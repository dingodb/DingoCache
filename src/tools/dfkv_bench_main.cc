/* dfkv_bench — throughput / latency benchmark for a dfkv cluster.
 *
 *   dfkv_bench (--members "n=ip:port,..." | --mds "ip:port,..." [--group g])
 *              [--size BYTES] [--count N] [--threads T] [--batch B]
 *              [--op put|get|both] [--key-seed S] [--ready-timeout SECS]
 *
 * Ring membership is EITHER a static --members list OR --mds discovery (exactly
 * one). With --mds the client populates the ring from the MDS group and waits
 * (up to --ready-timeout, default 30s) for a warmup Put to succeed before the
 * measured phases start, so you can point bench straight at a production ring
 * without hand-listing members from `dfkvctl ring`.
 *
 * Transport is chosen by the same env switch as the client: DFKV_RDMA=1 (+ device
 * DFKV_RDMA_DEV) uses RDMA, else TCP. --batch B issues B keys per BatchPut/
 * BatchGet call, which on RDMA pipelines up to B requests in flight on one
 * connection (cap by DFKV_RDMA_DEPTH); B=1 is the per-op path. Each op uses a
 * unique key ("dfkv-bench-<i>") so the write-once cache never collides. Reports
 * aggregate GB/s and per-call p50/p99/max latency for the PUT and GET phases.
 *
 * --key-seed S pins the object-key set to S instead of the default pid, so
 * phases can run in DIFFERENT processes or on DIFFERENT nodes: `--op put
 * --key-seed x` on node A, then `--op get --key-seed x` (same --size/--count)
 * on node B measures the cross-node read path. With an explicit seed,
 * re-running PUT with the same seed+size hits write-once dedup and measures
 * nothing; pick a fresh seed per experiment.
 * The pid default keeps ordinary reruns collision-free. */
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "client/kv_client.h"
#include "transport/transport_factory.h"
#include "common/version.h"
#include "utils/prom_parse.h"

using namespace dfkv;  // NOLINT
using Clock = std::chrono::steady_clock;

static std::vector<std::pair<std::string, std::string>> ParseMembers(const std::string& s) {
  std::vector<std::pair<std::string, std::string>> out;
  size_t i = 0;
  while (i <= s.size()) {
    size_t c = s.find(',', i);
    if (c == std::string::npos) c = s.size();
    std::string tok = s.substr(i, c - i);
    size_t eq = tok.find('=');
    if (eq != std::string::npos) out.emplace_back(tok.substr(0, eq), tok.substr(eq + 1));
    if (c == s.size()) break;
    i = c + 1;
  }
  return out;
}

// Comma-split MDS endpoints ("ip:port,ip:port") — no name= prefix, unlike members.
static std::vector<std::string> SplitComma(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i <= s.size()) {
    size_t c = s.find(',', i);
    if (c == std::string::npos) c = s.size();
    std::string tok = s.substr(i, c - i);
    if (!tok.empty()) out.push_back(tok);
    if (c == s.size()) break;
    i = c + 1;
  }
  return out;
}

static bool ParseSize(const char* text, size_t* value) {
  if (text == nullptr || *text == '\0') return false;
  const char* end = text + std::strlen(text);
  auto parsed = std::from_chars(text, end, *value);
  return parsed.ec == std::errc{} && parsed.ptr == end;
}

static void PrintUsage(FILE* stream) {
  std::fprintf(
      stream,
      "usage: dfkv_bench (--members name=ip:port,... | --mds ip:port,... [--group g])\n"
      "                  [--size BYTES] [--count N] [--threads T] [--batch B]\n"
      "                  [--bc N] [--op put|get|both] [--key-seed S]\n"
      "                  [--ready-timeout SECS]\n");
}

static double Pct(std::vector<double>& v, double p) {
  if (v.empty()) return 0;
  size_t k = static_cast<size_t>(p * (v.size() - 1));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

// Run `units` work-items across `threads`. fn(u) does one unit (a batch) and
// returns the number of FAILED ops in it. Records per-unit latency in ms.
static double RunPhase(size_t units, size_t threads,
                       const std::function<size_t(size_t)>& fn,
                       std::vector<double>* lat, std::atomic<size_t>* fails) {
  lat->assign(units, 0.0);
  std::atomic<size_t> next{0};
  auto worker = [&] {
    for (size_t u = next.fetch_add(1); u < units; u = next.fetch_add(1)) {
      auto t0 = Clock::now();
      size_t f = fn(u);
      auto t1 = Clock::now();
      (*lat)[u] = std::chrono::duration<double, std::milli>(t1 - t0).count();
      if (f) fails->fetch_add(f);
    }
  };
  auto start = Clock::now();
  std::vector<std::thread> ts;
  for (size_t w = 0; w < threads; ++w) ts.emplace_back(worker);
  for (auto& t : ts) t.join();
  return std::chrono::duration<double>(Clock::now() - start).count();
}

static void Report(const char* phase, size_t count, size_t size, size_t threads,
                   size_t batch, double secs, std::vector<double>& lat,
                   size_t fails) {
  const size_t succeeded = fails <= count ? count - fails : 0;
  const double good_gb = double(succeeded) * double(size) / 1e9;
  std::printf("%-5s n=%zu size=%zu threads=%zu batch=%zu | %.3fs  "
              "goodput %.2f GB/s  %.0f ok-ops/s  "
              "call-lat ms p50=%.3f p99=%.3f max=%.3f  ok=%zu fails=%zu\n",
              phase, count, size, threads, batch, secs,
              secs > 0 ? good_gb / secs : 0,
              secs > 0 ? succeeded / secs : 0, Pct(lat, 0.50),
              Pct(lat, 0.99), Pct(lat, 1.0), succeeded, fails);
}

static uint64_t CounterDelta(const std::string& before,
                             const std::string& after,
                             const char* name) {
  const uint64_t a = PromMetricSum(before, name);
  const uint64_t b = PromMetricSum(after, name);
  return b >= a ? b - a : b;
}

static void ReportDiagnostics(const char* phase, const std::string& before,
                              const std::string& after) {
  uint64_t ring = 0;
  const bool client_metrics_available =
      PromMetricValue(after, "dfkv_client_ring_members", &ring);
  uint64_t mds_reachable = 0;
  const bool mds_available =
      PromMetricValue(after, "dfkv_client_mds_reachable", &mds_reachable);
  uint64_t declared_max_block = 0;
  const bool rdma_available = PromMetricValue(
      after, "dfkv_rdma_client_declared_max_block_bytes",
      &declared_max_block);
  std::printf(
      "DIAG phase=%s client_metrics_available=%d ring=%llu "
      "mds_available=%d mds_reachable=%llu mds_unreachable_polls=%llu "
      "rdma_available=%d io_errors=%llu unhealthy_skips=%llu "
      "peer_bad=%llu endpoint_errors=%llu rail_errors=%llu cq_errors=%llu "
      "completion_timeouts=%llu stale_retries=%llu mr_rejections=%llu "
      "mr_failures=%llu adhoc_mr=%llu mr_registered_bytes=%llu "
      "oversize_rejects=%llu max_block_seen_bytes=%llu "
      "declared_max_block_bytes=%llu\n",
      phase, client_metrics_available ? 1 : 0,
      (unsigned long long)ring, mds_available ? 1 : 0,
      (unsigned long long)mds_reachable,
      (unsigned long long)CounterDelta(
          before, after, "dfkv_client_mds_unreachable_polls_total"),
      rdma_available ? 1 : 0,
      (unsigned long long)CounterDelta(
          before, after, "dfkv_client_io_errors_total"),
      (unsigned long long)CounterDelta(
          before, after, "dfkv_client_unhealthy_skips_total"),
      (unsigned long long)CounterDelta(
          before, after, "dfkv_client_peer_marked_bad_total"),
      (unsigned long long)CounterDelta(
          before, after, "dfkv_rdma_client_endpoint_errors_total"),
      (unsigned long long)CounterDelta(
          before, after, "dfkv_rdma_client_rail_errors_total"),
      (unsigned long long)CounterDelta(
          before, after, "dfkv_rdma_cq_errors_total"),
      (unsigned long long)CounterDelta(
          before, after, "dfkv_rdma_client_completion_timeouts_total"),
      (unsigned long long)CounterDelta(
          before, after, "dfkv_rdma_client_stale_pool_retries_total"),
      (unsigned long long)CounterDelta(
          before, after, "dfkv_rdma_client_mr_registration_rejections_total"),
      (unsigned long long)CounterDelta(
          before, after,
          "dfkv_rdma_client_pool_mr_registration_failures_total"),
      (unsigned long long)CounterDelta(
          before, after, "dfkv_rdma_client_adhoc_user_mr_total"),
      (unsigned long long)PromMetricValue(
          after, "dfkv_rdma_client_mr_registered_bytes"),
      (unsigned long long)CounterDelta(
          before, after, "dfkv_rdma_client_oversize_rejects_total"),
      (unsigned long long)PromMetricValue(
          after, "dfkv_rdma_client_max_block_seen_bytes"),
      (unsigned long long)declared_max_block);
}

int main(int argc, char** argv) {
  if (argc == 2 && dfkv::WantsVersion(argc, argv)) {
    std::printf("dfkv_bench %s\n", dfkv::Version());
    return 0;
  }
  if (argc == 2 && dfkv::WantsHelp(argc, argv)) {
    PrintUsage(stdout);
    return 0;
  }
  std::string members, mds, group = "default", op = "both", key_seed;
  size_t size = 2752512, count = 2000, threads = 8, batch = 1, bc = 0;
  size_t ready_timeout_s = 30, mds_poll_ms = 1000;
  for (int i = 1; i < argc; i += 2) {
    const char* flag = argv[i];
    const bool known =
        !std::strcmp(flag, "--members") || !std::strcmp(flag, "--mds") ||
        !std::strcmp(flag, "--group") || !std::strcmp(flag, "--ready-timeout") ||
        !std::strcmp(flag, "--size") || !std::strcmp(flag, "--count") ||
        !std::strcmp(flag, "--threads") || !std::strcmp(flag, "--batch") ||
        !std::strcmp(flag, "--bc") || !std::strcmp(flag, "--op") ||
        !std::strcmp(flag, "--key-seed");
    if (!known) {
      std::fprintf(stderr, "unknown argument: %s\n", flag);
      PrintUsage(stderr);
      return 2;
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "missing value for %s\n", flag);
      PrintUsage(stderr);
      return 2;
    }
    const char* value = argv[i + 1];
    if (!std::strcmp(flag, "--members")) members = value;
    else if (!std::strcmp(flag, "--mds")) mds = value;
    else if (!std::strcmp(flag, "--group")) group = value;
    else if (!std::strcmp(flag, "--op")) op = value;
    else if (!std::strcmp(flag, "--key-seed")) key_seed = value;
    else {
      size_t* target = nullptr;
      if (!std::strcmp(flag, "--ready-timeout")) target = &ready_timeout_s;
      else if (!std::strcmp(flag, "--size")) target = &size;
      else if (!std::strcmp(flag, "--count")) target = &count;
      else if (!std::strcmp(flag, "--threads")) target = &threads;
      else if (!std::strcmp(flag, "--batch")) target = &batch;
      else if (!std::strcmp(flag, "--bc")) target = &bc;
      if (!ParseSize(value, target)) {
        std::fprintf(stderr, "invalid numeric value for %s: %s\n", flag, value);
        return 2;
      }
    }
  }
  if (op != "put" && op != "get" && op != "both") {
    std::fprintf(stderr, "invalid --op: %s (expected put, get, or both)\n", op.c_str());
    return 2;
  }
  if (size == 0 || count == 0 || threads == 0 || batch == 0) {
    std::fprintf(stderr, "--size, --count, --threads, and --batch must be greater than zero\n");
    return 2;
  }
  // Ring membership: exactly one of a static --members list or --mds discovery.
  auto mem = ParseMembers(members);
  if (mem.empty() == mds.empty()) {
    std::fprintf(stderr,
                 "need exactly one of --members name=ip:port,... "
                 "or --mds ip:port[,...] [--group g]\n");
    return 2;
  }

  std::string reason;
  MakeClientTransport(&reason);
  if (mds.empty())
    std::printf("dfkv_bench transport=%s members=%zu\n", reason.c_str(), mem.size());
  else
    std::printf("dfkv_bench transport=%s mds=%s group=%s\n",
                reason.c_str(), mds.c_str(), group.c_str());

  KVClient c(mem, "dfkv/bench");  // empty members when discovering via MDS
  if (!mds.empty()) {
    // Populate the ring from MDS, then wait until it's usable (a warmup Put
    // succeeds) so the measured phases don't race discovery. Mirrors
    // dfkv_discover_smoke's readiness probe.
    c.StartMdsDiscovery(SplitComma(mds), group, static_cast<int>(mds_poll_ms));
    std::vector<char> probe(size, 'w');
    auto deadline = Clock::now() + std::chrono::seconds(ready_timeout_s);
    bool ready = false;
    while (!ready && Clock::now() < deadline) {
      if (c.Put("dfkv_bench/warmup_k", probe.data(), probe.size())) ready = true;
      else std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!ready) {
      std::fprintf(stderr,
                   "dfkv_bench: ring not ready after %zus (mds=%s group=%s)\n",
                   ready_timeout_s, mds.c_str(), group.c_str());
      ReportDiagnostics("READY", "", c.MetricsSnapshot());
      return 2;
    }
    // Settle: the single warmup key only reaches ONE ring member, leaving the
    // other nodes' RDMA connections cold. The measured phase's first batch to
    // those nodes fails on connection setup (observed as ~N/total fails where
    // N = unprobed node count on a freshly-restarted ring). Issue probe keys
    // with distinct hashes to cover every ring member, retrying individually
    // until each succeeds — a batched settle still fails on cold connections;
    // individual Puts block until the per-node QP is established.
    constexpr size_t kSettleKeys = 32;  // covers rings up to ~32 nodes
    size_t settle_failures = 0;
    for (size_t i = 0; i < kSettleKeys; ++i) {
      std::string sk = "dfkv_bench/settle_" + std::to_string(i);
      bool settled = false;
      for (int attempt = 0; attempt < 20; ++attempt) {
        if (c.Put(sk, probe.data(), probe.size())) {
          settled = true;
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      if (!settled) ++settle_failures;
    }
    if (settle_failures != 0) {
      std::fprintf(stderr,
                   "dfkv_bench: %zu/%zu connection-settle keys failed; "
                   "refusing a misleading measured phase\n",
                   settle_failures, kSettleKeys);
      ReportDiagnostics("SETTLE", "", c.MetricsSnapshot());
      return 2;
    }
  }
  // External --threads already provide concurrency; with multi-node members the
  // client's per-call internal RunParallel(batch_concurrency) would nest under
  // each external thread (threads x nodes x bc) and can exhaust threads/connections.
  // --bc 1 disables internal nesting so external threads drive load cleanly.
  if (bc) c.set_batch_concurrency(bc);

  std::string val(size, '\0');
  for (size_t i = 0; i < size; ++i) val[i] = static_cast<char>(i & 0xFF);
  // Unique object-key set per run (pid + size): dfkv is write-once
  // (idempotent Cache), so a fixed key set would make reruns at a different
  // --size skip PUT and then miss GET on stored-length mismatch. pid+size
  // guarantees fresh keys. --key-seed replaces pid so another process/node can
  // find a prior PUT; the caller owns collision avoidance.
  const std::string run_id =
      "dfkv-bench-" +
      (key_seed.empty() ? std::to_string(static_cast<long>(getpid())) : key_seed) +
      "-" + std::to_string(size) + "-";
  auto key = [&run_id](size_t i) { return run_id + std::to_string(i); };
  const size_t units = count / batch + (count % batch != 0);

  std::vector<double> lat;
  std::atomic<size_t> fails{0};
  size_t total_fails = 0;

  if (op == "put" || op == "both") {
    fails.store(0);
    const std::string diag_before = c.MetricsSnapshot();
    double s = RunPhase(units, threads, [&](size_t u) {
      size_t base = u * batch, w = std::min(batch, count - base);
      std::vector<KvPutItem> items(w);
      for (size_t j = 0; j < w; ++j) items[j] = {key(base + j), val.data(), val.size()};
      auto oks = c.BatchPut(items);
      size_t succeeded = 0;
      for (size_t j = 0; j < std::min(w, oks.size()); ++j) {
        if (oks[j]) ++succeeded;
      }
      return w - succeeded;
    }, &lat, &fails);
    const size_t phase_fails = fails.load();
    Report("PUT", count, size, threads, batch, s, lat, phase_fails);
    ReportDiagnostics("PUT", diag_before, c.MetricsSnapshot());
    total_fails += phase_fails;
  }
  if (op == "get" || op == "both") {
    fails.store(0);
    const std::string diag_before = c.MetricsSnapshot();
    // One contiguous, page-aligned destination arena for ALL threads, declared
    // once via RegisterMemory. A missing/failed arena invalidates an RDMA
    // throughput result, so fail closed instead of silently timing ad-hoc MRs.
    const size_t active_workers = std::min(threads, units);
    const size_t worker_slots = std::min(batch, count);
    if (size > std::numeric_limits<size_t>::max() - 4095 ||
        worker_slots > std::numeric_limits<size_t>::max() / active_workers) {
      std::fprintf(stderr, "dfkv_bench: GET arena size overflow\n");
      return 2;
    }
    const size_t stride = (size + 4095) & ~size_t{4095};
    const size_t slots = active_workers * worker_slots;
    if (stride > std::numeric_limits<size_t>::max() / slots) {
      std::fprintf(stderr, "dfkv_bench: GET arena size overflow\n");
      return 2;
    }
    const size_t arena_bytes = slots * stride;
    char* arena =
        static_cast<char*>(std::aligned_alloc(4096, arena_bytes));
    if (!arena) {
      std::fprintf(stderr,
                   "dfkv_bench: GET registered arena allocation failed "
                   "(bytes=%zu)\n", arena_bytes);
      ReportDiagnostics("GET_SETUP", diag_before, c.MetricsSnapshot());
      return 2;
    }
    if (!c.RegisterMemory(arena, arena_bytes)) {
      std::fprintf(stderr,
                   "dfkv_bench: GET arena registration failed (bytes=%zu); "
                   "refusing an ad-hoc-MR fallback\n", arena_bytes);
      std::free(arena);
      ReportDiagnostics("GET_SETUP", diag_before, c.MetricsSnapshot());
      return 2;
    }
    std::printf("GET_SETUP registered_arena_bytes=%zu slots=%zu stride=%zu\n",
                arena_bytes, slots, stride);
    std::atomic<size_t> arena_slot{0};
    double s = RunPhase(units, threads, [&](size_t u) {
      size_t base = u * batch, w = std::min(batch, count - base);
      thread_local char* mybuf = nullptr;
      if (!mybuf)
        mybuf = arena + arena_slot.fetch_add(1) * worker_slots * stride;
      std::vector<KvGetItem> items(w);
      for (size_t j = 0; j < w; ++j)
        items[j] = {key(base + j), mybuf + j * stride, size};
      // DFKV_BENCH_STALL_MS: log wall-clock timestamps of slow calls to stderr
      // so stalls can be time-correlated across processes/nodes (diagnostics).
      static const long stall_ms = [] {
        const char* e = std::getenv("DFKV_BENCH_STALL_MS");
        return e && *e ? std::strtol(e, nullptr, 10) : 0;
      }();
      const auto st0 = std::chrono::system_clock::now();
      auto hits = c.BatchGet(items);
      if (stall_ms > 0) {
        const auto st1 = std::chrono::system_clock::now();
        double ms = std::chrono::duration<double, std::milli>(st1 - st0).count();
        if (ms >= static_cast<double>(stall_ms))
          std::fprintf(stderr, "STALL %.3f dur=%.1fms u=%zu\n",
                       std::chrono::duration<double>(st1.time_since_epoch()).count(), ms, u);
      }
      size_t succeeded = 0;
      for (size_t j = 0; j < std::min(w, hits.size()); ++j) {
        if (hits[j]) ++succeeded;
      }
      return w - succeeded;
    }, &lat, &fails);
    const size_t phase_fails = fails.load();
    Report("GET", count, size, threads, batch, s, lat, phase_fails);
    ReportDiagnostics("GET", diag_before, c.MetricsSnapshot());
    total_fails += phase_fails;
    std::free(arena);
  }
  return total_fails == 0 ? 0 : 1;
}
