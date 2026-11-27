#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

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
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "client/kv_client.h"
#include "transport/transport_factory.h"
#include "common/version.h"
#include "utils/prom_parse.h"
#include "utils/sha256.h"

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
      "                  [--ready-timeout SECS] [--put-arena 0|1]\n"
      "                  [--put-refill cpu|odirect [--put-refill-file PATH]]\n"
      "                  (refill mode: --op put --threads 1 --batch 1; --count 1..8 pairs)\n");
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
  // Distinguish connection setup work from steady-state data transfers.
  std::printf(
      "DIAG_CONN phase=%s v2_probe_attempts=%llu v2_probe_failures=%llu "
      "endpoint_cache_hits=%llu endpoint_cache_misses=%llu "
      "endpoint_cache_evictions=%llu\n",
      phase,
      (unsigned long long)CounterDelta(
          before, after, "dfkv_rdma_client_v2_probe_attempts_total"),
      (unsigned long long)CounterDelta(
          before, after, "dfkv_rdma_client_v2_probe_failures_total"),
      (unsigned long long)CounterDelta(
          before, after, "dfkv_rdma_endpoint_cache_hits_total"),
      (unsigned long long)CounterDelta(
          before, after, "dfkv_rdma_endpoint_cache_misses_total"),
      (unsigned long long)CounterDelta(
          before, after, "dfkv_rdma_endpoint_cache_evictions_total"));
}

namespace {

constexpr size_t kHostFillAlign = 4096;
constexpr size_t kHostFillMaxBytes = 16ull << 20;
constexpr size_t kHostFillMaxPairs = 8;

struct HostFillStats {
  std::vector<double> fill_us;
  std::vector<double> register_us;
  std::vector<double> scalar_put_us;
  std::vector<double> batch_put_us;
  std::vector<double> readback_us;
  std::vector<double> total_put_us;
};

struct HostFillFd {
  ~HostFillFd() {
    if (fd >= 0) (void)::close(fd);
  }
  int fd = -1;
};

using HostFillAllocation = std::unique_ptr<void, decltype(&std::free)>;

static HostFillAllocation AllocateHostFill(size_t bytes) {
  void* p = nullptr;
  const size_t rounded = (bytes + kHostFillAlign - 1) & ~(kHostFillAlign - 1);
  if (::posix_memalign(&p, kHostFillAlign, rounded) != 0) p = nullptr;
  return HostFillAllocation(p, &std::free);
}

static double ElapsedUs(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::micro>(end - start).count();
}

static std::string PayloadSha256(const void* data, size_t size) {
  std::array<uint8_t, 32> digest{};
  Sha256(data, size, digest.data());
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out(64, '0');
  for (size_t i = 0; i < digest.size(); ++i) {
    out[2 * i] = kHex[digest[i] >> 4];
    out[2 * i + 1] = kHex[digest[i] & 0xF];
  }
  return out;
}

static int HostFillError(const char* stage, const std::string& detail) {
  std::fprintf(stderr, "HOSTFILL_ERROR stage=%s detail=%s\n", stage,
               detail.c_str());
  return 1;
}

static bool CheckHostFillPayload(const std::string& key, const void* expected,
                                 const void* actual, size_t size,
                                 std::string* actual_sha256) {
  *actual_sha256 = PayloadSha256(actual, size);
  if (std::memcmp(expected, actual, size) == 0) return true;
  size_t first = 0;
  const auto* e = static_cast<const unsigned char*>(expected);
  const auto* a = static_cast<const unsigned char*>(actual);
  while (first < size && e[first] == a[first]) ++first;
  std::fprintf(stderr,
               "HOSTFILL_PAYLOAD_MISMATCH key=%s first_byte=%zu "
               "expected_sha256=%s actual_sha256=%s\n",
               key.c_str(), first, PayloadSha256(expected, size).c_str(),
               actual_sha256->c_str());
  return false;
}

// One arm of one warmup/measured pair. The pool arm redeclares the exact same
// base/size after refilling; the out-of-pool arm never declares its source.
// Both use the production scalar and flat-batch entry points, then compare all
// returned bytes against the fill that immediately preceded those PUTs.
static int RunHostFillArm(KVClient* c, const std::string& run_id, size_t size,
                          bool direct, int direct_fd, bool pool_arm,
                          size_t iteration, size_t fill_sequence, char* source,
                          char* readback, HostFillStats* stats) {
  const char* arm = pool_arm ? "pool" : "out_of_pool";
  const auto fill_start = Clock::now();
  if (direct) {
    const off_t offset = static_cast<off_t>(fill_sequence * size);
    ssize_t got;
    do {
      got = ::pread(direct_fd, source, size, offset);
    } while (got < 0 && errno == EINTR);
    if (got < 0) {
      return HostFillError(
          "odirect_read",
          std::string("errno=") + std::to_string(errno) +
              " offset=" + std::to_string(offset) + " bytes=" +
              std::to_string(size));
    }
    if (static_cast<size_t>(got) != size) {
      return HostFillError("odirect_read",
                           "short_read=" + std::to_string(got) +
                               " expected=" + std::to_string(size));
    }
  } else {
    for (size_t i = 0; i < size; ++i) {
      source[i] = static_cast<char>(
          (i * 17 + (i >> 8) * 29 + fill_sequence * 53) & 0xFF);
    }
  }
  const auto fill_end = Clock::now();

  const std::string before_register = c->MetricsSnapshot();
  const auto register_start = Clock::now();
  if (pool_arm && !c->RegisterMemory(source, size)) {
    return HostFillError("pool_reregister",
                         "same base/size registration returned false");
  }
  const auto register_end = Clock::now();
  const std::string before = c->MetricsSnapshot();

  const std::string base =
      run_id + "refill-" + std::to_string(fill_sequence) + "-";
  const std::string scalar_key = base + "scalar";
  const auto scalar_start = Clock::now();
  const bool scalar_ok = c->Put(scalar_key, source, size);
  const auto scalar_end = Clock::now();
  if (!scalar_ok) return HostFillError("scalar_put", scalar_key);

  std::vector<KvPutItem> puts(2);
  for (size_t i = 0; i < puts.size(); ++i) {
    puts[i] = {base + "flat-" + std::to_string(i), source, size};
  }
  const auto batch_start = Clock::now();
  const std::vector<bool> put_oks = c->BatchPut(puts);
  const auto batch_end = Clock::now();
  if (put_oks.size() != puts.size() ||
      std::find(put_oks.begin(), put_oks.end(), false) != put_oks.end()) {
    return HostFillError("flat_batch_put", base + "flat-*");
  }

  const auto readback_start = Clock::now();
  std::memset(readback, 0xA5, 2 * size);
  if (!c->Get(scalar_key, readback, size)) {
    return HostFillError("scalar_readback", scalar_key);
  }
  std::array<std::string, 3> actual_hashes;
  if (!CheckHostFillPayload(scalar_key, source, readback, size,
                            &actual_hashes[0])) {
    return 1;
  }
  std::vector<KvGetItem> gets(2);
  for (size_t i = 0; i < gets.size(); ++i) {
    gets[i] = {base + "flat-" + std::to_string(i), readback + i * size, size};
  }
  const std::vector<bool> hits = c->BatchGet(gets);
  if (hits.size() != gets.size() ||
      std::find(hits.begin(), hits.end(), false) != hits.end()) {
    return HostFillError("flat_batch_readback", base + "flat-*");
  }
  for (size_t i = 0; i < gets.size(); ++i) {
    if (!CheckHostFillPayload(gets[i].key, source, readback + i * size, size,
                              &actual_hashes[i + 1])) {
      return 1;
    }
  }
  const auto readback_end = Clock::now();
  const std::string after = c->MetricsSnapshot();

  uint64_t adhoc = 0, active = 0, active_before = 0, pool_active = 0;
  uint64_t inline_max = 0, lease_fallbacks = 0;
  const bool metrics = PromMetricValue(
                           after, "dfkv_rdma_client_adhoc_user_mr_total",
                           &adhoc) &&
                       PromMetricValue(
                           after, "dfkv_rdma_client_transient_user_mr_active",
                           &active) &&
                       PromMetricValue(
                           before, "dfkv_rdma_client_transient_user_mr_active",
                           &active_before) &&
                       PromMetricValue(
                           after, "dfkv_rdma_client_pool_mr_active_registrations",
                           &pool_active);
  const bool lease_metrics = PromMetricValue(
                                 after, "dfkv_rdma_client_inline_put_max_bytes",
                                 &inline_max) &&
                             PromMetricValue(
                                 after,
                                 "dfkv_rdma_client_leaseput_path_fallbacks_total",
                                 &lease_fallbacks);
  const uint64_t adhoc_delta = CounterDelta(
      before, after, "dfkv_rdma_client_adhoc_user_mr_total");
  const uint64_t pool_reg_delta = CounterDelta(
      before_register, after, "dfkv_rdma_client_pool_mr_registrations_total");
  const uint64_t leaseput_delta = CounterDelta(
      before, after, "dfkv_rdma_client_leaseput_ops_total");
  const uint64_t lease_fallback_delta = CounterDelta(
      before, after, "dfkv_rdma_client_leaseput_path_fallbacks_total");
  const double fill_us = ElapsedUs(fill_start, fill_end);
  const double register_us =
      pool_arm ? ElapsedUs(register_start, register_end) : 0.0;
  const double scalar_us = ElapsedUs(scalar_start, scalar_end);
  const double batch_us = ElapsedUs(batch_start, batch_end);
  const double readback_us = ElapsedUs(readback_start, readback_end);
  const auto unix_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
  std::printf(
      "HOSTFILL_SAMPLE phase=%s pair=%zu fill_sequence=%zu arm=%s "
      "bytes=%zu fill=%s fill_us=%.3f register_us=%.3f scalar_put_us=%.3f "
      "flat_batch_put_us=%.3f total_put_us=%.3f readback_us=%.3f "
      "expected_sha256=%s scalar_sha256=%s flat0_sha256=%s flat1_sha256=%s "
      "rdma_metrics=%d adhoc_mr_delta=%llu transient_before=%llu "
      "transient_after=%llu pool_active=%llu pool_registration_delta=%llu "
      "inline_put_max_bytes=%llu leaseput_delta=%llu "
      "leaseput_fallback_delta=%llu timestamp_unix_ms=%lld\n",
      iteration == 0 ? "warmup" : "measure", iteration, fill_sequence, arm,
      size, direct ? "odirect" : "cpu", fill_us, register_us, scalar_us,
      batch_us, scalar_us + batch_us, readback_us,
      PayloadSha256(source, size).c_str(), actual_hashes[0].c_str(),
      actual_hashes[1].c_str(), actual_hashes[2].c_str(), metrics ? 1 : 0,
      (unsigned long long)adhoc_delta, (unsigned long long)active_before,
      (unsigned long long)active, (unsigned long long)pool_active,
      (unsigned long long)pool_reg_delta, (unsigned long long)inline_max,
      (unsigned long long)leaseput_delta,
      (unsigned long long)lease_fallback_delta, (long long)unix_ms);

  if (metrics) {
    const uint64_t expected_adhoc = pool_arm ? 0 : 3;
    if (adhoc_delta != expected_adhoc) {
      return HostFillError(
          "mr_selection",
          std::string(arm) + " expected_adhoc=" +
              std::to_string(expected_adhoc) + " actual=" +
              std::to_string(adhoc_delta));
    }
    if (active_before != 0 || active != 0 || pool_active == 0 ||
        pool_reg_delta != 0) {
      return HostFillError(
          "mr_lifetime",
          "transient_before=" + std::to_string(active_before) +
              " transient_after=" + std::to_string(active) +
              " pool_active=" + std::to_string(pool_active) +
              " pool_registration_delta=" + std::to_string(pool_reg_delta));
    }
  }
  if (metrics && lease_metrics) {
    const uint64_t expected_leases =
        inline_max != 0 && size > inline_max ? 3 : 0;
    if (leaseput_delta != expected_leases || lease_fallback_delta != 0) {
      return HostFillError(
          "put_path",
          "expected_leases=" + std::to_string(expected_leases) +
              " actual_leases=" + std::to_string(leaseput_delta) +
              " fallback_delta=" + std::to_string(lease_fallback_delta));
    }
  }
  if (iteration != 0) {
    stats->fill_us.push_back(fill_us);
    stats->register_us.push_back(register_us);
    stats->scalar_put_us.push_back(scalar_us);
    stats->batch_put_us.push_back(batch_us);
    stats->readback_us.push_back(readback_us);
    stats->total_put_us.push_back(scalar_us + batch_us);
  }
  return 0;
}

static void ReportHostFillSummary(const char* arm, size_t pairs, size_t size,
                                  HostFillStats* s) {
  std::printf(
      "HOSTFILL_SUMMARY arm=%s samples=%zu bytes=%zu "
      "fill_us_p50=%.3f fill_us_p90=%.3f fill_us_max=%.3f "
      "register_us_p50=%.3f scalar_put_us_p50=%.3f "
      "flat_batch_put_us_p50=%.3f total_put_us_p50=%.3f "
      "total_put_us_p90=%.3f total_put_us_max=%.3f readback_us_p50=%.3f\n",
      arm, pairs, size, Pct(s->fill_us, 0.50), Pct(s->fill_us, 0.90),
      Pct(s->fill_us, 1.0), Pct(s->register_us, 0.50),
      Pct(s->scalar_put_us, 0.50), Pct(s->batch_put_us, 0.50),
      Pct(s->total_put_us, 0.50), Pct(s->total_put_us, 0.90),
      Pct(s->total_put_us, 1.0), Pct(s->readback_us, 0.50));
}

static int RunHostFillBenchmark(KVClient* c, const std::string& run_id,
                                size_t size, size_t pairs, bool direct,
                                const std::string& direct_path) {
  HostFillFd file;
  const size_t required_segments = 2 * (pairs + 1);
  const size_t required_file_bytes = required_segments * size;
  if (direct) {
    file.fd = ::open(direct_path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (file.fd < 0) {
      return HostFillError("odirect_open",
                           "path=" + direct_path +
                               " errno=" + std::to_string(errno));
    }
    struct stat st {};
    if (::fstat(file.fd, &st) != 0) {
      return HostFillError("odirect_stat", std::to_string(errno));
    }
    if (!S_ISREG(st.st_mode)) {
      return HostFillError("odirect_regular_file", direct_path);
    }
    if (st.st_size < 0 || static_cast<uint64_t>(st.st_size) <
                              static_cast<uint64_t>(required_file_bytes)) {
      return HostFillError(
          "odirect_file_size",
          "required_bytes=" + std::to_string(required_file_bytes) +
              " actual_bytes=" + std::to_string(st.st_size));
    }
  }
  HostFillAllocation pool = AllocateHostFill(size);
  HostFillAllocation out_of_pool = AllocateHostFill(size);
  HostFillAllocation readback = AllocateHostFill(2 * size);
  if (!pool || !out_of_pool || !readback) {
    return HostFillError("allocation", "aligned source/readback allocation");
  }
  const std::string setup_before = c->MetricsSnapshot();
  if (!c->RegisterMemory(pool.get(), size) ||
      !c->RegisterMemory(readback.get(), 2 * size)) {
    return HostFillError("setup_registration",
                         "source/readback persistent registration failed; "
                         "refusing operation-scoped fallback");
  }
  const std::string setup_after = c->MetricsSnapshot();
  uint64_t pool_active = 0, transient_active = 0;
  const bool metrics = PromMetricValue(
                           setup_after,
                           "dfkv_rdma_client_pool_mr_active_registrations",
                           &pool_active) &&
                       PromMetricValue(
                           setup_after,
                           "dfkv_rdma_client_transient_user_mr_active",
                           &transient_active);
  std::printf(
      "HOSTFILL_CONFIG version=%s fill=%s bytes=%zu pairs=%zu "
      "odirect_alignment=%zu required_file_segments=%zu "
      "required_file_bytes=%zu source_path=%s\n",
      dfkv::Version(), direct ? "odirect" : "cpu", size, pairs,
      kHostFillAlign, required_segments, required_file_bytes,
      direct ? direct_path.c_str() : "-");
  std::printf(
      "HOSTFILL_SETUP rdma_metrics=%d pool_registrations=%llu "
      "pool_active=%llu transient_active=%llu\n",
      metrics ? 1 : 0,
      (unsigned long long)CounterDelta(
          setup_before, setup_after,
          "dfkv_rdma_client_pool_mr_registrations_total"),
      (unsigned long long)pool_active, (unsigned long long)transient_active);
  if (metrics && (pool_active == 0 || transient_active != 0)) {
    return HostFillError("setup_lifetime", "unexpected initial MR gauges");
  }

  HostFillStats pool_stats, out_of_pool_stats;
  size_t fill_sequence = 0;
  for (size_t iteration = 0; iteration <= pairs; ++iteration) {
    const bool pool_first = iteration % 2 == 0;
    for (size_t order = 0; order < 2; ++order) {
      const bool pool_arm = (order == 0) == pool_first;
      char* source = static_cast<char*>(pool_arm ? pool.get()
                                                : out_of_pool.get());
      HostFillStats* stats = pool_arm ? &pool_stats : &out_of_pool_stats;
      const int rc = RunHostFillArm(
          c, run_id, size, direct, file.fd, pool_arm, iteration, fill_sequence,
          source, static_cast<char*>(readback.get()), stats);
      if (rc != 0) return rc;
      ++fill_sequence;
    }
  }
  ReportHostFillSummary("pool", pairs, size, &pool_stats);
  ReportHostFillSummary("out_of_pool", pairs, size, &out_of_pool_stats);
  const size_t samples = 2 * (pairs + 1);
  std::printf(
      "HOSTFILL_RESULT status=PASS samples=%zu put_operations=%zu "
      "verified_full_readbacks=%zu failures=0\n",
      samples, samples * 3, samples * 3);
  return 0;
}

}  // namespace

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
  std::string put_refill, put_refill_file;
  size_t size = 2752512, count = 2000, threads = 8, batch = 1, bc = 0;
  size_t put_arena = 0;
  size_t ready_timeout_s = 30, mds_poll_ms = 1000;
  for (int i = 1; i < argc; i += 2) {
    const char* flag = argv[i];
    const bool known =
        !std::strcmp(flag, "--members") || !std::strcmp(flag, "--mds") ||
        !std::strcmp(flag, "--group") || !std::strcmp(flag, "--ready-timeout") ||
        !std::strcmp(flag, "--size") || !std::strcmp(flag, "--count") ||
        !std::strcmp(flag, "--threads") || !std::strcmp(flag, "--batch") ||
        !std::strcmp(flag, "--bc") || !std::strcmp(flag, "--op") ||
        !std::strcmp(flag, "--key-seed") || !std::strcmp(flag, "--put-arena") ||
        !std::strcmp(flag, "--put-refill") ||
        !std::strcmp(flag, "--put-refill-file");
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
    else if (!std::strcmp(flag, "--put-refill")) put_refill = value;
    else if (!std::strcmp(flag, "--put-refill-file")) put_refill_file = value;
    else {
      size_t* target = nullptr;
      if (!std::strcmp(flag, "--ready-timeout")) target = &ready_timeout_s;
      else if (!std::strcmp(flag, "--size")) target = &size;
      else if (!std::strcmp(flag, "--count")) target = &count;
      else if (!std::strcmp(flag, "--threads")) target = &threads;
      else if (!std::strcmp(flag, "--batch")) target = &batch;
      else if (!std::strcmp(flag, "--bc")) target = &bc;
      else if (!std::strcmp(flag, "--put-arena")) target = &put_arena;
      if (!ParseSize(value, target)) {
        std::fprintf(stderr, "invalid numeric value for %s: %s\n", flag, value);
        return 2;
      }
    }
  }
  if (put_arena > 1) {
    std::fprintf(stderr, "--put-arena must be 0 or 1\n");
    return 2;
  }
  if (op != "put" && op != "get" && op != "both") {
    std::fprintf(stderr, "invalid --op: %s (expected put, get, or both)\n", op.c_str());
    return 2;
  }
  if (size == 0 || count == 0 || threads == 0 || batch == 0) {
    std::fprintf(stderr, "--size, --count, --threads, and --batch must be greater than zero\n");
    return 2;
  }
  if (!put_refill_file.empty() && put_refill.empty()) {
    std::fprintf(stderr, "--put-refill-file requires --put-refill odirect\n");
    return 2;
  }
  if (!put_refill.empty()) {
    if (put_refill != "cpu" && put_refill != "odirect") {
      std::fprintf(stderr, "--put-refill must be cpu or odirect\n");
      return 2;
    }
    if (op != "put" || threads != 1 || batch != 1 || put_arena != 0) {
      std::fprintf(stderr,
                   "--put-refill requires --op put --threads 1 --batch 1 "
                   "--put-arena 0\n");
      return 2;
    }
    if (count == 0 || count > kHostFillMaxPairs) {
      std::fprintf(stderr, "--put-refill requires --count 1..8 measured pairs\n");
      return 2;
    }
    if (size > kHostFillMaxBytes) {
      std::fprintf(stderr, "--put-refill size must be at most 16777216 bytes\n");
      return 2;
    }
    const bool direct = put_refill == "odirect";
    if ((direct && put_refill_file.empty()) ||
        (!direct && !put_refill_file.empty())) {
      std::fprintf(stderr,
                   "--put-refill-file PATH is required for odirect and "
                   "forbidden for cpu\n");
      return 2;
    }
    if (direct && size % kHostFillAlign != 0) {
      std::fprintf(stderr,
                   "--put-refill odirect requires 4096-byte-aligned --size\n");
      return 2;
    }
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
    // One key per member, found by hashing candidates until every member
    // owns one (32 fixed keys left about 40 of 65 members unprobed).
    const std::vector<std::string> members = c.MemberAddresses();
    std::map<std::string, std::string> settle_keys;  // member -> key
    for (size_t i = 0; settle_keys.size() < members.size() &&
                       i < 256 * members.size() + 1024;
         ++i) {
      std::string sk = "dfkv_bench/settle_" + std::to_string(i);
      const std::string owner = c.OwnerOf(sk);
      if (!owner.empty()) settle_keys.emplace(owner, std::move(sk));
    }
    const size_t kSettleKeys = members.size();
    size_t settle_failures = members.size() - settle_keys.size();
    for (const auto& [owner, sk] : settle_keys) {
      (void)owner;
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
                   "dfkv_bench: %zu/%zu ring members failed to settle; "
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
  if (!put_refill.empty()) {
    return RunHostFillBenchmark(&c, run_id, size, count,
                                put_refill == "odirect", put_refill_file);
  }
  const size_t units = count / batch + (count % batch != 0);

  std::vector<double> lat;
  std::atomic<size_t> fails{0};
  size_t total_fails = 0;
  const char* stall_env = std::getenv("DFKV_BENCH_STALL_MS");
  const long stall_ms = stall_env && *stall_env
                            ? std::strtol(stall_env, nullptr, 10)
                            : 0;

  if (op == "put" || op == "both") {
    fails.store(0);
    const std::string diag_before = c.MetricsSnapshot();
    // --put-arena 1: register the shared source buffer so PUT payload SGEs
    // hit the pool path instead of a per-op ad-hoc MR (ibv_reg_mr per op).
    // The buffer is read-only for the wire lifetime of each op; a missing or
    // failed registration invalidates the comparison, so fail closed.
    if (put_arena) {
      if (!c.RegisterMemory(val.data(), val.size())) {
        std::fprintf(stderr,
                     "dfkv_bench: PUT source arena registration failed "
                     "(bytes=%zu); refusing an ad-hoc-MR fallback\n",
                     val.size());
        ReportDiagnostics("PUT_SETUP", diag_before, c.MetricsSnapshot());
        return 2;
      }
      std::printf("PUT_SETUP registered_source_bytes=%zu\n", val.size());
    }
    double s = RunPhase(units, threads, [&](size_t u) {
      size_t base = u * batch, w = std::min(batch, count - base);
      std::vector<KvPutItem> items(w);
      for (size_t j = 0; j < w; ++j) items[j] = {key(base + j), val.data(), val.size()};
      const auto st0 = stall_ms > 0 ? Clock::now() : Clock::time_point{};
      auto oks = c.BatchPut(items);
      if (stall_ms > 0) {
        const auto st1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(st1 - st0).count();
        if (ms >= static_cast<double>(stall_ms))
          std::fprintf(stderr, "STALL %.3f dur=%.1fms u=%zu phase=PUT\n",
                       std::chrono::duration<double>(
                           std::chrono::system_clock::now().time_since_epoch()).count(),
                       ms, u);
      }
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
      const auto st0 = stall_ms > 0 ? Clock::now() : Clock::time_point{};
      auto hits = c.BatchGet(items);
      if (stall_ms > 0) {
        const auto st1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(st1 - st0).count();
        if (ms >= static_cast<double>(stall_ms))
          std::fprintf(stderr, "STALL %.3f dur=%.1fms u=%zu phase=GET\n",
                       std::chrono::duration<double>(
                           std::chrono::system_clock::now().time_since_epoch()).count(),
                       ms, u);
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
