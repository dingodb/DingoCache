/* In-process server+fan-out-client memory benchmark for the staged-lease
 * datapath. Runs entirely on one RDMA host: an embedded cache node serves
 * real verbs, N threads open N independent client connections, and the tool
 * reports the server's residency split (connection-resident receive bytes vs
 * in-flight lease bytes) so the KDA-style deployment math is observable:
 *
 *   dfkvleasebench --clients 200 --obj-size 32MiB --ops 20
 *
 * Compare runs with --inline-bytes 0 (legacy connection-resident class) and
 * the default 4MiB threshold (staged-lease path) to see the receive-pool
 * footprint converge from "connections x max object class x depth" down to
 * "bytes actually in flight". */
#include "cache/kv_node_server.h"
#include "cache/rdma_server.h"
#include "client/kv_client.h"
#include "client/key_map.h"
#include <array>
#include "common/status.h"
#include "transport/rdma_transport.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {

size_t ParseSize(const char* text, size_t fallback) {
  char* end = nullptr;
  const unsigned long long v = std::strtoull(text, &end, 10);
  if (end == text) return fallback;
  unsigned long long scaled = v;
  if (end && *end) {
    switch (std::tolower(*end)) {
      case 'k': scaled = v << 10; break;
      case 'm': scaled = v << 20; break;
      case 'g': scaled = v << 30; break;
      default: return fallback;
    }
  }
  return static_cast<size_t>(scaled);
}

// One Prometheus value line (rfind skips HELP/TYPE lines).
long MetricOf(const RdmaServer& server, const std::string& name) {
  const std::string text = server.MetricsText();
  const size_t at = text.rfind(name + " ");
  if (at == std::string::npos) return -1;
  const size_t sp = text.find(' ', at);
  if (sp == std::string::npos) return -1;
  return std::strtol(text.c_str() + sp + 1, nullptr, 10);
}

// Labelled variant: takes the exact "name{label...}" key, rfinds it.
long MetricOf(const RdmaServer& server, const std::string& key,
               const std::string& name) {
  const std::string text = server.MetricsText();
  const size_t at = text.rfind(key + " ");
  if (at == std::string::npos) return -1;
  const size_t sp = text.find(' ', at);
  if (sp == std::string::npos) return -1;
  return std::strtol(text.c_str() + sp + 1, nullptr, 10);
}

std::string BytesHuman(size_t bytes) {
  char buf[64];
  if (bytes >= (1ull << 30))
    std::snprintf(buf, sizeof(buf), "%.2f GiB", bytes / 1073741824.0);
  else if (bytes >= (1ull << 20))
    std::snprintf(buf, sizeof(buf), "%.1f MiB", bytes / 1048576.0);
  else
    std::snprintf(buf, sizeof(buf), "%zu B", bytes);
  return buf;
}

}  // namespace

int main(int argc, const char* argv[]) {
  size_t clients = 200;
  size_t obj_size = 33554432;  // KDA TP16 temporal state ≈ 32 MiB lease step
  size_t ops_per_client = 10;
  size_t inline_bytes = 4194304;  // staged-lease threshold; 0 = legacy
  size_t depth = 4;
  size_t small_ratio_pct = 0;

  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string flag = argv[i];
    const char* value = argv[i + 1];
    if (flag == "--clients") clients = ParseSize(value, clients);
    else if (flag == "--obj-size") obj_size = ParseSize(value, obj_size);
    else if (flag == "--ops") ops_per_client = ParseSize(value, ops_per_client);
    else if (flag == "--inline-bytes") inline_bytes = ParseSize(value, inline_bytes);
    else if (flag == "--depth") depth = ParseSize(value, depth);
    else if (flag == "--small-ratio-pct") small_ratio_pct = ParseSize(value, 0);
    else {
      std::fprintf(stderr, "unknown flag %s\n", flag.c_str());
      return 2;
    }
  }

  if (!RdmaTransport::Available()) {
    std::fprintf(stderr, "no RDMA device\n");
    return 1;
  }

  const std::string dir = (fs::temp_directory_path() / "dfkvleasebench")
                              .string();
  fs::remove_all(dir);
  fs::create_directories(dir);

  // Server fixture (same shape as the loopback suite's node: KvNodeServer
  // owns disk, RdmaServer serves data). A hard receive budget that a
  // legacy 32MiB-class * resident connection model could not fit makes the
  // contrast part of correctness: legacy runs must hit oversize/busy
  // rejections, lease runs must pass.
  setenv("DFKV_RDMA_POOL_MAX", "2048", 1);  // let every client keep its conn
  setenv("DFKV_RDMA_RECV_SEGMENT_SIZE", std::to_string(4ull << 30).c_str(), 1);
  setenv("DFKV_RDMA_RECV_CHUNK_BYTES", std::to_string(256ull << 20).c_str(), 1);
  setenv("DFKV_RDMA_MAX_BLOCK_BYTES", std::to_string(1ull << 30).c_str(), 1);
  setenv("DFKV_RDMA_DEPTH", std::to_string(depth).c_str(), 1);
  setenv("DFKV_RDMA_INLINE_PUT_MAX_BYTES",
         std::to_string(inline_bytes).c_str(), 1);
  setenv("DFKV_RDMA_ENDPOINT_CACHE_MAX", "4096", 1);

  auto srv = std::make_unique<KvNodeServer>(dir, 8ull << 30);
  if (srv->Start(0) != Status::kOk) {
    std::fprintf(stderr, "storage node failed to start\n");
    return 1;
  }
  auto rsrv = std::make_unique<RdmaServer>(
      [&srv](uint8_t op, const BlockKey& key, uint64_t off, uint64_t len,
             const char* pl, uint64_t pll, std::string* out,
             size_t* value_len) {
        return srv->ProcessRequestForKey(op, key, off, len, pl, pll, out,
                                         value_len);
      },
      1ull << 30);
  rsrv->set_range_handler(
      [&srv](const BlockKey& key, uint64_t off, uint64_t len, char* io_buf,
             size_t cap, const char** out_data, size_t* out_len,
             size_t* value_len) {
        return srv->RangeDirectForKey(key, off, len, io_buf, cap, out_data,
                                      out_len, value_len);
      });
  rsrv->set_cache_direct_handler(
      [&srv](const BlockKey& key, char* data, size_t len, size_t cap) {
        return srv->CacheDirectForKey(key, data, len, cap);
      });
  if (rsrv->Start(0) != Status::kOk) {
    std::fprintf(stderr, "rdma server failed to start\n");
    return 1;
  }
  const std::string addr = "127.0.0.1:" + std::to_string(rsrv->port());

  std::vector<std::string> payload_large(1, std::string(obj_size, '\0'));
  for (size_t i = 0; i < obj_size; ++i)
    payload_large[0][i] = static_cast<char>((i * 131 + 11) & 0xFF);
  std::string payload_small(256ull << 10, '\0');
  for (size_t i = 0; i < payload_small.size(); ++i)
    payload_small[i] = static_cast<char>((i * 7 + 3) & 0xFF);

  std::atomic<size_t> put_ok{0}, put_fail{0};
  std::array<std::atomic<size_t>, 8> status_counts{};
  std::atomic<uint64_t> put_bytes{0};
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> workers;
  workers.reserve(clients);
  for (size_t c = 0; c < clients; ++c) {
    workers.emplace_back([&, c] {
      RdmaTransport rt(1ull << 30);
      KVClient client({{"n", addr}}, "bench/model-kda", &rt);
      for (size_t o = 0; o < ops_per_client; ++o) {
        const bool small = small_ratio_pct != 0 &&
                           (o * 100 / ops_per_client) < small_ratio_pct;
        const std::string key = "c" + std::to_string(c) + "/o" +
                                std::to_string(o);
        const std::string& payload = small ? payload_small : payload_large[0];
        if (client.Put(key, payload.data(), payload.size())) {
          put_ok.fetch_add(1, std::memory_order_relaxed);
          put_bytes.fetch_add(payload.size(), std::memory_order_relaxed);
        } else {
          put_fail.fetch_add(1, std::memory_order_relaxed);
          // Failure forensics: bucket the raw transport status once so a
          // fan-out regression names its stage (kIOError=4 kQuota=3
          // kFull=2 kInvalid=5).
          std::vector<CacheSrc> src{
              CacheSrc{ToBlockKey("bench/model-kda", key),
                       const_cast<char*>(payload.data()), payload.size()}};
          const Status st = rt.CacheFrom(addr, src)[0];
          size_t code = static_cast<size_t>(st);
          if (code >= status_counts.size()) code = status_counts.size() - 1;
          status_counts[code].fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }
  for (auto& t : workers) t.join();
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();

  const long resident_data =
      MetricOf(*rsrv, "dfkv_rdma_connection_bytes{class=\"data\"}",
               "dfkv_rdma_connection_bytes");
  const long resident_total =
      MetricOf(*rsrv, "dfkv_rdma_recv_segment_used_bytes");
  const long lease_ops = MetricOf(*rsrv, "dfkv_rdma_leaseput_ops_total");
  const long lease_active = MetricOf(*rsrv, "dfkv_rdma_leaseput_active");
  const long lease_bytes_active =
      MetricOf(*rsrv, "dfkv_rdma_leaseput_bytes_active");
  const long busy = MetricOf(*rsrv, "dfkv_rdma_leaseput_busy_rejects_total");
  const long conn_errors =
      MetricOf(*rsrv, "dfkv_rdma_completion_errors");
  const long v2conns = MetricOf(*rsrv, "dfkv_rdma_v2_conns_opened_total");
  const std::string srv_metrics = srv->MetricsText();
  auto line_value = [&srv_metrics](const std::string& key) -> long {
    const size_t at = srv_metrics.rfind(key + " ");
    if (at == std::string::npos) return -1;
    const size_t sp = srv_metrics.find(' ', at);
    if (sp == std::string::npos) return -1;
    return std::strtol(srv_metrics.c_str() + sp + 1, nullptr, 10);
  };
  const long srv_put_io = line_value(
      "dfkv_errors_total{op=\"put\",status=\"io\"}");
  const long srv_invalid = line_value(
      "dfkv_errors_total{op=\"any\",status=\"invalid\"}");
  const long conns = MetricOf(*rsrv, "dfkv_rdma_active_conns");
  const long completion_errors =
      MetricOf(*rsrv, "dfkv_rdma_completion_errors_total");

  const long evictions = MetricOf(*rsrv, "dfkv_rdma_segment_evictions_total");
  const long idle_reclaims = MetricOf(*rsrv, "dfkv_rdma_idle_reclaims_total");

  std::printf(
      "dfkvleasebench inline=%zu obj=%s clients=%zu ops/client=%zu\n"
      "  puts ok=%zu fail=%zu bytes=%llu in %.2fs (%.1f MB/s)"
      " fail[io=%zu quota=%zu full=%zu invalid=%zu other=%zu]\n"
      "  server: active_conns=%ld v2_conns=%ld class-data-resident=%s "
      "recv_used=%s leaseput_ops=%ld leaseput_active=%ld "
      "lease_bytes_active=%s busy_rejects=%ld "
      "completion_errors=%ld evictions=%ld idle_reclaims=%ld"
      " srv_put_io=%ld srv_invalid=%ld\n",
      inline_bytes, BytesHuman(obj_size).c_str(), clients, ops_per_client,
      put_ok.load(), put_fail.load(), put_bytes.load(), seconds,
      put_bytes.load() / seconds / (1ull << 20),
      status_counts[4].load(), status_counts[3].load(),
      status_counts[2].load(), status_counts[5].load(),
      status_counts[6].load() + status_counts[7].load(), conns, v2conns,
      BytesHuman(resident_data < 0 ? 0 : resident_data).c_str(),
      BytesHuman(resident_total < 0 ? 0 : resident_total).c_str(), lease_ops,
      lease_active,
      BytesHuman(lease_bytes_active < 0 ? 0 : lease_bytes_active).c_str(),
      busy, completion_errors, evictions, idle_reclaims, srv_put_io,
      srv_invalid);

  rsrv->Stop();
  srv->Stop();
  fs::remove_all(dir);
  return put_fail.load() == 0 ? 0 : 1;
}
