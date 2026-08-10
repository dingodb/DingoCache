#ifndef DFKV_ETCD_CLIENT_H_
#define DFKV_ETCD_CLIENT_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "utils/http_client.h"
#include "utils/latency_hist.h"

namespace dfkv {

struct RangeResult {
  std::vector<std::pair<std::string, std::string>> kvs;  // decoded key, value
  int64_t revision = 0;  // etcd header revision — used as the membership epoch
};

// Thin etcd v3 client over the gRPC-gateway JSON API. Holds a non-owning
// HttpTransport so it is unit-testable with a fake. Every call is one POST.
class EtcdClient {
 public:
  explicit EtcdClient(HttpTransport* http) : http_(http) {}

  std::optional<int64_t> LeaseGrant(int ttl_seconds);
  bool LeaseKeepAlive(int64_t lease_id);
  bool Put(const std::string& key, const std::string& value, int64_t lease_id);
  std::optional<RangeResult> RangePrefix(const std::string& prefix);
  // Same bounded range operation, accounted separately so Prometheus scrapes
  // cannot be mistaken for control-plane membership traffic.
  std::optional<RangeResult> RangePrefixForMetrics(
      const std::string& prefix);

  // Prometheus counters and latency histograms for the actual etcd HTTP calls.
  // EtcdClient is shared by every MDS request path, so this covers registration,
  // heartbeats, membership reads, health probes, and scrape-time aggregation.
  std::string MetricsText() const;

 private:
  enum class Op : size_t {
    kLeaseGrant, kLeaseKeepAlive, kPut, kRange, kMetricsRange, kCount
  };
  bool Post(Op op, const std::string& path, const std::string& body,
            HttpResponse* response);
  std::optional<RangeResult> RangePrefixAs(
      const std::string& prefix, Op op);

  HttpTransport* http_;  // not owned
  std::array<LatencyHist, static_cast<size_t>(Op::kCount)> latency_;
  std::array<std::atomic<uint64_t>, static_cast<size_t>(Op::kCount)> requests_{};
  std::array<std::atomic<uint64_t>, static_cast<size_t>(Op::kCount)> errors_{};
};

}  // namespace dfkv

#endif  // DFKV_ETCD_CLIENT_H_
