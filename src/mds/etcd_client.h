#ifndef DFKV_ETCD_CLIENT_H_
#define DFKV_ETCD_CLIENT_H_

#include <array>
#include <atomic>
#include <mutex>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "utils/http_client.h"
#include "utils/latency_hist.h"

namespace dfkv {

// One key as etcd holds it: absent, or its value and the lease it is
// attached to (0 = none).
struct KeyState {
  bool exists = false;
  std::string value;
  int64_t lease = 0;
};

struct RangeResult {
  std::vector<std::pair<std::string, std::string>> kvs;  // decoded key, value
  int64_t revision = 0;  // etcd header revision — used as the membership epoch
};

// Thin etcd v3 client over the gRPC-gateway JSON API. Holds a non-owning
// HttpTransport so it is unit-testable with a fake. Every call is one POST.
class EtcdClient {
 public:
  // write_idle_window_ms: see dfkv_mds_etcd_write_failing_seconds in
  // MetricsText(). The default is the member lease TTL.
  explicit EtcdClient(HttpTransport* http,
                      uint64_t write_idle_window_ms = 30000)
      : http_(http), write_idle_window_ms_(write_idle_window_ms) {}

  std::optional<int64_t> LeaseGrant(int ttl_seconds);
  bool LeaseKeepAlive(int64_t lease_id);
  bool Put(const std::string& key, const std::string& value, int64_t lease_id);
  std::optional<RangeResult> RangePrefix(const std::string& prefix);
  // One key, with its lease. nullopt = etcd did not answer.
  std::optional<KeyState> Get(const std::string& key);
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
    kLeaseGrant, kLeaseKeepAlive, kPut, kRange, kMetricsRange, kGet, kCount
  };
  bool Post(Op op, const std::string& path, const std::string& body,
            HttpResponse* response);
  std::optional<RangeResult> RangePrefixAs(
      const std::string& prefix, Op op);

  // A refused or failed request, logged with what etcd said, at most once
  // per operation every kFailureLogEveryMs (with the number not logged).
  void NoteFailure(Op op, bool transported, const HttpResponse& response);
  static constexpr uint64_t kFailureLogEveryMs = 10000;
  static uint64_t NowMs();

  HttpTransport* http_;  // not owned
  std::array<LatencyHist, static_cast<size_t>(Op::kCount)> latency_;
  std::array<std::atomic<uint64_t>, static_cast<size_t>(Op::kCount)> requests_{};
  std::array<std::atomic<uint64_t>, static_cast<size_t>(Op::kCount)> errors_{};
  std::mutex failure_log_mu_;
  std::array<uint64_t, static_cast<size_t>(Op::kCount)> failure_logged_ms_{};
  std::array<uint64_t, static_cast<size_t>(Op::kCount)> failures_unlogged_{};
  // Since when writes (lease grant, keepalive, put) have been failing: the
  // first failed write after the last successful one; 0 while they succeed.
  std::atomic<uint64_t> write_failing_since_ms_{0};
  // The last write attempted, successful or not; and how long without one
  // before an ongoing failure is no longer reported (nothing is being lost).
  std::atomic<uint64_t> last_write_attempt_ms_{0};
  const uint64_t write_idle_window_ms_;
};

}  // namespace dfkv

#endif  // DFKV_ETCD_CLIENT_H_
