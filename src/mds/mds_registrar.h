#ifndef DFKV_MDS_REGISTRAR_H_
#define DFKV_MDS_REGISTRAR_H_

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <functional>

#include "mds/mds_endpoints.h"
#include "common/membership.h"

namespace dfkv {

// Node-side agent: registers this cache node with the MDS and keeps its lease
// alive via periodic heartbeats. Uses MdsEndpoints for no-LB multi-endpoint
// selection + failover. One background thread; steady_clock drives both the
// endpoint backoff clock and the heartbeat cadence.
//
// With is_client=true the SAME agent registers a cache *consumer* (an inference
// connector instance) under the /clients/<id> prefix instead of a cache node.
// Only the wire op differs (kClientRegister/kClientHeartbeat vs kRegister/
// kHeartbeat); lease/keepalive/failover logic is identical, so consumers get the
// same lease-TTL auto-cleanup as nodes — a dead client's key expires out of etcd
// within kTtlSeconds with no explicit deregister.
class MdsRegistrar {
 public:
  // An MDS register/heartbeat may spend up to three sequential 2 s etcd
  // request budgets (failed keepalive, lease grant, Put). Leave one second for
  // TCP framing/scheduling so a valid slow MDS operation cannot self-timeout.
  static constexpr int kDefaultIoTimeoutMs = 7000;
  // first_registration_timeout_ms == 0 preserves an unbounded registration
  // loop for non-daemon consumers. Cache-node startup supplies its mandatory,
  // bounded deadline and exits fail-closed when it expires.
  MdsRegistrar(std::vector<std::string> mds_eps, std::string group, MemberInfo self,
               int heartbeat_ms = 10000,
               int io_timeout_ms = kDefaultIoTimeoutMs,
               bool is_client = false,
               int first_registration_timeout_ms = 0);
  ~MdsRegistrar();

  void Start();
  void Stop();

  // Optional dynamic-stats provider: called right before EVERY register/
  // heartbeat encode, so the STA1 payload carries values as of that beat
  // (~heartbeat_ms freshness at the MDS). Unset = no STA1 (legacy behavior).
  using StatsFn = std::function<MemberStats()>;
  void set_stats_provider(StatsFn fn) { stats_fn_ = std::move(fn); }

  // Called once after the first successful registration. Configure before
  // Start(); heartbeat failures after that success do not clear the latch.
  using RegisteredFn = std::function<void()>;
  void set_registered_callback(RegisteredFn fn) {
    registered_fn_ = std::move(fn);
  }
  bool registered() const {
    return registered_.load(std::memory_order_acquire);
  }
  bool first_registration_timed_out() const {
    return first_registration_timed_out_.load(std::memory_order_acquire);
  }

  uint64_t heartbeat_failures_consecutive() const {
    return heartbeat_failures_consecutive_.load(std::memory_order_relaxed);
  }
  uint64_t heartbeat_failures_total() const {
    return heartbeat_failures_total_.load(std::memory_order_relaxed);
  }
  uint64_t last_success_age_seconds() const;
  bool heartbeat_healthy() const {
    return registered() && heartbeat_failures_consecutive() == 0;
  }
  std::string MetricsText() const;

  bool RegisterOnce();
  bool HeartbeatOnce();

 private:
  bool SendOnce(uint8_t op, int io_timeout_ms, uint64_t deadline_ms);
  bool RegisterOnceBefore(int io_timeout_ms, uint64_t deadline_ms);
  void MarkFirstRegistrationTimedOut();
  void ReportHeartbeatResult(bool ok);
  void Loop();
  bool WaitMs(int ms);
  uint64_t NowMs() const;
  MdsEndpoints eps_;
  std::string group_;
  MemberInfo self_;
  StatsFn stats_fn_;
  RegisteredFn registered_fn_;
  int hb_ms_;
  int io_ms_;
  bool is_client_;
  int first_registration_timeout_ms_;
  std::atomic<bool> running_{false};
  std::atomic<bool> registered_{false};
  std::atomic<bool> first_registration_timed_out_{false};
  std::atomic<uint64_t> heartbeat_failures_consecutive_{0};
  std::atomic<uint64_t> heartbeat_failures_total_{0};
  std::atomic<uint64_t> last_success_ms_{0};
  std::atomic<uint64_t> last_failure_log_ms_{0};
  static constexpr uint64_t kFailureLogIntervalMs = 30000;
  std::thread th_;
  std::mutex mu_;
  std::condition_variable cv_;
};

}  // namespace dfkv

#endif  // DFKV_MDS_REGISTRAR_H_
