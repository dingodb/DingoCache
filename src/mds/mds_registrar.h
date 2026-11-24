#ifndef DFKV_MDS_REGISTRAR_H_
#define DFKV_MDS_REGISTRAR_H_

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
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
  // An MDS register/heartbeat may spend up to five sequential 2 s etcd
  // request budgets (with DFKV_MDS_FENCING, a read of the fenced ids whose
  // cache ran out; read of the record, failed keepalive of its expired
  // lease, lease grant, Put). Leave one second for TCP framing/scheduling so
  // a valid slow MDS operation cannot self-timeout.
  static constexpr int kDefaultIoTimeoutMs = 11000;
  // A connect that has not completed in this long goes to the next MDS: on a
  // host that drops packets it would otherwise take the whole I/O timeout.
  static constexpr int kConnectTimeoutMs = 2000;
  // How long a failed MDS is passed over. The registrar stays on one MDS
  // while it answers; on a failure it tries the others at once, in the same
  // heartbeat, and comes back to the failed one only after this.
  static constexpr uint64_t kEndpointBackoffMs = 30000;
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
  // Optional health provider (HLT1), called like the stats provider. A beat
  // for which it returns nullopt carries no HLT1 at all, which is what a node
  // without RDMA rails sends while it has nothing to report: to the MDS and
  // to clients "no HLT1" and "HLT1 with zero rails" are different things.
  using HealthFn = std::function<std::optional<MemberHealth>()>;
  void set_health_provider(HealthFn fn) { health_fn_ = std::move(fn); }
  // Optional provider of the self-description (MemberInfo::info), called like
  // the stats provider. Unset = the info given at construction, unchanged.
  using InfoFn = std::function<std::string()>;
  void set_info_provider(InfoFn fn) { info_fn_ = std::move(fn); }
  // Optional provider of MemberInfo::draining (DFKV_DRAIN), called like the
  // stats provider.
  using DrainFn = std::function<bool()>;
  void set_drain_provider(DrainFn fn) { drain_fn_ = std::move(fn); }
  // Wakes the background loop for a heartbeat now (one that starts after
  // this call) and waits up to timeout_ms for it to succeed. Used to get a
  // changed drain state to the MDS without waiting out the interval. False
  // if no such heartbeat succeeded in time, or the loop is stopped.
  bool BeatNow(int timeout_ms);
  // Wakes the background loop for a heartbeat now, without waiting for it.
  // No effect once the loop has stopped.
  void RequestBeat();
  // Deletes this member's record at the MDS (kDeregister). Call after Stop().
  bool Deregister();

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
  bool fenced() const { return fenced_.load(std::memory_order_relaxed); }
  bool heartbeat_healthy() const {
    return registered() && heartbeat_failures_consecutive() == 0;
  }
  std::string MetricsText() const;

  bool RegisterOnce();
  bool HeartbeatOnce();

 private:
  bool SendOnce(uint8_t op, int io_timeout_ms, uint64_t deadline_ms);
  enum class Answer { kOk, kFenced, kFailed };
  Answer SendTo(const std::string& ep, uint8_t op, int io_timeout_ms,
                uint64_t deadline_ms);
  void NoteFenced(bool fenced);
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
  HealthFn health_fn_;
  InfoFn info_fn_;
  DrainFn drain_fn_;
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
  // The MDS answers this member's registers and heartbeats with kNotFound:
  // it is fenced (dfkvctl member fence). Nothing is renewed until unfenced.
  std::atomic<bool> fenced_{false};
  static constexpr uint64_t kFailureLogIntervalMs = 30000;
  std::thread th_;
  std::mutex mu_;
  std::condition_variable cv_;
  // BeatNow: a heartbeat is wanted now; sequence numbers of the last beat
  // started and of the last one that succeeded (registers count). Under mu_.
  bool beat_now_ = false;
  uint64_t beat_started_seq_ = 0;
  uint64_t beat_ok_seq_ = 0;
  bool Beat(bool registering, int io_timeout_ms, uint64_t deadline_ms);
};

}  // namespace dfkv

#endif  // DFKV_MDS_REGISTRAR_H_
