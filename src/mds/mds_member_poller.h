#ifndef DFKV_MDS_MEMBER_POLLER_H_
#define DFKV_MDS_MEMBER_POLLER_H_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "mds/mds_endpoints.h"
#include "common/membership.h"

namespace dfkv {

// Guards ring adoption against transient view collapses. Two cases, same
// hysteresis: a *successful* ListMembers that is (a) empty, or (b) a shrink
// dropping more than shrink_pct% of the adopted baseline at once — both are
// almost always etcd-outage recovery (mass lease expiry, e.g. NOSPACE) rather
// than a genuine teardown. The old guard only covered (a); a partially-expired
// table ("non-empty but shrunken") sailed through and remapped the whole
// cluster twice (once on the collapse, once on recovery). Suspicious views
// must persist for views_to_accept consecutive polls to be believed. Growth
// and small shrinks (single-node failures) pass through immediately. Pure
// logic; unit-tested without etcd.
class MemberViewGuard {
 public:
  MemberViewGuard(int views_to_accept, int shrink_pct)
      : views_to_accept_(views_to_accept), shrink_pct_(shrink_pct) {}

  // May this successful view be adopted now? false = keep the last ring.
  bool Admit(size_t incoming) {
    if (!have_baseline_) {  // first successful view: adopt as-is
      have_baseline_ = true;
      baseline_ = incoming;
      return true;
    }
    if (incoming == 0 && baseline_ > 0) {
      if (++empty_streak_ >= views_to_accept_) {
        baseline_ = 0; empty_streak_ = 0; shrink_streak_ = 0;
        return true;  // persisted: believe the teardown
      }
      ++rejected_empty_;
      return false;
    }
    empty_streak_ = 0;
    using Wide = unsigned __int128;
    if (shrink_pct_ > 0 && baseline_ > 0 &&
        static_cast<Wide>(incoming) * 100 <
            static_cast<Wide>(baseline_) *
                static_cast<unsigned>(100 - shrink_pct_)) {
      if (++shrink_streak_ >= views_to_accept_) {
        baseline_ = incoming; shrink_streak_ = 0;
        return true;  // persisted: real mass drain
      }
      ++rejected_shrink_;
      return false;
    }
    shrink_streak_ = 0;
    baseline_ = incoming;
    return true;
  }

  uint64_t rejected_empty() const { return rejected_empty_; }
  uint64_t rejected_shrink() const { return rejected_shrink_; }

 private:
  const int views_to_accept_;
  const int shrink_pct_;  // 0 disables the shrink arm (empty arm always on)
  bool have_baseline_ = false;
  size_t baseline_ = 0;  // last admitted view's member count
  int empty_streak_ = 0;
  int shrink_streak_ = 0;
  uint64_t rejected_empty_ = 0;
  uint64_t rejected_shrink_ = 0;
};

// Client-side discovery: periodically polls the MDS for a group's member view
// and invokes on_change(members) whenever its placement-content epoch changes.
// Endpoint selection + failover via MdsEndpoints. One background thread.
class MdsMemberPoller {
 public:
  using OnChange = std::function<void(const std::vector<MemberInfo>&)>;
  MdsMemberPoller(std::vector<std::string> mds_eps, std::string group, OnChange cb,
                  int poll_ms = 3000, int io_timeout_ms = 2000);
  ~MdsMemberPoller();

  void Start();
  void Stop();
  bool PollOnce();
  // Views suppressed by the adoption guard (diagnostic).
  uint64_t empty_view_rejected() const { return guard_.rejected_empty(); }
  uint64_t shrink_view_rejected() const { return guard_.rejected_shrink(); }

  // MDS reachability, surfaced to metrics/logs. The poll thread is otherwise
  // silent, so a mistyped/unreachable mds endpoint leaves the ring empty forever
  // while every op just reports ok=0 with no hint why.
  // ReportHealth() is Loop()'s per-poll helper (query_ok = PollOnce()'s result);
  // it is public so it is unit-testable without a live MDS. The two accessors
  // are thread-safe (atomics) for KVClient::MetricsSnapshot().
  void ReportHealth(bool query_ok);
  uint64_t unreachable_polls_total() const { return unreachable_total_.load(); }
  bool reachable() const { return reachable_.load(); }

 private:
  bool Query(std::vector<MemberInfo>* out, uint64_t* epoch);
  void Loop();
  bool WaitMs(int ms);
  uint64_t NowMs() const;

  MdsEndpoints eps_;
  std::string group_;
  OnChange cb_;
  int poll_ms_;
  int io_ms_;
  uint64_t last_epoch_ = 0;
  bool have_epoch_ = false;
  // MDS reachability diagnostics (ReportHealth). Without these a bad mds
  // endpoint fails 100% silently — the exact trap where "ok=0" writes look like
  // a data/write bug instead of "the ring was never populated".
  bool ever_adopted_ = false;      // has a non-empty view ever been adopted?
  uint64_t consec_fail_ = 0;       // consecutive failed polls (poll-thread only)
  uint64_t last_fail_log_ms_ = 0;  // rate-limit anchor for the sustained-outage WARN
  std::atomic<uint64_t> unreachable_total_{0};  // cumulative failed polls (metric)
  std::atomic<bool> reachable_{true};           // last poll succeeded? (metric)
  static constexpr uint64_t kFailLogEveryMs = 30000;  // outage heartbeat cadence
  // Suspicious-view hysteresis depth (empty and mass-shrink arms alike).
  static constexpr int kViewsToAccept = 3;
  MemberViewGuard guard_;
  std::atomic<bool> running_{false};
  std::thread th_;
  std::mutex mu_;
  std::condition_variable cv_;
};

}  // namespace dfkv

#endif  // DFKV_MDS_MEMBER_POLLER_H_
