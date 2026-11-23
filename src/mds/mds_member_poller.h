#ifndef DFKV_MDS_MEMBER_POLLER_H_
#define DFKV_MDS_MEMBER_POLLER_H_

#include <algorithm>
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

// Guards ring adoption against transient view swings. All suspicion verdicts
// are taken against a trusted REFERENCE size, not the last adopted hop: the
// reference only follows a view size that has repeated views_to_accept_
// consecutive polls (or a persisted suspicious view, below), never a one-off
// hop. Three suspicious cases share one hysteresis: a successful topology's
// eligible view that is (a) empty, or (b) a shrink dropping more than
// shrink_pct% of the reference, or (c) growth ballooning past
// +shrink_pct% of it. (a)/(b) are
// almost always etcd-outage recovery (mass lease expiry, e.g. NOSPACE)
// rather than a genuine teardown, and (c) is that same outage unwinding as
// members re-register on their own schedule. Suspicious views must reappear
// as the SAME value for views_to_accept consecutive polls to be believed;
// the reference stays frozen meanwhile. That closes two ratchets an
// adjacent-hop guard has: diffuse mass expiry (heartbeat phases spread over
// polls, each hop below the bar) can no longer walk the anchor down hop by
// hop, and a recovery ramp of ever-changing views can no longer rebuild the
// ring once per epoch. Growth below the bar passes through immediately. A
// smaller view than the reference that is not a mass shrink is believed once
// it repeats (two polls in a row), so a recovery ramp whose views all sit
// above the shrink bar (members re-registering one by one) is not adopted view
// by view. The price: a member loss reaches the ring one poll later, and a
// view that keeps changing is not adopted while it changes. A steady decline
// (N-1, N-2, ... one member per poll) is adopted only once it stops, and a
// member flapping between N and N-1 never drops out of the ring. Clients
// still skip a dead member at once through their peer-health cooldown
// (PeerHealth), so the cost is misses on its keys, not wrong placement of
// data. After a believed empty ring the views of the returning members are
// adopted as they come (better than no ring); the last adopted view then
// stands in for the reference on the shrink side, so one bad poll far below
// the ramp is still held back, and the reference anchors only on a view that
// repeats views_to_accept_ times: the first few members back are not frozen
// in as the whole ring by the growth arm. Pure logic; unit-tested without
// etcd.
class MemberViewGuard {
 public:
  MemberViewGuard(int views_to_accept, int shrink_pct)
      : views_to_accept_(views_to_accept), shrink_pct_(shrink_pct) {}

  // May this successful view be adopted now? false = keep the last ring.
  bool Admit(size_t incoming) {
    // Consecutive sightings of one value: the trusted reference follows only
    // a size that repeats views_to_accept_ polls in a row.
    if (have_baseline_ && incoming == last_seen_) {
      ++same_streak_;
    } else {
      last_seen_ = incoming;
      same_streak_ = 1;
    }

    if (!have_baseline_) {  // first successful view: adopt as-is
      have_baseline_ = true;
      baseline_ = incoming;
      stable_ref_ = incoming;
      return true;
    }

    if (incoming == 0 && baseline_ > 0) {
      suspect_streak_ = 0;  // an empty sighting breaks any suspect streak
      if (++empty_streak_ >= views_to_accept_) {
        baseline_ = 0;
        stable_ref_ = 0;
        empty_streak_ = 0;
        return true;  // persisted: believe the teardown
      }
      ++rejected_empty_;
      return false;
    }
    empty_streak_ = 0;

    using Wide = unsigned __int128;
    // Before the reference has anchored (stable_ref_ == 0, members coming
    // back after a believed teardown) the last adopted view stands in for it,
    // on the shrink side only: one bad poll far below the ramp is held back,
    // and the ramp's growth stays free.
    if (shrink_pct_ > 0 && stable_ref_ == 0 && baseline_ > 0) {
      const Wide scaled = static_cast<Wide>(incoming) * 100;
      const Wide ref = static_cast<Wide>(baseline_);
      if (scaled < ref * static_cast<unsigned>(100 - shrink_pct_)) {
        if (incoming == suspect_value_) {
          ++suspect_streak_;
        } else {
          suspect_value_ = incoming;
          suspect_streak_ = 1;
        }
        if (suspect_streak_ < views_to_accept_) {
          ++rejected_shrink_;
          return false;
        }
        suspect_streak_ = 0;  // persisted: the ramp really fell back
      } else if (incoming < baseline_ &&
                 same_streak_ < std::min(views_to_accept_, kShrinkRepeats)) {
        suspect_streak_ = 0;
        ++rejected_shrink_;
        return false;
      } else {
        suspect_streak_ = 0;
      }
    }
    if (shrink_pct_ > 0 && stable_ref_ > 0) {
      // Deviation arms vs the FROZEN reference: a shrink below (100-pct)% of
      // it, or symmetric growth past (100+pct)% — wide math, no overflow.
      // Either must persist as ONE value; a still-draining or still-growing
      // view changes every poll and restarts the streak.
      const Wide scaled = static_cast<Wide>(incoming) * 100;
      const Wide ref = static_cast<Wide>(stable_ref_);
      const bool shrink_suspect =
          scaled < ref * static_cast<unsigned>(100 - shrink_pct_);
      const bool growth_suspect =
          scaled > ref * static_cast<unsigned>(100 + shrink_pct_);
      if (shrink_suspect || growth_suspect) {
        if (incoming == suspect_value_) {
          ++suspect_streak_;
        } else {
          suspect_value_ = incoming;
          suspect_streak_ = 1;
        }
        if (suspect_streak_ >= views_to_accept_) {
          baseline_ = incoming;
          stable_ref_ = incoming;  // persisted: real mass drain / rebuild
          suspect_streak_ = 0;
          return true;
        }
        ++(shrink_suspect ? rejected_shrink_ : rejected_growth_);
        return false;
      }
      suspect_streak_ = 0;
      // A smaller view above the bar: believed once it repeats.
      if (incoming < stable_ref_ &&
          same_streak_ < std::min(views_to_accept_, kShrinkRepeats)) {
        ++rejected_shrink_;
        return false;
      }
    }

    baseline_ = incoming;
    // The reference only advances to a value that proved stable across
    // consecutive polls, also after a believed teardown (stable_ref_ == 0),
    // where the views of returning members are adopted meanwhile.
    if (same_streak_ >= views_to_accept_) stable_ref_ = incoming;
    return true;
  }

  uint64_t rejected_empty() const { return rejected_empty_; }
  uint64_t rejected_shrink() const { return rejected_shrink_; }
  uint64_t rejected_growth() const { return rejected_growth_; }

 private:
  // Polls in a row a smaller view above the shrink bar must appear for.
  static constexpr int kShrinkRepeats = 2;
  const int views_to_accept_;
  const int shrink_pct_;  // 0 disables the deviation arms (empty arm stays on)
  bool have_baseline_ = false;
  size_t baseline_ = 0;       // last admitted view's member count
  size_t stable_ref_ = 0;     // trusted reference; frozen while under suspicion
  size_t last_seen_ = 0;      // last polled view size, adopted or not
  int same_streak_ = 0;       // consecutive polls reporting last_seen_
  size_t suspect_value_ = 0;  // view size currently held in hysteresis
  int suspect_streak_ = 0;    // consecutive polls of suspect_value_
  int empty_streak_ = 0;
  uint64_t rejected_empty_ = 0;
  uint64_t rejected_shrink_ = 0;
  uint64_t rejected_growth_ = 0;
};

// Client-side discovery: periodically polls the MDS for a group's complete
// topology. The legacy callback remains placement-only. The modern callback
// receives every topology generation plus whether the placement guard admits
// the corresponding eligible-member view; its consumer can forward topology
// immediately while independently deduplicating ring adoption.
// Endpoint selection + failover via MdsEndpoints. One background thread.
class MdsMemberPoller {
 public:
  using OnChange = std::function<void(const std::vector<MemberInfo>&)>;
  using OnTopologyChange =
      std::function<void(const std::vector<MemberInfo>&, uint64_t, bool)>;
  MdsMemberPoller(std::vector<std::string> mds_eps, std::string group,
                  OnChange cb, int poll_ms = 3000,
                  int io_timeout_ms = 2000);
  MdsMemberPoller(std::vector<std::string> mds_eps, std::string group,
                  OnTopologyChange cb, int poll_ms = 3000,
                  int io_timeout_ms = 2000);
  ~MdsMemberPoller();

  // Run `tick` on the poll thread every `every_ms` between polls (absent
  // grace uses it to drop members whose grace passed). Set before Start().
  void SetTick(std::function<void()> tick, int every_ms);
  void Start();
  void Stop();
  bool PollOnce();
  // Views suppressed by the adoption guard (diagnostic): empty, mass-shrink,
  // and the symmetric mass-growth arm (recovery ramp) respectively.
  uint64_t empty_view_rejected() const { return guard_.rejected_empty(); }
  uint64_t shrink_view_rejected() const { return guard_.rejected_shrink(); }
  uint64_t growth_view_rejected() const { return guard_.rejected_growth(); }

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
  OnTopologyChange topology_cb_;
  int poll_ms_;
  int io_ms_;
  std::function<void()> tick_;
  int tick_ms_ = 0;
  uint64_t last_topology_epoch_ = 0;
  uint64_t last_placement_epoch_ = 0;
  bool have_topology_epoch_ = false;
  bool have_placement_epoch_ = false;
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
