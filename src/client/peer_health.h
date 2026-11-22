#ifndef DFKV_PEER_HEALTH_H_
#define DFKV_PEER_HEALTH_H_

#include <atomic>
#include <cstdint>
#include <list>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "common/status.h"
#include "utils/prom_escape.h"

namespace dfkv {

// Whether a status is something the peer itself said. Every status a server
// puts on the wire is, whatever it says about the request: a hit, a miss, and
// also "cache full" or "quota exceeded", which only a live node that looked at
// the request can answer. Not included: kIOError (a transport failure, or a
// server-side I/O error, which the client cannot tell apart), kInvalid (also
// produced locally for requests that are never sent) and the statuses that
// only this process produces (kResourceExhausted, kNoCompatibleRail).
inline bool PeerAnswered(Status status) {
  return status == Status::kOk || status == Status::kNotFound ||
         status == Status::kCacheFull || status == Status::kQuotaExceeded;
}

// What the statuses of one operation say about the peer it was sent to.
// Answered: the peer is alive. Failed and not answered: a transport failure.
// Neither: the operation says nothing about the peer (it was refused locally,
// or never sent).
struct PeerOutcome {
  bool answered = false;
  bool failed = false;
  void Add(Status status) {
    if (PeerAnswered(status)) answered = true;
    else if (status == Status::kIOError) failed = true;
  }
};

// Layer-1 fast avoidance: a peer (keyed by "ip:port") that fails a transport IO
// is marked unhealthy for cooldown_ms; while unhealthy the client short-circuits
// requests to it (miss) WITHOUT touching the ring. A served response (hit or
// logical miss) clears it. Thread-safe: BatchGet/Put route from many threads.
//
// One incident, one step. A batch reaches a node through several shards and
// several threads, so one failure of the node arrives here as many MarkBad
// calls. Calls made while the peer is already being avoided are observations
// of the incident that started the avoidance: they are counted as errors and
// change nothing else. The back-off grows only when the peer is tried again
// after its cooldown and fails again.
//
// One trial. When a cooldown ends, the peer may still be dead, and finding
// that out can cost a connect or an operation timeout. The first Healthy()
// call after the cooldown returns true and makes its caller the trial; until
// that caller reports (MarkGood clears the peer, MarkBad starts the next,
// longer cooldown) every other caller is still told to skip the peer. A
// trial whose operation says nothing about the peer gives the trial back at
// once (Report with a neutral outcome), so the next caller tries; one that
// never reports at all, because it hangs, lapses after one base cooldown.
// There is therefore at most one trial per base cooldown inside a dead peer,
// not one in total: a trial stuck in a long timeout is joined by the next
// after that time, and when the stuck one finally fails after the cooldown
// that followed has run out, it counts as one more failure.
//
// It is also the client's uniform metrics chokepoint: every op calls Healthy()
// then exactly one of MarkBad()/MarkGood(), so the client-observed counters live
// here at no extra hot-path cost (the mutex is already taken on these calls).
class PeerHealth {
 public:
  // base_cooldown_ms doubles per consecutive failure up to max_cooldown_ms. A
  // FIXED cooldown shorter than the connect timeout (2s < 3s) meant a dead peer
  // was re-dialed on the data path every base period, stalling each batch on a
  // full connect timeout indefinitely; exponential backoff caps that cost at
  // one connect per max_cooldown once a peer is persistently down.
  //
  // busy_grace_ms distinguishes BUSY from DEAD (R2-B, 2026-07-17): under a
  // write burst one stalled op used to MarkBad a node that was concurrently
  // serving other ops fine — the cooldown then instant-failed EVERY op to it
  // for 2-30s (on a small ring: a total artificial outage; measured 4013
  // swallowed backup puts per cold round, and it defeats the hicache put
  // retry, which re-arrives well inside the cooldown). A failure now engages
  // the cooldown only if the peer has had no served response within
  // busy_grace_ms; a peer that answered recently is alive-busy, and the
  // failed op's honest error still reaches its caller. Dead peers never
  // MarkGood, so their first failure cools down exactly as before.
  // How far ahead of a caller's clock another thread's stamp may be and
  // still be taken as current.
  static constexpr uint64_t kStampSkewMs = 1000;
  explicit PeerHealth(uint64_t base_cooldown_ms = 2000,
                      uint64_t max_cooldown_ms = 30000,
                      uint64_t busy_grace_ms = 1000)
      : base_cooldown_ms_(base_cooldown_ms), max_cooldown_ms_(max_cooldown_ms),
        busy_grace_ms_(busy_grace_ms) {}

  // False: skip the peer. True: use it, and report the outcome with MarkGood
  // or MarkBad; for a peer whose cooldown has just ended the caller is the one
  // trial (see above).
  bool Healthy(const std::string& peer, uint64_t now_ms) const {
    std::lock_guard<std::mutex> lk(mu_);
    checks_.fetch_add(1, std::memory_order_relaxed);
    auto it = until_.find(peer);
    if (it == until_.end()) return true;
    if (it->second <= now_ms) {
      uint64_t& trial = trial_until_[peer];
      if (trial <= now_ms) {
        trial = now_ms + base_cooldown_ms_;
        return true;
      }
    }
    unhealthy_skips_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  void MarkBad(const std::string& peer, uint64_t now_ms) {
    std::lock_guard<std::mutex> lk(mu_);
    errors_.fetch_add(1, std::memory_order_relaxed);
    TrackLocked(peer);
    // Alive-busy: the peer served a response within the grace window, so this
    // failure is one slow/failed op on a live node, not a dead peer. Count the
    // error but do NOT engage the cooldown (which would instant-fail every
    // subsequent op to it, including the caller's retry).
    auto lg = last_good_.find(peer);
    // (now_ms may be a little older than the last answer: it was read before
    // the lock, and another thread recorded the answer since. Only a small
    // skew counts as "just answered"; a stamp far ahead is no answer.)
    if (lg != last_good_.end() &&
        (lg->second > now_ms ? lg->second - now_ms <= kStampSkewMs
                             : now_ms - lg->second < busy_grace_ms_)) {
      busy_suppressed_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    // Per-peer errors share the same bounded peer index as every other
    // address-keyed state map. Aggregate errors_ still counts evicted peers.
    ++peer_errors_[peer];
    auto it = until_.find(peer);
    // Already being avoided: this failure belongs to the incident that
    // started the cooldown (an operation that was in flight then, or another
    // shard of the same batch). It neither lengthens nor restarts it.
    if (it != until_.end() && it->second > now_ms) return;
    // A new incident, or the trial after a cooldown failed.
    // Exponential backoff: base << (fails-1), capped at max_cooldown.
    uint32_t fails = ++fail_streak_[peer];
    unsigned shift = fails > 31 ? 31u : (fails - 1);
    uint64_t cd = base_cooldown_ms_ << shift;
    if (cd > max_cooldown_ms_ || cd < base_cooldown_ms_ /*overflow*/) cd = max_cooldown_ms_;
    until_[peer] = now_ms + cd;
    trial_until_.erase(peer);
    marked_bad_.fetch_add(1, std::memory_order_relaxed);  // healthy->bad edge
  }
  // Data-path success: clears the cooldown and the backoff streak, and stamps
  // the peer's last-served time (the busy-vs-dead grace signal). Bumps the
  // served counter (an op completed), so it is NOT for probe-driven recovery.
  void MarkGood(const std::string& peer, uint64_t now_ms) {
    std::lock_guard<std::mutex> lk(mu_);
    served_.fetch_add(1, std::memory_order_relaxed);
    TrackLocked(peer);
    last_good_[peer] = now_ms;
    ClearLocked(peer);
  }
  // The one entry point for what an operation found out. Answered: the peer
  // is alive (MarkGood). Failed: a transport failure (MarkBad), unless the
  // transport refused the operation without dialing, which is no new
  // evidence. Otherwise the operation is no verdict on the peer: if it was
  // the trial, the trial is given back so that the next caller tries now.
  void Report(const std::string& peer, const PeerOutcome& outcome,
              uint64_t now_ms, bool refused_without_dial = false) {
    if (outcome.answered) {
      MarkGood(peer, now_ms);
    } else if (outcome.failed && !refused_without_dial) {
      MarkBad(peer, now_ms);
    } else {
      std::lock_guard<std::mutex> lk(mu_);
      trial_until_.erase(peer);
    }
  }

  // The background prober got an answer from the peer. That shows the peer is
  // reachable and responding; it does not show that reads and writes work (a
  // node whose store has failed answers an existence query with a miss). So
  // it does not clear anything: it ends the running cooldown early, which
  // makes the next request the trial, and the trial decides. It does that
  // once per incident: if the trial then fails, later probe answers are not
  // allowed to shorten the back-off, or the data path would walk into the
  // same failure once per probe period.
  void MarkProbeAlive(const std::string& peer, uint64_t now_ms) {
    std::lock_guard<std::mutex> lk(mu_);
    auto tracked = tracked_.find(peer);
    if (tracked != tracked_.end()) {
      peer_lru_.splice(peer_lru_.end(), peer_lru_, tracked->second);
    }
    auto it = until_.find(peer);
    if (it == until_.end() || it->second <= now_ms) return;  // not avoided
    if (!probe_used_.insert(peer).second) return;
    it->second = now_ms;
  }

  // Prometheus client-side metrics text. Aggregates + per-peer error counts.
  std::string Render() const {
    auto m = [](std::string& s, const char* name, const char* type, const char* help,
                uint64_t v) {
      s += "# HELP "; s += name; s += " "; s += help; s += "\n";
      s += "# TYPE "; s += name; s += " "; s += type; s += "\n";
      s += name; s += " "; s += std::to_string(v); s += "\n";
    };
    std::string s;
    m(s, "dfkv_client_ops_served_total", "counter", "Ops with a served response",
      served_.load(std::memory_order_relaxed));
    m(s, "dfkv_client_io_errors_total", "counter", "Transport IO failures (client-observed)",
      errors_.load(std::memory_order_relaxed));
    m(s, "dfkv_client_health_checks_total", "counter", "Peer health checks performed",
      checks_.load(std::memory_order_relaxed));
    m(s, "dfkv_client_unhealthy_skips_total", "counter",
      "Ops short-circuited (peer in cooldown, or its recovery trial in flight)",
      unhealthy_skips_.load(std::memory_order_relaxed));
    m(s, "dfkv_client_peer_marked_bad_total", "counter", "Healthy->bad peer transitions",
      marked_bad_.load(std::memory_order_relaxed));
    m(s, "dfkv_client_peer_recovered_total", "counter", "Bad->good peer transitions",
      recovered_.load(std::memory_order_relaxed));
    m(s, "dfkv_client_busy_suppressed_total", "counter",
      "Failures on alive-busy peers (cooldown suppressed by grace)",
      busy_suppressed_.load(std::memory_order_relaxed));
    // per-peer errors (one HELP/TYPE, one series per peer seen)
    std::map<std::string, uint64_t> snap;
    {
      std::lock_guard<std::mutex> lk(mu_);
      snap.insert(peer_errors_.begin(), peer_errors_.end());
    }
    s += "# HELP dfkv_client_peer_errors_total IO failures per peer\n";
    s += "# TYPE dfkv_client_peer_errors_total counter\n";
    for (const auto& [peer, n] : snap)
      s += "dfkv_client_peer_errors_total{peer=\"" + PromLabelEscape(peer) + "\"} " +
           std::to_string(n) + "\n";
    return s;
  }

  // test accessors
  uint64_t served() const { return served_.load(std::memory_order_relaxed); }
  uint64_t errors() const { return errors_.load(std::memory_order_relaxed); }
  uint64_t marked_bad() const { return marked_bad_.load(std::memory_order_relaxed); }
  uint64_t recovered() const { return recovered_.load(std::memory_order_relaxed); }
  uint64_t busy_suppressed() const { return busy_suppressed_.load(std::memory_order_relaxed); }
  size_t tracked_peers_for_test() const {
    std::lock_guard<std::mutex> lk(mu_);
    return tracked_.size();
  }
  size_t cooldown_peers_for_test() const {
    std::lock_guard<std::mutex> lk(mu_);
    return until_.size();
  }
  size_t failure_streak_peers_for_test() const {
    std::lock_guard<std::mutex> lk(mu_);
    return fail_streak_.size();
  }
  size_t error_peers_for_test() const {
    std::lock_guard<std::mutex> lk(mu_);
    return peer_errors_.size();
  }
  size_t last_good_peers_for_test() const {
    std::lock_guard<std::mutex> lk(mu_);
    return last_good_.size();
  }

 private:
  // Admit or refresh one peer in the shared LRU. Eviction is deterministic for
  // a given call order and removes the victim from every address-indexed map,
  // so no map can outgrow the single global cardinality bound. Caller holds mu_.
  void TrackLocked(const std::string& peer) {
    auto tracked = tracked_.find(peer);
    if (tracked != tracked_.end()) {
      peer_lru_.splice(peer_lru_.end(), peer_lru_, tracked->second);
      return;
    }
    if (tracked_.size() == kMaxPeers) {
      const std::string victim = peer_lru_.front();
      until_.erase(victim);
      trial_until_.erase(victim);
      probe_used_.erase(victim);
      fail_streak_.erase(victim);
      peer_errors_.erase(victim);
      last_good_.erase(victim);
      tracked_.erase(victim);
      peer_lru_.pop_front();
    }
    peer_lru_.push_back(peer);
    auto pos = peer_lru_.end();
    --pos;
    tracked_.emplace(peer, pos);
  }

  // Clear a peer's cooldown + backoff streak (recovery). Caller holds mu_.
  void ClearLocked(const std::string& peer) {
    fail_streak_.erase(peer);
    trial_until_.erase(peer);
    probe_used_.erase(peer);
    if (until_.erase(peer)) recovered_.fetch_add(1, std::memory_order_relaxed);  // bad->good edge
  }

  static constexpr size_t kMaxPeers = 4096;  // global address-state cardinality cap
  mutable std::mutex mu_;
  uint64_t base_cooldown_ms_;
  uint64_t max_cooldown_ms_;
  uint64_t busy_grace_ms_;
  std::unordered_map<std::string, uint64_t> until_;        // peer -> unhealthy-until (ms)
  // peer -> end of the lease of the trial admitted after its cooldown
  mutable std::unordered_map<std::string, uint64_t> trial_until_;
  // peers whose current incident has had its probe-granted early trial
  std::unordered_set<std::string> probe_used_;
  std::unordered_map<std::string, uint32_t> fail_streak_;  // peer -> consecutive failures
  std::unordered_map<std::string, uint64_t> peer_errors_;  // peer -> cumulative IO errors
  std::list<std::string> peer_lru_;  // oldest at front
  std::unordered_map<std::string, std::list<std::string>::iterator> tracked_;
  std::unordered_map<std::string, uint64_t> last_good_;    // peer -> last served-at (ms)
  mutable std::atomic<uint64_t> checks_{0}, unhealthy_skips_{0}, errors_{0},
      marked_bad_{0}, served_{0}, recovered_{0}, busy_suppressed_{0};
};

}  // namespace dfkv

#endif  // DFKV_PEER_HEALTH_H_
