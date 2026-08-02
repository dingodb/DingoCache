#ifndef DFKV_PEER_HEALTH_H_
#define DFKV_PEER_HEALTH_H_

#include <atomic>
#include <cstdint>
#include <list>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>

#include "utils/prom_escape.h"

namespace dfkv {

// Layer-1 fast avoidance: a peer (keyed by "ip:port") that fails a transport IO
// is marked unhealthy for cooldown_ms; while unhealthy the client short-circuits
// requests to it (miss) WITHOUT touching the ring. A served response (hit or
// logical miss) clears it. Thread-safe: BatchGet/Put route from many threads.
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
  explicit PeerHealth(uint64_t base_cooldown_ms = 2000,
                      uint64_t max_cooldown_ms = 30000,
                      uint64_t busy_grace_ms = 1000)
      : base_cooldown_ms_(base_cooldown_ms), max_cooldown_ms_(max_cooldown_ms),
        busy_grace_ms_(busy_grace_ms) {}

  bool Healthy(const std::string& peer, uint64_t now_ms) const {
    std::lock_guard<std::mutex> lk(mu_);
    checks_.fetch_add(1, std::memory_order_relaxed);
    auto it = until_.find(peer);
    bool ok = it == until_.end() || it->second <= now_ms;
    if (!ok) unhealthy_skips_.fetch_add(1, std::memory_order_relaxed);
    return ok;
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
    if (lg != last_good_.end() && now_ms - lg->second < busy_grace_ms_) {
      busy_suppressed_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    auto it = until_.find(peer);
    bool was_ok = it == until_.end() || it->second <= now_ms;
    // Exponential backoff: base << (fails-1), capped at max_cooldown.
    uint32_t fails = ++fail_streak_[peer];
    unsigned shift = fails > 31 ? 31u : (fails - 1);
    uint64_t cd = base_cooldown_ms_ << shift;
    if (cd > max_cooldown_ms_ || cd < base_cooldown_ms_ /*overflow*/) cd = max_cooldown_ms_;
    until_[peer] = now_ms + cd;
    if (was_ok) marked_bad_.fetch_add(1, std::memory_order_relaxed);  // healthy->bad edge
    // Per-peer errors share the same bounded peer index as every other
    // address-keyed state map. Aggregate errors_ still counts evicted peers.
    ++peer_errors_[peer];
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
  // Probe-driven recovery (off the data path): the background prober reached
  // the peer, so clear its cooldown/streak WITHOUT counting a served op. This
  // is what lets recovery happen without the data path ever having to re-dial a
  // dead peer inside its (now backed-off) cooldown.
  void MarkProbeAlive(const std::string& peer) {
    std::lock_guard<std::mutex> lk(mu_);
    auto tracked = tracked_.find(peer);
    if (tracked != tracked_.end()) {
      peer_lru_.splice(peer_lru_.end(), peer_lru_, tracked->second);
    }
    ClearLocked(peer);
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
    m(s, "dfkv_client_unhealthy_skips_total", "counter", "Ops short-circuited (peer in cooldown)",
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
    if (until_.erase(peer)) recovered_.fetch_add(1, std::memory_order_relaxed);  // bad->good edge
  }

  static constexpr size_t kMaxPeers = 4096;  // global address-state cardinality cap
  mutable std::mutex mu_;
  uint64_t base_cooldown_ms_;
  uint64_t max_cooldown_ms_;
  uint64_t busy_grace_ms_;
  std::unordered_map<std::string, uint64_t> until_;        // peer -> unhealthy-until (ms)
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
