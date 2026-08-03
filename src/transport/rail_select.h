/* Hardware-independent RDMA multi-rail admission and selection policy. */
#ifndef DFKV_RAIL_SELECT_H_
#define DFKV_RAIL_SELECT_H_

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <vector>

namespace dfkv {
namespace rdma {

// Choose a device index into `dev_node` (NUMA node per device; -1 = unknown).
// NUMA-aware: when numa_on && caller_node>=0 && some device sits on caller_node,
// round-robin among THOSE devices (by rr_tick); otherwise round-robin across all.
// Returns 0 if dev_node is empty (a sentinel — caller must not index an empty set).
inline size_t PickRail(const std::vector<int>& dev_node, int caller_node,
                       bool numa_on, size_t rr_tick) {
  const size_t n = dev_node.size();
  if (n == 0) return 0;
  if (!numa_on || n == 1 || caller_node < 0) return rr_tick % n;
  std::vector<size_t> local;
  for (size_t i = 0; i < n; ++i)
    if (dev_node[i] == caller_node) local.push_back(i);
  if (local.empty()) return rr_tick % n;        // no NUMA-local rail -> all
  return local[rr_tick % local.size()];
}

struct RailPolicyConfig {
  uint32_t credits_per_rail = 64;
  uint32_t error_threshold = 3;
  uint64_t quarantine_us = 5'000'000;
  uint32_t latency_weight = 1;
  uint64_t error_penalty_us = 100'000;
};

struct RailLease {
  size_t rail = 0;
  uint32_t credits = 0;
};

enum class RailCompletion : uint8_t {
  kSuccess,
  // The remote node/bootstrap/protocol failed. Return admission credits without
  // changing this host's HCA health or satisfying a pending recovery probe.
  kEndpointFailure,
  // A local verbs/device/post/CQ operation failed and is evidence against HCA.
  kRailFailure,
};

struct RailStats {
  uint32_t inflight = 0;
  uint32_t credits = 0;
  uint32_t consecutive_errors = 0;
  uint64_t latency_ewma_us = 0;
  uint64_t selections = 0;
  uint64_t credits_exhausted = 0;
  uint64_t errors = 0;
  uint64_t endpoint_errors = 0;
  uint64_t quarantines = 0;
  uint64_t recoveries = 0;
  uint64_t quarantined_until_us = 0;
  bool quarantined = false;
  bool recovery_probe = false;
};

// Thread-safe credit admission and least-inflight selection. Time-taking
// methods accept an explicit monotonic timestamp, keeping failure policy tests
// deterministic and independent of verbs/HCA hardware.
class RailPolicy {
 public:
  explicit RailPolicy(size_t rails, RailPolicyConfig config = {})
      : config_(config), rails_(rails) {
    config_.credits_per_rail =
        std::max<uint32_t>(1, config_.credits_per_rail);
    config_.error_threshold = std::max<uint32_t>(1, config_.error_threshold);
    config_.quarantine_us = std::max<uint64_t>(1, config_.quarantine_us);
    for (auto& rail : rails_) rail.credits = config_.credits_per_rail;
  }

  // Empty allowed means every rail. requested is clamped to the per-rail
  // credit limit; callers must clamp their actual posting window to lease.credits.
  std::optional<RailLease> TryAcquire(
      uint32_t requested, uint64_t now_us,
      const std::vector<uint8_t>& allowed = {}) {
    std::lock_guard<std::mutex> lock(mu_);
    return TryAcquireLocked(requested, now_us, allowed);
  }

  // Bounded backpressure: wait for a completion or cooldown expiry, but never
  // beyond timeout_us. A zero timeout is a single non-blocking attempt.
  std::optional<RailLease> Acquire(
      uint32_t requested, uint64_t timeout_us,
      const std::vector<uint8_t>& allowed = {}) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::microseconds(timeout_us);
    std::unique_lock<std::mutex> lock(mu_);
    for (;;) {
      const uint64_t now_us = NowMicros();
      if (auto lease = TryAcquireLocked(requested, now_us, allowed))
        return lease;
      if (timeout_us == 0 || std::chrono::steady_clock::now() >= deadline)
        return std::nullopt;
      auto wake = deadline;
      for (const auto& rail : rails_) {
        if (rail.quarantined_until_us > now_us) {
          const auto cooldown =
              std::chrono::steady_clock::now() +
              std::chrono::microseconds(rail.quarantined_until_us - now_us);
          wake = std::min(wake, cooldown);
        }
      }
      cv_.wait_until(lock, wake);
    }
  }

  // Completes exactly one lease. Endpoint failures return credits but are
  // neutral to rail health. A quarantined rail admits only one genuine local
  // recovery probe after cooldown; success restores it and local failure starts
  // a fresh cooldown.
  void Complete(const RailLease& lease, uint64_t latency_us,
                RailCompletion completion, uint64_t now_us) {
    std::lock_guard<std::mutex> lock(mu_);
    if (lease.rail >= rails_.size()) return;
    State& rail = rails_[lease.rail];
    const uint32_t returned = std::min(lease.credits, rail.inflight);
    rail.inflight -= returned;
    if (completion == RailCompletion::kSuccess) {
      rail.latency_ewma_us =
          rail.latency_ewma_us == 0
              ? latency_us
              : (rail.latency_ewma_us * 7 + latency_us) / 8;
      rail.consecutive_errors = 0;
      if (rail.recovery_probe) {
        ++rail.recoveries;
        rail.quarantined_until_us = 0;
      }
      rail.recovery_probe = false;
    } else if (completion == RailCompletion::kEndpointFailure) {
      ++rail.endpoint_errors;
      // Keep an expired quarantine marker so the next admission is another
      // real recovery probe. This endpoint did not prove the rail either way.
      rail.recovery_probe = false;
    } else {
      ++rail.errors;
      ++rail.consecutive_errors;
      if (rail.recovery_probe ||
          rail.consecutive_errors >= config_.error_threshold) {
        ++rail.quarantines;
        rail.quarantined_until_us = now_us + config_.quarantine_us;
        rail.recovery_probe = false;
      }
    }
    cv_.notify_all();
  }

  std::vector<RailStats> Snapshot(uint64_t now_us) const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<RailStats> out;
    out.reserve(rails_.size());
    for (const auto& rail : rails_) {
      RailStats stats;
      stats.inflight = rail.inflight;
      stats.credits = rail.credits;
      stats.consecutive_errors = rail.consecutive_errors;
      stats.latency_ewma_us = rail.latency_ewma_us;
      stats.selections = rail.selections;
      stats.credits_exhausted = rail.credits_exhausted;
      stats.errors = rail.errors;
      stats.endpoint_errors = rail.endpoint_errors;
      stats.quarantines = rail.quarantines;
      stats.recoveries = rail.recoveries;
      stats.quarantined = rail.quarantined_until_us != 0;
      stats.recovery_probe = rail.recovery_probe;
      stats.quarantined_until_us =
          rail.quarantined_until_us > now_us ? rail.quarantined_until_us : 0;
      out.push_back(stats);
    }
    return out;
  }

  static uint64_t NowMicros() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
  }

 private:
  struct State {
    uint32_t inflight = 0;
    uint32_t credits = 0;
    uint32_t consecutive_errors = 0;
    uint64_t latency_ewma_us = 0;
    uint64_t selections = 0;
    uint64_t credits_exhausted = 0;
    uint64_t errors = 0;
    uint64_t endpoint_errors = 0;
    uint64_t quarantines = 0;
    uint64_t recoveries = 0;
    uint64_t quarantined_until_us = 0;
    bool recovery_probe = false;
  };

  bool Allowed(size_t rail, const std::vector<uint8_t>& allowed) const {
    return allowed.empty() ||
           (rail < allowed.size() && allowed[rail] != 0);
  }

  std::optional<RailLease> TryAcquireLocked(
      uint32_t requested, uint64_t now_us,
      const std::vector<uint8_t>& allowed) {
    bool selected_recovery = false;
    if (rails_.empty()) return std::nullopt;
    requested =
        std::min(std::max<uint32_t>(1, requested), config_.credits_per_rail);
    size_t selected = rails_.size();
    uint64_t selected_score = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i < rails_.size(); ++i) {
      State& rail = rails_[i];
      if (!Allowed(i, allowed)) continue;
      if (rail.quarantined_until_us > now_us) continue;
      // Cooldown elapsed: this rail may serve as the single recovery probe.
      // The recovery_probe flag is set only when the lease is actually granted
      // to this rail below, never during the scan — a candidate that loses the
      // score comparison must not keep a phantom "probe in flight" state that
      // only a nonexistent completion could clear.
      const bool probe = rail.recovery_probe || rail.quarantined_until_us != 0;
      if (probe && rail.inflight != 0) continue;
      if (requested > rail.credits - rail.inflight) {
        ++rail.credits_exhausted;
        continue;
      }
      const uint64_t occupancy =
          static_cast<uint64_t>(rail.inflight) * 1'000'000 / rail.credits;
      const uint64_t score =
          occupancy + rail.latency_ewma_us * config_.latency_weight +
          static_cast<uint64_t>(rail.consecutive_errors) *
              config_.error_penalty_us;
      if ((probe && !selected_recovery) ||
          (probe == selected_recovery && score < selected_score)) {
        selected = i;
        selected_score = score;
        selected_recovery = probe;
      }
    }
    if (selected == rails_.size()) return std::nullopt;
    State& rail = rails_[selected];
    if (!rail.recovery_probe && rail.quarantined_until_us != 0) {
      // Granting the single post-cooldown admission on this rail: mark the
      // probe in flight. Complete clears the flag exactly once.
      rail.recovery_probe = true;
    }
    rail.inflight += requested;
    ++rail.selections;
    return RailLease{selected, requested};
  }

  RailPolicyConfig config_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::vector<State> rails_;
};

// Bounded admission that prefers one candidate mask and degrades to a
// fallback mask when no preferred rail can serve immediately — every
// preferred rail quarantined or credit-bound — while fallback rails still
// can. Locality stays strictly preferred whenever it is admissible; the
// fallback is only a backstop. At most one lease is ever granted per call;
// when neither mask can serve right away, the caller waits on the preferred
// rails exactly as RailPolicy::Acquire with `preferred` alone would.
inline std::optional<RailLease> AcquireWithFallback(
    RailPolicy& policy, uint32_t requested, uint64_t timeout_us,
    const std::vector<uint8_t>& preferred,
    const std::vector<uint8_t>& fallback) {
  const uint64_t now = RailPolicy::NowMicros();
  if (auto lease = policy.TryAcquire(requested, now, preferred)) return lease;
  if (!fallback.empty()) {
    if (auto lease = policy.TryAcquire(requested, now, fallback)) return lease;
  }
  return policy.Acquire(requested, timeout_us, preferred);
}

}  // namespace rdma
}  // namespace dfkv

#endif  // DFKV_RAIL_SELECT_H_
