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
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

#include "transport/transport.h"

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

// Parse DFKV_RDMA_RAIL_TIERS. A missing value synthesizes one homogeneous tier
// containing every configured local device. An explicit value is a strict
// semicolon-separated tier list whose alternatives use '|': "a|b;c". Empty
// names/tiers, duplicates, and names outside DFKV_RDMA_DEV are rejected.
inline bool ParseRailTiers(const std::string& spec,
                           const std::vector<std::string>& local_devices,
                           std::vector<std::vector<uint8_t>>* tiers,
                           std::string* error) {
  if (!tiers) return false;
  tiers->clear();
  if (error) error->clear();
  if (spec.empty()) {
    tiers->push_back(std::vector<uint8_t>(local_devices.size(), 1));
    return true;
  }
  std::vector<uint8_t> seen(local_devices.size(), 0);
  size_t tier_begin = 0;
  while (tier_begin <= spec.size()) {
    const size_t tier_end = spec.find(';', tier_begin);
    const size_t end =
        tier_end == std::string::npos ? spec.size() : tier_end;
    if (end == tier_begin) {
      if (error) *error = "DFKV_RDMA_RAIL_TIERS contains an empty tier";
      tiers->clear();
      return false;
    }
    std::vector<uint8_t> mask(local_devices.size(), 0);
    size_t name_begin = tier_begin;
    while (name_begin <= end) {
      const size_t separator = spec.find('|', name_begin);
      const size_t name_end =
          separator == std::string::npos || separator > end ? end : separator;
      if (name_end == name_begin) {
        if (error) *error = "DFKV_RDMA_RAIL_TIERS contains an empty rail name";
        tiers->clear();
        return false;
      }
      const std::string name = spec.substr(name_begin, name_end - name_begin);
      const auto found =
          std::find(local_devices.begin(), local_devices.end(), name);
      if (found == local_devices.end()) {
        if (error)
          *error = "DFKV_RDMA_RAIL_TIERS rail '" + name +
                   "' is not present in DFKV_RDMA_DEV";
        tiers->clear();
        return false;
      }
      const size_t index =
          static_cast<size_t>(std::distance(local_devices.begin(), found));
      if (seen[index]) {
        if (error)
          *error = "DFKV_RDMA_RAIL_TIERS repeats rail '" + name + "'";
        tiers->clear();
        return false;
      }
      seen[index] = 1;
      mask[index] = 1;
      if (name_end == end) break;
      name_begin = name_end + 1;
    }
    tiers->push_back(std::move(mask));
    if (tier_end == std::string::npos) break;
    tier_begin = tier_end + 1;
  }
  return true;
}

// Return the first (highest-priority) tier with at least one peer-healthy local
// rail. The returned mask contains only that tier's intersection, preventing
// credits, latency, NUMA, or quarantine from spilling into a lower tier.
inline std::vector<uint8_t> HighestCompatibleTier(
    const std::vector<std::vector<uint8_t>>& tiers,
    const std::vector<uint8_t>& peer_healthy) {
  for (const auto& tier : tiers) {
    std::vector<uint8_t> compatible(
        std::max(tier.size(), peer_healthy.size()), 0);
    bool any = false;
    for (size_t i = 0; i < compatible.size(); ++i) {
      compatible[i] =
          i < tier.size() && tier[i] != 0 && i < peer_healthy.size() &&
          peer_healthy[i] != 0;
      any = any || compatible[i] != 0;
    }
    if (any) return compatible;
  }
  return {};
}

struct PeerRailSnapshot {
  uint64_t generation = 0;
  bool complete = false;
  // Peer health by stable local device index. `compatible` is the highest tier
  // for an all-enabled local topology; transports recompute from peer_healthy
  // when runtime local eligibility removes rails.
  std::vector<uint8_t> peer_healthy;
  std::vector<uint8_t> compatible;
};

// Thread-safe immutable snapshot publication keyed by peer address. Explicit
// heterogeneous tiers fail closed until a complete HLT1 topology arrives.
// Without explicit tiers, absent/incomplete topology preserves the legacy
// homogeneous all-local behavior.
class PeerTopologyStore {
 public:
  PeerTopologyStore(std::vector<std::string> local_devices,
                    std::vector<std::vector<uint8_t>> tiers,
                    bool require_complete)
      : local_devices_(std::move(local_devices)),
        tiers_(std::move(tiers)),
        require_complete_(require_complete) {
    auto fallback = std::make_shared<PeerRailSnapshot>();
    fallback->complete = !require_complete_;
    if (!require_complete_) {
      fallback->peer_healthy.assign(local_devices_.size(), 1);
      fallback->compatible.assign(local_devices_.size(), 1);
    }
    fallback_ = std::move(fallback);
  }

  bool Update(const PeerTopology& topology) {
    if (topology.peer_addr.empty()) return false;
    auto next = std::make_shared<PeerRailSnapshot>();
    next->generation = topology.generation;
    next->complete = topology.complete;
    if (topology.complete) {
      std::vector<uint8_t> healthy(local_devices_.size(), 0);
      for (const auto& rail : topology.rails) {
        if (!rail.healthy) continue;
        const auto found =
            std::find(local_devices_.begin(), local_devices_.end(), rail.name);
        if (found != local_devices_.end()) {
          healthy[static_cast<size_t>(
              std::distance(local_devices_.begin(), found))] = 1;
        }
      }
      next->peer_healthy = healthy;
      next->compatible = HighestCompatibleTier(tiers_, healthy);
    } else if (!require_complete_) {
      next->peer_healthy.assign(local_devices_.size(), 1);
      next->compatible.assign(local_devices_.size(), 1);
    }
    std::lock_guard<std::mutex> lock(mu_);
    const auto found = snapshots_.find(topology.peer_addr);
    // generation is a canonical FNV topology token, not a numeric sequence.
    // MdsMemberPoller publishes callbacks in order; compare equality only to
    // deduplicate the same immutable snapshot. Ordering it with < or > would
    // reject arbitrary valid topology changes.
    if (found != snapshots_.end() &&
        topology.generation == found->second->generation)
      return false;
    snapshots_[topology.peer_addr] = std::move(next);
    return true;
  }

  std::shared_ptr<const PeerRailSnapshot> Snapshot(
      const std::string& peer_addr) const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto found = snapshots_.find(peer_addr);
    return found == snapshots_.end() ? fallback_ : found->second;
  }

  bool IsCurrent(const std::string& peer_addr, uint64_t generation) const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto found = snapshots_.find(peer_addr);
    return (found == snapshots_.end() ? fallback_->generation
                                      : found->second->generation) ==
           generation;
  }

 private:
  std::vector<std::string> local_devices_;
  std::vector<std::vector<uint8_t>> tiers_;
  bool require_complete_;
  std::shared_ptr<const PeerRailSnapshot> fallback_;
  mutable std::mutex mu_;
  std::unordered_map<std::string,
                     std::shared_ptr<const PeerRailSnapshot>> snapshots_;
};

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
  // Admission/resource control failure after a lease was selected. Return
  // credits without changing either local-rail or endpoint health.
  kAdmission,
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

  // Empty allowed means every rail. A non-empty excluded mask removes matching
  // rails after a failure in the current logical operation. requested is
  // clamped to the per-rail credit limit; callers must clamp their actual
  // posting window to lease.credits.
  std::optional<RailLease> TryAcquire(
      uint32_t requested, uint64_t now_us,
      const std::vector<uint8_t>& allowed = {},
      const std::vector<uint8_t>& excluded = {}) {
    std::lock_guard<std::mutex> lock(mu_);
    return TryAcquireLocked(requested, now_us, allowed, excluded);
  }

  // Bounded backpressure: wait for a completion or cooldown expiry, but never
  // beyond timeout_us. A zero timeout is a single non-blocking attempt.
  std::optional<RailLease> Acquire(
      uint32_t requested, uint64_t timeout_us,
      const std::vector<uint8_t>& allowed = {},
      const std::vector<uint8_t>& excluded = {}) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::microseconds(timeout_us);
    std::unique_lock<std::mutex> lock(mu_);
    for (;;) {
      const uint64_t now_us = NowMicros();
      if (auto lease =
              TryAcquireLocked(requested, now_us, allowed, excluded))
        return lease;
      if (timeout_us == 0 || std::chrono::steady_clock::now() >= deadline)
        return std::nullopt;
      auto wake = deadline;
      for (size_t i = 0; i < rails_.size(); ++i) {
        const auto& rail = rails_[i];
        if (!Allowed(i, allowed) || Excluded(i, excluded)) continue;
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
    } else if (completion == RailCompletion::kAdmission) {
      // Admission failure is not endpoint evidence. If this lease was the
      // single recovery probe, leave the expired quarantine marker intact and
      // allow a later admission to perform the real probe.
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

  bool Excluded(size_t rail, const std::vector<uint8_t>& excluded) const {
    return rail < excluded.size() && excluded[rail] != 0;
  }

  std::optional<RailLease> TryAcquireLocked(
      uint32_t requested, uint64_t now_us,
      const std::vector<uint8_t>& allowed,
      const std::vector<uint8_t>& excluded) {
    bool selected_recovery = false;
    if (rails_.empty()) return std::nullopt;
    requested =
        std::min(std::max<uint32_t>(1, requested), config_.credits_per_rail);
    size_t selected = rails_.size();
    uint64_t selected_score = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i < rails_.size(); ++i) {
      State& rail = rails_[i];
      if (!Allowed(i, allowed) || Excluded(i, excluded)) continue;
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

// True when the preferred/fallback topology masks contain an enabled rail that
// is not excluded by the current logical operation. This intentionally ignores
// credits, quarantine, cooldown, and current health.
inline bool HasUnexcludedRail(const std::vector<uint8_t>& preferred,
                              const std::vector<uint8_t>& fallback,
                              const std::vector<uint8_t>& excluded) {
  const size_t rails =
      std::max({preferred.size(), fallback.size(), excluded.size()});
  for (size_t i = 0; i < rails; ++i) {
    const bool enabled =
        (i < preferred.size() && preferred[i] != 0) ||
        (i < fallback.size() && fallback[i] != 0);
    if (enabled && !(i < excluded.size() && excluded[i] != 0)) return true;
  }
  return false;
}

inline std::vector<uint8_t> IntersectRailMasks(
    const std::vector<uint8_t>& left, const std::vector<uint8_t>& right) {
  std::vector<uint8_t> out(std::max(left.size(), right.size()), 0);
  for (size_t i = 0; i < out.size(); ++i)
    out[i] = i < left.size() && left[i] != 0 && i < right.size() &&
             right[i] != 0;
  return out;
}


inline std::vector<uint8_t> UnionRailMasks(
    const std::vector<uint8_t>& left, const std::vector<uint8_t>& right) {
  std::vector<uint8_t> out(std::max(left.size(), right.size()), 0);
  for (size_t i = 0; i < out.size(); ++i)
    out[i] = (i < left.size() && left[i] != 0) ||
             (i < right.size() && right[i] != 0);
  return out;
}
inline bool HasRail(const std::vector<uint8_t>& mask) {
  return std::any_of(mask.begin(), mask.end(),
                     [](uint8_t enabled) { return enabled != 0; });
}

// Apply locality only within an already selected highest compatible tier.
// Credit pressure waits on that tier; `fallback` is a NUMA fallback mask, never
// a lower heterogeneous tier.
inline std::optional<RailLease> AcquireHighestTier(
    RailPolicy& policy, uint32_t requested, uint64_t timeout_us,
    const std::vector<uint8_t>& highest_tier,
    const std::vector<uint8_t>& preferred,
    const std::vector<uint8_t>& fallback,
    const std::vector<uint8_t>& excluded = {}) {
  std::vector<uint8_t> local = IntersectRailMasks(highest_tier, preferred);
  std::vector<uint8_t> nonlocal =
      IntersectRailMasks(highest_tier, fallback);
  if (!HasRail(local)) {
    local = std::move(nonlocal);
    nonlocal.clear();
  }
  if (!HasRail(local)) return std::nullopt;
  const uint64_t now = RailPolicy::NowMicros();
  if (auto lease = policy.TryAcquire(requested, now, local, excluded))
    return lease;
  if (HasRail(nonlocal)) {
    if (auto lease = policy.TryAcquire(requested, now, nonlocal, excluded))
      return lease;
  }
  const bool local_available = HasUnexcludedRail(local, {}, excluded);
  return policy.Acquire(requested, timeout_us,
                        local_available || !HasRail(nonlocal) ? local
                                                             : nonlocal,
                        excluded);
}

// Bounded admission that prefers one candidate mask and degrades to a fallback
// mask when no preferred rail can serve immediately. Exclusion is applied to
// both masks. When every preferred topology rail is excluded, bounded waiting
// moves to the fallback mask instead of waiting on an impossible candidate set.
inline std::optional<RailLease> AcquireWithFallback(
    RailPolicy& policy, uint32_t requested, uint64_t timeout_us,
    const std::vector<uint8_t>& preferred,
    const std::vector<uint8_t>& fallback,
    const std::vector<uint8_t>& excluded = {}) {
  const uint64_t now = RailPolicy::NowMicros();
  if (auto lease = policy.TryAcquire(requested, now, preferred, excluded))
    return lease;
  if (!fallback.empty()) {
    if (auto lease = policy.TryAcquire(requested, now, fallback, excluded))
      return lease;
  }
  const bool preferred_available =
      HasUnexcludedRail(preferred, {}, excluded);
  return policy.Acquire(requested, timeout_us,
                        preferred_available || fallback.empty() ? preferred
                                                               : fallback,
                        excluded);
}

}  // namespace rdma
}  // namespace dfkv

#endif  // DFKV_RAIL_SELECT_H_
