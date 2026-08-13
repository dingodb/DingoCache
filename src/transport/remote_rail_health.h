#ifndef DFKV_TRANSPORT_REMOTE_RAIL_HEALTH_H_
#define DFKV_TRANSPORT_REMOTE_RAIL_HEALTH_H_

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dfkv {

struct RemoteRailHealthConfig {
  uint64_t base_cooldown_us = 1000000;
  uint64_t max_cooldown_us = 30000000;
};

enum class RemoteRailOutcome : uint8_t {
  kSuccess,
  kEndpointFailure,
  kAbandon,
};

struct RemoteRailLease {
  uint64_t generation = 0;
  uint64_t health_epoch = 0;
  bool recovery_probe = false;
};

enum class RemoteRailTransition : uint8_t {
  kNone,
  kCooled,
  kRecovered,
};

struct RemoteRailCompletion {
  RemoteRailTransition transition = RemoteRailTransition::kNone;
  uint64_t cooldown_until_us = 0;
};

// Per-peer, per-local-rail endpoint health. Time-taking operations receive an
// explicit monotonic timestamp so the policy is deterministic in tests and
// independent of the caller's clock source.
class RemoteRailHealth {
 public:
  explicit RemoteRailHealth(RemoteRailHealthConfig config = {});

  // Healthy, live endpoints admit ordinary leases. Once an endpoint has
  // failed, admissions are denied until its cooldown expires; exactly one
  // caller then receives the recovery-probe lease. Before the first Reconcile,
  // unknown identities remain usable for constructor-order compatibility.
  // Afterwards, an identity absent from the latest topology is denied.
  std::optional<RemoteRailLease> TryAcquire(const std::string& peer_id,
                                            size_t local_rail,
                                            uint64_t now_us);

  // Completes a lease at most once. Unknown generations, completions after
  // Reconcile removed an identity, and endpoint failures from health epochs
  // superseded by a successful recovery probe are ignored.
  RemoteRailCompletion Complete(const std::string& peer_id, size_t local_rail,
                                uint64_t generation,
                                RemoteRailOutcome outcome, uint64_t now_us);

  // Removes currently unavailable endpoint rails from a stable local-rail
  // candidate mask. This is an observation only: it never claims a probe.
  std::vector<uint8_t> AllowedMask(const std::string& peer_id,
                                   const std::vector<uint8_t>& candidate_mask,
                                   uint64_t now_us) const;

  // MDS topology becomes authoritative at the first call. All state and
  // outstanding leases belonging to identities absent from live_peer_ids are
  // discarded, and subsequent admissions remain fenced to the latest set.
  void Reconcile(const std::vector<std::string>& live_peer_ids);

  // Prometheus text contains process aggregates and fixed local-device labels
  // only. Peer identities and addresses are never exported as labels.
  std::string MetricsText(const std::vector<std::string>& dev_names) const;

 private:
  struct ActiveLease {
    uint64_t health_epoch = 0;
    bool recovery_probe = false;
  };

  struct RailState {
    uint64_t health_epoch = 0;
    uint64_t consecutive_failures = 0;
    uint64_t cooldown_until_us = 0;
    bool recovery_probe_active = false;
    std::unordered_map<uint64_t, ActiveLease> active_leases;
  };

  struct PeerState {
    std::unordered_map<size_t, RailState> rails;
  };

  struct Counters {
    uint64_t normal_leases = 0;
    uint64_t successes = 0;
    uint64_t admissions_denied = 0;
    uint64_t endpoint_failures = 0;
    uint64_t cooldowns = 0;
    uint64_t recovery_probes = 0;
    uint64_t recoveries = 0;
    uint64_t abandons = 0;
  };

  uint64_t CooldownFor(uint64_t consecutive_failures) const;
  uint64_t NextGenerationLocked();
  static bool Unavailable(const RailState& state, uint64_t now_us);

  RemoteRailHealthConfig config_;
  mutable std::mutex mu_;
  std::unordered_map<std::string, PeerState> peers_;
  std::unordered_set<std::string> live_peer_ids_;
  bool has_reconciled_ = false;
  std::unordered_map<size_t, Counters> rail_counters_;
  Counters total_counters_;
  uint64_t next_generation_ = 1;
};

}  // namespace dfkv

#endif  // DFKV_TRANSPORT_REMOTE_RAIL_HEALTH_H_
