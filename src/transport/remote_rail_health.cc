#include "transport/remote_rail_health.h"

#include <algorithm>
#include <limits>
#include <unordered_set>
#include <utility>

namespace dfkv {
namespace {

constexpr uint64_t kHardMaxCooldownUs = 30000000;

uint64_t SaturatingAdd(uint64_t lhs, uint64_t rhs) {
  const uint64_t limit = std::numeric_limits<uint64_t>::max();
  return lhs > limit - rhs ? limit : lhs + rhs;
}

std::string EscapeLabel(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (char ch : value) {
    switch (ch) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      default:
        escaped += ch;
        break;
    }
  }
  return escaped;
}

void AppendValue(std::string* text, const char* name, uint64_t value) {
  text->append(name);
  text->push_back(' ');
  text->append(std::to_string(value));
  text->push_back('\n');
}

void AppendRailValue(std::string* text, const char* name,
                     const std::string& escaped_dev, uint64_t value) {
  text->append(name);
  text->append("{dev=\"");
  text->append(escaped_dev);
  text->append("\"} ");
  text->append(std::to_string(value));
  text->push_back('\n');
}

}  // namespace

RemoteRailHealth::RemoteRailHealth(RemoteRailHealthConfig config)
    : config_(config) {
  config_.max_cooldown_us =
      std::max<uint64_t>(1, std::min(config_.max_cooldown_us,
                                     kHardMaxCooldownUs));
  config_.base_cooldown_us =
      std::max<uint64_t>(1, std::min(config_.base_cooldown_us,
                                     config_.max_cooldown_us));
}

std::optional<RemoteRailLease> RemoteRailHealth::TryAcquire(
    const std::string& peer_id, size_t local_rail, uint64_t now_us) {
  std::lock_guard<std::mutex> lock(mu_);
  Counters& rail_counters = rail_counters_[local_rail];
  if (has_reconciled_ &&
      live_peer_ids_.find(peer_id) == live_peer_ids_.end()) {
    ++total_counters_.admissions_denied;
    ++rail_counters.admissions_denied;
    return std::nullopt;
  }

  RailState& state = peers_[peer_id].rails[local_rail];
  if (Unavailable(state, now_us)) {
    ++total_counters_.admissions_denied;
    ++rail_counters.admissions_denied;
    return std::nullopt;
  }

  const bool recovery_probe = state.consecutive_failures != 0;
  const uint64_t generation = NextGenerationLocked();
  state.active_leases.emplace(
      generation, ActiveLease{state.health_epoch, recovery_probe});
  if (recovery_probe) {
    state.recovery_probe_active = true;
    ++total_counters_.recovery_probes;
    ++rail_counters.recovery_probes;
  } else {
    ++total_counters_.normal_leases;
    ++rail_counters.normal_leases;
  }
  return RemoteRailLease{generation, state.health_epoch, recovery_probe};
}

RemoteRailCompletion RemoteRailHealth::Complete(
    const std::string& peer_id, size_t local_rail, uint64_t generation,
    RemoteRailOutcome outcome, uint64_t now_us) {
  std::lock_guard<std::mutex> lock(mu_);
  auto peer = peers_.find(peer_id);
  if (peer == peers_.end()) return {};
  auto rail = peer->second.rails.find(local_rail);
  if (rail == peer->second.rails.end()) return {};

  RailState& state = rail->second;
  auto lease = state.active_leases.find(generation);
  if (lease == state.active_leases.end()) return {};
  const ActiveLease completed_lease = lease->second;
  state.active_leases.erase(lease);

  Counters& rail_counters = rail_counters_[local_rail];
  if (outcome == RemoteRailOutcome::kAbandon) {
    if (completed_lease.recovery_probe) state.recovery_probe_active = false;
    ++total_counters_.abandons;
    ++rail_counters.abandons;
    return {};
  }

  if (outcome == RemoteRailOutcome::kSuccess) {
    ++total_counters_.successes;
    ++rail_counters.successes;
    if (completed_lease.recovery_probe &&
        state.consecutive_failures != 0) {
      state.consecutive_failures = 0;
      state.cooldown_until_us = 0;
      state.recovery_probe_active = false;
      if (++state.health_epoch == 0) ++state.health_epoch;
      ++total_counters_.recoveries;
      ++rail_counters.recoveries;
      return {RemoteRailTransition::kRecovered, 0};
    }
    if (completed_lease.recovery_probe)
      state.recovery_probe_active = false;
    return {};
  }

  if (!completed_lease.recovery_probe &&
      completed_lease.health_epoch != state.health_epoch) {
    return {};
  }

  const bool entering_cooldown = state.cooldown_until_us <= now_us;
  if (completed_lease.recovery_probe) state.recovery_probe_active = false;
  if (state.consecutive_failures != std::numeric_limits<uint64_t>::max())
    ++state.consecutive_failures;
  const uint64_t deadline =
      SaturatingAdd(now_us, CooldownFor(state.consecutive_failures));
  state.cooldown_until_us = std::max(state.cooldown_until_us, deadline);

  ++total_counters_.endpoint_failures;
  ++rail_counters.endpoint_failures;
  if (entering_cooldown) {
    ++total_counters_.cooldowns;
    ++rail_counters.cooldowns;
    return {RemoteRailTransition::kCooled, state.cooldown_until_us};
  }
  return {RemoteRailTransition::kNone, state.cooldown_until_us};
}

std::vector<uint8_t> RemoteRailHealth::AllowedMask(
    const std::string& peer_id, const std::vector<uint8_t>& candidate_mask,
    uint64_t now_us) const {
  std::vector<uint8_t> allowed = candidate_mask;
  std::lock_guard<std::mutex> lock(mu_);
  auto peer = peers_.find(peer_id);
  if (peer == peers_.end()) return allowed;

  for (size_t local_rail = 0; local_rail < allowed.size(); ++local_rail) {
    if (allowed[local_rail] == 0) continue;
    auto rail = peer->second.rails.find(local_rail);
    if (rail != peer->second.rails.end() &&
        Unavailable(rail->second, now_us)) {
      allowed[local_rail] = 0;
    }
  }
  return allowed;
}

void RemoteRailHealth::Reconcile(
    const std::vector<std::string>& live_peer_ids) {
  std::unordered_set<std::string> live;
  live.reserve(live_peer_ids.size());
  for (const std::string& peer_id : live_peer_ids) live.insert(peer_id);

  std::lock_guard<std::mutex> lock(mu_);
  live_peer_ids_ = std::move(live);
  has_reconciled_ = true;
  for (auto peer = peers_.begin(); peer != peers_.end();) {
    if (live_peer_ids_.find(peer->first) == live_peer_ids_.end()) {
      peer = peers_.erase(peer);
    } else {
      ++peer;
    }
  }
}

std::string RemoteRailHealth::MetricsText(
    const std::vector<std::string>& dev_names) const {
  Counters totals;
  std::vector<Counters> per_rail(dev_names.size());
  {
    std::lock_guard<std::mutex> lock(mu_);
    totals = total_counters_;
    for (size_t rail = 0; rail < per_rail.size(); ++rail) {
      auto counters = rail_counters_.find(rail);
      if (counters != rail_counters_.end()) per_rail[rail] = counters->second;
    }
  }

  std::string text;
  text.reserve(2048 + dev_names.size() * 768);
  text += "# HELP dfkv_rdma_client_remote_rail_normal_leases_total Normal remote endpoint leases granted\n";
  text += "# TYPE dfkv_rdma_client_remote_rail_normal_leases_total counter\n";
  text += "# HELP dfkv_rdma_client_remote_rail_admissions_denied_total Admissions denied by authoritative topology, endpoint cooldown, or an active recovery probe\n";
  text += "# TYPE dfkv_rdma_client_remote_rail_admissions_denied_total counter\n";
  text += "# HELP dfkv_rdma_client_remote_rail_successes_total Completed successful endpoint leases\n";
  text += "# TYPE dfkv_rdma_client_remote_rail_successes_total counter\n";
  text += "# HELP dfkv_rdma_client_remote_rail_failures_total Endpoint failures accounted at lease completion\n";
  text += "# TYPE dfkv_rdma_client_remote_rail_failures_total counter\n";
  text += "# HELP dfkv_rdma_client_remote_rail_cooldowns_total Endpoint cooldown transitions\n";
  text += "# TYPE dfkv_rdma_client_remote_rail_cooldowns_total counter\n";
  text += "# HELP dfkv_rdma_client_remote_rail_recovery_probes_total Single recovery probe leases granted after cooldown\n";
  text += "# TYPE dfkv_rdma_client_remote_rail_recovery_probes_total counter\n";
  text += "# HELP dfkv_rdma_client_remote_rail_recoveries_total Successful endpoint recoveries\n";
  text += "# TYPE dfkv_rdma_client_remote_rail_recoveries_total counter\n";
  text += "# HELP dfkv_rdma_client_remote_rail_abandons_total Leases released without endpoint evidence\n";
  text += "# TYPE dfkv_rdma_client_remote_rail_abandons_total counter\n";
  text += "# TYPE dfkv_rdma_client_remote_rail_normal_leases_by_rail_total counter\n";
  text += "# TYPE dfkv_rdma_client_remote_rail_admissions_denied_by_rail_total counter\n";
  text += "# TYPE dfkv_rdma_client_remote_rail_successes_by_rail_total counter\n";
  text += "# TYPE dfkv_rdma_client_remote_rail_failures_by_rail_total counter\n";
  text += "# TYPE dfkv_rdma_client_remote_rail_cooldowns_by_rail_total counter\n";
  text += "# TYPE dfkv_rdma_client_remote_rail_recovery_probes_by_rail_total counter\n";
  text += "# TYPE dfkv_rdma_client_remote_rail_recoveries_by_rail_total counter\n";
  text += "# TYPE dfkv_rdma_client_remote_rail_abandons_by_rail_total counter\n";

  AppendValue(&text, "dfkv_rdma_client_remote_rail_normal_leases_total",
              totals.normal_leases);
  AppendValue(&text, "dfkv_rdma_client_remote_rail_admissions_denied_total",
              totals.admissions_denied);
  AppendValue(&text, "dfkv_rdma_client_remote_rail_successes_total",
              totals.successes);
  AppendValue(&text, "dfkv_rdma_client_remote_rail_failures_total",
              totals.endpoint_failures);
  AppendValue(&text, "dfkv_rdma_client_remote_rail_cooldowns_total",
              totals.cooldowns);
  AppendValue(&text, "dfkv_rdma_client_remote_rail_recovery_probes_total",
              totals.recovery_probes);
  AppendValue(&text, "dfkv_rdma_client_remote_rail_recoveries_total",
              totals.recoveries);
  AppendValue(&text, "dfkv_rdma_client_remote_rail_abandons_total",
              totals.abandons);

  for (size_t rail = 0; rail < dev_names.size(); ++rail) {
    const std::string dev = EscapeLabel(dev_names[rail]);
    const Counters& counters = per_rail[rail];
    AppendRailValue(&text,
                    "dfkv_rdma_client_remote_rail_normal_leases_by_rail_total",
                    dev, counters.normal_leases);
    AppendRailValue(
        &text,
        "dfkv_rdma_client_remote_rail_admissions_denied_by_rail_total", dev,
        counters.admissions_denied);
    AppendRailValue(&text,
                    "dfkv_rdma_client_remote_rail_successes_by_rail_total", dev,
                    counters.successes);
    AppendRailValue(&text,
                    "dfkv_rdma_client_remote_rail_failures_by_rail_total", dev,
                    counters.endpoint_failures);
    AppendRailValue(&text,
                    "dfkv_rdma_client_remote_rail_cooldowns_by_rail_total", dev,
                    counters.cooldowns);
    AppendRailValue(
        &text,
        "dfkv_rdma_client_remote_rail_recovery_probes_by_rail_total", dev,
        counters.recovery_probes);
    AppendRailValue(&text,
                    "dfkv_rdma_client_remote_rail_recoveries_by_rail_total", dev,
                    counters.recoveries);
    AppendRailValue(&text,
                    "dfkv_rdma_client_remote_rail_abandons_by_rail_total", dev,
                    counters.abandons);
  }
  return text;
}

uint64_t RemoteRailHealth::CooldownFor(
    uint64_t consecutive_failures) const {
  uint64_t cooldown = config_.base_cooldown_us;
  for (uint64_t failure = 1;
       failure < consecutive_failures && cooldown < config_.max_cooldown_us;
       ++failure) {
    if (cooldown > config_.max_cooldown_us / 2) {
      cooldown = config_.max_cooldown_us;
    } else {
      cooldown *= 2;
    }
  }
  return std::min(cooldown, config_.max_cooldown_us);
}

uint64_t RemoteRailHealth::NextGenerationLocked() {
  const uint64_t generation = next_generation_++;
  if (next_generation_ == 0) next_generation_ = 1;
  return generation;
}

bool RemoteRailHealth::Unavailable(const RailState& state, uint64_t now_us) {
  return state.consecutive_failures != 0 &&
         (state.cooldown_until_us > now_us || state.recovery_probe_active);
}

}  // namespace dfkv
