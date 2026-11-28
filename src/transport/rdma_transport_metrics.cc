#include "transport/rdma_transport_internal.h"

#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "transport/rdma_verbs.h"
#include "utils/prom_escape.h"

namespace dfkv {

std::string RdmaTransport::MetricsText() const {
  std::string s;
  s += "# HELP dfkv_rdma_client_conns_opened_total RDMA client connections opened\n";
  s += "# TYPE dfkv_rdma_client_conns_opened_total counter\n";
  s += "dfkv_rdma_client_conns_opened_total " +
       std::to_string(conns_opened_.load(std::memory_order_relaxed)) + "\n";
  std::map<ConnectionClass, ConnectionClassStats> class_stats;
  {
    std::lock_guard<std::mutex> lock(connection_class_mu_);
    class_stats = connection_class_stats_;
  }
  std::map<ConnectionClass, uint64_t> idle_classes;
  {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& [node, connections] : pool_) {
      (void)node;
      for (const Conn* conn : connections)
        ++idle_classes[{conn->declared_bytes, conn->depth}];
    }
  }
  s += "# HELP dfkv_rdma_client_connection_class_opened_total Data QPs opened by adaptive block/depth class\n";
  s += "# TYPE dfkv_rdma_client_connection_class_opened_total counter\n";
  s += "# HELP dfkv_rdma_client_connection_class_active Data QPs currently executing by adaptive block/depth class\n";
  s += "# TYPE dfkv_rdma_client_connection_class_active gauge\n";
  for (const auto& [geometry, stats] : class_stats) {
    const std::string labels =
        "{block_bytes=\"" + std::to_string(geometry.first) +
        "\",depth=\"" + std::to_string(geometry.second) + "\"} ";
    s += "dfkv_rdma_client_connection_class_opened_total" + labels +
         std::to_string(stats.opened) + "\n";
    s += "dfkv_rdma_client_connection_class_active" + labels +
         std::to_string(stats.active) + "\n";
  }
  s += "# HELP dfkv_rdma_client_connection_class_idle Data QPs currently idle by adaptive block/depth class\n";
  s += "# TYPE dfkv_rdma_client_connection_class_idle gauge\n";
  for (const auto& [geometry, idle] : idle_classes) {
    s += "dfkv_rdma_client_connection_class_idle{block_bytes=\"" +
         std::to_string(geometry.first) + "\",depth=\"" +
         std::to_string(geometry.second) + "\"} " + std::to_string(idle) +
         "\n";
  }
  const rdma::ResourceRequest resource_used = resource_budget_->used();
  const rdma::ResourceRequest resource_limit = resource_budget_->limit();
  const auto resource_gauge = [&](const char* name, const char* help,
                                  uint64_t used, uint64_t limit) {
    s += "# HELP " + std::string(name) + " " + help + "\n";
    s += "# TYPE " + std::string(name) + " gauge\n";
    s += std::string(name) + "{kind=\"used\"} " + std::to_string(used) + "\n";
    s += std::string(name) + "{kind=\"limit\"} " + std::to_string(limit) +
         "\n";
  };
  resource_gauge("dfkv_rdma_client_endpoint_budget",
                 "Process-wide RDMA endpoint budget", resource_used.endpoints,
                 resource_limit.endpoints);
  resource_gauge("dfkv_rdma_client_qp_budget",
                 "Process-wide RDMA QP budget", resource_used.qps,
                 resource_limit.qps);
  resource_gauge("dfkv_rdma_client_wr_slot_budget",
                 "Process-wide RDMA work-request slot budget",
                 resource_used.wr_slots, resource_limit.wr_slots);
  resource_gauge("dfkv_rdma_client_registered_slot_bytes_budget",
                 "Process-wide remote resident-slot byte admission budget",
                 resource_used.registered_bytes,
                 resource_limit.registered_bytes);
  std::vector<std::vector<uint64_t>> idle_by_lane_rail(
      3, std::vector<uint64_t>(devs_.size(), 0));
  {
    std::lock_guard<std::mutex> lock(mu_);
    const auto count = [&](const auto& pools) {
      for (const auto& [node, connections] : pools) {
        (void)node;
        for (const Conn* conn : connections) {
          const size_t lane =
              conn->lane == Lane::kData
                  ? 0
                  : (conn->lane == Lane::kSgData ? 1 : 2);
          if (conn->rail_index < devs_.size())
            ++idle_by_lane_rail[lane][conn->rail_index];
        }
      }
    };
    count(pool_);
    count(control_pool_);
  }
  s += "# HELP dfkv_rdma_client_pool_connections Endpoint ownership by operation lane, state, and rail\n";
  s += "# TYPE dfkv_rdma_client_pool_connections gauge\n";
  constexpr const char* lane_names[] = {"payload", "sg", "control"};
  for (size_t lane = 0; lane < idle_by_lane_rail.size(); ++lane) {
    for (size_t rail = 0; rail < devs_.size(); ++rail) {
      const std::string dev =
          devs_[rail].empty() ? std::string("default") : devs_[rail];
      const std::string labels =
          "{lane=\"" + std::string(lane_names[lane]) + "\",state=\"";
      s += "dfkv_rdma_client_pool_connections" + labels + "idle\",dev=\"" +
           dev + "\"} " +
           std::to_string(idle_by_lane_rail[lane][rail]) + "\n";
      s += "dfkv_rdma_client_pool_connections" + labels + "active\",dev=\"" +
           dev + "\"} " +
           std::to_string(
               active_lane_rail_[lane * devs_.size() + rail].load(
                   std::memory_order_relaxed)) +
           "\n";
    }
  }
  s += "# HELP dfkv_rdma_client_peer_connections Live endpoints by peer and local rail\n";
  s += "# TYPE dfkv_rdma_client_peer_connections gauge\n";
  {
    std::lock_guard<std::mutex> lock(peer_connections_mu_);
    for (const auto& [key, count] : peer_connections_) {
      const std::string dev =
          devs_[key.second].empty() ? std::string("default")
                                   : devs_[key.second];
      s += "dfkv_rdma_client_peer_connections{peer=\"" +
           PromLabelEscape(key.first) + "\",dev=\"" + PromLabelEscape(dev) +
           "\"} " + std::to_string(count) + "\n";
    }
  }
  s += "# HELP dfkv_rdma_client_pool_limit Idle endpoints retained per peer and pool\n";
  s += "# TYPE dfkv_rdma_client_pool_limit gauge\n";
  s += "dfkv_rdma_client_pool_limit " + std::to_string(pool_max_) + "\n";
  s += "# HELP dfkv_rdma_client_resource_budget_timeouts_total Connection admissions rejected after bounded resource wait\n";
  s += "# TYPE dfkv_rdma_client_resource_budget_timeouts_total counter\n";
  s += "dfkv_rdma_client_resource_budget_timeouts_total " +
       std::to_string(resource_budget_->timeouts()) + "\n";
  s += "# HELP dfkv_rdma_client_resource_budget_raises_total Autoscale limit raises applied to the process budget\n";
  s += "# TYPE dfkv_rdma_client_resource_budget_raises_total counter\n";
  s += "dfkv_rdma_client_resource_budget_raises_total " +
       std::to_string(resource_budget_->raises()) + "\n";
  s += "# HELP dfkv_rdma_client_admission_failures_total Operations failed kResourceExhausted by local budget starvation (peer never dialed)\n";
  s += "# TYPE dfkv_rdma_client_admission_failures_total counter\n";
  s += "dfkv_rdma_client_admission_failures_total " +
       std::to_string(admission_failures_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_depth_refunds_total Connections whose clamped depth negotiation refunded WR/registered budget\n";
  s += "# TYPE dfkv_rdma_client_depth_refunds_total counter\n";
  s += "dfkv_rdma_client_depth_refunds_total " +
       std::to_string(depth_refunds_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_topology_nodes Ring size last hinted to the transport budget autoscaler (0=never hinted or autoscale off)\n";
  s += "# TYPE dfkv_rdma_client_topology_nodes gauge\n";
  s += "dfkv_rdma_client_topology_nodes " +
       std::to_string(topology_nodes_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_endpoint_cache_hits_total Idle endpoint cache hits\n";
  s += "# TYPE dfkv_rdma_endpoint_cache_hits_total counter\n";
  s += "dfkv_rdma_endpoint_cache_hits_total " +
       std::to_string(endpoint_cache_hits_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_endpoint_cache_misses_total Endpoint acquisitions requiring a new QP\n";
  s += "# TYPE dfkv_rdma_endpoint_cache_misses_total counter\n";
  s += "dfkv_rdma_endpoint_cache_misses_total " +
       std::to_string(endpoint_cache_misses_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_endpoint_cache_evictions_total Idle endpoints evicted under process-wide resource pressure\n";
  s += "# TYPE dfkv_rdma_endpoint_cache_evictions_total counter\n";
  s += "dfkv_rdma_endpoint_cache_evictions_total " +
       std::to_string(
           endpoint_cache_evictions_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_v2_put_writes_total PUT requests sent with RDMA WRITE_WITH_IMM\n";
  s += "# TYPE dfkv_rdma_client_v2_put_writes_total counter\n";
  s += "dfkv_rdma_client_v2_put_writes_total " +
       std::to_string(v2_put_writes_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_mr_regions Successfully published host MR regions\n";
  s += "# TYPE dfkv_rdma_client_mr_regions gauge\n";
  s += "dfkv_rdma_client_mr_regions " +
       std::to_string(mr_regions_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_mr_registered_bytes Bytes in successfully published host MR declarations\n";
  s += "# TYPE dfkv_rdma_client_mr_registered_bytes gauge\n";
  s += "dfkv_rdma_client_mr_registered_bytes " +
       std::to_string(
           mr_registered_bytes_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_mr_registration_rejections_total Pool MR declarations rejected before publication\n";
  s += "# TYPE dfkv_rdma_client_mr_registration_rejections_total counter\n";
  s += "dfkv_rdma_client_mr_registration_rejections_total " +
       std::to_string(
           mr_registration_rejections_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_rail_conns_total Connections opened per rail (device)\n";
  s += "# TYPE dfkv_rdma_client_rail_conns_total counter\n";
  s += "# HELP dfkv_rdma_client_rail_put_ops_total Successful logical PUT objects per local RDMA rail\n";
  s += "# TYPE dfkv_rdma_client_rail_put_ops_total counter\n";
  s += "# HELP dfkv_rdma_client_rail_put_bytes_total Successful logical PUT payload bytes per local RDMA rail\n";
  s += "# TYPE dfkv_rdma_client_rail_put_bytes_total counter\n";
  s += "# HELP dfkv_rdma_client_rail_get_ops_total Successful logical GET objects per local RDMA rail\n";
  s += "# TYPE dfkv_rdma_client_rail_get_ops_total counter\n";
  s += "# HELP dfkv_rdma_client_rail_get_bytes_total Successful logical GET payload bytes per local RDMA rail\n";
  s += "# TYPE dfkv_rdma_client_rail_get_bytes_total counter\n";
  for (size_t i = 0; i < devs_.size(); ++i) {
    const std::string& d = devs_[i].empty() ? std::string("default") : devs_[i];
    s += "dfkv_rdma_client_rail_conns_total{dev=\"" + d + "\"} " +
         std::to_string(rail_conns_[i].load(std::memory_order_relaxed)) + "\n";
    const std::string labels = "{dev=\"" + d + "\"} ";
    s += "dfkv_rdma_client_rail_put_ops_total" + labels +
         std::to_string(rail_put_ops_[i].load(std::memory_order_relaxed)) + "\n";
    s += "dfkv_rdma_client_rail_put_bytes_total" + labels +
         std::to_string(rail_put_bytes_[i].load(std::memory_order_relaxed)) + "\n";
    s += "dfkv_rdma_client_rail_get_ops_total" + labels +
         std::to_string(rail_get_ops_[i].load(std::memory_order_relaxed)) + "\n";
    s += "dfkv_rdma_client_rail_get_bytes_total" + labels +
         std::to_string(rail_get_bytes_[i].load(std::memory_order_relaxed)) + "\n";
  }
  // Lazily populated: a series appears with its first failure, so the label
  // set stays at (rails x statuses that actually occurred).
  s += "# HELP dfkv_rdma_client_rail_wc_errors_total Connections lost to a failed work completion per local RDMA rail and completion status\n";
  s += "# TYPE dfkv_rdma_client_rail_wc_errors_total counter\n";
  for (size_t i = 0; i < devs_.size(); ++i) {
    for (size_t slot = 0; slot < rdma::kWcStatusSlots; ++slot) {
      const uint64_t count =
          rail_wc_errors_[i * rdma::kWcStatusSlots + slot].load(
              std::memory_order_relaxed);
      if (count == 0) continue;
      s += "dfkv_rdma_client_rail_wc_errors_total{dev=\"" +
           PromLabelEscape(devs_[i].empty() ? std::string("default")
                                            : devs_[i]) +
           "\",status=\"" + rdma::WcStatusName(static_cast<int>(slot)) +
           "\"} " + std::to_string(count) + "\n";
    }
  }
  const uint64_t now = rdma::RailPolicy::NowMicros();
  const auto rail_stats = rail_policy_->Snapshot(now);
  s += "# HELP dfkv_rdma_client_rail_selections_total Operations selected per RDMA rail\n";
  s += "# TYPE dfkv_rdma_client_rail_selections_total counter\n";
  s += "# HELP dfkv_rdma_client_rail_inflight Reserved outstanding request credits per RDMA rail\n";
  s += "# TYPE dfkv_rdma_client_rail_inflight gauge\n";
  s += "# HELP dfkv_rdma_client_rail_credits_available Available outstanding request credits per RDMA rail\n";
  s += "# TYPE dfkv_rdma_client_rail_credits_available gauge\n";
  s += "# HELP dfkv_rdma_client_rail_credits_exhausted_total Selection candidates skipped because request credits were exhausted\n";
  s += "# TYPE dfkv_rdma_client_rail_credits_exhausted_total counter\n";
  s += "# HELP dfkv_rdma_client_rail_errors_total Local verbs, post, CQ, or device failures attributed to an RDMA rail\n";
  s += "# TYPE dfkv_rdma_client_rail_errors_total counter\n";
  s += "# HELP dfkv_rdma_client_endpoint_errors_total Remote endpoint, bootstrap, or protocol failures released without penalizing the rail\n";
  s += "# TYPE dfkv_rdma_client_endpoint_errors_total counter\n";
  s += "# HELP dfkv_rdma_client_rail_consecutive_errors Consecutive local rail failures used by quarantine policy\n";
  s += "# TYPE dfkv_rdma_client_rail_consecutive_errors gauge\n";
  s += "# HELP dfkv_rdma_client_rail_quarantines_total Rail quarantine transitions after consecutive failures\n";
  s += "# TYPE dfkv_rdma_client_rail_quarantines_total counter\n";
  s += "# HELP dfkv_rdma_client_rail_recoveries_total Successful rail recovery probes after cooldown\n";
  s += "# TYPE dfkv_rdma_client_rail_recoveries_total counter\n";
  s += "# HELP dfkv_rdma_client_rail_quarantined Whether a rail is currently quarantined\n";
  s += "# TYPE dfkv_rdma_client_rail_quarantined gauge\n";
  s += "# HELP dfkv_rdma_client_rail_recovery_probe Whether the quarantined rail currently has its single recovery admission in flight\n";
  s += "# TYPE dfkv_rdma_client_rail_recovery_probe gauge\n";
  s += "# HELP dfkv_rdma_client_rail_latency_ewma_us Rail completion latency EWMA used by selection policy\n";
  s += "# TYPE dfkv_rdma_client_rail_latency_ewma_us gauge\n";
  for (size_t i = 0; i < rail_stats.size(); ++i) {
    const std::string& d =
        devs_[i].empty() ? std::string("default") : devs_[i];
    const std::string labels = "{dev=\"" + d + "\"} ";
    const auto& stats = rail_stats[i];
    s += "dfkv_rdma_client_rail_selections_total" + labels +
         std::to_string(stats.selections) + "\n";
    s += "dfkv_rdma_client_rail_inflight" + labels +
         std::to_string(stats.inflight) + "\n";
    s += "dfkv_rdma_client_rail_credits_available" + labels +
         std::to_string(stats.credits - stats.inflight) + "\n";
    s += "dfkv_rdma_client_rail_credits_exhausted_total" + labels +
         std::to_string(stats.credits_exhausted) + "\n";
    s += "dfkv_rdma_client_rail_errors_total" + labels +
         std::to_string(stats.errors) + "\n";
    s += "dfkv_rdma_client_endpoint_errors_total" + labels +
         std::to_string(stats.endpoint_errors) + "\n";
    s += "dfkv_rdma_client_rail_consecutive_errors" + labels +
         std::to_string(stats.consecutive_errors) + "\n";
    s += "dfkv_rdma_client_rail_quarantines_total" + labels +
         std::to_string(stats.quarantines) + "\n";
    s += "dfkv_rdma_client_rail_recoveries_total" + labels +
         std::to_string(stats.recoveries) + "\n";
    s += "dfkv_rdma_client_rail_quarantined" + labels +
         std::to_string(stats.quarantined ? 1 : 0) + "\n";
    s += "dfkv_rdma_client_rail_recovery_probe" + labels +
         std::to_string(stats.recovery_probe ? 1 : 0) + "\n";
    s += "dfkv_rdma_client_rail_latency_ewma_us" + labels +
         std::to_string(stats.latency_ewma_us) + "\n";
  }
  s += "# HELP dfkv_rdma_client_numa_fallbacks_total NUMA-aware admissions that fell back to all enabled rails\n";
  s += "# TYPE dfkv_rdma_client_numa_fallbacks_total counter\n";
  s += "dfkv_rdma_client_numa_fallbacks_total{reason=\"caller_unknown\"} " +
       std::to_string(
           numa_caller_unknown_fallbacks_.load(std::memory_order_relaxed)) +
       "\n";
  s += "dfkv_rdma_client_numa_fallbacks_total{reason=\"no_local_rail\"} " +
       std::to_string(
           numa_no_local_fallbacks_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_cq_completions_total Work completions reaped from RDMA completion queues\n";
  s += "# TYPE dfkv_rdma_cq_completions_total counter\n";
  s += "dfkv_rdma_cq_completions_total " +
       std::to_string(rdma::RcEndpoint::CqCompletions()) + "\n";
  s += "# HELP dfkv_rdma_cq_errors_total Completion queue polling and failed work completion errors\n";
  s += "# TYPE dfkv_rdma_cq_errors_total counter\n";
  s += "dfkv_rdma_cq_errors_total " +
       std::to_string(rdma::RcEndpoint::CqErrors()) + "\n";
  // Effective per-connection posting depth. Aggregate concurrency is bounded
  // independently by the per-rail credit policy above.
  s += "# HELP dfkv_rdma_client_pipeline_depth Configured RDMA pipeline depth ceiling (env DFKV_RDMA_DEPTH); per-node connections may open at a lower learned negotiated depth\n";
  s += "# TYPE dfkv_rdma_client_pipeline_depth gauge\n";
  s += "dfkv_rdma_client_pipeline_depth " + std::to_string(depth_) + "\n";
  s += "# HELP dfkv_rdma_client_adhoc_user_mr_total One-shot user MRs registered outside explicit pool regions\n";
  s += "# TYPE dfkv_rdma_client_adhoc_user_mr_total counter\n";
  s += "dfkv_rdma_client_adhoc_user_mr_total " +
       std::to_string(rdma::RcEndpoint::AdhocUserMrTotal()) + "\n";
  s += "# HELP dfkv_rdma_client_transient_user_mr_active Live one-shot out-of-pool user MRs\n";
  s += "# TYPE dfkv_rdma_client_transient_user_mr_active gauge\n";
  s += "dfkv_rdma_client_transient_user_mr_active " +
       std::to_string(rdma::RcEndpoint::TransientUserMrActive()) + "\n";
  s += "# HELP dfkv_rdma_client_pool_mr_registrations_total Actual pool-region ibv_reg_mr calls\n";
  s += "# TYPE dfkv_rdma_client_pool_mr_registrations_total counter\n";
  s += "dfkv_rdma_client_pool_mr_registrations_total " +
       std::to_string(rdma::RcEndpoint::PoolMrRegistrations()) + "\n";
  s += "# HELP dfkv_rdma_client_pool_mr_registration_failures_total Failed explicit pool-region registrations\n";
  s += "# TYPE dfkv_rdma_client_pool_mr_registration_failures_total counter\n";
  s += "dfkv_rdma_client_pool_mr_registration_failures_total " +
       std::to_string(rdma::RcEndpoint::PoolMrRegistrationFailures()) + "\n";
  s += "# HELP dfkv_rdma_client_pool_mr_active_registrations Live shared-PD pool MR generations\n";
  s += "# TYPE dfkv_rdma_client_pool_mr_active_registrations gauge\n";
  s += "dfkv_rdma_client_pool_mr_active_registrations " +
       std::to_string(rdma::RcEndpoint::PoolMrActiveRegistrations()) + "\n";
  s += "# HELP dfkv_rdma_client_max_block_seen_bytes Largest requested RDMA block or destination capacity\n";
  s += "# TYPE dfkv_rdma_client_max_block_seen_bytes gauge\n";
  s += "dfkv_rdma_client_max_block_seen_bytes " +
       std::to_string(max_block_seen_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_declared_max_block_bytes Logical RDMA block safety ceiling\n";
  s += "# TYPE dfkv_rdma_client_declared_max_block_bytes gauge\n";
  s += "dfkv_rdma_client_declared_max_block_bytes " +
       std::to_string(declared_) + "\n";
  s += "# HELP dfkv_rdma_client_connection_min_block_bytes Smallest adaptive data-connection declaration\n";
  s += "# TYPE dfkv_rdma_client_connection_min_block_bytes gauge\n";
  s += "dfkv_rdma_client_connection_min_block_bytes " +
       std::to_string(connection_min_block_bytes_) + "\n";
  s += "# HELP dfkv_rdma_client_oversize_rejects_total Operations rejected above the declared maximum block\n";
  s += "# TYPE dfkv_rdma_client_oversize_rejects_total counter\n";
  s += "dfkv_rdma_client_oversize_rejects_total " +
       std::to_string(oversize_rejects_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_inline_put_max_bytes Leased-PUT inline threshold (env DFKV_RDMA_INLINE_PUT_MAX_BYTES); objects above it use per-op staging leases\n";
  s += "# TYPE dfkv_rdma_client_inline_put_max_bytes gauge\n";
  s += "dfkv_rdma_client_inline_put_max_bytes " +
       std::to_string(inline_put_max_bytes_) + "\n";
  s += "# HELP dfkv_rdma_client_leaseput_ops_total Objects delivered through per-op staging leases\n";
  s += "# TYPE dfkv_rdma_client_leaseput_ops_total counter\n";
  s += "dfkv_rdma_client_leaseput_ops_total " +
       std::to_string(leaseput_ops_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_leaseput_path_fallbacks_total Lease-intent batches that fell back to a traditional largest-object connection\n";
  s += "# TYPE dfkv_rdma_client_leaseput_path_fallbacks_total counter\n";
  s += "dfkv_rdma_client_leaseput_path_fallbacks_total " +
       std::to_string(leaseput_path_fallbacks_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_v2_probe_attempts_total Required-protocol bootstrap probes\n";
  s += "# TYPE dfkv_rdma_client_v2_probe_attempts_total counter\n";
  s += "dfkv_rdma_client_v2_probe_attempts_total " +
       std::to_string(v2_probe_attempts_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_v2_probe_failures_total Failed required-protocol bootstrap probes\n";
  s += "# TYPE dfkv_rdma_client_v2_probe_failures_total counter\n";
  s += "dfkv_rdma_client_v2_probe_failures_total " +
       std::to_string(v2_probe_failures_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_session_closed_reaps_total Pooled connections dropped before reuse because the server had closed their session channel (DFKV_RDMA_SESSION_FD)\n";
  s += "# TYPE dfkv_rdma_client_session_closed_reaps_total counter\n";
  s += "dfkv_rdma_client_session_closed_reaps_total " +
       std::to_string(session_closed_reaps_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_pipelined_read_conns_total Connections opened with pipelined RDMA READs (DFKV_RDMA_PIPELINED_READS, server advertising its READ depth)\n";
  s += "# TYPE dfkv_rdma_client_pipelined_read_conns_total counter\n";
  s += "dfkv_rdma_client_pipelined_read_conns_total " +
       std::to_string(pipelined_read_conns_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_stale_pool_connections_total Pooled connections that failed on reuse and were discarded without cooling the peer rail\n";
  s += "# TYPE dfkv_rdma_client_stale_pool_connections_total counter\n";
  s += "dfkv_rdma_client_stale_pool_connections_total " +
       std::to_string(
           stale_pool_connections_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_peer_unreachable_total Connection attempts whose TCP connect to the peer's bootstrap address failed; no rail charged, not retried\n";
  s += "# TYPE dfkv_rdma_client_peer_unreachable_total counter\n";
  s += "dfkv_rdma_client_peer_unreachable_total " +
       std::to_string(peer_unreachable_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_peer_unreachable_skips_total Connection attempts not made because a connect to the same address failed within DFKV_RDMA_REMOTE_RAIL_COOLDOWN_MS\n";
  s += "# TYPE dfkv_rdma_client_peer_unreachable_skips_total counter\n";
  s += "dfkv_rdma_client_peer_unreachable_skips_total " +
       std::to_string(
           peer_unreachable_skips_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_peer_busy_total First connection attempts the peer closed before any reply; no rail charged, retried once after a pause\n";
  s += "# TYPE dfkv_rdma_client_peer_busy_total counter\n";
  s += "dfkv_rdma_client_peer_busy_total " +
       std::to_string(peer_busy_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_rail_background_probes_total Recovery probes of a rail out of its cooldown sent by the background thread instead of an application operation\n";
  s += "# TYPE dfkv_rdma_client_rail_background_probes_total counter\n";
  s += "dfkv_rdma_client_rail_background_probes_total " +
       std::to_string(rail_background_probes_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_port_down_events_total Local rails taken out of selection because their port left ACTIVE\n";
  s += "# TYPE dfkv_rdma_client_port_down_events_total counter\n";
  s += "dfkv_rdma_client_port_down_events_total " +
       std::to_string(port_down_events_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_rails_port_down Local rails out of selection now because their port is not ACTIVE\n";
  s += "# TYPE dfkv_rdma_client_rails_port_down gauge\n";
  s += "dfkv_rdma_client_rails_port_down " +
       std::to_string(rails_port_down_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_op_deadline_expired_total Operations whose deadline (DFKV_RDMA_OP_TIMEOUT_MS, or the batch timeout per attempt) ran out while their connection was being set up; nothing was charged\n";
  s += "# TYPE dfkv_rdma_client_op_deadline_expired_total counter\n";
  s += "dfkv_rdma_client_op_deadline_expired_total " +
       std::to_string(op_deadline_expired_.load(std::memory_order_relaxed)) + "\n";
  // TRN-29: device resets.
  s += "# HELP dfkv_rdma_client_async_events_total Device async events read (IBV_EVENT_*)\n";
  s += "# TYPE dfkv_rdma_client_async_events_total counter\n";
  {
    std::lock_guard<std::mutex> lock(device_events_mu_);
    for (const auto& [key, count] : device_events_)
      s += "dfkv_rdma_client_async_events_total{dev=\"" + key.first +
           "\",event=\"" + key.second + "\"} " + std::to_string(count) + "\n";
  }
  s += "# HELP dfkv_rdma_client_device_suspect_total Signals that a rail's device was reset: DEVICE_FATAL, or an endpoint Open that failed with ENODEV or EIO\n";
  s += "# TYPE dfkv_rdma_client_device_suspect_total counter\n";
  s += "# HELP dfkv_rdma_client_device_dead Rails out of selection until rebuilt on their reset device (DFKV_RDMA_DEVICE_REBUILD)\n";
  s += "# TYPE dfkv_rdma_client_device_dead gauge\n";
  s += "# HELP dfkv_rdma_client_device_rebuilds_total Rails rebuilt on a reset device (DFKV_RDMA_DEVICE_REBUILD)\n";
  s += "# TYPE dfkv_rdma_client_device_rebuilds_total counter\n";
  for (size_t rail = 0; rail < devs_.size(); ++rail) {
    const std::string dev = "{dev=\"" + devs_[rail] + "\"";
    s += "dfkv_rdma_client_device_suspect_total" + dev + ",reason=\"fatal\"} " +
         std::to_string(device_suspects_[rail * 2].load(std::memory_order_relaxed)) + "\n";
    s += "dfkv_rdma_client_device_suspect_total" + dev + ",reason=\"open\"} " +
         std::to_string(device_suspects_[rail * 2 + 1].load(std::memory_order_relaxed)) + "\n";
    s += "dfkv_rdma_client_device_dead" + dev + "} " +
         std::to_string(RailStale(rail) ? 1 : 0) + "\n";
    s += "dfkv_rdma_client_device_rebuilds_total" + dev + "} " +
         std::to_string(device_rebuilds_[rail].load(std::memory_order_relaxed)) + "\n";
  }
  s += "# HELP dfkv_rdma_client_dead_release_failures_total Releases (dereg, dealloc, close) that failed on a device proven gone and were dropped instead of aborting (DFKV_RDMA_DEVICE_REBUILD)\n";
  s += "# TYPE dfkv_rdma_client_dead_release_failures_total counter\n";
  s += "dfkv_rdma_client_dead_release_failures_total " +
       std::to_string(rdma::DeadReleaseFailures()) + "\n";
  s += "# HELP dfkv_rdma_client_rail_refusals_total Peer rails charged as failed because the peer closed a bootstrap on them, took one on another rail, and closed a confirming dial on them again (it does not serve that device)\n";
  s += "# TYPE dfkv_rdma_client_rail_refusals_total counter\n";
  s += "dfkv_rdma_client_rail_refusals_total " +
       std::to_string(rail_refusals_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_local_resource_held_total Operations failed at once because every rail's endpoint open failed with ENOMEM within the last few seconds\n";
  s += "# TYPE dfkv_rdma_client_local_resource_held_total counter\n";
  s += "dfkv_rdma_client_local_resource_held_total " +
       std::to_string(local_resource_held_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_local_resource_rail_fallbacks_total Endpoint opens refused for lack of descriptors or memory whose operation was then served on another rail\n";
  s += "# TYPE dfkv_rdma_client_local_resource_rail_fallbacks_total counter\n";
  s += "dfkv_rdma_client_local_resource_rail_fallbacks_total " +
       std::to_string(local_resource_rail_fallbacks_.load(
           std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_local_resource_failures_total Operations failed because this process could not register a caller buffer or create an endpoint for lack of descriptors or memory; no rail or peer charged\n";
  s += "# TYPE dfkv_rdma_client_local_resource_failures_total counter\n";
  s += "dfkv_rdma_client_local_resource_failures_total " +
       std::to_string(
           local_resource_failures_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_stale_pool_retries_total Fresh-connection retries after a pooled QP failure\n";
  s += "# TYPE dfkv_rdma_client_stale_pool_retries_total counter\n";
  s += "dfkv_rdma_client_stale_pool_retries_total " +
       std::to_string(stale_pool_retries_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_cross_rail_retries_total Logical operations retried with the failed local or remote endpoint rail excluded and another compatible healthy rail available\n";
  s += "# TYPE dfkv_rdma_client_cross_rail_retries_total counter\n";
  s += "dfkv_rdma_client_cross_rail_retries_total " +
       std::to_string(cross_rail_retries_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_cross_rail_retry_successes_total Cross-rail retries completed successfully\n";
  s += "# TYPE dfkv_rdma_client_cross_rail_retry_successes_total counter\n";
  s += "dfkv_rdma_client_cross_rail_retry_successes_total " +
       std::to_string(
           cross_rail_retry_successes_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_cross_rail_retry_exhausted_total Cross-rail retries whose second transport attempt failed\n";
  s += "# TYPE dfkv_rdma_client_cross_rail_retry_exhausted_total counter\n";
  s += "dfkv_rdma_client_cross_rail_retry_exhausted_total " +
       std::to_string(
           cross_rail_retry_exhausted_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_completion_timeouts_total Completion windows exhausting their absolute deadline\n";
  s += "# TYPE dfkv_rdma_client_completion_timeouts_total counter\n";
  s += "dfkv_rdma_client_completion_timeouts_total " +
       std::to_string(completion_timeouts_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_pull_prepares_total Pull-read PREPARE requests posted\n";
  s += "# TYPE dfkv_rdma_client_pull_prepares_total counter\n";
  s += "dfkv_rdma_client_pull_prepares_total " +
       std::to_string(pull_prepares_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_pull_reads_total Successful initiator READ operations\n";
  s += "# TYPE dfkv_rdma_client_pull_reads_total counter\n";
  s += "dfkv_rdma_client_pull_reads_total " +
       std::to_string(pull_reads_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_unregistered_dest_registrations_total Registrations made for GET destinations outside registered memory (segments that touch share one)\n";
  s += "# TYPE dfkv_rdma_client_unregistered_dest_registrations_total counter\n";
  s += "dfkv_rdma_client_unregistered_dest_registrations_total " +
       std::to_string(dest_registrations_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_unregistered_dest_memlock_fallbacks_total GETs whose unregistered destination the locked-memory limit refused as a whole, registered round by round instead\n";
  s += "# TYPE dfkv_rdma_client_unregistered_dest_memlock_fallbacks_total counter\n";
  s += "dfkv_rdma_client_unregistered_dest_memlock_fallbacks_total " +
       std::to_string(dest_memlock_fallbacks_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_unregistered_dest_span_fallbacks_total Merged GET destination ranges the device would not register, registered segment by segment instead\n";
  s += "# TYPE dfkv_rdma_client_unregistered_dest_span_fallbacks_total counter\n";
  s += "dfkv_rdma_client_unregistered_dest_span_fallbacks_total " +
       std::to_string(dest_span_fallbacks_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_pull_read_wrs_total READ work requests posted (one may scatter into several destination segments)\n";
  s += "# TYPE dfkv_rdma_client_pull_read_wrs_total counter\n";
  s += "dfkv_rdma_client_pull_read_wrs_total " +
       std::to_string(pull_read_wrs_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_pull_read_bytes_total Bytes completed by initiator READ\n";
  s += "# TYPE dfkv_rdma_client_pull_read_bytes_total counter\n";
  s += "dfkv_rdma_client_pull_read_bytes_total " +
       std::to_string(pull_read_bytes_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_pull_failures_total Pull-read operations that destroyed their connection\n";
  s += "# TYPE dfkv_rdma_client_pull_failures_total counter\n";
  s += "dfkv_rdma_client_pull_failures_total " +
       std::to_string(pull_failures_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_pull_releases_total Acknowledged dynamic pull releases\n";
  s += "# TYPE dfkv_rdma_client_pull_releases_total counter\n";
  s += "dfkv_rdma_client_pull_releases_total " +
       std::to_string(pull_releases_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_pull_release_failures_total Failed dynamic pull release exchanges\n";
  s += "# TYPE dfkv_rdma_client_pull_release_failures_total counter\n";
  s += "dfkv_rdma_client_pull_release_failures_total " +
       std::to_string(pull_release_failures_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_keepalive_attempts_total Idle pooled QP keepalive probes attempted\n";
  s += "# TYPE dfkv_rdma_client_keepalive_attempts_total counter\n";
  s += "dfkv_rdma_client_keepalive_attempts_total " +
       std::to_string(keepalive_attempts_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_keepalive_successes_total Idle pooled QP keepalive probes completed\n";
  s += "# TYPE dfkv_rdma_client_keepalive_successes_total counter\n";
  s += "dfkv_rdma_client_keepalive_successes_total " +
       std::to_string(keepalive_successes_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_keepalive_failures_total Idle pooled QP keepalive probes that retired a connection\n";
  s += "# TYPE dfkv_rdma_client_keepalive_failures_total counter\n";
  s += "dfkv_rdma_client_keepalive_failures_total " +
       std::to_string(keepalive_failures_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_peer_topology_updates_total Published peer topology snapshots\n";
  s += "# TYPE dfkv_rdma_client_peer_topology_updates_total counter\n";
  s += "dfkv_rdma_client_peer_topology_updates_total " +
       std::to_string(
           peer_topology_updates_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_no_compatible_rail_total Operations rejected before admission because no peer-compatible rail exists\n";
  s += "# TYPE dfkv_rdma_client_no_compatible_rail_total counter\n";
  s += "dfkv_rdma_client_no_compatible_rail_total " +
       std::to_string(no_compatible_rail_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_stale_generation_reaps_total Connections retired instead of reuse after a peer publication/incarnation change\n";
  s += "# TYPE dfkv_rdma_client_stale_generation_reaps_total counter\n";
  s += "dfkv_rdma_client_stale_generation_reaps_total " +
       std::to_string(
           stale_publication_reaps_.load(std::memory_order_relaxed)) + "\n";
  s += remote_rail_health_->MetricsText(devs_);
  return s;
}

}  // namespace dfkv
