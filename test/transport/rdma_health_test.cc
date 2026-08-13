#include "transport/rdma_health.h"
#include "transport/remote_rail_health.h"
#include "transport/rail_select.h"

#include <gtest/gtest.h>

#include <deque>
#include <vector>

using namespace dfkv;  // NOLINT

namespace {

IbDeviceHealth Device(const char* name, uint8_t port, uint8_t phys,
                      bool query_ok = true) {
  return IbDeviceHealth{name, port, phys, query_ok};
}

}  // namespace

TEST(RdmaHealth, AnyHealthyDeviceKeepsNodeEligible) {
  std::deque<std::vector<IbDeviceHealth>> samples{
      {Device("ib0", 4, 5), Device("ib1", 4, 5)},
      {Device("ib0", 4, 5), Device("ib1", 2, 2)}};
  rdma::RdmaHealthMonitor monitor([&] {
    auto sample = samples.front();
    samples.pop_front();
    return sample;
  });
  EXPECT_TRUE(monitor.Sample().ring_eligible);
  const MemberHealth partially_degraded = monitor.Sample();
  EXPECT_TRUE(partially_degraded.ring_eligible);
  ASSERT_EQ(partially_degraded.ib_devices.size(), 2u);
  EXPECT_TRUE(partially_degraded.ib_devices[0].healthy());
  EXPECT_FALSE(partially_degraded.ib_devices[1].healthy());
  const std::string metrics = monitor.MetricsText();
  EXPECT_NE(metrics.find("dfkv_server_rdma_rails_configured 2"),
            std::string::npos);
  EXPECT_NE(metrics.find("dfkv_server_rdma_rails_active 1"),
            std::string::npos);
}

TEST(RdmaHealth, ZeroActiveDevicesRemovesNodeImmediately) {
  std::deque<std::vector<IbDeviceHealth>> samples{
      {Device("ib0", 4, 5), Device("ib1", 2, 2)},
      {Device("ib0", 1, 2), Device("ib1", 2, 2)}};
  rdma::RdmaHealthMonitor monitor([&] {
    auto sample = samples.front();
    samples.pop_front();
    return sample;
  });
  EXPECT_TRUE(monitor.Sample().ring_eligible);
  EXPECT_FALSE(monitor.Sample().ring_eligible);
}

TEST(RdmaHealth, RecoveryRequiresConsecutiveHealthySamples) {
  std::deque<std::vector<IbDeviceHealth>> samples{
      {Device("ib0", 1, 2)}, {Device("ib0", 4, 5)},
      {Device("ib0", 1, 2)}, {Device("ib0", 4, 5)},
      {Device("ib0", 4, 5)}, {Device("ib0", 4, 5)}};
  rdma::RdmaHealthMonitor monitor([&] {
    auto sample = samples.front();
    samples.pop_front();
    return sample;
  }, 3);
  EXPECT_FALSE(monitor.Sample().ring_eligible);
  EXPECT_FALSE(monitor.Sample().ring_eligible);
  EXPECT_FALSE(monitor.Sample().ring_eligible);
  EXPECT_FALSE(monitor.Sample().ring_eligible);
  EXPECT_FALSE(monitor.Sample().ring_eligible);
  EXPECT_TRUE(monitor.Sample().ring_eligible);
}

TEST(RdmaHealth, SingleRailUsesTheSameRemovalAndRecoveryLifecycle) {
  std::deque<std::vector<IbDeviceHealth>> samples{
      {Device("ib0", 4, 5)}, {Device("ib0", 1, 2)},
      {Device("ib0", 4, 5)}, {Device("ib0", 4, 5)}};
  rdma::RdmaHealthMonitor monitor([&] {
    auto sample = samples.front();
    samples.pop_front();
    return sample;
  }, 2);
  EXPECT_TRUE(monitor.Sample().ring_eligible);
  EXPECT_FALSE(monitor.Sample().ring_eligible);
  EXPECT_FALSE(monitor.Sample().ring_eligible);
  EXPECT_TRUE(monitor.Sample().ring_eligible);
}

TEST(RdmaHealth, QueryFailureFailsClosed) {
  rdma::RdmaHealthMonitor monitor(
      [] { return std::vector<IbDeviceHealth>{Device("ib0", 0, 0, false)}; });
  const MemberHealth health = monitor.Sample();
  EXPECT_FALSE(health.ring_eligible);
  EXPECT_NE(monitor.MetricsText().find("dfkv_server_ring_eligible 0"),
            std::string::npos);
  EXPECT_NE(monitor.MetricsText().find("device=\"ib0\"} 0"),
            std::string::npos);
}

TEST(RemoteRailHealth, EndpointFailureIsIsolatedByPeerAndSiblingRail) {
  RemoteRailHealth health(
      RemoteRailHealthConfig{/*base_cooldown_us=*/100,
                             /*max_cooldown_us=*/800});

  auto failed = health.TryAcquire("peer-a", 0, 10);
  ASSERT_TRUE(failed);
  ASSERT_FALSE(failed->recovery_probe);
  const auto cooled =
      health.Complete("peer-a", 0, failed->generation,
                      RemoteRailOutcome::kEndpointFailure, 20);
  EXPECT_EQ(cooled.transition, RemoteRailTransition::kCooled);
  EXPECT_EQ(cooled.cooldown_until_us, 120u);

  EXPECT_EQ(health.AllowedMask("peer-a", {1, 1}, 20),
            (std::vector<uint8_t>{0, 1}));
  EXPECT_EQ(health.AllowedMask("peer-b", {1, 1}, 20),
            (std::vector<uint8_t>{1, 1}));
  EXPECT_FALSE(health.TryAcquire("peer-a", 0, 20));

  auto sibling = health.TryAcquire("peer-a", 1, 20);
  ASSERT_TRUE(sibling);
  EXPECT_FALSE(sibling->recovery_probe);
  auto other_peer = health.TryAcquire("peer-b", 0, 20);
  ASSERT_TRUE(other_peer);
  EXPECT_FALSE(other_peer->recovery_probe);
  health.Complete("peer-a", 1, sibling->generation,
                  RemoteRailOutcome::kSuccess, 21);
  health.Complete("peer-b", 0, other_peer->generation,
                  RemoteRailOutcome::kSuccess, 21);
}

TEST(RemoteRailHealth, CooldownAdmitsExactlyOneSuccessfulRecoveryProbe) {
  RemoteRailHealth health(
      RemoteRailHealthConfig{/*base_cooldown_us=*/100,
                             /*max_cooldown_us=*/800});
  auto failed = health.TryAcquire("peer", 0, 1);
  ASSERT_TRUE(failed);
  health.Complete("peer", 0, failed->generation,
                  RemoteRailOutcome::kEndpointFailure, 10);

  EXPECT_FALSE(health.TryAcquire("peer", 0, 109));
  auto probe = health.TryAcquire("peer", 0, 110);
  ASSERT_TRUE(probe);
  EXPECT_TRUE(probe->recovery_probe);
  EXPECT_FALSE(health.TryAcquire("peer", 0, 110))
      << "an active recovery probe must fence every competing admission";

  const auto recovered =
      health.Complete("peer", 0, probe->generation,
                      RemoteRailOutcome::kSuccess, 111);
  EXPECT_EQ(recovered.transition, RemoteRailTransition::kRecovered);
  EXPECT_EQ(recovered.cooldown_until_us, 0u);
  EXPECT_EQ(health.AllowedMask("peer", {1}, 111),
            (std::vector<uint8_t>{1}));
  auto ordinary = health.TryAcquire("peer", 0, 111);
  ASSERT_TRUE(ordinary);
  EXPECT_FALSE(ordinary->recovery_probe);
  health.Complete("peer", 0, ordinary->generation,
                  RemoteRailOutcome::kSuccess, 112);
}

TEST(RemoteRailHealth, LateOrdinarySuccessCannotBypassRecoveryProbe) {
  RemoteRailHealth health(
      RemoteRailHealthConfig{/*base_cooldown_us=*/100,
                             /*max_cooldown_us=*/800});
  auto failed = health.TryAcquire("peer", 0, 1);
  auto already_inflight = health.TryAcquire("peer", 0, 2);
  ASSERT_TRUE(failed);
  ASSERT_TRUE(already_inflight);
  ASSERT_FALSE(failed->recovery_probe);
  ASSERT_FALSE(already_inflight->recovery_probe);

  health.Complete("peer", 0, failed->generation,
                  RemoteRailOutcome::kEndpointFailure, 10);
  const auto late_success =
      health.Complete("peer", 0, already_inflight->generation,
                      RemoteRailOutcome::kSuccess, 11);
  EXPECT_EQ(late_success.transition, RemoteRailTransition::kNone);
  EXPECT_EQ(health.AllowedMask("peer", {1}, 11),
            (std::vector<uint8_t>{0}));
  EXPECT_FALSE(health.TryAcquire("peer", 0, 109));
  auto probe = health.TryAcquire("peer", 0, 110);
  ASSERT_TRUE(probe);
  EXPECT_TRUE(probe->recovery_probe);
  health.Complete("peer", 0, probe->generation,
                  RemoteRailOutcome::kSuccess, 111);
}

TEST(RemoteRailHealth, LateOrdinaryFailureCannotRecoolRecoveredRail) {
  RemoteRailHealth health(
      RemoteRailHealthConfig{/*base_cooldown_us=*/100,
                             /*max_cooldown_us=*/800});
  auto late_ordinary = health.TryAcquire("peer", 0, 1);
  auto cooling_failure = health.TryAcquire("peer", 0, 2);
  ASSERT_TRUE(late_ordinary);
  ASSERT_TRUE(cooling_failure);
  ASSERT_FALSE(late_ordinary->recovery_probe);
  ASSERT_EQ(late_ordinary->health_epoch, cooling_failure->health_epoch);

  health.Complete("peer", 0, cooling_failure->generation,
                  RemoteRailOutcome::kEndpointFailure, 10);
  auto probe = health.TryAcquire("peer", 0, 110);
  ASSERT_TRUE(probe);
  ASSERT_TRUE(probe->recovery_probe);
  ASSERT_EQ(probe->health_epoch, late_ordinary->health_epoch);
  const auto recovered =
      health.Complete("peer", 0, probe->generation,
                      RemoteRailOutcome::kSuccess, 111);
  ASSERT_EQ(recovered.transition, RemoteRailTransition::kRecovered);

  const auto late_failure =
      health.Complete("peer", 0, late_ordinary->generation,
                      RemoteRailOutcome::kEndpointFailure, 112);
  EXPECT_EQ(late_failure.transition, RemoteRailTransition::kNone);
  EXPECT_EQ(late_failure.cooldown_until_us, 0u);
  EXPECT_EQ(health.AllowedMask("peer", {1}, 112),
            (std::vector<uint8_t>{1}));
  EXPECT_NE(
      health.MetricsText({"ib0"}).find(
          "dfkv_rdma_client_remote_rail_failures_total 1\n"),
      std::string::npos)
      << "the superseded failure must not be accounted as fresh evidence";
  auto fresh = health.TryAcquire("peer", 0, 112);
  ASSERT_TRUE(fresh);
  EXPECT_FALSE(fresh->recovery_probe);
  EXPECT_EQ(fresh->health_epoch, probe->health_epoch + 1);
  health.Complete("peer", 0, fresh->generation,
                  RemoteRailOutcome::kSuccess, 113);
}

TEST(RemoteRailHealth, FailedProbeRecoolsWithCappedFakeClockBackoff) {
  RemoteRailHealth health(
      RemoteRailHealthConfig{/*base_cooldown_us=*/100,
                             /*max_cooldown_us=*/250});
  auto initial = health.TryAcquire("peer", 0, 1);
  ASSERT_TRUE(initial);
  auto completion =
      health.Complete("peer", 0, initial->generation,
                      RemoteRailOutcome::kEndpointFailure, 20);
  EXPECT_EQ(completion.cooldown_until_us, 120u);
  EXPECT_EQ(health.AllowedMask("peer", {1}, 119),
            (std::vector<uint8_t>{0}));

  auto first_probe = health.TryAcquire("peer", 0, 120);
  ASSERT_TRUE(first_probe);
  ASSERT_TRUE(first_probe->recovery_probe);
  completion = health.Complete("peer", 0, first_probe->generation,
                               RemoteRailOutcome::kEndpointFailure, 130);
  EXPECT_EQ(completion.transition, RemoteRailTransition::kCooled);
  EXPECT_EQ(completion.cooldown_until_us, 330u);
  EXPECT_FALSE(health.TryAcquire("peer", 0, 329));

  auto second_probe = health.TryAcquire("peer", 0, 330);
  ASSERT_TRUE(second_probe);
  ASSERT_TRUE(second_probe->recovery_probe);
  completion = health.Complete("peer", 0, second_probe->generation,
                               RemoteRailOutcome::kEndpointFailure, 340);
  EXPECT_EQ(completion.cooldown_until_us, 590u)
      << "the third cooldown must be capped at 250us";
  EXPECT_FALSE(health.TryAcquire("peer", 0, 589));
  auto capped_probe = health.TryAcquire("peer", 0, 590);
  ASSERT_TRUE(capped_probe);
  EXPECT_TRUE(capped_probe->recovery_probe);
  health.Complete("peer", 0, capped_probe->generation,
                  RemoteRailOutcome::kAbandon, 590);
}

TEST(RemoteRailHealth, ConfiguredCooldownIsHardCappedAtThirtySeconds) {
  RemoteRailHealth health(
      RemoteRailHealthConfig{/*base_cooldown_us=*/60'000'000,
                             /*max_cooldown_us=*/60'000'000});
  auto lease = health.TryAcquire("peer", 0, 1);
  ASSERT_TRUE(lease);
  const auto completion =
      health.Complete("peer", 0, lease->generation,
                      RemoteRailOutcome::kEndpointFailure, 10);
  EXPECT_EQ(completion.cooldown_until_us, 30'000'010u);
  EXPECT_FALSE(health.TryAcquire("peer", 0, 30'000'009));
  auto probe = health.TryAcquire("peer", 0, 30'000'010);
  ASSERT_TRUE(probe);
  EXPECT_TRUE(probe->recovery_probe);
  health.Complete("peer", 0, probe->generation,
                  RemoteRailOutcome::kAbandon, 30'000'010);
}

TEST(RemoteRailHealth, AbandonedProbeReleasesTheProbeSlot) {
  RemoteRailHealth health(
      RemoteRailHealthConfig{/*base_cooldown_us=*/100,
                             /*max_cooldown_us=*/800});
  auto failed = health.TryAcquire("peer", 0, 1);
  ASSERT_TRUE(failed);
  health.Complete("peer", 0, failed->generation,
                  RemoteRailOutcome::kEndpointFailure, 10);

  auto abandoned = health.TryAcquire("peer", 0, 110);
  ASSERT_TRUE(abandoned);
  ASSERT_TRUE(abandoned->recovery_probe);
  EXPECT_FALSE(health.TryAcquire("peer", 0, 110));
  const auto completion =
      health.Complete("peer", 0, abandoned->generation,
                      RemoteRailOutcome::kAbandon, 111);
  EXPECT_EQ(completion.transition, RemoteRailTransition::kNone);

  auto replacement = health.TryAcquire("peer", 0, 111);
  ASSERT_TRUE(replacement);
  EXPECT_TRUE(replacement->recovery_probe);
  EXPECT_NE(replacement->generation, abandoned->generation);
  health.Complete("peer", 0, replacement->generation,
                  RemoteRailOutcome::kSuccess, 112);
}

TEST(RemoteRailHealth, NonEndpointAbandonsAreHealthNeutral) {
  RemoteRailHealth health(
      RemoteRailHealthConfig{/*base_cooldown_us=*/100,
                             /*max_cooldown_us=*/800});

  // Local-HCA, resource-admission, and control-path failures are deliberately
  // represented by abandoning their remote leases: none is peer evidence.
  for (size_t rail = 0; rail < 3; ++rail) {
    auto lease = health.TryAcquire("peer", rail, 10);
    ASSERT_TRUE(lease) << rail;
    health.Complete("peer", rail, lease->generation,
                    RemoteRailOutcome::kAbandon, 20 + rail);
  }
  EXPECT_EQ(health.AllowedMask("peer", {1, 1, 1}, 23),
            (std::vector<uint8_t>{1, 1, 1}));
  for (size_t rail = 0; rail < 3; ++rail) {
    auto lease = health.TryAcquire("peer", rail, 23);
    ASSERT_TRUE(lease) << rail;
    EXPECT_FALSE(lease->recovery_probe) << rail;
    health.Complete("peer", rail, lease->generation,
                    RemoteRailOutcome::kSuccess, 24);
  }
}

TEST(RemoteRailHealth, ReconcileRemovalFencesStaleAcquireAndCompletion) {
  RemoteRailHealth health(
      RemoteRailHealthConfig{/*base_cooldown_us=*/100,
                             /*max_cooldown_us=*/800});

  // Constructor-order compatibility permits use before the first topology.
  auto outstanding = health.TryAcquire("removed", 0, 1);
  ASSERT_TRUE(outstanding);
  ASSERT_FALSE(outstanding->recovery_probe);

  health.Reconcile({"survivor"});
  EXPECT_FALSE(health.TryAcquire("removed", 0, 2));
  const auto late =
      health.Complete("removed", 0, outstanding->generation,
                      RemoteRailOutcome::kEndpointFailure, 3);
  EXPECT_EQ(late.transition, RemoteRailTransition::kNone);
  EXPECT_EQ(late.cooldown_until_us, 0u);
  EXPECT_FALSE(health.TryAcquire("removed", 0, 4))
      << "a stale completion must not recreate or readmit a removed identity";

  health.Reconcile({"survivor", "removed"});
  auto republished = health.TryAcquire("removed", 0, 5);
  ASSERT_TRUE(republished);
  EXPECT_FALSE(republished->recovery_probe);
  EXPECT_NE(republished->generation, outstanding->generation);
  health.Complete("removed", 0, republished->generation,
                  RemoteRailOutcome::kSuccess, 6);
}

TEST(RemoteRailHealth, StablePeerTopologyFlapsRetainCooldown) {
  RemoteRailHealth health(
      RemoteRailHealthConfig{/*base_cooldown_us=*/100,
                             /*max_cooldown_us=*/800});
  auto failed = health.TryAcquire("stable-peer-id", 0, 1);
  ASSERT_TRUE(failed);
  health.Complete("stable-peer-id", 0, failed->generation,
                  RemoteRailOutcome::kEndpointFailure, 10);

  // Address/topology generations may flap, but MDS continues publishing the
  // same stable identity. Reconcile must therefore retain its endpoint memory.
  health.Reconcile({"stable-peer-id"});
  health.Reconcile({"stable-peer-id", "another-peer"});
  health.Reconcile({"stable-peer-id"});
  EXPECT_EQ(health.AllowedMask("stable-peer-id", {1}, 109),
            (std::vector<uint8_t>{0}));
  EXPECT_FALSE(health.TryAcquire("stable-peer-id", 0, 109));
  auto probe = health.TryAcquire("stable-peer-id", 0, 110);
  ASSERT_TRUE(probe);
  EXPECT_TRUE(probe->recovery_probe);
  health.Complete("stable-peer-id", 0, probe->generation,
                  RemoteRailOutcome::kSuccess, 111);
}

TEST(RemoteRailHealth, EndpointFailureExcludesAttemptedRailForAlternate) {
  RemoteRailHealth remote(
      RemoteRailHealthConfig{/*base_cooldown_us=*/100,
                             /*max_cooldown_us=*/800});
  rdma::RailPolicy local(
      2, rdma::RailPolicyConfig{/*credits_per_rail=*/2,
                                /*error_threshold=*/1,
                                /*quarantine_us=*/1000,
                                /*latency_weight=*/1,
                                /*error_penalty_us=*/100});

  auto first_remote = remote.TryAcquire("peer", 0, 1);
  ASSERT_TRUE(first_remote);
  auto first_local = rdma::AcquireWithFallback(
      local, 1, 1, remote.AllowedMask("peer", {1, 1}, 1), {}, {});
  ASSERT_TRUE(first_local);
  ASSERT_EQ(first_local->rail, 0u);
  local.Complete(*first_local, 1, rdma::RailCompletion::kEndpointFailure, 2);
  remote.Complete("peer", 0, first_remote->generation,
                  RemoteRailOutcome::kEndpointFailure, 2);

  const auto retry_mask = remote.AllowedMask("peer", {1, 1}, 2);
  EXPECT_EQ(retry_mask, (std::vector<uint8_t>{0, 1}));
  auto retry = rdma::AcquireWithFallback(local, 1, 2, retry_mask, {}, {});
  ASSERT_TRUE(retry);
  EXPECT_EQ(retry->rail, 1u);
  local.Complete(*retry, 1, rdma::RailCompletion::kSuccess, 3);

  const auto local_stats = local.Snapshot(3);
  ASSERT_EQ(local_stats.size(), 2u);
  EXPECT_EQ(local_stats[0].endpoint_errors, 1u);
  EXPECT_EQ(local_stats[0].errors, 0u);
  EXPECT_FALSE(local_stats[0].quarantined)
      << "remote endpoint cooling must not change RailPolicy semantics";
}

TEST(RemoteRailHealth, SoleCooledRailFailsAdmissionWithoutFallback) {
  RemoteRailHealth health(
      RemoteRailHealthConfig{/*base_cooldown_us=*/100,
                             /*max_cooldown_us=*/800});
  auto lease = health.TryAcquire("peer", 0, 1);
  ASSERT_TRUE(lease);
  health.Complete("peer", 0, lease->generation,
                  RemoteRailOutcome::kEndpointFailure, 10);

  const auto allowed = health.AllowedMask("peer", {1}, 10);
  EXPECT_EQ(allowed, (std::vector<uint8_t>{0}));
  EXPECT_FALSE(health.TryAcquire("peer", 0, 10));
  rdma::RailPolicy local(1);
  EXPECT_FALSE(
      rdma::AcquireWithFallback(local, 1, 10, allowed, {}, {}))
      << "an empty remote-health mask is an explicit admission failure, not "
         "permission to bypass filtering";
}

TEST(RemoteRailHealth, CompletionIsAccountedOnceWithFixedCardinalityMetrics) {
  RemoteRailHealth health(
      RemoteRailHealthConfig{/*base_cooldown_us=*/100,
                             /*max_cooldown_us=*/800});
  auto lease = health.TryAcquire("secret-peer-id", 1, 1);
  ASSERT_TRUE(lease);
  const auto first =
      health.Complete("secret-peer-id", 1, lease->generation,
                      RemoteRailOutcome::kEndpointFailure, 10);
  ASSERT_EQ(first.transition, RemoteRailTransition::kCooled);
  const auto duplicate =
      health.Complete("secret-peer-id", 1, lease->generation,
                      RemoteRailOutcome::kEndpointFailure, 11);
  EXPECT_EQ(duplicate.transition, RemoteRailTransition::kNone);
  EXPECT_EQ(duplicate.cooldown_until_us, 0u);
  EXPECT_FALSE(health.TryAcquire("secret-peer-id", 1, 11));

  const std::string metrics =
      health.MetricsText({"mlx5_0", "mlx5_1", "unused\"rail"});
  EXPECT_NE(metrics.find("dfkv_rdma_client_remote_rail_failures_total 1\n"),
            std::string::npos);
  EXPECT_NE(metrics.find("dfkv_rdma_client_remote_rail_cooldowns_total 1\n"),
            std::string::npos);
  EXPECT_NE(
      metrics.find(
          "dfkv_rdma_client_remote_rail_admissions_denied_total 1\n"),
      std::string::npos);
  EXPECT_NE(
      metrics.find(
          "dfkv_rdma_client_remote_rail_failures_by_rail_total{dev=\"mlx5_0\"} 0\n"),
      std::string::npos);
  EXPECT_NE(
      metrics.find(
          "dfkv_rdma_client_remote_rail_failures_by_rail_total{dev=\"mlx5_1\"} 1\n"),
      std::string::npos);
  EXPECT_NE(
      metrics.find(
          "dfkv_rdma_client_remote_rail_failures_by_rail_total{dev=\"unused\\\"rail\"} 0\n"),
      std::string::npos)
      << "every configured rail must emit a bounded zero-valued series";
  EXPECT_EQ(metrics.find("secret-peer-id"), std::string::npos);
  EXPECT_EQ(metrics.find("peer_id="), std::string::npos);
}
