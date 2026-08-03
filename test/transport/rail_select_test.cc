#include "transport/rail_select.h"
// ClassifyCompletion is verbs-dependent (ibv_wc_status): only RDMA builds
// compile its coverage below; elsewhere those tests are compiled away.
#ifdef DFKV_WITH_RDMA
#include "transport/rail_classify.h"
#endif

#include <gtest/gtest.h>

using dfkv::rdma::PickRail;
using dfkv::rdma::RailCompletion;
using dfkv::rdma::RailPolicy;
using dfkv::rdma::RailPolicyConfig;

TEST(PickRail, NumaOffRoundRobinsAll) {
  std::vector<int> dn{0, 0, 1, 1};
  for (size_t t = 0; t < 6; ++t)
    EXPECT_EQ(PickRail(dn, /*caller=*/1, /*numa_on=*/false, t), t % 4);
}

TEST(PickRail, CallerUnknownRoundRobinsAll) {
  std::vector<int> dn{0, 1};
  EXPECT_EQ(PickRail(dn, -1, true, 0), 0u);
  EXPECT_EQ(PickRail(dn, -1, true, 1), 1u);
  EXPECT_EQ(PickRail(dn, -1, true, 2), 0u);
}

TEST(PickRail, SingleRailAlwaysZero) {
  std::vector<int> dn{-1};
  EXPECT_EQ(PickRail(dn, 0, true, 0), 0u);
  EXPECT_EQ(PickRail(dn, 0, true, 7), 0u);
}

TEST(PickRail, PicksLocalSubsetNode1) {  // dev 2,3 are on NUMA1
  std::vector<int> dn{0, 0, 1, 1};
  EXPECT_EQ(PickRail(dn, 1, true, 0), 2u);
  EXPECT_EQ(PickRail(dn, 1, true, 1), 3u);
  EXPECT_EQ(PickRail(dn, 1, true, 2), 2u);
}

TEST(PickRail, PicksLocalSubsetNode0) {  // dev 0,1 are on NUMA0
  std::vector<int> dn{0, 0, 1, 1};
  EXPECT_EQ(PickRail(dn, 0, true, 0), 0u);
  EXPECT_EQ(PickRail(dn, 0, true, 1), 1u);
  EXPECT_EQ(PickRail(dn, 0, true, 2), 0u);
}

TEST(PickRail, NoLocalRailFallsBackToAll) {  // caller NUMA1 but both devs NUMA0
  std::vector<int> dn{0, 0};
  EXPECT_EQ(PickRail(dn, 1, true, 0), 0u);
  EXPECT_EQ(PickRail(dn, 1, true, 1), 1u);
  EXPECT_EQ(PickRail(dn, 1, true, 2), 0u);
}

TEST(PickRail, AllUnknownFallsBackToAll) {  // single-NUMA box: all -1
  std::vector<int> dn{-1, -1};
  EXPECT_EQ(PickRail(dn, 0, true, 0), 0u);
  EXPECT_EQ(PickRail(dn, 0, true, 1), 1u);
}

TEST(PickRail, EmptyDeviceListReturnsZero) {
  std::vector<int> dn{};
  EXPECT_EQ(PickRail(dn, 0, true, 5), 0u);
  EXPECT_EQ(PickRail(dn, -1, false, 0), 0u);
}

TEST(RailPolicy, LeastInflightSpreadsConcurrentOperationsDeterministically) {
  RailPolicy policy(3, RailPolicyConfig{4, 3, 1000, 1, 100});
  const auto first = policy.TryAcquire(1, 10);
  const auto second = policy.TryAcquire(1, 10);
  const auto third = policy.TryAcquire(1, 10);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  ASSERT_TRUE(third);
  EXPECT_EQ(first->rail, 0u);
  EXPECT_EQ(second->rail, 1u);
  EXPECT_EQ(third->rail, 2u);
  const auto stats = policy.Snapshot(10);
  ASSERT_EQ(stats.size(), 3u);
  for (const auto& rail : stats) {
    EXPECT_EQ(rail.inflight, 1u);
    EXPECT_EQ(rail.selections, 1u);
  }
}

TEST(RailPolicy, CreditsBoundOutstandingAndExposeExhaustion) {
  RailPolicy policy(1, RailPolicyConfig{2, 3, 1000, 1, 100});
  const auto full = policy.TryAcquire(2, 10);
  ASSERT_TRUE(full);
  EXPECT_EQ(full->credits, 2u);
  EXPECT_FALSE(policy.TryAcquire(1, 10));
  auto stats = policy.Snapshot(10);
  EXPECT_FALSE(policy.Acquire(1, 0));
  ASSERT_EQ(stats.size(), 1u);
  EXPECT_EQ(stats[0].inflight, 2u);
  EXPECT_EQ(stats[0].credits, 2u);
  EXPECT_EQ(stats[0].credits_exhausted, 1u);

  policy.Complete(*full, 20, RailCompletion::kSuccess, 30);
  const auto next = policy.TryAcquire(3, 31);
  ASSERT_TRUE(next);
  EXPECT_EQ(next->credits, 2u);
  EXPECT_EQ(policy.Snapshot(31)[0].inflight, 2u);
}

TEST(RailPolicy, FailureQuarantinesThenSuccessfulProbeRecovers) {
  RailPolicy policy(2, RailPolicyConfig{2, 1, 1000, 1, 100});
  const auto degraded = policy.TryAcquire(1, 100);
  ASSERT_TRUE(degraded);
  ASSERT_EQ(degraded->rail, 0u);
  policy.Complete(*degraded, 50, RailCompletion::kRailFailure, 200);

  const auto healthy = policy.TryAcquire(1, 201);
  ASSERT_TRUE(healthy);
  EXPECT_EQ(healthy->rail, 1u);
  auto during = policy.Snapshot(201);
  EXPECT_EQ(during[0].quarantines, 1u);
  EXPECT_GT(during[0].quarantined_until_us, 201u);

  policy.Complete(*healthy, 25, RailCompletion::kSuccess, 300);
  const auto probe = policy.TryAcquire(1, 1200);
  ASSERT_TRUE(probe);
  EXPECT_EQ(probe->rail, 0u);
  const auto while_probing = policy.TryAcquire(1, 1201);
  ASSERT_TRUE(while_probing);
  EXPECT_EQ(while_probing->rail, 1u);
  const auto probing = policy.Snapshot(1201);
  EXPECT_TRUE(probing[0].quarantined);
  EXPECT_TRUE(probing[0].recovery_probe);
  policy.Complete(*probe, 30, RailCompletion::kSuccess, 1230);
  const auto after = policy.Snapshot(1230);
  EXPECT_EQ(after[0].recoveries, 1u);
  policy.Complete(*while_probing, 25, RailCompletion::kSuccess, 1230);
  EXPECT_EQ(after[0].consecutive_errors, 0u);
  EXPECT_EQ(after[0].quarantined_until_us, 0u);
}

TEST(RailPolicy, ErrorAndLatencyPenaltySteerEqualInflightAway) {
  RailPolicy policy(2, RailPolicyConfig{4, 3, 1000, 10, 10000});
  const auto slow = policy.TryAcquire(1, 10);
  ASSERT_TRUE(slow);
  ASSERT_EQ(slow->rail, 0u);
  policy.Complete(*slow, 500, RailCompletion::kSuccess, 510);

  const auto fast = policy.TryAcquire(1, 511);
  ASSERT_TRUE(fast);
  EXPECT_EQ(fast->rail, 1u);
  policy.Complete(*fast, 10, RailCompletion::kRailFailure, 521);

  const auto penalized = policy.TryAcquire(1, 522);
  ASSERT_TRUE(penalized);
  EXPECT_EQ(penalized->rail, 0u);
}

TEST(RailPolicy, CandidateMaskRestrictsNumaSelection) {
  RailPolicy policy(3);
  const std::vector<uint8_t> local{0, 1, 1};
  const auto first = policy.TryAcquire(1, 10, local);
  const auto second = policy.TryAcquire(1, 10, local);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_EQ(first->rail, 1u);
  EXPECT_EQ(second->rail, 2u);
}

TEST(RailPolicy, LocalMaskDoesNotEscapeWhenLocalCreditsAreBusy) {
  RailPolicy policy(2, RailPolicyConfig{1, 3, 1000, 1, 100});
  const std::vector<uint8_t> local{1, 0};
  const auto held = policy.TryAcquire(1, 10, local);
  ASSERT_TRUE(held);
  EXPECT_FALSE(policy.TryAcquire(1, 11, local));
  EXPECT_EQ(policy.Snapshot(11)[1].inflight, 0u);

  policy.Complete(*held, 20, RailCompletion::kSuccess, 30);
  const auto next = policy.TryAcquire(1, 31, local);
  ASSERT_TRUE(next);
  EXPECT_EQ(next->rail, 0u);
}

TEST(RailPolicy, EndpointFailuresReturnCreditsWithoutPenalizingRail) {
  RailPolicy policy(2, RailPolicyConfig{1, 1, 1000, 1, 100});
  const std::vector<uint8_t> first_rail{1, 0};

  for (uint64_t now = 10; now < 15; ++now) {
    const auto failed_endpoint = policy.TryAcquire(1, now, first_rail);
    ASSERT_TRUE(failed_endpoint);
    ASSERT_EQ(failed_endpoint->rail, 0u);
    policy.Complete(*failed_endpoint, 20, RailCompletion::kEndpointFailure,
                    now + 1);
  }

  const auto stats = policy.Snapshot(20);
  EXPECT_EQ(stats[0].inflight, 0u);
  EXPECT_EQ(stats[0].endpoint_errors, 5u);
  EXPECT_EQ(stats[0].errors, 0u);
  EXPECT_EQ(stats[0].consecutive_errors, 0u);
  EXPECT_EQ(stats[0].quarantines, 0u);
  const auto unrelated_node = policy.TryAcquire(1, 21, first_rail);
  ASSERT_TRUE(unrelated_node);
  EXPECT_EQ(unrelated_node->rail, 0u);
}

TEST(RailPolicy, EndpointFailureDoesNotConsumeRecoveryProbe) {
  RailPolicy policy(2, RailPolicyConfig{1, 1, 1000, 1, 100});
  const auto failed_rail = policy.TryAcquire(1, 10);
  ASSERT_TRUE(failed_rail);
  policy.Complete(*failed_rail, 20, RailCompletion::kRailFailure, 20);

  const auto endpoint_failed_probe = policy.TryAcquire(1, 1020);
  ASSERT_TRUE(endpoint_failed_probe);
  ASSERT_EQ(endpoint_failed_probe->rail, failed_rail->rail);
  policy.Complete(*endpoint_failed_probe, 20,
                  RailCompletion::kEndpointFailure, 1030);

  auto stats = policy.Snapshot(1030);
  EXPECT_EQ(stats[failed_rail->rail].endpoint_errors, 1u);
  EXPECT_EQ(stats[failed_rail->rail].recoveries, 0u);
  EXPECT_TRUE(stats[failed_rail->rail].quarantined);
  EXPECT_FALSE(stats[failed_rail->rail].recovery_probe);
  const auto genuine_probe = policy.TryAcquire(1, 1031);
  ASSERT_TRUE(genuine_probe);
  ASSERT_EQ(genuine_probe->rail, failed_rail->rail);
  policy.Complete(*genuine_probe, 20, RailCompletion::kSuccess, 1051);
  stats = policy.Snapshot(1051);
  EXPECT_EQ(stats[failed_rail->rail].recoveries, 1u);
  EXPECT_EQ(stats[failed_rail->rail].quarantined_until_us, 0u);
  EXPECT_FALSE(stats[failed_rail->rail].quarantined);
}

#ifdef DFKV_WITH_RDMA
using dfkv::rdma::ClassifyCompletion;

TEST(ClassifyCompletion, RemoteEvidenceNeverBlamesTheRail) {
  const ibv_wc_status remote[] = {
      IBV_WC_REM_ACCESS_ERR,   IBV_WC_REM_OP_ERR,
      IBV_WC_REM_INV_REQ_ERR,  IBV_WC_REM_ABORT_ERR,
      IBV_WC_RETRY_EXC_ERR,    IBV_WC_RNR_RETRY_EXC_ERR,
      IBV_WC_RESP_TIMEOUT_ERR, IBV_WC_WR_FLUSH_ERR,
  };
  for (ibv_wc_status status : remote) {
    for (bool had : {false, true}) {
      EXPECT_EQ(ClassifyCompletion(status, had),
                RailCompletion::kEndpointFailure)
          << "wc status " << status;
    }
  }
}

TEST(ClassifyCompletion, LocalEvidenceBlamesTheRail) {
  const ibv_wc_status local[] = {
      IBV_WC_LOC_LEN_ERR,    IBV_WC_LOC_QP_OP_ERR,
      IBV_WC_LOC_EEC_OP_ERR, IBV_WC_LOC_PROT_ERR,
      IBV_WC_LOC_ACCESS_ERR, IBV_WC_LOC_RDD_VIOL_ERR,
      IBV_WC_FATAL_ERR,      IBV_WC_GENERAL_ERR,
  };
  for (ibv_wc_status status : local) {
    for (bool had : {false, true}) {
      EXPECT_EQ(ClassifyCompletion(status, had), RailCompletion::kRailFailure)
          << "wc status " << status;
    }
  }
}

TEST(ClassifyCompletion, DeadlineExpiryIsPeerSilence) {
  // Zero completions: the peer went silent and the NIC surfaced no local
  // error evidence.
  EXPECT_EQ(ClassifyCompletion(IBV_WC_SUCCESS, /*had_completions=*/false),
            RailCompletion::kEndpointFailure);
  // Partial progress then stall: completions prove the rail was healthy when
  // last observed, so the same verdict holds.
  EXPECT_EQ(ClassifyCompletion(IBV_WC_SUCCESS, /*had_completions=*/true),
            RailCompletion::kEndpointFailure);
}

TEST(ClassifyCompletion, UnknownStatusSparesTheRail) {
  // Unlisted/ambiguous evidence must never quarantine a healthy HCA.
  const ibv_wc_status unknown[] = {
      IBV_WC_MW_BIND_ERR, IBV_WC_BAD_RESP_ERR, IBV_WC_REM_INV_RD_REQ_ERR,
      IBV_WC_INV_EECN_ERR, static_cast<ibv_wc_status>(0x7f),
  };
  for (ibv_wc_status status : unknown) {
    for (bool had : {false, true}) {
      EXPECT_EQ(ClassifyCompletion(status, had),
                RailCompletion::kEndpointFailure)
          << "wc status " << status;
    }
  }
}

TEST(ClassifyCompletion, DeadPeerWindowLeavesRailHealthy) {
  // End to end through the policy: dead-peer evidence (retry-chain exhaustion
  // or a silent deadline) repeated past the error threshold must not
  // quarantine the local rail.
  RailPolicy policy(1, RailPolicyConfig{1, 3, 1000, 1, 100});
  for (uint64_t now = 10; now < 15; ++now) {
    const auto lease = policy.TryAcquire(1, now);
    ASSERT_TRUE(lease);
    policy.Complete(*lease, 20,
                    ClassifyCompletion(IBV_WC_RETRY_EXC_ERR,
                                       /*had_completions=*/true),
                    now + 1);
  }
  const auto silent_lease = policy.TryAcquire(1, 20);
  ASSERT_TRUE(silent_lease);
  policy.Complete(*silent_lease, 20,
                  ClassifyCompletion(IBV_WC_SUCCESS,
                                     /*had_completions=*/false),
                  21);
  const auto stats = policy.Snapshot(22);
  EXPECT_EQ(stats[0].endpoint_errors, 6u);
  EXPECT_EQ(stats[0].errors, 0u);
  EXPECT_EQ(stats[0].consecutive_errors, 0u);
  EXPECT_EQ(stats[0].quarantines, 0u);
  EXPECT_TRUE(policy.TryAcquire(1, 23));
}

TEST(ClassifyCompletion, LocalFaultWindowFeedsQuarantine) {
  // Local evidence (the local-fault bucket also used for post/CQ API
  // failures) still counts toward quarantine.
  RailPolicy policy(1, RailPolicyConfig{1, 3, 1000, 1, 100});
  for (uint64_t now = 10; now < 13; ++now) {
    const auto lease = policy.TryAcquire(1, now);
    ASSERT_TRUE(lease);
    policy.Complete(*lease, 20,
                    ClassifyCompletion(IBV_WC_GENERAL_ERR,
                                       /*had_completions=*/true),
                    now + 1);
  }
  const auto stats = policy.Snapshot(14);
  EXPECT_EQ(stats[0].errors, 3u);
  EXPECT_EQ(stats[0].consecutive_errors, 3u);
  EXPECT_EQ(stats[0].quarantines, 1u);
  EXPECT_TRUE(stats[0].quarantined);
  EXPECT_FALSE(policy.TryAcquire(1, 15));
}
#endif  // DFKV_WITH_RDMA
