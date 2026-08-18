#include "transport/rdma_resource_budget.h"
#include "transport/rdma_protocol.h"

#include <atomic>
#include <chrono>
#include <thread>

#include <gtest/gtest.h>

namespace dfkv::rdma {
namespace {
using namespace std::chrono_literals;

TEST(RdmaResourceBudget, RejectsImpossibleReservationWithoutChangingUsage) {
  ResourceBudget budget({2, 2, 8, 4096});
  EXPECT_FALSE(budget.Acquire({1, 1, 9, 1}, 1ms));
  const auto used = budget.used();
  EXPECT_EQ(used.endpoints, 0u);
  EXPECT_EQ(used.wr_slots, 0u);
  EXPECT_EQ(budget.timeouts(), 0u);
}

TEST(RdmaResourceBudget, WaitsUntilEveryResourceIsReleased) {
  ResourceBudget budget({1, 1, 4, 4096});
  const ResourceRequest request{1, 1, 4, 4096};
  ASSERT_TRUE(budget.Acquire(request, 1ms));

  std::atomic<bool> acquired{false};
  std::thread waiter([&] {
    acquired.store(budget.Acquire(request, 500ms), std::memory_order_release);
  });
  std::this_thread::sleep_for(20ms);
  EXPECT_FALSE(acquired.load(std::memory_order_acquire));
  budget.Release(request);
  waiter.join();
  EXPECT_TRUE(acquired.load(std::memory_order_acquire));
  EXPECT_EQ(budget.used().endpoints, 1u);
  budget.Release(request);
  EXPECT_EQ(budget.used().endpoints, 0u);
}

TEST(RdmaResourceBudget, TimeoutDoesNotOversubscribe) {
  ResourceBudget budget({1, 1, 1, 1});
  const ResourceRequest request{1, 1, 1, 1};
  ASSERT_TRUE(budget.Acquire(request, 1ms));
  EXPECT_FALSE(budget.Acquire(request, 5ms));
  EXPECT_EQ(budget.used().endpoints, 1u);
  EXPECT_EQ(budget.timeouts(), 1u);
  budget.Release(request);
}
TEST(RdmaResourceBudget, TryAcquireNeverWaitsOrCountsTimeout) {
  ResourceBudget budget({1, 1, 1, 1});
  const ResourceRequest request{1, 1, 1, 1};
  ASSERT_TRUE(budget.TryAcquire(request));
  EXPECT_FALSE(budget.TryAcquire(request));
  EXPECT_EQ(budget.timeouts(), 0u);
  EXPECT_EQ(budget.used().endpoints, 1u);
  budget.Release(request);
  EXPECT_TRUE(budget.TryAcquire(request));
  budget.Release(request);
}

TEST(RdmaResourceBudget, RaiseLimitIsMonotonicPerDimension) {
  ResourceBudget budget({4, 4, 16, 1024});
  // A mixed target raises only the dimensions that grow.
  EXPECT_TRUE(budget.RaiseLimit({8, 2, 16, 4096}));
  const auto limit = budget.limit();
  EXPECT_EQ(limit.endpoints, 8u);
  EXPECT_EQ(limit.qps, 4u);           // 2 < 4 keeps the old ceiling
  EXPECT_EQ(limit.wr_slots, 16u);     // equal is not a raise
  EXPECT_EQ(limit.registered_bytes, 4096u);
  EXPECT_EQ(budget.raises(), 1u);
  // A target at-or-below every dimension changes nothing.
  EXPECT_FALSE(budget.RaiseLimit({8, 4, 16, 4096}));
  EXPECT_EQ(budget.raises(), 1u);
}

TEST(RdmaResourceBudget, RaiseLimitWakesBlockedAcquirer) {
  ResourceBudget budget({1, 1, 4, 4096});
  const ResourceRequest request{1, 1, 4, 4096};
  ASSERT_TRUE(budget.Acquire(request, 1ms));

  std::atomic<bool> acquired{false};
  std::thread waiter([&] {
    acquired.store(budget.Acquire(request, 2000ms), std::memory_order_release);
  });
  std::this_thread::sleep_for(20ms);
  EXPECT_FALSE(acquired.load(std::memory_order_acquire));
  // Raising the ceiling (nothing released) must admit the waiter: this is the
  // autoscale path racing an already-starved Acquire.
  EXPECT_TRUE(budget.RaiseLimit({2, 2, 8, 8192}));
  waiter.join();
  EXPECT_TRUE(acquired.load(std::memory_order_acquire));
  EXPECT_EQ(budget.used().endpoints, 2u);
  budget.Release(request);
  budget.Release(request);
}

TEST(RdmaResourceBudget, RaiseLimitAdmitsPreviouslyImpossibleRequest) {
  ResourceBudget budget({2, 2, 8, 4096});
  // Larger than the whole limit: rejected up-front, no timeout counted.
  EXPECT_FALSE(budget.Acquire({1, 1, 9, 1}, 1ms));
  EXPECT_TRUE(budget.RaiseLimit({2, 2, 16, 4096}));
  EXPECT_TRUE(budget.TryAcquire({1, 1, 9, 1}));
  budget.Release({1, 1, 9, 1});
}

TEST(AdaptiveBudgetTarget, SizesFromRingAndRetainedPools) {
  // Eight retained endpoints in each of the data and control pools. A 55-node
  // ring needs 880 base endpoints and 25% teardown/churn headroom.
  const auto target = AdaptiveBudgetTarget(55, 8, 4, 4198400, 256);
  EXPECT_EQ(target.endpoints, 55u * 16u + (55u * 16u) / 4u);  // 1100
  EXPECT_EQ(target.qps, target.endpoints);
  EXPECT_EQ(target.wr_slots, target.endpoints * 4u);
  EXPECT_EQ(target.registered_bytes,
            2 * static_cast<uint64_t>(4198400) * 4u * target.endpoints);
}

TEST(AdaptiveBudgetTarget, SmallRingsKeepTheFloor) {
  const auto target = AdaptiveBudgetTarget(3, 1, 4, 4198400, 256);
  EXPECT_EQ(target.endpoints, 256u);  // 3*(2 pools)*1.25 = 7 -> floor
  EXPECT_EQ(target.wr_slots, 256u * 4u);
}

TEST(AdaptiveBudgetTarget, SeventeenNodeDefaultPoolsExceedFloor) {
  const auto target = AdaptiveBudgetTarget(17, 8, 4, 4198400, 256);
  EXPECT_EQ(target.endpoints, 340u);  // 17*(2 pools)*8*1.25
  EXPECT_EQ(target.qps, 340u);
  EXPECT_EQ(target.wr_slots, 1360u);
}

TEST(AdaptiveBudgetTarget, UnifiedPoolFitsXb01FourteenTp8ClientsIn64GiB) {
  constexpr uint64_t kSegment64GiB = 64ull << 30;
  constexpr uint64_t kSegment128GiB = 128ull << 30;
  constexpr uint64_t kServingPods = 14;
  constexpr uint64_t kTpRanks = 8;
  constexpr uint64_t kPoolLimit = 8;
  constexpr size_t kDepth = 4;
  const uint64_t data_slot = V2SlotSize(4u << 20);
  const uint64_t control_slot = V2SlotSize(kV2ControlCap);
  const uint64_t pull_data_connection_bytes = 2 * kDepth * data_slot;
  const uint64_t pull_control_connection_bytes = 2 * kDepth * control_slot;
  ASSERT_EQ(pull_data_connection_bytes, 33587200u);
  ASSERT_EQ(pull_control_connection_bytes, 327680u);
  EXPECT_EQ(kSegment64GiB / pull_data_connection_bytes, 2046u);
  EXPECT_EQ(kSegment128GiB / pull_data_connection_bytes, 4092u);

  const uint64_t client_ranks = kServingPods * kTpRanks;
  const uint64_t data_connections = client_ranks * kPoolLimit;
  const uint64_t control_connections = client_ranks * kPoolLimit;
  const uint64_t base_bytes =
      data_connections * pull_data_connection_bytes +
      control_connections * pull_control_connection_bytes;
  const uint64_t with_churn_headroom = base_bytes * 5 / 4;
  EXPECT_EQ(data_connections, 896u);
  EXPECT_EQ(control_connections, 896u);
  EXPECT_EQ(base_bytes, 30387732480u);
  EXPECT_EQ(with_churn_headroom, 37984665600u);
  EXPECT_LT(with_churn_headroom, kSegment64GiB);
}

TEST(AdaptiveBudgetTarget, CapsAtTheEnvUpperBound) {
  const auto huge = AdaptiveBudgetTarget(100000, 8, 4, 4198400, 256);
  EXPECT_EQ(huge.endpoints, 65536u);
  // Zero-pool and zero-depth inputs stay defined (treated as 1).
  const auto degenerate = AdaptiveBudgetTarget(10, 0, 0, 0, 256);
  EXPECT_EQ(degenerate.endpoints, 256u);
  EXPECT_EQ(degenerate.wr_slots, 256u);
  EXPECT_EQ(degenerate.registered_bytes,
            std::numeric_limits<uint64_t>::max());
}

}  // namespace
}  // namespace dfkv::rdma
