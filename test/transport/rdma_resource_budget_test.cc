#include "transport/rdma_resource_budget.h"

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

TEST(AdaptiveBudgetTarget, SizesFromRingAndRails) {
  // 55 nodes x 8 rails (the 0812-004 incident shape): 2 data-family lanes per
  // rail + control, +25% headroom => comfortably above the observed ~500
  // endpoint demand and the manually verified 1024 fix.
  const auto target = AdaptiveBudgetTarget(55, 8, 4, 4198400, 256);
  EXPECT_EQ(target.endpoints, 55u * 17u + (55u * 17u) / 4u);  // 1168
  EXPECT_EQ(target.qps, target.endpoints);
  EXPECT_EQ(target.wr_slots, target.endpoints * 4u);
  EXPECT_EQ(target.registered_bytes,
            static_cast<uint64_t>(4198400) * 4u * target.endpoints);
}

TEST(AdaptiveBudgetTarget, SmallRingsKeepTheFloor) {
  const auto target = AdaptiveBudgetTarget(3, 1, 4, 4198400, 256);
  EXPECT_EQ(target.endpoints, 256u);  // 3*(2*1+1)*1.25 = 11 -> floor
  EXPECT_EQ(target.wr_slots, 256u * 4u);
}

TEST(AdaptiveBudgetTarget, CapsAtTheEnvUpperBound) {
  const auto huge = AdaptiveBudgetTarget(100000, 8, 4, 4198400, 256);
  EXPECT_EQ(huge.endpoints, 65536u);
  // Zero-rail and zero-depth inputs stay defined (treated as 1).
  const auto degenerate = AdaptiveBudgetTarget(10, 0, 0, 0, 256);
  EXPECT_EQ(degenerate.endpoints, 256u);
  EXPECT_EQ(degenerate.wr_slots, 256u);
  EXPECT_EQ(degenerate.registered_bytes,
            std::numeric_limits<uint64_t>::max());
}

}  // namespace
}  // namespace dfkv::rdma
