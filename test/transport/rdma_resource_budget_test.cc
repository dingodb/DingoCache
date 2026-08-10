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

}  // namespace
}  // namespace dfkv::rdma
