#include "utils/parallel_for.h"

#include <atomic>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace dfkv {
namespace {

TEST(ParallelExecutor, ExecutesEveryIndexExactlyOnce) {
  constexpr size_t kItems = 257;
  std::vector<std::atomic<int>> visits(kItems);
  for (auto& visit : visits) visit.store(0);

  RunParallel(kItems, 16, [&](size_t index) { visits[index].fetch_add(1); });

  for (const auto& visit : visits) EXPECT_EQ(visit.load(), 1);
}

TEST(ParallelExecutor, ItemExceptionDoesNotPoisonLaterJobs) {
  std::atomic<size_t> completed{0};
  RunParallel(32, 8, [&](size_t index) {
    if (index == 7) throw std::runtime_error("item failure");
    completed.fetch_add(1);
  });
  EXPECT_EQ(completed.load(), 31u);

  completed.store(0);
  RunParallel(19, 4, [&](size_t) { completed.fetch_add(1); });
  EXPECT_EQ(completed.load(), 19u);
}

TEST(ParallelExecutor, CompletedJobsLeaveNoStaleHelperTickets) {
  for (size_t run = 0; run < 100; ++run)
    RunParallel(2, 32, [](size_t) {});

  EXPECT_EQ(ParallelExecutor::Instance().PendingQueueEntriesForTest(), 0u);
}

}  // namespace
}  // namespace dfkv
