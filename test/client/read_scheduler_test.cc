#include "client/read_scheduler.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace dfkv {
namespace {

TEST(ReadScheduler, ExecutesEveryShardExactlyOnce) {
  constexpr size_t kShards = 257;
  std::vector<std::atomic<int>> visits(kShards);
  for (auto& visit : visits) visit.store(0);

  RunReadScheduled(kShards,
                   [&](size_t index) { visits[index].fetch_add(1); });

  for (const auto& visit : visits) EXPECT_EQ(visit.load(), 1);
}

TEST(ReadScheduler, BoundsProcessWideActiveShards) {
  constexpr size_t kCallers = 16;
  constexpr size_t kShards = 16;
  std::atomic<size_t> active{0};
  std::atomic<size_t> peak{0};
  std::vector<std::thread> callers;
  callers.reserve(kCallers);

  for (size_t caller = 0; caller < kCallers; ++caller) {
    callers.emplace_back([&] {
      RunReadScheduled(kShards, [&](size_t) {
        const size_t now = active.fetch_add(1) + 1;
        size_t observed = peak.load();
        while (observed < now && !peak.compare_exchange_weak(observed, now)) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        active.fetch_sub(1);
      });
    });
  }
  for (auto& caller : callers) caller.join();

  EXPECT_LE(peak.load(), ReadScheduler::WorkerCount());
  EXPECT_GT(peak.load(), 1u);
}

TEST(ReadScheduler, ExceptionDoesNotPoisonLaterBatches) {
  std::atomic<size_t> completed{0};
  RunReadScheduled(32, [&](size_t index) {
    if (index == 7) throw 7;
    completed.fetch_add(1);
  });
  EXPECT_EQ(completed.load(), 31u);

  completed.store(0);
  RunReadScheduled(19, [&](size_t) { completed.fetch_add(1); });
  EXPECT_EQ(completed.load(), 19u);
}

TEST(ReadScheduler, RendersBoundedResourceMetrics) {
  RunReadScheduled(3, [](size_t) {});
  const std::string metrics = ReadScheduler::Instance().MetricsText();
  EXPECT_NE(metrics.find("dfkv_read_scheduler_pending_batches 0"),
            std::string::npos);
  EXPECT_NE(metrics.find("dfkv_read_scheduler_pending_shards 0"),
            std::string::npos);
  EXPECT_NE(metrics.find("dfkv_read_scheduler_active_shards 0"),
            std::string::npos);
  EXPECT_NE(metrics.find("dfkv_read_scheduler_fairness_yields_total"),
            std::string::npos);
}

}  // namespace
}  // namespace dfkv
