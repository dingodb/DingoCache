#include "client/read_scheduler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
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

namespace {
// Shards that stay inside their node until released, counted per node.
struct HeldShards {
  std::mutex mu;
  std::condition_variable cv;
  bool release = false;
  int inside = 0, most = 0;
  void Enter() {
    std::unique_lock<std::mutex> lock(mu);
    most = std::max(most, ++inside);
    cv.notify_all();
    cv.wait(lock, [&] { return release; });
    --inside;
  }
  bool WaitInside(int count, std::chrono::milliseconds wait) {
    std::unique_lock<std::mutex> lock(mu);
    return cv.wait_for(lock, wait, [&] { return inside >= count; });
  }
  void Release() {
    std::lock_guard<std::mutex> lock(mu);
    release = true;
    cv.notify_all();
  }
};
const std::string kNodeX = "10.0.0.1:1", kNodeY = "10.0.0.2:1";
const std::string& NodeX(size_t) { return kNodeX; }
const std::string& NodeY(size_t) { return kNodeY; }

uint64_t NodeLimitWaits() {
  const std::string metrics = dfkv::ReadScheduler::Instance().MetricsText();
  const std::string name = "\ndfkv_read_scheduler_node_limit_waits_total ";
  const size_t at = metrics.find(name);
  if (at == std::string::npos) return 0;
  return std::stoull(metrics.substr(at + name.size()));
}
}  // namespace

// The scheduler is one per process and keeps no record of earlier batches
// that decides what a later one may do, so these tests pass in any order and
// in one process (--gtest_shuffle).
TEST(ReadScheduler, OneNodeLeavesAWorkerForTheOthers) {
  const size_t workers = dfkv::ReadScheduler::WorkerCount();
  ASSERT_GE(workers, 2u);
  const uint64_t waits_before = NodeLimitWaits();

  // More shards of a slow node than there are workers, with nothing else
  // queued or running: one worker is still kept free.
  HeldShards slow;
  std::thread batch([&] {
    dfkv::RunReadScheduled(workers + 1, [&](size_t) { slow.Enter(); }, NodeX);
  });
  if (!slow.WaitInside(static_cast<int>(workers) - 1,
                       std::chrono::seconds(10))) {
    slow.Release();
    batch.join();
    FAIL() << "the slow node's shards never filled their share";
  }
  // A read of another node that arrives now is served while they are all
  // still inside.
  std::atomic<int> other{0};
  dfkv::RunReadScheduled(3, [&](size_t) { other.fetch_add(1); }, NodeY);
  EXPECT_EQ(other.load(), 3);
  {
    std::lock_guard<std::mutex> lock(slow.mu);
    EXPECT_EQ(slow.inside, static_cast<int>(workers) - 1);
  }
  // The free worker, back from the other node's read, finds only shards of
  // the slow node left and counts that. Polled: on a loaded machine it may
  // get there after this thread does.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  bool waited = NodeLimitWaits() > waits_before;
  while (!waited && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    waited = NodeLimitWaits() > waits_before;
  }
  slow.Release();
  batch.join();
  EXPECT_TRUE(waited) << "no free worker found the slow node at its limit";
  EXPECT_EQ(slow.most, static_cast<int>(workers) - 1);
}

TEST(ReadScheduler, ShardsWithoutANodeAreNotLimited) {
  const size_t workers = dfkv::ReadScheduler::WorkerCount();
  HeldShards anonymous;
  std::thread batch([&] {
    dfkv::RunReadScheduled(workers + 1, [&](size_t) { anonymous.Enter(); });
  });
  EXPECT_TRUE(anonymous.WaitInside(static_cast<int>(workers),
                                   std::chrono::seconds(10)));
  anonymous.Release();
  batch.join();
  EXPECT_EQ(anonymous.most, static_cast<int>(workers));
}
