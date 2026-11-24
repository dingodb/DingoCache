// The quarantine bound and the bind-only region registry of the server's
// memory windows (DFKV_RDMA_SERVER_MW), without a device: registration and
// deregistration are fakes.
#include "transport/rdma_mw.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

using dfkv::rdma::BindRegionRegistry;
using dfkv::rdma::MwQuarantineUs;

namespace {

// Distinct fake registrations: never dereferenced.
ibv_mr* FakeMr(uintptr_t n) { return reinterpret_cast<ibv_mr*>(n * 64); }

// Waits for the registry's thread.
template <typename Done>
bool Eventually(Done done) {
  const auto until =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < until) {
    if (done()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

}  // namespace

TEST(MwQuarantine, IsTwiceTheWireTimePlusTheMarginNeverBelowTheFloor) {
  // 400 Gbit/s: 4 MiB is 84 us on the wire.
  EXPECT_EQ(MwQuarantineUs(4u << 20, 400000, 5000, 1000), 5000u);
  // 100 Gbit/s: 64 MiB is 5369 us; twice that plus 1 ms.
  EXPECT_EQ(MwQuarantineUs(64u << 20, 100000, 5000, 1000), 2 * 5369 + 1000u);
  // Unknown rate: taken as 10 Gbit/s.
  EXPECT_EQ(MwQuarantineUs(1u << 20, 0, 0, 0), 2 * 839u);
}

// The first grant into a region starts its registration and is served
// without it; one under way is not started again; then it is found.
TEST(BindRegionRegistry, RegistersARegionOncePerRailInTheBackground) {
  std::mutex mu;
  std::condition_variable cv;
  bool open = false;
  std::map<std::pair<size_t, char*>, int> calls;
  BindRegionRegistry registry;
  registry.Start(
      2,
      [&](size_t rail, char* base, size_t) {
        std::unique_lock<std::mutex> lock(mu);
        ++calls[{rail, base}];
        cv.wait(lock, [&] { return open; });
        return FakeMr(rail + 1);
      },
      [](ibv_mr*) { return 0; }, 0);
  std::vector<char> region(4096);
  using Kind = BindRegionRegistry::Kind;
  EXPECT_EQ(registry.Find(0, region.data(), region.size(), Kind::kChunk),
            nullptr);
  EXPECT_EQ(registry.Find(0, region.data(), region.size(), Kind::kChunk),
            nullptr);
  {
    std::lock_guard<std::mutex> lock(mu);
    open = true;
  }
  cv.notify_all();
  ASSERT_TRUE(Eventually([&] {
    return registry.Find(0, region.data(), region.size(), Kind::kChunk) ==
           FakeMr(1);
  }));
  EXPECT_EQ(registry.Find(1, region.data(), region.size(), Kind::kChunk),
            nullptr);
  ASSERT_TRUE(Eventually([&] {
    return registry.Find(1, region.data(), region.size(), Kind::kChunk) ==
           FakeMr(2);
  }));
  std::lock_guard<std::mutex> lock(mu);
  EXPECT_EQ(calls.size(), 2u);
  for (const auto& [where, count] : calls) EXPECT_EQ(count, 1) << where.first;
}

// A failed registration is not tried again at once; arena regions stop at
// the per-rail cap.
TEST(BindRegionRegistry, AFailedRegionWaitsAndArenaRegionsStopAtTheCap) {
  std::atomic<int> calls{0};
  BindRegionRegistry registry;
  std::vector<char> arena(4 * 4096);
  registry.Start(
      1,
      [&](size_t, char* base, size_t) -> ibv_mr* {
        calls.fetch_add(1);
        return base == arena.data() ? nullptr : FakeMr(1);
      },
      [](ibv_mr*) { return 0; }, 2 * 4096);
  using Kind = BindRegionRegistry::Kind;
  EXPECT_EQ(registry.Find(0, arena.data(), 4096, Kind::kArena), nullptr);
  ASSERT_TRUE(Eventually([&] { return registry.RegisterFailures() == 1; }));
  EXPECT_EQ(registry.Find(0, arena.data(), 4096, Kind::kArena), nullptr);
  EXPECT_EQ(registry.Find(0, arena.data() + 4096, 4096, Kind::kArena),
            nullptr);
  EXPECT_EQ(registry.Find(0, arena.data() + 8192, 4096, Kind::kArena),
            nullptr);
  ASSERT_TRUE(Eventually([&] { return registry.Registrations() == 2; }));
  // The failed one does not count against the cap.
  EXPECT_EQ(registry.RegisteredBytes(0, Kind::kArena), 2 * 4096u);
  EXPECT_EQ(registry.Find(0, arena.data(), 4096, Kind::kArena), nullptr);
  // Past the cap: not registered at all.
  EXPECT_EQ(registry.Find(0, arena.data() + 12288, 4096, Kind::kArena),
            nullptr);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_EQ(calls.load(), 3);
}

// Forget waits for a registration under way, deregisters on every rail,
// and reports a region a window still holds (EBUSY) so its memory is kept.
TEST(BindRegionRegistry, ForgetWaitsForARegistrationAndReportsABoundWindow) {
  std::mutex mu;
  std::condition_variable cv;
  bool open = false;
  bool entered = false;  // a registration is under way
  std::set<ibv_mr*> deregistered;
  int busy = 0;
  BindRegionRegistry registry;
  registry.Start(
      2,
      [&](size_t rail, char*, size_t) {
        std::unique_lock<std::mutex> lock(mu);
        entered = true;
        cv.wait(lock, [&] { return open; });
        return FakeMr(rail + 1);
      },
      [&](ibv_mr* mr) {
        std::lock_guard<std::mutex> lock(mu);
        deregistered.insert(mr);
        return mr == FakeMr(2) && busy-- > 0 ? EBUSY : 0;
      },
      0);
  std::vector<char> region(4096);
  using Kind = BindRegionRegistry::Kind;
  registry.Find(0, region.data(), region.size(), Kind::kChunk);
  registry.Find(1, region.data(), region.size(), Kind::kChunk);
  ASSERT_TRUE(Eventually([&] {
    std::lock_guard<std::mutex> lock(mu);
    return entered;
  }));
  std::thread opener([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::lock_guard<std::mutex> lock(mu);
    open = true;
    cv.notify_all();
  });
  // Rail 0's is under way: Forget waits for it, then deregisters it.
  EXPECT_TRUE(registry.Forget(region.data()));
  opener.join();
  {
    std::lock_guard<std::mutex> lock(mu);
    EXPECT_TRUE(deregistered.count(FakeMr(1)) == 1);
  }
  EXPECT_EQ(registry.Find(0, region.data(), region.size(), Kind::kChunk),
            nullptr);
  ASSERT_TRUE(Eventually([&] {
    return registry.Find(0, region.data(), region.size(), Kind::kChunk) !=
           nullptr;
  }));
  registry.Find(1, region.data(), region.size(), Kind::kChunk);
  ASSERT_TRUE(Eventually([&] {
    return registry.Find(1, region.data(), region.size(), Kind::kChunk) !=
           nullptr;
  }));
  {
    std::lock_guard<std::mutex> lock(mu);
    busy = 1;
  }
  EXPECT_FALSE(registry.Forget(region.data()));
}
