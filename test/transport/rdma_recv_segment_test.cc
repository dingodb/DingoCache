#include "transport/rdma_recv_segment.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace dfkv::rdma {

TEST(RecvSegment, AllocatesAlignedNonOverlappingLeasesAndCoalesces) {
  RecvSegment segment;
  ASSERT_TRUE(segment.Init(64u << 10));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(segment.data()) % 4096, 0u);

  auto first = segment.Allocate(8192);
  auto second = segment.Allocate(4096);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_EQ(first.offset() % 4096, 0u);
  EXPECT_EQ(second.offset() % 4096, 0u);
  EXPECT_GE(second.offset(), first.offset() + first.size());
  EXPECT_EQ(segment.free_bytes(), (64u << 10) - 12288);

  first.Reset();
  second.Reset();
  EXPECT_EQ(segment.free_bytes(), 64u << 10);
  auto whole = segment.Allocate(64u << 10);
  EXPECT_TRUE(whole);
}

TEST(RecvSegment, MoveTransfersOwnershipAndExhaustionFailsCleanly) {
  RecvSegment segment;
  ASSERT_TRUE(segment.Init(8192));
  auto lease = segment.Allocate(8192);
  ASSERT_TRUE(lease);
  EXPECT_FALSE(segment.Allocate(4096));

  RecvSegment::Lease moved = std::move(lease);
  EXPECT_FALSE(lease);
  EXPECT_TRUE(moved);
  moved.Reset();
  EXPECT_EQ(segment.free_bytes(), 8192u);
}

TEST(RecvSegment, StatsTrackOwnershipFragmentationAndFailures) {
  RecvSegment segment;
  ASSERT_TRUE(segment.Init(64u << 10));
  auto first = segment.Allocate(8192);
  auto second = segment.Allocate(16384);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);

  auto stats = segment.stats();
  EXPECT_EQ(stats.total_bytes, 64u << 10);
  EXPECT_EQ(stats.used_bytes, 24u << 10);
  EXPECT_EQ(stats.free_bytes, 40u << 10);
  EXPECT_EQ(stats.largest_free_range, 40u << 10);
  EXPECT_EQ(stats.allocation_failures, 0u);

  first.Reset();
  stats = segment.stats();
  EXPECT_EQ(stats.used_bytes, 16u << 10);
  EXPECT_EQ(stats.free_bytes, 48u << 10);
  EXPECT_EQ(stats.largest_free_range, 40u << 10);
  EXPECT_FALSE(segment.Allocate(48u << 10))
      << "fragmented free bytes must not masquerade as one range";
  EXPECT_EQ(segment.stats().allocation_failures, 1u);

  second.Reset();
  stats = segment.stats();
  EXPECT_EQ(stats.used_bytes, 0u);
  EXPECT_EQ(stats.free_bytes, 64u << 10);
  EXPECT_EQ(stats.largest_free_range, 64u << 10);
  EXPECT_EQ(stats.allocation_failures, 1u);
}

TEST(RecvSegment, RejectsInvalidGeometry) {
  RecvSegment segment;
  EXPECT_FALSE(segment.Init(0));
  EXPECT_FALSE(segment.Init(4097));
  EXPECT_FALSE(segment.Init(8192, 3000));
}

TEST(RecvSegment, LargerLeaseAlignmentUsesAbsoluteAddress) {
  RecvSegment segment;
  ASSERT_TRUE(segment.Init(64u << 10, 4096));
  auto lease = segment.Allocate(8192, 8192);
  ASSERT_TRUE(lease);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(lease.data()) % 8192, 0u);
}

TEST(RecvSegment, SizeParserIs64BitAlignedAndNotWireCapped) {
  constexpr size_t fallback = 2ull << 30;
  EXPECT_EQ(ResolveRecvSegmentBytes(nullptr, fallback), fallback);
  EXPECT_EQ(ResolveRecvSegmentBytes("", fallback), fallback);
  EXPECT_EQ(ResolveRecvSegmentBytes("0", fallback), fallback);
  EXPECT_EQ(ResolveRecvSegmentBytes("garbage", fallback), fallback);
  EXPECT_EQ(ResolveRecvSegmentBytes("8193", fallback), 12288u);
  EXPECT_EQ(ResolveRecvSegmentBytes("8192junk", fallback), fallback);
  if constexpr (sizeof(size_t) >= 8) {
    EXPECT_EQ(ResolveRecvSegmentBytes("8589934592", fallback),
              static_cast<size_t>(8ull << 30));
    EXPECT_EQ(ResolveRecvSegmentBytes("8589934593", fallback),
              static_cast<size_t>((8ull << 30) + 4096));
  }
  EXPECT_EQ(ResolveRecvSegmentBytes("8192", fallback, 3000), 0u);
}

TEST(RecvSegmentPool, GrowsLazilyWithinHardBudgetAndReusesLeases) {
  RecvSegmentPool pool;
  ASSERT_TRUE(pool.Init(16u << 10, 48u << 10));
  EXPECT_EQ(pool.stats().committed_bytes, 16u << 10);
  EXPECT_EQ(pool.stats().chunks, 1u);

  auto first = pool.Allocate(12u << 10);
  auto second = pool.Allocate(12u << 10);
  auto third = pool.Allocate(12u << 10);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  ASSERT_TRUE(third);
  EXPECT_NE(first.segment(), second.segment());
  EXPECT_NE(second.segment(), third.segment());

  auto stats = pool.stats();
  EXPECT_EQ(stats.max_bytes, 48u << 10);
  EXPECT_EQ(stats.committed_bytes, 48u << 10);
  EXPECT_EQ(stats.used_bytes, 36u << 10);
  EXPECT_EQ(stats.chunks, 3u);
  EXPECT_EQ(stats.growths, 2u);
  EXPECT_FALSE(pool.Allocate(12u << 10));
  EXPECT_EQ(pool.stats().growth_failures, 1u);

  RecvSegment* released_segment = second.segment();
  second.Reset();
  auto replacement = pool.Allocate(12u << 10);
  ASSERT_TRUE(replacement);
  EXPECT_EQ(replacement.segment(), released_segment);
  EXPECT_EQ(pool.stats().growths, 2u);
}

TEST(RecvSegmentPool, RejectsInvalidOrOversizedGeometry) {
  RecvSegmentPool pool;
  EXPECT_FALSE(pool.Init(0, 4096));
  EXPECT_FALSE(pool.Init(8192, 4096));
  ASSERT_TRUE(pool.Init(8192, 16384));
  EXPECT_EQ(pool.initial_segment()->size(), 8192u);
  EXPECT_FALSE(pool.Allocate(20u << 10));
  EXPECT_EQ(pool.stats().committed_bytes, 8192u);
}

TEST(RecvSegmentPool, AffinityPreventsCrossRailChunkReuse) {
  RecvSegmentPool pool;
  ASSERT_TRUE(pool.Init(16u << 10, 48u << 10));
  auto initial = pool.Allocate(12u << 10, 4096, 0);
  auto rail0 = pool.Allocate(12u << 10, 4096, 0);
  auto rail1 = pool.Allocate(12u << 10, 4096, 1);
  ASSERT_TRUE(initial);
  ASSERT_TRUE(rail0);
  ASSERT_TRUE(rail1);
  EXPECT_NE(rail0.segment(), rail1.segment());

  rail0.Reset();
  // rail1 does not borrow the empty rail0 chunk: the budget is full, so
  // that chunk is released at once (not after the idle hold) and rail1
  // gets one of its own.
  auto again = pool.Allocate(12u << 10, 4096, 1);
  ASSERT_TRUE(again);
  int affinity = -2;
  for (const auto& chunk : pool.Snapshot().chunks)
    if (chunk.segment == again.segment()) affinity = chunk.affinity;
  EXPECT_EQ(affinity, 1);
  EXPECT_EQ(pool.stats().chunks, 3u);
  EXPECT_EQ(pool.stats().shrinks, 1u);
}

// RDS-07: the free ranges pressure reclaim plans runs with.
TEST(RecvSegment, FreeRangesListsWhatIsUnleased) {
  RecvSegment segment;
  ASSERT_TRUE(segment.Init(64u << 10));
  auto a = segment.Allocate(16u << 10);
  auto b = segment.Allocate(16u << 10);
  auto c = segment.Allocate(16u << 10);  // 16K free after c
  ASSERT_TRUE(a && b && c);
  using Ranges = std::vector<std::pair<size_t, size_t>>;
  EXPECT_EQ(segment.FreeRanges(), (Ranges{{48u << 10, 16u << 10}}));
  b.Reset();
  EXPECT_EQ(segment.FreeRanges(),
            (Ranges{{16u << 10, 16u << 10}, {48u << 10, 16u << 10}}));
  c.Reset();
  EXPECT_EQ(segment.FreeRanges(), (Ranges{{16u << 10, 48u << 10}}));
}

// A refused growth releases only as many empty chunks as it needs.
TEST(RecvSegmentPool, RefusedGrowthTrimsOnlyWhatItNeeds) {
  RecvSegmentPool pool;
  ASSERT_TRUE(pool.Init(16u << 10, 64u << 10));
  auto initial = pool.Allocate(16u << 10, 4096, 0);
  auto one = pool.Allocate(16u << 10, 4096, 1);
  auto two = pool.Allocate(16u << 10, 4096, 2);
  auto three = pool.Allocate(16u << 10, 4096, 3);
  ASSERT_TRUE(initial && one && two && three);
  one.Reset();
  two.Reset();
  auto four = pool.Allocate(16u << 10, 4096, 4);
  ASSERT_TRUE(four);
  EXPECT_EQ(pool.stats().chunks, 4u) << "one empty chunk is enough";
  EXPECT_EQ(pool.stats().released_bytes, 16u << 10);
}

TEST(RecvSegmentPool, TrimsEmptyNonInitialChunksAfterIdleHold) {
  RecvSegmentPool pool;
  ASSERT_TRUE(pool.Init(16u << 10, 48u << 10));
  auto first = pool.Allocate(12u << 10);
  auto second = pool.Allocate(12u << 10);
  auto third = pool.Allocate(12u << 10);
  ASSERT_EQ(pool.stats().chunks, 3u);
  first.Reset();
  second.Reset();
  third.Reset();
  std::this_thread::sleep_for(std::chrono::milliseconds(3));
  EXPECT_EQ(pool.TrimIdle(1), 32u << 10);
  const auto stats = pool.stats();
  EXPECT_EQ(stats.chunks, 1u);
  EXPECT_EQ(stats.committed_bytes, 16u << 10);
  EXPECT_EQ(stats.shrinks, 1u);
  EXPECT_EQ(stats.released_bytes, 32u << 10);
}

// A chunk a trim takes is offered to the release hook first, outside the
// pool lock (the hook allocates: it would deadlock under it); a chunk the
// hook refuses is kept, never freed, and its budget is not given back.
TEST(RecvSegmentPool, TheReleaseHookRunsOutsideThePoolLockAndCanKeepAChunk) {
  RecvSegmentPool pool;
  std::vector<const char*> offered;
  bool keep = true;
  pool.set_release_hook([&](const RecvSegment* segment) {
    offered.push_back(segment->data());
    auto lease = pool.Allocate(4u << 10);  // the lock is not held
    EXPECT_TRUE(lease);
    return !keep;
  });
  ASSERT_TRUE(pool.Init(16u << 10, 48u << 10));
  auto first = pool.Allocate(12u << 10);
  auto second = pool.Allocate(12u << 10);
  ASSERT_EQ(pool.stats().chunks, 2u);
  const char* second_base = second.segment()->data();
  first.Reset();
  second.Reset();
  std::this_thread::sleep_for(std::chrono::milliseconds(3));
  EXPECT_EQ(pool.TrimIdle(1), 16u << 10);
  ASSERT_EQ(offered.size(), 1u);
  EXPECT_EQ(offered[0], second_base);
  EXPECT_EQ(pool.kept_chunks(), 1u);
  EXPECT_EQ(pool.stats().chunks, 1u);
}

TEST(RecvSegmentPool, ReservedStagingRemainsContiguousWhenConnectionsFillBudget) {
  RecvSegmentPool pool;
  constexpr size_t chunk = 64u << 10;
  ASSERT_TRUE(pool.Init(chunk, 3 * chunk, 4096, chunk));
  EXPECT_FALSE(pool.Init(chunk, 3 * chunk));
  EXPECT_EQ(pool.stats().staging_reserved_bytes, chunk);

  // Multiple connection-sized leases fill both general chunks. They cannot
  // fragment or consume the already committed staging-only initial chunk.
  std::vector<RecvSegmentPool::Lease> connections;
  for (int i = 0; i < 8; ++i) {
    auto lease = pool.Allocate(16u << 10, 4096, 0);
    ASSERT_TRUE(lease);
    EXPECT_NE(lease.segment(), pool.initial_segment());
    connections.push_back(std::move(lease));
  }
  EXPECT_FALSE(pool.Allocate(4096, 4096, 0));
  auto stats = pool.stats();
  EXPECT_EQ(stats.connection_free_bytes, 0u);
  EXPECT_EQ(stats.connection_largest_free_range, 0u);
  EXPECT_EQ(stats.staging_reserved_free_bytes, chunk);
  EXPECT_EQ(stats.free_bytes, chunk);
  EXPECT_EQ(stats.allocation_failures, 1u);

  auto read = pool.Allocate(32u << 10, 4096, 0, -1,
                            RecvSegmentPool::LeaseClass::kStaging);
  auto write = pool.Allocate(32u << 10, 4096, 0, -1,
                             RecvSegmentPool::LeaseClass::kStaging);
  ASSERT_TRUE(read);
  ASSERT_TRUE(write);
  EXPECT_EQ(read.segment(), pool.initial_segment());
  EXPECT_EQ(write.segment(), pool.initial_segment());
  EXPECT_FALSE(pool.Allocate(4096, 4096, 0, -1,
                             RecvSegmentPool::LeaseClass::kStaging));
  EXPECT_EQ(pool.stats().staging_reserved_free_bytes, 0u);

  read.Reset();
  write.Reset();
  EXPECT_EQ(pool.stats().staging_reserved_free_bytes, chunk);
  EXPECT_EQ(pool.stats().connection_free_bytes, 0u);

  // Two nonadjacent general holes contain enough aggregate free bytes, but
  // neither fits a 32-KiB GET. The staging reservation remains contiguous.
  connections[0].Reset();
  connections[2].Reset();
  stats = pool.stats();
  EXPECT_EQ(stats.connection_free_bytes, 32u << 10);
  EXPECT_EQ(stats.connection_largest_free_range, 16u << 10);
  auto fragmented_read = pool.Allocate(
      32u << 10, 4096, 0, -1,
      RecvSegmentPool::LeaseClass::kStaging);
  ASSERT_TRUE(fragmented_read);
  EXPECT_EQ(fragmented_read.segment(), pool.initial_segment());
}

TEST(RecvSegmentPool, RejectsImpossibleStagingReservation) {
  RecvSegmentPool pool;
  EXPECT_FALSE(pool.Init(16u << 10, 16u << 10, 4096, 16u << 10));
  EXPECT_FALSE(pool.Init(16u << 10, 48u << 10, 4096, 8u << 10));
  ASSERT_TRUE(pool.Init(16u << 10, 48u << 10));
  EXPECT_EQ(pool.stats().staging_reserved_bytes, 0u);
}

}  // namespace dfkv::rdma

// RDS-14: what a staging lease costs on the request path, the pool's one lock
// and its scans included, with many connections allocating at once. Opt-in:
// set DFKV_RECV_POOL_BENCH to the leases per thread.
TEST(RecvSegmentPool, StagingLeaseCostBench) {
  const char* requested = std::getenv("DFKV_RECV_POOL_BENCH");
  if (requested == nullptr || *requested == '\0')
    GTEST_SKIP() << "set DFKV_RECV_POOL_BENCH=<leases per thread> to run";
  const int leases = std::max(1, std::atoi(requested));
  // Production-like geometry: 64 MiB chunks, 4 GiB budget, chunks already
  // committed and partly held by long-lived connection leases.
  constexpr size_t kChunk = 64ull << 20;
  for (const int threads : {1, 8, 32, 64}) {
    dfkv::rdma::RecvSegmentPool pool;
    ASSERT_TRUE(pool.Init(kChunk, 64 * kChunk));
    std::vector<dfkv::rdma::RecvSegmentPool::Lease> resident;
    for (int i = 0; i < 128; ++i)  // one 4 MiB connection lease per QP
      resident.push_back(pool.Allocate(4ull << 20));
    std::atomic<uint64_t> failures{0};
    const auto started = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) {
      workers.emplace_back([&, t] {
        uint64_t x = 0x9e3779b97f4a7c15ull * static_cast<uint64_t>(t + 1);
        for (int i = 0; i < leases; ++i) {
          x ^= x << 13; x ^= x >> 7; x ^= x << 17;
          const size_t bytes = (size_t{4096} << (x % 9)) + 4096;  // 8K..1M
          auto lease = pool.Allocate(bytes, 4096, -1, -1,
              dfkv::rdma::RecvSegmentPool::LeaseClass::kStaging);
          if (!lease) failures.fetch_add(1);
        }
      });
    }
    for (auto& w : workers) w.join();
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    const double total = static_cast<double>(threads) * leases;
    // Each thread allocates and releases its leases one after another, as a
    // Serve thread does: the time per lease is what one request pays.
    std::printf("threads %3d: %7.0f ns per lease per thread, %6.2f M leases/s "
                "in all, %llu failures, %zu chunks\n",
                threads, seconds * 1e9 / leases, total / seconds / 1e6,
                static_cast<unsigned long long>(failures.load()),
                pool.stats().chunks);
  }
}
