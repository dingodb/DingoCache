#include "transport/rdma_recv_segment.h"

#include <chrono>
#include <cstdint>
#include <thread>
#include <utility>

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
  EXPECT_FALSE(pool.Allocate(12u << 10, 4096, 1))
      << "rail1 must not borrow an empty chunk bound to rail0";
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

}  // namespace dfkv::rdma
