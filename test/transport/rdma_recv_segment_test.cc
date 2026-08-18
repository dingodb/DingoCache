#include "transport/rdma_recv_segment.h"

#include <cstdint>
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

}  // namespace dfkv::rdma
