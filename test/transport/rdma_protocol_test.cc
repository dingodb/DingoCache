#include "transport/rdma_protocol.h"

#include <gtest/gtest.h>

namespace dfkv::rdma {

TEST(RdmaProtocol, ProbeRoundTrip) {
  char frame[kDevNameBytes];
  EncodeDevFrame(kV2ProbeDevice, 4u << 20, frame, kDevProtoV2);
  EXPECT_TRUE(IsV2Probe(frame));

  char reply[kV2ProbeReplyBytes];
  EncodeV2ProbeReply(reply);
  EXPECT_TRUE(ParseV2ProbeReply(reply));
  reply[4] = 1;
  EXPECT_FALSE(ParseV2ProbeReply(reply));
}

TEST(RdmaProtocol, SlotGeometryKeepsPayloadAligned) {
  const size_t slot = V2SlotSize(4u << 20);
  ASSERT_NE(slot, 0u);
  EXPECT_EQ(slot % kV2DataOffset, 0u);
  EXPECT_GE(slot - kV2DataOffset, (4u << 20) + ValueHeader::kSize);
  EXPECT_EQ(kV2PutPrefixOffset + kReqPrefix, kV2DataOffset);
  EXPECT_EQ(kV2MaxGetTargets, 29u);
}

TEST(RdmaProtocol, PutCompletionMustCoverExactFrame) {
  const size_t slot = V2SlotSize(4u << 20);
  ASSERT_NE(slot, 0u);
  constexpr uint64_t payload = 64u << 10;
  const uint32_t frame = static_cast<uint32_t>(kReqPrefix + payload);
  EXPECT_TRUE(V2PutCompletionCoversFrame(frame, payload, slot));
  EXPECT_FALSE(V2PutCompletionCoversFrame(frame - 1, payload, slot));
  EXPECT_FALSE(V2PutCompletionCoversFrame(frame + 1, payload, slot));
  EXPECT_FALSE(V2PutCompletionCoversFrame(
      static_cast<uint32_t>(kReqPrefix - 1), payload, slot));
  EXPECT_FALSE(V2PutCompletionCoversFrame(
      frame, static_cast<uint64_t>(slot), slot));
  EXPECT_TRUE(V2PutCompletionIsValid(
      /*is_rdma_write_with_imm=*/true, /*has_immediate=*/true, frame,
      payload, payload, slot));
  EXPECT_FALSE(V2PutCompletionIsValid(
      /*is_rdma_write_with_imm=*/false, /*has_immediate=*/true, frame,
      payload, payload, slot));  // SEND_WITH_IMM is not a shared-slot write
  EXPECT_FALSE(V2PutCompletionIsValid(
      /*is_rdma_write_with_imm=*/true, /*has_immediate=*/false, frame,
      payload, payload, slot));
  EXPECT_FALSE(V2PutCompletionIsValid(
      /*is_rdma_write_with_imm=*/true, /*has_immediate=*/true, frame,
      payload, payload - 1, slot));  // alignment padding is not logical capacity
}

TEST(RdmaProtocol, SegmentInfoRoundTripAndValidation) {
  const RecvSegmentInfo expected{0x12345000, 0xAABBCCDD, 4u << 20};
  char frame[kRecvSegmentInfoBytes];
  EncodeRecvSegmentInfo(expected, frame);
  RecvSegmentInfo actual;
  ASSERT_TRUE(DecodeRecvSegmentInfo(frame, &actual));
  EXPECT_EQ(actual.base_addr, expected.base_addr);
  EXPECT_EQ(actual.rkey, expected.rkey);
  EXPECT_EQ(actual.slot_size, expected.slot_size);

  frame[0] = 0;
  EXPECT_FALSE(DecodeRecvSegmentInfo(frame, &actual));
}

}  // namespace dfkv::rdma
