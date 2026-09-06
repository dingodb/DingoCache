/* Per-op staged-lease PUT protocol contracts (wire codec + capability bits).
 * Nordic verbs-free, so this suite runs on any build host. */
#include "transport/rdma_protocol.h"
#include "transport/dev_frame.h"
#include "transport/wire.h"

#include <cstring>
#include <limits>

#include <gtest/gtest.h>

namespace dfkv::rdma {

TEST(RdmaLeaseProtocol, ProbeAdvertisesLeasedPutByDefault) {
  char reply[kV2ProbeReplyBytes];
  EncodeV2ProbeReply(reply);
  EXPECT_TRUE(ParseV2ProbeReply(reply));
  EXPECT_TRUE(V2ProbeSupportsLeasedPut(reply));
  EXPECT_TRUE(V2ProbeSupportsWriterRetirement(reply));
  EXPECT_TRUE(V2ProbeSupportsPullRead(reply));

  // A rolling-upgraded server that predates the capability advertises only
  // the base bits. The lease bit is absent and the reply stays valid.
  EncodeV2ProbeReply(reply, kV2ProbeCapWriterRetirement | kV2ProbeCapPullRead);
  EXPECT_TRUE(ParseV2ProbeReply(reply));
  EXPECT_FALSE(V2ProbeSupportsLeasedPut(reply));

  // Zero capabilities keeps the reply parseable (old servers reserved the
  // byte) while expressing no optional datapath.
  EncodeV2ProbeReply(reply, /*capabilities=*/0);
  EXPECT_TRUE(ParseV2ProbeReply(reply));
  EXPECT_FALSE(V2ProbeSupportsLeasedPut(reply));
}

TEST(RdmaLeaseProtocol, DevFrameRequestsLeasedPut) {
  char frame[kDevNameBytes];
  EncodeDevFrame("mlx5_0", (4u << 20) | kDevFrameRequestWriterRetirement |
                                kDevFrameRequestPullRead |
                                kDevFrameRequestLeasedPut,
                  frame, kDevProtoV2);
  EXPECT_TRUE(DevFrameRequestsLeasedPut(frame));
  EXPECT_TRUE(DevFrameRequestsPullRead(frame));
  EXPECT_TRUE(DevFrameRequestsWriterRetirement(frame));
  // The request bit must never leak into the geometry value.
  EXPECT_EQ(ParseDevFrameMaxBlock(frame), 4u << 20);

  EncodeDevFrame("mlx5_0", 4u << 20, frame, kDevProtoV2);
  EXPECT_FALSE(DevFrameRequestsLeasedPut(frame));
  EXPECT_EQ(ParseDevFrameMaxBlock(frame), 4u << 20);
}

TEST(RdmaLeaseProtocol, LeasePutReadyRoundTrip) {
  const LeasePutReady expected{
      /*slot=*/3, /*generation=*/17, /*rkey=*/0x00c0ffee,
      /*write_base=*/0x0000abcd12345000ull, /*lease_bytes=*/4096 + (32u << 20)};
  char wire[kLeasePutReadyBytes];
  EncodeLeasePutReady(expected, wire);

  LeasePutReady actual;
  ASSERT_TRUE(DecodeLeasePutReady(wire, &actual));
  EXPECT_EQ(actual.slot, expected.slot);
  EXPECT_EQ(actual.generation, expected.generation);
  EXPECT_EQ(actual.rkey, expected.rkey);
  EXPECT_EQ(actual.write_base, expected.write_base);
  EXPECT_EQ(actual.lease_bytes, expected.lease_bytes);
}

TEST(RdmaLeaseProtocol, LeasePutReadyRejectsInvalidDescriptors) {
  const LeasePutReady valid{
      /*slot=*/1, /*generation=*/5, /*rkey=*/0x00c0ffee,
      /*write_base=*/0x1000, /*lease_bytes=*/8192};
  char wire[kLeasePutReadyBytes];

  LeasePutReady sink;
  // Wrong magic.
  EncodeLeasePutReady(valid, wire);
  wire[0] = 'X';
  EXPECT_FALSE(DecodeLeasePutReady(wire, &sink));

  // Zero rkey: nothing may be written.
  LeasePutReady bad_rkey = valid;
  bad_rkey.rkey = 0;
  EncodeLeasePutReady(bad_rkey, wire);
  EXPECT_FALSE(DecodeLeasePutReady(wire, &sink));

  // Zero write base.
  LeasePutReady bad_base = valid;
  bad_base.write_base = 0;
  EncodeLeasePutReady(bad_base, wire);
  EXPECT_FALSE(DecodeLeasePutReady(wire, &sink));

  // Below one data page: no frame would fit.
  LeasePutReady bad_size = valid;
  bad_size.lease_bytes = kV2DataOffset;
  EncodeLeasePutReady(bad_size, wire);
  EXPECT_FALSE(DecodeLeasePutReady(wire, &sink));

  // Not page aligned: the frame geometry assumes kV2DataOffset multiples.
  LeasePutReady misaligned = valid;
  misaligned.lease_bytes = 8192 + 1;
  EncodeLeasePutReady(misaligned, wire);
  EXPECT_FALSE(DecodeLeasePutReady(wire, &sink));

  // Overflowing range end.
  LeasePutReady overflow = valid;
  overflow.write_base = std::numeric_limits<uint64_t>::max() - 4096;
  overflow.lease_bytes = 65536;
  EncodeLeasePutReady(overflow, wire);
  EXPECT_FALSE(DecodeLeasePutReady(wire, &sink));

  // Generation zero is the reserved "absent" value.
  LeasePutReady zero_gen = valid;
  zero_gen.generation = 0;
  EncodeLeasePutReady(zero_gen, wire);
  EXPECT_FALSE(DecodeLeasePutReady(wire, &sink));
}

TEST(RdmaLeaseProtocol, LeasePutOpCodeStaysAppendOnly) {
  // New op codes are append-only and never reuse old values: pooled v2.25
  // peers classify unknown ops as invalid, so the staged-lease datapath must
  // be reachable only after both ends negotiate this code.
  EXPECT_EQ(WireOp::kPullRange, static_cast<WireOp>(16));
  EXPECT_EQ(WireOp::kPullRelease, static_cast<WireOp>(17));
  EXPECT_EQ(WireOp::kLeasePut, static_cast<WireOp>(18));
}

}  // namespace dfkv::rdma

namespace dfkv {
// Keep the wire status contract: staged-lease backpressure must ride the
// existing kCacheFull value, because DecodeRespVersion rejects anything above
// kInvalid (new status values would break every deployed peer).
static_assert(static_cast<uint8_t>(Status::kInvalid) == 5);
}  // namespace dfkv
