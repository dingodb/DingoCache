// Wire-framing encode/decode: round-trips plus the negative paths a hostile or
// version-skewed peer can take (bad version byte, oversized declared length).
#include "transport/wire.h"

#include <cstring>

#include <gtest/gtest.h>

using namespace dfkv;

TEST(Wire, ReqRoundTrip) {
  char buf[kReqPrefix];
  BlockKey k{0x1122334455667788ull, 0xAABBCCDD, 0x12345678};
  EncodeReq(buf, WireOp::kCache, k, 4096, 8192, 65536);
  ReqFields rq;
  ASSERT_TRUE(DecodeReq(buf, &rq));
  EXPECT_EQ(rq.op, static_cast<uint8_t>(WireOp::kCache));
  EXPECT_EQ(rq.id, k.id);
  EXPECT_EQ(rq.index, k.index);
  EXPECT_EQ(rq.size, k.size);
  EXPECT_EQ(rq.offset, 4096u);
  EXPECT_EQ(rq.length, 8192u);
  EXPECT_EQ(rq.payload_len, 65536u);
}

TEST(Wire, RespRoundTrip) {
  char buf[kRespPrefix];
  EncodeResp(buf, Status::kOk, 1u << 20);
  Status st = Status::kInvalid;
  uint64_t dlen = 0;
  ASSERT_TRUE(DecodeResp(buf, &st, &dlen));
  EXPECT_EQ(st, Status::kOk);
  EXPECT_EQ(dlen, 1u << 20);
}

TEST(Wire, ReqRejectsBadVersion) {
  char buf[kReqPrefix];
  EncodeReq(buf, WireOp::kCache, BlockKey{1, 2, 3}, 0, 0, 0);
  buf[0] = static_cast<char>(kProtoVersion + 1);  // unknown version
  ReqFields rq;
  EXPECT_FALSE(DecodeReq(buf, &rq));
}

TEST(Wire, RespRejectsBadVersion) {
  char buf[kRespPrefix];
  EncodeResp(buf, Status::kOk, 0);
  buf[0] = static_cast<char>(kProtoVersion + 1);  // unknown version
  Status st = Status::kInvalid;
  uint64_t dlen = 0;
  EXPECT_FALSE(DecodeResp(buf, &st, &dlen));
}

TEST(Wire, ReqRejectsOversizedPayload) {
  char buf[kReqPrefix];
  // A garbage/hostile 64-bit length must NOT decode (else it drives a huge alloc).
  EncodeReq(buf, WireOp::kCache, BlockKey{1, 2, 3}, 0, 0, kMaxFrameLen + 1);
  ReqFields rq;
  EXPECT_FALSE(DecodeReq(buf, &rq));               // rejected by default ceiling
  EXPECT_TRUE(DecodeReq(buf, &rq, kMaxFrameLen + 1));  // explicit higher bound accepts
  EXPECT_EQ(rq.payload_len, kMaxFrameLen + 1);
}

TEST(Wire, ReqAcceptsPayloadAtCeiling) {
  char buf[kReqPrefix];
  EncodeReq(buf, WireOp::kCache, BlockKey{1, 2, 3}, 0, 0, kMaxFrameLen);
  ReqFields rq;
  EXPECT_TRUE(DecodeReq(buf, &rq));  // exactly at the ceiling is allowed
  EXPECT_EQ(rq.payload_len, kMaxFrameLen);
}

TEST(Wire, ReqRejectsOverTighterBound) {
  char buf[kReqPrefix];
  EncodeReq(buf, WireOp::kCache, BlockKey{1, 2, 3}, 0, 0, 4097);
  ReqFields rq;
  EXPECT_FALSE(DecodeReq(buf, &rq, 4096));  // caller-supplied tighter cap enforced
  EXPECT_TRUE(DecodeReq(buf, &rq, 4097));
}

TEST(Wire, RespRejectsOversizedData) {
  char buf[kRespPrefix];
  EncodeResp(buf, Status::kOk, kMaxFrameLen + 1);
  Status st = Status::kInvalid;
  uint64_t dlen = 0;
  EXPECT_FALSE(DecodeResp(buf, &st, &dlen));
  EXPECT_TRUE(DecodeResp(buf, &st, &dlen, kMaxFrameLen + 1));
  EXPECT_EQ(dlen, kMaxFrameLen + 1);
}

TEST(Wire, V2GetScatterRoundTrip) {
  char buf[512];
  const BlockKey key{0x1122, 7, 4096};
  const std::vector<RdmaWriteTarget> targets{
      {0x100000, 0xA1, 1024}, {0x200000, 0xB2, 3024}};
  size_t encoded = 0;
  ASSERT_TRUE(EncodeRdmaGetReq(buf, sizeof(buf), key, 0, 4096, 48, targets,
                               &encoded));
  EXPECT_EQ(encoded, RdmaGetFrameSize(2));

  ReqFields req{};
  RdmaGetFields get;
  ASSERT_TRUE(DecodeRdmaGetReq(buf, encoded, &req, &get));
  EXPECT_EQ(req.op, static_cast<uint8_t>(WireOp::kRange));
  EXPECT_EQ(req.id, key.id);
  EXPECT_EQ(req.length, 4096u);
  EXPECT_EQ(get.header_len, 48u);
  ASSERT_EQ(get.targets.size(), 2u);
  EXPECT_EQ(get.targets[0].addr, targets[0].addr);
  EXPECT_EQ(get.targets[1].rkey, targets[1].rkey);
  EXPECT_EQ(get.Capacity(), 4048u);
}

TEST(Wire, V2GetRejectsShortCapacityAndMalformedTarget) {
  char buf[256];
  size_t encoded = 0;
  ASSERT_TRUE(EncodeRdmaGetReq(
      buf, sizeof(buf), BlockKey{1, 2, 3}, 0, 4096, 48,
      std::vector<RdmaWriteTarget>{{0x1000, 7, 1024}}, &encoded));
  ReqFields req{};
  RdmaGetFields get;
  EXPECT_FALSE(DecodeRdmaGetReq(buf, encoded, &req, &get));

  ASSERT_TRUE(EncodeRdmaGetReq(
      buf, sizeof(buf), BlockKey{1, 2, 3}, 0, 48, 0,
      std::vector<RdmaWriteTarget>{{0, 0, 48}}, &encoded));
  EXPECT_FALSE(DecodeRdmaGetReq(buf, encoded, &req, &get));
}

TEST(Wire, VersionedResponseRequiresNegotiatedVersion) {
  char buf[kRespPrefix];
  EncodeRespVersion(buf, kProtoVersionV2, Status::kOk, 99);
  Status status = Status::kInvalid;
  uint64_t data_len = 0;
  EXPECT_FALSE(DecodeResp(buf, &status, &data_len));
  ASSERT_TRUE(DecodeRespVersion(buf, kProtoVersionV2, &status, &data_len));
  EXPECT_EQ(status, Status::kOk);
  EXPECT_EQ(data_len, 99u);
}
