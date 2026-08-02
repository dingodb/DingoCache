// Wire-framing encode/decode: round-trips plus the negative paths a hostile or
// version-skewed peer can take (bad version byte, oversized declared length).
#include "transport/wire.h"

#include <cstring>

#include <gtest/gtest.h>

using namespace dfkv;

TEST(Wire, ReqRoundTrip) {
  char buf[kReqPrefix];
  BlockKey k{0x1122334455667788ULL, 0xaabbccdd12345678ULL,
             0x0102030405060708ULL};
  EncodeReq(buf, WireOp::kCache, k, 4096, 8192, 65536);
  ReqFields rq;
  ASSERT_TRUE(DecodeReq(buf, &rq));
  EXPECT_EQ(rq.op, static_cast<uint8_t>(WireOp::kCache));
  EXPECT_EQ(rq.tenant_hash, k.tenant_hash);
  EXPECT_EQ(rq.digest_hi, k.digest_hi);
  EXPECT_EQ(rq.digest_lo, k.digest_lo);
  EXPECT_EQ(rq.offset, 4096u);
  EXPECT_EQ(rq.length, 8192u);
  EXPECT_EQ(rq.payload_len, 65536u);
}

TEST(Wire, TenantScopedEpochAndPrefixAreExact) {
  EXPECT_EQ(kNativeProtoTcp, 6);
  EXPECT_EQ(kNativeProtoRdmaV2, 7);
  EXPECT_EQ(kReqPrefix, 50u);
}

TEST(Wire, RespRoundTripCarriesAuthoritativeStoredLength) {
  char buf[kRespPrefix];
  EncodeResp(buf, Status::kQuotaExceeded, 4096, 16384);
  Status status = Status::kInvalid;
  uint64_t data_len = 0;
  uint64_t stored_len = 0;
  ASSERT_TRUE(DecodeResp(buf, &status, &data_len, kMaxFrameLen, &stored_len));
  EXPECT_EQ(status, Status::kQuotaExceeded);
  EXPECT_EQ(data_len, 4096u);
  EXPECT_EQ(stored_len, 16384u);
}

TEST(Wire, ReqRejectsBadVersion) {
  char buf[kReqPrefix];
  EncodeReq(buf, WireOp::kCache, BlockKey{1, 2}, 0, 0, 0);
  buf[0] = static_cast<char>(kProtoVersion + 1);  // unknown version
  ReqFields rq;
  EXPECT_FALSE(DecodeReq(buf, &rq));
}

TEST(Wire, PreviousTcpEpochIsRejectedWithoutDecoding) {
  char buf[kReqPrefix] = {};
  EncodeReqVersion(buf, kNativeProtoTcp - 1, WireOp::kCache,
                   BlockKey{1, 2, 3}, 0, 0, 0);
  ReqFields request{};
  EXPECT_FALSE(DecodeReq(buf, &request));
}

TEST(Wire, RespRejectsBadVersion) {
  char buf[kRespPrefix];
  EncodeResp(buf, Status::kOk, 0);
  buf[0] = static_cast<char>(kProtoVersion + 1);  // unknown version
  Status st = Status::kInvalid;
  uint64_t dlen = 0;
  EXPECT_FALSE(DecodeResp(buf, &st, &dlen));
}

TEST(Wire, ResponseRejectsUnknownStatusWithoutPublishingOutputs) {
  char buf[kRespPrefix];
  EncodeRespVersion(buf, kNativeProtoRdmaV2, Status::kOk, 17, 23);
  buf[1] = static_cast<char>(
      static_cast<uint8_t>(Status::kInvalid) + 1);
  Status status = Status::kNotFound;
  uint64_t data_len = 41;
  uint64_t value_len = 43;
  EXPECT_FALSE(DecodeRespVersion(buf, kNativeProtoRdmaV2, &status, &data_len,
                                 kMaxFrameLen, &value_len));
  EXPECT_EQ(status, Status::kNotFound);
  EXPECT_EQ(data_len, 41u);
  EXPECT_EQ(value_len, 43u);

  buf[0] = static_cast<char>(kNativeProtoTcp);
  EXPECT_FALSE(DecodeResp(buf, &status, &data_len, kMaxFrameLen, &value_len));
}

TEST(Wire, HistoricalShortResponseEpochIsRejected) {
  char frame[kRespPrefix] = {};
  frame[0] = 1;
  frame[1] = static_cast<char>(Status::kOk);
  net::PutU64(frame + 2, 4096);
  Status status = Status::kInvalid;
  uint64_t data_len = 0;
  uint64_t stored_len = 0;
  EXPECT_FALSE(
      DecodeResp(frame, &status, &data_len, kMaxFrameLen, &stored_len));
}

TEST(Wire, ReqRejectsOversizedPayload) {
  char buf[kReqPrefix];
  // A garbage/hostile 64-bit length must NOT decode (else it drives a huge alloc).
  EncodeReq(buf, WireOp::kCache, BlockKey{1, 2}, 0, 0, kMaxFrameLen + 1);
  ReqFields rq;
  EXPECT_FALSE(DecodeReq(buf, &rq));               // rejected by default ceiling
  EXPECT_TRUE(DecodeReq(buf, &rq, kMaxFrameLen + 1));  // explicit higher bound accepts
  EXPECT_EQ(rq.payload_len, kMaxFrameLen + 1);
}

TEST(Wire, ReqAcceptsPayloadAtCeiling) {
  char buf[kReqPrefix];
  EncodeReq(buf, WireOp::kCache, BlockKey{1, 2}, 0, 0, kMaxFrameLen);
  ReqFields rq;
  EXPECT_TRUE(DecodeReq(buf, &rq));  // exactly at the ceiling is allowed
  EXPECT_EQ(rq.payload_len, kMaxFrameLen);
}

TEST(Wire, ReqRejectsOverTighterBound) {
  char buf[kReqPrefix];
  EncodeReq(buf, WireOp::kCache, BlockKey{1, 2}, 0, 0, 4097);
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
  const BlockKey key{0x1122, 0x33445566778899aaULL,
                     0xaabbccddeeff0011ULL};
  const std::vector<RdmaWriteTarget> targets{
      {0x100000, 0xA1, 1024}, {0x200000, 0xB2, 3072}};
  size_t encoded = 0;
  ASSERT_TRUE(EncodeRdmaGetReq(buf, sizeof(buf), key, 0, 4096, targets,
                               &encoded));
  EXPECT_EQ(encoded, RdmaGetFrameSize(2));

  ReqFields req{};
  RdmaGetFields get;
  ASSERT_TRUE(DecodeRdmaGetReq(buf, encoded, &req, &get));
  EXPECT_EQ(req.op, static_cast<uint8_t>(WireOp::kRange));
  EXPECT_EQ(req.Key(), key);
  EXPECT_EQ(req.length, 4096u);
  ASSERT_EQ(get.targets.size(), 2u);
  EXPECT_EQ(get.targets[0].addr, targets[0].addr);
  EXPECT_EQ(get.targets[1].rkey, targets[1].rkey);
  EXPECT_EQ(get.Capacity(), 4096u);
}

TEST(Wire, V2GetRejectsShortCapacityAndMalformedTarget) {
  char buf[256];
  size_t encoded = 0;
  EXPECT_FALSE(EncodeRdmaGetReq(
      buf, sizeof(buf), BlockKey{1, 2}, 0, 4096,
      std::vector<RdmaWriteTarget>{{0x1000, 7, 1024}}, &encoded));
  EXPECT_FALSE(EncodeRdmaGetReq(
      buf, sizeof(buf), BlockKey{1, 2}, 0, 48,
      std::vector<RdmaWriteTarget>{{0, 0, 48}}, &encoded));

  ASSERT_TRUE(EncodeRdmaGetReq(
      buf, sizeof(buf), BlockKey{1, 2}, 0, 48,
      std::vector<RdmaWriteTarget>{{0x1000, 7, 48}}, &encoded));
  net::PutU64(buf + kReqPrefix + kRdmaGetFixed, 0);
  ReqFields req{};
  RdmaGetFields get;
  EXPECT_FALSE(DecodeRdmaGetReq(buf, encoded, &req, &get));
}

TEST(Wire, RdmaResponseRequiresNegotiatedVersionAndStoredLength) {
  char buf[kRespPrefix];
  EncodeRespVersion(buf, kNativeProtoRdmaV2, Status::kOk, 99, 123);
  Status status = Status::kInvalid;
  uint64_t data_len = 0;
  uint64_t stored_len = 0;
  EXPECT_FALSE(DecodeResp(buf, &status, &data_len, kMaxFrameLen, &stored_len));
  ASSERT_TRUE(DecodeRespVersion(buf, kNativeProtoRdmaV2, &status, &data_len,
                                kMaxFrameLen, &stored_len));
  EXPECT_EQ(status, Status::kOk);
  EXPECT_EQ(data_len, 99u);
  EXPECT_EQ(stored_len, 123u);
}

TEST(Wire, PreviousRdmaEpochIsRejectedWithoutDecoding) {
  char buf[256] = {};
  size_t encoded = 0;
  ASSERT_TRUE(EncodeRdmaGetReq(
      buf, sizeof(buf), BlockKey{1, 2, 3}, 0, 16,
      std::vector<RdmaWriteTarget>{{0x1000, 7, 16}}, &encoded));
  buf[0] = static_cast<char>(kNativeProtoRdmaV2 - 1);
  ReqFields request{};
  RdmaGetFields get;
  EXPECT_FALSE(DecodeRdmaGetReq(buf, encoded, &request, &get));
}
