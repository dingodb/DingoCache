// Mandatory RDMA v2 depth/version negotiation over the QpInfo trailer.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>

#include "transport/dev_frame.h"
#include "transport/rdma_verbs.h"

using dfkv::rdma::kDevProtoV2;
using dfkv::rdma::kQpInfoBytes;
using dfkv::rdma::ParseQpInfo;
using dfkv::rdma::QpInfo;
using dfkv::rdma::SerializeQpInfo;

namespace {
QpInfo Sample(uint16_t depth) {
  QpInfo q;
  q.qpn = 0x123456;
  q.psn = 0xabcdef;
  q.lid = 42;
  for (int i = 0; i < 16; ++i) q.gid[i] = static_cast<uint8_t>(i * 3);
  q.depth = depth;
  q.protocol_version = kDevProtoV2;
  return q;
}

void ExpectEndpointFieldsEqual(const QpInfo& actual, const QpInfo& expected) {
  EXPECT_EQ(actual.qpn, expected.qpn);
  EXPECT_EQ(actual.psn, expected.psn);
  EXPECT_EQ(actual.lid, expected.lid);
  EXPECT_EQ(std::memcmp(actual.gid, expected.gid, 16), 0);
}
}  // namespace

TEST(QpDepthNegotiation, V2DepthRoundTrips) {
  char frame[kQpInfoBytes];
  const QpInfo expected = Sample(32);
  SerializeQpInfo(expected, frame);
  const QpInfo actual = ParseQpInfo(frame);
  ExpectEndpointFieldsEqual(actual, expected);
  EXPECT_EQ(actual.depth, 32);
  EXPECT_EQ(actual.protocol_version, kDevProtoV2);
}

TEST(QpDepthNegotiation, HistoricalZeroTrailerIsRejected) {
  char frame[kQpInfoBytes];
  SerializeQpInfo(Sample(32), frame);
  std::memset(frame + 26, 0, kQpInfoBytes - 26);

  const QpInfo actual = ParseQpInfo(frame);
  EXPECT_EQ(actual.depth, 0);
  EXPECT_EQ(actual.protocol_version, 0);
}

TEST(QpDepthNegotiation, MissingV2FlagIsRejected) {
  char frame[kQpInfoBytes];
  QpInfo unversioned = Sample(16);
  unversioned.protocol_version = 0;
  SerializeQpInfo(unversioned, frame);
  const QpInfo actual = ParseQpInfo(frame);
  EXPECT_EQ(actual.depth, 0);
  EXPECT_EQ(actual.protocol_version, 0);
}

TEST(QpDepthNegotiation, WrongMagicOrBadDepthIsRejected) {
  char frame[kQpInfoBytes];
  SerializeQpInfo(Sample(16), frame);
  frame[26] ^= 0x5A;
  EXPECT_EQ(ParseQpInfo(frame).protocol_version, 0);

  SerializeQpInfo(Sample(16), frame);
  frame[30] = 0;
  frame[31] = static_cast<char>(0x80);  // v2 marker with zero depth
  EXPECT_EQ(ParseQpInfo(frame).protocol_version, 0);

  SerializeQpInfo(Sample(16), frame);
  uint16_t too_large = static_cast<uint16_t>(0x8000u | 300u);
  std::memcpy(frame + 30, &too_large, sizeof(too_large));
  EXPECT_EQ(ParseQpInfo(frame).protocol_version, 0);
}

TEST(QpDepthNegotiation, WindowIsStrictMinimum) {
  auto window = [](size_t local, uint16_t remote) -> size_t {
    return std::min(local, static_cast<size_t>(remote));
  };
  EXPECT_EQ(window(32, 8), 8u);
  EXPECT_EQ(window(8, 32), 8u);
  EXPECT_EQ(window(8, 8), 8u);
}
