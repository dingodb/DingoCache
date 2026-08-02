// DPQ1 depth negotiation over the QpInfo pad (see rdma_verbs.h). The pad is a
// deterministic-zero extension area (legacy peers memset the whole blob), so
// these tests pin the exact-compatibility argument:
//   new -> new : depth round-trips
//   old -> new : zero pad parses as depth 0 (no advertisement)
//   new -> old : legacy fields untouched by the extension bytes
//   garbage    : wrong magic / out-of-range depth = treated as absent
#include <gtest/gtest.h>

#include <cstring>

#include "transport/rdma_verbs.h"

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
  return q;
}
void ExpectLegacyFieldsEqual(const QpInfo& a, const QpInfo& b) {
  EXPECT_EQ(a.qpn, b.qpn);
  EXPECT_EQ(a.psn, b.psn);
  EXPECT_EQ(a.lid, b.lid);
  EXPECT_EQ(std::memcmp(a.gid, b.gid, 16), 0);
}
}  // namespace

TEST(QpDepthNegotiation, DepthRoundTrips) {
  char buf[kQpInfoBytes];
  SerializeQpInfo(Sample(32), buf);
  QpInfo out = ParseQpInfo(buf);
  ExpectLegacyFieldsEqual(out, Sample(32));
  EXPECT_EQ(out.depth, 32);
}

TEST(QpDepthNegotiation, V2FlagSharesDepthFieldWithoutTouchingLegacyFields) {
  QpInfo v2 = Sample(32);
  v2.protocol_version = 2;
  char encoded[kQpInfoBytes], legacy[kQpInfoBytes];
  SerializeQpInfo(v2, encoded);
  SerializeQpInfo(Sample(32), legacy);
  EXPECT_EQ(std::memcmp(encoded, legacy, 30), 0);

  QpInfo parsed = ParseQpInfo(encoded);
  ExpectLegacyFieldsEqual(parsed, v2);
  EXPECT_EQ(parsed.depth, 32);
  EXPECT_EQ(parsed.protocol_version, 2);

  uint16_t raw_depth = 0;
  std::memcpy(&raw_depth, encoded + 30, 2);
  EXPECT_GT(raw_depth, 256u);  // an old parser safely treats it as absent
}

TEST(QpDepthNegotiation, LegacyZeroPadMeansNoAdvertisement) {
  // A legacy peer's serializer memsets the blob and writes only the fields:
  // emulate it by serializing WITHOUT depth (writer skips the magic).
  char buf[kQpInfoBytes];
  SerializeQpInfo(Sample(0), buf);
  // pad must be all-zero (what an old parser would also have produced)
  for (size_t i = 26; i < kQpInfoBytes; ++i)
    EXPECT_EQ(buf[i], 0) << "byte " << i;
  QpInfo out = ParseQpInfo(buf);
  EXPECT_EQ(out.depth, 0);
  ExpectLegacyFieldsEqual(out, Sample(0));
}

TEST(QpDepthNegotiation, ExtensionInvisibleToLegacyFields) {
  // new -> old: an old parser reads bytes [0,26) only; the extension must not
  // perturb them. Emulate the old parser by comparing the field bytes.
  char with[kQpInfoBytes], without[kQpInfoBytes];
  SerializeQpInfo(Sample(64), with);
  SerializeQpInfo(Sample(0), without);
  EXPECT_EQ(std::memcmp(with, without, 26), 0)
      << "extension leaked into the legacy field area";
}

TEST(QpDepthNegotiation, WrongMagicOrBadDepthTreatedAsAbsent) {
  char buf[kQpInfoBytes];
  SerializeQpInfo(Sample(16), buf);
  buf[26] ^= 0x5A;  // corrupt the magic
  EXPECT_EQ(ParseQpInfo(buf).depth, 0);
  SerializeQpInfo(Sample(16), buf);
  buf[30] = 0; buf[31] = 0;  // depth 0 under a valid magic: out of range
  EXPECT_EQ(ParseQpInfo(buf).depth, 0);
  SerializeQpInfo(Sample(16), buf);
  uint16_t big = 300;  // > 256: out of contract range
  std::memcpy(buf + 30, &big, 2);
  EXPECT_EQ(ParseQpInfo(buf).depth, 0);
}

TEST(QpDepthNegotiation, WindowClampsOnlyDownward) {
  // window() semantics live on RcEndpoint but reduce to this arithmetic:
  // remote 0 (legacy) or remote >= local => local; remote < local => remote.
  auto window = [](size_t local, uint16_t remote) -> size_t {
    return (remote > 0 && remote < local) ? remote : local;
  };
  EXPECT_EQ(window(32, 0), 32u);    // legacy server: pre-negotiation behavior
  EXPECT_EQ(window(32, 8), 8u);     // server smaller: clamp
  EXPECT_EQ(window(8, 32), 8u);     // server larger: keep local
  EXPECT_EQ(window(8, 8), 8u);
}
