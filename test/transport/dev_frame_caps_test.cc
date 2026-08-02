// DCP1/DCP2 device-frame capability declarations and every legacy /
// degenerate shape an old peer can produce. Hermetic (no RDMA device).
#include <gtest/gtest.h>

#include <cstring>

#include "transport/dev_frame.h"

namespace dfkv::rdma {

TEST(DevFrameCaps, RoundTrip) {
  char f[kDevNameBytes];
  EncodeDevFrame("ib7s400p0", 4u << 20, f);
  EXPECT_STREQ(f, "ib7s400p0");            // name intact for old servers
  EXPECT_EQ(ParseDevFrameCaps(f), 4u << 20);
}

TEST(DevFrameCaps, V2RoundTripAndLegacyName) {
  char f[kDevNameBytes];
  EncodeDevFrame("ib7s400p0", 4u << 20, f, kDevProtoV2);
  EXPECT_STREQ(f, "ib7s400p0");
  EXPECT_EQ(ParseDevFrameCaps(f), 4u << 20);
  EXPECT_EQ(ParseDevFrameProtocol(f), kDevProtoV2);
}

TEST(DevFrameCaps, V2RequiresDeclarationAndTailRoom) {
  char f[kDevNameBytes];
  EncodeDevFrame("ib7s400p0", 0, f, kDevProtoV2);
  EXPECT_EQ(ParseDevFrameProtocol(f), kDevProtoV1);

  std::string too_long(19, 'x');
  EncodeDevFrame(too_long, 4u << 20, f, kDevProtoV2);
  EXPECT_EQ(ParseDevFrameCaps(f), 0u);
  EXPECT_EQ(ParseDevFrameProtocol(f), kDevProtoV1);

  std::string edge(18, 'y');
  EncodeDevFrame(edge, 4u << 20, f, kDevProtoV2);
  EXPECT_EQ(ParseDevFrameCaps(f), 4u << 20);
  EXPECT_EQ(ParseDevFrameProtocol(f), kDevProtoV2);
}

TEST(DevFrameCaps, LegacyZeroTailParsesAsUndeclared) {
  char f[kDevNameBytes];                    // what every old client sends
  std::memset(f, 0, sizeof(f));
  std::memcpy(f, "mlx5_0", 6);
  EXPECT_EQ(ParseDevFrameCaps(f), 0u);
}

TEST(DevFrameCaps, ZeroDeclarationEncodesLegacyFrame) {
  char f[kDevNameBytes];
  EncodeDevFrame("ib7s400p0", 0, f);
  for (size_t i = 10; i < kDevNameBytes; ++i) EXPECT_EQ(f[i], 0) << i;
  EXPECT_EQ(ParseDevFrameCaps(f), 0u);
}

TEST(DevFrameCaps, NameTooLongSkipsDeclaration) {
  // 20+ chars leaves < 13 tail bytes: encoder must fall back to legacy.
  std::string longname(20, 'x');
  char f[kDevNameBytes];
  EncodeDevFrame(longname, 8u << 20, f);
  EXPECT_EQ(ParseDevFrameCaps(f), 0u);
  EXPECT_EQ(std::string(f), longname);
  // 19 chars is the longest name that still fits a declaration.
  std::string edge(19, 'y');
  EncodeDevFrame(edge, 8u << 20, f);
  EXPECT_EQ(ParseDevFrameCaps(f), 8u << 20);
}

TEST(DevFrameCaps, EmptyDeviceNameStillCarriesCaps) {
  char f[kDevNameBytes];                    // "" = use server default device
  EncodeDevFrame("", 60u << 20, f);
  EXPECT_EQ(f[0], '\0');
  EXPECT_EQ(ParseDevFrameCaps(f), 60u << 20);
}

TEST(DevFrameCaps, GarbageTailIsNotACapsDeclaration) {
  char f[kDevNameBytes];
  std::memset(f, 0, sizeof(f));
  std::memcpy(f, "ib0", 3);
  std::memset(f + 4, 0xAB, kDevNameBytes - 4);  // non-magic junk
  EXPECT_EQ(ParseDevFrameCaps(f), 0u);
}

TEST(DevFrameCaps, NoNulTreatedAsUndeclared) {
  char f[kDevNameBytes];
  std::memset(f, 'z', sizeof(f));           // hostile: no terminator at all
  EXPECT_EQ(ParseDevFrameCaps(f), 0u);
}

}  // namespace dfkv::rdma
