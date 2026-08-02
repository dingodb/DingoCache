// Mandatory RDMA v2 device-frame negotiation. Hermetic (no RDMA device).
#include <gtest/gtest.h>

#include <cstring>

#include "transport/dev_frame.h"

namespace dfkv::rdma {

TEST(DevFrameCaps, V2RoundTrip) {
  char frame[kDevNameBytes];
  EncodeDevFrame("ib7s400p0", 4u << 20, frame);
  EXPECT_STREQ(frame, "ib7s400p0");
  EXPECT_EQ(ParseDevFrameCaps(frame), 4u << 20);
  EXPECT_EQ(ParseDevFrameProtocol(frame), kDevProtoV2);
}

TEST(DevFrameCaps, V2RequiresDeclarationAndTailRoom) {
  char f[kDevNameBytes];
  EncodeDevFrame("ib7s400p0", 0, f, kDevProtoV2);
  EXPECT_EQ(ParseDevFrameProtocol(f), 0);

  std::string too_long(19, 'x');
  EncodeDevFrame(too_long, 4u << 20, f, kDevProtoV2);
  EXPECT_EQ(ParseDevFrameCaps(f), 0u);
  EXPECT_EQ(ParseDevFrameProtocol(f), 0);

  std::string edge(18, 'y');
  EncodeDevFrame(edge, 4u << 20, f, kDevProtoV2);
  EXPECT_EQ(ParseDevFrameCaps(f), 4u << 20);
  EXPECT_EQ(ParseDevFrameProtocol(f), kDevProtoV2);
}

TEST(DevFrameCaps, HistoricalProtocolFrameIsRejected) {
  char frame[kDevNameBytes] = {};
  std::memcpy(frame, "mlx5_0", 6);
  const uint32_t historical_magic = 0x31504344u;
  const uint64_t declaration = 4u << 20;
  std::memcpy(frame + 7, &historical_magic, sizeof(historical_magic));
  std::memcpy(frame + 11, &declaration, sizeof(declaration));
  EXPECT_EQ(ParseDevFrameCaps(frame), 0u);
  EXPECT_EQ(ParseDevFrameProtocol(frame), 0);
}

TEST(DevFrameCaps, ZeroDeclarationIsRejected) {
  char frame[kDevNameBytes];
  EncodeDevFrame("ib7s400p0", 0, frame);
  EXPECT_EQ(ParseDevFrameCaps(frame), 0u);
  EXPECT_EQ(ParseDevFrameProtocol(frame), 0);
}

TEST(DevFrameCaps, NameWithoutTailRoomIsRejected) {
  std::string long_name(19, 'x');
  char frame[kDevNameBytes];
  EncodeDevFrame(long_name, 8u << 20, frame);
  EXPECT_EQ(ParseDevFrameCaps(frame), 0u);
  EXPECT_EQ(ParseDevFrameProtocol(frame), 0);

  std::string edge(18, 'y');
  EncodeDevFrame(edge, 8u << 20, frame);
  EXPECT_EQ(ParseDevFrameCaps(frame), 8u << 20);
  EXPECT_EQ(ParseDevFrameProtocol(frame), kDevProtoV2);
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

TEST(DevFrameCaps, MissingTerminatorIsRejected) {
  char frame[kDevNameBytes];
  std::memset(frame, 'z', sizeof(frame));
  EXPECT_EQ(ParseDevFrameCaps(frame), 0u);
  EXPECT_EQ(ParseDevFrameProtocol(frame), 0);
}

}  // namespace dfkv::rdma
