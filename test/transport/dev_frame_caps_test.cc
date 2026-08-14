// Mandatory RDMA v2 device-frame negotiation. Hermetic (no RDMA device).
#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "transport/dev_frame.h"

// Startup fail-fast wiring lives in the RDMA transport/server: only RDMA
// builds compile that coverage below; elsewhere it is compiled away (same
// pattern as rail_select_test.cc).
#ifdef DFKV_WITH_RDMA
#include "cache/rdma_server.h"
#include "transport/rdma_transport.h"
#endif

namespace dfkv::rdma {

TEST(DevFrameCaps, V2RoundTrip) {
  char frame[kDevNameBytes];
  EncodeDevFrame("ib7s400p0", 4u << 20, frame);
  EXPECT_STREQ(frame, "ib7s400p0");
  EXPECT_EQ(ParseDevFrameCaps(frame), 4u << 20);
  EXPECT_EQ(ParseDevFrameProtocol(frame), kDevProtoV2);
}

TEST(DevFrameCaps, WriterRetirementRequestDoesNotChangeBlockGeometry) {
  constexpr uint64_t max_block = 4u << 20;
  char frame[kDevNameBytes];
  EncodeDevFrame("ib0", max_block | kDevFrameRequestWriterRetirement, frame);

  EXPECT_EQ(ParseDevFrameCaps(frame),
            max_block | kDevFrameRequestWriterRetirement);
  EXPECT_EQ(ParseDevFrameMaxBlock(frame), max_block);
  EXPECT_TRUE(DevFrameRequestsWriterRetirement(frame));
  EXPECT_EQ(ParseDevFrameProtocol(frame), kDevProtoV2);

  EncodeDevFrame("ib0", max_block, frame);
  EXPECT_EQ(ParseDevFrameMaxBlock(frame), max_block);
  EXPECT_FALSE(DevFrameRequestsWriterRetirement(frame));
}

TEST(DevFrameCaps, RawOpaqueControlValueKeepsBit63) {
  constexpr uint64_t token = kDevFrameRequestWriterRetirement | 0x1234;
  char frame[kDevNameBytes];
  EncodeDevFrame("__control__", token, frame);
  EXPECT_EQ(ParseDevFrameCaps(frame), token);
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

TEST(DevFrameCaps, DeviceNameFitBoundaryIsExported) {
  EXPECT_EQ(kMaxDeviceNameBytes, 18u);  // 32 - (1 NUL + 4 magic + 8 u64 + 1 u8)
  EXPECT_TRUE(DeviceNameFitsFrame(""));  // empty = peer default device
  EXPECT_TRUE(DeviceNameFitsFrame(std::string(18, 'a')));
  EXPECT_FALSE(DeviceNameFitsFrame(std::string(19, 'a')));
  EXPECT_FALSE(DeviceNameFitsFrame(std::string(64, 'a')));  // udev/bond names
}

TEST(DevFrameCaps, OversizedDiagnosticNamesDeviceAndLimit) {
  const std::string dev(19, 'x');
  const std::string error = OversizedDeviceNameError(dev);
  EXPECT_NE(error.find(dev), std::string::npos);
  EXPECT_NE(error.find(std::to_string(kMaxDeviceNameBytes)), std::string::npos);
}

// The deep-defense drop stays reachable for frames not built from a validated
// name, mirroring the historical silent-drop behavior the fit check prevents.
TEST(DevFrameCaps, FitCheckMirrorsEncodeBoundary) {
  char frame[kDevNameBytes];
  for (size_t len = 0; len <= kDevNameBytes; ++len) {
    const std::string dev(len, 'q');
    EncodeDevFrame(dev, 4u << 20, frame);
    if (DeviceNameFitsFrame(dev)) {
      EXPECT_EQ(ParseDevFrameProtocol(frame), kDevProtoV2) << "len=" << len;
    } else {
      EXPECT_EQ(ParseDevFrameProtocol(frame), 0) << "len=" << len;
    }
  }
}

#ifdef DFKV_WITH_RDMA

TEST(DevFrameCaps, OversizedConfiguredDeviceFailsClientConstruction) {
  // The whitelist check runs before any device discovery, so this is hermetic
  // on hosts with no RDMA hardware. 19 chars overflows the frame; 64 chars
  // matches a long custom udev/bond HCA name.
  for (size_t len : {19u, 64u}) {
    const std::string dev(len, 'x');
    try {
      dfkv::RdmaTransport transport(0, dev);
      FAIL() << "construction accepted a " << len << "-byte device name";
    } catch (const std::runtime_error& e) {
      const std::string what = e.what();
      EXPECT_NE(what.find(dev), std::string::npos);
      EXPECT_NE(what.find(std::to_string(kMaxDeviceNameBytes)),
                std::string::npos);
    }
  }
}

TEST(DevFrameCaps, BoundaryConfiguredDevicePassesNameValidation) {
  // 18 chars exactly fills the frame budget: construction must clear name
  // validation and fail only later at discovery (no hardware here, or no
  // ACTIVE device by that literal name) with no oversized-name complaint.
  const std::string dev(kMaxDeviceNameBytes, 'y');
  try {
    dfkv::RdmaTransport transport(0, dev);
    GTEST_SKIP() << "host has an ACTIVE device matching the test name";
  } catch (const std::runtime_error& e) {
    const std::string what = e.what();
    EXPECT_NE(what.find("ACTIVE"), std::string::npos);
    EXPECT_EQ(what.find(std::to_string(kMaxDeviceNameBytes)),
              std::string::npos);
  }
}

TEST(DevFrameCaps, OversizedAnchorWarnsButServerStillStarts) {
  // The server never encodes its anchor into a frame (empty-name clients fall
  // back to the default anchor), so startup diagnoses the name but continues.
  const std::string dev(20, 'a');
  dfkv::RdmaServer server(
      [](uint8_t, const dfkv::BlockKey&, uint64_t, uint64_t, const char*,
         uint64_t, std::string*, size_t*) { return dfkv::Status::kNotFound; },
      1u << 20, dev);
  testing::internal::CaptureStderr();
  const dfkv::Status status = server.Start(0);
  const std::string logs = testing::internal::GetCapturedStderr();
  EXPECT_NE(logs.find(dev), std::string::npos)
      << "startup diagnostics must name the oversized anchor";
  EXPECT_NE(logs.find(std::to_string(kMaxDeviceNameBytes)), std::string::npos)
      << "startup diagnostics must state the frame limit";
  EXPECT_NE(logs.find("serving anyway"), std::string::npos);
  // Startup continued past the warning: device discovery (not name
  // validation) reports availability. No ACTIVE device carries the test name.
  EXPECT_NE(logs.find("ACTIVE"), std::string::npos);
  EXPECT_EQ(status, dfkv::Status::kIOError);
}

#endif  // DFKV_WITH_RDMA

}  // namespace dfkv::rdma
