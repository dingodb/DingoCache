// Dual-protocol MDS acceptance: with DFKV_MDS_ACCEPT_LEGACY=1 the MDS serves
// v1.x control-plane peers using the legacy 42/10-byte framing, while native
// v2 framing and the fail-closed epoch gate are unchanged.
//
// Wire-level expectations run against a dead etcd address (List ops surface
// kIOError fast on ECONNREFUSED), so only the framing is exercised. The one
// functional cross-protocol round-trip is gated on DFKV_TEST_ETCD like the
// rest of the MDS suite.
#include "mds/mds_server.h"
#include "mds/mds_proto.h"
#include "common/membership.h"
#include "transport/wire.h"
#include "transport/wire_legacy.h"
#include "utils/net_util.h"
#include <gtest/gtest.h>
#include <unistd.h>
#include <cstdlib>
#include <string>
#include <vector>
using namespace dfkv;  // NOLINT

namespace {
const char* EtcdEp() { return std::getenv("DFKV_TEST_ETCD"); }
constexpr const char* kDeadEtcd = "127.0.0.1:59999";  // nothing listens: instant ECONNREFUSED
constexpr uint64_t kTestMaxResp = 64ull << 20;

// Legacy v1.x status byte order (v1.37..v1.40.x status.h — no kQuotaExceeded).
constexpr uint8_t kLegacyOk = 0;
constexpr uint8_t kLegacyIOError = 3;

void EnableLegacy() { ::setenv("DFKV_MDS_ACCEPT_LEGACY", "1", 1); }
void DisableLegacy() { ::unsetenv("DFKV_MDS_ACCEPT_LEGACY"); }

bool DoLegacyReq(int port, WireOp op, const std::string& payload, uint8_t* ver,
                 uint8_t* status, std::string* data) {
  int fd = net::Dial("127.0.0.1:" + std::to_string(port), 2000, 5000);
  if (fd < 0) return false;
  char pre[kLegacyReqPrefix];
  EncodeLegacyReq(pre, op, payload.size());
  bool ok = net::WriteAll(fd, pre, kLegacyReqPrefix) &&
            (payload.empty() || net::WriteAll(fd, payload.data(), payload.size()));
  if (ok) {
    char rp[kLegacyRespPrefix];
    if (!net::ReadAll(fd, rp, kLegacyRespPrefix)) {
      ok = false;
    } else {
      *ver = static_cast<uint8_t>(rp[0]);
      *status = static_cast<uint8_t>(rp[1]);
      const uint64_t dlen = net::GetU64(rp + 2);
      if (dlen > kTestMaxResp) {
        ok = false;
      } else {
        data->resize(dlen);
        ok = dlen == 0 || net::ReadAll(fd, &(*data)[0], dlen);
      }
    }
  }
  ::close(fd);
  return ok;
}

bool DoNativeReq(int port, WireOp op, const std::string& payload, Status* st,
                 std::string* data) {
  int fd = net::Dial("127.0.0.1:" + std::to_string(port), 2000, 5000);
  if (fd < 0) return false;
  char pre[kReqPrefix];
  EncodeReq(pre, op, BlockKey{}, 0, 0, payload.size());
  bool ok = net::WriteAll(fd, pre, kReqPrefix) &&
            (payload.empty() || net::WriteAll(fd, payload.data(), payload.size()));
  if (ok) {
    char rp[kRespPrefix];
    if (!net::ReadAll(fd, rp, kRespPrefix)) {
      ok = false;
    } else {
      uint64_t dlen = 0;
      if (!DecodeResp(rp, st, &dlen, kTestMaxResp)) {
        ok = false;
      } else {
        data->resize(dlen);
        ok = dlen == 0 || net::ReadAll(fd, &(*data)[0], dlen);
      }
    }
  }
  ::close(fd);
  return ok;
}
}  // namespace

TEST(MdsDualProto, LegacyFrameGetsLegacyResponseWhenEnabled) {
  EnableLegacy();
  MdsServer mds(kDeadEtcd);
  ASSERT_EQ(mds.Start(0), Status::kOk);

  uint8_t ver = 0, status = 0;
  std::string data;
  // ListGroups hits etcd, fails, and answers with a legacy-encoded kIOError.
  ASSERT_TRUE(DoLegacyReq(mds.port(), WireOp::kListGroups, "", &ver, &status, &data));
  EXPECT_EQ(ver, kNativeProtoLegacyTcp);
  EXPECT_EQ(status, kLegacyIOError);
  EXPECT_TRUE(data.empty());
  // The legacy frame counter is the migration-drain observable.
  EXPECT_NE(mds.MetricsText().find("dfkv_mds_legacy_frames_total 1"),
            std::string::npos);
  mds.Stop();
}

TEST(MdsDualProto, NativeFrameStillGetsNativeResponse) {
  EnableLegacy();
  MdsServer mds(kDeadEtcd);
  ASSERT_EQ(mds.Start(0), Status::kOk);

  Status st = Status::kOk;
  std::string data;
  ASSERT_TRUE(DoNativeReq(mds.port(), WireOp::kListGroups, "", &st, &data));
  EXPECT_EQ(st, Status::kIOError);
  mds.Stop();
}

TEST(MdsDualProto, LegacyFrameRejectedWhenDisabled) {
  DisableLegacy();
  MdsServer mds(kDeadEtcd);
  ASSERT_EQ(mds.Start(0), Status::kOk);

  uint8_t ver = 0, status = 0;
  std::string data;
  // Historic strict behavior: no reply, connection dropped.
  EXPECT_FALSE(DoLegacyReq(mds.port(), WireOp::kListGroups, "", &ver, &status, &data));
  mds.Stop();
}

TEST(MdsDualProto, UnknownEpochClosesConnection) {
  EnableLegacy();
  MdsServer mds(kDeadEtcd);
  ASSERT_EQ(mds.Start(0), Status::kOk);

  int fd = net::Dial("127.0.0.1:" + std::to_string(mds.port()), 2000, 5000);
  ASSERT_GE(fd, 0);
  char garbage = static_cast<char>(0x09);
  ASSERT_TRUE(net::WriteAll(fd, &garbage, 1));
  char x = 0;
  EXPECT_FALSE(net::ReadAll(fd, &x, 1));
  ::close(fd);
  mds.Stop();
}

// The money test: a member registered over the legacy protocol must be
// visible to a native-protocol client, and vice versa — op dispatch, payload
// codec, etcd schema and lease contract are generation-shared.
TEST(MdsDualProto, LegacyRegisterListCrossProtocolRoundTrip) {
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD=host:port";
  EnableLegacy();
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  const int port = mds.port();
  const std::string group = "itest-dual-" + std::to_string(port);

  const MemberInfo legacy_node{"legacy-n1", "10.1.1.1", 28000, 1};
  const MemberInfo native_node{"native-n2", "10.1.1.2", 28000, 3};

  uint8_t ver = 0, status = 0;
  std::string data;
  ASSERT_TRUE(DoLegacyReq(port, WireOp::kRegister,
                          EncodeMemberReq(group, legacy_node), &ver, &status, &data));
  EXPECT_EQ(ver, kNativeProtoLegacyTcp);
  EXPECT_EQ(status, kLegacyOk);
  ASSERT_TRUE(DoLegacyReq(port, WireOp::kHeartbeat,
                          EncodeMemberReq(group, legacy_node), &ver, &status, &data));
  EXPECT_EQ(status, kLegacyOk);

  Status st = Status::kOk;
  ASSERT_TRUE(DoNativeReq(port, WireOp::kRegister,
                          EncodeMemberReq(group, native_node), &st, &data));
  EXPECT_EQ(st, Status::kOk);

  // Native list sees both generations with placement intact.
  ASSERT_TRUE(DoNativeReq(port, WireOp::kListMembers, group, &st, &data));
  ASSERT_EQ(st, Status::kOk);
  std::vector<MemberInfo> got;
  uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(data.data(), data.size(), &got, &epoch));
  EXPECT_GT(epoch, 0u);
  ASSERT_EQ(got.size(), 2u);
  bool saw_legacy = false, saw_native = false;
  for (const auto& m : got) {
    if (m == legacy_node) saw_legacy = true;
    if (m == native_node) saw_native = true;
  }
  EXPECT_TRUE(saw_legacy);
  EXPECT_TRUE(saw_native);

  // Legacy list sees the same ring in the legacy framing.
  ASSERT_TRUE(DoLegacyReq(port, WireOp::kListMembers, group, &ver, &status, &data));
  ASSERT_EQ(ver, kNativeProtoLegacyTcp);
  ASSERT_EQ(status, kLegacyOk);
  ASSERT_TRUE(DecodeMembers(data.data(), data.size(), &got, &epoch));
  EXPECT_EQ(got.size(), 2u);
  mds.Stop();
}
