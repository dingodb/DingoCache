// TDD R13 — transport factory: always returns a working transport (TCP fallback
// when RDMA is not built / not requested / no device).
#include "transport/transport_factory.h"
#include "cache/kv_node_server.h"
#include "client/key_map.h"

#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {
struct EnvSave {
  explicit EnvSave(const char* n) : name(n) {
    const char* v = std::getenv(name);
    if (v) { had = true; old = v; }
  }
  ~EnvSave() {
    if (had) ::setenv(name, old.c_str(), 1);
    else ::unsetenv(name);
  }
  const char* name;
  bool had = false;
  std::string old;
};
}  // namespace

TEST(Factory, ReturnsWorkingTransportWithTcpFallback) {
  auto dir = fs::temp_directory_path() / "dfkv_factory";
  fs::remove_all(dir); fs::create_directories(dir);
  KvNodeServer srv(dir.string(), 1ull << 30);
  ASSERT_EQ(srv.Start(0), Status::kOk);
  std::string addr = "127.0.0.1:" + std::to_string(srv.port());

  std::string reason;
  auto t = MakeClientTransport(&reason);
  ASSERT_NE(t, nullptr);
  EXPECT_NE(reason.find("tcp"), std::string::npos) << reason;  // no DFKV_RDMA env here

  std::string v(128, 'f');
  ASSERT_EQ(t->Cache(addr, ToBlockKey("k"), v.data(), v.size()), Status::kOk);
  std::string out;
  ASSERT_EQ(t->Range(addr, ToBlockKey("k"), 0, v.size(), &out), Status::kOk);
  EXPECT_EQ(out, v);
  srv.Stop();
}

TEST(Factory, RequireRdmaRejectsImplicitTcpFallback) {
  EnvSave save_require("DFKV_REQUIRE_RDMA");
  EnvSave save_rdma("DFKV_RDMA");
  ::setenv("DFKV_REQUIRE_RDMA", "1", 1);
  ::unsetenv("DFKV_RDMA");

  std::string reason;
  auto t = MakeClientTransport(&reason);
  EXPECT_EQ(t, nullptr);
  EXPECT_NE(reason.find("rdma-required"), std::string::npos) << reason;
}

TEST(Factory, ExplicitDeviceFilterNeverFallsBackToTcp) {
  EnvSave save_require("DFKV_REQUIRE_RDMA");
  EnvSave save_rdma("DFKV_RDMA");
  EnvSave save_dev("DFKV_RDMA_DEV");
  ::unsetenv("DFKV_REQUIRE_RDMA");
  ::setenv("DFKV_RDMA", "1", 1);
  ::setenv("DFKV_RDMA_DEV", "dfkv-no-such-rdma-device", 1);

  std::string reason;
  auto t = MakeClientTransport(&reason);
  EXPECT_EQ(t, nullptr);
  EXPECT_NE(reason.find("rdma-configured-devices"), std::string::npos)
      << reason;
}
