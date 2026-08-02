// Transport factory: TCP is selected only when RDMA is not requested; an
// unavailable requested RDMA transport fails closed.
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

TEST(Factory, ReturnsWorkingTcpWhenRdmaNotRequested) {
  EnvSave save_rdma("DFKV_RDMA");
  ::unsetenv("DFKV_RDMA");
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
  ASSERT_EQ(t->Cache(addr, ToBlockKey("test/model", "k"), v.data(), v.size()), Status::kOk);
  std::string out;
  ASSERT_EQ(t->Range(addr, ToBlockKey("test/model", "k"), 0, v.size(), &out), Status::kOk);
  EXPECT_EQ(out, v);
  srv.Stop();
}


TEST(Factory, RequestedUnavailableDeviceFailsClosed) {
  EnvSave save_rdma("DFKV_RDMA");
  EnvSave save_dev("DFKV_RDMA_DEV");
  ::setenv("DFKV_RDMA", "1", 1);
  ::setenv("DFKV_RDMA_DEV", "dfkv-no-such-rdma-device", 1);

  std::string reason;
  auto t = MakeClientTransport(&reason);
  EXPECT_EQ(t, nullptr);
#ifdef DFKV_WITH_RDMA
  EXPECT_NE(reason.find("rdma-configured-devices"), std::string::npos) << reason;
#else
  EXPECT_NE(reason.find("rdma-requested-but-not-built"), std::string::npos)
      << reason;
#endif
}
