#include "cache/kv_node_server.h"
#include "compat/sgengine_tcp_frontend.h"
#include "transport/tcp_transport.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace dfkv::compat {
namespace {

class SgEngineTcpFrontendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           ("dfkv_sg_tcp_" +
            std::to_string(::testing::UnitTest::GetInstance()
                               ->current_test_info()
                               ->line()));
    fs::remove_all(dir_);
    fs::create_directories(dir_);
  }
  void TearDown() override { fs::remove_all(dir_); }

  fs::path dir_;
};

TEST_F(SgEngineTcpFrontendTest, LegacyAndNativePortsCannotAliasSameWireKey) {
  KvNodeServer server(dir_.string(), 1ull << 20);
  ASSERT_EQ(server.Start(0), Status::kOk);
  SgEngineTcpFrontend legacy(server);
  ASSERT_EQ(legacy.Start(0), Status::kOk);

  const std::string native_address =
      "127.0.0.1:" + std::to_string(server.port());
  const std::string legacy_address =
      "127.0.0.1:" + std::to_string(legacy.port());
  const BlockKey wire_key{42, (uint64_t{4096} << 32) | 7};
  const std::string native_value = "native-value";
  const std::string legacy_value = "legacy-value";

  {
    TcpTransport transport;
    ASSERT_EQ(transport.Cache(native_address, wire_key, native_value.data(),
                              native_value.size()),
              Status::kOk);
    ASSERT_EQ(transport.Cache(legacy_address, wire_key, legacy_value.data(),
                              legacy_value.size()),
              Status::kOk);

    std::string out;
    ASSERT_EQ(transport.Range(native_address, wire_key, 0,
                              native_value.size(), &out),
              Status::kOk);
    EXPECT_EQ(out, native_value);
    ASSERT_EQ(transport.Range(legacy_address, wire_key, 0,
                              legacy_value.size(), &out),
              Status::kOk);
    EXPECT_EQ(out, legacy_value);

    std::string members;
    EXPECT_EQ(transport.Members(legacy_address, &members), Status::kInvalid);
  }

  EXPECT_EQ(legacy.RequestCount(), 3u);
  EXPECT_EQ(legacy.RejectedOps(), 1u);
  const std::string metrics = legacy.MetricsText();
  EXPECT_NE(metrics.find(
                "dfkv_sgengine_tcp_info{wire=\"v1\",key_domain=\"sgengine-v1\"} 1"),
            std::string::npos);
  EXPECT_NE(metrics.find("dfkv_sgengine_tcp_requests_total 3"),
            std::string::npos);
  EXPECT_NE(metrics.find("dfkv_sgengine_tcp_rejected_ops_total 1"),
            std::string::npos);
  legacy.Stop();
  server.Stop();
}

}  // namespace
}  // namespace dfkv::compat
