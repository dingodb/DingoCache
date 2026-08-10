#include "cache/kv_node_server.h"
#include "client/key_map.h"
#include "transport/tcp_transport.h"
#include "utils/net_util.h"

#include <gtest/gtest.h>

#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <thread>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) {
      had_old_ = true;
      old_ = old;
    }
    ::setenv(name, value, 1);
  }
  ~ScopedEnv() {
    if (had_old_)
      ::setenv(name_.c_str(), old_.c_str(), 1);
    else
      ::unsetenv(name_.c_str());
  }

 private:
  std::string name_;
  std::string old_;
  bool had_old_ = false;
};

std::unique_ptr<KvNodeServer> StartServer(const char* suffix) {
  const fs::path dir = fs::temp_directory_path() /
      (std::string("dfkv_tcp_listener_") + suffix + "_" +
       std::to_string(::getpid()));
  fs::remove_all(dir);
  fs::create_directories(dir);
  auto server = std::make_unique<KvNodeServer>(dir.string(), 1ull << 20);
  EXPECT_EQ(server->Start(0), Status::kOk);
  return server;
}

bool WaitFor(const std::function<bool()>& predicate, int timeout_ms) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return predicate();
}

int Dial(const KvNodeServer& server) {
  return net::Dial("127.0.0.1:" + std::to_string(server.port()), 1000, 1000);
}
}  // namespace

TEST(KvNodeServerConfig, WriteBackDefaultsToRamAckAndDiskCanOverride) {
  ScopedEnv engine("DFKV_STORE_ENGINE", "file");
  ScopedEnv ram_tier("DFKV_RAM_TIER", "1");
  ScopedEnv ram_bytes("DFKV_RAM_TIER_BYTES", "1048576");
  ScopedEnv extent_bytes("DFKV_RAM_TIER_EXTENT_BYTES", "1048576");
  ScopedEnv write_mode("DFKV_RAM_WRITE_MODE", "writeback");

  {
    ScopedEnv ack_mode("DFKV_PUT_ACK_MODE", "");
    auto server = StartServer("ram_ack_default");
    EXPECT_TRUE(server->ram_ack_enabled());
    server->Stop();
  }
  {
    ScopedEnv ack_mode("DFKV_PUT_ACK_MODE", "disk");
    auto server = StartServer("disk_ack_override");
    EXPECT_FALSE(server->ram_ack_enabled());
    server->Stop();
  }
}

TEST(KvNodeServerConfig, WriteAroundDoesNotDefaultToRamAck) {
  ScopedEnv engine("DFKV_STORE_ENGINE", "file");
  ScopedEnv ram_tier("DFKV_RAM_TIER", "1");
  ScopedEnv ram_bytes("DFKV_RAM_TIER_BYTES", "1048576");
  ScopedEnv extent_bytes("DFKV_RAM_TIER_EXTENT_BYTES", "1048576");
  ScopedEnv write_mode("DFKV_RAM_WRITE_MODE", "writearound");
  ScopedEnv ack_mode("DFKV_PUT_ACK_MODE", "");

  auto server = StartServer("writearound_ack_default");
  EXPECT_FALSE(server->ram_ack_enabled());
  server->Stop();
}

TEST(KvNodeServerListener, SaturationRejectsBeyondHandlerLimit) {
  ScopedEnv engine("DFKV_STORE_ENGINE", "file");
  ScopedEnv max_connections("DFKV_TCP_MAX_CONNS", "2");
  ScopedEnv io_timeout("DFKV_TCP_IO_TIMEOUT_S", "10");
  auto server = StartServer("saturation");

  int first = Dial(*server);
  int second = Dial(*server);
  ASSERT_GE(first, 0);
  ASSERT_GE(second, 0);
  ASSERT_TRUE(WaitFor([&] { return server->live_conn_count() == 2; }, 1000));

  int rejected = Dial(*server);
  ASSERT_GE(rejected, 0);
  ASSERT_TRUE(WaitFor(
      [&] { return server->TcpRejectedConnections() == 1; }, 1000));
  EXPECT_EQ(server->live_conn_count(), 2u);

  timeval timeout{1, 0};
  ::setsockopt(rejected, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  char byte = 0;
  EXPECT_LE(::recv(rejected, &byte, 1, 0), 0);
  ::close(rejected);
  ::close(first);
  ::close(second);
  server->Stop();
}

TEST(KvNodeServerListener, SilentPeerIsReleasedBySocketTimeout) {
  ScopedEnv engine("DFKV_STORE_ENGINE", "file");
  ScopedEnv max_connections("DFKV_TCP_MAX_CONNS", "4");
  ScopedEnv io_timeout("DFKV_TCP_IO_TIMEOUT_S", "1");
  auto server = StartServer("silent");

  int fd = Dial(*server);
  ASSERT_GE(fd, 0);
  ASSERT_TRUE(WaitFor([&] { return server->live_conn_count() == 1; }, 1000));
  EXPECT_TRUE(WaitFor([&] { return server->live_conn_count() == 0; }, 3000));
  ::close(fd);
  server->Stop();
}

TEST(KvNodeServerListener, ShutdownInterruptsSilentHandlersPromptly) {
  ScopedEnv engine("DFKV_STORE_ENGINE", "file");
  ScopedEnv max_connections("DFKV_TCP_MAX_CONNS", "4");
  ScopedEnv io_timeout("DFKV_TCP_IO_TIMEOUT_S", "60");
  auto server = StartServer("shutdown");

  int fd = Dial(*server);
  ASSERT_GE(fd, 0);
  ASSERT_TRUE(WaitFor([&] { return server->live_conn_count() == 1; }, 1000));
  const auto start = std::chrono::steady_clock::now();
  server->Stop();
  const auto elapsed = std::chrono::steady_clock::now() - start;
  EXPECT_LT(elapsed, std::chrono::seconds(2));
  ::close(fd);
}

TEST(KvNodeServerListener, NormalPooledTcpRemainsFunctionalAtLimit) {
  ScopedEnv engine("DFKV_STORE_ENGINE", "file");
  ScopedEnv max_connections("DFKV_TCP_MAX_CONNS", "1");
  ScopedEnv io_timeout("DFKV_TCP_IO_TIMEOUT_S", "5");
  auto server = StartServer("pooled");
  const std::string endpoint =
      "127.0.0.1:" + std::to_string(server->port());
  TcpTransport transport;
  const BlockKey key = ToBlockKey("listener/model", "pooled-key");
  const std::string value = "normal pooled payload";

  ASSERT_EQ(transport.Cache(endpoint, key, value.data(), value.size()),
            Status::kOk);
  bool exists = false;
  ASSERT_EQ(transport.Exist(endpoint, key, &exists), Status::kOk);
  EXPECT_TRUE(exists);
  std::string got;
  ASSERT_EQ(transport.Range(endpoint, key, 0, value.size(), &got), Status::kOk);
  EXPECT_EQ(got, value);
  EXPECT_EQ(server->TcpRejectedConnections(), 0u);
  EXPECT_EQ(server->AcceptCount(), 1u);
  server->Stop();
}

// A kRange whose declared range length exceeds the payload ceiling can never
// name a real value: the TCP frontend drops the connection at the same gate
// as an oversized payload (the RDMA frontend already rejects this shape at
// its decode/serve gates). Well-formed frames on the same server are
// unaffected. Regression guard for the hostile-length coalesced-read finding.
TEST(KvNodeServerListener, RangeLengthBeyondPayloadBoundDropsConnection) {
  ScopedEnv engine("DFKV_STORE_ENGINE", "file");
  auto server = StartServer("range_len");
  server->set_max_request_payload(4096);

  {
    const int fd = Dial(*server);
    ASSERT_GE(fd, 0);
    char pre[kReqPrefix];
    EncodeReq(pre, WireOp::kRange, BlockKey{1, 2, 3}, 0, 8192, 0);
    ASSERT_TRUE(net::WriteAll(fd, pre, sizeof(pre)));
    char rp[kRespPrefix];
    EXPECT_FALSE(net::ReadAll(fd, rp, sizeof(rp)));
    ::close(fd);
  }
  {
    const int fd = Dial(*server);
    ASSERT_GE(fd, 0);
    const std::string value = "ok";
    char pre[kReqPrefix];
    EncodeReq(pre, WireOp::kCache, BlockKey{1, 2, 3}, 0, 0, value.size());
    ASSERT_TRUE(net::WriteAll(fd, pre, sizeof(pre)));
    ASSERT_TRUE(net::WriteAll(fd, value.data(), value.size()));
    char rp[kRespPrefix];
    ASSERT_TRUE(net::ReadAll(fd, rp, sizeof(rp)));
    Status st = Status::kIOError;
    uint64_t data_len = 0;
    ASSERT_TRUE(DecodeResp(rp, &st, &data_len));
    EXPECT_EQ(st, Status::kOk);
    ::close(fd);
  }
  server->Stop();
}

// SO_RCVTIMEO is a per-SYSCALL budget: a drip feeder sending one byte per
// (timeout - epsilon) never timed out and pinned a connection slot forever.
// DFKV_TCP_FIRST_REQ_MS gives the FIRST complete frame an absolute deadline
// anchored at accept, so the drip connection is closed while the per-syscall
// base timeout alone would never have fired (drip interval < base).
TEST(KvNodeServerListener, SlowDripFirstFrameIsClosedByAbsoluteDeadline) {
  ScopedEnv engine("DFKV_STORE_ENGINE", "file");
  ScopedEnv max_connections("DFKV_TCP_MAX_CONNS", "4");
  ScopedEnv io_timeout("DFKV_TCP_IO_TIMEOUT_S", "30");  // never fires under drip
  ScopedEnv first_req("DFKV_TCP_FIRST_REQ_MS", "300");
  auto server = StartServer("drip");

  const int fd = Dial(*server);
  ASSERT_GE(fd, 0);
  ASSERT_TRUE(WaitFor([&] { return server->live_conn_count() == 1; }, 1000));
  // Drip one partial-prefix byte every 100ms: each byte arrives far inside the
  // 30s base budget, so only the 300ms absolute deadline can close this.
  const char junk[16] = {0};
  ::send(fd, junk, 1, MSG_NOSIGNAL);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  ::send(fd, junk + 1, 1, MSG_NOSIGNAL);
  EXPECT_TRUE(WaitFor([&] { return server->live_conn_count() == 0; }, 3000))
      << "drip connection outlived its first-frame deadline";
  ::close(fd);
  server->Stop();
}

// The deadline covers only the FIRST frame on a connection: a client that
// completes a request promptly, then goes quiet, must keep the connection for
// normal keep-alive reuse (base per-syscall timeout semantics unchanged), and
// the restored per-syscall budget must still serve later requests.
TEST(KvNodeServerListener, FirstFrameDeadlineLeavesKeepAliveTrafficAlone) {
  ScopedEnv engine("DFKV_STORE_ENGINE", "file");
  ScopedEnv max_connections("DFKV_TCP_MAX_CONNS", "4");
  ScopedEnv io_timeout("DFKV_TCP_IO_TIMEOUT_S", "10");
  ScopedEnv first_req("DFKV_TCP_FIRST_REQ_MS", "300");
  auto server = StartServer("drip_ok");
  const std::string endpoint = "127.0.0.1:" + std::to_string(server->port());
  TcpTransport transport;
  const BlockKey key = ToBlockKey("listener/model", "drip-ok-key");
  const std::string value = "normal payload after deadline knob";

  // First frame completes well within the 300ms deadline.
  ASSERT_EQ(transport.Cache(endpoint, key, value.data(), value.size()),
            Status::kOk);
  // Idle LONGER than the first-frame deadline: the connection must NOT be
  // closed — the deadline stopped applying once the first frame completed.
  std::this_thread::sleep_for(std::chrono::milliseconds(600));
  bool exists = false;
  ASSERT_EQ(transport.Exist(endpoint, key, &exists), Status::kOk);
  EXPECT_TRUE(exists);
  std::string got;
  ASSERT_EQ(transport.Range(endpoint, key, 0, value.size(), &got), Status::kOk);
  EXPECT_EQ(got, value);
  // One accept for the whole session: the post-deadline requests were served
  // on the ORIGINAL keep-alive connection, not a transparent redial.
  EXPECT_EQ(server->AcceptCount(), 1u);
  server->Stop();
}

// Knob off (0) restores the legacy behavior exactly: no absolute deadline, a
// drip connection lives until the per-syscall base timeout fires.
TEST(KvNodeServerListener, DisabledFirstFrameDeadlineKeepsLegacyBehavior) {
  ScopedEnv engine("DFKV_STORE_ENGINE", "file");
  ScopedEnv max_connections("DFKV_TCP_MAX_CONNS", "4");
  ScopedEnv io_timeout("DFKV_TCP_IO_TIMEOUT_S", "30");  // would take 30s to fire
  ScopedEnv first_req("DFKV_TCP_FIRST_REQ_MS", "0");
  auto server = StartServer("drip_off");

  const int fd = Dial(*server);
  ASSERT_GE(fd, 0);
  ASSERT_TRUE(WaitFor([&] { return server->live_conn_count() == 1; }, 1000));
  // Drip for far longer than a 300ms-style deadline would allow; the conn must
  // still be alive because the deadline is disabled.
  const char junk[8] = {0};
  for (int i = 0; i < 6; ++i) {
    ::send(fd, junk, 1, MSG_NOSIGNAL);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  EXPECT_EQ(server->live_conn_count(), 1u)
      << "disabled deadline must not close a drip connection";
  ::close(fd);
  server->Stop();
}

TEST(KvNodeServerListener, ConfigIsHardBoundedAndExported) {
  ScopedEnv engine("DFKV_STORE_ENGINE", "file");
  ScopedEnv max_connections("DFKV_TCP_MAX_CONNS", "999999");
  ScopedEnv io_timeout("DFKV_TCP_IO_TIMEOUT_S", "999999");
  ScopedEnv first_req("DFKV_TCP_FIRST_REQ_MS", "99999999");
  auto server = StartServer("config");

  EXPECT_EQ(server->TcpMaxConnections(), 4096u);
  EXPECT_EQ(server->TcpIoTimeoutSeconds(), 3600);
  EXPECT_EQ(server->TcpFirstReqMs(), 3600000u);
  const std::string metrics = server->MetricsText();
  EXPECT_NE(metrics.find("dfkv_tcp_max_connections 4096"), std::string::npos)
      << metrics;
  EXPECT_NE(metrics.find("dfkv_tcp_io_timeout_seconds 3600"),
            std::string::npos) << metrics;
  EXPECT_NE(metrics.find("dfkv_tcp_first_req_ms 3600000"), std::string::npos)
      << metrics;
  EXPECT_NE(metrics.find("dfkv_tcp_rejected_connections_total 0"),
            std::string::npos) << metrics;
  server->Stop();
}
