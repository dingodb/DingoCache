#include "mds/mds_registrar.h"
#include "mds/mds_server.h"
#include "mds/mds_proto.h"
#include "common/membership.h"
#include "transport/wire.h"
#include "utils/net_util.h"
#include <gtest/gtest.h>
#include <atomic>
#include <unistd.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <chrono>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <thread>
#include <vector>
#include <mutex>
using namespace dfkv;  // NOLINT

namespace {
const char* EtcdEp() { return std::getenv("DFKV_TEST_ETCD"); }

bool ListMembers(int port, const std::string& group, std::vector<MemberInfo>* out) {
  int fd = net::Dial("127.0.0.1:" + std::to_string(port), 2000, 2000);
  if (fd < 0) return false;
  char pre[kReqPrefix];
  EncodeReq(pre, WireOp::kListMembers, BlockKey{}, 0, 0, group.size());
  bool ok = net::WriteAll(fd, pre, kReqPrefix) &&
            net::WriteAll(fd, group.data(), group.size());
  std::string data;
  if (ok) {
    char rp[kRespPrefix]; Status st; uint64_t dlen = 0;
    ok = net::ReadAll(fd, rp, kRespPrefix) && DecodeResp(rp, &st, &dlen) && st == Status::kOk;
    if (ok) { data.resize(dlen); ok = (dlen == 0) || net::ReadAll(fd, &data[0], dlen); }
  }
  ::close(fd);
  if (!ok) return false;
  uint64_t epoch = 0;
  return DecodeMembers(data.data(), data.size(), out, &epoch);
}

bool ListClients(int port, const std::string& group, std::vector<MemberInfo>* out) {
  int fd = net::Dial("127.0.0.1:" + std::to_string(port), 2000, 2000);
  if (fd < 0) return false;
  char pre[kReqPrefix];
  EncodeReq(pre, WireOp::kListClients, BlockKey{}, 0, 0, group.size());
  bool ok = net::WriteAll(fd, pre, kReqPrefix) &&
            (!group.empty() && net::WriteAll(fd, group.data(), group.size()));
  std::string data;
  if (ok) {
    char rp[kRespPrefix]; Status st; uint64_t dlen = 0;
    ok = net::ReadAll(fd, rp, kRespPrefix) && DecodeResp(rp, &st, &dlen) && st == Status::kOk;
    if (ok) { data.resize(dlen); ok = (dlen == 0) || net::ReadAll(fd, &data[0], dlen); }
  }
  ::close(fd);
  if (!ok) return false;
  uint64_t epoch = 0;
  return DecodeMembers(data.data(), data.size(), out, &epoch);
}

bool WaitForClient(int port, const std::string& group, const MemberInfo& m, int timeout_ms) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    std::vector<MemberInfo> cs;
    if (ListClients(port, group, &cs))
      for (auto& x : cs) if (x == m) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return false;
}

bool WaitForMember(int port, const std::string& group, const MemberInfo& m, int timeout_ms) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    std::vector<MemberInfo> ms;
    if (ListMembers(port, group, &ms))
      for (auto& x : ms) if (x == m) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return false;
}

class DelayedFakeMds {
 public:
  explicit DelayedFakeMds(std::vector<int> delays_ms)
      : delays_ms_(std::move(delays_ms)) {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) throw std::runtime_error("socket failed");
    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t sl = sizeof(sa);
    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) != 0 ||
        ::listen(listen_fd_, 8) != 0 ||
        ::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&sa), &sl) != 0) {
      ::close(listen_fd_);
      listen_fd_ = -1;
      throw std::runtime_error("fake MDS listener setup failed");
    }
    port_ = ntohs(sa.sin_port);
    thread_ = std::thread([this] { Serve(); });
  }

  ~DelayedFakeMds() {
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    if (thread_.joinable()) thread_.join();
  }

  std::string endpoint() const {
    return "127.0.0.1:" + std::to_string(port_);
  }

 private:
  void Serve() {
    for (int delay_ms : delays_ms_) {
      int fd = ::accept(listen_fd_, nullptr, nullptr);
      if (fd < 0) return;
      char pre[kReqPrefix];
      ReqFields req;
      if (!net::ReadAll(fd, pre, sizeof(pre)) || !DecodeReq(pre, &req)) {
        ::close(fd);
        continue;
      }
      std::string payload(req.payload_len, '\0');
      if (req.payload_len != 0 &&
          !net::ReadAll(fd, &payload[0], payload.size())) {
        ::close(fd);
        continue;
      }
      if (delay_ms < 0) {
        ::close(fd);
        continue;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
      char response[kRespPrefix];
      EncodeResp(response, Status::kOk, 0);
      net::WriteAll(fd, response, sizeof(response));
      ::close(fd);
    }
  }

  std::vector<int> delays_ms_;
  int listen_fd_ = -1;
  int port_ = 0;
  std::thread thread_;
};
}  // namespace

TEST(MdsRegistrar, RegisterOnceMakesNodeVisible) {
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  std::string group = "m2-grp-" + std::to_string(mds.port());
  MemberInfo self{"node-x", "10.9.9.9", 28000, 2};
  MdsRegistrar reg({"127.0.0.1:" + std::to_string(mds.port())}, group, self);
  std::atomic<int> registered_callbacks{0};
  reg.set_registered_callback(
      [&] { registered_callbacks.fetch_add(1, std::memory_order_relaxed); });
  EXPECT_FALSE(reg.registered());
  ASSERT_TRUE(reg.RegisterOnce());
  EXPECT_TRUE(reg.registered());
  EXPECT_EQ(registered_callbacks.load(std::memory_order_relaxed), 1);
  ASSERT_TRUE(reg.HeartbeatOnce());
  EXPECT_EQ(registered_callbacks.load(std::memory_order_relaxed), 1);
  std::vector<MemberInfo> ms;
  ASSERT_TRUE(ListMembers(mds.port(), group, &ms));
  ASSERT_EQ(ms.size(), 1u);
  EXPECT_EQ(ms[0], self);
  mds.Stop();
}

TEST(MdsRegistrar, FailoverSkipsDeadEndpoint) {
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  std::string group = "m2-fo-" + std::to_string(mds.port());
  MemberInfo self{"node-fo", "10.9.9.10", 28000, 1};
  MdsRegistrar reg({"127.0.0.1:1", "127.0.0.1:" + std::to_string(mds.port())}, group, self);
  bool ok = false;
  for (int i = 0; i < 4 && !ok; ++i) ok = reg.RegisterOnce();
  ASSERT_TRUE(ok);
  std::vector<MemberInfo> ms;
  ASSERT_TRUE(ListMembers(mds.port(), group, &ms));
  ASSERT_EQ(ms.size(), 1u);
  EXPECT_EQ(ms[0], self);
  mds.Stop();
}

TEST(MdsRegistrar, BackgroundLoopRegistersAndKeepsAlive) {
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  std::string group = "m2-bg-" + std::to_string(mds.port());
  MemberInfo self{"node-bg", "10.9.9.11", 28000, 3};
  MdsRegistrar reg({"127.0.0.1:" + std::to_string(mds.port())}, group, self, /*hb_ms=*/200);
  reg.Start();
  EXPECT_TRUE(WaitForMember(mds.port(), group, self, /*timeout_ms=*/5000));
  reg.Stop();
  mds.Stop();
}

// ---- Client (consumer) registration via is_client=true ----------------------
// The SAME MdsRegistrar, with is_client=true, must register under /clients/ and
// keep the lease alive — inheriting the lease-TTL auto-cleanup contract for free.

TEST(MdsRegistrar, ClientRegistrarRegistersVisibleUnderClientsPrefix) {
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  std::string group = "cli-grp-" + std::to_string(mds.port());
  MemberInfo self{"conn-x", "0.0.0.0", 0, 0, 0,
                  "type=vllm,model=glm51,role=kv_producer,tp_size=8,tp_rank=0"};
  MdsRegistrar reg({"127.0.0.1:" + std::to_string(mds.port())}, group, self,
                   /*hb_ms=*/10000, /*io_ms=*/2000, /*is_client=*/true);
  ASSERT_TRUE(reg.RegisterOnce());
  ASSERT_TRUE(reg.HeartbeatOnce());

  std::vector<MemberInfo> cs;
  ASSERT_TRUE(ListClients(mds.port(), group, &cs));
  ASSERT_EQ(cs.size(), 1u);
  EXPECT_EQ(cs[0], self);
  EXPECT_EQ(cs[0].info, self.info);

  // Must NOT leak into the member ring.
  std::vector<MemberInfo> ms;
  ASSERT_TRUE(ListMembers(mds.port(), group, &ms));
  EXPECT_TRUE(ms.empty());
  mds.Stop();
}

TEST(MdsRegistrar, ClientBackgroundLoopKeepsAlive) {
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  std::string group = "cli-bg-" + std::to_string(mds.port());
  MemberInfo self{"conn-bg", "0.0.0.0", 0, 0, 0, "type=vllm"};
  MdsRegistrar reg({"127.0.0.1:" + std::to_string(mds.port())}, group, self,
                   /*hb_ms=*/200, /*io_ms=*/2000, /*is_client=*/true);
  reg.Start();
  EXPECT_TRUE(WaitForClient(mds.port(), group, self, /*timeout_ms=*/5000));
  reg.Stop();
  mds.Stop();
}

TEST(MdsRegistrar, DefaultTimeoutAllowsValidSequentialMdsBudget) {
  // A valid MDS upsert can consume three sequential 2 s etcd budgets. Delay
  // just beyond the full 6 s composed server budget; the registrar's 7 s
  // default must still accept the successful response.
  static_assert(MdsRegistrar::kDefaultIoTimeoutMs > 3 * 2000,
                "registrar timeout must exceed composed MDS etcd budget");
  DelayedFakeMds fake({6100});
  MemberInfo self{"slow-node", "10.9.9.12", 28000, 1};
  MdsRegistrar reg({fake.endpoint()}, "slow-group", self);
  const auto start = std::chrono::steady_clock::now();
  EXPECT_TRUE(reg.RegisterOnce());
  EXPECT_GE(std::chrono::steady_clock::now() - start,
            std::chrono::milliseconds(6000));
  EXPECT_TRUE(reg.registered());
}

TEST(MdsRegistrar, HeartbeatDegradationIsObservableWithoutClearingReadiness) {
  DelayedFakeMds fake({0, -1, -1, 0});
  MemberInfo self{"health-node", "10.9.9.13", 28000, 1};
  MdsRegistrar reg({fake.endpoint()}, "health-group", self);
  std::atomic<int> callbacks{0};
  reg.set_registered_callback(
      [&] { callbacks.fetch_add(1, std::memory_order_relaxed); });

  ASSERT_TRUE(reg.RegisterOnce());
  EXPECT_EQ(callbacks.load(std::memory_order_relaxed), 1);
  EXPECT_TRUE(reg.registered());
  EXPECT_FALSE(reg.HeartbeatOnce());
  EXPECT_FALSE(reg.HeartbeatOnce());
  EXPECT_TRUE(reg.registered()) << "startup readiness must remain latched";
  EXPECT_EQ(callbacks.load(std::memory_order_relaxed), 1);
  EXPECT_EQ(reg.heartbeat_failures_consecutive(), 2u);
  EXPECT_EQ(reg.heartbeat_failures_total(), 2u);
  EXPECT_FALSE(reg.heartbeat_healthy());
  EXPECT_LE(reg.last_success_age_seconds(), 1u);
  const std::string degraded = reg.MetricsText();
  EXPECT_NE(degraded.find("dfkv_mds_registration_latched{group=\"health-group\",node=\"health-node\"} 1"),
            std::string::npos) << degraded;
  EXPECT_NE(degraded.find("dfkv_mds_heartbeat_healthy{group=\"health-group\",node=\"health-node\"} 0"),
            std::string::npos) << degraded;

  EXPECT_NE(degraded.find("dfkv_mds_last_success_age_seconds{group=\"health-group\",node=\"health-node\"}"),
            std::string::npos) << degraded;
  ASSERT_TRUE(reg.HeartbeatOnce());
  EXPECT_TRUE(reg.heartbeat_healthy());
  EXPECT_EQ(reg.heartbeat_failures_consecutive(), 0u);
  EXPECT_EQ(reg.heartbeat_failures_total(), 2u);
  EXPECT_EQ(callbacks.load(std::memory_order_relaxed), 1);
}

TEST(MdsRegistrar, FirstRegistrationDeadlineStopsRetryLoop) {
  MemberInfo self{"deadline-node", "10.9.9.14", 28000, 1};
  MdsRegistrar reg({"127.0.0.1:1"}, "deadline-group", self,
                   /*heartbeat_ms=*/10000, /*io_timeout_ms=*/50,
                   /*is_client=*/false,
                   /*first_registration_timeout_ms=*/150);

  const auto started = std::chrono::steady_clock::now();
  reg.Start();
  const auto wait_limit = started + std::chrono::seconds(2);
  while (!reg.first_registration_timed_out() &&
         std::chrono::steady_clock::now() < wait_limit) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  EXPECT_TRUE(reg.first_registration_timed_out());
  EXPECT_FALSE(reg.registered());
  EXPECT_LT(std::chrono::steady_clock::now() - started,
            std::chrono::seconds(1))
      << "unreachable MDS left the initial registration loop running";
  const std::string metrics = reg.MetricsText();
  EXPECT_NE(metrics.find(
                "dfkv_mds_first_registration_timeout_ms"
                "{group=\"deadline-group\",node=\"deadline-node\"} 150"),
            std::string::npos)
      << metrics;
  EXPECT_NE(metrics.find(
                "dfkv_mds_first_registration_timed_out"
                "{group=\"deadline-group\",node=\"deadline-node\"} 1"),
            std::string::npos)
      << metrics;
  reg.Stop();
}
