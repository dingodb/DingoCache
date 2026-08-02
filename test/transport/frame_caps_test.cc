// Frame-cap regressions: every socket that reads a declared length must bound
// it BEFORE allocating (utils/wire_limits.h). A forged prefix declaring a
// huge payload/body must drop the connection promptly -- not allocate.
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <string>
#include <thread>

#include "cache/kv_node_server.h"
#include "mds/mds_server.h"
#include "mds/mds_member_poller.h"
#include "transport/tcp_transport.h"
#include "transport/wire.h"
#include "utils/net_util.h"

using namespace dfkv;  // NOLINT
namespace fs = std::filesystem;

namespace {

constexpr uint64_t kHuge = 1ull << 33;  // 8 GiB: legal under the old 16 GiB default

// Sends a request prefix declaring `payload_len` bytes (without sending them)
// and returns true iff the server closed the connection promptly (recv == 0
// within the socket timeout) -- the post-fix behavior. Pre-fix the server
// would allocate and then block waiting for the payload.
bool ServerDropsOversizeFrame(int port, uint64_t payload_len) {
  int fd = net::Dial("127.0.0.1:" + std::to_string(port), 2000, 2000);
  if (fd < 0) return false;
  char prefix[kReqPrefix];
  EncodeReq(prefix, WireOp::kCache, BlockKey{1, 2}, 0, 0, payload_len);
  if (!net::WriteAll(fd, prefix, kReqPrefix)) { ::close(fd); return false; }
  char c;
  ssize_t r = ::recv(fd, &c, 1, 0);  // 0 = orderly close, <0 = timeout/err
  ::close(fd);
  return r == 0;
}

// One-shot fake server: accepts a single connection, consumes the request
// prefix (+payload), replies with a response prefix declaring `resp_dlen`
// body bytes, then closes without sending a body.
class FakeDlenServer {
 public:
  explicit FakeDlenServer(uint64_t resp_dlen) : dlen_(resp_dlen) {
    lfd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ::bind(lfd_, reinterpret_cast<sockaddr*>(&sa), sizeof(sa));
    ::listen(lfd_, 4);
    socklen_t sl = sizeof(sa);
    ::getsockname(lfd_, reinterpret_cast<sockaddr*>(&sa), &sl);
    port_ = ntohs(sa.sin_port);
    th_ = std::thread([this] {
      for (;;) {
        int fd = ::accept(lfd_, nullptr, nullptr);
        if (fd < 0) return;  // listener closed
        char prefix[kReqPrefix];
        if (net::ReadAll(fd, prefix, kReqPrefix)) {
          ReqFields rq;
          if (DecodeReq(prefix, &rq) && rq.payload_len) {
            std::string sink(rq.payload_len, '\0');
            net::ReadAll(fd, &sink[0], rq.payload_len);
          }
          char rp[kRespPrefix];
          EncodeResp(rp, Status::kOk, dlen_);
          net::WriteAll(fd, rp, kRespPrefix);
        }
        ::close(fd);
      }
    });
  }
  ~FakeDlenServer() {
    ::shutdown(lfd_, SHUT_RDWR);
    ::close(lfd_);
    if (th_.joinable()) th_.join();
  }
  int port() const { return port_; }

 private:
  uint64_t dlen_;
  int lfd_ = -1, port_ = 0;
  std::thread th_;
};

}  // namespace

TEST(FrameCaps, KvNodeServerDropsOversizeRequestAndStaysAlive) {
  fs::path dir = fs::temp_directory_path() / "dfkv_framecap_node";
  fs::remove_all(dir);
  fs::create_directories(dir);
  KvNodeServer srv(dir.string(), 1ull << 30);
  ASSERT_EQ(srv.Start(0), Status::kOk);

  EXPECT_TRUE(ServerDropsOversizeFrame(srv.port(), kHuge));

  // Server must still serve a well-formed request afterwards.
  TcpTransport t;
  std::string node = "127.0.0.1:" + std::to_string(srv.port());
  std::string blob(1024, 'x');
  EXPECT_EQ(t.Cache(node, BlockKey{7, 0}, blob.data(), blob.size()),
            Status::kOk);
  bool exist = false;
  EXPECT_EQ(t.Exist(node, BlockKey{7, 0}, &exist), Status::kOk);
  EXPECT_TRUE(exist);
  srv.Stop();
  fs::remove_all(dir);
}

TEST(FrameCaps, MdsServerDropsOversizeRequestAndStaysAlive) {
  // etcd is never contacted: the oversize frame is dropped at decode, and the
  // liveness probe below only needs the server to answer (kIOError from the
  // unreachable etcd endpoint is an ANSWER -- the connection survives).
  MdsServer mds("127.0.0.1:9");  // discard port: connection refused fast
  ASSERT_EQ(mds.Start(0), Status::kOk);

  EXPECT_TRUE(ServerDropsOversizeFrame(mds.port(), 2ull << 20));  // > 1 MiB cap

  int fd = net::Dial("127.0.0.1:" + std::to_string(mds.port()), 2000, 5000);
  ASSERT_GE(fd, 0);
  std::string group = "g";
  char prefix[kReqPrefix];
  EncodeReq(prefix, WireOp::kListMembers, BlockKey{}, 0, 0, group.size());
  ASSERT_TRUE(net::WriteAll(fd, prefix, kReqPrefix));
  ASSERT_TRUE(net::WriteAll(fd, group.data(), group.size()));
  char rp[kRespPrefix];
  EXPECT_TRUE(net::ReadAll(fd, rp, kRespPrefix)) << "server must still answer";
  ::close(fd);
  mds.Stop();
}

TEST(FrameCaps, ClientRejectsOversizeStatusResponse) {
  FakeDlenServer fake(kHuge);
  TcpTransport t;
  std::string node = "127.0.0.1:" + std::to_string(fake.port());
  bool exist = false;
  // Status-only op: a peer declaring an 8 GiB body is a protocol violation.
  EXPECT_EQ(t.Exist(node, BlockKey{1, 0}, &exist), Status::kIOError);
}

TEST(FrameCaps, ClientRejectsRangeResponseLargerThanAsked) {
  FakeDlenServer fake(4096 + 1000);  // more than the request asks for
  TcpTransport t;
  std::string node = "127.0.0.1:" + std::to_string(fake.port());
  std::string out;
  EXPECT_EQ(t.Range(node, BlockKey{1, 0}, 0, 4096, &out), Status::kIOError);
}

TEST(FrameCaps, PollerRejectsOversizeMemberList) {
  FakeDlenServer fake(kHuge);
  std::atomic<int> changes{0};
  MdsMemberPoller poller({"127.0.0.1:" + std::to_string(fake.port())}, "g",
                         [&](const std::vector<MemberInfo>&) { ++changes; });
  EXPECT_FALSE(poller.PollOnce());  // oversize dlen must fail the poll...
  EXPECT_EQ(changes.load(), 0);     // ...and never reach the on_change callback
}
