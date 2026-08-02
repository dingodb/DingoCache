// TDD R10 — transport IO timeout: a node that accepts but never replies must
// cause the call to FAIL within a bounded time, not hang forever.
#include "transport/tcp_transport.h"
#include "client/key_map.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>

using namespace dfkv;  // NOLINT

// A listener that accepts connections and then does nothing (black hole).
class BlackHole {
 public:
  void Start() {
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    int one = 1; ::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in sa{}; sa.sin_family = AF_INET; sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ::bind(fd_, (sockaddr*)&sa, sizeof(sa));
    ::listen(fd_, 16);
    socklen_t sl = sizeof(sa); ::getsockname(fd_, (sockaddr*)&sa, &sl); port_ = ntohs(sa.sin_port);
    run_ = true;
    th_ = std::thread([this] {
      while (run_) { int c = ::accept(fd_, nullptr, nullptr); if (c >= 0) held_.push_back(c); }
    });
  }
  void Stop() {
    run_ = false; ::shutdown(fd_, SHUT_RDWR); ::close(fd_);
    if (th_.joinable()) th_.join();
    for (int c : held_) ::close(c);
  }
  int port() const { return port_; }
 private:
  int fd_ = -1, port_ = 0;
  std::atomic<bool> run_{false};
  std::thread th_;
  std::vector<int> held_;
};

TEST(Timeout, SilentNodeFailsWithinBoundedTime) {
  BlackHole bh; bh.Start();
  std::string node = "127.0.0.1:" + std::to_string(bh.port());
  TcpTransport t;
  t.set_timeouts(/*connect_ms=*/1000, /*io_ms=*/300);

  auto t0 = std::chrono::steady_clock::now();
  std::string out;
  Status st = t.Range(node, ToBlockKey("test/model", "k"), 0, 16, &out);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();

  EXPECT_NE(st, Status::kOk);          // did not succeed
  EXPECT_LT(ms, 2000) << "took " << ms << "ms — likely hung";  // bounded by io timeout
  bh.Stop();
}
