#include "utils/net_util.h"

#include <gtest/gtest.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <unistd.h>

#include <string>
#include <vector>

using namespace dfkv;  // NOLINT

namespace {

// Loopback listener on an ephemeral port; returns {listen_fd, port}.
std::pair<int, int> Listen() {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  EXPECT_GE(fd, 0);
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  sa.sin_port = 0;
  EXPECT_EQ(::bind(fd, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)), 0);
  EXPECT_EQ(::listen(fd, 16), 0);
  socklen_t sl = sizeof(sa);
  EXPECT_EQ(::getsockname(fd, reinterpret_cast<sockaddr*>(&sa), &sl), 0);
  return {fd, ntohs(sa.sin_port)};
}

}  // namespace

TEST(NetDial, ConnectsWithTimeout) {
  auto [lfd, port] = Listen();
  int fd = net::Dial("127.0.0.1:" + std::to_string(port), 1000, 1000);
  EXPECT_GE(fd, 0);
  if (fd >= 0) ::close(fd);
  ::close(lfd);
}

TEST(NetDial, TimedConnectWorksWithFdAboveFdSetsize) {
  // Regression: the bounded-connect path used select()/FD_SET, which writes
  // past the fd_set when the socket's fd number is >= FD_SETSIZE (1024) —
  // stack corruption inside whatever host process links libdfkv (inference
  // engines routinely hold thousands of fds). Occupy the low fd space so the
  // dialed socket's fd lands above FD_SETSIZE, then exercise the timed path.
  rlimit rl{};
  ASSERT_EQ(::getrlimit(RLIMIT_NOFILE, &rl), 0);
  if (rl.rlim_cur < 1100) {
    rl.rlim_cur = std::min<rlim_t>(4096, rl.rlim_max);
    if (::setrlimit(RLIMIT_NOFILE, &rl) != 0 || rl.rlim_cur < 1100)
      GTEST_SKIP() << "cannot raise RLIMIT_NOFILE above 1100";
  }

  auto [lfd, port] = Listen();
  std::vector<int> hold;
  hold.reserve(1200);
  int dev_null = ::open("/dev/null", O_RDONLY);
  ASSERT_GE(dev_null, 0);
  hold.push_back(dev_null);
  while (hold.size() < 1100) {
    int d = ::dup(dev_null);
    ASSERT_GE(d, 0) << "dup failed at " << hold.size();
    hold.push_back(d);
  }

  int fd = net::Dial("127.0.0.1:" + std::to_string(port), 1000, 1000);
  EXPECT_GE(fd, 1024) << "test precondition: fd should land above FD_SETSIZE";
  EXPECT_GE(fd, 0) << "timed connect must work for fd >= FD_SETSIZE";
  if (fd >= 0) ::close(fd);

  for (int d : hold) ::close(d);
  ::close(lfd);
}

TEST(NetPeerAddress, NamesTheRemoteEndOfAConnectedSocket) {
  auto [lfd, port] = Listen();
  const int client = net::Dial("127.0.0.1:" + std::to_string(port), 1000, 1000);
  ASSERT_GE(client, 0);
  const int accepted = ::accept(lfd, nullptr, nullptr);
  ASSERT_GE(accepted, 0);

  EXPECT_EQ(net::PeerAddress(client), "127.0.0.1:" + std::to_string(port));
  // The accepted side sees the client's ephemeral port: what a server logs.
  sockaddr_in local{};
  socklen_t length = sizeof(local);
  ASSERT_EQ(::getsockname(client, reinterpret_cast<sockaddr*>(&local), &length),
            0);
  EXPECT_EQ(net::PeerAddress(accepted),
            "127.0.0.1:" + std::to_string(ntohs(local.sin_port)));

  ::close(accepted);
  ::close(client);
  ::close(lfd);
}

TEST(NetPeerAddress, SocketsWithoutAnIpPeerAreUnknown) {
  auto [lfd, port] = Listen();
  (void)port;
  EXPECT_EQ(net::PeerAddress(lfd), "unknown") << "listening, not connected";
  ::close(lfd);
  int pair[2] = {-1, -1};
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, pair), 0);
  EXPECT_EQ(net::PeerAddress(pair[0]), "unknown");
  ::close(pair[0]);
  ::close(pair[1]);
  EXPECT_EQ(net::PeerAddress(-1), "unknown");
}

TEST(NetPeerAddress, BracketsAnIpv6Peer) {
  const int lfd = ::socket(AF_INET6, SOCK_STREAM, 0);
  if (lfd < 0) GTEST_SKIP() << "no IPv6 sockets";
  sockaddr_in6 sa{};
  sa.sin6_family = AF_INET6;
  sa.sin6_addr = in6addr_loopback;
  if (::bind(lfd, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) != 0) {
    ::close(lfd);
    GTEST_SKIP() << "no IPv6 loopback address";
  }
  ASSERT_EQ(::listen(lfd, 1), 0);
  socklen_t length = sizeof(sa);
  ASSERT_EQ(::getsockname(lfd, reinterpret_cast<sockaddr*>(&sa), &length), 0);
  const int client = ::socket(AF_INET6, SOCK_STREAM, 0);
  ASSERT_GE(client, 0);
  ASSERT_EQ(::connect(client, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)), 0);

  EXPECT_EQ(net::PeerAddress(client),
            "[::1]:" + std::to_string(ntohs(sa.sin6_port)));

  ::close(client);
  ::close(lfd);
}
