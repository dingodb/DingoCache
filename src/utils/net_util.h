/* Minimal POSIX TCP + integer codec helpers for the portable test transport.
 * The codec is host-endian (raw memcpy, no byte-swap): it is correct only
 * between same-endianness peers, which holds for all current x86_64 deploys.
 * The real build uses brpc; this is only for the standalone, GPU-free harness. */
#ifndef DFKV_NET_UTIL_H_
#define DFKV_NET_UTIL_H_

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>

namespace dfkv {
namespace net {

inline bool WriteAll(int fd, const void* buf, size_t n) {
  const char* p = static_cast<const char*>(buf);
  size_t off = 0;
  while (off < n) {
    ssize_t w = ::send(fd, p + off, n - off, MSG_NOSIGNAL);
    if (w <= 0) {
      if (w < 0 && errno == EINTR) continue;
      return false;
    }
    off += static_cast<size_t>(w);
  }
  return true;
}

inline bool ReadAll(int fd, void* buf, size_t n) {
  char* p = static_cast<char*>(buf);
  size_t off = 0;
  while (off < n) {
    ssize_t r = ::recv(fd, p + off, n - off, 0);
    if (r == 0) return false;  // peer closed
    if (r < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    off += static_cast<size_t>(r);
  }
  return true;
}

// ReadAll with an ABSOLUTE deadline for the whole read. SO_RCVTIMEO alone is a
// per-SYSCALL budget: a slow-dribble peer (one byte just under each interval)
// would otherwise never time out. Here the per-syscall budget is recomputed
// before EVERY recv as min(base_timeout, deadline - now), so the read must
// complete before `deadline` (monotonic clock) no matter how the bytes arrive.
// Same return contract as ReadAll (false on close, error, or timeout/expiry).
// `base_timeout` of {0,0} means "no per-syscall cap, deadline only".
inline bool ReadAllWithin(int fd, void* buf, size_t n, timeval base_timeout,
                          std::chrono::steady_clock::time_point deadline) {
  char* p = static_cast<char*>(buf);
  size_t off = 0;
  while (off < n) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) return false;  // budget already spent => close
    const int64_t remain_us =
        std::chrono::duration_cast<std::chrono::microseconds>(deadline - now)
            .count();
    const int64_t base_us =
        static_cast<int64_t>(base_timeout.tv_sec) * 1000000 +
        base_timeout.tv_usec;
    int64_t budget_us = remain_us;
    if (base_us > 0 && base_us < budget_us) budget_us = base_us;
    timeval tv{budget_us / 1000000, budget_us % 1000000};
    if (tv.tv_sec == 0 && tv.tv_usec == 0) tv.tv_usec = 1;  // {0,0} = wait forever
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ssize_t r = ::recv(fd, p + off, n - off, 0);
    if (r == 0) return false;  // peer closed
    if (r < 0) {
      if (errno == EINTR) continue;  // budget recomputed next iteration
      return false;  // includes EAGAIN/EWOULDBLOCK from SO_RCVTIMEO expiry
    }
    off += static_cast<size_t>(r);
  }
  return true;
}

// host-endian (native byte order via memcpy); see file header for the caveat.
inline void PutU64(char* p, uint64_t v) { std::memcpy(p, &v, 8); }
inline void PutU32(char* p, uint32_t v) { std::memcpy(p, &v, 4); }
inline uint64_t GetU64(const char* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
inline uint32_t GetU32(const char* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }

// "ip:port" -> connected fd, or -1. IPv4 only (inet_pton AF_INET): an IPv6
// literal or hostname returns -1 (caller treats as a clean connect failure /
// marks the endpoint down). All current deploys are IPv4 (192.168.x); switch to
// getaddrinfo here if an IPv6/dual-stack fabric is ever introduced.
inline int Dial(const std::string& addr, int connect_ms = 0, int io_ms = 0) {
  auto pos = addr.rfind(':');
  if (pos == std::string::npos) return -1;
  std::string ip = addr.substr(0, pos);
  int port = std::atoi(addr.substr(pos + 1).c_str());
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_port = htons(static_cast<uint16_t>(port));
  if (::inet_pton(AF_INET, ip.c_str(), &sa.sin_addr) != 1) { ::close(fd); return -1; }
  int one = 1; ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

  if (connect_ms <= 0) {
    if (::connect(fd, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) != 0) { ::close(fd); return -1; }
  } else {
    // non-blocking connect bounded by connect_ms
    int fl = ::fcntl(fd, F_GETFL, 0);
    ::fcntl(fd, F_SETFL, fl | O_NONBLOCK);
    int rc = ::connect(fd, reinterpret_cast<sockaddr*>(&sa), sizeof(sa));
    if (rc != 0) {
      if (errno != EINPROGRESS) { ::close(fd); return -1; }
      // poll, not select: this header is linked into inference-engine hosts
      // that routinely hold thousands of fds, so a fresh socket's fd number
      // can exceed FD_SETSIZE (1024) — FD_SET would then write past the
      // fd_set (stack corruption in the host process, not just a failed
      // connect). poll has no fd-number limit.
      pollfd pf{fd, POLLOUT, 0};
      int pr;
      do { pr = ::poll(&pf, 1, connect_ms); } while (pr < 0 && errno == EINTR);
      if (pr <= 0) { ::close(fd); return -1; }
      int err = 0; socklen_t el = sizeof(err);
      ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &el);
      if (err != 0) { ::close(fd); return -1; }
    }
    ::fcntl(fd, F_SETFL, fl);  // back to blocking
  }

  if (io_ms > 0) {  // bound blocking recv/send so a silent peer can't hang us
    timeval tv{io_ms / 1000, (io_ms % 1000) * 1000};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
  }
  return fd;
}

}  // namespace net
}  // namespace dfkv

#endif  // DFKV_NET_UTIL_H_
