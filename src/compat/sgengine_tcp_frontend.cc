#include "compat/sgengine_tcp_frontend.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <string>
#include <vector>

#include "cache/kv_node_server.h"
#include "transport/wire.h"
#include "utils/net_util.h"
#include "utils/thread_name.h"
#include "utils/wire_limits.h"

namespace dfkv::compat {

Status SgEngineTcpFrontend::Start(int port) {
  if (running_.load(std::memory_order_acquire) || port < 0 || port > 65535)
    return Status::kInvalid;

  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) return Status::kIOError;
  int one = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  address.sin_port = htons(static_cast<uint16_t>(port));
  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&address),
             sizeof(address)) != 0 ||
      ::listen(listen_fd_, 128) != 0) {
    ::close(listen_fd_);
    listen_fd_ = -1;
    return Status::kIOError;
  }

  socklen_t address_len = sizeof(address);
  if (::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&address),
                    &address_len) != 0) {
    ::close(listen_fd_);
    listen_fd_ = -1;
    return Status::kIOError;
  }
  port_ = ntohs(address.sin_port);
  running_.store(true, std::memory_order_release);
  accept_thread_ = std::thread([this] {
    NameThisThread("sg-tcp-accept");
    AcceptLoop();
  });
  return Status::kOk;
}

void SgEngineTcpFrontend::Stop() {
  if (!running_.exchange(false, std::memory_order_acq_rel)) return;
  if (listen_fd_ >= 0) ::shutdown(listen_fd_, SHUT_RDWR);
  if (accept_thread_.joinable()) accept_thread_.join();
  if (listen_fd_ >= 0) {
    ::close(listen_fd_);
    listen_fd_ = -1;
  }

  std::vector<int> fds;
  std::vector<Conn> conns;
  {
    std::lock_guard<std::mutex> lock(conn_mu_);
    fds.assign(conn_fds_.begin(), conn_fds_.end());
    conns.swap(conns_);
  }
  for (int fd : fds) ::shutdown(fd, SHUT_RDWR);
  for (auto& conn : conns)
    if (conn.thread.joinable()) conn.thread.join();
}

void SgEngineTcpFrontend::ReapDoneLocked() {
  for (auto it = conns_.begin(); it != conns_.end();) {
    if (it->done->load(std::memory_order_acquire)) {
      if (it->thread.joinable()) it->thread.join();
      it = conns_.erase(it);
    } else {
      ++it;
    }
  }
}

void SgEngineTcpFrontend::AcceptLoop() {
  while (running_.load(std::memory_order_acquire)) {
    int fd = ::accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) {
      if (!running_.load(std::memory_order_acquire)) break;
      continue;
    }
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    accepts_.fetch_add(1, std::memory_order_relaxed);
    open_connections_.fetch_add(1, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lock(conn_mu_);
    if (!running_.load(std::memory_order_acquire)) {
      open_connections_.fetch_sub(1, std::memory_order_relaxed);
      ::close(fd);
      break;
    }
    ReapDoneLocked();
    conn_fds_.insert(fd);
    auto done = std::make_shared<std::atomic<bool>>(false);
    conns_.push_back(
        {std::thread([this, fd, done] {
           NameThisThread("sg-tcp-serve");
           Handle(fd);
           open_connections_.fetch_sub(1, std::memory_order_relaxed);
           {
             std::lock_guard<std::mutex> lock(conn_mu_);
             conn_fds_.erase(fd);
           }
           ::close(fd);
           done->store(true, std::memory_order_release);
         }),
         done});
  }
}

void SgEngineTcpFrontend::Handle(int fd) {
  const uint64_t max_payload = max_request_payload_
                                   ? max_request_payload_
                                   : wire_limits::MaxRequestPayload();
  while (running_.load(std::memory_order_acquire)) {
    char prefix[kReqPrefix];
    if (!net::ReadAll(fd, prefix, sizeof(prefix))) return;
    ReqFields request;
    if (!DecodeReqVersion(prefix, kSgEngineProtoV1, &request, max_payload))
      return;

    std::vector<char> payload(request.payload_len);
    if (request.payload_len &&
        !net::ReadAll(fd, payload.data(), request.payload_len))
      return;

    const WireOp op = static_cast<WireOp>(request.op);
    std::string data;
    Status status = Status::kInvalid;
    switch (op) {
      case WireOp::kCache:
      case WireOp::kRange:
      case WireOp::kExist:
      case WireOp::kRemove:
        status = server_.ProcessRequestForKey(
            request.op,
            BlockKey{request.digest_hi, request.digest_lo,
                     KeyDomain::kSgEngineV1},
            request.offset, request.length, payload.data(),
            request.payload_len, &data);
        break;
      case WireOp::kStats:
        status = server_.ProcessRequestForKey(
            request.op, BlockKey{0, 0, KeyDomain::kSgEngineV1},
            request.offset, request.length, payload.data(),
            request.payload_len, &data);
        break;
      default:
        // In particular, never return native kMembers addresses from this port.
        rejected_ops_.fetch_add(1, std::memory_order_relaxed);
        break;
    }
    requests_.fetch_add(1, std::memory_order_relaxed);

    char response[kRespPrefix];
    EncodeRespVersion(response, kSgEngineProtoV1, status, data.size());
    if (!net::WriteAll(fd, response, sizeof(response))) return;
    if (!data.empty() && !net::WriteAll(fd, data.data(), data.size())) return;
  }
}

std::string SgEngineTcpFrontend::MetricsText() const {
  std::string out;
  auto metric = [&out](const char* name, const char* type,
                       const char* help, size_t value) {
    out += "# HELP " + std::string(name) + " " + help + "\n";
    out += "# TYPE " + std::string(name) + " " + type + "\n";
    out += std::string(name) + " " + std::to_string(value) + "\n";
  };
  out += "# HELP dfkv_sgengine_tcp_info SGEngine compatibility TCP contract\n";
  out += "# TYPE dfkv_sgengine_tcp_info gauge\n";
  out += "dfkv_sgengine_tcp_info{wire=\"v1\",key_domain=\"sgengine-v1\"} 1\n";
  metric("dfkv_sgengine_tcp_accepts_total", "counter",
         "Connections accepted by the SGEngine compatibility TCP port",
         AcceptCount());
  metric("dfkv_sgengine_tcp_open_connections", "gauge",
         "Open SGEngine compatibility TCP connections", OpenConnections());
  metric("dfkv_sgengine_tcp_requests_total", "counter",
         "Requests served by the SGEngine compatibility TCP port",
         RequestCount());
  metric("dfkv_sgengine_tcp_rejected_ops_total", "counter",
         "Discovery or control operations rejected at the compatibility boundary",
         RejectedOps());
  return out;
}

}  // namespace dfkv::compat
