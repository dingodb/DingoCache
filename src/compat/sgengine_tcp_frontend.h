/* Isolated SGEngine v1 TCP compatibility listener.
 *
 * This frontend accepts the legacy v1 request frame on its own port and tags
 * every data key with KeyDomain::kSgEngineV1 before entering KvNodeServer.
 * Native listeners and native request handling remain unaware of legacy key
 * derivation. Discovery/MDS operations are intentionally not exposed here:
 * returning native membership would let a legacy client escape onto native
 * ports and bypass the domain boundary.
 */
#ifndef DFKV_SGENGINE_TCP_FRONTEND_H_
#define DFKV_SGENGINE_TCP_FRONTEND_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "common/status.h"

namespace dfkv {

class KvNodeServer;

namespace compat {

class SgEngineTcpFrontend {
 public:
  explicit SgEngineTcpFrontend(KvNodeServer& server) : server_(server) {}
  ~SgEngineTcpFrontend() { Stop(); }

  SgEngineTcpFrontend(const SgEngineTcpFrontend&) = delete;
  SgEngineTcpFrontend& operator=(const SgEngineTcpFrontend&) = delete;

  Status Start(int port);
  void Stop();
  int port() const { return port_; }
  void set_max_request_payload(uint64_t bytes) {
    max_request_payload_ = bytes;
  }

  size_t AcceptCount() const {
    return accepts_.load(std::memory_order_relaxed);
  }
  size_t OpenConnections() const {
    return open_connections_.load(std::memory_order_relaxed);
  }
  size_t RequestCount() const {
    return requests_.load(std::memory_order_relaxed);
  }
  size_t RejectedOps() const {
    return rejected_ops_.load(std::memory_order_relaxed);
  }
  std::string MetricsText() const;

 private:
  struct Conn {
    std::thread thread;
    std::shared_ptr<std::atomic<bool>> done;
  };

  void AcceptLoop();
  void Handle(int fd);
  void ReapDoneLocked();

  KvNodeServer& server_;
  uint64_t max_request_payload_ = 0;
  int listen_fd_ = -1;
  int port_ = 0;
  std::atomic<bool> running_{false};
  std::atomic<size_t> accepts_{0};
  std::atomic<size_t> open_connections_{0};
  std::atomic<size_t> requests_{0};
  std::atomic<size_t> rejected_ops_{0};
  std::thread accept_thread_;
  std::mutex conn_mu_;
  std::set<int> conn_fds_;
  std::vector<Conn> conns_;
};

}  // namespace compat
}  // namespace dfkv

#endif  // DFKV_SGENGINE_TCP_FRONTEND_H_
