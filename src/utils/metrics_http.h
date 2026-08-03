/* MetricsHttpServer — a tiny, self-contained HTTP/1.0 responder that exposes a
 * single render callback at GET /metrics (plus GET /healthz). No third-party HTTP
 * dependency: it accepts on its own port and thread and is OFF the datapath, so a
 * Prometheus scrape never touches the cache/RDMA hot path. Only the render
 * callback's relaxed-atomic reads run per scrape. Opt-in (a daemon enables it via
 * --metrics-port); when no port is given no listener exists and behavior is
 * unchanged. */
#ifndef DFKV_METRICS_HTTP_H_
#define DFKV_METRICS_HTTP_H_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "common/status.h"

namespace dfkv {

class MetricsHttpServer {
 public:
  // render() returns the Prometheus exposition text served at /metrics. It is
  // called once per scrape on the HTTP thread (never on the datapath).
  explicit MetricsHttpServer(std::function<std::string()> render)
      : render_(std::move(render)) {}
  ~MetricsHttpServer();

  // Optional /healthz predicate. When set and it returns false, /healthz
  // answers 503 (e.g. MDS reporting etcd unreachable) instead of a blanket
  // 200 "ok"; unset keeps the always-200 behavior. Called on the HTTP thread.
  void set_health_check(std::function<bool()> fn) { health_ = std::move(fn); }
  // Optional /readyz predicate — READINESS, distinct from liveness: the
  // process is alive (healthz 200) long before it can serve (arena pre-fault,
  // pool-MR anchor, RDMA listener, MDS registration: tens of seconds on big
  // arenas). Rolling upgrades must gate on THIS, not on the port being open.
  // Unset => mirrors the always-200 behavior.
  void set_ready_check(std::function<bool()> fn) { ready_ = std::move(fn); }

  // port 0 => ephemeral (query with port()). bind_addr empty => all interfaces
  // (back-compat); pass e.g. "127.0.0.1" to restrict /metrics to loopback.
  Status Start(int port, const std::string& bind_addr = "");
  void Stop();             // idempotent
  int port() const { return port_; }
  // Handler threads not yet reaped (test/diagnostic). Bounded under scrape churn
  // now that finished connections are joined as new ones arrive — Prometheus
  // scrapes are Connection: close, so without reaping this grew per scrape.
  size_t live_conn_count();
  // Live connections admitted (not yet closed). Test/diagnostic; the accept
  // loop gates on this so DFKV_METRICS_MAX_CONNS is an exact bound.
  size_t ActiveConnections() const {
    return active_connections_.load(std::memory_order_relaxed);
  }
  // Accepted connections closed instantly because the server was at
  // DFKV_METRICS_MAX_CONNS (oneshot port: no queue for a handler thread).
  size_t DroppedConnections() const {
    return dropped_connections_.load(std::memory_order_relaxed);
  }

 private:
  void AcceptLoop();
  // accepted_at stamps the accept() return so the first complete request line
  // is due within DFKV_METRICS_FIRST_REQ_MS of ACCEPT, not of scheduling.
  void Handle(int fd, std::chrono::steady_clock::time_point accepted_at);
  void ReapDoneLocked();  // join+erase finished handler threads; conn_mu_ held

  // Absolute deadline for the first complete request line after accept — a
  // drip feeder otherwise never hit the per-syscall SO_RCVTIMEO (0 = off).
  static constexpr uint64_t kDefaultFirstReqMs = 30000;
  static constexpr uint64_t kHardFirstReqMs = 3600000;
  // Connection cap for this oneshot/low-frequency port (none existed before).
  static constexpr size_t kDefaultMaxConns = 64;
  static constexpr size_t kHardMaxConns = 4096;
  uint64_t first_req_ms_ = kDefaultFirstReqMs;
  size_t max_conns_ = kDefaultMaxConns;
  std::atomic<size_t> active_connections_{0};
  std::atomic<size_t> dropped_connections_{0};

  // A live connection: its handler thread + a flag the thread sets last, so
  // AcceptLoop can join it without blocking. shared_ptr because conns_ may
  // reallocate; the thread captures its own copy.
  struct Conn {
    std::thread th;
    std::shared_ptr<std::atomic<bool>> done;
  };

  std::function<std::string()> render_;
  std::function<bool()> health_;  // optional /healthz predicate (null => always ok)
  std::function<bool()> ready_;   // optional /readyz predicate (null => always ok)
  int listen_fd_ = -1;
  int port_ = 0;
  std::atomic<bool> running_{false};
  std::thread accept_thread_;
  std::mutex conn_mu_;
  std::set<int> conn_fds_;
  std::vector<Conn> conns_;
};

}  // namespace dfkv

#endif  // DFKV_METRICS_HTTP_H_
