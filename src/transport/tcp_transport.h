#ifndef DFKV_TCP_TRANSPORT_H_
#define DFKV_TCP_TRANSPORT_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "transport/transport.h"
#include "transport/wire.h"

namespace dfkv {

// TCP transport with a bounded adaptive per-node connection pool. Concurrent
// calls to one routed node spread over healthy keep-alive sockets; no selection
// can cross the node boundary. Extra sockets grow only under contention and
// retire after idleness. Thread-safe.
class TcpTransport : public Transport {
 public:
  TcpTransport();
  explicit TcpTransport(EndpointPoolOptions options);
  ~TcpTransport() override;

  Status Cache(const std::string& node, const BlockKey& key, const void* data,
               size_t len) override;
  Status Range(const std::string& node, const BlockKey& key, uint64_t offset,
               uint64_t length, std::string* out,
               uint64_t* value_len = nullptr) override;
  std::vector<Status> RangeInto(
      const std::string& node, const std::vector<BlockKey>& keys,
      const std::vector<RangeDst>& dsts,
      std::vector<uint64_t>* value_lens) override;
  Status Lookup(const std::string& node, const BlockKey& key,
                uint64_t* value_len) override;
  Status Exist(const std::string& node, const BlockKey& key,
               bool* exist) override;
  Status Remove(const std::string& node, const BlockKey& key) override;

  // Fetch a node's Prometheus-format metrics text (not part of Transport iface).
  Status Stats(const std::string& node, std::string* out);
  Status Members(const std::string& node, std::string* out) override;
  std::string MetricsText() const override;

  // connect/IO timeouts (ms); 0 = block indefinitely. Defaults bound hangs.
  // Configure before issuing traffic.
  void set_timeouts(int connect_ms, int io_ms) {
    connect_ms_ = connect_ms; io_ms_ = io_ms;
  }

 private:
  using Clock = std::chrono::steady_clock;
  struct Connection {
    Connection(int socket_fd, uint64_t stable_id, Clock::time_point now)
        : fd(socket_fd), id(stable_id), last_used(now) {}
    ~Connection();

    int fd;
    uint64_t id;
    size_t inflight = 0;  // TCP wire is serial, therefore reserved as 0/1
    uint64_t latency_ns = 0;
    Clock::time_point last_used;
  };
  struct NodePool {
    std::vector<std::shared_ptr<Connection>> connections;
    size_t creating = 0;
    uint32_t failure_streak = 0;
    Clock::time_point backoff_until{};
  };
  struct Lease {
    std::shared_ptr<Connection> connection;
    bool reused = false;
    Clock::time_point started{};
  };

  Status RoundTrip(const std::string& node, WireOp op, const BlockKey& key,
                   uint64_t offset, uint64_t length, const void* payload,
                   uint64_t payload_len, std::string* out,
                   uint64_t max_data, uint64_t* value_len = nullptr);
  Status RoundTripInto(const std::string& node, const BlockKey& key,
                       uint64_t offset, uint64_t length, void* dst,
                       size_t dst_cap, uint64_t* value_len);
  bool Acquire(const std::string& node, Lease* lease);
  void Complete(const std::string& node, Lease lease, bool healthy,
                bool apply_backoff);
  void RetireIdleLocked(NodePool* pool, Clock::time_point now);
  void ApplyBackoffLocked(NodePool* pool, Clock::time_point now);
  static EndpointPoolOptions OptionsFromEnv();
  static EndpointPoolOptions NormalizeOptions(EndpointPoolOptions options);

  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::unordered_map<std::string, NodePool> pools_;
  EndpointPoolOptions options_;
  uint64_t next_connection_id_ = 1;
  int connect_ms_ = 3000;
  int io_ms_ = 10000;

  std::atomic<uint64_t> pool_connections_{0};
  std::atomic<uint64_t> pool_inflight_{0};
  std::atomic<uint64_t> pool_selections_{0};
  std::atomic<uint64_t> pool_retired_idle_{0};
  std::atomic<uint64_t> pool_retired_error_{0};
  std::atomic<uint64_t> pool_backoff_events_{0};
  std::atomic<uint64_t> pool_backoff_suppressed_{0};
};

}  // namespace dfkv

#endif  // DFKV_TCP_TRANSPORT_H_
