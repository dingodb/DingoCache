#include "cache/store_engine.h"
/* KvNodeServer — a cache-node daemon for the harness: wraps a DiskCacheGroup
 * (one or more NVMe SSDs) and serves Cache/Range/Exist over the TCP wire
 * protocol. The real cache node is dingo-cache (brpc + DiskCache); this proves
 * the semantics end-to-end, including multi-disk intra-node sharding. */
#ifndef DFKV_KV_NODE_SERVER_H_
#define DFKV_KV_NODE_SERVER_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "cache/disk_cache_group.h"
#include "cache/ram_tier.h"
#include "cache/read_coalescer.h"
#include "utils/latency_hist.h"

namespace dfkv {

class KvNodeServer {
 public:
  // single disk (back-compat)
  KvNodeServer(const std::string& cache_dir, uint64_t capacity_bytes);
  // multiple NVMe SSDs on this node (total capacity split across disks)
  KvNodeServer(const std::vector<std::string>& cache_dirs,
               uint64_t capacity_bytes);
  ~KvNodeServer();

  Status Start(int port);  // port 0 => ephemeral; query with port()
  void Stop();
  // Cluster member list this node advertises for discovery (kMembers wire op):
  // "name=ip:port,name=ip:port,...". Clients query any node to learn the cluster.
  void set_members(const std::string& members) { members_ = members; }
  // Node identity for Prometheus labels. When unset, series are emitted unlabeled
  // (back-compat with single-node scrapes / older tooling).
  void set_identity(const std::string& id, const std::string& group) {
    node_id_ = id; node_group_ = group;
  }
  // Upper bound for a request frame's declared payload_len. The TCP handler
  // allocates payload_len BEFORE reading the payload bytes, so without a real
  // bound a 42-byte forged prefix triggers a 16 GiB allocation (one-frame OOM
  // DoS). Defaults to the env-resolved raw-value ceiling — the same bound the
  // RDMA server enforces (utils/wire_limits.h).
  void set_max_request_payload(uint64_t n) { max_request_payload_ = n; }
  int port() const { return port_; }
  size_t Count() const { return group_.Count(); }
  uint64_t UsedBytes() const { return group_.UsedBytes(); }
  // Dynamic storage health used by /healthz and /readyz. MDS heartbeat state is
  // intentionally excluded: only the first successful registration gates
  // readiness; terminal local store/RAM failures remove it.
  bool Healthy() const {
    return group_.Healthy() && !ram_required_failed_ &&
           (!ram_ || ram_->healthy());
  }
  // Accessors for the MDS heartbeat stats provider (STA1) -- all cheap reads.
  uint64_t Evictions() const { return group_.Evictions(); }
  uint64_t PutBusyTotal() const { return put_busy_.load(std::memory_order_relaxed); }
  uint64_t DioWriteFallbacks() const { return group_.SlabStats().dio_write_fallbacks; }
  uint64_t RamUsedBytes() const { return ram_ ? ram_->UsedBytes() : 0; }
  uint64_t RamHits() const { return ram_ ? ram_->Hits() : 0; }
  uint64_t UptimeSeconds() const {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time_).count());
  }
  size_t DiskCount() const { return group_.DiskCount(); }
  size_t AcceptCount() const { return accept_count_.load(std::memory_order_relaxed); }
  size_t live_conn_count();  // handler threads not yet reaped (test/diagnostic)
  size_t TcpRejectedConnections() const {
    return tcp_rejected_connections_.load(std::memory_order_relaxed);
  }
  size_t TcpMaxConnections() const { return tcp_max_connections_; }
  int TcpIoTimeoutSeconds() const { return tcp_io_timeout_seconds_; }
  uint64_t TcpFirstReqMs() const { return tcp_first_req_ms_; }

  // metrics (relaxed atomics)
  size_t m_cache_put() const { return cache_put_.load(std::memory_order_relaxed); }
  size_t m_cache_hit() const { return cache_hit_.load(std::memory_order_relaxed); }
  size_t m_cache_miss() const { return cache_miss_.load(std::memory_order_relaxed); }
  size_t m_exist_hit() const { return exist_hit_.load(std::memory_order_relaxed); }
  size_t m_exist_miss() const { return exist_miss_.load(std::memory_order_relaxed); }
  size_t m_remove_ok() const { return remove_ok_.load(std::memory_order_relaxed); }
  size_t m_remove_miss() const { return remove_miss_.load(std::memory_order_relaxed); }
  std::string MetricsText() const;  // cache-node metrics only
  // Add transport/registrar/readiness metrics to both HTTP /metrics and the
  // kStats wire response used by dfkvctl. Assignment and scrape are thread-safe.
  void set_metrics_extension(std::function<std::string()> render);
  std::string FullMetricsText() const;

  // Transport-agnostic request processing (shared by the TCP handler and, when
  // built with DFKV_WITH_RDMA, the RDMA handler). Returns status and raw bytes;
  // `value_len` comes from the same pinned storage/RAM snapshot as those bytes.
  // Every frontend supplies the authoritative native digest identity.
  Status ProcessRequestForKey(uint8_t op, const BlockKey& key,
                              uint64_t offset, uint64_t length,
                              const char* payload, uint64_t payload_len,
                              std::string* out_data,
                              size_t* value_len = nullptr);
  // Payload-free lookup used by protocol response framing and connector
  // descriptor validation. RAM is authoritative while a write-back value is
  // visible but not yet committed to disk.
  Status LookupForKey(const BlockKey& key, ValueMetadata* out) const;


  // Zero-copy server-side GET: read the block straight into `dst` (e.g. the RDMA
  // send buffer), no intermediate std::string. *out_len = bytes read. Updates
  // hit/miss + bytes_read metrics like a Range.
  Status RangeIntoForKey(const BlockKey& key, uint64_t offset,
                         uint64_t length, char* dst, size_t dst_cap,
                         size_t* out_len, size_t* value_len = nullptr);

  // RDMA direct PUT: `data` is an aligned registered buffer containing raw
  // value bytes. Writes without a payload-sized CPU copy.
  Status CacheDirectForKey(const BlockKey& key, char* data, size_t len,
                           size_t cap);
  std::vector<Status> CacheDirectBatchForKeys(
      const std::vector<StoreEngine::CacheBatchItem>& items);

  // RDMA direct GET: read an O_DIRECT-aligned superset into `io_buf`; *out_data
  // points inside that same buffer at the exact requested range so the RDMA layer
  // can scatter-send it without copying into sbuf.
  Status RangeDirectForKey(const BlockKey& key, uint64_t offset,
                           uint64_t length, char* io_buf, size_t io_cap,
                           const char** out_data, size_t* out_len,
                           size_t* value_len = nullptr);

  // Prepare one RDMA read against its registered staging destination. The
  // returned move-only transaction owns every disk/RAM/coalescer obligation;
  // Commit(), Abort(), or destruction is the only terminal protocol.
  PreparedRead PrepareReadForKey(const BlockKey& key, uint64_t offset,
                                 uint64_t length, char* staging,
                                 size_t staging_cap);

  bool ram_enabled() const { return ram_ != nullptr; }
  // Resolved storage backend name ("file"|"slab"), for self-description reporting.
  const std::string& engine_name() const { return group_.EngineName(); }
  const std::string& write_mode() const { return group_.WriteMode(); }
  char* ram_arena() const { return ram_ ? ram_->arena() : nullptr; }
  uint64_t ram_arena_bytes() const { return ram_ ? ram_->arena_bytes() : 0; }
  bool ram_ack_enabled() const { return ram_ack_enabled_; }

 private:
  static void FinishDiskRead(void* owner, uint64_t flight, bool committed,
                             Status result, size_t bytes_read,
                             double elapsed_sec, const char* data) noexcept;
  struct PromotedRead {
    KvNodeServer* server = nullptr;
    RamTier::DurableReservation reservation;
  };
  static void FinishPromotedDiskRead(
      void* owner, uint64_t flight, bool committed, Status result,
      size_t bytes_read, double elapsed_sec, const char* data) noexcept;
  static void FinishRamRead(void* owner, uint64_t token, bool committed,
                            Status result, size_t bytes_read,
                            double elapsed_sec, const char* data) noexcept;
  void AcceptLoop();
  // accepted_at stamps the accept() return so the first complete request frame
  // is due within DFKV_TCP_FIRST_REQ_MS of ACCEPT, not of handler scheduling.
  void Handle(int fd, std::chrono::steady_clock::time_point accepted_at);
  void InitRamTier();     // construct ram_ if DFKV_RAM_TIER is enabled (env)
  // RAM tier write policy: write-back (PUT lands in arena, async flush) or
  // write-around (PUT direct to disk, arena filled by GET read-promotion).
  void set_ram_write_back(bool wb) { ram_write_back_ = wb; }
  void InitAdmission();   // read DFKV_PUT_INFLIGHT_LIMIT (0 = gate off)
  void ReapDoneLocked();  // join+erase finished handler threads; conn_mu_ held
  void InitTcpListenerConfig();
  std::atomic<size_t> cache_put_{0}, cache_hit_{0}, cache_miss_{0};
  std::atomic<size_t> exist_hit_{0}, exist_miss_{0};
  std::atomic<size_t> remove_ok_{0}, remove_miss_{0};
  std::atomic<size_t> bytes_written_{0}, bytes_read_{0};
  // Read-side convoy collapse (env DFKV_READ_COALESCE=1; see read_coalescer.h).
  ReadCoalescer read_coalescer_;
  bool coalesce_enabled_ = [] {
    const char* e = std::getenv("DFKV_READ_COALESCE");
    return e && *e && *e != '0';
  }();
  // depth metrics: errors by op, live connections, sampled op latency
  std::atomic<size_t> put_io_err_{0}, get_io_err_{0}, invalid_ops_{0};
  // PUT admission gate (I6): reject with kCacheFull once this many disk writes
  // are in flight (0 = unlimited/off). See --put-inflight-limit.
  size_t put_busy_limit_ = 0;
  bool ram_write_back_ = true;  // default write-back; DFKV_RAM_WRITE_MODE=writearound switches
  std::atomic<size_t> disk_put_inflight_{0};
  std::atomic<size_t> put_busy_{0};
  std::atomic<size_t> open_connections_{0};
  static constexpr size_t kDefaultTcpMaxConnections = 512;
  static constexpr size_t kHardTcpMaxConnections = 4096;
  static constexpr int kDefaultTcpIoTimeoutSeconds = 60;
  static constexpr int kHardTcpIoTimeoutSeconds = 3600;
  // Absolute deadline for the FIRST complete request frame after accept
  // (SO_RCVTIMEO alone is per-syscall, so a drip feeder never timed out).
  static constexpr uint64_t kDefaultTcpFirstReqMs = 30000;
  static constexpr uint64_t kHardTcpFirstReqMs = 3600000;
  size_t tcp_max_connections_ = kDefaultTcpMaxConnections;
  int tcp_io_timeout_seconds_ = kDefaultTcpIoTimeoutSeconds;
  uint64_t tcp_first_req_ms_ = kDefaultTcpFirstReqMs;  // 0 = deadline disabled
  std::atomic<size_t> tcp_rejected_connections_{0};
  Sampler lat_sampler_{64};        // 1-in-64 latency sampling (near-zero hot-path cost)
  LatencyHist get_lat_, put_lat_;  // server-side op latency (sampled)
  // Exist HANDLER latency (Contains + IsCached under the shard/reclaim locks).
  // Separate from get: exist is a key-only probe with a completely different
  // latency profile, and its tail is the first thing to check when L3 prefetch
  // stalls (a slow exist blocks the prefetch decision). This measures the
  // handler body only; serve-loop queueing (a large GET draining ahead of an
  // exist on the same connection) is a separate, connection-ordered cost the
  // client's control-lane QP split is meant to avoid.
  LatencyHist exist_lat_;
  std::string members_;  // advertised cluster membership (kMembers)
  uint64_t max_request_payload_ = 0;  // 0 = resolve default lazily (see .cc)
  std::string node_id_, node_group_;       // identity for Prometheus labels (optional)
  std::chrono::steady_clock::time_point start_time_ = std::chrono::steady_clock::now();
  mutable std::mutex metrics_extension_mu_;
  std::function<std::string()> metrics_extension_;

  DiskCacheGroup group_;
  // Optional RAM hot tier (P3). Declared AFTER group_ so it is destroyed FIRST:
  // ~RamTier stops+joins the flusher, whose callback calls group_.Cache, before
  // group_ is torn down. nullptr unless DFKV_RAM_TIER is enabled. In write-back
  // mode, PUT lands in RAM (sync-visible) and flushes to group_; write-around
  // populates RAM through born-durable GET promotion. GET checks RAM first.
  std::unique_ptr<RamTier> ram_;
  bool ram_ack_enabled_ = false;
  bool ram_required_failed_ = false;
  int listen_fd_ = -1;
  int port_ = 0;
  std::atomic<bool> running_{false};
  std::atomic<size_t> accept_count_{0};
  std::thread accept_thread_;
  // connection drain: track per-connection handler threads + their fds so Stop()
  // can unblock (shutdown) and join them before group_ is destroyed. Finished
  // handlers are reaped incrementally at accept time (done flag), so a long-lived
  // node with churning clients doesn't accumulate dead thread handles.
  struct Conn {
    std::thread th;
    std::shared_ptr<std::atomic<bool>> done;
  };
  std::mutex conn_mu_;
  std::set<int> conn_fds_;
  std::vector<Conn> conns_;
};

}  // namespace dfkv

#endif  // DFKV_KV_NODE_SERVER_H_
