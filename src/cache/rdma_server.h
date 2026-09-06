/* RDMA cache-node listener — native v2 libibverbs RC. Bounded 32,786-byte
 * per-QP control buffers share a process-wide registered payload segment.
 * Peers that cannot negotiate v2 are rejected.
 * The TCP listener bootstraps QPs and proves failed writer generations retired.
 * Startup discovers the first ACTIVE HCA automatically, or resolves every
 * explicitly configured HCA into a fixed topology (including initially inactive
 * ports), then anchors shared PD/MRs.
 * shutdown(listen_fd) keeps Stop() interruptible. */
#ifndef DFKV_RDMA_SERVER_H_
#define DFKV_RDMA_SERVER_H_

#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef DFKV_WITH_URING
#include "cache/uring_reader.h"
#endif
#include "cache/store_engine.h"
#include "common/kv_types.h"
#include "common/status.h"
#include "transport/rdma_recv_segment.h"
#include "transport/rdma_verbs.h"  // rdma::RcEndpoint
#include "transport/rdma_topology.h"

namespace dfkv {
class RdmaServerTestPeer;

class RdmaServer {
 public:
  using Handler = std::function<Status(
      uint8_t op, const BlockKey& key, uint64_t offset, uint64_t length,
      const char* payload, uint64_t payload_len, std::string* out_data,
      size_t* value_len)>;
  // Optional direct GET handler: reads raw bytes into a registered buffer and
  // reports both returned bytes and the authoritative full stored value size.
  using RangeHandler = std::function<Status(
      const BlockKey& key, uint64_t offset, uint64_t length, char* io_buf,
      size_t io_cap, const char** out_data, size_t* out_len,
      size_t* value_len)>;
  // Optional direct PUT handler: `data` points at an aligned registered buffer
  // containing exactly the raw stored value bytes.
  using CacheDirectHandler = std::function<Status(
      const BlockKey& key, char* data, size_t len, size_t cap)>;
  // Batched direct PUT: multiple keys in one call for io_uring flush.
  using CacheDirectBatchHandler = std::function<std::vector<Status>(
      const std::vector<StoreEngine::CacheBatchItem>& items)>;

  // One read-preparation entry point for disk, coalescer, and RAM sources. The
  // returned move-only transaction is the sole cleanup authority across
  // queueing, fallback, completion, timeout, and connection teardown.
  using PrepareReadHandler = std::function<PreparedRead(
      const BlockKey& key, uint64_t offset, uint64_t length, char* staging,
      size_t staging_cap)>;

  // dev_name empty => env DFKV_RDMA_DEV; both empty => first ACTIVE local HCA.
  // An explicit comma list enables multi-rail.
  explicit RdmaServer(Handler handler, size_t max_msg = (64u << 20),
                      const std::string& dev_name = "");
  void set_range_handler(RangeHandler h) { range_handler_ = std::move(h); }
  void set_cache_direct_handler(CacheDirectHandler h) {
    cache_direct_handler_ = std::move(h);
  }
  void set_cache_direct_batch_handler(CacheDirectBatchHandler h) {
    cache_direct_batch_handler_ = std::move(h);
  }
  void set_prepare_read_handler(PrepareReadHandler h) {
    prepare_read_handler_ = std::move(h);
  }
  // Register a caller memory region (the RAM arena) as a pool MR on every
  // connection's PD, so a RAM-hit payload resolves to an MR with no per-op
  // ibv_reg_mr. Call before Start(); regions are applied as each connection opens.
  void RegisterMemory(void* base, size_t size);

  ~RdmaServer();

  Status Start(int port);  // TCP bootstrap port
  void Stop();
  int port() const { return port_; }
  // Explicit-mode failures retain the complete requested order; diagnostic
  // probe evidence never replaces this with a partially discovered topology.
  const std::vector<std::string>& DeviceNames() const { return anchor_devs_; }
  // Fixed topology and successfully materialized anchors. Explicit startup is
  // fail-closed, so a running server has one initialized anchor per configured
  // rail; keeping both counts exposes startup/lifecycle truth independently.
  size_t ConfiguredRailCount() const { return anchor_devs_.size(); }
  size_t InitializedRailCount() const { return anchors_.size(); }

  // Observability (used by tests): number of Serve threads not yet reaped.
  // Bounded under churn now that finished connections are joined as new ones
  // arrive — previously this grew without bound until Stop().
  size_t live_conn_count();

  // RDMA completion counters (relaxed atomics, incremented in the serve loop).
  uint64_t Completions() const { return completions_.load(std::memory_order_relaxed); }
  uint64_t CompletionErrors() const { return completion_errors_.load(std::memory_order_relaxed); }
  uint64_t ActiveConns() const { return active_conns_.load(std::memory_order_relaxed); }
  uint64_t IdleReclaims() const { return idle_reclaims_.load(std::memory_order_relaxed); }
  // io_uring pipeline observability. A submit batch is one SQ flush group;
  // completions count logical descriptors (short-read residuals stay one).
  // All remain zero when the path is off.
  uint64_t UringReads() const {
    return uring_reads_.load(std::memory_order_relaxed);
  }
  uint64_t UringReadBatches() const {
    return uring_read_batches_.load(std::memory_order_relaxed);
  }
  uint64_t UringReadBatchMax() const {
    return uring_read_batch_max_.load(std::memory_order_relaxed);
  }
  uint64_t UringCompletions() const {
    return uring_completions_.load(std::memory_order_relaxed);
  }
  uint64_t UringInflight() const {
    return uring_inflight_.load(std::memory_order_relaxed);
  }
  uint64_t UringInflightMax() const {
    return uring_inflight_max_.load(std::memory_order_relaxed);
  }
  uint64_t UringSendFences() const {
    return uring_send_fences_.load(std::memory_order_relaxed);
  }
  uint64_t UringRepliesPosted() const {
    return uring_replies_posted_.load(std::memory_order_relaxed);
  }
  uint64_t UringSendPostErrors() const {
    return uring_send_post_errors_.load(std::memory_order_relaxed);
  }
  uint64_t UringInitFallbacks() const {
    return uring_init_fallbacks_.load(std::memory_order_relaxed);
  }
  uint64_t V2Conns() const { return v2_conns_.load(std::memory_order_relaxed); }
  uint64_t V2PutWrites() const {
    return v2_put_writes_.load(std::memory_order_relaxed);
  }
  uint64_t V2GetWrites() const {
    return v2_get_writes_.load(std::memory_order_relaxed);
  }
  uint64_t V2GetContinuationSlotChanges() const {
    return v2_get_continuation_slot_changes_.load(std::memory_order_relaxed);
  }
  // The server-side pipeline depth (env DFKV_RDMA_DEPTH, default 4) -- surfaced
  // in ring INFO because the CLIENT's depth must not exceed it: excess in-flight
  // requests hit receiver-not-ready retries and degrade SILENTLY (measured 3-4x
  // on pipelined GETs) and cause PUT batch failures during burst writes.
  size_t PipelineDepth() const;
  std::string MetricsText() const;  // Prometheus text (dfkv_rdma_*)
  // Whether the io_uring async-GET serve path should be used for new conns.
  // Default ON when built with DFKV_WITH_URING and the prepared-read handler is
  // set; DFKV_SERVER_URING=0 forces sync. Decided once per connection.
  bool UseUringPath() const;

 private:
  void AcceptLoop();
  void Serve(int boot_fd);
  void ReapDoneLocked();  // join+erase finished Serve threads; conn_mu_ held

  // A live connection: its Serve thread plus a flag the thread sets (last thing
  // it does) so AcceptLoop can tell it has finished and join it without blocking.
  // The flag is a shared_ptr because conns_ reallocates on push_back — the thread
  // captures its own copy, so the atomic outlives any vector move/erase.
  struct Conn {
    std::thread th;
    std::shared_ptr<std::atomic<bool>> done;
  };

  struct WriterState {
    std::mutex mu;
    std::condition_variable retired_cv;
    rdma::RcEndpoint* endpoint = nullptr;
    bool retired = false;
  };
  // Register a writer under an OS-random, nonzero token. Candidate insertion
  // and collision detection are atomic with respect to retire lookups.
  uint64_t RegisterWriter(const std::shared_ptr<WriterState>& writer);

  // Per-device counters make a partial rail failure or affinity imbalance
  // visible without polling host counters. The rail set is fixed by Start().
  struct RailStats {
    std::atomic<uint64_t> active_conns{0};
    std::atomic<uint64_t> completions{0};
    std::atomic<uint64_t> completion_errors{0};
    std::atomic<uint64_t> put_writes{0};
    std::atomic<uint64_t> put_bytes{0};
    std::atomic<uint64_t> get_writes{0};
    std::atomic<uint64_t> get_bytes{0};
  };

  Handler handler_;
  RangeHandler range_handler_;
  CacheDirectHandler cache_direct_handler_;
  CacheDirectBatchHandler cache_direct_batch_handler_;
  PrepareReadHandler prepare_read_handler_;
  std::vector<std::pair<void*, size_t>> user_regions_;  // RAM arena pool MRs (RegisterMemory)
  size_t max_msg_;
  std::string dev_name_;
  bool auto_device_ = true;
  int listen_fd_ = -1;
  int port_ = 0;
  std::atomic<bool> running_{false};
  std::thread accept_thread_;
  std::thread recv_trim_thread_;
  std::mutex recv_trim_mu_;
  std::condition_variable recv_trim_cv_;
  // Track per-connection Serve threads + their endpoints so Stop() can wake them
  // out of WaitComp and join them before the handler's owner is destroyed, and
  // so finished threads are reaped incrementally (see ReapDoneLocked).
  std::mutex conn_mu_;
  std::vector<Conn> conns_;
  std::unordered_set<rdma::RcEndpoint*> live_eps_;
  std::mutex writer_mu_;
  std::unordered_map<uint64_t, std::shared_ptr<WriterState>> writers_;
  // Receive memory is committed in fixed-size chunks on demand. Connections
  // lease only their negotiated slot geometry from any chunk; the old
  // DFKV_RDMA_RECV_SEGMENT_SIZE remains the hard process budget.
  rdma::RecvSegmentPool recv_segments_;
  std::atomic<uint64_t> segment_evictions_{0};
  size_t recv_segment_max_bytes_ = 0;
  size_t recv_segment_chunk_bytes_ = 0;
  uint64_t recv_chunk_idle_ms_ = 60000;
  size_t recv_segment_registered_rails_ = 0;
  std::atomic<uint64_t> pull_connections_{0};
  std::atomic<uint64_t> pull_memory_windows_{0};
  std::atomic<uint64_t> pull_mr_fallbacks_{0};
  std::atomic<uint64_t> legacy_connections_{0};
  std::atomic<uint64_t> data_connection_bytes_{0};
  std::atomic<uint64_t> control_connection_bytes_{0};
  // In-flight leased-PUT staging: receive memory held only while an
  // object's windows are actually in flight, unlike the connection-lifetime
  // leases above. Gauges make the pool/non-resident split observable.
  std::atomic<uint64_t> lease_put_ops_{0};
  std::atomic<uint64_t> lease_put_bytes_{0};
  std::atomic<uint64_t> lease_put_busy_rejects_{0};
  std::atomic<uint64_t> lease_put_active_{0};
  std::atomic<uint64_t> lease_put_bytes_active_{0};
  // One anchor per resolved rail holds a lifetime shared device reference and
  // registers the initial receive chunk and caller pools on that rail's PD.
  // Later chunks register lazily on the rail of the connection that leases
  // them. Auto mode anchors only the first ACTIVE local rail.
  std::vector<std::unique_ptr<rdma::RcEndpoint>> anchors_;
  std::vector<std::string> anchor_devs_;  // fixed resolved rail names
  std::vector<std::unique_ptr<RailStats>> rail_stats_;  // indexed with anchor_devs_
  // Narrow hardware-free startup seams, private and reachable only through the
  // test peer. Production always calls verbs discovery and materializes real
  // anchors.
  std::function<rdma::RdmaDiscoveryResult(
      const std::vector<std::string>&, rdma::RdmaDiscoveryPolicy)>
      discover_for_test_;
  std::function<std::unique_ptr<rdma::RcEndpoint>(
      const std::string&, const std::vector<std::pair<void*, size_t>>&,
      void*, size_t)>
      initialize_anchor_for_test_;
#ifdef DFKV_WITH_URING
  // Externally owned backend seam for deterministic queue/completion tests.
  // Production leaves this empty and UringReader selects liburing directly.
  std::function<UringReader::Backend*()> uring_backend_factory_for_test_;
#endif
  friend class RdmaServerTestPeer;
  std::atomic<uint64_t> uring_reads_{0}, uring_read_batches_{0},
      uring_read_batch_max_{0}, uring_completions_{0}, uring_inflight_{0},
      uring_inflight_max_{0}, uring_replies_posted_{0},
      uring_send_fences_{0}, uring_send_post_errors_{0},
      uring_init_fallbacks_{0};
  std::atomic<uint64_t> v2_conns_{0}, v2_put_writes_{0}, v2_get_writes_{0};
  std::atomic<uint64_t> v2_get_continuation_slot_changes_{0};
  std::atomic<uint64_t> completions_{0}, completion_errors_{0}, active_conns_{0},
      idle_reclaims_{0};
};

}  // namespace dfkv

#endif  // DFKV_RDMA_SERVER_H_
