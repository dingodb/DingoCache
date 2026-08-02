/* RDMA cache-node listener — native libibverbs RC. Mixed v2 uses small
 * per-QP control buffers plus leases from one process-wide registered receive
 * segment; legacy v1 remains available per connection.
 *
 * The listener is a TCP socket used only for capability/QP bootstrap. Startup
 * discovers ACTIVE HCAs (or applies the configured whitelist), anchors their
 * shared PD/MRs, and each accepted connection opens an RC QP on its requested
 * rail. shutdown(listen_fd) keeps Stop() interruptible. */
#ifndef DFKV_RDMA_SERVER_H_
#define DFKV_RDMA_SERVER_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/status.h"
#include "transport/rdma_verbs.h"  // rdma::RcEndpoint
#include "transport/rdma_recv_segment.h"

namespace dfkv {

class RdmaServer {
 public:
  using Handler = std::function<Status(
      uint8_t op, uint64_t id, uint32_t index, uint32_t ksize, uint64_t offset,
      uint64_t length, const char* payload, uint64_t payload_len,
      std::string* out_data)>;
  // Optional direct GET handler: read an O_DIRECT-aligned superset into `io_buf`
  // and return *out_data pointing inside that same registered buffer at the exact
  // requested range. When set, kRange replies scatter-send [resp | *out_data]
  // without copying the payload into sbuf.
  using RangeHandler = std::function<Status(
      uint64_t id, uint32_t index, uint32_t ksize, uint64_t offset,
      uint64_t length, char* io_buf, size_t io_cap, const char** out_data,
      size_t* out_len)>;
  // Optional direct PUT handler: `data` points at a 4096-aligned registered
  // buffer containing the full stored blob [ValueHeader|payload]. `cap` is the
  // usable direct-buffer capacity. The handler may zero O_DIRECT padding bytes
  // after len and write data directly to disk.
  using CacheDirectHandler = std::function<Status(
      uint64_t id, uint32_t index, uint32_t ksize, char* data, size_t len,
      size_t cap)>;

  // Optional async-GET prep handler (ADDITIVE, used only by the io_uring serve
  // path when DFKV_SERVER_URING=1). Does the cheap, lock-protected half of a
  // direct GET — index lookup, O_DIRECT open, range clamp + O_DIRECT alignment
  // math — but performs NO disk read. The serve loop issues the pread itself via
  // io_uring, then trims [head, head+payload_len) out of the aligned buffer. On
  // kOk the serve loop OWNS `fd` and must ::close it after the read completes.
  // payload_len==0 with fd<0 is a valid zero-length hit (no read needed).
  struct RangePrepResult {
    int fd = -1;               // owned by serve loop on kOk; -1 => no read
    uint64_t aligned_off = 0;  // O_DIRECT-aligned read start
    size_t aligned_len = 0;    // O_DIRECT-aligned read length (multiple of 4096)
    size_t head = 0;           // offset of the requested bytes within the read
    size_t payload_len = 0;    // exact requested bytes (post file-size clamp)
    // 0 = no hold. A slot-based engine (slab) pins the slot for the read's
    // duration; passed to range_release_handler_ wherever fd is closed.
    uint64_t release_token = 0;
    // 0 = no flight. Read-coalescer registration made at prep time: the serve
    // loop passes it to range_complete_handler_ with the payload once the async
    // read finishes, or to range_flight_abort_handler_ on any teardown path
    // where the read never completes — a registered flight MUST reach exactly
    // one of the two, or a convoy follower waits out its full timeout.
    uint64_t flight = 0;
  };
  using RangePrepHandler = std::function<Status(
      uint64_t id, uint32_t index, uint32_t ksize, uint64_t offset,
      uint64_t length, size_t io_cap, RangePrepResult* out)>;
  // Optional accounting hook invoked once an async read completes (mirrors the
  // hit/io-error counters the synchronous RangeDirect bumps).
  // elapsed_sec = wall time from prep (read submit) to completion, so the
  // accounting hook can observe the default (uring) path's disk-read latency —
  // the synchronous RangeDirect samples get_lat_ itself, but the async path
  // never did, leaving dfkv_op_latency_seconds{op="get"} blind to the default
  // read path since v1.27.0. Negative/zero-cost when the sampler skips (the
  // serve loop only stamps sampled reads).
  // `flight` is the prep's coalescer registration (0 = none) and `data` points
  // at the payload bytes in the connection's direct buffer (nullptr on
  // failure) — valid only for the duration of the call, which runs BEFORE the
  // reply send, so the handler may copy from it (coalesced waiters, RAM
  // promotion) but must not retain it.
  using RangeCompleteHandler =
      std::function<void(bool ok, size_t bytes_read, double elapsed_sec,
                         uint64_t flight, const char* data)>;
  // Releases a prep's release_token (slab slot pin) once its read is done.
  using RangeReleaseHandler = std::function<void(uint64_t token)>;
  // Aborts a prep's flight registration on paths where the async read never
  // completes (poisoned ring teardown, connection death): waiters fall back to
  // their own reads instead of hanging until timeout.
  using RangeFlightAbortHandler = std::function<void(uint64_t flight)>;

  // dev_name empty => env DFKV_RDMA_DEV; both empty => first ACTIVE local HCA
  // (legacy host-local semantics). An explicit comma list enables multi-rail.
  explicit RdmaServer(Handler handler, size_t max_msg = (64u << 20),
                      const std::string& dev_name = "");
  void set_range_handler(RangeHandler h) { range_handler_ = std::move(h); }
  void set_cache_direct_handler(CacheDirectHandler h) {
    cache_direct_handler_ = std::move(h);
  }
  // Wire the async-GET prep + completion accounting hooks. Optional: the io_uring
  // serve path is used when BOTH are set and the binary is built with
  // DFKV_WITH_URING (default ON since phase 10; DFKV_SERVER_URING=0 forces the
  // synchronous path). Otherwise the serve loop uses the sync path verbatim.
  void set_range_prep_handler(RangePrepHandler h) {
    range_prep_handler_ = std::move(h);
  }
  void set_range_complete_handler(RangeCompleteHandler h) {
    range_complete_handler_ = std::move(h);
  }
  void set_range_release_handler(RangeReleaseHandler h) {
    range_release_handler_ = std::move(h);
  }
  void set_range_flight_abort_handler(RangeFlightAbortHandler h) {
    range_flight_abort_handler_ = std::move(h);
  }

  // --- RAM hot-tier zero-copy GET (P3 B5-3, ADDITIVE + OFF unless wired) --------
  // On a kRange, the serve loop first asks ram_range_handler_ for a hit: on true
  // it returns (arena ptr, len, token) into a caller RAM arena, and the reply is
  // scatter-sent [resp | arena bytes] straight from the arena MR -- no copy into
  // the connection buffer, no disk. The NIC reads the shared arena during the
  // send, so the slot stays pinned until IBV_WC_SEND, when the serve loop calls
  // ram_release_handler_(token) (mirrors rearm_on_send). Both must be set AND the
  // arena registered via RegisterMemory for the path to activate; otherwise the
  // existing range_handler_ (copy-out) path runs unchanged.
  using RamRangeHandler = std::function<bool(
      uint64_t id, uint32_t index, uint32_t ksize, uint64_t offset,
      uint64_t length, const char** out_ptr, size_t* out_len, uint64_t* out_token)>;
  using RamReleaseHandler = std::function<void(uint64_t token)>;
  void set_ram_range_handler(RamRangeHandler h) { ram_range_handler_ = std::move(h); }
  void set_ram_release_handler(RamReleaseHandler h) { ram_release_handler_ = std::move(h); }
  // Register a caller memory region (the RAM arena) as a pool MR on every
  // connection's PD, so a RAM-hit payload resolves to an MR with no per-op
  // ibv_reg_mr. Call before Start(); regions are applied as each connection opens.
  void RegisterMemory(void* base, size_t size);

  ~RdmaServer();

  Status Start(int port);  // TCP bootstrap port
  void Stop();
  int port() const { return port_; }

  // Observability (used by tests): number of Serve threads not yet reaped.
  // Bounded under churn now that finished connections are joined as new ones
  // arrive — previously this grew without bound until Stop().
  size_t live_conn_count();

  // RDMA completion counters (relaxed atomics, incremented in the serve loop).
  uint64_t Completions() const { return completions_.load(std::memory_order_relaxed); }
  uint64_t CompletionErrors() const { return completion_errors_.load(std::memory_order_relaxed); }
  uint64_t ActiveConns() const { return active_conns_.load(std::memory_order_relaxed); }
  uint64_t IdleReclaims() const { return idle_reclaims_.load(std::memory_order_relaxed); }
  // io_uring async-GET path observability: reads actually submitted through a
  // ring, and connections that WANTED the path but fell back to sync (ring init
  // failed). Both zero when the path is off -- uring_reads_total > 0 is the
  // external proof the env-gated path is really active (the fallback is
  // otherwise silent by design, correctness-first).
  uint64_t UringReads() const { return uring_reads_.load(std::memory_order_relaxed); }
  uint64_t UringInitFallbacks() const { return uring_init_fallbacks_.load(std::memory_order_relaxed); }
  uint64_t V1Conns() const { return v1_conns_.load(std::memory_order_relaxed); }
  uint64_t V2Conns() const { return v2_conns_.load(std::memory_order_relaxed); }
  uint64_t V2PutWrites() const {
    return v2_put_writes_.load(std::memory_order_relaxed);
  }
  uint64_t V2GetWrites() const {
    return v2_get_writes_.load(std::memory_order_relaxed);
  }
  // The server-side pipeline depth (env DFKV_RDMA_DEPTH, default 1) -- surfaced
  // in ring INFO because the CLIENT's depth must not exceed it: excess in-flight
  // requests hit receiver-not-ready retries and degrade SILENTLY (measured 3-4x
  // on pipelined GETs), they don't fail.
  size_t PipelineDepth() const;
  std::string MetricsText() const;  // Prometheus text (dfkv_rdma_*)
  // Whether the io_uring async-GET serve path should be used for new conns.
  // Default ON when built with DFKV_WITH_URING and both range prep + complete
  // handlers are set; DFKV_SERVER_URING=0 forces sync. Decided once per conn.
  // Public so startup can force-resolve DFKV_SERVER_URING into the config dump.
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

  Handler handler_;
  RangeHandler range_handler_;
  CacheDirectHandler cache_direct_handler_;
  RangePrepHandler range_prep_handler_;
  RangeCompleteHandler range_complete_handler_;
  RangeReleaseHandler range_release_handler_;
  RangeFlightAbortHandler range_flight_abort_handler_;
  RamRangeHandler ram_range_handler_;
  RamReleaseHandler ram_release_handler_;
  std::vector<std::pair<void*, size_t>> user_regions_;  // RAM arena pool MRs (RegisterMemory)
  size_t max_msg_;
  size_t control_cap_;
  std::string dev_name_;
  bool v2_enabled_ = true;
  bool auto_device_ = true;
  int listen_fd_ = -1;
  int port_ = 0;
  std::atomic<bool> running_{false};
  std::thread accept_thread_;
  // Track per-connection Serve threads + their endpoints so Stop() can wake them
  // out of WaitComp and join them before the handler's owner is destroyed, and
  // so finished threads are reaped incrementally (see ReapDoneLocked).
  std::mutex conn_mu_;
  std::vector<Conn> conns_;
  std::unordered_set<rdma::RcEndpoint*> live_eps_;
  // One process-wide receive segment replaces per-connection multi-MiB
  // request/direct-I/O buffers for v2. Each connection leases depth*slot_size;
  // the segment MR is shared per PD through RcEndpoint's device registry.
  rdma::RecvSegment recv_segment_;
  size_t recv_segment_bytes_ = 0;
  size_t recv_segment_registered_rails_ = 0;
  // One anchor per resolved ACTIVE rail holds a lifetime shared-device ref and
  // registers the receive segment / caller pools once on that rail's shared PD.
  // With no explicit device filter only the first ACTIVE local rail is anchored,
  // preserving the legacy rule that HCA names are host-local, not peer IDs.
  std::vector<std::unique_ptr<rdma::RcEndpoint>> anchors_;
  std::vector<std::string> anchor_devs_;  // configured filter, then active rails
  std::atomic<uint64_t> uring_reads_{0}, uring_init_fallbacks_{0};
  std::atomic<uint64_t> v1_conns_{0}, v2_conns_{0}, v2_put_writes_{0},
      v2_get_writes_{0};
  std::atomic<uint64_t> completions_{0}, completion_errors_{0}, active_conns_{0},
      idle_reclaims_{0};
};

}  // namespace dfkv

#endif  // DFKV_RDMA_SERVER_H_
