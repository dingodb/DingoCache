#include "cache/store_engine.h"
/* KvNodeServer — a cache-node daemon for the harness: wraps a DiskCacheGroup
 * (one or more NVMe SSDs) and serves Cache/Range/Exist over the TCP wire
 * protocol. The real cache node is dingo-cache (brpc + DiskCache); this proves
 * the semantics end-to-end, including multi-disk intra-node sharding. */
#ifndef DFKV_KV_NODE_SERVER_H_
#define DFKV_KV_NODE_SERVER_H_

#include <atomic>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "cache/disk_cache_group.h"
#include "cache/ram_tier.h"
#include "cache/remove_memory.h"
#include "cache/write_memory.h"
#include "common/raw_clock.h"
#include "cache/read_coalescer.h"
#include "cache/recent_reads.h"
#include "common/block_key_hash.h"
#include "utils/accept_guard.h"
#include "utils/env_parse.h"
#include "utils/latency_hist.h"

namespace dfkv {

class KvNodeServer {
 public:
  // single disk (back-compat)
  KvNodeServer(const std::string& cache_dir, uint64_t capacity_bytes);
  // multiple NVMe SSDs on this node (total capacity split across disks)
  // `cancel_start` (optional, must outlive the constructor call): becomes
  // true when the start is no longer wanted, typically from a signal handler.
  // The constructor then gives up faulting in the RAM tier arena, still lets
  // the disks finish opening (seconds; they are closed again by Shutdown),
  // and returns with start_cancelled() true. Such a server never starts.
  KvNodeServer(const std::vector<std::string>& cache_dirs,
               uint64_t capacity_bytes,
               const std::atomic<bool>* cancel_start = nullptr);
  ~KvNodeServer();

  Status Start(int port);  // port 0 => ephemeral; query with port()
  void Stop();
  // The last thing a process does with this server before it exits: Stop(),
  // then get the disks to a clean state before any memory is released.
  //  1. The RAM tier gets DFKV_RAM_SHUTDOWN_DRAIN_TIMEOUT_S (default 30) to
  //     write its backlog to disk. What has not made it by then is dropped:
  //     it is a cache, and a stop that outlives the service manager's timeout
  //     loses every disk's cache instead of the newest few objects.
  //  2. All disks sync and record a clean shutdown, at the same time.
  // Each step is logged with its duration. Returns true if every disk is
  // clean; false also when a disk failed to open and so is not there to be
  // closed. Precondition: whatever else calls into this server (an RDMA server)
  // has been stopped. Afterwards the server can only be destroyed.
  // `started` is when the caller began stopping: the time it spent stopping
  // other request paths counts against the drain limit, because the flushers
  // were already writing the backlog then.
  bool Shutdown(std::chrono::steady_clock::time_point started =
                    std::chrono::steady_clock::now());
  // First step of a stop, before any request path is stopped: PUTs parked
  // until the flush of their key are answered kCacheFull, now and from here
  // on. Otherwise stopping a request path (it joins its threads) would wait
  // for those flushes, outside the drain limit. Stop() and Shutdown() do
  // this for the TCP path themselves; the owner of another request path (an
  // RDMA server) calls it before stopping that path. Idempotent.
  void CancelParkedPuts();
  // Cluster member list this node advertises for discovery (kMembers wire op):
  // "name=ip:port,name=ip:port,...". Clients query any node to learn the cluster.
  // May change while requests are served.
  void set_members(const std::string& members) {
    std::lock_guard<std::mutex> lock(members_mu_);
    members_ = members;
  }
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
    return group_.Healthy() && !ram_required_failed_ && !start_cancelled_ &&
           (!ram_ || ram_->healthy());
  }
  // The constructor saw its cancel_start flag set. The disks that opened are
  // open; Shutdown() closes them clean. Start() refuses.
  bool start_cancelled() const { return start_cancelled_; }
  // False once fewer than DFKV_DEGRADED_MIN_DISKS (default 1) disks are
  // left whose store has not failed for good. A failed store stores and
  // reads nothing more until the process is restarted; its keys go to the
  // remaining disks (DisksDegraded). Narrower than Healthy() on purpose: a
  // RAM tier that dropped an entry it could not flush also clears Healthy(),
  // but that node still serves every request.
  bool DisksHealthy() const { return group_.Healthy(); }
  // Some disk stores have failed for good and the node serves on the others.
  bool DisksDegraded() const {
    return group_.FailedDisks() > 0 && group_.Healthy();
  }
  size_t FailedDisks() const { return group_.FailedDisks(); }
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
  // accept() failures that made the TCP listener back off, and connections
  // refused because their handler thread could not be created.
  uint64_t TcpAcceptFailures() const { return accept_guard_.accept_failures(); }
  uint64_t TcpThreadStartFailures() const {
    return accept_guard_.thread_start_failures();
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
  // kDrain (DFKV_DRAIN): called with true to start draining, false to stop;
  // its status is the answer. Unset: kDrain is answered kInvalid.
  void set_drain_handler(std::function<Status(bool)> handler);
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
  // A GET's existence check before the RDMA server gives it a staging
  // buffer: a miss is counted as the read would have counted it.
  Status LookupForGet(const BlockKey& key);


  // Zero-copy server-side GET: read the block straight into `dst` (e.g. the RDMA
  // send buffer), no intermediate std::string. *out_len = bytes read. Updates
  // hit/miss + bytes_read metrics like a Range.
  Status RangeIntoForKey(const BlockKey& key, uint64_t offset,
                         uint64_t length, char* dst, size_t dst_cap,
                         size_t* out_len, size_t* value_len = nullptr);

  // RDMA direct PUT: `data` is an aligned registered buffer containing raw
  // value bytes. Writes without a payload-sized CPU copy. `batch_arrival` is
  // when the request batch this PUT came in arrived: a PUT parked for a flush
  // waits at most DFKV_RAM_PUT_WAIT_MS from then (default: from its wait).
  Status CacheDirectForKey(
      const BlockKey& key, char* data, size_t len, size_t cap,
      std::chrono::steady_clock::time_point batch_arrival = {});

  // RDMA direct GET: read an O_DIRECT-aligned superset into `io_buf`; *out_data
  // points inside that same buffer at the exact requested range so the RDMA layer
  // can scatter-send it without copying into sbuf.
  Status RangeDirectForKey(const BlockKey& key, uint64_t offset,
                           uint64_t length, char* io_buf, size_t io_cap,
                           const char** out_data, size_t* out_len,
                           size_t* value_len = nullptr);
  // Observe an arena hit without accounting. The prepared owner holds its pin
  // until the caller revokes remote access and commits, or fences and aborts.
  // Misses, dedicated values and registration fallbacks remain uncounted here.
  bool RamPinnedHitForKey(const BlockKey& key, uint64_t offset,
                          uint64_t length, PreparedRead* out);

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
  uint64_t ram_arena_bytes() const { return ram_ ? ram_->arena_bytes() : 0; }
  RamTier* ram_tier() const { return ram_.get(); }
  bool ram_ack_enabled() const { return ram_ack_enabled_; }

  // SIGHUP: read the tenant quotas file again (DiskCacheGroup::ReloadTenantQuotas).
  bool ReloadTenantQuotas(std::string* error) {
    return group_.ReloadTenantQuotas(error);
  }

 private:
  friend class KvNodeServerTestPeer;
  friend class KvNodeServerWiringTestPeer;
  // Reuse the storage group's engine factory in focused admission tests.
  explicit KvNodeServer(DiskCacheGroup::Options options,
                        const std::atomic<bool>* cancel_start = nullptr);
  static void FinishDiskRead(void* owner, uint64_t flight, bool committed,
                             Status result, size_t bytes_read,
                             double elapsed_sec, const char* data) noexcept;
  // A staged read under DFKV_RAM_PROMOTE=second that is not to be promoted.
  static void FinishUnpromotedDiskRead(
      void* owner, uint64_t flight, bool committed, Status result,
      size_t bytes_read, double elapsed_sec, const char* data) noexcept;
  void FinishStagedDiskRead(uint64_t flight, bool committed, Status result,
                            size_t bytes_read, double elapsed_sec,
                            const char* data, bool promote) noexcept;
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
  static void FinishPullRamRead(void* owner, uint64_t token, bool committed,
                                Status result, size_t bytes_read,
                                double elapsed_sec, const char* data) noexcept;
  void AcceptLoop();
  // Closes the listener and joins every handler thread; false if that had
  // been done already.
  bool StopRequests();
  // accepted_at stamps the accept() return so the first complete request frame
  // is due within DFKV_TCP_FIRST_REQ_MS of ACCEPT, not of handler scheduling.
  void Handle(int fd, std::chrono::steady_clock::time_point accepted_at);
  // What the environment asks of the RAM tier, read once before anything
  // slow starts. The tier itself is then built by StartRamTier on its own
  // thread while the disks open, and wired to this server by AdoptRamTier.
  struct RamTierPlan {
    bool enabled = false;
    bool write_back = true;
    bool ack_enabled = false;
    // DFKV_RAM_PROMOTE=second: a whole cold read is promoted only when the
    // key was read within promote_window_ms before.
    bool promote_second = false;
    uint64_t promote_window_ms = 30000;
    RamTier::Options options;
  };
  static RamTierPlan PlanRamTier(size_t disk_count);
  // The disk group is told to open its disks without activating them; the
  // constructor activates them once the RAM tier is ready too.
  static DiskCacheGroup::Options OpenedNotActivated(
      DiskCacheGroup::Options options);
  // The tier under construction should give up: the caller cancelled the
  // start, or a disk failed to open and the start is over anyway.
  bool RamStartUnwanted() const {
    return ram_start_abandoned_.load(std::memory_order_relaxed) ||
           (cancel_start_ != nullptr &&
            cancel_start_->load(std::memory_order_relaxed));
  }
  std::future<std::unique_ptr<RamTier>> StartRamTier();
  void AdoptRamTier(std::unique_ptr<RamTier> tier);
  // RAM tier write policy: write-back (PUT lands in arena, async flush) or
  // write-around (PUT direct to disk, arena filled by GET read-promotion).
  void set_ram_write_back(bool wb) { ram_write_back_ = wb; }
  void InitAdmission();   // read DFKV_PUT_INFLIGHT_LIMIT (0 = gate off)
  bool TryBeginPut();
  void EndPut();
  // Serializes the PUTs of one key, and of that key only: a PUT of another
  // key never waits for it. `lock` may be handed to the RAM tier, which
  // releases it before waiting for a flush; the key's entry goes with the
  // last holder.
  struct PutKeySlot {
    std::mutex mu;
    size_t users = 0;
  };
  class PutKeyLock {
   public:
    PutKeyLock() = default;
    PutKeyLock(KvNodeServer* owner, const BlockKey& key)
        : owner_(owner), key_(key) {}
    PutKeyLock(PutKeyLock&& other) noexcept
        : owner_(other.owner_), key_(other.key_),
          lock(std::move(other.lock)) {
      other.owner_ = nullptr;
    }
    PutKeyLock(const PutKeyLock&) = delete;
    PutKeyLock& operator=(const PutKeyLock&) = delete;
    PutKeyLock& operator=(PutKeyLock&&) = delete;
    ~PutKeyLock() {
      if (owner_) owner_->ReleasePutKey(key_, &lock);
    }
    std::unique_lock<std::mutex> lock;

   private:
    KvNodeServer* owner_ = nullptr;
    BlockKey key_{};
  };
  PutKeyLock LockPutKey(const BlockKey& key);
  void ReleasePutKey(const BlockKey& key, std::unique_lock<std::mutex>* lock);
  void ReapDoneLocked();  // join+erase finished handler threads; conn_mu_ held
  void InitTcpListenerConfig();
  std::atomic<size_t> cache_put_{0}, cache_hit_{0}, cache_miss_{0};
  std::atomic<size_t> exist_hit_{0}, exist_miss_{0};
  // kRetain: per-key outcomes by mode (0 SET, 1 REFRESH) and RetainCode.
  std::atomic<size_t> retain_keys_[2][8] = {};
  std::atomic<size_t> retain_requests_{0};
  Status ProcessRemovedWithin(const BlockKey& prefix_key, uint64_t count,
                              const char* payload, uint64_t payload_len,
                              std::string* out_data);
  Status ProcessRemovedWithinAges(const BlockKey& prefix_key, uint64_t count,
                                  const char* payload, uint64_t payload_len,
                                  std::string* out_data);
  // A PUT about to be acknowledged kOk (the recent-PUT memory, CLI-35).
  void NoteWrite(const BlockKey& key) {
    if (write_memory_) write_memory_->Record(key, RawMs());
  }
  Status ProcessRetain(const BlockKey& prefix_key, uint64_t count,
                       const char* payload, uint64_t payload_len,
                       std::string* out_data);
  // A SET on a write-back value still only in RAM is applied when its flush
  // commits (bounded; overflow answers DENIED). Under pending_retain_mu_.
  // ApplyPendingRetain takes the parked SET of a key: applied when the value
  // was written, forgotten when it was dropped unwritten or removed. Refused
  // and failed attempts that are tried again leave it parked.
  static constexpr size_t kPendingRetainMax = 1u << 20;
  mutable std::mutex pending_retain_mu_;  // also read by MetricsText()
  std::unordered_map<BlockKey, uint32_t, BlockKeyMixHash> pending_retain_;
  std::atomic<size_t> pending_retain_applied_{0};
  std::atomic<size_t> pending_retain_denied_{0};
  // The retention code of the applied SET; -1 when none was applied.
  int ApplyPendingRetain(const BlockKey& key, bool flushed);
  // The generation of the disk's value for a key (ValueMetadata), 0 if none.
  uint64_t DiskGeneration(const BlockKey& key) const;
  // Whether a whole-value cold read of `key` may be promoted into RAM. With
  // DFKV_RAM_PROMOTE=first (recent_reads_ null) always, without side
  // effects; with second only on a repeat read within the window. Asked
  // only for a read that would otherwise be promoted; `note` records the
  // read (once per read: false when it may still be given up and redone).
  bool PromoteRead(const BlockKey& key, bool note = true);
  void NoteRead(const BlockKey& key, bool promoted);
  static constexpr size_t kRecentReadsMax = 1u << 18;
  std::unique_ptr<RecentReads> recent_reads_;
  std::atomic<size_t> promote_deferred_{0};
  std::atomic<size_t> remove_ok_{0}, remove_miss_{0};
  // W has a distinct publication lock so a RAM flush can release only the
  // inner admission lock, not the PUT/REMOVE vs op-26 serialization.
  std::unique_lock<std::mutex> MemoryKey(const BlockKey& key) {
    const size_t bucket =
        BlockKeyMixHash{}(key) % memory_key_locks_.size();
    if (memory_key_expected_bucket_for_test_.load(
            std::memory_order_relaxed) == bucket) {
      std::lock_guard<std::mutex> lk(memory_key_test_mu_);
      memory_key_arrived_bucket_for_test_ = bucket;
      memory_key_test_cv_.notify_all();
    }
    return std::unique_lock<std::mutex>(memory_key_locks_[bucket]);
  }
  std::array<std::mutex, 64> memory_key_locks_;
  // The opt-in disk-barrier test proves op 26 reached this bucket's lock
  // before releasing storage; 64 is no valid bucket and disarms the hook.
  std::atomic<size_t> memory_key_expected_bucket_for_test_{64};
  std::mutex memory_key_test_mu_;
  std::condition_variable memory_key_test_cv_;
  size_t memory_key_arrived_bucket_for_test_ = 64;
  // Op-26 pauses before reading its joint M/W publication snapshot in a test.
  std::atomic<bool> pause_ages_for_test_{false};
  std::mutex ages_test_mu_;
  std::condition_variable ages_test_cv_;
  int ages_test_state_ = 0;
  void AfterRemovedAgesForTest();
  std::unique_ptr<RemoveMemory> remove_memory_;
  // Recent-PUT memory (DFKV_WRITE_MEMORY_ENTRIES, needs M; null = off).
  std::unique_ptr<WriteMemory> write_memory_;
  std::atomic<uint64_t> removed_within_ages_requests_{0};  // op 26 answered
  std::atomic<size_t> bytes_written_{0}, bytes_read_{0};
  // Read-side convoy collapse (env DFKV_READ_COALESCE=1; see read_coalescer.h).
  ReadCoalescer read_coalescer_;
  bool coalesce_enabled_ = envparse::EnvSwitch(
      "DFKV_READ_COALESCE", false, envparse::OffWhen::kStartsWithZero);
  // depth metrics: errors by op, live connections, sampled op latency
  std::atomic<size_t> put_io_err_{0}, get_io_err_{0}, invalid_ops_{0};
  // PUT admission gate: reserve before RAM admission or direct disk I/O.
  // Accepted requests retain their permit through commit/failure; background
  // flushes never take a permit. 0 = unlimited/off (--put-inflight-limit).
  size_t put_busy_limit_ = 0;
  bool ram_write_back_ = true;  // default write-back; DFKV_RAM_WRITE_MODE=writearound switches
  std::atomic<size_t> put_inflight_{0};
  std::atomic<size_t> put_busy_{0};
  // Cover the gap between RAM capacity bypass and the backing-store commit.
  // Flush workers and Remove never acquire these foreground-only locks.
  // Keys with a PUT in progress, in stripes; a stripe's mutex is held only
  // to find or drop a key's slot.
  struct PutKeyStripe {
    std::mutex mu;
    std::unordered_map<BlockKey, std::unique_ptr<PutKeySlot>, BlockKeyMixHash> keys;
    std::vector<std::unique_ptr<PutKeySlot>> free;  // unused slots to reuse
  };
  static constexpr size_t kPutKeyFreeSlots = 8;
  static constexpr size_t kPutKeyStripes = 256;
  PutKeyStripe put_keys_[kPutKeyStripes];
  std::atomic<uint64_t> put_key_lock_waits_{0}, put_key_lock_wait_us_{0};
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
  mutable std::mutex members_mu_;
  std::string members_;  // advertised cluster membership (kMembers)
  uint64_t max_request_payload_ = 0;  // 0 = resolve default lazily (see .cc)
  std::string node_id_, node_group_;       // identity for Prometheus labels (optional)
  std::chrono::steady_clock::time_point start_time_ = std::chrono::steady_clock::now();
  mutable std::mutex metrics_extension_mu_;
  std::function<Status(bool)> drain_handler_;  // under metrics_extension_mu_
  std::function<std::string()> metrics_extension_;
  struct RamFlushDiskMetrics {
    std::atomic<uint64_t> batches{0};
    std::atomic<uint64_t> objects{0};
    std::atomic<uint64_t> bytes{0};
    std::atomic<uint64_t> failures{0};
    LatencyHist latency;
  };
  std::vector<std::unique_ptr<RamFlushDiskMetrics>> ram_flush_disks_;

  // Declared BEFORE group_ on purpose: members are constructed in declaration
  // order, and the arena has to start faulting in before the disks start
  // opening for the two to overlap. Empty again once the constructor is done.
  const std::atomic<bool>* cancel_start_ = nullptr;
  std::atomic<bool> ram_start_abandoned_{false};
  RamTierPlan ram_plan_;
  std::future<std::unique_ptr<RamTier>> ram_starting_;
  DiskCacheGroup group_;
  // Optional RAM hot tier (P3). Declared AFTER group_ so it is destroyed FIRST:
  // ~RamTier stops+joins the flusher, whose callback calls group_.Cache, before
  // group_ is torn down. nullptr unless DFKV_RAM_TIER is enabled. In write-back
  // mode, PUT lands in RAM (sync-visible) and flushes to group_; write-around
  // populates RAM through born-durable GET promotion. GET checks RAM first.
  std::unique_ptr<RamTier> ram_;
  bool ram_ack_enabled_ = false;
  bool ram_required_failed_ = false;
  bool start_cancelled_ = false;
  int listen_fd_ = -1;
  int port_ = 0;
  std::atomic<bool> running_{false};
  uint64_t shutdown_drain_timeout_s_ = 30;  // DFKV_RAM_SHUTDOWN_DRAIN_TIMEOUT_S
  bool shutdown_done_ = false;       // Shutdown() ran; shutdown_clean_ is its result
  bool shutdown_clean_ = false;
  std::atomic<size_t> accept_count_{0};
  std::thread accept_thread_;
  net::AcceptGuard accept_guard_{"tcp listener"};
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
