/* DiskSlabStore — a disk-backed KV cache node built on SlabAllocator.
 *
 * Replaces the "one file per block" KVStore geometry (whose tmp-leak / ENOSPC-
 * dead-end / unbounded-inode / lock-held-unlink pathologies all stem from that
 * base) with a fixed pool of pre-allocated EXTENT files carved into slots by the
 * media-agnostic SlabAllocator. A compact slots.tbl records which key occupies
 * each slot so the index rebuilds on restart -- keeping cache warmth across a
 * rolling upgrade without the per-block file churn.
 *
 * Payload I/O defaults to O_DIRECT in the production daemon; the aligned
 * transport paths avoid the page cache and a bounded buffered mode remains an
 * explicit diagnostic option. Extent descriptors stay open for the store
 * lifetime. Each table record is CRC-checked and dirty-epoch recovery fails
 * safe by discarding uncommitted cache warmth rather than serving torn bytes. */
#ifndef DFKV_DISK_SLAB_STORE_H_
#define DFKV_DISK_SLAB_STORE_H_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cache/slab_allocator.h"
#include "cache/store_engine.h"
#include "common/kv_types.h"
#include "common/status.h"

namespace dfkv {

class DiskSlabStoreTestPeer;

class DiskSlabStore : public StoreEngine {

 public:
  struct Options {
    std::string dir;
    uint64_t capacity_bytes = (1ull << 30);
    uint64_t extent_bytes = (1ull << 30);    // 1 GiB per extent file
    // Slot-size quantum: every class slot_size is a multiple of this, so the
    // per-extent slot count (and thus slots.tbl size) is bounded. Real KV blocks
    // are MiB-scale, so 1 MiB is a fine default; tests use small values.
    uint64_t slot_granularity = (1ull << 20);
    // O_DIRECT on the aligned data paths (CacheDirect writes, RangeDirect reads,
    // async prep): zero page-cache footprint -- on GPU nodes the cache/dirty
    // growth of buffered I/O competes with training/inference memory, so the
    // DEPLOYMENT default is direct (DiskCacheGroup: DFKV_SLAB_WRITE=buffered
    // opts out); burst absorption comes from the explicit RAM tier instead.
    // Extents are fallocate'd in this mode: DIO OVERWRITES parallelize on XFS,
    // allocating writes into holes would serialize on the exclusive iolock.
    // If the filesystem rejects O_DIRECT (tmpfs in tests/CI), the store falls
    // back to buffered -- DirectWritesActive() reports the resolved truth.
    bool direct_writes = false;
    // Background free-slot reclaimer cadence (ms; 0 = off). Keeps a demand-
    // driven headroom of free slots per active class so Put's fast path stays
    // pop-free-slot; without it every Put under capacity pressure runs the
    // CLOCK eviction sweep inline under the store lock (the measured write-
    // path serialization on a full store).
    uint32_t reclaim_interval_ms = 50;
    // slots.tbl fdatasync cadence (ms; 0 = off). Bounds the crash window in
    // which a REUSED slot's new payload is on disk while the reused record is
    // still page-cache-only -- rebuild would then resurrect the PREVIOUS
    // occupant's key pointing at the new occupant's bytes. Payload writes give
    // no such window in direct mode (DIO is durable at return); this closes
    // the record side to <= the cadence instead of the ~30s dirty expiry.
    uint32_t table_sync_ms = 100;
  };

  // One store's runtime counters (see KvNodeServer's dfkv_slab_* metrics).
  struct Stats {
    uint64_t dio_write_fallbacks = 0;
    uint64_t dio_read_fallbacks = 0;
    uint64_t table_syncs = 0;
    uint64_t bind_wipes = 0;
    uint64_t steals = 0;
    uint64_t cold_steals = 0;
    uint64_t watermark_evictions = 0;
    uint64_t watermark_extent_clears = 0;
    uint64_t watermark_ticks = 0;
    uint64_t watermark_active = 0;
    uint64_t watermark_last_tick_us = 0;
    uint64_t watermark_max_tick_us = 0;
    uint64_t watermark_max_extents_per_tick = 0;
    uint64_t extent_returns = 0;
    uint64_t deferred_removes = 0;
    uint64_t inflight = 0;
    uint64_t prep_holds = 0;
    uint64_t reclaimed_slots = 0;
    uint64_t rebalanced_extents = 0;
    uint64_t batched_writes = 0;
    uint64_t uring_write_batches = 0;
    uint64_t metadata_io_errors = 0;
    uint64_t unclean_resets = 0;
    uint64_t eviction_record_clears = 0;
    uint64_t record_writes = 0;
    uint64_t table_rebuilt = 0;
    uint64_t rebuild_corrupt_records = 0;
    uint64_t rebuild_rejected_records = 0;
    uint64_t rebuild_scanned_bytes = 0;
    uint64_t rebuild_scan_chunks = 0;
    uint64_t rebuild_sparse_ranges = 0;
    uint64_t rebuild_sequential_fallbacks = 0;
    uint64_t rebuild_mmap_scans = 0;
    uint64_t capacity_bytes = 0;
    uint64_t allocated_bytes = 0;
    uint64_t payload_bytes = 0;
    uint64_t allocator_objects = 0;
    uint64_t committed_objects = 0;
    uint64_t class_count = 0;
    uint64_t bound_extents = 0;
    uint64_t pool_extents = 0;
    uint64_t failed_disks = 0;
    bool failed = false;
  };

  // Opens (or creates) the store under Options::dir, pre-allocating extents and
  // rebuilding the index from slots.tbl. `*ok` (nullable) reports success; on a
  // fatal open error the store is left empty and every op returns kIOError.
  explicit DiskSlabStore(Options opt, bool* ok = nullptr);
  ~DiskSlabStore();

  DiskSlabStore(const DiskSlabStore&) = delete;
  DiskSlabStore& operator=(const DiskSlabStore&) = delete;

  Status Cache(const BlockKey& key, const void* data, size_t len) override;
  // Writes an already aligned transport buffer without a value envelope.
  // Direct mode uses O_DIRECT; the bounded fallback preserves raw bytes.
  Status CacheDirect(const BlockKey& key, char* data, size_t len,
                     size_t cap) override;
  // Batched CacheDirect: ONE lock hold allocates every slot, payloads go out
  // unlocked (io_uring one-submit when built+enabled, else a sequential loop),
  // ONE lock hold commits. Per-item semantics identical to CacheDirect.
  std::vector<Status> CacheDirectBatch(const std::vector<CacheBatchItem>& items) override;
  Status Range(const BlockKey& key, uint64_t offset, uint64_t length,
               std::string* out, size_t* value_len = nullptr) override;
  Status RangeInto(const BlockKey& key, uint64_t offset, uint64_t length,
                   char* dst, size_t dst_cap, size_t* out_len,
                   size_t* value_len = nullptr) override;
  // Reads into the caller's bounded registered buffer and returns the payload
  // slice directly; direct mode uses its aligned O_DIRECT window.
  Status RangeDirect(const BlockKey& key, uint64_t offset, uint64_t length,
                     char* io_buf, size_t io_cap, const char** out_data,
                     size_t* out_len, size_t* value_len = nullptr) override;
  // Async prep pins the resolved slot and returns a move-only lease over both
  // the dup'd extent descriptor and that pin. Abandoning or completing the read
  // destroys the lease and balances the hold without a manual release protocol.
  Status RangeDirectPrep(const BlockKey& key, uint64_t offset, uint64_t length,
                         size_t io_cap, ReadLease* out) override;
  bool IsCached(const BlockKey& key) const override;
  Status Lookup(const BlockKey& key, ValueMetadata* out) const override;
  Status Remove(const BlockKey& key) override;

  size_t Count() const override;
  uint64_t UsedBytes() const override;
  uint64_t TenantUsedBytes(uint64_t tenant_hash) const override;
  uint64_t Capacity() const;
  uint64_t Evictions() const override;
  uint64_t EvictedBytes() const override;
  const std::string& Dir() const override { return opt_.dir; }
  uint64_t TableRebuilt() const { return table_rebuilt_; }  // records restored at open
  // Resolved I/O mode: direct_writes requested AND the filesystem took O_DIRECT.
  bool DirectWritesActive() const { return !extent_dio_fds_.empty(); }
  Stats GetStats() const;
  std::vector<SlabAllocator::ClassStat> ClassStats() const;
  bool Healthy() const override {
    return ok_ && !metadata_failed_.load(std::memory_order_acquire);
  }
  const std::string& StartupError() const override { return startup_error_; }

 private:
  friend class DiskSlabStoreTestPeer;
  static constexpr size_t kRecBytes = 64;

  bool ValidateOptions();
  bool SetStartupError(std::string error);
  bool OpenOrInit();
  bool Rebuild();
  bool OpenEpochState(bool fresh);
  bool ResetUncleanEpoch();
  bool WriteEpochState(uint32_t state, uint64_t epoch);
  bool MarkDirtyEpoch();
  bool MarkCleanEpoch();
  bool SyncPayloads();
  bool WriteRecord(const SlabAllocator::SlotRef& r, const BlockKey& key,
                   uint32_t payload_len, bool valid);
  bool SyncTable(uint64_t seen);
  bool FailMetadata();
  bool WritePayload(const SlabAllocator::SlotRef& r, const void* data, size_t len);
  // O_DIRECT payload write from the caller's ALIGNED buffer (CacheDirect path):
  // zeroes the padding bytes in place (cap allows) and pwrites the 4 KiB-rounded
  // length via the extent's DIO fd. Caller pre-checked alignment/cap fit.
  bool WritePayloadDirect(const SlabAllocator::SlotRef& r, char* data, size_t len);
  void CommitPayloadLocked(const std::string& key, uint32_t payload_len);
  uint32_t ErasePayloadLocked(const std::string& key);
  // Shared allocate/pin -> unlocked payload write -> commit skeleton for
  // Cache/CacheDirect; `write_payload` does just the payload I/O for the slot.
  template <typename WriteFn>
  Status CacheImpl(const BlockKey& key, size_t len, const WriteFn& write_payload);
  struct PutFlight {
    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    Status result = Status::kIOError;
  };
  static void CompletePut(const std::shared_ptr<PutFlight>& flight,
                          Status result);
  static Status WaitPut(const std::shared_ptr<PutFlight>& flight);
  // Resolve fn's slot + committed payload length and acquire it for an unlocked
  // read (pin + inflight count, under mu_). False = miss (absent, or a write
  // still in flight). Balanced by a Release{...}Locked call.
  bool AcquireForRead(const BlockKey& key, const std::string& fn,
                      SlabAllocator::SlotRef* ref, uint32_t* plen);
  // Drop one in-flight hold on fn (unpin + count). The LAST releaser performs
  // the durable invalidation for a Remove that arrived while I/O was active.
  // SlabAllocator itself also defers pinned removal, but this caller-level state
  // is still required to order the slots.tbl record before physical reuse.
  // Call with mu_ held.
  bool ReleaseInflightLocked(const BlockKey& key, const std::string& fn);
  static void ReleaseReadLeaseThunk(void* owner, uint64_t token) noexcept;
  void ReleaseReadLease(uint64_t token) noexcept;
  uint64_t TableOffset(uint32_t extent, uint32_t slot) const {
    return static_cast<uint64_t>(extent) * max_slots_per_extent_ * kRecBytes +
           static_cast<uint64_t>(slot) * kRecBytes;
  }

  Options opt_;
  uint32_t num_extents_ = 0;
  uint32_t max_slots_per_extent_ = 0;
  std::unique_ptr<SlabAllocator> alloc_;
  std::vector<int> extent_fds_;      // resident, one per extent (buffered)
  std::vector<int> extent_dio_fds_;  // O_DIRECT twins (only when direct_writes)
  int table_fd_ = -1;                // slots.tbl
  int state_fd_ = -1;                // slab_state clean/dirty epoch marker
  uint64_t run_epoch_ = 0;
  bool unclean_start_ = false;
  // Outstanding asynchronous leases. The opaque id remains entirely internal:
  // ReadLease destruction routes it back through ReleaseReadLeaseThunk.
  struct PrepHold { BlockKey key; std::string fn; };
  std::unordered_map<uint64_t, PrepHold> prep_holds_;
  uint64_t prep_token_seq_ = 0;
  // key.Filename() -> its true payload length (slot_size >= this). Rebuilt on
  // open from the table; the allocator only tracks slot size.
  std::unordered_map<std::string, uint32_t> payload_len_;
  std::unordered_map<uint64_t, uint64_t> tenant_used_bytes_;
  // Keys with an unlocked pread/pwrite in flight (value = op count), and keys
  // whose Remove arrived during that window (executed by the last releaser).
  std::unordered_map<std::string, uint32_t> inflight_;
  std::unordered_set<std::string> deferred_remove_;
  // Signaled by the last in-flight releaser right after it executes a deferred
  // Remove (mu_ held): a re-PUT of that key waits out the window (bounded)
  // thereon instead of failing hard against the still-resident slot.
  std::condition_variable remove_cv_;
  // A reserved-but-uncommitted key has exactly one leader. Scalar and batch
  // duplicates join this flight and observe its real commit result.
  std::unordered_map<std::string, std::shared_ptr<PutFlight>> put_flights_;
  // Guards the in-memory maps and the allocate/lookup+pin step ONLY -- never
  // held across payload I/O (a pin protects the slot from eviction in the
  // unlocked window, inflight_/deferred_remove_ protect it from Remove;
  // payload_len_ install after the write is the reader-visible commit).
  mutable std::mutex mu_;
  uint64_t committed_payload_bytes_ = 0;  // under mu_
  bool ok_ = false;
  std::atomic<bool> metadata_failed_{false};
  std::string startup_error_;
  uint64_t table_rebuilt_ = 0;
  uint64_t evicted_bytes_ = 0;
  uint64_t deferred_remove_total_ = 0;  // under mu_
  std::atomic<uint64_t> dio_write_fallbacks_{0};
  std::atomic<uint64_t> dio_read_fallbacks_{0};
  std::atomic<uint64_t> table_syncs_{0};
  std::atomic<uint64_t> bind_wipes_{0};
  std::atomic<uint64_t> metadata_io_errors_{0};
  std::function<bool()> payload_write_hook_for_test_;
  std::function<bool()> record_write_hook_for_test_;
  // Test-only seam (never set in production; same discipline as the hooks
  // above): when set, replaces EVERY CQE reap of the batched io_uring write
  // path -- the blocking wait AND the post-failure non-blocking drain -- so
  // tests can inject a hard reap failure. The void* signature keeps liburing
  // types out of the header (build-flag-independent class layout).
  std::function<int(void* ring, void* cqe)> uring_reap_hook_for_test_;
  std::atomic<uint64_t> unclean_resets_{0};
  std::atomic<uint64_t> eviction_record_clears_{0};
  std::atomic<uint64_t> watermark_extent_clears_{0};
  // slots.tbl sync thread (see Options::table_sync_ms): fdatasync only when
  uint64_t rebuild_corrupt_records_ = 0;
  uint64_t rebuild_rejected_records_ = 0;
  uint64_t rebuild_scanned_bytes_ = 0;
  uint64_t rebuild_scan_chunks_ = 0;
  uint64_t rebuild_sparse_ranges_ = 0;
  uint64_t rebuild_mmap_scans_ = 0;
  uint64_t rebuild_sequential_fallbacks_ = 0;
  // records were written since the last cycle.
  std::atomic<uint64_t> record_writes_{0};
  uint64_t synced_marker_ = 0;  // sync-thread-local snapshot of record_writes_
  std::thread sync_thread_;
  std::condition_variable sync_cv_;
  std::mutex sync_mu_;
  bool sync_stop_ = false;
  // Background free-slot reclaimer (see Options::reclaim_interval_ms): runs
  // ReclaimTick every interval, evicting ahead of demand in bounded batches
  // and rebalancing extents from cold classes to hot ones on a full store.
  void ReclaimTick(
      const std::function<void()>& after_watermark_evict = {});
  // Rebalance rate cap: extents moved per tick per hot class. Each move evicts
  // a donor extent's residents + wipes its table region under the lock, so it
  // is deliberately slow-drip (converges in seconds at the 50 ms tick).
  static constexpr size_t kGrowExtentsPerTick = 2;
  std::atomic<uint64_t> reclaimed_{0};
  std::atomic<uint64_t> rebalanced_{0};
  std::atomic<uint64_t> batched_writes_{0};
  std::atomic<uint64_t> uring_write_batches_{0};
  // Kept in every build so the class layout does not depend on io_uring.
  bool uring_write_enabled_ = false;
  // The batched-write submission ring is thread_local (see disk_slab_store.cc):
  // one per flush worker, no shared lock across the blocking CQE wait.
  std::vector<uint64_t> reclaim_last_puts_;  // reclaim-thread-local puts snapshot
  uint64_t evict_high_bytes_ = 0;  // high crossing activates eviction
  uint64_t evict_low_bytes_ = 0;   // active eviction drains to this target
  size_t evict_max_extents_per_tick_ = 1;
  std::atomic<bool> watermark_active_{false};
  std::atomic<uint64_t> watermark_ticks_{0};
  std::atomic<uint64_t> watermark_last_tick_us_{0};
  std::atomic<uint64_t> watermark_max_tick_us_{0};
  std::thread reclaim_thread_;
  std::condition_variable reclaim_cv_;
  std::mutex reclaim_mu_;
  bool reclaim_stop_ = false;
};

}  // namespace dfkv

#endif  // DFKV_DISK_SLAB_STORE_H_
