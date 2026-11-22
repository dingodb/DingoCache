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

#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cache/slab_allocator.h"
#include "cache/store_engine.h"
#include "cache/new_put_demand_latch.h"
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
    // What happens to slots.tbl after an unclean exit, i.e. when slab_state
    // still says dirty at startup.
    //   kOff      reset the table: the cache starts empty (every release up to
    //             2.35 did this unconditionally). The default.
    //   kAuto     keep the table when the previous process demonstrably died
    //             with the kernel -- and so the page cache holding its last
    //             record writes -- still up: same boot id, same unique mount
    //             id, same directory, its trust token still on the tmpfs, a
    //             supported local filesystem, and a clean fdatasync of the
    //             table and every extent now. Anything short of that resets
    //             as kOff does.
    //   kBootOnly as kAuto, but a run that recorded no unique mount id (a
    //             kernel before Linux 6.8) is accepted on its boot id. A
    //             filesystem that died and was mounted again without a reboot
    //             is then only caught if the old process noticed -- an I/O
    //             error, or a table sync in progress -- before it went away
    //             (either takes the token back).
    // DiskCacheGroup resolves DFKV_SLAB_WARM_RESTART into this.
    enum class WarmRestart { kOff, kAuto, kBootOnly };
    WarmRestart warm_restart = WarmRestart::kOff;
    // Where trust tokens go. A token says "the table of this directory may be
    // kept"; a run holds one only while it can vouch for the table. It has to
    // vanish with the kernel's page cache and must not depend on the cache
    // disk, which is the thing that fails: a tmpfs. Empty = /dev/shm, used
    // only if it really is a tmpfs; tests point this elsewhere.
    std::string trust_token_dir;
    // Keeping a table also requires a filesystem known to keep written data
    // with the local kernel (xfs, ext4, tmpfs). Tests that run on whatever
    // the build machine offers turn the check off.
    bool require_local_filesystem = true;
    // Tests: make the start of the n-th background thread of Activate() fail
    // as a refused thread creation does (1 = slab-sync, 2 = slab-reclaim).
    int fail_thread_start_for_test = 0;
    // Leave a store that is found clean un-activated: see Activate().
    // DiskCacheGroup sets this and activates its disks together. The default
    // activates at the end of the constructor, as a store used on its own
    // expects.
    bool defer_activation = false;
    // Identity of this kernel boot and of the mount holding `dir`. Read from
    // the system when empty; tests set them to stage a reboot or a remount.
    std::optional<std::array<uint8_t, 16>> boot_id;
    std::optional<uint64_t> mount_id;
    // Stands in for the fallocate that reserves slots.tbl at open: returns 0,
    // or -1 with errno set. Tests use it to stage a filesystem that has no
    // room for the table, or none that can preallocate.
    std::function<int(int fd, uint64_t bytes)> table_allocate_for_test;
  };

  // One store's runtime counters (see KvNodeServer's dfkv_slab_* metrics).
  struct Stats {
    uint64_t dio_write_fallbacks = 0;
    // PUTs answered kCacheFull because the allocator had no slot to give:
    // everything evictable was pinned or protected by retention.
    uint64_t put_no_slot = 0;
    // PUTs answered kCacheFull because the key could not be taken yet: it
    // still holds a slot from an unfinished operation, or its deferred
    // removal outlasted the wait bound.
    uint64_t put_busy = 0;
    uint64_t dio_read_fallbacks = 0;
    uint64_t table_syncs = 0;
    uint64_t bind_wipes = 0;
    uint64_t steals = 0;
    uint64_t cold_steals = 0;
    uint64_t watermark_evictions = 0;
    uint64_t watermark_extent_clears = 0;
    // Global slot reclamation and durable invalidation work.
    uint64_t watermark_slot_removals = 0;
    uint64_t watermark_slot_record_clears = 0;
    uint64_t watermark_record_clear_bytes = 0;
    uint64_t watermark_freed_slot_bytes = 0;
    uint64_t watermark_scan_visits = 0;
    uint64_t watermark_max_victims_per_tick = 0;
    uint64_t watermark_max_visits_per_tick = 0;
    uint64_t watermark_ticks = 0;
    uint64_t watermark_active = 0;
    uint64_t watermark_last_tick_us = 0;
    uint64_t watermark_max_tick_us = 0;
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
    uint64_t warm_restarts = 0;
    uint64_t trust_tokens_held = 0;  // 1 while this store holds its token
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
    // This open (not exported as metrics; the same numbers are in the
    // per-disk "opened in" log line). Extents whose fallocate was left out
    // because the previous run closed clean and the file had every block, and
    // the time spent opening the extents, scanning slots.tbl, and installing
    // what it held.
    uint64_t open_fallocate_skipped = 0;
    uint64_t open_extents_us = 0;
    uint64_t open_scan_us = 0;
    uint64_t open_restore_us = 0;
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

  // A narrow locked view: no allocator class enumeration or temporary vectors.
  struct WatermarkObservation {
    uint64_t watermark_high_observations;
    uint64_t watermark_last_high_bytes;
    uint64_t watermark_peak_high_bytes;  // lifetime sampled peak, not phase proof
    uint64_t run_epoch;
    uint64_t capacity_bytes;
    uint64_t watermark_high_bytes;
    uint64_t watermark_low_bytes;
    uint64_t reclaim_interval_ms;
    uint64_t watermark_active;
  };
  WatermarkObservation GetWatermarkObservation() const {
    std::lock_guard<std::mutex> lk(mu_);
    return {watermark_high_observations_, watermark_last_high_bytes_,
            watermark_peak_high_bytes_, run_epoch_,
            static_cast<uint64_t>(opt_.extent_bytes) * num_extents_,
            evict_high_bytes_, evict_low_bytes_, opt_.reclaim_interval_ms,
            watermark_active_.load(std::memory_order_relaxed) ? 1u : 0u};
  }

  // Opens (or creates) the store under Options::dir, pre-allocating extents and
  // rebuilding the index from slots.tbl. An interrupted first initialization
  // resumes only with a matching initialization marker, absent slab_state,
  // zeroed table and strictly validated directory/extent layout; unknown and
  // previously populated layouts are never auto-recreated. `*ok` (nullable)
  // reports success; on fatal open error every op returns kIOError and
  // StartupError() includes the failing syscall's errno when applicable.
  explicit DiskSlabStore(Options opt, bool* ok = nullptr);
  ~DiskSlabStore();
  // Stops the background threads, syncs payloads and table, and writes the
  // clean marker. Split from the destructor so a server can have its disks
  // clean before it spends time releasing memory; see StoreEngine::Close.
  // A store that was never activated only syncs what its open wrote and
  // leaves slab_state as found, which is clean; it reports clean.
  // Contract: every Cache/CacheDirect/batch write has returned before this is
  // called, and none starts afterwards. The clean marker tells the next start
  // that nothing of this run is still being written (it then skips the
  // per-extent fallocate that would wait for such a write). A payload write
  // found in flight here is logged and the marker is not written: the next
  // start takes the unclean path. Returns whether the marker was written.
  bool Close() override;

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

  // Retention (DFKV_RETAIN_*). Only committed keys can be retained; a SET or
  // upgrade rewrites the slot record so protection survives a restart.
  bool RetainEnabled() const override { return retain_enabled_; }
  uint8_t Retain(const BlockKey& key, uint32_t ttl_s, uint8_t mode,
                 uint32_t now_s) override;
  uint32_t RetainHeadroomPct() const override {
    return alloc_ ? alloc_->RetentionHeadroomPct() : 0;
  }
  uint64_t MaxValueBytes() const override { return opt_.extent_bytes; }
  uint32_t RetainMaxTtl() const override {
    return retain_enabled_ ? retain_max_ttl_s_ : 0;
  }
  SlabAllocator::RetentionStats RetentionStats() const {
    return alloc_ ? alloc_->GetRetentionStats()
                  : SlabAllocator::RetentionStats{};
  }
  const char* EngineName() const override { return "slab"; }
  ::dfkv::RetentionStats GetRetentionStats() const override {
    return RetentionStats();
  }
  bool Healthy() const override {
    return ok_ && !metadata_failed_.load(std::memory_order_acquire);
  }
  // Open and activate are two steps.
  //
  // The open (the constructor) reads the state, rebuilds the index and
  // changes nothing a cleanly closed run left behind: slab_state stays as
  // found, no token is written, no thread runs. It may allocate missing
  // extent blocks and clear table records that fail their checks, as every
  // open may; both are repeatable. An opened store answers Count(), stats and
  // health; every data operation is refused (kIOError, a miss) and nothing
  // can write to it.
  //
  // Activate() makes the store serve: it records this run's identity and the
  // dirty state durably, grants the trust token, and only then starts the
  // sync and reclaim threads and admits operations. From that moment an exit
  // without Close() is an unclean exit, exactly as before the split; before
  // it, a kill leaves the directory as the previous run closed it.
  //
  // A store found dirty (the previous run did not close) is recovered or
  // reset by its open, which rewrites slab_state and withdraws that run's
  // token; it is activated by the constructor whatever defer_activation
  // says. Idempotent; false if the dirty state could not be recorded (the
  // store is then unhealthy) or the store is closed.
  bool Activate() override;
  // Starts one of Activate()'s background threads (which: 1 sync, 2
  // reclaim). False, with the store failed, if the system refuses it.
  template <typename Body>
  bool StartThread(std::thread* thread, int which, const char* name,
                   Body body);
  void StopThreads();
  bool Activated() const {
    return activated_.load(std::memory_order_acquire);
  }
  const std::string& StartupError() const override { return startup_error_; }
  // Why this start reset the table after an unclean exit; empty if it did not.
  const std::string& UncleanResetReason() const { return unclean_reset_reason_; }

  // For a fatal-signal handler: takes back the trust token of every open
  // store, so that the next start resets its table. A process that crashed on
  // its own may have misbehaved before it died, and payloads carry no checksum
  // to catch that. Async-signal-safe: one unlink(2) per store on prepared
  // paths, on a tmpfs, so it neither blocks on nor depends on the cache disks.
  static void RevokeAllTrustForCrash() noexcept;

 private:
  friend class DiskSlabStoreTestPeer;
  static constexpr size_t kRecBytes = 64;

  bool ValidateOptions();
  bool SetStartupError(std::string error);
  bool SetStartupErrno(std::string error, int saved_errno);
  bool ReserveTable(uint64_t bytes, bool creating);
  bool ClearInitMarker();
  bool OpenOrInit();
  // *inconsistent is set when the records contradict each other (the same key
  // twice, mixed slot sizes in one extent, ...), as opposed to an I/O failure.
  bool Rebuild(bool* inconsistent = nullptr);
  bool OpenEpochState(bool fresh);
  bool LockDirectory();
  bool RecoverUnclean();
  // nullptr when every recorded condition for keeping the table holds;
  // otherwise the reason it has to be reset.
  const char* WhyNotWarm() const;
  bool SyncForWarmRestart();
  bool ResetUncleanEpoch();
  // with_identity stores this boot and mount in the record: only a dirty state
  // written after the table was rebuilt may carry them.
  bool WriteEpochState(uint32_t state, uint64_t epoch, bool with_identity);
  // The trust token on tmpfs: present, and carrying this run's epoch, exactly
  // while the run vouches for the table (see Options::trust_token_dir).
  bool GrantTrust();
  // forfeit: for good -- this run will not vouch for the table again.
  // withdraw_identity_if_stuck: if the token cannot be removed, take the
  // identity out of slab_state instead. Only valid while that state is a
  // dirty one carrying an identity; never after the clean marker was written,
  // which it would overwrite. False if the token is still there and nothing
  // was withdrawn in its place.
  bool RevokeTrust(bool forfeit, bool withdraw_identity_if_stuck);
  bool RevokeTrustLocked(bool forfeit, bool withdraw_identity_if_stuck);
  bool TrustTokenMatches(uint64_t epoch) const;
  // fdatasync(table_fd_) with the token taken back for its duration: the
  // kernel reports a writeback error once, to this very call, and a process
  // killed inside it never gets to act on that.
  bool SyncTableFile();
  bool MarkDirtyEpoch();
  bool MarkCleanEpoch();
  bool SyncPayloads();
  // Bytes 44..51 carry retain_until/retain_ttl_s (zero = none). They were
  // always zero and CRC-covered, so older binaries validate and ignore them.
  bool WriteRecord(const SlabAllocator::SlotRef& r, const BlockKey& key,
                   uint32_t payload_len, bool valid,
                   uint32_t retain_until = 0, uint32_t retain_ttl_s = 0);
  bool SyncTable(uint64_t seen);
  // Latches the store as failed; always returns false. `op` completes "cannot
  // ..." in the one log line the first failure writes, `err` is its errno
  // (0 = none applies).
  bool FailMetadata(const char* op, int err = errno);
  bool WritePayload(const SlabAllocator::SlotRef& r, const void* data, size_t len);
  // O_DIRECT payload write from the caller's ALIGNED buffer (CacheDirect path):
  // zeroes the padding bytes in place (cap allows) and pwrites the 4 KiB-rounded
  // length via the extent's DIO fd. Caller pre-checked alignment/cap fit.
  bool WritePayloadDirect(const SlabAllocator::SlotRef& r, char* data, size_t len);
  void CommitPayloadLocked(const BlockKey& key, uint32_t payload_len);
  uint32_t ErasePayloadLocked(const BlockKey& key);
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
  // the physical removal for a Remove that arrived while I/O was active. The
  // record itself is already invalid by then (Remove invalidates a committed
  // key before it returns, and a cancelled write never writes one); it is
  // written once more here, which orders it before the slot can be reused.
  // SlabAllocator itself also defers pinned removal, but this caller-level
  // state is still required for that ordering. Call with mu_ held.
  bool ReleaseInflightLocked(const BlockKey& key, const std::string& fn);
  static void ReleaseReadLeaseThunk(void* owner, uint64_t token) noexcept;
  void ReleaseReadLease(uint64_t token) noexcept;
  uint64_t TableOffset(uint32_t extent, uint32_t slot) const {
    return static_cast<uint64_t>(extent) * max_slots_per_extent_ * kRecBytes +
           static_cast<uint64_t>(slot) * kRecBytes;
  }

  Options opt_;
  bool retain_enabled_ = false;
  uint32_t retain_max_ttl_s_ = 0;
  uint32_t num_extents_ = 0;
  uint32_t max_slots_per_extent_ = 0;
  std::vector<char> zero_extent_records_;
  std::unique_ptr<SlabAllocator> alloc_;
  std::vector<int> extent_fds_;      // resident, one per extent (buffered)
  std::vector<int> extent_dio_fds_;  // O_DIRECT twins (only when direct_writes)
  int table_fd_ = -1;                // slots.tbl
  int state_fd_ = -1;                // slab_state clean/dirty epoch marker
  uint64_t run_epoch_ = 0;
  bool initializing_ = false;        // validated on-disk init marker to clear
  bool unclean_start_ = false;
  // This boot / mount, and what the state file recorded for the previous run
  // (all-zero / 0 = unknown or deliberately withheld).
  std::array<uint8_t, 16> boot_id_{};
  uint64_t mount_id_ = 0;
  std::array<uint8_t, 16> stored_boot_id_{};
  uint64_t stored_mount_id_ = 0;
  uint64_t stored_dir_ino_ = 0;
  int dir_fd_ = -1;                  // holds the directory lock
  bool dir_locked_ = false;
  uint64_t dir_dev_ = 0, dir_ino_ = 0;
  std::string token_path_;           // empty: no usable tmpfs for tokens
  std::string unclean_reset_reason_;
  std::mutex state_mu_;              // serializes slab_state writes
  std::atomic<bool> identity_recorded_{false};  // slab_state carries it now
  mutable std::mutex trust_mu_;      // token + the fields below
  bool trust_forfeited_ = false;     // this run no longer vouches for the table
  int table_syncs_in_flight_ = 0;    // the token is absent while > 0
  bool trust_token_held_ = false;    // the token file exists, written by us
  bool trust_failure_logged_ = false;
  std::atomic<bool> buffered_payload_written_{false};  // extents may be dirty
  // Cache/CacheDirect/batch calls between issuing a payload write and its
  // completion or failure. Writers only: inflight_ also counts readers and
  // prepared-read leases, which Close() has no reason to wait out.
  std::atomic<uint32_t> payload_writers_{0};
  // Set by Activate() once the dirty state is durable; never cleared.
  std::atomic<bool> activated_{false};
  // What every data operation asks first.
  bool Serving() const {
    return Healthy() && activated_.load(std::memory_order_acquire);
  }
  bool MayMutate();
  Status NoSlotStatusLocked(bool resident);
  Status BusyStatus();
  int crash_slot_ = -1;              // RevokeAllTrustForCrash registration
  bool closed_ = false;              // Close() ran; closed_clean_ is its result
  bool closed_clean_ = false;
  // Outstanding asynchronous leases. The opaque id remains entirely internal:
  // ReadLease destruction routes it back through ReleaseReadLeaseThunk.
  struct PrepHold { BlockKey key; std::string fn; };
  std::unordered_map<uint64_t, PrepHold> prep_holds_;
  uint64_t prep_token_seq_ = 0;
  // Key -> its true payload length (slot_size >= this). Rebuilt on open from
  // the table; the allocator only tracks slot size. Keyed by the 24-byte
  // BlockKey itself: a restart installs one entry per cached object, and a
  // 48-character name per entry was a second allocation and a longer hash for
  // nothing the key does not already say.
  // Digests come from clients. Each word goes through a full 64-bit mix
  // before the next is folded in, starting from a value drawn once per
  // process, so keys cannot be chosen to land in one bucket.
  struct BlockKeyHash {
    static uint64_t Mix(uint64_t h) noexcept {
      h ^= h >> 33;
      h *= 0xFF51AFD7ED558CCDull;
      h ^= h >> 33;
      h *= 0xC4CEB9FE1A85EC53ull;
      h ^= h >> 33;
      return h;
    }
    static uint64_t Seed() noexcept;
    size_t operator()(const BlockKey& k) const noexcept {
      static const uint64_t seed = Seed();
      return static_cast<size_t>(
          Mix(Mix(Mix(k.digest_lo ^ seed) ^ k.digest_hi) ^ k.tenant_hash));
    }
  };
  std::unordered_map<BlockKey, uint32_t, BlockKeyHash> payload_len_;
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
  // Guards the in-memory maps, the allocate/lookup+pin step and the 64 B
  // slots.tbl record writes -- never held across payload I/O (a pin protects
  // the slot from eviction in the unlocked window, inflight_/deferred_remove_
  // protect it from Remove). The record write and the payload_len_ install
  // form one critical section: that is the commit, on disk and for readers.
  mutable std::mutex mu_;
  uint64_t committed_payload_bytes_ = 0;  // under mu_
  bool ok_ = false;
  std::atomic<bool> metadata_failed_{false};
  std::string startup_error_;
  uint64_t table_rebuilt_ = 0;
  uint64_t evicted_bytes_ = 0;
  uint64_t deferred_remove_total_ = 0;  // under mu_
  std::atomic<uint64_t> dio_write_fallbacks_{0};
  std::atomic<uint64_t> put_no_slot_{0};
  std::atomic<uint64_t> put_busy_{0};
  std::atomic<uint64_t> dio_read_fallbacks_{0};
  std::atomic<uint64_t> table_syncs_{0};
  std::atomic<uint64_t> bind_wipes_{0};
  std::atomic<uint64_t> metadata_io_errors_{0};
  std::function<bool()> payload_write_hook_for_test_;
  // Runs inside SyncTableFile, after the token was taken back.
  std::function<void()> table_sync_hook_for_test_;
  // Runs under mu_ (records are written at commit): it must not call back into
  // the store, and blocking in it blocks every operation on this disk.
  std::function<bool()> record_write_hook_for_test_;
  // Test-only seam (never set in production; same discipline as the hooks
  // above): when set, replaces EVERY CQE reap of the batched io_uring write
  // path -- the blocking wait AND the post-failure non-blocking drain -- so
  // tests can inject a hard reap failure. The void* signature keeps liburing
  // types out of the header (build-flag-independent class layout).
  std::function<int(void* ring, void* cqe)> uring_reap_hook_for_test_;
  std::atomic<uint64_t> unclean_resets_{0};
  std::atomic<uint64_t> warm_restarts_{0};
  std::atomic<uint64_t> eviction_record_clears_{0};
  std::atomic<uint64_t> watermark_extent_clears_{0};
  std::atomic<uint64_t> watermark_slot_record_clears_{0};
  std::atomic<uint64_t> watermark_record_clear_bytes_{0};
  std::atomic<uint64_t> watermark_freed_slot_bytes_{0};
  // Under mu_: shared callbacks count global-selector pwrite completions only.
  bool watermark_selector_call_ = false;
  // slots.tbl sync thread (see Options::table_sync_ms): fdatasync only when
  uint64_t rebuild_corrupt_records_ = 0;
  uint64_t rebuild_rejected_records_ = 0;
  uint64_t rebuild_scanned_bytes_ = 0;
  uint64_t rebuild_scan_chunks_ = 0;
  uint64_t rebuild_sparse_ranges_ = 0;
  uint64_t rebuild_mmap_scans_ = 0;
  uint64_t rebuild_sequential_fallbacks_ = 0;
  uint32_t open_fallocate_skipped_ = 0;
  uint64_t open_extents_us_ = 0;
  uint64_t open_scan_us_ = 0;
  uint64_t open_restore_us_ = 0;
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
  // Wheel entries one retention expiry looks at per hold of the allocator's
  // lock (about a hundred microseconds of work).
  static constexpr size_t kRetainExpireBatch = 1024;
  static constexpr std::chrono::microseconds kRetainExpirePause{50};
  std::atomic<uint64_t> reclaimed_{0};
  std::atomic<uint64_t> rebalanced_{0};
  std::atomic<uint64_t> batched_writes_{0};
  std::atomic<uint64_t> uring_write_batches_{0};
  // Kept in every build so the class layout does not depend on io_uring.
  bool uring_write_enabled_ = false;
  // The batched-write submission ring is thread_local (see disk_slab_store.cc):
  // one per flush worker, no shared lock across the blocking CQE wait.
  std::vector<uint64_t> reclaim_last_puts_;  // reclaim-thread-local puts snapshot
  NewPutDemandLatch watermark_latch_;  // under mu_
  uint64_t new_put_bytes_since_tick_ = 0;  // successful NEW commits, under mu_
  uint64_t evict_high_bytes_ = 0;  // high crossing activates eviction
  uint64_t evict_low_bytes_ = 0;   // active eviction drains to this target
  size_t evict_max_victims_per_tick_ = 256;
  std::vector<BlockKey> watermark_evicted_;  // bounded reusable output, under mu_
  std::atomic<bool> watermark_active_{false};
  std::atomic<uint64_t> watermark_ticks_{0};
  std::atomic<uint64_t> watermark_last_tick_us_{0};
  std::atomic<uint64_t> watermark_max_tick_us_{0};
  // Reclaimer observations, read/written together under mu_.
  uint64_t watermark_high_observations_ = 0;
  uint64_t watermark_last_high_bytes_ = 0;
  uint64_t watermark_peak_high_bytes_ = 0;
  std::thread reclaim_thread_;
  std::condition_variable reclaim_cv_;
  std::mutex reclaim_mu_;
  bool reclaim_stop_ = false;
};

}  // namespace dfkv

#endif  // DFKV_DISK_SLAB_STORE_H_
