/* DiskCacheGroup — one cache node spanning multiple NVMe SSDs, mirroring
 * dingo-cache's intra-node layout (--cache_dir=d1,d2,d3). Each disk is an
 * independent KVStore (own LRU); a block is routed to one disk by a second-level
 * Ketama hash on BlockKey.Filename(). Total capacity is split evenly across
 * disks. A single-disk group degenerates to one KVStore. A disk that fails
 * while serving hands its keys to the others (see Healthy). */
#ifndef DFKV_DISK_CACHE_GROUP_H_
#define DFKV_DISK_CACHE_GROUP_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <map>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/con_hash.h"
#include "cache/disk_slab_store.h"
#include "cache/kv_store.h"
#include "cache/store_engine.h"
#include "common/kv_types.h"

namespace dfkv {

class DiskCacheGroup {
 public:
  struct Options {
    std::vector<std::string> cache_dirs;   // one per NVMe SSD
    uint64_t capacity_bytes = (1ull << 30);  // TOTAL, split evenly across disks
    // Empty reads DFKV_STORE_ENGINE, then selects the production slab backend.
    // "file" remains available only when explicitly requested for diagnostics.
    // "auto" (choose per directory by what it holds) and "log" (the
    // log-structured engine) need DFKV_LSE_EXPERIMENTAL=1 while that engine
    // is under construction; see docs/design/log-structured-engine.md §10.
    std::string engine{};
    // Dependency injection for focused concurrency tests. Production leaves
    // this empty and constructs the selected file/slab engine normally. Called
    // once per directory, from one thread per directory at the same time. An
    // exception it throws is reported as that directory's startup failure.
    std::function<std::unique_ptr<StoreEngine>(const std::string&, uint64_t)>
        engine_factory{};
    // Leave the disks opened but not activated; the owner calls Activate()
    // when the node is about to serve. See Activate().
    bool defer_activation = false;
    // Test seam: whether a disk failure record may be written next to this
    // (canonical) cache directory. Empty: the directory above it must be on
    // the directory's own filesystem.
    std::function<bool(const std::string&)> record_location_for_test{};
    // Test seam: replaces the Remove of a key found on a disk that is not
    // its home, at the start after a run with failed disks.
    std::function<Status(size_t disk, const BlockKey&)>
        misplaced_remove_for_test{};
    // Test seam: replaces the SyncRemovals of a disk after those Removes.
    std::function<bool(size_t disk)> sync_removals_for_test{};
  };

  explicit DiskCacheGroup(Options opt);
  ~DiskCacheGroup();
  // Closes every disk (StoreEngine::Close), all of them at the same time: each
  // waits on its own device. Same precondition as destruction; the destructor
  // calls it if nobody did. Reports one entry per disk, none on a second call.
  struct CloseReport {
    std::string dir;
    bool clean = false;
    double seconds = 0;
  };
  std::vector<CloseReport> Close();
  // Opening a disk changes nothing on it; activating it (StoreEngine::
  // Activate) marks it in use and lets it serve. All disks are activated
  // here, one after the other, and only if every one of them opened: until
  // then a process that is killed, or a start that fails or is cancelled,
  // leaves every disk as its previous run closed it. The constructor does
  // this itself unless Options::defer_activation is set. Idempotent; false
  // (and the group unhealthy) if a disk could not be activated.
  bool Activate();
  // Started, and at least DFKV_DEGRADED_MIN_DISKS (default 1) disks have
  // not failed; once one has, only those that take other disks' keys count
  // (see TakesNoOtherDisksKeys). A disk store that fails for good while serving (it refuses
  // every write until the process is restarted) loses its keys: while the
  // group is healthy they are routed to the healthy disks, where they miss
  // until they are written again. Keys of healthy disks never move.
  //
  // Before the first key moves, the failure is put on record next to every
  // healthy disk's directory (<dir>.dfkv-failed-disks), where the directory
  // above is on that disk's own filesystem; if it can be written nowhere,
  // nothing moves and the group is unhealthy. The next start that finds a
  // record resets the disks it names to empty (each by its engine; every
  // disk, if a copy cannot be read or names a directory this node does not
  // have) and drops from the other disks the keys whose home is another
  // disk (what the failed run wrote in place of a failed disk), so that
  // neither a value from before the failure nor one written in its place
  // comes back after a REMOVE or an overwrite. The record is cleared by a
  // clean close of a run in which no disk failed.
  bool Healthy() const;
  // Disks whose store has failed for good.
  size_t FailedDisks() const;
  // What the open's pass after a run with failed disks dropped: every key of
  // a disk that had failed, and keys found on a disk that is not their home.
  uint64_t OpenEmptiedKeys() const { return open_emptied_keys_; }
  uint64_t OpenMisplacedKeys() const { return open_misplaced_keys_; }
  // Keys of every disk, emptied because a record could not be used.
  uint64_t OpenUnusableKeys() const { return open_unusable_keys_; }
  // Whether disk i takes no other disk's keys in this run (see
  // DropMisplacedKeys).
  bool TakesNoOtherDisksKeys(size_t i) const { return no_takeover_[i]; }
  // The id a failure record uses for a cache directory: its canonical path.
  static std::string CanonicalDiskId(const std::string& dir);
  size_t MinHealthyDisks() const { return min_healthy_disks_; }
  // Whether the disk this key is written to has failed for good: only once
  // every disk has (see Healthy). Callers ask on every PUT.
  bool DiskFailedFor(const BlockKey& key) const;
  // The largest value every disk can store (the smallest engine limit).
  uint64_t MaxValueBytes() const;
  const std::string& StartupError() const { return startup_error_; }

  Status Cache(const BlockKey& key, const void* data, size_t len);
  Status CacheDirect(const BlockKey& key, char* data, size_t len, size_t cap);
  // Batched CacheDirect: split by disk route, one engine batch per disk.
  // `observe` runs once per participating disk after its engine call.
  using DirectBatchObserver = std::function<void(
      size_t disk, size_t objects, uint64_t bytes, size_t failures,
      double elapsed_seconds)>;
  std::vector<Status> CacheDirectBatch(
      const std::vector<StoreEngine::CacheBatchItem>& items,
      const DirectBatchObserver& observe = {});
  Status Range(const BlockKey& key, uint64_t offset, uint64_t length,
               std::string* out, size_t* value_len = nullptr);
  Status RangeInto(const BlockKey& key, uint64_t offset, uint64_t length,
                   char* dst, size_t dst_cap, size_t* out_len,
                   size_t* value_len = nullptr);
  Status RangeDirect(const BlockKey& key, uint64_t offset, uint64_t length,
                     char* io_buf, size_t io_cap, const char** out_data,
                     size_t* out_len, size_t* value_len = nullptr);
  // Cheap prep half of RangeDirect (no disk read); the returned lease carries
  // descriptor and storage-pin ownership through asynchronous queues.
  Status RangeDirectPrep(const BlockKey& key, uint64_t offset, uint64_t length,
                         size_t io_cap, ReadLease* out);
  bool IsCached(const BlockKey& key) const;
  Status Lookup(const BlockKey& key, ValueMetadata* out) const;

  // Drop a cached block from its owning disk (routes like IsCached). kOk if
  // removed, kNotFound if absent. Backs the LMCache L2 eviction path.
  Status Remove(const BlockKey& key);

  // Retention routes like Lookup. Enabled only when every disk supports it,
  // so one RETAIN reply describes the whole node consistently.
  bool RetainEnabled() const;
  uint8_t Retain(const BlockKey& key, uint32_t ttl_s, uint8_t mode,
                 uint32_t now_s);
  uint32_t RetainHeadroomPct() const;  // minimum across disks
  uint32_t RetainMaxTtl() const;       // minimum across disks
  SlabAllocator::RetentionStats RetentionStats() const;  // summed

  uint64_t UsedBytes() const;   // summed across disks
  size_t Count() const;         // summed across disks
  size_t DiskCount() const { return disks_.size(); }
  uint64_t Evictions() const;     // summed across disks
  uint64_t EvictedBytes() const;  // summed across disks
  // The storage backend actually constructed ("file" | "slab") -- the RESOLVED
  // choice (Options.engine / DFKV_STORE_ENGINE / default), not the flag intent.
  // "mixed" when "auto" chose different engines for different disks.
  const std::string& EngineName() const { return engine_; }
  // Whether slab metrics apply: the slab engine was selected, or "auto" chose
  // it for at least one disk.
  bool HasSlabDisks() const { return has_slab_; }
  // DFKV_SLAB_BALANCE: the slab disks move extents between size classes.
  bool SlabBalance() const { return slab_balance_; }
  // The engine disk i runs (its StoreEngine::EngineName).
  std::string DiskEngineName(size_t i) const { return disks_[i]->EngineName(); }
  void DiskEngineMetrics(size_t i,
                         std::vector<StoreEngine::EngineMetric>* out) const {
    disks_[i]->EngineMetrics(out);
  }
  // Slab I/O mode actually in effect ("direct" | "buffered"; empty for file
  // engine). Direct is the deployment default; a filesystem that rejects
  // O_DIRECT (tmpfs) resolves to buffered regardless of the env.
  const std::string& WriteMode() const { return write_mode_; }
  // Summed slab runtime counters across the slab disks (all-zero without).
  DiskSlabStore::Stats SlabStats() const;
  // Live size-class layout, aggregated by slot size across all slab disks.
  std::vector<SlabAllocator::ClassStat> SlabClasses() const;
  // Per-disk views for fine-grained metrics (i in [0, DiskCount)).
  const std::string& DiskPath(size_t i) const { return disks_[i]->Dir(); }
  uint64_t DiskUsedBytes(size_t i) const { return disks_[i]->UsedBytes(); }
  size_t DiskObjects(size_t i) const { return disks_[i]->Count(); }
  bool DiskHealthy(size_t i) const { return disks_[i]->Healthy(); }
  // Slab-only, indexed identically to DiskPath; one coherent observation view.
  // A store that an engine_factory built is not in the typed view: it has
  // no observation here, whatever its type.
  DiskSlabStore::WatermarkObservation DiskWatermarkObservation(size_t i) const {
    return i < slab_of_disk_.size() && slab_of_disk_[i] != nullptr
               ? slab_of_disk_[i]->GetWatermarkObservation()
               : DiskSlabStore::WatermarkObservation{};
  }
  uint64_t TenantUsedBytes(uint64_t tenant_hash) const;
  uint64_t TenantDefaultQuotaBytes() const { return tenant_default_quota_bytes_; }
  uint64_t TenantQuotaBytes(uint64_t tenant_hash) const {
    return TenantLimit(tenant_hash);
  }
  struct TenantQuotaMetric {
    uint64_t tenant_hash = 0;
    uint64_t limit_bytes = 0;
    uint64_t used_bytes = 0;
    uint64_t rejections = 0;
  };
  std::vector<TenantQuotaMetric> ConfiguredTenantQuotaMetrics() const;
  uint64_t TenantQuotaRejections() const {
    return tenant_quota_rejections_.load(std::memory_order_relaxed);
  }
  // Reads DFKV_TENANT_QUOTAS_FILE again (on SIGHUP) and replaces the per-tenant
  // quotas with it; a file that does not parse leaves them as they were
  // (false, *error says why). Rejection counts of tenants kept are kept.
  bool ReloadTenantQuotas(std::string* error);
  uint64_t TenantQuotaReloads() const {
    return tenant_quota_reloads_.load(std::memory_order_relaxed);
  }
  uint64_t TenantQuotaReloadFailures() const {
    return tenant_quota_reload_failures_.load(std::memory_order_relaxed);
  }
  // Index of the disk a key is stored on; DiskCount() if no disk takes it.
  size_t DiskIndexOf(const BlockKey& key) const {
    const size_t index = RouteIndex(key);
    return index == kInvalidDisk ? DiskCount() : index;
  }

 private:
  static constexpr size_t kInvalidDisk = std::numeric_limits<size_t>::max();

  size_t RouteIndex(const BlockKey& key) const;
  // The key's disk while none has failed.
  size_t HomeIndex(const std::string& name) const;
  // Puts disk `i`'s failure on record (see Healthy); false if it could not.
  bool FailureOnRecord(size_t i) const;
  bool WriteFailureRecords(const std::set<std::string>& ids) const;
  struct FailureRecords {
    bool found = false;   // some copy is there
    bool usable = true;   // every copy read in full, naming known disks only
    std::set<std::string> ids;
  };
  FailureRecords ReadFailureRecords() const;
  bool ResetAfterFailedRun();
  void DropMisplacedKeys();
  void ClearFailureRecords();
  StoreEngine* Route(const BlockKey& key) const;
  static constexpr size_t kTenantStripes = 256;
  size_t TenantStripe(uint64_t tenant_hash) const {
    return static_cast<size_t>(tenant_hash) & (kTenantStripes - 1);
  }
  uint64_t TenantLimit(uint64_t tenant_hash) const;
  // True when this tenant has a non-zero node quota. The quotas file can be
  // reloaded (ReloadTenantQuotas): evaluate this once per operation.
  bool TenantLimited(uint64_t tenant_hash) const {
    return TenantLimit(tenant_hash) != 0;
  }
  bool QuotaAllowsNewLocked(uint64_t tenant_hash, uint64_t current,
                            uint64_t additional) const;
  void RecordQuotaRejectionLocked(uint64_t tenant_hash);
  bool healthy_ = true;
  size_t min_healthy_disks_ = 1;  // DFKV_DEGRADED_MIN_DISKS
  // Failure records (see Healthy). failed_ids_ holds the ids on record:
  // those found at open and those this run recorded.
  mutable std::mutex record_mu_;
  mutable std::set<std::string> failed_ids_;
  mutable std::unique_ptr<std::atomic<bool>[]> on_record_;
  mutable std::atomic<bool> record_failed_{false};
  bool records_found_ = false;
  bool open_drop_incomplete_ = false;
  std::vector<bool> reset_;  // reset to empty by this start
  // Kept a key of another disk this start could not remove: takes no other
  // disk's keys in this run.
  std::vector<bool> no_takeover_;
  std::function<Status(size_t, const BlockKey&)> misplaced_remove_for_test_;
  std::function<bool(size_t)> sync_removals_for_test_;
  std::function<bool(const std::string&)> record_location_for_test_;
  uint64_t open_emptied_keys_ = 0;
  uint64_t open_misplaced_keys_ = 0;
  uint64_t open_unusable_keys_ = 0;
  bool closed_ = false;
  bool activated_ = false;
  std::string startup_error_;
  std::string engine_;      // resolved backend name (see EngineName)
  std::string write_mode_;  // resolved slab I/O mode (see WriteMode)
  std::vector<std::unique_ptr<StoreEngine>> disks_;
  std::vector<const DiskSlabStore*> slabs_;  // typed view of the slab disks
  // The same by disk index: nullptr for a disk that is not a slab store the
  // group built (another engine, or one an engine_factory made).
  std::vector<const DiskSlabStore*> slab_of_disk_;
  bool has_slab_ = false;
  bool slab_balance_ = false;  // DFKV_SLAB_BALANCE
  std::unordered_map<std::string, size_t> by_id_;  // disk id -> stable index
  std::vector<std::string> ids_;                   // stable index -> disk id
  std::vector<std::string> canonical_ids_;         // CanonicalDiskId(ids_[i])
  ConHash ring_;
  mutable std::array<std::mutex, kTenantStripes> tenant_stripes_;
  // At most this many stripes are held at once (CacheDirectBatch splits a
  // batch that spans more), well under what lock checkers track per thread.
  static constexpr size_t kMaxHeldStripes = 16;
  // Guards tenant_quotas_ and the shape of tenant_quota_rejections_by_hash_.
  // Written by ReloadTenantQuotas only, exclusively and without any stripe;
  // readers take it shared. A rejection count is incremented under the
  // shared lock and its tenant's stripe (lock order: a stripe, then this).
  mutable std::shared_mutex tenant_quotas_mu_;
  std::map<uint64_t, uint64_t> tenant_quotas_;
  std::map<uint64_t, uint64_t> tenant_quota_rejections_by_hash_;
  std::atomic<uint64_t> tenant_quota_reloads_{0}, tenant_quota_reload_failures_{0};
  uint64_t tenant_default_quota_bytes_ = 0;
  std::string tenant_quotas_file_;
  std::atomic<uint64_t> tenant_quota_rejections_{0};
};

}  // namespace dfkv

#endif  // DFKV_DISK_CACHE_GROUP_H_
