/* DiskCacheGroup — one cache node spanning multiple NVMe SSDs, mirroring
 * dingo-cache's intra-node layout (--cache_dir=d1,d2,d3). Each disk is an
 * independent KVStore (own LRU); a block is routed to one disk by a second-level
 * Ketama hash on BlockKey.Filename(). Total capacity is split evenly across
 * disks. A single-disk group degenerates to one KVStore. */
#ifndef DFKV_DISK_CACHE_GROUP_H_
#define DFKV_DISK_CACHE_GROUP_H_

#include <cstdint>
#include <memory>
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
    // Backend: "file" (KVStore, default) or "slab" (DiskSlabStore). Empty =>
    // read DFKV_STORE_ENGINE (default "file"). Selectable so slab lands off by
    // default with zero production impact.
    std::string engine{};
  };

  explicit DiskCacheGroup(Options opt);
  bool Healthy() const;
  const std::string& StartupError() const { return startup_error_; }

  Status Cache(const BlockKey& key, const void* data, size_t len);
  Status CacheDirect(const BlockKey& key, char* data, size_t len, size_t cap);
  // Batched CacheDirect: split by disk route, one engine batch per disk.
  std::vector<Status> CacheDirectBatch(const std::vector<StoreEngine::CacheBatchItem>& items);
  Status Range(const BlockKey& key, uint64_t offset, uint64_t length,
               std::string* out);
  Status RangeInto(const BlockKey& key, uint64_t offset, uint64_t length,
                   char* dst, size_t dst_cap, size_t* out_len);
  Status RangeDirect(const BlockKey& key, uint64_t offset, uint64_t length,
                     char* io_buf, size_t io_cap, const char** out_data,
                     size_t* out_len);
  // Cheap prep half of RangeDirect (no disk read); see KVStore::RangeDirectPrep.
  Status RangeDirectPrep(const BlockKey& key, uint64_t offset, uint64_t length,
                         size_t io_cap, KVStore::RangePrep* out);
  // Balance a prep whose RangePrep::token != 0 (slab holds the slot across the
  // caller's async read); the owning disk's index rides the token's top byte.
  void RangeRelease(uint64_t token);
  bool IsCached(const BlockKey& key) const;

  // Drop a cached block from its owning disk (routes like IsCached). kOk if
  // removed, kNotFound if absent. Backs the LMCache L2 eviction path.
  Status Remove(const BlockKey& key);

  uint64_t UsedBytes() const;   // summed across disks
  size_t Count() const;         // summed across disks
  size_t DiskCount() const { return disks_.size(); }
  uint64_t Evictions() const;     // summed across disks
  uint64_t EvictedBytes() const;  // summed across disks
  // The storage backend actually constructed ("file" | "slab") -- the RESOLVED
  // choice (Options.engine / DFKV_STORE_ENGINE / default), not the flag intent.
  const std::string& EngineName() const { return engine_; }
  // Slab I/O mode actually in effect ("direct" | "buffered"; empty for file
  // engine). Direct is the deployment default; a filesystem that rejects
  // O_DIRECT (tmpfs) resolves to buffered regardless of the env.
  const std::string& WriteMode() const { return write_mode_; }
  // Summed slab runtime counters across disks (all-zero for the file engine).
  DiskSlabStore::Stats SlabStats() const;
  // Per-disk views for fine-grained metrics (i in [0, DiskCount)).
  const std::string& DiskPath(size_t i) const { return disks_[i]->Dir(); }
  uint64_t DiskUsedBytes(size_t i) const { return disks_[i]->UsedBytes(); }
  size_t DiskObjects(size_t i) const { return disks_[i]->Count(); }

 private:
  static constexpr uint64_t kTokenMask = (1ull << 56) - 1;  // engine token bits

  StoreEngine* Route(const BlockKey& key) const;

  bool healthy_ = true;
  std::string startup_error_;
  std::string engine_;      // resolved backend name (see EngineName)
  std::string write_mode_;  // resolved slab I/O mode (see WriteMode)
  std::vector<std::unique_ptr<StoreEngine>> disks_;
  std::vector<const DiskSlabStore*> slabs_;  // typed view of disks_ (slab engine only)
  std::unordered_map<std::string, StoreEngine*> by_id_;  // disk id -> store
  ConHash ring_;
};

}  // namespace dfkv

#endif  // DFKV_DISK_CACHE_GROUP_H_
