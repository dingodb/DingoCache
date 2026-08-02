/* KVStore — cache-node local KV store: disk-backed, LRU-evicted, cache-only
 * (no S3 fallback; a miss is a clean NotFound). Cache() is synchronous: after
 * it returns the block is on disk and indexed (IsCached==true), giving the
 * cross-node read-after-write visibility the design requires. Portable (no
 * brpc/io_uring); the real cache node uses the dingofs DiskCache engine.
 *
 * Concurrency: the in-memory index + recency are SHARDED into `shards` stripes,
 * each guarded by its own std::shared_mutex (the dingofs `Shards` pattern). The
 * GET hot path (Range/RangeDirect/IsCached) takes a SHARED lock and only flips a
 * per-entry atomic CLOCK bit (no list mutation), so reads to a shard run
 * concurrently; Cache/eviction take the EXCLUSIVE lock. Eviction is per-shard
 * second-chance (CLOCK) — an LRU approximation; each shard owns capacity/shards
 * bytes. The bulk disk I/O always happens OUTSIDE the lock (the open fd pins the
 * inode, so a concurrent eviction can't pull bytes out from under a reader). */
#ifndef DFKV_KV_STORE_H_
#define DFKV_KV_STORE_H_

#include <atomic>
#include <functional>
#include <cstdint>
#include <list>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/kv_types.h"
#include "common/status.h"
#include "cache/store_engine.h"

namespace dfkv {

class KVStore : public StoreEngine {
 public:
  struct Options {
    std::string cache_dir;
    uint64_t capacity_bytes = (1ull << 30);
    // Index/LRU stripes. Each shard owns capacity_bytes/shards and its own lock,
    // so concurrent ops to distinct shards never contend. Default 16; the strict
    // cross-key LRU unit test pins it to 1.
    size_t shards = 16;
  };

  explicit KVStore(Options opt);
  bool Healthy() const override { return healthy_; }
  const std::string& StartupError() const override { return startup_error_; }

  // Synchronous, idempotent (skips if already present). No S3 upload.
  Status Cache(const BlockKey& key, const void* data, size_t len) override;
  // Same semantics as Cache(), but `data` must point at a 4096-aligned
  // buffer with at least `cap` bytes. The O_DIRECT write uses it directly, so the
  // server PUT path avoids a payload-sized bounce-buffer copy.
  Status CacheDirect(const BlockKey& key, char* data, size_t len, size_t cap) override;
  // [offset, offset+length) from the local block; NotFound if absent.
  Status Range(const BlockKey& key, uint64_t offset, uint64_t length,
               std::string* out, size_t* value_len = nullptr) override;
  // Like Range but reads straight into a caller buffer (no std::string), saving a
  // copy on the server GET path. Reads up to min(length, file-offset, dst_cap)
  // bytes into dst; *out_len = bytes read. NotFound if absent.
  Status RangeInto(const BlockKey& key, uint64_t offset, uint64_t length,
                   char* dst, size_t dst_cap, size_t* out_len,
                   size_t* value_len = nullptr) override;
  // O_DIRECT range read into a caller-provided aligned buffer. The disk read may
  // cover an aligned superset of the requested range; *out_data points inside
  // io_buf at the exact requested bytes and can be scatter-sent directly.
  Status RangeDirect(const BlockKey& key, uint64_t offset, uint64_t length,
                     char* io_buf, size_t io_cap, const char** out_data,
                     size_t* out_len, size_t* value_len = nullptr) override;

  // Async-friendly split of RangeDirect. Performs the cheap lookup/open/range
  // alignment half only; the returned move-only lease owns the descriptor until
  // the caller's asynchronous read completes or is abandoned.
  Status RangeDirectPrep(const BlockKey& key, uint64_t offset, uint64_t length,
                         size_t io_cap, ReadLease* out) override;

  bool IsCached(const BlockKey& key) const override;
  Status Lookup(const BlockKey& key, ValueMetadata* out) const override;

  // Explicitly drop a cached block: deletes the file, removes it from the
  // shard index + CLOCK ring, and reclaims its bytes (exclusive shard lock).
  // kOk if removed, kNotFound if absent. Used by the LMCache L2 eviction path
  // (dfkv_remove / DfkvL2Adapter.delete); distinct from capacity eviction so
  // it does NOT touch the eviction counters.
  Status Remove(const BlockKey& key) override;

  uint64_t UsedBytes() const override;
  uint64_t TenantUsedBytes(uint64_t tenant_hash) const override;
  size_t Count() const override;
  uint64_t Evictions() const override { return evictions_.load(std::memory_order_relaxed); }
  uint64_t EvictedBytes() const override { return evicted_bytes_.load(std::memory_order_relaxed); }
  // Orphan ".tmp" files removed at construction (crash-between-write-and-rename
  // leaks). Diagnostic: a persistently non-zero value across restarts points at
  // frequent mid-write kills.
  uint64_t TmpReclaimed() const { return tmp_reclaimed_.load(std::memory_order_relaxed); }
  // PUTs that hit a real ENOSPC, force-evicted, and retried successfully.
  uint64_t EnospcEvictions() const { return enospc_evictions_.load(std::memory_order_relaxed); }
  const std::string& Dir() const override { return opt_.cache_dir; }

  // Test-only write override shared by Cache and CacheDirect. It reports errno
  // so both paths can exercise the same bounded ENOSPC force-evict + retry.
  using WriteFn = std::function<bool(const std::string& path, const void* data,
                                     size_t len, int* out_errno)>;
  void SetWriteFnForTest(WriteFn fn) { write_fn_override_ = std::move(fn); }
  using RenameFn =
      std::function<bool(const std::string& from, const std::string& to)>;
  void SetRenameFnForTest(RenameFn fn) { rename_fn_override_ = std::move(fn); }

 private:
  struct Entry {
    std::string path;
    uint64_t size = 0;
    uint64_t tenant_hash = 0;
    std::atomic<bool> referenced{false};  // CLOCK bit: set on access (read lock)
    // This entry's own node in Shard::ring. std::list iterators are stable
    // across other insert/erase, so Remove() drops the ring node in O(1)
    // instead of scanning the whole ring (the old O(n), O(n^2) under RemoveMany
    // while holding the exclusive lock). Set right after the ring push_front.
    std::list<std::string>::iterator it{};
    Entry(std::string p, uint64_t s, uint64_t tenant)
        : path(std::move(p)), size(s), tenant_hash(tenant) {}
  };
  struct Shard {
    mutable std::shared_mutex mu;
    std::unordered_map<std::string, Entry> index;  // filename -> entry
    std::list<std::string> ring;                    // CLOCK ring, front = newest
    uint64_t used_bytes = 0;
    std::unordered_map<uint64_t, uint64_t> tenant_used_bytes;
    uint64_t capacity = 0;
    // Persistent CLOCK hand: the next eviction candidate, swept tail->front and
    // wrapped. Carrying it across Cache() calls amortizes the second-chance scan
    // so a hot, over-capacity shard does not re-clear the whole ring on every
    // write. end() means "(re)start at the tail (oldest)".
    std::list<std::string>::iterator hand;
    Shard() : hand(ring.end()) {}
  };

  Shard& ShardFor(const std::string& fname) const;
  // Eviction detaches victims from the index/ring under the exclusive lock but
  // only RENAMES their files to a unique sibling (a fast metadata op); the slow
  // block-freeing unlink is deferred to `*trash`, which the caller drains AFTER
  // releasing the lock so an eviction storm can't stall concurrent GETs.
  bool EvictLocked(Shard& sh,
                   std::vector<std::string>* trash);  // CLOCK 2nd-chance
  uint64_t ForceEvictLocked(Shard& sh, uint64_t target,
                            std::vector<std::string>* trash, bool* io_error);
  bool RenameToTrash(const std::string& path, std::string* trash);
  bool RebuildIndex();

  Options opt_;
  bool healthy_ = false;
  std::string startup_error_;
  std::vector<std::unique_ptr<Shard>> shards_;  // fixed after construction
  std::atomic<uint64_t> tmp_seq_{0};  // unique suffix for concurrent lock-free writes
  // Eviction counters (relaxed): incremented in EvictLocked across shards.
  std::atomic<uint64_t> evictions_{0}, evicted_bytes_{0};
  std::atomic<uint64_t> tmp_reclaimed_{0};   // orphan .tmp removed at startup
  std::atomic<uint64_t> enospc_evictions_{0};  // ENOSPC force-evict + retry succeeded
  WriteFn write_fn_override_;   // test-only Cache/CacheDirect write injection
  RenameFn rename_fn_override_; // test-only metadata failure injection
};

}  // namespace dfkv

#endif  // DFKV_KV_STORE_H_
