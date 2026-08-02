/* StoreEngine — the cache-node local storage backend interface.
 *
 * DiskCacheGroup routes a block to one backend per disk; both the original
 * file-per-block KVStore and the extent-slab DiskSlabStore implement this, so
 * the engine is selectable (--store-engine / DFKV_STORE_ENGINE) with the file
 * engine as the default. The RDMA zero-copy fast path (RangeDirect returning a
 * pointer into the caller's registered buffer) and its io_uring split
 * (RangeDirectPrep) are part of the interface; a buffered engine (slab) fills
 * io_buf with a plain read and declines the prep (returns non-kOk) so the RDMA
 * server falls back to the synchronous RangeDirect. */
#ifndef DFKV_STORE_ENGINE_H_
#define DFKV_STORE_ENGINE_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/kv_types.h"
#include "common/status.h"

namespace dfkv {

// Output of the cheap, lock-protected prep half of RangeDirect (see
// KVStore::RangeDirectPrep). A buffered engine leaves fd < 0 and returns a
// non-kOk status to signal "no async prep; use the synchronous RangeDirect".
struct RangePrep {
  int fd = -1;              // owned by caller on kOk (caller closes); -1 if no read
  uint64_t aligned_off = 0; // O_DIRECT-aligned read start
  size_t aligned_len = 0;   // O_DIRECT-aligned read length (multiple of 4096)
  size_t head = 0;          // offset of requested bytes within the aligned read
  size_t payload_len = 0;   // exact requested bytes (after clamp to file size)
  size_t value_len = 0;     // full stored value size — lets the server tell a
                            // whole-value read (offset 0, payload_len ==
                            // value_len) from a sub-range (RAM promotion gate)
  // 0 = nothing to release. A slot-based engine (slab) must hold the slot
  // against eviction/remove while the caller's async read is in flight -- the
  // caller passes this back via RangeRelease once the read is done (KVStore's
  // per-key files need no hold: the open fd pins the inode, so token stays 0).
  uint64_t token = 0;
};

class StoreEngine {
 public:
  virtual ~StoreEngine() = default;
  // Construction cannot return a Status; startup health is therefore explicit.
  // A node must refuse readiness when any configured engine is unhealthy.
  virtual bool Healthy() const = 0;
  virtual const std::string& StartupError() const = 0;

  virtual Status Cache(const BlockKey& key, const void* data, size_t len) = 0;
  virtual Status CacheDirect(const BlockKey& key, char* data, size_t len,
                             size_t cap) = 0;
  // Batched CacheDirect: same per-item semantics/status as CacheDirect, but an
  // engine may amortize its lock round-trips and submit the payload writes
  // together (the RAM-tier flusher's small-object drain is IOPS-bound at one
  // synchronous write per key). Default: per-item loop -- engines without a
  // batch win (file engine) inherit correct behavior.
  struct CacheBatchItem { BlockKey key; char* data; size_t len; size_t cap; };
  virtual std::vector<Status> CacheDirectBatch(const std::vector<CacheBatchItem>& items) {
    std::vector<Status> out;
    out.reserve(items.size());
    for (const auto& it : items) out.push_back(CacheDirect(it.key, it.data, it.len, it.cap));
    return out;
  }
  virtual Status Range(const BlockKey& key, uint64_t offset, uint64_t length,
                       std::string* out) = 0;
  virtual Status RangeInto(const BlockKey& key, uint64_t offset, uint64_t length,
                           char* dst, size_t dst_cap, size_t* out_len) = 0;
  virtual Status RangeDirect(const BlockKey& key, uint64_t offset, uint64_t length,
                             char* io_buf, size_t io_cap, const char** out_data,
                             size_t* out_len) = 0;
  virtual Status RangeDirectPrep(const BlockKey& key, uint64_t offset,
                                 uint64_t length, size_t io_cap, RangePrep* out) = 0;
  // Balance a prep whose RangePrep::token != 0, after the async read completes
  // (success or failure). Default: nothing to release.
  virtual void RangeRelease(uint64_t token) { (void)token; }
  virtual bool IsCached(const BlockKey& key) const = 0;
  virtual Status Remove(const BlockKey& key) = 0;

  virtual uint64_t UsedBytes() const = 0;
  virtual size_t Count() const = 0;
  virtual uint64_t Evictions() const = 0;
  virtual uint64_t EvictedBytes() const = 0;
  virtual const std::string& Dir() const = 0;
};

}  // namespace dfkv

#endif  // DFKV_STORE_ENGINE_H_
