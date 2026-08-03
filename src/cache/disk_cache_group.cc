#include "cache/disk_cache_group.h"

#include <cstdlib>
#include <map>

#include "cache/disk_slab_store.h"
#include "common/config_dump.h"
#include "utils/log.h"

namespace dfkv {

namespace {
// DFKV_DISK_HASH_WEIGHT: ketama vnode multiplier for the intra-server disk
// ring, clamped [1, 64]. Default 1 keeps the historical placement (existing
// cache entries stay routable); raising it re-routes a share of old keys
// (misses, refilled on demand) in exchange for a much flatter per-disk load.
int DiskHashWeight() {
  static const int v = [] {
    const char* e = std::getenv("DFKV_DISK_HASH_WEIGHT");
    if (e && *e) {
      long x = std::strtol(e, nullptr, 10);
      if (x >= 1 && x <= 64) return static_cast<int>(x);
    }
    return 1;
  }();
  return v;
}
}  // namespace

DiskCacheGroup::DiskCacheGroup(Options opt) {
  size_t n = opt.cache_dirs.empty() ? 1 : opt.cache_dirs.size();
  uint64_t per_disk = opt.capacity_bytes / n;
  if (per_disk == 0) per_disk = opt.capacity_bytes;  // tiny-cap safety
  std::string engine = opt.engine;
  if (engine.empty()) {
    const char* e = std::getenv("DFKV_STORE_ENGINE");
    engine = (e && *e) ? e : "file";
  }
  const bool use_slab = (engine == "slab");
  engine_ = use_slab ? "slab" : "file";  // resolved truth, reported via EngineName
  // Slab I/O mode: DIRECT by default -- on GPU nodes the page-cache/dirty
  // growth of buffered I/O competes with training/inference memory; burst
  // absorption belongs to the explicit RAM tier (DFKV_RAM_TIER), not the page
  // cache. DFKV_SLAB_WRITE=buffered opts out (benchmarks / non-GPU boxes).
  const char* wm = std::getenv("DFKV_SLAB_WRITE");
  const bool slab_direct = !(wm && std::string(wm) == "buffered");
  // Slot quantum (DFKV_SLAB_GRANULARITY, bytes): tune DOWN for small-value
  // workloads (default 1 MiB wastes ~94% on 64 KiB values). Changing it is a
  // layout change -> meta mismatch -> the store re-inits EMPTY (cold cache):
  // treat as a migration step, not a routine restart.
  uint64_t slab_gran = 0;
  if (const char* g = std::getenv("DFKV_SLAB_GRANULARITY")) {
    unsigned long long v = std::strtoull(g, nullptr, 10);
    if (v > 0) slab_gran = v;
  }
  // slots.tbl fdatasync cadence override (ms; 0 disables). Default 100.
  uint32_t sync_ms = 100;
  if (const char* t = std::getenv("DFKV_SLAB_TABLE_SYNC_MS"))
    sync_ms = static_cast<uint32_t>(std::strtoul(t, nullptr, 10));
  // Background free-slot reclaimer cadence override (ms; 0 disables). Default 50.
  uint32_t reclaim_ms = 50;
  if (const char* r = std::getenv("DFKV_SLAB_RECLAIM_MS"))
    reclaim_ms = static_cast<uint32_t>(std::strtoul(r, nullptr, 10));
  // Effective values into the startup config dump (defaults included).
  config_dump::RecordResolved("DFKV_STORE_ENGINE", engine_);
  config_dump::RecordResolved("DFKV_DISK_HASH_WEIGHT", std::to_string(DiskHashWeight()));
  config_dump::RecordResolved("DFKV_SLAB_WRITE", slab_direct ? "direct" : "buffered");
  config_dump::RecordResolved("DFKV_SLAB_GRANULARITY",
                              slab_gran ? std::to_string(slab_gran) : std::string("1048576"));
  config_dump::RecordResolved("DFKV_SLAB_TABLE_SYNC_MS", std::to_string(sync_ms));
  config_dump::RecordResolved("DFKV_SLAB_RECLAIM_MS", std::to_string(reclaim_ms));
  for (const auto& dir : opt.cache_dirs) {
    std::unique_ptr<StoreEngine> store;
    if (use_slab) {
      DiskSlabStore::Options so;
      so.dir = dir;
      so.capacity_bytes = per_disk;
      so.direct_writes = slab_direct;
      if (slab_gran) so.slot_granularity = slab_gran;
      so.table_sync_ms = sync_ms;
      so.reclaim_interval_ms = reclaim_ms;
      bool store_ok = false;
      auto slab = std::make_unique<DiskSlabStore>(so, &store_ok);
      if (!store_ok) failed_dirs_.push_back(dir);
      slabs_.push_back(slab.get());
      // Resolved truth (an fs that rejects O_DIRECT demotes to buffered).
      write_mode_ = slab->DirectWritesActive() ? "direct" : "buffered";
      store = std::move(slab);
    } else {
      store = std::make_unique<KVStore>(KVStore::Options{dir, per_disk});
    }
    by_id_[dir] = store.get();
    disks_.push_back(std::move(store));
    // Disk id = its dir path. The hash weight multiplies ketama vnodes
    // (160/weight-unit): at the default weight 1 the realized per-disk key
    // share on a 6-disk group varies ±20%, so the hottest disk saturates at
    // ~83% of aggregate disk bandwidth (measured: one disk at aqu-sz 600+
    // while its five peers idle at 20-50, capping a 6x6.9GB/s server at
    // ~33GB/s cold-read). Weight 10 (1600 vnodes/disk) shrinks the share
    // spread to ~±6%. Env-tunable; NOTE changing it re-routes existing keys
    // (cache semantics: old entries become misses, not corruption).
    ring_.AddNode(dir, DiskHashWeight());
  }
  ring_.Build();
  if (disks_.size() > 1) {
    // Realized routing-share spread (min/max ring points per disk): the ops
    // signal for placement skew — a hot disk saturates first and gates the
    // whole group's cold-read throughput.
    auto pts = ring_.NodePointCounts();
    size_t mn = SIZE_MAX, mx = 0;
    for (const auto& [_, c] : pts) { if (c < mn) mn = c; if (c > mx) mx = c; }
    DFKV_LOG_INFO("disk ring: " + std::to_string(disks_.size()) + " disks, weight=" +
                  std::to_string(DiskHashWeight()) + ", points min=" + std::to_string(mn) +
                  " max=" + std::to_string(mx));
  }
}

StoreEngine* DiskCacheGroup::Route(const BlockKey& key) const {
  if (disks_.size() == 1) return disks_[0].get();
  std::string id;
  if (!ring_.Lookup(key.Filename(), &id)) return nullptr;
  auto it = by_id_.find(id);
  return it == by_id_.end() ? nullptr : it->second;
}

Status DiskCacheGroup::Cache(const BlockKey& key, const void* data, size_t len) {
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->Cache(key, data, len);
}

Status DiskCacheGroup::Remove(const BlockKey& key) {
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->Remove(key);
}

Status DiskCacheGroup::CacheDirect(const BlockKey& key, char* data, size_t len,
                                   size_t cap) {
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->CacheDirect(key, data, len, cap);
}

Status DiskCacheGroup::Range(const BlockKey& key, uint64_t offset,
                             uint64_t length, std::string* out) {
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->Range(key, offset, length, out);
}

Status DiskCacheGroup::RangeInto(const BlockKey& key, uint64_t offset,
                                 uint64_t length, char* dst, size_t dst_cap,
                                 size_t* out_len) {
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->RangeInto(key, offset, length, dst, dst_cap, out_len);
}

Status DiskCacheGroup::RangeDirect(const BlockKey& key, uint64_t offset,
                                   uint64_t length, char* io_buf, size_t io_cap,
                                   const char** out_data, size_t* out_len) {
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->RangeDirect(key, offset, length, io_buf, io_cap, out_data, out_len);
}

Status DiskCacheGroup::RangeDirectPrep(const BlockKey& key, uint64_t offset,
                                       uint64_t length, size_t io_cap,
                                       KVStore::RangePrep* out) {
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  Status st = d->RangeDirectPrep(key, offset, length, io_cap, out);
  // Brand the engine's token with the disk index so RangeRelease -- which has
  // no key to route by -- finds its way back (0 stays 0 = nothing to release).
  if (st == Status::kOk && out && out->token != 0) {
    for (size_t i = 0; i < disks_.size(); ++i) {
      if (disks_[i].get() != d) continue;
      out->token = (static_cast<uint64_t>(i + 1) << 56) | (out->token & kTokenMask);
      break;
    }
  }
  return st;
}

std::vector<Status> DiskCacheGroup::CacheDirectBatch(
    const std::vector<StoreEngine::CacheBatchItem>& items) {
  // Split by owning disk (consistent-hash route), one engine batch per disk --
  // preserves per-key placement while letting the slab engine amortize locks
  // and submit the disk's payload writes together.
  std::vector<Status> out(items.size(), Status::kIOError);
  std::map<StoreEngine*, std::vector<size_t>> by_disk;
  for (size_t i = 0; i < items.size(); ++i) {
    StoreEngine* d = Route(items[i].key);
    if (d) by_disk[d].push_back(i);
  }
  for (auto& [d, idx] : by_disk) {
    std::vector<StoreEngine::CacheBatchItem> sub;
    sub.reserve(idx.size());
    for (size_t k : idx) sub.push_back(items[k]);
    std::vector<Status> sts = d->CacheDirectBatch(sub);
    for (size_t m = 0; m < idx.size(); ++m) out[idx[m]] = sts[m];
  }
  return out;
}

DiskSlabStore::Stats DiskCacheGroup::SlabStats() const {
  DiskSlabStore::Stats sum;
  for (const auto* d : slabs_) {
    const auto st = d->GetStats();
    sum.dio_write_fallbacks += st.dio_write_fallbacks;
    sum.dio_read_fallbacks += st.dio_read_fallbacks;
    sum.table_syncs += st.table_syncs;
    sum.steals += st.steals;
    sum.extent_returns += st.extent_returns;
    sum.deferred_removes += st.deferred_removes;
    sum.inflight += st.inflight;
    sum.prep_holds += st.prep_holds;
    sum.reclaimed_slots += st.reclaimed_slots;
    // rebalanced_extents was MISSING from this sum since its introduction --
    // dfkv_slab_rebalanced_total silently read 0 fleet-wide (found while adding
    // the batch counters; canary data showed 0 despite active rebalances).
    sum.rebalanced_extents += st.rebalanced_extents;
    sum.batched_writes += st.batched_writes;
    sum.uring_write_batches += st.uring_write_batches;
  }
  return sum;
}

void DiskCacheGroup::RangeRelease(uint64_t token) {
  const size_t i = static_cast<size_t>(token >> 56);
  if (i == 0 || i > disks_.size()) return;
  disks_[i - 1]->RangeRelease(token & kTokenMask);
}

bool DiskCacheGroup::IsCached(const BlockKey& key) const {
  StoreEngine* d = Route(key);
  return d != nullptr && d->IsCached(key);
}

uint64_t DiskCacheGroup::UsedBytes() const {
  uint64_t t = 0;
  for (const auto& d : disks_) t += d->UsedBytes();
  return t;
}

size_t DiskCacheGroup::Count() const {
  size_t t = 0;
  for (const auto& d : disks_) t += d->Count();
  return t;
}

uint64_t DiskCacheGroup::Evictions() const {
  uint64_t t = 0;
  for (const auto& d : disks_) t += d->Evictions();
  return t;
}

uint64_t DiskCacheGroup::EvictedBytes() const {
  uint64_t t = 0;
  for (const auto& d : disks_) t += d->EvictedBytes();
  return t;
}

}  // namespace dfkv
