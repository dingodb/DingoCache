#include "cache/disk_cache_group.h"

#include <cstdlib>
#include <charconv>
#include <cstring>
#include <limits>
#include <map>
#include <unordered_set>

#include "cache/disk_slab_store.h"
#include "common/config_dump.h"
#include "utils/log.h"

namespace dfkv {

namespace {
bool ParseUnsigned(const char* text, uint64_t max, uint64_t* out) {
  if (text == nullptr || *text == '\0') return false;
  uint64_t value = 0;
  const char* end = text + std::strlen(text);
  const auto parsed = std::from_chars(text, end, value);
  if (parsed.ec != std::errc() || parsed.ptr != end || value > max)
    return false;
  *out = value;
  return true;
}

bool ResolveDiskHashWeight(int* out) {
  const char* text = std::getenv("DFKV_DISK_HASH_WEIGHT");
  if (text == nullptr) {
    *out = 1;
    return true;
  }
  uint64_t value = 0;
  if (!ParseUnsigned(text, 64, &value) || value == 0) return false;
  *out = static_cast<int>(value);
  return true;
}
}  // namespace

DiskCacheGroup::DiskCacheGroup(Options opt) {
  auto fail = [&](std::string error) {
    healthy_ = false;
    if (startup_error_.empty()) startup_error_ = std::move(error);
  };
  if (opt.cache_dirs.empty()) {
    fail("no cache directories configured");
    return;
  }
  if (opt.capacity_bytes == 0) {
    fail("capacity_bytes must be non-zero");
    return;
  }
  std::unordered_set<std::string> unique_dirs;
  for (const auto& dir : opt.cache_dirs) {
    if (dir.empty()) {
      fail("cache directory is empty");
      return;
    }
    if (!unique_dirs.insert(dir).second) {
      fail("duplicate cache directory: " + dir);
      return;
    }
  }

  std::string engine = opt.engine;
  if (engine.empty()) {
    const char* value = std::getenv("DFKV_STORE_ENGINE");
    engine = (value && *value) ? value : "file";
  }
  if (engine != "file" && engine != "slab") {
    fail("DFKV_STORE_ENGINE must be 'file' or 'slab'");
    return;
  }
  engine_ = engine;
  const bool use_slab = engine == "slab";

  int disk_weight = 1;
  if (!ResolveDiskHashWeight(&disk_weight)) {
    fail("DFKV_DISK_HASH_WEIGHT must be an integer in [1,64]");
    return;
  }

  bool slab_direct = true;
  if (const char* mode = std::getenv("DFKV_SLAB_WRITE")) {
    if (std::strcmp(mode, "direct") == 0) {
      slab_direct = true;
    } else if (std::strcmp(mode, "buffered") == 0) {
      slab_direct = false;
    } else if (use_slab) {
      fail("DFKV_SLAB_WRITE must be 'direct' or 'buffered'");
      return;
    }
  }
  uint64_t slab_gran = 0;
  if (const char* value = std::getenv("DFKV_SLAB_GRANULARITY")) {
    if (!ParseUnsigned(value, std::numeric_limits<uint32_t>::max(),
                       &slab_gran) ||
        slab_gran == 0) {
      if (use_slab) {
        fail("DFKV_SLAB_GRANULARITY must be a positive uint32");
        return;
      }
      slab_gran = 0;
    }
  }
  uint64_t parsed = 0;
  uint32_t sync_ms = 100;
  if (const char* value = std::getenv("DFKV_SLAB_TABLE_SYNC_MS")) {
    if (!ParseUnsigned(value, std::numeric_limits<uint32_t>::max(), &parsed)) {
      if (use_slab) {
        fail("DFKV_SLAB_TABLE_SYNC_MS must be a uint32");
        return;
      }
    } else {
      sync_ms = static_cast<uint32_t>(parsed);
    }
  }
  uint32_t reclaim_ms = 50;
  if (const char* value = std::getenv("DFKV_SLAB_RECLAIM_MS")) {
    if (!ParseUnsigned(value, std::numeric_limits<uint32_t>::max(), &parsed)) {
      if (use_slab) {
        fail("DFKV_SLAB_RECLAIM_MS must be a uint32");
        return;
      }
    } else {
      reclaim_ms = static_cast<uint32_t>(parsed);
    }
  }

  config_dump::RecordResolved("DFKV_STORE_ENGINE", engine_);
  config_dump::RecordResolved("DFKV_DISK_HASH_WEIGHT",
                              std::to_string(disk_weight));
  config_dump::RecordResolved("DFKV_SLAB_WRITE",
                              slab_direct ? "direct" : "buffered");
  config_dump::RecordResolved(
      "DFKV_SLAB_GRANULARITY",
      slab_gran ? std::to_string(slab_gran) : std::string("1048576"));
  config_dump::RecordResolved("DFKV_SLAB_TABLE_SYNC_MS",
                              std::to_string(sync_ms));
  config_dump::RecordResolved("DFKV_SLAB_RECLAIM_MS",
                              std::to_string(reclaim_ms));

  const size_t n = opt.cache_dirs.size();
  const uint64_t base_capacity = opt.capacity_bytes / n;
  const uint64_t capacity_remainder = opt.capacity_bytes % n;
  write_mode_ = use_slab ? "direct" : "";
  for (size_t i = 0; i < n; ++i) {
    const std::string& dir = opt.cache_dirs[i];
    const uint64_t disk_capacity =
        base_capacity + (i < capacity_remainder ? 1 : 0);
    std::unique_ptr<StoreEngine> store;
    if (use_slab) {
      DiskSlabStore::Options so;
      so.dir = dir;
      so.capacity_bytes = disk_capacity;
      so.direct_writes = slab_direct;
      if (slab_gran) so.slot_granularity = slab_gran;
      so.table_sync_ms = sync_ms;
      so.reclaim_interval_ms = reclaim_ms;
      auto slab = std::make_unique<DiskSlabStore>(so);
      slabs_.push_back(slab.get());
      if (!slab->DirectWritesActive()) write_mode_ = "buffered";
      store = std::move(slab);
    } else {
      store =
          std::make_unique<KVStore>(KVStore::Options{dir, disk_capacity});
    }
    if (!store->Healthy()) {
      fail(dir + ": " + store->StartupError());
      DFKV_LOG_ERROR("disk store startup failed: " + dir + ": " +
                     store->StartupError());
    }
    by_id_[dir] = store.get();
    disks_.push_back(std::move(store));
    ring_.AddNode(dir, disk_weight);
  }
  ring_.Build();
  if (disks_.size() > 1) {
    auto points = ring_.NodePointCounts();
    size_t min_points = SIZE_MAX;
    size_t max_points = 0;
    for (const auto& [_, count] : points) {
      if (count < min_points) min_points = count;
      if (count > max_points) max_points = count;
    }
    DFKV_LOG_INFO("disk ring: " + std::to_string(disks_.size()) +
                  " disks, weight=" + std::to_string(disk_weight) +
                  ", points min=" + std::to_string(min_points) +
                  " max=" + std::to_string(max_points));
  }
}

bool DiskCacheGroup::Healthy() const {
  if (!healthy_) return false;
  for (const auto& disk : disks_)
    if (!disk->Healthy()) return false;
  return true;
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
    sum.bind_wipes += st.bind_wipes;
    sum.steals += st.steals;
    sum.cold_steals += st.cold_steals;
    sum.watermark_evictions += st.watermark_evictions;
    sum.extent_returns += st.extent_returns;
    sum.deferred_removes += st.deferred_removes;
    sum.inflight += st.inflight;
    sum.prep_holds += st.prep_holds;
    sum.reclaimed_slots += st.reclaimed_slots;
    sum.rebalanced_extents += st.rebalanced_extents;
    sum.batched_writes += st.batched_writes;
    sum.uring_write_batches += st.uring_write_batches;
    sum.metadata_io_errors += st.metadata_io_errors;
    sum.unclean_resets += st.unclean_resets;
    sum.eviction_record_clears += st.eviction_record_clears;
    sum.record_writes += st.record_writes;
    sum.table_rebuilt += st.table_rebuilt;
    sum.rebuild_corrupt_records += st.rebuild_corrupt_records;
    sum.rebuild_rejected_records += st.rebuild_rejected_records;
    sum.capacity_bytes += st.capacity_bytes;
    sum.allocated_bytes += st.allocated_bytes;
    sum.payload_bytes += st.payload_bytes;
    sum.allocator_objects += st.allocator_objects;
    sum.committed_objects += st.committed_objects;
    sum.class_count += st.class_count;
    sum.bound_extents += st.bound_extents;
    sum.pool_extents += st.pool_extents;
    sum.failed_disks += st.failed_disks;
    sum.failed = sum.failed || st.failed;
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
