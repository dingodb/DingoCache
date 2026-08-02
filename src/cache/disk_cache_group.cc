#include "cache/disk_cache_group.h"

#include <algorithm>
#include <cstdlib>
#include <charconv>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>

#include "cache/disk_slab_store.h"
#include "common/config_dump.h"
#include "utils/log.h"
#include "utils/parallel_for.h"

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

bool ParseTenantHash(const std::string& text, uint64_t* out) {
  if (text.size() != 16) return false;
  for (char c : text) {
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
      return false;
  }
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto parsed = std::from_chars(begin, end, *out, 16);
  return parsed.ec == std::errc() && parsed.ptr == end;
}

bool ParseTenantQuotasFile(const std::string& path,
                           std::map<uint64_t, uint64_t>* quotas,
                           std::string* error) {
  std::ifstream input(path);
  if (!input.is_open()) {
    *error = "cannot open DFKV_TENANT_QUOTAS_FILE: " + path;
    return false;
  }
  std::string line;
  size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    const size_t first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') continue;
    std::istringstream fields(line.substr(first));
    std::string hash_text;
    std::string bytes_text;
    std::string extra;
    if (!(fields >> hash_text >> bytes_text) || (fields >> extra)) {
      *error = "malformed tenant quota line " + std::to_string(line_number);
      return false;
    }
    uint64_t hash = 0;
    uint64_t bytes = 0;
    if (!ParseTenantHash(hash_text, &hash) ||
        !ParseUnsigned(bytes_text.c_str(), std::numeric_limits<uint64_t>::max(),
                       &bytes)) {
      *error = "malformed tenant quota line " + std::to_string(line_number);
      return false;
    }
    if (!quotas->emplace(hash, bytes).second) {
      *error = "duplicate tenant hash on quota line " +
               std::to_string(line_number);
      return false;
    }
  }
  if (input.bad()) {
    *error = "cannot read DFKV_TENANT_QUOTAS_FILE: " + path;
    return false;
  }
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
  std::set<std::string> unique_dirs;
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
  if (const char* value = std::getenv("DFKV_TENANT_DEFAULT_QUOTA_BYTES")) {
    if (*value != '\0' &&
        !ParseUnsigned(value, std::numeric_limits<uint64_t>::max(),
                       &tenant_default_quota_bytes_)) {
      fail("DFKV_TENANT_DEFAULT_QUOTA_BYTES must be a uint64");
      return;
    }
  }
  if (const char* path = std::getenv("DFKV_TENANT_QUOTAS_FILE");
      path != nullptr && *path != '\0') {
    tenant_quotas_file_ = path;
    std::string error;
    if (!ParseTenantQuotasFile(tenant_quotas_file_, &tenant_quotas_, &error)) {
      config_dump::RecordResolved("DFKV_TENANT_QUOTAS_FILE",
                                  tenant_quotas_file_);
      config_dump::RecordResolved(
          "DFKV_TENANT_DEFAULT_QUOTA_BYTES",
          std::to_string(tenant_default_quota_bytes_));
      fail(std::move(error));
      return;
    }
  }
  for (const auto& quota : tenant_quotas_)
    tenant_quota_rejections_by_hash_.emplace(quota.first, 0);
  config_dump::RecordResolved(
      "DFKV_TENANT_QUOTAS_FILE",
      tenant_quotas_file_.empty() ? "(unset)" : tenant_quotas_file_);
  config_dump::RecordResolved("DFKV_TENANT_DEFAULT_QUOTA_BYTES",
                              std::to_string(tenant_default_quota_bytes_));
  config_dump::RecordResolved("DFKV_TENANT_QUOTAS_COUNT",
                              std::to_string(tenant_quotas_.size()));


  std::string engine = opt.engine;
  if (engine.empty()) {
    const char* value = std::getenv("DFKV_STORE_ENGINE");
    engine = (value && *value) ? value : "slab";
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
    if (opt.engine_factory) {
      store = opt.engine_factory(dir, disk_capacity);
    } else if (use_slab) {
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
    if (!store) {
      fail(dir + ": engine factory returned null");
      return;
    }
    if (!store->Healthy()) {
      fail(dir + ": " + store->StartupError());
      DFKV_LOG_ERROR("disk store startup failed: " + dir + ": " +
                     store->StartupError());
    }
    by_id_[dir] = i;
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

size_t DiskCacheGroup::RouteIndex(const BlockKey& key) const {
  if (disks_.size() == 1) return 0;
  std::string id;
  if (!ring_.Lookup(key.Filename(), &id)) return kInvalidDisk;
  auto it = by_id_.find(id);
  return it == by_id_.end() ? kInvalidDisk : it->second;
}

StoreEngine* DiskCacheGroup::Route(const BlockKey& key) const {
  const size_t index = RouteIndex(key);
  return index == kInvalidDisk ? nullptr : disks_[index].get();
}
uint64_t DiskCacheGroup::TenantLimit(uint64_t tenant_hash) const {
  const auto it = tenant_quotas_.find(tenant_hash);
  return it == tenant_quotas_.end() ? tenant_default_quota_bytes_ : it->second;
}

bool DiskCacheGroup::QuotaAllowsNewLocked(uint64_t tenant_hash,
                                          uint64_t current,
                                          uint64_t additional) const {
  const uint64_t limit = TenantLimit(tenant_hash);
  return limit == 0 ||
         (current <= limit && additional <= limit - current);
}

void DiskCacheGroup::RecordQuotaRejectionLocked(uint64_t tenant_hash) {
  tenant_quota_rejections_.fetch_add(1, std::memory_order_relaxed);
  if (tenant_quotas_.find(tenant_hash) == tenant_quotas_.end()) return;
  auto it = tenant_quota_rejections_by_hash_.find(tenant_hash);
  if (it != tenant_quota_rejections_by_hash_.end()) ++it->second;
}


Status DiskCacheGroup::Cache(const BlockKey& key, const void* data, size_t len) {
  if (data == nullptr && len != 0) return Status::kInvalid;
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  std::lock_guard<std::mutex> tenant_lock(
      tenant_stripes_[TenantStripe(key.tenant_hash)]);
  ValueMetadata metadata;
  const Status lookup = d->Lookup(key, &metadata);
  if (lookup != Status::kOk && lookup != Status::kNotFound) return lookup;
  if (lookup == Status::kNotFound &&
      !QuotaAllowsNewLocked(key.tenant_hash,
                            TenantUsedBytes(key.tenant_hash), len)) {
    RecordQuotaRejectionLocked(key.tenant_hash);
    return Status::kQuotaExceeded;
  }
  return d->Cache(key, data, len);
}

Status DiskCacheGroup::Remove(const BlockKey& key) {
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  std::lock_guard<std::mutex> tenant_lock(
      tenant_stripes_[TenantStripe(key.tenant_hash)]);
  return d->Remove(key);
}
Status DiskCacheGroup::Lookup(const BlockKey& key, ValueMetadata* out) const {
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->Lookup(key, out);
}


Status DiskCacheGroup::CacheDirect(const BlockKey& key, char* data, size_t len,
                                   size_t cap) {
  if (data == nullptr && len != 0) return Status::kInvalid;
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  std::lock_guard<std::mutex> tenant_lock(
      tenant_stripes_[TenantStripe(key.tenant_hash)]);
  ValueMetadata metadata;
  const Status lookup = d->Lookup(key, &metadata);
  if (lookup != Status::kOk && lookup != Status::kNotFound) return lookup;
  if (lookup == Status::kNotFound &&
      !QuotaAllowsNewLocked(key.tenant_hash,
                            TenantUsedBytes(key.tenant_hash), len)) {
    RecordQuotaRejectionLocked(key.tenant_hash);
    return Status::kQuotaExceeded;
  }
  return d->CacheDirect(key, data, len, cap);
}

Status DiskCacheGroup::Range(const BlockKey& key, uint64_t offset,
                             uint64_t length, std::string* out,
                             size_t* value_len) {
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->Range(key, offset, length, out, value_len);
}

Status DiskCacheGroup::RangeInto(const BlockKey& key, uint64_t offset,
                                 uint64_t length, char* dst, size_t dst_cap,
                                 size_t* out_len, size_t* value_len) {
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->RangeInto(key, offset, length, dst, dst_cap, out_len, value_len);
}

Status DiskCacheGroup::RangeDirect(
    const BlockKey& key, uint64_t offset, uint64_t length, char* io_buf,
    size_t io_cap, const char** out_data, size_t* out_len, size_t* value_len) {
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->RangeDirect(key, offset, length, io_buf, io_cap, out_data, out_len,
                        value_len);
}

Status DiskCacheGroup::RangeDirectPrep(const BlockKey& key, uint64_t offset,
                                       uint64_t length, size_t io_cap,
                                       ReadLease* out) {
  if (out == nullptr) return Status::kInvalid;
  *out = ReadLease{};
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->RangeDirectPrep(key, offset, length, io_cap, out);
}

std::vector<Status> DiskCacheGroup::CacheDirectBatch(
    const std::vector<StoreEngine::CacheBatchItem>& items) {
  std::vector<Status> out(items.size(), Status::kInvalid);
  if (items.empty()) return out;

  std::array<bool, kTenantStripes> used_stripes{};
  for (const auto& item : items)
    used_stripes[TenantStripe(item.key.tenant_hash)] = true;
  std::array<std::unique_lock<std::mutex>, kTenantStripes> locks;
  for (size_t stripe = 0; stripe < kTenantStripes; ++stripe) {
    if (used_stripes[stripe])
      locks[stripe] = std::unique_lock<std::mutex>(tenant_stripes_[stripe]);
  }

  // Admission is evaluated in caller order. Existing keys and batch-internal
  // duplicates are idempotent and consume no additional budget.
  std::vector<uint8_t> charge(items.size(), 0);
  for (size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    const size_t disk = RouteIndex(item.key);
    if (disk == kInvalidDisk || (item.data == nullptr && item.len != 0))
      continue;
    ValueMetadata metadata;
    const Status lookup = disks_[disk]->Lookup(item.key, &metadata);
    if (lookup != Status::kOk && lookup != Status::kNotFound) {
      out[i] = lookup;
      continue;
    }
    if (lookup == Status::kOk) {
      out[i] = Status::kOk;
      continue;
    }
    bool duplicate = false;
    uint64_t additional = item.len;
    for (size_t j = 0; j < i; ++j) {
      if (out[j] != Status::kOk ||
          items[j].key.tenant_hash != item.key.tenant_hash) {
        continue;
      }
      if (items[j].key == item.key) {
        duplicate = true;
        break;
      }
      if (charge[j]) {
        if (additional > std::numeric_limits<uint64_t>::max() - items[j].len) {
          additional = std::numeric_limits<uint64_t>::max();
          break;
        }
        additional += items[j].len;
      }
    }
    if (duplicate) {
      out[i] = Status::kOk;
      continue;
    }
    if (!QuotaAllowsNewLocked(item.key.tenant_hash,
                              TenantUsedBytes(item.key.tenant_hash),
                              additional)) {
      out[i] = Status::kQuotaExceeded;
      RecordQuotaRejectionLocked(item.key.tenant_hash);
      continue;
    }
    charge[i] = 1;
    out[i] = Status::kOk;
  }

  // Stable disk-index groups avoid pointer-order routing and put every result
  // directly back in the caller's original position.
  std::vector<std::vector<size_t>> groups(disks_.size());
  std::vector<size_t> participating;
  for (size_t i = 0; i < items.size(); ++i) {
    if (out[i] != Status::kOk) continue;
    const size_t disk = RouteIndex(items[i].key);
    if (groups[disk].empty()) participating.push_back(disk);
    groups[disk].push_back(i);
    out[i] = Status::kIOError;  // actual engine result replaces this default
  }

  auto run_disk = [&](size_t disk) noexcept {
    try {
      const auto& indexes = groups[disk];
      std::vector<StoreEngine::CacheBatchItem> sub;
      sub.reserve(indexes.size());
      for (size_t index : indexes) sub.push_back(items[index]);
      const std::vector<Status> statuses =
          disks_[disk]->CacheDirectBatch(sub);
      const size_t count = std::min(indexes.size(), statuses.size());
      for (size_t i = 0; i < count; ++i) out[indexes[i]] = statuses[i];
    } catch (...) {
      // Per-item defaults remain kIOError; other disks still complete.
    }
  };
  if (participating.empty()) return out;

  if (participating.size() == 1) {
    run_disk(participating[0]);  // common single-disk case: no thread overhead
    return out;
  }

  RunParallel(participating.size(), participating.size(),
              [&](size_t group_index) {
                run_disk(participating[group_index]);
              });
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
    // Keep every backend counter in this aggregation. Omitting
    // rebalanced_extents once made dfkv_slab_rebalanced_total read zero
    // fleet-wide despite observed extent movement.
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
    sum.rebuild_scanned_bytes += st.rebuild_scanned_bytes;
    sum.rebuild_scan_chunks += st.rebuild_scan_chunks;
    sum.rebuild_sparse_ranges += st.rebuild_sparse_ranges;
    sum.rebuild_sequential_fallbacks += st.rebuild_sequential_fallbacks;
    sum.rebuild_mmap_scans += st.rebuild_mmap_scans;
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

std::vector<SlabAllocator::ClassStat> DiskCacheGroup::SlabClasses() const {
  std::map<uint32_t, SlabAllocator::ClassStat> by_size;
  for (const auto* disk : slabs_) {
    for (const auto& cls : disk->ClassStats()) {
      auto& sum = by_size[cls.slot_size];
      sum.slot_size = cls.slot_size;
      sum.free_slots += cls.free_slots;
      sum.resident += cls.resident;
      sum.puts += cls.puts;
      sum.extents += cls.extents;
      sum.slots_per_extent =
          std::max(sum.slots_per_extent, cls.slots_per_extent);
      sum.useful_bytes += cls.useful_bytes;
      sum.allocated_bytes += cls.allocated_bytes;
      sum.read_touches += cls.read_touches;
      sum.read_heat += cls.read_heat;
    }
  }
  std::vector<SlabAllocator::ClassStat> out;
  out.reserve(by_size.size());
  for (auto& entry : by_size) out.push_back(entry.second);
  return out;
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
uint64_t DiskCacheGroup::TenantUsedBytes(uint64_t tenant_hash) const {
  uint64_t total = 0;
  for (const auto& disk : disks_)
    total += disk->TenantUsedBytes(tenant_hash);
  return total;
}

std::vector<DiskCacheGroup::TenantQuotaMetric>
DiskCacheGroup::ConfiguredTenantQuotaMetrics() const {
  std::vector<TenantQuotaMetric> metrics;
  metrics.reserve(tenant_quotas_.size());
  for (const auto& quota : tenant_quotas_) {
    std::lock_guard<std::mutex> tenant_lock(
        tenant_stripes_[TenantStripe(quota.first)]);
    const auto rejected =
        tenant_quota_rejections_by_hash_.find(quota.first);
    metrics.push_back(
        TenantQuotaMetric{quota.first, quota.second,
                          TenantUsedBytes(quota.first),
                          rejected == tenant_quota_rejections_by_hash_.end()
                              ? 0
                              : rejected->second});
  }
  return metrics;
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
