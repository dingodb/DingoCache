#include "cache/disk_slab_store.h"

#include "common/config_dump.h"

#ifdef DFKV_WITH_URING
#include <liburing.h>
#endif

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <limits>
#include <sys/stat.h>
#include <vector>

#include "utils/net_util.h"
#include "utils/thread_name.h"  // PutU32/PutU64/GetU32/GetU64 (host-endian codec)

namespace fs = std::filesystem;

namespace dfkv {

namespace {
// Table-record + meta magics.
constexpr uint32_t kRecMagic = 0x424C5453u;   // "SLTB"
constexpr uint32_t kMetaMagic = 0x424C534Du;  // "SLBM"
constexpr uint32_t kFormatVersion = 1;
constexpr uint32_t kStateMagic = 0x53424C53u;  // "SLBS"
constexpr uint32_t kStateClean = 1;
constexpr uint32_t kStateDirty = 2;

// Small CRC32 (IEEE, reflected) over a byte range; enough to catch a torn record.
uint32_t Crc32(const uint8_t* p, size_t n) {
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < n; ++i) {
    crc ^= p[i];
    for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
  }
  return crc ^ 0xFFFFFFFFu;
}

bool PwriteAll(int fd, const void* buf, size_t n, uint64_t off) {
  const char* p = static_cast<const char*>(buf);
  size_t done = 0;
  while (done < n) {
    ssize_t w = ::pwrite(fd, p + done, n - done, static_cast<off_t>(off + done));
    if (w < 0) { if (errno == EINTR) continue; return false; }
    if (w == 0) return false;
    done += static_cast<size_t>(w);
  }
  return true;
}

bool PreadAll(int fd, void* buf, size_t n, uint64_t off) {
  char* p = static_cast<char*>(buf);
  size_t done = 0;
  while (done < n) {
    ssize_t r = ::pread(fd, p + done, n - done, static_cast<off_t>(off + done));
    if (r < 0) { if (errno == EINTR) continue; return false; }
    if (r == 0) return false;  // short/EOF
    done += static_cast<size_t>(r);
  }
  return true;
}

#ifdef DFKV_WITH_URING
// Per-thread batched-write submission ring (see the long comment at the use
// site in CacheDirectBatch). Store-agnostic: fds ride in the SQEs. Leaked at
// thread exit by design (flush workers are process-lifetime; retiring on the
// hygiene path recreates it lazily).
thread_local io_uring* tls_uring_w = nullptr;
#endif
}  // namespace

DiskSlabStore::DiskSlabStore(Options opt, bool* ok) : opt_(std::move(opt)) {
  if (!ValidateOptions()) {
    if (ok) *ok = false;
    return;
  }

  SlabAllocator::Options ao;
  ao.extent_bytes = opt_.extent_bytes;
  ao.num_extents = num_extents_;
  ao.align = static_cast<uint32_t>(opt_.slot_granularity);  // slot_size is a granularity multiple
  ao.max_waste = 0.25;
  // Runtime (re)bind of an extent -- initial, steal, or pool re-bind -- must
  // physically destroy the previous binding's records BEFORE any new-slot use:
  // a stale-but-CRC-valid record at a slot-grid position the new class never
  // writes would otherwise survive to the next rebuild and resurrect its old
  // key over the new occupant's bytes (cross-class poisoning). Durable (an
  // inline fdatasync) so a crash right after the rebind can't see old records;
  // rebinds are rare (workload-mix shifts), so the ms-scale sync under the
  // allocator lock is acceptable.
  ao.on_extent_bind = [this](uint32_t e) {
    const std::vector<char> zeros(
        static_cast<size_t>(max_slots_per_extent_) * kRecBytes, 0);
    if (!PwriteAll(table_fd_, zeros.data(), zeros.size(), TableOffset(e, 0)))
      return FailMetadata();
    if (::fdatasync(table_fd_) != 0) return FailMetadata();
    bind_wipes_.fetch_add(1, std::memory_order_relaxed);
    return true;
  };
  ao.on_slot_evict = [this](const SlabAllocator::SlotRef& ref) {
    const uint8_t zero[kRecBytes] = {0};
    if (!PwriteAll(table_fd_, zero, sizeof(zero),
                   TableOffset(ref.extent, ref.slot)))
      return FailMetadata();
    record_writes_.fetch_add(1, std::memory_order_relaxed);
    eviction_record_clears_.fetch_add(1, std::memory_order_relaxed);
    return true;
  };
  alloc_ = std::make_unique<SlabAllocator>(ao);

  // Phase 10: proactive watermark eviction. When used bytes cross the high
  // watermark, the reclaimer evicts globally-cold extents down to the low
  // watermark so a write burst never hits a 100%-full ring and has to
  // synchronously self-evict just-written pages (the phase-8 0%-hit cliff).
  // DFKV_SLAB_EVICT_HIGH_PCT / _LOW_PCT (of capacity); high=0 disables.
  auto env_pct = [](const char* name, unsigned dflt) -> unsigned {
    const char* v = std::getenv(name);
    if (!v || !*v) return dflt;
    long x = std::strtol(v, nullptr, 10);
    return (x >= 0 && x <= 100) ? static_cast<unsigned>(x) : dflt;
  };
  const unsigned high_pct = env_pct("DFKV_SLAB_EVICT_HIGH_PCT", 92);
  unsigned low_pct = env_pct("DFKV_SLAB_EVICT_LOW_PCT", 88);
  if (low_pct >= high_pct && high_pct > 0) low_pct = high_pct - 1;
  evict_high_bytes_ = high_pct ? opt_.capacity_bytes / 100 * high_pct : 0;
  evict_low_bytes_ = opt_.capacity_bytes / 100 * low_pct;

  ok_ = OpenOrInit();
  if (ok_ && unclean_start_) ok_ = ResetUncleanEpoch();
  if (ok_) ok_ = Rebuild();
  if (ok_) ok_ = MarkDirtyEpoch();
  if (Healthy() && opt_.table_sync_ms > 0) {
    sync_thread_ = std::thread([this] {
      NameThisThread("slab-sync");
      std::unique_lock<std::mutex> lk(sync_mu_);
      for (;;) {
        sync_cv_.wait_for(lk, std::chrono::milliseconds(opt_.table_sync_ms),
                          [this] { return sync_stop_; });
        if (sync_stop_) return;
        const uint64_t seen = record_writes_.load(std::memory_order_relaxed);
        if (seen == synced_marker_) continue;
        lk.unlock();
        const bool synced = SyncTable(seen);
        lk.lock();
        if (!synced) return;
      }
    });
  }
  // Batched-write submit path (io_uring). Only meaningful with a DFKV_WITH_URING
  // build AND direct mode; DFKV_SLAB_URING_WRITE=0 is the kill switch.
#ifdef DFKV_WITH_URING
  {
    const char* v = std::getenv("DFKV_SLAB_URING_WRITE");
    uring_write_enabled_ = !(v && *v == '0');
    config_dump::RecordResolved("DFKV_SLAB_URING_WRITE",
                                uring_write_enabled_ ? "on" : "off");
  }
#endif
  if (Healthy() && opt_.reclaim_interval_ms > 0) {
    reclaim_thread_ = std::thread([this] {
      NameThisThread("slab-reclaim");
      std::unique_lock<std::mutex> lk(reclaim_mu_);
      for (;;) {
        reclaim_cv_.wait_for(lk, std::chrono::milliseconds(opt_.reclaim_interval_ms),
                             [this] { return reclaim_stop_; });
        if (reclaim_stop_) return;
        lk.unlock();
        ReclaimTick();
        lk.lock();
      }
    });
  }
  if (ok) *ok = Healthy();
}

DiskSlabStore::~DiskSlabStore() {
  if (reclaim_thread_.joinable()) {
    {
      std::lock_guard<std::mutex> lk(reclaim_mu_);
      reclaim_stop_ = true;
    }
    reclaim_cv_.notify_all();
    reclaim_thread_.join();
  }
  if (sync_thread_.joinable()) {
    {
      std::lock_guard<std::mutex> lk(sync_mu_);
      sync_stop_ = true;
    }
    sync_cv_.notify_all();
    sync_thread_.join();
  }
  bool clean = Healthy();
  if (clean) clean = SyncPayloads();
  if (clean)
    clean = SyncTable(record_writes_.load(std::memory_order_relaxed));
  if (clean) MarkCleanEpoch();
  for (int fd : extent_fds_)
    if (fd >= 0) ::close(fd);
  for (int fd : extent_dio_fds_)
    if (fd >= 0) ::close(fd);
  if (table_fd_ >= 0) ::close(table_fd_);
  if (state_fd_ >= 0) ::close(state_fd_);
}

bool DiskSlabStore::SetStartupError(std::string error) {
  if (startup_error_.empty()) startup_error_ = std::move(error);
  return false;
}

bool DiskSlabStore::ValidateOptions() {
  if (opt_.dir.empty()) return SetStartupError("cache directory is empty");
  if (opt_.capacity_bytes == 0)
    return SetStartupError("capacity_bytes must be non-zero");
  if (opt_.extent_bytes == 0 ||
      opt_.extent_bytes > std::numeric_limits<uint32_t>::max())
    return SetStartupError("extent_bytes must fit uint32 and be non-zero");
  if (opt_.slot_granularity < 4096 ||
      opt_.slot_granularity > std::numeric_limits<uint32_t>::max() ||
      (opt_.slot_granularity % 4096) != 0)
    return SetStartupError(
        "slot_granularity must be a non-zero 4 KiB multiple fitting uint32");
  if ((opt_.extent_bytes % opt_.slot_granularity) != 0)
    return SetStartupError(
        "extent_bytes must be divisible by slot_granularity");
  if (opt_.capacity_bytes < opt_.extent_bytes ||
      (opt_.capacity_bytes % opt_.extent_bytes) != 0)
    return SetStartupError(
        "capacity_bytes must be an exact positive multiple of extent_bytes");

  const uint64_t extents = opt_.capacity_bytes / opt_.extent_bytes;
  const uint64_t slots = opt_.extent_bytes / opt_.slot_granularity;
  if (extents > std::numeric_limits<uint32_t>::max() || slots == 0 ||
      slots > std::numeric_limits<uint32_t>::max())
    return SetStartupError("slab geometry exceeds uint32 limits");
  if (extents > static_cast<uint64_t>(
                    std::numeric_limits<off_t>::max()) /
                    slots / kRecBytes)
    return SetStartupError("slots.tbl geometry exceeds off_t");

  num_extents_ = static_cast<uint32_t>(extents);
  max_slots_per_extent_ = static_cast<uint32_t>(slots);
  return true;
}

bool DiskSlabStore::OpenOrInit() {
  std::error_code ec;
  const fs::path root(opt_.dir);
  const fs::path extents_dir = root / "extents";
  fs::create_directories(root, ec);
  if (ec || !fs::is_directory(root, ec))
    return SetStartupError("cannot create or access cache directory");

  const std::string meta_path = (root / "slab_meta").string();
  const std::string tbl_path = (root / "slots.tbl").string();
  const uint64_t tbl_bytes =
      static_cast<uint64_t>(num_extents_) * max_slots_per_extent_ * kRecBytes;
  uint8_t meta[64] = {0};
  bool fresh = false;

  int mfd = ::open(meta_path.c_str(), O_RDONLY);
  if (mfd >= 0) {
    struct stat st {};
    const bool read_ok =
        ::fstat(mfd, &st) == 0 && st.st_size == static_cast<off_t>(sizeof(meta)) &&
        PreadAll(mfd, meta, sizeof(meta), 0);
    ::close(mfd);
    if (!read_ok)
      return SetStartupError("slab_meta is truncated or unreadable");
    if (net::GetU32(reinterpret_cast<char*>(meta)) != kMetaMagic)
      return SetStartupError("slab_meta magic mismatch");
    const uint32_t ver = net::GetU32(reinterpret_cast<char*>(meta) + 4);
    const uint64_t eb = net::GetU64(reinterpret_cast<char*>(meta) + 8);
    const uint64_t sg = net::GetU64(reinterpret_cast<char*>(meta) + 16);
    const uint32_t ne = net::GetU32(reinterpret_cast<char*>(meta) + 24);
    if (ver != kFormatVersion || eb != opt_.extent_bytes ||
        sg != opt_.slot_granularity || ne != num_extents_)
      return SetStartupError(
          "existing slab layout differs from requested geometry");
    ec.clear();
    if (!fs::is_directory(extents_dir, ec) || ec)
      return SetStartupError("extents directory is missing or unreadable");
  } else {
    if (errno != ENOENT)
      return SetStartupError("cannot open slab_meta");
    fresh = true;
    ec.clear();
    fs::directory_iterator root_it(root, ec);
    if (ec) return SetStartupError("cannot inspect cache directory");
    if (root_it != fs::directory_iterator())
      return SetStartupError(
          "cache directory contains a different or incomplete store layout");
    fs::create_directories(extents_dir, ec);
    if (ec || !fs::is_directory(extents_dir, ec))
      return SetStartupError("cannot create extents directory");
  }

  if (fresh) {
    net::PutU32(reinterpret_cast<char*>(meta), kMetaMagic);
    net::PutU32(reinterpret_cast<char*>(meta) + 4, kFormatVersion);
    net::PutU64(reinterpret_cast<char*>(meta) + 8, opt_.extent_bytes);
    net::PutU64(reinterpret_cast<char*>(meta) + 16, opt_.slot_granularity);
    net::PutU32(reinterpret_cast<char*>(meta) + 24, num_extents_);
    int wfd =
        ::open(meta_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (wfd < 0) return SetStartupError("cannot create slab_meta");
    const bool meta_ok = PwriteAll(wfd, meta, sizeof(meta), 0) &&
                         ::fdatasync(wfd) == 0;
    ::close(wfd);
    if (!meta_ok) return SetStartupError("cannot persist slab_meta");

    table_fd_ =
        ::open(tbl_path.c_str(), O_RDWR | O_CREAT | O_EXCL, 0644);
    if (table_fd_ < 0 ||
        ::ftruncate(table_fd_, static_cast<off_t>(tbl_bytes)) != 0 ||
        ::fdatasync(table_fd_) != 0)
      return SetStartupError("cannot create or size slots.tbl");
  } else {
    table_fd_ = ::open(tbl_path.c_str(), O_RDWR);
    if (table_fd_ < 0) return SetStartupError("cannot open slots.tbl");
    struct stat st {};
    if (::fstat(table_fd_, &st) != 0 ||
        st.st_size != static_cast<off_t>(tbl_bytes))
      return SetStartupError("slots.tbl size differs from slab geometry");
  }

  extent_fds_.assign(num_extents_, -1);
  if (opt_.direct_writes) extent_dio_fds_.assign(num_extents_, -1);
  for (uint32_t e = 0; e < num_extents_; ++e) {
    char name[32];
    std::snprintf(name, sizeof(name), "E%05u", e);
    const std::string ep = (extents_dir / name).string();
    const int flags = O_RDWR | (fresh ? (O_CREAT | O_EXCL) : 0);
    int fd = ::open(ep.c_str(), flags, 0644);
    if (fd < 0) return SetStartupError("cannot open extent " + ep);
    struct stat st {};
    bool sized = ::fstat(fd, &st) == 0;
    if (fresh) {
      sized = sized &&
              ::ftruncate(fd, static_cast<off_t>(opt_.extent_bytes)) == 0;
    } else {
      sized = sized &&
              st.st_size == static_cast<off_t>(opt_.extent_bytes);
    }
    if (!sized) {
      ::close(fd);
      return SetStartupError("extent size differs from slab geometry: " + ep);
    }
    if (opt_.direct_writes && !extent_dio_fds_.empty() &&
        ::fallocate(fd, 0, 0, static_cast<off_t>(opt_.extent_bytes)) != 0 &&
        errno != EOPNOTSUPP && errno != ENOSYS && errno != EINVAL) {
      ::close(fd);
      return SetStartupError("cannot materialize extent " + ep);
    }
    extent_fds_[e] = fd;
    if (opt_.direct_writes && !extent_dio_fds_.empty()) {
      int dfd = ::open(ep.c_str(), O_RDWR | O_DIRECT);
      if (dfd < 0) {
        for (int direct_fd : extent_dio_fds_)
          if (direct_fd >= 0) ::close(direct_fd);
        extent_dio_fds_.clear();
      } else {
        extent_dio_fds_[e] = dfd;
      }
    }
  }
  return OpenEpochState(fresh);
}

bool DiskSlabStore::WriteEpochState(uint32_t state, uint64_t epoch) {
  if (state_fd_ < 0) return false;
  uint8_t record[64] = {0};
  net::PutU32(reinterpret_cast<char*>(record), kStateMagic);
  net::PutU32(reinterpret_cast<char*>(record) + 4, kFormatVersion);
  net::PutU64(reinterpret_cast<char*>(record) + 8, epoch);
  net::PutU32(reinterpret_cast<char*>(record) + 16, state);
  net::PutU32(reinterpret_cast<char*>(record) + 60, Crc32(record, 60));
  return PwriteAll(state_fd_, record, sizeof(record), 0) &&
         ::fdatasync(state_fd_) == 0;
}

bool DiskSlabStore::OpenEpochState(bool fresh) {
  const std::string path = (fs::path(opt_.dir) / "slab_state").string();
  if (fresh) {
    state_fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL, 0644);
    if (state_fd_ < 0 || !WriteEpochState(kStateClean, 0))
      return SetStartupError("cannot create clean slab_state");
    run_epoch_ = 0;
    return true;
  }

  state_fd_ = ::open(path.c_str(), O_RDWR);
  if (state_fd_ < 0 && errno == ENOENT) {
    // Safe migration from the pre-epoch format: assume the prior process died
    // uncleanly, persist that fact first, then cold-reset slots.tbl.
    state_fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL, 0644);
    run_epoch_ = 1;
    unclean_start_ = true;
    if (state_fd_ < 0 || !WriteEpochState(kStateDirty, run_epoch_))
      return SetStartupError("cannot create migration slab_state");
    return true;
  }
  if (state_fd_ < 0) return SetStartupError("cannot open slab_state");

  struct stat st {};
  uint8_t record[64] = {0};
  if (::fstat(state_fd_, &st) != 0 ||
      st.st_size != static_cast<off_t>(sizeof(record)) ||
      !PreadAll(state_fd_, record, sizeof(record), 0))
    return SetStartupError("slab_state is truncated or unreadable");
  if (net::GetU32(reinterpret_cast<char*>(record)) != kStateMagic ||
      net::GetU32(reinterpret_cast<char*>(record) + 4) != kFormatVersion ||
      net::GetU32(reinterpret_cast<char*>(record) + 60) != Crc32(record, 60))
    return SetStartupError("slab_state checksum or format mismatch");
  const uint32_t state =
      net::GetU32(reinterpret_cast<char*>(record) + 16);
  if (state != kStateClean && state != kStateDirty)
    return SetStartupError("slab_state contains an invalid epoch state");
  run_epoch_ = net::GetU64(reinterpret_cast<char*>(record) + 8);
  unclean_start_ = state == kStateDirty;
  return true;
}

bool DiskSlabStore::ResetUncleanEpoch() {
  const uint64_t bytes =
      static_cast<uint64_t>(num_extents_) * max_slots_per_extent_ * kRecBytes;
  const size_t chunk_bytes =
      static_cast<size_t>(std::min<uint64_t>(bytes, 1ull << 20));
  const std::vector<char> zeros(chunk_bytes, 0);
  for (uint64_t offset = 0; offset < bytes; offset += chunk_bytes) {
    const size_t n =
        static_cast<size_t>(std::min<uint64_t>(chunk_bytes, bytes - offset));
    if (!PwriteAll(table_fd_, zeros.data(), n, offset)) {
      FailMetadata();
      return SetStartupError("cannot reset slots.tbl after unclean shutdown");
    }
  }
  if (::fdatasync(table_fd_) != 0) {
    FailMetadata();
    return SetStartupError("cannot persist unclean slots.tbl reset");
  }
  unclean_resets_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool DiskSlabStore::MarkDirtyEpoch() {
  if (run_epoch_ == std::numeric_limits<uint64_t>::max())
    return SetStartupError("slab_state epoch exhausted");
  ++run_epoch_;
  if (WriteEpochState(kStateDirty, run_epoch_)) return true;
  FailMetadata();
  return SetStartupError("cannot persist dirty slab_state");
}

bool DiskSlabStore::MarkCleanEpoch() {
  if (WriteEpochState(kStateClean, run_epoch_)) return true;
  return FailMetadata();
}

bool DiskSlabStore::SyncPayloads() {
  for (int fd : extent_fds_) {
    if (fd >= 0 && ::fdatasync(fd) != 0) return FailMetadata();
  }
  return true;
}

bool DiskSlabStore::Rebuild() {
  uint8_t rec[kRecBytes];
  bool repaired = false;
  for (uint32_t e = 0; e < num_extents_; ++e) {
    for (uint32_t s = 0; s < max_slots_per_extent_; ++s) {
      if (!PreadAll(table_fd_, rec, kRecBytes, TableOffset(e, s))) {
        FailMetadata();
        return SetStartupError("cannot read slots.tbl during rebuild");
      }
      const uint32_t magic =
          net::GetU32(reinterpret_cast<char*>(rec));
      if (magic != kRecMagic) {
        if (magic != 0) ++rebuild_corrupt_records_;
        continue;
      }
      const uint32_t crc = net::GetU32(reinterpret_cast<char*>(rec) + 4);
      if (Crc32(rec + 8, kRecBytes - 8) != crc) {
        ++rebuild_corrupt_records_;
        continue;
      }
      if (rec[8] != 1) continue;
      const uint32_t slot_size =
          net::GetU32(reinterpret_cast<char*>(rec) + 12);
      BlockKey key;
      key.id = net::GetU64(reinterpret_cast<char*>(rec) + 16);
      key.index = net::GetU32(reinterpret_cast<char*>(rec) + 24);
      key.size = net::GetU32(reinterpret_cast<char*>(rec) + 28);
      const uint32_t payload_len =
          net::GetU32(reinterpret_cast<char*>(rec) + 32);
      const bool geometry_ok =
          slot_size >= opt_.slot_granularity &&
          (slot_size % opt_.slot_granularity) == 0 &&
          slot_size <= opt_.extent_bytes && payload_len <= slot_size &&
          s < opt_.extent_bytes / slot_size;
      const std::string fn = key.Filename();
      if (geometry_ok && alloc_->Restore(fn, slot_size, e, s)) {
        CommitPayloadLocked(fn, payload_len);
        ++table_rebuilt_;
        continue;
      }
      ++rebuild_rejected_records_;
      uint8_t zero[kRecBytes] = {0};
      if (!PwriteAll(table_fd_, zero, kRecBytes, TableOffset(e, s))) {
        FailMetadata();
        return SetStartupError("cannot clear rejected slots.tbl record");
      }
      repaired = true;
    }
  }
  if (repaired && ::fdatasync(table_fd_) != 0) {
    FailMetadata();
    return SetStartupError("cannot persist slots.tbl rebuild repairs");
  }
  return true;
}

bool DiskSlabStore::FailMetadata() {
  metadata_io_errors_.fetch_add(1, std::memory_order_relaxed);
  metadata_failed_.store(true, std::memory_order_release);
  return false;
}

bool DiskSlabStore::SyncTable(uint64_t seen) {
  if (!Healthy()) return false;
  if (::fdatasync(table_fd_) != 0) return FailMetadata();
  synced_marker_ = seen;
  table_syncs_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool DiskSlabStore::WriteRecord(const SlabAllocator::SlotRef& r,
                                const BlockKey& key, uint32_t payload_len,
                                bool valid) {
  if (!Healthy()) return false;
  uint8_t rec[kRecBytes];
  std::memset(rec, 0, sizeof(rec));
  net::PutU32(reinterpret_cast<char*>(rec), kRecMagic);
  rec[8] = valid ? 1 : 0;
  net::PutU32(reinterpret_cast<char*>(rec) + 12, r.slot_size);
  net::PutU64(reinterpret_cast<char*>(rec) + 16, key.id);
  net::PutU32(reinterpret_cast<char*>(rec) + 24, key.index);
  net::PutU32(reinterpret_cast<char*>(rec) + 28, key.size);
  net::PutU32(reinterpret_cast<char*>(rec) + 32, payload_len);
  const uint32_t crc = Crc32(rec + 8, kRecBytes - 8);
  net::PutU32(reinterpret_cast<char*>(rec) + 4, crc);
  if (!PwriteAll(table_fd_, rec, kRecBytes, TableOffset(r.extent, r.slot)))
    return FailMetadata();
  record_writes_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool DiskSlabStore::WritePayload(const SlabAllocator::SlotRef& r, const void* data,
                                 size_t len) {
  if (len == 0) return true;
  return PwriteAll(extent_fds_[r.extent], data, len, r.offset);
}

bool DiskSlabStore::WritePayloadDirect(const SlabAllocator::SlotRef& r, char* data,
                                       size_t len) {
  const size_t alen = (len + 4095) & ~static_cast<size_t>(4095);
  std::memset(data + len, 0, alen - len);  // caller's cap covers the padding
  // Slot offsets are slot_granularity (>= 4 KiB) aligned, so offset + alen stay
  // inside the slot; the tail padding lands in bytes payload_len_ gates off.
  return PwriteAll(extent_dio_fds_[r.extent], data, alen, r.offset);
}

void DiskSlabStore::CommitPayloadLocked(const std::string& key,
                                        uint32_t payload_len) {
  auto [it, inserted] = payload_len_.try_emplace(key, payload_len);
  if (inserted) {
    committed_payload_bytes_ += payload_len;
    return;
  }
  committed_payload_bytes_ -= it->second;
  it->second = payload_len;
  committed_payload_bytes_ += payload_len;
}

uint32_t DiskSlabStore::ErasePayloadLocked(const std::string& key) {
  const auto it = payload_len_.find(key);
  if (it == payload_len_.end()) return 0;
  const uint32_t payload_len = it->second;
  committed_payload_bytes_ -= payload_len;
  payload_len_.erase(it);
  return payload_len;
}

// mu_ guards only the in-memory maps; the MB-scale payload I/O runs OUTSIDE it
// (same discipline as KVStore::Cache and RamTier::Put), so concurrent ops to one
// disk overlap in the kernel instead of serializing behind the store lock. The
// slot bytes are protected across the unlocked window by a write-pin (a pinned
// slot is never evicted/reused), and payload_len_ -- installed only after the
// I/O lands -- is the commit point readers gate on, so an in-flight key reads
// as a clean miss, never as torn bytes.
template <typename WriteFn>
Status DiskSlabStore::CacheImpl(const BlockKey& key, size_t len,
                                const WriteFn& write_payload) {
  if (!Healthy()) return Status::kIOError;
  const std::string fn = key.Filename();
  SlabAllocator::SlotRef ref;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (!Healthy()) return Status::kIOError;
    // Idempotent: committed, or mid-write by another thread (which will commit).
    if (alloc_->Contains(fn)) return Status::kOk;
    std::vector<std::string> evicted;
    if (!alloc_->Put(fn, len, &ref, &evicted)) return Status::kIOError;  // too big / all pinned
    alloc_->Pin(fn);  // write-pin for the unlocked I/O below
    inflight_[fn]++;  // guards the slot against a concurrent Remove (see ReleaseInflightLocked)
    // The allocator has already invalidated every evicted slot record through
    // on_slot_evict (or wiped the whole extent before a rebind). Remove the
    // corresponding runtime commit metadata in the same store critical section.
    for (const auto& ev : evicted)
      evicted_bytes_ += ErasePayloadLocked(ev);
  }

  const bool io_ok = write_payload(ref) &&
                     WriteRecord(ref, key, static_cast<uint32_t>(len), /*valid=*/true);

  std::lock_guard<std::mutex> lk(mu_);
  // A Remove that arrived while we wrote was deferred to us (the sole in-flight
  // holder of an uncommitted key): ReleaseInflightLocked executes it, so the key
  // must not be committed afterwards.
  const bool removed_while_writing = deferred_remove_.count(fn) > 0;
  const bool release_ok = ReleaseInflightLocked(key, fn);
  if (!io_ok || !release_ok) {
    if (!removed_while_writing) alloc_->Remove(fn);  // roll back the reservation
    return Status::kIOError;
  }
  if (removed_while_writing) return Status::kOk;  // acked put, then removed: a miss later is correct
  CommitPayloadLocked(fn, static_cast<uint32_t>(len));
  return Status::kOk;
}

Status DiskSlabStore::Cache(const BlockKey& key, const void* data, size_t len) {
  if (data == nullptr && len != 0) return Status::kInvalid;
  return CacheImpl(key, len, [&](const SlabAllocator::SlotRef& r) {
    return WritePayload(r, data, len);
  });
}

Status DiskSlabStore::CacheDirect(const BlockKey& key, char* data, size_t len,
                                  size_t cap) {
  if (data == nullptr && len != 0) return Status::kInvalid;
  return CacheImpl(key, len, [&](const SlabAllocator::SlotRef& r) {
    // O_DIRECT when enabled and the caller's buffer qualifies (RDMA recv buffers
    // are 4 KiB-aligned with padded cap); anything else falls back to buffered.
    const size_t alen = (len + 4095) & ~static_cast<size_t>(4095);
    if (!extent_dio_fds_.empty() && len != 0 &&
        (reinterpret_cast<uintptr_t>(data) & 4095) == 0 &&
        alen <= cap && alen <= r.slot_size)
      return WritePayloadDirect(r, data, len);
    if (!extent_dio_fds_.empty() && len != 0)
      dio_write_fallbacks_.fetch_add(1, std::memory_order_relaxed);
    return WritePayload(r, data, len);
  });
}


// Batched CacheDirect. Three phases mirror CacheImpl exactly, amortized:
//   L1 (one mu_): dedup + allocate + write-pin + inflight for every item;
//   IO (unlocked): zero-pad + payload writes -- ONE io_uring submit when the
//     build has uring and DFKV_SLAB_URING_WRITE isn't 0 (small-object flushing
//     is IOPS-bound at one synchronous DIO write per key), else a loop; then
//     the 64 B records (buffered pwrites, cheap) in a loop;
//   L2 (one mu_): per-item deferred-remove check, inflight release, failure
//     rollback, payload_len_ commit.
std::vector<Status> DiskSlabStore::CacheDirectBatch(
    const std::vector<CacheBatchItem>& items) {
  const size_t N = items.size();
  std::vector<Status> out(N, Status::kIOError);
  if (!Healthy() || N == 0) return out;
  struct Slot { std::string fn; SlabAllocator::SlotRef ref; bool active = false; bool io_ok = false; bool direct = false; };
  std::vector<Slot> st(N);
  // -- L1: allocate under one lock --
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t i = 0; i < N; ++i) {
      const auto& it = items[i];
      if (it.data == nullptr && it.len != 0) { out[i] = Status::kInvalid; continue; }
      st[i].fn = it.key.Filename();
      if (alloc_->Contains(st[i].fn)) { out[i] = Status::kOk; continue; }  // idempotent (incl. batch-internal dup)
      std::vector<std::string> evicted;
      if (!alloc_->Put(st[i].fn, it.len, &st[i].ref, &evicted)) { out[i] = Status::kIOError; continue; }
      alloc_->Pin(st[i].fn);
      inflight_[st[i].fn]++;
      for (const auto& ev : evicted)
        evicted_bytes_ += ErasePayloadLocked(ev);
      st[i].active = true;
      // Same eligibility test as CacheDirect: aligned buffer + padded cap.
      const size_t alen = (it.len + 4095) & ~static_cast<size_t>(4095);
      st[i].direct = !extent_dio_fds_.empty() && it.len != 0 &&
                     (reinterpret_cast<uintptr_t>(it.data) & 4095) == 0 &&
                     alen <= it.cap && alen <= st[i].ref.slot_size;
    }
  }
  // -- IO: payloads (uring one-submit when possible), then records --
#ifdef DFKV_WITH_URING
  size_t direct_n = 0;
  for (size_t i = 0; i < N; ++i) if (st[i].active && st[i].direct) direct_n++;
  bool did_uring = false;
  if (uring_write_enabled_ && direct_n >= 2) {
    // One submission ring PER THREAD: the previous single shared ring + mutex
    // serialized every flush worker behind one lock held ACROSS the blocking
    // CQE wait — at any instant only one worker was doing uring IO, which
    // capped the batched path's gain over plain pwrite at ~10% (measured
    // 45.0k -> 50.6k ops/s saturated 64 KiB PUT, 16 workers, B200 3xNVMe). A
    // ring is only a submission channel (the fds ride in each SQE), so it can
    // be thread-local and store-agnostic: no lock, no cross-worker convoy.
    // The batch-hygiene invariant is per-ring and unchanged (a batch fully
    // reaps its completions or the ring is retired). Rings leak at thread
    // exit by design — flush workers live for the process.
    if (!tls_uring_w) {
      auto* r = new io_uring;
      if (io_uring_queue_init(256, r, 0) == 0) tls_uring_w = r; else delete r;
    }
    if (tls_uring_w) {
      auto* ring = tls_uring_w;
      size_t submitted = 0;
      for (size_t i = 0; i < N; ++i) {
        if (!st[i].active || !st[i].direct) continue;
        const auto& it = items[i];
        const size_t alen = (it.len + 4095) & ~static_cast<size_t>(4095);
        std::memset(it.data + it.len, 0, alen - it.len);
        io_uring_sqe* sqe = io_uring_get_sqe(ring);
        if (!sqe) break;  // SQ full: leave the rest to the loop path
        io_uring_prep_write(sqe, extent_dio_fds_[st[i].ref.extent], it.data,
                            static_cast<unsigned>(alen), st[i].ref.offset);
        io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(static_cast<uintptr_t>(i)));  // set_data64 needs liburing>=2.2; CI ships 2.1
        st[i].io_ok = false;  // set true on CQE
        ++submitted;
      }
      if (submitted) {
        // io_uring_submit returns how many SQEs the kernel actually consumed.
        // On a short submit (rc < submitted, or rc < 0) the tail SQEs stay
        // queued in the SQ and would be pushed by the NEXT batch's submit —
        // by then their iovecs point at arena slots this flush has released.
        // Likewise, a CQE left unreaped here would be handed to the next batch
        // under a same-range user_data index and could mark an unwritten item
        // io_ok (a stale/short slot then serves reads). Invariant: a batch
        // either fully reaps the completions it submitted, or the ring is
        // retired with them; nothing may leak across batches.
        const int rc = io_uring_submit(ring);
        const size_t expect = rc > 0 ? static_cast<size_t>(rc) : 0;
        size_t done = 0;
        bool reap_ok = true;
        while (done < expect) {
          io_uring_cqe* cqe = nullptr;
          const int wrc = io_uring_wait_cqe(ring, &cqe);
          if (wrc == -EINTR) continue;         // signal: just retry the wait
          if (wrc != 0) { reap_ok = false; break; }
          // Bounds-check the completion's index BEFORE touching items[i]: a
          // corrupt/foreign user_data must not become an out-of-bounds read.
          const size_t i = static_cast<size_t>(
              reinterpret_cast<uintptr_t>(io_uring_cqe_get_data(cqe)));
          if (i < N && st[i].active && st[i].direct) {
            const auto& it = items[i];
            const size_t alen = (it.len + 4095) & ~static_cast<size_t>(4095);
            if (cqe->res == static_cast<int>(alen)) st[i].io_ok = true;
          }
          io_uring_cqe_seen(ring, cqe);
          ++done;
        }
        if (!reap_ok || expect != submitted) {
          // Unreaped CQEs or unsubmitted SQEs remain: retire the ring so they
          // die with it; this thread's next batch re-inits a fresh one. The
          // affected items keep io_ok=false and are rewritten by the
          // sequential path below (their slots are exclusively owned by this
          // flush, so the rewrite is idempotent).
          io_uring_queue_exit(ring);
          delete ring;
          tls_uring_w = nullptr;
        }
        did_uring = true;
        size_t batch_ok = 0;
        for (size_t i = 0; i < N; ++i)
          if (st[i].active && st[i].io_ok) ++batch_ok;
        if (batch_ok) {
          uring_write_batches_.fetch_add(1, std::memory_order_relaxed);
          // Count only CQE-confirmed writes: submitted-but-failed items fall
          // through to the sequential path and are tallied there (previously
          // they were double-counted as both batched and dio-fallback).
          batched_writes_.fetch_add(batch_ok, std::memory_order_relaxed);
        }
      }
    }
  }
  (void)did_uring;
#endif
  for (size_t i = 0; i < N; ++i) {
    if (!st[i].active) continue;
    const auto& it = items[i];
    if (!st[i].io_ok) {  // not written by uring (loop path / fallback / uring failure)
      bool w;
      if (st[i].direct) {
        w = WritePayloadDirect(st[i].ref, it.data, it.len);
      } else {
        if (!extent_dio_fds_.empty() && it.len != 0)
          dio_write_fallbacks_.fetch_add(1, std::memory_order_relaxed);
        w = WritePayload(st[i].ref, it.data, it.len);
      }
      st[i].io_ok = w;
    }
    if (st[i].io_ok)
      st[i].io_ok = WriteRecord(st[i].ref, it.key, static_cast<uint32_t>(it.len), /*valid=*/true);
  }
  // -- L2: commit under one lock --
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t i = 0; i < N; ++i) {
      if (!st[i].active) continue;
      const bool removed = deferred_remove_.count(st[i].fn) > 0;
      const bool release_ok =
          ReleaseInflightLocked(items[i].key, st[i].fn);
      if (!st[i].io_ok || !release_ok || !Healthy()) {
        if (!removed) alloc_->Remove(st[i].fn);
        out[i] = Status::kIOError;
        continue;
      }
      if (!removed)
        CommitPayloadLocked(st[i].fn,
                            static_cast<uint32_t>(items[i].len));
      out[i] = Status::kOk;
    }
  }
  return out;
}

Status DiskSlabStore::RangeDirect(const BlockKey& key, uint64_t offset,
                                  uint64_t length, char* io_buf, size_t io_cap,
                                  const char** out_data, size_t* out_len) {
  // Direct mode + aligned io_buf (the RDMA path's contract): O_DIRECT read of
  // the slot-absolute aligned window, payload pointer trimmed by head -- the
  // GET path stays page-cache-free. Anything else: buffered RangeInto.
  if (!extent_dio_fds_.empty() && Healthy() &&
      (reinterpret_cast<uintptr_t>(io_buf) & 4095) == 0) {
    const std::string fn = key.Filename();
    SlabAllocator::SlotRef ref;
    uint32_t plen = 0;
    if (!AcquireForRead(fn, &ref, &plen))
      return Healthy() ? Status::kNotFound : Status::kIOError;
    Status st = Status::kOk;
    size_t got = 0, head = 0;
    if (offset < plen) {
      const uint64_t n = std::min(length ? length : (plen - offset),
                                  static_cast<uint64_t>(plen - offset));
      const uint64_t abs = ref.offset + offset;
      const uint64_t aoff = abs & ~static_cast<uint64_t>(4095);
      head = static_cast<size_t>(abs - aoff);
      const size_t alen =
          (head + static_cast<size_t>(n) + 4095) & ~static_cast<size_t>(4095);
      if (alen <= io_cap) {
        if (PreadAll(extent_dio_fds_[ref.extent], io_buf, alen, aoff))
          got = static_cast<size_t>(n);
        else
          st = Status::kIOError;
      } else {  // window larger than the caller's buffer: buffered exact read
        dio_read_fallbacks_.fetch_add(1, std::memory_order_relaxed);
        if (!PreadAll(extent_fds_[ref.extent], io_buf,
                      static_cast<size_t>(std::min<uint64_t>(n, io_cap)),
                      ref.offset + offset))
          st = Status::kIOError;
        else { got = static_cast<size_t>(std::min<uint64_t>(n, io_cap)); head = 0; }
      }
    }
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (!ReleaseInflightLocked(key, fn)) st = Status::kIOError;
    }
    if (st != Status::kOk) return st;
    if (out_data) *out_data = io_buf + head;
    if (out_len) *out_len = got;
    return Status::kOk;
  }
  if (!extent_dio_fds_.empty() && Healthy())
    dio_read_fallbacks_.fetch_add(1, std::memory_order_relaxed);  // unaligned io_buf
  size_t got = 0;
  Status st = RangeInto(key, offset, length, io_buf, io_cap, &got);
  if (st == Status::kOk) { if (out_data) *out_data = io_buf; if (out_len) *out_len = got; }
  return st;
}

Status DiskSlabStore::RangeDirectPrep(const BlockKey& key, uint64_t offset,
                                      uint64_t length, size_t io_cap,
                                      RangePrep* out) {
  if (out) *out = RangePrep{};
  if (!Healthy()) return Status::kIOError;
  const std::string fn = key.Filename();
  SlabAllocator::SlotRef ref;
  uint32_t plen = 0;
  if (!AcquireForRead(fn, &ref, &plen))
    return Healthy() ? Status::kNotFound : Status::kIOError;
  auto bail = [&](Status status) {
    std::lock_guard<std::mutex> lk(mu_);
    return ReleaseInflightLocked(key, fn) ? status : Status::kIOError;
  };
  out->value_len = plen;
  if (offset >= plen) { out->payload_len = 0; return bail(Status::kOk); }  // zero-len hit, fd=-1
  const uint64_t n = std::min(length ? length : (plen - offset),
                              static_cast<uint64_t>(plen - offset));
  const uint64_t abs = ref.offset + offset;  // slot-absolute within the extent
  const uint64_t aoff = abs & ~static_cast<uint64_t>(4095);
  const size_t head = static_cast<size_t>(abs - aoff);
  const size_t alen = (head + static_cast<size_t>(n) + 4095) & ~static_cast<size_t>(4095);
  if (alen > io_cap) return bail(Status::kInvalid);  // caller uses sync RangeDirect
  // Direct mode dups the DIO twin (page-cache-free); buffered mode dups the
  // buffered fd so a page-cache-hot slot completes the read in memory. The dup
  // is the caller's to close; the PIN (not the fd) is what keeps the slot's
  // bytes intact -- extents are shared files, so an open fd pins nothing.
  int fd = ::dup(extent_dio_fds_.empty() ? extent_fds_[ref.extent]
                                         : extent_dio_fds_[ref.extent]);
  if (fd < 0) return bail(Status::kIOError);
  uint64_t tok;
  {
    std::lock_guard<std::mutex> lk(mu_);
    tok = ++prep_token_seq_;
    prep_holds_.emplace(tok, PrepHold{key, fn});
  }
  out->fd = fd;
  out->aligned_off = aoff;
  out->aligned_len = alen;
  out->head = head;
  out->payload_len = static_cast<size_t>(n);
  out->token = tok;
  return Status::kOk;
}

void DiskSlabStore::RangeRelease(uint64_t token) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = prep_holds_.find(token);
  if (it == prep_holds_.end()) return;
  ReleaseInflightLocked(it->second.key, it->second.fn);
  prep_holds_.erase(it);
}

// Reads take the lock only to resolve the slot + commit length and acquire the
// key (pin + inflight count), then pread OUTSIDE the lock. The pin keeps the
// slot away from eviction or explicit-remove reuse; the inflight count orders a
// concurrent Remove's durable invalidation before the slot becomes reusable.
bool DiskSlabStore::AcquireForRead(const std::string& fn, SlabAllocator::SlotRef* ref,
                                   uint32_t* plen) {
  if (!Healthy()) return false;
  std::lock_guard<std::mutex> lk(mu_);
  auto it = payload_len_.find(fn);
  if (it == payload_len_.end()) return false;  // in-flight write / removed: no commit
  if (!alloc_->GetAndPin(fn, ref)) return false;
  *plen = it->second;
  inflight_[fn]++;
  return true;
}

bool DiskSlabStore::ReleaseInflightLocked(const BlockKey& key,
                                          const std::string& fn) {
  alloc_->Unpin(fn);
  auto it = inflight_.find(fn);
  if (it == inflight_.end()) return true;
  if (--it->second > 0) return true;
  inflight_.erase(it);
  if (!deferred_remove_.erase(fn)) return true;
  SlabAllocator::SlotRef ref;
  if (alloc_->Get(fn, &ref)) {
    if (!WriteRecord(ref, key, 0, /*valid=*/false)) {
      ErasePayloadLocked(fn);
      return false;
    }
    alloc_->Remove(fn);
  }
  ErasePayloadLocked(fn);
  return true;
}

Status DiskSlabStore::Range(const BlockKey& key, uint64_t offset, uint64_t length,
                            std::string* out) {
  if (!Healthy()) return Status::kIOError;
  const std::string fn = key.Filename();
  SlabAllocator::SlotRef ref;
  uint32_t plen = 0;
  if (!AcquireForRead(fn, &ref, &plen))
    return Healthy() ? Status::kNotFound : Status::kIOError;
  Status st = Status::kOk;
  if (offset >= plen) {
    out->clear();
  } else {
    const uint64_t avail = plen - offset;
    const uint64_t n = std::min(length ? length : avail, avail);
    out->resize(static_cast<size_t>(n));
    if (n && !PreadAll(extent_fds_[ref.extent], &(*out)[0], static_cast<size_t>(n),
                       ref.offset + offset))
      st = Status::kIOError;
  }
  std::lock_guard<std::mutex> lk(mu_);
  if (!ReleaseInflightLocked(key, fn)) st = Status::kIOError;
  return st;
}

Status DiskSlabStore::RangeInto(const BlockKey& key, uint64_t offset, uint64_t length,
                                char* dst, size_t dst_cap, size_t* out_len) {
  if (!Healthy()) return Status::kIOError;
  const std::string fn = key.Filename();
  SlabAllocator::SlotRef ref;
  uint32_t plen = 0;
  if (!AcquireForRead(fn, &ref, &plen))
    return Healthy() ? Status::kNotFound : Status::kIOError;
  Status st = Status::kOk;
  size_t got = 0;
  if (offset < plen) {
    uint64_t n = std::min(length ? length : (plen - offset), static_cast<uint64_t>(plen - offset));
    if (n > dst_cap) n = dst_cap;
    if (n && !PreadAll(extent_fds_[ref.extent], dst, static_cast<size_t>(n), ref.offset + offset))
      st = Status::kIOError;
    else got = static_cast<size_t>(n);
  }
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (!ReleaseInflightLocked(key, fn)) st = Status::kIOError;
  }
  if (st == Status::kOk && out_len) *out_len = got;
  return st;
}

bool DiskSlabStore::IsCached(const BlockKey& key) const {
  if (!Healthy()) return false;
  std::lock_guard<std::mutex> lk(mu_);
  // Commit-gated (matches what a Range would hit): an in-flight write or a
  // deferred-removed key reads as absent.
  return payload_len_.count(key.Filename()) > 0;
}

Status DiskSlabStore::Remove(const BlockKey& key) {
  if (!Healthy()) return Status::kIOError;
  const std::string fn = key.Filename();
  std::lock_guard<std::mutex> lk(mu_);
  SlabAllocator::SlotRef ref;
  if (!alloc_->Get(fn, &ref)) return Status::kNotFound;
  if (inflight_.count(fn)) {
    // An unlocked pread/pwrite holds this slot. Defer durable invalidation and
    // physical removal to the last releaser; readers gate off immediately via
    // the payload_len_ erase. SlabAllocator's own deferred-remove state remains
    // a final defense for callers that miss this higher-level ordering.
    deferred_remove_.insert(fn);
    ++deferred_remove_total_;
    ErasePayloadLocked(fn);
    return Status::kOk;
  }
  if (!WriteRecord(ref, key, 0, /*valid=*/false))
    return Status::kIOError;
  alloc_->Remove(fn);
  ErasePayloadLocked(fn);
  return Status::kOk;
}

size_t DiskSlabStore::Count() const { return alloc_ ? alloc_->Count() : 0; }
uint64_t DiskSlabStore::UsedBytes() const {
  return alloc_ ? alloc_->UsedBytes() : 0;
}
uint64_t DiskSlabStore::Capacity() const {
  return alloc_ ? alloc_->Capacity() : 0;
}
uint64_t DiskSlabStore::Evictions() const {
  return alloc_ ? alloc_->Evictions() : 0;
}
uint64_t DiskSlabStore::EvictedBytes() const {
  std::lock_guard<std::mutex> lk(mu_);
  return evicted_bytes_;
}

// One reclaimer pass, two phases per hot class (new inserts since last pass):
//
// GROW: on a demand shift over a full store, the hot class should absorb
// capacity from cold classes instead of eating itself -- Put's inline order
// (EvictOneFrom before StealExtentFor) keeps a new class at its current size
// forever (self-eviction always succeeds once it has any unpinned resident),
// so a workload's new value size retains only a sliver of its writes. Move up
// to kGrowExtentsPerTick extents per tick from the coldest donors (no recent
// inserts, or 4x less demand) whose capacity stays above the kStripeWays
// striping floor -- that floor is the per-class quota protection.
//
// RECLAIM: then top free slots up to the demand-driven watermark (~2 intervals
// of the insert rate, capped at 1/4 of bound capacity) in bounded batches per
// lock hold, so Put never waits behind an inline CLOCK sweep.
void DiskSlabStore::ReclaimTick(
    const std::function<void()>& after_watermark_evict) {
  if (!Healthy()) return;
  // Watermark eviction is a composite mutation: allocator occupancy and the
  // payload-length commit map must change under the same store lock. Otherwise
  // a concurrent rewrite can commit a reused key between those two mutations,
  // only for this pass to erase the new commit state.
  if (evict_high_bytes_ != 0 && alloc_->UsedBytes() > evict_high_bytes_) {
    std::lock_guard<std::mutex> lk(mu_);
    if (alloc_->UsedBytes() > evict_high_bytes_) {
      std::vector<std::string> evicted;
      alloc_->EvictColdToTarget(evict_low_bytes_, /*max_extents=*/64,
                               &evicted);
      if (!evicted.empty()) {
        if (after_watermark_evict) after_watermark_evict();
        for (const auto& fn : evicted) ErasePayloadLocked(fn);
      }
    }
  }
  const auto classes = alloc_->Classes();
  if (reclaim_last_puts_.size() < classes.size())
    reclaim_last_puts_.resize(classes.size(), 0);
  std::vector<uint64_t> delta(classes.size(), 0);
  std::vector<uint32_t> extents(classes.size(), 0);
  for (size_t i = 0; i < classes.size(); ++i) {
    delta[i] = classes[i].puts - reclaim_last_puts_[i];
    reclaim_last_puts_[i] = classes[i].puts;
    extents[i] = classes[i].extents;
  }
  for (size_t i = 0; i < classes.size(); ++i) {
    if (delta[i] == 0) continue;  // no write demand on this class
    const auto& c = classes[i];
    size_t free_now = c.free_slots;
    const size_t want = std::max<size_t>(64, static_cast<size_t>(2 * delta[i]));
    // -- GROW --
    if (free_now < want && c.slots_per_extent > 0 && alloc_->PoolExtents() == 0) {
      size_t need_ext =
          (want - free_now + c.slots_per_extent - 1) / c.slots_per_extent;
      need_ext = std::min<size_t>(need_ext, kGrowExtentsPerTick);
      while (need_ext > 0) {
        size_t donor = classes.size();
        for (size_t d = 0; d < classes.size(); ++d) {  // coldest, then biggest
          if (d == i || extents[d] <= SlabAllocator::kStripeWays) continue;
          if (delta[d] != 0 && delta[d] * 4 > delta[i]) continue;  // not cold enough
          if (donor == classes.size() ||
              std::make_pair(delta[d], ~uint64_t(extents[d])) <
                  std::make_pair(delta[donor], ~uint64_t(extents[donor])))
            donor = d;
        }
        if (donor == classes.size()) break;
        std::vector<std::string> evicted;
        bool ok;
        {
          std::lock_guard<std::mutex> lk(mu_);
          ok = alloc_->StealFrom(donor, i, &evicted);
          for (const auto& ev : evicted)
            evicted_bytes_ += ErasePayloadLocked(ev);
        }
        if (!ok) { extents[donor] = 0; continue; }  // donor all pinned: try next
        extents[donor]--;
        rebalanced_.fetch_add(1, std::memory_order_relaxed);
        free_now += c.slots_per_extent;
        --need_ext;
      }
    }
    // -- RECLAIM --
    const size_t capacity = c.resident + free_now;
    const size_t target = std::min(want, capacity / 4);
    if (free_now >= target) continue;
    size_t budget = 4096;  // per-tick per-class bound
    for (;;) {
      std::vector<std::string> evicted;
      const size_t batch = std::min<size_t>(128, budget);
      size_t got;
      {
        std::lock_guard<std::mutex> lk(mu_);
        got = alloc_->ReclaimClass(i, target, batch, &evicted);
        for (const auto& ev : evicted)
          evicted_bytes_ += ErasePayloadLocked(ev);
      }
      if (got > 0) reclaimed_.fetch_add(got, std::memory_order_relaxed);
      budget -= std::min(budget, got);
      // A partial batch means ReclaimClass stopped for an internal reason --
      // target reached, everything pinned, or an extent went back to the pool
      // (its cascade-shrink guard). Re-invoking would defeat that guard.
      if (got < batch || budget == 0) break;
    }
  }
}

DiskSlabStore::Stats DiskSlabStore::GetStats() const {
  Stats st;
  st.dio_write_fallbacks =
      dio_write_fallbacks_.load(std::memory_order_relaxed);
  st.dio_read_fallbacks =
      dio_read_fallbacks_.load(std::memory_order_relaxed);
  st.table_syncs = table_syncs_.load(std::memory_order_relaxed);
  st.bind_wipes = bind_wipes_.load(std::memory_order_relaxed);
  st.metadata_io_errors =
      metadata_io_errors_.load(std::memory_order_relaxed);
  st.unclean_resets = unclean_resets_.load(std::memory_order_relaxed);
  st.eviction_record_clears =
      eviction_record_clears_.load(std::memory_order_relaxed);
  st.record_writes = record_writes_.load(std::memory_order_relaxed);
  st.table_rebuilt = table_rebuilt_;
  st.rebuild_corrupt_records = rebuild_corrupt_records_;
  st.rebuild_rejected_records = rebuild_rejected_records_;
  st.capacity_bytes =
      static_cast<uint64_t>(opt_.extent_bytes) * num_extents_;
  st.failed = !Healthy();
  st.failed_disks = st.failed ? 1 : 0;
  if (alloc_) {
    st.steals = alloc_->Steals();
    st.cold_steals = alloc_->ColdSteals();
    st.watermark_evictions = alloc_->WatermarkEvictions();
    st.extent_returns = alloc_->ExtentReturns();
    st.allocated_bytes = alloc_->UsedBytes();
    st.allocator_objects = alloc_->Count();
    const auto classes = alloc_->Classes();
    st.class_count = classes.size();
    for (const auto& cls : classes) st.bound_extents += cls.extents;
    st.pool_extents = alloc_->PoolExtents();
  }
  std::lock_guard<std::mutex> lk(mu_);
  st.payload_bytes = committed_payload_bytes_;
  st.committed_objects = payload_len_.size();
  st.deferred_removes = deferred_remove_total_;
  st.inflight = inflight_.size();
  st.prep_holds = prep_holds_.size();
  st.reclaimed_slots = reclaimed_.load(std::memory_order_relaxed);
  st.rebalanced_extents = rebalanced_.load(std::memory_order_relaxed);
  st.batched_writes = batched_writes_.load(std::memory_order_relaxed);
  st.uring_write_batches =
      uring_write_batches_.load(std::memory_order_relaxed);
  return st;
}

}  // namespace dfkv
