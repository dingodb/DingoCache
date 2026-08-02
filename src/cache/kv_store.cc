#include "cache/kv_store.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <shared_mutex>

namespace fs = std::filesystem;

namespace dfkv {

namespace {

// All block payloads go through O_DIRECT, which requires the file offset, the
// transfer length, AND the memory buffer to be aligned to (at least) the
// device's logical sector size. 4096 is a safe superset for xfs/NVMe (and any
// 512-byte-sector device), so we align everything to it.
constexpr size_t kDioAlign = 4096;

inline uint64_t AlignDown(uint64_t x) { return x & ~(uint64_t{kDioAlign} - 1); }

// Rounds up to a kDioAlign multiple; returns false on overflow.
inline bool AlignUp(uint64_t x, uint64_t* out) {
  if (x > UINT64_MAX - (kDioAlign - 1)) return false;
  *out = (x + (kDioAlign - 1)) & ~(uint64_t{kDioAlign} - 1);
  return true;
}

// Move-only RAII file descriptor (close on scope exit).
class Fd {
 public:
  Fd() = default;
  explicit Fd(int fd) : fd_(fd) {}
  ~Fd() { reset(); }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  Fd(Fd&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }
  Fd& operator=(Fd&& o) noexcept {
    if (this != &o) { reset(); fd_ = o.fd_; o.fd_ = -1; }
    return *this;
  }
  int get() const { return fd_; }
  bool valid() const { return fd_ >= 0; }
  void reset(int fd = -1) {
    // close() can fail with EINTR, but on Linux the fd is freed regardless, so a
    // retry would close an unrelated fd. Close once and discard the result.
    if (fd_ >= 0) { int r = ::close(fd_); (void)r; }
    fd_ = fd;
  }

 private:
  int fd_ = -1;
};

// RAII aligned heap buffer for O_DIRECT transfers (posix_memalign + free).
class AlignedBuf {
 public:
  // n must be > 0 and a multiple of kDioAlign (the only way we call it).
  explicit AlignedBuf(size_t n) {
    // posix_memalign returns the error code (e.g. ENOMEM) and does NOT set errno.
    if (posix_memalign(&p_, kDioAlign, n) != 0) p_ = nullptr;
  }
  ~AlignedBuf() { free(p_); }
  AlignedBuf(const AlignedBuf&) = delete;
  AlignedBuf& operator=(const AlignedBuf&) = delete;
  bool valid() const { return p_ != nullptr; }
  char* data() const { return static_cast<char*>(p_); }

 private:
  void* p_ = nullptr;
};

// Writes [data, data+len) to `path` via O_DIRECT: fallocate the (aligned) space,
// write an aligned superset from a bounce buffer, then ftruncate to the exact
// `len`. Returns true on success. The caller owns tmp-file cleanup on failure.
bool WriteFileDirect(const std::string& path, const void* data, size_t len,
                     int* out_errno) {
  if (out_errno) *out_errno = 0;  // 0 unless a syscall below fails (see ENOSPC path)
  Fd fd(::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644));
  if (!fd.valid()) { if (out_errno) *out_errno = errno; return false; }
  if (len == 0) return true;  // empty file already created + truncated by O_TRUNC

  uint64_t alen64 = 0;
  if (!AlignUp(len, &alen64)) return false;
  const size_t alen = static_cast<size_t>(alen64);

  // Reserve the space first (user-requested). fallocate is a pure optimization;
  // if the FS doesn't support it the O_DIRECT write below still extends the file,
  // so tolerate "unsupported"; treat other errors (e.g. ENOSPC) as fatal.
  if (::fallocate(fd.get(), 0, 0, static_cast<off_t>(alen)) != 0 &&
      errno != EOPNOTSUPP && errno != ENOSYS && errno != EINVAL) {
    if (out_errno) *out_errno = errno;
    return false;
  }

  AlignedBuf buf(alen);
  if (!buf.valid()) return false;
  std::memcpy(buf.data(), data, len);
  std::memset(buf.data() + len, 0, alen - len);  // zero the pad (no stale heap on disk)

  size_t done = 0;
  while (done < alen) {
    ssize_t w = ::pwrite(fd.get(), buf.data() + done, alen - done,
                         static_cast<off_t>(done));
    if (w < 0) { if (errno == EINTR) continue; if (out_errno) *out_errno = errno; return false; }
    if (w == 0) return false;  // not expected for a regular file
    done += static_cast<size_t>(w);
    // A sub-block resume offset would make the next O_DIRECT pwrite EINVAL.
    if (done < alen && (done % kDioAlign) != 0) return false;
  }

  if (::ftruncate(fd.get(), static_cast<off_t>(len)) != 0) {  // exact size
    if (out_errno) *out_errno = errno;
    return false;
  }
  return true;
}

// Same as WriteFileDirect(), but writes directly from a caller-owned aligned
// buffer. The caller allows us to zero the O_DIRECT padding bytes in-place.
bool WriteFileDirectAligned(const std::string& path, char* data, size_t len,
                            size_t cap, int* out_errno) {
  if (out_errno) *out_errno = 0;
  Fd fd(::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644));
  if (!fd.valid()) {
    if (out_errno) *out_errno = errno;
    return false;
  }
  if (len == 0) return true;
  if (!data) return false;
  if ((reinterpret_cast<uintptr_t>(data) & (kDioAlign - 1)) != 0) return false;

  uint64_t alen64 = 0;
  if (!AlignUp(len, &alen64)) return false;
  const size_t alen = static_cast<size_t>(alen64);
  if (alen > cap) return false;

  if (::fallocate(fd.get(), 0, 0, static_cast<off_t>(alen)) != 0 &&
      errno != EOPNOTSUPP && errno != ENOSYS && errno != EINVAL) {
    if (out_errno) *out_errno = errno;
    return false;
  }

  std::memset(data + len, 0, alen - len);
  size_t done = 0;
  while (done < alen) {
    ssize_t w = ::pwrite(fd.get(), data + done, alen - done,
                         static_cast<off_t>(done));
    if (w < 0) {
      if (errno == EINTR) continue;
      if (out_errno) *out_errno = errno;
      return false;
    }
    if (w == 0) return false;
    done += static_cast<size_t>(w);
    if (done < alen && (done % kDioAlign) != 0) return false;
  }

  if (::ftruncate(fd.get(), static_cast<off_t>(len)) != 0) {
    if (out_errno) *out_errno = errno;
    return false;
  }
  return true;
}

// Reads the aligned superset of [offset, offset+n) from the already-open
// O_DIRECT fd into `io_buf` (which must be kDioAlign-aligned). *out_data points
// inside io_buf at the exact requested slice. Returns true on success.
bool PreadRangeDirectTo(int fd, uint64_t offset, size_t n, char* io_buf,
                        size_t io_cap, const char** out_data) {
  if (!io_buf) return false;
  if ((reinterpret_cast<uintptr_t>(io_buf) & (kDioAlign - 1)) != 0) return false;
  const uint64_t astart = AlignDown(offset);
  uint64_t aend = 0;
  if (!AlignUp(offset + n, &aend)) return false;  // offset+n can't overflow: both bounded by fsize
  const size_t alen = static_cast<size_t>(aend - astart);
  if (alen > io_cap) return false;

  size_t total = 0;
  while (total < alen) {
    ssize_t r = ::pread(fd, io_buf + total, alen - total,
                        static_cast<off_t>(astart + total));
    if (r < 0) { if (errno == EINTR) continue; return false; }
    if (r == 0) break;  // EOF (the aligned tail can legitimately extend past EOF)
    total += static_cast<size_t>(r);
  }

  const size_t head = static_cast<size_t>(offset - astart);
  if (total < head + n) return false;  // file shrank/corrupt under us
  *out_data = io_buf + head;
  return true;
}

// Reads exactly [offset, offset+n) from the already-open O_DIRECT fd into `dst`
// (n > 0, dst holds >= n bytes). Reads the aligned superset into a bounce buffer
// and trims out the requested slice. Returns true on success.
bool PreadRangeDirect(int fd, uint64_t offset, size_t n, char* dst) {
  const uint64_t astart = AlignDown(offset);
  uint64_t aend = 0;
  if (!AlignUp(offset + n, &aend)) return false;
  const size_t alen = static_cast<size_t>(aend - astart);

  AlignedBuf buf(alen);
  if (!buf.valid()) return false;
  const char* data = nullptr;
  if (!PreadRangeDirectTo(fd, offset, n, buf.data(), alen, &data)) return false;
  std::memcpy(dst, data, n);
  return true;
}

}  // namespace

KVStore::KVStore(Options opt) : opt_(std::move(opt)) {
  if (opt_.cache_dir.empty()) {
    startup_error_ = "cache directory is empty";
    return;
  }
  if (opt_.capacity_bytes == 0) {
    startup_error_ = "capacity_bytes must be non-zero";
    return;
  }

  // Adapt the shard count to capacity: each shard should own a meaningful slice
  // (>= kMinShardBytes) so a single value can't exceed its shard's capacity.
  constexpr uint64_t kMinShardBytes = 64ull << 20;
  const size_t want = opt_.shards ? opt_.shards : 1;
  size_t fit = static_cast<size_t>(opt_.capacity_bytes / kMinShardBytes);
  if (fit < 1) fit = 1;
  const size_t n = want < fit ? want : fit;
  shards_.reserve(n);
  const uint64_t base = opt_.capacity_bytes / n;
  const uint64_t rem = opt_.capacity_bytes % n;
  for (size_t i = 0; i < n; ++i) {
    auto sh = std::make_unique<Shard>();
    sh->capacity = base + (i < rem ? 1 : 0);
    shards_.push_back(std::move(sh));
  }

  std::error_code ec;
  fs::create_directories(opt_.cache_dir, ec);
  if (ec || !fs::is_directory(opt_.cache_dir, ec)) {
    startup_error_ = "cannot create or access cache directory";
    return;
  }
  healthy_ = RebuildIndex();
}

KVStore::Shard& KVStore::ShardFor(const std::string& fname) const {
  return *shards_[std::hash<std::string>{}(fname) % shards_.size()];
}

bool KVStore::RebuildIndex() {  // constructor-time, single-threaded: no locks
  std::error_code ec;
  const fs::path root(opt_.cache_dir);
  if (!fs::exists(root, ec) || ec) {
    startup_error_ = "cache directory disappeared during startup";
    return false;
  }

  std::vector<fs::path> orphans;
  fs::recursive_directory_iterator it(root, ec);
  if (ec) {
    startup_error_ = "cannot traverse cache directory";
    return false;
  }
  for (; it != fs::recursive_directory_iterator(); it.increment(ec)) {
    if (ec) {
      startup_error_ = "cannot traverse cache directory";
      return false;
    }
    const fs::file_status symlink_status = it->symlink_status(ec);
    if (ec || fs::is_symlink(symlink_status)) {
      startup_error_ = "cache directory contains an unreadable entry or symlink";
      return false;
    }
    const fs::path rel = it->path().lexically_relative(root);
    if (rel.empty()) {
      startup_error_ = "cannot resolve cache directory entry";
      return false;
    }
    const bool regular = fs::is_regular_file(symlink_status);
    const std::string fname = it->path().filename().string();
    if (regular && fname.size() >= 4 &&
        fname.substr(fname.size() - 4) == ".tmp") {
      orphans.push_back(it->path());
      continue;
    }
    const std::string rel_text = rel.generic_string();
    const bool native_layout =
        rel_text == "blocks" || rel_text.rfind("blocks/", 0) == 0;
    if (!native_layout) {
      startup_error_ =
          "cache directory contains a different store layout: " + rel_text;
      return false;
    }
    if (!regular) {
      if (!fs::is_directory(symlink_status)) {
        startup_error_ = "cache layout contains a non-file entry";
        return false;
      }
      continue;
    }
    BlockKey parsed_key;
    if (!BlockKey::ParseFilename(fname, &parsed_key) ||
        rel_text != parsed_key.StoreKey()) {
      startup_error_ = "cache layout contains an invalid block path: " +
                       rel_text;
      return false;
    }
    const uint64_t sz = static_cast<uint64_t>(fs::file_size(it->path(), ec));
    if (ec) {
      startup_error_ = "cannot stat cached block";
      return false;
    }
    Shard& sh = ShardFor(fname);
    auto [entry, inserted] = sh.index.try_emplace(
        fname, it->path().string(), sz, parsed_key.tenant_hash);
    if (!inserted) {
      startup_error_ = "duplicate cached block filename: " + fname;
      return false;
    }
    sh.ring.push_front(fname);
    entry->second.it = sh.ring.begin();
    sh.used_bytes += sz;
    sh.tenant_used_bytes[parsed_key.tenant_hash] += sz;
  }
  if (ec) {
    startup_error_ = "cannot traverse cache directory";
    return false;
  }
  for (const auto& path : orphans) {
    ec.clear();
    if (!fs::remove(path, ec) || ec) {
      startup_error_ = "cannot reclaim orphan temporary file";
      return false;
    }
    tmp_reclaimed_.fetch_add(1, std::memory_order_relaxed);
  }
  return true;
}

// Advance the CLOCK hand one step toward the front (newer). Past the front,
// return end() so the next iteration wraps back to the tail (oldest).
static std::list<std::string>::iterator HandNext(
    std::list<std::string>& ring, std::list<std::string>::iterator it) {
  return (it == ring.begin()) ? ring.end() : std::prev(it);
}

// CLOCK second-chance eviction within one shard (exclusive lock held). A newly
// inserted entry starts unreferenced; an access (read lock) sets its bit. A
// PERSISTENT hand (sh.hand) sweeps tail->front across calls: a referenced entry
// is cleared and given a second chance (hand advances, no list reorder), an
// unreferenced one is evicted. Carrying the hand across Cache() calls amortizes
// the scan so a hot, over-capacity shard does not re-clear the whole ring on
// every write (the previous splice-to-front did). The per-call work is also
// bounded (`limit`): after sweeping ~two full cycles without freeing enough, the
// current victim is evicted regardless of its bit, guaranteeing forward progress
// and termination even under ring/index drift.
// Rename a victim block to a unique sibling ".tmp" name: a fast metadata-only
// op, so the slow block-freeing unlink can run OUTSIDE the shard lock. A failed
// rename is an I/O failure: callers retain the original index/accounting entry.
bool KVStore::RenameToTrash(const std::string& path, std::string* trash) {
  if (trash == nullptr) return false;
  *trash = path + "." +
           std::to_string(tmp_seq_.fetch_add(1, std::memory_order_relaxed)) +
           ".tmp";
  if (rename_fn_override_) {
    if (!rename_fn_override_(path, *trash)) {
      trash->clear();
      return false;
    }
    return true;
  }
  std::error_code ec;
  fs::rename(path, *trash, ec);
  if (ec) {
    trash->clear();
    return false;
  }
  return true;
}

bool KVStore::EvictLocked(Shard& sh, std::vector<std::string>* trash) {
  // size() > 1 (not !empty()): never evict a shard's last entry, so a value larger
  // than the per-shard capacity still stays cached (it just keeps the shard over).
  size_t spins = 0;
  while (sh.used_bytes > sh.capacity && sh.ring.size() > 1) {
    const size_t limit = 2 * sh.ring.size();  // recomputed: ring shrinks as we evict
    if (sh.hand == sh.ring.end()) sh.hand = std::prev(sh.ring.end());  // (re)start at tail
    auto cur = sh.hand;
    auto it = sh.index.find(*cur);
    if (it == sh.index.end()) {  // ring/index drift (shouldn't happen): drop the node
      sh.hand = HandNext(sh.ring, cur);
      sh.ring.erase(cur);
      continue;
    }
    if (it->second.referenced.load(std::memory_order_relaxed) && ++spins <= limit) {
      it->second.referenced.store(false, std::memory_order_relaxed);  // second chance
      sh.hand = HandNext(sh.ring, cur);
      continue;
    }
    std::string renamed;
    if (!RenameToTrash(it->second.path, &renamed)) return false;
    sh.hand = HandNext(sh.ring, cur);  // move off only after durable namespace detach
    trash->push_back(std::move(renamed));  // slow unlink deferred off-lock
    sh.used_bytes -= it->second.size;
    auto tenant = sh.tenant_used_bytes.find(it->second.tenant_hash);
    if (tenant != sh.tenant_used_bytes.end()) {
      tenant->second -= it->second.size;
      if (tenant->second == 0) sh.tenant_used_bytes.erase(tenant);
    }
    evictions_.fetch_add(1, std::memory_order_relaxed);
    evicted_bytes_.fetch_add(it->second.size, std::memory_order_relaxed);
    sh.ring.erase(cur);
    sh.index.erase(it);
  }
  return true;
}

// Reclaim up to `target` bytes from one shard regardless of the capacity
// watermark, for the ENOSPC self-heal path. Unlike normal capacity eviction,
// this may remove a shard's last entry: ENOSPC is filesystem-wide and retaining
// one object per shard can otherwise prevent any space from being reclaimed.
// Returns the logical bytes detached; the caller drains `trash` after unlocking.
uint64_t KVStore::ForceEvictLocked(Shard& sh, uint64_t target,
                                   std::vector<std::string>* trash,
                                   bool* io_error) {
  if (io_error) *io_error = false;
  uint64_t freed = 0;
  size_t swept = 0;
  const size_t budget = 3 * sh.ring.size() + 4;  // bounded sweep (progress guarantee)
  while (freed < target && !sh.ring.empty() && swept++ < budget) {
    if (sh.hand == sh.ring.end()) sh.hand = std::prev(sh.ring.end());
    auto cur = sh.hand;
    auto it = sh.index.find(*cur);
    if (it == sh.index.end()) {
      sh.hand = HandNext(sh.ring, cur);
      sh.ring.erase(cur);
      continue;
    }
    // Under real pressure we don't grant second chances (we need bytes now).
    std::string renamed;
    if (!RenameToTrash(it->second.path, &renamed)) {
      if (io_error) *io_error = true;
      break;
    }
    sh.hand = HandNext(sh.ring, cur);
    trash->push_back(std::move(renamed));
    sh.used_bytes -= it->second.size;
    auto tenant = sh.tenant_used_bytes.find(it->second.tenant_hash);
    if (tenant != sh.tenant_used_bytes.end()) {
      tenant->second -= it->second.size;
      if (tenant->second == 0) sh.tenant_used_bytes.erase(tenant);
    }
    freed += it->second.size;
    evictions_.fetch_add(1, std::memory_order_relaxed);
    evicted_bytes_.fetch_add(it->second.size, std::memory_order_relaxed);
    sh.ring.erase(cur);
    sh.index.erase(it);
  }
  return freed;
}

Status KVStore::Cache(const BlockKey& key, const void* data, size_t len) {
  if (!healthy_) return Status::kIOError;
  if (len == 0 || data == nullptr) return Status::kInvalid;
  const std::string fname = key.Filename();
  Shard& sh = ShardFor(fname);
  {  // best-effort early idempotent skip (avoids a needless 2.74 MiB write)
    std::shared_lock<std::shared_mutex> rl(sh.mu);
    if (sh.index.count(fname)) return Status::kOk;
  }
  // Write the payload OUTSIDE the lock so concurrent writes to the same disk run
  // in parallel (the lock only guards the in-memory index/ring). A unique tmp
  // name per writer avoids collisions; the index re-check below resolves races.
  fs::path full = fs::path(opt_.cache_dir) / key.StoreKey();
  std::error_code ec;
  fs::create_directories(full.parent_path(), ec);
  auto write_tmp = [&](fs::path* tmp, int* werr) -> bool {
    *tmp = full;
    *tmp += "." + std::to_string(tmp_seq_.fetch_add(1, std::memory_order_relaxed)) + ".tmp";
    return write_fn_override_
               ? write_fn_override_(tmp->string(), data, len, werr)
               : WriteFileDirect(tmp->string(), data, len, werr);
  };
  fs::path tmp;
  int werr = 0;
  std::vector<std::string> trash;
  if (!write_tmp(&tmp, &werr)) {
    fs::remove(tmp, ec);
    // ENOSPC self-heal: the disk filled before logical capacity did (tmp/FS
    // overhead / shared-disk tenant), so capacity-triggered eviction never
    // fires and PUTs would fail forever. Force-evict this shard and retry once.
    if (werr == ENOSPC) {
      uint64_t remaining = std::max<uint64_t>(2 * len, 64ull << 20);
      bool eviction_error = false;
      {
        std::lock_guard<std::shared_mutex> wl(sh.mu);
        const uint64_t freed =
            ForceEvictLocked(sh, remaining, &trash, &eviction_error);
        remaining = freed >= remaining ? 0 : remaining - freed;
      }
      for (const auto& candidate : shards_) {
        if (remaining == 0 || eviction_error) break;
        if (candidate.get() == &sh) continue;
        bool failed = false;
        std::lock_guard<std::shared_mutex> wl(candidate->mu);
        const uint64_t freed =
            ForceEvictLocked(*candidate, remaining, &trash, &failed);
        eviction_error = eviction_error || failed;
        remaining = freed >= remaining ? 0 : remaining - freed;
      }
      for (auto& t : trash) {
        std::error_code e2;
        fs::remove(t, e2);
      }
      if (eviction_error) return Status::kIOError;
      werr = 0;
      if (!write_tmp(&tmp, &werr)) { fs::remove(tmp, ec); return Status::kIOError; }
      enospc_evictions_.fetch_add(1, std::memory_order_relaxed);
    } else {
      return Status::kIOError;
    }
  }
  bool eviction_ok = true;
  {
    std::lock_guard<std::shared_mutex> wl(sh.mu);  // exclusive
    if (sh.index.count(fname)) { fs::remove(tmp, ec); return Status::kOk; }  // lost the race; keep first
    fs::rename(tmp, full, ec);  // atomic publish
    if (ec) { fs::remove(tmp, ec); return Status::kIOError; }
    sh.ring.push_front(fname);
    auto res =
        sh.index.try_emplace(fname, full.string(), len, key.tenant_hash);
    res.first->second.it = sh.ring.begin();  // O(1) removal handle
    sh.used_bytes += len;
    sh.tenant_used_bytes[key.tenant_hash] += len;
    eviction_ok = EvictLocked(sh, &trash);
  }
  for (auto& t : trash) { std::error_code e2; fs::remove(t, e2); }  // slow unlink off-lock
  return eviction_ok ? Status::kOk : Status::kIOError;
}

Status KVStore::Remove(const BlockKey& key) {
  if (!healthy_) return Status::kIOError;
  const std::string fname = key.Filename();
  Shard& sh = ShardFor(fname);
  std::string trash;
  {
    std::lock_guard<std::shared_mutex> wl(sh.mu);  // exclusive
    auto it = sh.index.find(fname);
    if (it == sh.index.end()) return Status::kNotFound;
    if (!RenameToTrash(it->second.path, &trash))
      return Status::kIOError;  // retain index/ring/tenant accounting
    sh.used_bytes -= it->second.size;
    auto tenant = sh.tenant_used_bytes.find(it->second.tenant_hash);
    if (tenant != sh.tenant_used_bytes.end()) {
      tenant->second -= it->second.size;
      if (tenant->second == 0) sh.tenant_used_bytes.erase(tenant);
    }
    // O(1) ring drop via the entry's own iterator (was an O(n) scan, O(n^2)
    // under RemoveMany while holding the exclusive lock). If the CLOCK hand
    // points at the victim, advance it off first (same discipline as eviction).
    auto rit = it->second.it;
    if (sh.hand == rit) sh.hand = HandNext(sh.ring, rit);
    sh.ring.erase(rit);
    sh.index.erase(it);
  }
  if (!trash.empty()) { std::error_code ec; fs::remove(trash, ec); }  // off-lock
  return Status::kOk;
}

Status KVStore::CacheDirect(const BlockKey& key, char* data, size_t len,
                            size_t cap) {
  if (!healthy_) return Status::kIOError;
  if (len == 0 || data == nullptr) return Status::kInvalid;
  const std::string fname = key.Filename();
  Shard& sh = ShardFor(fname);
  {
    std::shared_lock<std::shared_mutex> rl(sh.mu);
    if (sh.index.count(fname)) return Status::kOk;
  }

  fs::path full = fs::path(opt_.cache_dir) / key.StoreKey();
  std::error_code ec;
  fs::create_directories(full.parent_path(), ec);
  auto write_tmp = [&](fs::path* tmp, int* werr) {
    *tmp = full;
    *tmp += "." +
            std::to_string(tmp_seq_.fetch_add(1, std::memory_order_relaxed)) +
            ".tmp";
    return write_fn_override_
               ? write_fn_override_(tmp->string(), data, len, werr)
               : WriteFileDirectAligned(tmp->string(), data, len, cap, werr);
  };
  fs::path tmp;
  int werr = 0;
  std::vector<std::string> trash;
  if (!write_tmp(&tmp, &werr)) {
    fs::remove(tmp, ec);
    if (werr != ENOSPC) return Status::kIOError;
    uint64_t remaining = std::max<uint64_t>(2 * len, 64ull << 20);
    bool eviction_error = false;
    {
      std::lock_guard<std::shared_mutex> wl(sh.mu);
      const uint64_t freed =
          ForceEvictLocked(sh, remaining, &trash, &eviction_error);
      remaining = freed >= remaining ? 0 : remaining - freed;
    }
    for (const auto& candidate : shards_) {
      if (remaining == 0 || eviction_error) break;
      if (candidate.get() == &sh) continue;
      bool failed = false;
      std::lock_guard<std::shared_mutex> wl(candidate->mu);
      const uint64_t freed =
          ForceEvictLocked(*candidate, remaining, &trash, &failed);
      eviction_error = eviction_error || failed;
      remaining = freed >= remaining ? 0 : remaining - freed;
    }
    for (auto& path : trash) {
      std::error_code remove_error;
      fs::remove(path, remove_error);
    }
    if (eviction_error) return Status::kIOError;
    werr = 0;
    if (!write_tmp(&tmp, &werr)) {
      fs::remove(tmp, ec);
      return Status::kIOError;
    }
    enospc_evictions_.fetch_add(1, std::memory_order_relaxed);
  }
  bool direct_eviction_ok = true;
  {
    std::lock_guard<std::shared_mutex> wl(sh.mu);
    if (sh.index.count(fname)) {
      fs::remove(tmp, ec);
      return Status::kOk;
    }
    fs::rename(tmp, full, ec);
    if (ec) {
      fs::remove(tmp, ec);
      return Status::kIOError;
    }
    sh.ring.push_front(fname);
    auto res =
        sh.index.try_emplace(fname, full.string(), len, key.tenant_hash);
    res.first->second.it = sh.ring.begin();
    sh.used_bytes += len;
    sh.tenant_used_bytes[key.tenant_hash] += len;
    direct_eviction_ok = EvictLocked(sh, &trash);
  }
  for (auto& path : trash) {
    std::error_code remove_error;
    fs::remove(path, remove_error);
  }
  return direct_eviction_ok ? Status::kOk : Status::kIOError;
}

Status KVStore::Range(const BlockKey& key, uint64_t offset, uint64_t length,
                      std::string* out, size_t* value_len) {
  if (!healthy_) return Status::kIOError;
  // The lock protects the in-memory index/ring, NOT the bulk file read. The GET
  // hot path takes a SHARED lock (concurrent reads per shard) and only flips the
  // entry's CLOCK bit; it opens the file under the lock so a concurrent eviction
  // (exclusive lock) can't fs::remove it first — the open fd then pins the inode
  // (POSIX unlink keeps it alive until close) — then releases the lock and does
  // the 2.74 MiB O_DIRECT read concurrently.
  const std::string fname = key.Filename();
  Shard& sh = ShardFor(fname);
  Fd fd;
  uint64_t fsize = 0;
  {
    std::shared_lock<std::shared_mutex> rl(sh.mu);
    auto it = sh.index.find(fname);
    if (it == sh.index.end()) return Status::kNotFound;  // cache-only: clean miss
    fd.reset(::open(it->second.path.c_str(), O_RDONLY | O_DIRECT));
    if (!fd.valid()) return Status::kIOError;  // entry exists; open failure is I/O
    fsize = it->second.size;
    it->second.referenced.store(true, std::memory_order_relaxed);  // CLOCK touch (read lock OK)
  }
  if (offset > fsize) return Status::kInvalid;
  if (value_len) *value_len = static_cast<size_t>(fsize);
  // length==0 means the entire remainder. Clamp by subtraction so offset +
  // length cannot overflow.
  const uint64_t remainder = fsize - offset;
  const uint64_t n = std::min(length == 0 ? remainder : length, remainder);
  out->resize(n);
  if (n == 0) return Status::kOk;
  if (!PreadRangeDirect(fd.get(), offset, static_cast<size_t>(n), &(*out)[0]))
    return Status::kIOError;
  return Status::kOk;
}

Status KVStore::RangeInto(const BlockKey& key, uint64_t offset, uint64_t length,
                          char* dst, size_t dst_cap, size_t* out_len,
                          size_t* value_len) {
  *out_len = 0;
  if (!healthy_) return Status::kIOError;
  Fd fd;
  uint64_t fsize = 0;
  {  // index lookup + open under a SHARED lock; bulk read outside (see Range)
    const std::string fname = key.Filename();
    Shard& sh = ShardFor(fname);
    std::shared_lock<std::shared_mutex> rl(sh.mu);
    auto it = sh.index.find(fname);
    if (it == sh.index.end()) return Status::kNotFound;
    fd.reset(::open(it->second.path.c_str(), O_RDONLY | O_DIRECT));
    if (!fd.valid()) return Status::kIOError;
    fsize = it->second.size;
    it->second.referenced.store(true, std::memory_order_relaxed);  // CLOCK touch
  }
  if (offset > fsize) return Status::kInvalid;
  if (value_len) *value_len = static_cast<size_t>(fsize);
  // length==0 means remainder, still bounded by the destination capacity.
  const uint64_t remainder = fsize - offset;
  uint64_t n = std::min(length == 0 ? remainder : length, remainder);
  if (n > dst_cap) n = dst_cap;
  if (n == 0) return Status::kOk;  // *out_len already 0
  if (!PreadRangeDirect(fd.get(), offset, static_cast<size_t>(n), dst))
    return Status::kIOError;
  *out_len = static_cast<size_t>(n);
  return Status::kOk;
}

Status KVStore::RangeDirect(const BlockKey& key, uint64_t offset,
                            uint64_t length, char* io_buf, size_t io_cap,
                            const char** out_data, size_t* out_len,
                            size_t* value_len) {
  if (!healthy_) return Status::kIOError;
  *out_data = nullptr;
  *out_len = 0;
  Fd fd;
  uint64_t fsize = 0;
  {  // index lookup + open under a SHARED lock; bulk read outside (see Range)
    const std::string fname = key.Filename();
    Shard& sh = ShardFor(fname);
    std::shared_lock<std::shared_mutex> rl(sh.mu);
    auto it = sh.index.find(fname);
    if (it == sh.index.end()) return Status::kNotFound;
    fd.reset(::open(it->second.path.c_str(), O_RDONLY | O_DIRECT));
    if (!fd.valid()) return Status::kIOError;
    fsize = it->second.size;
    it->second.referenced.store(true, std::memory_order_relaxed);  // CLOCK touch
  }
  if (offset > fsize) return Status::kInvalid;
  if (value_len) *value_len = static_cast<size_t>(fsize);
  const uint64_t remainder = fsize - offset;
  const uint64_t n =
      std::min(length == 0 ? remainder : length, remainder);
  if (n == 0) {
    *out_data = io_buf;
    return Status::kOk;
  }
  const char* data = nullptr;
  if (!PreadRangeDirectTo(fd.get(), offset, static_cast<size_t>(n), io_buf, io_cap, &data))
    return Status::kIOError;
  *out_data = data;
  *out_len = static_cast<size_t>(n);
  return Status::kOk;
}

Status KVStore::RangeDirectPrep(const BlockKey& key, uint64_t offset,
                                uint64_t length, size_t io_cap,
                                ReadLease* out) {
  if (out == nullptr) return Status::kInvalid;
  *out = ReadLease{};
  if (!healthy_) return Status::kIOError;
  ReadLease lease;
  uint64_t fsize = 0;
  {  // index lookup + open under a SHARED lock; bulk read happens in the caller
    const std::string fname = key.Filename();
    Shard& sh = ShardFor(fname);
    std::shared_lock<std::shared_mutex> rl(sh.mu);
    auto it = sh.index.find(fname);
    if (it == sh.index.end()) return Status::kNotFound;
    int fd = ::open(it->second.path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) return Status::kIOError;
    lease = ReadLease::Adopt(fd);
    fsize = it->second.size;
    it->second.referenced.store(true, std::memory_order_relaxed);  // CLOCK touch
  }
  if (offset > fsize) return Status::kInvalid;
  lease.value_len = static_cast<size_t>(fsize);
  const uint64_t remainder = fsize - offset;
  const uint64_t n =
      std::min(length == 0 ? remainder : length, remainder);
  if (n == 0) {
    lease = ReadLease{};
    lease.value_len = static_cast<size_t>(fsize);
    *out = std::move(lease);
    return Status::kOk;
  }
  // Mirror PreadRangeDirectTo's aligned-superset math so the caller's read covers
  // exactly the bytes the synchronous path would have read.
  const uint64_t astart = AlignDown(offset);
  uint64_t aend = 0;
  if (!AlignUp(offset + n, &aend)) return Status::kIOError;
  const size_t alen = static_cast<size_t>(aend - astart);
  if (alen > io_cap) return Status::kIOError;
  lease.aligned_off = astart;
  lease.aligned_len = alen;
  lease.head = static_cast<size_t>(offset - astart);
  lease.payload_len = static_cast<size_t>(n);
  *out = std::move(lease);
  return Status::kOk;
}

bool KVStore::IsCached(const BlockKey& key) const {
  if (!healthy_) return false;
  const std::string fname = key.Filename();
  Shard& sh = ShardFor(fname);
  std::shared_lock<std::shared_mutex> rl(sh.mu);
  return sh.index.count(fname) != 0;
}
Status KVStore::Lookup(const BlockKey& key, ValueMetadata* out) const {
  if (!healthy_) return Status::kIOError;
  if (out == nullptr) return Status::kInvalid;
  const std::string fname = key.Filename();
  Shard& sh = ShardFor(fname);
  std::shared_lock<std::shared_mutex> rl(sh.mu);
  auto it = sh.index.find(fname);
  if (it == sh.index.end()) return Status::kNotFound;
  out->value_len = it->second.size;
  return Status::kOk;
}


uint64_t KVStore::UsedBytes() const {
  uint64_t total = 0;
  for (const auto& sh : shards_) {
    std::shared_lock<std::shared_mutex> rl(sh->mu);
    total += sh->used_bytes;
  }
  return total;
}
uint64_t KVStore::TenantUsedBytes(uint64_t tenant_hash) const {
  uint64_t total = 0;
  for (const auto& sh : shards_) {
    std::shared_lock<std::shared_mutex> rl(sh->mu);
    const auto it = sh->tenant_used_bytes.find(tenant_hash);
    if (it != sh->tenant_used_bytes.end()) total += it->second;
  }
  return total;
}


size_t KVStore::Count() const {
  size_t n = 0;
  for (const auto& sh : shards_) {
    std::shared_lock<std::shared_mutex> rl(sh->mu);
    n += sh->index.size();
  }
  return n;
}

}  // namespace dfkv
