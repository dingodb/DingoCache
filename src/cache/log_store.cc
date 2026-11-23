#include "cache/log_store.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>

#include "common/version.h"
#include "utils/crc32c.h"
#include "utils/env_parse.h"
#include "utils/log.h"

namespace fs = std::filesystem;

namespace dfkv {

using lse::kBlockBytes;

namespace {

constexpr uint64_t kIncarnationRange = 1024;
constexpr uint64_t kLsnRange = 1ull << 20;
constexpr uint64_t kMinSegmentBytes = 4ull << 20;
constexpr uint64_t kMaxSegmentBytes = 1ull << 30;
constexpr uint32_t kMinSegments = 8;
constexpr size_t kSealerQueue = 2;
constexpr const char* kSuperName = "lse_super";
constexpr const char* kJournalName = "lse_journal";
constexpr const char* kMarkerName = "lse_initializing";
constexpr const char* kSegmentsDir = "segments";

uint64_t AlignDown(uint64_t v) { return v & ~(kBlockBytes - 1); }
uint64_t AlignUp(uint64_t v) { return (v + kBlockBytes - 1) & ~(kBlockBytes - 1); }
bool IsPowerOfTwo(uint64_t v) { return v != 0 && (v & (v - 1)) == 0; }

struct AlignedBuf {
  char* p = nullptr;
  explicit AlignedBuf(size_t n) {
    if (::posix_memalign(reinterpret_cast<void**>(&p), kBlockBytes,
                         n == 0 ? kBlockBytes : n) != 0)
      p = nullptr;
  }
  ~AlignedBuf() { std::free(p); }
  AlignedBuf(const AlignedBuf&) = delete;
  AlignedBuf& operator=(const AlignedBuf&) = delete;
};

// Room the summary of a segment with `records` records will take (§6.1); its
// checksum is in the trailer.
uint64_t SummaryRoom(uint64_t records) {
  return AlignUp(records * lse::kSummaryEntryBytes);
}

bool PwriteAll(lse::IoBackend* io, int fd, const struct iovec* iov, int n,
               uint64_t offset, bool dsync) {
  size_t total = 0;
  for (int i = 0; i < n; ++i) total += iov[i].iov_len;
  std::vector<struct iovec> rest(iov, iov + n);
  size_t done = 0;
  while (done < total) {
    const ssize_t w = io->Pwritev(fd, rest.data(), static_cast<int>(rest.size()),
                                  static_cast<off_t>(offset + done), dsync);
    if (w <= 0) {
      if (w == 0) errno = EIO;
      return false;
    }
    done += static_cast<size_t>(w);
    size_t skip = static_cast<size_t>(w);
    while (!rest.empty() && skip >= rest.front().iov_len) {
      skip -= rest.front().iov_len;
      rest.erase(rest.begin());
    }
    if (!rest.empty() && skip != 0) {
      rest.front().iov_base = static_cast<char*>(rest.front().iov_base) + skip;
      rest.front().iov_len -= skip;
    }
  }
  return true;
}

bool PwriteBlock(lse::IoBackend* io, int fd, const void* data, size_t len,
                 uint64_t offset, bool dsync) {
  AlignedBuf buf(AlignUp(len));
  if (buf.p == nullptr) return false;
  std::memset(buf.p, 0, AlignUp(len));
  std::memcpy(buf.p, data, len);
  struct iovec iov = {buf.p, static_cast<size_t>(AlignUp(len))};
  return PwriteAll(io, fd, &iov, 1, offset, dsync);
}

bool PreadAll(lse::IoBackend* io, int fd, void* buf, size_t n,
              uint64_t offset) {
  size_t done = 0;
  while (done < n) {
    const ssize_t r = io->Pread(fd, static_cast<char*>(buf) + done, n - done,
                                static_cast<off_t>(offset + done));
    if (r < 0) return false;
    if (r == 0) {
      std::memset(static_cast<char*>(buf) + done, 0, n - done);  // past EOF
      return true;
    }
    done += static_cast<size_t>(r);
  }
  return true;
}

lse::Uuid RandomUuid() {
  std::random_device rd;
  lse::Uuid u;
  for (size_t i = 0; i < u.size(); i += 4) {
    const uint32_t v = rd();
    std::memcpy(u.data() + i, &v, 4);
  }
  return u;
}

int OpenJournalFile(lse::IoBackend* io, const std::string& path, int flags) {
  int fd = io->Open(path, flags | O_DIRECT, 0644);
  if (fd < 0 && errno == EINVAL) fd = io->Open(path, flags, 0644);  // tmpfs
  return fd;
}

}  // namespace

// ---- the metadata journal's writer (design §4.5) ---------------------------
// One thread per store. Entries come in through a queue, each with a ticket;
// the thread writes them as 4 KiB blocks, one block per synchronous
// pwritev2(RWF_DSYNC), and advances the durable ticket after each write. It
// never takes the store's mutex, so the store may enqueue while holding it.
// It owns the journal's descriptor.
class LogStore::Journal {
 public:
  Journal(LogStore* store, int fd, uint64_t offset, uint64_t next_seq)
      : store_(store),
        io_(store->io_.get()),
        fd_(fd),
        uuid_(store->sb_.store_uuid),
        offset_(offset),
        next_seq_(next_seq) {
    thread_ = std::thread([this] { Loop(); });
  }
  ~Journal() {
    Stop();
    if (fd_ >= 0) io_->Close(fd_);
  }

  // Called with the store mutex held, so journal order follows store order.
  uint64_t Enqueue(const lse::JournalEntry& e) {
    std::lock_guard<std::mutex> lk(mu_);
    queue_.push_back({e, ++enqueued_});
    cv_.notify_one();
    return enqueued_;
  }
  uint64_t LastTicket() {
    std::lock_guard<std::mutex> lk(mu_);
    return enqueued_;
  }
  bool WaitDurable(uint64_t ticket) {
    std::unique_lock<std::mutex> lk(mu_);
    done_cv_.wait(lk, [&] { return durable_ >= ticket || failed_; });
    return durable_ >= ticket;
  }
  bool Failed() {
    std::lock_guard<std::mutex> lk(mu_);
    return failed_;
  }
  // Writes what is queued, then ends the thread.
  void Stop() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
      cv_.notify_one();
    }
    if (thread_.joinable()) thread_.join();
  }

 private:
  struct Queued {
    lse::JournalEntry entry;
    uint64_t ticket;
  };
  void Fail(std::unique_lock<std::mutex>& lk, int err) {
    failed_ = true;
    DFKV_LOG_ERROR(std::string("lse: journal write failed: ") +
                   std::strerror(err));
    done_cv_.notify_all();
    // Nothing more is written: an acknowledged kill is never followed by
    // a lost one (design §4.5).
    cv_.wait(lk, [&] { return stop_; });
  }
  void Loop() {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
      cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
      if (queue_.empty()) return;  // stopping, nothing left
      lse::JournalBlock block;
      block.store_uuid = uuid_;
      block.sequence = next_seq_;
      uint64_t last = 0;
      while (!queue_.empty() &&
             block.entries.size() < lse::kJournalEntriesPerBlock) {
        block.entries.push_back(queue_.front().entry);
        last = queue_.front().ticket;
        queue_.pop_front();
      }
      const uint64_t offset = offset_;
      lk.unlock();
      store_->Hook("journal.before_write");
      lse::Block bytes;
      // The loop above never takes more than a block holds.
      bool ok = lse::EncodeJournalBlock(block, &bytes) &&
                PwriteBlock(io_, fd_, bytes.data(), kBlockBytes, offset,
                            /*dsync=*/true);
      const int err = errno;
      lk.lock();
      if (!ok) return Fail(lk, err);
      offset_ += kBlockBytes;
      ++next_seq_;
      durable_ = last;
      done_cv_.notify_all();
    }
  }
  LogStore* const store_;
  lse::IoBackend* io_;
  int fd_;
  const lse::Uuid uuid_;
  std::mutex mu_;
  std::condition_variable cv_, done_cv_;
  std::deque<Queued> queue_;
  uint64_t enqueued_ = 0;
  uint64_t durable_ = 0;
  uint64_t offset_;
  uint64_t next_seq_;
  bool failed_ = false;
  bool stop_ = false;
  std::thread thread_;
};

// ---- settings and geometry -------------------------------------------------
void LogStore::ApplyEnvironment(Options* o) {
  auto env = [](const char* name, uint64_t dflt, uint64_t lo, uint64_t hi) {
    envparse::U64Spec spec;
    spec.fallback = dflt;
    spec.min = lo;
    spec.max = hi;
    return envparse::EnvU64(name, spec);
  };
  o->sync_ms = static_cast<uint32_t>(env("DFKV_LSE_SYNC_MS", o->sync_ms, 1, 3600000));
}

bool LogStore::GeometryFor(uint64_t capacity, uint64_t segment_override,
                           Geometry* out, std::string* error) {
  uint64_t seg = segment_override;
  if (seg != 0) {
    if (!IsPowerOfTwo(seg) || seg < kMinSegmentBytes || seg > kMaxSegmentBytes) {
      *error = "segment size must be a power of two in [4 MiB, 1 GiB]";
      return false;
    }
  } else {
    seg = kMaxSegmentBytes;
    while (seg > kMinSegmentBytes && seg > capacity / 32) seg >>= 1;
  }
  const uint64_t count = capacity / seg;
  if (count < kMinSegments) {
    *error = "capacity " + std::to_string(capacity) +
             " is below the log engine's minimum of 8 segments of " +
             std::to_string(seg) + " bytes";
    return false;
  }
  if (count > UINT32_MAX) {
    *error = "capacity too large";
    return false;
  }
  out->segment_bytes = seg;
  out->segment_count = static_cast<uint32_t>(count);
  return true;
}

// ---- construction, init and recovery ---------------------------------------
LogStore::LogStore(Options opt, bool* ok) : opt_(std::move(opt)) {
  io_ = opt_.io ? opt_.io : std::shared_ptr<lse::IoBackend>(
                                lse::MakeSystemIoBackend());
  ok_ = [&] {
    if (opt_.dir.empty()) return SetError("no directory");
    std::error_code ec;
    fs::create_directories(opt_.dir, ec);
    if (ec) return SetError("cannot create " + opt_.dir + ": " + ec.message());
    if (!LockDirectory()) return false;
    // Re-checked under the lock (design §10.1): what the directory holds.
    const fs::path root(opt_.dir);
    bool foreign = false, marker = false, super = false;
    std::string first_foreign;
    for (fs::directory_iterator it(root, ec), end; !ec && it != end;
         it.increment(ec)) {
      const std::string name = it->path().filename().string();
      if (name == "lost+found") {
        // A file system check put files there: refused like foreign
        // content (design §10.1), the directory left as it is.
        std::error_code lf_ec;
        if (!it->is_directory(lf_ec) || lf_ec ||
            !fs::is_empty(it->path(), lf_ec) || lf_ec) {
          foreign = true;
          if (first_foreign.empty()) first_foreign = "lost+found (not empty)";
        }
        continue;
      }
      if (name == kMarkerName) {
        marker = true;
        continue;
      }
      if (name == kSuperName) super = true;
      if (name == kSuperName || name == kJournalName || name == kSegmentsDir ||
          name == std::string(kJournalName) + ".new")
        continue;
      foreign = true;
      if (first_foreign.empty()) first_foreign = name;
    }
    if (ec) return SetError("cannot list " + opt_.dir + ": " + ec.message());
    if (foreign)
      return SetError("cache directory holds files of no log store (" +
                      first_foreign +
                      "); inspect them, nothing was changed");
    if (super && !marker) return OpenStore();
    if (!marker && (fs::exists(root / kJournalName, ec) ||
                    fs::exists(root / kSegmentsDir, ec)))
      return SetError(
          "log store without lse_super; inspect it, nothing was changed");
    Geometry g;
    std::string error;
    if (!GeometryFor(opt_.capacity_bytes, opt_.segment_bytes, &g, &error))
      return SetError(error);
    return InitStore(g);
  }();
  if (ok_ && !opt_.defer_activation) ok_ = Activate();
  if (ok) *ok = ok_;
}

LogStore::~LogStore() {
  if (activated_.load() && !closed_.load()) Close();
  journal_.reset();
  for (auto& s : segments_) {
    if (s.fd >= 0) io_->Close(s.fd);
    if (s.dio_fd >= 0) io_->Close(s.dio_fd);
  }
  if (super_fd_ >= 0) io_->Close(super_fd_);
  if (journal_fd_ >= 0) io_->Close(journal_fd_);
  if (dir_fd_ >= 0) ::close(dir_fd_);  // releases the lock
}

bool LogStore::SetError(std::string error) {
  if (startup_error_.empty()) startup_error_ = std::move(error);
  return false;
}
bool LogStore::SetErrno(std::string what, int err) {
  return SetError(what + ": " + std::strerror(err));
}

bool LogStore::LockDirectory() {
  dir_fd_ = ::open(opt_.dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (dir_fd_ < 0) return SetErrno("cannot open " + opt_.dir, errno);
  if (::flock(dir_fd_, LOCK_EX | LOCK_NB) != 0) {
    if (errno == EWOULDBLOCK)
      return SetError(opt_.dir + " is in use by another process");
    return SetErrno("cannot lock " + opt_.dir, errno);
  }
  return true;
}

std::string LogStore::SegmentPath(uint32_t index) const {
  char name[32];
  std::snprintf(name, sizeof(name), "G%06u", index);
  return (fs::path(opt_.dir) / kSegmentsDir / name).string();
}

bool LogStore::SyncDir(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fd < 0) return false;
  const bool ok = io_->Fsync(fd) == 0;
  ::close(fd);
  return ok;
}

bool LogStore::WriteSuperblock() {
  ++sb_.sequence;
  const lse::Block b = lse::EncodeSuperblock(sb_);
  return PwriteBlock(io_.get(), super_fd_, b.data(), kBlockBytes,
                     (sb_.sequence % 2) * kBlockBytes, /*dsync=*/true);
}

bool LogStore::OpenSegmentFile(uint32_t i, bool create) {
  const std::string path = SegmentPath(i);
  Segment& s = segments_[i];
  s.fd = io_->Open(path, O_RDWR | (create ? O_CREAT : 0), 0644);
  if (s.fd < 0) return SetErrno("cannot open segment " + path, errno);
  struct stat st {};
  if (::fstat(s.fd, &st) != 0) return SetErrno("cannot inspect " + path, errno);
  if (create) {
    const int rc =
        opt_.direct_writes
            ? io_->Fallocate(s.fd, 0, static_cast<off_t>(sb_.segment_bytes))
            : io_->Ftruncate(s.fd, static_cast<off_t>(sb_.segment_bytes));
    if (rc != 0) return SetErrno("cannot size segment " + path, errno);
    // The size is metadata: durable only with the file's own sync.
    if (io_->Fdatasync(s.fd) != 0)
      return SetErrno("cannot sync segment " + path, errno);
  } else if (static_cast<uint64_t>(st.st_size) != sb_.segment_bytes) {
    return SetError("segment " + path + " has the wrong size");
  }
  if (opt_.direct_writes) {
    s.dio_fd = io_->Open(path, O_RDWR | O_DIRECT, 0);
    if (s.dio_fd < 0 && errno == EINVAL) {
      // The filesystem takes no O_DIRECT (tmpfs): buffered, as slab does.
      DFKV_LOG_WARN("lse: " + opt_.dir +
                    ": O_DIRECT not supported here; writing buffered");
      opt_.direct_writes = false;
      return true;
    }
    if (s.dio_fd < 0) return SetErrno("cannot open " + path + " for O_DIRECT", errno);
    // The barrier (design §8.3 step 2): fallocate waits for direct I/O a
    // killed predecessor may still have in flight on this file.
    if (!create &&
        io_->Fallocate(s.fd, 0, static_cast<off_t>(sb_.segment_bytes)) != 0)
      return SetErrno("cannot fallocate segment " + path, errno);
  }
  return true;
}

bool LogStore::OpenSegmentFiles(bool create) {
  segments_.assign(sb_.segment_count, Segment{});
  for (uint32_t i = 0; i < sb_.segment_count; ++i)
    if (!OpenSegmentFile(i, create)) return false;
  return true;
}

bool LogStore::InitStore(const Geometry& g) {
  const fs::path root(opt_.dir);
  const std::string marker = (root / kMarkerName).string();
  int fd = io_->Open(marker, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) return SetErrno("cannot create " + marker, errno);
  io_->Close(fd);
  if (io_->Fsync(dir_fd_) != 0) return SetErrno("cannot sync " + opt_.dir, errno);
  Hook("init.after_marker");
  const std::string segdir = (root / kSegmentsDir).string();
  if (io_->Mkdir(segdir, 0755) != 0 && errno != EEXIST)
    return SetErrno("cannot create " + segdir, errno);

  sb_ = lse::Superblock{};
  sb_.incompat_features = lse::kIncompatDevelopmentFormat;
  sb_.store_uuid = RandomUuid();
  sb_.segment_bytes = g.segment_bytes;
  sb_.segment_count = g.segment_count;
  sb_.incarnation_reserved = uint64_t(g.segment_count) + 1 + kIncarnationRange;
  sb_.lsn_reserved = 1 + kLsnRange;
  sb_.created_by = Version();
  if (!OpenSegmentFiles(/*create=*/true)) return false;
  next_incarnation_ = 1;
  for (uint32_t i = 0; i < g.segment_count; ++i)
    if (!FreshHeader(i, /*empty=*/true))
      return SetErrno("cannot write a segment header", errno);
  if (!SyncDir(segdir)) return SetErrno("cannot sync " + segdir, errno);

  journal_fd_ = OpenJournalFile(io_.get(), (root / kJournalName).string(),
                                O_RDWR | O_CREAT | O_TRUNC);
  if (journal_fd_ < 0) return SetErrno("cannot create the journal", errno);
  if (io_->Fdatasync(journal_fd_) != 0)
    return SetErrno("cannot sync the journal", errno);
  super_fd_ = io_->Open((root / kSuperName).string(),
                        O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (super_fd_ < 0) return SetErrno("cannot create lse_super", errno);
  if (!WriteSuperblock() || !WriteSuperblock())  // both copies
    return SetErrno("cannot write lse_super", errno);
  if (io_->Fsync(dir_fd_) != 0) return SetErrno("cannot sync " + opt_.dir, errno);
  Hook("init.after_super");
  if (io_->Unlink(marker) != 0) return SetErrno("cannot remove " + marker, errno);
  if (io_->Fsync(dir_fd_) != 0) return SetErrno("cannot sync " + opt_.dir, errno);

  next_incarnation_ = sb_.incarnation_reserved;
  next_lsn_ = sb_.lsn_reserved;
  for (uint32_t i = 0; i < g.segment_count; ++i) free_.push_back(i);
  return true;
}

bool LogStore::ReadSuperblock() {
  AlignedBuf buf(2 * kBlockBytes);
  if (buf.p == nullptr) return SetError("out of memory");
  if (!PreadAll(io_.get(), super_fd_, buf.p, 2 * kBlockBytes, 0))
    return SetErrno("cannot read lse_super", errno);
  bool have = false;
  lse::DecodeError why = lse::DecodeError::kBadMagic;
  for (int copy = 0; copy < 2; ++copy) {
    lse::Superblock sb;
    const auto e = lse::DecodeSuperblock(
        reinterpret_cast<const uint8_t*>(buf.p) + copy * kBlockBytes,
        /*allow_development=*/true, &sb);
    if (e != lse::DecodeError::kOk) {
      if (why == lse::DecodeError::kBadMagic) why = e;
      continue;
    }
    if (!have || sb.sequence > sb_.sequence) sb_ = sb;
    have = true;
  }
  if (!have)
    return SetError(std::string("lse_super is not usable: ") +
                    lse::DecodeErrorName(why));
  return true;
}

bool LogStore::ReadJournal() {
  struct stat st {};
  if (::fstat(journal_fd_, &st) != 0) {
    needs_reset_ = true;  // an unreadable journal resets (design §4.5)
    return true;
  }
  const uint64_t size = static_cast<uint64_t>(st.st_size);
  AlignedBuf buf(kBlockBytes);
  if (buf.p == nullptr) return SetError("out of memory");
  uint64_t offset = 0, seq = 1;
  bool stopped = false;
  for (; offset + kBlockBytes <= size; offset += kBlockBytes) {
    if (!PreadAll(io_.get(), journal_fd_, buf.p, kBlockBytes, offset)) {
      DFKV_LOG_ERROR("lse: " + opt_.dir + ": journal read error; resetting");
      needs_reset_ = true;
      recovery_.journal_damaged = true;
      return true;
    }
    lse::JournalBlock jb;
    const auto e = lse::DecodeJournalBlock(
        reinterpret_cast<const uint8_t*>(buf.p), sb_.store_uuid, &jb);
    if (!stopped) {
      if (e != lse::DecodeError::kOk || jb.sequence != seq) {
        stopped = true;
        journal_stop_offset_ = offset;
        journal_next_seq_ = seq;
        continue;
      }
      ++recovery_.journal_blocks;
      ++seq;
      for (const auto& entry : jb.entries)
        if (entry.type == lse::kJournalKill)
          kills_.insert({entry.segment_index, entry.incarnation,
                         entry.offset_blocks});
    } else if (e == lse::DecodeError::kOk && jb.sequence >= seq) {
      // A valid block beyond the stopping point: a block inside the durable
      // range was damaged after it was written.
      DFKV_LOG_ERROR("lse: " + opt_.dir +
                     ": journal damaged inside its durable range; the store "
                     "will be reset to empty");
      needs_reset_ = true;
      recovery_.journal_damaged = true;
      return true;
    }
  }
  if (!stopped) {
    journal_stop_offset_ = offset;
    journal_next_seq_ = seq;
  }
  return true;
}

// The trailer and summary of a sealed segment (§8.3 step 4): `*loaded` false
// when there is none valid for the current incarnation (the caller walks).
bool LogStore::LoadSummary(uint32_t index,
                           std::vector<lse::SummaryEntry>* records,
                           bool* loaded) {
  *loaded = false;
  Segment& s = segments_[index];
  const int fd = s.dio_fd >= 0 ? s.dio_fd : s.fd;
  AlignedBuf tb(kBlockBytes);
  if (tb.p == nullptr) return SetError("out of memory");
  if (!PreadAll(io_.get(), fd, tb.p, kBlockBytes, sb_.segment_bytes - kBlockBytes))
    return SetErrno("cannot read segment " + SegmentPath(index), errno);
  lse::Trailer t;
  if (lse::DecodeTrailer(reinterpret_cast<const uint8_t*>(tb.p), sb_.store_uuid,
                         s.incarnation, sb_.segment_bytes,
                         &t) != lse::DecodeError::kOk)
    return true;
  std::vector<lse::SummaryEntry> entries;
  if (t.summary_length != 0) {
    AlignedBuf sbuf(AlignUp(t.summary_length));
    if (sbuf.p == nullptr) return SetError("out of memory");
    if (!PreadAll(io_.get(), fd, sbuf.p, AlignUp(t.summary_length),
                  t.summary_offset))
      return SetErrno("cannot read segment " + SegmentPath(index), errno);
    const auto* bytes = reinterpret_cast<const uint8_t*>(sbuf.p);
    if (Crc32c(bytes, t.summary_length) != t.summary_crc) return true;  // C10
    if (!lse::DecodeSummary(bytes, t.summary_length, sb_.segment_bytes, &entries))
      return true;
    for (const auto& e : entries)
      if (uint64_t(e.offset_blocks) * kBlockBytes >= t.summary_offset)
        return true;  // a record after the summary: not ours to trust
  } else if (Crc32c(nullptr, 0) != t.summary_crc) {
    return true;
  }
  *records = std::move(entries);
  s.append_off = t.summary_offset;
  s.last_lsn = t.last_write_lsn;
  *loaded = true;
  return true;
}

bool LogStore::WalkSegment(uint32_t index,
                           std::vector<lse::SummaryEntry>* records) {
  Segment& s = segments_[index];
  const int fd = s.dio_fd >= 0 ? s.dio_fd : s.fd;
  const uint64_t end = sb_.segment_bytes - kBlockBytes;  // the trailer block
  AlignedBuf header(kBlockBytes);
  if (header.p == nullptr) return SetError("out of memory");
  // One payload buffer for the walk, grown to the largest value met.
  std::unique_ptr<AlignedBuf> data;
  size_t data_cap = 0;
  uint64_t off = kBlockBytes;
  uint64_t last_end = kBlockBytes;
  bool resync = false;
  while (off + kBlockBytes <= end) {
    if (!PreadAll(io_.get(), fd, header.p, kBlockBytes, off))
      return SetErrno("cannot read segment " + SegmentPath(index), errno);
    lse::RecordHeader h;
    const auto e = lse::DecodeRecordHeader(
        reinterpret_cast<const uint8_t*>(header.p), sb_.store_uuid,
        s.incarnation, sb_.segment_bytes, &h);
    if (e != lse::DecodeError::kOk) {
      off += kBlockBytes;  // resynchronize block by block (design §8.3)
      resync = true;
      continue;
    }
    const uint64_t footprint = lse::RecordFootprint(h);
    if (h.value_len == 0 || off + footprint > end) {
      off += kBlockBytes;
      resync = true;
      continue;
    }
    bool payload_ok = false;
    if (h.flags & lse::kRecordFlagInline) {
      payload_ok = Crc32c(header.p + h.header_len, h.value_len) == h.payload_crc;
    } else {
      if (AlignUp(h.value_len) > data_cap) {
        data_cap = AlignUp(h.value_len);
        data = std::make_unique<AlignedBuf>(data_cap);
        if (data->p == nullptr) return SetError("out of memory");
      }
      if (!PreadAll(io_.get(), fd, data->p, AlignUp(h.value_len),
                    off + kBlockBytes))
        return SetErrno("cannot read segment " + SegmentPath(index), errno);
      payload_ok = Crc32c(data->p, h.value_len) == h.payload_crc;
    }
    if (!payload_ok) {
      ++recovery_.records_rejected;
      if (resync) {  // a resynchronized header must also verify its payload
        off += kBlockBytes;
        continue;
      }
      off += footprint;  // the header's length is trustworthy (its CRC passed)
      continue;
    }
    resync = false;
    lse::SummaryEntry r;
    r.key = h.key;
    r.value_len = h.value_len;
    r.generation = h.generation;
    r.write_lsn = h.write_lsn;
    r.offset_blocks = static_cast<uint32_t>(off / kBlockBytes);
    r.payload_crc = h.payload_crc;
    r.flags = h.flags;
    r.header_len = h.header_len;
    records->push_back(r);
    s.last_lsn = std::max(s.last_lsn, h.write_lsn);
    off += footprint;
    last_end = off;
  }
  s.append_off = last_end;
  return true;
}

bool LogStore::OpenStore() {
  const fs::path root(opt_.dir);
  super_fd_ = io_->Open((root / kSuperName).string(), O_RDWR, 0);
  if (super_fd_ < 0) return SetErrno("cannot open lse_super", errno);
  if (!ReadSuperblock()) return false;
  if (!OpenSegmentFiles(/*create=*/false)) return false;
  journal_fd_ = OpenJournalFile(io_.get(), (root / kJournalName).string(), O_RDWR);
  if (journal_fd_ < 0) return SetErrno("cannot open the journal", errno);
  if (!ReadJournal()) return false;
  if (sb_.reset_in_progress) needs_reset_ = true;

  // Segment headers; then each segment's summary, or its records by a walk.
  AlignedBuf buf(kBlockBytes);
  if (buf.p == nullptr) return SetError("out of memory");
  for (uint32_t i = 0; i < sb_.segment_count; ++i) {
    Segment& s = segments_[i];
    const int fd = s.dio_fd >= 0 ? s.dio_fd : s.fd;
    if (!PreadAll(io_.get(), fd, buf.p, kBlockBytes, 0))
      return SetErrno("cannot read the header of " + SegmentPath(i), errno);
    lse::SegmentHeader h;
    const auto e = lse::DecodeSegmentHeader(
        reinterpret_cast<const uint8_t*>(buf.p), sb_.store_uuid, &h);
    if (e == lse::DecodeError::kForeignStore)
      return SetError(SegmentPath(i) + " belongs to another store");
    s.header_valid = e == lse::DecodeError::kOk && h.segment_index == i;
    s.incarnation = s.header_valid ? h.incarnation : 0;
  }
  std::vector<bool> walked(sb_.segment_count, false);
  if (!needs_reset_) {
    for (uint32_t i = 0; i < sb_.segment_count; ++i) {
      Segment& s = segments_[i];
      if (!s.header_valid) continue;
      std::vector<lse::SummaryEntry> records;
      bool loaded = false;
      if (!LoadSummary(i, &records, &loaded)) return false;
      if (loaded) {
        ++recovery_.summaries_loaded;
      } else {
        records.clear();
        s.last_lsn = 0;
        if (!WalkSegment(i, &records)) return false;
        ++recovery_.segments_walked;
        walked[i] = true;
      }
      s.records = static_cast<uint32_t>(records.size());
      for (const auto& r : records) {
        const Location loc{i, s.incarnation, r.offset_blocks};
        if (kills_.count({i, s.incarnation, r.offset_blocks})) {
          ++recovery_.records_killed;
          continue;
        }
        Entry entry;
        entry.loc = loc;
        entry.value_len = r.value_len;
        entry.generation = r.generation;
        entry.payload_crc = r.payload_crc;
        entry.inline_at =
            (r.flags & lse::kRecordFlagInline) ? r.header_len : uint16_t(0);
        entry.write_lsn = r.write_lsn;
        auto it = index_.find(r.key);
        if (it == index_.end()) {
          index_.emplace(r.key, entry);
          ++s.live;
          continue;
        }
        // Newest wins; at equal generation, the later write (a relocated
        // copy). The loser stays accounted for as a shadow (design §8.3).
        Entry& cur = it->second;
        const bool newer = r.generation > cur.generation ||
                           (r.generation == cur.generation &&
                            r.write_lsn > cur.write_lsn);
        if (newer) {
          AddShadowLocked(r.key, cur.loc, cur.generation);
          --segments_[cur.loc.segment].live;
          cur = entry;
          ++s.live;
        } else {
          AddShadowLocked(r.key, loc, r.generation);
        }
        ++recovery_.records_superseded;
      }
      s.summary = std::move(records);
    }
    for (auto& kv : index_) {
      used_bytes_ += Footprint(kv.second);
      tenant_bytes_[kv.first.tenant_hash] += kv.second.value_len;
    }
    recovery_.records_recovered = index_.size();
  }
  // A segment is free when nothing in it is live or shadowed; it gets a
  // fresh header before its first new record. Anything else holds records
  // of its current incarnation and is never appended to again (§8.3 step 9):
  // sealed already, or sealed at Activate() if it was walked.
  for (uint32_t i = 0; i < sb_.segment_count; ++i) {
    Segment& s = segments_[i];
    if (needs_reset_ || !s.header_valid || (s.live == 0 && s.shadows == 0)) {
      s.state = SegState::kFree;
      free_.push_back(i);
    } else if (walked[i]) {
      s.state = SegState::kWalked;
    } else {
      s.state = SegState::kSealed;
      fifo_.push_back(i);
    }
  }
  std::sort(fifo_.begin(), fifo_.end(), [&](uint32_t a, uint32_t b) {
    return segments_[a].last_lsn < segments_[b].last_lsn;
  });
  next_incarnation_ = sb_.incarnation_reserved;
  next_lsn_ = sb_.lsn_reserved;
  return true;
}

bool LogStore::FreshHeader(uint32_t index, bool empty) {
  if (next_incarnation_ >= sb_.incarnation_reserved) {
    sb_.incarnation_reserved = next_incarnation_ + kIncarnationRange;
    if (!WriteSuperblock()) return false;
  }
  const uint64_t inc = next_incarnation_++;
  const lse::Block b = lse::EncodeSegmentHeader({sb_.store_uuid, index, inc});
  Segment& s = segments_[index];
  if (!PwriteBlock(io_.get(), s.dio_fd >= 0 ? s.dio_fd : s.fd, b.data(),
                   kBlockBytes, 0, /*dsync=*/true))
    return false;
  // The new header is durable: the old incarnation's records are rejected
  // from now on, and the kills naming it are garbage (§4.5).
  if (empty) {
    lse::Trailer t;
    t.store_uuid = sb_.store_uuid;
    t.incarnation = inc;
    t.summary_offset = kBlockBytes;
    const lse::Block tb = lse::EncodeTrailer(t);
    if (!PwriteBlock(io_.get(), s.dio_fd >= 0 ? s.dio_fd : s.fd, tb.data(),
                     kBlockBytes, sb_.segment_bytes - kBlockBytes, true))
      return false;
  }
  s.incarnation = inc;
  s.header_valid = true;
  s.append_off = kBlockBytes;
  s.records = 0;
  s.last_lsn = 0;
  s.summary.clear();
  return true;
}

bool LogStore::ResetToEmpty() {
  sb_.reset_in_progress = 1;
  sb_.incarnation_reserved = next_incarnation_ + sb_.segment_count +
                             kIncarnationRange;
  if (!WriteSuperblock()) return false;
  Hook("reset.after_flag");
  for (uint32_t i = 0; i < sb_.segment_count; ++i)
    if (!FreshHeader(i, /*empty=*/true)) return false;
  Hook("reset.after_headers");
  const fs::path root(opt_.dir);
  const std::string fresh = (root / (std::string(kJournalName) + ".new")).string();
  int fd = io_->Open(fresh, O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) return false;
  const bool synced = io_->Fdatasync(fd) == 0;
  io_->Close(fd);
  if (!synced || io_->Rename(fresh, (root / kJournalName).string()) != 0 ||
      io_->Fsync(dir_fd_) != 0)
    return false;
  Hook("reset.after_rename");
  io_->Close(journal_fd_);
  journal_fd_ = OpenJournalFile(io_.get(), (root / kJournalName).string(), O_RDWR);
  if (journal_fd_ < 0) return false;
  journal_stop_offset_ = 0;
  journal_next_seq_ = 1;
  sb_.reset_in_progress = 0;
  if (!WriteSuperblock()) return false;
  index_.clear();
  shadows_.clear();
  tenant_bytes_.clear();
  used_bytes_ = 0;
  free_.clear();
  fifo_.clear();
  for (uint32_t i = 0; i < sb_.segment_count; ++i) {
    Segment& s = segments_[i];
    s.state = SegState::kFree;
    s.live = s.shadows = 0;
    free_.push_back(i);
  }
  ++stats_.journal_resets;
  DFKV_LOG_WARN("lse: " + opt_.dir + ": store reset to empty");
  return true;
}

bool LogStore::Activate() {
  if (!ok_ || failed_.load()) return false;
  if (activated_.load()) return true;
  Lock lk(mu_);
  uint64_t headers = 0;
  for (const auto& s : segments_)
    if (!s.header_valid) ++headers;
  if (needs_reset_) headers = sb_.segment_count;
  // One durable write first (design §8.3 step 7).
  sb_.clean = 0;
  sb_.incarnation_reserved = next_incarnation_ + headers + kIncarnationRange;
  sb_.lsn_reserved = next_lsn_ + kLsnRange;
  if (!WriteSuperblock()) {
    FailStore("cannot write lse_super", errno);
    return false;
  }
  Hook("activate.after_clean");
  if (needs_reset_) {
    if (!ResetToEmpty()) {
      FailStore("cannot reset the store", errno);
      return false;
    }
  } else if (io_->Ftruncate(journal_fd_,
                            static_cast<off_t>(journal_stop_offset_)) != 0 ||
             io_->Fdatasync(journal_fd_) != 0) {
    FailStore("cannot truncate the journal", errno);
    return false;
  }
  // Seal every walked segment as it is (step 9): never appended to again.
  for (uint32_t i = 0; i < segments_.size(); ++i) {
    if (segments_[i].state != SegState::kWalked) continue;
    segments_[i].state = SegState::kSealing;
    if (!SealSegment(lk, i)) return false;
  }
  std::sort(fifo_.begin(), fifo_.end(), [&](uint32_t a, uint32_t b) {
    return segments_[a].last_lsn < segments_[b].last_lsn;
  });
  Hook("activate.after_seal");
  // Fresh headers where they were invalid (step 10).
  for (uint32_t i = 0; i < segments_.size(); ++i)
    if (!segments_[i].header_valid && !FreshHeader(i, /*empty=*/true)) {
      FailStore("cannot write a segment header", errno);
      return false;
    }
  kills_.clear();
  journal_ = std::make_unique<Journal>(this, journal_fd_, journal_stop_offset_,
                                       journal_next_seq_);
  journal_fd_ = -1;
  uint32_t first = 0;
  if (TakeFreeSegmentLocked(&first)) {
    open_segment_ = first;
    segments_[first].state = SegState::kOpen;
  }
  if (failed_.load()) return false;
  sealer_ = std::thread([this] { SealerLoop(); });
  activated_.store(true);
  return true;
}

bool LogStore::Close() {
  if (closed_.exchange(true)) return true;
  if (!activated_.load()) return ok_;
  {
    Lock lk(mu_);
    stop_threads_ = true;
    state_cv_.notify_all();
  }
  if (sealer_.joinable()) sealer_.join();
  Lock lk(mu_);
  // Operations already admitted finish; none is admitted any more.
  state_cv_.wait(lk, [&] {
    if (!flights_.empty()) return false;
    for (const auto& s : segments_)
      if (s.writers != 0) return false;
    return true;
  });
  // Seal what is still open or queued (§8.5).
  std::vector<uint32_t> to_seal(seal_queue_.begin(), seal_queue_.end());
  seal_queue_.clear();
  if (open_segment_ != UINT32_MAX) to_seal.push_back(open_segment_);
  open_segment_ = UINT32_MAX;
  bool ok = Healthy();
  for (uint32_t i : to_seal) {
    segments_[i].state = SegState::kSealing;
    if (ok && !SealSegment(lk, i)) ok = false;
  }
  lk.unlock();
  Hook("close.after_seal");
  if (journal_) journal_->Stop();
  lk.lock();
  ok = ok && Healthy() && !(journal_ && journal_->Failed());
  // Only the segments this run wrote can hold unsynced records.
  for (auto& s : segments_)
    if (s.dirty && s.fd >= 0 && io_->Fdatasync(s.fd) != 0) ok = false;
  if (ok) {
    sb_.clean = 1;
    ok = WriteSuperblock();
  }
  return ok;
}

bool LogStore::Healthy() const { return ok_ && !failed_.load(); }

bool LogStore::Serving() const {
  return Healthy() && activated_.load() && !closed_.load();
}

void LogStore::FailStore(const char* what, int err) {
  if (!failed_.exchange(true))
    DFKV_LOG_ERROR("lse: " + opt_.dir + ": " + what + ": " +
                   std::strerror(err) + "; the store stops serving");
  state_cv_.notify_all();
}

uint64_t LogStore::MaxValueBytes() const {
  return sb_.segment_bytes > 4 * kBlockBytes
             ? sb_.segment_bytes - 4 * kBlockBytes
             : 0;
}

// ---- bookkeeping -------------------------------------------------------------
// `kill_ticket` != 0: the location's KILL is already enqueued at that ticket.
void LogStore::AddShadowLocked(const BlockKey& key, const Location& loc,
                               uint64_t generation, uint64_t kill_ticket) {
  shadows_.emplace(key, Shadow{loc, generation, kill_ticket != 0, kill_ticket});
  ++segments_[loc.segment].shadows;
}

void LogStore::EraseEntryLocked(const BlockKey& key) {
  auto it = index_.find(key);
  if (it == index_.end()) return;
  const Entry e = it->second;
  index_.erase(it);
  used_bytes_ -= Footprint(e);
  auto t = tenant_bytes_.find(key.tenant_hash);
  if (t != tenant_bytes_.end()) t->second -= e.value_len;
  --segments_[e.loc.segment].live;
  MaybeFreeLocked(e.loc.segment);
}

void LogStore::MaybeFreeLocked(uint32_t index) {
  if (index >= segments_.size()) return;
  Segment& s = segments_[index];
  if (s.state != SegState::kSealed || s.live != 0 ||
      s.shadows != 0 || s.pins != 0 || s.writers != 0)
    return;
  // Every record in it is killed: it can be reused as it is.
  RemoveFromFifoLocked(index);
  s.state = SegState::kFree;
  free_.push_back(index);
  state_cv_.notify_all();
}

void LogStore::RemoveFromFifoLocked(uint32_t index) {
  auto it = std::find(fifo_.begin(), fifo_.end(), index);
  if (it != fifo_.end()) fifo_.erase(it);
}

uint64_t LogStore::EnqueueKillLocked(const BlockKey& key, const Location& loc,
                                     uint64_t generation) {
  lse::JournalEntry e;
  e.type = lse::kJournalKill;
  e.segment_index = loc.segment;
  e.incarnation = loc.incarnation;
  e.offset_blocks = loc.offset_blocks;
  e.key = key;
  e.generation = generation;
  return journal_->Enqueue(e);
}

// ---- write path --------------------------------------------------------------
uint64_t LogStore::NextLsnLocked() {
  const uint64_t lsn = next_lsn_++;
  if (next_lsn_ >= sb_.lsn_reserved) {
    sb_.lsn_reserved = next_lsn_ + kLsnRange;
    if (!WriteSuperblock()) FailStore("cannot write lse_super", errno);
  }
  return lsn;
}

bool LogStore::TakeFreeSegmentLocked(uint32_t* out) {
  if (free_.empty()) return false;
  const uint32_t next = free_.front();
  free_.erase(free_.begin());
  if (!FreshHeader(next)) {
    FailStore("cannot write a segment header", errno);
    return false;
  }
  *out = next;
  return true;
}

void LogStore::HandToSealerLocked(Lock& lk, uint32_t segment) {
  segments_[segment].state = SegState::kSealing;
  // The queue is bounded (§6.3), which bounds what an unclean exit leaves
  // unsealed.
  state_cv_.wait(lk, [&] {
    return seal_queue_.size() < kSealerQueue || stop_threads_ || failed_.load();
  });
  seal_queue_.push_back(segment);
  state_cv_.notify_all();
}

bool LogStore::PlaceLocked(uint32_t segment, uint64_t footprint,
                           uint64_t* offset) {
  Segment& s = segments_[segment];
  if (s.append_off + footprint + SummaryRoom(s.records + 1) + kBlockBytes >
      sb_.segment_bytes)
    return false;
  *offset = s.append_off;
  s.append_off += footprint;
  ++s.records;
  ++s.pins;
  ++s.writers;
  s.dirty = true;
  return true;
}

bool LogStore::ReserveLocked(Lock& lk, uint64_t footprint, uint32_t* segment,
                             uint64_t* offset) {
  for (int attempt = 0; attempt < 64 && !failed_.load(); ++attempt) {
    if (open_segment_ != UINT32_MAX) {
      if (PlaceLocked(open_segment_, footprint, offset)) {
        *segment = open_segment_;
        return true;
      }
      const uint32_t full = open_segment_;
      open_segment_ = UINT32_MAX;
      HandToSealerLocked(lk, full);
      continue;
    }
    uint32_t next = 0;
    if (!TakeFreeSegmentLocked(&next)) return false;  // full (no eviction yet)
    open_segment_ = next;
    segments_[next].state = SegState::kOpen;
  }
  return false;
}

bool LogStore::WriteRecord(uint32_t segment, uint64_t offset,
                           const lse::RecordHeader& h, const char* payload,
                           size_t len, char* direct_buf, size_t direct_cap) {
  const bool is_inline = (h.flags & lse::kRecordFlagInline) != 0;
  lse::Block header;
  if (!lse::EncodeRecordHeader(h, is_inline ? payload : nullptr, &header)) {
    errno = EINVAL;
    return false;
  }
  AlignedBuf hbuf(kBlockBytes);
  if (hbuf.p == nullptr) return false;
  std::memcpy(hbuf.p, header.data(), kBlockBytes);
  struct iovec iov[2];
  iov[0] = {hbuf.p, kBlockBytes};
  int n = 1;
  std::unique_ptr<AlignedBuf> bounce;
  Segment& s = segments_[segment];
  const bool direct = s.dio_fd >= 0;
  if (!is_inline) {
    const size_t padded = AlignUp(len);
    if (!direct) {
      iov[1] = {const_cast<char*>(payload), len};
    } else if (direct_buf != nullptr &&
               (reinterpret_cast<uintptr_t>(direct_buf) & (kBlockBytes - 1)) == 0 &&
               direct_cap >= padded) {
      std::memset(direct_buf + len, 0, padded - len);
      iov[1] = {direct_buf, padded};
    } else {
      bounce = std::make_unique<AlignedBuf>(padded);
      if (bounce->p == nullptr) return false;
      std::memcpy(bounce->p, payload, len);
      std::memset(bounce->p + len, 0, padded - len);
      iov[1] = {bounce->p, padded};
    }
    n = 2;
  }
  return PwriteAll(io_.get(), direct ? s.dio_fd : s.fd, iov, n, offset,
                   /*dsync=*/false);
}

void LogStore::RecordWrittenLocked(uint32_t segment, const BlockKey& key,
                                   const lse::RecordHeader& h, uint64_t offset) {
  Segment& s = segments_[segment];
  lse::SummaryEntry e;
  e.key = key;
  e.value_len = h.value_len;
  e.generation = h.generation;
  e.write_lsn = h.write_lsn;
  e.offset_blocks = static_cast<uint32_t>(offset / kBlockBytes);
  e.payload_crc = h.payload_crc;
  e.flags = h.flags;
  e.header_len = h.header_len;
  s.summary.push_back(e);
  s.last_lsn = std::max(s.last_lsn, h.write_lsn);
}

template <typename WriteFn>
Status LogStore::Put(const BlockKey& key, size_t len, const WriteFn& write) {
  if (!Serving()) return Status::kIOError;
  if (len == 0 || len > MaxValueBytes()) return Status::kInvalid;
  Lock lk(mu_);
  for (;;) {
    if (index_.count(key) != 0) return Status::kOk;  // keep-first
    auto f = flights_.find(key);
    if (f == flights_.end()) break;
    std::shared_ptr<Flight> flight = f->second;
    if (!flight->cancelled) {  // join the PUT in flight
      flight->cv.wait(lk, [&] { return flight->done; });
      return flight->result;
    }
    // A REMOVE took the key while that PUT was written: this PUT, which
    // arrived after the REMOVE, must not share its result. Wait, then judge
    // again.
    flight->cv.wait(lk, [&] { return flight->done; });
  }
  lse::RecordHeader h;
  h.flags = lse::RecordFitsInline(len) ? lse::kRecordFlagInline : 0;
  h.value_len = len;
  const uint64_t footprint = lse::RecordFootprint(h);
  // The flight is registered before the reservation, which may wait (for the
  // sealer, or an inline reclaim): joiners and REMOVEs see it meanwhile.
  auto flight = std::make_shared<Flight>();
  flights_[key] = flight;
  auto finish = [&](Status result) {
    flight->result = result;
    flight->done = true;
    auto it = flights_.find(key);
    if (it != flights_.end() && it->second == flight) flights_.erase(it);
    flight->cv.notify_all();
    state_cv_.notify_all();
    return result;
  };
  uint32_t segment = 0;
  uint64_t offset = 0;
  if (!ReserveLocked(lk, footprint, &segment, &offset)) {
    if (!Serving()) return finish(Status::kIOError);
    ++stats_.cache_full;
    return finish(Status::kCacheFull);
  }
  if (opt_.on_reserve) opt_.on_reserve(key);
  Segment& seg = segments_[segment];
  if (flight->cancelled) {
    // A REMOVE came while this PUT waited for space: nothing is written, so
    // nothing can come back after the REMOVE; answered kOk (it began first).
    --seg.pins;
    --seg.writers;
    return finish(Status::kOk);
  }
  const uint64_t generation = NextLsnLocked();
  flight->loc = {segment, seg.incarnation,
                 static_cast<uint32_t>(offset / kBlockBytes)};
  flight->has_loc = true;
  flight->generation = generation;
  h.store_uuid = sb_.store_uuid;
  h.incarnation = flight->loc.incarnation;
  h.key = key;
  h.generation = generation;
  h.write_lsn = generation;
  lk.unlock();

  Hook("put.after_reserve");
  uint32_t crc = 0;
  const bool written = write(&h, segment, offset, &crc);
  const int err = errno;
  Hook("put.after_write");

  lk.lock();
  --segments_[segment].pins;
  --segments_[segment].writers;
  // Marked for the periodic sync only now that the write is complete: a
  // sync pass between the reservation and the write must not clear it.
  if (written) segments_[segment].unsynced = true;
  state_cv_.notify_all();
  Status result = Status::kOk;
  if (!written) {
    FailStore("cannot write a record", err);
    result = Status::kIOError;
  } else {
    RecordWrittenLocked(segment, key, h, offset);
  }
  if (written && !flight->cancelled) {
    // Older values of the key that a crash could bring back are killed
    // durably before this one is acknowledged (design §6.1, I3).
    auto range = shadows_.equal_range(key);
    if (range.first != range.second) {
      uint64_t ticket = 0;
      for (auto it = range.first; it != range.second; ++it) {
        if (!it->second.kill_pending) {
          it->second.ticket =
              EnqueueKillLocked(key, it->second.loc, it->second.generation);
          it->second.kill_pending = true;
        }
        ticket = std::max(ticket, it->second.ticket);
      }
      lk.unlock();
      Hook("put.after_shadow_enqueue");
      const bool durable = journal_->WaitDurable(ticket);
      lk.lock();
      if (!durable) {
        FailStore("cannot write the journal", EIO);
        result = Status::kIOError;
      } else {
        auto again = shadows_.equal_range(key);
        std::vector<uint32_t> segs;
        for (auto it = again.first; it != again.second;) {
          if (it->second.kill_pending && it->second.ticket <= ticket) {
            segs.push_back(it->second.loc.segment);
            --segments_[it->second.loc.segment].shadows;
            it = shadows_.erase(it);
          } else {
            ++it;
          }
        }
        for (uint32_t s : segs) MaybeFreeLocked(s);
      }
    }
    if (result == Status::kOk && !flight->cancelled) {
      Entry e;
      e.loc = flight->loc;
      e.value_len = len;
      e.generation = generation;
      e.payload_crc = crc;
      e.inline_at = (h.flags & lse::kRecordFlagInline) ? h.header_len : uint16_t(0);
      e.write_lsn = generation;
      index_[key] = e;
      ++segments_[segment].live;
      used_bytes_ += footprint;
      tenant_bytes_[key.tenant_hash] += len;
      if (opt_.on_commit) opt_.on_commit(key, generation);
    }
  }
  // A PUT that a REMOVE cancelled while it was written is answered kOk (it
  // started before the REMOVE) and not published.
  MaybeFreeLocked(segment);
  return finish(result);
}

Status LogStore::Cache(const BlockKey& key, const void* data, size_t len) {
  if (data == nullptr) return Status::kInvalid;
  return Put(key, len, [&](lse::RecordHeader* h, uint32_t segment,
                           uint64_t offset, uint32_t* crc) {
    *crc = h->payload_crc = Crc32c(data, len);
    return WriteRecord(segment, offset, *h, static_cast<const char*>(data),
                       len, nullptr, 0);
  });
}

Status LogStore::CacheDirect(const BlockKey& key, char* data, size_t len,
                             size_t cap) {
  if (data == nullptr) return Status::kInvalid;
  return Put(key, len, [&](lse::RecordHeader* h, uint32_t segment,
                           uint64_t offset, uint32_t* crc) {
    *crc = h->payload_crc = Crc32c(data, len);
    return WriteRecord(segment, offset, *h, data, len, data, cap);
  });
}

// ---- sealing (§6.3) ----------------------------------------------------------
// With `lk` held; released for the I/O. The segment is in kSealing.
bool LogStore::SealSegment(Lock& lk, uint32_t index) {
  Segment& s = segments_[index];
  state_cv_.wait(lk, [&] { return s.writers == 0 || failed_.load(); });
  if (failed_.load()) return false;
  std::vector<lse::SummaryEntry> entries = s.summary;
  std::sort(entries.begin(), entries.end(),
            [](const lse::SummaryEntry& a, const lse::SummaryEntry& b) {
              return a.offset_blocks < b.offset_blocks;
            });
  const uint64_t summary_off = s.append_off;
  const uint64_t summary_len = entries.size() * lse::kSummaryEntryBytes;
  const int fd = s.dio_fd >= 0 ? s.dio_fd : s.fd;
  const int sync_fd = s.fd;
  lse::Trailer t;
  t.store_uuid = sb_.store_uuid;
  t.incarnation = s.incarnation;
  t.record_count = static_cast<uint32_t>(entries.size());
  t.summary_offset = summary_off;
  t.summary_length = summary_len;
  t.last_write_lsn = s.last_lsn;
  // A walked segment's summary always fits (its room was reserved with its
  // records); if it did not, it stays without a trailer and is walked again.
  const bool fits =
      summary_off + SummaryRoom(entries.size()) + kBlockBytes <= sb_.segment_bytes;
  lk.unlock();
  bool ok = io_->Fdatasync(sync_fd) == 0;  // step 1: every record durable
  if (ok && fits) {
    const std::vector<uint8_t> bytes = lse::EncodeSummary(entries);
    t.summary_crc = Crc32c(bytes.data(), bytes.size());
    if (!bytes.empty())
      ok = PwriteBlock(io_.get(), fd, bytes.data(), bytes.size(), summary_off,
                       false) &&
           io_->Fdatasync(sync_fd) == 0;  // step 2
    Hook("seal.after_summary");
    const lse::Block tb = lse::EncodeTrailer(t);
    ok = ok && PwriteBlock(io_.get(), fd, tb.data(), kBlockBytes,
                           sb_.segment_bytes - kBlockBytes, true);  // step 3
    Hook("seal.after_trailer");
  }
  const int err = errno;
  lk.lock();
  if (!ok) {
    FailStore("cannot seal a segment", err);
    return false;
  }
  s.state = SegState::kSealed;
  fifo_.push_back(index);
  ++stats_.segments_sealed;
  MaybeFreeLocked(index);
  state_cv_.notify_all();
  return true;
}

void LogStore::SealerLoop() {
  Lock lk(mu_);
  auto next_sync = std::chrono::steady_clock::now() +
                   std::chrono::milliseconds(opt_.sync_ms);
  while (!failed_.load()) {
    if (!seal_queue_.empty()) {
      const uint32_t index = seal_queue_.front();
      if (!SealSegment(lk, index)) break;
      seal_queue_.pop_front();  // only now: the queue bounds the unsealed
      state_cv_.notify_all();
      continue;
    }
    if (stop_threads_) break;
    if (std::chrono::steady_clock::now() >= next_sync) {
      // Bounds what a power loss can take of acknowledged writes (§6.5):
      // every segment written since the previous pass, the one just left
      // behind as well. A failed fdatasync fails the store (C19): Linux
      // reports the error once, so a later sync could succeed with the
      // writes lost.
      std::vector<int> fds;
      for (auto& seg : segments_)
        if (seg.unsynced && seg.fd >= 0) {
          seg.unsynced = false;
          fds.push_back(seg.fd);
        }
      lk.unlock();
      int err = 0;
      for (int fd : fds)
        if (io_->Fdatasync(fd) != 0) err = errno ? errno : EIO;
      lk.lock();
      if (err != 0) {
        FailStore("cannot sync a segment", err);
        break;
      }
      next_sync = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(opt_.sync_ms);
      continue;
    }
    state_cv_.wait_until(lk, next_sync);
  }
}

// ---- REMOVE ------------------------------------------------------------------
Status LogStore::Remove(const BlockKey& key) {
  if (!Serving()) return Status::kIOError;
  Lock lk(mu_);
  if (opt_.on_remove) opt_.on_remove(key);
  bool had = false;
  // Until its KILL is durable, a killed record could come back after a power
  // loss: it stays a kill-pending shadow, which a PUT of the key waits for
  // before it is acknowledged (I3), and which keeps its segment in use.
  auto it = index_.find(key);
  if (it != index_.end()) {
    const Entry e = it->second;
    AddShadowLocked(key, e.loc, e.generation,
                    EnqueueKillLocked(key, e.loc, e.generation));
    EraseEntryLocked(key);
    had = true;
  }
  auto f = flights_.find(key);
  if (f != flights_.end() && !f->second->cancelled) {
    f->second->cancelled = true;
    if (f->second->has_loc) {
      const Location loc = f->second->loc;
      const uint64_t generation = f->second->generation;
      AddShadowLocked(key, loc, generation, EnqueueKillLocked(key, loc, generation));
    }
    had = true;
  }
  auto range = shadows_.equal_range(key);
  for (auto s = range.first; s != range.second; ++s) {
    if (!s->second.kill_pending) {
      s->second.ticket = EnqueueKillLocked(key, s->second.loc, s->second.generation);
      s->second.kill_pending = true;
    }
  }
  // Every kill enqueued before this point, by this REMOVE or anyone else,
  // is durable before it returns (design §4.5, I2).
  const uint64_t ticket = journal_->LastTicket();
  lk.unlock();
  Hook("remove.after_enqueue");
  if (!journal_->WaitDurable(ticket)) {
    FailStore("cannot write the journal", EIO);
    return Status::kIOError;
  }
  Hook("remove.after_wait");
  lk.lock();
  auto again = shadows_.equal_range(key);
  std::vector<uint32_t> segs;
  for (auto s = again.first; s != again.second;) {
    if (s->second.kill_pending && s->second.ticket <= ticket) {
      segs.push_back(s->second.loc.segment);
      --segments_[s->second.loc.segment].shadows;
      s = shadows_.erase(s);
    } else {
      ++s;
    }
  }
  for (uint32_t seg : segs) MaybeFreeLocked(seg);
  lk.unlock();
  Hook("remove.after_durable");
  return had ? Status::kOk : Status::kNotFound;
}

// ---- stats -------------------------------------------------------
LogStore::Stats LogStore::GetStats() const {
  std::lock_guard<std::mutex> lk(mu_);
  Stats st = stats_;
  st.segments_free = static_cast<uint32_t>(free_.size());
  st.segments_sealed_now = static_cast<uint32_t>(fifo_.size());
  st.sealer_queue = static_cast<uint32_t>(seal_queue_.size());
  st.shadows = shadows_.size();
  st.segment_count = static_cast<uint32_t>(segments_.size());
  return st;
}

// ---- read path ---------------------------------------------------------------
bool LogStore::AcquireLocked(const BlockKey& key, Entry* out) {
  auto it = index_.find(key);
  if (it == index_.end()) return false;
  *out = it->second;
  ++segments_[out->loc.segment].pins;
  return true;
}

void LogStore::Unpin(uint32_t segment) {
  std::lock_guard<std::mutex> lk(mu_);
  Segment& s = segments_[segment];
  if (s.pins > 0) --s.pins;
  MaybeFreeLocked(segment);
  state_cv_.notify_all();
}

uint64_t LogStore::Footprint(const Entry& e) {
  return e.inline_at != 0 ? kBlockBytes : kBlockBytes + AlignUp(e.value_len);
}

uint64_t LogStore::PayloadOffset(const Entry& e) const {
  const uint64_t record = uint64_t(e.loc.offset_blocks) * kBlockBytes;
  return e.inline_at != 0 ? record + e.inline_at : record + kBlockBytes;
}

bool LogStore::ReadPayload(const Entry& e, uint64_t offset, size_t n,
                           char* dst) {
  const Segment& s = segments_[e.loc.segment];
  const uint64_t pos = PayloadOffset(e) + offset;
  if (s.dio_fd < 0) return PreadAll(io_.get(), s.fd, dst, n, pos);
  const uint64_t start = AlignDown(pos);
  const uint64_t len = AlignUp(pos + n) - start;
  AlignedBuf buf(len);
  if (buf.p == nullptr) return false;
  if (!PreadAll(io_.get(), s.dio_fd, buf.p, len, start)) return false;
  std::memcpy(dst, buf.p + (pos - start), n);
  return true;
}

void LogStore::DropCorruptLocked(const BlockKey& key, const Entry& e) {
  auto it = index_.find(key);
  if (it == index_.end() || it->second.generation != e.generation) return;
  ++stats_.read_crc_failures;
  DFKV_LOG_WARN("lse: " + opt_.dir + ": payload checksum mismatch in segment " +
                std::to_string(e.loc.segment) + " at block " +
                std::to_string(e.loc.offset_blocks) + "; served as a miss");
  // The record on disk may be intact (the damage may have been in memory or
  // in transfer): it stays accounted for as a shadow (design §5.2). The
  // shadow is added first, so the segment does not become free meanwhile.
  AddShadowLocked(key, e.loc, e.generation);
  EraseEntryLocked(key);
}

Status LogStore::Range(const BlockKey& key, uint64_t offset, uint64_t length,
                       std::string* out, size_t* value_len) {
  if (!Serving()) return Status::kIOError;
  Entry e;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (!AcquireLocked(key, &e)) return Status::kNotFound;
  }
  struct Unpinner {
    LogStore* store;
    uint32_t segment;
    ~Unpinner() { store->Unpin(segment); }
  } unpin{this, e.loc.segment};
  if (offset > e.value_len) return Status::kInvalid;
  if (value_len) *value_len = static_cast<size_t>(e.value_len);
  const uint64_t remainder = e.value_len - offset;
  const uint64_t n = std::min(length == 0 ? remainder : length, remainder);
  if (n == e.value_len) {  // the whole value: verified
    std::string all(e.value_len, '\0');
    if (!ReadPayload(e, 0, all.size(), &all[0])) return Status::kIOError;
    if (Crc32c(all.data(), all.size()) != e.payload_crc) {
      std::lock_guard<std::mutex> lk(mu_);
      DropCorruptLocked(key, e);
      return Status::kNotFound;
    }
    *out = std::move(all);
    return Status::kOk;
  }
  out->resize(n);
  if (n != 0 && !ReadPayload(e, offset, n, &(*out)[0])) return Status::kIOError;
  return Status::kOk;
}

Status LogStore::RangeInto(const BlockKey& key, uint64_t offset,
                           uint64_t length, char* dst, size_t dst_cap,
                           size_t* out_len, size_t* value_len) {
  *out_len = 0;
  if (!Serving()) return Status::kIOError;
  std::string bytes;
  size_t total = 0;
  Status st = Range(key, offset, length, &bytes, &total);
  if (st != Status::kOk) return st;
  if (value_len) *value_len = total;
  const size_t n = std::min(bytes.size(), dst_cap);
  if (n) std::memcpy(dst, bytes.data(), n);
  *out_len = n;
  return Status::kOk;
}

Status LogStore::RangeDirect(const BlockKey& key, uint64_t offset,
                             uint64_t length, char* io_buf, size_t io_cap,
                             const char** out_data, size_t* out_len,
                             size_t* value_len) {
  *out_data = nullptr;
  *out_len = 0;
  if (!Serving()) return Status::kIOError;
  if (io_buf == nullptr ||
      (reinterpret_cast<uintptr_t>(io_buf) & (kBlockBytes - 1)) != 0)
    return Status::kIOError;
  Entry e;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (!AcquireLocked(key, &e)) return Status::kNotFound;
  }
  struct Unpinner {
    LogStore* store;
    uint32_t segment;
    ~Unpinner() { store->Unpin(segment); }
  } unpin{this, e.loc.segment};
  if (offset > e.value_len) return Status::kInvalid;
  if (value_len) *value_len = static_cast<size_t>(e.value_len);
  const uint64_t remainder = e.value_len - offset;
  const uint64_t n = std::min(length == 0 ? remainder : length, remainder);
  if (n == 0) {
    *out_data = io_buf;
    return Status::kOk;
  }
  const uint64_t pos = PayloadOffset(e) + offset;
  const uint64_t start = AlignDown(pos);
  const uint64_t alen = AlignUp(pos + n) - start;
  if (alen > io_cap) return Status::kIOError;
  const Segment& s = segments_[e.loc.segment];
  if (!PreadAll(io_.get(), s.dio_fd >= 0 ? s.dio_fd : s.fd, io_buf, alen, start))
    return Status::kIOError;
  const char* data = io_buf + (pos - start);
  if (n == e.value_len && Crc32c(data, n) != e.payload_crc) {
    std::lock_guard<std::mutex> lk(mu_);
    DropCorruptLocked(key, e);
    return Status::kNotFound;
  }
  *out_data = data;
  *out_len = static_cast<size_t>(n);
  return Status::kOk;
}

void LogStore::ReleaseLeaseThunk(void* owner, uint64_t token) noexcept {
  static_cast<LogStore*>(owner)->Unpin(static_cast<uint32_t>(token - 1));
}

Status LogStore::RangeDirectPrep(const BlockKey& key, uint64_t offset,
                                 uint64_t length, size_t io_cap,
                                 ReadLease* out) {
  if (out == nullptr) return Status::kInvalid;
  *out = ReadLease{};
  if (!Serving()) return Status::kIOError;
  Entry e;
  int fd = -1;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (!AcquireLocked(key, &e)) return Status::kNotFound;
    const Segment& s = segments_[e.loc.segment];
    fd = ::dup(s.dio_fd >= 0 ? s.dio_fd : s.fd);
  }
  // The lease owns the descriptor and the segment pin from here on.
  ReadLease lease = ReadLease::Adopt(fd, this, uint64_t(e.loc.segment) + 1,
                                     &LogStore::ReleaseLeaseThunk);
  if (fd < 0) return Status::kIOError;
  if (offset > e.value_len) return Status::kInvalid;
  const uint64_t remainder = e.value_len - offset;
  const uint64_t n = std::min(length == 0 ? remainder : length, remainder);
  if (n == 0) {
    lease = ReadLease{};
    lease.value_len = static_cast<size_t>(e.value_len);
    *out = std::move(lease);
    return Status::kOk;
  }
  const uint64_t pos = PayloadOffset(e) + offset;
  const uint64_t start = AlignDown(pos);
  const uint64_t alen = AlignUp(pos + n) - start;
  if (alen > io_cap) return Status::kIOError;
  lease.aligned_off = start;
  lease.aligned_len = static_cast<size_t>(alen);
  lease.head = static_cast<size_t>(pos - start);
  lease.payload_len = static_cast<size_t>(n);
  lease.value_len = static_cast<size_t>(e.value_len);
  *out = std::move(lease);
  return Status::kOk;
}

bool LogStore::IsCached(const BlockKey& key) const {
  if (!Serving()) return false;
  std::lock_guard<std::mutex> lk(mu_);
  return index_.count(key) != 0;
}

Status LogStore::Lookup(const BlockKey& key, ValueMetadata* out) const {
  if (!Serving()) return Status::kIOError;
  if (out == nullptr) return Status::kInvalid;
  std::lock_guard<std::mutex> lk(mu_);
  auto it = index_.find(key);
  if (it == index_.end()) return Status::kNotFound;
  out->value_len = it->second.value_len;
  out->generation = it->second.generation;
  return Status::kOk;
}

uint64_t LogStore::UsedBytes() const {
  std::lock_guard<std::mutex> lk(mu_);
  return used_bytes_;
}

uint64_t LogStore::TenantUsedBytes(uint64_t tenant_hash) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = tenant_bytes_.find(tenant_hash);
  return it == tenant_bytes_.end() ? 0 : it->second;
}

size_t LogStore::Count() const {
  std::lock_guard<std::mutex> lk(mu_);
  return index_.size();
}

}  // namespace dfkv
