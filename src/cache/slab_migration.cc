#include "cache/slab_migration.h"

#include <dirent.h>
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>
#include <set>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "cache/log_store.h"
#include "cache/lse_format.h"
#include "cache/slab_allocator.h"
#include "common/block_key_hash.h"
#include "utils/crc32.h"
#include "utils/crc32c.h"
#include "utils/net_util.h"
#include "utils/sha256.h"

namespace dfkv {
namespace slab_migration {
namespace {

namespace fs = std::filesystem;
constexpr uint32_t kRecMagic = 0x334C5453u;    // STL3
constexpr uint32_t kMetaMagic = 0x424C534Du;   // SLBM
constexpr uint32_t kStateMagic = 0x53424C53u;  // SLBS
constexpr uint32_t kFormatVersion = 3;
constexpr uint32_t kStateClean = 1;
constexpr size_t kRecordBytes = 64;
constexpr size_t kMetadataBytes = 64;
constexpr uint8_t kRecordPayloadCrc = 1;
constexpr uint64_t kMinSegmentBytes = 4ull << 20;
constexpr uint64_t kMaxSegmentBytes = 1ull << 30;

bool Fail(std::string* error, const std::string& message) {
  if (error != nullptr) *error = message;
  return false;
}

std::string ErrnoText(const std::string& operation, int error) {
  return operation + ": " + std::error_code(error, std::generic_category()).message() +
         " (errno " + std::to_string(error) + ")";
}

uint64_t Ns(const timespec& ts) {
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
         static_cast<uint64_t>(ts.tv_nsec);
}

bool PreadAll(int fd, void* data, size_t size, uint64_t offset) {
  char* out = static_cast<char*>(data);
  size_t done = 0;
  while (done < size) {
    const ssize_t n = ::pread(fd, out + done, size - done,
                              static_cast<off_t>(offset + done));
    if (n > 0) {
      done += static_cast<size_t>(n);
    } else if (n < 0 && errno == EINTR) {
      continue;
    } else {
      return false;
    }
  }
  return true;
}

class Sha256Stream {
 public:
  Sha256Stream() = default;
  void Update(const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    bit_bytes_ += size;
    while (size != 0) {
      const size_t take = std::min(size, 64 - tail_size_);
      std::memcpy(tail_ + tail_size_, bytes, take);
      tail_size_ += take;
      bytes += take;
      size -= take;
      if (tail_size_ == 64) {
        sha256_detail::Compress(state_, tail_);
        tail_size_ = 0;
      }
    }
  }
  std::array<uint8_t, 32> Finish() const {
    uint32_t state[8];
    std::memcpy(state, state_, sizeof(state));
    uint8_t final_blocks[128]{};
    std::memcpy(final_blocks, tail_, tail_size_);
    final_blocks[tail_size_] = 0x80;
    const size_t padded = tail_size_ < 56 ? 64 : 128;
    const uint64_t bit_size = bit_bytes_ * 8;
    for (size_t i = 0; i < 8; ++i)
      final_blocks[padded - 1 - i] = static_cast<uint8_t>(bit_size >> (8 * i));
    sha256_detail::Compress(state, final_blocks);
    if (padded == 128) sha256_detail::Compress(state, final_blocks + 64);
    std::array<uint8_t, 32> out{};
    for (size_t i = 0; i < 8; ++i) sha256_detail::StoreBe32(out.data() + i * 4, state[i]);
    return out;
  }

 private:
  uint32_t state_[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                        0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                        0x1f83d9abu, 0x5be0cd19u};
  uint8_t tail_[64]{};
  size_t tail_size_ = 0;
  uint64_t bit_bytes_ = 0;
};

std::string Hex(const uint8_t* bytes, size_t size) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string out(size * 2, '0');
  for (size_t i = 0; i < size; ++i) {
    out[i * 2] = kDigits[bytes[i] >> 4];
    out[i * 2 + 1] = kDigits[bytes[i] & 15];
  }
  return out;
}

std::string Sha256(const void* data, size_t size) {
  std::array<uint8_t, 32> digest{};
  dfkv::Sha256(data, size, digest.data());
  return Hex(digest.data(), digest.size());
}

class ScopedFd {
 public:
  ScopedFd() = default;
  explicit ScopedFd(int fd) : fd_(fd) {}
  ~ScopedFd() { Reset(); }
  ScopedFd(const ScopedFd&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;
  ScopedFd(ScopedFd&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  ScopedFd& operator=(ScopedFd&& other) noexcept {
    if (this != &other) {
      Reset();
      fd_ = std::exchange(other.fd_, -1);
    }
    return *this;
  }
  int get() const { return fd_; }
  int Release() { return std::exchange(fd_, -1); }
  void Reset(int fd = -1) {
    if (fd_ >= 0) ::close(fd_);
    fd_ = fd;
  }

 private:
  int fd_ = -1;
};

struct HeldFile {
  std::string path;
  ScopedFd fd;
  FileManifest initial;
};

bool CanonicalExistingPath(const fs::path& path, std::string* error) {
  if (!path.is_absolute() || path != path.lexically_normal())
    return Fail(error, "path must be absolute and lexically normalized: " + path.string());
  std::error_code ec;
  const fs::path canonical = fs::canonical(path, ec);
  if (ec) return Fail(error, "cannot canonicalize " + path.string() + ": " + ec.message());
  if (canonical != path)
    return Fail(error, "path contains a symlink or non-canonical alias: " + path.string());
  return true;
}

bool PathAbsent(const fs::path& path, std::string* error) {
  std::error_code ec;
  const auto status = fs::symlink_status(path, ec);
  if (ec == std::errc::no_such_file_or_directory ||
      status.type() == fs::file_type::not_found)
    return true;
  if (ec) return Fail(error, "cannot inspect " + path.string() + ": " + ec.message());
  return Fail(error, "path already exists; refusing to reuse or replace it: " + path.string());
}

bool ListNames(int directory_fd, std::vector<std::string>* names, std::string* error) {
  ScopedFd dup(::dup(directory_fd));
  if (dup.get() < 0) return Fail(error, ErrnoText("dup source directory", errno));
  // dup shares the directory description and its enumeration offset. Rewind
  // the duplicate so repeated invariance scans start at directory entry zero.
  if (::lseek(dup.get(), 0, SEEK_SET) != 0)
    return Fail(error, ErrnoText("rewind source directory", errno));
  DIR* raw = ::fdopendir(dup.get());
  if (raw == nullptr) return Fail(error, ErrnoText("fdopendir source directory", errno));
  (void)dup.Release();  // directory stream now owns it; closedir closes it.
  names->clear();
  errno = 0;
  while (dirent* entry = ::readdir(raw)) {
    const std::string_view name(entry->d_name);
    if (name != "." && name != "..") names->emplace_back(name);
    errno = 0;
  }
  const int saved = errno;
  ::closedir(raw);
  if (saved != 0) return Fail(error, ErrnoText("readdir source directory", saved));
  std::sort(names->begin(), names->end());
  return true;
}

bool OpenReadonly(int directory_fd, const std::string& name, uint64_t expected_size,
                  HeldFile* out, std::string* error) {
  struct stat path_stat {};
  if (::fstatat(directory_fd, name.c_str(), &path_stat, AT_SYMLINK_NOFOLLOW) != 0)
    return Fail(error, ErrnoText("inspect source file " + name, errno));
  if (!S_ISREG(path_stat.st_mode) || path_stat.st_nlink != 1)
    return Fail(error, "source file must be an unshared regular file: " + name);
  if (expected_size != std::numeric_limits<uint64_t>::max() &&
      static_cast<uint64_t>(path_stat.st_size) != expected_size)
    return Fail(error, "source file size differs from its STL3 geometry: " + name);
  ScopedFd fd(::openat(directory_fd, name.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC));
  if (fd.get() < 0) return Fail(error, ErrnoText("open source read-only " + name, errno));
  struct stat opened {};
  if (::fstat(fd.get(), &opened) != 0)
    return Fail(error, ErrnoText("fstat source file " + name, errno));
  if (!S_ISREG(opened.st_mode) || opened.st_nlink != 1 ||
      opened.st_dev != path_stat.st_dev || opened.st_ino != path_stat.st_ino)
    return Fail(error, "source file changed identity while opening: " + name);
  out->path = name;
  out->initial.path = name;
  out->fd = std::move(fd);
  out->initial.device = opened.st_dev;
  out->initial.inode = opened.st_ino;
  out->initial.size = opened.st_size;
  out->initial.blocks_512 = opened.st_blocks;
  out->initial.mtime_ns = Ns(opened.st_mtim);
  out->initial.ctime_ns = Ns(opened.st_ctim);
  out->initial.links = opened.st_nlink;
  return true;
}

bool HashHeld(HeldFile* file, std::string* error) {
  Sha256Stream hash;
  std::array<char, 1 << 20> buffer{};
  uint64_t offset = 0;
  while (offset < file->initial.size) {
    const size_t want = static_cast<size_t>(
        std::min<uint64_t>(buffer.size(), file->initial.size - offset));
    ssize_t got;
    do {
      got = ::pread(file->fd.get(), buffer.data(), want, static_cast<off_t>(offset));
    } while (got < 0 && errno == EINTR);
    if (got <= 0) return Fail(error, ErrnoText("hash source " + file->path, got < 0 ? errno : EIO));
    hash.Update(buffer.data(), static_cast<size_t>(got));
    offset += static_cast<uint64_t>(got);
  }
  file->initial.sha256 = Hex(hash.Finish().data(), 32);
  return true;
}

std::string ManifestJson(const FileManifest& f) {
  return "{\"path\":\"" + f.path + "\",\"device\":" + std::to_string(f.device) +
         ",\"inode\":" + std::to_string(f.inode) + ",\"size\":" +
         std::to_string(f.size) + ",\"blocks_512\":" + std::to_string(f.blocks_512) +
         ",\"mtime_ns\":" + std::to_string(f.mtime_ns) + ",\"ctime_ns\":" +
         std::to_string(f.ctime_ns) + ",\"links\":" + std::to_string(f.links) +
         ",\"sha256\":\"" + f.sha256 + "\"}";
}

std::string JsonEscape(const std::string& text) {
  std::string out;
  out.reserve(text.size() + 2);
  for (unsigned char c : text) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
      out.push_back(static_cast<char>(c));
    } else if (c < 0x20) {
      static constexpr char kHex[] = "0123456789abcdef";
      out += "\\u00";
      out.push_back(kHex[c >> 4]);
      out.push_back(kHex[c & 15]);
    } else {
      out.push_back(static_cast<char>(c));
    }
  }
  return out;
}

using KnownKeys = std::unordered_map<BlockKey, size_t, BlockKeyMixHash>;

struct ExpectedValue {
  uint64_t generation = 0;
  std::string sha256;
};

bool SyncDirectory(const fs::path& path, std::string* error) {
  ScopedFd fd(::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
  if (fd.get() < 0) return Fail(error, ErrnoText("open directory for fsync " + path.string(), errno));
  if (::fsync(fd.get()) != 0) return Fail(error, ErrnoText("fsync directory " + path.string(), errno));
  return true;
}

bool MarkIncomplete(int marker_fd, int stage_fd, const std::string& reason,
                    std::string* error) {
  if (marker_fd < 0) return Fail(error, "incomplete marker descriptor was not reserved");
  if (stage_fd < 0) return Fail(error, "incomplete staging directory descriptor was not reserved");
  if (::ftruncate(marker_fd, 0) != 0 || ::lseek(marker_fd, 0, SEEK_SET) != 0)
    return Fail(error, ErrnoText("reset incomplete staging marker", errno));
  const std::string unnamed_fd = "/proc/self/fd/" + std::to_string(marker_fd);
  if (::linkat(AT_FDCWD, unnamed_fd.c_str(), stage_fd, "STO14-MIGRATION-INCOMPLETE",
               AT_SYMLINK_FOLLOW) != 0)
    return Fail(error, ErrnoText("materialize incomplete staging marker", errno));
  size_t done = 0;
  while (done < reason.size()) {
    const ssize_t n = ::write(marker_fd, reason.data() + done, reason.size() - done);
    if (n > 0) done += static_cast<size_t>(n);
    else if (n < 0 && errno == EINTR) continue;
    else return Fail(error, ErrnoText("write incomplete staging marker", n < 0 ? errno : EIO));
  }
  if (::fdatasync(marker_fd) != 0) return Fail(error, ErrnoText("sync incomplete staging marker", errno));
  if (::fsync(stage_fd) != 0) return Fail(error, ErrnoText("fsync incomplete staging directory", errno));
  return true;
}

bool ReadBlock(int fd, uint64_t offset, lse::Block* block, std::string* error) {
  if (!PreadAll(fd, block->data(), block->size(), offset))
    return Fail(error, ErrnoText("read candidate LSE block", errno));
  return true;
}

bool InspectCandidate(const fs::path& root, const KnownKeys& expected_keys,
                      const std::unordered_map<BlockKey, ExpectedValue, BlockKeyMixHash>& expected,
                      std::vector<MigratedRecord>* records, std::string* error) {
  ScopedFd root_fd(::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
  if (root_fd.get() < 0) return Fail(error, ErrnoText("open candidate read-only", errno));
  ScopedFd super_fd(::openat(root_fd.get(), "lse_super", O_RDONLY | O_NOFOLLOW | O_CLOEXEC));
  if (super_fd.get() < 0) return Fail(error, ErrnoText("open candidate superblock", errno));
  std::array<lse::Block, 2> copies{};
  for (size_t i = 0; i < copies.size(); ++i)
    if (!ReadBlock(super_fd.get(), i * lse::kBlockBytes, &copies[i], error)) return false;
  lse::Superblock sb{};
  bool have_super = false;
  for (const auto& copy : copies) {
    lse::Superblock current{};
    if (lse::DecodeSuperblock(copy.data(), false, &current) != lse::DecodeError::kOk)
      continue;
    if (!have_super || current.sequence > sb.sequence) sb = current;
    have_super = true;
  }
  if (!have_super) return Fail(error, "candidate has no valid LSE superblock");
  ScopedFd segments_fd(::openat(root_fd.get(), "segments",
                                O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
  if (segments_fd.get() < 0) return Fail(error, ErrnoText("open candidate segments", errno));
  records->clear();
  std::unordered_set<BlockKey, BlockKeyMixHash> seen;
  for (uint32_t index = 0; index < sb.segment_count; ++index) {
    char name[32];
    std::snprintf(name, sizeof(name), "G%06u", index);
    ScopedFd fd(::openat(segments_fd.get(), name, O_RDONLY | O_NOFOLLOW | O_CLOEXEC));
    if (fd.get() < 0) return Fail(error, ErrnoText("open candidate segment " + std::string(name), errno));
    lse::Block block{};
    if (!ReadBlock(fd.get(), 0, &block, error)) return false;
    lse::SegmentHeader segment_header{};
    if (lse::DecodeSegmentHeader(block.data(), sb.store_uuid, &segment_header) !=
            lse::DecodeError::kOk ||
        segment_header.segment_index != index)
      return Fail(error, "candidate segment header is invalid: " + std::string(name));
    if (!ReadBlock(fd.get(), sb.segment_bytes - lse::kBlockBytes, &block, error)) return false;
    lse::Trailer trailer{};
    std::vector<lse::SummaryEntry> summary;
    if (lse::DecodeTrailer(block.data(), sb.store_uuid, segment_header.incarnation,
                           sb.segment_bytes, &trailer) == lse::DecodeError::kOk) {
      std::vector<uint8_t> summary_bytes(trailer.summary_length);
      if (!summary_bytes.empty() &&
          !PreadAll(fd.get(), summary_bytes.data(), summary_bytes.size(), trailer.summary_offset))
        return Fail(error, "cannot read candidate summary");
      if (Crc32c(summary_bytes.data(), summary_bytes.size()) != trailer.summary_crc)
        return Fail(error, "candidate summary CRC mismatch");
      if (!lse::DecodeSummary(summary_bytes.data(), summary_bytes.size(), sb.segment_bytes, &summary) ||
          summary.size() != trailer.record_count)
        return Fail(error, "candidate summary is inconsistent");
    } else {
      // Reserved/open segments may have a valid header but no sealed trailer.
      // Walk exactly the same record layout as ordinary LSE recovery. A fresh
      // candidate's unused blocks are zero; nonzero invalid blocks are damage,
      // not silently dropped history.
      const uint64_t end = sb.segment_bytes - lse::kBlockBytes;
      uint64_t offset = lse::kBlockBytes;
      while (offset + lse::kBlockBytes <= end) {
        if (!ReadBlock(fd.get(), offset, &block, error)) return false;
        lse::RecordHeader header{};
        if (lse::DecodeRecordHeader(block.data(), sb.store_uuid, segment_header.incarnation,
                                    sb.segment_bytes, &header) == lse::DecodeError::kOk) {
          const uint64_t footprint = lse::RecordFootprint(header);
          if (header.value_len == 0 || footprint > end - offset)
            return Fail(error, "candidate unsealed record crosses its segment boundary: " +
                                   std::string(name));
          lse::SummaryEntry entry{};
          entry.key = header.key;
          entry.value_len = header.value_len;
          entry.generation = header.generation;
          entry.write_lsn = header.write_lsn;
          entry.offset_blocks = static_cast<uint32_t>(offset / lse::kBlockBytes);
          entry.flags = header.flags;
          entry.header_len = header.header_len;
          entry.payload_crc = header.payload_crc;
          summary.push_back(entry);
          offset += footprint;
        } else {
          if (std::any_of(block.begin(), block.end(), [](uint8_t c) { return c != 0; }))
            return Fail(error, "candidate unsealed segment has an invalid nonzero block: " +
                                   std::string(name));
          offset += lse::kBlockBytes;
        }
      }
    }
    for (const auto& entry : summary) {
      const uint64_t offset = static_cast<uint64_t>(entry.offset_blocks) * lse::kBlockBytes;
      if (!ReadBlock(fd.get(), offset, &block, error)) return false;
      lse::RecordHeader header{};
      if (lse::DecodeRecordHeader(block.data(), sb.store_uuid, segment_header.incarnation,
                                  sb.segment_bytes, &header) != lse::DecodeError::kOk)
        return Fail(error, "candidate record header failed decode");
      if (!(header.key == entry.key) || header.value_len != entry.value_len ||
          header.generation != entry.generation || header.write_lsn != entry.write_lsn ||
          header.flags != entry.flags || header.header_len != entry.header_len ||
          header.payload_crc != entry.payload_crc)
        return Fail(error, "candidate header and summary disagree");
      if (!seen.insert(header.key).second) return Fail(error, "duplicate candidate live record");
      const auto exp = expected.find(header.key);
      if (exp == expected.end() || expected_keys.count(header.key) == 0)
        return Fail(error, "candidate contains a key absent from the source");
      std::string payload(header.value_len, '\0');
      if (header.flags & lse::kRecordFlagInline) {
        if (header.value_len != 0)
          std::memcpy(payload.data(), block.data() + header.header_len, header.value_len);
      } else if (!PreadAll(fd.get(), payload.data(), payload.size(), offset + lse::kBlockBytes)) {
        return Fail(error, "cannot read candidate non-inline payload");
      }
      if (Crc32c(payload.data(), payload.size()) != header.payload_crc ||
          Sha256(payload.data(), payload.size()) != exp->second.sha256)
        return Fail(error, "candidate record payload/CRC differs from source");
      MigratedRecord record{};
      record.key = header.key;
      record.value_len = header.value_len;
      record.generation = header.generation;
      record.write_lsn = header.write_lsn;
      record.incarnation = header.incarnation;
      record.segment = index;
      record.offset_blocks = entry.offset_blocks;
      record.header_len = header.header_len;
      record.flags = header.flags;
      record.payload_crc = header.payload_crc;
      record.footprint = lse::RecordFootprint(header);
      record.value_sha256 = exp->second.sha256;
      if (exp->second.generation != header.generation)
        return Fail(error, "candidate Lookup and record generation disagree");
      records->push_back(std::move(record));
    }
  }
  if (records->size() != expected.size() || seen.size() != expected.size())
    return Fail(error, "candidate record count differs from source");
  return true;
}

bool HashDestination(const fs::path& root, std::vector<FileManifest>* out,
                     uint64_t* allocated, std::string* error) {
  out->clear();
  *allocated = 0;
  std::error_code ec;
  for (fs::recursive_directory_iterator it(root, ec), end; !ec && it != end;
       it.increment(ec)) {
    if (it->is_directory(ec)) continue;
    if (ec) break;
    const fs::path path = it->path();
    const std::string relative = fs::relative(path, root, ec).string();
    if (ec) break;
    ScopedFd fd(::open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC));
    if (fd.get() < 0) return Fail(error, ErrnoText("open output file " + relative, errno));
    struct stat st {};
    if (::fstat(fd.get(), &st) != 0) return Fail(error, ErrnoText("stat output file", errno));
    if (!S_ISREG(st.st_mode) || st.st_nlink != 1)
      return Fail(error, "output contains an unexpected non-regular/shared file: " + relative);
    Sha256Stream hash;
    std::array<char, 1 << 20> buffer{};
    uint64_t offset = 0;
    while (offset < static_cast<uint64_t>(st.st_size)) {
      ssize_t got;
      do {
        got = ::pread(fd.get(), buffer.data(),
                      std::min<uint64_t>(buffer.size(), st.st_size - offset),
                      static_cast<off_t>(offset));
      } while (got < 0 && errno == EINTR);
      if (got <= 0) return Fail(error, "short read hashing output " + relative);
      hash.Update(buffer.data(), static_cast<size_t>(got));
      offset += static_cast<uint64_t>(got);
    }
    FileManifest file{};
    file.path = relative;
    file.device = st.st_dev;
    file.inode = st.st_ino;
    file.size = st.st_size;
    file.blocks_512 = st.st_blocks;
    file.mtime_ns = Ns(st.st_mtim);
    file.ctime_ns = Ns(st.st_ctim);
    file.links = st.st_nlink;
    file.sha256 = Hex(hash.Finish().data(), 32);
    *allocated += file.blocks_512 * 512;
    out->push_back(std::move(file));
  }
  if (ec) return Fail(error, "cannot enumerate output: " + ec.message());
  std::sort(out->begin(), out->end(), [](const auto& a, const auto& b) { return a.path < b.path; });
  return true;
}

}  // namespace

struct SlabSnapshot::Impl {
  std::string dir;
  ScopedFd root_fd;
  ScopedFd extents_fd;
  HeldFile meta;
  HeldFile state;
  HeldFile table;
  std::vector<HeldFile> extents;
  std::vector<FileManifest> initial;
  std::vector<SourceEntry> entries;
  uint64_t extent_bytes = 0;
  uint64_t slot_granularity = 0;
  uint32_t extent_count = 0;
  uint64_t state_epoch = 0;

  ~Impl() = default;  // every descriptor remains O_RDONLY for its lifetime.

  bool VerifyUnchanged(std::string* error) const {
    std::vector<std::string> expected_root{"extents", "slab_meta", "slab_state", "slots.tbl"};
    std::vector<std::string> actual_root;
    if (!ListNames(root_fd.get(), &actual_root, error)) return false;
    if (actual_root != expected_root) {
      std::string found;
      for (const auto& name : actual_root) found += (found.empty() ? "" : ",") + name;
      return Fail(error, "source root inventory changed; found: " + found);
    }
    std::vector<std::string> expected_extents;
    expected_extents.reserve(extents.size());
    for (const auto& file : extents) expected_extents.push_back(file.path);
    std::vector<std::string> actual_extents;
    if (!ListNames(extents_fd.get(), &actual_extents, error)) return false;
    if (!ExtentInventoryMatches(expected_extents, actual_extents)) {
      std::string found;
      for (const auto& name : actual_extents) found += (found.empty() ? "" : ",") + name;
      return Fail(error, "source extent inventory changed; found: " + found);
    }
    std::vector<std::pair<int, const HeldFile*>> checks;
    checks.push_back({root_fd.get(), &meta});
    checks.push_back({root_fd.get(), &state});
    checks.push_back({root_fd.get(), &table});
    for (const auto& extent : extents) checks.push_back({extents_fd.get(), &extent});
    std::vector<FileManifest> after;
    after.reserve(checks.size());
    for (const auto& check : checks) {
      const HeldFile& original = *check.second;
      struct stat path_st {};
      if (::fstatat(check.first, original.path.c_str(), &path_st, AT_SYMLINK_NOFOLLOW) != 0)
        return Fail(error, "source file disappeared: " + original.path);
      struct stat held_st {};
      if (::fstat(original.fd.get(), &held_st) != 0)
        return Fail(error, "cannot restat source file: " + original.path);
      HeldFile current;
      current.path = original.path;
      current.fd = ScopedFd(::openat(check.first, original.path.c_str(),
                                     O_RDONLY | O_NOFOLLOW | O_CLOEXEC));
      if (current.fd.get() < 0) return Fail(error, "cannot reopen source " + original.path);
      current.initial = original.initial;
      current.initial.blocks_512 = held_st.st_blocks;
      if (!HashHeld(&current, error)) return false;
      if (path_st.st_dev != held_st.st_dev || path_st.st_ino != held_st.st_ino ||
          held_st.st_dev != original.initial.device || held_st.st_ino != original.initial.inode ||
          static_cast<uint64_t>(held_st.st_size) != original.initial.size ||
          static_cast<uint64_t>(held_st.st_blocks) != original.initial.blocks_512 ||
          Ns(held_st.st_mtim) != original.initial.mtime_ns ||
          Ns(held_st.st_ctim) != original.initial.ctime_ns ||
          held_st.st_nlink != 1 || current.initial.sha256 != original.initial.sha256)
        return Fail(error, "source file changed during migration: " + original.path);
      current.initial.blocks_512 = original.initial.blocks_512;
      after.push_back(std::move(current.initial));
    }
    return true;
  }
};

SlabSnapshot::SlabSnapshot(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SlabSnapshot::~SlabSnapshot() = default;
const std::string& SlabSnapshot::dir() const { return impl_->dir; }
uint64_t SlabSnapshot::extent_bytes() const { return impl_->extent_bytes; }
uint64_t SlabSnapshot::slot_granularity() const { return impl_->slot_granularity; }
uint32_t SlabSnapshot::extent_count() const { return impl_->extent_count; }
uint64_t SlabSnapshot::state_epoch() const { return impl_->state_epoch; }
const std::vector<SourceEntry>& SlabSnapshot::entries() const { return impl_->entries; }
const std::vector<FileManifest>& SlabSnapshot::initial_manifest() const { return impl_->initial; }

std::unique_ptr<SlabSnapshot> SlabSnapshot::Open(const std::string& dir, std::string* error) {
  auto impl = std::make_unique<Impl>();
  impl->dir = dir;
  const fs::path root(dir);
  if (!CanonicalExistingPath(root, error)) return nullptr;
  struct stat root_st {};
  if (::lstat(root.c_str(), &root_st) != 0 || !S_ISDIR(root_st.st_mode)) {
    Fail(error, "source root is not a readable directory");
    return nullptr;
  }
  impl->root_fd = ScopedFd(::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
  if (impl->root_fd.get() < 0) {
    Fail(error, ErrnoText("open source root read-only", errno));
    return nullptr;
  }
  if (::flock(impl->root_fd.get(), LOCK_EX | LOCK_NB) != 0) {
    Fail(error, ErrnoText("lock source offline (another engine may be using it)", errno));
    return nullptr;
  }
  std::vector<std::string> names;
  if (!ListNames(impl->root_fd.get(), &names, error)) return nullptr;
  const std::vector<std::string> expected{"extents", "slab_meta", "slab_state", "slots.tbl"};
  if (names != expected) {
    Fail(error, "source is not a complete, initialized STL3-only root; missing/foreign/history files are rejected");
    return nullptr;
  }
  impl->extents_fd = ScopedFd(::openat(impl->root_fd.get(), "extents",
                                       O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
  if (impl->extents_fd.get() < 0) {
    Fail(error, ErrnoText("open source extents read-only", errno));
    return nullptr;
  }
  if (!OpenReadonly(impl->root_fd.get(), "slab_meta", kMetadataBytes, &impl->meta, error) ||
      !OpenReadonly(impl->root_fd.get(), "slab_state", kMetadataBytes, &impl->state, error))
    return nullptr;
  std::array<uint8_t, kMetadataBytes> meta{};
  if (!PreadAll(impl->meta.fd.get(), meta.data(), meta.size(), 0)) {
    Fail(error, "cannot read slab_meta");
    return nullptr;
  }
  if (net::GetU32(reinterpret_cast<char*>(meta.data())) != kMetaMagic ||
      net::GetU32(reinterpret_cast<char*>(meta.data()) + 4) != kFormatVersion ||
      std::any_of(meta.begin() + 28, meta.end(), [](uint8_t c) { return c != 0; })) {
    Fail(error, "unsupported slab_meta magic/version/reserved fields");
    return nullptr;
  }
  impl->extent_bytes = net::GetU64(reinterpret_cast<char*>(meta.data()) + 8);
  impl->slot_granularity = net::GetU64(reinterpret_cast<char*>(meta.data()) + 16);
  impl->extent_count = net::GetU32(reinterpret_cast<char*>(meta.data()) + 24);
  if (impl->extent_count == 0 || impl->extent_bytes == 0 ||
      impl->extent_bytes > std::numeric_limits<uint32_t>::max() ||
      impl->slot_granularity < 4096 ||
      impl->slot_granularity > std::numeric_limits<uint32_t>::max() ||
      impl->slot_granularity % 4096 != 0 ||
      impl->extent_bytes % impl->slot_granularity != 0) {
    Fail(error, "invalid or mixed slab slot geometry");
    return nullptr;
  }
  const uint64_t slots_per_extent = impl->extent_bytes / impl->slot_granularity;
  if (slots_per_extent == 0 || slots_per_extent > std::numeric_limits<uint32_t>::max() ||
      impl->extent_count > std::numeric_limits<uint64_t>::max() / slots_per_extent / kRecordBytes ||
      impl->extent_count * slots_per_extent * kRecordBytes >
          static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
    Fail(error, "slab table geometry overflows its native representation");
    return nullptr;
  }
  SlabAllocator::Options allocator_options;
  allocator_options.align = impl->slot_granularity;
  allocator_options.extent_bytes = impl->extent_bytes;
  const uint64_t table_bytes = impl->extent_count * slots_per_extent * kRecordBytes;
  if (!OpenReadonly(impl->root_fd.get(), "slots.tbl", table_bytes, &impl->table, error))
    return nullptr;
  std::array<uint8_t, kMetadataBytes> state{};
  if (!PreadAll(impl->state.fd.get(), state.data(), state.size(), 0)) {
    Fail(error, "cannot read slab_state");
    return nullptr;
  }
  if (net::GetU32(reinterpret_cast<char*>(state.data())) != kStateMagic ||
      net::GetU32(reinterpret_cast<char*>(state.data()) + 4) != kFormatVersion ||
      net::GetU32(reinterpret_cast<char*>(state.data()) + 60) != Crc32(state.data(), 60) ||
      net::GetU32(reinterpret_cast<char*>(state.data()) + 16) != kStateClean) {
    Fail(error, "slab_state is missing history, dirty, corrupt or not a clean STL3 epoch");
    return nullptr;
  }
  impl->state_epoch = net::GetU64(reinterpret_cast<char*>(state.data()) + 8);
  impl->extents.reserve(impl->extent_count);
  for (uint32_t i = 0; i < impl->extent_count; ++i) {
    char name[32];
    std::snprintf(name, sizeof(name), "E%05u", i);
    HeldFile extent;
    if (!OpenReadonly(impl->extents_fd.get(), name, impl->extent_bytes, &extent, error))
      return nullptr;
    impl->extents.push_back(std::move(extent));
  }
  std::vector<std::string> extent_names;
  if (!ListNames(impl->extents_fd.get(), &extent_names, error)) return nullptr;
  std::vector<std::string> expected_extents;
  expected_extents.reserve(impl->extent_count);
  for (const auto& file : impl->extents) expected_extents.push_back(file.path);
  if (!ExtentInventoryMatches(expected_extents, extent_names)) {
    Fail(error, "source extents contain missing, foreign or history files");
    return nullptr;
  }

  KnownKeys live_keys;
  std::set<std::pair<uint32_t, uint32_t>> occupied_slots;
  std::unordered_map<uint32_t, uint32_t> extent_slot_sizes;
  std::vector<uint8_t> buffer(static_cast<size_t>(std::min<uint64_t>(table_bytes, 1 << 20)));
  for (uint64_t base = 0; base < table_bytes; base += buffer.size()) {
    const size_t bytes = static_cast<size_t>(std::min<uint64_t>(buffer.size(), table_bytes - base));
    if (!PreadAll(impl->table.fd.get(), buffer.data(), bytes, base)) {
      Fail(error, "cannot scan source slots.tbl");
      return nullptr;
    }
    for (size_t pos = 0; pos < bytes; pos += kRecordBytes) {
      const uint8_t* rec = buffer.data() + pos;
      const uint64_t index = (base + pos) / kRecordBytes;
      const uint32_t extent = static_cast<uint32_t>(index / slots_per_extent);
      const uint32_t slot = static_cast<uint32_t>(index % slots_per_extent);
      const uint32_t magic = net::GetU32(reinterpret_cast<const char*>(rec));
      if (magic == 0) {
        if (std::any_of(rec, rec + kRecordBytes, [](uint8_t c) { return c != 0; })) {
          Fail(error, "partial or high-shape zero-magic table row at extent " +
                          std::to_string(extent) + " slot " + std::to_string(slot));
          return nullptr;
        }
        continue;
      }
      if (magic != kRecMagic) {
        Fail(error, "unknown nonzero table row magic; source is not silently reset");
        return nullptr;
      }
      if (net::GetU32(reinterpret_cast<const char*>(rec) + 4) !=
          Crc32(rec + 8, kRecordBytes - 8)) {
        Fail(error, "corrupt STL3 table row CRC; source left untouched");
        return nullptr;
      }
      if (rec[8] != 1) {
        Fail(error, "tombstone/history/non-live STL3 row is not representable by this migration");
        return nullptr;
      }
      // digest_hi and digest_lo occupy the whole [16,32) interval; the upper
      // half of digest_lo is identity, not reserved zero padding.
      if ((rec[9] & ~kRecordPayloadCrc) != 0 ||
          std::any_of(rec + 10, rec + 12, [](uint8_t c) { return c != 0; }) ||
          std::any_of(rec + 56, rec + kRecordBytes, [](uint8_t c) { return c != 0; })) {
        Fail(error, "unsupported STL3 flags or reserved fields");
        return nullptr;
      }
      SourceEntry entry{};
      entry.extent = extent;
      entry.slot = slot;
      entry.slot_size = net::GetU32(reinterpret_cast<const char*>(rec) + 12);
      entry.key.digest_hi = net::GetU64(reinterpret_cast<const char*>(rec) + 16);
      entry.key.digest_lo = net::GetU64(reinterpret_cast<const char*>(rec) + 24);
      entry.payload_len = net::GetU32(reinterpret_cast<const char*>(rec) + 32);
      entry.key.tenant_hash = net::GetU64(reinterpret_cast<const char*>(rec) + 36);
      const uint32_t retain_until = net::GetU32(reinterpret_cast<const char*>(rec) + 44);
      const uint32_t retain_ttl = net::GetU32(reinterpret_cast<const char*>(rec) + 48);
      if (retain_until != 0 || retain_ttl != 0) {
        Fail(error, "retention/TTL records are rejected instead of erasing their semantics");
        return nullptr;
      }
      if (entry.payload_len == 0) {
        Fail(error, "a zero-length live value is unsupported by ordinary LSE recovery");
        return nullptr;
      }
      const bool geometry = entry.slot_size >= impl->slot_granularity &&
                            entry.slot_size % impl->slot_granularity == 0 &&
                            entry.slot_size <= impl->extent_bytes &&
                            entry.payload_len <= entry.slot_size &&
                            slot < impl->extent_bytes / entry.slot_size &&
                            SlabAllocator::CanonicalSlotSizeForGeometry(
                                entry.slot_size, allocator_options) == entry.slot_size;
      if (!geometry) {
        Fail(error, "STL3 row has invalid, mixed or noncanonical slot geometry");
        return nullptr;
      }
      const auto [extent_class, inserted_class] =
          extent_slot_sizes.emplace(extent, entry.slot_size);
      if (!inserted_class && extent_class->second != entry.slot_size) {
        Fail(error, "mixed slot classes in one source extent are inconsistent");
        return nullptr;
      }
      if (!occupied_slots.insert({extent, slot}).second) {
        Fail(error, "two live records name one physical slab slot");
        return nullptr;
      }
      if (!live_keys.emplace(entry.key, impl->entries.size()).second) {
        Fail(error, "duplicate/conflicting live BlockKey (tenant_hash,digest_hi,digest_lo)");
        return nullptr;
      }
      entry.has_payload_crc = (rec[9] & kRecordPayloadCrc) != 0;
      if (entry.has_payload_crc)
        entry.payload_crc = net::GetU32(reinterpret_cast<const char*>(rec) + 52);
      impl->entries.push_back(entry);
    }
  }
  if (!impl->entries.empty() && impl->state_epoch == 0) {
    Fail(error, "live rows cannot belong to the never-activated clean epoch zero");
    return nullptr;
  }
  for (HeldFile* file : {&impl->meta, &impl->state, &impl->table}) {
    if (!HashHeld(file, error)) return nullptr;
    impl->initial.push_back(file->initial);
  }
  for (auto& extent : impl->extents) {
    if (!HashHeld(&extent, error)) return nullptr;
    FileManifest manifest = extent.initial;
    manifest.path = "extents/" + manifest.path;
    impl->initial.push_back(std::move(manifest));
  }
  return std::unique_ptr<SlabSnapshot>(new SlabSnapshot(std::move(impl)));
}

bool SlabSnapshot::ReadValue(size_t entry_index, std::string* value,
                             std::string* sha256, std::string* error) const {
  if (entry_index >= impl_->entries.size() || value == nullptr || sha256 == nullptr)
    return Fail(error, "source value index is out of bounds");
  const SourceEntry& entry = impl_->entries[entry_index];
  value->assign(entry.payload_len, '\0');
  // SlotRef::offset is extent-local: the descriptor already selects E%05u.
  const uint64_t offset = static_cast<uint64_t>(entry.slot) * entry.slot_size;
  if (!PreadAll(impl_->extents[entry.extent].fd.get(), value->data(), value->size(), offset))
    return Fail(error, "cannot read source value for " + entry.key.Filename());
  if (entry.has_payload_crc && Crc32c(value->data(), value->size()) != entry.payload_crc)
    return Fail(error, "source payload CRC mismatch for " + entry.key.Filename());
  *sha256 = Sha256(value->data(), value->size());
  return true;
}

bool SlabSnapshot::VerifyUnchanged(std::string* error) const {
  return impl_->VerifyUnchanged(error);
}

MigrationResult MigrateToLse(const SlabSnapshot& source, const std::string& destination) {
  const auto began = std::chrono::steady_clock::now();
  MigrationResult result;
  result.source_dir = source.dir();
  result.destination_dir = destination;
  result.staging_dir = destination + ".sto14-staging";
  result.source_before = source.initial_manifest();
  KnownKeys source_keys;
  std::unordered_map<BlockKey, ExpectedValue, BlockKeyMixHash> expected;
  uint64_t max_value = 0;
  auto finish = [&]() {
    result.elapsed_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - began).count());
    return result;
  };
  ScopedFd marker_fd;
  ScopedFd stage_fd;
  auto fail_stage = [&](const std::string& message) {
    result.error = message;
    std::string marker_error;
    if (!MarkIncomplete(marker_fd.get(), stage_fd.get(), message, &marker_error))
      result.error += "; also failed to mark staging: " + marker_error;
    return finish();
  };
  uint64_t source_allocated = 0;
  for (const auto& file : result.source_before) {
    result.source_table_bytes += file.path == "slots.tbl" ? file.size : 0;
    source_allocated += file.blocks_512 * 512;
  }
  result.source_allocated_bytes = source_allocated;
  for (const auto& entry : source.entries()) {
    source_keys.emplace(entry.key, expected.size());
    if (result.source_slot_charge >
        std::numeric_limits<uint64_t>::max() - entry.slot_size) {
      result.error = "source slot charge overflows uint64";
      return finish();
    }
    result.source_slot_charge += entry.slot_size;
    max_value = std::max<uint64_t>(max_value, entry.payload_len);
  }
  const auto fits_largest_value = [max_value](uint64_t segment) {
    // MaxRecordValue accounts for segment header, record header and trailer.
    // Ordinary LogStore admission also reserves the minimum sealed summary.
    const uint64_t raw_capacity = lse::MaxRecordValue(segment);
    return raw_capacity >= lse::kBlockBytes &&
           raw_capacity - lse::kBlockBytes >= max_value;
  };
  uint64_t segment_bytes = kMinSegmentBytes;
  while (segment_bytes < kMaxSegmentBytes && !fits_largest_value(segment_bytes))
    segment_bytes *= 2;
  if (!fits_largest_value(segment_bytes)) {
    result.error = "largest slab value exceeds ordinary LSE MaxValueBytes";
    return finish();
  }
  if (source.entries().size() >
      (std::numeric_limits<uint64_t>::max() - (lse::kBlockBytes - 1)) /
          lse::kSummaryEntryBytes) {
    result.error = "LSE summary sizing overflows uint64";
    return finish();
  }
  // Model the writer selected below: exactly one client segment and no
  // preopening. PlaceLocked charges each footprint, the next 64-byte summary
  // entry (rounded to a block), and fixed header/trailer blocks. Closing a
  // segment leaves it sealed, not free; client admission retains two free
  // cleaner-reserve segments. Modeling that admission prevents eviction and
  // accounts for nonuniform record sizes as well as the all-inline boundary.
  uint64_t active_segments = source.entries().empty() ? 0 : 1;
  uint64_t append_offset = lse::kBlockBytes;
  uint64_t records_in_segment = 0;
  auto summary_room = [](uint64_t records) {
    return (records * lse::kSummaryEntryBytes + lse::kBlockBytes - 1) /
           lse::kBlockBytes * lse::kBlockBytes;
  };
  for (const auto& entry : source.entries()) {
    lse::RecordHeader shape{};
    shape.value_len = entry.payload_len;
    shape.flags = lse::RecordFitsInline(entry.payload_len) ? lse::kRecordFlagInline : 0;
    const uint64_t footprint = lse::RecordFootprint(shape);
    if (append_offset + footprint + summary_room(records_in_segment + 1) +
            lse::kBlockBytes > segment_bytes) {
      ++active_segments;
      append_offset = lse::kBlockBytes;
      records_in_segment = 0;
    }
    if (append_offset + footprint + summary_room(records_in_segment + 1) +
            lse::kBlockBytes > segment_bytes) {
      result.error = "one source record cannot fit a fresh LSE segment";
      return finish();
    }
    append_offset += footprint;
    ++records_in_segment;
  }
  const uint64_t count = std::max<uint64_t>(8, active_segments + 2);
  if (count > std::numeric_limits<uint32_t>::max() ||
      count > std::numeric_limits<uint64_t>::max() / segment_bytes) {
    result.error = "required LSE segment count/capacity is unsupported";
    return finish();
  }
  result.segment_bytes = segment_bytes;
  result.segment_count = static_cast<uint32_t>(count);
  result.capacity_bytes = count * segment_bytes;
  LogStore::Geometry geometry{};
  if (!LogStore::GeometryFor(result.capacity_bytes, result.segment_bytes, &geometry, &result.error))
    return finish();
  if (geometry.segment_bytes != segment_bytes || geometry.segment_count != count) {
    result.error = "internal LSE geometry disagreement";
    return finish();
  }
  std::string error;
  const fs::path dest(destination);
  const fs::path stage(result.staging_dir);
  if (!dest.is_absolute() || dest != dest.lexically_normal()) {
    result.error = "destination must be absolute and lexically normalized";
    return finish();
  }
  if (!PathAbsent(dest, &result.error) || !PathAbsent(stage, &result.error)) return finish();
  const fs::path parent = dest.parent_path();
  if (!CanonicalExistingPath(parent, &result.error)) return finish();
  const fs::path canonical_source = fs::canonical(source.dir());
  auto within = [](const fs::path& path, const fs::path& root) {
    auto p = path.begin(), r = root.begin();
    for (; r != root.end(); ++r, ++p)
      if (p == path.end() || *p != *r) return false;
    return true;
  };
  if (within(dest, canonical_source) || within(stage, canonical_source) ||
      dest == canonical_source || stage == canonical_source) {
    result.error = "destination/staging may not alias or be inside the protected source";
    return finish();
  }
  if (!source.VerifyUnchanged(&result.error)) return finish();
  std::error_code ec;
  if (!fs::create_directory(stage, ec) || ec) {
    result.error = ec ? "cannot atomically create staging: " + ec.message()
                      : "staging appeared concurrently; refusing existing tree";
    return finish();
  }
  // Keep the staging directory available for every failure-marking path,
  // including a partially opened verifier under RLIMIT_NOFILE exhaustion.
  stage_fd.Reset(::open(stage.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
  if (stage_fd.get() < 0) {
    result.error = ErrnoText("open staging directory for failure ownership", errno);
    return finish();
  }
  // No O_EXCL: /proc/self/fd/N may materialize this unnamed inode on failure.
  // The directory inventory stays clear for the ordinary LSE constructor.
  marker_fd.Reset(::open(stage.c_str(), O_TMPFILE | O_WRONLY | O_CLOEXEC, 0600));
  if (marker_fd.get() < 0) {
    result.error = ErrnoText("reserve unnamed failure marker on staging filesystem", errno);
    return finish();
  }

  {
    LogStore::Options options;
    options.dir = stage.string();
    options.capacity_bytes = result.capacity_bytes;
    options.segment_bytes = result.segment_bytes;
    options.direct_writes = false;
    options.cleaner = false;
    options.preopen = false;
    options.open_segments = 1;
    bool opened = false;
    std::optional<LogStore> writer(std::in_place, options, &opened);
    if (!opened) {
      const auto startup_error = writer->StartupError();
      writer.reset();  // release partial-open FDs before failure marking.
      return fail_stage("cannot create ordinary LSE writer: " + startup_error);
    }
    for (size_t i = 0; i < source.entries().size(); ++i) {
      const SourceEntry& entry = source.entries()[i];
      std::string value;
      std::string sha;
      if (!source.ReadValue(i, &value, &sha, &error))
        return fail_stage(error);
      const Status status = writer->Cache(entry.key, value.data(), value.size());
      if (status != Status::kOk)
        return fail_stage("ordinary LSE Cache rejected source value; no subset is published: " +
                          entry.key.Filename() + " status " + std::to_string(static_cast<int>(status)));
      expected.emplace(entry.key, ExpectedValue{0, std::move(sha)});
    }
    if (writer->Count() != source.entries().size())
      return fail_stage("LSE writer live count differs before close; refusing eviction/subset");
    if (!writer->Close()) return fail_stage("ordinary LSE writer did not close cleanly");
  }
  {
    LogStore::Options options;
    options.dir = stage.string();
    options.capacity_bytes = result.capacity_bytes;
    options.direct_writes = false;
    options.cleaner = false;
    options.preopen = false;
    options.open_segments = 1;
    bool reopened = false;
    std::optional<LogStore> verifier(std::in_place, options, &reopened);
    if (!reopened) {
      const auto startup_error = verifier->StartupError();
      verifier.reset();  // release partial-open FDs before failure marking.
      return fail_stage("candidate cannot reopen through ordinary LSE: " + startup_error);
    }
    if (verifier->Count() != source.entries().size())
      return fail_stage("reopened LSE live count differs from source");
    KnownKeys reopened_keys;
    verifier->ForEachKey([&](const BlockKey& key) { reopened_keys.emplace(key, 0); });
    if (reopened_keys.size() != source_keys.size())
      return fail_stage("reopened LSE key set size differs from source");
    for (const auto& entry : source.entries()) {
      if (reopened_keys.count(entry.key) == 0)
        return fail_stage("reopened LSE is missing exact source key " + entry.key.Filename());
      std::string actual;
      size_t actual_len = 0;
      if (verifier->Range(entry.key, 0, 0, &actual, &actual_len) != Status::kOk ||
          actual_len != entry.payload_len || Sha256(actual.data(), actual.size()) != expected[entry.key].sha256)
        return fail_stage("reopened LSE payload/length differs for " + entry.key.Filename());
      ValueMetadata metadata{};
      if (verifier->Lookup(entry.key, &metadata) != Status::kOk || metadata.generation == 0 ||
          metadata.value_len != entry.payload_len)
        return fail_stage("reopened LSE Lookup differs for " + entry.key.Filename());
      expected[entry.key].generation = metadata.generation;
    }
    std::vector<MigratedRecord> audited;
    if (!InspectCandidate(stage, source_keys, expected, &audited, &error))
      return fail_stage(error);
    std::unordered_map<BlockKey, const SourceEntry*, BlockKeyMixHash> source_by_key;
    for (const auto& entry : source.entries()) source_by_key.emplace(entry.key, &entry);
    for (auto& record : audited) {
      const SourceEntry& original = *source_by_key.at(record.key);
      record.source_slot_charge = original.slot_size;
      record.source_has_payload_crc = original.has_payload_crc;
      record.source_payload_crc = original.payload_crc;
    }
    result.records = std::move(audited);
    if (!verifier->Close()) return fail_stage("candidate LSE verifier did not close cleanly");
  }  // verifier is destroyed, its lock and descriptors released before publish.
  if (!source.VerifyUnchanged(&result.error)) return fail_stage(result.error);
  result.source_after = result.source_before;
  result.source_unchanged = true;
  if (!HashDestination(stage, &result.destination_files,
                       &result.destination_allocated_bytes, &result.error))
    return fail_stage(result.error);
  if (!SyncDirectory(stage / "segments", &result.error) || !SyncDirectory(stage, &result.error))
    return fail_stage(result.error);
  long rc;
  do {
    rc = ::syscall(SYS_renameat2, AT_FDCWD, stage.c_str(), AT_FDCWD, dest.c_str(),
                   RENAME_NOREPLACE);
  } while (rc != 0 && errno == EINTR);
  if (rc != 0) return fail_stage(ErrnoText("atomic no-replace publish failed", errno));
  result.published = true;
  if (!SyncDirectory(parent, &result.error)) {
    result.error += "; candidate was atomically published but parent-directory durability is uncertain";
    return finish();
  }
  result.ok = true;
  return finish();
}

std::string ResultJson(const MigrationResult& r) {
  auto manifests = [](const std::vector<FileManifest>& files) {
    std::string out = "[";
    for (size_t i = 0; i < files.size(); ++i) {
      if (i != 0) out += ',';
      out += ManifestJson(files[i]);
    }
    return out + "]";
  };
  std::string out = "{\"ok\":" + std::string(r.ok ? "true" : "false") +
                    ",\"published\":" + (r.published ? "true" : "false") +
                    ",\"source_unchanged\":" + (r.source_unchanged ? "true" : "false") +
                    ",\"error\":\"" + JsonEscape(r.error) + "\",\"source_dir\":\"" +
                    JsonEscape(r.source_dir) + "\",\"destination_dir\":\"" +
                    JsonEscape(r.destination_dir) + "\",\"staging_dir\":\"" +
                    JsonEscape(r.staging_dir) + "\",\"capacity_bytes\":" +
                    std::to_string(r.capacity_bytes) + ",\"segment_bytes\":" +
                    std::to_string(r.segment_bytes) + ",\"segment_count\":" +
                    std::to_string(r.segment_count) + ",\"source_slot_charge\":" +
                    std::to_string(r.source_slot_charge) + ",\"source_table_bytes\":" +
                    std::to_string(r.source_table_bytes) + ",\"source_allocated_bytes\":" +
                    std::to_string(r.source_allocated_bytes) + ",\"destination_allocated_bytes\":" +
                    std::to_string(r.destination_allocated_bytes) + ",\"elapsed_us\":" +
                    std::to_string(r.elapsed_us) + ",\"record_count\":" +
                    std::to_string(r.records.size()) + ",\"records\":[";
  for (size_t i = 0; i < r.records.size(); ++i) {
    const auto& v = r.records[i];
    if (i != 0) out += ',';
    out += "{\"key\":\"" + v.key.Filename() + "\",\"tenant_hash\":" +
           std::to_string(v.key.tenant_hash) + ",\"digest_hi\":" +
           std::to_string(v.key.digest_hi) + ",\"digest_lo\":" +
           std::to_string(v.key.digest_lo) + ",\"value_len\":" +
           std::to_string(v.value_len) + ",\"source_slot_charge\":" +
           std::to_string(v.source_slot_charge) + ",\"source_has_payload_crc\":" +
           (v.source_has_payload_crc ? "true" : "false") +
           ",\"source_payload_crc\":" + std::to_string(v.source_payload_crc) +
           ",\"generation\":" + std::to_string(v.generation) + ",\"write_lsn\":" +
           std::to_string(v.write_lsn) + ",\"incarnation\":" +
           std::to_string(v.incarnation) + ",\"segment\":" + std::to_string(v.segment) +
           ",\"offset_blocks\":" + std::to_string(v.offset_blocks) +
           ",\"header_len\":" + std::to_string(v.header_len) + ",\"flags\":" +
           std::to_string(v.flags) + ",\"payload_crc\":" + std::to_string(v.payload_crc) +
           ",\"footprint\":" + std::to_string(v.footprint) + ",\"value_sha256\":\"" +
           v.value_sha256 + "\"}";
  }
  out += "],\"source_before\":" + manifests(r.source_before) +
         ",\"source_after\":" + manifests(r.source_after) +
         ",\"destination_files\":" + manifests(r.destination_files) + "}";
  return out;
}

}  // namespace slab_migration
}  // namespace dfkv
