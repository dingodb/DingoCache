// Opt-in, evidence-preserving M3 store-level capacity, recovery, and I/O tests.
// Existing capacity acceptance remains distinct from comparative I/O evidence.
// The runner pins and reports the exact source HEAD separately; this test never
// invokes git or discovers a benchmark directory on its own.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <fcntl.h>
#include <gtest/gtest.h>
#include <poll.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "cache/disk_slab_store.h"
#include "cache/log_store.h"
#include "common/status.h"

namespace dfkv {

// Test-only barriers.  Neither one records a clean shutdown: the process is
// killed only after the writes and their persistent index metadata are synced.
class LogStoreTestPeer {
 public:
  static bool SyncDirtySegments(LogStore* store) {
    LogStore::Lock lock(store->mu_);
    for (auto& segment : store->segments_) {
      if (!segment.dirty) continue;
      if (segment.fd < 0 || store->io_->Fdatasync(segment.fd) != 0) return false;
    }
    return true;
  }
};

class DiskSlabStoreTestPeer {
 public:
  static bool SyncPayloadsAndTable(DiskSlabStore* store) {
    // A valid table record must never become durable ahead of its payload.
    // SyncTable also finishes withdrawing and regranting the trust token.
    return store->SyncPayloads() &&
           store->SyncTable(store->TableWriteStamp());
  }
};

}  // namespace dfkv

namespace {

using dfkv::BlockKey;
using dfkv::DiskSlabStore;
using dfkv::LogStore;
using dfkv::Status;
using dfkv::StoreEngine;
using dfkv::ValueMetadata;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr uint64_t kValueBytes = 2873098;  // round(2.74 * 2^20)
constexpr uint64_t kStandardCapacity = 4ull << 30;
constexpr uint64_t kRequiredFreeBytes = 400ull << 30;
constexpr uint64_t kSlabGranularity = 1ull << 20;
constexpr uint64_t kSlabExtent = 1ull << 30;
constexpr uint64_t kLogSegment = 64ull << 20;
constexpr uint32_t kWriters = 8;
constexpr uint32_t kStandardRepetitions = 3;
constexpr size_t kIoAlignment = 4096;
constexpr char kJsonlName[] = "lse_m3_capacity.jsonl";

static_assert(kValueBytes == static_cast<uint64_t>(2.74 * (1ull << 20) + 0.5),
              "the acceptance value size must remain exact");

struct Config {
  fs::path root;
  uint64_t capacity = kStandardCapacity;
  uint32_t repetitions = kStandardRepetitions;
  uint64_t puts_per_run = 0;
  bool capacity_override = false;
  bool acceptance_eligible = true;
};

struct BackingCandidate {
  fs::path source;
  fs::path resolved;
  bool physical_leaf = false;
  bool verified = false;
  fs::path nvme_subsystem_link;
  fs::path nvme_controller_link;
  fs::path nvme_controller_device;
  fs::path nvme_block_link;
  fs::path block_device;
  std::string error;
};

struct RootEvidence {
  fs::path supplied_root;
  fs::path normalized_root;
  fs::path normalized_parent;
  uint64_t available_bytes = 0;
  unsigned device_major = 0;
  unsigned device_minor = 0;
  std::string sysfs_dev_block;
  std::string resolved_device;
  std::string nvme_subsystem_link;
  std::string nvme_subsystem;
  fs::path nvme_controller_link;
  fs::path nvme_controller_device;
  std::vector<BackingCandidate> candidates;
  size_t physical_leaf_count = 0;
  bool all_leaves_nvme = false;
};

enum class EngineKind { kSlab, kLog };

const char* EngineName(EngineKind kind) {
  return kind == EngineKind::kSlab ? "slab" : "log";
}

std::string JsonString(const std::string& value) {
  std::ostringstream out;
  out << '"';
  constexpr char kHex[] = "0123456789abcdef";
  for (const unsigned char c : value) {
    switch (c) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) {
          out << "\\u00" << kHex[c >> 4] << kHex[c & 0xf];
        } else {
          out << static_cast<char>(c);
        }
    }
  }
  out << '"';
  return out.str();
}

std::string JsonStrings(const std::vector<std::string>& values) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ',';
    out << JsonString(values[i]);
  }
  out << ']';
  return out.str();
}

std::string BackingCandidatesJson(const RootEvidence& evidence) {
  std::ostringstream out;
  out << std::boolalpha << '[';
  for (size_t i = 0; i < evidence.candidates.size(); ++i) {
    const BackingCandidate& candidate = evidence.candidates[i];
    if (i != 0) out << ',';
    out << "{\"source\":" << JsonString(candidate.source.string())
        << ",\"resolved\":" << JsonString(candidate.resolved.string())
        << ",\"physical_leaf\":" << candidate.physical_leaf
        << ",\"verified\":" << candidate.verified
        << ",\"nvme_subsystem_link\":"
        << JsonString(candidate.nvme_subsystem_link.string())
        << ",\"nvme_controller_link\":"
        << JsonString(candidate.nvme_controller_link.string())
        << ",\"nvme_controller_device\":"
        << JsonString(candidate.nvme_controller_device.string())
        << ",\"nvme_block_link\":"
        << JsonString(candidate.nvme_block_link.string())
        << ",\"block_device\":" << JsonString(candidate.block_device.string())
        << ",\"error\":" << JsonString(candidate.error) << '}';
  }
  out << ']';
  return out.str();
}

double Seconds(Clock::duration elapsed) {
  return std::chrono::duration<double>(elapsed).count();
}

void EmitStdout(const std::string& line) {
  std::cout << line << '\n';
  std::cout.flush();
}

class JsonlReporter {
 public:
  explicit JsonlReporter(const fs::path& path)
      : path_(path), out_(path, std::ios::out | std::ios::app) {}

  bool good() const { return out_.is_open() && out_.good() && output_ok_; }
  const fs::path& path() const { return path_; }

  void Emit(const std::string& line) {
    EmitStdout(line);
    if (!out_.is_open()) {
      output_ok_ = false;
      return;
    }
    out_ << line << '\n';
    out_.flush();
    if (!out_.good()) output_ok_ = false;
  }

 private:
  fs::path path_;
  std::ofstream out_;
  bool output_ok_ = true;
};

bool ParseUnsigned(const char* text, uint64_t* value) {
  if (text == nullptr || *text == '\0') return false;
  const char* const end = text + std::strlen(text);
  const auto parsed = std::from_chars(text, end, *value);
  return parsed.ec == std::errc{} && parsed.ptr == end;
}

std::string ErrnoMessage(int error) {
  return std::error_code(error, std::generic_category()).message();
}

class ScopedFd {
 public:
  ScopedFd() = default;
  explicit ScopedFd(int fd) : fd_(fd) {}
  ~ScopedFd() { Reset(-1); }
  ScopedFd(const ScopedFd&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;

  int get() const { return fd_; }
  void Reset(int fd) {
    if (fd_ >= 0) ::close(fd_);
    fd_ = fd;
  }

 private:
  int fd_ = -1;
};

struct NvmeController {
  fs::path class_link;
  fs::path resolved;
};

bool SameOrAncestorPath(const fs::path& ancestor, const fs::path& descendant) {
  auto a = ancestor.begin();
  auto d = descendant.begin();
  for (; a != ancestor.end(); ++a, ++d) {
    if (d == descendant.end() || *a != *d) return false;
  }
  return true;
}

std::vector<NvmeController> NvmeControllerInventory() {
  std::vector<NvmeController> controllers;
  // The class is the primary authority.  The bus-device directory is an
  // equivalent kernel-owned controller inventory on systems exposing both.
  for (const char* root : {"/sys/class/nvme", "/sys/bus/nvme/devices"}) {
    std::error_code ec;
    fs::directory_iterator it(root, ec), end;
    if (ec) continue;
    for (; !ec && it != end; it.increment(ec)) {
      const fs::path resolved = fs::canonical(it->path(), ec);
      if (!ec) controllers.push_back({it->path(), resolved});
      ec.clear();
    }
  }
  std::sort(controllers.begin(), controllers.end(),
            [](const NvmeController& a, const NvmeController& b) {
              return a.class_link.string() < b.class_link.string();
            });
  return controllers;
}

// Validate one physical block leaf without trusting its filename.  Prefer a
// real NVMe subsystem link; otherwise require the class-block device identity
// to match a resolved NVMe controller (or its namespace/partition descendant).
bool VerifyPhysicalNvmeLeaf(const fs::path& nvme_bus,
                            const std::vector<NvmeController>& controllers,
                            BackingCandidate* candidate) {
  if (!nvme_bus.empty()) {
    for (fs::path current = candidate->resolved;; current = current.parent_path()) {
      for (const char* suffix : {"subsystem", "device/subsystem"}) {
        const fs::path link = current / suffix;
        std::error_code ec;
        const fs::file_status status = fs::symlink_status(link, ec);
        if (ec || !fs::is_symlink(status)) continue;
        const fs::path subsystem = fs::canonical(link, ec);
        if (!ec && subsystem == nvme_bus) {
          candidate->nvme_subsystem_link = link;
          candidate->verified = true;
          break;
        }
      }
      if (candidate->verified || current == "/sys" || current == "/" ||
          current.parent_path() == current) {
        break;
      }
    }
  }

  // /sys/class/block/<leaf>/device is an actual kernel link to the device whose
  // class identity is being validated, not a lookup based on a name prefix.
  const fs::path block_device_link =
      "/sys/class/block" / candidate->resolved.filename() / "device";
  std::error_code block_ec;
  const fs::path block_device = fs::canonical(block_device_link, block_ec);
  if (!block_ec) {
    candidate->nvme_block_link = block_device_link;
    candidate->block_device = block_device;
  }
  for (const auto& controller : controllers) {
    const bool class_match =
        !block_device.empty() && controller.resolved == block_device;
    const bool ancestry_match =
        SameOrAncestorPath(controller.resolved, candidate->resolved);
    if (class_match || ancestry_match) {
      candidate->nvme_controller_link = controller.class_link;
      candidate->nvme_controller_device = controller.resolved;
      candidate->verified = true;
      break;
    }
  }
  if (!candidate->verified) {
    candidate->error = "physical leaf has no NVMe subsystem link and no resolved "
                       "NVMe class/controller match";
  }
  return candidate->verified;
}

// Recursively resolve device-mapper slaves and record every root/slave/leaf.
// Mixed NVMe/non-NVMe, empty, cyclic, and unresolvable trees all fail closed.
bool ResolveBackingTree(const fs::path& source,
                        const std::vector<NvmeController>& controllers,
                        const fs::path& nvme_bus, RootEvidence* evidence,
                        std::vector<std::string>* active,
                        std::vector<std::string>* problems, unsigned depth) {
  BackingCandidate candidate;
  candidate.source = source;
  const size_t candidate_index = evidence->candidates.size();
  evidence->candidates.push_back(candidate);
  if (depth > 32) {
    evidence->candidates[candidate_index].error =
        "device-slave nesting exceeds 32 levels";
    problems->push_back(source.string() + ": nesting exceeds 32 levels");
    return false;
  }

  std::error_code ec;
  const fs::path resolved = fs::canonical(source, ec);
  if (ec) {
    evidence->candidates[candidate_index].error =
        "canonical resolution failed: " + ec.message();
    problems->push_back(source.string() + ": " + ec.message());
    return false;
  }
  evidence->candidates[candidate_index].resolved = resolved;
  if (std::find(active->begin(), active->end(), resolved.string()) !=
      active->end()) {
    evidence->candidates[candidate_index].error = "cycle in device-slave links";
    problems->push_back(resolved.string() + ": device-slave cycle");
    return false;
  }
  active->push_back(resolved.string());

  bool has_slaves = false;
  bool inspection_failed = false;
  std::vector<fs::path> slaves;
  const fs::path slaves_dir = resolved / "slaves";
  const fs::file_status slaves_status = fs::symlink_status(slaves_dir, ec);
  if (!ec && fs::is_directory(slaves_status)) {
    has_slaves = true;
    fs::directory_iterator it(slaves_dir, ec), end;
    for (; !ec && it != end; it.increment(ec)) slaves.push_back(it->path());
    if (ec) {
      inspection_failed = true;
      problems->push_back("cannot enumerate " + slaves_dir.string() + ": " +
                          ec.message());
    }
  } else if (ec && ec != std::errc::no_such_file_or_directory) {
    inspection_failed = true;
    problems->push_back("cannot inspect " + slaves_dir.string() + ": " +
                        ec.message());
  } else if (!ec && slaves_status.type() != fs::file_type::not_found &&
             !fs::is_directory(slaves_status)) {
    inspection_failed = true;
    problems->push_back(slaves_dir.string() +
                        ": slaves entry is not a real directory");
  }
  std::sort(slaves.begin(), slaves.end());

  if (inspection_failed)
    evidence->candidates[candidate_index].error =
        "device-slave inspection failed";
  if (has_slaves && !slaves.empty()) {
    bool children_ok = !inspection_failed;
    for (const fs::path& slave : slaves) {
      const bool child_ok = ResolveBackingTree(
          slave, controllers, nvme_bus, evidence, active, problems, depth + 1);
      children_ok = children_ok && child_ok;
    }
    evidence->candidates[candidate_index].verified = children_ok;
    active->pop_back();
    return children_ok;
  }
  if (inspection_failed) {
    active->pop_back();
    return false;
  }

  // Empty and absent slaves both mean this candidate itself is the physical
  // leaf.  A virtual/device-mapper leaf still fails NVMe verification below.
  evidence->candidates[candidate_index].physical_leaf = true;
  ++evidence->physical_leaf_count;
  const bool verified = VerifyPhysicalNvmeLeaf(
      nvme_bus, controllers, &evidence->candidates[candidate_index]);
  if (verified) {
    if (evidence->nvme_subsystem_link.empty() &&
        !evidence->candidates[candidate_index].nvme_subsystem_link.empty()) {
      evidence->nvme_subsystem_link =
          evidence->candidates[candidate_index].nvme_subsystem_link.string();
    }
    if (evidence->nvme_controller_link.empty() &&
        !evidence->candidates[candidate_index].nvme_controller_link.empty()) {
      evidence->nvme_controller_link =
          evidence->candidates[candidate_index].nvme_controller_link;
      evidence->nvme_controller_device =
          evidence->candidates[candidate_index].nvme_controller_device;
    }
  } else {
    problems->push_back(resolved.string() + ": " +
                        evidence->candidates[candidate_index].error);
  }
  active->pop_back();
  return verified;
}

bool ResolveNvmeBacking(dev_t device, RootEvidence* evidence,
                        std::string* error) {
  evidence->device_major = major(device);
  evidence->device_minor = minor(device);
  const std::string device_id = std::to_string(evidence->device_major) + ":" +
                                std::to_string(evidence->device_minor);
  evidence->sysfs_dev_block = "/sys/dev/block/" + device_id;
  std::error_code ec;
  const fs::path resolved = fs::canonical(evidence->sysfs_dev_block, ec);
  if (!ec) evidence->resolved_device = resolved.string();
  const fs::path nvme_bus = fs::canonical("/sys/bus/nvme", ec);
  if (!ec) evidence->nvme_subsystem = nvme_bus.string();
  const auto controllers = NvmeControllerInventory();
  std::vector<std::string> active;
  std::vector<std::string> problems;
  const bool tree_ok = ResolveBackingTree(
      evidence->sysfs_dev_block, controllers, nvme_bus, evidence, &active,
      &problems, 0);
  evidence->all_leaves_nvme = tree_ok && evidence->physical_leaf_count > 0;
  if (evidence->all_leaves_nvme) return true;
  if (problems.empty()) problems.push_back("no physical NVMe leaf was resolved");
  *error = "mounted backing device " + device_id + " resolved to " +
           evidence->resolved_device + ": ";
  for (size_t i = 0; i < problems.size(); ++i) {
    if (i != 0) *error += "; ";
    *error += problems[i];
  }
  return false;
}

// Open every supplied parent component relative to its already validated
// parent, without following symlinks.  mkdirat then creates only the final
// component in that exact parent, so no existing or aliased root is reused.
bool PrepareFreshNvmeRoot(const fs::path& supplied, RootEvidence* evidence,
                          std::string* error) {
  if (!supplied.is_absolute()) {
    *error = "benchmark root must be absolute";
    return false;
  }
  evidence->supplied_root = supplied;
  fs::path normalized = supplied.lexically_normal();
  while (normalized != normalized.root_path() && normalized.filename().empty())
    normalized = normalized.parent_path();
  if (normalized == normalized.root_path()) {
    *error = "filesystem root cannot be used as a fresh benchmark root";
    return false;
  }
  evidence->normalized_root = normalized;
  evidence->normalized_parent = normalized.parent_path();
  const std::string final_component = normalized.filename().string();
  if (final_component.empty() || final_component == "." ||
      final_component == "..") {
    *error = "benchmark root has no safe final component";
    return false;
  }

  ScopedFd parent(::open("/", O_PATH | O_DIRECTORY | O_CLOEXEC));
  if (parent.get() < 0) {
    *error = "open filesystem root for no-symlink walk: " + ErrnoMessage(errno);
    return false;
  }
  for (const fs::path& component : evidence->normalized_parent.relative_path()) {
    const std::string name = component.string();
    if (name.empty() || name == ".") continue;
    if (name == "..") {
      *error = "normalized parent unexpectedly contains ..";
      return false;
    }
    const int next = ::openat(parent.get(), name.c_str(),
                              O_PATH | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (next < 0) {
      *error = "no-symlink parent walk refused component " + name + ": " +
               ErrnoMessage(errno);
      return false;
    }
    parent.Reset(next);
  }

  struct stat existing {};
  if (::fstatat(parent.get(), final_component.c_str(), &existing,
                AT_SYMLINK_NOFOLLOW) == 0) {
    *error = "refusing existing/outdated candidate benchmark root: " +
             normalized.string();
    return false;
  }
  if (errno != ENOENT) {
    *error = "no-follow inspection of final benchmark-root component failed: " +
             ErrnoMessage(errno);
    return false;
  }

  struct stat parent_stat {};
  if (::fstat(parent.get(), &parent_stat) != 0) {
    *error = "fstat validated benchmark parent: " + ErrnoMessage(errno);
    return false;
  }
  struct statvfs filesystem {};
  if (::fstatvfs(parent.get(), &filesystem) != 0) {
    *error = "fstatvfs validated benchmark parent: " + ErrnoMessage(errno);
    return false;
  }
  const uint64_t fragment =
      filesystem.f_frsize != 0 ? filesystem.f_frsize : filesystem.f_bsize;
  evidence->available_bytes =
      fragment != 0 && filesystem.f_bavail >
                           std::numeric_limits<uint64_t>::max() / fragment
          ? std::numeric_limits<uint64_t>::max()
          : static_cast<uint64_t>(filesystem.f_bavail) * fragment;
  if (evidence->available_bytes < kRequiredFreeBytes) {
    std::ostringstream reason;
    reason << "benchmark filesystem has " << evidence->available_bytes
           << " bytes available; 400 GiB (" << kRequiredFreeBytes
           << ") is required; refusing root " << normalized;
    *error = reason.str();
    return false;
  }
  if (!ResolveNvmeBacking(parent_stat.st_dev, evidence, error)) return false;

  if (::mkdirat(parent.get(), final_component.c_str(), 0700) != 0) {
    *error = "atomic mkdirat fresh benchmark root failed: " +
             ErrnoMessage(errno);
    return false;
  }
  const ScopedFd created(
      ::openat(parent.get(), final_component.c_str(),
               O_PATH | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
  if (created.get() < 0) {
    *error = "created root failed no-follow descriptor verification: " +
             ErrnoMessage(errno);
    return false;
  }
  struct stat created_stat {};
  if (::fstat(created.get(), &created_stat) != 0) {
    *error = "fstat created benchmark root: " + ErrnoMessage(errno);
    return false;
  }
  if (created_stat.st_dev != parent_stat.st_dev) {
    *error = "created benchmark root changed backing device between preflight "
             "and mkdirat; refusing the raced mount";
    return false;
  }
  return true;
}

std::string ZeroPadding(uint32_t value) {
  std::ostringstream out;
  out << std::setw(2) << std::setfill('0') << value;
  return out.str();
}

BlockKey MakeKey(uint64_t id) {
  return BlockKey{id * 0x9e3779b97f4a7c15ULL + 0x6a09e667f3bcc909ULL,
                  id, 0};
}

void FillValue(uint64_t id, char* data, size_t size) {
  // One explicitly little-endian SplitMix64 word per eight bytes is exactly
  // reproducible from a key on every host, without a per-byte hash dependency.
  uint64_t state = id * 0x9e3779b97f4a7c15ULL;
  uint64_t word = 0;
  for (size_t i = 0; i < size; ++i) {
    if ((i & 7) == 0) {
      state += 0x9e3779b97f4a7c15ULL;
      uint64_t z = state;
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
      word = z ^ (z >> 31);
    }
    data[i] = static_cast<char>(word & 0xff);
    word >>= 8;
  }
}

class AlignedBuffer {
 public:
  explicit AlignedBuffer(size_t size) : size_(size) {
    void* memory = nullptr;
    if (::posix_memalign(&memory, kIoAlignment, size) == 0) {
      data_.reset(static_cast<char*>(memory));
      std::memset(data_.get(), 0, size);
    }
  }

  char* data() const { return data_.get(); }
  size_t size() const { return size_; }
  explicit operator bool() const { return data_ != nullptr; }

 private:
  struct Free {
    void operator()(char* data) const { std::free(data); }
  };
  std::unique_ptr<char, Free> data_;
  size_t size_;
};

// LogStore normally falls back to buffered payload I/O when O_DIRECT is
// rejected with EINVAL.  Its public test I/O injection point lets this
// acceptance test turn exactly that refusal into a startup/runtime failure,
// while leaving every successful operation on the unchanged system backend.
class StrictDirectIo final : public dfkv::lse::IoBackend {
 public:
  StrictDirectIo() : backend_(dfkv::lse::MakeSystemIoBackend()) {}

  int Open(const std::string& path, int flags, mode_t mode) override {
    const int fd = backend_->Open(path, flags, mode);
    if ((flags & O_DIRECT) != 0) {
      if (fd >= 0) {
        direct_successes_.fetch_add(1, std::memory_order_relaxed);
      } else if (errno == EINVAL) {
        direct_refusals_.fetch_add(1, std::memory_order_relaxed);
        errno = EIO;  // Bypass LogStore's EINVAL-only buffered fallback.
      }
    }
    return fd;
  }
  int Close(int fd) override { return backend_->Close(fd); }
  ssize_t Pwritev(int fd, const struct iovec* iov, int iovcnt, off_t offset,
                  bool dsync) override {
    return backend_->Pwritev(fd, iov, iovcnt, offset, dsync);
  }
  ssize_t Pread(int fd, void* buf, size_t len, off_t offset) override {
    return backend_->Pread(fd, buf, len, offset);
  }
  int Fdatasync(int fd) override { return backend_->Fdatasync(fd); }
  int Fsync(int fd) override { return backend_->Fsync(fd); }
  int Fallocate(int fd, off_t offset, off_t len) override {
    return backend_->Fallocate(fd, offset, len);
  }
  int Ftruncate(int fd, off_t len) override { return backend_->Ftruncate(fd, len); }
  int Rename(const std::string& from, const std::string& to) override {
    return backend_->Rename(from, to);
  }
  int Mkdir(const std::string& path, mode_t mode) override {
    return backend_->Mkdir(path, mode);
  }
  int Unlink(const std::string& path) override { return backend_->Unlink(path); }

  uint64_t direct_successes() const {
    return direct_successes_.load(std::memory_order_relaxed);
  }
  uint64_t direct_refusals() const {
    return direct_refusals_.load(std::memory_order_relaxed);
  }

 private:
  std::unique_ptr<dfkv::lse::IoBackend> backend_;
  std::atomic<uint64_t> direct_successes_{0};
  std::atomic<uint64_t> direct_refusals_{0};
};

bool ProbeDirectIo(const fs::path& path, std::string* error) {
  const int fd = ::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_DIRECT,
                        0600);
  if (fd < 0) {
    *error = "open(O_CREAT|O_EXCL|O_DIRECT): " + ErrnoMessage(errno);
    return false;
  }
  alignas(kIoAlignment) char block[kIoAlignment] = {};
  bool ok = true;
  if (::write(fd, block, sizeof(block)) != static_cast<ssize_t>(sizeof(block))) {
    *error = "aligned 4 KiB write: " + ErrnoMessage(errno);
    ok = false;
  } else if (::fdatasync(fd) != 0) {
    *error = "fdatasync: " + ErrnoMessage(errno);
    ok = false;
  }
  if (::close(fd) != 0 && ok) {
    *error = "close: " + ErrnoMessage(errno);
    ok = false;
  }
  return ok;  // The fresh probe file is retained with the run evidence.
}

struct RunRecord {
  uint32_t pair = 0;
  uint32_t order = 0;
  EngineKind engine = EngineKind::kSlab;
  fs::path dir;
  uint64_t capacity_bytes = 0;
  uint64_t first_key = 0;
  uint64_t planned_puts = 0;
  uint64_t attempted_puts = 0;
  uint64_t put_ok = 0;
  uint64_t cache_full = 0;
  uint64_t put_not_found = 0;
  uint64_t put_quota = 0;
  uint64_t put_io_error = 0;
  uint64_t put_invalid = 0;
  uint64_t put_resource_exhausted = 0;
  uint64_t put_no_rail = 0;
  uint32_t writers_started = 0;
  uint64_t writer_allocation_failures = 0;
  bool direct_requested = true;
  bool direct_probe_ok = false;
  bool slab_direct_active = false;
  bool fresh_directory = false;
  bool open_ok = false;
  bool healthy_start = false;
  bool healthy_before_close = false;
  bool close_ok = false;
  bool slab_settled = true;
  double write_seconds = 0;
  double settle_seconds = 0;
  double readback_seconds = 0;
  double close_seconds = 0;
  std::string startup_error;
  std::string thread_error;

  uint64_t enumerated_keys = 0;
  uint64_t duplicate_enumerated_keys = 0;
  uint64_t unexpected_enumerated_keys = 0;
  uint64_t key_count_at_end = 0;
  uint64_t lookup_ok = 0;
  uint64_t lookup_not_found = 0;
  uint64_t lookup_io_error = 0;
  uint64_t lookup_other_error = 0;
  uint64_t lookup_length_mismatch = 0;
  uint64_t resident_value_bytes = 0;
  uint64_t read_ok = 0;
  uint64_t read_not_found = 0;
  uint64_t read_io_error = 0;
  uint64_t read_other_error = 0;
  uint64_t read_length_mismatch = 0;
  uint64_t read_value_length_mismatch = 0;
  uint64_t payload_mismatch = 0;
  uint64_t used_bytes = 0;
  uint64_t evictions = 0;

  uint64_t slab_put_no_slot = 0;
  uint64_t slab_put_busy = 0;
  uint64_t slab_dio_write_fallbacks = 0;
  uint64_t slab_payload_crc_failures = 0;
  uint64_t slab_metadata_io_errors = 0;
  uint64_t slab_balance_moves = 0;
  uint64_t slab_watermark_high = 0;
  uint64_t slab_watermark_low = 0;
  uint64_t slab_watermark_capacity = 0;
  uint64_t slab_reclaim_interval_ms = 0;
  uint64_t log_segment_bytes = 0;
  uint64_t log_segment_count = 0;
  uint64_t log_open_segments = 0;
  uint64_t log_stat_cache_full = 0;
  uint64_t log_records_relocated = 0;
  uint64_t log_relocated_bytes = 0;
  uint64_t log_read_crc_failures = 0;
  uint64_t log_direct_open_successes = 0;
  uint64_t log_direct_open_refusals = 0;

  bool valid = false;
  std::vector<std::string> violations;

  uint64_t UnexpectedPuts() const {
    return put_not_found + put_quota + put_io_error + put_invalid +
           put_resource_exhausted + put_no_rail;
  }
  double Fraction() const;
  double CacheFullRatio() const {
    return attempted_puts == 0
               ? 1.0
               : static_cast<double>(cache_full) /
                     static_cast<double>(attempted_puts);
  }
};

double RunRecord::Fraction() const {
  return static_cast<double>(resident_value_bytes) /
         static_cast<double>(capacity_bytes);
}

void Violate(RunRecord* run, const std::string& reason) {
  run->violations.push_back(reason);
}

struct PutCounters {
  std::atomic<uint64_t> allocation_failures{0};
  std::atomic<uint64_t> ok{0};
  std::atomic<uint64_t> cache_full{0};
  std::atomic<uint64_t> not_found{0};
  std::atomic<uint64_t> quota{0};
  std::atomic<uint64_t> io_error{0};
  std::atomic<uint64_t> invalid{0};
  std::atomic<uint64_t> resource_exhausted{0};
  std::atomic<uint64_t> no_rail{0};

  void Add(Status status) {
    switch (status) {
      case Status::kOk: ok.fetch_add(1, std::memory_order_relaxed); break;
      case Status::kCacheFull:
        cache_full.fetch_add(1, std::memory_order_relaxed); break;
      case Status::kNotFound:
        not_found.fetch_add(1, std::memory_order_relaxed); break;
      case Status::kQuotaExceeded:
        quota.fetch_add(1, std::memory_order_relaxed); break;
      case Status::kIOError:
        io_error.fetch_add(1, std::memory_order_relaxed); break;
      case Status::kInvalid:
        invalid.fetch_add(1, std::memory_order_relaxed); break;
      case Status::kResourceExhausted:
        resource_exhausted.fetch_add(1, std::memory_order_relaxed); break;
      case Status::kNoCompatibleRail:
        no_rail.fetch_add(1, std::memory_order_relaxed); break;
    }
  }
};

bool KeyOrder(const BlockKey& left, const BlockKey& right) {
  if (left.digest_lo != right.digest_lo) return left.digest_lo < right.digest_lo;
  if (left.digest_hi != right.digest_hi) return left.digest_hi < right.digest_hi;
  return left.tenant_hash < right.tenant_hash;
}

void EnumerateAndReadBack(StoreEngine* store, RunRecord* run) {
  std::vector<BlockKey> keys;
  store->ForEachKey([&](const BlockKey& key) { keys.push_back(key); });
  run->enumerated_keys = keys.size();
  std::sort(keys.begin(), keys.end(), KeyOrder);
  for (size_t i = 1; i < keys.size(); ++i) {
    if (keys[i] == keys[i - 1]) ++run->duplicate_enumerated_keys;
  }
  std::string expected(kValueBytes, '\0');

  for (const BlockKey& key : keys) {
    if (key.digest_lo < run->first_key ||
        key.digest_lo >= run->first_key + run->planned_puts ||
        !(key == MakeKey(key.digest_lo))) {
      ++run->unexpected_enumerated_keys;
    }
    ValueMetadata metadata;
    const Status lookup = store->Lookup(key, &metadata);
    if (lookup == Status::kOk) {
      ++run->lookup_ok;
      run->resident_value_bytes += metadata.value_len;
      if (metadata.value_len != kValueBytes) ++run->lookup_length_mismatch;
    } else if (lookup == Status::kNotFound) {
      ++run->lookup_not_found;
    } else if (lookup == Status::kIOError) {
      ++run->lookup_io_error;
    } else {
      ++run->lookup_other_error;
    }

    std::string payload;
    size_t value_len = 0;
    const Status read = store->Range(key, 0, 0, &payload, &value_len);
    if (read == Status::kOk) {
      ++run->read_ok;
      if (payload.size() != kValueBytes) ++run->read_length_mismatch;
      if (value_len != kValueBytes) ++run->read_value_length_mismatch;
      FillValue(key.digest_lo, expected.data(), expected.size());
      if (payload != expected) ++run->payload_mismatch;
    } else if (read == Status::kNotFound) {
      ++run->read_not_found;
    } else if (read == Status::kIOError) {
      ++run->read_io_error;
    } else {
      ++run->read_other_error;
    }
  }
}

RunRecord RunOne(EngineKind kind, uint32_t pair, uint32_t order,
                 const Config& config, bool direct_probe_ok) {
  RunRecord run;
  run.pair = pair;
  run.order = order;
  run.engine = kind;
  run.dir = config.root / ("pair_" + ZeroPadding(pair)) / EngineName(kind);
  run.capacity_bytes = config.capacity;
  run.first_key = static_cast<uint64_t>(pair) * config.puts_per_run;
  run.planned_puts = config.puts_per_run;
  run.direct_probe_ok = direct_probe_ok;

  std::error_code ec;
  run.fresh_directory = fs::create_directory(run.dir, ec) && !ec;
  if (!run.fresh_directory) {
    Violate(&run, "fresh per-engine directory creation refused/failed: " +
                      (ec ? ec.message() : "path already exists"));
    return run;
  }

  std::unique_ptr<StoreEngine> store;
  DiskSlabStore* slab = nullptr;
  LogStore* log = nullptr;
  std::shared_ptr<StrictDirectIo> strict_io;
  if (kind == EngineKind::kSlab) {
    DiskSlabStore::Options options;
    options.dir = run.dir.string();
    options.capacity_bytes = config.capacity;
    options.extent_bytes = kSlabExtent;
    options.slot_granularity = kSlabGranularity;
    options.direct_writes = true;
    options.balance = false;
    options.warm_restart = DiskSlabStore::Options::WarmRestart::kOff;
    bool ok = false;
    auto opened = std::make_unique<DiskSlabStore>(options, &ok);
    slab = opened.get();
    store = std::move(opened);
    run.open_ok = ok;
    run.slab_direct_active = slab->DirectWritesActive();
    const auto watermark = slab->GetWatermarkObservation();
    run.slab_watermark_high = watermark.watermark_high_bytes;
    run.slab_watermark_low = watermark.watermark_low_bytes;
    run.slab_watermark_capacity = watermark.capacity_bytes;
    run.slab_reclaim_interval_ms = watermark.reclaim_interval_ms;
  } else {
    LogStore::Options options;
    options.dir = run.dir.string();
    options.capacity_bytes = config.capacity;
    options.direct_writes = true;
    options.segment_bytes = kLogSegment;
    options.open_segments = 0;
    options.relocate_pct = 25;
    options.sync_ms = 1000;
    options.cleaner = true;
    options.preopen = true;
    options.retain_enabled = false;
    strict_io = std::make_shared<StrictDirectIo>();
    options.io = strict_io;
    bool ok = false;
    auto opened = std::make_unique<LogStore>(options, &ok);
    log = opened.get();
    store = std::move(opened);
    run.open_ok = ok;
    LogStore::Geometry geometry;
    std::string geometry_error;
    if (!LogStore::GeometryFor(config.capacity, kLogSegment, &geometry,
                               &geometry_error)) {
      Violate(&run, "log geometry rejected: " + geometry_error);
    } else {
      run.log_segment_bytes = geometry.segment_bytes;
      const auto stats = log->GetStats();
      run.log_segment_count = stats.segment_count;
      run.log_open_segments = stats.open_segments;
      const uint64_t expected_open =
          std::max<uint64_t>(1, std::min<uint64_t>(4, geometry.segment_count / 32));
      if (geometry.segment_bytes != kLogSegment ||
          geometry.segment_count != config.capacity / kLogSegment ||
          stats.segment_count != geometry.segment_count ||
          stats.open_segments != expected_open) {
        Violate(&run, "resolved log geometry/open_segments differs from request");
      }
    }
  }
  if (strict_io) {
    run.log_direct_open_successes = strict_io->direct_successes();
    run.log_direct_open_refusals = strict_io->direct_refusals();
  }
  run.startup_error = store->StartupError();
  run.healthy_start = store->Healthy();
  if (!run.open_ok || !run.healthy_start) {
    Violate(&run, "engine did not open healthy: " + run.startup_error);
    return run;
  }
  if (kind == EngineKind::kSlab) {
    const uint64_t expected_high = config.capacity / 100 * 92;
    const uint64_t expected_low = config.capacity / 100 * 88;
    if (!run.slab_direct_active)
      Violate(&run, "slab direct I/O was requested but is not active");
    if (run.slab_watermark_high != expected_high ||
        run.slab_watermark_low != expected_low ||
        run.slab_watermark_capacity != config.capacity ||
        run.slab_reclaim_interval_ms != 50) {
      Violate(&run, "slab does not expose the required 92/88 percentage watermark");
    }
  } else if (run.log_direct_open_successes == 0 ||
             run.log_direct_open_refusals != 0) {
    Violate(&run, "log direct I/O opens failed or were refused");
  }
  if (!run.violations.empty()) {
    // Never replace a required configuration with a buffered/different-geometry
    // measurement, even one explicitly labelled as a failed acceptance run.
    const auto close_start = Clock::now();
    run.healthy_before_close = store->Healthy();
    run.key_count_at_end = store->Count();
    run.used_bytes = store->UsedBytes();
    run.close_ok = store->Close();
    run.close_seconds = Seconds(Clock::now() - close_start);
    return run;
  }

  const auto write_start = Clock::now();
  constexpr size_t kIoCapacity = (kValueBytes + kIoAlignment - 1) /
                                 kIoAlignment * kIoAlignment;
  std::atomic<uint64_t> next{0};
  std::atomic<bool> stop{false};
  PutCounters counters;
  std::vector<std::thread> writers;
  try {
    writers.reserve(kWriters);
    for (uint32_t writer = 0; writer < kWriters; ++writer) {
      writers.emplace_back([&] {
        AlignedBuffer buffer(kIoCapacity);
        if (!buffer) {
          counters.allocation_failures.fetch_add(1, std::memory_order_relaxed);
          stop.store(true, std::memory_order_relaxed);
          return;
        }
        while (!stop.load(std::memory_order_relaxed)) {
          const uint64_t sequence = next.fetch_add(1, std::memory_order_relaxed);
          if (sequence >= config.puts_per_run) break;
          const uint64_t id = run.first_key + sequence;
          FillValue(id, buffer.data(), kValueBytes);
          const Status status = store->CacheDirect(
              MakeKey(id), buffer.data(), kValueBytes, buffer.size());
          counters.Add(status);
          if (status != Status::kOk && status != Status::kCacheFull)
            stop.store(true, std::memory_order_relaxed);
        }
      });
    }
  } catch (const std::exception& error) {
    run.thread_error = error.what();
    stop.store(true, std::memory_order_relaxed);
  } catch (...) {
    run.thread_error = "unknown std::thread creation failure";
    stop.store(true, std::memory_order_relaxed);
  }
  run.writers_started = writers.size();
  for (auto& writer : writers) writer.join();
  run.write_seconds = Seconds(Clock::now() - write_start);
  run.writer_allocation_failures =
      counters.allocation_failures.load(std::memory_order_relaxed);
  run.put_ok = counters.ok.load(std::memory_order_relaxed);
  run.cache_full = counters.cache_full.load(std::memory_order_relaxed);
  run.put_not_found = counters.not_found.load(std::memory_order_relaxed);
  run.put_quota = counters.quota.load(std::memory_order_relaxed);
  run.put_io_error = counters.io_error.load(std::memory_order_relaxed);
  run.put_invalid = counters.invalid.load(std::memory_order_relaxed);
  run.put_resource_exhausted =
      counters.resource_exhausted.load(std::memory_order_relaxed);
  run.put_no_rail = counters.no_rail.load(std::memory_order_relaxed);
  run.attempted_puts = run.put_ok + run.cache_full + run.UnexpectedPuts();

  if (run.writers_started != kWriters || !run.thread_error.empty() ||
      run.writer_allocation_failures != 0) {
    Violate(&run, "could not run exactly eight writers: " + run.thread_error);
  }
  if (run.attempted_puts != run.planned_puts) {
    Violate(&run, "actual PUT attempts differ from ceil(2*capacity/value)");
  }
  if (run.UnexpectedPuts() != 0) {
    Violate(&run, "PUT returned a status other than Ok or CacheFull");
  }

  // Do not close/reopen before enumeration: in particular slab's kOff warm
  // restart would discard exactly the resident set this test must measure.
  if (run.attempted_puts == run.planned_puts && run.UnexpectedPuts() == 0) {
    const auto settle_start = Clock::now();
    if (kind == EngineKind::kSlab) {
      const auto deadline = settle_start + std::chrono::seconds(30);
      for (;;) {
        const auto observation = slab->GetWatermarkObservation();
        if (observation.watermark_active == 0 &&
            slab->UsedBytes() <= observation.watermark_high_bytes) {
          break;
        }
        if (Clock::now() >= deadline) {
          run.slab_settled = false;
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      if (!run.slab_settled)
        Violate(&run, "slab watermark did not settle within 30 seconds");
    }
    run.settle_seconds = Seconds(Clock::now() - settle_start);
    const auto read_start = Clock::now();
    EnumerateAndReadBack(store.get(), &run);
    run.readback_seconds = Seconds(Clock::now() - read_start);
  }

  run.healthy_before_close = store->Healthy();
  run.key_count_at_end = store->Count();
  run.used_bytes = store->UsedBytes();
  run.evictions = store->Evictions();
  if (kind == EngineKind::kSlab) {
    const auto stats = slab->GetStats();
    run.slab_put_no_slot = stats.put_no_slot;
    run.slab_put_busy = stats.put_busy;
    run.slab_dio_write_fallbacks = stats.dio_write_fallbacks;
    run.slab_payload_crc_failures = stats.payload_crc_failures;
    run.slab_metadata_io_errors = stats.metadata_io_errors;
    run.slab_balance_moves = stats.balance_moves;
  } else {
    const auto stats = log->GetStats();
    run.log_stat_cache_full = stats.cache_full;
    run.log_records_relocated = stats.records_relocated;
    run.log_relocated_bytes = stats.relocated_bytes;
    run.log_read_crc_failures = stats.read_crc_failures;
    run.log_direct_open_successes = strict_io->direct_successes();
    run.log_direct_open_refusals = strict_io->direct_refusals();
  }

  const auto close_start = Clock::now();
  run.close_ok = store->Close();
  run.close_seconds = Seconds(Clock::now() - close_start);
  store.reset();

  if (run.CacheFullRatio() >= 0.001)
    Violate(&run, "raw CacheFull/PUT-attempt ratio is not below 0.1%");
  if (run.enumerated_keys != run.lookup_ok || run.lookup_not_found != 0 ||
      run.lookup_io_error != 0 || run.lookup_other_error != 0 ||
      run.lookup_length_mismatch != 0 || run.duplicate_enumerated_keys != 0 ||
      run.unexpected_enumerated_keys != 0) {
    Violate(&run, "resident enumeration/per-key Lookup is incomplete or inconsistent");
  }
  if (run.enumerated_keys != run.read_ok || run.read_not_found != 0 ||
      run.read_io_error != 0 || run.read_other_error != 0 ||
      run.read_length_mismatch != 0 || run.read_value_length_mismatch != 0 ||
      run.payload_mismatch != 0) {
    Violate(&run, "not every enumerated resident key read back its exact payload");
  }
  if (!run.healthy_before_close || !run.close_ok)
    Violate(&run, "engine health or clean Close failed");
  if (run.used_bytes > config.capacity)
    Violate(&run, "UsedBytes exceeds capacity");
  if (run.resident_value_bytes > config.capacity)
    Violate(&run, "sum of resident value_len exceeds capacity");
  if (kind == EngineKind::kSlab) {
    if (run.slab_dio_write_fallbacks != 0)
      Violate(&run, "slab reports buffered fallbacks for aligned direct PUTs");
    if (run.slab_payload_crc_failures != 0 ||
        run.slab_metadata_io_errors != 0 || run.slab_balance_moves != 0) {
      Violate(&run, "slab CRC/metadata/balance invariants failed");
    }
  } else {
    if (run.log_read_crc_failures != 0)
      Violate(&run, "log reports payload CRC failures");
    if (run.log_direct_open_successes == 0 ||
        run.log_direct_open_refusals != 0) {
      Violate(&run, "log direct I/O opens failed or were refused during the run");
    }
  }
  run.valid = run.violations.empty();
  return run;
}

std::string RunJson(const RunRecord& run, const Config& config) {
  std::ostringstream out;
  out << std::setprecision(17) << std::boolalpha;
  out << "{\"record\":\"run\",\"schema_version\":1"
      << ",\"pair\":" << run.pair
      << ",\"engine_order\":" << run.order
      << ",\"engine\":" << JsonString(EngineName(run.engine))
      << ",\"directory\":" << JsonString(run.dir.string())
      << ",\"value_bytes\":" << kValueBytes
      << ",\"capacity_bytes\":" << run.capacity_bytes
      << ",\"first_key_id\":" << run.first_key
      << ",\"planned_puts\":" << run.planned_puts
      << ",\"attempted_puts\":" << run.attempted_puts
      << ",\"put_ok\":" << run.put_ok
      << ",\"cache_full_numerator\":" << run.cache_full
      << ",\"cache_full_denominator\":" << run.attempted_puts
      << ",\"raw_cache_full_ratio\":" << run.CacheFullRatio()
      << ",\"put_not_found\":" << run.put_not_found
      << ",\"put_quota_exceeded\":" << run.put_quota
      << ",\"put_io_error\":" << run.put_io_error
      << ",\"put_invalid\":" << run.put_invalid
      << ",\"put_resource_exhausted\":" << run.put_resource_exhausted
      << ",\"put_no_compatible_rail\":" << run.put_no_rail
      << ",\"writers_started\":" << run.writers_started
      << ",\"writer_allocation_failures\":" << run.writer_allocation_failures
      << ",\"direct_writes_requested\":" << run.direct_requested
      << ",\"direct_io_probe_ok\":" << run.direct_probe_ok
      << ",\"slab_direct_writes_active\":" << run.slab_direct_active
      << ",\"fresh_directory\":" << run.fresh_directory
      << ",\"open_ok\":" << run.open_ok
      << ",\"healthy_start\":" << run.healthy_start
      << ",\"healthy_before_close\":" << run.healthy_before_close
      << ",\"close_ok\":" << run.close_ok
      << ",\"slab_settled\":" << run.slab_settled
      << ",\"write_seconds\":" << run.write_seconds
      << ",\"settle_seconds\":" << run.settle_seconds
      << ",\"readback_seconds\":" << run.readback_seconds
      << ",\"close_seconds\":" << run.close_seconds
      << ",\"startup_error\":" << JsonString(run.startup_error)
      << ",\"thread_error\":" << JsonString(run.thread_error)
      << ",\"enumerated_keys\":" << run.enumerated_keys
      << ",\"duplicate_enumerated_keys\":" << run.duplicate_enumerated_keys
      << ",\"unexpected_enumerated_keys\":" << run.unexpected_enumerated_keys
      << ",\"count_at_end\":" << run.key_count_at_end
      << ",\"lookup_ok\":" << run.lookup_ok
      << ",\"lookup_not_found\":" << run.lookup_not_found
      << ",\"lookup_io_error\":" << run.lookup_io_error
      << ",\"lookup_other_error\":" << run.lookup_other_error
      << ",\"lookup_value_len_mismatch\":" << run.lookup_length_mismatch
      << ",\"resident_value_bytes\":" << run.resident_value_bytes
      << ",\"resident_fraction\":" << run.Fraction()
      << ",\"read_ok\":" << run.read_ok
      << ",\"read_not_found\":" << run.read_not_found
      << ",\"read_io_error\":" << run.read_io_error
      << ",\"read_other_error\":" << run.read_other_error
      << ",\"read_length_mismatch\":" << run.read_length_mismatch
      << ",\"read_value_len_mismatch\":" << run.read_value_length_mismatch
      << ",\"payload_mismatch\":" << run.payload_mismatch
      << ",\"used_bytes\":" << run.used_bytes
      << ",\"used_within_capacity\":" << (run.used_bytes <= config.capacity)
      << ",\"evictions\":" << run.evictions
      << ",\"slab_put_no_slot\":" << run.slab_put_no_slot
      << ",\"slab_put_busy\":" << run.slab_put_busy
      << ",\"slab_dio_write_fallbacks\":" << run.slab_dio_write_fallbacks
      << ",\"slab_payload_crc_failures\":" << run.slab_payload_crc_failures
      << ",\"slab_metadata_io_errors\":" << run.slab_metadata_io_errors
      << ",\"slab_balance_configured\":false"
      << ",\"slab_balance_moves\":" << run.slab_balance_moves
      << ",\"slab_watermark_high_bytes\":" << run.slab_watermark_high
      << ",\"slab_watermark_low_bytes\":" << run.slab_watermark_low
      << ",\"slab_watermark_capacity_bytes\":" << run.slab_watermark_capacity
      << ",\"slab_reclaim_interval_ms\":" << run.slab_reclaim_interval_ms
      << ",\"log_segment_bytes\":" << run.log_segment_bytes
      << ",\"log_segment_count\":" << run.log_segment_count
      << ",\"log_open_segments\":" << run.log_open_segments
      << ",\"log_open_segments_from_zero\":true"
      << ",\"log_relocate_pct\":25,\"log_sync_ms\":1000"
      << ",\"log_cleaner\":true,\"log_preopen\":true"
      << ",\"log_retain_enabled\":false"
      << ",\"log_stat_cache_full\":" << run.log_stat_cache_full
      << ",\"log_records_relocated\":" << run.log_records_relocated
      << ",\"log_relocated_bytes\":" << run.log_relocated_bytes
      << ",\"log_read_crc_failures\":" << run.log_read_crc_failures
      << ",\"log_direct_open_successes\":" << run.log_direct_open_successes
      << ",\"log_direct_open_refusals\":" << run.log_direct_open_refusals
      << ",\"result\":" << JsonString(run.valid ? "VALID" : "FAIL")
      << ",\"violations\":" << JsonStrings(run.violations) << '}';
  return out.str();
}

const RunRecord* FindRun(const std::vector<RunRecord>& runs, EngineKind engine,
                         uint32_t pair) {
  for (const auto& run : runs) {
    if (run.engine == engine && run.pair == pair) return &run;
  }
  return nullptr;
}

double Spread(const std::vector<double>& values) {
  if (values.empty()) return 0;
  const auto bounds = std::minmax_element(values.begin(), values.end());
  return *bounds.second - *bounds.first;
}

void PreflightFailure(const std::string& reason,
                      const RootEvidence* evidence = nullptr) {
  std::ostringstream line;
  line << std::boolalpha
       << "{\"record\":\"preflight\",\"schema_version\":1"
       << ",\"result\":\"FAIL\",\"reason\":" << JsonString(reason);
  if (evidence != nullptr && !evidence->candidates.empty()) {
    line << ",\"backing_device\":\"" << evidence->device_major << ':'
         << evidence->device_minor << '"'
         << ",\"sysfs_dev_block\":" << JsonString(evidence->sysfs_dev_block)
         << ",\"resolved_device\":" << JsonString(evidence->resolved_device)
         << ",\"backing_physical_leaf_count\":" << evidence->physical_leaf_count
         << ",\"backing_all_leaves_nvme\":" << evidence->all_leaves_nvme
         << ",\"backing_candidates\":" << BackingCandidatesJson(*evidence);
  }
  line << '}';
  EmitStdout(line.str());
  FAIL() << reason;
}

constexpr uint32_t kRecoveryKeysPerBatch = 16;
constexpr size_t kRecoveryIoCapacity =
    (kValueBytes + kIoAlignment - 1) / kIoAlignment * kIoAlignment;
constexpr char kRecoveryJsonlName[] = "lse_m3_recovery.jsonl";
constexpr uint32_t kRecoveryReportMagic = 0x44464b56;  // "DFKV"
constexpr int kRecoveryReadyTimeoutMs = 120000;

struct RecoveryOptions {
  EngineKind kind = EngineKind::kSlab;
  DiskSlabStore::Options slab;
  LogStore::Options log;
  std::shared_ptr<StrictDirectIo> log_io;
};

struct RecoveryOpened {
  std::unique_ptr<StoreEngine> store;
  DiskSlabStore* slab = nullptr;
  LogStore* log = nullptr;
};

RecoveryOptions MakeRecoveryOptions(EngineKind kind, const fs::path& dir) {
  RecoveryOptions options;
  options.kind = kind;
  if (kind == EngineKind::kSlab) {
    options.slab.dir = dir.string();
    options.slab.capacity_bytes = kStandardCapacity;
    options.slab.extent_bytes = kSlabExtent;
    options.slab.slot_granularity = kSlabGranularity;
    options.slab.direct_writes = true;
    options.slab.balance = false;
    options.slab.table_sync_ms = 0;
    options.slab.reclaim_interval_ms = 0;
    options.slab.warm_restart = DiskSlabStore::Options::WarmRestart::kAuto;
    // Deliberately use the constructor's actual boot/mount identity, the real
    // /dev/shm token location, and the default local-filesystem requirement.
    options.slab.require_local_filesystem = true;
  } else {
    options.log.dir = dir.string();
    options.log.capacity_bytes = kStandardCapacity;
    options.log.direct_writes = true;
    options.log.segment_bytes = kLogSegment;
    options.log.open_segments = 0;
    options.log.relocate_pct = 25;
    options.log.sync_ms = 1000;
    options.log.cleaner = true;
    options.log.preopen = true;
    options.log.retain_enabled = false;
    options.log_io = std::make_shared<StrictDirectIo>();
    options.log.io = options.log_io;
  }
  return options;
}

RecoveryOpened OpenRecoveryEngine(const RecoveryOptions& options, bool* ok) {
  RecoveryOpened opened;
  bool open_ok = false;
  if (options.kind == EngineKind::kSlab) {
    auto store = std::make_unique<DiskSlabStore>(options.slab, &open_ok);
    opened.slab = store.get();
    opened.store = std::move(store);
  } else {
    auto store = std::make_unique<LogStore>(options.log, &open_ok);
    opened.log = store.get();
    opened.store = std::move(store);
  }
  *ok = open_ok && opened.store->Healthy();
  return opened;
}

std::string RecoveryDirectError(const RecoveryOptions& options,
                                const RecoveryOpened& opened) {
  if (options.kind == EngineKind::kSlab) {
    if (!opened.slab->DirectWritesActive()) return "slab O_DIRECT is not active";
    if (opened.slab->GetStats().dio_write_fallbacks != 0)
      return "slab unexpectedly used a buffered payload-write fallback";
  } else if (options.log_io->direct_successes() == 0 ||
             options.log_io->direct_refusals() != 0) {
    return "log O_DIRECT open was not observed or was refused";
  }
  return {};
}

bool RecoveryWriteBatch(StoreEngine* store, uint64_t first_key,
                        uint32_t* completed, std::string* error) {
  AlignedBuffer value(kRecoveryIoCapacity);
  if (!value) {
    *error = "cannot allocate the aligned PUT buffer";
    return false;
  }
  *completed = 0;
  for (uint64_t id = first_key; id < first_key + kRecoveryKeysPerBatch; ++id) {
    FillValue(id, value.data(), kValueBytes);
    const Status status =
        store->CacheDirect(MakeKey(id), value.data(), kValueBytes, value.size());
    if (status != Status::kOk) {
      *error = "PUT key " + std::to_string(id) + " returned status " +
               std::to_string(static_cast<int>(status));
      return false;
    }
    ++*completed;
  }
  return true;
}

bool RecoveryVerifyValues(StoreEngine* store, uint64_t first_key,
                          uint64_t last_key_exclusive, std::string* error) {
  if (!store->Healthy()) {
    *error = "store is unhealthy: " + store->StartupError();
    return false;
  }
  const size_t expected_count = last_key_exclusive - first_key;
  if (store->Count() != expected_count) {
    *error = "Count is " + std::to_string(store->Count()) + ", expected " +
             std::to_string(expected_count);
    return false;
  }
  std::vector<BlockKey> keys;
  store->ForEachKey([&](const BlockKey& key) { keys.push_back(key); });
  if (keys.size() != expected_count) {
    *error = "ForEachKey returned " + std::to_string(keys.size()) +
             " keys, expected " + std::to_string(expected_count);
    return false;
  }
  std::vector<uint32_t> seen(expected_count, 0);
  for (const BlockKey& key : keys) {
    const uint64_t id = key.digest_lo;
    if (id < first_key || id >= last_key_exclusive || !(key == MakeKey(id))) {
      *error = "ForEachKey returned an unexpected key (digest_lo=" +
               std::to_string(id) + ")";
      return false;
    }
    ++seen[id - first_key];
  }
  for (size_t i = 0; i < seen.size(); ++i) {
    if (seen[i] != 1) {
      *error = "key " + std::to_string(first_key + i) + " was enumerated " +
               std::to_string(seen[i]) + " times";
      return false;
    }
  }

  AlignedBuffer read_buffer(kRecoveryIoCapacity);
  if (!read_buffer) {
    *error = "cannot allocate the aligned read buffer";
    return false;
  }
  std::string expected(kValueBytes, '\0');
  for (uint64_t id = first_key; id < last_key_exclusive; ++id) {
    const BlockKey key = MakeKey(id);
    ValueMetadata metadata;
    const Status lookup = store->Lookup(key, &metadata);
    if (lookup != Status::kOk || metadata.value_len != kValueBytes) {
      *error = "Lookup key " + std::to_string(id) + " returned status " +
               std::to_string(static_cast<int>(lookup)) + " and value_len " +
               std::to_string(metadata.value_len);
      return false;
    }
    const char* data = nullptr;
    size_t out_len = 0;
    size_t value_len = 0;
    const Status read = store->RangeDirect(
        key, 0, 0, read_buffer.data(), read_buffer.size(), &data, &out_len,
        &value_len);
    if (read != Status::kOk || out_len != kValueBytes ||
        value_len != kValueBytes) {
      *error = "RangeDirect key " + std::to_string(id) + " returned status " +
               std::to_string(static_cast<int>(read)) + ", out_len " +
               std::to_string(out_len) + ", value_len " +
               std::to_string(value_len);
      return false;
    }
    FillValue(id, expected.data(), expected.size());
    if (std::string(data, out_len) != expected) {
      *error = "full payload mismatch for key " + std::to_string(id);
      return false;
    }
  }
  return true;
}

enum RecoveryChildStep : uint32_t {
  kRecoveryChildOpen = 1,
  kRecoveryChildDirect = 2,
  kRecoveryChildPut = 3,
  kRecoveryChildCount = 4,
  kRecoveryChildSync = 5,
  kRecoveryChildReady = 6,
};

struct RecoveryChildReport {
  uint32_t magic = kRecoveryReportMagic;
  uint32_t ready = 0;
  uint32_t step = 0;
  uint32_t writes_completed = 0;
  int32_t status_code = 0;
  int32_t saved_errno = 0;
  char detail[192] = {};
};

static_assert(sizeof(RecoveryChildReport) <= 512,
              "the child report must fit an atomic bounded pipe write");

[[noreturn]] void RecoveryChildExit(int code) {
  ::syscall(SYS_exit_group, code);
  __builtin_unreachable();
}

[[noreturn]] void RecoveryChildFail(int fd, RecoveryChildStep step,
                                    int status_code, int saved_errno,
                                    uint32_t writes_completed,
                                    const std::string& detail) {
  RecoveryChildReport report;
  report.step = step;
  report.writes_completed = writes_completed;
  report.status_code = status_code;
  report.saved_errno = saved_errno;
  std::snprintf(report.detail, sizeof(report.detail), "%s", detail.c_str());
  ssize_t written;
  do {
    written = ::write(fd, &report, sizeof(report));
  } while (written < 0 && errno == EINTR);
  RecoveryChildExit(written == sizeof(report) ? 0 : 90);
}

[[noreturn]] void RecoveryChildMain(const RecoveryOptions& options, int fd) {
  bool ok = false;
  RecoveryOpened opened = OpenRecoveryEngine(options, &ok);
  if (!ok) {
    RecoveryChildFail(fd, kRecoveryChildOpen, 0, errno, 0,
                      opened.store->StartupError());
  }
  const std::string direct_error = RecoveryDirectError(options, opened);
  if (!direct_error.empty()) {
    RecoveryChildFail(fd, kRecoveryChildDirect, 0, errno, 0, direct_error);
  }
  if (opened.store->Count() != kRecoveryKeysPerBatch) {
    RecoveryChildFail(fd, kRecoveryChildCount, 0, 0, 0,
                      "clean dataset count before crash writes is " +
                          std::to_string(opened.store->Count()));
  }
  uint32_t completed = 0;
  std::string error;
  if (!RecoveryWriteBatch(opened.store.get(), kRecoveryKeysPerBatch, &completed,
                          &error)) {
    RecoveryChildFail(fd, kRecoveryChildPut, 0, errno, completed, error);
  }
  if (!opened.store->Healthy() ||
      opened.store->Count() != 2 * kRecoveryKeysPerBatch) {
    RecoveryChildFail(fd, kRecoveryChildCount, 0, 0, completed,
                      "post-PUT health/count check failed: count=" +
                          std::to_string(opened.store->Count()) + ", error=" +
                          opened.store->StartupError());
  }
  const bool synced = options.kind == EngineKind::kSlab
                          ? dfkv::DiskSlabStoreTestPeer::SyncPayloadsAndTable(
                                opened.slab)
                          : dfkv::LogStoreTestPeer::SyncDirtySegments(opened.log);
  if (!synced) {
    RecoveryChildFail(fd, kRecoveryChildSync, 0, errno, completed,
                      ErrnoMessage(errno));
  }

  RecoveryChildReport report;
  report.ready = 1;
  report.step = kRecoveryChildReady;
  report.writes_completed = completed;
  ssize_t written;
  do {
    written = ::write(fd, &report, sizeof(report));
  } while (written < 0 && errno == EINTR);
  if (written != sizeof(report)) RecoveryChildExit(90);
  for (;;) ::pause();
}

class RecoveryChildGuard {
 public:
  explicit RecoveryChildGuard(pid_t pid) : pid_(pid) {}
  RecoveryChildGuard(const RecoveryChildGuard&) = delete;
  RecoveryChildGuard& operator=(const RecoveryChildGuard&) = delete;
  ~RecoveryChildGuard() {
    if (pid_ <= 0) return;
    ::kill(pid_, SIGKILL);
    int status = 0;
    // A kernel-stuck child cannot be forced to reap.  Bound cleanup as well,
    // and leave its ownership marked unresolved rather than claiming success.
    (void)WaitUntil(pid_, Clock::now() + std::chrono::seconds(30), &status);
  }

  bool Reap(int* status) {
    if (pid_ <= 0) {
      errno = ECHILD;
      return false;
    }
    const bool reaped =
        WaitUntil(pid_, Clock::now() + std::chrono::seconds(30), status);
    if (reaped) pid_ = -1;
    return reaped;
  }

 private:
  static bool WaitUntil(pid_t pid, Clock::time_point deadline, int* status) {
    for (;;) {
      const pid_t result = ::waitpid(pid, status, WNOHANG);
      if (result == pid) return true;
      if (result < 0) {
        if (errno != EINTR || Clock::now() >= deadline) return false;
        continue;
      }
      const auto remaining =
          std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now());
      if (remaining <= std::chrono::milliseconds::zero()) return false;
      std::this_thread::sleep_for(
          std::min(remaining, std::chrono::milliseconds(10)));
    }
  }

  pid_t pid_;
};

bool RecoveryWaitForReport(int fd, RecoveryChildReport* report,
                           std::string* error) {
  const auto deadline =
      Clock::now() + std::chrono::milliseconds(kRecoveryReadyTimeoutMs);
  for (;;) {
    const int64_t remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now())
            .count();
    if (remaining <= 0) {
      *error = "timed out waiting for the crash child's sync-ready report";
      return false;
    }
    pollfd waiter{fd, POLLIN, 0};
    const int polled = ::poll(&waiter, 1, static_cast<int>(remaining));
    if (polled < 0) {
      if (errno == EINTR) continue;
      *error = "poll of crash-child report failed: " + ErrnoMessage(errno);
      return false;
    }
    if (polled == 0) continue;
    if ((waiter.revents & POLLNVAL) != 0) {
      *error = "crash-child report pipe is invalid";
      return false;
    }
    if ((waiter.revents & (POLLIN | POLLHUP | POLLERR)) == 0) continue;
    ssize_t n;
    do {
      n = ::read(fd, report, sizeof(*report));
    } while (n < 0 && errno == EINTR);
    if (n != sizeof(*report)) {
      *error = n < 0 ? "read of crash-child report failed: " + ErrnoMessage(errno)
                     : "crash-child report EOF/short read: " + std::to_string(n) +
                           " of " + std::to_string(sizeof(*report)) + " bytes";
      return false;
    }
    if (report->magic != kRecoveryReportMagic) {
      *error = "crash-child report has an invalid magic value";
      return false;
    }
    return true;
  }
}

bool RecoveryEmit(JsonlReporter* reporter, const std::string& line) {
  reporter->Emit(line);
  if (!reporter->good()) {
    ADD_FAILURE() << "cannot append recovery evidence to " << reporter->path();
    return false;
  }
  return true;
}

void RunRecoveryEngine(EngineKind kind, const fs::path& root,
                       JsonlReporter* reporter) {
  const fs::path dir = root / EngineName(kind);
  std::error_code ec;
  ASSERT_TRUE(fs::create_directory(dir, ec) && !ec)
      << "cannot create fresh per-engine recovery directory " << dir << ": "
      << ec.message();
  const RecoveryOptions options = MakeRecoveryOptions(kind, dir);

  bool ok = false;
  RecoveryOpened initial = OpenRecoveryEngine(options, &ok);
  ASSERT_TRUE(ok) << initial.store->StartupError();
  ASSERT_TRUE(RecoveryDirectError(options, initial).empty())
      << RecoveryDirectError(options, initial);
  uint32_t completed = 0;
  std::string operation_error;
  ASSERT_TRUE(RecoveryWriteBatch(initial.store.get(), 0, &completed,
                                 &operation_error))
      << operation_error;
  ASSERT_EQ(completed, kRecoveryKeysPerBatch);
  ASSERT_TRUE(initial.store->Healthy() &&
              initial.store->Count() == kRecoveryKeysPerBatch);
  ASSERT_TRUE(initial.store->Close());
  initial.store.reset();

  RecoveryOpened clean = OpenRecoveryEngine(options, &ok);
  ASSERT_TRUE(ok) << clean.store->StartupError();
  ASSERT_TRUE(RecoveryDirectError(options, clean).empty())
      << RecoveryDirectError(options, clean);
  ASSERT_TRUE(RecoveryVerifyValues(clean.store.get(), 0, kRecoveryKeysPerBatch,
                                   &operation_error))
      << operation_error;
  ASSERT_TRUE(clean.store->Close());
  clean.store.reset();

  std::ostringstream clean_record;
  clean_record << "{\"record\":\"clean_reopen\",\"schema_version\":1"
               << ",\"engine\":" << JsonString(EngineName(kind))
               << ",\"value_bytes\":" << kValueBytes
               << ",\"keys\":" << kRecoveryKeysPerBatch
               << ",\"count\":" << kRecoveryKeysPerBatch
               << ",\"lookup_verified\":true,\"out_len_verified\":true"
               << ",\"value_len_verified\":true,\"full_payload_verified\":true"
               << ",\"clean_close_after_verification\":true,\"result\":\"PASS\"}";
  ASSERT_TRUE(RecoveryEmit(reporter, clean_record.str()));

  int report_pipe[2];
  ASSERT_EQ(::pipe(report_pipe), 0) << ErrnoMessage(errno);
  ScopedFd report_read(report_pipe[0]);
  ScopedFd report_write(report_pipe[1]);
  const pid_t child = ::fork();
  ASSERT_GE(child, 0) << ErrnoMessage(errno);
  if (child == 0) {
    report_read.Reset(-1);
    RecoveryChildMain(options, report_write.get());
  }
  report_write.Reset(-1);
  RecoveryChildGuard child_guard(child);
  RecoveryChildReport child_report;
  ASSERT_TRUE(RecoveryWaitForReport(report_read.get(), &child_report,
                                    &operation_error))
      << operation_error;
  if (child_report.ready != 1 || child_report.step != kRecoveryChildReady) {
    FAIL() << "crash child did not reach explicit sync readiness: step="
           << child_report.step << ", status=" << child_report.status_code
           << ", errno=" << child_report.saved_errno << ", writes="
           << child_report.writes_completed << ", detail=" << child_report.detail;
  }
  ASSERT_EQ(child_report.writes_completed, kRecoveryKeysPerBatch);
  ASSERT_EQ(::kill(child, SIGKILL), 0) << ErrnoMessage(errno);
  int child_status = 0;
  ASSERT_TRUE(child_guard.Reap(&child_status)) << ErrnoMessage(errno);
  ASSERT_TRUE(WIFSIGNALED(child_status) && WTERMSIG(child_status) == SIGKILL)
      << "crash child status " << child_status;

  std::ostringstream kill_record;
  kill_record << "{\"record\":\"sigkill\",\"schema_version\":1"
              << ",\"engine\":" << JsonString(EngineName(kind))
              << ",\"pid\":" << child
              << ",\"ready_step\":" << child_report.step
              << ",\"synced_writes\":" << child_report.writes_completed
              << ",\"signal\":" << WTERMSIG(child_status)
              << ",\"close_before_kill\":false,\"result\":\"PASS\"}";
  ASSERT_TRUE(RecoveryEmit(reporter, kill_record.str()));

  RecoveryOpened recovered = OpenRecoveryEngine(options, &ok);
  ASSERT_TRUE(ok) << recovered.store->StartupError();
  ASSERT_TRUE(RecoveryDirectError(options, recovered).empty())
      << RecoveryDirectError(options, recovered);
  uint64_t warm_restarts = 0;
  uint64_t unclean_resets = 0;
  std::string unclean_reset_reason;
  if (kind == EngineKind::kSlab) {
    const auto stats = recovered.slab->GetStats();
    warm_restarts = stats.warm_restarts;
    unclean_resets = stats.unclean_resets;
    unclean_reset_reason = recovered.slab->UncleanResetReason();
    ASSERT_EQ(warm_restarts, 1u);
    ASSERT_EQ(unclean_resets, 0u);
    ASSERT_TRUE(unclean_reset_reason.empty()) << unclean_reset_reason;
  }
  ASSERT_TRUE(RecoveryVerifyValues(recovered.store.get(), 0,
                                   2 * kRecoveryKeysPerBatch, &operation_error))
      << operation_error;
  ASSERT_TRUE(recovered.store->Close());
  recovered.store.reset();

  std::ostringstream recovery_record;
  recovery_record << "{\"record\":\"recovery\",\"schema_version\":1"
                  << ",\"engine\":" << JsonString(EngineName(kind))
                  << ",\"expected_keys\":" << 2 * kRecoveryKeysPerBatch
                  << ",\"verified_keys\":" << 2 * kRecoveryKeysPerBatch
                  << ",\"value_bytes\":" << kValueBytes
                  << ",\"all_clean_and_synced_keys_present\":true"
                  << ",\"unexpected_keys\":0,\"missing_keys\":0"
                  << ",\"corrupt_keys\":0"
                  << ",\"warm_restarts\":" << warm_restarts
                  << ",\"unclean_resets\":" << unclean_resets
                  << ",\"unclean_reset_reason\":"
                  << JsonString(unclean_reset_reason)
                  << ",\"clean_close_after_verification\":true"
                  << ",\"result\":\"PASS\"}";
  ASSERT_TRUE(RecoveryEmit(reporter, recovery_record.str()));
}

TEST(LseM3Recovery, CleanReopenAndSigkillAfterExplicitSync) {
  const char* const root_text = std::getenv("DFKV_LSE_M3_RECOVERY_ROOT");
  if (root_text == nullptr || *root_text == '\0') {
    EmitStdout(
        "{\"record\":\"recovery_preflight\",\"schema_version\":1,"
        "\"result\":\"SKIP\",\"reason\":\"DFKV_LSE_M3_RECOVERY_ROOT is required\"}");
    GTEST_SKIP() << "set DFKV_LSE_M3_RECOVERY_ROOT to a new absolute directory; "
                    "every parent component must be a real directory (no "
                    "symlinks), its mounted backing must resolve to NVMe, and "
                    "at least 400 GiB must be free";
  }

  RootEvidence root_evidence;
  std::string root_error;
  if (!PrepareFreshNvmeRoot(fs::path(root_text), &root_evidence, &root_error)) {
    PreflightFailure(root_error, &root_evidence);
    return;
  }
  const fs::path root = root_evidence.normalized_root;
  const fs::path probe = root / "recovery_direct_io.probe";
  std::string probe_error;
  if (!ProbeDirectIo(probe, &probe_error)) {
    PreflightFailure("recovery O_DIRECT probe failed: " + probe_error,
                     &root_evidence);
    return;
  }

  JsonlReporter reporter(root / kRecoveryJsonlName);
  std::ostringstream configuration;
  configuration << std::boolalpha
                << "{\"record\":\"recovery_configuration\",\"schema_version\":1"
                << ",\"supplied_root\":"
                << JsonString(root_evidence.supplied_root.string())
                << ",\"root\":" << JsonString(root.string())
                << ",\"parent_component_walk\":"
                   "\"openat O_PATH|O_DIRECTORY|O_NOFOLLOW\""
                << ",\"root_creation\":\"mkdirat in validated parent descriptor\""
                << ",\"backing_device\":\"" << root_evidence.device_major << ':'
                << root_evidence.device_minor << '"'
                << ",\"sysfs_dev_block\":"
                << JsonString(root_evidence.sysfs_dev_block)
                << ",\"resolved_device\":"
                << JsonString(root_evidence.resolved_device)
                << ",\"backing_physical_leaf_count\":"
                << root_evidence.physical_leaf_count
                << ",\"backing_all_leaves_nvme\":"
                << root_evidence.all_leaves_nvme
                << ",\"backing_candidates\":" << BackingCandidatesJson(root_evidence)
                << ",\"filesystem_available_bytes\":"
                << root_evidence.available_bytes
                << ",\"minimum_free_bytes\":" << kRequiredFreeBytes
                << ",\"capacity_bytes_per_engine\":" << kStandardCapacity
                << ",\"value_bytes\":" << kValueBytes
                << ",\"keys_per_batch\":" << kRecoveryKeysPerBatch
                << ",\"direct_io_probe\":" << JsonString(probe.string())
                << ",\"direct_io_probe_ok\":true"
                << ",\"slab_warm_restart\":\"auto_real_boot_mount_dev_shm\""
                << ",\"slab_table_sync_ms\":0,\"slab_reclaim_interval_ms\":0"
                << ",\"log_segment_bytes\":" << kLogSegment
                << ",\"prekill_barrier\":"
                   "\"fdatasync payloads before persistent index while unclean\""
                << ",\"jsonl\":" << JsonString(reporter.path().string())
                << ",\"power_cycle_tested\":false,\"reboot_or_remount_tested\":false"
                << ",\"cold_cache_proof\":false}";
  ASSERT_TRUE(RecoveryEmit(&reporter, configuration.str()));

  for (const EngineKind kind : {EngineKind::kSlab, EngineKind::kLog}) {
    SCOPED_TRACE(EngineName(kind));
    RunRecoveryEngine(kind, root, &reporter);
    if (::testing::Test::HasFatalFailure()) return;
  }
  ASSERT_TRUE(reporter.good()) << reporter.path();
}

// ---- opt-in sustained-I/O and Zipf eviction evidence --------------------------
//
// These are measurements, not a cross-engine performance gate.  Cold fill is a
// separate interval.  Write throughput uses the union of the eight workers'
// CacheDirect call intervals, so payload generation and gaps with no API call in
// flight are not reported as I/O time.  Whole-resident reads use the same rule
// for RangeDirect.  Zipf hit, miss, and miss-admission latencies are disjoint.
constexpr uint64_t kQualityObjectsPerCapacity = kStandardCapacity / kValueBytes;
constexpr uint64_t kQualityZipfKeys = 2 * kQualityObjectsPerCapacity;
constexpr uint64_t kQualityWarmupRequests = kQualityZipfKeys;
constexpr uint64_t kQualityMeasuredRequests = 2 * kQualityZipfKeys;
constexpr uint64_t kQualityZipfKeyBase = 1ull << 48;
constexpr uint64_t kQualityZipfValueBase = 1ull << 50;
constexpr uint64_t kQualityTraceSeedBase = 0x4d335a6970663930ull;
constexpr size_t kQualityIoCapacity =
    (kValueBytes + kIoAlignment - 1) / kIoAlignment * kIoAlignment;
constexpr char kQualityJsonlName[] = "lse_m3_io.jsonl";
static_assert(kQualityObjectsPerCapacity > 0 && kQualityZipfKeys > 0,
              "the quality workload needs a nonempty capacity and key space");

struct QualityPutStats {
  uint64_t ok = 0;
  uint64_t cache_full = 0;
  uint64_t not_found = 0;
  uint64_t quota = 0;
  uint64_t io_error = 0;
  uint64_t invalid = 0;
  uint64_t resource_exhausted = 0;
  uint64_t no_rail = 0;

  void Add(Status status) {
    switch (status) {
      case Status::kOk: ++ok; break;
      case Status::kCacheFull: ++cache_full; break;
      case Status::kNotFound: ++not_found; break;
      case Status::kQuotaExceeded: ++quota; break;
      case Status::kIOError: ++io_error; break;
      case Status::kInvalid: ++invalid; break;
      case Status::kResourceExhausted: ++resource_exhausted; break;
      case Status::kNoCompatibleRail: ++no_rail; break;
    }
  }
  void Merge(const QualityPutStats& other) {
    ok += other.ok;
    cache_full += other.cache_full;
    not_found += other.not_found;
    quota += other.quota;
    io_error += other.io_error;
    invalid += other.invalid;
    resource_exhausted += other.resource_exhausted;
    no_rail += other.no_rail;
  }
  uint64_t Calls() const {
    return ok + cache_full + not_found + quota + io_error + invalid +
           resource_exhausted + no_rail;
  }
  uint64_t UnexpectedForChurn() const { return Calls() - ok - cache_full; }
};

struct QualityCallInterval {
  uint64_t begin_ns = 0;
  uint64_t end_ns = 0;
  int32_t status = 0;  // actual Status enum value returned by this call
};

struct QualityLatency {
  uint64_t calls = 0;
  uint64_t total_ns = 0;
  uint64_t p50_ns = 0;
  uint64_t p99_ns = 0;
};

struct QualityCallMetrics {
  QualityLatency latency;
  uint64_t api_span_ns = 0;
  uint64_t api_union_ns = 0;  // wall time with at least one measured call active
};

uint64_t QualityNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             Clock::now().time_since_epoch())
      .count();
}

uint64_t QualityNearestRank(const std::vector<uint64_t>& sorted,
                            uint32_t numerator) {
  if (sorted.empty()) return 0;
  return sorted[(static_cast<uint64_t>(sorted.size()) * numerator + 99) / 100 - 1];
}

QualityLatency QualityLatencyOf(std::vector<uint64_t>* samples) {
  QualityLatency out;
  out.calls = samples->size();
  std::sort(samples->begin(), samples->end());
  for (const uint64_t sample : *samples) out.total_ns += sample;
  out.p50_ns = QualityNearestRank(*samples, 50);
  out.p99_ns = QualityNearestRank(*samples, 99);
  return out;
}

QualityCallMetrics QualityMetricsOf(std::vector<QualityCallInterval>* calls) {
  QualityCallMetrics out;
  if (calls->empty()) return out;
  std::sort(calls->begin(), calls->end(),
            [](const QualityCallInterval& a, const QualityCallInterval& b) {
              return a.begin_ns != b.begin_ns ? a.begin_ns < b.begin_ns
                                              : a.end_ns < b.end_ns;
            });
  std::vector<uint64_t> durations;
  durations.reserve(calls->size());
  out.api_span_ns = calls->front().begin_ns;
  uint64_t union_begin = calls->front().begin_ns;
  uint64_t union_end = calls->front().end_ns;
  uint64_t last_end = 0;
  for (const auto& call : *calls) {
    durations.push_back(call.end_ns - call.begin_ns);
    out.api_span_ns = std::min(out.api_span_ns, call.begin_ns);
    last_end = std::max(last_end, call.end_ns);
    if (call.begin_ns > union_end) {
      out.api_union_ns += union_end - union_begin;
      union_begin = call.begin_ns;
      union_end = call.end_ns;
    } else {
      union_end = std::max(union_end, call.end_ns);
    }
  }
  out.api_union_ns += union_end - union_begin;
  out.api_span_ns = last_end - out.api_span_ns;
  out.latency = QualityLatencyOf(&durations);
  return out;
}

QualityLatency QualityLatencyForStatus(
    const std::vector<QualityCallInterval>& calls, Status wanted) {
  std::vector<uint64_t> durations;
  durations.reserve(calls.size());
  for (const auto& call : calls) {
    if (call.status == static_cast<int32_t>(wanted))
      durations.push_back(call.end_ns - call.begin_ns);
  }
  return QualityLatencyOf(&durations);
}

double QualitySeconds(uint64_t nanoseconds) {
  return static_cast<double>(nanoseconds) / 1000000000.0;
}

double QualityMiBPerSecond(uint64_t bytes, uint64_t nanoseconds) {
  return nanoseconds == 0
             ? 0.0
             : static_cast<double>(bytes) / QualitySeconds(nanoseconds) /
                   static_cast<double>(1ull << 20);
}

void QualitySamplesJson(std::ostringstream* out, const std::string& name,
                        const std::vector<QualityCallInterval>& calls) {
  *out << ',' << JsonString(name) << ":[";
  for (size_t i = 0; i < calls.size(); ++i) {
    if (i != 0) *out << ',';
    *out << '[' << calls[i].begin_ns << ',' << calls[i].end_ns << ','
         << calls[i].status << ']';
  }
  *out << ']';
}

void QualityPutJson(std::ostringstream* out, const std::string& prefix,
                    const QualityPutStats& stats) {
  *out << ',' << JsonString(prefix + "_ok") << ':' << stats.ok
       << ',' << JsonString(prefix + "_cache_full") << ':' << stats.cache_full
       << ',' << JsonString(prefix + "_not_found") << ':' << stats.not_found
       << ',' << JsonString(prefix + "_quota_exceeded") << ':' << stats.quota
       << ',' << JsonString(prefix + "_io_error") << ':' << stats.io_error
       << ',' << JsonString(prefix + "_invalid") << ':' << stats.invalid
       << ',' << JsonString(prefix + "_resource_exhausted") << ':'
       << stats.resource_exhausted
       << ',' << JsonString(prefix + "_no_compatible_rail") << ':' << stats.no_rail;
}

void QualityLatencyJson(std::ostringstream* out, const std::string& prefix,
                        const QualityLatency& latency) {
  *out << ',' << JsonString(prefix + "_calls") << ':' << latency.calls
       << ',' << JsonString(prefix + "_p50_ns") << ':' << latency.p50_ns
       << ',' << JsonString(prefix + "_p99_ns") << ':' << latency.p99_ns
       << ',' << JsonString(prefix + "_cumulative_seconds")
       << ':' << QualitySeconds(latency.total_ns);
}

struct QualityWriteInterval {
  uint64_t first_key = 0;
  uint64_t planned_calls = 0;
  uint32_t writers_started = 0;
  uint64_t allocation_failures = 0;
  double wall_seconds_including_generation = 0;
  QualityPutStats puts;
  QualityCallMetrics metrics;
  std::vector<QualityCallInterval> calls;
  std::string thread_error;
};

QualityWriteInterval RunQualityWrites(StoreEngine* store, uint64_t first_key,
                                      uint64_t count) {
  QualityWriteInterval out;
  out.first_key = first_key;
  out.planned_calls = count;
  std::atomic<uint64_t> next{0};
  std::atomic<bool> stop{false};
  std::vector<std::thread> writers;
  struct WorkerResult {
    QualityPutStats puts;
    uint64_t allocation_failures = 0;
    std::vector<QualityCallInterval> calls;
  };
  std::vector<WorkerResult> results(kWriters);
  const auto wall_start = Clock::now();
  try {
    writers.reserve(kWriters);
    for (uint32_t worker = 0; worker < kWriters; ++worker) {
      writers.emplace_back([&, worker] {
        auto& result = results[worker];
        AlignedBuffer buffer(kQualityIoCapacity);
        if (!buffer) {
          ++result.allocation_failures;
          stop.store(true, std::memory_order_relaxed);
          return;
        }
        result.calls.reserve(count / kWriters + 1);
        while (!stop.load(std::memory_order_relaxed)) {
          const uint64_t sequence = next.fetch_add(1, std::memory_order_relaxed);
          if (sequence >= count) break;
          const uint64_t id = first_key + sequence;
          FillValue(id, buffer.data(), kValueBytes);
          const uint64_t begin = QualityNowNs();
          const Status status = store->CacheDirect(
              MakeKey(id), buffer.data(), kValueBytes, buffer.size());
          const uint64_t end = QualityNowNs();
          result.calls.push_back({begin, end, static_cast<int32_t>(status)});
          result.puts.Add(status);
          if (status != Status::kOk && status != Status::kCacheFull)
            stop.store(true, std::memory_order_relaxed);
        }
      });
    }
  } catch (const std::exception& error) {
    out.thread_error = error.what();
    stop.store(true, std::memory_order_relaxed);
  } catch (...) {
    out.thread_error = "unknown std::thread creation failure";
    stop.store(true, std::memory_order_relaxed);
  }
  out.writers_started = writers.size();
  for (auto& writer : writers) writer.join();
  out.wall_seconds_including_generation = Seconds(Clock::now() - wall_start);
  std::vector<QualityCallInterval> calls;
  for (auto& result : results) {
    out.puts.Merge(result.puts);
    out.allocation_failures += result.allocation_failures;
    calls.insert(calls.end(), std::make_move_iterator(result.calls.begin()),
                 std::make_move_iterator(result.calls.end()));
  }
  out.calls = std::move(calls);
  out.metrics = QualityMetricsOf(&out.calls);
  return out;
}

void QualityWriteJson(std::ostringstream* out, const std::string& prefix,
                      const QualityWriteInterval& interval) {
  *out << ',' << JsonString(prefix + "_first_key") << ':' << interval.first_key
       << ',' << JsonString(prefix + "_planned_calls") << ':'
       << interval.planned_calls
       << ',' << JsonString(prefix + "_actual_calls") << ':' << interval.puts.Calls()
       << ',' << JsonString(prefix + "_writers_started") << ':'
       << interval.writers_started
       << ',' << JsonString(prefix + "_allocation_failures") << ':'
       << interval.allocation_failures
       << ',' << JsonString(prefix + "_wall_seconds_including_generation") << ':'
       << interval.wall_seconds_including_generation
       << ',' << JsonString(prefix + "_api_span_seconds")
       << ':' << QualitySeconds(interval.metrics.api_span_ns)
       << ',' << JsonString(prefix + "_api_union_seconds")
       << ':' << QualitySeconds(interval.metrics.api_union_ns)
       << ',' << JsonString(prefix + "_successful_payload_mib_per_second")
       << ':' << QualityMiBPerSecond(interval.puts.ok * kValueBytes,
                                    interval.metrics.api_union_ns);
  QualityPutJson(out, prefix + "_put", interval.puts);
  QualityLatencyJson(out, prefix + "_cache_direct_latency",
                     interval.metrics.latency);
  QualitySamplesJson(out, prefix + "_cache_direct_samples", interval.calls);
}

struct QualityReadback {
  uint64_t enumerated = 0;
  uint64_t distinct = 0;
  uint64_t duplicates = 0;
  uint64_t unexpected = 0;
  uint64_t allocation_failures = 0;
  uint64_t lookup_ok = 0;
  uint64_t lookup_not_found = 0;
  uint64_t lookup_other = 0;
  uint64_t lookup_length_mismatch = 0;
  uint64_t read_ok = 0;
  uint64_t read_not_found = 0;
  uint64_t read_io_error = 0;
  uint64_t read_other = 0;
  uint64_t out_length_mismatch = 0;
  uint64_t value_length_mismatch = 0;
  uint64_t payload_mismatch = 0;
  uint64_t verified_payload_bytes = 0;
  QualityCallMetrics metrics;
  std::vector<QualityCallInterval> calls;
};

QualityReadback ReadEveryResidentValue(StoreEngine* store, uint64_t first_key,
                                       uint64_t last_key_exclusive) {
  QualityReadback out;
  std::vector<BlockKey> keys;
  store->ForEachKey([&](const BlockKey& key) { keys.push_back(key); });
  out.enumerated = keys.size();
  std::sort(keys.begin(), keys.end(), KeyOrder);
  for (size_t i = 1; i < keys.size(); ++i)
    if (keys[i] == keys[i - 1]) ++out.duplicates;
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  out.distinct = keys.size();  // exactly this snapshot is read-eligible

  AlignedBuffer read_buffer(kQualityIoCapacity);
  if (!read_buffer) {
    ++out.allocation_failures;
    return out;
  }
  std::string expected(kValueBytes, '\0');
  out.calls.reserve(keys.size());
  for (const BlockKey& key : keys) {
    const uint64_t id = key.digest_lo;
    if (id < first_key || id >= last_key_exclusive || !(key == MakeKey(id)))
      ++out.unexpected;
    ValueMetadata metadata;
    const Status lookup = store->Lookup(key, &metadata);
    if (lookup == Status::kOk) {
      ++out.lookup_ok;
      if (metadata.value_len != kValueBytes) ++out.lookup_length_mismatch;
    } else if (lookup == Status::kNotFound) {
      ++out.lookup_not_found;
    } else {
      ++out.lookup_other;
    }
    FillValue(id, expected.data(), expected.size());
    const char* data = nullptr;
    size_t out_len = 0;
    size_t value_len = 0;
    const uint64_t begin = QualityNowNs();
    const Status read = store->RangeDirect(
        key, 0, 0, read_buffer.data(), read_buffer.size(), &data, &out_len,
        &value_len);
    const uint64_t end = QualityNowNs();
    out.calls.push_back({begin, end, static_cast<int32_t>(read)});
    if (read == Status::kOk) {
      ++out.read_ok;
      if (out_len != kValueBytes) ++out.out_length_mismatch;
      if (value_len != kValueBytes) ++out.value_length_mismatch;
      if (out_len == kValueBytes && value_len == kValueBytes &&
          std::string(data, out_len) == expected) {
        out.verified_payload_bytes += out_len;
      } else if (out_len == kValueBytes && value_len == kValueBytes) {
        ++out.payload_mismatch;
      }
    } else if (read == Status::kNotFound) {
      ++out.read_not_found;
    } else if (read == Status::kIOError) {
      ++out.read_io_error;
    } else {
      ++out.read_other;
    }
  }
  out.metrics = QualityMetricsOf(&out.calls);
  return out;
}

void QualityReadbackJson(std::ostringstream* out, const QualityReadback& read) {
  *out << ",\"resident_enumerated\":" << read.enumerated
       << ",\"resident_distinct_eligible\":" << read.distinct
       << ",\"resident_duplicates\":" << read.duplicates
       << ",\"resident_unexpected_keys\":" << read.unexpected
       << ",\"resident_allocation_failures\":" << read.allocation_failures
       << ",\"resident_lookup_ok\":" << read.lookup_ok
       << ",\"resident_lookup_not_found\":" << read.lookup_not_found
       << ",\"resident_lookup_other_error\":" << read.lookup_other
       << ",\"resident_lookup_length_mismatch\":" << read.lookup_length_mismatch
       << ",\"resident_direct_read_ok\":" << read.read_ok
       << ",\"resident_direct_read_not_found\":" << read.read_not_found
       << ",\"resident_direct_read_io_error\":" << read.read_io_error
       << ",\"resident_direct_read_other_error\":" << read.read_other
       << ",\"resident_out_length_mismatch\":" << read.out_length_mismatch
       << ",\"resident_value_length_mismatch\":" << read.value_length_mismatch
       << ",\"resident_payload_mismatch\":" << read.payload_mismatch
       << ",\"resident_verified_payload_bytes\":" << read.verified_payload_bytes
       << ",\"resident_read_api_span_seconds\":"
       << QualitySeconds(read.metrics.api_span_ns)
       << ",\"resident_read_api_union_seconds\":"
       << QualitySeconds(read.metrics.api_union_ns)
       << ",\"resident_verified_payload_mib_per_second\":"
       << QualityMiBPerSecond(read.verified_payload_bytes,
                              read.metrics.api_union_ns);
  QualityLatencyJson(out, "resident_range_direct_latency", read.metrics.latency);
  QualitySamplesJson(out, "resident_range_direct_samples", read.calls);
}

struct QualityStore {
  std::unique_ptr<StoreEngine> store;
  DiskSlabStore* slab = nullptr;
  LogStore* log = nullptr;
  std::shared_ptr<StrictDirectIo> log_io;
  bool open_ok = false;
  bool fresh_directory = false;
};

QualityStore OpenQualityStore(EngineKind kind, const fs::path& dir) {
  QualityStore opened;
  std::error_code ec;
  opened.fresh_directory = fs::create_directory(dir, ec) && !ec;
  if (!opened.fresh_directory) return opened;
  if (kind == EngineKind::kSlab) {
    DiskSlabStore::Options options;
    options.dir = dir.string();
    options.capacity_bytes = kStandardCapacity;
    options.extent_bytes = kSlabExtent;
    options.slot_granularity = kSlabGranularity;
    options.direct_writes = true;
    options.balance = false;
    // Keep ordinary defaults: warm restart kOff, table sync 100 ms, reclaim 50 ms.
    auto store = std::make_unique<DiskSlabStore>(options, &opened.open_ok);
    opened.slab = store.get();
    opened.store = std::move(store);
  } else {
    LogStore::Options options;
    options.dir = dir.string();
    options.capacity_bytes = kStandardCapacity;
    options.direct_writes = true;
    options.segment_bytes = kLogSegment;
    options.open_segments = 0;
    options.relocate_pct = 25;
    options.sync_ms = 1000;
    options.cleaner = true;
    options.preopen = true;
    options.retain_enabled = false;
    opened.log_io = std::make_shared<StrictDirectIo>();
    options.io = opened.log_io;
    auto store = std::make_unique<LogStore>(options, &opened.open_ok);
    opened.log = store.get();
    opened.store = std::move(store);
  }
  opened.open_ok = opened.open_ok && opened.store->Healthy();
  return opened;
}

struct QualityFinalStats {
  uint64_t evictions = 0;
  uint64_t used_bytes = 0;
  uint64_t slab_dio_write_fallbacks = 0;
  uint64_t slab_dio_read_fallbacks = 0;
  uint64_t slab_payload_crc_failures = 0;
  uint64_t slab_metadata_io_errors = 0;
  uint64_t log_read_crc_failures = 0;
};

QualityFinalStats QualityStatsOf(EngineKind kind, const QualityStore& opened) {
  QualityFinalStats stats;
  if (opened.store == nullptr) return stats;
  stats.evictions = opened.store->Evictions();
  stats.used_bytes = opened.store->UsedBytes();
  if (kind == EngineKind::kSlab) {
    const auto slab = opened.slab->GetStats();
    stats.slab_dio_write_fallbacks = slab.dio_write_fallbacks;
    stats.slab_dio_read_fallbacks = slab.dio_read_fallbacks;
    stats.slab_payload_crc_failures = slab.payload_crc_failures;
    stats.slab_metadata_io_errors = slab.metadata_io_errors;
  } else {
    stats.log_read_crc_failures = opened.log->GetStats().read_crc_failures;
  }
  return stats;
}

void QualityFinalStatsJson(std::ostringstream* out, const QualityFinalStats& stats) {
  *out << ",\"evictions\":" << stats.evictions
       << ",\"used_bytes\":" << stats.used_bytes
       << ",\"slab_dio_write_fallbacks\":" << stats.slab_dio_write_fallbacks
       << ",\"slab_dio_read_fallbacks\":" << stats.slab_dio_read_fallbacks
       << ",\"slab_payload_crc_failures\":" << stats.slab_payload_crc_failures
       << ",\"slab_metadata_io_errors\":" << stats.slab_metadata_io_errors
       << ",\"log_read_crc_failures\":" << stats.log_read_crc_failures;
}

bool QualityFinalStatsClean(EngineKind kind, const QualityFinalStats& stats) {
  return stats.slab_dio_write_fallbacks == 0 && stats.slab_dio_read_fallbacks == 0 &&
         stats.slab_payload_crc_failures == 0 && stats.slab_metadata_io_errors == 0 &&
         (kind == EngineKind::kSlab || stats.log_read_crc_failures == 0);
}

bool QualityDirectActive(EngineKind kind, const QualityStore& opened) {
  if (opened.store == nullptr) return false;
  return kind == EngineKind::kSlab
             ? opened.slab->DirectWritesActive()
             : opened.log_io->direct_successes() > 0 &&
                   opened.log_io->direct_refusals() == 0;
}

struct QualityIoResult {
  uint32_t pair = 0;
  uint32_t order = 0;
  EngineKind kind = EngineKind::kSlab;
  bool valid = false;
  std::vector<std::string> violations;
  double write_mib_s = 0;
  double write_p50_ns = 0;
  double write_p99_ns = 0;
  double read_mib_s = 0;
  double read_p50_ns = 0;
  double read_p99_ns = 0;
};

void QualityViolation(QualityIoResult* result, const std::string& reason) {
  result->violations.push_back(reason);
}

QualityIoResult RunQualityIo(EngineKind kind, uint32_t pair, uint32_t order,
                             const fs::path& dir, JsonlReporter* reporter) {
  QualityIoResult result;
  result.pair = pair;
  result.order = order;
  result.kind = kind;
  QualityStore opened = OpenQualityStore(kind, dir);
  if (!opened.fresh_directory || !opened.open_ok) {
    QualityViolation(&result, "fresh directory or healthy direct open failed: " +
                                  (opened.store ? opened.store->StartupError()
                                                : std::string("directory exists")));
  } else if (!QualityDirectActive(kind, opened)) {
    QualityViolation(&result, "requested O_DIRECT is not active");
  }
  const uint64_t first =
      static_cast<uint64_t>(pair) * 3 * kQualityObjectsPerCapacity;
  QualityWriteInterval cold, steady;
  uint64_t evictions_before_steady = 0;
  uint64_t steady_evictions = 0;
  QualityReadback readback;
  double settle_seconds = 0;
  bool slab_settled = true;
  if (opened.open_ok && QualityDirectActive(kind, opened)) {
    cold = RunQualityWrites(opened.store.get(), first, kQualityObjectsPerCapacity);
    // The measured interval begins only after the separate full-capacity fill.
    evictions_before_steady = opened.store->Evictions();
    steady = RunQualityWrites(opened.store.get(), first + kQualityObjectsPerCapacity,
                              2 * kQualityObjectsPerCapacity);
    steady_evictions = opened.store->Evictions() - evictions_before_steady;
    const auto settle_start = Clock::now();
    if (kind == EngineKind::kSlab) {
      const auto deadline = settle_start + std::chrono::seconds(30);
      for (;;) {
        const auto observation = opened.slab->GetWatermarkObservation();
        if (observation.watermark_active == 0 &&
            opened.store->UsedBytes() <= observation.watermark_high_bytes)
          break;
        if (Clock::now() >= deadline) {
          slab_settled = false;
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    } else {
      // LSE exposes no idle gauge.  Its writers are joined; after this explicit
      // quiet delay, any lookup/read loss from the sorted snapshot is still an
      // error, never silently discarded from the resident-read denominator.
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    settle_seconds = Seconds(Clock::now() - settle_start);
    readback = ReadEveryResidentValue(opened.store.get(), first,
                                      first + 3 * kQualityObjectsPerCapacity);
  }

  for (const auto* interval : {&cold, &steady}) {
    if (interval->writers_started != kWriters || !interval->thread_error.empty() ||
        interval->allocation_failures != 0)
      QualityViolation(&result, "could not run eight writers without setup failures");
    if (interval->puts.Calls() != interval->planned_calls ||
        interval->metrics.latency.calls != interval->planned_calls)
      QualityViolation(&result, "actual/timed PUT calls differ from planned calls");
    if (interval->puts.UnexpectedForChurn() != 0)
      QualityViolation(&result, "PUT returned a non-Ok/non-CacheFull status");
    if (interval->metrics.api_union_ns == 0)
      QualityViolation(&result, "no real CacheDirect measurement interval");
  }
  if (steady.puts.ok == 0)
    QualityViolation(&result, "steady interval admitted no successful values");
  if (steady_evictions == 0)
    QualityViolation(&result, "steady interval produced no eviction delta");
  if (!slab_settled) QualityViolation(&result, "slab watermark did not settle");
  if (readback.distinct == 0 || readback.enumerated != readback.distinct ||
      readback.duplicates != 0 || readback.unexpected != 0 ||
      readback.allocation_failures != 0 ||
      readback.lookup_ok != readback.distinct || readback.lookup_not_found != 0 ||
      readback.lookup_other != 0 || readback.lookup_length_mismatch != 0 ||
      readback.read_ok != readback.distinct || readback.read_not_found != 0 ||
      readback.read_io_error != 0 || readback.read_other != 0 ||
      readback.out_length_mismatch != 0 || readback.value_length_mismatch != 0 ||
      readback.payload_mismatch != 0 || readback.metrics.latency.calls != readback.distinct ||
      readback.verified_payload_bytes != readback.distinct * kValueBytes ||
      readback.metrics.api_union_ns == 0) {
    QualityViolation(&result, "not every distinct resident value was fully verified");
  }
  const QualityFinalStats stats = QualityStatsOf(kind, opened);
  if (!QualityFinalStatsClean(kind, stats))
    QualityViolation(&result, "direct-I/O, CRC, or metadata error counter is nonzero");
  if (opened.store != nullptr && !opened.store->Healthy())
    QualityViolation(&result, "engine became unhealthy");
  const bool close_ok = opened.store != nullptr && opened.store->Close();
  if (!close_ok) QualityViolation(&result, "final Close failed");
  opened.store.reset();
  result.valid = result.violations.empty();
  result.write_mib_s = QualityMiBPerSecond(steady.puts.ok * kValueBytes,
                                           steady.metrics.api_union_ns);
  result.write_p50_ns = steady.metrics.latency.p50_ns;
  result.write_p99_ns = steady.metrics.latency.p99_ns;
  result.read_mib_s = QualityMiBPerSecond(readback.verified_payload_bytes,
                                          readback.metrics.api_union_ns);
  result.read_p50_ns = readback.metrics.latency.p50_ns;
  result.read_p99_ns = readback.metrics.latency.p99_ns;

  std::ostringstream out;
  out << std::setprecision(17) << std::boolalpha
      << "{\"record\":\"steady_io\",\"schema_version\":1,\"pair\":" << pair
      << ",\"engine_order\":" << order
      << ",\"engine\":" << JsonString(EngineName(kind))
      << ",\"capacity_bytes\":" << kStandardCapacity
      << ",\"value_bytes\":" << kValueBytes
      << ",\"objects_per_payload_capacity\":" << kQualityObjectsPerCapacity
      << ",\"cold_fill_is_timed_steady_state\":false"
      << ",\"steady_write_payload_bytes_requested\":"
      << 2 * kQualityObjectsPerCapacity * kValueBytes
      << ",\"steady_crosses_full_capacity\":true"
      << ",\"evictions_before_steady\":" << evictions_before_steady
      << ",\"steady_evictions\":" << steady_evictions
      << ",\"steady_admitted_values\":" << steady.puts.ok
      << ",\"settle_seconds\":" << settle_seconds
      << ",\"slab_settled\":" << slab_settled
      << ",\"close_ok\":" << close_ok;
  QualityWriteJson(&out, "cold_fill", cold);
  QualityWriteJson(&out, "steady", steady);
  QualityReadbackJson(&out, readback);
  QualityFinalStatsJson(&out, stats);
  out << ",\"result\":" << JsonString(result.valid ? "VALID" : "FAIL")
      << ",\"violations\":" << JsonStrings(result.violations) << '}';
  reporter->Emit(out.str());
  if (!reporter->good()) {
    QualityViolation(&result, "steady_io JSONL write failed");
    result.valid = false;
  }
  return result;
}

uint64_t QualitySplitMix64(uint64_t* state) {
  *state += 0x9e3779b97f4a7c15ULL;
  uint64_t z = *state;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

struct QualityTrace {
  uint64_t seed = 0;
  uint64_t key_base = 0;
  std::vector<uint32_t> warmup;
  std::vector<uint32_t> measured;
  uint64_t warmup_unique = 0;
  uint64_t measured_unique = 0;
  uint64_t all_unique = 0;
  uint64_t hash = 0;
  uint64_t measured_hash = 0;
  bool valid = false;
};

void QualityTraceHashAdd(uint64_t value, uint64_t* hash) {
  for (unsigned byte = 0; byte < 8; ++byte) {
    *hash ^= static_cast<uint8_t>(value >> (byte * 8));
    *hash *= 0x100000001b3ULL;
  }
}

QualityTrace MakeQualityTrace(uint32_t pair) {
  QualityTrace trace;
  trace.seed = kQualityTraceSeedBase + pair;
  trace.key_base = kQualityZipfKeyBase + pair * kQualityZipfKeys;
  std::vector<double> cdf(kQualityZipfKeys);
  double total = 0;
  for (uint64_t i = 0; i < kQualityZipfKeys; ++i) {
    total += std::pow(static_cast<double>(i + 1), -0.9);
    cdf[i] = total;
  }
  // Warm-up is declared initialization, not a sampled Zipf measurement.  Every
  // key is requested exactly once, in a reproducible per-pair permutation.
  trace.warmup.reserve(kQualityWarmupRequests);
  for (uint64_t i = 0; i < kQualityZipfKeys; ++i)
    trace.warmup.push_back(static_cast<uint32_t>(i));
  uint64_t shuffle_rng = trace.seed;
  for (uint64_t remaining = kQualityZipfKeys; remaining > 1; --remaining) {
    const double uniform =
        (QualitySplitMix64(&shuffle_rng) >> 11) * (1.0 / 9007199254740992.0);
    const size_t selected = static_cast<size_t>(uniform * remaining);
    std::swap(trace.warmup[remaining - 1], trace.warmup[selected]);
  }

  // Keep the measured vector independent of shuffle consumption.  Advancing a
  // separate stream by the old warm-up draw count preserves its prior derivation.
  uint64_t measured_rng = trace.seed;
  for (uint64_t i = 0; i < kQualityWarmupRequests; ++i)
    (void)QualitySplitMix64(&measured_rng);
  auto draw = [&] {
    const double uniform =
        (QualitySplitMix64(&measured_rng) >> 11) * (1.0 / 9007199254740992.0);
    return static_cast<uint32_t>(
        std::lower_bound(cdf.begin(), cdf.end(), uniform * total) - cdf.begin());
  };
  trace.measured.reserve(kQualityMeasuredRequests);
  for (uint64_t i = 0; i < kQualityMeasuredRequests; ++i)
    trace.measured.push_back(draw());
  std::vector<bool> seen(kQualityZipfKeys, false);
  for (const uint32_t id : trace.warmup) {
    if (!seen[id]) {
      seen[id] = true;
      ++trace.warmup_unique;
    }
  }
  std::fill(seen.begin(), seen.end(), false);
  for (const uint32_t id : trace.measured) {
    if (!seen[id]) {
      seen[id] = true;
      ++trace.measured_unique;
    }
  }
  std::fill(seen.begin(), seen.end(), false);
  for (const auto* requests : {&trace.warmup, &trace.measured}) {
    for (const uint32_t id : *requests) {
      if (!seen[id]) {
        seen[id] = true;
        ++trace.all_unique;
      }
    }
  }
  trace.measured_hash = 1469598103934665603ULL;
  QualityTraceHashAdd(trace.seed, &trace.measured_hash);
  QualityTraceHashAdd(trace.measured.size(), &trace.measured_hash);
  for (const uint32_t id : trace.measured)
    QualityTraceHashAdd(id, &trace.measured_hash);
  trace.hash = 1469598103934665603ULL;  // FNV-1a, little-endian request ids
  QualityTraceHashAdd(trace.seed, &trace.hash);
  QualityTraceHashAdd(trace.warmup.size(), &trace.hash);
  for (const uint32_t id : trace.warmup) QualityTraceHashAdd(id, &trace.hash);
  QualityTraceHashAdd(trace.measured.size(), &trace.hash);
  for (const uint32_t id : trace.measured) QualityTraceHashAdd(id, &trace.hash);
  trace.valid = trace.warmup.size() == kQualityWarmupRequests &&
                trace.measured.size() == kQualityMeasuredRequests &&
                trace.warmup_unique == kQualityZipfKeys &&
                trace.measured_unique >= kQualityZipfKeys / 4;
  return trace;
}

uint64_t QualityZipfValueId(uint32_t object, uint64_t epoch) {
  return kQualityZipfValueBase + epoch * kQualityZipfKeys + object;
}

struct QualityZipfPhase {
  uint64_t requests = 0;
  uint64_t allocation_failures = 0;
  uint64_t hits = 0;
  uint64_t misses = 0;
  uint64_t range_io_error = 0;
  uint64_t other_range_error = 0;
  uint64_t out_length_mismatch = 0;
  uint64_t value_length_mismatch = 0;
  uint64_t payload_mismatch = 0;
  uint64_t verified_hit_bytes = 0;
  QualityPutStats admissions;
  QualityLatency hit_latency;
  QualityLatency miss_latency;
  QualityLatency admission_latency;
  std::vector<QualityCallInterval> read_calls;
  std::vector<QualityCallInterval> admission_calls;
  double wall_seconds = 0;
};

QualityZipfPhase RunQualityZipfPhase(StoreEngine* store, const QualityTrace& trace,
                                     const std::vector<uint32_t>& requests,
                                     std::vector<uint64_t>* epochs) {
  QualityZipfPhase out;
  out.requests = requests.size();
  const auto wall_start = Clock::now();
  AlignedBuffer buffer(kQualityIoCapacity);
  if (!buffer) {
    out.allocation_failures = 1;
    out.wall_seconds = Seconds(Clock::now() - wall_start);
    return out;
  }
  std::string expected(kValueBytes, '\0');
  out.read_calls.reserve(requests.size());
  out.admission_calls.reserve(requests.size());
  for (const uint32_t object : requests) {
    const BlockKey key = MakeKey(trace.key_base + object);
    const char* data = nullptr;
    size_t out_len = 0;
    size_t value_len = 0;
    const uint64_t read_begin = QualityNowNs();
    const Status read = store->RangeDirect(
        key, 0, 0, buffer.data(), buffer.size(), &data, &out_len, &value_len);
    const uint64_t read_end = QualityNowNs();
    out.read_calls.push_back(
        {read_begin, read_end, static_cast<int32_t>(read)});
    if (read == Status::kOk) {
      ++out.hits;
      FillValue(QualityZipfValueId(object, (*epochs)[object]), expected.data(),
                expected.size());
      if (out_len != kValueBytes) ++out.out_length_mismatch;
      if (value_len != kValueBytes) ++out.value_length_mismatch;
      if (out_len == kValueBytes && value_len == kValueBytes) {
        if (std::string(data, out_len) != expected) {
          ++out.payload_mismatch;
        } else {
          out.verified_hit_bytes += out_len;
        }
      }
      continue;
    }
    if (read == Status::kIOError) {
      ++out.range_io_error;
      continue;
    }
    if (read != Status::kNotFound) {
      ++out.other_range_error;
      continue;
    }
    ++out.misses;
    const uint64_t next_epoch = (*epochs)[object] + 1;
    FillValue(QualityZipfValueId(object, next_epoch), buffer.data(), kValueBytes);
    const uint64_t put_begin = QualityNowNs();
    const Status put = store->CacheDirect(key, buffer.data(), kValueBytes, buffer.size());
    const uint64_t put_end = QualityNowNs();
    out.admission_calls.push_back(
        {put_begin, put_end, static_cast<int32_t>(put)});
    out.admissions.Add(put);
    if (put == Status::kOk) (*epochs)[object] = next_epoch;
  }
  out.hit_latency = QualityLatencyForStatus(out.read_calls, Status::kOk);
  out.miss_latency = QualityLatencyForStatus(out.read_calls, Status::kNotFound);
  out.admission_latency =
      QualityLatencyForStatus(out.admission_calls, Status::kOk);
  out.wall_seconds = Seconds(Clock::now() - wall_start);
  return out;
}

void QualityZipfPhaseJson(std::ostringstream* out, const std::string& prefix,
                          const QualityZipfPhase& phase) {
  *out << ',' << JsonString(prefix + "_requests") << ':' << phase.requests
       << ',' << JsonString(prefix + "_allocation_failures") << ':'
       << phase.allocation_failures
       << ',' << JsonString(prefix + "_actual_gets")
       << ':' << phase.hits + phase.misses + phase.range_io_error +
                    phase.other_range_error
       << ',' << JsonString(prefix + "_hits") << ':' << phase.hits
       << ',' << JsonString(prefix + "_misses") << ':' << phase.misses
       << ',' << JsonString(prefix + "_range_io_error") << ':'
       << phase.range_io_error
       << ',' << JsonString(prefix + "_other_range_error") << ':'
       << phase.other_range_error
       << ',' << JsonString(prefix + "_out_length_mismatch") << ':'
       << phase.out_length_mismatch
       << ',' << JsonString(prefix + "_value_length_mismatch") << ':'
       << phase.value_length_mismatch
       << ',' << JsonString(prefix + "_payload_mismatch") << ':'
       << phase.payload_mismatch
       << ',' << JsonString(prefix + "_verified_hit_bytes") << ':'
       << phase.verified_hit_bytes
       << ',' << JsonString(prefix + "_wall_seconds") << ':' << phase.wall_seconds;
  QualityPutJson(out, prefix + "_admission_put", phase.admissions);
  QualityLatencyJson(out, prefix + "_hit_range_direct_latency", phase.hit_latency);
  QualityLatencyJson(out, prefix + "_miss_range_direct_latency", phase.miss_latency);
  QualityLatencyJson(out, prefix + "_admission_cache_direct_latency",
                     phase.admission_latency);
  QualitySamplesJson(out, prefix + "_range_direct_samples", phase.read_calls);
  QualitySamplesJson(out, prefix + "_admission_cache_direct_samples",
                     phase.admission_calls);
}

struct QualityZipfResult {
  uint32_t pair = 0;
  uint32_t order = 0;
  EngineKind kind = EngineKind::kSlab;
  bool valid = false;
  std::vector<std::string> violations;
  uint64_t evictions = 0;
  double hit_fraction = 0;
  double hit_p50_ns = 0;
  double hit_p99_ns = 0;
  double miss_p50_ns = 0;
  double miss_p99_ns = 0;
  double admission_p50_ns = 0;
  double admission_p99_ns = 0;
};

QualityZipfResult RunQualityZipf(EngineKind kind, uint32_t pair, uint32_t order,
                                 const fs::path& dir, const QualityTrace& trace,
                                 JsonlReporter* reporter) {
  QualityZipfResult result;
  result.pair = pair;
  result.order = order;
  result.kind = kind;
  QualityStore opened = OpenQualityStore(kind, dir);
  if (!opened.fresh_directory || !opened.open_ok) {
    result.violations.push_back("fresh directory or healthy direct open failed");
  } else if (!QualityDirectActive(kind, opened)) {
    result.violations.push_back("requested O_DIRECT is not active");
  }
  std::vector<uint64_t> epochs(kQualityZipfKeys, 0);
  QualityZipfPhase warmup, measured;
  const uint64_t evictions_before =
      opened.store == nullptr ? 0 : opened.store->Evictions();
  if (opened.open_ok && QualityDirectActive(kind, opened)) {
    // Each directory is fresh and empty.  Warm-up is declared full-keyspace
    // permutation initialization; only the following vector is Zipf measurement.
    // One synchronous worker consumes the exact request vectors supplied here.
    warmup = RunQualityZipfPhase(opened.store.get(), trace, trace.warmup, &epochs);
    measured = RunQualityZipfPhase(opened.store.get(), trace, trace.measured, &epochs);
  }
  result.evictions =
      opened.store == nullptr ? 0 : opened.store->Evictions() - evictions_before;
  const QualityFinalStats stats = QualityStatsOf(kind, opened);
  if (opened.store != nullptr && !opened.store->Healthy())
    result.violations.push_back("engine became unhealthy");
  const bool close_ok = opened.store != nullptr && opened.store->Close();
  if (!close_ok) result.violations.push_back("final Close failed");
  opened.store.reset();

  for (const auto* phase : {&warmup, &measured}) {
    const uint64_t gets = phase->hits + phase->misses + phase->range_io_error +
                          phase->other_range_error;
    if (gets != phase->requests || phase->read_calls.size() != phase->requests ||
        phase->admission_calls.size() != phase->admissions.Calls() ||
        phase->allocation_failures != 0 || phase->range_io_error != 0 ||
        phase->other_range_error != 0 ||
        phase->out_length_mismatch != 0 ||
        phase->value_length_mismatch != 0 || phase->payload_mismatch != 0 ||
        phase->verified_hit_bytes != phase->hits * kValueBytes ||
        phase->hit_latency.calls != phase->hits ||
        phase->miss_latency.calls != phase->misses ||
        phase->admission_latency.calls != phase->admissions.ok ||
        phase->admissions.UnexpectedForChurn() != 0) {
      result.violations.push_back("Zipf request, payload, latency, or status invariant failed");
    }
  }
  if (measured.hits == 0 || measured.misses == 0 || measured.admissions.Calls() == 0 ||
      measured.admissions.ok == 0 || warmup.misses == 0 || result.evictions == 0) {
    result.violations.push_back("Zipf trace produced no representative hit/miss/admission/eviction");
  }
  if (!trace.valid) result.violations.push_back("fixed Zipf trace coverage is not representative");
  if (!QualityFinalStatsClean(kind, stats))
    result.violations.push_back("direct-I/O, CRC, or metadata error counter is nonzero");
  result.valid = result.violations.empty();
  result.hit_fraction =
      measured.requests == 0 ? 0.0
                            : static_cast<double>(measured.hits) / measured.requests;
  result.hit_p50_ns = measured.hit_latency.p50_ns;
  result.hit_p99_ns = measured.hit_latency.p99_ns;
  result.miss_p50_ns = measured.miss_latency.p50_ns;
  result.miss_p99_ns = measured.miss_latency.p99_ns;
  result.admission_p50_ns = measured.admission_latency.p50_ns;
  result.admission_p99_ns = measured.admission_latency.p99_ns;

  std::ostringstream out;
  out << std::setprecision(17) << std::boolalpha
      << "{\"record\":\"zipf_eviction\",\"schema_version\":1,\"pair\":" << pair
      << ",\"engine_order\":" << order
      << ",\"engine\":" << JsonString(EngineName(kind))
      << ",\"trace_seed\":" << trace.seed << ",\"trace_fnv1a64\":\""
      << std::hex << std::setw(16) << std::setfill('0') << trace.hash << std::dec
      << "\",\"trace_key_base\":" << trace.key_base
      << ",\"zipf_exponent\":0.9,\"zipf_keyspace\":" << kQualityZipfKeys
      << ",\"warmup_unique_keys\":" << trace.warmup_unique
      << ",\"warmup_is_full_keyspace_initialization\":true"
      << ",\"trace_value_transform\":\"value_base+epoch*keyspace+object\""
      << ",\"trace_value_base\":" << kQualityZipfValueBase
      << ",\"warmup_then_measured_same_epoch_table\":true"
      << ",\"one_synchronous_worker\":true" << ",\"zipf_phase_evictions\":"
      << result.evictions << ",\"close_ok\":" << close_ok;
  QualityZipfPhaseJson(&out, "warmup", warmup);
  QualityZipfPhaseJson(&out, "measured", measured);
  QualityFinalStatsJson(&out, stats);
  out << ",\"result\":" << JsonString(result.valid ? "VALID" : "FAIL")
      << ",\"violations\":" << JsonStrings(result.violations) << '}';
  reporter->Emit(out.str());
  if (!reporter->good()) {
    result.violations.push_back("zipf JSONL write failed");
    result.valid = false;
  }
  return result;
}

void QualitySeriesJson(std::ostringstream* out, const std::string& name,
                       std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const double minimum = values.empty() ? 0 : values.front();
  const double maximum = values.empty() ? 0 : values.back();
  const double median = values.empty() ? 0 : values[values.size() / 2];
  *out << ',' << JsonString(name + "_samples") << ':' << values.size()
       << ',' << JsonString(name + "_median") << ':' << median
       << ',' << JsonString(name + "_min") << ':' << minimum
       << ',' << JsonString(name + "_max") << ':' << maximum
       << ',' << JsonString(name + "_spread") << ':' << maximum - minimum;
}

TEST(LseM3Quality, SteadyIoAndZipfEviction) {
  const char* const root_text = std::getenv("DFKV_LSE_M3_IO_ROOT");
  if (root_text == nullptr || *root_text == '\0') {
    EmitStdout(
        "{\"record\":\"io_preflight\",\"schema_version\":1,"
        "\"result\":\"SKIP\",\"reason\":\"DFKV_LSE_M3_IO_ROOT is required\"}");
    GTEST_SKIP() << "set DFKV_LSE_M3_IO_ROOT to a fresh absolute real-NVMe "
                    "directory satisfying the existing no-symlink/space checks";
  }
  RootEvidence root_evidence;
  std::string root_error;
  if (!PrepareFreshNvmeRoot(fs::path(root_text), &root_evidence, &root_error)) {
    PreflightFailure(root_error, &root_evidence);
    return;
  }
  const fs::path root = root_evidence.normalized_root;
  const fs::path probe = root / "io_direct_io.probe";
  std::string probe_error;
  if (!ProbeDirectIo(probe, &probe_error)) {
    PreflightFailure("quality O_DIRECT probe failed: " + probe_error, &root_evidence);
    return;
  }
  JsonlReporter reporter(root / kQualityJsonlName);
  std::ostringstream configuration;
  configuration << std::setprecision(17) << std::boolalpha
                << "{\"record\":\"io_configuration\",\"schema_version\":1"
                << ",\"root\":" << JsonString(root.string())
                << ",\"backing_device\":\"" << root_evidence.device_major << ':'
                << root_evidence.device_minor << '"'
                << ",\"backing_candidates\":" << BackingCandidatesJson(root_evidence)
                << ",\"backing_all_leaves_nvme\":true"
                << ",\"direct_io_probe_ok\":true"
                << ",\"capacity_bytes\":" << kStandardCapacity
                << ",\"value_bytes\":" << kValueBytes
                << ",\"slab_granularity_bytes\":" << kSlabGranularity
                << ",\"slab_warm_restart\":\"ordinary_default_kOff\""
                << ",\"slab_table_sync_ms\":100,\"slab_reclaim_interval_ms\":50"
                << ",\"steady_writers\":" << kWriters
                << ",\"cold_fill_calls\":" << kQualityObjectsPerCapacity
                << ",\"steady_calls_per_engine_pair\":"
                << 2 * kQualityObjectsPerCapacity
                << ",\"write_time_definition\":"
                   "\"union of eight CacheDirect intervals; generation excluded\""
                << ",\"read_time_definition\":"
                   "\"union of full-whole-value RangeDirect intervals\""
                << ",\"pairs_required\":3"
                << ",\"zipf_exponent\":0.9,\"zipf_keyspace\":" << kQualityZipfKeys
                << ",\"zipf_warmup_requests\":" << kQualityWarmupRequests
                << ",\"zipf_measured_requests\":" << kQualityMeasuredRequests
                << ",\"warmup_recipe\":"
                   "\"seeded Fisher-Yates permutation containing every key exactly once\""
                << ",\"sequence_semantics\":"
                   "\"(warmup by seeded full-keyspace permutation) -> (measured by independent Zipf(0.9))\""
                << ",\"trace_recipe\":"
                   "\"SplitMix64 top53 / 2^53; lower_bound in normalized N weights i^-0.9; measured RNG independent of shuffle\""
                << ",\"trace_seed_base\":" << kQualityTraceSeedBase
                << ",\"trace_epoch_value\":"
                   "\"value_base+epoch*keyspace+object; epoch advances only on admitted miss\""
                << ",\"sample_clock\":\"steady_clock time_since_epoch in nanoseconds\""
                << ",\"sample_tuple\":\"[begin_ns,end_ns,actual Status enum value]\""
                << ",\"status_enum\":{\"kOk\":"
                << static_cast<int32_t>(Status::kOk) << ",\"kNotFound\":"
                << static_cast<int32_t>(Status::kNotFound) << ",\"kCacheFull\":"
                << static_cast<int32_t>(Status::kCacheFull) << ",\"kQuotaExceeded\":"
                << static_cast<int32_t>(Status::kQuotaExceeded) << ",\"kIOError\":"
                << static_cast<int32_t>(Status::kIOError) << ",\"kInvalid\":"
                << static_cast<int32_t>(Status::kInvalid)
                << ",\"kResourceExhausted\":"
                << static_cast<int32_t>(Status::kResourceExhausted)
                << ",\"kNoCompatibleRail\":"
                << static_cast<int32_t>(Status::kNoCompatibleRail) << '}'
                << ",\"percentile_rule\":"
                   "\"nearest-rank: sort end_ns-begin_ns ascending; zero-based index ceil(N*p/100)-1\""
                << ",\"api_union_rule\":"
                   "\"sort by begin_ns/end_ns; merge overlapping/touching intervals; sum union durations\""
                << ",\"admission_latency_rule\":"
                   "\"kOk admissions only; every refused PUT remains separately visible in raw all-attempt samples\""
                << ",\"jsonl\":" << JsonString(reporter.path().string())
                << ",\"performance_comparison_threshold\":\"none\"}";
  reporter.Emit(configuration.str());

  std::vector<QualityIoResult> io_results;
  std::vector<QualityZipfResult> zipf_results;
  bool traces_valid = true;
  bool infrastructure_ok = reporter.good();
  for (uint32_t pair = 0; pair < kStandardRepetitions; ++pair) {
    const fs::path pair_dir = root / ("pair_" + ZeroPadding(pair));
    std::error_code ec;
    const bool pair_ok = fs::create_directory(pair_dir, ec) && !ec;
    const bool steady_ok = pair_ok && fs::create_directory(pair_dir / "steady", ec) && !ec;
    const bool zipf_ok = steady_ok && fs::create_directory(pair_dir / "zipf", ec) && !ec;
    if (!zipf_ok) infrastructure_ok = false;
    const QualityTrace trace = MakeQualityTrace(pair);
    traces_valid = traces_valid && trace.valid;
    std::ostringstream trace_line;
    trace_line << std::boolalpha << "{\"record\":\"zipf_trace\",\"schema_version\":1"
               << ",\"pair\":" << pair << ",\"seed\":" << trace.seed
               << ",\"key_base\":" << trace.key_base
               << ",\"warmup_requests\":" << trace.warmup.size()
               << ",\"warmup_unique_keys\":" << trace.warmup_unique
               << ",\"warmup_recipe\":"
                  "\"Fisher-Yates full-keyspace permutation; SplitMix64 top53/2^53; selected=floor(u*remaining)\""
               << ",\"measured_recipe\":"
                  "\"independent Zipf(0.9) inverse-CDF; separate RNG advanced by old warmup draw count\""
               << ",\"semantic_recipe\":"
                  "\"(warmup by seeded full-keyspace permutation) -> (measured by independent Zipf(0.9))\""
               << ",\"measured_requests\":" << trace.measured.size()
               << ",\"measured_unique_keys\":" << trace.measured_unique
               << ",\"all_unique_keys\":" << trace.all_unique
               << ",\"fnv1a64\":\"" << std::hex << std::setw(16)
               << std::setfill('0') << trace.hash << std::dec << '"'
               << ",\"measured_fnv1a64\":\"" << std::hex << std::setw(16)
               << std::setfill('0') << trace.measured_hash << std::dec << '"'
               << ",\"same_vector_supplied_to_both_engines\":true"
               << ",\"valid\":" << trace.valid << '}';
    reporter.Emit(trace_line.str());
    if (!zipf_ok) continue;
    const EngineKind first = pair % 2 == 0 ? EngineKind::kSlab : EngineKind::kLog;
    const EngineKind second = first == EngineKind::kSlab ? EngineKind::kLog
                                                         : EngineKind::kSlab;
    int order = 0;
    for (const EngineKind kind : {first, second}) {
      io_results.push_back(RunQualityIo(kind, pair, order,
                                        pair_dir / "steady" / EngineName(kind), &reporter));
      zipf_results.push_back(RunQualityZipf(kind, pair, order,
                                            pair_dir / "zipf" / EngineName(kind), trace,
                                            &reporter));
      ++order;
    }
  }

  bool all_valid = infrastructure_ok && traces_valid && reporter.good() &&
                   io_results.size() == 2 * kStandardRepetitions &&
                   zipf_results.size() == 2 * kStandardRepetitions;
  for (const auto& result : io_results) all_valid = all_valid && result.valid;
  for (const auto& result : zipf_results) all_valid = all_valid && result.valid;
  std::ostringstream aggregate;
  aggregate << std::setprecision(17) << std::boolalpha
            << "{\"record\":\"io_aggregate\",\"schema_version\":1"
            << ",\"actual_pairs\":" << io_results.size() / 2
            << ",\"steady_runs\":" << io_results.size()
            << ",\"zipf_runs\":" << zipf_results.size()
            << ",\"performance_gate\":\"none; invariant validity only\"";
  for (const EngineKind kind : {EngineKind::kSlab, EngineKind::kLog}) {
    const std::string prefix = std::string(kind == EngineKind::kSlab ? "slab" : "log");
    std::vector<double> write_rate, write_p50, write_p99, read_rate, read_p50, read_p99;
    std::vector<double> hit_fraction, hit_p50, hit_p99, miss_p50, miss_p99;
    std::vector<double> admission_p50, admission_p99;
    for (const auto& result : io_results) {
      if (result.kind != kind) continue;
      write_rate.push_back(result.write_mib_s);
      write_p50.push_back(result.write_p50_ns);
      write_p99.push_back(result.write_p99_ns);
      read_rate.push_back(result.read_mib_s);
      read_p50.push_back(result.read_p50_ns);
      read_p99.push_back(result.read_p99_ns);
    }
    for (const auto& result : zipf_results) {
      if (result.kind != kind) continue;
      hit_fraction.push_back(result.hit_fraction);
      hit_p50.push_back(result.hit_p50_ns);
      hit_p99.push_back(result.hit_p99_ns);
      miss_p50.push_back(result.miss_p50_ns);
      miss_p99.push_back(result.miss_p99_ns);
      admission_p50.push_back(result.admission_p50_ns);
      admission_p99.push_back(result.admission_p99_ns);
    }
    aggregate << ',' << JsonString(prefix + "_actual_runs") << ':'
              << write_rate.size();
    QualitySeriesJson(&aggregate, prefix + "_steady_write_mib_per_second", write_rate);
    QualitySeriesJson(&aggregate, prefix + "_steady_write_p50_ns", write_p50);
    QualitySeriesJson(&aggregate, prefix + "_steady_write_p99_ns", write_p99);
    QualitySeriesJson(&aggregate, prefix + "_resident_read_mib_per_second", read_rate);
    QualitySeriesJson(&aggregate, prefix + "_resident_read_p50_ns", read_p50);
    QualitySeriesJson(&aggregate, prefix + "_resident_read_p99_ns", read_p99);
    QualitySeriesJson(&aggregate, prefix + "_zipf_hit_fraction", hit_fraction);
    QualitySeriesJson(&aggregate, prefix + "_zipf_hit_p50_ns", hit_p50);
    QualitySeriesJson(&aggregate, prefix + "_zipf_hit_p99_ns", hit_p99);
    QualitySeriesJson(&aggregate, prefix + "_zipf_miss_p50_ns", miss_p50);
    QualitySeriesJson(&aggregate, prefix + "_zipf_miss_p99_ns", miss_p99);
    QualitySeriesJson(&aggregate, prefix + "_zipf_admission_p50_ns", admission_p50);
    QualitySeriesJson(&aggregate, prefix + "_zipf_admission_p99_ns", admission_p99);
  }
  aggregate << ",\"validity\":" << JsonString(all_valid ? "PASS" : "FAIL") << '}';
  reporter.Emit(aggregate.str());
  if (!all_valid || !reporter.good()) {
    ADD_FAILURE() << "LSE M3 I/O evidence validity failed; see " << reporter.path();
  }
}

TEST(LseM3Capacity, EffectiveCapacity274MiB) {
  const char* const root_text = std::getenv("DFKV_LSE_M3_BENCH_ROOT");
  if (root_text == nullptr || *root_text == '\0') {
    EmitStdout(
        "{\"record\":\"preflight\",\"schema_version\":1,\"result\":\"SKIP\","
        "\"reason\":\"DFKV_LSE_M3_BENCH_ROOT is required\"}");
    GTEST_SKIP() << "set DFKV_LSE_M3_BENCH_ROOT to a new absolute directory; "
                    "every parent component must be a real directory (no "
                    "symlinks), its mounted backing must resolve to NVMe, and "
                    "at least 400 GiB must be free";
  }

  Config config;
  config.root = fs::path(root_text);
  std::string config_error;
  if (!config.root.is_absolute()) config_error = "benchmark root must be absolute";
  if (const char* repetitions = std::getenv("DFKV_LSE_M3_REPETITIONS")) {
    uint64_t parsed = 0;
    if (!ParseUnsigned(repetitions, &parsed) || (parsed != 1 && parsed != 3)) {
      config_error = "DFKV_LSE_M3_REPETITIONS must be exactly 1 (development "
                     "smoke, INCONCLUSIVE) or 3 (acceptance)";
    } else {
      config.repetitions = static_cast<uint32_t>(parsed);
    }
  }
  if (const char* capacity = std::getenv("DFKV_LSE_M3_CAP_BYTES")) {
    config.capacity_override = true;
    if (!ParseUnsigned(capacity, &config.capacity) || config.capacity == 0 ||
        config.capacity % kSlabExtent != 0) {
      config_error = "DFKV_LSE_M3_CAP_BYTES must be a positive 1 GiB multiple";
    }
  }
  if (config.capacity > (std::numeric_limits<uint64_t>::max() - kValueBytes + 1) /
                            2) {
    config_error = "capacity overflows the 2*capacity PUT calculation";
  }
  config.puts_per_run =
      (2 * config.capacity + kValueBytes - 1) / kValueBytes;
  if (config.puts_per_run > std::numeric_limits<uint64_t>::max() /
                                config.repetitions) {
    config_error = "key id calculation would overflow";
  }
  // Direct store options do not call LogStore::ApplyEnvironment.  Slab does
  // read its watermark environment in the constructor, so reject absolute
  // overrides rather than silently benchmarking a different configuration.
  for (const char* name : {"DFKV_SLAB_EVICT_HEADROOM_BYTES",
                           "DFKV_SLAB_EVICT_BAND_BYTES"}) {
    const char* const value = std::getenv(name);
    if (value != nullptr) {
      uint64_t parsed = 0;
      if (!ParseUnsigned(value, &parsed) || parsed != 0) {
        config_error = std::string(name) +
                       " must be unset or zero: absolute slab watermark "
                       "overrides are forbidden for this comparison";
      }
    }
  }
  if (!config_error.empty()) {
    PreflightFailure(config_error);
    return;
  }
  config.acceptance_eligible = config.repetitions == kStandardRepetitions &&
                               !config.capacity_override;

  RootEvidence root_evidence;
  std::string root_error;
  if (!PrepareFreshNvmeRoot(config.root, &root_evidence, &root_error)) {
    PreflightFailure(root_error, &root_evidence);
    return;
  }
  config.root = root_evidence.normalized_root;
  std::error_code ec;

  JsonlReporter reporter(config.root / kJsonlName);
  std::ostringstream configuration;
  configuration << std::boolalpha
                << "{\"record\":\"configuration\",\"schema_version\":1"
                << ",\"supplied_root\":"
                << JsonString(root_evidence.supplied_root.string())
                << ",\"root\":" << JsonString(config.root.string())
                << ",\"normalized_parent\":"
                << JsonString(root_evidence.normalized_parent.string())
                << ",\"parent_component_walk\":"
                   "\"openat O_PATH|O_DIRECTORY|O_NOFOLLOW\""
                << ",\"root_creation\":\"mkdirat in validated parent descriptor\""
                << ",\"backing_device\":\""
                << root_evidence.device_major << ':' << root_evidence.device_minor
                << '"'
                << ",\"sysfs_dev_block\":"
                << JsonString(root_evidence.sysfs_dev_block)
                << ",\"resolved_device\":"
                << JsonString(root_evidence.resolved_device)
                << ",\"nvme_subsystem_link\":"
                << JsonString(root_evidence.nvme_subsystem_link)
                << ",\"nvme_subsystem\":"
                << JsonString(root_evidence.nvme_subsystem)
                << ",\"nvme_controller_link\":"
                << JsonString(root_evidence.nvme_controller_link.string())
                << ",\"nvme_controller_device\":"
                << JsonString(root_evidence.nvme_controller_device.string())
                << ",\"backing_physical_leaf_count\":"
                << root_evidence.physical_leaf_count
                << ",\"backing_all_leaves_nvme\":"
                << root_evidence.all_leaves_nvme
                << ",\"backing_candidates\":"
                << BackingCandidatesJson(root_evidence)
                << ",\"backing_nvme_verified\":true"
                << ",\"filesystem_available_bytes\":"
                << root_evidence.available_bytes
                << ",\"minimum_free_bytes\":" << kRequiredFreeBytes
                << ",\"value_bytes\":" << kValueBytes
                << ",\"capacity_bytes\":" << config.capacity
                << ",\"puts_per_engine_run\":" << config.puts_per_run
                << ",\"writers\":" << kWriters
                << ",\"repetitions\":" << config.repetitions
                << ",\"acceptance_eligible\":" << config.acceptance_eligible
                << ",\"development_smoke\":" << (config.repetitions == 1)
                << ",\"capacity_override\":" << config.capacity_override
                << ",\"standard_segments\":64"
                << ",\"log_segment_bytes\":" << kLogSegment
                << ",\"log_buffered_direct_fallback_allowed\":false"
                << ",\"slab_granularity_bytes\":" << kSlabGranularity
                << ",\"slab_extent_bytes\":" << kSlabExtent
                << ",\"slab_watermark_high_pct\":92"
                << ",\"slab_watermark_low_pct\":88"
                << ",\"slab_absolute_watermark_override\":false"
                << ",\"slab_balance\":false"
                << ",\"jsonl\":" << JsonString(reporter.path().string())
                << ",\"head_provenance\":\"exact HEAD reported separately by runner\""
                << '}';
  reporter.Emit(configuration.str());
  if (!reporter.good()) {
    FAIL() << "cannot create/write " << reporter.path();
  }

  std::vector<RunRecord> runs;
  std::string infrastructure_error;
  for (uint32_t pair = 0; pair < config.repetitions; ++pair) {
    const fs::path pair_dir = config.root / ("pair_" + ZeroPadding(pair));
    if (!fs::create_directory(pair_dir, ec) || ec) {
      infrastructure_error = "cannot create fresh pair directory " +
                             pair_dir.string() + ": " +
                             (ec ? ec.message() : "already exists");
      break;
    }
    const fs::path probe = pair_dir / "direct_io.probe";
    std::string probe_error;
    const bool direct_ok = ProbeDirectIo(probe, &probe_error);
    std::ostringstream probe_record;
    probe_record << std::boolalpha
                 << "{\"record\":\"direct_io_probe\",\"schema_version\":1"
                 << ",\"pair\":" << pair
                 << ",\"path\":" << JsonString(probe.string())
                 << ",\"ok\":" << direct_ok
                 << ",\"error\":" << JsonString(probe_error) << '}';
    reporter.Emit(probe_record.str());
    if (!direct_ok) {
      infrastructure_error = "O_DIRECT acceptance probe failed: " + probe_error;
      break;
    }

    const EngineKind first = pair % 2 == 0 ? EngineKind::kSlab : EngineKind::kLog;
    const EngineKind second = first == EngineKind::kSlab ? EngineKind::kLog
                                                         : EngineKind::kSlab;
    runs.push_back(RunOne(first, pair, 0, config, direct_ok));
    reporter.Emit(RunJson(runs.back(), config));
    runs.push_back(RunOne(second, pair, 1, config, direct_ok));
    reporter.Emit(RunJson(runs.back(), config));
  }

  struct PairResult {
    uint32_t pair = 0;
    const RunRecord* slab = nullptr;
    const RunRecord* log = nullptr;
    double delta = 0;
    bool valid = false;
  };
  std::vector<PairResult> pairs;
  std::vector<double> slab_fractions;
  std::vector<double> log_fractions;
  std::vector<double> deltas;
  for (uint32_t pair = 0; pair < config.repetitions; ++pair) {
    const RunRecord* const slab = FindRun(runs, EngineKind::kSlab, pair);
    const RunRecord* const log = FindRun(runs, EngineKind::kLog, pair);
    if (slab == nullptr || log == nullptr) continue;
    PairResult result;
    result.pair = pair;
    result.slab = slab;
    result.log = log;
    result.delta = log->Fraction() - slab->Fraction();
    result.valid = slab->valid && log->valid;
    slab_fractions.push_back(slab->Fraction());
    log_fractions.push_back(log->Fraction());
    deltas.push_back(result.delta);
    pairs.push_back(result);
  }
  const double slab_spread = Spread(slab_fractions);
  const double log_spread = Spread(log_fractions);
  const double max_engine_spread = std::max(slab_spread, log_spread);
  std::vector<double> sorted_deltas = deltas;
  std::sort(sorted_deltas.begin(), sorted_deltas.end());
  const double median_delta = sorted_deltas.empty()
                                  ? 0
                                  : sorted_deltas[sorted_deltas.size() / 2];
  const double min_delta = sorted_deltas.empty() ? 0 : sorted_deltas.front();
  // Emit each pair only after all runs so it carries both its own D and the
  // complete within-engine spread used by the three-run decision.
  for (const auto& pair : pairs) {
    const EngineKind first = pair.pair % 2 == 0 ? EngineKind::kSlab
                                                : EngineKind::kLog;
    std::ostringstream line;
    line << std::setprecision(17) << std::boolalpha
         << "{\"record\":\"pair\",\"schema_version\":1"
         << ",\"pair\":" << pair.pair
         << ",\"engine_order\":[" << JsonString(EngineName(first)) << ','
         << JsonString(EngineName(first == EngineKind::kSlab ? EngineKind::kLog
                                                            : EngineKind::kSlab))
         << ']'
         << ",\"slab_fraction\":" << pair.slab->Fraction()
         << ",\"log_fraction\":" << pair.log->Fraction()
         << ",\"D\":" << pair.delta
         << ",\"slab_spread\":" << slab_spread
         << ",\"log_spread\":" << log_spread
         << ",\"max_engine_spread\":" << max_engine_spread
         << ",\"D_gt_max_engine_spread\":" << (pair.delta > max_engine_spread)
         << ",\"result\":" << JsonString(pair.valid ? "VALID" : "FAIL")
         << '}';
    reporter.Emit(line.str());
  }

  const size_t valid_pairs = static_cast<size_t>(std::count_if(
      pairs.begin(), pairs.end(), [](const PairResult& pair) { return pair.valid; }));
  const bool all_run_constraints =
      runs.size() == static_cast<size_t>(2 * config.repetitions) &&
      valid_pairs == pairs.size() && pairs.size() == config.repetitions;
  const bool three_actual_pairs = config.repetitions == kStandardRepetitions &&
                                  pairs.size() == kStandardRepetitions &&
                                  valid_pairs == kStandardRepetitions;
  const bool median_gate = median_delta >= 0.10;
  // Requiring every pair's D above the largest same-engine spread is stronger
  // than testing only the median against that spread; it prevents one favorable
  // outlier from turning noisy repetitions into a PASS.
  const bool separation_gate = !deltas.empty() && min_delta > max_engine_spread;
  std::vector<std::string> reasons;
  if (!infrastructure_error.empty()) reasons.push_back(infrastructure_error);
  if (config.repetitions == 1)
    reasons.push_back("one-pair development smoke is always INCONCLUSIVE");
  if (config.capacity_override)
    reasons.push_back("capacity override is a non-acceptance development pilot");
  if (!all_run_constraints)
    reasons.push_back("one or more engine runs/pairs failed raw constraints");
  if (config.repetitions == kStandardRepetitions && !three_actual_pairs)
    reasons.push_back("three valid actual pairs are required");
  if (!median_gate) reasons.push_back("median(D) is below 0.10");
  if (!separation_gate)
    reasons.push_back("not every D exceeds max(engine spread)");
  if (!reporter.good()) reasons.push_back("JSONL evidence output failed");

  const bool hard_failure = !infrastructure_error.empty() ||
                            !all_run_constraints || !reporter.good();
  const bool pass = config.acceptance_eligible && three_actual_pairs &&
                    all_run_constraints && median_gate && separation_gate &&
                    reporter.good();
  const std::string verdict = pass ? "PASS" : hard_failure ? "FAIL" : "INCONCLUSIVE";
  const std::string acceptance_verdict =
      config.acceptance_eligible ? verdict : "INCONCLUSIVE";
  std::ostringstream aggregate;
  aggregate << std::setprecision(17) << std::boolalpha
            << "{\"record\":\"aggregate\",\"schema_version\":1"
            << ",\"value_bytes\":" << kValueBytes
            << ",\"capacity_bytes\":" << config.capacity
            << ",\"repetitions_requested\":" << config.repetitions
            << ",\"run_records\":" << runs.size()
            << ",\"pair_records\":" << pairs.size()
            << ",\"valid_pairs\":" << valid_pairs
            << ",\"three_actual_valid_pairs\":" << three_actual_pairs
            << ",\"slab_spread\":" << slab_spread
            << ",\"log_spread\":" << log_spread
            << ",\"max_engine_spread\":" << max_engine_spread
            << ",\"median_D\":" << median_delta
            << ",\"min_D\":" << min_delta
            << ",\"median_D_gte_0_10\":" << median_gate
            << ",\"every_D_gt_max_engine_spread\":" << separation_gate
            << ",\"acceptance_eligible\":" << config.acceptance_eligible
            << ",\"head_provenance\":\"exact HEAD reported separately by runner\""
            << ",\"verdict\":" << JsonString(verdict)
            << ",\"acceptance_verdict\":" << JsonString(acceptance_verdict)
            << ",\"reasons\":" << JsonStrings(reasons) << '}';
  reporter.Emit(aggregate.str());

  if (verdict != "PASS" || !reporter.good()) {
    ADD_FAILURE() << "LSE M3 effective-capacity verdict "
                  << (reporter.good() ? verdict : "FAIL") << " ("
                  << (!reporter.good()
                          ? "aggregate JSONL evidence write failed"
                          : reasons.empty() ? "no reason recorded" : reasons.front())
                  << "); evidence path: " << reporter.path();
  }
}

}  // namespace
