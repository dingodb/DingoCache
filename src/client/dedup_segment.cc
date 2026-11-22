#include "client/dedup_segment.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>

#include "utils/log.h"

namespace dfkv {
namespace dedup_segment {

namespace {

constexpr int kInitWaitMs = 1000;

std::atomic<uint32_t>* Field(void* base, size_t off) {
  return reinterpret_cast<std::atomic<uint32_t>*>(static_cast<char*>(base) +
                                                  off);
}

bool Dead(uint32_t pid) {
  return ::kill(static_cast<pid_t>(pid), 0) != 0 && errno == ESRCH;
}

// Sizes and backs the segment. tmpfs allocates lazily: posix_fallocate turns
// a later SIGBUS mid-publish (a full /dev/shm) into a clean failure here.
// Both are idempotent, so a process finishing a dead creator's work repeats
// them safely.
int SizeAndBack(int fd, size_t len) {
  if (::ftruncate(fd, static_cast<off_t>(len)) != 0) return errno;
  return ::posix_fallocate(fd, 0, static_cast<off_t>(len));
}

void* Map(int fd, size_t len) {
  void* base = ::mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  return base == MAP_FAILED ? nullptr : base;
}

// Waits for init_done on a segment of the right size. True: laid out.
bool WaitLaidOut(void* base, const Layout& l) {
  for (int spin = 0; spin < kInitWaitMs; ++spin) {
    if (Field(base, l.init_done_off)->load(std::memory_order_acquire) == 1)
      return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

}  // namespace

void* Attach(const std::string& name, const Layout& l, const char* tag,
             bool* creator) {
  *creator = false;
  const std::string prefix = std::string(tag) + ": ";
  int fd = ::shm_open(name.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
  if (fd >= 0) {
    // Name the creator first: the segment then has a size (creator_off + 4)
    // that tells an opener a creation is under way, and a pid it can check.
    const uint32_t me = static_cast<uint32_t>(::getpid());
    int rc = ::pwrite(fd, &me, sizeof(me), static_cast<off_t>(l.creator_off)) ==
                     static_cast<ssize_t>(sizeof(me))
                 ? 0
                 : errno;
    if (rc == 0) rc = SizeAndBack(fd, l.len);
    void* base = rc == 0 ? Map(fd, l.len) : nullptr;
    ::close(fd);
    if (!base) {
      DFKV_LOG_INFO(prefix + "cannot create a " + std::to_string(l.len >> 20) +
                    " MiB segment in /dev/shm (" + std::to_string(rc) +
                    "), running without");
      ::shm_unlink(name.c_str());  // don't leave a half-backed segment
      return nullptr;
    }
    *creator = true;
    return base;
  }
  if (errno != EEXIST) return nullptr;
  fd = ::shm_open(name.c_str(), O_RDWR, 0600);
  if (fd < 0) return nullptr;
  const size_t naming = l.creator_off + sizeof(uint32_t);
  void* base = nullptr;
  for (int spin = 0; spin <= kInitWaitMs && !base; ++spin) {
    struct stat st{};
    if (::fstat(fd, &st) != 0) break;
    const size_t size = static_cast<size_t>(st.st_size);
    if (size == l.len) {
      base = Map(fd, l.len);
      break;
    }
    // 0 or `naming`: a creation is under way. Anything else is a segment of
    // another layout.
    if (size != 0 && size != naming) {
      DFKV_LOG_INFO(prefix + "existing segment layout mismatch, running without");
      ::close(fd);
      return nullptr;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (base && WaitLaidOut(base, l)) {
    ::close(fd);
    return base;
  }
  // Unfinished after the wait. Finish it if its creator is dead, or if it is
  // still empty: a creator is between its create and its pid write for a
  // system call, not for the whole wait, so an empty segment's creator was
  // killed there (older releases adopted it as the creator of a new one).
  struct stat st{};
  const bool empty = ::fstat(fd, &st) == 0 && st.st_size == 0;
  uint32_t owner = 0;
  if (::pread(fd, &owner, sizeof(owner), static_cast<off_t>(l.creator_off)) !=
      static_cast<ssize_t>(sizeof(owner)))
    owner = 0;
  if (!empty && (owner == 0 || !Dead(owner))) {
    DFKV_LOG_INFO(prefix + "segment " + name +
                  " was never finished by the process that created it (" +
                  (owner == 0 ? std::string("unknown") : "pid " + std::to_string(owner) +
                                                             ", alive") +
                  "), running without");
    if (base) ::munmap(base, l.len);
    ::close(fd);
    return nullptr;
  }
  const int rc = SizeAndBack(fd, l.len);
  if (rc == 0 && !base) base = Map(fd, l.len);
  ::close(fd);
  if (!base) {
    DFKV_LOG_INFO(prefix + "cannot finish the segment its dead creator left (" +
                  std::to_string(rc) + "), running without");
    return nullptr;
  }
  uint32_t expect = owner;  // 0 for an empty segment
  if (Field(base, l.creator_off)
          ->compare_exchange_strong(expect, static_cast<uint32_t>(::getpid()),
                                    std::memory_order_acq_rel)) {
    DFKV_LOG_WARN(prefix + "segment " + name + " was left unfinished by " +
                  (empty ? std::string("a creator that never sized it")
                         : "pid " + std::to_string(owner) + ", which died") +
                  "; finishing it");
    *creator = true;
    return base;
  }
  // Another opener is finishing it.
  if (WaitLaidOut(base, l)) return base;
  ::munmap(base, l.len);
  DFKV_LOG_INFO(prefix + "segment " + name + " was not finished in time, running without");
  return nullptr;
}

bool AbandonForTest(const std::string& name, const Layout& l, pid_t creator,
                    bool sized) {
  const int fd = ::shm_open(name.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
  if (fd < 0) return false;
  const uint32_t pid = static_cast<uint32_t>(creator);
  bool ok = ::pwrite(fd, &pid, sizeof(pid), static_cast<off_t>(l.creator_off)) ==
            static_cast<ssize_t>(sizeof(pid));
  if (ok && sized) ok = ::ftruncate(fd, static_cast<off_t>(l.len)) == 0;
  ::close(fd);
  return ok;
}

}  // namespace dedup_segment
}  // namespace dfkv
