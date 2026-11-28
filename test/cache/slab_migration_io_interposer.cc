#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>

namespace {

using OpenFn = int (*)(const char*, int, ...);
using CloseFn = int (*)(int);
using PreadFn = ssize_t (*)(int, void*, size_t, off_t);
using Pwritev2Fn = ssize_t (*)(int, const struct iovec*, int, off_t, int);

OpenFn RealOpen = reinterpret_cast<OpenFn>(dlsym(RTLD_NEXT, "open"));
CloseFn RealClose = reinterpret_cast<CloseFn>(dlsym(RTLD_NEXT, "close"));
PreadFn RealPread = reinterpret_cast<PreadFn>(dlsym(RTLD_NEXT, "pread"));
Pwritev2Fn RealPwritev2 =
    reinterpret_cast<Pwritev2Fn>(dlsym(RTLD_NEXT, "pwritev2"));

struct State {
  std::string scenario;
  unsigned pread_stage = 0;
  unsigned pwrite_stage = 0;
  uint64_t committed_writes = 0;
  std::set<int> direct_fds;
  std::set<int> direct_read_fds;
};
State state;
std::mutex mutex;

bool Contains(const std::string& text, const char* needle) {
  return text.find(needle) != std::string::npos;
}

std::string Environment(const char* name) {
  const char* value = ::getenv(name);
  return value == nullptr ? std::string() : std::string(value);
}

void EventUnlocked(const char* event, const std::string& path,
                   off_t offset = -1, size_t bytes = 0, ssize_t result = -2) {
  const std::string log = Environment("DFKV_MIGRATION_IO_INTERPOSE_LOG");
  if (log.empty()) return;
  const int fd = RealOpen(log.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
  if (fd < 0) return;
  char line[2048];
  const int len = std::snprintf(
      line, sizeof(line),
      "event=%s pid=%ld path=%s offset=%lld bytes=%zu result=%lld\n", event,
      static_cast<long>(::getpid()), path.c_str(), static_cast<long long>(offset),
      bytes, static_cast<long long>(result));
  if (len > 0) {
    const char* p = line;
    size_t left = static_cast<size_t>(len) < sizeof(line) ? static_cast<size_t>(len)
                                                          : sizeof(line) - 1;
    while (left != 0) {
      const ssize_t n = ::write(fd, p, left);
      if (n > 0) {
        p += n;
        left -= static_cast<size_t>(n);
      } else if (n < 0 && errno == EINTR) {
        continue;
      } else {
        break;
      }
    }
  }
  RealClose(fd);
}

// Changing scenarios resets injection state but preserves the observed direct
// descriptor inventory only until its close. Each runtime trial uses one
// scenario at a time and its own evidence file.
void RefreshScenarioUnlocked() {
  const std::string scenario = Environment("DFKV_MIGRATION_IO_SCENARIO");
  if (scenario != state.scenario) {
    state.scenario = scenario;
    state.pread_stage = 0;
    state.pwrite_stage = 0;
    state.committed_writes = 0;
  }
}

bool StagingSegment(const std::string& path) {
  return Contains(path, ".sto14-staging/segments/");
}

bool SourceExtent(const std::string& path) {
  return Contains(path, "/source/extents/E");
}

size_t IovBytes(const struct iovec* iov, int iovcnt) {
  size_t total = 0;
  for (int i = 0; i < iovcnt; ++i) total += iov[i].iov_len;
  return total;
}

}  // namespace

__attribute__((constructor)) void MigrationIoInterposerReady() {
  ::setenv("DFKV_MIGRATION_IO_INTERPOSE_ACTIVE", "1", 1);
}

extern "C" int open(const char* path, int flags, ...) {
  mode_t mode = 0;
  if ((flags & O_CREAT) != 0
#ifdef O_TMPFILE
      || (flags & O_TMPFILE) == O_TMPFILE
#endif
  ) {
    va_list ap;
    va_start(ap, flags);
    mode = static_cast<mode_t>(va_arg(ap, int));
    va_end(ap);
  }
  const int fd = RealOpen(path, flags, mode);
  std::lock_guard<std::mutex> lock(mutex);
  RefreshScenarioUnlocked();
  const std::string watch = Environment("DFKV_MIGRATION_IO_WATCH");
  const std::string name(path);
  if (fd >= 0 && (flags & O_DIRECT) != 0 && !watch.empty() &&
      Contains(name, watch.c_str()) && Contains(name, "/segments/")) {
    state.direct_fds.insert(fd);
    EventUnlocked("odirect_open", name, -1, 0, fd);
  }
  return fd;
}

extern "C" int close(int fd) {
  const int rc = RealClose(fd);
  std::lock_guard<std::mutex> lock(mutex);
  state.direct_fds.erase(fd);
  state.direct_read_fds.erase(fd);
  return rc;
}

extern "C" ssize_t pread(int fd, void* buf, size_t count, off_t offset) {
  std::lock_guard<std::mutex> lock(mutex);
  RefreshScenarioUnlocked();
  char link[64];
  std::snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
  char path_buf[4096];
  const ssize_t path_len = ::readlink(link, path_buf, sizeof(path_buf) - 1);
  const std::string path =
      path_len > 0 ? std::string(path_buf, static_cast<size_t>(path_len)) : std::string();

  if (!path.empty() && state.direct_fds.count(fd) != 0 &&
      state.direct_read_fds.insert(fd).second) {
    EventUnlocked("odirect_pread", path, offset, count, fd);
  }

  const std::string target_text = Environment("DFKV_MIGRATION_IO_TARGET_READ_BYTES");
  const size_t target = target_text.empty()
                            ? 0
                            : static_cast<size_t>(std::strtoull(target_text.c_str(), nullptr, 10));
  if (target != 0 && count == target && SourceExtent(path) &&
      state.scenario == "retry") {
    if (state.pread_stage == 0) {
      state.pread_stage = 1;
      EventUnlocked("source_pread_eintr", path, offset, count, -EINTR);
      errno = EINTR;
      return -1;
    }
    if (state.pread_stage == 1) {
      state.pread_stage = 2;
      const ssize_t rc = RealPread(fd, buf, count - 1, offset);
      EventUnlocked("source_pread_short", path, offset, count, rc);
      return rc;
    }
    if (state.pread_stage == 2) {
      state.pread_stage = 3;
      const ssize_t rc = RealPread(fd, buf, count, offset);
      EventUnlocked("source_pread_retry_completed", path, offset, count, rc);
      return rc;
    }
  }
  if (target != 0 && count == target && SourceExtent(path) &&
      state.scenario == "source_eof" && state.pread_stage == 0) {
    state.pread_stage = 1;
    EventUnlocked("source_eof_injected", path, offset, count, 0);
    return 0;
  }
  return RealPread(fd, buf, count, offset);
}

extern "C" ssize_t pwritev2(int fd, const struct iovec* iov, int iovcnt,
                            off_t offset, int flags) {
  std::lock_guard<std::mutex> lock(mutex);
  RefreshScenarioUnlocked();
  char link[64];
  std::snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
  char path_buf[4096];
  const ssize_t path_len = ::readlink(link, path_buf, sizeof(path_buf) - 1);
  const std::string path =
      path_len > 0 ? std::string(path_buf, static_cast<size_t>(path_len)) : std::string();
  const size_t total = IovBytes(iov, iovcnt);
  const bool record_write = StagingSegment(path) && offset >= 4096 && total >= 4096;
  if (!record_write) return RealPwritev2(fd, iov, iovcnt, offset, flags);

  if (state.scenario == "retry" && state.pwrite_stage == 2 &&
      offset == 8193 && total == 5000) {
    state.pwrite_stage = 3;
    const ssize_t rc = RealPwritev2(fd, iov, iovcnt, offset, flags);
    EventUnlocked("destination_pwritev2_retry_completed", path, offset, total, rc);
    return rc;
  }

  if (state.scenario == "retry" && offset == 4096 && total > 8192) {
    if (state.pwrite_stage == 0) {
      state.pwrite_stage = 1;
      EventUnlocked("destination_pwritev2_eintr", path, offset, total, -EINTR);
      errno = EINTR;
      return -1;
    }
    if (state.pwrite_stage == 1) {
      state.pwrite_stage = 2;
      // Return a real positive short write that crosses from the 4 KiB record
      // header into the payload iovec. PwriteAll must adjust both iovecs and
      // the next file offset rather than merely retrying the same bytes.
      struct iovec shortened[8];
      const int copied = iovcnt < 8 ? iovcnt : 8;
      for (int i = 0; i < copied; ++i) shortened[i] = iov[i];
      size_t left = 4097;
      int used = 0;
      for (; used < copied && left != 0; ++used) {
        if (shortened[used].iov_len > left) shortened[used].iov_len = left;
        left -= shortened[used].iov_len;
      }
      const ssize_t rc = RealPwritev2(fd, shortened, used, offset, flags);
      EventUnlocked("destination_pwritev2_short", path, offset, total, rc);
      return rc;
    }
  }

  if (state.scenario == "destination_fail" && state.committed_writes != 0 &&
      state.pwrite_stage == 0) {
    state.pwrite_stage = 1;
    EventUnlocked("destination_pwritev2_eio", path, offset, total, -EIO);
    errno = EIO;
    return -1;
  }

  const ssize_t rc = RealPwritev2(fd, iov, iovcnt, offset, flags);
  if (rc > 0 && offset + rc >= offset + static_cast<off_t>(total)) {
    ++state.committed_writes;
    EventUnlocked("destination_record_write_ok", path, offset, total, rc);
  }
  if (rc > 0 && state.committed_writes != 0 &&
      (state.scenario == "source_eof" || state.scenario == "destination_fail")) {
    EventUnlocked("destination_committed_before_fault", path, offset, total,
                  static_cast<ssize_t>(state.committed_writes));
  }
  return rc;
}
