#include "cache/lse_io.h"

#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

#include <cerrno>

namespace dfkv {
namespace lse {
namespace {

class SystemIoBackend final : public IoBackend {
 public:
  int Open(const std::string& path, int flags, mode_t mode) override {
    return ::open(path.c_str(), flags | O_CLOEXEC, mode);
  }
  int Close(int fd) override { return ::close(fd); }
  ssize_t Pwritev(int fd, const struct iovec* iov, int iovcnt, off_t offset,
                  bool dsync) override {
    ssize_t n;
    do {
      n = ::pwritev2(fd, iov, iovcnt, offset, dsync ? RWF_DSYNC : 0);
    } while (n < 0 && errno == EINTR);
    return n;
  }
  ssize_t Pread(int fd, void* buf, size_t len, off_t offset) override {
    ssize_t n;
    do {
      n = ::pread(fd, buf, len, offset);
    } while (n < 0 && errno == EINTR);
    return n;
  }
  int Fdatasync(int fd) override { return ::fdatasync(fd); }
  int Fsync(int fd) override { return ::fsync(fd); }
  int Fallocate(int fd, off_t offset, off_t len) override {
    int rc;
    do {
      rc = ::fallocate(fd, 0, offset, len);
    } while (rc != 0 && errno == EINTR);
    return rc;
  }
  int Ftruncate(int fd, off_t len) override { return ::ftruncate(fd, len); }
  int Rename(const std::string& from, const std::string& to) override {
    return ::rename(from.c_str(), to.c_str());
  }
  int Mkdir(const std::string& path, mode_t mode) override {
    return ::mkdir(path.c_str(), mode);
  }
  int Unlink(const std::string& path) override {
    return ::unlink(path.c_str());
  }
};

}  // namespace

std::unique_ptr<IoBackend> MakeSystemIoBackend() {
  return std::make_unique<SystemIoBackend>();
}

}  // namespace lse
}  // namespace dfkv
