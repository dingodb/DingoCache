/* The I/O the log-structured engine does, behind one interface
 * (docs/design/log-structured-engine.md §11.4), so that tests can record it
 * and build crash images from it: which writes a power loss may drop, tear or
 * reorder, and which are covered by a completed sync.
 *
 * Every call is synchronous: it has finished, or failed, when it returns.
 * The journal relies on that (design §4.5): no journal write can still be in
 * flight once its process has died. Segment payloads may later also go through
 * io_uring; that path is outside this interface and covered by the open-time
 * barrier instead. */
#ifndef DFKV_LSE_IO_H_
#define DFKV_LSE_IO_H_

#include <sys/types.h>
#include <sys/uio.h>

#include <cstdint>
#include <memory>
#include <string>

namespace dfkv {
namespace lse {

class IoBackend {
 public:
  virtual ~IoBackend() = default;
  // As open(2); returns a descriptor or -1 with errno set.
  virtual int Open(const std::string& path, int flags, mode_t mode) = 0;
  virtual int Close(int fd) = 0;
  // As pwritev2(2). `dsync`: RWF_DSYNC, the write is durable when it returns.
  // Returns the bytes written or -1 with errno set.
  virtual ssize_t Pwritev(int fd, const struct iovec* iov, int iovcnt,
                          off_t offset, bool dsync) = 0;
  virtual ssize_t Pread(int fd, void* buf, size_t len, off_t offset) = 0;
  virtual int Fdatasync(int fd) = 0;
  virtual int Fsync(int fd) = 0;  // also for directories
  virtual int Fallocate(int fd, off_t offset, off_t len) = 0;
  virtual int Ftruncate(int fd, off_t len) = 0;
  virtual int Rename(const std::string& from, const std::string& to) = 0;
  virtual int Mkdir(const std::string& path, mode_t mode) = 0;
  virtual int Unlink(const std::string& path) = 0;
};

// The real system calls.
std::unique_ptr<IoBackend> MakeSystemIoBackend();

}  // namespace lse
}  // namespace dfkv

#endif  // DFKV_LSE_IO_H_
