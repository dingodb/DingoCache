/* StoreEngine — the cache-node local storage backend interface.
 *
 * DiskCacheGroup routes a block to one backend per disk and defaults to
 * extent-slab DiskSlabStore on every construction path; file-per-block KVStore
 * is an explicit diagnostic fallback. The RDMA RangeDirect path and its asynchronous
 * RangeDirectPrep split are part of the interface. Prepared reads return one
 * move-only lease carrying descriptor and storage-pin ownership through the
 * asynchronous transport. */
#ifndef DFKV_STORE_ENGINE_H_
#define DFKV_STORE_ENGINE_H_

#include <cstddef>
#include <cstring>
#include <cstdint>
#include <utility>
#include <unistd.h>
#include <string>
#include <vector>

#include "common/kv_types.h"
#include "common/status.h"

namespace dfkv {

class DiskSlabStore;
class KvNodeServer;
class KVStore;

// Move-only ownership returned by the cheap prep half of RangeDirect. The lease
// owns the opened descriptor and, for slot-based engines, the storage pin that
// prevents reuse while an asynchronous read is outstanding. Moving transfers
// both resources; destruction closes/releases them exactly once. A lease with a
// storage action must be destroyed before its owning StoreEngine.
class ReadLease {
 public:

  ReadLease() noexcept = default;
  ~ReadLease() { Release(); }
  ReadLease(const ReadLease&) = delete;
  ReadLease& operator=(const ReadLease&) = delete;
  ReadLease(ReadLease&& other) noexcept { MoveFrom(std::move(other)); }
  ReadLease& operator=(ReadLease&& other) noexcept {
    if (this != &other) {
      Release();
      MoveFrom(std::move(other));
    }
    return *this;
  }


  int fd() const noexcept { return fd_; }

  uint64_t aligned_off = 0;
  size_t aligned_len = 0;
  size_t head = 0;
  size_t payload_len = 0;
  size_t value_len = 0;

 private:
  friend class DiskSlabStore;
  friend class KVStore;
  using ReleaseFn = void (*)(void* owner, uint64_t token) noexcept;
  static ReadLease Adopt(int fd, void* owner = nullptr, uint64_t token = 0,
                         ReleaseFn release = nullptr) noexcept {
    ReadLease lease;
    lease.fd_ = fd;
    lease.owner_ = owner;
    lease.token_ = token;
    lease.release_ = release;
    return lease;
  }
  void AdoptDescriptor(int fd) noexcept { fd_ = fd; }
  void Release() noexcept {
    const int fd = std::exchange(fd_, -1);
    ReleaseFn release = std::exchange(release_, nullptr);
    void* owner = std::exchange(owner_, nullptr);
    const uint64_t token = std::exchange(token_, 0);
    if (fd >= 0) ::close(fd);
    if (release != nullptr) release(owner, token);
  }
  void MoveFrom(ReadLease&& other) noexcept {
    fd_ = std::exchange(other.fd_, -1);
    owner_ = std::exchange(other.owner_, nullptr);
    token_ = std::exchange(other.token_, 0);
    release_ = std::exchange(other.release_, nullptr);
    aligned_off = other.aligned_off;
    aligned_len = other.aligned_len;
    head = other.head;
    payload_len = other.payload_len;
    value_len = other.value_len;
    other.aligned_off = 0;
    other.aligned_len = 0;
    other.head = 0;
    other.payload_len = 0;
    other.value_len = 0;
  }

  int fd_ = -1;
  void* owner_ = nullptr;
  uint64_t token_ = 0;
  ReleaseFn release_ = nullptr;
};

// One prepared transport read and its complete terminal lifecycle. The owner is
// move-only: storage descriptors/pins, a coalescer flight, a RAM send pin, and
// the caller's staging destination travel together. Commit(), Abort(), or the
// destructor wins exactly once; subsequent terminal calls are harmless.
class PreparedRead {
 public:
  PreparedRead() noexcept = default;
  ~PreparedRead() { Abort(); }
  PreparedRead(const PreparedRead&) = delete;
  PreparedRead& operator=(const PreparedRead&) = delete;
  PreparedRead(PreparedRead&& other) noexcept { MoveFrom(std::move(other)); }
  PreparedRead& operator=(PreparedRead&& other) noexcept {
    if (this != &other) {
      Abort();
      MoveFrom(std::move(other));
    }
    return *this;
  }

  Status status() const noexcept { return status_; }
  bool owns_cleanup() const noexcept { return active_; }
  bool needs_io() const noexcept { return needs_io_; }
  int fd() const noexcept { return lease_.fd(); }
  uint64_t aligned_off() const noexcept { return aligned_off_; }
  size_t aligned_len() const noexcept { return aligned_len_; }
  size_t head() const noexcept { return head_; }
  size_t payload_len() const noexcept { return payload_len_; }
  size_t value_len() const noexcept { return value_len_; }
  char* staging() const noexcept { return staging_; }
  size_t staging_cap() const noexcept { return staging_cap_; }
  const char* data() const noexcept { return data_; }
  bool source_registered() const noexcept { return source_registered_; }
  // Copy a registered source into the owned destination staging when the
  // connection cannot resolve its pool MR. The transaction and its pin remain
  // intact; only the send source changes.
  bool Stage() noexcept {
    if (!active_ || !source_registered_ || staging_ == nullptr ||
        payload_len_ > staging_cap_ ||
        (payload_len_ != 0 && data_ == nullptr)) {
      return false;
    }
    if (payload_len_ != 0) std::memcpy(staging_, data_, payload_len_);
    data_ = staging_;
    source_registered_ = false;
    return true;
  }

  // Commit a completed storage read or a completed RAM-backed send. `bytes_read`
  // is the exact payload byte count exposed to completion accounting.
  bool Commit(Status result, size_t bytes_read, double elapsed_sec) noexcept {
    if (!active_) return false;
    active_ = false;
    FinishFn finish = std::exchange(finish_, nullptr);
    void* owner = std::exchange(owner_, nullptr);
    const uint64_t token = std::exchange(token_, 0);
    if (finish != nullptr) {
      finish(owner, token, true, result, bytes_read, elapsed_sec,
             result == Status::kOk ? data_ : nullptr);
    }
    lease_ = ReadLease{};
    return true;
  }

  // Abort preparation/queueing/send teardown. The destructor calls this when a
  // caller leaves any path without choosing a terminal action.
  bool Abort() noexcept {
    if (!active_) return false;
    active_ = false;
    FinishFn finish = std::exchange(finish_, nullptr);
    void* owner = std::exchange(owner_, nullptr);
    const uint64_t token = std::exchange(token_, 0);
    if (finish != nullptr) {
      finish(owner, token, false, Status::kIOError, 0, 0.0, nullptr);
    }
    lease_ = ReadLease{};
    return true;
  }

 private:
  friend class KvNodeServer;
  using FinishFn = void (*)(void* owner, uint64_t token, bool committed,
                            Status result, size_t bytes_read,
                            double elapsed_sec, const char* data) noexcept;

  static PreparedRead Result(Status status) noexcept {
    PreparedRead read;
    read.status_ = status;
    return read;
  }
  static PreparedRead Storage(ReadLease lease, char* staging,
                              size_t staging_cap, void* owner, uint64_t token,
                              FinishFn finish) noexcept {
    PreparedRead read;
    read.status_ = Status::kOk;
    read.needs_io_ = lease.aligned_len != 0;
    read.aligned_off_ = lease.aligned_off;
    read.aligned_len_ = lease.aligned_len;
    read.head_ = lease.head;
    read.payload_len_ = lease.payload_len;
    read.value_len_ = lease.value_len;
    read.staging_ = staging;
    read.staging_cap_ = staging_cap;
    read.data_ = staging == nullptr ? nullptr : staging + lease.head;
    read.owner_ = owner;
    read.token_ = token;
    read.finish_ = finish;
    read.active_ = true;
    read.lease_ = std::move(lease);
    return read;
  }
  static PreparedRead Ready(const char* data, size_t payload_len,
                            size_t value_len, bool source_registered,
                            char* staging, size_t staging_cap, void* owner,
                            uint64_t token, FinishFn finish) noexcept {
    PreparedRead read;
    read.status_ = Status::kOk;
    read.payload_len_ = payload_len;
    read.value_len_ = value_len;
    read.staging_ = staging;
    read.staging_cap_ = staging_cap;
    read.data_ = data;
    read.source_registered_ = source_registered;
    read.owner_ = owner;
    read.token_ = token;
    read.finish_ = finish;
    read.active_ = true;
    return read;
  }
  void MoveFrom(PreparedRead&& other) noexcept {
    status_ = other.status_;
    active_ = std::exchange(other.active_, false);
    needs_io_ = other.needs_io_;
    source_registered_ = other.source_registered_;
    aligned_off_ = other.aligned_off_;
    aligned_len_ = other.aligned_len_;
    head_ = other.head_;
    payload_len_ = other.payload_len_;
    value_len_ = other.value_len_;
    staging_ = other.staging_;
    staging_cap_ = other.staging_cap_;
    data_ = other.data_;
    owner_ = std::exchange(other.owner_, nullptr);
    token_ = std::exchange(other.token_, 0);
    finish_ = std::exchange(other.finish_, nullptr);
    lease_ = std::move(other.lease_);
  }

  Status status_ = Status::kInvalid;
  bool active_ = false;
  bool needs_io_ = false;
  bool source_registered_ = false;
  uint64_t aligned_off_ = 0;
  size_t aligned_len_ = 0;
  size_t head_ = 0;
  size_t payload_len_ = 0;
  size_t value_len_ = 0;
  char* staging_ = nullptr;
  size_t staging_cap_ = 0;
  const char* data_ = nullptr;
  void* owner_ = nullptr;
  uint64_t token_ = 0;
  FinishFn finish_ = nullptr;
  ReadLease lease_;
};
struct ValueMetadata {
  uint64_t value_len = 0;
};


class StoreEngine {
 public:
  virtual ~StoreEngine() = default;
  // Construction cannot return a Status; startup health is therefore explicit.
  // A node must refuse readiness when any configured engine is unhealthy.
  virtual bool Healthy() const = 0;
  virtual const std::string& StartupError() const = 0;

  virtual Status Cache(const BlockKey& key, const void* data, size_t len) = 0;
  virtual Status CacheDirect(const BlockKey& key, char* data, size_t len,
                             size_t cap) = 0;
  // Batched CacheDirect: same per-item semantics/status as CacheDirect, but an
  // engine may amortize its lock round-trips and submit the payload writes
  // together (the RAM-tier flusher's small-object drain is IOPS-bound at one
  // synchronous write per key). Default: per-item loop -- engines without a
  // batch win (file engine) inherit correct behavior.
  struct CacheBatchItem { BlockKey key; char* data; size_t len; size_t cap; };
  virtual std::vector<Status> CacheDirectBatch(const std::vector<CacheBatchItem>& items) {
    std::vector<Status> out;
    out.reserve(items.size());
    for (const auto& it : items) out.push_back(CacheDirect(it.key, it.data, it.len, it.cap));
    return out;
  }
  virtual Status Range(const BlockKey& key, uint64_t offset, uint64_t length,
                       std::string* out, size_t* value_len = nullptr) = 0;
  virtual Status RangeInto(const BlockKey& key, uint64_t offset, uint64_t length,
                           char* dst, size_t dst_cap, size_t* out_len,
                           size_t* value_len = nullptr) = 0;
  virtual Status RangeDirect(const BlockKey& key, uint64_t offset,
                             uint64_t length, char* io_buf, size_t io_cap,
                             const char** out_data, size_t* out_len,
                             size_t* value_len = nullptr) = 0;
  virtual Status RangeDirectPrep(const BlockKey& key, uint64_t offset,
                                 uint64_t length, size_t io_cap,
                                 ReadLease* out) = 0;
  virtual bool IsCached(const BlockKey& key) const = 0;
  // Payload-free metadata lookup. kOk means the value is committed/readable;
  // kNotFound also covers a write that has not reached its commit point.
  virtual Status Lookup(const BlockKey& key, ValueMetadata* out) const = 0;
  virtual Status Remove(const BlockKey& key) = 0;

  virtual uint64_t UsedBytes() const = 0;
  // Committed payload bytes owned by one tenant. Implementations that do not
  // support tenant accounting may keep the zero default; production stores
  // override it and rebuild the value from persistent identity metadata.
  virtual uint64_t TenantUsedBytes(uint64_t tenant_hash) const {
    (void)tenant_hash;
    return 0;
  }
  virtual size_t Count() const = 0;
  virtual uint64_t Evictions() const = 0;
  virtual uint64_t EvictedBytes() const = 0;
  virtual const std::string& Dir() const = 0;
};

}  // namespace dfkv

#endif  // DFKV_STORE_ENGINE_H_
