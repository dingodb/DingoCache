/* Shared, aligned receive segment allocator for RDMA v2 server connections. */
#ifndef DFKV_TRANSPORT_RDMA_RECV_SEGMENT_H_
#define DFKV_TRANSPORT_RDMA_RECV_SEGMENT_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>

namespace dfkv::rdma {
// Parse and align the process-wide receive-segment size without applying the
// wire payload's uint32 ceiling. Invalid/zero/overflow values use fallback;
// returns 0 only when the fallback geometry itself is invalid.
size_t ResolveRecvSegmentBytes(const char* value, size_t fallback,
                               size_t alignment = 4096);

class RecvSegment {
 public:
  class Lease {
   public:
    Lease() = default;
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& other) noexcept;
    Lease& operator=(Lease&& other) noexcept;
    ~Lease();

    explicit operator bool() const { return owner_ != nullptr; }
    char* data() const;
    size_t offset() const { return offset_; }
    size_t size() const { return size_; }
    void Reset();

   private:
    friend class RecvSegment;
    Lease(RecvSegment* owner, size_t offset, size_t size)
        : owner_(owner), offset_(offset), size_(size) {}

    RecvSegment* owner_ = nullptr;
    size_t offset_ = 0;
    size_t size_ = 0;
  };

  RecvSegment() = default;
  RecvSegment(const RecvSegment&) = delete;
  RecvSegment& operator=(const RecvSegment&) = delete;
  ~RecvSegment();

  struct Stats {
    size_t total_bytes = 0;
    size_t used_bytes = 0;
    size_t free_bytes = 0;
    size_t largest_free_range = 0;
    uint64_t allocation_failures = 0;
  };

  bool Init(size_t bytes, size_t alignment = 4096);
  Lease Allocate(size_t bytes, size_t alignment = 4096);

  char* data() const { return data_; }
  size_t size() const { return size_; }
  size_t free_bytes() const;
  Stats stats() const;

 private:
  void Release(size_t offset, size_t bytes);

  char* data_ = nullptr;
  size_t size_ = 0;
  size_t alignment_ = 0;
  mutable std::mutex mu_;
  std::map<size_t, size_t> free_;  // offset -> length, sorted and coalesced
  std::atomic<uint64_t> allocation_failures_{0};
};

}  // namespace dfkv::rdma

#endif  // DFKV_TRANSPORT_RDMA_RECV_SEGMENT_H_
