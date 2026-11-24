/* Shared, aligned receive segment allocator for RDMA v2 server connections. */
#ifndef DFKV_TRANSPORT_RDMA_RECV_SEGMENT_H_
#define DFKV_TRANSPORT_RDMA_RECV_SEGMENT_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

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

  // huge_pages: ask for transparent huge pages (a segment of at least one
  // huge page, aligned to it), as the RAM tier arena does: registering a
  // range of it for one RDMA operation then pins 2 MiB pages instead of
  // 4 KiB ones, several times cheaper for large ranges.
  bool Init(size_t bytes, size_t alignment = 4096, int numa_node = -1,
            bool huge_pages = false);
  bool huge_pages() const { return huge_pages_; }
  Lease Allocate(size_t bytes, size_t alignment = 4096);

  char* data() const { return data_; }
  size_t size() const { return size_; }
  size_t free_bytes() const;
  Stats stats() const;
  // When the last lease was released (or the segment initialized); 0 while
  // a lease is out.
  uint64_t EmptySinceMs() const;
  // The free ranges, (offset, length) in offset order. Leases and free
  // ranges begin and end on the segment's alignment, so ranges that touch
  // form one range a lease of that alignment can take whole.
  std::vector<std::pair<size_t, size_t>> FreeRanges() const;

 private:
  void Release(size_t offset, size_t bytes);

  char* data_ = nullptr;
  size_t size_ = 0;
  size_t alignment_ = 0;
  mutable std::mutex mu_;
  std::map<size_t, size_t> free_;  // offset -> length, sorted and coalesced
  bool huge_pages_ = false;
  bool mapped_ = false;  // data_ is an mmap of its own (huge pages)
  size_t used_ = 0;                // under mu_
  uint64_t empty_since_ms_ = 0;    // under mu_
  std::atomic<uint64_t> allocation_failures_{0};
};

// Lazily commits fixed-size registered-memory candidates up to one hard
// process budget. A lease retains its owning chunk, so callers can register
// only that chunk on the selected rail instead of one monolithic arena.
class RecvSegmentPool {
 public:
  enum class LeaseClass { kConnection, kStaging };
  class Lease {
   public:
    Lease() = default;
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& other) noexcept;
    Lease& operator=(Lease&& other) noexcept;
    ~Lease();

    explicit operator bool() const { return static_cast<bool>(lease_); }
    char* data() const { return lease_.data(); }
    size_t size() const { return lease_.size(); }
    RecvSegment* segment() const { return segment_; }
    void Reset();

   private:
    friend class RecvSegmentPool;
    Lease(RecvSegmentPool* pool, RecvSegment* segment,
          RecvSegment::Lease lease)
        : pool_(pool), lease_(std::move(lease)), segment_(segment) {}

    RecvSegmentPool* pool_ = nullptr;
    RecvSegment::Lease lease_;
    RecvSegment* segment_ = nullptr;
  };

  struct Stats {
    size_t max_bytes = 0;
    size_t committed_bytes = 0;
    size_t used_bytes = 0;
    size_t free_bytes = 0;
    size_t largest_free_range = 0;
    size_t connection_free_bytes = 0;
    size_t connection_largest_free_range = 0;
    size_t staging_reserved_bytes = 0;
    size_t staging_reserved_free_bytes = 0;
    size_t chunks = 0;
    uint64_t growths = 0;
    uint64_t shrinks = 0;
    uint64_t released_bytes = 0;
    uint64_t allocation_failures = 0;
    uint64_t growth_failures = 0;
  };

  // When a whole initial chunk is reserved for staging, connection-lifetime
  // leases cannot consume it. The reservation stays within max_bytes.
  bool Init(size_t chunk_bytes, size_t max_bytes,
            size_t alignment = 4096, size_t staging_reserve_bytes = 0);
  // Chunks committed from now on ask for transparent huge pages (see
  // RecvSegment::Init). Call before Init to cover the initial chunk.
  void set_huge_pages(bool on) { huge_pages_ = on; }
  Lease Allocate(size_t bytes, size_t alignment = 4096,
                 int affinity = -1, int numa_node = -1,
                 LeaseClass lease_class = LeaseClass::kConnection);
  size_t TrimIdle(uint64_t idle_ms);
  // Called for a chunk about to be freed, outside the pool lock, before its
  // memory goes (a trim unlinks it under the lock first). False keeps the
  // chunk's memory for good. Set before the first allocation.
  void set_release_hook(std::function<bool(const RecvSegment*)> hook) {
    release_hook_ = std::move(hook);
  }
  // Chunks the release hook refused, kept for good.
  uint64_t kept_chunks() const {
    return kept_chunks_.load(std::memory_order_relaxed);
  }
  RecvSegment* initial_segment() const;
  size_t max_bytes() const;
  // The pool as it is now, for planning which leases to give up: every
  // chunk with its free ranges, and the budget.
  struct ChunkView {
    const RecvSegment* segment = nullptr;
    size_t size = 0;
    int affinity = -1;
    bool staging_only = false;
    bool initial = false;  // never released
    std::vector<std::pair<size_t, size_t>> free;
  };
  struct Geometry {
    size_t max_bytes = 0;
    size_t chunk_bytes = 0;
    size_t committed_bytes = 0;
    std::vector<ChunkView> chunks;
  };
  Geometry Snapshot() const;
  Stats stats() const;

 private:
  struct Chunk {
    std::unique_ptr<RecvSegment> segment;
    int affinity = -1;
    bool staging_only = false;
  };
  std::unique_ptr<Chunk> NewChunk(size_t minimum_bytes, int affinity,
                                  int numa_node);
  // First fit over the committed chunks; mu_ held.
  Lease AllocateFromChunks(size_t bytes, size_t alignment, int affinity,
                           LeaseClass lease_class);
  // Unlinks empty chunks other than the initial one, newest first, until
  // `wanted` bytes of the budget are uncommitted (or none is left), into
  // *unlinked; mu_ held.
  size_t TrimEmptyLocked(size_t wanted,
                         std::vector<std::unique_ptr<Chunk>>* unlinked);
  // Frees unlinked chunks through the release hook; mu_ not held.
  void FreeUnlinked(std::vector<std::unique_ptr<Chunk>>* chunks);

  size_t chunk_bytes_ = 0;
  size_t max_bytes_ = 0;
  size_t staging_reserve_bytes_ = 0;
  size_t alignment_ = 0;
  bool huge_pages_ = false;
  std::function<bool(const RecvSegment*)> release_hook_;
  std::atomic<uint64_t> kept_chunks_{0};
  // Allocations, growth, trimming. A release takes only its chunk's own
  // lock: it no longer goes through the pool.
  mutable std::mutex mu_;
  std::vector<std::unique_ptr<Chunk>> chunks_;
  std::atomic<uint64_t> growths_{0};
  std::atomic<uint64_t> shrinks_{0};
  std::atomic<uint64_t> released_bytes_{0};
  std::atomic<uint64_t> allocation_failures_{0};
  std::atomic<uint64_t> growth_failures_{0};
};

}  // namespace dfkv::rdma

#endif  // DFKV_TRANSPORT_RDMA_RECV_SEGMENT_H_
