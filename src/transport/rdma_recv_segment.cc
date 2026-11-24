#include "transport/rdma_recv_segment.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <utility>

#include <sys/mman.h>

#include "utils/numa_util.h"

namespace dfkv::rdma {
namespace {

bool IsPowerOfTwo(size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

size_t AlignUpChecked(size_t value, size_t alignment) {
  if (!IsPowerOfTwo(alignment) ||
      value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
    return std::numeric_limits<size_t>::max();
  }
  return (value + alignment - 1) & ~(alignment - 1);
}

uint64_t SteadyMs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

}  // namespace

size_t ResolveRecvSegmentBytes(const char* value, size_t fallback,
                               size_t alignment) {
  if (!IsPowerOfTwo(alignment)) return 0;
  size_t bytes = fallback;
  if (value && *value) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno == 0 && end != value && *end == '\0' && parsed != 0 &&
        parsed <= std::numeric_limits<size_t>::max()) {
      bytes = static_cast<size_t>(parsed);
    }
  }
  const size_t aligned = AlignUpChecked(bytes, alignment);
  return aligned == std::numeric_limits<size_t>::max() ? 0 : aligned;
}

RecvSegment::Lease::Lease(Lease&& other) noexcept
    : owner_(other.owner_), offset_(other.offset_), size_(other.size_) {
  other.owner_ = nullptr;
  other.offset_ = 0;
  other.size_ = 0;
}

RecvSegment::Lease& RecvSegment::Lease::operator=(Lease&& other) noexcept {
  if (this == &other) return *this;
  Reset();
  owner_ = other.owner_;
  offset_ = other.offset_;
  size_ = other.size_;
  other.owner_ = nullptr;
  other.offset_ = 0;
  other.size_ = 0;
  return *this;
}

RecvSegment::Lease::~Lease() { Reset(); }

char* RecvSegment::Lease::data() const {
  return owner_ ? owner_->data_ + offset_ : nullptr;
}

void RecvSegment::Lease::Reset() {
  if (!owner_) return;
  owner_->Release(offset_, size_);
  owner_ = nullptr;
  offset_ = 0;
  size_ = 0;
}

RecvSegment::~RecvSegment() {
  if (mapped_)
    ::munmap(data_, size_);
  else
    std::free(data_);
}

bool RecvSegment::Init(size_t bytes, size_t alignment, int numa_node,
                       bool huge_pages) {
  if (data_) return size_ == bytes && alignment_ == alignment;
  if (bytes == 0 || !IsPowerOfTwo(alignment) || bytes % alignment != 0 ||
      alignment < sizeof(void*)) {
    return false;
  }
  constexpr size_t kHugePageBytes = size_t{2} << 20;
  huge_pages = huge_pages && bytes >= kHugePageBytes;
  void* raw = nullptr;
  if (huge_pages) {
    // A mapping of its own, aligned to the huge page, rather than heap
    // memory: madvise marks the whole mapping it falls in, and a heap
    // mapping is shared with unrelated allocations.
    const size_t align = std::max(alignment, kHugePageBytes);
    const size_t span = bytes + align;
    void* map = ::mmap(nullptr, span, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (map == MAP_FAILED) return false;
    const uintptr_t base = reinterpret_cast<uintptr_t>(map);
    const uintptr_t start = (base + align - 1) & ~(uintptr_t{align} - 1);
    if (start > base) ::munmap(map, start - base);
    const uintptr_t end = start + bytes;
    if (base + span > end)
      ::munmap(reinterpret_cast<void*>(end), base + span - end);
    raw = reinterpret_cast<void*>(start);
    mapped_ = true;
  } else if (::posix_memalign(&raw, alignment, bytes) != 0) {
    return false;
  }
#ifdef MADV_HUGEPAGE
  // Refused where the kernel has no transparent huge pages: small pages.
  if (huge_pages && ::madvise(raw, bytes, MADV_HUGEPAGE) != 0)
    huge_pages = false;
#else
  huge_pages = false;
#endif
  huge_pages_ = huge_pages;
  data_ = static_cast<char*>(raw);
  numa::BindMemory(raw, bytes, numa_node);
  size_ = bytes;
  alignment_ = alignment;
  free_.emplace(0, bytes);
  empty_since_ms_ = SteadyMs();
  return true;
}

RecvSegment::Lease RecvSegment::Allocate(size_t bytes, size_t alignment) {
  if (!data_ || bytes == 0 || !IsPowerOfTwo(alignment)) return {};
  std::lock_guard<std::mutex> lock(mu_);
  for (auto it = free_.begin(); it != free_.end(); ++it) {
    const size_t range_begin = it->first;
    const size_t range_size = it->second;
    const uintptr_t base = reinterpret_cast<uintptr_t>(data_);
    if (range_begin > std::numeric_limits<uintptr_t>::max() - base) continue;
    const size_t absolute =
        AlignUpChecked(static_cast<size_t>(base + range_begin), alignment);
    if (absolute == std::numeric_limits<size_t>::max() || absolute < base)
      continue;
    const size_t begin = absolute - base;
    if (begin < range_begin || begin - range_begin > range_size ||
        bytes > range_size - (begin - range_begin)) {
      continue;
    }
    const size_t range_end = range_begin + range_size;
    const size_t end = begin + bytes;
    free_.erase(it);
    if (begin > range_begin) free_.emplace(range_begin, begin - range_begin);
    if (end < range_end) free_.emplace(end, range_end - end);
    used_ += bytes;
    empty_since_ms_ = 0;
    return Lease(this, begin, bytes);
  }
  allocation_failures_.fetch_add(1, std::memory_order_relaxed);
  return {};
}

size_t RecvSegment::free_bytes() const { return stats().free_bytes; }

uint64_t RecvSegment::EmptySinceMs() const {
  std::lock_guard<std::mutex> lock(mu_);
  return used_ == 0 ? empty_since_ms_ : 0;
}

std::vector<std::pair<size_t, size_t>> RecvSegment::FreeRanges() const {
  std::lock_guard<std::mutex> lock(mu_);
  return {free_.begin(), free_.end()};
}

RecvSegment::Stats RecvSegment::stats() const {
  std::lock_guard<std::mutex> lock(mu_);
  Stats out;
  out.total_bytes = size_;
  for (const auto& range : free_) {
    out.free_bytes += range.second;
    out.largest_free_range = std::max(out.largest_free_range, range.second);
  }
  out.used_bytes = out.total_bytes - out.free_bytes;
  out.allocation_failures =
      allocation_failures_.load(std::memory_order_relaxed);
  return out;
}

void RecvSegment::Release(size_t offset, size_t bytes) {
  if (bytes == 0) return;
  std::lock_guard<std::mutex> lock(mu_);
  size_t begin = offset;
  size_t end = offset + bytes;

  auto next = free_.lower_bound(begin);
  if (next != free_.begin()) {
    auto prev = std::prev(next);
    if (prev->first + prev->second == begin) {
      begin = prev->first;
      free_.erase(prev);
    }
  }
  next = free_.lower_bound(end);
  if (next != free_.end() && next->first == end) {
    end += next->second;
    free_.erase(next);
  }
  free_.emplace(begin, end - begin);
  used_ -= bytes;
  if (used_ == 0) empty_since_ms_ = SteadyMs();
}

RecvSegmentPool::Lease::Lease(Lease&& other) noexcept
    : pool_(other.pool_),
      lease_(std::move(other.lease_)),
      segment_(other.segment_) {
  other.pool_ = nullptr;
  other.segment_ = nullptr;
}

RecvSegmentPool::Lease& RecvSegmentPool::Lease::operator=(
    Lease&& other) noexcept {
  if (this == &other) return *this;
  Reset();
  pool_ = other.pool_;
  lease_ = std::move(other.lease_);
  segment_ = other.segment_;
  other.pool_ = nullptr;
  other.segment_ = nullptr;
  return *this;
}

RecvSegmentPool::Lease::~Lease() { Reset(); }

void RecvSegmentPool::Lease::Reset() {
  lease_.Reset();
  pool_ = nullptr;
  segment_ = nullptr;
}

bool RecvSegmentPool::Init(size_t chunk_bytes, size_t max_bytes,
                           size_t alignment, size_t staging_reserve_bytes) {
  if (!IsPowerOfTwo(alignment) || alignment < sizeof(void*) ||
      chunk_bytes == 0 || max_bytes == 0) {
    return false;
  }
  const size_t aligned_chunk = AlignUpChecked(chunk_bytes, alignment);
  const size_t aligned_max = AlignUpChecked(max_bytes, alignment);
  const size_t aligned_reserve = staging_reserve_bytes == 0
                                     ? 0
                                     : AlignUpChecked(staging_reserve_bytes,
                                                      alignment);
  if (aligned_chunk == std::numeric_limits<size_t>::max() ||
      aligned_max == std::numeric_limits<size_t>::max() ||
      aligned_reserve == std::numeric_limits<size_t>::max() ||
      aligned_chunk > aligned_max ||
      (aligned_reserve != 0 &&
       (aligned_reserve != aligned_chunk ||
        aligned_reserve == aligned_max))) {
    return false;
  }
  std::lock_guard<std::mutex> lock(mu_);
  if (!chunks_.empty()) {
    return chunk_bytes_ == aligned_chunk && max_bytes_ == aligned_max &&
           alignment_ == alignment &&
           staging_reserve_bytes_ == aligned_reserve;
  }
  chunk_bytes_ = aligned_chunk;
  max_bytes_ = aligned_max;
  staging_reserve_bytes_ = aligned_reserve;
  alignment_ = alignment;
  auto initial = std::make_unique<Chunk>();
  initial->segment = std::make_unique<RecvSegment>();
  initial->staging_only = staging_reserve_bytes_ != 0;
  if (!initial->segment->Init(chunk_bytes_, alignment_, -1, huge_pages_)) {
    chunk_bytes_ = 0;
    max_bytes_ = 0;
    alignment_ = 0;
    staging_reserve_bytes_ = 0;
    return false;
  }
  chunks_.push_back(std::move(initial));
  return true;
}

std::unique_ptr<RecvSegmentPool::Chunk> RecvSegmentPool::NewChunk(
    size_t minimum_bytes, int affinity, int numa_node) {
  const size_t aligned_min = AlignUpChecked(minimum_bytes, alignment_);
  if (aligned_min == std::numeric_limits<size_t>::max()) return nullptr;
  size_t committed = 0;
  for (const auto& chunk : chunks_) committed += chunk->segment->size();
  if (committed >= max_bytes_ || aligned_min > max_bytes_ - committed)
    return nullptr;
  const size_t bytes =
      std::min(std::max(chunk_bytes_, aligned_min), max_bytes_ - committed);
  auto chunk = std::make_unique<Chunk>();
  chunk->segment = std::make_unique<RecvSegment>();
  chunk->affinity = affinity;
  if (!chunk->segment->Init(bytes, alignment_, numa_node, huge_pages_))
    return nullptr;
  return chunk;
}

RecvSegmentPool::Lease RecvSegmentPool::AllocateFromChunks(
    size_t bytes, size_t alignment, int affinity, LeaseClass lease_class) {
  auto try_chunks = [&](int wanted_affinity, bool staging_only) -> Lease {
    for (const auto& chunk : chunks_) {
      if (chunk->affinity != wanted_affinity ||
          chunk->staging_only != staging_only) continue;
      auto lease = chunk->segment->Allocate(bytes, alignment);
      if (lease) return Lease(this, chunk->segment.get(), std::move(lease));
    }
    return {};
  };
  if (lease_class == LeaseClass::kStaging && staging_reserve_bytes_ != 0) {
    auto reserved = try_chunks(-1, true);
    if (reserved) return reserved;
  }
  if (affinity >= 0) {
    auto lease = try_chunks(affinity, false);
    if (lease) return lease;
  }
  return try_chunks(-1, false);
}

RecvSegmentPool::Lease RecvSegmentPool::Allocate(size_t bytes,
                                                 size_t alignment,
                                                 int affinity,
                                                 int numa_node,
                                                 LeaseClass lease_class) {
  if (bytes == 0 || !IsPowerOfTwo(alignment)) return {};
  // Chunks a trim unlinks below are freed after the lock is released.
  std::vector<std::unique_ptr<Chunk>> unlinked;
  struct FreeAfterUnlock {
    RecvSegmentPool* pool;
    std::vector<std::unique_ptr<Chunk>>* chunks;
    ~FreeAfterUnlock() { pool->FreeUnlinked(chunks); }
  } free_after_unlock{this, &unlinked};
  std::lock_guard<std::mutex> lock(mu_);
  auto lease = AllocateFromChunks(bytes, alignment, affinity, lease_class);
  if (lease) return lease;
  auto chunk = NewChunk(bytes, affinity, numa_node);
  // Empty chunks (of another rail, say, or left by a failover) hold budget
  // until TrimIdle releases them, up to DFKV_RDMA_RECV_CHUNK_IDLE_MS later.
  // A growth the budget refuses releases them now and tries again.
  if (!chunk &&
      TrimEmptyLocked(AlignUpChecked(bytes, alignment_), &unlinked) != 0)
    chunk = NewChunk(bytes, affinity, numa_node);
  if (!chunk) {
    allocation_failures_.fetch_add(1, std::memory_order_relaxed);
    growth_failures_.fetch_add(1, std::memory_order_relaxed);
    return {};
  }
  RecvSegment* owner = chunk->segment.get();
  chunks_.push_back(std::move(chunk));
  growths_.fetch_add(1, std::memory_order_relaxed);
  auto grown = owner->Allocate(bytes, alignment);
  if (!grown) {
    allocation_failures_.fetch_add(1, std::memory_order_relaxed);
    return {};
  }
  return Lease(this, owner, std::move(grown));
}

size_t RecvSegmentPool::TrimIdle(uint64_t idle_ms) {
  if (idle_ms == 0) return 0;
  const uint64_t now = SteadyMs();
  size_t released = 0;
  // Two phases: unlinked under the pool lock, freed outside it.
  std::vector<std::unique_ptr<Chunk>> unlinked;
  struct FreeAfterUnlock {
    RecvSegmentPool* pool;
    std::vector<std::unique_ptr<Chunk>>* chunks;
    ~FreeAfterUnlock() { pool->FreeUnlinked(chunks); }
  } free_after_unlock{this, &unlinked};
  std::lock_guard<std::mutex> lock(mu_);
  for (size_t i = chunks_.size(); i > 1; --i) {
    auto& chunk = chunks_[i - 1];
    // Under the pool lock no allocation can lease from it meanwhile; a
    // release in progress holds the segment's lock, so 0 is read until it
    // is done.
    const uint64_t empty_since = chunk->segment->EmptySinceMs();
    if (empty_since == 0 || now < empty_since ||
        now - empty_since < idle_ms)
      continue;
    released += chunk->segment->size();
    unlinked.push_back(std::move(chunk));
    chunks_.erase(chunks_.begin() + static_cast<std::ptrdiff_t>(i - 1));
  }
  if (released != 0) {
    shrinks_.fetch_add(1, std::memory_order_relaxed);
    released_bytes_.fetch_add(released, std::memory_order_relaxed);
  }
  return released;
}

void RecvSegmentPool::FreeUnlinked(
    std::vector<std::unique_ptr<Chunk>>* chunks) {
  for (auto& chunk : *chunks) {
    if (release_hook_ && !release_hook_(chunk->segment.get())) {
      // The hook could not take its registrations off: kept, never freed.
      (void)chunk.release();
      kept_chunks_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  chunks->clear();
}

size_t RecvSegmentPool::TrimEmptyLocked(
    size_t wanted, std::vector<std::unique_ptr<Chunk>>* unlinked) {
  size_t committed = 0;
  for (const auto& chunk : chunks_) committed += chunk->segment->size();
  size_t released = 0;
  for (size_t i = chunks_.size(); i > 1; --i) {
    if (committed <= max_bytes_ && max_bytes_ - committed >= wanted) break;
    auto& chunk = chunks_[i - 1];
    if (chunk->segment->stats().used_bytes != 0) continue;
    released += chunk->segment->size();
    committed -= chunk->segment->size();
    unlinked->push_back(std::move(chunk));
    chunks_.erase(chunks_.begin() + static_cast<std::ptrdiff_t>(i - 1));
  }
  if (released != 0) {
    shrinks_.fetch_add(1, std::memory_order_relaxed);
    released_bytes_.fetch_add(released, std::memory_order_relaxed);
  }
  return released;
}

RecvSegmentPool::Geometry RecvSegmentPool::Snapshot() const {
  std::lock_guard<std::mutex> lock(mu_);
  Geometry out;
  out.max_bytes = max_bytes_;
  out.chunk_bytes = chunk_bytes_;
  for (size_t i = 0; i < chunks_.size(); ++i) {
    const auto& chunk = chunks_[i];
    ChunkView view;
    view.segment = chunk->segment.get();
    view.size = chunk->segment->size();
    view.affinity = chunk->affinity;
    view.staging_only = chunk->staging_only;
    view.initial = i == 0;
    view.free = chunk->segment->FreeRanges();
    out.committed_bytes += view.size;
    out.chunks.push_back(std::move(view));
  }
  return out;
}

size_t RecvSegmentPool::max_bytes() const {
  std::lock_guard<std::mutex> lock(mu_);
  return max_bytes_;
}

RecvSegment* RecvSegmentPool::initial_segment() const {
  std::lock_guard<std::mutex> lock(mu_);
  return chunks_.empty() ? nullptr : chunks_.front()->segment.get();
}

RecvSegmentPool::Stats RecvSegmentPool::stats() const {
  std::lock_guard<std::mutex> lock(mu_);
  Stats out;
  out.max_bytes = max_bytes_;
  out.chunks = chunks_.size();
  for (const auto& chunk : chunks_) {
    const RecvSegment::Stats current = chunk->segment->stats();
    out.committed_bytes += current.total_bytes;
    out.used_bytes += current.used_bytes;
    out.free_bytes += current.free_bytes;
    out.largest_free_range =
        std::max(out.largest_free_range, current.largest_free_range);
    if (chunk->staging_only) {
      out.staging_reserved_bytes += current.total_bytes;
      out.staging_reserved_free_bytes += current.free_bytes;
    } else {
      out.connection_free_bytes += current.free_bytes;
      out.connection_largest_free_range =
          std::max(out.connection_largest_free_range,
                   current.largest_free_range);
    }
  }
  out.growths = growths_.load(std::memory_order_relaxed);
  out.shrinks = shrinks_.load(std::memory_order_relaxed);
  out.released_bytes = released_bytes_.load(std::memory_order_relaxed);
  out.allocation_failures =
      allocation_failures_.load(std::memory_order_relaxed);
  out.growth_failures = growth_failures_.load(std::memory_order_relaxed);
  return out;
}

}  // namespace dfkv::rdma
