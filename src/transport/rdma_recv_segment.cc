#include "transport/rdma_recv_segment.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <utility>

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

RecvSegment::~RecvSegment() { std::free(data_); }

bool RecvSegment::Init(size_t bytes, size_t alignment) {
  if (data_) return size_ == bytes && alignment_ == alignment;
  if (bytes == 0 || !IsPowerOfTwo(alignment) || bytes % alignment != 0 ||
      alignment < sizeof(void*)) {
    return false;
  }
  void* raw = nullptr;
  if (::posix_memalign(&raw, alignment, bytes) != 0) return false;
  data_ = static_cast<char*>(raw);
  size_ = bytes;
  alignment_ = alignment;
  free_.emplace(0, bytes);
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
    return Lease(this, begin, bytes);
  }
  allocation_failures_.fetch_add(1, std::memory_order_relaxed);
  return {};
}

size_t RecvSegment::free_bytes() const { return stats().free_bytes; }

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
}

bool RecvSegmentPool::Init(size_t chunk_bytes, size_t max_bytes,
                           size_t alignment) {
  if (!IsPowerOfTwo(alignment) || alignment < sizeof(void*) ||
      chunk_bytes == 0 || max_bytes == 0) {
    return false;
  }
  const size_t aligned_chunk = AlignUpChecked(chunk_bytes, alignment);
  const size_t aligned_max = AlignUpChecked(max_bytes, alignment);
  if (aligned_chunk == std::numeric_limits<size_t>::max() ||
      aligned_max == std::numeric_limits<size_t>::max() ||
      aligned_chunk > aligned_max) {
    return false;
  }
  std::lock_guard<std::mutex> lock(mu_);
  if (!chunks_.empty()) {
    return chunk_bytes_ == aligned_chunk && max_bytes_ == aligned_max &&
           alignment_ == alignment;
  }
  chunk_bytes_ = aligned_chunk;
  max_bytes_ = aligned_max;
  alignment_ = alignment;
  auto initial = std::make_unique<RecvSegment>();
  if (!initial->Init(chunk_bytes_, alignment_)) {
    chunk_bytes_ = 0;
    max_bytes_ = 0;
    alignment_ = 0;
    return false;
  }
  chunks_.push_back(std::move(initial));
  return true;
}

std::unique_ptr<RecvSegment> RecvSegmentPool::NewChunk(
    size_t minimum_bytes) {
  const size_t aligned_min = AlignUpChecked(minimum_bytes, alignment_);
  if (aligned_min == std::numeric_limits<size_t>::max()) return nullptr;
  size_t committed = 0;
  for (const auto& chunk : chunks_) committed += chunk->size();
  if (committed >= max_bytes_ || aligned_min > max_bytes_ - committed)
    return nullptr;
  const size_t bytes =
      std::min(std::max(chunk_bytes_, aligned_min), max_bytes_ - committed);
  auto chunk = std::make_unique<RecvSegment>();
  if (!chunk->Init(bytes, alignment_)) return nullptr;
  return chunk;
}

RecvSegmentPool::Lease RecvSegmentPool::Allocate(size_t bytes,
                                                 size_t alignment) {
  if (bytes == 0 || !IsPowerOfTwo(alignment)) return {};
  std::lock_guard<std::mutex> lock(mu_);
  for (const auto& chunk : chunks_) {
    auto lease = chunk->Allocate(bytes, alignment);
    if (lease) return Lease(chunk.get(), std::move(lease));
  }
  auto chunk = NewChunk(bytes);
  if (!chunk) {
    allocation_failures_.fetch_add(1, std::memory_order_relaxed);
    growth_failures_.fetch_add(1, std::memory_order_relaxed);
    return {};
  }
  RecvSegment* owner = chunk.get();
  chunks_.push_back(std::move(chunk));
  growths_.fetch_add(1, std::memory_order_relaxed);
  auto lease = owner->Allocate(bytes, alignment);
  if (!lease) {
    allocation_failures_.fetch_add(1, std::memory_order_relaxed);
    return {};
  }
  return Lease(owner, std::move(lease));
}

RecvSegment* RecvSegmentPool::initial_segment() const {
  std::lock_guard<std::mutex> lock(mu_);
  return chunks_.empty() ? nullptr : chunks_.front().get();
}

RecvSegmentPool::Stats RecvSegmentPool::stats() const {
  std::lock_guard<std::mutex> lock(mu_);
  Stats out;
  out.max_bytes = max_bytes_;
  out.chunks = chunks_.size();
  for (const auto& chunk : chunks_) {
    const RecvSegment::Stats current = chunk->stats();
    out.committed_bytes += current.total_bytes;
    out.used_bytes += current.used_bytes;
    out.free_bytes += current.free_bytes;
    out.largest_free_range =
        std::max(out.largest_free_range, current.largest_free_range);
  }
  out.growths = growths_.load(std::memory_order_relaxed);
  out.allocation_failures =
      allocation_failures_.load(std::memory_order_relaxed);
  out.growth_failures = growth_failures_.load(std::memory_order_relaxed);
  return out;
}

}  // namespace dfkv::rdma
