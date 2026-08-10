#include "cache/ram_tier.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "utils/log.h"
#include "common/config_dump.h"
#include "utils/thread_name.h"
#include "utils/numa_util.h"

namespace dfkv {
namespace {
bool AlignCapacity(size_t len, uint64_t align, uint64_t* out) {
  if (align == 0 || len > UINT64_MAX - (align - 1)) return false;
  uint64_t cap = (static_cast<uint64_t>(len) + align - 1) / align * align;
  if (cap == 0) cap = align;
  *out = cap;
  return true;
}
}  // namespace

struct RamTier::DurableReservation::State {
  RamTier* owner = nullptr;
  size_t shard_index = 0;
  BlockKey key;
  size_t len = 0;
  uint64_t offset = 0;
  uint64_t cap = 0;
  SlabAllocator::SlotHandle slot_handle;
  std::shared_ptr<PutCompletion> completion;
  bool finished = false;
  bool installed = false;
};
RamTier::DurableReservation::DurableReservation() noexcept = default;


RamTier::DurableReservation::~DurableReservation() { Reset(); }

RamTier::DurableReservation::DurableReservation(
    DurableReservation&& other) noexcept
    : state_(std::move(other.state_)) {}

RamTier::DurableReservation& RamTier::DurableReservation::operator=(
    DurableReservation&& other) noexcept {
  if (this != &other) {
    Reset();
    state_ = std::move(other.state_);
  }
  return *this;
}

char* RamTier::DurableReservation::data() const noexcept {
  return state_ ? state_->owner->arena_ + state_->offset : nullptr;
}

size_t RamTier::DurableReservation::capacity() const noexcept {
  return state_ ? static_cast<size_t>(state_->cap) : 0;
}

bool RamTier::DurableReservation::Commit() noexcept {
  return state_ && state_->owner->CommitReservation(this);
}

void RamTier::DurableReservation::Reset() noexcept {
  if (state_) state_->owner->AbortReservation(this);
}


RamTier::RamTier(Options opt, FlushFn flush)
    : opt_(opt), flush_(std::move(flush)) {
  if (opt_.slot_granularity < 4096)
    opt_.slot_granularity = 4096;  // O_DIRECT floor
  if (opt_.bytes < opt_.slot_granularity)
    opt_.bytes = opt_.slot_granularity;

  uint64_t ext = 32ull << 20;
  const bool extent_explicit =
      std::getenv("DFKV_RAM_TIER_EXTENT_BYTES") != nullptr;
  if (const char* e = std::getenv("DFKV_RAM_TIER_EXTENT_BYTES")) {
    const uint64_t value = std::strtoull(e, nullptr, 10);
    if (value >= (1ull << 20)) ext = value;
  }
  ext = std::min(ext, opt_.bytes);

  uint64_t large_reserve = opt_.large_reserve_bytes;
  if (const char* value =
          std::getenv("DFKV_RAM_TIER_LARGE_RESERVE_BYTES")) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end != value && *end == '\0')
      large_reserve = static_cast<uint64_t>(parsed);
  }
  if (large_reserve == UINT64_MAX) {
    if (!extent_explicit && ext == opt_.bytes) {
      large_reserve = 0;
    } else {
      large_reserve =
          ext > UINT64_MAX / 2 ? UINT64_MAX : 2 * ext;
    }
  }
  const uint64_t max_reserve = opt_.bytes - opt_.slot_granularity;
  large_reserve = std::min(large_reserve, max_reserve);
  arena_bytes_ =
      ((opt_.bytes - large_reserve) / opt_.slot_granularity) *
      opt_.slot_granularity;
  if (arena_bytes_ < opt_.slot_granularity)
    arena_bytes_ = opt_.slot_granularity;
  large_budget_bytes_ = opt_.bytes - arena_bytes_;

  ext = std::min(ext, arena_bytes_);
  ext = (ext / opt_.slot_granularity) * opt_.slot_granularity;
  if (ext < opt_.slot_granularity) ext = opt_.slot_granularity;
  extent_bytes_ = ext;
  config_dump::RecordResolved("DFKV_RAM_TIER_EXTENT_BYTES",
                              std::to_string(extent_bytes_));
  config_dump::RecordResolved("DFKV_RAM_TIER_LARGE_RESERVE_BYTES",
                              std::to_string(large_budget_bytes_));

  if (arena_bytes_ > static_cast<uint64_t>(
                         std::numeric_limits<size_t>::max()))
    return;
  const size_t arena_size = static_cast<size_t>(arena_bytes_);
  void* p = nullptr;
  // 4096-aligned base => slot addresses (offset is a slot_size multiple, itself
  // a granularity multiple) are O_DIRECT-aligned for the flusher.
  if (posix_memalign(&p, 4096, arena_size) != 0 || p == nullptr) return;
  arena_ = static_cast<char*>(p);
  // Interleave the fixed-slot arena across NUMA nodes before first touch. A
  // single prefaulting thread would otherwise place every page on its socket,
  // charging remote CPU workers and NIC rails an interconnect hop on every
  // access. Dedicated oversize allocations remain ordinary aligned heap
  // allocations and are independently bounded by large_budget_bytes_.
  const char* nm = std::getenv("DFKV_RAM_TIER_NUMA");
  const std::string numa_mode = (nm && *nm) ? nm : "interleave";
  if (numa_mode != "off" && numa_mode != "interleave") {
    config_dump::RecordResolved("DFKV_RAM_TIER_NUMA", "invalid");
    DFKV_LOG_ERROR("DFKV_RAM_TIER_NUMA must be off or interleave");
    return;
  }
  const bool interleave = numa_mode == "interleave";
  config_dump::RecordResolved("DFKV_RAM_TIER_NUMA", numa_mode);
  const int nodes = numa::OnlineNodeCount();
  if (interleave) numa::InterleaveMemory(arena_, arena_size, nodes);
  {
    const auto t0 = std::chrono::steady_clock::now();
    const unsigned hw = std::thread::hardware_concurrency();
    const size_t nt =
        std::max(1u, std::min(hw ? hw / 4 : 4u, 16u));
    const size_t chunk =
        arena_size / nt + (arena_size % nt != 0 ? 1 : 0);
    std::vector<std::thread> ws;
    for (size_t i = 0; i < nt; ++i) {
      const size_t off = i * chunk;
      if (off >= arena_size) break;
      const size_t n = std::min(chunk, arena_size - off);
      ws.emplace_back([this, off, n] {
        std::memset(arena_ + off, 0, n);
      });
    }
    for (auto& worker : ws) worker.join();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    DFKV_LOG_INFO(
        "ram tier arena pre-faulted: " +
        std::to_string(arena_bytes_ >> 20) + " MiB (total budget " +
        std::to_string(opt_.bytes >> 20) + " MiB, large reserve " +
        std::to_string(large_budget_bytes_ >> 20) + " MiB) in " +
        std::to_string(ms) + " ms (threads=" +
        std::to_string(ws.size()) + ", numa=" +
        (interleave && nodes > 1
             ? "interleave/" + std::to_string(nodes) + "nodes"
             : "default") +
        ")");
  }

  const uint64_t total_extents =
      std::max<uint64_t>(1, arena_bytes_ / extent_bytes_);

  // Shard count: enough shards to take the single lock off the hot path, few
  // enough that every shard keeps a deep extent pool for class coexistence.
  size_t nshards = 8;
  if (const char* s = std::getenv("DFKV_RAM_TIER_SHARDS")) {
    const long value = std::strtol(s, nullptr, 10);
    if (value >= 1 && value <= static_cast<long>(kMaxShards))
      nshards = static_cast<size_t>(value);
  }
  nshards = std::min(nshards, kMaxShards);
  while (nshards > 1 && total_extents / nshards < 32) nshards /= 2;
  if (nshards < 1) nshards = 1;
  config_dump::Record(
      "DFKV_RAM_TIER_SHARDS", std::to_string(nshards),
      std::getenv("DFKV_RAM_TIER_SHARDS")
          ? config_dump::Source::kEnv
          : config_dump::Source::kDefault);

  const uint64_t ext_per_shard = total_extents / nshards;
  const uint64_t ext_remainder = total_extents % nshards;
  shards_.reserve(nshards);
  uint64_t base_extent = 0;
  for (size_t shard_index = 0; shard_index < nshards; ++shard_index) {
    const uint64_t shard_extents =
        ext_per_shard + (shard_index < ext_remainder ? 1 : 0);
    if (shard_extents > std::numeric_limits<uint32_t>::max()) return;
    auto shard = std::make_unique<Shard>();
    shard->base_off = base_extent * extent_bytes_;
    base_extent += shard_extents;
    SlabAllocator::Options allocator_options;
    allocator_options.extent_bytes = extent_bytes_;
    allocator_options.num_extents =
        static_cast<uint32_t>(shard_extents);
    allocator_options.align = opt_.slot_granularity;
    allocator_options.max_waste = 0.25;
    shard->alloc =
        std::make_unique<SlabAllocator>(allocator_options);
    shards_.push_back(std::move(shard));
  }
  if (nshards > 1)
    DFKV_LOG_INFO("ram tier sharded: " + std::to_string(nshards) +
                  " shards across " + std::to_string(total_extents) +
                  " extents");

  // Flush workers: distribute round-robin over shards, at least one per shard.
  const uint32_t nf = std::max<uint32_t>(opt_.flush_threads ? opt_.flush_threads : 1,
                                         static_cast<uint32_t>(nshards));
  config_dump::RecordResolved("DFKV_RAM_FLUSH_THREADS", std::to_string(nf));
  flushers_.reserve(nf);
  for (uint32_t i = 0; i < nf; ++i) {
    Shard& s = *shards_[i % nshards];
    flushers_.emplace_back([this, &s, i] { NameThisThread("rt-flush-", i); FlushLoop(s); });
  }
  if (opt_.reclaim_interval_ms > 0) {
    reclaim_thread_ = std::thread([this] {
      NameThisThread("rt-reclaim");
      std::unique_lock<std::mutex> lk(reclaim_mu_);
      for (;;) {
        reclaim_cv_.wait_for(lk, std::chrono::milliseconds(opt_.reclaim_interval_ms),
                             [this] { return reclaim_stop_; });
        if (reclaim_stop_) return;
        lk.unlock();
        for (size_t s = 0; s < shards_.size(); ++s) ReclaimTick(*shards_[s], s);
        lk.lock();
      }
    });
  }
  ready_ = true;
}

RamTier::~RamTier() {
  if (reclaim_thread_.joinable()) {
    { std::lock_guard<std::mutex> lk(reclaim_mu_); reclaim_stop_ = true; }
    reclaim_cv_.notify_all();
    reclaim_thread_.join();
  }
  for (auto& sh : shards_) {
    std::lock_guard<std::mutex> lk(sh->mu);
    sh->stop = true;
  }
  for (auto& sh : shards_) sh->cv.notify_all();
  for (auto& f : flushers_) if (f.joinable()) f.join();
  if (arena_) std::free(arena_);
}

// One reclaimer pass over one shard (mirror of DiskSlabStore::ReclaimTick,
// arena flavor): for every class with new inserts since the last pass, top its
// free slots up to a demand-driven watermark, in small batches per lock hold.
// Victims are unpinned (== durable, no send in flight), so dropping them from
// the index is the same operation Put performs for its own inline evictions.
void RamTier::ReclaimTick(Shard& s, size_t /*shard_idx*/) {
  const auto classes = s.alloc->Classes();
  if (s.reclaim_last_puts.size() < classes.size())
    s.reclaim_last_puts.resize(classes.size(), 0);
  std::vector<uint64_t> delta(classes.size(), 0);
  std::vector<uint32_t> extents(classes.size(), 0);
  for (size_t i = 0; i < classes.size(); ++i) {
    delta[i] = classes[i].puts - s.reclaim_last_puts[i];
    s.reclaim_last_puts[i] = classes[i].puts;
    extents[i] = classes[i].extents;
  }
  // -- GROW -- (mirror of DiskSlabStore::ReclaimTick; see its comment)
  // Runs even while flush-gated, ON PURPOSE: a cold donor's extents hold
  // DURABLE residents, so moving them to the hot class frees admission
  // capacity precisely when the flusher can't -- the shard's pinned mass is
  // the hot class's own unflushed writes, not the donors'.
  for (size_t i = 0; i < classes.size(); ++i) {
    if (delta[i] == 0) continue;
    const auto& c = classes[i];
    size_t free_now = c.free_slots;
    const size_t want = std::max<size_t>(64, static_cast<size_t>(2 * delta[i]));
    if (free_now >= want || c.slots_per_extent == 0 || s.alloc->PoolExtents() > 0)
      continue;
    size_t need_ext =
        (want - free_now + c.slots_per_extent - 1) / c.slots_per_extent;
    need_ext = std::min<size_t>(need_ext, kGrowExtentsPerTick);
    while (need_ext > 0) {
      size_t donor = classes.size();
      for (size_t d = 0; d < classes.size(); ++d) {  // coldest, then biggest
        if (d == i || extents[d] <= SlabAllocator::kStripeWays) continue;
        if (delta[d] != 0 && delta[d] * 4 > delta[i]) continue;  // not cold enough
        if (donor == classes.size() ||
            std::make_pair(delta[d], ~uint64_t(extents[d])) <
                std::make_pair(delta[donor], ~uint64_t(extents[donor])))
          donor = d;
      }
      if (donor == classes.size()) break;
      std::vector<BlockKey> evicted;
      bool ok;
      {
        std::lock_guard<std::mutex> lk(s.mu);
        ok = s.alloc->StealFrom(donor, i, &evicted);
        EraseEvictedLocked(s, evicted);
      }
      if (!ok) { extents[donor] = 0; continue; }  // donor all pinned: try next
      extents[donor]--;
      rebalanced_.fetch_add(1, std::memory_order_relaxed);
      --need_ext;
    }
  }
  // Flush-gated regime: when the flush queue is deep, the shard is mostly
  // PINNED (not-yet-durable) slots -- free slots are not the admission
  // constraint, and a CLOCK sweep over a pinned-heavy ring just burns lock
  // time skipping entries. Skip the self-eviction phase; it resumes as the
  // flusher catches up. The 4096 budget is per shard.
  {
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.flushq.size() > 4096) return;
  }
  // -- RECLAIM --
  for (size_t i = 0; i < classes.size(); ++i) {
    if (delta[i] == 0) continue;  // no write demand on this class
    const auto& c = classes[i];
    const size_t capacity = c.resident + c.free_slots;
    size_t target = std::max<size_t>(64, static_cast<size_t>(2 * delta[i]));
    target = std::min(target, capacity / 4);
    if (c.free_slots >= target) continue;
    size_t budget = 4096;  // per-tick per-class bound
    for (;;) {
      std::vector<BlockKey> evicted;
      const size_t batch = std::min<size_t>(64, budget);
      size_t got;
      {
        std::lock_guard<std::mutex> lk(s.mu);
        got = s.alloc->ReclaimClass(i, target, batch, &evicted);
        EraseEvictedLocked(s, evicted);
      }
      if (got > 0) reclaimed_.fetch_add(got, std::memory_order_relaxed);
      budget -= std::min(budget, got);
      // Partial batch = ReclaimClass stopped early (target reached, everything
      // pinned, or its cascade-shrink guard fired) -- don't re-invoke.
      if (got < batch || budget == 0) break;
    }
  }
}

void RamTier::SetArenaMr(void* mr) {
  arena_mr_.store(mr, std::memory_order_release);
}

bool RamTier::TryReserve(uint64_t bytes) {
  uint64_t used = budget_used_.load(std::memory_order_relaxed);
  for (;;) {
    if (bytes > opt_.bytes || used > opt_.bytes - bytes) return false;
    if (budget_used_.compare_exchange_weak(
            used, used + bytes, std::memory_order_acq_rel,
            std::memory_order_relaxed))
      return true;
  }
}

bool RamTier::TryReserveLarge(uint64_t bytes) {
  uint64_t used = large_used_.load(std::memory_order_relaxed);
  for (;;) {
    if (bytes > large_budget_bytes_ ||
        used > large_budget_bytes_ - bytes)
      return false;
    if (large_used_.compare_exchange_weak(
            used, used + bytes, std::memory_order_acq_rel,
            std::memory_order_relaxed))
      break;
  }
  if (TryReserve(bytes)) return true;
  large_used_.fetch_sub(bytes, std::memory_order_acq_rel);
  return false;
}

void RamTier::ReleaseBudget(uint64_t bytes) {
  budget_used_.fetch_sub(bytes, std::memory_order_acq_rel);
}

void RamTier::ReleaseLargeBudget(uint64_t bytes) {
  large_used_.fetch_sub(bytes, std::memory_order_acq_rel);
  ReleaseBudget(bytes);
}

void RamTier::EraseEvictedLocked(
    Shard& s, const std::vector<BlockKey>& keys) {
  for (const BlockKey& key : keys) {
    auto it = s.index.find(key);
    if (it == s.index.end()) continue;
    ReleaseBudget(it->second.cap);
    s.index.erase(it);
  }
}

void RamTier::ReclaimLargeFor(uint64_t bytes) {
  if (bytes > large_budget_bytes_) return;
  auto has_room = [this, bytes] {
    const uint64_t used = large_used_.load(std::memory_order_acquire);
    return used <= large_budget_bytes_ - bytes;
  };
  for (auto& shard : shards_) {
    if (has_room()) return;
    Shard& s = *shard;
    std::lock_guard<std::mutex> lk(s.mu);
    for (auto it = s.index.begin(); it != s.index.end();) {
      Entry& entry = it->second;
      if (!entry.in_arena() && entry.durable &&
          entry.send_pins == 0 && !entry.remove_pending) {
        ReleaseLargeBudget(entry.cap);
        it = s.index.erase(it);
        large_evictions_.fetch_add(1, std::memory_order_relaxed);
        if (has_room()) return;
      } else {
        ++it;
      }
    }
  }
}

void RamTier::CompletePut(
    const std::shared_ptr<PutCompletion>& completion, bool success) {
  if (!completion) return;
  {
    std::lock_guard<std::mutex> lk(completion->mu);
    if (completion->done) return;
    completion->success = success;
    completion->done = true;
  }
  completion->cv.notify_all();
}

Status RamTier::WaitPut(
    const std::shared_ptr<PutCompletion>& completion) {
  if (!completion) return Status::kIOError;
  std::unique_lock<std::mutex> lk(completion->mu);
  completion->cv.wait(lk, [&completion] { return completion->done; });
  return completion->success ? Status::kOk : Status::kIOError;
}

RamTier::Admission RamTier::Admit(
    const BlockKey& key, const void* data, size_t len,
    std::shared_ptr<PutCompletion>* out_completion) {
  if (out_completion) out_completion->reset();
  if (!arena_ || len == 0 || data == nullptr) return Admission::kBypass;
  uint64_t requested_cap = 0;
  if (!AlignCapacity(len, opt_.slot_granularity, &requested_cap) ||
      requested_cap > opt_.bytes) {
    put_bypass_.fetch_add(1, std::memory_order_relaxed);
    return Admission::kBypass;
  }

  Shard& s = ShardFor(key);
  auto completion = std::make_shared<PutCompletion>();
  {
    std::lock_guard<std::mutex> lk(s.mu);
    auto existing = s.index.find(key);
    if (existing != s.index.end()) {
      if (existing->second.remove_pending) {
        put_bypass_.fetch_add(1, std::memory_order_relaxed);
        return Admission::kBypass;
      }
      if (!existing->second.completion) {
        existing->second.completion = std::make_shared<PutCompletion>();
        CompletePut(existing->second.completion, existing->second.durable);
      }
      if (out_completion) *out_completion = existing->second.completion;
      return Admission::kDuplicate;
    }
    auto writing = s.writing.find(key);
    if (writing != s.writing.end()) {
      if (out_completion) *out_completion = writing->second;
      return Admission::kDuplicate;
    }
    s.writing.emplace(key, completion);
  }
  if (out_completion) *out_completion = completion;

  const bool large = static_cast<uint64_t>(len) > extent_bytes_;
  uint64_t offset = 0;
  uint64_t cap = requested_cap;
  AlignedBuffer dedicated;
  SlabAllocator::SlotHandle slot_handle;
  bool admitted = false;

  if (large) {
    if (!TryReserveLarge(cap)) {
      ReclaimLargeFor(cap);
      admitted = TryReserveLarge(cap);
    } else {
      admitted = true;
    }
    if (admitted) {
      void* p = nullptr;
      if (posix_memalign(&p, 4096, cap) != 0 || p == nullptr) {
        ReleaseLargeBudget(cap);
        admitted = false;
      } else {
        dedicated.reset(static_cast<char*>(p));
      }
    }
  } else {
    for (int attempt = 0; attempt < 2 && !admitted; ++attempt) {
      {
        std::lock_guard<std::mutex> lk(s.mu);
        SlabAllocator::SlotRef ref;
        std::vector<BlockKey> evicted;
        if (!s.alloc->Put(key, len, &ref, &evicted, &slot_handle)) break;
        EraseEvictedLocked(s, evicted);
        cap = ref.slot_size;
        if (!TryReserve(cap)) {
          s.alloc->Remove(key);
        } else {
          s.alloc->Pin(key, slot_handle);  // flush-pin
          offset = s.base_off + ref.extent * extent_bytes_ + ref.offset;
          admitted = true;
        }
      }
      if (!admitted) ReclaimLargeFor(cap);
    }
  }

  if (!admitted) {
    {
      std::lock_guard<std::mutex> lk(s.mu);
      auto writing = s.writing.find(key);
      if (writing != s.writing.end() && writing->second == completion)
        s.writing.erase(writing);
    }
    CompletePut(completion, false);
    put_bypass_.fetch_add(1, std::memory_order_relaxed);
    return Admission::kBypass;
  }

  char* dst = large ? dedicated.get() : arena_ + offset;
  if (len != 0) std::memcpy(dst, data, len);

  // The canceled check, the writing-set erase and the index install must be
  // ONE s.mu section: Remove() sets canceled while holding s.mu, so reading
  // the flag anywhere else lets the cancel land between the read and the
  // install and resurrects an explicitly Removed key. Nested lock order here
  // is s.mu -> completion->mu, matching Remove() and every CompletePut call
  // site under s.mu -- no path ever takes s.mu while holding completion->mu.
  bool canceled = false;
  {
    std::lock_guard<std::mutex> lk(s.mu);
    {
      std::lock_guard<std::mutex> state_lk(completion->mu);
      canceled = completion->canceled;
    }
    s.writing.erase(key);
    if (!canceled) {
      Entry e;
      e.key = key;
      e.offset = offset;
      e.len = len;
      e.cap = cap;
      e.slot_handle = slot_handle;
      e.large = std::move(dedicated);
      e.completion = completion;
      e.durable = false;
      e.flush_pin = true;
      s.index.emplace(key, std::move(e));
      s.flushq.push_back(QItem{key, 0});
    } else if (!large) {
      s.alloc->Unpin(key, slot_handle);
      s.alloc->Remove(key);
    }
  }
  if (canceled) {
    if (large)
      ReleaseLargeBudget(cap);
    else
      ReleaseBudget(cap);
    CompletePut(completion, false);
    return Admission::kAccepted;
  }

  s.cv.notify_one();
  puts_.fetch_add(1, std::memory_order_relaxed);
  return Admission::kAccepted;
}

bool RamTier::Put(const BlockKey& key, const void* data, size_t len) {
  std::shared_ptr<PutCompletion> completion;
  return Admit(key, data, len, &completion) != Admission::kBypass;
}

Status RamTier::PutCommitted(const BlockKey& key, const void* data, size_t len) {
  if (len == 0 || data == nullptr) return Status::kInvalid;
  std::shared_ptr<PutCompletion> completion;
  const Admission admission = Admit(key, data, len, &completion);
  if (admission == Admission::kBypass) return Status::kCacheFull;
  return WaitPut(completion);
}

bool RamTier::ReserveDurable(const BlockKey& key, size_t len,
                             DurableReservation* out) {
  if (out == nullptr) return false;
  out->Reset();
  if (!arena_ || len == 0 || static_cast<uint64_t>(len) > extent_bytes_)
    return false;

  uint64_t requested_cap = 0;
  if (!AlignCapacity(len, opt_.slot_granularity, &requested_cap) ||
      requested_cap > opt_.bytes)
    return false;

  const size_t shard_index = KeyHash{}(key) % shards_.size();
  Shard& s = *shards_[shard_index];
  auto completion = std::make_shared<PutCompletion>();
  uint64_t offset = 0;
  uint64_t cap = requested_cap;
  SlabAllocator::SlotHandle slot_handle;
  bool admitted = false;

  for (int attempt = 0; attempt < 2 && !admitted; ++attempt) {
    {
      std::lock_guard<std::mutex> lk(s.mu);
      auto existing = s.index.find(key);
      if (existing != s.index.end() || s.writing.count(key)) return false;
      SlabAllocator::SlotRef ref;
      std::vector<BlockKey> evicted;
      if (!s.alloc->Put(key, len, &ref, &evicted, &slot_handle)) break;
      EraseEvictedLocked(s, evicted);
      cap = ref.slot_size;
      if (!TryReserve(cap)) {
        s.alloc->Remove(key);
      } else {
        s.alloc->Pin(key, slot_handle);
        s.writing.emplace(key, completion);
        offset = s.base_off + ref.extent * extent_bytes_ + ref.offset;
        admitted = true;
      }
    }
    if (!admitted) ReclaimLargeFor(cap);
  }
  if (!admitted) return false;

  auto state = std::make_unique<DurableReservation::State>();
  state->owner = this;
  state->shard_index = shard_index;
  state->key = key;
  state->len = len;
  state->offset = offset;
  state->cap = cap;
  state->slot_handle = slot_handle;
  state->completion = std::move(completion);
  out->state_ = std::move(state);
  return true;
}

bool RamTier::CommitReservation(
    DurableReservation* reservation) noexcept {
  if (reservation == nullptr || !reservation->state_ ||
      reservation->state_->finished)
    return false;
  DurableReservation::State& state = *reservation->state_;
  Shard& s = *shards_[state.shard_index];
  bool canceled = false;
  {
    std::lock_guard<std::mutex> lk(s.mu);
    {
      std::lock_guard<std::mutex> state_lk(state.completion->mu);
      canceled = state.completion->canceled;
    }
    auto writing = s.writing.find(state.key);
    if (writing != s.writing.end() &&
        writing->second == state.completion)
      s.writing.erase(writing);
    if (!canceled) {
      Entry entry;
      entry.key = state.key;
      entry.offset = state.offset;
      entry.len = state.len;
      entry.cap = state.cap;
      entry.slot_handle = state.slot_handle;
      entry.durable = true;
      entry.send_pins = 1;
      entry.completion = state.completion;
      CompletePut(state.completion, true);
      s.index.emplace(state.key, std::move(entry));
      state.installed = true;
    } else {
      // Remove() canceled the promotion while storage was filling the slot.
      // Keep the physical slot pinned for the in-flight reply, but hide it and
      // let the reservation destructor release the deferred allocation.
      s.alloc->Remove(state.key);
      CompletePut(state.completion, false);
    }
    state.finished = true;
  }
  if (!canceled) promotes_.fetch_add(1, std::memory_order_relaxed);
  return !canceled;
}

void RamTier::AbortReservation(
    DurableReservation* reservation) noexcept {
  if (reservation == nullptr || !reservation->state_) return;
  std::unique_ptr<DurableReservation::State> state =
      std::move(reservation->state_);
  Shard& s = *shards_[state->shard_index];
  if (state->finished) {
    bool release_budget = !state->installed;
    {
      std::lock_guard<std::mutex> lk(s.mu);
      if (state->installed) {
        auto it = s.index.find(state->key);
        if (it != s.index.end() && it->second.send_pins != 0) {
          Entry& entry = it->second;
          --entry.send_pins;
          s.alloc->Unpin(entry.key, entry.slot_handle);
          if (entry.remove_pending && entry.send_pins == 0 &&
              !entry.flushing)
            DropLocked(s, state->key);
        }
      } else {
        s.alloc->Unpin(state->key, state->slot_handle);
      }
    }
    if (release_budget) ReleaseBudget(state->cap);
    return;
  }

  {
    std::lock_guard<std::mutex> lk(s.mu);
    auto writing = s.writing.find(state->key);
    if (writing != s.writing.end() &&
        writing->second == state->completion)
      s.writing.erase(writing);
    s.alloc->Unpin(state->key, state->slot_handle);
    s.alloc->Remove(state->key);
  }
  ReleaseBudget(state->cap);
  CompletePut(state->completion, false);
}

bool RamTier::PutDurable(const BlockKey& key, const void* data, size_t len) {
  if (!arena_ || len == 0 || data == nullptr) return false;
  uint64_t requested_cap = 0;
  if (!AlignCapacity(len, opt_.slot_granularity, &requested_cap) ||
      requested_cap > opt_.bytes)
    return false;

  Shard& s = ShardFor(key);
  const bool large = static_cast<uint64_t>(len) > extent_bytes_;
  uint64_t offset = 0;
  uint64_t cap = requested_cap;
  AlignedBuffer dedicated;
  SlabAllocator::SlotHandle slot_handle;
  auto completion = std::make_shared<PutCompletion>();

  if (large) {
    {
      std::lock_guard<std::mutex> lk(s.mu);
      auto existing = s.index.find(key);
      if (existing != s.index.end())
        return !existing->second.remove_pending;
      if (s.writing.count(key)) return false;
      s.writing.emplace(key, completion);
    }
    if (!TryReserveLarge(cap)) {
      ReclaimLargeFor(cap);
      if (!TryReserveLarge(cap)) {
        std::lock_guard<std::mutex> lk(s.mu);
        s.writing.erase(key);
        CompletePut(completion, false);
        return false;
      }
    }
    void* p = nullptr;
    if (posix_memalign(&p, 4096, cap) != 0 || p == nullptr) {
      ReleaseLargeBudget(cap);
      std::lock_guard<std::mutex> lk(s.mu);
      s.writing.erase(key);
      CompletePut(completion, false);
      return false;
    }
    dedicated.reset(static_cast<char*>(p));
  } else {
    bool admitted = false;
    for (int attempt = 0; attempt < 2 && !admitted; ++attempt) {
      {
        std::lock_guard<std::mutex> lk(s.mu);
        auto existing = s.index.find(key);
        if (existing != s.index.end())
          return !existing->second.remove_pending;
        if (s.writing.count(key)) return false;
        SlabAllocator::SlotRef ref;
        std::vector<BlockKey> evicted;
        if (!s.alloc->Put(key, len, &ref, &evicted, &slot_handle)) break;
        EraseEvictedLocked(s, evicted);
        cap = ref.slot_size;
        if (!TryReserve(cap)) {
          s.alloc->Remove(key);
        } else {
          s.alloc->Pin(key, slot_handle);  // protects the copy window
          s.writing.emplace(key, completion);
          offset = s.base_off + ref.extent * extent_bytes_ + ref.offset;
          admitted = true;
        }
      }
      if (!admitted) ReclaimLargeFor(cap);
    }
    if (!admitted) return false;
  }

  char* dst = large ? dedicated.get() : arena_ + offset;
  if (len != 0) std::memcpy(dst, data, len);

  // Same cancel/install atomicity as Admit(): a Remove() that flagged the
  // shared completion while the copy ran must suppress the install -- before
  // this the promotion installed unconditionally and revived a Removed key.
  bool canceled = false;
  {
    std::lock_guard<std::mutex> lk(s.mu);
    {
      std::lock_guard<std::mutex> state_lk(completion->mu);
      canceled = completion->canceled;
    }
    s.writing.erase(key);
    if (!canceled) {
      Entry e;
      e.key = key;
      e.offset = offset;
      e.len = len;
      e.cap = cap;
      e.slot_handle = slot_handle;
      e.large = std::move(dedicated);
      e.durable = true;
      e.completion = completion;
      CompletePut(completion, true);
      s.index.emplace(key, std::move(e));
      if (!large) s.alloc->Unpin(key, slot_handle);
    } else if (!large) {
      s.alloc->Unpin(key, slot_handle);  // release the copy-window pin
      s.alloc->Remove(key);  // free the slot
    }
  }
  if (canceled) {
    if (large)
      ReleaseLargeBudget(cap);
    else
      ReleaseBudget(cap);
    CompletePut(completion, false);  // release Remove's WaitPut
    return false;
  }
  promotes_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool RamTier::GetPrep(const BlockKey& key, uint64_t offset, uint64_t length,
                      Hit* out) {
  if (out != nullptr) *out = Hit{};
  if (shards_.empty()) return false;
  const size_t sidx = KeyHash{}(key) % shards_.size();
  Shard& s = *shards_[sidx];
  std::lock_guard<std::mutex> lk(s.mu);
  auto it = s.index.find(key);
  if (it == s.index.end() || it->second.remove_pending) {
    misses_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  Entry& e = it->second;
  if (offset > e.len) return false;
  const uint64_t start = offset;
  const uint64_t avail = e.len - start;
  const uint64_t n = std::min(length == 0 ? avail : length, avail);
  if (out) {
    if (e.in_arena()) {
      if (!s.alloc->Pin(key, e.slot_handle)) return false;
    }
    ++e.send_pins;
    out->ptr = e.data(arena_) + start;
    out->len = static_cast<size_t>(n);
    out->value_len = static_cast<size_t>(e.len);
    out->in_arena = e.in_arena();
    out->mr = out->in_arena
                  ? arena_mr_.load(std::memory_order_acquire)
                  : nullptr;
    out->owner_ = this;
    out->token_ =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&e));
  }
  hits_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void RamTier::ReleaseToken(uint64_t token) {
  if (shards_.empty() || token == 0) return;
  Entry* entry =
      reinterpret_cast<Entry*>(static_cast<uintptr_t>(token));
  Shard& s = ShardFor(entry->key);
  std::lock_guard<std::mutex> lk(s.mu);
  if (entry->send_pins == 0) return;
  --entry->send_pins;
  if (entry->in_arena()) s.alloc->Unpin(entry->key, entry->slot_handle);
  if (entry->remove_pending && entry->send_pins == 0 && !entry->flushing) {
    const BlockKey key = entry->key;
    DropLocked(s, key);
  }
}

bool RamTier::Contains(const BlockKey& key) const {
  if (shards_.empty()) return false;
  const Shard& s = ShardFor(key);
  std::lock_guard<std::mutex> lk(s.mu);
  auto it = s.index.find(key);
  return it != s.index.end() && !it->second.remove_pending;
}
bool RamTier::Lookup(const BlockKey& key, size_t* value_len) const {
  if (shards_.empty() || value_len == nullptr) return false;
  const Shard& s = ShardFor(key);
  std::lock_guard<std::mutex> lk(s.mu);
  auto it = s.index.find(key);
  if (it == s.index.end() || it->second.remove_pending) return false;
  *value_len = static_cast<size_t>(it->second.len);
  return true;
}


bool RamTier::Remove(const BlockKey& key) {
  if (shards_.empty()) return false;
  Shard& s = ShardFor(key);
  std::shared_ptr<PutCompletion> completion;
  bool had = false;
  {
    std::lock_guard<std::mutex> lk(s.mu);
    auto writing = s.writing.find(key);
    if (writing != s.writing.end()) {
      completion = writing->second;
      had = true;
      std::lock_guard<std::mutex> state_lk(completion->mu);
      completion->canceled = true;
    } else {
      auto it = s.index.find(key);
      if (it == s.index.end() || it->second.remove_pending) return false;
      had = true;
      Entry& e = it->second;
      e.remove_pending = true;  // visibility fence precedes every wait
      completion = e.completion;
      for (auto q = s.flushq.begin(); q != s.flushq.end();) {
        if (q->key == key)
          q = s.flushq.erase(q);
        else
          ++q;
      }
      if (!e.flushing) {
        if (e.flush_pin) {
          if (e.in_arena()) s.alloc->Unpin(e.key, e.slot_handle);
          e.flush_pin = false;
        }
        if (!e.durable) CompletePut(completion, false);
        if (e.send_pins == 0) DropLocked(s, key);
      }
    }
  }
  if (completion) (void)WaitPut(completion);
  {
    // Fallback tombstone: no install path may outlive Remove(). If the key
    // nevertheless sits in the index untombstoned, mark it now (purging its
    // queued flush, mirroring phase one) and run the normal drop path, so
    // Remove never returns while the key is still visible.
    std::lock_guard<std::mutex> lk(s.mu);
    auto it = s.index.find(key);
    if (it != s.index.end()) {
      Entry& e = it->second;
      if (!e.remove_pending) {
        e.remove_pending = true;
        for (auto q = s.flushq.begin(); q != s.flushq.end();) {
          if (q->key == key)
            q = s.flushq.erase(q);
          else
            ++q;
        }
        if (!e.flushing) {
          if (e.flush_pin) {
            if (e.in_arena()) s.alloc->Unpin(e.key, e.slot_handle);
            e.flush_pin = false;
          }
          if (!e.durable) CompletePut(e.completion, false);
        }
      }
      if (!e.flushing && e.send_pins == 0) DropLocked(s, key);
    }
  }
  return had;
}

void RamTier::DropLocked(Shard& s, const BlockKey& key) {
  auto it = s.index.find(key);
  if (it == s.index.end()) return;
  Entry& e = it->second;
  if (e.send_pins != 0 || e.flushing) {
    e.remove_pending = true;
    return;
  }
  const uint64_t cap = e.cap;
  const bool large = !e.in_arena();
  if (!large) s.alloc->Remove(e.key);
  s.index.erase(it);
  if (large)
    ReleaseLargeBudget(cap);
  else
    ReleaseBudget(cap);
}

void RamTier::FlushLoop(Shard& s) {
  for (;;) {
    // Drain up to kFlushBatchMax queued items in one pass (one worker's batch).
    std::vector<QItem> batch;
    {
      std::unique_lock<std::mutex> lk(s.mu);
      s.cv.wait(lk, [&s] { return s.stop || !s.flushq.empty(); });
      if (s.stop && s.flushq.empty()) return;
      while (!s.flushq.empty() && batch.size() < kFlushBatchMax) {
        batch.push_back(std::move(s.flushq.front()));
        s.flushq.pop_front();
      }
      s.flush_inflight += batch.size();
    }

    // Snapshot the slots (guaranteed present: queued items are flush-pinned,
    // so they can't be evicted or Removed). live[] marks items still to flush.
    const size_t B = batch.size();
    std::vector<FlushItem> items(B);
    std::vector<char> live(B, 0);
    {
      std::lock_guard<std::mutex> lk(s.mu);
      for (size_t i = 0; i < B; ++i) {
        auto it = s.index.find(batch[i].key);
        if (it == s.index.end() || it->second.durable ||
            it->second.remove_pending)
          continue;
        it->second.flushing = true;
        items[i] = FlushItem{batch[i].key, it->second.data(arena_),
                             static_cast<size_t>(it->second.len),
                             static_cast<size_t>(it->second.cap)};
        live[i] = 1;
      }
    }

    // Flush: batched sink when wired (one store visit for the whole dequeue),
    // else the per-item sink. Per-item ok/fail semantics identical either way.
    std::vector<char> ok(B, 0);
    if (flush_batch_) {
      std::vector<FlushItem> sub;
      std::vector<size_t> map;
      for (size_t i = 0; i < B; ++i)
        if (live[i]) { sub.push_back(items[i]); map.push_back(i); }
      if (!sub.empty()) {
        std::vector<bool> r = flush_batch_(sub);
        for (size_t m = 0; m < map.size() && m < r.size(); ++m) ok[map[m]] = r[m] ? 1 : 0;
      }
    } else {
      for (size_t i = 0; i < B; ++i)
        if (live[i])
          ok[i] = (flush_ ? flush_(items[i].key, items[i].data, items[i].len, items[i].cap) : true) ? 1 : 0;
    }
    if (!flush_ && !flush_batch_) for (size_t i = 0; i < B; ++i) ok[i] = live[i];

    {
      std::lock_guard<std::mutex> lk(s.mu);
      for (size_t i = 0; i < B; ++i) {
        if (!live[i]) continue;
        auto it = s.index.find(batch[i].key);
        if (it == s.index.end()) continue;  // defensive
        Entry& entry = it->second;
        entry.flushing = false;
        if (entry.remove_pending) {
          if (entry.flush_pin) {
            if (entry.in_arena())
              s.alloc->Unpin(entry.key, entry.slot_handle);
            entry.flush_pin = false;
          }
          CompletePut(entry.completion, false);
          if (entry.send_pins == 0) DropLocked(s, batch[i].key);
        } else if (ok[i]) {
          entry.durable = true;
          if (entry.flush_pin) {
            if (entry.in_arena())
              s.alloc->Unpin(entry.key,
                             entry.slot_handle);  // release the flush-pin
            entry.flush_pin = false;
          }
          CompletePut(entry.completion, true);
          flushed_.fetch_add(1, std::memory_order_relaxed);
        } else if (++batch[i].tries < opt_.flush_retries) {
          s.flushq.push_back(std::move(batch[i]));  // retry later
          s.cv.notify_one();
        } else {
          if (entry.flush_pin) {
            if (entry.in_arena())
              s.alloc->Unpin(entry.key, entry.slot_handle);
            entry.flush_pin = false;
          }
          CompletePut(entry.completion, false);
          DropLocked(s, batch[i].key);
          healthy_.store(false, std::memory_order_release);
          flush_dropped_.fetch_add(1, std::memory_order_relaxed);
        }
      }
      s.flush_inflight -= B;
    }
  }
}

uint64_t RamTier::Evictions() const {
  uint64_t n = large_evictions_.load(std::memory_order_relaxed);
  for (const auto& sh : shards_) n += sh->alloc->Evictions();
  return n;
}

uint64_t RamTier::UsedBytes() const {
  return budget_used_.load(std::memory_order_acquire);
}

size_t RamTier::Count() const {
  size_t n = 0;
  for (const auto& sh : shards_) {
    std::lock_guard<std::mutex> lk(sh->mu);
    for (const auto& entry : sh->index)
      if (!entry.second.remove_pending) ++n;
  }
  return n;
}

size_t RamTier::FlushBacklog() const {
  size_t n = 0;
  for (const auto& sh : shards_) {
    std::lock_guard<std::mutex> lk(sh->mu);
    n += sh->flushq.size() + sh->flush_inflight;
  }
  return n;
}

}  // namespace dfkv
