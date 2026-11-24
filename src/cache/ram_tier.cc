#include "utils/env_parse.h"
#include "cache/ram_tier.h"

#include <cassert>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>

#include "utils/log.h"
#include "common/config_dump.h"
#include "utils/thread_name.h"
#include "utils/numa_util.h"

namespace dfkv {

uint64_t AnonHugePageBytesIn(const void* start, size_t bytes) {
  std::FILE* f = std::fopen("/proc/self/smaps", "r");
  if (f == nullptr) return 0;
  const auto first = reinterpret_cast<uintptr_t>(start);
  // To the end of the last page: the kernel rounds the length of an madvise,
  // mlock or mbind up to whole pages, so a mapping made by one of them for
  // this range ends there, not at start + bytes.
  const auto page = static_cast<uintptr_t>(::sysconf(_SC_PAGESIZE));
  const uintptr_t last = (first + bytes + page - 1) / page * page;
  char line[512];
  bool inside = false;
  uint64_t kb_total = 0;
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    unsigned long long from = 0, to = 0, kb = 0;
    char perms[8];
    // A mapping starts with "from-to perms ..."; its counters follow, each
    // on a line of its own, until the next mapping. Mappings are listed in
    // address order, so the first one past the range ends the pass.
    if (std::sscanf(line, "%llx-%llx %7s", &from, &to, perms) == 3) {
      if (from >= last) break;
      inside = from >= first && to <= last;
    } else if (inside &&
               std::sscanf(line, "AnonHugePages: %llu kB", &kb) == 1) {
      kb_total += kb;
    }
  }
  std::fclose(f);
  return kb_total << 10;
}

namespace {
uint64_t SteadyMs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

constexpr size_t kHugePageBytes = 2u << 20;

// The bracketed (active) value of a transparent_hugepage sysfs setting, for
// example "never" out of "always madvise [never]"; empty if it cannot be read.
std::string ActiveHugePageSetting(const char* path) {
  std::FILE* f = std::fopen(path, "r");
  if (f == nullptr) return {};
  char line[128] = {0};
  const bool read = std::fgets(line, sizeof(line), f) != nullptr;
  std::fclose(f);
  if (!read) return {};
  const char* open = std::strchr(line, '[');
  const char* close = open ? std::strchr(open, ']') : nullptr;
  return close ? std::string(open + 1, close) : std::string();
}

// The system never gives a 2 MiB huge page: madvise is accepted but there is
// nothing to measure. The per-size setting (kernels from 6.8) wins over the
// global one unless it says "inherit". Unknown counts as "may give".
bool SystemHugePagesNever() {
  const std::string per_size = ActiveHugePageSetting(
      "/sys/kernel/mm/transparent_hugepage/hugepages-2048kB/enabled");
  if (!per_size.empty() && per_size != "inherit") return per_size == "never";
  return ActiveHugePageSetting(
             "/sys/kernel/mm/transparent_hugepage/enabled") == "never";
}
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
  uint64_t generation = 0;  // of the disk value being read
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
  {
    envparse::U64Spec spec;
    spec.above = envparse::Above::kClamp;  // strtoull saturates
    spec.legacy = envparse::Legacy::kPrefixOrZero;
    spec.negative = envparse::Negative::kWraps;
    spec.fallback = ext;
    spec.min = 1ull << 20;
    ext = envparse::EnvU64("DFKV_RAM_TIER_EXTENT_BYTES", spec);
  }
  ext = std::min(ext, opt_.bytes);

  uint64_t large_reserve = opt_.large_reserve_bytes;
  {
    envparse::U64Spec spec;
    spec.above = envparse::Above::kClamp;  // strtoull saturates
    spec.legacy = envparse::Legacy::kFull;
    spec.negative = envparse::Negative::kWraps;
    spec.fallback = large_reserve;
    large_reserve = envparse::EnvU64("DFKV_RAM_TIER_LARGE_RESERVE_BYTES", spec);
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
  // Transparent huge pages: with 4 KiB pages the kernel hands out about
  // 4 GiB/s however many threads touch the arena (192 GiB: 39-47 s) and needs
  // another 13-21 s to get the pages back at exit; with 2 MiB pages the same
  // arena is resident in seconds and gone in about one. Asked for with
  // madvise, so it works where the system setting is "madvise" or "always"
  // and is a no-op where it is "never" or the kernel has no huge page to give
  // (the fault then falls back to small pages, page by page). An arena
  // smaller than one huge page does not ask. Not asking does not mean small
  // pages: under "always" the kernel uses huge pages for the arena anyway.
  arena_hugepages_requested_ =
      config_dump::EnvBool("DFKV_RAM_TIER_HUGEPAGES", true) &&
      arena_size >= kHugePageBytes;
  // 4096-aligned base => slot addresses (offset is a slot_size multiple, itself
  // a granularity multiple) are O_DIRECT-aligned for the flusher. Aligned to
  // the huge page size when huge pages are wanted, so the first one is whole.
  if (posix_memalign(&p, arena_hugepages_requested_ ? kHugePageBytes : 4096,
                     arena_size) != 0 ||
      p == nullptr)
    return;
  arena_ = static_cast<char*>(p);
#ifdef MADV_HUGEPAGE
  if (arena_hugepages_requested_ &&
      ::madvise(arena_, arena_size, MADV_HUGEPAGE) != 0) {
    // A kernel built without transparent huge pages: small pages, as before.
    const int refused_errno = errno;
    DFKV_LOG_WARN(std::string("ram tier arena: the kernel refused "
                              "transparent huge pages (") +
                  std::strerror(refused_errno) + "); using small pages");
    arena_hugepages_requested_ = false;
  }
#else
  arena_hugepages_requested_ = false;
#endif
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
  // Keep the arena resident. It used to be pinned as a side effect of being
  // registered with every RDMA rail; without that it is ordinary anonymous
  // memory, and a host with swap would page a "RAM tier" out under pressure.
  // MLOCK_ONFAULT locks each page as the pre-fault below touches it instead of
  // faulting the range in itself. Best effort: without the privilege or the
  // memlock limit the tier works as before, just unlocked.
  if (config_dump::EnvBool("DFKV_RAM_TIER_LOCK", true)) {
    int lock_errno = ENOSYS;
#ifdef SYS_mlock2
    constexpr int kMlockOnFault = 0x01;  // MLOCK_ONFAULT
    arena_locked_ =
        ::syscall(SYS_mlock2, arena_, arena_size, kMlockOnFault) == 0;
    lock_errno = errno;
#endif
    if (!arena_locked_)
      DFKV_LOG_WARN(std::string("ram tier arena is not locked in memory (") +
                    std::strerror(lock_errno) +
                    "; check the memlock limit): it can be swapped out under "
                    "memory pressure");
  }
  {
    // Reading smaps walks the page tables of what it lists, which costs
    // seconds for an arena on small pages: skipped where the answer is known
    // to be zero.
    const bool measure_huge =
        arena_hugepages_requested_ && !SystemHugePagesNever();
    const auto t0 = std::chrono::steady_clock::now();
    const unsigned hw = std::thread::hardware_concurrency();
    const size_t nt =
        std::max(1u, std::min(hw ? hw / 4 : 4u, 16u));
    const size_t chunk =
        arena_size / nt + (arena_size % nt != 0 ? 1 : 0);
    std::vector<std::thread> ws;
    // Said before, not only after: on small pages this takes tens of seconds
    // for a large arena, during which the log would otherwise be silent.
    DFKV_LOG_INFO("ram tier arena: faulting in " +
                  std::to_string(arena_bytes_ >> 20) + " MiB (threads=" +
                  std::to_string(nt) + ")");
    for (size_t i = 0; i < nt; ++i) {
      const size_t off = i * chunk;
      if (off >= arena_size) break;
      const size_t n = std::min(chunk, arena_size - off);
      ws.emplace_back([this, off, n] {
        // One write per small page faults every page, huge or not. The
        // contents do not matter: nothing reads arena bytes it has not
        // written. volatile, so the store cannot be turned into "store only
        // if it differs", which would map the shared zero page and never
        // allocate.
        volatile char* const base = arena_ + off;
        for (size_t at = 0; at < n; at += 4096) {
          // A cancelled start stops here, at most 2 MiB later.
          if (at % kHugePageBytes == 0 && opt_.cancel_start &&
              opt_.cancel_start())
            return;
          base[at] = 0;
        }
      });
    }
    for (auto& worker : ws) worker.join();
    if (opt_.cancel_start && opt_.cancel_start()) {
      start_cancelled_ = true;
      DFKV_LOG_INFO("ram tier arena pre-fault given up: the start is not "
                    "going ahead");
      return;
    }
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    if (measure_huge) {
      // Counted in the arena's own mappings, not as a change of the whole
      // process: the disks are opened while the arena is faulted in, and
      // what they allocate meanwhile may sit on huge pages too. The madvise
      // above gave the arena mappings of its own, split at its two ends.
      arena_hugepage_bytes_ = AnonHugePageBytesIn(arena_, arena_size);
    }
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
        ", locked=" + (arena_locked_ ? "yes" : "no") + ", huge pages=" +
        (arena_hugepages_requested_
             ? std::to_string(arena_hugepage_bytes_ >> 20) + " MiB"
             : std::string("not requested")) +
        ")");
  }

  const uint64_t total_extents =
      std::max<uint64_t>(1, arena_bytes_ / extent_bytes_);

  // Shard count: enough shards to take the single lock off the hot path, few
  // enough that every shard keeps a deep extent pool for class coexistence.
  size_t nshards = 8;
  {
    envparse::U64Spec spec;
    spec.representable_max = envparse::kSignedMax;  // read with strtol
    spec.legacy = envparse::Legacy::kPrefixOrZero;
    spec.fallback = nshards;
    spec.min = 1;
    spec.max = kMaxShards;
    nshards = static_cast<size_t>(envparse::EnvU64("DFKV_RAM_TIER_SHARDS", spec));
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
    allocator_options.evict_clear_limit = kEvictClearLimit;
    // Writes here are memory copies: nothing to stripe, so a class keeps one
    // extent with free slots, not eight, and gives up all but one to others.
    allocator_options.stripe_ways = 1;
    shard->alloc =
        std::make_unique<SlabAllocator>(allocator_options);
    shards_.push_back(std::move(shard));
  }
  if (nshards > 1)
    DFKV_LOG_INFO("ram tier sharded: " + std::to_string(nshards) +
                  " shards across " + std::to_string(total_extents) +
                  " extents");

  // Flush workers: distribute round-robin over shards, at least one per shard,
  // and two with more than one disk (see Options::flush_lane).
  const size_t lanes = std::max<size_t>(opt_.flush_lanes, 1);
  uint32_t nf = std::max<uint32_t>(opt_.flush_threads ? opt_.flush_threads : 1,
                                   static_cast<uint32_t>(nshards));
  if (lanes > 1)
    nf = std::max<uint32_t>(nf, static_cast<uint32_t>(2 * nshards));
  config_dump::RecordResolved("DFKV_RAM_FLUSH_THREADS", std::to_string(nf));
  lane_dirty_ = std::make_unique<std::atomic<uint64_t>[]>(lanes);
  for (size_t lane = 0; lane < lanes; ++lane) lane_dirty_[lane].store(0);
  for (size_t i = 0; i < nshards; ++i) {
    Shard& shard = *shards_[i];
    const size_t workers = nf / nshards + (i < nf % nshards ? 1 : 0);
    shard.flushq.resize(lanes);
    shard.lane_busy.assign(lanes, 0);
    shard.lane_limit = lanes > 1 && workers > 1 ? workers - 1 : workers;
  }
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
  StopBackground(/*abandon_backlog=*/false);
  if (arena_registrations_.load(std::memory_order_acquire) != 0) {
    DFKV_LOG_ERROR("ram tier: the arena is freed while an RDMA server still "
                   "holds registrations over it; aborting");
    std::abort();
  }
  if (arena_) std::free(arena_);
}

void RamTier::StopBackground(bool abandon_backlog) {
  if (reclaim_thread_.joinable()) {
    { std::lock_guard<std::mutex> lk(reclaim_mu_); reclaim_stop_ = true; }
    reclaim_cv_.notify_all();
    reclaim_thread_.join();
  }
  for (auto& sh : shards_) {
    std::lock_guard<std::mutex> lk(sh->mu);
    sh->stop = true;
    if (abandon_backlog) sh->abandon = true;
  }
  background_stopping_.store(true, std::memory_order_release);
  for (auto& sh : shards_) sh->cv.notify_all();
  for (auto& f : flushers_) if (f.joinable()) f.join();
}

bool RamTier::WaitForDrain(std::chrono::milliseconds timeout) {
  // From here on a failed flush is tried again at once: a stop does not wait
  // out a backoff, it gives the failing object its remaining attempts and
  // drops it, as it did before there was a backoff.
  draining_.store(true, std::memory_order_release);
  for (auto& sh : shards_) {
    std::lock_guard<std::mutex> lk(sh->mu);
    sh->cv.notify_all();
  }
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (DirtyObjects() != 0) {
    if (std::chrono::steady_clock::now() >= deadline) {
      shutdown_drain_timeouts_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return true;
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
    const size_t want = SlabAllocator::FreeSlotTarget(
        delta[i], c.slot_size, std::numeric_limits<size_t>::max());
    if (free_now >= want || c.slots_per_extent == 0 || s.alloc->PoolExtents() > 0)
      continue;
    size_t need_ext =
        (want - free_now + c.slots_per_extent - 1) / c.slots_per_extent;
    need_ext = std::min<size_t>(need_ext, kGrowExtentsPerTick);
    while (need_ext > 0) {
      const size_t donor =
          SlabAllocator::ColdestGrowDonor(classes, delta, extents, i,
                                          s.alloc->StripeWays());
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
    if (s.queued > 4096) return;
  }
  // -- RECLAIM --
  for (size_t i = 0; i < classes.size(); ++i) {
    if (delta[i] == 0) continue;  // no write demand on this class
    const auto& c = classes[i];
    const size_t target = SlabAllocator::FreeSlotTarget(
        delta[i], c.slot_size, c.resident + c.free_slots);
    if (c.free_slots >= target) continue;
    size_t budget = 4096;  // per-tick per-class bound
    size_t clear_limited_calls = 0;
    for (;;) {
      std::vector<BlockKey> evicted;
      const size_t batch = std::min<size_t>(64, budget);
      size_t got;
      bool clear_limited = false;
      {
        std::lock_guard<std::mutex> lk(s.mu);
        got = s.alloc->ReclaimClass(i, target, batch, &evicted, &clear_limited);
        EraseEvictedLocked(s, evicted);
      }
      if (got > 0) reclaimed_.fetch_add(got, std::memory_order_relaxed);
      budget -= std::min(budget, got);
      if (budget == 0) break;
      // A call cut short by its reference-clearing bound has freed as little
      // as one slot. Continue the sweep in further short lock holds, up to a
      // cap per tick; beyond it admissions evict inline, from this class (the
      // allocator remembers that the last call ended this way). std::mutex is
      // not fair: yield so a read waiting on the shard lock gets it first.
      if (clear_limited) {
        if (++clear_limited_calls >= kReclaimClearLimitedCalls) break;
        std::this_thread::yield();
        continue;
      }
      // Partial batch = ReclaimClass stopped early (target reached, everything
      // pinned, or its cascade-shrink guard fired) -- don't re-invoke.
      if (got < batch) break;
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
    NoteEvicted(it->second);
    ReleaseBudget(it->second.cap);
    payload_bytes_.fetch_sub(it->second.len, std::memory_order_relaxed);
    s.index.erase(it);
  }
}

bool RamTier::LargeIsCurrentLocked(
    const Shard& s, const std::pair<BlockKey, uint64_t>& item) const {
  auto it = s.index.find(item.first);
  return it != s.index.end() && !it->second.in_arena() &&
         it->second.large_seq == item.second;
}

void RamTier::NoteEvicted(const Entry& entry) {
  if (!entry.read) evicted_unread_.fetch_add(1, std::memory_order_relaxed);
  evicted_age_ms_.fetch_add(
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() -
                                entry.admitted_at)
                                .count()),
      std::memory_order_relaxed);
}

void RamTier::NoteLargeLocked(Shard& s, const BlockKey& key) {
  auto it = s.index.find(key);
  if (it == s.index.end()) return;
  it->second.large_seq = ++s.large_seq;
  s.large_keys.emplace_back(key, it->second.large_seq);
  // Pairs of values removed (or put again) since are stale; drop them once
  // the list is twice as long as the budget has values (each is larger than
  // an extent). What stays is one pair per resident value, so this runs
  // again only after as many more insertions.
  const size_t bound = 2 * (large_budget_bytes_ / extent_bytes_ + 1);
  if (s.large_keys.size() <= bound) return;
  std::deque<std::pair<BlockKey, uint64_t>> live;
  for (const auto& item : s.large_keys)
    if (LargeIsCurrentLocked(s, item)) live.push_back(item);
  s.large_keys.swap(live);
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
    // Oldest first; a value that cannot go yet (unwritten, being sent,
    // being removed) keeps its place at the back.
    for (size_t n = s.large_keys.size(); n != 0 && !has_room(); --n) {
      const auto item = s.large_keys.front();
      s.large_keys.pop_front();
      if (!LargeIsCurrentLocked(s, item)) continue;  // stale
      auto it = s.index.find(item.first);
      Entry& entry = it->second;
      if (entry.durable && entry.send_pins == 0 && !entry.remove_pending) {
        NoteEvicted(entry);
        ReleaseLargeBudget(entry.cap);
        payload_bytes_.fetch_sub(entry.len, std::memory_order_relaxed);
        s.index.erase(it);
        large_evictions_.fetch_add(1, std::memory_order_relaxed);
      } else {
        s.large_keys.push_back(item);
      }
    }
  }
}

void RamTier::CompletePut(
    const std::shared_ptr<PutCompletion>& completion, bool success) {
  CompletePut(completion, success ? FlushOutcome::kFlushed
                                  : FlushOutcome::kCanceled);
}

void RamTier::CompletePut(
    const std::shared_ptr<PutCompletion>& completion, FlushOutcome outcome,
    Status refused) {
  if (!completion) return;
  {
    std::lock_guard<std::mutex> lk(completion->mu);
    if (completion->done) return;
    completion->success = outcome == FlushOutcome::kFlushed;
    completion->outcome = outcome;
    completion->refused = refused;
    completion->done = true;
  }
  completion->cv.notify_all();
}

Status RamTier::WaitPut(const std::shared_ptr<PutCompletion>& completion,
                        FlushOutcome* outcome) {
  if (outcome) *outcome = FlushOutcome::kCanceled;
  if (!completion) return Status::kIOError;
  std::unique_lock<std::mutex> lk(completion->mu);
  // The shared completion owns the admitted value, not the caller's payload.
  // Returning a bypass here could persist conflicting duplicate bytes while
  // the original flush still owns its slot (or outlive Remove's disk fence).
  completion->cv.wait(lk, [&completion] { return completion->done; });
  if (outcome) *outcome = completion->outcome;
  return completion->success ? Status::kOk : Status::kIOError;
}

RamTier::PutResult RamTier::AwaitPut(
    const std::shared_ptr<PutCompletion>& completion,
    bool accept_acked_resident,
    std::unique_lock<std::mutex>* admission_lock,
    std::chrono::steady_clock::time_point arrival) {
  using Clock = std::chrono::steady_clock;
  PutResult result{Status::kIOError};
  result.flush = FlushOutcome::kCanceled;
  if (!completion) return result;
  // The shared completion owns the admitted value, not the caller's payload:
  // this PUT writes nothing from here on, so nothing needs the lock that
  // serializes PUTs of the key. Every later PUT of it finds the same owner.
  if (admission_lock && admission_lock->owns_lock()) admission_lock->unlock();
  const bool bounded = opt_.put_wait_ms != 0;
  const Clock::time_point deadline =
      (arrival == Clock::time_point{} ? Clock::now() : arrival) +
      std::chrono::milliseconds(opt_.put_wait_ms);
  const auto answered = [&] {
    return completion->done ||
           (accept_acked_resident && completion->acked_resident);
  };
  const auto ready = [&] {
    return answered() || put_waits_stopped_.load(std::memory_order_acquire);
  };
  const auto expired = [&] { return bounded && Clock::now() >= deadline; };
  std::unique_lock<std::mutex> lk(completion->mu);
  // Only a PUT that has to wait is registered, so an answer that is already
  // there (a RAM-ACK duplicate of a resident value) costs no global lock.
  const bool parked = !ready() && !expired();
  if (parked) {
    // Lock order: put_wait_mu_ before a completion's mutex. ready() is
    // checked again under the completion's mutex before every wait, after
    // registering: CancelPutWaits either finds this waiter or this waiter
    // sees its flag.
    lk.unlock();
    {
      std::lock_guard<std::mutex> wl(put_wait_mu_);
      put_waiters_.insert(completion.get());
    }
    lk.lock();
    if (bounded)
      completion->cv.wait_until(lk, deadline, ready);
    else
      completion->cv.wait(lk, ready);
  }
  bool ended = false, timed_out = false;
  if (completion->done) {
    result.flush = completion->outcome;
    result.status = completion->success ? Status::kOk
                    : completion->outcome == FlushOutcome::kNoRoom
                        ? Status::kCacheFull
                    : completion->outcome == FlushOutcome::kRefused
                        ? completion->refused
                        : Status::kIOError;
  } else if (answered()) {
    result.flush = FlushOutcome::kNone;  // acknowledged from RAM
    result.status = Status::kOk;
  } else {
    // Ran out of time, or the server is stopping. A wait whose time was up
    // counts as a timeout even if the stop has begun meanwhile.
    ended = true;
    timed_out = expired();
    result.status = Status::kCacheFull;
    result.flush = FlushOutcome::kWaitEnded;
  }
  lk.unlock();
  if (parked) {
    std::lock_guard<std::mutex> wl(put_wait_mu_);
    put_waiters_.erase(put_waiters_.find(completion.get()));
  }
  if (!ended) return result;
  // Returning a bypass here could persist conflicting duplicate bytes while
  // the original flush still owns its slot (or outlive Remove's disk fence).
  // The caller is only told that this PUT did not see the value reach disk.
  if (timed_out)
    put_wait_timeouts_.fetch_add(1, std::memory_order_relaxed);
  else
    put_waits_canceled_.fetch_add(1, std::memory_order_relaxed);
  return result;
}

void RamTier::CancelPutWaits() {
  put_waits_stopped_.store(true, std::memory_order_release);
  std::lock_guard<std::mutex> lk(put_wait_mu_);
  for (PutCompletion* completion : put_waiters_) {
    // Taking the waiter's mutex orders this wake-up after its check of the
    // flag: it either saw the flag or is waiting and gets the notification.
    { std::lock_guard<std::mutex> state_lk(completion->mu); }
    completion->cv.notify_all();
  }
}

RamTier::Admission RamTier::Admit(
    const BlockKey& key, const void* data, size_t len, bool client_ack,
    std::shared_ptr<PutCompletion>* out_completion) {
  if (out_completion) out_completion->reset();
  if (!arena_ || len == 0 || data == nullptr) return Admission::kBypass;
  uint64_t requested_cap = 0;
  const uint32_t lane = LaneOf(key);  // outside the shard lock
  // Asked before the shard lock: the disk store's check takes the store's
  // lock, which a slow metadata write can hold, and every RAM hit of this
  // shard would wait behind it. The PUT takes effect at this check: a
  // REMOVE of the key that completes before the lock below orders after it,
  // as it would after the lock is released.
  const bool persisted = opt_.is_persisted && opt_.is_persisted(key);

  Shard& s = ShardFor(key);
  std::shared_ptr<PutCompletion> completion;
  // Keep admission and the first arena reservation in one critical section;
  // releasing and immediately reacquiring this shard only adds contention.
  std::unique_lock<std::mutex> admission_lock(s.mu);
  {
    auto existing = s.index.find(key);
    if (existing != s.index.end()) {
      // A stale copy (below) that a reader still holds goes on to the disk
      // as well; only a copy a REMOVE is taking out refuses the PUT.
      if (existing->second.remove_pending)
        return existing->second.stale ? Admission::kBypass
                                      : Admission::kRejected;
      // A written copy of a key the disk no longer has (it evicted the key,
      // or the key was removed from the disk alone): this PUT is not a
      // duplicate of a stored value. The stale copy goes, and the PUT goes
      // to the disk like any new key there.
      if (existing->second.durable && opt_.is_persisted && !persisted) {
        existing->second.remove_pending = true;
        existing->second.stale = true;
        DropLocked(s, key);
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
      // A cold read is filling a reserved slot with what the disk holds. Its
      // holder may be this very thread, so do not wait for it: the PUT goes
      // to the disk, which answers it as for any key it has.
      if (writing->second->reservation) return Admission::kBypass;
      if (out_completion) *out_completion = writing->second;
      return Admission::kDuplicate;
    }
    if (persisted) return Admission::kPersisted;
    if (client_ack && opt_.ack_high_watermark_pct != 0) {
      const uint64_t high =
          opt_.bytes * std::min<uint32_t>(opt_.ack_high_watermark_pct, 100) / 100;
      // A disk's values may hold twice its share of the watermark.
      const size_t lanes = std::max<size_t>(opt_.flush_lanes, 1);
      const bool lane_full =
          lanes > 1 &&
          lane_dirty_[lane].load(std::memory_order_relaxed) >=
              high / lanes * 2;
      if (lane_full || dirty_bytes_.load(std::memory_order_relaxed) >= high) {
        // Only a NEW key may be rejected. No completion, allocation or flush
        // ownership has been created, and this is NOT a direct-disk bypass.
        ack_backpressure_.fetch_add(1, std::memory_order_relaxed);
        return Admission::kBackpressure;
      }
    }
    // An oversized conflicting duplicate must still join the first value,
    // rather than bypassing RAM with different bytes.
    if (!AlignCapacity(len, opt_.slot_granularity, &requested_cap) ||
        requested_cap > opt_.bytes) {
      put_bypass_.fetch_add(1, std::memory_order_relaxed);
      return Admission::kBypass;
    }
    completion = std::make_shared<PutCompletion>();
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
    admission_lock.unlock();
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
        if (!admission_lock.owns_lock()) admission_lock.lock();
        SlabAllocator::SlotRef ref;
        std::vector<BlockKey> evicted;
        if (!s.alloc->Put(key, len, &ref, &evicted, &slot_handle)) {
          admission_lock.unlock();
          break;
        }
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
      admission_lock.unlock();
      if (!admitted) ReclaimLargeFor(cap);
    }
  }

  if (!admitted) {
    bool canceled = false;
    {
      std::lock_guard<std::mutex> lk(s.mu);
      auto writing = s.writing.find(key);
      if (writing != s.writing.end() && writing->second == completion)
        s.writing.erase(writing);
      std::lock_guard<std::mutex> state_lk(completion->mu);
      canceled = completion->canceled;
    }
    CompletePut(completion, false);
    if (canceled) return Admission::kRejected;
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
      e.client_acked = client_ack;
      e.lane = lane;
      e.dirty_since = std::chrono::steady_clock::now();
      if (client_ack) {
        // Second RAM-ACK writers of this key need not wait any longer.
        {
          std::lock_guard<std::mutex> state_lk(completion->mu);
          completion->acked_resident = true;
        }
        completion->cv.notify_all();
      }
      payload_bytes_.fetch_add(len, std::memory_order_relaxed);
      // Every index insert is new (the writing set and the existing entry stop
      // a duplicate earlier); the payload gauge relies on it.
      const bool inserted = s.index.emplace(key, std::move(e)).second;
      assert(inserted);
      (void)inserted;
      QItem queued;
      queued.key = key;
      queued.lane = lane;
      Enqueue(s, std::move(queued));
      AddDirty(lane, cap);
      if (large) NoteLargeLocked(s, key);
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
    return Admission::kRejected;
  }

  s.cv.notify_one();
  puts_.fetch_add(1, std::memory_order_relaxed);
  return Admission::kAccepted;
}

bool RamTier::Put(const BlockKey& key, const void* data, size_t len) {
  std::shared_ptr<PutCompletion> completion;
  const Admission admission = Admit(key, data, len, false, &completion);
  return admission == Admission::kAccepted || admission == Admission::kDuplicate ||
         admission == Admission::kPersisted;
}

RamTier::PutResult RamTier::PutCommitted(
    const BlockKey& key, const void* data, size_t len,
    std::unique_lock<std::mutex>* admission_lock,
    std::chrono::steady_clock::time_point arrival) {
  std::shared_ptr<PutCompletion> completion;
  const Admission admission = Admit(key, data, len, false, &completion);
  if (admission == Admission::kPersisted) return {Status::kOk};
  if (admission == Admission::kBypass)
    return {Status::kCacheFull, PutDisposition::kBypass};
  if (admission == Admission::kRejected) return {Status::kIOError};
  return AwaitPut(completion, /*accept_acked_resident=*/false,
                  admission_lock, arrival);
}

bool RamTier::ReserveDurable(const BlockKey& key, size_t len,
                             DurableReservation* out, uint64_t generation) {
  const bool reserved = ReserveDurableImpl(key, len, out, generation);
  if (!reserved) promote_declined_.fetch_add(1, std::memory_order_relaxed);
  return reserved;
}

bool RamTier::ReserveDurableImpl(const BlockKey& key, size_t len,
                             DurableReservation* out, uint64_t generation) {
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
  completion->reservation = true;
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
  state->generation = generation;
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
    // As in PutDurable: only a copy of the value the disk still holds.
    if (!canceled && !DiskStillHolds(state.key, state.generation))
      canceled = true;
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
      payload_bytes_.fetch_add(entry.len, std::memory_order_relaxed);
      // Every index insert is new (the writing set and the existing entry stop
      // a duplicate earlier); the payload gauge relies on it.
      const bool inserted = s.index.emplace(state.key, std::move(entry)).second;
      assert(inserted);
      (void)inserted;
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
  if (!canceled)
    promotes_.fetch_add(1, std::memory_order_relaxed);
  else
    reservations_aborted_.fetch_add(1, std::memory_order_relaxed);
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

  reservations_aborted_.fetch_add(1, std::memory_order_relaxed);
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

RamTier::PutResult RamTier::PutWriteBack(
    const BlockKey& key, const void* data, size_t len,
    std::unique_lock<std::mutex>* admission_lock,
    std::chrono::steady_clock::time_point arrival) {
  const bool synchronous = opt_.ack_high_watermark_pct == 0;
  std::shared_ptr<PutCompletion> completion;
  const Admission admission =
      Admit(key, data, len, !synchronous, &completion);
  if (admission == Admission::kPersisted) return {Status::kOk};
  if (admission == Admission::kBypass)
    return {Status::kCacheFull, PutDisposition::kBypass};
  if (admission == Admission::kBackpressure)
    return {Status::kCacheFull, PutDisposition::kBackpressure};
  if (admission == Admission::kRejected) return {Status::kIOError};
  if (synchronous) {
    ack_backpressure_.fetch_add(1, std::memory_order_relaxed);
    return AwaitPut(completion, /*accept_acked_resident=*/false,
                    admission_lock, arrival);
  }
  if (admission == Admission::kDuplicate) {
    // A second writer of a key whose first value is in the tier. If that
    // value was acknowledged from RAM, this writer gets the same
    // acknowledgement, with the same guarantee: the value is resident and
    // its flush is pending. It does not wait for that flush, which can now
    // take seconds while failed attempts are spaced out. If the value is
    // dropped later, that is one lost acknowledged value in the counter,
    // whoever wrote it. Otherwise (the first PUT has ended, or its value is
    // not acknowledged from RAM) the outcome of the first PUT is this one's
    // too.
    const PutResult result =
        AwaitPut(completion, /*accept_acked_resident=*/true, admission_lock,
                 arrival);
    if (result.status == Status::kOk && result.flush == FlushOutcome::kNone)
      ram_acks_.fetch_add(1, std::memory_order_relaxed);
    return result;
  }
  ram_acks_.fetch_add(1, std::memory_order_relaxed);
  return {Status::kOk};
}

bool RamTier::PutDurable(const BlockKey& key, const void* data, size_t len,
                         uint64_t generation) {
  const bool taken = PutDurableImpl(key, data, len, generation);
  if (!taken) promote_declined_.fetch_add(1, std::memory_order_relaxed);
  return taken;
}

bool RamTier::PutDurableImpl(const BlockKey& key, const void* data, size_t len,
                             uint64_t generation) {
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
  // A copy of a value the disk no longer holds is not installed either: the
  // key was removed while it was read (it would come back), or removed and
  // written again (the copy would hide the new value). Judged under the
  // shard lock, so a Remove that follows its disk remove with
  // DropDurableCopy() either finds the copy or it is never installed.
  bool canceled = false;
  {
    std::lock_guard<std::mutex> lk(s.mu);
    {
      std::lock_guard<std::mutex> state_lk(completion->mu);
      canceled = completion->canceled;
    }
    if (!canceled && !DiskStillHolds(key, generation)) canceled = true;
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
      payload_bytes_.fetch_add(len, std::memory_order_relaxed);
      // Every index insert is new (the writing set and the existing entry stop
      // a duplicate earlier); the payload gauge relies on it.
      const bool inserted = s.index.emplace(key, std::move(e)).second;
      assert(inserted);
      (void)inserted;
      if (large) NoteLargeLocked(s, key);
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
                      Hit* out, bool count_access) {
  if (out != nullptr) *out = Hit{};
  if (shards_.empty()) return false;
  const size_t sidx = KeyHash{}(key) % shards_.size();
  Shard& s = *shards_[sidx];
  std::lock_guard<std::mutex> lk(s.mu);
  auto it = s.index.find(key);
  if (it == s.index.end() || it->second.remove_pending) {
    if (count_access) misses_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  Entry& e = it->second;
  if (offset > e.len) return false;
  const uint64_t start = offset;
  const uint64_t avail = e.len - start;
  const uint64_t n = std::min(length == 0 ? avail : length, avail);
  if (out) {
    if (e.in_arena()) {
      // The transfer pin is also the read's CLOCK reference: without it the
      // class hand never sees a read and eviction is FIFO by admission.
      if (!s.alloc->PinForRead(key, e.slot_handle)) return false;
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
  // A pinned hit is a value served from RAM, also on paths that count the
  // hit later (count_access false: the RDMA pull path).
  if (out) e.read = true;
  if (count_access) hits_.fetch_add(1, std::memory_order_relaxed);
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
bool RamTier::Lookup(const BlockKey& key, size_t* value_len,
                     bool* dirty) const {
  if (shards_.empty() || value_len == nullptr) return false;
  const Shard& s = ShardFor(key);
  std::lock_guard<std::mutex> lk(s.mu);
  auto it = s.index.find(key);
  if (it == s.index.end() || it->second.remove_pending) return false;
  *value_len = static_cast<size_t>(it->second.len);
  if (dirty) *dirty = !it->second.durable;
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
      // A canceled reservation is never installed (CommitReservation sees
      // the flag under this lock). Its holder may be this very thread: do
      // not wait for it.
      if (completion->reservation) return true;
    } else {
      auto it = s.index.find(key);
      if (it == s.index.end() || it->second.remove_pending) return false;
      had = true;
      Entry& e = it->second;
      e.remove_pending = true;  // visibility fence precedes every wait
      completion = e.completion;
      for (auto& lane_queue : s.flushq) {
        for (auto q = lane_queue.begin(); q != lane_queue.end();) {
          if (q->key == key) {
            q = lane_queue.erase(q);
            --s.queued;
          } else {
            ++q;
          }
        }
      }
      EraseRetry(s, key);
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
    // A new generation may already have been admitted after this completion
    // retired. Only the removed generation belongs to this cleanup.
    std::lock_guard<std::mutex> lk(s.mu);
    auto it = s.index.find(key);
    if (it != s.index.end() && it->second.completion == completion) {
      Entry& e = it->second;
      if (!e.remove_pending) {
        e.remove_pending = true;
        for (auto& lane_queue : s.flushq) {
          for (auto q = lane_queue.begin(); q != lane_queue.end();) {
            if (q->key == key) {
              q = lane_queue.erase(q);
              --s.queued;
            } else {
              ++q;
            }
          }
        }
        EraseRetry(s, key);
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

bool RamTier::DropDurableCopy(const BlockKey& key) {
  if (shards_.empty()) return false;
  Shard& s = ShardFor(key);
  std::lock_guard<std::mutex> lk(s.mu);
  auto it = s.index.find(key);
  if (it == s.index.end() || it->second.remove_pending ||
      !it->second.durable)
    return false;
  it->second.remove_pending = true;  // hidden now; gone with its last send
  DropLocked(s, key);
  return true;
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
  if (!e.durable) SubDirty(e.lane, cap);
  if (!large) s.alloc->Remove(e.key);
  payload_bytes_.fetch_sub(e.len, std::memory_order_relaxed);
  s.index.erase(it);
  if (large)
    ReleaseLargeBudget(cap);
  else
    ReleaseBudget(cap);
}

void RamTier::FlushLoop(Shard& s) {
  const size_t lanes = s.flushq.size();
  constexpr size_t kNoLane = static_cast<size_t>(-1);
  for (;;) {
    // Drain up to kFlushBatchMax queued items of one disk in one pass (one
    // worker's batch), from a disk that has fewer than lane_limit batches in
    // flight: the next such disk after the one picked last.
    std::vector<QItem> batch;
    std::vector<FlushItem> items;
    size_t lane = kNoLane;
    {
      std::unique_lock<std::mutex> lk(s.mu);
      for (;;) {
        // Items whose pause is over go back to the queue. Once the tier is
        // stopping or draining there are no pauses left to wait out.
        const bool hurry =
            s.stop || draining_.load(std::memory_order_acquire);
        const auto now = std::chrono::steady_clock::now();
        while (!s.retryq.empty() &&
               (hurry || s.retryq.begin()->first <= now)) {
          s.retry_at.erase(s.retryq.begin()->second.key);
          QItem item = std::move(s.retryq.begin()->second);
          s.retryq.erase(s.retryq.begin());
          Enqueue(s, std::move(item));
        }
        if (s.stop && (s.abandon || s.queued == 0)) return;
        for (size_t k = 0; k < lanes; ++k) {
          const size_t candidate = (s.next_lane + k) % lanes;
          if (!s.flushq[candidate].empty() &&
              s.lane_busy[candidate] < s.lane_limit) {
            lane = candidate;
            s.next_lane = (candidate + 1) % lanes;
            break;
          }
        }
        if (lane != kNoLane) break;
        if (s.retryq.empty() || s.stop) {
          s.cv.wait(lk);
        } else {
          const auto earliest = s.retryq.begin()->first;
          s.cv.wait_until(lk, earliest);
        }
      }
      auto& lane_queue = s.flushq[lane];
      const size_t capacity = std::min(lane_queue.size(), kFlushBatchMax);
      batch.reserve(capacity);
      items.reserve(capacity);
      while (!lane_queue.empty() && batch.size() < kFlushBatchMax) {
        QItem queued = std::move(lane_queue.front());
        lane_queue.pop_front();
        --s.queued;
        auto it = s.index.find(queued.key);
        if (it == s.index.end() || it->second.durable ||
            it->second.remove_pending)
          continue;
        // Dequeue and acquire flush ownership in the same critical section.
        // Otherwise Remove/reinsert can replace the generation in between.
        it->second.flushing = true;
        items.push_back(FlushItem{queued.key, it->second.data(arena_),
                                  static_cast<size_t>(it->second.len),
                                  static_cast<size_t>(it->second.cap)});
        batch.push_back(std::move(queued));
      }
      s.flush_inflight += batch.size();
      ++s.lane_busy[lane];
    }
    const size_t B = batch.size();

    // Flush: batched sink when wired (one store visit for the whole dequeue),
    // else the per-item sink. Every item is owned until this batch completes.
    std::vector<Status> result(B, Status::kIOError);
    if (flush_batch_ && B != 0) {
      std::vector<Status> results = flush_batch_(items);
      for (size_t i = 0; i < B && i < results.size(); ++i)
        result[i] = results[i];
    } else {
      for (size_t i = 0; i < B; ++i)
        result[i] = flush_ ? flush_(items[i].key, items[i].data, items[i].len,
                                    items[i].cap)
                           : Status::kOk;
    }

    // Outside the shard lock: the health streak, and whether a failed
    // object's disk has failed for good (then it is not tried again). A disk
    // with no room has done neither.
    std::vector<char> target_failed(B, 0);
    for (size_t i = 0; i < B; ++i) {
      if (result[i] == Status::kCacheFull) {
        flush_no_room_.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      if (Refused(result[i])) continue;
      NoteFlushOutcome(result[i] == Status::kOk);
      if (result[i] != Status::kOk && opt_.flush_target_failed)
        target_failed[i] = opt_.flush_target_failed(batch[i].key) ? 1 : 0;
    }
    bool dropped_now = false, dropped_target_failed = false;
    {
      std::lock_guard<std::mutex> lk(s.mu);
      for (size_t i = 0; i < B; ++i) {
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
        } else if (result[i] == Status::kOk) {
          entry.durable = true;
          if (entry.client_acked) {
            const double seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - entry.dirty_since)
                    .count();
            ack_to_durable_latency_.Observe(seconds);
          }
          SubDirty(entry.lane, entry.cap);
          if (entry.flush_pin) {
            if (entry.in_arena())
              s.alloc->Unpin(entry.key,
                             entry.slot_handle);  // release the flush-pin
            entry.flush_pin = false;
          }
          CompletePut(entry.completion, true);
          flushed_.fetch_add(1, std::memory_order_relaxed);
        } else if (result[i] == Status::kCacheFull) {
          // No room on a healthy disk. An acknowledged value waits for room,
          // for a while, or while the tier drains for a stop gets
          // flush_retries attempts without pauses, as a failing one does; a
          // waiting PUT is told kCacheFull now.
          const auto now = std::chrono::steady_clock::now();
          QItem& item = batch[i];
          if (item.no_room_tries++ == 0) item.no_room_since = now;
          const bool draining = draining_.load(std::memory_order_acquire);
          if (draining) ++item.no_room_drain_tries;
          const bool hold =
              entry.client_acked &&
              (draining
                   ? item.no_room_drain_tries < opt_.flush_retries
                   : now - item.no_room_since <
                         std::chrono::milliseconds(opt_.flush_no_room_hold_ms));
          if (hold) {
            const uint64_t shift =
                2 * std::min<uint32_t>(item.no_room_tries - 1, 3);
            item.not_before =
                draining
                    ? now
                    : now + std::chrono::milliseconds(
                                static_cast<uint64_t>(opt_.flush_retry_base_ms)
                                << shift);
            const BlockKey retry_key = item.key;
            const auto due = item.not_before;
            s.retry_at[retry_key] = s.retryq.emplace(due, std::move(item));
            s.cv.notify_one();
          } else {
            const auto completion = entry.completion;
            if (entry.flush_pin) {
              if (entry.in_arena())
                s.alloc->Unpin(entry.key, entry.slot_handle);
              entry.flush_pin = false;
            }
            if (opt_.flush_given_up) opt_.flush_given_up(item.key);
            // Counted before the drop: the drop is what a WaitForDrain sees.
            flush_no_room_dropped_.fetch_add(1, std::memory_order_relaxed);
            DropLocked(s, item.key);
            if (after_drop_for_test_) after_drop_for_test_();
            CompletePut(completion, FlushOutcome::kNoRoom);
          }
        } else if (Refused(result[i])) {
          // The disk is fine and will never take this value (too large for
          // it, or over its tenant's quota): no retry, no health change, and
          // a waiting PUT gets the disk's answer.
          const auto completion = entry.completion;
          if (entry.flush_pin) {
            if (entry.in_arena())
              s.alloc->Unpin(entry.key, entry.slot_handle);
            entry.flush_pin = false;
          }
          if (opt_.flush_given_up) opt_.flush_given_up(batch[i].key);
          flush_refused_.fetch_add(1, std::memory_order_relaxed);
          DropLocked(s, batch[i].key);
          if (after_drop_for_test_) after_drop_for_test_();
          CompletePut(completion, FlushOutcome::kRefused, result[i]);
        } else if (!target_failed[i] &&
                   ++batch[i].tries < opt_.flush_retries) {
          // Try again: after a pause if the value was acknowledged already,
          // at once if its PUT is still waiting or the tier is being drained
          // for a stop. The object stays resident, dirty and pinned.
          flush_retries_.fetch_add(1, std::memory_order_relaxed);
          const uint64_t shift = 2 * std::min<uint32_t>(batch[i].tries - 1, 10);
          const bool pause = entry.client_acked &&
                             !draining_.load(std::memory_order_acquire);
          const uint64_t pause_ms =
              pause ? static_cast<uint64_t>(opt_.flush_retry_base_ms) << shift
                    : 0;
          batch[i].not_before = std::chrono::steady_clock::now() +
                                std::chrono::milliseconds(pause_ms);
          const BlockKey retry_key = batch[i].key;
          const auto due = batch[i].not_before;
          s.retry_at[retry_key] = s.retryq.emplace(due, std::move(batch[i]));
          s.cv.notify_one();
        } else {
          const auto completion = entry.completion;
          const bool client_acked = entry.client_acked;
          if (entry.flush_pin) {
            if (entry.in_arena())
              s.alloc->Unpin(entry.key, entry.slot_handle);
            entry.flush_pin = false;
          }
          if (client_acked)
            post_ack_flush_failures_.fetch_add(1, std::memory_order_relaxed);
          if (opt_.flush_given_up) opt_.flush_given_up(batch[i].key);
          // Every count and the health before the drop: the drop takes the
          // value out of the dirty count, and a WaitForDrain that sees the
          // tier drained must see the drop counted.
          flush_dropped_.fetch_add(1, std::memory_order_relaxed);
          NoteDrop();
          DropLocked(s, batch[i].key);
          if (after_drop_for_test_) after_drop_for_test_();
          dropped_now = true;
          dropped_target_failed = dropped_target_failed || target_failed[i];
          // Publish every terminal side effect before waking PutCommitted()
          // leaders and duplicate followers, and tell them why.
          CompletePut(completion,
                      target_failed[i]
                          ? FlushOutcome::kDroppedStoreFailed
                          : draining_.load(std::memory_order_acquire)
                                ? FlushOutcome::kDroppedAtShutdown
                                : FlushOutcome::kDroppedAfterRetries);
        }
      }
      s.flush_inflight -= B;
      --s.lane_busy[lane];
    }
    // A worker may be waiting for this disk to drop below its limit.
    if (lanes > 1) s.cv.notify_all();
    if (dropped_now) LogDrop(dropped_target_failed);
  }
}

// Health follows the flushes: a drop makes the tier unhealthy (in FlushLoop);
// health_recovery_flushes successes in a row, no sooner than
// health_recovery_min_ms after the last drop, make it healthy again. A failed
// attempt, also one that is retried, starts the count again.
void RamTier::NoteFlushOutcome(bool ok) {
  // The usual case: healthy and the flush succeeded. No lock.
  if (ok && healthy_.load(std::memory_order_acquire)) return;
  std::lock_guard<std::mutex> lk(health_mu_);
  if (!ok) {
    healthy_streak_ = 0;
    return;
  }
  if (healthy_streak_ < opt_.health_recovery_flushes) ++healthy_streak_;
  (void)RecoveredLocked();
}

void RamTier::NoteDrop() {
  std::lock_guard<std::mutex> lk(health_mu_);
  last_drop_ms_ = SteadyMs();
  healthy_streak_ = 0;
  healthy_.store(false, std::memory_order_release);
}

// With health_mu_ held: whether the tier is healthy, making it so when the
// streak is complete and the minimum time since the last drop has passed.
bool RamTier::RecoveredLocked() const {
  if (healthy_.load(std::memory_order_acquire)) return true;
  if (healthy_streak_ < opt_.health_recovery_flushes) return false;
  if (SteadyMs() - last_drop_ms_ < opt_.health_recovery_min_ms) return false;
  healthy_.store(true, std::memory_order_release);
  DFKV_LOG_INFO("ram tier: no object dropped for " +
                std::to_string(opt_.health_recovery_min_ms / 1000) +
                " s and " + std::to_string(opt_.health_recovery_flushes) +
                " flushes in a row have succeeded; the tier reports itself "
                "healthy again");
  return true;
}

bool RamTier::healthy() const {
  if (healthy_.load(std::memory_order_acquire)) return true;
  std::lock_guard<std::mutex> lk(health_mu_);
  return RecoveredLocked();
}

// One line per interval for the objects dropped since the previous line.
// Their counters (dfkv_ram_flush_dropped_total,
// dfkv_ram_post_ack_flush_failures_total) see every one.
void RamTier::LogDrop(bool target_failed) {
  constexpr uint64_t kIntervalMs = 10000;
  const uint64_t now_ms = SteadyMs();
  uint64_t last = drop_log_ms_.load(std::memory_order_relaxed);
  if (last != 0 && now_ms - last < kIntervalMs) return;
  if (!drop_log_ms_.compare_exchange_strong(last, now_ms == 0 ? 1 : now_ms))
    return;
  const uint64_t total = flush_dropped_.load(std::memory_order_relaxed);
  const uint64_t acked =
      post_ack_flush_failures_.load(std::memory_order_relaxed);
  const uint64_t new_total =
      total - drop_logged_total_.exchange(total, std::memory_order_relaxed);
  const uint64_t new_acked =
      acked - drop_logged_acked_.exchange(acked, std::memory_order_relaxed);
  DFKV_LOG_WARN(
      "ram tier: dropped " + std::to_string(new_total) +
      " object(s) that could not be written to disk, " +
      std::to_string(new_acked) +
      " of them already acknowledged to the client (now lost: a later GET "
      "misses)" +
      (target_failed
           ? "; the disk store they belong to has failed for good"
           : "; every attempt to write them failed, see the disk store's "
             "errors") +
      ". The tier reports itself unhealthy until flushes succeed again.");
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
    n += sh->queued + sh->retryq.size() + sh->flush_inflight;
  }
  return n;
}

uint32_t RamTier::LaneOf(const BlockKey& key) const {
  if (opt_.flush_lanes <= 1 || !opt_.flush_lane) return 0;
  const size_t lane = opt_.flush_lane(key);
  // A key no disk takes is written (and refused) like any other: lane 0.
  return lane < opt_.flush_lanes ? static_cast<uint32_t>(lane) : 0;
}

void RamTier::Enqueue(Shard& s, QItem item) {
  const size_t lane = item.lane < s.flushq.size() ? item.lane : 0;
  s.flushq[lane].push_back(std::move(item));
  ++s.queued;
}

void RamTier::AddDirty(uint32_t lane, uint64_t bytes) {
  dirty_bytes_.fetch_add(bytes, std::memory_order_relaxed);
  dirty_objects_.fetch_add(1, std::memory_order_relaxed);
  lane_dirty_[lane].fetch_add(bytes, std::memory_order_relaxed);
}

void RamTier::SubDirty(uint32_t lane, uint64_t bytes) {
  dirty_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
  // Release: what was published before (a drop's counters) is seen by the
  // reader that sees the object leave the count (DirtyObjects, acquire).
  dirty_objects_.fetch_sub(1, std::memory_order_release);
  lane_dirty_[lane].fetch_sub(bytes, std::memory_order_relaxed);
}

uint64_t RamTier::OldestDirtyAgeSeconds() const {
  const auto now = std::chrono::steady_clock::now();
  std::chrono::steady_clock::duration oldest{};
  for (const auto& sh : shards_) {
    std::lock_guard<std::mutex> lk(sh->mu);
    for (const auto& [_, entry] : sh->index) {
      if (!entry.durable && !entry.remove_pending) {
        oldest = std::max(oldest, now - entry.dirty_since);
      }
    }
  }
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::seconds>(oldest).count());
}

}  // namespace dfkv
