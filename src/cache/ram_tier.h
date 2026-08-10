/* RamTier — a registered RAM hot tier that fronts the disk cache.
 *
 * Motivation (P3): a COLD dfkv load is disk-bound (~480 MB/s O_DIRECT), which
 * dominates PD decode TTFT. For PD-warm GETs, serving the KV straight from a
 * pre-registered RAM arena over RDMA -- no open, no pread, no disk -- removes
 * that bottleneck. In server write-back mode, PUT becomes synchronously visible
 * in a RAM slot and flushes to disk before acknowledgement. A whole-value cold
 * GET can instead read directly into a hidden slot and publish it born-durable.
 *
 * Ordinary values use SlabAllocator's dense fixed-slot arena. Values larger
 * than one allocator extent use a dedicated aligned allocation instead: they
 * remain bounded by the same configured byte budget and use copy-backed
 * transport because they are outside the arena MR.
 *
 * Both storage kinds follow the same lifecycle:
 *   - reservation-pin: protects an unpublished cold-read destination.
 *   - flush-pin: protects a write-back value until disk commit.
 *   - send-pin: protects a visible value through transport completion.
 *   durable && no send in flight  <=>  evictable.
 * Arena pins are maintained by SlabAllocator; dedicated entries maintain the
 * equivalent counters directly.
 *
 * Backpressure (gap 10.3): if the arena is full of non-evictable slots (flush
 * fell behind), Put returns false and the caller falls back to the normal
 * synchronous disk write -- never blocks, never fails read-after-write.
 * Alignment (gap 10.4): the arena base is posix_memalign(4096) and slot sizes
 * are 4096-multiples, so a slot address is O_DIRECT-aligned for the flusher.
 *
 * SHARDING: every op used to funnel through ONE mutex (index/alloc/flushq/pins)
 * — with 16 serve threads + 16 flush workers the single lock was the measured
 * write-path serialization point (each RAM op also took the allocator's lock
 * UNDER it: two locks per op). The arena is still one allocation (one reg_mr),
 * but it is partitioned into DFKV_RAM_TIER_SHARDS (default 8) independent
 * shards — own SlabAllocator, mutex, index, flush queue and flush workers —
 * routed by hash(key) so a key's whole lifecycle stays on one shard. Hot paths
 * only contend within a shard. Backpressure is per shard (a full shard
 * declines while others admit); the hash keeps that rare and unbiased.
 *
 * Off by default (DFKV_RAM_TIER=0); this class is the standalone core, wired in
 * by later PRs. RDMA MR registration is deferred (SetArenaMr) so this stays
 * hermetically testable without libibverbs. */
#ifndef DFKV_RAM_TIER_H_
#define DFKV_RAM_TIER_H_
#include <cstdlib>
#include <chrono>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cache/slab_allocator.h"
#include "common/kv_types.h"
#include "common/status.h"

namespace dfkv {
class KvNodeServer;

class RamTier {
 public:
  struct Options {
    uint64_t bytes = (16ull << 30);       // total RAM-tier byte budget
    // Bytes reserved outside the registered fixed-slot arena for dedicated
    // oversized values. UINT64_MAX selects two allocator extents (or zero when
    // the default extent spans the whole budget); 0 disables dedicated
    // residency. DFKV_RAM_TIER_LARGE_RESERVE_BYTES overrides this value.
    uint64_t large_reserve_bytes = UINT64_MAX;
    uint32_t slot_granularity = 4096;    // slot quantum (>= O_DIRECT align)
    uint32_t flush_retries = 3;          // per-item flush attempts before drop
    // Flush worker threads draining the shard queues (distributed round-robin
    // over the shards, at least one per shard). One DIO stream sustains only
    // ~1.6-3 GB/s -- far below the arena's ingest -- so the drain rate, not
    // the arena size, decides when Put backpressure kicks in. The server
    // defaults this to 4x the disk count, capped at 16.
    uint32_t flush_threads = 1;
    // Background free-slot reclaimer cadence (ms; 0 = off). Keeps demand-driven
    // free-slot headroom per class so Put's allocator call stays pop-free-slot;
    // without it a full arena runs the CLOCK eviction sweep inline under the
    // shard lock on every admission.
    uint32_t reclaim_interval_ms = 10;
    // Dirty-byte watermark for RAM-ACK mode. Once reached, new PUTs wait for
    // their flush result instead of acknowledging asynchronously.
    uint32_t ack_high_watermark_pct = 80;
  };

  // Persists a resident value to disk. `data` is 4 KiB aligned and `cap` is
  // the allocation's full aligned size. Ordinary data points into the arena;
  // oversize data points into a dedicated allocation. Both may use O_DIRECT.
  // Zeroing padding bytes inside [len, cap) is fine; readers only see [0, len).
  using FlushFn =
      std::function<bool(const BlockKey& key, char* data, size_t len, size_t cap)>;
  // Batched flush sink (optional). One resident allocation per item, same
  // contract as FlushFn; the server wires this to CacheDirectBatch so a
  // worker's whole dequeue rides one lock-amortized (and, with uring, one-
  // submit) store visit -- small-object flushing is IOPS-bound at one
  // synchronous DIO write per key otherwise. Unset = per-item FlushFn.
  struct FlushItem { BlockKey key; char* data; size_t len; size_t cap; };
  using FlushBatchFn = std::function<std::vector<bool>(const std::vector<FlushItem>&)>;
  void set_flush_batch(FlushBatchFn fn) { flush_batch_ = std::move(fn); }

  // A move-only pinned RAM location. Destruction releases the send pin; moving
  // transfers it. Callers cannot access or manually replay the release token.
  struct Hit {
    Hit() noexcept = default;
    ~Hit();
    Hit(const Hit&) = delete;
    Hit& operator=(const Hit&) = delete;
    Hit(Hit&& other) noexcept;
    Hit& operator=(Hit&& other) noexcept;

    const char* ptr = nullptr;
    size_t len = 0;
    size_t value_len = 0;
    void* mr = nullptr;
    bool in_arena = false;

   private:
    friend class RamTier;
    friend class KvNodeServer;
    void Reset() noexcept;
    uint64_t TransferToken() noexcept;
    RamTier* owner_ = nullptr;
    uint64_t token_ = 0;
  };

  // Move-only arena reservation for a value already durable on disk. The
  // storage read may target data() directly, eliminating PutDurable's
  // payload-sized promotion memcpy. Until Commit(), the key stays hidden and
  // the slot stays pinned; destruction aborts and releases the reservation.
  class DurableReservation {
   public:
    DurableReservation() noexcept;
    ~DurableReservation();
    DurableReservation(const DurableReservation&) = delete;
    DurableReservation& operator=(const DurableReservation&) = delete;
    DurableReservation(DurableReservation&& other) noexcept;
    DurableReservation& operator=(DurableReservation&& other) noexcept;

    explicit operator bool() const noexcept { return state_ != nullptr; }
    char* data() const noexcept;
    size_t capacity() const noexcept;
    bool Commit() noexcept;

   private:
    friend class RamTier;
    struct State;
    void Reset() noexcept;
    std::unique_ptr<State> state_;
  };

  RamTier(Options opt, FlushFn flush);
  ~RamTier();                // stops the flushers, joins

  RamTier(const RamTier&) = delete;
  RamTier& operator=(const RamTier&) = delete;

  bool ok() const { return ready_; }
  bool healthy() const { return healthy_.load(std::memory_order_acquire); }

  // Asynchronous admission used by standalone tier tests and read-promotion
  // internals. true means the key was admitted or joined an identical in-flight
  // PUT; it does NOT claim durability. A duplicate never falls through as
  // capacity backpressure.
  bool Put(const BlockKey& key, const void* data, size_t len);

  // Disk-ACK contract: wait for the admitted key's actual flush result.
  // kCacheFull is the only disk-fallback result; kIOError means an admitted
  // leader exhausted its flush retries (duplicates observe the same result).
  Status PutCommitted(const BlockKey& key, const void* data, size_t len);

  // Cache RAM-ACK contract: acknowledge after the payload is copied into a
  // visible, flush-pinned RAM entry. At the dirty-byte high watermark this
  // request waits for disk commit, bounding acknowledged volatile data.
  Status PutWriteBack(const BlockKey& key, const void* data, size_t len);

  // Read-promotion entry: install a value that is ALREADY durable on disk
  // (a coalesced cold read with fan-in evidence). Same allocation/index logic
  // as Put, but the slot is born durable — never enqueued for flush, never
  // holds a flush-pin, so it costs zero DIO write bandwidth and is evictable
  // by CLOCK the moment no send is in flight. Best-effort: a full shard
  // declines (false) and the promotion is silently skipped; promoted slots can
  // never crowd out the write path's flush resources.
  bool PutDurable(const BlockKey& key, const void* data, size_t len);

  // Reserve a registered-arena slot before an O_DIRECT read, so the read can
  // populate the durable RAM entry in place. Oversized values live outside the
  // registered arena and intentionally decline this zero-copy path.
  bool ReserveDurable(const BlockKey& key, size_t len,
                      DurableReservation* out);

  // On hit, pins the resident allocation and returns one move-only owner.
  // Miss leaves `out` empty.
  bool GetPrep(const BlockKey& key, uint64_t offset, uint64_t length, Hit* out);

  bool Contains(const BlockKey& key) const;
  bool Lookup(const BlockKey& key, size_t* value_len) const;
  bool Remove(const BlockKey& key);  // drop from RAM (cache); true if present


  // Registers the arena's RDMA MR (set once by the RDMA server after reg_mr).
  void SetArenaMr(void* mr);
  char* arena() const { return arena_; }
  uint64_t arena_bytes() const { return arena_bytes_; }
  uint64_t budget_bytes() const { return opt_.bytes; }
  uint64_t large_budget_bytes() const { return large_budget_bytes_; }
  uint64_t large_used_bytes() const {
    return large_used_.load(std::memory_order_acquire);
  }
  size_t shards() const { return shards_.size(); }  // test/diagnostic
  size_t flusher_count() const { return flushers_.size(); }

  // metrics (relaxed)
  uint64_t Hits() const { return hits_.load(std::memory_order_relaxed); }
  uint64_t Misses() const { return misses_.load(std::memory_order_relaxed); }
  uint64_t Puts() const { return puts_.load(std::memory_order_relaxed); }
  uint64_t Promoted() const { return promotes_.load(std::memory_order_relaxed); }
  uint64_t PutBypass() const { return put_bypass_.load(std::memory_order_relaxed); }
  uint64_t Flushed() const { return flushed_.load(std::memory_order_relaxed); }
  uint64_t FlushDropped() const { return flush_dropped_.load(std::memory_order_relaxed); }
  uint64_t Reclaimed() const { return reclaimed_.load(std::memory_order_relaxed); }
  uint64_t Rebalanced() const { return rebalanced_.load(std::memory_order_relaxed); }
  uint64_t Evictions() const;
  uint64_t UsedBytes() const;   // arena slot + dedicated allocation bytes
  size_t Count() const;
  size_t FlushBacklog() const;  // queued or flushing, not-yet-durable items
  uint64_t DirtyBytes() const {
    return dirty_bytes_.load(std::memory_order_relaxed);
  }
  uint64_t DirtyObjects() const {
    return dirty_objects_.load(std::memory_order_relaxed);
  }
  uint64_t OldestDirtyAgeSeconds() const;
  uint64_t RamAcks() const {
    return ram_acks_.load(std::memory_order_relaxed);
  }
  uint64_t AckBackpressure() const {
    return ack_backpressure_.load(std::memory_order_relaxed);
  }
  uint64_t PostAckFlushFailures() const {
    return post_ack_flush_failures_.load(std::memory_order_relaxed);
  }

 private:
  struct FreeAligned {
    void operator()(char* p) const noexcept { std::free(p); }
  };
  using AlignedBuffer = std::unique_ptr<char, FreeAligned>;
  struct PutCompletion {
    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    bool success = false;
    bool canceled = false;
  };
  struct Entry {
    BlockKey key;
    uint64_t offset = 0;   // global arena offset; ignored for dedicated values
    uint64_t len = 0;      // exact payload length
    uint64_t cap = 0;      // charged/aligned allocation size
    SlabAllocator::SlotHandle slot_handle;
    AlignedBuffer large;   // non-null iff this entry is outside the arena
    std::shared_ptr<PutCompletion> completion;
    uint32_t send_pins = 0;
    bool durable = false;
    bool flushing = false;
    bool flush_pin = false;
    bool client_acked = false;
    std::chrono::steady_clock::time_point dirty_since;
    bool remove_pending = false;  // hidden immediately; physical free after pins
    bool in_arena() const { return !large; }
    char* data(char* arena) const { return large ? large.get() : arena + offset; }
  };
  struct KeyHash {
    size_t operator()(const BlockKey& key) const noexcept {
      auto mix = [](uint64_t value) {
        value ^= value >> 30;
        value *= 0xbf58476d1ce4e5b9ULL;
        value ^= value >> 27;
        value *= 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
      };
      uint64_t hash = mix(key.digest_hi + 0x9e3779b97f4a7c15ULL);
      hash ^= mix(key.digest_lo + 0x3c6ef372fe94f82aULL);
      hash ^= mix(key.tenant_hash + 0xdaa66d2c7ddef743ULL);
      return static_cast<size_t>(mix(hash));
    }
  };
  struct QItem { BlockKey key; uint32_t tries = 0; };

  // One independent slice of the tier. All state a key's lifecycle touches
  // (allocator, index, writing set, send-pins, flush queue) lives here, under
  // this shard's mutex only. base_off is the shard's byte offset into arena_.
  struct Shard {
    uint64_t base_off = 0;
    std::unique_ptr<SlabAllocator> alloc;
    mutable std::mutex mu;
    std::unordered_map<BlockKey, Entry, KeyHash> index;
    std::unordered_map<BlockKey, std::shared_ptr<PutCompletion>, KeyHash> writing;
    std::deque<QItem> flushq;
    size_t flush_inflight = 0;
    std::condition_variable cv;
    bool stop = false;
    std::vector<uint64_t> reclaim_last_puts;  // reclaim-thread-local snapshot
  };

  friend class KvNodeServer;
  void ReleaseToken(uint64_t token);
  bool TryReserve(uint64_t bytes);
  bool TryReserveLarge(uint64_t bytes);
  void ReleaseBudget(uint64_t bytes);
  void ReleaseLargeBudget(uint64_t bytes);
  void EraseEvictedLocked(Shard& s, const std::vector<BlockKey>& keys);
  enum class Admission { kAccepted, kDuplicate, kBypass };
  Admission Admit(const BlockKey& key, const void* data, size_t len,
                  bool client_ack,
                  std::shared_ptr<PutCompletion>* completion);
  static void CompletePut(const std::shared_ptr<PutCompletion>& completion,
                          bool success);
  static Status WaitPut(const std::shared_ptr<PutCompletion>& completion);
  bool CommitReservation(DurableReservation* reservation) noexcept;
  void AbortReservation(DurableReservation* reservation) noexcept;
  void ReclaimLargeFor(uint64_t bytes);

  Shard& ShardFor(const BlockKey& key) {
    return *shards_[KeyHash{}(key) % shards_.size()];
  }
  const Shard& ShardFor(const BlockKey& key) const {
    return *shards_[KeyHash{}(key) % shards_.size()];
  }

  void FlushLoop(Shard& s);
  // Max items one flush worker drains per store visit. Big enough to amortize
  // the two store-lock hops + reach a useful uring submit width, small enough
  // that a batch's slots stay flush-pinned only briefly.
  static constexpr size_t kFlushBatchMax = 16;
  // The entry itself is the send-pin release token. unordered_map keeps node
  // addresses stable across rehash, and every removal path defers erasure while
  // send_pins is nonzero, so completion can release without a second token map
  // insertion, allocation, lookup, and index lookup.
  static constexpr size_t kMaxShards = 64;
  void ReclaimTick(Shard& s, size_t shard_idx);
  // Rebalance rate cap: extents moved per tick per hot class (32 MiB default
  // extents; converges in well under a second at the 10 ms tick).
  static constexpr size_t kGrowExtentsPerTick = 8;
  void DropLocked(Shard& s, const BlockKey& key);  // s.mu held

  Options opt_;
  FlushFn flush_;
  FlushBatchFn flush_batch_;
  char* arena_ = nullptr;
  bool ready_ = false;
  std::atomic<bool> healthy_{true};
  uint64_t arena_bytes_ = 0;
  uint64_t large_budget_bytes_ = 0;
  std::atomic<void*> arena_mr_{nullptr};
  uint64_t extent_bytes_ = 0;   // allocator class/extent upper bound
  std::vector<std::unique_ptr<Shard>> shards_;
  std::vector<std::thread> flushers_;
  // Background free-slot reclaimer (own cv/mutex so shutdown never races the
  // flushers' cv protocol). One thread sweeps all shards.
  std::thread reclaim_thread_;
  std::condition_variable reclaim_cv_;
  std::mutex reclaim_mu_;
  bool reclaim_stop_ = false;

  std::atomic<uint64_t> budget_used_{0};
  std::atomic<uint64_t> large_used_{0};
  std::atomic<uint64_t> large_evictions_{0};
  std::atomic<uint64_t> hits_{0}, misses_{0}, puts_{0}, put_bypass_{0}, promotes_{0};
  std::atomic<uint64_t> flushed_{0}, flush_dropped_{0}, reclaimed_{0}, rebalanced_{0};
  std::atomic<uint64_t> dirty_bytes_{0}, dirty_objects_{0};
  std::atomic<uint64_t> ram_acks_{0}, ack_backpressure_{0};
  std::atomic<uint64_t> post_ack_flush_failures_{0};
};

inline RamTier::Hit::~Hit() { Reset(); }

inline RamTier::Hit::Hit(Hit&& other) noexcept {
  *this = std::move(other);
}

inline RamTier::Hit& RamTier::Hit::operator=(Hit&& other) noexcept {
  if (this != &other) {
    Reset();
    ptr = other.ptr;
    len = other.len;
    value_len = other.value_len;
    mr = other.mr;
    in_arena = other.in_arena;
    owner_ = std::exchange(other.owner_, nullptr);
    token_ = std::exchange(other.token_, 0);
    other.ptr = nullptr;
    other.len = 0;
    other.value_len = 0;
    other.mr = nullptr;
    other.in_arena = false;
  }
  return *this;
}

inline void RamTier::Hit::Reset() noexcept {
  RamTier* owner = std::exchange(owner_, nullptr);
  const uint64_t token = std::exchange(token_, 0);
  if (owner != nullptr) owner->ReleaseToken(token);
}

inline uint64_t RamTier::Hit::TransferToken() noexcept {
  owner_ = nullptr;
  return std::exchange(token_, 0);
}

}  // namespace dfkv

#endif  // DFKV_RAM_TIER_H_
