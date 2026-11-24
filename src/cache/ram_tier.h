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
 * Eviction among evictable arena slots is per-size-class CLOCK second chance:
 * a GetPrep hit that hands out a pin (out != nullptr) on an arena slot sets
 * the slot's reference bit, and the class hand clears a set bit and moves on
 * instead of evicting. Existence probes (out == nullptr) and dedicated large
 * values set nothing. A slot read at least once per lap of the hand stays;
 * one that is no longer read goes within one more lap. The sweep runs under
 * the shard lock that reads take, so one eviction clears at most
 * kEvictClearLimit bits and then evicts the first evictable slot it met.
 * Reads do not reorder the ring and do not feed the allocator's class/extent
 * read heat, so whole-extent rebalancing between size classes still judges a
 * class by its writes only.
 *
 * Backpressure: capacity rejection happens before a value is admitted; only
 * then may the caller fall back to synchronous disk write. Once admitted, a
 * disk-ACK PUT (including a conflicting duplicate) waits for that value's
 * terminal flush result. A deadline cannot transfer ownership to another PUT.
 * Alignment (gap 10.4): the arena base is aligned to 4096 (to 2 MiB when
 * huge pages are asked for) and slot sizes are 4096-multiples, so a slot
 * address is O_DIRECT-aligned for the flusher.
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
#include <map>
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
#include "utils/latency_hist.h"

namespace dfkv {

// Bytes backed by transparent huge pages in the mappings of this process
// that lie entirely inside [start, start + bytes), from one pass over
// /proc/self/smaps. A range that mbind, mlock or madvise split into several
// mappings is the sum of them; a mapping that reaches outside the range is
// not counted. The range ends with the page that holds its last byte. 0 if
// the kernel does not say.
uint64_t AnonHugePageBytesIn(const void* start, size_t bytes);
class KvNodeServer;

class RamTier {
 public:
  // Reference bits one eviction may clear (SlabAllocator::Options::
  // evict_clear_limit). The sweep holds the shard and allocator locks, which
  // every read of the shard takes twice, and each step loads one 128-byte
  // slot record from a table far larger than the CPU caches. 512 bounds the
  // bits an inline eviction clears, and a background reclaim call clears
  // fewer than 2 x 512, however many slots the class has (estimated, not
  // measured: about 0.1 us per step, so about 0.05 ms and 0.1 ms). Slots
  // pinned by an unfinished flush are skipped without counting, so a flush
  // backlog at the hand can still lengthen one call, as before. A referenced
  // slot is only evicted when 512 referenced slots in a row stand at the
  // hand; a class in that state evicts in admission order, as it did before
  // reads were recorded.
  static constexpr uint32_t kEvictClearLimit = 512;
  // Reclaim calls per class and tick that may end on that bound before the
  // tick gives up on the class, each call in its own lock hold. A call can
  // clear up to 2 x 512 - 1 bits, so a tick clears just under 64 x 1024 per
  // class and shard; while 512 referenced slots stand at the hand each call
  // frees one slot, so such a tick frees 64.
  static constexpr size_t kReclaimClearLimitedCalls = 64;

  struct Options {
    uint64_t bytes = (16ull << 30);       // total RAM-tier byte budget
    // Bytes reserved outside the registered fixed-slot arena for dedicated
    // oversized values. UINT64_MAX selects two allocator extents (or zero when
    // the default extent spans the whole budget); 0 disables dedicated
    // residency. DFKV_RAM_TIER_LARGE_RESERVE_BYTES overrides this value.
    uint64_t large_reserve_bytes = UINT64_MAX;
    uint32_t slot_granularity = 4096;    // slot quantum (>= O_DIRECT align)
    // A flush that fails is tried again, flush_retries attempts in all.
    //
    // For a value that was already acknowledged to the client (RAM-ACK) a
    // drop is a silent loss, so the attempts are spaced out:
    // flush_retry_base_ms before the second one, four times the previous
    // pause before each further one. The defaults give 50 + 200 + 800 + 3200
    // ms = 4.25 s for the disk to take the write after all (an I/O error
    // that passes, a store that could not place the value under load). The
    // object stays readable and pinned meanwhile and counts as dirty, so the
    // ACK watermark turns a disk that keeps refusing into backpressure. The
    // span is short on purpose: a second PUT of the same key waits for the
    // outcome, under its put-key stripe.
    //
    // A value whose PUT is still waiting for the flush (disk-ACK) is tried
    // again without pauses, as it always was: the client gets its error at
    // once and nothing is lost by failing fast.
    //
    // After the last attempt the object is dropped and counted.
    //
    // A disk that answers kCacheFull has not failed: it has no slot for the
    // value now (every slot of its size class is held by an operation in
    // flight or protected by retention). Such an attempt is not a failure
    // and does not count against flush_retries or the tier's health. A value
    // whose PUT waits for the flush is not tried again: that PUT is answered
    // kCacheFull, as the disk would answer it without a RAM tier. An
    // acknowledged value stays resident, readable and dirty, and is tried
    // again with the same pauses, at most 3.2 s apart, for up to
    // flush_no_room_hold_ms from the first refusal; then it is dropped like
    // a value the cache evicts, counted apart from the lost ones. While the
    // tier drains for a stop it gets flush_retries more attempts without
    // pauses, as a failing value does, and is then dropped the same way.
    uint32_t flush_retries = 5;
    uint32_t flush_retry_base_ms = 50;
    uint32_t flush_no_room_hold_ms = 60000;
    // Whether the disk a key is written to has failed for good. A flush that
    // fails and gets true here is not tried again: nothing will change until
    // the process is restarted. Called from a flush worker, without locks.
    std::function<bool(const BlockKey&)> flush_target_failed;
    // Called for every value dropped without having been written: its
    // flush failed for good, or the disk had no room for it. Not for a
    // removed value (Remove's caller knows) and not for a value written
    // after refused or failed attempts. From a flush worker, under the shard
    // lock of the key, in the same hold that drops the value: a new PUT of
    // the key, and anything the caller does for it, comes after the call.
    // Must not call into the tier.
    std::function<void(const BlockKey&)> flush_given_up;
    // After an object has been dropped the tier reports itself unhealthy
    // until this many flushes in a row have succeeded with no failed attempt
    // in between: a failure that has passed stops being reported, one that
    // keeps hitting part of the writes does not. And for at least
    // health_recovery_min_ms after the last drop, whatever the flush rate: a
    // busy node runs through 64 flushes in milliseconds, and the unhealthy
    // state has to last long enough to be scraped and not to flap.
    uint32_t health_recovery_flushes = 64;
    uint32_t health_recovery_min_ms = 30000;
    // Flush worker threads draining the shard queues (distributed round-robin
    // over the shards, at least one per shard). One DIO stream sustains only
    // ~1.6-3 GB/s -- far below the arena's ingest -- so the drain rate, not
    // the arena size, decides when Put backpressure kicks in. The server
    // defaults this to 4x the disk count, capped at 16.
    uint32_t flush_threads = 1;
    // The disk a key is written to (flush_lane, in [0, flush_lanes)). With
    // more than one disk a flush batch holds the values of one disk only, a
    // shard keeps at least two flush workers and a disk is written by all
    // but one of them at most, so one disk that hangs leaves a worker to the
    // others (two hung disks can still take every worker of a shard that
    // has only two); and a disk's values may make up at most twice its share of
    // the dirty watermark, so the RAM-ACK backpressure it causes is its own.
    std::function<size_t(const BlockKey&)> flush_lane;
    size_t flush_lanes = 1;
    // Background free-slot reclaimer cadence (ms; 0 = off). Keeps demand-driven
    // free-slot headroom per class so Put's allocator call stays pop-free-slot;
    // without it a full arena runs the CLOCK eviction sweep inline under the
    // shard lock on every admission.
    uint32_t reclaim_interval_ms = 10;
    // Positive dirty-byte watermark for RAM-ACK mode: reject NEW keys before
    // admission once reached. Existing owners still resolve normally.
    // 0 explicitly selects synchronous disk-ACK, not perpetual rejection.
    uint32_t ack_high_watermark_pct = 80;
    // Longest a PUT waits for the value that owns its key: in disk-ACK mode
    // every admitted PUT, first writer or duplicate, waits for the flush; a
    // second RAM-ACK writer waits until the first value is resident. After
    // that it is answered kCacheFull -- not stored as far as this request
    // knows, try again -- and the value's flush goes on. The time counts from
    // the arrival of the request batch the PUT came in, so the parked PUTs of
    // one batch share it. Keep it below the clients' operation timeout: a
    // client that times out first treats the node as failed. 0 = no limit.
    uint32_t put_wait_ms = 3000;
    // Optional backing-index check for a key without a RAM owner, and for a
    // written RAM copy of a PUT's key. Invoked before the RAM shard lock (it
    // may wait for the disk store's lock); must be metadata-only and must not
    // reenter RAM.
    // An existing disk value is authoritative after RAM eviction.
    std::function<bool(const BlockKey&)> is_persisted;
    // The generation of the value the disk holds for a key (ValueMetadata),
    // 0 if it holds none. A promotion is installed only if this still equals
    // the generation its read was given: the disk has neither lost the key
    // nor taken a new value of it since. Invoked under the key's RAM shard
    // lock; metadata-only, and must not reenter RAM.
    std::function<uint64_t(const BlockKey&)> disk_generation;
    // Asked while the arena is being faulted in (once per 2 MiB and thread):
    // true gives the start up. The constructor then returns without the
    // arena ready and without threads; ok() is false and start_cancelled()
    // true. Faulting in a large arena on small pages takes tens of seconds,
    // which a stop request during startup should not have to wait for.
    std::function<bool()> cancel_start;
  };

  // Persists a resident value to disk. `data` is 4 KiB aligned and `cap` is
  // the allocation's full aligned size. Ordinary data points into the arena;
  // oversize data points into a dedicated allocation. Both may use O_DIRECT.
  // Zeroing padding bytes inside [len, cap) is fine; readers only see [0, len).
  // Returns the disk's answer: kOk written, kCacheFull no room for it now
  // (see Options::flush_no_room_hold_ms), anything else a failed attempt.
  using FlushFn = std::function<Status(const BlockKey& key, char* data,
                                       size_t len, size_t cap)>;
  // Batched flush sink (optional). One resident allocation per item, same
  // contract as FlushFn; the server wires this to CacheDirectBatch so a
  // worker's whole dequeue rides one lock-amortized (and, with uring, one-
  // submit) store visit -- small-object flushing is IOPS-bound at one
  // synchronous DIO write per key otherwise. Unset = per-item FlushFn.
  struct FlushItem { BlockKey key; char* data; size_t len; size_t cap; };
  using FlushBatchFn =
      std::function<std::vector<Status>(const std::vector<FlushItem>&)>;
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
  ~RamTier();                // StopBackground(false), then frees the arena
  // Stops the reclaim and flush threads and keeps the arena. Without
  // `abandon_backlog` the flushers first write out everything that is queued,
  // however long that takes; with it they finish the batch in hand and leave
  // the rest dirty in RAM, to be discarded with the tier. Precondition, as for
  // destruction: no request is in flight. Idempotent.
  void StopBackground(bool abandon_backlog);
  // The flushers have been told to stop (test/diagnostic).
  bool background_stopping() const {
    return background_stopping_.load(std::memory_order_acquire);
  }

  RamTier(const RamTier&) = delete;
  RamTier& operator=(const RamTier&) = delete;

  bool ok() const { return ready_; }
  // The constructor gave up because Options::cancel_start said so.
  bool start_cancelled() const { return start_cancelled_; }
  // False from a dropped flush until health_recovery_flushes flushes in a row
  // have succeeded and health_recovery_min_ms have passed since the last
  // drop. Evaluated here as well as on a flush, so that a node that goes
  // idle after its streak still recovers when the time is up.
  bool healthy() const;

  // Asynchronous admission used by standalone tier tests. true means the key
  // was admitted, joined an existing PUT, or already exists in the backing
  // store. Admitted duplicates keep the first payload, including its size.
  bool Put(const BlockKey& key, const void* data, size_t len);

  // Only kBypass permits the caller to persist its own payload directly.
  // kBackpressure is a terminal pre-admission rejection, even though both
  // dispositions carry wire status kCacheFull.
  enum class PutDisposition { kComplete, kBypass, kBackpressure };
  // How the flush of an admitted value ended. A PUT that waited for it (a
  // disk-ACK writer, or a second writer of the same key) gets this next to
  // its status; anything but kFlushed is a failure of that PUT.
  enum class FlushOutcome {
    kNone,              // this PUT did not wait for a flush
    kFlushed,           // written to disk
    kDroppedAfterRetries,   // every attempt failed; the value is gone
    kDroppedStoreFailed,    // its disk store has failed for good; gone
    kDroppedAtShutdown,     // still failing when the tier was being drained
    kCanceled,          // removed, or given up before it was admitted
    // The disk had no room for it (kCacheFull); the value is gone and the
    // PUT is answered kCacheFull: not stored, the node is fine.
    kNoRoom,
    // The disk will never take it (kInvalid: too large for it, or
    // kQuotaExceeded); the value is gone, the PUT gets that answer.
    kRefused,
    // The PUT stopped waiting before the flush ended (Options::put_wait_ms
    // passed, or the server is stopping). Status kCacheFull: the value still
    // owns the key and may yet reach disk; this PUT did not see it.
    kWaitEnded,
  };
  struct PutResult {
    Status status;
    PutDisposition disposition = PutDisposition::kComplete;
    FlushOutcome flush = FlushOutcome::kNone;
  };

  // Disk-ACK contract: wait for the admitted key's actual flush result.
  // Admitted failures/cancellations return kIOError, never disk fallback.
  //
  // Both PUT entries may have to wait for a flush. `admission_lock`, if
  // given and held, is the caller's lock serializing PUTs of this key; it is
  // released before such a wait, because from then on this PUT writes
  // nothing: the key's owner is in the RAM index or still being written, and
  // every later PUT of the key joins it. The wait ends with the flush result
  // (kOk / kIOError), or with kCacheFull, disposition kComplete and flush
  // kWaitEnded when Options::put_wait_ms has passed or CancelPutWaits() was
  // called: the value is not known to be on disk, and the caller must not
  // write it itself.
  // put_wait_ms counts from `arrival`, when the request batch this PUT came
  // in arrived (the PUTs of one batch are served one after another and share
  // the client's deadline); a default `arrival` means the start of the wait.
  PutResult PutCommitted(const BlockKey& key, const void* data, size_t len,
                         std::unique_lock<std::mutex>* admission_lock = nullptr,
                         std::chrono::steady_clock::time_point arrival = {});

  // Cache RAM-ACK contract: acknowledge a visible, flush-pinned copy. A
  // positive dirty watermark rejects new keys before admission, without disk
  // fallback. Duplicates join their first owner's terminal completion; zero
  // watermark forces disk-ACK for every admission.
  PutResult PutWriteBack(const BlockKey& key, const void* data, size_t len,
                         std::unique_lock<std::mutex>* admission_lock = nullptr,
                         std::chrono::steady_clock::time_point arrival = {});

  // The server is stopping: answer every PUT that waits for a flush, now and
  // from here on, with kCacheFull instead of making the stop wait for it.
  // Flushing itself is not affected, and a PUT whose flush has already
  // ended, or which RAM may acknowledge, still gets that answer.
  void CancelPutWaits();

  // Read-promotion entry: install a value that is ALREADY durable on disk
  // (a coalesced cold read with fan-in evidence). Same allocation/index logic
  // as Put, but the slot is born durable — never enqueued for flush, never
  // holds a flush-pin, so it costs zero DIO write bandwidth and is evictable
  // by CLOCK the moment no send is in flight. Best-effort: a full shard
  // declines (false) and the promotion is silently skipped; promoted slots can
  // never crowd out the write path's flush resources. `generation`: that of
  // the disk value the copy was read from, taken before the read. With
  // Options::disk_generation set, the copy is not installed unless the disk
  // still holds that value (not if it was removed, or removed and written
  // again, during the read); the same holds for a reservation's Commit.
  bool PutDurable(const BlockKey& key, const void* data, size_t len,
                  uint64_t generation = 0);
  bool PutDurableImpl(const BlockKey& key, const void* data, size_t len,
                      uint64_t generation);

  // Reserve a registered-arena slot before an O_DIRECT read, so the read can
  // populate the durable RAM entry in place. Oversized values live outside the
  // registered arena and intentionally decline this zero-copy path.
  bool ReserveDurable(const BlockKey& key, size_t len,
                      DurableReservation* out, uint64_t generation = 0);
  bool ReserveDurableImpl(const BlockKey& key, size_t len,
                          DurableReservation* out, uint64_t generation);

  // On hit with `out`, pins the resident allocation, marks an arena slot
  // referenced for CLOCK eviction, and returns one move-only owner. Without
  // `out` it is an existence probe: no pin, no reference.
  // Miss leaves `out` empty.
  // Observational probes acquire the same pin but defer hit/miss accounting.
  bool GetPrep(const BlockKey& key, uint64_t offset, uint64_t length, Hit* out,
               bool count_access = true);

  bool Contains(const BlockKey& key) const;
  // `dirty`, if given: the value is still to be written to disk.
  bool Lookup(const BlockKey& key, size_t* value_len,
              bool* dirty = nullptr) const;
  bool Remove(const BlockKey& key);  // drop from RAM (cache); true if present
  // Drops the key's entry if it is a written (durable) copy, as a cache may;
  // a value still to be written is left alone. The second step of a REMOVE,
  // after the disk remove: it takes out a promotion that was installed while
  // the REMOVE was between the tiers. True if a copy was dropped.
  bool DropDurableCopy(const BlockKey& key);


  // Registers the arena's RDMA MR (set once by the RDMA server after reg_mr).
  void SetArenaMr(void* mr);
  uint64_t arena_bytes() const { return arena_bytes_; }
  // The arena and the extent its values never cross, for an RDMA server
  // that registers it per extent (DFKV_RDMA_SERVER_MW).
  char* arena_base() const { return arena_; }
  uint64_t extent_bytes() const { return extent_bytes_; }
  // An RDMA server holds registrations over the arena while this count is
  // nonzero (DFKV_RDMA_SERVER_MW). The arena is mapped once and never
  // released, remapped or madvised away for the tier's life; with a
  // registration over it that is required, and the destructor aborts if
  // one is still held rather than free memory a device can still read.
  void HoldArenaRegistrations(int delta) {
    arena_registrations_.fetch_add(delta, std::memory_order_acq_rel);
  }
  // The arena is locked in memory (mlock2); false if that was turned off or
  // could not be done.
  bool arena_locked() const { return arena_locked_; }
  // Transparent huge pages were asked for and the kernel accepted the request
  // (DFKV_RAM_TIER_HUGEPAGES, default on; never for an arena smaller than one
  // huge page), and how much of the arena they backed right after the
  // pre-fault. The kernel gives what it has: 0 with the system setting
  // "never", less than the arena when free memory is fragmented. Without a
  // request nothing is measured, so 0 then says nothing about the page size:
  // under the system setting "always" the arena is on huge pages regardless.
  bool arena_hugepages_requested() const { return arena_hugepages_requested_; }
  uint64_t arena_hugepage_bytes() const { return arena_hugepage_bytes_; }
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
  // A read this tier would have missed, answered without asking it (a GET
  // found absent by its existence check).
  void NoteMiss() { misses_.fetch_add(1, std::memory_order_relaxed); }
  uint64_t Puts() const { return puts_.load(std::memory_order_relaxed); }
  uint64_t Promoted() const { return promotes_.load(std::memory_order_relaxed); }
  // Promotions the tier did not take (PutDurable or ReserveDurable said
  // no), and reservations that ended without a copy installed (the read
  // failed, or a REMOVE or a newer disk value overtook it).
  uint64_t PromoteDeclined() const {
    return promote_declined_.load(std::memory_order_relaxed);
  }
  uint64_t ReservationsAborted() const {
    return reservations_aborted_.load(std::memory_order_relaxed);
  }
  // Evicted values (capacity) never read from RAM since they were taken in,
  // and the time evicted values had been resident, summed in milliseconds.
  uint64_t EvictedUnread() const {
    return evicted_unread_.load(std::memory_order_relaxed);
  }
  uint64_t EvictedAgeMs() const {
    return evicted_age_ms_.load(std::memory_order_relaxed);
  }
  uint64_t PutBypass() const { return put_bypass_.load(std::memory_order_relaxed); }
  uint64_t Flushed() const { return flushed_.load(std::memory_order_relaxed); }
  uint64_t FlushDropped() const { return flush_dropped_.load(std::memory_order_relaxed); }
  uint64_t Reclaimed() const { return reclaimed_.load(std::memory_order_relaxed); }
  uint64_t Rebalanced() const { return rebalanced_.load(std::memory_order_relaxed); }
  uint64_t Evictions() const;
  uint64_t UsedBytes() const;   // arena slot + dedicated allocation bytes
  // Payload bytes of the values resident in the arena and in dedicated
  // allocations. UsedBytes() minus this is what slot rounding costs.
  uint64_t PayloadBytes() const {
    return payload_bytes_.load(std::memory_order_relaxed);
  }
  size_t Count() const;
  size_t FlushBacklog() const;  // queued or flushing, not-yet-durable items
  uint64_t DirtyBytes() const {
    return dirty_bytes_.load(std::memory_order_relaxed);
  }
  uint64_t DirtyObjects() const {
    return dirty_objects_.load(std::memory_order_acquire);
  }
  uint64_t OldestDirtyAgeSeconds() const;
  uint64_t RamAcks() const {
    return ram_acks_.load(std::memory_order_relaxed);
  }
  uint64_t AckBackpressure() const {
    return ack_backpressure_.load(std::memory_order_relaxed);
  }
  // PUTs parked right now, waiting for the value that owns their key.
  size_t ParkedPuts() const {
    std::lock_guard<std::mutex> lk(put_wait_mu_);
    return put_waiters_.size();
  }
  uint64_t PutWaitTimeouts() const {
    return put_wait_timeouts_.load(std::memory_order_relaxed);
  }
  uint64_t PutWaitsCanceled() const {
    return put_waits_canceled_.load(std::memory_order_relaxed);
  }
  uint64_t PostAckFlushFailures() const {
    return post_ack_flush_failures_.load(std::memory_order_relaxed);
  }
  // Failed flush attempts that were followed by another attempt.
  uint64_t FlushRetries() const {
    return flush_retries_.load(std::memory_order_relaxed);
  }
  // Flush attempts the disk answered kCacheFull (no room now), and the values
  // dropped for that reason (acknowledged ones after flush_no_room_hold_ms).
  uint64_t FlushNoRoom() const {
    return flush_no_room_.load(std::memory_order_relaxed);
  }
  uint64_t FlushNoRoomDropped() const {
    return flush_no_room_dropped_.load(std::memory_order_relaxed);
  }
  // Values dropped because the disk refused them for good (Refused()).
  uint64_t FlushRefused() const {
    return flush_refused_.load(std::memory_order_relaxed);
  }
  std::string AckToDurableLatencyMetrics(const std::string& labels) const {
    return ack_to_durable_latency_.Render(
        "dfkv_ram_ack_to_durable_latency_seconds", labels);
  }
  bool WaitForDrain(std::chrono::milliseconds timeout);
  uint64_t ShutdownDrainTimeouts() const {
    return shutdown_drain_timeouts_.load(std::memory_order_relaxed);
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
    // Set with `done`: kFlushed exactly when `success`.
    FlushOutcome outcome = FlushOutcome::kNone;
    // The value is resident and was acknowledged from RAM to its first
    // writer. A second RAM-ACK writer of the key is given the same
    // acknowledgement at this point instead of waiting for the flush.
    bool acked_resident = false;
    // The completion of a cold-read reservation (ReserveDurable). Only the
    // thread that holds the reservation finishes it, after its disk read and
    // reply, so nothing else may wait for it: that thread may be the waiter.
    bool reservation = false;
    // With outcome kRefused: what the disk answered.
    Status refused = Status::kIOError;
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
    uint32_t lane = 0;  // Options::flush_lane
    std::chrono::steady_clock::time_point dirty_since;
    uint64_t large_seq = 0;  // see Shard::large_keys
    // When the value was taken in, and whether RAM has served it since.
    std::chrono::steady_clock::time_point admitted_at =
        std::chrono::steady_clock::now();
    bool read = false;
    bool remove_pending = false;  // hidden immediately; physical free after pins
    // remove_pending because the disk no longer had the key (a stale written
    // copy dropped by a PUT), not because of a REMOVE.
    bool stale = false;
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
  struct QItem {
    BlockKey key;
    uint32_t tries = 0;
    uint32_t lane = 0;  // Options::flush_lane
    // Attempts the disk answered kCacheFull, and when the first of them was.
    uint32_t no_room_tries = 0;
    std::chrono::steady_clock::time_point no_room_since{};
    // Refused attempts since the tier began to drain for a stop.
    uint32_t no_room_drain_tries = 0;
    // For an item waiting to be tried again: not before this.
    std::chrono::steady_clock::time_point not_before{};
  };

  // One independent slice of the tier. All state a key's lifecycle touches
  // (allocator, index, writing set, send-pins, flush queue) lives here, under
  // this shard's mutex only. base_off is the shard's byte offset into arena_.
  struct Shard {
    uint64_t base_off = 0;
    std::unique_ptr<SlabAllocator> alloc;
    mutable std::mutex mu;
    std::unordered_map<BlockKey, Entry, KeyHash> index;
    std::unordered_map<BlockKey, std::shared_ptr<PutCompletion>, KeyHash> writing;
    // One queue per disk (Options::flush_lane), the batches in flight for
    // each, how many a disk may have (lane_limit), and where the next pick
    // starts. queued: items in all queues.
    std::vector<std::deque<QItem>> flushq;
    std::vector<size_t> lane_busy;
    size_t lane_limit = 0;
    size_t next_lane = 0;
    size_t queued = 0;
    // The dedicated (above-extent) values, oldest first, as (key, the
    // value's large_seq): the victims of ReclaimLargeFor, which so visits
    // these and not the whole index. A pair whose sequence is not that of
    // the key's value now (removed, or removed and put again) is stale and
    // skipped; large_seq numbers the shard's dedicated insertions.
    std::deque<std::pair<BlockKey, uint64_t>> large_keys;
    uint64_t large_seq = 0;
    // Items whose flush failed and that wait for their next attempt, by the
    // time it is due; still resident, dirty and flush-pinned. With a disk
    // that refuses writes this holds every dirty object of that disk, so the
    // flusher takes the due ones from the front and Remove finds its key
    // through retry_at.
    using RetryQueue =
        std::multimap<std::chrono::steady_clock::time_point, QItem>;
    RetryQueue retryq;
    std::unordered_map<BlockKey, RetryQueue::iterator, KeyHash> retry_at;
    size_t flush_inflight = 0;
    std::condition_variable cv;
    bool stop = false;
    bool abandon = false;  // with stop: leave the queue instead of draining it
    std::vector<uint64_t> reclaim_last_puts;  // reclaim-thread-local snapshot
  };

  friend class KvNodeServer;
  friend class RamTierTestPeer;
  // Tests: called with the shard lock held right after a flush that gave up
  // dropped its value.
  std::function<void()> after_drop_for_test_;
  void ReleaseToken(uint64_t token);
  bool TryReserve(uint64_t bytes);
  bool TryReserveLarge(uint64_t bytes);
  void ReleaseBudget(uint64_t bytes);
  void ReleaseLargeBudget(uint64_t bytes);
  void EraseEvictedLocked(Shard& s, const std::vector<BlockKey>& keys);
  void NoteLargeLocked(Shard& s, const BlockKey& key);
  bool LargeIsCurrentLocked(const Shard& s,
                            const std::pair<BlockKey, uint64_t>& item) const;
  enum class Admission {
    kAccepted, kDuplicate, kPersisted, kBypass, kRejected, kBackpressure
  };
  Admission Admit(const BlockKey& key, const void* data, size_t len,
                  bool client_ack,
                  std::shared_ptr<PutCompletion>* completion);
  // The bool form is kFlushed or kCanceled.
  // Whether a promotion read from disk value `generation` may be installed.
  // Under the key's shard lock.
  bool DiskStillHolds(const BlockKey& key, uint64_t generation) const {
    if (!opt_.disk_generation) return true;
    const uint64_t current = opt_.disk_generation(key);
    return current != 0 && current == generation;
  }
  static void CompletePut(const std::shared_ptr<PutCompletion>& completion,
                          bool success);
  static void CompletePut(const std::shared_ptr<PutCompletion>& completion,
                          FlushOutcome outcome,
                          Status refused = Status::kIOError);
  // A flush answer of a healthy disk that will never take the value.
  static bool Refused(Status st) {
    return st == Status::kInvalid || st == Status::kQuotaExceeded;
  }
  static Status WaitPut(const std::shared_ptr<PutCompletion>& completion,
                        FlushOutcome* outcome = nullptr);
  // Every wait of a PUT for the value that owns its key: bounded by
  // Options::put_wait_ms, ended by CancelPutWaits(), and without the
  // caller's admission lock. With `accept_acked_resident` (a second RAM-ACK
  // writer) the wait also ends, with kOk and flush kNone, as soon as that
  // value is resident and acknowledged from RAM. A PUT whose answer is
  // already there (the common RAM-ACK duplicate) is not registered as parked.
  PutResult AwaitPut(const std::shared_ptr<PutCompletion>& completion,
                     bool accept_acked_resident,
                     std::unique_lock<std::mutex>* admission_lock,
                     std::chrono::steady_clock::time_point arrival);
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
  static void EraseRetry(Shard& s, const BlockKey& key) {
    const auto at = s.retry_at.find(key);
    if (at == s.retry_at.end()) return;
    s.retryq.erase(at->second);
    s.retry_at.erase(at);
  }
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
  // The health state: the flag for the fast path, the rest under health_mu_.
  // A drop, a failed attempt and the recovery all take the mutex, so a drop
  // that races with a recovery always wins.
  mutable std::atomic<bool> healthy_{true};
  mutable std::mutex health_mu_;
  uint32_t healthy_streak_ = 0;   // successes since the last failed attempt
  uint64_t last_drop_ms_ = 0;     // steady clock
  void NoteDrop();
  bool RecoveredLocked() const;
  uint64_t arena_bytes_ = 0;
  uint64_t large_budget_bytes_ = 0;
  std::atomic<void*> arena_mr_{nullptr};
  std::atomic<int> arena_registrations_{0};  // see HoldArenaRegistrations
  bool arena_locked_ = false;
  bool arena_hugepages_requested_ = false;
  uint64_t arena_hugepage_bytes_ = 0;
  bool start_cancelled_ = false;
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
  std::atomic<uint64_t> payload_bytes_{0};  // see PayloadBytes()
  std::atomic<uint64_t> large_used_{0};
  std::atomic<uint64_t> large_evictions_{0};
  std::atomic<uint64_t> promote_declined_{0}, reservations_aborted_{0};
  std::atomic<uint64_t> evicted_unread_{0}, evicted_age_ms_{0};
  void NoteEvicted(const Entry& entry);
  std::atomic<uint64_t> hits_{0}, misses_{0}, puts_{0}, put_bypass_{0}, promotes_{0};
  std::atomic<uint64_t> flushed_{0}, flush_dropped_{0}, reclaimed_{0}, rebalanced_{0};
  std::atomic<uint64_t> dirty_bytes_{0}, dirty_objects_{0};
  // Dirty bytes by disk (Options::flush_lane); one entry per lane.
  std::unique_ptr<std::atomic<uint64_t>[]> lane_dirty_;
  uint32_t LaneOf(const BlockKey& key) const;
  void Enqueue(Shard& s, QItem item);
  void AddDirty(uint32_t lane, uint64_t bytes);
  void SubDirty(uint32_t lane, uint64_t bytes);
  std::atomic<uint64_t> ram_acks_{0}, ack_backpressure_{0};
  // PUTs parked in AwaitPut, so CancelPutWaits can wake exactly them. Lock
  // order: put_wait_mu_ before a completion's mutex.
  mutable std::mutex put_wait_mu_;
  std::unordered_multiset<PutCompletion*> put_waiters_;
  std::atomic<bool> put_waits_stopped_{false};
  std::atomic<uint64_t> put_wait_timeouts_{0}, put_waits_canceled_{0};
  std::atomic<uint64_t> post_ack_flush_failures_{0};
  std::atomic<uint64_t> flush_retries_{0};
  std::atomic<uint64_t> flush_no_room_{0};
  std::atomic<uint64_t> flush_no_room_dropped_{0};
  std::atomic<uint64_t> flush_refused_{0};
  // The stop sequence is waiting for the backlog (WaitForDrain): attempts
  // are no longer spaced out, so a failing object is given up at once
  // instead of holding the stop for its backoff.
  std::atomic<bool> draining_{false};
  // Drops are logged at most once per interval, with what the line stands for.
  std::atomic<uint64_t> drop_log_ms_{0};
  std::atomic<uint64_t> drop_logged_total_{0}, drop_logged_acked_{0};
  void NoteFlushOutcome(bool ok);
  void LogDrop(bool target_failed);
  std::atomic<uint64_t> shutdown_drain_timeouts_{0};
  std::atomic<bool> background_stopping_{false};
  LatencyHist ack_to_durable_latency_;
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
