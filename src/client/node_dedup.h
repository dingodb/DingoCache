/* NodeDedup — same-host GET/EXIST rendezvous over POSIX shared memory.
 *
 * Motivation (phase 5/6): TP-replicated KV (MLA/DSA latent) is written by one
 * rank but read back by EVERY rank — 8x fabric traffic and 8x server read
 * load per L3 prefix load (measured on GLM-5.2 TP8). The ranks are same-host
 * processes requesting the SAME key set within milliseconds (measured 12 ms
 * spread vs ~44 ms per fetch batch), so a plain look-aside cache misses: by
 * the time rank N checks, rank 0's fetch is still in flight. What dedups the
 * fetch is a RENDEZVOUS: the first arrival claims a key (FETCHING), fetches
 * it remotely and publishes; the others wait briefly and copy — with a hard
 * deadline that falls back to a direct fetch, so no failure mode is worse
 * than today's 8x behavior. EXIST probes (2 per page per rank) ride the same
 * machinery with a 1-byte payload; a stale exist answer is safe either way
 * (stale absent -> recompute, stale present -> GET miss -> recompute).
 *
 * Correctness model (lock-free, crash-robust):
 *  - Index slot identity is (key, kind); the payload length is an attribute
 *    (strict readers require len == n, auto readers accept len <= cap). A
 *    reader re-checks identity inside its generation-validated copy: the
 *    slot it looked up may have been recycled for another key meanwhile.
 *  - State machine EMPTY -> FETCHING -> READY guarded by CAS; a FETCHING
 *    older than takeover_ms (fetcher died/hung) can be re-claimed.
 *  - A waiter waits only while a fetch is in flight (FETCHING/PUBLISHING).
 *    An owner that has nothing to publish aborts its slot back to EMPTY, and
 *    the waiter then returns at once instead of running out its deadline.
 *  - Payload arena is a ring with a MONOTONIC allocation cursor; readers
 *    validate "my payload's lap has not been overwritten" (cursor - alloc_seq
 *    <= arena - len) before AND after the copy, plus a per-slot seqlock
 *    generation for slot reuse. A failed validation is just a miss.
 *  - Nothing here is load-bearing: every exit path (no slot, timeout, torn
 *    read, size mismatch) degrades to the caller's normal remote op.
 *
 * SCOPE: host-memory destinations (the SGLang HiCache path). GPUDirect
 * destinations are ROUTED AWAY from this class by the batch wrappers (a
 * memcpy to a device VA is invalid) and served by the CUDA-IPC flavor in
 * node_dedup_gpu.h (phase 2b, DFKV_CLIENT_NODE_DEDUP_GPU=1 opts in on top).
 * Off by default (DFKV_CLIENT_NODE_DEDUP=1 opts in). */
#ifndef DFKV_NODE_DEDUP_H_
#define DFKV_NODE_DEDUP_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "client/dedup_segment.h"
#include "client/dedup_token.h"
#include "common/kv_types.h"

namespace dfkv {

class NodeDedup {
 public:
  struct Options {
    std::string name;              // shm name (leading '/'); see FromEnv()
    uint64_t arena_bytes = 512ull << 20;   // DFKV_NODE_DEDUP_ARENA_MB
    uint32_t slots = 65536;                // DFKV_NODE_DEDUP_SLOTS (pow2 forced)
    int wait_ms = 500;                     // DFKV_NODE_DEDUP_WAIT_MS
    int takeover_ms = 2000;                // DFKV_NODE_DEDUP_TAKEOVER_MS
    int ttl_ms = 5000;                     // DFKV_NODE_DEDUP_TTL_MS
  };

  // Entry namespaces sharing one index: a key's data payload and its exist
  // answer are distinct entries.
  enum class Kind : uint32_t { kData = 1, kExist = 2 };

  // Builds Options from the environment; returns nullptr (feature off) unless
  // DFKV_CLIENT_NODE_DEDUP=1. namespace_hash isolates shared-memory segments.
  static std::unique_ptr<NodeDedup> FromEnv(uint64_t namespace_hash);
  // The env-derived segment name. The shm LAYOUT VERSION is part of the name:
  // a layout change must never collide with a segment an older lib left
  // behind — v1.23.0 shipped a layout bump behind the same name, the header
  // check "safely" refused the stale segment, and the feature silently
  // disabled itself across the whole upgrade (canary re-measured 8x). The
  // header magic stays as a backstop only. Exposed so tests clean up the
  // exact segment FromEnv would use.
  static std::string EnvSegmentName(uint64_t namespace_hash);
  // Opens (or creates) the shm segment. Returns nullptr on any failure — the
  // caller just runs without dedup. A parameter mismatch with an existing
  // segment also returns nullptr (never reinterpret another config's layout).
  static std::unique_ptr<NodeDedup> Open(const Options& opt);
  // Size of the segment Open maps for opt, and where its header keeps the
  // fields dedup_segment needs (tests leave a creation unfinished with it).
  static size_t SegmentBytes(const Options& opt);
  static dedup_segment::Layout SegmentLayout(size_t len);
  ~NodeDedup();

  NodeDedup(const NodeDedup&) = delete;
  NodeDedup& operator=(const NodeDedup&) = delete;

  // Per-key verdict for a batch pass. A kFetch claim also returns an opaque
  // ownership token. The caller MUST pass that token to Publish/Abort; it
  // fences a late fetcher after Remove invalidates or a takeover supersedes it.
  enum class Role {
    kHit,    // result delivered (skip the remote op)
    kFetch,  // caller owns the remote op; MUST call Publish or Abort afterwards
    kWait,   // someone else is fetching; call the matching Wait* afterwards
  };

  // Strict GET (BatchGet semantics: exact n bytes). A published payload of a
  // different length is not a hit (lockstep peers request the same n).
  Role Claim(const BlockKey& key, size_t n, char* dst, uint64_t* token);
  // Wait* return true once the owner's result has been copied out, false
  // when the caller must fetch for itself: the owner aborted, the published
  // entry is unusable, or the deadline passed. deadline_ms is a WaitDeadline()
  // value shared by every key of one batch, so a batch waits wait_ms in
  // total, not per key; 0 gives the call a deadline of its own. A key reached
  // after the deadline still gets one look at its slot.
  uint64_t WaitDeadline() const;
  bool WaitCopy(const BlockKey& key, size_t n, char* dst,
                uint64_t deadline_ms = 0);

  // Variable-size GET (BatchGetAuto semantics): hit iff published len <= cap;
  // *got receives the payload length.
  Role ClaimAuto(const BlockKey& key, size_t cap, char* dst, size_t* got,
                 uint64_t* token);
  bool WaitCopyAuto(const BlockKey& key, size_t cap, char* dst, size_t* got,
                    uint64_t deadline_ms = 0);

  // EXIST probe (1-byte answer). Only an answer the server gave may be
  // published by the caller: present, or absent in either form a transport
  // reports it (kNotFound, or kOk with exists=false).
  Role ClaimExist(const BlockKey& key, bool* val, uint64_t* token);
  bool WaitExist(const BlockKey& key, bool* val, uint64_t deadline_ms = 0);

  // Fetch outcomes for keys claimed as kFetch (kind selects the namespace;
  // exist publishes a 1-byte payload). token=0 is an unclaimed/direct fetch.
  // Both act only on the slot the token was issued for, and only while that
  // slot is still the same fetch of the same key.
  void Publish(const BlockKey& key, Kind kind, uint64_t token,
               const char* data, size_t n);
  void Abort(const BlockKey& key, Kind kind, uint64_t token);
  // Eviction barrier: removes every READY data/exist answer of the key and
  // supersedes every FETCHING owner (a key can hold more than one slot of its
  // probe window). A publisher already copying loses its slot and commits
  // nothing; a publisher of a release that cannot be fenced is joined.
  void Invalidate(const BlockKey& key);

  // diagnostics (relaxed; process-local)
  uint64_t hits() const { return hits_.load(std::memory_order_relaxed); }
  uint64_t fetches() const { return fetches_.load(std::memory_order_relaxed); }
  uint64_t wait_hits() const { return wait_hits_.load(std::memory_order_relaxed); }
  uint64_t wait_timeouts() const { return wait_timeouts_.load(std::memory_order_relaxed); }

 private:
  struct Slot;    // 64-byte shm index slot
  struct Header;  // shm header
  friend struct NodeDedupTestAccess;  // steps a reader one stage at a time
  NodeDedup() = default;

  Slot* Find(const BlockKey& key, Kind kind) const;
  Slot* Reserve(const BlockKey& key, Kind kind);
  // A READY entry whose payload the ring has overwritten since.
  bool Lapped(const Slot& s) const;
  // Validated copy of (key, kind)'s payload out of slot s. strict_n: exact
  // length required (SIZE_MAX = any <= cap). Returns payload length via *got
  // on success.
  bool CopyOut(Slot* s, const BlockKey& key, Kind kind, size_t cap,
               size_t strict_n, char* dst, size_t* got) const;
  Role ClaimImpl(const BlockKey& key, Kind kind, size_t cap, size_t strict_n,
                 char* dst, size_t* got, uint64_t* token);
  bool WaitImpl(const BlockKey& key, Kind kind, size_t cap, size_t strict_n,
                char* dst, size_t* got, uint64_t deadline_ms);
  void InvalidateKind(const BlockKey& key, Kind kind);
  static uint64_t NowMs();
  // First slot of the key's 8-slot probe window, before masking.
  static uint32_t ProbeStart(const BlockKey& key, Kind kind);
  // Ownership tokens (client/dedup_token.h) name a slot and its generation.
  uint64_t TokenFor(const Slot* s, uint64_t gen) const;
  // The slot the token names, if it still has the token's generation.
  Slot* OwnedSlot(uint64_t token) const;
  // FETCHING -> PUBLISHING for the token's holder. False, with the slot left
  // as it was, when the slot is not (or no longer) this holder's fetch of
  // (key, kind).
  bool LockOwned(Slot* s, const BlockKey& key, Kind kind, uint64_t token);
  // Publish and Abort after the token check: Publish takes the slot
  // (PublishBegin), then copies and commits (PublishFinish).
  struct PublishHold {
    Slot* slot = nullptr;
    uint64_t gen = 0;
  };
  bool PublishBegin(Slot* s, const BlockKey& key, Kind kind, uint64_t token,
                    PublishHold* hold);
  void PublishFinish(const PublishHold& hold, const char* data, size_t n);
  void AbortSlot(Slot* s, const BlockKey& key, Kind kind, uint64_t token);
  // Makes a held slot look like the publisher of an older release, which
  // does not mark itself as fenceable.
  void SetFenceableForTest(const PublishHold& hold, bool on);
  // A stale Publish/Abort of another process inside its re-check: the slot
  // is PUBLISHING for a moment and then FETCHING again.
  bool LockSlotForTest(Slot* s);
  void UnlockSlotForTest(Slot* s);
  // Takes a PUBLISHING slot away from a publisher that can be fenced and
  // leaves it EMPTY. False when the holder is not such a publisher, or when
  // another forcer is at work on the slot. Begin wins the slot (generation
  // odd from then on), Finish clears and releases it.
  bool ForceOutPublisher(Slot* s);
  bool ForceOutBegin(Slot* s);
  void ForceOutFinish(Slot* s);

  Header* hdr_ = nullptr;
  Slot* slots_ = nullptr;
  char* arena_ = nullptr;
  uint64_t arena_bytes_ = 0;
  uint32_t nslots_ = 0;       // power of two
  int wait_ms_ = 500;
  int takeover_ms_ = 2000;
  int ttl_ms_ = 5000;
  int lock_wait_ms_ = kDedupLockWaitMs;  // see LockOwned; tests raise it
  void* map_base_ = nullptr;
  size_t map_len_ = 0;

  std::atomic<uint64_t> hits_{0}, fetches_{0}, wait_hits_{0}, wait_timeouts_{0};
};

}  // namespace dfkv

#endif  // DFKV_NODE_DEDUP_H_
