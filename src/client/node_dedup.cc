#include "client/node_dedup.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "client/dedup_segment.h"
#include "client/dedup_token.h"
#include "utils/log.h"
#include "utils/env_parse.h"

namespace dfkv {

namespace {
constexpr uint32_t kMagic = 0x44445634u;  // "DDV4" (publisher ownership tokens)
constexpr uint32_t kStateEmpty = 0;
constexpr uint32_t kStateFetching = 1;
constexpr uint32_t kStateReady = 2;
constexpr uint32_t kStatePublishing = 3;
constexpr size_t kAnyLen = SIZE_MAX;  // CopyOut strict_n wildcard
// Slot::publisher while a publisher that can be fenced holds PUBLISHING.
constexpr uint32_t kFenceablePublisher = 0x46454e43u;

uint64_t EnvU64(const char* name, uint64_t dflt) {
  const char* v = std::getenv(name);
  if (!v || !*v) return dflt;
  unsigned long long x = std::strtoull(v, nullptr, 10);
  const uint64_t resolved = x > 0 ? static_cast<uint64_t>(x) : dflt;
  envparse::ReportUnlessPlain(name, 1, std::numeric_limits<uint64_t>::max(),
                              std::to_string(resolved));
  return resolved;
}

// Age of a stamp taken by another thread or process. `now` is often read
// before the lookup or lock that finds the stamp, so the stamp can be a
// little newer than now: within kStampSkewMs it is fresh (age 0), not
// ~2^64 ms old. A stamp further ahead is no skew (a torn write, a clock of
// another time namespace): it counts as ancient, so that an entry carrying
// it expires instead of being served for good.
constexpr uint64_t kStampSkewMs = 1000;
uint64_t AgeMs(uint64_t now, uint64_t then) {
  if (then <= now) return now - then;
  return then - now <= kStampSkewMs ? 0 : UINT64_MAX;
}

uint32_t Pow2AtLeast(uint32_t v) {
  uint32_t p = 1024;
  while (p < v && p < (1u << 24)) p <<= 1;
  return p;
}
}  // namespace

// 64-byte slot. gen is a seqlock over {identity, payload location}: odd while
// a writer mutates, +1 to even when coherent. state transitions are CAS-only
// (EMPTY->FETCHING, expired-READY->FETCHING, FETCHING->READY/EMPTY).
//
// publisher was padding in every release that shares this layout: no release
// reads it and none before this one writes it, so it is zero unless a
// publisher of this or a later release is inside Publish. Such a publisher
// commits with a compare-and-swap on gen, which is what lets an invalidation
// take the slot away from it (see ForceOutPublisher); a publisher of an older
// release finishes with unconditional stores and must be waited for.
struct NodeDedup::Slot {
  std::atomic<uint64_t> gen;
  std::atomic<uint32_t> state;
  uint32_t len;                        // payload byte count (attribute, not identity)
  uint64_t key_hi;
  uint64_t key_lo;
  uint32_t kind;                       // Kind: data vs exist namespace
  std::atomic<uint32_t> publisher;     // kFenceablePublisher or 0
  std::atomic<uint64_t> fetch_start_ms;
  uint64_t payload_off;                // absolute offset into the arena
  uint64_t alloc_seq;                  // arena cursor value at allocation (lap check)
  // exactly 64 B (cache line / cross-process ABI)
};

struct NodeDedup::Header {
  uint32_t magic;
  uint32_t nslots;
  uint64_t arena_bytes;
  std::atomic<uint64_t> alloc_cursor;  // monotonic bytes allocated (ring = mod arena)
  std::atomic<uint32_t> init_done;     // 1 once the creator finished laying out
  // pid of the creating process (dedup_segment.h). Padding in older
  // releases, which leave it 0.
  std::atomic<uint32_t> creator;
};

uint32_t NodeDedup::ProbeStart(const BlockKey& key, Kind kind) {
  return static_cast<uint32_t>(
      key.digest_hi ^ (key.digest_hi >> 32) ^ key.digest_lo ^
      (key.digest_lo >> 32) ^ (static_cast<uint32_t>(kind) << 13));
}

uint64_t NodeDedup::NowMs() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}

std::string NodeDedup::EnvSegmentName(uint64_t namespace_hash) {
  char name[128];
  // Layout version FIRST: a layout bump gets a fresh name and can never be
  // silently disabled by (or corrupt) a segment an older lib left behind —
  // old and new processes each rendezvous within their own generation during
  // a rolling upgrade, then the old segment ages out with its processes.
  // uid-scoped so unrelated users on one host cannot share a segment; the
  // canonical namespace digest separates independent keyspaces.
  std::snprintf(name, sizeof(name), "/dfkv-dedup-v4-%u-%016llx", ::getuid(),
                static_cast<unsigned long long>(namespace_hash));
  return name;
}

std::unique_ptr<NodeDedup> NodeDedup::FromEnv(uint64_t namespace_hash) {
  const char* on = std::getenv("DFKV_CLIENT_NODE_DEDUP");
  const bool enabled = on && std::strcmp(on, "1") == 0;
  envparse::ReportSwitch("DFKV_CLIENT_NODE_DEDUP", enabled, "only 1 is on");
  if (!enabled) return nullptr;
  Options o;
  o.arena_bytes = EnvU64("DFKV_NODE_DEDUP_ARENA_MB", 512) << 20;
  o.slots = static_cast<uint32_t>(EnvU64("DFKV_NODE_DEDUP_SLOTS", 65536));
  o.wait_ms = static_cast<int>(EnvU64("DFKV_NODE_DEDUP_WAIT_MS", 500));
  o.takeover_ms = static_cast<int>(EnvU64("DFKV_NODE_DEDUP_TAKEOVER_MS", 2000));
  o.ttl_ms = static_cast<int>(EnvU64("DFKV_NODE_DEDUP_TTL_MS", 5000));
  if (o.ttl_ms < o.wait_ms) {
    // A published result is claimable only for ttl_ms; a peer rank that
    // reaches it later (rank skew, a large leader batch) re-fetches it.
    DFKV_LOG_WARN("node-dedup: DFKV_NODE_DEDUP_TTL_MS=" + std::to_string(o.ttl_ms) +
                  " is below DFKV_NODE_DEDUP_WAIT_MS=" + std::to_string(o.wait_ms) +
                  "; late peers will re-fetch instead of sharing (default 5000)");
  }
  o.name = EnvSegmentName(namespace_hash);
  return Open(o);
}

std::unique_ptr<NodeDedup> NodeDedup::Open(const Options& opt) {
  static_assert(sizeof(Slot) == 64, "shm slot layout is cross-process ABI");
  const uint32_t nslots = Pow2AtLeast(opt.slots);
  const size_t len = SegmentBytes(opt);
  // One creator, which backs the whole segment eagerly (a full /dev/shm is
  // a clean failure here, not a SIGBUS mid-publish); see dedup_segment.h,
  // also for a creator that died before it finished.
  bool creator = false;
  void* base = dedup_segment::Attach(opt.name, SegmentLayout(len), "node-dedup",
                                     &creator);
  if (!base) return nullptr;

  auto d = std::unique_ptr<NodeDedup>(new NodeDedup());
  d->map_base_ = base;
  d->map_len_ = len;
  d->hdr_ = static_cast<Header*>(base);
  d->slots_ = reinterpret_cast<Slot*>(static_cast<char*>(base) + sizeof(Header));
  d->arena_ = reinterpret_cast<char*>(d->slots_) + static_cast<size_t>(nslots) * sizeof(Slot);
  d->arena_bytes_ = opt.arena_bytes;
  d->nslots_ = nslots;
  d->wait_ms_ = opt.wait_ms;
  d->takeover_ms_ = opt.takeover_ms;
  d->ttl_ms_ = opt.ttl_ms;

  if (creator) {
    // Stamp the header last so concurrent openers either spin briefly on
    // init_done or see a finished layout. The ring cursor is not written:
    // ftruncate zero-filled it, and a segment laid out a second time (one
    // whose creator died, or one an older release's opener laid out inside
    // this creator's window) may already be in use; a rewound cursor would
    // hand the bytes of published entries to new allocations while their
    // lap check still passes.
    d->hdr_->nslots = nslots;
    d->hdr_->arena_bytes = opt.arena_bytes;
    d->hdr_->magic = kMagic;
    d->hdr_->init_done.store(1, std::memory_order_release);
  } else {
    if (d->hdr_->magic != kMagic || d->hdr_->nslots != nslots ||
        d->hdr_->arena_bytes != opt.arena_bytes) {
      DFKV_LOG_INFO("node-dedup: segment header mismatch, running without");
      return nullptr;  // unique_ptr dtor munmaps
    }
  }
  DFKV_LOG_INFO("node-dedup: attached " + opt.name + " (slots=" +
                std::to_string(nslots) + ", arena=" +
                std::to_string(opt.arena_bytes >> 20) + " MiB)");
  return d;
}

dedup_segment::Layout NodeDedup::SegmentLayout(size_t len) {
  return dedup_segment::Layout{len, offsetof(Header, init_done),
                               offsetof(Header, creator)};
}

size_t NodeDedup::SegmentBytes(const Options& opt) {
  return sizeof(Header) +
         static_cast<size_t>(Pow2AtLeast(opt.slots)) * sizeof(Slot) +
         opt.arena_bytes;
}

NodeDedup::~NodeDedup() {
  if (map_base_) ::munmap(map_base_, map_len_);
  // The segment itself is intentionally left in place: peers may still use it,
  // and a later process re-attaches. Bounded size; no growth over time.
}

NodeDedup::Slot* NodeDedup::Find(const BlockKey& key, Kind kind) const {
  const uint32_t mask = nslots_ - 1;
  uint32_t idx = ProbeStart(key, kind) & mask;
  for (int probe = 0; probe < 8; ++probe, idx = (idx + 1) & mask) {
    Slot& s = slots_[idx];
    if (s.state.load(std::memory_order_acquire) == kStateEmpty) continue;
    if (s.key_hi == key.digest_hi && s.key_lo == key.digest_lo &&
        s.kind == static_cast<uint32_t>(kind))
      return &s;
  }
  return nullptr;
}

NodeDedup::Slot* NodeDedup::Reserve(const BlockKey& key, Kind kind) {
  const uint64_t now = NowMs();
  const uint32_t mask = nslots_ - 1;
  uint32_t idx = ProbeStart(key, kind) & mask;
  for (int probe = 0; probe < 8; ++probe, idx = (idx + 1) & mask) {
    Slot& s = slots_[idx];
    uint32_t st = s.state.load(std::memory_order_acquire);
    bool claim = false;
    bool locked = false;  // claimed a stale fetch: holds PUBLISHING (below)
    if (st == kStateEmpty) {
      claim = s.state.compare_exchange_strong(st, kStateFetching,
                                              std::memory_order_acq_rel);
    } else if (st == kStateReady &&
               (AgeMs(now, s.fetch_start_ms.load(std::memory_order_relaxed)) >
                    static_cast<uint64_t>(ttl_ms_) ||
                Lapped(s))) {
      // Expired, or overwritten in the ring: nobody can copy it any more,
      // and leaving it in place would only keep the key from being shared
      // until it expires (every later claim finds it first and fetches for
      // itself).
      claim = s.state.compare_exchange_strong(st, kStateFetching,
                                              std::memory_order_acq_rel);
    } else if (st == kStateFetching) {
      // A fetch older than the takeover age, of any key: its owner is dead or
      // hung (a claim of its own key would take it over the same way). Left
      // alone, such slots fill the probe window of keys nobody asks for
      // again. Taken like a takeover does, under PUBLISHING, so that no
      // takeover or late publish of the old claim acts on the slot while its
      // identity changes; the old owner's token no longer matches after.
      const uint64_t started = s.fetch_start_ms.load(std::memory_order_relaxed);
      if (started < now && now - started > static_cast<uint64_t>(takeover_ms_)) {
        claim = locked = s.state.compare_exchange_strong(
            st, kStatePublishing, std::memory_order_acq_rel);
      }
    }
    if (!claim) continue;
    // Slot reserved: mutate identity under an odd generation so a concurrent
    // Find/CopyOut on the OLD occupant tears predictably (gen mismatch).
    s.gen.fetch_add(1, std::memory_order_acq_rel);  // -> odd
    s.key_hi = key.digest_hi;
    s.key_lo = key.digest_lo;
    s.kind = static_cast<uint32_t>(kind);
    s.len = 0;
    s.payload_off = 0;
    s.alloc_seq = 0;
    s.fetch_start_ms.store(now, std::memory_order_relaxed);
    s.gen.fetch_add(1, std::memory_order_release);  // -> even, coherent
    if (locked) s.state.store(kStateFetching, std::memory_order_release);
    return &s;
  }
  return nullptr;  // probe window full: run without dedup for this key
}

// Read in READY, where payload_off, alloc_seq and len are not written (they
// change only after a claim has swapped the state away from READY).
bool NodeDedup::Lapped(const Slot& s) const {
  return hdr_->alloc_cursor.load(std::memory_order_acquire) - s.alloc_seq >
         arena_bytes_ - s.len;
}

bool NodeDedup::CopyOut(Slot* s, const BlockKey& key, Kind kind, size_t cap,
                        size_t strict_n, char* dst, size_t* got) const {
  const uint64_t g0 = s->gen.load(std::memory_order_acquire);
  if (g0 & 1) return false;  // writer active
  if (s->state.load(std::memory_order_acquire) != kStateReady) return false;
  // The caller looked this slot up by key before it got here, and the slot
  // may have been recycled for another key since. Identity is rewritten only
  // under an odd generation, so reading it between the two generation checks
  // ties the bytes copied below to the key they are returned for.
  if (s->key_hi != key.digest_hi || s->key_lo != key.digest_lo ||
      s->kind != static_cast<uint32_t>(kind))
    return false;
  const size_t n = s->len;
  if ((strict_n != kAnyLen && n != strict_n) || n > cap) {
    // Another process published this key with a length this reader does not
    // expect: the two disagree about what the key holds. Dedup rests on one
    // key meaning the same bytes in every process of the host (a key that
    // leaves out a rank axis of a sharded payload breaks it), so this is
    // reported. The entry is not used either way. The generation is read
    // again so that a slot being recycled under the reader is not reported.
    if (s->gen.load(std::memory_order_acquire) == g0)
      NoteLengthMismatch(&length_mismatches_, "node-dedup", n,
                         strict_n != kAnyLen ? strict_n : cap);
    return false;
  }
  const uint64_t off = s->payload_off;
  const uint64_t seq = s->alloc_seq;
  if (off + n > arena_bytes_) return false;  // torn metadata: bail
  // Lap check BEFORE the copy: if the ring cursor has advanced more than one
  // lap past this allocation, the payload bytes may already be overwritten.
  if (hdr_->alloc_cursor.load(std::memory_order_acquire) - seq > arena_bytes_ - n)
    return false;
  std::memcpy(dst, arena_ + off, n);
  // ... and AFTER: the copy is only valid if the payload stayed un-lapped and
  // the slot generation did not move during the copy.
  if (hdr_->alloc_cursor.load(std::memory_order_acquire) - seq > arena_bytes_ - n)
    return false;
  if (s->gen.load(std::memory_order_acquire) != g0) return false;
  if (got) *got = n;
  return true;
}

void NoteLengthMismatch(std::atomic<uint64_t>* counter, const char* tag,
                        size_t published, size_t expected) {
  counter->fetch_add(1, std::memory_order_relaxed);
  static std::atomic<bool> said{false};
  if (said.exchange(true, std::memory_order_relaxed)) return;
  DFKV_LOG_WARN(std::string(tag) + ": a key was published by another process "
                "on this host with " + std::to_string(published) +
                " bytes where this one expects " + std::to_string(expected) +
                ". Same-host dedup assumes that a key holds the same bytes in "
                "every process; if the keys leave out a rank axis of a "
                "sharded payload, turn dedup off. (Logged once; counted in "
                "dfkv_client_dedup_length_mismatches_total / "
                "dfkv_client_gpu_dedup_length_mismatches_total.)");
}

NodeDedup::Role NodeDedup::ClaimImpl(const BlockKey& key, Kind kind, size_t cap,
                                     size_t strict_n, char* dst, size_t* got,
                                     uint64_t* token) {
  if (token) *token = 0;
  if (cap == 0 || cap > arena_bytes_ / 2) return Role::kFetch;
  const uint64_t now = NowMs();
  if (Slot* s = Find(key, kind)) {
    const uint32_t st = s->state.load(std::memory_order_acquire);
    if (st == kStateReady) {
      if (AgeMs(now, s->fetch_start_ms.load(std::memory_order_relaxed)) <=
              static_cast<uint64_t>(ttl_ms_) &&
          CopyOut(s, key, kind, cap, strict_n, dst, got)) {
        hits_.fetch_add(1, std::memory_order_relaxed);
        return Role::kHit;
      }
      // Expired, torn or size-mismatched READY data falls through to Reserve;
      // the stale slot may be recycled but is never returned as a hit.
    } else if (st == kStateFetching) {
      // (`now` was read before the lookup: a fetch begun since is newer.)
      const uint64_t started = s->fetch_start_ms.load(std::memory_order_relaxed);
      if (started >= now ||
          now - started <= static_cast<uint64_t>(takeover_ms_))
        return Role::kWait;
      uint32_t expect = kStateFetching;
      if (s->state.compare_exchange_strong(expect, kStatePublishing,
                                           std::memory_order_acq_rel)) {
        s->gen.fetch_add(1, std::memory_order_acq_rel);
        s->fetch_start_ms.store(now, std::memory_order_relaxed);
        const uint64_t mine = s->gen.fetch_add(1, std::memory_order_release) + 1;
        s->state.store(kStateFetching, std::memory_order_release);
        if (token) *token = TokenFor(s, mine);
        fetches_.fetch_add(1, std::memory_order_relaxed);
        return Role::kFetch;
      }
      return Role::kWait;
    } else if (st == kStatePublishing) {
      // A publisher killed inside Publish never leaves this state. Once it
      // has been publishing for longer than the takeover age, take the slot
      // away from it (possible only for a publisher that can be fenced) and
      // fetch.
      // The mark is read before the stamp: PublishBegin writes the stamp
      // first, so a marked slot carries the time this publish began, not the
      // time of the fetch before it. (`now` was read before the lookup: a
      // publish begun since is newer.)
      if (s->publisher.load(std::memory_order_acquire) != kFenceablePublisher)
        return Role::kWait;
      const uint64_t started = s->fetch_start_ms.load(std::memory_order_relaxed);
      if (started >= now ||
          now - started <= static_cast<uint64_t>(takeover_ms_) ||
          !ForceOutPublisher(s))
        return Role::kWait;
    }
  }
  if (Slot* mine = Reserve(key, kind)) {
    // Concurrent lockstep claimers can reserve two slots for one key when the
    // second Find races the first identity write; this was observed live as
    // exactly twice as many server reads as unique keys. Re-scan immediately
    // and after a short settle spin, retaining only the lowest probe position
    // because that is the canonical slot every waiter finds.
    for (int pass = 0; pass < 2; ++pass) {
      Slot* first = Find(key, kind);
      if (first && first != mine) {
        uint32_t st = kStateFetching;
        mine->state.compare_exchange_strong(st, kStateEmpty,
                                            std::memory_order_acq_rel);
        return Role::kWait;
      }
      if (pass == 0)
        for (volatile int spin = 0; spin < 2000; ++spin) {}
    }
    if (token)
      *token = TokenFor(mine, mine->gen.load(std::memory_order_acquire));
    fetches_.fetch_add(1, std::memory_order_relaxed);
    return Role::kFetch;
  }
  return Role::kFetch;
}

NodeDedup::Role NodeDedup::Claim(const BlockKey& key, size_t n, char* dst,
                                 uint64_t* token) {
  return ClaimImpl(key, Kind::kData, n, n, dst, nullptr, token);
}

NodeDedup::Role NodeDedup::ClaimAuto(const BlockKey& key, size_t cap, char* dst,
                                     size_t* got, uint64_t* token) {
  return ClaimImpl(key, Kind::kData, cap, kAnyLen, dst, got, token);
}

NodeDedup::Role NodeDedup::ClaimExist(const BlockKey& key, bool* val,
                                      uint64_t* token) {
  char b = 0;
  const Role r = ClaimImpl(key, Kind::kExist, 1, 1, &b, nullptr, token);
  if (r == Role::kHit && val) *val = (b != 0);
  return r;
}

uint64_t NodeDedup::TokenFor(const Slot* s, uint64_t gen) const {
  return dedup_token::Make(static_cast<uint32_t>(s - slots_), gen);
}

NodeDedup::Slot* NodeDedup::OwnedSlot(uint64_t token) const {
  if (!dedup_token::Valid(token)) return nullptr;
  const uint32_t idx = dedup_token::SlotIndex(token);
  if (idx >= nslots_) return nullptr;
  Slot* s = &slots_[idx];
  if (!dedup_token::SameGen(token, s->gen.load(std::memory_order_acquire)))
    return nullptr;
  return s;
}

bool NodeDedup::LockOwned(Slot* s, const BlockKey& key, Kind kind,
                          uint64_t token) {
  // The slot can be PUBLISHING with this claim's generation only while
  // someone else's stale Publish or Abort holds it for the comparison below,
  // which hands it back within a few instructions. Giving up at once would
  // leave this fetch unpublished and the slot FETCHING until a takeover, so
  // wait for it, bounded in case that holder was killed right there.
  const auto give_up =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(lock_wait_ms_);
  for (;;) {
    uint32_t st = kStateFetching;
    if (s->state.compare_exchange_strong(st, kStatePublishing,
                                         std::memory_order_acq_rel))
      break;
    if (st != kStatePublishing ||
        !dedup_token::SameGen(token, s->gen.load(std::memory_order_acquire)) ||
        std::chrono::steady_clock::now() >= give_up)
      return false;
    std::this_thread::yield();
  }
  // The generation was last compared before the lock was taken. An
  // invalidation followed by a new claim of this slot in between leaves it
  // FETCHING again, for a later fetch of this key or for another key, and
  // the lock is then on that claim. Nothing moves the generation while the
  // slot is PUBLISHING, so this comparison settles it.
  if (!dedup_token::SameGen(token, s->gen.load(std::memory_order_acquire)) ||
      s->key_hi != key.digest_hi || s->key_lo != key.digest_lo ||
      s->kind != static_cast<uint32_t>(kind)) {
    s->state.store(kStateFetching, std::memory_order_release);
    return false;
  }
  return true;
}

bool NodeDedup::ForceOutBegin(Slot* s) {
  // Generation first: every way out of PUBLISHING moves it, so if it still
  // has this value at the swap below, the state and the mark read in between
  // belong to one and the same publish.
  uint64_t g = s->gen.load(std::memory_order_acquire);
  if (g & 1) return false;  // a publisher or another forcer is mid-write
  if (s->state.load(std::memory_order_acquire) != kStatePublishing)
    return false;
  if (s->publisher.load(std::memory_order_acquire) != kFenceablePublisher)
    return false;
  // The publisher commits by swapping gen from g to g + 1, and so does this.
  // Exactly one swap succeeds. If it is this one, the publisher, alive or
  // not, can no longer write the slot, and the odd generation keeps every
  // other forcer and every reader out until ForceOutFinish is through.
  return s->gen.compare_exchange_strong(g, g + 1, std::memory_order_acq_rel);
}

void NodeDedup::ForceOutFinish(Slot* s) {
  s->publisher.store(0, std::memory_order_release);
  s->gen.fetch_add(1, std::memory_order_release);  // even: token superseded
  s->state.store(kStateEmpty, std::memory_order_release);
}

bool NodeDedup::ForceOutPublisher(Slot* s) {
  if (!ForceOutBegin(s)) return false;
  ForceOutFinish(s);
  return true;
}

bool NodeDedup::PublishBegin(Slot* s, const BlockKey& key, Kind kind,
                             uint64_t token, PublishHold* hold) {
  if (!LockOwned(s, key, kind, token)) return false;
  hold->slot = s;
  hold->gen = s->gen.load(std::memory_order_acquire);
  // The age of a PUBLISHING slot is the time spent publishing, not the time
  // the fetch took before: a claim gives the copy takeover_ms from here.
  s->fetch_start_ms.store(NowMs(), std::memory_order_relaxed);
  // From here an invalidation may take the slot at any moment.
  s->publisher.store(kFenceablePublisher, std::memory_order_release);
  return true;
}

bool NodeDedup::LockSlotForTest(Slot* s) {
  uint32_t st = kStateFetching;
  return s->state.compare_exchange_strong(st, kStatePublishing,
                                          std::memory_order_acq_rel);
}

void NodeDedup::UnlockSlotForTest(Slot* s) {
  s->state.store(kStateFetching, std::memory_order_release);
}

bool NodeDedup::ShiftStampForTest(const BlockKey& key, Kind kind,
                                  int64_t delta_ms) {
  Slot* s = Find(key, kind);
  if (!s) return false;
  s->fetch_start_ms.fetch_add(static_cast<uint64_t>(delta_ms),
                              std::memory_order_relaxed);
  return true;
}

void NodeDedup::SetFenceableForTest(const PublishHold& hold, bool on) {
  hold.slot->publisher.store(on ? kFenceablePublisher : 0,
                             std::memory_order_release);
}

void NodeDedup::PublishFinish(const PublishHold& hold, const char* data,
                              size_t n) {
  Slot* s = hold.slot;
  // Ring-allocate: reserve [cur, cur+n) contiguously; skip the tail remainder
  // when a lap boundary would split the payload.
  uint64_t cur = hdr_->alloc_cursor.load(std::memory_order_relaxed);
  uint64_t off, seq;
  for (;;) {
    const uint64_t in_lap = cur % arena_bytes_;
    const uint64_t need =
        (in_lap + n > arena_bytes_) ? (arena_bytes_ - in_lap) + n : n;
    if (hdr_->alloc_cursor.compare_exchange_weak(cur, cur + need,
                                                 std::memory_order_acq_rel)) {
      seq = cur + need;
      off = (cur + need - n) % arena_bytes_;
      break;
    }
  }
  std::memcpy(arena_ + off, data, n);
  // Commit. A failed swap means an invalidation took the slot during the
  // copy: it is no longer this publisher's, so nothing more is written.
  uint64_t g = hold.gen;
  if (!s->gen.compare_exchange_strong(g, hold.gen + 1,
                                      std::memory_order_acq_rel))
    return;
  s->len = static_cast<uint32_t>(n);
  s->payload_off = off;
  s->alloc_seq = seq;
  s->fetch_start_ms.store(NowMs(), std::memory_order_relaxed);
  s->publisher.store(0, std::memory_order_release);
  s->gen.fetch_add(1, std::memory_order_release);
  s->state.store(kStateReady, std::memory_order_release);
}

void NodeDedup::Publish(const BlockKey& key, Kind kind, uint64_t token,
                        const char* data, size_t n) {
  PublishHold hold;
  Slot* s = OwnedSlot(token);
  if (s && PublishBegin(s, key, kind, token, &hold))
    PublishFinish(hold, data, n);
}

void NodeDedup::AbortSlot(Slot* s, const BlockKey& key, Kind kind,
                          uint64_t token) {
  if (!LockOwned(s, key, kind, token)) return;
  s->state.store(kStateEmpty, std::memory_order_release);
}

void NodeDedup::Abort(const BlockKey& key, Kind kind, uint64_t token) {
  if (Slot* s = OwnedSlot(token)) AbortSlot(s, key, kind, token);
}

void NodeDedup::InvalidateKind(const BlockKey& key, Kind kind) {
  // A key can occupy more than one slot of its probe window: a claimer that
  // cannot use the published entry (expired, or another length) reserves the
  // first free position, which may lie before the entry it leaves behind,
  // and two lockstep claimers can each keep a slot. Lookups stop at the
  // first match, so an entry left behind comes back into view as soon as the
  // one in front of it goes. Every entry of the key is removed here.
  const uint32_t mask = nslots_ - 1;
  uint32_t idx = ProbeStart(key, kind) & mask;
  for (int probe = 0; probe < 8; ++probe, idx = (idx + 1) & mask) {
    Slot& s = slots_[idx];
    for (;;) {
      uint32_t st = s.state.load(std::memory_order_acquire);
      if (st == kStateEmpty) break;
      if (s.key_hi != key.digest_hi || s.key_lo != key.digest_lo ||
          s.kind != static_cast<uint32_t>(kind))
        break;
      if (st == kStatePublishing) {
        // What is being published is from before this invalidation. A
        // publisher that can be fenced loses the slot here and now, whether
        // it is still running or was killed; any other holder is waited for.
        if (ForceOutPublisher(&s)) break;
        std::this_thread::yield();
        continue;
      }
      if (!s.state.compare_exchange_weak(st, kStatePublishing,
                                         std::memory_order_acq_rel))
        continue;
      // Supersede every outstanding ownership token before making the slot
      // reusable. A late publisher can find a newly claimed slot for this
      // key, but its old token can never publish into that generation.
      s.gen.fetch_add(1, std::memory_order_acq_rel);
      s.gen.fetch_add(1, std::memory_order_release);
      s.state.store(kStateEmpty, std::memory_order_release);
      break;
    }
  }
}

void NodeDedup::Invalidate(const BlockKey& key) {
  InvalidateKind(key, Kind::kData);
  InvalidateKind(key, Kind::kExist);
}

bool NodeDedup::WaitImpl(const BlockKey& key, Kind kind, size_t cap,
                         size_t strict_n, char* dst, size_t* got,
                         uint64_t deadline_ms) {
  const uint64_t deadline = deadline_ms ? deadline_ms : WaitDeadline();
  int backoff_us = 50;
  bool gone = false;
  for (;;) {
    Slot* s = Find(key, kind);
    const uint32_t st =
        s ? s->state.load(std::memory_order_acquire) : kStateEmpty;
    if (st == kStateReady) {
      // Same freshness rule as a claim. What a waiter finds is normally the
      // result just published for it, but when that fetch was aborted it can
      // be an older entry of the key further along the probe window.
      if (AgeMs(NowMs(), s->fetch_start_ms.load(std::memory_order_relaxed)) <=
              static_cast<uint64_t>(ttl_ms_) &&
          CopyOut(s, key, kind, cap, strict_n, dst, got)) {
        wait_hits_.fetch_add(1, std::memory_order_relaxed);
        return true;
      }
      break;  // expired, lapped, recycled or another length: it will not heal
    }
    if (st == kStateFetching || st == kStatePublishing) {
      gone = false;
      if (NowMs() >= deadline) break;
      std::this_thread::sleep_for(std::chrono::microseconds(backoff_us));
      if (backoff_us < 1000) backoff_us *= 2;
      continue;
    }
    // No fetch in flight for this key: the owner aborted (or the entry was
    // invalidated), so nothing will be published and waiting out the deadline
    // only delays the caller's own fetch. A live slot never passes through
    // EMPTY, but two lockstep claimers may hold one slot each for an instant
    // until the higher probe position is released; a probe that crosses that
    // release sees neither, so look once more before giving up.
    if (gone) break;
    gone = true;
  }
  wait_timeouts_.fetch_add(1, std::memory_order_relaxed);
  return false;
}

uint64_t NodeDedup::WaitDeadline() const {
  return NowMs() + static_cast<uint64_t>(wait_ms_);
}

bool NodeDedup::WaitCopy(const BlockKey& key, size_t n, char* dst,
                         uint64_t deadline_ms) {
  return WaitImpl(key, Kind::kData, n, n, dst, nullptr, deadline_ms);
}

bool NodeDedup::WaitCopyAuto(const BlockKey& key, size_t cap, char* dst,
                             size_t* got, uint64_t deadline_ms) {
  return WaitImpl(key, Kind::kData, cap, kAnyLen, dst, got, deadline_ms);
}

bool NodeDedup::WaitExist(const BlockKey& key, bool* val,
                          uint64_t deadline_ms) {
  char b = 0;
  if (!WaitImpl(key, Kind::kExist, 1, 1, &b, nullptr, deadline_ms))
    return false;
  if (val) *val = (b != 0);
  return true;
}

}  // namespace dfkv
