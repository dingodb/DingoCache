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

#include "utils/log.h"

namespace dfkv {

namespace {
constexpr uint32_t kMagic = 0x44445633u;  // "DDV3" (128-bit identity)
constexpr uint32_t kStateEmpty = 0;
constexpr uint32_t kStateFetching = 1;
constexpr uint32_t kStateReady = 2;
constexpr size_t kAnyLen = SIZE_MAX;  // CopyOut strict_n wildcard

uint64_t EnvU64(const char* name, uint64_t dflt) {
  const char* v = std::getenv(name);
  if (!v || !*v) return dflt;
  unsigned long long x = std::strtoull(v, nullptr, 10);
  return x > 0 ? static_cast<uint64_t>(x) : dflt;
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
struct NodeDedup::Slot {
  std::atomic<uint64_t> gen;
  std::atomic<uint32_t> state;
  uint32_t len;                        // payload byte count (attribute, not identity)
  uint64_t key_hi;
  uint64_t key_lo;
  uint32_t kind;                       // Kind: data vs exist namespace
  uint32_t pad;
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
  uint32_t pad;
};

uint64_t NodeDedup::NowMs() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}

std::string NodeDedup::EnvSegmentName(uint64_t model_hash) {
  char name[128];
  // Layout version FIRST: a layout bump gets a fresh name and can never be
  // silently disabled by (or corrupt) a segment an older lib left behind —
  // old and new processes each rendezvous within their own generation during
  // a rolling upgrade, then the old segment ages out with its processes.
  // uid-scoped so unrelated users on one host can't share (or collide on) a
  // segment; model_hash separates keyspaces (same salting as the block keys).
  std::snprintf(name, sizeof(name), "/dfkv-dedup-v3-%u-%016llx", ::getuid(),
                static_cast<unsigned long long>(model_hash));
  return name;
}

std::unique_ptr<NodeDedup> NodeDedup::FromEnv(uint64_t model_hash) {
  const char* on = std::getenv("DFKV_CLIENT_NODE_DEDUP");
  if (!on || std::strcmp(on, "1") != 0) return nullptr;
  Options o;
  o.arena_bytes = EnvU64("DFKV_NODE_DEDUP_ARENA_MB", 512) << 20;
  o.slots = static_cast<uint32_t>(EnvU64("DFKV_NODE_DEDUP_SLOTS", 65536));
  o.wait_ms = static_cast<int>(EnvU64("DFKV_NODE_DEDUP_WAIT_MS", 500));
  o.takeover_ms = static_cast<int>(EnvU64("DFKV_NODE_DEDUP_TAKEOVER_MS", 2000));
  o.ttl_ms = static_cast<int>(EnvU64("DFKV_NODE_DEDUP_TTL_MS", 5000));
  o.name = EnvSegmentName(model_hash);
  return Open(o);
}

std::unique_ptr<NodeDedup> NodeDedup::Open(const Options& opt) {
  static_assert(sizeof(Slot) == 64, "shm slot layout is cross-process ABI");
  const uint32_t nslots = Pow2AtLeast(opt.slots);
  const size_t len = sizeof(Header) + static_cast<size_t>(nslots) * sizeof(Slot) +
                     opt.arena_bytes;
  int fd = ::shm_open(opt.name.c_str(), O_RDWR | O_CREAT, 0600);
  if (fd < 0) return nullptr;
  struct stat st{};
  if (::fstat(fd, &st) != 0) { ::close(fd); return nullptr; }
  const bool creator = (st.st_size == 0);
  if (creator && ::ftruncate(fd, static_cast<off_t>(len)) != 0) {
    ::close(fd);
    return nullptr;
  }
  if (creator) {
    // Eagerly back the whole segment with pages. tmpfs allocates lazily on
    // first write: with /dev/shm full (other tenants), ftruncate+mmap succeed
    // and the process later takes a SIGBUS on a page fault mid-publish —
    // a crash, not a degrade. posix_fallocate converts that into a clean
    // ENOSPC HERE, where returning nullptr just runs without dedup. Cost:
    // the segment occupies its full size from creation (bounded, documented).
    const int frc = ::posix_fallocate(fd, 0, static_cast<off_t>(len));
    if (frc != 0) {
      DFKV_LOG_INFO("node-dedup: cannot pre-reserve " +
                    std::to_string(len >> 20) + " MiB in /dev/shm (" +
                    std::to_string(frc) + "), running without");
      ::close(fd);
      ::shm_unlink(opt.name.c_str());  // don't leave a half-backed segment
      return nullptr;
    }
  }
  if (!creator && static_cast<size_t>(st.st_size) != len) {
    DFKV_LOG_INFO("node-dedup: existing segment layout mismatch, running without");
    ::close(fd);
    return nullptr;
  }
  void* base = ::mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ::close(fd);
  if (base == MAP_FAILED) return nullptr;

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
    // Fresh (ftruncate zero-fills): stamp the header last so concurrent
    // openers either spin briefly on init_done or see a finished layout.
    d->hdr_->nslots = nslots;
    d->hdr_->arena_bytes = opt.arena_bytes;
    d->hdr_->alloc_cursor.store(0, std::memory_order_relaxed);
    d->hdr_->magic = kMagic;
    d->hdr_->init_done.store(1, std::memory_order_release);
  } else {
    for (int spin = 0; spin < 1000 && d->hdr_->init_done.load(std::memory_order_acquire) != 1; ++spin)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
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

NodeDedup::~NodeDedup() {
  if (map_base_) ::munmap(map_base_, map_len_);
  // The segment itself is intentionally left in place: peers may still use it,
  // and a later process re-attaches. Bounded size; no growth over time.
}

NodeDedup::Slot* NodeDedup::Find(const BlockKey& key, Kind kind) const {
  const uint32_t mask = nslots_ - 1;
  uint32_t idx = static_cast<uint32_t>(
      key.digest_hi ^ (key.digest_hi >> 32) ^ key.digest_lo ^
      (key.digest_lo >> 32) ^ (static_cast<uint32_t>(kind) << 13)) & mask;
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
  uint32_t idx = static_cast<uint32_t>(
      key.digest_hi ^ (key.digest_hi >> 32) ^ key.digest_lo ^
      (key.digest_lo >> 32) ^ (static_cast<uint32_t>(kind) << 13)) & mask;
  for (int probe = 0; probe < 8; ++probe, idx = (idx + 1) & mask) {
    Slot& s = slots_[idx];
    uint32_t st = s.state.load(std::memory_order_acquire);
    bool claim = false;
    if (st == kStateEmpty) {
      claim = s.state.compare_exchange_strong(st, kStateFetching,
                                              std::memory_order_acq_rel);
    } else if (st == kStateReady &&
               now - s.fetch_start_ms.load(std::memory_order_relaxed) >
                   static_cast<uint64_t>(ttl_ms_)) {
      claim = s.state.compare_exchange_strong(st, kStateFetching,
                                              std::memory_order_acq_rel);
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
    return &s;
  }
  return nullptr;  // probe window full: run without dedup for this key
}

bool NodeDedup::CopyOut(Slot* s, size_t cap, size_t strict_n, char* dst,
                        size_t* got) const {
  const uint64_t g0 = s->gen.load(std::memory_order_acquire);
  if (g0 & 1) return false;  // writer active
  if (s->state.load(std::memory_order_acquire) != kStateReady) return false;
  const size_t n = s->len;
  if (strict_n != kAnyLen && n != strict_n) return false;
  if (n > cap) return false;
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

NodeDedup::Role NodeDedup::ClaimImpl(const BlockKey& key, Kind kind, size_t cap,
                                     size_t strict_n, char* dst, size_t* got) {
  if (cap == 0 || cap > arena_bytes_ / 2) return Role::kFetch;  // oversize: no dedup
  const uint64_t now = NowMs();
  if (Slot* s = Find(key, kind)) {
    const uint32_t st = s->state.load(std::memory_order_acquire);
    if (st == kStateReady) {
      if (now - s->fetch_start_ms.load(std::memory_order_relaxed) <=
              static_cast<uint64_t>(ttl_ms_) &&
          CopyOut(s, cap, strict_n, dst, got)) {
        hits_.fetch_add(1, std::memory_order_relaxed);
        return Role::kHit;
      }
      // Expired/torn/size-mismatch: fall through to Reserve (may recycle it).
    } else if (st == kStateFetching) {
      const uint64_t started = s->fetch_start_ms.load(std::memory_order_relaxed);
      if (now - started <= static_cast<uint64_t>(takeover_ms_)) return Role::kWait;
      // Fetcher looks dead: take the fetch over (CAS on fetch_start_ms so
      // exactly one waiter wins; the losers keep waiting on the winner).
      uint64_t expect = started;
      if (s->fetch_start_ms.compare_exchange_strong(expect, now,
                                                    std::memory_order_acq_rel)) {
        fetches_.fetch_add(1, std::memory_order_relaxed);
        return Role::kFetch;
      }
      return Role::kWait;
    }
  }
  if (Slot* mine = Reserve(key, kind)) {
    // Concurrent claimers can reserve TWO slots for one key when the second
    // claimer's Find() races the first's identity write (near-synchronous
    // claim loops hit that window almost every key; found live on the GPU
    // flavor as exactly-2x server reads). Re-scan twice — immediately and
    // after a ~1 us settle spin — and keep only the LOWEST probe position
    // (what every waiter's Find returns); other winners release and wait.
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
    fetches_.fetch_add(1, std::memory_order_relaxed);
    return Role::kFetch;
  }
  return Role::kFetch;  // no slot available: plain fetch, nothing to publish
}

NodeDedup::Role NodeDedup::Claim(const BlockKey& key, size_t n, char* dst) {
  return ClaimImpl(key, Kind::kData, n, n, dst, nullptr);
}

NodeDedup::Role NodeDedup::ClaimAuto(const BlockKey& key, size_t cap, char* dst,
                                     size_t* got) {
  return ClaimImpl(key, Kind::kData, cap, kAnyLen, dst, got);
}

NodeDedup::Role NodeDedup::ClaimExist(const BlockKey& key, bool* val) {
  char b = 0;
  const Role r = ClaimImpl(key, Kind::kExist, 1, 1, &b, nullptr);
  if (r == Role::kHit && val) *val = (b != 0);
  return r;
}

void NodeDedup::Publish(const BlockKey& key, Kind kind, const char* data, size_t n) {
  Slot* s = Find(key, kind);
  if (!s || s->state.load(std::memory_order_acquire) != kStateFetching) return;
  // Ring-allocate: reserve [cur, cur+n) contiguously; skip the tail remainder
  // when a lap boundary would split the payload.
  uint64_t cur = hdr_->alloc_cursor.load(std::memory_order_relaxed);
  uint64_t off, seq;
  for (;;) {
    const uint64_t in_lap = cur % arena_bytes_;
    const uint64_t need = (in_lap + n > arena_bytes_) ? (arena_bytes_ - in_lap) + n : n;
    if (hdr_->alloc_cursor.compare_exchange_weak(cur, cur + need,
                                                 std::memory_order_acq_rel)) {
      seq = cur + need;                       // cursor AFTER this allocation
      off = (cur + need - n) % arena_bytes_;  // payload sits at the end of the grab
      break;
    }
  }
  std::memcpy(arena_ + off, data, n);
  s->gen.fetch_add(1, std::memory_order_acq_rel);   // odd: mutating
  s->len = static_cast<uint32_t>(n);
  s->payload_off = off;
  s->alloc_seq = seq;
  s->fetch_start_ms.store(NowMs(), std::memory_order_relaxed);  // TTL from publish
  s->gen.fetch_add(1, std::memory_order_release);   // even: coherent
  s->state.store(kStateReady, std::memory_order_release);
}

void NodeDedup::Abort(const BlockKey& key, Kind kind) {
  Slot* s = Find(key, kind);
  if (!s) return;
  // Only the fetcher aborts. Freeing the slot immediately (instead of letting
  // it age into a takeover) lets a falling-back waiter re-claim at once; CAS
  // so a concurrent takeover winner isn't clobbered.
  uint32_t st = kStateFetching;
  s->state.compare_exchange_strong(st, kStateEmpty, std::memory_order_acq_rel);
}

bool NodeDedup::WaitImpl(const BlockKey& key, Kind kind, size_t cap,
                         size_t strict_n, char* dst, size_t* got) {
  const uint64_t deadline = NowMs() + static_cast<uint64_t>(wait_ms_);
  int backoff_us = 50;
  for (;;) {
    Slot* s = Find(key, kind);
    if (s && s->state.load(std::memory_order_acquire) == kStateReady &&
        CopyOut(s, cap, strict_n, dst, got)) {
      wait_hits_.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
    if (NowMs() >= deadline) {
      wait_timeouts_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(backoff_us));
    if (backoff_us < 1000) backoff_us *= 2;
  }
}

bool NodeDedup::WaitCopy(const BlockKey& key, size_t n, char* dst) {
  return WaitImpl(key, Kind::kData, n, n, dst, nullptr);
}

bool NodeDedup::WaitCopyAuto(const BlockKey& key, size_t cap, char* dst, size_t* got) {
  return WaitImpl(key, Kind::kData, cap, kAnyLen, dst, got);
}

bool NodeDedup::WaitExist(const BlockKey& key, bool* val) {
  char b = 0;
  if (!WaitImpl(key, Kind::kExist, 1, 1, &b, nullptr)) return false;
  if (val) *val = (b != 0);
  return true;
}

}  // namespace dfkv
