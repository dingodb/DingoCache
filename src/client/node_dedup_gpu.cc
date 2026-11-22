#include "client/node_dedup_gpu.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <random>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "client/dedup_segment.h"
#include "client/dedup_token.h"
#include "utils/log.h"
#include "utils/env_parse.h"

namespace dfkv {

namespace {
constexpr uint32_t kMagic = 0x44444733u;  // "DDG3" (publisher ownership tokens)
constexpr uint32_t kStateEmpty = 0;
constexpr uint32_t kStateFetching = 1;
constexpr uint32_t kStateReady = 2;
constexpr uint32_t kStatePublishing = 3;
// Slot::alloc_seq while a publisher that can be fenced holds PUBLISHING (see
// MarkedFenceable). An arena cursor never reaches the top bit, so no release
// writes this value there for any other reason, and an older reader that
// meets it in a torn snapshot fails its lap check.
constexpr uint64_t kFenceablePublisher = 0x8000000046454e43ull;
// Slot::alloc_seq while a claim or an invalidation is taking a slot from such
// a publisher (ForceOutBegin until ForceOutFinish; see
// RecoverStrandedForceOut): the two top bits set and the odd generation the
// force-out holds. It names that one generation, so once anyone has moved the
// generation on, the slot no longer reads as that force-out. The top bit has
// the same reasons as above; the second one keeps it apart from
// kFenceablePublisher (a generation would need 2^62 steps to wrap the tag).
constexpr uint64_t ForcingOutTag(uint64_t odd_gen) {
  return 0xC000000000000000ull | (odd_gen & 0x3FFFFFFFFFFFFFFFull);
}

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

// First slot of a key's 8-slot probe window, before masking.
uint32_t ProbeStart(const BlockKey& key) {
  return static_cast<uint32_t>(key.digest_hi ^ (key.digest_hi >> 32) ^
                               key.digest_lo ^ (key.digest_lo >> 32));
}

bool PidDead(uint32_t pid) {
  return ::kill(static_cast<pid_t>(pid), 0) != 0 && errno == ESRCH;
}

// Start time of a process (field 22 of /proc/<pid>/stat, clock ticks since
// boot), 0 if it cannot be read. Unchanged by exec; differs between two
// processes that had the same pid one after the other.
uint64_t StartTicks(uint32_t pid) {
  char path[64];
  std::snprintf(path, sizeof(path), "/proc/%u/stat", pid);
  FILE* f = std::fopen(path, "r");
  if (!f) return 0;
  char buf[1024];
  const size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
  std::fclose(f);
  buf[n] = '\0';
  const char* p = std::strrchr(buf, ')');  // the command name may hold anything
  if (!p) return 0;
  ++p;
  for (int field = 2; field < 21 && p; ++field) {  // to the start of field 22
    p = std::strchr(p + 1, ' ');
  }
  return p ? std::strtoull(p + 1, nullptr, 10) : 0;
}

// Random, fixed for the life of this process image (a new one after exec).
uint32_t ImageNonce() {
  static const uint32_t nonce = [] {
    std::random_device rd;
    uint32_t v = 0;
    while (v == 0) v = rd();
    return v;
  }();
  return nonce;
}
}  // namespace

// Same 64-byte seqlock slot as the host flavor, but the payload location is a
// DESCRIPTOR into a peer's GPU arena: (owner registry index, owner generation,
// arena offset, arena cursor at allocation).
struct GpuNodeDedup::Slot {
  std::atomic<uint64_t> gen;
  std::atomic<uint32_t> state;
  uint32_t len;
  uint64_t key_hi;
  uint64_t key_lo;
  std::atomic<uint64_t> fetch_start_ms;
  uint64_t payload_off;   // offset into the OWNER's GPU arena
  // Owner's arena cursor value at allocation (lap check), valid in READY.
  // kFenceablePublisher while a publisher that can be fenced holds the slot.
  uint64_t alloc_seq;
  // Registry entry of the publishing process and its truncated generation.
  // Valid in READY; kMaxProcs otherwise.
  uint32_t owner_idx;
  uint32_t owner_gen32;
};

// One per participating process. pid claims the entry (CAS, dead pids are
// reclaimable); generation counts claims so descriptors and cached IPC
// mappings against a previous occupant are detectable. The arena allocation
// cursor lives HERE (per arena), not in the header.
struct GpuNodeDedup::ProcEntry {
  std::atomic<uint32_t> pid;         // 0 = free
  std::atomic<uint32_t> ready;       // 1 once handle/device/arena_bytes valid
  std::atomic<uint64_t> generation;  // bumped on every claim
  std::atomic<uint64_t> alloc_cursor;
  uint64_t arena_bytes;
  int32_t device;
  // Which process holds the entry, beyond its pid: the claimant's image
  // nonce (ImageNonce) and start time (StartTicks), valid when identity_gen
  // equals generation. They were padding in older releases, which neither
  // write nor read them; an entry an older release claimed therefore shows
  // an identity_gen of a previous claim, and is judged by its pid alone.
  uint32_t image;
  CUipcMemHandle handle;
  // The exporter's own arena VA: cuIpcOpenMemHandle refuses handles from the
  // EXPORTING process, so a second instance in the same process (same address
  // space) reads through this pointer directly. Only trusted for an entry of
  // this very process image (pid, identity and image nonce all match).
  uint64_t owner_va;
  uint64_t start_ticks;
  uint64_t identity_gen;  // -> 128 B (cross-process ABI)
};

struct GpuNodeDedup::Header {
  uint32_t magic;
  uint32_t nslots;
  uint64_t arena_bytes;  // configured per-process arena size (must match)
  std::atomic<uint32_t> init_done;
  // pid of the creating process (dedup_segment.h). Padding in older
  // releases, which leave it 0.
  std::atomic<uint32_t> creator;
};

uint64_t GpuNodeDedup::NowMs() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}

std::string GpuNodeDedup::EnvSegmentName(uint64_t namespace_hash) {
  char name[128];
  // Layout version FIRST in the name (v1.23.0 lesson, see node_dedup.h):
  // a layout bump must never meet a segment an older lib left behind.
  std::snprintf(name, sizeof(name), "/dfkv-dedup-gpuv3-%u-%016llx", ::getuid(),
                static_cast<unsigned long long>(namespace_hash));
  return name;
}

std::unique_ptr<GpuNodeDedup> GpuNodeDedup::FromEnv(
    uint64_t namespace_hash, const void* device_dst_hint) {
  const char* on = std::getenv("DFKV_CLIENT_NODE_DEDUP");
  if (!on || std::strcmp(on, "1") != 0) return nullptr;  // reported by NodeDedup::FromEnv, which KVClient calls first
  const char* gpu = std::getenv("DFKV_CLIENT_NODE_DEDUP_GPU");
  const bool enabled = gpu && std::strcmp(gpu, "1") == 0;
  envparse::ReportSwitch("DFKV_CLIENT_NODE_DEDUP_GPU", enabled, "only 1 is on");
  if (!enabled) return nullptr;
  // The caller remembers this answer, so it is not taken from a throttled probe.
  const CudaLib* cu = CudaLib::Probe();
  if (!cu) {
    DFKV_LOG_INFO(
        "gpu-dedup: CUDA is not initialized in this process, running without");
    return nullptr;
  }
  if (!cu->HasCurrentCtx()) {
    // The connectors' transfer threads are fresh pthreads that never touched
    // CUDA: bind the primary context of the device the caller is about to
    // scatter into (already alive — the framework holds it).
    if (!device_dst_hint || !cu->BindPrimaryCtx(cu->DeviceOf(device_dst_hint))) {
      DFKV_LOG_INFO("gpu-dedup: no CUDA context and no bindable device, running without");
      return nullptr;
    }
  }
  Options o;
  o.arena_bytes = EnvU64("DFKV_NODE_DEDUP_GPU_ARENA_MB", 512) << 20;
  o.slots = static_cast<uint32_t>(EnvU64("DFKV_NODE_DEDUP_GPU_SLOTS", 16384));
  o.wait_ms = static_cast<int>(EnvU64("DFKV_NODE_DEDUP_WAIT_MS", 500));
  o.takeover_ms = static_cast<int>(EnvU64("DFKV_NODE_DEDUP_TAKEOVER_MS", 2000));
  o.ttl_ms = static_cast<int>(EnvU64("DFKV_NODE_DEDUP_TTL_MS", 5000));
  if (o.ttl_ms < o.wait_ms) {
    // A published result is claimable only for ttl_ms; a peer rank that
    // reaches it later (rank skew, a large leader batch) re-fetches it.
    DFKV_LOG_WARN("gpu-dedup: DFKV_NODE_DEDUP_TTL_MS=" + std::to_string(o.ttl_ms) +
                  " is below DFKV_NODE_DEDUP_WAIT_MS=" + std::to_string(o.wait_ms) +
                  "; late peers will re-fetch instead of sharing (default 5000)");
  }
  o.name = EnvSegmentName(namespace_hash);
  return Open(o);
}

std::unique_ptr<GpuNodeDedup> GpuNodeDedup::Open(const Options& opt) {
  static_assert(sizeof(Slot) == 64, "shm slot layout is cross-process ABI");
  static_assert(sizeof(ProcEntry) == 128, "shm registry layout is cross-process ABI");
  const CudaLib* cu = CudaLib::Probe();
  if (!cu || !cu->HasCurrentCtx()) return nullptr;

  const uint32_t nslots = Pow2AtLeast(opt.slots);
  const size_t len = SegmentBytes(opt);
  // Small segment (~1-2 MiB: descriptors only, payloads live in GPU memory),
  // created and backed as the host flavor's is (dedup_segment.h).
  bool creator = false;
  void* base = dedup_segment::Attach(opt.name, SegmentLayout(len), "gpu-dedup",
                                     &creator);
  if (!base) return nullptr;

  auto d = std::unique_ptr<GpuNodeDedup>(new GpuNodeDedup());
  d->cu_ = cu;
  d->map_base_ = base;
  d->map_len_ = len;
  d->hdr_ = static_cast<Header*>(base);
  d->reg_ = reinterpret_cast<ProcEntry*>(static_cast<char*>(base) + sizeof(Header));
  d->slots_ = reinterpret_cast<Slot*>(reinterpret_cast<char*>(d->reg_) +
                                      sizeof(ProcEntry) * kMaxProcs);
  d->nslots_ = nslots;
  d->arena_bytes_ = opt.arena_bytes;
  d->wait_ms_ = opt.wait_ms;
  d->takeover_ms_ = opt.takeover_ms;
  d->ttl_ms_ = opt.ttl_ms;

  if (creator) {
    d->hdr_->nslots = nslots;
    d->hdr_->arena_bytes = opt.arena_bytes;
    d->hdr_->magic = kMagic;
    d->hdr_->init_done.store(1, std::memory_order_release);
  } else {
    if (d->hdr_->magic != kMagic || d->hdr_->nslots != nslots ||
        d->hdr_->arena_bytes != opt.arena_bytes) {
      DFKV_LOG_INFO("gpu-dedup: segment header mismatch, running without");
      return nullptr;
    }
  }

  // Allocate this process' staging arena on the caller's device and export it.
  if (cu->MemAlloc(&d->arena_base_, opt.arena_bytes) != kCudaSuccess) {
    DFKV_LOG_INFO("gpu-dedup: cannot allocate " +
                  std::to_string(opt.arena_bytes >> 20) +
                  " MiB GPU staging arena, running without");
    return nullptr;
  }
  CUipcMemHandle handle{};
  if (cu->IpcGetMemHandle(&handle, d->arena_base_) != kCudaSuccess) {
    DFKV_LOG_INFO("gpu-dedup: cuIpcGetMemHandle failed, running without");
    return nullptr;  // dtor frees the arena
  }

  // Claim a registry entry: a free one, one whose owner died, or one whose
  // owner is provably gone although its pid is in use (see Stale).
  const uint32_t mypid = static_cast<uint32_t>(::getpid());
  for (uint32_t i = 0; i < kMaxProcs && d->self_idx_ == kMaxProcs; ++i) {
    ProcEntry& e = d->reg_[i];
    uint32_t cur = e.pid.load(std::memory_order_acquire);
    if (cur == 0 || (cur != mypid && PidDead(cur))) {
      if (!e.pid.compare_exchange_strong(cur, mypid, std::memory_order_acq_rel))
        continue;
      e.ready.store(0, std::memory_order_release);
      d->self_gen_ = e.generation.fetch_add(1, std::memory_order_acq_rel) + 1;
    } else {
      // The pid is in use. Several claimants may judge the entry stale at
      // once (instances of this process, for its own pid), so the claim is
      // a swap of the generation the judgement was made at; then the pid,
      // in case the owner died meanwhile and an older release took the
      // entry by its pid.
      uint64_t g = e.generation.load(std::memory_order_acquire);
      if (!Stale(e, cur, g) ||
          !e.generation.compare_exchange_strong(g, g + 1,
                                                std::memory_order_acq_rel) ||
          !e.pid.compare_exchange_strong(cur, mypid, std::memory_order_acq_rel))
        continue;
      e.ready.store(0, std::memory_order_release);
      d->self_gen_ = g + 1;
    }
    e.alloc_cursor.store(0, std::memory_order_relaxed);
    e.arena_bytes = opt.arena_bytes;
    e.device = cu->CurrentDevice();
    d->self_device_ = e.device;
    std::memcpy(&e.handle, &handle, sizeof(handle));
    e.owner_va = static_cast<uint64_t>(d->arena_base_);
    e.image = ImageNonce();
    e.start_ticks = StartTicks(mypid);
    // Last, and a release store: whoever sees identity_gen equal to the
    // generation sees the image and start time written for it (Stale).
    __atomic_store_n(&e.identity_gen, d->self_gen_, __ATOMIC_RELEASE);
    e.ready.store(1, std::memory_order_release);
    d->self_idx_ = i;
  }
  if (d->self_idx_ == kMaxProcs) {
    DFKV_LOG_INFO("gpu-dedup: registry full (" + std::to_string(kMaxProcs) +
                  " live processes), running without");
    return nullptr;
  }
  DFKV_LOG_INFO("gpu-dedup: attached " + opt.name + " (slots=" +
                std::to_string(nslots) + ", arena=" +
                std::to_string(opt.arena_bytes >> 20) + " MiB on device " +
                std::to_string(d->reg_[d->self_idx_].device) + ", entry " +
                std::to_string(d->self_idx_) + ")");
  return d;
}

size_t GpuNodeDedup::SegmentBytes(const Options& opt) {
  return sizeof(Header) + sizeof(ProcEntry) * kMaxProcs +
         static_cast<size_t>(Pow2AtLeast(opt.slots)) * sizeof(Slot);
}

dedup_segment::Layout GpuNodeDedup::SegmentLayout(size_t len) {
  return dedup_segment::Layout{len, offsetof(Header, init_done),
                               offsetof(Header, creator)};
}

// The holder of an entry whose pid cur is in use is gone when the entry's
// identity (written by a claimant of this or a later release, for generation
// g) names another process: for this process's own pid another image nonce
// (a previous process with the same pid, typically a container restarted in
// place, or this process before an exec); for another pid another start time
// (the pid was reused). An entry without an identity for g, written by an
// older release, is judged by its pid alone, as before.
bool GpuNodeDedup::Stale(const ProcEntry& e, uint32_t cur, uint64_t g) {
  // Acquire: pairs with the claimant's release store of identity_gen, after
  // its image and start time (otherwise a sibling's fresh entry could be
  // judged by the fields of its previous holder).
  if (__atomic_load_n(&e.identity_gen, __ATOMIC_ACQUIRE) != g) return false;
  if (cur == static_cast<uint32_t>(::getpid())) return e.image != ImageNonce();
  const uint64_t t = StartTicks(cur);
  return t != 0 && t != e.start_ticks;
}

void GpuNodeDedup::ForgeRegistryForTest(int how) {
  const uint32_t me = static_cast<uint32_t>(::getpid());
  const uint32_t parent = static_cast<uint32_t>(::getppid());
  for (uint32_t i = 0; i < kMaxProcs; ++i) {
    ProcEntry& e = reg_[i];
    if (how == 4 && i != self_idx_) continue;
    const uint64_t g = e.generation.load(std::memory_order_acquire);
    switch (how) {
      case 0:  // a previous process with this pid
      case 4:  // ... in this entry only
        e.pid.store(me);
        e.image = ImageNonce() + 1;
        e.identity_gen = g;
        break;
      case 1:  // an older release's process with this pid
        e.pid.store(me);
        e.identity_gen = g + 1;
        break;
      case 2:  // a process whose pid now belongs to another (live) one
        e.pid.store(parent);
        e.start_ticks = StartTicks(parent) + 1;
        e.identity_gen = g;
        break;
      case 3:  // a live process
        e.pid.store(parent);
        e.start_ticks = StartTicks(parent);
        e.identity_gen = g;
        break;
    }
    e.ready.store(1, std::memory_order_release);
  }
}

GpuNodeDedup::~GpuNodeDedup() {
  if (self_idx_ < kMaxProcs && reg_) {
    ProcEntry& e = reg_[self_idx_];
    // Invalidate before the arena goes away: readers check generation before
    // AND after the copy, so a mid-copy free degrades to a miss (the driver
    // rejects the unmapped VA range; it does not corrupt).
    e.ready.store(0, std::memory_order_release);
    e.generation.fetch_add(1, std::memory_order_acq_rel);
    e.pid.store(0, std::memory_order_release);
  }
  if (cu_) {
    for (auto& pm : peer_map_)
      if (pm.base) cu_->IpcCloseMemHandle(pm.base);  // best-effort at teardown
    if (stream_) cu_->StreamDestroy(stream_);
    if (arena_base_) cu_->MemFree(arena_base_);
  }
  if (map_base_) ::munmap(map_base_, map_len_);
}

void GpuNodeDedup::EnsureThreadCtx() const {
  // First bind on a thread retains the primary ctx once (bounded by thread
  // count); afterwards HasCurrentCtx short-circuits.
  if (!cu_->HasCurrentCtx()) cu_->BindPrimaryCtx(self_device_);
}

CUstream GpuNodeDedup::Stream() {
  std::lock_guard<std::mutex> lk(stream_mu_);
  if (!stream_ &&
      cu_->StreamCreate(&stream_, kCuStreamNonBlocking) != kCudaSuccess)
    stream_ = nullptr;
  return stream_;
}

GpuNodeDedup::Slot* GpuNodeDedup::Find(const BlockKey& key) const {
  const uint32_t mask = nslots_ - 1;
  uint32_t idx = ProbeStart(key) & mask;
  for (int probe = 0; probe < 8; ++probe, idx = (idx + 1) & mask) {
    Slot& s = slots_[idx];
    if (s.state.load(std::memory_order_acquire) == kStateEmpty) continue;
    if (s.key_hi == key.digest_hi && s.key_lo == key.digest_lo) return &s;
  }
  return nullptr;
}

GpuNodeDedup::Slot* GpuNodeDedup::Reserve(const BlockKey& key) {
  const uint64_t now = NowMs();
  const uint32_t mask = nslots_ - 1;
  uint32_t idx = ProbeStart(key) & mask;
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
                Unusable(s))) {
      // Expired, or nobody can copy it any more (see Unusable): left in
      // place it would only keep the key from being shared until it expires.
      claim = s.state.compare_exchange_strong(st, kStateFetching,
                                              std::memory_order_acq_rel);
    } else if (st == kStateFetching) {
      // A fetch older than the takeover age, of any key: see NodeDedup::
      // Reserve. Taken under PUBLISHING, as a takeover is.
      const uint64_t started = s.fetch_start_ms.load(std::memory_order_relaxed);
      if (started < now && now - started > static_cast<uint64_t>(takeover_ms_)) {
        claim = locked = s.state.compare_exchange_strong(
            st, kStatePublishing, std::memory_order_acq_rel);
      }
    }
    if (!claim) continue;
    s.gen.fetch_add(1, std::memory_order_acq_rel);  // odd: mutating identity
    s.key_hi = key.digest_hi;
    s.key_lo = key.digest_lo;
    s.len = 0;
    s.payload_off = 0;
    __atomic_store_n(&s.alloc_seq, 0, __ATOMIC_RELAXED);
    s.owner_idx = kMaxProcs;
    s.owner_gen32 = 0;
    s.fetch_start_ms.store(now, std::memory_order_relaxed);
    s.gen.fetch_add(1, std::memory_order_release);  // even: coherent
    if (locked) s.state.store(kStateFetching, std::memory_order_release);
    return &s;
  }
  return nullptr;
}

// A READY entry no reader can copy any more: its owner's registry entry is
// gone or has another occupant, or the owner's arena ring has overwritten the
// payload. Read in READY, where the descriptor is not written.
bool GpuNodeDedup::Unusable(const Slot& s) const {
  const uint32_t oi = s.owner_idx;
  if (oi >= kMaxProcs) return true;
  const ProcEntry& e = reg_[oi];
  if (e.ready.load(std::memory_order_acquire) != 1 ||
      static_cast<uint32_t>(e.generation.load(std::memory_order_acquire)) !=
          s.owner_gen32)
    return true;
  return e.alloc_cursor.load(std::memory_order_acquire) -
             __atomic_load_n(&s.alloc_seq, __ATOMIC_RELAXED) >
         e.arena_bytes - s.len;
}

CUdeviceptr GpuNodeDedup::PeerBase(uint32_t idx, uint64_t gen) {
  if (idx == self_idx_)
    return gen == self_gen_ ? arena_base_ : 0;  // own publishes: no IPC open
  {
    // Another instance in THIS process: cuIpcOpenMemHandle refuses handles
    // the process itself exported, but its arena VA is directly valid here.
    ProcEntry& e = reg_[idx];
    if (e.pid.load(std::memory_order_acquire) ==
        static_cast<uint32_t>(::getpid())) {
      // Same pid is not same process: an entry left by a previous process
      // with this pid, or by this process before an exec, carries an
      // address of an address space that is gone.
      if (e.ready.load(std::memory_order_acquire) != 1 ||
          e.generation.load(std::memory_order_acquire) != gen ||
          __atomic_load_n(&e.identity_gen, __ATOMIC_ACQUIRE) != gen ||
          e.image != ImageNonce())
        return 0;
      return static_cast<CUdeviceptr>(e.owner_va);
    }
  }
  std::lock_guard<std::mutex> lk(peer_mu_);
  PeerMap& pm = peer_map_[idx];
  if (pm.base && pm.gen == gen) return pm.base;
  if (pm.base) {  // stale mapping of a previous occupant
    cu_->IpcCloseMemHandle(pm.base);
    pm.base = 0;
  }
  ProcEntry& e = reg_[idx];
  CUipcMemHandle h;
  std::memcpy(&h, &e.handle, sizeof(h));
  // Re-check the generation AFTER snapshotting the handle: a concurrent
  // re-claim can't have half-written the handle we just validated.
  if (e.ready.load(std::memory_order_acquire) != 1 ||
      e.generation.load(std::memory_order_acquire) != gen)
    return 0;
  CUdeviceptr base = 0;
  if (cu_->IpcOpenMemHandle(&base, h, kCuIpcMemLazyEnablePeerAccess) != kCudaSuccess)
    return 0;
  pm.gen = gen;
  pm.base = base;
  return base;
}

bool GpuNodeDedup::IssueCopySg(Slot* s, const BlockKey& key, const Seg* segs,
                               size_t nsegs, size_t total_cap, CUstream st,
                               CopySnap* snap) {
  const uint64_t g0 = s->gen.load(std::memory_order_acquire);
  if (g0 & 1) return false;
  if (s->state.load(std::memory_order_acquire) != kStateReady) return false;
  // The slot was looked up by key earlier and may have been recycled for
  // another key since. Identity is rewritten only under an odd generation,
  // and ValidateCopy re-reads the generation after the copy.
  if (s->key_hi != key.digest_hi || s->key_lo != key.digest_lo) return false;
  const size_t n = s->len;
  if (n == 0 || n > total_cap) return false;
  const uint32_t oi = s->owner_idx;
  const uint32_t og = s->owner_gen32;
  const uint64_t off = s->payload_off;
  const uint64_t seq = __atomic_load_n(&s->alloc_seq, __ATOMIC_RELAXED);
  if (oi >= kMaxProcs) return false;
  ProcEntry& e = reg_[oi];
  const uint64_t egen = e.generation.load(std::memory_order_acquire);
  if (e.ready.load(std::memory_order_acquire) != 1 ||
      static_cast<uint32_t>(egen) != og)
    return false;
  if (off + n > e.arena_bytes) return false;  // torn metadata: bail
  // Lap check BEFORE the copy (owner's cursor lives in its registry entry).
  if (e.alloc_cursor.load(std::memory_order_acquire) - seq > e.arena_bytes - n)
    return false;
  CUdeviceptr base = PeerBase(oi, egen);
  if (!base) return false;
  size_t done = 0;
  for (size_t j = 0; j < nsegs && done < n; ++j) {
    const size_t m = std::min(n - done, segs[j].cap);
    if (m == 0) continue;
    if (cu_->MemcpyAsync(reinterpret_cast<CUdeviceptr>(segs[j].ptr),
                         base + off + done, m, st) != kCudaSuccess)
      return false;  // unmapped/raced VA: the driver rejects, we miss
    done += m;
  }
  if (done != n) return false;
  *snap = CopySnap{s, g0, oi, egen, seq, n};
  return true;
}

bool GpuNodeDedup::ValidateCopy(const CopySnap& snap) const {
  // AFTER the sync: un-lapped payload, same entry generation (arena still the
  // one we copied from), untouched slot.
  ProcEntry& e = reg_[snap.oi];
  if (e.alloc_cursor.load(std::memory_order_acquire) - snap.seq >
      e.arena_bytes - snap.n)
    return false;
  if (e.generation.load(std::memory_order_acquire) != snap.egen) return false;
  if (snap.s->gen.load(std::memory_order_acquire) != snap.g0) return false;
  return true;
}

bool GpuNodeDedup::CopyOutSg(Slot* s, const BlockKey& key, const Seg* segs,
                             size_t nsegs, size_t total_cap, size_t* got) {
  CUstream st = Stream();
  if (!st) return false;
  CopySnap snap{};
  if (!IssueCopySg(s, key, segs, nsegs, total_cap, st, &snap)) return false;
  // The scatter must be ON the device before we declare a hit — async D2D
  // "returns" immediately; the framework would otherwise consume the blocks
  // before the bytes land.
  if (cu_->StreamSynchronize(st) != kCudaSuccess) return false;
  if (!ValidateCopy(snap)) return false;
  if (got) *got = snap.n;
  return true;
}

void GpuNodeDedup::WaitManySg(WaitItem* items, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    items[i].ok = false;
    items[i].got = 0;
  }
  if (n == 0) return;
  EnsureThreadCtx();
  CUstream st = Stream();
  if (!st) {
    wait_timeouts_.fetch_add(n, std::memory_order_relaxed);
    return;
  }
  std::vector<size_t> pending(n);
  for (size_t i = 0; i < n; ++i) pending[i] = i;
  struct InFlight {
    size_t idx;
    CopySnap snap;
  };
  std::vector<InFlight> inflight;
  inflight.reserve(n);
  // Per key: its slot was not there on the previous look (see below).
  std::vector<char> gone(n, 0);
  const uint64_t deadline = NowMs() + static_cast<uint64_t>(wait_ms_);
  int backoff_us = 50;
  size_t failed_final = 0;
  while (!pending.empty()) {
    // Issue round: enqueue every key that is READY right now, keep the keys
    // whose fetch is still in flight, and give up on the rest.
    size_t w = 0;
    bool look_again = false;
    for (size_t r = 0; r < pending.size(); ++r) {
      const size_t idx = pending[r];
      WaitItem& it = items[idx];
      Slot* s = Find(it.key);
      const uint32_t state =
          s ? s->state.load(std::memory_order_acquire) : kStateEmpty;
      if (state == kStateReady) {
        CopySnap snap{};
        // Same freshness rule as a claim: with the fetch in front of it
        // aborted, a waiter can find an older entry of the key.
        if (AgeMs(NowMs(), s->fetch_start_ms.load(std::memory_order_relaxed)) <=
                static_cast<uint64_t>(ttl_ms_) &&
            IssueCopySg(s, it.key, it.segs, it.nsegs, it.total_cap, st,
                        &snap)) {
          inflight.push_back(InFlight{idx, snap});
        } else {
          ++failed_final;  // expired, lapped, recycled, too long: will not heal
        }
      } else if (state == kStateFetching || state == kStatePublishing) {
        gone[idx] = 0;
        pending[w++] = idx;  // not published yet
      } else if (gone[idx]) {
        // No fetch in flight for this key: the owner aborted, or the entry
        // was invalidated. Nothing will be published, and waiting out the
        // deadline only delays the caller's own fetch.
        ++failed_final;
      } else {
        // A live slot never passes through EMPTY, but two lockstep claimers
        // may hold one slot each for an instant until the higher probe
        // position is released; a look that crosses that release sees
        // neither. Look once more, at once, before giving up.
        gone[idx] = 1;
        look_again = true;
        pending[w++] = idx;
      }
    }
    pending.resize(w);
    if (!inflight.empty()) {
      // ONE sync amortized over the whole round, then post-validation.
      const bool synced = cu_->StreamSynchronize(st) == kCudaSuccess;
      for (const InFlight& f : inflight) {
        if (synced && ValidateCopy(f.snap)) {
          items[f.idx].ok = true;
          items[f.idx].got = f.snap.n;
          wait_hits_.fetch_add(1, std::memory_order_relaxed);
        } else {
          ++failed_final;  // lapped mid-copy: it will not heal — re-fetch
        }
      }
      inflight.clear();
      backoff_us = 50;  // publishes are flowing; poll eagerly
      continue;
    }
    if (look_again) continue;
    if (pending.empty() || NowMs() >= deadline) break;
    std::this_thread::sleep_for(std::chrono::microseconds(backoff_us));
    if (backoff_us < 1000) backoff_us *= 2;
  }
  wait_timeouts_.fetch_add(pending.size() + failed_final,
                           std::memory_order_relaxed);
}

GpuNodeDedup::Role GpuNodeDedup::ClaimSg(const BlockKey& key, const Seg* segs,
                                         size_t nsegs, size_t total_cap,
                                         size_t* got, uint64_t* token) {
  if (token) *token = 0;
  if (total_cap == 0 || total_cap > arena_bytes_ / 2) return Role::kFetch;
  EnsureThreadCtx();
  const uint64_t now = NowMs();
  if (Slot* s = Find(key)) {
    const uint32_t st = s->state.load(std::memory_order_acquire);
    if (st == kStateReady) {
      if (AgeMs(now, s->fetch_start_ms.load(std::memory_order_relaxed)) <=
              static_cast<uint64_t>(ttl_ms_) &&
          CopyOutSg(s, key, segs, nsegs, total_cap, got)) {
        hits_.fetch_add(1, std::memory_order_relaxed);
        return Role::kHit;
      }
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
      // A publisher killed inside PublishSg never leaves this state. Once it
      // has been publishing for longer than the takeover age, take the slot
      // away from it (possible only for a publisher that can be fenced) and
      // fetch. The generation is read first and is the value the swap in
      // ForceOutBegin expects: if the swap succeeds, the mark and the stamp
      // read in between belong to the publish that held the slot then, not
      // to a newer one that took the slot while this thread was descheduled.
      // The mark is read before the stamp: PublishBegin writes the stamp
      // first, so a marked slot carries the time this publish began.
      // (`now` was read before the lookup: a publish begun since is newer.)
      const uint64_t g = s->gen.load(std::memory_order_acquire);
      if (!RecoverStrandedForceOut(s, now)) {
        if (g & 1) return Role::kWait;
        if (!MarkedFenceable(s)) return Role::kWait;  // (an acquire load)
        const uint64_t started =
            s->fetch_start_ms.load(std::memory_order_relaxed);
        if (started >= now ||
            now - started <= static_cast<uint64_t>(takeover_ms_) ||
            !ForceOutBegin(s, g))
          return Role::kWait;
        ForceOutFinish(s, g + 1);
      }
    }
  }
  if (Slot* mine = Reserve(key)) {
    // Concurrent lockstep claimers can reserve two slots for one key when the
    // second Find races the first identity write (a roughly 100 ns window,
    // observed live as exactly twice as many server reads as unique keys).
    // Re-scan immediately and after a short settle spin. The lowest probe
    // position is canonical because it is what every waiter finds; a
    // non-canonical winner releases its slot and waits.
    for (int pass = 0; pass < 2; ++pass) {
      Slot* first = Find(key);
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

uint64_t GpuNodeDedup::TokenFor(const Slot* s, uint64_t gen) const {
  return dedup_token::Make(static_cast<uint32_t>(s - slots_), gen);
}

GpuNodeDedup::Slot* GpuNodeDedup::OwnedSlot(uint64_t token) const {
  if (!dedup_token::Valid(token)) return nullptr;
  const uint32_t idx = dedup_token::SlotIndex(token);
  if (idx >= nslots_) return nullptr;
  Slot* s = &slots_[idx];
  if (!dedup_token::SameGen(token, s->gen.load(std::memory_order_acquire)))
    return nullptr;
  return s;
}

bool GpuNodeDedup::LockOwned(Slot* s, const BlockKey& key, uint64_t token) {
  // Bounded wait for a stale party's re-check to hand the slot back; see
  // NodeDedup::LockOwned.
  const auto give_up =
      std::chrono::steady_clock::now() +
      std::chrono::milliseconds(kDedupLockWaitMs);
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
  // See NodeDedup::LockOwned: the slot may have been invalidated and claimed
  // again between the generation check and the lock.
  if (!dedup_token::SameGen(token, s->gen.load(std::memory_order_acquire)) ||
      s->key_hi != key.digest_hi || s->key_lo != key.digest_lo) {
    s->state.store(kStateFetching, std::memory_order_release);
    return false;
  }
  return true;
}

bool GpuNodeDedup::FenceLoads() {
  const char* fence = std::getenv("DFKV_GPU_LOAD_FENCE");
  if (fence && std::strcmp(fence, "1") != 0) return true;  // switched off
  EnsureThreadCtx();
  if (!cu_->CtxSynchronize || cu_->CtxSynchronize() != kCudaSuccess) {
    load_fence_failures_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  load_fences_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

// A publisher that can be fenced says so in the slot while it holds it: it
// stores kFenceablePublisher into alloc_seq as soon as it has the slot. The
// slot has no spare field, but alloc_seq is read by nobody outside READY, and
// every other value it ever holds, in any release, is 0 (Reserve) or an arena
// cursor (a commit). The mark therefore cannot be mistaken for part of an
// older release's commit, whose stores another core may see in any order
// (owner fields before len, say), nor for a READY entry that an invalidation
// holds PUBLISHING for a moment. A slot held without the mark belongs to an
// older release, whose commit is a set of unconditional stores: it is never
// taken away, and a killed one blocks its key as before. So does a current
// publisher killed between LockOwned and the mark, or inside the odd-
// generation window of its commit or of PublishAbandon.
bool GpuNodeDedup::MarkedFenceable(const Slot* s) {
  return __atomic_load_n(&s->alloc_seq, __ATOMIC_ACQUIRE) ==
         kFenceablePublisher;
}

bool GpuNodeDedup::ForceOutBegin(Slot* s, uint64_t g) {
  // g was read before the state and the mark: every way out of PUBLISHING
  // moves the generation, so if it still has this value at the swap below,
  // the state and the mark read in between belong to one and the same
  // publish.
  if (g & 1) return false;  // a publisher or another forcer is mid-write
  if (s->state.load(std::memory_order_acquire) != kStatePublishing)
    return false;
  if (!MarkedFenceable(s)) return false;
  // The publisher commits, and gives up, by swapping gen from g to g + 1; so
  // does this. Exactly one swap succeeds. If it is this one, the publisher,
  // alive or not, writes nothing more to the slot, and the odd generation
  // keeps every other forcer and every reader out until ForceOutFinish.
  if (!s->gen.compare_exchange_strong(g, g + 1, std::memory_order_acq_rel))
    return false;
  // Say so in the slot, with the time, so that a forcer killed before it
  // finishes can be finished for (RecoverStrandedForceOut).
  s->fetch_start_ms.store(NowMs(), std::memory_order_relaxed);
  __atomic_store_n(&s->alloc_seq, ForcingOutTag(g + 1), __ATOMIC_RELEASE);
  return true;
}

bool GpuNodeDedup::ForceOutFinish(Slot* s, uint64_t odd) {
  // A swap again, from the odd value the force-out left, to the next odd
  // one: the forcer and a recoverer (RecoverStrandedForceOut) may both try
  // to finish, and exactly one does. A forcer that comes back after a
  // recoverer finished for it finds the generation moved and writes nothing.
  uint64_t g = odd;
  if (!s->gen.compare_exchange_strong(g, odd + 2, std::memory_order_acq_rel))
    return false;
  // The sentinel goes before the slot leaves this odd window. Left in an
  // EMPTY slot, it would still be there while a Reserve that has swapped
  // the state and not yet cleared alloc_seq races an invalidation of the
  // old key, which holds the slot PUBLISHING with an odd generation: that
  // looks exactly like a stranded force-out, and a recovery would step the
  // generation under the other two.
  __atomic_store_n(&s->alloc_seq, 0, __ATOMIC_RELAXED);
  s->gen.fetch_add(1, std::memory_order_release);  // even: token superseded
  s->state.store(kStateEmpty, std::memory_order_release);
  return true;
}

bool GpuNodeDedup::ForceOutPublisher(Slot* s) {
  const uint64_t g = s->gen.load(std::memory_order_acquire);
  if (!ForceOutBegin(s, g)) return false;
  ForceOutFinish(s, g + 1);  // false: a recoverer finished for it
  return true;
}

bool GpuNodeDedup::RecoverStrandedForceOut(Slot* s, uint64_t now) {
  // A forcer killed between its swap and its finish leaves the slot
  // PUBLISHING with an odd generation for good. Nobody can commit to it: a
  // commit or an abandon needs the swap from the even value the forcer
  // already moved. The forcer marked the slot with the tag of its odd
  // generation and stamped it (only it writes in this odd window, and the
  // mark is a release store after the stamp); once that is older than the
  // takeover age, finish for it. If it is merely slow, the swap in
  // ForceOutFinish decides which of the two finishes. The tag must name the
  // generation read here: the winning finisher's swap moves the generation
  // to the next odd value before it clears the tag, and a second party that
  // looks in between must not take that for a stranded force-out too (it
  // would finish the same release a second time and leave the generation
  // odd at rest). The tag exists only inside a force-out's own odd window
  // (ForceOutFinish clears it before it leaves). A forcer killed after its
  // swap and before its mark, or a finisher killed after its swap and
  // before its last store, still strands the slot.
  const uint64_t g = s->gen.load(std::memory_order_acquire);
  if (!(g & 1)) return false;
  if (s->state.load(std::memory_order_acquire) != kStatePublishing)
    return false;
  if (__atomic_load_n(&s->alloc_seq, __ATOMIC_ACQUIRE) != ForcingOutTag(g))
    return false;
  const uint64_t started = s->fetch_start_ms.load(std::memory_order_relaxed);
  if (started >= now || now - started <= static_cast<uint64_t>(takeover_ms_))
    return false;
  return ForceOutFinish(s, g);
}

bool GpuNodeDedup::PublishBegin(const BlockKey& key, uint64_t token,
                                size_t len, PublishHold* hold) {
  Slot* s = OwnedSlot(token);
  if (!s || !LockOwned(s, key, token)) return false;
  if (len == 0 || len > arena_bytes_ / 2) {
    // Held without the mark: nobody can have taken the slot.
    s->state.store(kStateEmpty, std::memory_order_release);
    return false;
  }
  hold->slot = s;
  hold->gen = s->gen.load(std::memory_order_acquire);
  hold->len = len;
  // The age of a PUBLISHING slot is the time spent publishing, not the time
  // the fetch took before: a claim gives the copy takeover_ms from here. The
  // stamp is written before the mark (a release store), so whoever sees the
  // mark sees this stamp.
  s->fetch_start_ms.store(NowMs(), std::memory_order_relaxed);
  // From here an invalidation may take the slot at any moment.
  __atomic_store_n(&s->alloc_seq, kFenceablePublisher, __ATOMIC_RELEASE);
  return true;
}

void GpuNodeDedup::PublishAbandon(const PublishHold& hold) {
  // The same swap as a commit. If an invalidation has taken the slot it may
  // already be someone else's claim: then there is nothing to give up.
  Slot* s = hold.slot;
  uint64_t g = hold.gen;
  if (!s->gen.compare_exchange_strong(g, hold.gen + 1,
                                      std::memory_order_acq_rel))
    return;
  __atomic_store_n(&s->alloc_seq, 0, __ATOMIC_RELAXED);
  s->gen.fetch_add(1, std::memory_order_release);
  s->state.store(kStateEmpty, std::memory_order_release);
}

void GpuNodeDedup::PublishFinish(const PublishHold& hold, const Seg* segs,
                                 size_t nsegs) {
  Slot* s = hold.slot;
  const size_t n = hold.len;
  ProcEntry& e = reg_[self_idx_];
  // Ring-allocate on this process's arena cursor. Skip the tail remainder when
  // a lap boundary would split one payload, matching the host rendezvous.
  uint64_t cur = e.alloc_cursor.load(std::memory_order_relaxed);
  uint64_t off, seq;
  for (;;) {
    const uint64_t in_lap = cur % arena_bytes_;
    const uint64_t need =
        (in_lap + n > arena_bytes_) ? (arena_bytes_ - in_lap) + n : n;
    if (e.alloc_cursor.compare_exchange_weak(cur, cur + need,
                                             std::memory_order_acq_rel)) {
      seq = cur + need;
      off = (cur + need - n) % arena_bytes_;
      break;
    }
  }
  // Gather caller-owned device segments into the rendezvous arena on its
  // private CUDA stream, then synchronize before publishing READY. Without the
  // synchronization, peers can copy stale arena bytes; this was observed as
  // silently corrupted KV payloads rather than a transport error.
  CUstream st = Stream();
  if (!st) return PublishAbandon(hold);
  size_t done = 0;
  for (size_t j = 0; j < nsegs && done < n; ++j) {
    const size_t m = std::min(n - done, segs[j].cap);
    if (m == 0) continue;
    if (cu_->MemcpyAsync(arena_base_ + off + done,
                         reinterpret_cast<CUdeviceptr>(segs[j].ptr), m,
                         st) != kCudaSuccess)
      return PublishAbandon(hold);
    done += m;
  }
  if (done != n || cu_->StreamSynchronize(st) != kCudaSuccess)
    return PublishAbandon(hold);
  // Commit. A failed swap means an invalidation (or a claim past the takeover
  // age) took the slot during the copy: it is no longer this publisher's, so
  // nothing more is written.
  uint64_t g = hold.gen;
  if (!s->gen.compare_exchange_strong(g, hold.gen + 1,
                                      std::memory_order_acq_rel))
    return;
  s->len = static_cast<uint32_t>(n);
  s->payload_off = off;
  __atomic_store_n(&s->alloc_seq, seq, __ATOMIC_RELAXED);  // replaces the mark
  s->owner_idx = self_idx_;
  s->owner_gen32 = static_cast<uint32_t>(self_gen_);
  s->fetch_start_ms.store(NowMs(), std::memory_order_relaxed);
  s->gen.fetch_add(1, std::memory_order_release);
  s->state.store(kStateReady, std::memory_order_release);
}

void GpuNodeDedup::PublishSg(const BlockKey& key, uint64_t token,
                             const Seg* segs, size_t nsegs, size_t len) {
  if (!dedup_token::Valid(token)) return;
  EnsureThreadCtx();
  PublishHold hold;
  if (PublishBegin(key, token, len, &hold)) PublishFinish(hold, segs, nsegs);
}

void GpuNodeDedup::InterleaveForTest(const PublishHold& hold, int step) {
  Slot* s = hold.slot;
  switch (step) {
    case 0: {  // Reserve: the state swap
      uint32_t st = kStateEmpty;
      s->state.compare_exchange_strong(st, kStateFetching,
                                       std::memory_order_acq_rel);
      break;
    }
    case 1: {  // Invalidate of the old key: the state swap, first step
      uint32_t st = kStateFetching;
      s->state.compare_exchange_strong(st, kStatePublishing,
                                       std::memory_order_acq_rel);
      s->gen.fetch_add(1, std::memory_order_acq_rel);
      break;
    }
    case 2:  // Invalidate: the rest
      s->gen.fetch_add(1, std::memory_order_release);
      s->state.store(kStateEmpty, std::memory_order_release);
      break;
    case 3:  // Reserve: the rest
      s->gen.fetch_add(1, std::memory_order_acq_rel);
      __atomic_store_n(&s->alloc_seq, 0, __ATOMIC_RELAXED);
      s->gen.fetch_add(1, std::memory_order_release);
      break;
    case 4: {  // a finisher of a stranded force-out: its swap only
      uint64_t g = s->gen.load(std::memory_order_acquire);
      s->gen.compare_exchange_strong(g, g + 2, std::memory_order_acq_rel);
      break;
    }
    case 6: {  // a forcer's swap and tag, whatever the slot's mark
      uint64_t g = s->gen.load(std::memory_order_acquire);
      if (s->gen.compare_exchange_strong(g, g + 1, std::memory_order_acq_rel)) {
        s->fetch_start_ms.store(NowMs(), std::memory_order_relaxed);
        __atomic_store_n(&s->alloc_seq, ForcingOutTag(g + 1), __ATOMIC_RELEASE);
      }
      break;
    }
    case 5:  // that finisher: the rest
      __atomic_store_n(&s->alloc_seq, 0, __ATOMIC_RELAXED);
      s->gen.fetch_add(1, std::memory_order_release);
      s->state.store(kStateEmpty, std::memory_order_release);
      break;
  }
}

bool GpuNodeDedup::RecoverForTest(const PublishHold& hold) {
  return RecoverStrandedForceOut(hold.slot, NowMs());
}

bool GpuNodeDedup::GenEvenForTest(const PublishHold& hold) const {
  return (hold.slot->gen.load(std::memory_order_acquire) & 1) == 0;
}

bool GpuNodeDedup::ShiftStampForTest(const BlockKey& key, int64_t delta_ms) {
  Slot* s = Find(key);
  if (!s) return false;
  s->fetch_start_ms.fetch_add(static_cast<uint64_t>(delta_ms),
                              std::memory_order_relaxed);
  return true;
}

void GpuNodeDedup::SetFenceableForTest(const PublishHold& hold, bool on) {
  if (on) {
    __atomic_store_n(&hold.slot->alloc_seq, kFenceablePublisher,
                     __ATOMIC_RELEASE);
    return;
  }
  // An older release's publisher as another core may see it in its commit:
  // the owner fields already stored, len and alloc_seq not yet.
  __atomic_store_n(&hold.slot->alloc_seq, 0, __ATOMIC_RELAXED);
  hold.slot->owner_idx = self_idx_;
  hold.slot->owner_gen32 = static_cast<uint32_t>(self_gen_);
}

void GpuNodeDedup::Abort(const BlockKey& key, uint64_t token) {
  Slot* s = OwnedSlot(token);
  if (!s || !LockOwned(s, key, token)) return;
  s->state.store(kStateEmpty, std::memory_order_release);
}

void GpuNodeDedup::Invalidate(const BlockKey& key) {
  // Every entry of the key in its probe window, not only the first one a
  // lookup finds: see NodeDedup::InvalidateKind.
  const uint32_t mask = nslots_ - 1;
  uint32_t idx = ProbeStart(key) & mask;
  for (int probe = 0; probe < 8; ++probe, idx = (idx + 1) & mask) {
    Slot& s = slots_[idx];
    uint64_t spin_since = 0;  // when this call began to wait for the slot
    uint64_t finishing_since = 0;  // since when FinishingRelease has held
    bool reported = false;
    for (;;) {
      uint32_t st = s.state.load(std::memory_order_acquire);
      if (st == kStateEmpty) break;
      if (s.key_hi != key.digest_hi || s.key_lo != key.digest_lo) break;
      if (st == kStatePublishing) {
        // What is being published is from before this invalidation. A
        // publisher that can be fenced loses the slot here and now, whether
        // it is still running or was killed. A force-out whose forcer was
        // killed is finished once it is older than the takeover age. A
        // release that is past its finishing swap can only end EMPTY: a live
        // finisher is through in a few instructions, so it is waited for
        // like any holder, but once that state has held for longer than the
        // takeover age the finisher is taken to be dead, and the slot is
        // left behind (reported). Any other holder is waited for, without bound:
        // an older release's publisher, or one of the short windows listed
        // at MarkedFenceable. Past the takeover age that is reported once.
        const uint64_t now = NowMs();
        if (ForceOutPublisher(&s) || RecoverStrandedForceOut(&s, now))
          break;
        if (FinishingRelease(&s)) {
          if (finishing_since == 0) {
            finishing_since = now;
          } else if (now - finishing_since >
                     static_cast<uint64_t>(takeover_ms_)) {
            if (!reported)  // at most one report per slot and call
              ReportStuck(idx, &s,
                          "a release past its finishing swap has not ended "
                          "within DFKV_NODE_DEDUP_TAKEOVER_MS; it can only "
                          "end EMPTY, so not waiting for it any longer");
            break;
          }
          std::this_thread::yield();
          continue;
        }
        finishing_since = 0;
        if (spin_since == 0) {
          spin_since = now;
        } else if (!reported &&
                   now - spin_since > static_cast<uint64_t>(takeover_ms_)) {
          ReportStuck(idx, &s,
                      "waited longer than DFKV_NODE_DEDUP_TAKEOVER_MS for its "
                      "holder (an older release's publisher, or a process "
                      "killed in a short window); still waiting");
          reported = true;
        }
        std::this_thread::yield();
        continue;
      }
      if (!s.state.compare_exchange_weak(st, kStatePublishing,
                                         std::memory_order_acq_rel))
        continue;
      s.gen.fetch_add(1, std::memory_order_acq_rel);
      s.gen.fetch_add(1, std::memory_order_release);
      s.state.store(kStateEmpty, std::memory_order_release);
      break;
    }
  }
}

// A finisher (ForceOutFinish) has swapped the generation from the force-out's
// odd value g - 2 to g and not yet cleared the force-out's tag: the slot is
// PUBLISHING at odd g with alloc_seq == ForcingOutTag(g - 2). Only that
// finisher writes the slot from here, and only to clear the tag, step the
// generation and store EMPTY: nothing can make it READY again. Nobody else
// can produce the combination: the tag is written only by ForceOutBegin, for
// the odd generation its own swap made, and older releases never write it.
// A commit, an abandon or an older release's publisher is told apart this
// way and is still waited for: their odd windows can end READY.
bool GpuNodeDedup::FinishingRelease(const Slot* s) const {
  const uint64_t g = s->gen.load(std::memory_order_acquire);
  if (!(g & 1)) return false;
  if (s->state.load(std::memory_order_acquire) != kStatePublishing)
    return false;
  if (__atomic_load_n(&s->alloc_seq, __ATOMIC_ACQUIRE) != ForcingOutTag(g - 2))
    return false;
  return s->gen.load(std::memory_order_acquire) == g;
}

void GpuNodeDedup::ReportStuck(uint32_t idx, const Slot* s, const char* what) {
  invalidate_stuck_.fetch_add(1, std::memory_order_relaxed);
  char line[256];
  std::snprintf(
      line, sizeof(line),
      "gpu-dedup: invalidation, slot %u (gen=%llu state=%u "
      "alloc_seq=0x%016llx): %s",
      idx,
      static_cast<unsigned long long>(s->gen.load(std::memory_order_acquire)),
      s->state.load(std::memory_order_acquire),
      static_cast<unsigned long long>(
          __atomic_load_n(&s->alloc_seq, __ATOMIC_RELAXED)),
      what);
  DFKV_LOG_WARN(line);
}

bool GpuNodeDedup::WaitSg(const BlockKey& key, const Seg* segs, size_t nsegs,
                          size_t total_cap, size_t* got) {
  EnsureThreadCtx();
  const uint64_t deadline = NowMs() + static_cast<uint64_t>(wait_ms_);
  int backoff_us = 50;
  bool gone = false;
  for (;;) {
    Slot* s = Find(key);
    const uint32_t state =
        s ? s->state.load(std::memory_order_acquire) : kStateEmpty;
    if (state == kStateReady) {
      if (AgeMs(NowMs(), s->fetch_start_ms.load(std::memory_order_relaxed)) <=
              static_cast<uint64_t>(ttl_ms_) &&
          CopyOutSg(s, key, segs, nsegs, total_cap, got)) {
        wait_hits_.fetch_add(1, std::memory_order_relaxed);
        return true;
      }
      break;  // expired, lapped, recycled or too long: it will not heal
    }
    if (state == kStateFetching || state == kStatePublishing) {
      gone = false;
      if (NowMs() >= deadline) break;
      std::this_thread::sleep_for(std::chrono::microseconds(backoff_us));
      if (backoff_us < 1000) backoff_us *= 2;
      continue;
    }
    // No fetch in flight for this key (see WaitManySg): look once more, then
    // give up at once.
    if (gone) break;
    gone = true;
  }
  wait_timeouts_.fetch_add(1, std::memory_order_relaxed);
  return false;
}

}  // namespace dfkv
