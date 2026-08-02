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
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "utils/log.h"

namespace dfkv {

namespace {
constexpr uint32_t kMagic = 0x44444733u;  // "DDG3" (publisher ownership tokens)
constexpr uint32_t kStateEmpty = 0;
constexpr uint32_t kStateFetching = 1;
constexpr uint32_t kStateReady = 2;
constexpr uint32_t kStatePublishing = 3;

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

bool PidDead(uint32_t pid) {
  return ::kill(static_cast<pid_t>(pid), 0) != 0 && errno == ESRCH;
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
  uint64_t alloc_seq;     // owner's arena cursor value at allocation (lap check)
  uint32_t owner_idx;     // registry entry of the publishing process
  uint32_t owner_gen32;   // truncated entry generation at publish time
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
  uint32_t pad0;
  CUipcMemHandle handle;
  // The exporter's own arena VA: cuIpcOpenMemHandle refuses handles from the
  // EXPORTING process, so a second instance in the same process (same address
  // space) reads through this pointer directly. Only trusted when pid matches.
  uint64_t owner_va;
  uint64_t pad1[2];  // -> 128 B (cross-process ABI)
};

struct GpuNodeDedup::Header {
  uint32_t magic;
  uint32_t nslots;
  uint64_t arena_bytes;  // configured per-process arena size (must match)
  std::atomic<uint32_t> init_done;
  uint32_t pad;
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
  if (!on || std::strcmp(on, "1") != 0) return nullptr;
  const char* gpu = std::getenv("DFKV_CLIENT_NODE_DEDUP_GPU");
  if (!gpu || std::strcmp(gpu, "1") != 0) return nullptr;
  const CudaLib* cu = CudaLib::Get();
  if (!cu) {
    DFKV_LOG_INFO("gpu-dedup: no usable CUDA driver, running without");
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
  o.name = EnvSegmentName(namespace_hash);
  return Open(o);
}

std::unique_ptr<GpuNodeDedup> GpuNodeDedup::Open(const Options& opt) {
  static_assert(sizeof(Slot) == 64, "shm slot layout is cross-process ABI");
  static_assert(sizeof(ProcEntry) == 128, "shm registry layout is cross-process ABI");
  const CudaLib* cu = CudaLib::Get();
  if (!cu || !cu->HasCurrentCtx()) return nullptr;

  const uint32_t nslots = Pow2AtLeast(opt.slots);
  const size_t len = sizeof(Header) + sizeof(ProcEntry) * kMaxProcs +
                     static_cast<size_t>(nslots) * sizeof(Slot);
  int fd = ::shm_open(opt.name.c_str(), O_RDWR | O_CREAT, 0600);
  if (fd < 0) return nullptr;
  struct stat st{};
  if (::fstat(fd, &st) != 0) { ::close(fd); return nullptr; }
  const bool creator = (st.st_size == 0);
  if (creator) {
    // Small segment (~1-2 MiB: descriptors only, payloads live in GPU
    // memory), but the eager-backing rule from the host flavor still holds:
    // ENOSPC here beats a SIGBUS mid-publish.
    if (::ftruncate(fd, static_cast<off_t>(len)) != 0 ||
        ::posix_fallocate(fd, 0, static_cast<off_t>(len)) != 0) {
      ::close(fd);
      ::shm_unlink(opt.name.c_str());
      return nullptr;
    }
  } else if (static_cast<size_t>(st.st_size) != len) {
    DFKV_LOG_INFO("gpu-dedup: existing segment layout mismatch, running without");
    ::close(fd);
    return nullptr;
  }
  void* base = ::mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ::close(fd);
  if (base == MAP_FAILED) return nullptr;

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
    for (int spin = 0; spin < 1000 && d->hdr_->init_done.load(std::memory_order_acquire) != 1; ++spin)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
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

  // Claim a registry entry: a free one, or one whose owner died.
  const uint32_t mypid = static_cast<uint32_t>(::getpid());
  for (uint32_t i = 0; i < kMaxProcs && d->self_idx_ == kMaxProcs; ++i) {
    ProcEntry& e = d->reg_[i];
    uint32_t cur = e.pid.load(std::memory_order_acquire);
    if (cur == mypid) continue;  // stale self entry from an exec'd ancestor: skip
    if (cur != 0 && !PidDead(cur)) continue;
    if (!e.pid.compare_exchange_strong(cur, mypid, std::memory_order_acq_rel))
      continue;
    e.ready.store(0, std::memory_order_release);
    d->self_gen_ = e.generation.fetch_add(1, std::memory_order_acq_rel) + 1;
    e.alloc_cursor.store(0, std::memory_order_relaxed);
    e.arena_bytes = opt.arena_bytes;
    e.device = cu->CurrentDevice();
    d->self_device_ = e.device;
    std::memcpy(&e.handle, &handle, sizeof(handle));
    e.owner_va = static_cast<uint64_t>(d->arena_base_);
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
  uint32_t idx = static_cast<uint32_t>(
      key.digest_hi ^ (key.digest_hi >> 32) ^ key.digest_lo ^
      (key.digest_lo >> 32)) & mask;
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
  uint32_t idx = static_cast<uint32_t>(
      key.digest_hi ^ (key.digest_hi >> 32) ^ key.digest_lo ^
      (key.digest_lo >> 32)) & mask;
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
    s.gen.fetch_add(1, std::memory_order_acq_rel);  // odd: mutating identity
    s.key_hi = key.digest_hi;
    s.key_lo = key.digest_lo;
    s.len = 0;
    s.payload_off = 0;
    s.alloc_seq = 0;
    s.owner_idx = kMaxProcs;
    s.owner_gen32 = 0;
    s.fetch_start_ms.store(now, std::memory_order_relaxed);
    s.gen.fetch_add(1, std::memory_order_release);  // even: coherent
    return &s;
  }
  return nullptr;
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
      if (e.ready.load(std::memory_order_acquire) != 1 ||
          e.generation.load(std::memory_order_acquire) != gen)
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

bool GpuNodeDedup::IssueCopySg(Slot* s, const Seg* segs, size_t nsegs,
                               size_t total_cap, CUstream st, CopySnap* snap) {
  const uint64_t g0 = s->gen.load(std::memory_order_acquire);
  if (g0 & 1) return false;
  if (s->state.load(std::memory_order_acquire) != kStateReady) return false;
  const size_t n = s->len;
  if (n == 0 || n > total_cap) return false;
  const uint32_t oi = s->owner_idx;
  const uint32_t og = s->owner_gen32;
  const uint64_t off = s->payload_off;
  const uint64_t seq = s->alloc_seq;
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

bool GpuNodeDedup::CopyOutSg(Slot* s, const Seg* segs, size_t nsegs,
                             size_t total_cap, size_t* got) {
  CUstream st = Stream();
  if (!st) return false;
  CopySnap snap{};
  if (!IssueCopySg(s, segs, nsegs, total_cap, st, &snap)) return false;
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
  const uint64_t deadline = NowMs() + static_cast<uint64_t>(wait_ms_);
  int backoff_us = 50;
  size_t failed_final = 0;
  while (!pending.empty()) {
    // Issue round: enqueue every key that is READY right now.
    size_t w = 0;
    for (size_t r = 0; r < pending.size(); ++r) {
      const size_t idx = pending[r];
      WaitItem& it = items[idx];
      Slot* s = Find(it.key);
      CopySnap snap{};
      if (s && s->state.load(std::memory_order_acquire) == kStateReady &&
          IssueCopySg(s, it.segs, it.nsegs, it.total_cap, st, &snap)) {
        inflight.push_back(InFlight{idx, snap});
      } else {
        pending[w++] = idx;  // not published yet (or torn): retry next round
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
    if (NowMs() >= deadline) break;
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
      if (now - s->fetch_start_ms.load(std::memory_order_relaxed) <=
              static_cast<uint64_t>(ttl_ms_) &&
          CopyOutSg(s, segs, nsegs, total_cap, got)) {
        hits_.fetch_add(1, std::memory_order_relaxed);
        return Role::kHit;
      }
    } else if (st == kStateFetching) {
      const uint64_t started = s->fetch_start_ms.load(std::memory_order_relaxed);
      if (now - started <= static_cast<uint64_t>(takeover_ms_))
        return Role::kWait;
      uint32_t expect = kStateFetching;
      if (s->state.compare_exchange_strong(expect, kStatePublishing,
                                           std::memory_order_acq_rel)) {
        s->gen.fetch_add(1, std::memory_order_acq_rel);
        s->fetch_start_ms.store(now, std::memory_order_relaxed);
        const uint64_t mine = s->gen.fetch_add(1, std::memory_order_release) + 1;
        s->state.store(kStateFetching, std::memory_order_release);
        if (token) *token = mine;
        fetches_.fetch_add(1, std::memory_order_relaxed);
        return Role::kFetch;
      }
      return Role::kWait;
    } else if (st == kStatePublishing) {
      return Role::kWait;
    }
  }
  if (Slot* mine = Reserve(key)) {
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
    if (token) *token = mine->gen.load(std::memory_order_acquire);
    fetches_.fetch_add(1, std::memory_order_relaxed);
    return Role::kFetch;
  }
  return Role::kFetch;
}

void GpuNodeDedup::PublishSg(const BlockKey& key, uint64_t token,
                             const Seg* segs, size_t nsegs, size_t len) {
  if (token == 0) return;
  EnsureThreadCtx();
  Slot* s = Find(key);
  if (!s || s->gen.load(std::memory_order_acquire) != token) return;
  uint32_t state = kStateFetching;
  if (!s->state.compare_exchange_strong(state, kStatePublishing,
                                        std::memory_order_acq_rel))
    return;
  const size_t n = len;
  auto abandon = [&] { s->state.store(kStateEmpty, std::memory_order_release); };
  if (n == 0 || n > arena_bytes_ / 2) return abandon();
  ProcEntry& e = reg_[self_idx_];
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
  // Gather on the rendezvous stream and synchronize before READY.
  CUstream st = Stream();
  if (!st) return abandon();
  size_t done = 0;
  for (size_t j = 0; j < nsegs && done < n; ++j) {
    const size_t m = std::min(n - done, segs[j].cap);
    if (m == 0) continue;
    if (cu_->MemcpyAsync(arena_base_ + off + done,
                         reinterpret_cast<CUdeviceptr>(segs[j].ptr), m,
                         st) != kCudaSuccess)
      return abandon();
    done += m;
  }
  if (done != n || cu_->StreamSynchronize(st) != kCudaSuccess) return abandon();
  s->gen.fetch_add(1, std::memory_order_acq_rel);
  s->len = static_cast<uint32_t>(n);
  s->payload_off = off;
  s->alloc_seq = seq;
  s->owner_idx = self_idx_;
  s->owner_gen32 = static_cast<uint32_t>(self_gen_);
  s->fetch_start_ms.store(NowMs(), std::memory_order_relaxed);
  s->gen.fetch_add(1, std::memory_order_release);
  s->state.store(kStateReady, std::memory_order_release);
}

void GpuNodeDedup::Abort(const BlockKey& key, uint64_t token) {
  if (token == 0) return;
  Slot* s = Find(key);
  if (!s || s->gen.load(std::memory_order_acquire) != token) return;
  uint32_t st = kStateFetching;
  s->state.compare_exchange_strong(st, kStateEmpty, std::memory_order_acq_rel);
}

void GpuNodeDedup::Invalidate(const BlockKey& key) {
  for (;;) {
    Slot* s = Find(key);
    if (!s) return;
    uint32_t st = s->state.load(std::memory_order_acquire);
    if (st == kStatePublishing) {
      std::this_thread::yield();
      continue;
    }
    if (st != kStateFetching && st != kStateReady) return;
    if (!s->state.compare_exchange_weak(st, kStatePublishing,
                                        std::memory_order_acq_rel))
      continue;
    s->gen.fetch_add(1, std::memory_order_acq_rel);
    s->gen.fetch_add(1, std::memory_order_release);
    s->state.store(kStateEmpty, std::memory_order_release);
    return;
  }
}

bool GpuNodeDedup::WaitSg(const BlockKey& key, const Seg* segs, size_t nsegs,
                          size_t total_cap, size_t* got) {
  EnsureThreadCtx();
  const uint64_t deadline = NowMs() + static_cast<uint64_t>(wait_ms_);
  int backoff_us = 50;
  for (;;) {
    Slot* s = Find(key);
    if (s && s->state.load(std::memory_order_acquire) == kStateReady &&
        CopyOutSg(s, segs, nsegs, total_cap, got)) {
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

}  // namespace dfkv
