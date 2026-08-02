/* GpuNodeDedup — same-host GET rendezvous for GPU (device-memory) destinations.
 *
 * Phase 2b of the NodeDedup design (see node_dedup.h for the host flavor and
 * the full correctness model): the vLLM connector's GETs land via GPUDirect
 * straight in device memory, so the host rendezvous' memcpy-through-shm can
 * not serve them. Same lockstep pattern though — TP ranks re-fetch identical
 * TP-replicated KV within milliseconds — so the same slot machinery applies;
 * only the payload channel changes:
 *
 *  - Every participating process owns a private GPU STAGING ARENA (cuMemAlloc
 *    on its current device) and exports it once via a CUDA IPC handle kept in
 *    a small shm registry. A publisher gathers its fetched segments into its
 *    own arena (D2D); a waiter opens the publisher's arena (cached handle,
 *    lazy peer access = NVLink) and scatters straight into its destination
 *    segments — the copy never touches host memory.
 *  - The index slots and per-arena allocation cursors live in shm exactly
 *    like the host flavor: monotonic cursor ring + before/after lap check +
 *    per-slot seqlock. The cursor of arena X lives in X's registry entry.
 *  - Registry entries are pid-claimed; a claim bumps the entry generation, so
 *    a slot published against a died-and-replaced arena fails its generation
 *    check and degrades to a direct fetch. The driver validates VA ranges on
 *    cuMemcpy, so a racing unmap yields an error, not corruption.
 *
 * CUDA is a dlopen'd soft dependency (cuda_ipc.h); no driver, no current
 * context, or any CUDA error on the copy path => the caller just fetches
 * directly. Nothing here is load-bearing.
 *
 * Off by default: DFKV_CLIENT_NODE_DEDUP=1 (master) plus
 * DFKV_CLIENT_NODE_DEDUP_GPU=1 opt in. */
#ifndef DFKV_NODE_DEDUP_GPU_H_
#define DFKV_NODE_DEDUP_GPU_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "client/cuda_ipc.h"
#include "common/kv_types.h"

namespace dfkv {

class GpuNodeDedup {
 public:
  struct Options {
    std::string name;                       // shm name; see FromEnv()
    uint64_t arena_bytes = 512ull << 20;    // per-PROCESS GPU staging arena
                                            // (DFKV_NODE_DEDUP_GPU_ARENA_MB)
    uint32_t slots = 16384;                 // DFKV_NODE_DEDUP_GPU_SLOTS (pow2)
    int wait_ms = 500;                      // shared with the host flavor:
    int takeover_ms = 2000;                 //   DFKV_NODE_DEDUP_WAIT_MS /
    int ttl_ms = 5000;                      //   _TAKEOVER_MS / _TTL_MS
  };

  // One destination (or source) segment of a scatter-gather item. ptr may be
  // device or host — cuMemcpy is unified-addressing — but the feature is only
  // routed to when the first segment is device memory.
  struct Seg {
    void* ptr;
    size_t cap;
  };

  enum class Role { kHit, kFetch, kWait };  // same contract as NodeDedup::Role

  // nullptr unless BOTH env switches are set AND the CUDA driver resolves.
  // device_dst_hint is a device pointer the caller is about to scatter into:
  // when the calling thread has no current CUDA context (the connectors'
  // pure-Python transfer threads never touched CUDA), the hint's device
  // selects which PRIMARY context to bind — the framework already holds it,
  // so this never creates a context, only makes it current here.
  static std::unique_ptr<GpuNodeDedup> FromEnv(uint64_t namespace_hash,
                                               const void* device_dst_hint);
  static std::string EnvSegmentName(uint64_t namespace_hash);
  static std::unique_ptr<GpuNodeDedup> Open(const Options& opt);
  ~GpuNodeDedup();

  GpuNodeDedup(const GpuNodeDedup&) = delete;
  GpuNodeDedup& operator=(const GpuNodeDedup&) = delete;

  // Variable-size SG semantics (BatchGetAutoSg): a published payload hits iff
  // its length fits sum(caps); it is scattered across segs in order. *got
  // receives the payload length. kFetch returns an ownership token that must
  // be passed to PublishSg/Abort so Remove can fence late publishers.
  Role ClaimSg(const BlockKey& key, const Seg* segs, size_t nsegs,
               size_t total_cap, size_t* got, uint64_t* token);
  bool WaitSg(const BlockKey& key, const Seg* segs, size_t nsegs,
              size_t total_cap, size_t* got);

  // One wait-item of a WaitManySg batch. segs/nsegs/total_cap mirror the
  // per-key API; ok/got are outputs (ok=false => caller re-fetches the key).
  struct WaitItem {
    BlockKey key;
    const Seg* segs;
    size_t nsegs;
    size_t total_cap;
    bool ok;
    size_t got;
  };
  // Waits for MANY keys with amortized synchronization: copies are issued
  // asynchronously the moment each key turns READY, and the stream is
  // synchronized ONCE per issue round, with post-validation (lap/seqlock)
  // after the sync. The per-key WaitSg costs a StreamSynchronize per payload
  // (~ms under load); a lockstep batch of >1000 keys blew straight through
  // any reasonable wait budget that way (measured 2.3 ms/key in the PD A/B —
  // the dedup win leaked back out through the fallback re-fetches). wait_ms
  // bounds how long the batch waits for publishes as a whole.
  void WaitManySg(WaitItem* items, size_t n);
  // Gathers the fetched segments (total payload = len bytes) into this
  // process' arena and flips the slot READY.
  void PublishSg(const BlockKey& key, uint64_t token, const Seg* segs,
                 size_t nsegs, size_t len);
  void Abort(const BlockKey& key, uint64_t token);
  // Eviction barrier for READY and in-flight entries.
  void Invalidate(const BlockKey& key);

  int wait_ms() const { return wait_ms_; }

  // diagnostics (relaxed; process-local)
  uint64_t hits() const { return hits_.load(std::memory_order_relaxed); }
  uint64_t fetches() const { return fetches_.load(std::memory_order_relaxed); }
  uint64_t wait_hits() const { return wait_hits_.load(std::memory_order_relaxed); }
  uint64_t wait_timeouts() const { return wait_timeouts_.load(std::memory_order_relaxed); }

  static constexpr uint32_t kMaxProcs = 64;

 private:
  struct Slot;       // 64-byte shm index slot (owner-descriptor flavor)
  struct ProcEntry;  // 128-byte shm registry entry (one per process)
  struct Header;
  GpuNodeDedup() = default;

  // Threads that never touched CUDA (fresh pthreads) lack a current context;
  // bind our device's primary context on demand so every entry point works
  // from any caller thread.
  void EnsureThreadCtx() const;
  // The rendezvous' own non-blocking stream (lazy; guarded by stream_mu_).
  // Copies must NOT ride the legacy default stream: that falsely serializes
  // them with the framework's in-flight kernels, and unsynchronized D2D
  // "completions" published stale arena bytes (both observed live).
  CUstream Stream();
  Slot* Find(const BlockKey& key) const;
  Slot* Reserve(const BlockKey& key);
  // Split copy-out: Issue pre-validates and ENQUEUES the scatter on st (no
  // sync); Validate re-checks lap/generation AFTER the caller synchronized.
  // CopyOutSg composes them for the single-key path; WaitManySg amortizes one
  // sync over a whole issue round.
  struct CopySnap {
    Slot* s;
    uint64_t g0;
    uint32_t oi;
    uint64_t egen;
    uint64_t seq;
    size_t n;
  };
  bool IssueCopySg(Slot* s, const Seg* segs, size_t nsegs, size_t total_cap,
                   CUstream st, CopySnap* snap);
  bool ValidateCopy(const CopySnap& snap) const;
  bool CopyOutSg(Slot* s, const Seg* segs, size_t nsegs, size_t total_cap,
                 size_t* got);
  // Cached cuIpcOpenMemHandle of registry entry idx at generation gen;
  // 0 on any failure. Self resolves to arena_base_ without an IPC open.
  CUdeviceptr PeerBase(uint32_t idx, uint64_t gen);
  static uint64_t NowMs();

  const CudaLib* cu_ = nullptr;
  Header* hdr_ = nullptr;
  ProcEntry* reg_ = nullptr;
  Slot* slots_ = nullptr;
  uint32_t nslots_ = 0;
  uint64_t arena_bytes_ = 0;
  int wait_ms_ = 500;
  int takeover_ms_ = 2000;
  int ttl_ms_ = 5000;
  void* map_base_ = nullptr;
  size_t map_len_ = 0;

  CUdeviceptr arena_base_ = 0;  // this process' staging arena (device memory)
  uint32_t self_idx_ = kMaxProcs;
  uint64_t self_gen_ = 0;
  int self_device_ = -1;  // arena's device (EnsureThreadCtx re-binds to it)
  std::mutex stream_mu_;
  CUstream stream_ = nullptr;

  std::mutex peer_mu_;  // guards peer_map_
  struct PeerMap {
    uint64_t gen = 0;
    CUdeviceptr base = 0;
  };
  PeerMap peer_map_[kMaxProcs];

  std::atomic<uint64_t> hits_{0}, fetches_{0}, wait_hits_{0}, wait_timeouts_{0};
};

}  // namespace dfkv

#endif  // DFKV_NODE_DEDUP_GPU_H_
