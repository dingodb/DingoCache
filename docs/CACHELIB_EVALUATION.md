# CacheLib / Navy evaluation for dfkv v2.0.0 (F25)

**Decision:** **reject integration for the v2.0.0 line**. CacheLib is not a dfkv
build dependency or supported storage engine. No adapter, CMake option, runtime
selector, compatibility reader, or placeholder hook is added by this decision.
The existing `DiskSlabStore` plus optional `RamTier` remains the production
storage path; `KVStore` remains the explicit diagnostic fallback.

**Decision date:** 2026-08-03
**CacheLib baseline evaluated:** upstream
[`faa9ef1e7c57cfb193c8bb0ca404c3ad239231af`](https://github.com/facebook/CacheLib/tree/faa9ef1e7c57cfb193c8bb0ca404c3ad239231af).
Pinning the source revision matters: CacheLib, RAM, and NVM each have independent
format versions, and upstream explicitly warns that incompatible version bumps
can drop a cache.

This is a technical rejection for this release line, not a claim that CacheLib
or Navy is unsuitable in general. CacheLib provides a mature in-process DRAM
allocator and Navy provides useful flash policies and asynchronous I/O. Their
published object lifecycle does not implement dfkv's current storage contract,
and adapting it would replace rather than wrap the most important v2
correctness and data-path invariants.

## 1. Scope and evidence

The comparison covers the server-local storage implementation only. Client
routing, the native TCP/RDMA v2 wire protocol, and connector identity are not
candidates for replacement.

The dfkv side was evaluated from these current source contracts:

- `src/cache/store_engine.h`: `StoreEngine`, direct/batched writes, partial
  ranges, caller-buffer reads, move-only `ReadLease`, health, committed-byte
  accounting, and removal.
- `src/cache/slab_allocator.{h,cc}`: fixed-size `BlockKey` lookup, value-semantic
  slot metadata, size classes, pin/deferred-remove lifecycle, restore, CLOCK
  eviction, and extent rebalance.
- `src/cache/disk_slab_store.{h,cc}`: preallocated extents, v3 fixed records,
  direct I/O, async read preparation, commit ordering, rebuild, and the
  clean/dirty run epoch.
- `src/cache/ram_tier.{h,cc}`: one bounded aligned arena, durable-acknowledged
  write-back/direct read-promotion, lifecycle pins, and the arena MR hand-off.
- `src/cache/disk_cache_group.{h,cc}`: per-disk routing and exact per-node tenant
  quota admission/accounting.
- `src/cache/rdma_server.{h,cc}` and `src/transport/rdma_recv_segment.*`:
  process-lifetime multi-rail registration, bounded registered receive storage,
  and completion-fenced release.
- `src/cache/kv_node_server.cc` and `docs/METRICS.md`: the existing Prometheus
  operational contract.
- `CMakeLists.txt`: the current C++17 core and the only production optional
  dependencies, `libibverbs` and `liburing`.

The CacheLib side uses source and documentation at the pinned revision:

- [CacheLib README](https://github.com/facebook/CacheLib/blob/faa9ef1e7c57cfb193c8bb0ca404c3ad239231af/README.md)
  describes an **in-process** DRAM/SSD cache.
- [HybridCache lifecycle](https://github.com/facebook/CacheLib/blob/faa9ef1e7c57cfb193c8bb0ca404c3ad239231af/website/docs/Cache_Library_User_Guides/HybridCache.md)
  says items are allocated in DRAM, spill to NVM when cold, and are promoted to
  DRAM on access; it also states that pools partition DRAM but not HybridCache.
- [Item and Handle](https://github.com/facebook/CacheLib/blob/faa9ef1e7c57cfb193c8bb0ca404c3ad239231af/website/docs/Cache_Library_User_Guides/Item_and_Handle.md)
  defines move-only handles that prevent eviction/reclamation and reports the
  per-item metadata overhead.
- [Navy configuration](https://github.com/facebook/CacheLib/blob/faa9ef1e7c57cfb193c8bb0ca404c3ad239231af/website/docs/Cache_Library_User_Guides/Configure_HybridCache.md)
  documents file/RAID devices, BlockCache/BigHash, region buffering, request
  schedulers, and the io_uring/libaio switch.
- [`Device.cpp`](https://github.com/facebook/CacheLib/blob/faa9ef1e7c57cfb193c8bb0ca404c3ad239231af/cachelib/navy/common/Device.cpp#L1151-L1177)
  shows Navy attempting `O_DIRECT` and falling back without it on `EINVAL`.
- [Cache persistence](https://github.com/facebook/CacheLib/blob/faa9ef1e7c57cfb193c8bb0ca404c3ad239231af/website/docs/Cache_Library_User_Guides/Cache_persistence.md)
  requires draining access and calling `shutdown()` for a recoverable shared
  memory attach, and documents separate NVM-drop controls.
- [`CacheVersion.h`](https://github.com/facebook/CacheLib/blob/faa9ef1e7c57cfb193c8bb0ca404c3ad239231af/cachelib/allocator/CacheVersion.h)
  defines CacheLib/RAM/NVM format versions 19/5/3 at the evaluated revision.
- [`CacheAllocator::getSlabMemoryInfo()`](https://github.com/facebook/CacheLib/blob/faa9ef1e7c57cfb193c8bb0ca404c3ad239231af/cachelib/allocator/CacheAllocator.h#L1293-L1306)
  exposes the DRAM slab base and size for pre-registration. The same surface
  exposes allocator and NVM statistics maps.
- [BUILD.md](https://github.com/facebook/CacheLib/blob/faa9ef1e7c57cfb193c8bb0ca404c3ad239231af/BUILD.md)
  requires C++20 and lists the Folly/FBThrift-centered dependency graph.

## 2. Required dfkv invariants

An alternative backend is acceptable only if it implements the observable
`StoreEngine` contract, not merely key/value insertion and lookup.

| Area | v2.0.0 invariant |
|---|---|
| Identity and values | A tenant-scoped, fixed-size `BlockKey` is the storage identity. Values are opaque raw bytes: no adapter-owned envelope, text conversion, or compatibility decode. |
| PUT acknowledgement | `kOk` means the disk commit used by this cache node succeeded. With `RamTier`, a value may become RAM-visible before its flush, but the server PUT waits for the shared flush result. Same-key followers observe the leader's exact result. |
| GET data path | Full and partial range reads are supported. RDMA reads can target a caller-owned aligned registered buffer. Async preparation returns one move-only lease owning the descriptor and storage pin through completion, fallback, error, or teardown. |
| Removal and reuse | REMOVE hides the key before returning and fences in-flight I/O. Pinned storage cannot be reused; deferred completion cannot republish a removed key. |
| Capacity | Capacity, eviction, and tenant usage are committed payload-byte semantics, not allocator-reserved bytes or an approximate object count. |
| Tenant admission | Per-node quotas serialize PUT/REMOVE by fixed stripes, charge only new committed keys, roll back failures, subtract eviction/removal, and rebuild from tenant-scoped records. One rejected tenant does not reject unrelated batch items. |
| Disk layout | Each disk is a bounded extent pool with resident descriptors, deterministic geometry, v3 CRC-protected fixed records, and no per-object files. Format/geometry mismatch fails closed. |
| Crash epoch | Startup persists a dirty epoch before serving. Graceful shutdown syncs payload/table then marks clean. An unclean epoch cold-resets metadata instead of serving possibly stale or torn bytes. |
| RAM/MR ownership | The fixed arena is aligned, prefaulted, bounded, and registered once on every selected rail. Arena slots remain pinned until the completion that fences NIC reads. Oversized allocations never borrow the arena MR. |
| I/O | Production payload I/O bypasses the page cache with `O_DIRECT`; bounded buffered fallback is observable. io_uring is build-gated and submits prepared reads or direct-write batches without weakening lease/commit semantics. |
| Operations | Existing `dfkv_*` metrics retain their meanings, including per-disk bytes/objects, tenant limits/usage/rejections, slab allocation/rebuild/epoch failures, direct-I/O fallbacks, io_uring activity, RAM health/backlog/pins, and RDMA registration/resources. |

## 3. Compatibility findings

### 3.1 Allocator and item lifetime: useful mechanisms, different contract

CacheLib handles are a real ownership primitive: a live handle prevents item
reclamation. Its DRAM allocator also exposes one slab memory range specifically
for pre-registration. These are the strongest reasons to reconsider CacheLib in
a future redesign; they are not empty extension points.

They still do not directly satisfy dfkv. dfkv's allocator manages media
locations and exposes slot pins to disk reads, asynchronous transport leases,
RAM flushes, and transfer completion. Its key index stores the 24-byte
`BlockKey` by value and its reverse eviction metadata is also value-semantic;
lookup, existing-key PUT, and REMOVE do not allocate or rehash. A CacheLib
backend would use CacheLib's item/key/index representation and handle state
instead. The adapter would have to prove the same remove/replace ordering and
translate every handle into the existing completion-fenced lease model.

That translation is feasible engineering, but it is a new backend, not a thin
allocator substitution. Running `SlabAllocator` beside CacheLib would retain two
indexes, two eviction policies, two pin models, and double accounting—the exact
complexity an allocator integration is meant to remove.

### 3.2 Disk and acknowledgement semantics: incompatible

Navy overlaps with dfkv at the device layer: it supports file or RAID devices,
tries `O_DIRECT`, and can choose io_uring or libaio with configurable queue
depth. Its BlockCache/BigHash policies and region writes are credible
alternatives to a custom flash cache.

The published HybridCache object lifecycle is the blocking mismatch:

1. a new item starts in DRAM;
2. it spills to NVM when evicted from DRAM;
3. an NVM hit is promoted back into DRAM before application access.

That is not dfkv's optional registered RAM tier over an authoritative disk
cache. A normal CacheLib insert can succeed while the object is only in DRAM,
whereas dfkv does not return PUT `kOk` until the disk result is known.
Likewise, the documented HybridCache read API returns a ready/not-ready handle
to a DRAM item; it does not implement `RangeDirectPrep` over a resident extent
into dfkv's caller-provided registered I/O buffer. Partial range reads, exact
full-value length reporting, and `CacheDirectBatch` commit results would all
need new code above or inside Navy.

Using Navy alone through its lower-level engine APIs would avoid the
DRAM-spill policy, but would also forgo most of the proposed CacheAllocator
integration and require dfkv to own scheduling, key index semantics, commit
ordering, leases, tenant metadata, and metrics. That is a replacement project
with a fork-sensitive internal API surface, not a justified optional adapter for
v2.0.0.

### 3.3 GPU/RDMA memory registration: promising DRAM seam, no end-to-end owner

`getSlabMemoryInfo()` means CacheLib DRAM is not inherently impossible to
register. A proof-of-concept could register that stable range on every selected
HCA and keep a CacheLib handle live until the signaled SEND completion. That
would have to be the real design.

No evaluated CacheLib API owns dfkv's `ibv_mr` objects, selected-rail PD
registrations, QP teardown obligations, receive-segment leases, or client GPU
MRs. Those remain dfkv transport responsibilities. In particular:

- every selected rail must register successfully or startup must fail;
- a handle must cover all RDMA WRITEs and the completion fence, including
  connection teardown;
- allocator slab release/rebalance must not invalidate a registered address;
- Navy promotion must not introduce an unregistered transient pointer;
- a Navy disk hit needs either a direct read into the registered transport
  buffer or a measured and explicitly accepted DRAM promotion/copy.

The existing `RamTier` already has this ownership chain and distinguishes arena
from dedicated oversized allocations. Replacing it without an end-to-end MR
owner would regress correctness even if ordinary CacheLib lookups worked.

### 3.4 Persistence and storage epochs: no in-place compatibility

DiskSlabStore format v3 stores tenant identity, exact payload length, slot
geometry, and a CRC in each fixed record. `slab_state` has a separate
CRC-protected clean/dirty epoch. Unclean startup destroys warmth deliberately;
format or geometry mismatch refuses startup.

CacheLib persistence has a different contract and different RAM/NVM format
epochs. Its documented recoverable path drains all cache access and calls
`shutdown()`, then uses `SharedMemAttach`; NVM can be dropped independently.
Those mechanisms may be robust for their intended lifecycle, but they cannot
read `slots.tbl`/`slab_state` and do not prove dfkv's dirty-epoch rule.

Therefore a future CacheLib backend must use a separate directory/device and
start cold. It must not reuse, convert in place, or add a compatibility reader
for v3. Selecting it must not change TCP/RDMA wire epoch or raw-value semantics.
Rollback must likewise be an explicit cold switch, not two implementations
mutating the same media.

### 3.5 Tenant accounting: not provided by HybridCache pools

CacheLib pools are attractive as allocation domains, but upstream documents
that pools partition only DRAM, not HybridCache/NVM. Mapping each dfkv tenant to
a pool therefore cannot enforce the current disk quota. It also would not by
itself provide dfkv's exact committed payload bytes across same-key replace,
per-item batch rejection, disk eviction, remove, failure rollback, and restart
rebuild.

An adapter could maintain a second tenant ledger and encode tenant identity into
its own persistent metadata. That duplicates a load-bearing part of
`DiskCacheGroup`/DiskSlabStore and requires observing every CacheLib/Navy
eviction and recovery transition without races. Until that is implemented and
stress-tested, the backend cannot return the current `kQuotaExceeded` result or
export truthful tenant metrics.

### 3.6 I/O and operational metrics: capability overlap, semantic gap

Navy's O_DIRECT attempt and async I/O configuration meet the basic device
capability requirement. They do not establish parity with dfkv's behavior:
resolved buffered fallback must be exposed, prepared read buffers must remain
owned through io_uring completion, and a write completion must map to the
per-item server acknowledgement. Navy region buffering and HybridCache
promotion also change latency, memory, and write-amplification behavior that
would need target-hardware measurement.

CacheLib exposes global, pool, allocator, and NVM statistics maps. dfkv's
Prometheus endpoint can consume such APIs, but names alone are not parity. A
production adapter would need a documented mapping for every existing metric
and new metrics for CacheLib/Navy queue depth, admission rejection, promotion,
region reclaim, checksum/recovery, and resolved I/O engine. The HybridCache
upstream documentation itself notes that uniform DRAM+NVM statistics semantics
are not available for all stats. Silently publishing CacheLib counters under
existing `dfkv_*` names would be misleading.

### 3.7 Dependency, build, and release footprint: disproportionate for v2.0.0

The current core is C++17. Its baseline link surface is Threads, `libdl`, and
`librt`; RDMA and io_uring add only explicit `libibverbs` and `liburing` gates.
The release package is intended to stay portable and self-contained.

At the pinned revision CacheLib requires C++20 and its supported build uses
`getdeps.py`. The primary dependencies include Folly and FBThrift, with a
transitive graph including Fizz, Wangle, glog, gflags, fmt, sparse-map, xxHash,
Boost, libevent, compression, crypto, unwind, and sodium libraries. CacheLib's
build guide also warns that it is tightly coupled to matching revisions of
frequently updated dependencies.

Adding this graph would require a separately maintained compiler baseline,
pinned dependency manifest, SBOM/license review, portable package strategy,
security update process, and CI/release matrix. It cannot be hidden behind a
header-only option, and it is not acceptable for the default binary to gain
those runtime dependencies accidentally.

## 4. Migration cost if the decision is reopened

A credible implementation is at least the following work, all before it can be
called supported:

1. **Backend semantics:** implement every `StoreEngine` operation, including
   partial ranges, direct caller buffers, batched per-item status, metadata-only
   lookup, exact remove ordering, health/readiness, and move-only read leases.
2. **Durable PUT:** define and prove the Navy persistence point that controls
   server `kOk`; preserve same-key leader/follower results and batch rollback.
3. **MR owner:** register CacheLib slab memory on all selected rails, bind handles
   to RDMA completion/teardown with RAII, and prove allocator maintenance cannot
   invalidate registered addresses. Keep client GPU and receive-segment
   ownership unchanged.
4. **Tenant ledger:** enforce exact per-node committed-byte quotas through
   replace, eviction, remove, failures, and recovery; do not equate DRAM pools
   with NVM tenant isolation.
5. **Persistence:** allocate separate media, define clean/dirty crash behavior,
   validate CacheLib and dfkv geometry/version, and fail closed. There is no v3
   in-place conversion.
6. **Metrics and operations:** preserve current metric meanings, add Navy
   operational counters, surface resolved direct/async I/O, and provide a
   readiness/failure/runbook contract.
7. **Build and packaging:** add a C++20-compatible build, pinned CacheLib
   dependency set, package/SBOM/license/security ownership, and hermetic CI for
   TCP, RDMA, io_uring, sanitizers, restart, and real NVMe/RDMA hardware.
8. **Rollout:** use a distinct backend name and distinct cache directory/device;
   canary from cold, compare hit/latency/write amplification/RSS, and roll back
   cold. Never let two engines share storage.

This cost is justified only by measured gains that the current implementation
cannot reasonably deliver, not by reducing the local source line count.

## 5. Reopening criteria

Reopen F25 only when **all** of these gates have owners and measurable acceptance
criteria:

1. A production-shaped trace and target NVMe hardware show a material, repeatable
   deficit in DiskSlabStore/RamTier after current tuning. CacheBench may be used
   for exploration, but the final A/B must exercise dfkv's real block-size mix,
   same-key concurrency, partial GETs, tenant distribution, RDMA fan-out, and
   restart behavior.
2. The team accepts C++20 and the complete pinned dependency/package/security
   footprint, including release artifacts for supported distributions.
3. A prototype implements the full `StoreEngine` contract—no unsupported
   methods, synchronous wrappers that change acknowledgement, or copy-only
   fallback presented as zero-copy.
4. Multi-rail MR registration and handle-to-completion ownership pass teardown,
   eviction, replace, slab maintenance, and error-injection tests; direct disk
   reads into registered transport buffers are demonstrated or the measured copy
   cost is explicitly accepted.
5. Crash/restart tests prove no stale resurrection, no torn value, fail-closed
   version/geometry handling, and a cold reset after unclean shutdown consistent
   with the selected v2 policy.
6. Exact tenant quota and accounting tests pass across scalar/batch PUT,
   replacement, concurrent admission, eviction, remove, I/O failure, and restart.
7. The Prometheus and readiness contracts have semantic parity and the new Navy
   queues/recovery/admission behavior is operationally observable.
8. A cold canary/rollback migration plan demonstrates a worthwhile end-to-end
   gain in throughput or tail latency without unacceptable RSS, write
   amplification, startup time, or deployment complexity.

Passing only the device-level O_DIRECT/io_uring comparison, or only a CacheBench
microbenchmark, is insufficient.

## 6. Why there is deliberately no empty integration hook

A rejected dependency does not justify speculative code. An unused
`DFKV_WITH_CACHELIB` option, empty `StoreEngine` subclass, or runtime
`--store-engine=cachelib` selector would be harmful because it would:

- create a configuration that compiles or starts without implementing the
  durable PUT, range, lease, tenant, persistence, and metric contracts;
- imply support to packagers and operators despite having no working backend;
- freeze a premature adapter ABI before the MR and persistence owners are
  designed;
- add conditional build paths and CI obligations with no executable behavior;
- encourage a no-op/error fallback, which can turn an operator typo into cache
  loss or persistent unready nodes.

If a future evaluation passes the reopening gates, the first code change must be
a **real** build-gated adapter: enabling it must require the pinned CacheLib
headers and libraries and fail CMake configuration when they are absent. There
must never be a no-op substitute or automatic fallback to another engine. Until
then, keeping CMake and the backend selector unchanged is the only truthful
interface.
