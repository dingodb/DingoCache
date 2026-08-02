# dfkv Architecture

A tour of how dfkv is put together: the layers a request flows through, the
pluggable storage engines, the optional RAM hot tier, and the wire protocol.
For deployment see [DEPLOY.md](DEPLOY.md); for metrics see
[METRICS.md](METRICS.md); for engine connectors and the **client-side**
env/config reference see [CONNECTORS.md](CONNECTORS.md).

> **Client vs server config:** the storage engine (§5) and RAM tier (§6) are
> **server-side only**. Clients must supply an explicit namespace and object key;
> the supported connectors derive both from runtime model/layout metadata. See
> [CONNECTORS.md](CONNECTORS.md) §1.

---

## 1. Layers

```
        LLM inference engine (SGLang HiCache / LMCache / vLLM)
                              │  thin adapter (ctypes / KVConnector)
                              ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  libdfkv client — namespace + object key→128-bit id, Ketama route │
   │                   opaque raw Put/Get/Exist/Remove + batch/SG APIs │
   └───────────────┬──────────────────────────────┬───────────────┘
        MDS discovery (etcd epoch)          transport: TCP or RDMA (RC)
                    │                               │  versioned wire frame
                    ▼                               ▼
   ┌──────────────────────┐        ┌──────────────────────────────────────┐
   │ dfkv_mds (+ etcd)    │        │  dfkv_server (cache node)             │
   │ membership directory │        │  ┌────────────────────────────────┐  │
   └──────────────────────┘        │  │ RamTier (opt) — write-through   │  │
                                    │  │  RAM arena, RDMA zero-copy GET  │  │
                                    │  └───────────────┬────────────────┘  │
                                    │      miss ▼      │ async flush        │
                                    │  ┌──────────────────────────────────┐│
                                    │  │ DiskCacheGroup — Ketama over N    ││
                                    │  │  disks, each a StoreEngine:       ││
                                    │  │   file (KVStore)  |  slab         ││
                                    │  └──────────────────────────────────┘│
                                    └──────────────────────────────────────┘
```

Distribution is **client-side consistent hashing**; there is no replication
(regenerable KV → a node loss is a miss → recompute). Membership is dynamic via
the MDS tier — see the README for the two-layer offline detection.

---

## 2. Request data path

**PUT** `client.Put(key, value)`
1. The connector supplies the handle's namespace and a canonical object key.
   The client derives the native block identity, routes it by Ketama, and sends
   the caller's bytes unchanged.
2. Server `ProcessRequest(kCache)` / `CacheDirect` stores the exact raw bytes and
   their length. If the RAM tier is on, admission makes the value visible in RAM,
   but the request does not return `kOk` until its disk flush commits. An
   overlapping same-key PUT joins that commit result rather than reporting
   uncommitted success. Genuine RAM capacity backpressure uses the synchronous
   disk path.

**GET** `client.Get(key, dst)`
1. The same namespace and object key select the same block. The server checks
   RAM first, then disk; RDMA writes the raw payload directly into the caller's
   registered buffer.
2. The stored length is returned separately. `GetAuto` accepts any destination
   capacity large enough for that length; dfkv does not prepend or validate a
   value envelope.
3. Every backend has one range contract: `length=0` means the complete
   remainder; `offset == stored_length` succeeds with zero bytes;
   `offset > stored_length` is `kInvalid`; an oversized length is clamped.
   `value_len` always reports the exact full stored payload on success.

---

## 3. Canonical identity and raw-value contract

The native C ABI has one versioned constructor:
`dfkv_open_v2(const dfkv_client_options_v2*)`. The size-delimited descriptor
owns static members or MDS discovery/registration settings, immutable binary
namespace bytes, and batch concurrency. There are no post-open membership
mutators and no model-hash or geometry arguments.

Every scalar and batch/SG operation passes object keys as `(const void*, uint64_t)`
binary spans (batch APIs add a parallel `key_lens[]` array); embedded NUL bytes
are identity bytes, not terminators. Null, zero-length, or unrepresentable key
spans fail closed. PUT values must also be non-empty: scalar, batch, SG, TCP,
RDMA, RAM-tier, and persistent-store admission all reject zero-byte objects.
Every exported `extern C` function is a no-throw boundary: output slots are
initialized to failure and C++ exceptions translate to the documented result.
MDS group and registration client IDs are validated synchronously by
`dfkv_open_v2` before either background worker starts.

Connectors construct a binary namespace in one of two disjoint forms:

- automatic: the cross-language `NamespaceDescriptor` encoding beginning with
  `b"DFKVNS\x00\x02"`. Length-framed fields bind tenant, exact model ID and
  revision, connector raw-layout ID, dtype, layout fingerprint, block-token and
  layer geometry, group, and replicated topology sizes. Connector-specific
  shape/stride fields feed the deterministic 64-bit layout fingerprint.
- explicit `key_namespace` override:
  `b"DFKVNS\x00\xff" + u32le(len(override_utf8)) + override_utf8`.

The automatic layout IDs are `sglang-hicache/raw-v1`, `vllm/raw-v1`, and
`lmcache/raw-v1`. Model/revision strings are preserved verbatim. The disjoint
magic bytes prevent an explicit override from accidentally aliasing an
automatic namespace. Python and C++ serializers share a checked golden vector.

Every connector renders an object key as:

```
dfkv/pool/v2|pool=...|hash=...|dp=S:R|tp=S:R|pcp=S:R|dcp=S:R|pp=S:R|group=N|component=...
```

Scatter-grouped objects append `|sg=W:G`. Both size and rank are identity;
`R=-1` means replicated. Object identity is SHA-256 over the length-framed
namespace and object-key bytes, truncated to 128 bits. `BlockKey` also carries a
stable 64-bit tenant hash. For canonical `DFKVNS\0\2`, tenant identity is the
first length-framed field; malformed or other namespace forms use the complete
namespace bytes. The hash is the first eight SHA-256 bytes (big-endian) of
`"DFKVTENANT1" || u64le(identity_length) || identity`. C++ and Python share
golden vectors.

Identity ownership is strict:

- **namespace:** exact model/deployment identity plus a deployment-wide raw
  layout/schema version; connector defaults include their raw-layout ID
- **object key:** pool, full content hash, parallel coordinates, cache group,
  component, and optional scatter width/group
- **control metadata:** members, MDS endpoints/group, transport, library path,
  telemetry, and capacity knobs; none changes cache identity
- **raw payload:** exactly the caller-provided bytes; dfkv stores no envelope,
  geometry, dtype, checksum, or model tag in the value

Different namespace or key bytes produce a cold miss. Reusing the same
namespace and key for a different byte layout is an operator error, not a
guarded miss: encode every identity-bearing geometry/layout change in the
namespace or object key, and coordinate an explicit override only among
byte-compatible producers and consumers. This is a clean cutover: native
clients do not read old keys or dual-write the retired identity/value format.

---

## 4. Wire protocol

`src/transport/wire.h` defines two deliberately distinct transports:

| path | request/control frame | response/control frame | payload |
|---|---:|---:|---|
| native TCP | 50-byte fixed prefix + inline payload | 18-byte stored-length prefix + inline payload | versioned stream |
| native RDMA v2 | 50-byte prefix; GET adds 4 bytes + 16 bytes per destination | 18-byte stored-length prefix; `Members` allows at most 32 KiB data | PUT `WRITE_WITH_IMM` into a leased server slot; GET server `RDMA_WRITE` into client MRs |

Both prefixes start with an explicit 1-byte protocol epoch: TCP accepts epoch 6
and native RDMA v2 accepts epoch 7. An unknown or unexpected epoch fails fast
instead of being mis-parsed. RDMA v2 is negotiated during the TCP bootstrap
before QPs exchange frames. Replies remain strict FIFO on each RC QP. RDMA
control buffers have an explicit `18-byte prefix + 32-KiB response` capacity;
`kMembers` uses the control lane and an oversized declaration is rejected before
response copy/post rather than truncated or allowed to resize a receive buffer.
Production discovery uses MDS.


### 4.1 RDMA v2 resource and failure invariants

- `RdmaServer::Start` allocates one aligned `RecvSegment` per process, then
  registers it once on every anchored rail's shared PD. Every endpoint on that
  rail reuses the same MR; endpoint close drops only its shared-device reference.
- DCP2 advertises the client's exact `DFKV_RDMA_MAX_BLOCK_BYTES`. The server
  validates it against `--max-msg`, negotiates `qd=min(client depth, server depth)`,
  and leases `qd` contiguous slots. The connection keeps that lease—including
  while idle in the client pool—until QP teardown or idle reclaim.
- A data slot is `align4K(4096 + declared_block)`: 4 KiB of wire-prefix space
  plus the declared maximum raw payload, with no value-envelope allowance.
  Segment sizing must cover every live and pooled data/control QP across all
  client ranks/processes, not only currently in-flight operations. Exhaustion
  rejects the connection; it is never reinterpreted as another protocol.
- Client host/device pools are registered once per rail at declaration time.
  Re-declaring the same base with a larger size registers the larger extent; the
  registration call returns false/nonzero unless the full range is ready. Buffers
  outside those pools use operation-scoped MRs and are deregistered only after
  the corresponding completion; no cached MR may outlive caller-owned
  `std::string`, `CacheSrc`, destination, or SG buffers.
- Every multi-completion wait uses one absolute window deadline. A partial CQ
  drain reduces the remaining budget and cannot restart it.
  `DFKV_RDMA_BATCH_OP_TIMEOUT_MS` applies to every multi-item method, including
  Cache/Range/Exist batches, zero-copy variants, and both SG variants; unset
  follows `DFKV_RDMA_OP_TIMEOUT_MS`.
- With an empty device selector each side chooses its own first ACTIVE local HCA,
  so host-local names may differ. An explicit comma list is a cross-host
  whitelist and therefore requires the same names/fabric on both peers.
- Rail health is host-local evidence only. Local device open, verbs/QP
  transition, post, or CQ failures increment the selected rail's failure streak
  and may quarantine that rail. TCP bootstrap, unreachable/incompatible peer,
  and malformed peer frames return the admission credit and feed client
  `PeerHealth`; they do not penalize an HCA shared with unrelated nodes.
- `DFKV_RDMA_NUMA=1` derives the calling thread's NUMA node for every admission.
  If any enabled configured rail has matching discovery metadata, only that
  stable-index subset competes for credits. Unknown caller topology or no local
  rail falls back to all enabled rails; an explicit device whitelist remains the
  authoritative candidate set.

`dfkv_rdma_v2_ready`, receive-segment total/free bytes, registered-rail count,
opened connections, v2 PUT/GET WRITE counters, client rail-vs-endpoint failure
and quarantine counters, and bounded NUMA-fallback counters make these
invariants observable. See [CONNECTORS.md](CONNECTORS.md) §1.2.1 for capacity
arithmetic and [DEPLOY.md](DEPLOY.md) §3 for rollout settings.

---

## 5. Storage engine layer

`DiskCacheGroup` routes each block to one NVMe disk (Ketama over the full
`BlockKey.Filename()`) and owns one `StoreEngine` per disk. With no option and
no `DFKV_STORE_ENGINE`, every `DiskCacheGroup` user (including `dfkv_server`)
selects `slab`. `--store-engine=file` / `DFKV_STORE_ENGINE=file` selects the
explicit diagnostic fallback; invalid slab capacity or persistent geometry
fails startup and never retries with `file`:

```
StoreEngine: Cache / CacheDirect[Batch] / Range[*] / Lookup / Remove + stats
   ├── slab  →  DiskSlabStore  (production: extent pool + sparse slots.tbl)
   └── file  →  KVStore        (diagnostic fallback: one file per block)
```

### 5a. `slab` engine — production default

Each configured disk is a fixed pool of pre-allocated **extent files**. A
media-agnostic `SlabAllocator` binds extents to a deterministic startup-built
size-class lattice and carves them into uniform slots:

- **Dense resident metadata:** one reusable `SlotMeta` per live value; class and
  extent lists store integer indices, and the only owned key string lives in the
  key→slot map. There is no per-class key mirror or stale slot-state scan.
- **Variable-size workloads:** aligned geometric classes keep bounded internal
  waste across partial chunks and mixed model layouts. Placement is independent
  of request/restore order. Rebalancing uses useful bytes and decayed read heat,
  not object count alone.
- **Concurrent I/O:** payload reads/writes run outside metadata locks while
  slot pins and deferred removal prevent reuse. Prepared disk reads return one
  move-only RAII lease; async completion, fallback, error, and teardown release
  descriptor + pin exactly once. Batched flushes group by disk and run
  participating NVMe devices concurrently.
- **Commit and removal ordering:** a reserved same-key PUT has one leader;
  scalar and batch followers wait for its payload+record result. REMOVE hides
  the key immediately and fences in-flight I/O, so a deferred write cannot
  republish it after REMOVE returns.
- **Restart:** each published slot has a CRC-protected 64-byte tenant-scoped
  fixed-offset record in sparse `slots.tbl`. Rebuild visits allocated sparse
  ranges with `SEEK_DATA`/`SEEK_HOLE` when available (bounded chunked fallback
  otherwise), parses chunks, validates all records, restores allocator state in
  one bulk transaction, and rebuilds per-tenant committed-byte usage—no
  per-object open and no per-record syscall.
- **Crash safety:** `slab_state` marks the running epoch dirty before traffic and
  clean only after payload/table sync on graceful shutdown. An unclean epoch is
  cold-reset rather than risking stale resurrection. Torn/invalid records are
  rejected and cleared durably; metadata I/O failure fails the store closed.
  Format/geometry mismatches are refused without rewriting existing data.

Payload I/O uses resident extent descriptors and O_DIRECT by default; a
filesystem that cannot support it reports and uses the bounded buffered
fallback. The v3 tenant-scoped on-disk format is intentionally incompatible
with earlier slab metadata, so upgrading a slab node requires a clean cache
directory.

### 5b. `file` engine — explicit fallback

`KVStore` keeps one file per block under `blocks/<bucket>/…`, with O_DIRECT
read/write and a sharded CLOCK index. It remains useful for diagnosis or
rollback, but is not the daemon's production default: inode growth,
open-per-GET, tmp cleanup, and unlink/ENOSPC behavior make it unsuitable as the
main large-capacity path.
Its v3 path is strict lower-case hex:
`blocks/<tenant-prefix>/<tenant-prefix>/<16-hex-tenant><32-hex-object>`;
startup rejects old or malformed cache filenames rather than decoding them.
Rename-to-trash is part of the file engine's logical commit: rename failure
returns `kIOError` and leaves the index, global bytes, and tenant bytes intact.
Both `Cache` and `CacheDirect` make exactly one bounded force-evict-and-retry
attempt after `ENOSPC`.

### 5c. Per-node tenant capacity admission

`DiskCacheGroup` optionally loads immutable startup quotas from
`DFKV_TENANT_QUOTAS_FILE`, with `DFKV_TENANT_DEFAULT_QUOTA_BYTES` as the
unlisted-tenant limit (`0`/unset means unlimited). Quotas are deliberately
**per cache node**, not a synchronous cluster-global reservation protocol.

Admission uses 256 fixed tenant stripes. PUT/REMOVE for the same stripe
serialize; GET remains lock-free relative to quota. A batch acquires unique
stripes in ascending order, charges only new keys, and may reject one tenant
with `kQuotaExceeded` while admitting unrelated tenants. Locks remain held
through storage writes, so concurrent single/batch requests cannot oversubscribe.
Both engines account committed payload bytes through publish, failure rollback,
eviction, removal, and restart rebuild. A limited tenant bypasses the
asynchronous RAM write-back admission path so the PUT result reflects durable
quota admission immediately.

---

## 6. RAM hot tier (P3, opt-in)

A COLD dfkv load is disk-bound (~480 MB/s O_DIRECT) and dominates PD-decode TTFT.
The RAM tier fronts the disk with a pre-registered RAM arena so a PD-warm GET is
served straight from RAM over RDMA — no open, no pread, no disk. Enabled with
`DFKV_RAM_TIER=1` (`DFKV_RAM_TIER_BYTES` sizes the arena; default off).

**Write-through with durable acknowledgement**: admission copies the value into
a RAM slot and makes it visible to concurrent reads, then enqueues its disk
flush. The server PUT waits for that flush result before returning `kOk`.
Overlapping same-key callers share the leader's exact success/failure.

**State machine = allocator pin refcount.** The slot lifecycle maps directly onto
the allocator's pin count — this is why the allocator was built media-agnostic:

| state | pin holders | evictable? |
|-------|-------------|------------|
| RAM_ONLY (flush pending) | flush-pin | no |
| in-flight (RDMA transfer reading arena) | transfer-pin | no |
| DURABLE & idle | none (refcount 0) | **yes** |

- **flush-pin** — taken on admission, released when the flush reaches disk or is
  canceled.
- **transfer-pin** — taken on `GetPrep`, released only after the completion that
  fences the payload transfer (`IBV_WC_SEND`). REMOVE hides the entry
  immediately but defers physical reuse until this final pin releases.

**RDMA zero-copy serve**: the arena is registered once on each selected device's
shared PD; all endpoint QPs on that rail reuse the pool MR. On a v2 RAM hit the
server RDMA-WRITEs arena bytes directly into the client's advertised targets,
then sends the small status response. Its signaled SEND completion fences the
preceding WRITEs and releases the transfer-pin. Connection teardown releases
any outstanding pins, so a dead QP cannot leak an allocator pin. With RAM tier
off none of this path is wired.

Values no larger than one RAM extent live inside that registered arena and keep
the zero-copy path. Larger accepted values use a dedicated aligned allocation
charged to the same hard byte budget; they retain the same flush/durable/send-pin
lifecycle but copy through the connection's bounded registered buffer because
their address is not part of the arena MR. A large pointer is never paired with
the arena's memory registration.
The fixed arena and dedicated allocations partition `DFKV_RAM_TIER_BYTES`;
their resident sum cannot exceed that limit. The automatic dedicated reserve is
two extents. `DFKV_RAM_TIER_LARGE_RESERVE_BYTES=0` gives all bytes to the arena;
set an explicit reserve when values larger than an extent are expected.

**Backpressure, durability, and removal**: if the arena fills with
non-evictable slots, admission declines and the caller takes one synchronous
disk path. A queued REMOVE cancels the queue item; an active flush is fenced and
then compensated on disk, so no later worker can resurrect the key. Exhausting
flush retries is terminal RAM/store health and removes readiness. Only a
DURABLE slot may be capacity-evicted. Requested arena allocation or an
unsupported NUMA mode rejects startup instead of silently running disk-only.

Observability: `dfkv_ram_hit/miss/put/put_bypass/flushed/flush_dropped/
evictions_total`, `dfkv_ram_healthy`, actual `dfkv_ram_flush_threads`,
object/backlog gauges, and explicit total/arena/dedicated budget/usage gauges
(emitted only when enabled) — see [METRICS.md](METRICS.md).

---

## 7. Configuration matrix (feature defaults)

The production disk path is slab by default; RAM and RDMA remain explicit.
RDMA requires `DFKV_RDMA=1`; selected peers must negotiate the v2 one-sided
data plane or the connection fails.

| Feature | Enable with | Default | Notes |
|---------|-------------|---------|-------|
| disk storage engine | `--store-engine=slab|file` / `DFKV_STORE_ENGINE` | `slab` (all server/store construction paths) | option > env > default; v3 slab needs a clean cache directory; invalid capacity, format, or geometry fails closed without a file fallback |
| RAM hot tier | `DFKV_RAM_TIER=1` (+ `DFKV_RAM_TIER_BYTES`) | off | durable-acknowledged write-through + RDMA zero-copy GET; requested allocation failure rejects startup |
| RAM NUMA policy | `--ram-tier-numa=interleave|off` / `DFKV_RAM_TIER_NUMA` | `interleave` | numeric node IDs are not accepted |
| RAM flush workers | `--ram-flush-threads` / `DFKV_RAM_FLUSH_THREADS` | 4× disk count, cap 16 | actual count is at least shard count and is reported after adjustment |
| RAM dedicated-value reserve | `DFKV_RAM_TIER_LARGE_RESERVE_BYTES` | two extents | partitions the hard total budget; `0` disables oversized residency |
| RAM tier lock shards | `--ram-tier-shards` / `DFKV_RAM_TIER_SHARDS` | 8 | 1-64; auto-halved while a shard would hold <32 extents |
| RDMA transport | build `-DDFKV_WITH_RDMA=ON`, `DFKV_RDMA=1` | TCP | active-HCA discovery; `DFKV_RDMA_DEV` is an optional whitelist |
| RDMA v2 | `DFKV_RDMA=1` | TCP when RDMA was not requested | bounded 32,786-byte control buffers (32-KiB Members data) + mandatory shared registered receive segment |
| io_uring async GET | build `-DDFKV_WITH_URING`; `DFKV_SERVER_URING=0` disables | on when built, unavailable otherwise | RDMA v2 disk-read path |

---

## 8. Source map

```
src/common/     BlockKey, namespace codec, Status — portable shared types
src/transport/  wire.h (current TCP/RDMA v2 frames) · rdma_protocol / rdma_recv_segment ·
                rdma_topology · tcp_transport · rdma_transport / rdma_verbs
src/cache/      store_engine.h (interface) · kv_store (file) ·
                slab_allocator + disk_slab_store (slab) · ram_tier (RAM hot tier) ·
                disk_cache_group (per-disk routing) · kv_node_server · rdma_server ·
                dfkv_server_main
src/client/     kv_client + C ABI · key_map (128-bit native identity)
src/mds/        membership service + dfkv_mds
src/tools/      dfkvctl / dfkv_smoke / dfkv_bench
```
