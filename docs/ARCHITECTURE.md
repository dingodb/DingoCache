# dfkv Architecture

A tour of how dfkv is put together: the layers a request flows through, the
pluggable storage engines, the optional RAM hot tier, and the wire protocol.
For deployment see [DEPLOY.md](DEPLOY.md); for metrics see
[METRICS.md](METRICS.md); for engine connectors and the **client-side**
env/config reference see [CONNECTORS.md](CONNECTORS.md).

> **Client vs server config:** the storage engine (§5) and RAM tier (§6) are
> **server-side only** — a client needs no config for them and can't tell which a
> node runs. No client-side config is required for any v1.7.x feature (the
> 96-bit key, §3, is automatic). See [CONNECTORS.md](CONNECTORS.md) §1.

---

## 1. Layers

```
        LLM inference engine (SGLang HiCache / LMCache / vLLM)
                              │  thin adapter (ctypes / KVConnector)
                              ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  libdfkv client  — key→96-bit id, Ketama route, header guard  │
   │                    Put / Get / Exist / Remove + batch/SG APIs  │
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
1. Client computes the block identity and wraps the value with a 48-byte
   `ValueHeader` (model / page-size / dtype / layer geometry guard), then routes
   to one node by Ketama over the key.
2. Server `ProcessRequest(kCache)` / `CacheDirect` (RDMA aligned): if the RAM
   tier is on, the value is **written through** to a RAM slot (synchronously
   visible) and flushed to disk in the background; otherwise it goes straight to
   the disk `StoreEngine`.

**GET** `client.Get(key, dst)`
1. Route + request. Server checks the RAM tier first (served from the arena, no
   disk); on a miss it reads from the disk engine. On RDMA the payload is
   scatter-sent zero-copy (from the arena MR, or straight from the O_DIRECT read
   buffer) into the caller's registered buffer.
2. The client verifies the `ValueHeader` geometry — a mismatch (wrong model /
   page-size / dtype / layer, or a stale/corrupt block) is treated as a **miss**
   (recompute), never silent wrong bytes.

---

## 3. Block identity (96-bit)

A block's identity is `BlockKey{id, index, size}`:

- `id` = MD5[0..8) of the key string (little-endian u64) — the ring routing hash
  input and the primary identity.
- `index` = MD5[8..12) (little-endian u32) — extends identity to **96 bits**.
- `size` = a fixed identity constant (never the payload length, so Put/Get/Exist
  build the same `Filename()`).

Why 96 bits: at the ~1e9 lifetime-write scale of a 5 TiB × N-node ring, a 64-bit
id alone has a few-percent birthday-collision probability — and a collision is
**not a clean miss but a silent cross-key read** (the geometry header only checks
model/page/dtype/layer, so two same-model pages that collide pass validation).
Filling the previously-always-zero `index` with more hash bits cuts collision
probability by 2³² with no wire or storage change; `id` is unchanged so routing
is unaffected.

---

## 4. Wire protocol

`src/transport/wire.h` defines two deliberately distinct transports:

| path | request/control frame | response/control frame | payload |
|---|---:|---:|---|
| TCP and RDMA fallback v1 | 42-byte fixed prefix + inline payload | 10-byte prefix + inline payload | two-sided SEND/RECV (RDMA) or stream (TCP) |
| negotiated RDMA transport v2 | 42-byte prefix; GET adds 8 bytes + 16 bytes per destination | 10-byte prefix + small header | PUT `WRITE_WITH_IMM` into a leased server slot; GET server `RDMA_WRITE` into client MRs |

Both prefixes start with an explicit 1-byte protocol version. RDMA v2 is
negotiated during the TCP bootstrap before QPs exchange frames; an unknown or
unexpected version fails fast instead of being mis-parsed. Replies remain strict
FIFO on each RC QP. The legacy `kMembers` discovery response deliberately uses a
dedicated v1 control QP because a membership list may exceed the 4-KiB v2
control frame; production MDS discovery is unaffected.

> Historical name collision: v1.7.0/v1.7.1 shipped an unrelated experimental
> TCP "wire v2" (`DFKV_WIRE_VERSION=2`) that only echoed a request `seq`. It was
> removed in v1.7.2 and that environment variable remains a no-op. The current
> **RDMA transport v2** is a different, bootstrap-negotiated one-sided data plane;
> it does not resurrect the retired TCP sequence protocol.

### 4.1 RDMA v2 resource and fallback invariants

- `RdmaServer::Start` allocates one aligned `RecvSegment` per process, then
  registers it once on every anchored rail's shared PD. Every endpoint on that
  rail reuses the same MR; endpoint close drops only its shared-device reference.
- DCP2 advertises the client's exact `DFKV_RDMA_MAX_BLOCK_BYTES` (or the explicit
  legacy DCP1 cap). The server validates it against `--max-msg`, negotiates
  `qd=min(client depth, server depth)`, and leases `qd` contiguous slots. The
  connection keeps that lease—including while idle in the client pool—until QP
  teardown or idle reclaim.
- Slot geometry is `align4K(4096 + ValueHeader::kSize + max(declared_block, 4096))`.
  Segment sizing must cover every live and pooled data/control QP across all
  client ranks/processes, not only currently in-flight operations. Exhaustion
  rejects that v2 bootstrap cleanly; the client reconnects the operation over v1.
- Client host/device pools are registered once per rail at declaration time.
  Buffers outside those pools use operation-scoped MRs and are deregistered only
  after the corresponding completion; no cached MR may outlive caller-owned
  `std::string` or transient buffers.
- With an empty device selector each side chooses its own first ACTIVE local HCA,
  so host-local names may differ. An explicit comma list is a cross-host
  whitelist and therefore requires the same names/fabric on both peers.

`dfkv_rdma_v2_ready`, receive-segment total/free bytes, registered-rail count,
v1/v2 opened connections, and v2 PUT/GET WRITE counters make each invariant
observable. See [CONNECTORS.md](CONNECTORS.md) §1.2.1 for capacity arithmetic and
[DEPLOY.md](DEPLOY.md) §3 for rollout settings.

---

## 5. Storage engine layer

`DiskCacheGroup` routes a block to one disk (Ketama over `BlockKey.Filename()`)
and holds one `StoreEngine` per disk. The backend is selected by
`--store-engine` / `DFKV_STORE_ENGINE` (default `file`):

```
StoreEngine (interface): Cache / CacheDirect / Range / RangeInto /
                         RangeDirect / RangeDirectPrep / IsCached / Remove + stats
   ├── file  →  KVStore        (one file per block; the original engine)
   └── slab  →  DiskSlabStore  (extent files + slots.tbl; the rework)
```

### 5a. `file` engine — KVStore (default)

One file per block under `blocks/<bucket>/…`, O_DIRECT read/write, per-shard
`shared_mutex` index + CLOCK-second-chance LRU. Battle-tested; the default. Its
"one file per block" geometry is the root of several operational hazards (tmp
leak, ENOSPC dead-end, unbounded inode growth, lock-held unlink, open-per-GET),
which the slab engine removes.

### 5b. `slab` engine — DiskSlabStore (opt-in)

A fixed pool of pre-allocated **extent files**, each bound on demand to one
**size class** and carved into uniform slots by the media-agnostic
`SlabAllocator`:

- **SlabAllocator** — owns slot *layout* (which extent, which offset), not bytes.
  Size-class slab + per-class CLOCK eviction + pin refcount. `slot_granularity`
  bounds the per-extent slot count. Reused by the RAM tier (§6).
- **DiskSlabStore** — maps a key's slot to an extent-file offset (buffered I/O;
  extent fds stay resident, no open-per-GET) and records `{key → slot}` in a
  compact 64-byte-per-slot **`slots.tbl`**. On restart the index **rebuilds from
  `slots.tbl`**, keeping cache warmth across a rolling upgrade without per-block
  file churn.

**Crash safety**: every `slots.tbl` record is CRC32-checked, so a torn record
reads as free (its key becomes a clean miss = recompute, never corruption). A
meta magic mismatch is refused; an extent/granularity config mismatch re-inits
fresh. Eviction leaves a freed slot's record as-is (a slot's record always
reflects its last occupant), so an unreused evicted slot merely "resurrects" its
still-valid content-addressed key on restart.

**When to use slab**: it is a strategic replacement for the file engine on nodes
where the per-block-file hazards bite. It is **off by default**; switching a node
needs a clean-disk cold start (a separate ops migration).

---

## 6. RAM hot tier (P3, opt-in)

A COLD dfkv load is disk-bound (~480 MB/s O_DIRECT) and dominates PD-decode TTFT.
The RAM tier fronts the disk with a pre-registered RAM arena so a PD-warm GET is
served straight from RAM over RDMA — no open, no pread, no disk. Enabled with
`DFKV_RAM_TIER=1` (`DFKV_RAM_TIER_BYTES` sizes the arena; default off).

**Write-through**: `Put` copies the value into a RAM slot (via a `SlabAllocator`
over the arena), makes it **synchronously visible** (read-after-write), and
enqueues a background flush to the disk engine.

**State machine = allocator pin refcount.** The slot lifecycle maps directly onto
the allocator's pin count — this is why the allocator was built media-agnostic:

| state | pin holders | evictable? |
|-------|-------------|------------|
| RAM_ONLY (flush pending) | flush-pin | no |
| in-flight (RDMA transfer reading arena) | transfer-pin | no |
| DURABLE & idle | none (refcount 0) | **yes** |

- **flush-pin** — taken on `Put`, released when the async flush reaches disk.
- **transfer-pin** — taken on `GetPrep`, released only after the completion that
  fences the payload transfer (`IBV_WC_SEND`).

**RDMA zero-copy serve**: the arena is registered once on each selected device's
shared PD; all endpoint QPs on that rail reuse the pool MR. On a v2 RAM hit the
server RDMA-WRITEs arena bytes directly into the client's advertised targets,
then sends the small status response. Its signaled SEND completion fences the
preceding WRITEs and releases the transfer-pin. A v1 fallback QP instead
scatter-SENDs `[response | arena bytes]` and releases the same token on that
SEND completion. Connection teardown releases any outstanding pins, so a dead
QP cannot leak an allocator pin. With RAM tier off none of this path is wired.

**Backpressure & durability**: if the arena fills with non-evictable
(flush-pending / in-flight) slots, `Put` declines and the caller takes the normal
synchronous disk write — never blocks, never breaks read-after-write. Only a
DURABLE slot may be evicted, so an eviction never loses the sole copy. RAM is
volatile but dfkv is a cache (a miss = recompute), so a crash costs only a
recompute of the unflushed tail.

Observability: `dfkv_ram_hit/miss/put/put_bypass/flushed/flush_dropped/
evictions_total` + `ram_objects` / `ram_flush_backlog` (emitted only when
enabled) — see [METRICS.md](METRICS.md).

---

## 7. Configuration matrix (feature defaults)

Storage engines remain opt-in. RDMA itself still requires `DFKV_RDMA=1`; once
selected, peers negotiate mixed v2 by default and fall back per connection.

| Feature | Enable with | Default | Notes |
|---------|-------------|---------|-------|
| slab storage engine | `--store-engine=slab` / `DFKV_STORE_ENGINE=slab` | `file` | needs clean-disk cold start |
| RAM hot tier | `DFKV_RAM_TIER=1` (+ `DFKV_RAM_TIER_BYTES`) | off | write-through + RDMA zero-copy GET |
| RAM tier lock shards | `--ram-tier-shards` / `DFKV_RAM_TIER_SHARDS` | 8 | 1-64; per-shard lock is the >8-connection concurrency ceiling; auto-halved while a shard would hold <32 extents |
| RDMA transport | build `-DDFKV_WITH_RDMA=ON`, `DFKV_RDMA=1` | TCP | active-HCA discovery; `DFKV_RDMA_DEV` is an optional whitelist |
| RDMA mixed v2 | `DFKV_RDMA_PROTOCOL=auto-v2` | auto-v2 with per-connection v1 fallback | 4-KiB control buffers + shared registered receive segment; server rollback `DFKV_RDMA_SERVER_PROTOCOL=1` |
| io_uring async GET | build `-DDFKV_WITH_URING`; `DFKV_SERVER_URING=0` disables | on when built, unavailable otherwise | disk-read path; applies to RDMA v1/v2 |

---

## 8. Source map

```
src/common/     ValueHeader, BlockKey, Status — portable shared types
src/transport/  wire.h (v1/v2 frames) · rdma_protocol / rdma_recv_segment ·
                rdma_topology · tcp_transport · rdma_transport / rdma_verbs
src/cache/      store_engine.h (interface) · kv_store (file) ·
                slab_allocator + disk_slab_store (slab) · ram_tier (RAM hot tier) ·
                disk_cache_group (per-disk routing) · kv_node_server · rdma_server ·
                dfkv_server_main
src/client/     kv_client + C ABI · key_map (96-bit identity)
src/mds/        membership service + dfkv_mds
src/tools/      dfkvctl / dfkv_smoke / dfkv_bench
```
