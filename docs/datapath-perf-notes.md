# dfkv RDMA datapath — perf notes & investigated-but-deferred items

## Current one-sided RDMA v2 payload plane

RDMA v2 uses small two-sided SEND/RECV messages only for requests, descriptors,
and status:

- PUT uses client `RDMA_WRITE_WITH_IMM` into a lease from the server's registered
  process-wide receive segment.
- GET sends the client `{addr,rkey,len}` targets, then the server uses
  `RDMA_WRITE` to place the value directly into the registered HiCache buffers.
- Cold values still require a request because their source is NVMe rather than a
  remotely addressable persistent MR; the server stages the disk read before its
  one-sided WRITE.

The retired two-sided payload protocol is not a fallback. Capability, QP,
receive-segment, or registration mismatch rejects the RDMA connection.

The negotiated pipeline depth contract is `qd = min(client depth, server depth)`
(`DFKV_RDMA_DEPTH` on the client is an upper bound). The server's resolved
runtime value can be audited live through the `qd=` column of `dfkvctl ring`
INFO — check it before reasoning about pipelining behavior.

## Retired two-sided plane (187d241, superseded by v2)

- RC **two-sided SEND/RECV**, two-end **zero-copy** (server reads disk straight into
  the registered send buffer; client SEND/RECV scatters straight into the caller's
  registered HiCache page). One round-trip per key.
- Batch ops (`CacheFrom` for PUT, `RangeInto` for GET) take N keys per call and
  **pipeline** them in windows of `ep.depth()` (`DFKV_RDMA_DEPTH`): up to `depth`
  requests in flight on one connection, hiding per-op latency. Each key is still its
  own wire SEND/RECV.

## #2 — coalescing contiguous pages into larger RDMA transfers: investigated, deferred

**Question:** should we merge multiple KV pages into fewer, larger RDMA transfers to
beat a small-object penalty?

**Finding — the premise mostly doesn't apply to MLA:**
- GLM-5.1 is MLA → **one object per page ≈ 2.6 MiB**. That is already in the bench's
  large-transfer sweet spot (3-node bench: 2 MiB→15.5, 4 MiB→23.5 GB/s GET). The
  per-page wire transfer is NOT small, so there is no small-object penalty to fix.
- Coalescing N keys into one wire message is **not free**: each key is a separate
  server-side raw-value object/file, so it needs a new multi-key wire op (server
  reads N files into N regions of one buffer → one big RDMA) plus
  larger registered direct buffers (`DFKV_RDMA_MAX_PAYLOAD_BYTES`, default 64 MiB,
  while ordinary control buffers stay small). Cost is real; gain on already-large, bandwidth-bound transfers is
  marginal.
- The genuine latency lever for multi-key batches is **pipelining (#1, `DFKV_RDMA_DEPTH`)**,
  which keeps `depth` pages in flight without any protocol change. That is implemented.
- (MHA models split a page into `_k`+`_v` = 2 sub-objects/page → there coalescing the
  pair could halve ops. Not relevant to MLA/GLM-5.1.)

**Decision:** do not add key-coalescing now. Use depth pipelining (#1). Revisit only
if a future model is MHA with small pages (small-object, message-rate-bound).

## #1 — write pipelining: measured on an 8x400G IB testbed (depth flat; multi-connection is the lever)

Empirical (8x400G IB testbed, 3-node, 1 MiB PUT, single writer thread, batch 64):

| knob | PUT GB/s |
|------|----------|
| `DFKV_RDMA_DEPTH` 1 / 8 / 16 (1 connection) | 2.50 / 2.55 / 2.50 — **flat** |
| `batch_concurrency` 1 / 8 / 16 | 1.34 / **3.35** / 3.28 — **2.5x at 8, saturates** |

- **Depth pipelining does NOT raise PUT.** The server's per-connection serve loop
  processes requests serially (each does the O_DIRECT disk write inline), so a
  client that pipelines `depth` PUTs just queues them at the server. Depth is kept
  as a knob defaulting to 4 (`rdma_depth` extra_config / `DFKV_RDMA_DEPTH`) — it can help on
  a network-latency-bound link — but on a disk-bound path it is flat.
- **Multi-connection fan-out is the real write lever**: splitting a batch across N
  connections hits N parallel server serve threads. `batch_concurrency` auto mode
  (the default, `0`) now floors at `min(max(groups, 8), 32)` — single-node rings
  (groups=1) get 8-way parallelism automatically, while multi-node rings (groups
  ≥ 8) keep their existing one-worker-per-group scaling. It saturates at ~8 on
  this hardware; the write ceiling is the single-rank r0 path + server
  serial-per-connection writes, mitigated by the fan-out.

- **GET is depth-flat too** (2026-06-20, `dfkv_bench --threads 1 --bc 1 --size 512KiB
  --count 2000`, single RC connection, io_uring server on a local SSD, 3 interleaved
  runs): GET held **1.22–1.25 GB/s with `DFKV_RDMA_DEPTH` 1 vs 32 — indistinguishable**
  (p50 call-lat ~0.41 ms both). Same reason as PUT: the server's per-connection serve
  loop is strictly in-order, so pipelining N GETs on one connection just queues them.
  Depth is a network-latency-hider, not a throughput knob; **do not expect a load
  speedup from raising `DFKV_RDMA_DEPTH` alone.** Caution when benchmarking on a shared
  node: the SSD's read latency varies several-fold with other tenants' I/O, which can
  masquerade as a depth effect — interleave depth values within one run to cancel it.

The depth-flat and multi-connection results above were collected before the
one-sided cutover. Under the v2 shared-segment PUT the server no longer receives
the payload inline in its serve loop (it lands directly in the leased slot via
`WRITE_WITH_IMM`), and the slab store commits through the async/io_uring write
path — the mechanism premise behind those numbers has changed. They remain
useful topology guidance, but absolute throughput and latency claims for
current v2 require a new hardware run.

## v2 datapath performance: TBD

No post-cutover hardware run exists yet. The items a v2 perf chapter must
measure before any absolute claims are made:

- **PUT slot-to-durable latency**: client `WRITE_WITH_IMM` slot submission →
  server commit acknowledgment, at 1 MiB and page-scale payloads, across
  negotiated `qd` values.
- **Server `RDMA_WRITE` GET**: warm (RAM-tier) and cold (NVMe → stage → WRITE)
  paths, single connection vs `batch_concurrency` fan-out; compare against the
  retired two-sided numbers above.
- **Commit granularity**: per-slot store batches vs depth on the slab
  io_uring write path (`dfkv_slab_uring_write_batches_total`).
- **Depth curve under v2**: re-prove or refute depth-flatness on the one-sided
  plane (`qd = min(client, server)`; audit the runtime value via `dfkvctl ring`
  INFO `qd=`).
- **Shared receive-segment capacity effects**: `DFKV_RDMA_RECV_SEGMENT_SIZE`
  sizing vs concurrent connection count (`qd`×slot-size lease per connection);
  exhaustion behavior and its impact on admitted concurrency.
