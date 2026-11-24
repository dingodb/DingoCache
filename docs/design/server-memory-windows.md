# Server memory windows for GET grants

Status: implemented (ledger RDS-09, part A), on top of #539, #527, #540,
#542 and #553. Switch: `DFKV_RDMA_SERVER_MW`, server side, default off.
Nothing changes on the wire. Section 13 lists where the code differs from
this design and why.

## 1. Why

Every GET the server serves by READ registers a memory region on exactly the
grant's bytes (`RegisterLeaseReadRegion`) and deregisters it at RELEASE; every
leased PUT does the same for WRITE (`RegisterLeaseWriteRegion`). Both system
calls are on the client's critical path: the READY waits for the
registration, the RELEASE reply for the deregistration.

Measured on a ConnectX host (400G InfiniBand, main 1dad0f44), touched host
memory, p50 (p99) in microseconds:

| size | `ibv_reg_mr` | `ibv_dereg_mr` | pair |
|---|---|---|---|
| 4 KiB | 28 (201) | 27 (183) | ~55 |
| 64 KiB | 28 (53) | 27 (36) | ~55 |
| 256 KiB | 32 (39) | 28 (35) | ~60 |
| 1 MiB | 47 (74) | 32 (41) | ~79 |
| 4 MiB | 163 (441) | 60 (505) | ~223 |

A type-2 memory window on a region registered once, bound to a 64 KiB range
by an `IBV_WR_BIND_MW` work request and invalidated by an `IBV_WR_LOCAL_INV`,
each signalled on the connection's QP: bind 3.5 us (9.0), invalidate 3.4 us
(10.0), about 7 us per grant whatever the size. A small single GET takes about
170-200 us in event mode; some 55 us of that is the registration pair.

## 2. Scope

- **GET grants (READ) use windows.** This is the change.
- **Leased PUT grants (WRITE) keep today's exact per-grant MR** with
  synchronous deregistration. A WRITE that raced a window's invalidation could
  write into a range already reused for another connection's data; the
  deregistration's device guarantee is kept there, and no WRITE windows are
  bound in this change.
- Fallback to today's per-grant MR for GET, per grant or per connection, in
  each case of sections 6 and 7; never a failed connection.

## 3. What changes, with the switch on

- **Probe at startup, per device:** `IBV_DEVICE_MEM_WINDOW_TYPE_2B` in
  `device_cap_flags` and an `ibv_alloc_mw(pd, IBV_MW_TYPE_2)` that succeeds
  (and is freed again). A device without them serves as today; the startup
  log and a gauge say which mode each rail uses. NVIDIA documents type-2B
  windows for mlx5 physical functions only, so virtual functions fall back.
- **Bind-only regions.** After #539 there are no chunk-wide MRs: each
  connection's slots have their own registration, which is what isolates
  connections. Window mode adds one region per receive chunk, registered
  with `IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_MW_BIND` and **no remote access**:
  its rkey authorizes nothing remotely, only the binding of windows, so #539's
  isolation is unchanged. Likewise one per RAM-arena extent (section 7). Both
  are registered outside `g_dev_mu` (#540's rule), off the request path where
  possible (a chunk's at its creation, the arena's per extent, section 7).
- **Windows per connection:** one per pull slot (depth; 16 with #553), allocated
  when the connection is set up and deallocated when it ends. If
  `ibv_alloc_mw` fails, that connection serves its GETs with per-grant MRs,
  counted; it never fails (section 6).
- **A GET grant** posts, in one `ibv_post_send`, an `IBV_WR_BIND_MW` (slot's
  window, a fresh rkey from `ibv_inc_rkey`, range exactly the grant's bytes,
  `IBV_ACCESS_REMOTE_READ`) and then the READY SEND carrying that rkey and
  address, as the per-grant MR's are today.
- **RELEASE** posts an `IBV_WR_LOCAL_INV` of that rkey and the reply SEND in
  one post; the range then goes into quarantine (section 4) when the
  invalidation's completion is seen.

### Ordering (bind, then the READY)

The bind and the SEND that carries its rkey are posted together, bind first,
on the connection's RC QP. `ibv_post_send(3)` lists `IBV_WR_BIND_MW` as a
send-queue work request on RC, and `ibv_bind_mw(3)` states that a send after a
bind request on the same QP is always ordered, so the rkey may be sent on the
same QP right behind the bind, but by no other means before the bind's
successful completion is read. This follows the InfiniBand rule that work
requests on one send queue are executed in order; a bind is a send-queue
operation. The client learns the rkey only from that SEND, so its READ
reaches the responder after the bind took effect. No fence: `IBV_SEND_FENCE`
makes a work request wait for this QP's own outstanding RDMA READ and atomic
requests, which is not what is ordered here. The bind is signalled, so a
failed bind is seen (the QP goes to error, and the client's READ fails
without touching memory).

## 4. Revocation and reuse

### The problem

Deregistration today is synchronous: when `ibv_dereg_mr` returns, the device
no longer accesses the range, and only then is it reused. A window's
invalidation gives no such guarantee against the QP's **inbound** side.
Nothing orders the responder serving the client's READ against the outbound
`LOCAL_INV`, and the reply SEND's completion says nothing about the
responder either. A client that READs after its RELEASE (a faulty or
hostile client) can have a READ in the responder while the invalidation
runs. If the range were reused at once, that READ could return bytes of
whatever value reuses it, which may belong to another tenant.

### The rule

**A revoked GET range is reused only after both:**
1. its window is known dead: the `LOCAL_INV`'s own completion observed with
   `IBV_WC_SUCCESS`, or the QP destroyed and the window deallocated
   (`ibv_dealloc_mw` unbinds; a type-2B window is usable only through its
   own QP); and
2. a **quarantine** has passed since then: long enough for any READ that was
   already in the responder to have been sent.

A READ request that reaches the responder after the invalidation fails its
rkey check. What is left is a READ the responder accepted before the
invalidation, and **its retransmissions**: a requester that lost part of a
READ's response resends the request (from the missing PSN), and the responder
executes it again. Whether that re-execution checks the rkey again decides
the bound:
- **rxe re-checks, per response packet.** `read_reply()` in
  `drivers/infiniband/sw/rxe/rxe_resp.c` calls `rxe_recheck_mr()` on the saved
  `res->read.rkey` for every packet it builds, original or duplicate, and
  `rxe_recheck_mr()` fails for a window that is no longer valid or whose
  rkey changed. On rxe an invalidated window stops the READ at the next
  packet.
- **mlx5: not established.** No vendor or firmware reference found states
  whether a duplicate READ re-checks the R_Key; the InfiniBand specification
  text on duplicate READ handling was not available to check. On the test host there
  is no reliable way to make the fabric drop a READ response, so no test.

So the doc does not claim a tight bound on mlx5.

### Quarantine time

Per grant: `max(floor, 2 x rail_bytes / slowest_link_rate + margin)`, where
- `rail_bytes` is the bytes of GET grants outstanding on the grant's rail
  (READY sent, not yet released or revoked), this one included, when the
  window goes: what the port could still have to send ahead of a READ
  already in the responder. The design first used the grant's own bytes;
  the incast measurement below showed a READ's lag follows the port's
  backlog, not its own size;
- `slowest_link_rate` is the lowest active rate among the server's rails
  (`ibv_query_port`: active speed x width) or a configured lower bound,
  whichever is lower;
- `floor` defaults to 5 ms (`DFKV_RDMA_SERVER_MW_QUARANTINE_MS` can only raise
  it). The HCA reads host memory as it builds each packet, so credit stalls
  and incast queueing can stretch a READ well past its nominal wire time.
- `margin`: 1 ms, from the measurement below.

**Measured** (host with real InfiniBand, 400 Gb/s; server on one port, 16
client processes on the other port of the same host, through the switch;
each reads RAM-tier values one READ at a time, 2000 READs per process and
size; the READ timed from its post to its completion; the host otherwise
idle):

| READ | wire time | p50 | p99 | p99.9 | p99.99 | max |
|---|---|---|---|---|---|---|
| 1 MiB, per-grant MRs | 21 us | 306 us | 471 us | 657 us | 741 us | 884 us |
| 4 MiB, per-grant MRs | 84 us | 655 us | 1160 us | 1270 us | 1333 us | 1392 us |
| 1 MiB, windows | 21 us | 123 us | 459 us | 611 us | 764 us | 855 us |
| 4 MiB, windows | 84 us | 634 us | 1168 us | 1310 us | 1439 us | 1514 us |

The lag at p99.99 is 0.7 ms (1 MiB) and 1.4 ms (4 MiB): 35 and 17 times the
READ's own wire time, but within twice the backlog of 16 such grants on
the port (0.7 ms and 2.7 ms) plus 1 ms. Every such grant here quarantines
the 5 ms floor.

For example, with the 5 ms floor at 400 Gb/s, up to about 95 MiB of
outstanding grants on a rail quarantine 5 ms; 1 GiB about 44 ms; past
`DFKV_RDMA_SERVER_MW_QUARANTINE_MAX_MS` (50 ms) a grant takes a per-grant
MR.

### What remains

This rule replaces the deregistration's device guarantee with a time bound,
and the bound is only as good as the rkey re-check above:
- **On a device that re-checks the rkey on every response packet (rxe):** a
  READ being served when the invalidation ran stops at its next packet; the
  quarantine covers the packets already being built.
- **On mlx5, until established otherwise:** a READ accepted before the
  invalidation, or its retransmission, may read the reused range for up to
  the client's transport retry horizon, about 3.7 s at the defaults
  (`DFKV_RDMA_QP_TIMEOUT` x `DFKV_RDMA_QP_RETRY_CNT`, measured 3.5-3.7 s).
  The quarantine does not cover that.

The exposure is **confidentiality only, toward a client that violates the
protocol** (reading after its own RELEASE, or keeping READs in flight across
it); a READ never writes, so it is **never data corruption**. A correct
client has finished its READs before it sends the RELEASE. The DEPLOY entry
of the switch says exactly this, with the mlx5 retry-horizon residual, and a
deployment that cannot accept it leaves the switch off.

**Condition on the default.** The switch stays off by default until it is
established on real hardware whether mlx5 re-checks the R_Key on a
retransmitted READ, by a vendor statement or a reliable test. Turning the default on
requires that first; the CHANGELOG and the PR say so.

### Quarantine accounting

- Quarantined bytes are counted (gauge) and capped:
  `DFKV_RDMA_SERVER_MW_QUARANTINE_BYTES`, default one eighth of the receive
  pool budget. A range is released from quarantine by the serve loop's timer
  (it already wakes for idle and grant deadlines).
- When the cap is reached, or a grant would need a quarantine longer than
  `DFKV_RDMA_SERVER_MW_QUARANTINE_MAX_MS` (default 50 ms), **that grant uses a
  per-grant MR**, counted. Quarantine never blocks a grant and never starves
  the pool.
- For a RAM-arena grant, the range is the arena slot: the grant's pin on it
  (`PreparedRead`) is held until the quarantine ends, so the RAM tier does not
  evict or overwrite the value under it; the pin counts in the same cap.
- **Ownership:** a quarantine entry outlives the connection and its Serve
  thread (a connection may close right after a RELEASE). Entries go into one
  process-wide quarantine list, ordered by expiry, which the receive-chunk
  trim thread (`recv_trim`) drains; it already wakes periodically. A Serve
  thread never waits a quarantine out.

### Per path

1. **RELEASE.** `LOCAL_INV` (signalled) and the reply SEND in one post. The
   slot becomes *revoking*: busy, range held. On the invalidation's success
   completion the range moves to quarantine and the slot is free for the next
   request. A request on that slot that arrives earlier is held until then.
2. **Grant timeout** (the client never releases). The connection ends, as
   today, by path 4.
3. **Invalidation fails** (error completion, or the post fails): the QP is in
   error; it neither posts nor serves READs. The connection ends by path 4;
   the range is not released before.
4. **Connection close, QP in error, server stop.** In `RcEndpoint::Close()`:
   destroy the QP, then deallocate the connection's windows; then the slots
   reset and their ranges go into quarantine. A window bound to a QP in
   error cannot be invalidated by a work request (an NVIDIA answer for
   type-2B windows), so this path deallocates instead. In `Serve()` the
   endpoint is destroyed before the slot state, so the order holds. Server
   stop takes this path for every connection and then waits out the
   quarantine (at most the max) before the chunks and arena are freed.

A bind-only region is deregistered only after every window on it is
deallocated (`ibv_dereg_mr` fails while one is bound) and its quarantine is
empty.

#553 raises the response READ depth to 16, so up to 16 READs per connection
can be in the responder; the quarantine bound is per grant and holds for
each.

## 5. Serve-loop changes

- The sync and io_uring loops (`process_wc` in both, including #542's) treat
  every completion that is not a receive as a SEND today. Bind and
  invalidation completions get their own `wr_id` space (a tag bit plus slot
  and generation) and must not touch `posted_sends` or `free_send`.
- Revoking slots count in `has_async_work` and in `grant_view`, so idle
  reclaim and the stop drain do not take the connection for quiescent while
  an invalidation is pending; `wait_bound_ms` includes the earliest pending
  invalidation deadline. Quarantine is not per connection (section 4,
  ownership), so it does not hold a connection open.
- The send queue and CQ grow by two work requests per pull slot (bind and
  invalidate; about three per slot with the reply), sized at connection set
  up.

## 6. Window resources

- 16 windows per connection with #553. At startup the server compares
  `max_mw` with depth x its expected connections (the endpoint budget) and
  logs a WARN if it does not fit. That count also covers batched GETs
  (#551's windows of keys, #553's 16 READs in flight): they use the same pull
  slots, one window per slot.
- `ibv_alloc_mw` failing for a connection: that connection's GETs use
  per-grant MRs, counted (`dfkv_rdma_server_mw_fallback_connections_total`);
  the connection is served.

## 7. The RAM arena: one bind-only region per extent and device

The arena (`DFKV_RAM_TIER_BYTES`, extents of `DFKV_RAM_TIER_EXTENT_BYTES`,
32 MiB by default) is not registered up front today, by design.

- **Registered per (device, extent)**: each rail has its own PD, so an extent
  used by grants on N rails is registered N times. A registry per device
  holds each extent's state: none, in progress, registered, failed (with a
  retry time). The first grant into an extent starts the registration
  (outside `g_dev_mu`); a grant that finds it in progress, or failed, uses a
  per-grant MR for that GET. No extent is registered twice on one device.
- **Values up to an extent** sit inside one extent (slab classes). Values
  larger than an extent live in the large reserve
  (`DFKV_RAM_TIER_LARGE_RESERVE_BYTES`, 0 by default) and keep per-grant MRs.
- **Cap**: `DFKV_RDMA_SERVER_MW_ARENA_BYTES` per device (default the whole
  arena). Past it, grants into unregistered extents use per-grant MRs.
- **Footprint.** Physical memory: none added; the arena's pages are resident,
  `mlock2(MLOCK_ONFAULT)`ed and charged to the cgroup's `MemoryMax` whether
  registered or not. Locked-memory limit: the kernel checks pinned pages
  (`pinned_vm`) and `mlock`ed pages (`locked_vm`) each against
  `RLIMIT_MEMLOCK`, not their sum, so the limit must cover the arena once
  (the shipped unit sets `LimitMEMLOCK=infinity`).
- **Invariant**: arena pages are never released or remapped
  (`MADV_DONTNEED`, `munmap`, `mremap`) while an extent is registered. The
  arena is mapped once and kept for the process's life today; the RAM tier
  gets an assertion and a comment so a later change cannot break it unseen.
- **First-use cost**: about 1-2 ms per (device, extent) on 2 MiB pages, to be
  measured. If it shows in tail latency, with the switch on the extents are
  registered when the arena is populated (in the background, per device).

## 8. Receive chunks

- Registering a chunk's bind-only region takes 50-100 ms for a large chunk;
  it must not run under the receive pool's lock (`mu_`) or `g_dev_mu`. Each
  chunk carries a registration state per device (none, in progress,
  registered, failed); `NewChunk` only creates the chunk, and its region is
  registered afterwards by a thread outside the locks (the first grant into
  it starts it, or the trim thread ahead of need). A GET grant into a chunk
  whose region is not ready on that device uses a per-grant MR.
- Trimming (#527, `TrimIdle`) runs in two phases: under the pool lock, unlink
  a chunk that is empty (no leases, nothing in quarantine); outside it,
  deregister its bind-only regions and free it.
- `ibv_dereg_mr` returning `EBUSY` at trim means a window is still bound to a
  chunk the server took for empty: a bug. The chunk is kept (never freed
  under a live window) and the server logs it at ERROR.

## 9. Security

Scoped: **grants served through windows are reachable only by their own
connection** (a type-2B window is used only through the QP that bound it),
where a per-grant MR's rkey is valid on every QP of the device's PD. Fallback
grants, leased PUT grants and #539's own-slot MRs stay PD-wide, as today.
The trust model in DEPLOY does not change.

## 10. rxe in CI

rxe has had type-1 and type-2B windows since about Linux 5.15, and it
re-checks the rkey on every READ response packet (section 4). The startup
probe's result is printed in the CI job. If CI's rxe reports 2B, the window
tests are required there (#528's skip allowlist gets no entry for them); if
not, CI tests the fallback and the window path is tested on InfiniBand hosts
only, which the PR states. rxe passing says nothing about section 4 on mlx5;
that is measured on a host with real InfiniBand.

## 11. Measurement and tests

Before and after, switch off and on, one rail and two, RAM tier and disk:
- server registration time per grant (both modes);
- single GET p50/p99 at 4 KiB, 64 KiB, 1 MiB, 4 MiB; batched GET (16 keys)
  and the SG shape as for TRN-12/13;
- first grant into an extent; quarantined bytes under load.

Tests (seams where the device would not fail on its own):
- a READ with a revoked window's rkey fails; the range is not reused before
  the invalidation's completion and the quarantine (a hook asserts the
  order);
- grant timeout, QP forced to error, server stop: windows deallocated before
  any range is released, then quarantine;
- quarantine cap reached: the grant falls back to a per-grant MR;
- `ibv_alloc_mw` failure (seam): the connection serves with per-grant MRs;
- extent registration failure and an in-progress extent (seams): fallback,
  no double registration;
- chunk trim with a window bound: kept;
- leased PUT unchanged (per-grant MR);
- interop: a 2.42.0 client against a server with the switch on; nothing
  changes on the wire.

## 12. Metrics

- `dfkv_rdma_server_mw_mode{dev}` (gauge).
- `dfkv_rdma_server_mw_binds_total`, `_invalidations_total`,
  `_invalidation_failures_total`.
- `dfkv_rdma_server_mw_quarantine_bytes` (gauge), `_quarantined_total`,
  `_quarantine_fallbacks_total`.
- `dfkv_rdma_server_mw_arena_registered_bytes{dev}`,
  `_region_registrations_total`, `_region_register_failures_total` (chunks
  and extents), `_fallback_grants_total`, `_fallback_connections_total`.
- `dfkv_rdma_server_grant_register_seconds` (summary: `_sum`, `_count`;
  both modes).

### Measured with the code

Host with real InfiniBand (400 Gb/s), one rail, one GET at a time, the
switch off and on alternated three times per size (the host is shared;
medians of the three runs), in microseconds:

| GET | auth. per grant, off -> on | p50 off -> on | p99 off -> on |
|---|---|---|---|
| RAM tier 4 KiB | 27.6 -> 1.6 | 214 -> 164 | 292 -> 388 |
| RAM tier 64 KiB | 24.2 -> 1.2 | 221 -> 146 | 287 -> 285 |
| RAM tier 1 MiB | 20.3 -> 1.0 | 220 -> 198 | 385 -> 397 |
| RAM tier 4 MiB | 29.9 -> 1.5 | 424 -> 508 | 1822 -> 1578 |
| disk 4 KiB | 18.4 -> 1.0 | 157 -> 103 | 217 -> 166 |
| disk 64 KiB | 20.0 -> 1.0 | 170 -> 118 | 228 -> 455 |
| disk 1 MiB | 14.5 -> 0.7 | 187 -> 155 | 225 -> 431 |
| disk 4 MiB | 21.1 -> 1.3 | 448 -> 383 | 793 -> 786 |

"auth." is `dfkv_rdma_server_grant_register_seconds` per grant: the exact
MR's registration, or the bind's post. A GET's phases timed alone: the
PREPARE 67 -> 38 us (1 MiB), the READ unchanged, the RELEASE 35-67 -> about
135 us: its reply waits behind the invalidation on the send queue, which
shows at 4 MiB in the RAM tier. Raw verbs on the same host, one QP pair:
bind, READ and invalidate 31 us (64 KiB) and 76 us (1 MiB), against 50 and
116 us for a per-operation MR. The p99 column moves with the host's other
load between runs.

## 13. Where the code differs

- **Completions**: the bind and the invalidation are posted unsignalled. A
  completion of each, signalled, woke the serve thread twice more per GET,
  which put up to 0.5 ms on the GET's tail (measured, phases above). The
  READY's successful completion, after the bind on the send queue, means the
  bind took effect, and the RELEASE reply's means the invalidation did (an
  RC send queue completes in order; a failure of either ends the QP, and its
  completion, with its `wr_id`, still comes). The range goes into
  quarantine at the reply's completion.
- **RELEASE**: the invalidation and the reply are posted on the send queue
  in that order, as designed; the slot stays busy until the reply's
  completion. A PREPARE on a slot still revoking is
  held in the receive FIFO until the reply's completion retires the slot,
  as designed (the reply's completion comes with the client's ACK, a
  separate packet from its next request, so either may be polled first).
- **Quarantine time**: the rail's outstanding grant bytes, not the grant's
  own (section 4, from the measurement); the admission check at the grant
  uses the same bound. The end is clamped to the maximum: grants added
  to the rail after the admission do not stretch it. A server sees only its
  own outstanding bytes; queueing at one client's link from several servers
  is covered by the 5 ms floor alone.
- **Registration**: one background thread registers both receive chunks
  and arena extents (section 7 and 8), on the first grant into each; there
  is no registration ahead of need.
- **Chunk trim**: two-phase for every trim, also with the switch off (the
  chunk is freed after the pool lock is released); a chunk whose
  deregistration fails is kept, never freed, and counted.
- **Close order**: `Close()` deallocates the windows, then destroys the
  QP (section 4 path 4 has it the other way). Deallocation invalidates and
  unbinds each window, so a READ through it fails from then on; on
  Soft-RoCE a still-bound window holds its QP, whose destruction then
  waited (about 50 s in CI). The ranges are still released only after the
  endpoint is gone. A deallocation the device refuses (`EBUSY` on a live
  QP in the middle of a transfer) is tried again after the QP is moved to
  the error state, which ends its work in flight, up to 10 times within
  10 ms, and only then is the QP destroyed: destroying it with the window
  still bound waited about 50 s on Soft-RoCE. A window still refused is
  tried once more after the QP's destruction; only a failure then aborts.
- **Throughput with many clients, measured** (2.35.7 `dfkv_bench`, 4
  client threads on pinned CPUs, off and on alternated, medians of 3-6
  runs):
  - values in the RAM tier: windows equal or faster. 64 KiB in batches of
    8: 26600 -> 42700 GET/s; 256 KiB one key at a time, four runs on one
    node: 12500-19900 off, 15100-26100 on.
  - GETs staged from disk, under that concurrency: slower with windows,
    e.g. 3900 -> 3400 and 6100 -> 4700 GET/s at 256 KiB. Cause open
    (RDS-29): not the per-grant bind and invalidation (a window bound once
    per slot over its whole region was as slow), not the quarantine (none
    at all was as slow), not the READ (raw verbs: a READ through a window
    takes as long as one through an MR, 64 KiB to 1 MiB, one or four
    connections). To be resolved before the default changes.
  - An earlier "batched GETs halve" (5981 -> 2883 GET/s) came from a RAM
    tier too small for the bench's data: those GETs were staged from disk.
    The PUT refusals (`kCacheFull`) seen in such runs come from the RAM
    tier's writeback backlog and happen with the switch off too.
- **Link rate**: the slowest rail's active rate, or
  `DFKV_RDMA_SERVER_MW_LINK_GBPS` if lower; 10 Gb/s if neither is known.
