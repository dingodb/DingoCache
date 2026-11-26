# Changelog

## Unreleased

- Server, RAM tier: a RAM-ACK PUT refused at the dirty watermark can wait
  for the flush instead (REQ-26, `DFKV_RAM_ACK_BACKPRESSURE_WAIT_MS`, 0 by
  default: refused at once, as before). Over the watermark (80% of the tier
  unflushed) a PUT of a new key was answered kCacheFull, and the connectors
  take that as not stored: measured on a 256 MiB tier after 250 MiB written,
  a burst of 4 x 250 PUTs of 256 KiB had 625, 527 and 586 of 1000 refused in
  three runs, though the flush cleared the backlog within tens of
  milliseconds. With a wait of up to 50 ms none were refused, p99 4.7-6.9 ms,
  at most 8.8 ms. The PUT
  waits without its key's lock, is admitted again each time a value is
  flushed, and is refused when the wait or the batch's
  `DFKV_RAM_PUT_WAIT_MS` ends, or the server stops. New
  `dfkv_ram_ack_backpressure_waits_total`, `_wait_timeouts_total`,
  `_wait_microseconds_total`. `dfkv_ram_ack_backpressure_total` counts each
  refused PUT once, after any wait.
- Server and client: rewrite after REMOVE (CLI-35), behind
  `DFKV_WRITE_MEMORY_ENTRIES` (server, 0 = off; needs
  `DFKV_REMOVE_MEMORY_MS`) and `DFKV_RING_READBACK_REWRITE` (client, 0 = off;
  needs `DFKV_RING_READBACK_MS`). A node remembers how long ago it
  acknowledged each recent PUT and answers a new op, `kRemovedWithinAges`
  (26, TCP and RDMA): the ages of each key's latest REMOVE and PUT, on its
  `CLOCK_MONOTONIC_RAW`. With W on, per-key locks order each PUT/REMOVE with
  the op-26 evidence read, including within one millisecond. The read-back
  then also reads back a key removed
  at an asked node if the previous owner's write followed every reported
  REMOVE, compared through the client's own request times with a clock-rate
  bound of 1000 ppm and a 1 s pause guard. The node clears the memory when
  `CLOCK_REALTIME` or `CLOCK_BOOTTIME` moves against the raw clock by more
  than 1 s within the window. Nodes without the op, or without the memory,
  get the old rule. The recent-REMOVE memory's ages move from
  `CLOCK_MONOTONIC` to `CLOCK_MONOTONIC_RAW` (its answers do not change).
  For hosts whose clocks keep running through pauses only (DEPLOY §4c-6).
  New `dfkv_write_memory_entries`, `dfkv_write_memory_dropped_total`,
  `dfkv_write_memory_cleared_total{reason}`,
  `dfkv_client_readback_reads_total{result="hit_rewritten"}`.
  The effective config printout reflects clamped/dependent M/W/rewrite
  settings after resolution. W's timestamp checks advance each shard even
  when nonempty, preventing a stale per-shard watermark from losing the
  same-millisecond PUT/REMOVE ordering. The unused internal C++ server-only
  `KvNodeServer::CacheDirectBatchForKeys` wrapper is removed because it would
  bypass W admission/publication; the client C ABI is unchanged.

## 2.44.0 — 2026-11-26

- vLLM connector: support vLLM's manager-owned KV-cache allocation policy
  (introduced by upstream #61452) and current main while retaining the legacy
  pinned policy-typed-spec API. Groups resolve their real manager config and
  DCP block geometry, mixed layouts no longer use an arbitrary first-layer
  spec, and full/SWA/Mamba lookup masks follow manager-owned retention/replay
  settings. Saves stop at emitted lookahead hashes, finish-time tail and
  partial-tail sources keep their blocks alive, KV events carry lookahead
  token IDs, async-load admission follows the connector policy, and DCP
  ranks use actual group membership on main (tested against upstream main
  `e7f220fdfaf71bb85d8c4bb833070101e9a6ced6` plus the legacy CI pin).
  Attention-free models legitimately report zero KV heads on main; their rank
  identity now bypasses attention-head replication without dividing by zero.

- vLLM connector identity: `vllm-multiwr-v5` binds cache layout, manager
  policy, each group's logical bytes per block, and draft/lookahead semantics.
  Existing objects intentionally cold-miss; no legacy reads, dual writes,
  aliases, server upgrade, ABI/wire change, or cache deletion is required.

- Server, RDMA: GET grants can be bound by memory windows (RDS-09,
  `DFKV_RDMA_SERVER_MW=1`, off by default; design in
  `docs/design/server-memory-windows.md`). Each GET registered a memory
  region over the value it served and deregistered it at the RELEASE, about
  55 us per GET (220 us at 4 MiB). With the switch on, each RAM-arena
  extent is registered once per rail, with no remote access (without a RAM
  tier the switch does nothing, as the startup log says), and a GET of a
  value in the RAM tier binds a type-2B window on its own connection (3.5 us) that the
  RELEASE invalidates (3.4 us); the range is reused only after the
  invalidation's completion and a quarantine of at least 5 ms (longer with
  many GET bytes outstanding on the rail). Measured, one GET at a time:
  the server's authorization of a grant 15-30 -> 1-1.5 us; GET p50 from the
  RAM tier 221 -> 146 us at 64 KiB, 220 -> 198 us at 1 MiB, 424 -> 508 us
  at 4 MiB (its RELEASE reply waits behind the invalidation). A GET staged
  from disk keeps its per-grant MR: a READ of memory the disk has just
  written by DMA is slow through any registration older than the write,
  and fast through one made after it (RDS-29).
  Nothing changes on the wire; clients of every version are served. A rail
  whose device has no type-2B windows, a connection that cannot allocate
  them, an arena extent not registered yet, a full quarantine: each uses the
  per-grant MR, counted. Leased PUTs keep their per-grant MR. The residual
  risk is confidentiality only, toward a client violating the protocol
  (reading after its own RELEASE); never data corruption. On mlx5 a READ
  accepted before the invalidation, or its retransmission, may read the
  reused range for up to the client's transport retry horizon (about 3.7 s
  at the defaults); the quarantine does not cover that. The default stays
  off until it is established on real hardware whether mlx5 re-checks the
  R_Key on a retransmitted READ. Under concurrency (4 client threads),
  windows are faster from the RAM tier (64 KiB in batches of 8: 29000 ->
  42000 GET/s) and as fast for GETs staged from disk (256 KiB: 5040 ->
  5390 GET/s). New
  `dfkv_rdma_server_mw_*` metrics and
  `dfkv_rdma_server_grant_register_seconds` (both modes). A receive chunk
  is now freed after the pool lock is released when it is trimmed. Quarantine
  time uses the configured maximum hold, instead of hoping a bind-time rail
  estimate bounds later grants; that estimate only selects a window versus a
  per-grant MR at authorization. The byte cap accounts whole arena allocations
  (including padding), plus temporary staging leases. Both reserved grants and
  actual quarantine are charged under one accounting lock, with each reserved
  amount moving unambiguously into quarantine rather than being recomputed
  from the later data length. Concurrent admissions cannot overshoot;
  the gauge continues to count actual quarantine only. Teardown checks the
  QP-to-error transition before retrying refused windows; it stays fail-closed
  when revocation cannot be proven, without assuming a provider-wide timeout.

## 2.43.0 — 2026-11-25

- Server, failed disks (STO-16 follow-up): a disk excluded from takeover (it
  kept a key of another disk the start could not remove) no longer counts
  toward `DFKV_DEGRADED_MIN_DISKS` once a disk has failed. With every
  healthy disk excluded, the failed disk's keys failed with I/O errors while
  the node stayed in placement; it now leaves placement. The keys that start
  drops from a disk that is not their home are synced before the failure
  record can be cleared (the file engine syncs its filesystem; its REMOVE
  alone is not durable across a power loss); if that sync fails, the record
  is kept for the next start.


- Client: keys a node refused to store are counted,
  `dfkv_client_node_refusals_total{status="cache_full"|"quota_exceeded"}`
  (also in the Python metrics mirror). A write-back RAM tier over its dirty
  watermark answers new PUTs kCacheFull while its flush backlog drains; the
  connectors take that as not stored, and it was not visible on the client.

- RDMA client: `DFKV_RDMA_CONNECT_MS` and `DFKV_RDMA_IO_MS` above 2^31-1
  resolved to a negative or zero timeout (the number was cast to an int);
  they now resolve to the default, with the warning they already gave. The
  client's numeric RDMA settings are read through the shared setting parser
  (TRN-30); every other value, and every warning, is as before.

- RDMA client: a rail recovers from a device reset without a restart
  (TRN-29, `DFKV_RDMA_DEVICE_REBUILD`, off by default). A firmware or PCI
  reset or a driver reload leaves the device's contexts dead; a transport
  that registered memory held the dead context, so the rail stayed broken
  until the process restarted. With the switch on, `IBV_EVENT_DEVICE_FATAL`
  or an endpoint Open failing with ENODEV/EIO retires the device's binding,
  the rail leaves the transport's selection, and the background thread
  rebuilds it on a fresh context with every declared region registered
  again, then puts it back. A release that fails on a context proven
  disassociated is counted (`dfkv_rdma_client_dead_release_failures_total`)
  instead of aborting; anything short of proof still aborts. With the
  switch off the signals are only counted (`dfkv_rdma_client_async_events_total`,
  `dfkv_rdma_client_device_suspect_total`). Client only; servers still need
  a restart after a device reset. Design: docs/design/rdma-device-reset.md.

- Client, absent grace: a member's return to the view ends its PeerHealth
  cooldown (once, as a probe answer) in every client, not only in one with
  REMOVEs queued for it. A reader that had cooled the member before it
  vanished skipped it until its backoff ended (up to 30 s); its next read
  now goes to the member (CLI-36).
- Client: previous-owner read-back (CLI-02 stage B), behind
  `DFKV_RING_READBACK_MS` (default 0 = off; MDS discovery only; needs the
  nodes' recent-REMOVE memory). Within the window after a write-ring change
  (or the end of a cooldown-reroute episode), a GET or EXIST the current
  owner answers "not found" asks the memory of the key's first successors
  on every recent ring and the ring before the change (not the current
  owner) whether the key was removed, and reads the previous owner once
  only if every one answers "not removed"; "unknown" or an unreachable node
  refuses. Requires `DFKV_RING_REMOVE_FANOUT` >= 3. With cooldown reroute,
  and only while every registered client advertises B (`readback=1` in its
  registry marker), an ended episode's keys stay on the substitute until
  its window passes (they count toward `DFKV_RING_REROUTE_KEYS`; a close
  removes them at once); a client with B asks the same question before
  reading a substitute's copy it did not write in its current episode (GET,
  EXIST and RETAIN), and
  a client with cooldown reroute alone treats such a read as a miss while
  any registered client advertises B. The deferral ends at once (every
  deferred key cleaned up) when a client without B registers or one drops
  it, the registry answer is stale, or the gate closes. New
  `dfkv_client_readback_reads_total{result}`,
  `dfkv_client_readback_changes_refused_total{reason}`,
  `dfkv_client_readback_window_active`,
  `dfkv_client_readback_memory_short_total{reason}`. Absent-owner read-back (design
  §3.4): not in this stage. Off, no
  `kRemovedWithin` is sent and every read is unchanged.

- Server: recent-REMOVE memory (CLI-02 stage M), behind
  `DFKV_REMOVE_MEMORY_MS` (default 0 = off, up to 3600000). A node records
  every REMOVE it acknowledges (key and tenant, before the reply; RAM only,
  at most `DFKV_REMOVE_MEMORY_ENTRIES`, default 1048576, 40 MiB when
  full: 40 B an entry, CLI-33) and answers a new op,
  `kRemovedWithin` (25, TCP and RDMA): was each key removed here within the
  last `age_ms`, or unknown when that reaches before the node's start or
  its oldest kept record. It exists for the previous-owner read-back of a
  later stage; no other op's answer changes, a PUT after a REMOVE is stored
  and served as before. A node with the switch off, like an older one,
  answers `kInvalid`; the client's `KVClient::RemovedWithin` then reports
  every key unknown, does not charge the node in PeerHealth and does not ask
  it again for 60 s. New `dfkv_remove_memory_entries`,
  `dfkv_remove_memory_covered_seconds`, `dfkv_remove_memory_dropped_total`,
  `dfkv_removed_within_queries_total{result}` (only with the switch on).
- Client, RDMA: a batch GET into the callers' buffers runs in windows. The
  zero-copy batch reads the connectors use (`dfkv_batch_get`,
  `dfkv_batch_get_auto` and `dfkv_batch_get_auto_sg`) read their keys one at
  a time: a connection, a request, the READs, a release, then the next key.
  The keys of a batch now share one connection, and each window of keys (up
  to the connection's depth) has its requests, READs and releases in flight
  together. Measured, 16 keys from a RAM-tier node, one rail: 64 KiB values
  2.54 -> 0.95 ms, 512 KiB 2.60 -> 2.37 ms, 4 MiB 17.3 -> 4.8 ms; 61
  segments per key of 4 KiB 18.9 -> 4.1 ms, of 64 KiB 22.3 -> 8.3 ms.
  Nothing changes on the wire. A batch that finds a stale pooled connection
  retries from the window that failed; finished windows keep their results.
  Any other failed window fails only its own keys: the windows after it are
  read once more on a fresh connection (another rail when there is one),
  except after a timeout. Exact reads in a batch keep TRN-26's rule: a value
  of another length is released without a READ.
- Client, RDMA: a completion wait can poll briefly before it sleeps. A GET
  is three round trips (request, READ, release), and each sleep on the
  completion channel and wake-up cost some 30-50 us. With
  `DFKV_RDMA_SPIN_US` (default 0, off) a client endpoint first polls its CQ
  for up to that long, within the wait's timeout. At 50, recommended for
  latency-sensitive small-value deployments: one GET at a time from a
  RAM-tier node, 64 KiB p50 139-177 -> 45-66 us, about what
  `DFKV_RDMA_BUSY_POLL` gives; the cost is up to about 150 us of spinning
  per GET when the server is slower than that (some 1.5 cores at 10k GETs/s).
  Server endpoints do not spin. The busy-poll wait no longer makes a
  `poll()` system call on its wake pipe on every spin. A signal during a
  completion wait no longer starts its timeout again. Nothing changes on
  the wire.
- Client, RDMA: a scatter-gather GET into memory that is not registered
  registers touching segments together (TRN-28). Each destination segment
  outside registered memory got its own registration for every GET: 16 keys
  of 61 contiguous 16 KiB segments took 976 registrations and 28-68 ms
  per call, against 1.6-2.6 ms with the memory registered. Segments that
  overlap or touch now share one registration (the same pages, so nothing
  else of the caller's is pinned or, with fork support, kept from a child):
  16 registrations and 2.2-2.9 ms. Scattered segments still take one each
  (30-32 ms); registering the buffers once with `dfkv_register_memory` is the fix
  there, and a rate-limited WARN says so. A merged range the device refuses
  is registered segment by segment. The registrations now last the whole
  GET rather than one READ round; where the locked-memory limit refuses
  that (ENOMEM, EAGAIN), the GET registers round by round as before; in a
  batch window, a key whose round the limit refuses waits for the next one. New
  `dfkv_rdma_client_unregistered_dest_registrations_total`,
  `dfkv_rdma_client_unregistered_dest_span_fallbacks_total`,
  `dfkv_rdma_client_unregistered_dest_memlock_fallbacks_total`.


- Server, RAM tier: a value a flush gives up on is counted before it is
  dropped (REQ-25). The drop takes the value out of the dirty count, and
  the counters (`dfkv_ram_flush_dropped_total`, `_refused_total`,
  `_no_room_dropped_total`) and the health came after it, so a reader that
  saw the tier drained (the stop sequence's drain wait, a scrape) could
  still see the drop uncounted. Seen once as a flaky unit test.

- Server: the node's in-memory key tables hash every key field, seeded per
  process (SRV-01), as the log store's index does since #532: the parked
  RETAIN SETs, the PUT key stripes and the recent reads of
  `DFKV_RAM_PROMOTE=second`. Their hash, digest_hi ^ digest_lo * C, put
  keys whose fields are related into a handful of buckets: measured with
  100000 keys {id * C + 1, id}, filling the parked-SET table took 28 s
  instead of 7 ms (one chain of 50000 keys), 8 of the 256 PUT stripes held
  every key, so PUTs of different keys waited for each other's stripe lock,
  and two passes of recent reads took 114 s instead of 37 ms. In-memory
  only: nothing on the wire or on disk changes.

- Client, RDMA: no `getenv` on the data path. Five test-only fault hooks
  (`DFKV_RDMA_TEST_*`) read the environment on every READ, PUT batch and
  attempt, and `DFKV_RDMA_BUSY_POLL` was read for every connection. In a
  host that calls `setenv` meanwhile (Python does), that is a data race.
  The hooks are now read only in test binaries that turn them on, and
  `DFKV_RDMA_BUSY_POLL` once, when the transport is built.

- Client, RDMA: an operation keeps to its timeout. `DFKV_RDMA_OP_TIMEOUT_MS`
  bounded each completion wait, but not the connection setup before it:
  the capability probe and the bootstrap read waited `DFKV_RDMA_IO_MS`
  (10 s) on their own, so a GET to a node that accepts TCP and never
  answers failed after 10.2-10.5 s with the default 5 s timeout. An
  operation now has one deadline from its start: admission, the budget
  wait, the probe, the dial, the bootstrap reads, the GET's request, READs
  and release, and a retry get only what is left of it (5.0-5.3 s in the
  same case), except the one retry after a stale pooled connection, which
  gets the full timeout again (that attempt reached no server); such an
  operation can take up to two timeouts, as before. A batch bounds each attempt's connection setup by its batch
  timeout; its windows keep their own. A connect that times out at a limit
  the deadline cut short marks nothing unreachable (a refused one still
  does), and a bootstrap read or a release that runs out of the operation's
  time charges no rail: a server under receive-pool pressure holds its
  bootstrap reply for up to 5 s. The confirming dial of a refused rail
  (2.42) also stays within the deadline. An endpoint's completion wait no
  longer starts its full timeout again after a wake-up that brought no
  completion. New `dfkv_rdma_client_op_deadline_expired_total`.

- Server: one failed disk no longer takes the node out of placement
  (STO-16). A disk store that fails for good while serving used to make the
  whole node report itself out of the ring, so its healthy disks got no
  traffic until a restart. The failed disk's keys now go to the node's
  healthy disks (spread evenly; a healthy disk's keys never move), where a
  GET is a miss (`kNotFound`, not an I/O error, so clients do not cool the
  node down) until the key is written again; RAM-tier values on their way
  to the failed disk are written to the disk that takes over their keys.
  The node stays in placement and `/healthz` stays 200 while at least
  `DFKV_DEGRADED_MIN_DISKS` (default 1) disks are healthy; set it to the
  disk count for the previous behaviour. New `dfkv_disks_failed`,
  `dfkv_disk_failed{disk}`, `dfkv_server_degraded`, alert
  `DfkvServerDegraded`, and `store=degraded` in the node's INFO. Before
  the first key moves, the failure is recorded next to each healthy disk's
  directory (`<dir>.dfkv-failed-disks`), only where the directory above is
  on that disk's own filesystem: each cache directory must be a
  subdirectory on its disk (as `/mnt/diskN/dfkv`), not a mount root or a
  container volume, and the service must be able to create files above it;
  with no such place nothing moves and the node leaves placement as
  before. The next start resets the disks on record to empty (each engine
  its own way) and drops from the other disks the keys that are not
  theirs, so neither a value from before the failure nor one written in
  its place comes back after a REMOVE or an overwrite; a record that
  cannot be read in full or names a directory the node does not have
  empties every disk (new `dfkv_disks_open_dropped_keys{reason}`). A
  clean stop of a run without failures removes the record. Below the
  threshold a failed disk's keys do not move. A failed disk's bytes no longer count against tenant quotas, and its retention
  limits no longer bound the node's. A disk that cannot be opened at
  startup still stops the start. Capacity and disk changes still need a
  cold start; DEPLOY.md now says so and how to replace one disk without
  losing the others' data.

- Client: an allocation that fails in a client's one-time set-ups no longer
  aborts the process. The GPU dedup rendezvous of a client's first
  all-device batch, the configuration dump of the first client and the
  SYNC_MEMOPS warning ran under `std::call_once`, through which an exception
  aborts (SIGABRT). Now a failed GPU dedup set-up leaves that client without
  GPU dedup (with a WARNING), and a failed log line is skipped.

- Client: cooldown reroute (CLI-02 stage C), behind
  `DFKV_RING_COOLDOWN_REROUTE` (default 0 = off; MDS discovery only). A key
  whose owner this client's PeerHealth has in cooldown goes to a substitute
  for PUT, GET, EXIST and RETAIN: the first stable, non-cooling member among
  the key's first `DFKV_RING_REMOVE_FANOUT` (default 3) successors on the
  union of the write rings this client adopted or saw in the last S (11 of
  the group's longest poll intervals). The keys written to a substitute are
  removed from it when its episode (that cooldown) ends (at most
  `DFKV_RING_REROUTE_KEYS`, default 65536, tracked or waiting for their
  cleanup; past that a rerouted write is refused), and by a
  clean close within `DFKV_RING_REROUTE_CLOSE_MS` (default 2000). A REMOVE
  fans out to the key's successors on every recent ring, the drainer and the
  absent owner, and is ok only if every target confirms in the call: while
  one of N members cools, about (`DFKV_RING_REMOVE_FANOUT` + 1)/N of the
  REMOVEs fail, against 1/N without the switch. The reroute acts only while
  every client of the group's MDS client registry carries the fan-out marker
  this release adds to its registration (`dfkv-remove-fanout=1;poll_ms=`);
  enable the switch on every registered client, readers included. Measured
  on a test ring of eight server processes (64-block prefixes, a miss
  recomputed and PUT again): a member killed for 90 s and back with its
  data, full-prefix hit rate in the first 10 s 36 % (0 % off), 88 % in the
  next 20 s (3 % off), 90 MiB recomputed instead of 149; killed for 3 s,
  49 % in the first 10 s (32 % off). With clients removing keys throughout,
  no key was served after its REMOVE returned ok while the reroute acted.
  New `dfkv_client_cooldown_reroutes_total{op}`,
  `dfkv_client_reroute_refused_total`, `dfkv_client_reroute_cleanup_total{result}`,
  `dfkv_client_remove_fanout_total{result}`,
  `dfkv_client_remove_unconfirmed_total{reason}`,
  `dfkv_client_registry_gate_open`, `dfkv_client_registry_gate_failures_total{reason}`,
  `dfkv_client_registry_unmarked_clients`. Off, routing, REMOVE and close are
  unchanged.
- MDS: `kListClients` is answered from a client-list cache, as member lists
  are, for `DFKV_MDS_LIST_CACHE_MS` but at most 1000 ms (member lists keep
  the value as before, up to 10000 ms). This holds for every caller,
  `dfkvctl clients` included: a client registered at another MDS can be
  missing from its answer for up to that TTL. New
  `dfkv_mds_client_list_cache_hits_total`.
- Client, GPU dedup: publishing fetched keys and copying hits synchronize
  the CUDA stream once per batch (per chunk of at most half the arena for
  publishes) instead of once per key. Measured on one GPU: 512 keys of 64
  KiB published in 2.6 ms instead of 6.9 ms and hit in 2.6 ms instead of 6.8
  ms; 64 keys in 0.37 and 0.34 ms instead of 0.91 and 0.99 ms.

- RDMA client: registering or growing memory with `RegisterMemory` no
  longer stops the transport's operations (TRN-17). Every rail's region
  was pinned under the lock every connection checkout and return takes:
  growing a region to 2 GiB took 666-850 ms on two rails, and an operation
  started meanwhile waited all of it (0.03 ms otherwise). The rails are now
  registered without that lock, one after another (registering them at
  once was slower: 620-683 ms against 411-417 ms for 1 GiB on two rails),
  and the lock is taken only to publish the new declaration, whole. A
  destroyed transport waits for a registration in flight.


- RDMA client: the device context stays open while the transport lives
  (TRN-16). Each rail's device context and protection domain are now held
  from the rail's first connection until the transport is destroyed. A
  transport that never called `RegisterMemory` closed them whenever its
  last connection went (idle reclaim, failures), and the next connection
  opened them again: opening an endpoint on a device not held took
  12-13 ms against 1.4-1.5 ms, and closing the last one 3 ms against
  1.5 ms. A transport that registered memory already held them. An Open
  that fails with ENODEV or EIO (device gone or reset) drops the rail's
  reference, so the device is opened again once its endpoints are gone; a
  transport with `RegisterMemory` anchors still needs a restart after a
  device reset, as before (DEPLOY.md).

- Server, read coalescing (`DFKV_READ_COALESCE=1`, REQ-20): a read nobody
  else is waiting for goes straight into its own buffer. The leader of a
  synchronous read allocated a scratch of the full capacity and copied it
  over, followers or not: measured about 600 us extra on a 1 MiB read
  (2.5 ms on 4 MiB), more than the read itself from the page cache. A fill
  that fails by throwing no longer leaves its flight registered, and a
  follower that times out takes a stuck flight out, so later identical
  reads no longer each wait the full timeout (500 ms by default) for it.
- Server, slab: extents can follow a change in the mix of value sizes
  (STO-35, `DFKV_SLAB_BALANCE=1`, off by default). When the share of the
  bytes of each size changes while every size class is still written (for
  example large values rising from a third to 80% of the bytes), the
  classes kept the extents they had: in a unit measurement the large class
  then held 72-77% of its most recent capacity's worth of writes and the
  small ones 100%. With the setting, each reclaim tick moves at most one
  extent from the class whose next victim is oldest to the one whose next
  victim is youngest, when the older is more than 1.1 times the younger;
  the three classes then held the same share (87%). A move never takes an
  extent with a slot in a read or write or a RETAIN-protected value, and it
  is the existing extent rebind (the table wipe is written before any
  value is forgotten, and kept by the page cache for a warm restart), so it
  is safe across a warm restart. New `dfkv_slab_class_tail_age{slot_size}`,
  `dfkv_slab_balance_moves_total`, `dfkv_slab_balance_max_hold_us`. The RAM
  tier is not affected.
- Client: absent grace (CLI-02 stage A), behind `DFKV_RING_ABSENT_GRACE_MS`
  (default 0 = off, cap 600000; MDS discovery only). A member that vanishes
  from an admitted view is kept as absent for the grace: no new writes, and
  REMOVE and RETAIN go to it as well as to the new owner; it is dropped when
  the grace passes (checked once a second on the poll thread) and owns its
  keys again if it returns first. An absent member is not read after a miss
  (a client that started while it was out of the view sends its REMOVEs to
  the new owner only, so it may hold removed copies); that read is the
  read-back of a later stage, under its rules. At most `max(1, N/8)`
  members are absent; a member whose last record was draining is not kept.
  A draining member stays its keys' second owner while another is absent
  (an absent member is looked up on a ring of its own).
  A REMOVE an absent member did not confirm succeeds on the write owner's
  confirmation and is sent again when the member returns (its return ends
  the PeerHealth cooldown the absence started, for that first send, so the
  retry is not skipped), then each second until confirmed, for up to the grace
  (`DFKV_RING_ABSENT_REMOVE_QUEUE`, default 4096 queued). What it buys is
  REMOVE correctness on a warm return: on a test ring of eight server
  processes with clients removing keys throughout, a member killed for
  90 s and back with its data served 4 removed keys off and none on. The
  hit rate is unchanged (a member's record deleted for 40 s: the same 0 %
  worst second and 27 MiB recomputed either way). New
  `dfkv_client_ring_absent_members`,
  `dfkv_client_ring_absent_dropped_total{reason}`,
  `dfkv_client_ring_absent_removes_{unconfirmed,retried,dropped}_total`.
  Off, routing is unchanged.

- Server, file store engine (`--store-engine file`, the explicit
  diagnostic and rollback engine; slab is the default): CLOCK eviction no
  longer evicts the object a PUT has just stored, and unread PUTs no longer
  evict each other for good (REQ-16). A new entry went in at the front,
  unreferenced, where the hand, sweeping toward the front, reached it right
  after clearing the residents' bits: under pressure on read objects a PUT
  returned kOk for an object already evicted, and with the hand at the
  front each unread PUT evicted the one before while the older objects
  behind the hand were never examined again. A new entry now goes in just
  behind the hand, so it is examined last, and the eviction a PUT triggers
  never takes that PUT's own object.
- Server, RDMA: `dfkv_rdma_rail_retry_exhausted_peers{dev}`, the distinct
  peer IP addresses whose connection on a device ended with RETRY_EXC or
  RNR_RETRY_EXC in the last 30 s (addresses, not QPs, so clients in
  separate pods of one host count separately; at most 64 a device), and
  the alert `DfkvRdmaRailRetriesExhaustedForManyPeers` (a maximum of 3 or
  more within 5 minutes: clients stop using the device soon after a burst).
  One address is usually that client; several on one device while the
  node's others stay clean point at that device's path, which a port check
  (ACTIVE) does not see. Observation only: placement is unchanged.


- Client: `GetAuto` into a `std::string` (used by `dfkvctl get`) no longer
  zero-fills its whole bound (64 MiB by default) before each read; that cost
  about 20 ms per call whatever the value's size.

- Client, host and GPU dedup: an exception on the calling thread between
  claiming a key and publishing or giving it up (an allocation failing, in
  practice) left the key claimed, and every other process of the host waited
  on it until the takeover age; the batch now gives up its claims on the way
  out. And a worker thread that could not be started after a batch had queued
  its shards no longer lets the batch return while a worker could still run
  them on freed memory.

- Client: a read that the node answered but whose bytes this process could
  not copy into a CUDA destination (TCP transport with a device destination)
  is `DFKV_ST_INVALID` instead of `DFKV_ST_IO_ERROR`, and no longer puts the
  node in cooldown.

- C API: `dfkv_batch_retain` refuses a nonzero `reserved0` and unknown
  `flags` (returns -1), as `dfkv_open_v2` does for its options; they were
  ignored.

- Client and MDS registration: stopping the member poller, the registrar and
  the latency probe can no longer miss its wakeup and wait a whole period
  (up to the 10 s heartbeat) in `dfkv_close`; the stop flag is now set under
  the mutex their loops wait with.

- Client, batch PUT over RDMA: a node's keys go out in shards, as the
  scatter-gather PUT and the reads do (`DFKV_READ_SHARD_KEYS`,
  `DFKV_READ_MAX_CONNS`), instead of all on one connection; and a transport
  that answers for fewer keys than it was given fails the rest instead of
  reading past its answer. Measured over RDMA, 1024 values of 256 KiB to one
  node: 0.74-0.95 GB/s on one connection, 3.6-3.9 GB/s sharded. As with
  the scatter-gather PUT, copies of one key in one batch can now go out on
  different connections, so which copy a node keeps no longer follows the
  batch order.

- Server, RDMA: a capability probe gives its pending-handshake slot back
  before it answers. It held it until its thread returned, so at the
  `DFKV_RDMA_MAX_PENDING_HANDSHAKES` limit the client's bootstrap, sent as
  soon as it read the answer, could be refused as if the server were busy.

- Client, RDMA: a scatter-gather GET reads several destination segments per
  READ. Each segment of a key's destination took its own READ and its own
  wait; one READ now scatters into as many segments as the device allows
  (`max_sge_rd`, at most the QP's SGE count), since the value's range on the
  server is contiguous. Measured, 16 keys of 61 segments from a RAM-tier
  node: 4 KiB segments 17.8-31.8 -> 3.1 ms, 64 KiB segments 22.8-29.5 ->
  15.9-19.1 ms, 16 KiB segments outside any registered memory 59-61 -> 34-36
  ms (each such segment is still registered for the call: 976 per call).
  Nothing changes on the wire.

- Design (no code): unavailable ring owners, in
  `docs/design/ring-unavailable-owner.md`. Behind switches that default off:
  a member that vanishes from the MDS view is kept as draining for a grace
  period; writes for an owner in PeerHealth cooldown go to the next member,
  with REMOVE fanned out to the key's successors; a server-side memory of
  recent REMOVEs (one new op); and a miss read back once from the previous
  owner, allowed only when that memory confirms the key was not removed.

- Server, RAM tier: promotion on a key's second read (REQ-12), off by
  default. With `DFKV_RAM_PROMOTE=second` a whole cold read is installed in
  RAM only when the key was read within `DFKV_RAM_PROMOTE_WINDOW_MS`
  (default 30000) before; a first read only remembers the key (a
  fingerprint, at most 262144 keys) and reserves nothing in RAM. A one-time
  scan then no longer pushes out values that are read again. Measured by
  replay on a 64 MiB tier with 256 KiB values, a hot set re-read while a
  scan streams past: 15.0% and 18.8% of hot reads hit RAM in two runs with
  first-read promotion, 100% in both with second-read. The cost is one
  more disk read per key: keys read exactly three times hit on reads 2 and
  3 (200 of 200) with first, on read 3 only (100 of 200) with second.
  `DFKV_RAM_PROMOTE=first`, the default, is promotion as before. DEPLOY.md
  says how to choose the window; new `dfkv_ram_promote_deferred_total` and
  `dfkv_ram_recent_reads`, only with `second`.

- Server, RDMA with io_uring: a request that arrives while a disk read is in
  flight is taken at once (RDS-11). The serve loop waited on the ring alone
  for up to 1 ms at a time and saw the request only then; measured, the
  request's own disk read started 612 us after it was posted instead of
  147 us. The loop now waits for RDMA completions, disk completions (an
  eventfd the ring signals) and Stop() in one poll().

- Server, RAM tier: the payload it holds is visible (REQ-13). New
  `dfkv_ram_payload_bytes`, the payload bytes of the resident values;
  `dfkv_ram_used_bytes` counts whole slots. With large values the arena
  holds much less than its size: an extent holds a whole number of slots
  and the rest is idle, so 17 MiB values fill about half of an arena of
  32 MiB extents (53%, 80% with 128 MiB extents) and 10 MiB values 63%
  (86%), measured in a single-size unit test. DEPLOY.md says when to raise
  `DFKV_RAM_TIER_EXTENT_BYTES` and what it costs.

- vLLM: the per-step connector metadata no longer carries two fields no
  worker read: the ids of all unfinished requests and each request's
  `is_last_chunk`. On a decode step with 1024 running requests and no
  request metadata the ids were 43 KB of the 43.1 KB pickled to every
  worker; with 32 requests to save or load and 512 running, 20 KB of 180 KB.

- LMCache: the health ping asks the ring. `ping` only checked that the
  client handle was open, so a ring with every node down or hung answered as
  healthy. It now GETs eight sentinel keys (one byte, no write; a node only
  looks them up) and needs one node's answer within 2 s; a probe still
  running is waited for by the next ping, not repeated, and the ping runs
  off the event loop. Measured on a two-node ring: a live ring answers in
  under 1 ms (about 12 ms on a fresh RDMA connection), a killed one fails in
  under 1 ms on TCP and within the timeout on RDMA, and a hung one
  (processes stopped, natively 10 to 51 s per call) within the timeout. The
  retention-limits call was the other candidate; it answers from what it
  learned at the first contact and kept reporting a dead ring as supported.
  Each ping adds eight GET misses to the client's and the nodes' GET
  counters and access logs. `close()` waits at most 5 s for a probe still
  in the native call; past that it logs a WARNING and leaves the native
  client open rather than close it under the call.

- vLLM connector and LMCache L2 adapter: a misspelt identity key fails
  startup. A `kv_connector_extra_config` key one or two characters away from
  `model_revision`, `require_model_revision`, `share_across_dp_sizes`,
  `tenant_id` or `enable_cross_layers_blocks` (vLLM), or an L2 adapter key
  that close to `model_name`, `model_revision`, `tenant_id` or
  `key_includes_model`, was not read, so its default applied and deployments
  meant to be told apart shared keys; it now refuses to start, with the
  rules HiCache applies since 2.42 (now shared in
  `dfkv_common.config_check`). vLLM integer settings that are not integers
  (`batch_concurrency`, `mds_poll_ms`, `recv_workers`, `load_window_keys`,
  `lookup_cache_ttl_ms`, `lookup_timeout_ms`, `lookup_rpc_port`) also refuse
  to start; `mds_poll_ms` and `batch_concurrency` were truncated before,
  and `lookup_rpc_port` must now be an integer (a token such as `dp0`
  refuses startup).
  Other unknown keys are not checked. Behaviour change: a configuration with
  such a typo that started before does not start now.

- vLLM: a failed load is not billed as a cache read. For a request with a
  retention TTL the connector reports `cached_tokens` and
  `cache_write_tokens` from its lookup; when the load failed and vLLM
  recomputed those tokens, the hit was still reported as cached. A
  request-level failure now reports only the local prefix hit, and a
  block-level failure the tokens before the first invalid block.

- Server, slab: the eviction watermarks can be set as an absolute headroom
  below capacity: `DFKV_SLAB_EVICT_HEADROOM_BYTES` (high = capacity minus
  it) and `DFKV_SLAB_EVICT_BAND_BYTES` (low = high minus it; default a
  quarter of the headroom). They take precedence over
  `DFKV_SLAB_EVICT_HIGH_PCT` / `_LOW_PCT` when set; the band is at least one
  extent and the headroom at least two bands; a resolved high below the
  percentage high or below the band (a small disk) keeps the percentages,
  with a warning. Unset by default, so 92/88 still applies; see DEPLOY for
  suggested values.
- Log engine (experimental): a PUT that fills its segment while the next
  one is being pre-opened waits for that header (one write) instead of
  reclaiming a segment inline. Since the pre-opened segment (STO-33), such
  a PUT found one free segment fewer than the cleaner keeps, reclaimed
  inline, and evicted values the cleaner would have relocated (values that
  had been read among them). `segments_free` now counts a segment being
  pre-opened, as the cleaner does. New `dfkv_lse_preopen_waits_total`,
  `dfkv_lse_preopen_wait_microseconds_total`.


- Client: an exact read of a key stored with another length leaves the
  destination as it was (TRN-26). Such a read is a miss, yet the stored
  bytes reached the caller's buffer: over TCP a shorter value or the first
  bytes of a longer one, over RDMA a shorter value (a longer one was already
  refused before the READ). A caller that kept or reused that buffer after
  the miss saw them. The client now marks its exact reads (`dfkv_get`,
  `dfkv_batch_get` and their `_st` forms); RDMA releases a value of another
  length without a READ, and TCP reads the reply off the socket into a
  scratch buffer. Results, statuses and lengths are unchanged; variable-size
  reads (`*_auto`) still receive a shorter value.

- Server, RDMA: the server can watch its rails' ports (RDS-02;
  `DFKV_RDMA_PORT_CHECK_MS`, off by default on the server; 1000 ms is the
  recommended setting). It took handshakes on a rail whose port was down;
  each client then failed only after RC retries (about 3.7 s per
  operation), the connections held their pins and receive slots until the
  10-minute idle timeout, and clients learned of the rail through the next
  heartbeat (up to 10 s) and their poll. With the check on, from a thread
  of its own that holds no lock a handshake or a serve loop takes:
  - a handshake on a rail whose port is not ACTIVE is closed before the
    reply. A client takes that as a busy server once (about 20 ms pause),
    then cools that rail of the server and uses another; a client with no
    other rail fails at once;
  - when a rail's port is not ACTIVE on two polls in a row, that rail's
    connections are ended, and no other rail's (a link that retrains
    within one poll keeps them); a connection registering meanwhile ends
    itself;
  - a change sends the MDS a heartbeat at once, at most one a second per
    rail; a change within the second is sent once it is over, if it still
    holds, so clients see the rail's state within their next poll;
  - a client that names no device gets the first rail not known down.
  New `dfkv_rdma_rail_port_active{dev}`,
  `dfkv_rdma_inactive_rail_refusals_total`,
  `dfkv_rdma_rail_down_connections_ended_total`,
  `dfkv_rdma_rail_port_changes_total`.

- Server, RDMA: a connection can write only into its own receive slots
  (RDS-13). The rkey a connection got covered the whole receive chunk, which
  holds other connections' slots: a client writing to a wrong address (a
  bug) silently changed another connection's PUT in flight (reproduced: a
  WRITE with one connection's rkey into another's slot completed and the
  bytes landed). Each connection's slots are now registered on their own
  and its rkey covers them only; a WRITE outside them fails with a remote
  access error and ends that connection. Slots are zeroed before they are
  given to a new connection, so a PUT that does not cover what it claims
  cannot store a previous connection's bytes. Receive chunks are no longer
  registered as a whole (nothing used those registrations but the rkey), so
  a chunk growth no longer registers 256 MiB under the device lock. No
  change for clients: they use the rkey and address from the handshake, as
  every release does. Startup still registers the initial chunk once on
  every rail (and releases it), so a rail that cannot register memory, for
  example with too low a `LimitMEMLOCK`, still refuses startup. Behaviour
  change: `dfkv_rdma_recv_segment_registered_rails` now counts the rails
  opened and checked at startup, not rails holding a chunk registration.
  DEPLOY §0 states the trust model.

- RDMA, client and server: endpoint setup costs less, and a large
  registration no longer stalls every rail (RDS-15). Measured on 400G
  InfiniBand, median of 40, depth 4: opening an endpoint took 4.05 ms and
  closing it 2.89 ms, now 1.19 ms and 1.49 ms (depth 64: 8.74 and 8.77 ms,
  now 2.51 and 1.86 ms). The control buffers are one page-aligned
  allocation with one registration instead of two per slot, and their NUMA
  bind now takes effect (it failed on unaligned heap memory). A device's
  port attributes and source GID are queried once and reused for a second
  while the port is ACTIVE (`ibv_query_port` took about 1.5 ms per open).
  Shared pool registrations (host KV pools, the RAM arena) no longer run
  under the process-wide device lock: registering 256 MiB took about 117 ms,
  and every endpoint open and close on every rail waited for it.

- RDMA: pipelined READs (RDS-06; client switch `DFKV_RDMA_PIPELINED_READS`,
  off by default). Every QP allowed one RDMA READ in flight
  (max_rd_atomic 1), so the READs of a GET window ran one after another on
  the NIC: measured on one connection, a window of four 1-MiB READs took
  179 us (23 GB/s) and sixteen 64-KiB READs 101 us; with sixteen in flight,
  127 us (33 GB/s) and 38 us. The server now accepts up to 16 READs in flight
  on every QP (responder resources only; a client that sends one at a time
  sees no change; measured, 20000 QPs at 16 are created as at 1, about
  2 ms each, and the device reports read resources for 16 on each of its
  131072 QPs, so none is taken from another) and advertises it in the probe
  answer (bit 6) when every rail's device accepts 16. A client with the
  switch on asks a server that advertises it (bootstrap bit 57) and raises
  its own depth to 16 (clamped to its device) only when the server confirms
  it for that connection (readiness byte 0, bit 2); otherwise it keeps 1,
  since more than the server's QP accepts would end the QP. A capability
  the client cached from a server since replaced by an older one under the
  same identity therefore costs one busy-peer retry (the older server
  refuses the bootstrap that asks), never a failed READ: a bootstrap that asked
  for one of these and was closed before any answer, taken as a busy server
  once, now also drops the peer's cached capabilities, so the next connection probes afresh. Before,
  whole-object GETs (which do not retry their connection) were refused on
  every new connection until a PUT or a control operation refreshed the
  cache; the session channel's request bit (58) had the same flaw. New `dfkv_rdma_client_pipelined_read_conns_total`. The path MTU,
  still the local port's active MTU, is logged per connection at debug
  level.

- Server, RDMA: a GET of a key the node does not have is answered
  `kNotFound` before any staging lease. It leased one first, read nothing,
  and under pool pressure waited for memory or evicted another client's
  idle connection to get it (reproduced: a miss evicted an idle QP). A GET
  served from the RAM arena through the prepared read gives its unused
  staging lease back with the READY instead of at the RELEASE. A lease
  release no longer takes the pool lock and walks the chunk's free list;
  measured with 8 to 64 threads leasing at once, a lease costs half as
  long and the pool serves 1.3 to 2.2 times as many (one thread: 0.39 µs,
  was 0.45 µs). New `dfkv_rdma_unstaged_misses_total`.
- Server, RDMA: receive-pool chunks are on transparent huge pages
  (`DFKV_RDMA_RECV_HUGEPAGES`, default on), as the RAM tier arena already
  was. Every GET staged from disk and every leased PUT registers its range
  of a chunk for that operation alone; measured on 400G InfiniBand, the
  registration and deregistration of 4 MiB took about 0.3 ms on 4 KiB pages
  and 0.05 ms on 2 MiB pages, of 64 MiB 2.4 ms and 0.4 ms (the transfer
  itself: 0.08 ms and 1.3 ms). Below about 1 MiB both cost 20-40 µs, the
  system call itself; avoiding that needs registrations that outlive one
  operation, which this release does not do.

- Server, slab disks: a damaged value is a miss, not a hit (STO-15). The
  slot table's record checksum covers the record only; a value whose bytes
  on disk were damaged, or never landed before a crash that a warm restart
  survived, was served as a hit. A PUT now stores a CRC32C of the value in
  the record (bytes 52..55, flagged in byte 9; older releases ignore both,
  so the format stays readable both ways). Every whole-value read checks
  it: GET, the buffered and direct read paths, and the RDMA lease's
  `Verify`. A mismatch drops the key, answers `kNotFound`, and counts in the
  new `dfkv_slab_payload_crc_failures_total`. Records written by older
  releases are served unchecked until they are rewritten; reads of a slice
  of a value are not checked, and the check runs at read time: a record a
  warm restart kept whose value never landed still answers EXISTS until
  its first whole-value read. The checksum costs one CRC32C pass per PUT
  and per whole-value GET: in a single-writer loop of 1 MiB values on NVMe,
  about 60-70 us per PUT at the median (+17%, single-writer throughput
  -12%) and 15-40 us per GET. `DFKV_SLAB_PAYLOAD_CRC=0` (default 1) turns
  it off: records are written without it and no read checks one; records
  written while it is off stay unchecked when it is turned on again. Memory:
  the store's per-key index entry grows from 4 to 12 bytes of value data
  (length, CRC, flag).

- SGLang HiCache: a parked retention TTL lapses with its TTL (HIC-28). A
  `batch_touch` that finds pages not stored yet parks their TTL for the
  write; it never lapsed, so a page written long after the touch was
  protected for a full TTL the request no longer needed, and keys whose
  write never came stayed until 2^20 newer ones pushed them out. The write
  now protects such a page for what is left of the parked TTL, not at all
  once it has run out, and lapsed entries leave the table.

- SGLang HiCache: `retain_precheck` reuses an admission for a tenth of its
  TTL (HIC-27). SGLang asks it for every request that carries a retention
  TTL, and each call built the object keys of the whole prefix and made two
  round trips (the ring's limits and a headroom check): for a 1563-page
  prefix, 7815 keys and about 4.3 ms of Python per call in a stand-in
  benchmark. An admission of the same prefix, TTL and key form is now
  answered from memory for a tenth of the TTL (the renewal interval
  `batch_touch` already uses), at most 30 s, in about 0.03 ms; a refusal is
  always asked again.

- SGLang HiCache: a key stored with another length is rewritten once
  (HIC-26). A read that found a key stored with another length than this
  process reads (a shorter or larger object, for example written by an
  instance with another page layout in the same namespace) missed, and the
  backup's existence gate then reported the key stored, while a node keeps
  an existing key's value on PUT, so the page was recomputed on every read.
  The connector now reads with the per-key status calls and records keys
  that read `DFKV_ST_LENGTH_MISMATCH` (or, on the device path, a hit shorter
  than the destination) in a bounded set. The next backup reads each again
  at the size it stores and, if it still mismatches, removes it and writes
  it again; a key that now reads correctly, is absent or cannot be read is
  left alone, and `DFKV_ST_INVALID` never leads to a removal. A key is
  rewritten at most once per process; later mismatches of it are only
  counted in the new `dfkv_hicache_length_mismatch_keys_total`, and each
  backend logs one WARNING. A steady rise means two layouts share a
  namespace: give them distinct `model_revision` values or set
  `bind_runtime_layout`. For the same reason, a rolling upgrade that
  changes object sizes without a new `model_revision` makes old and new
  instances delete each other's objects (once per key per process) instead
  of only missing them.

- Client, per-key read status: a value larger than the destination is
  `DFKV_ST_LENGTH_MISMATCH` on every transport. Over RDMA it was
  `DFKV_ST_INVALID`, the code also given for this process's block bound, a
  bad buffer and a node's refusal (TCP already answered
  `LENGTH_MISMATCH`). A transport now reports a stored length with a
  refusal only when it refused the value for its size. The plain calls and
  hits are unchanged; only the `*_st` codes move from 5 to 8 for this case,
  and every caller of them treats the two codes alike (a miss) except the
  HiCache heal above.

- Server: `SIGHUP` reloads `DFKV_TENANT_QUOTAS_FILE`; new limits apply from
  the next PUT, and a file that does not parse keeps the old quotas (WARN).
  The reload takes only the quotas lock: writes keep going, and one in
  flight is checked against the quotas it started with.
  SIGHUP used to stop the server (its default action); a server run in a
  foreground terminal without nohup now survives that session closing. New
  `dfkv_tenant_quota_reloads_total`, `dfkv_tenant_quota_reload_failures_total`.
  The MDS heartbeat interval is `DFKV_MDS_HEARTBEAT_MS` (1000-10000, default
  10000) instead of a fixed 10 s.

- Client, host and GPU dedup: the shared-memory segment goes with its last
  user. Each process holds a shared lock on it; the last one to leave
  removes its name. A segment (516 MiB by default for the host flavor) used
  to stay in `/dev/shm` for good, one more for every namespace a host ever
  ran, counted against the container's memory. A process of an older
  release takes no lock and can be left on a removed segment, unshared,
  until it exits. A segment that cannot be created for want of `/dev/shm`
  is now a WARN.

- Client, routing: the consistent-hash ring is a sorted vector instead of a
  map; a lookup took 33 ns on a 16-member ring and 55 ns on 64, now 18 and
  21 ns. Routes are unchanged. New `dfkv_client_ring_point_collisions`.

- `dfkvctl`: new `locate <key>` (the member a key routes to) and `rm <key>`;
  `put`, `get`, `exist`, `locate` and `rm` find the ring through `--mds`
  (and `--group`, `--ring-wait-ms`) as well as `--members`. An unknown
  option now exits 2 (it used to be taken for an argument), so a positional
  value starting with `-` must come after `--`; `--help` prints the usage.
  `dfkvctl` and `dfkv_smoke` ignore `DFKV_CLIENT_NODE_DEDUP` and
  `DFKV_CLIENT_NODE_DEDUP_GPU`: one command no longer creates a 516 MiB
  shared-memory segment. `dfkv_smoke` round-trips one key per member
  instead of one key for the whole ring before printing "N nodes OK".

- `dfkv_bench`: the connection settle before the measured phase now puts
  one key on every ring member (found by hashing candidates until each
  member owns one) instead of 32 fixed keys, which on a large ring left
  most members unprobed and their cold connections failing in the
  measured phase. New `KVClient::OwnerOf` and `MemberAddresses`.

- Client, RDMA with `DFKV_RDMA_NUMA=1`: the calling CPU's NUMA node is
  looked up in a table read from sysfs once, not by opening up to 256
  sysfs files on every connection acquire: 14.9 us to 0.055 us per lookup.

- RDMA, client and server: session channels (`DFKV_RDMA_SESSION_FD`, off by
  default on each side). The bootstrap TCP socket of a connection stays open
  as the connection's session channel, and whichever side ends the
  connection closes it. A client checks it (non-blocking `MSG_PEEK`) before
  reusing a pooled connection and drops one the server has ended (idle
  reclaim, receive-pool eviction, grant timeout, protocol error, stop)
  instead of spending its next operation on RC retry exhaustion: measured
  on InfiniBand, the first operation after an idle reclaim took 3.5-3.7 s
  without it and 9-13 ms with it. A server ends a connection as soon as its
  client drops it, instead of at `DFKV_RDMA_IDLE_MS`. Negotiated: a client
  asks (bootstrap bit 58) only a server whose probe advertises the
  capability, and the server acknowledges in the readiness byte only to a
  client that asked, so either side of an older release works as before.
  The keepalive skips (and drops) a pooled connection whose session is
  closed. A session the server closes between that check and the
  operation still costs the old RC retry timeout. One more descriptor per
  connection on each side: the client raises its soft `RLIMIT_NOFILE` to
  the hard limit when the switch is on; the server's `LimitNOFILE` must
  cover its connection count. The socket is close-on-exec from creation,
  but a plain `fork()` child (Python multiprocessing "fork") holds a copy
  until it exits, and for those connections the server falls back to
  `DFKV_RDMA_IDLE_MS`. A proxy on the bootstrap path that closes the TCP
  connection after the handshake makes the server end every connection.
  New `dfkv_rdma_client_session_closed_reaps_total`,
  `dfkv_rdma_sessions_kept_total`, `dfkv_rdma_session_peer_closed_total`.

- SGLang HiCache connector, device-direct (L2-bypass): a host-form
  `batch_set_v1` while primary KV is stored device-direct is logged and
  counted. It happens when the controller falls back to the host path after
  the device pool was registered, for example when a hybrid model's sidecar
  registration failed: the pages are stored under bare keys, while discovery
  probes the scatter-gather keys of the device form, so they are never
  found. The write is done as before; the first one logs a WARNING saying
  so, and `dfkv_hicache_host_writes_while_device_direct_pages_total` counts
  the pages.

- Server, slab: binding an extent to a size class (first fill, steals) no
  longer fdatasyncs slots.tbl while holding the disk's locks; its record
  wipe is made durable by the table syncs that follow anyway. Every read and
  write of the disk used to wait about 0.4-1.7 ms per bind. Warm restart is
  unaffected (it keeps a table only under the same boot id, where the page
  cache holds the wipe).

- Client, RDMA, rail selection:
  - The wait for rail credits is one budget per operation, its retries
    and rail changes included: `DFKV_RDMA_RAIL_BACKPRESSURE_MS`, now
    200 ms by default (at most the operation timeout), counted from the
    operation's first admission. Before, each admission waited 10 ms on
    its own, so with one credit per rail and 16 threads 25 to 57 of 320
    PUTs failed in a run of about 0.1 s; with the 200 ms budget none
    failed in three runs (0.11-0.17 s). While it waits, a non-local rail
    that frees serves the operation; local rails still go first.
  - A rail's latency estimate no longer counts connection setup (about
    12 ms against 0.2 ms for an operation: the rail that set up second was
    then left unused for the life of the process), counts operations above
    64 KiB per 64 KiB, and halves for every second without a sample.
  - A rail out of its cooldown is probed by the background thread with a
    keepalive to a node already connected to (up to four are tried), and
    application operations go to a healthy rail first, local or not;
    before, the next operation was the probe, ahead of every healthy rail.
    New `dfkv_rdma_client_rail_background_probes_total`.

- Client, RDMA: a local rail whose port goes down leaves selection. The
  client only checked port state at startup; a port that went down later
  kept being selected, and was given up only through the failures of the
  operations sent on it. The port of every
  rail is now queried every `DFKV_RDMA_PORT_CHECK_MS` (default 1000, 0 =
  off); a rail whose port is not ACTIVE is out of selection and its idle
  connections are closed, and it comes back when the port is ACTIVE again.
  A rail not ACTIVE at startup is still left out until restart. New
  `dfkv_rdma_client_rails_port_down`, `dfkv_rdma_client_port_down_events_total`.

- Client, RDMA: a peer rail that refuses our device is learned. Rails pair
  by device name, so a server that does not serve the device of our first
  rail closes the bootstrap before replying, which the client takes as
  "busy": a pause, then the other rail. Nothing was charged, so every
  operation paid that again (20 PUTs to such a peer took 1.16 s). When the
  operation then connects to the peer on another rail, the client dials the
  closed rail once more; if that is closed before a reply too, the rail is
  charged as an endpoint failure and cools like one (with the usual
  recovery probe); the same 20 PUTs take a few milliseconds. A peer out of
  handshake slots or stopping is not mistaken for a refusal, and a peer's
  last usable rail is never cooled this way. Peers without an MDS identity
  (static members) are tracked by address for this. New
  `dfkv_rdma_client_rail_refusals_total`.

- Client, RDMA: client QPs grant no remote READ or WRITE, and the memory a
  client registers (`dfkv_register_memory`, the host KV pool) is registered
  for local access only. Since 2.31 servers never write into client memory
  (a GET is the client's READ of server staging), so any connected peer
  that guessed a 32-bit rkey could write into the client's KV pool for no
  purpose. Servers keep their permissions; nothing on the wire changes.

- RDMA: the QP transport retry settings are `DFKV_RDMA_QP_TIMEOUT` (default
  14), `DFKV_RDMA_QP_RETRY_CNT` (7) and `DFKV_RDMA_QP_RNR_RETRY` (7, no
  limit) instead of constants; defaults unchanged. Measured on a ConnectX
  host, a SEND to a peer whose QP is gone ends with `RETRY_EXC` after
  3.5-3.7 s whatever the timeout (12, 14 or 16), not the 0.5 s the formula
  gives, so the retry budget is below the 5 s operation timeout but not
  far below it.
- Tests: two drift guards for duplicated connector code. The ctypes bindings
  of `libdfkv` are declared in the HiCache plugin, the vLLM connector, the
  LMCache connector and `dfkv_common`, and nothing kept them in step with
  each other or with `src/client/dfkv_c_api.h`; a test now loads each
  binding against a recording stand-in for the library and fails if a symbol
  is bound differently in two places, or with another argument count than
  the header's (`python_cabi_bindings`). The telemetry sync test compared a
  fixed list of files; it now also fails if the canonical or a vendored
  telemetry package holds a module outside that list, or if
  `deploy/sync_telemetry.sh` copies another list. The bindings agree today.

- SGLang HiCache connector, device-direct (L2-bypass): the host read
  `batch_get_v1` reports every page missing when primary KV is stored
  device-direct, and leaves the host buffer as it was. It cannot load those
  objects (layer-major scatter-gather groups) into a host buffer, yet it
  reported a stored page as present: a caller that then read the host buffer
  would have used whatever the buffer held as KV. The SGLang builds with the
  bypass never call it for device-direct KV (prefetch only records the
  request, discovery uses `batch_exists`, loads go through the device
  entries), so for them nothing changes; whether a page is stored is
  `batch_exists`'s answer. A WARNING is logged once if it is called.

- Server, RDMA: a leased PUT's lease slot is chosen among the slots free
  in the index space its windows name (their immediate), not taken as the
  receive slot its request happened to land in. With two leases granted on
  a depth-2 connection and their windows not yet sent, the next request
  lands in the first receive slot again, the first lease's: the server
  dropped the connection (reproduced). It now answers `kCacheFull` while
  no lease slot is free. A control op no longer fails the connection when
  a multi-window PUT is in progress on the data slot that happens to have
  its receive slot's number.
- Server, RDMA: receive-pool pressure reclaim evicts only idle connections
  whose leases make room for the request: a run of free ranges and idle
  leases that touch end to end in a chunk of the request's rail (or of
  none) and hold the request together (the fewest connections, then the
  least recently active), or else chunks it can empty, each holding only
  idle leases, until the budget can grow a chunk for it. It used to take
  the longest idle connection when none was big enough alone, on any rail:
  reproduced, a small idle control connection was evicted, the request
  still failed after a 1 s wait, and that client lost its connection. A
  request no eviction can serve now fails at once
  (`dfkv_rdma_pressure_unsatisfiable_total`), as does one larger than the
  whole budget. A connection that never sent a request, never quiescent
  before, may be evicted 10 s after its last activity (a client that gave
  up after the handshake). A growth the budget refuses releases empty chunks
  (as many as it needs) instead of after `DFKV_RDMA_RECV_CHUNK_IDLE_MS`, so
  an empty chunk of another rail (after a failover, say) no longer blocks
  it.
- Docs: `DFKV_RDMA_IDLE_MS` on the server must be at least three times the
  client's `DFKV_RDMA_KEEPALIVE_MS` (the defaults, 600000 and 15000, are);
  with one beat of margin a client that is throttled or paused for a moment
  finds a pooled QP reclaimed, and its next operation on it waits out RC
  retry exhaustion, measured at about 3.7 s, before it reconnects. With the
  keepalive off (`0`) every idle reclaim and every server restart costs that
  on the first use of each pooled QP.

- Drain (MDS-04, off by default): a node with `DFKV_DRAIN=1` can be put in
  a draining state, by `dfkvctl drain <node> [--undrain]` or by a stop with
  `DFKV_STOP_DRAIN_MS` set (it drains that long, then stops; a second
  SIGTERM ends the wait). A draining node keeps its ring position and
  serves reads; clients of this release send it no new writes (a key goes
  to its owner on the ring without the node, where it belongs once the
  node leaves), read it after a miss at that owner, and send REMOVE and
  RETAIN to both (a REMOVE succeeds when both confirm). A node drained by
  `dfkvctl` deletes its MDS record when it stops, so clients drop it within
  a poll instead of after its lease. The state rides a new optional member
  extension (DRN1), written only while a member drains; older clients,
  MDS and dfkvctl ignore it, and the MDS must be upgraded first for it to
  take effect: do not drain until every MDS of the group runs 2.43 or
  later (a heartbeat that reaches a 2.42 MDS rewrites the record without
  the flag, so the drain turns on and off between polls, and a REMOVE made
  while it is off reaches only the drainer: a removed key can come back).
  Known limit with older clients: their REMOVE of a key a new
  client wrote during the drain reaches only the drainer. New ops
  `kDrain` (node) and `kDeregister` (MDS); new metrics
  `dfkv_client_ring_draining_members`, `dfkv_client_drain_requests_total`,
  `dfkv_mds_deregister_requests_total`. See DEPLOY §4c-1, including
  `terminationGracePeriodSeconds` against `DFKV_STOP_DRAIN_MS`.
- CI: two new jobs. One runs the MDS, registrar, member poller, etcd client
  and client discovery tests against a real etcd (3.5.17, checksum pinned);
  they skip themselves without `DFKV_TEST_ETCD`, so no job ran them, and a
  skip for want of etcd now fails the job. The other runs the suites under
  AddressSanitizer and UndefinedBehaviorSanitizer. The release workflow
  now starts only after main's CI run of the tagged commit has passed
  (waiting for it if it is still running): it checked only that the commit
  is on main, and repeats a few of CI's jobs, not the sanitizer, etcd,
  RDMA, container and connector ones. Runs of pushes to main are no
  longer cancelled by a later push (PR runs still are), so that run stays.
- Release: the binaries are built with `-fstack-protector-strong`,
  `-D_FORTIFY_SOURCE=2` and `-z relro -z now` (new CMake option
  `DFKV_HARDENING`, on by default; a compiler that sets a fortify level
  itself keeps it, Debug builds get none), and the
  release carries their debug symbols in a separate
  `dfkv-<version>-linux-x86_64-debug.tar.gz` (`bin/<tool>.debug`,
  `lib/libdfkv.so.<version>.debug`; extracted over the release, gdb finds
  them through each binary's debuglink). The shipped binaries are stripped
  as before. The release workflow now checks the hardening and the glibc
  floor of what it ships, as CI's portable build does. The floor is now
  glibc 2.34 (it was 2.35): both reject a binary that needs GLIBC_2.35 or
  later, so the release runs on 2.34 hosts (RHEL 9 family).

- SGLang HiCache connector: object keys are built from affixes packed once,
  and a batch's keys share one native buffer. Every key was validated and
  packed in full (pool, coordinates, component), and every key got a buffer
  of its own for the native call; on a 1,563-page prefix of an MHA model
  that took about 7 ms to build the 3,126 keys and 2.8 ms to hand them to
  the native client, under the GIL, on every lookup, read, write and touch.
  The connector now caches `pool_key_affixes` per pool and coordinates and
  adds only the page hash field per key, and `make_key_array`
  (`dfkv_common`) packs a batch into one buffer: 1.2 ms and 0.5 ms for the
  same prefix. Keys are byte-identical.

- SGLang HiCache connector, models whose primary KV pool is a logical
  anchor: the one-byte marker writes and reads no longer register memory for
  RDMA on every batch. Their buffers were not registered, so the transport
  registered a one-shot memory region (`ibv_reg_mr`, counted in
  `dfkv_rdma_client_adhoc_user_mr_total`) for each operation; a registration
  failure also counts against the connection. Markers now use a 64 KiB
  scratch buffer registered once per client; a call with more markers than
  it holds uses a buffer of its own as before.

- SGLang HiCache connector, `batch_exists_v2`: all side pools are probed in
  one existence round trip. It made one round trip for the KV prefix and
  then one per side pool, one after the other: a model with five side pools
  paid six round trips per lookup. The side-pool keys are known once the KV
  prefix is, so they now go in one call: two round trips per lookup whatever
  the number of side pools, three on a ring with retention, where the probe
  of the side pools no longer renews and one REFRESH renews the pools that
  were judged. As in SGLang's reference, the pools after one that leaves no
  usable prefix get no verdict, and now are not renewed either. Results are
  unchanged.

- SGLang HiCache connector, EAGLE draft KV (L2-bypass): `batch_touch` and
  `retain_precheck` cover the draft's objects too. A draft page is written
  with the write's retention TTL, but a touch renewed only the target
  model's KV and side pools, so the draft's protection ran out first while
  the target's was renewed, and loaded pages then ran without their draft
  KV. The draft objects this rank writes (all ranks for a draft split over
  TP, the replica writer for a replicated one) are now renewed with the rest
  of the prefix, once this process has moved a draft page and so knows its
  layout. Because `retain_precheck` now includes them, a rank that owns
  draft objects on nodes without headroom vetoes the prefix's TTL like any
  other owner. Draft objects not stored yet are not parked for their write
  (draft pages are written best-effort and may never be), so they cannot
  fill the pending table.

- perf: CRC-32C over long inputs runs three instruction chains at once and
  joins them (x86-64 SSE4.2 and AArch64 CRC paths), 2.6 times the throughput
  of one chain (measured). Values are bit-identical; the run-time dispatch
  and the software fallback are unchanged. It is the log engine's payload
  checksum, on every PUT and whole-value GET.

- perf: log engine (experimental): a segment's new header is written without
  holding the store mutex, and the sealer keeps one client segment opened
  ahead of time. PUTs that finished their write no longer wait behind a
  header write (about 1 ms, once per segment opened or reclaimed). New
  metrics `dfkv_lse_segment_open_waits_total`,
  `dfkv_lse_segment_open_wait_microseconds_total`,
  `dfkv_lse_preopened_taken_total`. Design §6.7.

- perf: log engine index hash mixes every key field (experimental `log`
  engine only). Keys whose fields are related, such as `{id * C + 1, id}`,
  fell into a few buckets of the index, shadow and flight tables, so each
  PUT spent long in hash chains under the store mutex: 1-byte PUTs with 8
  writers had a p99 of 168 ms, now 2.5 ms. The hash is now seeded per
  process and mixes every field, as slab's index does.

- Server, TCP: the first frame of a connection gets the same payload
  allowance as later frames (1 MiB/s on top of `DFKV_TCP_FIRST_REQ_MS`), so a
  large first PUT on a slow link is not cut off at the first-request
  deadline.

- Server, RAM tier metrics: `dfkv_ram_evicted_unread_total` (values
  evicted for room without having been read from RAM),
  `dfkv_ram_evicted_age_ms_total` (their resident time, summed),
  `dfkv_ram_promote_declined_total` and
  `dfkv_ram_reservations_aborted_total` (promotions that did not end in a
  RAM copy).

- Server, RAM tier: values larger than an arena extent are no longer kept in
  RAM by default; they are written to and read from the disk.
  `DFKV_RAM_TIER_LARGE_RESERVE_BYTES` sets a budget for them and brings that
  back. By default two extents (64 MiB) of the RAM budget were kept for such
  values, which the default `--max-msg` (one extent) never sends; the arena
  gets them back. With a budget set, its values now give way oldest first,
  and finding one visits only them: before, every admission at a full budget
  walked the whole index of every shard under its lock and took victims in
  hash order (with 150 000 small values resident, such a PUT took 3.2 ms, up
  to 16 ms). A value removed and put again is as new as its last PUT.

- Server, RAM tier: a PUT's check of whether the disk already holds its key
  is made before the RAM shard lock, not under it. The disk store's check
  takes the store's lock, which a slow metadata write (`slots.tbl`) can
  hold; meanwhile every RAM hit of that shard waited. In a test where the
  check takes 300 ms, the slowest RAM hit of the shard took 299 ms; now it
  does not wait for it.
- MDS and dfkvctl: one step takes a node out of the ring, behind
  `DFKV_MDS_FENCING=1` (off by default; it adds `/fenced/<id>` records to
  the etcd layout, and every MDS of the group needs it).
  `dfkvctl member fence <id> --mds <eps> --group <g>` writes a fence record
  without a lease and deletes the node's member record: clients drop it at
  their next poll, its registers and heartbeats are refused (node log
  `is fenced`, `dfkv_mds_fenced 1`) while it keeps running, and member lists
  leave it out, until `dfkvctl member unfence <id>`; `dfkvctl member fenced`
  lists them. It took three steps (change the ring, stop the process,
  disable it), as a running or restarted node registered again within 10 s.
  Fence and unfence are unauthenticated, as registration is. With fencing a
  heartbeat may take five etcd requests, so the registrar's default I/O
  timeout is now 11 s.
- MDS: member lists are reused for `DFKV_MDS_LIST_CACHE_MS` (default
  500 ms, 0 = read etcd on every poll), and polls of one group that arrive
  while one is reading etcd wait for that read instead of each starting
  their own. A record this MDS writes is in its next answer; one written
  through another MDS appears at most that much later. With 50 clients
  polling every 3 s through 3 MDS, etcd ranges per client and minute went
  from 19.9 to 0.4. New `dfkv_mds_list_cache_hits_total`.
- Node and client registrar: it stays on one MDS while that one answers,
  and when it fails tries the others at once, in the same heartbeat,
  instead of rotating over them one heartbeat at a time. A connect to an
  MDS gives up after 2 s (it used to take the 7 s I/O timeout), and a failed
  MDS is passed over for 30 s (2 s, less than a heartbeat interval). Each
  registrar starts on the MDS its member id hashes to, so members spread
  over the MDS rather than all loading the first and failing over together. With one
  of three MDS unreachable a member used to go 27 s between renewals of its
  30 s lease, and with two 44 s, dropping out of the ring; now an
  unreachable MDS costs one heartbeat at most 2 s.
- MDS: a heartbeat no longer writes to etcd unless the member's record
  changed. It reads the record, renews the lease the record is attached to
  (any MDS may renew any lease), and writes only a changed record; a change
  of the stats (STA1) alone, which every beat of a serving node carries, is
  written at most every `DFKV_MDS_STATS_PUT_MS` (default 60 s, 0 = every
  beat). A record that is gone or whose lease expired is written under a new
  lease, as before. Every heartbeat used to be a Put, and behind a registrar
  rotating over three MDS also a lease grant, because each MDS's own lease
  had lapsed by the time the rotation came back to it. Measured with 10
  members heartbeating through 3 MDS against a real etcd for 3 minutes: per
  member and minute 6 Puts and 6 lease grants (6 etcd revisions) before, 1
  Put and no lease grant (1 revision) after. A member that stops heartbeating
  still leaves within the 30 s lease TTL (25.2 s measured, 25.3 s before).
  A heartbeat now takes up to four etcd requests instead of three, so the
  registrar's default I/O timeout is 9 s (was 7 s). Registrars of 2.35 to
  2.42 keep their 7 s against this path; that matters only while etcd is
  slow (four requests of up to 2 s each), when such a heartbeat may time out
  and be retried at the next beat. New `dfkv_mds_record_puts_total`,
  `dfkv_mds_record_puts_skipped_total`, and etcd request op `get`.
- Tests: `MetricsHttp.DescriptorExhaustionBacksOffAndRecovers` no longer
  waits for the lowest free descriptor number to come back after the queued
  scrape, which other descriptors of the process can move (it failed once
  on CI after waiting out its 3 s); it waits until the server has closed
  every descriptor it accepted. The metrics server now closes a
  connection's descriptor under the lock that removes it from its records.
- Client, RDMA: ENOMEM while building an endpoint no longer fails the
  operation at once as if the process were out of memory. It may be the
  device's own limit on QPs or CQs (RLIMIT_MEMLOCK is the process-wide
  case), so the other rails are tried with a new connection each; only when
  none can be opened does the operation fail with `kResourceExhausted`. No
  rail is charged and the peer is not cooled, as before. A rail that failed
  so opens no new endpoint for 2 s (its idle connections still serve);
  when every rail without an idle connection is in that state, operations
  fail at once without dialing
  (`dfkv_rdma_client_local_resource_held_total`): each attempt cost the
  peer a connection and a handshake it saw abandoned.
- Client, RDMA: an operation that cannot open an endpoint for lack of file
  descriptors (in `socket()` for the bootstrap connection, or in building
  the endpoint) is now served on an idle connection to the same peer on
  another rail, if there is one, instead of failing at once. A `socket()`
  failure also no longer marks the peer unreachable and holds off new
  connections to it. The rail it had chosen, without an idle
  connection, needed a new one; a connection already open on another rail
  needs no descriptor. No rail is charged either way. Counted in
  `dfkv_rdma_client_local_resource_rail_fallbacks_total`.
- Server, RDMA: stopping no longer waits for connections still in their
  handshake, and lets reads and writes in flight finish. A connection
  thread waiting for its peer's bootstrap frames held `Stop()` up until the
  bootstrap deadline (10 s; measured 10.0 s with three peers that had only
  connected, now under 1 ms): their sockets are now shut down. The receive
  pool's pressure wait (up to 5 s) also ends on a stop. Connections that
  have sent a GET's READY or a PUT lease and wait for the client go on
  serving, for up to `DFKV_RDMA_STOP_DRAIN_MS` (default 2 s, 0 = no wait),
  so the client's READ or WRITE completes instead of ending in a completion
  error; each ends as soon as it holds no grant. New GETs are answered as a
  miss and new leased PUTs with `kCacheFull` meanwhile, so the drain waits
  only for the grants in flight (`dfkv_rdma_stop_grants_refused_total`).
  The others are closed at once, as before. Stops that ran out of drain
  time count in `dfkv_rdma_stop_drains_cut_total`.
- Server, RDMA: a GET grant or a PUT lease no longer outlives a dead client
  by the connection's idle timeout. After sending a GET's READY the server
  holds the value (a staging range, or a pin on a RAM tier slot that then
  cannot be evicted) until the client's RELEASE, and after granting a PUT
  lease it holds the staging range until the client writes. A client that
  crashed or gave up sends nothing, so the connection, and with it the
  pin, stayed until `DFKV_RDMA_IDLE_MS` (10 min by default, forever with 0).
  Now a connection that holds a grant, has nothing of the server's in
  flight (no disk read, no reply) and stays silent for
  `DFKV_RDMA_GRANT_TIMEOUT_MS` (default 60 s, 0 = no bound), counted from
  its last completion or the last grant it sent, is closed: the QP and its
  memory registrations go first, then the pins and ranges, so nothing the
  peer could still read or write is freed under it. A batch GET's clock
  thus starts with its last READY, however long its reads take. Counted in
  `dfkv_rdma_grant_expiries_total`, with a WARN naming the peer. A client
  whose deadline (`DFKV_RDMA_OP_TIMEOUT_MS`, or the batch one) is longer,
  or 0, needs a longer grant timeout, or 0, as a single READ or write that
  outlasts it is cut.
- Server, RAM tier (unreleased since the flush-lane change): the number of
  flush lanes was read from the disk group before the group was
  constructed, so it was whatever that memory held; a start with the RAM
  tier on could fail with `std::bad_array_new_length` or run with a wrong
  lane count. It is the number of configured directories now, as for the
  flush threads.


- Client C ABI: per-key outcome codes. The batch calls report a key as hit
  or not (0/1); a miss, a node in cooldown, an empty ring, a full disk, a
  used-up quota, a refused request and an I/O error all read the same, so
  an engine could not tell "recompute" from "back off". New
  `dfkv_batch_put_st`, `dfkv_batch_get_st`, `dfkv_batch_get_auto_st`,
  `dfkv_batch_put_sg_st` and `dfkv_batch_get_auto_sg_st` take one more
  array, `out_status`, with a `DFKV_ST_*` code per key (the server's or
  transport's answer, or the client's: length mismatch, peer cooling, no
  route, bad argument). The existing calls and their results are unchanged.

- Client, host dedup: a key is handed to the processes waiting for it as
  soon as it has arrived, not when the fetching process's whole batch has.
  A waiter's budget (`DFKV_NODE_DEDUP_WAIT_MS`, 500 ms) covers one batch; a
  fetch whose last part was slow cost the waiters every key, which they then
  read themselves. In a test where the fetching process has half of 16 keys
  and hangs on the rest, the waiter used to wait out its budget and fetch
  all 16; now it copies 8 and fetches 8. GPU dedup is unchanged: its
  publication waits for one load fence per batch.

- Client, per-key cost. Every key's identity hashed the namespace again (a
  canonical namespace is about 175 bytes, two of the six SHA-256 blocks of a
  vLLM key) and recomputed the tenant hash; the client now hashes the
  namespace's part once: 1160 to 635 ns per key. Routing computed the key's
  MD5 under the ring lock that every routed key of every thread takes; with
  8 threads routing 200 000 keys took 335 ms, now 60 ms (one thread: 115 ms
  either way). Keys and routes are unchanged.

- Client, MDS discovery: a shrink the placement guard refuses no longer cuts
  the data path to the omitted peers. The guard keeps such peers on the ring
  (the typical case is etcd leases of live nodes expiring together), but the
  client retired their transport topology all the same; with
  `DFKV_RDMA_RAIL_TIERS` set, every request to them then failed
  `kNoCompatibleRail` until they registered again, and without it their
  pooled connections and remote rail health were dropped. Ring members a
  refused view omits now keep their last published topology and stay in the
  live peer set; they are retired once a view without them is admitted.
- Server, TCP: every request frame has a deadline, not only the first. A
  later frame trickled in a byte at a time (each byte inside
  `DFKV_TCP_IO_TIMEOUT_S`) held its connection and handler thread for good;
  it is now due within the IO timeout plus its payload at 1 MiB/s, from its
  first byte. The payload buffer grows as bytes arrive instead of being
  allocated and zeroed at the declared length before any data: 512
  connections declaring 32 MiB each no longer take 16 GiB.

- Server: PUTs of different keys no longer wait for each other. PUTs were
  serialized by 1024 lock stripes held across the RAM admission copy and,
  on a capacity bypass, the disk write; with 64 concurrent 1 MiB PUTs of
  distinct keys 4.5% of them waited for an unrelated key, about 1.15 ms
  each. The lock is now per key. New `dfkv_put_key_lock_waits_total` and
  `dfkv_put_key_lock_wait_us_total`.

- Server, read coalescing: two connections whose batches name the same
  keys in opposite order no longer wait for each other. Each serve thread
  registered the read of its first key (submitted only after its whole
  batch was prepared) and then waited for the other's read of its second
  key: both waited out `DFKV_READ_COALESCE_TIMEOUT_MS` (500 ms) and then
  read for themselves. A follower whose thread holds an unfinished read of
  its own now reads for itself at once; same-order convoys still coalesce.
  New `dfkv_read_coalesce_held_reads_total`.

- Server, RAM tier: a slow disk no longer holds up the flushes of the
  others. Flush queues were per RAM shard and a flush batch mixed the
  values of every disk; the batch returned when its slowest disk was done,
  so a disk that hung (without failing) stalled every shard, dirty data
  climbed to the RAM-ACK watermark and new keys of healthy disks got
  `kCacheFull`. Each shard now queues by disk, a batch holds one disk's
  values, a disk occupies at most all but one of a shard's flush workers
  (with more than one disk a shard gets at least two), and a disk's dirty
  values may fill at most twice its share of the watermark: past that only
  its own new keys are refused.

- vLLM connector, namespace: each cache group's object size
  (`logical_bytes_per_block`) is not in the namespace. It is not added now:
  that changes every key and so waits for the next layout-ID bump, which
  cold-starts every deployment anyway; until then an object of another size
  is caught by the exact-length load and a shorter one is removed. A test
  now fails if the layout ID changes without the field.

- vLLM connector, lookup cost. Every lookup key copied the key metadata
  and validated and encoded all of its namespace fields again; a full-hit
  lookup of 15 624 chunks took about 61 ms with one rank namespace and
  465 ms with eight, now 13 ms and 80 ms. Each namespace is now encoded
  once per lookup (new `dfkv_common.pool_key_affixes` and
  `pool_key_hash_field`; `pool_key` is built from them). Keys are
  unchanged.

- vLLM connector, scheduler step cost. A request's token ids, read only by
  KV events, were pickled to every worker every step: a 32k-token request's
  metadata was about 115 KiB, now 19 KiB without KV events. The scan for
  pending loads checked each unfinished request against lists of the
  scheduled ones; with 2048 of each it took 8.7 ms per step, now 0.1 ms.

- vLLM connector: a failed block-level LOAD drops the request's cached
  lookup. vLLM rolls back the requests holding an invalid block and asks for
  them again; the scheduler reused a completed lookup for
  `lookup_cache_ttl_ms` and answered the retry with the hit whose load had
  just failed, so the load could fail again until the entry expired (the
  README already said failed loads drop it; request-level layouts did).
  Requests whose allocated blocks include an invalid block now have their
  cached lookup discarded.

- vLLM connector, small fixes. The client took its TP rank from the
  environment (`DFKV_TP_RANK` / `LOCAL_RANK` / `RANK`, none of which vLLM
  sets): every rank labelled its metrics rank 0 and wrote the same
  access-log file; the worker now passes its TP rank, so metrics carry the
  real rank and each rank writes its own `<path>.r<rank>` access log (a
  collector that tails only `.r0` now sees rank 0 only). The retention SET of a
  SAVE is timed from its start (`save_retain_set` durations were always 0).
  `lookup_async` is parsed as a boolean (the string "false" turned it on;
  it now really keeps lookups synchronous).
  The hot-config watcher (all three copies) kept its stop event in
  `threading.Thread._stop`, which Python up to 3.12 calls from `join()`;
  it is renamed. Corrected docs: `require_rdma=false` does allow the TCP
  transport (the vLLM README said there was no fallback), and the C header
  no longer says SG vectors wider than `dfkv_max_sg_segs` fail (they are
  posted as several work requests).

- Server, RDMA: with the default inline threshold, a PUT over RDMA of an
  object larger than the server's `--max-msg` (default 32 MiB) is answered
  `kInvalid` at once. A client whose `DFKV_RDMA_MAX_BLOCK_BYTES` is higher
  (64 MiB, for example) sends such an object down the leased path, and the
  server dropped the connection without an answer: the client waited out
  `DFKV_RDMA_OP_TIMEOUT_MS`, or `DFKV_RDMA_BATCH_OP_TIMEOUT_MS` when set
  (5 s by default), counted a completion timeout and an I/O error, and
  cooled a healthy node, so the PUTs after it were skipped as well. The same
  happens with 2.35.7. The client takes `kInvalid` as not cached and keeps
  the node; the server counts `dfkv_rdma_leaseput_oversize_rejects_total`
  and logs a WARN with the peer address the first time and every 1024th.
  Paths that declare a connection by object size (an inline threshold of 0
  or at or above `--max-msg`, a retry on a fresh connection) still have the
  connection refused, and TCP still drops it: keep the client's bound at or
  below the server's `--max-msg`, and always against 2.42.0 or earlier.
- Server, slab store and RAM tier: the free slots the background pass keeps
  ahead of a size class's writers are bounded in bytes, at least 64 MiB and
  at most 1 GiB of slots (still at most a quarter of the class), instead of
  at least 64 slots. A class of 32 MiB slots kept 2 GiB of each disk, or a
  quarter of the class, evicted and empty ahead of need; one of small slots
  written fast could hold a quarter of its capacity free.
- Server, slab store and RAM tier: a size class that is not being written
  gives its extents to one that is. The background growth took extents only
  from classes holding more than eight, so a class written once and then
  only read, or not at all, kept eight extents of each disk for good, most
  of their slots free, while the classes being written evicted their own
  values.
  A class written since the last pass keeps its eight; an idle one gives
  its empty extents first and then, as before, only extents much colder
  than the growing class's own. The RAM tier keeps one extent with free
  slots per class instead of eight (its writes are memory copies, there is
  nothing to stripe), so a small arena no longer binds eight extents to
  every class it has seen; for the same reason a RAM tier class being
  written keeps one extent against the background growth, not eight.
- Server, slab store: a cold steal (a growing size class taking an extent
  from a much colder class) now takes the qualifying extent whose residents
  were used least recently, as the other extent steals do. It ordered the
  candidates by read heat and then by fewest useful bytes; read heat decays
  to nothing within some hundreds of operations, so it took the emptiest
  qualifying extent, however recently its residents had been read. And a
  write that finds no free slot of its class and no unbound extent now
  takes an extent another class holds empty before it evicts anything, if
  that class has written nothing for a cold window (a class that is being
  written keeps its striping width of extents with free slots); it evicted
  one of its own values when the other class was at its floor of eight
  extents.
- Docs: `docs/ARCHITECTURE.md` no longer says that a PUT through the RAM
  tier waits for its disk flush. In the default write-back mode it returns
  once the value is in RAM (RAM-ACK); `DFKV_PUT_ACK_MODE=disk` or a zero
  watermark makes it wait. At the dirty watermark new keys are refused
  with `kCacheFull`, not made to wait, as the configuration table said.
- Server, slab store: the periodic sync of the slot table (`slots.tbl`,
  every 100 ms) now keeps that cadence by default only with
  `DFKV_SLAB_WARM_RESTART=boot`, the one mode whose recovery depends on it.
  Under `off` (the default) an unclean exit resets the table anyway, and
  the sync is off. `auto` keeps a table only when the kernel and the mount
  survived, whose page cache holds every record write whether synced or
  not; it syncs every 5 s, only to find a writeback error of the table
  during the run rather than at the next stop or restart. A server killed
  under `auto` is far less likely to lose its cache because the kill fell
  into one of these syncs, and each disk saves most of the syncs' I/O.
  Clean stops and extent rebinds still sync the table;
  `DFKV_SLAB_TABLE_SYNC_MS` still sets the cadence explicitly.
- Server, slab store: a disk whose background thread cannot be started (the
  process or its cgroup at its thread limit) now fails to start like any
  other disk error, instead of ending the process. Starting the table sync
  thread and then failing to start the reclaim thread terminated the server
  (`std::terminate` on a joinable thread); failing to start the first one
  threw out of the store's constructor, past the code that closes the
  disk's files and releases its directory lock. The disk is left marked as
  in use, so its next start treats this one as an unclean exit.

- Server, storage: the on-disk format library of the log-structured engine
  (superblock, segment header, record header, summary, trailer, journal
  block; CRC-32C with the CPU instruction where there is one) is added with
  its tests. Nothing uses it yet.

- Server, storage: groundwork for the log-structured storage engine
  (`docs/design/log-structured-engine.md`), with no change for any existing
  configuration. `DFKV_STORE_ENGINE` accepts two experimental values that
  are refused unless `DFKV_LSE_EXPERIMENTAL=1`: `auto`, which chooses each
  directory's engine from what it holds (slab markers: slab; the file
  engine's `blocks`: file; empty: slab for now), and `log`, which is not
  built yet and refuses to start. A directory with both slab and log-engine
  files, or with files of no known engine, is refused under `auto` without
  being changed. New per-disk metric `dfkv_disk_engine_info{disk,engine}`.
  The per-disk slab watermark metrics are now looked up by disk, so a node
  whose disks run different engines labels them correctly; on such a node
  they are emitted for the slab disks only, retention metrics are the slab
  disks', and the registered self-description keeps the slab write mode.

- Server, storage: the log-structured engine's core, selectable only with
  `DFKV_LSE_EXPERIMENTAL=1` (`DFKV_STORE_ENGINE=log`, or `auto` on a
  directory that already holds a log store). Stores carry a development-
  format bit that later builds refuse until the format is frozen. This
  stage writes checksummed records to one open segment, keeps the index in
  memory and rebuilds it by walking every segment, journals REMOVE as
  durable kills (a REMOVE returns once every earlier kill is durable), and
  answers `kCacheFull` when full (no eviction yet). The open segment is
  synced every `DFKV_LSE_SYNC_MS` (default 1000), which bounds what a power
  loss can take of acknowledged writes. Whole-value reads through `Range`,
  `RangeInto` and `RangeDirect` are checked against the stored checksum;
  prepared reads (`RangeDirectPrep`: RDMA pulls and the RAM tier's direct
  promotion) are not checked yet, so the engine is not for real data. The
  slab engine refuses a log store's directory without changing it. Nothing
  changes for any configuration that does not select the engine.

- Server, storage, log-structured engine (experimental): segments are
  sealed. A full segment goes to a sealer thread, which syncs it, writes a
  summary of its records and a trailer; `Close` seals what is open. A
  restart loads the summaries and walks only the segments without a valid
  trailer (after a clean close, none), and seals what it walked. A sealed
  segment whose records are all removed is reused. The summary entry now
  holds the record's header length (development format: no older store to
  read).

- Server, storage, log-structured engine (experimental): eviction. A
  cleaner keeps a few segments free: it takes the oldest sealed segment,
  relocates the records read since they were written (within
  `DFKV_LSE_RELOCATE_PCT`, default 25, of the bytes clients wrote), evicts
  the rest, and reuses the segment; a segment whose reads have not ended
  within `DFKV_LSE_PIN_WAIT_MS` (default 5000) is skipped for now. A PUT
  that finds no free segment reclaims one itself. The cleaner starts only
  when a PUT first needs space after an open, so a restart keeps everything
  it recovered until then. The metadata journal is compacted once it holds
  four times its live entries. A configured capacity that changes grows or
  shrinks the store at the next open. New per-disk `dfkv_lse_*` metrics.

- Server, storage, log-structured engine (experimental): RETAIN, with
  slab's codes, rules and `DFKV_RETAIN_*` settings. The budget counts each
  protected value's on-disk footprint; a protected value is relocated by
  the cleaner rather than evicted, and evicted only as a last resort
  (`dfkv_lse_protected_evictions_total`). Protection is kept in the
  metadata journal without the call waiting for it, and comes back,
  re-clamped to the TTL cap, after a restart. On a node whose disks run
  different engines, the retention metrics sum the disks that keep
  retention.

- Server, storage, log-structured engine (experimental): several open
  segments (`DFKV_LSE_OPEN_SEGMENTS`, default one per 32 segments, at most
  four; a PUT goes to the fullest one it fits in), RAM-tier flush batches
  written back to back with one write per segment, and whole-value reads
  prepared for RDMA checked against the stored checksum before anything
  uses the bytes: a mismatch is a miss for the client, the coalesced
  readers and the RAM tier, and the engine stops serving that copy. Slab
  reads are unchanged.

## 2.42.0 — 2026-11-23

- Client, RDMA: the rail name a scatter-gather batch read reports (the
  `dev` field of the per-node read statistics) no longer repeats the rail
  once per key ("ib0->ib0->ib0..." for a batch of N keys on one rail). It
  names the sequence of rails the call used.
- vLLM connector: a request's SAVEs resume where its earlier steps left
  off. Each step enumerated the prefix from chunk 0, built every chunk's key
  and probed the existence of all of them, so a chunked prefill cost
  exists keys (and Python time) quadratic in the prompt length. Per request
  and cache group the worker now keeps how far this rank's chunks are known
  stored, by a PUT or a probe, and starts there. A chunk whose PUT failed,
  or that the store mask left out (a volatile tail), holds the mark back, so
  a later step still saves it; stripe positions and KV-event parents are
  those of a full enumeration, and recurrent-state groups keep no mark. A
  request with a retention TTL still enumerates its whole prefix, so its
  SET keeps re-protecting the early chunks (prefix closure). One
  change: a chunk below the mark that the ring evicts while the request is
  still running is not stored again by that request. Stand-in benchmark,
  1024 blocks in prefill steps of 32: 16896 keys probed (45 ms) before, 1024
  (7 ms) after; the same 1024 keys are put.

- vLLM connector: a lookup that backs off probes each object once. A
  recurrent-state group's load mask is true only at the candidate's tail,
  so a missing state there makes the lookup try a shorter candidate; each
  round probed (and built the keys of) the whole prefix again, in the
  scheduler's lookup path. The first candidate is now probed as before; if
  it fails and a group's load mask moves with the candidate (a recurrent
  state's tail, a window), that group's chunks below the hit are probed in
  one call, and the rounds are answered from what was found. Pure full
  attention needs no second probe. Hits are
  unchanged (400 randomized lookups: identical results). Stand-in
  benchmark, 128 LCM chunks with no state saved: 128 probes of 8384 keys
  before, 2 probes of 256 keys after; a full hit still takes one probe.

- vLLM connector: a LOAD split by `load_window_keys` no longer re-marshals
  its descriptors window by window. Slicing the flat descriptor batch gave
  plain lists, so each window rebuilt its pointer and size arrays segment by
  segment in Python, holding the GIL. A window of the batch now passes the
  batch's own arrays to the native call from the window's first key, without
  copying; the native call reads the same descriptors as before. Stand-in
  benchmark, 4096 keys of 61 segments in windows of 256: 102 ms of Python
  per batch before, 4 ms after.

- LMCache connector (in-process): lookups no longer queue behind I/O, and
  the synchronous `batched_contains` is supported. PUT, GET and existence
  probes shared one executor (`DFKV_CONNECTOR_GET_PARALLELISM` workers, 1 by
  default), so a lookup waited for any PUT or GET in flight. Existence
  probes now have a worker of their own; PUTs (and removes) and GETs each
  have `DFKV_CONNECTOR_GET_PARALLELISM` workers, so no deployment has less
  PUT concurrency than before. The synchronous native calls (`exists_sync`,
  the new `batch_exists_sync`, `remove_sync`) are admitted only while the
  client is open, and `close()` waits for those in flight before closing
  the native handle.
  LMCache's RemoteBackend, seeing no `batched_contains`, asked `contains`
  key by key: one round trip per chunk of the prefix. It now gets the same
  answer as `batched_async_contains` (exists-cache prefix, then one batched
  probe per `DFKV_CONNECTOR_BATCH_MAX_KEYS` slice). Stand-in benchmark: a
  contains check over 256 chunks went from 256 round trips (224 ms) to 1
  (1 ms); an exists probe issued during a 1 s PUT waited 959 ms before and
  1 ms after.

- LMCache connector (in-process): a GET no longer stalls LMCache's event
  loop when the CPU arena is full. The connector allocated its buffer with
  LMCache's default `busy_loop=True`, which retries in `time.sleep(0.1)`
  steps until memory frees: on the event loop that held every other
  request. It now asks with `busy_loop=False`, and a full arena is a miss
  (the connector's existing allocation-failure path). In a stand-in
  benchmark with an arena that frees after one second, the event loop was
  stalled 975 ms before and 12 ms after.

- Server, RAM tier: a PUT of a key whose RAM copy is still there after the
  disk evicted the key now reaches the disk. A value written from RAM stays
  in RAM as a written copy; when the disk evicted its copy (or the key was
  removed from the disk alone), a later PUT of the key was answered `kOk`
  as a duplicate of the RAM copy and never written, so the key was gone
  once RAM evicted it too. Now such a PUT drops the stale RAM copy and is
  written to the disk directly, as a PUT of a new key the RAM tier cannot
  take is. A PUT of a key whose RAM value is still to be written is a
  duplicate as before. No setting, API or wire change.

- MDS: what etcd answers when it refuses a request is now logged, and
  refused writes and damaged member records are visible as metrics and
  alerts. The MDS request path logged nothing: an etcd out of space only
  raised `dfkv_mds_etcd_request_errors_total`, and the error body
  (`database space exceeded`) was dropped. Now each failed etcd request is
  logged with its operation, HTTP status and the first 256 bytes of etcd's
  answer, at most one line per operation every 10 s (with the number of
  failures not logged). New gauge `dfkv_mds_etcd_write_failing_seconds`:
  how long writes (lease grant, keepalive, put) have been failing, 0 while
  they succeed or when none was attempted in the last 30 s; alert `DfkvMdsEtcdWritesRefused` at 10 s, before members
  drop out of etcd when their 30 s lease runs out. A member or client
  record that cannot be decoded is still skipped, but counted
  (`dfkv_mds_member_decode_errors_total`, alert
  `DfkvMdsMemberRecordsUndecodable`) and logged with its key. New alert
  `DfkvMdsGroupMembersDropped`: a group has fewer than 80% of its members
  of the last 15 minutes (`dfkv_mds_group_nodes`). No setting, API or wire
  change. See `docs/METRICS.md`.

- Client, MDS membership: the ring is no longer rebuilt on every view of a
  recovery ramp, and the first members back after an empty ring are no
  longer frozen in as the whole ring. A smaller view than the trusted
  reference used to be adopted at once unless it dropped more than
  `DFKV_MDS_SHRINK_GUARD_PCT` (50%) of it; after an outage, members
  re-registering one by one gave a run of such views, each adopted and each
  rebuilding the ring on an incomplete view. Such a view is now adopted
  once it repeats on two polls in a row, so a single member loss reaches
  the ring one poll (3 s) later than before, a steady decline of one
  member per poll is adopted only once it stops, and a member flapping
  between present and absent stays in the ring; clients still skip a dead
  member through their peer-health cooldown, so the cost is misses on its
  keys. After an empty ring had been
  believed, the first non-empty view (perhaps 2 or 3 members) became the
  reference at once, and the growth arm then held every larger view back
  until one repeated three times, leaving all keys on those members for
  about 16 s. The views of the returning members are now adopted as they
  come, and the reference anchors on a view that repeats three times; until
  then the last adopted view is the reference for shrinking, so one bad
  poll during the ramp does not shrink the ring. No setting, API or wire
  change. See `docs/DEPLOY.md`.
- Client, host and GPU dedup: on weakly ordered CPUs (aarch64) a copy out
  of a shared entry that raced a writer could pass its check and return
  torn bytes. The copy re-reads the entry's generation and the ring cursor
  after copying, and accepts the bytes only if neither moved; those
  re-reads were acquire loads, which keep later loads after them but not
  the copy's own loads before them, so the copy could read bytes written
  after a re-read that still saw the old values. A fence now orders the
  copy (and the identity and length reads) before the re-reads. x86 was
  not affected.
- Client, host and GPU dedup: a key that two processes on the host expect
  with different lengths is now reported. Dedup rests on one key meaning
  the same bytes in every process; keys that leave out a rank axis of a
  sharded payload break that, and so far nothing said so. Such a lookup
  already ignored the entry and fetched for itself; it is now also counted
  (`dfkv_client_dedup_length_mismatches_total`,
  `dfkv_client_gpu_dedup_length_mismatches_total`) and the first one is
  logged as a WARN. Keys that disagree only in content, not in length,
  cannot be detected this way.
- Client: a timestamp another thread or process wrote a moment after this
  one read its clock is no longer taken for one from the distant past. The
  clock was read before the lookup or lock that found the stamp, and the
  unsigned difference of the two then came out near 2^64 milliseconds. A
  node that had just answered was put in cooldown after a failure as if it
  had been silent for long (`DFKV_PEER_BAD_GRACE_MS` did not apply), and a
  dedup entry published at that moment was treated as expired: the lookup
  fetched the key itself and recycled the fresh entry, and a waiter gave
  up on it. Such a stamp now counts as current when it is at most one
  second ahead; a stamp further ahead (a torn write, a clock of another
  time namespace) counts as ancient, as before, so that an entry carrying
  it expires rather than being served for good.
- Client, batch reads: the one retry of missed keys (`DFKV_GET_MISS_RETRIES`)
  now covers only the transient misses it exists for, on all three batch
  read paths: a key the node reported absent (a block moving between the
  node's tiers), and a key whose read failed with an I/O error (a
  connection being replaced) while its node is still healthy. It used to
  re-send every key that was not a hit, also keys whose request was
  refused or whose length did not match, and keys of a node that had just
  been put in cooldown. A read error of a node that answered for other keys
  of the batch moments before (within `DFKV_PEER_BAD_GRACE_MS`) still
  leaves the node healthy and is retried: the client cannot tell a timeout
  from a lost connection, so such a batch can still take two operation
  timeouts. The variable-length batch read (`BatchGetAuto`) had no retry;
  it now has the same one.
- Client, retention: a node found without retention support is asked
  again after `DFKV_RETAIN_REPROBE_MS` (new, default 30000; `0` asks every
  time), and the existence fallback for such a node no longer holds up the
  client's other retention calls. The verdict was kept for the life of the
  client: a client that started during a rolling upgrade kept treating
  the upgraded nodes as unsupported (SET always `kRetUnsupported`, REFRESH
  always an existence check, `RetainLimits` and `RetainPrecheck` saying
  no). `RetainPrecheck`, which asks no node, keeps saying no until a
  retention call or `RetainLimits` has heard otherwise. And the fallback,
  a network round trip, ran while holding the lock that every other node
  group of the batch, `RetainPrecheck` and `RetainLimits` wait for.
- Client, retention (`dfkv_batch_retain`, REFRESH with a TTL hint): a
  REFRESH skipped by the client's debounce no longer reports a key present
  that is gone or has moved. A REFRESH that finds the key protected counts
  as a hit, and the client skips asking the node again for a tenth of the
  hinted TTL. That cache knew only the key: after the key was removed or
  written again through the same client, after its owner changed in the
  ring, or once the protection the node actually granted (the SET's TTL,
  clamped to the node's cap) had run out, the skip still said "present".
  The cache now also records the owner and the most the protection can
  last, a skip needs both to match, and a successful PUT or remove of the
  key drops it, also when it races a retention call still on its way (the
  reply about the old value is then not recorded). A node that restarted and lost its protections is not
  noticed until the skip interval has passed, as before.
- SGLang HiCache connector: concrete extra_config checks at startup. A key
  that is not one of `model_revision`, `tenant_id`, `kv_cache_dtype`,
  `prefill_cp_storage_layout`, `bind_runtime_layout` but differs from one by
  a character or two (`modle_revision`) is refused: it was ignored, the
  default applied, and instances that should be told apart shared keys. An
  integer key (`page_size`, `layer_num`, `dtype_tag`, `head_num`,
  `head_dim`, `dp_size`, `mds_poll_ms`, `batch_concurrency`, `rdma_depth`)
  that is not an integer is refused with the key named; `"dtype_tag":"fp8"`
  used to fail with a bare int() error. `"rdma_numa":"0"` turned NUMA rail
  selection on (any non-empty string was true); it is now off, like `0`,
  `false`, `no` and `off` for the other switches. When `page_size` is given,
  it must be the primary host pool's page size (SGLang's `--page-size`) or,
  under decode CP, its logical page (`page_size × dcp_size`); otherwise
  registration is refused with both numbers. The docs now state that
  HiCache's `rdma_depth` defaults to 1, not to the native default of 4.

- Connectors (SGLang HiCache, vLLM, LMCache), client metrics: every counter
  and gauge the native client declares is mirrored to the inference server's
  `/metrics`. The allowlist had fallen behind by 45 families, among them the
  rail failure signals: `dfkv_rdma_client_no_compatible_rail_total` (a rank
  that shares no healthy rail with a server), the
  `dfkv_rdma_client_remote_rail_*` health counters, the cross-rail retry
  counters, admission and pull failures, and
  `dfkv_rdma_client_rail_wc_errors_total{dev,status}`. They were only in the
  native snapshot. A test now scans the native sources for declared families
  and fails on one that is neither mirrored nor listed as deliberately not
  mirrored.

- SGLang HiCache connector and client: retention results and calls are
  counted. A retention SET's per-key result was logged at debug level only,
  so a denied, failed or unsupported retention went unnoticed.
  `dfkv_hicache_retain_keys_total{result}` now counts the keys of every SET
  by result (protected, miss = parked for the write, unprotected, denied,
  unsupported, error), and a denied, failed or unsupported key logs a
  rate-limited WARNING. In the native client `BatchRetain` was not counted
  in `dfkv_client_op_*`, so once a ring had retention the connectors'
  existence probes (REFRESH without a TTL) disappeared from `op="exist"`. A
  REFRESH without a TTL now counts as `op="exist"`; a SET or a REFRESH with
  a TTL counts as the new `op="retain"`. On rings with retention the
  `op="exist"` request and key rates therefore step up after the upgrade:
  the probes were always sent, they are only counted now, and the step is
  not a load increase. A REFRESH with a TTL whose keys are all skipped by
  the client's debounce counts as an `op="retain"` request although no RPC
  is sent.

- SGLang HiCache connector, models whose primary KV pool is a logical
  anchor: the anchor's one-byte markers get the retention a `batch_touch`
  asked for before they were written. A touch that finds a page not yet
  stored parks its TTL, and the page's write applies it; the marker write
  applied a TTL only when the write itself carried one. With the touch first
  and a backup without a TTL, the side pools were protected and the markers
  were not; an evicted marker makes the prefix a miss, so the protected side
  pools held capacity no read could use.

- SGLang HiCache connector, `batch_exists_v2`: the reported prefix is one
  every side pool can serve. Each pool's boundary was computed on its own
  and the minimum returned: with an ALL_PAGES pool complete up to page 6 and
  a TRAILING_PAGES pool (sliding window, recurrent state) whose window is
  complete only at pages 4 and 10, it reported 6, where the window is
  missing, so the load that followed missed. It now reports the longest
  prefix length at which every ALL_PAGES pool is complete and every trailing
  window is (4 in the example). On SGLang builds whose `PoolTransferResult`
  has `restorable_prefix_pages`, all such lengths are returned too: with a
  trailing pool they are not every length up to the longest, and the
  controller intersects them across ranks rather than taking the minimum of
  each rank's longest.

- LMCache connector, construction and the L2 adapter's size map. In
  LMCache's scheduler role the in-process connector failed with
  `AttributeError`: it read the engine config and metadata from the
  context's `local_cpu_backend`, which is then a stand-in without them; it
  now takes them from the context. A construction failure after the native
  client was open leaked the client, and a failed RDMA pool registration
  leaked the native handle; both are closed now. LMCache logs a
  construction error as a warning and runs without the backend, so the
  connector also logs it as an error. The MP-server L2 adapter kept the size
  of every key it ever stored when LMCache does no eviction (nothing removes
  an entry then: dfkv evicts on its own), so the map grew without bound; it
  is kept only when an eviction policy or `max_capacity_gb` is configured.
  Without one a store cannot be told from a re-store, so stores report 0
  bytes to LMCache's usage counters (which LMCache only uses for eviction);
  keys are still reported to its listeners.
  A partly failed store task now accounts and reports the keys it did
  store, and the map is updated under the adapter's lock.

- LMCache connector (in-process): protections the vLLM and SGLang
  connectors had. Request tags (`lmcache.tag.*`) are now part of the
  object key, as they are of LMCache's own key: chunks of equal tokens under
  other tags were the same object, so a request loaded a chunk stored under
  another tag (another user, or anything the tag stands for). Untagged keys
  are unchanged; a deployment that already sends tags gets new keys for its
  tagged requests (one cold start for those). A chunk larger than the RDMA
  client's block bound is refused at startup instead of failing every PUT
  (with a libdfkv that predates the bound query the check is skipped, with a
  warning). A PUT that stored not every key is counted as a failed `put` in
  the metrics (it was "ok"; only the access log, off by default, recorded
  it). A warning, once per process, when `PYTHONHASHSEED` is unset: vLLM
  then seeds the prefix hash chain at random and chunk keys change with
  every engine start. Retention and a real liveness probe are still missing
  and are documented as such.

- vLLM connector, pipeline parallelism: the scheduler's lookup checks the
  objects of the first pipeline stage only, and this is now documented and
  warned about. Each stage's cache groups list only its own layers, so the
  other stages' objects are in namespaces of their own and rank 0 cannot
  probe them. A prefix another stage has lost is admitted and that stage's
  load fails; with `kv_load_failure_policy="fail"`, vLLM's default, the
  request fails rather than recomputing. With PP > 1 and that policy the
  connector now logs a warning at startup recommending `"recompute"`.

- vLLM connector, object identity: `share_across_dp_sizes` (default off)
  leaves the data-parallel size out of the namespace and keys. Every DP
  replica writes the same bytes for a prefix (`dp_rank` is already -1 in
  every key), but the DP size was always bound, so deployments of different
  DP sizes, or one scaled to another, missed each other's objects. Turning
  it on changes every key (one cold start); the default keeps today's keys.
  The weights version fell back to the model path when neither
  `model_revision`, the engine's revision nor a hub commit hash was known:
  for a local directory that stays the same when other weights are put
  there, and objects computed with the old weights were loaded for the new
  ones. That now logs a warning at startup, or with `require_model_revision`
  fails it. Keys are unchanged.

- vLLM connector: which cache groups' SAVE must finish inside the model step
  is now decided default-deny. Only full attention (exactly
  `FullAttentionSpec` / `MLAAttentionSpec`, without a window or chunk size),
  whose blocks vLLM keeps with the request until it finishes, is saved
  asynchronously. Every other spec type is fenced, including subclasses and
  types the connector does not know. The old list allowed by default: a
  group was fenced only for `MambaSpec` or a `sliding_window` /
  `attention_chunk_size` / `rswa_window` field, so a new spec type that
  frees or rewrites blocks while its request runs would have been read
  after its blocks were reused, and stored under a valid key. Spec types of
  current vLLM affected: subclasses such as the hidden-state cache and sink
  attention are now fenced too (correct, a little slower).
## 2.41.0 — 2026-11-23

- Server, RAM tier: a REMOVE is no longer undone by a read of the key that
  was in flight. A whole-value read of a key that is only on disk installs
  a copy in RAM when it is done. If a REMOVE of the key returned while the
  disk read was in flight, or ran while the copy was installed between the
  RAM and the disk step of the REMOVE, the copy was installed anyway: after
  a successful REMOVE, EXISTS and GET found the key, and a later PUT of it
  was answered from RAM as a duplicate without reaching the disk. Now a
  copy is installed only if the disk still has the key, judged under the
  RAM lock (for the copied promotion and for the reserved, zero-copy one),
  and a REMOVE drops a written RAM copy of the key once more after the disk
  remove. A value still to be written is not touched by that second step.
  The check is of the value, not only the key: a copy is not installed
  either if the key was removed and written again during the read with a
  value that reached the disk without passing through RAM (a quota tenant,
  a capacity bypass); the copy would have hidden the new value. Each stored
  value now carries an in-memory generation (nothing changes on disk), and
  a reader that joined another reader's coalesced read does not promote
  the bytes it was handed under its own generation.
  No setting, API or wire change.
- Server, RETAIN: a SET for a value that is in RAM but not on disk is parked
  only when the value is still to be written, and is no longer lost when
  the write lands at the same time. A value in RAM as a written copy of a
  key the disk no longer has (evicted there, or removed from the disk
  alone) was answered `kProtectedNew` and its SET parked for good: no flush
  would ever apply it, and parked SETs that are never taken fill the table
  (2^20 entries), after which every SET for a RAM-only value is `kDenied`.
  Such a SET is now answered as a hit without protection, or by the disk
  if the write has landed meanwhile. A SET parked just as its value's
  write landed, after the flush had looked for parked SETs, stayed parked
  without effect; the SET is now applied at once in that case. New
  `dfkv_retain_pending_denied_total` (parked SETs the disk denied when they
  were applied) and `dfkv_retain_pending_entries` (SETs parked now). No
  setting, API or wire change.
- Server, RAM tier: a REMOVE or a PUT of a key whose whole value is being
  read from disk into a reserved RAM slot no longer waits for that read.
  Only the thread serving the read finishes the reservation, after its disk
  read and reply; on the same connection that thread may be the one that
  then handles a REMOVE or PUT of the key. A REMOVE waited for it with no
  limit (the thread hung, and a stop could not join it); a PUT waited
  `DFKV_RAM_PUT_WAIT_MS` (with `0`, for ever) while holding its key's put
  lock. Now a REMOVE cancels the reservation and returns (the read still
  answers, its copy is not installed), and a PUT goes to the disk, which
  already holds the key and answers it as for any key it has. Stock
  clients send these on separate connections; a peer that pipelines them
  on one could hang a server thread. No setting, API or wire change.
- Server, RAM tier: a value the disks can never store is no longer
  acknowledged from RAM and then lost. A PUT larger than a slab disk's
  extent was taken into RAM (answered `kOk` with RAM-ACK), and its flush
  got `kInvalid`; the tier treated that as a failed write, tried five
  times over 4.25 s, dropped the value as acknowledged data lost
  (`DfkvRamAcknowledgedDataLost`) and reported itself unhealthy for at
  least 30 s. Such a PUT now skips the RAM tier and gets the disk's
  `kInvalid` at once, as without a RAM tier. A flush the disk answers
  `kInvalid` or `kQuotaExceeded` all the same (neither should reach it now:
  tenants with a quota already skip RAM write-back) is a final answer of a
  healthy disk: the value is dropped at once, without retries or health
  change, counted in the new `dfkv_ram_flush_refused_total` and not as
  lost, and a waiting PUT gets that answer. No setting, API or wire change.

- Client, GPU dedup (`DFKV_CLIENT_NODE_DEDUP_GPU=1`): restarting the
  inference processes in place no longer uses up the GPU dedup registry.
  Each process holds one of 64 registry entries, and an entry is reclaimed
  when its process has died. A container restarted in place gets the same
  pids, so the entries its previous processes left behind looked like
  entries of the new processes (whose pids they carry) and were neither
  reused nor reclaimed: with eight ranks, the registry was full after
  about seven such restarts, and from then on every process ran without
  GPU dedup, with one INFO line. An entry now also records which process
  holds it (its start time and a random value of the process image), and
  one whose pid is in use by another process (or by a new image of this
  one) is reclaimed. Such an entry is also no longer read through the
  arena address it records, which belongs to an address space that is
  gone. Entries left by older releases record no such identity and are
  still judged by their pid only, so the first restart after an upgrade
  can still leave entries behind. The processes sharing the registry must
  share a PID namespace (one pod, or `hostPID`): with separate ones, a
  process can take another's live entry (dedup then comes and goes; no
  wrong data). Layout and segment name are unchanged.

- Client, host and GPU dedup: a process killed while it created the
  shared segment no longer switches dedup off on the host. The first
  process creates the segment and sizes it in `/dev/shm` (512 MiB by
  default on the host path, which takes a while); killed before it
  finished, it left a segment that every later process waited one second
  for and then ran without dedup, until the segment was removed by hand.
  A process now finds the segment unfinished, sees that the process named
  as its creator has died (or that the segment was never even sized), and
  finishes it. Two processes of this release starting at the same moment
  no longer both take themselves for the creator. Laying a segment out a
  second time, which can still happen with older releases on the host (an
  older process that starts in the first microseconds of a creation takes
  itself for the creator too), no longer rewinds the host ring: the
  second layout used to reset the allocation cursor under processes that
  were already sharing entries, so that new entries reused the bytes of
  published ones and a lookup could return another key's data. A segment
  whose older-release creator died after sizing it does not name its
  creator and is waited for as before. A process of an older release
  that starts while a current one is creating the segment (a few
  milliseconds) runs without dedup for as long as it runs; it does not
  try again. The processes that share a segment must share a PID
  namespace (one pod, or `hostPID`), as all of dedup's liveness checks
  compare pids. Layout and segment names are unchanged.

- Client, host dedup (`DFKV_CLIENT_NODE_DEDUP=1`) and GPU dedup: a shared
  entry that can no longer be copied no longer keeps its key from being
  shared, and fetches whose process died no longer fill up the table. An
  entry is unusable once the staging ring has overwritten its bytes (a
  batch larger than `DFKV_NODE_DEDUP_ARENA_MB` /
  `DFKV_NODE_DEDUP_GPU_ARENA_MB`, or a rank that falls behind by a full
  ring) or, on the GPU path, once its publishing process has exited. Until
  it expired (`DFKV_NODE_DEDUP_TTL_MS`, 5 s), every later lookup of the key
  found it first and fetched the key for itself, without waiting (since
  2.38.0, 2.39.1 on the GPU path) but also without sharing. The next lookup
  now fetches the key again in that slot and the others wait for it. A
  fetch whose process died stayed in the table until its own key was asked
  for again; keys that never were filled their eight-slot probe window
  with such entries, and other keys landing there ran without dedup. A
  claim of any key now takes over a fetch older than
  `DFKV_NODE_DEDUP_TAKEOVER_MS`, as a lookup of the same key already did.
  A fetch that is merely slow and is taken over loses its chance to share
  its result, as before. So may a fetch that is only just starting: until
  its claim has written its own time, the slot shows the time of the
  previous occupant, and a claim of any key in that moment can take it
  over (both then fetch, and at most one of them publishes). Mixed
  versions on one host are safe: the slots
  are taken the way an expired entry or a takeover already took them, and
  the layout and segment names are unchanged.

- vLLM connector: a scheduler lookup that rank 0 does not answer within
  `lookup_timeout_ms` (new, default 10 s; `0` = wait forever) is a miss.
  The scheduler asked rank 0 over a ZMQ REQ socket with no receive timeout:
  if rank 0's lookup server had exited or a retention call there hung, the
  scheduler loop blocked for good (in the default synchronous mode), aborts
  were not processed, and `close()` hung as well. The timed-out socket is
  replaced, since a REQ socket cannot send again before it has its reply.
  For one timeout period after an unanswered lookup, lookups are misses at
  once: they share one socket thread, so N queued lookups would otherwise
  stall the scheduler N times the timeout. The first lookup after the
  period asks rank 0 again.
  `lookup_async` is now documented.

- vLLM connector: a LOAD that is rejected (receive queue full, connector
  closing), cancelled, or fails unexpectedly reports only its own
  destination blocks as invalid. It reported the request's whole block
  table, including the blocks of its local prefix-cache hit. Those are
  shared with other running requests: vLLM rolled each of them back (with
  `kv_load_failure_policy=fail`, failed them) and dropped the blocks from its
  prefix cache, e.g. every request sharing a system prompt with one whose
  load overflowed the queue after a restart. Misses and short reads already
  reported their own blocks only. If a request's load geometry cannot be
  resolved, its whole block table is still reported. Block 0, vLLM's shared
  null block, is never reported.

- vLLM connector: two settings that reorder an object's bytes without
  changing its length are now part of the namespace. Under DCP or PCP,
  `cp_kv_cache_interleave_size` decides which tokens each rank's block
  holds (`i` tokens on one rank, the next `i` on the next); with
  `enable_cross_layers_blocks` on vLLM up to 0.26 hands the connector one
  tensor in its own layer order, where per-layer registration concatenates
  layers by name (`layers.10` before `layers.2`). Deployments differing in
  either shared keys, the exact length check passed, and a hit put the bytes
  in the wrong places. Each is bound only when it differs from the default
  (interleave 1, no cross-layer blocks): a deployment with the defaults
  keeps its keys; one with another interleave size under CP, or with
  cross-layer blocks, starts cold once.

- LMCache connector (in-process `RemoteConnector`), models with several KV
  layer groups: a GET buffer now holds every group. LMCache describes such a
  chunk with one shape per group and PUTs all of them; the connector
  allocated the GET buffer from group 0's shape alone. A full chunk did not
  fit and always missed, while `contains` kept reporting it present, so it
  was neither loaded nor stored again. An unfull chunk short enough to fit
  was a hit and came back shaped as group 0, its bytes read as that group's.
  Unfull chunks of several groups are now a miss (LMCache's reshape of an
  unfull chunk sets a single shape); chunks of one group are unchanged. The
  connector also refuses at startup what it cannot store correctly:
  `use_layerwise=True` and a `remote_serde` other than `naive` (LMCache then
  runs without this remote backend and logs why). Keys are unchanged: the
  namespace already bound every group's shape.
- Connectors (SGLang HiCache, vLLM, LMCache), `rail_affinity`: the fallback
  rails are spread over the rail list instead of being the next ones. A rank
  used its primary rail and, by default, the one after it. Neighbouring
  rails are often the two ports of one adapter or share a switch and fail
  together: when both were down on a cache server, the rank had no rail in
  common with that server and every operation to it failed
  (`dfkv_rdma_client_no_compatible_rail_total`), although the server's other
  rails were healthy; the same held for the rank's own host. Fallback k of N
  is now `rank + k * rails / (N + 1)`: with 8 rails and one fallback, rank r
  uses r and r+4. The number of rails per rank is unchanged
  (`rail_affinity_fallbacks`, default 1), so is the primary; which rail is
  the fallback changes. The fallback is chosen for failure independence, not
  locality: it can sit on the other CPU socket, so traffic on it may cost
  bandwidth across the sockets.
- SGLang HiCache connector, MLA with decode context parallelism (DCP): a
  rank layout under which some DCP shards would have no writer now refuses
  to start. The writer of DCP shard d is the attention-TP rank d, which
  gives every shard exactly one writer per attention group when the DCP rank
  is the attention-TP rank modulo the DCP width, as SGLang builds it
  (checked against its rank derivation). Under another mapping, for example
  DCP coordinates given in the extra_config in blocks, some shards were
  never written while backup reported them done, so they never hit. A rank
  whose DCP rank is not its attention-TP rank modulo the DCP width is
  refused with the coordinates in the message. Deployments with SGLang's own
  coordinates are unaffected.
- SGLang HiCache connector: a side pool with one object per page is taken as
  the same on every TP rank only for an MLA model. For any model it was
  stored under one key shared by all TP ranks. An MHA model's single-buffer
  side pool can hold each rank's own heads: the sparse index-K pool of a
  model with sparse attention splits its index heads over TP. Every rank
  then wrote its heads under the same key, the last write won, and a rank
  read another rank's heads back at the right size as a hit. Such pools of
  non-MLA models are now stored per TP rank and written by every rank, and
  the same holds for device-form sidecars of L2-bypass; a device sidecar
  registered after its host pool under the same name keeps the host pool's
  layout (registrations that disagree are refused). Their keys change: one
  cold start for those pools. Stored per rank they take tp_size times the L3
  capacity they took under one shared key. MLA models are unchanged.
- SGLang HiCache connector, device-direct (L2-bypass): the ordering fence
  after a read orders the GPU the bytes landed on. It synchronized the
  calling thread's current CUDA device, which is per thread; SGLang's
  storage threads do not set it, and a thread that never did is on device 0.
  On a rank whose GPU is not 0 the fence therefore waited for GPU 0 and not
  for the GPUDirect writes, which can leave kernels reading KV that has not
  arrived (all pages hit, output degraded); the native GPU dedup's fence
  before it publishes had the same assumption. The CUDA device is now
  recorded when the device pool is registered (from the pool's buffers if
  they are CUDA tensors, else the registering thread's current device), each
  device-direct read runs the native call with that device current, and the
  fence synchronizes only it. `DFKV_GPU_LOAD_FENCE=0` still turns both
  fences off; the read still runs on the recorded device. Without CUDA
  nothing changes.
## 2.40.0 — 2026-11-23

- vLLM connector, hybrid models (recurrent state or sliding window next to
  full attention): a step that saves no longer waits on the model thread
  for every PUT. It waited for the whole send queue -- exists probe, PUT
  and retention of every group of every request saving in that step, and
  whatever earlier steps had left queued -- while only the state and window
  sources can change under a running request. The full-attention blocks
  stay with the request until it is reported finished sending. A SAVE is
  now split: the state and window groups go to a send worker of their own,
  and the step waits for exactly those; the attention part runs on the
  regular send worker and overlaps the following steps. A request is
  reported finished sending only when both parts are done, and each part's
  failures are counted on their own (a failed state PUT no longer fails the
  attention PUT of the same batch, and the reverse). The regular send queue
  can now fill while state PUTs still go through; an attention part it
  rejects is counted as the `save_rejected` operation, and that prefix is
  not admitted by LOOKUP. KV events of attention groups arrive when their
  PUT completes, possibly a later step than the state's.

- vLLM connector: a recurrent-state layer with several state tensors is
  saved and loaded whole. In vLLM 0.26.x and earlier such a layer (Mamba2,
  GDN, KDA, ...) reaches the connector as a list of views into one page, one
  per state (conv state, SSM / recurrent state); the connector transferred
  the first only. On a hit the other states kept whatever the target block
  held, and the model continued from a wrong state with nothing reported.
  Every tensor's bytes are in the object now. Objects saved before hold the
  first tensor only under the same chunk hashes: a load would reject them on
  length, but a SAVE finds them present and never replaces them, so those
  groups use a pool of their own (`mamba-states`) and start cold once. From
  vLLM 0.27 a layer is one view over its whole page (the layer splits it
  into its states itself); that was saved in full before, and its keys are
  unchanged. A deployment that ran 0.26.x with an older connector and
  upgrades vLLM can still meet first-tensor objects under the `mamba` pool,
  if its namespace did not change with the upgrade (the MambaSpec fields
  are the same in 0.26.0 and 0.27.0). A load that finds a shorter object
  under its key (by the stored length the server reports) now removes it,
  in the background and best effort, on rings with the remove RPC
  (`load_remove_stale` in the connector's operation metrics), so the next
  SAVE writes it again: one failed load per such object instead of failed
  loads until the server evicts it. A longer object is a miss and is left
  alone.

- Server, RAM tier: a disk that is healthy but has no slot for a value is no
  longer treated as a failing disk. The flush of a RAM value now gets the
  disk's status instead of a yes or no. A slab disk answers `kCacheFull` when
  every slot of the value's size class is held by a read or write in flight
  or protected by retention (`dfkv_slab_put_no_slot_total`). The RAM tier
  took that for a failed write: an acknowledged value was tried five times
  over 4.25 s and then dropped, counted as acknowledged data lost
  (`dfkv_ram_post_ack_flush_failures_total`, alert
  `DfkvRamAcknowledgedDataLost`), and the tier reported itself unhealthy
  (`/healthz` 503) for at least 30 s; a disk-ACK PUT got `kIOError`, on
  which clients cool the node down. Now such an attempt is neither a failure
  nor a retry and does not touch the tier's health. An acknowledged value
  stays in RAM, readable and dirty, and is tried again with the same pauses,
  at most 3.2 s apart, for up to 60 s; if the disk has no slot for it all
  that time it is dropped like an evicted value and counted in the new
  `dfkv_ram_flush_no_room_dropped_total`, not as lost. A disk-ACK PUT is
  answered `kCacheFull` at once, as the disk answers it without a RAM tier.
  While the tier drains for a stop there is no waiting for room: such a
  value gets `flush_retries` (5) more attempts without pauses, as a failing
  one does, and is then dropped and counted the same way. Attempts the disk
  refuses this way are counted in the new `dfkv_ram_flush_no_room_total`.
  A RETAIN SET parked for a value that is only in RAM (answered
  `kProtectedNew`) is now applied when the value is written, also after
  refused or failed attempts; it used to be thrown away on the first
  attempt that did not succeed, so the value landed without protection and
  could be evicted early. This was true for retried write failures as well
  (2.39). A parked SET is forgotten when its value is dropped unwritten or
  removed. Failed writes keep the behaviour of 2.39
  (retries with pauses, drop, loss counter, health). Limit: while values
  wait for room they count as dirty, so a disk without room for long fills
  the RAM-ACK watermark like a disk that refuses writes, and new keys are
  then answered `kCacheFull`. No setting, API or wire change. See
  `docs/DEPLOY.md` and `docs/METRICS.md`.
- Server, slab: two more answers of a healthy store that cannot take a key
  yet are `kCacheFull` instead of `kIOError`, on which clients cool the
  node down. A PUT of a key that still holds a slot from an earlier
  operation that has not let go, and a PUT of a key whose removal waits for
  a read or write in flight that outlasts the 5 s bound. Both used to be
  `kIOError`. `kCacheFull` has been a wire status since the first release;
  clients of 2.35 and later count it as "not stored" and keep using the
  node, as they do for a full disk. Counted in the new
  `dfkv_slab_put_busy_total`, so a pin or slot that never lets go shows;
  not in `dfkv_slab_put_no_slot_total`: neither is a missing slot. No
  setting, API or wire change.

- Client, GPU dedup (`DFKV_CLIENT_NODE_DEDUP_GPU=1`): a write or remove of
  a key that hangs on a blocked shared slot is no longer silent, and one
  such case no longer hangs. Before a write or remove, the client removes
  the key's shared entries; a slot held by a party it cannot take the slot
  from is waited for without limit, because that party may still publish
  the bytes from before the write. Those parties are a publisher of an
  older release on the same host and a process killed in one of the short
  windows listed for 2.39.1. Once the wait for one slot exceeds
  `DFKV_NODE_DEDUP_TAKEOVER_MS`, the client now logs one WARN line
  (`gpu-dedup: invalidation, slot ...`, with the slot's state) and counts
  it in the new `dfkv_client_gpu_dedup_invalidate_stuck_total`; it keeps
  waiting. One of the windows is no longer waited for without limit: a
  process killed while completing the release of a slot taken from a
  publisher, after its last decisive step. Such a slot can only end empty,
  never published, so once it has stayed in that state for longer than
  `DFKV_NODE_DEDUP_TAKEOVER_MS` the write or remove goes ahead (also
  logged and counted; a process that is merely completing the release is
  waited for and not counted). The slot itself stays blocked for lookups
  of the key, which fetch for themselves. The counter is in the native
  metrics snapshot.

- SGLang HiCache connector: new extra_config switch `bind_runtime_layout`
  (default `false`). Objects are found by key, and a read accepts an object
  only if its length is exact, so a reader with another page size, layer
  count, element width or TP/PP/CP split misses. It did not miss when the
  byte format differs at the same length: an MHA host pool in `page_first`
  (token-major) and one in `page_first_direct` or `page_head` (layer- or
  head-major), and an MLA host pool in `page_first` and `page_first_direct`,
  give pages of the same size in different orders, as element types of one
  width (bf16 and fp16) and pool classes that pack rows of one width
  differently do; such pages were served as hits with the wrong bytes,
  unless `model_revision` or `kv_cache_dtype` in the extra_config told them
  apart. With the switch on, a tag of the pool that lays out an object goes
  into the pool field of its key. Host-form keys are tagged by the host pool
  (class, `layout`, `dtype`, objects per page); the scatter-gather objects
  of L2-bypass by the device pool (class, `dtype`, and for the primary KV
  the per-token strides and layer count), so instances that differ only in
  host layout still share them. A reader with another format misses. A pool
  without one of these fields is reported by a WARNING naming it. It is off
  by default because it changes every object key: when it is turned on,
  nothing stored before is found (one cold start), so turn it on for all
  instances of a namespace together; while it is on, an SGLang upgrade that
  renames a pool class is one cold start as well. Not covered: another
  weight version or weight quantization under the same served model name
  (the connector cannot see the weights; set `model_revision`), and fp8
  variants where the pool's `dtype` is the storage type (uint8). See
  docs/CONNECTORS.md §2.2.
- Connectors (vLLM, LMCache, SGLang HiCache): the GPU dedup counters added
  in 2.39.1, `dfkv_client_gpu_dedup_load_fences_total` and
  `dfkv_client_gpu_dedup_load_fence_failures_total`, are now forwarded
  like the other dedup counters: to the metrics push as
  `dfkv_connector_gpu_dedup_load_fences_total` and
  `dfkv_connector_gpu_dedup_load_fence_failures_total`, and to the
  inference server's metrics endpoint under their native names. They were
  in the native snapshot only. `dfkv_client_gpu_dedup_invalidate_stuck_total`
  is forwarded the same way once the native client reports it; a counter
  the native client does not report is not sent.

## 2.39.1 — 2026-11-22

- Client, GPU dedup (`DFKV_CLIENT_NODE_DEDUP_GPU=1`): a process killed while
  it published a key no longer blocks that key, except in the cases listed
  at the end of this entry. A publisher holds its
  shared slot locked while it copies the fetched bytes into the staging
  arena and synchronizes the stream, which takes milliseconds or more
  under load. Killed there, it left the slot locked for good: every later
  write or remove of the key spun forever in the client, and every lookup
  of it waited out `DFKV_NODE_DEDUP_WAIT_MS`. The host-memory path got a
  remedy in 2.38.0; the device path now has the same one. A publisher
  marks the slot as its own while it copies and commits in a way that can
  be refused. An invalidation takes the slot from a publisher that is
  copying, whether it is alive or not (what it was about to publish is from
  before the write), and a lookup does the same once the publisher has been
  copying for longer than `DFKV_NODE_DEDUP_TAKEOVER_MS`; a publisher that
  lost its slot writes nothing more to it. A process killed while it takes a
  slot away leaves a second mark; once that is older than
  `DFKV_NODE_DEDUP_TAKEOVER_MS`, an invalidation or a lookup completes the
  release for it, and a process that was only slow and comes back finds
  the slot no longer its own and writes nothing. The shared-memory layout,
  the slot states and the segment name are unchanged: the marks are values
  that no release ever writes for another reason, kept in a slot field that
  is otherwise read only once an entry is published. After a slot is taken
  away it reads empty until the next fetch claims it, so a process that
  was waiting for the key may fetch it itself (counted in
  `dfkv_client_gpu_dedup_wait_timeouts_total`, one more read of the key).
  Known cases that still block the key (a write or remove of it then does
  not return, and lookups wait out `DFKV_NODE_DEDUP_WAIT_MS`): a publisher
  of an older release on the same host, which cannot be refused and is
  waited for as before; and a current process killed in one of a few
  windows of a few instructions each: after it locked the slot and before
  it marked it, inside its commit or its giving up after a CUDA error
  (between the generation swap and the last store), after it took a slot
  away and before it marked that, or, completing a release, after its swap
  and before its last store.
- Client, GPU dedup (`DFKV_CLIENT_NODE_DEDUP_GPU=1`): a lookup that another
  process on the host is fetching no longer waits out
  `DFKV_NODE_DEDUP_WAIT_MS` (500 ms by default) when that process has
  nothing to share. The host-memory path got this in 2.38.0; the device
  path still treated only a published result as final, so a batch waited
  the full time, once, whenever the owner of any of its keys gave up (a
  miss, a node in cooldown, an I/O error, or a device that could not be
  synchronized before publishing). A waiter now returns a key at once when
  no fetch is in flight for it any more, or when the published entry is one
  it cannot use (expired, overwritten, larger than its buffers), and the
  caller fetches those keys in one direct batch as before; the keys that
  were published are still delivered. `dfkv_client_gpu_dedup_wait_timeouts_total`
  keeps counting every wait that ended in a direct fetch, early or not.
  Mixed versions on one host are safe: only the waiting side behaves
  differently.
- Client, GPU dedup (`DFKV_CLIENT_NODE_DEDUP_GPU=1`): the process that
  fetched a key for the others published it before the fetched bytes were
  known to have reached device memory. A GPUDirect read writes the caller's
  device buffers directly, and a completion from the NIC does not order
  those writes before CUDA work issued right after it; that is why the
  connectors synchronize the device after a load before they use the
  buffers. The publish is such CUDA work (a device copy from the buffers
  into the shared staging arena) and ran before the connector's
  synchronization, so the other processes could be handed whatever the
  buffers held before the write arrived, while the fetching process itself
  saw correct data. The client now synchronizes the device context once per
  batch after the fetch and before it publishes anything. If that fails
  nothing of the batch is published and the others fetch for themselves.
  `DFKV_GPU_LOAD_FENCE=0`, which switches the connectors' fence off, also
  switches this one off. Cost: one `cuCtxSynchronize` per fetched batch in
  the fetching process. It waits for all work in the context, the
  framework's kernels included; the connector's own synchronization right
  after it then finds nothing left to wait for. That time comes out of the
  waiting processes' `DFKV_NODE_DEDUP_WAIT_MS`: on a busy device the publish
  can come too late, and they fetch for themselves. The synchronization is
  of the fetching thread's current context, so the destinations are assumed
  to be on its device, as the connectors' own fence assumes. Whether real hardware shows the reordering depends on the
  platform; the fix follows the ordering rule of the GPUDirect RDMA design
  guide and was verified against a simulated driver that holds such writes
  back. New counters `dfkv_client_gpu_dedup_load_fences_total` and
  `dfkv_client_gpu_dedup_load_fence_failures_total`. Host-memory dedup and
  processes without GPU dedup are unaffected.
- SGLang HiCache connector, device-direct (L2-bypass): a process that has
  not moved a page yet now addresses every scatter-gather group of a page.
  With `SGLANG_HICACHE_L2_BYPASS` requested, registering the device pool
  records how many groups a page is stored in (from the segments of one
  page, no I/O); before, the count was known only after the first device
  transfer, so until then existence probes, the presence answer of
  `batch_get_v1` and retention addressed group 0 only. `batch_exists` now
  probes every known group of a page, like `batch_get_v1`; on hardware whose
  scatter-gather width is below the layer count that is one key more per
  group per page and probe. If the SGLang build's device page metadata
  cannot be called at registration, a WARNING says so and the previous
  behaviour applies until the first transfer.
- SGLang HiCache connector, `batch_set_v2_device`: the side-pool write metric
  (`on_set_v2` counters) now counts every side pool that did I/O, with its
  bytes and time. Host side pools were counted without their bytes and time,
  and on a rank that does not write the replicated KV nothing was counted,
  although rank-sharded side pools are written on every rank. The number of
  puts that succeeded on the retry (`retry_ok=` in the access log) is now
  returned with each put instead of being left on the storage object, where
  a concurrent put overwrote it, and `batch_set_v2_device` reports the sum
  over all of its puts instead of the last one.
- SGLang HiCache connector: `batch_touch` with a retention TTL no longer
  repeats its whole work on every request. To protect a prefix it sends a SET,
  which the native client does not debounce (only a REFRESH with a TTL hint
  is), and before that it built the object keys of the entire prefix: for a
  long prefix on a writer rank thousands of keys, a precheck and a SET per
  request, under the GIL. A prefix this process has protected for at least the
  requested TTL within the last tenth of that TTL is now skipped before any
  key is built; the interval is the one the native client uses for refreshes.
  A touch whose SET left an object unprotected, or whose precheck was refused,
  is not remembered and is repeated. A prefix is remembered with the key form
  of its objects (bare or scatter-gather keys, and the number of groups), so
  it is touched again when that form changes. The table is bounded (4,096
  prefixes per storage backend). Touches without a TTL and `retain_precheck`
  are unchanged.
- SGLang HiCache connector: storage coordinates are no longer assumed when
  SGLang's parallel context cannot be read. The connector takes the
  context-parallel coordinates of a rank (prefill CP, decode CP) and the
  attention-TP rank used to elect the MLA writer from that context. When
  reading it failed for any reason, it silently used pcp=0/1 and dcp=0/1 and
  elected the writer by `tp_rank`: in a deployment with context parallelism
  every rank then shared one key, one rank wrote, and the others read that
  rank's shard as their own (same size, so the read check passes). Now the
  coordinates come from `HiCacheStorageConfig` where it carries both size
  and rank of an axis: PCP from `attn_cp_size`/`attn_cp_rank`, DCP from
  `dcp_size`/`dcp_rank`. Each storage backend created this way logs a
  WARNING naming the source of each coordinate, and
  `dfkv_hicache_parallel_context_fallbacks_total` counts it. Without DCP
  fields there, DCP is 0/1 if the SGLang build has no parallel-context API
  at all (such builds have no decode CP); if the API is there but fails
  (state not initialized yet, or changed), or a field of the context raises
  when read, the DCP coordinate is unknown and the backend refuses to be
  created unless `dcp_size`/`dcp_rank` are given in the extra_config. A host
  pool that declares a `dcp_size` above 1 different from the resolved one is
  refused at registration. Deployments whose coordinates were read from the
  parallel context, or given in the extra_config, are unaffected. One case
  that started before now refuses to: an SGLang build without the
  parallel-context API, an MLA model and `attn_cp_size` above 1 in
  `HiCacheStorageConfig`. The prefill CP axis is now seen there, and MLA
  with prefill CP needs an explicit `prefill_cp_storage_layout` in the
  extra_config, as it already did when the coordinates came from the context
  or the extra_config.
- SGLang HiCache connector: a v2 backup and a v2 prefetch that run at the same
  time no longer take each other's result. The pool-aware transfer
  (`batch_set_v2`, `batch_get_v2`, and the host side pools of the `_device`
  variants) handed its byte count, I/O time and the names of the pools it
  touched to its caller through attributes of the storage object. SGLang backs
  up and prefetches on different threads, so a set that had finished its
  transfer could read what a concurrent get had just left there: its metrics
  counted the get's bytes, and when the two calls carried different pools the
  set failed with a `KeyError` and the backup was lost. The transfer now
  returns these values.
- SGLang HiCache connector, device-direct (L2-bypass): the host read no
  longer reports pages as retrieved that are not in the store, and the key
  form no longer follows from a device pool merely being registered.
  `batch_get_v1` cannot load device-direct objects into a host buffer; with
  a device pool registered it returned True for every page without any I/O.
  It still loads nothing, but now answers per page whether every
  scatter-gather group of it is stored. A missing group or a failed probe is
  a miss.
  The KV itself is loaded, and every group length-checked, by the device
  entries as before. Second, existence probes, retention and that host read
  switched to the device key form as soon as a primary device pool was
  registered. SGLang can register the pool without using the bypass; the
  host path then wrote bare keys that the probes never found (no L3 hits,
  retention sent to keys that do not exist). The device form is now used
  only when `SGLANG_HICACHE_L2_BYPASS` was requested and its registration
  held, or once the process has done a device transfer of primary KV. Cost:
  one existence round trip per `batch_get_v1` call in device-direct mode,
  where there was none.
- LMCache connector (in-process `RemoteConnector`): a cancelled GET no
  longer gives its buffer back to the L1 arena while the transport is still
  writing into it. LMCache cancels a `get` that exceeds its blocking timeout
  (10 s by default); the native GET runs on a worker thread and does not
  stop. The connector released the buffer at the cancellation, so the late
  write landed in a chunk the arena had already handed to another request:
  memory corruption in the host process, whenever a GET to a slow or
  degraded node outlived the timeout. On `batched_get` and
  `batched_get_non_blocking` a cancellation skipped the release instead and
  leaked the whole batch, and a failing chunk released the buffers of its
  still-running siblings. Buffers are now released only when the native
  call can no longer touch them: when it returns, or at once if it had not
  started. Nothing waits on the event loop for that. The uncancelled paths
  are unchanged.
- LMCache connector (in-process): a stale "present" answer of the exists
  cache no longer lasts for the life of the process. The cache remembered
  every key it had stored or seen, with no expiry, and only an explicit
  remove cleared an entry. After the server had evicted a chunk, every
  request for that prefix was told it exists, failed to load it, recomputed
  it and did not store it again, because LMCache's put path skips what
  `contains` reports. Now a GET that misses (or returns an unusable length)
  forgets the key at once, on the scalar and both batched paths, and an
  entry expires 60 s after it was last confirmed by the server
  (`DFKV_CONNECTOR_EXISTS_TTL_S`, `0` = no expiry), so a key that is never
  read again is still re-checked. More `exists` calls reach the ring than
  before: at most one per key per TTL.
- LMCache MP-server L2 adapter: a load no longer reports an object loaded
  when the stored value has another length than its buffer. The adapter
  ignored the length that came back, so a value stored under the same key
  for another layout (a dtype change without a new `model_revision`) or a
  truncated one was handed to LMCache as the object. A hit must now have
  exactly the buffer's length; others count as misses
  (`load_length_mismatches`, one warning). With a `serde_config` (serialized,
  variable-length values) the check is off. The object key also did not
  include `ObjectKey.model_name`: two models served through one adapter
  shared a key wherever chunk hash, rank, group and salt agreed. The new
  `adapter_params.key_includes_model` (default `false`) adds it; the adapter
  warns once when it sees a second model name while it is off. It is off by
  default because it changes every key: an upgraded deployment keeps finding
  what it stored, and turning it on costs one cold start.
  With it on, a `cache_salt` that would push the key component past 256 bytes
  goes into the key as its SHA-256 (`salth-<hex>`); with it off a salt over
  251 bytes is still refused, as before.
- LMCache MP-server L2 adapter: works with LMCache 0.5.x, and the
  connector's LMCache dependency is bounded (`>=0.4.5,<0.5.7`; it was
  unbounded). From 0.5.0 LMCache passes a layout argument to
  `submit_lookup_and_lock_task`, which the adapter did not take: the server
  started and the first prefetch raised a `TypeError`. From 0.5.4 `Bitmap`
  lives in `lmcache.lmcache_native`, and the adapter failed to import. Both
  forms are accepted now (the layout is advisory and not used). The upper
  bound: 0.5.7 adds an abstract L2 method the adapter does not implement,
  and moves `lmcache.v1.storage_backend` to
  `lmcache.inprocess.v1.storage_backend`, so the in-process connector does
  not import there either. A fresh install therefore gets LMCache 0.5.6,
  and an environment whose vLLM requires LMCache 0.5.7 or later gets a
  resolver conflict. Unit tests check the adapter against the interface of
  0.4.7, 0.5.0, 0.5.3 and 0.5.4; end-to-end runs so far are on 0.4.7.
- RDMA client: a PUT, and a read into the caller's memory, that find a pooled
  connection the server has reclaimed or lost in a restart are made again on
  a new connection instead of failing. Reads through staging buffers and
  existence checks were already tried again on such a stale pooled
  connection, and since 2.39.0 it no longer cools the peer rail. A PUT that
  had been posted was not tried again: in general it is unknown whether the
  server stored it. Nor
  was a whole-object read into caller memory (`RangeInto`,
  `RangeIntoMulti`, the zero-copy and scatter-gather GET), which had a
  single attempt. Both failed with `kIOError` after the transport's
  retries ran out (`status=12 status_name=retry_exc_err` in the CQ failure
  line). Now, when nothing has succeeded on the connection since it was
  taken from the pool, not even the peer NIC's acknowledgement of the
  request, the request was almost certainly not delivered, and it is sent
  once more on a new connection (same rail when it is the only usable one).
  If it was delivered after all (every acknowledgement lost on the way back,
  or its completion arriving just as the window's deadline ran out), the PUT
  sent again carries the same key and bytes, and the server answers it from
  the value it already has; the first copy may still be handled after the
  second, as could any PUT the client gave up on before this change. Such a
  PUT, batch or whole-object read can take twice as long in the worst case:
  the first window has to fail, then the second runs in full. A PUT that
  failed on a connection on which something had succeeded is still not sent
  again, and a whole-object read is still not tried again after any other
  failure. The retries count in `dfkv_rdma_client_stale_pool_retries_total`.
  No setting, API or wire change; servers of any version are unaffected.
- RDMA client: a keepalive that fails no longer cools the peer rail. The
  keepalive (`DFKV_RDMA_KEEPALIVE_MS`, every 15 s by default) waits at most
  100 ms for its answer, where an operation waits `DFKV_RDMA_OP_TIMEOUT_MS`
  (5 s by default). A server that was busy for longer than that, or that had
  reclaimed the idle connection or restarted, failed the keepalive, and the
  pair (peer, rail) was cooled: with one usable rail every operation to the
  peer was refused for the cooldown although the peer was serving. The
  failed keepalive now drops that connection only, as a pooled connection
  that fails on reuse does since 2.39.0. The next operation makes a new
  connection, and if that one fails the rail is cooled as before. So a rail
  that is really down is found by an operation (which then retries on
  another rail) rather than by the keepalive in advance. The limit stays at
  100 ms: a missed keepalive now costs one reconnect, and a longer limit
  would hold a dead connection out of the pool for longer. Still counted in
  `dfkv_rdma_client_keepalive_failures_total`; no longer in
  `dfkv_rdma_client_remote_rail_failures_total`. The client CQ failure log
  line gains a last field, `peer_rail_charged=<0|1>`: whether the failure
  counts against the peer's rail. It is 0 for a failed keepalive, a stale
  pooled connection and a peer without identity, where `verdict=endpoint`
  only says that the local rail is not to blame. No setting, API or wire
  change.
- RDMA client: a keepalive round no longer empties the connection pools. It
  took every idle connection out of the pools first and then probed them one
  after another, so for the whole round, which lasts 100 ms for each
  connection that does not answer, an operation to any node found no pooled
  connection and dialed a new one; those were later retired as surplus. Now
  each connection leaves its pool only for its own probe. A connection that
  an operation is using when its turn comes is skipped in that round. No
  setting, metric, API or wire change.

## 2.39.0 — 2026-11-22

- Client: a cache node that answers slowly no longer holds up reads of the
  other nodes. Batch reads over RDMA run on a fixed set of workers shared by
  the whole process (`DFKV_RDMA_READ_WORKERS`, 7 by default), and a batch
  reaches one node through up to eight shards. A node whose operations take
  long could occupy every worker; reads of all other nodes then waited for
  one of its operations to finish. Shards of one node now run on at most
  all but one of the workers (6 by default), so one worker is always free
  for the others, also for a read that arrives while the slow node's batch
  is running. A process whose ring is a single node therefore reads from it
  with one worker fewer than before (raise `DFKV_RDMA_READ_WORKERS` if that
  matters); with `DFKV_RDMA_READ_WORKERS=1` nothing is reserved, with 2
  one node's shards run on one worker at a time. With a
  scripted transport: beside a node whose shards took 600 ms each, a read
  of another node used to wait the full 600 ms and is now served at once
  (the test requires under 200 ms). Limit: two slow nodes at the same time
  can still take every worker between them (six and one), and reads of
  other nodes then wait for one of their operations. New counter
  `dfkv_read_scheduler_node_limit_waits_total`. No setting, API or wire
  change. See `docs/DEPLOY.md` and `docs/METRICS.md`.
- Server, RAM tier: a PUT that waits for the value owning its key no longer
  blocks other keys, waits without limit, or holds up a stop. In disk-ACK mode
  (`DFKV_PUT_ACK_MODE=disk`, or a zero RAM-ACK watermark) every admitted PUT,
  first writer or duplicate, waits for the flush of that value. (In RAM-ACK
  mode a duplicate of an acknowledged value is acknowledged at once since the
  flush-failure change below; it only waits until the first value is
  resident.) A waiting PUT kept the lock that serializes PUTs of its key
  stripe (1,024 stripes) for the whole wait, so PUTs of unrelated keys on the
  same stripe queued behind it; the wait had no limit, so with a flush backlog
  it outlasted the client's operation timeout and the client cooled the node;
  and a stopping server waited for these PUTs before it started counting its
  drain limit, so a stop was not bounded by
  `DFKV_RAM_SHUTDOWN_DRAIN_TIMEOUT_S`. Now the lock is released before the
  wait. The wait is limited by the new `DFKV_RAM_PUT_WAIT_MS` (default 3000,
  below the clients' default 5 s operation timeout; `0` = no limit, the old
  behaviour): after it the PUT is answered `kCacheFull`. The limit counts
  from the arrival of the request batch the PUT came in, because an RDMA
  connection serves a client window's requests one after another under the
  window's one deadline: the parked PUTs of one window share the limit. A stop answers all
  such PUTs `kCacheFull` first, so stopping the request paths does not wait
  for a flush. `kCacheFull` here means that this request did not see the value
  reach disk: the value that owns the key stays in RAM and keeps being
  flushed, nothing is written twice, and clients, 2.35.x included, treat the
  answer as "not stored" without marking the node failed. What a successful
  answer means is unchanged, a duplicate with different bytes still gets the
  first value's result, and a flush that fails still answers `kIOError`. In
  disk-ACK mode a PUT whose flush takes longer than the limit is now answered
  `kCacheFull` although its value reaches disk later; raise the limit together
  with the client timeout if such PUTs must wait. New metrics:
  `dfkv_ram_parked_puts`, `dfkv_ram_put_wait_timeouts_total`,
  `dfkv_ram_put_waits_canceled_total`.
- Client: a cache node that fails is avoided for as long as the failure
  warrants, and finding out whether it is back no longer stalls reads of
  other nodes. Three things in the per-node health record (`PeerHealth`)
  made one incident worse than it was.
  - Every failed shard of a batch counted as one more consecutive failure.
    A batch reaches a node through up to eight shards, so the first
    failure of a node doubled its cooldown once per shard: a node that was
    restarted and back within seconds was avoided for the 30 s maximum
    instead of 2 s. Failures that arrive while the node is already being
    avoided now belong to the incident that started the cooldown. The
    cooldown doubles only when the node is tried again and fails again
    (2 s, 4 s, 8 s, up to `DFKV_PEER_COOLDOWN_MAX_MS`).
  - When a cooldown ended, every request was sent to the node again. With
    a node that drops connections silently each of them waited for a
    connect or operation timeout, and a batch for that node could occupy
    all read workers of the process, so reads of healthy nodes waited
    behind it for the length of a timeout, once per cooldown. Now one
    request tries the node and the others keep skipping it (a miss, as
    during the cooldown) until that request has succeeded or failed. This
    is one request per cooldown window, not one in total: a trial that is
    stuck in a timeout for longer than `DFKV_PEER_COOLDOWN_MS` is joined by
    the next, and if it finally fails after the following cooldown has run
    out, that counts as another failure. A trial that says nothing about
    the node hands the trial to the next request at once: the request was
    refused inside this process (transport budget, no usable rail), or the
    RDMA transport refused it without dialing because a connect to that
    address had just failed. Such a refusal used to count as a failure of
    the node; it no longer does, in a trial or otherwise.
  - The optional probe (`DFKV_PROBE_INTERVAL_MS`, off by default; a
    connector may switch it on for per-node latency metrics) cleared a
    node's cooldown and its back-off whenever the node answered. The probe
    is an existence query, which a node whose store has failed still
    answers, so requests were sent into the same failure once per probe
    period. A probe answer now ends the running cooldown early once per
    incident and makes the next request the trial; it clears nothing by
    itself. If that trial fails, the probe cannot shorten that incident
    again until a request has succeeded, and the back-off runs its course.
  Any status a node itself sends now shows it alive: besides a hit and a
  miss also "cache full" and "quota exceeded", which only a node that is up
  and looked at the request can answer. A full node that refused every
  write used to look silent, so one transport failure made the client
  avoid it although it had answered milliseconds before. An I/O error is
  not counted (the client cannot tell a server-side one from a transport
  failure).
  The tests use a scripted transport and scaled timings (500 ms for 2 s,
  5 s for 30 s) and assert: after seven shards failed together the node is
  tried again in less than four base periods, not at the cap; while a batch
  re-tries a dead node (600 ms timeout), one request is inside the dead
  node and a read of another node takes less than 200 ms; a node with a
  failed store and a 20 ms probe receives at most 4 reads in a second.
  Behaviour changes to be aware of: after a cooldown, requests to the node
  other than the trial are skipped until the trial reports;
  `dfkv_client_unhealthy_skips_total` counts those too.
  `dfkv_client_peer_recovered_total` no longer counts a probe answer, only
  a request that was answered. `dfkv_client_ops_served_total` includes
  "cache full" and "quota exceeded" answers. A node that answers probes
  while its requests fail is no longer retried at the probe rate. No
  setting, API or wire change; servers of any version are unaffected. Not
  changed here: a node that is slow but answers can still occupy all read
  workers. See `docs/DEPLOY.md` and `docs/METRICS.md`.
- Client library and transports: a mistyped `DFKV_*` setting is now reported.
  This is the client-side half of the configuration change for `dfkv_server`
  and `dfkv_mds` (see that entry below), with the same rule: nothing changes
  its effective value. Every reader keeps the parse it had; what is new is a
  check next to it that logs when the text is not a plain decimal number in
  the range the reader accepts, or not `1`/`0` (or a word that says the same)
  for a switch.
  For users of the client library (`libdfkv` inside an inference engine):
  the HOST PROCESS MAY NOW SEE WARN LINES it did not see before, at most
  one per variable and process, through the library's logger on stderr
  (never stdout; `DFKV_LOG=error` hides them): `[WARN] dfkv: invalid
  DFKV_X='v': not a plain decimal number from A to B; using N, as before. A
  later release will not accept this value.` A bad value still never makes
  the library throw, abort or exit, and nothing is logged on a
  per-operation path: the settings that are read per call or per connection
  (`DFKV_RETAIN_PRECHECK_MIN_HEADROOM_PCT`, `DFKV_RDMA_BUSY_POLL`) are
  still read there, but reported only the first time. No value is cached
  that was not cached before.
  The inputs most worth looking for, all read as they always were:
  `DFKV_RDMA=off` (or any word but lower-case `0`, `false`, `no`) asks for
  RDMA; `DFKV_CLIENT_NODE_DEDUP=true` leaves dedup off, only `1` is on;
  `DFKV_CLIENT_PERNODE_STATS=off` and `DFKV_RDMA_BUSY_POLL=off` switch
  those ON; `DFKV_PEER_COOLDOWN_MS=2s` is 2 ms; a non-numeric
  `DFKV_RDMA_OP_TIMEOUT_MS` means no per-operation deadline and a
  non-numeric or empty `DFKV_MDS_SHRINK_GUARD_PCT` switches the member
  shrink guard off; an empty `DFKV_RETAIN_PRECHECK_MIN_HEADROOM_PCT` is 0,
  not the default 5; an empty `DFKV_RDMA_INLINE_PUT_MAX_BYTES` disables inline PUT; an
  RDMA budget variable that is set but empty switches the automatic budget
  scaling off. A 0 that is documented to mean something (auto, off, no
  deadline, follow another setting) is a clean value and is not reported.
  `docs/CONNECTORS.md` §1.5 has the table.
  Covered: the client (`src/client`), the TCP and RDMA client transports,
  the transport choice, and `DFKV_MDS_SHRINK_GUARD_PCT`. Not covered: the
  `DFKV_RDMA_TEST_*` hooks, string-valued settings (device lists, rail
  tiers, paths), the CUDA bounce-pool sizes (they already warn, on every
  pool construction), and the bench and smoke tools.
- Server, retention (`DFKV_RETAIN_*`): the expiry wheel no longer grows with
  every renewal, and expiring a full bucket no longer holds up the disk.
  - Every renewal that moved a deadline (one per second and object at most)
    left an entry on the wheel until its bucket expired; an object renewed
    once a second for an hour had thousands, each 32 bytes and each looked
    at again at expiry. In a test, 500 objects renewed once a second for ten
    minutes had 300,500 entries. A protected object now has exactly one
    entry; a renewal only moves its deadline, and the expiry moves the entry
    on when it finds the deadline later than the bucket. The same test now
    has 500. `dfkv_retain_wheel_entries` is the number of protected objects
    plus the entries of protected objects that were removed, which stay
    until their bucket expires (at most the maximum TTL).
  - Objects protected with one deadline expire in one bucket, and the whole
    bucket was processed under the allocator's lock, which every read and
    write of that disk waits for: 45-72 ms for a million objects in the
    test. The expiry now looks at 1,024 entries per hold of the lock and
    pauses 50 microseconds in between. In the same test the longest hold is
    0.24 ms and a concurrent reader waited at most 1.7 ms; the expiry as a
    whole takes longer (180 ms), in the background. Without the pause the
    batches did not help: the expiring thread took the lock back before a
    waiter woke up.
  - Retention itself is unchanged: an object is protected until its latest
    deadline and at most 10 s beyond it, and is released then.
- Server, slab disks: a PUT that finds no slot is answered `kCacheFull`, not
  `kIOError`. When every slot of a size class is held by a read or write in
  flight or is protected by retention (`DFKV_RETAIN_*`), and no extent can
  be taken from another class, the disk cannot take another value. It
  answered that with the status of a failed disk. Clients (every release so
  far) mark a node bad on `kIOError` and stay away from it for 2 s and up,
  reads included, although the node was healthy; `kCacheFull` they count as
  "not stored" and go on using the node.
  - The status is now `kCacheFull` on every path that reaches the disk: TCP
    and RDMA, single and batched writes, and the RAM tier's bypass to disk.
    New counter `dfkv_slab_put_no_slot_total`; these PUTs no longer count in
    `dfkv_errors_total{op="put",status="io"}`.
  - A failure while room is being made (the table record of an eviction, an
    extent rebind) is still `kIOError`, and the store is unhealthy
    afterwards, as before.
  - A value larger than one extent is answered `kInvalid`; it was `kIOError`
    too.
  - Not changed: the RAM tier's flush to disk treats this refusal like any
    failed flush (it retries with pauses, then drops the value), whatever
    the status.
  No client change is needed; `kCacheFull` has always been part of the
  protocol.
- Server, RAM tier and slab disks: whole extents now move between size
  classes toward the data that is used, not away from the class that is not
  written. Two paths move an extent from one size class to another and
  evict everything on it: the background growth of a class that is short of
  free slots, and the cold steal an admission tries before it evicts from
  its own class. Both judged "cold" by writes alone: the growth picked a
  donor class by the bytes written to it in the last tick, the cold steal
  by how long ago an extent was last written. Reads only ordered the
  candidates, and in the RAM tier not even that, because its reads recorded
  nothing but the per-slot reference bit. With one model's pages written
  and then only read while another page size was being written, the first
  lost extents until it was at its floor of 8 (growth) or had none left
  (cold steal); and two classes written in turns handed the same extents
  back and forth.
  - A donor's extent is now taken only if its residents are much colder
    than what the growing class would otherwise evict of its own: the mean
    time since they were last read or written must be more than twice that
    of the growing class's own least recently used extent (among those it
    could lose: not one with a pinned slot or a retained value). Otherwise
    the growing class evicts from its own slots. Two steals inside an
    admission make no such comparison, as before: the one that brings a
    class with fewer than 8 extents up to 8, and the last resort when the
    class has neither a free slot nor anything of its own to evict.
  - After a restart, restored values carry no access time, so a growing
    class takes no extent from another until its own least recently used
    extent holds values written since the restart, about one turnover of
    that class. On a disk behind a RAM tier, recency sees only the reads
    the RAM tier misses.
  - So a class that is read and not written keeps its extents against a
    class that is only written; a class that is read less than a newer one
    is used still gives way to it, down to the same floors as before; and
    classes of about the same age no longer trade extents.
  - RAM-tier reads now record when a slot was last read. This costs no
    additional lock and touches no memory a read did not already write: the
    time goes into the slot and extent records that the read's pin updates
    anyway, and it is stored again only after an admission has moved the
    allocator's clock.
  - Unchanged: the order of eviction inside a class, the 8-extent floor of
    a donor in the background growth, the write-age gate of the cold steal
    (`DFKV_SLAB_COLD_STEAL_WINDOW`), and a class that is neither read nor
    written, which gives its extents as before.
  See `docs/ARCHITECTURE.md`.
- Server, RAM tier: a failed flush to disk is handled less rigidly, and a
  node no longer acknowledges PUTs it can never store. Until now a flush
  that failed was tried three times within microseconds, then the object
  was dropped and the tier reported itself unhealthy (`/healthz`, `/readyz`
  503) until the process was restarted; and after a disk store had failed
  for good, every further PUT for that disk was still acknowledged from RAM
  and its value dropped right after (measured: 20 of 20 acknowledged, none
  readable), until clients had dropped the node from their ring.
  - A PUT whose disk store has failed for good is no longer taken into RAM:
    it gets the store's I/O error, as on a node without a RAM tier. Clients
    of every release treat that status as a failure of the node and stop
    sending to it for their peer cooldown (2 s by default), which is what
    is wanted until the node has left their ring. Keys of the node's other
    disks are unaffected. The check is one atomic read per disk on a PUT. A
    flush to a failed store is not tried again. Values acknowledged before
    the store failed and not yet written cannot be saved; they are dropped
    at once and counted.
  - A value that was acknowledged from RAM and whose flush fails is tried
    again after 50 ms, 200 ms, 800 ms and 3.2 s (five attempts, about 4 s)
    before it is dropped. It stays readable and counts as dirty meanwhile,
    so a disk that keeps refusing turns into `kCacheFull` for new keys at
    the ACK watermark. A failure that passes within that time (an I/O error,
    a store that could not place the value under load) loses nothing. One
    that lasts minutes, like a full filesystem, still drops the values
    written into it; that is now visible, see below. A PUT that waits for
    its flush (disk-ACK) is tried again without pauses, as before, and gets
    its error at once. A stop does not wait out the pauses.
  - Behaviour change: the RAM tier's health can recover. After a drop the
    tier is unhealthy until 64 flushes in a row have succeeded (a failed
    attempt starts the count again) and at least 30 s have passed since the
    last drop, instead of until restart. So
    `dfkv_ram_healthy`, and through it `/healthz` and `/readyz`, are no
    longer latched: a node whose filesystem was full and has room again
    recovers by itself, and a probe that relied on the 503 staying must use
    the counters instead. `dfkv_ram_flush_dropped_total` and
    `dfkv_ram_post_ack_flush_failures_total` stay monotonic.
  - What a disk that takes no writes holds in RAM: the values waiting for
    another attempt stay pinned, and they are dirty, so at most the ACK
    watermark's share of the RAM tier (`DFKV_RAM_ACK_HIGH_WATERMARK_PCT`,
    default 80 %) is held this way, however long the disk refuses. The
    watermark is one number for the whole tier, not one per disk: a single
    refusing disk among several can fill it, and then new keys of EVERY
    disk of that node get `kCacheFull` (no peer cooldown) until its values
    have been dropped or written. Reads and keys already resident are
    served as before.
  - A second PUT of a key whose value was acknowledged from RAM and is
    still waiting for its flush is acknowledged at once, like the first: the
    value is resident and its flush pending, which is all the first writer
    was told. It used to wait for the flush; with attempts spaced out that
    would have parked a handler thread and one of the 1,024 put-key stripes
    for seconds. If the value is later dropped it is counted once, whoever
    wrote it. A second writer in disk-ACK mode still waits for the first
    one's flush and gets its outcome.
  - A PUT that waits for a flush (a disk-ACK writer, or a second disk-ACK
    writer of a key whose first value is still being flushed) is told how
    the flush
    ended, `RamTier::FlushOutcome`: flushed, dropped after retries, dropped
    because the store failed, or dropped at shutdown. Only "flushed" is a
    success. No lock is held across a pause between attempts, and no PUT
    waits through one.
  - Every acknowledged value that is dropped is visible: the existing
    counter `dfkv_ram_post_ack_flush_failures_total`, the new alert
    `DfkvRamAcknowledgedDataLost` on it, and a WARN line `ram tier: dropped
    N object(s) that could not be written to disk, K of them already
    acknowledged to the client ...` (at most one per 10 s). New counter
    `dfkv_ram_flush_retries_total`: failed attempts that were followed by
    another one.
  The healthy path, the wire and the stop sequence are unchanged. Ring
  eligibility is unchanged too: an unhealthy RAM tier alone does not take a
  node out of placement, a failed disk store does. See `docs/DEPLOY.md` and
  `docs/METRICS.md`.
- Server: a slab disk is marked in use only when the server is about to
  serve, not when the disk has been opened. Until now each disk recorded
  the dirty state at the end of its own open. From then until the rest of
  the start was done (the other disks, the RAM tier arena, on small pages
  tens of seconds) a process that was killed, by SIGKILL or for lack of
  memory, left that disk dirty although it had not written a byte, and with
  `DFKV_SLAB_WARM_RESTART=off`, the default, the next start emptied it.
  - Opening a disk now changes nothing a cleanly stopped run left behind:
    `slab_state` stays as found, no trust token is written, no background
    thread runs, and the store refuses every operation. What an open may
    still do is what every open could always do again: allocate missing
    extent blocks, reserve the table's space, and clear table records that
    fail their checks. All disks are activated together once every disk
    is open and the RAM tier is ready, right before the server starts
    listening: the dirty state and this run's identity are written and
    synced, the trust token is granted, and only then do the sync and
    reclaim threads start and operations pass.
    From that point an exit without a clean stop is an unclean exit, as
    before.
  - A start that is killed before that point, is cancelled by a stop signal,
    or fails (a disk that does not open, a RAM tier that cannot be set up)
    leaves every disk exactly as its previous run closed it: the next start
    is a clean start with the cache in place, whatever
    `DFKV_SLAB_WARM_RESTART` says. Before, a failed or cancelled start
    relied on an orderly close of the disks it had already marked.
  - A disk found dirty (the previous run did not stop cleanly) is treated as
    before: its open resets it or keeps its table, and marks it in use at
    once.
  - New INFO line `disk stores activated: N in X ms; an exit without a clean
    stop is an unclean exit from here on`.
  No change to the files on disk; older releases open a directory written by
  this one and the other way round.
- Configuration: a mistyped setting of `dfkv_server` or `dfkv_mds` is no
  longer read as something else without a word. Numbers were parsed in
  several ways, most of them with `strtol`/`atoi` and little checking: a
  value that is not a number read as 0, a negative one wrapped, text after
  the number was dropped, an out-of-range port was truncated. The rule of
  this change:
  - ENVIRONMENT values (and the flags that are passed on as their variable):
    nothing changes its effective value. Every input that starts a server
    today resolves to exactly what it resolved to before. What is new is
    one WARN per variable and process for every value that is not a plain
    decimal number in range, or not `1`/`0` for a switch, naming the
    variable, the value and what is used: `invalid NAME='v': ...; using N as
    before`, `...; using the default D`, or `NAME='v' clamped to the maximum
    M`. After an upgrade, `grep -E "invalid DFKV_|clamped to the maximum"`
    in the start-up log finds what to clean up. A later release will read
    these by their meaning or reject them; `docs/DEPLOY.md` §3b has the
    table of inputs that are accepted with a WARN until then. The ones most
    worth looking for: an empty or non-numeric
    `DFKV_RAM_ACK_HIGH_WATERMARK_PCT` or `DFKV_RAM_RECLAIM_MS` reads as 0
    (acknowledge from disk; no background reclaim); a non-numeric or
    negative `DFKV_RDMA_IDLE_MS` reads as 0 (no idle reaper); `off`, `false`
    and `no` switch `DFKV_READ_COALESCE`, `DFKV_RDMA_NUMA`,
    `DFKV_SERVER_URING` and `DFKV_SLAB_URING_WRITE` ON, because only `0` is
    off for them; `DFKV_RAM_TIER=ON` leaves the RAM tier off, because its
    words are lower case; `64G` is 64.
  - `--log` / `DFKV_LOG` take the level in any case. `--log WARN`, the
    spelling `--help` gives, was read as INFO. This is the one environment
    value whose effect changes: a level that was asked for now applies.
    This reaches users of the client library too: a host process that has
    had `DFKV_LOG=DEBUG` (or `Warn`, `ERROR`) set without effect now gets
    debug lines on stderr (or only warnings, or only errors).
  - COMMAND-LINE values that are parsed directly and were silently
    truncated or wrapped now stop the start with exit 2: `--port`,
    `--rdma-port`, `--metrics-port` and `dfkv_mds --listen` outside 0..65535
    (`--port 70000` listened on 4464; an explicit `-1` for the optional
    listeners still means off), a negative `--cap` or `--max-msg` (`--cap
    -1` was 2^64-1), an integer that does not fit an int (4294967297 was 1),
    and a `dfkvctl --stat-port` that is not a port in 1..65535 (it was read
    as "use the member's port"). `--mds` without `--id`/`--advertise` is
    refused (exit 1, as before) before any disk is opened instead of after.
    A start without `--dir` or `--cap` still uses the defaults but logs a
    WARN. The values in the shipped Dockerfile and `deploy/` scripts are
    unaffected.
  Settings that already refused to start on a bad value (storage engine and
  slab geometry, tenant quotas, the first-registration timeout) still do. No
  setting is renamed and no default changes. All of this goes through one
  parser, `src/utils/env_parse.h`, which the existing strict parsers now use
  as well. The variables the client library shares with the server
  (`DFKV_LOG`, `DFKV_RDMA_NUMA`, `DFKV_FANOUT_THREADS`, the payload limits)
  are reported the same way there and never abort the host process; the
  client-only variables are not part of this change.
- Client RDMA: three kinds of failure that have nothing to do with a rail are
  no longer charged to one.
  - The TCP connect to a peer's bootstrap address fails (node down, listener
    gone). Every rail dials the same address, so no (peer, rail) pair is
    cooled and the operation no longer makes a second connect over another
    rail; with a dropped SYN that was two connect timeouts per operation, and
    the rails were cooled two at a time. The operation fails with `kIOError`
    as before, which is what makes `KVClient` cool the node. The address is
    then not dialed again for `DFKV_RDMA_REMOTE_RAIL_COOLDOWN_MS` (2 s by
    default, the time its rails used to be cooled for): operations that need
    a new connection to it, and the client's background probe of the node,
    fail at once with `kIOError` and do not each wait for a connect timeout.
    Connections that are already established are not affected. Counters
    `dfkv_rdma_client_peer_unreachable_total` and
    `dfkv_rdma_client_peer_unreachable_skips_total`.
  - The peer accepts the bootstrap connection and closes it before sending
    anything, which is what a server at `DFKV_RDMA_MAX_PENDING_HANDSHAKES`
    does, and also what a server does that does not serve the requested
    device. On the first attempt of an operation nothing is charged: the
    client waits 20 to 40 ms (random per caller) and tries once more, on
    another compatible rail when it has one, as it did before, and on the
    same rail otherwise, where it used to give up. Only a close on that
    second attempt is charged to the (peer, rail) pair it happened on. An
    operation still makes at most two connection attempts. A read that times
    out, or a reply that is invalid, is charged at once as before. Counter
    `dfkv_rdma_client_peer_busy_total`.
  - This process cannot register a caller's buffer because of the
    locked-memory limit or an address that cannot be pinned (`ENOMEM`,
    `EAGAIN`, `EFAULT`, `EINVAL`), or cannot create an endpoint because it is
    out of file descriptors or memory (`EMFILE`, `ENFILE`, `ENOMEM`). These
    were recorded as local rail faults: after three of them every rail was
    quarantined and the metrics looked like a hardware problem. Now no rail
    is charged, no other rail is tried, the connection is closed without
    blaming the peer, and the operation returns `kResourceExhausted`, which
    does not cool the peer. A WARN line names what failed and the errno.
    Registration and endpoint failures with any other errno are rail faults
    as before. Counter `dfkv_rdma_client_local_resource_failures_total`.
  No wire change.
- Client RDMA: a failure is charged to the peer's rail only when it says
  something about that rail, and one event counts once.
  - A pooled connection that fails when it is reused, before anything has
    succeeded on it again (the server reclaimed it while idle, or restarted),
    no longer cools the (peer, rail) pair. Cooling
    it also refused the retry on a fresh connection wherever that rail was the
    only one usable for the peer (one rail configured, or one rail in common),
    so for a peer known by identity through the MDS a single stale connection
    failed the operation with "no compatible rail" and kept the peer unusable
    for the cooldown; with several rails it pushed traffic off the preferred
    rail for the cooldown. Now the dead connection is discarded and the
    retry runs; the pair is cooled only if a connection made now fails as
    well. For a peer known by identity the other idle connections to it on
    that rail are discarded with it. New counter
    `dfkv_rdma_client_stale_pool_connections_total`. Peers given as a static
    address list were never cooled and still are not; their other idle
    connections are left in the pool as before, and each is counted when it
    is found dead. A failed keepalive on an idle connection is treated as
    before.
    Behaviour change to be aware of: a rail on which pooled connections keep
    dying while new connections work is no longer cooled at all. The counter
    above is the signal for that, not
    `dfkv_rdma_client_remote_rail_cooldowns_total`.
  - Operations that were in flight when a peer rail failed and then fail too
    are observations of the same failure. They are still counted in
    `dfkv_rdma_client_remote_rail_failures_total` but no longer double the
    cooldown each: five operations failing together used to give a 30 s
    cooldown where one gave the base 2 s. The cooldown now grows only when the
    single recovery probe made after it fails.
    Behaviour change to be aware of: the cooldown of a failing rail is the
    base one however many operations fail with it, also when the rail has
    just recovered through a successful probe and then fails again; it used
    to be longer the more operations were in flight.
  No wire change; retry counts, rail choice on retry and local rail
  quarantine are unchanged.
- Server: a stop signal (SIGTERM or SIGINT) that arrives while the server
  is still starting now cancels the start. Until now the signal was only
  looked at once the server was fully up: the start ran to its end (disks
  opened, RAM tier arena faulted in, RDMA listener started, registration
  with the MDS begun) and only then was undone. With a large arena on 4 KiB
  pages that is tens of seconds, and a service manager whose stop timeout
  is shorter killed the process with its disks marked dirty, which costs
  the cache unless `DFKV_SLAB_WARM_RESTART` is `auto` or `boot`.
  - The arena pre-fault is given up within one 2 MiB step of the signal.
    The disk opens are not interrupted: they take seconds, and a disk that
    has finished opening is closed clean again.
  - The signal is acted on after the disks are open, after the RDMA listener
    has started, and right before the health endpoints come up and the
    registration with the MDS begins. A cancelled start never listens for
    TCP requests in the first case and never registers in any of them. A
    signal that arrives after the registration loop has started is an
    ordinary stop, as before.
  - Every disk that was opened is synced and left as it was found (it has
    not been marked in use yet), so the next start is a clean start. The
    last log line is `dfkv_server start cancelled: not registered with the
    MDS, every opened disk closed clean,
    N s after the process started; exiting`, and the exit status is 0, as
    for any stop that was asked for. If a disk failed to open or could not
    be closed clean, the line is a WARN (`... but a disk failed to open or
    is not clean ...`) and the status is 1.
  - New INFO line `ram tier arena: faulting in N MiB (threads=T)` when the
    pre-fault begins; `ram tier arena pre-fault given up: the start is not
    going ahead` when it is abandoned, which also happens when a disk fails
    to open: a failed start no longer waits for the arena.
  See `docs/DEPLOY.md`.
- Client, RDMA: a topology change of one cache node no longer makes a client
  drop its pooled connections to every node. The client published each member
  to the transport under the epoch of the whole member list, and that epoch
  changes when anything about any member does (a port state, an address, a
  node joining or leaving). The transport therefore saw a new generation for
  every peer: it tore down all idle connections, retired the active ones as
  they were released, and forgot what it had learned about each server's
  capabilities, so one port flap on one node made every client reconnect to
  all nodes at once. Each member is now published under an epoch of its own
  entry (`MemberTopologyEpoch`: id, address, weight, eligibility and rails
  with their states, the same fields as before, of that member only). A member
  whose entry is unchanged keeps its connections; the member that changed, was
  removed, or was removed and added again loses its own, as before. The order
  in which the MDS lists members does not matter, and a list the MDS withholds
  changes nothing. No wire or MDS change, and nothing to configure. A server
  that restarts without any change to its entry was not announced by the MDS
  before either, unless some other member happened to change at the same time:
  the client notices through the failure of the first request or keepalive on
  a dead connection, which drops that node's idle connections on that rail and
  its cached capabilities. Idle connections on the node's other rails are not
  dropped then; each fails on its own first use.
  `dfkv_rdma_client_stale_generation_reaps_total` and
  `dfkv_rdma_client_peer_topology_updates_total` now grow by the affected
  member only.

## 2.38.0 — 2026-11-22

- Client, host and GPU dedup (`DFKV_CLIENT_NODE_DEDUP=1`), the publishing
  side: three ways a process that fetched a key for the others could do
  damage after the key had been written or removed in the meantime.
  - A publisher compared its claim with the shared slot and then took the
    slot in a second step. If the key was invalidated and the slot claimed
    again in between, for a new fetch of the key or for another key, the
    old bytes were published into that claim. A late abort could drop it
    the same way. Both now compare again after taking the slot and put it
    back untouched when it is no longer theirs. The claim's real owner,
    if it publishes at that very moment, waits for the slot to come back
    (at most 50 ms) instead of giving up.
  - A claim was identified by a small per-slot counter only, and a late
    publish went to whichever slot the key had by then. If the key had
    moved to a slot whose counter happened to match, bytes from before the
    write were published as current. A claim now names its slot, and
    publish and abort act on that slot only.
  - Host path only: a publisher killed while copying left its slot locked
    for good, so every later write or remove of that key spun forever in
    the client and every lookup of it waited out
    `DFKV_NODE_DEDUP_WAIT_MS`. An invalidation now takes the slot away from
    a publisher that is copying, whether it is alive or not (what it was
    about to publish is from before the write anyway), and a lookup does
    the same once the publisher has been copying for longer than
    `DFKV_NODE_DEDUP_TAKEOVER_MS`. Two windows of a few instructions each
    remain in which a kill still blocks the key as before: between taking
    the slot and marking it, and during the last stores of a commit or of
    taking a slot away. On the GPU path a publisher killed anywhere while
    publishing still blocks its key as before.
  The shared-memory layout, the slot states and the segment names are
  unchanged. Taking a slot away relies on the publisher committing in a way
  that can be refused, which only this release and later do; they say so
  in a slot field that older releases neither read nor write. A publisher
  of an older release on the same host is therefore still waited for, and
  if it is killed while copying, its key stays blocked until the segment
  is removed.
- Server: a start with a RAM tier and populated slab disks takes less time.
  Nothing changes on disk or on the wire, and a restarted node holds exactly
  the state it held before: same eviction order, same order in which free
  slots are handed out.
  - The RAM tier arena is faulted in while the disks are opened, instead of
    after them. The two were independent and each took seconds (192 GiB
    arena: about 3 s; a disk with 2.3 million objects: 2-3 s). A start now
    takes about as long as the slower of the two. A disk that fails to open,
    or an arena that cannot be set up, still ends the start as before; in
    the first case the process now also waits for the arena to finish
    faulting in before it exits. The memory a disk needs only while its
    table is being restored (up to about 250 MiB per disk at that size) is
    now in use while the arena is faulted in, not released before.
    `dfkv_ram_arena_hugepage_bytes` and the `huge pages=N MiB` figure of the
    arena log line are now read from the arena's own mappings; as a change
    of the whole process they would have included what the disk opens
    allocate meanwhile.
  - Restoring a disk's table into the allocator is faster: the per-object
    bookkeeping of the restore no longer allocates a heap node per object,
    taking a slot out of an extent's free list no longer searches the list,
    duplicate keys are found by sorting 8-byte hashes instead of 24-byte
    keys, and index buckets are prefetched. Measured with the test
    `SlabAllocator.RestoreBulkBench` (2.3 million objects in 2,808 extents of
    1 MiB slots, no disk involved) on a machine that was not idle: 1.47-2.13 s
    before, 0.64-1.12 s after; runs taken back to back were about 1.6 s and
    0.95 s.
- Server: the slab slot table (`slots.tbl`) has all of its space reserved
  when a disk is opened. It was sized with `ftruncate` and written through
  the page cache, so its blocks were allocated when an extent was first
  used, while in direct mode the extents are allocated at startup. On a
  cache filesystem that filled up (capacity set close to the filesystem
  size, or something else using the filesystem) a record write then failed
  with ENOSPC, which fails the store for good and takes the node out of
  placement; nodes sharing one capacity setting could go one after another
  as they filled. Now the table is allocated in full with `fallocate` when
  it is created, and a table that has its blocks is left alone.
  A new cache directory on a filesystem with no room for the table does not
  start: that disk fails with `cannot reserve N bytes for slots.tbl ...: No
  space left on device` instead of starting a store that fails later.
  An existing directory always starts as before. A table written by an
  earlier release is completed at the first start after the upgrade (INFO
  line `reserved slots.tbl, N bytes (M were allocated)`). If the filesystem
  has no room for that, the store starts with the lazy allocation it had and
  logs, at every start until there is room, a WARN line `no room to reserve
  slots.tbl, X of N bytes are not allocated`: free X bytes on the cache
  filesystem and restart to close the exposure.
  The table takes `capacity / granularity x 64` bytes per disk, in addition
  to `--cap`. File size and record format are unchanged: older releases open
  a directory written by this one and the reverse. What a restart restores
  is unchanged; preallocated ranges read as zeros and the restore scan still
  skips them where the filesystem reports them as holes (measured on xfs and
  tmpfs: a 1 GiB table holding ten objects opens in the same time as
  before). Buffered mode (`--slab-write buffered`, or a filesystem without
  O_DIRECT) still allocates extent blocks as payloads are written: a full
  filesystem there fails the PUT, not the store. A filesystem without
  `fallocate` logs one WARN line and keeps the lazy allocation. See
  `docs/DEPLOY.md`.
- Server and MDS: the two remaining accept loops, the cache TCP listener of
  `dfkv_server` and the metrics HTTP listener of `dfkv_server` and `dfkv_mds`,
  get the same protection as the RDMA bootstrap and MDS listeners. A
  connection whose handler thread cannot be created is closed and counted
  instead of the exception ending the process; `accept()` failing for lack of
  descriptors or memory is counted, logged at a bounded rate and retried after
  100 ms instead of at once, which used to pin a core for as long as the
  shortage lasted. Stopping is not delayed by the wait. New metrics:
  `dfkv_tcp_accept_failures_total`, `dfkv_tcp_thread_start_failures_total`.
  The metrics listener reports the same two events in the log only (WARN
  lines starting `metrics listener:`). Connection limits, timeouts, the wire
  protocol and existing metrics are unchanged.
- Client, host and GPU dedup (`DFKV_CLIENT_NODE_DEDUP=1`): two ways a lookup
  could be answered from shared memory with bytes it should not have got.
  Both need an unusual interleaving.
  - A write or a remove of a key cleared only the first shared entry of that
    key. A key can have a second, older entry further along its probe
    window, left behind when a process that could not use it (expired, or
    another length) fetched the key again into an earlier free slot. That
    entry survived the invalidation and was served, with the bytes from
    before the write, once the entry in front of it was gone. An
    invalidation now removes every entry of the key.
  - A reader looks a slot up by key and then copies out of it. If the slot
    was recycled and republished for another key of the same length in
    between, the reader got that key's bytes. The copy now re-checks the
    slot's key under the same generation check that already guards the
    bytes.
  A process waiting for another's fetch also no longer accepts an entry
  older than `DFKV_NODE_DEDUP_TTL_MS`, on the host and the GPU path; a claim
  never did.
  The shared-memory layout, the slot states and the segment names are
  unchanged, and processes of older versions can share a segment with this
  one. Each fix protects the process that runs it: an older process still
  clears only the first entry when it invalidates, and still copies without
  the key check, until it is upgraded.
- RAM tier: eviction now honours reads. A GET served from the RAM tier pinned
  its slot but never set the slot's CLOCK reference bit, so the class hand
  never met a referenced slot and the tier evicted in strict admission order:
  a value that every request reads was pushed out after one turnover of its
  size class, read from disk and promoted again, repeatedly. A RAM hit now
  sets the reference bit (`SlabAllocator::PinForRead`), under the allocator
  lock the pin already takes and only when the bit is clear; the hand clears
  it and passes the slot over once. A value read at least once per lap of the
  hand over its size class stays resident; one that stops being read is
  evicted within one further lap. Pins taken for writes (copy window, flush)
  give no such credit. Reads still do not reorder the ring or feed the
  class/extent read heat, so whole-extent rebalancing between size classes is
  unchanged and can still take an extent whose values are read but no longer
  written. The disk tier already recorded reads and is unchanged. Expect
  `dfkv_ram_promoted_total` and disk reads to drop for workloads that reread a
  hot subset while writing. The sweep that clears reference bits runs under
  the shard lock that reads take, so in the RAM tier one eviction clears at
  most 512 bits (new `SlabAllocator::Options::evict_clear_limit`, unbounded by
  default and in the disk tier) and then evicts the first evictable slot it
  met. A background reclaim call ends after 512 clears and releases the lock;
  the reclaimer repeats such calls, up to 64 per class and tick. While 512
  referenced slots stand at the hand each call frees one slot, so a tick frees
  at most 64 slots per class and shard; admissions beyond that evict inline
  under the same bound, from their own class, without first taking a whole
  cold extent from another size class as an admission without a free slot
  otherwise does. In a class where nearly every slot has been read, a
  referenced slot can therefore be evicted, and the hand laps faster than
  slots are evicted: there reads give no protection and values leave in about
  the order they were admitted, as before this change. A value that now stays
  in RAM is no longer re-read from disk once per turnover, so those re-reads
  no longer refresh the recency of its disk copy: the disk tier may evict it,
  and it is then absent after a restart until it is written again.
- Server: opening a slab disk at startup costs less. Nothing changes on disk
  or on the wire: older releases open a directory written by this one and
  the other way round.
  - Checksum: the CRC-32 of each 64-byte table record is computed eight bytes
    per step from tables instead of one bit at a time. Same polynomial, same
    values.
  - Index: the per-object length index is keyed by the 24-byte key instead of
    its 48-character name, so restoring an object no longer builds and hashes
    a string, and the index is sized once instead of rehashing as it grows.
  - `fallocate`: in direct mode an extent file is no longer passed to it when
    the previous run closed clean and the file already has all its blocks. On
    XFS the call cost a transaction per extent file even when it allocated
    nothing. After an unclean exit every extent is still passed to it,
    because there the call also waits for direct writes the previous process
    left in flight; a new extent and one with holes are allocated as before.
  - Clean close: since a clean marker now lets the next start leave that
    call out, a store that is closed while a payload write call is still in
    flight does not write the marker. It logs a WARN with the number of
    write calls (a batch is one call) and the next start takes the unclean
    path, as after a kill. The server stops its request paths and RAM-tier
    flushers before it closes the disks, so an orderly stop is not affected.
  - Measured with the test `DiskSlabTest.OpenCostBench` (400,000 objects of
    4 KiB in 1,968 extent files, clean reopen, direct mode): 871-945 ms
    before, 257-321 ms after, of which 15 ms for the extents, 18-31 ms for
    the table scan and about 225 ms for the restore (the allocator's own bulk
    restore is most of that and is unchanged).
  - New INFO line per disk at startup: `slab <dir>: opened in N ms, K objects
    restored (extents A ms, allocation skipped for S of E; table scan B ms;
    restore C ms)`. The `allocation skipped` part appears in direct mode
    only. See `docs/DEPLOY.md`.
- Server and MDS: the accept loops of the RDMA bootstrap listener
  (`dfkv_server`) and of `dfkv_mds` no longer take the process down or spin
  when the host runs short of threads or file descriptors.
  - A connection whose thread cannot be created is refused and counted. In
    `dfkv_server` this used to be an uncaught exception from the accept
    thread, which ended the process and with it the RAM tier contents.
  - `accept()` failing for lack of descriptors or memory (`EMFILE`, `ENFILE`,
    `ENOBUFS`, `ENOMEM`, or any error that is not about the one connection) is
    counted, logged at a bounded rate, and retried after 100 ms. It used to be
    retried at once, which pinned a core for as long as the shortage lasted.
    Errors of a single connection (`ECONNABORTED`, `EINTR`, ...) are retried
    at once as before. Stopping the process is not delayed by the wait.
  - RDMA bootstrap: the peer's two bootstrap frames must arrive within 10 s
    of `accept()`. The 10 s used to apply to each `recv` separately, so a
    peer sending a byte at a time could hold a thread for minutes. A client
    sends both frames once it has created its queue pair, which comes after
    its `connect()`; that has to fit in the 10 s, as it already had to fit in
    the client's own bootstrap timeout.
  - RDMA bootstrap: at most `DFKV_RDMA_MAX_PENDING_HANDSHAKES` connections
    (default 4096, `0` = unlimited) may be waiting to deliver those frames;
    further ones are closed without a thread and counted. Established RDMA
    connections do not count and are not limited by this. A client holds a
    slot from `accept()` until its frames arrive, which includes the time it
    takes to create its queue pair, so when many clients reconnect at once
    (after a server restart, for example) the number pending is roughly the
    accept rate times that time. A refused client sees a failed bootstrap and
    retries. The limit is there for peers that connect and send nothing (port
    scans, health checkers that only connect, stuck peers);
    `dfkv_rdma_rejected_connections_total` growing during a mass reconnect
    means the limit is too low for the deployment and should be raised.
  - `dfkv_mds` fits `DFKV_MDS_MAX_CONNS` to the open-file limit at startup.
    Serving N connections takes up to `2N + 128` descriptors (each handler
    also dials etcd), 8320 for the default 4096, while a service usually
    starts with a soft limit of 1024. The MDS now raises its soft limit as far
    as the hard limit allows and, if that is still short, lowers the
    connection cap to what fits and says so in a WARN line. Before, the cap
    could not be reached: descriptors ran out first, and then the listener
    and the MDS's own etcd requests failed together. The sample unit in
    `docs/DEPLOY.md` now sets `LimitNOFILE=65536`.
  - New metrics: `dfkv_rdma_pending_handshakes`,
    `dfkv_rdma_rejected_connections_total`, `dfkv_rdma_accept_failures_total`,
    `dfkv_rdma_thread_start_failures_total`, `dfkv_mds_max_conns`,
    `dfkv_mds_accept_failures_total`, `dfkv_mds_thread_start_failures_total`.
  The wire protocol and existing metrics are unchanged. The TCP data listener
  and the metrics HTTP listener are not changed by this entry.
- Client, host dedup (`DFKV_CLIENT_NODE_DEDUP=1`): a lookup that another
  process on the host is fetching no longer waits out
  `DFKV_NODE_DEDUP_WAIT_MS` (default 500 ms) when that process has nothing
  to share. Until now a waiter treated only a published result as final, and
  `dfkv_batch_get`, `dfkv_batch_get_auto` and `dfkv_batch_exist` waited key
  by key, each key with a deadline of its own, then fetched the key with one
  scalar call. Every key whose owner gave up (a miss, a peer in cooldown, an
  empty ring, an I/O error) therefore cost the other processes the full wait:
  a batch with a hundred such keys took close to a minute where the same
  call without dedup takes milliseconds. Three changes:
  - a waiter returns as soon as no fetch is in flight for its key (the owner
    aborted, or the entry was invalidated) or the published entry is one it
    cannot use (overwritten in the arena, or another length), and fetches
    for itself;
  - a batch has one deadline for all the keys it waits on, and the keys it
    did not get are fetched in one direct batch. A hung or killed owner now
    costs a batch one wait instead of one per key;
  - `dfkv_batch_exist` shares "does not exist" whichever way the transport
    reports it. Over TCP it arrives with an OK status, which was treated as
    a failure, so every absent key was aborted and never shared.
  The one deadline has a cost when the owner is slow rather than failed. An
  owner publishes its keys only after its whole direct batch has returned.
  If that takes longer than `DFKV_NODE_DEDUP_WAIT_MS`, every key still being
  fetched is read again by each waiting process in one direct batch at the
  deadline, and nothing of that batch is shared. Before, the key-by-key
  waits added up, so a waiter usually outlasted a slow owner and shared most
  of the keys. Set `DFKV_NODE_DEDUP_WAIT_MS` to at least the time the
  largest batch takes; a batch that takes longer is read twice. The default
  is unchanged.
  Mixed versions on one host are safe: the shared-memory layout, the slot
  states and the segment name are unchanged, and only the waiting side
  behaves differently. An older process next to a newer one keeps its old
  waiting behaviour and the newer one returns early whichever of them owns
  the fetch. `dfkv_client_dedup_wait_timeouts_total` keeps counting every
  wait that ended in a direct fetch, early or not. The device (GPU dedup)
  path is unchanged: it already had one deadline and one fallback batch,
  and still waits that deadline once when an owner gives up. Processes that
  do not enable dedup are unaffected; the wire protocol is unchanged.
- Membership: a cache node whose disk store has failed for good now leaves
  the placement ring. A slab store whose metadata could not be written or
  synced (or that lost io_uring completions) stores and reads nothing more
  until the process is restarted: PUT and GET fail with an I/O error, Exist
  and Retain answer a miss, and only what a RAM tier in front of it still
  holds is served. The node kept renewing its membership as ring-eligible
  all the same: clients kept routing its share of the keys to it, where
  writes failed and reads missed for as long as the process ran. The store
  now also logs one ERROR line when it fails, naming the cache directory,
  the operation and the errno. Only RDMA
  rail health was part of the heartbeat, and a node without an RDMA listener
  reported no health at all. Now the next heartbeat (within 10 s) reports the
  node as not ring-eligible, through the existing health extension of the
  member record (HLT1), so the MDS and clients need no change: the MDS leaves
  the node out of the list clients build their ring from, and its keys move
  to the other members once. The node stays registered: `dfkvctl ring` shows
  it as `DEGRADED` with `store=failed` appended to INFO, until it is
  restarted or replaced (`deploy/dfkv_node_replace.py` handles an old node
  that is registered but out of placement). The process does not exit on its
  own: a restart resets the failed store and rejoins, and with a disk that
  is really broken a supervisor's restart loop would move the keys back and
  forth. The condition is a latch that only failed local metadata I/O sets
  and nothing clears, so it cannot make the ring flap. One failed disk of a
  multi-disk node takes the whole node out. A RAM tier that dropped an entry
  it could not flush still turns `/healthz` to 503 but does not take the
  node out of placement, because that node serves every request. Not
  covered: data-plane threads that are stuck while the process is alive.
  `dfkv_server_ring_eligible` is now what the node reports (no failed disk
  store and, with an RDMA listener, a healthy rail) and is also emitted by
  nodes without an RDMA listener. New alert `DfkvGroupMembersOutOfPlacement`
  on `dfkv_mds_group_degraded_nodes`. Mixed versions: an MDS older than
  2.15.0 ignores the health extension and keeps the node in the ring as
  before; a healthy node sends exactly what it sent before. See
  `docs/DEPLOY.md`.
- Server: once its disks are clean, a stopping `dfkv_server` exits instead of
  running destructors. What was left after the clean marker was only giving
  memory back, and most of that time went into freeing the disk indexes one
  heap node per cached object: about 4 s of a 5 s stop with 13 million
  objects. The kernel releases the same memory when the process ends. On a
  serving node the stop went from 5.2-7.5 s to 1.6-1.7 s. Sanitizer builds
  still return from `main` and run the destructors.

## 2.37.0 — 2026-11-22

- Observability: a failed RDMA work completion can now be attributed to a
  peer, a local device and a completion status on both sides. Until now the
  client logged nothing for it and counted it only in the process-wide
  `dfkv_rdma_cq_errors_total`; the server logged the status number without
  the peer address or the device and counted it without the status.
  Client: new WARN line `rdma client CQ failure peer=... dev=... op=...
  status=<n> status_name=<name> vendor=<n> ...` when a connection is torn
  down because a completion failed, and new counter
  `dfkv_rdma_client_rail_wc_errors_total{dev,status}`. Server: the existing
  `rdma server CQ failure` line, still one per failed connection, keeps its
  fields and gains `status_name=`, `dev=`, `peer=<ip:port>` and `age_ms=` at
  the end; the `rdma conn:` line gains `dev=` and `peer=`; new counter
  `dfkv_rdma_rail_wc_errors_total{dev,status}`, which adds up to
  `dfkv_rdma_rail_completion_errors_total{dev}`. `status` is the verbs
  status name without its `IBV_WC_` prefix (`retry_exc_err` is status 12).
  Peer addresses are not a metric label; they are in the log lines. The new
  client line is limited to one line per (peer, device, status, vendor
  syndrome) and 32 such combinations per 10 s; `suppressed=` on a line is the
  number of failures not logged since the previous line. The per-status
  series appear with the first failure of a (device, status) pair, so keep
  alerting on the existing totals and use the new series to break an alert
  down.
  Existing metrics and their labels are unchanged; retry, cooldown and
  failover behaviour are unchanged. See `docs/METRICS.md`.
- Server: the RAM tier arena asks for transparent huge pages
  (`madvise(MADV_HUGEPAGE)`, effective where the system setting is `madvise`
  or `always`). With 4 KiB pages the kernel hands out about 4 GiB/s no matter
  how many threads touch the arena, and taking the pages back at exit is slow
  as well. Measured with 192 GiB: resident in 3.3 s instead of
  39-47 s, released in about 1 s instead of 13-21 s; more threads, batched
  population and parallel release did not help with small pages. On a serving
  node the stop went from 27 s to 5-6 s. Reads from the RAM tier got faster as
  well: GET of 1 MiB objects by 30-40 %, of 2.74 MiB objects by 5-20 %; PUT
  and 64 KiB objects did not change. The startup log line and the new gauge
  `dfkv_ram_arena_hugepage_bytes` say how much of the arena huge pages
  actually back: the kernel falls back to small pages where it has none to
  give, and on a host with fragmented free memory it compacts memory while the
  arena is pre-faulted, which can take longer than small pages did (not
  measured; size the start timeout for at least the small-page time, with
  margin). `DFKV_RAM_TIER_HUGEPAGES=off` does not ask; the system setting then
  decides, as before this change (with `always` the arena is on huge pages
  either way, and the gauge reads 0 because nothing is measured). The
  pre-fault now writes one byte per page instead of the whole arena.
- MDS: a member list that has lost members because etcd stopped taking writes
  is no longer handed out. Once its space quota is exhausted etcd refuses
  writes and lease grants but still answers reads and keeps expiring leases.
  Members then cannot renew, and where registrars rotate over three or more
  MDS endpoints every member key runs out within one lease TTL. `ListMembers`
  and `ListTopology` returned the shrinking, finally empty view with `kOk`;
  clients dropped the missing peers' connections at the first such answer and
  adopted an empty ring after three polls, with every cache node healthy. The
  MDS now remembers each member whose heartbeat it received but could not
  write, and answers `kIOError` to a member list from which such a member is
  missing. Only heartbeats count: a registrar sends one only after a
  registration of the same process went through, so a node that cannot get in
  (a first registration that keeps failing) is not taken for a missing member.
  Clients of every version keep their last ring when a list fails, so deployed
  clients are covered once all MDS instances run this version; an MDS that has
  not been upgraded still serves the broken view. The list is served again
  when each of those members is back in etcd, written by this or another MDS,
  or has not heartbeat at this MDS for `DFKV_MDS_UNWRITTEN_HOLD_MS` (default
  120000, four lease TTLs; `0` restores the previous behaviour; it has to
  exceed the number of MDS endpoints times the 10 s heartbeat interval). A
  failed heartbeat whose member etcd still lists withholds nothing; other
  groups and `ListClients` are unaffected. Members leave as before, with one
  exception: after a member's heartbeat failed at an MDS and the member then
  really went away, that MDS withholds the list for up to the hold. Not
  covered: an MDS that has no record (just started, or not among the
  registrars' endpoints) serves the etcd view as before; nodes restarted
  during the outage only register, so a group whose nodes were all restarted
  is still served as empty; with four or more MDS endpoints the MDS that last
  wrote a member hears from it again only after its key has run out and may,
  for about 10 s per endpoint beyond three, serve a list without it. While a
  list is withheld a newly started client gets no ring and `dfkvctl` reports a
  failed MDS query. New counter `dfkv_mds_list_withheld_total`, gauge
  `dfkv_mds_unwritten_members`, a WARN line at most every 30 s, and the alert
  `DfkvMdsMemberListWithheld` (lists withheld for four minutes). No protocol
  change.
  MDS now remembers each member whose register or heartbeat it received but
  could not write, and answers `kIOError` to a member list from which such a
  member is missing. Clients of every version keep their last ring when a
  list fails, so deployed clients are covered once all MDS instances run this
  version; an MDS that has not been upgraded still serves the broken view.
  The list is served again when each of those members is back in etcd,
  written by this or another MDS, or has not heartbeat at this MDS for
  `DFKV_MDS_UNWRITTEN_HOLD_MS` (default 120000, four lease TTLs; `0` restores
  the previous behaviour). A failed heartbeat whose member etcd still lists,
  members that really leave, other groups and `ListClients` are unaffected.
  While a list is withheld a newly started client gets no ring and `dfkvctl`
  reports a failed MDS query. New counter `dfkv_mds_list_withheld_total`,
  gauge `dfkv_mds_unwritten_members`, a WARN line at most every 30 s, and the
  alert `DfkvMdsMemberListWithheld`. No protocol change.
- Tools: `deploy/dfkv_membership_audit.py` and `deploy/dfkv_node_replace.py`
  work again. Both parsed the table printed by `dfkvctl ring`, whose header
  and columns have since gained eligibility, status and IB health: with a
  non-empty ring the audit reported every MDS as `mds_unreachable`
  (`dfkv_membership_audit_ok` stayed 0) and a replacement stopped in its
  preflight. `dfkvctl ring --json` (and `topology --json`) prints the same view
  as one JSON document, and the scripts read that; the table itself is
  unchanged. The scripts need the `dfkvctl` of this release. The ring epoch
  they report is now the document's `placement_epoch`, the value the MDS and
  clients compare; their own calculation left out eligibility, never matched
  it, and is gone. The audit also reads a registration whose strings are not
  UTF-8 instead of giving up on it: byte for byte, the way `ring --json`
  writes such strings (`\u00XX` for every byte outside printable ASCII), so
  that member is compared like any other and gets the issues that are its
  own. One such registration used to make the audit report the whole group as
  `etcd_unreachable_or_invalid`.
- Tools: what `deploy/dfkv_node_replace.py` requires of the two nodes.
  - The replacement has to be ring-eligible (own vnodes) before the old node
    is stopped. The old node only has to be registered: one the MDS already
    holds out of placement, with all its rails down for instance, can be
    replaced, and a rollback then only waits for it to register again.
  - Readiness is the replacement's own row of `dfkvctl stat --all`. That
    command exits 1 as soon as any member of the group is unreachable,
    unhealthy or of unknown health, and the script took the exit status for a
    failed command: one such member -- the node being replaced, say -- kept the
    replacement from ever counting as ready. The row has to show what
    `dfkvctl` itself requires of it (reachable, READY and HEALTH `yes`, MDS
    and RDMA not `NO`).
- Tools: `deploy/dfkv_node_replace.py` no longer waits for `systemctl stop`
  under `--command-timeout` (10 s by default, while a stop takes from tens of
  seconds to minutes). The timeout started a rollback that asked systemd to
  start the node it was still stopping and timed out on that as well, so the
  script exited with "rollback also failed"; systemd runs such a start once
  the stop is over, which can bring the old node back next to its
  replacement. Stops and starts are now queued with `--no-block`.
  - A replacement is complete when the old node's registration has expired
    and the replacement is placed and ready. A stopping server stops
    heartbeating first, so that takes about the lease time however long the
    unit goes on flushing; the script does not wait for the unit.
  - A rollback has to see the old unit through its stop, because systemd runs
    the start only then. It waits for the unit to be active again for up to
    the new `--stop-timeout` (660 s by default; set it above the unit's
    `TimeoutStopSec`) plus `--timeout`, and only then looks at the ring.
    When it cannot confirm the restart, the error says that systemd already
    has the start instead of asking for one.
- Server: the disks of a node are opened at the same time instead of one after
  the other. A slab store reads its whole table back before it can serve, so
  startup grew by seconds per disk: on a multi-disk node with a full cache,
  34-51 s passed before the RAM tier even started to pre-fault; it is now
  about 9 s, also on the path that keeps the cache after an unclean exit
  (each disk syncs and rebuilds for 4-7 s there). With more than one disk, one
  log line,
  `disk stores opened: N disks in S s (slowest ...)`, reports the time. The
  scratch memory of the table rebuilds (roughly 150-200 bytes per cached
  object, freed before the RAM tier is allocated) is now in use for all disks
  at once instead of one disk at a time. An engine that throws while opening
  is reported as a startup failure of that disk, so the disks that did open
  are closed in order. An engine factory (tests only) is now called from one
  thread per directory.
- Server: a stopping `dfkv_server` gets its disks clean before it releases
  memory. The clean marker used to be written by the disk stores' destructors,
  which run after the RDMA registrations are released and the RAM tier arena is
  unmapped; with a large arena that is tens of seconds during which a stop
  timeout (SIGKILL) left every disk dirty, and a dirty disk is reset at the
  next start. The order is now: stop RDMA and TCP requests, write the RAM
  tier's backlog, sync all disks and mark them clean, then release memory.
  - The disks sync at the same time instead of one after the other.
  - The RAM tier's backlog gets `DFKV_RAM_SHUTDOWN_DRAIN_TIMEOUT_S` (default
    30, as before). **Behaviour change:** when the time is up, what is still
    only in RAM is dropped and the disks are closed clean; until now the
    timeout only logged a warning and the flush went on without a limit, with
    the clean marker behind it. Raise the setting (up to 3600) to keep the old
    trade-off; keep it below the service manager's stop timeout. The limit
    counts from the start of the stop. The value is now parsed strictly: a
    non-number or a value with a unit falls back to 30 with a warning instead
    of being read as 0 or as its leading digits, and a value above 3600 is
    clamped instead of ignored.
  - Every step logs one `shutdown: ...` line with its duration, the drain
    reports its progress every 5 s, and `dfkv_server shut down cleanly ...
    releasing memory` marks the point after which a kill is harmless.
  - `KvNodeServer::Shutdown()`, `DiskCacheGroup::Close()`,
    `StoreEngine::Close()`, `RamTier::StopBackground()` and
    `RdmaServer::Quiesce()` are the pieces; `Stop()` and destruction behave as
    before for embedders and tests that do not call them.

## 2.36.0 — 2026-11-22

- Server: the RAM tier arena is no longer registered as a whole with every
  RDMA rail. Nothing on the server referred to that registration any more
  (each zero-copy GET registers an exact MR over the value it serves; PUTs
  land in the receive pool), but it pinned the arena once per rail, serially,
  at startup and unpinned it at shutdown. Measured with a 192 GiB arena and
  eight rails: start to ready 128-142 s -> 69-88 s, stop under load 34-42 s ->
  about 25 s, GET and PUT throughput unchanged. `RdmaServer::RegisterMemory` is gone with it. The
  arena stays resident: it is now locked with `mlock2(MLOCK_ONFAULT)` while it
  is pre-faulted (`DFKV_RAM_TIER_LOCK=off` disables that; a failed lock only
  warns; `dfkv_ram_arena_locked` shows the outcome). A TCP-only node, whose
  arena was never pinned, gets the lock as well. New counter
  `dfkv_rdma_pull_zerocopy_register_failures_total`:
  per-read registrations that failed and fell back to a staged copy -- what a
  too-low memlock limit used to turn into a failed startup.
- Server: opt-in, the slab cache survives an unclean exit when the kernel did
  not go down with the process. A dirty `slab_state` resets `slots.tbl`, so a
  server killed by a stop timeout or the OOM killer comes back empty. With
  `DFKV_SLAB_WARM_RESTART=auto` the table is instead rebuilt as after a clean
  shutdown, provided that the boot id, the unique mount id (Linux 6.8) and the
  directory recorded in `slab_state` still match, the previous run's trust
  token is still on `/dev/shm` (a run gives it up on a metadata I/O failure,
  during every table sync, in the new fatal-signal handler when the server
  itself crashes, and while a recovery is unfinished), the filesystem is xfs,
  ext4 or tmpfs, and the table and extents sync cleanly.
  Everything else still resets, and records that contradict each other reset
  on that path rather than failing startup. `boot` also accepts runs without
  a unique mount id; `off` is the default and the behaviour of 2.35. New
  counter `dfkv_slab_warm_restarts_total` and gauge
  `dfkv_slab_trust_tokens_held`; the reason for a reset is logged.
  The state record stays readable by 2.35, whose own records are treated as
  "reset". Independent of the setting, a cache directory is now locked: a
  second process opening it fails to start.
- Server: once a `Remove` has returned successfully, the key's record in
  `slots.tbl` is invalid as far as the next process to read the table is
  concerned (the write is in the page cache; syncing it stays with the
  periodic table sync). While a read or write of the key was in flight the
  removal is deferred, and until the last holder finished the record stayed
  valid (a read in flight) or could briefly become valid (a write in flight
  wrote its record before it learned about the `Remove`). Nothing observed
  this so far, because a restart after an unclean exit resets the table; it
  is a prerequisite for keeping the cache across such restarts. PUT records
  are now written at commit, under the store mutex.
- Server: `dfkv_rdma_active_conns` and `dfkv_rdma_rail_active_conns` now drop only
  after a closing connection has released its endpoint, receive-segment lease,
  source pins and connection accounting. They used to drop first, so "no
  active connections" could be observed while `dfkv_rdma_recv_segment_used_bytes`,
  `dfkv_rdma_pull_connections` and `dfkv_rdma_connection_bytes` still counted
  the connection -- visible to anything that waits on the gauge before reading
  the others, including one of the loopback tests on a real NIC.
- Client: libdfkv no longer loads or initializes CUDA. Classifying an
  unregistered GET destination (and `RegisterMemory`) used to
  `dlopen("libcuda.so.1")` and call `cuInit` on first use, so every dfkv
  client process on a GPU host -- schedulers, tools, benchmarks, unit tests --
  opened the driver devices even though it only ever handled host memory. With
  one driver build (595.71.05 open kernel modules, UVM HMM enabled) that was
  enough to hit a kernel NULL dereference in `uvm_hmm_unregister_gpu` when
  such a process exited. `CudaLib` now only attaches (`RTLD_NOLOAD`, no
  `cuInit`) to a library the host process has loaded and initialized itself;
  until then every pointer is host memory. The attach is retried, so a
  framework that brings CUDA up after the client was created is picked up:
  immediately by `RegisterMemory`, and within 100 ms by the per-GET
  classification of unregistered destinations. A C++ program that is its own
  CUDA host and relied on dfkv to initialize the driver must now do that
  itself or call the new `CudaLib::LoadForHost()`.
- Server: a tenant without a node quota no longer serializes on the tenant
  admission stripe. `DiskCacheGroup` took the stripe unconditionally in
  `Cache` / `CacheDirect` / `CacheDirectBatch` / `Remove` and held it across
  the storage write, so with a single tenant (the connector default) only one
  RAM-tier flush batch -- or one direct PUT -- was writing at any time,
  regardless of `DFKV_RAM_FLUSH_THREADS`. Quotas are immutable after startup;
  unlimited tenants now skip the stripe and the admission lookup entirely.
  Tenants with a quota keep the existing strict serialization. Same-key
  ordering for unlimited tenants is left to the storage engine, and the slab
  engine closes the one gap that exposed: a PUT arriving after a REMOVE had
  returned could join the in-flight write that REMOVE cancelled and be told
  `kOk` for a value that was never published. It now waits out the removal
  and stores its own bytes.
- Metrics: `dfkv_evicted_bytes_total` counts watermark evictions again. Since
  v2.0.0 the watermark reclaimer dropped its victims' payload accounting
  without adding it to the counter, so the metric only reflected inline `Put`
  eviction (single and batch), extent rebalance and class-local reclaim while
  `dfkv_evictions_total` counted every victim. Expect the rate of this
  counter to step up after upgrading; that is the corrected value.
- Server: a tenant with a node quota can PUT over RDMA again when the RAM
  tier is enabled. `CacheDirectForKey` gated both the RAM write-back branch and
  the disk branch on "tenant has no quota", so a limited tenant skipped both:
  every direct PUT returned `kCacheFull`, stored nothing, and was not counted
  as an error or a quota rejection. The TCP entry was not affected. Limited
  tenants still bypass RAM write-back admission and now fall through to the
  quota-checked disk write, which reports `kQuotaExceeded` when over quota.

## 2.35.6 — 2026-11-15

- Client per-node stats: a shard timed at 0 us (sub-microsecond) no longer
  drops its rail from the `pernode-get`/`pernode-exist` line. The merge kept
  `dev` only from a shard strictly slower than the running maximum, so a
  0 us shard never set it; this also made
  `ClientPernodeStats.RetryRailPathPreservesAttemptOrderAndSingleRail` flaky.

## 2.35.4 — 2026-11-11

- vLLM connector: partial hash hits for DCP + hybrid linear-attention models.
  When vLLM exports `KVCacheConfig.cache_hit_alignment_tokens` below the
  scheduler block (hash-block prefix hits, e.g. Kimi-Linear-48B-A3B TP8 DCP8:
  KDA state blocks 256, attention 256 x dcp 8 = 2048), lookups land on the
  hash block, ported from vLLM's Mooncake store connector: the first SAVE
  that covers the prompt also stores the attention block holding the prompt
  checkpoint `floor((P - 1) / 256) * 256` whole, under that checkpoint's
  hash, next to the recurrent states vLLM hands off there (an offload-only
  SAVE when the hand-off step has no other SAVE). A hit inside an attention
  block loads that object into a whole block; core recomputes the rest.
  Lookups probe whole attention blocks, the hash boundaries inside the first
  block not stored whole, and states at those and the scheduler block
  boundaries, under every rank namespace; an external hit is declined when
  the local prefix hit ends inside a block (core's copy-on-write of that
  block would race the load), and block free waits for a tail-only SAVE.
  Object sizes and key formats are unchanged; engines without the attribute,
  or with hits on the scheduler block, keep the previous behaviour and the
  off-grid hand-off warning below.

## 2.35.3 — 2026-11-10

- vLLM connector: warn (rate-limited, with a running count) when the engine
  hands off recurrent-state boundaries off the scheduler block grid. Saves and
  lookups use the LCM of the group block sizes; under decode context
  parallelism that LCM (attention block x dcp_size, e.g. 2048 or 16384)
  exceeds the linear-attention state block (256 or 2048), and vLLM hands off
  state only at the state-block-aligned prompt end, so hybrid prefixes were
  silently never saved and never hit (Kimi-Linear-48B-A3B, TP8 DCP8: 0
  external hits). Saves and lookups are unchanged; engines that align the
  hand-off to the LCM do not trigger the warning.

## 2.35.2 — 2026-11-10

- Client: warn at startup when `DFKV_NODE_DEDUP_TTL_MS` is below
  `DFKV_NODE_DEDUP_WAIT_MS`. A published node-dedup result is claimable only
  for the TTL; a peer rank that reaches it later (rank skew, a large leader
  batch) re-fetches the object instead of sharing it. Seen in production as
  `DFKV_NODE_DEDUP_TTL_MS=100` against the 500 ms default wait.
- SGLang HiCache: declare `DfkvHiCache.supports_dcp_storage = True` so SGLang
  builds that gate HiCache L3 under decode context parallelism admit the dfkv
  `dynamic` backend, and probe each host pool's layout over one whole logical
  page (`page_size x dcp_size` slots) at registration; under DCP
  `get_page_buffer_meta` rejects a bare physical page. The span is computed
  locally rather than imported from SGLang, so builds without the DCP storage
  helpers load unchanged; non-DCP keys and transfers are unchanged.
- Docs: `load_async=True` validated on a KDA+MLA hybrid (Kimi-Linear-48B-A3B,
  TP8): L3-restored prompts reproduce their cold outputs and the same
  external/local/computed token accounting as `False`. Recommendation updated;
  DSA-indexer hybrids (Kimi K3.1) stay on `False` until validated.

- vLLM connector: reuse a waiting request's completed lookup for
  `lookup_cache_ttl_ms` (default 1000, `0` disables). vLLM re-polls every
  waiting request on every scheduler step, and each poll was a synchronous
  rank-0 round trip plus a retention REFRESH of every key of the prefix. With
  the KV pool full, a 256K-token Kimi K3.1 request was probed ~140 times while
  queued (5.7% of scheduler wall time, 26x more refreshed keys than fetched
  keys). Entries are keyed by request, prefix length and retention TTL and are
  dropped on finish, abort, preemption and failed load; RPC errors are not
  cached.

## 2.34.0 — 2026-10-30

- Stop whole-extent steals from wiping the newest residents. Cross-class
  steals (`StealFrom`, `StealExtentFor`) ranked extents by decayed read heat
  and then by fewest useful bytes; once read heat had decayed, that picked the
  extent still being filled by new writes, so fresh objects were evicted in
  bulk while extents of old residents survived. Steals now take an empty
  extent first, otherwise the extent whose residents are least recently used
  on average (per-resident admission/read sequence), with heat and useful
  bytes only breaking ties. The disk and RAM-tier background rebalancers also
  judge cold donor classes by bytes written instead of object counts, so a
  large-slot class (for example Kimi-K3 recurrent state next to its KV pages)
  is no longer treated as cold and drained. Reproduced on a 3-disk slab server
  with a RAM tier where watermark reclaim stays idle: before, new KV probes
  were lost while the oldest KV survived 80% capacity of later writes; after,
  the oldest residents go first.

## 2.33.3 — 2026-10-24

- Save vLLM connector recurrent-state (Mamba) objects only from the blocks
  the engine hands off for committed boundaries (`kv_connector_block_state`),
  instead of locating them by position in an append-only mirror of the
  block table. With MTP the engine's last cacheable boundary backs off one
  block, so a prompt's final chunk can cross a boundary while its slot still
  holds the running state; the positional save then published that running
  state under the boundary's key and a cross-instance load decoded garbage
  (reproduced on GLM-5.3-Flash at the default 8192-token budget with 2.33.1).
  Engines without the hand-off keep 2.33.2's fail-closed rule; with the
  hand-off this also restores the external hits 2.33.2 lost for prompts
  prefilled in one step.

## 2.33.2 — 2026-10-24

- Fail closed on vLLM connector recurrent-state (Mamba) SAVEs whose
  boundary was not computed in the current step. The connector locates
  state blocks by position in an append-only mirror of vLLM's block table,
  which vLLM align mode does not keep append-only; such a SAVE now skips
  the state objects (logging a warning) so the prefix is never admitted,
  instead of possibly publishing another position's state.

- Make vLLM connector KV events survive cross-worker aggregation. Each
  BlockStored event now chains to its predecessor in the request instead
  of to the previous chunk this rank stored, so all ranks emit identical
  events. Replicated (MLA) payloads are striped across TP ranks, so one
  report per block now suffices for those groups while TP-sharded state
  still needs every rank; workers with no events still count toward the
  quorum. Previously, striped MLA events at TP > 1 were almost all dropped.

- Make vLLM connector lookup probe every key coordinate that the TP x PCP
  workers of a pipeline stage write, instead of only the lookup rank's own
  PCP/DCP coordinate. Under MLA with DCP, a prefix missing another DCP
  rank's shard was admitted and then failed to load; under GQA with DCP the
  lookup probed coordinates nobody writes and never reported a hit.

- vLLM connector fixes carried over from upstream MooncakeStore and
  vllm-ascend: skip pending loads that MultiConnector assigned to another
  connector (no SAVE from blocks it is filling); refuse sleep mode, whose
  page remap invalidates GPUDirect registrations; key the lookup IPC path
  by the global DP index so colocated external-LB engines do not share it;
  and derive the PP stage with PCP in vLLM's rank layout.

- Track cached prefill progress from vLLM's computed token position even
  when a step allocates no new block. Previously the connector skipped the
  update, so one sub-block prefill chunk permanently desynchronized the
  request tracker; a later SAVE could then select a retired source page
  and publish bytes that did not match the key. Keep the tracker's token
  IDs in step with that position for KV events, skip zero-token steps
  without an allocation, and resynchronize a lagging tracker with a
  warning.

## 2.33.0

- Reclaim globally by bounded slot-level read/write recency instead of dropping
  whole extents at the disk high watermark. New admissions and successful reads
  move behind the scan cursor; pins and metadata-invalidation failures still
  prevent reuse. Successful new writes can renew recent pressure below high,
  but inactive pressure history expires after 128 completed quiet observations.
- Replace `DFKV_SLAB_EVICT_MAX_EXTENTS_PER_TICK` with
  `DFKV_SLAB_EVICT_MAX_VICTIMS_PER_TICK` (default 256 slots; scan cap four times
  the victim cap). Watermark eviction counts now measure objects; the obsolete
  max-extents gauge is replaced with victim/visit gauges.
- Read logical HiCache anchor markers through native GET on consumption,
  refreshing their recency and propagating missing markers rather than
  reporting unconditional success. Existence probes remain non-promoting.
- No wire protocol, native client ABI, namespace or persisted-format change.
  Existing compatible clients may continue to connect; logical-anchor HiCache
  users should update the Python connector for marker recency/miss handling.
  The vLLM and LMCache connector versions follow the release, without a new
  compatibility requirement from this change.
- PUT tail-latency overhead and cross-workload/long-term stability remain
  qualification risks; bounded reclamation does not guarantee zero overhead.

## 2.32.0

- Skip synchronous disk-follower queue scans when a connection has no deferred
  follower. Pinned-RAM dynamic GET and RELEASE replies keep the existing ordered
  queue, exact READ grants, revocation, and SEND-buffer fences.
- Cover duplicate sliced disk reads, missing keys, and out-of-order disk
  completions in one real-RDMA `RangeMany` window with coalescing enabled.
- Document matched dynamic-GET performance comparisons. Controlled public
  2.30/2.31 runs did not reproduce a previously observed small-object slowdown;
  the queue-scan optimization reduces CPU work but does not establish a material
  end-to-end throughput gain. Protocol and v2.28 default-client compatibility
  are unchanged.

## 2.31.0

- Make every RDMA GET, including string `Range`/`RangeMany`, use dynamic
  one-sided READ grants with explicit RELEASE/ACK. Remove responder-WRITE GET,
  fixed pull arenas, writer cancellation/CQ-drain ownership and the
  `DFKV_RDMA_DYNAMIC_PULL` escape hatch. PUT WRITE operations remain.
- Give new clients a dynamic-only handshake and token-free 25-byte readiness.
  Servers retain a bounded v2.28 default-client adapter: legacy dynamic
  bootstrap gets its original 33-byte readiness and a real live-connection
  token. Retirement proofs validate that connection and the invariant that
  this server never issues GET WRITE; unknown/closed tokens fail.
- Upgrade servers first, then clients. New clients reject old servers;
  v2.28 clients with dynamic pull disabled and historical WRITE GET APIs are
  not supported. Do not roll a server back after activating new clients
  without restoring matching clients.
- Promote disk reads into the RAM tier only when the returned bytes contain
  the complete object. Cold slices can no longer replace the whole cached
  value. Keep offset, full stored length and returned slice length distinct.
- Preserve io_uring preparation and source ownership for dynamic GET,
  depth-windowed string batches, whole-object capacity rejection and
  explicit release of empty EOF grants. Remove retired GET-WRITE/fixed-pull
  metric series rather than exposing misleading zero aliases.

## 2.30.0 — 2026-10-05

- Preserve an initial, hard-budgeted receive chunk for dynamic GET and
  leased-PUT staging when enough connection capacity remains. Connection
  leases cannot fragment that chunk; pressure recovery reclaims only
  quiescent, aged QPs and never revokes pending replies, disk reads, or
  client READ grants. Truly exhausted pools still report backpressure.
  Separate connection-eligible and staging-reserved free-space gauges.
- Resume a marked interrupted first slab initialization only when its
  geometry, empty table, and existing extents prove an unpopulated layout.
  Preserve unknown or populated stores, and report the original startup
  syscall errno (including soft file-descriptor exhaustion) instead of
  silently creating missing data.
- Add local and peer QP identities to failed server CQ diagnostics without
  suppressing completion errors. The historical intermittent retry-exceeded
  failure has not yet been attributed to a particular peer lifecycle.

## 2.29.0 — 2026-10-04

- Keep reply buffers behind their SEND fence while queueing early incoming
  requests in a depth-bounded FIFO. Control/PUT receives are ready before the
  reply is posted; receive-backed data sources retain their existing fence.
  This avoids both depth-one reply-buffer races and receiver-not-ready stalls
  in synchronous and io_uring loops, without timeout or retry changes.
- Reuse RDMA capability observations only within a known peer publication;
  invalidate them on endpoint failure, forced recovery and topology changes.
  Anonymous/static peers retain fresh-QP negotiation.
- Retain a bounded burst of idle data endpoints, retire excess endpoints after
  application inactivity, and keep explicit `DFKV_RDMA_POOL_MAX` limits exact.
  Keepalives no longer extend burst retention.
- Serve arena-resident dynamic-pull GETs through exact READ grants while keeping
  their arena pins until grant revocation. Account successful pull reads at
  release without double-counting fallback probes.
  Preserve the storage-handler latency boundary independently of client grant
  hold time.
- Apply the existing foreground PUT admission limit before RAM or disk work.
  Preserve admitted RAM first-writer completion ownership across conflicting
  duplicates, cancellation and remove/reinsert; do not bypass on a timer.
  At a positive RAM-ACK dirty watermark, reject only new keys before admission;
  this rejection never falls through to direct persistence. Zero watermark
  retains its explicit disk-ACK behavior.
  Keep admission and the first arena reservation under the same shard lock.
- Preserve the authoritative backing value after RAM eviction. A conflicting
  duplicate cannot create a RAM copy that differs from an existing disk key
  and then misinterpret an idempotent disk write as persistence of those bytes.
  Serialize foreground same-key PUTs across RAM admission and synchronous
  capacity bypass, so an in-flight disk owner cannot race a new RAM generation.
- Add per-phase connection diagnostics and optional registered PUT-source
  benchmarking. PUT and GET stall timing uses a monotonic clock and is disabled
  when not requested.

## 2.28.0 — 2026-09-22

- Add a read-only native `dfkv_max_block_bytes` query and startup layout checks
  for HiCache host pools and vLLM cache groups. Undersized effective object
  bounds fail before cache traffic instead of only surfacing as runtime misses.
  Explicit limits and runtime transport rejection remain unchanged. Updated
  connectors require the matching native library export.

## 2.27.2 — 2026-09-17

### v2.27.2 — Legacy and native vLLM compatibility

- Restore one connector package for engines with the legacy `get_finished`
  lifecycle and engines with native `get_transfer_results` support.
- On legacy V1 engines, carry failed request identities through supported
  worker metadata and enter the existing scheduler recovery/error path only
  after all participating workers finish. No fabricated block IDs or installed
  engine source-file patches are used.
- Preserve originating-child routing through nested `MultiConnector` metadata;
  do not retain or replay other connectors' historical metadata.
- Register the external connector's canonical name so MultiConnector can
  reconstruct cross-process statistics without terminating API output handling.
  Preserve an operator's existing registration of the same class.
- Keep serialized hybrid loads parked and fenced, quarantine failed external
  hits, and preserve cancellation/deferred-free ownership. Native engines
  continue to use their native request-failure protocol.
- Register CPU connector and worker lifecycle regressions in CTest, and cover
  old/native scheduler and executor behavior in their actual engine runtimes.
- Correct v2.27.1's native-only connector compatibility restriction. Legacy
  support requires the V1 worker-metadata, completion aggregation and scheduler
  recovery hooks; arbitrary V0 engines or partial protocol backports are not
  implied. No functional C++ changes, ABI/wire changes or stored-key migration.

## 2.27.1 — 2026-09-17

### v2.27.1 — Request-level hybrid KV load recovery

- Adopt vLLM's native `KVConnectorTransferResults` protocol for hybrid and
  multi-group loads instead of reporting ambiguous, group-less block IDs.
- Publish paired receive completion/failure only after native writes and the
  owning CUDA device are fenced; cover misses, malformed results, exceptions,
  queue rejection and cancellation without losing or duplicating outcomes.
- Keep `load_async=false` I/O serialized on the model thread while parking
  hybrid requests until receive completion. Explicit device synchronization
  also handles runners that defer the load hook until after a forward pass.
- Bypass failed external hits for the rest of a request and rebuild retry
  metadata from fresh allocations, including resumed MRV1 and MRV2 requests.
- Remove the obsolete hybrid invalid-block scheduler patch. This connector
  requires native `get_transfer_results` / `failed_recving` support throughout
  the vLLM runner, executor and scheduler; older runtimes need an engine upgrade.
- Preserve single full-attention block-level recovery and stored-key layout.
  No functional C++/server, native ABI or wire-format changes.

## 2.27.0 — 2026-09-13

### v2.27.0 — Runtime RDMA providers and native hybrid-state integration

- Include `ibverbs-providers` in the runtime image; exposing RDMA devices does
  not make them discoverable when the userspace hardware plugins are absent.
- Exercise real RDMA PUT/GET through the built runtime image when the CI
  hardware probe succeeds, rather than relying on version or TCP-only checks.
- Refresh the shipped SGLang hybrid Mamba/DSA patch for the pinned native
  GLM-5.3-Flash image, preserving indexer sidecars and actual scaled-MLA host
  geometry with the current allocator API.
- Add a bounded real-GPU offload/poison/restore regression for KV, indexer,
  temporal and convolution state, without model weights.
- Ship a targeted native vLLM GLM-5.3-Flash V2-runner patch: exclude the
  single-block kpool tail from generic position-indexed slot mapping, retaining
  the dedicated circular metadata builder and existing allocation policy.
- Document strict patch applicability and fresh cache generations. Native
  client/server wire formats and connector runtime code are unchanged.

## 2.26.4 — 2026-09-11

### v2.26.4 — SGLang DeepSeek-V4.1 HiCache deployment guidance

- Add the verified preview-image HiCache/L3 recipe and physical side-pool
  requirements, including separate DSpark layout isolation.
- Correct the object-bound configuration: the payload ceiling does not raise
  the default 4 MiB RDMA object declaration. Explicitly select the data fabric.
- Correct HiCache size units and distinguish GPU/host write-through from
  asynchronous L3 backup, and local cache salt from L3 key isolation.
- Document paired cold/GET-backed hot verification and workload-dependent
  cache benefits. No connector runtime or functional C++ changes.

## 2.26.3 — 2026-09-11

### v2.26.3 — vLLM acceptance coverage

- Replace the forwarding-only preemption test with a source-lifetime regression:
  block reuse must wait for the old SAVE, which must retain the original bytes.
- Cover consumer MRV1 synchronous/asynchronous recovery, prior-step/same-step
  preemption, continued prefill, decode and completion without duplicate LOAD
  or unauthorized SAVE.
- Clarify that compatible object layouts do not validate old cache contents.
  This release changes tests and documentation, not connector runtime behavior.

## 2.26.2 — 2026-09-11

### vLLM preemption safety

- Fence cancelled receive work and in-flight SAVE sources through vLLM's
  pre-forward `handle_preemptions` hook, before recycled blocks can be
  overwritten. Synchronous LOAD submission remains in `start_load_kv`.
- Preserve LOAD metadata for consumer-only MRV1 requests resumed through the
  cached-request path while continuing to suppress SAVE work for that role.

## 2.26.1 — 2026-09-11

### vLLM stable hybrid and speculative cache restoration

- Separate stable SAVE masks from LOAD requirements; exclude volatile draft
  state and validate required objects at each shortened lookup boundary.
- Bind full effective cache-spec geometry, including wrapped/DCP-adjusted
  specs, into the `vllm-multiwr-v4` identity. Earlier Python layouts cold-miss;
  the native C ABI and server wire protocol are unchanged.
- Reject unallocated LOAD destinations and incomplete source chunks instead
  of reading or publishing invalid cache state.
- Drain mutable/windowed SAVE sources before the next model step can overwrite
  them, while retaining overlap for immutable full-attention sources.
- Fence asynchronous GET completion on the owning CUDA device captured from
  the model thread, rather than the receive thread's default device.
- Preserve fresh allocation and LOAD state when preemption and resumption
  happen in the same scheduler step; clear the resume marker when MRV2
  introduces the request again.
- Restore MRV1 resumed requests from the complete allocated block table,
  including asynchronously loaded blocks, rather than the step's delta.
- Keep resumed SAVE completion accounting until finished GPU blocks can be
  released; invalidate old SAVE generations before reuse so queued or cancelled
  work cannot revive or decrement a new generation's counter.

## 2.26.0 — 2026-09-08

### Operation-scoped RDMA staging

- Large scalar and scatter/gather PUTs negotiate exact operation-scoped
  staging leases. Successful STORE revokes the remote MR before recycling
  storage; failed connections keep storage until QP/MR teardown completes.
- Direct scalar/SG GETs negotiate dynamic pull descriptors and explicitly
  acknowledge release after local READ completions. Supported endpoints no
  longer reserve a connection-lifetime pull arena.
- `DFKV_RDMA_MAX_BLOCK_BYTES` remains the common PUT/GET object ceiling.
  `DFKV_RDMA_INLINE_PUT_MAX_BYTES` defaults to 4 MiB (0 disables leased PUT);
  `DFKV_RDMA_DYNAMIC_PULL` defaults to enabled (0 retains legacy pull).
  Both capabilities are optional and negotiated before connection sizing.
- Preserve mixed-batch invalid results and first-write-wins ordering; reuse
  pooled legacy endpoints without repeated capability-probe/QP churn.
  Insufficient GET capacity returns `kInvalid` without cooling a healthy peer.
- Add delayed-DMA, MR/chunk reclamation, pressure, generation, old-peer and
  posted-READ abandonment regressions. Hardware CI invokes the lease suites
  in SYNC/io_uring and TSan modes and distinguishes skipped hardware probes.
- Replace the in-process memory benchmark with an external-server client:
  retain clients during idle sampling, record sampled server RSS/committed/
  leased peaks, and validate every successful PUT with byte-exact GET.
  Samples are not exact hardware high-water counters.

### Hybrid inference cache correctness

- Revalidate vLLM's length-dependent state mask after shortening a lookup
  prefix, and reserve the sampling tail before lookup rather than truncating
  an already validated hit.
- Preserve each cache group's effective physical block coverage instead of
  treating a small attention block as a whole scheduler-LCM region. Respect
  nonpersistent scratch pools without forcing them into the hash geometry.
- Store every TP-sharded recurrent checkpoint under its physical rank in the
  `mamba` pool; retain attention replica striping without eliding required
  state-shard clients. Old collapsed state keys and affected underfilled
  heterogeneous layouts cold-miss after the identity correction; native wire,
  raw payload and C ABI formats remain compatible.
- Include the tested SGLang engine patch registering the missing DSA indexer
  sidecar in hybrid Mamba stacks. Correct recurrent bytes alone do not make a
  complete DSA cache; the indexer must survive the same fresh-process reload.

### GPUDirect load memory ordering

- Fence device-direct GET completions with a post-batch device
  synchronization so kernels launched after the load observe the RDMA BAR
  writes; arming `CU_POINTER_ATTRIBUTE_SYNC_MEMOPS` at registration where the
  driver accepts it (VMM/cuMemMap pools reject it with a one-shot warning).
  Observed live as exact-hit loads whose generation degraded into empty
  reasoning until reordered. `DFKV_GPU_LOAD_FENCE=0` disables the fence.

## 2.25.0 — 2026-09-03

### SGLang Prefill-CP storage identity and physical rail affinity

- Fixed SGLang DP-attention rail affinity to use the world-group node-local
  process rank instead of TP/PCP coordinates. Eight TP-of-1 schedulers no
  longer collapse onto rail 0; missing physical rank now fails startup.
- Added the explicit `prefill_cp_storage_layout=replicated` contract for
  rank-replicated MLA Prefill-CP. All CP readers retain the deployed CP size but
  address the elected writer's `pcp_rank=0` key. DCP, DSA cache-layer split,
  sharded, and ambiguous layouts fail closed.
- Added HiCache parallel-identity and contiguous-prefix existence metrics, plus
  native per-rail successful logical PUT/GET operation and payload-byte
  counters.
- Fixed three latent device-direct backup branches that referenced an
  uninitialized page count/component count, and refreshed the CPU-only vLLM
  test shims for the current world-group and asynchronous-load contracts.

## 2.24.2 — 2026-08-30

### RDMA receive-pool default capacity

- Raised the default `DFKV_RDMA_RECV_SEGMENT_SIZE` hard budget from 16 GiB to
  128 GiB. The receive pool remains lazy: startup commits only the initial
  256 MiB chunk, so the larger default expands the growth ceiling without
  eagerly pinning or registering 128 GiB. Explicit operator overrides are
  unchanged.

## 2.24.1 — 2026-08-30

### vLLM long-context node-dedup load windows

- Added opt-in `load_window_keys` and `load_window_min_keys` connector settings.
  Large replicated loads can now publish bounded native GET results before the
  same-host GPU dedup wait deadline, while short requests retain one native GET.
- Defaults remain disabled. Values are validated at startup, preserve key/result
  order, and retain one outer connector metric for the logical load.
- On a production node, a 1,000,000-token TP8 GLM-5.3 hot load with 128-key windows
  reduced TTFT from 39.35 s to 22.92 s and reduced native remote fetches from
  approximately eight copies to one, with zero dedup fallback. A 4,096-key
  threshold kept the 65,536-token/C10 path unwindowed at 35,622 total tok/s.

## 2.24.0 — 2026-08-28

### Adaptive RDMA receive capacity and bounded high-water reclaim

- Replaced the eagerly committed monolithic RDMA receive segment with a lazy
  chunk pool. `DFKV_RDMA_RECV_SEGMENT_SIZE` remains the hard process budget;
  `DFKV_RDMA_RECV_CHUNK_BYTES` (default 256 MiB) controls incremental commits.
  Each connection publishes and registers only the chunk containing its lease.
- Decoupled `DFKV_RDMA_MAX_BLOCK_BYTES` from physical connection geometry. It
  remains the logical safety ceiling, while data QPs now negotiate a
  power-of-two class sized to the current operation, with
  `DFKV_RDMA_CONNECTION_MIN_BLOCK_BYTES` (default 256 KiB) as the floor.
- Idle endpoint reuse selects the smallest sufficient class, and a full idle
  pool prefers a smaller returning QP over its largest retained QP.
- Extended adaptive geometry to QP depth: scalar operations open depth-1 QPs,
  while batched operations select the smallest sufficient power-of-two depth.
  Pool reuse and resource accounting now match both block and depth classes,
  with bounded per-class opened/active/idle metrics.
- Replaced per-connection pull-read MR registration with type-2 Memory Windows
  over shared per-PD chunk MRs. Hardware without Memory Window support retains
  the exact-range MR fallback and exposes which path each connection used.
- Affinitized growth chunks to their first rail/NUMA node, and added idle
  shrinking for empty non-initial chunks (`DFKV_RDMA_RECV_CHUNK_IDLE_MS`,
  default 60 s). The initial shared chunk is never released.
- Changed watermark eviction to a persistent high/low hysteresis drain bounded
  by `DFKV_SLAB_EVICT_MAX_EXTENTS_PER_TICK` (default one). Whole extents clear
  `slots.tbl` in one contiguous write instead of one 64-byte pwrite per slot.
- Added receive-pool growth/shrink/budget, pull Memory Window/fallback,
  adaptive connection-class, and watermark active/tick/duration/clear metrics.
- Reused one preallocated zero-record buffer for extent metadata clears,
  removing allocation and repeated zero-fill work from the watermark lock.
- Completed vLLM replicated-MLA rank convergence: converged client-rank mode
  automatically enables native same-host GPU rendezvous, so one rank performs
  each remote GET and CUDA IPC publishes identical bytes to TP followers.
- On a production node, 80 live 1 MiB data QPs under a 64 MiB logical ceiling used
  674 MiB across three lazy chunks and completed 10,000/10,000 PUTs; v2.23.3's
  fixed geometry exhausted an 8 GiB segment at roughly 15 such QPs. Sustained
  64 GiB high-water PUT improved from 5.39 to 6.71 GB/s, p99 from 63.8 to
  24.1 ms, and max latency from 496.8 to 50.8 ms.

## 2.23.3 — 2026-08-28

### vLLM hybrid state and explicit TCP staging

- Accepted align-mode hybrid cache groups whose Mamba block size differs from
  the attention cache block size. The connector already resolves the scheduler
  LCM and preserves each group's physical block geometry; the old equality
  gate incorrectly rejected Qwen3.8-Flash-Next (`400` vs `4`) at startup.
- Added opt-in `require_rdma=false` for the vLLM connector. The default remains
  fail-closed GPUDirect RDMA; the opt-in path uses libdfkv's bounded host
  staging plus final CUDA publication and enables a correctness-first fallback
  when a platform's inbound GPUDirect GET path is unusable.

## 2.23.2 — 2026-08-19

### LMCache rank-local rail affinity

- Added opt-in `rail_affinity` and `rail_affinity_fallbacks` plugin settings to
  the in-process LMCache dfkv connector. Each worker now uses LMCache's explicit
  per-host `local_worker_id` to apply the shared primary/fallback policy before
  opening the native client; missing or invalid local metadata fails closed.
- Kept the MP-server L2 adapter unchanged because it owns one shared dfkv
  client and has no single physical GPU rank at construction.

## 2.23.1 — 2026-08-19

### vLLM rank-local rail affinity

- Added `rail_affinity` and `rail_affinity_fallbacks` to
  `DfkvStoreConnector`. Before opening the native client, each vLLM worker now
  maps its world-group local rank to one primary plus a bounded ordered
  fallback set, then sets `DFKV_RDMA_DEV`, `DFKV_RDMA_PRIMARY_DEV`, and
  `DFKV_RDMA_NUMA=0`.
- Moved the rail-list parsing, fallback bounds, duplicate rejection, and
  environment update contract into `dfkv-common`; SGLang HiCache and vLLM now
  share one implementation while retaining their framework-specific physical
  rank resolution.

## 2.23.0 — 2026-08-18

### RDMA connection capacity

- Unified scalar and scatter-gather data operations on one exclusive-ownership
  endpoint pool. Control operations remain isolated. Ring autoscaling now sizes
  `nodes × 2 pools × max(pool limit, rails) × 1.25`.
- Lowered the per-peer/pool idle default from 16 to 8: seven process-wide read
  workers plus one warm spare avoid reconnect churn, while a cap of four is
  proven to reopen three endpoints under a seven-way concurrent wave.
- With 4 MiB blocks and depth four, one pull data connection leases
  33,587,200 bytes. A server needs that per data connection (client
  processes × the per-pool limit of 8) plus as many smaller control
  connections, and 25% for churn; the measured production envelope stays
  within a 64 GiB setting, so increasing it to 128 GiB is unnecessary.
- On a production node, 64 GiB and 128 GiB segments both registered on one
  rail with zero allocation failures. Readiness took 21 s and 43 s
  respectively, so 128 GiB doubles the pinned MR and adds 22 s startup latency
  without serving the current capacity envelope.
- Added client idle/active pool gauges by operation lane and rail, live
  connection gauges by peer and local rail, and the pool limit. Added server
  receive-segment used/free/largest-range, allocation-failure, pull/legacy
  connection, and data/control leased-byte metrics.
- `rail_affinity=true` now assigns each rank one primary plus one ordered
  neighboring fallback HCA by default. `rail_affinity_fallbacks=0` restores
  strict one-rank/one-rail binding; larger values remain bounded by the
  configured HCA list.

### Metrics HTTP clean close

- Consume the complete HTTP request-head section, bounded to 32 KiB and by the
  existing accept-time absolute deadline, before rendering server or MDS
  metrics. Closing no longer leaves unread Prometheus headers that force Linux
  to send a TCP RST and truncate large remote scrapes.
- Preserve the existing one-request connection model, connection cap,
  slow-drip protection, health/readiness semantics, and send timeout. Successful
  responses now finish with an explicit write-side shutdown and clean EOF.
- Add realistic Prometheus headers, a 256 KiB byte-exact response, clean-EOF,
  split-header, header-drip, and oversized unterminated-header regressions.

## 2.20.0 — 2026-08-14

### CUDA GET pinned publication pool

- Replaced per-operation CUDA host registration for GET publication with a
  process-wide, bounded, reusable portable-pinned bounce pool. Slots are fixed
  size, allocated and pinned/registered once on first use, exclusively leased,
  and returned only after the stream-ordered host-to-device copy synchronizes.
- Added `DFKV_CUDA_PINNED_POOL_BYTES` (default 67108864, 64 MiB; maximum
  4 GiB) and `DFKV_CUDA_PINNED_SLOT_BYTES` (default 4194304, 4 MiB; accepted
  range 4 KiB–64 MiB). Values are positive decimal byte counts; the budget
  rounds down to whole slots with at least one and at most 4096 slots. Invalid
  sizing warns and falls back to both defaults. Allocation is lazy, so
  pinned-memory and memlock consumption grows only to the concurrent high-water
  and remains bounded by that rounded per-process budget.
- When every slot is leased and the budget is exhausted, CUDA GET publication
  blocks until a slot is returned rather than allocating beyond the bound.
  Payloads larger than one slot are published in fixed-size chunks.
- Kept RDMA retry data in operation-owned pageable staging. Only the final
  winning attempt is copied through the pinned pool to caller CUDA memory, so a
  failed rail cannot partially publish into the caller buffer. Direct
  GPUDirect GET would violate that retry fence; GPUDirect PUT remains unchanged.
- CUDA context preservation, multi-device publication, byte-exact mixed
  scatter-gather behavior, synchronous completion, fail-closed CUDA errors,
  parent-process teardown, and fail-closed post-fork child behavior are
  preserved. A slot whose stream synchronization fails is quarantined and kept
  pinned until process exit rather than risking release while the driver may
  still reference it. Deployments must provision `RLIMIT_MEMLOCK` for the
  pool's lazy high-water plus other locked memory.

## v2.19.0 — 2026-08-13

### CUDA GET correctness

- Fixed a `RangeInto`/`RangeIntoMulti` failure in which the host publication
  path could call `memcpy` with a CUDA device destination and terminate the
  process with `SIGSEGV`.
- Added an explicit destination kind to range reads. CUDA GETs now receive RDMA
  data into operation-owned pinned host staging and publish it with asynchronous
  host-to-device copies only after the final healthy rail attempt succeeds.
  Completion remains synchronous: the call returns success only after the CUDA
  stream has completed publication.
- Applied the same publication and retry fence to contiguous ranges, mixed
  host/CUDA scatter-gather ranges, and rail retry, so a failed attempt cannot
  publish partial CUDA data before a later attempt succeeds.
- PUT and memory-registration behavior are unchanged and remain GPUDirect.
  CUDA GET adds temporary pinned host memory, a host-to-device copy, stream
  synchronization, and associated allocation/registration overhead. This is a
  known performance tradeoff pending safe direct-GPU retry fencing; deployments
  should validate pinned-memory and memlock capacity for their concurrent GET
  payload.

## v2.11.1 — 2026-08-10

### Release

- Corrected the tag workflow to keep the generic no-HCA release runner from
  executing hardware-only `RdmaLoopback.*` tests. RDMA remains gated by the
  dedicated compile and probe-gated Soft-RoCE CI jobs and by qualification on
  real RDMA hardware.
- Require every release tag commit to be reachable from `origin/main`.
- The immutable `v2.11.0` tag published no release assets because its generic
  runner attempted those hardware-only tests; v2.11.1 supersedes that failed
  publication attempt without data-path behavior changes.

## v2.11.0 — 2026-08-09

### Observability

- Added one bounded native `KVClient` snapshot covering operation, peer health,
  ring/MDS, RDMA rail/CQ/MR/timeout, TCP pool, and same-host dedup state.
- Mirrored the same native allowlist into SGLang HiCache and vLLM Prometheus
  endpoints without adding work to the transfer request path.
- Added opt-in connector-fleet OTLP push for SGLang HiCache, LMCache, and vLLM,
  with stable connector, host, rank, model, deployment, connector-package, and
  native-library identity.
- Added explicit last-good poll health (`success`, failure count, last-success
  time) so snapshot failures cannot masquerade as healthy zero traffic.
- Added MDS per-operation etcd request/error/latency metrics, MDS readiness,
  consumer counts, group capacity/hit/version/stats aggregates, and server
  startup/readiness/health gauges.
- Added bounded per-rail server and client traffic/error series, receive-segment
  capacity, io_uring activation/fallback, storage engine, RAM-tier, slab
  allocator/rebuild, and TCP pool/backoff diagnostics.
- Expanded the provisioned Grafana connector and cluster dashboards with
  readiness, poll freshness, empty-ring/MDS state, dedup, transport failures,
  receive-segment, rail bandwidth/errors, group capacity/hit ratio, and etcd
  latency views.
- Added Prometheus alert rules for server/MDS readiness, storage health, etcd
  errors, version skew, missing heartbeat stats, RDMA failures/capacity,
  connector poll staleness, empty rings, MDS reachability, TCP backoff, and
  dedup rendezvous timeouts.
- Corrected the default zero-dependency exporter contract: it sends OTLP/HTTP
  JSON to the HTTP receiver (normally port 4318); port 4317 is only for an
  explicitly selected OpenTelemetry SDK gRPC exporter.
- Made connector telemetry lifecycle-safe: process-shared recorders are reference
  counted, final metrics/traces flush on the last close, failed exports preserve
  window maxima and uncommitted native counter deltas, and constructor failures
  release native handles and telemetry acquisitions.
- Made one scrape-time MDS group range serve both readiness and aggregation;
  scrape traffic is separately labeled `op="metrics_range"`, and latency
  histograms now resolve bounded failures through 60 seconds.
- Added explicit missing-scrape/missing-connector alerts, bounded dashboard
  resource variables, and canonical `slot_size` slab labels.

### Operations and release

- Added `dfkvctl stat --all` health classification from stable server gauges,
  explicit handling for legacy metrics, and all-group MDS discovery/stat views.
- Added atomic operational artifacts for membership audit, load regression,
  tenant quota, and node-replacement workflows, plus GPU same-host dedup
  reproduction.
- Added a least-privilege, tag-gated GitHub Release workflow. It reruns tests,
  validates observability configuration, builds portable RDMA/io_uring binaries
  and all three connector wheels, emits SHA-256 checksums, and publishes assets
  once through a draft-to-immutable release transition.
- Made root `VERSION`, C++ binaries, and Python connector package versions a
  CI-enforced release contract.
- Tightened the install/package manifest so generated caches and local build
  artifacts are excluded while supported sources, deployment files, tests, and
  documentation remain available in the release archive.
- Hardened `dfkv_bench` as a regression gate: GET uses one up-front registered
  MR arena, setup fails closed instead of falling back to ad-hoc registration,
  goodput excludes failed operations, diagnostics expose control-plane and RDMA
  availability/errors, and both current and legacy output remain parseable.
- Release and CI now validate the runtime container end to end, preserve and
  verify versioned `libdfkv.so` SONAME symlinks, smoke-import the shipped SGLang
  and vLLM packages under Python 3.12, and verify installed wheel metadata and
  vendored telemetry contents.
- Fixed signed-integer overflow in the etcd Base64 decode accumulator; long
  membership values now remain defined under UBSan and decode byte-exactly.
- Made the RAM flush-backlog gauge include dequeued in-flight writes, closing a
  false-zero readiness window that could race the background reclaimer.
- Pinned release Actions, runtime/CI base images, and the bundled observability
  stack to reviewed immutable digests; Grafana now fails closed unless the
  operator supplies a non-default administrator password.

### Compatibility

- No data-format migration is required.
- Metrics and tracing remain opt-in. Existing deployments that do not enable a
  metrics port or connector telemetry retain their prior data-path behavior.
- Dashboards and alert rules target the v2.11.0 metric contract; missing series
  on older nodes indicate rollout state, not a numeric zero.
