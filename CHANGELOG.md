# Changelog

## Unreleased

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
  startup grew by seconds per disk: with six disks and 13.5 million objects,
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
- On xb01-0064, a 1,000,000-token TP8 GLM-5.3 hot load with 128-key windows
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
- On xb01-0064, 80 live 1 MiB data QPs under a 64 MiB logical ceiling used
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
  33,587,200 bytes. The xb01 14×TP8, per-pool-limit-8 envelope is 896 data plus
  896 smaller control connections per server. Including 25% churn consumes
  37,984,665,600 bytes (35.4 GiB), so xb01's 64 GiB setting remains sufficient;
  increasing it to 128 GiB is unnecessary.
- On xb01-gpu-200b-0064, 64 GiB and 128 GiB segments both registered on
  `ib7s400p6` with zero allocation failures. Readiness took 21 s and 43 s
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
