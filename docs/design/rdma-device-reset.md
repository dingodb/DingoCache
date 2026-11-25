# RDMA client: recovering a rail after a device reset (TRN-29)

Status: design, revision 3 (after review-519 round 2). Nothing here is implemented. Client transport
only; the server's rails are out of scope (§8). Builds on #572 (TRN-17,
registration outside the pool lock) and #574 (TRN-16, a device reference per
rail).

## 1. Problem

A firmware or PCI reset, or a driver reload, disassociates every open context
of the device: verbs on it fail, completions flush, and the hardware behind it
is gone or new. The client opens a device once and shares the context and PD
by device name (`AcquireSharedDevice`): while anything references the entry,
every Open of that name gets the same context back. A transport that called
`RegisterMemory` has one anchor endpoint per rail for its lifetime, and since
#574 every transport also holds a `DeviceRef` per rail. So after a reset the
name stays bound to the dead context, every new endpoint on that rail is
created on it and fails, and the rail is broken until the process restarts.
#574 drops the `DeviceRef` on an Open that fails with ENODEV or EIO; the
anchors still hold the context. The three connectors (SGLang HiCache,
vLLM, LMCache) all call `register_memory`, so the anchored case is the usual
one.

Two things already in place do not help: the port watcher queries the port
through its own context (`PortWatcher`, rdma_topology.cc), which is dead too,
so `State()` returns -1 ("unknown, keep the last state") for good; and the
rail cooldown only delays the next attempt on the same dead context.

## 2. Detection, and what it changes

Detection is always on and only counts: with `DFKV_RDMA_DEVICE_REBUILD` off
(the default) nothing else changes, no binding is retired and no abort is
skipped. Two signals:

1. **IBV_EVENT_DEVICE_FATAL** on the context's async event fd. Authoritative
   where the driver delivers it (mlx5 does on a fatal error and a reset).
   Read by one reader per shared context, in the shared-device layer: the fd
   is set O_NONBLOCK once when the context is opened, and whoever polls
   shared devices (the transports' background tick) drains it through
   `PollSharedDeviceEvents()`, which takes a per-device lock so one context
   is read by one thread at a time. Other events (port active/error, LID
   change) are counted and logged only.
2. **ENODEV or EIO from an endpoint Open** on the rail, as #574 handles.
   Covers drivers without the event. A failing `ibv_query_port` alone is not
   a signal: it also fails transiently.

Counted: `dfkv_rdma_client_async_events_total{dev,event}`,
`dfkv_rdma_client_device_suspect_total{dev,reason}`.

**With the switch on**, a signal retires the device's name binding: the
`g_devs` entry moves from the name map to a retired list, and the device's
generation (a counter in the shared-device layer, per name) is bumped. The
next `AcquireSharedDevice(name)` then opens a fresh context and PD. The
retired entry is closed when its last reference goes (endpoints, anchors,
`DeviceRef`s). Every lookup by context (`FindDeviceLocked` and its callers:
`ReleaseSharedDevice`, `SharedAddPoolMr`, `SharedReleasePoolMr`,
`ReferenceDevice`) searches the retired list too, so releases on the old
context still find their entry instead of doing nothing and leaking.
Idempotent: a second signal for a context already retired does nothing.

Retiring the binding is not proof that the context is dead (§3, releases).

## 3. Rebuild

Each transport compares, on every background tick, each rail's device
generation with the one its anchor was built on. A rail whose device
generation moved is out of that transport's selection from the moment the
tick sees it until that transport has rebuilt it, so no other transport's
recovery half-recovers it: a transport whose tick has not run yet opens no
unanchored connection on the new PD, because its rail is not selectable.
That leaves a race no pre-check closes: a transport can Open on the new
context after the bump and before its own tick sees it. So the invariant is
checked after the Open, at run time: the shared entry records the device
generation it was opened under, and the endpoint reports it. After a
successful Open, Acquire compares it with the rail's anchor generation; on a
mismatch it destroys the endpoint (`kAdmission`, nothing charged), takes the
rail out of this transport's selection, wakes its tick, and returns a
local-rail failure, so the operation is retried on another rail. With the
switch off the generations never move and the check never fires.

One attempt, run by the background thread:

1. Without any lock: take the rail out of selection (`SetDeviceEnabled(rail,
   false)`, `RetireIdleRail(rail)`), drop its `DeviceRef`, set
   `dfkv_rdma_client_device_dead{dev}` to 1.
2. `register_mu_.try_lock()`; if it is held (a `RegisterMemory`, or the
   destructor), the attempt is skipped and the next tick tries again.
3. Open a new anchor on the device name (a fresh context, the old binding is
   retired) and stage every region in `pools_` on it without `mu_`, one
   after another, checking `keepalive_stop_` between regions so that a large
   pool cannot hold the destructor's join for seconds. Any failure, or the
   stop flag, rolls back what was staged, releases `register_mu_`, and the
   rail stays out; the next attempt waits for the backoff.
4. Under `mu_`: commit the staged registrations, swap the rail's anchor and
   `DeviceRef`, record the device generation the anchor was built on.
5. Release `register_mu_`. After it: destroy the old anchor (its MRs go
   through #572's `RetiredPoolMrs`), put the rail back in selection if its
   port is ACTIVE, count `dfkv_rdma_client_device_rebuilds_total{dev}`, clear
   the dead gauge.

`register_mu_` is held for one attempt (an Open plus one registration per
region, the cost of a `RegisterMemory`), never while waiting. Between failed
attempts the thread waits for the backoff (1 s, doubling to 30 s) on the
keepalive condition variable with no lock held, so the stop flag ends the
wait at once.

`pools_` does not change: the regions are the same, only one rail's
registrations are new. There is no new declaration generation and the other
rails are not touched.

**Port watcher.** Its own context for the rail is reopened when the device
generation moves (it is keyed by name like the shared layer); until then its
`State()` returns -1 and the rail's selection follows the rebuild alone.

**Releases on a dead context.** `RequireVerbsRelease` aborts when a dereg or
dealloc fails, because a failed revocation may leave the NIC able to DMA into
reused memory. It stays that way unless the context is proven disassociated:
- DEVICE_FATAL was received for that context; or
- the failing release returned EIO or ENODEV, and a confirming verb on the
  same context (`ibv_query_device`) fails with ENODEV; or
- the confirming verb fails with EIO only, and the device is gone: its name
  is no longer in `ibv_get_device_list()`, or its sysfs ibdev node
  (`/sys/class/infiniband/<dev>`) was removed or replaced (another inode than
  when the context was opened). An EIO alone is not enough: wedged firmware
  answers EIO and can still DMA.

Then the failure is logged and counted
(`dfkv_rdma_client_dead_release_failures_total`) and the object dropped: the
device that could DMA is gone. Any other errno (EBUSY, EINVAL, ...), an EIO
the confirming verb does not confirm, or a confirmed EIO on a device still
listed under the same node, still aborts (mlx5 maps some firmware command
errors to EIO on a live device). On mlx5 a disassociated
context's releases return success, so the exemption is for other drivers.
With the switch off there is no exemption.

## 4. Interaction with #572

- The destructor's order changes: it sets `keepalive_stop_`, notifies and
  joins the background thread first, then takes `register_mu_` (for a
  `RegisterMemory` in flight) and tears down. The rebuild runs on that thread
  and only ever `try_lock`s `register_mu_`, so neither order can deadlock.
- The rebuild stages without `mu_` and commits under it, like
  `RegisterMemory`: Acquire sees a rail either out of selection or anchored on
  the new context with every region registered.
- `RegisterMemory` while a rail is dead (switch on): today `StagePoolMr` on
  the dead anchor fails and the whole declaration is rejected, so one dead
  rail would block new memory everywhere. It skips rails whose device
  generation moved (the `anchors_.size() != devs_.size()` check counts them
  as present); the region is published in `pools_` and that rail's rebuild
  registers it with the rest.

## 5. What callers see

Callers never hold lkeys: they declare memory by address (`RegisterMemory`),
and each endpoint resolves lkeys from its own cache. So nothing a caller
holds goes stale. Operations in flight on the dead rail fail with flushed or
fatal completions and take today's paths: `kRailFailure`, replay-safe
operations retried on another rail, a posted PUT never replayed (the caller
gets the failure). While the rail is out of selection new operations use the
other rails; a client with only that rail fails operations with `kLocalRail`
until its rebuild publishes. After it, new connections on the rail resolve
their lkeys from the new anchor's registrations.

## 6. Switch and observability

`DFKV_RDMA_DEVICE_REBUILD`, default off. Off: detection and counting only
(§2), behaviour as today, a device reset still needs a client restart. On:
retirement, rebuild and the proven-dead release exemption. Client-only and
in-process: no wire, MDS or format change. New metrics: the async event and
suspect counters (§2), `dfkv_rdma_client_device_dead{dev}`,
`dfkv_rdma_client_device_rebuilds_total{dev}`,
`dfkv_rdma_client_dead_release_failures_total`. DEPLOY says what the switch
does and that a restart is needed without it.

## 7. Tests

No reset or link-down on real HCAs, ever: not on the dev ring, not on any
host whose NICs serve something.

Seams (unit, any device): `MarkDeviceDeadForTest(name)` delivers the §2
signal (as an event for the context, or through the existing per-rail
Open-errno seam with ENODEV), and a release seam returns a chosen errno for a
chosen context and controls what the confirming verb returns.

- Switch off: a signal is counted; no binding is retired (`DeviceOpens()`
  unchanged), no rail leaves selection, a failing release still aborts.
- Switch on: the next Open opens a fresh context (`DeviceOpens()` + 1); the
  rebuild registers every region once on it (`PoolMrRegistrations()`); the
  old entry closes once the old anchor and connections are gone
  (`DeviceReferences()` back to its baseline), its releases found through the
  retired list; Exist and GET on the rail succeed after the rebuild.
- Two transports in one process: after the signal, the one whose tick has not
  yet run has the rail out of selection and opens no connection on the new
  PD until it has rebuilt its own anchors.
- `RegisterMemory` during the dead window succeeds; the rebuild registers the
  new region.
- Destruction with seams holding each phase: during an attempt (anchor open,
  staging) and during the backoff; it returns, with no deadlock, and leaves
  no pool MR behind.
- An attempt while `register_mu_` is held (a `RegisterMemory` held by the
  registration seam) is skipped and done on the next tick.
- Releases: DEVICE_FATAL, or a confirming ENODEV, is counted, not an abort; a
  confirming EIO with the device gone from the list (seam) is counted; death
  tests: a live context returning EIO once (the confirming verb succeeds)
  aborts, a confirming EIO with the device still listed aborts, EBUSY aborts.
- The race: two transports on one device, a seam bumps the generation after
  transport B's tick ran and before its Open. B's Open lands on the new
  context, the post-Open check destroys it, the operation is served on
  another rail (or fails as local-rail with one rail), and B rebuilds on its
  next tick.

**Soft-RoCE: not possible.** rxe cannot disassociate a context: `rdma link
delete` blocks until every context of the device closes, so a live reset
can't be exercised on Soft-RoCE (seen in CI: the delete waited for the test
process until it exited). The seam tests are the proof; a real
disassociation needs a driver with disassociate support (mlx5), and no real
HCA is reset for a test.

## 8. Out of scope

- The server's rails (its anchors and receive-pool registrations): a server
  restart recovers them, and a server restart is already routine (warm
  restart keeps the disks).
- GPU memory the connectors declare through `register_memory` (vLLM's
  GPUDirect path) is in `pools_` and is re-registered like host memory; a
  reset of the GPU itself is not covered.
- A device that comes back with another name: the rail is the name.
