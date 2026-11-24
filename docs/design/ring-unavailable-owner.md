# Unavailable owners: absent grace, cooldown reroute, recent-REMOVE memory, read-back

Status: design, revision 8 (stage C reviewed). Stage A is
implemented; the rest is design. Each
mechanism sits behind its own switch, off by default; none changes a released
binary. A and C are client-side only. The recent-REMOVE memory (M) adds one
wire op and server state; B depends on M and does not ship without it.

Terms: the *ring* is the client's consistent-hash ring of members; a key's
*owner* is the member the ring names for it; the *write ring* is the ring PUTs
route by (today: the adopted view without draining members); the *full ring*
also holds draining members (MDS-04). A member is *unavailable* when the
client should not write to it for now: it drains (MDS-04), it vanished from
the MDS view a short while ago (A), or this client's PeerHealth has it in
cooldown (C). A key's *successors* are the first distinct members clockwise
from the key on a ring, the owner first.

---

## 1. Problem and measurement

A member that fails without draining costs far more than its 1/N of the keys.
Measured on a test ring of eight server processes (TCP, file engine) with a
workload that reads 64-block prefixes and recomputes and re-PUTs a prefix from
its first miss (16 sampled prefixes per second; full-prefix hit rate / mean
hit length), with prototypes of the mechanisms:

| Case | Today | With the mechanism |
|---|---|---|
| Node killed for 90 s, back with its data | 0 % / 7.4 for ~27 s (until the lease expires and the view drops it), recovering to 99 % ~40 s later | C: 38 % in the first 10 s, 96 % by 20-30 s; 62 instead of 157 MiB recomputed |
| Node killed for 3 s | 32 % in the first 10 s | C: 50 % |
| Node alive, its member record deleted for 40 s | 51 % / 36.1 at worst, ~40 s to recover | A reading the absent member after a miss: 100 % throughout (that read is B's now, §3) |
| Node killed 90 s, back empty | after the return 46 % / 33.1, ~40 s to recover | A + read-back: 100 % on the return |

With one of eight nodes gone, (7/8)^64 ≈ 0.02 % of 64-block prefixes are
intact, so almost every prefix breaks. Until the view drops the node, the
recomputed blocks are PUT to the dead node and lost. A warm return needs
nothing; a cold return, or a member that leaves the view while alive, does.

## 2. Assumptions

- **Values of one key are interchangeable within a namespace.** A key is
  derived from what the value encodes (model identity, layout, token hashes),
  so any stored copy is a valid answer. Today the first PUT of a key wins on
  each node (PUT is write-once per node, measured for HIC-26), and different
  nodes may hold different copies. Non-bitwise-deterministic kernels give
  different bytes for one key; every copy is still acceptable to a reader,
  which is what today's behaviour relies on too.
- **Where the assumption breaks: two layouts in one namespace** (the case
  HIC-26 heals: one key, different lengths). An exact-length read rejects a
  copy of another length wherever it comes from (it is *length mismatch*,
  never a hit). A variable-length read accepts any length that fits; a
  read-back (B) of such a copy returns the other layout's bytes. Rule: B reads
  back only with the read's own length check, and B is refused in a
  namespace whose client has counted a length mismatch
  (`dfkv_hicache_length_mismatch_keys_total`, `dfkv_client_dedup_length_mismatches_total`)
  since start; the misconfiguration needs distinct identities, not read-back.
- **Removal is rare** (a vLLM LOAD that found an object damaged, LMCache's
  remove, HiCache's one-time heal). Mechanisms may make REMOVE costlier.

## 3. Mechanisms

All reuse the drain path of MDS-04: a key whose full-ring owner is
unavailable is written to its owner on the write ring, read first there and,
after a miss, from a draining owner. A read after a miss from an *absent*
owner (A) is a read-back and belongs to B (§3.4), under B's rules: a client
that started while the member was out of the view never saw it and sent its
REMOVEs to the new owner only, so the absent member may hold removed copies
(measured with stage A's first version: 3 removed keys read in 5 s).

### 3.1 A: absent grace (`DFKV_RING_ABSENT_GRACE_MS`, default 0 = off)

A member that disappears from the adopted view is kept as *absent* for the
grace period, then dropped: no writes and no reads; REMOVE and RETAIN reach
it; its unconfirmed REMOVEs are retried when it returns.

- **Its own ring.** An absent member is looked up on the view's members plus
  the absent ones, separately from the drain ring (the view's members,
  drainers included). A key can so have three owners: the write owner, a
  drainer, and an absent member; REMOVE and RETAIN go to each that differs.
  The drainer must confirm a REMOVE, the absent member need not. Putting the
  absent members into the drain ring instead would hide a drainer behind an
  absent member for the keys the absent member owns there: their REMOVEs
  would miss the drainer, and after the grace a read after a miss would find
  the removed value on it.

- **Timer.** Each absent member carries `absent_since_ms`. A one-second tick
  of the member poller's loop (no new thread) drops members whose grace has
  passed and rebuilds the rings. With static members nothing vanishes.
- **Bound.** At most the grace (cap 600 s; larger values clamped with a
  WARNING); at most `max(1, N/8)` members absent at once, a further vanished
  member is dropped at once.
- **Planned leaves are not kept:** a member whose last record was draining
  (DRN1) is dropped at once.
- **Member-view guard (#497).** A applies to views the guard admitted; a
  refused view changes nothing. A member back within its grace is an
  ordinary member again at once (owning the keys it owned before, as
  without A: the ring is deterministic).
- **REMOVE and RETAIN** go to the write-ring owner and to the absent member.
  A REMOVE does not need the absent member's confirmation (it is usually
  down); it is counted and queued (bounded, `DFKV_RING_ABSENT_REMOVE_QUEUE`).
  When the member returns the queue is sent at once: the failures during the
  absence put the member in PeerHealth cooldown (up to 30 s), so the return
  to the view counts as a probe answer and ends the cooldown for that first
  send, making the retry the trial. What is still unconfirmed is sent again
  at each tick, with PeerHealth as it is, for up to one grace after the
  return, then dropped and counted. A PUT that reaches the returned member
  before a later retry can be deleted by it: a miss, never a stale value. An absent
  member's RETAIN answer does not stand for the key (no read finds that copy
  until the member returns).
- **No read after a miss.** See §3 above: reading an absent member is B's.
- **Warm return with unconfirmed REMOVEs.** A member that returns serves
  copies whose REMOVE it did not confirm: this client's (racing the retry, or
  dropped by the retry queue's bound) and other clients' (which never sent
  it there). This is **no worse than today**: today a member that left the
  view receives no REMOVE at all and serves the same copies on its return.

### 3.2 C: cooldown reroute (`DFKV_RING_COOLDOWN_REROUTE`, default 0 = off)

A key whose owner this client's PeerHealth has in cooldown is routed, for
PUT, GET, EXIST and RETAIN alike, to a *substitute*: a successor that is not
in cooldown, chosen by the rule below. A miss at the substitute is a miss
(the cooling owner is not asked; reading it back is B's).

- **Episode end.** The cooldown check is non-mutating (a cooldown that has not
  expired; no trial taken). A cooling owner receives no traffic from this
  client, so its cooldown ends only by expiry; from then routing returns to
  the owner, and the first operation after expiry takes the PeerHealth trial
  as today. A failed trial starts the next (longer) cooldown and a new
  episode. While the registry gate is closed (below), C routes as today:
  closing it mid-episode ends the substitution at once.
- **Successor order.** Every member has fixed points on the hash circle, so
  the members clockwise from a key come in one global order, and the
  successors of the key on any ring are that order restricted to the ring's
  members. The *write ring* is the adopted view without drainers and absent
  members; substitutes are picked there.
- **Recent rings and stable members.** Each client keeps the write rings of
  every view it adopted or its member poller saw (admitted by #497's guard or
  not) within the last `S = settle + poll_max = 11 × poll_max`, where
  `poll_max` is the largest poll interval among this client and the marked
  clients of the last registry answer (the marker carries it, below), and
  `settle = 10 × poll_max` as in §3.4 (B uses the same `poll_max`). Readers
  and removers use the same S. The current write ring is always among them.
  A member is *stable* for this client when it is in every one of its
  recent rings.
- **When S changes.** A registry answer can change `poll_max`. When S grows
  (a slower client registers), the recent-ring history may not yet span the
  new S: C and B stay off (the gate counts as closed, `reason="s_growing"`)
  until the new S has elapsed since the change, so that the history covers
  it. When S shrinks (the slowest client leaves), the client keeps using the
  old S for one old S before it switches. Both rules apply to B's `settle`
  too.
- **Reader: the substitute.** For key K whose owner O cools in this client:
  walk the union of this client's recent rings in K's successor order. The
  first `DFKV_RING_REMOVE_FANOUT` members of that walk are the candidates;
  members that are unusable (cooling, or not stable) still count toward the
  rank. The substitute is the first candidate that is stable and not
  cooling; with none, there is no substitute and the key behaves as today
  (the cooling owner is skipped: a miss, a refused write).
- **Remover: the fan-out targets.** Whenever C or B is configured, a REMOVE
  goes, for each of this client's recent rings, to every member of that ring
  in K's successor order up to and including the `DFKV_RING_REMOVE_FANOUT`-th
  member stable for this client; plus the drainer (MDS-04) and the absent
  owner (§3.1); each target once. With no view change in S that is one
  ring, and the targets are its first `DFKV_RING_REMOVE_FANOUT` members (all
  of them on a smaller ring).
- **Why the targets hold the substitute.** Let reader Y use substitute T for
  K, and remover Z REMOVE K. T is in every ring of Y's recent set and is
  among the first `DFKV_RING_REMOVE_FANOUT` members of their union. If Y's
  and Z's recent sets share a ring V, then T is in V, and every member before
  T in V is in Y's union before T: fewer than `DFKV_RING_REMOVE_FANOUT`
  members, stable for Z or not. Z's walk of V goes up to its
  `DFKV_RING_REMOVE_FANOUT`-th Z-stable member, which therefore comes at or
  after T: T is a target. The two sets share a ring when the view the MDS
  served at Z's REMOVE was seen by Y's poller within S of Y's read and by Z's
  within S of the REMOVE: a poller sees every view served for at least one
  poll interval, and S covers the slowest marked poller. One hole remains,
  stated as a limit: **a view served for less than one poll interval**, the
  only view naming T for one of the two (as in §3.4's residual).
- **Substitute copies outlive the episode; the episode end removes them.**
  The argument above covers a read near the REMOVE. A copy on a substitute
  stays after the episode, and a later episode may substitute the same
  member again under another view: T ranks third under V0 and is
  substituted; V1 adds N before T; much later a REMOVE under V1 walks to
  its third stable member and misses T; V2 drops N; a new episode
  substitutes T and serves K. So each client records the keys it wrote (PUT
  or RETAIN SET) to a substitute, per (owner, episode), at most
  `DFKV_RING_REROUTE_KEYS` in total (default 65536). At the cap, further
  rerouted PUTs are refused (a miss for the caller, counted); nothing is
  written to a substitute untracked. When the episode ends (the owner's
  cooldown expires, or the gate closes), the client REMOVEs those keys from
  their substitutes; a failure goes to stage A's per-member queue (sent from
  the tick, cleanup only). A substitute copy therefore lives at most one
  episode plus its cleanup, and no later episode finds it. Cost: one REMOVE
  per key written to a substitute. A clean close (`dfkv_close`, which a
  restart goes through) runs the same cleanup for every open episode first,
  bounded by `DFKV_RING_REROUTE_CLOSE_MS` (default 2000); what it could not
  send is counted (`dfkv_client_reroute_cleanup_total{result="abandoned"}`)
  and logged. Residual: a client killed mid-episode (or whose close ran out
  of time) leaves its substitute copies until the substitute evicts them (or
  their RETAIN TTL ends); a later episode of another client may serve them
  under the view-change sequence above.
- **REMOVE: who must confirm.** Whenever C or B is configured (fan-out on),
  a REMOVE is ok only if every target confirms in this call: the owner, the
  drainer and every fan-out target (counted as above). A target that cools in
  this client, or that fails, makes the REMOVE not ok; the owner included
  (today a REMOVE to a cooling owner already fails at once, without a
  dial). A cooling target's part is still queued (stage A's queue, by
  member) and sent from the poll thread's tick once its cooldown has
  expired, for cleanup only: it is never grounds for ok. The one exception
  is the absent owner (§3.1): its part goes to its queue entry and does not
  decide the result, since no reader asks an absent member until B, and B
  asks M first. Counted as `dfkv_client_remove_unconfirmed_total{reason}`
  (`cooling`, `failed`). Cost: up to `DFKV_RING_REMOVE_FANOUT` + 2 round
  trips per REMOVE, in parallel; more while views change.
- **The cost of that rule during an outage.** While one of N members cools,
  a REMOVE fails for the keys that have it among their targets: about
  (`DFKV_RING_REMOVE_FANOUT` + 1)/N of the keys, against 1/N today (only its
  own keys), for the cooldown's length. The caller sees the failure and may
  retry. A smaller `DFKV_RING_REMOVE_FANOUT` is the lever (it also narrows
  the substitutes: fewer cooling members can be routed around); answering
  ok on a queued part is not. A member that left the view stays a REMOVE
  target while it is in a recent ring, for S after it left: if it left
  because it died, REMOVEs of the keys in its neighbourhood are not ok for
  S after each removal (`reason="failed"`, or `"cooling"` once it cools).
  That is required: a reader whose recent rings still hold it may use it as
  substitute. DEPLOY states this when C ships.
- **The registry gate: every registered remover must fan out.** A REMOVE by
  a client without the fan-out (an older release, or fan-out off) reaches the
  owner only, and a C-enabled reader could keep serving its substitute's copy
  for up to one cooldown. C (and B, §3.4) is therefore enabled only while the
  MDS client registry shows every client registered in the group as one that
  fans out.
  - The registry holds a free-form info string per client (today the
    caller's `client_info`); the native library of this release appends a
    capability marker (`dfkv-remove-fanout=1;poll_ms=<n>`, its member-poll
    interval) whenever fan-out is on. A registered client without the
    marker (an older release, or fan-out off) counts as not fanning out, and
    turns C and B off at the next poll (logged once, counted); so does a
    marked client without the `poll_ms` field. The largest `poll_ms` sets S
    (above).
  - The gate reads the registry (`kListClients`) with the member poller, at
    every poll. A registry answer older than two poll intervals fails the
    gate (C and B off until a fresh one passes); so does a client that
    cannot see the registry (no MDS discovery, or registration off).
  - **Cost and staleness.** The MDS answers `kListClients` from the
    per-group list cache that serves `kListMembers` (`DFKV_MDS_LIST_CACHE_MS`),
    invalidated by a client register or deregister on that MDS; the wire is
    unchanged. The cache TTL is capped below the member-poll interval: the
    MDS clamps `DFKV_MDS_LIST_CACHE_MS` to at most 1000 ms (with a WARNING),
    and C and B stay off (with a WARNING) on a client polling faster than
    every 1000 ms. With several MDS
    instances an invalidation is local to the one that handled it, so another
    may serve a list up to the TTL old; the unsafe direction is a new
    unmarked client hidden from the gate, which extends the window below by
    up to the TTL.
  - No registration wait is added to open. A client with C or B configured
    fans its REMOVEs out from its first one, so it is safe before its
    registration is visible; a client with all switches off opens exactly
    as today.
  - **Limits, stated plainly:** clients that do not register at all cannot be
    seen and are not covered; C and B are safe only in groups where every
    remover registers, and the switch documentation says so. A client of
    this release with C and B off (no fan-out) that REMOVEs right after it
    opens is not seen by the gate until its registration is visible: within
    about one registration attempt plus one registry poll plus the MDS list
    cache's TTL (at most 1000 ms). In that interval
    such a REMOVE reaches the owner only, as a REMOVE by any client does
    today.
  - REMOVE fan-out itself is on whenever C or B is configured, even while the
    gate keeps them off, so a mixed group converges. There is no switch for
    fan-out alone: the marker means "C or B is configured here".
  - **Rollout.** The gate needs every registered client marked, read-only
    clients included (it cannot tell who removes): C or B must be configured
    on every registered client of the group. A registered client of an older
    release, a read-only connector among them, keeps the gate closed; it is
    visible as `dfkv_client_registry_unmarked_clients`.
- **Scatter.** With a warm return the substitute's copies are superfluous.
  With an empty return (a flap with a wipe, a replacement) the keys written
  during the episode are on the substitute only: B reads them back (§3.4).
- **The substitute's copies afterwards** are ordinary objects there. Nothing
  routes to them after the episode (and B's window): they occupy capacity
  until the substitute's eviction removes them (least recently used first),
  or until their RETAIN TTL ends if a RETAIN SET protected them. Bound: what
  the clients wrote for the cooling member's keys during one cooldown (at
  most `DFKV_PEER_COOLDOWN_MAX_MS`, 30 s by default); counted by
  `dfkv_client_cooldown_reroutes_total{op="put"}`.

### 3.3 M: recent-REMOVE memory (server; `DFKV_REMOVE_MEMORY_MS`, default 0 = off)

A node remembers the keys removed on it recently and answers the question
"was this key removed here within the last *age* milliseconds?". B uses it to
refuse a read-back of a removed key, wherever the remover sent its REMOVE.

- **State.** A RAM table `BlockKey` (tenant hash included) → time of the last
  REMOVE of the key on this node (the node's steady clock), recorded for
  every REMOVE the node acknowledges (present key or not) *before* the
  acknowledgement is sent. A PUT does not clear an entry.
- **Bounds.** Entries older than `DFKV_REMOVE_MEMORY_MS` are dropped
  (default when on: 600 s; at least B's lookback H, §3.4). Capacity
  `DFKV_REMOVE_MEMORY_ENTRIES` (default 1 Mi, about 48 B each, so ~48 MiB);
  when full the oldest entry is dropped and the node's *coverage* shrinks.
- **Coverage.** The node keeps `covered_since`: the latest of its start time
  and the time of the newest entry it dropped for capacity. It knows every
  REMOVE since `covered_since`. A question about an age reaching before that
  is answered *unknown*.
- **Crash and restart.** The table is RAM only. After any start, coverage
  begins at the start: questions about earlier times are *unknown* (a warm
  restart keeps the data, not the memory). Nothing is written to disk.
- **Wire: one new op, `kRemovedWithin = 25`.** TCP and RDMA. Request: the
  usual prefix (tenant hash in it), payload `u32 count | u32 age_ms |
  count × {u64 digest_hi, u64 digest_lo}`. Reply `kOk` with `u32 covered_ms`
  (how far back coverage reaches, saturated) and one byte per key: 1 = removed
  within `age_ms`, 0 = not removed, 2 = unknown (age beyond coverage).
  Batched like kRetain.
- **Interop.** A server without M (older release, or the switch off) answers
  an unknown op with `kInvalid`, as every release does (kDrain relies on the
  same). On op 25 the client reads `kInvalid` as *unknown* for every key: not
  an I/O error, no PeerHealth charge, the connection kept. It then remembers
  "no M" for that node for 60 s, so an older server's invalid-op counter is
  not bumped on every candidate miss. Tested against an older-release server
  over TCP and RDMA. Old clients never send the op. No existing op, status or
  format changes; the response status set is unchanged.
- **Metrics:** `dfkv_remove_memory_entries`, `dfkv_remove_memory_covered_seconds`,
  `dfkv_remove_memory_dropped_total` (capacity), `dfkv_removed_within_queries_total{result}`.

### 3.4 B: previous-owner read-back (`DFKV_RING_READBACK_MS`, default 0 = off; requires M)

For a window after a key's owner changed, a GET or EXIST that misses at the
current owner may ask the previous owner once.

- **Absent owners.** With A on, a miss whose full-ring owner is absent asks
  the absent member under the same rules (M asked, gate passing), as a
  previous owner.
- **Owner changes.** Every write-ring change this client adopts (a member
  added or removed, a drain or an absence starting or ending) keeps the write
  ring before it for the window; a C episode is a per-member change (keys of
  member X have X's substitute as previous owner for the window after the
  episode ends). Only the most recent previous owner is asked.
- **Gate.** B is enabled only while the registry gate of §3.2 passes: every
  registered client fans out its REMOVEs. Without it a remover ahead of this
  reader by one added member N that it does not fan out to sends its REMOVE
  to N only, which this reader does not know and does not ask.
- **Fixed lookback H.** Another client can be on a newer ring before this
  one: #497's guard adopts a shrink only when it repeats on two polls and a
  steady decline only once it stops, and a freshly started client adopts its
  first view at once. B therefore does not try to track when views diverged.
  Every M question asks about the same age,
  `H = DFKV_RING_READBACK_MS + settle`, with `settle = 10 × poll_max`
  (30 s at the default 3 s poll; `poll_max` as in §3.2, one value for B and
  C, and when it changes §3.2's rule for S applies). Why ten polls: #497 adopts a held shrink
  after two polls, a decline of one member per poll is adopted when it
  stops, and a client that starts or reconnects during the change takes up
  to a poll to see it; ten polls covers a decline of several members plus
  those lags with room for clock reading and A's tick.
- **Held changes get no window.** H covers a REMOVE only if this reader's
  guard held the change briefly. The member poller records
  `F`, the first time since its last adoption that it saw an MDS view
  different from the adopted ring (admitted or not). A flap does not reset
  `F`; only an adoption does. On adopting a change, if `adoption − F >
  settle − poll_max` (nine polls), the change gets no read-back window:
  B is off for it, its misses stand, and it is counted in
  `dfkv_client_readback_changes_refused_total{reason="held"}`. Otherwise:
  any other client's newer ring was served by the MDS at most a poll before
  `F`, so a remover acting on it did so no earlier than `F − poll_max ≥
  adoption − settle`, and a read inside the window, at most
  `DFKV_RING_READBACK_MS` after adoption, asks about an age that covers it.
- **The read-back rule.** Key K, missed (*not found*) at current owner O,
  previous owner P:
  1. Ask M (`kRemovedWithin`, age H) at K's first `DFKV_RING_REMOVE_FANOUT`
     successors on the union of the current and the previous full ring.
     With the gate passing, every remover fans out to the first successors
     of its own ring, so for a ring that differs from this client's by fewer
     than that many member changes the asked successors include a node that
     received the REMOVE.
  2. Read back from P only if every asked node answered *not removed*. Any
     *removed* or *unknown* (no M, coverage shorter than H, node
     unreachable) and there is no read-back: the miss stands. B fails
     closed.
  3. The read-back value must pass the read's own length check (§2); a
     namespace with a counted length mismatch does not read back.
  4. A hit at the owner always wins; read-back happens only on *not found*,
     never on *length mismatch*.
- **M must cover H.** The nodes need `DFKV_REMOVE_MEMORY_MS ≥ H`. A node
  whose coverage is shorter (a short setting, a capacity drop, or the first
  H after its start) answers *unknown*, so B fails closed there and reads
  nothing back. At configuration time the client clamps
  `DFKV_RING_READBACK_MS` to `600000 − settle` (so that H fits M's
  default of 600 s) with a WARNING, against the current `poll_max`; if that
  leaves nothing (a `poll_max` of 60 s or more), B is off with a WARNING.
  The server documentation states the rule: M at least the clients'
  read-back window plus ten of the group's longest poll intervals.
- **What remains.** A REMOVE older than H, or one that went to a node outside
  the asked successors (a remover whose ring differs from the reader's by
  `DFKV_RING_REMOVE_FANOUT` or more member changes). A change given a
  window rules out a REMOVE older than H by a remover acting on the views
  this reader saw, but not one made on a view the reader never saw: a
  REMOVE made on a shrink view that was served for less than one reader
  poll interval, more than `settle` before the reader's change (P vanished
  briefly; the remover's fan-out on the ring without P skips P, and the
  REMOVE is older than H when the reader asks). In these cases the reader
  may read back a removed key from P; that is no worse than plain routing
  today, where a reader still on the old ring reads P directly.
  **Mitigation:** with A on in the group, a briefly vanished member stays in
  the removers' full rings as absent, so their REMOVEs reach it as well. Run
  B together with A; that closes this residual in practice. B
  refuses read-back while more than one change is held in the window (a
  second change inside the window clears the previous ring instead of
  chaining it).
- **No write-back** of a read-back value (a later option, measured
  separately). **Cost:** for each read-back candidate one batched M query to
  up to 3 nodes and one read at P; nothing for a hit at the owner.

## 4. Retention, tenants, debounce

- **Tenants.** M and read-back use the same `BlockKey`, tenant hash included;
  quota and accounting are those of the node read, as for any read.
- **Retention.** RETAIN SET and REFRESH go to both owners while a key has a
  previous owner (the drain rule), so a retained key keeps its protection
  where it can be read back. Retention in dfkv is a minimum lifetime against
  eviction, not an expiry: no expired data can be served; an object the
  previous owner evicted is a miss there.
- **Debounce.** A read-back hit does not seed the retention debounce for the
  current owner (#496 records the node that answered).

## 5. Interop and switches

- **Wire and disk.** A and C: unchanged. M: one new op (§3.3); servers
  without it answer `kInvalid`, which disables B (fail closed). The MDS
  protocol is unchanged (C reads the existing client registry; the marker
  rides its existing info string). The MDS serves `kListClients` from its
  list cache, whose TTL is capped at 1000 ms (§3.2).
- **Older clients** route by the plain ring and see no difference. With
  C-enabled clients present they read a cooling owner's rerouted keys as
  misses, as today. Their REMOVEs do not fan out, which is why C is gated on
  the registry (§3.2) and why B asks M instead of trusting removers (§3.4).
- **Switches compose independently, with these dependencies:** B requires M
  on the nodes (checked per query, fail closed) and does nothing without it;
  REMOVE fan-out is on whenever C or B is configured; C and B enable
  themselves only while the registry gate passes (§3.2).

| Switch | Side | Default | Range |
|---|---|---|---|
| `DFKV_RING_ABSENT_GRACE_MS` | client | 0 (off) | 0-600000 |
| `DFKV_RING_COOLDOWN_REROUTE` | client | 0 (off) | 0 or 1 |
| `DFKV_RING_REMOVE_FANOUT` | client | 3 | 1-8 |
| `DFKV_RING_REROUTE_KEYS` | client | 65536 | 1024-16777216; keys written to substitutes, tracked for the episode-end cleanup |
| `DFKV_RING_REROUTE_CLOSE_MS` | client | 2000 | 0-60000; how long a clean close spends on the cleanup |
| `DFKV_REMOVE_MEMORY_MS` | server | 0 (off) | 0-3600000 |
| `DFKV_REMOVE_MEMORY_ENTRIES` | server | 1048576 | 1024-67108864 |
| `DFKV_RING_READBACK_MS` | client | 0 (off) | 0-600000; window plus settle (ten polls) at most the nodes' M |

## 6. Client metrics (mirrored in `dfkv_common.client_metrics`)

| Name | Type | Meaning |
|---|---|---|
| `dfkv_client_ring_absent_members` | gauge | members kept absent now |
| `dfkv_client_ring_absent_dropped_total` | counter | absent members dropped (grace over, over the bound, planned leave) |
| `dfkv_client_ring_absent_removes_unconfirmed_total` | counter | REMOVEs an absent or cooling owner did not confirm; `..._retried_total`, `..._dropped_total` |
| `dfkv_client_cooldown_reroutes_total{op}` | counter | keys routed past a cooling owner (get, put, exist, retain); `op="none"`: the owner cooled and no candidate was usable |
| `dfkv_client_registry_gate_open` | gauge | 1 while the registry gate passes (C and B may act) |
| `dfkv_client_registry_gate_failures_total{reason}` | counter | gate transitions from open to closed, by first failing reason: unmarked_client (or a marker without poll_ms), stale_registry, no_registry, s_growing |
| `dfkv_client_registry_unmarked_clients` | gauge | registered clients without the fan-out marker in the last registry answer (operators find such clients by this, then by the registry) |
| `dfkv_client_remove_fanout_total{result}` | counter | REMOVE fan-out messages: ok, error, queued_cooling (sent for cleanup after the target's cooldown) |
| `dfkv_client_reroute_cleanup_total{result}` | counter | episode-end REMOVEs of keys written to substitutes: ok, queued (failed, sent from the tick), abandoned (not sent before a close's bound) |
| `dfkv_client_reroute_refused_total` | counter | rerouted PUTs refused because `DFKV_RING_REROUTE_KEYS` keys are tracked |
| `dfkv_client_remove_unconfirmed_total{reason}` | counter | REMOVEs answered not ok because a target did not confirm in the call: cooling (in this client), failed |
| `dfkv_client_readback_reads_total{result}` | counter | read-back attempts: hit, miss, refused_removed, refused_unknown, refused_length, refused_gate |
| `dfkv_client_readback_changes_refused_total{reason}` | counter | adopted changes given no read-back window: held (guard held the change longer than `settle − poll_max`) |
| `dfkv_client_readback_window_active` | gauge | 1 while a previous ring or a C episode is held |

## 7. Measurement plan for the code PRs

The harness of §1 (eight server processes, a private etcd, the prefix
workload, per-second full-prefix hit rate and mean hit length), each variant
against the same code with its switches off:

1. kill 90 s, warm return: base vs C vs A+C;
2. flap 3 s, warm return: base vs C;
3. flap 3 s, empty return: base vs C vs C+M+B;
4. member record deleted 40 s, node alive: base vs A vs A+M+B;
5. kill 90 s, empty return: base vs A+M+B.

**Correctness under churn**, in every variant: removers REMOVE a sample of
present keys throughout (and PUT some again); every reader hit is checked
against the log of REMOVEs: a violation is a hit on a key whose last REMOVE
*returned* before the read *started* and which no PUT followed. Expected 0 in
all variants. The removers and readers include:

- a remover restarted inside a read-back window (it has no previous ring);
- a reader whose member poller lags (poll interval raised to 10 s);
- two readers with independent PeerHealth, only one of them cooling the owner;
- a warm return with unconfirmed REMOVEs, and with the retry queue overflowed
  (bound lowered): violations here are counted separately, as today's
  behaviour for a member that left the view (§3.1);
- one remover at an older release, registered: C and B must report
  themselves disabled (registry gate), with 0 violations;
- **an older-release remover ahead of the reader by one added member**: the
  remover adopts a view with member N first and REMOVEs keys N now owns;
  with the gate off this is the case M alone cannot see (expected violations,
  shown once); with the gate on, B is off and 0 violations;
- a remover that adopts a change before the reader (reader poll 10 s,
  remover 1 s), REMOVEs right after: 0 violations;
- a steady decline of one member per poll (the guard adopts it only when it
  stops) while a remover started mid-decline adopts its first view at once
  and REMOVEs: 0 violations;
- a flap seen by the reader as adopted → divergent → adopted → divergent
  (a member leaves, returns, leaves again), with a remover acting during the
  first divergence: 0 violations (the fixed lookback H still covers that
  REMOVE);
- a decline held by the reader's guard longer than `settle` (reader poll
  3 s, one member per poll for 15 polls), with a fresh remover that adopts
  its first view at once and REMOVEs early in the decline: 0 violations, and
  the change is counted as refused (`held`) with no read-back;
- a registry answer older than two poll intervals (MDS made unreachable for
  the reader only): C and B report the gate failed.
- for C, each must fail on revision 7's fan-out rule:
  - view skew, three cases with `DFKV_RING_REMOVE_FANOUT=3` and two members
    cooling in the reader: a member added in the reader's view only (the
    reader substitutes it), a drain in the reader's view only, an absence in
    the reader's view only; a REMOVE by a remover on the other view, then a
    read: 0 violations;
  - a fan-out target that is not cooling and times out (busy, not down): the
    REMOVE is not ok, and the reader does not serve the key;
  - a successor cooling in the remover only, and an owner cooling in the
    remover only (both reachable from the reader): each REMOVE is not ok,
    counted with `reason="cooling"`, and the reader misses the key once the
    cleanup send has landed (each must fail on revision 8's first rule,
    which answered ok on a queued part);
  - two MDS instances: a new unmarked client registered at one, the reader
    polling the other; the time until the gate closes is at most one poll
    plus the list cache TTL plus one registration attempt;
  - small rings (N = 2 and 3, `DFKV_RING_REMOVE_FANOUT` ≥ N): targets are
    distinct, also with a drainer or an absent owner that is also a
    successor;
  - the gate closing mid-episode: routing returns to today's at once;
  - an older read-only client registered: the gate stays closed;
  - the five-step sequence (T third under V0 and substituted; V1 adds N;
    a REMOVE under V1; V2 drops N; a new episode substitutes T): the
    episode-end cleanup removed T's copy, 0 violations;
  - a failed cleanup (the substitute refuses at the episode end): the key is
    queued, sent from the tick, and not served meanwhile by a later episode;
  - the reroute cap (`DFKV_RING_REROUTE_KEYS` lowered): rerouted PUTs past it
    are refused and counted, none is written untracked;
  - a marked client without `poll_ms`: the gate stays closed; clients with
    different poll intervals: S follows the largest;
  - a slow client registering mid-episode (S grows): the gate holds C and B
    off until the recent-ring history covers the new S; the slowest client
    leaving (S shrinks): the old S is kept for one old S;
  - a clean close mid-episode: no tracked key is left on the substitute
    (with a substitute refusing, the abandoned keys are counted);
  - C and B unconfigured: routing and REMOVE are today's, byte for byte
    (owners of 2000 keys compared, REMOVE targets recorded).

Footprint as before: under 2 GiB written, private etcd and ports, everything
removed afterwards.

## 8. Stages

1. This design.
2. A (timer, bound, planned-leave rule, REMOVE rule; no read after a miss)
   with tests and metrics.
3. C with REMOVE fan-out, the registry gate and tests.
4. M on the server (table, coverage, `kRemovedWithin`) with tests, including
   interop with older clients and servers.
5. B on top of M, then the measurements of §7. B does not ship without M.

## 9. Revisions

- Revision 1: first version, after two prototype measurement runs.
- Revision 8 (stage C's design review): C picks a substitute from the union
  of the write rings it adopted or saw within S (settle plus a poll), stable
  members only, ranked with unusable members counted; a remover fans out
  along each of its recent rings up to its `DFKV_RING_REMOVE_FANOUT`-th
  stable member, plus the drainer and the absent owner, each once, with the
  argument and its remaining hole (a view served for less than a poll); whenever C or B is configured a REMOVE is ok
  only if every target (owner, drainer, fan-out set) confirms in the call,
  a cooling or failing one makes it not ok, and a cooling target's queued
  part is cleanup only (the absent owner keeps stage A's treatment), with
  the cost during an outage stated (a departed member stays a target for S);
  substitute copies are removed at the episode end (tracked per owner and
  episode, capped by `DFKV_RING_REROUTE_KEYS`, refused past the cap), which
  replaces the misstated cooldown bound; the marker carries `poll_ms` and S
  follows the largest marked poller (one `poll_max` for B and C; when S
  grows, C and B wait until the history spans it, and a shrink waits one old
  S); a clean close runs the episode-end cleanup, bounded and counted; the MDS serves the client registry from its
  list cache with the TTL capped at 1000 ms and the gate's window states
  it; rollout needs C or B on every registered client (no fan-out-only
  switch); eight more measurement cases.
- Revision 7 (stage A's measurement): A does not read an absent member after
  a miss (a fresh remover's REMOVE does not reach it); that read moves to B
  under its rules. A REMOVE an absent member missed is sent when it returns,
  with its cooldown ended for that first send, then at each tick for up to
  one grace, instead of once. Absent members are looked up on a ring of
  their own, beside the drain ring, so that an absent member never hides a
  drainer (review of stage A).
- Revision 6 (review round 5): H covers a REMOVE only when the reader's guard
  held the change briefly; the poller records the first sighting of a
  different view since its last adoption (not reset on a flap), and a
  change held longer than `settle − mds_poll_ms` gets no read-back window
  (counted); residual restated; a churn case for a long-held decline with
  an early remover; residual stated as a REMOVE on a briefly served shrink
  view, with A as its mitigation (run B together with A).
- Revision 5 (review round 4): the lookback no longer starts at the first
  divergent sighting (resetting it on a flap lost an earlier REMOVE); every M
  question asks a fixed age H = read-back window + ten poll intervals, and
  the nodes' M must cover H, so B fails closed for the first H after a
  server start; residual restated (a REMOVE older than H, or sent outside
  the asked successors, no worse than plain routing); a churn case for a
  flap with a remover in the first divergence; gate failures counted per
  transition, with a gauge of unmarked clients.
- Revision 4 (review round 3): the lookback starts at the reader's first
  sighting of an MDS view different from its adopted ring (admitted or not),
  replacing the fixed skew bound, which #497's guard and fresh clients can
  exceed; B off with a WARNING when the clamp leaves no window; no
  registration wait in open (fan-out makes it unnecessary), with the residual
  for a client with C and B off stated; a churn case for a steady decline
  with a fresh remover.
- Revision 3 (review round 2): M's lookback adds the view-skew bound S, and M
  must cover window plus S; B is gated on the registry like C; the gate fails
  on a stale registry answer; this release registers before its first REMOVE;
  `kInvalid` on op 25 is *unknown* with no PeerHealth charge, cached per node;
  the gate's limit for unregistered clients stated plainly; churn cases for an
  older remover ahead by one member and for view skew.
- Revision 2 (review round 1): B can no longer rely on removers (a remover
  restarted in the window, views adopted at different times, per-client
  cooldowns): it asks the new server-side memory M and fails closed. C alone
  was not safe across clients: REMOVEs fan out to the key's successors from
  every C-enabled client, and C is gated on every registered client fanning
  out. Stages reordered (A, C, M, B); the "A+B+C recommended" advice is
  withdrawn. Added: the interchangeable-values assumption and the two-layout
  case; A's warm return stated as no worse than today; C's episode ending at
  cooldown expiry; the measurement plan's churn cases.
