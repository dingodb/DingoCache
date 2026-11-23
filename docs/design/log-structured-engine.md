# Log-structured storage engine (`log`): design

Status: design, revision 3 (after review rounds 1 and 2). No code exists yet. Nothing
here changes the behaviour of a released binary until the stages in §14 land,
and the default engine changes only after the acceptance measurements in §13.

This document describes a second disk engine for `dfkv_server`, a
log-structured store, to sit beside the slab engine (`DiskSlabStore`) behind
the existing `StoreEngine` interface. It covers the on-disk format, the read
and write paths, eviction and segment cleaning, crash consistency and
recovery, interaction with the RAM tier and with RETAIN, the rule that picks an
engine per directory, the test plan, the measurement plan, and a staged
implementation plan. §16 lists what changed in each revision.

Terms: a *value* is the payload a PUT stores under a `BlockKey`; a *record* is
a value as laid out on disk; a *segment* is a fixed-size region of the disk
that records are appended to; *sealing* a segment makes it immutable;
*reclaiming* a segment makes its space available again; *cleaning* is the
copying of live records out of a segment before it is reclaimed; a segment's
*incarnation* is the number its current use carries (§4.3).

---

## 1. Why a second engine

The slab engine places each value in a slot of a size class. The 2.35.7 code
review measured what that costs and what it leaves open:

| Slab property | Consequence |
|---|---|
| A value is rounded up to the 1 MiB granularity, then into a geometric class (ratio about 1.33) | 17–49% of the disk is padding for real object sizes (2.74 MiB page in a 4 MiB slot: 31%; 1.02 MiB page in a 2 MiB slot: 49%; a 1-byte marker in a 1 MiB slot) |
| Free slots belong to one class; space moves between classes a whole extent at a time | a whole extent of live values is dropped to give another class room; idle classes hold empty extents |
| Index persistence is a fixed-offset table that is correct only if its last writes reached disk | after an unclean exit the table is reset unless a trust token proves the page cache survived; after a power loss the cache is always empty |
| Payloads carry no checksum | a torn write or a misplaced record is served as data |
| Percentage watermarks (92% / 88%) | 8–12% of each disk is kept empty |

Some of these are being improved inside the slab engine (exact-size classes,
absolute headroom, boot-id-based warm restart). Two of them are structural and
are what this engine is for:

1. **Placement without classes.** Values are appended in arrival order, aligned
   only to 4 KiB. There is no size class to round into, no slot to steal and
   no class-to-class competition for space.
2. **Self-describing, checksummed data.** Every record carries its key, its
   length and a CRC32C of its payload; every sealed segment carries a summary
   of what it holds. Any crash, including a power loss, can be recovered by
   reading the disk alone, with no trust token and no reset, and nothing that
   fails its checks is ever served.

The cost is the classic one of log-structured storage: space is reclaimed a
segment at a time, live values that should survive are copied (write
amplification), and a delete does not free space at once. §7 bounds these.

### Non-goals

- Not a general-purpose KV store: values are immutable once written (PUT is
  keep-first), there are no partial updates, and there is no ordered scan.
- No change on the wire, in the client, in the MDS, or in the RAM tier's
  contract. A 2.35.x client or server cannot tell which engine a node uses.
- No automatic migration of slab data into the log engine. A directory keeps
  the engine that created it (§10).
- No support for rotating disks as a target. The design assumes NVMe-class
  random read latency; it would work on other media but is not tuned or
  measured for them.
- Supported filesystems are those slab's warm restart already requires (XFS,
  ext4; tmpfs for tests), because the open-time barrier of §8.3 relies on
  their `fallocate` waiting for in-flight direct I/O.

---

## 2. Requirements

From the task contract, in priority order (correctness first):

| # | Requirement | Where addressed |
|---|---|---|
| R1 | No crash of any kind (process kill, kill -9, OOM kill, panic, simulated power loss) hands out wrong data | §8 (invariants I1–I5), §11.3–11.5 |
| R2 | Data that was sealed before a crash is recoverable after it | §7.5, §8, §11.4 |
| R3 | Lives side by side with slab behind `StoreEngine`; passes all existing storage-interface tests | §3, §4.7, §11.1 |
| R4 | Interoperable with 2.35.x clients and servers | §3.3 (no wire change) |
| R5 | After acceptance, an empty directory defaults to `log`; a directory holding slab data keeps using slab, never wiped on upgrade | §10 |
| R6 | Compared with slab on the same disks and real object sizes on effective capacity, read and write throughput, eviction quality and restart time | §13 |

"Wrong data" is defined precisely in §8.1. In short: the engine never returns
bytes for a key that are not exactly the bytes some PUT of that key stored;
never returns a value that a REMOVE which had returned took out; and never
returns a value older than one a later acknowledged PUT of the same key stored,
whether that later value survived or not.

---

## 3. Place in the system

### 3.1 The interface it implements

`LogStore` implements `StoreEngine` (`src/cache/store_engine.h`) in full:

| `StoreEngine` member | Log engine behaviour |
|---|---|
| `Cache`, `CacheDirect` | append a record; keep-first; a PUT that starts after a REMOVE returned is stored after it (§6.1) |
| `CacheDirectBatch` | append a batch contiguously into one open segment, one submission (§6.4) |
| `Range`, `RangeInto`, `RangeDirect` | read from the record's payload (§5) |
| `RangeDirectPrep` | return a `ReadLease` over the segment's descriptor and a segment pin (§5.3) |
| `IsCached`, `Lookup` | index lookup; `ValueMetadata::generation` is the value's LSN (§9.3) |
| `Remove` | erase from the index, then durable kill entries before returning (§6.6) |
| `UsedBytes`, `TenantUsedBytes`, `Count`, `Evictions`, `EvictedBytes` | from the index and counters; tenant use is rebuilt at open from record keys |
| `Retain`, `RetainEnabled`, `RetainHeadroomPct`, `RetainMaxTtl` | same codes and budget semantics as slab (§9.2) |
| `MaxValueBytes` | what one record can hold in a segment (§4.7) |
| `Activate`, `Close`, `Healthy`, `StartupError` | two-step open like slab: the open reads, `Activate` is the first write (§8.3); `Close` seals and records a clean shutdown (§8.5) |

The semantics a caller can observe are the ones the interface documents today:
keep-first PUT, REMOVE that fences in-flight writes, `Lookup` that reports only
committed values, `kCacheFull` for "healthy but no room now", `kInvalid` for a
value larger than `MaxValueBytes`, `kIOError` only when the store has failed.

### 3.2 What changes around it

- `DiskCacheGroup` learns the engine names `log` and `auto` and the
  per-directory detection rule (§10).
- `DiskCacheGroup` today reaches into `DiskSlabStore` directly for some
  metrics, the watermark observation and retention statistics (`slabs_`). Those
  become engine-neutral virtuals (an `EngineStats` snapshot), so the group and
  `KvNodeServer::MetricsText` do not need to know which engine a disk runs.
- `ReadLease::Adopt` is private to the engines that may create leases
  (`DiskSlabStore`, `KVStore`); `LogStore` is added to them.
- `ReadLease` gains an optional verification hook (§5.3): the engine can attach
  the expected payload CRC of a whole-value read, and `KvNodeServer`'s finish
  callbacks (`FinishDiskRead`, `FinishPromotedDiskRead`) call it before the
  bytes are installed in the RAM tier or handed to coalesced followers. This is
  the one change in `KvNodeServer`; slab leases carry no hook and behave as
  today.
- Nothing else in `KvNodeServer`, the RAM tier, the RDMA server or the
  transports changes. They already talk only to `StoreEngine` and `ReadLease`.

### 3.3 Compatibility

The engine is local to a node. No frame, status code, RETAIN code or MDS
record changes, so 2.35.x clients and servers interoperate with a node running
it as with any other node. The MDS member statistics (capacity, used bytes,
evictions) are filled from the same engine-neutral numbers.

The log engine answers `kCacheFull` in situations where slab would have evicted
inline (§6.1, §7.1). A 2.35.x client treats `kCacheFull` as "not stored" for
that PUT and does not cool the peer down (its peer-health update counts only
`kIOError` as a failure), so the effect is a missed store, not a lost node. The
rate is an acceptance metric (§13).

A node may run different engines on different disks (§10.3), which is what
lets a fleet move to the new engine one emptied directory at a time.

---

## 4. On-disk format

All integers are little-endian. All checksums are CRC32C (Castagnoli). The
repository's `utils/crc32.h` is IEEE CRC-32 with a software table; the engine
adds a CRC32C with the hardware instruction (SSE4.2 `crc32` or ARMv8 CRC) and a
software fallback, because payload checksums run at disk bandwidth (§12.3).

### 4.1 Directory layout

```
<dir>/
  lse_super          superblock, two copies (offsets 0 and 4096)
  lse_journal        append-only metadata journal (kills, retention)
  segments/
    G000000 … Gnnnnnn  segment files, each exactly segment_bytes, preallocated
  lse_initializing   present only while a first initialization is in progress
```

The `lse_` prefix and the `segments/` name are disjoint from every name the
slab engine uses (`slab_meta`, `slots.tbl`, `slab_state`, `slab_initializing`,
`extents/`) and from the file engine's `blocks/`, which is what the detection
rule in §10 relies on.

The directory is `flock`ed for the life of the process, as slab does: one
process per directory.

### 4.2 Superblock

Two 4 KiB copies at offsets 0 and 4096 of `lse_super`. A write updates the copy
with the lower sequence number with `RWF_DSYNC`; a reader takes the valid copy
with the higher sequence number. A torn write therefore always leaves one good
copy. Every superblock write is synchronous; none is on a per-request path
except the rare reservation bumps below.

| Field | Size | Meaning |
|---|---|---|
| magic | 8 | `DFKVLSE1` |
| format_major / format_minor | 4 + 4 | major changes are incompatible; minor are compatible |
| incompat_features / ro_compat_features / compat_features | 8 + 8 + 8 | bit sets, as in ext4: an unknown incompat bit refuses the open, an unknown ro_compat bit makes the store unhealthy rather than serve, an unknown compat bit is ignored. Bit 63 of incompat is *development format* (§14) |
| store_uuid | 16 | random at creation; stamped into every segment header, record and journal block; never leaves the node |
| sequence | 8 | increments on every superblock write |
| segment_bytes | 8 | fixed at creation (§4.7) |
| segment_count | 4 | number of segment files in use; may change (§7.6) |
| align | 4 | 4096 |
| incarnation_reserved | 8 | every incarnation ever written to a segment header is below this |
| lsn_reserved | 8 | every LSN ever assigned (record `generation`, `write_lsn`, journal entries) is below this |
| clean | 4 | 1 if the last run closed cleanly (§8.5) |
| reset_in_progress | 4 | 1 while a reset of the store to empty is under way (§4.5); an open that finds it set redoes the reset |
| created_by | 32 | version string of the release that created the store |
| reserved | … | zero |
| crc | 4 | CRC32C of the 4 KiB copy with this field zero |

**Reservations.** Incarnations and LSNs are handed out from ranges reserved in
the superblock: before the engine uses a number at or above
`incarnation_reserved` (or `lsn_reserved`), it writes the superblock with the
bound raised by a range (1024 incarnations, 2²⁰ LSNs) and waits for that write
to be durable. The bump is started in the background when half a range is used,
so a PUT normally never waits for it. (Not yet in the implementation: the bump is written
when the range is used up, synchronously under the store mutex, once per 2²⁰
PUTs; so is a segment's fresh header when a PUT opens it, once per segment
filled. Each is one 4 KiB `RWF_DSYNC` write, 16–19 µs at the median and
0.3–0.5 ms at the 99th percentile on the development node's NVMe, during
which the store's other operations wait. Moving both off the PUT path is
left to the performance stages.) At open, the next numbers are the
reserved bounds of the valid superblock (not the highest numbers seen on disk),
so no number is ever handed out twice, whatever headers or records a crash
tore. LSNs start at 1; 0 never names a value.

### 4.3 Segment file

```
offset 0                 4 KiB segment header
offset 4096 …            records, each 4 KiB aligned
…                        summary (sealed segments only), right after the last record
segment_bytes − 4096     4 KiB trailer (sealed segments only)
```

**Segment header** (4 KiB): magic `DFLSSEG1`, store_uuid, segment index,
*incarnation* (unique within the store and never reused, §4.2: every time a
segment file is put to a new use it gets a new one), and a CRC32C. A header is
written with `RWF_DSYNC`, and no record of an incarnation is written before its
header is durable.

**Record** (4 KiB header block, then the payload):

| Field | Size | Meaning |
|---|---|---|
| magic | 4 | `DLR1` |
| type | 1 | `VALUE` (others reserved) |
| flags | 1 | bit 0: payload is inline in this block; bit 1: record is a relocated copy (§7.4) |
| header_len | 2 | bytes of header used (forward compatibility) |
| store_uuid | 16 | must match the superblock |
| incarnation | 8 | must match the segment header |
| key | 24 | `BlockKey` (digest_hi, digest_lo, tenant_hash) |
| value_len | 8 | caller-visible length |
| generation | 8 | LSN at which this value was first committed; kept by relocation |
| write_lsn | 8 | LSN of this write (equals generation unless relocated) |
| payload_crc | 4 | CRC32C of the `value_len` payload bytes |
| header_crc | 4 | CRC32C of the header bytes with this field zero |
| inline payload | ≤ 4096 − header_len | when flag bit 0 is set |
| reserved | … | zero |

A header longer than 88 bytes is an extension: this format's readers
require the extension bytes to be zero, so a writer may use them only
together with an incompat or ro_compat feature bit (§4.2) that tells older
readers what they cannot read. Format minor versions alone stay compatible
because they never add header fields.

A payload is inline exactly when it fits, that is when value_len ≤ 4096 −
header_len; a decoder refuses a record whose flag says otherwise, and a
value_len larger than the segment can hold. A non-inline payload starts at
the next 4 KiB boundary after the header block
and is padded with zeros to a 4 KiB multiple. The next record starts after the
padding. So a record occupies `4096 + round_up(value_len, 4096)` bytes, or
exactly 4096 bytes when the payload fits inline (any value up to about
3.9 KiB, which covers the 1-byte markers some connectors write).

The header is in its own block rather than in front of the payload in the same
block so that the payload begins on a 4 KiB boundary: the PUT path can write
the caller's aligned buffer with `O_DIRECT` and no copy (§6.2), and a read of
the whole value is an aligned read with no head offset (§5).

**Summary** (sealed segments): an array of 64-byte entries, one per record in
the segment (key, value_len, generation, write_lsn, offset in 4 KiB units,
payload_crc, flags, and the record's header_len, so that an inline payload
is found without reading the header). It carries no checksum of its own: its CRC32C is the
trailer's summary CRC, so a summary is trusted only through a valid trailer.
Room for it is reserved as
records are placed (§6.1), so it always fits. **Trailer** (fixed last 4 KiB):
magic `DFLSEND1`, store_uuid, incarnation, record count, summary offset,
summary length, summary CRC, last write_lsn, trailer CRC. A free segment's
fresh header (at initialization, after a reclaim, a reset or a grow) is
followed by a trailer of the same incarnation with no records, so that
recovery loads an empty summary instead of walking the whole segment. The
trailer is
written only after the summary is durable (§6.3). A trailer is valid only for
the incarnation it names, and a segment with a valid trailer is never appended
to within that incarnation.

### 4.4 Space overhead

For a value of length *s*: 4 KiB header block plus padding to 4 KiB, plus its
64-byte summary entry; on average 4 KiB + 2 KiB + 64 B. Per segment, the
segment header, trailer and the unusable tail when the next value does not
fit (§4.7). For the object sizes measured in production:

| Value | Slab slot (2.41) | Log engine footprint | Log overhead |
|---|---:|---:|---:|
| 1 B marker | 1 MiB | 4 KiB | — |
| 1.02 MiB page | 2 MiB | 1.02 MiB + ≤ 8 KiB | ≤ 0.8% |
| 2.74 MiB page | 4 MiB | 2.74 MiB + ≤ 8 KiB | ≤ 0.3% |
| 25.9 MiB state | 34 MiB | 25.9 MiB + ≤ 8 KiB | ≤ 0.03% |
| 51.75 MiB state | 62 MiB | 51.75 MiB + ≤ 8 KiB | ≤ 0.02% |

These are format numbers; §13 measures the real effective capacity, which also
includes reserved segments and the tail of each segment.

### 4.5 Metadata journal

`lse_journal` holds the small amount of state that is not part of any record:

| Entry | Fields | Purpose |
|---|---|---|
| `KILL` | segment index, incarnation, offset, key, generation | a record that must never be recovered again (§6.6, §8.1) |
| `RETAIN` | key, generation, retain_until, retain_ttl | persisted retention state (§9.2) |

**Blocks.** Entries are packed into 4 KiB blocks. Each block carries magic
`DFLSJRN1`, store_uuid, a block sequence number (1, 2, 3, … with no gaps), the
entry count and a CRC32C. Blocks are **write-once**: a block is written to a
new offset, completely, exactly once, and never rewritten; a group commit that
has fewer entries than fit in a block still uses a block of its own. The file is
created and synced (with its directory) at initialization.

**One ordered writer.** All journal writes go through one writer thread per
store. Entries are handed to it in a queue, each with a *journal ticket* (its
position in the queue). The writer takes what has accumulated (the group) and
writes it as the next block, one block per write, with
`O_DIRECT | RWF_DSYNC`; a group larger than a block becomes several blocks
written one after the other. Only when a write is durable does the writer
advance the *durable ticket* and wake the waiters whose tickets it covers, and
only then does it write the next block. So block *N*+1 is never written, let
alone acknowledged, before block *N* is durable, blocks of one group cannot
tear out of order, and a crash can lose only blocks that nobody was told were
durable: a suffix of the file.

**Synchronous writes only.** The writer issues every journal write as a
synchronous `pwritev2(..., RWF_DSYNC)` on its own thread, never through
io_uring or any other asynchronous interface. A synchronous write is finished,
or failed, when the call returns, so no journal write of a process can still be
in flight after that process has died. This matters because the open-time
barrier (§8.3 step 2) covers only segment files: a late asynchronous journal
write from a killed predecessor could otherwise land after `Activate()` has
truncated the journal to its stopping offset, and put a stale block back
beyond the new tail. (Truncation and the journal's `fdatasync` are likewise
synchronous calls.)

**Ordering with the store.** KILL entries are enqueued **while the store mutex
is held**, by the operation that decided them (REMOVE, a PUT killing shadows,
§6.1, §6.6), so journal order follows the order in which the store made its
decisions. Every REMOVE, including one that returns `kNotFound`, takes under
the mutex the ticket of the last entry enqueued so far, and returns only when
the durable ticket has reached it. A REMOVE therefore never returns before any
kill decided before it, by itself or by anyone else, is durable (I2; crash
point C34).

**Reading.** Recovery reads blocks in sequence order from the start of the file
and stops at the first block that is not a valid block of this store with the
next sequence number (the *stopping offset*). Everything before it is the
durable journal. If any valid block of this store *with a higher sequence
number* appears after the stopping offset, a block inside the durable range has
been damaged after it was written (media corruption: the ordered writer cannot
produce that state). The engine then cannot know which kills it lost, and it
**resets the store to empty** (below). An I/O error while reading the journal
leads to the same reset rather than to an unhealthy store, by the same
argument: after a reset nothing is served, so nothing wrong can be.

**Resuming after a crash.** `Activate()` truncates the journal to the stopping
offset (`ftruncate`, `fdatasync`) before the writer appends anything, and the
writer continues there with the next sequence number. So the unacknowledged
tail a crash left (torn or partly written blocks) is discarded, and no block
beyond the new tail can survive to look like damage at the next open: two
ordinary crashes in a row never cause a reset.

**Reset to empty**, after damage or an I/O error, in an order that survives a
crash at any step (crash point C35):

1. Write the superblock with `reset_in_progress = 1` and the incarnation
   reservation raised by at least the number of segments plus a range
   (§4.2); durable.
2. Give every segment a fresh header with a new incarnation, each durable.
3. Create the new, empty journal as `lse_journal.new`, `fdatasync`, rename it
   over `lse_journal`, `fsync` the directory.
4. Write the superblock with `reset_in_progress = 0`; durable.

An open that finds `reset_in_progress = 1` treats the store as empty without
reading the journal and repeats steps 2–4. The reset is logged and counted
(`dfkv_lse_journal_integrity_resets_total`).

**Residual risk.** Damage confined to the *last* durable block or blocks
cannot be told apart from a torn, unacknowledged tail: the reader stops there
and no valid higher block follows. The kills in those blocks would be lost
silently, and the values they removed could come back. NVMe devices report
unreadable data as an uncorrectable read error rather than returning wrong
bytes, and an I/O error resets the store (above), so the case left is silent
corruption of the tail of the journal by the device. It is accepted and
documented; it could be closed later by writing every journal block to two
places through the same ordered writer, at twice the journal's write cost.

**Liveness.** A `KILL` entry is live while the incarnation it names is still
the durable current incarnation of its segment; once that segment's next header
is durable, the entry is garbage. A `RETAIN` entry is live while its
(key, generation) is in the index and the deadline has not passed.

**Compaction.** When the journal holds more than a bound (by default four times
the live entries, at least 64 MiB), the writer thread compacts it: it stops
taking new groups (REMOVE and RETAIN callers keep waiting for their
acknowledgement), writes the live entries to `lse_journal.new` as blocks
numbered from 1, `fdatasync`s it, renames it over `lse_journal`, `fsync`s the
directory, and only then resumes, appending to the new file. A REMOVE that
arrives during compaction is therefore written to the new file after the rename
is durable, never to the old one. Compaction rewrites the live set, which is
bounded by the records on disk (§15 Q1), so its pause is bounded; §13 measures
it as part of REMOVE latency, with an explicit criterion (§13).

### 4.6 Why these choices

- **Header block per record, not a per-segment table only.** A segment that
  was open at a crash has no summary; recovery must be able to walk it and
  check every record on its own. A forward walk needs the length of each record
  before its payload, hence a header in front. 4 KiB is the cost of keeping the
  payload aligned for zero-copy `O_DIRECT`; at the object sizes in use it is
  under 1%.
- **Incarnations instead of zeroing.** Reusing a segment does not erase it.
  Old records stay on disk until overwritten. Every record names its segment's
  incarnation, and incarnations are never reused (§4.2), so after a crash bytes
  left over from an earlier use are rejected by number.
- **Kills in a journal, not tombstone records in segments.** A tombstone must
  outlive every older copy of the key, which ties its lifetime to other
  segments and makes cleaning order matter. A kill names the exact
  (segment, incarnation, offset) it invalidates, so it is needed exactly as long
  as that incarnation exists and can be dropped without looking at anything
  else.
- **No zeroing of segment data at creation.** The reviewer suggested writing
  every segment once at creation so that later `fdatasync`s convert no
  unwritten extents. That is about 20 minutes per 3.5 TiB disk at first start,
  for a cost that §13 can measure instead: correctness does not depend on what
  unwritten space contains (every byte is checked by uuid, incarnation and
  CRC), and the extent conversion is a metadata commit folded into the per-file
  `fdatasync` the engine issues anyway (§6.5). The superblock, the journal and
  each segment's header block are written at creation. If §13 shows the
  conversion cost matters, a background pass that writes zeros into segments
  before their first use can be added without a format change.

### 4.7 Geometry from capacity

`capacity_bytes` (per disk, as the group computes it today) is all the space
the store may use, reserves included. From it:

- `segment_bytes` = the largest power of two that is at most
  `capacity_bytes / 32`, clamped to [4 MiB, 1 GiB]. `DFKV_LSE_SEGMENT_BYTES`
  overrides it (a power of two in the same range). It is fixed at creation and
  recorded in the superblock; a later start with a different setting uses the
  recorded value.
- `segment_count` = `capacity_bytes / segment_bytes`.
- *K* (open client segments) = `segment_count / 32`, clamped to [1, 4];
  `DFKV_LSE_OPEN_SEGMENTS` overrides it.
- Reserves inside the capacity (§7.1): *K* segments for the next openings,
  *C* = 2 for the cleaner only, and the relocation segment.
- Minimum: `segment_count` ≥ *K* + *C* + 1 + 4, that is at least 8 segments;
  with the 4 MiB floor the smallest store is 32 MiB. A smaller capacity is
  refused at startup, as slab refuses a geometry it cannot build.
- `MaxValueBytes` = `segment_bytes` − 4 KiB (segment header) − 4 KiB
  (trailer) − 4 KiB (record header block) − 4 KiB (summary room for one
  record, rounded to a block): `segment_bytes` − 16 KiB.

Examples: a 1 GiB test store has 32 MiB segments, 32 of them, *K* = 1, and
`MaxValueBytes` = 32 MiB − 16 KiB; a 3.5 TiB disk has 1 GiB segments, about
3,500 of them, *K* = 4, and `MaxValueBytes` = 1 GiB − 16 KiB.

`MaxValueBytes` is therefore 16 KiB below slab's (`extent_bytes`, exactly
1 GiB by default) on a full-size disk, and much lower on small test stores.
`DiskCacheGroup::MaxValueBytes` takes the minimum over disks, so a node that
mixes engines gets the log engine's limit on every disk. The largest values in
use are about 52 MiB, so this has no effect at production sizes; a test that
stores values larger than its store's limit sets `DFKV_LSE_SEGMENT_BYTES`.

Tail waste. Larger segments waste less at their tail when large values do not
fit: with 52 MiB values and one open segment, a 256 MiB segment loses about 10%
on average to its tail and a 1 GiB segment about 2.5%; best fit across *K* open
segments (§6.1) brings that down further. Smaller segments make eviction finer.
1 GiB at full size is the starting point; §13 measures the effect.

---

## 5. Read path

### 5.1 Index

An in-memory hash map from `BlockKey` to an entry of about 48 bytes:
segment index (4), offset in 4 KiB units (4), value_len (8), generation (8),
retain_until and retain_ttl (8), flags (referenced, protected, relocating,
inline) and a pin count where needed. With 2.74 MiB values a 3.5 TiB disk
holds about 1.3 million entries, roughly 64 MiB with hash overhead.

A small-value workload can create far more entries per byte of disk than slab
ever allowed (a 1-byte value now costs 4 KiB of disk instead of 1 MiB). The
index therefore has a configured maximum entry count; when it is reached, the
oldest segment is reclaimed as if the disk were full (§7). The bound is
reported as a metric.

Beside the index the engine keeps the **shadow set** (§7.5): for each key, the
disk locations of copies of its values that are no longer indexed but would
still pass recovery's checks. It is indexed by key (for REMOVE and re-PUT) and
by segment (to drop entries when a segment is reused).

### 5.2 `Range`, `RangeInto`, `RangeDirect`

Under the store mutex: look the key up; a miss is `kNotFound`. Otherwise take a
*segment pin* (§5.3) and note the record's location, then release the mutex.
Read the requested slice of the payload (inline values come from the header
block). Release the pin.

A pin keeps a segment from being reused while bytes are being read from it.
It does not keep the key alive: a REMOVE or an eviction that happens during
the read erases the index entry at once, later reads miss, and the read in
progress still sees the old, unchanged bytes because segments are not written
in place within an incarnation. This replaces slab's deferred-removal windows
and the `kCacheFull` "busy" answers that came with them.

**Verification.** A read that brings a whole value into memory checks its
payload CRC before returning it, unless `DFKV_LSE_VERIFY_READS=0`. A mismatch
returns `kNotFound`, counts `dfkv_lse_payload_crc_failures_total`, logs the
location, and removes the index entry; the location becomes a **shadow** of the
key (§7.5), because the record on disk may in fact be intact (the corruption
may have been in memory or in transfer) and must stay accounted for until its
segment is reused. The client sees a miss, never the bytes. Partial reads
cannot be verified against a whole-value CRC; §12.3 discusses per-4 KiB-block
checksums for a later format minor.

### 5.3 Zero-copy and RDMA reads

`RangeDirectPrep` returns a `ReadLease` exactly as slab does: the lease owns a
duplicated descriptor of the segment file and the segment pin, and carries
`aligned_off`, `aligned_len`, `head`, `payload_len` and `value_len`. The
transport (io_uring or a plain read) reads `aligned_len` bytes at
`aligned_off` into the registered buffer or a RAM-tier reservation; destroying
the lease closes the descriptor and drops the pin.

Because payloads start on a 4 KiB boundary, a whole-value read has `head = 0`
and goes straight into a RAM reservation (the RAM tier's direct promotion
path). For a slice, `aligned_off` rounds down and `head` is the offset within
the first block, as today.

**Verification hook.** For a whole-value prepared read the engine attaches a
verifier to the lease: the expected payload CRC and a callback for a mismatch.
`KvNodeServer`'s finish callbacks call it after the read completed and before
anything uses the bytes: before `FinishPromotedDiskRead` commits a RAM
reservation (the value would otherwise be installed in RAM), before
`FinishDiskRead` hands the bytes to the coalescer's followers or to the RAM
tier, and before the reply is posted. The bytes are in host memory at that
point, so the check is a CPU pass and no extra I/O. A mismatch turns the
read into a miss for every consumer, aborts the reservation, and the engine
shadows the location as in §5.2. Slab leases carry no verifier.

Until stage 4 adds the hook, prepared reads (`RangeDirectPrep`, the RDMA
pull path and the RAM tier's direct promotion) are not verified: the
experimental engine is not for real data before then.

---

## 6. Write path

### 6.1 Open segments and space reservation

A store keeps *K* open client segments (§4.7), so that concurrent PUTs write to
different files and the device sees parallel streams. Each open segment has an
append offset and a count of records placed in it.

`CacheImpl` (shared by `Cache` and `CacheDirect`), under the store mutex
(the flight is registered before space is reserved, because step 5 may
wait for the sealer or run an inline reclaim; a PUT that a REMOVE cancels
during that wait writes nothing and is answered `kOk`, so nothing it wrote
can come back after the REMOVE, and a REMOVE kills a flight's location only
once it has one):

1. If the key is committed: return `kOk` (keep-first).
2. If a PUT of the key is in flight: join it (a `PutFlight`, as slab), unless a
   REMOVE returned since that PUT began, in which case this PUT waits for the
   flight to end and then starts its own (the same rule slab uses so that a PUT
   after a REMOVE is never answered by a write the REMOVE cancelled).
3. If `value_len > MaxValueBytes`: `kInvalid`.
4. Choose the open segment with the least free space that still fits the
   record **and** one more 64-byte summary entry: a record of size *r* fits if
   `append_offset + r + round_up((n + 1) × 64, 4096) + 4096 ≤
   segment_bytes`, with *n* the records already placed (best fit reduces tail
   waste). Advance its append offset and count.
5. If no open segment fits: hand the fullest one to the sealer (§6.3, without
   waiting) and replace it with a fresh segment from the free list. Clients may
   take a free segment only while more than *C* (the cleaner's reserve, §7.1)
   remain. If none is available, run an inline reclaim (§7.7); if that finds
   nothing either, return `kCacheFull` (healthy, no room now), the status the
   RAM tier and clients already handle.
6. Assign `write_lsn = generation = next_lsn++`, register the flight, pin the
   segment.

Then, without the mutex: build the header block, compute the payload CRC and
write header and payload with one `pwritev` at the reserved offset
(`O_DIRECT`, see §6.2). Then, under the mutex:

- if a REMOVE of the key arrived meanwhile, do not publish; the record is on
  disk and is covered by the REMOVE's kill (§6.6);
- otherwise, if the key has shadows (older values that a crash could bring
  back, §7.5): enqueue `KILL` entries for them **while still holding the
  mutex** (§4.5), keep them in the shadow set marked *kill pending*, release the
  mutex, wait until the durable ticket covers them, then retake the mutex,
  drop them from the set and check for a REMOVE again (§8.1 I3). A REMOVE that
  runs meanwhile still finds them (kill pending) and in any case waits for
  every kill enqueued before it (§4.5). Shadows from eviction exist only while
  a reclaim is in
  progress (seconds); the others are rare (§7.5 table). So this wait happens
  only when a key is stored again within seconds of being evicted, or after a
  CRC failure or an interrupted reclaim;
- insert the index entry.

Complete the flight and unpin. A write that fails with an I/O error fails the
store closed (unhealthy, `kIOError` from then on), as slab does.

The commit point is the index insert. `Lookup` and every read see only
committed entries, as the interface requires.

### 6.2 Zero-copy PUT

`CacheDirect` receives the transport's 4 KiB-aligned buffer with capacity
slack. The payload is written from that buffer with `O_DIRECT` and no copy;
padding to 4 KiB is zeroed in place when the capacity allows, as slab does
today, and otherwise the last partial block goes through a small bounce
buffer. The header block comes from a per-thread aligned buffer. Buffered
writes (`DFKV_SLAB_WRITE=buffered`, tmpfs in tests) use the same layout.

The payload CRC is computed from the caller's buffer before the write. At
hardware CRC32C rates this is well above one disk's bandwidth per core
(§12.3 measures it).

### 6.3 Sealing

Sealing runs on a background sealer thread, never on a PUT. A segment is handed
to the sealer when a PUT finds it cannot take the next record (§6.1 step 5), and
at `Close`. The sealer waits for the segment's in-flight writes to finish (their
reservations pin it), then:

1. `fdatasync` the segment file: every record is durable.
2. Write the summary after the last record (its room was reserved, §6.1),
   `fdatasync`.
3. Write the trailer with `RWF_DSYNC`.

The segment is then immutable within its incarnation until it is reclaimed. Its
summary is also kept in memory in compact form for the cleaner. The sealer
queue is bounded (2 segments by default); if it is full, step 5 of §6.1 waits
for it, which bounds the segments an unclean exit can leave unsealed (§8.4).

### 6.4 Batches (`CacheDirectBatch`)

The RAM tier's flusher hands batches of values to `CacheDirectBatch`. The log
engine reserves one contiguous range in a single open segment for the whole
batch (or as much of it as fits), so the batch is one `pwritev` or one io_uring
submission of consecutive iovecs, then commits all its entries in one mutex
hold. Per-item semantics (keep-first, flights, `kCacheFull`, `kInvalid`) are
those of `CacheDirect`.

### 6.5 Durability and write ordering

The engine relies only on these properties of the filesystem:

- Durability is per file: a write to a file is durable once that file's own
  `fdatasync` (or the write's own `RWF_DSYNC`) has completed. There is no
  ordering between files. Every step that needs one file's write durable before
  another file's write is issued waits explicitly for the first.
- In direct mode, segments are preallocated with `fallocate` (mode 0), so
  their data area starts as unwritten extents. In buffered mode (tests, tmpfs)
  segment files are created when first opened for writing and not
  preallocated, and the open-time barrier is skipped (it exists only for
  direct I/O, §8.3 step 2): preallocating would take the whole capacity in RAM
  on tmpfs. A completed `O_DIRECT` write into such an extent
  is in the device, but the extent conversion that makes it readable after a
  crash is metadata, durable only with that file's next `fdatasync`. So for
  power-loss purposes a record is durable only when its segment file has been
  synced; a process kill does not lose it (the conversion is in the kernel).
- Headers, the superblock and journal blocks are written with
  `RWF_DSYNC` (FUA where the device has it).
- Creating, growing or shrinking the set of segment files is followed by an
  `fsync` of `segments/` and of the store directory.

Every `DFKV_LSE_SYNC_MS` (default 1000) the open segment files are
`fdatasync`ed. This bounds what a power loss can lose to roughly the last
interval of acknowledged writes; it does not affect correctness (§8).

### 6.6 REMOVE

Under the mutex: erase the index entry (later reads miss at once), and mark any
in-flight PUT of the key as cancelled. Collect every disk location that could
hold a recoverable value of the key: the committed record, a record being
written by a cancelled PUT, the destination of a relocation in progress
(§7.4), and the key's shadows (§7.5), including those whose kill is already
pending. Enqueue one `KILL` entry per location **while still holding the
mutex**, and take the journal ticket of the last entry enqueued so far (by this
REMOVE or by anyone before it). Release the mutex, wait until the durable
ticket reaches that ticket (§4.5: one ordered writer; concurrent REMOVEs share
blocks), then return. A REMOVE that finds nothing to kill (`kNotFound`) still
takes and waits for the ticket.

So a REMOVE that returned is durable against any crash (I2 in §8.1),
including when it returned `kNotFound` because the key had just been evicted or
because another REMOVE of the same key, whose kill was not yet durable, got
there first (crash point C34).

REMOVE waits on the request dispatch thread today
(`KvNodeServer::ProcessRequestForKey`), so its latency, and its effect on the
other requests that thread serves, is a measured item (§13); whether a batch
REMOVE is needed is decided from that measurement (§15).

---

## 7. Eviction and segment cleaning

### 7.1 Free space and reserves

Inside `capacity_bytes` (§4.7) the store holds back:

- the next *K* segments for client openings;
- *C* = 2 segments that only the cleaner may take (clients stop at *C*,
  §6.1 step 5), so the cleaner always has somewhere to relocate protected
  records;
- the relocation segment the cleaner is filling.

The cleaner runs whenever the free count falls below *K* + *C* and keeps it
there, from the first time a client needs a segment beyond those `Activate`
opened. Until then it does not reclaim: a store reopened full would
otherwise evict about one segment's values at every restart (measured on
the development node: 22% of the values of an 8-segment store, 3.6% of a
32-segment one; about 1/(*N* − *K* − *C*) of them), and a restart keeps
everything it recovered until writes need the space. For the first second
after that, a PUT that finds no free segment and nothing to reclaim inline
(§7.7; every sealed segment holds a protected record) waits for the cleaner
instead of getting `kCacheFull`. There is no percentage watermark: at full size the space held back is
about (*K* + *C* + 1) × 1 GiB per disk, roughly 0.2% of a 3.5 TiB disk.

### 7.2 Choosing the victim

Sealed segments form a FIFO in seal order. The cleaner takes the oldest sealed
segment. Segments are therefore reclaimed in the order they were filled, which
approximates "evict the oldest writes first"; §7.3 adds the reuse signal that
makes it approximate least-recently-*used*.

Two conditions move the cleaner past a segment to the next oldest, and the
skipped segment is retried on the next pass:

- **Pins.** The cleaner waits at most `DFKV_LSE_PIN_WAIT_MS` (default 5000)
  for a victim's pins (leases, reads in progress) to clear before reclaiming it
  (§7.5 step 1). A lease held longer, for example by a stalled RDMA reply,
  delays that one segment, not all cleaning.
- **Protection.** A segment whose protected footprint (§9.2) is more than half
  of its size is skipped while an older-or-equal choice exists that is not.

When every sealed segment is skipped, the cleaner takes the oldest one that is
not pinned, and only if all are pinned the oldest one: a long lease on the
oldest segment does not stop the reclaim of mostly protected ones.

### 7.3 What survives a reclaim

For each record in the victim's summary whose index entry still points at this
location (dead records, superseded or removed or already relocated, are
skipped):

| Record | Action |
|---|---|
| protected by RETAIN (deadline in the future) | relocate |
| *referenced*: read at least once since it was written or last relocated | relocate, clear the bit |
| otherwise | evict: drop the index entry, count it, make the location a shadow |

This is second-chance reinsertion at segment granularity (a FIFO with
reinsertion of accessed objects, the structure of CLOCK and of S3-FIFO's main
queue). The *referenced* bit is set by every disk read hit and, once the RAM
tier gains its touch hook, by the RAM tier when it evicts a value it served
(§9.1).

**Write-amplification budget.** Over a sliding window, relocated bytes may be
at most `DFKV_LSE_RELOCATE_PCT` (default 25) percent of the bytes clients
wrote. When the budget is exhausted, referenced but unprotected records are
evicted instead of relocated until the window recovers.

**Protected records** are relocated as long as the cleaner can. The retention
budget (`DFKV_RETAIN_MAX_PCT` of the usable capacity, counted as on-disk
footprint, §9.2) bounds their total, but not their share of any one segment: a
single segment can be entirely protected. So the cleaner does not rely on a
per-segment bound. It skips mostly protected segments (§7.2) and, because the
*C* reserve is its own, it can always relocate what it must. As a last resort,
when no segment can be reclaimed without relocating protected records and the
relocation would not fit, it evicts protected records of the oldest segment
(counted as `dfkv_lse_protected_evictions_total`): RETAIN is a best-effort
minimum lifetime in both engines, and the cache must keep accepting writes.

A relocation read that fails its CRC check evicts the record and makes its
location a shadow, as in §5.2 (§15 Q3).

### 7.4 Relocation

The cleaner reads a live record (header and payload) with `O_DIRECT`, verifies
its CRC, and appends it to the *relocation* open segment, with `generation`
unchanged, a new `write_lsn` and the relocated flag. The relocation segment is
separate from the client open segments so that long-lived values cluster
together and are not mixed with new writes (this reduces how often they are
relocated again). It is sealed like any other segment: when it is full or
at `Close`. Relocated values can therefore stay in an unsealed segment for
a long time when little is relocated; after an unclean exit that segment is
one of the walked ones (§8.4).

Before writing, under the mutex, the cleaner marks the entry *relocating* and
records the destination location. After the write, under the mutex:

- if the entry still names the same generation at the source location, it is
  switched to the destination, and the **source becomes a shadow** of the key;
- otherwise (the key was removed, replaced, or dropped for any reason) the copy
  at the destination is dead, and the **destination becomes a shadow** of the
  key. If a REMOVE caused it, the REMOVE already listed the destination in its
  kill set (§6.6); a shadow that is also killed is harmless.

Whatever erases the entry while the copy is being written (REMOVE, a CRC
failure on a read) makes the destination a shadow at that moment, not after
the write: a location may be killed before it is written, and a PUT of the
key that completes before the copy lands then kills it too (I3).

### 7.5 Reclaim and shadows

1. Wait (bounded, §7.2) until the victim has no pins, before anything is
   relocated or evicted (a victim that cannot be reused keeps its values), and
   again once every live record of the victim has been relocated or evicted
   (reads that started meanwhile). An attempt that times out gives up the
   victim and retries it later.
2. `fdatasync` every segment that received a copy taken from the victim's
   current incarnation, by this attempt or by an earlier one that gave up
   after relocating: every relocated copy is durable. The store keeps that
   list per victim until its next header, not per attempt.
3. Write a new segment header for the victim with a new incarnation, with
   `RWF_DSYNC`.
4. Drop the victim's shadows.
5. Garbage-collect journal kill entries for the old incarnation (lazily, at the
   next compaction).
6. Put the segment on the free list.

**Step 2 before step 3** is what keeps sealed data recoverable (R2): without
it, a power loss after the victim's new header is durable but before the
relocated copies are would lose values that were sealed, and protected (crash
point C20).

**Shadows.** A shadow is a disk location that holds a copy of some value of a
key, is not the key's indexed location, and would still pass recovery's checks
(right uuid, incarnation not yet replaced, valid CRCs). Shadows arise from:

| Source | Shadow location | Released at |
|---|---|---|
| a record evicted from the victim (§7.3) | its location in the victim | the victim's step 3 |
| a relocation that switched the index (§7.4) | the source, in the victim | the victim's step 3 |
| a relocation whose copy turned out dead (§7.4) | the destination | that destination segment's next reuse |
| a whole-value read or relocation read that failed its CRC (§5.2, §7.3) | the record | its segment's next reuse |
| a recovery loser (§8.3 step 5) | the record | its segment's next reuse |

A REMOVE kills the shadows of its key along with the indexed copy (§6.6), and a
PUT that stores a new value of a key kills its shadows before it is
acknowledged (§6.1). Shadows are what make invariant I5 (§8.1) hold, and with
it I2 and I3.

**Steps 3 before 4 and 5** are required for the same reason: until the new
header is durable, the victim's old records are still recoverable, so their
shadows and kills must stay (C11, C12).

### 7.6 Capacity changes

The segment count is in the superblock; changes are ordered so that a crash at
any point leaves a store that opens:

- **Grow**: create the new segment files (preallocated, each with a fresh
  header), `fdatasync` them, `fsync` `segments/` and the directory, then write
  the superblock with the higher `segment_count`. Files beyond `segment_count`
  found at open are leftovers of an interrupted grow: they are ignored, and the
  next grow reuses them, writing a fresh header with a new incarnation into
  each (never keeping a leftover's header).
- **Shrink**: reclaim (evict, with shadows and kills handled as in §7.5) the
  highest-numbered segments, write the superblock with the lower count, then
  remove their files and `fsync` the directories. Files beyond `segment_count`
  are again ignored.

Recovery accepts leftover segment files beyond `segment_count`; it refuses only
when files *within* the count are missing or have the wrong size. Slab today
refuses any capacity change without a wipe.

### 7.7 Inline reclaim

Slab evicts inline when a PUT finds no slot; the log engine depends on the
cleaner, and a cleaner that is behind would otherwise make `kCacheFull`
frequent. So when a PUT finds no client-usable free segment (§6.1 step 5), it
reclaims one itself before giving up: the oldest sealed segment that has no
pins and no protected records, evict-only (no relocation): every live record is
evicted and shadowed, and steps 2, 3, 4 and 6 of §7.5 follow (step 2 syncs
only the copies a cleaner attempt that gave up on the victim's pins left
behind; usually there are none). The victim is claimed under the store mutex (marked *reclaiming*, which
the cleaner skips and which no second inline reclaim can claim), and the header
write is done after releasing the mutex. The PUT pays one synchronous header
write. Only if no such segment
exists does the PUT get `kCacheFull`, after waiting for the cleaner if it was
armed less than a second ago (§7.1). The `kCacheFull` rate and the inline
reclaims are counted and are acceptance metrics (§13).

### 7.8 Prefix chains

The review found that recency-only eviction loses prefix chains head first.
The log engine does not solve that by itself, but it helps two ways: pages of
one request are written together and land in the same or adjacent segments,
so they age and are evicted together; and the summary gives the cleaner a
cheap view of everything in a segment, which is where a chain-aware policy
(milestone M5) can plug in without changing the format.

---

## 8. Crash consistency and recovery

### 8.1 Invariants

After any crash and recovery, for every key:

- **I1 (no wrong bytes).** A value returned for key *K* is byte-for-byte the
  payload some PUT of *K* stored. Enforced by the record header CRC, the
  payload CRC, the store_uuid and incarnation match on every record, and
  incarnations that are never reused (§4.2).
- **I2 (no resurrection after REMOVE).** A value present before a REMOVE that
  returned (`kOk`, or `kNotFound` for a key just evicted) is never returned
  again. Enforced by durable kills of the indexed copy and of every shadow
  (§6.6, I5).
- **I3 (newest wins).** A value older than one a later acknowledged PUT of the
  same key stored is never returned, even if that later value is lost to a
  power failure. Among recoverable copies recovery keeps the highest
  `generation` (§8.3 step 5); and a PUT that stores a key with shadows kills
  them before it is acknowledged (§6.1), so the older value cannot come back if
  the newer one is lost.
- **I4 (sealed data survives).** Every record in a segment whose trailer was
  durable is recovered, unless it was killed, superseded, or evicted before
  the crash (an evicted record may come back; that is allowed, see below). A
  record relocated out of a segment is durable in its new place before the old
  place is given up (§7.5 step 2).

The engine maintains at run time the invariant the others rest on:

- **I5 (no unaccounted copies).** Every record on disk whose write completed,
  that would pass recovery's checks, and that is not killed is either the
  indexed location of its key or a shadow of its key (§7.5). The rule applies
  to every path that stops indexing a location without making it unreadable:
  eviction, relocation (both outcomes), CRC failures, and recovery's losers.
  An older value of a key cannot linger beside a newer one unaccounted: a new
  value is stored only after the old one was removed (killed) or evicted (a
  shadow until its segment is reused), and storing it kills those shadows.

What is allowed after a crash: losing acknowledged values that were not yet
covered by a sync (power loss only; §6.5 bounds it), and the return of values
that had been *evicted* (not removed, not overwritten) shortly before the
crash. An evicted value is a correct value of its key; a cache may hold it
again. A REMOVE or PUT that had not returned when the crash happened may or may
not have taken effect.

The contract's wording, "no stale value after an acknowledged overwrite", is
met by I3: an acknowledged later PUT kills the older value's copies durably
before it is acknowledged.

### 8.2 Crash-point table

"Kill" means a process exit at that point (page cache, kernel state and device
contents survive; I/O already submitted may still complete). "Power" means a
loss of power or a kernel crash at that point (writes not covered by a
completed sync may be lost, torn at 512 B or 4 KiB granularity, or reordered
among themselves; nothing is ordered across files).

| # | Operation | Crash point | Kill: outcome | Power: outcome |
|---|---|---|---|---|
| C1 | first init | after `lse_initializing`, before superblock | next open resumes the init (only marker and partial files present) | same |
| C2 | first init | superblock durable, segment files partly created | resume completes them (the init marker is removed only after every file and the directories are synced) | same |
| C3 | PUT | space reserved, nothing written | bytes there are from another incarnation or unwritten: rejected; the walk resynchronizes past the hole | same |
| C4 | PUT | header written, payload partial | not acknowledged; payload CRC fails: rejected, walk skips by the header's length | same; or header lost: rejected |
| C5 | PUT | written, index not yet updated | not acknowledged; record valid: recovered (allowed) | may be lost or torn: rejected if torn |
| C6 | PUT | acknowledged | recovered (completed `O_DIRECT`; resynchronization keeps it past holes) | recovered if covered by a sync; else lost or rejected, never wrong |
| C7 | REMOVE | index erased, kill not durable | not acknowledged: value may come back (allowed) | same |
| C8 | REMOVE | kill durable, returned | never comes back | never comes back |
| C9 | seal | summary written, trailer not | segment walked as an open segment | same; summary may be torn, ignored |
| C10 | seal | trailer written, summary torn | — (the trailer is written only after the summary is durable) | trailer's summary CRC fails: walk |
| C11 | reclaim | entries dropped, new header not durable | old incarnation still on disk: its records recovered (evicted values back, allowed; relocated ones as same-generation duplicates); removed keys stay out (their shadows were killed, kills not yet collected) | same |
| C12 | reclaim | new header durable, kills not collected | old records rejected by incarnation; stale kills ignored | same |
| C13 | relocation | copy written, index not switched | two copies, same generation: the relocated one is kept (§8.3 step 5) | copy may be torn: rejected, source still there |
| C14 | relocation | REMOVE during or after relocation | kills cover destination and source | same |
| C15 | journal compaction | new file written, rename not durable | old journal used (no append went to it after compaction began, §4.5) | same |
| C16 | superblock update | torn | other copy used | same |
| C17 | RETAIN SET or REFRESH | returned, journal block not yet written (the next block the writer writes) | protection lost (allowed, §9.2) | same; also lost if written but not durable |
| C18 | `Close` | some open segments sealed, clean flag not set | unclean open: walks every segment without a valid trailer | same |
| C19 | any | device returns an I/O error | the store fails closed (unhealthy, `kIOError`), as slab | — |
| C20 | reclaim | victim's new header durable, relocated copy not | impossible: every segment holding a copy taken from the victim, also by an earlier attempt that gave up, is synced first (§7.5 step 2) | same; the copy is durable before the victim's old records are given up |
| C21 | reclaim | victim's new header torn | header invalid at open: segment classed free; at `Activate` it gets a fresh incarnation from the reserved range (never the torn one's), and its old records are rejected for ever; its kills are kept until then | same |
| C22 | any PUT or relocation | process killed with `O_DIRECT` / io_uring writes in flight | the open-time barrier (§8.3 step 2) waits for them before anything is read or reused; tested with real kills on each supported filesystem (§11.3) | — (writes in flight are lost or torn) |
| C23 | journal | group *N*+1 pending while block *N* in flight | block *N*+1 is not written before *N* is durable (§4.5): acknowledged kills survive | a lost suffix holds only unacknowledged groups; a damaged block before a valid one resets the store (§4.5) |
| C24 | journal compaction | REMOVE arrives during compaction | waits; written to the new file after the rename is durable | same |
| C25 | segment reuse | new header not durable, records being written | no record is written before its header is durable (§4.3) | records of the new incarnation without their header: rejected |
| C26 | grow | files created, superblock not updated | leftover files ignored, reused by the next grow | same |
| C27 | shrink | superblock lowered, files not removed | leftover files ignored | same |
| C28 | `Activate` | sealing walked segments, refreshing free headers | next open walks again (idempotent) | same |
| C29 | `Activate` | `clean = 0` not durable | no operation was admitted yet | same |
| C30 | inline reclaim | as C11/C12 | as C11/C12 | as C11/C12 |
| C31 | PUT of a key with shadows | shadow kills not durable | PUT not acknowledged: old value may come back (allowed) | same |
| C32 | CRC failure on read | location shadowed | REMOVE and re-PUT kill it | same |
| C33 | reservation bump | superblock with the higher bound not durable | numbers above the old bound were not used (§4.2) | same |
| C34 | REMOVE while a PUT's shadow kill (or another REMOVE's kill) of the same key is pending | the later REMOVE waits for every kill enqueued before it (§4.5) and so returns only after the pending kill is durable | same |
| C35 | reset to empty | interrupted at any of its steps | `reset_in_progress` is set from step 1 to step 4: the next open treats the store as empty and repeats the reset | same |
| C36 | journal resume | two crashes in a row with torn tails | `Activate` truncates the journal to its stopping offset before appending: no stale block survives beyond the tail, no false reset | same |

### 8.3 Recovery procedure

**Open** (the constructor) reads and checks; it changes nothing on disk except
what the barrier in step 2 is (an `fallocate` over space already allocated,
the kind of repeatable allocation slab's open also does). The exception is a
directory that is empty or holds an unfinished initialization: the
constructor creates (or finishes) the store, as slab does, so when another
disk of the node then fails to open, this one is not left as found.

1. Lock the directory. Re-check, under the lock, that the directory still
   classifies as LOG (§10.1). Read both superblock copies; take the valid newer
   one. Refuse on an unknown incompat feature, the development-format bit
   without the experimental flag (§14), a store_uuid mismatch in any file, or
   missing or wrong-sized segment files within `segment_count`.
2. **Barrier**: `fallocate` every segment file over its full length. On XFS
   and ext4 this waits for direct I/O still in flight on the file, including
   io_uring writes that a killed predecessor submitted (the property slab's
   open already relies on, see `DiskSlabStore::LockDirectory`). Nothing is read
   before the barrier has passed for that file. The assumption is tested with
   real kills (C22, §11.3); if it fails on a supported kernel and filesystem,
   the fallback is to quarantine every segment that was open at the crash
   (seal it with what the walk found and never reuse it until a later open
   after a clean close), at the cost of their space for one more run. Buffered
   mode skips the barrier (§6.5).
3. Read the journal (§4.5): build the kill set {(segment, incarnation,
   offset)} and the retention map; detect a damaged block inside the durable
   range (marks the store for reset at `Activate`).
4. For each segment file, in parallel across a small thread pool, read the
   header. An I/O error fails the open. A header with this store's uuid and a
   valid CRC gives the segment's incarnation. A header that is not valid
   (never written, torn) marks the segment free; a valid header with *another*
   store's uuid refuses the open. Then read the trailer. If the trailer is valid
   for this incarnation and its summary CRC matches, load the summary. Otherwise
   walk the segment:
   - Read the 4 KiB block at the walk position. If it is a valid record header
     of this store and incarnation, read and check the payload CRC (inline
     payloads are in the block); keep the record if it passes. Either way the
     header's length is trustworthy (its CRC passed), so move to the next record
     boundary it gives.
   - If the block is not a valid header (a write reserved there never reached
     the disk, or was torn), *resynchronize*: advance one 4 KiB block at a time
     and resume at the next block that is a valid header of this incarnation
     **and** whose payload CRC verifies. Concurrent writers fill an open segment
     at different offsets, so a hole left by one writer must not hide records
     another writer completed and had acknowledged. (A payload could only pass
     as a header by containing this store's uuid, which never leaves the node,
     the current incarnation and a valid header CRC, and then its supposed
     payload would also have to verify.)
   - The walk always runs to the end of the segment's record area.
   - An I/O error during the walk fails the open.
5. Merge: for each record not in the kill set, insert or replace by
   `generation` (I3); between copies of equal generation keep the higher
   `write_lsn` (the relocated copy, so an interrupted reclaim does not undo its
   relocation). Every record that loses is registered as a shadow of its key
   until its segment is reused (I5). Rebuild per-tenant used bytes and the
   protected flags (from the retention map, matched on key and generation);
   referenced bits start cleared.
6. Rebuild the free list (free-classed segments and segments with no live
   records), the seal FIFO (by last write_lsn), and set `next_incarnation` and
   `next_lsn` to the superblock's reserved bounds (§4.2). A segment on the free
   list is not yet reusable as it stands: every opening from the free list
   writes a fresh header with a new incarnation, durable, before its first
   record (§4.3), and the kills of the incarnation it replaces are collected
   only after that.

The store is now opened: counts and stats are available and no data operation
is admitted. **Activation** (`Activate()`) does every write:

7. First, one durable superblock write: `clean = 0`, and the reservation
   bounds raised so this run's numbers start in a fresh range, with the
   incarnation bound raised by at least the number of headers steps 8 and 10
   are about to write (which can be thousands, more than one range).
8. If the open found journal damage or a journal I/O error, or
   `reset_in_progress` set, reset the store to empty (§4.5, steps 1–4 there).
   Then truncate the journal to its stopping offset (§4.5).
9. Seal every walked segment as it is: write a summary of the records the walk
   kept and a trailer, `fdatasync`. A walked segment is never appended to again
   within its incarnation, so whatever lies between or after its recovered
   records (torn writes, holes) can never be reached by a later walk or mixed
   with new records.
10. Give every segment whose header was invalid (free-classed in step 4) a
    fresh header with a new incarnation (durable); then collect the kills of
    the incarnations so replaced. (Segments that are free because they hold no
    live records get their fresh header when they are opened, step 6.)
11. Open *K* fresh client segments from the free list (the relocation
    segment is opened when the cleaner first needs it), start the sealer, the
    cleaner (held until a client needs space, §7.1) and the journal writer,
    and admit operations.

### 8.4 What a walk costs

A walk reads every header block and every payload of the segments it walks,
to check their CRCs. After a clean `Close` nothing is walked (§8.5). After an
unclean exit, the segments without a valid trailer are walked: the *K* open
client segments, the relocation segment, at most the sealer queue's 2 segments
in the middle of sealing, and one segment per opening in progress. That is at
most *K* + 4 segments per disk, (*K* + 4) × 1 GiB of sequential reads at full
size, a few seconds per disk on current NVMe (to be measured, §13). All other
segments contribute their summary only.

### 8.5 Clean close

`Close()` (the server's shutdown and the destructor): stop the cleaner, wait
for in-flight writes, seal every open segment (§6.3), let the journal writer
finish, then write the superblock with `clean = 1`. The next open finds only
sealed segments and loads their summaries.

The clean flag is only an expectation: whatever it says, the open walks every
segment that lacks a valid trailer for its incarnation. `Activate()` makes
`clean = 0` durable before it admits the first operation (§8.3 step 7).

A segment, once it has a trailer, is never appended to again within that
incarnation, after a clean close as after a walk. The price is the unused space
at the end of up to *K* + 1 segments per restart, until those segments age
out.

Unlike slab, nothing depends on the clean flag for correctness: it only saves
the walk.

---

## 9. Interaction with the rest of the server

### 9.1 RAM tier

The RAM tier talks to the disk through the group and `StoreEngine` only, so it
needs no change:

- **Flush**: `CacheDirectBatch`, one contiguous append per batch (§6.4).
  `kCacheFull` from a full store is handled as "no room now" (held in RAM,
  retried; 2.40). `kInvalid` for a value above `MaxValueBytes` is refused at
  admission (2.41).
- **Promotion**: whole-value reads through `RangeDirectPrep` with `head = 0`,
  straight into a RAM reservation, verified before the reservation commits
  (§5.3).
- **`is_persisted` / `disk_generation`**: `Lookup` gives committed state and
  the value's generation (§9.3). The REMOVE fence and the promotion checks
  added in 2.41 work unchanged.

Two hooks are planned for later, optional on the engine side (no-ops on slab
until implemented there): `Touch(key, generation)` so that the RAM tier can set
the disk's *referenced* bit when it evicts a value it served, and an eviction
callback so that the RAM tier drops its written copy when the disk evicts the
key.

### 9.2 RETAIN and TTL

Semantics are those of slab: SET and REFRESH, the same `RetainCode`s, the
`DFKV_RETAIN_MAX_TTL_S` cap and `RetainHeadroomPct`. The class budget
(`DFKV_RETAIN_MAX_CLASS_PCT`) has no meaning without classes and is ignored.

- **Budget**: `DFKV_RETAIN_MAX_PCT` of the usable capacity, counted as each
  protected record's **on-disk footprint** (header block, payload, padding), as
  slab counts slot size and not payload: a protected 1-byte marker costs 4 KiB
  of budget.
- Protection lives in the index entry; deadlines expire through a timing wheel
  with one entry per protected value, cleared in bounded batches (the structure
  slab adopted in 2.39).
- **Persistence**: a SET, or a REFRESH that moves the deadline by at least
  `DFKV_RETAIN_PERSIST_MIN_DELTA_S`, queues a `RETAIN` journal entry
  (key, generation, retain_until, ttl). The call returns without waiting; the
  journal writer puts queued entries in the next block it writes, at once when
  it is idle. So a process kill or a power loss may lose the protection set
  in the last block's time, a few milliseconds (C17). Protection is
  a best-effort minimum lifetime in both engines, and losing it never produces
  wrong data.
- **Cleaning**: a protected record is relocated, and evicted only as the last
  resort of §7.3.

### 9.3 Generation

`ValueMetadata::generation` is the record's `generation` field: the LSN at
which the value was first committed. It is never 0, unique per store, increases
with every commit, never repeats after a crash (§4.2), and is kept when the
cleaner relocates the value. So the RAM tier's "the disk still holds the value
this copy was read from" check (2.41) holds across relocations. It is also
stable across restarts, which has no use today because the RAM tier starts
empty.

### 9.4 Tenants and quotas

The tenant is part of the key and so of every record; `TenantUsedBytes` is
kept in memory and rebuilt at open. Quotas are enforced by the group, as
today.

### 9.5 Metrics

A `dfkv_lse_*` family per disk: segments (free / open / sealed / relocation /
sealer queue), bytes (payload, record overhead, dead in sealed segments),
cleaner (segments reclaimed, inline reclaims, records relocated / evicted,
protected evictions, relocated bytes, budget exhaustion events, victims
skipped for pins), shadows (entries), PUT `kCacheFull` answers, recovery
(segments walked, records recovered, records rejected by reason: header CRC,
payload CRC, incarnation, killed; journal integrity resets), journal (entries,
blocks, compactions and their duration, write latency), payload CRC failures on
reads, index entries and its bound. The engine-neutral numbers
(`dfkv_used_bytes`, evictions, objects) come from the same `EngineStats` the
slab engine fills.

---

## 10. Engine selection per directory

### 10.1 Rule

`DFKV_STORE_ENGINE` / `--store-engine` takes `slab`, `log`, `file` or `auto`.
Each configured directory is classified before any engine opens it, by the
names directly inside it:

| Markers present | Class |
|---|---|
| no entry other than `lost+found` | EMPTY |
| slab markers (`slab_meta`, `slots.tbl`, `slab_state`, `slab_initializing`, `extents`), no log markers | SLAB, whatever else is present |
| log markers (`lse_super`, `lse_journal`, `lse_initializing`, `segments`), no slab markers | LOG, whatever else is present |
| `blocks`, and neither slab nor log markers | FILE |
| slab and log markers both | CONFLICT |
| none of the above | FOREIGN |

"No entry other than `lost+found`" is EMPTY for the classification, but the
slab engine's own open of a new store still refuses a directory that is not
completely empty (`OpenOrInit` requires an empty root), so under `slab`, and
under `auto` before acceptance (when EMPTY maps to slab), such a directory is
refused as it is today; the log engine accepts it.

"Whatever else is present" keeps today's behaviour for slab: a slab directory
with extra entries, such as `lost+found`, opens today (the slab open path does
not enumerate the root of an existing store), and it must keep opening under
`auto`.

| Setting \ class | EMPTY | SLAB | LOG | FILE | CONFLICT / FOREIGN |
|---|---|---|---|---|---|
| `auto` | default engine (§10.2) | slab | log | file | refuse to start |
| `slab` | slab | slab | refuse | refuse | refuse (as slab does today for a non-empty directory without its markers) |
| `log` | log | refuse | log | refuse | refuse |
| `file` | file | refuse | refuse | file | file (the file engine's own checks apply, as today) |

"Refuse" is a startup failure that names the directory and the entries that
decided its class, and it never deletes or rewrites anything. Going from one
engine to another on a directory is an operator action: empty the directory
(or point the server at a new one). This keeps the slab rule "unknown and
previously populated layouts are never auto-recreated".

The classification is advisory: the chosen engine re-checks its own markers
under its own directory lock before it reads anything (§8.3 step 1; slab
already validates its metadata after taking its lock), so a directory changed
between the check and the open is refused, not misread.

### 10.2 Defaults over time

- Until acceptance (§13): the default stays `slab`, so nothing changes for
  anyone who does not ask for `log`.
- After acceptance: the default (no `DFKV_STORE_ENGINE`, no option) becomes
  `auto`. An empty directory then gets the log engine, a directory with slab
  data keeps slab, and nothing is wiped by the upgrade (R5).

### 10.3 Mixed nodes, upgrade and downgrade

- Each disk is classified on its own; a node may run slab on some disks and
  log on others. Per-disk metrics carry the engine name.
- **Upgrade** from a release without the log engine: directories hold slab
  data and stay slab under `auto`.
- **Downgrade** to a release without the log engine, on a directory the log
  engine created: the slab engine of every release from 2.33.3 on refuses a
  non-empty directory that has neither `slab_meta` nor `slab_state` ("cache
  directory contains a different or incomplete store layout",
  `DiskSlabStore::OpenOrInit`), after taking its directory lock and before
  changing anything; the file engine of those releases also fails before it
  touches anything. So an older binary refuses to start on such a directory and
  does not wipe it. The operator either empties the directory or stays on the
  newer release. Tested from stage 3 on (§11.6).
- **Configuration and rollback**: releases before the log engine reject any
  `DFKV_STORE_ENGINE` value other than `file` and `slab`. Deployments should
  leave the variable unset rather than set `auto` (unset becomes `auto` after
  the default switch), so that rolling the binary back still starts on slab
  disks.

---

## 11. Test plan

### 11.1 Interface conformance

The existing storage-interface tests run against both engines:

- `store_engine_test` (round trips, the range conformance table, engine
  selection, startup refusals), `disk_cache_group_test` (including the
  activation tests: a group that is never activated leaves its disks as found,
  and a disk that fails to open keeps the others from being activated, which
  the open/activate split of §8.3 satisfies), the RAM tier wiring tests that
  build real stores, the node-health tests, the RDMA loopback tests' prepared
  read paths, and the client end-to-end tests.
- These are parameterized over the engine (`slab`, `log`; `file` where it
  applies today). The tests' store sizes (1–3 GiB) are within the log engine's
  geometry (§4.7); a test that needs values above its store's `MaxValueBytes`
  sets `DFKV_LSE_SEGMENT_BYTES`. The parameterization is not purely mechanical:
  tests that build slab stores below 32 MiB (for example the 1 MiB stores of
  the RAM tier wiring tests) set a capacity of at least 32 MiB in their log
  variants.
- Tests of slab internals (classes, steals, trust tokens, `slots.tbl` rebuild)
  stay slab-only; each gets a log-engine counterpart where the property applies
  (for example "a REMOVE during an in-flight read never serves torn bytes",
  "keep-first under concurrent PUTs", "a PUT after a REMOVE is stored after
  it").

### 11.2 Format

Unit tests for every encoder and decoder, CRC32C against published test
vectors (hardware and software paths agree), and a fuzz target per decoder
(superblock, segment header, trailer, summary, record header, journal block):
arbitrary input never crashes or hangs the decoder and is either rejected or
decoded to a value whose re-encoding is byte-identical.

### 11.3 Crash points

Every row of the table in §8.2 is a test. The engine has named crash hooks
(in the style of slab's crash slots): a test arms a hook, runs the operation in
a child process that `_exit`s at the hook, reopens the directory in the parent
and checks the outcome column. Power-loss rows run under §11.4. C22 (late
writes) is tested by a child that submits io_uring batch writes (including
ones the kernel punts to its worker threads) and is killed with SIGKILL before
they complete, with a parent that opens while they may still be in flight and
verifies every byte; it runs on XFS and on ext4 on the kernels in use.

**Journal writes are synchronous.** Enforced in code and checked by tests:
the journal writer owns its file descriptor and has no path to the store's
io_uring rings (the `IoBackend` interface used for the journal has no
asynchronous write call); a unit test drives the journal through a recording
`IoBackend` and asserts that every journal write it saw was a synchronous
`RWF_DSYNC` write completed before the next was issued; and the C22 kill test
(§11.3) also kills a child in the middle of a REMOVE burst, so that journal
writes are in progress, and checks after the restart that no journal block
appears beyond the tail `Activate()` truncated to.

### 11.4 Simulated power loss

All engine I/O goes through a small `IoBackend` interface (pwrite/pwritev with
and without `RWF_DSYNC`, pread, fdatasync, fallocate, rename, fsync of a
directory). A recording backend used in tests keeps, per file, the writes
issued since that file's last completed sync. A *crash image* is produced by:

- keeping every write covered by a completed sync of its file (or written with
  `RWF_DSYNC`);
- for the rest, independently per file and per write: dropping it, keeping it,
  or tearing it (keeping a prefix, at 512 B and at 4 KiB granularity), and
  applying the kept ones in a random order, with no ordering across files;
- treating renames and file creations as durable only after the directory
  sync.

The test then opens the image and checks I1–I5 against a model: the workload
generator knows, for every key, which PUTs and REMOVEs were acknowledged and
which syncs had completed. Values are derived deterministically from
(key, generation) so any served byte can be checked without storing a copy.
The suite runs a few thousand seeds per workload (PUT-heavy, REMOVE-heavy,
re-PUT after eviction, cleaning under pressure with RETAIN, CRC-failure
injection, journal compaction under REMOVEs, grow and shrink, concurrent reads
with leases) in CI and many more on the development node.

### 11.5 Real crashes

- **kill -9 soak**: a child `dfkv_server` on real disks under a mixed load
  (PUT, GET, REMOVE, RETAIN, batch flushes from the RAM tier), killed with
  SIGKILL at random intervals, restarted, verified with the deterministic
  values, a thousand cycles per run. Also OOM-kill (memory limit in a cgroup)
  and SIGSEGV via a debug hook.
- **Kernel-level power loss**: on the development node, the store on a
  device-mapper target (`dm-log-writes`, replayed to each flush and FUA mark
  and to points between them, or `dm-flakey` dropping writes), then opened and
  verified as in §11.4. This needs root on the test node; if it is not
  available, the gap is recorded and §11.4 stands alone.

### 11.6 Engine selection

Every cell of the table in §10.1, plus: a slab directory with `lost+found` and
other extra entries stays SLAB; an empty directory gets the log engine under
the post-acceptance default; a slab directory written by a release binary is
detected and kept; a log directory is refused by the slab engine's current
open path (the downgrade case, unchanged since 2.33.3) without any file being
changed, from stage 3 on, plus a recorded check with a release binary; mixed
disks in one node; a directory changed between classification and open is
refused.

### 11.7 Concurrency and endurance

ThreadSanitizer and AddressSanitizer builds of the engine tests; a stress test
with cleaner, inline reclaims, readers holding leases, REMOVEs, relocations and
batches running together for hours; a long fill-and-evict run checking that the
cleaner's reserve is never taken by clients, that a stalled lease delays only
its segment, and that `kCacheFull` appears only when every sealed segment is
pinned or protected.

---

## 12. Performance considerations

### 12.1 Writes

A PUT is one `pwritev` of header and payload into a reserved range: no table
write, no allocator search, no class. Up to *K* open segments write in
parallel. Batched flushes are one submission. The store mutex is held only for
reservation and commit, as in slab. Sealing is off the request path (§6.3).

### 12.2 Reads

A read is one aligned read at a known offset, as in slab. Payload verification
adds a CPU pass (§12.3).

### 12.3 Checksums

CRC32C with the hardware instruction runs at several GB/s per core on current
servers (to be measured on the development node as part of stage 2), against a
few GB/s per NVMe device, so one core per disk is the upper bound of the cost at
full disk bandwidth on both reads and writes. `DFKV_LSE_VERIFY_READS=0` turns
read verification off if the measurement says it costs too much; write-side
CRCs are always computed, recovery depends on them. Per-4 KiB-block checksums
(so that slices can be verified) would cost 4 bytes per block in a later format
minor; not in the first version.

### 12.4 Cleaning

Relocation costs a read and a write of each relocated value, bounded by the
budget in §7.3. Reclaim itself is one sync of the relocation segment and one
synchronous header write.

### 12.5 Restart

Restart runs the barrier (one `fallocate` per segment file), reads the
superblock, the journal, and one summary per sealed segment (64 bytes per
record: about 80 MiB for 1.3 million records), and walks at most *K* + 4
segments only after an unclean exit (§8.4). Disks open in parallel, as they
already do.

---

## 13. Measurement plan for acceptance

**Setup.** Same development node, same NVMe disks and filesystem, same
`dfkv_server` binary; the engine is selected per run on emptied directories.
The slab baseline is the slab engine of the same release, with whatever slab
improvements (exact-size classes, absolute headroom, warm restart) have been
merged by then; its version and settings are recorded with the results. Each
measurement is repeated three times; the report gives the median and the
spread, and a difference smaller than the larger spread is reported as noise.

**Workloads.**

- Object sizes: 1 B (markers), 1.02 MiB, 2.74 MiB, 25.9 MiB and 51.75 MiB (the
  sizes observed for attention pages and linear-attention state in
  production), and a mix drawn from production access logs with their size
  distribution. The traces used are named by their capture date in the
  results.
- Eviction scenarios: the prefix-reuse replay scenario that the 2.34 eviction
  changes were measured on (99.8% prefix reuse under pressure with slab
  today), and replays of the production access logs at 0.5×, 1× and 2×
  overcommit.
- RAM tier off and on, for throughput and for eviction quality.
- **Steady state**: after writing twice the capacity and then 10 minutes of
  continued load with evictions running.
- **First fill and reuse separately**: throughput is measured both while the
  store is first filled (writes into unwritten extents, whose conversion is
  committed with each sync, §4.6) and at steady state (every segment reused),
  and reported per phase.

**Metrics and proposed criteria** (to be confirmed in review before measuring):

| Metric | Measured as | Proposed criterion |
|---|---|---|
| Effective capacity | committed payload bytes / configured capacity at steady state, per size and for the mix; object count | against the slab engine of 2.41 (geometric classes): log ≥ slab + 10 percentage points for every real size except the 1 B marker, where the object count is compared; against a slab baseline with exact-size classes, if merged by then: log ≥ slab |
| Write throughput | `dfkv_bench` PUT over RDMA (direct write path) and TCP; 1, 8 and 32 clients; 1.02, 2.74 and 25.9 MiB; at steady state and during cleaning | GB/s within 10% of slab, p50 within 10%, p99 within 20%, in every cell |
| Read throughput | cold GETs from disk, RDMA prepared reads and TCP, random keys over a full disk; same client counts and sizes; with and without read verification | as for writes, with verification on |
| Eviction quality | object and byte hit ratio, reusable prefix tokens per request, for each scenario and overcommit, RAM tier off and on | reusable tokens ≥ slab − 1 percentage point in every case |
| Restart time | time from start to serving per disk and per node, after clean stop, kill -9 and simulated power loss (dm target where available), at full disk; with the fraction of objects recovered | after clean stop and kill -9: ≤ slab's time; after kill -9: every acknowledged object that was not evicted or removed is recovered (100%); after power loss: every sealed object recovered (slab recovers 0%) |
| PUT `kCacheFull` rate | share of PUTs answered `kCacheFull` at steady state, inline reclaims per second | < 0.1% of PUTs with the cleaner within its budget |
| Write amplification | relocated bytes per client byte | within `DFKV_LSE_RELOCATE_PCT` |
| REMOVE latency | REMOVE p50/p99, and GET p99 on the same connections during a REMOVE burst; separately, the worst REMOVE wait during a journal compaction with a large live set | REMOVE p99 and the GET p99 change reported, compaction pause included; decides §15 Q8 |
| Resources | index and shadow memory per million objects; CPU per GB written and read | reported |

If a criterion is not met, the report says which, by how much, and why, and the
default does not change.

---

## 14. Staged implementation plan

Each stage is one PR with its tests, mergeable on its own, and changes nothing
for a node that does not select the log engine.

**Gating.** From stage 3 until stage 10, selecting `log` additionally requires
`DFKV_LSE_EXPERIMENTAL=1`, and stores are created with the development-format
incompat bit (§4.2) set, so that a store from an unfinished stage is never
opened by a later build as if it were final. Stage 10 freezes the format,
clears the bit for new stores, and refuses stores that still carry it (they are
emptied by the operator).

| Stage | Content | Tests in the PR |
|---|---|---|
| 1 | Engine-neutral `EngineStats` in place of the group's direct `DiskSlabStore` use; `DFKV_STORE_ENGINE=auto|log` parsing and the directory classification of §10.1 (selecting `log` refuses with "not built yet") | classification table including `lost+found` and `blocks`, refusal messages, unchanged slab behaviour |
| 2 | Format library: CRC32C (hardware and software), superblock with reservations, segment header and trailer, record header, summary, journal block | unit tests, test vectors, fuzz targets, CRC throughput benchmark |
| 3 | `IoBackend` and the crash-image generator; `LogStore` core: init and resume, the open/activate split with the barrier, one open segment, buffered and direct appends, index, keep-first and flights, all read calls, the journal (ordered writer, write-once blocks, tickets enqueued under the store mutex, truncation to the stopping offset at `Activate`), REMOVE with kills, `Close`; recovery by walking every segment; no eviction yet (a full store answers `kCacheFull`) | conformance suites of §11.1 over `log`; crash points C1–C8, C16, C19, C22, C23, C25, C29, C33, C36 including their power-loss images; downgrade refusal via the slab open path |
| 4 | Several open segments, zero-copy `CacheDirect`, `CacheDirectBatch` (io_uring), `RangeDirectPrep` leases and segment pins, read verification and the `ReadLease` verifier hook in `KvNodeServer` | RDMA loopback and RAM tier wiring over `log`; lease and pin tests; C32 |
| 5 | Asynchronous sealing, summary reservation, summaries and trailers, clean close, summary-based recovery, sealing of walked segments at `Activate` | C9, C10, C18, C28; restart-time test at scale |
| 6 | Cleaner: reserves, victim FIFO with pin and protection skips, second-chance relocation, budget, reclaim with the step 2 sync, shadows, inline reclaim, re-PUT shadow kills, journal compaction, the reset to empty, capacity growth and shrink, metrics | C11–C15, C20, C21, C24, C26, C27, C30, C31, C34, C35; fill-and-evict endurance |
| 7 | RETAIN and TTL with footprint budget and protected-eviction fallback, generation semantics, tenant accounting | retention conformance against slab's tests; C17 |
| 8 | kill -9 soak tool, dm-target script, the long power-loss runs | §11.4 at full scale and §11.5 |
| 9 | Measurements of §13 on the development node; results added to this document | — |
| 10 | Format frozen; default switch to `auto` (empty directory gets `log`); ARCHITECTURE, DEPLOY and METRICS updated; CHANGELOG | §11.6 in full, including the recorded downgrade check with a release binary |

Stages 5–7 were built before stage 4, with *K* = 1 open client segment;
stage 4 raises *K*.

Stages 1–2 can proceed in parallel; 3 depends on both; 4–7 depend on 3 and can
partly overlap; 8 builds on the harness from 3 and is complete before 10; 10
waits for 9 and its review.

---

## 15. Open questions

Answered in this revision:

1. *How large can shadow sets get?* An entry is about 40 bytes (key and
   location). The cleaner works on one victim at a time per disk, so the
   shadows of a reclaim in progress are at most one segment's records: about
   400 for 2.74 MiB values in a 1 GiB segment, up to about 262,000 (10 MiB)
   for a segment full of 4 KiB records. The other sources (dead relocation
   copies, CRC failures, recovery losers) are rare and are released when their
   segments are reused; a loser in a young relocation segment can be held for
   a whole turnover of the disk, which is bounded by the records on the disk.
   The count is a metric (§9.5).
2. *What does a 2.35.x client do with a PUT `kCacheFull`?* It reports that PUT
   as not stored and does not count it against the peer: only `kIOError` cools a
   peer down (§3.3).
3. *A relocation read that fails its CRC check?* The record is evicted and its
   location becomes a shadow (§7.3).
4. *Segments walked per unclean exit?* At most *K* + 4, counting the
   relocation segment, the sealer queue and openings in progress (§8.4).

Still open:

5. Default segment size at full scale: 1 GiB (low tail waste for 50 MiB
   values) or smaller for finer eviction. Settled by the eviction-quality
   measurement.
6. Read verification on by default: depends on the measured CRC cost (§12.3).
7. Relocation budget default (25%) and whether the second-chance signal should
   require more than one read for large values.
8. Whether REMOVE's durable kill on the dispatch thread is acceptable when
   connectors remove a whole request's pages at once, or a batch REMOVE (one
   group for many keys) is needed. Settled by the REMOVE latency measurement.
9. Index bound for small-value workloads: a fixed entry count, or memory based.

---

## 16. Revision history

### Changes in revision 2 (review round 1)

Review round 1 found six must-fix problems, fifteen should-fix items and four
questions. All are addressed; where this revision does something other than
what the review proposed, the reason is given in place.

| Item | Change |
|---|---|
| M1 reclaim before relocated copy durable | §7.5 step 2 syncs every segment holding a copy before the victim's new header; C20 |
| M2 unaccounted copies | I5 restated as a rule over every path (§8.1); CRC failures, dead relocation copies and recovery losers become shadows (§5.2, §7.4, §7.5 table, §8.3 step 5); C32 |
| M3 kill journal | write-once blocks, one ordered writer with `RWF_DSYNC`, appends paused during compaction, a damaged block inside the durable range resets the store (§4.5); C23, C24 |
| M4 incarnation reuse | incarnations and LSNs reserved in the superblock and never reused (§4.2); free segments get a fresh durable header at `Activate` (§8.3 step 10); header I/O errors fail the open (step 4); LSNs never 0; C21, C33 |
| M5 late writes after a kill | `fallocate` barrier on every segment file before anything is read or reused (§8.3 step 2); supported filesystems stated (§1); C22 |
| M6 interface tests and `MaxValueBytes` | geometry derived from capacity with reserves inside it, minimum 32 MiB, `MaxValueBytes` = segment − 16 KiB and the mixed-node consequence (§4.7); the earlier claim of equality with slab is withdrawn |
| S1 cleaner starvation | cleaner-only reserve *C* (§7.1), bounded pin wait (§7.2), footprint budget (§9.2), protected eviction as last resort (§7.3); the false per-segment claim removed |
| S2 `kCacheFull` frequency | inline evict-only reclaim on the PUT path (§7.7); rate is an acceptance metric (§13) |
| S3 sealing and summary room | sealing asynchronous with a bounded queue (§6.3); summary room reserved per record (§6.1 step 4) |
| S4 write ordering | §6.5 states per-file durability and no cross-file order, `RWF_DSYNC` for headers, superblock and journal, directory syncs; grow and shrink ordered, leftovers tolerated (§7.6); C26, C27. Zeroing every segment at creation is not adopted (§4.6 gives the reason); the superblock, journal and headers are written at creation |
| S5 I/O errors at recovery | header and walk I/O errors fail the open; a foreign uuid refuses (§8.3 step 4) |
| S6 clean flag | every segment without a valid trailer is walked whatever the flag; `clean = 0` durable before the first operation (§8.5, §8.3 step 8); C29 |
| S7 open writes to disk | all writes moved to `Activate` (§8.3 steps 7–11); the open only reads, apart from the barrier, which is the repeatable allocation slab's open also performs |
| S8 equal generation | the higher `write_lsn` (the relocated copy) wins (§8.3 step 5) |
| S9 classification | precedence rules, `lost+found`, a FILE class marked by `blocks`, re-check under the engine's lock (§10.1) |
| S10 verification hook | `ReadLease` verifier called by the server before RAM install and coalesced hand-off (§3.2, §5.3); stage 4 |
| S11 stale value after overwrite | a PUT that stores a key with shadows kills them durably before it is acknowledged (§6.1, I3); C31 |
| S12 RETAIN persistence | SET and REFRESH return before their group write; the kill and power columns of C17 say the last few milliseconds may be lost (§9.2) |
| S13 REMOVE latency | stated as a measured item with its effect on the dispatch thread (§6.6, §13); batch REMOVE decided from it (§15 Q8) |
| S14 stages | experimental flag and development-format bit until stage 10; `IoBackend` and crash images from stage 3; downgrade refusal tested from stage 3 (§14) |
| S15 measurement | baseline, scenarios, steady state, noise band, cells and thresholds, RAM tier on and off, recovered fraction, `kCacheFull` rate, write amplification, REMOVE latency, memory and CPU (§13) |
| Verified points | downgrade refusal from 2.33.3 and the `DFKV_STORE_ENGINE` rollback caveat (§10.3); generation's restart stability stated as having no current use (§9.3); resynchronization skips by a header's length and runs to the end (§8.3 step 4); "never appended after a trailer" stated per incarnation (§4.3, §8.5) |

### Changes in revision 3 (review round 2)

| Item | Change |
|---|---|
| MF1 shadow kill durable after a later REMOVE | KILL entries are enqueued under the store mutex with journal tickets; every REMOVE, also one returning `kNotFound`, waits until all kills enqueued before it are durable; taken shadows stay in the set as *kill pending* (§4.5, §6.1, §6.6); C34 |
| MF2 crash-safe reset | reset ordered behind a durable `reset_in_progress` flag: superblock flag and incarnation bump, fresh headers, new journal, flag cleared; an open that finds the flag repeats it (§4.2, §4.5, §8.3 step 8); C35 |
| Divergence (b) conditions | residual risk of silent damage to the last journal blocks documented; a journal I/O error resets instead of failing the open (§4.5) |
| Divergence (a) condition | first fill and steady-state reuse measured separately (§13) |
| S1 barrier assumption | C22 is a real SIGKILL test with io_uring batch writes on XFS and ext4; quarantine of open segments as the fallback (§8.3 step 2, §11.3) |
| S2 buffered preallocation | buffered mode creates segment files on first use, without preallocation and without the barrier (§6.5) |
| S3 false resets | one block per journal write; the journal is truncated to its stopping offset at `Activate` before appending (§4.5); C36 |
| S4 activation order | the reservation bump, sized to the headers about to be written, is the first write of `Activate` (§8.3 step 7) |
| S5 inline reclaim | victim claimed under the mutex and excluded from the cleaner, header written outside it (§7.7) |
| S6 criteria | kill -9 criterion is 100% of acknowledged, not evicted, not removed; capacity criterion stated against both possible slab baselines (§13) |
| S7 free definition | every opening from the free list writes a fresh header first; `Activate` refreshes only invalid headers (§8.3 steps 6, 10) |
| S8 `lost+found` | slab's own open still refuses a non-empty root for a new store; stated (§10.1) |
| S9 compaction pause | part of the REMOVE latency criterion (§4.5, §13) |
| Open question | a grow that reuses a leftover file writes a fresh-incarnation header into it (§7.6) |
| §11.1 | log variants of tests with slab stores below 32 MiB set their own capacity |
