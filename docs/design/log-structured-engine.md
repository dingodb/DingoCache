# Log-structured storage engine (`log`): design

Status: design for review. No code exists yet. Nothing here changes the
behaviour of a released binary until the stages in §14 land, and the default
engine changes only after the acceptance measurements in §13.

This document describes a second disk engine for `dfkv_server`, a
log-structured store, to sit beside the slab engine (`DiskSlabStore`) behind
the existing `StoreEngine` interface. It covers the on-disk format, the read
and write paths, eviction and segment cleaning, crash consistency and
recovery, interaction with the RAM tier and with RETAIN, the rule that picks an
engine per directory, the test plan, the measurement plan, and a staged
implementation plan.

Terms: a *value* is the payload a PUT stores under a `BlockKey`; a *record* is
a value as laid out on disk; a *segment* is a fixed-size region of the disk
that records are appended to; *sealing* a segment makes it immutable;
*reclaiming* a segment makes its space available again; *cleaning* is the
copying of live records out of a segment before it is reclaimed.

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

Many of these are being improved inside the slab engine (exact-size classes,
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

---

## 2. Requirements

From the task contract, in priority order (correctness first):

| # | Requirement | Where addressed |
|---|---|---|
| R1 | No crash of any kind (process kill, kill -9, OOM kill, panic, simulated power loss) hands out wrong data | §8 (invariants I1–I5), §11.3–11.5 |
| R2 | Data that was sealed before a crash is recoverable after it | §8, §11.4 |
| R3 | Lives side by side with slab behind `StoreEngine`; passes all existing storage-interface tests | §3, §11.1 |
| R4 | Interoperable with 2.35.x clients and servers | §3.3 (no wire change) |
| R5 | After acceptance, an empty directory defaults to `log`; a directory holding slab data keeps using slab, never wiped on upgrade | §10 |
| R6 | Compared with slab on the same disks and real object sizes on effective capacity, read and write throughput, eviction quality and restart time | §13 |

"Wrong data" is defined precisely in §8.1. In short: the engine never returns
bytes for a key that are not exactly the bytes some PUT of that key stored,
never returns a value that a REMOVE which had returned removed, and never
returns a value older than a newer one that is still recoverable.

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
| `IsCached`, `Lookup` | index lookup; `ValueMetadata::generation` is the value's LSN, stable across restarts (§9.3) |
| `Remove` | erase from the index, then durable kill entries before returning (§6.6) |
| `UsedBytes`, `TenantUsedBytes`, `Count`, `Evictions`, `EvictedBytes` | from the index and counters; tenant use is rebuilt at open from record keys |
| `Retain`, `RetainEnabled`, `RetainHeadroomPct`, `RetainMaxTtl` | same codes and budget semantics as slab (§9.2) |
| `MaxValueBytes` | what one record can hold in a segment: segment size less its header, trailer and the record header block (§4.3, §4.6) |
| `Activate`, `Close`, `Healthy`, `StartupError` | two-step open like slab; `Close` seals summaries and records a clean shutdown (§8.5) |

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
- Nothing in `KvNodeServer`, the RAM tier, the RDMA server or the transports
  changes. They already talk only to `StoreEngine` and `ReadLease`.

### 3.3 Compatibility

The engine is local to a node. No frame, status code, RETAIN code or MDS
record changes, so 2.35.x clients and servers interoperate with a node running
it as with any other node. The MDS member statistics (capacity, used bytes,
evictions) are filled from the same engine-neutral numbers.

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
`extents/`) and from the file engine's block files, which is what the detection
rule in §10 relies on.

The directory is `flock`ed for the life of the process, as slab does: one
process per directory.

### 4.2 Superblock

Two 4 KiB copies at offsets 0 and 4096 of `lse_super`. A write updates the copy
with the lower sequence number, then `fdatasync`; a reader takes the valid copy
with the higher sequence number. A torn write therefore always leaves one good
copy.

| Field | Size | Meaning |
|---|---|---|
| magic | 8 | `DFKVLSE1` |
| format_major / format_minor | 4 + 4 | major changes are incompatible; minor are compatible |
| incompat_features / ro_compat_features / compat_features | 8 + 8 + 8 | bit sets, as in ext4: an unknown incompat bit refuses the open, an unknown ro_compat bit opens read-only (the engine then reports unhealthy rather than serve), an unknown compat bit is ignored |
| store_uuid | 16 | random at creation; stamped into every segment header, record and journal block |
| sequence | 8 | increments on every superblock write |
| segment_bytes | 8 | fixed at creation (default 1 GiB, §4.6 discusses the choice) |
| segment_count | 4 | number of segment files; may grow (§7.6) |
| align | 4 | 4096 |
| next_incarnation | 8 | lower bound for the next segment incarnation (a hint; the real value is recovered from segment headers) |
| next_lsn | 8 | lower bound for the next LSN (hint, as above) |
| clean | 4 | 1 if the last run closed cleanly and wrote open-segment summaries (§8.5) |
| created_by | 32 | version string of the release that created the store |
| reserved | … | zero |
| crc | 4 | CRC32C of the 4 KiB copy with this field zero |

### 4.3 Segment file

```
offset 0                 4 KiB segment header
offset 4096 …            records, each 4 KiB aligned
…                        summary (sealed segments only), right after the last record
segment_bytes − 4096     4 KiB trailer (sealed segments only)
```

**Segment header** (4 KiB): magic `DFLSSEG1`, store_uuid, segment index,
*incarnation* (a 64-bit number unique within the store, never reused: every
time a segment file is reused for new data it gets a new one), the first LSN
of the incarnation, and a CRC32C.

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

A non-inline payload starts at the next 4 KiB boundary after the header block
and is padded with zeros to a 4 KiB multiple. The next record starts after the
padding. So a record occupies `4096 + round_up(value_len, 4096)` bytes, or
exactly 4096 bytes when the payload fits inline (any value up to about
3.9 KiB, which covers the 1-byte markers some connectors write).

The header is in its own block rather than in front of the payload in the same
block so that the payload begins on a 4 KiB boundary: the PUT path can write
the caller's aligned buffer with `O_DIRECT` and no copy (§4.4), and a read of
the whole value is an aligned read with no head offset (§5).

**Summary** (sealed segments): an array of 64-byte entries, one per record in
the segment (key, value_len, generation, write_lsn, offset in 4 KiB units,
payload_crc, flags), followed by its own CRC32C. **Trailer** (fixed last
4 KiB): magic `DFLSEND1`, store_uuid, incarnation, record count, summary
offset, summary length, summary CRC, last write_lsn, trailer CRC. The trailer
is written only after the summary has been made durable (§8.2).

### 4.4 Space overhead

For a value of length *s*: 4 KiB header block plus padding to 4 KiB, plus its
64-byte summary entry; on average 4 KiB + 2 KiB + 64 B. Per segment, the
segment header, trailer and the unusable tail when the next value does not
fit (§4.6). For the object sizes measured in production:

| Value | Slab slot (today) | Log engine footprint | Log overhead |
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
| `KILL` | incarnation, segment index, offset, key, generation | a record that a REMOVE took out; recovery must not bring it back (§4.6, §8) |
| `RETAIN` | key, generation, retain_until, retain_ttl | persisted retention state (§9.2) |

Entries are packed into 4 KiB blocks; each block carries magic `DFLSJRN1`,
store_uuid, a block sequence number and a CRC32C, and is written with
`O_DIRECT`. The journal is append-only and compacted when it has grown past a
bound: the live entries are written to `lse_journal.new`, which is synced,
renamed over `lse_journal`, and the directory is synced. A crash during
compaction leaves either the old file (a superset) or the new one.

A `KILL` entry is live while the incarnation it names is still the current
incarnation of its segment; once that segment has been reused and its new
header is durable, the entry is garbage (§8.2, crash point C12). A `RETAIN`
entry is live while its (key, generation) is in the index and the deadline has
not passed.

### 4.6 Why these choices

- **Header block per record, not a per-segment table only.** A segment that
  was open at a crash has no summary; recovery must be able to walk it and
  check every record on its own. A forward walk needs the length of each record
  before its payload, hence a header in front. 4 KiB is the cost of keeping the
  payload aligned for zero-copy `O_DIRECT`; at the object sizes in use it is
  under 1%.
- **Incarnations instead of zeroing.** Reusing a segment does not erase it.
  Old records stay on disk until overwritten. Every record names its segment's
  incarnation, so after a crash bytes left over from an earlier incarnation are
  rejected by number, not by hoping they were cleared.
- **Kills in a journal, not tombstone records in segments.** A tombstone must
  outlive every older copy of the key, which ties its lifetime to other
  segments and makes cleaning order matter. A kill names the exact
  (segment, incarnation, offset) it invalidates, so it is needed exactly as long
  as that incarnation exists and can be dropped without looking at anything
  else.
- **Segment size.** Larger segments waste less at their tail when large values
  do not fit (with values up to 52 MiB and one open segment, a 256 MiB segment
  would lose about 10% on average to its tail; 1 GiB, about 2.5%; with several
  open segments filled best-fit, much less) and keep the summary count low.
  Smaller segments make eviction finer. The default is 1 GiB, the same as a
  slab extent, so `MaxValueBytes` stays what slab offers today; the choice is
  one of the measured parameters (§13).

---

## 5. Read path

### 5.1 Index

An in-memory hash map from `BlockKey` to an entry of about 48 bytes:
segment index (4), offset in 4 KiB units (4), value_len (8), generation (8),
retain_until and retain_ttl (8), flags (referenced, protected, relocating,
inline) and a per-entry pin count where needed. With 2.74 MiB values a 3.5 TiB
disk holds about 1.3 million entries, roughly 64 MiB with hash overhead.

A small-value workload can create far more entries per byte of disk than slab
ever allowed (a 1-byte value now costs 4 KiB of disk instead of 1 MiB). The
index therefore has a configured maximum entry count; when it is reached, the
oldest segment is reclaimed as if the disk were full (§7). The bound is
reported as a metric.

### 5.2 `Range`, `RangeInto`, `RangeDirect`

Under the store mutex: look the key up; a miss is `kNotFound`. Otherwise take a
*segment pin* (§5.3) and note the record's location, then release the mutex.
Read the requested slice of the payload (inline values come from the header
block). Release the pin.

A pin keeps a segment from being reused while bytes are being read from it.
It does not keep the key alive: a REMOVE or an eviction that happens during
the read erases the index entry at once, later reads miss, and the read in
progress still sees the old, unchanged bytes because segments are never
written in place. This replaces slab's deferred-removal windows and the
`kCacheFull` "busy" answers that came with them.

**Verification.** A read that brings a whole value into memory checks its
payload CRC before returning it, unless `DFKV_LSE_VERIFY_READS=0`. A mismatch
returns `kNotFound`, drops the index entry, counts
`dfkv_lse_payload_crc_failures_total` and logs the location: the client sees a
miss, never the bytes. Partial reads cannot be verified against a whole-value
CRC; §12.3 discusses whether to add per-4 KiB-block checksums in a later format
minor.

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

With verification on, the prepared path checks the CRC in the finish callback,
after the read completed and before the reply is posted. The bytes are in host
memory at that point (the registered buffer), so this is a CPU pass at memory
bandwidth and no extra I/O. A failure turns the reply into a miss.

---

## 6. Write path

### 6.1 Open segments and space reservation

A store keeps *K* open segments (default 4, `DFKV_LSE_OPEN_SEGMENTS`), so that
concurrent PUTs write to different files and the device sees parallel streams.
Each open segment has an append offset.

`CacheImpl` (shared by `Cache` and `CacheDirect`), under the store mutex:

1. If the key is committed: return `kOk` (keep-first).
2. If a PUT of the key is in flight: join it (a `PutFlight`, as slab), unless a
   REMOVE returned since that PUT began, in which case this PUT waits for the
   flight to end and then starts its own (the same rule slab uses so that a PUT
   after a REMOVE is never answered by a write the REMOVE cancelled).
3. If `value_len > MaxValueBytes`: `kInvalid`.
4. Choose the open segment with the least free space that still fits the
   record (best fit; reduces tail waste) and advance its append offset by the
   record size. If none fits, seal the fullest open segment (§6.3) and open a
   fresh one from the free list. If the free list is empty, the cleaner is
   behind: return `kCacheFull` (healthy, no room now), exactly the status the
   RAM tier and clients already handle.
5. Assign `write_lsn = generation = ++next_lsn`, register the flight, pin the
   segment.

Then, without the mutex: build the header block, compute the payload CRC and
write header and payload with one `pwritev` at the reserved offset
(`O_DIRECT`, see §6.2). Then, under the mutex: if a REMOVE of the key arrived
meanwhile, do not publish (the record stays on disk unpublished and is covered
by a kill, §6.6); otherwise insert the index entry. Complete the flight and
unpin.

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

A segment is sealed when the next record does not fit and the segment is the
fullest of the open ones, or at `Close`. Sealing waits for the segment's
in-flight writes to finish (the reservation pins it), then:

1. `fdatasync` the segment file: every record is durable.
2. Write the summary after the last record, `fdatasync`.
3. Write the trailer, `fdatasync`.

The segment is then immutable until it is reclaimed. Its summary is also kept
in memory in compact form for the cleaner.

### 6.4 Batches (`CacheDirectBatch`)

The RAM tier's flusher hands batches of values to `CacheDirectBatch`. The log
engine reserves one contiguous range in a single open segment for the whole
batch (or as much of it as fits), so the batch is one `pwritev` or one io_uring
submission of consecutive iovecs, then commits all its entries in one mutex
hold. Per-item semantics (keep-first, flights, `kCacheFull`, `kInvalid`) are
those of `CacheDirect`.

### 6.5 Periodic sync

Every `DFKV_LSE_SYNC_MS` (default 1000) the open segment files are
`fdatasync`ed. This bounds what a power loss can lose to roughly the last
interval of acknowledged writes; it does not affect correctness (§8), and a
process kill loses nothing that was acknowledged, because `O_DIRECT` writes
are in the device when they complete.

### 6.6 REMOVE

Under the mutex: erase the index entry (later reads miss at once), and mark any
in-flight PUT of the key as cancelled. Collect every disk location that could
hold a recoverable value of the key: the committed record, a record being
written by a cancelled PUT, the destination of a relocation in progress
(§7.4), and the key's *shadows* (§7.5): copies the cleaner has moved away from
or evicted in a segment whose reuse is not yet durable. Release the mutex.
Append one `KILL` entry per location to the journal and wait until the journal
block is durable (group commit: concurrent REMOVEs share one write and one
flush). Then return.

So a REMOVE that returned is durable against any crash (I2 in §8.1),
including when it returned `kNotFound` because the key had just been evicted.
REMOVE is rare in the workloads dfkv serves (eviction is not a REMOVE and
needs no kill, §8.1), so the flush per group is affordable; §13 measures it.

---

## 7. Eviction and segment cleaning

### 7.1 Free space

The store keeps a reserve of free segments: the *K* that the next openings
need, plus a cleaning reserve (default 2). The cleaner runs whenever the free
count falls below the reserve and keeps it there. There is no percentage
watermark: the space held back is (*K* + 2) segments per disk, 6 GiB at the
defaults, about 0.2% of a 3.5 TiB disk.

### 7.2 Choosing the victim

Sealed segments form a FIFO in seal order. The cleaner takes the oldest sealed
segment that has no pins it would have to wait for long, and that is not
dominated by protected values (§9.2). Segments are therefore reclaimed in the
order they were filled, which approximates "evict the oldest writes first";
§7.3 adds the reuse signal that makes it approximate least-recently-*used*.

### 7.3 What survives a reclaim

For each record in the victim's summary whose index entry still points at this
location (dead records, superseded or removed or already relocated, are
skipped):

| Record | Action |
|---|---|
| protected by RETAIN (deadline in the future) | relocate |
| *referenced*: read at least once since it was written or last relocated | relocate, clear the bit |
| otherwise | evict: drop the index entry, count it |

This is second-chance reinsertion at segment granularity (a FIFO with
reinsertion of accessed objects, the structure of CLOCK and of S3-FIFO's main
queue). The *referenced* bit is set by every disk read hit and, once REQ-17's
follow-up lands, by the RAM tier when it evicts a value it served (§9.1).

**Write-amplification budget.** Relocation is bounded: over a sliding window,
relocated bytes may be at most `DFKV_LSE_RELOCATE_PCT` (default 25) percent of
the bytes clients wrote. When the budget is exhausted, referenced but
unprotected records are evicted instead of relocated until the window
recovers. Protected records are always relocated; their total is bounded by
the retention budget (`DFKV_RETAIN_MAX_PCT`), which is what guarantees the
cleaner can always make progress: at most that share of any segment can be
protected, and the cleaner skips a segment that is mostly protected in favour
of the next oldest.

### 7.4 Relocation

The cleaner reads a live record (header and payload) with `O_DIRECT`, verifies
its CRC, and appends it to a dedicated *relocation* open segment, with
`generation` unchanged, a new `write_lsn` and the relocated flag. The
relocation segment is separate from the client open segments so that
long-lived values cluster together and are not mixed with new writes (this
reduces how often they are relocated again).

Before writing, under the mutex, the cleaner marks the entry *relocating* and
records the destination location. After the write, under the mutex: if the
entry still names the same generation at the source location, it is switched
to the destination; otherwise (the key was removed or replaced meanwhile) the
copy is dead. A REMOVE that arrives while the entry is *relocating* includes
the destination in its kill set (§6.6), so a dead copy can never be recovered.
After a successful switch the source location becomes a shadow of the key
until the source segment is reclaimed (§7.5).

### 7.5 Reclaim and shadows

When every live record of the victim has been relocated or evicted:

1. Wait until the victim has no pins (leases and reads in progress).
2. Write a new segment header with a new incarnation, `fdatasync`.
3. Drop the victim's shadows (below).
4. Garbage-collect journal kill entries for the old incarnation (lazily, at
   the next compaction).
5. Put the segment on the free list.

Until step 2 is durable the victim's records are still recoverable: a crash
there brings back the values that were relocated away (as duplicates of the
same generation) or evicted. The cleaner therefore keeps, for the victim, the
set of keys and locations it moved or evicted, its *shadows*, until step 2
completes. A REMOVE of a key kills its shadows as well as its indexed copy
(§6.6). This is what makes invariant I5 (§8.1) hold, and with it I2 even for a
REMOVE that finds the key already evicted.

Step 2 before steps 3 and 4 is required for correctness (crash points C11,
C12).

### 7.6 Capacity changes

The segment count is in the superblock. Growing the configured capacity adds
segment files and rewrites the superblock; it never touches existing data.
Shrinking reclaims (evicts) the highest-numbered segments first, then removes
their files. Slab today refuses any capacity change without a wipe.

### 7.7 Prefix chains

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
  payload CRC, and the store_uuid and incarnation match on every record.
- **I2 (no resurrection after REMOVE).** A value present before a REMOVE that
  returned (`kOk`, or `kNotFound` for a key just evicted) is never returned
  again. Enforced by durable kills of the indexed copy and of every shadow
  (§6.6, §7.5, I5).
- **I3 (newest wins).** If several recoverable records of *K* exist, recovery
  keeps the one with the highest `generation`; equal generations are copies of
  the same value (relocation) and either may be kept.
- **I4 (sealed data survives).** Every record in a segment whose trailer was
  durable is recovered, unless it was killed, superseded, or evicted before
  the crash (an evicted record may come back; that is allowed, see below).

The engine maintains one invariant at run time that the others rest on:

- **I5 (no unaccounted copies).** Every record on disk that would pass
  recovery's checks and is not killed is either the indexed location of its
  key or a shadow of its key (§7.5). An older value of a key cannot linger
  beside a newer one: a new value is stored only after the old one was removed
  (killed) or evicted (a shadow until its segment is reused). Hence a REMOVE,
  which kills the indexed location and the shadows, leaves no recoverable copy
  of the value it removed.

What is allowed after a crash: losing acknowledged values that were not yet
covered by a sync (power loss only; §6.5 bounds it), and the return of values
that had been *evicted* (not removed) shortly before the crash. An evicted
value is a correct value of its key; a cache may hold it again. A REMOVE that
had not returned when the crash happened may or may not have taken effect.

### 8.2 Crash-point table

"Kill" means a process exit at that point (page cache and device contents
survive); "power" means a loss of power or kernel crash at that point (writes
not covered by a completed `fdatasync` may be lost, torn at 512 B or 4 KiB
granularity, or reordered among themselves).

| # | Operation | Crash point | Kill: outcome | Power: outcome |
|---|---|---|---|---|
| C1 | first init | after `lse_initializing`, before superblock | next open resumes the init (only marker and partial files present) | same |
| C2 | first init | superblock durable, segments partly preallocated | resume completes preallocation | same |
| C3 | PUT | space reserved, nothing written | bytes at that offset are old (other incarnation or zero): rejected; the walk resynchronizes past the hole and keeps later records | same |
| C4 | PUT | header written, payload partial | not acknowledged; payload CRC fails: rejected | same; or header lost: rejected |
| C5 | PUT | written, index not yet updated | not acknowledged; record valid: recovered (allowed, a PUT that completed its write) | may be lost or torn: rejected if torn |
| C6 | PUT | acknowledged | recovered (`O_DIRECT` completed; the walk's resynchronization keeps it even if an earlier reservation left a hole) | recovered if covered by a sync; else lost or rejected, never wrong |
| C7 | REMOVE | index erased, kill not durable | not acknowledged: value may come back (allowed) | same |
| C8 | REMOVE | kill durable, returned | never comes back | never comes back |
| C9 | seal | summary written, trailer not | segment recovered by walking records (as an open segment) | same; summary may be torn, ignored |
| C10 | seal | trailer written, summary torn | — (trailer is written only after the summary is durable) | trailer's summary CRC fails: walk records |
| C11 | reclaim | index entries dropped, new header not durable | old incarnation still on disk: its records recovered (evicted values back, allowed; relocated ones as same-generation duplicates); records of keys removed meanwhile stay out, because REMOVE killed the shadows and kills are not yet collected | same |
| C12 | reclaim | new header durable, kills not collected | old records rejected by incarnation; stale kills ignored (incarnation mismatch) | same |
| C13 | relocation | copy written, index not switched | two copies, same generation: either kept (identical bytes, I3) | copy may be torn: rejected, source still there |
| C14 | relocation | REMOVE during or after relocation | kills cover the destination and the source (a shadow until the source is reclaimed) | same |
| C15 | journal compaction | new file written, rename not durable | old journal (superset) used | same |
| C16 | superblock update | torn | other copy used | same |
| C17 | RETAIN SET | journal block not yet written | protection lost (allowed; §9.2) | same |
| C18 | `Close` | some open segments sealed, clean flag not set | open treats it as unclean: walks the segments without a valid trailer | same |
| C19 | any | device returns an I/O error | the store fails closed (unhealthy, `kIOError`), as slab does | — |

### 8.3 Recovery procedure

Open, in order:

1. Lock the directory. Read both superblock copies; take the valid newer one.
   Refuse on unknown incompat features, store_uuid mismatch with any file, or
   a geometry the files do not match.
2. Read the journal: build the kill set {(segment, incarnation, offset)} and
   the retention map. Ignore blocks with a bad CRC or another store_uuid; stop
   at the first block whose sequence number breaks the run (a torn tail).
3. For each segment file, in parallel across a small thread pool: read the
   header. A bad or foreign header means the segment holds nothing usable: it
   is free. Otherwise read the trailer. If the trailer is valid for this
   incarnation and its summary CRC matches, load the summary. Otherwise walk
   the segment: read the 4 KiB block at the walk position; if it is a valid
   record header of this store and incarnation, read and check the payload CRC
   (inline payloads are in the block), keep the record if it passes, and move
   to the next record boundary its header gives. If the block is not a valid
   header (a write reserved there never reached the disk, or was torn), the
   walk *resynchronizes*: it advances one 4 KiB block at a time and resumes at
   the next block that is a valid header of this incarnation **and** whose
   payload CRC verifies. Concurrent writers fill an open segment at different
   offsets, so a hole left by one writer must not hide the records another
   writer completed and had acknowledged. (A payload could only pass as a
   header by containing this store's uuid, which never leaves the node, the
   current incarnation and a valid header CRC, and then its supposed payload
   would also have to verify.)
4. Merge: for each record not in the kill set, insert or replace by
   `generation` (I3). A record that loses (a lower generation, or a duplicate
   of the winner's generation) is registered as a shadow of its key, held
   until its segment is reclaimed, so that a REMOVE after the restart kills it
   too (I5). Such losers exist only when a crash interrupted a reclaim (C11,
   C13), so the set is small. Rebuild per-tenant used bytes and the referenced/protected
   flags (protected from the retention map, matched on key and generation;
   referenced is cleared, every value is as old as the restart).
5. Rebuild the free list (segments with no live records, or free headers), the
   seal FIFO (by last write_lsn), and `next_lsn` / `next_incarnation` as one
   past the highest seen.
6. Seal every walked segment as it is: write a summary of the records the walk
   kept and a trailer, `fdatasync`. A walked segment is never appended to
   again, so whatever lies between or after its recovered records (torn
   writes, holes) can never be reached by a later walk or mixed with new
   records; the cost is the unused tail of at most *K* + 1 segments per
   unclean exit. Open *K* fresh segments from the free list.

After step 6 the store is opened (counts and stats available, no data
operation admitted); `Activate()` makes it serve, as with slab.

### 8.4 What a walk costs

A walk reads every header block and every payload of the segments it walks,
to check their CRCs. After a clean `Close` nothing is walked (§8.5). After an
unclean exit only the segments that were open are walked: *K* + 1 segments per
disk (the client open segments and the relocation segment), at most
(*K* + 1) × 1 GiB of sequential reads, roughly 2 s per disk at the defaults on
current NVMe (to be measured). All other segments contribute their summary
only.

### 8.5 Clean close

`Close()` (the server's shutdown and the destructor): stop the cleaner, wait
for in-flight writes, seal every open segment (§6.3), then write the
superblock with `clean = 1`. The next open finds only sealed segments, trusts
every trailer and walks nothing; it opens fresh segments for new writes.
Activation writes `clean = 0` before admitting the first operation.

A segment, once it has a trailer, is never appended to again, after a clean
close as after a walk (§8.3 step 6). A trailer that did not cover the whole
segment would otherwise go stale as soon as a record was appended after it.
The price is the unused space at the end of up to *K* + 1 segments per
restart, until those segments age out.

Unlike slab, nothing depends on the clean flag for correctness: it only saves
the walk.

---

## 9. Interaction with the rest of the server

### 9.1 RAM tier

The RAM tier talks to the disk through the group and `StoreEngine` only, so it
needs no change:

- **Flush**: `CacheDirectBatch`, one contiguous append per batch (§6.4).
  `kCacheFull` from a cleaner that is behind is handled as "no room now" (held
  in RAM, retried; added in 2.40). `kInvalid` for a value above
  `MaxValueBytes` is refused at admission (2.41).
- **Promotion**: whole-value reads through `RangeDirectPrep` with `head = 0`,
  straight into a RAM reservation.
- **`is_persisted` / `disk_generation`**: `Lookup` gives committed state and
  the value's generation (§9.3). The REMOVE fence and the promotion checks
  added in 2.41 work unchanged.

Two hooks are planned for later, optional on the engine side (no-ops on slab
until implemented there): `Touch(key, generation)` so that the RAM tier can
set the disk's *referenced* bit when it evicts a value it served (the review's
E9/REQ-17 follow-up), and an eviction callback so that the RAM tier drops its
written copy when the disk evicts the key.

### 9.2 RETAIN and TTL

Semantics are those of slab: SET and REFRESH, the same `RetainCode`s, the
`DFKV_RETAIN_MAX_TTL_S` cap, the `DFKV_RETAIN_MAX_PCT` budget (as a share of
the store's payload capacity) and `RetainHeadroomPct`. The class budget
(`DFKV_RETAIN_MAX_CLASS_PCT`) has no meaning without classes and is ignored.

- Protection lives in the index entry; deadlines expire through a timing
  wheel with one entry per protected value, cleared in bounded batches (the
  same structure slab adopted in 2.39).
- **Persistence**: a SET, or a REFRESH that moves the deadline by at least
  `DFKV_RETAIN_PERSIST_MIN_DELTA_S`, appends a `RETAIN` journal entry
  (key, generation, retain_until, ttl). Entries are written in groups every
  few milliseconds with `O_DIRECT`, so a process kill keeps them; a power loss
  may lose the last interval (crash point C17). Protection is a best-effort
  minimum lifetime in both engines, and losing it never produces wrong data.
- **Cleaning**: a protected record is always relocated, never evicted (§7.3).

### 9.3 Generation

`ValueMetadata::generation` is the record's `generation` field: the LSN at
which the value was first committed. It is unique per store, increases with
every commit, survives restarts (it is on disk), and is kept when the cleaner
relocates the value. So the RAM tier's "the disk still holds the value this
copy was read from" check (2.41) holds across restarts and relocations with
this engine, which it does not with slab, whose generations are in-memory
only.

### 9.4 Tenants and quotas

The tenant is part of the key and so of every record; `TenantUsedBytes` is
kept in memory and rebuilt at open. Quotas are enforced by the group, as
today.

### 9.5 Metrics

A `dfkv_lse_*` family per disk: segments (free / open / sealed / relocation),
bytes (payload, record overhead, dead in sealed segments), cleaner (segments
reclaimed, records relocated / evicted, relocated bytes, budget exhaustion
events), recovery (segments walked, records recovered, records rejected by
reason: header CRC, payload CRC, incarnation, killed), journal (entries,
compactions, sync latency), payload CRC failures on reads, index entries and
its bound. The engine-neutral numbers (`dfkv_used_bytes`, evictions, objects)
come from the same `EngineStats` the slab engine fills.

---

## 10. Engine selection per directory

### 10.1 Rule

`DFKV_STORE_ENGINE` / `--store-engine` takes `slab`, `log`, `file` or `auto`.
Each configured directory is classified before any engine opens it:

| Directory contains | Class |
|---|---|
| nothing | EMPTY |
| any of `slab_meta`, `slots.tbl`, `slab_state`, `slab_initializing`, `extents` | SLAB |
| any of `lse_super`, `lse_journal`, `lse_initializing`, `segments` | LOG |
| entries of both kinds | CONFLICT |
| anything else | FOREIGN |

| Setting \ class | EMPTY | SLAB | LOG | CONFLICT / FOREIGN |
|---|---|---|---|---|
| `auto` | default engine (§10.2) | slab | log | refuse to start |
| `slab` | slab | slab | refuse | refuse (as slab does today) |
| `log` | log | refuse | log | refuse |
| `file` | file | refuse | refuse | file (as today: the file engine's own checks apply) |

"Refuse" is a startup failure that names the directory and the files that
decided its class, and it never deletes or rewrites anything. Going from one
engine to another on a directory is an operator action: empty the directory
(or point the server at a new one). This keeps the slab rule "unknown and
previously populated layouts are never auto-recreated".

### 10.2 Defaults over time

- Until acceptance (§13): the default stays `slab`, so nothing changes for
  anyone who does not ask for `log`.
- After acceptance: the default becomes `auto`. An empty directory then gets
  the log engine, a directory with slab data keeps slab, and nothing is wiped
  by the upgrade (R5).

### 10.3 Mixed nodes, upgrade and downgrade

- Each disk is classified on its own; a node may run slab on some disks and
  log on others. Per-disk metrics carry the engine name.
- **Upgrade** from a release without the log engine: directories hold slab
  data and stay slab under `auto`.
- **Downgrade** to a release without the log engine, on a directory the log
  engine created: the slab engine of every release from 2.35 on refuses a
  non-empty directory without `slab_meta` ("cache directory contains a
  different or incomplete store layout", `DiskSlabStore::OpenOrInit`), so an
  older binary refuses to start on it and does not wipe it. The operator either
  empties the directory or stays on the newer release. This is tested (§11.6).

---

## 11. Test plan

### 11.1 Interface conformance

The existing storage-interface tests run against both engines:

- `store_engine_test` (round trips, the range conformance table, engine
  selection, startup refusals), `disk_cache_group_test`, `kv_store`-style
  tests of the `StoreEngine` contract, the RAM tier wiring tests that build
  real stores, the node-health tests, the RDMA loopback tests' prepared-read
  paths, and the client end-to-end tests.
- These are parameterized over the engine (`slab`, `log`; `file` where it
  applies today). Tests of slab internals (classes, steals, trust tokens,
  `slots.tbl` rebuild) stay slab-only; each gets a log-engine counterpart
  where the property applies (for example "a REMOVE during an in-flight read
  never serves torn bytes", "keep-first under concurrent PUTs", "a PUT after a
  REMOVE is stored after it").

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
and checks the outcome column. Power-loss rows run under §11.4.

### 11.4 Simulated power loss

All engine I/O goes through a small `IoBackend` interface (pwrite/pwritev,
pread, fdatasync, fallocate, rename, fsync of a directory). A recording backend
used in tests keeps, per file, the writes issued since the last completed sync.
A *crash image* is produced by:

- keeping every write covered by a completed sync;
- for the rest, independently: dropping it, keeping it, or tearing it (keeping
  a prefix, at 512 B and at 4 KiB granularity), and applying the kept ones in a
  random order;
- treating renames as durable only after the directory sync.

The test then opens the image and checks I1–I4 against a model: the workload
generator knows, for every key, which PUTs and REMOVEs were acknowledged and
when the covering syncs completed. Values are derived deterministically from
(key, generation) so any served byte can be checked without storing a copy.
The suite runs a few thousand seeds per workload (PUT-heavy, REMOVE-heavy,
cleaning under pressure, RETAIN churn, concurrent reads with leases) in CI and
many more on the development node.

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

Every cell of the table in §10.1, plus: an empty directory gets the log engine
under the post-acceptance default; a slab directory written by a release
binary is detected and kept; a log directory is refused by the slab engine
(the downgrade case) without any file being changed; mixed disks in one node.

### 11.7 Concurrency and endurance

ThreadSanitizer and AddressSanitizer builds of the engine tests; a stress test
with cleaner, readers holding leases, REMOVEs, relocations and batches running
together for hours; a long fill-and-evict run checking that free segments never
run out while the cleaner is within budget, and that `kCacheFull` appears only
when everything is protected or pinned.

---

## 12. Performance considerations

### 12.1 Writes

A PUT is one `pwritev` of header and payload into a reserved range: no table
write, no allocator search, no class. Up to *K* open segments write in
parallel. Batched flushes are one submission. The store mutex is held only for
reservation and commit, as in slab.

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
budget in §7.3. Reclaim itself is one 4 KiB header write and a sync.

### 12.5 Restart

Restart reads the superblock, the journal, and one summary per sealed segment
(64 bytes per record: about 80 MiB for 1.3 million records), and walks the open
segments only after an unclean exit (§8.4). Disks open in parallel, as they
already do.

---

## 13. Measurement plan for acceptance

Same development node, same NVMe disks, same filesystem, same
`dfkv_server` binary; the engine is selected per run on emptied directories.
Each run is repeated three times; the report gives the median and the spread.

Object sizes: 1 B (markers), 1.02 MiB, 2.74 MiB, 25.9 MiB and 51.75 MiB values
(the sizes observed for attention pages and linear-attention state in
production), and a mix drawn from production access logs with their size
distribution.

| Metric | How it is measured | Reported as |
|---|---|---|
| **Effective capacity** | fill a disk to twice its capacity with each size and with the mix; at steady state (evictions running) read the committed payload bytes | payload bytes / configured capacity, and object count, per size, per engine |
| **Write throughput** | `dfkv_bench` PUT over RDMA (direct write path) and TCP, RAM tier off and on, 1–N clients, at the sizes above; also during cleaning at full disk | GB/s, ops/s, p50/p99 latency, CPU per GB |
| **Read throughput** | cold GETs from disk (RAM tier off), RDMA prepared reads and TCP, random keys over a full disk; with and without read verification | GB/s, ops/s, p50/p99, CPU per GB |
| **Eviction quality** | the existing prefix-reuse replay scenario and replays of production access logs at 0.5×, 1× and 2× overcommit | object hit ratio, byte hit ratio, reusable prefix tokens per request, and for the log engine relocated bytes per client byte |
| **Restart time** | full disk; restart after clean stop, after kill -9, after a simulated power loss (dm target where available) | time from start to serving per disk and per node, and fraction of objects recovered |

Acceptance criteria, to be confirmed in review before measurement starts: the
log engine is at least as good as slab on restart time and on eviction quality
(within measurement noise), better on effective capacity for every real size,
and within 10% of slab on read and write throughput, with no correctness
failure in §11. If one of these is not met, the report says which, by how much,
and why, and the default does not change.

---

## 14. Staged implementation plan

Each stage is one PR with its tests, mergeable on its own, and changes nothing
for a node that does not select the log engine.

| Stage | Content | Tests in the PR |
|---|---|---|
| 1 | Engine-neutral `EngineStats` in place of the group's direct `DiskSlabStore` use; `DFKV_STORE_ENGINE=auto|log` parsing and the directory classification of §10.1 (selecting `log` refuses with "not built yet") | classification table, refusal messages, unchanged slab behaviour |
| 2 | Format library: CRC32C (hardware and software), superblock, segment header and trailer, record header, summary, journal block | unit tests, test vectors, fuzz targets, CRC throughput benchmark |
| 3 | `LogStore` core: init and resume, one open segment, buffered and direct appends, index, keep-first and flights, all read calls, REMOVE with the journal, `Close`; recovery by walking every segment; no eviction yet (a full store answers `kCacheFull`) | conformance suites of §11.1 parameterized over `log`; crash points C1–C8 |
| 4 | Several open segments, zero-copy `CacheDirect`, `CacheDirectBatch` (io_uring), `RangeDirectPrep` leases and segment pins, read verification | RDMA loopback and RAM tier wiring over `log`; lease and pin tests |
| 5 | Sealing, summaries, trailers, clean close, summary-based recovery | C9, C10, C18; restart-time test at scale |
| 6 | Cleaner: free reserve, victim FIFO, second-chance relocation, budget, reclaim with incarnations, journal compaction, capacity growth and shrink, metrics | C11–C16; fill-and-evict endurance; cleaner metrics |
| 7 | RETAIN and TTL, generation semantics, tenant accounting | retention conformance against slab's tests; C17 |
| 8 | `IoBackend`, the power-loss image generator, kill -9 soak tool, dm-target script | §11.4 and §11.5 suites |
| 9 | Measurements of §13 on the development node; results added to this document | — |
| 10 | Default switch to `auto` (empty directory gets `log`); ARCHITECTURE, DEPLOY and METRICS updated; CHANGELOG | §11.6 in full, including the downgrade refusal |

Stages 1–2 can proceed in parallel; 3 depends on both; 4–7 depend on 3 and can
partly overlap; 8 starts with 3 (crash hooks) and is complete before 10; 10
waits for 9 and its review.

---

## 15. Open questions for review

1. Default segment size: 1 GiB (as proposed, same `MaxValueBytes` as slab, low
   tail waste for 50 MiB values) or smaller for finer eviction. To be settled by
   the eviction-quality measurement.
2. Read verification on by default: depends on the measured CRC cost (§12.3).
3. Relocation budget default (25%) and whether the second-chance signal should
   require more than one read for large values.
4. Whether REMOVE's journal flush is acceptable on workloads that remove in
   bulk (some connectors delete a whole request's pages at once); a batch
   REMOVE would share one flush.
5. Index bound for small-value workloads: a fixed entry count, or memory based.
6. Whether to give the file engine (diagnostic) a detection marker as well, so
   that `auto` can recognise it rather than call it FOREIGN.
