# Explicit offline STL3 slab → LSE migration

`dfkv_slab_migrate` is an opt-in offline copy tool. It does **not** select an
engine, rewrite a configuration, rename a live directory, replace an existing
store, resume/merge into a previous candidate, or switch a service. Activation
of the resulting candidate remains a separate operator decision.

```bash
build/dfkv_slab_migrate \
  /absolute/offline/already-closed-slab-root \
  /absolute/new/candidate-lse-root \
  /absolute/evidence/migration.json
```

All three paths must be absolute and lexically normalized. Source and report
parents must be canonical (not symlink aliases). The candidate,
`candidate.sto14-staging`, and report must be absent. The source must be
quiesced/offline. The reader takes the slab directory's cooperative exclusive
lock and uses only `O_RDONLY|O_NOFOLLOW` descriptors, with relative `openat`
access for source members. This excludes a cooperating `DiskSlabStore` writer;
as with any advisory lock, it is not a defense against an arbitrary process
that ignores that lock. Full source re-enumeration and per-file SHA-256,
identity, size, timestamps, link count and `st_blocks` checks run before and
after copying.

## Accepted source and explicit limits

The reader derives geometry from `slab_meta`; it never asks `DiskSlabStore` to
open or rebuild the original. It requires a complete STL3-only root, exact
extent inventory and sizes, a CRC-valid **clean** `slab_state`, CRC-valid live
records, known flags/reserved fields, exact slot geometry, one class per
extent, unique physical locations, no foreign/history files, and unique full
`(tenant_hash,digest_hi,digest_lo)` keys. CRC-protected source payloads are
verified when read.

The following are errors, not implicit cache resets or dropped values:

- missing/dirty/corrupt epoch state, an initializing/pre-STL3 or mixed root;
- corrupt, partial, rejected or unknown rows, duplicate/conflicting keys;
- mixed slot classes or source/LSE size incompatibility;
- tombstones or retention/TTL records (their history is not erased);
- zero-length live records, which ordinary LSE put/recovery cannot preserve;
- a destination/staging path that exists, or unsafe source/output aliasing;
- any source change, failed write, close, reopen, header, CRC or reconciliation.

The accepted history scope is deliberately narrow: a clean source whose table
contains only live values and zeroed free rows. No retention deadline, old
slab generation, chain ownership, tombstone or deleted-data history is claimed
as preserved. `BlockKey` fields and opaque value bytes (including zero bytes,
one-byte `0x00`, ordinary one-byte `0x7f`, and non-default tenant hashes) are
copied without interpreting a value as a connector marker.

## Failure and publication protocol

The tool creates a new `candidate.sto14-staging`, chooses a segment size that
can hold the largest source value, and forward-packs the ordered source record
footprints as an ordinary LSE writer will place them. The budget includes each
record's next 64-byte summary entry rounded to a block, segment headers and
trailers, and the ordinary client-write reserve of two cleaner segments. It
also respects the minimum segment count and rejects count/capacity overflow.

For this offline transfer only, `open_segments=1` and `preopen=false` keep that
forward-packing model and actual admission behavior aligned for nonuniform
record sizes as well as small inline entries. This is not an engine-default or
production configuration change. The data is written through ordinary
`LogStore::Cache`; LSE generations/write LSNs are new writer state. Source slab
generation numbers are not lineage and are not copied.

Before opening the LSE writer, the tool reserves an unnamed file on the staging
filesystem and keeps its descriptor open. This leaves the new LSE root empty
while reserving the ability to record a later `EMFILE`/writer failure. On that
failure the inode is materialized as `STO14-MIGRATION-INCOMPLETE` with the
failure reason; no new output file descriptor is needed to open the marker.
This is a Linux `O_TMPFILE` plus `/proc/self/fd/<fd>` `linkat` operation without
`O_EXCL` or additional capabilities. If that mechanism is unavailable, or the
staging directory still cannot be opened for synchronization, the failure is
reported and no partial candidate is published. A pre-writer reservation failure
starts no LSE at all. Successfully verified staging closes the unused inode
before publication, so no failure marker enters the valid candidate.

A failed LSE write/reopen/validation therefore leaves clearly marked staging
and never publishes a partial candidate. Such staging is retained for diagnosis
and is never reused automatically.

Only after a clean writer close, ordinary LSE close/reopen, exact key-set and
full-payload SHA comparison, live `Lookup` checks, superblock/segment/summary/
record-header decoding, payload CRC verification, and repeated source
invariance checks does the tool use `renameat2(RENAME_NOREPLACE)` to publish.
The verifier is destroyed before this atomic publication. Parent-directory
sync failure after rename is reported as published-but-durability-uncertain,
never as an absent destination.

## Report interpretation and bounded verification

The JSON report includes exact source and output paths, byte size, elapsed
microseconds, complete key tuples and value SHA-256, record flags,
`header_len`, `value_len`, `generation`, `write_lsn`, `incarnation`,
`payload_crc`, `RecordFootprint`, source slot charge, and full source/output
file inventories with SHA-256 and `st_blocks` data.

These are distinct quantities:

- a slab live row costs **64 B** in `slots.tbl` and charges its slot size;
- with 1 MiB granularity, each small live value has a 1 MiB logical slot
  charge, not proof of 1 MiB physical allocation, RAM, or PUT bytes;
- LSE inline small values have an 88 B header within a 4096 B record block and
  `RecordFootprint=4096`; sealed summaries add 64 B per record plus segment,
  trailer, superblock and journal metadata;
- `st_blocks` reports filesystem allocation and is neither resident memory nor
  logical occupancy. No fixed space-saving or performance ratio is promised.

Focused tests cover native 1 MiB-extent fixtures, read-only source trees,
both genuine connector no-layout forms (`None` and `(None,None)`), complete
four-page marker/side sets, a missing fourth anchor, a missing fourth side
value, ordinary `0x7f`, non-default tenants, exact reopen payloads and decoded
LSE headers. Reject fixtures cover epoch, geometry, corruption, duplicates,
retention/history, foreign roots, existing output, and real staged-open
failure. To select just these native cases in a configured build:

```bash
build/slab_migration_test
DFKV_BUILD="$PWD/build" DFKV_LIB="$PWD/build/libdfkv.so" \
DFKV_STORE_ENGINE=file DFKV_SLAB_WRITE=buffered \
PYTHONPATH="$PWD/integration/common/src" \
python3 -B test/python/test_dfkv_hicache.py \
  DingoFSHiCacheTest.test_sto14_migrates_real_slab_anchor_variants_to_lse
```

The source/output commands above are bounded examples for the test fixture,
not production sizing or activation instructions. Actual evidence should
retain the report, focused output and source/output hash manifests from the
same commit and run.
