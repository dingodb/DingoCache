# STO-18 optional ordered IO / prefix observation

## Scope and activation

This is an optional observation layer for reconstructing ordered cache-object and
prefix-chain evidence. It is not a replay cache backend, never stores payload
bytes, and never changes cache results, retries, retention, routes, or schedules.

Both native `KVClient` construction and `DfkvHiCache` construction read
`DFKV_REPLAY_TRACE_PATH` once. No request reads the environment. The path is a
prefix: each writer creates files named
`<prefix>.<role>.<pid>.<capture-id>.g<generation>.jsonl`, where role is `native`
or `connector-prefix`. Native and connector streams therefore have independent
exclusive owners even when configured with the same prefix. Files use mode 0600
and `O_CREAT|O_EXCL` (plus `O_NOFOLLOW` where available); existing files are never
truncated and missing directories are not created. Open, allocation, identity
serialization, record serialization, write, rotation, or sync failures stay at
the observability boundary: recorder entry catches them before they can reach a
cache caller, so they do not terminate the client, alter API results, or become
successful trace claims. Public cache calls are not given a blanket exception
handler, and their normal defensive result paths remain unchanged.

`DFKV_REPLAY_TRACE_MAX_BYTES` selects size rotation (default 64 MiB, 0 disables
rotation). Every rotation closes the previous generation with a `rotation` record
and opens the next with a `start` record. Global `sequence` remains monotonic
across generations. **Sequence is record-publication order, not invocation
order**: overlapping calls may publish in a different order from their starts,
so intervals must be reconstructed from their measured monotonic start/end
timestamps. Batch end time is sampled at recorder entry, immediately after the
public cache call returns and before identity derivation, allocation,
serialization, or file I/O; observation work is not charged to cache execution.
`drop` records expose failed observations and sequence loss; an ordinary clean
close samples capture wall/monotonic end timestamps and ends with `end` after an
fsync. A crash, partial line, missing generation, failed rotation/open, or broken
chain is not complete.

With no path, the native pointer and connector recorder are null. Batch methods do
not take observation timestamps, hash keys, copy SG descriptors, allocate native
status/length arrays, build page metadata, or write files. Native observation is
attached at these public boundaries after real results are available:

- `BatchPut`, `BatchGet`, `BatchGetAuto`, `BatchPutSg`, `BatchGetAutoSg`;
- `BatchExist`, `BatchRetain`, and `BatchRemove`.

The existing bodies perform all actual work. When tracing is enabled and a caller
did not request statuses or lengths, wrapper-local vectors observe those real
results without changing the caller-facing API. Native `input_index` is the index
at the `KVClient` public boundary. A C ABI wrapper can filter invalid arguments
before that boundary; the native record does not invent the pre-filter C ABI
index. The connector prefix event retains its own full logical input order.

## Record envelope

Every JSONL object has `schema_version: 1`, `record_type`, `capture_id`,
`capture_start_epoch_ns`, `capture_start_monotonic_ns`, `sequence`, and
`generation`. Capture start uses the Unix epoch plus a process-local monotonic
boot marker; IO records also carry wall and monotonic start/end as integer ns.

`start.initial_inventory` is `{"state":"unknown","object_count":null}` for the
normal environment-enabled path. Programmatic construction may explicitly set
`{"state":"empty","object_count":0}` only when its caller actually knows that
initial condition. Rotation uses `carried_from_same_capture`; it does not assert a
new inventory. Empty storage at process start is never assumed. `end` includes
`capture_end_epoch_ns`, `capture_end_monotonic_ns`, writer error/drop counts,
expected record count, the unchanged initial-inventory assertion, final fsync
status, and the explicit `fsync_before_end` durability descriptor.

### `batch_io`

Batch fields include:

- `operation`: `put`, `get`, `get_auto`, `exist`, `retain`, or `remove`;
- `namespace_layout_generation`: the 48-hex BlockKey filename of the empty object
  key in this binary namespace, used only as a namespace/layout generation tag;
- `wall_*`, `monotonic_*`, `expected_item_count`, `item_count`, and complete
  `items` in original call order (no first-K truncation).

Each item includes `input_index`, `block_key`, `output_available`, `returned`,
`status`, `status_code`, explicit request/capacity/stored-length fields, and a
`payload.kind`:

- `block_key` is exactly `BlockKey::Filename()`: 16 lower-case hex tenant-hash
  chars followed by 32 lower-case hex object-digest chars. No `PernodeKeyFp` or
  process-random HMAC is used.
- `status` / `status_code` are the actual `DFKV_ST_*` values for PUT/GET/SG.
  `UNKNOWN_ITEM_STATUS` with its numeric code is not accepted as replay complete.
  The public EXIST and REMOVE APIs return only booleans, so status is explicitly
  null rather than a fabricated OK/miss classification. REMOVE `returned` means
  owner confirmation, not proof that an object existed.
- `BatchRetain` uses nullable `retain_result` / `retain_result_code` with actual
  `DFKV_RET_*`, plus `retention.{mode,mode_code,ttl_s,no_debounce,min_refresh_ms}`.
  It distinguishes SET and REFRESH, including the zero-TTL REFRESH used by
  connector existence probes.
- PUT `requested_length` is the requested byte count. Exact GET exposes only
  `capacity`; `stored_length` is null with `not_returned`. Auto GET reports
  `actual_stored_length` only for a successful item with a returned length. A
  miss, short result, or unavailable output has null, never a zero masquerading
  as an actual stored length.
- `contiguous_put`, `contiguous_get`, and `key_only` need no invented segment
  geometry. SG records include actual segment sizes/capacities, pointer/count
  descriptor counts (counts, never addresses), total-sum validity, and
  `payload_form: scatter_gather`. Memory address space (host/device) is not
  exposed by these batch items and remains unknown rather than guessed.

### `prefix_exists`

`DfkvHiCache.batch_exists()` records every ordered logical page hash and every
native key actually probed for that page, with aligned presence outputs. These
`present` values are explicitly `connector_boolean_result_not_dfkv_status` and
`dfkv_status` is null: a false connector result is not promoted to a confirmed
server miss or eviction verdict. Binary
page hash bytes appear as `logical_page_hash_hex` from the existing
`pool_key_hash_field()`; they are not token IDs or prompt text. Native identities
use the same canonical namespace/object bytes as production `pool_key()`,
`pool_key_affixes()`, and `sg_key()` followed by the native BlockKey mapping.

Each native item has explicit `storage_kind` (`pool` or `sg_group`), producer
component and component index, and SG group/width where used. The event includes
real DP/TP/PCP/storage-PCP/DCP/PP/physical-rank coordinates present in the
connector. `probe_operation` distinguishes `exist` from `retain_refresh`.
`batch_exists_v2` emits its nested primary event and per-side-pool events. Side
pool events cover exactly the already-known KV prefix actually probed; an
unprobed tail is not allocated a speculative native key or result. Actual pool
policy, judged state, boundary, final usable prefix, and later REFRESH are kept
separate. Native REFRESH per-item codes remain in the native batch stream.

`parent` is always explicitly `{"state":"unknown","reason":"not_supplied"}` in
this slice. Disjoint calls are not stitched by time or key position; no fabricated
root, empty prefix relationship, or historical parent is emitted. Events retain
the exact ordered path itself, from which later analysis can prove relationships
that are actually present.

## Data protection

Records intentionally omit raw namespace/model/tenant/model-revision strings,
raw token IDs, prompt text, source object-key bytes, route anchors, pointer
values, and payload bytes. BlockKey filenames and logical page hashes are
**sensitive pseudonymous identities**, not anonymous data: they can be correlated
across requests/ranks and may be attacked with candidate inputs. Protect trace
files as production diagnostic data, restrict access, and define retention and
deletion policy separately from cache retention. The digest is not an erasure or
anonymization claim.

## Validation and non-eviction outcomes

Use `dfkv_common.replay_trace.validate_replay_trace(paths)` with every generation
of one capture. `complete_for_replay` requires structured parseability, one
capture, complete contiguous sequence coverage, start/rotation/end consistency,
finite ordered times, original item/page order and counts, supported
status/payload descriptors, no drops or writer I/O failures, and an explicitly
empty initial inventory. Unknown initial inventory alone prevents this claim.
The validator requires a unique nonempty string capture ID, exact integer
schema/types, stable capture-start timestamps, generation-0 capture start,
rotation immediately followed by the next generation's carried-inventory start,
and a final active-generation end with a zero-error fsync descriptor. It also
checks operation-specific result/length/RETAIN/SG matrices and real boolean
prefix probe outcomes. Missing end, partial lines, duplicate JSON keys,
nonfinite numbers (including overflow such as `1e400`), excessive recursion,
malformed inventory, unavailable probe outcomes, unknown status values,
SG/schema errors, file-level sequence reordering, missing rotations, and sequence
gaps all become structured validation failures rather than uncaught parser or
validator exceptions.

`DFKV_ST_BAD_ARGUMENT`, `DFKV_ST_PEER_COOLING`, `DFKV_ST_NO_ROUTE`, I/O, rail,
resource, and peer errors are preserved outcomes/diagnostics. Their presence can
be part of a complete observation stream, but consumers must not turn them into
eviction evidence. `complete_for_replay` means integrity of this finite stream,
not proof of empty production storage, proof that a miss was eviction, or
completion of a live production acceptance run.

Current limitations are explicit: native EXIST/REMOVE have no public per-item
`DFKV_ST_*`; connector per-item RETAIN codes are in the native stream; host/device
memory address kind is unavailable at the native item boundary; historical
parents and live inventory are not discovered; routing/membership generations are
outside this slice; and no actual production capture/replay verdict is asserted.
