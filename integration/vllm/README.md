# dfkv connector for vLLM

A direct vLLM `KVConnectorBase_V1` connector (`DfkvStoreConnector`) that stores
and loads KV cache to/from a **dfkv** cluster over GPUDirect RDMA, bypassing
LMCache. It is the dfkv analogue of vLLM's `MooncakeStoreConnector`: both
producer and consumer read/write KV to a shared pool, enabling prefix-cache
reuse across requests and instances.

The connector is pure Python (ctypes over `libdfkv.so`); there is no native
build. It talks to dfkv on **raw GPU device pointers** — the paged KV cache is
registered once via `dfkv_register_memory` (an `ibv_reg_mr` that, under
nvidia-peermem, yields a GPUDirect MR). Only return code `0` is accepted;
registration failure aborts startup rather than issuing I/O with an unregistered
pointer. Transfers then run directly between RDMA and GPU memory with no host
bounce. Each logical chunk is one dfkv object whose complete ordered GPU
segment vector is passed to the **scatter-gather** batch API. When the vector
exceeds one HCA WR's SGE limit, libdfkv posts ordered bounded WR windows under
that single object operation; it does not split the chunk into physical keys.

> **Full deployment walkthrough + recommended settings: [`docs/CONNECTORS.md`](../../docs/CONNECTORS.md) §3.**
> This README is the quick reference.

## Enable

```
--kv-transfer-config '{
  "kv_connector": "DfkvStoreConnector",
  "kv_connector_module_path": "dfkv_vllm.connector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "members": "c1=<server-ip>:<rdma-port>",
    "lib": "/path/to/libdfkv.so",
    "rail_affinity": true,
    "rail_affinity_fallbacks": 1
  }
}'
```

`DfkvStoreConnector` is RDMA-only because every data method receives raw GPU
device pointers. Set `DFKV_RDMA=1` in every engine process. Construction rejects
and closes any native handle whose reported transport is not `rdma`; there is no
TCP or host-bounce fallback for this connector.

## Environment variables (engine process)

Read by `libdfkv.so` (the C client) and the connector, so set them in **every**
vLLM engine process — each DP rank is its own process.
The connector requires vLLM's content-defined block hashing and a stable root:
set `--prefix-caching-hash-algo sha256` and the same fixed
`PYTHONHASHSEED` in every engine process. Current vLLM releases initialize the
first block's parent hash from `os.urandom()` when the seed is unset, which
makes every restart miss all previously stored keys. Startup fails before
traffic if either requirement is missing.


| env | default | meaning |
|---|---|---|
| `DFKV_RDMA` | **required: `1`** | Selects the required GPUDirect RDMA transport. Unset/TCP is rejected during connector construction; there is no TCP fallback. |
| `PYTHONHASHSEED` | **required: fixed value** | Stabilizes vLLM's root block hash across processes and restarts. Use the same value (for example `0`) on every producer and consumer sharing a store. |
| `DFKV_RDMA_DEV` | first local `ACTIVE` HCA | Optional ordered device/fabric list. With `rail_affinity=true`, the connector narrows this full per-host list to the worker's world-group local-rank primary plus bounded fallbacks before native open. |
| `DFKV_RDMA_DEPTH` | `4` ceiling | Scalar QPs open at depth1; batches select the smallest sufficient power-of-two depth up to this ceiling and the server cap. |
| `DFKV_RDMA_NUMA` | `0` | `1` pins buffers/threads to the rail's NUMA node and picks a NUMA-local rail per connection. Optional. |
| `DFKV_RDMA_CONNECTION_MIN_BLOCK_BYTES` | `256 KiB` | Minimum adaptive block class; logical max remains `DFKV_RDMA_MAX_BLOCK_BYTES`. |
| `DFKV_CONNECTOR_CLIENT_RANKS` | unset=`TP` | `auto` or `N` converges fully TP-replicated MLA stores to N evenly spread ranks. Sharded DCP/PCP layouts clamp to TP. Converged loads automatically enable native GPU same-host dedup so one rank performs each remote GET and CUDA IPC publishes to followers. |
| `DFKV_CONNECTOR_CLIENT_ELIDE` | auto in converged mode | Producer non-participants skip eager native-client creation. Explicit `0` disables elision without disabling load convergence. |
| `DFKV_NODE_DEDUP_GPU_ARENA_MB` | `512` | Per-rank GPU rendezvous arena used by converged loads; reduce only after the largest concurrent logical object/window fits. |
| `DFKV_READ_SHARD_KEYS` | `16` | Target keys per read shard: splits one node's batched GET into parallel shards. The real read-throughput lever on few-node rings / large batches landing on one node (single connection drains ~166 MB/s serially); no-op on wide rings. |
| `DFKV_READ_MAX_CONNS` | `8` | Per-node cap on concurrent read-shard connections (pairs with the above; `1` disables sharding). |
| `DFKV_FANOUT_THREADS` | `32` | Client batch-op fan-out pool cap (clamped [1,1024]). Raise when callers × node-groups ≫ 32, or batch calls degrade to caller-serial and per-call latency grows from max(group) to sum(group). |
| `DFKV_LIB` / `DFKV_BUILD` | — | `libdfkv.so` path (overridden by the `lib` extra-config key). |
| `DFKV_ACCESS_LOG_ENABLED` | `0` | `1` turns on the per-op access log (one line per dfkv client op: `batch_get_auto_sg`/`batch_put_sg`/`batch_exist`/…). Off ⇒ ~100 ns/call no-op; on ⇒ async (background thread), ~µs on the hot path. |
| `DFKV_ACCESS_LOG_PATH` | (stderr) | access-log file path; empty ⇒ stderr. |
| `DFKV_ACCESS_LOG_THRESHOLD_US` | `0` | only log ops slower than this many µs (`0` = log every call). Set e.g. `1000` to surface only ≥1 ms ops. |
| `DFKV_CLIENT_STATS_POLL_S` | `15` | cadence (seconds) of the background poller that mirrors the C client's ring/MDS health onto Prometheus (`vllm:dfkv_client_ring_members`, `vllm:dfkv_client_mds_reachable`, `vllm:dfkv_client_mds_unreachable_polls_total`, `vllm:dfkv_client_transport_info`). Sleeping daemon thread, off the request path. `0` disables. |

The access log shares the same env vars and line format as the dfkv HiCache /
LMCache connector access logs, so one setting covers every integration. Format:
`<op>(<args>) : <result> <duration_s>`, e.g.
`batch_get_auto_sg(20 keys, 1240 segments) : hits=20/20, 1310720 bytes <0.007234>`.

## `kv_connector_extra_config` keys

| key | default | meaning |
|---|---|---|
| `members` | (required) | dfkv member string. **The port MUST be the server's `--rdma-port`** (the RDMA bootstrap listener), not the main `--port`, when RDMA is enabled. |
| `lib` | env `DFKV_LIB` / `$DFKV_BUILD/libdfkv.so` | path to `libdfkv.so`. |
| `batch_concurrency` | `0`=auto | client fan-out for batch ops; the real throughput lever (depth is flat). Auto = `min(max(nodes, 8), 32)`: 8-way parallel on single-node, one-per-node on multi-node. Set >0 to pin a fixed value. |
| `rail_affinity` | `False` | Bind each vLLM worker process to a primary rail selected by world-group local rank; requires an ordered multi-rail `DFKV_RDMA_DEV`. |
| `rail_affinity_fallbacks` | `1` | Number of ordered neighboring fallback rails when affinity is enabled. `0` keeps strict one-rank/one-rail; values above the available rail count are bounded. |
| `load_async` | `True` | `True` returns `WAITING_FOR_REMOTE_KVS` and overlaps GPUDirect loads with unrelated model work. `False` performs each requested load synchronously in `start_load_kv`, before the forward pass. Use `False` for hybrid state-cache models when the engine cannot guarantee that remote writes target blocks disjoint from concurrent compute. |
| `transfer_queue_capacity` | `256` | Maximum queued requests in each direction (`1..65536`). All receive workers consume one shared receive queue of this capacity; capacity is not multiplied by `recv_workers`. Submission is non-blocking: a full queue rejects new saves as completed (releasing finish/free fences) and rejects new loads as load errors (forcing recompute), so overload cannot grow memory or pin blocks indefinitely. Invalid or out-of-range values abort connector construction. |
| `recv_workers` | `1` | Receive/load worker count (`1..32`). Workers consume the shared bounded receive queue and may execute independent native GETs concurrently. Invalid, boolean, or out-of-range values abort connector construction. |
| `load_window_keys` | `0` (disabled) | Maximum keys per native GET window (`0..65536`). Use a value whose worst-case result bytes fit inside the node-dedup GPU arena. Windowing lets follower ranks consume published results before the dedup wait deadline instead of re-fetching a large replicated-MLA batch. |
| `load_window_min_keys` | `0` | Apply `load_window_keys` only when the request contains at least this many keys (`0..65536`). Set a threshold to keep small, latency-sensitive loads as one native GET while windowing long-context loads. Has no effect when `load_window_keys=0`. |
| `enable_cross_layers_blocks` | `False` | opt-in for engines whose paged layout interleaves layers within a block. Leave `False` unless you know the layout needs it. |
| `lookup_rpc_port` | (ipc auto) | port for the rank-0 scheduler-side prefix-lookup RPC; set only if the default IPC socket name collides. |

Transfer threads are shut down through vLLM's connector `shutdown()` lifecycle
hook. Shutdown stops admission, cancels queued transfers with the same visible
save/load outcomes described above, waits for active native operations, joins
the send thread and every receive worker, and closes native dfkv clients exactly
once. Accepted work can also be drained explicitly by the thread-level
`stop(cancel_pending=False)` path; normal framework teardown uses cancellation
to avoid extending engine shutdown behind an accumulated queue.

Start with `recv_workers=1`. Increase it only in controlled, byte-identical hot
rounds when `vllm:dfkv_receive_queue_wait_time_seconds` and
`vllm:dfkv_receive_queue_depth` show sustained receive-side queuing while native
GET latency and the dfkv service still have headroom. Confirm actual parallelism
with `vllm:dfkv_receive_active_workers`; all three are Prometheus histograms
(with the standard `_bucket`, `_sum`, and `_count` series). Compare TTFT,
queue-wait quantiles, active-worker samples, dfkv GET latency, and failed/recompute
counts at each setting rather than assuming more workers are faster. Startup
must contain the evidence line
`dfkv transfer queues: capacity=<N> per direction, recv_workers=<N>,
load_window_keys=<N>, load_window_min_keys=<N>, overload=reject-new,
shutdown=cancel-pending`; capture it together with the before/after Prometheus
snapshots.

For same-host replicated loads, a single native GET publishes deduplicated GPU
results only after its storage fetch completes. If a large batch takes longer
than `DFKV_NODE_DEDUP_WAIT_MS`, follower ranks time out and independently
re-fetch it. Set `load_window_keys` so one window completes within that deadline
and its result bytes fit `DFKV_NODE_DEDUP_GPU_ARENA_MB`. Then set
`load_window_min_keys` above ordinary request sizes to avoid adding native-call
overhead to short loads. Validate with `DFKV_CLIENT_NODE_DEDUP_LOG=1`: the
windowed long-context path should report zero `fallback` and aggregate
`fetched` counts near one logical copy, not one copy per TP rank.

### Choosing load-window values

Keep both settings at `0` unless same-host node-dedup logs show follower
`fallback` or aggregate `fetched` counts approaching `TP size × logical keys`.
Windowing adds native-call overhead and does not improve a load that already
publishes before the follower deadline.

Treat the values as **key counts, not token counts**. Derive batch key-count and
result-byte distributions from representative `batch_get_auto_sg` access logs;
use connector geometry or a conservative maximum for per-key sizing. Repeat
this after changing model, KV dtype, block size, cache-group geometry, or TP/DCP
layout.

Choose `load_window_keys` with both constraints below:
1. **Arena bound.** Choose a conservative `sizing_bytes_per_key` (p99 or the
   geometry maximum). As an operational starting point, keep one window below
   75% of `DFKV_NODE_DEDUP_GPU_ARENA_MB`:
   `window_keys <= floor(0.75 × arena_bytes / sizing_bytes_per_key)`.
2. **Deadline bound.** The p99 fetch-and-publish time of one window should be
   comfortably below `DFKV_NODE_DEDUP_WAIT_MS` (target at most 50–70%). A
   larger wait is not a substitute for a batch whose publication is too large.

Start from the smaller bound, round down to a convenient value such as 32, 64,
or 128, then increase one step at a time only while `fallback=0`, aggregate
`fetched` stays near one logical copy, and TTFT improves. Smaller windows reduce
publication latency but add calls; larger windows reduce call overhead but can
overflow/lap the arena or cross the wait deadline.

Choose `load_window_min_keys` only after separating the workload classes:

- set it strictly above the p99 key count of latency-sensitive short loads;
- keep it at or below the smallest long load that must be windowed;
- if those ranges overlap, use separate engine pools or accept a measured
  tradeoff rather than hiding it with an arbitrary threshold.

When `load_window_keys=0`, the minimum has no effect. A batch also remains one
native GET when it is below `load_window_min_keys` or no larger than
`load_window_keys`; consequently, a threshold only changes behavior when it is
larger than the window.

Example from one GLM-5.3 TP8 deployment (not a portable default): the per-key
result size used for sizing was about 3.05 MB and the dedup arena was 512 MiB.
A 128-key window was about 391 MB, completed inside a 1,000 ms wait, and
reduced a 15,624-key load from about eight remote copies to one.
`load_window_min_keys=4096` kept
approximately 1,024-key short loads on the single-GET path:

```json
{
  "recv_workers": 2,
  "load_window_keys": 128,
  "load_window_min_keys": 4096,
  "load_async": true
}
```

Production acceptance requires two byte-identical hot tests:

- **long path:** access logs show batches no larger than the window, zero
  `fallback`, aggregate `fetched` near logical keys rather than `TP × keys`,
  and no failed/recomputed keys;
- **short path:** access logs show one native GET and throughput/TTFT do not
  regress beyond the deployment's acceptance threshold.

## Reproducible external-cache benchmark

`test/python/vllm_external_cache_benchmark.py` never resets caches, deploys
software, or runs local/remote version-discovery commands. Supply immutable
deployment identity explicitly:

| CLI option | summary/corpus key | default |
|---|---|---|
| `--vllm-version` | `vllm_version` | `null` |
| `--vllm-commit` | `vllm_commit` | `null` |
| `--dfkv-version` | `dfkv_version` | `null` |
| `--dfkv-build-commit` | `dfkv_build_commit` | `null` |
| `--connector-layout` | `connector_layout` | `null` |
| `--connector-version` | `connector_version` | `null` |
| `--model-revision` | `model_revision` | `null` |
| `--deployment-manifest-hash` | `deployment_manifest_hash` | `null` |

Values are stored verbatim in the fixed-schema `deployment_identity` object;
the required `--model` is stored as its `model` member. Provide an immutable
manifest digest, not a manifest path or manifest contents. Do not put
credentials, URLs, or host paths in identity values. Summary files store
endpoint hashes and artifact basenames rather than endpoint credentials or host
paths.

For a populate/restart/hot measurement, generate the corpus once, preserve the
external cache, restart only the intended vLLM deployment, then reuse the corpus
with `--corpus-file`. Keep model, model revision, prefix target, request count,
seed, concurrency, and generation settings fixed. The corpus checksum proves
byte-identical prompts; corpus reuse rejects a different model or model
revision. Pass the populate summary as
`--require-identity-summary FILE` when the compared rounds must use exactly the
same model, vLLM, dfkv, connector, layout, and deployment-manifest identity.
This check runs before metrics collection or inference requests, and the new
summary records the reference summary's SHA-256. Capture vLLM and dfkv
Prometheus endpoints with the respective repeatable
`--vllm-metrics-endpoint` and `--dfkv-metrics-endpoint` options so every round
contains before/after metric evidence.

## Identity and raw-value contract

`model_name` is the exact vLLM `model_config.model`, not an extra-config key.
The binary namespace and every object key bind it to the source-controlled
`vllm-multiwr-v2` storage-layout ID; operator-supplied aliases are rejected.

Object keys are self-delimiting binary bytes: `DFKVPOOL\x02`, uint32-LE
length-framed pool and full page hash, fixed `(uint32 size, int32 rank)` pairs
for DP/TP/PCP/DCP/PP, uint32 KV-cache group, and the length-framed
`vllm-multiwr-v2` component. There is exactly one key per logical chunk,
independent of its GPU segment count or the negotiated HCA `max_sge`. Every
native operation receives a pointer plus its exact uint64 length (parallel
pointer/length arrays for batch and SG); no C-string, decode/re-encode,
delimiter, Python-hash, or legacy ABI path exists. The KV-cache group field is
the semantic vLLM hybrid `group_id`, never a WR-window or sibling index. dfkv
stores only the raw GPU segments in canonical layer-name/run/physical-block
order and returns their total length separately; it has no geometry/dtype
envelope. A read is accepted only when the object hits and its returned length
exactly equals the complete destination-vector capacity; otherwise that logical
chunk is failed closed and recomputed.

`multiwr-v2` is a clean cutover from the former `sg-v1` physical-key layout.
Old objects are cold misses by namespace/key identity: readers do not probe,
assemble, or clean up v1 scatter-group siblings. Roll producers and consumers
together; expect a cold external cache after rollout. Do not mix connector
versions in one deployment.

Different namespace/key bytes are a cold miss. The same namespace+key with a
different dtype, page/block size, shape, layer order, or KV memory layout is a
type-safety violation, not a guarded miss. Such changes require another
source-controlled layout-ID bump and coordinated writer/reader deployment.
Cross-runtime sharing requires both identical object keys and byte-compatible
payload layouts.

## Gotchas (validated on hd04 H100 + IB)

- **member port = rdma-port.** Pointing at the main `--port` makes every RDMA
  `put` fail (`rc=-1`); RDMA QP bootstrap listens on `--rdma-port`.
- **Use `--prefix-caching-hash-algo sha256` and a fixed `PYTHONHASHSEED` on
  every engine.** Without the seed, current vLLM releases derive the first
  block's parent hash from `os.urandom()`, so every restart silently turns all
  previously stored entries into cold misses. The connector rejects both
  unsafe configurations before traffic.
- **Identity mismatches are cold misses.** If a read hits but the decoded
  shape/content is wrong, stop mixed writers: the same namespace+key has been
  reused for incompatible raw layouts. Bump the source-controlled layout ID.
- **DCP (`--decode-context-parallel-size > 1`) needs client >= v1.10.0**.
  Older builds run the put_step stride under the replicated-KV assumption and
  store only 1/dcp_size of each rank's shard, so external prefix hits
  collapse to ~1/dcp (field-reported "low dfkv hit rate with DCP"). v1.10.0
  (#70) shrinks the stride by dcp_size; coverage under the default
  token-level interleave is guarded by
  `tests/test_dcp_lookup_geometry.py`.
- **GPUDirect needs nvidia-peermem loaded** on the GPU node (`lsmod | grep
  nvidia_peermem`).
- **First request per DP rank pays a one-time ~2s Triton JIT** (resumed-prefill +
  SWA-index kernels). Warm it with a synthetic hit per rank at startup if first-token
  latency matters.
