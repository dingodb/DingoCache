# dfkv connector for vLLM

A direct vLLM `KVConnectorBase_V1` connector (`DfkvStoreConnector`) that stores
and loads KV cache to/from a **dfkv** cluster over GPUDirect RDMA, bypassing
LMCache. It is the dfkv analogue of vLLM's `MooncakeStoreConnector`: both
producer and consumer read/write KV to a shared pool, enabling prefix-cache
reuse across requests and instances.

The connector is pure Python (ctypes over `libdfkv.so`); there is no native
build. It talks to dfkv on **raw GPU device pointers** — the paged KV cache is
registered once via `dfkv_register_memory` (an `ibv_reg_mr` that, under
nvidia-peermem, yields a GPUDirect MR), and transfers RDMA directly to/from GPU
memory with no host bounce. Per-chunk layer segments are coalesced into one dfkv
key via the **scatter-gather** batch API (one multi-SGE RDMA per chunk instead of
one per layer-segment), cutting key/disk-read count ~20x.

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
    "model_hash": "<uint64>",
    "lib": "/path/to/libdfkv.so"
  }
}'
```

## Environment variables (engine process)

Read by `libdfkv.so` (the C client) and the connector, so set them in **every**
vLLM engine process — each DP rank is its own process.

| env | default | meaning |
|---|---|---|
| `DFKV_RDMA` | unset (TCP) | `1` selects the RDMA transport; unset ⇒ TCP fallback. |
| `DFKV_RDMA_DEV` | — | RDMA rail by name (`ib7s400p0`; comma-list = multi-rail). Required when `DFKV_RDMA=1`. |
| **`PYTHONHASHSEED`** | unset | **Set to `0` (any fixed value, identical on every rank and instance).** vLLM's block hashes feed the dfkv key and use Python `hash()`, which is per-process randomized by default. Without a fixed seed, DP ranks and restarts compute **different keys for the same tokens** ⇒ cross-process / cross-restart prefix reuse silently drops to ~0. The single most common "no hit" misconfig. |
| `DFKV_RDMA_DEPTH` | `1` | Requests in flight per connection. A latency hider, **not** a throughput knob (GET/PUT are depth-flat — the per-connection serve loop is in-order). Leave at default. |
| `DFKV_RDMA_NUMA` | `0` | `1` pins buffers/threads to the rail's NUMA node and picks a NUMA-local rail per connection. Optional. |
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
`batch_get_auto_sg(20 keys) : hits=20/20, 1310720 bytes <0.007234>`.

## `kv_connector_extra_config` keys

| key | default | meaning |
|---|---|---|
| `members` | (required) | dfkv member string. **The port MUST be the server's `--rdma-port`** (the RDMA bootstrap listener), not the main `--port`, when RDMA is enabled. |
| `model_hash` | `0` | uint64 namespace for keys; isolates this model's KV from others sharing the pool. A shared `model_hash` requires identical kv-cache-dtype / page-size / layout (see geometry guard below). |
| `lib` | env `DFKV_LIB` / `$DFKV_BUILD/libdfkv.so` | path to `libdfkv.so`. |
| `batch_concurrency` | `8` | client fan-out across nodes for batch ops; the real throughput lever (depth is flat). Raise toward the cluster's node count for wider pools. |
| `load_async` | `True` | async KV load: the scheduler returns `WAITING_FOR_REMOTE_KVS` and the load runs off the critical path. Keep `True`. |
| `enable_cross_layers_blocks` | `False` | opt-in for engines whose paged layout interleaves layers within a block. Leave `False` unless you know the layout needs it. |
| `lookup_rpc_port` | (ipc auto) | port for the rank-0 scheduler-side prefix-lookup RPC; set only if the default IPC socket name collides. |

## Geometry guard (instances sharing one pool)

KV written by instance A is only safe to read by instance B when their tensor
geometry matches. Before sharing a `model_hash` across instances, confirm all of:
**`--kv-cache-dtype`, page/block size, KV memory layout, `--max-model-len`** are
identical. The dfkv value header guards `payload_len` (byte size) only — same-size
but different-layout KV would be read back as **silent garbage**. To isolate
instead of share, give each a distinct `served-model-name` / `model_hash`.

## Gotchas (validated on hd04 H100 + IB)

- **member port = rdma-port.** Pointing at the main `--port` makes every RDMA
  `put` fail (`rc=-1`); RDMA QP bootstrap listens on `--rdma-port`.
- **`PYTHONHASHSEED=0` on every rank** (see env table) — the #1 cause of "writes
  succeed but reads never hit".
- **GPU buffers use the batch path only.** The single `dfkv_get_auto` computes a
  CRC over the destination on the CPU and segfaults on device memory; the batch
  `dfkv_batch_get_auto` is zero-copy and GPU-safe.
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
