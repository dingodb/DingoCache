# dfkv — distributed KV cache for LLM inference (SGLang · LMCache · vLLM)

[![CI](https://github.com/dingodb/DingoCache/actions/workflows/ci.yml/badge.svg)](https://github.com/dingodb/DingoCache/actions/workflows/ci.yml)
[![Release](https://github.com/dingodb/DingoCache/actions/workflows/release.yml/badge.svg)](https://github.com/dingodb/DingoCache/releases)

A small, **self-contained** distributed key-value cache that pools GPU-node NVMe
SSDs into a shared, large-capacity KV pool for LLM inference (e.g. GLM-5.1 / MLA,
DeepSeek-V4), **without any DingoFS / brpc / S3-RADOS dependency** — it runs on
its own (only its built-in MDS + etcd for dynamic membership). It plugs into
three engines through thin adapters over one portable core:

- **SGLang HiCache** as an L3 external KV store (`--hicache-storage-backend dynamic`).
- **LMCache** as a `RemoteConnector`.
- **vLLM** directly as a `KVConnectorBase_V1` (GPUDirect RDMA, no LMCache).

> Origin: extracted from the DingoFS branch `feat/kvcache-sglang`
> (`src/cache/kvclient`). The portable core has zero coupling to DingoFS, so it
> lives here as an independent repo. To instead fuse these semantics into the
> production `dingo-cache` (brpc + MDS), see `docs/INTEGRATION.md`.

## What it is
- **`dfkv_server`** — a cache-node daemon. Disk + bounded eviction,
  **cache-only** (a miss is a clean NotFound; no object-store fallback),
  synchronous durable-visible writes. Supports **multiple NVMe SSDs per node**
  (`--dir d1,d2,d3`, intra-node Ketama). With `--mds`, `--group`, `--id`,
  `--advertise`, `--weight` it registers into the MDS tier. With neither a
  `--store-engine` option nor `DFKV_STORE_ENGINE`, every server/store path
  selects the restart-safe slab backend. `file` remains an explicit
  diagnostic/rollback choice; invalid slab capacity/geometry refuses startup
  rather than falling back. An optional registered **RAM hot tier**
  (`DFKV_RAM_TIER=1`) provides direct cold-read promotion and disk-free
  zero-copy warm GETs — see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- **`dfkv_mds`** — stateless Membership Directory Service daemon. Flags:
  `--listen <port>` and `--etcd <host:port>` (default `127.0.0.1:2379`). The only
  etcd client in the system; holds each node's etcd lease on its behalf. Deploy as
  N replicas — no load-balancer needed; nodes and clients each pick any reachable
  MDS and fail over automatically.
- **`libdfkv.so`** — C ABI client (canonical namespace + object-key hashing,
  Ketama routing, opaque raw-value Put/Get/Exist).
- **`integration/hicache/dfkv_hicache.py`** — SGLang `HiCacheStorage` plugin loaded via
  `--hicache-storage-backend dynamic` (no SGLang fork). MLA: one packed-latent
  object per page, no tp_rank suffix, `backup_skip` (only tp_rank 0 writes).

## Design in one breath
SGLang HiCache (zero-copy `interface_v1`) → `dfkv_hicache.py` (ctypes) →
`libdfkv` v2 client (length-framed namespace/object identity + Ketama route) →
TCP/RDMA → `dfkv_server` → optional RAM hot tier → DiskCacheGroup over N NVMe
(production slab extents; explicit file fallback).
Distributed = client-side consistent hashing; no replication (regenerable KV →
node loss = miss → recompute). Full architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

**Membership and peer topology** are managed by the MDS tier (`dfkv_mds` +
etcd). Nodes register and heartbeat; etcd leases (TTL 30 s) are the liveness
signal. Native clients use the immutable `dfkv_client_options_v2` constructor:
set either static `members` or `mds_endpoints` + `mds_group`. Modern discovery
uses `kListTopology`. Its existing optional `HLT1` membership extension carries
each member's placement eligibility plus repeated
`{device name, port_state, phys_state, query_ok}` rail health; no second rail
protocol is used. Clients independently deduplicate placement changes with
`MembersEpoch` and peer-topology changes with `MembersTopologyEpoch`. The latter
is canonical over stable member-id/device-name order and includes the member's
address/placement binding, eligibility, and all rail health fields, so a rail
transition is forwarded to the transport even when the Ketama ring is
unchanged. Two-layer offline detection remains: etcd lease expiry updates the
placement view, while `PeerHealth` quickly avoids endpoint IO failures. There
is no post-open membership mutation API in v2. For rings still running v1.x
nodes/clients, `DFKV_MDS_ACCEPT_LEGACY=1` enables a dual-protocol shim on the
MDS control plane (ring-level isolation still applies) — see `docs/DEPLOY.md`
§2b.

**RDMA server rail admission** separates immutable startup resources from
runtime eligibility. With no device list, discovery remains `ACTIVE`-only. An
explicit `dfkv_server --rdma-dev` list defines fixed first-occurrence order: a
present `DOWN` rail is still opened, initialized (anchor plus every required
MR), and monitored. Missing/unopenable devices, port/GID-query failure, or any
anchor/MR initialization failure abort startup; startup never shrinks the
configured topology. After initialization, the node is placement-eligible when
**at least one** initialized rail is healthy. Losing one of several rails is
`PARTIAL` and keeps serving; losing the final healthy rail makes the node
`DEGRADED` immediately. Recovery from zero to nonzero healthy rails retains the
`DFKV_RDMA_HEALTH_RECOVERY_SAMPLES` gate (default 3), then rejoins without a
restart as `PARTIAL` or `ACTIVE`. Startup cardinalities, current health, and
bounded metrics are described in [`docs/DEPLOY.md`](docs/DEPLOY.md) and
[`docs/METRICS.md`](docs/METRICS.md).

**Client registration** (who is using dfkv): cache *consumers* (inference
connector instances — vLLM / LMCache / SGLang HiCache) register themselves with
the MDS under a disjoint etcd prefix (`/dfkv/v1/groups/<g>/clients/<id>`) so
they never enter the placement ring. The same lease/heartbeat contract as nodes
applies — a dead connector's key expires after the lease TTL plus etcd
processing delay, not within a strict TTL upper bound; no explicit deregister
or permanent stale key is required. Connectors set
`DFKV_CLIENT_OPT_REGISTER_WITH_MDS`, `client_id`, `client_info`, and heartbeat
fields in `dfkv_client_options_v2` when MDS discovery is used (opt out with
`DFKV_CLIENT_REGISTER=0`).
Observe with `dfkvctl clients --mds <ep,...> --group <g>` or the
`dfkv_mds_group_clients` gauge. Only upgraded clients register, so an empty list
means "none of the current consumers are registered," not "no one is using dfkv."

## Build & test (no GPU / no RDMA needed)
```bash
cmake -S . -B build            # add -DDFKV_STATIC_LIBSTDCXX=ON for portable binaries
cmake --build build -j
ctest --test-dir build --output-on-failure   # C++ gtests + the Python plugin test
```
Artifacts: `build/dfkv_server`, `build/dfkv_mds`, `build/libdfkv.so`.
## Download a release

Tagged releases are published at
[github.com/dingodb/DingoCache/releases](https://github.com/dingodb/DingoCache/releases).
Each release contains:

- `dfkv-<version>-linux-x86_64.tar.gz`: RDMA + io_uring binaries, `libdfkv.so`,
  headers, deployment/observability configs, connector sources, tests and docs;
- `dfkv_common`, `dfkv_connector` (LMCache), and `dfkv_vllm` pure-Python wheels;
- `SHA256SUMS` covering every downloadable payload.

```bash
sha256sum -c SHA256SUMS
tar xzf dfkv-<version>-linux-x86_64.tar.gz
sudo cp -a dfkv-<version>-linux-x86_64/bin/. /usr/local/bin/
sudo cp -a dfkv-<version>-linux-x86_64/lib/. /usr/local/lib/
```

The tag, root `VERSION`, C++ `--version`, and all three wheel versions are one
CI-enforced contract. The release workflow uses a read-only build job and a
separate publish-only job, reruns the full test suite, validates the
Prometheus/Grafana deployment, builds from a digest-pinned Ubuntu 22.04 base
(glibc 2.35), checksums the payloads, then publishes the GitHub Release once.


## Run a cluster
```bash
# 1. Start etcd (one or three nodes, external)

# 2. Start MDS replicas (stateless, any number)
dfkv_mds --listen 9400 --etcd 127.0.0.1:2379

# 3. On each cache node (--mds requires --id and --advertise)
dfkv_server --dir /mnt/disk1/dfkv,/mnt/disk2/dfkv,/mnt/disk3/dfkv \
            --port 12000 --cap 6597069766656 \
            --mds 10.0.0.1:9400,10.0.0.2:9400 \
            --group default --id n1 --advertise 10.0.0.10:12000

# 4. Client: one immutable v2 construction descriptor (recommended MDS mode)
#    dfkv_client_options_v2 o = { ... .mds_endpoints = "10.0.0.1:9400,10.0.0.2:9400",
#                                 .mds_group = "default" };
#    dfkv_client_t c = dfkv_open_v2(&o);
# Static single-node mode sets o.members instead; no mutable follow-up calls.
```

## Observe the cluster
```bash
dfkvctl ring    --mds 10.0.0.1:9400 --group default        # cache nodes + ring share
dfkvctl clients --mds 10.0.0.1:9400 --group default        # inference consumers
dfkvctl stats   --mds 10.0.0.1:9400 --group default        # ring stats + clients=N
dfkvctl stat    --mds 10.0.0.1:9400 --group default --all  # per-node deep-dive
```
Full dfkv CLUSTER deploy runbook (etcd + MDS + systemd units): `docs/DEPLOY.md`.
Per-engine connect/config + client env/config reference (all connectors): `docs/CONNECTORS.md`.

## Layout
```
src/        portable C++ core: common/ (shared types) · utils/ (generic helpers) ·
            transport/ (TCP/RDMA + wire protocol) · cache/ (StoreEngine: file
            KVStore | slab SlabAllocator+DiskSlabStore · RamTier · dfkv_server) ·
            client/ (KV client + C ABI) · mds/ (membership service + dfkv_mds) · tools/ (CLIs)
integration/hicache/  dfkv_hicache.py (SGLang dynamic backend plugin) + dfkv_telemetry/
                      (canonical shared telemetry pkg, vendored by the other connectors)
integration/common/   dfkv_common shared identity and C ABI schema package
integration/lmcache/  dfkv_connector  (LMCache RemoteConnector, ctypes over libdfkv.so)
integration/vllm/     dfkv_vllm       (vLLM KVConnectorBase_V1, GPUDirect RDMA, bypass LMCache)
test/       gtest suites + test/python (unittest + no-torch sglang shim)
docs/       ARCHITECTURE.md (layers · storage engines · RAM hot tier · wire protocol) ·
            CONNECTORS.md (engine connectors: HiCache · vLLM · LMCache + client env/config reference) ·
            DEPLOY.md (dfkv CLUSTER deploy: etcd + MDS + server + systemd) · INTEGRATION.md (fuse into dingo-cache)
            CACHELIB_EVALUATION.md (F25 decision: no CacheLib/Navy backend in v2.0.0)
```

## Engine integrations
- **SGLang HiCache**: `integration/hicache/dfkv_hicache.py` — see `docs/CONNECTORS.md` §2
  (connect/config/use; cluster deploy is `docs/DEPLOY.md`).
- **LMCache**: `integration/lmcache/` (`dfkv_connector`) — see `docs/CONNECTORS.md` §4
  (deploy + design + implementation).
- **vLLM (direct)**: `integration/vllm/` (`dfkv_vllm`) — a `KVConnectorBase_V1`
  connector occupying the same `--kv-transfer-config` slot as `MooncakeStoreConnector`,
  storing/loading KV **directly over GPUDirect RDMA** (no LMCache, no host bounce).
  Pure-Python ctypes over `libdfkv.so`; uses the scatter-gather batch API to coalesce
  per-chunk keys. Validated on H100 + IB with DeepSeek-V4 (multi kv_cache_group / MLA +
  SWA), full cross-restart and cross-DP prefix hit. See `docs/CONNECTORS.md` §3 (config
  reference + recommended settings) and `integration/vllm/README.md`.

## Operability & performance features
- **Slab-first storage** (no flag/env required; `--store-engine=slab` may pin it
  explicitly): a fixed pool of pre-allocated extent files carved into
  deterministic size classes. Sparse `slots.tbl` metadata is rebuilt in bulk
  after restart, preserving cache warmth without directory walks, per-object
  opens, or file-per-block inode pressure. CRC-protected records, dirty-epoch
  cold reset, deferred removal, O_DIRECT payload I/O, and byte-aware eviction
  keep failures fail-closed. `--store-engine=file` is an explicit diagnostic
  fallback, not a silent recovery path or the production default.
- **RAM hot tier** (`DFKV_RAM_TIER=1`, off by default): a pre-registered arena
  fronting the disk — PUT can be write-back or write-around, a cold whole-value
  GET is read directly from NVMe into its final arena slot, and a warm GET is
  served **straight from that slot over RDMA**. The read-promotion path removes
  both the staging-buffer copy and the duplicate follower payload copy. Send
  pins, identity-checked slot handles, and flush backpressure prevent slot reuse
  while disk or network I/O is in flight; `dfkv_ram_*` metrics expose hit rate,
  promotion, and backpressure. See
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §5–6.
- **128-bit native block identity**: SHA-256 of length-framed canonical
  namespace + object-key bytes, truncated to 128 bits for wire/storage identity.
  Connector object keys are self-delimiting binary bytes encoding pool, full
  chunk hash, DP/TP/PCP/DCP/PP coordinates, cache-group/component, and optional
  scatter geometry; namespace remains separate binary identity. C ABI v2 passes
  every key as pointer + exact length (parallel arrays for batch/SG), so embedded
  NUL and non-UTF-8 bytes are preserved. Values remain opaque raw bytes with no
  hidden header or compatibility unwrap.
- **Connection pooling + keep-alive** (TCP_NODELAY): ~250× lower latency vs dial-per-call.
- **Batch APIs** with concurrent fan-out across nodes and local NVMe disks.
- **Connect/IO timeouts + stale-connection retry**: a hung node fails fast, never hangs.
- **Observability** ([docs/METRICS.md](docs/METRICS.md)): opt-in embedded
  Prometheus endpoints plus client snapshots. New unlabeled client process
  counters are `dfkv_rdma_client_peer_topology_updates_total`,
  `dfkv_rdma_client_no_compatible_rail_total`, and
  `dfkv_rdma_client_stale_generation_reaps_total`; bounded retry counters remain
  `dfkv_rdma_client_cross_rail_{retries,retry_successes,retry_exhausted}_total`.
  Server partial health is exposed by `dfkv_server_rdma_rails_configured`,
  `dfkv_rdma_rails_initialized`, `dfkv_server_rdma_rails_active`,
  `dfkv_server_ib_device_healthy{device}`, and
  `dfkv_server_ring_eligible`. RDMA families exist only when the corresponding
  RDMA listener/transport is active; TCP-only absence is not zero and must not
  be coerced to unhealthy. No metrics port means no HTTP listener.
  The three connectors (vLLM / LMCache / SGLang HiCache) can also **push** fleet
  metrics (ops/keys/bytes, op latency, per-peer latency) over OTLP to a central
  Collector → Grafana — opt-in via `DFKV_METRICS_ENABLED=1`, **zero-dependency stdlib
  exporter by default**; see [deploy/observability/CONNECTOR-USAGE.md](deploy/observability/CONNECTOR-USAGE.md)
  and [docs/METRICS.md](docs/METRICS.md) §3.4.
  Health endpoints differ by daemon: `dfkv_server /readyz` means startup
  finished + first MDS registration done + current store/RAM health OK
  (`src/cache/dfkv_server_main.cc:389`); `dfkv_mds /healthz` is pure process
  liveness (it no longer probes etcd, avoiding CrashLoop storms), while
  `dfkv_mds /readyz` runs a TTL-debounced etcd probe
  (`DFKV_MDS_PROBE_CACHE_MS`, default 2500 ms).
- **Dynamic membership**: the immutable `dfkv_client_options_v2` descriptor
  selects MDS discovery at construction; the client polls MDS and atomically
  rebuilds its weighted Ketama ring on etcd-epoch changes. No mutable
  membership or post-open discovery API is exported.
- **CLI tools**: `dfkv_smoke` (roundtrip check), `dfkvctl` — per-node ops
  (`put/get/exist/stat`) plus cluster views: `dfkvctl ring` (membership + ring vnode
  share + each node's **self-reported version/config** — engine, capacity, RAM tier,
  RDMA dev — carried on register/heartbeat, so fleet-wide version/config audit is one
  command, no per-node ssh) and `dfkvctl stat --all` (per-node metrics + aggregate) via MDS.
- **Peer-aware heterogeneous RDMA rails** (gated `-DDFKV_WITH_RDMA=ON`,
  native libibverbs RC): with no device override each host chooses its first
  `ACTIVE` local HCA. `DFKV_RDMA_DEV` fixes stable local device indices.
  `DFKV_RDMA_RAIL_TIERS` optionally groups that list as priority tiers using
  `a|b;c` syntax (leftmost Tier 0, `|` peers within a tier, `;` separates
  tiers); every name must exist in `DFKV_RDMA_DEV`. When absent, all configured
  devices form one homogeneous tier. For each endpoint, selection intersects
  local enabled rails with the peer's HLT1 healthy rails and uses only the
  highest nonempty tier. A configured-tier client fails closed with
  `Status::kNoCompatibleRail` when peer topology is missing/incomplete or the
  intersection is empty. QPs bootstrap over TCP, so the data fabric needs no
  IP. A v2 capability probe is mandatory; once RDMA is requested it never
  switches transports.
- **One-sided zero-copy data plane**: control descriptors/status use small 4-KiB
  SEND/RECV buffers. PUT payloads RDMA-WRITE into leases from one process-wide,
  pre-registered receive segment; GET payloads RDMA-WRITE directly into the
  caller's registered buffer. No connection-sized block buffers or payload copy
  are used.
- **Optional pipelining** (`DFKV_RDMA_DEPTH=K`): K requests in flight per connection.
  A network-latency hider, **not a throughput knob** — GET and PUT are both
  depth-flat (the per-connection serve loop is in-order; benchmarked GET ~1.24 GB/s at
  depth 1 == 32). The throughput levers are **multi-connection fan-out**
  (`batch_concurrency`) and **fewer/larger keys**. See `docs/datapath-perf-notes.md`.
- **Tier-bounded NUMA and congestion selection** (`DFKV_RDMA_NUMA=1`):
  NUMA locality, per-rail credits, latency, quarantine, and recovery select only
  inside the highest compatible tier. A healthy Tier-0 rail with exhausted
  credits waits under the existing bounded backpressure contract; congestion
  never overflows traffic into Tier 1. Lower tiers are considered only when the
  peer-health intersection makes every higher tier unavailable. Server threads
  follow their QP rail NUMA node; the shared receive segment is registered on
  each selected rail but is not separately NUMA-allocated per rail.
- **Generation-fenced endpoint recovery**: each public operation captures one
  peer-topology generation for both of its possible physical attempts.
  Connections record that generation. A stale-generation pool entry is never
  reused or returned to a pool; it is retired through the existing single-owner
  teardown lifecycle. A first-attempt client-local `kRailFailure` retires its
  endpoint before at most one fresh retry, excluding the failed rail when
  another compatible rail is available. The operation never widens beyond two
  physical attempts.
- **Replay contract**: replay-safe GET/batch/SG paths retain byte-exact completed
  items/windows and retry only unfinished work. A PUT may cross rails only when
  failure is known to precede request submission. Once a PUT may have been
  submitted, its outcome is ambiguous and it is returned without cross-rail
  replay; dfkv does not promise exactly-once execution or broaden the existing
  ambiguous-write policy. `kNoCompatibleRail` is client-local and peer-health
  neutral: it neither cools the peer nor counts as a served peer response.
  A current MDS can expose a legacy member's absent HLT1; configured tiers then
  fail closed, while no-tier clients retain homogeneous behavior. A legacy MDS
  does not understand `kListTopology`: modern polls fail and preserve the
  last-good ring, so rollout is MDS-first rather than silent fallback.
- **SGLang HiCache pool-aware v2 interface** (`batch_set_v2`/`batch_get_v2` +
  PoolTransfer) for multi-pool models (Mamba/SWA/DeepSeek-V4).
- **Packaging**: CPack (deb/rpm/tgz) + Dockerfile; **graceful shutdown**; leveled logging.

## Recommended tuning (v2.11)

Validated on a 5-node ring (8×B200 hosts, 6× Gen4 NVMe + 8×400G IB per node,
128 GiB RAM arena): cold read 97 → **156 GB/s**, hot read 48 → **97 GB/s**
single-pair / **147 GB/s** mesh, cold-read p99 0.9–2.2 s → **~50 ms**.
Defaults are safe for a single-fabric node; production overrides below make
fabric selection and capacity explicit.

**Server** (per cache node):

| Knob | Recommended | Why |
|---|---|---|
| `--rdma-dev` | leave unset for one local HCA; list the fabric explicitly for multi-rail | Unset resolves and initializes the first `ACTIVE` local HCA (peer names may differ). An explicit comma list defines a fixed topology in first-occurrence order: every listed device must be present/openable with complete provider metadata, including a successful GID query, and complete anchor/MR initialization; a present `DOWN` entry remains initialized and monitored rather than being rejected. Both hosts still require compatible names/fabric. |
| `DFKV_DISK_HASH_WEIGHT` | `10` | Flattens the intra-server disk ring share from ±20 % to ±3 % so the hottest disk stops gating the whole node (+5–6 % cold read, ~2× lower p99). **Re-routes existing keys** (cache miss + refill) — flip together with a restart/upgrade window. |
| `--rdma-depth` | `4` (default) | Handshake window is `min(client, server)`. Single-connection PUT/GET bandwidth is depth-flat once connected, but production connector batches require both sides to expose the same bounded window; a 1-vs-4 mismatch caused burst PUT failures and a 29.8% hot-round regression on GLM-5.2. Depth 4 with the default 4 MiB declaration leases about 16 MiB per data QP, so a 16 GiB receive segment admits about 1024 QPs. Increase only for a measured latency-bound path and budget `depth × slot_size` per pooled QP. |
| `DFKV_RDMA_IDLE_MS` | `30000` for frequently replaced clients; otherwise default `600000` | Server dead-client reaper. A short interval bounds leaked receive-segment leases, but live clients must set `DFKV_RDMA_KEEPALIVE_MS` below this value or their next GET pays stale-QP recovery. |
| `DFKV_RDMA_HEALTH_RECOVERY_SAMPLES` | `3` | Any healthy initialized rail keeps placement eligible; a partial loss stays online. Loss of the final healthy rail removes the node immediately. Only recovery from zero to nonzero waits the consecutive sample gate, then rejoins as `PARTIAL` or `ACTIVE`. |
| `DFKV_RDMA_HEALTH_FILE` | unset | Diagnostic-only health input (`device port_state phys_state`, one per initialized/resolved rail, including an auto-discovered rail) for controlled fault injection. Production MUST leave this unset so health comes from sysfs. |
| `--ram-tier` / `--ram-tier-bytes` / `--ram-tier-shards` | on / sized to the node / `16` for ≥100 GiB arenas | Large arenas contend on the shard locks under mixed load (+40 % mixed R/W at 16 shards on a 128 GiB arena); small (≤16 GiB) arenas are fine at the default 8. |
| `--store-engine` | `slab` | Index rebuilds on restart; removes file-per-block hazards. |
| `DFKV_TCP_FIRST_REQ_MS` / `DFKV_MDS_FIRST_REQ_MS` / `DFKV_METRICS_FIRST_REQ_MS` | `30000` (default) / `0` = off | First-request deadline per listener: a connection that sends nothing before the deadline is dropped, capping idle pre-auth connections. |
| `DFKV_METRICS_MAX_CONNS` | `64` (default) | Connection cap on the metrics listener; protects the exporter from connection floods. |

**Production client** (inference connectors, one process per TP rank):

| Knob | Recommended | Why |
|---|---|---|
| `DFKV_RDMA_DEV` | leave unset for one local HCA; set each host's actual stable local whitelist for multi-rail | The local list may differ in cardinality on GPU and CPU hosts. Peer-aware selection uses exact shared names from HLT1; configured tiers require an explicit list. |
| `DFKV_RDMA_RAIL_TIERS` | unset for homogeneous hosts; e.g. `mlx5_0|mlx5_1;mlx5_2` for heterogeneous hosts | Every name must be present in `DFKV_RDMA_DEV`; leftmost is highest priority. GPU and CPU hosts may expose different subsets, but interoperating rail names must describe the same fabric. The client uses the highest tier in `local enabled ∩ peer healthy`; missing/incomplete peer HLT1 fails closed only when tiers are configured. Credits, NUMA, latency, and quarantine never cause tier overflow. |
| `DFKV_RDMA_DEPTH` | `4` (default) | Keep client/server defaults aligned for connector batch correctness. Throughput scaling comes from multiple pooled connections, not raising one QP's depth; lowering depth reduces receive-segment consumption only after validating the real engine workload. |
| `DFKV_RDMA_RAIL_ERROR_THRESHOLD` | `3` (default) | Consecutive client-local rail failures required to quarantine that rail; endpoint/peer failures do not contribute. |
| `DFKV_RDMA_RAIL_COOLDOWN_MS` | `5000` (default) | Quarantine duration. No admission occurs during cooldown; afterward exactly one real operation is admitted as the recovery probe. Probe success clears the rail state, local failure starts another cooldown, and endpoint failure proves neither recovery nor a new local fault. |
| local-rail attempts | fixed at `2` (not configurable) | One initial attempt plus at most one fresh retry. GET retries only unfinished work. PUT crosses rails only before request post; ambiguous post-submit failure is returned without replay. Both attempts retain one captured peer generation. |
| `DFKV_RDMA_KEEPALIVE_MS` | `15000` (default); `0` = off | Sends lightweight membership probes over every idle pooled QP. Live clients keep their QPs and avoid first-GET reconnect tails; exited clients send no probes and remain reclaimable. Keep the interval strictly below the server reap interval. |
| `DFKV_RDMA_POOL_MAX` | `16` (default) | Maximum idle QPs retained per server and per lane, across all rails. It is not a process-wide connection cap. Raise only when connection-open metrics grow on repeated steady-state rounds; each retained QP leases `depth × align4K(4096 + declared_max_block)` from the server receive segment. |
| `DFKV_RDMA_ENDPOINT_CACHE_MAX` | autoscaled (floor `256`, process-wide) | Hard cap on live client RDMA endpoints across all `RdmaTransport` instances. Unset, the cap follows the adopted ring: `nodes x (2 x rails + 1) x 1.25`, raised (never shrunk) on every membership adoption, so large rings cannot starve admission into 10 s `kResourceExhausted` timeouts. Setting this env (or any budget env) pins the whole budget and disables autoscaling. Under pressure, the requesting transport applies a SIEVE second-chance scan to its idle data/SG/control endpoints, destroys one cold QP, then retries admission. A hot cache hit sets only one visited bit; no LRU mutation occurs on the datapath. |
| `DFKV_RDMA_QP_BUDGET` | endpoint cap (default) | Process-wide QP reservation limit. Keep equal to the endpoint cap while each endpoint owns one QP. |
| `DFKV_RDMA_WR_BUDGET` | endpoint cap × `DFKV_RDMA_DEPTH` | Process-wide negotiated WR-slot reservation. SG endpoints reserve one slot; data/control endpoints reserve configured depth. |
| `DFKV_RDMA_REGISTERED_BYTES_BUDGET` | endpoint cap × depth × negotiated receive-slot bytes | Bounds aggregate server receive-segment bytes represented by live client endpoints. |
| `DFKV_RDMA_RESOURCE_ACQUIRE_MS` | `10000` | Bounded wait for all endpoint/QP/WR/receive-slot credits before opening a new QP. Timeout fails the shard without oversubscribing resources. |
| `DFKV_FANOUT_THREADS` | unset (default 32) | Bounds TCP batch, RDMA write, and compatibility helpers; RDMA reads use the separate bounded scheduler below. |
| `DFKV_RDMA_READ_WORKERS` | `7` (default) | Process-wide hard cap for active RDMA GET/GET-Auto/SG-GET/Exist shards across every client handle. Concurrent batches advance round-robin. Seven workers replace the legacy executor's seven helpers per rank without adding a thread; unlike the legacy caller-participates path, active read QPs cannot grow with concurrent callers. Raise only when an A/B proves seven active depth-four QPs cannot saturate the target fabric. |

**Read-side convoy collapse and direct promotion (opt-in)** — for MLA + TP-N
inference rings, where every rank is a separate process fetching the SAME page
and NVMe would otherwise execute N identical reads per page. With both knobs
enabled, the io_uring leader reserves the final registered arena slot before
submitting the read; followers wait for publication and then RDMA-send from the
same resident slot. Repeated reads hit zero disk.

| Knob | Recommended | Why |
|---|---|---|
| `DFKV_READ_COALESCE` | `1` | Master switch. Concurrent identical GETs share one disk read. A whole-value io_uring miss that fits the arena reads NVMe directly into a hidden RAM reservation and publishes it born-durable before waking followers: no staging copy, no extra flush, and no follower payload copy. Partial, oversized, or capacity-constrained requests retain the bounded staging fallback. Unset restores independent disk reads. |
| `DFKV_READ_COALESCE_RECUR_MS` | `1000` (default) | Diagnostic recurrence window for staged fallback flights: a lone whole read leaves a bounded key fingerprint and an identical read inside the window increments `recur`. Direct-to-RAM flights create no fingerprint because the resident itself covers the recurrence. `0` disables this diagnostic. |
| `DFKV_READ_COALESCE_TIMEOUT_MS` | `500` (default) | Follower wait bound; a waiter whose leader connection dies falls back to its own read instead of hanging. |

Watch `dfkv_read_coalesce_{leaders,coalesced,timeouts}_total` and
`dfkv_ram_promoted_total`; healthy cold-to-warm runs have `timeouts=0`,
`promoted` rising on first whole-value reads, then `dfkv_ram_hit_total` rising
without disk bytes. Direct promotion requires the RAM tier (`--ram-tier`);
coalescing alone works without it. Note for env-file deployments: the file is
*sourced*, so new knobs must be explicitly **exported** by the start script or
a systemd drop-in to reach the server process.

**Benchmark reproduction** (`dfkv_bench`): set `DFKV_RDMA=1` explicitly (otherwise
the client measures TCP), select the intended 8-rail `DFKV_RDMA_DEV`, keep
`DFKV_RDMA_DEPTH=4`, and use `DFKV_FANOUT_THREADS=256`. The 0064 cold-read
sweet spot was `--threads 16 --batch 8 --bc 1 --size 4194304` per client node.
`--bc 1` prevents nested client fan-out under the external benchmark threads.
Keep `--count` no larger than the seeded key count and let the drives settle
after bulk writes before cold-read A/Bs. GET allocates and registers one bounded,
page-aligned arena up front; registration failure aborts instead of silently
benchmarking ad-hoc MRs. `DIAG` lines report ring/MDS/RDMA availability,
transport errors, MR coverage, oversize rejects, and completion timeouts.
The command exits `1` if any requested operation failed, `2` for setup/CLI
failure, and `0` only when every requested operation succeeded.

## Status
CI gates GCC/Clang builds, C++ and Python contract tests, ThreadSanitizer,
Soft-RoCE RDMA datapath tests when the runner supports RXE, the shipped runtime
container, observability configuration, portable glibc-2.35 artifacts, wheel
contents, version alignment, and release-package smoke imports.
Architecture & design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Rollout: `docs/DEPLOY.md`.
