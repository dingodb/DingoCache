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

**Membership** is managed by the MDS tier (`dfkv_mds` + etcd). Nodes register
with the MDS on startup and send periodic heartbeats; etcd leases (TTL 30 s)
are the liveness signal. Native clients use the single immutable
`dfkv_client_options_v2` constructor: set either static `members` or
`mds_endpoints` + `mds_group`. MDS mode polls the directory and rebuilds the
weighted consistent-hash ring whenever its epoch advances. Two-layer offline
detection: **layer-2** — etcd lease expiry → MDS view changes → client epoch →
ring rebuild (authoritative removal after the 30 s lease TTL plus etcd, MDS RPC,
and client-polling delay; this is not a 30 s upper bound); **layer-1** —
`PeerHealth` fast avoidance short-circuits transport failures to misses during
a cooldown. There is no post-open membership mutation API in v2. For rings still running v1.x
nodes/clients, `DFKV_MDS_ACCEPT_LEGACY=1` enables a dual-protocol shim on the
MDS control plane (ring-level isolation still applies) — see
`docs/DEPLOY.md` §2b.

**RDMA server rail admission** is deliberately asymmetric. With no device
list, automatic discovery remains `ACTIVE`-only. An explicit
`dfkv_server --rdma-dev` list instead defines a fixed topology in configured
first-occurrence order: a present but `DOWN` rail is still opened, initialized
(anchor plus every required MR), and monitored. A missing/unopenable device,
port- or GID-query failure, or any anchor/MR initialization failure aborts startup.
The daemon may start with an initialized `DOWN` rail, but the whole node stays
out of the placement ring until **every initialized/resolved rail**, including
the auto-discovered rail when no list is configured, is `ACTIVE`/`LinkUp` for
the recovery sample streak. The default is three successful sampling
opportunities. At the nominal 10 s registrar cadence this usually takes about
20–30 s depending on phase, plus registrar connection/RPC delays, and can take
longer while MDS is unreachable; it is not a wall-clock upper bound. The node
then rejoins without a restart. This fail-closed policy trades the node's entire
cache capacity while any one resolved rail is unhealthy for immutable,
index-aligned multi-rail topology. Startup
cardinalities and current health are described in
[`docs/DEPLOY.md`](docs/DEPLOY.md) and
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
- **Observability** ([docs/METRICS.md](docs/METRICS.md)): opt-in embedded Prometheus
  `/metrics` on `dfkv_server` and `dfkv_mds` (`--metrics-port`); sampled op-latency
  histogram, eviction/error/per-disk/RDMA counters server-side; client-side counters
  (peer health, IO errors, per-rail quarantine/recovery,
  `dfkv_rdma_client_cross_rail_retries_total`,
  `dfkv_rdma_client_cross_rail_retry_successes_total`, and
  `dfkv_rdma_client_cross_rail_retry_exhausted_total`) via
  `dfkv_stats_snapshot` + a plugin poller. Client RDMA families exist only
  when the RDMA transport is in use and that snapshot is exported; absence from
  a TCP-only process is expected. Server `dfkv_server_ring_eligible` and
  `dfkv_server_ib_device_healthy` families are emitted only by an RDMA-enabled
  binary running an RDMA listener. Their absence from a TCP-only binary or an
  RDMA build without a listener must not be coerced to zero by dashboards or
  alerts. **Opt-in and
  off the datapath** — no `--metrics-port` ⇒ no metrics listener, behavior unchanged.
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
- **RDMA transport v2** (gated `-DDFKV_WITH_RDMA=ON`, native libibverbs RC):
  with no device override each host chooses its first `ACTIVE` local HCA and
  sends an empty device selector to its peer. An explicit comma whitelist opts
  into multi-rail round-robin and therefore must name the intended fabric on
  both hosts. Device names are limited to 18 bytes in the wire dev frame —
  longer names fail fast instead of silently truncating (7837f0b). QPs bootstrap over a tiny TCP channel, so the data fabric needs no
  IP. A v2 capability probe is mandatory; protocol, QP, receive-segment, or
  registration mismatch rejects the connection. Unset `DFKV_RDMA` selects TCP;
  once RDMA is requested it never switches transports.
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
- **NUMA-aware rail selection** (`DFKV_RDMA_NUMA=1`): with an explicit
  multi-rail whitelist, prefers an `ACTIVE` rail local to the calling thread,
  then round-robins the listed healthy rails; when every NUMA-local rail is
  inadmissible, it degrades to the full enabled rail set instead of refusing
  traffic (5ef39f1). Server threads follow their QP's
  rail NUMA node; the single shared receive segment is registered on each
  selected rail but is not separately NUMA-allocated per rail. Off by default;
  vendor-neutral (sysfs + `sched_getcpu`, no libnuma/CUDA).
- **Client-local rail recovery**: local device/verbs evidence (`Open`, local QP
  transition, MR registration/refresh, post/CQ API failure, or a locally
  classified WC) is `kRailFailure`; peer/bootstrap/protocol failure, remote/RNR/
  retry/flush WC, silent completion timeout, resource admission, cancellation,
  and invalid input are not. A first-attempt `kRailFailure` retires the failed
  endpoint and synchronously tears down its QP/MRs before one fresh retry. The
  retry excludes that rail from both NUMA-preferred and fallback selection, so
  it uses a distinct topology-enabled local rail whenever one exists. There are
  at most two physical attempts. A one-rail topology may instead make the one
  fresh retry on that same rail, but it never reuses the failed QP or bypasses
  quarantine, cooldown, credits, or the single recovery-probe gate.
- **Bounded at-least-once replay**: a rail retry replays the existing whole
  logical PUT/GET/batch/SG operation after teardown; it does not promise
  exactly-once PUT execution when a remote commit preceded a lost completion.
  This is client-only recovery and is separate from server node/ring health
  (`DFKV_RDMA_HEALTH_FILE` and `dfkv_server_ring_eligible`). It changes neither
  the RDMA wire protocol nor server behavior, so upgraded clients remain
  compatible with mixed-version servers that already speak the same RDMA v2
  protocol.
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
| `DFKV_RDMA_HEALTH_RECOVERY_SAMPLES` | `3` | With an RDMA listener, the server gates placement on every initialized/resolved IB rail, including the auto-discovered rail. Any query failure or non-ACTIVE/LinkUp rail removes the node immediately; three successful sampling opportunities re-admit it. At the nominal 10 s registrar cadence this is usually about 20–30 s depending on phase, plus registrar connection/RPC delays, and can be longer while MDS is unreachable; it is not an upper bound. |
| `DFKV_RDMA_HEALTH_FILE` | unset | Diagnostic-only health input (`device port_state phys_state`, one per initialized/resolved rail, including an auto-discovered rail) for controlled fault injection. Production MUST leave this unset so health comes from sysfs. |
| `--ram-tier` / `--ram-tier-bytes` / `--ram-tier-shards` | on / sized to the node / `16` for ≥100 GiB arenas | Large arenas contend on the shard locks under mixed load (+40 % mixed R/W at 16 shards on a 128 GiB arena); small (≤16 GiB) arenas are fine at the default 8. |
| `--store-engine` | `slab` | Index rebuilds on restart; removes file-per-block hazards. |
| `DFKV_TCP_FIRST_REQ_MS` / `DFKV_MDS_FIRST_REQ_MS` / `DFKV_METRICS_FIRST_REQ_MS` | `30000` (default) / `0` = off | First-request deadline per listener: a connection that sends nothing before the deadline is dropped, capping idle pre-auth connections. |
| `DFKV_METRICS_MAX_CONNS` | `64` (default) | Connection cap on the metrics listener; protects the exporter from connection floods. |

**Production client** (inference connectors, one process per TP rank):

| Knob | Recommended | Why |
|---|---|---|
| `DFKV_RDMA_DEV` | leave unset for one local HCA; use the same fabric whitelist on both hosts for multi-rail | Unset lets each endpoint select its own first `ACTIVE` local HCA, so local names may differ. A comma list explicitly enables multi-rail and is sent to the peer; every listed name must exist and be on the intended interoperable fabric at both ends. |
| `DFKV_RDMA_DEPTH` | `4` (default) | Keep client/server defaults aligned for connector batch correctness. Throughput scaling comes from multiple pooled connections, not raising one QP's depth; lowering depth reduces receive-segment consumption only after validating the real engine workload. |
| `DFKV_RDMA_RAIL_ERROR_THRESHOLD` | `3` (default) | Consecutive client-local rail failures required to quarantine that rail; endpoint/peer failures do not contribute. |
| `DFKV_RDMA_RAIL_COOLDOWN_MS` | `5000` (default) | Quarantine duration. No admission occurs during cooldown; afterward exactly one real operation is admitted as the recovery probe. Probe success clears the rail state, local failure starts another cooldown, and endpoint failure proves neither recovery nor a new local fault. |
| local-rail attempts | fixed at `2` (not configurable) | One initial attempt plus at most one fresh retry after `kRailFailure`; a distinct enabled local rail is mandatory when present. A one-rail topology may retry fresh on the same rail without bypassing quarantine. |
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
