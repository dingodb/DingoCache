# Changelog

## Unreleased

### vLLM hybrid state and explicit TCP staging

- Accepted align-mode hybrid cache groups whose Mamba block size differs from
  the attention cache block size. The connector already resolves the scheduler
  LCM and preserves each group's physical block geometry; the old equality
  gate incorrectly rejected Qwen3.8-Flash-Next (`400` vs `4`) at startup.
- Added opt-in `require_rdma=false` for the vLLM connector. The default remains
  fail-closed GPUDirect RDMA; the opt-in path uses libdfkv's bounded host
  staging plus final CUDA publication and enables a correctness-first fallback
  when a platform's inbound GPUDirect GET path is unusable.

### LMCache rank-local rail affinity

- Added opt-in `rail_affinity` and `rail_affinity_fallbacks` plugin settings to
  the in-process LMCache dfkv connector. Each worker now uses LMCache's explicit
  per-host `local_worker_id` to apply the shared primary/fallback policy before
  opening the native client; missing or invalid local metadata fails closed.
- Kept the MP-server L2 adapter unchanged because it owns one shared dfkv
  client and has no single physical GPU rank at construction.

### vLLM rank-local rail affinity

- Added `rail_affinity` and `rail_affinity_fallbacks` to
  `DfkvStoreConnector`. Before opening the native client, each vLLM worker now
  maps its world-group local rank to one primary plus a bounded ordered
  fallback set, then sets `DFKV_RDMA_DEV`, `DFKV_RDMA_PRIMARY_DEV`, and
  `DFKV_RDMA_NUMA=0`.
- Moved the rail-list parsing, fallback bounds, duplicate rejection, and
  environment update contract into `dfkv-common`; SGLang HiCache and vLLM now
  share one implementation while retaining their framework-specific physical
  rank resolution.

### RDMA connection capacity

- Unified scalar and scatter-gather data operations on one exclusive-ownership
  endpoint pool. Control operations remain isolated. Ring autoscaling now sizes
  `nodes × 2 pools × max(pool limit, rails) × 1.25`.
- Lowered the per-peer/pool idle default from 16 to 8: seven process-wide read
  workers plus one warm spare avoid reconnect churn, while a cap of four is
  proven to reopen three endpoints under a seven-way concurrent wave.
- With 4 MiB blocks and depth four, one pull data connection leases
  33,587,200 bytes. The xb01 14×TP8, per-pool-limit-8 envelope is 896 data plus
  896 smaller control connections per server. Including 25% churn consumes
  37,984,665,600 bytes (35.4 GiB), so xb01's 64 GiB setting remains sufficient;
  increasing it to 128 GiB is unnecessary.
- On xb01-gpu-200b-0064, 64 GiB and 128 GiB segments both registered on
  `ib7s400p6` with zero allocation failures. Readiness took 21 s and 43 s
  respectively, so 128 GiB doubles the pinned MR and adds 22 s startup latency
  without serving the current capacity envelope.
- Added client idle/active pool gauges by operation lane and rail, live
  connection gauges by peer and local rail, and the pool limit. Added server
  receive-segment used/free/largest-range, allocation-failure, pull/legacy
  connection, and data/control leased-byte metrics.
- `rail_affinity=true` now assigns each rank one primary plus one ordered
  neighboring fallback HCA by default. `rail_affinity_fallbacks=0` restores
  strict one-rank/one-rail binding; larger values remain bounded by the
  configured HCA list.

### Metrics HTTP clean close

- Consume the complete HTTP request-head section, bounded to 32 KiB and by the
  existing accept-time absolute deadline, before rendering server or MDS
  metrics. Closing no longer leaves unread Prometheus headers that force Linux
  to send a TCP RST and truncate large remote scrapes.
- Preserve the existing one-request connection model, connection cap,
  slow-drip protection, health/readiness semantics, and send timeout. Successful
  responses now finish with an explicit write-side shutdown and clean EOF.
- Add realistic Prometheus headers, a 256 KiB byte-exact response, clean-EOF,
  split-header, header-drip, and oversized unterminated-header regressions.

### CUDA GET pinned publication pool

- Replaced per-operation CUDA host registration for GET publication with a
  process-wide, bounded, reusable portable-pinned bounce pool. Slots are fixed
  size, allocated and pinned/registered once on first use, exclusively leased,
  and returned only after the stream-ordered host-to-device copy synchronizes.
- Added `DFKV_CUDA_PINNED_POOL_BYTES` (default 67108864, 64 MiB; maximum
  4 GiB) and `DFKV_CUDA_PINNED_SLOT_BYTES` (default 4194304, 4 MiB; accepted
  range 4 KiB–64 MiB). Values are positive decimal byte counts; the budget
  rounds down to whole slots with at least one and at most 4096 slots. Invalid
  sizing warns and falls back to both defaults. Allocation is lazy, so
  pinned-memory and memlock consumption grows only to the concurrent high-water
  and remains bounded by that rounded per-process budget.
- When every slot is leased and the budget is exhausted, CUDA GET publication
  blocks until a slot is returned rather than allocating beyond the bound.
  Payloads larger than one slot are published in fixed-size chunks.
- Kept RDMA retry data in operation-owned pageable staging. Only the final
  winning attempt is copied through the pinned pool to caller CUDA memory, so a
  failed rail cannot partially publish into the caller buffer. Direct
  GPUDirect GET would violate that retry fence; GPUDirect PUT remains unchanged.
- CUDA context preservation, multi-device publication, byte-exact mixed
  scatter-gather behavior, synchronous completion, fail-closed CUDA errors,
  parent-process teardown, and fail-closed post-fork child behavior are
  preserved. A slot whose stream synchronization fails is quarantined and kept
  pinned until process exit rather than risking release while the driver may
  still reference it. Deployments must provision `RLIMIT_MEMLOCK` for the
  pool's lazy high-water plus other locked memory.

## v2.19.0 — 2026-08-13

### CUDA GET correctness

- Fixed a `RangeInto`/`RangeIntoMulti` failure in which the host publication
  path could call `memcpy` with a CUDA device destination and terminate the
  process with `SIGSEGV`.
- Added an explicit destination kind to range reads. CUDA GETs now receive RDMA
  data into operation-owned pinned host staging and publish it with asynchronous
  host-to-device copies only after the final healthy rail attempt succeeds.
  Completion remains synchronous: the call returns success only after the CUDA
  stream has completed publication.
- Applied the same publication and retry fence to contiguous ranges, mixed
  host/CUDA scatter-gather ranges, and rail retry, so a failed attempt cannot
  publish partial CUDA data before a later attempt succeeds.
- PUT and memory-registration behavior are unchanged and remain GPUDirect.
  CUDA GET adds temporary pinned host memory, a host-to-device copy, stream
  synchronization, and associated allocation/registration overhead. This is a
  known performance tradeoff pending safe direct-GPU retry fencing; deployments
  should validate pinned-memory and memlock capacity for their concurrent GET
  payload.

## v2.11.1 — 2026-08-10

### Release

- Corrected the tag workflow to keep the generic no-HCA release runner from
  executing hardware-only `RdmaLoopback.*` tests. RDMA remains gated by the
  dedicated compile and probe-gated Soft-RoCE CI jobs and by qualification on
  real RDMA hardware.
- Require every release tag commit to be reachable from `origin/main`.
- The immutable `v2.11.0` tag published no release assets because its generic
  runner attempted those hardware-only tests; v2.11.1 supersedes that failed
  publication attempt without data-path behavior changes.

## v2.11.0 — 2026-08-09

### Observability

- Added one bounded native `KVClient` snapshot covering operation, peer health,
  ring/MDS, RDMA rail/CQ/MR/timeout, TCP pool, and same-host dedup state.
- Mirrored the same native allowlist into SGLang HiCache and vLLM Prometheus
  endpoints without adding work to the transfer request path.
- Added opt-in connector-fleet OTLP push for SGLang HiCache, LMCache, and vLLM,
  with stable connector, host, rank, model, deployment, connector-package, and
  native-library identity.
- Added explicit last-good poll health (`success`, failure count, last-success
  time) so snapshot failures cannot masquerade as healthy zero traffic.
- Added MDS per-operation etcd request/error/latency metrics, MDS readiness,
  consumer counts, group capacity/hit/version/stats aggregates, and server
  startup/readiness/health gauges.
- Added bounded per-rail server and client traffic/error series, receive-segment
  capacity, io_uring activation/fallback, storage engine, RAM-tier, slab
  allocator/rebuild, and TCP pool/backoff diagnostics.
- Expanded the provisioned Grafana connector and cluster dashboards with
  readiness, poll freshness, empty-ring/MDS state, dedup, transport failures,
  receive-segment, rail bandwidth/errors, group capacity/hit ratio, and etcd
  latency views.
- Added Prometheus alert rules for server/MDS readiness, storage health, etcd
  errors, version skew, missing heartbeat stats, RDMA failures/capacity,
  connector poll staleness, empty rings, MDS reachability, TCP backoff, and
  dedup rendezvous timeouts.
- Corrected the default zero-dependency exporter contract: it sends OTLP/HTTP
  JSON to the HTTP receiver (normally port 4318); port 4317 is only for an
  explicitly selected OpenTelemetry SDK gRPC exporter.
- Made connector telemetry lifecycle-safe: process-shared recorders are reference
  counted, final metrics/traces flush on the last close, failed exports preserve
  window maxima and uncommitted native counter deltas, and constructor failures
  release native handles and telemetry acquisitions.
- Made one scrape-time MDS group range serve both readiness and aggregation;
  scrape traffic is separately labeled `op="metrics_range"`, and latency
  histograms now resolve bounded failures through 60 seconds.
- Added explicit missing-scrape/missing-connector alerts, bounded dashboard
  resource variables, and canonical `slot_size` slab labels.

### Operations and release

- Added `dfkvctl stat --all` health classification from stable server gauges,
  explicit handling for legacy metrics, and all-group MDS discovery/stat views.
- Added atomic operational artifacts for membership audit, load regression,
  tenant quota, and node-replacement workflows, plus GPU same-host dedup
  reproduction.
- Added a least-privilege, tag-gated GitHub Release workflow. It reruns tests,
  validates observability configuration, builds portable RDMA/io_uring binaries
  and all three connector wheels, emits SHA-256 checksums, and publishes assets
  once through a draft-to-immutable release transition.
- Made root `VERSION`, C++ binaries, and Python connector package versions a
  CI-enforced release contract.
- Tightened the install/package manifest so generated caches and local build
  artifacts are excluded while supported sources, deployment files, tests, and
  documentation remain available in the release archive.
- Hardened `dfkv_bench` as a regression gate: GET uses one up-front registered
  MR arena, setup fails closed instead of falling back to ad-hoc registration,
  goodput excludes failed operations, diagnostics expose control-plane and RDMA
  availability/errors, and both current and legacy output remain parseable.
- Release and CI now validate the runtime container end to end, preserve and
  verify versioned `libdfkv.so` SONAME symlinks, smoke-import the shipped SGLang
  and vLLM packages under Python 3.12, and verify installed wheel metadata and
  vendored telemetry contents.
- Fixed signed-integer overflow in the etcd Base64 decode accumulator; long
  membership values now remain defined under UBSan and decode byte-exactly.
- Made the RAM flush-backlog gauge include dequeued in-flight writes, closing a
  false-zero readiness window that could race the background reclaimer.
- Pinned release Actions, runtime/CI base images, and the bundled observability
  stack to reviewed immutable digests; Grafana now fails closed unless the
  operator supplies a non-default administrator password.

### Compatibility

- No data-format migration is required.
- Metrics and tracing remain opt-in. Existing deployments that do not enable a
  metrics port or connector telemetry retain their prior data-path behavior.
- Dashboards and alert rules target the v2.11.0 metric contract; missing series
  on older nodes indicate rollout state, not a numeric zero.
