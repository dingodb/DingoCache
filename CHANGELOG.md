# Changelog

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
