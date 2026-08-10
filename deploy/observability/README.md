# dfkv local observability stack

A vendor-neutral, runnable-locally backend for the dfkv telemetry layer:

```
connectors (vllm/lmcache/hicache) --OTLP push--> otel-collector --+--> prometheus --> grafana
dfkv_server / dfkv_mds  <--Prometheus pull (/metrics)-------------/        ^
                                                  otel-collector --traces--> tempo --/
```

- **Connectors PUSH** their fleet metrics over OTLP to the Collector (they are
  many and dynamically scheduled, so pull-discovery is impractical).
- **dfkv C++ daemons are PULLED**: Prometheus scrapes `dfkv_server`/`dfkv_mds`
  `/metrics` directly (few, long-lived). Start them with `--metrics-port`.
- The Collector re-exposes pushed metrics at `:8889` for Prometheus and forwards
  connector traces to Tempo.

## Bring it up

The compose file pins Collector `0.158.0`, Prometheus `3.13.2`, Tempo `3.0.2`,
and Grafana `13.1.3`. Mainland-China hosts default to
`docker.m.daocloud.io`; set `DFKV_IMAGE_REGISTRY` to the approved private
registry when required. All published ports bind to loopback.

```bash
export DFKV_GRAFANA_ADMIN_PASSWORD='replace-me'
docker compose -f deploy/observability/docker-compose.yml up -d
curl -fsS http://127.0.0.1:13133/
```

Prometheus, Tempo, and Grafana data persist in named Docker volumes.

| Service | URL | Notes |
|---|---|---|
| Grafana | http://127.0.0.1:3000 | admin / configured password; anonymous access disabled; **Connector** and **Cluster** dashboards auto-provisioned |
| Prometheus | http://127.0.0.1:9090 | alert rules loaded from `alerts.yml` |
| Collector OTLP | HTTP `127.0.0.1:4318` (default stdlib); gRPC `127.0.0.1:4317` (explicit SDK exporter) | connectors push here |
| Collector health/scrape | http://127.0.0.1:13133 / http://127.0.0.1:8889/metrics | liveness / pushed metrics |
| Tempo | http://127.0.0.1:3200 | traces |

If a default host port is occupied, set the corresponding compose variable:
`DFKV_GRAFANA_PORT`, `DFKV_PROMETHEUS_PORT`, `DFKV_TEMPO_PORT`,
`DFKV_OTLP_HTTP_PORT`, `DFKV_OTLP_GRPC_PORT`,
`DFKV_COLLECTOR_HEALTH_PORT`, or `DFKV_COLLECTOR_PROM_PORT`. These change only
the loopback host port; container-to-container addresses remain unchanged.

## Point a connector at it

Telemetry is **off by default** (zero cost). Turn it on per connector process:

```bash
export DFKV_METRICS_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318      # default stdlib OTLP/HTTP JSON
# optional: a stable, human-readable id (else auto = host_pid_tp_rank)
export DFKV_CONNECTOR_ID=myhost-rank0
# optional: active per-cache-node latency probe (every 5s) so idle nodes still
# show avg/max latency. Off unless set. (The SGLang plugin auto-enables it.)
export DFKV_PROBE_INTERVAL_MS=5000
# No package is required for the default exporter. For the SDK exporter only:
# pip install 'dfkv-vllm[otel]' && export DFKV_METRICS_EXPORTER=otel
```

Then run vLLM / LMCache / SGLang as usual. Metrics appear in Grafana within ~15s.

## Configuration reference

All telemetry is opt-in. Configure via env vars (and, for the SGLang plugin,
equivalent `extra_config` keys, precedence **extra_config > env > default**).

| Env var | extra_config key | Default | Effect |
|---|---|---|---|
| `DFKV_METRICS_ENABLED` | `metrics` | `0` (off) | master switch for push metrics |
| `DFKV_TELEMETRY_ENABLED` | `telemetry` | `0` | umbrella switch (metrics + traces); explicit per-signal switch wins |
| `DFKV_METRICS_EXPORTER` | `metrics_exporter` | `stdlib` | `stdlib` = built-in OTLP/HTTP JSON; `otel` = optional OpenTelemetry SDK |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `otlp_endpoint` | stdlib: `http://localhost:4318`; otel: SDK default | HTTP `:4318` for stdlib; gRPC `:4317` only with an SDK gRPC exporter |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `otlp_protocol` | stdlib: fixed HTTP/JSON; otel: SDK default | only interpreted by the SDK exporter |
| `DFKV_CONNECTOR_ID` | `connector_id` | `<host>_<pid>_<tp_rank>` | stable instance id label; `[A-Za-z0-9._-]` |
| `DFKV_METRICS_EXPORT_INTERVAL_MS` | `metrics_export_interval_ms` | `10000` | OTLP push cadence (min 1000) |
| `DFKV_PROBE_INTERVAL_MS` | `probe_interval_ms` | `0` off (`5000` when metrics on) | C++ active per-peer latency probe |
| `DFKV_PEER_LATENCY_POLL_S` | `peer_latency_poll_s` | `10` | snapshot→push cadence for per-peer latency |

**Cost model.** When metrics are **off**: the connector op path evaluates no
metric args (a falsy no-op guard), the C++ datapath is byte-for-byte unchanged,
and the OTel SDK is never imported. When **on**: each op updates bounded
in-memory aggregates; a sleeping background poller snapshots the native C client
outside the request path, and a second thread pushes aggregates over OTLP every
`DFKV_METRICS_EXPORT_INTERVAL_MS`. Export failures emit one warning, retain
bounded trace data for retry, and emit a recovery warning; no request is blocked.

## Health and alert rules

Prometheus loads `alerts.yml` from this directory. It evaluates server readiness/
storage health, MDS scrape and etcd readiness, version skew, missing heartbeat
stats, RDMA completion/receive-segment failures, connector poll staleness, empty
rings, MDS reachability, transport backoff, and rendezvous timeouts.

Rules are visible under Prometheus **Alerts** and as the `ALERTS` series. This
local stack intentionally does not embed an Alertmanager destination; production
deployments must route Prometheus alerts to their existing Alertmanager/on-call
system. `promtool check config` validates both `prometheus.yml` and `alerts.yml`
in CI.


## Dashboards

Two dashboards split by audience: **connector / business-traffic view** vs
**backend / infrastructure-health view**. Both read the same Prometheus.

### 1. "Connector" (connector fleet — pushed via OTLP)

- Fleet inventory and request volume by connector type/id and package/native
  version.
- Request/key/byte rates; average, p99, and max latency; success/failure ratio.
- Per-cache-node active-probe latency.
- Native ring/MDS health, transport-pool connections/backoff, peer
  quarantine/recovery, same-host CPU/GPU rendezvous funnels, and client
  operation/I/O errors.
- RDMA rail connection/completion/error rates, CQ/MR state, completion
  timeouts, pipeline depth, and NUMA fallback reasons.

Use the **Connector type** / **Connector id** variables to drill into one type
or process. `dfkv_connector_client_stats_poll_success` and
`dfkv_connector_client_stats_last_success_unixtime` distinguish a healthy zero
from a stale last-good native snapshot.

### 2. "Cluster" (cache nodes + MDS — scraped via pull)

Fed by the `dfkv_server` / `dfkv_mds` Prometheus scrape jobs (start daemons with
`--metrics-port`):

- **Cache nodes** — composite readiness, startup completion, local storage/RAM
  health, MDS registration/heartbeat health, hit ratio, request latency and
  throughput, capacity/objects/evictions, open/rejected connections, and node
  version/uptime inventory.
- **Disk/RAM/slab** — physical/logical slab occupancy, per-class reserved vs
  used capacity, reclaim progress, watermarks, failed disks, io_uring batches
  and fallback/errors, RAM admission/promotion/flush backlog.
- **RDMA** — active connections, completions and errors per rail, receive
  segment registered/free bytes and evictions, one-sided PUT/GET payload rates,
  and io_uring read/fallback counters.
- **MDS** — scrape/etcd readiness, member/client counts, capacity and hit/miss
  aggregates per group, registration/list/keepalive and etcd latency/error
  rates, heartbeat completeness, version skew, and replica inventory.

Use the **Cache node** variable to filter by `instance`.

The connector snapshot poller mirrors the bounded native C-client allowlist to
central OTLP, so client I/O, peer, MDS, RDMA, transport-pool, and rendezvous
families are available on the Connector dashboard for vLLM, LMCache, and
SGLang. SGLang also keeps its process-local Prometheus mirror.

## Metric reference (what the connectors emit)

| Metric | Type | Labels |
|---|---|---|
| `dfkv_connector_info` | gauge (=1) | identity: `dfkv_connector_id/type/host/pid/tp_rank` + version: `dfkv_version` (connector pkg), `dfkv_native_version` (libdfkv.so) |
| `dfkv_connector_ops_total` | counter | `op`, `status` |
| `dfkv_connector_keys_total` | counter | `op` |
| `dfkv_connector_bytes_total` | counter | `op` |
| `dfkv_connector_op_seconds` | histogram | `op` |
| `dfkv_connector_op_max_seconds` | gauge | `op` |
| `dfkv_connector_client_stats_poll_success` | gauge | connector identity |
| `dfkv_connector_client_stats_last_success_unixtime` | gauge | connector identity |
| `dfkv_connector_{ring,mds,rdma,transport_pool,dedup,gpu_dedup,...}` | counter/gauge | native labels such as `peer`, `dev`, `reason` |

The complete authoritative family list and producer ownership are in
[`docs/METRICS.md`](../../docs/METRICS.md) §3.4.

Identity rides on OTLP **resource attributes**; the Collector's
`resource_to_telemetry_conversion` turns them into the `dfkv_connector_*` labels
above (with `.`→`_`). Metric-name suffixing is disabled so names are verbatim.

## Trace backend: Tempo vs Jaeger

Tempo provides Grafana trace search and connector request attributes. To use a
standalone Jaeger UI instead, point the Collector's `otlp/tempo` exporter at a
compatible Jaeger OTLP endpoint. On mainland-China hosts, mirror and pin the
Jaeger image in the approved registry rather than pulling Docker Hub directly.
