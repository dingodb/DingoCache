# SPDX-License-Identifier: Apache-2.0
"""Pure-standard-library OTLP/HTTP-JSON metrics exporter — zero third-party deps.

The OTLP spec allows HTTP transport with JSON encoding; the OpenTelemetry
Collector accepts it on ``:4318/v1/metrics`` (``Content-Type: application/json``).
That lets a connector push its fleet metrics using only ``urllib``/``json``/
``threading`` — no ``opentelemetry`` SDK, no ``protobuf``/``grpcio``/``requests`` —
so there's nothing to ``pip install`` inside the inference container, and nothing
that can shadow the engine's own dependencies. The OTel SDK stays an opt-in
alternative (``DFKV_METRICS_EXPORTER=otel``).

It reads the recorder's in-process aggregates each interval and emits:
  dfkv_connector_ops_total{op,status}   Sum (cumulative, monotonic)
  dfkv_connector_keys_total{op}         Sum
  dfkv_connector_bytes_total{op}        Sum
  dfkv_connector_op_seconds{op}         Histogram (buckets -> p99 in Grafana)
  dfkv_connector_op_max_seconds{op}     Gauge (max since last export)
  dfkv_connector_info                   Gauge (=1; identity via resource attrs)
  dfkv_client_peer_latency_{avg,max}_seconds{peer}  Gauge
Identity rides on OTLP *resource* attributes (the Collector's
resource_to_telemetry_conversion turns them into Prometheus labels).
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.request
import warnings


def _attr(key, value):
    if isinstance(value, bool):
        v = {"boolValue": value}
    elif isinstance(value, int):
        v = {"intValue": str(value)}
    elif isinstance(value, float):
        v = {"doubleValue": value}
    else:
        v = {"stringValue": str(value)}
    return {"key": key, "value": v}


def metrics_url(endpoint):
    """Resolve a generic or signal-specific OTLP endpoint to ``/v1/metrics``."""
    ep = (endpoint or "http://localhost:4318").strip()
    if "://" not in ep:
        ep = "http://" + ep
    ep = ep.rstrip("/")
    if ep.endswith("/v1/metrics"):
        return ep
    if ep.endswith("/v1/traces"):
        return ep[:-len("/v1/traces")] + "/v1/metrics"
    return ep + "/v1/metrics"


def _host():
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def build_payload(rec, snap, start_ns, now_ns):
    """Build the OTLP/HTTP-JSON ExportMetricsServiceRequest from a recorder + an
    aggregate snapshot. Pure data; unit-testable without a network."""
    res_attrs = [
        _attr("service.name", "dfkv-%s-connector" % (rec.connector_type or "unknown")),
        _attr("service.namespace", "dfkv"),
        _attr("dfkv.connector_id", rec.connector_id),
        _attr("dfkv.connector_type", rec.connector_type),
        _attr("dfkv.host", _host()),
        _attr("dfkv.pid", os.getpid()),
        _attr("dfkv.tp_rank", int(rec.tp_rank)),
    ]
    for key, value in (
            ("dfkv.model", getattr(rec, "model", "")),
            ("dfkv.deployment", getattr(rec, "deployment", "")),
            ("dfkv.cache_role", getattr(rec, "cache_role", "")),
            ("dfkv.team", getattr(rec, "team", ""))):
        if value:
            res_attrs.append(_attr(key, value))
    if getattr(rec, "version", ""):
        res_attrs.append(_attr("dfkv.version", rec.version))
    if getattr(rec, "native_version", ""):
        res_attrs.append(_attr("dfkv.native_version", rec.native_version))

    t = str(now_ns)
    st = str(start_ns)
    metrics = []

    # --- counters (cumulative, monotonic Sums) ---
    ops_dps = []
    for (op, status), c in snap["ops"].items():
        if c:
            ops_dps.append({"attributes": [_attr("op", op), _attr("status", status)],
                            "startTimeUnixNano": st, "timeUnixNano": t, "asInt": str(c)})
    if ops_dps:
        metrics.append({"name": "dfkv_connector_ops_total",
                        "sum": {"aggregationTemporality": 2, "isMonotonic": True,
                                "dataPoints": ops_dps}})

    for name, key in (("dfkv_connector_keys_total", "keys"),
                      ("dfkv_connector_bytes_total", "bytes")):
        dps = [{"attributes": [_attr("op", op)], "startTimeUnixNano": st,
                "timeUnixNano": t, "asInt": str(v)}
               for op, v in snap[key].items() if v]
        if dps:
            metrics.append({"name": name,
                            "sum": {"aggregationTemporality": 2, "isMonotonic": True,
                                    "dataPoints": dps}})

    # --- op latency histogram (buckets -> p99) ---
    bounds = list(snap["bounds"])
    hist_dps = []
    for op, cnt in snap["dur_cnt"].items():
        if cnt:
            hist_dps.append({
                "attributes": [_attr("op", op)],
                "startTimeUnixNano": st, "timeUnixNano": t,
                "count": str(cnt), "sum": snap["dur_sum"][op],
                "bucketCounts": [str(c) for c in snap["dur_buckets"][op]],
                "explicitBounds": bounds,
            })
    if hist_dps:
        metrics.append({"name": "dfkv_connector_op_seconds",
                        "histogram": {"aggregationTemporality": 2, "dataPoints": hist_dps}})

    # --- gauges: per-op max, info heartbeat, per-peer latency ---
    max_dps = [{"attributes": [_attr("op", op)], "timeUnixNano": t,
                "asDouble": value}
               for op, value in snap["dur_max"].items() if value]
    if max_dps:
        metrics.append({"name": "dfkv_connector_op_max_seconds",
                        "gauge": {"dataPoints": max_dps}})

    metrics.append({"name": "dfkv_connector_info",
                    "gauge": {"dataPoints": [{"timeUnixNano": t, "asInt": "1"}]}})

    peer = snap["peer"]
    if peer:
        for name, field in (
                ("dfkv_client_peer_latency_avg_seconds", "avg"),
                ("dfkv_client_peer_latency_max_seconds", "max")):
            metrics.append({"name": name, "gauge": {"dataPoints": [
                {"attributes": [_attr("peer", peer_name)],
                 "timeUnixNano": t, "asDouble": values[field]}
                for peer_name, values in peer.items()]}})
        for name, field in (
                ("dfkv_client_peer_latency_seconds_sum", "sum"),
                ("dfkv_client_peer_latency_seconds_count", "count")):
            metrics.append({"name": name,
                            "sum": {"aggregationTemporality": 2,
                                    "isMonotonic": True,
                                    "dataPoints": [
                                        {"attributes": [_attr("peer", peer_name)],
                                         "startTimeUnixNano": st,
                                         "timeUnixNano": t,
                                         "asDouble": values[field]}
                                        for peer_name, values in peer.items()]}})

    # --- per-op metrics forwarded from the C++ KVClient snapshot (dfkv_client_op_*),
    #     re-emitted with this connector's identity. One name set across all
    #     connectors since they share the C chokepoint. Values are absolute/
    #     cumulative (counters) so no per-window delta is needed.
    cops = snap.get("client_ops") or {}
    if cops:
        for fld, mname in (("requests", "dfkv_connector_op_requests_total"),
                           ("keys", "dfkv_connector_op_keys_total"),
                           ("hits", "dfkv_connector_op_hits_total"),
                           ("bytes", "dfkv_connector_op_bytes_total")):
            dps = [{"attributes": [_attr("op", op)], "startTimeUnixNano": st,
                    "timeUnixNano": t,
                    "asDouble": float(values[fld])}
                   for op, values in cops.items() if fld in values]
            if dps:
                metrics.append({"name": mname,
                                "sum": {"aggregationTemporality": 2,
                                        "isMonotonic": True, "dataPoints": dps}})
        max_dps = [{"attributes": [_attr("op", op)], "timeUnixNano": t,
                    "asDouble": float(values["max"])}
                   for op, values in cops.items() if "max" in values]
        if max_dps:
            metrics.append({"name": "dfkv_connector_op_lifetime_max_seconds",
                            "gauge": {"dataPoints": max_dps}})
        hist_dps = []
        for op, d in cops.items():
            cnt = d.get("lat_count")
            buckets = d.get("buckets")
            if cnt is None or buckets is None:
                continue
            # Prometheus cumulative (le) buckets -> OTLP explicitBounds + per-bucket
            # counts. The +Inf bucket = count - last cumulative.
            fin = sorted(((float(le), cum) for le, cum in buckets
                          if le not in ("+Inf", "Inf")), key=lambda x: x[0])
            bounds = [b for b, _ in fin]
            total = float(cnt)
            counts, prev = [], 0.0
            for _, c in fin:
                counts.append(str(int(max(0.0, c - prev))))
                prev = c
            counts.append(str(int(max(0.0, total - prev))))  # +Inf bucket
            hist_dps.append({
                "attributes": [_attr("op", op)],
                "startTimeUnixNano": st, "timeUnixNano": t,
                "count": str(int(total)), "sum": float(d.get("lat_sum", 0.0)),
                "bucketCounts": counts, "explicitBounds": bounds,
            })
        if hist_dps:
            metrics.append({"name": "dfkv_connector_op_latency_seconds",
                            "histogram": {"aggregationTemporality": 2,
                                          "dataPoints": hist_dps}})

    # --- same-host rendezvous counters (dfkv_client_[gpu_]dedup_*), forwarded
    #     under this connector's identity. The top of the hit-rate funnel: how
    #     much of the TP-replicated L3 read load was collapsed to 1x vs fanned
    #     out per rank. Label-free cumulative counters -> per-name OTLP sums.
    dedup = snap.get("client_dedup") or {}
    for cname, val in sorted(dedup.items()):
        metrics.append({"name": cname.replace("dfkv_client_", "dfkv_connector_"),
                        "sum": {"aggregationTemporality": 2, "isMonotonic": True,
                                "dataPoints": [{"startTimeUnixNano": st,
                                                "timeUnixNano": t,
                                                "asInt": str(int(val))}]}})

    # --- bounded native client health/transport/MR metrics. Group labeled
    # samples into one OTLP metric family and preserve counter vs gauge type.
    native = {}
    for sample in snap.get("client_status") or []:
        native.setdefault(
            (sample["name"], sample["kind"]), []).append(sample)
    for (name, kind), samples in sorted(native.items()):
        points = []
        for sample in samples:
            point = {
                "attributes": [
                    _attr(key, value)
                    for key, value in sorted(sample["labels"].items())
                ],
                "timeUnixNano": t,
            }
            if kind == "counter":
                point["startTimeUnixNano"] = st
                point["asInt"] = str(int(sample["value"]))
            else:
                point["asDouble"] = float(sample["value"])
            points.append(point)
        if kind == "counter":
            metrics.append({
                "name": name,
                "sum": {
                    "aggregationTemporality": 2,
                    "isMonotonic": True,
                    "dataPoints": points,
                },
            })
        else:
            metrics.append({"name": name, "gauge": {"dataPoints": points}})

    poll = snap.get("client_poll") or {}
    if poll.get("observed"):
        for name, field in (
                ("dfkv_connector_client_stats_poll_success", "success"),
                ("dfkv_connector_client_stats_last_success_unixtime",
                 "last_success_unixtime")):
            metrics.append({"name": name, "gauge": {"dataPoints": [{
                "timeUnixNano": t, "asDouble": float(poll[field])}]}})
        metrics.append({
            "name": "dfkv_connector_client_stats_poll_errors_total",
            "sum": {
                "aggregationTemporality": 2,
                "isMonotonic": True,
                "dataPoints": [{
                    "startTimeUnixNano": st,
                    "timeUnixNano": t,
                    "asInt": str(int(poll["errors"])),
                }],
            },
        })

    return {"resourceMetrics": [{
        "resource": {"attributes": res_attrs},
        "scopeMetrics": [{"scope": {"name": "dfkv.connector"}, "metrics": metrics}],
    }]}


class StdlibExporter:
    """Background thread that POSTs the recorder's aggregates as OTLP/HTTP-JSON
    every ``interval_s``. Off the request hot path. Never routes through a proxy
    (the Collector is an internal endpoint)."""

    def __init__(self, recorder, endpoint, interval_s=10.0):
        self._rec = recorder
        self._url = metrics_url(endpoint)
        self._interval = float(interval_s)
        self._start_ns = time.time_ns()
        self._stop = threading.Event()
        self._thread = None
        # An opener with an empty ProxyHandler ignores HTTP(S)_PROXY for this POST.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._failed = False
        self._push_lock = threading.Lock()

    def build(self, now_ns=None):
        now_ns = now_ns or time.time_ns()
        snap = self._rec.export_snapshot()
        return build_payload(self._rec, snap, self._start_ns, now_ns)

    def push_once(self):
        # Serialize the periodic and shutdown flushes. The snapshot's window max
        # is committed only after delivery; cumulative counters remain absolute.
        with self._push_lock:
            snap = self._rec.export_snapshot()
            try:
                payload = build_payload(
                    self._rec, snap, self._start_ns, time.time_ns())
                body = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    self._url, data=body,
                    headers={"Content-Type": "application/json"})
                resp = self._opener.open(req, timeout=5)
                try:
                    resp.read()
                    return resp.status
                finally:
                    resp.close()
            except Exception:
                self._rec.restore_maxima(snap)
                raise
    def _push_guarded(self):
        try:
            self.push_once()
        except Exception as exc:
            if not self._failed:
                warnings.warn(
                    "dfkv OTLP metrics export failed for {}: {}; "
                    "will retry in the background".format(self._url, exc),
                    RuntimeWarning, stacklevel=2)
            self._failed = True
            return False
        if self._failed:
            warnings.warn(
                "dfkv OTLP metrics export recovered for {}".format(self._url),
                RuntimeWarning, stacklevel=2)
        self._failed = False
        return True

    def _loop(self):
        while not self._stop.wait(self._interval):
            self._push_guarded()

    def start(self):
        if self._interval <= 0:
            return
        self._thread = threading.Thread(target=self._loop, name="dfkv-otlp-json",
                                        daemon=True)
        self._thread.start()

    def stop(self, flush=True):
        self._stop.set()
        if self._thread is not None:
            # urllib's request timeout is 5s. Wait past it so the final flush
            # cannot race an in-flight periodic flush over a drained max window.
            self._thread.join(timeout=6)
            self._thread = None
        if flush:
            self._push_guarded()
