# SPDX-License-Identifier: Apache-2.0
"""Pure-standard-library OTLP/HTTP-JSON *trace* exporter — zero third-party deps.

The sibling of ``otlp_json`` (which pushes metrics): the OTLP spec allows HTTP
transport with JSON encoding, and the OpenTelemetry Collector accepts spans on
``:4318/v1/traces`` (``Content-Type: application/json``). So a connector can push
a span per sampled request using only ``urllib``/``json``/``threading`` — no
``opentelemetry`` SDK, no ``protobuf``/``grpcio`` — nothing to ``pip install``
inside the inference container, nothing that can shadow the engine's own deps.

Tracing is connector-side only (see ``tracing.py``): every span is a root span
for one connector op (batch_get_v1 / batch_set_v1 / exist / ...), with the
connector identity on the OTLP *resource* (so Jaeger/Tempo group spans by
connector_id / type / host / tp_rank). The wire protocol and the C++ cache
server are untouched; cross-process child spans are a later milestone.

This module is pure data + a background pusher; the sampling decision lives in
``tracing.py``. Finished span dicts are enqueued into a bounded buffer and a
daemon thread POSTs them in batches.
"""

from __future__ import annotations

import collections
import json
import threading
import urllib.request
import warnings

# Reuse the metrics exporter's attribute encoder + hostname helper so the two
# OTLP/HTTP-JSON paths can't drift in how they encode values / resolve the host.
from .otlp_json import _attr, _host


def traces_url(endpoint):
    """Resolve a generic or signal-specific OTLP endpoint to ``/v1/traces``."""
    ep = (endpoint or "http://localhost:4318").strip()
    if "://" not in ep:
        ep = "http://" + ep
    ep = ep.rstrip("/")
    if ep.endswith("/v1/traces"):
        return ep
    if ep.endswith("/v1/metrics"):
        return ep[:-len("/v1/metrics")] + "/v1/traces"
    return ep + "/v1/traces"


# OTLP span status codes: 0 UNSET, 1 OK, 2 ERROR. The spec says instrumentation
# should leave a successful span UNSET and only set ERROR on failure, so that's
# what we do; a queryable ``status`` attribute ("ok"/"fail") is added separately.
STATUS_ERROR = 2


def make_span(name, trace_id, span_id, start_ns, end_ns, attributes,
              failed=False, error=""):
    """Build one OTLP span dict. ``attributes`` is a flat {key: value} map (the
    encoder picks int/double/bool/string). ``trace_id``/``span_id`` are hex
    strings (16/8 bytes). Pure data; unit-testable without a network."""
    span = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "kind": 3,  # SPAN_KIND_CLIENT — the connector calls into the cache tier
        "startTimeUnixNano": str(int(start_ns)),
        "endTimeUnixNano": str(int(end_ns)),
        "attributes": [_attr(k, v) for k, v in attributes.items()],
    }
    if failed:
        st = {"code": STATUS_ERROR}
        if error:
            st["message"] = str(error)
        span["status"] = st
    return span


def _resource_attrs(identity):
    """Connector identity as OTLP resource attributes (the Collector turns these
    into span resource labels). Same family ``otlp_json.build_payload`` puts on
    metrics, kept here as a small copy so touching one path can't break the other."""
    attrs = [
        _attr("service.name",
              "dfkv-%s-connector" % (identity.connector_type or "unknown")),
        _attr("service.namespace", "dfkv"),
        _attr("dfkv.connector_id", identity.connector_id),
        _attr("dfkv.connector_type", identity.connector_type),
        _attr("dfkv.host", _host()),
        _attr("dfkv.pid", identity.pid),
        _attr("dfkv.tp_rank", int(identity.tp_rank)),
    ]
    for key, value in (
            ("dfkv.model", getattr(identity, "model", "")),
            ("dfkv.deployment", getattr(identity, "deployment", "")),
            ("dfkv.cache_role", getattr(identity, "cache_role", "")),
            ("dfkv.team", getattr(identity, "team", ""))):
        if value:
            attrs.append(_attr(key, value))
    if getattr(identity, "version", ""):
        attrs.append(_attr("dfkv.version", identity.version))
    if getattr(identity, "native_version", ""):
        attrs.append(_attr("dfkv.native_version", identity.native_version))
    return attrs


def build_payload(identity, spans):
    """Build the OTLP/HTTP-JSON ExportTraceServiceRequest from a connector
    identity + a list of span dicts (from ``make_span``). Pure data."""
    return {"resourceSpans": [{
        "resource": {"attributes": _resource_attrs(identity)},
        "scopeSpans": [{"scope": {"name": "dfkv.connector"}, "spans": list(spans)}],
    }]}


class StdlibSpanExporter:
    """Bounded in-process span buffer + a daemon thread that POSTs batches as
    OTLP/HTTP-JSON every ``interval_s``. Off the request hot path. Never routes
    through a proxy (the Collector is an internal endpoint).

    The buffer is a ``deque(maxlen=N)``: when full, the oldest span is dropped on
    append (and counted) rather than blocking the connector. ``identity`` is any
    object exposing connector_type/connector_id/tp_rank/pid/version/native_version."""

    def __init__(self, identity, endpoint, interval_s=5.0, max_spans=2048):
        self._identity = identity
        self._url = traces_url(endpoint)
        self._interval = float(interval_s)
        self._max = max(1, int(max_spans))
        self._buf = collections.deque(maxlen=self._max)
        self._lock = threading.Lock()
        self._dropped = 0
        self._stop = threading.Event()
        self._thread = None
        # An opener with an empty ProxyHandler ignores HTTP(S)_PROXY for this POST.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._failed = False
        self._flush_lock = threading.Lock()

    def enqueue(self, span):
        """Append a finished span. Drops (and counts) the oldest if the buffer is
        already full — bounded memory, never blocks the caller."""
        with self._lock:
            if len(self._buf) >= self._max:
                self._dropped += 1  # deque(maxlen) discards the oldest on append
            self._buf.append(span)

    def dropped(self):
        with self._lock:
            return self._dropped

    def buffered(self):
        with self._lock:
            return len(self._buf)

    def _drain(self):
        with self._lock:
            if not self._buf:
                return []
            spans = list(self._buf)
            self._buf.clear()
            return spans
    def _restore_after_failure(self, spans):
        with self._lock:
            combined = spans + list(self._buf)
            overflow = max(0, len(combined) - self._max)
            if overflow:
                self._dropped += overflow
                combined = combined[overflow:]
            self._buf = collections.deque(combined, maxlen=self._max)


    def flush(self):
        """Drain and POST one batch, restoring it on any delivery-path failure."""
        with self._flush_lock:
            spans = self._drain()
            if not spans:
                return None
            try:
                body = json.dumps(
                    build_payload(self._identity, spans)).encode("utf-8")
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
                self._restore_after_failure(spans)
                raise

    def _flush_guarded(self):
        try:
            status = self.flush()
        except Exception as exc:
            if not self._failed:
                warnings.warn(
                    "dfkv OTLP trace export failed for {}: {}; buffered spans "
                    "were retained and will be retried".format(self._url, exc),
                    RuntimeWarning, stacklevel=2)
            self._failed = True
            return False
        if status is not None and self._failed:
            warnings.warn(
                "dfkv OTLP trace export recovered for {}".format(self._url),
                RuntimeWarning, stacklevel=2)
            self._failed = False
        return True

    def _loop(self):
        while not self._stop.wait(self._interval):
            self._flush_guarded()

    def start(self):
        if self._interval <= 0:
            return
        self._thread = threading.Thread(target=self._loop, name="dfkv-otlp-traces",
                                        daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            # urllib's request timeout is 5s; do not race a final flush against
            # a periodic request that already drained the shared buffer.
            self._thread.join(timeout=6)
            self._thread = None
        self._flush_guarded()
