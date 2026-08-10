# SPDX-License-Identifier: Apache-2.0
"""Connector-side distributed tracing for the dfkv connectors.

Every connector op (batch_get_v1 / batch_set_v1 / exist / put / get / ...) funnels
through a ``span()`` context manager, exactly like the ``access_log`` / push-metrics
chokepoints. When a span is *kept* it is pushed as one OTLP root span to a central
Collector's ``/v1/traces`` (Jaeger/Tempo), tagged with this connector's identity on
the OTLP resource — so spans from many inference processes / TP ranks / hosts
converge into one trace backend (the "distributed" view). The wire protocol and the
C++ cache server are untouched; cross-process child spans on the server are a later
milestone.

Which requests are reported (decided at op end, when the latency is known):
  * slow request  — duration >= ``trace_slow_request_ms`` (the core knob; 0 = off)
  * sampled       — an extra ``trace_sample_percent`` (0..100) of all requests
  * failed        — an op that raised / set status=fail is always reported (cheap,
                    and exactly what you want when chasing errors)

Design (mirrors ``dfkv_access_log`` / ``metrics_push``):
  * Off by default => ``span()`` returns a frozen no-op singleton (~tens of ns,
    no timer, no OTLP import). Master switch DFKV_TRACING_ENABLED (or the umbrella
    DFKV_TELEMETRY_ENABLED).
  * On but not kept => one ``perf_counter`` enter/exit + the keep decision (~100 ns);
    no span dict, no IDs, no allocation.
  * Kept => generate trace/span IDs, build the span dict, enqueue to a bounded
    buffer; a daemon thread POSTs batches off the hot path (see ``otlp_traces``).

Config precedence (extra_config key > env var > default) matches the rest of the
telemetry layer; see ``config.py`` for the knob names.
"""

from __future__ import annotations

import atexit
import os
import random
import threading
import time
import warnings
from urllib.parse import urlsplit
from typing import Any, Optional

from . import config, otlp_traces


def _to_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _to_int(v: Any, default: int) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _new_trace_id() -> str:
    return os.urandom(16).hex()  # 16-byte trace id, 32 hex chars (OTLP)


def _new_span_id() -> str:
    return os.urandom(8).hex()   # 8-byte span id, 16 hex chars (OTLP)


# ---------------------------------------------------------------------------
# No-op (disabled) path — what span() returns when tracing is off. Zero cost.
# ---------------------------------------------------------------------------

class _NoopSpan:
    __slots__ = ()
    # Writable-looking attributes the caller may set; all dropped by __setattr__.
    keys: Any = 0
    hits: Any = None
    bytes: Any = None
    status: Any = None
    attrs: Any = None

    def __enter__(self) -> "_NoopSpan":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def __bool__(self) -> bool:
        return False  # falsy: callers can `if _sp:` to skip any expensive attr eval

    def __setattr__(self, name: str, value: Any) -> None:
        pass  # allow `_sp.hits = ...` without raising; just drop it


_NOOP_SPAN = _NoopSpan()


class _NoopTracer:
    enabled = False

    def span(self, op_name, num_keys=0):
        return _NOOP_SPAN

    def shutdown(self):
        pass


# ---------------------------------------------------------------------------
# Real (enabled) span context manager — times the op, samples + emits on exit.
# ---------------------------------------------------------------------------

class _RealSpan:
    """Per-op context manager. Optionally set ``.hits`` / ``.bytes`` / ``.status``
    ('ok'|'fail') / ``.attrs`` (extra {key:value}) before exit; status is inferred
    from whether the block raised when not set."""

    __slots__ = ("_tracer", "_name", "keys", "hits", "bytes", "status", "attrs",
                 "_t0", "_start_ns")

    def __init__(self, tracer: "_Tracer", name: str, num_keys: int):
        self._tracer = tracer
        self._name = name
        self.keys = int(num_keys)
        self.hits: Optional[int] = None
        self.bytes: Optional[int] = None
        self.status: Optional[str] = None
        self.attrs: Optional[dict] = None
        self._t0 = 0.0
        self._start_ns = 0

    def __enter__(self) -> "_RealSpan":
        self._t0 = time.perf_counter()
        self._start_ns = time.time_ns()
        return self

    def __bool__(self) -> bool:
        return True

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        duration = time.perf_counter() - self._t0
        failed = (self.status == "fail") or (exc_type is not None)
        if self._tracer.should_sample(duration * 1000.0, failed):
            self._tracer.emit(self._name, self._start_ns, duration, self.keys,
                              self.hits, self.bytes, failed, exc_val, self.attrs)
        return False  # never swallow the caller's exception


# ---------------------------------------------------------------------------
# Real tracer — holds identity + sampling knobs + the background span exporter.
# Doubles as the `identity` object the OTLP exporter reads (connector_type, ...).
# ---------------------------------------------------------------------------

class _Tracer:
    enabled = True

    def __init__(self, connector_type: str, connector_id: str, tp_rank: int,
                 model: str, deployment: str, cache_role: str, team: str,
                 endpoint: str, slow_ms: float, sample_percent: float,
                 interval_ms: int, max_spans: int, version: str = "",
                 native_version: str = "", exporter=None):
        self.connector_type = connector_type
        self.connector_id = connector_id
        self.tp_rank = int(tp_rank)
        self.model = model
        self.deployment = deployment
        self.cache_role = cache_role
        self.team = team
        self.pid = os.getpid()
        self.version = version or ""
        self.native_version = native_version or ""
        # slow_ms<=0 disables the latency trigger; sample_percent clamped to 0..100.
        self.slow_ms = max(0.0, float(slow_ms))
        self.sample_percent = min(100.0, max(0.0, float(sample_percent)))
        if exporter is not None:  # test seam: caller-managed exporter
            self._exporter = exporter
        else:
            self._exporter = otlp_traces.StdlibSpanExporter(
                self, endpoint, max(0, int(interval_ms)) / 1000.0, max_spans)
            self._exporter.start()

    def span(self, op_name: str, num_keys: int = 0) -> _RealSpan:
        return _RealSpan(self, op_name, num_keys)

    def should_sample(self, duration_ms: float, failed: bool) -> bool:
        if failed:
            return True
        if self.slow_ms > 0 and duration_ms >= self.slow_ms:
            return True
        if self.sample_percent > 0 and random.random() * 100.0 < self.sample_percent:
            return True
        return False

    def emit(self, name: str, start_ns: int, duration_s: float, keys: int,
             hits: Optional[int], nbytes: Optional[int], failed: bool,
             exc_val, extra: Optional[dict]) -> None:
        end_ns = start_ns + int(duration_s * 1e9)
        attributes = {
            "op": name,
            "dfkv.keys": int(keys),
            "dfkv.duration_ms": round(duration_s * 1000.0, 3),
            "status": "fail" if failed else "ok",
        }
        if hits is not None:
            attributes["dfkv.hits"] = int(hits)
        if nbytes is not None:
            attributes["dfkv.bytes"] = int(nbytes)
        if extra:
            attributes.update(extra)
        error_msg = ""
        if failed and exc_val is not None:
            error_msg = "{}: {}".format(type(exc_val).__name__, exc_val)
            attributes["dfkv.error"] = error_msg
        span = otlp_traces.make_span(
            name, _new_trace_id(), _new_span_id(), start_ns, end_ns,
            attributes, failed=failed, error=error_msg)
        self._exporter.enqueue(span)

    def shutdown(self) -> None:
        try:
            self._exporter.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level facade (mirrors metrics_push / dfkv_access_log).
# ---------------------------------------------------------------------------

_tracer = _NoopTracer()  # type: ignore[assignment]
_configured = False
_users = 0
_atexit_registered = False
_cfg_lock = threading.Lock()


def _grpc_target(endpoint: str, protocol: str) -> bool:
    if "grpc" in (protocol or "").lower():
        return True
    try:
        candidate = endpoint if "://" in endpoint else "http://" + endpoint
        return urlsplit(candidate).port == 4317
    except ValueError:
        return False


def configure(cfg: Optional[dict] = None, connector_type: str = "",
              tp_rank: int = 0, model: str = "", version: str = "",
              native_version: str = "", deployment: str = "",
              cache_role: str = "", team: str = "", _exporter=None) -> None:
    """Acquire the process-global tracer; first caller sets its identity."""
    global _tracer, _configured, _users, _atexit_registered
    with _cfg_lock:
        _users += 1
        if _configured:
            return
        _configured = True
        cfg = cfg or {}
        if not config.tracing_enabled(cfg):
            _tracer = _NoopTracer()
            return
        connector_id = config.resolve_connector_id(cfg, tp_rank)
        generic_endpoint = str(config.resolve(
            cfg, "otlp_endpoint", config.ENV_OTLP_ENDPOINT, "")).strip()
        endpoint = str(config.resolve(
            cfg, "trace_otlp_endpoint", config.ENV_TRACE_OTLP_ENDPOINT,
            "")).strip()
        protocol = str(config.resolve(
            cfg, "otlp_protocol", config.ENV_OTLP_PROTOCOL, "")).strip()
        if not endpoint:
            endpoint = generic_endpoint
        if _exporter is None and _grpc_target(endpoint, protocol):
            warnings.warn(
                "dfkv trace exporter requires OTLP/HTTP; the selected trace "
                "endpoint/protocol is gRPC. Set DFKV_TRACE_OTLP_ENDPOINT to "
                "the Collector HTTP receiver (normally :4318).",
                RuntimeWarning, stacklevel=2)
            _tracer = _NoopTracer()
            if not _atexit_registered:
                atexit.register(shutdown)
                _atexit_registered = True
            return
        model = str(model or config.resolve(
            cfg, "model", config.ENV_MODEL, "")).strip()
        deployment = str(deployment or config.resolve(
            cfg, "deployment", config.ENV_DEPLOYMENT, "")).strip()
        cache_role = str(cache_role or config.resolve(
            cfg, "cache_role", config.ENV_CACHE_ROLE, "")).strip()
        team = str(team or config.resolve(
            cfg, "team", config.ENV_TEAM, "")).strip()
        slow_ms = _to_float(config.resolve(
            cfg, "trace_slow_request_ms",
            config.ENV_TRACE_SLOW_REQUEST_MS, 1000), 1000.0)
        sample_percent = _to_float(config.resolve(
            cfg, "trace_sample_percent",
            config.ENV_TRACE_SAMPLE_PERCENT, 0), 0.0)
        interval_ms = _to_int(config.resolve(
            cfg, "trace_export_interval_ms",
            config.ENV_TRACE_EXPORT_INTERVAL_MS, 5000), 5000)
        max_spans = _to_int(config.resolve(
            cfg, "trace_max_buffered_spans",
            config.ENV_TRACE_MAX_BUFFERED_SPANS, 2048), 2048)
        try:
            _tracer = _Tracer(
                connector_type, connector_id, tp_rank, model, deployment,
                cache_role, team, endpoint, slow_ms, sample_percent,
                interval_ms, max_spans, version=version,
                native_version=native_version, exporter=_exporter)
        except Exception as exc:
            warnings.warn(
                "dfkv tracing initialization failed; connector will continue "
                "without tracing: {}".format(exc),
                RuntimeWarning, stacklevel=2)
            _tracer = _NoopTracer()
        if not _atexit_registered:
            atexit.register(shutdown)
            _atexit_registered = True


def span(op_name: str, num_keys: int = 0):
    """Cheap when disabled (frozen singleton); times + samples + emits when on."""
    return _tracer.span(op_name, num_keys)


def is_enabled() -> bool:
    return _tracer.enabled


def release() -> None:
    """Release one connector's tracing lifecycle reference."""
    global _tracer, _configured, _users
    with _cfg_lock:
        if _users <= 0:
            return
        _users -= 1
        if _users:
            return
        tracer = _tracer
        _tracer = _NoopTracer()
        _configured = False
    tracer.shutdown()


def shutdown() -> None:
    """Force process-wide shutdown (the atexit path)."""
    global _tracer, _configured, _users
    with _cfg_lock:
        tracer = _tracer
        _tracer = _NoopTracer()
        _configured = False
        _users = 0
    tracer.shutdown()


def _reset_for_test() -> None:
    """Test seam: drop config + tracer so a test can reconfigure from env."""
    try:
        shutdown()
    except Exception:
        pass
