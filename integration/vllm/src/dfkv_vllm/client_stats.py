"""Expose the native dfkv client's bounded metrics through vLLM Prometheus.

The connector's ``vllm:dfkv_store_*`` metrics describe Python-level KV
operations. This poller complements them with the authoritative native
operation, peer-health, MDS, RDMA rail, memory-registration, and timeout state
from ``dfkv_stats_snapshot``. Polling runs on one sleeping daemon thread and
never touches the KV transfer request path.
"""

import threading
import time

from dfkv_common.client_metrics import (
    CLIENT_METRIC_SPECS,
    SnapshotDelta,
    read_native_snapshot,
)

try:
    from prometheus_client import Counter as _PromCounter, Gauge as _PromGauge
    _HAVE_PROM = True
except Exception:
    _HAVE_PROM = False


_PROM = {}
_POLL_PROM = {}
if _HAVE_PROM:
    try:
        for _source, _spec in CLIENT_METRIC_SPECS.items():
            _name = f"vllm:{_source}"
            _labels = ["tp_rank", *_spec.labels]
            if _spec.kind == "counter":
                _PROM[_source] = _PromCounter(
                    _name, _spec.description, _labels)
            else:
                _PROM[_source] = _PromGauge(
                    _name, _spec.description, _labels,
                    multiprocess_mode="liveall")
        _POLL_PROM["success"] = _PromGauge(
            "vllm:dfkv_client_stats_snapshot_success",
            "1 when the latest native client metrics snapshot poll succeeded",
            ["tp_rank"], multiprocess_mode="liveall")
        _POLL_PROM["timestamp"] = _PromGauge(
            "vllm:dfkv_client_stats_snapshot_timestamp_seconds",
            "Unix timestamp of the latest successful native client metrics snapshot",
            ["tp_rank"], multiprocess_mode="liveall")
        _POLL_PROM["errors"] = _PromCounter(
            "vllm:dfkv_client_stats_snapshot_errors_total",
            "Native client metrics snapshot poll failures", ["tp_rank"])
    except Exception:
        _HAVE_PROM = False
        _PROM = {}
        _POLL_PROM = {}


def read_snapshot(lib, h) -> str:
    """Read one complete native Prometheus snapshot through the C ABI."""
    return read_native_snapshot(lib, h)


class ClientStatsPoller:
    """Mirror every allow-listed native client metric off the request path."""

    def __init__(self, get_text, tp_rank, interval_s=15.0):
        self._get_text = get_text
        self._rank = str(int(tp_rank))
        self._interval = float(interval_s)
        self._mirror = SnapshotDelta()
        self._health = {
            "success": 0,
            "timestamp_seconds": 0.0,
            "errors_total": 0,
        }
        self._stop = threading.Event()
        self._thread = None
        self._warned = False

    def _record_failure(self):
        self._health["success"] = 0
        self._health["errors_total"] += 1
        if _HAVE_PROM:
            _POLL_PROM["success"].labels(self._rank).set(0)
            _POLL_PROM["errors"].labels(self._rank).inc()

    def poll_once(self):
        try:
            text = self._get_text() or ""
            if not text:
                raise RuntimeError("empty dfkv client metrics snapshot")
            updates = self._mirror.prepare(text)
            if not updates:
                raise RuntimeError("no recognized dfkv client metrics")
            for update in updates:
                if _HAVE_PROM:
                    metric = _PROM[update.name].labels(
                        self._rank, *update.labels)
                    if update.kind == "counter":
                        metric.inc(update.value)
                    else:
                        metric.set(update.value)
                self._mirror.commit(update)
            now = time.time()
            self._health["success"] = 1
            self._health["timestamp_seconds"] = now
            if _HAVE_PROM:
                _POLL_PROM["success"].labels(self._rank).set(1)
                _POLL_PROM["timestamp"].labels(self._rank).set(now)
        except Exception:
            self._record_failure()
            raise

    def totals(self):
        return self._mirror.totals()

    def gauges(self):
        return self._mirror.gauges()

    def health(self):
        return dict(self._health)

    def _loop(self):
        while not self._stop.wait(self._interval):
            try:
                self.poll_once()
            except Exception:
                if not self._warned:
                    self._warned = True
                    import warnings
                    warnings.warn(
                        "dfkv client stats poll failed; metric marks it failed "
                        "(further warnings suppressed)", stacklevel=2)

    def start(self):
        if self._interval <= 0:
            return
        try:
            self.poll_once()
        except Exception:
            pass
        self._thread = threading.Thread(target=self._loop,
                                        name="dfkv-client-stats", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
