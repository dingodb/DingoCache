"""Bounded parser and delta tracker for ``dfkv_stats_snapshot`` metrics.

The native client already owns the authoritative transport, peer-health, and
operation counters. Connectors poll that snapshot off their request path and use
this module to expose the same signals through their inference server's metrics
endpoint. Only fixed-cardinality labels emitted by the native client are
accepted: operation, peer, RDMA device, NUMA fallback reason, and transport.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
import re
from typing import Optional


@dataclass(frozen=True)
class MetricSpec:
    kind: str
    labels: tuple[str, ...] = ()
    description: str = "Native dfkv client metric mirrored from dfkv_stats_snapshot."


@dataclass(frozen=True)
class MetricUpdate:
    name: str
    kind: str
    labels: tuple[str, ...]
    value: float
    # Absolute native sample used to commit a prepared counter delta only after
    # its external Prometheus update succeeds. Hidden from normal consumers.
    source_value: Optional[float] = None


# Counters and gauges with operational value at the inference endpoint. Native
# latency histogram buckets are intentionally not re-registered through the
# Python client: prometheus_client cannot ingest cumulative external buckets.
# The request counters plus lifetime max gauges remain available here, while
# connector-owned histograms cover per-batch latency distributions.
_COUNTERS: dict[str, tuple[str, ...]] = {
    "dfkv_client_ops_served_total": (),
    "dfkv_client_io_errors_total": (),
    "dfkv_client_health_checks_total": (),
    "dfkv_client_unhealthy_skips_total": (),
    "dfkv_client_peer_marked_bad_total": (),
    "dfkv_client_peer_recovered_total": (),
    "dfkv_client_busy_suppressed_total": (),
    "dfkv_client_peer_errors_total": ("peer",),
    "dfkv_client_op_requests_total": ("op",),
    "dfkv_client_op_keys_total": ("op",),
    "dfkv_client_op_hits_total": ("op",),
    "dfkv_client_op_bytes_total": ("op",),
    "dfkv_client_dedup_hits_total": (),
    "dfkv_client_dedup_fetches_total": (),
    "dfkv_client_dedup_wait_hits_total": (),
    "dfkv_client_dedup_wait_timeouts_total": (),
    "dfkv_client_dedup_length_mismatches_total": (),
    "dfkv_client_gpu_dedup_hits_total": (),
    "dfkv_client_gpu_dedup_fetches_total": (),
    "dfkv_client_gpu_dedup_wait_hits_total": (),
    "dfkv_client_gpu_dedup_wait_timeouts_total": (),
    "dfkv_client_gpu_dedup_load_fences_total": (),
    "dfkv_client_gpu_dedup_load_fence_failures_total": (),
    "dfkv_client_gpu_dedup_invalidate_stuck_total": (),
    "dfkv_client_gpu_dedup_length_mismatches_total": (),
    "dfkv_client_mds_unreachable_polls_total": (),
    "dfkv_rdma_client_conns_opened_total": (),
    "dfkv_rdma_client_v2_put_writes_total": (),
    "dfkv_rdma_client_mr_registration_rejections_total": (),
    "dfkv_rdma_client_rail_conns_total": ("dev",),
    "dfkv_rdma_client_rail_selections_total": ("dev",),
    "dfkv_rdma_client_rail_put_ops_total": ("dev",),
    "dfkv_rdma_client_rail_put_bytes_total": ("dev",),
    "dfkv_rdma_client_rail_get_ops_total": ("dev",),
    "dfkv_rdma_client_rail_get_bytes_total": ("dev",),
    "dfkv_rdma_client_rail_credits_exhausted_total": ("dev",),
    "dfkv_rdma_client_rail_errors_total": ("dev",),
    "dfkv_rdma_client_endpoint_errors_total": ("dev",),
    "dfkv_rdma_client_rail_quarantines_total": ("dev",),
    "dfkv_rdma_client_rail_recoveries_total": ("dev",),
    "dfkv_rdma_client_numa_fallbacks_total": ("reason",),
    "dfkv_rdma_cq_completions_total": (),
    "dfkv_rdma_cq_errors_total": (),
    "dfkv_rdma_client_adhoc_user_mr_total": (),
    "dfkv_rdma_client_pool_mr_registrations_total": (),
    "dfkv_rdma_client_pool_mr_registration_failures_total": (),
    "dfkv_rdma_client_oversize_rejects_total": (),
    "dfkv_rdma_client_v2_probe_attempts_total": (),
    "dfkv_rdma_client_v2_probe_failures_total": (),
    "dfkv_rdma_client_stale_pool_retries_total": (),
    "dfkv_rdma_client_completion_timeouts_total": (),
    "dfkv_rdma_client_keepalive_attempts_total": (),
    "dfkv_rdma_client_keepalive_successes_total": (),
    "dfkv_rdma_client_keepalive_failures_total": (),
    "dfkv_transport_pool_selections_total": (),
    "dfkv_transport_pool_retirements_total": ("reason",),
    "dfkv_transport_pool_backoff_events_total": (),
    "dfkv_transport_pool_backoff_suppressed_total": (),
    "dfkv_read_scheduler_queue_delay_us_total": (),
    "dfkv_read_scheduler_fairness_yields_total": (),
    "dfkv_rdma_client_resource_budget_timeouts_total": (),
    "dfkv_rdma_endpoint_cache_hits_total": (),
    "dfkv_rdma_endpoint_cache_misses_total": (),
    "dfkv_rdma_endpoint_cache_evictions_total": (),
    # Rail and peer failure signals. Without them a rank cut off from a
    # server's rails (no compatible rail) or a server's failing rails were
    # invisible at the inference endpoint.
    # Second requests to a key's draining owner (MDS-04).
    "dfkv_client_drain_requests_total": ("op", "result"),
    "dfkv_rdma_client_no_compatible_rail_total": (),
    "dfkv_rdma_client_admission_failures_total": (),
    "dfkv_rdma_client_local_resource_failures_total": (),
    "dfkv_rdma_client_local_resource_rail_fallbacks_total": (),
    "dfkv_rdma_client_local_resource_held_total": (),
    "dfkv_rdma_client_rail_wc_errors_total": ("dev", "status"),
    "dfkv_rdma_client_cross_rail_retries_total": (),
    "dfkv_rdma_client_cross_rail_retry_successes_total": (),
    "dfkv_rdma_client_cross_rail_retry_exhausted_total": (),
    "dfkv_rdma_client_peer_unreachable_total": (),
    "dfkv_rdma_client_peer_unreachable_skips_total": (),
    "dfkv_rdma_client_peer_busy_total": (),
    "dfkv_rdma_client_rail_refusals_total": (),
    "dfkv_rdma_client_port_down_events_total": (),
    "dfkv_rdma_client_rail_background_probes_total": (),
    "dfkv_rdma_client_peer_topology_updates_total": (),
    "dfkv_rdma_client_remote_rail_normal_leases_total": (),
    "dfkv_rdma_client_remote_rail_admissions_denied_total": (),
    "dfkv_rdma_client_remote_rail_successes_total": (),
    "dfkv_rdma_client_remote_rail_failures_total": (),
    "dfkv_rdma_client_remote_rail_cooldowns_total": (),
    "dfkv_rdma_client_remote_rail_recovery_probes_total": (),
    "dfkv_rdma_client_remote_rail_recoveries_total": (),
    "dfkv_rdma_client_remote_rail_abandons_total": (),
    "dfkv_rdma_client_remote_rail_normal_leases_by_rail_total": ("dev",),
    "dfkv_rdma_client_remote_rail_admissions_denied_by_rail_total": ("dev",),
    "dfkv_rdma_client_remote_rail_successes_by_rail_total": ("dev",),
    "dfkv_rdma_client_remote_rail_failures_by_rail_total": ("dev",),
    "dfkv_rdma_client_remote_rail_cooldowns_by_rail_total": ("dev",),
    "dfkv_rdma_client_remote_rail_recovery_probes_by_rail_total": ("dev",),
    "dfkv_rdma_client_remote_rail_recoveries_by_rail_total": ("dev",),
    "dfkv_rdma_client_remote_rail_abandons_by_rail_total": ("dev",),
    # Data path, connection and pull accounting.
    "dfkv_rdma_client_pull_prepares_total": (),
    "dfkv_rdma_client_pull_reads_total": (),
    "dfkv_rdma_client_pull_read_bytes_total": (),
    "dfkv_rdma_client_pull_read_wrs_total": (),
    "dfkv_rdma_client_pull_failures_total": (),
    "dfkv_rdma_client_pull_releases_total": (),
    "dfkv_rdma_client_pull_release_failures_total": (),
    "dfkv_rdma_client_leaseput_ops_total": (),
    "dfkv_rdma_client_leaseput_path_fallbacks_total": (),
    "dfkv_rdma_client_stale_pool_connections_total": (),
    "dfkv_rdma_client_session_closed_reaps_total": (),
    "dfkv_rdma_client_stale_generation_reaps_total": (),
    "dfkv_rdma_client_depth_refunds_total": (),
    "dfkv_rdma_client_resource_budget_raises_total": (),
    "dfkv_rdma_client_connection_class_opened_total": ("block_bytes", "depth"),
}

_GAUGES: dict[str, tuple[str, ...]] = {
    "dfkv_client_transport_info": ("transport",),
    "dfkv_client_ring_members": (),
    "dfkv_client_ring_draining_members": (),
    "dfkv_client_ring_point_collisions": (),
    "dfkv_client_mds_reachable": (),
    "dfkv_client_peer_latency_max_seconds": ("peer",),
    "dfkv_client_op_max_seconds": ("op",),
    "dfkv_rdma_client_mr_regions": (),
    "dfkv_rdma_client_mr_registered_bytes": (),
    "dfkv_rdma_client_rail_inflight": ("dev",),
    "dfkv_rdma_client_rail_credits_available": ("dev",),
    "dfkv_rdma_client_rail_consecutive_errors": ("dev",),
    "dfkv_rdma_client_rail_quarantined": ("dev",),
    "dfkv_rdma_client_rail_recovery_probe": ("dev",),
    "dfkv_rdma_client_rail_latency_ewma_us": ("dev",),
    "dfkv_rdma_client_rails_port_down": (),
    "dfkv_rdma_client_pipeline_depth": (),
    "dfkv_rdma_client_transient_user_mr_active": (),
    "dfkv_rdma_client_pool_mr_active_registrations": (),
    "dfkv_rdma_client_max_block_seen_bytes": (),
    "dfkv_rdma_client_declared_max_block_bytes": (),
    "dfkv_transport_pool_connections": (),
    "dfkv_transport_pool_inflight": (),
    "dfkv_transport_pool_backoff_endpoints": (),
    "dfkv_read_scheduler_pending_batches": (),
    "dfkv_read_scheduler_pending_shards": (),
    "dfkv_read_scheduler_active_shards": (),
    "dfkv_rdma_client_endpoint_budget": ("kind",),
    "dfkv_rdma_client_qp_budget": ("kind",),
    "dfkv_rdma_client_wr_slot_budget": ("kind",),
    "dfkv_rdma_client_registered_slot_bytes_budget": ("kind",),
    "dfkv_rdma_client_pool_connections": ("lane", "state", "dev"),
    "dfkv_rdma_client_peer_connections": ("peer", "dev"),
    "dfkv_rdma_client_pool_limit": (),
    "dfkv_rdma_client_topology_nodes": (),
    "dfkv_rdma_client_inline_put_max_bytes": (),
    "dfkv_rdma_client_connection_min_block_bytes": (),
    "dfkv_rdma_client_connection_class_active": ("block_bytes", "depth"),
    "dfkv_rdma_client_connection_class_idle": ("block_bytes", "depth"),
}

CLIENT_METRIC_SPECS: dict[str, MetricSpec] = {
    **{name: MetricSpec("counter", labels) for name, labels in _COUNTERS.items()},
    **{name: MetricSpec("gauge", labels) for name, labels in _GAUGES.items()},
}

_LABEL_RE = re.compile(r'(?:^|,)([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')

def read_native_snapshot(lib, handle, max_attempts: int = 4) -> str:
    """Fetch one complete C-client snapshot, retrying if it grows mid-read."""
    needed = int(lib.dfkv_stats_snapshot(handle, None, 0))
    if needed <= 0:
        return ""
    capacity = needed + 1
    for _ in range(max_attempts):
        buf = ctypes.create_string_buffer(capacity)
        full = int(lib.dfkv_stats_snapshot(handle, buf, capacity))
        if full <= 0:
            return ""
        if full < capacity:
            return buf.value.decode("utf-8", "replace")
        capacity = full + 1
    raise RuntimeError("dfkv client metrics snapshot kept growing during fetch")


def _unescape_label(value: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(value):
        if value[i] != "\\" or i + 1 == len(value):
            out.append(value[i])
            i += 1
            continue
        i += 1
        escaped = value[i]
        out.append("\n" if escaped == "n" else escaped)
        i += 1
    return "".join(out)


def _labels(raw: str) -> dict[str, str] | None:
    if not raw:
        return {}
    parsed: dict[str, str] = {}
    end = 0
    for match in _LABEL_RE.finditer(raw):
        if match.start() != end:
            return None
        parsed[match.group(1)] = _unescape_label(match.group(2))
        end = match.end()
    return parsed if end == len(raw) else None


def parse_snapshot(text: str) -> list[MetricUpdate]:
    """Parse the allow-listed samples from one Prometheus text snapshot."""
    samples: list[MetricUpdate] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.rsplit(None, 1)
        if len(fields) != 2:
            continue
        metric, raw_value = fields
        brace = metric.find("{")
        if brace == -1:
            name, raw_labels = metric, ""
        elif metric.endswith("}"):
            name, raw_labels = metric[:brace], metric[brace + 1 : -1]
        else:
            continue
        spec = CLIENT_METRIC_SPECS.get(name)
        if spec is None:
            continue
        parsed_labels = _labels(raw_labels)
        if parsed_labels is None or any(label not in parsed_labels for label in spec.labels):
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if not math.isfinite(value) or value < 0:
            continue
        samples.append(
            MetricUpdate(name, spec.kind,
                         tuple(parsed_labels[label] for label in spec.labels), value)
        )
    return samples


def sample_key(name: str, labels: tuple[str, ...]) -> str:
    if not labels:
        return name
    spec = CLIENT_METRIC_SPECS[name]
    inner = ",".join(f'{key}="{value}"' for key, value in zip(spec.labels, labels))
    return f"{name}{{{inner}}}"


class SnapshotDelta:
    """Convert cumulative native counters to deltas and retain gauge values.

    ``prepare`` is side-effect free and ``commit`` advances one sample. Pollers
    commit only after the corresponding Prometheus update succeeds, so a failed
    update is replayed rather than silently lost on the next native snapshot.
    ``consume`` preserves the convenient atomic-in-memory API for other callers.
    """

    def __init__(self):
        self._last: dict[tuple[str, tuple[str, ...]], float] = {}
        self._totals: dict[str, float] = {}
        self._gauges: dict[str, float] = {}

    def prepare(self, text: str) -> list[MetricUpdate]:
        updates: list[MetricUpdate] = []
        for sample in parse_snapshot(text):
            key = (sample.name, sample.labels)
            first_observation = key not in self._last
            if sample.kind == "counter":
                previous = self._last.get(key, 0.0)
                delta = (
                    sample.value - previous
                    if sample.value >= previous
                    else sample.value
                )
                # Emit an initial zero once. Prometheus then has a real labeled
                # zero series; absence remains reserved for a missing native
                # metric or a failed poll rather than being ambiguous with zero.
                if delta > 0 or first_observation:
                    updates.append(MetricUpdate(
                        sample.name, sample.kind, sample.labels, delta,
                        source_value=sample.value,
                    ))
            else:
                updates.append(MetricUpdate(
                    sample.name, sample.kind, sample.labels, sample.value,
                    source_value=sample.value,
                ))
        return updates

    def commit(self, update: MetricUpdate) -> None:
        key = (update.name, update.labels)
        display = sample_key(*key)
        source_value = (
            update.value if update.source_value is None else update.source_value
        )
        if update.kind == "counter":
            self._last[key] = source_value
            if update.value > 0:
                self._totals[display] = (
                    self._totals.get(display, 0.0) + update.value
                )
        else:
            self._gauges[display] = source_value

    def consume(self, text: str) -> list[MetricUpdate]:
        updates = self.prepare(text)
        for update in updates:
            self.commit(update)
        return updates

    def totals(self) -> dict[str, float]:
        return dict(self._totals)

    def gauges(self) -> dict[str, float]:
        return dict(self._gauges)
