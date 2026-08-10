from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DASHBOARDS = ROOT / "deploy" / "observability" / "grafana" / "dashboards"
sys.path.insert(0, str(ROOT / "integration" / "common" / "src"))
sys.path.insert(0, str(ROOT / "integration" / "hicache"))

from dfkv_common.client_metrics import CLIENT_METRIC_SPECS  # noqa: E402
from dfkv_telemetry.metrics_push import _client_status_export_name  # noqa: E402

ALERTS = ROOT / "deploy" / "observability" / "alerts.yml"

# Public metric families consumed by the shipped dashboards and alert rules.
# This is deliberately exact: a typo or a producer rename must fail CI instead
# of silently rendering an empty Grafana panel after release.
PUBLIC_FAMILIES = {
    "ALERTS",
    "up",
    "dfkv_accepts_total",
    "dfkv_build_info",
    "dfkv_bytes_read_total",
    "dfkv_bytes_written_total",
    "dfkv_cache_hit_total",
    "dfkv_cache_miss_total",
    "dfkv_cache_put_total",
    "dfkv_client_peer_latency_avg_seconds",
    "dfkv_client_peer_latency_max_seconds",
    "dfkv_connector_bytes_total",
    "dfkv_connector_client_stats_last_success_unixtime",
    "dfkv_connector_client_stats_poll_success",
    "dfkv_connector_dedup_fetches_total",
    "dfkv_connector_dedup_hits_total",
    "dfkv_connector_dedup_wait_hits_total",
    "dfkv_connector_dedup_wait_timeouts_total",
    "dfkv_connector_gpu_dedup_fetches_total",
    "dfkv_connector_gpu_dedup_hits_total",
    "dfkv_connector_gpu_dedup_wait_hits_total",
    "dfkv_connector_gpu_dedup_wait_timeouts_total",
    "dfkv_connector_info",
    "dfkv_connector_keys_total",
    "dfkv_connector_mds_reachable",
    "dfkv_connector_op_bytes_total",
    "dfkv_connector_op_hits_total",
    "dfkv_connector_op_keys_total",
    "dfkv_connector_op_latency_seconds_bucket",
    "dfkv_connector_op_latency_seconds_count",
    "dfkv_connector_op_latency_seconds_sum",
    "dfkv_connector_op_max_seconds",
    "dfkv_connector_op_requests_total",
    "dfkv_connector_op_seconds_bucket",
    "dfkv_connector_op_seconds_count",
    "dfkv_connector_op_seconds_sum",
    "dfkv_connector_ops_total",
    "dfkv_connector_rdma_completion_timeouts_total",
    "dfkv_connector_rdma_cq_errors_total",
    "dfkv_connector_rdma_rail_errors_total",
    "dfkv_connector_ring_members",
    "dfkv_connector_transport_pool_backoff_events_total",
    "dfkv_disk_used_bytes",
    "dfkv_errors_total",
    "dfkv_evicted_bytes_total",
    "dfkv_evictions_total",
    "dfkv_exist_hit_total",
    "dfkv_exist_miss_total",
    "dfkv_mds_etcd_errors_total",
    "dfkv_mds_etcd_request_duration_seconds_bucket",
    "dfkv_mds_etcd_request_errors_total",
    "dfkv_mds_group_capacity_bytes",
    "dfkv_mds_group_clients",
    "dfkv_mds_group_hits_sum",
    "dfkv_mds_group_misses_sum",
    "dfkv_mds_group_stats_missing",
    "dfkv_mds_group_used_bytes",
    "dfkv_mds_heartbeat_healthy",
    "dfkv_mds_last_success_age_seconds",
    "dfkv_mds_heartbeat_failures_consecutive",
    "dfkv_mds_group_version_skew",
    "dfkv_mds_keepalives_total",
    "dfkv_mds_lease_grants_total",
    "dfkv_mds_list_requests_total",
    "dfkv_mds_members",
    "dfkv_mds_ready",
    "dfkv_mds_register_requests_total",
    "dfkv_objects",
    "dfkv_op_latency_seconds_bucket",
    "dfkv_op_latency_seconds_count",
    "dfkv_op_latency_seconds_sum",
    "dfkv_open_connections",
    "dfkv_rdma_active_conns",
    "dfkv_rdma_completion_errors_total",
    "dfkv_rdma_completions_total",
    "dfkv_rdma_idle_reclaims_total",
    "dfkv_rdma_rail_completion_errors_total",
    "dfkv_rdma_rail_get_bytes_total",
    "dfkv_rdma_rail_put_bytes_total",
    "dfkv_rdma_recv_segment_bytes",
    "dfkv_ram_arena_bytes",
    "dfkv_ram_budget_bytes",
    "dfkv_ram_evictions_total",
    "dfkv_ram_flush_backlog",
    "dfkv_ram_flush_dropped_total",
    "dfkv_ram_healthy",
    "dfkv_ram_large_used_bytes",
    "dfkv_ram_put_bypass_total",
    "dfkv_ram_used_bytes",
    "dfkv_read_coalesce_leaders_total",
    "dfkv_read_coalesce_timeouts_total",
    "dfkv_read_coalesced_total",
    "dfkv_rdma_recv_segment_free_bytes",
    "dfkv_rdma_segment_evictions_total",
    "dfkv_server_healthy",
    "dfkv_server_ready",
    "dfkv_server_mds_registration_ready",
    "dfkv_server_startup_complete",
    "dfkv_slab_healthy",
    "dfkv_slab_allocated_bytes",
    "dfkv_slab_capacity_bytes",
    "dfkv_slab_class_allocated_bytes",
    "dfkv_slab_class_fragmentation_bytes",
    "dfkv_slab_class_useful_bytes",
    "dfkv_slab_dio_read_fallback_total",
    "dfkv_slab_dio_write_fallback_total",
    "dfkv_slab_extent_steals_total",
    "dfkv_slab_inflight_keys",
    "dfkv_slab_internal_fragmentation_bytes",
    "dfkv_slab_metadata_io_errors_total",
    "dfkv_slab_payload_bytes",
    "dfkv_slab_prep_holds",
    "dfkv_slab_rebuild_corrupt_records_total",
    "dfkv_slab_rebuild_rejected_records_total",
    "dfkv_slab_unclean_resets_total",
    "dfkv_slab_watermark_evictions_total",
    "dfkv_storage_healthy",
    "dfkv_tcp_max_connections",
    "dfkv_tcp_rejected_connections_total",
    "dfkv_uring_reads_total",
    "dfkv_uptime_seconds",
    "dfkv_uring_init_fallbacks_total",
    "dfkv_used_bytes",
}
_SYNTHETIC_CONNECTOR_FAMILIES = {
    "dfkv_connector_bytes_total",
    "dfkv_connector_client_stats_last_success_unixtime",
    "dfkv_connector_client_stats_poll_success",
    "dfkv_connector_info",
    "dfkv_connector_keys_total",
    "dfkv_connector_op_max_seconds",
    "dfkv_connector_op_seconds_bucket",
    "dfkv_connector_op_seconds_count",
    "dfkv_connector_op_seconds_sum",
    "dfkv_connector_ops_total",
}
_NATIVE_CONNECTOR_FAMILIES = {
    "dfkv_connector_op_bytes_total",
    "dfkv_connector_op_hits_total",
    "dfkv_connector_op_keys_total",
    "dfkv_connector_op_latency_seconds_bucket",
    "dfkv_connector_op_latency_seconds_count",
    "dfkv_connector_op_latency_seconds_sum",
    "dfkv_connector_op_requests_total",
} | {
    _client_status_export_name(source) for source in CLIENT_METRIC_SPECS
}
PUBLIC_FAMILIES = {
    family for family in PUBLIC_FAMILIES
    if not family.startswith("dfkv_connector_")
} | _SYNTHETIC_CONNECTOR_FAMILIES | _NATIVE_CONNECTOR_FAMILIES

_METRIC = re.compile(r"(?<![A-Za-z0-9_:])((?:vllm:)?dfkv_[A-Za-z0-9_:]+|ALERTS|up)\b")


def metric_families(expression: str) -> set[str]:
    # Label keys/values such as job="dfkv_server" are not metric names.
    without_labels = re.sub(r"\{[^{}]*\}", "", expression)
    without_labels = re.sub(
        r"\b(?:by|without|on|ignoring)\s*\([^)]*\)", "", without_labels)
    without_strings = re.sub(r'"(?:\\.|[^"\\])*"', '""', without_labels)
    return set(_METRIC.findall(without_strings))


class ObservabilityContractTest(unittest.TestCase):
    def test_dashboard_queries_use_public_metric_families(self):
        seen_ids: set[tuple[str, int]] = set()
        query_count = 0
        for path in sorted(DASHBOARDS.glob("*.json")):
            dashboard = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(dashboard.get("uid"), path)
            for panel in dashboard.get("panels", []):
                key = (dashboard["uid"], int(panel["id"]))
                self.assertNotIn(key, seen_ids)
                seen_ids.add(key)
                if panel.get("type") == "row":
                    continue
                targets = panel.get("targets", [])
                self.assertTrue(targets, f"{path.name}: panel {panel['id']} has no query")
                for target in targets:
                    expression = target.get("expr", "")
                    self.assertTrue(expression, f"{path.name}: panel {panel['id']} has an empty query")
                    query_count += 1
                    unknown = metric_families(expression) - PUBLIC_FAMILIES
                    self.assertFalse(
                        unknown,
                        f"{path.name}: panel {panel['id']} references unknown metrics {sorted(unknown)}",
                    )
        self.assertGreater(query_count, 30)

    def test_alert_queries_use_public_metric_families(self):
        expressions = [
            line.split("expr:", 1)[1].strip()
            for line in ALERTS.read_text(encoding="utf-8").splitlines()
            if line.lstrip().startswith("expr:")
        ]
        self.assertGreaterEqual(len(expressions), 10)
        for expression in expressions:
            unknown = metric_families(expression) - PUBLIC_FAMILIES
            self.assertFalse(unknown, f"alert references unknown metrics {sorted(unknown)}")

    def test_prometheus_mounts_the_checked_rule_file(self):
        prometheus = (ROOT / "deploy" / "observability" / "prometheus.yml").read_text()
        compose = (ROOT / "deploy" / "observability" / "docker-compose.yml").read_text()
        self.assertIn("/etc/prometheus/alerts.yml", prometheus)
        self.assertIn("./alerts.yml:/etc/prometheus/alerts.yml:ro", compose)
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        self.assertIn(
            '-v "$PWD/deploy/observability:/etc/prometheus:ro"', ci
        )
        self.assertIn("check config /etc/prometheus/prometheus.yml", ci)


    def test_deployment_defaults_are_fail_closed_and_immutable(self):
        compose = (
            ROOT / "deploy" / "observability" / "docker-compose.yml"
        ).read_text(encoding="utf-8")
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertEqual(compose.count("@sha256:"), 4)
        self.assertIn("${DFKV_GRAFANA_ADMIN_PASSWORD:?", compose)
        self.assertNotIn("DFKV_GRAFANA_ADMIN_PASSWORD:-admin", compose)
        self.assertEqual(compose.count("no-new-privileges:true"), 4)
        self.assertEqual(compose.count("cap_drop: [ALL]"), 4)
        self.assertIn("GF_PLUGINS_PREINSTALL_DISABLED=true", compose)
        self.assertIn("GF_ANALYTICS_REPORTING_ENABLED=false", compose)
        self.assertIn("GF_NEWS_NEWS_FEED_ENABLED=false", compose)
        collector = (
            ROOT / "deploy" / "observability" / "otel-collector-config.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("otlp_grpc/tempo", collector)
        self.assertNotIn("otlp/tempo", collector)
        self.assertIn("readers:", collector)
        self.assertIn("port: 8888", collector)
        self.assertIn("DFKV_BASE_IMAGE=", dockerfile)
        self.assertIn("@sha256:", dockerfile)

if __name__ == "__main__":
    unittest.main()
