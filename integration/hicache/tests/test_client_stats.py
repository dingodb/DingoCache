"""Native dfkv metrics are mirrored into SGLang's Prometheus registry."""

import sys
from pathlib import Path

import pytest

_INTEGRATION = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_INTEGRATION / "common" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dfkv_metrics import ClientStatsPoller, Metrics


_SNAPSHOT = """\
dfkv_client_transport_info{transport="rdma"} 1
dfkv_client_ring_members 1
dfkv_client_mds_reachable 1
dfkv_client_ops_served_total 42
dfkv_client_op_requests_total{op="get"} 8
dfkv_client_io_errors_total 2
dfkv_client_peer_errors_total{peer="192.168.5.1:28101"} 3
dfkv_rdma_client_rail_selections_total{dev="ib7s400p0"} 17
dfkv_rdma_client_rail_put_ops_total{dev="ib7s400p0"} 4
dfkv_rdma_client_rail_put_bytes_total{dev="ib7s400p0"} 4096
dfkv_rdma_client_rail_get_ops_total{dev="ib7s400p0"} 3
dfkv_rdma_client_rail_get_bytes_total{dev="ib7s400p0"} 3072
dfkv_rdma_client_rail_errors_total{dev="ib7s400p0"} 1
dfkv_rdma_client_mr_registered_bytes 134217728
dfkv_rdma_client_completion_timeouts_total 4
"""


def test_poller_tracks_native_families_and_reset_safe_deltas():
    snapshots = iter((_SNAPSHOT, _SNAPSHOT.replace(
        "dfkv_client_ops_served_total 42", "dfkv_client_ops_served_total 5")))
    poller = ClientStatsPoller(lambda: next(snapshots), tp_rank=2, interval_s=0)
    poller.poll_once()
    poller.poll_once()

    assert poller.gauges()["dfkv_client_ring_members"] == 1
    assert poller.gauges()["dfkv_rdma_client_mr_registered_bytes"] == 134217728
    assert poller.totals()["dfkv_client_ops_served_total"] == 47
    assert poller.totals()['dfkv_client_op_requests_total{op="get"}'] == 8
    assert poller.totals()[
        'dfkv_rdma_client_rail_errors_total{dev="ib7s400p0"}'] == 1
    assert poller.totals()[
        'dfkv_rdma_client_rail_put_bytes_total{dev="ib7s400p0"}'] == 4096
    assert poller.totals()[
        'dfkv_rdma_client_rail_get_bytes_total{dev="ib7s400p0"}'] == 3072
    assert poller.health()["success"] == 1



def test_hicache_identity_and_exist_outcomes_are_observable():
    metrics = Metrics(tp_rank=913)
    metrics.set_identity(
        physical_rank=6,
        pcp_rank=6,
        dcp_rank=0,
        storage_pcp_rank=0,
        primary_dev="ib7s400p6",
    )
    metrics.on_exists(8, 0, 0, seconds=0.001)
    metrics.on_exists(8, 6, 4, seconds=0.002)
    metrics.on_exists(8, 8, 8, seconds=0.003)
    snapshot = metrics.snapshot()
    assert snapshot["identity"] == {
        "tp_rank": "913",
        "physical_rank": "6",
        "pcp_rank": "6",
        "dcp_rank": "0",
        "storage_pcp_rank": "0",
        "primary_dev": "ib7s400p6",
    }
    assert snapshot["exist_calls"] == 3
    assert snapshot["exist_probe_pages"] == 24
    assert snapshot["exist_present_pages"] == 14
    assert snapshot["exist_contiguous_pages"] == 12
    assert snapshot["exist_result_full_miss"] == 1
    assert snapshot["exist_result_partial_prefix"] == 1
    assert snapshot["exist_result_full_hit"] == 1
    assert snapshot["exist_observations"] == 3

def test_failed_snapshot_is_observable_and_recovery_clears_failure():
    snapshots = iter(("", _SNAPSHOT))
    poller = ClientStatsPoller(lambda: next(snapshots), tp_rank=4, interval_s=0)
    with pytest.raises(RuntimeError):
        poller.poll_once()
    assert poller.health() == {
        "success": 0,
        "timestamp_seconds": 0.0,
        "errors_total": 1,
    }
    poller.poll_once()
    assert poller.health()["success"] == 1
    assert poller.health()["errors_total"] == 1
    assert poller.health()["timestamp_seconds"] > 0


def test_prometheus_scrape_contains_native_failure_and_rail_signals():
    prometheus_client = pytest.importorskip("prometheus_client")
    poller = ClientStatsPoller(lambda: _SNAPSHOT, tp_rank=78, interval_s=60)
    poller.start()
    registry = prometheus_client.REGISTRY
    assert registry.get_sample_value(
        "dfkv_client_ops_served_total", {"tp_rank": "78"}) == 42
    assert registry.get_sample_value(
        "dfkv_client_peer_errors_total",
        {"tp_rank": "78", "peer": "192.168.5.1:28101"}) == 3
    assert registry.get_sample_value(
        "dfkv_rdma_client_rail_errors_total",
        {"tp_rank": "78", "dev": "ib7s400p0"}) == 1
    assert registry.get_sample_value(
        "dfkv_rdma_client_completion_timeouts_total",
        {"tp_rank": "78"}) == 4
    assert registry.get_sample_value(
        "dfkv_client_stats_snapshot_success", {"tp_rank": "78"}) == 1
    poller.stop()
