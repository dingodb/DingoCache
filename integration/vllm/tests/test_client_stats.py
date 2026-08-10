"""Native client snapshots are mirrored into bounded vLLM metrics."""

import sys
from pathlib import Path

import pytest

_INTEGRATION = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_INTEGRATION / "common" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dfkv_vllm.client_stats import ClientStatsPoller


_HEALTHY = """\
# HELP dfkv_client_ring_members ring size
# TYPE dfkv_client_ring_members gauge
dfkv_client_transport_info{transport="rdma"} 1
dfkv_client_ring_members 5
dfkv_client_mds_reachable 1
dfkv_client_mds_unreachable_polls_total 0
dfkv_client_ops_served_total 123456
dfkv_client_op_requests_total{op="get"} 19
dfkv_rdma_client_rail_selections_total{dev="ib7s400p3"} 11
dfkv_rdma_client_rail_quarantined{dev="ib7s400p3"} 0
"""

_EMPTY_RING = """\
dfkv_client_transport_info{transport="rdma"} 1
dfkv_client_ring_members 0
dfkv_client_mds_reachable 0
dfkv_client_mds_unreachable_polls_total 7
"""


def test_parses_health_operations_and_rail_metrics():
    poller = ClientStatsPoller(lambda: _HEALTHY, tp_rank=0, interval_s=0)
    poller.poll_once()
    assert poller.gauges() == {
        'dfkv_client_transport_info{transport="rdma"}': 1.0,
        "dfkv_client_ring_members": 5.0,
        "dfkv_client_mds_reachable": 1.0,
        'dfkv_rdma_client_rail_quarantined{dev="ib7s400p3"}': 0.0,
    }
    assert poller.totals() == {
        "dfkv_client_ops_served_total": 123456.0,
        'dfkv_client_op_requests_total{op="get"}': 19.0,
        'dfkv_rdma_client_rail_selections_total{dev="ib7s400p3"}': 11.0,
    }
    assert poller.health()["success"] == 1
    assert poller.health()["timestamp_seconds"] > 0


def test_empty_ring_and_counter_deltas():
    snapshots = iter((_EMPTY_RING, _EMPTY_RING.replace(" 7", " 9")))
    poller = ClientStatsPoller(lambda: next(snapshots), tp_rank=3, interval_s=0)
    poller.poll_once()
    poller.poll_once()
    assert poller.gauges()["dfkv_client_ring_members"] == 0
    assert poller.gauges()["dfkv_client_mds_reachable"] == 0
    assert poller.totals()["dfkv_client_mds_unreachable_polls_total"] == 9


def test_missing_metrics_are_absent_not_false_zeroes():
    poller = ClientStatsPoller(
        lambda: "dfkv_client_ring_members 2\\n", tp_rank=0, interval_s=0)
    poller.poll_once()
    assert poller.gauges() == {"dfkv_client_ring_members": 2.0}


@pytest.mark.parametrize(
    "text",
    ("", None, "garbage-with-no-space\\n",
     "dfkv_client_ring_members not-a-number\\n"),
)
def test_empty_or_malformed_snapshot_marks_poller_failed(text):
    poller = ClientStatsPoller(lambda: text, tp_rank=0, interval_s=0)
    with pytest.raises(RuntimeError):
        poller.poll_once()
    assert poller.health()["success"] == 0
    assert poller.health()["errors_total"] == 1


def test_zero_interval_disables_thread():
    poller = ClientStatsPoller(lambda: _HEALTHY, tp_rank=0, interval_s=0)
    poller.start()
    assert poller._thread is None
    poller.stop()


def test_prometheus_metrics_reflect_native_snapshot():
    prometheus_client = pytest.importorskip("prometheus_client")
    poller = ClientStatsPoller(lambda: _HEALTHY, tp_rank=77, interval_s=60)
    poller.start()
    registry = prometheus_client.REGISTRY
    assert registry.get_sample_value(
        "vllm:dfkv_client_ring_members", {"tp_rank": "77"}) == 5
    assert registry.get_sample_value(
        "vllm:dfkv_client_ops_served_total", {"tp_rank": "77"}) == 123456
    assert registry.get_sample_value(
        "vllm:dfkv_client_op_requests_total",
        {"tp_rank": "77", "op": "get"}) == 19
    assert registry.get_sample_value(
        "vllm:dfkv_rdma_client_rail_selections_total",
        {"tp_rank": "77", "dev": "ib7s400p3"}) == 11
    assert registry.get_sample_value(
        "vllm:dfkv_client_transport_info",
        {"tp_rank": "77", "transport": "rdma"}) == 1
    assert registry.get_sample_value(
        "vllm:dfkv_client_stats_snapshot_success",
        {"tp_rank": "77"}) == 1
    poller.stop()
