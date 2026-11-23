import glob
import os
import re
import unittest

from dfkv_common.client_metrics import (
    CLIENT_METRIC_SPECS,
    SnapshotDelta,
    parse_snapshot,
    read_native_snapshot,
)

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "..", "src")
# Client-side families of the native snapshot.
_CLIENT_FAMILY = re.compile(
    r"# TYPE ((?:dfkv_client_|dfkv_rdma_client_|dfkv_transport_pool_|"
    r"dfkv_read_scheduler_|dfkv_rdma_endpoint_cache_|dfkv_rdma_cq_)"
    r"[a-z0-9_]+) (counter|gauge)")
# Native client families deliberately not mirrored, with the reason.
_NOT_MIRRORED: dict[str, str] = {}


class NativeFamiliesAreMirrored(unittest.TestCase):
    """Every counter and gauge family the native client declares is either
    mirrored or listed as deliberately not mirrored. The list lagged: rail
    failure counters (no compatible rail, remote rail health, cross-rail
    retries) never reached the inference server's /metrics."""

    def test_every_declared_family_is_mirrored_or_exempt(self):
        if not os.path.isdir(_SRC):
            self.skipTest("native sources not available")
        declared = {}
        for path in glob.glob(os.path.join(_SRC, "**", "*.[ch]*"),
                              recursive=True):
            with open(path, errors="replace") as source:
                for name, kind in _CLIENT_FAMILY.findall(source.read()):
                    declared[name] = kind
        self.assertIn("dfkv_rdma_client_no_compatible_rail_total", declared)
        missing = sorted(name for name in declared
                         if name not in CLIENT_METRIC_SPECS
                         and name not in _NOT_MIRRORED)
        self.assertEqual(missing, [], "add to client_metrics or _NOT_MIRRORED")
        wrong_kind = sorted(
            name for name, kind in declared.items()
            if name in CLIENT_METRIC_SPECS
            and CLIENT_METRIC_SPECS[name].kind != kind)
        self.assertEqual(wrong_kind, [])

    def test_labelled_rail_families_parse(self):
        text = (
            'dfkv_rdma_client_rail_wc_errors_total'
            '{dev="r0",status="RETRY_EXC_ERR"} 2\n'
            'dfkv_rdma_client_remote_rail_failures_by_rail_total{dev="r1"} 3\n'
            'dfkv_rdma_client_connection_class_active'
            '{block_bytes="1048576",depth="4"} 1\n'
            'dfkv_rdma_client_no_compatible_rail_total 5\n'
        )
        self.assertEqual(
            [(s.name, s.labels, s.value) for s in parse_snapshot(text)],
            [("dfkv_rdma_client_rail_wc_errors_total",
              ("r0", "RETRY_EXC_ERR"), 2.0),
             ("dfkv_rdma_client_remote_rail_failures_by_rail_total",
              ("r1",), 3.0),
             ("dfkv_rdma_client_connection_class_active",
              ("1048576", "4"), 1.0),
             ("dfkv_rdma_client_no_compatible_rail_total", (), 5.0)])


class ClientMetricsTest(unittest.TestCase):
    def test_parser_preserves_bounded_labels_and_values(self):
        text = (
            '# TYPE dfkv_client_op_requests_total counter\n'
            'dfkv_client_op_requests_total{op="get"} 7\n'
            'dfkv_client_peer_errors_total'
            '{peer="node\\\"1\\\\rail\\nA"} 3\n'
            'dfkv_rdma_client_rail_quarantined{dev="mlx5_3"} 1\n'
            'unknown_metric_total 99\n'
        )
        samples = parse_snapshot(text)
        self.assertEqual(
            [(sample.name, sample.labels, sample.value)
             for sample in samples],
            [
                ("dfkv_client_op_requests_total", ("get",), 7.0),
                ("dfkv_client_peer_errors_total",
                 ('node"1\\rail\nA',), 3.0),
                ("dfkv_rdma_client_rail_quarantined",
                 ("mlx5_3",), 1.0),
            ],
        )

    def test_parser_accepts_gpu_dedup_fence_and_stuck_counters(self):
        text = (
            'dfkv_client_gpu_dedup_load_fences_total 12\n'
            'dfkv_client_gpu_dedup_load_fence_failures_total 1\n'
            'dfkv_client_gpu_dedup_invalidate_stuck_total 2\n'
        )
        self.assertEqual(
            [(sample.name, sample.value) for sample in parse_snapshot(text)],
            [
                ("dfkv_client_gpu_dedup_load_fences_total", 12.0),
                ("dfkv_client_gpu_dedup_load_fence_failures_total", 1.0),
                ("dfkv_client_gpu_dedup_invalidate_stuck_total", 2.0),
            ],
        )

    def test_delta_handles_labels_and_native_counter_reset(self):
        delta = SnapshotDelta()
        first = delta.consume(
            'dfkv_client_op_requests_total{op="get"} 5\n'
            'dfkv_client_op_requests_total{op="put"} 2\n'
            'dfkv_client_ring_members 4\n'
        )
        self.assertEqual(
            [update.value for update in first], [5.0, 2.0, 4.0])

        second = delta.consume(
            'dfkv_client_op_requests_total{op="get"} 8\n'
            'dfkv_client_op_requests_total{op="put"} 1\n'
            'dfkv_client_ring_members 3\n'
        )
        self.assertEqual(
            [update.value for update in second], [3.0, 1.0, 3.0])
        self.assertEqual(delta.totals(), {
            'dfkv_client_op_requests_total{op="get"}': 8.0,
            'dfkv_client_op_requests_total{op="put"}': 3.0,
        })
        self.assertEqual(
            delta.gauges(), {"dfkv_client_ring_members": 3.0})

    def test_initial_zero_counter_is_emitted_once(self):
        delta = SnapshotDelta()
        sample = "dfkv_client_io_errors_total 0\n"
        first = delta.consume(sample)
        self.assertEqual(
            [(update.name, update.value) for update in first],
            [("dfkv_client_io_errors_total", 0.0)],
        )
        self.assertEqual(delta.consume(sample), [])
        third = delta.consume("dfkv_client_io_errors_total 2\n")
        self.assertEqual([update.value for update in third], [2.0])
        self.assertEqual(
            delta.totals(), {"dfkv_client_io_errors_total": 2.0})

    def test_parser_accepts_bounded_transport_metrics(self):
        samples = parse_snapshot(
            'dfkv_transport_pool_retirements_total{reason="idle"} 4\n'
            'dfkv_transport_pool_connections 3\n'
            'dfkv_rdma_client_keepalive_attempts_total 9\n'
            'dfkv_rdma_client_keepalive_successes_total 8\n'
            'dfkv_rdma_client_keepalive_failures_total 1\n'
            'dfkv_rdma_client_pool_connections'
            '{lane="sg",state="idle",dev="mlx5_0"} 6\n'
            'dfkv_rdma_client_peer_connections'
            '{peer="dfkv-3",dev="mlx5_0"} 9\n'
            'dfkv_rdma_client_pool_limit 8\n'
        )
        self.assertEqual(
            [(sample.name, sample.labels, sample.value)
             for sample in samples],
            [
                ("dfkv_transport_pool_retirements_total", ("idle",), 4.0),
                ("dfkv_transport_pool_connections", (), 3.0),
                ("dfkv_rdma_client_keepalive_attempts_total", (), 9.0),
                ("dfkv_rdma_client_keepalive_successes_total", (), 8.0),
                ("dfkv_rdma_client_keepalive_failures_total", (), 1.0),
                ("dfkv_rdma_client_pool_connections",
                 ("sg", "idle", "mlx5_0"), 6.0),
                ("dfkv_rdma_client_peer_connections",
                 ("dfkv-3", "mlx5_0"), 9.0),
                ("dfkv_rdma_client_pool_limit", (), 8.0),
            ],
        )

    def test_parser_rejects_malformed_or_unsafe_samples(self):
        self.assertEqual(parse_snapshot(
            'dfkv_client_peer_errors_total 2\n'
            'dfkv_client_ring_members NaN\n'
            'dfkv_client_ring_members -1\n'
            'dfkv_client_ring_members{unexpected="x" 2\n'
        ), [])

    def test_snapshot_fetch_retries_after_growth(self):
        class GrowingLib:
            def __init__(self):
                self.fetches = 0

            def dfkv_stats_snapshot(self, _handle, buf, cap):
                if buf is None:
                    return 4
                self.fetches += 1
                payload = b"abcdefgh"
                buf.value = payload[:cap - 1]
                return len(payload)

        lib = GrowingLib()
        self.assertEqual(read_native_snapshot(lib, object()), "abcdefgh")
        self.assertEqual(lib.fetches, 2)

    def test_snapshot_fetch_rejects_perpetual_growth(self):
        class UnstableLib:
            def dfkv_stats_snapshot(self, _handle, buf, cap):
                if buf is None:
                    return 1
                buf.value = b"x" * (cap - 1)
                return cap

        with self.assertRaisesRegex(RuntimeError, "kept growing"):
            read_native_snapshot(UnstableLib(), object(), max_attempts=2)


if __name__ == "__main__":
    unittest.main()
