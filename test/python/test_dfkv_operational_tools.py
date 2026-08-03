#!/usr/bin/env python3
"""Deterministic, cluster-free tests for deploy operational automation."""

from __future__ import annotations

import math
import struct
import subprocess
import sys
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
sys.path.insert(0, str(DEPLOY))

from dfkv_load_regression import bench_once, histogram_delta, histogram_quantile, percentile  # noqa: E402
from dfkv_membership_audit import decode_registration, info_fields, range_end  # noqa: E402
from dfkv_node_replace import Replacer, WorkflowError  # noqa: E402
from dfkv_tenant_quota import load_quotas, main as quota_main  # noqa: E402
from dfkv_ops_common import (  # noqa: E402
    members_epoch,
    parse_bench,
    parse_clients,
    parse_ring,
    parse_stat_reachability,
)


RING = """group=glm members=2 ring_points=300
ID               ADDR                   WEIGHT   VNODES   SHARE  INFO
n1               10.0.0.1:28001             1      100   33.3%  ver=2.0,engine=slab,disks=3,cap=9,ram=0,rdma=on
n2               10.0.0.2:28001             2      200   66.7%  ver=2.0,engine=slab,disks=3,cap=9,ram=0,rdma=on
"""

CLIENTS = """group=glm clients=2 (only upgraded clients register)
ID                           TYPE       MODEL                    ROLE           TP       INFO
host-a_1                     vllm       model-a                  producer       8        type=vllm,model=model-a
host-b_1                     sglang     model-a                  -              8        type=sglang,model=model-a
"""

STATS = """ID             ADDR                    USED_MB   OBJECTS      HITS    MISSES   HIT%
n1             10.0.0.1:28000             1.0        10         9         1  90.0%
n2             10.0.0.2:28000       (unreachable)
TOTAL used=1.0MB objects=10 hits=9 misses=1 hit%=90.0 bytes_w=1 bytes_r=1
"""

BENCH = """dfkv_bench transport=tcp members=2
PUT   n=8000 size=4096 threads=8 batch=4 | 1.000s  0.03 GB/s  8000 ops/s  call-lat ms p50=0.100 p99=0.900 max=1.100  fails=0
GET   n=8000 size=4096 threads=8 batch=4 | 0.500s  0.07 GB/s  16000 ops/s  call-lat ms p50=0.050 p99=0.400 max=0.800  fails=2
"""


class CommonParserTest(unittest.TestCase):
    def test_ring_clients_and_readiness(self) -> None:
        view = parse_ring(RING)
        self.assertEqual(view.group, "glm")
        self.assertEqual([member.node_id for member in view.members], ["n1", "n2"])
        self.assertEqual(view.epoch, members_epoch(reversed(view.members)))
        self.assertNotEqual(view.epoch, 0)
        self.assertEqual(parse_clients(CLIENTS), ("glm", ("host-a_1", "host-b_1")))
        self.assertEqual(parse_stat_reachability(STATS), {"n1": True, "n2": False})

    def test_bench_parser_rejects_partial_output(self) -> None:
        parsed = parse_bench(BENCH)
        self.assertEqual(parsed["put"]["throughput_gbps"], 0.03)
        self.assertEqual(parsed["get"]["fails"], 2)
        with self.assertRaises(ValueError):
            parse_bench(BENCH.split("GET", 1)[0])


class TenantQuotaToolTest(unittest.TestCase):
    def test_set_list_remove_round_trips_strict_server_format(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tenant-quotas"
            self.assertEqual(
                quota_main([
                    "--file", str(path), "set",
                    "--tenant", "tenant-a", "1234",
                ]),
                0,
            )
            self.assertEqual(
                load_quotas(path), {"6164af84acbc9ade": 1234})
            self.assertEqual(
                quota_main([
                    "--file", str(path), "set",
                    "--hash", "0000000000000001", "0",
                ]),
                0,
            )
            self.assertEqual(
                load_quotas(path),
                {"0000000000000001": 0, "6164af84acbc9ade": 1234},
            )
            self.assertEqual(
                quota_main([
                    "--file", str(path), "remove",
                    "--tenant", "tenant-a",
                ]),
                0,
            )
            self.assertEqual(load_quotas(path), {"0000000000000001": 0})

    def test_malformed_existing_file_is_never_rewritten(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tenant-quotas"
            original = "ABCDEF0123456789 10\\n"
            path.write_text(original, encoding="utf-8")
            self.assertEqual(
                quota_main([
                    "--file", str(path), "set",
                    "--tenant", "tenant-a", "1234",
                ]),
                2,
            )
            self.assertEqual(path.read_text(encoding="utf-8"), original)

class ReplacementWorkflowTest(unittest.TestCase):
    @staticmethod
    def args(dry_run: bool = True) -> SimpleNamespace:
        return SimpleNamespace(
            old_id="n1", new_id="n2", old_host="old", new_host="new",
            dry_run=dry_run, min_clients=0, timeout=10.0, command_timeout=1.0,
            poll_interval=0.01, stable_samples=1, state_file=None,
            dfkvctl="dfkvctl", mds="mds:1", group="glm", ssh="ssh", service="dfkv",
        )

    def test_dry_run_with_ready_replacement_never_stops_old(self) -> None:
        class Fixture(Replacer):
            def __init__(self, args):
                super().__init__(args)
                self.actions = []

            def ring(self):
                return parse_ring(RING)

            def clients(self):
                return {"client-1"}

            def healthy(self, node_id):
                return True

            def wait_stable_ring(self, required, forbidden):
                return parse_ring(RING)

            def remote(self, host, action):
                self.actions.append((host, action))

        workflow = Fixture(self.args())
        workflow.run()
        self.assertEqual(workflow.actions, [])
        self.assertFalse(workflow.old_stopped)

    def test_rollback_only_requires_old_node_to_return(self) -> None:
        class Fixture(Replacer):
            def __init__(self, args):
                super().__init__(args)
                self.required = []
                self.actions = []

            def remote(self, host, action):
                self.actions.append((host, action))

            def wait_stable_ring(self, required, forbidden):
                self.required.append(required)
                return parse_ring(RING)

            def wait_for(self, description, predicate):
                return True

        workflow = Fixture(self.args(dry_run=False))
        workflow.old_stopped = True
        workflow.rollback(RuntimeError("cutover failed"))
        self.assertEqual(workflow.actions, [("old", "start")])
        self.assertEqual(workflow.required, [{"n1"}])

    def test_wait_for_retries_command_timeouts_until_predicate_succeeds(self) -> None:
        transient = subprocess.TimeoutExpired(["dfkvctl", "ring"], 1.0)
        with patch("dfkv_node_replace.run_command", side_effect=[transient, transient, RING]) as command:
            workflow = Replacer(self.args())
            view = workflow.wait_for("replacement ring", workflow.ring)
        self.assertEqual([member.node_id for member in view.members], ["n1", "n2"])
        self.assertEqual(command.call_count, 3)

    def test_wait_for_deadline_still_fails_under_persistent_timeouts(self) -> None:
        args = self.args()
        args.timeout = 0.05
        transient = subprocess.TimeoutExpired(["dfkvctl", "ring"], 1.0)
        with patch("dfkv_node_replace.run_command", side_effect=transient) as command:
            workflow = Replacer(args)
            with self.assertRaises(WorkflowError) as raised:
                workflow.wait_for("replacement ring", workflow.ring)
        self.assertIn("timed out", str(raised.exception))
        self.assertGreaterEqual(command.call_count, 2)

    def test_wait_for_retries_os_errors_as_transient(self) -> None:
        reset = OSError(104, "connection reset by peer")
        with patch("dfkv_node_replace.run_command", side_effect=[reset, reset, RING]) as command:
            workflow = Replacer(self.args())
            view = workflow.wait_for("replacement ring", workflow.ring)
        self.assertEqual([member.node_id for member in view.members], ["n1", "n2"])
        self.assertEqual(command.call_count, 3)


class MembershipDecodeTest(unittest.TestCase):
    @staticmethod
    def registration() -> bytes:
        node_id = b"node-7"
        ip = b"10.2.3.7"
        info = b"ver=2.0,engine=slab,disks=4,cap=9,ram=0,rdma=on"
        result = struct.pack("=QI", 0, 1)
        result += struct.pack("=I", len(node_id)) + node_id
        result += struct.pack("=I", len(ip)) + ip
        result += struct.pack("=II", 28001, 3)
        result += struct.pack("=II", 0x54435031, 28000)
        result += struct.pack("=II", 0x4E464F31, len(info)) + info
        result += struct.pack("=I", 0x31415453) + bytes([12, 0])
        return result

    def test_registration_identity_and_info(self) -> None:
        member = decode_registration(self.registration())
        self.assertEqual((member.node_id, member.address, member.weight), ("node-7", "10.2.3.7:28001", 3))
        self.assertEqual(info_fields(member.info)["engine"], "slab")
        with self.assertRaises(ValueError):
            decode_registration(self.registration()[:15])

    def test_prefix_range_end(self) -> None:
        self.assertEqual(range_end(b"/members/"), b"/members0")
        self.assertEqual(range_end(b"\xff"), b"\0")


class LoadRegressionMathTest(unittest.TestCase):
    def test_histogram_delta_and_quantiles(self) -> None:
        before = {
            "put": {0.001: 10, 0.01: 20, math.inf: 20},
            "get": {0.001: 1, 0.01: 2, math.inf: 2},
        }
        after = {
            "put": {0.001: 60, 0.01: 120, math.inf: 120},
            "get": {0.001: 51, 0.01: 102, math.inf: 102},
        }
        delta = histogram_delta(before, after)
        self.assertAlmostEqual(histogram_quantile(delta["put"], 0.50), 0.001)
        self.assertGreater(histogram_quantile(delta["put"], 0.95), 0.001)
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.95), 2.9)

    def test_histogram_reset_fails_closed(self) -> None:
        before = {"put": {0.1: 2, math.inf: 2}, "get": {0.1: 2, math.inf: 2}}
        after = {"put": {0.1: 1, math.inf: 1}, "get": {0.1: 3, math.inf: 3}}
        with self.assertRaises(ValueError):
            histogram_delta(before, after)

    def test_bench_exit_one_preserves_measured_error_rate(self) -> None:
        args = SimpleNamespace(
            baseline_bench=None, dfkv_bench="dfkv_bench",
            baseline_members="n=127.0.0.1:1", baseline_mds=None,
            baseline_group="default", size=4096, threads=8, batch=4, bc=1,
            ready_timeout=30.0, run_timeout=60.0,
        )
        completed = Mock(returncode=1, stdout=BENCH, stderr="")
        with patch("dfkv_load_regression.subprocess.run", return_value=completed):
            phases = bench_once(
                args, "baseline", 8000, "seed", allow_operation_failures=True,
            )
            self.assertEqual(phases["get"]["fails"], 2)
            with self.assertRaises(RuntimeError):
                bench_once(
                    args, "baseline", 8000, "warmup",
                    allow_operation_failures=False,
                )


if __name__ == "__main__":
    unittest.main()
