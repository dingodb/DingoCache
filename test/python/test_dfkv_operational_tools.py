#!/usr/bin/env python3
"""Deterministic, cluster-free tests for deploy operational automation."""

from __future__ import annotations

import hashlib
import json
import math
import os
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
from dfkv_node_replace import Replacer, WorkflowError, main as replace_main  # noqa: E402
from dfkv_tenant_quota import load_quotas, main as quota_main  # noqa: E402
from dfkv_ops_common import (  # noqa: E402
    atomic_write,
    atomic_write_with_checksum,
    parse_bench,
    parse_bench_diagnostics,
    parse_clients,
    parse_ring,
    parse_stat_reachability,
)


# `dfkvctl ring --json` documents. test_dfkvctl_all_groups.py feeds the real
# binary's output through the same parsers; these only drive the workflows.
INFO = "ver=2.0,engine=slab,disks=3,cap=9,ram=0,rdma=on"


def ring_member(node_id: str, ip: str, weight: int, vnodes: int, share: float,
                *, eligible: bool = True, info: str = INFO) -> dict:
    return {
        "id": node_id, "ip": ip, "port": 28001,
        "status": "ACTIVE" if eligible else "DEGRADED", "eligible": eligible,
        "weight": weight, "vnodes": vnodes, "share_percent": share,
        "ib_health": "unknown", "info": info,
    }


def ring_document(epoch: int, *members: dict) -> str:
    return json.dumps({
        "group": "glm", "ring_points": 300 if members else 0,
        "placement_epoch": str(epoch), "members": list(members),
    })


N1 = ring_member("n1", "10.0.0.1", 1, 100, 33.3)
N2 = ring_member("n2", "10.0.0.2", 2, 200, 66.7)
N1_ALONE = ring_member("n1", "10.0.0.1", 1, 300, 100.0)
RING = ring_document(2**64 - 59, N1, N2)
RING_OLD = ring_document(7, N1_ALONE)
# n2 is registered but held out of placement: it owns no vnodes.
RING_DEGRADED = ring_document(
    7, N1_ALONE, ring_member("n2", "10.0.0.2", 2, 0, 0.0, eligible=False, info=""))
EMPTY_RING = ring_document(1469598103934665603)

# What a dfkvctl without --json prints for the same command: the table.
RING_TABLE = f"""group=glm members=1 eligible=1 ring_points=300
ID               ADDR                   STATUS    WEIGHT   VNODES   SHARE IB-HEALTH                      INFO
n1               10.0.0.1:28001         ACTIVE         1      300  100.0% unknown                        {INFO}
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
PUT   n=8000 size=4096 threads=8 batch=4 | 1.000s  goodput 0.03 GB/s  8000 ok-ops/s  call-lat ms p50=0.100 p99=0.900 max=1.100  ok=8000 fails=0
DIAG phase=PUT ring=2 io_errors=0 unhealthy_skips=0 peer_bad=0 endpoint_errors=0 rail_errors=0 cq_errors=0 completion_timeouts=0 stale_retries=0 mr_rejections=0 mr_failures=0 adhoc_mr=0 mr_registered_bytes=134217728
GET   n=8000 size=4096 threads=8 batch=4 | 0.500s  goodput 0.07 GB/s  15996 ok-ops/s  call-lat ms p50=0.050 p99=0.400 max=0.800  ok=7998 fails=2
DIAG phase=GET ring=2 io_errors=2 unhealthy_skips=0 peer_bad=1 endpoint_errors=1 rail_errors=0 cq_errors=0 completion_timeouts=0 stale_retries=1 mr_rejections=0 mr_failures=0 adhoc_mr=0 mr_registered_bytes=134217728
"""
LEGACY_BENCH = """dfkv_bench transport=tcp members=2
PUT   n=8000 size=4096 threads=8 batch=4 | 1.000s 0.03 GB/s 8000 ops/s call-lat ms p50=0.100 p99=0.900 max=1.100 ok=8000 fails=0
GET   n=8000 size=4096 threads=8 batch=4 | 0.500s 0.07 GB/s 15996 ops/s call-lat ms p50=0.050 p99=0.400 max=0.800 ok=7998 fails=2
"""





class CommonParserTest(unittest.TestCase):
    def test_ring_clients_and_readiness(self) -> None:
        view = parse_ring(RING)
        self.assertEqual(view.group, "glm")
        self.assertEqual(view.ring_points, 300)
        self.assertEqual([member.node_id for member in view.members], ["n1", "n2"])
        self.assertEqual(
            [(m.address, m.ip, m.port, m.weight, m.vnodes, m.share_percent, m.info, m.eligible)
             for m in view.members],
            [("10.0.0.1:28001", "10.0.0.1", 28001, 1, 100, 33.3, INFO, True),
             ("10.0.0.2:28001", "10.0.0.2", 28001, 2, 200, 66.7, INFO, True)],
        )
        # The epoch is dfkvctl's, bit for bit: 2**64 - 59 does not fit a double.
        self.assertEqual(view.epoch, 2**64 - 59)
        self.assertEqual(parse_clients(CLIENTS), ("glm", ("host-a_1", "host-b_1")))
        self.assertEqual(parse_stat_reachability(STATS), {"n1": True, "n2": False})
        empty = parse_ring(EMPTY_RING)
        self.assertEqual(empty.group, "glm")
        self.assertEqual(empty.members, ())

    def test_ring_eligibility_and_missing_self_description(self) -> None:
        active, degraded = parse_ring(RING_DEGRADED).members
        self.assertTrue(active.eligible)
        self.assertFalse(degraded.eligible)
        self.assertEqual(degraded.info, "-")
        self.assertNotEqual(
            parse_ring(RING_DEGRADED).fingerprint(),
            parse_ring(ring_document(7, N1_ALONE, ring_member("n2", "10.0.0.2", 2, 0, 0.0, info=""))).fingerprint(),
        )

    def test_ring_parser_rejects_anything_but_the_json_document(self) -> None:
        document = json.loads(RING)
        del document["members"][0]["vnodes"]
        duplicate = json.loads(RING)
        duplicate["members"][1]["id"] = "n1"
        for text in (RING_TABLE, "", "[]", json.dumps(document), json.dumps(duplicate),
                     RING.replace(str(2**64 - 59), "soon")):
            with self.assertRaises(ValueError, msg=text):
                parse_ring(text)


    def test_bench_parser_rejects_partial_output(self) -> None:
        parsed = parse_bench(BENCH)
        self.assertEqual(parsed["put"]["throughput_gbps"], 0.03)
        self.assertEqual(parsed["get"]["fails"], 2)
        self.assertEqual(parsed["get"]["successful_operations"], 7998)
        with self.assertRaises(ValueError):
            parse_bench(BENCH.split("GET", 1)[0])
        diagnostics = parse_bench_diagnostics(BENCH)
        self.assertEqual(diagnostics["get"]["io_errors"], 2)
        self.assertEqual(diagnostics["get"]["mr_registered_bytes"], 134217728)
        with self.assertRaises(ValueError):
            parse_bench_diagnostics(BENCH.split("GET", 1)[0])
        self.assertEqual(
            parse_bench_diagnostics(BENCH.replace(
                BENCH.splitlines()[2] + "\n", "").replace(
                BENCH.splitlines()[4] + "\n", ""), required=False),
            {},
        )
        legacy = parse_bench(LEGACY_BENCH)
        self.assertEqual(legacy["put"]["throughput_gbps"], 0.03)
        self.assertEqual(legacy["get"]["successful_operations"], 7998)



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

    def test_atomic_rewrite_preserves_mode_and_owner(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text("old", encoding="utf-8")
            path.chmod(0o640)
            before = path.stat()
            atomic_write(path, b"new")
            after = path.stat()
            self.assertEqual(after.st_mode & 0o777, 0o640)
            self.assertEqual((after.st_uid, after.st_gid),
                             (before.st_uid, before.st_gid))
            self.assertEqual(path.read_bytes(), b"new")


class FakeFleet:
    """Stands in for run_command: one MDS group and two hosts running the unit.

    A stop takes ``stop_polls`` state queries; until it is over the server is
    still registered and still answers, as one draining for tens of seconds
    is. systemctl without --no-block waits for its job, so under a command
    timeout shorter than the stop the caller times out while the job stays
    queued -- a start queued behind a stop runs once the stop is over.
    """

    MEMBER = {"old": N1, "new": N2}

    def __init__(self, stop_polls: int = 5) -> None:
        self.state = {"old": "active", "new": "inactive"}
        self.stop_polls = stop_polls
        self.stopping: dict[str, int] = {}
        self.start_queued: set[str] = set()
        self.jobs: list[tuple[str, ...]] = []

    def run_command(self, argv, timeout, **_) -> str:
        argv = list(argv)
        if argv[0] == "dfkvctl":
            return self.dfkvctl(argv[1:])
        host = argv[argv.index("--") + 1]
        arguments = argv[argv.index("systemctl") + 1:-1]
        if arguments[0] == "show":
            return f"ActiveState={self.poll(host)}\n"
        self.jobs.append((host, *arguments))
        waits = False
        if arguments[0] == "stop" and self.state[host] == "active":
            self.state[host] = "deactivating"
            self.stopping[host] = self.stop_polls
            waits = True
        elif arguments[0] == "start" and self.state[host] == "deactivating":
            self.start_queued.add(host)
            waits = True
        elif arguments[0] == "start":
            self.state[host] = "active"
        if waits and "--no-block" not in arguments:
            raise subprocess.TimeoutExpired(argv, timeout)
        return ""

    def poll(self, host: str) -> str:
        if self.state[host] == "deactivating":
            self.stopping[host] -= 1
            if self.stopping[host] <= 0:
                self.state[host] = "active" if host in self.start_queued else "inactive"
                self.start_queued.discard(host)
        return self.state[host]

    def dfkvctl(self, arguments: list[str]) -> str:
        up = [self.MEMBER[host] for host in ("old", "new") if self.state[host] != "inactive"]
        if arguments[0] == "ring":
            return ring_document(len(up), *up) if "--json" in arguments else RING_TABLE
        if arguments[0] == "clients":
            return CLIENTS
        rows = "".join(f"{member['id']} {member['ip']}:28000 2.0 yes\n" for member in up)
        return "ID ADDR VERSION READY\n" + rows + "TOTAL used=0.0MB\n"


class ReplacementWorkflowTest(unittest.TestCase):
    ARGV = [
        "--mds", "mds:1", "--group", "glm", "--old-id", "n1", "--new-id", "n2",
        "--old-host", "old", "--new-host", "new", "--timeout", "10",
        "--command-timeout", "1", "--poll-interval", "0.01", "--stable-samples", "1",
    ]

    def setUp(self) -> None:
        # No test may reach a real dfkvctl or ssh.
        blocked = patch(
            "dfkv_node_replace.run_command", side_effect=AssertionError("unmocked command"))
        blocked.start()
        self.addCleanup(blocked.stop)

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

    def test_start_timeout_marks_started_new_and_rollback_stops_replacement(self) -> None:
        class Fixture(Replacer):
            def __init__(self, args):
                super().__init__(args)
                self.actions = []

            def ring(self):
                return parse_ring(RING_OLD)

            def clients(self):
                return {"client-1"}

            def remote(self, host, action):
                self.actions.append((host, action))
                if (host, action) == ("new", "start"):
                    raise subprocess.TimeoutExpired(["ssh", "new"], 1.0)

            def unit_state(self, host):
                return "inactive"

        workflow = Fixture(self.args(dry_run=False))
        with self.assertRaises(subprocess.TimeoutExpired):
            workflow.run()
        # The local SSH timeout must not hide a remotely running replacement.
        self.assertTrue(workflow.started_new)
        self.assertFalse(workflow.old_stopped)
        workflow.rollback(RuntimeError("start ssh timed out"))
        self.assertEqual(workflow.actions, [("new", "start"), ("new", "stop")])
        events = workflow.events
        self.assertTrue(any(event["phase"] == "rollback" for event in events))
        self.assertFalse(any("safe abort" in str(event["message"]) for event in events))

    def test_successful_cutover_keeps_started_new_without_rollback(self) -> None:
        class Fixture(Replacer):
            def __init__(self, args):
                super().__init__(args)
                self.actions = []

            def ring(self):
                return parse_ring(RING_OLD)

            def clients(self):
                return {"client-1"}

            def remote(self, host, action):
                self.actions.append((host, action))

            def wait_stable_ring(self, required, forbidden):
                return parse_ring(RING)

            def wait_for(self, description, predicate):
                return True

        workflow = Fixture(self.args(dry_run=False))
        workflow.run()
        self.assertEqual(workflow.actions, [("new", "start"), ("old", "stop")])
        self.assertTrue(workflow.started_new)
        self.assertTrue(workflow.old_stopped)
        phases = [event["phase"] for event in workflow.events]
        self.assertNotIn("rollback", phases)
        self.assertNotIn("abort", phases)

    def test_rollback_fails_loudly_when_replacement_stop_fails(self) -> None:
        class Fixture(Replacer):
            def remote(self, host, action):
                raise subprocess.TimeoutExpired(["ssh", host], 1.0)

        workflow = Fixture(self.args(dry_run=False))
        workflow.started_new = True
        with self.assertRaises(WorkflowError) as raised:
            workflow.rollback(RuntimeError("start ssh timed out"))
        message = str(raised.exception)
        self.assertIn("CRITICAL", message)
        self.assertIn("still be registered in the ring", message)
        self.assertIn("stop dfkv on new manually", message)
        self.assertFalse(any("safe abort" in str(event["message"]) for event in workflow.events))

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

    def test_stop_longer_than_the_command_timeout_is_observed_not_awaited(self) -> None:
        fleet = FakeFleet()
        with patch("dfkv_node_replace.run_command", side_effect=fleet.run_command):
            status = replace_main(self.ARGV)
        self.assertEqual(status, 0)
        self.assertEqual(
            fleet.jobs, [("new", "start", "--no-block"), ("old", "stop", "--no-block")])
        self.assertEqual(fleet.state, {"old": "inactive", "new": "active"})

    def test_rollback_start_waits_behind_a_stop_in_progress(self) -> None:
        fleet = FakeFleet()
        fleet.state = {"old": "deactivating", "new": "active"}
        fleet.stopping["old"] = fleet.stop_polls
        with patch("dfkv_node_replace.run_command", side_effect=fleet.run_command):
            workflow = Replacer(self.args(dry_run=False))
            workflow.old_stopped = True
            workflow.rollback(RuntimeError("cutover failed"))
        self.assertEqual(fleet.jobs, [("old", "start", "--no-block")])
        # Not declared restored on the registration of the process going away.
        self.assertEqual(fleet.state["old"], "active")
        self.assertIn("old node restored", str(workflow.events[-1]["message"]))

    def test_stable_ring_requires_placed_members_and_forbids_registered_ones(self) -> None:
        args = self.args()
        args.timeout = 0.05
        with patch("dfkv_node_replace.run_command", return_value=RING_DEGRADED):
            workflow = Replacer(args)
            self.assertEqual(workflow.wait_stable_ring({"n1"}, set()).group, "glm")
            with self.assertRaises(WorkflowError):
                workflow.wait_stable_ring({"n1", "n2"}, set())
            with self.assertRaises(WorkflowError):
                workflow.wait_stable_ring({"n1"}, {"n2"})


class AtomicArtifactTest(unittest.TestCase):
    def test_atomic_write_mode_and_sha256_sidecar(self) -> None:
        with TemporaryDirectory() as temp:
            output = Path(temp) / "audit.json"
            atomic_write(str(output), "first\n")
            self.assertEqual(output.stat().st_mode & 0o777, 0o644)
            os.chmod(output, 0o640)
            content = '{"healthy":true}\n'
            atomic_write_with_checksum(str(output), content)
            self.assertEqual(output.stat().st_mode & 0o777, 0o640)
            expected = hashlib.sha256(content.encode()).hexdigest()
            self.assertEqual(
                Path(str(output) + ".sha256").read_text(),
                f"{expected}  audit.json\n",
            )


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

    def test_bench_retries_legacy_binary_without_ready_timeout(self) -> None:
        args = SimpleNamespace(
            baseline_bench=None, dfkv_bench="dfkv_bench",
            baseline_members="n=127.0.0.1:1", baseline_mds=None,
            baseline_group="default", size=4096, threads=8, batch=4, bc=1,
            ready_timeout=30.0, run_timeout=60.0,
        )
        unsupported = Mock(
            returncode=2,
            stdout="",
            stderr="usage: dfkv_bench\nunknown option --ready-timeout",
        )
        legacy_output = "\n".join(
            line for line in BENCH.splitlines()
            if not line.startswith("DIAG ")
        ) + "\n"
        completed = Mock(returncode=0, stdout=legacy_output, stderr="")
        with patch(
                "dfkv_load_regression.subprocess.run",
                side_effect=[unsupported, completed]) as run:
            phases = bench_once(
                args, "baseline", 8000, "seed",
                allow_operation_failures=True,
            )
        self.assertEqual(run.call_count, 2)
        self.assertNotIn("--ready-timeout", run.call_args.args[0])
        self.assertFalse(
            phases["get"]["compatibility"]["ready_timeout_supported"])
        self.assertNotIn("diagnostics", phases["get"])
        self.assertFalse(
            phases["get"]["compatibility"]["diagnostics_available"])


if __name__ == "__main__":
    unittest.main()
