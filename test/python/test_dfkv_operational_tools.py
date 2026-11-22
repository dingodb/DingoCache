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
    CommandError,
    atomic_write,
    atomic_write_with_checksum,
    parse_bench,
    parse_bench_diagnostics,
    parse_clients,
    parse_ring,
    parse_stat_health,
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

# `dfkvctl stat --all --group` in the layout of its printf formats; the header
# and the unreachable row are pinned to the real binary by
# test_dfkvctl_all_groups.py.
STAT_HEADER = "%-14s %-20s %-10s %6s %6s %5s %5s %7s %10s %9s %9s %9s %6s\n" % (
    "ID", "ADDR", "VERSION", "READY", "HEALTH", "MDS", "RDMA", "RD-ERR",
    "USED_MB", "OBJECTS", "HITS", "MISSES", "HIT%")
STAT_TOTAL = (
    "TOTAL used=1.0MB objects=10 hits=9 misses=1 hit%=90.0 bytes_w=1 bytes_r=1 "
    "unreachable=1 unhealthy=3 health-unknown=1\n")


def stat_row(node_id: str, ip: str, ready: str = "yes", health: str = "yes",
             mds: str = "yes", rdma: str = "-") -> str:
    return "%-14s %-20s %-10s %6s %6s %5s %5s %7d %10.1f %9d %9d %9d %5.1f%%\n" % (
        node_id, f"{ip}:28000", "2.0", ready, health, mds, rdma, 0, 1.0, 10, 9, 1, 90.0)


def stat_unreachable(node_id: str, ip: str) -> str:
    return "%-14s %-20s %-10s   (unreachable)\n" % (node_id, f"{ip}:28000", "2.0")


STATS = (
    STAT_HEADER
    + stat_row("n1", "10.0.0.1")
    + stat_unreachable("n2", "10.0.0.2")
    + stat_row("n3", "10.0.0.3", ready="NO")
    + stat_row("n4", "10.0.0.4", ready="-", health="-", mds="-")
    + stat_row("n5", "10.0.0.5", mds="NO")
    + stat_row("n6", "10.0.0.6", rdma="NO")
    + stat_row("n7", "10.0.0.7", rdma="yes")
    + STAT_TOTAL
)

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
        # One row is healthy exactly when dfkvctl counts it as neither
        # unreachable, unhealthy nor health-unknown.
        self.assertEqual(
            parse_stat_health(STATS),
            {"n1": True, "n2": False, "n3": False, "n4": False, "n5": False, "n6": False, "n7": True},
        )
        short_row = STAT_HEADER + stat_row("n1", "10.0.0.1").rsplit(None, 1)[0] + "\n"
        for text in ("", STATS.replace("READY", "READY?"), short_row):
            with self.assertRaises(ValueError, msg=text):
                parse_stat_health(text)
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
        for text in (RING_TABLE, "", "[]", json.dumps(document), json.dumps(duplicate)):
            with self.assertRaises(ValueError, msg=text):
                parse_ring(text)

    def test_ring_parser_rejects_values_of_the_wrong_type(self) -> None:
        def with_member(name: str, value: object) -> str:
            document = json.loads(RING)
            document["members"][0][name] = value
            return json.dumps(document)

        def with_top(name: str, value: object) -> str:
            document = json.loads(RING)
            document[name] = value
            return json.dumps(document)

        for text in (
            with_member("id", 5), with_member("id", ["n1"]), with_member("ip", None),
            with_member("port", "abc"), with_member("weight", "heavy"), with_member("weight", True),
            with_member("vnodes", 1.5), with_member("share_percent", "33.3"),
            with_member("info", 0), with_member("eligible", "false"), with_member("eligible", 1),
            with_top("members", {}), with_top("group", 1), with_top("ring_points", "300"),
            with_top("placement_epoch", 7), with_top("placement_epoch", 1.5),
            with_top("placement_epoch", True), with_top("placement_epoch", "-5"),
            with_top("placement_epoch", "soon"), with_top("placement_epoch", ""),
            RING.replace(f'"{2**64 - 59}"', "Infinity"), RING.replace(f'"{2**64 - 59}"', "1e400"),
        ):
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


class FakeUnit:
    def __init__(self, member: dict, running: bool, answers: bool = True) -> None:
        self.member = member
        self.state = "active" if running else "inactive"
        self.ready_at = 0  # from this tick the running process is registered and serves
        self.stopped_at = 0
        self.lease_until = 0
        self.serves_until = 0
        self.start_queued = False
        self.answers = answers
        # Ticks between registering and owning vnodes / reporting READY;
        # None: never.
        self.placed_lag: int | None = 0
        self.ready_lag: int | None = 0
        self.starts = 0


class FakeFleet:
    """Stands in for run_command: one MDS group and hosts running the unit.

    Time is the number of commands issued, so a stop makes progress whatever a
    workflow polls. A server registers and serves START ticks after its unit
    was started. A stopping one behaves like the real server: heartbeats stop
    at once, so the registration is gone after LEASE; requests are refused
    after GRACE; the unit stays deactivating for ``stop_ticks``. `stat --all`
    lists the placement members and exits 1 with its table if any of them is
    unreachable or not ready. systemctl without --no-block waits for its job, so under a
    command timeout shorter than the job the caller times out while the job
    stays queued -- and a start queued behind a stop runs when the stop is over.
    """

    START, GRACE, LEASE = 3, 6, 8

    def __init__(self, *, stop_ticks: int = 20, old_eligible: bool = True,
                 new_eligible: bool = True, new_ready: bool = True,
                 bystander: bool = False) -> None:
        self.now = 0
        self.stop_ticks = stop_ticks
        self.jobs: list[tuple[str, ...]] = []
        # Per member, the number of ring views that showed it placed, and
        # what that count was when a host was asked to stop.
        self.seen_placed: dict[str, int] = {}
        self.seen_placed_at_stop: dict[str, dict[str, int]] = {}
        self.units = {"old": FakeUnit(N1, True), "new": FakeUnit(N2, False)}
        if not old_eligible:
            self.units["old"].placed_lag = None
        if not new_eligible:
            self.units["new"].placed_lag = None
        if not new_ready:
            self.units["new"].ready_lag = None
        if bystander:  # a member nobody is replacing, whose stat port is dead
            self.units["other"] = FakeUnit(
                ring_member("n3", "10.0.0.3", 1, 100, 33.3), True, answers=False)

    def registered(self, unit: FakeUnit) -> bool:
        return (unit.state == "active" and self.now >= unit.ready_at) or self.now < unit.lease_until

    def serving(self, unit: FakeUnit) -> bool:
        return unit.answers and (
            (unit.state == "active" and self.now >= unit.ready_at) or self.now < unit.serves_until)

    def placed(self, unit: FakeUnit) -> bool:
        return self.registered(unit) and unit.placed_lag is not None and (
            unit.state != "active" or self.now >= unit.ready_at + unit.placed_lag)

    def ready(self, unit: FakeUnit) -> bool:
        return self.serving(unit) and unit.ready_lag is not None and (
            unit.state != "active" or self.now >= unit.ready_at + unit.ready_lag)

    def start(self, unit: FakeUnit) -> None:
        unit.state, unit.ready_at, unit.start_queued = "active", self.now + self.START, False
        unit.starts += 1

    def run_command(self, argv, timeout, **_) -> str:
        argv = list(argv)
        self.now += 1
        for unit in self.units.values():
            if unit.state == "deactivating" and self.now >= unit.stopped_at + self.stop_ticks:
                unit.state = "inactive"
                if unit.start_queued:
                    self.start(unit)
        if argv[0] == "dfkvctl":
            return self.dfkvctl(argv)
        host = argv[argv.index("--") + 1]
        unit = self.units[host]
        arguments = argv[argv.index("systemctl") + 1:-1]
        if arguments[0] == "show":
            return f"ActiveState={unit.state}\n"
        self.jobs.append((host, *arguments))
        waits = False
        if arguments[0] == "stop":
            self.seen_placed_at_stop[host] = dict(self.seen_placed)
        if arguments[0] == "stop" and unit.state == "active":
            up = self.now >= unit.ready_at
            unit.state, unit.stopped_at = "deactivating", self.now
            unit.lease_until = self.now + (self.LEASE if up else 0)
            unit.serves_until = self.now + (self.GRACE if up else 0)
            waits = True
        elif arguments[0] == "start" and unit.state == "deactivating":
            unit.start_queued = True
            waits = True
        elif arguments[0] == "start" and unit.state == "inactive":
            self.start(unit)
        if waits and "--no-block" not in arguments:
            raise subprocess.TimeoutExpired(argv, timeout)
        return ""

    def dfkvctl(self, argv: list[str]) -> str:
        registered = [unit for unit in self.units.values() if self.registered(unit)]
        if argv[1] == "ring":
            if "--json" not in argv:
                return RING_TABLE
            out_of_placement = {
                "eligible": False, "status": "DEGRADED", "vnodes": 0, "share_percent": 0.0}
            members = []
            for unit in registered:
                if self.placed(unit):
                    node_id = unit.member["id"]
                    self.seen_placed[node_id] = self.seen_placed.get(node_id, 0) + 1
                    members.append(unit.member)
                else:
                    members.append({**unit.member, **out_of_placement})
            return ring_document(len(registered), *members)
        if argv[1] == "clients":
            return CLIENTS
        placement = [unit for unit in registered if self.placed(unit)]
        if not placement:
            raise CommandError(argv, 1, "", "stat --all: group=glm has an empty placement ring")
        def row(unit: FakeUnit) -> str:
            node_id, ip = unit.member["id"], unit.member["ip"]
            if not self.serving(unit):
                return stat_unreachable(node_id, ip)
            return stat_row(node_id, ip, ready="yes" if self.ready(unit) else "NO")

        table = STAT_HEADER + "".join(row(unit) for unit in placement) + STAT_TOTAL
        if not all(self.ready(unit) for unit in placement):
            raise CommandError(argv, 1, table, "")
        return table


class ReplacementWorkflowTest(unittest.TestCase):
    ARGV = [
        "--mds", "mds:1", "--group", "glm", "--old-id", "n1", "--new-id", "n2",
        "--old-host", "old", "--new-host", "new", "--timeout", "10", "--stop-timeout", "10",
        "--command-timeout", "1", "--poll-interval", "0.01", "--stable-samples", "1",
    ]

    def setUp(self) -> None:
        # No test may reach a real dfkvctl or ssh.
        blocked = patch(
            "dfkv_node_replace.run_command", side_effect=AssertionError("unmocked command"))
        blocked.start()
        self.addCleanup(blocked.stop)

    def replace(self, run_command, timeout: str) -> tuple[int, list[dict]]:
        """main() against a fleet: its exit status and its journal."""
        with TemporaryDirectory() as directory:
            journal = Path(directory) / "journal.json"
            argv = [*self.ARGV, "--state-file", str(journal)]
            argv[argv.index("--timeout") + 1] = timeout
            with patch("dfkv_node_replace.run_command", side_effect=run_command):
                status = replace_main(argv)
            return status, json.loads(journal.read_text())["events"]

    def assert_join_undone(self, fleet: FakeFleet, status: int, events: list[dict],
                           refused_at: str) -> None:
        self.assertEqual(status, 1)
        # The join and its undoing are the only jobs: the old node was never
        # asked to stop, and placement is what it was.
        self.assertEqual(
            fleet.jobs, [("new", "start", "--no-block"), ("new", "stop", "--no-block")])
        old, new = fleet.units["old"], fleet.units["new"]
        self.assertEqual((old.state, old.starts, fleet.registered(old)), ("active", 0, True))
        self.assertEqual(new.state, "inactive")
        reasons = [str(event["message"]) for event in events if event["phase"] == "rollback"]
        self.assertIn(f"waiting for {refused_at}", reasons[0])

    @staticmethod
    def args(dry_run: bool = True) -> SimpleNamespace:
        return SimpleNamespace(
            old_id="n1", new_id="n2", old_host="old", new_host="new",
            dry_run=dry_run, min_clients=0, timeout=10.0, stop_timeout=10.0, command_timeout=1.0,
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

            def wait_stable_ring(self, **sets):
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

            def wait_stable_ring(self, **sets):
                self.required.append(sets)
                return parse_ring(RING)

            def wait_for(self, description, predicate, timeout=None):
                return True

        workflow = Fixture(self.args(dry_run=False))
        workflow.old_stopped = True
        workflow.rollback(RuntimeError("cutover failed"))
        self.assertEqual(workflow.actions, [("old", "start")])
        self.assertEqual(workflow.required, [{"placed": {"n1"}}])

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

            def wait_stable_ring(self, **sets):
                self.actions.append(sets)
                return parse_ring(RING)

            def wait_for(self, description, predicate, timeout=None):
                self.actions.append(description)
                return True

        workflow = Fixture(self.args(dry_run=False))
        workflow.run()
        # The old node is stopped only after the replacement was seen placed
        # and then ready; the cutover is over only when it still is both and
        # the old node is no longer a member.
        self.assertEqual(workflow.actions, [
            ("new", "start"),
            {"placed": {"n2"}, "registered": {"n1"}},
            "replacement node readiness",
            ("old", "stop"),
            {"placed": {"n2"}, "forbidden": {"n1"}},
            "replacement readiness after old-node lease expiry",
        ])
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

    def test_success_does_not_depend_on_how_long_the_old_unit_takes_to_stop(self) -> None:
        fleet = FakeFleet(stop_ticks=10**9)
        with patch("dfkv_node_replace.run_command", side_effect=fleet.run_command):
            status = replace_main(self.ARGV)
        self.assertEqual(status, 0)
        self.assertEqual(
            fleet.jobs, [("new", "start", "--no-block"), ("old", "stop", "--no-block")])
        old = fleet.units["old"]
        self.assertEqual((old.state, fleet.registered(old)), ("deactivating", False))

    def test_readiness_is_judged_on_the_replacement_not_on_the_whole_group(self) -> None:
        # stat --all exits 1 for as long as any member is unreachable.
        fleet = FakeFleet(bystander=True)
        with patch("dfkv_node_replace.run_command", side_effect=fleet.run_command):
            status = replace_main(self.ARGV)
        self.assertEqual(status, 0)
        self.assertEqual(
            fleet.jobs, [("new", "start", "--no-block"), ("old", "stop", "--no-block")])

    def test_old_node_is_not_stopped_for_a_replacement_that_is_not_placed(self) -> None:
        # Registered, but held out of placement for good.
        fleet = FakeFleet(new_eligible=False)
        status, events = self.replace(fleet.run_command, timeout="0.5")
        self.assert_join_undone(fleet, status, events, refused_at="a stable MDS ring")

    def test_old_node_is_stopped_only_after_the_replacement_was_seen_placed(self) -> None:
        # The replacement registers first and owns vnodes later. `stat --all`
        # lists it from then on, so readiness alone would also let the stop
        # through -- without a ring view ever having shown it placed.
        fleet = FakeFleet()
        fleet.units["new"].placed_lag = 12
        status, _ = self.replace(fleet.run_command, timeout="10")
        self.assertEqual(status, 0)
        self.assertGreaterEqual(fleet.seen_placed_at_stop["old"].get("n2", 0), 1)

    def test_old_node_is_not_stopped_for_a_replacement_that_is_not_ready(self) -> None:
        # Placed, but its row never shows READY.
        fleet = FakeFleet(new_ready=False)
        status, events = self.replace(fleet.run_command, timeout="0.5")
        self.assert_join_undone(fleet, status, events, refused_at="replacement node readiness")

    def test_cutover_is_undone_when_the_replacement_drops_out_of_placement(self) -> None:
        fleet = FakeFleet()

        def run_command(argv, timeout, **options) -> str:
            output = fleet.run_command(argv, timeout, **options)
            # Once the old node has been asked to stop, the replacement's
            # rails fail: it stays registered and loses its vnodes.
            if fleet.jobs[-1:] == [("old", "stop", "--no-block")]:
                fleet.units["new"].placed_lag = None
            return output

        status, events = self.replace(run_command, timeout="0.5")
        self.assertEqual(status, 1)
        self.assertEqual(fleet.jobs, [
            ("new", "start", "--no-block"), ("old", "stop", "--no-block"),
            ("old", "start", "--no-block")])
        old = fleet.units["old"]
        self.assertEqual((old.state, old.starts), ("active", 1))
        self.assertTrue(fleet.ready(old))
        reasons = [str(event["message"]) for event in events if event["phase"] == "rollback"]
        self.assertIn("waiting for a stable MDS ring", reasons[0])
        self.assertIn("old node restored", reasons[-1])

    def test_rollback_waits_until_the_restarted_old_node_is_ready(self) -> None:
        fleet = FakeFleet()
        fleet.units["new"].state = "active"
        fleet.units["old"].ready_lag = 6  # registered and placed before it reports READY
        with patch("dfkv_node_replace.run_command", side_effect=fleet.run_command):
            workflow = Replacer(self.args(dry_run=False))
            workflow.remote("old", "stop")
            workflow.old_stopped = True
            workflow.rollback(RuntimeError("cutover failed"))
        old = fleet.units["old"]
        self.assertEqual((old.state, old.starts), ("active", 1))
        self.assertTrue(fleet.ready(old))

    def test_readiness_takes_the_table_of_exit_1_and_nothing_else(self) -> None:
        workflow = Replacer(self.args())
        verdict = CommandError(["dfkvctl", "stat"], 1, STATS, "")
        with patch("dfkv_node_replace.run_command", side_effect=verdict):
            self.assertTrue(workflow.healthy("n1"))
            self.assertFalse(workflow.healthy("n2"))
            self.assertFalse(workflow.healthy("absent"))
        for failure in (
            CommandError(["dfkvctl", "stat"], 1, "", "stat --all: MDS query failed"),
            CommandError(["dfkvctl", "stat"], 2, STATS, "stat --all needs --mds ip:port[,...]"),
        ):
            with patch("dfkv_node_replace.run_command", side_effect=failure):
                with self.assertRaises(CommandError):
                    workflow.healthy("n1")

    def test_an_old_node_already_out_of_placement_can_be_replaced(self) -> None:
        fleet = FakeFleet(old_eligible=False)
        with patch("dfkv_node_replace.run_command", side_effect=fleet.run_command):
            workflow = Replacer(self.args(dry_run=False))
            workflow.run()
        self.assertEqual(
            fleet.jobs, [("new", "start", "--no-block"), ("old", "stop", "--no-block")])
        self.assertFalse(workflow.old_placed)
        self.assertFalse(fleet.registered(fleet.units["old"]))
        self.assertEqual(workflow.events[-1]["phase"], "complete")

    def test_rollback_waits_for_the_restarted_process_not_the_one_going_away(self) -> None:
        fleet = FakeFleet()
        fleet.units["new"].state = "active"
        with patch("dfkv_node_replace.run_command", side_effect=fleet.run_command):
            workflow = Replacer(self.args(dry_run=False))
            workflow.remote("old", "stop")
            workflow.old_stopped = True
            workflow.rollback(RuntimeError("cutover failed"))
        self.assertEqual(
            fleet.jobs, [("old", "stop", "--no-block"), ("old", "start", "--no-block")])
        # When the rollback began, the process being stopped was still
        # registered and still answering; restored is the one started after it.
        old = fleet.units["old"]
        self.assertEqual((old.state, old.starts), ("active", 1))
        self.assertTrue(fleet.registered(old) and fleet.serving(old))
        self.assertIn("old node restored", str(workflow.events[-1]["message"]))

    def test_rollback_outlasts_a_stop_longer_than_the_transition_timeout(self) -> None:
        args = self.args(dry_run=False)
        args.timeout = 0.5  # 80 polls take at least 0.8 s
        fleet = FakeFleet(stop_ticks=80)
        fleet.units["new"].state = "active"
        with patch("dfkv_node_replace.run_command", side_effect=fleet.run_command):
            workflow = Replacer(args)
            workflow.remote("old", "stop")
            workflow.old_stopped = True
            workflow.rollback(RuntimeError("cutover failed"))
        self.assertEqual((fleet.units["old"].state, fleet.units["old"].starts), ("active", 1))

    def test_rollback_of_an_old_node_out_of_placement_only_needs_it_registered(self) -> None:
        fleet = FakeFleet(old_eligible=False)
        fleet.units["new"].state = "active"
        with patch("dfkv_node_replace.run_command", side_effect=fleet.run_command):
            workflow = Replacer(self.args(dry_run=False))
            workflow.old_placed = False
            workflow.remote("old", "stop")
            workflow.old_stopped = True
            workflow.rollback(RuntimeError("cutover failed"))
        old = fleet.units["old"]
        self.assertEqual((old.state, old.starts, fleet.registered(old)), ("active", 1, True))

    def test_unconfirmed_rollback_reports_what_systemd_already_has(self) -> None:
        class Fixture(Replacer):
            def remote(self, host, action):
                pass

            def unit_state(self, host):
                return "deactivating"

        args = self.args(dry_run=False)
        args.timeout = args.stop_timeout = 0.02
        workflow = Fixture(args)
        workflow.old_stopped = True
        with self.assertRaises(WorkflowError) as raised:
            workflow.rollback(RuntimeError("cutover failed"))
        message = str(raised.exception)
        self.assertIn("CRITICAL rollback not confirmed", message)
        self.assertIn("systemd accepted the start of dfkv on old", message)
        self.assertNotIn("immediately", message)

        workflow = Fixture(args)
        workflow.started_new = True
        with self.assertRaises(WorkflowError) as raised:
            workflow.rollback(RuntimeError("join failed"))
        message = str(raised.exception)
        self.assertIn("accepted by systemd but not seen to finish", message)
        self.assertIn("check that dfkv on new stops", message)
        self.assertNotIn("manually", message)

    def test_rollback_whose_start_command_fails_asks_for_a_manual_start(self) -> None:
        class Fixture(Replacer):
            def remote(self, host, action):
                raise subprocess.TimeoutExpired(["ssh", host], 1.0)

        workflow = Fixture(self.args(dry_run=False))
        workflow.old_stopped = True
        with self.assertRaises(WorkflowError) as raised:
            workflow.rollback(RuntimeError("cutover failed"))
        self.assertIn("CRITICAL rollback also failed", str(raised.exception))
        self.assertIn("start dfkv on old immediately", str(raised.exception))

    def test_stable_ring_tells_placed_from_merely_registered(self) -> None:
        args = self.args()
        args.timeout = 0.05
        with patch("dfkv_node_replace.run_command", return_value=RING_DEGRADED):
            workflow = Replacer(args)
            # n2 is registered but owns no vnodes.
            self.assertEqual(
                workflow.wait_stable_ring(placed={"n1"}, registered={"n2"}).group, "glm")
            with self.assertRaises(WorkflowError):
                workflow.wait_stable_ring(placed={"n1", "n2"})
            with self.assertRaises(WorkflowError):
                workflow.wait_stable_ring(placed={"n1"}, forbidden={"n2"})
            with self.assertRaises(WorkflowError):
                workflow.wait_stable_ring(registered={"n3"})


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
        # Bytes that are not UTF-8 read as `dfkvctl ring --json` writes them.
        garbled = decode_registration(self.registration().replace(b"engine=slab", b"engine=\xffl\xfeb"))
        self.assertEqual(info_fields(garbled.info)["engine"], "\xffl\xfeb")
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
