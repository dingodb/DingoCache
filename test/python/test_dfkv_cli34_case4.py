#!/usr/bin/env python3
"""Deterministic CLI-34 case-4 workload against real dfkv processes.

Scenario (design docs/design/ring-unavailable-owner.md §1 case 4, §3.1 A,
§3.3 M, §3.4 B): the MDS membership record of a LIVE owner disappears while
its process, data and metrics stay untouched.  Unlike CLI-35 case 5 there is
no SIGKILL and no empty restart: the same server process keeps serving the
whole time, is later re-announced by its own heartbeat, and must resume
ordinary reads.

Full timeline (seconds, relative to the start of workload traffic):
    0..30 three real nodes n0..n2 with deterministic traffic, 30 fence+delete
    n2's member record, 120 unfence (the server's own heartbeat re-creates the
    record), 195 stop the workload.  The absence therefore lasts ~90 s, longer
    than the read-back horizon H = DFKV_RING_READBACK_MS + 10*poll = 60 s, so
    both the within-H read-back phase and the over-H refusal phase of the
    dedicated absent route (PlanReadBack's absent_since >= H => unknown) are
    exercised in one run.

How the record is made to disappear (and STAY disappeared): a raw etcd delete
does not last, because the MDS re-PUTs the member record on every victim
heartbeat (MdsServer::UpsertLeased: "Absent ... a new lease ... put"), and a
briefly re-listed member would end every client's absent episode and reset
absent_since.  This runner therefore uses the MDS's own coordinated deletion:
`dfkvctl member fence n2` writes the fence marker first and then deletes the
member record from etcd; every later heartbeat of the still-running victim is
refused (dfkv_mds_fenced=1 on the victim, dfkv_mds_fenced_refusals_total on
the MDS), so the record stays absent deterministically.  A record-keeper
thread independently polls the exact etcd key every 0.4 s and re-deletes it
if it ever reappears (repeated-deletion backstop; expected recreations: 0),
recording every observation as evidence.  `dfkvctl member unfence n2` then
lets the SAME process re-announce itself on its next heartbeat (1 s), which
the keeper observes as the record's reappearance under a fresh real lease.

Every variant gets an independent real etcd/MDS/ring.  The run needs only
Linux loopback TCP, the built binaries and an etcd executable (DFKV_ETCD,
ETCD_BIN, or PATH).  No GPU or production resource is used.  --smoke shortens
only the timeline and is never an acceptance run.  Runtime data is retained
below the supplied output root, which must be empty.

Variants: base (A/B/M off), a (absent grace only), amb (A+M+B).  CLI-35's W
rewrite switch is off everywhere: case 4 is an op25-only scenario, asserted by
dfkv_removed_within_ages_requests_total staying 0 while
dfkv_removed_within_queries_total is positive in amb.

Operation-log semantics, per-second coverage evidence, the ctypes client, the
process/JSONL plumbing, the churn remover and the offline violation checker
(confirmed/possible tombstone + definite non-store vs ambiguous PUT witness)
are reused from the proven CLI-35 runner (imported, not copied); only the
case-4-specific workload identity, membership semantics, metrics focus and
acceptance checks live here.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import traceback
from typing import Any
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_dfkv_cli35_case5 as c35  # proven CLI-35 helpers, reused unchanged

ALL_VARIANTS = ("base", "a", "amb")
NODES = 3
VICTIM = "n2"
# Batch geometry must match CLI-35's: its NativeClient methods read these
# module constants, and case 4 reuses those methods unchanged.
PREFIXES = 128
BLOCKS = 64
BLOCK_BYTES = 4096
assert (c35.PREFIXES, c35.BLOCKS, c35.BLOCK_BYTES) == (PREFIXES, BLOCKS, BLOCK_BYTES), \
    "case4 reuses CLI-35 NativeClient batch geometry"
READS_PER_SECOND = 16
NAMESPACE = b"dfkv/cli34/case4/native-v1"
POLL_MS = 1000
ABSENT_GRACE_MS = 600000          # A: far beyond the ~90 s absence
READBACK_MS = 50000               # B window
SETTLE_MS = 10 * POLL_MS
H_MS = READBACK_MS + SETTLE_MS    # dedicated absent route horizon: 60 s
MEMORY_MS = 120000                # server M window, >= H as B requires
MEMORY_COVERAGE_S = MEMORY_MS // 1000
HEARTBEAT_MS = 1000               # victim re-announces ~1 s after unfence
MDS_LEASE_TTL_SECONDS = 30        # src/mds/mds_server.h kTtlSeconds (real lease expiry)
SCHEDULE_TOLERANCE_NS = c35.SCHEDULE_TOLERANCE_NS
# Evidence-window margins around MDS-observed membership transitions.  A
# client adopts a 3->2 shrink only after the guard sees the same value on two
# consecutive 1 s polls; 5 s covers polls + adoption + scrape cadence with room.
MARGIN_S = 5
DISRUPTION_BAD_RATE = 0.5         # one second is "disrupted" at <=50% full prefixes
DISRUPTION_MIN_BAD_SECONDS = 3
DISRUPTION_MAX_MEAN = 0.75
# The deterministic remover churn caps full-prefix recovery near 0.76-0.78:
# a not-yet-healed REMOVE breaks its whole 64-block prefix.  Judge recovery
# against the same variant's own churn-limited pre-absence rate instead of an
# absolute clean-run number (the pattern CLI-35 uses).
AMB_MIN_PRE_ABSENCE_FRACTION = 0.90
POST_MIN_PRE_ABSENCE_FRACTION = 0.90
# A is a stronger control than base: its absent-member REMOVE path heals much
# of the churn disruption without reading the stale copy.  Require a clear
# 10-point full-prefix gain over both controls; the binary victim hit/refusal
# counters remain the primary AMB proof.
MIN_AMB_CONTRAST = 0.10
VICTIM_MIN_READBACK_HITS = 100    # amb within-H dfkv_cache_hit delta on n2
KEEPER_PERIOD_S = 0.4
REANNOUNCE_TIMEOUT_S = 30.0

DEFERRED_CHECKS = [
    {
        "name": "no_M_server_kInvalid_fail_closed",
        "reason": "a mixed ring with one M-off server in an M+B variant is a "
                  "second server-configuration matrix; the identical "
                  "fail-closed path (kInvalid => unknown => refused) is "
                  "exercised by CLI-35's ambw-now (all servers W-off answer "
                  "op26 with kInvalid) and ambw-old (a real op25-only older "
                  "server).  Case 4 keeps one homogeneous server matrix.",
    },
    {
        "name": "op25_only_old_server_binary",
        "reason": "case 4 never sends op26 (W off by design), so an op25-only "
                  "older server is wire-identical to a current one here; the "
                  "old-binary interop is CLI-35's ambw-old variant.",
    },
    {
        "name": "churn_skew_cases",
        "reason": "remover restarted inside a read-back window, lagging "
                  "member pollers and per-client cooldown skew are CLI-35 "
                  "case5's churn matrix; case 4 runs one continuous remover "
                  "plus the read-back reader.",
    },
]


class Workload4(c35.Workload):
    """CLI-35's deterministic workload shape with case-4 identity."""

    def __init__(self) -> None:
        self.keys: list[list[bytes]] = []
        self.values: list[list[bytes]] = []
        self.flat: dict[bytes, int] = {}
        digest = hashlib.sha256()
        for prefix in range(PREFIXES):
            key_row: list[bytes] = []
            value_row: list[bytes] = []
            for block in range(BLOCKS):
                key = f"cli34/case4/p{prefix:03d}/b{block:02d}".encode()
                seed = hashlib.sha256(b"dfkv-cli34-value-v1\0" + key).digest()
                header = b"DFKV-CLI34-CASE4\0" + key + b"\0"
                value = (header + seed * 128)[:BLOCK_BYTES]
                assert len(value) == BLOCK_BYTES
                key_row.append(key)
                value_row.append(value)
                self.flat[key] = prefix * BLOCKS + block
                digest.update(key)
                digest.update(value)
            self.keys.append(key_row)
            self.values.append(value_row)
        self.identity_sha256 = digest.hexdigest()

    @staticmethod
    def indices(kind: str, second: int, count: int, universe: int) -> list[int]:
        answer: list[int] = []
        attempt = 0
        while len(answer) < count:
            material = f"dfkv-cli34-case4\0{kind}\0{second}\0{attempt}".encode()
            index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % universe
            if index not in answer:
                answer.append(index)
            attempt += 1
        return answer


class NativeClient4(c35.NativeClient):
    """CLI-35's ctypes client with the case-4 namespace and client info."""

    def __init__(self, lib: ctypes.CDLL, client_id: str, role: str,
                 mds: str, group: str, operations: c35.OperationLog) -> None:
        self.lib = lib
        self.client_id = client_id
        self.role = role
        self.operations = operations
        self.last_put_start_ns = 0
        self.last_flat_start_ns = 0
        options = c35.make_client_options_v2(
            NAMESPACE, mds_endpoints=mds, mds_group=group,
            mds_poll_ms=POLL_MS, register_client=True, client_id=client_id,
            client_info=f"type=cli34-case4;role={role}",
            client_heartbeat_ms=1000, batch_concurrency=8,
        )
        self._options = options
        self.handle = lib.dfkv_open_v2(ctypes.byref(options))
        if not self.handle:
            raise RuntimeError(f"dfkv_open_v2 failed for {client_id}")
        mode = lib.dfkv_transport_mode(self.handle)
        self.transport = mode.decode(errors="replace") if mode else ""
        if not self.transport.startswith("tcp"):
            lib.dfkv_close(self.handle)
            self.handle = None
            raise RuntimeError(f"{client_id}: TCP required, got {self.transport!r}")


@dataclass
class Node4:
    node_id: str
    binary: Path
    child: c35.Child
    data_dir: Path
    port: int
    metrics_port: int

    @property
    def key(self) -> str:
        return f"{self.node_id}-g0"


def _server_core4(text: str) -> dict[str, float | None]:
    """Per-node scalars; the victim's read counters are case-4's direct
    stale-copy evidence, its registrar gauges prove the process stayed alive
    and fenced the whole absence."""
    return {
        "uptime_seconds": c35._metric_sum(text, "dfkv_uptime_seconds"),
        "cache_put": c35._metric_sum(text, "dfkv_cache_put_total"),
        "cache_hit": c35._metric_sum(text, "dfkv_cache_hit_total"),
        "cache_miss": c35._metric_sum(text, "dfkv_cache_miss_total"),
        "exist_hit": c35._metric_sum(text, "dfkv_exist_hit_total"),
        "exist_miss": c35._metric_sum(text, "dfkv_exist_miss_total"),
        "remove_ok": c35._metric_sum(text, "dfkv_remove_ok_total"),
        "remove_miss": c35._metric_sum(text, "dfkv_remove_miss_total"),
        "removed_within_queries": c35._metric_sum(
            text, "dfkv_removed_within_queries_total"),
        "removed_within_ages_requests": c35._metric_sum(
            text, "dfkv_removed_within_ages_requests_total"),
        "remove_memory_entries": c35._metric_sum(text, "dfkv_remove_memory_entries"),
        "remove_memory_covered_seconds": c35._metric_sum(
            text, "dfkv_remove_memory_covered_seconds"),
        "mds_fenced": c35._metric_sum(text, "dfkv_mds_fenced"),
        "mds_heartbeat_healthy": c35._metric_sum(text, "dfkv_mds_heartbeat_healthy"),
        "mds_heartbeat_failures_consecutive": c35._metric_sum(
            text, "dfkv_mds_heartbeat_failures_consecutive"),
        "mds_registration_latched": c35._metric_sum(
            text, "dfkv_mds_registration_latched"),
    }


def _mds_core(text: str) -> dict[str, float | None]:
    return {
        "fenced_refusals": c35._metric_sum(text, "dfkv_mds_fenced_refusals_total"),
        "record_puts": c35._metric_sum(text, "dfkv_mds_record_puts_total"),
        "lease_grants": c35._metric_sum(text, "dfkv_mds_lease_grants_total"),
    }


class MetricsCollector4(c35.MetricsCollector):
    """CLI-35's collector loop with case-4 membership semantics: the victim
    record disappears by fence+etcd-delete and returns by re-announce."""

    def __init__(self, run: "VariantRun4", origin_ns: int) -> None:
        super().__init__(run, origin_ns)
        self.gone_ns = 0
        self.gone_record: dict = {}
        self.back_ns = 0
        self.back_record: dict = {}
        self.final_victim_eligible = False
        self.final_victim_member: dict = {}
        self.last_mds_raw = ""

    def capture(self, phase: str | None) -> None:
        with self.capture_lock:
            now = time.monotonic_ns()
            second = ((now - self.origin_ns) / 1e9
                      if self.origin_ns else None)
            row: dict[str, Any] = {
                "monotonic_ns": now, "workload_second": second,
                "phase": phase, "clients": {}, "servers": {}, "mds": {},
            }
            for client in self.run.client_snapshot():
                try:
                    text = client.stats()
                    self.last_client_raw[client.client_id] = text
                    row["clients"][client.client_id] = c35._client_core(text)
                    if phase:
                        path = self.run.root / "metrics" / f"client-{client.client_id}-{phase}.prom"
                        path.write_text(text)
                except Exception as exc:
                    row["clients"][client.client_id] = {"error": str(exc)}
            for node in self.run.node_snapshot():
                try:
                    text = self.scrape(node.metrics_port)
                    self.last_server_raw[node.key] = text
                    row["servers"][node.key] = _server_core4(text)
                    if phase:
                        path = self.run.root / "metrics" / f"server-{node.key}-{phase}.prom"
                        path.write_text(text)
                except Exception as exc:
                    row["servers"][node.key] = {"error": str(exc)}
            if self.run.mds_metrics_port:
                try:
                    text = self.scrape(self.run.mds_metrics_port)
                    self.last_mds_raw = text
                    row["mds"] = _mds_core(text)
                    if phase:
                        (self.run.root / "metrics" / f"mds-{phase}.prom").write_text(text)
                except Exception as exc:
                    row["mds"] = {"error": str(exc)}
            if phase == "final":
                self.final_complete = all(
                    "error" not in values
                    for group in (row["clients"], row["servers"])
                    for values in group.values()) and "error" not in row["mds"]
            self.run.sinks["metrics"].write(row)
            for key, text in self.last_server_raw.items():
                (self.run.root / "metrics" / f"server-{key}-last.prom").write_text(text)
            if not phase:
                return
            for client_id, text in self.last_client_raw.items():
                (self.run.root / "metrics" / f"client-{client_id}-last.prom").write_text(text)
            if self.last_mds_raw:
                (self.run.root / "metrics" / "mds-last.prom").write_text(self.last_mds_raw)

    def poll_ring(self, *, final: bool = False) -> None:
        record: dict[str, Any] = {"poll_start_ns": time.monotonic_ns(),
                                  "wall_time": c35._iso(), "final": final,
                                  "transitions": []}
        try:
            ring = self.run.ring()
            observed_ns = time.monotonic_ns()
            record["observed_ns"] = observed_ns
            record["members"] = [
                {"id": m["id"], "ip": m["ip"], "port": m["port"],
                 "eligible": m["eligible"]} for m in ring["members"]]
            self.last_ring = ring
            victim = next((m for m in ring["members"] if m["id"] == VICTIM), None)
            if (self.run.fence_confirmed_ns and not self.run.unfence_done
                    and victim is None
                    and observed_ns >= self.run.fence_confirmed_ns
                    and not self.gone_ns):
                self.gone_ns = observed_ns
                self.gone_record = {
                    "transition": "victim_absent_after_confirmed_fence",
                    "observed_ns": observed_ns,
                    "fence_confirmed_ns": self.run.fence_confirmed_ns,
                }
                self.transitions.append(self.gone_record)
                record["transitions"].append(self.gone_record)
            if (self.run.unfence_done and victim is not None
                    and victim.get("eligible") is True
                    and observed_ns >= self.run.unfence_confirmed_ns
                    and not self.back_ns):
                self.back_ns = observed_ns
                self.back_record = {
                    "transition": "victim_eligible_after_unfence",
                    "observed_ns": observed_ns,
                    "unfence_confirmed_ns": self.run.unfence_confirmed_ns,
                    "member": dict(victim),
                }
                self.transitions.append(self.back_record)
                record["transitions"].append(self.back_record)
            if final:
                self.final_ring_ns = observed_ns
                self.final_victim_eligible = (
                    victim is not None and victim.get("eligible") is True)
                self.final_victim_member = dict(victim) if victim else {}
                record["final_victim_eligible"] = self.final_victim_eligible
                record["final_victim_member"] = self.final_victim_member
        except Exception as exc:
            record["error"] = str(exc)
        self.run.sinks["membership"].write(record)


class RecordKeeper:
    """Watches the victim's exact etcd membership key every KEEPER_PERIOD_S.

    While fenced, any reappearance (the MDS/server restoring its own
    registration despite the fence) is deleted again at once and recorded:
    the repeated-deletion backstop, expected to fire zero times.  After the
    unfence, the first reappearance is the victim's own heartbeat
    re-announcing the same process under a fresh real lease.
    """

    def __init__(self, run: "VariantRun4", etcd_endpoint: str, member_key: str) -> None:
        self.run = run
        self.etcd_endpoint = etcd_endpoint
        self.member_key = member_key
        self.key_b64 = base64.b64encode(member_key.encode()).decode()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="record-keeper",
                                       daemon=True)
        self.pre_fence_record: dict = {}
        self.samples = 0
        self.absent_samples = 0
        self.first_absent_ns = 0
        self.recreated_and_redeleted: list[dict] = []
        self.reannounce_ns = 0
        self.reannounce_record: dict = {}
        self.errors: list[str] = []
        self.started_ns = 0

    def _etcd_post(self, path: str, payload: dict) -> dict:
        request = UrlRequest(
            f"http://{self.etcd_endpoint}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=2) as response:
            return json.loads(response.read())

    def sample(self) -> dict | None:
        body = self._etcd_post("/v3/kv/range", {"key": self.key_b64})
        kvs = body.get("kvs", [])
        if not kvs:
            return None
        kv = kvs[0]
        return {
            "lease": str(kv.get("lease", "0")),
            "create_revision": str(kv.get("create_revision", "")),
            "mod_revision": str(kv.get("mod_revision", "")),
            "value_bytes": len(base64.b64decode(kv.get("value", ""))),
        }

    def delete(self) -> None:
        self._etcd_post("/v3/kv/deleterange", {"key": self.key_b64})

    def start(self) -> None:
        self.started_ns = time.monotonic_ns()
        self.thread.start()

    def _loop(self) -> None:
        last_present: bool | None = None
        try:
            while not self.stop_event.is_set():
                now = time.monotonic_ns()
                try:
                    record = self.sample()
                except Exception as exc:
                    self.errors.append(f"sample: {type(exc).__name__}: {exc}")
                    self.stop_event.wait(KEEPER_PERIOD_S)
                    continue
                present = record is not None
                self.samples += 1
                if not present:
                    self.absent_samples += 1
                    if not self.first_absent_ns:
                        self.first_absent_ns = now
                if present != last_present:
                    last_present = present
                    self.run.sinks["membership"].write({
                        "event": "etcd_member_record_presence",
                        "present": present, "monotonic_ns": now,
                        "wall_time": c35._iso(), "record": record,
                        "key": self.member_key,
                    })
                if not self.run.unfence_done:
                    if present:
                        # The MDS/server restored its own registration despite
                        # the fence: delete it again (repeated deletion).
                        try:
                            self.delete()
                        except Exception as exc:
                            self.errors.append(
                                f"delete: {type(exc).__name__}: {exc}")
                        entry = {"observed_ns": now, "record": record}
                        self.recreated_and_redeleted.append(entry)
                        self.run.sinks["membership"].write({
                            "event": "member_record_recreated_redeleted",
                            "monotonic_ns": now, "wall_time": c35._iso(),
                            "record": record, "key": self.member_key,
                        })
                else:
                    if present:
                        self.reannounce_ns = now
                        self.reannounce_record = record or {}
                        self.run.sinks["membership"].write({
                            "event": "member_record_reannounced_by_victim_heartbeat",
                            "monotonic_ns": now, "wall_time": c35._iso(),
                            "record": record, "key": self.member_key,
                            "unfence_confirmed_ns": self.run.unfence_confirmed_ns,
                        })
                        return
                    if now - self.run.unfence_confirmed_ns > REANNOUNCE_TIMEOUT_S * 1e9:
                        self.errors.append(
                            f"no reannounce within {REANNOUNCE_TIMEOUT_S}s of unfence")
                        return
                self.stop_event.wait(KEEPER_PERIOD_S)
        except BaseException as exc:
            self.errors.append(f"keeper: {type(exc).__name__}: {exc}")

    def stop_and_join(self) -> None:
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError("record keeper did not stop")


class VariantRun4:
    def __init__(self, args: argparse.Namespace, variant: str,
                 timing: dict[str, int], workload: Workload4, etcd: Path) -> None:
        self.args = args
        self.variant = variant
        self.timing = timing
        self.workload = workload
        self.etcd = etcd
        self.root = args.output_root / variant
        self.root.mkdir()
        (self.root / "logs").mkdir()
        (self.root / "metrics").mkdir()
        (self.root / "client-access").mkdir()
        (self.root / "server-access").mkdir()
        self.sinks = {
            name: c35.Jsonl(self.root / f"{name}.jsonl")
            for name in ("operations", "per_second", "prefix_results", "timeline",
                         "metrics", "membership", "processes", "churn",
                         "memory_coverage")
        }
        self.processes = c35.Processes(self.root, self.sinks["processes"])
        self.operations = c35.OperationLog(self.sinks["operations"], workload)
        self.errors: queue.Queue[str] = queue.Queue()
        self.clients: dict[str, NativeClient4] = {}
        self.clients_lock = threading.Lock()
        self.nodes: list[Node4] = []
        self.nodes_lock = threading.Lock()
        self.loader: NativeClient4 | None = None
        self.reader: NativeClient4 | None = None
        self.removers: list[c35.Remover] = []
        self.executor: concurrent.futures.ThreadPoolExecutor | None = None
        self.metrics: MetricsCollector4 | None = None
        self.keeper: RecordKeeper | None = None
        self.stats = c35.PerSecond(self.sinks["per_second"], READS_PER_SECOND)
        self.origin_ns = 0
        self.fence_confirmed_ns = 0
        self.unfence_confirmed_ns = 0
        self.unfence_done = False
        self.fence_evidence: dict = {}
        self.unfence_evidence: dict = {}
        self.mds_metrics_port = 0
        self.result: dict[str, Any] = {
            "variant": variant, "acceptance_eligible": not args.smoke,
            "smoke": args.smoke, "timing": timing,
            "scenario": "live owner's MDS membership record disappears "
                        "(fence-coordinated etcd deletion, victim process, data "
                        "and metrics untouched) and is later re-announced by "
                        "the same process's own heartbeat",
            "h_boundary": {
                "readback_ms": READBACK_MS, "settle_ms": SETTLE_MS,
                "poll_ms": POLL_MS, "h_ms": H_MS,
                "rule": "the dedicated absent-owner read-back route holds only "
                        "while now - absent_since < H; an older absence is "
                        "refused as unknown without reading the absent member",
            },
            "workload": {
                "prefixes": PREFIXES, "blocks_per_prefix": BLOCKS,
                "block_bytes": BLOCK_BYTES, "initial_mib": 32,
                "sampled_prefixes_per_second": READS_PER_SECOND,
                "workload_identity_sha256": workload.identity_sha256,
                "reader_schedule": "SHA-256 kind=reader (dfkv-cli34-case4), 16 distinct prefixes/second",
                "churn": "CLI-35 Remover unchanged: existent-key probe; up to 8 REMOVEs/s; half PUT again after 2 s",
            },
            "client_env": self.client_env(),
            "server_env": self.server_env(),
            "status": "RUNNING", "errors": [],
            "deferred_checks": DEFERRED_CHECKS,
        }
        self.group = f"cli34-case4-{variant}-{os.getpid()}"
        self.mds_endpoint = ""
        self.etcd_endpoint = ""
        self.member_key = f"/dfkv/v1/groups/{self.group}/members/{VICTIM}"
        self._saved_env: dict[str, str | None] = {}
        self._final_captured = False
        self._correctness: dict = {}
        self._cleaned_up = False

    def client_env(self) -> dict[str, str]:
        a_on = self.variant in ("a", "amb")
        b_on = self.variant == "amb"
        return {
            "DFKV_RING_ABSENT_GRACE_MS": str(ABSENT_GRACE_MS if a_on else 0),
            "DFKV_RING_READBACK_MS": str(READBACK_MS if b_on else 0),
            "DFKV_RING_READBACK_REWRITE": "0",
            "DFKV_RING_COOLDOWN_REROUTE": "0",
            "DFKV_RING_REMOVE_FANOUT": "3",
            "DFKV_RING_REROUTE_KEYS": "65536",
            "DFKV_RING_REROUTE_CLOSE_MS": "2000",
            "DFKV_CLIENT_NODE_DEDUP": "0",
            "DFKV_CLIENT_NODE_DEDUP_GPU": "0",
            "DFKV_RDMA": "0",
            "DFKV_OPEN_RING_WAIT_MS": "30000",
            "DFKV_GET_MISS_RETRIES": "1",
            "DFKV_LOG": "WARN",
            "DFKV_ACCESS_DIR": str(self.root / "client-access"),
        }

    def server_env(self) -> dict[str, str]:
        return {
            "DFKV_REMOVE_MEMORY_MS": str(MEMORY_MS if self.variant == "amb" else 0),
            "DFKV_REMOVE_MEMORY_ENTRIES": "1048576",
            "DFKV_WRITE_MEMORY_ENTRIES": "0",
            "DFKV_MDS_HEARTBEAT_MS": str(HEARTBEAT_MS),
            "DFKV_LOG": "INFO",
            "DFKV_RDMA": "0",
            "DFKV_ACCESS_DIR": str(self.root / "server-access"),
        }

    def apply_client_env(self) -> None:
        wanted = self.client_env()
        for key in c35.CLIENT_ENV_KEYS:
            self._saved_env[key] = os.environ.get(key)
            os.environ[key] = wanted[key]
        self.result["client_env"] = wanted

    def restore_client_env(self) -> None:
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._saved_env.clear()

    def client_snapshot(self) -> list[NativeClient4]:
        with self.clients_lock:
            return list(self.clients.values())

    def node_snapshot(self) -> list[Node4]:
        with self.nodes_lock:
            return list(self.nodes)

    def victim_node(self) -> Node4:
        with self.nodes_lock:
            return self.nodes[int(VICTIM[1:])]

    def open_client(self, role: str, number: int) -> NativeClient4:
        client_id = f"{self.variant}-{role}-{number}-{os.getpid()}"
        client = NativeClient4(self.args.lib_handle, client_id, role,
                               self.mds_endpoint, self.group, self.operations)
        with self.clients_lock:
            self.clients[client_id] = client
        self.sinks["timeline"].write({
            "event": "client_open", "client": client_id, "role": role,
            "transport": client.transport, "monotonic_ns": time.monotonic_ns(),
        })
        return client

    def close_client(self, client: NativeClient4) -> None:
        with self.clients_lock:
            self.clients.pop(client.client_id, None)
        client.close()

    def start_etcd(self) -> None:
        for attempt in range(1, 6):
            client_port, peer_port = c35._free_port(), c35._free_port()
            data = self.root / "runtime" / f"etcd-attempt-{attempt}"
            data.mkdir(parents=True)
            client_url = f"http://127.0.0.1:{client_port}"
            peer_url = f"http://127.0.0.1:{peer_port}"
            argv = [
                str(self.etcd), "--data-dir", str(data),
                "--name", f"cli34-{self.variant}",
                "--listen-client-urls", client_url,
                "--advertise-client-urls", client_url,
                "--listen-peer-urls", peer_url,
                "--initial-advertise-peer-urls", peer_url,
                "--initial-cluster", f"cli34-{self.variant}={peer_url}",
            ]
            child = self.processes.launch(f"etcd-{attempt}", argv, dict(os.environ))
            self.etcd_endpoint = f"127.0.0.1:{client_port}"
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if child.popen.poll() is not None:
                    break
                try:
                    with urlopen(client_url + "/health", timeout=0.5) as response:
                        health = json.loads(response.read())
                    if health.get("health") == "true":
                        self.result["etcd"] = {
                            "executable": str(self.etcd), "endpoint": self.etcd_endpoint,
                            "data_dir": str(data), "real_lease_expiry": True,
                            "member_key": self.member_key,
                        }
                        return
                except Exception:
                    pass
                time.sleep(0.1)
            if child.popen.poll() is None:
                raise RuntimeError(f"etcd did not become healthy; see {child.log_path}")
        raise RuntimeError("etcd failed on five independent port/data-dir attempts")

    def start_mds(self) -> None:
        metrics_port = c35._free_port()
        child = self.processes.launch(
            "mds", [str(self.args.mds), "--listen", "0",
                    "--etcd", self.etcd_endpoint, "--etcd-probe-ms", "10000",
                    "--metrics-port", str(metrics_port),
                    "--metrics-bind", "127.0.0.1"],
            {**os.environ, "DFKV_LOG": "INFO", "DFKV_MDS_FENCING": "1"},
        )
        match = self.processes.wait_log(child, r"^dfkv_mds listening on (\d+),")
        self.mds_endpoint = f"127.0.0.1:{int(match.group(1))}"
        metrics_match = self.processes.wait_log(
            child, r"dfkv_mds /metrics on port (\d+)")
        self.mds_metrics_port = int(metrics_match.group(1))
        self.result["mds"] = {
            "executable": str(self.args.mds), "endpoint": self.mds_endpoint,
            "group": self.group, "fencing_enabled": True,
            "metrics_port": self.mds_metrics_port,
            "member_lease_ttl_seconds": MDS_LEASE_TTL_SECONDS,
            "member_lease_ttl_source": "src/mds/mds_server.h kTtlSeconds",
        }

    def start_node(self, number: int) -> Node4:
        node_id = f"n{number}"
        for attempt in range(1, 6):
            port = c35._free_port()
            data = self.root / "runtime" / f"{node_id}-attempt-{attempt}"
            data.mkdir(parents=True)
            env = {**os.environ, **self.server_env()}
            argv = [
                str(self.args.server), "--dir", str(data), "--port", str(port),
                "--cap", str(c35.CAP_BYTES), "--store-engine", "file",
                "--mds", self.mds_endpoint, "--group", self.group,
                "--id", node_id, "--advertise", f"127.0.0.1:{port}",
                "--metrics-port", "0", "--metrics-bind", "127.0.0.1",
                "--mds-registration-timeout-ms", "10000",
            ]
            child = self.processes.launch(f"{node_id}-a{attempt}", argv, env)
            try:
                port_match = self.processes.wait_log(child, r"^PORT (\d+)\s*$")
                if int(port_match.group(1)) != port:
                    raise RuntimeError(f"{node_id}: listened on an unexpected port")
                metrics_match = self.processes.wait_log(
                    child, r"dfkv_server /metrics on port (\d+)")
                return Node4(node_id, self.args.server, child, data, port,
                             int(metrics_match.group(1)))
            except RuntimeError:
                if child.popen.poll() is None:
                    raise
                text = child.log_path.read_text(errors="replace")
                if f"failed to start on port {port}" not in text:
                    raise
        raise RuntimeError(f"{node_id}: no usable port in five attempts")

    def ring(self) -> dict:
        completed = subprocess.run(
            [str(self.args.ctl), "ring", "--json", "--allow-empty",
             "--group", self.group, "--mds", self.mds_endpoint],
            text=True, capture_output=True, timeout=8, check=True)
        return json.loads(completed.stdout)

    def wait_initial_membership(self) -> None:
        expected = {f"n{i}" for i in range(NODES)}

        def ready() -> bool:
            try:
                members = self.ring()["members"]
                return {m["id"] for m in members} == expected and all(
                    m["eligible"] for m in members)
            except Exception:
                return False

        c35._until("three real MDS members", ready, 40)

    def fence_victim(self, planned_s: int) -> None:
        started = time.monotonic_ns()
        target = self.origin_ns + planned_s * 1_000_000_000
        assert self.keeper is not None
        try:
            self.keeper.pre_fence_record = self.keeper.sample() or {}
        except Exception as exc:
            raise RuntimeError(f"pre-fence etcd sample failed: {exc}") from exc
        completed = subprocess.run(
            [str(self.args.ctl), "member", "fence", VICTIM,
             "--mds", self.mds_endpoint, "--group", self.group],
            text=True, capture_output=True, timeout=15)
        confirmed = time.monotonic_ns()
        fenced = subprocess.run(
            [str(self.args.ctl), "member", "fenced",
             "--mds", self.mds_endpoint, "--group", self.group],
            text=True, capture_output=True, timeout=15)
        self.fence_evidence = {
            "mechanism": "dfkvctl member fence: the MDS writes the fence marker "
                         "first, then deletes the member record from etcd; the "
                         "live victim's later heartbeats are refused, so the "
                         "record stays absent (coordinated deletion)",
            "planned_second": planned_s, "actual_start_ns": started,
            "planned_target_ns": target, "start_delay_ns": started - target,
            "schedule_tolerance_ns": SCHEDULE_TOLERANCE_NS,
            "within_schedule_tolerance": abs(started - target) <= SCHEDULE_TOLERANCE_NS,
            "confirmed_ns": confirmed,
            "workload_second": (started - self.origin_ns) / 1e9,
            "victim": VICTIM, "etcd_member_key": self.member_key,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip(),
            "fenced_list_returncode": fenced.returncode,
            "fenced_list": fenced.stdout.strip(),
            "etcd_record_before": self.keeper.pre_fence_record,
        }
        self.sinks["timeline"].write({"event": "victim_fenced_record_deleted",
                                      **self.fence_evidence})
        if completed.returncode != 0 or VICTIM not in fenced.stdout:
            raise RuntimeError(f"dfkvctl member fence failed: {self.fence_evidence}")
        self.fence_confirmed_ns = confirmed
        self.keeper.start()

    def unfence_victim(self, planned_s: int) -> None:
        started = time.monotonic_ns()
        target = self.origin_ns + planned_s * 1_000_000_000
        completed = subprocess.run(
            [str(self.args.ctl), "member", "unfence", VICTIM,
             "--mds", self.mds_endpoint, "--group", self.group],
            text=True, capture_output=True, timeout=15)
        confirmed = time.monotonic_ns()
        fenced = subprocess.run(
            [str(self.args.ctl), "member", "fenced",
             "--mds", self.mds_endpoint, "--group", self.group],
            text=True, capture_output=True, timeout=15)
        self.unfence_evidence = {
            "mechanism": "dfkvctl member unfence: the fence marker is deleted; "
                         "the same victim process re-announces itself on its "
                         "next heartbeat (DFKV_MDS_HEARTBEAT_MS=1000), "
                         "re-creating the etcd record under a fresh lease",
            "planned_second": planned_s, "actual_start_ns": started,
            "planned_target_ns": target, "start_delay_ns": started - target,
            "schedule_tolerance_ns": SCHEDULE_TOLERANCE_NS,
            "within_schedule_tolerance": abs(started - target) <= SCHEDULE_TOLERANCE_NS,
            "confirmed_ns": confirmed,
            "workload_second": (started - self.origin_ns) / 1e9,
            "victim": VICTIM, "etcd_member_key": self.member_key,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip(),
            "fenced_list_after": fenced.stdout.strip(),
        }
        self.sinks["timeline"].write({"event": "victim_unfenced_reannounce_pending",
                                      **self.unfence_evidence})
        if completed.returncode != 0:
            raise RuntimeError(f"dfkvctl member unfence failed: {self.unfence_evidence}")
        self.unfence_confirmed_ns = confirmed
        self.unfence_done = True

    def wait_client_gate(self) -> None:
        expected = {client.client_id for client in self.client_snapshot()}

        def ready() -> bool:
            clients = self.client_snapshot()
            try:
                snapshots = {client.client_id: client.stats() for client in clients}
            except Exception:
                return False
            if any(c35._metric(text, "dfkv_client_ring_members") != NODES
                   for text in snapshots.values()):
                return False
            if self.variant == "amb" and any(
                    c35._metric(text, "dfkv_client_registry_gate_open") != 1
                    for text in snapshots.values()):
                return False
            try:
                output = subprocess.run(
                    [str(self.args.ctl), "clients", "--group", self.group,
                     "--mds", self.mds_endpoint], text=True, capture_output=True,
                    timeout=8, check=True).stdout
            except Exception:
                return False
            return all(client_id in output for client_id in expected)

        c35._until("client MDS registrations and registry gate", ready, 40)
        self.sinks["timeline"].write({
            "event": "initial_clients_registered_gate_ready",
            "clients": sorted(expected), "monotonic_ns": time.monotonic_ns(),
        })

    def wait_memory_coverage(self) -> None:
        required = self.variant == "amb"
        common = {
            "required": required,
            "required_seconds": MEMORY_COVERAGE_S if required else 0,
            "remove_memory_ms": MEMORY_MS,
            "readback_ms": READBACK_MS if required else 0,
            "settle_ms": SETTLE_MS,
            "lookback_h_ms": H_MS,
            "smoke": self.args.smoke,
        }
        if not required:
            self.result["memory_coverage"] = {**common, "waited": False,
                                               "reason": f"{self.variant} has M off"}
            return

        def sample() -> list[dict]:
            observed_ns = time.monotonic_ns()
            rows = []
            for node in self.node_snapshot():
                row: dict[str, Any] = {"node": node.node_id, "key": node.key}
                try:
                    text = MetricsCollector4.scrape(node.metrics_port)
                    row["covered_seconds"] = c35._metric_sum(
                        text, "dfkv_remove_memory_covered_seconds")
                except Exception as exc:
                    row["error"] = str(exc)
                rows.append(row)
            self.sinks["memory_coverage"].write({
                "monotonic_ns": observed_ns, "wall_time": c35._iso(),
                "required_seconds": MEMORY_COVERAGE_S, "nodes": rows,
            })
            return rows

        if self.args.smoke:
            rows = sample()
            self.result["memory_coverage"] = {
                **common, "waited": False, "warmup_bypassed": True,
                "non_acceptance": True, "nodes": rows,
                "reason": "smoke explicitly skips the 120 s warmup; read-backs "
                          "fail closed as unknown (coverage < H): mechanics only",
            }
            return

        latest: list[dict] = []
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            latest = sample()
            if len(latest) == NODES and all(
                    row.get("covered_seconds") is not None
                    and row["covered_seconds"] >= MEMORY_COVERAGE_S
                    for row in latest):
                self.result["memory_coverage"] = {
                    **common, "waited": True, "warmup_bypassed": False,
                    "ready_monotonic_ns": time.monotonic_ns(),
                    "ready_before_workload_origin": True,
                    "nodes": latest,
                }
                return
            time.sleep(1.0)
        raise RuntimeError(
            f"nodes did not reach {MEMORY_COVERAGE_S}s real M coverage; last={latest}")

    def read_task(self, second: int, prefix: int) -> None:
        result: dict[str, Any] = {}
        try:
            assert self.reader is not None
            get = self.reader.prefix_get(prefix, phase="workload", second=second)
            result.update(get)
            if get["first_miss"] < BLOCKS:
                oks = self.reader.prefix_put(prefix, get["first_miss"],
                                             phase="workload", second=second)
                result["put_keys"] = len(oks)
                result["put_ok"] = sum(oks)
        except BaseException as exc:
            result.update({
                "error": f"{type(exc).__name__}: {exc}", "full_hit": False,
                "leading_hits": 0, "hits": 0, "payload_mismatches": 0,
            })
            self.errors.put(f"reader second={second} prefix={prefix}: {exc}")
        result.update({"second": second, "prefix": prefix,
                       "client": self.reader.client_id if self.reader else ""})
        result.update(self.stats.start_evidence(second, result.get("start_ns")))
        self.sinks["prefix_results"].write(result)
        self.stats.record(second, result)

    def check_errors(self) -> None:
        try:
            error = self.errors.get_nowait()
        except queue.Empty:
            return
        others = []
        while True:
            try:
                others.append(self.errors.get_nowait())
            except queue.Empty:
                break
        raise RuntimeError("; ".join([error, *others]))

    def join_readers(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None

    def execute(self) -> None:
        self.start_etcd()
        self.start_mds()
        for number in range(NODES):
            self.nodes.append(self.start_node(number))
        self.wait_initial_membership()
        self.sinks["timeline"].write({
            "event": "initial_ring_ready", "members": self.ring()["members"],
            "monotonic_ns": time.monotonic_ns(),
        })
        self.apply_client_env()
        loader = self.open_client("loader", 0)
        self.loader = loader
        for prefix in range(PREFIXES):
            oks = loader.prefix_put(prefix, 0, phase="initial", second=None)
            if not all(oks):
                raise RuntimeError(f"initial load failed at prefix {prefix}: {oks}")
        self.result["initial_load"] = {
            "keys": PREFIXES * BLOCKS, "bytes": PREFIXES * BLOCKS * BLOCK_BYTES,
            "all_puts_ok": True, "scope_excluded_from_violation_put_witnesses": True,
        }
        self.close_client(loader)
        self.loader = None
        self.reader = self.open_client("reader", 0)
        remover0_client = self.open_client("remover", 0)
        self.wait_client_gate()
        self.wait_memory_coverage()
        self.keeper = RecordKeeper(self, self.etcd_endpoint, self.member_key)
        self.metrics = MetricsCollector4(self, 0)
        self.metrics.capture("baseline")
        self.result["victim_baseline"] = self._victim_baseline()
        self.origin_ns = time.monotonic_ns()
        self.result["workload_origin"] = {
            "monotonic_ns": self.origin_ns, "wall_time_ns": time.time_ns(),
            "wall_time": c35._iso(), "stamped_after_baseline_capture": True,
        }
        self.metrics.origin_ns = self.origin_ns
        self.stats.configure(self.origin_ns, self.timing["end_s"])
        self.result["schedule_tolerance_ns"] = SCHEDULE_TOLERANCE_NS
        self.metrics.start()
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=READS_PER_SECOND, thread_name_prefix="case4-reader")
        remover0 = c35.Remover(self, remover0_client, 0, 0)
        self.removers.append(remover0)
        remover0.start()
        self.sinks["timeline"].write({
            "event": "workload_start", "monotonic_ns": self.origin_ns,
            "planned": self.timing,
        })
        for second in range(self.timing["end_s"]):
            c35._wait_until(self.origin_ns, second)
            if second == self.timing["fence_s"]:
                self.fence_victim(second)
            if second == self.timing["unfence_s"]:
                self.unfence_victim(second)
            assert self.executor is not None
            for prefix in Workload4.indices("reader", second, READS_PER_SECOND, PREFIXES):
                self.executor.submit(self.read_task, second, prefix)
            self.check_errors()
        c35._wait_until(self.origin_ns, self.timing["end_s"])
        self.join_readers()
        for remover in self.removers:
            remover.join_finished()
        self.check_errors()
        self.metrics.stop_and_join()
        self.metrics.capture("final")
        self._final_captured = True
        self.metrics.poll_ring(final=True)
        assert self.keeper is not None
        self.keeper.stop_and_join()
        self._collect_results()

    def _victim_baseline(self) -> dict:
        victim = self.victim_node()
        assert self.metrics is not None and self.keeper is not None
        text = self.metrics.last_server_raw.get(victim.key, "")
        return {
            "node": victim.node_id, "key": victim.key,
            "pid": victim.child.popen.pid, "port": victim.port,
            "metrics_port": victim.metrics_port, "data_dir": str(victim.data_dir),
            "server_core": _server_core4(text),
            "etcd_member_record": self.keeper.pre_fence_record or "captured at fence time",
        }

    # ---- offline evidence evaluation ----

    @staticmethod
    def _series(rows: list[dict], group: str, key: str,
                field: str) -> list[tuple[int, float]]:
        points = []
        for row in rows:
            entry = row.get(group, {}).get(key, {})
            if "error" in entry:
                continue
            value = entry.get(field)
            if isinstance(value, (int, float)):
                points.append((int(row["monotonic_ns"]), float(value)))
        return points

    @staticmethod
    def _value_at(points: list[tuple[int, float]], ns: int) -> float | None:
        answer = None
        for point_ns, value in points:
            if point_ns <= ns:
                answer = value
            else:
                break
        return answer

    @classmethod
    def _delta(cls, points: list[tuple[int, float]], start_ns: int,
               end_ns: int) -> dict:
        first = cls._value_at(points, start_ns)
        last = cls._value_at(points, end_ns)
        complete = first is not None and last is not None
        return {
            "start_ns": start_ns, "end_ns": end_ns,
            "start_value": first, "end_value": last,
            "delta": (last - first) if complete else None,
            "complete": complete,
        }

    def _collect_results(self) -> None:
        assert self.metrics is not None and self.keeper is not None
        gone_ns = self.metrics.gone_ns
        back_ns = self.metrics.back_ns
        origin = self.origin_ns
        end_ns = origin + self.timing["end_s"] * 1_000_000_000
        gone_s = int((gone_ns - origin) // 1e9) if gone_ns else 0
        back_s = int((back_ns - origin) // 1e9) if back_ns else 0
        margin = MARGIN_S * 1_000_000_000

        rows = [json.loads(line)
                for line in (self.root / "metrics.jsonl").open(encoding="utf-8")
                if line.strip()]
        victim_key = self.victim_node().key
        victim = {
            field: self._series(rows, "servers", victim_key, field)
            for field in ("uptime_seconds", "cache_put", "cache_hit", "cache_miss",
                          "exist_hit", "exist_miss", "remove_ok", "remove_miss",
                          "removed_within_queries", "removed_within_ages_requests",
                          "remove_memory_covered_seconds", "mds_fenced",
                          "mds_heartbeat_healthy", "mds_heartbeat_failures_consecutive")
        }
        victim_error_rows = sum(
            1 for row in rows if "error" in row.get("servers", {}).get(victim_key, {}))
        # mds rows are stored under row["mds"] without a per-key layer.
        mds_series = {
            field: [(int(row["monotonic_ns"]), float(row["mds"][field]))
                    for row in rows
                    if isinstance(row.get("mds"), dict)
                    and isinstance(row["mds"].get(field), (int, float))]
            for field in ("fenced_refusals", "record_puts", "lease_grants")
        }
        client_ids = sorted({cid for row in rows for cid in row.get("clients", {})})
        client_series = {
            cid: {field: self._series(rows, "clients", cid, field)
                  for field in ("ring_members", "registry_gate_open",
                                "absent_members", "readback_hit",
                                "readback_refused_removed", "readback_refused_unknown")}
            for cid in client_ids
        }

        def summed_client(field: str) -> list[tuple[int, float]]:
            points = []
            for row in rows:
                total = 0.0
                complete = True
                for cid in client_ids:
                    entry = row.get("clients", {}).get(cid, {})
                    value = entry.get(field) if "error" not in entry else None
                    if not isinstance(value, (int, float)):
                        complete = False
                        break
                    total += float(value)
                if complete:
                    points.append((int(row["monotonic_ns"]), total))
            return points

        readback_hit_series = summed_client("readback_hit")
        refused_removed_series = summed_client("readback_refused_removed")
        refused_unknown_series = summed_client("readback_refused_unknown")
        gate_min_absence = None
        absent_members_min = None
        absent_members_max_after_adoption = None
        if gone_ns and back_ns:
            gate_values = [value for series in client_series.values()
                           for ns, value in series["registry_gate_open"]
                           if gone_ns <= ns <= back_ns]
            gate_min_absence = min(gate_values) if gate_values else None
            absent_values = [value for series in client_series.values()
                             for ns, value in series["absent_members"]
                             if gone_ns + margin <= ns <= back_ns - margin]
            absent_members_min = min(absent_values) if absent_values else None
        if back_ns:
            post_adoption = [value for series in client_series.values()
                             for ns, value in series["absent_members"]
                             if ns >= back_ns + 2 * 1e9]
            absent_members_max_after_adoption = (
                max(post_adoption) if post_adoption else None)

        # Per-second rate windows (workload seconds).
        def rates(start_s: int, end_s: int) -> dict:
            window_rows = [self.stats.rows[s] for s in range(start_s, end_s)
                           if s in self.stats.rows]
            per_second = {s: self.stats.rows[s]["full_prefix_hit_rate"]
                          for s in range(start_s, end_s) if s in self.stats.rows}
            complete = len(window_rows) == end_s - start_s
            mean = (sum(r["full_prefixes"] for r in window_rows)
                    / sum(r["prefix_results"] for r in window_rows)) if window_rows else None
            bad = [s for s, rate in per_second.items() if rate <= DISRUPTION_BAD_RATE]
            return {"start_s": start_s, "end_s_exclusive": end_s,
                    "complete": complete, "observed_seconds": len(window_rows),
                    "mean_full_prefix_rate": mean, "per_second": per_second,
                    "seconds_at_or_below_50pct": sorted(bad)}

        if self.args.smoke:
            unavailable = {"complete": False, "available": False,
                           "reason": "window exceeds shortened smoke timeline; non-acceptance"}
            self.result["core"] = {
                "pre_absence": dict(unavailable),
                "absence_first10s": dict(unavailable),
                "absence_over_h_first10s": dict(unavailable),
                "absence_steady20s": dict(unavailable),
                "post_restore_15_35s": dict(unavailable),
                "whole_run": self.stats.window(0, self.timing["end_s"]),
            }
        else:
            self.result["core"] = {
                "pre_absence": self.stats.window(10, 25),
                "absence_first10s": self.stats.window(gone_s + 3, gone_s + 13),
                "absence_over_h_first10s": self.stats.window(gone_s + 63, gone_s + 73),
                "absence_steady20s": self.stats.window(gone_s + 13, gone_s + 33),
                "post_restore_15_35s": self.stats.window(back_s + 15, back_s + 35),
                "whole_run": self.stats.window(0, self.timing["end_s"]),
            }
            self.result["rate_windows"] = {
                "absence_first10s": rates(gone_s + 3, gone_s + 13),
                "absence_over_h_first10s": rates(gone_s + 63, gone_s + 73),
                "absence_steady20s": rates(gone_s + 13, gone_s + 33),
                "post_restore_15_35s": rates(back_s + 15, back_s + 35),
            }

        self.result["fence_evidence"] = self.fence_evidence
        self.result["unfence_evidence"] = self.unfence_evidence
        self.result["timeline_evidence"] = {
            "tolerance_ns": SCHEDULE_TOLERANCE_NS,
            "fence": self.fence_evidence,
            "unfence": self.unfence_evidence,
            "all_within_tolerance": all(
                event.get("within_schedule_tolerance") is True
                for event in (self.fence_evidence, self.unfence_evidence)),
        }
        membership_chain = (
            self.fence_confirmed_ns > 0 and gone_ns > 0
            and self.unfence_confirmed_ns > 0 and self.keeper.reannounce_ns > 0
            and back_ns > 0 and self.metrics.final_ring_ns > 0
            and self.fence_confirmed_ns <= gone_ns <= self.unfence_confirmed_ns
            < self.keeper.reannounce_ns
            and self.keeper.reannounce_ns - int((KEEPER_PERIOD_S + 0.5) * 1e9)
            <= back_ns <= self.metrics.final_ring_ns
            and self.metrics.final_victim_eligible)
        self.result["membership_evidence"] = {
            "fence_confirmed_ns": self.fence_confirmed_ns,
            "victim_absent_ns": gone_ns,
            "victim_absence": self.metrics.gone_record,
            "unfence_confirmed_ns": self.unfence_confirmed_ns,
            "record_reannounced_ns": self.keeper.reannounce_ns,
            "record_reannounced": self.keeper.reannounce_record,
            "reannounce_latency_ms": ((self.keeper.reannounce_ns
                                        - self.unfence_confirmed_ns) / 1e6
                                       if self.keeper.reannounce_ns else None),
            "victim_eligible_ns": back_ns,
            "victim_eligible": self.metrics.back_record,
            "final_ring_ns": self.metrics.final_ring_ns,
            "final_victim_eligible": self.metrics.final_victim_eligible,
            "final_victim_member": self.metrics.final_victim_member,
            "transitions": self.metrics.transitions,
            "absence_observed_seconds": ((back_ns - gone_ns) / 1e9
                                          if gone_ns and back_ns else None),
            "h_seconds": H_MS / 1000,
            "absence_exceeds_h": (back_ns - gone_ns) * 1e-9 > H_MS / 1000
            if gone_ns and back_ns else None,
            "chain_valid": membership_chain,
        }
        self.result["etcd_record_evidence"] = {
            "member_key": self.member_key,
            "pre_fence_record": self.keeper.pre_fence_record,
            "pre_fence_present_with_real_lease": bool(
                self.keeper.pre_fence_record
                and self.keeper.pre_fence_record.get("lease", "0") != "0"),
            "keeper_started_ns": self.keeper.started_ns,
            "keeper_samples": self.keeper.samples,
            "absent_samples_while_fenced": self.keeper.absent_samples,
            "first_absent_ns": self.keeper.first_absent_ns,
            "recreated_and_redeleted": self.keeper.recreated_and_redeleted,
            "recreated_and_redeleted_count": len(self.keeper.recreated_and_redeleted),
            "reannounce_ns": self.keeper.reannounce_ns,
            "reannounce_record": self.keeper.reannounce_record,
            "reannounce_under_fresh_lease": bool(
                self.keeper.reannounce_record
                and self.keeper.reannounce_record.get("lease", "0") != "0"),
            "keeper_errors": self.keeper.errors,
            "real_lease_expiry": {
                "private_real_etcd": True,
                "member_lease_ttl_seconds": MDS_LEASE_TTL_SECONDS,
                "evidence": "member records carry real etcd leases (pre-fence "
                            "and reannounce records above, lease != 0) and the "
                            "MDS issues real grants (dfkv_mds_lease_grants_total); "
                            "the fence path itself deletes the key in etcd",
            },
        }
        uptime = [value for _, value in victim["uptime_seconds"]]
        # Heartbeat-health windows end a second before the unfence: the first
        # successful heartbeat may land within one heartbeat interval (1 s) of
        # it, and the MDS fence cache (1 s) may admit the first heartbeat after
        # the fence, so both boundaries are trimmed.
        hc_start = gone_ns + margin if gone_ns else 0
        hc_end = (self.unfence_confirmed_ns - 1_000_000_000
                  if self.unfence_confirmed_ns else 0)
        hc_valid = bool(hc_start and hc_end and hc_start < hc_end)
        fenced_absence = [value for ns, value in victim["mds_fenced"]
                          if hc_valid and hc_start <= ns <= hc_end]
        healthy_absence = [value for ns, value in victim["mds_heartbeat_healthy"]
                           if hc_valid and hc_start <= ns <= hc_end]
        failures = victim["mds_heartbeat_failures_consecutive"]
        failure_at = lambda ns: self._value_at(failures, ns)  # noqa: E731
        self.result["victim_evidence"] = {
            "node": VICTIM, "key": victim_key,
            "pid": self.victim_node().child.popen.pid,
            "single_generation_same_process": True,
            "never_signaled": True,
            "metrics_error_rows": victim_error_rows,
            "metrics_rows": len(rows),
            "uptime_monotonic_non_decreasing": all(
                b >= a for a, b in zip(uptime, uptime[1:])),
            "uptime_first": uptime[0] if uptime else None,
            "uptime_last": uptime[-1] if uptime else None,
            "health_window_valid": hc_valid,
            "health_window": {"start_ns": hc_start, "end_ns": hc_end},
            "fenced_during_absence_min": min(fenced_absence) if fenced_absence else None,
            "heartbeat_healthy_during_absence_max": (
                max(healthy_absence) if healthy_absence else None),
            "heartbeat_failures_at_gone": failure_at(gone_ns) if gone_ns else None,
            # The successful post-unfence heartbeat can reset this gauge before
            # the client observes n2 back in the ring.  Anchor the evidence at
            # the unfence reply, before any possible reset.
            "heartbeat_failures_at_unfence": (
                failure_at(self.unfence_confirmed_ns)
                if self.unfence_confirmed_ns else None),
            "heartbeat_failures_at_back": failure_at(back_ns) if back_ns else None,
            "mds_fenced_refusals": self._delta(
                mds_series["fenced_refusals"], origin, end_ns),
            "mds_lease_grants_final": self._value_at(mds_series["lease_grants"], end_ns),
        }

        # Direct stale-copy evidence: the victim's own read counters.  Windows
        # are clamped to the observed membership events so a shortened (smoke)
        # absence never inverts them; an over-H window exists only when the
        # absence really crossed H.
        fence_ns = self.fence_confirmed_ns or origin + self.timing["fence_s"] * 1e9
        pre = ({"start_ns": origin + 2 * 1e9, "end_ns": fence_ns - 2 * 1e9}
               if fence_ns - 2 * 1e9 > origin + 2 * 1e9 else None)
        within_h = None
        if gone_ns and back_ns:
            wh_end = min(gone_ns + (H_MS / 1000 - MARGIN_S) * 1e9, back_ns)
            if gone_ns + margin < wh_end:
                within_h = {"start_ns": gone_ns + margin, "end_ns": wh_end}
        over_h = None
        if gone_ns and back_ns:
            oh_start = gone_ns + (H_MS / 1000 + MARGIN_S) * 1e9
            if oh_start < back_ns:
                over_h = {"start_ns": oh_start, "end_ns": back_ns}
        post = ({"start_ns": back_ns + margin, "end_ns": back_ns + 25 * 1e9}
                if back_ns and back_ns + margin < end_ns else None)
        freeze_delay_ns = (READBACK_MS + SETTLE_MS + MARGIN_S * 1000) * 1_000_000
        freeze = ({"start_ns": back_ns + freeze_delay_ns, "end_ns": end_ns}
                  if back_ns and back_ns + freeze_delay_ns < end_ns else None)

        def counter_windows(field_points: list[tuple[int, float]]) -> dict:
            answer: dict[str, Any] = {}
            for name, window in (("pre_absence", pre),
                                 ("within_h", within_h),
                                 ("over_h", over_h),
                                 ("post_restore", post)):
                answer[name] = (self._delta(field_points, window["start_ns"],
                                            window["end_ns"])
                                if window else {
                                    "complete": False,
                                    "reason": "window does not exist for this "
                                              "timeline (e.g. smoke absence "
                                              "never crosses H)"})
            return answer

        self.result["destination_evidence"] = {
            "note": "deltas of the victim's own /metrics counters over "
                    "membership-referenced windows; the victim process and its "
                    "metrics endpoint are never touched",
            "victim_cache_hit": counter_windows(victim["cache_hit"]),
            "victim_exist_hit": counter_windows(victim["exist_hit"]),
            "victim_remove_ok_plus_miss": counter_windows(
                [(ns, ok + miss)
                 for (ns, ok), (_, miss) in zip(victim["remove_ok"],
                                                 victim["remove_miss"])]
                if len(victim["remove_ok"]) == len(victim["remove_miss"]) else []),
            "client_readback_hit": counter_windows(readback_hit_series),
            "client_readback_refused_removed": counter_windows(refused_removed_series),
            "client_readback_refused_unknown": counter_windows(refused_unknown_series),
            "readback_hit_freeze_after_restore": (
                self._delta(readback_hit_series, freeze["start_ns"], freeze["end_ns"])
                if freeze else {"complete": False}),
            "gate_min_during_absence": gate_min_absence,
            "absent_members_min_during_absence": absent_members_min,
            "absent_members_max_after_two_polls": absent_members_max_after_adoption,
            "held_ring_freeze_start_ns": freeze["start_ns"] if freeze else None,
        }

        self._correctness = c35.check_operations(self.root / "operations.jsonl")
        if self.variant == "base" and back_ns:
            # A client may adopt the restored ring up to one poll before the
            # harness's ring poller observes it, so the warm-return window
            # starts 3 s before the observed transition.  No removed_value_hit
            # can occur earlier in base: during the absence routing excludes
            # the victim, so every hit is a legitimately re-PUT copy.
            warm_threshold_ns = back_ns - 3_000_000_000
            warm, unexpected = [], []
            for violation in self._correctness["violations"]:
                read_start = violation.get("read", {}).get("start_ns", 0)
                if (violation.get("type") == "removed_value_hit"
                        and isinstance(read_start, int)
                        and read_start > warm_threshold_ns):
                    warm.append(violation)
                else:
                    unexpected.append(violation)
            self._correctness["warm_return_threshold_ns"] = warm_threshold_ns
            self._correctness["warm_return_today_behavior_violations"] = warm
            self._correctness["warm_return_today_behavior_count"] = len(warm)
            self._correctness["unexpected_violations"] = unexpected
            self._correctness["unexpected_violation_count"] = len(unexpected)
            self._correctness["warm_return_classification"] = (
                "base (A off) sends no REMOVE to the departed member, so the "
                "restored same process may serve copies whose REMOVE it never "
                "saw: design §3.1/§7 'warm return with unconfirmed REMOVEs', "
                "today's behaviour, counted separately and never masked")
        else:
            self._correctness["unexpected_violations"] = self._correctness["violations"]
            self._correctness["unexpected_violation_count"] = \
                self._correctness["violation_count"]
        self.result["correctness"] = self._correctness

        removers = []
        for remover in self.removers:
            removers.append({
                "client": remover.client.client_id,
                "start_second": remover.start_second,
                "removed": remover.removed,
                "rewrite_scheduled": remover.rewrite_scheduled,
                "rewritten": remover.rewritten,
                "after_cutoff_rewrites": remover.after_cutoff_rewrites,
                "pending_rewrite_not_run_after_end": len(remover.pending_rewrite),
                "operation_timing": remover.operation_timing,
                "remove_keys_by_actual_second": remover.remove_keys_by_actual_second,
                "timing_violations": remover.timing_violations,
                "max_rewrite_lateness_ns": max(remover.rewrite_lateness_ns, default=0),
                "actual_churn_timing_valid": (
                    bool(remover.operation_timing)
                    and not remover.timing_violations
                    and not remover.pending_rewrite
                    and remover.rewrite_scheduled == remover.rewritten
                    and all(row["actual_start_valid"]
                            for row in remover.operation_timing)
                    and all(count <= c35.REMOVES_PER_SECOND
                            for count in remover.remove_keys_by_actual_second.values())
                    and max(remover.rewrite_lateness_ns, default=0)
                    <= c35.REWRITE_LATENESS_TOLERANCE_NS),
            })
        self.result["churn"] = {
            "remove_keys_per_actual_second_cap": c35.REMOVES_PER_SECOND,
            "rewrite_lateness_tolerance_ns": c35.REWRITE_LATENESS_TOLERANCE_NS,
            "removers": removers,
        }
        client_metrics = {client_id: c35._client_core(text)
                          for client_id, text in self.metrics.last_client_raw.items()}
        server_metrics = []
        for node in self.node_snapshot():
            text = self.metrics.last_server_raw.get(node.key, "")
            server_metrics.append({
                "node": node.node_id, "key": node.key,
                "binary": str(node.binary), "data_dir": str(node.data_dir),
                "metrics_available": bool(text), **_server_core4(text),
            })
        readback = {
            label: sum(value or 0 for value in (
                metrics.get(f"readback_{label}") for metrics in client_metrics.values()))
            for label in c35.READBACK_RESULTS
        }
        self.result["metrics"] = {
            "clients": client_metrics,
            "client_readback_totals": readback,
            "servers": server_metrics,
            "mds": _mds_core(self.metrics.last_mds_raw),
            "final_capture_complete": self.metrics.final_complete,
            "every_node_has_metrics": all(
                row["metrics_available"] for row in server_metrics),
        }
        self.result["evidence"] = {
            "operations": "operations.jsonl", "per_second": "per_second.jsonl",
            "prefix_results": "prefix_results.jsonl", "timeline": "timeline.jsonl",
            "metrics": "metrics.jsonl and metrics/*.prom",
            "membership": "membership.jsonl (ring polls + etcd record keeper)",
            "processes": "processes.jsonl",
            "memory_coverage": "memory_coverage.jsonl",
            "logs": "logs/ (retained, including the never-restarted victim)",
            "victim_data_dir": str(self.victim_node().data_dir),
        }

    def cleanup(self) -> list[str]:
        if self._cleaned_up:
            return []
        self._cleaned_up = True
        errors = []
        for remover in self.removers:
            remover.stop_event.set()
        try:
            self.join_readers()
        except Exception as exc:
            errors.append(f"reader cleanup: {exc}")
        for remover in self.removers:
            try:
                remover.stop_and_join()
            except Exception as exc:
                errors.append(f"remover cleanup: {exc}")
        if self.keeper is not None:
            try:
                self.keeper.stop_and_join()
            except Exception as exc:
                errors.append(f"keeper cleanup: {exc}")
        if self.metrics is not None:
            try:
                self.metrics.stop_and_join()
            except Exception as exc:
                errors.append(f"metrics cleanup: {exc}")
            if not self._final_captured:
                try:
                    self.metrics.capture("final")
                    self._final_captured = True
                except Exception as exc:
                    errors.append(f"final metrics: {exc}")
        for client in self.client_snapshot():
            try:
                client.close()
            except Exception as exc:
                errors.append(f"client {client.client_id} close: {exc}")
            finally:
                with self.clients_lock:
                    self.clients.pop(client.client_id, None)
        if self.loader is not None:
            try:
                self.loader.close()
            except Exception as exc:
                errors.append(f"loader close: {exc}")
            self.loader = None
        self.restore_client_env()
        try:
            self.processes.close()
        except Exception as exc:
            errors.append(f"process cleanup: {exc}")
        live = [child.name for child in self.processes.children
                if child.popen.poll() is None]
        if live:
            errors.append(f"processes still alive: {live}")
        live_threads = [thread.name for thread in threading.enumerate()
                        if thread.name.startswith(
                            ("case4-reader", "remover-", "metrics", "record-keeper"))
                        and thread.is_alive() and thread is not threading.current_thread()]
        if live_threads:
            errors.append(f"threads still alive: {live_threads}")
        for sink in self.sinks.values():
            try:
                sink.close()
            except Exception as exc:
                errors.append(f"evidence close: {exc}")
        self.result["cleanup"] = {
            "native_handles_open": len(self.clients),
            "processes_alive": live, "worker_threads_alive": live_threads,
            "errors": errors,
        }
        return errors

    def run(self) -> dict:
        try:
            self.execute()
        except BaseException as exc:
            self.result["errors"].append(f"{type(exc).__name__}: {exc}")
            (self.root / "error.log").write_text(traceback.format_exc())
        cleanup_errors = self.cleanup()
        self.result["errors"].extend(cleanup_errors)
        self.result["status"] = "ERROR" if self.result["errors"] else "COMPLETE"
        c35._write_json(self.root / "result.json", self.result)
        return self.result


def selected_checks4(result: dict, smoke: bool) -> None:
    variant = result["variant"]
    add = lambda name, passed, detail: c35.add_check(result, name, passed, detail)  # noqa: E731
    correctness = result.get("correctness", {})
    add("zero_unexpected_violations",
        correctness.get("unexpected_violation_count") == 0,
        {"count": correctness.get("unexpected_violation_count"),
         "first": correctness.get("unexpected_violations", [])[:3],
         "warm_return_today_behavior_count":
         correctness.get("warm_return_today_behavior_count"),
         "rule": "base: hits on keys whose REMOVE the restored process never "
                 "saw are §3.1 today's behaviour, counted separately; a/amb "
                 "and everything before the restore: zero"})
    add("complete_operation_intervals",
        correctness.get("incomplete_attempt_count") == 0,
        correctness.get("incomplete_attempts", [])[:3])
    add("operation_identity_evidence_clean",
        correctness.get("evidence_error_count") == 0,
        {"count": correctness.get("evidence_error_count"),
         "first": correctness.get("evidence_errors", [])[:3]})
    coverage_windows = [result.get("core", {}).get("whole_run", {})] if smoke else [
        result.get("core", {}).get("whole_run", {}),
        result.get("core", {}).get("pre_absence", {})]
    full_coverage = all(window.get("complete") and not window.get("errors")
                        for window in coverage_windows)
    add("complete_per_second_workload", full_coverage, result.get("core", {}))
    churn = result.get("churn", {})
    removers = churn.get("removers", [])
    add("actual_churn_starts_caps_and_rewrite_deadlines",
        len(removers) == 1
        and churn.get("remove_keys_per_actual_second_cap") == c35.REMOVES_PER_SECOND
        and all(remover.get("actual_churn_timing_valid") is True
                for remover in removers),
        {"removers": [{
            "client": remover.get("client"),
            "valid": remover.get("actual_churn_timing_valid"),
            "max_rewrite_lateness_ns": remover.get("max_rewrite_lateness_ns"),
            "rewrite_scheduled": remover.get("rewrite_scheduled"),
            "rewritten": remover.get("rewritten"),
            "timing_violations": remover.get("timing_violations", [])[:5],
        } for remover in removers]})
    timeline = result.get("timeline_evidence", {})
    add("fence_unfence_within_schedule_tolerance",
        timeline.get("all_within_tolerance") is True
        and timeline.get("tolerance_ns") == SCHEDULE_TOLERANCE_NS, timeline)
    member = result.get("membership_evidence", {})
    add("ordered_absence_reannounce_and_final_membership",
        member.get("chain_valid") is True, member)
    add("record_reannounced_promptly_by_same_process",
        member.get("reannounce_latency_ms") is not None
        and member["reannounce_latency_ms"] <= 15000
        and member.get("record_reannounced", {}).get("lease", "0") != "0",
        {"reannounce_latency_ms": member.get("reannounce_latency_ms"),
         "record": member.get("record_reannounced"),
         "heartbeat_ms": HEARTBEAT_MS})
    etcd_ev = result.get("etcd_record_evidence", {})
    add("real_etcd_record_deleted_with_lease_and_kept_absent",
        etcd_ev.get("pre_fence_present_with_real_lease") is True
        and etcd_ev.get("recreated_and_redeleted_count") == 0
        and not etcd_ev.get("keeper_errors")
        and etcd_ev.get("absent_samples_while_fenced", 0) > 0
        and etcd_ev.get("reannounce_under_fresh_lease") is True,
        etcd_ev)
    victim = result.get("victim_evidence", {})
    add("victim_process_never_touched_and_live",
        victim.get("metrics_error_rows") == 0
        and victim.get("uptime_monotonic_non_decreasing") is True
        and victim.get("single_generation_same_process") is True
        and victim.get("health_window_valid") is True
        and victim.get("fenced_during_absence_min") == 1
        and victim.get("heartbeat_healthy_during_absence_max") == 0
        and (victim.get("heartbeat_failures_at_unfence") or 0)
        > (victim.get("heartbeat_failures_at_gone") or 0)
        and (victim.get("mds_fenced_refusals", {}).get("delta") or 0) > 0,
        victim)
    baseline = result.get("victim_baseline", {}).get("server_core", {})
    add("victim_was_a_live_owner",
        (baseline.get("cache_put") or 0) > 0
        and (result.get("destination_evidence", {})
             .get("victim_cache_hit", {}).get("pre_absence", {}).get("delta") or 0) > 0,
        {"baseline": baseline,
         "pre_absence_reads": result.get("destination_evidence", {})
         .get("victim_cache_hit", {}).get("pre_absence", {})})
    dest = result.get("destination_evidence", {})
    victim_hit = dest.get("victim_cache_hit", {})
    victim_exist = dest.get("victim_exist_hit", {})
    metrics = result.get("metrics", {})
    servers = metrics.get("servers", [])
    queries = sum(row.get("removed_within_queries") or 0 for row in servers)
    ages = sum(row.get("removed_within_ages_requests") or 0 for row in servers)
    totals = metrics.get("client_readback_totals", {})

    if variant in ("base", "a"):
        over_h_ok = (
            victim_hit.get("over_h", {}).get("complete") is not True
            or (victim_hit.get("over_h", {}).get("delta") == 0
                and victim_exist.get("over_h", {}).get("delta") == 0))
        add("base_or_A_never_reads_the_stale_copy",
            victim_hit.get("within_h", {}).get("complete") is True
            and victim_hit.get("within_h", {}).get("delta") == 0
            and victim_exist.get("within_h", {}).get("delta") == 0
            and over_h_ok,
            {"victim_cache_hit": victim_hit, "victim_exist_hit": victim_exist,
             "note": "over_h window absent only when the absence never crossed "
                     "H (smoke); whenever it exists the victim's read counters "
                     "must stay frozen across it"})
        add("no_readback_machinery_active",
            all(totals.get(label, 0) == 0 for label in c35.READBACK_RESULTS)
            and queries == 0 and ages == 0,
            {"client_readback_totals": totals,
             "removed_within_queries": queries, "ages_requests": ages})
    if variant == "base":
        removes = dest.get("victim_remove_ok_plus_miss", {})
        over_h_removes_ok = (removes.get("over_h", {}).get("complete") is not True
                             or removes.get("over_h", {}).get("delta") == 0)
        add("base_sends_no_removes_to_departed_member",
            removes.get("within_h", {}).get("complete") is True
            and removes.get("within_h", {}).get("delta") == 0
            and over_h_removes_ok,
            removes)
    if variant == "a":
        add("A_keeps_member_absent_and_reaches_it_with_removes",
            (dest.get("absent_members_min_during_absence") or 0) >= 1
            and (dest.get("victim_remove_ok_plus_miss", {})
                 .get("within_h", {}).get("delta") or 0) > 0,
            {"absent_members_min": dest.get("absent_members_min_during_absence"),
             "victim_removes": dest.get("victim_remove_ok_plus_miss", {})})
    if variant == "amb":
        add("gate_open_and_member_absent_throughout_absence",
            dest.get("gate_min_during_absence") == 1
            and (dest.get("absent_members_min_during_absence") or 0) >= 1,
            {"gate_min": dest.get("gate_min_during_absence"),
             "absent_members_min": dest.get("absent_members_min_during_absence")})
        add("op25_only_readback_path_exercised_no_op26",
            queries > 0 and ages == 0,
            {"removed_within_queries": queries, "ages_requests": ages,
             "note": "case 4 has W off everywhere: B asks op25 only"})
        if smoke:
            add("smoke_readback_fail_closed_path_exercised",
                totals.get("hit", 0) + totals.get("refused_removed", 0)
                + totals.get("refused_unknown", 0) > 0,
                {"client_readback_totals": totals,
                 "note": "smoke bypasses the M warmup, so read-backs fail "
                         "closed (unknown): mechanics only, non-acceptance"})
    if smoke:
        coverage = result.get("memory_coverage", {})
        if variant == "amb":
            add("smoke_memory_warmup_bypass_is_explicitly_non_acceptance",
                coverage.get("warmup_bypassed") is True
                and coverage.get("non_acceptance") is True,
                coverage)
        add("smoke_over_h_window_not_reached_is_explicitly_non_acceptance",
            result.get("membership_evidence", {}).get("absence_exceeds_h") is not True,
            {"note": "the smoke absence (~20 s) must never cross H (60 s); the "
                     "over-H refusal control is not evaluated in smoke.  This "
                     "check FAILS if a smoke run somehow observed an over-H "
                     "absence, which would mean the smoke label is wrong",
             "absence_observed_seconds":
             result.get("membership_evidence", {}).get("absence_observed_seconds"),
             "h_seconds": H_MS / 1000})
        return

    # ---- full-timeline acceptance checks below ----
    add("absence_longer_than_h",
        result.get("membership_evidence", {}).get("absence_exceeds_h") is True,
        result.get("membership_evidence", {}))
    if variant != "amb":
        coverage = result.get("memory_coverage", {})
        add("memory_coverage_not_required", coverage.get("required") is False, coverage)
    else:
        coverage = result.get("memory_coverage", {})
        nodes = coverage.get("nodes", [])
        add("all_nodes_have_full_real_memory_coverage_before_origin",
            coverage.get("waited") is True
            and coverage.get("warmup_bypassed") is False
            and len(nodes) == NODES
            and all((row.get("covered_seconds") or 0) >= MEMORY_COVERAGE_S
                    for row in nodes)
            and coverage.get("ready_monotonic_ns", 0)
            <= result.get("workload_origin", {}).get("monotonic_ns", 0),
            coverage)
    core = result.get("core", {})
    rate_windows = result.get("rate_windows", {})
    if variant in ("base", "a"):
        first10 = rate_windows.get("absence_first10s", {})
        add("absence_visibly_disrupts_prefix_reads",
            first10.get("complete") is True
            and len(first10.get("seconds_at_or_below_50pct", []))
            >= DISRUPTION_MIN_BAD_SECONDS
            and (first10.get("mean_full_prefix_rate") or 1) <= DISRUPTION_MAX_MEAN,
            first10)
    if variant == "amb":
        pre_absence = core.get("pre_absence", {})
        pre_rate = pre_absence.get("full_prefix_hit_rate")
        first10 = rate_windows.get("absence_first10s", {})
        steady = rate_windows.get("absence_steady20s", {})
        add("AMB_recovers_valid_values_within_H",
            first10.get("complete") is True
            and steady.get("complete") is True
            and pre_absence.get("complete") is True
            and pre_rate is not None
            and (steady.get("mean_full_prefix_rate") or 0)
            >= AMB_MIN_PRE_ABSENCE_FRACTION * pre_rate
            and (victim_hit.get("within_h", {}).get("delta") or 0)
            >= VICTIM_MIN_READBACK_HITS
            and totals.get("hit", 0) > 0,
            {"first10s": first10, "steady20s": steady,
             "victim_cache_hit_within_h": victim_hit.get("within_h"),
             "client_readback_hit": totals.get("hit"),
             "note": "valid M negative case: never-removed keys are read back "
                     "from the absent live owner"})
        over_h10 = rate_windows.get("absence_over_h_first10s", {})
        add("over_H_absence_refused_without_reading",
            over_h10.get("complete") is True
            and victim_hit.get("over_h", {}).get("complete") is True
            and victim_hit.get("over_h", {}).get("delta") == 0
            and victim_exist.get("over_h", {}).get("delta") == 0
            and (dest.get("client_readback_refused_unknown", {})
                 .get("over_h", {}).get("delta") or 0) > 0
            and len(over_h10.get("seconds_at_or_below_50pct", []))
            >= DISRUPTION_MIN_BAD_SECONDS
            and (over_h10.get("mean_full_prefix_rate") or 1) <= DISRUPTION_MAX_MEAN,
            {"over_h_first10s": over_h10,
             "victim_cache_hit_over_h": victim_hit.get("over_h"),
             "victim_exist_hit_over_h": victim_exist.get("over_h"),
             "refused_unknown_over_h": dest.get("client_readback_refused_unknown",
                                                 {}).get("over_h"),
             "note": "the H boundary: absent_since >= H refuses the dedicated "
                     "absent route as unknown; M/W cannot exclude a REMOVE "
                     "older than H, so no read happens"})
        add("removed_keys_fail_closed",
            totals.get("refused_removed", 0) > 0,
            {"refused_removed": totals.get("refused_removed"),
             "note": "removed negative case: churn-removed keys are refused "
                     "by M and never served"})
        post = rate_windows.get("post_restore_15_35s", {})
        add("restore_resumes_ordinary_reads_with_no_new_readback_hit",
            post.get("complete") is True
            and pre_absence.get("complete") is True
            and pre_rate is not None
            and (post.get("mean_full_prefix_rate") or 0)
            >= POST_MIN_PRE_ABSENCE_FRACTION * pre_rate
            and (victim_hit.get("post_restore", {}).get("delta") or 0) > 0
            and dest.get("absent_members_max_after_two_polls") == 0
            and dest.get("readback_hit_freeze_after_restore", {}).get("complete") is True
            and dest.get("readback_hit_freeze_after_restore", {}).get("delta") == 0,
            {"post_restore_15_35s": post,
             "victim_cache_hit_post_restore": victim_hit.get("post_restore"),
             "absent_members_max_after_two_polls":
                 dest.get("absent_members_max_after_two_polls"),
             "readback_hit_freeze": dest.get("readback_hit_freeze_after_restore"),
             "held_ring_freeze_start_ns": dest.get("held_ring_freeze_start_ns")})
    metrics_ok = metrics.get("final_capture_complete") is True and metrics.get(
        "every_node_has_metrics") is True
    add("authoritative_final_metrics", metrics_ok,
        {"final_capture_complete": metrics.get("final_capture_complete"),
         "every_node_has_metrics": metrics.get("every_node_has_metrics")})


def cross_variant_checks4(by_name: dict[str, dict]) -> dict:
    required = {"base", "a", "amb"}
    if not required.issubset(by_name):
        return {"evaluated": False,
                "missing": sorted(required.difference(by_name))}
    windows = {name: by_name[name].get("rate_windows", {})
               .get("absence_first10s", {}) for name in required}
    if any(not window.get("complete") or window.get("mean_full_prefix_rate") is None
           for window in windows.values()):
        c35.add_check(by_name["amb"], "cross_amb_recovers_where_base_and_A_cannot",
                      False, {"reason": "incomplete absence_first10s windows",
                              "windows": windows})
        return {"evaluated": False, "reason": "incomplete absence_first10s windows"}
    amb_pre = by_name["amb"].get("core", {}).get("pre_absence", {})
    amb_pre_rate = amb_pre.get("full_prefix_hit_rate")
    if amb_pre.get("complete") is not True or amb_pre_rate is None:
        c35.add_check(by_name["amb"], "cross_amb_recovers_where_base_and_A_cannot",
                      False, {"reason": "incomplete amb pre_absence window",
                              "pre_absence": amb_pre})
        return {"evaluated": False, "reason": "incomplete amb pre_absence window"}
    amb_steady = by_name["amb"].get("rate_windows", {}).get(
        "absence_steady20s", {})
    amb_steady_mean = amb_steady.get("mean_full_prefix_rate")
    if amb_steady.get("complete") is not True or amb_steady_mean is None:
        c35.add_check(by_name["amb"], "cross_amb_recovers_where_base_and_A_cannot",
                      False, {"reason": "incomplete amb steady window",
                              "steady": amb_steady})
        return {"evaluated": False, "reason": "incomplete amb steady window"}
    base_mean = windows["base"]["mean_full_prefix_rate"]
    a_mean = windows["a"]["mean_full_prefix_rate"]
    amb_mean = windows["amb"]["mean_full_prefix_rate"]
    c35.add_check(
        by_name["amb"], "cross_amb_recovers_where_base_and_A_cannot",
        amb_steady_mean >= AMB_MIN_PRE_ABSENCE_FRACTION * amb_pre_rate
        and base_mean <= DISRUPTION_MAX_MEAN
        and a_mean <= DISRUPTION_MAX_MEAN
        and amb_mean - base_mean >= MIN_AMB_CONTRAST
        and amb_mean - a_mean >= MIN_AMB_CONTRAST,
        {"amb_first10s_mean": amb_mean, "base_first10s_mean": base_mean,
         "a_first10s_mean": a_mean,
         "amb_steady20s_mean": amb_steady_mean,
         "amb_pre_absence_mean": amb_pre_rate,
         "amb_minus_base": amb_mean - base_mean,
         "amb_minus_a": amb_mean - a_mean,
         "thresholds": {"amb_min_pre_absence_fraction": AMB_MIN_PRE_ABSENCE_FRACTION,
                        "others_max": DISRUPTION_MAX_MEAN,
                        "min_contrast": MIN_AMB_CONTRAST},
         "meaning": "during the first 10 s of the absence base and A cannot "
                    "read the stale copy (prefix reads collapse) while A+M+B "
                    "reads it back under M control and recovers valid values"})
    return {
        "evaluated": True,
        "amb_first10s_mean": amb_mean,
        "base_first10s_mean": base_mean,
        "a_first10s_mean": a_mean,
        "amb_minus_base": amb_mean - base_mean,
        "amb_minus_a": amb_mean - a_mean,
    }


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--variants", help=",".join(ALL_VARIANTS) + " (CSV debug subset)")
    parser.add_argument("--smoke", action="store_true",
                        help="shorten only timing; mechanics evidence is non-acceptance")
    args = parser.parse_args(argv)
    variants = tuple(args.variants.split(",")) if args.variants else ALL_VARIANTS
    if not variants or any(v not in ALL_VARIANTS for v in variants):
        parser.error(f"--variants must be a CSV of {','.join(ALL_VARIANTS)}")
    if len(set(variants)) != len(variants):
        parser.error("duplicate --variants entries are not allowed")
    args.variants = variants
    build = args.build_dir.resolve()
    args.server = c35.executable(build / "dfkv_server", "dfkv_server")
    args.mds = c35.executable(build / "dfkv_mds", "dfkv_mds")
    args.ctl = c35.executable(build / "dfkvctl", "dfkvctl")
    args.lib = build / "libdfkv.so"
    if not args.lib.is_file():
        parser.error(f"missing libdfkv.so: {args.lib}")
    args.lib = args.lib.resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if any(args.output_root.iterdir()):
        parser.error(
            f"output root must be empty so evidence is never overwritten: {args.output_root}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    etcd = c35.discover_etcd()
    args.lib_handle = c35.load_library(args.lib)
    version = args.lib_handle.dfkv_version()
    workload = Workload4()
    timing = ({"fence_s": 8, "unfence_s": 28, "end_s": 40, "planned_absence_s": 20}
              if args.smoke else
              {"fence_s": 30, "unfence_s": 120, "end_s": 195, "planned_absence_s": 90})
    metadata = {
        "wall_time": c35._iso(),
        "argv": sys.argv if argv is None else [sys.argv[0], *argv],
        "cwd": os.getcwd(), "python": sys.version, "platform": sys.platform,
        "build_dir": str(args.build_dir.resolve()), "output_root": str(args.output_root),
        "requested_variants": args.variants,
        "full_matrix": set(args.variants) == set(ALL_VARIANTS),
        "smoke": args.smoke, "timing": timing,
        "scenario": "CLI-34 case 4: a live owner's MDS membership record is "
                    "deleted (fence-coordinated, with an etcd-level repeated-"
                    "deletion backstop) while its process, data and metrics "
                    "stay untouched; the same process is later re-announced "
                    "by its own heartbeat",
        "binaries": {
            key: {"path": str(path), "sha256": c35._sha256_file(path)}
            for key, path in (("server", args.server), ("mds", args.mds),
                              ("dfkvctl", args.ctl), ("libdfkv", args.lib),
                              ("etcd", etcd))
        },
        "libdfkv_version": version.decode(errors="replace") if version else "",
        "environment_whitelist": {key: os.environ.get(key)
                                  for key in c35.CLIENT_ENV_KEYS},
        "etcd_discovery": "DFKV_ETCD, ETCD_BIN, then PATH; real process and /health required",
        "h_ms": H_MS,
        "assumptions": [
            "Linux loopback TCP; no RDMA or GPU; no production endpoints",
            "enough disk below output-root for retained data/logs and 32 MiB initial values per variant",
            "C is off because case 4 specifies A(+M+B), not cooldown reroute; W is off everywhere (op25-only read-back)",
            "a raw etcd delete of the member record would be re-PUT by the MDS on the next victim heartbeat (UpsertLeased), so the fence's coordinated deletion is what keeps the record absent; the keeper thread re-deletes any reappearance anyway and would record it",
        ],
        "deferred_checks": DEFERRED_CHECKS,
        "workload_identity_sha256": workload.identity_sha256,
    }
    c35._write_json(args.output_root / "run-metadata.json", metadata)
    results = []
    for variant in args.variants:
        print(f"[case4] starting {variant}; evidence={args.output_root / variant}",
              flush=True)
        result = VariantRun4(args, variant, timing, workload, etcd).run()
        results.append(result)
        core = result.get("core", {})
        first = core.get("absence_first10s", {})
        totals = result.get("metrics", {}).get("client_readback_totals", {})
        print(
            f"[case4:{variant}] status={result['status']} "
            f"absence_first10={first.get('full_prefix_hit_rate')} "
            f"readback_hit={totals.get('hit', 0)} "
            f"refused_removed={totals.get('refused_removed', 0)} "
            f"refused_unknown={totals.get('refused_unknown', 0)} "
            f"violations={result.get('correctness', {}).get('unexpected_violation_count', 'unknown')} "
            f"warm_return_today={result.get('correctness', {}).get('warm_return_today_behavior_count', 0)}",
            flush=True,
        )
        if result["status"] == "ERROR":
            print(f"[case4:{variant}] {result['errors']}", file=sys.stderr, flush=True)
            break
    by_name = {result["variant"]: result for result in results}
    comparisons: dict = {"evaluated": False}
    for result in results:
        if result["status"] == "ERROR":
            continue
        selected_checks4(result, args.smoke)
        failures = [check for check in result["acceptance_checks"] if not check["passed"]]
        if failures:
            result["status"] = "FAILED"
        elif args.smoke:
            result["status"] = "SMOKE_MECHANICS_COMPLETE_NON_ACCEPTANCE"
        else:
            result["status"] = "CHECKED"
    if not args.smoke and all(result["status"] != "ERROR" for result in results):
        comparisons = cross_variant_checks4(by_name)
        for name in ("base", "a", "amb"):
            if name in by_name:
                failures = [check for check in by_name[name].get("acceptance_checks", [])
                            if not check["passed"]]
                if failures:
                    by_name[name]["status"] = "FAILED"
    full_matrix = set(args.variants) == set(ALL_VARIANTS)
    all_ran = len(results) == len(args.variants)
    failed = any(result["status"] in ("FAILED", "ERROR") for result in results)
    if args.smoke:
        overall = "SMOKE_FAILED" if failed else "SMOKE_MECHANICS_COMPLETE_NON_ACCEPTANCE"
    elif not full_matrix or not all_ran:
        overall = "FAILED" if failed else "DEBUG_SUBSET_COMPLETE_NON_ACCEPTANCE"
    elif failed:
        overall = "ACCEPTANCE_FAIL"
    else:
        overall = "ACCEPTANCE_PASS"
    summary = {
        "wall_time": c35._iso(), "overall": overall, "smoke": args.smoke,
        "full_matrix": full_matrix, "all_requested_variants_ran": all_ran,
        "acceptance": overall == "ACCEPTANCE_PASS",
        "non_acceptance_reason": (
            "--smoke validates mechanics only; the smoke absence never crosses "
            "H and the M warmup is bypassed" if args.smoke else
            "explicit --variants subset is debugging, not full acceptance"
            if not full_matrix else None),
        "comparison": comparisons, "deferred_checks": DEFERRED_CHECKS,
        "variants": results,
    }
    c35._write_json(args.output_root / "result.json", summary)
    for result in results:
        c35._write_json(args.output_root / result["variant"] / "result.json", result)
    print(f"[case4] comparison={json.dumps(comparisons, sort_keys=True)}", flush=True)
    if overall == "ACCEPTANCE_PASS":
        print("[case4] ACCEPTANCE PASS", flush=True)
    elif overall.startswith("SMOKE"):
        print(f"[case4] {overall}; no acceptance PASS claimed", flush=True)
    else:
        print(f"[case4] {overall}; result={args.output_root / 'result.json'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
