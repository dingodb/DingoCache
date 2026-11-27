#!/usr/bin/env python3
"""CLI-39 churn acceptance against real etcd/MDS/file-engine processes.

The default matrix contains five roots for four independent scenarios:
poll-skew, independent-cooling, added-member-negative/added-member-gated,
and held-decline.  Every native operation has exact start/end intervals in
operations.jsonl; scheduled starts are independently checked in
per_second.jsonl.  All runtime state and evidence stay below a new, empty
output root.  The negative root must produce a removed-value hit: it is a
control, not a failure to hide.  --smoke bypasses only the real M-coverage
warmup and is never an acceptance run.

The added-member scenario needs --old-lib: a registered, pre-fanout libdfkv
that supports ABI v2, the C calls used here, dfkv_version,
dfkv_stats_snapshot and the ordinary dfkv_client_ring_members metric.  The
negative control deletes that closed old client's registry record after its
REMOVE (and records doing so)
to model a missing registry gate.  The gated control retains the same old
registration and requires C/B to close before the final read.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Callable
from urllib.request import Request as UrlRequest, urlopen

import test_dfkv_cli35_case5 as c35

ALL_SCENARIOS = (
    "poll-skew", "independent-cooling", "added-member-negative",
    "added-member-gated", "held-decline",
)
NAMESPACE = b"dfkv/cli39/churn/native-v1"
KEY_COUNT = 1024
VALUE_BYTES = 1024
READBACK_MS = 60000
SETTLE_POLLS = 10
REMOVE_FANOUT = 3
CAP_BYTES = 64 << 20
ENV_KEYS = tuple(dict.fromkeys((*c35.CLIENT_ENV_KEYS,
    "DFKV_PEER_COOLDOWN_MS", "DFKV_PEER_COOLDOWN_MAX_MS",
    "DFKV_PEER_BAD_GRACE_MS")))


class ChurnWorkload(c35.Workload):
    """One deterministic key/value per flat index; exact 1 KiB payloads."""

    def __init__(self) -> None:
        self.keys: list[list[bytes]] = []
        self.values: list[list[bytes]] = []
        self.flat: dict[bytes, int] = {}
        digest = hashlib.sha256()
        for flat in range(KEY_COUNT):
            key = f"cli39/churn/k{flat:04d}".encode()
            seed = hashlib.sha256(b"dfkv-cli39-value-v1\0" + key).digest()
            header = b"DFKV-CLI39-CHURN\0" + key + b"\0"
            value = (header + seed * 64)[:VALUE_BYTES]
            assert len(value) == VALUE_BYTES
            self.keys.append([key])
            self.values.append([value])
            self.flat[key] = flat
            digest.update(key)
            digest.update(value)
        self.identity_sha256 = digest.hexdigest()

    def key(self, flat: int) -> bytes:
        return self.keys[flat][0]

    def value(self, flat: int) -> bytes:
        return self.values[flat][0]

    @staticmethod
    def indices(kind: str, second: int, count: int, universe: int) -> list[int]:
        answer: list[int] = []
        attempt = 0
        while len(answer) < count:
            material = f"dfkv-cli39-churn\0{kind}\0{second}\0{attempt}".encode()
            index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % universe
            if index not in answer:
                answer.append(index)
            attempt += 1
        return answer


class ChurnClient(c35.NativeClient):
    """CLI-35 native client with per-client polling and flat-key GETs."""

    def __init__(self, lib: ctypes.CDLL, client_id: str, role: str,
                 mds: str, group: str, operations: c35.OperationLog,
                 poll_ms: int) -> None:
        self.lib = lib
        self.client_id = client_id
        self.role = role
        self.operations = operations
        self.last_put_start_ns = 0
        self.last_flat_start_ns = 0
        self.opened_ns = 0
        options = c35.make_client_options_v2(
            NAMESPACE, mds_endpoints=mds, mds_group=group,
            mds_poll_ms=poll_ms, register_client=True, client_id=client_id,
            client_info=f"type=cli39-churn;role={role}",
            client_heartbeat_ms=200, batch_concurrency=8,
        )
        self._options = options
        self.handle = lib.dfkv_open_v2(ctypes.byref(options))
        if not self.handle:
            raise RuntimeError(f"dfkv_open_v2 failed for {client_id}")
        self.opened_ns = time.monotonic_ns()
        mode = lib.dfkv_transport_mode(self.handle)
        self.transport = mode.decode(errors="replace") if mode else ""
        if not self.transport.startswith("tcp"):
            lib.dfkv_close(self.handle)
            self.handle = None
            raise RuntimeError(f"{client_id}: TCP required, got {self.transport!r}")

    def put_indices(self, indices: list[int], *, phase: str,
                    second: int | None) -> list[bool]:
        workload = self.operations.workload
        return self.put_keys([workload.key(i) for i in indices],
                             [workload.value(i) for i in indices],
                             phase=phase, second=second, prefix=None)

    def get_indices(self, indices: list[int], *, phase: str,
                    second: int | None) -> dict[str, Any]:
        workload = self.operations.workload
        keys = [workload.key(i) for i in indices]
        expected = [workload.value(i) for i in indices]
        n = len(keys)
        key_array, key_lengths, key_owners = c35.make_key_array(keys)
        storage = ctypes.create_string_buffer(n * VALUE_BYTES)
        base = ctypes.addressof(storage)
        pointers = (ctypes.c_void_p * n)(*[
            base + i * VALUE_BYTES for i in range(n)])
        capacities = (ctypes.c_uint64 * n)(*([VALUE_BYTES] * n))
        hits = (ctypes.c_int * n)()
        lengths = (ctypes.c_uint64 * n)()
        statuses = (ctypes.c_uint8 * n)()
        attempt = self.operations.attempt()
        started = time.monotonic_ns()
        self.operations.starts(attempt, "GET", keys, client=self.client_id,
                               phase=phase, second=second, prefix=None,
                               start_ns=started)
        try:
            call_rc = self.lib.dfkv_batch_get_auto_st(
                self.handle, key_array, key_lengths, pointers, capacities,
                n, hits, lengths, statuses)
        except BaseException as exc:
            self._exception_ends(attempt, "GET", keys, exc)
            raise
        ended = time.monotonic_ns()
        rows: list[dict[str, Any]] = []
        hit_indices: list[int] = []
        mismatches: list[int] = []
        for i, flat in enumerate(indices):
            status = int(statuses[i])
            hit = call_rc == 0 and hits[i] == 1
            length = int(lengths[i]) if hit else 0
            correct: bool | None = None
            if hit:
                actual = (ctypes.string_at(base + i * VALUE_BYTES, length)
                          if length <= VALUE_BYTES else b"")
                correct = length == VALUE_BYTES and actual == expected[i]
                hit_indices.append(flat)
                if not correct:
                    mismatches.append(flat)
            row: dict[str, Any] = {
                "call_rc": call_rc, "status": status,
                "status_name": c35.STATUS_NAMES.get(status, str(status)),
                "hit": hit, "value_length": length, "payload_correct": correct,
            }
            if call_rc != 0:
                row["error"] = f"dfkv_batch_get_auto_st rc={call_rc}"
            rows.append(row)
        self.operations.ends(attempt, "GET", keys, rows, end_ns=ended)
        del key_owners, storage
        if call_rc != 0:
            raise RuntimeError(f"dfkv_batch_get_auto_st rc={call_rc}")
        return {
            "attempt": attempt, "start_ns": started, "end_ns": ended,
            "indices": indices, "hits": len(hit_indices),
            "hit_indices": hit_indices, "miss_indices": [
                flat for flat in indices if flat not in hit_indices],
            "full_hit": len(hit_indices) == n,
            "payload_mismatches": len(mismatches),
            "payload_mismatch_indices": mismatches,
        }


class ChurnRun:
    def __init__(self, args: argparse.Namespace, scenario: str,
                 workload: ChurnWorkload, etcd: Path) -> None:
        self.args = args
        self.scenario = scenario
        self.workload = workload
        self.etcd = etcd
        self.root = args.output_root / scenario
        self.root.mkdir()
        for name in ("logs", "metrics", "client-access", "server-access"):
            (self.root / name).mkdir()
        self.sinks = {
            name: c35.Jsonl(self.root / f"{name}.jsonl")
            for name in ("operations", "per_second", "timeline", "metrics",
                         "membership", "processes", "churn")
        }
        self.processes = c35.Processes(self.root, self.sinks["processes"])
        self.operations = c35.OperationLog(self.sinks["operations"], workload)
        self.clients: dict[str, ChurnClient] = {}
        self.nodes: list[c35.Node] = []
        self.group = f"cli39-{scenario}-{os.getpid()}"
        self.mds_endpoint = ""
        self.etcd_endpoint = ""
        self.loader: ChurnClient | None = None
        self.reader: ChurnClient | None = None
        self.reader_b: ChurnClient | None = None
        self.remover: ChurnClient | None = None
        self.writer: ChurnClient | None = None
        self.origin_ns = 0
        self._second_cursor = 0
        self._client_serial = 0
        self._saved_env: dict[str, str | None] = {}
        self._last_views: dict[str, int] = {}
        self.scheduled_rows: list[dict[str, Any]] = []
        self.specific: dict[str, Any] = {}
        self.result: dict[str, Any] = {
            "scenario": scenario, "smoke": args.smoke,
            "acceptance_eligible": not args.smoke, "status": "RUNNING",
            "errors": [], "acceptance_checks": [],
            "workload": {
                "keys": KEY_COUNT, "value_bytes": VALUE_BYTES,
                "identity_sha256": workload.identity_sha256,
                "key_schedule": "explicit flat indices; SHA-256 only for untouched controls",
            },
            "expected_negative_control": scenario == "added-member-negative",
        }
        self._cleaned_up = False

    # ---- process, registry and metrics evidence ----

    def timeline(self, event: str, **fields: Any) -> None:
        self.sinks["timeline"].write({
            "event": event, "monotonic_ns": time.monotonic_ns(),
            "wall_time": c35._iso(), **fields,
        })

    def client_env(self) -> dict[str, str]:
        independent = self.scenario == "independent-cooling"
        return {
            "DFKV_RING_ABSENT_GRACE_MS": "0" if independent else "60000",
            "DFKV_RING_READBACK_MS": str(READBACK_MS),
            "DFKV_RING_READBACK_REWRITE": "0",
            "DFKV_RING_COOLDOWN_REROUTE": "1",
            "DFKV_RING_REMOVE_FANOUT": str(REMOVE_FANOUT),
            "DFKV_RING_REROUTE_KEYS": "65536",
            "DFKV_RING_REROUTE_CLOSE_MS": "2000",
            "DFKV_CLIENT_NODE_DEDUP": "0",
            "DFKV_CLIENT_NODE_DEDUP_GPU": "0",
            "DFKV_RDMA": "0",
            "DFKV_OPEN_RING_WAIT_MS": "30000",
            "DFKV_GET_MISS_RETRIES": "1",
            "DFKV_PEER_COOLDOWN_MS": "60000" if independent else "2000",
            "DFKV_PEER_COOLDOWN_MAX_MS": "60000" if independent else "30000",
            "DFKV_PEER_BAD_GRACE_MS": "1",
            "DFKV_LOG": "WARN",
            "DFKV_ACCESS_DIR": str(self.root / "client-access"),
        }

    def server_env(self, h_ms: int) -> dict[str, str]:
        return {
            "DFKV_REMOVE_MEMORY_MS": str(h_ms),
            "DFKV_REMOVE_MEMORY_ENTRIES": "1048576",
            "DFKV_WRITE_MEMORY_ENTRIES": "0",
            "DFKV_MDS_HEARTBEAT_MS": "200",
            "DFKV_DRAIN": "1",
            "DFKV_STOP_DRAIN_MS": "0",
            "DFKV_LOG": "INFO",
            "DFKV_RDMA": "0",
            "DFKV_ACCESS_DIR": str(self.root / "server-access"),
        }

    def apply_client_env(self) -> None:
        wanted = self.client_env()
        for key in ENV_KEYS:
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

    def start_etcd(self) -> None:
        for attempt in range(1, 6):
            client_port, peer_port = c35._free_port(), c35._free_port()
            data = self.root / "runtime" / f"etcd-attempt-{attempt}"
            data.mkdir(parents=True)
            client_url = f"http://127.0.0.1:{client_port}"
            peer_url = f"http://127.0.0.1:{peer_port}"
            name = f"cli39-{self.scenario}"
            child = self.processes.launch(f"etcd-{attempt}", [
                str(self.etcd), "--data-dir", str(data), "--name", name,
                "--listen-client-urls", client_url,
                "--advertise-client-urls", client_url,
                "--listen-peer-urls", peer_url,
                "--initial-advertise-peer-urls", peer_url,
                "--initial-cluster", f"{name}={peer_url}",
            ], dict(os.environ))
            self.etcd_endpoint = f"127.0.0.1:{client_port}"
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if child.popen.poll() is not None:
                    break
                try:
                    with urlopen(client_url + "/health", timeout=0.5) as response:
                        health = json.loads(response.read())
                    if health.get("health") == "true":
                        self.timeline("etcd_healthy", endpoint=self.etcd_endpoint,
                                      data_dir=str(data))
                        return
                except Exception:
                    pass
                time.sleep(0.1)
            if child.popen.poll() is None:
                raise RuntimeError(f"etcd not healthy; see {child.log_path}")
        raise RuntimeError("etcd failed on five independent attempts")

    def start_mds(self) -> None:
        child = self.processes.launch("mds", [
            str(self.args.mds), "--listen", "0", "--etcd", self.etcd_endpoint,
            "--etcd-probe-ms", "10000", "--metrics-port", "0",
            "--metrics-bind", "127.0.0.1",
        ], {**os.environ, "DFKV_LOG": "INFO"})
        match = self.processes.wait_log(child, r"^dfkv_mds listening on (\d+),")
        self.mds_endpoint = f"127.0.0.1:{int(match.group(1))}"
        self.timeline("mds_ready", endpoint=self.mds_endpoint)

    def start_node(self, number: int, generation: int = 0, phase: str = "initial",
                   h_ms: int | None = None, port: int | None = None) -> c35.Node:
        assert h_ms is not None
        node_id = f"n{number}"
        for attempt in range(1, 6):
            selected_port = port if port is not None else c35._free_port()
            data = self.root / "runtime" / f"{node_id}-{phase}-attempt-{attempt}"
            data.mkdir(parents=True)
            argv = [
                str(self.args.server), "--dir", str(data), "--port", str(selected_port),
                "--cap", str(CAP_BYTES), "--store-engine", "file",
                "--mds", self.mds_endpoint, "--group", self.group,
                "--id", node_id, "--advertise", f"127.0.0.1:{selected_port}",
                "--metrics-port", "0", "--metrics-bind", "127.0.0.1",
                "--mds-registration-timeout-ms", "10000",
            ]
            child = self.processes.launch(
                f"{node_id}-g{generation}-a{attempt}", argv,
                {**os.environ, **self.server_env(h_ms)})
            try:
                port_match = self.processes.wait_log(child, r"^PORT (\d+)\s*$")
                if int(port_match.group(1)) != selected_port:
                    raise RuntimeError(f"{node_id}: unexpected listening port")
                metrics_match = self.processes.wait_log(
                    child, r"dfkv_server /metrics on port (\d+)")
                return c35.Node(node_id, generation, "current", self.args.server,
                                child, data, selected_port, int(metrics_match.group(1)))
            except RuntimeError:
                if child.popen.poll() is None:
                    raise
                text = child.log_path.read_text(errors="replace")
                if f"failed to start on port {selected_port}" not in text:
                    raise
            if port is not None:
                raise RuntimeError(f"{node_id}: required reuse port {port} is unavailable")
        raise RuntimeError(f"{node_id}: no usable port in five attempts")

    def ring(self) -> dict:
        completed = subprocess.run([
            str(self.args.ctl), "ring", "--json", "--allow-empty",
            "--group", self.group, "--mds", self.mds_endpoint,
        ], text=True, capture_output=True, timeout=8, check=True)
        return json.loads(completed.stdout)

    def registry(self) -> str:
        completed = subprocess.run([
            str(self.args.ctl), "clients", "--group", self.group,
            "--mds", self.mds_endpoint,
        ], text=True, capture_output=True, timeout=8, check=True)
        return completed.stdout

    def locate(self, flat: int) -> str:
        key = self.workload.key(flat).decode()
        completed = subprocess.run([
            str(self.args.ctl), "locate", key, "--group", self.group,
            "--mds", self.mds_endpoint, "--ring-wait-ms", "10000",
        ], text=True, capture_output=True, timeout=12, check=True)
        return completed.stdout.strip()

    def wait_mds_members(self, count: int, timeout: float = 40.0) -> dict:
        def predicate() -> dict | None:
            try:
                value = self.ring()
                if len(value["members"]) == count and all(
                        member["eligible"] for member in value["members"]):
                    return value
            except Exception:
                pass
            return None
        result = c35._until(f"MDS ring has {count} eligible members", predicate, timeout)
        self.sinks["membership"].write({
            "event": "mds_ring", "observed_ns": time.monotonic_ns(),
            "expected_count": count, "members": result["members"],
        })
        return result

    def open_client(self, role: str, poll_ms: int, *, old: bool = False) -> ChurnClient:
        self._client_serial += 1
        client_id = f"{self.scenario}-{role}-{self._client_serial}-{os.getpid()}"
        client = ChurnClient(self.args.old_lib_handle if old else self.args.lib_handle,
                             client_id, role, self.mds_endpoint, self.group,
                             self.operations, poll_ms)
        self.clients[client_id] = client
        self.timeline("client_open", client=client_id, role=role,
                      poll_ms=poll_ms, old_library=old, transport=client.transport)
        return client

    def close_client(self, client: ChurnClient) -> None:
        self.clients.pop(client.client_id, None)
        client.close()
        self.timeline("client_close", client=client.client_id)

    def raw_client(self, client: ChurnClient) -> str:
        return client.stats()

    def client_core(self, client: ChurnClient) -> dict[str, Any]:
        text = self.raw_client(client)
        answer: dict[str, Any] = dict(c35._client_core(text))
        answer.update({
            "unmarked_clients": c35._metric(
                text, "dfkv_client_registry_unmarked_clients"),
            "gate_unmarked_failures": c35._metric(
                text, "dfkv_client_registry_gate_failures_total",
                reason="unmarked_client"),
            "unhealthy_skips": c35._metric_sum(
                text, "dfkv_client_unhealthy_skips_total"),
        })
        for op in ("get", "put", "exist", "none"):
            answer[f"reroutes_{op}"] = c35._metric(
                text, "dfkv_client_cooldown_reroutes_total", op=op)
        answer["readback_total"] = sum(answer.get(f"readback_{label}") or 0
                                       for label in c35.READBACK_RESULTS)
        return answer

    def wait_client(self, client: ChurnClient, what: str,
                    predicate: Callable[[dict[str, Any]], bool],
                    timeout: float = 40.0) -> dict[str, Any]:
        def sample() -> dict[str, Any] | None:
            try:
                core = self.client_core(client)
                ring = core.get("ring_members")
                if isinstance(ring, (int, float)) and self._last_views.get(client.client_id) != ring:
                    self._last_views[client.client_id] = int(ring)
                    self.timeline("client_ring_view", client=client.client_id,
                                  ring_members=int(ring))
                return core if predicate(core) else None
            except Exception:
                return None
        return c35._until(f"{client.client_id}: {what}", sample, timeout)

    def wait_ring(self, client: ChurnClient, count: int,
                  timeout: float = 40.0) -> dict[str, Any]:
        return self.wait_client(client, f"ring={count}",
                                lambda core: core.get("ring_members") == count, timeout)

    def wait_gate(self, client: ChurnClient, open_: bool,
                  timeout: float = 40.0) -> dict[str, Any]:
        expected = 1 if open_ else 0
        return self.wait_client(client, f"registry_gate_open={expected}",
                                lambda core: core.get("registry_gate_open") == expected,
                                timeout)

    def running_nodes(self) -> list[c35.Node]:
        return [node for node in self.nodes if node.child.popen.poll() is None]

    def capture_metrics(self, phase: str) -> dict[str, Any]:
        row: dict[str, Any] = {
            "phase": phase, "monotonic_ns": time.monotonic_ns(),
            "wall_time": c35._iso(), "clients": {}, "servers": {},
        }
        for client in self.clients.values():
            try:
                text = self.raw_client(client)
                (self.root / "metrics" / f"client-{client.client_id}-{phase}.prom").write_text(text)
                row["clients"][client.client_id] = self.client_core(client)
            except Exception as exc:
                row["clients"][client.client_id] = {"error": str(exc)}
        for node in self.nodes:
            try:
                text = c35.MetricsCollector.scrape(node.metrics_port)
                (self.root / "metrics" / f"server-{node.key}-{phase}.prom").write_text(text)
                row["servers"][node.key] = c35._server_core(text)
            except Exception as exc:
                row["servers"][node.key] = {"error": str(exc)}
        self.sinks["metrics"].write(row)
        return row

    def wait_memory_coverage(self, h_ms: int) -> None:
        required_s = (h_ms + 999) // 1000
        common = {"required_seconds": required_s, "h_ms": h_ms,
                  "smoke": self.args.smoke}
        if self.args.smoke:
            self.result["memory_coverage"] = {
                **common, "waited": False, "warmup_bypassed": True,
                "non_acceptance": True,
            }
            return
        deadline = time.monotonic() + required_s + 60
        latest: list[dict[str, Any]] = []
        initial_nodes = list(self.running_nodes())
        expected = {node.key for node in initial_nodes}
        if not expected:
            raise RuntimeError("H coverage requires a nonempty initial ring")
        while time.monotonic() < deadline:
            latest = []
            for node in initial_nodes:
                row: dict[str, Any] = {"node": node.key}
                try:
                    text = c35.MetricsCollector.scrape(node.metrics_port)
                    row["covered_seconds"] = c35._singleton_metric(
                        text, "dfkv_remove_memory_covered_seconds")
                except Exception as exc:
                    row["error"] = str(exc)
                latest.append(row)
            self.sinks["churn"].write({
                "event": "memory_coverage", "monotonic_ns": time.monotonic_ns(),
                "required_seconds": required_s, "servers": latest,
            })
            if ({row["node"] for row in latest} == expected and all(
                    (row.get("covered_seconds") or 0) >= required_s
                    for row in latest)):
                self.result["memory_coverage"] = {
                    **common, "waited": True, "warmup_bypassed": False,
                    "ready_ns": time.monotonic_ns(), "servers": latest,
                }
                return
            time.sleep(2.0)
        raise RuntimeError(f"M coverage did not reach {required_s}s; last={latest}")

    # ---- exact scheduled native starts ----

    def start_clock(self) -> None:
        self.origin_ns = time.monotonic_ns()
        self._second_cursor = 0
        self.timeline("workload_origin", origin_ns=self.origin_ns)

    def allocate_second(self) -> int:
        elapsed = ((time.monotonic_ns() - self.origin_ns) // 1_000_000_000
                   if self.origin_ns else 0)
        second = max(self._second_cursor, int(elapsed) + 1)
        self._second_cursor = second + 1
        return second

    def wait_second(self, second: int) -> None:
        if not self.origin_ns:
            raise RuntimeError("workload clock was not started")
        c35._wait_until(self.origin_ns, second)
        self._second_cursor = max(self._second_cursor, second + 1)

    def record_start(self, second: int, op: str, client: ChurnClient,
                     start_ns: int, detail: dict[str, Any]) -> None:
        scheduled_start = self.origin_ns + second * 1_000_000_000
        scheduled_end = scheduled_start + 1_000_000_000
        valid = scheduled_start <= start_ns < scheduled_end
        row = {
            "second": second, "op": op, "client": client.client_id,
            "scheduled_start_ns": scheduled_start, "scheduled_end_ns": scheduled_end,
            "actual_start_ns": start_ns, "actual_start_valid": valid,
            "detail": detail,
        }
        self.scheduled_rows.append(row)
        self.sinks["per_second"].write(row)
        if not valid:
            raise RuntimeError(f"actual {op} start outside second {second}: {row}")

    def read(self, client: ChurnClient, indices: list[int], second: int | None = None,
             phase: str = "workload") -> dict[str, Any]:
        second = self.allocate_second() if second is None else second
        self.wait_second(second)
        result = client.get_indices(indices, phase=phase, second=second)
        result["second"] = second
        self.record_start(second, "GET", client, result["start_ns"], {
            "indices": indices, "hits": result["hits"],
            "hit_indices": result["hit_indices"],
            "payload_mismatches": result["payload_mismatches"],
        })
        return result

    def put(self, client: ChurnClient, indices: list[int], second: int | None = None,
            phase: str = "workload") -> dict[str, Any]:
        second = self.allocate_second() if second is None else second
        self.wait_second(second)
        oks = client.put_indices(indices, phase=phase, second=second)
        started = client.last_put_start_ns
        result = {"second": second, "indices": indices, "oks": oks,
                  "all_ok": all(oks), "start_ns": started}
        self.record_start(second, "PUT", client, started, {
            "indices": indices, "oks": oks,
        })
        return result

    def remove(self, client: ChurnClient, indices: list[int],
               second: int | None = None) -> dict[str, Any]:
        second = self.allocate_second() if second is None else second
        self.wait_second(second)
        oks, ended = client.remove(indices, second=second)
        started = client.last_flat_start_ns
        result = {"second": second, "indices": indices, "oks": oks,
                  "confirmed_count": sum(oks), "start_ns": started,
                  "end_ns": ended}
        self.record_start(second, "REMOVE", client, started, {
            "indices": indices, "per_key_ok": oks,
            "confirmed_count": sum(oks),
        })
        self.sinks["churn"].write({"event": "remove_completed", **result})
        return result

    def exist(self, client: ChurnClient, indices: list[int],
              second: int | None = None) -> dict[str, Any]:
        second = self.allocate_second() if second is None else second
        self.wait_second(second)
        present = client.exist(indices, second=second)
        started = client.last_flat_start_ns
        self.record_start(second, "GET_EXIST", client, started, {
            "indices": indices, "present": present,
        })
        return {"second": second, "indices": indices, "present": present,
                "start_ns": started}

    def align_after_poll(self, client: ChurnClient, poll_ms: int) -> None:
        # Leave nearly a whole slow-reader interval for add/PUT/REMOVE.  The
        # subsequent reader gauge must still prove the old view immediately
        # before the old REMOVE; timing alone is never accepted as proof.
        period = poll_ms * 1_000_000
        now = time.monotonic_ns()
        next_tick = client.opened_ns + ((now - client.opened_ns) // period + 1) * period
        target = next_tick + 300_000_000
        if target > now:
            time.sleep((target - now) / 1e9)
        self.timeline("aligned_after_reader_poll", client=client.client_id,
                      poll_ms=poll_ms, target_ns=target)

    # ---- scenario setup and checks ----

    def setup(self, node_count: int, reader_poll_ms: int, *,
              writer: bool = False, two_readers: bool = False,
              start_clock: bool = True) -> int:
        remover_poll_ms = 3000 if self.scenario == "held-decline" else 1000
        poll_max_ms = max(reader_poll_ms, remover_poll_ms, 1000)
        h_ms = READBACK_MS + SETTLE_POLLS * poll_max_ms
        self.result["h_evidence"] = {
            "reader_poll_ms": reader_poll_ms, "remover_poll_ms": remover_poll_ms,
            "poll_max_ms": poll_max_ms, "readback_ms": READBACK_MS,
            "settle_ms": SETTLE_POLLS * poll_max_ms, "h_ms": h_ms,
            "formula": "H = DFKV_RING_READBACK_MS + 10 * max(all registered poll_ms)",
        }
        self.result["server_env"] = self.server_env(h_ms)
        self.start_etcd()
        self.start_mds()
        for number in range(node_count):
            self.nodes.append(self.start_node(number, h_ms=h_ms))
        self.wait_mds_members(node_count)
        self.apply_client_env()
        self.loader = self.open_client("loader", reader_poll_ms)
        for first in range(0, KEY_COUNT, 64):
            indices = list(range(first, min(first + 64, KEY_COUNT)))
            oks = self.loader.put_indices(indices, phase="initial", second=None)
            if not all(oks):
                raise RuntimeError(f"initial load failed at {first}: {oks}")
        self.reader = self.open_client("reader-a", reader_poll_ms)
        if two_readers:
            self.reader_b = self.open_client("reader-b", reader_poll_ms)
        if writer:
            self.writer = self.open_client("writer", 1000)
        if self.scenario in ("poll-skew", "independent-cooling"):
            self.remover = self.open_client("remover", 1000)
        for client in self.clients.values():
            self.wait_gate(client, True)
        registry = self.registry()
        polls = [int(value) for value in re.findall(r"poll_ms=(\d+)", registry)]
        self.result["h_evidence"].update({
            "registry_at_start": registry,
            "registry_poll_values_ms": polls,
            "registry_poll_max_ms": max(polls) if polls else None,
            "matches_configured_poll_max": bool(polls) and max(polls) == poll_max_ms,
        })
        self.capture_metrics("baseline")
        self.wait_memory_coverage(h_ms)
        self.result["initial_load"] = {"keys": KEY_COUNT, "all_puts_ok": True}
        if start_clock:
            self.start_clock()
        return h_ms

    def require_hit(self, result: dict[str, Any], what: str,
                    count: int | None = None) -> None:
        expected = len(result["indices"]) if count is None else count
        if result["hits"] != expected or result["payload_mismatches"]:
            raise RuntimeError(f"{what}: required exact hits={expected}, got {result}")

    def owners(self, indices: list[int]) -> dict[int, str]:
        return {flat: self.locate(flat) for flat in indices}

    def execute(self) -> None:
        if self.scenario == "poll-skew":
            self.run_poll_skew()
        elif self.scenario == "independent-cooling":
            self.run_independent_cooling()
        elif self.scenario in ("added-member-negative", "added-member-gated"):
            self.run_added_member()
        elif self.scenario == "held-decline":
            self.run_held_decline()
        else:
            raise RuntimeError(f"unknown scenario {self.scenario}")
        self.capture_metrics("final")

    def run_poll_skew(self) -> None:
        h_ms = self.setup(8, 10000, start_clock=False)
        assert self.reader is not None and self.remover is not None
        assert self.remover is not self.reader
        # Keep every scheduled key off the killed initial owner so a scheduled
        # start is not delayed by an intentional dead-endpoint timeout.
        candidates = list(range(160))
        old_owners = self.owners(candidates)
        killed_address = f"127.0.0.1:{self.nodes[3].port}"
        usable = [flat for flat in candidates if old_owners[flat] != killed_address]
        if len(usable) < 80:
            raise RuntimeError("not enough deterministic non-victim keys")
        early = usable[:16]
        scheduled = usable[16:]
        self.start_clock()
        early_evidence: dict[str, Any] = {}
        converged_second: int | None = None
        execute_seconds: list[int] = []
        for second in range(120):
            if second == 2:
                self.wait_second(second)
                killed = self.nodes[3]
                self.processes.scenario_kill(killed.child)
                self.timeline("poll_skew_victim_killed", node=killed.node_id,
                              pid=killed.child.popen.pid, port=killed.port)
            indices = [scheduled[(2 * second) % len(scheduled)],
                       scheduled[(2 * second + 1) % len(scheduled)]]
            self.read(self.reader, indices, second)
            self.remove(self.remover, indices, second)
            after = self.read(self.reader, indices, second)
            execute_seconds.append(second)
            if not early_evidence:
                remover_core = self.client_core(self.remover)
                reader_core = self.client_core(self.reader)
                if (remover_core.get("ring_members") == 7
                        and reader_core.get("ring_members") == 8):
                    before = {
                        "reader": reader_core, "remover": remover_core,
                        "observed_ns": time.monotonic_ns(),
                    }
                    removed = self.remove(self.remover, early, second)
                    post = self.read(self.reader, early, second)
                    early_evidence = {
                        "before_remove": before,
                        "remove": removed, "post_remove_read": post,
                        "reader_ring_after": self.client_core(self.reader).get("ring_members"),
                        "h_ms": h_ms,
                    }
                    self.timeline("early_removes_during_view_skew", **early_evidence)
            if early_evidence and converged_second is None:
                if self.client_core(self.reader).get("ring_members") == 7:
                    converged_second = second
                    self.timeline("slow_reader_converged", second=second)
            if converged_second is not None and second >= converged_second + 2:
                break
        if not early_evidence:
            raise RuntimeError("never observed remover=7 while reader=8")
        self.specific = {
            "early": early_evidence, "converged_second": converged_second,
            "scheduled_seconds": execute_seconds,
            "reader_poll_ms": 10000, "remover_poll_ms": 1000,
        }
        self.check("poll_skew_early_removes_proved",
                   early_evidence["remove"]["confirmed_count"] > 0
                   and early_evidence["before_remove"]["reader"]["ring_members"] == 8
                   and early_evidence["before_remove"]["remover"]["ring_members"] == 7,
                   early_evidence)
        self.check("poll_skew_no_direct_stale_hits",
                   early_evidence["post_remove_read"]["hits"] == 0,
                   early_evidence["post_remove_read"])
        self.check("slow_reader_eventually_adopted", converged_second is not None,
                   {"converged_second": converged_second})
        self.check("h_uses_actual_registry_poll_max",
                   self.result["h_evidence"]["matches_configured_poll_max"]
                   and h_ms == READBACK_MS + 10 * 10000,
                   self.result["h_evidence"])

    def run_independent_cooling(self) -> None:
        self.setup(8, 1000, two_readers=True)
        assert self.reader is not None and self.reader_b is not None
        assert self.remover is not None
        candidates = list(range(192))
        owners = self.owners(candidates)
        target = self.nodes[2]
        target_address = f"127.0.0.1:{target.port}"
        target_keys = [flat for flat in candidates if owners[flat] == target_address][:8]
        healthy_keys = [flat for flat in candidates if owners[flat] != target_address][:8]
        if len(target_keys) < 4 or len(healthy_keys) < 4:
            raise RuntimeError("not enough owner-targeted deterministic keys")
        target_keys = target_keys[:4]
        healthy_keys = healthy_keys[:4]
        baseline_a = self.read(self.reader, target_keys)
        baseline_b = self.read(self.reader_b, healthy_keys)
        self.require_hit(baseline_a, "reader-a baseline")
        self.require_hit(baseline_b, "reader-b unaffected baseline")
        kill_second = self.allocate_second()
        self.wait_second(kill_second)
        self.processes.scenario_kill(target.child)
        self.timeline("cooling_owner_killed", node=target.node_id,
                      address=target_address, pid=target.child.popen.pid)
        # Only A contacts the killed owner. B keeps using a healthy, exact-hit
        # route and therefore never charges its independent PeerHealth.
        self.read(self.reader, target_keys)
        healthy_during_outage = self.read(self.reader_b, healthy_keys)
        self.require_hit(healthy_during_outage, "reader-b healthy route during A outage")
        self.read(self.reader, target_keys)
        cooled_a = self.wait_client(
            self.reader, "owner cooling used by A",
            lambda core: (core.get("reroutes_get") or 0) > 0, 15)
        unaffected_b = self.client_core(self.reader_b)
        restart_second = self.allocate_second()
        self.wait_second(restart_second)
        replacement = self.start_node(2, 1, "return-same-address", h_ms=self.result["h_evidence"]["h_ms"],
                                      port=target.port)
        self.nodes[2] = replacement
        self.wait_mds_members(8)
        self.wait_ring(self.reader, 8)
        self.wait_ring(self.reader_b, 8)
        self.timeline("cooling_owner_returned_same_address", node="n2",
                      address=target_address, old_pid=target.child.popen.pid,
                      new_pid=replacement.child.popen.pid)
        # B replenishes the new owner's empty directory. A still has the old
        # endpoint cooling and puts the same exact value on its substitute.
        put_b = self.put(self.reader_b, target_keys)
        put_a = self.put(self.reader, target_keys)
        if not put_b["all_ok"] or not put_a["all_ok"]:
            raise RuntimeError(f"independent owner/substitute PUT failed: {put_b} {put_a}")
        hit_b = self.read(self.reader_b, target_keys)
        hit_a = self.read(self.reader, target_keys)
        self.require_hit(hit_b, "reader-b direct owner after return")
        self.require_hit(hit_a, "reader-a substitute after return")
        removed = self.remove(self.remover, target_keys)
        if removed["confirmed_count"] != len(target_keys):
            raise RuntimeError(f"fanout REMOVE was not fully confirmed: {removed}")
        miss_b = self.read(self.reader_b, target_keys)
        miss_a = self.read(self.reader, target_keys)
        final_a = self.client_core(self.reader)
        final_b = self.client_core(self.reader_b)
        self.specific = {
            "target_keys": target_keys, "healthy_control_keys": healthy_keys,
            "healthy_during_outage": healthy_during_outage,
            "cooled_a": cooled_a, "unaffected_b_before_return": unaffected_b,
            "final_a": final_a, "final_b": final_b, "remove": removed,
            "post_remove_a": miss_a, "post_remove_b": miss_b,
            "same_owner_address": target_address,
            "old_and_replacement_pids": [target.child.popen.pid,
                                          replacement.child.popen.pid],
        }
        self.check("owner_cooling_only_in_reader_a",
                   (final_a.get("reroutes_get") or 0) > 0
                   and (final_a.get("reroutes_put") or 0) > 0
                   and (final_a.get("unhealthy_skips") or 0) > 0,
                   final_a)
        self.check("reader_b_route_and_peer_health_unaffected",
                   all((final_b.get(f"reroutes_{op}") or 0) == 0
                       for op in ("get", "put", "exist", "none"))
                   and (final_b.get("unhealthy_skips") or 0) == 0
                   and healthy_during_outage["full_hit"] and hit_b["full_hit"],
                   {"before": unaffected_b, "after": final_b,
                    "healthy_during_outage": healthy_during_outage,
                    "exact_owner_hits": hit_b})
        self.check("both_readers_miss_after_confirmed_remove",
                   miss_a["hits"] == 0 and miss_b["hits"] == 0,
                   {"a": miss_a, "b": miss_b})

    def etcd_post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = UrlRequest(f"http://{self.etcd_endpoint}{path}",
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=2) as response:
            return json.loads(response.read())

    def delete_client_record(self, client_id: str) -> dict[str, Any]:
        key = f"/dfkv/v1/groups/{self.group}/clients/{client_id}"
        before = self.etcd_post("/v3/kv/range", {
            "key": base64.b64encode(key.encode()).decode()})
        response = self.etcd_post("/v3/kv/deleterange", {
            "key": base64.b64encode(key.encode()).decode()})
        evidence = {"client": client_id, "key": key,
                    "present_before_delete": bool(before.get("kvs")),
                    "delete_response": response,
                    "monotonic_ns": time.monotonic_ns()}
        self.sinks["membership"].write({"event": "negative_gate_registry_bypass",
                                        **evidence})
        return evidence

    def gate_cooling_probe(self) -> dict[str, Any]:
        """Distinguish C from B after the added-member read-back control.

        The ring/address does not change: the same owner dies briefly and
        returns on its original port. Only rerouting (C) can make the second
        down-owner GET leave that owner. No REMOVE participates in this probe.
        """
        assert self.reader is not None
        target = self.nodes[2]
        address = f"127.0.0.1:{target.port}"
        candidates = list(range(192, 320))
        keys = [flat for flat in candidates if self.locate(flat) == address][:4]
        if len(keys) < 2:
            raise RuntimeError("no deterministic added-member cooling-probe owner keys")
        before = self.client_core(self.reader)
        self.processes.scenario_kill(target.child)
        self.timeline("gate_cooling_probe_owner_killed", node=target.node_id,
                      address=address, pid=target.child.popen.pid)
        self.read(self.reader, keys)
        self.read(self.reader, keys)
        after = self.client_core(self.reader)
        replacement = self.start_node(
            2, 1, "gate-probe-return-same-address",
            h_ms=self.result["h_evidence"]["h_ms"], port=target.port)
        self.nodes[2] = replacement
        self.wait_mds_members(9)
        self.timeline("gate_cooling_probe_owner_returned", node=target.node_id,
                      address=address, new_pid=replacement.child.popen.pid)
        return {
            "keys": keys, "owner_address": address, "before": before,
            "after_down_owner_reads": after,
            "reroutes_get_delta": ((after.get("reroutes_get") or 0)
                                   - (before.get("reroutes_get") or 0)),
            "old_pid": target.child.popen.pid,
            "replacement_pid": replacement.child.popen.pid,
        }

    def run_added_member(self) -> None:
        self.setup(8, 10000, writer=True)
        assert self.reader is not None and self.writer is not None
        negative = self.scenario == "added-member-negative"
        candidates = list(range(160))
        old_owners = self.owners(candidates)
        self.align_after_poll(self.reader, 10000)
        added = self.start_node(8, 0, "added", h_ms=self.result["h_evidence"]["h_ms"])
        self.nodes.append(added)
        self.wait_mds_members(9)
        new_address = f"127.0.0.1:{added.port}"
        # The old client's first view is the nine-member view: no inferred
        # polling race. It is ahead of the still-eight-member reader.
        old = self.open_client("old-remover", 1000, old=True)
        self.remover = old
        self.wait_ring(old, 9, 15)
        new_keys: list[int] = []
        new_locations: dict[int, str] = {}
        for flat in candidates:
            location = self.locate(flat)
            if location == new_address and old_owners[flat] != new_address:
                new_keys.append(flat)
                new_locations[flat] = location
                if len(new_keys) == 8:
                    break
        if len(new_keys) < 4:
            raise RuntimeError("no deterministic keys moved to the added member")
        new_keys = new_keys[:4]
        registered_text = c35._until(
            "old client visible in registry",
            lambda: (text if old.client_id in (text := self.registry()) else None), 10)
        old_line = next((line for line in registered_text.splitlines()
                         if old.client_id in line), "")
        if "dfkv-remove-fanout=1" in old_line:
            raise RuntimeError(f"old library unexpectedly advertised fanout: {old_line}")
        self.wait_ring(self.writer, 9, 15)
        written = self.put(self.writer, new_keys)
        if not written["all_ok"]:
            raise RuntimeError(f"could not place exact values on added owner: {written}")
        remove_second = self.allocate_second()
        self.wait_second(remove_second)
        reader_before = self.client_core(self.reader)
        old_before = self.client_core(old)
        if (reader_before.get("ring_members") != 8
                or old_before.get("ring_members") != 9):
            raise RuntimeError(
                f"old-remover view skew vanished before actual REMOVE: "
                f"reader={reader_before}, old={old_before}")
        removed = self.remove(old, new_keys, remove_second)
        if removed["confirmed_count"] != len(new_keys):
            raise RuntimeError(f"old REMOVE did not confirm every new-owner key: {removed}")
        bypass: dict[str, Any] = {}
        if negative:
            # The old registration existed for the REMOVE. Retire that exact
            # handle and record afterward so this control has no effective
            # registry gate; do not pretend the old remover was unregistered.
            self.close_client(old)
            bypass = self.delete_client_record(old.client_id)
            c35._until("old registry record absent for negative control",
                       lambda: old.client_id not in self.registry(), 10)
            final_gate = self.wait_gate(self.reader, True, 20)
        else:
            final_gate = self.wait_gate(self.reader, False, 20)
        final_view = self.wait_ring(self.reader, 9, 20)
        before_read = self.client_core(self.reader)
        final_read = self.read(self.reader, new_keys)
        after_read = self.client_core(self.reader)
        cooling_probe = self.gate_cooling_probe()
        self.specific = {
            "new_keys": new_keys, "old_owners": {str(k): old_owners[k] for k in new_keys},
            "new_owners": new_locations, "added_member": new_address,
            "old_registered_at_remove": old.client_id in registered_text,
            "old_registry_row": old_line,
            "reader_before_remove": reader_before, "old_before_remove": old_before,
            "remove": removed, "registry_bypass_after_remove": bypass,
            "gate_at_final": final_gate, "final_view": final_view,
            "before_final_read": before_read, "final_read": final_read,
            "after_final_read": after_read, "cooling_probe": cooling_probe,
        }
        self.check("old_remover_was_ahead_and_registered",
                   reader_before.get("ring_members") == 8
                   and old_before.get("ring_members") == 9
                   and old.client_id in registered_text
                   and "dfkv-remove-fanout=1" not in old_line, self.specific)
        if negative:
            self.check("negative_control_actually_stale_via_open_gate",
                       final_gate.get("registry_gate_open") == 1
                       and final_read["hits"] > 0
                       and final_read["payload_mismatches"] == 0
                       and (after_read.get("readback_hit") or 0)
                       > (before_read.get("readback_hit") or 0)
                       and bypass.get("present_before_delete") is True
                       and cooling_probe["reroutes_get_delta"] > 0
                       and cooling_probe["after_down_owner_reads"].get(
                           "registry_gate_open") == 1,
                       self.specific)
        else:
            self.check("unmarked_old_remover_disables_C_and_B",
                       final_gate.get("registry_gate_open") == 0
                       and (final_gate.get("unmarked_clients") or 0) >= 1
                       and final_read["hits"] == 0
                       and after_read.get("readback_total")
                       == before_read.get("readback_total")
                       and cooling_probe["reroutes_get_delta"] == 0
                       and cooling_probe["after_down_owner_reads"].get(
                           "registry_gate_open") == 0,
                       self.specific)

    def drain_and_stop(self, node: c35.Node, step: int, expected_count: int) -> dict[str, Any]:
        started = time.monotonic_ns()
        address = f"127.0.0.1:{node.port}"
        completed = subprocess.run([str(self.args.ctl), "drain", address],
                                   text=True, capture_output=True, timeout=20)
        if completed.returncode != 0:
            raise RuntimeError(f"drain {address} failed: {completed.stdout} {completed.stderr}")
        node.child.popen.terminate()
        node.child.popen.wait(timeout=10)
        node.child.log_stream.flush()
        ring = self.wait_mds_members(expected_count, 2.8)
        evidence = {
            "step": step, "node": node.node_id, "pid": node.child.popen.pid,
            "returncode": node.child.popen.returncode, "started_ns": started,
            "observed_ns": time.monotonic_ns(), "expected_count": expected_count,
            "drain_stdout": completed.stdout.strip(), "members": ring["members"],
        }
        self.timeline("decline_member_left", **evidence)
        return evidence

    def run_held_decline(self) -> None:
        self.setup(18, 3000)
        assert self.reader is not None
        reserved = list(range(32)
                          )
        control = list(range(32, 160))
        early: dict[str, Any] = {}
        declines: list[dict[str, Any]] = []
        reader_counts: list[int] = []
        final_pre: dict[str, Any] = {}
        final_core: dict[str, Any] = {}
        final_read: dict[str, Any] = {}
        fresh: ChurnClient | None = None
        fresh_view: dict[str, Any] = {}
        control_cursor = 0
        final_second: int | None = None
        # One member leaves per 3 s reader poll, fifteen times. After the last
        # distinct view, GET_EXIST keeps exact per-second starts without any
        # possibility of issuing B read-back while the final value stabilizes.
        for second in range(90):
            step = (second - 3) // 3 if second >= 3 and (second - 3) % 3 == 0 and second <= 48 else None
            if control_cursor < len(control):
                flat = control[control_cursor]
                control_cursor += 1
            else:
                flat = control[-1]
            if second <= 48:
                self.read(self.reader, [flat], second)
            else:
                self.exist(self.reader, [flat], second)
            if step is not None and 0 <= step < 15:
                node_index = 17 - step
                evidence = self.drain_and_stop(self.nodes[node_index], step,
                                               expected_count=17 - step)
                declines.append(evidence)
                reader_counts.append(int(self.client_core(self.reader)["ring_members"]))
                if step == 4:
                    fresh = self.open_client("fresh-remover", 3000)
                    self.remover = fresh
                    fresh_view = self.wait_ring(fresh, 13, 2)
                    removed = self.remove(fresh, reserved, second)
                    # A fresh client's old-view REMOVE can be a confirmed miss:
                    # no value exists at its new owner.  Do not dress such an
                    # unconfirmed tombstone up as a read-back obligation while
                    # the reader is still on the old direct-owner view.  The
                    # final post-adoption read below covers every candidate.
                    confirmed = [flat for flat, ok in zip(reserved, removed["oks"]) if ok]
                    if not confirmed:
                        raise RuntimeError("fresh remover confirmed no early REMOVE")
                    post = self.read(self.reader, confirmed, second)
                    early = {"remove": removed, "confirmed_early_keys": confirmed,
                             "post_remove_read": post, "fresh_first_view": fresh_view,
                             "reader_ring_at_remove": reader_counts[-1]}
                    self.timeline("fresh_remover_early_remove", **early)
                if step == 14:
                    final_pre = self.client_core(self.reader)
            if len(declines) == 15 and second > 48:
                core = self.client_core(self.reader)
                if core.get("ring_members") == 3:
                    final_core = core
                    final_second = second
                    final_read = self.read(self.reader, reserved, second)
                    break
        if fresh is None or len(declines) != 15 or not final_core:
            raise RuntimeError("decline/fresh-remover/final-adoption sequence did not complete")
        after_final_read = self.client_core(self.reader)
        # The first drain is scheduled one 3 s poll after the workload origin;
        # the fifteenth is at second 45. Measure the whole fifteen-poll sequence,
        # not only the fourteen intervals between successive drains.
        duration_ms = ((declines[-1]["observed_ns"] - self.origin_ns)
                       / 1_000_000)
        self.specific = {
            "declines": declines, "reader_counts_during_decline": reader_counts,
            "decline_from_workload_origin_ms": duration_ms, "required_decline_ms": 15 * 3000,
            "fresh": early, "final_adoption_second": final_second,
            "before_final_stabilization": final_pre, "at_final_adoption": final_core,
            "final_read": final_read, "after_final_read": after_final_read,
        }
        self.check("fifteen_real_member_poll_declines_while_reader_held",
                   len(declines) == 15 and duration_ms >= 15 * 3000
                   and len(reader_counts) == 15
                   and all(count == 18 for count in reader_counts)
                   and fresh_view.get("ring_members") == 13, self.specific)
        self.check("fresh_remover_early_remove_exercised",
                   early["remove"]["confirmed_count"] > 0
                   and early["post_remove_read"]["hits"] == 0, early)
        held_delta = ((final_core.get("readback_changes_refused_held") or 0)
                      - (final_pre.get("readback_changes_refused_held") or 0))
        self.check("long_guard_hold_refused_without_readback",
                   held_delta > 0
                   and final_core.get("readback_total") == final_pre.get("readback_total")
                   and after_final_read.get("readback_total") == final_core.get("readback_total")
                   and final_read["hits"] == 0,
                   {"held_delta": held_delta, "before": final_pre,
                    "adopted": final_core, "after_final_read": after_final_read})

    # ---- offline verdict and lifecycle ----

    def check(self, name: str, passed: bool, detail: Any) -> None:
        c35.add_check(self.result, name, passed, detail)

    def collect_offline(self) -> None:
        correctness = c35.check_operations(self.root / "operations.jsonl")
        self.result["correctness"] = correctness
        negative = self.scenario == "added-member-negative"
        removed_hits = [row for row in correctness["violations"]
                        if row["type"] == "removed_value_hit"]
        self.check("complete_and_well_formed_operation_evidence",
                   correctness["incomplete_attempt_count"] == 0
                   and correctness["evidence_error_count"] == 0,
                   {"incomplete": correctness["incomplete_attempts"][:3],
                    "errors": correctness["evidence_errors"][:3]})
        self.check("zero_payload_corruption",
                   not any(row["type"] == "payload_mismatch"
                           for row in correctness["violations"]),
                   correctness["violations"][:3])
        if negative:
            self.check("required_stale_negative_control_fires",
                       len(removed_hits) > 0,
                       {"removed_value_hits": len(removed_hits),
                        "first": removed_hits[:3],
                        "all_violations": correctness["violation_count"]})
        else:
            self.check("zero_stale_or_removed_value_violations",
                       correctness["violation_count"] == 0,
                       {"count": correctness["violation_count"],
                        "first": correctness["violations"][:3]})
        invalid = [row for row in self.scheduled_rows if not row["actual_start_valid"]]
        self.check("every_actual_native_start_is_in_its_scheduled_second",
                   bool(self.scheduled_rows) and not invalid,
                   {"scheduled_native_operations": len(self.scheduled_rows),
                    "invalid": invalid[:3]})
        coverage = self.result.get("memory_coverage", {})
        self.check("real_H_memory_coverage_or_explicit_non_acceptance_smoke",
                   (self.args.smoke and coverage.get("non_acceptance") is True)
                   or (not self.args.smoke and coverage.get("waited") is True
                       and coverage.get("warmup_bypassed") is False), coverage)
        self.result["evidence"] = {
            "operations": "operations.jsonl", "per_second": "per_second.jsonl",
            "timeline": "timeline.jsonl", "metrics": "metrics.jsonl and metrics/*.prom",
            "membership": "membership.jsonl", "churn": "churn.jsonl",
            "processes": "processes.jsonl", "logs": "logs/",
        }

    def cleanup(self) -> list[str]:
        if self._cleaned_up:
            return []
        self._cleaned_up = True
        errors: list[str] = []
        for client in list(self.clients.values())[::-1]:
            try:
                client.close()
            except Exception as exc:
                errors.append(f"close {client.client_id}: {exc}")
        self.clients.clear()
        self.restore_client_env()
        try:
            self.processes.close()
        except Exception as exc:
            errors.append(f"process cleanup: {exc}")
        live = [child.name for child in self.processes.children
                if child.popen.poll() is None]
        live_threads = [thread.name for thread in threading.enumerate()
                        if thread is not threading.current_thread()
                        and thread.name.startswith("cli39") and thread.is_alive()]
        if live:
            errors.append(f"processes alive: {live}")
        if live_threads:
            errors.append(f"threads alive: {live_threads}")
        for sink in self.sinks.values():
            try:
                sink.close()
            except Exception as exc:
                errors.append(f"evidence close: {exc}")
        self.result["cleanup"] = {"processes_alive": live,
                                  "worker_threads_alive": live_threads,
                                  "errors": errors}
        return errors

    def run(self) -> dict[str, Any]:
        try:
            self.execute()
        except BaseException as exc:
            self.result["errors"].append(f"{type(exc).__name__}: {exc}")
            (self.root / "error.log").write_text(traceback.format_exc())
        self.result["errors"].extend(self.cleanup())
        try:
            self.collect_offline()
        except BaseException as exc:
            self.result["errors"].append(f"offline evaluation: {type(exc).__name__}: {exc}")
            (self.root / "evaluation-error.log").write_text(traceback.format_exc())
        failures = [check for check in self.result["acceptance_checks"]
                    if not check["passed"]]
        if self.result["errors"]:
            self.result["status"] = "ERROR"
        elif failures:
            self.result["status"] = "FAILED"
        elif self.args.smoke:
            self.result["status"] = "SMOKE_MECHANICS_COMPLETE_NON_ACCEPTANCE"
        else:
            self.result["status"] = "CHECKED"
        self.result["specific"] = self.specific
        c35._write_json(self.root / "result.json", self.result)
        return self.result


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--old-lib", type=Path,
                        help="registered pre-fanout libdfkv for both added-member roots")
    parser.add_argument("--scenarios", help=",".join(ALL_SCENARIOS) + " (CSV debug subset)")
    parser.add_argument("--smoke", action="store_true",
                        help="bypass M warmup only; mechanics are not acceptance")
    args = parser.parse_args(argv)
    scenarios = tuple(args.scenarios.split(",")) if args.scenarios else ALL_SCENARIOS
    if not scenarios or any(name not in ALL_SCENARIOS for name in scenarios):
        parser.error(f"--scenarios must be a CSV of {','.join(ALL_SCENARIOS)}")
    if len(set(scenarios)) != len(scenarios):
        parser.error("duplicate --scenarios entries are not allowed")
    args.scenarios = scenarios
    build = args.build_dir.resolve()
    args.server = c35.executable(build / "dfkv_server", "dfkv_server")
    args.mds = c35.executable(build / "dfkv_mds", "dfkv_mds")
    args.ctl = c35.executable(build / "dfkvctl", "dfkvctl")
    args.lib = (build / "libdfkv.so").resolve()
    if not args.lib.is_file():
        parser.error(f"missing libdfkv.so: {args.lib}")
    if any(name.startswith("added-member-") for name in scenarios):
        if args.old_lib is None:
            parser.error("added-member scenarios require --old-lib PATH")
        args.old_lib = args.old_lib.expanduser().resolve()
        if not args.old_lib.is_file():
            parser.error(f"pre-fanout libdfkv is not a regular file: {args.old_lib}")
    elif args.old_lib is not None:
        args.old_lib = args.old_lib.expanduser().resolve()
        if not args.old_lib.is_file():
            parser.error(f"pre-fanout libdfkv is not a regular file: {args.old_lib}")
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if any(args.output_root.iterdir()):
        parser.error(f"output root must be empty: {args.output_root}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    etcd = c35.discover_etcd()
    args.lib_handle = c35.load_library(args.lib)
    args.old_lib_handle = c35.load_library(args.old_lib) if args.old_lib else None
    workload = ChurnWorkload()
    version = args.lib_handle.dfkv_version()
    old_version = (args.old_lib_handle.dfkv_version()
                   if args.old_lib_handle is not None else None)
    binaries = [("server", args.server), ("mds", args.mds), ("dfkvctl", args.ctl),
                ("libdfkv", args.lib), ("etcd", etcd)]
    if args.old_lib:
        binaries.append(("old_libdfkv", args.old_lib))
    metadata = {
        "wall_time": c35._iso(),
        "argv": sys.argv if argv is None else [sys.argv[0], *argv],
        "cwd": os.getcwd(), "python": sys.version, "platform": sys.platform,
        "build_dir": str(args.build_dir.resolve()),
        "output_root": str(args.output_root), "requested_scenarios": args.scenarios,
        "full_matrix": set(args.scenarios) == set(ALL_SCENARIOS), "smoke": args.smoke,
        "binaries": {name: {"path": str(path), "sha256": c35._sha256_file(path)}
                     for name, path in binaries},
        "libdfkv_version": version.decode(errors="replace") if version else "",
        "old_libdfkv_version": old_version.decode(errors="replace") if old_version else "",
        "workload_identity_sha256": workload.identity_sha256,
        "environment_whitelist": {key: os.environ.get(key) for key in ENV_KEYS},
        "assumptions": [
            "Linux loopback TCP; real etcd, MDS and file-engine dfkv_server; no GPU/RDMA/production",
            "old lib is wire/MDS compatible, registers without fanout marker, and exposes ABI-v2 plus stats/ring metrics",
            "negative registry deletion happens only after the old REMOVE and is explicit evidence",
            "decline uses dfkvctl drain followed by graceful stop for immediate real deregistration, one member per reader poll",
            "H is readback_ms + 10 * actual maximum registered poll_ms; full runs wait that real M coverage before origin",
        ],
    }
    c35._write_json(args.output_root / "run-metadata.json", metadata)
    results: list[dict[str, Any]] = []
    for scenario in args.scenarios:
        print(f"[cli39] starting {scenario}; evidence={args.output_root / scenario}", flush=True)
        result = ChurnRun(args, scenario, workload, etcd).run()
        results.append(result)
        print(f"[cli39:{scenario}] status={result['status']} "
              f"violations={result.get('correctness', {}).get('violation_count', 'unknown')} "
              f"errors={result['errors']}", flush=True)
    full = set(args.scenarios) == set(ALL_SCENARIOS)
    all_ran = len(results) == len(args.scenarios)
    failed = any(result["status"] in ("ERROR", "FAILED") for result in results)
    if args.smoke:
        overall = "SMOKE_FAILED" if failed else "SMOKE_MECHANICS_COMPLETE_NON_ACCEPTANCE"
    elif not full or not all_ran:
        overall = "FAILED" if failed else "DEBUG_SUBSET_COMPLETE_NON_ACCEPTANCE"
    else:
        overall = "ACCEPTANCE_FAIL" if failed else "ACCEPTANCE_PASS"
    summary = {
        "wall_time": c35._iso(), "overall": overall,
        "acceptance": overall == "ACCEPTANCE_PASS", "smoke": args.smoke,
        "full_matrix": full, "all_requested_scenarios_ran": all_ran,
        "negative_control_required_and_observed": next((
            result["correctness"]["violation_count"] > 0
            for result in results if result["scenario"] == "added-member-negative"
            and "correctness" in result), False),
        "scenarios": results,
    }
    c35._write_json(args.output_root / "result.json", summary)
    print(f"[cli39] {overall}; result={args.output_root / 'result.json'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
