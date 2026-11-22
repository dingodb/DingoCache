#!/usr/bin/env python3
"""Live multi-group readiness/version/failure coverage for ``dfkvctl stat --all``,
and the deploy/ operational scripts run against what the real ``dfkvctl`` prints."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy"))

from dfkv_membership_audit import main as audit_main  # noqa: E402
from dfkv_node_replace import Replacer  # noqa: E402
from dfkv_ops_common import parse_ring  # noqa: E402


class _EtcdState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.values: dict[bytes, bytes] = {}
        self.revision = 1
        self.next_lease = 100


class _EtcdHandler(BaseHTTPRequestHandler):
    state: _EtcdState

    def log_message(self, _format: str, *_args) -> None:
        pass

    def _reply(self, value: dict) -> None:
        body = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        state = self.state
        if self.path == "/v3/lease/grant":
            with state.lock:
                lease = state.next_lease
                state.next_lease += 1
            self._reply({"ID": str(lease), "TTL": request.get("TTL", "30")})
            return
        if self.path == "/v3/lease/keepalive":
            self._reply({"result": {"ID": request.get("ID", "0"), "TTL": "30"}})
            return
        if self.path == "/v3/kv/put":
            key = base64.b64decode(request["key"])
            value = base64.b64decode(request["value"])
            with state.lock:
                state.values[key] = value
                state.revision += 1
                revision = state.revision
            self._reply({"header": {"revision": str(revision)}})
            return
        if self.path == "/v3/kv/range":
            start = base64.b64decode(request["key"])
            end = base64.b64decode(request.get("range_end", ""))
            with state.lock:
                rows = [
                    {"key": base64.b64encode(key).decode(),
                     "value": base64.b64encode(value).decode()}
                    for key, value in sorted(state.values.items())
                    if key >= start and (not end or key < end)
                ]
                revision = state.revision
            response = {"header": {"revision": str(revision)}, "count": str(len(rows))}
            if rows:
                response["kvs"] = rows
            self._reply(response)
            return
        self.send_error(404)


# A port number that was free a moment ago is not a port this test owns: the
# kernel hands the same number to the next listener or outgoing connection on
# the host. So nothing here is started on a number picked in advance and then
# trusted. A process that can report its own port listens on port 0; one that
# must be told its port beforehand is restarted on another number when the
# first is gone; and a port that must stay closed is held bound for as long
# as the test needs it closed.


def _free_port() -> int:
    """A candidate only: it may be taken before the caller listens on it."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _ClosedPort:
    """A port nothing listens on, and nothing else can take, while open."""

    def __enter__(self) -> int:
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))  # bound, never listening: refuses
        return int(self._sock.getsockname()[1])

    def __exit__(self, *exc) -> None:
        self._sock.close()


class _Exited(RuntimeError):
    pass


def _wait_line(log: Path, process: subprocess.Popen, prefix: str,
               timeout: float = 10.0) -> str:
    """The first line of `log` starting with `prefix`: what the process prints
    once it is listening. A connect to the port would also succeed against a
    stranger that took the number first."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith(prefix):
                return line
        if process.poll() is not None:
            raise _Exited(
                f"process exited {process.returncode} before printing {prefix!r}")
        time.sleep(0.05)
    raise RuntimeError(f"{prefix!r} not printed within {timeout}s")


def _start_server(dfkv_server: Path, root: Path, node: str, group: str,
                  mds_port: int, processes: list, logs: list):
    """dfkv_server registers the address it is given, so its port is chosen
    before it starts; when that port has been taken meanwhile, try another."""
    data = root / node
    data.mkdir()
    for attempt in range(1, 6):
        port = _free_port()
        log = (root / f"{node}.{attempt}.log").open("w+")
        logs.append(log)
        process = subprocess.Popen(
            [str(dfkv_server), "--dir", str(data), "--port", str(port),
             "--cap", str(64 << 20), "--store-engine", "file",
             "--mds", f"127.0.0.1:{mds_port}", "--group", group,
             "--id", node, "--advertise", f"127.0.0.1:{port}",
             "--mds-registration-timeout-ms", "5000"],
            stdout=log, stderr=subprocess.STDOUT, text=True,
            env={**os.environ, "DFKV_LOG_LEVEL": "ERROR"},
        )
        processes.append(process)
        try:
            _wait_line(Path(log.name), process, "PORT ")
            return process, port
        except _Exited:
            if f"failed to start on port {port}" not in Path(log.name).read_text(
                    errors="replace"):
                raise
    raise RuntimeError(f"{node}: no free port in 5 attempts")


def _stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _run_until(dfkvctl: Path, mds_port: int, predicate, timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    latest = None
    while time.monotonic() < deadline:
        latest = subprocess.run(
            [str(dfkvctl), "stat", "--all", "--mds", f"127.0.0.1:{mds_port}"],
            text=True, capture_output=True, timeout=10, check=False,
        )
        if predicate(latest):
            return latest
        time.sleep(0.1)
    detail = "" if latest is None else f"rc={latest.returncode}\nout={latest.stdout}\nerr={latest.stderr}"
    raise RuntimeError(f"dfkvctl stat --all did not reach expected state\n{detail}")


def _registration(node_id: str, ip: str, port: int, info: bytes) -> bytes:
    """An etcd registration value as the MDS writes it: EncodeMembers of one member."""
    value = struct.pack("=QI", 0, 1)
    for text in (node_id.encode(), ip.encode()):
        value += struct.pack("=I", len(text)) + text
    value += struct.pack("=II", port, 1)
    value += struct.pack("=II", 0x54435031, 0)
    value += struct.pack("=II", 0x4E464F31, len(info)) + info
    return value


def _members_epoch(members) -> int:
    """MembersEpoch of src/common/membership.h, restated: FNV-1a over the
    eligible members in id order. Scripts take the epoch from dfkvctl; this is
    here so that a change of the definition does not pass unnoticed."""
    value = 1469598103934665603
    for member in sorted((m for m in members if m.eligible), key=lambda m: m.node_id):
        fields = (
            struct.pack("=I", len(member.node_id)) + member.node_id.encode()
            + struct.pack("=I", len(member.ip)) + member.ip.encode()
            + struct.pack("=II", member.port, member.weight) + b"\x01"
        )
        for byte in fields:
            value = ((value ^ byte) * 1099511628211) % 2**64
    return value


def _check_operational_tools(dfkvctl: Path, mds: str, etcd: str, state: _EtcdState,
                             address: str, version: str, root: Path) -> None:
    """deploy/ scripts read `ring --json`, `clients` and `stat --all` of the real binary."""
    def ctl(*arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(dfkvctl), *arguments, "--mds", mds], text=True, capture_output=True,
            timeout=10, check=False)

    def require(condition: bool, what: str, detail: object) -> None:
        if not condition:
            raise RuntimeError(f"{what}\n{detail}")

    # A registered consumer, so that `dfkvctl clients` has a row to print.
    with state.lock:
        state.values[b"/dfkv/v1/groups/alpha/clients/client-1"] = _registration(
            "client-1", "127.0.0.1", 0, b"type=vllm,model=m")
    workflow = Replacer(SimpleNamespace(
        dfkvctl=str(dfkvctl), mds=mds, group="alpha", command_timeout=10.0))
    view = workflow.ring()
    require(len(view.members) == 1, "ring --json member count", view)
    member = view.members[0]
    require(
        (member.node_id, member.address, member.weight, member.eligible)
        == ("alpha-node", address, 1, True)
        and member.vnodes == view.ring_points > 0 and member.share_percent == 100.0
        and f"ver={version}" in member.info,
        "ring --json member", view)
    require(view.epoch == _members_epoch(view.members), "ring --json placement_epoch", view)
    require(workflow.clients() == {"client-1"}, "clients", ctl("clients", "--group", "alpha"))
    require(workflow.healthy("alpha-node") is True, "stat --all readiness",
            ctl("stat", "--all", "--group", "alpha"))

    # The table is what people and scripts outside this repository read.
    table = ctl("ring", "--group", "alpha").stdout.splitlines()
    require(
        len(table) == 3
        and table[0] == f"group=alpha members=1 eligible=1 ring_points={view.ring_points}"
        and table[1].split() == [
            "ID", "ADDR", "STATUS", "WEIGHT", "VNODES", "SHARE", "IB-HEALTH", "INFO"]
        and table[2].split()[:3] == ["alpha-node", address, "ACTIVE"],
        "ring table changed", table)

    empty = ctl("ring", "--json", "--group", "unused")
    require(empty.returncode == 1 and parse_ring(empty.stdout).members == (),
            "ring --json of an empty group", empty)
    empty = ctl("ring", "--json", "--allow-empty", "--group", "unused")
    require(empty.returncode == 0 and parse_ring(empty.stdout).members == (),
            "ring --json --allow-empty", empty)
    require(ctl("clients", "--json", "--group", "alpha").returncode == 2,
            "--json outside ring must be refused", "")

    report_path = root / "membership-audit.json"
    audit_arguments = [
        "--mds", mds, "--etcd", etcd, "--group", "alpha", "--dfkvctl", str(dfkvctl),
        "--timeout", "10", "--output", str(report_path)]
    status = audit_main(audit_arguments)
    report = json.loads(report_path.read_text())
    require(
        status == 0 and report["healthy"] and not report["issues"]
        and report["ring_epoch"] == view.epoch
        and [item["node_id"] for item in report["registrations"]] == ["alpha-node"],
        "membership audit of a consistent group", report)

    # A second member that is registered but not there: nothing listens on its
    # port, and part of its self-description is not text. `stat --all` then
    # exits 1, which must not hide the row of the healthy member.
    ghost_key = b"/dfkv/v1/groups/alpha/members/ghost"
    closed = _ClosedPort()
    ghost_port = closed.__enter__()
    with state.lock:
        state.values[ghost_key] = _registration(
            "ghost", "127.0.0.1", ghost_port, b"ver=0.0,note=\xff\xfe")
    try:
        stat = ctl("stat", "--all", "--group", "alpha")
        rows = [line.split() for line in stat.stdout.splitlines()]
        require(
            stat.returncode == 1
            and rows[0] == [
                "ID", "ADDR", "VERSION", "READY", "HEALTH", "MDS", "RDMA", "RD-ERR",
                "USED_MB", "OBJECTS", "HITS", "MISSES", "HIT%"]
            and [len(row) for row in rows if row[0] == "alpha-node"] == [len(rows[0])]
            and ["ghost", f"127.0.0.1:{ghost_port}", "0.0", "(unreachable)"] in rows,
            "stat --all table with an unreachable member", stat)
        require(workflow.healthy("alpha-node") is True and workflow.healthy("ghost") is False,
                "readiness of one member next to an unreachable one", stat)
        infos = {item.node_id: item.info for item in workflow.ring().members}
        require(infos.get("ghost") == "ver=0.0,note=\xff\xfe",
                "ring --json of bytes that are not UTF-8", infos)
        status = audit_main(audit_arguments)
        issues = json.loads(report_path.read_text())["issues"]
        require(
            status == 2
            and [(issue["code"], issue.get("node")) for issue in issues]
            == [("self_report_incomplete", "ghost")],
            "membership audit next to a member with a non-text self-description", issues)
    finally:
        with state.lock:
            del state.values[ghost_key]
        closed.__exit__()

    # No table at all is dfkvctl failing, not a verdict on the members.
    with _ClosedPort() as no_mds_port:
        no_mds = Replacer(SimpleNamespace(
            dfkvctl=str(dfkvctl), mds=f"127.0.0.1:{no_mds_port}", group="alpha",
            command_timeout=10.0))
        try:
            no_mds.healthy("alpha-node")
        except RuntimeError as error:
            require("MDS query failed" in str(error), "stat --all without an MDS", error)
        else:
            raise RuntimeError("stat --all without an MDS must stay an error")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_dfkvctl_all_groups.py <build-dir>")
    build = Path(sys.argv[1]).resolve()
    dfkvctl = build / "dfkvctl"
    dfkv_mds = build / "dfkv_mds"
    dfkv_server = build / "dfkv_server"
    for binary in (dfkvctl, dfkv_mds, dfkv_server):
        if not binary.is_file():
            raise RuntimeError(f"missing test binary: {binary}")

    state = _EtcdState()
    handler = type("EtcdHandler", (_EtcdHandler,), {"state": state})
    etcd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    etcd_thread = threading.Thread(target=etcd.serve_forever, daemon=True)
    etcd_thread.start()

    processes: list[subprocess.Popen] = []
    logs = []
    stopped_server: subprocess.Popen | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="dfkvctl-all-groups-") as temp:
            root = Path(temp)
            # The MDS says which port it got: let the kernel choose it.
            mds_log = (root / "mds.log").open("w+")
            logs.append(mds_log)
            mds = subprocess.Popen(
                [str(dfkv_mds), "--listen", "0", "--etcd",
                 f"127.0.0.1:{etcd.server_port}"],
                stdout=mds_log, stderr=subprocess.STDOUT, text=True,
            )
            processes.append(mds)
            listening = _wait_line(root / "mds.log", mds, "dfkv_mds listening on ")
            mds_port = int(listening.split()[3].rstrip(","))

            servers = []
            addresses = {}
            for group, node in (("alpha", "alpha-node"), ("beta", "beta-node")):
                process, port = _start_server(
                    dfkv_server, root, node, group, mds_port, processes, logs)
                addresses[node] = f"127.0.0.1:{port}"
                servers.append(process)

            version = subprocess.check_output(
                [str(dfkvctl), "--version"], text=True, timeout=5).strip().split()[1]
            healthy = _run_until(
                dfkvctl, mds_port,
                lambda result: (
                    result.returncode == 0
                    and "== group=alpha ==" in result.stdout
                    and "== group=beta ==" in result.stdout
                    and "alpha-node" in result.stdout
                    and "beta-node" in result.stdout
                    and result.stdout.count(version) >= 2
                    and result.stdout.count("yes") >= 6
                ),
            )
            if "unreachable=0 unhealthy=0 health-unknown=0" not in healthy.stdout:
                raise RuntimeError(f"healthy fleet summary missing\n{healthy.stdout}")

            _check_operational_tools(
                dfkvctl, f"127.0.0.1:{mds_port}", f"http://127.0.0.1:{etcd.server_port}",
                state, addresses["alpha-node"], version, root)

            stopped_server = servers[1]
            _stop(stopped_server)
            failed = _run_until(
                dfkvctl, mds_port,
                lambda result: (
                    result.returncode == 1
                    and "== group=alpha ==" in result.stdout
                    and "== group=beta ==" in result.stdout
                    and "alpha-node" in result.stdout
                    and "beta-node" in result.stdout
                    and "(unreachable)" in result.stdout
                    and "unreachable=1" in result.stdout
                ),
            )
            if failed.stdout.count(version) < 2:
                raise RuntimeError(f"version inventory disappeared on failure\n{failed.stdout}")
    except Exception:
        for log in logs:
            try:
                log.flush()
                log.seek(0)
                print(f"--- {log.name} ---\n{log.read()}", file=sys.stderr)
            except Exception:
                pass
        raise
    finally:
        for process in reversed(processes):
            if process is not stopped_server:
                _stop(process)
        etcd.shutdown()
        etcd.server_close()
        etcd_thread.join(timeout=5)
        for log in logs:
            log.close()
    print("dfkvctl multi-group readiness/version/failure and operational-tools smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
