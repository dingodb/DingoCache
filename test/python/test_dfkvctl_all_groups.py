#!/usr/bin/env python3
"""Live multi-group readiness/version/failure coverage for ``dfkvctl stat --all``."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


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


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_port(port: int, process: subprocess.Popen, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"process exited {process.returncode} before port {port} opened")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"port {port} did not open within {timeout}s")


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
            mds_port = _free_port()
            mds_log = (root / "mds.log").open("w+")
            logs.append(mds_log)
            mds = subprocess.Popen(
                [str(dfkv_mds), "--listen", str(mds_port), "--etcd",
                 f"127.0.0.1:{etcd.server_port}"],
                stdout=mds_log, stderr=subprocess.STDOUT, text=True,
            )
            processes.append(mds)
            _wait_port(mds_port, mds)

            servers = []
            for group, node in (("alpha", "alpha-node"), ("beta", "beta-node")):
                port = _free_port()
                data = root / node
                data.mkdir()
                log = (root / f"{node}.log").open("w+")
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
                servers.append(process)
                _wait_port(port, process)

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
    print("dfkvctl multi-group readiness/version/failure smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
