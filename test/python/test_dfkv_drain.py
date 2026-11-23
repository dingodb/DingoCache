#!/usr/bin/env python3
"""MDS-04 drain, end to end: dfkv_server with DFKV_DRAIN, dfkv_mds over a
stand-in etcd, and dfkvctl.

- `dfkvctl drain` marks a node draining in the MDS view (ring --json and the
  table) and `--undrain` clears it; a node without DFKV_DRAIN refuses.
- A node drained by dfkvctl and then stopped deletes its record.
- With DFKV_STOP_DRAIN_MS a stop drains first: the node is listed draining,
  keeps serving for that long, then stops and keeps its record (a restart).
  A second stop signal ends the wait.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_dfkvctl_all_groups import (  # noqa: E402
    _EtcdHandler,
    _EtcdState,
    _start_server,
    _stop,
    _wait_line,
)

GROUP = "drain"


def _ring(dfkvctl: Path, mds: str) -> dict[str, dict]:
    out = subprocess.run(
        [str(dfkvctl), "ring", "--json", "--allow-empty", "--group", GROUP, "--mds", mds],
        text=True, capture_output=True, timeout=10, check=True).stdout
    return {member["id"]: member for member in json.loads(out)["members"]}


def _until(what: str, predicate, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise RuntimeError(f"timed out: {what}")


def _ctl(dfkvctl: Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(dfkvctl), *arguments], text=True, capture_output=True,
                          timeout=30, check=False)


def _require(condition: bool, what: str, detail: object = "") -> None:
    if not condition:
        raise RuntimeError(f"{what}\n{detail}")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_dfkv_drain.py <build-dir>")
    build = Path(sys.argv[1]).resolve()
    dfkvctl, dfkv_mds, dfkv_server = (build / "dfkvctl", build / "dfkv_mds",
                                      build / "dfkv_server")
    state = _EtcdState()
    handler = type("EtcdHandler", (_EtcdHandler,), {"state": state})
    etcd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=etcd.serve_forever, daemon=True).start()
    processes: list[subprocess.Popen] = []
    logs = []
    try:
        with tempfile.TemporaryDirectory(prefix="dfkv-drain-") as temp:
            root = Path(temp)
            mds_log = (root / "mds.log").open("w+")
            logs.append(mds_log)
            mds_process = subprocess.Popen(
                [str(dfkv_mds), "--listen", "0", "--etcd", f"127.0.0.1:{etcd.server_port}"],
                stdout=mds_log, stderr=subprocess.STDOUT, text=True)
            processes.append(mds_process)
            line = _wait_line(root / "mds.log", mds_process, "dfkv_mds listening on ")
            mds = f"127.0.0.1:{int(line.split()[3].rstrip(','))}"
            mds_port = int(mds.rsplit(":", 1)[1])

            def start(node: str, env: dict) -> tuple[subprocess.Popen, str]:
                process, port = _start_server(dfkv_server, root, node, GROUP, mds_port,
                                              processes, logs, env)
                return process, f"127.0.0.1:{port}"

            ctl_node, ctl_addr = start("ctl", {"DFKV_DRAIN": "1"})
            stop_node, _ = start("stop", {"DFKV_DRAIN": "1", "DFKV_STOP_DRAIN_MS": "1500"})
            long_node, _ = start("long", {"DFKV_DRAIN": "1", "DFKV_STOP_DRAIN_MS": "60000"})
            plain_node, plain_addr = start("plain", {})
            _until("four members", lambda: len(_ring(dfkvctl, mds)) == 4)
            _require(not any(m["draining"] for m in _ring(dfkvctl, mds).values()),
                     "nobody drains yet", _ring(dfkvctl, mds))

            # dfkvctl drain / --undrain; refused by a node without DFKV_DRAIN.
            result = _ctl(dfkvctl, "drain", ctl_addr)
            _require(result.returncode == 0 and "draining" in result.stdout,
                     "dfkvctl drain", result)
            ring = _ring(dfkvctl, mds)  # the node answered once the MDS had it
            _require(ring["ctl"]["draining"] and not ring["plain"]["draining"],
                     "ctl is draining in ring --json", ring)
            _require(ring["ctl"]["status"] == "ACTIVE" and ring["ctl"]["eligible"],
                     "draining keeps status and placement", ring)
            table = _ctl(dfkvctl, "ring", "--group", GROUP, "--mds", mds).stdout
            rows = {line.split()[0]: line.split() for line in table.splitlines()[2:]}
            _require(rows["ctl"][2] == "DRAINING" and rows["plain"][2] == "ACTIVE",
                     "the table shows DRAINING", table)
            result = _ctl(dfkvctl, "drain", plain_addr)
            _require(result.returncode == 1 and "does not drain" in result.stderr,
                     "a node without DFKV_DRAIN refuses", result)
            result = _ctl(dfkvctl, "drain", ctl_addr, "--undrain")
            _require(result.returncode == 0 and "not draining" in result.stdout,
                     "dfkvctl drain --undrain", result)
            _require(not _ring(dfkvctl, mds)["ctl"]["draining"], "undrained",
                     _ring(dfkvctl, mds))

            # Drained by dfkvctl, then stopped: the record is deleted at once
            # (the stand-in etcd never expires a lease on its own).
            _require(_ctl(dfkvctl, "drain", ctl_addr).returncode == 0, "drain again")
            _stop(ctl_node)
            _require(ctl_node.returncode == 0, "the drained node stopped cleanly",
                     ctl_node.returncode)
            _until("ctl deregistered", lambda: "ctl" not in _ring(dfkvctl, mds), 5.0)

            # A stop drains first for DFKV_STOP_DRAIN_MS, keeps serving, then
            # stops and keeps its record.
            started = time.monotonic()
            stop_node.send_signal(signal.SIGTERM)
            _until("stop is draining", lambda: _ring(dfkvctl, mds)["stop"]["draining"], 1.4)
            _require(stop_node.poll() is None, "still serving while it drains")
            stop_node.wait(timeout=15)
            elapsed = time.monotonic() - started
            _require(stop_node.returncode == 0 and elapsed >= 1.5,
                     "the stop waited out DFKV_STOP_DRAIN_MS", (stop_node.returncode, elapsed))
            _require("stop" in _ring(dfkvctl, mds), "a stop keeps the record (restart)",
                     _ring(dfkvctl, mds))

            # A second stop signal ends a long drain.
            long_node.send_signal(signal.SIGTERM)
            _until("long is draining", lambda: _ring(dfkvctl, mds)["long"]["draining"])
            again = time.monotonic()
            long_node.send_signal(signal.SIGTERM)
            long_node.wait(timeout=15)
            _require(time.monotonic() - again < 10.0 and long_node.returncode == 0,
                     "a second signal ends the drain", long_node.returncode)

            # Without DFKV_DRAIN a stop is as before: no drain, record kept.
            _stop(plain_node)
            _require(not _ring(dfkvctl, mds)["plain"]["draining"], "plain never drained")
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
            _stop(process)
        etcd.shutdown()
        etcd.server_close()
        for log in logs:
            log.close()
    print("drain: dfkvctl drain/undrain, deregister on stop, stop-time drain OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
