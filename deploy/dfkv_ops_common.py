#!/usr/bin/env python3
"""Shared parsers and bounded process helpers for dfkv operational scripts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class RingMember:
    node_id: str
    address: str
    weight: int
    vnodes: int
    share_percent: float
    info: str
    # False: registered but held out of the placement ring (no vnodes).
    eligible: bool = True

    @property
    def ip(self) -> str:
        return self.address.rsplit(":", 1)[0]

    @property
    def port(self) -> int:
        return int(self.address.rsplit(":", 1)[1])


@dataclass(frozen=True)
class RingView:
    group: str
    ring_points: int
    members: tuple[RingMember, ...]
    # Placement epoch as computed by dfkvctl (C++ MembersEpoch over the
    # eligible members): the value the MDS and clients compare.
    epoch: int

    def fingerprint(self) -> str:
        payload = json.dumps(
            [asdict(member) for member in sorted(self.members, key=lambda item: item.node_id)],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


_CLIENT_HEADER = re.compile(r"^group=(\S+) clients=(\d+)")
_BENCH_LINE = re.compile(
    r"^(PUT|GET)\s+n=(\d+)\s+size=(\d+)\s+threads=(\d+)\s+batch=(\d+)\s+\|\s+"
    r"([0-9.]+)s\s+goodput\s+([0-9.]+) GB/s\s+([0-9.]+) ok-ops/s\s+"
    r"call-lat ms p50=([0-9.]+) p99=([0-9.]+) max=([0-9.]+)\s+"
    r"ok=(\d+) fails=(\d+)$"
)
_LEGACY_BENCH_LINE = re.compile(
    r"^(PUT|GET)\s+n=(\d+)\s+size=(\d+)\s+threads=(\d+)\s+batch=(\d+)\s+\|\s+"
    r"([0-9.]+)s\s+([0-9.]+) GB/s\s+([0-9.]+) ops/s\s+"
    r"call-lat ms p50=([0-9.]+) p99=([0-9.]+) max=([0-9.]+)\s+"
    r"(?:ok=(\d+)\s+)?fails=(\d+)$"
)
_BENCH_DIAG = re.compile(r"^DIAG phase=(READY|SETTLE|GET_SETUP|PUT|GET)\s+(.+)$")


class CommandError(RuntimeError):
    """A command exited non-zero. Its output is kept: some exit codes are a
    verdict on what was printed rather than a failure to print it."""

    def __init__(self, argv: Sequence[str], returncode: int, stdout: str, stderr: str):
        detail = stderr.strip() or stdout.strip() or "no diagnostic"
        super().__init__(f"command exited {returncode}: {' '.join(argv)}: {detail}")
        self.returncode = returncode
        self.stdout = stdout


def run_command(argv: Sequence[str], timeout: float, *, env: dict[str, str] | None = None) -> str:
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    completed = subprocess.run(
        list(argv),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise CommandError(argv, completed.returncode, completed.stdout, completed.stderr)
    return completed.stdout


def parse_ring(text: str) -> RingView:
    """Parse the document printed by ``dfkvctl ring --json``."""
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"dfkvctl ring --json did not print JSON (dfkvctl without --json support?): {error}"
        ) from error
    def field(source: dict, name: str, kind: type):
        value = source[name]
        if type(value) is not kind:
            raise TypeError(f"{name} must be {kind.__name__}, got {value!r}")
        return value

    try:
        members = tuple(
            RingMember(
                node_id=field(item, "id", str),
                address=f"{field(item, 'ip', str)}:{field(item, 'port', int)}",
                weight=field(item, "weight", int),
                vnodes=field(item, "vnodes", int),
                share_percent=field(item, "share_percent", float),
                # "-" = the node reports no self-description.
                info=field(item, "info", str) or "-",
                eligible=field(item, "eligible", bool),
            )
            for item in field(document, "members", list)
        )
        epoch = field(document, "placement_epoch", str)
        if not re.fullmatch(r"[0-9]+", epoch):
            raise TypeError(f"placement_epoch must be a decimal string, got {epoch!r}")
        view = RingView(
            field(document, "group", str), field(document, "ring_points", int), members, int(epoch)
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"invalid dfkvctl ring --json document: {error}") from error
    if len({member.node_id for member in members}) != len(members):
        raise ValueError("ring contains duplicate node IDs")
    return view


def parse_clients(text: str) -> tuple[str, tuple[str, ...]]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("dfkvctl clients output is empty")
    match = _CLIENT_HEADER.match(lines[0])
    if not match:
        raise ValueError(f"invalid dfkvctl clients header: {lines[0]!r}")
    group, expected = match.group(1), int(match.group(2))
    if expected == 0:
        return group, ()
    if len(lines) < 2 or lines[1].split()[:2] != ["ID", "TYPE"]:
        raise ValueError("dfkvctl clients column header is missing")
    ids = tuple(line.split(None, 1)[0] for line in lines[2:])
    if len(ids) != expected:
        raise ValueError(f"clients header says {expected} clients but parsed {len(ids)}")
    return group, tuple(sorted(ids))


def parse_stat_health(text: str) -> dict[str, bool]:
    """Per member of ``dfkvctl stat --all --group``, the verdict dfkvctl itself
    applies to that row for its exit status: reachable, READY and HEALTH known
    and yes, MDS and RDMA not NO."""
    result: dict[str, bool] = {}
    header: list[str] | None = None
    for line in text.splitlines():
        columns = line.split()
        if columns[:2] == ["ID", "ADDR"]:
            header = columns
            if not {"READY", "HEALTH", "MDS", "RDMA"}.issubset(header):
                raise ValueError(f"dfkvctl stat --all has no readiness columns: {line!r}")
            continue
        if header is None or not columns or columns[0] == "TOTAL":
            continue
        if "(unreachable)" in columns:
            result[columns[0]] = False
            continue
        if len(columns) != len(header):
            raise ValueError(f"invalid dfkvctl stat --all row: {line!r}")
        row = dict(zip(header, columns))
        result[row["ID"]] = (
            row["READY"] == "yes" and row["HEALTH"] == "yes"
            and row["MDS"] != "NO" and row["RDMA"] != "NO"
        )
    if header is None:
        raise ValueError("dfkvctl stat --all column header is missing")
    return result


def parse_bench(text: str) -> dict[str, dict[str, float | int]]:
    phases: dict[str, dict[str, float | int]] = {}
    for raw in text.splitlines():
        match = _BENCH_LINE.match(raw.strip())
        legacy = False
        if not match:
            match = _LEGACY_BENCH_LINE.match(raw.strip())
            legacy = match is not None
        if not match:
            continue
        phase = match.group(1).lower()
        if phase in phases:
            raise ValueError(f"dfkv_bench emitted duplicate {phase.upper()} result")
        count = int(match.group(2))
        fails = int(match.group(13))
        phases[phase] = {
            "count": count,
            "size_bytes": int(match.group(3)),
            "threads": int(match.group(4)),
            "batch": int(match.group(5)),
            "seconds": float(match.group(6)),
            "throughput_gbps": float(match.group(7)),
            "ops_per_second": float(match.group(8)),
            "bench_p50_ms": float(match.group(9)),
            "bench_p99_ms": float(match.group(10)),
            "bench_max_ms": float(match.group(11)),
            "successful_operations": (
                count - fails if match.group(12) is None else int(match.group(12))
            ),
            "fails": fails,
        }
    if set(phases) != {"put", "get"}:
        raise ValueError("dfkv_bench must emit exactly one PUT and one GET result (use --op both)")
    return phases


def parse_bench_diagnostics(
    text: str, *, required: bool = True
) -> dict[str, dict[str, int]]:
    phases: dict[str, dict[str, int]] = {}
    for raw in text.splitlines():
        match = _BENCH_DIAG.match(raw.strip())
        if not match:
            continue
        values: dict[str, int] = {}
        for token in match.group(2).split():
            key, separator, value = token.partition("=")
            if not separator:
                raise ValueError(f"invalid dfkv_bench diagnostic token: {token!r}")
            try:
                values[key] = int(value)
            except ValueError as error:
                raise ValueError(
                    f"invalid dfkv_bench diagnostic value: {token!r}"
                ) from error
        phases[match.group(1).lower()] = values
    if not phases:
        if not required:
            return {}
        raise ValueError(
            "dfkv_bench must emit one DIAG line for each PUT and GET phase"
        )
    if required and not {"put", "get"}.issubset(phases):
        raise ValueError(
            "dfkv_bench must emit one DIAG line for each PUT and GET phase"
        )
    return phases


def atomic_write(path: str, content: str | bytes, *, mode: int = 0o644) -> None:
    """Atomically replace content, preserving existing mode/owner when present."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = destination.stat()
    except FileNotFoundError:
        existing = None
    target_mode = (
        stat.S_IMODE(existing.st_mode) if existing is not None else mode)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=str(destination.parent))
    try:
        if isinstance(content, bytes):
            stream = os.fdopen(fd, "wb")
        else:
            stream = os.fdopen(fd, "w", encoding="utf-8")
        with stream:
            os.fchmod(stream.fileno(), target_mode)
            if existing is not None:
                os.fchown(stream.fileno(), existing.st_uid, existing.st_gid)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_with_checksum(path: str, content: str) -> None:
    """Write an artifact and a standard SHA-256 sidecar."""
    atomic_write(path, content)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    destination = Path(path)
    atomic_write(
        str(destination) + ".sha256",
        f"{digest}  {destination.name}\n",
    )
