#!/usr/bin/env python3
"""Real v2.44.0-to-candidate per-directory storage activation acceptance.

The three scenarios are a release-produced slab retained by an unset-selector
candidate, a release-produced development LSE deliberately refused by that
candidate without mutation, and a fresh candidate directory selecting LSE and
surviving public-API I/O and a real server reopen.  Fixtures and the CPU-only
control tests are not evidence that any of these native scenarios passed.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import signal
import socket
import stat
import struct
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import uuid
from typing import Callable, Mapping, Optional, Sequence


RELEASE_VERSION = "2.44.0"
RELEASE_SOURCE_COMMIT = "ef851f2617fdf4bf2ae15747550349eb7ee28dfe"
RELEASE_ASSET_NAME = "dfkv-2.44.0-linux-x86_64.tar.gz"
RELEASE_ASSET_URL = (
    "https://github.com/dingodb/DingoCache/releases/download/v2.44.0/"
    + RELEASE_ASSET_NAME
)
RELEASE_ASSET_SHA256 = "da8d736baa5761d5c504110862e9453ff5d0b425be0d13f026693c7d824f945e"
RELEASE_ASSET_BYTES = 12_864_294
RELEASE_FILE_SHA256 = {
    "dfkv_server": "e5f4cb1e7d1c15e06e1fbf4b525e2e507fa39e3b537c25a4fc75c6d78eb21510",
    "dfkv_mds": "a7c1aab276cf6e6e390cb86dab5b74613de12bde42754f633cd1aad702f72d91",
    "libdfkv.so": "90060a29ab65bbc40dca79fdb9f964668a041b7d16becf0f461924cc04acca9b",
}
FORBIDDEN_CURRENT = Path("/root/dfkv-dev/current")
SELECTOR_ENV = ("DFKV_STORE_ENGINE", "DFKV_LSE_EXPERIMENTAL")
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class AcceptanceFailure(RuntimeError):
    """A required native proof is missing, contradictory, or incomplete."""


class CleanupFailure(AcceptanceFailure):
    """An identity-checked owned process did not stop completely."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise AcceptanceFailure(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bytes_new(path: Path, data: bytes) -> None:
    require(not path.exists(), f"refusing to replace evidence: {path}")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def write_json_new(path: Path, value: object) -> None:
    write_bytes_new(
        path,
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(),
    )


def read_log_tail(path: Path, limit: int = 16000) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            return stream.read().decode("utf-8", "replace")
    except FileNotFoundError:
        return "<log absent>"


def is_mnt_disk_path(path: Path) -> bool:
    parts = path.parts
    return len(parts) >= 3 and parts[0] == "/" and parts[1] == "mnt" and bool(
        re.fullmatch(r"disk[^/]+", parts[2])
    )


def resolve_roots(run_root: Path, cache_root: Path) -> tuple[Path, Path]:
    run = run_root.expanduser().resolve(strict=False)
    cache = cache_root.expanduser().resolve(strict=False)
    for role, path in (("run", run), ("cache", cache)):
        require(path != Path(path.anchor), f"{role} root cannot be a filesystem root")
        require(not path.exists(), f"existing {role} root is preserved, refusing: {path}")
        require(FORBIDDEN_CURRENT != path and FORBIDDEN_CURRENT not in path.parents,
                f"current deployment is not a test root: {path}")
    require(not is_mnt_disk_path(run),
            "--run-root is local runtime/evidence only; use --cache-root for /mnt/disk*")
    require(run != cache and run not in cache.parents and cache not in run.parents,
            "run and cache roots must be independent, non-nested fresh roots")
    for path in (run, cache):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir()
    return run, cache


@dataclasses.dataclass(frozen=True)
class RootIdentity:
    path: str
    device: int
    inode: int

    @staticmethod
    def read(path: Path) -> "RootIdentity":
        resolved = path.expanduser().resolve(strict=True)
        info = resolved.stat()
        require(stat.S_ISDIR(info.st_mode), f"store root is not a directory: {resolved}")
        return RootIdentity(str(resolved), info.st_dev, info.st_ino)


@dataclasses.dataclass(frozen=True)
class EntrySnapshot:
    kind: str
    mode: int
    size: int
    sha256: Optional[str] = None
    symlink_target: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class DirectorySnapshot:
    root: RootIdentity
    entries: Mapping[str, EntrySnapshot]
    regular_files_hashed: bool

    def to_dict(self) -> dict:
        return {
            "root": dataclasses.asdict(self.root),
            "regular_files_hashed": self.regular_files_hashed,
            "entries": {name: dataclasses.asdict(entry)
                        for name, entry in sorted(self.entries.items())},
        }


def snapshot_directory(root: Path, *, hash_regular_files: bool = True) -> DirectorySnapshot:
    identity = RootIdentity.read(root)
    base = Path(identity.path)
    entries: dict[str, EntrySnapshot] = {}

    def visit(directory: Path, prefix: str) -> None:
        names = sorted(item.name for item in os.scandir(directory))
        for name in names:
            require(name not in ("", ".", "..") and "/" not in name,
                    f"invalid directory entry under {base}: {name!r}")
            relative = f"{prefix}{name}"
            path = directory / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                entries[relative] = EntrySnapshot(
                    "symlink", stat.S_IMODE(info.st_mode), info.st_size,
                    symlink_target=os.readlink(path))
            elif stat.S_ISDIR(info.st_mode):
                entries[relative] = EntrySnapshot(
                    "directory", stat.S_IMODE(info.st_mode), 0)
                visit(path, relative + "/")
            elif stat.S_ISREG(info.st_mode):
                entries[relative] = EntrySnapshot(
                    "file", stat.S_IMODE(info.st_mode), info.st_size,
                    sha256=sha256_file(path) if hash_regular_files else None)
            else:
                entries[relative] = EntrySnapshot(
                    f"other-{stat.S_IFMT(info.st_mode)}", stat.S_IMODE(info.st_mode),
                    info.st_size)

    visit(base, "")
    return DirectorySnapshot(identity, entries, hash_regular_files)


def require_same_root(before: DirectorySnapshot, after: DirectorySnapshot) -> None:
    require(before.root == after.root,
            f"store root identity changed: {before.root} -> {after.root}")


def require_byte_preservation(before: DirectorySnapshot, after: DirectorySnapshot) -> None:
    require(before.regular_files_hashed and after.regular_files_hashed,
            "byte preservation requires complete regular-file hashes")
    require_same_root(before, after)
    missing = sorted(before.entries.keys() - after.entries.keys())
    added = sorted(after.entries.keys() - before.entries.keys())
    changed = sorted(name for name in before.entries.keys() & after.entries.keys()
                     if before.entries[name] != after.entries[name])
    require(not missing and not added and not changed,
            f"refused startup changed store bytes/entries: missing={missing[:8]} "
            f"added={added[:8]} changed={changed[:8]}")


def require_compatible_root_shape(before: DirectorySnapshot,
                                  after: DirectorySnapshot) -> dict:
    """Slab state may change, but a successful reopen cannot replace the layout."""
    require_same_root(before, after)
    missing = sorted(before.entries.keys() - after.entries.keys())
    added = sorted(after.entries.keys() - before.entries.keys())
    retyped = sorted(name for name in before.entries.keys() & after.entries.keys()
                     if before.entries[name].kind != after.entries[name].kind)
    require(not missing and not added and not retyped,
            f"successful slab reopen changed layout shape: missing={missing[:8]} "
            f"added={added[:8]} retyped={retyped[:8]}")
    changed = sorted(name for name in before.entries.keys() & after.entries.keys()
                     if before.entries[name] != after.entries[name])
    return {"changed_entries_while_retaining_layout": changed}


# ---- Prometheus evidence ---------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MetricSample:
    name: str
    labels: Mapping[str, str]
    value: float


def _split_metric_labels(raw: str) -> dict[str, str]:
    if not raw:
        return {}
    labels: dict[str, str] = {}
    decoder = json.JSONDecoder()
    index = 0
    while index < len(raw):
        match = re.match(r"[a-zA-Z_][a-zA-Z0-9_]*=", raw[index:])
        require(match is not None, f"invalid Prometheus label at {raw[index:]!r}")
        name = match.group(0)[:-1]
        index += len(match.group(0))
        require(index < len(raw) and raw[index] == '"',
                f"Prometheus label {name} is not quoted")
        value, consumed = decoder.raw_decode(raw[index:])
        require(isinstance(value, str), f"Prometheus label {name} is not a string")
        require(name not in labels, f"duplicate Prometheus label: {name}")
        labels[name] = value
        index += consumed
        if index == len(raw):
            break
        require(raw[index] == ",", f"missing Prometheus label comma: {raw[index:]!r}")
        index += 1
    return labels


def parse_prometheus(text: str) -> list[MetricSample]:
    samples: list[MetricSample] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(
            r"([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+([^\s]+)(?:\s+[^\s]+)?",
            line)
        require(match is not None, f"invalid Prometheus sample: {raw_line!r}")
        value = float(match.group(3))
        require(math.isfinite(value), f"non-finite metric value: {raw_line!r}")
        samples.append(MetricSample(match.group(1), _split_metric_labels(match.group(2) or ""),
                                    value))
    return samples


def require_engine_observation(text: str, root: RootIdentity,
                               expected_engine: str) -> dict:
    samples = [sample for sample in parse_prometheus(text)
               if sample.name == "dfkv_disk_engine_info"]
    active = [sample for sample in samples if sample.value == 1.0]
    require(active, "no active dfkv_disk_engine_info sample from the real server")
    observed = [(sample.labels.get("disk"), sample.labels.get("engine"))
                for sample in active]
    require(len(active) == 1, f"single-disk scenario has ambiguous engine evidence: {observed}")
    disk, engine = observed[0]
    require(disk is not None and engine is not None, "engine metric lacks disk/engine labels")
    require(Path(disk) == Path(root.path),
            f"engine metric names a different root: {disk!r} != {root.path!r}")
    require(engine == expected_engine,
            f"actual backend for {root.path} is {engine!r}, expected {expected_engine!r}")
    builds = [sample for sample in parse_prometheus(text)
              if sample.name == "dfkv_build_info" and sample.value == 1.0]
    require(len(builds) == 1, "exactly one real dfkv_build_info sample is required")
    return {"disk": disk, "engine": engine,
            "build": dict(builds[0].labels)}


# ---- Fail-closed trace and verdict controls -------------------------------


def require_open_trace(events: Sequence[str], *, refusal: bool = False) -> None:
    """The harness's snapshot/spawn/open order; injected constructors never qualify."""
    require(events and events[0] == "root_snapshot",
            f"open trace must classify/snapshot before process creation: {events}")
    require("injected_constructor" not in events,
            "an injected engine constructor is not native activation evidence")
    require(events.count("process_spawn") == 1,
            f"open trace needs exactly one real data-server spawn: {events}")
    spawn = events.index("process_spawn")
    require(spawn > 0 and all(event != "engine_observed" for event in events[:spawn]),
            "engine evidence cannot precede the real server open")
    terminal = "startup_refusal" if refusal else "engine_observed"
    wrong = "engine_observed" if refusal else "startup_refusal"
    require(events.count(terminal) == 1 and terminal in events[spawn + 1:],
            f"missing post-open {terminal}: {events}")
    require(wrong not in events, f"contradictory startup outcome: {events}")
    if refusal:
        require("payload_verified" not in events and "client_closed" not in events,
                "a refused server cannot acquire payload/client success evidence")
        require(events[-1] == "root_snapshot",
                f"failed open must end with a preservation snapshot: {events}")
    else:
        require(events[-1] == "process_stopped",
                f"successful server phase must prove stop: {events}")
        require(events.count("payload_verified") == 1,
                f"successful server phase lacks whole-payload proof: {events}")
        require(events.index("payload_verified") > events.index("engine_observed"),
                "payload proof cannot precede actual engine observation")


def verdict_old_slab(release: Mapping[str, object], candidate: Mapping[str, object]) -> str:
    build = release.get("build")
    require(release.get("engine") == "slab" and isinstance(build, Mapping) and
            build.get("version") == RELEASE_VERSION,
            "old slab stage lacks exact live release/engine evidence")
    require(release.get("whole_payload_written") is True and
            release.get("whole_payload_read_by_release") is True and
            release.get("client_closed") is True,
            "release writer did not prove whole payload and native close")
    require(candidate.get("engine") == "slab" and
            candidate.get("whole_payload_read") is True and
            candidate.get("client_closed") is True and
            candidate.get("root_shape_retained") is True,
            "candidate did not retain and read the same slab root")
    return "PASS_RELEASE_SLAB_RETAINED"


def classify_development_refusal(log_text: str, root: Path) -> dict:
    """Require one real refusal event; unrelated lines and prior opens do not qualify."""
    root_text = str(root)
    matches = [line.strip() for line in log_text.splitlines()
               if root_text in line and
               re.search(r"development[ -]format", line, re.IGNORECASE) and
               re.search(r"(?:startup[^\n]*(?:failed|refus)|(?:failed|refus)[^\n]*startup)",
                         line, re.IGNORECASE)]
    successful_listener = bool(
        re.search(r"(?m)^PORT [0-9]+\s*$", log_text) or
        "dfkv_server listening on port" in log_text or
        "dfkv_server /metrics on port" in log_text)
    return {"reason": "development_format" if matches and not successful_listener
                      else "generic_startup_failure",
            "matching_diagnostics": matches,
            "successful_listener_observed": successful_listener}


def verdict_development_refusal(old_control: Mapping[str, object],
                                refusal: Mapping[str, object]) -> str:
    build = old_control.get("build")
    require(old_control.get("engine") == "log" and isinstance(build, Mapping) and
            build.get("version") == RELEASE_VERSION and
            old_control.get("whole_payload_read_by_release") is True and
            old_control.get("client_closed") is True,
            "directory was not positively opened/read by old release LSE")
    require(refusal.get("reason") == "development_format" and
            refusal.get("nonzero_exit") is True and
            refusal.get("health_absent") is True and
            refusal.get("health_or_ready_observed") is False and
            refusal.get("startup_observation_count", 0) > 0 and
            refusal.get("engine_observed") is False and
            refusal.get("root_preserved_byte_for_byte") is True,
            "development data did not fail by the dedicated safe-refusal gate")
    return "EXPECTED_DEVELOPMENT_FORMAT_REFUSAL"


# ---- Release/candidate artifact binding -----------------------------------


def elf_metadata(path: Path) -> dict:
    with path.open("rb") as stream:
        header = stream.read(64)
    require(len(header) == 64 and header[:4] == b"\x7fELF",
            f"native artifact is not ELF64: {path}")
    require(header[4] == 2 and header[5] == 1,
            f"native artifact is not little-endian ELF64: {path}")
    e_type, machine = struct.unpack_from("<HH", header, 16)
    require(machine == 62, f"native artifact is not x86-64 (e_machine={machine}): {path}")
    require(e_type in (2, 3), f"unexpected ELF type {e_type}: {path}")
    return {"class": "ELF64", "endianness": "little", "machine": "x86-64",
            "elf_type": e_type}


def reported_version(executable: Path) -> str:
    completed = subprocess.run(
        [str(executable), "--version"], cwd=str(executable.parent),
        stdin=subprocess.DEVNULL, capture_output=True, timeout=10, check=False)
    text = (completed.stdout + completed.stderr).decode("utf-8", "replace").strip()
    require(completed.returncode == 0 and len(text.splitlines()) == 1,
            f"cannot execute/read {executable} --version: {text}")
    fields = text.split()
    require(len(fields) >= 2 and fields[-1], f"unrecognised version output: {text!r}")
    return fields[-1]


def library_version(library: Path) -> str:
    loaded = ctypes.CDLL(str(library), mode=ctypes.RTLD_LOCAL)
    loaded.dfkv_version.argtypes = []
    loaded.dfkv_version.restype = ctypes.c_char_p
    raw = loaded.dfkv_version()
    require(raw, f"dfkv_version returned no value for {library}")
    return raw.decode("utf-8", "strict")


def _locate(root: Path, name: str, subdirectory: str) -> Path:
    candidates = [root / name, root / subdirectory / name]
    present = [path for path in candidates if path.exists()]
    require(len(present) == 1, f"expected one unambiguous {name} under {root}: {present}")
    resolved = present[0].resolve(strict=True)
    require(resolved.is_file(), f"native artifact is not a file: {resolved}")
    return resolved


@dataclasses.dataclass(frozen=True)
class NativeLayout:
    root: Path
    server: Path
    mds: Path
    library: Path

    @staticmethod
    def resolve(root: Path) -> "NativeLayout":
        resolved = root.expanduser().resolve(strict=True)
        require(resolved.is_dir(), f"native layout is not a directory: {resolved}")
        require(FORBIDDEN_CURRENT != resolved and FORBIDDEN_CURRENT not in resolved.parents,
                f"/root/dfkv-dev/current is never a provenance source: {resolved}")
        return NativeLayout(resolved, _locate(resolved, "dfkv_server", "bin"),
                            _locate(resolved, "dfkv_mds", "bin"),
                            _locate(resolved, "libdfkv.so", "lib"))


def verify_release_layout(root: Path) -> tuple[NativeLayout, dict]:
    layout = NativeLayout.resolve(root)
    version_file = layout.root / "VERSION"
    if version_file.exists():
        require(version_file.read_text().strip() == RELEASE_VERSION,
                f"release VERSION is not {RELEASE_VERSION}: {version_file}")
    files = {"dfkv_server": layout.server, "dfkv_mds": layout.mds,
             "libdfkv.so": layout.library}
    rows = {}
    for name, path in files.items():
        digest = sha256_file(path)
        require(digest == RELEASE_FILE_SHA256[name],
                f"{path} is not the public v2.44.0 {name}: sha256={digest}")
        rows[name] = {"path": str(path), "sha256": digest, "elf": elf_metadata(path)}
    server_version = reported_version(layout.server)
    mds_version = reported_version(layout.mds)
    native_version = library_version(layout.library)
    require(server_version == mds_version == native_version == RELEASE_VERSION,
            f"release version mismatch: server={server_version} mds={mds_version} "
            f"lib={native_version}")
    provenance = {
        "version": RELEASE_VERSION,
        "source_commit": RELEASE_SOURCE_COMMIT,
        "asset_name": RELEASE_ASSET_NAME,
        "asset_url": RELEASE_ASSET_URL,
        "asset_sha256": RELEASE_ASSET_SHA256,
        "asset_bytes": RELEASE_ASSET_BYTES,
        "layout_root": str(layout.root),
        "files": rows,
        "reported_versions": {"dfkv_server": server_version,
                              "dfkv_mds": mds_version, "libdfkv": native_version},
    }
    return layout, provenance


@dataclasses.dataclass(frozen=True)
class CandidateLayout:
    root: Path
    server: Path
    library: Path


def verify_candidate_layout(root: Path, release: NativeLayout) -> tuple[CandidateLayout, dict]:
    resolved = root.expanduser().resolve(strict=True)
    require(resolved.is_dir(), f"candidate layout is not a directory: {resolved}")
    require(FORBIDDEN_CURRENT != resolved and FORBIDDEN_CURRENT not in resolved.parents,
            f"/root/dfkv-dev/current is never a candidate source: {resolved}")
    layout = CandidateLayout(resolved, _locate(resolved, "dfkv_server", "bin"),
                             _locate(resolved, "libdfkv.so", "lib"))
    server_sha = sha256_file(layout.server)
    library_sha = sha256_file(layout.library)
    require(server_sha != sha256_file(release.server),
            "candidate server is byte-identical to v2.44.0, not a final activation candidate")
    server_version = reported_version(layout.server)
    native_version = library_version(layout.library)
    require(server_version == native_version,
            f"candidate server/lib versions differ: {server_version} != {native_version}")
    return layout, {
        "layout_root": str(layout.root),
        "server": {"path": str(layout.server), "sha256": server_sha,
                   "elf": elf_metadata(layout.server), "version": server_version},
        "library": {"path": str(layout.library), "sha256": library_sha,
                    "elf": elf_metadata(layout.library), "version": native_version},
    }


def extract_verified_release(archive: Path, destination: Path) -> Path:
    source = archive.expanduser().resolve(strict=True)
    require(source.name == RELEASE_ASSET_NAME,
            f"release asset filename must be {RELEASE_ASSET_NAME}: {source}")
    require(source.stat().st_size == RELEASE_ASSET_BYTES,
            f"release archive has wrong byte size: {source.stat().st_size}")
    digest = sha256_file(source)
    require(digest == RELEASE_ASSET_SHA256,
            f"release archive checksum mismatch: {digest}")
    require(not destination.exists(), f"release extraction destination exists: {destination}")
    destination.mkdir()
    with tarfile.open(source, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            require(not name.is_absolute() and ".." not in name.parts,
                    f"unsafe release archive path: {member.name}")
            require(member.isdir() or member.isreg() or member.issym() or member.islnk(),
                    f"unsupported release archive member type: {member.name}")
            if member.issym() or member.islnk():
                target = PurePosixPath(member.linkname)
                require(not target.is_absolute() and ".." not in target.parts,
                        f"unsafe release archive link: {member.name} -> {member.linkname}")
        bundle.extractall(destination)
    expected = destination / "dfkv-2.44.0-linux-x86_64"
    require(expected.is_dir(), f"release archive root is absent after extraction: {expected}")
    return expected


# ---- Identity-checked owned processes (same lifecycle model as SGLang E2E) --


@dataclasses.dataclass(frozen=True)
class ProcIdentity:
    pid: int
    ppid: int
    pgid: int
    start_ticks: int
    state: str

    @staticmethod
    def read(pid: int) -> Optional["ProcIdentity"]:
        try:
            text = Path(f"/proc/{pid}/stat").read_text()
            closing = text.rfind(")")
            require(closing > 0, f"malformed /proc/{pid}/stat")
            fields = text[closing + 1:].split()
            require(len(fields) > 19, f"short /proc/{pid}/stat")
            return ProcIdentity(pid, int(fields[1]), int(fields[2]),
                                int(fields[19]), fields[0])
        except (FileNotFoundError, ProcessLookupError):
            return None

    def active(self) -> bool:
        return self.state != "Z"

    def same_process(self, other: Optional["ProcIdentity"]) -> bool:
        return (other is not None and self.pid == other.pid and
                self.start_ticks == other.start_ticks and self.pgid == other.pgid)


def owned_group_id(identity: ProcIdentity) -> int:
    require(identity.pgid > 0 and identity.pgid == identity.pid,
            f"process is not its owned group leader: {identity}")
    return identity.pgid


class OwnedProcess:
    """A new session, plus only descendants observed with pid/start/PGID identity."""

    def __init__(self, name: str, popen: subprocess.Popen, identity: ProcIdentity,
                 argv: Sequence[str], actual_argv: Optional[Sequence[str]], log_path: Path) -> None:
        self.name = name
        self.popen = popen
        self.original_identity = identity
        self.argv = list(argv)
        self.actual_argv = list(actual_argv) if actual_argv is not None else None
        self.log_path = log_path
        self.known = {identity.pid: identity}
        self.expected_stop = False
        self.exit_recorded = False

    @classmethod
    def start(cls, name: str, argv: Sequence[str], *, cwd: Path,
              env: Mapping[str, str], log_path: Path,
              allow_early_exit: bool = False) -> "OwnedProcess":
        require(not log_path.exists(), f"log already exists: {log_path}")
        with log_path.open("xb") as log:
            popen = subprocess.Popen(list(argv), cwd=str(cwd), env=dict(env),
                                     stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     close_fds=True)
        identity = ProcIdentity.read(popen.pid)
        if identity is None:
            popen.wait()
            raise AcceptanceFailure(f"{name} exited before identity could be recorded: "
                                    f"{read_log_tail(log_path)}")
        owned_group_id(identity)
        try:
            raw_argv = Path(f"/proc/{popen.pid}/cmdline").read_bytes()
            actual = [part.decode("utf-8", "surrogateescape")
                      for part in raw_argv.rstrip(b"\0").split(b"\0")]
        except FileNotFoundError:
            if not allow_early_exit or popen.poll() is None:
                popen.wait()
                raise AcceptanceFailure(f"{name} exited before argv could be captured")
            raw_argv = b""
            actual = None
        if allow_early_exit and (not raw_argv or actual == [""]) and popen.poll() is not None:
            actual = None
        require(actual is None or actual == [str(item) for item in argv],
                f"actual argv mismatch for {name}: {actual!r} != {list(argv)!r}")
        return cls(name, popen, identity, [str(item) for item in argv], actual, log_path)

    def command_row(self) -> dict:
        return {"name": self.name, "pid": self.popen.pid,
                "start_ticks": self.original_identity.start_ticks,
                "pgid": self.original_identity.pgid, "argv": self.argv,
                "actual_argv_at_start": self.actual_argv,
                "actual_argv_captured": self.actual_argv is not None,
                "log_path": str(self.log_path)}

    def refresh(self) -> dict[int, ProcIdentity]:
        current: dict[int, ProcIdentity] = {}
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit():
                identity = ProcIdentity.read(int(entry.name))
                if identity is not None:
                    current[identity.pid] = identity
        original = current.get(self.popen.pid)
        owned: dict[int, ProcIdentity] = {}
        if self.original_identity.same_process(original):
            owned[original.pid] = original
        changed = True
        while changed:
            changed = False
            for identity in current.values():
                if identity.pid not in owned and identity.ppid in owned:
                    owned[identity.pid] = identity
                    changed = True
        for pid, old in self.known.items():
            if old.same_process(current.get(pid)):
                owned[pid] = current[pid]
        self.known.update(owned)
        return owned

    def check_running(self) -> None:
        identity = ProcIdentity.read(self.popen.pid)
        code = self.popen.poll()
        if not self.original_identity.same_process(identity) or code is not None:
            raise AcceptanceFailure(
                f"owned process {self.name} exited unexpectedly (code={code}); "
                f"log tail:\n{read_log_tail(self.log_path)}")
        require(identity.active(), f"owned process {self.name} is a zombie")
        self.refresh()

    @staticmethod
    def _signal_checked(identity: ProcIdentity, sig: int) -> None:
        current = ProcIdentity.read(identity.pid)
        if identity.same_process(current) and current.active():
            try:
                os.kill(identity.pid, sig)
            except ProcessLookupError:
                pass

    def _signal_owned(self, sig: int) -> None:
        owned = self.refresh()
        root = owned.get(self.popen.pid)
        if self.original_identity.same_process(root) and root.active():
            try:
                os.killpg(owned_group_id(root), sig)
            except ProcessLookupError:
                pass
        for identity in owned.values():
            self._signal_checked(identity, sig)
        self.popen.poll()

    def _active_identities(self) -> list[ProcIdentity]:
        return [identity for identity in self.refresh().values() if identity.active()]

    def terminate(self, graceful_s: float = 20, kill_s: float = 5) -> dict:
        self.expected_stop = True
        started = time.monotonic()
        try:
            self._signal_owned(signal.SIGTERM)
            while self._active_identities() and time.monotonic() - started < graceful_s:
                time.sleep(0.05)
                self.popen.poll()
            if self._active_identities():
                self._signal_owned(signal.SIGKILL)
                kill_started = time.monotonic()
                while self._active_identities() and time.monotonic() - kill_started < kill_s:
                    time.sleep(0.05)
                    self.popen.poll()
            try:
                code = self.popen.wait(timeout=max(0.1, kill_s))
            except subprocess.TimeoutExpired as exc:
                raise CleanupFailure(f"root process {self.name} could not be reaped") from exc
            remaining = self._active_identities()
            if remaining:
                raise CleanupFailure(f"owned descendants of {self.name} remain active: "
                                     f"{[dataclasses.asdict(item) for item in remaining]}")
            for entry in Path("/proc").iterdir():
                if entry.name.isdigit():
                    identity = ProcIdentity.read(int(entry.name))
                    if (identity is not None and identity.active() and
                            identity.pgid == self.original_identity.pgid):
                        raise CleanupFailure(
                            f"unaccounted process remains in owned PGID {identity.pgid}: "
                            f"pid={identity.pid} start={identity.start_ticks}")
            return {"name": self.name, "pid": self.popen.pid,
                    "start_ticks": self.original_identity.start_ticks,
                    "pgid": self.original_identity.pgid, "status": "terminated",
                    "returncode": code, "elapsed_s": time.monotonic() - started}
        except Exception:
            self.expected_stop = False
            raise

    def record_expected_exit(self, timeout_s: float,
                             poll_observation: Optional[Callable[[], None]] = None) -> dict:
        if poll_observation is not None:
            poll_observation()
        deadline = time.monotonic() + timeout_s
        while self.popen.poll() is None and time.monotonic() < deadline:
            self.refresh()
            if poll_observation is not None:
                poll_observation()
            time.sleep(0.05)
        if self.popen.poll() is None:
            raise AcceptanceFailure(f"expected development-format refusal did not exit: {self.name}")
        code = self.popen.wait(timeout=1)
        if poll_observation is not None:
            poll_observation()
        self.expected_stop = True
        remaining = self._active_identities()
        if remaining:
            self.expected_stop = False
            raise CleanupFailure(f"refused server left owned children: {remaining}")
        return {"name": self.name, "pid": self.popen.pid,
                "start_ticks": self.original_identity.start_ticks,
                "pgid": self.original_identity.pgid, "status": "expected_exit",
                "returncode": code}


class ProcessSupervisor:
    def __init__(self) -> None:
        self.processes: list[OwnedProcess] = []
        self.cleanup: list[dict] = []
        self.cleanup_failures: list[str] = []
        self.unexpected_exits: list[dict] = []

    def add(self, process: OwnedProcess) -> OwnedProcess:
        self.processes.append(process)
        return process

    def check_running(self, process: OwnedProcess) -> None:
        try:
            process.check_running()
        except AcceptanceFailure:
            if not process.exit_recorded:
                process.exit_recorded = True
                self.unexpected_exits.append(
                    {"name": process.name, "pid": process.popen.pid,
                     "returncode": process.popen.poll(), "log_path": str(process.log_path)})
            raise

    def stop(self, process: OwnedProcess, graceful_s: float, kill_s: float) -> None:
        self.cleanup.append(process.terminate(graceful_s, kill_s))

    def stop_all(self, graceful_s: float = 20, kill_s: float = 5) -> None:
        for process in reversed(self.processes):
            if process.expected_stop and any(
                    row["pid"] == process.popen.pid and
                    row["start_ticks"] == process.original_identity.start_ticks
                    for row in self.cleanup):
                continue
            try:
                self.stop(process, graceful_s, kill_s)
            except Exception as exc:
                self.cleanup_failures.append(f"{process.name}: {exc}")


# ---- HTTP, environment, and native public C ABI ----------------------------


def http_bytes(url: str, timeout: float = 1.0) -> tuple[int, bytes]:
    try:
        with HTTP.open(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def wait_http_ready(url: str, process: OwnedProcess, supervisor: ProcessSupervisor,
                    timeout_s: float, *, etcd: bool = False) -> bytes:
    deadline = time.monotonic() + timeout_s
    last = "not contacted"
    while time.monotonic() < deadline:
        supervisor.check_running(process)
        try:
            status, body = http_bytes(url)
            last = f"HTTP {status}: {body[:300]!r}"
            if status == 200:
                if etcd:
                    observed = json.loads(body)
                    require(observed.get("health") == "true", f"etcd is not healthy: {observed}")
                return body
        except (OSError, ValueError, AcceptanceFailure) as exc:
            last = str(exc)
        time.sleep(0.1)
    raise AcceptanceFailure(f"readiness timed out for {url}: {last}; "
                            f"log:\n{read_log_tail(process.log_path)}")


def assert_http_down(url: str) -> None:
    try:
        status, body = http_bytes(url)
    except (OSError, urllib.error.URLError):
        return
    raise CleanupFailure(f"health endpoint still answers HTTP {status}: {url}: {body[:200]!r}")


def engine_env(*, selector: Optional[str] = None) -> dict[str, str]:
    env = dict(os.environ)
    for key in SELECTOR_ENV:
        env.pop(key, None)
    env.update({"DFKV_RDMA": "0", "DFKV_LOG": "INFO"})
    if selector is not None:
        require(selector == "log", "only old development-LSE generation may force a selector")
        env["DFKV_STORE_ENGINE"] = "log"
        env["DFKV_LSE_EXPERIMENTAL"] = "1"
    require("DFKV_STORE_ENGINE" not in env if selector is None else env["DFKV_STORE_ENGINE"] == "log",
            "unexpected engine selector")
    require("DFKV_LSE_EXPERIMENTAL" not in env if selector is None else
            env["DFKV_LSE_EXPERIMENTAL"] == "1", "unexpected experimental flag")
    return env


class ClientOptionsV2(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("members", ctypes.c_char_p), ("key_namespace", ctypes.c_void_p),
        ("key_namespace_len", ctypes.c_uint64), ("batch_concurrency", ctypes.c_uint64),
        ("mds_endpoints", ctypes.c_char_p), ("mds_group", ctypes.c_char_p),
        ("mds_poll_ms", ctypes.c_int32), ("flags", ctypes.c_uint32),
        ("client_id", ctypes.c_char_p), ("client_info", ctypes.c_char_p),
        ("client_heartbeat_ms", ctypes.c_int32), ("reserved0", ctypes.c_uint32),
    ]


class NativeClient:
    """Actual libdfkv C ABI, with binary lifetime retention until each call returns."""

    def __init__(self, library: Path, mds: str, group: str, namespace: bytes) -> None:
        require(namespace, "empty native client namespace")
        self.lib = ctypes.CDLL(str(library), mode=ctypes.RTLD_LOCAL)
        self.lib.dfkv_open_v2.argtypes = [ctypes.POINTER(ClientOptionsV2)]
        self.lib.dfkv_open_v2.restype = ctypes.c_void_p
        self.lib.dfkv_close.argtypes = [ctypes.c_void_p]
        self.lib.dfkv_close.restype = None
        void_pp = ctypes.POINTER(ctypes.c_void_p)
        u64_p = ctypes.POINTER(ctypes.c_uint64)
        int_p = ctypes.POINTER(ctypes.c_int)
        u8_p = ctypes.POINTER(ctypes.c_uint8)
        self.lib.dfkv_put.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64,
                                      ctypes.c_void_p, ctypes.c_uint64]
        self.lib.dfkv_put.restype = ctypes.c_int
        self.lib.dfkv_get_auto.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64,
                                           ctypes.c_void_p, ctypes.c_uint64, u64_p]
        self.lib.dfkv_get_auto.restype = ctypes.c_int
        self.lib.dfkv_exist.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
        self.lib.dfkv_exist.restype = ctypes.c_int
        self.lib.dfkv_remove.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
        self.lib.dfkv_remove.restype = ctypes.c_int
        self.lib.dfkv_batch_put_st.argtypes = [ctypes.c_void_p, void_pp, u64_p, void_pp,
                                               u64_p, ctypes.c_int, int_p, u8_p]
        self.lib.dfkv_batch_put_st.restype = ctypes.c_int
        self.lib.dfkv_batch_get_auto_st.argtypes = [ctypes.c_void_p, void_pp, u64_p, void_pp,
                                                    u64_p, ctypes.c_int, int_p, u64_p, u8_p]
        self.lib.dfkv_batch_get_auto_st.restype = ctypes.c_int
        self.lib.dfkv_batch_remove.argtypes = [ctypes.c_void_p, void_pp, u64_p,
                                               ctypes.c_int, int_p]
        self.lib.dfkv_batch_remove.restype = ctypes.c_int
        namespace_buffer = ctypes.create_string_buffer(namespace, len(namespace))
        options = ClientOptionsV2()
        options.struct_size = ctypes.sizeof(options)
        options.abi_version = 2
        options.key_namespace = ctypes.cast(namespace_buffer, ctypes.c_void_p)
        options.key_namespace_len = len(namespace)
        options.mds_endpoints = mds.encode()
        options.mds_group = group.encode()
        options.mds_poll_ms = 100
        self.handle = self.lib.dfkv_open_v2(ctypes.byref(options))
        require(self.handle, f"actual dfkv_open_v2 failed for {group} at {mds}")

    @staticmethod
    def _buffer(data: bytes) -> ctypes.Array:
        require(len(data) > 0, "native C API requires non-empty keys/values")
        return ctypes.create_string_buffer(data, len(data))

    @staticmethod
    def _void_pointer(buffer: ctypes.Array) -> ctypes.c_void_p:
        return ctypes.cast(buffer, ctypes.c_void_p)

    def close(self) -> None:
        if self.handle:
            handle = self.handle
            self.handle = None
            self.lib.dfkv_close(handle)

    def __enter__(self) -> "NativeClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def put(self, key: bytes, value: bytes) -> None:
        key_buffer, value_buffer = self._buffer(key), self._buffer(value)
        result = self.lib.dfkv_put(self.handle, key_buffer, len(key), value_buffer, len(value))
        require(result == 0, f"public dfkv_put failed for {key!r}: {result}")

    def get(self, key: bytes, expected: bytes) -> None:
        key_buffer = self._buffer(key)
        output = ctypes.create_string_buffer(len(expected) + 97)
        output.raw = b"\xa5" * (len(expected) + 97)
        length = ctypes.c_uint64(0)
        hit = self.lib.dfkv_get_auto(self.handle, key_buffer, len(key), output,
                                     len(output), ctypes.byref(length))
        require(hit == 1 and length.value == len(expected),
                f"public dfkv_get_auto missed/short read for {key!r}: hit={hit} len={length.value}")
        require(output.raw[:length.value] == expected,
                f"whole payload mismatch for {key!r}: sha256={sha256_bytes(output.raw[:length.value])}")

    def batch_put(self, keys: Sequence[bytes], values: Sequence[bytes]) -> None:
        require(len(keys) == len(values) and keys, "invalid native batch put")
        key_buffers = [self._buffer(key) for key in keys]
        value_buffers = [self._buffer(value) for value in values]
        count = len(keys)
        key_ptrs = (ctypes.c_void_p * count)(*[self._void_pointer(item) for item in key_buffers])
        value_ptrs = (ctypes.c_void_p * count)(*[self._void_pointer(item) for item in value_buffers])
        key_lens = (ctypes.c_uint64 * count)(*[len(key) for key in keys])
        value_lens = (ctypes.c_uint64 * count)(*[len(value) for value in values])
        oks = (ctypes.c_int * count)(*([-1] * count))
        statuses = (ctypes.c_uint8 * count)(*([255] * count))
        result = self.lib.dfkv_batch_put_st(self.handle, key_ptrs, key_lens, value_ptrs,
                                            value_lens, count, oks, statuses)
        require(result == 0 and list(oks) == [1] * count and list(statuses) == [0] * count,
                f"public dfkv_batch_put_st failed: result={result} ok={list(oks)} "
                f"status={list(statuses)}")

    def batch_get(self, keys: Sequence[bytes], values: Sequence[bytes]) -> None:
        require(len(keys) == len(values) and keys, "invalid native batch get")
        key_buffers = [self._buffer(key) for key in keys]
        output_buffers = [ctypes.create_string_buffer(len(value) + 31) for value in values]
        for output in output_buffers:
            output.raw = b"\x5a" * len(output)
        count = len(keys)
        key_ptrs = (ctypes.c_void_p * count)(*[self._void_pointer(item) for item in key_buffers])
        output_ptrs = (ctypes.c_void_p * count)(*[self._void_pointer(item) for item in output_buffers])
        key_lens = (ctypes.c_uint64 * count)(*[len(key) for key in keys])
        caps = (ctypes.c_uint64 * count)(*[len(item) for item in output_buffers])
        hits = (ctypes.c_int * count)(*([-1] * count))
        lengths = (ctypes.c_uint64 * count)(*([0] * count))
        statuses = (ctypes.c_uint8 * count)(*([255] * count))
        result = self.lib.dfkv_batch_get_auto_st(self.handle, key_ptrs, key_lens, output_ptrs,
                                                 caps, count, hits, lengths, statuses)
        require(result == 0 and list(hits) == [1] * count and
                list(lengths) == [len(value) for value in values] and
                list(statuses) == [0] * count,
                f"public batch read incomplete: result={result} hits={list(hits)} "
                f"lengths={list(lengths)} status={list(statuses)}")
        for index, expected in enumerate(values):
            actual = output_buffers[index].raw[:len(expected)]
            require(actual == expected, f"batch payload {index} differs")

    def require_absent(self, key: bytes) -> None:
        key_buffer = self._buffer(key)
        require(self.lib.dfkv_exist(self.handle, key_buffer, len(key)) == 0,
                f"removed key still exists: {key!r}")

    def remove(self, key: bytes) -> None:
        key_buffer = self._buffer(key)
        require(self.lib.dfkv_remove(self.handle, key_buffer, len(key)) == 1,
                f"public dfkv_remove failed: {key!r}")

    def batch_remove(self, keys: Sequence[bytes]) -> None:
        require(keys, "empty native batch remove")
        key_buffers = [self._buffer(key) for key in keys]
        count = len(keys)
        key_ptrs = (ctypes.c_void_p * count)(*[self._void_pointer(item) for item in key_buffers])
        key_lens = (ctypes.c_uint64 * count)(*[len(key) for key in keys])
        oks = (ctypes.c_int * count)(*([-1] * count))
        result = self.lib.dfkv_batch_remove(self.handle, key_ptrs, key_lens, count, oks)
        require(result == 0 and list(oks) == [1] * count,
                f"public dfkv_batch_remove failed: result={result} ok={list(oks)}")


# ---- Real isolated ring and scenario orchestration -------------------------


class PortAllocator:
    def __init__(self) -> None:
        self.used: set[int] = set()

    def take(self) -> int:
        for _ in range(128):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            if port not in self.used:
                self.used.add(port)
                return port
        raise AcceptanceFailure("could not allocate fresh loopback endpoints")


@dataclasses.dataclass
class Ring:
    name: str
    local_root: Path
    run_root: Path
    args: argparse.Namespace
    release: NativeLayout
    supervisor: ProcessSupervisor
    result: dict
    sequence: list[str]

    def __post_init__(self) -> None:
        self.group = f"storage-activation-{self.name}-{self.args.run_id}"
        self.node = f"node-{self.name}-{self.args.run_id}"
        self.ports = {role: self.args.ports.take() for role in
                      ("etcd_client", "etcd_peer", "mds", "mds_metrics", "data", "data_metrics")}
        self.mds_endpoint = f"127.0.0.1:{self.ports['mds']}"
        self.data_base = f"http://127.0.0.1:{self.ports['data_metrics']}"
        self.etcd: Optional[OwnedProcess] = None
        self.mds: Optional[OwnedProcess] = None
        self.server: Optional[OwnedProcess] = None
        self.urls: dict[OwnedProcess, str] = {}
        self.metrics_path = self.run_root / "evidence" / f"{self.name}-metrics.prom"

    def _launch(self, role: str, argv: Sequence[str], env: Mapping[str, str],
                *, allow_early_exit: bool = False) -> OwnedProcess:
        process = self.supervisor.add(OwnedProcess.start(
            f"{self.name}-{role}", argv, cwd=self.local_root, env=env,
            log_path=self.local_root / f"{role}.log",
            allow_early_exit=allow_early_exit))
        self.result.setdefault("processes", []).append(process.command_row())
        self.result.setdefault("endpoints", []).append(
            {"ring": self.name, "role": role, "group": self.group,
             "ports": dict(self.ports), "log_path": str(process.log_path)})
        return process

    def start_control(self) -> None:
        env = engine_env()
        client_url = f"http://127.0.0.1:{self.ports['etcd_client']}"
        peer_url = f"http://127.0.0.1:{self.ports['etcd_peer']}"
        token = f"{self.name}-{self.args.run_id}"
        (self.local_root / "etcd-data").mkdir()
        self.etcd = self._launch("etcd", [
            str(self.args.etcd_binary), "--name", token, "--data-dir",
            str(self.local_root / "etcd-data"), "--listen-client-urls", client_url,
            "--advertise-client-urls", client_url, "--listen-peer-urls", peer_url,
            "--initial-advertise-peer-urls", peer_url, "--initial-cluster",
            f"{token}={peer_url}", "--initial-cluster-state", "new",
            "--initial-cluster-token", token], env)
        self.urls[self.etcd] = client_url + "/health"
        wait_http_ready(self.urls[self.etcd], self.etcd, self.supervisor,
                        self.args.startup_timeout_s, etcd=True)
        self.mds = self._launch("dfkv-mds", [
            str(self.release.mds), "--listen", str(self.ports["mds"]), "--etcd",
            f"127.0.0.1:{self.ports['etcd_client']}", "--metrics-port",
            str(self.ports["mds_metrics"]), "--metrics-bind", "127.0.0.1"], env)
        self.urls[self.mds] = f"http://127.0.0.1:{self.ports['mds_metrics']}/healthz"
        wait_http_ready(f"http://127.0.0.1:{self.ports['mds_metrics']}/readyz",
                        self.mds, self.supervisor, self.args.startup_timeout_s)

    def start_server(self, store: Path, binary: Path, *, selector: Optional[str],
                     allow_early_exit: bool = False) -> None:
        self.server_binary_version = reported_version(binary)
        argv = [str(binary), "--dir", str(store), "--port", str(self.ports["data"]),
                "--cap", str(self.args.capacity_bytes), "--id", self.node,
                "--group", self.group, "--mds", self.mds_endpoint, "--advertise",
                f"127.0.0.1:{self.ports['data']}", "--metrics-port",
                str(self.ports["data_metrics"]), "--metrics-bind", "127.0.0.1",
                "--ram-tier", "off", "--log", "INFO"]
        if selector is not None:
            require(selector == "log", "release slab and all candidate opens must be unset")
            argv.extend(["--store-engine", "log"])
        self.server = self._launch("dfkv-data-server", argv, engine_env(selector=selector),
                                   allow_early_exit=allow_early_exit)
        self.urls[self.server] = self.data_base + "/healthz"
        self.sequence.append("process_spawn")
        require("--store-engine" not in argv if selector is None else "--store-engine" in argv,
                f"selector argument contract violated: {argv}")

    def observe(self, root: RootIdentity, expected_engine: str) -> dict:
        require(self.server is not None, "data server was not started")
        wait_http_ready(self.data_base + "/healthz", self.server, self.supervisor,
                        self.args.startup_timeout_s)
        wait_http_ready(self.data_base + "/readyz", self.server, self.supervisor,
                        self.args.startup_timeout_s)
        status, body = http_bytes(self.data_base + "/metrics", timeout=5)
        require(status == 200, f"metrics endpoint failed: HTTP {status}: {body[:300]!r}")
        text = body.decode("utf-8", "strict")
        write_bytes_new(self.metrics_path, body)
        observed = require_engine_observation(text, root, expected_engine)
        require(observed["build"].get("version") == self.server_binary_version,
                f"running server build differs from staged binary: {observed} != "
                f"{self.server_binary_version}")
        self.sequence.append("engine_observed")
        return observed

    def stop_process(self, process: OwnedProcess, *, clean_server: bool = False) -> None:
        self.supervisor.check_running(process)
        self.supervisor.stop(process, self.args.graceful_stop_s, self.args.kill_stop_s)
        url = self.urls.get(process)
        if url:
            assert_http_down(url)
        row = self.supervisor.cleanup[-1]
        if clean_server:
            require(row["returncode"] == 0,
                    f"data server did not close cleanly: {row}; log={read_log_tail(process.log_path)}")
            self.sequence.append("process_stopped")

    def stop_server(self) -> None:
        if self.server is not None:
            self.stop_process(self.server, clean_server=True)

    def stop_control(self) -> None:
        if self.mds is not None:
            self.stop_process(self.mds)
        if self.etcd is not None:
            self.stop_process(self.etcd)

    def stop_all(self) -> None:
        for process in (self.server, self.mds, self.etcd):
            if process is None:
                continue
            already = any(row["pid"] == process.popen.pid and
                          row["start_ticks"] == process.original_identity.start_ticks
                          for row in self.supervisor.cleanup)
            if not already:
                self.supervisor.stop(process, self.args.graceful_stop_s, self.args.kill_stop_s)
            url = self.urls.get(process)
            if url:
                assert_http_down(url)

    def client(self, library: Path, namespace: bytes) -> NativeClient:
        return NativeClient(library, self.mds_endpoint, self.group, namespace)


def payload(label: str, size: int) -> bytes:
    require(size > 32768, "activation payload must exercise whole nontrivial contents")
    seed = hashlib.sha256(label.encode()).digest()
    body = bytes(seed[index % len(seed)] ^ (index * 131 % 251)
                 for index in range(size - 32))
    return body + hashlib.sha256(body).digest()


def make_ring(label: str, args: argparse.Namespace, release: NativeLayout,
              supervisor: ProcessSupervisor, result: dict, sequence: list[str]) -> Ring:
    local = args.run_root / "runtime" / label
    local.mkdir()
    return Ring(label, local, args.run_root, args, release, supervisor, result, sequence)


def run_old_slab(args: argparse.Namespace, release: NativeLayout,
                 candidate: CandidateLayout, supervisor: ProcessSupervisor,
                 result: dict) -> dict:
    store = args.cache_root / "release-slab"
    store.mkdir()
    key = b"activation/release-slab/whole-payload"
    value = payload("old-release-slab", 65_537)
    namespace = f"dfkv/storage-activation/{args.run_id}/slab".encode()
    old_trace: list[str] = []
    old: dict = {"events": old_trace, "key_sha256": sha256_bytes(key),
                 "payload_sha256": sha256_bytes(value), "payload_bytes": len(value),
                 "namespace_sha256": sha256_bytes(namespace)}
    old["initial_root"] = snapshot_directory(store, hash_regular_files=False).to_dict()
    old_trace.append("root_snapshot")
    stage_evidence = {"release": old, "root": str(store)}
    result["active_stage"] = stage_evidence
    ring = make_ring("slab-release", args, release, supervisor, result, old_trace)
    try:
        ring.start_control()
        ring.start_server(store, release.server, selector=None)
        observed = ring.observe(RootIdentity.read(store), "slab")
        old.update(observed)
        require(old["build"].get("version") == RELEASE_VERSION,
                f"running release slab reports wrong build: {observed}")
        with ring.client(release.library, namespace) as client:
            client.put(key, value)
            old["whole_payload_written"] = True
            client.get(key, value)
            old["whole_payload_read_by_release"] = True
            old_trace.append("payload_verified")
        old["client_closed"] = True
        old_trace.append("client_closed")
        ring.stop_server()
        ring.stop_control()
        require_open_trace(old_trace)
        old["verdict"] = "PASS_OLD_RELEASE_POSITIVE_CONTROL"
    finally:
        ring.stop_all()
    before = snapshot_directory(store, hash_regular_files=False)
    write_json_new(args.run_root / "evidence" / "slab-before-candidate.json",
                   before.to_dict())

    candidate_trace: list[str] = ["root_snapshot"]
    new: dict = {"events": candidate_trace, "before_root": before.to_dict()}
    stage_evidence["candidate"] = new
    ring = make_ring("slab-candidate", args, release, supervisor, result, candidate_trace)
    try:
        ring.start_control()
        ring.start_server(store, candidate.server, selector=None)
        new.update(ring.observe(RootIdentity.read(store), "slab"))
        with ring.client(candidate.library, namespace) as client:
            client.get(key, value)
            new["whole_payload_read"] = True
            candidate_trace.append("payload_verified")
        new["client_closed"] = True
        candidate_trace.append("client_closed")
        ring.stop_server()
        ring.stop_control()
        require_open_trace(candidate_trace)
    finally:
        ring.stop_all()
    after = snapshot_directory(store, hash_regular_files=False)
    write_json_new(args.run_root / "evidence" / "slab-after-candidate.json",
                   after.to_dict())
    new["after_root"] = after.to_dict()
    new.update(require_compatible_root_shape(before, after))
    new["root_shape_retained"] = True
    stage_evidence["verdict"] = verdict_old_slab(old, new)
    return stage_evidence


def run_development_refusal(args: argparse.Namespace, release: NativeLayout,
                            candidate: CandidateLayout, supervisor: ProcessSupervisor,
                            result: dict) -> dict:
    store = args.cache_root / "release-development-lse"
    store.mkdir()
    key = b"activation/release-development-lse/whole-payload"
    value = payload("old-development-lse", 65_681)
    namespace = f"dfkv/storage-activation/{args.run_id}/development".encode()
    old_trace: list[str] = ["root_snapshot"]
    old: dict = {"events": old_trace, "payload_sha256": sha256_bytes(value),
                 "payload_bytes": len(value),
                 "initial_root": snapshot_directory(store, hash_regular_files=False).to_dict()}
    stage_evidence = {"old_positive_control": old, "root": str(store)}
    result["active_stage"] = stage_evidence
    ring = make_ring("development-release", args, release, supervisor, result, old_trace)
    try:
        ring.start_control()
        ring.start_server(store, release.server, selector="log")
        old.update(ring.observe(RootIdentity.read(store), "log"))
        require(old["build"].get("version") == RELEASE_VERSION,
                f"running development control reports wrong build: {old}")
        with ring.client(release.library, namespace) as client:
            client.put(key, value)
            client.get(key, value)
            old["whole_payload_read_by_release"] = True
            old_trace.append("payload_verified")
        old["client_closed"] = True
        old_trace.append("client_closed")
        ring.stop_server()
        ring.stop_control()
        require_open_trace(old_trace)
        old["verdict"] = "PASS_OLD_DEVELOPMENT_LSE_POSITIVE_CONTROL"
    finally:
        ring.stop_all()
    before = snapshot_directory(store, hash_regular_files=True)
    write_json_new(args.run_root / "evidence" / "development-before-candidate.json",
                   before.to_dict())

    refusal_trace: list[str] = ["root_snapshot"]
    refusal: dict = {"events": refusal_trace, "startup_observation_count": 0,
                     "health_or_ready_observed": False, "engine_observed": None,
                     "before_root": before.to_dict()}
    stage_evidence["candidate_refusal"] = refusal
    ring = make_ring("development-candidate-refusal", args, release, supervisor,
                     result, refusal_trace)
    try:
        ring.start_control()
        ring.start_server(store, candidate.server, selector=None, allow_early_exit=True)
        assert ring.server is not None

        def reject_any_open_interval() -> None:
            refusal["startup_observation_count"] += 1
            for endpoint in (ring.data_base + "/healthz", ring.data_base + "/readyz"):
                try:
                    status, body = http_bytes(endpoint)
                except (OSError, urllib.error.URLError):
                    continue
                if status == 200:
                    refusal["health_or_ready_observed"] = True
                    raise AcceptanceFailure(
                        f"development candidate became healthy/ready before exit: "
                        f"{endpoint}: {body[:300]!r}")
            try:
                status, body = http_bytes(ring.data_base + "/metrics")
            except (OSError, urllib.error.URLError):
                return
            if status == 200:
                active_engines = [sample for sample in parse_prometheus(
                    body.decode("utf-8", "strict"))
                    if sample.name == "dfkv_disk_engine_info" and sample.value == 1.0]
                if active_engines:
                    refusal["engine_observed"] = True
                    raise AcceptanceFailure(
                        "development candidate exposed a selected engine before exit: "
                        f"{active_engines}")

        exit_row = ring.server.record_expected_exit(
            args.startup_timeout_s, reject_any_open_interval)
        refusal["engine_observed"] = refusal["engine_observed"] is True
        supervisor.cleanup.append(exit_row)
        refusal_trace.append("startup_refusal")
        refusal["nonzero_exit"] = exit_row["returncode"] != 0
        after = snapshot_directory(store, hash_regular_files=True)
        write_json_new(args.run_root / "evidence" / "development-after-candidate.json",
                       after.to_dict())
        refusal_trace.append("root_snapshot")
        refusal["after_root"] = after.to_dict()
        require(refusal["nonzero_exit"], f"development data started instead of refusing: {exit_row}")
        log_text = read_log_tail(ring.server.log_path, limit=1_000_000)
        refusal.update(classify_development_refusal(log_text, store))
        require(refusal["reason"] == "development_format",
                "generic startup failure/prior open is not a development-format verdict; "
                "log:\n" + log_text)
        assert_http_down(ring.data_base + "/healthz")
        assert_http_down(ring.data_base + "/metrics")
        refusal["health_absent"] = True
        require_byte_preservation(before, after)
        refusal["root_preserved_byte_for_byte"] = True
        ring.stop_control()
        require_open_trace(refusal_trace, refusal=True)
    finally:
        ring.stop_all()
    stage_evidence["verdict"] = verdict_development_refusal(old, refusal)
    return stage_evidence


def fresh_public_api_phase(ring: Ring, store: Path, library: Path, namespace: str,
                           trace: list[str], *, reopen: bool) -> dict:
    scalar_key = b"activation/fresh/scalar-direct"
    batch_keys = [b"activation/fresh/batch-0", b"activation/fresh/batch-1"]
    scalar_value = payload("fresh-scalar-direct", 65_699)
    batch_values = [payload("fresh-batch-0", 36_869), payload("fresh-batch-1", 49_171)]
    observed = ring.observe(RootIdentity.read(store), "log")
    with ring.client(library, namespace.encode()) as client:
        if not reopen:
            client.put(scalar_key, scalar_value)
            client.batch_put(batch_keys, batch_values)
        client.get(scalar_key, scalar_value)
        client.batch_get(batch_keys, batch_values)
        if reopen:
            client.remove(scalar_key)
            client.require_absent(scalar_key)
            client.batch_remove(batch_keys)
            for batch_key in batch_keys:
                client.require_absent(batch_key)
    trace.append("payload_verified")
    return {"engine_observation": observed, "scalar_direct_bytes": len(scalar_value),
            "scalar_direct_sha256": sha256_bytes(scalar_value),
            "batch_payloads": [{"key_sha256": sha256_bytes(key), "bytes": len(value),
                                "sha256": sha256_bytes(value)}
                               for key, value in zip(batch_keys, batch_values)],
            "operations": "whole payload reads" + (", scalar/batch remove" if reopen else
                                                   ", scalar direct-disk put + batch put")}


def run_fresh_default(args: argparse.Namespace, release: NativeLayout,
                      candidate: CandidateLayout, supervisor: ProcessSupervisor,
                      result: dict) -> dict:
    store = args.cache_root / "fresh-default-lse"
    store.mkdir()
    namespace = f"dfkv/storage-activation/{args.run_id}/fresh"
    stage_evidence = {"root": str(store),
                      "initial_root": snapshot_directory(store, hash_regular_files=False).to_dict()}
    result["active_stage"] = stage_evidence
    first_trace: list[str] = ["root_snapshot"]
    ring = make_ring("fresh-candidate-write", args, release, supervisor, result, first_trace)
    try:
        ring.start_control()
        ring.start_server(store, candidate.server, selector=None)
        first = fresh_public_api_phase(ring, store, candidate.library, namespace,
                                       first_trace, reopen=False)
        stage_evidence["write"] = first
        first_trace.append("client_closed")
        ring.stop_server()
        ring.stop_control()
        require_open_trace(first_trace)
    finally:
        ring.stop_all()

    stage_evidence["before_reopen_root"] = snapshot_directory(
        store, hash_regular_files=False).to_dict()
    second_trace: list[str] = ["root_snapshot"]
    ring = make_ring("fresh-candidate-reopen", args, release, supervisor, result, second_trace)
    try:
        ring.start_control()
        ring.start_server(store, candidate.server, selector=None)
        second = fresh_public_api_phase(ring, store, candidate.library, namespace,
                                        second_trace, reopen=True)
        stage_evidence["reopen_remove"] = second
        second_trace.append("client_closed")
        ring.stop_server()
        ring.stop_control()
        require_open_trace(second_trace)
    finally:
        ring.stop_all()
    stage_evidence.update({
        "verdict": "PASS_FRESH_DEFAULT_LSE_PUBLIC_API_REOPEN_REMOVE",
        "root_identity_after_close": dataclasses.asdict(RootIdentity.read(store))})
    return stage_evidence


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--release-archive", type=Path,
                        help="exact unmodified public dfkv-2.44.0-linux-x86_64.tar.gz")
    source.add_argument("--release-dir", type=Path,
                        help="extracted canonical or flat directory; exact file hashes are required")
    parser.add_argument("--candidate-build-dir", type=Path, required=True,
                        help="candidate build root or bin/lib layout (never /root/dfkv-dev/current)")
    parser.add_argument("--etcd-binary", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True,
                        help="fresh local evidence/runtime root; must not be on /mnt/disk*")
    parser.add_argument("--cache-root", type=Path, required=True,
                        help="fresh store root on the caller-reserved test filesystem; children are preserved")
    parser.add_argument("--capacity-bytes", type=int, default=1 << 30)
    parser.add_argument("--startup-timeout-s", type=float, default=45)
    parser.add_argument("--graceful-stop-s", type=float, default=20)
    parser.add_argument("--kill-stop-s", type=float, default=5)
    return parser.parse_args(argv)


def execute(args: argparse.Namespace, supervisor: ProcessSupervisor, result: dict) -> dict:
    require(platform.system() == "Linux" and platform.machine() == "x86_64",
            "the public release binding is Linux x86-64 only")
    require(args.capacity_bytes > 0 and args.startup_timeout_s > 0 and
            args.graceful_stop_s > 0 and args.kill_stop_s > 0, "invalid capacities/timeouts")
    args.run_id = uuid.uuid4().hex
    args.ports = PortAllocator()
    run_root, cache_root = resolve_roots(args.run_root, args.cache_root)
    args.run_root = Path(run_root)
    args.cache_root = Path(cache_root)
    args.owned_run_root = args.run_root
    (args.run_root / "runtime").mkdir()
    (args.run_root / "evidence").mkdir()
    args.etcd_binary = args.etcd_binary.expanduser().resolve(strict=True)
    require(args.etcd_binary.is_file(), f"etcd is not a file: {args.etcd_binary}")
    if args.release_archive is not None:
        release_root = extract_verified_release(args.release_archive, args.run_root / "release")
    else:
        release_root = args.release_dir
    release, release_proof = verify_release_layout(release_root)
    candidate, candidate_proof = verify_candidate_layout(args.candidate_build_dir, release)
    etcd_version = subprocess.run([str(args.etcd_binary), "--version"],
                                  capture_output=True, timeout=10, check=False)
    require(etcd_version.returncode == 0, "selected etcd cannot report its version")
    result.update({
        "schema": "dfkv-storage-activation-e2e-v1",
        "status": "INCOMPLETE",
        "run_id": args.run_id,
        "policy": {"release_slab": "write/read/close by exact old release, then candidate unset reopen",
                   "development_lse": "old positive control, then dedicated candidate refusal and byte preservation",
                   "fresh_default": "candidate unset selects log; public API read/reopen/remove",
                   "engine_authority": "dfkv_disk_engine_info at canonical root; never filenames or selector intent",
                   "roots": "fresh exclusive run_root/cache_root; never clear or replace"},
        "run_root": str(args.run_root), "cache_root": str(args.cache_root),
        "release": release_proof, "candidate": candidate_proof,
        "etcd": {"path": str(args.etcd_binary), "sha256": sha256_file(args.etcd_binary),
                 "version_output": (etcd_version.stdout + etcd_version.stderr).decode(
                     "utf-8", "replace").strip()},
        "processes": [], "endpoints": [],
    })
    write_json_new(args.run_root / "evidence" / "release-provenance.json", release_proof)
    result["stages"] = {}
    result["stages"]["release_slab"] = run_old_slab(
        args, release, candidate, supervisor, result)
    result["stages"]["development_lse"] = run_development_refusal(
        args, release, candidate, supervisor, result)
    result["stages"]["fresh_default"] = run_fresh_default(
        args, release, candidate, supervisor, result)
    result["status"] = "ACCEPTANCE_PASS_PENDING_CLEANUP"
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    supervisor = ProcessSupervisor()
    result: dict = {"schema": "dfkv-storage-activation-e2e-v1", "status": "HARNESS_ERROR"}
    exit_code = 3
    try:
        result = execute(args, supervisor, result)
        exit_code = 0
    except CleanupFailure as exc:
        result.setdefault("errors", []).append(str(exc))
        result["status"] = "CLEANUP_FAILED"
        exit_code = 4
    except AcceptanceFailure as exc:
        result.setdefault("errors", []).append(str(exc))
        result["status"] = "ACCEPTANCE_FAIL"
        exit_code = 1
    except Exception as exc:
        result.setdefault("errors", []).append(f"{type(exc).__name__}: {exc}")
        result["status"] = "HARNESS_ERROR"
        exit_code = 3
    finally:
        supervisor.stop_all(args.graceful_stop_s, args.kill_stop_s)
    result["cleanup"] = supervisor.cleanup
    result["cleanup_failures"] = supervisor.cleanup_failures
    result["unexpected_exits"] = supervisor.unexpected_exits
    if supervisor.cleanup_failures or supervisor.unexpected_exits:
        result["status"] = "CLEANUP_FAILED"
        result.setdefault("errors", []).append("owned process cleanup did not complete")
        exit_code = 4
    elif result["status"] == "ACCEPTANCE_PASS_PENDING_CLEANUP":
        result["status"] = "ACCEPTANCE_PASS"
        exit_code = 0
    owned_run_root = getattr(args, "owned_run_root", None)
    result_path = (owned_run_root / "result.json"
                   if isinstance(owned_run_root, Path) else None)
    require(result_path is None or not result_path.exists(),
            f"refusing to replace this run's result evidence: {result_path}")
    if result_path is not None:
        write_json_new(result_path, result)
    print(json.dumps({"status": result["status"], "exit_code": exit_code,
                      "result_path": str(result_path) if result_path is not None else None},
                     sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
