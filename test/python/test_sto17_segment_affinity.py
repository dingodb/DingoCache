#!/usr/bin/env python3
"""STO-17 segment-affinity placement/read acceptance and warm-read evidence.

Usage: test_sto17_segment_affinity.py BUILD_DIR OUTPUT_ROOT [--smoke]

Starts one real etcd, MDS and three loopback TCP file-engine servers; uses only
stdlib, the real libdfkv C ABI and this repository's CLI-35 process helpers.
OUTPUT_ROOT must be absent or empty and all runtime data/logs are retained.
A full successful run returns status ACCEPTANCE in result.json and exit 0;
--smoke runs the same correctness checks with fewer samples and is never
acceptance.  No latency threshold is a correctness gate.
"""

from __future__ import annotations

import argparse
import ctypes
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any
from urllib.request import urlopen

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "integration" / "common" / "src"))
sys.path.insert(0, str(REPO / "test" / "python"))

import test_dfkv_cli35_case5 as c35  # noqa: E402
from dfkv_common.cabi import make_key_array, make_route_key_array  # noqa: E402

GROUP = "sto17"
NAMESPACE = b"dfkv/sto17/segment-affinity/v1"
NODE_IDS = ("n1", "n2", "n3")
CHAIN_IDS = ("a", "b")
OBJECTS_PER_CHAIN = 64
SEGMENT_SIZES = (16, 64)
VALUE_BYTES = 4096
CAP_BYTES = 64 << 20
LOCATE_CANDIDATE_LIMIT = 4096
ENV_KEYS = (
    "DFKV_RING_ABSENT_GRACE_MS", "DFKV_RING_READBACK_MS",
    "DFKV_RING_READBACK_REWRITE", "DFKV_RING_COOLDOWN_REROUTE",
    "DFKV_RING_REMOVE_FANOUT", "DFKV_RING_REROUTE_KEYS",
    "DFKV_CLIENT_NODE_DEDUP", "DFKV_CLIENT_NODE_DEDUP_GPU", "DFKV_RDMA",
    "DFKV_OPEN_RING_WAIT_MS", "DFKV_GET_MISS_RETRIES", "DFKV_LOG",
    "DFKV_ACCESS_DIR",
)


def _key_text(key: bytes) -> str:
    return key.decode("ascii")


def _flat(chains: dict[str, list[bytes]]) -> list[bytes]:
    return [key for chain_id in CHAIN_IDS for key in chains[chain_id]]


def _values(chains: dict[str, list[bytes]]) -> dict[bytes, bytes]:
    answer = {}
    for key in _flat(chains):
        header = b"DFKV-STO17-SEGMENT-AFFINITY-V1\0" + key + b"\0"
        seed = hashlib.sha256(b"dfkv-sto17-value-v1\0" + key).digest()
        answer[key] = (header + seed * 160)[:VALUE_BYTES]
        if len(answer[key]) != VALUE_BYTES:
            raise AssertionError("value construction length")
    return answer


def _identity(chains: dict[str, list[bytes]], values: dict[bytes, bytes]) -> str:
    digest = hashlib.sha256()
    for chain_id in CHAIN_IDS:
        for index, key in enumerate(chains[chain_id]):
            for item in (chain_id.encode(), str(index).encode(), key, values[key]):
                digest.update(len(item).to_bytes(8, "big"))
                digest.update(item)
    return digest.hexdigest()


def _routes(chains: dict[str, list[bytes]], segment_size: int) -> list[bytes]:
    return [
        chains[chain_id][(index // segment_size) * segment_size]
        for chain_id in CHAIN_IDS
        for index, keys in enumerate(chains[chain_id])
    ]


def _percentile(ordered: list[int], percentile: int) -> int:
    return ordered[max(0, math.ceil(percentile * len(ordered) / 100) - 1)]


def _timing_summary(samples: list[int]) -> dict[str, Any]:
    ordered = sorted(samples)
    return {
        "available": bool(samples), "count": len(samples),
        "min_ns": ordered[0], "p50_ns": _percentile(ordered, 50),
        "p95_ns": _percentile(ordered, 95), "p99_ns": _percentile(ordered, 99),
        "max_ns": ordered[-1], "mean_ns": sum(samples) / len(samples),
        "samples_ns": samples,
        "quantile_method": "nearest-rank, ceil(p*N/100)-1",
    }


def _git_head() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True,
        capture_output=True, timeout=10, check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
        text=True, capture_output=True, timeout=10, check=True,
    ).stdout
    return {"head": head, "tracked_worktree_dirty": bool(status), "status": status}


def _load_library(path: Path) -> ctypes.CDLL:
    lib = c35.load_library(path)
    pointer = ctypes.POINTER
    anchored_argtypes = {
        "dfkv_batch_put_anchored": [
            ctypes.c_void_p, pointer(ctypes.c_void_p), pointer(ctypes.c_uint64),
            pointer(ctypes.c_void_p), pointer(ctypes.c_uint64),
            pointer(ctypes.c_void_p), pointer(ctypes.c_uint64), ctypes.c_int,
            pointer(ctypes.c_int),
        ],
        "dfkv_batch_get_auto_anchored": [
            ctypes.c_void_p, pointer(ctypes.c_void_p), pointer(ctypes.c_uint64),
            pointer(ctypes.c_void_p), pointer(ctypes.c_uint64),
            pointer(ctypes.c_void_p), pointer(ctypes.c_uint64), ctypes.c_int,
            pointer(ctypes.c_int), pointer(ctypes.c_uint64),
        ],
        "dfkv_batch_remove_anchored": [
            ctypes.c_void_p, pointer(ctypes.c_void_p), pointer(ctypes.c_uint64),
            pointer(ctypes.c_void_p), pointer(ctypes.c_uint64), ctypes.c_int,
            pointer(ctypes.c_int),
        ],
    }
    for name, argtypes in anchored_argtypes.items():
        function = getattr(lib, name, None)
        if function is None:
            raise RuntimeError(f"{path} lacks required anchored C ABI: {name}")
        function.restype = ctypes.c_int
        function.argtypes = argtypes
    lib.dfkv_batch_put.restype = ctypes.c_int
    lib.dfkv_batch_put.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.POINTER(ctypes.c_int),
    ]
    lib.dfkv_batch_get_auto.restype = ctypes.c_int
    lib.dfkv_batch_get_auto.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint64),
    ]
    return lib


class NativeClient:
    def __init__(self, lib: ctypes.CDLL, client_id: str, *, mds: str = "",
                 members: str = "") -> None:
        self.lib = lib
        self.client_id = client_id
        options = c35.make_client_options_v2(
            NAMESPACE, members=members, mds_endpoints=mds, mds_group=GROUP,
            mds_poll_ms=200, register_client=False, client_id=client_id,
            client_info="type=sto17-segment-affinity;direct-probe=" +
                        ("1" if members else "0"),
            batch_concurrency=8,
        )
        self._options = options
        self.handle = lib.dfkv_open_v2(ctypes.byref(options))
        if not self.handle:
            raise RuntimeError(f"dfkv_open_v2 failed: {client_id}")
        mode = lib.dfkv_transport_mode(self.handle)
        self.transport = mode.decode(errors="replace") if mode else ""
        if not self.transport.startswith("tcp"):
            self.close()
            raise RuntimeError(f"{client_id}: TCP required, got {self.transport!r}")

    def close(self) -> None:
        if self.handle:
            self.lib.dfkv_close(self.handle)
            self.handle = None


def _route_arrays(keys: list[bytes], mode: str,
                  anchors: list[bytes] | None) -> tuple[Any, Any, list[Any]]:
    if mode == "null":
        return None, None, []
    if mode == "empty":
        return make_route_key_array([b""] * len(keys))
    if mode == "nonempty":
        if anchors is None or len(anchors) != len(keys) or any(not x for x in anchors):
            raise ValueError("nonempty mode needs one non-empty actual segment-first key/item")
        return make_route_key_array(anchors)
    raise ValueError(mode)


def _put(client: NativeClient, keys: list[bytes], values: dict[bytes, bytes], mode: str,
         anchors: list[bytes] | None = None) -> None:
    n = len(keys)
    karr, klens, key_owners = make_key_array(keys)
    value_blob = ctypes.create_string_buffer(b"".join(values[key] for key in keys),
                                              n * VALUE_BYTES + 1)
    base = ctypes.addressof(value_blob)
    pointers = (ctypes.c_void_p * n)(*[base + i * VALUE_BYTES for i in range(n)])
    sizes = (ctypes.c_uint64 * n)(*([VALUE_BYTES] * n))
    output = (ctypes.c_int * n)()
    if mode == "legacy":
        rc = client.lib.dfkv_batch_put(
            client.handle, karr, klens, pointers, sizes, n, output)
    else:
        rarr, rlens, route_owners = _route_arrays(keys, mode, anchors)
        rc = client.lib.dfkv_batch_put_anchored(
            client.handle, karr, klens, rarr, rlens, pointers, sizes, n, output)
        del route_owners
    if rc != 0 or list(output) != [1] * n:
        raise RuntimeError(f"{mode} PUT failed: rc={rc}, out={list(output)}")
    del key_owners, value_blob


def _get(client: NativeClient, keys: list[bytes], mode: str,
         anchors: list[bytes] | None = None) -> dict[str, list[Any]]:
    n = len(keys)
    karr, klens, key_owners = make_key_array(keys)
    storage = ctypes.create_string_buffer(n * VALUE_BYTES + 1)
    base = ctypes.addressof(storage)
    pointers = (ctypes.c_void_p * n)(*[base + i * VALUE_BYTES for i in range(n)])
    caps = (ctypes.c_uint64 * n)(*([VALUE_BYTES] * n))
    hits = (ctypes.c_int * n)()
    lengths = (ctypes.c_uint64 * n)()
    if mode == "legacy":
        rc = client.lib.dfkv_batch_get_auto(
            client.handle, karr, klens, pointers, caps, n, hits, lengths)
    else:
        rarr, rlens, route_owners = _route_arrays(keys, mode, anchors)
        rc = client.lib.dfkv_batch_get_auto_anchored(
            client.handle, karr, klens, rarr, rlens, pointers, caps, n,
            hits, lengths)
        del route_owners
    if rc != 0:
        raise RuntimeError(f"{mode} GET failed: rc={rc}")
    result: dict[str, list[Any]] = {"hits": list(hits), "lengths": list(lengths),
                                    "values": []}
    for i in range(n):
        hit = hits[i] == 1
        length = int(lengths[i])
        result["values"].append(
            ctypes.string_at(base + i * VALUE_BYTES, length)
            if hit and length <= VALUE_BYTES else None)
    del key_owners, storage
    return result


def _remove(client: NativeClient, keys: list[bytes], mode: str,
            anchors: list[bytes] | None = None) -> None:
    n = len(keys)
    karr, klens, key_owners = make_key_array(keys)
    output = (ctypes.c_int * n)()
    if mode == "legacy":
        rc = client.lib.dfkv_batch_remove(client.handle, karr, klens, n, output)
    else:
        rarr, rlens, route_owners = _route_arrays(keys, mode, anchors)
        rc = client.lib.dfkv_batch_remove_anchored(
            client.handle, karr, klens, rarr, rlens, n, output)
        del route_owners
    if rc != 0 or list(output) != [1] * n:
        raise RuntimeError(f"{mode} REMOVE failed: rc={rc}, out={list(output)}")
    del key_owners


def _read_check(keys: list[bytes], result: dict[str, list[Any]],
                values: dict[bytes, bytes],
                expected_hits: list[bool] | None = None) -> dict[str, Any]:
    wrong_bytes, unexpected_hits, unexpected_misses = [], [], []
    for i, key in enumerate(keys):
        hit = result["hits"][i] == 1
        value = result["values"][i]
        if hit and value != values[key]:
            wrong_bytes.append(_key_text(key))
        if expected_hits is not None:
            if hit and not expected_hits[i]:
                unexpected_hits.append(_key_text(key))
            if expected_hits[i] and not hit:
                unexpected_misses.append(_key_text(key))
    return {
        "request_keys": len(keys), "hit_count": sum(result["hits"]),
        "miss_count": len(keys) - sum(result["hits"]),
        "wrong_byte_count": len(wrong_bytes), "wrong_byte_keys": wrong_bytes,
        "unexpected_hit_count": len(unexpected_hits),
        "unexpected_hit_keys": unexpected_hits,
        "unexpected_miss_count": len(unexpected_misses),
        "unexpected_miss_keys": unexpected_misses,
    }


class Ring:
    def __init__(self, args: argparse.Namespace, etcd: Path, root: Path,
                 events: c35.Jsonl) -> None:
        self.args = args
        self.etcd_binary = etcd
        self.root = root
        self.events = events
        self.processes = c35.Processes(root, events)
        self.core_children: list[c35.Child] = []
        self.etcd_endpoint = ""
        self.mds_endpoint = ""
        self.nodes: dict[str, c35.Node] = {}

    def _event(self, event: str, **fields: Any) -> None:
        self.events.write({"event": event, "wall_time": c35._iso(),
                           "monotonic_ns": time.monotonic_ns(), **fields})

    def start(self) -> None:
        client_port, peer_port = c35._free_port(), c35._free_port()
        client_url = f"http://127.0.0.1:{client_port}"
        peer_url = f"http://127.0.0.1:{peer_port}"
        data = self.root / "runtime" / "etcd"
        data.mkdir(parents=True)
        child = self.processes.launch("etcd", [
            str(self.etcd_binary), "--data-dir", str(data), "--name", "sto17",
            "--listen-client-urls", client_url, "--advertise-client-urls", client_url,
            "--listen-peer-urls", peer_url, "--initial-advertise-peer-urls", peer_url,
            "--initial-cluster", f"sto17={peer_url}",
        ], dict(os.environ))
        self.core_children.append(child)
        self.etcd_endpoint = f"127.0.0.1:{client_port}"

        def healthy() -> bool:
            if child.popen.poll() is not None:
                raise RuntimeError(f"etcd exited {child.popen.returncode}; log={child.log_path}")
            try:
                with urlopen(client_url + "/health", timeout=0.5) as response:
                    health = json.loads(response.read())
                return health.get("health") == "true"
            except Exception:
                return False

        c35._until("real etcd /health", healthy, 30.0)
        self._event("etcd_healthy", endpoint=self.etcd_endpoint)
        mds = self.processes.launch("mds", [
            str(self.args.mds), "--listen", "0", "--etcd", self.etcd_endpoint,
            "--etcd-probe-ms", "10000", "--metrics-port", "0",
            "--metrics-bind", "127.0.0.1",
        ], {**os.environ, "DFKV_LOG": "INFO"})
        self.core_children.append(mds)
        match = self.processes.wait_log(mds, r"^dfkv_mds listening on (\d+),")
        self.mds_endpoint = f"127.0.0.1:{int(match.group(1))}"
        self._event("mds_ready", endpoint=self.mds_endpoint)
        for node_id in NODE_IDS:
            port = c35._free_port()
            node_data = self.root / "runtime" / node_id
            node_data.mkdir(parents=True)
            server = self.processes.launch(node_id, [
                str(self.args.server), "--dir", str(node_data), "--port", str(port),
                "--cap", str(CAP_BYTES), "--store-engine", "file",
                "--mds", self.mds_endpoint, "--group", GROUP, "--id", node_id,
                "--advertise", f"127.0.0.1:{port}", "--metrics-port", "0",
                "--metrics-bind", "127.0.0.1", "--mds-registration-timeout-ms", "10000",
            ], {
                **os.environ, "DFKV_LOG": "INFO", "DFKV_RDMA": "0",
                "DFKV_MDS_HEARTBEAT_MS": "200", "DFKV_WRITE_MEMORY_ENTRIES": "0",
                "DFKV_ACCESS_DIR": str(self.root / "server-access"),
            })
            self.core_children.append(server)
            observed = self.processes.wait_log(server, r"^PORT (\d+)\s*$")
            if int(observed.group(1)) != port:
                raise RuntimeError(f"{node_id}: unexpected listening port")
            metrics = self.processes.wait_log(server, r"dfkv_server /metrics on port (\d+)")
            self.nodes[node_id] = c35.Node(
                node_id, 0, "current", self.args.server, server, node_data,
                port, int(metrics.group(1)))
            self._event("file_server_ready", node=node_id, endpoint=f"127.0.0.1:{port}")
        self.membership = c35._until(
            "three eligible MDS members", self._eligible_ring, 40.0)
        self._event("ring_ready", membership=self.membership)

    def _eligible_ring(self) -> dict[str, Any] | None:
        try:
            value = self.ring()
            if len(value["members"]) == 3 and all(x["eligible"] for x in value["members"]):
                return value
        except Exception:
            pass
        return None

    def ring(self) -> dict[str, Any]:
        completed = subprocess.run([
            str(self.args.ctl), "ring", "--json", "--allow-empty", "--group", GROUP,
            "--mds", self.mds_endpoint,
        ], text=True, capture_output=True, timeout=8, check=True)
        return json.loads(completed.stdout)

    def locate(self, key: bytes) -> str:
        completed = subprocess.run([
            str(self.args.ctl), "locate", _key_text(key), "--group", GROUP,
            "--mds", self.mds_endpoint, "--ring-wait-ms", "10000",
        ], text=True, capture_output=True, timeout=12, check=True)
        reported_endpoint = completed.stdout.strip()
        endpoint_to_node = {
            f"127.0.0.1:{node.port}": node_id
            for node_id, node in self.nodes.items()
        }
        owner = endpoint_to_node.get(reported_endpoint)
        if owner is None:
            raise RuntimeError(
                f"locate returned unadvertised endpoint {reported_endpoint!r} for {key!r}; "
                f"advertised={sorted(endpoint_to_node)}")
        return owner


def _candidate(chain_id: str, index: int, candidate: int) -> bytes:
    return (f"sto17/segment-affinity/v1/chain-{chain_id}/object-{index:02d}/"
            f"candidate-{candidate:05d}").encode("ascii")


def _build_deterministic_chains(ring: Ring) -> tuple[dict[str, list[bytes]], dict[str, Any]]:
    """Select fixed ring-independent candidate order; direct probes remain the proof.

    The first object of every 16-object absolute segment in chain A/B is chosen
    on n1/n2 respectively.  Thus both S=16 and S=64 meet the requested whole-
    chain one-owner invariant.  Non-anchor objects retain candidate zero unless
    needed to guarantee independent off placement.  locate is construction and
    supporting evidence only, never substituted for direct storage probes.
    """
    chains: dict[str, list[bytes]] = {}
    locate_owners: dict[str, str] = {}
    attempts: dict[str, int] = {}
    for chain_id, target in zip(CHAIN_IDS, NODE_IDS):
        row = []
        for index in range(OBJECTS_PER_CHAIN):
            candidate = 0
            while True:
                if candidate >= LOCATE_CANDIDATE_LIMIT:
                    raise RuntimeError(f"no anchor candidate for chain {chain_id} index {index}")
                key = _candidate(chain_id, index, candidate)
                if index % 16:
                    break
                owner = ring.locate(key)
                attempts[_key_text(key)] = candidate + 1
                if owner == target:
                    break
                candidate += 1
            row.append(key)
        owners = {ring.locate(key) for key in row}
        if len(owners) < 2:
            for index in range(1, OBJECTS_PER_CHAIN):
                if index % 16 == 0:
                    continue
                for candidate in range(1, LOCATE_CANDIDATE_LIMIT):
                    key = _candidate(chain_id, index, candidate)
                    owner = ring.locate(key)
                    attempts[_key_text(key)] = candidate + 1
                    if owner != target:
                        row[index] = key
                        break
                else:
                    raise RuntimeError(f"no independently placed key for chain {chain_id}")
                break
        for key in row:
            locate_owners[_key_text(key)] = ring.locate(key)
        if len({ring.locate(key) for key in row}) < 2:
            raise RuntimeError(f"deterministic chain {chain_id} lacks off placement diversity")
        chains[chain_id] = row
    return chains, {
        "algorithm": "bounded ascending candidates; segment-first keys co-located by locate; "
                     "candidate zero elsewhere; direct static-node GET is sole placement proof",
        "candidate_limit": LOCATE_CANDIDATE_LIMIT,
        "selected_anchor_targets": dict(zip(CHAIN_IDS, NODE_IDS)),
        "candidate_attempts_for_anchor_searches": attempts,
        "supporting_locate_owners": locate_owners,
    }


def _probe_owners(static: dict[str, NativeClient], chains: dict[str, list[bytes]],
                  values: dict[bytes, bytes], *, expect_absent: bool = False) -> dict[str, Any]:
    """Directly request every actual key from every physical node over TCP."""
    owners: dict[bytes, list[str]] = {key: [] for key in _flat(chains)}
    per_node, wrong_bytes = {}, []
    for node_id, client in static.items():
        node_hits = 0
        for keys in chains.values():
            result = _get(client, keys, "legacy")
            node_hits += sum(result["hits"])
            for i, key in enumerate(keys):
                if result["hits"][i] == 1:
                    owners[key].append(node_id)
                    if result["values"][i] != values[key]:
                        wrong_bytes.append({"key": _key_text(key), "node": node_id})
        per_node[node_id] = {"direct_hit_objects": node_hits,
                             "transport": client.transport,
                             "members": client._options.members.decode()}
    counts = Counter(len(found) for found in owners.values())
    multiplicity = {str(number): counts.get(number, 0) for number in range(4)}
    absent = [key for key, found in owners.items() if not found]
    multiple = [key for key, found in owners.items() if len(found) > 1]
    owner_map = {_key_text(key): found[0] if len(found) == 1 else found
                 for key, found in owners.items()}
    report = {
        "placement_proof": "direct exact-payload TCP GET of every actual key from each "
                           "independent one-node static client; no owner inferred from locate",
        "per_node": per_node, "owner_map": owner_map,
        "owner_multiplicity_counts": multiplicity,
        "objects_on_zero_nodes": len(absent), "objects_on_multiple_nodes": len(multiple),
        "zero_node_keys": [_key_text(x) for x in absent],
        "multiple_node_keys": [_key_text(x) for x in multiple],
        "wrong_byte_count": len(wrong_bytes), "wrong_bytes": wrong_bytes,
        "unique_exactly_one_owner_objects": counts.get(1, 0),
    }
    if expect_absent:
        if counts.get(0, 0) != 2 * OBJECTS_PER_CHAIN or wrong_bytes:
            raise RuntimeError(f"post-remove direct probe not empty: {report}")
    elif counts.get(1, 0) != 2 * OBJECTS_PER_CHAIN or wrong_bytes:
        raise RuntimeError(f"cannot directly prove unique exact placement: {report}")
    return report


def _chain_placement(probe: dict[str, Any], chains: dict[str, list[bytes]],
                     segment_size: int) -> dict[str, Any]:
    answer = {}
    for chain_id, keys in chains.items():
        rows = [probe["owner_map"][_key_text(key)] for key in keys]
        counts = Counter(rows)
        named_counts = {node: counts.get(node, 0) for node in NODE_IDS}
        segments = []
        if segment_size:
            for start in range(0, OBJECTS_PER_CHAIN, segment_size):
                segment_owners = rows[start:start + segment_size]
                unique = sorted(set(segment_owners))
                segments.append({
                    "absolute_start": start, "object_count": segment_size,
                    "actual_segment_first_key": _key_text(keys[start]),
                    "owners": unique,
                    "owner": unique[0] if len(unique) == 1 else None,
                    "owner_counts": dict(Counter(segment_owners)),
                })
        answer[chain_id] = {
            "distinct_owner_count": len(set(rows)), "owners": sorted(set(rows)),
            "owner_counts": named_counts,
            "owner_skew": {
                "max_objects_on_one_node": max(named_counts.values()),
                "min_objects_on_any_node": min(named_counts.values()),
                "max_minus_min_objects": max(named_counts.values()) - min(named_counts.values()),
                "max_owner_share": max(named_counts.values()) / OBJECTS_PER_CHAIN,
            },
            "segments": segments,
            "distinct_segment_owners": sorted({row["owner"] for row in segments
                                                if row["owner"] is not None}),
        }
    return answer


class BenchChain:
    def __init__(self, keys: list[bytes]) -> None:
        self.keys = keys
        self.n = len(keys)
        self.karr, self.klens, self.key_owners = make_key_array(keys)
        self.storage = ctypes.create_string_buffer(self.n * VALUE_BYTES + 1)
        base = ctypes.addressof(self.storage)
        self.base = base
        self.pointers = (ctypes.c_void_p * self.n)(
            *[base + i * VALUE_BYTES for i in range(self.n)])
        self.caps = (ctypes.c_uint64 * self.n)(*([VALUE_BYTES] * self.n))
        self.hits = (ctypes.c_int * self.n)()
        self.lengths = (ctypes.c_uint64 * self.n)()

    def exact(self, values: dict[bytes, bytes]) -> bool:
        for i, key in enumerate(self.keys):
            length = int(self.lengths[i])
            if self.hits[i] != 1 or length != VALUE_BYTES:
                return False
            if ctypes.string_at(self.base + i * VALUE_BYTES, length) != values[key]:
                return False
        return True


def _benchmark(client: NativeClient, chains: dict[str, list[bytes]],
               values: dict[bytes, bytes], segment_size: int,
               warmup: int, samples: int) -> dict[str, Any]:
    answer = {
        "correctness_gate": "every warm request must return 64/64 exact payloads; "
                            "latency has no pass threshold",
        "timing_boundaries": {
            "python_anchor_prep": "actual segment-first route-list derivation plus ctypes route arrays",
            "native_batch_wall": "monotonic_ns around one libdfkv batch GET only",
            "end_to_end_batch_wall": "anchor prep plus native batch wall; payload verification excluded",
        },
        "chains": {},
    }
    mode = "legacy" if segment_size == 0 else "nonempty"
    for chain_id, keys in chains.items():
        state = BenchChain(keys)
        for _ in range(warmup):
            rarr = rlens = None
            route_owners: list[Any] = []
            if mode == "nonempty":
                anchors = [keys[(i // segment_size) * segment_size]
                           for i in range(len(keys))]
                rarr, rlens, route_owners = make_route_key_array(anchors)
                rc = client.lib.dfkv_batch_get_auto_anchored(
                    client.handle, state.karr, state.klens, rarr, rlens,
                    state.pointers, state.caps, state.n, state.hits, state.lengths)
            else:
                rc = client.lib.dfkv_batch_get_auto(
                    client.handle, state.karr, state.klens, state.pointers,
                    state.caps, state.n, state.hits, state.lengths)
            del route_owners
            if rc != 0 or not state.exact(values):
                raise RuntimeError(f"warmup read failed: S={segment_size}, chain={chain_id}, rc={rc}")
        native_samples, prep_samples, total_samples = [], [], []
        sample_rows = []
        for sample in range(samples):
            total_start = time.monotonic_ns()
            rarr = rlens = None
            route_owners = []
            anchors = None
            if mode == "nonempty":
                anchors = [keys[(i // segment_size) * segment_size]
                           for i in range(len(keys))]
                rarr, rlens, route_owners = make_route_key_array(anchors)
            prep_end = time.monotonic_ns()
            if mode == "nonempty":
                rc = client.lib.dfkv_batch_get_auto_anchored(
                    client.handle, state.karr, state.klens, rarr, rlens,
                    state.pointers, state.caps, state.n, state.hits, state.lengths)
            else:
                rc = client.lib.dfkv_batch_get_auto(
                    client.handle, state.karr, state.klens, state.pointers,
                    state.caps, state.n, state.hits, state.lengths)
            native_end = time.monotonic_ns()
            prep_ns = prep_end - total_start if mode == "nonempty" else 0
            native_ns = native_end - prep_end
            total_ns = native_end - total_start
            exact = rc == 0 and state.exact(values)
            if not exact:
                raise RuntimeError(f"benchmark read mismatch: S={segment_size}, "
                                   f"chain={chain_id}, sample={sample}, rc={rc}")
            del route_owners
            native_samples.append(native_ns)
            prep_samples.append(prep_ns)
            total_samples.append(total_ns)
            sample_rows.append({
                "sample": sample, "keys": len(keys), "hits": sum(state.hits),
                "exact_payloads": len(keys), "python_anchor_prep_ns": prep_ns,
                "native_batch_wall_ns": native_ns, "end_to_end_batch_wall_ns": total_ns,
            })
        answer["chains"][chain_id] = {
            "keys_per_request": len(keys), "request_batch_count": samples,
            "warmup_batch_count": warmup, "total_measured_keys": samples * len(keys),
            "python_anchor_prep": _timing_summary(prep_samples),
            "native_batch_wall": _timing_summary(native_samples),
            "end_to_end_batch_wall": _timing_summary(total_samples),
            "sample_rows": sample_rows,
        }
    return answer


def _check(report: dict[str, Any], name: str, passed: bool, detail: Any) -> None:
    report.setdefault("checks", []).append(
        {"name": name, "passed": bool(passed), "detail": detail})
    if not passed:
        raise RuntimeError(f"acceptance invariant failed: {name}: {detail}")


def _run_scenario(name: str, segment_size: int, ring: Ring,
                  chains: dict[str, list[bytes]], values: dict[bytes, bytes],
                  static: dict[str, NativeClient], clients: list[NativeClient],
                  off_owners: dict[str, Any] | None, warmup: int,
                  samples: int) -> dict[str, Any]:
    keys = _flat(chains)
    client = NativeClient(ring.args.lib_handle, f"sto17-{name}-{os.getpid()}",
                          mds=ring.mds_endpoint)
    clients.append(client)
    report: dict[str, Any] = {
        "name": name, "segment_size": segment_size, "mode": "legacy" if segment_size == 0 else "anchored",
        "namespace": NAMESPACE.decode(), "object_layout_changed": False,
        "keys_per_chain": OBJECTS_PER_CHAIN, "chains": CHAIN_IDS,
    }
    try:
        if segment_size == 0:
            _put(client, chains["a"], values, "null")
            _put(client, chains["b"], values, "empty")
            report["write_modes"] = {"a": "anchored ABI with NULL vectors",
                                     "b": "anchored ABI with per-item empty anchors"}
        else:
            _put(client, keys, values, "nonempty", _routes(chains, segment_size))
            report["write_modes"] = {"all": "actual segment-first anchors inserted after key vectors"}
        probe = _probe_owners(static, chains, values)
        report["direct_placement"] = probe
        report["chain_placement"] = _chain_placement(probe, chains, segment_size)
        if segment_size == 0:
            for chain_id, placement in report["chain_placement"].items():
                _check(report, f"off chain {chain_id} independent placement",
                       placement["distinct_owner_count"] > 1, placement)
        else:
            for chain_id, placement in report["chain_placement"].items():
                _check(report, f"S{segment_size} chain {chain_id} exactly one whole-chain owner",
                       placement["distinct_owner_count"] == 1, placement)
            segment_owners = {owner for placement in report["chain_placement"].values()
                              for owner in placement["distinct_segment_owners"]}
            _check(report, f"S{segment_size} at least two segment owners across chains",
                   len(segment_owners) >= 2, sorted(segment_owners))
        _check(report, "128 distinct actual objects without dedup collapse",
               len(set(keys)) == 128 and len(set(values.values())) == 128 and
               probe["unique_exactly_one_owner_objects"] == 128 and
               probe["objects_on_zero_nodes"] == 0 and
               probe["objects_on_multiple_nodes"] == 0 and
               probe["wrong_byte_count"] == 0,
               {"distinct_keys": len(set(keys)), "distinct_values": len(set(values.values())),
                "unique_exactly_one_owner_objects": probe["unique_exactly_one_owner_objects"]})

        report["read_back"] = {}
        if segment_size == 0:
            for mode in ("legacy", "null", "empty"):
                checked = _read_check(keys, _get(client, keys, mode), values, [True] * len(keys))
                report["read_back"][mode] = checked
                _check(report, f"off actual-key exact GET via {mode}",
                       checked["hit_count"] == 128 and checked["wrong_byte_count"] == 0 and
                       checked["unexpected_hit_count"] == 0 and
                       checked["unexpected_miss_count"] == 0, checked)
            report["legacy_objects_under_enabled_reads"] = {}
            for enabled_size in SEGMENT_SIZES:
                enabled_anchors = _routes(chains, enabled_size)
                # locate supplies the route expectation only.  The old objects'
                # existence, bytes and unchanged owner are independently proved
                # by direct one-node reads before and after each enabled read.
                route_owners = [ring.locate(anchor) for anchor in enabled_anchors]
                expected = [probe["owner_map"][_key_text(key)] == route_owners[i]
                            for i, key in enumerate(keys)]
                checked = _read_check(
                    keys, _get(client, keys, "nonempty", enabled_anchors),
                    values, expected)
                after = _probe_owners(static, chains, values)
                evidence = {
                    "expected_cold_miss_count": expected.count(False),
                    "read": checked, "post_read_direct_placement": after,
                    "owner_map_unchanged": after["owner_map"] == probe["owner_map"],
                }
                report["legacy_objects_under_enabled_reads"][f"s{enabled_size}"] = evidence
                _check(report, f"legacy objects under S{enabled_size} expected cold misses",
                       expected.count(False) > 0 and
                       checked["miss_count"] == expected.count(False) and
                       checked["unexpected_hit_count"] == 0 and
                       checked["unexpected_miss_count"] == 0 and
                       checked["wrong_byte_count"] == 0, evidence)
                _check(report, f"S{enabled_size} reads do not migrate legacy objects",
                       after["owner_map"] == probe["owner_map"] and
                       after["wrong_byte_count"] == 0,
                       {"before": probe["owner_map"], "after": after["owner_map"]})
            anchors = None
        else:
            anchors = _routes(chains, segment_size)
            checked = _read_check(keys, _get(client, keys, "nonempty", anchors), values,
                                  [True] * len(keys))
            report["read_back"]["nonempty"] = checked
            _check(report, f"S{segment_size} anchored actual-key exact GET",
                   checked["hit_count"] == 128 and checked["wrong_byte_count"] == 0 and
                   checked["unexpected_miss_count"] == 0, checked)
            if off_owners is None:
                raise RuntimeError("enabled scenario lacks directly probed off baseline")
            expected = [off_owners[_key_text(key)] == probe["owner_map"][_key_text(key)]
                        for key in keys]
            report["legacy_layout_cold_miss"] = {
                "basis": "same keys/namespace/payload/layout; expected off owner comes from direct off probes",
                "placement_changed_objects": expected.count(False),
                "reads": {},
            }
            for mode in ("legacy", "null", "empty"):
                checked = _read_check(keys, _get(client, keys, mode), values, expected)
                report["legacy_layout_cold_miss"]["reads"][mode] = checked
                _check(report, f"S{segment_size} {mode} legacy-layout outcomes",
                       checked["unexpected_hit_count"] == 0 and
                       checked["unexpected_miss_count"] == 0 and
                       checked["wrong_byte_count"] == 0 and
                       checked["miss_count"] == expected.count(False), checked)
            _check(report, f"S{segment_size} demonstrates expected cold miss, not wrong bytes",
                   expected.count(False) > 0, report["legacy_layout_cold_miss"])
            after = _probe_owners(static, chains, values)
            report["post_cold_miss_direct_placement"] = after
            _check(report, f"S{segment_size} cold misses do not migrate/rewrite objects",
                   after["owner_map"] == probe["owner_map"] and
                   after["wrong_byte_count"] == 0,
                   {"before": probe["owner_map"], "after": after["owner_map"]})
        report["benchmark"] = _benchmark(
            client, chains, values, segment_size, warmup, samples)
        _check(report, "all warm-read batches returned exact full payloads", True, {
            "samples_per_chain": samples, "keys_per_request": OBJECTS_PER_CHAIN,
            "speed_used_as_acceptance_gate": False})

        if segment_size == 0:
            _remove(client, chains["a"], "null")
            _remove(client, chains["b"], "empty")
        else:
            _remove(client, keys, "nonempty", anchors)
        removed = _probe_owners(static, chains, values, expect_absent=True)
        report["post_remove_probe"] = removed
        _check(report, "scenario objects removed through matching actual-key identity",
               removed["objects_on_zero_nodes"] == 128 and
               removed["objects_on_multiple_nodes"] == 0 and
               removed["wrong_byte_count"] == 0,
               removed["owner_multiplicity_counts"])
        report["completed"] = True
        return report
    finally:
        client.close()


def _latency_comparison(scenarios: dict[str, dict[str, Any]]) -> dict[str, Any]:
    answer = {"description": "observed ratio to off; informational only, never a speed gate",
              "chains": {}}
    for chain_id in CHAIN_IDS:
        off = scenarios["off"]["benchmark"]["chains"][chain_id]
        row = {}
        for name in ("s16", "s64"):
            enabled = scenarios[name]["benchmark"]["chains"][chain_id]
            metrics = {}
            for timing_name in ("python_anchor_prep", "native_batch_wall", "end_to_end_batch_wall"):
                ratios = {}
                for percentile in ("p50_ns", "p95_ns", "p99_ns"):
                    numerator = enabled[timing_name][percentile]
                    denominator = off[timing_name][percentile]
                    ratios[percentile] = numerator / denominator if denominator else None
                metrics[timing_name] = {"ratio_to_off": ratios}
            row[name] = metrics
        answer["chains"][chain_id] = row
    return answer


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--smoke", action="store_true",
                        help="same semantics, four measured batches/chain; never ACCEPTANCE")
    args = parser.parse_args(argv)
    args.build_dir = args.build_dir.expanduser().resolve()
    args.server = c35.executable(args.build_dir / "dfkv_server", "dfkv_server")
    args.mds = c35.executable(args.build_dir / "dfkv_mds", "dfkv_mds")
    args.ctl = c35.executable(args.build_dir / "dfkvctl", "dfkvctl")
    args.lib = args.build_dir / "libdfkv.so"
    if not args.lib.is_file():
        parser.error(f"missing libdfkv.so: {args.lib}")
    args.lib = args.lib.resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if any(args.output_root.iterdir()):
        parser.error("output_root must be absent or empty; evidence is never overwritten")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    root = args.output_root
    for name in ("logs", "runtime", "client-access", "server-access"):
        (root / name).mkdir()
    events = c35.Jsonl(root / "lifecycle.jsonl")
    saved_env = {key: os.environ.get(key) for key in ENV_KEYS}
    wanted_env = {
        "DFKV_RING_ABSENT_GRACE_MS": "0", "DFKV_RING_READBACK_MS": "0",
        "DFKV_RING_READBACK_REWRITE": "0", "DFKV_RING_COOLDOWN_REROUTE": "0",
        "DFKV_RING_REMOVE_FANOUT": "0", "DFKV_RING_REROUTE_KEYS": "0",
        "DFKV_CLIENT_NODE_DEDUP": "0", "DFKV_CLIENT_NODE_DEDUP_GPU": "0",
        "DFKV_RDMA": "0", "DFKV_OPEN_RING_WAIT_MS": "30000",
        "DFKV_GET_MISS_RETRIES": "1", "DFKV_LOG": "WARN",
        "DFKV_ACCESS_DIR": str(root / "client-access"),
    }
    os.environ.update(wanted_env)
    etcd = c35.discover_etcd()
    args.lib_handle = _load_library(args.lib)
    version = args.lib_handle.dfkv_version()
    binaries = {name: {"path": str(path), "sha256": c35._sha256_file(path)}
                for name, path in (("dfkv_server", args.server), ("dfkv_mds", args.mds),
                                   ("dfkvctl", args.ctl), ("libdfkv", args.lib),
                                   ("etcd", etcd))}
    git = _git_head()
    warmup, samples = (1, 4) if args.smoke else (8, 64)
    report: dict[str, Any] = {
        "wall_time": c35._iso(), "argv": sys.argv if argv is None else [sys.argv[0], *argv],
        "python": sys.version, "build_dir": str(args.build_dir),
        "output_root": str(root), "smoke": args.smoke,
        "status": "RUNNING", "git": git, "binaries": binaries,
        "libdfkv_version": version.decode(errors="replace") if version else "",
        "process_model": "one real etcd, one MDS, three file-engine TCP servers on 127.0.0.1",
        "client_env": {key: os.environ.get(key) for key in ENV_KEYS},
        "benchmark_plan": {"warmup_batches_per_chain": warmup,
                           "measured_batches_per_chain": samples,
                           "keys_per_request": OBJECTS_PER_CHAIN,
                           "speed_threshold": None},
        "scenarios": {},
    }
    c35._write_json(root / "run-metadata.json", report)
    ring = Ring(args, etcd, root, events)
    clients: list[NativeClient] = []
    failure = ""
    succeeded = False
    try:
        ring.start()
        report["membership"] = ring.membership
        chains, construction = _build_deterministic_chains(ring)
        values = _values(chains)
        identity = _identity(chains, values)
        report["workload"] = {
            "chains": {name: [_key_text(key) for key in keys]
                       for name, keys in chains.items()},
            "objects_per_chain": OBJECTS_PER_CHAIN, "distinct_objects": 128,
            "value_bytes": VALUE_BYTES, "distinct_values": 128,
            "identity_sha256": identity, "construction": construction,
            "value_sha256": {_key_text(key): hashlib.sha256(value).hexdigest()
                             for key, value in values.items()},
        }
        c35._write_json(root / "workload.json", report["workload"])
        static = {}
        for node_id, node in ring.nodes.items():
            client = NativeClient(args.lib_handle, f"sto17-direct-{node_id}-{os.getpid()}",
                                  members=f"{node_id}=127.0.0.1:{node.port}")
            clients.append(client)
            static[node_id] = client
        off = None
        for name, size in (("off", 0), ("s16", 16), ("s64", 64)):
            scenario = _run_scenario(name, size, ring, chains, values, static,
                                     clients, off, warmup, samples)
            report["scenarios"][name] = scenario
            if name == "off":
                off = scenario["direct_placement"]["owner_map"]
            c35._write_json(root / "result.json", report)
        report["latency_comparison"] = _latency_comparison(report["scenarios"])
        core_alive = [child for child in ring.core_children if child.popen.poll() is None]
        report["pre_cleanup_core_alive_count"] = len(core_alive)
        if len(core_alive) != 5:
            raise RuntimeError(f"core process died during run: alive={len(core_alive)}/5")
        succeeded = True
    except BaseException:
        failure = traceback.format_exc()
        report["error"] = failure
    finally:
        for client in reversed(clients):
            try:
                client.close()
            except Exception:
                failure += "\nclient cleanup:\n" + traceback.format_exc()
                succeeded = False
        report["client_lifecycle"] = {
            "opened_count": len(clients),
            "closed_count": sum(1 for client in clients if not client.handle),
            "alive_count": sum(1 for client in clients if client.handle),
        }
        cleanup_error = ""
        try:
            ring.processes.close()
        except Exception:
            cleanup_error = traceback.format_exc()
            failure += "\nprocess cleanup:\n" + cleanup_error
            succeeded = False
        alive = [child.name for child in ring.processes.children
                 if child.popen.poll() is None]
        report["process_lifecycle"] = {
            "started_count": len(ring.processes.children),
            "pre_cleanup_core_alive_count": report.get("pre_cleanup_core_alive_count"),
            "post_cleanup_alive_count": len(alive), "post_cleanup_alive_names": alive,
            "children": [{"name": child.name, "pid": child.popen.pid,
                          "returncode": child.popen.returncode,
                          "log": str(child.log_path)}
                         for child in ring.processes.children],
            "cleanup_error": cleanup_error or None,
        }
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        report["finished_wall_time"] = c35._iso()
        if succeeded and not failure and not alive:
            report["status"] = "SMOKE_COMPLETE_NON_ACCEPTANCE" if args.smoke else "ACCEPTANCE"
            report["acceptance"] = not args.smoke
        else:
            report["status"] = "SMOKE_FAILED" if args.smoke else "ACCEPTANCE_FAILED"
            report["acceptance"] = False
        if failure:
            report["error"] = failure
        c35._write_json(root / "result.json", report)
        events.write({"event": "runner_finished", "wall_time": c35._iso(),
                      "status": report["status"],
                      "post_cleanup_alive_count": len(alive)})
        events.close()
    print(json.dumps({"status": report["status"], "smoke": args.smoke,
                      "git_head": git["head"], "result": str(root / "result.json"),
                      "post_cleanup_alive": report["process_lifecycle"]["post_cleanup_alive_count"]},
                     sort_keys=True), flush=True)
    return 0 if report["status"] in ("ACCEPTANCE", "SMOKE_COMPLETE_NON_ACCEPTANCE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
