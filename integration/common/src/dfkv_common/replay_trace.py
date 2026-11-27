"""Opt-in ordered IO/prefix evidence and its fail-closed stream validator.

This is an observation format, not a cache-key or payload replay encoder.  The
only object identities emitted are native BlockKey filenames (16 hex tenant
hash + 32 hex object digest).  They remain sensitive pseudonyms: protect trace
files and apply an explicit retention policy.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import struct
import threading
import time
from pathlib import Path

from .identity import namespace_tenant_hash, pool_key_hash_field

SCHEMA_VERSION = 1
_BLOCK_KEY = re.compile(r"[0-9a-f]{48}\Z")
_OPERATIONS = {"put", "get", "get_auto", "exist", "retain", "remove"}
_STATUSES = {
    0: "DFKV_ST_OK", 1: "DFKV_ST_NOT_FOUND", 2: "DFKV_ST_CACHE_FULL",
    3: "DFKV_ST_QUOTA_EXCEEDED", 4: "DFKV_ST_IO_ERROR",
    5: "DFKV_ST_INVALID", 6: "DFKV_ST_RESOURCE_EXHAUSTED",
    7: "DFKV_ST_NO_COMPATIBLE_RAIL", 8: "DFKV_ST_LENGTH_MISMATCH",
    9: "DFKV_ST_PEER_COOLING", 10: "DFKV_ST_NO_ROUTE",
    11: "DFKV_ST_BAD_ARGUMENT",
}
_RETAINS = {
    0: "DFKV_RET_MISS", 1: "DFKV_RET_HIT_UNPROTECTED",
    2: "DFKV_RET_PROTECTED_NEW", 3: "DFKV_RET_REFRESHED",
    4: "DFKV_RET_UPGRADED", 5: "DFKV_RET_UNCHANGED", 6: "DFKV_RET_DENIED",
    7: "DFKV_RET_UNSUPPORTED", 8: "DFKV_RET_SKIPPED_DEBOUNCE",
    9: "DFKV_RET_ERROR",
}
_INSTANCES = itertools.count()
_INSTANCES_LOCK = threading.Lock()


def block_key_filename(key_namespace: bytes, object_key: bytes) -> str:
    """The exact native ``ToBlockKey(...).Filename()`` identity.

    No random per-process fingerprint is used, so the same canonical namespace
    and object key match across ranks and capture processes.
    """
    if not isinstance(key_namespace, bytes) or not isinstance(object_key, bytes):
        raise TypeError("namespace and object key must be bytes")
    framed = (b"DFKVKEY2" + struct.pack("<Q", len(key_namespace)) +
              key_namespace + struct.pack("<Q", len(object_key)) + object_key)
    return (f"{namespace_tenant_hash(key_namespace):016x}" +
            hashlib.sha256(framed).hexdigest()[:32])


def _next_instance() -> int:
    with _INSTANCES_LOCK:
        return next(_INSTANCES)


class ReplayTraceWriter:
    """One exclusively-owned JSONL stream, optionally rotated by byte size.

    ``path_prefix`` is a prefix, not an existing file to truncate.  Every open
    adds role/pid/capture/generation and uses O_EXCL|O_NOFOLLOW with mode 0600.
    No directory is created or removed.  The normal constructor records unknown
    initial inventory; ``initial_inventory_empty=True`` is solely an explicit
    caller assertion and is never inferred by the connector.
    """

    def __init__(self, path_prefix, *, role: str, key_namespace: bytes,
                 coordinates=None, initial_inventory_empty: bool = False,
                 max_bytes: int = 64 << 20):
        if not path_prefix:
            raise ValueError("empty replay trace path")
        if role not in {"native", "connector-prefix", "test"}:
            raise ValueError("unsupported replay trace role")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer")
        self._base = os.fspath(path_prefix)
        self._role = role
        self._key_namespace = key_namespace
        self._coordinates = dict(coordinates or {})
        self._initial_empty = bool(initial_inventory_empty)
        self._max_bytes = max_bytes
        self._lock = threading.RLock()
        self._capture_wall_ns = time.time_ns()
        self._capture_monotonic_ns = time.monotonic_ns()
        self._capture_id = (
            f"{self._capture_wall_ns}-{os.getpid()}-{_next_instance()}")
        self._generation = 0
        self._sequence = 0
        self._fd = None
        self._bytes = 0
        self._paths = []
        self._closed = False
        self._write_errors = 0
        self._dropped_total = 0
        self._pending_drops = []
        self._open_generation(0)
        self._append("start", {
            "reason": "capture_start",
            "initial_inventory": {
                "state": "empty" if self._initial_empty else "unknown",
                "object_count": 0 if self._initial_empty else None,
            },
        })

    @classmethod
    def from_env(cls, *, role: str, key_namespace: bytes, coordinates=None):
        """Read opt-in configuration once; return ``None`` when disabled/failed."""
        prefix = os.environ.get("DFKV_REPLAY_TRACE_PATH")
        if not prefix:
            return None
        raw_max = os.environ.get("DFKV_REPLAY_TRACE_MAX_BYTES")
        try:
            max_bytes = 64 << 20 if not raw_max else int(raw_max, 10)
            if max_bytes < 0:
                raise ValueError("negative rotation size")
            return cls(prefix, role=role, key_namespace=key_namespace,
                       coordinates=coordinates, max_bytes=max_bytes)
        except (OSError, TypeError, ValueError):
            # Observation setup cannot become a cache startup/runtime error.
            return None

    @staticmethod
    def mark():
        return {"wall_start_ns": time.time_ns(),
                "monotonic_start_ns": time.monotonic_ns()}

    @property
    def paths(self):
        with self._lock:
            return tuple(self._paths)

    def _path_for(self, generation):
        return (f"{self._base}.{self._role}.{os.getpid()}."
                f"{self._capture_id}.g{generation}.jsonl")

    def _open_generation(self, generation):
        path = self._path_for(generation)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o600)
        self._fd = fd
        self._generation = generation
        self._bytes = 0
        self._paths.append(path)

    def _base_record(self, record_type, sequence):
        return {
            "schema_version": SCHEMA_VERSION,
            "record_type": record_type,
            "capture_id": self._capture_id,
            "capture_start_epoch_ns": self._capture_wall_ns,
            "capture_start_monotonic_ns": self._capture_monotonic_ns,
            "sequence": sequence,
            "generation": self._generation,
        }

    @staticmethod
    def _line(record):
        return (json.dumps(record, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")

    def _write_line(self, line):
        view = memoryview(line)
        while view:
            count = os.write(self._fd, view)
            if count <= 0:
                raise OSError("short replay trace write")
            view = view[count:]
            self._bytes += count

    def _write_record(self, record_type, body):
        sequence = self._sequence
        self._sequence += 1
        record = self._base_record(record_type, sequence)
        record.update(body)
        try:
            self._write_line(self._line(record))
        except (OSError, TypeError, ValueError):
            self._write_errors += 1
            self._pending_drops.append(sequence)

    def _note_unsequenced_drop(self):
        self._write_errors += 1
        self._pending_drops.append(None)

    def _flush_drops(self):
        if not self._pending_drops or self._fd is None:
            return
        drops, self._pending_drops = self._pending_drops, []
        numbered = [seq for seq in drops if seq is not None]
        body = {
            "dropped_sequence_begin": min(numbered) if numbered else None,
            "dropped_sequence_end": max(numbered) if numbered else None,
            "dropped_count": len(drops),
            "expected_dropped_count": len(drops),
        }
        self._dropped_total += len(drops)
        self._write_record("drop", body)

    def _rotate(self):
        next_generation = self._generation + 1
        self._write_record("rotation", {"next_generation": next_generation})
        os.close(self._fd)
        self._fd = None
        try:
            self._open_generation(next_generation)
        except OSError:
            self._write_errors += 1
            return
        self._write_record("start", {
            "reason": "rotation",
            "initial_inventory": {
                "state": "carried_from_same_capture", "object_count": None,
            },
        })

    def _append(self, record_type, body):
        with self._lock:
            if self._closed:
                return
            self._flush_drops()
            if self._fd is None:
                self._sequence += 1
                self._pending_drops.append(self._sequence - 1)
                return
            estimated = len(body) * 8 if body else 0
            if (self._max_bytes and self._bytes and
                    self._bytes + estimated + 512 > self._max_bytes):
                self._rotate()
            if self._fd is None:
                self._sequence += 1
                self._pending_drops.append(self._sequence - 1)
                return
            self._flush_drops()
            self._write_record(record_type, body)

    def observation_failed(self):
        """Make an observation serialization/alignment failure visible."""
        try:
            with self._lock:
                self._dropped_total += 1
            self._append("drop", {
                "dropped_sequence_begin": None, "dropped_sequence_end": None,
                "dropped_count": 1, "expected_dropped_count": 1,
                "reason": "record_serialization_failed",
            })
        except Exception:
            pass

    def record_prefix(self, mark, *, operation, probe_operation, pages,
                      coordinates=None, refresh=None, result=None):
        """Append one complete logical prefix event; never truncate its pages."""
        wall_end = time.time_ns()
        monotonic_end = time.monotonic_ns()
        try:
            native_pages = build_prefix_pages(
                pages["logical_keys"], pages["native_pages"],
                pages["native_results"], self._key_namespace)
            body = {
                "operation": operation,
                "probe_operation": probe_operation,
                "namespace_layout_generation": block_key_filename(
                    self._key_namespace, b""),
                "coordinates": dict(self._coordinates if coordinates is None
                                    else coordinates),
                "wall_start_ns": mark["wall_start_ns"],
                "wall_end_ns": wall_end,
                "monotonic_start_ns": mark["monotonic_start_ns"],
                "monotonic_end_ns": monotonic_end,
                "parent": {"state": "unknown", "reason": "not_supplied"},
                "expected_page_count": len(pages["logical_keys"]),
                "page_count": len(native_pages),
                "pages": native_pages,
            }
            if refresh is not None:
                body["refresh"] = refresh
            if result is not None:
                body["result"] = result
            self._append("prefix_exists", body)
        except Exception:
            self.observation_failed()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._fd is None:
                return
            capture_end_epoch_ns = time.time_ns()
            capture_end_monotonic_ns = time.monotonic_ns()
            sync_failed = False
            try:
                os.fsync(self._fd)
            except OSError:
                sync_failed = True
                self._write_errors += 1
            self._flush_drops()
            after_end = self._sequence + 1
            self._write_record("end", {
                "capture_end_epoch_ns": capture_end_epoch_ns,
                "capture_end_monotonic_ns": capture_end_monotonic_ns,
                "initial_inventory_state_at_end": (
                    "empty" if self._initial_empty else "unknown"),
                "write_error_count": self._write_errors,
                "final_fsync_error": sync_failed,
                "expected_record_count": after_end,
                "dropped_record_count": self._dropped_total + len(self._pending_drops),
                "durability": "fsync_before_end",
            })
            os.close(self._fd)
            self._fd = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def build_prefix_pages(logical_keys, native_pages, native_results, key_namespace):
    """Project exact ordered producer inputs to safe structured prefix pages.

    Each native descriptor carries the actual bytes used for its BlockKey, a
    producer-supplied storage kind and optional explicit component/SG fields.
    No pool geometry is inferred or rebuilt here.  Raw object bytes are consumed
    only for their stable digest and are not present in the returned records.
    """
    if not (len(logical_keys) == len(native_pages) == len(native_results)):
        raise ValueError("prefix page/native/result lengths differ")
    projected = []
    for page_index, (logical, descriptors, results) in enumerate(
            zip(logical_keys, native_pages, native_results)):
        field = pool_key_hash_field(logical)
        if len(field) < 4 or struct.unpack_from("<I", field)[0] != len(field) - 4:
            raise ValueError("invalid pool key hash field")
        raw_hash = field[4:]
        native_items = []
        if len(descriptors) != len(results):
            raise ValueError("native key/result lengths differ")
        for native_index, (descriptor, present) in enumerate(zip(descriptors, results)):
            if present is not None and type(present) is not bool:
                raise ValueError("prefix probe result must be bool or None")
            key = descriptor["key"]
            item = {
                "native_input_index": native_index,
                "block_key": block_key_filename(key_namespace, key),
                "storage_kind": descriptor["storage_kind"],
                "component": descriptor.get("component"),
                "component_index": descriptor.get("component_index"),
                "sg_group": descriptor.get("sg_group"),
                "sg_width": descriptor.get("sg_width"),
                "present": bool(present) if present is not None else None,
                "present_semantics": "connector_boolean_result_not_dfkv_status",
                "dfkv_status": None,
                "output_available": present is not None,
            }
            native_items.append(item)
        projected.append({
            "input_index": page_index,
            "logical_page_hash_hex": raw_hash.hex(),
            "logical_page_hash_byte_length": len(raw_hash),
            "native_items": native_items,
        })
    return projected


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def _parse_float(value):
    import math
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"nonfinite JSON number: {value}")
    return number


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _is_uint(value):
    return type(value) is int and value >= 0


def _is_bool(value):
    return type(value) is bool


def _is_block_key(value):
    return isinstance(value, str) and _BLOCK_KEY.fullmatch(value) is not None


def _validate_times(record, errors, prefix=""):
    for field in ("wall_start_ns", "wall_end_ns", "monotonic_start_ns",
                  "monotonic_end_ns"):
        if not _is_uint(record.get(field)):
            errors.append(f"invalid_{prefix}{field}")
    for kind in ("wall", "monotonic"):
        start, end = record.get(f"{kind}_start_ns"), record.get(f"{kind}_end_ns")
        if _is_uint(start) and _is_uint(end) and end < start:
            errors.append(f"{prefix}{kind}_time_reversed")


def _validate_segment_payload(operation, payload, expected_total, errors):
    is_put = operation == "put"
    lengths_field = "segment_lengths" if is_put else "segment_capacities"
    count_field = "length_descriptor_count" if is_put else "capacity_descriptor_count"
    lengths = payload.get(lengths_field)
    pointer_count = payload.get("pointer_descriptor_count")
    length_count = payload.get(count_field)
    if not isinstance(lengths, list) or not all(_is_uint(v) for v in lengths):
        errors.append("invalid_sg_segments")
        return
    if (not _is_uint(pointer_count) or pointer_count != len(lengths) or
            not _is_uint(length_count) or length_count != len(lengths)):
        errors.append("sg_descriptor_count_mismatch")
    if payload.get("sum_valid") is not True:
        errors.append("invalid_sg_sum")
    elif sum(lengths) != expected_total:
        errors.append("sg_total_length_mismatch")


def _validate_batch(record, errors):
    operation = record.get("operation")
    if operation not in _OPERATIONS:
        errors.append("unknown_batch_operation")
    _validate_times(record, errors)
    items = record.get("items")
    expected = record.get("expected_item_count")
    if (not isinstance(items, list) or not _is_uint(expected) or
            not _is_uint(record.get("item_count")) or
            record.get("item_count") != expected or len(items) != expected):
        errors.append("batch_item_count_mismatch")
        return
    if not _is_block_key(record.get("namespace_layout_generation")):
        errors.append("invalid_namespace_layout_generation")
    valid_kinds = {
        "put": {"contiguous_put", "scatter_gather_put"},
        "get": {"contiguous_get"},
        "get_auto": {"contiguous_get", "scatter_gather_get"},
        "exist": {"key_only"}, "retain": {"key_only"}, "remove": {"key_only"},
    }
    indices = []
    saw_sg = False
    for item in items:
        if not isinstance(item, dict):
            errors.append("invalid_batch_item")
            continue
        required = {"input_index", "block_key", "output_available", "returned",
                    "status", "status_code", "requested_length", "capacity",
                    "stored_length", "stored_length_semantics", "payload"}
        if operation == "retain":
            required |= {"retain_result", "retain_result_code"}
        if not required.issubset(item):
            errors.append("missing_batch_item_fields")
        indices.append(item.get("input_index"))
        if not _is_uint(item.get("input_index")):
            errors.append("invalid_input_index")
        if not _is_block_key(item.get("block_key")):
            errors.append("invalid_block_key")
        available = item.get("output_available")
        if not _is_bool(available):
            errors.append("missing_output_availability")
        if available is not True:
            errors.append("incomplete_item_output")
        returned = item.get("returned")
        if available is not True or not _is_bool(returned):
            errors.append("invalid_item_output")
        status, code = item.get("status"), item.get("status_code")
        if operation in {"put", "get", "get_auto"}:
            if (not _is_uint(code) or code not in _STATUSES or
                    status != _STATUSES[code]):
                errors.append("unknown_or_mismatched_item_status")
            elif returned is not (code == 0):
                errors.append("returned_status_mismatch")
        elif status is not None or code is not None:
            errors.append("unexpected_key_only_status")

        requested = item.get("requested_length")
        capacity = item.get("capacity")
        stored = item.get("stored_length")
        semantics = item.get("stored_length_semantics")
        if operation == "put":
            if not _is_uint(requested) or capacity is not None or stored is not None:
                errors.append("invalid_put_lengths")
            if semantics != "not_returned":
                errors.append("invalid_stored_length_semantics")
        elif operation in {"get", "get_auto"}:
            if requested is not None or not _is_uint(capacity):
                errors.append("invalid_get_capacity")
            if operation == "get":
                if stored is not None or semantics != "not_returned":
                    errors.append("exact_get_capacity_mislabelled_as_length")
            elif returned is True:
                if not _is_uint(stored) or semantics != "actual_stored_length":
                    errors.append("missing_actual_stored_length")
                elif _is_uint(capacity) and stored > capacity:
                    errors.append("stored_length_exceeds_capacity")
            elif stored is not None or semantics not in {"not_returned", "not_available"}:
                errors.append("missing_output_relabelled_as_zero_or_length")
        elif (requested is not None or capacity is not None or stored is not None or
              semantics != "not_returned"):
            errors.append("invalid_key_only_lengths")

        if operation == "retain":
            retain = item.get("retain_result_code")
            if (not _is_uint(retain) or retain not in _RETAINS or
                    item.get("retain_result") != _RETAINS[retain]):
                errors.append("unknown_or_mismatched_retain_result")
            elif returned is not (retain in {1, 2, 3, 4, 5, 6, 8}):
                errors.append("retain_returned_result_mismatch")
        elif "retain_result" in item or "retain_result_code" in item:
            errors.append("unexpected_retain_result")

        payload = item.get("payload")
        if (not isinstance(payload, dict) or
                payload.get("kind") not in valid_kinds.get(operation, set())):
            errors.append("unknown_or_mismatched_payload_descriptor_kind")
            continue
        kind = payload["kind"]
        is_sg = kind in {"scatter_gather_put", "scatter_gather_get"}
        saw_sg = saw_sg or is_sg
        if is_sg:
            if record.get("payload_form") != "scatter_gather":
                errors.append("missing_sg_payload_form")
            _validate_segment_payload(
                operation, payload, requested if operation == "put" else capacity, errors)
        elif "payload_form" in record and record["payload_form"] == "scatter_gather":
            errors.append("unexpected_sg_payload_form")
        if any(key in item for key in ("pointer", "payload_bytes", "route_key", "value")):
            errors.append("unsafe_item_field")
    if saw_sg and record.get("payload_form") != "scatter_gather":
        errors.append("missing_sg_payload_form")
    if not saw_sg and record.get("payload_form") == "scatter_gather":
        errors.append("unexpected_sg_payload_form")
    if indices != list(range(expected)):
        errors.append("original_item_order_broken")
    if operation == "retain":
        retention = record.get("retention")
        if not isinstance(retention, dict):
            errors.append("missing_retention")
        else:
            mode_code = retention.get("mode_code")
            expected_mode = {0: "set", 1: "refresh"}.get(mode_code) if _is_uint(mode_code) else None
            if expected_mode is None or retention.get("mode") != expected_mode:
                errors.append("missing_or_unknown_retain_mode")
            if (not _is_uint(retention.get("ttl_s")) or
                    not _is_uint(retention.get("min_refresh_ms")) or
                    not _is_bool(retention.get("no_debounce"))):
                errors.append("invalid_retention_parameters")
    elif "retention" in record:
        errors.append("unexpected_retention")


def _validate_prefix(record, errors):
    if record.get("operation") not in {"batch_exists", "batch_exists_v2_pool"}:
        errors.append("unknown_prefix_operation")
    if record.get("probe_operation") not in {"exist", "retain_refresh"}:
        errors.append("unknown_prefix_probe_operation")
    refresh = record.get("refresh")
    if record.get("probe_operation") == "retain_refresh":
        if not isinstance(refresh, dict) or refresh.get("mode") != "refresh":
            errors.append("missing_prefix_refresh_mode")
    if refresh is not None:
        if (not isinstance(refresh, dict) or refresh.get("mode") != "refresh" or
                not _is_uint(refresh.get("ttl_s")) or refresh.get("ttl_s") != 0 or
                not _is_bool(refresh.get("attempted"))):
            errors.append("invalid_prefix_refresh")
    _validate_times(record, errors, "prefix_")
    if (not isinstance(record.get("parent"), dict) or
            record["parent"].get("state") != "unknown"):
        errors.append("unverified_parent_relationship")
    if not _is_block_key(record.get("namespace_layout_generation")):
        errors.append("invalid_namespace_layout_generation")
    pages = record.get("pages")
    expected = record.get("expected_page_count")
    if (not isinstance(pages, list) or not _is_uint(expected) or
            len(pages) != expected or not _is_uint(record.get("page_count")) or
            record.get("page_count") != expected):
        errors.append("prefix_page_count_mismatch")
        return
    indices = []
    for page in pages:
        if not isinstance(page, dict):
            errors.append("invalid_prefix_page")
            continue
        indices.append(page.get("input_index"))
        if not _is_uint(page.get("input_index")):
            errors.append("invalid_prefix_input_index")
        digest = page.get("logical_page_hash_hex")
        declared_length = page.get("logical_page_hash_byte_length")
        if (not isinstance(digest, str) or not digest or len(digest) % 2 or
                re.fullmatch(r"[0-9a-f]+", digest) is None):
            errors.append("invalid_logical_page_hash")
        elif (not _is_uint(declared_length) or declared_length == 0 or
                len(bytes.fromhex(digest)) != declared_length):
            errors.append("logical_page_hash_length_mismatch")
        native_items = page.get("native_items")
        if not isinstance(native_items, list):
            errors.append("invalid_prefix_native_items")
            continue
        native_indices = []
        for item in native_items:
            if not isinstance(item, dict):
                errors.append("invalid_prefix_native_item")
                continue
            native_indices.append(item.get("native_input_index"))
            if not _is_uint(item.get("native_input_index")):
                errors.append("invalid_prefix_native_input_index")
            required = {"native_input_index", "block_key", "storage_kind",
                        "component", "component_index", "sg_group", "sg_width",
                        "present", "present_semantics", "dfkv_status",
                        "output_available"}
            if not required.issubset(item):
                errors.append("missing_prefix_native_item_fields")
            if item.get("output_available") is not True:
                errors.append("incomplete_prefix_output")
            if type(item.get("present")) is not bool:
                errors.append("invalid_prefix_presence")
            if (item.get("present_semantics") !=
                    "connector_boolean_result_not_dfkv_status" or
                    "dfkv_status" not in item or item.get("dfkv_status") is not None):
                errors.append("invalid_prefix_presence_semantics")
            if not _is_block_key(item.get("block_key")):
                errors.append("invalid_prefix_block_key")
            storage_kind = item.get("storage_kind")
            component = item.get("component")
            component_index = item.get("component_index")
            if not isinstance(component, str) or not _is_uint(component_index):
                errors.append("invalid_prefix_component")
            if storage_kind == "pool":
                if item.get("sg_group") is not None or item.get("sg_width") is not None:
                    errors.append("unexpected_pool_sg_geometry")
            elif storage_kind == "sg_group":
                if (not _is_uint(item.get("sg_group")) or
                        not _is_uint(item.get("sg_width")) or item.get("sg_width") == 0):
                    errors.append("invalid_sg_group_geometry")
            else:
                errors.append("unknown_prefix_storage_kind")
        if native_indices != list(range(len(native_items))):
            errors.append("original_native_order_broken")
    if indices != list(range(expected)):
        errors.append("original_page_order_broken")


def validate_replay_trace(paths):
    """Fail closed unless every record belongs to one complete generation chain."""
    records = []
    errors = []
    for path in paths:
        previous_sequence = None
        try:
            with open(path, "rb") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.endswith(b"\n"):
                        errors.append(f"partial_line:{os.fspath(path)}:{line_number}")
                    try:
                        record = json.loads(
                            line, parse_constant=_reject_constant,
                            parse_float=_parse_float,
                            object_pairs_hook=_object_without_duplicate_keys)
                    except (UnicodeDecodeError, ValueError, TypeError, RecursionError):
                        errors.append(f"invalid_json:{os.fspath(path)}:{line_number}")
                        continue
                    records.append(record)
                    if isinstance(record, dict) and _is_uint(record.get("sequence")):
                        sequence = record["sequence"]
                        if previous_sequence is not None and sequence <= previous_sequence:
                            errors.append(f"file_sequence_not_increasing:{os.fspath(path)}")
                        previous_sequence = sequence
        except OSError:
            errors.append(f"io_error:{os.fspath(path)}")

    by_sequence = {}
    capture_ids = set()
    for record in records:
        if not isinstance(record, dict):
            errors.append("nonobject_record")
            continue
        capture_id = record.get("capture_id")
        if not isinstance(capture_id, str) or not capture_id:
            errors.append("invalid_capture_id")
        else:
            capture_ids.add(capture_id)
        sequence = record.get("sequence")
        if not _is_uint(sequence):
            errors.append("invalid_sequence")
            continue
        if sequence in by_sequence:
            errors.append("duplicate_sequence")
        else:
            by_sequence[sequence] = record
        if type(record.get("schema_version")) is not int or record["schema_version"] != SCHEMA_VERSION:
            errors.append("unsupported_schema_version")
        if (not _is_uint(record.get("capture_start_epoch_ns")) or
                not _is_uint(record.get("capture_start_monotonic_ns"))):
            errors.append("invalid_capture_start_timestamp")
        if not _is_uint(record.get("generation")):
            errors.append("invalid_generation")
        record_type = record.get("record_type")
        try:
            if record_type == "drop":
                errors.append("dropped_records")
                if (not _is_uint(record.get("dropped_count")) or
                        not _is_uint(record.get("expected_dropped_count")) or
                        record["dropped_count"] != record["expected_dropped_count"]):
                    errors.append("drop_count_mismatch")
            elif record_type == "batch_io":
                _validate_batch(record, errors)
            elif record_type == "prefix_exists":
                _validate_prefix(record, errors)
            elif record_type not in {"start", "rotation", "end"}:
                errors.append("unknown_record_type")
        except (AttributeError, TypeError, ValueError, IndexError, KeyError,
                OverflowError, RecursionError):
            errors.append("invalid_record_schema")
    if len(capture_ids) != 1:
        errors.append("mixed_or_missing_capture_id")

    ordered = [by_sequence[sequence] for sequence in sorted(by_sequence)]
    if [record["sequence"] for record in ordered] != list(range(len(ordered))):
        errors.append("sequence_gap")
    starts = [record for record in ordered if record.get("record_type") == "start"]
    rotations = [record for record in ordered if record.get("record_type") == "rotation"]
    ends = [record for record in ordered if record.get("record_type") == "end"]
    if not ordered or ordered[0].get("record_type") != "start":
        errors.append("missing_start")
    if len(ends) != 1 or not ordered or ordered[-1].get("record_type") != "end":
        errors.append("missing_or_unterminated_end")

    initial_inventory_state = None
    capture_start = None
    active_generation = None
    rotation_generation = None
    ended = False
    if starts:
        first = starts[0]
        first_inventory = first.get("initial_inventory")
        if first.get("generation") != 0 or first.get("reason") != "capture_start":
            errors.append("invalid_initial_generation")
        if not isinstance(first_inventory, dict):
            errors.append("invalid_initial_inventory")
        elif first_inventory.get("state") == "unknown":
            initial_inventory_state = "unknown"
            if first_inventory != {"state": "unknown", "object_count": None}:
                errors.append("invalid_initial_inventory")
            errors.append("initial_inventory_unknown")
        elif (first_inventory == {"state": "empty", "object_count": 0} and
              _is_uint(first_inventory.get("object_count"))):
            initial_inventory_state = "empty"
        else:
            errors.append("untrusted_initial_inventory")
        capture_start = (first.get("capture_start_epoch_ns"),
                         first.get("capture_start_monotonic_ns"))
    for record in ordered:
        if capture_start is not None and (
                record.get("capture_start_epoch_ns"),
                record.get("capture_start_monotonic_ns")) != capture_start:
            errors.append("capture_start_changed")
        if ended:
            errors.append("record_after_end")
            continue
        record_type = record.get("record_type")
        generation = record.get("generation")
        if rotation_generation is not None:
            if (record_type != "start" or record.get("reason") != "rotation" or
                    generation != rotation_generation or
                    record.get("initial_inventory") != {
                        "state": "carried_from_same_capture", "object_count": None}):
                errors.append("broken_rotation_chain")
            else:
                active_generation = generation
                rotation_generation = None
            continue
        if record_type == "start":
            if record is not ordered[0]:
                errors.append("unexpected_start")
        elif record_type == "rotation":
            if (active_generation is None or generation != active_generation or
                    not _is_uint(record.get("next_generation")) or
                    record.get("next_generation") != active_generation + 1):
                errors.append("broken_rotation_chain")
            rotation_generation = active_generation + 1 if active_generation is not None else None
            active_generation = None
        elif record_type == "end":
            if active_generation is None or generation != active_generation:
                errors.append("end_outside_active_generation")
            if (not _is_uint(record.get("capture_end_epoch_ns")) or
                    not _is_uint(record.get("capture_end_monotonic_ns"))):
                errors.append("invalid_capture_end_timestamp")
            elif (_is_uint(record.get("capture_start_monotonic_ns")) and
                  record["capture_end_monotonic_ns"] < record["capture_start_monotonic_ns"]):
                errors.append("capture_monotonic_time_reversed")
            if record.get("durability") != "fsync_before_end":
                errors.append("invalid_final_durability")
            if not _is_uint(record.get("write_error_count")) or record["write_error_count"] != 0:
                errors.append("writer_io_error")
            if record.get("final_fsync_error") is not False:
                errors.append("invalid_final_fsync")
            if record.get("initial_inventory_state_at_end") != initial_inventory_state:
                errors.append("end_inventory_state_mismatch")
            ended = True
        elif active_generation is None or generation != active_generation:
            errors.append("record_outside_active_generation")
        if record is ordered[0] and record_type == "start":
            active_generation = 0
    if rotation_generation is not None:
        errors.append("rotation_without_next_start")
    if ordered:
        final = ordered[-1]
        if final.get("record_type") == "end":
            if (not _is_uint(final.get("expected_record_count")) or
                    final.get("expected_record_count") != final["sequence"] + 1):
                errors.append("end_expected_record_count_mismatch")
            dropped = sum(record.get("dropped_count", 0) for record in ordered
                          if record.get("record_type") == "drop" and
                          _is_uint(record.get("dropped_count")))
            if (not _is_uint(final.get("dropped_record_count")) or
                    final.get("dropped_record_count") != dropped):
                errors.append("end_dropped_record_count_mismatch")
    return {
        "complete_for_replay": not errors,
        "errors": sorted(set(errors)),
        "capture_id": next(iter(capture_ids)) if len(capture_ids) == 1 else None,
        "record_count": len(ordered),
        "batch_count": sum(record.get("record_type") == "batch_io" for record in ordered),
        "prefix_event_count": sum(record.get("record_type") == "prefix_exists" for record in ordered),
        "rotation_count": len(rotations),
        "drop_count": sum(record.get("record_type") == "drop" for record in ordered),
    }
