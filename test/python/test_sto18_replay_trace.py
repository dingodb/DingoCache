"""STO-18 real ordered prefix records, connector primitives and fail-closed checks."""
import contextlib
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "integration", "hicache"))
sys.path.insert(0, os.path.join(ROOT, "integration", "common", "src"))

from dfkv_common import namespace_tenant_hash, pool_key, pool_key_hash_field, sg_key
from dfkv_common import replay_trace as rt
import dfkv_hicache
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolTransfer


def _independent_block_key(namespace, key):
    # Independently restate the C++ framing instead of calling the producer.
    digest_input = (b"DFKVKEY2" + len(namespace).to_bytes(8, "little") +
                    namespace + len(key).to_bytes(8, "little") + key)
    return (f"{namespace_tenant_hash(namespace):016x}" +
            hashlib.sha256(digest_input).digest()[:16].hex())


def _records(paths):
    result = []
    for path in paths:
        with open(path, encoding="ascii") as stream:
            result.extend(json.loads(line) for line in stream if line)
    return sorted(result, key=lambda record: record["sequence"])


def _page(logical, present=True, component="all", group=None, width=None):
    key = pool_key(logical, pool="kv", tp_size=2, tp_rank=1,
                   pcp_size=3, pcp_rank=2, dcp_size=4, dcp_rank=3,
                   pp_size=1, pp_rank=0, component=component)
    if group is not None:
        key = sg_key(key, width, group)
    descriptor = {"key": key, "storage_kind": "sg_group" if group is not None else "pool",
                  "component": component, "component_index": 0,
                  "sg_group": group, "sg_width": width}
    return logical, [descriptor], [present]


def _valid_batch(operation="get", code=1):
    item = {
        "input_index": 0, "block_key": "1" * 48,
        "output_available": True, "returned": False,
        "status": None, "status_code": None,
        "requested_length": None, "capacity": None, "stored_length": None,
        "stored_length_semantics": "not_returned",
        "payload": {"kind": "key_only"},
    }
    record = {
        "schema_version": 1, "record_type": "batch_io", "capture_id": "x",
        "capture_start_epoch_ns": 1, "capture_start_monotonic_ns": 1,
        "sequence": 1, "generation": 0, "operation": operation,
        "namespace_layout_generation": "0" * 48,
        "wall_start_ns": 1, "wall_end_ns": 2,
        "monotonic_start_ns": 1, "monotonic_end_ns": 2,
        "expected_item_count": 1, "item_count": 1, "items": [item],
    }
    if operation in {"put", "get", "get_auto"}:
        item["status_code"] = code
        item["status"] = rt._STATUSES[code]
    if operation == "put":
        item.update(requested_length=3,
                    payload={"kind": "contiguous_put"})
    elif operation == "get":
        item.update(capacity=3, payload={"kind": "contiguous_get"})
    elif operation == "get_auto":
        item.update(capacity=4, stored_length_semantics="not_available",
                    payload={"kind": "contiguous_get"})
    elif operation == "retain":
        item.update(retain_result="DFKV_RET_MISS", retain_result_code=0)
        record["retention"] = {"mode": "refresh", "mode_code": 1,
                               "ttl_s": 0, "no_debounce": False,
                               "min_refresh_ms": 0}
    return record


class ReplayTraceWriterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="dfkv_sto18_trace_")
        self.addCleanup(self.tmp.cleanup)
        self.base = os.path.join(self.tmp.name, "capture.jsonl")
        self.namespace = b"sto18-test-namespace"

    def writer(self, **kwargs):
        kwargs.setdefault("role", "test")
        kwargs.setdefault("key_namespace", self.namespace)
        kwargs.setdefault("initial_inventory_empty", True)
        return rt.ReplayTraceWriter(self.base, **kwargs)

    def one_event(self, writer, logical="aa" * 32, present=True, **kwargs):
        page = _page(logical, present)
        writer.record_prefix(
            writer.mark(), operation="batch_exists", probe_operation="exist",
            pages={"logical_keys": [page[0]], "native_pages": [page[1]],
                   "native_results": [page[2]]}, **kwargs)

    def test_complete_records_parse_with_exact_identities_order_and_rotation(self):
        writer = self.writer(max_bytes=1100, coordinates={"tp_size": 2, "tp_rank": 1})
        logical_keys = [f"{value:064x}" for value in (1, 2, 3)]
        native_pages = [
            [_page(logical_keys[0])[1][0],
             _page(logical_keys[0], component="v", group=1, width=29)[1][0]],
            [_page(logical_keys[1])[1][0]],
            [_page(logical_keys[2], group=0, width=29)[1][0]],
        ]
        native_results = [[True, False], [True], [False]]
        writer.record_prefix(
            writer.mark(), operation="batch_exists", probe_operation="retain_refresh",
            pages={"logical_keys": logical_keys, "native_pages": native_pages,
                   "native_results": native_results},
            refresh={"mode": "refresh", "ttl_s": 0},
            result={"contiguous_prefix": 1})
        # Real additional calls force more than one actual size rotation.
        for index in range(6, 12):
            self.one_event(writer, f"{index:064x}")
        paths = writer.paths
        writer.close()
        self.assertGreaterEqual(len(paths), 2)
        parsed = _records(paths)
        report = rt.validate_replay_trace(paths)
        self.assertTrue(report["complete_for_replay"], report)
        self.assertEqual(report["batch_count"], 0)
        self.assertEqual(report["prefix_event_count"], 7)
        self.assertGreaterEqual(report["rotation_count"], 1)
        self.assertEqual(report["drop_count"], 0)
        first = next(record for record in parsed if record["record_type"] == "prefix_exists")
        self.assertEqual([page["input_index"] for page in first["pages"]], [0, 1, 2])
        self.assertEqual(
            [page["logical_page_hash_hex"] for page in first["pages"]],
            [pool_key_hash_field(key)[4:].hex() for key in logical_keys])
        native_items = first["pages"][0]["native_items"]
        self.assertEqual([item["present"] for item in native_items], [True, False])
        self.assertEqual(native_items[1]["storage_kind"], "sg_group")
        self.assertEqual(native_items[1]["sg_group"], 1)
        self.assertEqual(native_items[1]["sg_width"], 29)
        self.assertEqual(first["parent"], {"state": "unknown", "reason": "not_supplied"})
        self.assertEqual(first["coordinates"], {"tp_size": 2, "tp_rank": 1})
        for page, descriptors in zip(first["pages"], native_pages):
            for actual, descriptor in zip(page["native_items"], descriptors):
                self.assertRegex(actual["block_key"], r"^[0-9a-f]{48}$")
                self.assertEqual(actual["block_key"],
                                 _independent_block_key(self.namespace, descriptor["key"]))
        for record in parsed:
            self.assertGreaterEqual(record["capture_start_epoch_ns"], 1)
            if record["record_type"] == "end":
                self.assertGreaterEqual(record["capture_end_epoch_ns"], 1)
                self.assertGreaterEqual(record["capture_end_monotonic_ns"],
                                        record["capture_start_monotonic_ns"])
            if record["record_type"] == "prefix_exists":
                self.assertLessEqual(record["monotonic_start_ns"], record["monotonic_end_ns"])
                self.assertLessEqual(record["wall_start_ns"], record["wall_end_ns"])
        raw = b"".join(Path(path).read_bytes() for path in paths)
        self.assertNotIn(self.namespace, raw)
        self.assertNotIn(b'raw_token', raw)
        self.assertNotIn(b'payload_bytes', raw)
        self.assertNotIn(b'route_key', raw)

    def test_unknown_inventory_prevents_complete_claim(self):
        writer = self.writer(initial_inventory_empty=False)
        self.one_event(writer)
        paths = writer.paths
        writer.close()
        report = rt.validate_replay_trace(paths)
        self.assertFalse(report["complete_for_replay"])
        self.assertIn("initial_inventory_unknown", report["errors"])

    def test_real_write_failure_leaves_drop_marker_and_fails_validation(self):
        writer = self.writer()
        real_write = os.write
        failed = [False]

        def fail_once(fd, data):
            if not failed[0]:
                failed[0] = True
                raise OSError("injected test write failure")
            return real_write(fd, data)

        with mock.patch.object(rt.os, "write", side_effect=fail_once):
            self.one_event(writer, "01" * 32)
            self.one_event(writer, "02" * 32)
        paths = writer.paths
        writer.close()
        parsed = _records(paths)
        drops = [record for record in parsed if record["record_type"] == "drop"]
        self.assertEqual(len(drops), 1)
        self.assertEqual(drops[0]["dropped_count"], 1)
        self.assertEqual(drops[0]["dropped_count"], drops[0]["expected_dropped_count"])
        self.assertEqual(parsed[-1]["record_type"], "end")
        self.assertGreaterEqual(parsed[-1]["write_error_count"], 1)
        report = rt.validate_replay_trace(paths)
        self.assertFalse(report["complete_for_replay"])
        self.assertIn("dropped_records", report["errors"])

    def test_alignment_failure_is_visible_not_silently_complete(self):
        writer = self.writer()
        page = _page("03" * 32)
        writer.record_prefix(
            writer.mark(), operation="batch_exists", probe_operation="exist",
            pages={"logical_keys": [page[0]], "native_pages": [page[1]],
                   "native_results": [[]]})
        paths = writer.paths
        writer.close()
        report = rt.validate_replay_trace(paths)
        self.assertFalse(report["complete_for_replay"])
        self.assertIn("dropped_records", report["errors"])

    def test_exclusive_never_overwrites_base_and_disabled_does_no_work(self):
        Path(self.base).write_bytes(b"other-process-data")
        writer = self.writer()
        paths = writer.paths
        writer.close()
        self.assertEqual(Path(self.base).read_bytes(), b"other-process-data")
        self.assertTrue(all(path != self.base for path in paths))

        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(rt.hashlib, "sha256",
                                  side_effect=AssertionError("disabled hashing")) as digest:
            self.assertIsNone(rt.ReplayTraceWriter.from_env(
                role="test", key_namespace=self.namespace))
            digest.assert_not_called()
        with mock.patch.dict(os.environ, {
                "DFKV_REPLAY_TRACE_PATH": os.path.join(
                    self.tmp.name, "missing-parent", "trace")}, clear=True):
            self.assertIsNone(rt.ReplayTraceWriter.from_env(
                role="test", key_namespace=self.namespace))
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "missing-parent")))

    def test_validator_rejects_nonfinite_partial_and_unsupported_status(self):
        start = {"schema_version": 1, "record_type": "start", "capture_id": "x",
                 "capture_start_epoch_ns": 1, "capture_start_monotonic_ns": 1,
                 "sequence": 0, "generation": 0, "reason": "capture_start",
                 "initial_inventory": {"state": "empty", "object_count": 0}}
        end = {"schema_version": 1, "record_type": "end", "capture_id": "x",
               "capture_start_epoch_ns": 1, "capture_start_monotonic_ns": 1,
               "capture_end_epoch_ns": 2, "capture_end_monotonic_ns": 2,
               "sequence": 2, "generation": 0, "write_error_count": 0,
               "final_fsync_error": False, "expected_record_count": 3,
               "dropped_record_count": 0, "durability": "fsync_before_end",
               "initial_inventory_state_at_end": "empty"}
        batch = {"schema_version": 1, "record_type": "batch_io", "capture_id": "x",
                 "capture_start_epoch_ns": 1, "capture_start_monotonic_ns": 1,
                 "sequence": 1, "generation": 0, "operation": "get",
                 "namespace_layout_generation": "0" * 48,
                 "wall_start_ns": 1, "wall_end_ns": 2,
                 "monotonic_start_ns": 1, "monotonic_end_ns": 2,
                 "expected_item_count": 1, "item_count": 1,
                 "items": [{"input_index": 0, "block_key": "1" * 48,
                            "output_available": True, "returned": False,
                            "status": "UNKNOWN_ITEM_STATUS", "status_code": 99,
                            "capacity": 5, "stored_length": None,
                            "payload": {"kind": "contiguous_get"}}]}
        bad = Path(self.base)
        bad.write_text("\n".join(json.dumps(record) for record in (start, batch, end)) + "\n")
        report = rt.validate_replay_trace([bad])
        self.assertFalse(report["complete_for_replay"])
        self.assertIn("unknown_or_mismatched_item_status", report["errors"])
        with bad.open("ab") as stream:
            stream.write(b'{"sequence":NaN}')
        report = rt.validate_replay_trace([bad])
        self.assertFalse(report["complete_for_replay"])
        self.assertTrue(any(error.startswith("partial_line") for error in report["errors"]))
        self.assertTrue(any(error.startswith("invalid_json") for error in report["errors"]))

    def assert_capture(self, records, complete=None, needed_error=None, path=None):
        path = Path(path or self.base)
        path.write_text("".join(json.dumps(record, separators=(",", ":")) + "\n"
                                for record in records))
        report = rt.validate_replay_trace([path])
        if complete is not None:
            self.assertEqual(report["complete_for_replay"], complete, report)
        if needed_error is not None:
            self.assertIn(needed_error, report["errors"], report)
        return report

    def start_end(self, sequence=1):
        start = {"schema_version": 1, "record_type": "start", "capture_id": "x",
                 "capture_start_epoch_ns": 1, "capture_start_monotonic_ns": 1,
                 "sequence": 0, "generation": 0, "reason": "capture_start",
                 "initial_inventory": {"state": "empty", "object_count": 0}}
        end = {"schema_version": 1, "record_type": "end", "capture_id": "x",
               "capture_start_epoch_ns": 1, "capture_start_monotonic_ns": 1,
               "capture_end_epoch_ns": 2, "capture_end_monotonic_ns": 2,
               "sequence": sequence + 1, "generation": 0, "write_error_count": 0,
               "final_fsync_error": False, "expected_record_count": sequence + 2,
               "dropped_record_count": 0, "durability": "fsync_before_end",
               "initial_inventory_state_at_end": "empty"}
        return start, end

    def test_batch_matrix_accepts_diagnostics_but_never_unavailable_or_bool_codes(self):
        start, end = self.start_end(6)
        batches = []
        for sequence, operation in enumerate(("put", "get", "get_auto", "exist",
                                               "retain", "remove"), 1):
            batch = _valid_batch(operation)
            batch["sequence"] = sequence
            batches.append(batch)
        report = self.assert_capture([start, *batches, end], complete=True)
        self.assertEqual(report["batch_count"], 6)
        for code in (4, 9, 10, 11):
            batch = _valid_batch("get_auto", code=code)
            start, end = self.start_end()
            self.assert_capture([start, batch, end], complete=True)
        hit = _valid_batch("get_auto", code=0)
        hit["items"][0].update(returned=True, stored_length=3,
                               stored_length_semantics="actual_stored_length")
        start, end = self.start_end()
        self.assert_capture([start, hit, end], complete=True)
        hit["items"][0]["stored_length"] = 5
        self.assert_capture([start, hit, end], complete=False,
                            needed_error="stored_length_exceeds_capacity")
        for operation, mutate, expected in (
                ("put", lambda b: b["items"][0].pop("requested_length"), "invalid_put_lengths"),
                ("get", lambda b: b["items"][0].update(capacity=None), "invalid_get_capacity"),
                ("get", lambda b: b["items"][0].update(status_code=True),
                 "unknown_or_mismatched_item_status"),
                ("get_auto", lambda b: b["items"][0].update(output_available=True,
                    returned=True, stored_length=None), "missing_actual_stored_length"),
                ("retain", lambda b: b["items"][0].update(retain_result_code=None),
                 "unknown_or_mismatched_retain_result"),
                ("retain", lambda b: b["items"][0].update(returned=True),
                 "retain_returned_result_mismatch"),
                ("exist", lambda b: b["items"][0].update(status_code=0),
                 "unexpected_key_only_status")):
            batch = _valid_batch(operation)
            mutate(batch)
            start, end = self.start_end()
            self.assert_capture([start, batch, end], complete=False, needed_error=expected)
        for field in ("ttl_s", "min_refresh_ms", "no_debounce", "mode_code"):
            batch = _valid_batch("retain")
            batch["retention"].pop(field)
            start, end = self.start_end()
            self.assert_capture([start, batch, end], complete=False)
        for code in (1, 2, 3, 4, 5, 6, 8):
            batch = _valid_batch("retain")
            batch["items"][0].update(retain_result_code=code,
                retain_result=rt._RETAINS[code], returned=True)
            start, end = self.start_end()
            self.assert_capture([start, batch, end], complete=True)

    def test_sg_geometry_is_complete_and_disjoint_sizes_must_sum(self):
        batch = _valid_batch("put")
        batch["payload_form"] = "scatter_gather"
        batch["items"][0]["payload"] = {
            "kind": "scatter_gather_put", "segment_lengths": [1, 2],
            "pointer_descriptor_count": 2, "length_descriptor_count": 2,
            "sum_valid": True}
        start, end = self.start_end()
        self.assert_capture([start, batch, end], complete=True)
        for field, value in (("sum_valid", False), ("pointer_descriptor_count", 1),
                             ("length_descriptor_count", 3)):
            broken = copy.deepcopy(batch)
            broken["items"][0]["payload"][field] = value
            self.assert_capture([start, broken, end], complete=False)
        broken = copy.deepcopy(batch)
        broken["items"][0]["payload"]["segment_lengths"] = [1, 1]
        self.assert_capture([start, broken, end], complete=False,
                            needed_error="sg_total_length_mismatch")
        broken = copy.deepcopy(batch)
        broken.pop("payload_form")
        self.assert_capture([start, broken, end], complete=False,
                            needed_error="missing_sg_payload_form")

    def test_prefix_requires_boolean_result_and_coherent_layout(self):
        writer = self.writer()
        self.one_event(writer)
        paths = writer.paths
        writer.close()
        records = _records(paths)
        prefix = next(r for r in records if r["record_type"] == "prefix_exists")
        for mutate, expected in (
                (lambda item: item.update(present=None, output_available=True),
                 "invalid_prefix_presence"),
                (lambda item: item.update(present=None, output_available=False),
                 "incomplete_prefix_output"),
                (lambda item: item.update(storage_kind="pool", sg_width=29),
                 "unexpected_pool_sg_geometry"),
                (lambda item: item.update(component=None), "invalid_prefix_component")):
            broken = copy.deepcopy(records)
            event = next(r for r in broken if r["record_type"] == "prefix_exists")
            mutate(event["pages"][0]["native_items"][0])
            self.assert_capture(broken, complete=False, needed_error=expected,
                                path=self.base + ".prefix-mutation")
        for mutate, expected in (
                (lambda event: event.update(namespace_layout_generation="short"),
                 "invalid_namespace_layout_generation"),
                (lambda event: event["pages"][0].update(logical_page_hash_byte_length=1),
                 "logical_page_hash_length_mismatch")):
            broken = copy.deepcopy(records)
            event = next(r for r in broken if r["record_type"] == "prefix_exists")
            mutate(event)
            self.assert_capture(broken, complete=False, needed_error=expected,
                                path=self.base + ".prefix-layout")

    def test_malformed_json_and_envelope_mutations_are_structured_failures(self):
        start, end = self.start_end(0)
        malformed = [
            b'{"schema_version":1,"schema_version":1}\n',
            b'[' * 2000 + b'0' + b']' * 2000 + b'\n',
            json.dumps(start).replace('"capture_id":"x"', '"capture_id":[]').encode() + b'\n',
            json.dumps(start).replace('"initial_inventory":{"state":"empty","object_count":0}',
                                      '"initial_inventory":[]').encode() + b'\n',
            json.dumps(start).replace('"capture_start_monotonic_ns":1',
                                      '"capture_start_monotonic_ns":1e400').encode() + b'\n',
        ]
        for index, line in enumerate(malformed):
            path = Path(self.base + f".bad{index}")
            path.write_bytes(line)
            try:
                report = rt.validate_replay_trace([path])
            except (TypeError, AttributeError, RecursionError) as exc:
                self.fail(f"validator exposed {type(exc).__name__}: {exc}")
            self.assertFalse(report["complete_for_replay"], report)
        for mutate, expected in (
                (lambda s: s.update(schema_version=True), "unsupported_schema_version"),
                (lambda s: s.update(generation=1), "invalid_initial_generation"),
                (lambda s: s.update(capture_start_monotonic_ns=2), "capture_start_changed")):
            broken_start = copy.deepcopy(start)
            mutate(broken_start)
            self.assert_capture([broken_start, end], complete=False, needed_error=expected)
        for field in ("capture_end_epoch_ns", "capture_end_monotonic_ns",
                      "durability", "initial_inventory_state_at_end"):
            broken_end = copy.deepcopy(end)
            broken_end.pop(field)
            self.assert_capture([start, broken_end], complete=False)

    def test_rotation_state_and_per_file_sequence_order_fail_closed(self):
        writer = self.writer(max_bytes=1100)
        for index in range(8):
            self.one_event(writer, f"{index + 30:064x}")
        paths = writer.paths
        writer.close()
        records = _records(paths)
        self.assertGreater(sum(r["record_type"] == "rotation" for r in records), 0)
        rotation_index = next(i for i, r in enumerate(records)
                              if r["record_type"] == "rotation")
        for broken_records in (
                records[:rotation_index + 1] + records[rotation_index + 2:],
                records[:rotation_index] + records[rotation_index + 1:]):
            broken_paths = [Path(self.base + ".broken")]
            broken_paths[0].write_text("".join(json.dumps(r) + "\n" for r in broken_records))
            report = rt.validate_replay_trace(broken_paths)
            self.assertFalse(report["complete_for_replay"], report)
        reversed_single = [records[0], records[2], records[1], *records[3:]]
        path = Path(self.base + ".reordered")
        path.write_text("".join(json.dumps(r) + "\n" for r in reversed_single))
        report = rt.validate_replay_trace([path])
        self.assertFalse(report["complete_for_replay"])
        self.assertTrue(any(e.startswith("file_sequence_not_increasing")
                            for e in report["errors"]))

    def test_positive_native_known_empty_fixture_when_supplied(self):
        fixture = os.environ.get("DFKV_REPLAY_TRACE_NATIVE_EMPTY_FIXTURE")
        if not fixture:
            self.skipTest("set DFKV_REPLAY_TRACE_NATIVE_EMPTY_FIXTURE to native test JSONL")
        paths = sorted(Path(fixture).glob("*.jsonl"))
        self.assertTrue(paths)
        report = rt.validate_replay_trace(paths)
        self.assertTrue(report["complete_for_replay"], report)
        operations = [(record["operation"], record["items"][0]["payload"]["kind"])
                      for record in _records(paths)
                      if record["record_type"] == "batch_io" and record["items"]]
        for operation in ("put", "get", "get_auto", "exist", "retain", "remove"):
            self.assertIn(operation, {item[0] for item in operations})
        self.assertIn("scatter_gather_put", {item[1] for item in operations})
        self.assertIn("scatter_gather_get", {item[1] for item in operations})

    def test_native_fixture_when_supplied_is_parsed_not_fabricated(self):
        fixture = os.environ.get("DFKV_REPLAY_TRACE_NATIVE_FIXTURE")
        if not fixture:
            self.skipTest("set DFKV_REPLAY_TRACE_NATIVE_FIXTURE to native test JSONL")
        paths = sorted(Path(fixture).glob("*.jsonl"))
        self.assertTrue(paths)
        report = rt.validate_replay_trace(paths)
        self.assertFalse(report["complete_for_replay"], report)
        self.assertEqual(report["errors"], ["initial_inventory_unknown"], report)
        operations = {record["operation"] for record in _records(paths)
                      if record["record_type"] == "batch_io"}
        self.assertEqual(operations, {"put", "get", "get_auto", "exist", "retain", "remove"})
        payloads = [item["payload"]["kind"] for record in _records(paths)
                    if record["record_type"] == "batch_io" for item in record["items"]]
        self.assertIn("scatter_gather_put", payloads)
        self.assertIn("scatter_gather_get", payloads)


class _Metrics:
    def __init__(self):
        self.calls = []

    def on_exists(self, **kwargs):
        self.calls.append(("exists", kwargs))

    def on_exists_v2(self, **kwargs):
        self.calls.append(("exists_v2", kwargs))


class _ExistLib:
    def __init__(self, present):
        self.present = set(present)
        self.calls = []

    def dfkv_batch_exist(self, _handle, keys, lengths, count, out):
        call = [ctypes.string_at(keys[index], lengths[index]) for index in range(count)]
        self.calls.append(call)
        for index, key in enumerate(call):
            out[index] = int(key in self.present)
        return 0


def _dummy_connector(writer, namespace=b"connector-prefix-test"):
    connector = object.__new__(dfkv_hicache.DfkvHiCache)
    connector.tp_size = 2
    connector.tp_rank = 1
    connector.is_mla = True
    connector.pp_size = 1
    connector.pp_rank = 0
    connector.enable_pp = False
    connector.pcp_size = 1
    connector.pcp_rank = 0
    connector.dcp_size = 1
    connector.dcp_rank = 0
    connector._storage_pcp_rank = 0
    connector.prefill_cp_storage_layout = "rank"
    connector._bind_runtime_layout = False
    connector._device_direct_requested = False
    connector._device_registration_failed = False
    connector._has_retain = False
    connector._replay_trace = writer
    connector._alog_tag = "r1"
    connector._metrics = _Metrics()
    connector._h = 1
    connector._log_exist_probe = lambda *args: None
    connector._pool_component_names = {}
    connector._pool_replicated = {}
    connector._sg_chunks = {}
    connector._key_namespace = namespace
    return connector


class ConnectorPrefixTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="dfkv_sto18_connector_")
        self.addCleanup(self.tmp.cleanup)
        self.base = os.path.join(self.tmp.name, "prefix.jsonl")
        self.namespace = b"connector-prefix-test"
        self.keys = [f"{value:064x}" for value in (11, 12, 13)]

    def writer(self):
        return rt.ReplayTraceWriter(
            self.base, role="connector-prefix", key_namespace=self.namespace,
            coordinates={"tp_size": 2, "tp_rank": 1, "pcp_size": 1, "pcp_rank": 0,
                         "dcp_size": 1, "dcp_rank": 0, "pp_size": 1, "pp_rank": 0},
            initial_inventory_empty=True)

    def test_real_batch_exists_and_v2_preserve_pipeline_and_exact_paths(self):
        writer = self.writer()
        connector = _dummy_connector(writer, self.namespace)
        connector._pool_component_names = {"mamba": ("temporal", "conv")}
        connector._pool_replicated = {"mamba": False}
        all_keys = [connector._keys(key)[0] for key in self.keys]
        side_keys = [connector._pool_keys("mamba", key)[component]
                     for key in self.keys[:2] for component in (0, 1)]
        present = set(all_keys[:2] + side_keys[:3])
        connector._lib = _ExistLib(present)
        logged = []

        @contextlib.contextmanager
        def capture_access(operation, _message):
            record = type("AccessResult", (), {"result": None})()
            try:
                yield record
            finally:
                logged.append((operation, record.result))

        with mock.patch.object(dfkv_hicache, "access_log", capture_access):
            primary = connector.batch_exists(self.keys)
            transfer = PoolTransfer(name="mamba", keys=self.keys[:2],
                                    hit_policy=PoolHitPolicy.ALL_PAGES)
            combined = connector.batch_exists_v2(self.keys, [transfer])
        self.assertEqual(primary, 2)
        self.assertEqual(combined.kv_hit_pages, 1)  # side pool narrows usable prefix
        self.assertEqual(logged, [("batch_exists", "prefix=2/3"),
                                  ("batch_exists", "prefix=2/3"),
                                  ("batch_exists_v2", "kv=1/3 mamba=1")])
        paths = writer.paths
        writer.close()
        records = _records(paths)
        events = [record for record in records if record["record_type"] == "prefix_exists"]
        # There are three events: standalone, v2 primary, v2 side.
        self.assertEqual(len(events), 3)
        for event in events[:2]:
            self.assertEqual(event["operation"], "batch_exists")
            self.assertEqual([page["input_index"] for page in event["pages"]], [0, 1, 2])
            for page, key in zip(event["pages"], self.keys):
                self.assertEqual(page["logical_page_hash_hex"],
                                 pool_key_hash_field(key)[4:].hex())
                native = page["native_items"][0]
                native_key = connector._keys(key)[0]
                self.assertEqual(native["block_key"],
                                 _independent_block_key(self.namespace, native_key))
                self.assertEqual(native["component"], "all")
                self.assertIsNone(native["sg_group"])
        side = events[2]
        self.assertEqual(side["operation"], "batch_exists_v2_pool")
        self.assertEqual(side["probe_operation"], "exist")
        self.assertEqual([page["input_index"] for page in side["pages"]], [0, 1])
        self.assertEqual([item["component"] for page in side["pages"]
                          for item in page["native_items"]],
                         ["temporal", "conv", "temporal", "conv"])
        self.assertEqual([item["present"] for page in side["pages"]
                          for item in page["native_items"]],
                         [True, True, True, False])
        self.assertEqual(side["result"]["pool_prefix_boundary"], 1)
        self.assertEqual(side["result"]["final_usable_prefix"], 1)
        self.assertEqual(side["result"]["hit_policy"], PoolHitPolicy.ALL_PAGES)
        report = rt.validate_replay_trace(paths)
        self.assertTrue(report["complete_for_replay"], report)
        self.assertEqual(report["prefix_event_count"], 3)

    def test_empty_retain_probe_makes_no_native_attempt(self):
        writer = self.writer()
        connector = _dummy_connector(writer, self.namespace)
        connector._has_retain = True
        connector._lib = _ExistLib(set())
        with mock.patch.object(dfkv_hicache.dfkv_cabi, "call_batch_retain") as retain:
            self.assertEqual(connector.batch_exists([]), 0)
        retain.assert_not_called()
        self.assertEqual(connector._lib.calls, [])
        paths = writer.paths
        writer.close()
        event = next(record for record in _records(paths)
                     if record["record_type"] == "prefix_exists")
        self.assertEqual(event["pages"], [])
        self.assertFalse(event["refresh"]["attempted"])
        self.assertTrue(rt.validate_replay_trace(paths)["complete_for_replay"])

    def test_v2_refresh_is_separate_from_nonpromoting_probe(self):
        writer = self.writer()
        connector = _dummy_connector(writer, self.namespace)
        connector._pool_component_names = {"mamba": ("temporal", "conv")}
        connector._pool_replicated = {"mamba": False}
        connector._has_retain = True
        side_keys = [connector._pool_keys("mamba", key)[component]
                     for key in self.keys[:1] for component in (0, 1)]
        connector._lib = _ExistLib(set(side_keys))
        transfer = PoolTransfer(name="mamba", keys=self.keys[:1],
                                hit_policy=PoolHitPolicy.ALL_PAGES)
        refresh_results = [[2, 0, 0], [3, 3]]
        with mock.patch.object(dfkv_hicache.dfkv_cabi, "call_batch_retain",
                               side_effect=refresh_results) as retain:
            combined = connector.batch_exists_v2(self.keys, [transfer])
        self.assertEqual(combined.kv_hit_pages, 1)
        self.assertEqual(retain.call_count, 2)  # primary probe + judged side refresh
        paths = writer.paths
        writer.close()
        events = [record for record in _records(paths)
                  if record["record_type"] == "prefix_exists"]
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["probe_operation"], "retain_refresh")
        self.assertEqual(events[1]["probe_operation"], "exist")
        self.assertEqual(events[1]["refresh"]["mode"], "refresh")
        self.assertEqual(events[1]["refresh"]["ttl_s"], 0)
        self.assertTrue(events[1]["refresh"]["attempted"])
        self.assertEqual(events[1]["refresh"]["native_item_results"],
                         "not_available_at_connector_boundary")
        report = rt.validate_replay_trace(paths)
        self.assertTrue(report["complete_for_replay"], report)

    def test_real_sg_probe_tags_both_groups_without_payload_or_parent_invention(self):
        writer = self.writer()
        connector = _dummy_connector(writer, self.namespace)
        connector.mem_pool_device = object()
        connector._device_direct_requested = True
        connector._sg_width_cache = 29
        connector._sg_chunks = {"kv": 2}
        keys = self.keys[:2]
        sks = [sk for key in keys for sk in connector._stored_keys("kv", key)]
        connector._lib = _ExistLib(set(sks[:3]))
        self.assertEqual(connector.batch_exists(keys), 0)
        paths = writer.paths
        writer.close()
        event = next(record for record in _records(paths)
                     if record["record_type"] == "prefix_exists")
        self.assertEqual([item["sg_group"] for page in event["pages"]
                          for item in page["native_items"]], [0, 1, 0, 1])
        self.assertTrue(all(item["sg_width"] == 29 for page in event["pages"]
                            for item in page["native_items"]))
        self.assertEqual([item["present"] for page in event["pages"]
                          for item in page["native_items"]],
                         [True, True, True, False])
        self.assertEqual(event["result"]["contiguous_prefix"], 0)
        self.assertEqual(event["parent"]["state"], "unknown")


class _OpenOnlyLib:
    def dfkv_open_v2(self, *_args):
        return ctypes.c_void_p(0xBEEF)

    def dfkv_transport_mode(self, *_args):
        return b"tcp"

    def dfkv_close(self, *_args):
        return None

    def dfkv_version(self, *_args):
        return b"0.0.0"


class ConnectorTraceConstructionTest(unittest.TestCase):
    def test_nonpositive_dp_size_matches_namespace_and_trace_coordinates(self):
        parallel = types.SimpleNamespace(
            tp_rank=0, tp_size=1, attn_tp_rank=0, attn_tp_size=1,
            attn_dp_rank=0, attn_cp_size=1, attn_cp_rank=0)
        cfg = dfkv_hicache.HiCacheStorageConfig(
            tp_rank=0, tp_size=1, is_mla_model=False, model_name="dp-trace",
            extra_config={"members": "n0=127.0.0.1:1", "interface_v1": 1,
                          "client_stats_poll_s": 0, "dp_size": 0})
        trace_coordinates = []

        def capture_trace(**kwargs):
            trace_coordinates.append(kwargs["coordinates"])
            return None

        with mock.patch.object(dfkv_hicache, "_load_lib", return_value=_OpenOnlyLib()), \
                mock.patch.object(dfkv_hicache, "_probe_parallel_context",
                                  return_value=(parallel, None)), \
                mock.patch.object(dfkv_hicache, "_ReplayTraceWriter") as writer_type, \
                mock.patch.object(dfkv_hicache, "canonical_namespace",
                                  wraps=dfkv_hicache.canonical_namespace) as namespace:
            writer_type.from_env.side_effect = capture_trace
            connector = dfkv_hicache.DfkvHiCache(cfg, cfg.extra_config)
        self.assertEqual(trace_coordinates[0]["dp_size"], 1)
        self.assertEqual(namespace.call_args.kwargs["dp_size"], 1)
        self.assertIsNone(connector._replay_trace)


if __name__ == "__main__":
    unittest.main()
