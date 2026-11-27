#!/usr/bin/env python3
"""CPU-only controls for the real storage activation driver.

Temporary files, marker names and metric strings here test the harness gates;
they are NOT release-produced stores and are NOT native acceptance evidence.
No dfkv daemon, etcd, native library, model, GPU or third-party package is used.
Only the process tests launch real short-lived owned Python children.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
import types
import unittest


MODULE_PATH = (Path(__file__).resolve().parents[1] / "e2e" /
               "dfkv_storage_activation.py")
SPEC = importlib.util.spec_from_file_location("dfkv_storage_activation_harness", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HARNESS
SPEC.loader.exec_module(HARNESS)


class FreshRootAndSnapshotTest(unittest.TestCase):
    def test_existing_roots_are_preserved_and_mnt_roles_are_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            existing = parent / "existing"
            existing.mkdir()
            sentinel = existing / "do-not-replace"
            sentinel.write_bytes(b"original")
            with self.assertRaisesRegex(HARNESS.AcceptanceFailure, "preserved"):
                HARNESS.resolve_roots(existing, parent / "cache")
            self.assertEqual(sentinel.read_bytes(), b"original")
            with self.assertRaisesRegex(HARNESS.AcceptanceFailure, "run-root"):
                HARNESS.resolve_roots(Path("/mnt/disk-test-run/root"), parent / "cache-two")

    def test_nested_run_and_cache_roots_are_rejected(self):
        # The run role rejects /mnt/disk* before checking nesting. Do not let
        # a task-scoped TMPDIR on such a mount select that different rejection.
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            parent = Path(temporary)
            with self.assertRaisesRegex(HARNESS.AcceptanceFailure, "independent"):
                HARNESS.resolve_roots(parent / "run", parent / "run" / "cache")
            self.assertFalse((parent / "run").exists())

    def test_main_never_finalizes_into_preexisting_run_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            existing = parent / "existing-run"
            existing.mkdir()
            sentinel = existing / "sentinel"
            sentinel.write_bytes(b"preserve")
            before = sorted(path.name for path in existing.iterdir())
            argv = [
                "--release-dir", str(parent / "release-not-reached"),
                "--candidate-build-dir", str(parent / "candidate-not-reached"),
                "--etcd-binary", str(parent / "etcd-not-reached"),
                "--run-root", str(existing),
                "--cache-root", str(parent / "fresh-cache"),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                status = HARNESS.main(argv)
            self.assertEqual(status, 1)
            self.assertEqual(sorted(path.name for path in existing.iterdir()), before)
            self.assertEqual(sentinel.read_bytes(), b"preserve")
            self.assertFalse((existing / "result.json").exists())
            self.assertFalse((parent / "fresh-cache").exists())

    def test_snapshot_hashes_real_contents_and_detects_same_size_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "store"
            root.mkdir()
            (root / "segments").mkdir()
            payload = root / "segments" / "000001.bin"
            payload.write_bytes(b"payload-A")
            before = HARNESS.snapshot_directory(root)
            payload.write_bytes(b"payload-B")
            after = HARNESS.snapshot_directory(root)
            with self.assertRaisesRegex(HARNESS.AcceptanceFailure, "changed="):
                HARNESS.require_byte_preservation(before, after)

    def test_partial_root_clone_is_not_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "store"
            (root / "segments").mkdir(parents=True)
            (root / "lse_super").write_bytes(b"control-only-marker")
            (root / "lse_journal").write_bytes(b"journal")
            (root / "segments" / "000000.bin").write_bytes(b"whole-value")
            before = HARNESS.snapshot_directory(root)
            (root / "segments" / "000000.bin").unlink()
            after = HARNESS.snapshot_directory(root)
            with self.assertRaisesRegex(HARNESS.AcceptanceFailure, "missing=.*000000"):
                HARNESS.require_byte_preservation(before, after)

    def test_added_conversion_layout_is_rejected_even_without_file_hashing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "store"
            root.mkdir()
            (root / "slab_meta").write_bytes(b"fixture")
            before = HARNESS.snapshot_directory(root, hash_regular_files=False)
            (root / "lse_super").write_bytes(b"unexpected-conversion")
            after = HARNESS.snapshot_directory(root, hash_regular_files=False)
            with self.assertRaisesRegex(HARNESS.AcceptanceFailure, "added=.*lse_super"):
                HARNESS.require_compatible_root_shape(before, after)

    def test_replaced_root_with_identical_files_fails_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "store"
            root.mkdir()
            (root / "value").write_bytes(b"same-bytes")
            before = HARNESS.snapshot_directory(root)
            moved = parent / "moved-original"
            root.rename(moved)
            root.mkdir()
            (root / "value").write_bytes(b"same-bytes")
            after = HARNESS.snapshot_directory(root)
            self.assertNotEqual(before.root.inode, after.root.inode)
            with self.assertRaisesRegex(HARNESS.AcceptanceFailure, "identity changed"):
                HARNESS.require_byte_preservation(before, after)

    def test_symlink_target_is_part_of_snapshot_not_followed(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "store"
            root.mkdir()
            outside = parent / "outside"
            outside.write_bytes(b"outside-must-not-be-read")
            link = root / "link"
            link.symlink_to(outside)
            before = HARNESS.snapshot_directory(root)
            self.assertEqual(before.entries["link"].kind, "symlink")
            self.assertNotIn("outside", before.entries)
            link.unlink()
            link.symlink_to(parent / "different")
            with self.assertRaises(HARNESS.AcceptanceFailure):
                HARNESS.require_byte_preservation(before, HARNESS.snapshot_directory(root))


class MetricAuthorityTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = HARNESS.RootIdentity.read(Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def text(self, engine="log", disk=None, build="2.45.0", extra=""):
        disk = self.root.path if disk is None else disk
        return (
            f"dfkv_disk_engine_info{{disk={json.dumps(disk)},engine={json.dumps(engine)}}} 1\n"
            f"dfkv_build_info{{version={json.dumps(build)},engine={json.dumps(engine)}}} 1\n"
            + extra
        )

    def test_actual_root_engine_and_build_are_bound(self):
        observed = HARNESS.require_engine_observation(self.text(), self.root, "log")
        self.assertEqual(observed["disk"], self.root.path)
        self.assertEqual(observed["engine"], "log")
        self.assertEqual(observed["build"]["version"], "2.45.0")

    def test_live_release_observation_feeds_slab_verdict_shape(self):
        observed = HARNESS.require_engine_observation(
            self.text(engine="slab", build=HARNESS.RELEASE_VERSION),
            self.root, "slab")
        release = dict(observed)
        release.update({"whole_payload_written": True,
                        "whole_payload_read_by_release": True, "client_closed": True})
        candidate = {"engine": "slab", "whole_payload_read": True,
                     "client_closed": True, "root_shape_retained": True}
        self.assertEqual(HARNESS.verdict_old_slab(release, candidate),
                         "PASS_RELEASE_SLAB_RETAINED")

    def test_slab_directory_cannot_be_called_lse(self):
        with self.assertRaisesRegex(HARNESS.AcceptanceFailure, "actual backend"):
            HARNESS.require_engine_observation(self.text(engine="slab"), self.root, "log")

    def test_wrong_root_is_rejected(self):
        other = Path(self.temporary.name) / "other"
        with self.assertRaisesRegex(HARNESS.AcceptanceFailure, "different root"):
            HARNESS.require_engine_observation(self.text(disk=str(other)), self.root, "log")

    def test_zero_engine_sample_does_not_count(self):
        text = self.text().replace("} 1\n", "} 0\n", 1)
        with self.assertRaisesRegex(HARNESS.AcceptanceFailure, "no active"):
            HARNESS.require_engine_observation(text, self.root, "log")

    def test_multiple_active_engines_are_ambiguous_not_winner_selection(self):
        extra = (f"dfkv_disk_engine_info{{disk={json.dumps(self.root.path)},"
                 'engine="slab"} 1\n')
        with self.assertRaisesRegex(HARNESS.AcceptanceFailure, "ambiguous"):
            HARNESS.require_engine_observation(self.text(extra=extra), self.root, "log")

    def test_engine_without_real_build_metadata_fails(self):
        text = self.text().splitlines()[0] + "\n"
        with self.assertRaisesRegex(HARNESS.AcceptanceFailure, "build_info"):
            HARNESS.require_engine_observation(text, self.root, "log")

    def test_escaped_path_labels_are_parsed_exactly(self):
        quoted = Path(self.temporary.name) / 'quote"and\\backslash'
        quoted.mkdir()
        identity = HARNESS.RootIdentity.read(quoted)
        observed = HARNESS.require_engine_observation(
            self.text(disk=str(quoted)), identity, "log")
        self.assertEqual(observed["disk"], str(quoted))


class ArtifactBindingTest(unittest.TestCase):
    def test_public_release_identity_is_complete_and_not_current(self):
        self.assertEqual(HARNESS.RELEASE_VERSION, "2.44.0")
        self.assertEqual(HARNESS.RELEASE_SOURCE_COMMIT,
                         "ef851f2617fdf4bf2ae15747550349eb7ee28dfe")
        self.assertEqual(HARNESS.RELEASE_ASSET_BYTES, 12_864_294)
        self.assertEqual(HARNESS.RELEASE_ASSET_SHA256,
                         "da8d736baa5761d5c504110862e9453ff5d0b425be0d13f026693c7d824f945e")
        self.assertEqual(HARNESS.RELEASE_FILE_SHA256, {
            "dfkv_server": "e5f4cb1e7d1c15e06e1fbf4b525e2e507fa39e3b537c25a4fc75c6d78eb21510",
            "dfkv_mds": "a7c1aab276cf6e6e390cb86dab5b74613de12bde42754f633cd1aad702f72d91",
            "libdfkv.so": "90060a29ab65bbc40dca79fdb9f964668a041b7d16becf0f461924cc04acca9b",
        })
        self.assertEqual(
            HARNESS.RELEASE_ASSET_URL,
            "https://github.com/dingodb/DingoCache/releases/download/v2.44.0/"
            "dfkv-2.44.0-linux-x86_64.tar.gz")
        self.assertNotIn("/root/dfkv-dev/current", HARNESS.RELEASE_ASSET_URL)

    def test_elf_architecture_is_read_from_header(self):
        header = bytearray(64)
        header[:4] = b"\x7fELF"
        header[4:6] = b"\x02\x01"
        struct.pack_into("<HH", header, 16, 3, 62)
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "libfixture.so"
            binary.write_bytes(header)
            metadata = HARNESS.elf_metadata(binary)
            self.assertEqual(metadata["machine"], "x86-64")
            self.assertEqual(metadata["elf_type"], 3)
            struct.pack_into("<H", header, 18, 183)
            binary.write_bytes(header)
            with self.assertRaisesRegex(HARNESS.AcceptanceFailure, "not x86-64"):
                HARNESS.elf_metadata(binary)

    def test_wrong_archive_size_fails_before_extracting_anything(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            archive = parent / HARNESS.RELEASE_ASSET_NAME
            archive.write_bytes(b"not the release")
            destination = parent / "new-release-root"
            with self.assertRaisesRegex(HARNESS.AcceptanceFailure, "wrong byte size"):
                HARNESS.extract_verified_release(archive, destination)
            self.assertFalse(destination.exists())

    def test_exact_size_but_wrong_checksum_cannot_extract(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            archive = parent / HARNESS.RELEASE_ASSET_NAME
            with archive.open("wb") as stream:
                stream.truncate(HARNESS.RELEASE_ASSET_BYTES)
            destination = parent / "new-release-root"
            with self.assertRaisesRegex(HARNESS.AcceptanceFailure, "checksum mismatch"):
                HARNESS.extract_verified_release(archive, destination)
            self.assertFalse(destination.exists())

    def test_ambiguous_flat_and_packaged_binary_layout_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bin").mkdir()
            (root / "dfkv_server").write_bytes(b"flat")
            (root / "bin" / "dfkv_server").write_bytes(b"nested")
            with self.assertRaisesRegex(HARNESS.AcceptanceFailure, "unambiguous"):
                HARNESS._locate(root, "dfkv_server", "bin")


class TraceAndVerdictTest(unittest.TestCase):
    def successful_events(self):
        return ["root_snapshot", "process_spawn", "engine_observed",
                "payload_verified", "client_closed", "process_stopped"]

    def old_slab_success(self):
        return {"build": {"version": HARNESS.RELEASE_VERSION}, "engine": "slab",
                "whole_payload_written": True, "whole_payload_read_by_release": True,
                "client_closed": True}

    def candidate_slab_success(self):
        return {"engine": "slab", "whole_payload_read": True,
                "client_closed": True, "root_shape_retained": True}

    def old_development_success(self):
        return {"build": {"version": HARNESS.RELEASE_VERSION}, "engine": "log",
                "whole_payload_read_by_release": True, "client_closed": True}

    def development_refusal_success(self):
        return {"reason": "development_format", "nonzero_exit": True,
                "health_absent": True, "health_or_ready_observed": False,
                "startup_observation_count": 1, "engine_observed": False,
                "root_preserved_byte_for_byte": True}

    def test_valid_native_open_order_and_old_slab_verdict(self):
        HARNESS.require_open_trace(self.successful_events())
        self.assertEqual(
            HARNESS.verdict_old_slab(self.old_slab_success(), self.candidate_slab_success()),
            "PASS_RELEASE_SLAB_RETAINED")

    def test_injected_constructor_cannot_open_before_classification(self):
        events = ["root_snapshot", "injected_constructor"] + self.successful_events()[1:]
        with self.assertRaisesRegex(HARNESS.AcceptanceFailure, "injected"):
            HARNESS.require_open_trace(events)
        reversed_events = ["process_spawn", "root_snapshot", "engine_observed",
                           "process_stopped"]
        with self.assertRaisesRegex(HARNESS.AcceptanceFailure, "before process"):
            HARNESS.require_open_trace(reversed_events)

    def test_missing_whole_payload_read_cannot_pass_old_slab(self):
        candidate = self.candidate_slab_success()
        candidate["whole_payload_read"] = False
        with self.assertRaises(HARNESS.AcceptanceFailure):
            HARNESS.verdict_old_slab(self.old_slab_success(), candidate)

    def test_missing_payload_event_cannot_be_saved_by_engine_metric(self):
        events = self.successful_events()
        events.remove("payload_verified")
        with self.assertRaisesRegex(HARNESS.AcceptanceFailure, "whole-payload proof"):
            HARNESS.require_open_trace(events)

    def test_development_refusal_is_a_distinct_expected_verdict(self):
        self.assertEqual(
            HARNESS.verdict_development_refusal(
                self.old_development_success(), self.development_refusal_success()),
            "EXPECTED_DEVELOPMENT_FORMAT_REFUSAL")
        HARNESS.require_open_trace(
            ["root_snapshot", "process_spawn", "startup_refusal", "root_snapshot"],
            refusal=True)

    def test_generic_startup_failure_is_not_development_refusal(self):
        refusal = self.development_refusal_success()
        refusal["reason"] = "address_in_use"
        with self.assertRaisesRegex(HARNESS.AcceptanceFailure, "dedicated safe-refusal"):
            HARNESS.verdict_development_refusal(self.old_development_success(), refusal)

    def test_refusal_classifier_requires_root_and_format_on_one_event(self):
        root = Path("/fresh/development-root")
        line = f"disk store startup failed: {root}: log superblock: development format"
        observed = HARNESS.classify_development_refusal(
            line + "\nrefusing startup: local storage initialization failed\n", root)
        self.assertEqual(observed["reason"], "development_format")
        mixed_lines = (f"root {root} could not be listed\n"
                       "unrelated decoder warning: development format\n")
        self.assertEqual(HARNESS.classify_development_refusal(mixed_lines, root)["reason"],
                         "generic_startup_failure")
        warning = f"configuration warning for {root}: development format is documented"
        self.assertEqual(HARNESS.classify_development_refusal(warning, root)["reason"],
                         "generic_startup_failure")

    def test_refusal_classifier_rejects_accepted_then_crashed_logs(self):
        root = Path("/fresh/development-root")
        line = f"disk store startup failed: {root}: development format"
        accepted = "PORT 12345\ndfkv_server listening on port 12345\n" + line + "\n"
        self.assertEqual(HARNESS.classify_development_refusal(accepted, root)["reason"],
                         "generic_startup_failure")
        self.assertTrue(HARNESS.classify_development_refusal(
            accepted, root)["successful_listener_observed"])

    def test_grave_marker_without_old_positive_control_cannot_pass(self):
        old = self.old_development_success()
        old["engine"] = False
        with self.assertRaisesRegex(HARNESS.AcceptanceFailure, "positively opened"):
            HARNESS.verdict_development_refusal(old, self.development_refusal_success())

    def test_staged_refusal_with_root_mutation_cannot_pass(self):
        refusal = self.development_refusal_success()
        refusal["root_preserved_byte_for_byte"] = False
        with self.assertRaises(HARNESS.AcceptanceFailure):
            HARNESS.verdict_development_refusal(self.old_development_success(), refusal)

    def test_refusal_trace_cannot_claim_client_or_payload_success(self):
        events = ["root_snapshot", "process_spawn", "startup_refusal",
                  "payload_verified", "client_closed", "root_snapshot"]
        with self.assertRaisesRegex(HARNESS.AcceptanceFailure, "refused server"):
            HARNESS.require_open_trace(events, refusal=True)

    def test_unclosed_success_phase_fails_trace(self):
        with self.assertRaisesRegex(HARNESS.AcceptanceFailure, "prove stop"):
            HARNESS.require_open_trace(self.successful_events()[:-1])

    def test_selector_scrub_is_real_and_only_old_dev_may_opt_in(self):
        saved = {key: os.environ.get(key) for key in HARNESS.SELECTOR_ENV}
        try:
            os.environ["DFKV_STORE_ENGINE"] = "slab"
            os.environ["DFKV_LSE_EXPERIMENTAL"] = "1"
            clean = HARNESS.engine_env()
            self.assertNotIn("DFKV_STORE_ENGINE", clean)
            self.assertNotIn("DFKV_LSE_EXPERIMENTAL", clean)
            development = HARNESS.engine_env(selector="log")
            self.assertEqual(development["DFKV_STORE_ENGINE"], "log")
            self.assertEqual(development["DFKV_LSE_EXPERIMENTAL"], "1")
            with self.assertRaises(HARNESS.AcceptanceFailure):
                HARNESS.engine_env(selector="slab")
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class OwnedPythonLifetimeTest(unittest.TestCase):
    def python(self, source, root, name="python-child"):
        return HARNESS.OwnedProcess.start(
            name, [sys.executable, "-c", source], cwd=root,
            env=dict(os.environ), log_path=root / f"{name}.log")

    def test_short_lived_owned_python_is_signalled_and_reaped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            supervisor = HARNESS.ProcessSupervisor()
            child = supervisor.add(self.python("import time; time.sleep(60)", root))
            supervisor.check_running(child)
            supervisor.stop(child, graceful_s=1, kill_s=1)
            row = supervisor.cleanup[-1]
            self.assertEqual(row["pid"], child.popen.pid)
            self.assertEqual(row["pgid"], child.popen.pid)
            self.assertIsNotNone(row["returncode"])
            self.assertEqual(row["status"], "terminated")

    def test_detached_descendant_is_found_by_identity_and_stopped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = (
                "import subprocess,sys,time; "
                "subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], "
                "start_new_session=True,stdin=subprocess.DEVNULL,"
                "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
                "time.sleep(60)"
            )
            supervisor = HARNESS.ProcessSupervisor()
            child = supervisor.add(self.python(source, root))
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if len(child.refresh()) >= 2:
                    break
                supervisor.check_running(child)
                time.sleep(0.02)
            self.assertGreaterEqual(len(child.refresh()), 2,
                                    "detached Python grandchild was not observed")
            supervisor.stop(child, graceful_s=1, kill_s=1)
            self.assertEqual(supervisor.cleanup[-1]["status"], "terminated")

    def test_expected_exit_has_exit_code_not_healthy_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child = self.python("import time; time.sleep(0.15); raise SystemExit(7)", root)
            observations = []
            row = child.record_expected_exit(
                timeout_s=3, poll_observation=lambda: observations.append(time.monotonic()))
            self.assertEqual(row["status"], "expected_exit")
            self.assertEqual(row["returncode"], 7)
            self.assertNotEqual(row["returncode"], 0)
            self.assertGreaterEqual(len(observations), 2)

    def test_preexisting_control_daemon_exit_is_not_successful_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            supervisor = HARNESS.ProcessSupervisor()
            child = supervisor.add(self.python(
                "import time; time.sleep(0.15); raise SystemExit(7)", root))
            self.assertEqual(child.popen.wait(timeout=3), 7)
            ring = object.__new__(HARNESS.Ring)
            ring.supervisor = supervisor
            ring.args = types.SimpleNamespace(graceful_stop_s=0.1, kill_stop_s=0.1)
            ring.urls = {}
            ring.sequence = []
            with self.assertRaisesRegex(HARNESS.AcceptanceFailure, "exited unexpectedly"):
                HARNESS.Ring.stop_process(ring, child)
            self.assertEqual(len(supervisor.unexpected_exits), 1)
            self.assertEqual(supervisor.unexpected_exits[0]["returncode"], 7)
            supervisor.stop_all(graceful_s=0.1, kill_s=0.1)
            self.assertFalse(supervisor.cleanup_failures)

    def test_pid_start_ticks_and_pgid_prevent_identity_reuse(self):
        original = HARNESS.ProcIdentity(100, 1, 100, 12345, "S")
        self.assertTrue(original.same_process(
            HARNESS.ProcIdentity(100, 2, 100, 12345, "Z")))
        self.assertFalse(original.same_process(
            HARNESS.ProcIdentity(100, 1, 100, 12346, "S")))
        self.assertFalse(original.same_process(
            HARNESS.ProcIdentity(100, 1, 101, 12345, "S")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
