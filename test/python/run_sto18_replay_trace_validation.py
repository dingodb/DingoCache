#!/usr/bin/env python3
"""Generate real native replay fixtures and validate them with the common parser."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
OPTIONAL_TRACE_KEYS = (
    "DFKV_REPLAY_TRACE_PATH",
    "DFKV_REPLAY_TRACE_TEST_ARTIFACTS",
    "DFKV_REPLAY_TRACE_NATIVE_FIXTURE",
    "DFKV_REPLAY_TRACE_NATIVE_EMPTY_FIXTURE",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_test", type=Path)
    parser.add_argument("--artifacts-dir", type=Path,
                        help="fresh directory for retained fixtures; otherwise a private temporary one")
    args = parser.parse_args()

    executable = args.native_test.expanduser().resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        parser.error("native_test must be an executable file")
    if args.artifacts_dir is None:
        artifacts = Path(tempfile.mkdtemp(prefix="dfkv-sto18-fixtures-"))
        supplied = False
    else:
        artifacts = args.artifacts_dir.expanduser().resolve(strict=False)
        if artifacts.exists():
            parser.error(f"preserve existing artifact directory: {artifacts}")
        artifacts.parent.mkdir(parents=True, exist_ok=True)
        artifacts.mkdir()
        supplied = True

    env = os.environ.copy()
    for key in OPTIONAL_TRACE_KEYS:
        env.pop(key, None)
    env["DFKV_REPLAY_TRACE_TEST_ARTIFACTS"] = str(artifacts)
    common = str(ROOT / "integration" / "common" / "src")
    env["PYTHONPATH"] = common + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    native_library = executable.parent / "libdfkv.so"
    if native_library.is_file():
        env["DFKV_LIB"] = str(native_library)

    def preserve(reason: str) -> None:
        print(f"{reason}; retained native fixtures at {artifacts}", file=sys.stderr)

    success = False
    try:
        native = subprocess.run(
            [str(executable), "--gtest_filter=ReplayTrace.*"],
            cwd=ROOT, env=env, check=False)
        if native.returncode != 0:
            preserve(f"native ReplayTrace tests failed: {native.returncode}")
            return native.returncode
        for label in ("batches", "rotation"):
            directory = artifacts / label
            files = sorted(directory.glob("*.jsonl"))
            if not directory.is_dir() or not files:
                preserve(f"missing required native fixture {label}")
                return 1
            print(f"REQUIRED_NATIVE_FIXTURE {label} path={directory} files={len(files)}")

        env["DFKV_REPLAY_TRACE_NATIVE_FIXTURE"] = str(artifacts / "batches")
        env["DFKV_REPLAY_TRACE_NATIVE_EMPTY_FIXTURE"] = str(artifacts / "rotation")
        validated = subprocess.run(
            [sys.executable, str(ROOT / "test/python/test_sto18_replay_trace.py"), "-v"],
            cwd=ROOT, env=env, check=False)
        if validated.returncode != 0:
            preserve(f"native-to-Python replay validation failed: {validated.returncode}")
            return validated.returncode
        print("NATIVE_TO_PYTHON_REPLAY_VALIDATION=PASS")
        success = True
        return 0
    finally:
        if not supplied and success:
            # Only an output root created uniquely by this invocation is disposable.
            shutil.rmtree(artifacts, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
