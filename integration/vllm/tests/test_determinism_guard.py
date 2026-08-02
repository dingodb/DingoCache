"""Unit tests for the block-hash determinism guard.

``_determinism.py`` has no vllm/torch imports, so unlike this directory's
other tests these run anywhere: the module is loaded by file path to avoid
importing the (vllm-dependent) package __init__.

Run: python3 -m unittest test_determinism_guard  (from this directory)
"""

import importlib.util
import pathlib
import types
import unittest

_MOD_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "src"
    / "dfkv_vllm"
    / "_determinism.py"
)
_spec = importlib.util.spec_from_file_location("dfkv_determinism", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ensure_deterministic_block_hashing = _mod.ensure_deterministic_block_hashing


def _cfg(algo):
    return types.SimpleNamespace(prefix_caching_hash_algo=algo)


class DeterminismGuardTest(unittest.TestCase):

    def test_builtin_raises(self):
        with self.assertRaisesRegex(RuntimeError, "sha256"):
            ensure_deterministic_block_hashing(_cfg("builtin"))

    def test_builtin_rejects_pinned_python_seed(self):
        with self.assertRaisesRegex(RuntimeError, "PYTHONHASHSEED"):
            ensure_deterministic_block_hashing(_cfg("builtin"))

    def test_sha256_family_passes(self):
        ensure_deterministic_block_hashing(_cfg("sha256"))
        ensure_deterministic_block_hashing(_cfg("sha256_cbor_64bit"))

    def test_enum_like_algo_value_passes(self):
        # vLLM may hand an enum whose str() embeds the name.

        class Algo:
            def __str__(self):
                return "HashAlgo.SHA256"

        ensure_deterministic_block_hashing(_cfg(Algo()))

    def test_missing_attr_defaults_to_builtin(self):
        with self.assertRaises(RuntimeError):
            ensure_deterministic_block_hashing(types.SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
