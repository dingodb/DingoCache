"""Which cache groups' SAVE must finish inside the model step.

An asynchronous PUT reads its source blocks after the step returns. That is
safe only for blocks vLLM keeps with the request until it finishes. The
decision used to be an allow-by-default name list (Mamba, or a spec with a
``sliding_window`` / ``attention_chunk_size`` / ``rswa_window`` field): a new
spec type that frees or rewrites blocks while the request runs, unknown to
the list, was saved asynchronously. It is now default deny. Runs on the
stand-ins of test_dfkv_vllm_worker, with spec classes of its own.
"""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import dataclass
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_dfkv_vllm_worker  # noqa: E402,F401  (installs the stand-ins)

from dfkv_vllm import worker as worker_module  # noqa: E402


@dataclass(frozen=True)
class FullAttentionSpec:
    block_size: int = 16
    sliding_window: int | None = None
    attention_chunk_size: int | None = None


@dataclass(frozen=True)
class MLAAttentionSpec(FullAttentionSpec):
    pass


@dataclass(frozen=True)
class SinkFullAttentionSpec(FullAttentionSpec):
    """A subclass with a manager of its own."""
    sink_len: int = 16


@dataclass(frozen=True)
class SomeNewSpec:
    """A spec type this connector has never seen, with none of the window
    fields the old list looked for."""
    block_size: int = 16


@dataclass(frozen=True)
class MambaSpec:
    block_size: int = 16


class UniformTypeKVCacheSpecs:
    def __init__(self, **specs) -> None:
        self.kv_cache_specs = specs


class SaveFenceIsDefaultDeny(unittest.TestCase):
    def setUp(self) -> None:
        for name, value in (
            ("FullAttentionSpec", FullAttentionSpec),
            ("MLAAttentionSpec", MLAAttentionSpec),
            ("MambaSpec", MambaSpec),
            ("UniformTypeKVCacheSpecs", UniformTypeKVCacheSpecs),
            ("_SAVE_ASYNC_SPEC_TYPES", (FullAttentionSpec, MLAAttentionSpec)),
        ):
            p = patch.object(worker_module, name, value)
            p.start()
            self.addCleanup(p.stop)

    def fenced(self, spec) -> bool:
        return worker_module._save_needs_step_fence(spec)

    def test_full_attention_is_saved_asynchronously(self) -> None:
        self.assertFalse(self.fenced(FullAttentionSpec()))
        self.assertFalse(self.fenced(MLAAttentionSpec()))
        self.assertFalse(self.fenced(UniformTypeKVCacheSpecs(
            a=FullAttentionSpec(), b=MLAAttentionSpec())))

    def test_known_recycling_specs_are_fenced(self) -> None:
        self.assertTrue(self.fenced(MambaSpec()))
        self.assertTrue(self.fenced(FullAttentionSpec(sliding_window=128)))
        self.assertTrue(self.fenced(FullAttentionSpec(attention_chunk_size=64)))

    def test_an_unknown_spec_type_is_fenced(self) -> None:
        self.assertTrue(self.fenced(SomeNewSpec()))
        self.assertTrue(self.fenced(SinkFullAttentionSpec()))
        self.assertTrue(self.fenced(UniformTypeKVCacheSpecs(
            a=FullAttentionSpec(), b=SomeNewSpec())))


if __name__ == "__main__":
    unittest.main(verbosity=2)
