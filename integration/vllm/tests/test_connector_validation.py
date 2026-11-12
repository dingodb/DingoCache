"""Startup validation for vLLM cache-group layouts."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec  # noqa: E402

from dfkv_vllm.connector import DfkvStoreConnector  # noqa: E402

try:
    from vllm.config.kv_transfer import KVCacheAccess
except ImportError:  # vLLM before #66605
    KVCacheAccess = None


def test_accepts_align_mamba_groups_with_distinct_block_size():
    spec = MambaSpec(
        block_size=400,
        shapes=((1,),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=4, mamba_cache_mode="align")
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[KVCacheGroupSpec(["model.layers.0.mamba"], spec)]
    )

    DfkvStoreConnector._validate_kv_cache_config(
        vllm_config, kv_cache_config
    )


def test_refuses_sleep_mode():
    # GPUDirect memory registrations cannot follow sleep/wake page remaps.
    assert DfkvStoreConnector.supports_sleep_mode(SimpleNamespace()) is False


@pytest.mark.skipif(KVCacheAccess is None, reason="vLLM predates KVCacheAccess")
def test_declares_kv_cache_registration():
    # vLLM then backs each extensible KV cache segment with one physical chunk,
    # which ibv_reg_mr needs.
    access = DfkvStoreConnector.kv_cache_access(SimpleNamespace())
    assert access == KVCacheAccess.REGISTERS
    assert access.needs_single_chunk()


def test_refuses_to_release_registered_kv_cache_memory():
    connector = object.__new__(DfkvStoreConnector)
    with pytest.raises(RuntimeError, match="registrations cannot follow"):
        connector.unregister_kv_caches()
