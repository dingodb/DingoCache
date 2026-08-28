"""Startup validation for vLLM cache-group layouts."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec  # noqa: E402

from dfkv_vllm.connector import DfkvStoreConnector  # noqa: E402


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
