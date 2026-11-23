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


def _construct(extra):
    """Build the connector up to its extra_config check; a valid config
    reaches the layout check, which is stopped there."""
    from unittest import mock
    from dfkv_vllm import connector as connector_module
    config = SimpleNamespace(kv_transfer_config=SimpleNamespace(
        kv_connector_extra_config=extra, kv_role="kv_both"))
    with mock.patch.object(DfkvStoreConnector, "_validate_kv_cache_config",
                           side_effect=RuntimeError("validated")), \
            mock.patch.object(connector_module.KVConnectorBase_V1, "__init__",
                              return_value=None):
        DfkvStoreConnector(config, None, object())


@pytest.mark.parametrize("key", ["modle_revision", "require_model_revison",
                                 "share_across_dp_size", "tenantid"])
def test_a_misspelt_identity_key_fails_startup(key):
    # Unread, its default would apply: deployments meant to be told apart
    # would share keys.
    with pytest.raises(ValueError, match="did you mean"):
        _construct({key: "x"})


def test_a_non_integer_setting_fails_startup():
    with pytest.raises(ValueError, match="lookup_timeout_ms"):
        _construct({"lookup_timeout_ms": "1.5"})


def test_a_valid_extra_config_passes_the_check():
    with pytest.raises(RuntimeError, match="validated"):
        _construct({"model_revision": "x", "members": "a=b:1",
                    "lookup_timeout_ms": 10000})
