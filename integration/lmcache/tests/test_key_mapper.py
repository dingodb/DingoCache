# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LMCache's binary dfkv object-key mapping."""

from __future__ import annotations

import importlib.util
import struct
import os
import sys
import types

# Stub lmcache.utils only when LMCache is genuinely unavailable. Keeping an
# installed package intact makes this module safe to collect with the adapter
# tests in one pytest process.
try:
    _HAS_LMCACHE = importlib.util.find_spec("lmcache.utils") is not None
except (ImportError, ModuleNotFoundError, ValueError):
    _HAS_LMCACHE = False

if not _HAS_LMCACHE:
    sys.modules.setdefault("lmcache", types.ModuleType("lmcache"))
    _stub = types.ModuleType("lmcache.utils")

    class CacheEngineKey:
        pass

    _stub.CacheEngineKey = CacheEngineKey
    sys.modules["lmcache.utils"] = _stub

# Load key_mapper.py directly by path (bypassing the dfkv_connector package
# __init__, which pulls in adapter → lmcache.logging and would need the full
# lmcache install). key_mapper has no other dfkv_connector-internal deps.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "_dfkv_key_mapper_under_test",
    os.path.join(_HERE, "..", "src", "dfkv_connector", "key_mapper.py"),
)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
cache_engine_key_to_dfkv_key = _MOD.cache_engine_key_to_dfkv_key


def _base_key(model="glm-5.1", world_size=8, worker_id=0, chunk_hash=0xABCDEF):
    """A base CacheEngineKey stand-in (no layer_id attribute)."""
    return types.SimpleNamespace(
        model_name=model,
        world_size=world_size,
        worker_id=worker_id,
        chunk_hash=chunk_hash,
    )


def _layer_key(
    model="glm-5.1", world_size=8, worker_id=0, chunk_hash=0xABCDEF, layer_id=0
):
    """A LayerCacheEngineKey stand-in (has layer_id)."""
    k = _base_key(model, world_size, worker_id, chunk_hash)
    k.layer_id = layer_id
    return k


def _decode_key(key: bytes):
    assert key.startswith(b"DFKVPOOL\x02")
    offset = len(b"DFKVPOOL\x02")

    def field():
        nonlocal offset
        size = struct.unpack_from("<I", key, offset)[0]
        offset += 4
        value = key[offset : offset + size]
        offset += size
        return value

    pool = field()
    page_hash = field()
    axes = struct.unpack_from("<IiIiIiIiIi", key, offset)
    offset += struct.calcsize("<IiIiIiIiIi")
    group = struct.unpack_from("<I", key, offset)[0]
    offset += 4
    component = field()
    assert offset == len(key)
    return pool, page_hash, axes, group, component


def test_base_key_canonical_golden_format():
    key = cache_engine_key_to_dfkv_key(_base_key(worker_id=2, chunk_hash=0x1234))
    assert _decode_key(key) == (
        b"kv",
        b"1234",
        (1, -1, 8, 2, 1, 0, 1, 0, 1, 0),
        0,
        b"all",
    )


def test_base_key_full_hash_preserved():
    key = cache_engine_key_to_dfkv_key(_base_key(chunk_hash=0xDEADBEEFCAFE))
    assert _decode_key(key)[1] == b"deadbeefcafe"


def test_layer_key_has_explicit_component():
    key = cache_engine_key_to_dfkv_key(_layer_key(chunk_hash=0x99, layer_id=7))
    assert _decode_key(key)[4] == b"layer7"


def test_layer_keys_same_chunk_different_layers_distinct():
    # THE BUG THIS FIXES: under use_layerwise=True, LMCache splits one chunk
    # into num_layers per-layer keys (same chunk_hash + worker, distinct
    # layer_id). Before the fix, these all collapsed to one dfkv key and
    # overwrote each other. They must now be distinct.
    keys = [_layer_key(chunk_hash=0x100, layer_id=L) for L in range(8)]
    binary_keys = [cache_engine_key_to_dfkv_key(k) for k in keys]
    assert len(set(binary_keys)) == 8
    for layer, key in enumerate(keys):
        assert _decode_key(cache_engine_key_to_dfkv_key(key))[4] == (
            f"layer{layer}".encode()
        )


def test_base_and_layer_zero_distinct():
    # A base key (no layer_id) and a layer key with layer_id=0 must NOT be equal:
    # the base key represents the whole-chunk blob (non-layerwise), the layer=0
    # key represents just layer 0's blob (layerwise). Encoding layer_id=0 makes
    # them distinct, which is correct — they are different objects.
    base = _base_key(chunk_hash=0x200)
    layer0 = _layer_key(chunk_hash=0x200, layer_id=0)
    assert base.chunk_hash == layer0.chunk_hash
    assert _decode_key(cache_engine_key_to_dfkv_key(base))[4] == b"all"
    assert _decode_key(cache_engine_key_to_dfkv_key(layer0))[4] == b"layer0"
    assert cache_engine_key_to_dfkv_key(base) != cache_engine_key_to_dfkv_key(layer0)


def test_layer_id_zero_is_encoded():
    key = cache_engine_key_to_dfkv_key(_layer_key(layer_id=0))
    assert _decode_key(key)[4] == b"layer0"


def test_worker_rank_is_preserved():
    key = cache_engine_key_to_dfkv_key(_base_key(worker_id=5, chunk_hash=0xABC))
    assert _decode_key(key)[2][3] == 5


def test_distinct_worker_ranks_stay_distinct():
    rendered = {
        cache_engine_key_to_dfkv_key(_base_key(worker_id=worker, chunk_hash=0xDEAD))
        for worker in range(8)
    }
    assert len(rendered) == 8


def test_layerwise_key_preserves_worker_rank():
    key = cache_engine_key_to_dfkv_key(
        _layer_key(worker_id=3, chunk_hash=0x1, layer_id=7)
    )
    assert _decode_key(key)[2][3] == 3
    assert _decode_key(key)[4] == b"layer7"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            import traceback

            print(f"FAIL {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
