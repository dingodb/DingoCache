"""Regression tests for exact hybrid-pool GPU segment geometry."""

import sys
import struct
from pathlib import Path
from types import SimpleNamespace

import torch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheGroupSpec,
)


from dfkv_vllm.data import (  # noqa: E402
    ChunkedTokenDatabase,
    KeyMetadata,
    PoolKey,
    split_block_contiguous_runs,
)
from dfkv_vllm import worker as worker_module  # noqa: E402
from dfkv_vllm.worker import DfkvStoreWorker  # noqa: E402

_METADATA = KeyMetadata(
    model_name="model",
    dp_size=1,
    dp_rank=-1,
    tp_size=1,
    tp_rank=0,
    pcp_size=1,
    pcp_rank=0,
    dcp_size=1,
    dcp_rank=0,
    pp_size=1,
    pp_rank=0,
)


def _db() -> ChunkedTokenDatabase:
    return ChunkedTokenDatabase(_METADATA, block_size=16)

def test_logical_block_ids_expand_compact_stateful_table():
    assert worker_module._logical_block_ids([False, False, True], [7]) == [
        -1,
        -1,
        7,
    ]
    assert worker_module._logical_block_ids([True, True], [5, 11]) == [5, 11]
    with pytest.raises(ValueError, match="1 ids for 2 selected state slots"):
        worker_module._logical_block_ids([True, True], [5])


def test_pool_key_uses_cross_runtime_binary_schema():
    assert PoolKey(_METADATA, "0123abcd").to_bytes() == (
        b"DFKVPOOL\x02"
        + struct.pack("<I", 2) + b"kv"
        + struct.pack("<I", 8) + b"0123abcd"
        + struct.pack(
            "<IiIiIiIiIiI",
            1, -1, 1, 0, 1, 0, 1, 0, 1, 0, 0,
        )
        + struct.pack("<I", 3) + b"all"
    )


def test_pool_key_preserves_embedded_nul_and_non_utf8_hash_bytes():
    binary = PoolKey(_METADATA, b"a\x00\xff").to_bytes()
    shorter = PoolKey(_METADATA, b"a").to_bytes()
    assert binary != shorter
    assert struct.pack("<I", 3) + b"a\x00\xff" in binary


def test_binary_key_log_label_is_digest_only():
    label = worker_module._key_label(b"\xff\x00raw-secret")
    assert label.startswith("len=12 sha256=")
    assert len(label.removeprefix("len=12 sha256=")) == 16
    assert "raw-secret" not in label


def test_prepare_value_uses_actual_noncontiguous_block_ids():
    db = _db()
    db.set_seg_layout(
        [
            (0x1000, 1000, 100),
            (0x2000, 1000, 200),
        ]
    )

    addrs, sizes, first_block_id = db.prepare_value(0, 32, [5, 11])

    assert addrs == [
        0x1000 + 5 * 1000,
        0x1000 + 11 * 1000,
        0x2000 + 5 * 1000,
        0x2000 + 11 * 1000,
    ]
    assert sizes == [100, 100, 200, 200]
    assert first_block_id == 5


def test_prepare_value_legacy_tables_use_actual_noncontiguous_block_ids():
    db = _db()
    db.set_kv_caches_base_addr([0x1000])
    db.set_block_len([128])

    addrs, sizes, _ = db.prepare_value(0, 32, [5, 11])

    assert addrs == [0x1000 + 5 * 128, 0x1000 + 11 * 128]
    assert sizes == [128, 128]


@pytest.mark.parametrize("start,end", [(1, 17), (0, 17), (16, 16), (-16, 0)])
def test_prepare_value_rejects_unaligned_or_empty_ranges(start: int, end: int):
    db = _db()
    db.set_seg_layout([(0x1000, 1000, 100)])

    with pytest.raises(ValueError, match="must align"):
        db.prepare_value(start, end, [5, 11])


def test_prepare_value_rejects_short_block_table():
    db = _db()
    db.set_seg_layout([(0x1000, 1000, 100)])

    with pytest.raises(ValueError, match="block table has 1 ids"):
        db.prepare_value(0, 32, [5])


def test_contiguous_block_is_one_run():
    assert split_block_contiguous_runs((8, 2, 4), (8, 4, 1), 1) == (
        8,
        [(0, 8)],
    )


def test_padded_rows_are_split_without_copying_padding():
    assert split_block_contiguous_runs((8, 2, 4), (100, 10, 1), 1) == (
        100,
        [(0, 4), (10, 4)],
    )


def test_transposed_rows_are_split_in_logical_order():
    assert split_block_contiguous_runs((8, 2, 2), (100, 1, 10), 1) == (
        100,
        [(0, 1), (10, 1), (1, 1), (11, 1)],
    )


def test_overlapping_or_cross_block_runs_fail_closed():
    with pytest.raises(ValueError, match="overlap or exceed"):
        split_block_contiguous_runs((8, 2, 4), (6, 4, 1), 1)


def test_invalid_segment_layout_fails_closed():
    db = _db()
    with pytest.raises(ValueError, match="invalid seg layout"):
        db.set_seg_layout([(-1, 100, 10)])
    with pytest.raises(ValueError, match="invalid seg layout"):
        db.set_seg_layout([(0x1000, 10, 11)])



def test_cross_layer_registration_uses_real_group_zero_layer(monkeypatch):
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float16,
    )
    groups = [
        KVCacheGroupSpec(["model.layers.0.self_attn"], spec),
    ]
    db = ChunkedTokenDatabase(_METADATA, block_size=16)
    registered = []

    class FakeClient:
        def register_memory(self, base, size):
            registered.append((base, size))

    class FakeRecvThread:
        def __init__(self, *args, **_kwargs):
            self._ready = args[5]

        def start(self):
            self._ready.set()

    monkeypatch.setattr(
        worker_module, "KVCacheStoreRecvingThread", FakeRecvThread
    )
    worker = DfkvStoreWorker.__new__(DfkvStoreWorker)
    worker._kv_cache_groups = groups
    worker.token_dbs = [db]
    worker.cache_config = SimpleNamespace(num_gpu_blocks=8)
    worker._kv_pool_regions = []
    worker.client = FakeClient()
    worker.kv_role = "kv_consumer"
    worker.coord = object()
    worker.block_size = 16
    worker.tp_rank = 0
    worker.kv_recv_thread = None

    unified = torch.empty((8, 2, 4), dtype=torch.float16)
    worker.register_cross_layers_kv_caches(unified)

    region_size = unified.untyped_storage().nbytes()
    block_size = region_size // 8
    assert registered == [(unified.untyped_storage().data_ptr(), region_size)]
    assert db.kv_caches_base_addr == [unified.data_ptr()]
    assert db.block_len == [block_size]
    assert db._seg_layout == [(unified.data_ptr(), block_size, block_size)]


def test_cross_layer_registration_rejects_group_without_layer_names():
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float16,
    )
    worker = DfkvStoreWorker.__new__(DfkvStoreWorker)
    worker._kv_cache_groups = [KVCacheGroupSpec([], spec)]

    with pytest.raises(RuntimeError, match="group 0 has no real layer names"):
        worker.register_cross_layers_kv_caches(
            torch.empty((8, 2, 4), dtype=torch.float16)
        )