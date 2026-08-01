"""Regression tests for exact hybrid-pool GPU segment geometry."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dfkv_vllm.data import (  # noqa: E402
    ChunkedTokenDatabase,
    KeyMetadata,
    split_block_contiguous_runs,
)

_METADATA = KeyMetadata("model", 0, 0, 0, 0)


def _db() -> ChunkedTokenDatabase:
    return ChunkedTokenDatabase(_METADATA, block_size=16)


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
