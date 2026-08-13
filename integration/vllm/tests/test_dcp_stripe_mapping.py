"""DCP store/load stripe identity regression tests.

These tests exercise the canonical rank mapping without the transfer data path,
so they need no GPU or dfkv server.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dfkv_vllm.worker import _key_stripe_identity


def _mla_identity(
    tp_rank: int,
    *,
    pcp_rank: int = 0,
    pcp_size: int = 1,
    dcp_size: int = 1,
):
    dcp_rank = (tp_rank * pcp_size + pcp_rank) % dcp_size
    return _key_stripe_identity(
        tp_rank=tp_rank,
        tp_size=8,
        pcp_rank=pcp_rank,
        pcp_size=pcp_size,
        dcp_rank=dcp_rank,
        dcp_size=dcp_size,
        num_kv_heads=1,
        use_mla=True,
    )


def test_tp8_dcp1_stripe_mapping_is_unchanged():
    mapping = [
        (
            tp_rank,
            identity.tp_rank,
            identity.dcp_rank,
            identity.stripe_idx,
            identity.stripe_step,
        )
        for tp_rank in range(8)
        for identity in [_mla_identity(tp_rank)]
    ]
    assert mapping == [
        (0, -1, 0, 0, 8),
        (1, -1, 0, 1, 8),
        (2, -1, 0, 2, 8),
        (3, -1, 0, 3, 8),
        (4, -1, 0, 4, 8),
        (5, -1, 0, 5, 8),
        (6, -1, 0, 6, 8),
        (7, -1, 0, 7, 8),
    ]


def test_tp8_dcp4_uses_dcp_replica_identity_for_stripes():
    mapping = [
        (
            tp_rank,
            identity.tp_rank,
            identity.dcp_rank,
            identity.stripe_idx,
            identity.stripe_step,
        )
        for tp_rank in range(8)
        for identity in [_mla_identity(tp_rank, dcp_size=4)]
    ]
    assert mapping == [
        (0, 0, 0, 0, 2),
        (1, 0, 1, 0, 2),
        (2, 0, 2, 0, 2),
        (3, 0, 3, 0, 2),
        (4, 0, 0, 1, 2),
        (5, 0, 1, 1, 2),
        (6, 0, 2, 1, 2),
        (7, 0, 3, 1, 2),
    ]


def test_mla_pcp_dcp_store_and_load_key_geometry():
    # (pcp_size, dcp_size, key namespaces, replica stripes per namespace)
    cases = [
        (1, 1, 1, 8),
        (1, 4, 4, 2),
        (2, 1, 2, 8),
        (2, 4, 4, 4),
        (2, 8, 8, 2),
        (4, 2, 4, 8),
    ]
    chunks = range(64)
    for pcp_size, dcp_size, namespace_count, stripe_count in cases:
        stored: dict[tuple[int, int, int], set[int]] = {}
        writes: dict[tuple[tuple[int, int, int], int], int] = {}
        stripes: dict[tuple[int, int, int], set[int]] = {}

        for pcp_rank in range(pcp_size):
            for tp_rank in range(8):
                identity = _mla_identity(
                    tp_rank,
                    pcp_rank=pcp_rank,
                    pcp_size=pcp_size,
                    dcp_size=dcp_size,
                )
                namespace = (identity.tp_rank, pcp_rank, identity.dcp_rank)
                assert identity.stripe_step == stripe_count
                stripes.setdefault(namespace, set()).add(identity.stripe_idx)
                for chunk in chunks:
                    if chunk % identity.stripe_step != identity.stripe_idx:
                        continue
                    stored.setdefault(namespace, set()).add(chunk)
                    write = (namespace, chunk)
                    writes[write] = writes.get(write, 0) + 1

        assert len(stored) == namespace_count
        assert sum(map(len, stored.values())) == namespace_count * len(chunks)
        assert all(count == 1 for count in writes.values())
        assert all(indices == set(range(stripe_count)) for indices in stripes.values())

        # Every hot/load worker derives a namespace whose complete cold/store
        # key set exists, regardless of which replica populated each stripe.
        for pcp_rank in range(pcp_size):
            for tp_rank in range(8):
                identity = _mla_identity(
                    tp_rank,
                    pcp_rank=pcp_rank,
                    pcp_size=pcp_size,
                    dcp_size=dcp_size,
                )
                namespace = (identity.tp_rank, pcp_rank, identity.dcp_rank)
                assert stored[namespace] == set(chunks)
