"""Settings that reorder a vLLM object's bytes must reach its namespace.

Two deployments whose objects have the same length but a different byte
order must not share keys: the exact length check on load would pass and
the bytes would land in the wrong places. Runs on the stand-ins of
test_dfkv_vllm_worker (no vLLM or torch needed).
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_dfkv_vllm_worker  # noqa: E402,F401  (installs the stand-ins)

from dfkv_common import canonical_namespace  # noqa: E402
from dfkv_vllm import worker as worker_module  # noqa: E402


def _fields(interleave=1, dcp=1, pcp=1, extra=None):
    return worker_module._byte_order_layout_fields(
        SimpleNamespace(cp_kv_cache_interleave_size=interleave),
        extra or {},
        dcp_size=dcp,
        pcp_size=pcp,
    )


def _namespace(fields):
    return canonical_namespace(
        "model", "vllm-multiwr-v5", layout_fields={"dcp_size": 2, **fields},
    )


class ByteOrderReachesTheNamespace(unittest.TestCase):
    def test_interleave_size_under_context_parallelism(self) -> None:
        # Token i sits on DCP rank i % 2 with interleave 1, on rank
        # (i // 16) % 2 with 16: same block length, other tokens.
        for dcp, pcp in ((2, 1), (1, 2)):
            with self.subTest(dcp=dcp, pcp=pcp):
                self.assertNotEqual(
                    _namespace(_fields(1, dcp, pcp)),
                    _namespace(_fields(16, dcp, pcp)),
                )
                self.assertNotEqual(
                    _namespace(_fields(16, dcp, pcp)),
                    _namespace(_fields(64, dcp, pcp)),
                )

    def test_cross_layer_blocks(self) -> None:
        on = {"enable_cross_layers_blocks": "true"}
        self.assertNotEqual(_namespace(_fields()), _namespace(_fields(extra=on)))
        self.assertEqual(_fields(extra={"enable_cross_layers_blocks": "False"}), {})

    def test_defaults_keep_todays_namespace(self) -> None:
        # Interleave 1, or any interleave without CP, and no cross-layer
        # blocks: no field, so keys stay those written so far.
        self.assertEqual(_fields(), {})
        self.assertEqual(_fields(interleave=1, dcp=4), {})
        self.assertEqual(_fields(interleave=16, dcp=1, pcp=1), {})
        self.assertEqual(_namespace(_fields()), _namespace({}))
        # An engine with only the older dcp_ name: that one is read.
        self.assertEqual(
            worker_module._byte_order_layout_fields(
                SimpleNamespace(dcp_kv_cache_interleave_size=16), {},
                dcp_size=2, pcp_size=1),
            {"cp_kv_cache_interleave_size": 16},
        )
        # A config object without either attribute counts as 1.
        self.assertEqual(
            worker_module._byte_order_layout_fields(
                SimpleNamespace(), {}, dcp_size=2, pcp_size=1),
            {},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
