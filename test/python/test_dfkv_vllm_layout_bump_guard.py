"""Guard: the next vLLM layout-ID bump must bind each group's object size.

The namespace binds the cache geometry through the cache specs
(``group_layout``) but not the byte size of one logical object per group
(``logical_bytes_per_block``), which the connector derives at registration.
Adding it changes every key, so it waits for the next layout-ID bump, which
cold-starts every deployment anyway; until then a length mismatch is caught
on load (and a shorter stale object removed). This test fails when the
layout ID changes without the field in ``layout_fields``. Runs on the
stand-ins of test_dfkv_vllm_worker.
"""

from __future__ import annotations

import ast
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_dfkv_vllm_worker  # noqa: E402,F401  (installs the stand-ins)

from dfkv_vllm.data import VLLM_RAW_LAYOUT  # noqa: E402

WORKER = (
    Path(__file__).resolve().parents[2]
    / "integration" / "vllm" / "src" / "dfkv_vllm" / "worker.py"
)
# The layout ID this decision was taken at.
DECIDED_AT = b"vllm-multiwr-v4"


def _namespace_layout_fields() -> set[str]:
    tree = ast.parse(WORKER.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "canonical_namespace"
        ):
            for keyword in node.keywords:
                if keyword.arg == "layout_fields" and isinstance(keyword.value, ast.Dict):
                    return {
                        k.value for k in keyword.value.keys
                        if isinstance(k, ast.Constant)
                    }
    raise AssertionError("canonical_namespace(layout_fields={...}) not found in worker.py")


class LayoutBumpGuard(unittest.TestCase):
    def test_a_new_layout_id_binds_logical_bytes_per_block(self) -> None:
        fields = _namespace_layout_fields()
        if VLLM_RAW_LAYOUT == DECIDED_AT:
            self.assertNotIn("logical_bytes_per_block", fields)
            return
        self.assertIn(
            "logical_bytes_per_block", fields,
            "the vLLM layout ID changed: put each cache group's "
            "logical_bytes_per_block into layout_fields with it",
        )

    def test_the_namespace_builder_is_found(self) -> None:
        self.assertIn("group_layout", _namespace_layout_fields())


if __name__ == "__main__":
    unittest.main(verbosity=2)
