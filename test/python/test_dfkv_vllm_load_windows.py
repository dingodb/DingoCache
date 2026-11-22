"""Windowed LOADs pass the flat descriptor batch to the native call in place.

``load_window_keys`` splits one LOAD into native calls of at most that many
keys. Slicing the flat ``SgDescriptorBatch`` views gave plain lists, so each
window was re-marshalled segment by segment in Python. A window of the batch
now hands the native call the batch's own arrays from the window's first key.
The native call must see exactly what it saw before. Runs on the stand-ins of
test_dfkv_vllm_worker.
"""

from __future__ import annotations

import hashlib
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_dfkv_vllm_worker  # noqa: E402,F401  (installs the stand-ins)

from dfkv_vllm import dfkv_client as dc  # noqa: E402
from dfkv_vllm import worker as worker_module  # noqa: E402


class _Lib:
    """Records what each native GET reads from its descriptor arrays."""

    def __init__(self) -> None:
        self.calls = []

    def dfkv_batch_get_auto_sg(self, _h, karr, klens, parr, carr, narr, n, hit, length):
        call = []
        for i in range(n):
            count = narr[i]
            call.append((
                [parr[i][j] for j in range(count)],
                [carr[i][j] for j in range(count)],
            ))
            hit[i] = 1
            length[i] = sum(carr[i][j] for j in range(count))
        self.calls.append(call)
        return 0


def _client():
    client = object.__new__(dc.DfkvDeviceClient)
    client._lib = _Lib()
    client._h = 1
    return client


def _batch(counts):
    batch = dc.SgDescriptorBatch(counts, [64 * c for c in counts])
    for i in range(batch.num_segments):
        batch._pointers[i] = 0x10000 + 0x40 * i
        batch._sizes[i] = 64
    return batch


def _keys(n):
    return [hashlib.sha256(b"%d" % i).digest() for i in range(n)]


class WindowedLoad(unittest.TestCase):
    def test_windows_read_exactly_the_unwindowed_descriptors(self) -> None:
        counts = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        batch = _batch(counts)
        keys = _keys(len(counts))
        whole, windowed = _client(), _client()
        hits_a, lens_a = worker_module._batch_get_auto_sg_windowed(
            whole, keys, batch.ptrs, batch.caps, 0, 0)
        hits_b, lens_b = worker_module._batch_get_auto_sg_windowed(
            windowed, keys, batch.ptrs, batch.caps, 3, 0)
        self.assertEqual(len(whole._lib.calls), 1)
        self.assertEqual(len(windowed._lib.calls), 4)
        flat = [entry for call in windowed._lib.calls for entry in call]
        self.assertEqual(flat, whole._lib.calls[0])
        self.assertEqual((hits_a, lens_a), (hits_b, lens_b))

    def test_a_window_is_a_prebuilt_batch(self) -> None:
        batch = _batch([2, 2, 2])
        window = batch.window(1, 3)
        self.assertIs(dc._prebuilt_sg_batch(window.ptrs, window.caps), window)
        self.assertEqual(window.num_segments, 4)
        self.assertEqual([list(v) for v in window.ptrs], [list(v) for v in batch.ptrs[1:3]])
        with self.assertRaises(IndexError):
            batch.window(2, 4)

    def test_plain_lists_are_still_sliced(self) -> None:
        keys = _keys(3)
        ptrs = [[0x100], [0x200, 0x240], [0x300]]
        caps = [[64], [64, 64], [64]]
        client = _client()
        worker_module._batch_get_auto_sg_windowed(client, keys, ptrs, caps, 2, 0)
        self.assertEqual(
            [entry for call in client._lib.calls for entry in call],
            [([0x100], [64]), ([0x200, 0x240], [64, 64]), ([0x300], [64])],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
