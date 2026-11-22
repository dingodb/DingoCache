# SPDX-License-Identifier: Apache-2.0
"""A chunk of a model with several KV layer groups is read back whole.

LMCache describes such a chunk with one shape per group
(``metadata.get_shapes()``), and PUTs the whole chunk: every group's bytes.
A GET has to give the transport a buffer of all of them; one of group 0 only
is too small for the stored object, so every GET missed while ``contains``
kept reporting the chunk present.
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import _lmcache_stub  # noqa: E402

_lmcache_stub.install()

from dfkv_connector.exists_cache import ExistsLRU  # noqa: E402
from dfkv_connector.remote_connector import DfkvConnector  # noqa: E402

TOKENS = 4
# Two layer groups, as LMCache's kv_layer_groups give them: a main KV group
# and a narrower one (e.g. an indexer). Element size 2 bytes for both.
SHAPES = [(1, 3, TOKENS, 8), (1, 2, TOKENS, 4)]
DTYPES = [2, 2]


def _nbytes(shapes, dtypes) -> int:
    if not isinstance(shapes, list):
        shapes, dtypes = [shapes], [dtypes]
    total = 0
    for shape, size in zip(shapes, dtypes):
        n = 1
        for d in shape:
            n *= d
        total += n * size
    return total


FULL = _nbytes(SHAPES, DTYPES)
TOKEN = FULL // TOKENS


class _Obj:
    def __init__(self, nbytes: int) -> None:
        self.buf = bytearray(nbytes)
        self.released = False

    @property
    def byte_array(self) -> memoryview:
        return memoryview(self.buf)

    def ref_count_down(self) -> None:
        self.released = True


class _CpuBackend:
    def __init__(self) -> None:
        self.allocations = []

    def allocate(self, shapes, dtypes, fmt=None):
        self.allocations.append((shapes, dtypes, fmt))
        return _Obj(_nbytes(shapes, dtypes))


class _Ring:
    """Holds one stored object per key."""

    def __init__(self) -> None:
        self.objects: dict[bytes, bytes] = {}

    async def batch_get(self, keys, bufs, *, release_buffers=None):
        per_key, lengths = [], []
        for key, buf in zip(keys, bufs):
            value = self.objects.get(key)
            if value is None or len(value) > len(buf):
                # dfkv: a stored value larger than the buffer is a miss.
                per_key.append(False)
                lengths.append(0)
                continue
            buf[: len(value)] = value
            per_key.append(True)
            lengths.append(len(value))
        return all(per_key), per_key, lengths


def _connector(shapes=SHAPES, dtypes=DTYPES):
    connector = object.__new__(DfkvConnector)
    connector._client = _Ring()
    connector._local_cpu_backend = _CpuBackend()
    connector._exists_lru = ExistsLRU(64)
    connector._batch_max_keys = 512
    connector._get_parallelism = 1
    connector.meta_shapes = list(shapes)
    connector.meta_dtypes = list(dtypes)
    connector.meta_fmt = "fmt"
    connector.full_chunk_size_bytes = _nbytes(list(shapes), list(dtypes))
    connector.single_token_size = connector.full_chunk_size_bytes // TOKENS
    connector._key = lambda key: key
    return connector


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class MultiGroupChunks(unittest.TestCase):
    def test_a_get_buffer_holds_every_group(self) -> None:
        connector = _connector()
        connector._client.objects[b"k"] = bytes(range(256))[:FULL] * 1
        obj = _run(connector.get(b"k"))
        self.assertIsNotNone(obj, "the stored chunk did not fit the GET buffer")
        self.assertEqual(bytes(obj.byte_array), connector._client.objects[b"k"])
        self.assertEqual(
            connector._local_cpu_backend.allocations,
            [(SHAPES, DTYPES, "fmt")],
        )

    def test_batched_gets_hold_every_group(self) -> None:
        connector = _connector()
        for key in (b"a", b"b"):
            connector._client.objects[key] = key * FULL
        out = _run(connector.batched_get([b"a", b"b"]))
        self.assertEqual([bytes(o.byte_array) for o in out], [b"a" * FULL, b"b" * FULL])
        out = _run(connector.batched_get_non_blocking("lookup", [b"a", b"b"]))
        self.assertEqual([bytes(o.byte_array) for o in out], [b"a" * FULL, b"b" * FULL])

    def test_a_partial_chunk_of_several_groups_is_a_miss(self) -> None:
        # LMCache's reshape of an unfull chunk sets one shape: it cannot
        # describe several groups, so the bytes would be read as group 0.
        connector = _connector()
        connector._client.objects[b"k"] = b"p" * (2 * TOKEN)
        connector._exists_lru.add(b"k")
        self.assertIsNone(_run(connector.get(b"k")))
        self.assertFalse(connector._exists_lru.has(b"k"))
        self.assertEqual(_run(connector.batched_get([b"k"])), [None])

    def test_one_group_is_unchanged(self) -> None:
        connector = _connector(shapes=SHAPES[:1], dtypes=DTYPES[:1])
        full = connector.full_chunk_size_bytes
        connector._client.objects[b"k"] = b"x" * full
        self.assertEqual(bytes(_run(connector.get(b"k")).byte_array), b"x" * full)
        # An unfull chunk of one group is still reshaped by LMCache.
        connector._client.objects[b"p"] = b"y" * (full // TOKENS)
        obj = _run(connector.get(b"p"))
        self.assertIsNotNone(obj)
        self.assertEqual(obj.valid_bytes, full // TOKENS)


class UnsupportedConfigurations(unittest.TestCase):
    def _config(self, **values):
        defaults = {"use_layerwise": False, "remote_serde": "naive"}
        defaults.update(values)
        return type("Config", (), defaults)()

    def test_layerwise_and_non_raw_serde_are_refused(self) -> None:
        for config in (
            self._config(use_layerwise=True),
            self._config(remote_serde="cachegen"),
        ):
            with self.subTest(config=vars(type(config))):
                with self.assertRaises(ValueError):
                    DfkvConnector._check_supported_config(config)

    def test_the_default_configuration_is_accepted(self) -> None:
        DfkvConnector._check_supported_config(self._config())
        DfkvConnector._check_supported_config(self._config(remote_serde=None))
        DfkvConnector._check_supported_config(type("Old", (), {})())


if __name__ == "__main__":
    unittest.main(verbosity=2)
