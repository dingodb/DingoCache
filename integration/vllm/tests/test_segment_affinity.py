"""Opt-in absolute-segment routing across the vLLM native/worker boundary."""
import ctypes
import hashlib
import os
import struct
import sys
import threading
from types import SimpleNamespace

import msgspec
import pytest
import torch
from vllm.distributed.kv_events import BlockStored
from vllm.v1.core.kv_cache_utils import BlockHash

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from _cache_api import cache_group, full_attention_spec  # noqa: E402
from _coordinator_fake import fake_coordinator  # noqa: E402
from dfkv_common import cabi as dfkv_cabi  # noqa: E402
from dfkv_common.replay_trace import block_key_filename  # noqa: E402
from dfkv_vllm import dfkv_client as client_module  # noqa: E402
from dfkv_vllm import worker as worker_module  # noqa: E402
from dfkv_vllm.coordinator import DfkvStoreCoordinator  # noqa: E402
from dfkv_vllm.data import (  # noqa: E402
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    PoolKey,
    ReqMeta,
)
from dfkv_vllm.dfkv_client import DfkvDeviceClient  # noqa: E402
from dfkv_vllm.worker import (  # noqa: E402
    DfkvStoreWorker,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    _batch_get_auto_sg_windowed,
    _parse_segment_affinity_pages,
    _segment_anchor_key,
)

BLOCK = 4
PAGES = 2


def _metadata():
    return KeyMetadata(
        model_name="affinity", dp_size=1, dp_rank=-1,
        tp_size=1, tp_rank=0, pcp_size=1, pcp_rank=0,
        dcp_size=1, dcp_rank=0, pp_size=1, pp_rank=0,
    )


def _hashes(count=6):
    return [
        BlockHash(hashlib.sha256(f"page-{i}".encode()).digest())
        for i in range(count)
    ]


def _db():
    db = ChunkedTokenDatabase(_metadata(), BLOCK)
    db.set_seg_layout([(0x1000, 8, 8)])
    return db


def _key(db, h):
    return PoolKey(db.metadata, h.hex()).to_bytes()


class _CaptureStore:
    """Capture the exact logical object and routing bytes crossing SAVE."""

    def __init__(self):
        self.objects = set()
        self.probes = []
        self.puts = []

    def batch_exist(self, keys, *, route_keys=None):
        keys = list(keys)
        routes = None if route_keys is None else list(route_keys)
        self.probes.append((keys, routes))
        return [int(key in self.objects) for key in keys]

    def batch_put_sg(self, keys, _ptrs, _sizes, *, route_keys=None):
        keys = list(keys)
        routes = None if route_keys is None else list(route_keys)
        self.puts.append((keys, routes))
        self.objects.update(keys)
        return [0] * len(keys)


def _decode_pool_key(wire):
    """Strictly decode all self-delimiting fields in an actual PoolKey."""
    magic = b"DFKVPOOL\x02"
    assert wire.startswith(magic)
    offset = len(magic)

    def field():
        nonlocal offset
        size, = struct.unpack_from("<I", wire, offset)
        offset += 4
        value = wire[offset:offset + size]
        assert len(value) == size
        offset += size
        return value

    pool = field()
    page_hash = field()
    # Exactly five (size, rank) pairs precede group_id/component.
    axes = struct.unpack_from("<IiIiIiIiIi", wire, offset)
    offset += 5 * 8
    group_id, = struct.unpack_from("<I", wire, offset)
    offset += 4
    component = field()
    assert offset == len(wire)
    return {
        "pool": pool,
        "page_hash": page_hash,
        "axes": axes,
        "group_id": group_id,
        "component": component,
    }


def _native_block_key(object_key):
    """Decode the native (tenant, digest_hi, digest_lo) eviction identity."""
    encoded = block_key_filename(b"affinity-test-namespace", object_key)
    assert len(encoded) == 48
    return tuple(int(encoded[i:i + 16], 16) for i in (0, 16, 32))


def _decode_events(events):
    return [
        msgspec.msgpack.decode(msgspec.msgpack.encode(event), type=BlockStored)
        for event in events
    ]


def test_config_validation_defaults_to_zero_and_rejects_non_integers():
    assert _parse_segment_affinity_pages(0) == 0
    assert _parse_segment_affinity_pages(" 3 ") == 3
    assert _parse_segment_affinity_pages("0") == 0
    for value in (True, False, -1, 1.5, "x", "-1", "", None):
        with pytest.raises(ValueError, match="segment_affinity_pages"):
            _parse_segment_affinity_pages(value)


def test_anchor_uses_absolute_page_and_preserves_boundary_partial_state_tail():
    hashes = _hashes()
    db = _db()
    full = [PoolKey(db.metadata, h.hex()) for h in hashes]
    for page, anchor_page in ((0, 0), (1, 0), (2, 2), (3, 2), (4, 4), (5, 4)):
        key = full[page]
        anchor = _segment_anchor_key(
            db, page * BLOCK, key, hashes, PAGES
        )
        assert key.to_bytes() == full[page].to_bytes()
        assert anchor.to_bytes() == full[anchor_page].to_bytes()

    # A boundary/state/tail key at the segment's first page is itself the
    # route anchor: its full-block counterpart is a different object key.
    partial_hash = hashlib.sha256(b"partial-boundary").hexdigest()
    partial = PoolKey(db.metadata, partial_hash)
    assert _segment_anchor_key(
        db, 2 * BLOCK, partial, hashes, PAGES
    ).to_bytes() == partial.to_bytes()
    assert partial.to_bytes() != full[2].to_bytes()

    # Once beyond the first page, all such keys route to that page's real
    # full object, not to a positional/batch-relative neighbor.
    for start_page in (3, 5):
        boundary = PoolKey(db.metadata, partial_hash)
        assert _segment_anchor_key(
            db, start_page * BLOCK, boundary, hashes, PAGES
        ).to_bytes() == full[start_page - 1].to_bytes()


def test_anchor_fails_closed_when_absolute_first_page_hash_is_unavailable():
    hashes = _hashes()
    key = PoolKey(_metadata(), hashes[-1].hex())
    db = SimpleNamespace(
        block_size=BLOCK,
        chunk_hash=lambda *_args: (_ for _ in ()).throw(IndexError("missing")),
    )
    with pytest.raises(ValueError, match="absolute segment-first hash"):
        _segment_anchor_key(db, 5 * BLOCK, key, hashes, PAGES)


def test_shared_prefix_dedup_routes_and_native_identity_have_no_request_lineage():
    hashes = _hashes(2)

    def page_hash(label):
        return BlockHash(hashlib.sha256(label.encode()).digest())

    requests = []
    for req_id, first_block, token_base in (
        ("request-a", 1, 200),
        ("request-b", 11, 300),
    ):
        divergent = [page_hash(f"{req_id}-{i}") for i in range(2)]
        requests.append(ReqMeta(
            req_id=req_id, token_len_chunk=4 * BLOCK,
            block_ids=(list(range(first_block, first_block + 4)),),
            block_hashes=hashes + divergent, can_save=True,
            token_ids=list(range(100, 100 + 2 * BLOCK)) + list(
                range(token_base, token_base + 2 * BLOCK)
            ),
        ))

    store = _CaptureStore()
    db = _db()
    sender = KVCacheStoreSendingThread(
        store,
        fake_coordinator(
            BLOCK, store_mask=lambda n: [[True] * (n // BLOCK)]
        ),
        [db], BLOCK, 0, 0, 1, "kv_producer", threading.Event(),
        enable_kv_event=True, segment_affinity_pages=PAGES,
    )

    def save(request):
        sender.add_stored_request(request)
        sender._handle_request(request)
        return _decode_events(sender.get_kv_events())

    # Reverse construction order so content position, not construction/batch
    # position, determines the common anchors. The second request dedups its
    # two common objects and only PUTs/publishes its divergent suffix.
    second_events = save(requests[1])
    first_events = save(requests[0])
    second_probe, second_routes = store.probes[0]
    first_probe, first_routes = store.probes[1]
    assert store.probes[0][0][:2] == first_probe[:2]
    assert second_routes[:2] == first_routes[:2] == [
        second_probe[0], second_probe[0]
    ]
    assert second_probe[2:] != first_probe[2:]
    assert store.puts[0][0] == second_probe
    assert store.puts[1][0] == first_probe[2:]

    # Every captured key strictly decodes to pool/hash/rank/group/component;
    # metadata stops at the end of those fields. Native storage/eviction then
    # sees only the corresponding tenant+digest triples. Equal common keys are
    # shared content identity, never a unique request chain identifier.
    for key in (*second_probe, *first_probe):
        decoded = _decode_pool_key(key)
        assert decoded["page_hash"]
        assert decoded["axes"] == (1, -1, 1, 0, 1, 0, 1, 0, 1, 0)
        assert decoded["group_id"] == 0
    assert [_native_block_key(key) for key in second_probe[:2]] == [
        _native_block_key(key) for key in first_probe[:2]
    ]
    assert [_native_block_key(key) for key in second_probe[2:]] != [
        _native_block_key(key) for key in first_probe[2:]
    ]

    converted = worker_module.maybe_convert_block_hash
    second_hashes = requests[1].block_hashes
    first_hashes = requests[0].block_hashes
    assert [event.block_hashes for event in second_events] == [
        [converted(h)] for h in second_hashes
    ]
    assert [event.parent_block_hash for event in second_events] == [
        None, converted(second_hashes[0]), converted(second_hashes[1]),
        converted(second_hashes[2]),
    ]
    assert [event.block_hashes for event in first_events] == [
        [converted(h)] for h in first_hashes[2:]
    ]
    assert [event.parent_block_hash for event in first_events] == [
        converted(first_hashes[1]), converted(first_hashes[2])
    ]

    # Route hints are parallel call data only: with affinity disabled the same
    # real SAVE presents byte-identical primary object keys to dfkv.
    plain_store = _CaptureStore()
    plain_sender = KVCacheStoreSendingThread(
        plain_store,
        fake_coordinator(
            BLOCK, store_mask=lambda n: [[True] * (n // BLOCK)]
        ),
        [_db()], BLOCK, 0, 0, 1, "kv_producer", threading.Event(),
        segment_affinity_pages=0,
    )
    plain_sender.add_stored_request(requests[1])
    plain_sender._handle_request(requests[1])
    assert plain_store.probes == [(second_probe, None)]
    assert plain_store.puts == [(second_probe, None)]


def test_full_block_events_skip_prefix_but_exclude_partial_boundary_tail():
    # Half-block hashes let the prompt checkpoint at token 10 address the
    # whole block [8, 12), while full positional objects combine two hashes.
    db = ChunkedTokenDatabase(
        _metadata(), BLOCK, hash_block_size=BLOCK // 2
    )
    db.set_seg_layout([(0x1000, 8, 8)])
    hashes = _hashes(6)
    store = _CaptureStore()
    sender = KVCacheStoreSendingThread(
        store,
        fake_coordinator(
            BLOCK, hash_block_size=BLOCK // 2,
            enable_partial_hash_hits=True,
            store_mask=lambda n: [[True] * (n // BLOCK)],
            load_mask=lambda _h, boundary: [[True] * (
                (boundary + BLOCK - 1) // BLOCK
            )],
            partial_tail_boundary=lambda _n: 10,
        ),
        [db], BLOCK, 0, 0, 1, "kv_producer", threading.Event(),
        enable_kv_event=True, segment_affinity_pages=PAGES,
    )

    def save(token_len, num_prompt_tokens=None):
        request = ReqMeta(
            req_id="partial-tail", token_len_chunk=token_len,
            block_ids=([1, 2, 3],), block_hashes=hashes, can_save=True,
            token_ids=list(range(3 * BLOCK)),
            num_prompt_tokens=num_prompt_tokens,
        )
        sender.add_stored_request(request)
        sender._handle_request(request)
        return _decode_events(sender.get_kv_events())

    events = save(2 * BLOCK, num_prompt_tokens=11)
    assert len(events) == 2
    assert [event.parent_block_hash for event in events] == [
        None, events[0].block_hashes[0]
    ]
    assert [event.token_ids for event in events] == [
        list(range(BLOCK)), list(range(BLOCK, 2 * BLOCK))
    ]

    # The actual third probe/PUT is the partial checkpoint H(tokens[:10]).
    # It has neither a root nor any other BlockStored event: off-grid objects
    # deliberately carry no parent lineage, including None-parent events.
    keys, routes = store.probes[0]
    assert len(keys) == 3
    tail_key = keys[-1]
    assert _decode_pool_key(tail_key)["page_hash"] == hashes[4].hex().encode()
    assert routes[-1] == tail_key
    converted_tail = worker_module.maybe_convert_block_hash(hashes[4])
    assert all(converted_tail not in event.block_hashes for event in events)

    # The next SAVE skips the two confirmed full objects, but its event still
    # parents to the preceding *full* block, not the tail at the same start.
    preceding_full_hash = events[1].block_hashes[0]
    events = save(3 * BLOCK)
    assert len(events) == 1
    full_third_key, full_third_routes = store.probes[1]
    assert len(full_third_key) == 1
    assert full_third_key[0] != tail_key
    assert full_third_routes == full_third_key
    assert events[0].parent_block_hash == preceding_full_hash
    assert events[0].token_ids == list(range(2 * BLOCK, 3 * BLOCK))


def test_offloaded_state_has_an_anchor_but_no_placeholder_parent_event():
    hashes = _hashes(2)
    store = _CaptureStore()
    db = _db()
    sender = KVCacheStoreSendingThread(
        store,
        fake_coordinator(
            BLOCK, enable_partial_hash_hits=True,
            store_mask=lambda n: [[True] * (n // BLOCK)],
        ),
        [db], BLOCK, 0, 0, 1, "kv_producer", threading.Event(),
        enable_kv_event=True, segment_affinity_pages=PAGES,
        tp_sharded_groups=frozenset({0}),
    )
    request = ReqMeta(
        req_id="state-only", token_len_chunk=0,
        block_ids=([1, 2],), block_hashes=hashes, can_save=True,
        token_ids=list(range(2 * BLOCK)),
        state_blocks={0: {2 * BLOCK: 9}},
    )
    sender.add_stored_request(request)
    sender._handle_request(request)

    (keys, routes), = store.probes
    assert keys == [_key(db, hashes[1])]
    assert routes == [_key(db, hashes[0])]
    assert _decode_pool_key(keys[0])["page_hash"] == hashes[1].hex().encode()
    assert sender.get_kv_events() == []


def test_window_get_slices_routes_exactly_with_keys_and_descriptors():
    class Client:
        def __init__(self):
            self.calls = []

        def batch_get_auto_sg(self, keys, ptrs, caps, *, route_keys):
            self.calls.append((list(keys), list(ptrs), list(caps), list(route_keys)))
            return [True] * len(keys), [sum(c) for c in caps]

    client = Client()
    keys = [f"k{i}".encode() for i in range(5)]
    routes = [f"r{i}".encode() for i in range(5)]
    ptrs = [[i] for i in range(5)]
    caps = [[i + 1] for i in range(5)]
    hits, lengths = _batch_get_auto_sg_windowed(
        client, keys, ptrs, caps, 2, 0, routes
    )
    assert [call[0] for call in client.calls] == [keys[:2], keys[2:4], keys[4:]]
    assert [call[3] for call in client.calls] == [
        routes[:2], routes[2:4], routes[4:]
    ]
    assert hits == [True] * 5
    assert lengths == [1, 2, 3, 4, 5]


def test_save_retain_probe_put_and_retain_keep_filtered_anchor_alignment():
    hashes = _hashes(3)
    db = _db()
    keys = [_key(db, h) for h in hashes]
    routes = [keys[0], keys[0], keys[2]]

    class Store:
        def __init__(self):
            self.objects = {keys[1]}
            self.retains = []
            self.puts = []

        def batch_retain(self, batch_keys, ttl_s, mode, *, route_keys):
            self.retains.append((list(batch_keys), list(route_keys)))
            return [
                dfkv_cabi.DFKV_RET_PROTECTED_NEW
                if key in self.objects else dfkv_cabi.DFKV_RET_MISS
                for key in batch_keys
            ]

        def batch_put_sg(self, batch_keys, _ptrs, _sizes, *, route_keys):
            self.puts.append((list(batch_keys), list(route_keys)))
            self.objects.update(batch_keys)
            return [0] * len(batch_keys)

    store = Store()
    sender = KVCacheStoreSendingThread(
        store,
        fake_coordinator(
            BLOCK, store_mask=lambda n: [[True] * (n // BLOCK)]
        ),
        [db], BLOCK, 0, 0, 1, "kv_producer", threading.Event(),
        segment_affinity_pages=PAGES,
    )
    request = ReqMeta(
        req_id="save-affinity", token_len_chunk=3 * BLOCK,
        block_ids=([1, 2, 3],), block_hashes=hashes, can_save=True,
        retain_ttl_s=60,
    )
    sender.add_stored_request(request)
    sender._handle_request(request)

    assert store.retains == [
        (keys, routes),
        ([keys[0], keys[2]], [routes[0], routes[2]]),
    ]
    assert store.puts == [
        ([keys[0], keys[2]], [routes[0], routes[2]])
    ]


def test_load_rotation_windows_and_stale_remove_preserve_anchor_alignment(
    monkeypatch,
):
    hashes = _hashes(4)
    db = _db()
    keys = [_key(db, h) for h in hashes]
    routes = [keys[0], keys[0], keys[2], keys[2]]

    class Store:
        def __init__(self):
            self.gets = []
            self.removes = []

        def batch_get_auto_sg(self, batch_keys, _ptrs, caps, *, route_keys):
            self.gets.append((list(batch_keys), list(route_keys)))
            return [True] * len(batch_keys), [sum(c) for c in caps]

        def supports_remove(self):
            return True

        def batch_remove(self, batch_keys, *, route_keys):
            self.removes.append((list(batch_keys), list(route_keys)))
            return [1] * len(batch_keys)

    store = Store()
    receiver = KVCacheStoreRecvingThread(
        store,
        fake_coordinator(
            BLOCK, load_mask=lambda _h, n: [[True] * (n // BLOCK)]
        ),
        [db], BLOCK, 0, threading.Event(),
        load_window_keys=2, segment_affinity_pages=PAGES,
    )
    monkeypatch.setattr(worker_module, "_batch_rotation_offset", lambda *a: 1)
    request = ReqMeta(
        req_id="load-affinity", token_len_chunk=0,
        block_ids=([1, 2, 3, 4],), block_hashes=hashes,
        load_spec=LoadSpec(0, 4 * BLOCK, True, 4 * BLOCK),
    )
    receiver._handle_request(request)
    assert store.gets == [
        ([keys[1], keys[2]], [keys[0], keys[2]]),
        ([keys[3], keys[0]], [keys[2], keys[0]]),
    ]

    receiver._remove_stale(store, [keys[3]], [keys[2]])
    assert store.removes == [([keys[3]], [keys[2]])]


def test_lookup_refresh_and_retention_precheck_use_absolute_anchors():
    hashes = _hashes(4)
    db = _db()
    keys = [_key(db, h) for h in hashes]
    routes = [keys[0], keys[0], keys[2], keys[2]]

    class Store:
        def __init__(self):
            self.probes = []
            self.prechecks = []

        def batch_retain(self, batch_keys, _ttl, _mode, *, route_keys):
            self.probes.append((list(batch_keys), list(route_keys)))
            return [dfkv_cabi.DFKV_RET_PROTECTED_NEW] * len(batch_keys)

        def retain_limits(self):
            return 1, 120

        def retain_precheck(self, batch_keys, *, route_keys):
            self.prechecks.append((list(batch_keys), list(route_keys)))
            return True

    spec = full_attention_spec(
        block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.uint8
    )
    groups = [cache_group(["layer"], spec)]
    store = Store()
    worker = SimpleNamespace(
        coord=DfkvStoreCoordinator(groups, BLOCK, BLOCK),
        token_dbs=[db], client=store, tp_size=1, num_kv_head=8, pp_size=1,
        segment_affinity_pages=PAGES, _kv_cache_groups=groups,
        _record_kv_connector_operation=lambda *a, **k: None,
    )
    assert DfkvStoreWorker.lookup(worker, 4 * BLOCK, hashes, 30) == (
        4 * BLOCK, 30
    )
    assert store.probes == [(keys, routes)]
    assert store.prechecks == [(keys, routes)]


def test_each_anchor_consumes_binary_routes_and_empty_native_fallback():
    object_keys = [b"object\x00a", b"object\xffb"]
    route_keys = [b"", b"route\x00second"]

    class Lib:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            if not name.endswith("_anchored"):
                raise AttributeError(name)

            def call(*args):
                _handle, keys, lens, routes, route_lens = args[:5]
                n = len(keys)
                objects = [ctypes.string_at(keys[i], lens[i]) for i in range(n)]
                anchors = []
                for i in range(n):
                    if route_lens[i] == 0:
                        assert not routes[i]
                        anchors.append(b"")
                    else:
                        anchors.append(
                            ctypes.string_at(routes[i], route_lens[i])
                        )
                assert objects == object_keys
                assert anchors == route_keys
                self.calls.append(name)
                if "precheck" in name:
                    ok = ctypes.cast(args[-1], ctypes.POINTER(ctypes.c_int))
                    ok.contents.value = 1
                elif "retain" in name:
                    for i in range(n):
                        args[-1][i] = dfkv_cabi.DFKV_RET_PROTECTED_NEW
                elif "get_auto" in name:
                    for i in range(n):
                        args[-2][i] = 1
                        args[-1][i] = 8
                else:
                    for i in range(n):
                        args[-1][i] = 1
                return 0

            return call

    client = object.__new__(DfkvDeviceClient)
    client._lib = Lib()
    client._h = 0xBEEF
    client._has_retain = True
    client._segment_affinity = True
    ptrs = [0x1000, 0x2000]
    caps = [8, 8]

    assert client.batch_put(
        object_keys, ptrs, caps, route_keys=route_keys
    ) == [0, 0]
    assert client.batch_get(
        object_keys, ptrs, caps, route_keys=route_keys
    ) == ([1, 1], [8, 8])
    assert client.batch_put_sg(
        object_keys, [[p] for p in ptrs], [[c] for c in caps],
        route_keys=route_keys,
    ) == [0, 0]
    assert client.batch_get_auto_sg(
        object_keys, [[p] for p in ptrs], [[c] for c in caps],
        route_keys=route_keys,
    ) == ([1, 1], [8, 8])
    assert client.batch_exist(object_keys, route_keys=route_keys) == [1, 1]
    assert client.batch_remove(object_keys, route_keys=route_keys) == [1, 1]
    assert client.batch_retain(
        object_keys, 60, dfkv_cabi.DFKV_RETAIN_SET, route_keys=route_keys
    ) == [dfkv_cabi.DFKV_RET_PROTECTED_NEW] * 2
    assert client.retain_precheck(object_keys, route_keys=route_keys)

    assert client._lib.calls == [
        "dfkv_batch_put_anchored",
        "dfkv_batch_get_auto_anchored",
        "dfkv_batch_put_sg_anchored",
        "dfkv_batch_get_auto_sg_anchored",
        "dfkv_batch_exist_anchored",
        "dfkv_batch_remove_anchored",
        "dfkv_batch_retain_anchored",
        "dfkv_retain_precheck_anchored",
    ]

    with pytest.raises(ValueError, match="object key must not be empty"):
        client.batch_exist([b""], route_keys=[b""])


def test_enabled_old_library_fails_before_open_and_off_needs_no_anchor(
    monkeypatch,
):
    opened = []

    class OldLib:
        def dfkv_open_v2(self, _options):
            opened.append(True)
            return 0xBEEF

        def dfkv_batch_exist(self, _h, keys, lens, n, out):
            for i in range(n):
                out[i] = 1
            return 0

    monkeypatch.setattr(client_module, "load_lib", lambda _path: OldLib())
    with pytest.raises(RuntimeError, match="complete anchored dfkv C ABI"):
        DfkvDeviceClient(
            members="n1=127.0.0.1:28001", key_namespace=b"affinity",
            segment_affinity=True,
        )
    assert opened == []

    client = object.__new__(DfkvDeviceClient)
    client._lib = OldLib()
    client._h = 0xBEEF
    client._segment_affinity = False
    assert client.batch_exist([b"legacy"]) == [1]
