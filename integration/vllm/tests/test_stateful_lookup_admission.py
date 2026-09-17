"""Hybrid resume admission must validate the checkpoint at the returned depth."""
import ctypes
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from dfkv_vllm import worker as worker_module
from dfkv_vllm.data import KeyMetadata, LoadSpec, PoolKey, ReqMeta
from dfkv_vllm.worker import DfkvStoreWorker

BLOCK = 64
_MLA_METADATA = {
    "model_name": "stateful-admission", "dp_size": 1, "dp_rank": -1,
    "tp_size": 1, "tp_rank": -1, "pcp_size": 1, "pcp_rank": 0,
    "dcp_size": 1, "pp_size": 1, "pp_rank": 0,
}


def _hashes(count=2):
    return [BlockHash(hashlib.sha256(str(i).encode()).digest()) for i in range(count)]



def test_shorter_prefix_requires_its_own_state_checkpoints():
    """A longer prefix's tail mask cannot authorize a shorter resume."""
    hashes = _hashes()[:2]
    metadata = [
        KeyMetadata(**{**_MLA_METADATA, "dcp_rank": 0, "group_id": group})
        for group in range(5)
    ]
    store = {
        PoolKey(metadata[0], h.hex()).to_bytes() for h in hashes
    }
    # The original query finds four of six objects: two dense chunks and
    # two of the four recurrent checkpoints at the final boundary.
    store.update(
        PoolKey(metadata[group], hashes[1].hex()).to_bytes()
        for group in (1, 2)
    )
    coord = SimpleNamespace(
        lcm_block_size=BLOCK,
        load_mask=lambda hashes, length: [
            [True] * (length // BLOCK),
            *[
                [False] * (length // BLOCK - 1) + [True]
                for _ in range(4)
            ],
        ],
        block_hashes_for_spec=lambda all_hashes, spec: all_hashes,
    )
    worker = SimpleNamespace(
        coord=coord,
        token_dbs=[
            SimpleNamespace(metadata=md, block_size=BLOCK) for md in metadata
        ],
        client=SimpleNamespace(
            batch_exist=lambda keys: [int(key in store) for key in keys]
        ),
        tp_size=1,
        num_kv_head=1,
        _kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=None) for _ in metadata
        ],
        _record_kv_connector_operation=lambda *args, **kwargs: None,
    )
    assert DfkvStoreWorker.lookup(worker, 2 * BLOCK, hashes) == 0
    # An actual checkpoint at the earlier boundary makes that prefix safe.
    store.update(
        PoolKey(metadata[group], hashes[0].hex()).to_bytes()
        for group in range(1, 5)
    )
    assert DfkvStoreWorker.lookup(worker, 2 * BLOCK, hashes) == BLOCK


class _MemoryClient:
    """Native-boundary substitute that gathers/scatters actual CPU buffers."""

    def __init__(self, objects, namespace):
        self.objects = objects
        self.namespace = namespace

    def register_memory(self, base, size):
        pass

    def batch_exist(self, keys):
        return [int((self.namespace, key) in self.objects) for key in keys]

    def batch_put_sg(self, keys, pointers, sizes):
        for key, ptrs, lens in zip(keys, pointers, sizes, strict=True):
            self.objects[self.namespace, key] = b"".join(
                ctypes.string_at(ptr, length)
                for ptr, length in zip(ptrs, lens, strict=True)
            )
        return [0] * len(keys)

    def batch_get_auto_sg(self, keys, pointers, capacities):
        hits, lengths = [], []
        for key, ptrs, caps in zip(keys, pointers, capacities, strict=True):
            value = self.objects.get((self.namespace, key))
            hits.append(value is not None)
            lengths.append(len(value) if value is not None else 0)
            if value is None:
                continue
            assert len(value) == sum(caps)
            offset = 0
            for ptr, cap in zip(ptrs, caps, strict=True):
                ctypes.memmove(ptr, value[offset:offset + cap], cap)
                offset += cap
        return hits, lengths

    def close(self):
        pass

class _ScratchSpec(MLAAttentionSpec):
    @property
    def prefix_cacheable(self):
        return False

    @property
    def participates_in_prefix_caching(self):
        return False



@pytest.fixture
def hybrid_workers(monkeypatch):
    """Use real startup, group registration, send/load and lookup on CPU."""
    objects = {}
    workers = []
    rank = [0]
    dcp_size = [1]
    monkeypatch.setattr(
        worker_module, "DfkvDeviceClient",
        lambda **kwargs: _MemoryClient(objects, kwargs["key_namespace"]),
    )
    monkeypatch.setattr(
        worker_module, "LookupKeyServer",
        lambda *args: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        worker_module, "apply_rank_local_rail_affinity",
        lambda *args: SimpleNamespace(enabled=False),
    )
    monkeypatch.setattr(worker_module, "get_dp_engine_index", lambda config: 0)
    monkeypatch.setattr(
        worker_module, "get_tensor_model_parallel_rank", lambda: rank[0],
    )
    monkeypatch.setattr(
        worker_module, "get_tensor_model_parallel_world_size", lambda: 4,
    )
    monkeypatch.setattr(
        worker_module, "get_world_group",
        lambda: SimpleNamespace(local_rank=rank[0]),
    )
    monkeypatch.setattr(
        worker_module, "get_pcp_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0),
    )
    monkeypatch.setattr(
        worker_module, "get_dcp_group",
        lambda: SimpleNamespace(
            world_size=dcp_size[0], rank_in_group=rank[0] % dcp_size[0],
        ),
    )
    monkeypatch.setattr(
        worker_module, "ensure_deterministic_block_hashing", lambda config: None,
    )
    # Exercise synchronous handlers without starting background threads.
    for cls in (
        worker_module.KVCacheStoreSendingThread,
        worker_module.KVCacheStoreRecvingThread,
    ):
        monkeypatch.setattr(cls, "start", lambda self: self.ready_event.set())
    monkeypatch.setenv("DFKV_CONNECTOR_CLIENT_RANKS", "1")
    monkeypatch.setenv("DFKV_CONNECTOR_CLIENT_ELIDE", "1")
    monkeypatch.setenv("DFKV_CLIENT_NODE_DEDUP", "0")
    monkeypatch.setenv("DFKV_CLIENT_NODE_DEDUP_GPU", "0")
    monkeypatch.setenv("DFKV_CLIENT_LOG_SUFFIX", "state-test")

    def create(tp_rank, *, dcp=1, states=True, attention_block_size=None, scratch=False, kernel_tiles=1, wrapped_attention=False, num_blocks=4):
        rank[0], dcp_size[0] = tp_rank, dcp
        physical_block_size = attention_block_size or BLOCK // dcp
        mla = MLAAttentionSpec(
            block_size=physical_block_size, num_kv_heads=1, head_size=1, dtype=torch.uint8,
        )
        if wrapped_attention:
            mla = UniformTypeKVCacheSpecs(
                block_size=physical_block_size, kv_cache_specs={"mla": mla}
            )
        state = MambaSpec(
            block_size=BLOCK, shapes=((16,),), dtypes=(torch.uint8,),
            mamba_cache_mode="align",
        )
        groups = [
            KVCacheGroupSpec(["mla"], mla),
            KVCacheGroupSpec(["state"], state),
            KVCacheGroupSpec(
                ["wrapped_state"],
                UniformTypeKVCacheSpecs(
                    block_size=BLOCK, kv_cache_specs={"wrapped_state": state},
                ),
            ),
        ]
        if not states:
            groups = groups[:1]
        if scratch:
            groups.append(KVCacheGroupSpec(
                ["scratch"],
                _ScratchSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.uint8),
            ))
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                model="test/hybrid-state", use_mla=True,
                get_num_layers=lambda parallel: len(groups),
            ),
            parallel_config=SimpleNamespace(
                data_parallel_size=1, pipeline_parallel_size=1, rank=tp_rank,
                decode_context_parallel_size=dcp,
            ),
            kv_transfer_config=SimpleNamespace(
                kv_role="kv_producer",
                kv_connector_extra_config={"members": "127.0.0.1:1"},
            ),
            cache_config=SimpleNamespace(
                num_gpu_blocks=num_blocks, block_size=physical_block_size,
                enable_prefix_caching=True, prefix_match_unit=None,
            ),
            kv_events_config=None,
        )
        worker = DfkvStoreWorker(
            config, SimpleNamespace(kv_cache_groups=groups),
        )
        workers.append(worker)
        buffers = {
            "mla": torch.zeros((num_blocks * kernel_tiles, physical_block_size // kernel_tiles, 1), dtype=torch.uint8),
            "state": torch.zeros((num_blocks, 16), dtype=torch.uint8),
            "wrapped_state": torch.zeros((num_blocks, 16), dtype=torch.uint8),
        }
        if not states:
            buffers = {"mla": buffers["mla"]}
        if scratch:
            buffers["scratch"] = torch.zeros((1, 4, 1), dtype=torch.uint8)
        worker.register_kv_caches(buffers)
        return worker, buffers

    yield create, objects
    for worker in workers:
        worker.close()


@pytest.mark.parametrize("dcp", [1, 2])
def test_hybrid_state_round_trip_preserves_every_tp_payload(hybrid_workers, dcp):
    create, objects = hybrid_workers
    hashes = _hashes()
    expected = []
    for rank in range(4):
        producer, buffers = create(rank, dcp=dcp)
        for group, tensor in enumerate(buffers.values()):
            for block in range(4):
                tensor[block].fill_(10 + block + group * 20 + (rank * 4 if group else 0))
        expected.append({name: tensor.clone() for name, tensor in buffers.items()})
        request = ReqMeta(
            req_id=f"save-{rank}", token_len_chunk=2 * BLOCK,
            block_ids=([1, 3], [3], [3]), block_hashes=hashes,
        )
        producer.kv_send_thread.add_stored_request(request)
        producer.kv_send_thread._handle_request(request)

    for rank in range(4):
        consumer, buffers = create(rank, dcp=dcp)
        assert consumer.lookup(2 * BLOCK, hashes) == 2 * BLOCK
        request = ReqMeta(
            req_id=f"load-{rank}", token_len_chunk=2 * BLOCK,
            block_ids=([1, 2], [2], [2]), block_hashes=hashes,
            load_spec=LoadSpec(0, 2 * BLOCK, True, token_len=2 * BLOCK),
        )
        consumer.kv_recv_thread.load_request_sync(request)
        assert consumer.get_block_ids_with_load_errors() == set()
        for name, tensor in buffers.items():
            result = torch.zeros_like(tensor)
            result[2].copy_(expected[rank][name][3])
            if name == "mla":
                result[1].copy_(expected[rank][name][1])
            assert torch.equal(tensor, result), (rank, name)

    # An old state object under attention's pool, even with the same numeric
    # rank coordinate, cannot substitute for a missing corrected shard.
    victim, _ = create(3, dcp=dcp)
    metadata = victim.token_dbs[2].metadata
    key = PoolKey(metadata, hashes[1].hex()).to_bytes()
    legacy_key = PoolKey(replace(metadata, pool_name="kv"), hashes[1].hex()).to_bytes()
    objects[victim.client.namespace, legacy_key] = objects.pop((victim.client.namespace, key))
    for rank in range(4):
        consumer, _ = create(rank, dcp=dcp)
        assert consumer.lookup(2 * BLOCK, hashes) == 0


@pytest.mark.parametrize("client_ranks", ["1", "4"])
def test_replicated_mla_round_trip_with_converged_or_striped_writers(
    hybrid_workers, monkeypatch, client_ranks,
):
    create, _ = hybrid_workers
    monkeypatch.setenv("DFKV_CONNECTOR_CLIENT_RANKS", client_ranks)
    hashes = _hashes()
    for rank in range(4):
        producer, buffers = create(rank, states=False)
        buffers["mla"][1].fill_(11)
        buffers["mla"][3].fill_(33)
        request = ReqMeta(
            req_id=f"mla-save-{rank}", token_len_chunk=2 * BLOCK,
            block_ids=([1, 3],), block_hashes=hashes,
        )
        producer.kv_send_thread.add_stored_request(request)
        producer.kv_send_thread._handle_request(request)

    for rank in range(4):
        consumer, buffers = create(rank, states=False)
        # Producer-only elided clients are lazily created when loading.
        consumer._ensure_client_for_load()
        assert consumer.lookup(2 * BLOCK, hashes) == 2 * BLOCK
        consumer.kv_recv_thread.load_request_sync(ReqMeta(
            req_id=f"mla-load-{rank}", token_len_chunk=2 * BLOCK,
            block_ids=([1, 2],), block_hashes=hashes,
            load_spec=LoadSpec(0, 2 * BLOCK, True, token_len=2 * BLOCK),
        ))
        assert consumer.get_block_ids_with_load_errors() == set()
        expected = torch.zeros_like(buffers["mla"])
        expected[1].fill_(11)
        expected[2].fill_(33)
        assert torch.equal(buffers["mla"], expected)


@pytest.mark.parametrize("dcp", [1, 2])
def test_hybrid_lookup_covers_every_smaller_attention_block(hybrid_workers, dcp):
    create, _ = hybrid_workers
    hashes = _hashes(4)
    expected = []
    for rank in range(4):
        producer, buffers = create(rank, dcp=dcp, attention_block_size=16 // dcp, num_blocks=5)
        for group, tensor in enumerate(buffers.values()):
            for block in range(5):
                tensor[block].fill_(10 + block + group * 20 + (rank * 4 if group else 0))
        expected.append({name: tensor.clone() for name, tensor in buffers.items()})
        request = ReqMeta(
            req_id=f"mixed-save-{rank}", token_len_chunk=BLOCK,
            block_ids=([3, 4, 2, 1], [3], [3]), block_hashes=hashes,
        )
        producer.kv_send_thread.add_stored_request(request)
        producer.kv_send_thread._handle_request(request)

    for rank in range(4):
        consumer, buffers = create(rank, dcp=dcp, attention_block_size=16 // dcp, num_blocks=5)
        assert consumer.lookup(BLOCK, hashes) == BLOCK
        request = ReqMeta(
            req_id=f"mixed-load-{rank}", token_len_chunk=BLOCK,
            block_ids=([1, 2, 3, 4], [1], [1]), block_hashes=hashes,
            load_spec=LoadSpec(0, BLOCK, True, token_len=BLOCK),
        )
        consumer.kv_recv_thread.load_request_sync(request)
        assert consumer.get_block_ids_with_load_errors() == set()
        expected_mla = torch.zeros_like(buffers["mla"])
        expected_mla[1:].copy_(expected[rank]["mla"][[3, 4, 2, 1]])
        assert torch.equal(buffers["mla"], expected_mla)
        for name in ("state", "wrapped_state"):
            result = torch.zeros_like(buffers[name])
            result[1].copy_(expected[rank][name][3])
            assert torch.equal(buffers[name], result)


def test_nonpersistent_scratch_does_not_constrain_or_overwrite_prefix_cache(hybrid_workers):
    create, _ = hybrid_workers
    hashes = _hashes(1)
    for rank in range(4):
        producer, buffers = create(rank, scratch=True)
        buffers["mla"][1].fill_(17)
        buffers["state"][3].fill_(30 + rank)
        buffers["wrapped_state"][3].fill_(50 + rank)
        buffers["scratch"].fill_(42)
        request = ReqMeta(
            req_id=f"scratch-save-{rank}", token_len_chunk=BLOCK,
            block_ids=([1], [3], [3], [0]), block_hashes=hashes,
        )
        producer.kv_send_thread.add_stored_request(request)
        producer.kv_send_thread._handle_request(request)

    for rank in range(4):
        consumer, buffers = create(rank, scratch=True)
        buffers["scratch"].fill_(99)
        assert consumer.lookup(BLOCK, hashes) == BLOCK
        request = ReqMeta(
            req_id=f"scratch-load-{rank}", token_len_chunk=BLOCK,
            block_ids=([1], [2], [2], [0]), block_hashes=hashes,
            load_spec=LoadSpec(0, BLOCK, True, token_len=BLOCK),
        )
        consumer.kv_recv_thread.load_request_sync(request)
        assert consumer.get_block_ids_with_load_errors() == set()
        assert torch.all(buffers["mla"][1] == 17)
        assert torch.all(buffers["state"][2] == 30 + rank)
        assert torch.all(buffers["wrapped_state"][2] == 50 + rank)
        assert torch.all(buffers["scratch"] == 99)


def test_logical_block_ids_restore_every_kernel_tile(hybrid_workers):
    create, _ = hybrid_workers
    hashes = _hashes()
    expected_source = torch.arange(4 * BLOCK, dtype=torch.uint8).reshape(4, BLOCK)
    for rank in range(4):
        producer, buffers = create(rank, states=False, kernel_tiles=4)
        buffers["mla"].copy_(expected_source.reshape(buffers["mla"].shape))
        request = ReqMeta(
            req_id=f"tiles-save-{rank}", token_len_chunk=2 * BLOCK,
            block_ids=([1, 3],), block_hashes=hashes,
        )
        producer.kv_send_thread.add_stored_request(request)
        producer.kv_send_thread._handle_request(request)

    for rank in range(4):
        consumer, buffers = create(rank, states=False, kernel_tiles=4)
        request = ReqMeta(
            req_id=f"tiles-load-{rank}", token_len_chunk=2 * BLOCK,
            block_ids=([1, 2],), block_hashes=hashes,
            load_spec=LoadSpec(0, 2 * BLOCK, True, token_len=2 * BLOCK),
        )
        consumer.kv_recv_thread.load_request_sync(request)
        assert consumer.get_block_ids_with_load_errors() == set()
        expected = torch.zeros_like(expected_source)
        expected[1].copy_(expected_source[1])
        expected[2].copy_(expected_source[3])
        assert torch.equal(buffers["mla"].reshape(4, BLOCK), expected)


def test_full_block_tables_do_not_shift_to_uncomputed_tail(hybrid_workers):
    create, _ = hybrid_workers
    hashes = _hashes()
    expected_source = []
    for rank in range(4):
        producer, buffers = create(rank)
        for group, tensor in enumerate(buffers.values()):
            for block in range(4):
                tensor[block].fill_(10 + block + group * 20 + (rank * 4 if group else 0))
        expected_source.append({name: tensor.clone() for name, tensor in buffers.items()})
        request = ReqMeta(
            req_id=f"tail-save-{rank}", token_len_chunk=2 * BLOCK,
            block_ids=([1, 3, 2], [1, 3, 2], [1, 3, 2]), block_hashes=hashes,
        )
        producer.kv_send_thread.add_stored_request(request)
        producer.kv_send_thread._handle_request(request)

    for rank in range(4):
        consumer, buffers = create(rank)
        for tensor in buffers.values():
            tensor[3].fill_(99)
        request = ReqMeta(
            req_id=f"tail-load-{rank}", token_len_chunk=2 * BLOCK,
            block_ids=([1, 2, 3], [1, 2, 3], [1, 2, 3]), block_hashes=hashes,
            load_spec=LoadSpec(0, 2 * BLOCK, True, token_len=2 * BLOCK),
        )
        consumer.kv_recv_thread.load_request_sync(request)
        assert consumer.get_block_ids_with_load_errors() == set()
        for name, tensor in buffers.items():
            expected = torch.zeros_like(tensor)
            expected[2].copy_(expected_source[rank][name][3])
            expected[3].fill_(99)
            if name == "mla":
                expected[1].copy_(expected_source[rank][name][1])
            assert torch.equal(tensor, expected), (rank, name)


def test_wrapped_attention_uses_engine_dcp_hash_geometry(hybrid_workers):
    # Two physical 32-token pages span 128 global tokens under DCP=2.
    create, _ = hybrid_workers
    hashes = _hashes()
    for rank in range(4):
        producer, buffers = create(rank, dcp=2, wrapped_attention=True)
        buffers["mla"][1].fill_(17)
        buffers["mla"][3].fill_(19)
        buffers["state"][3].fill_(30 + rank)
        buffers["wrapped_state"][3].fill_(50 + rank)
        request = ReqMeta(
            req_id=f"wrapped-save-{rank}", token_len_chunk=2 * BLOCK,
            block_ids=([1, 3], [3], [3]), block_hashes=hashes,
        )
        producer.kv_send_thread.add_stored_request(request)
        producer.kv_send_thread._handle_request(request)

    for rank in range(4):
        consumer, buffers = create(rank, dcp=2, wrapped_attention=True)
        assert consumer.lookup(2 * BLOCK, hashes) == 2 * BLOCK
        request = ReqMeta(
            req_id=f"wrapped-load-{rank}", token_len_chunk=2 * BLOCK,
            block_ids=([1, 2], [2], [2]), block_hashes=hashes,
            load_spec=LoadSpec(0, 2 * BLOCK, True, token_len=2 * BLOCK),
        )
        consumer.kv_recv_thread.load_request_sync(request)
        assert consumer.get_block_ids_with_load_errors() == set()
        assert torch.all(buffers["mla"][1] == 17)
        assert torch.all(buffers["mla"][2] == 19)
        assert torch.all(buffers["state"][2] == 30 + rank)
        assert torch.all(buffers["wrapped_state"][2] == 50 + rank)


def test_unallocated_source_is_not_published_as_an_immutable_object(hybrid_workers):
    create, objects = hybrid_workers
    producer, buffers = create(0, states=False)
    hashes = _hashes()
    buffers["mla"][0].fill_(99)
    buffers["mla"][1].fill_(17)
    request = ReqMeta(
        req_id="null-source", token_len_chunk=2 * BLOCK,
        block_ids=([0, 1],), block_hashes=hashes,
    )
    producer.kv_send_thread.add_stored_request(request)
    producer.kv_send_thread._handle_request(request)
    metadata = producer.token_dbs[0].metadata
    missing = PoolKey(metadata, hashes[0].hex()).to_bytes()
    present = PoolKey(metadata, hashes[1].hex()).to_bytes()
    assert (producer.client.namespace, missing) not in objects
    assert objects[producer.client.namespace, present] == bytes([17]) * BLOCK
    assert producer.lookup(2 * BLOCK, hashes) == 0


def test_unallocated_destination_fails_without_overwriting_null_memory(hybrid_workers):
    create, _ = hybrid_workers
    producer, source = create(0, states=False)
    hashes = _hashes(1)
    source["mla"][1].fill_(17)
    request = ReqMeta(
        req_id="valid-source", token_len_chunk=BLOCK,
        block_ids=([1],), block_hashes=hashes,
    )
    producer.kv_send_thread.add_stored_request(request)
    producer.kv_send_thread._handle_request(request)
    consumer, target = create(0, states=False)
    target["mla"][0].fill_(99)
    before = target["mla"].clone()
    consumer.kv_recv_thread.load_request_sync(ReqMeta(
        req_id="null-destination", token_len_chunk=BLOCK,
        block_ids=([0],), block_hashes=hashes,
        load_spec=LoadSpec(0, BLOCK, True, token_len=BLOCK),
    ))
    assert consumer.get_block_ids_with_load_errors() == {0}
    assert torch.equal(target["mla"], before)


def test_mutable_state_cannot_advance_before_its_put_finishes(hybrid_workers, monkeypatch):
    create, objects = hybrid_workers
    producer, buffers = create(0)
    hashes = _hashes(1)
    buffers["state"][3].fill_(30)
    entered, release, advanced = threading.Event(), threading.Event(), threading.Event()
    put = producer.client.batch_put_sg

    def delayed_put(*args):
        entered.set()
        assert release.wait(5)
        return put(*args)

    monkeypatch.setattr(producer.client, "batch_put_sg", delayed_put)
    monkeypatch.setattr(
        torch.cuda, "Event",
        lambda: SimpleNamespace(record=lambda: None, synchronize=lambda: None),
    )
    sender = producer.kv_send_thread
    sender.ready_event.clear()
    threading.Thread.start(sender)
    assert sender.ready_event.wait(5)
    request = ReqMeta(
        req_id="mutable-state", token_len_chunk=BLOCK, can_save=True,
        block_ids=([1], [3], [3]), block_hashes=hashes,
    )
    metadata = SimpleNamespace(requests=[request], preempted_req_ids=set())

    def next_model_step():
        producer.get_transfer_results(set(), metadata)
        buffers["state"][3].fill_(99)
        advanced.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(next_model_step)
        try:
            assert entered.wait(5)
            advanced_before_put = advanced.wait(1)
        finally:
            release.set()
        future.result(timeout=5)
    key = PoolKey(producer.token_dbs[1].metadata, hashes[0].hex()).to_bytes()
    assert objects[producer.client.namespace, key] == bytes([30]) * 16
    assert not advanced_before_put


def test_same_step_resume_save_releases_finished_blocks(hybrid_workers, monkeypatch):
    create, objects = hybrid_workers
    producer, buffers = create(0)
    hashes = _hashes(1)
    buffers["state"][3].fill_(30)
    monkeypatch.setattr(
        torch.cuda, "Event",
        lambda: SimpleNamespace(record=lambda: None, synchronize=lambda: None),
    )
    sender = producer.kv_send_thread
    sender.ready_event.clear()
    threading.Thread.start(sender)
    assert sender.ready_event.wait(5)
    request = ReqMeta(
        req_id="same-step-save", token_len_chunk=BLOCK, can_save=True,
        block_ids=([1], [3], [3]), block_hashes=hashes,
    )
    metadata = SimpleNamespace(requests=[request], preempted_req_ids={request.req_id})
    producer.handle_preemptions(metadata)
    producer.get_transfer_results(set(), metadata)
    key = PoolKey(producer.token_dbs[1].metadata, hashes[0].hex()).to_bytes()
    assert objects[producer.client.namespace, key] == bytes([30]) * 16
    result = producer.get_transfer_results(
        {request.req_id}, SimpleNamespace(requests=[], preempted_req_ids=set()),
    )
    assert result.finished_sending == {request.req_id}


def test_cancelled_save_generation_cannot_revive(hybrid_workers, monkeypatch):
    create, objects = hybrid_workers
    producer, buffers = create(0, states=False)
    hashes = _hashes(3)
    buffers["mla"][1].fill_(11)
    buffers["mla"][2].fill_(22)
    buffers["mla"][3].fill_(33)
    entered, release = threading.Event(), threading.Event()
    exists = producer.client.batch_exist

    def delay_first_exists(keys):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return exists(keys)

    monkeypatch.setattr(producer.client, "batch_exist", delay_first_exists)
    sender = producer.kv_send_thread
    sender.ready_event.clear()
    threading.Thread.start(sender)
    assert sender.ready_event.wait(5)
    blocker = ReqMeta(
        req_id="other-request", token_len_chunk=BLOCK,
        block_ids=([3],), block_hashes=[hashes[2]],
    )
    stale = ReqMeta(
        req_id="resumed", token_len_chunk=BLOCK,
        block_ids=([1],), block_hashes=[hashes[0]],
    )
    fresh = ReqMeta(
        req_id="resumed", token_len_chunk=BLOCK,
        block_ids=([2],), block_hashes=[hashes[1]],
    )
    sender.add_stored_request(blocker)
    sender.add_request(blocker)
    try:
        assert entered.wait(5)
        sender.add_stored_request(stale)
        sender.add_request(stale)
        producer.handle_preemptions(
            SimpleNamespace(requests=[], preempted_req_ids={stale.req_id}),
        )
        buffers["mla"][1].fill_(99)
        sender.add_stored_request(fresh)
        sender.add_request(fresh)
    finally:
        release.set()
    sender.request_queue.join()
    old_key = PoolKey(producer.token_dbs[0].metadata, hashes[0].hex()).to_bytes()
    new_key = PoolKey(producer.token_dbs[0].metadata, hashes[1].hex()).to_bytes()
    assert (producer.client.namespace, old_key) not in objects
    assert objects[producer.client.namespace, new_key] == bytes([22]) * BLOCK
    result = producer.get_transfer_results(
        {fresh.req_id}, SimpleNamespace(requests=[], preempted_req_ids=set()),
    )
    assert result.finished_sending == {fresh.req_id}
