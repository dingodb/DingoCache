"""Hybrid resume admission must validate the checkpoint at the returned depth."""
import ctypes
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

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



def test_noallocation_prefill_saves_state_before_page_recycling(hybrid_workers, monkeypatch):
    """Real allocator/serializer steps must SAVE the live state at each boundary."""
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager
    from dfkv_vllm.data import RequestTracker
    from dfkv_vllm.scheduler import DfkvStoreScheduler

    # Use 640-token pages and a deliberately small 256-token test budget
    # (not the deployment budget) to force steps without new allocations.
    monkeypatch.setitem(globals(), "BLOCK", 640)
    monkeypatch.setattr(worker_module, "get_tensor_model_parallel_world_size", lambda: 1)
    create, _ = hybrid_workers
    producer, buffers = create(0, num_blocks=40)
    hashes = _hashes(20)
    pools = [BlockPool(40, True, 64) for _ in range(3)]
    attention = MLAAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.uint8,
    )
    state = MambaSpec(
        block_size=BLOCK, shapes=((16,),), dtypes=(torch.uint8,),
        mamba_cache_mode="align", num_speculative_blocks=5,
    )
    managers = [FullAttentionManager(attention, pools[0], True, 0, BLOCK)] + [
        MambaManager(state, pools[group], enable_caching=True,
                     kv_cache_group_id=group, scheduler_block_size=BLOCK)
        for group in (1, 2)
    ]
    request = SimpleNamespace(
        request_id="chunked", num_computed_tokens=0, num_prompt_tokens=4000,
        num_tokens=4000, shared_prefix_boundary=0, all_token_ids=list(range(4000)),
        block_hashes=hashes, num_output_tokens=0, num_output_placeholders=0,
    )
    upstream = SimpleNamespace(
        use_pp=False, use_v2_model_runner=False,
        prev_step_scheduled_req_ids={request.request_id},
        cache_config=SimpleNamespace(block_size=BLOCK), max_num_scheduled_tokens=256,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        hash_block_size=64, use_eagle_block_drop=False,
        mamba_has_prefill_checkpoint_blocks=False, mamba_partial_cache_hit=True,
        mamba_shared_prefix_checkpoint=False,
    )
    scheduler = object.__new__(DfkvStoreScheduler)
    for name, value in dict(
        kv_role="kv_producer", _block_size=BLOCK,
        _unfinished_requests={request.request_id: (request, ([], [], []))},
        _unfinished_request_ids={request.request_id}, _preempted_req_ids=set(),
        _failed_load_req_ids=set(), _allocated_req_ids=set(), load_specs={},
        client=SimpleNamespace(discard=lambda _: None),
        _request_trackers={request.request_id: RequestTracker(
            request.request_id, 0, ([], [], []), prefill_end_tokens=4000, token_ids=[],
        )},
    ).items():
        setattr(scheduler, name, value)
    saves, empty_steps, recycled_pages, state_pages = [], 0, set(), set()
    for _ in range(13):
        start = request.num_computed_tokens
        count = Scheduler._mamba_block_aligned_split(upstream, request, 256)
        end = start + count
        for manager in managers:
            manager.remove_skipped_blocks(request.request_id, start, num_prompt_tokens=4000)
        groups = tuple(manager.allocate_new_blocks(request.request_id, end, end)
                       for manager in managers)
        cached = Scheduler._make_cached_request_data(
            upstream, [request], [], {request.request_id: count}, {},
            {request.request_id: KVCacheBlocks(groups)},
        )
        empty_steps += cached.new_block_ids[0] is None
        metadata = scheduler.build_connector_meta(SimpleNamespace(
            finished_req_ids=set(), preempted_req_ids=set(), scheduled_new_reqs=[],
            scheduled_cached_reqs=cached, num_scheduled_tokens={request.request_id: count},
        ))
        # Substitute only numerical kernel output; real managers own page life.
        for name, manager in zip(("state", "wrapped_state"), managers[1:], strict=True):
            active = manager.req_to_blocks[request.request_id][(end - 1) // BLOCK].block_id
            state_pages.add(active)
            buffers[name][active].fill_(end // 64)
        for block in managers[0].req_to_blocks[request.request_id]:
            buffers["mla"][block.block_id].fill_(17)
        if start >= 2 * BLOCK:
            for name, pool in zip(("state", "wrapped_state"), pools[1:], strict=True):
                held = pool.get_new_blocks(pool.get_num_free_blocks())
                recycled_pages.update(block.block_id for block in held)
                for block in held:
                    buffers[name][block.block_id].fill_(231)
                pool.free_blocks(held)
        for job in metadata.requests:
            producer.kv_send_thread.add_stored_request(job)
            producer.kv_send_thread._handle_request(job)
            saves.append(job.token_len_chunk)
        request.num_computed_tokens = end
    assert empty_steps > 1
    assert state_pages & recycled_pages  # real state pages were reclaimed
    assert saves

    # Verify every saved boundary, not merely the final hit or attention stripe.
    for depth in saves:
        consumer, target = create(0, num_blocks=40)
        assert consumer.lookup(depth, hashes) == depth
        get = consumer.client.batch_get_auto_sg
        transfers = []

        def checked_get(keys, pointers, capacities):
            hits, lengths = get(keys, pointers, capacities)
            assert all(hit and length == sum(caps) for hit, length, caps
                       in zip(hits, lengths, capacities, strict=True))
            transfers.extend(keys)
            return hits, lengths

        monkeypatch.setattr(consumer.client, "batch_get_auto_sg", checked_get)
        load = ReqMeta(
            req_id=f"restore-{depth}", token_len_chunk=depth,
            block_ids=(list(range(20, 20 + depth // BLOCK)), [25], [26]),
            block_hashes=hashes, load_spec=LoadSpec(0, depth, True, token_len=depth),
        )
        consumer.kv_recv_thread.load_request_sync(load)
        assert transfers
        assert consumer.get_block_ids_with_load_errors() == set()
        finished, failed = consumer.kv_recv_thread.get_and_clear_receive_results()
        assert finished == {load.req_id} and failed == set()
        assert torch.all(target["mla"][20:20 + depth // BLOCK] == 17)
        assert torch.all(target["state"][25] == depth // 64)
        assert torch.all(target["wrapped_state"][26] == depth // 64)
    assert saves == [640, 1280, 1920, 2560]


def test_noallocation_prefill_kv_events_carry_block_token_ids(hybrid_workers):
    """KV events from SAVE must name the tokens of each stored block."""
    from dfkv_vllm.data import RequestTracker
    from dfkv_vllm.scheduler import DfkvStoreScheduler

    create, _ = hybrid_workers
    producer, _ = create(0, states=False, kv_events=True)
    prompt = list(range(1000, 1000 + 3 * BLOCK))
    request = SimpleNamespace(
        request_id="events", all_token_ids=prompt + [7, 8],
        block_hashes=_hashes(3),
    )
    scheduler = object.__new__(DfkvStoreScheduler)
    for name, value in dict(
        kv_role="kv_producer", _block_size=BLOCK, client=MagicMock(),
        load_specs={}, _failed_load_req_ids=set(), _preempted_req_ids=set(),
        _unfinished_request_ids={"events"}, _allocated_req_ids=set(),
        _unfinished_requests={"events": (request, ([1, 2, 3],))},
        _request_trackers={"events": RequestTracker(
            "events", 0, ([1, 2, 3],), token_ids=[],
            prefill_end_tokens=len(prompt),
        )},
    ).items():
        setattr(scheduler, name, value)

    # All blocks are allocated up front, so every chunk has new_block_ids=None.
    saved = []
    for computed, count in ((0, BLOCK // 2), (BLOCK // 2, BLOCK // 2),
                            (BLOCK, 2 * BLOCK)):
        metadata = scheduler.build_connector_meta(SimpleNamespace(
            finished_req_ids=set(), preempted_req_ids=set(), scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=["events"], new_block_ids=[None],
                num_computed_tokens=[computed],
            ),
            num_scheduled_tokens={"events": count},
        ))
        for job in metadata.requests:
            producer.kv_send_thread.add_stored_request(job)
            producer.kv_send_thread._handle_request(job)
            saved.append(job.token_len_chunk)

    assert saved == [BLOCK, 3 * BLOCK]
    events = producer.get_kv_events()
    converted = [worker_module.maybe_convert_block_hash(h) for h in _hashes(3)]
    assert [event.block_hashes for event in events] == [[h] for h in converted]
    # Block 1 is the first block of the second SAVE; it still chains to block 0.
    assert [event.parent_block_hash for event in events] == [None, *converted[:2]]
    assert [event.token_ids for event in events] == [
        prompt[i * BLOCK:(i + 1) * BLOCK] for i in range(3)
    ]


def test_striped_kv_events_survive_aggregation_with_request_parents(hybrid_workers):
    """Every TP rank stores only its stripe of a replicated (MLA) payload; the
    published events must still cover every block, chained by request order."""
    from dfkv_vllm.connector import DfkvStoreKVEvents

    create, _ = hybrid_workers
    hashes = _hashes(3)
    tokens = list(range(500, 500 + 3 * BLOCK))
    combined = None
    for rank in range(4):
        producer, _ = create(rank, states=False, kv_events=True)
        job = ReqMeta(
            req_id=f"striped-{rank}", token_len_chunk=3 * BLOCK,
            block_ids=([1, 2, 3],), block_hashes=hashes, token_ids=tokens,
        )
        producer.kv_send_thread.add_stored_request(job)
        producer.kv_send_thread._handle_request(job)
        container = DfkvStoreKVEvents(
            1, group_tp_replication_factors=producer.group_tp_replication_factors,
        )
        container.add_events(producer.get_kv_events())
        if combined is None:
            combined = container
        else:
            combined.add_events(container.get_all_events())
            combined.increment_workers(1)

    events = combined.aggregate().get_all_events()
    converted = [worker_module.maybe_convert_block_hash(h) for h in hashes]
    assert [event.block_hashes for event in events] == [[h] for h in converted]
    assert [event.parent_block_hash for event in events] == [None, *converted[:2]]
    assert [event.token_ids for event in events] == [
        tokens[i * BLOCK:(i + 1) * BLOCK] for i in range(3)
    ]


@pytest.mark.parametrize("step_end,save_state", [(2 * BLOCK, True), (2 * BLOCK + 7, False)])
def test_state_saves_only_at_a_boundary_this_step_computed(step_end, save_state):
    from dfkv_vllm.data import RequestTracker

    tracker = RequestTracker("r", step_end, ([1, 2, 3],))
    job = ReqMeta.from_request_tracker(tracker, BLOCK)
    assert job.token_len_chunk == 2 * BLOCK
    assert job.save_state is save_state


def test_unverified_state_boundary_fails_closed(hybrid_workers):
    """Attention may still be saved, but without the boundary's recurrent
    state the prefix must not be admitted."""
    create, objects = hybrid_workers
    hashes = _hashes()
    producer, _ = create(0)
    job = ReqMeta(
        req_id="unverified", token_len_chunk=2 * BLOCK,
        block_ids=([1, 3], [3], [3]), block_hashes=hashes, save_state=False,
    )
    producer.kv_send_thread.add_stored_request(job)
    producer.kv_send_thread._handle_request(job)

    stored = [key for _, key in objects]
    assert stored  # attention chunks are still saved
    assert not any(b"mamba" in key for key in stored)
    consumer, _ = create(0)
    assert consumer.lookup(2 * BLOCK, hashes) == 0


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

    def validate_block_sizes(self, sizes, *, context):
        # CPU objects have no transport size limit, but invalid geometry must
        # still fail registration rather than being hidden by this substitute.
        assert context
        assert all(size > 0 for size in sizes)

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

    def create(tp_rank, *, dcp=1, states=True, attention_block_size=None, scratch=False, kernel_tiles=1, wrapped_attention=False, num_blocks=4, kv_events=False):
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
            kv_events_config=(
                SimpleNamespace(enable_kv_cache_events=True) if kv_events else None
            ),
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
