from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dfkv_vllm.data import LoadSpec
from dfkv_vllm.scheduler import DfkvStoreScheduler


def test_new_request_metadata_uses_complete_allocated_block_table():
    scheduler = object.__new__(DfkvStoreScheduler)
    scheduler._failed_load_req_ids = set()
    scheduler.kv_role = "kv_both"
    scheduler.client = MagicMock()
    scheduler.load_specs = {}
    scheduler._request_trackers = {}
    scheduler._preempted_req_ids = set()
    scheduler._unfinished_request_ids = set()
    scheduler._allocated_req_ids = set()
    scheduler._block_size = 4

    request_real = SimpleNamespace(request_id="req-1", block_hashes=[])
    complete_block_ids = ([10, 11], [20, 21])
    scheduler._unfinished_requests = {}
    blocks = SimpleNamespace(get_block_ids=lambda: complete_block_ids)
    scheduler.update_state_after_alloc(request_real, blocks, num_external_tokens=0)

    scheduled_new_request = SimpleNamespace(
        req_id="req-1",
        num_computed_tokens=0,
        # vLLM's NewRequestData exposes only blocks allocated this step.
        block_ids=([11], [21]),
        prefill_token_ids=None,
        prompt_token_ids=list(range(8)),
    )
    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[scheduled_new_request],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={"req-1": 8},
    )

    metadata = scheduler.build_connector_meta(scheduler_output)

    assert len(metadata.requests) == 1
    assert metadata.requests[0].block_ids == complete_block_ids
    assert metadata.requests[0].can_save is True


def test_sampling_tail_does_not_authorize_an_absent_checkpoint():
    scheduler = object.__new__(DfkvStoreScheduler)
    scheduler._failed_load_req_ids = set()
    scheduler.request_level_loads = False
    scheduler._block_size = 64
    scheduler.lookup_async = False
    scheduler.load_async = False
    scheduler.load_specs = {}
    checkpoints = {128}
    scheduler.client = SimpleNamespace(
        lookup=lambda request_id, length, hashes, non_block: max(
            (depth for depth in checkpoints if depth <= length), default=0
        )
    )
    request = SimpleNamespace(
        request_id="stateful-full-hit", num_tokens=128, block_hashes=[]
    )
    # The final token must be recomputed for sampling. A checkpoint at 128
    # cannot be used to resume at 64 when the latter checkpoint is absent.
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    assert request.request_id not in scheduler.load_specs
    checkpoints.add(64)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (64, False)


def test_cache_bypass_discards_stale_external_admission():
    scheduler = object.__new__(DfkvStoreScheduler)
    scheduler._failed_load_req_ids = set()
    scheduler.request_level_loads = False
    scheduler._block_size = 64
    scheduler.lookup_async = False
    scheduler.load_async = False
    cached = {"replay": 128}
    other_spec = object()
    scheduler.load_specs = {"replay": object(), "unrelated": other_spec}
    scheduler.client = SimpleNamespace(
        lookup=lambda request_id, *args, **kwargs: cached[request_id],
        discard=lambda request_id: cached.pop(request_id, None),
    )
    request = SimpleNamespace(
        request_id="replay", num_tokens=256, block_hashes=[],
        skip_reading_prefix_cache=True,
    )
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    assert cached == {}
    assert scheduler.load_specs == {"unrelated": other_spec}


@pytest.mark.parametrize("same_step_preemption", [False, True])
@pytest.mark.parametrize("async_pending", [False, True])
def test_consumer_cached_resume_emits_load_without_save(
    async_pending, same_step_preemption,
):
    scheduler = object.__new__(DfkvStoreScheduler)
    scheduler._failed_load_req_ids = set()
    scheduler.request_level_loads = False
    scheduler.kv_role = "kv_consumer"
    scheduler.client = MagicMock()
    scheduler.client.lookup.return_value = 4
    scheduler.lookup_async = False
    scheduler.load_async = async_pending
    scheduler.load_specs = {}
    scheduler._request_trackers = {}
    scheduler._preempted_req_ids = set()
    scheduler._unfinished_request_ids = set()
    scheduler._unfinished_requests = {}
    scheduler._allocated_req_ids = set()
    scheduler._block_size = 4
    request = SimpleNamespace(
        request_id="resumed", block_hashes=[], all_token_ids=list(range(16)),
        num_tokens=16, num_computed_tokens=0,
    )
    old_blocks = ([1, 2, 3, 4], [5, 6, 7, 8])
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda: old_blocks), 0,
    )
    step = SimpleNamespace(
        finished_req_ids=set(), preempted_req_ids=set(),
        scheduled_new_reqs=[SimpleNamespace(
            req_id="resumed", num_computed_tokens=0, block_ids=old_blocks,
            prefill_token_ids=None, prompt_token_ids=list(range(8)),
        )],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={"resumed": 8},
    )
    assert scheduler.build_connector_meta(step).requests == []

    step.preempted_req_ids = {"resumed"}
    step.scheduled_new_reqs = []
    step.num_scheduled_tokens = {}
    if not same_step_preemption:
        assert scheduler.build_connector_meta(step).requests == []
        step.preempted_req_ids = set()

    # Lookup and allocation admit a real LOAD; the resumed MRV1 request only
    # carries this step's delta, not the complete destination block table.
    assert scheduler.get_num_new_matched_tokens(request, 0) == (4, async_pending)
    blocks = ([10, 11], [20, 21])
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda: blocks), 4,
    )
    request.num_computed_tokens = 4
    resumed_cached = SimpleNamespace(
        req_ids=["resumed"], new_block_ids=[([11], [21])],
        num_computed_tokens=[4],
    )
    if not async_pending:
        step.scheduled_cached_reqs = resumed_cached
        step.num_scheduled_tokens = {"resumed": 4}

    metadata = scheduler.build_connector_meta(step)
    assert len(metadata.requests) == 1
    restored = metadata.requests[0]
    assert restored.req_id == "resumed"
    assert restored.load_spec is not None and restored.load_spec.can_load
    assert restored.load_spec.vllm_cached_tokens == 0
    assert restored.load_spec.kvpool_cached_tokens == 4
    assert restored.can_save is False
    assert restored.block_ids == blocks

    if async_pending:
        # Waiting another tick must not submit the already issued LOAD again.
        step.preempted_req_ids = set()
        assert scheduler.build_connector_meta(step).requests == []
        # After completion MRV1 can resume without allocating another block.
        resumed_cached.new_block_ids = [None]
        step.scheduled_cached_reqs = resumed_cached
        step.num_scheduled_tokens = {"resumed": 4}
        assert scheduler.build_connector_meta(step).requests == []

    # Finish re-prefilling prompt + generated history, then decode. Consumers
    # must neither reload the prefix nor SAVE newly computed chunk/decode KV.
    for computed, count, delta in (
        (8, 4, ([12], [22])),
        (12, 4, ([13], [23])),
        (16, 1, ([14], [24])),
        (17, 1, None),
    ):
        if computed >= request.num_tokens:
            request.all_token_ids.append(computed)
            request.num_tokens += 1
        request.num_computed_tokens = computed
        step.preempted_req_ids = set()
        step.scheduled_cached_reqs = SimpleNamespace(
            req_ids=["resumed"], new_block_ids=[delta],
            num_computed_tokens=[computed],
        )
        step.num_scheduled_tokens = {"resumed": count}
        assert scheduler.build_connector_meta(step).requests == []

    step.finished_req_ids = {"resumed"}
    step.scheduled_cached_reqs = SimpleNamespace(req_ids=[])
    step.num_scheduled_tokens = {}
    assert scheduler.build_connector_meta(step).requests == []


@pytest.mark.parametrize("cached_resume", [False, True])
@pytest.mark.parametrize("async_pending", [False, True])
def test_same_step_preemption_keeps_new_allocation_and_load(async_pending, cached_resume):
    scheduler = object.__new__(DfkvStoreScheduler)
    scheduler._failed_load_req_ids = set()
    scheduler.kv_role = "kv_both"
    scheduler.client = MagicMock()
    scheduler.load_specs = {}
    scheduler._request_trackers = {}
    scheduler._preempted_req_ids = set()
    scheduler._unfinished_request_ids = set()
    scheduler._unfinished_requests = {}
    scheduler._allocated_req_ids = set()
    scheduler._block_size = 4
    request = SimpleNamespace(
        request_id="resumed", block_hashes=[], all_token_ids=list(range(16)),
        num_computed_tokens=4,
    )
    new_blocks = ([10, 11], [20, 21])
    scheduler.load_specs["resumed"] = LoadSpec(0, 4, False)
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda: new_blocks), 4,
    )
    new_request = SimpleNamespace(
        req_id="resumed", num_computed_tokens=4, block_ids=([11], [21]),
        prefill_token_ids=list(range(16)), prompt_token_ids=list(range(8)),
    )
    resumed_cached = SimpleNamespace(
        req_ids=["resumed"], new_block_ids=[([11], [21])],
        num_computed_tokens=[4],
    )
    step = SimpleNamespace(
        finished_req_ids=set(), preempted_req_ids={"resumed"},
        scheduled_new_reqs=[] if async_pending or cached_resume else [new_request],
        scheduled_cached_reqs=(
            resumed_cached if cached_resume and not async_pending
            else SimpleNamespace(req_ids=[])
        ),
        num_scheduled_tokens={} if async_pending else {"resumed": 4},
    )
    metadata = scheduler.build_connector_meta(step)
    assert len(metadata.requests) == 1
    restored = metadata.requests[0]
    assert restored.load_spec is not None and restored.load_spec.can_load
    assert restored.block_ids == new_blocks

    if async_pending:
        # The load finishes later; either runner can now resume computation.
        if cached_resume:
            # The asynchronously restored table already has room for this step.
            resumed_cached.new_block_ids = [None]
        step.preempted_req_ids = set()
        step.scheduled_new_reqs = [] if cached_resume else [new_request]
        step.scheduled_cached_reqs = resumed_cached if cached_resume else SimpleNamespace(req_ids=[])
        step.num_scheduled_tokens = {"resumed": 4}
        scheduler.build_connector_meta(step)

    # The next chunk extends the full table; it is not another resume whose
    # delta-only allocation can replace the previously restored blocks.
    request.num_computed_tokens = 8
    step.preempted_req_ids = set()
    step.scheduled_new_reqs = []
    step.scheduled_cached_reqs = SimpleNamespace(
        req_ids=["resumed"], new_block_ids=[([12], [22])],
        num_computed_tokens=[8],
    )
    step.num_scheduled_tokens = {"resumed": 4}
    metadata = scheduler.build_connector_meta(step)
    assert metadata.requests[0].block_ids == ([10, 11, 12], [20, 21, 22])

    # A later preemption without reallocation must not reuse the earlier
    # step's allocation marker or emit a load into recycled blocks.
    scheduler.load_specs["resumed"] = LoadSpec(0, 4, True)
    step.preempted_req_ids = {"resumed"}
    step.scheduled_cached_reqs = SimpleNamespace(req_ids=[])
    step.num_scheduled_tokens = {}
    metadata = scheduler.build_connector_meta(step)
    assert metadata.requests == []


@pytest.mark.parametrize(
    "group_kinds, load_async, parked",
    [
        (["full"], False, False),
        (["full"], True, True),
        (["full", "full"], False, True),
        (["state"], False, True),
        (["full", "state"], True, True),
    ],
)
def test_load_admission_parks_request_level_layouts(
    monkeypatch, group_kinds, load_async, parked,
):
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from dfkv_vllm import scheduler as scheduler_module

    client = MagicMock()
    monkeypatch.setattr(scheduler_module, "LookupKeyClient", lambda config: client)
    monkeypatch.setattr(
        scheduler_module, "ensure_deterministic_block_hashing", lambda config: None,
    )
    monkeypatch.setattr(
        scheduler_module, "resolve_kv_cache_block_sizes", lambda *args: (4, 4),
    )
    groups = [
        SimpleNamespace(kv_cache_spec=(
            object.__new__(FullAttentionSpec) if kind == "full" else object()
        ))
        for kind in group_kinds
    ]
    scheduler = DfkvStoreScheduler(
        SimpleNamespace(
            cache_config=SimpleNamespace(),
            kv_transfer_config=SimpleNamespace(
                kv_role="kv_both", kv_connector_extra_config={"load_async": load_async},
            ),
        ),
        SimpleNamespace(kv_cache_groups=groups, transfer_groups=groups),
    )
    request = SimpleNamespace(request_id="load", num_tokens=16, block_hashes=[])
    client.lookup.return_value = 8
    assert scheduler.get_num_new_matched_tokens(request, 0) == (8, parked)
    client.lookup.return_value = 0
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    client.lookup.return_value = None
    assert scheduler.get_num_new_matched_tokens(request, 0) == (None, False)


@pytest.mark.parametrize("cached_resume", [False, True])
@pytest.mark.parametrize("load_async", [False, True])
def test_failed_load_bypasses_persistent_hit_and_rebuilds_computed_save(
    cached_resume, load_async,
):
    scheduler = object.__new__(DfkvStoreScheduler)
    scheduler.kv_role = "kv_both"
    scheduler.request_level_loads = True
    scheduler.load_async = load_async
    scheduler.lookup_async = False
    scheduler._block_size = 4
    scheduler.load_specs = {}
    scheduler._failed_load_req_ids = set()
    scheduler._request_trackers = {}
    scheduler._preempted_req_ids = set()
    scheduler._unfinished_request_ids = set()
    scheduler._unfinished_requests = {}
    scheduler._allocated_req_ids = set()
    advertised = {"retry": 12}
    cached = {}

    def lookup(req_id, *args, **kwargs):
        # Discarding the lookup result alone does not repair a remote server
        # that still advertises keys whose GET failed.
        return cached.setdefault(req_id, advertised[req_id])

    scheduler.client = SimpleNamespace(
        lookup=lookup, discard=lambda req_id: cached.pop(req_id, None),
    )
    request = SimpleNamespace(
        request_id="retry", num_tokens=20, num_computed_tokens=0,
        block_hashes=[], all_token_ids=list(range(20)),
    )
    assert scheduler.get_num_new_matched_tokens(request, 0) == (12, True)
    old_blocks = ([1, 2, 3], [4, 5, 6])
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda: old_blocks), 12,
    )
    step = SimpleNamespace(
        finished_req_ids=set(), preempted_req_ids=set(),
        scheduled_new_reqs=[], scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={},
    )
    pending = scheduler.build_connector_meta(step)
    assert pending.requests[0].load_spec.can_load
    assert pending.requests[0].can_save is False

    scheduler.update_connector_output(SimpleNamespace(failed_recving={"retry"}))
    assert cached == {}
    assert scheduler.build_connector_meta(step).requests == []
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    assert advertised == {"retry": 12}
    assert "retry" not in scheduler.load_specs

    # Upstream freed the failed allocation and retries from locally computed
    # tokens. MRV1 may resume via cached data without a new preemption event.
    new_blocks = ([10], [20])
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda: new_blocks), 0,
    )
    step.num_scheduled_tokens = {"retry": 4}
    if cached_resume:
        step.scheduled_cached_reqs = SimpleNamespace(
            req_ids=["retry"], new_block_ids=[new_blocks], num_computed_tokens=[0],
        )
    else:
        step.scheduled_new_reqs = [SimpleNamespace(
            req_id="retry", num_computed_tokens=0, prefill_token_ids=None,
            prompt_token_ids=request.all_token_ids,
        )]
    recomputed = scheduler.build_connector_meta(step).requests[0]
    assert recomputed.load_spec is None
    assert recomputed.can_save is True
    assert recomputed.token_len_chunk == 4
    assert recomputed.block_ids == new_blocks
    assert recomputed.token_ids == list(range(4))
    # A later preemption must not re-enable the same failed remote source.
    step.preempted_req_ids = {"retry"}
    step.scheduled_new_reqs = []
    step.scheduled_cached_reqs = SimpleNamespace(req_ids=[])
    step.num_scheduled_tokens = {}
    assert scheduler.build_connector_meta(step).requests == []
    step.preempted_req_ids = set()
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)


    # Failure quarantine ends only at terminal completion.
    scheduler.request_finished(request, new_blocks)
    step.finished_req_ids = {"retry"}
    step.scheduled_new_reqs = []
    step.scheduled_cached_reqs = SimpleNamespace(req_ids=[])
    step.num_scheduled_tokens = {}
    assert scheduler.build_connector_meta(step).requests == []
    # Late receive completion of an aborted/finalized request cannot resurrect
    # quarantine or stale allocation metadata.
    scheduler.update_connector_output(SimpleNamespace(failed_recving={"retry"}))
    assert scheduler.get_num_new_matched_tokens(request, 0) == (12, True)
