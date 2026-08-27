from types import SimpleNamespace
from unittest.mock import MagicMock

from dfkv_vllm.scheduler import DfkvStoreScheduler


def test_new_request_metadata_uses_complete_allocated_block_table():
    scheduler = object.__new__(DfkvStoreScheduler)
    scheduler.kv_role = "kv_both"
    scheduler.client = MagicMock()
    scheduler.load_specs = {}
    scheduler._request_trackers = {}
    scheduler._preempted_req_ids = set()
    scheduler._unfinished_request_ids = set()
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
