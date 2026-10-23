"""KV event aggregation must account for striped (replicated) groups."""
from vllm.distributed.kv_events import BlockStored

from dfkv_vllm.connector import DfkvStoreKVEvents


def _event(block, group):
    return BlockStored(
        block_hashes=[block], parent_block_hash=None, token_ids=[block],
        block_size=4, lora_id=None, medium="cpu", lora_name=None,
        group_idx=group,
    )


def _poll(per_worker_events, factors):
    """Combine worker containers the way vLLM's output aggregator does."""
    combined = None
    for events in per_worker_events:
        container = DfkvStoreKVEvents(1, group_tp_replication_factors=factors)
        container.add_events(events)
        if combined is None:
            combined = container
        else:
            combined.add_events(container.get_all_events())
            combined.increment_workers(1)
    return combined.aggregate().get_all_events()


def test_striped_block_needs_one_report_sharded_block_needs_all():
    factors = (4, 1)  # group 0: MLA striped over 4 ranks; group 1: TP-sharded
    striped = _event(10, 0)
    sharded = _event(20, 1)
    partial = _event(30, 1)
    published = _poll(
        [[striped, sharded, partial], [sharded, partial], [sharded, partial], [sharded]],
        factors,
    )
    assert striped in published
    assert sharded in published
    assert partial not in published


def test_empty_containers_count_toward_the_quorum():
    sharded = _event(20, 1)
    assert _poll([[sharded], [], [], []], (4, 1)) == []
    assert _poll([[sharded]] * 4, (4, 1)) == [sharded]
