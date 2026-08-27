from types import SimpleNamespace

from dfkv_vllm.coordinator import DfkvStoreCoordinator


def test_equal_block_size_stateful_group_uses_manager_mask():
    class StatefulManager:
        @staticmethod
        def find_longest_cache_hit(**kwargs):
            pool = kwargs["block_pool"]
            assert pool.hash_block_size == 16
            return ([pool.null_block, pool._present_block],)

    coordinator = object.__new__(DfkvStoreCoordinator)
    coordinator.lcm_block_size = 16
    coordinator.hash_block_size = 16
    coordinator.eagle_attn_group_indices = set()
    stateful_spec = SimpleNamespace(block_size=16)
    coordinator.kv_cache_groups = [
        SimpleNamespace(kv_cache_spec=stateful_spec),
    ]
    coordinator.attention_groups = [(stateful_spec, [0], StatefulManager)]

    masks = coordinator.store_mask(32)

    assert masks == ([False, True],)
