"""A receive thread must fence the cache owner's GPU, not its default device."""
import threading

import pytest
import torch
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

from dfkv_vllm.coordinator import DfkvStoreCoordinator
from dfkv_vllm.data import ChunkedTokenDatabase, KeyMetadata, LoadSpec, ReqMeta
from dfkv_vllm.worker import KVCacheStoreRecvingThread


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("request_level_loads", [False, True])
def test_receive_completion_waits_for_the_owning_gpu(monkeypatch, request_level_loads):
    monkeypatch.setenv("DFKV_GPU_LOAD_FENCE", "0" if request_level_loads else "1")
    for device in (0, 1):
        with torch.cuda.device(device):
            torch.cuda._sleep(1)
            torch.cuda.synchronize()
    with torch.cuda.device(1):
        pool = torch.zeros((3, 64), dtype=torch.uint8, device="cuda:1")
        stream = torch.cuda.Stream(device=1)
        complete = torch.cuda.Event()
        metadata = KeyMetadata(
            model_name="async-device", dp_size=1, dp_rank=-1,
            tp_size=1, tp_rank=0, pcp_size=1, pcp_rank=0,
            dcp_size=1, dcp_rank=0, pp_size=1, pp_rank=0,
        )
        database = ChunkedTokenDatabase(metadata, block_size=64, hash_block_size=64)
        database.set_seg_layout([(pool.data_ptr(), 64, 64)])
        spec = FullAttentionSpec(
            block_size=64, num_kv_heads=1, head_size=1,
            head_size_v=0, dtype=torch.uint8,
        )
        coordinator = DfkvStoreCoordinator([KVCacheGroupSpec(["kv"], spec)], 64, 64)

        class PendingDeviceWrite:
            def batch_get_auto_sg(self, keys, pointers, capacities):
                # Model a transfer whose completion notification precedes GPU
                # visibility. Real GPU work remains pending on device 1.
                with torch.cuda.device(1), torch.cuda.stream(stream):
                    torch.cuda._sleep(2_000_000_000)
                    pool[1].fill_(17)
                    complete.record(stream)
                return [True], [64]

        receiver = KVCacheStoreRecvingThread(
            PendingDeviceWrite(), coordinator, [database], 64,
            tp_rank=0, ready_event=threading.Event(),
            request_level_loads=request_level_loads,
        )
    request = ReqMeta(
        req_id="async-device", token_len_chunk=64,
        block_ids=([1],), block_hashes=[BlockHash(b"x" * 32)],
        load_spec=LoadSpec(0, 64, True, token_len=64),
    )
    thread = threading.Thread(target=receiver.load_request_sync, args=(request,))
    try:
        thread.start()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert complete.query(), "receive completed before the owning GPU's writes"
        assert receiver.get_and_clear_block_ids_with_load_errors() == set()
        assert torch.equal(pool[1].cpu(), torch.full((64,), 17, dtype=torch.uint8))
    finally:
        torch.cuda.synchronize(1)
        thread.join(timeout=10)
