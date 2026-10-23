"""The lookup IPC path must be unique per DP engine, not per host-local rank."""
from types import SimpleNamespace

from dfkv_vllm.worker import get_zmq_rpc_path_lookup


def _config(dp_index, *, local_rank=0, local_engines_only=True):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={}),
        parallel_config=SimpleNamespace(
            data_parallel_index=dp_index,
            data_parallel_rank_local=local_rank,
            local_engines_only=local_engines_only,
        ),
    )


def test_colocated_external_lb_engines_get_distinct_paths():
    # External LB: every engine on the host has local DP rank 0.
    paths = {get_zmq_rpc_path_lookup(_config(index)) for index in range(4)}
    assert len(paths) == 4


def test_path_is_stable_for_one_engine():
    assert get_zmq_rpc_path_lookup(_config(3)) == get_zmq_rpc_path_lookup(
        _config(3, local_rank=0, local_engines_only=False)
    )
