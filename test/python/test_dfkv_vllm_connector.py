"""Deterministic scheduler and connector-metrics contracts for vLLM."""

from __future__ import annotations

import logging
from importlib.util import find_spec
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def _install_runtime_dependency_stubs() -> None:
    """Keep imports CPU-only when torch and pyzmq are not installed."""
    if "torch" not in sys.modules and find_spec("torch") is None:
        class Tensor:
            pass

        class Event:
            pass

        torch_module = ModuleType("torch")
        torch_cuda_module = ModuleType("torch.cuda")
        torch_cuda_module.Event = Event
        torch_module.Tensor = Tensor
        torch_module.cuda = torch_cuda_module
        sys.modules["torch"] = torch_module
        sys.modules["torch.cuda"] = torch_cuda_module

    if "zmq" not in sys.modules and find_spec("zmq") is None:
        class Again(Exception):
            pass

        class ZMQError(Exception):
            pass

        class Context:
            pass

        zmq_module = ModuleType("zmq")
        zmq_module.Again = Again
        zmq_module.ZMQError = ZMQError
        zmq_module.Context = Context
        zmq_module.REQ = 1
        zmq_module.REP = 2
        zmq_module.RCVTIMEO = 3
        sys.modules["zmq"] = zmq_module


def _install_vllm_stubs() -> None:
    """Install the import surface used by the connector under standalone Python."""
    if "vllm" in sys.modules or find_spec("vllm") is not None:
        return

    def stub(name: str, **attributes):
        module = ModuleType(name)
        module.__path__ = []
        module.__dict__.update(attributes)
        sys.modules[name] = module
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            setattr(sys.modules[parent_name], child_name, module)
        return module

    class Placeholder:
        pass

    class BlockHash(bytes):
        pass

    @dataclass
    class KVCacheBlock:
        block_id: int

    @dataclass
    class KVConnectorStats:
        data: dict = field(default_factory=dict)

        def is_empty(self) -> bool:
            return not self.data

    class KVConnectorPromMetrics:
        pass

    class PromMetric:
        pass

    class BlockStored:
        def __init__(self, **values) -> None:
            self.__dict__.update(values)

    class KVCacheSpecRegistry:
        @classmethod
        def get_manager_class(cls, _spec):
            return Placeholder

    stub("vllm")
    stub("vllm.envs", VLLM_RPC_BASE_PATH="/tmp")
    stub("vllm.config", VllmConfig=Placeholder, ParallelConfig=Placeholder)
    stub(
        "vllm.distributed",
        get_dcp_group=lambda: Placeholder(),
        get_pcp_group=lambda: Placeholder(),
        get_tensor_model_parallel_rank=lambda: 0,
        get_tensor_model_parallel_world_size=lambda: 1,
    )
    stub("vllm.distributed.kv_events", BlockStored=BlockStored)
    stub("vllm.distributed.kv_transfer")
    stub("vllm.distributed.kv_transfer.kv_connector")
    stub("vllm.distributed.kv_transfer.kv_connector.v1")
    stub(
        "vllm.distributed.kv_transfer.kv_connector.v1.base",
        KVConnectorMetadata=Placeholder,
    )
    stub(
        "vllm.distributed.kv_transfer.kv_connector.v1.metrics",
        KVConnectorPromMetrics=KVConnectorPromMetrics,
        KVConnectorStats=KVConnectorStats,
        PromMetric=PromMetric,
        PromMetricT=PromMetric,
    )
    stub("vllm.logger", init_logger=logging.getLogger)
    stub("vllm.utils")
    stub(
        "vllm.utils.math_utils",
        cdiv=lambda numerator, denominator: (
            numerator + denominator - 1
        )
        // denominator,
    )
    stub(
        "vllm.utils.network_utils",
        make_zmq_socket=lambda *_args, **_kwargs: None,
    )
    stub("vllm.v1")
    stub("vllm.v1.core")
    stub("vllm.v1.core.block_pool", BlockPool=Placeholder)
    stub("vllm.v1.core.kv_cache_manager", KVCacheBlocks=Placeholder)
    stub(
        "vllm.v1.core.kv_cache_utils",
        BlockHash=BlockHash,
        BlockHashList=list[BlockHash],
        BlockHashListWithBlockSize=tuple[list[BlockHash], int],
        KVCacheBlock=KVCacheBlock,
        maybe_convert_block_hash=lambda value: value,
        resolve_kv_cache_block_sizes=lambda *_args, **_kwargs: (64, 64),
    )
    stub(
        "vllm.v1.core.single_type_kv_cache_manager",
        SingleTypeKVCacheManager=Placeholder,
    )
    stub("vllm.v1.core.sched")
    stub(
        "vllm.v1.core.sched.output",
        NewRequestData=Placeholder,
        SchedulerOutput=Placeholder,
    )
    stub(
        "vllm.v1.kv_cache_interface",
        FullAttentionSpec=Placeholder,
        KVCacheConfig=Placeholder,
        KVCacheGroupSpec=Placeholder,
        KVCacheSpec=Placeholder,
        UniformTypeKVCacheSpecs=Placeholder,
    )
    stub(
        "vllm.v1.kv_cache_spec_registry",
        KVCacheSpecRegistry=KVCacheSpecRegistry,
    )
    stub("vllm.v1.request", Request=Placeholder)


_install_runtime_dependency_stubs()
_install_vllm_stubs()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integration" / "common" / "src"))
sys.path.insert(0, str(ROOT / "integration" / "vllm" / "src"))

from dfkv_common import VLLM_RAW_V1, pool_key  # noqa: E402
from dfkv_vllm.data import (  # noqa: E402
    KeyMetadata,
    PoolKey,
)
from dfkv_vllm.metrics import (  # noqa: E402
    DfkvStoreConnectorStats,
    DfkvStorePromMetrics,
)
from dfkv_vllm.scheduler import DfkvStoreScheduler  # noqa: E402


class _LookupClient:
    def __init__(self, results: dict[str, list[int | None]]) -> None:
        self.results = {request_id: list(values) for request_id, values in results.items()}
        self.calls: list[tuple[str, int, tuple[bytes, ...], bool]] = []
        self.discarded: list[str] = []
        self.closed = False

    def lookup(self, request_id, token_len, block_hashes, non_block=False):
        self.calls.append(
            (request_id, token_len, tuple(bytes(value) for value in block_hashes), non_block)
        )
        values = self.results[request_id]
        if len(values) == 1:
            return values[0]
        return values.pop(0)

    def discard(self, request_id: str) -> None:
        self.discarded.append(request_id)

    def close(self) -> None:
        self.closed = True


class SchedulerLookupTest(unittest.TestCase):
    def _scheduler(
        self,
        client: _LookupClient,
        *,
        lookup_async: bool,
        load_async: bool = True,
    ) -> DfkvStoreScheduler:
        transfer = SimpleNamespace(
            kv_role="kv_both",
            kv_connector_extra_config={
                "lookup_async": lookup_async,
                "load_async": load_async,
            },
        )
        config = SimpleNamespace(
            kv_transfer_config=transfer,
            cache_config=SimpleNamespace(cache_salt="stable"),
        )
        with (
            patch("dfkv_vllm.scheduler.ensure_deterministic_block_hashing"),
            patch("dfkv_vllm.scheduler.resolve_kv_cache_block_sizes", return_value=(64, 64)),
            patch("dfkv_vllm.scheduler.LookupKeyClient", return_value=client),
        ):
            return DfkvStoreScheduler(config, SimpleNamespace())

    @staticmethod
    def _request(request_id: str, blocks: int = 2):
        return SimpleNamespace(
            request_id=request_id,
            num_tokens=blocks * 64,
            block_hashes=[bytes([index + 1]) * 32 for index in range(blocks)],
        )

    def test_sync_lookup_remains_single_call_and_returns_existing_shape(self) -> None:
        client = _LookupClient({"sync": [64]})
        scheduler = self._scheduler(client, lookup_async=False)

        self.assertEqual(
            scheduler.get_num_new_matched_tokens(self._request("sync"), 0),
            (64, True),
        )
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][0], "sync")
        self.assertFalse(client.calls[0][3])

    def test_async_first_call_is_pending_then_returns_completed_result(self) -> None:
        client = _LookupClient({"async": [None, 64]})
        scheduler = self._scheduler(client, lookup_async=True)
        request = self._request("async")

        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 0), (None, False))
        self.assertNotIn("async", scheduler.load_specs)
        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 0), (64, True))
        self.assertTrue(all(call[3] for call in client.calls))

    def test_concurrent_request_ids_do_not_share_lookup_state(self) -> None:
        client = _LookupClient({"left": [None, 64], "right": [None, 128]})
        scheduler = self._scheduler(client, lookup_async=True)
        left = self._request("left")
        right = self._request("right", blocks=3)

        self.assertEqual(scheduler.get_num_new_matched_tokens(left, 0), (None, False))
        self.assertEqual(scheduler.get_num_new_matched_tokens(right, 0), (None, False))
        self.assertEqual(scheduler.get_num_new_matched_tokens(right, 0), (128, True))
        self.assertEqual(scheduler.get_num_new_matched_tokens(left, 0), (64, True))
        self.assertEqual([call[0] for call in client.calls], ["left", "right", "right", "left"])

    def test_finish_preemption_abort_and_close_discard_pending_lookups(self) -> None:
        client = _LookupClient(
            {"finished": [None], "preempted": [None], "aborted": [None]}
        )
        scheduler = self._scheduler(client, lookup_async=True)
        for request_id in ("finished", "preempted", "aborted"):
            self.assertEqual(
                scheduler.get_num_new_matched_tokens(self._request(request_id), 0),
                (None, False),
            )

        scheduler.request_finished(self._request("aborted"), ())
        output = SimpleNamespace(
            finished_req_ids={"finished"},
            preempted_req_ids={"preempted"},
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
            num_scheduled_tokens={},
        )
        scheduler.build_connector_meta(output)
        scheduler.close()
        scheduler.close()

        self.assertCountEqual(client.discarded, ["aborted", "finished", "preempted"])
        self.assertTrue(client.closed)

    def test_empty_hash_list_is_a_normal_miss(self) -> None:
        client = _LookupClient({"empty": [0]})
        scheduler = self._scheduler(client, lookup_async=False)
        request = self._request("empty")
        request.block_hashes = []

        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 0), (0, False))
        self.assertEqual(client.calls[0][2], ())


class LogicalChunkNamespaceTest(unittest.TestCase):
    def test_multiwr_v2_component_isolated_from_legacy_layouts(self) -> None:
        metadata = KeyMetadata(
            model_name="model",
            dp_size=2,
            dp_rank=1,
            tp_size=4,
            tp_rank=3,
            pcp_size=1,
            pcp_rank=0,
            dcp_size=1,
            dcp_rank=0,
            pp_size=2,
            pp_rank=1,
            group_id=5,
        )
        chunk_hash = bytes(range(32))
        generated = PoolKey(metadata, chunk_hash).to_bytes()
        coordinates = dict(
            pool="kv",
            dp_size=2,
            dp_rank=1,
            tp_size=4,
            tp_rank=3,
            pcp_size=1,
            pcp_rank=0,
            dcp_size=1,
            dcp_rank=0,
            pp_size=2,
            pp_rank=1,
            group_id=5,
        )
        expected_v2 = pool_key(
            chunk_hash,
            component="vllm-multiwr-v2",
            **coordinates,
        )
        legacy_raw = pool_key(
            chunk_hash,
            component=VLLM_RAW_V1,
            **coordinates,
        )
        legacy_sg = pool_key(
            chunk_hash,
            component="sg-v1",
            **coordinates,
        )

        self.assertEqual(generated, expected_v2)
        self.assertNotEqual(generated, legacy_raw)
        self.assertNotEqual(generated, legacy_sg)


class _PromMetric:
    def __init__(self) -> None:
        self.labels_seen: list[tuple[object, ...]] = []
        self.observed: list[float] = []
        self.increments: list[int] = []

    def labels(self, *values):
        self.labels_seen.append(values)
        return self

    def observe(self, value: float) -> None:
        self.observed.append(value)

    def inc(self, value: int = 1) -> None:
        self.increments.append(value)


class ConnectorStatsTest(unittest.TestCase):
    def test_aggregate_preserves_physical_and_logical_key_accounting(self) -> None:
        first = DfkvStoreConnectorStats()
        first.record_operation(
            "lookup_exists",
            0.001,
            12,
            num_logical_keys=3,
        )
        second = DfkvStoreConnectorStats()
        second.record_operation(
            "lookup_exists",
            0.003,
            8,
            num_logical_keys=2,
            status="partial_failure",
            num_failed_keys=1,
        )

        self.assertIs(first.aggregate(second), first)
        reduced = first.reduce()
        self.assertEqual(reduced["lookup_exists_count"], 2)
        self.assertEqual(reduced["lookup_exists_total_keys"], 20)
        self.assertEqual(reduced["lookup_exists_total_logical_keys"], 5)
        self.assertEqual(reduced["lookup_exists_failed_keys"], 1)
        self.assertEqual(reduced["lookup_exists_error_count"], 0)

    def test_default_logical_count_keeps_existing_producers_compatible(self) -> None:
        stats = DfkvStoreConnectorStats()
        stats.record_operation("save_put", 0.001, 7)
        self.assertEqual(stats.reduce()["save_put_total_logical_keys"], 7)

    def test_phase_observations_aggregate_without_dynamic_labels(self) -> None:
        first = DfkvStoreConnectorStats()
        second = DfkvStoreConnectorStats()
        for name in (
            "lookup_ipc",
            "lookup_queue_wait",
            "receive_queue_wait",
            "geometry_preparation",
        ):
            first.record_observation(name, 0.001)
            second.record_observation(name, 0.003)
        first.aggregate(second)

        self.assertEqual(
            set(first.data),
            {"_observations"},
        )
        reduced = first.reduce()
        for name in (
            "lookup_ipc",
            "lookup_queue_wait",
            "receive_queue_wait",
            "geometry_preparation",
        ):
            self.assertEqual(reduced[f"{name}_count"], 2)
            self.assertEqual(reduced[f"{name}_avg_ms"], 2.0)
            self.assertEqual(reduced[f"{name}_p90_ms"], 3.0)

    def test_prometheus_observe_separates_bounded_phases_and_key_counters(
        self,
    ) -> None:
        operation_metrics = {
            name: _PromMetric()
            for name in (
                "time",
                "calls",
                "keys",
                "logical_keys",
                "bytes",
                "failed_keys",
            )
        }
        observation_metric = _PromMetric()
        prometheus = DfkvStorePromMetrics.__new__(DfkvStorePromMetrics)
        prometheus.per_engine_labelvalues = {0: ["engine-0"]}
        prometheus._metric_cache = {
            (0, "lookup_exists", "ok"): operation_metrics
        }
        prometheus._observation_cache = {}
        prometheus._histogram_observations = {
            "lookup_ipc": observation_metric
        }

        stats = DfkvStoreConnectorStats()
        stats.record_operation(
            "lookup_exists", 0.004, 12, num_logical_keys=3
        )
        stats.record_observation("lookup_ipc", 0.002)
        prometheus.observe(stats.data)

        self.assertEqual(operation_metrics["keys"].increments, [12])
        self.assertEqual(operation_metrics["logical_keys"].increments, [3])
        self.assertEqual(observation_metric.labels_seen, [("engine-0",)])
        self.assertEqual(observation_metric.observed, [0.002])


    def test_metric_dimensions_reject_unbounded_values(self) -> None:
        stats = DfkvStoreConnectorStats()
        with self.assertRaises(ValueError):
            stats.record_operation("lookup_exists:request-123", 0.1, 1)
        with self.assertRaises(ValueError):
            stats.record_operation("lookup_exists", 0.1, 1, status="request-123")
        with self.assertRaises(ValueError):
            stats.record_observation("request-123", 0.1)


if __name__ == "__main__":
    unittest.main()
