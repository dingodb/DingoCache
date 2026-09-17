"""Exercise legacy metadata fencing through vLLM's real aggregator/scheduler."""

from dataclasses import dataclass
from types import SimpleNamespace
import weakref

import pytest

pytest.importorskip("vllm")

from vllm import SamplingParams
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1 import base
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from dfkv_vllm import transfer_protocol as protocol


def request(req_id, status=RequestStatus.WAITING_FOR_REMOTE_KVS):
    req = Request(req_id, list(range(32)), SamplingParams(max_tokens=4), None)
    req.status = status
    req.num_computed_tokens = 16
    return req


def worker_output(finished=(), failed=(), *, blocks=(), metadata=True):
    return ModelRunnerOutput(
        req_ids=[], req_id_to_index={},
        kv_connector_output=KVConnectorOutput(
            finished_recving=set(finished), invalid_block_ids=set(blocks),
            kv_connector_worker_meta=(
                protocol.LegacyReceiveFailures(set(failed)) if metadata else None
            ),
        ),
    )


@pytest.fixture
def legacy_scheduler(monkeypatch):
    if protocol.HAS_NATIVE_TRANSFER_RESULTS:
        pytest.skip("Legacy integration runs against the old engine installations")
    # Restore process-wide hooks after this test, even when another connector
    # has already installed them during test collection.
    monkeypatch.setattr(Scheduler, "update_from_output", Scheduler.update_from_output)
    monkeypatch.setattr(Scheduler, "_handle_invalid_blocks", Scheduler._handle_invalid_blocks)
    protocol.install_legacy_failure_bridge()
    protocol.install_legacy_failure_bridge()

    def make(*requests, recompute=True):
        scheduler = object.__new__(Scheduler)
        scheduler.requests = {req.request_id: req for req in requests}
        scheduler.running = []
        scheduler.waiting = FCFSRequestQueue()
        scheduler.skipped_waiting = FCFSRequestQueue(requests)
        scheduler.recompute_kv_load_failures = recompute
        scheduler.failed_recving_kv_req_ids = set()
        scheduler.finished_recving_kv_req_ids = set()
        scheduler.finished_req_ids = set()
        scheduler.finished_req_ids_dict = None
        scheduler.grammar_compile_error_reqs = set()
        scheduler.defer_block_free = False
        scheduler.perf_metrics = None
        scheduler.ec_connector = None
        scheduler._inflight_prefills = set()
        scheduler.needs_kv_cache_zeroing = False
        scheduler.block_size = 4
        scheduler.make_stats = lambda *args: None
        scheduler._connector_finished = lambda req: (False, None)
        scheduler.encoder_cache_manager = SimpleNamespace(free=lambda req: None)
        scheduler.freed = []
        scheduler.cached = []
        scheduler.evicted = []
        scheduler.tables = {}
        scheduler.kv_cache_manager = SimpleNamespace(
            take_events=lambda: [],
            free=lambda req: scheduler.freed.append(req.request_id),
            cache_blocks=lambda req, count: scheduler.cached.append((req.request_id, count)),
            get_block_ids=lambda req_id: scheduler.tables[req_id],
            evict_blocks=lambda blocks: scheduler.evicted.append(set(blocks)),
        )
        scheduler.callbacks = []
        scheduler.connector = SimpleNamespace(
            update_connector_output=lambda output: scheduler.callbacks.append(
                set(protocol.failed_requests_from_output(output))
            ),
            get_kv_connector_stats=lambda: None,
            take_events=lambda: [],
        )
        return scheduler

    return make


def step(scheduler, output):
    return scheduler.update_from_output(
        SimpleNamespace(num_scheduled_tokens={}, total_num_scheduled_tokens=0), output,
    )


def multi_connector(*children):
    connector = object.__new__(MultiConnector)
    connector._connectors = list(children)
    return connector


def multi_worker_output(finished=(), *, metadata):
    def worker(child):
        if isinstance(child, tuple):
            return multi_connector(*(worker(item) for item in child))
        return SimpleNamespace(build_connector_worker_meta=lambda: child)

    output = worker_output(finished, metadata=False)
    output.kv_connector_output.kv_connector_worker_meta = (
        worker(metadata).build_connector_worker_meta()
    )
    return output


def dfkv_child(*requests):
    from dfkv_vllm.scheduler import DfkvStoreScheduler

    state = object.__new__(DfkvStoreScheduler)
    state.request_level_loads = True
    state._unfinished_request_ids = {req.request_id for req in requests}
    state._failed_load_req_ids = set()
    state._preempted_req_ids = set()
    state._allocated_req_ids = set(state._unfinished_request_ids)
    state._unfinished_requests = {}
    state._request_trackers = {}
    state.load_specs = {}
    discarded = []
    state.client = SimpleNamespace(discard=discarded.append)
    return SimpleNamespace(
        state=state, discarded=discarded,
        update_connector_output=state.update_connector_output,
        get_kv_connector_stats=lambda: None,
        take_events=lambda: [],
    )


@dataclass
class ForeignMetadata(base.KVConnectorWorkerMetadata):
    events: tuple[str, ...]

    def aggregate(self, other):
        return ForeignMetadata(self.events + other.events)


def foreign_child():
    received = []
    return SimpleNamespace(
        received=received,
        update_connector_output=lambda output: received.append(
            output.kv_connector_worker_meta
        ),
        get_kv_connector_stats=lambda: None,
        take_events=lambda: [],
    )


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("late_foreign", [False, True])
def test_multi_connector_routes_fenced_failures_without_replaying_foreign_metadata(
    legacy_scheduler, nested, recompute, late_foreign,
):
    first, second, good = request("first"), request("second"), request("good")
    scheduler = legacy_scheduler(first, second, good, recompute=recompute)
    # Both dfkv children know every request, so misrouting would quarantine a
    # healthy source too. Only the child whose worker failed may be quarantined.
    left, right = dfkv_child(first, second, good), dfkv_child(first, second, good)
    foreign = foreign_child()
    scheduler.connector = (
        multi_connector(left, multi_connector(right, foreign))
        if nested else multi_connector(left, right, foreign)
    )

    def layout(left_meta, right_meta, foreign_meta):
        return (
            (left_meta, (right_meta, foreign_meta))
            if nested else (left_meta, right_meta, foreign_meta)
        )

    aggregator = KVOutputAggregator(2)
    early = aggregator.aggregate([
        multi_worker_output({"first"}, metadata=layout(
            protocol.LegacyReceiveFailures({"first"}), None, ForeignMetadata(("rank0",)),
        )),
        multi_worker_output({"second"}, metadata=layout(
            None, protocol.LegacyReceiveFailures({"second"}), ForeignMetadata(("rank1",)),
        )),
    ])
    original_metadata = early.kv_connector_output.kv_connector_worker_meta
    assert step(scheduler, early) == {}
    assert early.kv_connector_output.kv_connector_worker_meta is original_metadata
    assert (first.num_computed_tokens, second.num_computed_tokens) == (16, 16)
    assert left.discarded == right.discarded == []
    assert foreign.received[0].events == ("rank0", "rank1")
    original_foreign = (
        original_metadata.metadata[1].metadata[1]
        if nested else original_metadata.metadata[2]
    )
    assert foreign.received[0] is original_foreign

    fresh_foreign = ForeignMetadata(("completion",)) if late_foreign else None
    late = aggregator.aggregate([
        multi_worker_output({"second"}, metadata=layout(None, None, fresh_foreign)),
        multi_worker_output({"first"}, metadata=layout(None, None, None)),
    ])
    original_metadata = late.kv_connector_output.kv_connector_worker_meta
    outputs = step(scheduler, late)
    assert late.kv_connector_output.kv_connector_worker_meta is original_metadata
    assert left.discarded == ["first"]
    assert right.discarded == ["second"]
    assert left.state._failed_load_req_ids == {"first"}
    assert right.state._failed_load_req_ids == {"second"}
    assert foreign.received == [original_foreign, fresh_foreign]
    assert foreign.received[-1] is fresh_foreign
    assert good.num_computed_tokens == 16
    if recompute:
        assert outputs == {}
        assert (first.num_computed_tokens, second.num_computed_tokens) == (0, 0)
        assert scheduler.failed_recving_kv_req_ids == {"first", "second"}
        scheduler._update_waiting_for_remote_kv(first)
        scheduler._update_waiting_for_remote_kv(second)
        assert scheduler.freed == ["first", "second"]
        assert scheduler.cached == []
    else:
        assert first.status == second.status == RequestStatus.FINISHED_ERROR
        assert {item.request_id for item in outputs[0].outputs} == {"first", "second"}
        assert set(scheduler.freed) == {"first", "second"}


def test_nested_pending_failures_complete_independently(legacy_scheduler):
    first, second = request("first"), request("second")
    scheduler = legacy_scheduler(first, second)
    child, foreign = dfkv_child(first, second), foreign_child()
    scheduler.connector = multi_connector(foreign, multi_connector(child))
    aggregator = KVOutputAggregator(2)
    step(scheduler, aggregator.aggregate([
        multi_worker_output({"first", "second"}, metadata=(
            ForeignMetadata(("early",)),
            (protocol.LegacyReceiveFailures({"first", "second"}),),
        )),
        multi_worker_output(metadata=(None, (None,))),
    ]))
    assert child.discarded == []

    step(scheduler, aggregator.aggregate([
        multi_worker_output(metadata=(None, (None,))),
        multi_worker_output({"first"}, metadata=(
            ForeignMetadata(("partial",)), (None,),
        )),
    ]))
    assert child.discarded == ["first"]
    assert first.num_computed_tokens == 0
    assert second.num_computed_tokens == 16
    step(scheduler, aggregator.aggregate([
        multi_worker_output(metadata=(None, (None,))),
        multi_worker_output({"second"}, metadata=(None, (None,))),
    ]))
    assert child.discarded == ["first", "second"]
    assert second.num_computed_tokens == 0
    assert [
        metadata.events if metadata is not None else None
        for metadata in foreign.received
    ] == [("early",), ("partial",), None]


@pytest.mark.parametrize("retirement", ["cancel", "replace", "remove", "packetless"])
def test_nested_pending_failures_do_not_poison_retired_requests(
    legacy_scheduler, retirement,
):
    req = request("retired")
    scheduler = legacy_scheduler(req)
    child, foreign = dfkv_child(req), foreign_child()
    scheduler.connector = multi_connector(foreign, multi_connector(child))
    aggregator = KVOutputAggregator(2)
    step(scheduler, aggregator.aggregate([
        multi_worker_output({"retired"}, metadata=(
            None, (protocol.LegacyReceiveFailures({"retired"}),),
        )),
        multi_worker_output(metadata=(None, (None,))),
    ]))
    if retirement == "cancel":
        scheduler.finish_requests({"retired"}, RequestStatus.FINISHED_ABORTED)
        assert scheduler.freed == []
    else:
        scheduler.requests.clear()
        scheduler.skipped_waiting.clear()
        if retirement == "packetless":
            step(scheduler, ModelRunnerOutput(req_ids=[], req_id_to_index={}))
        if retirement == "remove":
            reference = weakref.ref(req)
            del req
            assert reference() is None
        replacement = request("retired")
        scheduler.requests["retired"] = replacement
        scheduler.skipped_waiting.add_request(replacement)
    repeat = (
        protocol.LegacyReceiveFailures({"retired"})
        if retirement in {"cancel", "replace"} else None
    )
    step(scheduler, aggregator.aggregate([
        multi_worker_output(metadata=(None, (None,))),
        multi_worker_output({"retired"}, metadata=(None, (repeat,))),
    ]))
    assert child.discarded == []
    assert not child.state._failed_load_req_ids
    assert not scheduler.failed_recving_kv_req_ids
    if retirement == "cancel":
        assert scheduler.freed == ["retired"]
        assert "retired" not in scheduler.requests
    else:
        assert replacement.num_computed_tokens == 16
        assert scheduler.freed == []
        if retirement == "replace":
            assert not scheduler.finished_recving_kv_req_ids


def test_foreign_only_multi_metadata_is_not_intercepted(legacy_scheduler):
    scheduler = legacy_scheduler(request("foreign"))
    left, right = foreign_child(), foreign_child()
    scheduler.connector = multi_connector(left, multi_connector(right))
    output = multi_worker_output(metadata=(
        ForeignMetadata(("left",)), (ForeignMetadata(("right",)),),
    ))
    original = output.kv_connector_output.kv_connector_worker_meta
    step(scheduler, output)
    assert output.kv_connector_output.kv_connector_worker_meta is original
    assert left.received == [original.metadata[0]]
    assert left.received[0] is original.metadata[0]
    assert right.received == [original.metadata[1].metadata[0]]
    assert right.received[0] is original.metadata[1].metadata[0]


@pytest.mark.parametrize("recompute", [False, True])
def test_staggered_workers_wait_for_all_ranks_before_recovery(legacy_scheduler, recompute):
    failed, unaffected = request("failed"), request("unaffected")
    scheduler = legacy_scheduler(failed, unaffected, recompute=recompute)
    aggregator = KVOutputAggregator(expected_finished_count=2)
    early = aggregator.aggregate([
        worker_output({"failed"}, {"failed"}), worker_output(),
    ])
    assert step(scheduler, early) == {}
    assert failed.num_computed_tokens == 16
    assert scheduler.freed == []
    assert scheduler.callbacks == [set()]
    assert not scheduler.failed_recving_kv_req_ids
    # Rank zero has already drained its failure metadata. Only the final
    # worker completes now, so the bridge must retain the earlier failure.
    late = aggregator.aggregate([
        worker_output(metadata=False), worker_output({"failed"}, metadata=False),
    ])
    outputs = step(scheduler, late)
    assert scheduler.callbacks[-1] == {"failed"}
    assert unaffected.num_computed_tokens == 16
    assert unaffected.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    if recompute:
        assert outputs == {}
        assert failed.num_computed_tokens == 0
        assert scheduler.failed_recving_kv_req_ids == {"failed"}
        assert scheduler.finished_recving_kv_req_ids == {"failed"}
        scheduler._update_waiting_for_remote_kv(failed)
        assert scheduler.freed == ["failed"]
        assert scheduler.cached == []
        assert not scheduler.failed_recving_kv_req_ids
    else:
        assert failed.status == RequestStatus.FINISHED_ERROR
        assert [item.request_id for item in outputs[0].outputs] == ["failed"]
        assert outputs[0].outputs[0].finish_reason == failed.get_finished_reason()
        assert "failed" not in scheduler.requests
        assert scheduler.freed == ["failed"]


def test_same_step_worker_metadata_unions_failures(legacy_scheduler):
    first, second, good = request("first"), request("second"), request("good")
    scheduler = legacy_scheduler(first, second, good)
    aggregator = KVOutputAggregator(2)
    completed = {"first", "second", "good"}
    result = aggregator.aggregate([
        worker_output(completed, {"first"}),
        worker_output(completed, {"second"}),
    ])
    step(scheduler, result)
    assert (first.num_computed_tokens, second.num_computed_tokens) == (0, 0)
    assert good.num_computed_tokens == 16
    assert scheduler.callbacks == [{"first", "second"}]


def test_request_errors_do_not_scan_or_alias_hybrid_block_ids(legacy_scheduler):
    failed, other = request("failed"), request("other")
    running = request("running", RequestStatus.RUNNING)
    scheduler = legacy_scheduler(failed, other, running)
    # No tables are supplied: any block lookup would fail. Request-level
    # identities remain unambiguous even if physical pools reuse block IDs.
    step(scheduler, worker_output({"failed"}, {"failed", "running", "missing"}))
    assert failed.num_computed_tokens == 0
    assert (other.num_computed_tokens, running.num_computed_tokens) == (16, 16)
    assert scheduler.failed_recving_kv_req_ids == {"failed"}
    assert scheduler.callbacks == [{"failed"}]


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("request_error", [False, True])
def test_real_block_errors_keep_engine_prefix_recovery(
    legacy_scheduler, recompute, request_error,
):
    full, untouched, hybrid = request("full"), request("untouched"), request("hybrid")
    scheduler = legacy_scheduler(full, untouched, hybrid, recompute=recompute)
    scheduler.tables = {
        "full": ([10, 11, 12, 13],),
        "untouched": ([20, 21, 22, 23],),
        "hybrid": ([30, 31, 32, 33],),
    }
    result = step(scheduler, worker_output(
        {"full", "hybrid"}, {"hybrid"} if request_error else (), blocks={12},
    ))
    if recompute:
        assert full.num_computed_tokens == 8
        assert hybrid.num_computed_tokens == (0 if request_error else 16)
        assert scheduler.failed_recving_kv_req_ids == (
            {"full", "hybrid"} if request_error else {"full"}
        )
    else:
        assert full.status == RequestStatus.FINISHED_ERROR
        assert {item.request_id for item in result[0].outputs} == (
            {"full", "hybrid"} if request_error else {"full"}
        )
    assert untouched.num_computed_tokens == 16
    assert untouched.status == RequestStatus.WAITING_FOR_REMOTE_KVS


def test_abort_before_last_worker_releases_only_after_completion(legacy_scheduler):
    req = request("cancelled")
    scheduler = legacy_scheduler(req)
    step(scheduler, worker_output(failed={"cancelled"}))
    scheduler.finish_requests({"cancelled"}, RequestStatus.FINISHED_ABORTED)
    assert scheduler.freed == []
    # The engine intentionally keeps a cancelled receive in requests until
    # its all-worker completion. The bridge must not quarantine it again.
    step(scheduler, worker_output({"cancelled"}, {"cancelled", "unknown"}))
    assert scheduler.freed == ["cancelled"]
    assert scheduler.callbacks[-1] == set()
    assert "cancelled" not in scheduler.requests
    assert not scheduler.failed_recving_kv_req_ids
    # Duplicate/late dfkv metadata and completions cannot resurrect state.
    step(scheduler, worker_output({"cancelled"}, {"cancelled"}))
    replacement = request("cancelled")
    scheduler.requests["cancelled"] = replacement
    scheduler.skipped_waiting.add_request(replacement)
    step(scheduler, worker_output({"cancelled"}))
    assert replacement.num_computed_tokens == 16
    assert scheduler.callbacks[-1] == set()
    assert scheduler.freed == ["cancelled"]


def test_pending_failures_do_not_hold_removed_requests_alive(legacy_scheduler):
    req = request("removed")
    scheduler = legacy_scheduler(req)
    step(scheduler, worker_output(failed={"removed"}))
    reference = weakref.ref(req)
    scheduler.requests.clear()
    scheduler.skipped_waiting.clear()
    del req
    assert reference() is None
    replacement = request("removed")
    scheduler.requests["removed"] = replacement
    step(scheduler, worker_output({"removed"}))
    assert replacement.num_computed_tokens == 16
    assert scheduler.callbacks[-1] == set()


def test_replaced_request_does_not_consume_previous_failure_or_completion(legacy_scheduler):
    old = request("reused")
    scheduler = legacy_scheduler(old)
    step(scheduler, worker_output(failed={"reused"}))
    replacement = request("reused")
    scheduler.requests["reused"] = replacement
    scheduler.skipped_waiting.clear()
    scheduler.skipped_waiting.add_request(replacement)
    step(scheduler, worker_output({"reused"}, {"reused"}))
    assert replacement.num_computed_tokens == 16
    assert not scheduler.finished_recving_kv_req_ids
    assert not scheduler.failed_recving_kv_req_ids
    assert scheduler.callbacks[-1] == set()


def test_packetless_step_prunes_retired_failures(legacy_scheduler):
    req = request("retired")
    scheduler = legacy_scheduler(req)
    step(scheduler, worker_output(failed={"retired"}))
    scheduler.requests.clear()
    scheduler.skipped_waiting.clear()
    step(scheduler, ModelRunnerOutput(req_ids=[], req_id_to_index={}))
    replacement = request("retired")
    scheduler.requests["retired"] = replacement
    step(scheduler, worker_output({"retired"}))
    assert replacement.num_computed_tokens == 16
    assert scheduler.callbacks[-1] == set()


def test_switching_connector_drops_dfkv_pending_scope(legacy_scheduler):
    req = request("old")
    scheduler = legacy_scheduler(req)
    step(scheduler, worker_output(failed={"old"}))
    marker = object()
    observed = []
    scheduler.connector = SimpleNamespace(
        update_connector_output=lambda packet: observed.append(
            packet.kv_connector_worker_meta
        ),
        get_kv_connector_stats=lambda: None,
        take_events=lambda: [],
    )
    output = worker_output({"old"}, metadata=False)
    output.kv_connector_output.kv_connector_worker_meta = marker
    step(scheduler, output)
    assert observed == [marker]
    assert req.num_computed_tokens == 16
    assert not scheduler.failed_recving_kv_req_ids


@pytest.mark.parametrize("failure_point", ["callback", "block_handler"])
def test_temporary_packet_is_restored_when_engine_raises(
    legacy_scheduler, failure_point,
):
    req = request("load")
    scheduler = legacy_scheduler(req)
    output = worker_output({"load"}, {"load"}, blocks={12} if failure_point == "block_handler" else ())
    packet = output.kv_connector_output
    original_blocks = packet.invalid_block_ids
    original_finished = packet.finished_recving
    original_metadata = packet.kv_connector_worker_meta

    def fail(*args):
        raise RuntimeError("injected engine failure")

    if failure_point == "callback":
        scheduler.connector.update_connector_output = fail
    else:
        scheduler.kv_cache_manager.get_block_ids = fail
    with pytest.raises(RuntimeError, match="injected engine failure"):
        step(scheduler, output)
    assert packet.invalid_block_ids is original_blocks
    assert packet.finished_recving is original_finished
    assert packet.kv_connector_worker_meta is original_metadata
    assert protocol.failed_requests_from_output(packet) == set()


def test_unrelated_connector_is_not_intercepted(legacy_scheduler):
    req = request("other")
    scheduler = legacy_scheduler(req)
    marker = object()
    output = worker_output(metadata=False)
    output.kv_connector_output.kv_connector_worker_meta = marker
    observed = []
    scheduler.connector.update_connector_output = lambda packet: observed.append(
        packet.kv_connector_worker_meta
    )
    step(scheduler, output)
    assert observed == [marker]
    assert req.num_computed_tokens == 16


def test_native_engine_uses_native_result_without_scheduler_hooks():
    if not protocol.HAS_NATIVE_TRANSFER_RESULTS:
        pytest.skip("Native route runs against the native engine installation")
    original_update = Scheduler.update_from_output
    original_handler = Scheduler._handle_invalid_blocks
    protocol.install_legacy_failure_bridge()
    protocol.install_legacy_failure_bridge()
    assert Scheduler.update_from_output is original_update
    assert Scheduler._handle_invalid_blocks is original_handler
    assert protocol.TransferResults is base.KVConnectorTransferResults
    output = KVConnectorOutput(finished_recving={"load"}, failed_recving={"load"})
    assert protocol.failed_requests_from_output(output) == {"load"}


def test_raw_worker_metadata_is_not_a_scheduler_failure():
    metadata = protocol.LegacyReceiveFailures({"load"})
    output = SimpleNamespace(kv_connector_worker_meta=metadata)
    assert protocol.failed_requests_from_output(output) == set()
    metadata._ready = True
    assert protocol.failed_requests_from_output(output) == {"load"}
    metadata.aggregate(protocol.LegacyReceiveFailures({"second"}))
    assert protocol.failed_requests_from_output(output) == set()
