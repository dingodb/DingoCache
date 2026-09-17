"""Native transfer snapshots and a metadata bridge for older vLLM schedulers."""

from dataclasses import dataclass, field, fields
from functools import wraps
from inspect import signature
from weakref import WeakValueDictionary
from threading import Lock

from vllm.distributed.kv_transfer.kv_connector.v1 import base
from vllm.v1.outputs import KVConnectorOutput


def _require_method(owner, name: str, *parameters: str) -> None:
    method = getattr(owner, name, None)
    if not callable(method):
        raise RuntimeError(f"dfkv requires vLLM {owner.__name__}.{name}")
    actual = signature(method).parameters
    if not all(parameter in actual for parameter in parameters):
        raise RuntimeError(f"Unsupported vLLM {owner.__name__}.{name} signature")


def _require_fields(cls, names: set[str]) -> None:
    if not names.issubset(getattr(cls, "__dataclass_fields__", {})):
        raise RuntimeError(f"Unsupported vLLM {cls.__name__} transfer fields")


_native_results = getattr(base, "KVConnectorTransferResults", None)
_native_method = getattr(base.KVConnectorBase_V1, "get_transfer_results", None)
_native_output = "failed_recving" in {item.name for item in fields(KVConnectorOutput)}
HAS_NATIVE_TRANSFER_RESULTS = _native_results is not None
if HAS_NATIVE_TRANSFER_RESULTS or _native_method is not None or _native_output:
    if not (HAS_NATIVE_TRANSFER_RESULTS and callable(_native_method) and _native_output):
        raise RuntimeError("Incomplete native vLLM request-level transfer protocol")
    _require_fields(
        _native_results, {"finished_sending", "finished_recving", "failed_recving"},
    )
    _require_method(base.KVConnectorBase_V1, "get_transfer_results", "finished_req_ids")
    TransferResults = _native_results
else:
    _require_method(base.KVConnectorBase_V1, "get_finished", "finished_req_ids")
    _require_method(base.KVConnectorBase_V1, "build_connector_worker_meta")
    _require_method(base.KVConnectorBase_V1, "update_connector_output", "connector_output")
    _require_fields(
        KVConnectorOutput,
        {"finished_recving", "invalid_block_ids", "kv_connector_worker_meta"},
    )

    @dataclass
    class TransferResults:
        """One worker snapshot; failed receives also finish receiving."""

        finished_sending: set[str] = field(default_factory=set)
        finished_recving: set[str] = field(default_factory=set)
        failed_recving: set[str] = field(default_factory=set)


_require_method(base.KVConnectorWorkerMetadata, "aggregate", "other")


@dataclass
class LegacyReceiveFailures(base.KVConnectorWorkerMetadata):
    """Worker errors, unioned by vLLM's supported metadata aggregation."""

    failed_recving: set[str] = field(default_factory=set)
    # Only the scheduler bridge marks a local callback packet ready. Worker
    # metadata alone does not prove completion on every worker.
    _ready: bool = field(default=False, init=False, repr=False, compare=False)

    def aggregate(self, other: base.KVConnectorWorkerMetadata):
        if not isinstance(other, LegacyReceiveFailures):
            raise TypeError("Cannot aggregate dfkv receive failures with other metadata")
        self.failed_recving.update(other.failed_recving)
        self._ready = False
        return self


def failed_requests_from_output(output) -> set[str]:
    """Return native failures or failures fenced by the legacy scheduler."""
    native = getattr(output, "failed_recving", None)
    if native is not None:
        return native
    metadata = getattr(output, "kv_connector_worker_meta", None)
    if isinstance(metadata, LegacyReceiveFailures) and metadata._ready:
        return metadata.failed_recving
    return set()


class _RequestErrorBatch(set):
    """Local dispatch only, never sent through the executor or worker wire.

    Its elements are exclusively the original real block IDs. The private
    request set makes the old invalid-block guard dispatch even when there are
    no block errors; it is never interpreted as, or converted to, block IDs.
    """

    def __init__(self, block_ids: set[int], failed_recving: set[str]):
        super().__init__(block_ids)
        self.block_ids = block_ids
        self.failed_recving = failed_recving

    def __bool__(self):
        return bool(self.block_ids or self.failed_recving)


class _PendingFailures:
    def __init__(self, connector):
        self.connector = connector
        # Remember the actual request, not just a reusable string ID, without
        # keeping removed requests (and their prompt/cache state) alive.
        self.requests = WeakValueDictionary()
        # This tree contains only dfkv failure leaves and None foreign leaves.
        # Retaining a whole worker packet would replay other connectors' state.
        self.metadata = None

    def prune(self, scheduler, waiting_status) -> set[str]:
        replaced = set()
        for req_id, request in list(self.requests.items()):
            if scheduler.requests.get(req_id) is not request:
                replaced.add(req_id)
            if (
                req_id in replaced
                or request.status != waiting_status
            ):
                del self.requests[req_id]
        return replaced

_install_lock = Lock()


def install_legacy_failure_bridge() -> None:
    """Install legacy hooks once, including concurrent engine construction."""
    with _install_lock:
        _install_legacy_failure_bridge()


def _install_legacy_failure_bridge() -> None:
    """Install the two legacy dispatch hooks once; native engines stay untouched."""
    if HAS_NATIVE_TRANSFER_RESULTS:
        return

    # Lazy import: connector discovery can occur while vLLM imports Scheduler.
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiKVConnectorWorkerMetadata,
    )
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus

    if getattr(Scheduler.update_from_output, "_dfkv_legacy_failure_bridge", False):
        return
    for name, parameters in (
        ("update_from_output", ("scheduler_output", "model_runner_output")),
        ("_handle_invalid_blocks", ("invalid_block_ids", "num_scheduled_tokens")),
        ("_update_from_kv_xfer_finished", ("kv_connector_output",)),
        ("_update_waiting_for_remote_kv", ("request",)),
        ("finish_requests", ("request_ids", "finished_status")),
    ):
        _require_method(Scheduler, name, *parameters)
    if not hasattr(RequestStatus, "WAITING_FOR_REMOTE_KVS"):
        raise RuntimeError("Unsupported vLLM asynchronous receive lifecycle")

    original_update = Scheduler.update_from_output
    original_invalid = Scheduler._handle_invalid_blocks
    waiting = RequestStatus.WAITING_FOR_REMOTE_KVS

    def failure_leaves(metadata):
        if isinstance(metadata, LegacyReceiveFailures):
            yield metadata
        elif isinstance(metadata, MultiKVConnectorWorkerMetadata):
            for child in metadata.metadata:
                yield from failure_leaves(child)

    def graft_failures(metadata, failures, request_ids, *, ready=False):
        """Copy only allowed failure leaves, preserving current foreign data."""
        if isinstance(metadata, LegacyReceiveFailures) or (
            metadata is None and isinstance(failures, LegacyReceiveFailures)
        ):
            leaf = LegacyReceiveFailures(
                failures.failed_recving.intersection(request_ids)
                if isinstance(failures, LegacyReceiveFailures) else set()
            )
            leaf._ready = ready
            return leaf
        if isinstance(metadata, MultiKVConnectorWorkerMetadata) or (
            metadata is None and isinstance(failures, MultiKVConnectorWorkerMetadata)
        ):
            template = metadata if metadata is not None else failures
            children = tuple(
                graft_failures(
                    metadata.metadata[index] if metadata is not None else None,
                    failures.metadata[index]
                    if isinstance(failures, MultiKVConnectorWorkerMetadata) else None,
                    request_ids,
                    ready=ready,
                )
                for index in range(len(template.metadata))
            )
            if metadata is not None and all(
                child is original for child, original in zip(children, metadata.metadata)
            ):
                return metadata
            return MultiKVConnectorWorkerMetadata(metadata=children)
        return metadata

    @wraps(original_invalid)
    def handle_invalid(self, invalid_block_ids, num_scheduled_tokens):
        if not isinstance(invalid_block_ids, _RequestErrorBatch):
            return original_invalid(self, invalid_block_ids, num_scheduled_tokens)
        batch = invalid_block_ids
        affected = (
            original_invalid(self, batch.block_ids, num_scheduled_tokens)
            if batch.block_ids else set()
        )
        for req_id in batch.failed_recving:
            request = self.requests.get(req_id)
            if request is None or request.status != waiting:
                continue
            if self.recompute_kv_load_failures:
                request.num_computed_tokens = 0
                self.failed_recving_kv_req_ids.add(req_id)
            else:
                affected.add(req_id)
        return affected

    @wraps(original_update)
    def update(self, scheduler_output, model_runner_output):
        packet = model_runner_output.kv_connector_output
        metadata = getattr(packet, "kv_connector_worker_meta", None)
        pending = getattr(self, "_dfkv_pending_receive_failures", None)
        if pending is not None and pending.connector is not self.connector:
            del self._dfkv_pending_receive_failures
            pending = None
        if metadata is None and pending is None:
            return original_update(self, scheduler_output, model_runner_output)
        leaves = tuple(failure_leaves(metadata))
        if not leaves and pending is None:
            return original_update(self, scheduler_output, model_runner_output)
        if pending is None:
            pending = _PendingFailures(self.connector)
            self._dfkv_pending_receive_failures = pending

        replaced = pending.prune(self, waiting)
        pending.metadata = graft_failures(None, pending.metadata, pending.requests)
        for leaf in leaves:
            for req_id in leaf.failed_recving:
                request = self.requests.get(req_id)
                if (
                    req_id not in replaced and request is not None
                    and request.status == waiting
                ):
                    pending.requests[req_id] = request
        if leaves:
            incoming = graft_failures(None, metadata, pending.requests)
            pending.metadata = (
                pending.metadata.aggregate(incoming)
                if pending.metadata is not None else incoming
            )

        if packet is None:
            try:
                return original_update(self, scheduler_output, model_runner_output)
            finally:
                pending.prune(self, waiting)
                if not pending.requests:
                    del self._dfkv_pending_receive_failures

        ready = set(pending.requests).intersection(packet.finished_recving or ())
        original_blocks = packet.invalid_block_ids
        original_finished = packet.finished_recving
        callback_metadata = graft_failures(
            metadata, pending.metadata, ready, ready=True,
        )
        try:
            # Ignore already-retired receive notifications only in this dfkv
            # scope. Keep terminal requests still owned by the engine: its
            # existing completion path must free their deferred allocations.
            packet.finished_recving = {
                req_id for req_id in original_finished or ()
                if req_id not in replaced
                and (request := self.requests.get(req_id)) is not None
                and (request.status == waiting or request.is_finished())
            }
            packet.kv_connector_worker_meta = callback_metadata
            if ready:
                packet.invalid_block_ids = _RequestErrorBatch(original_blocks, ready)
            result = original_update(self, scheduler_output, model_runner_output)
            for req_id in ready:
                pending.requests.pop(req_id, None)
            return result
        finally:
            packet.invalid_block_ids = original_blocks
            packet.finished_recving = original_finished
            packet.kv_connector_worker_meta = metadata
            pending.prune(self, waiting)
            if not pending.requests:
                del self._dfkv_pending_receive_failures

    update._dfkv_legacy_failure_bridge = True
    Scheduler._handle_invalid_blocks = handle_invalid
    Scheduler.update_from_output = update
