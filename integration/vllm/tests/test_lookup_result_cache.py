"""Scheduler-side lookup reuse for requests that stay in the waiting queue."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dfkv_vllm import worker as worker_module
from dfkv_vllm.worker import LookupKeyClient, _parse_lookup_cache_ttl_ms


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _client(monkeypatch, extra=None, answers=None):
    monkeypatch.setattr(worker_module, "make_zmq_socket", lambda *a, **k: MagicMock())
    monkeypatch.setattr(worker_module.zmq, "Context", MagicMock)
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra or {}),
        parallel_config=SimpleNamespace(data_parallel_index=0),
    )
    client = LookupKeyClient(config)
    clock = _Clock()
    client._clock = clock
    calls = []
    answers = answers if answers is not None else {}

    def fake_lookup(token_len, hashes, retain_ttl_s):
        calls.append((token_len, retain_ttl_s))
        result = answers.get(token_len, (token_len, 0))
        if isinstance(result, Exception):
            raise result
        return result

    client._lookup = fake_lookup
    return client, clock, calls


def test_waiting_request_reuses_lookup_within_ttl(monkeypatch):
    client, clock, calls = _client(monkeypatch)
    try:
        # One request re-polled for 150 scheduler steps inside one TTL window.
        for _ in range(150):
            assert client.lookup("r", 4096, [b"h"] * 2) == (4096, 0)
            clock.now += 0.005
        assert len(calls) == 1
    finally:
        client.close()


def test_expired_entry_probes_again(monkeypatch):
    client, clock, calls = _client(monkeypatch)
    try:
        client.lookup("r", 4096, [b"h"])
        clock.now += 1.0  # default TTL is 1000 ms
        client.lookup("r", 4096, [b"h"])
        assert len(calls) == 2
    finally:
        client.close()


def test_changed_length_or_retention_is_a_new_lookup(monkeypatch):
    client, _, calls = _client(monkeypatch)
    try:
        client.lookup("r", 4096, [b"h"])
        client.lookup("r", 6144, [b"h"] * 3)
        client.lookup("r", 6144, [b"h"] * 3, retain_ttl_s=300)
        assert calls == [(4096, 0), (6144, 0), (6144, 300)]
    finally:
        client.close()


def test_requests_do_not_share_entries(monkeypatch):
    client, _, calls = _client(monkeypatch)
    try:
        client.lookup("a", 4096, [b"h"])
        client.lookup("b", 4096, [b"h"])
        assert len(calls) == 2
    finally:
        client.close()


def test_discard_drops_entry(monkeypatch):
    # discard() is what finish/abort/preemption/failed-load paths call.
    client, _, calls = _client(monkeypatch)
    try:
        client.lookup("r", 4096, [b"h"])
        client.discard("r")
        client.lookup("r", 4096, [b"h"])
        assert len(calls) == 2
    finally:
        client.close()


def test_rpc_error_is_not_cached(monkeypatch):
    client, _, calls = _client(
        monkeypatch, answers={4096: RuntimeError("rank-0 gone")}
    )
    try:
        assert client.lookup("r", 4096, [b"h"]) == (0, 0)
        assert client.lookup("r", 4096, [b"h"]) == (0, 0)
        assert len(calls) == 2
    finally:
        client.close()


def test_zero_ttl_disables_reuse(monkeypatch):
    client, _, calls = _client(monkeypatch, extra={"lookup_cache_ttl_ms": 0})
    try:
        for _ in range(3):
            client.lookup("r", 4096, [b"h"])
        assert len(calls) == 3
    finally:
        client.close()


def test_non_blocking_lookup_is_cached_once_answered(monkeypatch):
    client, _, calls = _client(monkeypatch)
    gate = threading.Event()
    original = client._lookup

    def slow(token_len, hashes, retain_ttl_s):
        gate.wait(5)
        return original(token_len, hashes, retain_ttl_s)

    client._lookup = slow
    try:
        assert client.lookup("r", 4096, [b"h"], non_block=True) is None
        gate.set()
        client.futures["r"].result(5)
        assert client.lookup("r", 4096, [b"h"], non_block=True) == (4096, 0)
        assert client.lookup("r", 4096, [b"h"], non_block=True) == (4096, 0)
        assert len(calls) == 1
    finally:
        gate.set()
        client.close()


def test_closed_client_drops_entries(monkeypatch):
    client, _, _ = _client(monkeypatch)
    client.lookup("r", 4096, [b"h"])
    client.close()
    assert client._results == {}
    assert client.lookup("r", 4096, [b"h"]) == (0, 0)


@pytest.mark.parametrize("value,expected", [(0, 0), (1000, 1000), ("250", 250)])
def test_ttl_parsing(value, expected):
    assert _parse_lookup_cache_ttl_ms(value) == expected


@pytest.mark.parametrize("value", [True, -1, 60001, "1.5", None])
def test_ttl_parsing_rejects_invalid(value):
    with pytest.raises(ValueError):
        _parse_lookup_cache_ttl_ms(value)
