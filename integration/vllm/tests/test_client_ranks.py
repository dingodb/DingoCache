"""CLIENT_RANKS clamp semantics (issue #111): must never reject, must never
converge non-replicated layouts, and participant spread must cover exactly
the effective count."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dfkv_vllm.client_ranks import (
    configure_load_convergence,
    participant,
    resolve_client_ranks,
)


def test_default_is_full_participation():
    assert resolve_client_ranks(None, 8, True)[0] == 8
    assert resolve_client_ranks("", 8, True)[0] == 8


def test_auto_converges_only_replicated():
    assert resolve_client_ranks("auto", 8, True)[0] == 1
    eff, reason = resolve_client_ranks("auto", 8, False)
    assert eff == 8 and "clamped" in reason


def test_explicit_n_clamps_but_never_errors():
    assert resolve_client_ranks("2", 8, True)[0] == 2
    eff, reason = resolve_client_ranks("2", 8, False)   # sharded layout
    assert eff == 8 and "kv-not-replicated" in reason
    assert resolve_client_ranks("99", 8, True)[0] == 8  # over TP
    assert resolve_client_ranks("0", 8, True)[0] == 1   # under 1
    eff, reason = resolve_client_ranks("garbage", 8, True)
    assert eff == 8 and "unparseable" in reason


def test_participant_spread_is_even_and_exact():
    # N=2 of TP8: ranks 0 and 4 (different NUMA/NIC halves), indices 0,1.
    got = {r: participant(r, 8, 2) for r in range(8)}
    assert got == {0: 0, 1: None, 2: None, 3: None, 4: 1, 5: None, 6: None, 7: None}
    # N=TP: everyone participates with idx == rank (legacy identity).
    assert [participant(r, 4, 4) for r in range(4)] == [0, 1, 2, 3]
    # N=1: only rank 0.
    assert [participant(r, 4, 1) for r in range(4)] == [0, None, None, None]

def test_load_convergence_enables_native_gpu_rendezvous():
    env: dict[str, str] = {}
    active, reason = configure_load_convergence(env, 8, 1, True)
    assert active and "native-gpu-dedup" in reason
    assert env["DFKV_CLIENT_NODE_DEDUP"] == "1"
    assert env["DFKV_CLIENT_NODE_DEDUP_GPU"] == "1"


def test_load_convergence_clamps_sharded_and_honors_explicit_disable():
    env: dict[str, str] = {}
    assert not configure_load_convergence(env, 8, 1, False)[0]
    assert env == {}
    env = {"DFKV_CLIENT_NODE_DEDUP_GPU": "0"}
    active, reason = configure_load_convergence(env, 8, 1, True)
    assert not active and "disabled" in reason
    assert "DFKV_CLIENT_NODE_DEDUP" not in env
    assert env["DFKV_CLIENT_NODE_DEDUP_GPU"] == "0"


# ---- Phase 2a: producer-side client elision (should_create_client) ----

from dfkv_vllm.client_ranks import should_create_client  # noqa: E402


def test_elide_off_always_creates():
    for role in ("kv_producer", "kv_consumer", "kv_both"):
        for r in range(8):
            ok, reason = should_create_client(role, r, 8, 2, False)
            assert ok and reason == "default(create)"


def test_elide_only_producer_non_participants():
    # tp=8, N=2 -> participants are ranks 0 and 4.
    for r in range(8):
        ok, reason = should_create_client("kv_producer", r, 8, 2, True)
        if r in (0, 4):
            assert ok and reason == "participant"
        else:
            assert not ok and reason.startswith("elided(")


def test_elide_never_touches_consumers_or_both():
    for role in ("kv_consumer", "kv_both"):
        for r in range(8):
            ok, reason = should_create_client(role, r, 8, 2, True)
            assert ok and reason.startswith("clamped(role=")


def test_elide_noop_without_convergence():
    for r in range(8):
        ok, reason = should_create_client("kv_producer", r, 8, 8, True)
        assert ok and reason == "clamped(no-convergence)"


def test_elide_n1_keeps_rank0_only():
    created = [should_create_client("kv_producer", r, 8, 1, True)[0] for r in range(8)]
    assert created == [True] + [False] * 7


class ClientRanksContractTest(unittest.TestCase):
    def test_all_client_rank_contracts(self):
        test_default_is_full_participation()
        test_auto_converges_only_replicated()
        test_explicit_n_clamps_but_never_errors()
        test_participant_spread_is_even_and_exact()
        test_load_convergence_enables_native_gpu_rendezvous()
        test_load_convergence_clamps_sharded_and_honors_explicit_disable()
        test_elide_off_always_creates()
        test_elide_only_producer_non_participants()
        test_elide_never_touches_consumers_or_both()
        test_elide_noop_without_convergence()
        test_elide_n1_keeps_rank0_only()


if __name__ == "__main__":
    unittest.main()
