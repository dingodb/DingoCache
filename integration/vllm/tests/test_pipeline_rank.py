"""PP stage derivation must follow vLLM's DP x PP x PCP x TP rank layout."""
import itertools

from dfkv_vllm.worker import pipeline_rank


def test_matches_vllm_rank_layout():
    tp, pcp, pp = 2, 2, 3
    for pp_rank, pcp_rank, tp_rank in itertools.product(
        range(pp), range(pcp), range(tp)
    ):
        rank = (pp_rank * pcp + pcp_rank) * tp + tp_rank
        assert pipeline_rank(rank, tp, pcp, pp) == pp_rank


def test_without_pcp_keeps_the_previous_formula():
    for rank in range(8):
        assert pipeline_rank(rank, 2, 1, 4) == (rank // 2) % 4
