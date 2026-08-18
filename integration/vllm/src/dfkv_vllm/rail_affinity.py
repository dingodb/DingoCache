from __future__ import annotations


def physical_affinity_rank(
    *,
    local_rank: int,
    tp_rank: int,
    pcp_rank: int,
    dcp_rank: int,
    pp_rank: int,
) -> int:
    """Return the per-host process/GPU coordinate used for HCA affinity.

    TP/PCP/DCP/PP are storage and communication coordinates and can repeat
    across stages or subgroups. vLLM's world-group ``local_rank`` is the
    physical per-host process coordinate, so it alone anchors the ordered HCA
    list. The logical ranks are accepted explicitly to keep their
    non-participation visible and testable.
    """
    physical = int(local_rank)
    logical = (int(tp_rank), int(pcp_rank), int(dcp_rank), int(pp_rank))
    if physical < 0 or any(rank < 0 for rank in logical):
        raise ValueError("vLLM rail affinity ranks must be non-negative")
    return physical
