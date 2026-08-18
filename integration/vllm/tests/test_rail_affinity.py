import unittest

from dfkv_common import apply_rank_local_rail_affinity
from dfkv_vllm.rail_affinity import physical_affinity_rank


class VllmRailAffinityTest(unittest.TestCase):
    def test_physical_affinity_is_anchored_to_world_local_rank(self):
        layouts = [
            (0, 0, 0, 0, 0),
            (7, 7, 0, 7, 0),  # TP8 / DCP8 / PP1
            (5, 1, 1, 3, 1),  # PCP/DCP/PP coordinates differ
            (2, 6, 0, 2, 0),  # DP engine reuses logical TP coordinates
        ]
        for local_rank, tp_rank, pcp_rank, dcp_rank, pp_rank in layouts:
            with self.subTest(
                layout=(local_rank, tp_rank, pcp_rank, dcp_rank, pp_rank)
            ):
                self.assertEqual(
                    physical_affinity_rank(
                        local_rank=local_rank,
                        tp_rank=tp_rank,
                        pcp_rank=pcp_rank,
                        dcp_rank=dcp_rank,
                        pp_rank=pp_rank,
                    ),
                    local_rank,
                )

    def test_negative_logical_or_physical_rank_is_rejected(self):
        for field in ("local_rank", "tp_rank", "pcp_rank", "dcp_rank", "pp_rank"):
            ranks = dict(
                local_rank=0,
                tp_rank=0,
                pcp_rank=0,
                dcp_rank=0,
                pp_rank=0,
            )
            ranks[field] = -1
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "must be non-negative"):
                    physical_affinity_rank(**ranks)

    def test_tp8_dcp8_rank_to_primary_and_fallback_mapping(self):
        rails = ",".join(f"ib7s400p{i}" for i in range(8))
        for rank in range(8):
            env = {
                "DFKV_RDMA_DEV": rails,
                "DFKV_RDMA_NUMA": "1",
                "DFKV_RDMA_RAIL_TIERS": "unused",
            }
            physical = physical_affinity_rank(
                local_rank=rank,
                tp_rank=rank,
                pcp_rank=0,
                dcp_rank=rank,
                pp_rank=0,
            )
            selection = apply_rank_local_rail_affinity(
                {"rail_affinity": True, "rail_affinity_fallbacks": 1},
                physical,
                env,
            )
            expected = (
                f"ib7s400p{rank}",
                f"ib7s400p{(rank + 1) % 8}",
            )
            self.assertEqual(selection.selected, expected)
            self.assertEqual(env["DFKV_RDMA_DEV"], ",".join(expected))
            self.assertEqual(env["DFKV_RDMA_PRIMARY_DEV"], expected[0])
            self.assertEqual(env["DFKV_RDMA_NUMA"], "0")
            self.assertNotIn("DFKV_RDMA_RAIL_TIERS", env)


if __name__ == "__main__":
    unittest.main()
