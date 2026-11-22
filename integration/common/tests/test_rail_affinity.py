import unittest

from dfkv_common import apply_rank_local_rail_affinity


class RailAffinityTest(unittest.TestCase):
    def test_disabled_leaves_environment_unchanged(self):
        env = {
            "DFKV_RDMA_DEV": "ib0,ib1,ib2,ib3",
            "DFKV_RDMA_NUMA": "1",
            "DFKV_RDMA_RAIL_TIERS": "ib0|ib1;ib2|ib3",
        }
        before = dict(env)
        result = apply_rank_local_rail_affinity({}, 2, env)
        self.assertFalse(result.enabled)
        self.assertFalse(result.applied)
        self.assertEqual(env, before)

    def test_primary_and_default_fallback_wrap_by_physical_rank(self):
        env = {
            "DFKV_RDMA_DEV": "ib0,ib1,ib2,ib3",
            "DFKV_RDMA_NUMA": "1",
            "DFKV_RDMA_RAIL_TIERS": "ib0|ib1;ib2|ib3",
        }
        result = apply_rank_local_rail_affinity({"rail_affinity": True}, 3, env)
        self.assertEqual(result.selected, ("ib3", "ib1"))
        self.assertEqual(result.primary, "ib3")
        self.assertEqual(result.fallback_count, 1)
        self.assertEqual(env["DFKV_RDMA_DEV"], "ib3,ib1")
        self.assertEqual(env["DFKV_RDMA_PRIMARY_DEV"], "ib3")
        self.assertEqual(env["DFKV_RDMA_NUMA"], "0")
        self.assertNotIn("DFKV_RDMA_RAIL_TIERS", env)

    def test_fallback_is_not_a_neighbouring_rail(self):
        # Two neighbouring rails (two ports of one adapter) down on a server:
        # with the next rail as fallback, the rank whose primary is the first
        # of them shares no rail with that server.
        rails = [f"ib{i}" for i in range(8)]
        for down in range(8):
            failed = {rails[down], rails[(down + 1) % 8]}
            for rank in range(8):
                env = {"DFKV_RDMA_DEV": ",".join(rails)}
                result = apply_rank_local_rail_affinity(
                    {"rail_affinity": True}, rank, env)
                self.assertTrue(set(result.selected) - failed,
                                (rank, sorted(failed), result.selected))

    def test_fallbacks_are_spread_and_distinct(self):
        rails = ",".join(f"ib{i}" for i in range(8))
        for count, expected in ((1, ("ib1", "ib5")), (2, ("ib1", "ib3", "ib6")),
                                (3, ("ib1", "ib3", "ib5", "ib7")),
                                (7, tuple(f"ib{(1 + i) % 8}" for i in range(8)))):
            result = apply_rank_local_rail_affinity(
                {"rail_affinity": True, "rail_affinity_fallbacks": count}, 1,
                {"DFKV_RDMA_DEV": rails})
            self.assertEqual(result.selected, expected, count)

    def test_explicit_zero_disables_fallback_not_affinity(self):
        env = {"DFKV_RDMA_DEV": "ib0,ib1,ib2,ib3"}
        result = apply_rank_local_rail_affinity(
            {"rail_affinity": "1", "rail_affinity_fallbacks": 0}, 6, env
        )
        self.assertTrue(result.applied)
        self.assertEqual(result.selected, ("ib2",))
        self.assertEqual(env["DFKV_RDMA_PRIMARY_DEV"], "ib2")

    def test_fallback_count_is_bounded_to_available_rails(self):
        env = {"DFKV_RDMA_DEV": "ib0,ib1,ib2"}
        result = apply_rank_local_rail_affinity(
            {"rail_affinity": True, "rail_affinity_fallbacks": 99}, 1, env
        )
        self.assertEqual(result.selected, ("ib1", "ib2", "ib0"))  # all 3
        self.assertEqual(result.fallback_count, 2)

    def test_single_or_empty_rail_is_a_clean_noop(self):
        for devs, expected_primary in (("ib0", "ib0"), ("", None)):
            env = {"DFKV_RDMA_DEV": devs, "DFKV_RDMA_NUMA": "1"}
            before = dict(env)
            result = apply_rank_local_rail_affinity({"rail_affinity": True}, 0, env)
            self.assertFalse(result.applied)
            self.assertEqual(result.primary, expected_primary)
            self.assertEqual(env, before)

    def test_invalid_configuration_fails_before_native_open(self):
        bad_cases = [
            (
                {"rail_affinity": True, "rail_affinity_fallbacks": True},
                {"DFKV_RDMA_DEV": "ib0,ib1"},
            ),
            (
                {"rail_affinity": True, "rail_affinity_fallbacks": "bad"},
                {"DFKV_RDMA_DEV": "ib0,ib1"},
            ),
            ({"rail_affinity": True}, {"DFKV_RDMA_DEV": "ib0,ib0"}),
        ]
        for cfg, env in bad_cases:
            with self.subTest(cfg=cfg, env=env):
                with self.assertRaises(ValueError):
                    apply_rank_local_rail_affinity(cfg, 0, env)

    def test_negative_rank_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_rank_local_rail_affinity(
                {"rail_affinity": True},
                -1,
                {"DFKV_RDMA_DEV": "ib0,ib1"},
            )


if __name__ == "__main__":
    unittest.main()
