import importlib.util
import os
import sys
import types
import unittest

_TEST_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_TEST_DIR, "..", "..", "common", "src"))

_MODULE_PATH = os.path.join(
    _TEST_DIR, "..", "src", "dfkv_connector", "rail_affinity.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_dfkv_lmcache_rail_affinity", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

from dfkv_common import apply_rank_local_rail_affinity  # noqa: E402

parse_plugin_rail_affinity = _MODULE.parse_plugin_rail_affinity
physical_affinity_rank = _MODULE.physical_affinity_rank


class LMCacheRailAffinityTest(unittest.TestCase):
    def test_plugin_config_defaults_to_disabled_with_one_fallback(self):
        config = parse_plugin_rail_affinity({}, "remote_storage_plugin.dfkv")
        self.assertFalse(config.enabled)
        self.assertEqual(config.fallback_count, 1)

    def test_plugin_config_parses_explicit_affinity(self):
        config = parse_plugin_rail_affinity(
            {
                "remote_storage_plugin.dfkv.rail_affinity": True,
                "remote_storage_plugin.dfkv.rail_affinity_fallbacks": 2,
            },
            "remote_storage_plugin.dfkv",
        )
        self.assertTrue(config.enabled)
        self.assertEqual(config.fallback_count, 2)

    def test_invalid_plugin_config_fails_closed(self):
        bad_configs = [
            {"remote_storage_plugin.dfkv.rail_affinity": "true"},
            {"remote_storage_plugin.dfkv.rail_affinity_fallbacks": True},
            {"remote_storage_plugin.dfkv.rail_affinity_fallbacks": -1},
            {"remote_storage_plugin.dfkv.rail_affinity_fallbacks": "1"},
        ]
        for config in bad_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    parse_plugin_rail_affinity(
                        config, "remote_storage_plugin.dfkv"
                    )

    def test_physical_rank_uses_local_worker_not_global_worker(self):
        metadata = types.SimpleNamespace(
            worker_id=13,
            world_size=16,
            local_worker_id=5,
            local_world_size=8,
        )
        self.assertEqual(physical_affinity_rank(metadata), 5)

    def test_missing_or_invalid_local_metadata_fails_closed(self):
        bad_metadata = [
            types.SimpleNamespace(worker_id=0, world_size=8),
            types.SimpleNamespace(local_worker_id=-1, local_world_size=8),
            types.SimpleNamespace(local_worker_id=8, local_world_size=8),
            types.SimpleNamespace(local_worker_id=0, local_world_size=0),
            types.SimpleNamespace(local_worker_id=True, local_world_size=8),
        ]
        for metadata in bad_metadata:
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    physical_affinity_rank(metadata)

    def test_two_node_tp16_maps_each_hosts_local_workers_to_eight_rails(self):
        rails = ",".join(f"ib7s400p{i}" for i in range(8))
        for global_rank in range(16):
            local_rank = global_rank % 8
            metadata = types.SimpleNamespace(
                worker_id=global_rank,
                world_size=16,
                local_worker_id=local_rank,
                local_world_size=8,
            )
            env = {"DFKV_RDMA_DEV": rails}
            selection = apply_rank_local_rail_affinity(
                {"rail_affinity": True, "rail_affinity_fallbacks": 1},
                physical_affinity_rank(metadata),
                env,
            )
            expected = (
                f"ib7s400p{local_rank}",
                f"ib7s400p{(local_rank + 1) % 8}",
            )
            self.assertEqual(selection.selected, expected)
            self.assertEqual(env["DFKV_RDMA_PRIMARY_DEV"], expected[0])


if __name__ == "__main__":
    unittest.main()
