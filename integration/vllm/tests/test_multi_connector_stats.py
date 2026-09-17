"""External connectors must survive MultiConnector statistics deserialization."""
import subprocess
import sys

import pytest


@pytest.mark.parametrize("pre_registered", [False, True])
def test_multiconn_reconstructs_serialized_dfkv_metrics(pre_registered):
    # Fresh interpreters exercise both module-path discovery and an operator's
    # explicit factory registration, without polluting another test's registry.
    setup = (
        "KVConnectorFactory.register_connector("
        "'DfkvStoreConnector', 'dfkv_vllm.connector', 'DfkvStoreConnector')\n"
        if pre_registered else ""
    )
    code = """
import json
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
""" + setup + """
from dfkv_vllm.connector import DfkvStoreConnector
from dfkv_vllm.metrics import DfkvStoreConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector, MultiKVConnectorStats,
)
first = DfkvStoreConnectorStats()
first.record_operation('load_get', 0.01, 7, num_bytes=512,
                       num_failed_keys=2, status='partial_failure')
second = DfkvStoreConnectorStats()
second.record_operation('load_get', 0.02, 3, num_bytes=128)
combined = MultiKVConnectorStats(data={'DfkvStoreConnector': first})
combined.aggregate(MultiKVConnectorStats(data={'DfkvStoreConnector': second}))
payload = json.loads(json.dumps(combined.to_dict()))
restored = MultiConnector.build_kv_connector_stats(payload).reduce()['DfkvStoreConnector']
assert restored['load_get_count'] == 2, restored
assert restored['load_get_total_keys'] == 10, restored
assert restored['load_get_total_bytes'] == 640, restored
assert restored['load_get_failed_keys'] == 2, restored
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
