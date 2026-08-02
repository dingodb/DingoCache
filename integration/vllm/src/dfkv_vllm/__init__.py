"""dfkv connector for vLLM (direct KVConnectorBase_V1, GPUDirect RDMA).

``DfkvStoreConnector`` is loaded by vLLM via ``kv_connector_module_path``:

    --kv-transfer-config '{"kv_connector":"DfkvStoreConnector",
      "kv_connector_module_path":"dfkv_vllm.connector",
      "kv_role":"kv_both",
      "kv_connector_extra_config":{"members":"c1=<ip>:<rdma-port>",
        "lib":"/path/to/libdfkv.so"}}'

The exact model identity comes from vLLM. ``key_namespace`` is an optional
explicit schema override; omit it for the safe ``vllm/raw-v1`` default.
"""
from .dfkv_client import DfkvDeviceClient

__all__ = ["DfkvDeviceClient"]

# vLLM resolves DfkvStoreConnector from .connector through
# kv_connector_module_path="dfkv_vllm.connector".
