# SPDX-License-Identifier: Apache-2.0
"""Map LMCache cache coordinates onto dfkv's canonical pool-key schema.

The model identity is carried once by ``dfkv_open``'s binary namespace. Object
keys therefore contain only pool coordinates and the full content hash.
"""

from __future__ import annotations

from lmcache.utils import CacheEngineKey
from dfkv_common import pool_key

__all__ = ["cache_engine_key_to_dfkv_key"]


def cache_engine_key_to_dfkv_key(key: CacheEngineKey) -> bytes:
    """Encode a CacheEngineKey without duplicating model identity.

    ``world_size`` and ``worker_id`` are preserved exactly. LMCache owns
    replicated-layout detection and writer selection; rewriting rank identity
    in this connector could alias non-equivalent pipeline stages. Layerwise
    objects use an explicit component coordinate.
    """
    layer_id = getattr(key, "layer_id", None)
    component = "all" if layer_id is None else f"layer{int(layer_id)}"
    return pool_key(
        f"{int(key.chunk_hash):x}",
        pool="kv",
        tp_size=max(1, int(key.world_size)),
        tp_rank=int(key.worker_id),
        component=component,
    )
