# SPDX-License-Identifier: Apache-2.0
"""Map LMCache cache coordinates onto dfkv's canonical pool-key schema.

The model identity is carried once by ``dfkv_open``'s binary namespace. Object
keys therefore contain only pool coordinates and the full content hash.
"""

from __future__ import annotations

from lmcache.utils import CacheEngineKey
from dfkv_common import pool_key

__all__ = ["cache_engine_key_to_dfkv_key"]


def cache_engine_key_to_dfkv_key(
    key: CacheEngineKey,
    canonicalize_worker: bool = False,
) -> bytes:
    """Encode a CacheEngineKey without duplicating model identity.

    Replicated MLA uses ``tp=-1``; sharded layouts retain the worker rank.
    Layerwise LMCache objects use an explicit component coordinate.
    """
    layer_id = getattr(key, "layer_id", None)
    component = "all" if layer_id is None else f"layer{int(layer_id)}"
    return pool_key(
        f"{int(key.chunk_hash):x}",
        pool="kv",
        tp_size=max(1, int(key.world_size)),
        tp_rank=-1 if canonicalize_worker else int(key.worker_id),
        component=component,
    )
