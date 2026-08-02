"""Fail-fast guard: dfkv store keys require cross-process block hashes.

Kept free of vllm/torch imports (duck-typed on cache_config) so it can be
unit-tested anywhere; both the scheduler- and worker-side components call it
at construction time.
"""



def ensure_deterministic_block_hashing(cache_config) -> None:
    """Require content-defined vLLM block hashes.

    The native client hashes the complete binary namespace/object key with
    SHA-256, but it cannot repair a process-local hash already embedded in the
    object key. vLLM's ``builtin`` algorithm uses Python ``hash()``; even a
    pinned PYTHONHASHSEED only works when every producer, consumer, and future
    restart shares an external process setting. Production identity must be
    intrinsic instead: require vLLM's SHA-256 family before any store traffic.
    """
    algo = str(getattr(cache_config, "prefix_caching_hash_algo", "builtin"))
    if "sha256" in algo.lower():
        return
    raise RuntimeError(
        "DfkvStoreConnector requires content-defined vLLM block hashes; "
        f"prefix_caching_hash_algo={algo!r} is process-local. Start every "
        "engine sharing this store with --prefix-caching-hash-algo sha256. "
        "PYTHONHASHSEED is intentionally not accepted as an identity contract."
    )
