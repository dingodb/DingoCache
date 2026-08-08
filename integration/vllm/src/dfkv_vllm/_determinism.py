"""Fail-fast guard: dfkv store keys require cross-process block hashes.

Kept free of vllm/torch imports (duck-typed on cache_config) so it can be
unit-tested anywhere; both the scheduler- and worker-side components call it
at construction time.
"""
import os




def ensure_deterministic_block_hashing(cache_config) -> None:
    """Require block hashes that remain stable across engine restarts.

    vLLM chains the first token block from a process-global ``NONE_HASH``.
    Current releases initialize that root with ``os.urandom(32)`` unless
    ``PYTHONHASHSEED`` is set, including for the SHA-256 algorithms.  The
    connector cannot repair that random root after it has entered the block
    hash, so both the content hash algorithm and root seed are store identity.
    """
    algo = str(getattr(cache_config, "prefix_caching_hash_algo", "builtin"))
    if "sha256" not in algo.lower():
        raise RuntimeError(
            "DfkvStoreConnector requires content-defined vLLM block hashes; "
            f"prefix_caching_hash_algo={algo!r} is process-local. Start every "
            "engine sharing this store with --prefix-caching-hash-algo sha256."
        )
    if os.environ.get("PYTHONHASHSEED") is None:
        raise RuntimeError(
            "DfkvStoreConnector requires PYTHONHASHSEED to be set before "
            "vLLM starts. vLLM otherwise seeds the first block-hash parent "
            "from os.urandom(), so cache keys change on every engine restart. "
            "Use the same fixed value (for example PYTHONHASHSEED=0) for "
            "every producer and consumer sharing the store."
        )
