# SPDX-License-Identifier: Apache-2.0
"""Per-process LRU set short-circuiting Exists() lookups.

Rationale: LMCache's batched_async_contains is on the prefetch hot path. The
overwhelming common case is "I just put this key, now I'm asking does it
exist?" — answering yes from memory avoids a remote round-trip.

A positive answer from here can be stale: the server evicts on its own, and
nothing tells the client. A stale positive is worse than a miss, because
LMCache then neither finds the chunk (the GET misses) nor stores it again (its
put path asks ``contains`` first and skips what "exists"). So positives are
bounded two ways:

* an entry expires ``ttl_s`` seconds after it was last confirmed (added), so
  a key is re-checked remotely at least that often;
* the connector discards an entry as soon as a GET for it misses.

Does NOT tolerate false negatives — on miss we must always check remotely.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Callable, Iterable, Optional, Sequence

__all__ = ["ExistsLRU"]


class ExistsLRU:
    def __init__(
        self,
        capacity: int = 1_000_000,
        ttl_s: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """``ttl_s``: seconds a positive stays valid after it was confirmed;
        ``None`` or ``0`` keeps entries until they are evicted or discarded."""
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if ttl_s is not None and ttl_s < 0:
            raise ValueError("ttl_s must not be negative")
        self._capacity = capacity
        self._ttl_s = float(ttl_s) if ttl_s else 0.0
        self._clock = clock
        # key -> time it expires (0.0 = never)
        self._entries: "OrderedDict[bytes, float]" = OrderedDict()
        self._lock = threading.Lock()

    @property
    def ttl_s(self) -> float:
        return self._ttl_s

    def _confirm_unlocked(self, key: bytes, expires: float) -> None:
        """The key was just seen on the server (stored, found, read)."""
        self._entries[key] = expires
        self._entries.move_to_end(key)
        if len(self._entries) > self._capacity:
            self._entries.popitem(last=False)

    def _expiry(self) -> float:
        return self._clock() + self._ttl_s if self._ttl_s else 0.0

    def _live_unlocked(self, key: bytes, now: float) -> bool:
        """Whether the key is known and not expired. Drops it if expired;
        marks it recently used otherwise (its expiry is not extended: only a
        confirmation from the server does that)."""
        expires = self._entries.get(key)
        if expires is None:
            return False
        if expires and expires <= now:
            del self._entries[key]
            return False
        self._entries.move_to_end(key)
        return True

    def add(self, key: bytes) -> None:
        with self._lock:
            self._confirm_unlocked(key, self._expiry())

    def add_many(self, keys: Iterable[bytes]) -> None:
        with self._lock:
            expires = self._expiry()
            for key in keys:
                self._confirm_unlocked(key, expires)

    def discard(self, key: bytes) -> None:
        """Forget a key (after a remove, or a GET that missed) so a later
        has() doesn't report a stale hit. No-op if absent."""
        with self._lock:
            self._entries.pop(key, None)

    def discard_many(self, keys: Iterable[bytes]) -> None:
        with self._lock:
            for key in keys:
                self._entries.pop(key, None)

    def has(self, key: bytes) -> bool:
        with self._lock:
            return self._live_unlocked(key, self._clock() if self._ttl_s else 0.0)

    def prefix_len(self, keys: Sequence[bytes]) -> int:
        """Return the longest prefix already known to exist.

        This is the batched form of ``has()`` for LMCache's prefetch path.
        It keeps the lock for one contiguous scan instead of taking it once
        per key, which matters when prompts expand to thousands of KV chunks.
        """
        with self._lock:
            now = self._clock() if self._ttl_s else 0.0
            prefix = 0
            for key in keys:
                if not self._live_unlocked(key, now):
                    break
                prefix += 1
            return prefix

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
