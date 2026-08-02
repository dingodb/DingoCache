# SPDX-License-Identifier: Apache-2.0
"""Per-process LRU set short-circuiting Exists() lookups.

Rationale: LMCache's batched_async_contains is on the prefetch hot path. The
overwhelming common case is "I just put this key, now I'm asking does it
exist?" — answering yes from memory avoids a remote round-trip.

Tolerates false positives (server evicts a key we know about; the subsequent
Get returns NotFound and LMCache handles it). Does NOT tolerate false
negatives — on miss we must always check remotely.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Iterable, Sequence

__all__ = ["ExistsLRU"]


class ExistsLRU:
    def __init__(self, capacity: int = 1_000_000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._entries: "OrderedDict[bytes, None]" = OrderedDict()
        self._lock = threading.Lock()

    def _touch_unlocked(self, key: bytes) -> None:
        if key in self._entries:
            self._entries.move_to_end(key)
            return
        self._entries[key] = None
        if len(self._entries) > self._capacity:
            self._entries.popitem(last=False)

    def add(self, key: bytes) -> None:
        with self._lock:
            self._touch_unlocked(key)

    def add_many(self, keys: Iterable[bytes]) -> None:
        with self._lock:
            for key in keys:
                self._touch_unlocked(key)

    def discard(self, key: bytes) -> None:
        """Forget a key (e.g. after remove) so a later has() doesn't report a
        stale hit. No-op if absent."""
        with self._lock:
            self._entries.pop(key, None)

    def has(self, key: bytes) -> bool:
        with self._lock:
            if key not in self._entries:
                return False
            self._entries.move_to_end(key)
            return True

    def prefix_len(self, keys: Sequence[bytes]) -> int:
        """Return the longest prefix already known to exist.

        This is the batched form of ``has()`` for LMCache's prefetch path.
        It keeps the lock for one contiguous scan instead of taking it once
        per key, which matters when prompts expand to thousands of KV chunks.
        """
        with self._lock:
            prefix = 0
            for key in keys:
                if key not in self._entries:
                    break
                self._entries.move_to_end(key)
                prefix += 1
            return prefix

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
