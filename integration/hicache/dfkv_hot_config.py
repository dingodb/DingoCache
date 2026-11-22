# SPDX-License-Identifier: Apache-2.0
"""Runtime hot-reload of the dfkv client's OBSERVABILITY knobs — no restart.

A background daemon thread polls a small JSON control file. On change it
re-applies each registered subsystem's hot knobs, layered over that subsystem's
launch config, so editing the file turns knobs on live and deleting it reverts
to launch behavior. Multi-process safe by construction: every SGLang TP-rank
process runs its own watcher over the SAME file, so one edit flips all ranks.

Scope (Phase 1): access_log family only — knobs with ZERO correctness impact.
Structural knobs (rdma / lib / register / node_dedup) and C-client-cached knobs
(peer cooldown, get_miss_retries, node_dedup_log, ...) stay restart-only: they
are frozen at construction / first-op inside the native client and cannot be
flipped from Python. Those are a separate (Phase 2) native change.

Opt-in: the watcher only runs when a control-file path is EXPLICITLY set. With
no path configured, nothing new happens (zero threads, zero cost) — hot-reload
is off, exactly as before. This keeps it out of the way of runs that never asked
for it.

Control file:
    path = extra_config["hot_config_path"] | $DFKV_HOT_CONFIG   (unset -> disabled)
    poll = extra_config["hot_config_poll_s"] | $DFKV_HOT_CONFIG_POLL_S | 5.0 (s)
    poll <= 0 disables the watcher entirely.

File format (all keys optional; an absent key -> that subsystem's launch value)::

    {
      "access_log": true,
      "access_log_threshold_us": 500,
      "access_log_path": "/var/log/dfkv/access.log"
    }

A missing control file means "no overrides" (revert to launch); a malformed file
is ignored (last good config stays) and retried next tick. The file is polled by
stat() — one syscall per interval, off the request path.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from typing import Callable, List, Optional, Tuple

_lock = threading.Lock()
_appliers: List[Tuple[str, Callable[[Optional[dict]], None]]] = []
_watcher: "Optional[_Watcher]" = None


def register(name: str, apply_fn: Callable[[Optional[dict]], None]) -> None:
    """Register a subsystem hot-apply callback ``apply_fn(overrides: dict)``.

    Idempotent by name so repeated DfkvHiCache.__init__ calls don't stack
    duplicate appliers in one process.
    """
    with _lock:
        if any(n == name for n, _ in _appliers):
            return
        _appliers.append((name, apply_fn))


class _Watcher(threading.Thread):
    def __init__(self, path: str, poll_s: float, tp_rank: int) -> None:
        super().__init__(name="dfkv-hot-config", daemon=True)
        self._path = path
        self._interval = float(poll_s)
        self._tp_rank = tp_rank
        # Not ``_stop``: that name is threading.Thread's own (a method it
        # calls from join() up to Python 3.12).
        self._stop_event = threading.Event()
        # (mtime, size) of the file at the last successful apply; None = absent.
        self._seen: object = "<never>"  # sentinel: force an apply on first tick

    def stop(self) -> None:
        self._stop_event.set()

    def _read(self) -> Optional[dict]:
        """Return the override dict ({} if the file is absent), or None on a
        transient read/parse error (so the caller keeps the last good config)."""
        try:
            with open(self._path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as exc:
            # Catch-all on purpose: a garbage/binary file (UnicodeDecodeError,
            # JSONDecodeError), a pathologically nested one (RecursionError), a
            # permission/IO error (OSError) — none of it may crash or kill the
            # watcher thread. Keep the last good config and retry next tick.
            sys.stderr.write(
                f"[dfkv.hot] r{self._tp_rank} bad control file {self._path!r}: "
                f"{type(exc).__name__}: {exc}; keeping current config\n")
            return None
        return data if isinstance(data, dict) else {}

    def _apply_all(self, overrides: dict) -> None:
        with _lock:
            appliers = list(_appliers)
        for name, fn in appliers:
            try:
                fn(overrides)
            except Exception as exc:  # one bad applier must not kill the watcher
                sys.stderr.write(
                    f"[dfkv.hot] r{self._tp_rank} applier {name!r} failed: {exc}\n")

    def run(self) -> None:
        first = True
        while first or not self._stop_event.wait(self._interval):
            first = False
            try:
                st = os.stat(self._path)
                sig: object = (st.st_mtime, st.st_size)
            except FileNotFoundError:
                sig = None
            except OSError:
                continue  # transient; retry next tick
            if sig == self._seen:
                continue
            overrides = self._read()
            if overrides is None:
                continue  # parse error: retry next tick, don't advance _seen
            self._apply_all(overrides)
            self._seen = sig
            sys.stderr.write(
                f"[dfkv.hot] r{self._tp_rank} applied {len(overrides)} override(s) "
                f"from {self._path}\n")


def _resolve(cfg: dict, key: str, env: str, default):
    if cfg.get(key) is not None:
        return cfg[key]
    if env in os.environ:
        return os.environ[env]
    return default


def start(cfg: Optional[dict] = None, tp_rank: int = 0) -> "Optional[_Watcher]":
    """Start the control-file watcher (idempotent; first call in the process
    wins). Returns the watcher, or None if disabled / nothing registered.

    Opt-in: returns None unless a control-file path is explicitly set (via
    extra_config["hot_config_path"] or $DFKV_HOT_CONFIG). Call AFTER registering
    appliers and AFTER each subsystem's configure(), so the first tick layers
    file overrides on a populated launch config.
    """
    global _watcher
    with _lock:
        if _watcher is not None:
            return _watcher
        if not _appliers:
            return None
        cfg = cfg or {}
        path = str(_resolve(cfg, "hot_config_path", "DFKV_HOT_CONFIG", "")).strip()
        try:
            poll_s = float(_resolve(cfg, "hot_config_poll_s",
                                    "DFKV_HOT_CONFIG_POLL_S", 5.0))
        except (TypeError, ValueError):
            poll_s = 5.0
        if not path or poll_s <= 0:
            return None
        _watcher = _Watcher(path, poll_s, tp_rank)
        _watcher.start()
        return _watcher


def stop() -> None:
    global _watcher
    with _lock:
        w = _watcher
        _watcher = None
    if w is not None:
        w.stop()
