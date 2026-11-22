# SPDX-License-Identifier: Apache-2.0
"""Per-operation access log for the dfkv vLLM and LMCache connectors.

This file is shared: integration/vllm/src/dfkv_vllm/access_log.py is the
canonical copy and deploy/sync_telemetry.sh copies it byte for byte into the
LMCache connector (test_telemetry_vendor_sync guards drift).

Mirrors the format used by dfkv's HiCache / dingofs vfs access log, one line
per operation::

    <op>(<args>) : <result> <duration_seconds>

e.g.::

    batch_get_auto_sg(20 keys) : hits=20/20, 1310720 bytes <0.007234>
    batch_put_sg(1 keys, 65536 bytes) : ok=1/1 <0.000821>
    batched_get(20 keys) : hits=20/20 <0.007234>
    get(model@1@0@cafebabe) : not_found <0.001012>
    batch_put(8 keys, 524288 bytes) : FAIL RuntimeError: dfkv_batch_put rc=-1 <0.001230>

Off by default. The SAME environment variables as every other dfkv access log,
so one setting covers all integrations:

    DFKV_ACCESS_LOG_ENABLED       "1" to turn on (default off)
    DFKV_ACCESS_LOG_PATH          file path; empty = stderr (default empty)
    DFKV_ACCESS_LOG_THRESHOLD_US  only log ops whose wall time exceeds this
                                  (default 0 = log every call)
    DFKV_ACCESS_LOG_MAX_BYTES     size-rotation threshold (default 128MiB;
                                  0 disables rotation = single unbounded file)
    DFKV_ACCESS_LOG_BACKUP_COUNT  rotated backups to keep (default 5)

The launch baseline is parsed once in :func:`configure` (called from the client
``__init__``); after that ``access_log()`` reads the enable flag per call, so a
control-file watcher (:mod:`._hot_config`) can toggle it at runtime WITHOUT a
restart. See docs/access_log.md → 运行时热开关.

Performance:
  - Disabled: ~100 ns/call. A frozen _NoopLog singleton is returned, so the
    context-manager protocol skips contextmanager machinery / timer / arg
    formatting entirely.
  - Enabled: file writes go through a logging.handlers.QueueHandler with a
    background QueueListener thread, so foreground emit cost is ~3 µs
    (enqueue only) — no synchronous write/flush in the hot path.
"""

from __future__ import annotations

import atexit
import logging
import logging.handlers
import os
import queue
import sys
import threading
import time
from typing import Any, Callable, Optional

# Module state. configure() (first call in the process) snapshots the launch
# config; apply_hot() (from the _hot_config watcher thread) may re-apply it
# later, so all mutating paths take _lock. access_log() reads _ENABLED /
# _THRESHOLD_US WITHOUT the lock — plain bool/int reads under the GIL; a stale
# read at worst mislabels one op, acceptable for a diagnostic log.
_ENABLED: bool = False
_THRESHOLD_US: int = 0
_logger: Optional[logging.Logger] = None
_listener: Optional[logging.handlers.QueueListener] = None
_configured: bool = False
_lock = threading.Lock()
_sink_want: Optional[tuple] = None
_launch_cfg: dict = {}
_tp_rank: int = 0
_model: str = ""


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _resolve(cfg: dict, key: str, env: str, default: Any) -> Any:
    """extra_config key wins; then env var; then default."""
    if cfg.get(key) is not None:
        return cfg[key]
    if env in os.environ:
        return os.environ[env]
    return default


def _int(cfg: dict, key: str, env: str, default: int) -> int:
    try:
        return int(_resolve(cfg, key, env, default))
    except (TypeError, ValueError):
        return default


def _stop_listener(listener) -> None:
    if listener is not None and getattr(listener, "_thread", None) is not None:
        listener.stop()


def _build_sink(path: str, max_bytes: int, backup_count: int,
                tp_rank: int, model: str) -> None:
    """(Re)build the logger + async QueueListener. Tears down any previous
    listener/handlers first so a hot path change never double-writes. Caller
    holds _lock. Size-rotation bounds disk per rank to
    max_bytes * (backup_count + 1); max_bytes=0 disables rotation."""
    global _logger, _listener
    if path:
        if "{rank}" in path or "{model}" in path:
            path = path.format(rank=tp_rank, model=model)
        else:
            path = f"{path}.r{tp_rank}"

    if path:
        try:
            if max_bytes > 0:
                sink: logging.Handler = logging.handlers.RotatingFileHandler(
                    path, mode="a", maxBytes=max_bytes, backupCount=backup_count)
            else:
                sink = logging.FileHandler(path, mode="a")
        except OSError as exc:
            sys.stderr.write(
                f"[dfkv.access] cannot open {path!r}: {exc}; "
                f"falling back to stderr\n")
            sink = logging.StreamHandler(sys.stderr)
    else:
        sink = logging.StreamHandler(sys.stderr)
    sink.setFormatter(
        logging.Formatter("%(asctime)s.%(msecs)03d %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )

    _stop_listener(_listener)
    _listener = None
    log = logging.getLogger("dfkv.access")
    log.setLevel(logging.INFO)
    log.propagate = False  # keep out of the root logger and the engine's logging stack
    for h in list(log.handlers):
        log.removeHandler(h)

    # Unbounded queue: enqueueing never blocks; the background listener does the
    # write/flush, keeping foreground emit at a few microseconds.
    q: "queue.Queue[logging.LogRecord]" = queue.Queue(-1)
    _listener = logging.handlers.QueueListener(q, sink, respect_handler_level=False)
    _listener.start()
    atexit.register(_stop_listener, _listener)
    log.addHandler(logging.handlers.QueueHandler(q))
    _logger = log


def _apply(cfg: Optional[dict], tp_rank: int, model: str) -> None:
    """Resolve the access-log knobs from cfg and (re)apply. Caller holds _lock.
    Enables/disables live; builds the sink lazily on first enable and rebuilds
    only when path/rotation params change."""
    global _ENABLED, _THRESHOLD_US, _sink_want
    cfg = cfg or {}
    _THRESHOLD_US = max(0, _int(cfg, "access_log_threshold_us",
                                "DFKV_ACCESS_LOG_THRESHOLD_US", 0))
    if not _truthy(_resolve(cfg, "access_log", "DFKV_ACCESS_LOG_ENABLED", False)):
        _ENABLED = False
        return
    path = str(_resolve(cfg, "access_log_path", "DFKV_ACCESS_LOG_PATH", "")).strip()
    max_bytes = _int(cfg, "access_log_max_bytes", "DFKV_ACCESS_LOG_MAX_BYTES",
                     128 * 1024 * 1024)
    backup_count = _int(cfg, "access_log_backup_count",
                        "DFKV_ACCESS_LOG_BACKUP_COUNT", 5)
    if max_bytes < 0:
        max_bytes = 0
    if backup_count < 0:
        backup_count = 0
    want = (path, max_bytes, backup_count)
    if _logger is None or _sink_want != want:
        _build_sink(path, max_bytes, backup_count, tp_rank, model)
        _sink_want = want
    _ENABLED = True


def configure(cfg: Optional[dict] = None, tp_rank: int = 0, model: str = "") -> None:
    """Parse the launch baseline (first call in the process wins) and apply it.
    Snapshots the config so :func:`apply_hot` can layer live control-file
    overrides on top and revert cleanly. cfg may be None (env-only)."""
    global _configured, _launch_cfg, _tp_rank, _model
    with _lock:
        if _configured:
            return
        _configured = True
        _launch_cfg = dict(cfg or {})
        _tp_rank = tp_rank
        _model = model
        _apply(_launch_cfg, tp_rank, model)


def apply_hot(overrides: Optional[dict]) -> None:
    """Re-apply access-log knobs from control-file overrides, layered over the
    launch baseline (absent key -> launch default, so deleting the control file
    reverts). Thread-safe; called by the _hot_config watcher. No-op before
    :func:`configure`."""
    with _lock:
        if not _configured:
            return
        merged = dict(_launch_cfg)
        if overrides:
            merged.update(overrides)
        _apply(merged, _tp_rank, _model)


def is_enabled() -> bool:
    return _ENABLED


# ---------------------------------------------------------------------------
# Real (enabled) context manager — keeps a timer and emits on exit.
# ---------------------------------------------------------------------------

class _RealLog:
    __slots__ = ("_op", "_args_fn", "_start", "result")

    def __init__(self, op: str, args_fn: Callable[[], str]) -> None:
        self._op = op
        self._args_fn = args_fn
        self._start = 0.0
        self.result: str = "OK"

    def __enter__(self) -> "_RealLog":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        duration = time.perf_counter() - self._start
        if exc_type is not None:
            self.result = f"FAIL {exc_type.__name__}: {exc_val}"
        if duration * 1_000_000 >= _THRESHOLD_US and _logger is not None:
            _logger.info("%s(%s) : %s <%.6f>",
                         self._op, self._args_fn(), self.result, duration)
        return False  # don't swallow exceptions


# ---------------------------------------------------------------------------
# Noop singleton — what disabled access_log returns. ~100 ns / call.
# ---------------------------------------------------------------------------

class _NoopLog:
    """Frozen singleton used when the access log is disabled.

    Implements just enough of the context-manager protocol to be a drop-in
    for _RealLog. Has a writable `result` attribute that nobody reads.
    """
    __slots__ = ()

    result: Any = "OK"  # class attribute; .result = ... below is a no-op

    def __enter__(self) -> "_NoopLog":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def __setattr__(self, name: str, value: Any) -> None:
        # Allow `r.result = "..."` from callers without raising; just drop it.
        pass


_NOOP = _NoopLog()


# ---------------------------------------------------------------------------
# Public entry: cheap singleton when disabled, real timer when enabled. Reads
# the enable flag PER CALL so the _hot_config watcher can toggle it at runtime.
# ---------------------------------------------------------------------------

def access_log(op: str = "", args_fn: Callable[[], str] = lambda: ""):
    if _ENABLED:
        return _RealLog(op, args_fn)
    return _NOOP
