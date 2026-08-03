"""ctypes declarations for the dfkv C ABI (``libdfkv.so``).

Self-contained: declares only the symbols the vLLM connector uses, so this
package does not depend on the LMCache integration package. Symbol signatures
mirror ``src/client/dfkv_c_api.h``.

The library is found via (highest precedence first):
  1. explicit ``lib_path`` argument
  2. env ``DFKV_LIB``
  3. ``$DFKV_BUILD/libdfkv.so``
"""
import ctypes
import os
from typing import Optional
from dfkv_common import DfkvClientOptionsV2

c_void_p = ctypes.c_void_p
c_char_p = ctypes.c_char_p
c_uint64 = ctypes.c_uint64
c_uint32 = ctypes.c_uint32
c_int = ctypes.c_int
POINTER = ctypes.POINTER


def resolve_lib_path(lib_path: Optional[str] = None) -> str:
    if lib_path:
        return lib_path
    env = os.environ.get("DFKV_LIB")
    if env:
        return env
    build = os.environ.get("DFKV_BUILD")
    if build:
        return os.path.join(build, "libdfkv.so")
    # Installed packages resolve through ldconfig / LD_LIBRARY_PATH without
    # requiring a workstation-specific build directory.
    return "libdfkv.so"


def load_lib(lib_path: Optional[str] = None) -> ctypes.CDLL:
    """Load libdfkv.so and declare the C-ABI signatures used by the connector."""
    lib = ctypes.CDLL(resolve_lib_path(lib_path))

    # handle = dfkv_open_v2(const dfkv_client_options_v2*)
    lib.dfkv_open_v2.restype = c_void_p
    lib.dfkv_open_v2.argtypes = [POINTER(DfkvClientOptionsV2)]

    # GPUDirect MR registration (ibv_reg_mr on a host OR device pointer).
    lib.dfkv_register_memory.restype = c_int
    lib.dfkv_register_memory.argtypes = [c_void_p, c_void_p, c_uint64]

    # Live SG width (negotiated max_sge - 1). Older libs lack the symbol;
    # callers must tolerate AttributeError and fall back to 29.
    try:
        lib.dfkv_max_sg_segs.restype = c_uint32
        lib.dfkv_max_sg_segs.argtypes = [c_void_p]
    except AttributeError:
        pass

    # Batch primitives (raw void** pointers -> may be GPU device pointers).
    lib.dfkv_batch_put.restype = c_int
    lib.dfkv_batch_put.argtypes = [
        c_void_p, POINTER(c_void_p), POINTER(c_uint64),
        POINTER(c_void_p), POINTER(c_uint64), c_int, POINTER(c_int),
    ]
    # Variable-size read: each buffer filled to its true stored length.
    lib.dfkv_batch_get_auto.restype = c_int
    lib.dfkv_batch_get_auto.argtypes = [
        c_void_p, POINTER(c_void_p), POINTER(c_uint64),
        POINTER(c_void_p), POINTER(c_uint64),
        c_int, POINTER(c_int), POINTER(c_uint64),
    ]
    lib.dfkv_batch_exist.restype = c_int
    lib.dfkv_batch_exist.argtypes = [
        c_void_p, POINTER(c_void_p), POINTER(c_uint64),
        c_int, POINTER(c_int)]

    # Scatter-gather: one key gathers num_bufs[i] non-contiguous source buffers
    # (ptrs[i][..], sizes[i][..]) on put / scatters into num_dsts[i] dst buffers
    # (dsts[i][..], caps[i][..]) on get. Coalesces a chunk's per-layer segments
    # into ONE key + one RDMA multi-SGE op (<=29 segs/key on max_sge=30 HCAs).
    lib.dfkv_batch_put_sg.restype = c_int
    lib.dfkv_batch_put_sg.argtypes = [
        c_void_p, POINTER(c_void_p), POINTER(c_uint64),
        POINTER(POINTER(c_void_p)), POINTER(POINTER(c_uint64)),
        POINTER(c_int), c_int, POINTER(c_int),
    ]
    lib.dfkv_batch_get_auto_sg.restype = c_int
    lib.dfkv_batch_get_auto_sg.argtypes = [
        c_void_p, POINTER(c_void_p), POINTER(c_uint64),
        POINTER(POINTER(c_void_p)), POINTER(POINTER(c_uint64)),
        POINTER(c_int), c_int, POINTER(c_int), POINTER(c_uint64),
    ]


    # Remove support remains capability-detected for deployments that omit the
    # optional eviction RPC; partial-SG cleanup degrades to a no-op then.
    if hasattr(lib, "dfkv_remove"):
        lib.dfkv_remove.restype = c_int
        lib.dfkv_remove.argtypes = [c_void_p, c_void_p, c_uint64]
    if hasattr(lib, "dfkv_batch_remove"):
        lib.dfkv_batch_remove.restype = c_int
        lib.dfkv_batch_remove.argtypes = [
            c_void_p, POINTER(c_void_p), POINTER(c_uint64),
            c_int, POINTER(c_int)]

    lib.dfkv_transport_mode.restype = c_char_p
    lib.dfkv_transport_mode.argtypes = [c_void_p]

    # u64 = dfkv_stats_snapshot(c, buf, cap): Prometheus-text dump of the C
    # client's observed counters (ring size, MDS reachability, per-peer health).
    # Call with buf=NULL/cap=0 to size, then fetch. Off the request path.
    lib.dfkv_stats_snapshot.restype = c_uint64
    lib.dfkv_stats_snapshot.argtypes = [c_void_p, c_char_p, c_uint64]

    # const char* dfkv_version(void) -> libdfkv.so version (process-global).
    lib.dfkv_version.restype = c_char_p
    lib.dfkv_version.argtypes = []

    lib.dfkv_close.restype = None
    lib.dfkv_close.argtypes = [c_void_p]
    return lib


def native_version(lib: ctypes.CDLL) -> str:
    """libdfkv.so version via the C dfkv_version(); "" if missing/older lib."""
    try:
        v = lib.dfkv_version()
        return v.decode("utf-8", "replace") if v else ""
    except Exception:
        return ""
