"""Drift guard for the ctypes bindings of libdfkv.

The HiCache plugin, the vLLM connector, the LMCache connector and
dfkv_common each declare argtypes/restype for the native C ABI. Nothing
kept them in step with each other or with src/client/dfkv_c_api.h: a
changed prototype could be fixed in one binding and left wrong in another,
and ctypes would pass the wrong types without an error. This test loads
every binding against a recording stand-in for the library and checks
that each symbol bound in more than one place is bound with the same
argtypes and restype, and that every bound symbol is declared in the
header with the same number of arguments. Argument types are compared
between binding sites only, not against the header (the header's C types
are not parsed).
"""
import ctypes
import importlib
import os
import re
import sys
import types
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
HEADER = os.path.join(ROOT, "src", "client", "dfkv_c_api.h")


class _Fn:
    def __init__(self):
        self.argtypes = None
        self.restype = ctypes.c_int

    def __call__(self, *_args):
        return 1        # e.g. the block-size preflight wants a non-zero bound


class _RecordingLib:
    """Every attribute is a function object whose argtypes/restype are kept."""

    def __init__(self, *_args, **_kwargs):
        object.__setattr__(self, "fns", {})

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self.fns.setdefault(name, _Fn())


def _signature(fn):
    def name(t):
        return getattr(t, "__name__", repr(t))
    return (tuple(name(t) for t in fn.argtypes or ()), name(fn.restype))


def _bindings():
    """{module: {symbol: (argtypes, restype)}} for every binding site."""
    sys.path[:0] = [
        HERE,
        os.path.join(ROOT, "integration", "hicache"),
        os.path.join(ROOT, "integration", "common", "src"),
        os.path.join(ROOT, "integration", "vllm", "src"),
        os.path.join(ROOT, "integration", "lmcache", "src"),
    ]
    sites = {
        "hicache": ("dfkv_hicache", "_load_lib"),
        "vllm": ("dfkv_vllm._cabi", "load_lib"),
        "lmcache": ("dfkv_connector.native_client", "load_lib"),
    }
    # The connector packages' __init__ imports their framework (vLLM,
    # LMCache); the binding modules themselves do not. Import them without it.
    for package, path in (
            ("dfkv_vllm", os.path.join(ROOT, "integration", "vllm", "src",
                                       "dfkv_vllm")),
            ("dfkv_connector", os.path.join(ROOT, "integration", "lmcache",
                                            "src", "dfkv_connector"))):
        if package not in sys.modules:
            stub = types.ModuleType(package)
            stub.__path__ = [path]
            sys.modules[package] = stub
    def recorded(lib):
        return {name: _signature(fn) for name, fn in lib.fns.items()
                if fn.argtypes is not None}

    out = {}
    with mock.patch.object(ctypes, "CDLL", _RecordingLib):
        for site, (module, loader) in sites.items():
            mod = importlib.import_module(module)
            out[site] = recorded(getattr(mod, loader)("libdfkv-recording.so"))
    # dfkv_common binds lazily, on a library a connector loaded.
    from dfkv_common import block_size, cabi
    lib = _RecordingLib()
    cabi.bind_retain(lib)
    block_size.get_max_block_bytes(lib, 1)
    out["common"] = recorded(lib)
    return out


def _strip_attributes(text):
    """Remove every ``__attribute__((...))``, nested parentheses included."""
    while True:
        m = re.search(r"__attribute__\s*\(", text)
        if not m:
            return text
        depth, i = 0, m.end() - 1
        for i in range(m.end() - 1, len(text)):
            depth += {"(": 1, ")": -1}.get(text[i], 0)
            if depth == 0:
                break
        text = text[:m.start()] + text[i + 1:]


def _top_level_args(params):
    """Argument count, not counting commas inside nested parentheses."""
    params = params.strip()
    if params in ("", "void"):
        return 0
    depth, count = 0, 1
    for ch in params:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            count += 1
    return count


def header_declarations(text):
    """{name: [argument count of each declaration]} of a C header.

    Every declaration counts, also one under ``#if 0`` or in another
    ``#ifdef`` branch: a stale duplicate must not silently decide. A
    declaration may end in attributes or an upper-case macro.
    """
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
    text = _strip_attributes(text)
    found = {}
    for m in re.finditer(r"\b(dfkv_\w+)\s*\(", text):
        depth, i = 0, m.end() - 1
        for i in range(m.end() - 1, len(text)):
            depth += {"(": 1, ")": -1}.get(text[i], 0)
            if depth == 0:
                break
        rest = text[i + 1:]
        if not re.match(r"\s*(?:[A-Z][A-Z0-9_]*(?:\([^()]*\))?\s*)*;", rest):
            continue        # not a declaration (a call, or a body follows)
        found.setdefault(m.group(1), []).append(
            _top_level_args(text[m.end():i]))
    return found


def _header_arity():
    with open(HEADER) as header:
        found = header_declarations(header.read())
    conflicting = {n: c for n, c in found.items() if len(set(c)) > 1}
    if conflicting:
        raise AssertionError(
            f"dfkv_c_api.h declares these with different argument counts: "
            f"{conflicting}")
    return {name: counts[0] for name, counts in found.items()}


class HeaderParsing(unittest.TestCase):
    def test_duplicate_declarations_must_agree(self):
        text = ("int dfkv_put(void* c, const void* k, uint64_t n);\n"
                "#if 0\nint dfkv_put(void* c, const void* k);\n#endif\n")
        self.assertEqual(header_declarations(text), {"dfkv_put": [3, 2]})

    def test_attributes_macros_and_function_pointers(self):
        text = (
            "int dfkv_a(void* c, int (*cb)(void*, int), int x) "
            "__attribute__((nonnull(1, 2)));\n"
            "uint64_t dfkv_b(void) DFKV_DEPRECATED;\n"
            "int dfkv_c(int x) DFKV_SINCE(2, 40);\n"
            "static inline int dfkv_d(int x) { return x; }\n")
        self.assertEqual(header_declarations(text),
                         {"dfkv_a": [3], "dfkv_b": [0], "dfkv_c": [1]})


class CabiBindingsAgree(unittest.TestCase):
    def test_bindings_agree_with_each_other_and_the_header(self):
        sites = _bindings()
        self.assertEqual(set(sites), {"hicache", "vllm", "lmcache", "common"})
        self.assertIn("dfkv_max_block_bytes", sites["common"])
        arity = _header_arity()
        seen = {}
        problems = []
        for site, symbols in sites.items():
            for name, sig in symbols.items():
                if name not in arity:
                    problems.append(f"{site}: {name} is not in dfkv_c_api.h")
                elif len(sig[0]) != arity[name]:
                    problems.append(
                        f"{site}: {name} binds {len(sig[0])} args, the header "
                        f"declares {arity[name]}")
                first = seen.setdefault(name, (site, sig))
                if first[1] != sig:
                    problems.append(
                        f"{name}: {first[0]} binds {first[1]}, {site} {sig}")
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
