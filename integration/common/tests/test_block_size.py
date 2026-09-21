import ctypes
import unittest
from types import SimpleNamespace

from dfkv_common.block_size import get_max_block_bytes, validate_object_sizes


class BlockSizeTest(unittest.TestCase):
    def test_native_bound_preserves_unsigned_64_bit_value(self):
        effective = (1 << 63) + 17
        expected_handle = (1 << 32) + 7
        getter = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p)(
            lambda handle: effective if handle == expected_handle else 0
        )
        # Emulate ctypes' default integer return type before binding the ABI.
        getter.restype = ctypes.c_int
        lib = SimpleNamespace(dfkv_max_block_bytes=getter)
        self.assertEqual(
            get_max_block_bytes(lib, ctypes.c_void_p(expected_handle)), effective
        )

    def test_missing_native_export_fails_closed(self):
        with self.assertRaises(RuntimeError) as raised:
            get_max_block_bytes(object(), ctypes.c_void_p(1))
        message = str(raised.exception)
        self.assertIn("dfkv_max_block_bytes", message)
        self.assertIn("upgrade", message)

    def test_zero_native_bound_fails_closed(self):
        getter = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p)(
            lambda _handle: 0
        )
        lib = SimpleNamespace(dfkv_max_block_bytes=getter)
        with self.assertRaisesRegex(RuntimeError, "0 bytes"):
            get_max_block_bytes(lib, None)

    def test_independent_objects_fit_at_equality_without_summing(self):
        validate_object_sizes(16, iter([9, 16]), context="separate K/V pools")
        with self.assertRaises(ValueError):
            validate_object_sizes(16, [9 + 16], context="one SG object")

    def test_oversize_reports_largest_object_and_effective_bound(self):
        with self.assertRaises(ValueError) as raised:
            validate_object_sizes(16, iter([8, 17, 32]), context="HiCache pool")
        message = str(raised.exception)
        for detail in (
            "HiCache pool", "required", "32 bytes", "effective", "16 bytes",
            "DFKV_RDMA_MAX_BLOCK_BYTES", "payload cap", "server limits",
        ):
            self.assertIn(detail, message)

    def test_invalid_object_sizes_are_not_coerced_or_ignored(self):
        for size in (0, -1, True, 1.0, "1", None):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    validate_object_sizes(16, [8, size], context="vLLM layout")

    def test_empty_geometry_is_not_assumed_to_fit(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_object_sizes(16, iter(()), context="unregistered pool")

    def test_invalid_effective_bound_is_rejected(self):
        for limit in (0, -1, True, 16.0):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    validate_object_sizes(limit, [1], context="RDMA client")


if __name__ == "__main__":
    unittest.main()
