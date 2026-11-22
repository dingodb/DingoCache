// The environment every unit test starts from, linked into each unit test
// binary. Set before main() and only where the variable is not set already,
// so a run that sets one itself (from a shell, say) keeps its value.
//
// It used to be a CTest ENVIRONMENT property with two items. The tests are
// registered with gtest_discover_tests, which passes PROPERTIES as a flat
// list through several CMake layers; the ';' between the items split the
// value, and every test ran with the first item only. Here the list cannot be
// split, and a test binary run by hand gets the same environment as under
// ctest.
//
// DFKV_STORE_ENGINE=file: tests that do not ask for an engine use the file
// backend, so a full run does not create 1 GiB slab extents in /tmp.
// DFKV_SLAB_WRITE=buffered: a slab store built through the disk group writes
// buffered; the tests of direct writes ask for them.

#include <cstdlib>

namespace {
struct UnitTestDefaults {
  UnitTestDefaults() {
    ::setenv("DFKV_STORE_ENGINE", "file", /*overwrite=*/0);
    ::setenv("DFKV_SLAB_WRITE", "buffered", /*overwrite=*/0);
  }
};
const UnitTestDefaults kUnitTestDefaults;
}  // namespace
