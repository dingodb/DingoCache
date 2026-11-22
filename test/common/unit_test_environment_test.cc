// The build gives every unit test the environment in DFKV_UNIT_TEST_ENV
// (CMakeLists.txt). Each item must arrive, not only the first: a test that
// builds a store from the environment depends on all of them.

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

namespace {
std::string Env(const char* name) {
  const char* value = std::getenv(name);
  return value ? value : "<unset>";
}
}  // namespace

TEST(UnitTestEnvironment, EveryItemOfTheTestEnvironmentIsSet) {
  EXPECT_EQ(Env("DFKV_STORE_ENGINE"), "file");
  EXPECT_EQ(Env("DFKV_SLAB_WRITE"), "buffered");
}
