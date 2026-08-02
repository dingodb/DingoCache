#include "common/config_dump.h"

#include <cstdlib>
#include <string>

#include "gtest/gtest.h"

namespace dfkv {
namespace {
namespace cd = config_dump;

class ConfigDumpTest : public ::testing::Test {
 protected:
  void SetUp() override { cd::ResetForTest(); }
  void TearDown() override {
    cd::ResetForTest();
    for (const char* k : {"DFKV_TEST_I", "DFKV_TEST_U", "DFKV_TEST_B",
                          "DFKV_TEST_S", "DFKV_TEST_SCAN"})
      ::unsetenv(k);
  }
};

TEST_F(ConfigDumpTest, EnvI64SetEmptyUnsetMalformed) {
  ::unsetenv("DFKV_TEST_I");
  EXPECT_EQ(cd::EnvI64("DFKV_TEST_I", 7), 7);   // unset -> default
  ::setenv("DFKV_TEST_I", "", 1);
  EXPECT_EQ(cd::EnvI64("DFKV_TEST_I", 7), 7);   // empty -> default
  ::setenv("DFKV_TEST_I", "42", 1);
  EXPECT_EQ(cd::EnvI64("DFKV_TEST_I", 7), 42);  // set
  ::setenv("DFKV_TEST_I", "-5", 1);
  EXPECT_EQ(cd::EnvI64("DFKV_TEST_I", 7), -5);  // negative ok
  ::setenv("DFKV_TEST_I", "abc", 1);
  EXPECT_EQ(cd::EnvI64("DFKV_TEST_I", 7), 7);   // malformed -> default
}

TEST_F(ConfigDumpTest, EnvU64StrBool) {
  ::setenv("DFKV_TEST_U", "1000", 1);
  EXPECT_EQ(cd::EnvU64("DFKV_TEST_U", 1), 1000u);
  ::unsetenv("DFKV_TEST_U");
  EXPECT_EQ(cd::EnvU64("DFKV_TEST_U", 9), 9u);

  ::setenv("DFKV_TEST_S", "hello", 1);
  EXPECT_EQ(cd::EnvStr("DFKV_TEST_S", "def"), "hello");
  ::unsetenv("DFKV_TEST_S");
  EXPECT_EQ(cd::EnvStr("DFKV_TEST_S", "def"), "def");

  for (const char* t : {"1", "true", "yes", "on", "TRUE"}) {
    ::setenv("DFKV_TEST_B", t, 1);
    EXPECT_TRUE(cd::EnvBool("DFKV_TEST_B", false)) << t;
  }
  for (const char* f : {"0", "false", "no", "off", "OFF"}) {
    ::setenv("DFKV_TEST_B", f, 1);
    EXPECT_FALSE(cd::EnvBool("DFKV_TEST_B", true)) << f;
  }
  ::unsetenv("DFKV_TEST_B");
  EXPECT_TRUE(cd::EnvBool("DFKV_TEST_B", true));    // unset -> default
  EXPECT_FALSE(cd::EnvBool("DFKV_TEST_B", false));
}

TEST_F(ConfigDumpTest, EmitShowsRecordsWithSourceAndFoldsEnvScan) {
  cd::Record("key_namespace", "test/model", cd::Source::kFlag);
  ::setenv("DFKV_TEST_SCAN", "scanned", 1);  // not recorded -> scan folds it in
  testing::internal::CaptureStderr();
  cd::Emit("client");
  const std::string out = testing::internal::GetCapturedStderr();
  EXPECT_NE(out.find("effective config (client)"), std::string::npos);
  EXPECT_NE(out.find("key_namespace"), std::string::npos);
  EXPECT_NE(out.find("test/model  (flag)"), std::string::npos);
  EXPECT_NE(out.find("DFKV_TEST_SCAN"), std::string::npos);
  EXPECT_NE(out.find("scanned  (env)"), std::string::npos);
}

TEST_F(ConfigDumpTest, ExplicitRecordWinsOverEnvScan) {
  // A clamped knob: raw env 16 but effective 4 recorded — dump must show 4.
  ::setenv("DFKV_TEST_SCAN", "16", 1);
  cd::Record("DFKV_TEST_SCAN", "4", cd::Source::kEnv);
  testing::internal::CaptureStderr();
  cd::Emit("server");
  const std::string out = testing::internal::GetCapturedStderr();
  EXPECT_NE(out.find("DFKV_TEST_SCAN"), std::string::npos);
  // Alignment width depends on any ambient DFKV_* variables on the host.
  EXPECT_NE(out.find(" = 4  (env)"), std::string::npos);
  EXPECT_EQ(out.find(" = 16  (env)"), std::string::npos);
}

TEST_F(ConfigDumpTest, EmitClearsRegistry) {
  cd::Record("uniqkey_zzz", "1", cd::Source::kFlag);
  testing::internal::CaptureStderr();
  cd::Emit("a");
  EXPECT_NE(testing::internal::GetCapturedStderr().find("uniqkey_zzz"),
            std::string::npos);
  // Second Emit: the key was cleared, so it must not reappear (robust to any
  // ambient DFKV_* env the test host may carry).
  testing::internal::CaptureStderr();
  cd::Emit("b");
  EXPECT_EQ(testing::internal::GetCapturedStderr().find("uniqkey_zzz"),
            std::string::npos);
}

TEST_F(ConfigDumpTest, SourceLabelsDistinguishExternalFromDefault) {
  cd::Record("a_flag", "1", cd::Source::kFlag);
  cd::Record("a_arg", "81", cd::Source::kArg);
  cd::Record("a_default", "8", cd::Source::kDefault);
  testing::internal::CaptureStderr();
  cd::Emit("mix");
  const std::string out = testing::internal::GetCapturedStderr();
  // Names are right-padded for alignment, so match on the stable " = value
  // (source)" tail rather than the name-to-equals spacing.
  EXPECT_NE(out.find(" = 1  (flag)"), std::string::npos);
  EXPECT_NE(out.find(" = 81  (arg)"), std::string::npos);
  EXPECT_NE(out.find(" = 8  (default)"), std::string::npos);
}

TEST_F(ConfigDumpTest, RecordResolvedSourceFollowsEnvPresence) {
  ::unsetenv("DFKV_TEST_SCAN");
  cd::RecordResolved("DFKV_TEST_SCAN", "4");        // env unset -> default
  ::setenv("DFKV_TEST_S", "x", 1);
  cd::RecordResolved("DFKV_TEST_S", "buffered");    // env set -> env (effective value)
  testing::internal::CaptureStderr();
  cd::Emit("srv");
  const std::string out = testing::internal::GetCapturedStderr();
  EXPECT_NE(out.find(" = 4  (default)"), std::string::npos);
  EXPECT_NE(out.find(" = buffered  (env)"), std::string::npos);
}

TEST_F(ConfigDumpTest, DfkvLogEnvIsNotFoldedIn) {
  ::setenv("DFKV_LOG", "info", 1);
  cd::Record("uniqkey_zzz", "1", cd::Source::kFlag);
  testing::internal::CaptureStderr();
  cd::Emit("x");
  const std::string out = testing::internal::GetCapturedStderr();
  EXPECT_EQ(out.find("DFKV_LOG"), std::string::npos);  // logger's own knob, skipped
}

}  // namespace
}  // namespace dfkv
