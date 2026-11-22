// MDS settings read from the environment, at the places that read them: what
// an unclean value resolves to (the same as up to 2.38) and that it is
// reported.
#include "mds/mds_server.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <functional>
#include <string>

#include "fake_etcd.h"
#include "utils/env_parse.h"

using namespace dfkv;  // NOLINT

namespace {

std::string LogOf(const char* name, const char* value,
                  const std::function<void()>& read) {
  envparse::ForgetWarningsForTest();
  if (value != nullptr)
    ::setenv(name, value, 1);
  else
    ::unsetenv(name);
  testing::internal::CaptureStderr();
  read();
  std::string log = testing::internal::GetCapturedStderr();
  ::unsetenv(name);
  return log;
}
bool Names(const std::string& log, const char* name, const char* value) {
  return log.find(std::string(name) + "='" + value + "'") != std::string::npos;
}

// DFKV_MDS_MAX_CONNS was read with atol and "> 0".
TEST(MdsEnv, MaxConnsKeepsItsReading) {
  size_t fallback = 0;
  {
    ::unsetenv("DFKV_MDS_MAX_CONNS");
    FakeEtcd etcd;
    MdsServer mds(etcd.ep());
    ASSERT_EQ(mds.Start(0), Status::kOk);
    fallback = mds.max_conns();
    mds.Stop();
  }
  struct Case { const char* value; size_t conns; bool reported; };
  const Case cases[] = {{"", fallback, false},     {"16", 16, false},
                        {"16conns", 16, true},     {"abc", fallback, true},
                        {"0", fallback, true},     {"-4", fallback, true},
                        {" 16", 16, false},        {"+16", 16, true},
                        {"1e3", 1, true}};
  for (const Case& c : cases) {
    size_t conns = 0;
    const std::string log = LogOf("DFKV_MDS_MAX_CONNS", c.value, [&] {
      FakeEtcd etcd;
      MdsServer mds(etcd.ep());
      EXPECT_EQ(mds.Start(0), Status::kOk);
      conns = mds.max_conns();
      mds.Stop();
    });
    EXPECT_EQ(conns, c.conns) << "'" << c.value << "'";
    EXPECT_EQ(Names(log, "DFKV_MDS_MAX_CONNS", c.value), c.reported)
        << "'" << c.value << "': " << log;
  }
}

// DFKV_MDS_FIRST_REQ_MS was read with strtol, a full end check and errno: a
// number that does not fit a long gave the default, not the one-hour clamp.
TEST(MdsEnv, FirstRequestDeadlineKeepsItsReading) {
  struct Case { const char* value; long ms; bool reported; };
  const Case cases[] = {
      {nullptr, 30000, false},
      {"", 30000, false},
      {"0", 0, false},
      {"5000", 5000, false},
      {"7200000", 3600000, true},                // clamped to one hour
      {"9223372036854775807", 3600000, true},    // LONG_MAX: still a long
      {"9223372036854775808", 30000, true},      // 2^63: ERANGE, the default
      {"18446744073709551615", 30000, true},     // 2^64-1
      {"99999999999999999999999999", 30000, true},
      {"-1", 30000, true},
      {"abc", 30000, true},
      {"5000ms", 30000, true},
      {" 5000", 5000, false}};
  for (const Case& c : cases) {
    const char* shown = c.value ? c.value : "(unset)";
    long ms = -1;
    const std::string log = LogOf("DFKV_MDS_FIRST_REQ_MS", c.value,
                                  [&] { ms = MdsFirstReqMsFromEnv(); });
    EXPECT_EQ(ms, c.ms) << shown;
    EXPECT_EQ(Names(log, "DFKV_MDS_FIRST_REQ_MS", c.value ? c.value : ""),
              c.reported)
        << shown << ": " << log;
  }
}

}  // namespace
