// TRN-30: the RDMA client's numeric settings are read through envparse
// specs. Each reader must resolve every text as the transport's own reader
// did, and warn with the same words. The old readers are copied here as
// they were; the only intended difference is ReadInt above INT_MAX.
#include "transport/rdma_settings.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "utils/env_parse.h"

using namespace dfkv;  // NOLINT

namespace {

// ---- the readers rdma_transport.cc had, unchanged ----
int OldEnvInt(const char* name, int dflt) {
  const char* v = std::getenv(name);
  if (!v || !*v) return dflt;
  long x = std::strtol(v, nullptr, 10);
  const int resolved = x > 0 ? static_cast<int>(x) : dflt;
  envparse::ReportUnlessPlain(name, 1, std::numeric_limits<int>::max(),
                              std::to_string(resolved));
  return resolved;
}

int OldEnvBoundedInt(const char* name, int dflt, int upper) {
  const char* v = std::getenv(name);
  if (!v || !*v) return dflt;
  errno = 0;
  char* end = nullptr;
  long x = std::strtol(v, &end, 10);
  const int resolved =
      (errno != 0 || end == v || *end != '\0' || x <= 0)
          ? dflt
          : static_cast<int>(std::min<long>(x, upper));
  envparse::ReportUnlessPlain(name, dflt == 0 ? 0 : 1,
                              static_cast<uint64_t>(upper),
                              std::to_string(resolved));
  return resolved;
}

constexpr uint64_t kMaxSgeBytes = std::numeric_limits<uint32_t>::max() - 8192;

size_t OldEnvBytes(const char* name, size_t dflt) {
  const char* v = std::getenv(name);
  if (!v || !*v) return dflt;
  errno = 0;
  char* end = nullptr;
  unsigned long long x = std::strtoull(v, &end, 10);
  size_t resolved = dflt;
  if (!(errno != 0 || end == v || x == 0))
    resolved = static_cast<size_t>(x > kMaxSgeBytes ? kMaxSgeBytes : x);
  envparse::ReportUnlessPlain(name, 1, kMaxSgeBytes, std::to_string(resolved));
  return resolved;
}

uint64_t OldEnvU64(const char* name, uint64_t dflt) {
  const char* value = std::getenv(name);
  if (!value || !*value) return dflt;
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  const uint64_t resolved =
      errno == 0 && end != value && *end == '\0' && parsed > 0
          ? static_cast<uint64_t>(parsed)
          : dflt;
  envparse::ReportUnlessPlain(name, 1, std::numeric_limits<uint64_t>::max(),
                              std::to_string(resolved));
  return resolved;
}
// ----

constexpr const char* kName = "DFKV_RDMA_TEST_SETTING";

std::vector<std::string> Corpus(const std::vector<std::string>& bounds) {
  std::vector<std::string> texts = {
      "",     "0",       "5",          " 5",          "\t3000",
      "+5",   "-5",      "-0",         "5s",          "5 ",
      "abc",  "64M",     "2147483647", "2147483648",  "4294967296",
      "9223372036854775807",           "9223372036854775808",
      "18446744073709551615",          "18446744073709551616",
      "18446744073709551617",          "-18446744073709551615",
      "99999999999999999999999"};
  texts.insert(texts.end(), bounds.begin(), bounds.end());
  return texts;
}

// The value a reader gives for `text` (nullptr: unset) and the warning it
// logs, from "invalid" on (the log line's prefix carries the time).
template <class Reader>
std::pair<std::string, std::string> Read(Reader reader, const char* text) {
  if (text)
    ::setenv(kName, text, 1);
  else
    ::unsetenv(kName);
  envparse::ForgetWarningsForTest();
  testing::internal::CaptureStderr();
  const std::string value = std::to_string(reader());
  std::string log = testing::internal::GetCapturedStderr();
  const auto at = log.find("invalid ");
  log = at == std::string::npos ? std::string() : log.substr(at);
  ::unsetenv(kName);
  return {value, log};
}

template <class Old, class New>
void ExpectSame(Old old_reader, New new_reader,
                const std::vector<std::string>& texts) {
  auto unset_old = Read(old_reader, nullptr);
  auto unset_new = Read(new_reader, nullptr);
  EXPECT_EQ(unset_new, unset_old) << "unset";
  size_t warned = 0;
  for (const auto& text : texts) {
    const auto old_read = Read(old_reader, text.c_str());
    EXPECT_EQ(Read(new_reader, text.c_str()), old_read) << "'" << text << "'";
    warned += !old_read.second.empty();
  }
  EXPECT_GE(warned, 10u) << "the warnings were not captured";
}

TEST(RdmaSettings, BoundedIntsResolveAndWarnAsBefore) {
  // One setting of each shape the transport has: a positive default with a
  // bound, the bound equal to the default, and a default of 0 (not set).
  for (const auto& [dflt, upper] :
       std::vector<std::pair<int, int>>{{2000, 30000},
                                        {30000, 30000},
                                        {64, 4096},
                                        {100000, 60000000},
                                        {0, 60000},
                                        {0, std::numeric_limits<int>::max()}}) {
    SCOPED_TRACE(std::to_string(dflt) + " up to " + std::to_string(upper));
    ExpectSame([&] { return OldEnvBoundedInt(kName, dflt, upper); },
               [&] { return rdma_settings::ReadBoundedInt(kName, dflt, upper); },
               Corpus({std::to_string(upper), std::to_string(upper + 1LL)}));
  }
}

TEST(RdmaSettings, BytesResolveAndWarnAsBefore) {
  for (const size_t dflt : {size_t{4} << 20, size_t{256} << 10,
                            size_t{64} << 20}) {
    SCOPED_TRACE(dflt);
    ExpectSame([&] { return OldEnvBytes(kName, dflt); },
               [&] { return rdma_settings::ReadBytes(kName, dflt,
                                                     kMaxSgeBytes); },
               Corpus({std::to_string(kMaxSgeBytes),
                       std::to_string(kMaxSgeBytes + 1)}));
  }
}

TEST(RdmaSettings, U64sResolveAndWarnAsBefore) {
  for (const uint64_t dflt : {uint64_t{1024}, uint64_t{545259520}}) {
    SCOPED_TRACE(dflt);
    ExpectSame([&] { return OldEnvU64(kName, dflt); },
               [&] { return rdma_settings::ReadU64(kName, dflt); }, Corpus({}));
  }
}

// ReadInt resolves and warns as before for every text the old reader got
// right; a number above INT_MAX, which the old reader cast to a negative or
// zero timeout, is now the default.
TEST(RdmaSettings, IntsResolveAsBeforeAndAboveIntMaxGiveTheDefault) {
  const int dflt = 3000;
  const auto old_reader = [&] { return OldEnvInt(kName, dflt); };
  const auto new_reader = [&] { return rdma_settings::ReadInt(kName, dflt); };
  EXPECT_EQ(Read(new_reader, nullptr), Read(old_reader, nullptr));
  for (const auto& text : Corpus({})) {
    // Above INT_MAX the strtol value does not fit an int.
    const long parsed = std::strtol(text.c_str(), nullptr, 10);
    const bool above_int = parsed > std::numeric_limits<int>::max();
    const auto fresh = Read(new_reader, text.c_str());
    if (above_int) {
      EXPECT_EQ(fresh.first, std::to_string(dflt)) << "'" << text << "'";
      EXPECT_NE(fresh.second.find("using 3000, as before"), std::string::npos)
          << fresh.second;
    } else {
      EXPECT_EQ(fresh, Read(old_reader, text.c_str())) << "'" << text << "'";
    }
  }
  // The old reader's results there, for the record: what CONNECT_MS and
  // IO_MS became.
  EXPECT_LE(Read(old_reader, "2147483648").first[0], '0')
      << "the old reader did not go negative";
  EXPECT_EQ(Read(old_reader, "4294967296").first, "0");
}

}  // namespace
