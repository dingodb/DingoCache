// What a setting read as text resolves to, per class of unclean input, for
// each way a setting used to be read. Two things are checked throughout: the
// value is the one the old reader gave, and anything unclean is reported.
#include "utils/env_parse.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <functional>
#include <limits>
#include <string>

using namespace dfkv;            // NOLINT
using namespace dfkv::envparse;  // NOLINT

namespace {

constexpr uint64_t kTop = std::numeric_limits<uint64_t>::max();

struct Read {
  uint64_t value;
  bool warned;
  bool from_text;
  std::string warning;
};
Read U64(const char* text, const U64Spec& spec) {
  Read r{};
  r.value = ParseU64("DFKV_X", text, spec, &r.warning, &r.from_text);
  r.warned = !r.warning.empty();
  return r;
}
U64Spec Spec(Legacy legacy, uint64_t fallback, uint64_t min, uint64_t max,
             Above above = Above::kDefault,
             Negative negative = Negative::kInvalid) {
  U64Spec spec;
  spec.legacy = legacy;
  spec.fallback = fallback;
  spec.min = min;
  spec.max = max;
  spec.above = above;
  spec.negative = negative;
  return spec;
}
const Legacy kAllLegacy[] = {Legacy::kStrict, Legacy::kFull, Legacy::kPrefix,
                             Legacy::kPrefixOrZero};

// The value is the default and the line names setting, value and default.
void ExpectDefault(const Read& r, const char* text) {
  EXPECT_EQ(r.value, 7u) << "'" << text << "'";
  EXPECT_FALSE(r.from_text) << text;
  ASSERT_TRUE(r.warned) << "'" << text << "'";
  EXPECT_NE(r.warning.find("DFKV_X='" + std::string(text) + "'"),
            std::string::npos)
      << r.warning;
  EXPECT_NE(r.warning.find("using the default 7"), std::string::npos)
      << r.warning;
}
// The value is what the old reader gave, and the line says "as before".
void ExpectKept(const Read& r, const char* text, uint64_t value) {
  EXPECT_EQ(r.value, value) << "'" << text << "'";
  EXPECT_TRUE(r.from_text) << text;
  ASSERT_TRUE(r.warned) << "'" << text << "'";
  EXPECT_NE(r.warning.find("DFKV_X='" + std::string(text) + "'"),
            std::string::npos)
      << r.warning;
  EXPECT_NE(r.warning.find("using " + std::to_string(value) + " as before"),
            std::string::npos)
      << r.warning;
  EXPECT_NE(r.warning.find("later release"), std::string::npos) << r.warning;
}

TEST(EnvParse, PlainNumberInRangeIsTakenSilently) {
  for (Legacy legacy : kAllLegacy) {
    for (const char* text : {"30", "030", "1", "100"}) {
      const Read r = U64(text, Spec(legacy, 7, 1, 100));
      EXPECT_EQ(r.value, std::strtoull(text, nullptr, 10)) << text;
      EXPECT_FALSE(r.warned) << text << ": " << r.warning;
      EXPECT_TRUE(r.from_text);
    }
    EXPECT_EQ(U64("18446744073709551615", Spec(legacy, 7, 0, kTop)).value, kTop);
  }
}

TEST(EnvParse, UnsetAndEmptyAreTheDefaultWithoutAWarning) {
  for (Legacy legacy : kAllLegacy) {
    for (const char* text : {static_cast<const char*>(nullptr), ""}) {
      const Read r = U64(text, Spec(legacy, 7, 0, 100));
      EXPECT_EQ(r.value, 7u);
      EXPECT_FALSE(r.warned) << r.warning;
      EXPECT_FALSE(r.from_text);
    }
  }
}

// ...except for the readers that did not test for an empty string: it was 0
// and stays 0, with a line that says what 0 means and what to write.
TEST(EnvParse, EmptyStaysZeroWhereItWasReadAsZero) {
  U64Spec spec = Spec(Legacy::kPrefixOrZero, 80, 0, 100);
  spec.empty_is_zero = true;
  spec.zero_means = "every PUT is acknowledged from disk";
  const Read r = U64("", spec);
  EXPECT_EQ(r.value, 0u);
  EXPECT_TRUE(r.from_text);
  ASSERT_TRUE(r.warned);
  EXPECT_NE(r.warning.find("DFKV_X=''"), std::string::npos) << r.warning;
  EXPECT_NE(r.warning.find("an empty value is read as 0"), std::string::npos)
      << r.warning;
  EXPECT_NE(r.warning.find("every PUT is acknowledged from disk"),
            std::string::npos)
      << r.warning;
  EXPECT_NE(r.warning.find("unset for the default 80"), std::string::npos)
      << r.warning;
  // An explicit 0 is the same value without a line.
  EXPECT_EQ(U64("0", spec).value, 0u);
  EXPECT_FALSE(U64("0", spec).warned);
  // Below the minimum the 0 was thrown away by the reader's own check.
  spec.min = 1;
  EXPECT_EQ(U64("", spec).value, 80u);
  EXPECT_TRUE(U64("", spec).warned);
}

TEST(EnvParse, NotANumberIsTheDefaultOrZeroForUncheckedReaders) {
  const char* const words[] = {"abc", "on", "x10", ".5", "e3", "s30", " ", "-",
                               "+", "-x"};
  for (Legacy legacy : {Legacy::kStrict, Legacy::kFull, Legacy::kPrefix})
    for (const char* text : words)
      ExpectDefault(U64(text, Spec(legacy, 7, 0, 100)), text);
  // atoi("abc") == 0, and 0 was in range: the setting has been running on 0.
  for (const char* text : words)
    ExpectKept(U64(text, Spec(Legacy::kPrefixOrZero, 7, 0, 100)), text, 0);
  // Where the reader then required n > 0, it has been running on the default.
  for (const char* text : words)
    ExpectDefault(U64(text, Spec(Legacy::kPrefixOrZero, 7, 1, 100)), text);
}

TEST(EnvParse, TrailingCharactersAreInvalidOrIgnoredAsBefore) {
  struct Case { const char* text; uint64_t prefix; };
  const Case cases[] = {{"30s", 30}, {"64G", 64}, {"1e3", 1}, {"30.5", 30},
                        {"30 s", 30}, {"30,5", 30}, {"5-", 5}, {"0x10", 0},
                        {"30 ", 30}, {"30\n", 30}};
  for (const Case& c : cases) {
    ExpectDefault(U64(c.text, Spec(Legacy::kStrict, 7, 0, 100000)), c.text);
    ExpectDefault(U64(c.text, Spec(Legacy::kFull, 7, 0, 100000)), c.text);
    ExpectKept(U64(c.text, Spec(Legacy::kPrefix, 7, 0, 100000)), c.text,
               c.prefix);
    ExpectKept(U64(c.text, Spec(Legacy::kPrefixOrZero, 7, 0, 100000)), c.text,
               c.prefix);
  }
  // The leading number still has to be in range.
  ExpectDefault(U64("500s", Spec(Legacy::kPrefix, 7, 0, 100)), "500s");
  ExpectDefault(U64("0s", Spec(Legacy::kPrefix, 7, 1, 100)), "0s");
}

// strtoul skips leading blanks and takes a sign; the strict readers looked at
// the first character.
TEST(EnvParse, LeadingBlanksAndPlusSign) {
  ExpectDefault(U64(" 30", Spec(Legacy::kStrict, 7, 0, 100)), " 30");
  ExpectDefault(U64("+30", Spec(Legacy::kStrict, 7, 0, 100)), "+30");
  for (Legacy legacy :
       {Legacy::kFull, Legacy::kPrefix, Legacy::kPrefixOrZero}) {
    const Read blank = U64(" \t30", Spec(legacy, 7, 0, 100));
    EXPECT_EQ(blank.value, 30u);
    EXPECT_FALSE(blank.warned) << blank.warning;
    ExpectKept(U64("+30", Spec(legacy, 7, 0, 100)), "+30", 30);
  }
}

TEST(EnvParse, NegativeIsInvalidOrWrapsOrIsZeroAsTheReaderDid) {
  for (Legacy legacy : kAllLegacy)
    for (const char* text : {"-1", "-5", "-99999999999999999999999"})
      ExpectDefault(U64(text, Spec(legacy, 7, 0, kTop)), text);

  // strtoull("-1") is 2^64-1; the reader then clamped it...
  ExpectKept(U64("-1", Spec(Legacy::kPrefix, 7, 0, kTop, Above::kClamp,
                            Negative::kWraps)),
             "-1", kTop);
  ExpectKept(U64("-5", Spec(Legacy::kFull, 7, 0, kTop, Above::kClamp,
                            Negative::kWraps)),
             "-5", kTop - 4);
  Read r = U64("-1", Spec(Legacy::kPrefix, 7, 0, 600000, Above::kClamp,
                          Negative::kWraps));
  EXPECT_EQ(r.value, 600000u);
  EXPECT_TRUE(r.from_text);
  EXPECT_NE(r.warning.find("DFKV_X='-1' clamped to the maximum 600000"),
            std::string::npos)
      << r.warning;
  // ...or threw it away with its range check.
  ExpectDefault(U64("-1", Spec(Legacy::kFull, 7, 1, 1024, Above::kDefault,
                               Negative::kWraps)),
                "-1");
  // strtol readers that treated v <= 0 alike.
  U64Spec idle = Spec(Legacy::kPrefixOrZero, 600000, 0, 86400000,
                      Above::kClamp, Negative::kZero);
  idle.zero_means = "the idle reaper is off";
  r = U64("-1", idle);
  EXPECT_EQ(r.value, 0u);
  ASSERT_TRUE(r.warned);
  EXPECT_NE(r.warning.find("a negative number is read as 0"), std::string::npos)
      << r.warning;
  EXPECT_NE(r.warning.find("the idle reaper is off"), std::string::npos);
  // -0 is 0.
  EXPECT_EQ(U64("-0", Spec(Legacy::kFull, 7, 0, 100)).value, 0u);
}

TEST(EnvParse, OutOfRangeIsTheDefaultOrTheMaximumForClampingSettings) {
  for (Legacy legacy : kAllLegacy) {
    ExpectDefault(U64("101", Spec(legacy, 7, 1, 100)), "101");
    const Read r = U64("101", Spec(legacy, 7, 1, 100, Above::kClamp));
    EXPECT_EQ(r.value, 100u);
    EXPECT_TRUE(r.from_text);
    EXPECT_NE(r.warning.find("DFKV_X='101' clamped to the maximum 100"),
              std::string::npos)
        << r.warning;
    // Below the minimum is never clamped up.
    ExpectDefault(U64("0", Spec(legacy, 7, 1, 100, Above::kClamp)), "0");
  }
}

TEST(EnvParse, OverflowSaturatesOrIsTheDefaultForReadersThatCheckedErrno) {
  const char* huge = "99999999999999999999999999";  // > 2^64
  for (Legacy legacy : kAllLegacy) {
    // strtoull returns 2^64-1 and the unchecked reader went on with it.
    ExpectKept(U64(huge, Spec(legacy, 7, 0, kTop)), huge, kTop);
    ExpectKept(U64("18446744073709551616", Spec(legacy, 7, 0, kTop)),
               "18446744073709551616", kTop);
    EXPECT_EQ(U64(huge, Spec(legacy, 7, 0, 3600, Above::kClamp)).value, 3600u);
    ExpectDefault(U64(huge, Spec(legacy, 7, 0, 3600)), huge);
    U64Spec errno_checked = Spec(legacy, 7, 0, 3600, Above::kClamp);
    errno_checked.overflow_is_default = true;
    ExpectDefault(U64(huge, errno_checked), huge);
  }
}

// strtol and atol hold 2^63-1: one more is their overflow.
TEST(EnvParse, SignedReadersOverflowAboveTwoToThe63) {
  const char* fits = "9223372036854775807";
  const char* over = "9223372036854775808";
  U64Spec spec = Spec(Legacy::kFull, 7, 0, kTop);
  spec.representable_max = kSignedMax;
  Read r = U64(fits, spec);
  EXPECT_EQ(r.value, kSignedMax);
  EXPECT_FALSE(r.warned) << r.warning;
  ExpectKept(U64(over, spec), over, kSignedMax);  // saturated
  spec.max = 3600000;
  spec.above = Above::kClamp;
  EXPECT_EQ(U64(fits, spec).value, 3600000u);
  EXPECT_EQ(U64(over, spec).value, 3600000u);
  spec.overflow_is_default = true;  // the reader checked errno
  EXPECT_EQ(U64(fits, spec).value, 3600000u);
  ExpectDefault(U64(over, spec), over);
  ExpectDefault(U64("18446744073709551615", spec), "18446744073709551615");
}

// What is wrong with the text comes first, not the range it then falls out of.
TEST(EnvParse, ReasonNamesWhatIsUncleanBeforeTheRange) {
  Read r = U64("abc", Spec(Legacy::kPrefixOrZero, 7, 1, 100));
  EXPECT_NE(r.warning.find("not a number"), std::string::npos) << r.warning;
  r = U64("500s", Spec(Legacy::kPrefix, 7, 0, 100));
  EXPECT_NE(r.warning.find("'s' after the number is ignored"),
            std::string::npos)
      << r.warning;
  EXPECT_NE(r.warning.find("above the maximum 100"), std::string::npos)
      << r.warning;
}

// A negative that wraps back into range under strtoull was a usable value.
TEST(EnvParse, NegativeThatWrapsIntoRangeIsKept) {
  const char* five = "-18446744073709551611";  // 2^64 - 5
  ExpectKept(U64(five, Spec(Legacy::kPrefixOrZero, 7, 1, 64, Above::kDefault,
                            Negative::kWraps)),
             five, 5);
  ExpectDefault(U64(five, Spec(Legacy::kPrefixOrZero, 7, 1, 64)), five);
}

// ---- switches ---------------------------------------------------------------

struct Flip {
  bool value;
  std::string warning;
};
Flip Sw(const char* text, bool fallback, OffWhen off_when) {
  Flip f{};
  f.value = ParseSwitch("DFKV_B", text, fallback, off_when, &f.warning);
  return f;
}
const OffWhen kAllRules[] = {OffWhen::kExactlyZero, OffWhen::kStartsWithZero,
                             OffWhen::kOffWord};

TEST(EnvSwitch, WordsThatSayWhatTheSwitchDoesAreSilent) {
  for (OffWhen rule : kAllRules) {
    for (const char* on : {"1", "on", "ON", "On", "true", "TRUE", "yes"}) {
      EXPECT_TRUE(Sw(on, false, rule).value) << on;
      EXPECT_TRUE(Sw(on, false, rule).warning.empty()) << on;
    }
    EXPECT_FALSE(Sw("0", true, rule).value);
    EXPECT_TRUE(Sw("0", true, rule).warning.empty());
    for (const char* unset : {static_cast<const char*>(nullptr), ""}) {
      EXPECT_TRUE(Sw(unset, true, rule).value);
      EXPECT_FALSE(Sw(unset, false, rule).value);
      EXPECT_TRUE(Sw(unset, true, rule).warning.empty());
    }
  }
}

// The switches that only knew "0": off, false and no have been switching the
// feature ON. They still do, and now say so.
TEST(EnvSwitch, OffWordsKeepSwitchingOnWhereOnlyZeroWasOff) {
  for (OffWhen rule : {OffWhen::kExactlyZero, OffWhen::kStartsWithZero}) {
    for (const char* word : {"off", "false", "no", "OFF", "False", "No"}) {
      const Flip f = Sw(word, false, rule);
      EXPECT_TRUE(f.value) << word;
      EXPECT_NE(f.warning.find("DFKV_B='" + std::string(word) + "'"),
                std::string::npos)
          << f.warning;
      EXPECT_NE(f.warning.find("read as on, as before"), std::string::npos)
          << f.warning;
      EXPECT_NE(f.warning.find("0 switches it off"), std::string::npos)
          << f.warning;
      EXPECT_NE(f.warning.find("later release will read on/off words by "
                               "their meaning"),
                std::string::npos)
          << f.warning;
    }
  }
  // The one reader that knew the words is silent about them.
  for (const char* word : {"off", "false", "no", "OFF", "False"}) {
    const Flip f = Sw(word, true, OffWhen::kOffWord);
    EXPECT_FALSE(f.value) << word;
    EXPECT_TRUE(f.warning.empty()) << f.warning;
  }
}

// "00" and "0abc": off where the first character decided, on where the whole
// value had to be "0".
TEST(EnvSwitch, LeadingZeroKeepsItsReading) {
  for (const char* text : {"00", "0abc", "0 ", "01"}) {
    Flip f = Sw(text, true, OffWhen::kStartsWithZero);
    EXPECT_FALSE(f.value) << text;
    EXPECT_NE(f.warning.find("read as off, as before"), std::string::npos)
        << f.warning;
    f = Sw(text, false, OffWhen::kExactlyZero);
    EXPECT_TRUE(f.value) << text;
    EXPECT_NE(f.warning.find("read as on, as before"), std::string::npos)
        << f.warning;
  }
}

TEST(EnvSwitch, AnyOtherWordIsOnAsBeforeAndReported) {
  for (OffWhen rule : kAllRules) {
    for (const char* word : {"2", "enable", "y", "offf", "-1", " on", "abc"}) {
      const Flip f = Sw(word, false, rule);
      EXPECT_TRUE(f.value) << word;
      EXPECT_NE(f.warning.find("read as on, as before"), std::string::npos)
          << word << ": " << f.warning;
    }
  }
}

// ---- reading the environment, once per setting ------------------------------

TEST(EnvParse, EnvFunctionsReadTheEnvironment) {
  ::setenv("DFKV_ENV_PARSE_TEST", "42", 1);
  bool from_env = false;
  EXPECT_EQ(EnvU64("DFKV_ENV_PARSE_TEST", Spec(Legacy::kStrict, 7, 0, 100),
                   &from_env),
            42u);
  EXPECT_TRUE(from_env);
  ::setenv("DFKV_ENV_PARSE_TEST", "forty-two", 1);
  EXPECT_EQ(EnvU64("DFKV_ENV_PARSE_TEST", Spec(Legacy::kStrict, 7, 0, 100),
                   &from_env),
            7u);
  EXPECT_FALSE(from_env);
  ::setenv("DFKV_ENV_PARSE_TEST", "Off", 1);
  EXPECT_FALSE(EnvSwitch("DFKV_ENV_PARSE_TEST", true, OffWhen::kOffWord));
  ::unsetenv("DFKV_ENV_PARSE_TEST");
  EXPECT_EQ(EnvU64("DFKV_ENV_PARSE_TEST", Spec(Legacy::kStrict, 7, 0, 100)), 7u);
  EXPECT_TRUE(EnvSwitch("DFKV_ENV_PARSE_TEST", true, OffWhen::kOffWord));
}

// Settings are read per disk, per tier and per connection: a bad value is
// reported once per process, not on each of those.
TEST(EnvParse, ABadValueIsLoggedOncePerSetting) {
  ForgetWarningsForTest();
  ::setenv("DFKV_ENV_PARSE_ONCE", "12parsecs", 1);
  ::setenv("DFKV_ENV_PARSE_OTHER", "nope", 1);
  testing::internal::CaptureStderr();
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(EnvU64("DFKV_ENV_PARSE_ONCE", Spec(Legacy::kPrefix, 7, 0, 100)),
              12u);
    EXPECT_TRUE(EnvSwitch("DFKV_ENV_PARSE_OTHER", false, OffWhen::kExactlyZero));
  }
  const std::string log = testing::internal::GetCapturedStderr();
  ::unsetenv("DFKV_ENV_PARSE_ONCE");
  ::unsetenv("DFKV_ENV_PARSE_OTHER");
  auto count = [&](const std::string& what) {
    size_t n = 0;
    for (size_t at = log.find(what); at != std::string::npos;
         at = log.find(what, at + 1))
      ++n;
    return n;
  };
  EXPECT_EQ(count("DFKV_ENV_PARSE_ONCE='12parsecs'"), 1u) << log;
  EXPECT_EQ(count("DFKV_ENV_PARSE_OTHER='nope'"), 1u) << log;
}

// The level names --help documents are upper case.
TEST(LogLevel, IsReadInAnyCase) {
  using namespace dfkv::logging;  // NOLINT
  bool known = false;
  EXPECT_EQ(ParseLevel("WARN", &known), kWarn);
  EXPECT_TRUE(known);
  EXPECT_EQ(ParseLevel("warn", &known), kWarn);
  EXPECT_EQ(ParseLevel("Warning", &known), kWarn);
  EXPECT_EQ(ParseLevel("DEBUG", &known), kDebug);
  EXPECT_EQ(ParseLevel("debug", &known), kDebug);
  EXPECT_EQ(ParseLevel("Error", &known), kError);
  EXPECT_EQ(ParseLevel("INFO", &known), kInfo);
  EXPECT_TRUE(known);
  EXPECT_EQ(ParseLevel(nullptr, &known), kInfo);
  EXPECT_TRUE(known);
  EXPECT_EQ(ParseLevel("", &known), kInfo);
  EXPECT_TRUE(known);
  EXPECT_EQ(ParseLevel("verbose", &known), kInfo);
  EXPECT_FALSE(known);
  EXPECT_EQ(ParseLevel("3", &known), kInfo);
  EXPECT_FALSE(known);
}

}  // namespace

// ---- report only: the reader keeps its own parse ------------------------------
namespace {
// What ReportUnlessPlain / ReportSwitch logged for one value of one variable.
std::string Reported(const char* value, const std::function<void()>& report) {
  ForgetWarningsForTest();
  if (value != nullptr)
    ::setenv("DFKV_REPORT_TEST", value, 1);
  else
    ::unsetenv("DFKV_REPORT_TEST");
  testing::internal::CaptureStderr();
  report();
  std::string log = testing::internal::GetCapturedStderr();
  ::unsetenv("DFKV_REPORT_TEST");
  return log;
}
}  // namespace

TEST(EnvReport, PlainNumberIsDigitsOnlyAndInRange) {
  EXPECT_TRUE(IsPlainNumber("0", 0, 10));
  EXPECT_TRUE(IsPlainNumber("10", 0, 10));
  EXPECT_TRUE(IsPlainNumber("007", 0, 10));
  EXPECT_TRUE(IsPlainNumber("18446744073709551615", 0, kTop));
  for (const char* bad : {"", " 5", "5 ", "+5", "-5", "5s", "1e3", "0x5", "abc",
                          "11", "18446744073709551616", "5.0"})
    EXPECT_FALSE(IsPlainNumber(bad, 0, 10)) << bad;
  EXPECT_FALSE(IsPlainNumber(nullptr, 0, 10));
  EXPECT_FALSE(IsPlainNumber("0", 1, 10));
}

TEST(EnvReport, NumberIsReportedUnlessPlainWithTheValueInUse) {
  auto report = [] { ReportUnlessPlain("DFKV_REPORT_TEST", 1, 64, "8"); };
  for (const char* clean : {static_cast<const char*>(nullptr), "", "1", "64"})
    EXPECT_EQ(Reported(clean, report), "") << (clean ? clean : "(unset)");
  for (const char* bad : {"abc", "-1", "0", "65", "8 ", " 8", "8k", "+8",
                          "99999999999999999999999999"}) {
    const std::string log = Reported(bad, report);
    EXPECT_NE(log.find("DFKV_REPORT_TEST='" + std::string(bad) + "'"),
              std::string::npos)
        << bad << ": " << log;
    EXPECT_NE(log.find("not a plain decimal number from 1 to 64"),
              std::string::npos)
        << log;
    EXPECT_NE(log.find("using 8, as before"), std::string::npos) << log;
    EXPECT_NE(log.find("[WARN]"), std::string::npos) << log;
  }
  // An empty value where the reader gives it a meaning.
  const std::string log = Reported("", [] {
    ReportUnlessPlain("DFKV_REPORT_TEST", 0, 64, "0 (disabled)", false);
  });
  EXPECT_NE(log.find("DFKV_REPORT_TEST=''"), std::string::npos) << log;
  EXPECT_NE(log.find("set but empty"), std::string::npos) << log;
  EXPECT_NE(log.find("using 0 (disabled), as before"), std::string::npos) << log;
}

TEST(EnvReport, SwitchIsReportedUnlessTheWordSaysWhatWasDone) {
  auto on = [] { ReportSwitch("DFKV_REPORT_TEST", true, "only 0 is off"); };
  auto off = [] { ReportSwitch("DFKV_REPORT_TEST", false, "only 1 is on"); };
  for (const char* word : {static_cast<const char*>(nullptr), "", "1", "on",
                           "true", "TRUE", "Yes"})
    EXPECT_EQ(Reported(word, on), "") << (word ? word : "(unset)");
  for (const char* word : {static_cast<const char*>(nullptr), "", "0", "off",
                           "false", "No"})
    EXPECT_EQ(Reported(word, off), "") << (word ? word : "(unset)");
  // The word says the opposite of what the reader did, or nothing at all.
  for (const char* word : {"off", "false", "NO", "0", "enable", "2", " 1"}) {
    const std::string log = Reported(word, on);
    EXPECT_NE(log.find("DFKV_REPORT_TEST='" + std::string(word) + "'"),
              std::string::npos)
        << word << ": " << log;
    EXPECT_NE(log.find("read as on, as before (only 0 is off)"),
              std::string::npos)
        << log;
  }
  for (const char* word : {"on", "true", "YES", "1", "enable", "2"}) {
    const std::string log = Reported(word, off);
    EXPECT_NE(log.find("read as off, as before (only 1 is on)"),
              std::string::npos)
        << word << ": " << log;
  }
}

TEST(EnvReport, IsMadeOncePerSettingAndProcess) {
  ForgetWarningsForTest();
  ::setenv("DFKV_REPORT_TEST", "nope", 1);
  testing::internal::CaptureStderr();
  for (int i = 0; i < 50; ++i) {
    ReportUnlessPlain("DFKV_REPORT_TEST", 1, 64, "8");
    ReportSwitch("DFKV_REPORT_TEST", true, "only 0 is off");
  }
  const std::string log = testing::internal::GetCapturedStderr();
  ::unsetenv("DFKV_REPORT_TEST");
  size_t lines = 0;
  for (size_t at = log.find("DFKV_REPORT_TEST='nope'"); at != std::string::npos;
       at = log.find("DFKV_REPORT_TEST='nope'", at + 1))
    ++lines;
  EXPECT_EQ(lines, 1u) << log;
}
