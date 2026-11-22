/* One parser for the numeric and on/off settings that arrive as text: DFKV_*
 * environment variables, and the command-line flags that are passed on as
 * their environment twin.
 *
 * Up to 2.38 these were read in several ways, most of them with strtol/atoi
 * and little or no checking, so that a mistyped value was silently read as
 * something else. The rule of this parser:
 *
 *   A value never fails a start and never throws (the client library runs
 *   inside its host process), and it never changes what a setting has been
 *   resolving to: whatever a given text gave before, it still gives. What is
 *   new is that every value that is not a plain decimal number in range (or,
 *   for a switch, not one of the words the switch really knows) is reported:
 *   one WARN per setting and process, naming the setting, the value and what
 *   is used.
 *
 * So a spec says two things: the range and default, and how the setting used
 * to be read (Legacy, Negative, ...), which decides what an unclean value
 * resolves to. New settings should use the defaults of U64Spec: full-string
 * decimal, anything else is the default.
 *
 * Header-only. The Parse* functions are pure (the warning is returned as
 * text) and unit-tested in test/utils/env_parse_test.cc; the Env* functions
 * read the environment and log. */
#ifndef DFKV_ENV_PARSE_H_
#define DFKV_ENV_PARSE_H_

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <set>
#include <string>

#include "utils/log.h"

namespace dfkv {
namespace envparse {

// 2^63-1, what strtol/atol saturate at.
constexpr uint64_t kSignedMax =
    static_cast<uint64_t>(std::numeric_limits<int64_t>::max());

// A number above the maximum: the default, or the maximum.
enum class Above { kDefault, kClamp };

// How much of the text has to be a number.
enum class Legacy {
  kStrict,        // first character a digit, nothing after the number
  kFull,          // strtoul with a full end check: leading blanks and a sign
  kPrefix,        // strtoul with "anything converted": also text after it
  kPrefixOrZero,  // strtoul/atoi unchecked: also no number at all, read as 0
};

// A negative number (never with kStrict).
enum class Negative {
  kInvalid,  // the default
  kWraps,    // strtoull's 2^64 - n, then the range like any other number
  kZero,     // 0, for a setting whose reader treated v <= 0 alike
};

struct U64Spec {
  uint64_t fallback = 0;
  uint64_t min = 0;
  uint64_t max = std::numeric_limits<uint64_t>::max();
  Above above = Above::kDefault;
  Legacy legacy = Legacy::kStrict;
  Negative negative = Negative::kInvalid;
  // The largest number the old reader's type could hold: 2^64-1 for strtoull,
  // 2^63-1 for strtol/atol. A larger one "overflows": the reader saturated at
  // this value and went on with its range check, or -- if it checked errno
  // (overflow_is_default) -- used the default.
  uint64_t representable_max = std::numeric_limits<uint64_t>::max();
  bool overflow_is_default = false;
  // The reader did not test for an empty string and so read it as 0.
  bool empty_is_zero = false;
  // What 0 means for this setting, if it is not just a small number. Added
  // to the warning when an unclean value resolves to 0.
  const char* zero_means = nullptr;
};

namespace detail {
inline bool IsBlank(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' ||
         c == '\f';
}
inline std::string Quoted(const char* name, const char* text) {
  return std::string(name) + "='" + text + "'";
}
// Leading decimal digits. *overflow when they do not fit 64 bits.
inline const char* Digits(const char* p, uint64_t* out, bool* overflow) {
  uint64_t value = 0;
  *overflow = false;
  for (; *p >= '0' && *p <= '9'; ++p) {
    const uint64_t digit = static_cast<uint64_t>(*p - '0');
    if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10)
      *overflow = true;
    else
      value = value * 10 + digit;
  }
  *out = value;
  return p;
}
inline void Note(std::string* notes, const std::string& note) {
  if (!notes->empty()) *notes += ", ";
  *notes += note;
}
}  // namespace detail

// `text` may be null (unset). `warning` is cleared, or set to the line to log.
// `from_text`, if given, says whether the result was taken from the text
// rather than being the default.
inline uint64_t ParseU64(const char* name, const char* text,
                         const U64Spec& spec, std::string* warning,
                         bool* from_text = nullptr) {
  warning->clear();
  bool unused = false;
  if (from_text == nullptr) from_text = &unused;
  *from_text = false;
  if (text == nullptr) return spec.fallback;

  auto use_default = [&](const std::string& why) {
    *warning = "invalid " + detail::Quoted(name, text) + ": " + why +
               "; using the default " + std::to_string(spec.fallback);
    return spec.fallback;
  };

  // What the text amounts to, and what is unclean about it.
  uint64_t parsed = 0;
  bool overflow = false;
  std::string notes;
  if (*text == '\0') {
    if (!spec.empty_is_zero) return spec.fallback;  // empty counts as unset
    detail::Note(&notes, "an empty value is read as 0");
  } else {
    const char* p = text;
    bool negative = false;
    if (spec.legacy != Legacy::kStrict) {
      while (detail::IsBlank(*p)) ++p;  // strtoul skips these; not worth a line
      if (*p == '+' || *p == '-') {
        negative = *p == '-';
        if (!negative) detail::Note(&notes, "the sign is ignored");
        ++p;
      }
    }
    const char* rest = detail::Digits(p, &parsed, &overflow);
    if (rest == p) {
      if (spec.legacy != Legacy::kPrefixOrZero)
        return use_default("not a non-negative decimal number");
      parsed = 0;
      negative = false;
      detail::Note(&notes, "not a number, read as 0");
    } else if (*rest != '\0') {
      if (spec.legacy == Legacy::kStrict || spec.legacy == Legacy::kFull)
        return use_default(
            "only a plain decimal number is accepted (no unit, sign or "
            "exponent)");
      detail::Note(&notes, "'" + std::string(rest) + "' after the number is "
                           "ignored");
    }
    if (negative && parsed != 0) {
      switch (spec.negative) {
        case Negative::kInvalid:
          return use_default("negative");
        case Negative::kZero:
          parsed = 0;
          overflow = false;
          detail::Note(&notes, "a negative number is read as 0");
          break;
        case Negative::kWraps:
          // As strtoull does; one that does not fit stays "too large".
          if (!overflow) parsed = ~parsed + 1;
          detail::Note(&notes, "a negative number wraps to a large one");
          break;
      }
    }
  }

  // What is unclean about the text comes first in the reason: "abc" on a
  // setting with a minimum is "not a number", not "below the minimum".
  const std::string unclean = notes.empty() ? std::string() : notes + "; ";
  if (overflow || parsed > spec.representable_max) {
    if (spec.overflow_is_default)
      return use_default(unclean + "too large a number");
    parsed = spec.representable_max;  // the reader's type saturated
    detail::Note(&notes, "too large a number, read as " +
                         std::to_string(parsed));
  }
  if (parsed > spec.max) {
    if (spec.above == Above::kDefault)
      return use_default(unclean + "above the maximum " +
                         std::to_string(spec.max));
    *warning = detail::Quoted(name, text) + " clamped to the maximum " +
               std::to_string(spec.max);
    *from_text = true;
    return spec.max;
  }
  if (parsed < spec.min)
    return use_default(unclean + "below the minimum " +
                       std::to_string(spec.min));
  *from_text = true;
  if (!notes.empty()) {
    *warning = "invalid " + detail::Quoted(name, text) + ": " + notes +
               "; using " + std::to_string(parsed) + " as before";
    if (parsed == 0 && spec.zero_means != nullptr)
      *warning += std::string(", which means ") + spec.zero_means +
                  " (write 0 if that is intended, a number, or leave it "
                  "unset for the default " + std::to_string(spec.fallback) +
                  ")";
    *warning += ". Only a plain decimal number is valid; a later release "
                "will not accept this value.";
  }
  return parsed;
}

// Logs `warning` unless a warning for `name` was already logged in this
// process: settings are read per disk, per tier, per connection, and one bad
// value must not be reported on every one of those.
namespace detail {
// name == nullptr forgets every setting (tests).
inline bool FirstWarning(const char* name) {
  static std::mutex mu;
  static std::set<std::string>* const warned = new std::set<std::string>();
  std::lock_guard<std::mutex> lock(mu);
  if (name == nullptr) {
    warned->clear();
    return false;
  }
  return warned->insert(name).second;
}
}  // namespace detail

inline void WarnOnce(const char* name, const std::string& warning) {
  if (warning.empty() || !detail::FirstWarning(name)) return;
  DFKV_LOG_WARN(warning);
}
inline void ForgetWarningsForTest() { detail::FirstWarning(nullptr); }

inline uint64_t EnvU64(const char* name, const U64Spec& spec,
                       bool* from_env = nullptr) {
  std::string warning;
  const uint64_t value =
      ParseU64(name, std::getenv(name), spec, &warning, from_env);
  WarnOnce(name, warning);
  return value;
}

// ---- switches ---------------------------------------------------------------

// The on/off words, in any case: 1 on true yes / 0 off false no.
enum class Word { kUnset, kOn, kOff, kOther };
inline Word ParseWord(const char* text) {
  if (text == nullptr || *text == '\0') return Word::kUnset;
  std::string word;
  for (const char* p = text; *p; ++p)
    word.push_back(*p >= 'A' && *p <= 'Z' ? static_cast<char>(*p - 'A' + 'a')
                                          : *p);
  if (word == "1" || word == "on" || word == "true" || word == "yes")
    return Word::kOn;
  if (word == "0" || word == "off" || word == "false" || word == "no")
    return Word::kOff;
  return Word::kOther;
}

// Which values switch a setting off. Every other non-empty value switches it
// on -- including, for the first two, the words "off", "false" and "no".
enum class OffWhen {
  kExactlyZero,     // "0"
  kStartsWithZero,  // "0", "00", "0abc"
  kOffWord,         // 0/off/false/no in any case
};

// The value is what the setting has always resolved to. The warning says so
// whenever that is not what the text says.
inline bool ParseSwitch(const char* name, const char* text, bool fallback,
                        OffWhen off_when, std::string* warning) {
  warning->clear();
  const Word word = ParseWord(text);
  if (word == Word::kUnset) return fallback;
  bool off = false;
  const char* rule = "";
  switch (off_when) {
    case OffWhen::kExactlyZero:
      off = text[0] == '0' && text[1] == '\0';
      rule = "only 0 switches it off";
      break;
    case OffWhen::kStartsWithZero:
      off = text[0] == '0';
      rule = "only a value starting with 0 switches it off";
      break;
    case OffWhen::kOffWord:
      off = word == Word::kOff;
      rule = "only 0, off, false and no switch it off";
      break;
  }
  const bool value = !off;
  const bool says_so = (value && word == Word::kOn) ||
                       (!value && word == Word::kOff);
  if (!says_so) {
    *warning = "invalid " + detail::Quoted(name, text) + ": read as " +
               (value ? "on" : "off") + ", as before (" + rule +
               "). Write 1 or 0; a later release will read on/off words by "
               "their meaning and not accept other values.";
  }
  return value;
}

inline bool EnvSwitch(const char* name, bool fallback, OffWhen off_when) {
  std::string warning;
  const bool value =
      ParseSwitch(name, std::getenv(name), fallback, off_when, &warning);
  WarnOnce(name, warning);
  return value;
}

// ---- report only ------------------------------------------------------------
//
// For a reader that keeps its own parse (the client library and the
// transports, whose readers differ in small ways that are not worth a spec
// each): the value is whatever that reader resolved, untouched. These only
// say, once per setting and process, that the text is not a clean one.

// A plain decimal number, nothing else, in [min, max].
inline bool IsPlainNumber(const char* text, uint64_t min, uint64_t max) {
  if (text == nullptr || *text < '0' || *text > '9') return false;
  uint64_t value = 0;
  bool overflow = false;
  const char* rest = detail::Digits(text, &value, &overflow);
  return *rest == '\0' && !overflow && value >= min && value <= max;
}

// `used` is what the reader resolved, as it should be shown. An empty value
// counts as unset unless the reader gives it a meaning of its own.
inline void ReportUnlessPlain(const char* name, uint64_t min, uint64_t max,
                              const std::string& used,
                              bool empty_is_unset = true) {
  const char* text = std::getenv(name);
  if (text == nullptr) return;
  if (*text == '\0' && empty_is_unset) return;
  if (IsPlainNumber(text, min, max)) return;
  WarnOnce(name,
           "invalid " + detail::Quoted(name, text) + ": " +
               (*text == '\0' ? std::string("set but empty")
                              : "not a plain decimal number from " +
                                    std::to_string(min) + " to " +
                                    std::to_string(max)) +
               "; using " + used +
               ", as before. A later release will not accept this value.");
}

// `on` is what the reader resolved; `rule` says in words what it looks at.
inline void ReportSwitch(const char* name, bool on, const char* rule) {
  const char* text = std::getenv(name);
  // A word that says what the reader did with it is clean, in any case.
  const Word word = ParseWord(text);
  if (word == Word::kUnset || (word == Word::kOn && on) ||
      (word == Word::kOff && !on))
    return;
  WarnOnce(name,
           "invalid " + detail::Quoted(name, text) + ": read as " +
               (on ? "on" : "off") + ", as before (" + rule +
               "). Write 1 or 0; a later release will read on/off words by "
               "their meaning and not accept other values.");
}

}  // namespace envparse
}  // namespace dfkv

#endif  // DFKV_ENV_PARSE_H_
