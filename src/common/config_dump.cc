#include "common/config_dump.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "utils/env_parse.h"
#include "utils/log.h"

extern char** environ;

namespace dfkv {
namespace config_dump {
namespace {

struct Entry {
  std::string value;
  Source src;
};

std::mutex g_mu;
// Keyed + sorted by name; last write wins so a re-resolved knob does not
// duplicate. Ordered map => stable, alphabetical Emit output.
std::map<std::string, Entry> g_entries;

const char* SourceLabel(Source s) {
  switch (s) {
    case Source::kFlag: return "flag";
    case Source::kEnv: return "env";
    case Source::kArg: return "arg";
    case Source::kDefault: return "default";
  }
  return "?";
}

// Only DFKV_-namespaced vars are dfkv knobs; other env (PATH, etc.) is noise.
bool IsDfkvEnv(const char* kv) { return std::strncmp(kv, "DFKV_", 5) == 0; }

}  // namespace

void Record(const std::string& name, const std::string& value, Source src) {
  std::lock_guard<std::mutex> lk(g_mu);
  g_entries[name] = Entry{value, src};
}

void RecordResolved(const std::string& name, const std::string& value) {
  Record(name, value,
         std::getenv(name.c_str()) ? Source::kEnv : Source::kDefault);
}

std::string EnvStr(const char* name, const std::string& def) {
  const char* e = std::getenv(name);
  bool set = (e != nullptr && *e != '\0');
  std::string val = set ? std::string(e) : def;
  Record(name, val, set ? Source::kEnv : Source::kDefault);
  return val;
}

// How these helpers always read a value is kept (utils/env_parse.h): a
// leading number is used and the rest ignored, a negative unsigned wraps (the
// callers clamp), and every word but 0/false/no/off is on. What is new is the
// WARN for a value that is not clean.
namespace {
bool IsSet(const char* name) {
  const char* e = std::getenv(name);
  return e != nullptr && *e != '\0';
}
}  // namespace

int64_t EnvI64(const char* name, int64_t def) {
  // The one signed reader. Its strtoll stays as it was; what is added is the
  // check whether the text is a plain number, and the report if it is not.
  const char* e = std::getenv(name);
  int64_t val = def;
  Source src = Source::kDefault;
  if (e != nullptr && *e != '\0') {
    errno = 0;
    char* end = nullptr;
    const long long parsed = std::strtoll(e, &end, 10);
    const bool converted = end != e;
    const bool out_of_range = errno == ERANGE;
    if (converted) { val = static_cast<int64_t>(parsed); src = Source::kEnv; }
    const char* p = e;
    if (*p == '-') ++p;
    bool clean = converted && !out_of_range && *end == '\0' &&
                 *p >= '0' && *p <= '9';
    if (!clean) {
      envparse::WarnOnce(
          name,
          "invalid " + envparse::detail::Quoted(name, e) +
              ": only a plain decimal number is valid; using " +
              (converted ? std::to_string(val) + " as before. A later "
                           "release will not accept this value."
                         : "the default " + std::to_string(def)));
    }
  }
  Record(name, std::to_string(val), src);
  return val;
}

uint64_t EnvU64(const char* name, uint64_t def) {
  envparse::U64Spec spec;
  spec.fallback = def;
  spec.above = envparse::Above::kClamp;  // strtoull saturates
  spec.legacy = envparse::Legacy::kPrefix;
  spec.negative = envparse::Negative::kWraps;
  bool from_env = false;
  const uint64_t val = envparse::EnvU64(name, spec, &from_env);
  Record(name, std::to_string(val),
         from_env ? Source::kEnv : Source::kDefault);
  return val;
}

bool EnvBool(const char* name, bool def) {
  const bool val =
      envparse::EnvSwitch(name, def, envparse::OffWhen::kOffWord);
  Record(name, val ? "1" : "0", IsSet(name) ? Source::kEnv : Source::kDefault);
  return val;
}

void Emit(const std::string& section) {
  std::lock_guard<std::mutex> lk(g_mu);
  // Fold in DFKV_* env vars that no component recorded, so a set-but-unwired
  // knob is still visible (marked env). Recorded entries are authoritative.
  for (char** ep = environ; ep && *ep; ++ep) {
    if (!IsDfkvEnv(*ep)) continue;
    const char* eq = std::strchr(*ep, '=');
    if (!eq) continue;
    std::string name(*ep, eq - *ep);
    if (name == "DFKV_LOG") continue;  // logger level, echoed by the logger
    if (g_entries.find(name) == g_entries.end())
      g_entries[name] = Entry{std::string(eq + 1), Source::kEnv};
  }
  if (g_entries.empty()) return;
  DFKV_LOG_INFO("effective config (" + section + "): " +
                std::to_string(g_entries.size()) + " parameter(s)");
  size_t width = 0;
  for (const auto& kv : g_entries) width = std::max(width, kv.first.size());
  for (const auto& kv : g_entries) {
    std::string pad(width - kv.first.size(), ' ');
    DFKV_LOG_INFO("  " + kv.first + pad + " = " + kv.second.value +
                  "  (" + SourceLabel(kv.second.src) + ")");
  }
  g_entries.clear();
}

void ResetForTest() {
  std::lock_guard<std::mutex> lk(g_mu);
  g_entries.clear();
}

}  // namespace config_dump
}  // namespace dfkv
