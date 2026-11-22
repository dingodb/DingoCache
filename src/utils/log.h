/* Tiny leveled logger (header-only). Level via env DFKV_LOG=debug|info|warn|error
 * in any case (default info; "warning" is accepted for warn). Writes
 * timestamped lines to stderr. Not on the hot path. */
#ifndef DFKV_LOG_H_
#define DFKV_LOG_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>

namespace dfkv {
namespace logging {

enum Level { kDebug = 0, kInfo = 1, kWarn = 2, kError = 3 };

// The level a DFKV_LOG value names, in any case. Unset or empty is info.
// *known is false for a word that is not a level; the result is then info.
// Up to 2.38 only lower case was matched, so that "WARN", which --help
// documents, was read as info.
inline Level ParseLevel(const char* text, bool* known) {
  *known = true;
  if (text == nullptr) return kInfo;
  std::string word;
  for (const char* p = text; *p; ++p)
    if (*p != ' ' && *p != '\t')
      word.push_back(*p >= 'A' && *p <= 'Z' ? static_cast<char>(*p - 'A' + 'a')
                                            : *p);
  if (word.empty() || word == "info") return kInfo;
  if (word == "debug") return kDebug;
  if (word == "warn" || word == "warning") return kWarn;
  if (word == "error") return kError;
  *known = false;
  return kInfo;
}

inline Level CurrentLevel() {
  static Level lvl = [] {
    const char* e = std::getenv("DFKV_LOG");
    bool known = true;
    const Level level = ParseLevel(e, &known);
    if (!known)
      std::fprintf(stderr,
                   "dfkv: [WARN] invalid DFKV_LOG='%s': expected debug, info, "
                   "warn or error; using info\n",
                   e);
    return level;
  }();
  return lvl;
}

inline void Emit(Level lvl, const char* tag, const std::string& msg) {
  if (lvl < CurrentLevel()) return;
  static std::mutex mu;
  char ts[32];
  std::time_t t = std::time(nullptr);
  std::tm tm{};
  localtime_r(&t, &tm);
  std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm);
  std::lock_guard<std::mutex> lk(mu);
  std::fprintf(stderr, "%s [%s] dfkv: %s\n", ts, tag, msg.c_str());
}

}  // namespace logging

#define DFKV_LOG_INFO(msg) ::dfkv::logging::Emit(::dfkv::logging::kInfo, "INFO", (msg))
#define DFKV_LOG_WARN(msg) ::dfkv::logging::Emit(::dfkv::logging::kWarn, "WARN", (msg))
#define DFKV_LOG_ERROR(msg) ::dfkv::logging::Emit(::dfkv::logging::kError, "ERROR", (msg))

}  // namespace dfkv

#endif  // DFKV_LOG_H_
