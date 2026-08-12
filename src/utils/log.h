/* Tiny leveled logger (header-only). Level via env DFKV_LOG=debug|info|warn|error
 * (default info). Writes timestamped lines to stderr. Not on the hot path. */
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

inline Level CurrentLevel() {
  static Level lvl = [] {
    const char* e = std::getenv("DFKV_LOG");
    if (!e) return kInfo;
    if (!std::strcmp(e, "debug")) return kDebug;
    if (!std::strcmp(e, "warn")) return kWarn;
    if (!std::strcmp(e, "error")) return kError;
    return kInfo;
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

#define DFKV_LOG_DEBUG(msg) ::dfkv::logging::Emit(::dfkv::logging::kDebug, "DEBUG", (msg))
#define DFKV_LOG_INFO(msg) ::dfkv::logging::Emit(::dfkv::logging::kInfo, "INFO", (msg))
#define DFKV_LOG_WARN(msg) ::dfkv::logging::Emit(::dfkv::logging::kWarn, "WARN", (msg))
#define DFKV_LOG_ERROR(msg) ::dfkv::logging::Emit(::dfkv::logging::kError, "ERROR", (msg))

}  // namespace dfkv

#endif  // DFKV_LOG_H_
