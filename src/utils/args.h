/* Strict CLI argument parser for the daemons/tools. Replaces the ad-hoc
 * `for (i=1; i+1<argc; i+=2)` loops, which silently ignored unknown flags
 * (a typo'd --capp fell back to the 1 GiB default -> eviction storm), desynced
 * on a valueless flag (eating the next flag as its value), and used atoi/strtoull
 * that accept trailing garbage (--cap 5TiB parsed as 5). Every parse error here
 * is fatal-by-design: the caller prints usage and exit(2), so a misconfigured
 * unit fails loudly at startup instead of running with silent wrong values.
 *
 * Header-only, no deps, unit-tested directly (see test/utils/args_test.cc). */
#ifndef DFKV_ARGS_H_
#define DFKV_ARGS_H_

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>

namespace dfkv {

class Args {
 public:
  // valued = flags that consume the next token ("--port" -> "8080");
  // boolean = flags that stand alone ("--foo"). A leading token that is neither
  // (and starts with '-') is an unknown-flag error; --version/-V/--help/-h are
  // always accepted (the caller handles them before constructing Args, but
  // tolerating them here keeps parsing order-independent).
  Args(int argc, char** argv, std::set<std::string> valued,
       std::set<std::string> boolean = {})
      : valued_(std::move(valued)), boolean_(std::move(boolean)) {
    static const std::set<std::string> kAlways = {"--version", "-V", "--help", "-h"};
    for (int i = 1; i < argc; ++i) {
      std::string tok = argv[i];
      if (kAlways.count(tok)) { flags_.insert(tok); continue; }
      if (tok.rfind("--", 0) != 0) {
        Fail("unexpected argument: " + tok);
        return;
      }
      if (boolean_.count(tok)) { flags_.insert(tok); continue; }
      if (valued_.count(tok)) {
        if (i + 1 >= argc) { Fail("missing value for " + tok); return; }
        values_[tok] = argv[++i];
        continue;
      }
      Fail("unknown flag: " + tok);
      return;
    }
  }

  bool ok() const { return ok_; }
  const std::string& error() const { return error_; }

  bool Has(const std::string& flag) const {
    return flags_.count(flag) || values_.count(flag);
  }
  std::string Get(const std::string& flag, const std::string& dflt) const {
    auto it = values_.find(flag);
    return it == values_.end() ? dflt : it->second;
  }

  // Typed getters. On a present-but-malformed value they set the parser to
  // not-ok (so the caller's single `if (!args.ok())` catches it) and return dflt.
  int GetInt(const std::string& flag, int dflt) {
    auto it = values_.find(flag);
    if (it == values_.end()) return dflt;
    errno = 0;
    char* end = nullptr;
    long v = std::strtol(it->second.c_str(), &end, 10);
    // Out of int range is an error too: the cast below used to wrap, so that
    // 4294967297 was read as 1.
    if (errno != 0 || end == it->second.c_str() || *end != '\0' ||
        v < std::numeric_limits<int>::min() ||
        v > std::numeric_limits<int>::max()) {
      Fail("invalid integer for " + flag + ": '" + it->second + "'");
      return dflt;
    }
    return static_cast<int>(v);
  }
  // A TCP port: 0..65535 (0 = let the system choose, where the caller allows
  // it). A port outside that range used to be truncated to 16 bits by the
  // socket call: --port 70000 listened on 4464.
  int GetPort(const std::string& flag, int dflt) {
    const int v = GetInt(flag, dflt);
    // The caller's default (-1 = "off" for the optional listeners) may also be
    // written out.
    if (values_.count(flag) && ok_ && v != dflt && (v < 0 || v > 65535)) {
      Fail("invalid port for " + flag + " (0..65535): '" + values_[flag] + "'");
      return dflt;
    }
    return v;
  }
  uint64_t GetU64(const std::string& flag, uint64_t dflt) {
    auto it = values_.find(flag);
    if (it == values_.end()) return dflt;
    errno = 0;
    char* end = nullptr;
    unsigned long long v = std::strtoull(it->second.c_str(), &end, 10);
    // strtoull accepts a sign and negates: "-1" used to be read as 2^64-1.
    if (errno != 0 || end == it->second.c_str() || *end != '\0' ||
        it->second.find('-') != std::string::npos) {
      Fail("invalid number for " + flag + " (expects bytes, decimal): '" +
           it->second + "'");
      return dflt;
    }
    return static_cast<uint64_t>(v);
  }

 private:
  void Fail(const std::string& msg) {
    if (ok_) { ok_ = false; error_ = msg; }  // keep the first error
  }
  std::set<std::string> valued_, boolean_;
  std::unordered_map<std::string, std::string> values_;
  std::set<std::string> flags_;
  bool ok_ = true;
  std::string error_;
};

// "host:port" validity: exactly one ':' with a non-empty host and a port in
// 1..65535 and no trailing garbage. Used to reject a mis-typed --advertise
// (whose rfind(':')==npos used to wrap to a garbage port).
inline bool IsValidHostPort(const std::string& s) {
  auto pos = s.rfind(':');
  if (pos == std::string::npos || pos == 0 || pos + 1 >= s.size()) return false;
  const std::string port = s.substr(pos + 1);
  errno = 0;
  char* end = nullptr;
  long p = std::strtol(port.c_str(), &end, 10);
  return errno == 0 && *end == '\0' && p >= 1 && p <= 65535;
}

}  // namespace dfkv

#endif  // DFKV_ARGS_H_
