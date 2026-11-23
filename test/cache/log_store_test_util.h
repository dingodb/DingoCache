// Helpers shared by the LogStore tests of the later stages (sealing,
// cleaning, retention): small stores, deterministic values, and crash
// points staged in a forked child.
#ifndef DFKV_TEST_CACHE_LOG_STORE_TEST_UTIL_H_
#define DFKV_TEST_CACHE_LOG_STORE_TEST_UTIL_H_

#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

#include "cache/log_store.h"

namespace lse_test {

namespace fs = std::filesystem;
using dfkv::BlockKey;
using dfkv::LogStore;
using dfkv::Status;

inline constexpr uint8_t kMiss = 0, kHitUnprotected = 1, kProtectedNew = 2,
                  kRefreshed = 3, kUpgraded = 4, kUnchanged = 5, kDenied = 6;
inline constexpr uint64_t kSmall = 32ull << 20;  // 8 segments of 4 MiB
inline constexpr uint32_t kT0 = 1'800'000'000u;

inline fs::path TestDir(const std::string& tag) {
  const fs::path p =
      fs::temp_directory_path() /
      ("dfkv_lsec_" + tag + "_" +
       std::to_string(::testing::UnitTest::GetInstance()
                          ->current_test_info()->line()));
  fs::remove_all(p);
  return p;
}

inline LogStore::Options Opts(const fs::path& dir, uint64_t capacity = kSmall) {
  LogStore::Options o;
  o.dir = dir.string();
  o.capacity_bytes = capacity;
  return o;
}
// For opens that check what came back (the same as Opts until the engine
// has a cleaner).
inline LogStore::Options Quiet(const fs::path& dir, uint64_t capacity = kSmall) {
  return Opts(dir, capacity);
}

inline BlockKey Key(uint64_t id) { return BlockKey{id * 7 + 3, id, id % 2}; }

inline std::string Val(uint64_t id, size_t len, uint32_t version = 0) {
  std::string v(len, '\0');
  uint64_t x = id * 0x9e3779b97f4a7c15ULL + version * 0x51ULL + 1;
  for (size_t i = 0; i < len; ++i) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    v[i] = static_cast<char>(x);
  }
  return v;
}

inline bool Has(LogStore& s, uint64_t id, size_t len, uint32_t version = 0) {
  std::string out;
  return s.Range(Key(id), 0, 0, &out) == Status::kOk &&
         out == Val(id, len, version);
}

// Runs `body` in a child process and returns its exit code (-1 if it did
// not exit normally). A crash hook in the child may _exit(0) at its point.
inline int InChild(const std::function<int()>& body) {
  const pid_t child = ::fork();
  if (child < 0) return -1;
  if (child == 0) ::_exit(body());
  int status = 0;
  if (::waitpid(child, &status, 0) != child || !WIFEXITED(status)) return -1;
  return WEXITSTATUS(status);
}

// A hook that _exits at the nth time `point` is reached.
inline std::function<void(const char*)> ExitAt(const char* point, int nth = 1) {
  auto seen = std::make_shared<int>(0);
  return [point, nth, seen](const char* p) {
    if (std::strcmp(p, point) == 0 && ++*seen == nth) ::_exit(0);
  };
}

}  // namespace lse_test

#endif  // DFKV_TEST_CACHE_LOG_STORE_TEST_UTIL_H_
