// LogThrottle bounds a failure log site: one line per key per window, a fixed
// number of keys per window, and an honest count of what was left out.
#include "utils/log_throttle.h"

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

using namespace dfkv;  // NOLINT

namespace {
constexpr uint64_t kWindowUs = 1000;
}  // namespace

TEST(LogThrottle, AdmitsEachKeyOncePerWindowAndReportsWhatItSwallowed) {
  LogThrottle throttle(kWindowUs);
  uint64_t suppressed = 99;
  ASSERT_TRUE(throttle.Admit(7, 100, &suppressed));
  EXPECT_EQ(suppressed, 0u);
  // A storm from the same source inside the window stays silent.
  for (uint64_t t = 101; t < 400; ++t) {
    suppressed = 99;
    EXPECT_FALSE(throttle.Admit(7, t, &suppressed));
    EXPECT_EQ(suppressed, 99u) << "untouched when nothing is logged";
  }
  // A different source is not hidden by the noisy one, and its line says how
  // many events were dropped since the previous line.
  ASSERT_TRUE(throttle.Admit(8, 400, &suppressed));
  EXPECT_EQ(suppressed, 299u);
  // The next window admits the first source again; nothing was dropped since.
  ASSERT_TRUE(throttle.Admit(7, 100 + kWindowUs, &suppressed));
  EXPECT_EQ(suppressed, 0u);
}

TEST(LogThrottle, CapsDistinctKeysPerWindow) {
  LogThrottle throttle(kWindowUs);
  uint64_t suppressed = 0;
  size_t admitted = 0;
  for (uint64_t key = 0; key < 10 * LogThrottle::kMaxKeys; ++key)
    admitted += throttle.Admit(key, 50, &suppressed) ? 1 : 0;
  EXPECT_EQ(admitted, LogThrottle::kMaxKeys)
      << "an unbounded key space cannot raise the line rate";
  ASSERT_TRUE(throttle.Admit(12345, 50 + kWindowUs, &suppressed));
  EXPECT_EQ(suppressed, 9 * LogThrottle::kMaxKeys);
}

TEST(LogThrottle, AnOlderTimestampDoesNotReopenTheWindow) {
  // Callers read the clock before taking the lock, so timestamps can arrive
  // slightly out of order; that must not look like an elapsed window.
  LogThrottle throttle(kWindowUs);
  uint64_t suppressed = 0;
  ASSERT_TRUE(throttle.Admit(1, 500, &suppressed));
  EXPECT_FALSE(throttle.Admit(1, 499, &suppressed));
  EXPECT_FALSE(throttle.Admit(1, 500 + kWindowUs - 1, &suppressed));
  EXPECT_TRUE(throttle.Admit(1, 500 + kWindowUs, &suppressed));
}

TEST(LogThrottle, KeySeparatesEveryPart) {
  const uint64_t base = LogThrottle::Key("10.0.0.1", 0, 12);
  EXPECT_EQ(base, LogThrottle::Key("10.0.0.1", 0, 12));
  EXPECT_NE(base, LogThrottle::Key("10.0.0.2", 0, 12));
  EXPECT_NE(base, LogThrottle::Key("10.0.0.1", 1, 12));
  EXPECT_NE(base, LogThrottle::Key("10.0.0.1", 0, 5));
  EXPECT_NE(LogThrottle::Key("a", 1, 2), LogThrottle::Key("a", 2, 1));
}

TEST(LogThrottle, ConcurrentStormAdmitsOneLineAndCountsTheRest) {
  LogThrottle throttle(kWindowUs);
  constexpr int kThreads = 8;
  constexpr int kEvents = 2000;
  std::atomic<int> admitted{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&] {
      for (int event = 0; event < kEvents; ++event) {
        uint64_t suppressed = 0;
        if (throttle.Admit(42, 10, &suppressed)) admitted.fetch_add(1);
      }
    });
  }
  for (auto& thread : threads) thread.join();
  EXPECT_EQ(admitted.load(), 1);
  uint64_t suppressed = 0;
  ASSERT_TRUE(throttle.Admit(43, 10, &suppressed));
  EXPECT_EQ(suppressed, static_cast<uint64_t>(kThreads) * kEvents - 1);
}
