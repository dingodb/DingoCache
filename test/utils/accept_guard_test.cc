#include "utils/accept_guard.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <thread>

using namespace dfkv;  // NOLINT

TEST(AcceptGuard, OnlyResourceErrorsCountAndWait) {
  net::AcceptGuard guard("test listener");
  std::atomic<bool> running{true};

  // Errors of the one connection: no count, no wait.
  const auto started = std::chrono::steady_clock::now();
  for (int err : {EINTR, ECONNABORTED, EAGAIN, EPROTO, ENETDOWN, EHOSTUNREACH})
    guard.OnAcceptError(err, running);
  EXPECT_LT(std::chrono::steady_clock::now() - started,
            std::chrono::milliseconds(net::AcceptGuard::kBackoffMs));
  EXPECT_EQ(guard.accept_failures(), 0u);

  // Exhaustion: counted, and the caller is held back.
  uint64_t expected = 0;
  for (int err : {EMFILE, ENFILE, ENOBUFS, ENOMEM}) {
    const auto before = std::chrono::steady_clock::now();
    guard.OnAcceptError(err, running);
    EXPECT_GE(std::chrono::steady_clock::now() - before,
              std::chrono::milliseconds(net::AcceptGuard::kBackoffMs - 10));
    EXPECT_EQ(guard.accept_failures(), ++expected);
  }
}

TEST(AcceptGuard, StoppedListenerDoesNotWait) {
  net::AcceptGuard guard("test listener");
  std::atomic<bool> running{false};
  const auto before = std::chrono::steady_clock::now();
  guard.OnAcceptError(EMFILE, running);
  EXPECT_LT(std::chrono::steady_clock::now() - before,
            std::chrono::milliseconds(net::AcceptGuard::kBackoffMs / 2));
}

TEST(AcceptGuard, StartThreadReportsFailureInsteadOfThrowing) {
  std::atomic<int> ran{0};
  std::thread thread;
  net::g_fail_thread_starts_for_test.store(1);
  EXPECT_FALSE(net::StartThread(&thread, [&] { ++ran; }));
  EXPECT_FALSE(thread.joinable());
  // The seam is used up; the next start is a real one.
  EXPECT_TRUE(net::StartThread(&thread, [&] { ++ran; }));
  thread.join();
  EXPECT_EQ(ran.load(), 1);
}
