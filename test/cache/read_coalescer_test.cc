// ReadCoalescer: sync leader/follower collapse plus the async-flight protocol
// used by the io_uring serve path (TryRegisterAsync / CompleteAsync). Hermetic:
// "disk reads" are lambdas. Covers: async register -> follower join -> complete
// hands the payload; fan-in evidence (waiters) reporting; abort and failure ->
// followers fall back to their own read; waiter timeout fallback; same-thread
// pipelined duplicate never deadlocks; duplicate registration declines.
#include "cache/read_coalescer.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using dfkv::BlockKey;
using dfkv::ReadCoalescer;
using dfkv::Status;
using namespace std::chrono_literals;

namespace {

BlockKey K(uint64_t id) { return BlockKey{id, 0}; }

// The wait bound is latched on first use (function-local static); pin it to a
// test-friendly value before any test can latch the 500ms default.
[[maybe_unused]] const bool kEnvSet = [] {
  ::setenv("DFKV_READ_COALESCE_TIMEOUT_MS", "200", 1);
  ::setenv("DFKV_READ_COALESCE_RECUR_MS", "200", 1);  // recurrence-window tests
  return true;
}();

TEST(ReadCoalescer, AsyncFlightHandsPayloadToFollower) {
  ReadCoalescer c;
  const std::string v = "payload-bytes";
  uint64_t t = c.TryRegisterAsync(K(1), 0, v.size(), /*whole_value=*/true);
  ASSERT_NE(t, 0u);
  ASSERT_TRUE(c.InFlight(K(1), 0, v.size()));

  std::atomic<int> own_reads{0};
  std::string got(v.size(), '\0');
  size_t got_len = 0;
  std::thread follower([&] {
    Status st = c.Read(K(1), 0, v.size(), &got[0], got.size(), &got_len,
                       [&](char* buf, size_t, size_t* n) {
                         own_reads.fetch_add(1);
                         std::memcpy(buf, v.data(), v.size());
                         *n = v.size();
                         return Status::kOk;
                       });
    EXPECT_EQ(st, Status::kOk);
  });
  // Give the follower time to join the flight, then complete it.
  std::this_thread::sleep_for(50ms);
  BlockKey key{};
  bool whole = false;
  EXPECT_TRUE(c.CompleteAsync(t, Status::kOk, v.data(), v.size(), &key, &whole));
  follower.join();
  EXPECT_EQ(own_reads.load(), 0);  // served from the flight, no duplicate read
  EXPECT_EQ(got, v);
  EXPECT_EQ(got_len, v.size());
  EXPECT_EQ(key.digest_hi, 1u);
  EXPECT_TRUE(whole);
  EXPECT_EQ(c.coalesced(), 1u);
}

TEST(ReadCoalescer, CompleteWithoutWaitersReportsNoFanIn) {
  ReadCoalescer c;
  uint64_t t = c.TryRegisterAsync(K(2), 0, 8, false);
  ASSERT_NE(t, 0u);
  EXPECT_FALSE(c.CompleteAsync(t, Status::kOk, "abcdefgh", 8));
  EXPECT_FALSE(c.InFlight(K(2), 0, 8));       // flight erased
  EXPECT_FALSE(c.CompleteAsync(t, Status::kOk, "abcdefgh", 8));  // idempotent
}

TEST(ReadCoalescer, DuplicateRegistrationDeclines) {
  ReadCoalescer c;
  uint64_t t1 = c.TryRegisterAsync(K(3), 0, 8, false);
  ASSERT_NE(t1, 0u);
  EXPECT_EQ(c.TryRegisterAsync(K(3), 0, 8, false), 0u);   // identical -> decline
  EXPECT_NE(c.TryRegisterAsync(K(3), 8, 8, false), 0u);   // other range -> own flight
  c.CompleteAsync(t1, Status::kIOError, nullptr, 0);
}

TEST(ReadCoalescer, AbortedFlightFollowerFallsBackToOwnRead) {
  ReadCoalescer c;
  uint64_t t = c.TryRegisterAsync(K(4), 0, 4, false);
  ASSERT_NE(t, 0u);
  std::atomic<int> own_reads{0};
  char buf[4];
  size_t n = 0;
  std::thread follower([&] {
    Status st = c.Read(K(4), 0, 4, buf, sizeof(buf), &n,
                       [&](char* b, size_t, size_t* on) {
                         own_reads.fetch_add(1);
                         std::memcpy(b, "sane", 4);
                         *on = 4;
                         return Status::kOk;
                       });
    EXPECT_EQ(st, Status::kOk);  // fallback read succeeded despite the abort
  });
  std::this_thread::sleep_for(50ms);
  c.CompleteAsync(t, Status::kIOError, nullptr, 0);  // teardown-path abort
  follower.join();
  EXPECT_EQ(own_reads.load(), 1);
  EXPECT_EQ(std::string(buf, 4), "sane");
}

TEST(ReadCoalescer, WaiterTimeoutFallsBackToOwnRead) {
  // The flight is NEVER completed: an abandoned flight (leader connection
  // died without reaching complete/abort) must not hang its waiters.
  ReadCoalescer c;
  uint64_t t = c.TryRegisterAsync(K(5), 0, 4, false);
  ASSERT_NE(t, 0u);
  std::atomic<int> own_reads{0};
  char buf[4];
  size_t n = 0;
  std::thread follower([&] {
    Status st = c.Read(K(5), 0, 4, buf, sizeof(buf), &n,
                       [&](char* b, size_t, size_t* on) {
                         own_reads.fetch_add(1);
                         std::memcpy(b, "wake", 4);
                         *on = 4;
                         return Status::kOk;
                       });
    EXPECT_EQ(st, Status::kOk);
  });
  follower.join();  // returns after the latched timeout, via its own read
  EXPECT_EQ(own_reads.load(), 1);
  EXPECT_EQ(c.timeouts(), 1u);
  (void)t;  // never completed on purpose: an abandoned flight must not hang waiters
}

TEST(ReadCoalescer, SameThreadDuplicateReadsItselfImmediately) {
  ReadCoalescer c;
  // Register on THIS thread (as the async prep does), then issue the duplicate
  // GET's sync fallback on the SAME thread: it must not wait (the "leader"
  // read would only run after this call returned -> deadlock-until-timeout).
  uint64_t t = c.TryRegisterAsync(K(6), 0, 4, false);
  ASSERT_NE(t, 0u);
  char buf[4];
  size_t n = 0;
  const auto t0 = std::chrono::steady_clock::now();
  Status st = c.Read(K(6), 0, 4, buf, sizeof(buf), &n,
                     [&](char* b, size_t, size_t* on) {
                       std::memcpy(b, "self", 4);
                       *on = 4;
                       return Status::kOk;
                     });
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0).count();
  EXPECT_EQ(st, Status::kOk);
  EXPECT_EQ(std::string(buf, 4), "self");
  EXPECT_LT(ms, 100);  // no wait: read executed inline
  c.CompleteAsync(t, Status::kIOError, nullptr, 0);
}

TEST(ReadCoalescer, SyncLeaderReportsFanIn) {
  ReadCoalescer c;
  const std::string v(8192, 'x');
  std::atomic<int> fills{0};
  std::atomic<bool> leader_entered{false};
  ReadCoalescer::Outcome oc;
  std::string lbuf(v.size(), '\0');
  size_t ln = 0;
  std::thread leader([&] {
    c.Read(K(7), 0, v.size(), &lbuf[0], lbuf.size(), &ln,
           [&](char* b, size_t, size_t* on) {
             leader_entered.store(true);
             std::this_thread::sleep_for(100ms);  // hold the flight open
             fills.fetch_add(1);
             std::memcpy(b, v.data(), v.size());
             *on = v.size();
             return Status::kOk;
           },
           &oc);
  });
  while (!leader_entered.load()) std::this_thread::sleep_for(1ms);
  std::string fbuf(v.size(), '\0');
  size_t fn = 0;
  Status st = c.Read(K(7), 0, v.size(), &fbuf[0], fbuf.size(), &fn,
                     [&](char* b, size_t, size_t* on) {
                       fills.fetch_add(1);
                       std::memcpy(b, v.data(), v.size());
                       *on = v.size();
                       return Status::kOk;
                     });
  leader.join();
  EXPECT_EQ(st, Status::kOk);
  EXPECT_EQ(fills.load(), 1);  // one disk read served both
  EXPECT_EQ(fbuf, v);
  EXPECT_EQ(lbuf, v);
  EXPECT_TRUE(oc.leader);
  EXPECT_TRUE(oc.had_waiters);
}

}  // namespace

// v2 recurrence window: a lone whole-value read leaves a key fingerprint; a
// re-read within DFKV_READ_COALESCE_RECUR_MS reports recurrence evidence on
// completion (the drift-tolerant promotion gate). Env pinned to 200ms below.
TEST(ReadCoalescer, RecurrenceTombstoneReportsOnReRead) {
  ReadCoalescer c;
  const char v[8] = "payload";
  uint64_t t1 = c.TryRegisterAsync(K(20), 0, 8, /*whole=*/true);
  ASSERT_NE(t1, 0u);
  EXPECT_FALSE(c.CompleteAsync(t1, Status::kOk, v, 8));  // lone read: tombstone laid
  BlockKey key{};
  bool whole = false, recur = false;
  uint64_t t2 = c.TryRegisterAsync(K(20), 0, 8, true);   // re-read inside window
  ASSERT_NE(t2, 0u);
  EXPECT_EQ(c.recur_hits(), 1u);
  c.CompleteAsync(t2, Status::kOk, v, 8, &key, &whole, &recur);
  EXPECT_TRUE(recur);      // promotion evidence without any in-flight waiter
  EXPECT_TRUE(whole);
  EXPECT_EQ(key.digest_hi, 20u);
  // The hit consumed the tombstone, and a recurrent completion lays none
  // (promotion put the page in RAM): a third read reports no recurrence.
  uint64_t t3 = c.TryRegisterAsync(K(20), 0, 8, true);
  ASSERT_NE(t3, 0u);
  recur = false;
  c.CompleteAsync(t3, Status::kOk, v, 8, nullptr, nullptr, &recur);
  EXPECT_FALSE(recur);
}

TEST(ReadCoalescer, RecurrenceTombstoneExpires) {
  ReadCoalescer c;
  const char v[8] = "payload";
  uint64_t t1 = c.TryRegisterAsync(K(21), 0, 8, true);
  c.CompleteAsync(t1, Status::kOk, v, 8);
  std::this_thread::sleep_for(250ms);  // past the 200ms env-pinned window
  bool recur = true;
  uint64_t t2 = c.TryRegisterAsync(K(21), 0, 8, true);
  ASSERT_NE(t2, 0u);
  c.CompleteAsync(t2, Status::kOk, v, 8, nullptr, nullptr, &recur);
  EXPECT_FALSE(recur);
}

TEST(ReadCoalescer, NoTombstoneForPartialFailedOrFannedReads) {
  ReadCoalescer c;
  const char v[8] = "payload";
  // Partial (whole=false) completion: no tombstone.
  uint64_t t1 = c.TryRegisterAsync(K(22), 4, 4, false);
  c.CompleteAsync(t1, Status::kOk, v, 4);
  bool recur = true;
  uint64_t t2 = c.TryRegisterAsync(K(22), 4, 4, false);
  c.CompleteAsync(t2, Status::kOk, v, 4, nullptr, nullptr, &recur);
  EXPECT_FALSE(recur);
  // Failed/aborted completion: no tombstone.
  uint64_t t3 = c.TryRegisterAsync(K(23), 0, 8, true);
  c.CompleteAsync(t3, Status::kIOError, nullptr, 0);
  recur = true;
  uint64_t t4 = c.TryRegisterAsync(K(23), 0, 8, true);
  c.CompleteAsync(t4, Status::kOk, v, 8, nullptr, nullptr, &recur);
  EXPECT_FALSE(recur);
}
