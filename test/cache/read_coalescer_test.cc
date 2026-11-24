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
#include <stdexcept>
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
  uint64_t t = c.TryRegisterAsync(K(1), 0, v.size(),
                                  /*whole_value=*/true, /*value_len=*/97);
  ASSERT_NE(t, 0u);
  ASSERT_TRUE(c.InFlight(K(1), 0, v.size()));

  std::atomic<int> own_reads{0};
  std::string got(v.size(), '\0');
  size_t got_len = 0;
  size_t got_value_len = 0;
  std::thread follower([&] {
    Status st = c.Read(
        K(1), 0, v.size(), &got[0], got.size(), &got_len, &got_value_len,
        [&](char* buf, size_t, size_t* n, size_t* value_len) {
          own_reads.fetch_add(1);
          std::memcpy(buf, v.data(), v.size());
          *n = v.size();
          *value_len = 1;
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
  EXPECT_EQ(got_value_len, 97u);
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
    Status st = c.Read(K(4), 0, 4, buf, sizeof(buf), &n, nullptr,
                       [&](char* b, size_t, size_t* on, size_t*) {
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
    Status st = c.Read(K(5), 0, 4, buf, sizeof(buf), &n, nullptr,
                       [&](char* b, size_t, size_t* on, size_t*) {
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
  Status st = c.Read(K(6), 0, 4, buf, sizeof(buf), &n, nullptr,
                     [&](char* b, size_t, size_t* on, size_t*) {
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
    c.Read(K(7), 0, v.size(), &lbuf[0], lbuf.size(), &ln, nullptr,
           [&](char* b, size_t, size_t* on, size_t*) {
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
  Status st = c.Read(K(7), 0, v.size(), &fbuf[0], fbuf.size(), &fn, nullptr,
                     [&](char* b, size_t, size_t* on, size_t*) {
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

TEST(ReadCoalescer, AsyncFollowerWaitsWithoutPayloadCopy) {
  ReadCoalescer c;
  const std::string value(8192, 'z');
  const uint64_t token = c.TryRegisterAsync(
      K(8), 0, value.size(), /*whole_value=*/true, value.size(),
      /*publishes_to_ram=*/true);
  ASSERT_NE(token, 0u);

  std::atomic<bool> entered{false};
  ReadCoalescer::PublishedWait waited =
      ReadCoalescer::PublishedWait::kNotEligible;
  std::thread follower([&] {
    entered.store(true, std::memory_order_release);
    waited = c.WaitForPublished(K(8), 0, value.size());
  });
  while (!entered.load(std::memory_order_acquire))
    std::this_thread::sleep_for(1ms);
  std::this_thread::sleep_for(20ms);

  // No payload is supplied: a direct follower needs only the completion fence,
  // then obtains bytes through the separately published RAM reservation.
  EXPECT_TRUE(c.CompleteAsync(
      token, Status::kOk, nullptr, value.size()));
  follower.join();
  EXPECT_EQ(waited, ReadCoalescer::PublishedWait::kReady);
  EXPECT_EQ(c.coalesced(), 0u);
  c.RecordCoalesced();
  EXPECT_EQ(c.coalesced(), 1u);
}

TEST(ReadCoalescer, NoCopyWaitRequiresDirectRamPublication) {
  ReadCoalescer c;
  const std::string value(4096, 's');
  const uint64_t token = c.TryRegisterAsync(
      K(9), 0, value.size(), /*whole_value=*/true, value.size(),
      /*publishes_to_ram=*/false);
  ASSERT_NE(token, 0u);
  EXPECT_EQ(c.WaitForPublished(K(9), 0, value.size()),
            ReadCoalescer::PublishedWait::kNotEligible);
  EXPECT_TRUE(c.InFlight(K(9), 0, value.size()));
  EXPECT_FALSE(c.CompleteAsync(token, Status::kOk, value.data(), value.size()));
}

TEST(ReadCoalescer, DirectPublicationSkipsRecurrenceTombstone) {
  ReadCoalescer c;
  const uint64_t direct = c.TryRegisterAsync(
      K(10), 0, 4096, /*whole_value=*/true, 4096,
      /*publishes_to_ram=*/true);
  ASSERT_NE(direct, 0u);
  EXPECT_FALSE(c.CompleteAsync(direct, Status::kOk, nullptr, 4096));

  bool recurrent = true;
  const uint64_t staged =
      c.TryRegisterAsync(K(10), 0, 4096, /*whole_value=*/true);
  ASSERT_NE(staged, 0u);
  c.CompleteAsync(
      staged, Status::kOk, nullptr, 4096, nullptr, nullptr, &recurrent);
  EXPECT_FALSE(recurrent);
  EXPECT_EQ(c.recur_hits(), 0u);
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

// Two serve threads whose batches name the same keys in opposite order:
// each registers its first key (the read is submitted only after its whole
// prep), then finds the other's flight for its second key. Waiting there
// is a cycle, broken only by the follower timeout; neither may wait.
TEST(ReadCoalescer, OppositeOrderBatchesDoNotWaitForEachOther) {
  ReadCoalescer c;
  constexpr size_t kLen = 4096;
  std::atomic<int> registered{0};
  std::atomic<int> looked{0};
  std::atomic<long long> waited_ms[2] = {{0}, {0}};
  ReadCoalescer::PublishedWait result[2] = {
      ReadCoalescer::PublishedWait::kNotEligible,
      ReadCoalescer::PublishedWait::kNotEligible};
  auto serve = [&](int self, uint64_t first, uint64_t second) {
    const uint64_t token = c.TryRegisterAsync(
        K(first), 0, kLen, /*whole_value=*/true, kLen,
        /*publishes_to_ram=*/true);
    ASSERT_NE(token, 0u);
    ++registered;
    while (registered.load() < 2) std::this_thread::sleep_for(1ms);
    const auto t0 = std::chrono::steady_clock::now();
    result[self] = c.WaitForPublished(K(second), 0, kLen);
    waited_ms[self] = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    // The prep is over: this thread's read is submitted and completes, once
    // the other has looked for it.
    ++looked;
    while (looked.load() < 2) std::this_thread::sleep_for(1ms);
    c.CompleteAsync(token, Status::kOk, nullptr, kLen);
  };
  std::thread a(serve, 0, 1, 2);
  std::thread b(serve, 1, 2, 1);
  a.join();
  b.join();
  for (int i = 0; i < 2; ++i) {
    EXPECT_LT(waited_ms[i].load(), 100) << "thread " << i;
    EXPECT_EQ(result[i], ReadCoalescer::PublishedWait::kFallback);
  }
}

// The convoy case is unchanged: a thread that holds no flight of its own
// waits for the leader and is served by it.
TEST(ReadCoalescer, AFollowerWithoutFlightsStillWaits) {
  ReadCoalescer c;
  constexpr size_t kLen = 4096;
  const uint64_t a = c.TryRegisterAsync(K(11), 0, kLen, true, kLen, true);
  const uint64_t b = c.TryRegisterAsync(K(12), 0, kLen, true, kLen, true);
  ASSERT_NE(a, 0u);
  ASSERT_NE(b, 0u);
  ReadCoalescer::PublishedWait first = ReadCoalescer::PublishedWait::kNotEligible;
  ReadCoalescer::PublishedWait second = ReadCoalescer::PublishedWait::kNotEligible;
  std::atomic<bool> first_done{false};
  std::thread follower([&] {
    first = c.WaitForPublished(K(11), 0, kLen);
    first_done = true;
    second = c.WaitForPublished(K(12), 0, kLen);
  });
  std::this_thread::sleep_for(20ms);
  c.CompleteAsync(a, Status::kOk, nullptr, kLen);
  while (!first_done.load()) std::this_thread::sleep_for(1ms);
  std::this_thread::sleep_for(20ms);
  c.CompleteAsync(b, Status::kOk, nullptr, kLen);
  follower.join();
  EXPECT_EQ(first, ReadCoalescer::PublishedWait::kReady);
  EXPECT_EQ(second, ReadCoalescer::PublishedWait::kReady);
  EXPECT_EQ(c.held_fallbacks(), 0u);
  // Once its flights are completed, the leader's thread may wait again.
  const uint64_t d = c.TryRegisterAsync(K(13), 0, kLen, true, kLen, true);
  std::thread other([&] {
    const uint64_t own = c.TryRegisterAsync(K(14), 0, kLen, true, kLen, true);
    c.CompleteAsync(own, Status::kOk, nullptr, kLen);
    EXPECT_EQ(c.WaitForPublished(K(13), 0, kLen),
              ReadCoalescer::PublishedWait::kReady);
  });
  std::this_thread::sleep_for(20ms);
  c.CompleteAsync(d, Status::kOk, nullptr, kLen);
  other.join();
}

// An aborted flight (CompleteAsync without data, as PreparedRead's abort
// does, also from another thread) no longer counts as held: its thread
// waits for other flights again.
TEST(ReadCoalescer, AnAbortedFlightReleasesItsThread) {
  ReadCoalescer c;
  constexpr size_t kLen = 4096;
  uint64_t own = 0;
  std::thread leader([&] {
    own = c.TryRegisterAsync(K(21), 0, kLen, true, kLen, true);
  });
  leader.join();
  ASSERT_NE(own, 0u);
  // Aborted here, not on the thread that registered it.
  c.CompleteAsync(own, Status::kIOError, nullptr, 0);
  const uint64_t mine = c.TryRegisterAsync(K(22), 0, kLen, true, kLen, true);
  ASSERT_NE(mine, 0u);
  c.CompleteAsync(mine, Status::kIOError, nullptr, 0);  // aborted on its own thread
  const uint64_t other = c.TryRegisterAsync(K(23), 0, kLen, true, kLen, true);
  std::thread waiter([&] {
    // This thread registered nothing.
    EXPECT_EQ(c.WaitForPublished(K(23), 0, kLen),
              ReadCoalescer::PublishedWait::kReady);
  });
  std::this_thread::sleep_for(20ms);
  c.CompleteAsync(other, Status::kOk, nullptr, kLen);
  waiter.join();
  // The main thread's flights are all completed: it may wait again.
  std::thread registrar([&] {
    const uint64_t t = c.TryRegisterAsync(K(24), 0, kLen, true, kLen, true);
    std::this_thread::sleep_for(20ms);
    c.CompleteAsync(t, Status::kOk, nullptr, kLen);
  });
  std::this_thread::sleep_for(5ms);
  EXPECT_EQ(c.WaitForPublished(K(24), 0, kLen),
            ReadCoalescer::PublishedWait::kReady);
  registrar.join();
  EXPECT_EQ(c.held_fallbacks(), 0u);
}

// REQ-20: a leader with no followers reads straight into the caller's buffer.
// It read into a scratch it allocated (posix_memalign of the whole capacity)
// and copied it over: measured 597 us extra on a 1 MiB read, 2.5 ms on 4 MiB,
// for a fill of 49 us and 339 us.
TEST(ReadCoalescer, ALeaderWithoutFollowersReadsIntoTheCallersBuffer) {
  ReadCoalescer c;
  std::string dst(1 << 20, '\0');
  char* seen = nullptr;
  size_t n = 0;
  ASSERT_EQ(c.Read(K(31), 0, dst.size(), &dst[0], dst.size(), &n, nullptr,
                   [&](char* b, size_t cap, size_t* on, size_t*) {
                     seen = b;
                     std::memset(b, 'd', cap);
                     *on = cap;
                     return Status::kOk;
                   }),
            Status::kOk);
  EXPECT_EQ(seen, &dst[0]) << "the leader read into a scratch of its own";
  EXPECT_EQ(n, dst.size());
  EXPECT_EQ(dst, std::string(dst.size(), 'd'));
}

// REQ-20: a fill that throws leaves no flight behind. It stayed in the map,
// and every later identical read waited the full timeout for it.
TEST(ReadCoalescer, AThrowingFillLeavesNoFlightBehind) {
  ReadCoalescer c;
  char buf[4];
  size_t n = 0;
  EXPECT_THROW(c.Read(K(32), 0, 4, buf, sizeof(buf), &n, nullptr,
                      [](char*, size_t, size_t*, size_t*) -> Status {
                        throw std::runtime_error("fill failed");
                      }),
               std::runtime_error);
  ReadCoalescer::Outcome oc;
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_EQ(c.Read(K(32), 0, 4, buf, sizeof(buf), &n, nullptr,
                   [](char* b, size_t, size_t* on, size_t*) {
                     std::memcpy(b, "next", 4);
                     *on = 4;
                     return Status::kOk;
                   },
                   &oc),
            Status::kOk);
  EXPECT_LT(std::chrono::steady_clock::now() - t0, 100ms)
      << "waited for the thrown read's flight";
  EXPECT_TRUE(oc.leader);
  EXPECT_EQ(c.timeouts(), 0u);
}

// REQ-20: a follower that times out takes the flight out of the map, so a
// read arriving after it does not wait again for a leader that is stuck.
TEST(ReadCoalescer, ATimedOutFollowerRemovesTheStuckFlight) {
  ReadCoalescer c;
  uint64_t t = c.TryRegisterAsync(K(33), 0, 4, false);  // never completed
  ASSERT_NE(t, 0u);
  char buf[4];
  size_t n = 0;
  const auto own = [](char* b, size_t, size_t* on, size_t*) {
    std::memcpy(b, "self", 4);
    *on = 4;
    return Status::kOk;
  };
  std::thread first([&] {  // waits out the timeout, then reads itself
    char b2[4];
    size_t n2 = 0;
    EXPECT_EQ(c.Read(K(33), 0, 4, b2, sizeof(b2), &n2, nullptr, own),
              Status::kOk);
  });
  first.join();
  ASSERT_EQ(c.timeouts(), 1u);
  // The next read, on a thread holding no flight (this one holds the async
  // flight, and would read for itself at once whatever happened).
  ReadCoalescer::Outcome oc;
  std::chrono::steady_clock::duration took{};
  std::thread second([&] {
    const auto t0 = std::chrono::steady_clock::now();
    EXPECT_EQ(c.Read(K(33), 0, 4, buf, sizeof(buf), &n, nullptr, own, &oc),
              Status::kOk);
    took = std::chrono::steady_clock::now() - t0;
  });
  second.join();
  EXPECT_LT(took, 100ms) << "waited again for the stuck flight";
  EXPECT_TRUE(oc.leader);
  EXPECT_EQ(c.timeouts(), 1u);
  (void)t;
}

