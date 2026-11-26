// Recent-PUT memory W (CLI-35; docs/design/ring-unavailable-owner.md §3.5):
// ages of each key's latest PUT, expiry before answering, and the node-wide
// clear when CLOCK_REALTIME or CLOCK_BOOTTIME moves against the raw clock by
// more than G within the window, however the movement is spread.
#include "cache/write_memory.h"

#include <gtest/gtest.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "common/block_key_hash.h"
#include "common/config_dump.h"
#include "utils/net_util.h"

using namespace dfkv;  // NOLINT

namespace {

// Offsets the test moves by hand; `bump_on_call` adds to them from that
// call on (a stop in the middle of an answer).
struct FakeClocks {
  WriteMemory::Offsets off;
  int calls = 0, bump_on_call = -1;
  int64_t bump = 0;
};

std::unique_ptr<WriteMemory> Make(FakeClocks* clocks, uint64_t window = 10000,
                                  size_t capacity = 100000) {
  return std::make_unique<WriteMemory>(
      window, capacity, 0,
      [clocks] {
        if (clocks->calls++ == clocks->bump_on_call)
          clocks->off.realtime += clocks->bump;
        return clocks->off;
      },
      /*tick_ms=*/0);
}

std::vector<uint32_t> Ages(WriteMemory& w, const std::vector<BlockKey>& keys,
                           uint64_t age, uint64_t now) {
  std::string d(keys.size() * 16, '\0');
  for (size_t i = 0; i < keys.size(); ++i) {
    net::PutU64(&d[i * 16], keys[i].digest_hi);
    net::PutU64(&d[i * 16 + 8], keys[i].digest_lo);
  }
  std::vector<uint32_t> out(keys.size(), 7);
  w.Ages(9, d.data(), keys.size(), age, now, out.data());
  return out;
}

constexpr uint32_t kNone = RemoveMemory::kAgeNone;
const BlockKey kA{1, 2, 9}, kB{3, 4, 9};

}  // namespace

TEST(WriteMemory, AnswersTheAgeOfTheLatestPutAndExpiresFirst) {
  FakeClocks clocks;
  auto w = Make(&clocks, /*window=*/1000);
  w->Record(kA, 100);
  w->Record(kA, 300);  // written again: the latest counts
  EXPECT_EQ(Ages(*w, {kA, kB}, 1000, 350), (std::vector<uint32_t>{50, kNone}));
  EXPECT_EQ(Ages(*w, {kA}, 40, 350), (std::vector<uint32_t>{kNone}))
      << "older than the age asked about";
  // Past the window: expired before the answer, not reported.
  EXPECT_EQ(Ages(*w, {kA}, 5000, 1400), (std::vector<uint32_t>{kNone}));
  EXPECT_EQ(w->Entries(), 0u);
}

TEST(WriteMemory, AJumpAboveGClearsEveryShardAndOneBelowDoesNot) {
  for (const bool boot : {false, true}) {
    FakeClocks clocks;
    auto w = Make(&clocks);
    std::vector<BlockKey> keys;
    for (uint64_t i = 0; i < 200; ++i) keys.push_back({i * 977, i, 9});
    for (const auto& k : keys) w->Record(k, 100);  // spread over the shards
    (boot ? clocks.off.boottime : clocks.off.realtime) += 900;
    w->Check(1100);
    EXPECT_EQ(w->Entries(), 200u) << "below G";
    (boot ? clocks.off.boottime : clocks.off.realtime) += 200;  // 1100 > G
    const auto ages = Ages(*w, keys, 5000, 2100);
    for (uint32_t a : ages) EXPECT_EQ(a, kNone);
    EXPECT_EQ(w->Entries(), 0u);
    EXPECT_EQ(w->Cleared(boot ? WriteMemory::kBoottime : WriteMemory::kRealtime), 1u);
    EXPECT_EQ(w->Cleared(boot ? WriteMemory::kRealtime : WriteMemory::kBoottime), 0u);
    // After the clear, a new PUT is remembered again.
    w->Record(kA, 2200);
    EXPECT_EQ(Ages(*w, {kA}, 5000, 2300), (std::vector<uint32_t>{100}));
  }
}

// Short stops add up: 0.9 s at every tick never trips a tick-to-tick check,
// but the spread over the window passes G at the second one.
TEST(WriteMemory, RepeatedStopsBelowGClearOnceTheirSumExceedsG) {
  FakeClocks clocks;
  auto w = Make(&clocks);
  w->Record(kA, 100);
  clocks.off.realtime += 900;
  w->Check(1100);
  EXPECT_EQ(w->Entries(), 1u);
  clocks.off.realtime += 900;  // 900 since the last sample, 1800 in all
  w->Check(2100);
  EXPECT_EQ(w->Entries(), 0u) << "the sum of short stops went unseen";
  EXPECT_EQ(w->Cleared(WriteMemory::kRealtime), 1u);
}

// While the table is empty the samples start again, so that slow drift
// before the first PUT does not add to what follows.
TEST(WriteMemory, TheSamplesStartAgainWhileTheTableIsEmpty) {
  FakeClocks clocks;
  auto w = Make(&clocks);
  clocks.off.realtime += 600;
  w->Check(1100);  // empty: starts again here
  w->Record(kA, 1200);
  clocks.off.realtime += 600;
  w->Check(2200);
  EXPECT_EQ(w->Entries(), 1u) << "drift from before the PUT was counted";
  // Not empty: the same drift again does add up.
  clocks.off.realtime += 600;
  w->Check(3200);
  EXPECT_EQ(w->Entries(), 0u);
}

// The samples cover the window: a movement older than it no longer counts.
TEST(WriteMemory, MovementOlderThanTheWindowIsForgotten) {
  FakeClocks clocks;
  auto w = Make(&clocks, /*window=*/3000);
  w->Record(kA, 0);
  for (uint64_t t = 1000; t <= 3000; t += 1000) {
    w->Record(kB, t);  // keeps the table busy
    w->Check(t);
  }
  clocks.off.realtime += 700;
  w->Check(4000);
  for (uint64_t t = 5000; t <= 8000; t += 1000) {
    w->Record(kB, t);
    w->Check(t);
  }
  clocks.off.realtime += 700;  // 1400 in all, but 700 within the window
  w->Check(9000);
  EXPECT_GT(w->Entries(), 0u);
  EXPECT_EQ(w->Cleared(WriteMemory::kRealtime), 0u);
}

// A stop between the reading of the ages and the second check voids them.
TEST(WriteMemory, AStopWhileAnsweringVoidsTheAnswer) {
  FakeClocks clocks;
  auto w = Make(&clocks);
  w->Record(kA, 100);
  clocks.bump_on_call = clocks.calls + 1;  // the check after the reading
  clocks.bump = 1500;
  EXPECT_EQ(Ages(*w, {kA}, 5000, 200), (std::vector<uint32_t>{kNone}));
  EXPECT_EQ(w->Cleared(WriteMemory::kRealtime), 1u);
}

TEST(WriteMemory, EveryPutChecksThePauseBeforePublishing) {
  FakeClocks clocks;
  auto w = Make(&clocks);
  w->Record(kA, 100);
  clocks.off.realtime += 1500;
  w->Record(kB, 200);
  const auto ages = Ages(*w, {kA, kB}, 5000, 250);
  EXPECT_EQ(ages[0], kNone) << "old record must not outlive the epoch check";
  EXPECT_EQ(ages[1], 50u);
  EXPECT_EQ(w->Cleared(WriteMemory::kRealtime), 1u);
}

TEST(WriteMemory, ANewerCheckCannotReageAWriteWhoseCallerReadAnOlderClock) {
  FakeClocks clocks;
  auto w = Make(&clocks);
  // The two callers read RAW 99 / 100; the PUT entered the serial domain
  // first. Raw time 100 covers up to its new sample but this PUT has not
  // been acknowledged: age zero must remain "no completed PUT".
  w->Record(kA, 99);
  w->Check(100);
  EXPECT_EQ(Ages(*w, {kA}, /*age=*/0, /*now=*/100),
            (std::vector<uint32_t>{kNone}));
  // The same PUT is not "already aged" over this zero-width epoch. Its age
  // over one ms is still the ordinary physical age, without hiding it.
  EXPECT_EQ(Ages(*w, {kA}, /*age=*/1, /*now=*/100),
            (std::vector<uint32_t>{1}));
}

TEST(WriteMemory, ALaterPauseClearCoversAWriteWithTheOlderCallerTime) {
  FakeClocks clocks;
  auto w = Make(&clocks);
  w->Record(kA, /*now=*/99);
  clocks.off.realtime = 1500;
  w->Check(/*now=*/100);
  EXPECT_EQ(w->Entries(), 0u);
  EXPECT_EQ(Ages(*w, {kA}, /*age=*/1, /*now=*/101),
            (std::vector<uint32_t>{kNone}));
  w->Record(kB, 101);
  EXPECT_EQ(Ages(*w, {kB}, /*age=*/0, /*now=*/101),
            (std::vector<uint32_t>{0}));
}

TEST(WriteMemory, EffectiveConfigReportsTheResolvedCapacity) {
  struct ClearEnv {
    ~ClearEnv() {
      ::unsetenv("DFKV_WRITE_MEMORY_ENTRIES");
      config_dump::ResetForTest();
    }
  } clear;
  for (const auto& [entries, window, expected] :
       {std::tuple<const char*, uint64_t, const char*>{"4096", 600000u, "4096"},
        {"1", 600000u, "1024"},
        {"67108865", 600000u, "67108864"},
        {"4096", 0u, "0"}}) {
    config_dump::ResetForTest();
    testing::internal::CaptureStderr();
    ::setenv("DFKV_WRITE_MEMORY_ENTRIES", entries, 1);
    auto w = WriteMemory::FromEnv(window, 0);
    config_dump::Emit("test");
    const auto log = testing::internal::GetCapturedStderr();
    if (window == 0)
      EXPECT_EQ(w, nullptr);
    else
      EXPECT_NE(w, nullptr);
    EXPECT_NE(log.find("DFKV_WRITE_MEMORY_ENTRIES = " +
                       std::string(expected) + "  ("),
              std::string::npos)
        << log;
  }
  ::unsetenv("DFKV_WRITE_MEMORY_ENTRIES");
}

TEST(WriteMemory, OffUnlessConfigured) {
  EXPECT_EQ(WriteMemory::FromEnv(600000, 0), nullptr);
  ::setenv("DFKV_WRITE_MEMORY_ENTRIES", "4096", 1);
  EXPECT_EQ(WriteMemory::FromEnv(0, 0), nullptr) << "needs M's window";
  EXPECT_NE(WriteMemory::FromEnv(600000, 0), nullptr);
  ::unsetenv("DFKV_WRITE_MEMORY_ENTRIES");
}

// Opt-in because filling the configured table is deliberately not a unit-test
// cost. The default is the documented 1 Mi entries; the environment selects a
// smaller reproducible run without changing the production-sized default.
TEST(WriteMemory, FullConfiguredClearBenchmark) {
  const char* configured = std::getenv("DFKV_WRITE_CLEAR_BENCH_ENTRIES");
  if (configured == nullptr) {
    GTEST_SKIP() << "set DFKV_WRITE_CLEAR_BENCH_ENTRIES (empty = 1048576)";
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed =
      *configured == '\0' ? 1048576 : std::strtoull(configured, &end, 10);
  if (errno != 0 || (*configured != '\0' && (end == configured || *end != '\0')) ||
      parsed == 0 || parsed > std::numeric_limits<size_t>::max()) {
    ADD_FAILURE() << "invalid DFKV_WRITE_CLEAR_BENCH_ENTRIES=" << configured;
    return;
  }
  const size_t capacity = static_cast<size_t>(parsed);
  ASSERT_GE(capacity, WriteMemory::kShards);
  FakeClocks clocks;
  auto w = Make(&clocks, /*window=*/600000, capacity);

  const size_t per_shard =
      (capacity + WriteMemory::kShards - 1) / WriteMemory::kShards;
  const size_t actual_capacity = per_shard * WriteMemory::kShards;
  std::vector<BlockKey> witnesses(WriteMemory::kShards);
  uint64_t serial = 0;
  for (size_t shard = 0; shard < WriteMemory::kShards; ++shard) {
    for (size_t i = 0; i < per_shard; ++i) {
      BlockKey key;
      do {
        key = BlockKey{serial * 0x9E3779B97F4A7C15ull,
                       serial ^ 0xD1B54A32D192ED03ull, 9};
        ++serial;
      } while ((BlockKeyMixHash{}(key) >> 60) != shard);
      w->Record(key, 100);
      // The last record in every full shard survives its oldest-entry drops.
      witnesses[shard] = key;
    }
  }
  ASSERT_EQ(w->Entries(), actual_capacity);
  ASSERT_EQ(Ages(*w, witnesses, 5000, 200),
            std::vector<uint32_t>(WriteMemory::kShards, 100));

  clocks.off.realtime += WriteMemory::kPauseGuardMs + 100;
  const auto started = std::chrono::steady_clock::now();
  w->Check(1200);
  const auto elapsed = std::chrono::steady_clock::now() - started;
  EXPECT_EQ(w->Entries(), 0u);
  EXPECT_EQ(w->Cleared(WriteMemory::kRealtime), 1u);
  EXPECT_EQ(Ages(*w, witnesses, 5000, 1200),
            std::vector<uint32_t>(WriteMemory::kShards, kNone));
  const auto clear_us =
      std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  std::printf("WRITE_CLEAR_BENCH requested_entries=%zu entries=%zu clear_us=%lld\n",
              capacity, actual_capacity, static_cast<long long>(clear_us));
}
