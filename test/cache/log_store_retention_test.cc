// LogStore, the log-structured engine (docs/design/log-structured-engine.md),
// stage "RETAIN" (§14 stage 7): slab's retention rules over the log engine,
// the footprint budget, persistence in the journal and its crash point
// (C17), and protected values through cleaning and compaction.
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "cache/log_store.h"
#include "log_store_test_util.h"

using namespace dfkv;      // NOLINT
using namespace lse_test;  // NOLINT
namespace fs = std::filesystem;

// Slab's retention rules (SlabAllocatorRetention.*, DiskSlabTest.Retention*)
// over the log engine: codes, REFRESH with the stored TTL, never shortening,
// the TTL cap, the budget (counted as on-disk footprint), REMOVE releasing
// it, expiry by 10 s buckets.
TEST(LogStoreRetention, FollowsSlabsRules) {
  const fs::path dir = TestDir("retain");
  auto o = Opts(dir);
  o.cleaner = false;
  o.retain_max_pct = 1;  // 1% of 16 MiB usable: one 128 KiB value's footprint
  bool ok = false;
  LogStore s(o, &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  ASSERT_TRUE(s.RetainEnabled());
  EXPECT_EQ(s.RetainMaxTtl(), 3600u);
  constexpr size_t kLen = 128 << 10;
  for (uint64_t id : {1, 2, 3}) {
    const std::string v = Val(id, kLen);
    ASSERT_EQ(s.Cache(Key(id), v.data(), v.size()), Status::kOk);
  }
  EXPECT_EQ(s.Retain(Key(99), 60, 0, kT0), kMiss);
  EXPECT_EQ(s.Retain(Key(1), 0, 1, kT0), kHitUnprotected)
      << "REFRESH of a never-protected value";
  EXPECT_EQ(s.Retain(Key(1), 300, 0, kT0), kProtectedNew);
  EXPECT_EQ(s.Retain(Key(1), 3600, 1, kT0 + 240), kRefreshed)
      << "REFRESH renews with the stored TTL";
  EXPECT_EQ(s.Retain(Key(1), 60, 0, kT0 + 240), kUnchanged)
      << "a shorter SET never lowers anything";
  EXPECT_EQ(s.Retain(Key(1), 3600, 0, kT0 + 242), kUpgraded);
  EXPECT_EQ(s.Retain(Key(2), 60, 0, kT0), kDenied) << "the budget is used";
  EXPECT_EQ(s.Retain(Key(1), 0, 1, kT0 + 300), kRefreshed)
      << "renewing protected bytes is always allowed";
  EXPECT_LT(s.RetainHeadroomPct(), 25u);
  auto st = s.GetRetentionStats();
  EXPECT_TRUE(st.enabled);
  EXPECT_EQ(st.protected_slots, 1u);
  EXPECT_EQ(st.protected_bytes, 4096u + kLen) << "counted as footprint";
  EXPECT_EQ(st.max_ttl_s, 3600u);
  EXPECT_EQ(st.results[kProtectedNew], 1u);
  EXPECT_EQ(st.results[kDenied], 1u);
  // REMOVE ignores protection and gives the budget back.
  ASSERT_EQ(s.Remove(Key(1)), Status::kOk);
  EXPECT_EQ(s.GetRetentionStats().protected_bytes, 0u);
  EXPECT_EQ(s.Retain(Key(2), 60, 0, kT0 + 300), kProtectedNew);
  // Expiry: not before the deadline's 10 s bucket has passed.
  EXPECT_EQ(s.ExpireRetention(kT0 + 359), 0u);
  EXPECT_EQ(s.ExpireRetention(kT0 + 361), 0u) << "the bucket is not over";
  EXPECT_EQ(s.ExpireRetention(kT0 + 380), 1u);
  EXPECT_EQ(s.GetRetentionStats().protected_slots, 0u);
  // After expiry a REFRESH protects again with the stored TTL.
  EXPECT_EQ(s.Retain(Key(2), 0, 1, kT0 + 400), kProtectedNew);
  fs::remove_all(dir);
}

TEST(LogStoreRetention, TheTtlIsCappedAndRetentionCanBeTurnedOff) {
  const fs::path dir = TestDir("retaincap");
  {
    auto o = Opts(dir);
    o.cleaner = false;
    o.retain_max_ttl_s = 600;
    bool ok = false;
    LogStore s(o, &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    const std::string v = Val(1, 5000);
    ASSERT_EQ(s.Cache(Key(1), v.data(), v.size()), Status::kOk);
    EXPECT_EQ(s.Retain(Key(1), 3600, 0, kT0), kProtectedNew);
    EXPECT_EQ(s.ExpireRetention(kT0 + 605), 0u);
    EXPECT_EQ(s.ExpireRetention(kT0 + 620), 1u) << "capped at 600 s";
  }
  fs::remove_all(dir);
  auto o = Opts(dir);
  o.retain_enabled = false;
  bool ok = false;
  LogStore s(o, &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  EXPECT_FALSE(s.RetainEnabled());
  EXPECT_EQ(s.RetainMaxTtl(), 0u);
  const std::string v = Val(1, 100);
  ASSERT_EQ(s.Cache(Key(1), v.data(), v.size()), Status::kOk);
  EXPECT_EQ(s.Retain(Key(1), 60, 0, kT0), StoreEngine::kRetainUnsupported);
  fs::remove_all(dir);
}

// Protection persists in the journal (matched on key and generation) and is
// re-clamped to the configured cap at the next open; an expired one keeps
// only its TTL, ready for a REFRESH (slab's RestoreBulk rules).
TEST(LogStoreRetention, ProtectionPersistsAcrossRestarts) {
  const fs::path dir = TestDir("retainpersist");
  std::atomic<uint32_t> now{kT0};
  auto opts = [&](uint32_t max_ttl) {
    auto o = Opts(dir);
    o.clock = [&] { return now.load(); };
    o.retain_max_ttl_s = max_ttl;
    return o;
  };
  {
    bool ok = false;
    LogStore s(opts(3600), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    for (uint64_t id : {1, 2}) {
      const std::string v = Val(id, 5000);
      ASSERT_EQ(s.Cache(Key(id), v.data(), v.size()), Status::kOk);
    }
    ASSERT_EQ(s.Retain(Key(1), 3600, 0, kT0), kProtectedNew);
    ASSERT_EQ(s.Retain(Key(2), 60, 0, kT0), kProtectedNew);
  }
  now = kT0 + 100;  // Key(2)'s protection has passed
  {
    bool ok = false;
    LogStore s(opts(600), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    EXPECT_EQ(s.GetRecoveryStats().retains_restored, 1u);
    EXPECT_EQ(s.GetRetentionStats().protected_slots, 1u);
    EXPECT_EQ(s.ExpireRetention(kT0 + 695), 0u);
    EXPECT_EQ(s.ExpireRetention(kT0 + 730), 1u) << "re-clamped to 600 s";
    EXPECT_EQ(s.Retain(Key(2), 0, 1, kT0 + 100), kProtectedNew)
        << "the stored TTL came back";
  }
  fs::remove_all(dir);
}

// C17: a RETAIN returns before its journal write; once a later journal write
// is acknowledged (any REMOVE waits for everything queued before it), the
// protection survives a kill. Without that, it may be lost (allowed).
TEST(LogStoreRetention, AProtectionSurvivesAKillOnceTheJournalHasIt) {
  for (const bool flushed : {true, false}) {
    SCOPED_TRACE(flushed ? "flushed" : "not waited for");
    const fs::path dir = TestDir(flushed ? "retainkill" : "retainkill_nowait");
    ASSERT_EQ(InChild([&] {
                bool ok = false;
                LogStore s(Opts(dir), &ok);
                if (!ok) return 2;
                const std::string v = Val(1, 5000);
                if (s.Cache(Key(1), v.data(), v.size()) != Status::kOk) return 3;
                if (s.Retain(Key(1), 3600, 0,
                             static_cast<uint32_t>(::time(nullptr))) != kProtectedNew)
                  return 4;
                if (flushed) s.Remove(Key(999));
                ::_exit(0);  // killed
              }),
              0);
    bool ok = false;
    LogStore s(Quiet(dir), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    EXPECT_TRUE(Has(s, 1, 5000));
    if (flushed) EXPECT_EQ(s.GetRetentionStats().protected_slots, 1u);
    fs::remove_all(dir);
  }
}

// A protected value is relocated, never evicted, until its protection
// expires; then it ages out like any other (§7.3, §9.2).
TEST(LogStoreCleaner, AProtectedValueSurvivesCleaningUntilItExpires) {
  const fs::path dir = TestDir("protected");
  std::atomic<uint32_t> now{kT0};
  auto o = Opts(dir);
  o.clock = [&] { return now.load(); };
  constexpr size_t kLen = 500000;
  auto fill = [&](LogStore& s, uint64_t from, uint64_t n) {
    for (uint64_t id = from; id < from + n; ++id) {
      const std::string v = Val(id, kLen);
      const Status st = s.Cache(Key(id), v.data(), v.size());
      ASSERT_TRUE(st == Status::kOk || st == Status::kCacheFull);
    }
  };
  {
    bool ok = false;
    LogStore s(o, &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    const std::string p = Val(7777, kLen);
    ASSERT_EQ(s.Cache(Key(7777), p.data(), p.size()), Status::kOk);
    ASSERT_EQ(s.Retain(Key(7777), 600, 0, kT0), kProtectedNew);
    fill(s, 0, 200);  // three times the capacity
    ASSERT_EQ(s.GetRetentionStats().protected_slots, 1u);
    EXPECT_GT(s.GetStats().records_relocated, 0u);
    EXPECT_EQ(s.GetStats().protected_evictions, 0u);
    EXPECT_GT(s.GetStats().records_evicted, 100u);
    EXPECT_TRUE(Has(s, 7777, kLen));
    // Expired (past the deadline's 10 s bucket): ordinary from then on.
    now = kT0 + 700;
    s.ExpireRetention(now);
    EXPECT_EQ(s.GetRetentionStats().protected_slots, 0u);
  }
  // It sits in the relocation segment, sealed when full or at a close; after
  // the restart it ages out like the values written around it.
  bool ok = false;
  LogStore s(o, &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  EXPECT_EQ(s.GetRetentionStats().protected_slots, 0u);
  fill(s, 1000, 300);
  std::string out;
  EXPECT_EQ(s.Range(Key(7777), 0, 0, &out), Status::kNotFound);
  fs::remove_all(dir);
}

// A compaction keeps the RETAIN entries of protected values (§4.5).
TEST(LogStoreRetention, CompactionKeepsTheRetainEntries) {
  const fs::path dir = TestDir("retaincompact");
  {
    auto o = Opts(dir);
    o.journal_compact_bytes = 16 * 4096;
    bool ok = false;
    LogStore s(o, &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    const std::string v = Val(1, 5000);
    ASSERT_EQ(s.Cache(Key(1), v.data(), v.size()), Status::kOk);
    const uint32_t now = static_cast<uint32_t>(::time(nullptr));
    ASSERT_EQ(s.Retain(Key(1), 3000, 0, now), kProtectedNew);
    // TTL raises persist every time: far more entries than live ones.
    for (uint32_t i = 1; i <= 400; ++i) {
      s.Retain(Key(1), 3000 + i, 0, now);
      if (i % 4 == 0) s.Remove(Key(999));  // waits: one block per few raises
    }
    EXPECT_GT(s.GetStats().journal_compactions, 0u);
  }
  EXPECT_LT(fs::file_size(dir / "lse_journal"), 64u * 4096) << "compacted";
  bool ok = false;
  LogStore s(Quiet(dir), &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  EXPECT_TRUE(Has(s, 1, 5000));
  EXPECT_EQ(s.GetRetentionStats().protected_slots, 1u) << "the RETAIN entry was kept";
  fs::remove_all(dir);
}
