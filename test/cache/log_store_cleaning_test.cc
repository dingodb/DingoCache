// LogStore, the log-structured engine (docs/design/log-structured-engine.md),
// stage "cleaner and eviction" (§14 stage 6): second-chance eviction,
// relocation, inline reclaim, shadows, journal compaction, capacity changes,
// the per-disk metrics, and the crash points those paths add (§8.2:
// C11-C15, C20, C24, C26, C27, C30).
#include <fcntl.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdio>
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

// ---- eviction and the cleaner ------------------------------------------------------
// A store reopened full keeps everything it recovered until a PUT needs a
// segment beyond the one the open took: only then does the cleaner start
// (one segment's values of N would otherwise go at every restart).
TEST(LogStoreSealing, ARestartKeepsEverythingUntilAPutNeedsSpace) {
  const fs::path dir = TestDir("restartkeeps");
  constexpr size_t kLen = 500000;
  size_t before = 0;
  {
    bool ok = false;
    LogStore s(Opts(dir), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    for (uint64_t id = 0; id < 200; ++id) {
      const std::string v = Val(id, kLen);
      const Status st = s.Cache(Key(id), v.data(), v.size());
      ASSERT_TRUE(st == Status::kOk || st == Status::kCacheFull);
    }
    ASSERT_GT(s.Evictions(), 0u) << "the store was full";
    before = s.Count();
  }
  bool ok = false;
  LogStore s(Opts(dir), &ok);  // with its cleaner
  ASSERT_TRUE(ok) << s.StartupError();
  EXPECT_EQ(s.Count(), before);
  std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  EXPECT_EQ(s.Count(), before) << "nothing evicted while no PUT needs space";
  EXPECT_EQ(s.Evictions(), 0u);
  // Writes that fill the open segment and more: cleaning resumes.
  for (uint64_t id = 1000; id < 1040; ++id) {
    const std::string v = Val(id, kLen);
    const Status st = s.Cache(Key(id), v.data(), v.size());
    ASSERT_TRUE(st == Status::kOk || st == Status::kCacheFull);
  }
  EXPECT_GT(s.Evictions(), 0u);
  EXPECT_TRUE(Has(s, 1039, kLen));
  fs::remove_all(dir);
}

// FIFO eviction with a second chance (§7.3): a value read since it was
// written is relocated, its unread neighbours are evicted.
TEST(LogStoreCleaner, AReadValueGetsASecondChance) {
  const fs::path dir = TestDir("secondchance");
  bool ok = false;
  LogStore s(Opts(dir), &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  constexpr size_t kLen = 500000;
  const std::string hot = Val(1000, kLen), cold = Val(1001, kLen);
  ASSERT_EQ(s.Cache(Key(1000), hot.data(), hot.size()), Status::kOk);
  ASSERT_EQ(s.Cache(Key(1001), cold.data(), cold.size()), Status::kOk);
  for (uint64_t id = 0; id < 200; ++id) {
    const std::string v = Val(id, kLen);
    const Status st = s.Cache(Key(id), v.data(), v.size());
    ASSERT_TRUE(st == Status::kOk || st == Status::kCacheFull);
    if (id % 4 == 0) EXPECT_TRUE(Has(s, 1000, kLen)) << "hot lost at " << id;
  }
  EXPECT_TRUE(Has(s, 1000, kLen));
  std::string out;
  EXPECT_EQ(s.Range(Key(1001), 0, 0, &out), Status::kNotFound);
  const auto st = s.GetStats();
  EXPECT_GT(st.records_relocated, 0u);
  EXPECT_GT(st.records_evicted, 100u);
  EXPECT_GT(st.segments_reclaimed, 5u);
  EXPECT_LE(s.UsedBytes(), kSmall);
  fs::remove_all(dir);
}

// S2 of the #512 review: a long lease on the oldest segment does not stall
// the cleaner while every other sealed segment is mostly protected; it
// reclaims those (relocating what is protected) instead.
TEST(LogStoreCleaner, APinnedOldestSegmentDoesNotStallTheCleaner) {
  const fs::path dir = TestDir("pinnedoldest");
  auto o = Opts(dir, 64ull << 20);
  o.pin_wait_ms = 100;
  o.retain_max_pct = 95;
  bool ok = false;
  LogStore s(o, &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  constexpr size_t kLen = 600000;  // six to a segment
  const uint32_t now = static_cast<uint32_t>(::time(nullptr));
  for (uint64_t id = 0; id < 6; ++id) {  // the oldest segment, unprotected
    const std::string v = Val(id, kLen);
    ASSERT_EQ(s.Cache(Key(id), v.data(), v.size()), Status::kOk);
  }
  ReadLease lease;
  ASSERT_EQ(s.RangeDirectPrep(Key(0), 0, 0, 1 << 20, &lease), Status::kOk);
  // Then two of every three values protected: every later segment is
  // more than half protected.
  uint64_t cache_full = 0;
  for (uint64_t id = 100; id < 200; ++id) {
    const std::string v = Val(id, kLen);
    const Status st = s.Cache(Key(id), v.data(), v.size());
    ASSERT_TRUE(st == Status::kOk || st == Status::kCacheFull);
    cache_full += st == Status::kCacheFull;
    if (st == Status::kOk && id % 3 != 2)
      EXPECT_EQ(s.Retain(Key(id), 600, 0, now), kProtectedNew) << id;
  }
  const auto st = s.GetStats();
  RecordProperty("cache_full", static_cast<int>(cache_full));
  RecordProperty("skipped_pins", static_cast<int>(st.victims_skipped_pins));
  EXPECT_GT(st.segments_reclaimed - st.inline_reclaims, 0u)
      << "the cleaner reclaimed nothing while the oldest segment was pinned";
  EXPECT_EQ(st.protected_evictions, 0u);
  fs::remove_all(dir);
}

// N2 of the #512 review: a victim still pinned keeps its values; the
// cleaner waits for the pins before it evicts or relocates anything.
TEST(LogStoreCleaner, APinnedVictimKeepsItsValues) {
  const fs::path dir = TestDir("pinnedvictim");
  auto o = Opts(dir);
  o.pin_wait_ms = 100;
  bool ok = false;
  LogStore s(o, &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  constexpr size_t kLen = 500000;
  std::vector<uint64_t> ids;
  for (uint64_t id = 0; id < 200; ++id) {  // full: the cleaner has started
    const std::string v = Val(id, kLen);
    const Status st = s.Cache(Key(id), v.data(), v.size());
    ASSERT_TRUE(st == Status::kOk || st == Status::kCacheFull);
  }
  // Every value is leased: every sealed segment is pinned.
  std::vector<std::unique_ptr<ReadLease>> leases;
  for (uint64_t id = 0; id < 200; ++id) {
    auto l = std::make_unique<ReadLease>();
    if (s.RangeDirectPrep(Key(id), 0, 0, 1 << 20, l.get()) != Status::kOk) continue;
    ids.push_back(id);
    leases.push_back(std::move(l));
  }
  ASSERT_GT(ids.size(), 20u);
  const uint64_t evicted = s.Evictions();
  const uint64_t relocated = s.GetStats().records_relocated;
  // Writes until a free segment is taken: the cleaner looks for a victim
  // and finds only pinned ones.
  const uint32_t free_before = s.GetStats().segments_free;
  for (uint64_t id = 1000; id < 1020 && s.GetStats().segments_free >= free_before;
       ++id) {
    const std::string v = Val(id, kLen);
    s.Cache(Key(id), v.data(), v.size());
  }
  for (int i = 0; i < 100 && s.GetStats().victims_skipped_pins == 0; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  EXPECT_GT(s.GetStats().victims_skipped_pins, 0u);
  EXPECT_EQ(s.Evictions(), evicted) << "values evicted from a pinned victim";
  EXPECT_EQ(s.GetStats().records_relocated, relocated)
      << "values relocated out of a pinned victim";
  for (uint64_t id : ids) EXPECT_TRUE(Has(s, id, kLen)) << id;
  leases.clear();
  fs::remove_all(dir);
}

// S3 of the #512 review: a store reopened full, every sealed segment
// holding a protected value, has nothing to reclaim inline; the first PUTs
// that need space wait for the cleaner just armed instead of getting
// kCacheFull.
TEST(LogStoreCleaner, AStoreReopenedFullOfProtectedValuesTakesPutsAtOnce) {
  const fs::path dir = TestDir("reopenprotected");
  constexpr size_t kLen = 500000;
  {
    bool ok = false;
    LogStore s(Opts(dir), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    const uint32_t now = static_cast<uint32_t>(::time(nullptr));
    // Full, a quarter protected (within the budget: no RETAIN is denied).
    for (uint64_t id = 0; id < 60; ++id) {
      const std::string v = Val(id, kLen);
      const Status st = s.Cache(Key(id), v.data(), v.size());
      ASSERT_TRUE(st == Status::kOk || st == Status::kCacheFull);
      if (st == Status::kOk && id % 4 == 0)
        EXPECT_EQ(s.Retain(Key(id), 3000, 0, now), kProtectedNew) << id;
    }
    ASSERT_GT(s.Evictions(), 0u) << "the store was full";
    ASSERT_GT(s.GetRetentionStats().protected_slots, 0u);
  }
  bool ok = false;
  LogStore s(Opts(dir), &ok);  // with its cleaner
  ASSERT_TRUE(ok) << s.StartupError();
  uint64_t cache_full = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (uint64_t id = 1000; id < 1012; ++id) {  // a segment and a half
    const std::string v = Val(id, kLen);
    const Status st = s.Cache(Key(id), v.data(), v.size());
    ASSERT_TRUE(st == Status::kOk || st == Status::kCacheFull);
    cache_full += st == Status::kCacheFull;
  }
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
  RecordProperty("puts_ms", static_cast<int>(ms));
  EXPECT_EQ(cache_full, 0u);
  EXPECT_EQ(s.GetStats().protected_evictions, 0u);
  fs::remove_all(dir);
}

// §7.7: with no cleaner running, a PUT that finds no free segment reclaims
// the oldest sealed one itself (C30 and its crash points).
TEST(LogStoreCleaner, InlineReclaimKeepsAStoreWithoutACleanerWriting) {
  const fs::path dir = TestDir("inline");
  {
    auto o = Opts(dir);
    o.cleaner = false;
    bool ok = false;
    LogStore s(o, &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    constexpr size_t kLen = 1 << 20;
    size_t stored = 0;
    for (uint64_t id = 0; id < 100; ++id) {
      const std::string v = Val(id, kLen);
      const Status st = s.Cache(Key(id), v.data(), v.size());
      ASSERT_TRUE(st == Status::kOk || st == Status::kCacheFull);
      stored += st == Status::kOk;
    }
    EXPECT_GT(stored, 80u);
    EXPECT_GT(s.GetStats().inline_reclaims, 5u);
    EXPECT_TRUE(Has(s, 99, kLen));
  }
  for (const char* point :
       {"inline_reclaim.before_header", "inline_reclaim.after_header"}) {
    SCOPED_TRACE(point);
    fs::remove_all(dir);
    ASSERT_EQ(InChild([&] {
                auto o = Opts(dir);
                o.cleaner = false;
                o.crash_hook = ExitAt(point);
                bool ok = false;
                LogStore s(o, &ok);
                if (!ok) return 2;
                for (uint64_t id = 0; id < 100; ++id) {
                  const std::string v = Val(id, 1 << 20);
                  s.Cache(Key(id), v.data(), v.size());
                }
                return 9;
              }),
              0);
    bool ok = false;
    LogStore s(Quiet(dir), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    EXPECT_GT(s.Count(), 0u);
    for (uint64_t id = 0; id < 100; ++id) {
      std::string out;
      if (s.Range(Key(id), 0, 0, &out) == Status::kOk)
        EXPECT_EQ(out, Val(id, 1 << 20)) << id;
    }
  }
  fs::remove_all(dir);
}

// C11 (victim's entries dropped, new header not durable), C12 (new header
// durable, kills not collected), C13 (relocated copy written, index not
// switched), C20 (the relocation segment is synced before the victim's new
// header: the relocated value survives every one of them). Removed keys
// stay removed throughout.
TEST(LogStoreCleaner, CrashPointsWhileReclaiming) {
  for (const char* point : {"reclaim.before_header", "reclaim.after_header",
                            "relocate.after_copy"}) {
    SCOPED_TRACE(point);
    const fs::path dir = TestDir("reclaimcrash");
    // The REMOVEs that returned, written by the child as each returns: the
    // crash comes from the cleaner's thread, possibly while a REMOVE is in
    // progress, or between a PUT and its REMOVE, and only a REMOVE that
    // returned must stay removed (I2; C7 for the others).
    const std::string removed_log = dir.string() + ".removed";
    std::remove(removed_log.c_str());
    constexpr size_t kLen = 400000;
    ASSERT_EQ(InChild([&] {
                const int log = ::open(removed_log.c_str(),
                                       O_WRONLY | O_CREAT | O_APPEND, 0644);
                if (log < 0) return 6;
                auto o = Opts(dir);
                o.crash_hook = ExitAt(point);
                bool ok = false;
                LogStore s(o, &ok);
                if (!ok) return 2;
                const std::string p = Val(5000, kLen);
                if (s.Cache(Key(5000), p.data(), p.size()) != Status::kOk) return 3;
                // Read: it gets a second chance at the first reclaim.
                if (!Has(s, 5000, kLen)) return 4;
                for (uint64_t id = 0; id < 300; ++id) {
                  const std::string v = Val(id, kLen);
                  const Status st = s.Cache(Key(id), v.data(), v.size());
                  if (id % 10 == 3 && st == Status::kOk) {
                    if (s.Remove(Key(id)) != Status::kOk) return 5;
                    if (::write(log, &id, sizeof(id)) != sizeof(id)) return 7;
                  }
                }
                return 9;
              }),
              0);
    std::set<uint64_t> removed;
    {
      std::ifstream in(removed_log, std::ios::binary);
      uint64_t id = 0;
      while (in.read(reinterpret_cast<char*>(&id), sizeof(id))) removed.insert(id);
    }
    std::remove(removed_log.c_str());
    EXPECT_FALSE(removed.empty()) << "no REMOVE returned before the crash";
    bool ok = false;
    LogStore s(Quiet(dir), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    EXPECT_TRUE(Has(s, 5000, kLen)) << "the value read, relocated in that reclaim";
    for (uint64_t id = 0; id < 300; ++id) {
      std::string out;
      const Status st = s.Range(Key(id), 0, 0, &out);
      if (removed.count(id)) {
        EXPECT_EQ(st, Status::kNotFound) << "removed " << id << " (the REMOVE returned)";
      } else if (st == Status::kOk) {
        EXPECT_EQ(out, Val(id, kLen)) << id;
      }
    }
    fs::remove_all(dir);
  }
}

// C14: a REMOVE while the value is being relocated kills both copies, in
// the running store and after a restart, whether the process goes on or is
// killed right after the REMOVE returned.
TEST(LogStoreCleaner, ARemoveDuringARelocationKillsBothCopies) {
  for (const bool kill : {false, true}) {
    SCOPED_TRACE(kill ? "killed after the REMOVE" : "running on");
    const fs::path dir = TestDir(kill ? "relocremove_kill" : "relocremove");
    constexpr size_t kLen = 400000;
    const int code = InChild([&] {
      LogStore* store = nullptr;
      std::atomic<bool> done{false};
      auto o = Opts(dir);
      o.crash_hook = [&](const char* p) {
        if (std::strcmp(p, "relocate.after_copy") != 0 || done.exchange(true))
          return;
        // On the cleaner's thread, between the copy and the index switch.
        if (store->Remove(Key(5000)) != Status::kOk) Kill(6);
        if (kill) Kill(0);
      };
      bool ok = false;
      LogStore s(o, &ok);
      store = &s;
      if (!ok) return 2;
      const std::string p = Val(5000, kLen);
      if (s.Cache(Key(5000), p.data(), p.size()) != Status::kOk) return 3;
      if (!Has(s, 5000, kLen)) return 4;  // read: relocated, not evicted
      for (uint64_t id = 0; id < 300 && !done; ++id) {
        const std::string v = Val(id, kLen);
        s.Cache(Key(id), v.data(), v.size());
      }
      if (!done) return 9;
      // Let the reclaim finish, then more cleaning over the relocation
      // segment.
      for (uint64_t id = 300; id < 400; ++id) {
        const std::string v = Val(id, kLen);
        s.Cache(Key(id), v.data(), v.size());
      }
      std::string out;
      if (s.Range(Key(5000), 0, 0, &out) != Status::kNotFound) return 7;
      return s.Close() ? 0 : 8;
    });
    ASSERT_EQ(code, 0);
    bool ok = false;
    LogStore s(Quiet(dir), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    std::string out;
    EXPECT_EQ(s.Range(Key(5000), 0, 0, &out), Status::kNotFound);
    fs::remove_all(dir);
  }
}

// ---- journal compaction (§4.5) ---------------------------------------------------
namespace {
// A workload whose journal grows far beyond its live entries: values written
// and removed until their segments are reused, which makes their kills
// garbage (§4.5).
int CompactionWorkload(LogStore& s) {
  const std::string v = Val(1, 5000);
  for (uint64_t id = 0; id < 20; ++id) {
    const std::string w = Val(id + 10, 5000);
    if (s.Cache(Key(id + 10), w.data(), w.size()) != Status::kOk) return 3;
  }
  for (uint64_t id = 0; id < 10; ++id)
    if (s.Remove(Key(id + 10)) != Status::kOk) return 4;
  if (s.Cache(Key(1), v.data(), v.size()) != Status::kOk) return 5;
  // 1 MiB values: few records per segment, so segments come back into use
  // quickly and the kills naming them become garbage.
  const std::string big(1 << 20, 'c');
  for (uint64_t id = 1000; id < 1300; ++id) {
    if (s.Cache(Key(id), big.data(), big.size()) != Status::kOk) return 6;
    if (s.Remove(Key(id)) != Status::kOk) return 7;
  }
  return 0;
}
void CheckAfterCompaction(LogStore& s) {
  for (uint64_t id = 0; id < 20; ++id) {
    std::string out;
    const Status st = s.Range(Key(id + 10), 0, 0, &out);
    if (id < 10) EXPECT_EQ(st, Status::kNotFound) << "removed " << id;
    else EXPECT_TRUE(Has(s, id + 10, 5000)) << id;
  }
  for (uint64_t id = 1000; id < 1300; id += 7) {
    std::string out;
    EXPECT_EQ(s.Range(Key(id), 0, 0, &out), Status::kNotFound) << "removed " << id;
  }
  EXPECT_TRUE(Has(s, 1, 5000));
}
}  // namespace

TEST(LogStoreJournal, CompactionKeepsTheLiveEntriesOnly) {
  const fs::path dir = TestDir("compact");
  {
    auto o = Opts(dir);
    o.journal_compact_bytes = 16 * 4096;
    bool ok = false;
    LogStore s(o, &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    ASSERT_EQ(CompactionWorkload(s), 0);
    EXPECT_GT(s.GetStats().journal_compactions, 0u);
    CheckAfterCompaction(s);
  }
  // Uncompacted it would hold one block per REMOVE: 310 blocks.
  EXPECT_LT(fs::file_size(dir / "lse_journal"), 150u * 4096) << "compacted";
  bool ok = false;
  LogStore s(Quiet(dir), &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  CheckAfterCompaction(s);
  fs::remove_all(dir);
}

// C15: a crash after the new journal is written but before the rename is
// durable, or right after it: either journal is whole.
TEST(LogStoreJournal, CrashPointsWhileCompacting) {
  for (const char* point :
       {"compact.begin", "compact.after_write", "compact.after_rename"}) {
    SCOPED_TRACE(point);
    const fs::path dir = TestDir("compactcrash");
    ASSERT_EQ(InChild([&] {
                auto o = Opts(dir);
                o.journal_compact_bytes = 16 * 4096;
                o.crash_hook = ExitAt(point);
                bool ok = false;
                LogStore s(o, &ok);
                if (!ok) return 2;
                const int rc = CompactionWorkload(s);
                return rc != 0 ? rc : 9;
              }),
              0);
    bool ok = false;
    LogStore s(Quiet(dir), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    for (uint64_t id = 0; id < 10; ++id) {
      std::string out;
      EXPECT_EQ(s.Range(Key(id + 10), 0, 0, &out), Status::kNotFound) << id;
    }
    std::string out;
    if (s.Range(Key(1), 0, 0, &out) == Status::kOk) EXPECT_EQ(out, Val(1, 5000));
    fs::remove_all(dir);
  }
}

// C24: a REMOVE that arrives during a compaction waits, is written to the
// new journal, and survives a restart.
TEST(LogStoreJournal, ARemoveDuringACompactionIsWrittenToTheNewJournal) {
  const fs::path dir = TestDir("compactremove");
  std::thread remover;
  std::atomic<bool> claimed{false}, spawned{false};
  std::atomic<int> removed{-1};
  {
    LogStore* store = nullptr;
    auto o = Opts(dir);
    o.journal_compact_bytes = 16 * 4096;
    o.crash_hook = [&](const char* p) {
      if (std::strcmp(p, "compact.begin") != 0 || claimed.exchange(true)) return;
      remover = std::thread([&] {
        removed = static_cast<int>(store->Remove(Key(25)));
      });
      spawned = true;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      EXPECT_EQ(removed.load(), -1) << "returned before the compaction ended";
    };
    bool ok = false;
    LogStore s(o, &ok);
    store = &s;
    ASSERT_TRUE(ok) << s.StartupError();
    ASSERT_EQ(CompactionWorkload(s), 0);
    // The workload's last REMOVE waited for the compaction: `remover` was
    // assigned before `spawned` was set.
    ASSERT_TRUE(spawned.load()) << "no compaction";
    remover.join();
    EXPECT_EQ(removed.load(), static_cast<int>(Status::kOk));
  }
  bool ok = false;
  LogStore s(Quiet(dir), &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  std::string out;
  EXPECT_EQ(s.Range(Key(25), 0, 0, &out), Status::kNotFound);
  EXPECT_TRUE(Has(s, 26, 5000));
  fs::remove_all(dir);
}

// ---- capacity changes (§7.6) -------------------------------------------------------
TEST(LogStoreCapacity, GrowAndShrinkFollowTheConfiguredCapacity) {
  const fs::path dir = TestDir("capacity");
  constexpr size_t kLen = 900000;
  {
    bool ok = false;
    LogStore s(Opts(dir, kSmall), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    for (uint64_t id = 0; id < 12; ++id) {
      const std::string v = Val(id, kLen);
      ASSERT_EQ(s.Cache(Key(id), v.data(), v.size()), Status::kOk);
    }
  }
  {
    bool ok = false;
    LogStore s(Quiet(dir, 2 * kSmall), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    EXPECT_EQ(s.GetStats().segment_count, 16u);
    EXPECT_EQ(s.GetStats().segments_grown, 8u);
    for (uint64_t id = 0; id < 12; ++id) EXPECT_TRUE(Has(s, id, kLen)) << id;
    // Fill the new segments too.
    for (uint64_t id = 100; id < 140; ++id) {
      const std::string v = Val(id, kLen);
      ASSERT_EQ(s.Cache(Key(id), v.data(), v.size()), Status::kOk);
    }
  }
  EXPECT_TRUE(fs::exists(dir / "segments" / "G000015"));
  bool ok = false;
  {
    LogStore s(Quiet(dir, kSmall), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    EXPECT_EQ(s.GetStats().segment_count, 8u);
    EXPECT_EQ(s.GetStats().segments_shrunk, 8u);
    EXPECT_FALSE(fs::exists(dir / "segments" / "G000008"));
    for (uint64_t id : {0ull, 5ull, 100ull, 139ull}) {
      std::string out;
      if (s.Range(Key(id), 0, 0, &out) == Status::kOk) EXPECT_EQ(out, Val(id, kLen));
    }
    const std::string v = Val(500, kLen);
    EXPECT_EQ(s.Cache(Key(500), v.data(), v.size()), Status::kOk);
  }
  LogStore again(Opts(dir, kSmall), &ok);
  ASSERT_TRUE(ok) << again.StartupError();
  EXPECT_TRUE(Has(again, 500, kLen));
  fs::remove_all(dir);
}

// C26 (grow: files created, superblock not updated) and C27 (shrink:
// superblock lowered, files not removed): leftovers are ignored, and the
// next grow reuses them.
TEST(LogStoreCapacity, CrashPointsWhileResizing) {
  const fs::path dir = TestDir("resizecrash");
  constexpr size_t kLen = 900000;
  {
    bool ok = false;
    LogStore s(Opts(dir, kSmall), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    for (uint64_t id = 0; id < 6; ++id) {
      const std::string v = Val(id, kLen);
      ASSERT_EQ(s.Cache(Key(id), v.data(), v.size()), Status::kOk);
    }
  }
  ASSERT_EQ(InChild([&] {
              auto o = Opts(dir, 2 * kSmall);
              o.crash_hook = ExitAt("grow.after_files");
              bool ok = false;
              LogStore s(o, &ok);
              return 9;
            }),
            0);
  EXPECT_TRUE(fs::exists(dir / "segments" / "G000015")) << "leftovers";
  {
    auto o = Opts(dir, kSmall);
    bool ok = false;
    LogStore s(o, &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    EXPECT_EQ(s.GetStats().segment_count, 8u) << "leftovers ignored";
    for (uint64_t id = 0; id < 6; ++id) EXPECT_TRUE(Has(s, id, kLen)) << id;
  }
  {
    bool ok = false;
    LogStore s(Quiet(dir, 2 * kSmall), &ok);  // reuses the leftovers
    ASSERT_TRUE(ok) << s.StartupError();
    EXPECT_EQ(s.GetStats().segment_count, 16u);
    for (uint64_t id = 0; id < 6; ++id) EXPECT_TRUE(Has(s, id, kLen)) << id;
  }
  ASSERT_EQ(InChild([&] {
              auto o = Opts(dir, kSmall);
              o.crash_hook = ExitAt("shrink.after_super");
              bool ok = false;
              LogStore s(o, &ok);
              return 9;
            }),
            0);
  EXPECT_TRUE(fs::exists(dir / "segments" / "G000015")) << "not removed yet";
  bool ok = false;
  LogStore s(Quiet(dir, kSmall), &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  EXPECT_EQ(s.GetStats().segment_count, 8u);
  for (uint64_t id = 0; id < 6; ++id) {
    std::string out;
    if (s.Range(Key(id), 0, 0, &out) == Status::kOk) EXPECT_EQ(out, Val(id, kLen));
  }
  fs::remove_all(dir);
}

// ---- endurance and metrics ------------------------------------------------------------
// Many times the capacity of random PUTs, REMOVEs, reads and RETAINs: every
// read is right, the store stays healthy and within its capacity, tenant
// accounting matches, and after a restart every value that was neither
// removed nor evicted since it was stored is there.
TEST(LogStoreCleaner, FillAndEvictEndurance) {
  const fs::path dir = TestDir("endurance");
  constexpr uint64_t kKeys = 96;
  std::map<uint64_t, std::pair<uint32_t, size_t>> latest;  // id -> version, len
  std::set<uint64_t> gone;  // removed or evicted since the latest store
  std::mutex mu;
  auto o = Opts(dir);
  o.on_evict = [&](const BlockKey& k, uint64_t) {
    std::lock_guard<std::mutex> lk(mu);
    gone.insert(k.digest_lo);  // Key(id).digest_lo == id
  };
  {
    bool ok = false;
    LogStore s(o, &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    std::mt19937_64 rng(17);
    std::map<uint64_t, uint32_t> version;
    const uint32_t now = static_cast<uint32_t>(::time(nullptr));
    uint64_t written = 0, max_generation = 0;
    while (written < 12 * kSmall) {
      const uint64_t id = rng() % kKeys;
      const int what = static_cast<int>(rng() % 10);
      if (what < 5) {
        const size_t len = 1 + rng() % 300000;
        const uint32_t ver = ++version[id];
        const std::string v = Val(id, len, ver);
        const Status st = s.Cache(Key(id), v.data(), v.size());
        ASSERT_TRUE(st == Status::kOk || st == Status::kCacheFull) << int(st);
        // Stored (not a keep-first answer): a generation above all seen.
        ValueMetadata m;
        if (st == Status::kOk && s.Lookup(Key(id), &m) == Status::kOk &&
            m.generation > max_generation) {
          max_generation = m.generation;
          std::lock_guard<std::mutex> lk(mu);  // never held across a store call
          latest[id] = {ver, len};
          gone.erase(id);
        }
        written += len;
      } else if (what < 7) {
        std::string out;
        if (s.Range(Key(id), 0, 0, &out) == Status::kOk) {
          bool known = false;
          for (uint32_t v = 1; v <= version[id] && !known; ++v)
            known = out == Val(id, out.size(), v);
          ASSERT_TRUE(known) << "wrong bytes for " << id;
        }
      } else if (what < 8) {
        s.Retain(Key(id), 30 + rng() % 600, 0, now);
      } else {
        s.Remove(Key(id));
        std::lock_guard<std::mutex> lk(mu);
        gone.insert(id);
      }
    }
    EXPECT_TRUE(s.Healthy());
    EXPECT_LE(s.UsedBytes(), kSmall);
    uint64_t t0 = 0, t1 = 0;
    for (uint64_t id = 0; id < kKeys; ++id) {
      ValueMetadata m;
      if (s.Lookup(Key(id), &m) == Status::kOk)
        (Key(id).tenant_hash == 0 ? t0 : t1) += m.value_len;
    }
    EXPECT_EQ(s.TenantUsedBytes(0), t0);
    EXPECT_EQ(s.TenantUsedBytes(1), t1);
    // Before the restart too: what was stored and not taken away is there.
    for (const auto& kv : latest) {
      if (gone.count(kv.first)) continue;
      EXPECT_TRUE(Has(s, kv.first, kv.second.second, kv.second.first))
          << "running store, key " << kv.first;
    }
    const auto st = s.GetStats();
    EXPECT_EQ(st.read_crc_failures, 0u);
    EXPECT_GT(st.segments_reclaimed, 20u);
    EXPECT_GT(st.records_relocated, 0u);
    RecordProperty("reclaimed", static_cast<int>(st.segments_reclaimed));
    RecordProperty("relocated", static_cast<int>(st.records_relocated));
    RecordProperty("inline_reclaims", static_cast<int>(st.inline_reclaims));
    RecordProperty("cache_full", static_cast<int>(st.cache_full));
  }
  bool ok = false;
  LogStore s(Quiet(dir), &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  EXPECT_EQ(s.GetRecoveryStats().segments_walked, 0u);
  for (const auto& kv : latest) {
    if (gone.count(kv.first)) continue;
    std::string out;
    const Status st = s.Range(Key(kv.first), 0, 0, &out);
    EXPECT_TRUE(st == Status::kOk &&
                out == Val(kv.first, kv.second.second, kv.second.first))
        << "after the restart, key " << kv.first << " status " << int(st)
        << " len " << out.size() << " want " << kv.second.second;
  }
  fs::remove_all(dir);
}

TEST(LogStoreMetrics, TheEngineReportsTheLseFamily) {
  const fs::path dir = TestDir("metrics");
  bool ok = false;
  LogStore s(Opts(dir), &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  std::vector<StoreEngine::EngineMetric> m;
  s.EngineMetrics(&m);
  std::set<std::string> names;
  for (const auto& e : m) {
    names.insert(e.name);
    EXPECT_TRUE(std::string(e.type) == "gauge" || std::string(e.type) == "counter");
  }
  for (const char* want : {"dfkv_lse_segments", "dfkv_lse_segments_free",
                           "dfkv_lse_records_evicted_total",
                           "dfkv_lse_records_relocated_total",
                           "dfkv_lse_inline_reclaims_total",
                           "dfkv_lse_protected_evictions_total",
                           "dfkv_lse_journal_compactions_total",
                           "dfkv_lse_journal_integrity_resets_total"})
    EXPECT_TRUE(names.count(want)) << want;
  fs::remove_all(dir);
}
