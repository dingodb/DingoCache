// LogStore, the log-structured engine (docs/design/log-structured-engine.md),
// stage "sealing" (§14 stage 5): summaries and trailers, the clean close,
// summary-based recovery, and the crash points around them (§8.2: C9, C18,
// C21, C28), with the REMOVE and PUT orderings this stage's hooks expose
// (C31, C32, C34) and the interrupted reset (C35).
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

// ---- sealing and recovery ---------------------------------------------------------
// After a clean close every segment has a trailer: the open loads summaries
// and walks nothing (§8.5).
TEST(LogStoreSealing, ACleanCloseLeavesOnlySummariesToLoad) {
  const fs::path dir = TestDir("summaries");
  constexpr uint64_t kCount = 1500;
  constexpr size_t kLen = 8000;
  {
    bool ok = false;
    LogStore s(Opts(dir, 64ull << 20), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    for (uint64_t id = 0; id < kCount; ++id) {
      const std::string v = Val(id, kLen);
      ASSERT_EQ(s.Cache(Key(id), v.data(), v.size()), Status::kOk) << id;
    }
    // After the close: the sealer seals asynchronously until then.
    ASSERT_TRUE(s.Close());
    EXPECT_GE(s.GetStats().segments_sealed, 4u);
  }
  const auto t0 = std::chrono::steady_clock::now();
  bool ok = false;
  LogStore s(Quiet(dir, 64ull << 20), &ok);
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
  ASSERT_TRUE(ok) << s.StartupError();
  RecordProperty("reopen_ms", static_cast<int>(ms));
  const auto r = s.GetRecoveryStats();
  EXPECT_EQ(r.segments_walked, 0u);
  EXPECT_GE(r.summaries_loaded, 4u);
  EXPECT_EQ(s.Count(), kCount);
  for (uint64_t id = 0; id < kCount; id += 7) EXPECT_TRUE(Has(s, id, kLen)) << id;
  fs::remove_all(dir);
}

// After a kill only segments without a trailer are walked (§8.4); Activate
// seals what it walked, so the next unclean exit walks only what was open
// then (C28).
TEST(LogStoreSealing, AKillLeavesOnlyTheUnsealedSegmentsToWalk) {
  const fs::path dir = TestDir("walk");
  constexpr uint64_t kCount = 1500;
  constexpr size_t kLen = 8000;
  ASSERT_EQ(InChild([&] {
              bool ok = false;
              LogStore s(Opts(dir, 64ull << 20), &ok);
              if (!ok) return 2;
              for (uint64_t id = 0; id < kCount; ++id) {
                const std::string v = Val(id, kLen);
                if (s.Cache(Key(id), v.data(), v.size()) != Status::kOk) return 3;
              }
              ::_exit(0);  // killed: no Close
            }),
            0);
  {
    bool ok = false;
    LogStore s(Quiet(dir, 64ull << 20), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    const auto r = s.GetRecoveryStats();
    EXPECT_GE(r.segments_walked, 1u);
    EXPECT_LE(r.segments_walked, 4u) << "open + relocation + sealer queue";
    EXPECT_GE(r.summaries_loaded, 2u);
    EXPECT_EQ(s.Count(), kCount);
    for (uint64_t id = 0; id < kCount; id += 5) EXPECT_TRUE(Has(s, id, kLen)) << id;
  }
  // The open above sealed the walked segments; a process that opens and is
  // killed at once leaves at most its own empty open segment to walk.
  ASSERT_EQ(InChild([&]() -> int {
              bool ok = false;
              LogStore s(Opts(dir, 64ull << 20), &ok);
              ::_exit(ok ? 0 : 2);
            }),
            0);
  bool ok = false;
  LogStore s(Quiet(dir, 64ull << 20), &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  EXPECT_LE(s.GetRecoveryStats().segments_walked, 1u);
  EXPECT_EQ(s.Count(), kCount);
  fs::remove_all(dir);
}

// C9 (summary written, trailer not), C18 (some segments sealed at Close,
// clean flag not set), C28 (Activate interrupted while sealing walked
// segments): the next open walks what has no trailer and loses nothing.
TEST(LogStoreSealing, CrashPointsWhileSealing) {
  struct Case {
    const char* point;
    bool via_close;      // the point is reached in Close()
    bool via_activate;   // ... in the Activate() of a second process
  };
  for (const Case& c : std::vector<Case>{{"seal.after_summary", false, false},
                                         {"seal.after_trailer", false, false},
                                         {"close.after_seal", true, false},
                                         {"seal.after_summary", false, true}}) {
    SCOPED_TRACE(std::string(c.point) + (c.via_close ? " close" : "") +
                 (c.via_activate ? " activate" : ""));
    const fs::path dir = TestDir("sealcrash");
    constexpr size_t kLen = 300000;
    const int code = InChild([&] {
      auto o = Opts(dir);
      if (!c.via_activate) o.crash_hook = ExitAt(c.point);
      bool ok = false;
      LogStore s(o, &ok);
      if (!ok) return 2;
      for (uint64_t id = 0; id < 60; ++id) {
        const std::string v = Val(id, kLen);
        if (s.Cache(Key(id), v.data(), v.size()) != Status::kOk) return 3;
      }
      if (c.via_close) s.Close();
      return c.via_activate ? 0 : 9;  // 9: the point was not reached
    });
    ASSERT_EQ(code, 0);
    if (c.via_activate) {
      ASSERT_EQ(InChild([&] {
                  auto o = Opts(dir);
                  o.crash_hook = ExitAt(c.point);
                  bool ok = false;
                  LogStore s(o, &ok);
                  return 9;
                }),
                0);
    }
    bool ok = false;
    LogStore s(Quiet(dir), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    // What is served is right, and the first values (sealed or walked) are
    // there.
    for (uint64_t id = 0; id < 60; ++id) {
      std::string out;
      if (s.Range(Key(id), 0, 0, &out) == Status::kOk)
        EXPECT_EQ(out, Val(id, kLen)) << id;
    }
    for (uint64_t id = 0; id < 5; ++id) EXPECT_TRUE(Has(s, id, kLen)) << id;
    fs::remove_all(dir);
  }
}

// C21: a victim's new header torn: the segment is classed free at open, gets
// a fresh incarnation at Activate, and its old records never come back.
TEST(LogStoreCleaner, ATornSegmentHeaderGivesUpItsRecordsForGood) {
  const fs::path dir = TestDir("tornheader");
  constexpr size_t kLen = 300000;
  {
    bool ok = false;
    LogStore s(Opts(dir), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    for (uint64_t id = 0; id < 40; ++id) {
      const std::string v = Val(id, kLen);
      ASSERT_EQ(s.Cache(Key(id), v.data(), v.size()), Status::kOk);
    }
    ASSERT_TRUE(s.Close());
  }
  // Segment 0 is the first one opened: it holds the first values.
  {
    std::fstream f(dir / "segments" / "G000000",
                   std::ios::in | std::ios::out | std::ios::binary);
    const std::string junk(512, '\x5a');
    f.write(junk.data(), junk.size());
  }
  size_t count = 0;
  {
    bool ok = false;
    LogStore s(Quiet(dir), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    count = s.Count();
    EXPECT_LT(count, 40u);
    EXPECT_GT(count, 0u);
    EXPECT_FALSE(Has(s, 0, kLen)) << "a record of the torn segment";
    for (uint64_t id = 0; id < 40; ++id) {
      std::string out;
      if (s.Range(Key(id), 0, 0, &out) == Status::kOk) EXPECT_EQ(out, Val(id, kLen));
    }
  }
  bool ok = false;
  LogStore s(Quiet(dir), &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  EXPECT_EQ(s.Count(), count);
  EXPECT_FALSE(Has(s, 0, kLen));
  fs::remove_all(dir);
}

// ---- REMOVE and PUT ordering with pending kills ----------------------------------
// C34: a second REMOVE of a key whose first REMOVE's kill is not durable yet
// answers kNotFound only once that kill is durable: killed right after it
// returned, the value stays removed.
TEST(LogStoreJournal, ASecondRemoveWaitsForTheFirstsKill) {
  const fs::path dir = TestDir("remove2");
  ASSERT_EQ(InChild([&] {
              LogStore* store = nullptr;
              auto o = Opts(dir);
              o.crash_hook = [&](const char* p) {
                if (std::strcmp(p, "remove.after_enqueue") != 0) return;
                static bool once = false;
                if (once) return;
                once = true;
                // The first REMOVE has enqueued its kill and not waited yet.
                const Status st = store->Remove(Key(1));
                ::_exit(st == Status::kNotFound ? 0 : 5);
              };
              bool ok = false;
              LogStore s(o, &ok);
              store = &s;
              if (!ok) return 2;
              const std::string v = Val(1, 7000);
              if (s.Cache(Key(1), v.data(), v.size()) != Status::kOk) return 3;
              s.Remove(Key(1));
              return 9;
            }),
            0);
  bool ok = false;
  LogStore s(Quiet(dir), &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  std::string out;
  EXPECT_EQ(s.Range(Key(1), 0, 0, &out), Status::kNotFound);
  fs::remove_all(dir);
}

// C31 and C32: a value whose read failed its checksum is a shadow; storing
// the key again kills the shadow before the PUT is acknowledged.
TEST(LogStoreJournal, APutOfAKeyWithShadowsKillsThemFirst) {
  const fs::path dir = TestDir("shadowput");
  constexpr size_t kLen = 9000;
  auto corrupt_first_record = [&] {
    std::fstream f(dir / "segments" / "G000000",
                   std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(8192 + 10);
    f.put('\x33');
  };
  {
    bool ok = false;
    LogStore s(Opts(dir), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    const std::string v = Val(1, kLen);
    ASSERT_EQ(s.Cache(Key(1), v.data(), v.size()), Status::kOk);
    corrupt_first_record();
    std::string out;
    EXPECT_EQ(s.Range(Key(1), 0, 0, &out), Status::kNotFound) << "C32";
    EXPECT_EQ(s.GetStats().shadows, 1u);
    const std::string w = Val(1, kLen, 1);
    ASSERT_EQ(s.Cache(Key(1), w.data(), w.size()), Status::kOk);
    EXPECT_EQ(s.GetStats().shadows, 0u) << "killed, then dropped";
    EXPECT_TRUE(Has(s, 1, kLen, 1));
  }
  fs::remove_all(dir);
  ASSERT_EQ(InChild([&] {
              auto o = Opts(dir);
              o.crash_hook = ExitAt("put.after_shadow_enqueue");
              bool ok = false;
              LogStore s(o, &ok);
              if (!ok) return 2;
              const std::string v = Val(1, kLen);
              if (s.Cache(Key(1), v.data(), v.size()) != Status::kOk) return 3;
              corrupt_first_record();
              std::string out;
              if (s.Range(Key(1), 0, 0, &out) != Status::kNotFound) return 4;
              const std::string w = Val(1, kLen, 1);
              s.Cache(Key(1), w.data(), w.size());
              return 9;
            }),
            0);
  bool ok = false;
  LogStore s(Quiet(dir), &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  std::string out;
  if (s.Range(Key(1), 0, 0, &out) == Status::kOk)
    EXPECT_EQ(out, Val(1, kLen, 1)) << "only the new value can be served";
  fs::remove_all(dir);
}

// C35: a reset interrupted at any step is repeated by the next open.
TEST(LogStoreJournal, AnInterruptedResetIsRepeated) {
  for (const char* point :
       {"reset.after_flag", "reset.after_headers", "reset.after_rename"}) {
    SCOPED_TRACE(point);
    const fs::path dir = TestDir("resetcrash");
    {
      bool ok = false;
      LogStore s(Opts(dir), &ok);
      ASSERT_TRUE(ok) << s.StartupError();
      for (uint64_t id = 0; id < 4; ++id) {
        const std::string v = Val(id, 5000);
        ASSERT_EQ(s.Cache(Key(id), v.data(), v.size()), Status::kOk);
      }
      ASSERT_EQ(s.Remove(Key(0)), Status::kOk);
      ASSERT_EQ(s.Remove(Key(1)), Status::kOk);
    }
    {
      std::fstream f(dir / "lse_journal",
                     std::ios::in | std::ios::out | std::ios::binary);
      f.seekp(100);
      f.put('\x55');  // damage block 1; block 2 stays valid
    }
    ASSERT_EQ(InChild([&] {
                auto o = Opts(dir);
                o.crash_hook = ExitAt(point);
                bool ok = false;
                LogStore s(o, &ok);
                return 9;
              }),
              0);
    bool ok = false;
    LogStore s(Quiet(dir), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    EXPECT_EQ(s.Count(), 0u) << "empty";
    EXPECT_TRUE(s.Healthy());
    const std::string v = Val(9, 5000);
    EXPECT_EQ(s.Cache(Key(9), v.data(), v.size()), Status::kOk);
    fs::remove_all(dir);
  }
}
