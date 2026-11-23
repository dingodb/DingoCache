// LogStore, the log-structured engine (docs/design/log-structured-engine.md),
// stage "sealing" (§14 stage 5): summaries and trailers, the clean close,
// summary-based recovery, and the crash points around them (§8.2: C9, C18,
// C21, C28), with the REMOVE and PUT orderings this stage's hooks expose
// (C31, C32, C34) and the interrupted reset (C35).
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
              Kill(0);  // killed: no Close
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
              Kill(ok ? 0 : 2);
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
                Kill(st == Status::kNotFound ? 0 : 5);
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

// ---- segment headers outside the store mutex (§6.7) ---------------------------------
namespace {
// Blocks the calling thread until `flag` is set or `ms` pass; true if set.
bool WaitFor(const std::atomic<bool>& flag, int ms) {
  for (int i = 0; i < ms && !flag.load(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  return flag.load();
}
thread_local int tl_role = 0;  // 1: the PUT that opens; 2: the PUT that finishes
}  // namespace

// A PUT that finishes its write while another PUT writes a new segment's
// header does not wait for that header (the mutex is free meanwhile); the
// opener publishes the segment even though a REMOVE cancelled its own PUT
// meanwhile, and the next PUT writes into it.
TEST(LogStoreHeaders, PutsFinishWhileASegmentHeaderIsWritten) {
  const fs::path dir = TestDir("unlockedheader");
  std::atomic<bool> release_preopen{false}, a_in_open{false}, b_done{false},
      release_a{false}, b_reserved{false};
  auto o = Quiet(dir);  // 4 MiB segments, one open
  o.crash_hook = [&](const char* p) {
    if (std::strcmp(p, "preopen.before_header") == 0) {
      WaitFor(release_preopen, 30000);  // no pre-opened segment: PUTs open
    } else if (std::strcmp(p, "put.after_reserve") == 0 && tl_role == 2) {
      b_reserved = true;
      WaitFor(a_in_open, 10000);
    } else if (std::strcmp(p, "open.before_header") == 0 && tl_role == 1) {
      a_in_open = true;
      WaitFor(release_a, 10000);
    }
  };
  bool ok = false;
  LogStore s(o, &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  constexpr size_t kLen = 1 << 20;  // three to a 4 MiB segment
  for (uint64_t id = 0; id < 2; ++id) {
    const std::string v = Val(id, kLen);
    ASSERT_EQ(s.Cache(Key(id), v.data(), v.size()), Status::kOk);
  }
  Status b_status = Status::kIOError, a_status = Status::kIOError;
  std::thread b([&] {  // the segment's last record: reserved, then held
    tl_role = 2;
    const std::string v = Val(2, kLen);
    b_status = s.Cache(Key(2), v.data(), v.size());
    b_done = true;
  });
  std::thread a([&] {  // does not fit: opens the next segment
    WaitFor(b_reserved, 10000);
    tl_role = 1;
    const std::string v = Val(3, kLen);
    a_status = s.Cache(Key(3), v.data(), v.size());
  });
  ASSERT_TRUE(WaitFor(a_in_open, 10000)) << "the opener never reached its header";
  EXPECT_TRUE(WaitFor(b_done, 5000))
      << "a PUT finishing its write waited for another PUT's header write";
  EXPECT_EQ(b_status, Status::kOk);
  // A REMOVE cancels the opener's own PUT while its header is written.
  EXPECT_EQ(s.Remove(Key(3)), Status::kOk);
  release_a = true;
  a.join();
  b.join();
  EXPECT_EQ(a_status, Status::kOk);  // began before the REMOVE
  std::string out;
  EXPECT_EQ(s.Range(Key(3), 0, 0, &out), Status::kNotFound);
  // The segment was published all the same: the next PUT goes into it.
  const std::string v = Val(4, kLen);
  EXPECT_EQ(s.Cache(Key(4), v.data(), v.size()), Status::kOk);
  EXPECT_TRUE(Has(s, 4, kLen));
  EXPECT_TRUE(Has(s, 2, kLen));
  EXPECT_EQ(s.GetStats().open_segments, 1u);
  release_preopen = true;
  ASSERT_TRUE(s.Close());
  fs::remove_all(dir);
}

// A clean close gives the pre-opened segment an empty trailer: nothing is
// walked at the next open. After a kill it is walked, holds nothing, and is
// free again (C39).
TEST(LogStoreHeaders, APreopenedSegmentAtCloseAndAtAKill) {
  constexpr size_t kLen = 500000;
  for (const bool kill : {false, true}) {
    SCOPED_TRACE(kill ? "kill" : "clean close");
    const fs::path dir = TestDir(kill ? "preopen_kill" : "preopen_close");
    ASSERT_EQ(InChild([&] {
                bool ok = false;
                LogStore s(Quiet(dir), &ok);
                if (!ok) return 2;
                for (uint64_t id = 0; id < 20; ++id) {
                  const std::string v = Val(id, kLen);
                  if (s.Cache(Key(id), v.data(), v.size()) != Status::kOk) return 3;
                }
                // Up to 10 s: the sealer may be slow (TSan, a loaded host).
                for (int i = 0; i < 1000 && s.GetStats().preopened == 0; ++i)
                  std::this_thread::sleep_for(std::chrono::milliseconds(10));
                if (s.GetStats().preopened != 1) return 4;
                if (kill) Kill(0);
                return s.Close() ? 0 : 5;
              }),
              0);
    bool ok = false;
    LogStore s(Quiet(dir), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    const auto r = s.GetRecoveryStats();
    if (kill) {
      EXPECT_GE(r.segments_walked, 2u) << "the open and the pre-opened segment";
      EXPECT_LE(r.segments_walked, 1u + 5) << "K + 5 (§8.4)";
    } else {
      EXPECT_EQ(r.segments_walked, 0u);
    }
    for (uint64_t id = 0; id < 20; ++id) EXPECT_TRUE(Has(s, id, kLen)) << id;
    // Every segment is accounted for: nothing leaked by the pre-open.
    const auto st = s.GetStats();
    EXPECT_EQ(st.segment_count, 8u);
    fs::remove_all(dir);
  }
}

// Crash points of the unlocked windows (C37-C39b): at each, the store opens,
// every REMOVE that returned stays removed, and every value served is one a
// PUT stored. Pre-opening is slowed so that PUTs also open segments
// themselves.
TEST(LogStoreHeaders, CrashPointsAroundSegmentHeaders) {
  for (const char* point : {"open.before_header", "open.after_header",
                            "preopen.before_header", "preopen.after_header",
                            "close.preopen_trailer"}) {
    SCOPED_TRACE(point);
    const fs::path dir = TestDir("headercrash");
    const std::string removed_log = dir.string() + ".removed";
    std::remove(removed_log.c_str());
    constexpr size_t kLen = 400000;
    ASSERT_EQ(InChild([&] {
                const int log = ::open(removed_log.c_str(),
                                       O_WRONLY | O_CREAT | O_APPEND, 0644);
                if (log < 0) return 6;
                auto o = Opts(dir);
                // The first time Activate or the sealer reaches it is before
                // any write: exit at a later one.
                auto exit_at = ExitAt(
                    point, std::strcmp(point, "close.preopen_trailer") == 0 ? 1 : 3);
                // The open.* points are reached only by a PUT that fills its
                // segment and finds no segment pre-opened for it. A writer
                // slower than the sealer's pre-open (under TSan, about 10
                // values of 400 KB per segment take longer than any fixed
                // delay here) always finds one: hold the pre-open until the
                // crash ends the child. The sealer then seals nothing either;
                // its queue holds two segments, and the third open comes at
                // the second switch, so the writer never waits for it.
                const bool open_point = std::strncmp(point, "open.", 5) == 0;
                o.crash_hook = [&](const char* p) {
                  if (std::strcmp(p, "preopen.before_header") == 0)
                    std::this_thread::sleep_for(
                        open_point ? std::chrono::milliseconds(60000)
                                   : std::chrono::milliseconds(20));
                  exit_at(p);
                };
                bool ok = false;
                LogStore s(o, &ok);
                if (!ok) return 2;
                for (uint64_t id = 0; id < 300; ++id) {
                  const std::string v = Val(id, kLen);
                  const Status st = s.Cache(Key(id), v.data(), v.size());
                  if (id % 10 == 3 && st == Status::kOk) {
                    if (s.Remove(Key(id)) != Status::kOk) return 5;
                    if (::write(log, &id, sizeof(id)) != sizeof(id)) return 7;
                  }
                }
                s.Close();  // reaches close.preopen_trailer
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
    EXPECT_LE(s.GetRecoveryStats().segments_walked, 1u + 5) << "K + 5 (§8.4)";
    for (uint64_t id = 0; id < 300; ++id) {
      std::string out;
      const Status st = s.Range(Key(id), 0, 0, &out);
      if (removed.count(id)) {
        EXPECT_EQ(st, Status::kNotFound) << "removed " << id;
      } else if (st == Status::kOk) {
        EXPECT_EQ(out, Val(id, kLen)) << id;
      }
    }
    // The store goes on: a PUT after the crash is stored.
    const std::string v = Val(9999, kLen);
    EXPECT_EQ(s.Cache(Key(9999), v.data(), v.size()), Status::kOk);
    fs::remove_all(dir);
  }
}
