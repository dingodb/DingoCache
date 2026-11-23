// LogStore, the log-structured engine (docs/design/log-structured-engine.md),
// stage "LogStore core": the StoreEngine contract, restarts after a clean
// close and after a kill, REMOVE durability, journal rules, and crash images
// built from the recorded I/O (simulated process kills and power losses).
#include "cache/log_store.h"

#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "cache/disk_slab_store.h"
#include "utils/crc32c.h"

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {

fs::path TestDir(const std::string& tag) {
  const fs::path p =
      fs::temp_directory_path() /
      ("dfkv_lse_" + tag + "_" +
       std::to_string(::testing::UnitTest::GetInstance()
                          ->current_test_info()->line()));
  fs::remove_all(p);
  return p;
}

LogStore::Options Opts(const fs::path& dir, uint64_t capacity = 64ull << 20) {
  LogStore::Options o;
  o.dir = dir.string();
  o.capacity_bytes = capacity;
  return o;
}

// The value of (key id, version): deterministic, so any served byte can be
// checked without keeping a copy. Sizes cover inline values and several
// blocks.
std::string Value(uint64_t id, uint32_t version) {
  static const size_t kSizes[] = {1, 100, 3000, 4096, 5000, 70000, 200001};
  const size_t len = kSizes[(id + version) % 7];
  std::string v(len, '\0');
  uint64_t x = id * 0x9e3779b97f4a7c15ULL + version;
  for (size_t i = 0; i < len; ++i) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    v[i] = static_cast<char>(x);
  }
  return v;
}
BlockKey Key(uint64_t id) { return BlockKey{id, id * 3 + 1, id % 3}; }

std::map<std::string, std::string> Snapshot(const fs::path& dir) {
  std::map<std::string, std::string> out;
  for (const auto& e : fs::recursive_directory_iterator(dir)) {
    if (!e.is_regular_file()) continue;
    std::ifstream in(e.path(), std::ios::binary);
    out[fs::relative(e.path(), dir).string()] =
        std::string((std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());
  }
  return out;
}

}  // namespace

// ---- the StoreEngine contract ------------------------------------------------
TEST(LogStore, CreatesAStoreAndRoundTripsEveryReadCall) {
  const fs::path dir = TestDir("roundtrip");
  bool ok = false;
  LogStore s(Opts(dir), &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  EXPECT_STREQ(s.EngineName(), "log");
  EXPECT_TRUE(fs::exists(dir / "lse_super"));
  EXPECT_TRUE(fs::exists(dir / "lse_journal"));
  EXPECT_FALSE(fs::exists(dir / "lse_initializing"));
  for (uint64_t id = 0; id < 20; ++id) {
    const std::string v = Value(id, 0);
    ASSERT_EQ(s.Cache(Key(id), v.data(), v.size()), Status::kOk) << id;
  }
  EXPECT_EQ(s.Count(), 20u);
  uint64_t tenant0 = 0;
  for (uint64_t id = 0; id < 20; ++id) {
    const std::string v = Value(id, 0);
    if (Key(id).tenant_hash == 0) tenant0 += v.size();
    std::string out;
    size_t len = 0;
    ASSERT_EQ(s.Range(Key(id), 0, 0, &out, &len), Status::kOk);
    EXPECT_EQ(out, v);
    EXPECT_EQ(len, v.size());
    if (v.size() > 10) {
      ASSERT_EQ(s.Range(Key(id), 3, 7, &out), Status::kOk);
      EXPECT_EQ(out, v.substr(3, 7));
    }
    char small[64];
    size_t got = 0;
    ASSERT_EQ(s.RangeInto(Key(id), 0, 0, small, sizeof(small), &got),
              Status::kOk);
    EXPECT_EQ(std::string(small, got), v.substr(0, std::min<size_t>(64, v.size())));
    void* mem = nullptr;
    ASSERT_EQ(::posix_memalign(&mem, 4096, 1 << 20), 0);
    const char* data = nullptr;
    size_t n = 0;
    ASSERT_EQ(s.RangeDirect(Key(id), 0, 0, static_cast<char*>(mem), 1 << 20,
                            &data, &n),
              Status::kOk);
    EXPECT_EQ(std::string(data, n), v);
    ReadLease lease;
    ASSERT_EQ(s.RangeDirectPrep(Key(id), 1, 0, 1 << 20, &lease), Status::kOk);
    EXPECT_EQ(lease.value_len, v.size());
    if (v.size() == 1) {  // nothing after offset 1: an empty lease
      EXPECT_LT(lease.fd(), 0);
      std::free(mem);
      continue;
    }
    ASSERT_GE(lease.fd(), 0);
    ASSERT_EQ(::pread(lease.fd(), mem, lease.aligned_len,
                      static_cast<off_t>(lease.aligned_off)),
              static_cast<ssize_t>(lease.aligned_len));
    EXPECT_EQ(std::string(static_cast<char*>(mem) + lease.head,
                          lease.payload_len),
              v.substr(1));
    EXPECT_EQ(lease.value_len, v.size());
    std::free(mem);
    ValueMetadata meta;
    ASSERT_EQ(s.Lookup(Key(id), &meta), Status::kOk);
    EXPECT_EQ(meta.value_len, v.size());
    EXPECT_GE(meta.generation, 1u);
  }
  EXPECT_EQ(s.TenantUsedBytes(0), tenant0);
  std::string out;
  EXPECT_EQ(s.Range(Key(99), 0, 0, &out), Status::kNotFound);
  EXPECT_EQ(s.Range(Key(1), Value(1, 0).size() + 1, 0, &out),
            Status::kInvalid);
  fs::remove_all(dir);
}

TEST(LogStore, KeepFirstAndAPutAfterARemoveIsStored) {
  const fs::path dir = TestDir("keepfirst");
  bool ok = false;
  LogStore s(Opts(dir), &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  const std::string a = Value(1, 0), b = Value(1, 1);
  ASSERT_EQ(s.Cache(Key(1), a.data(), a.size()), Status::kOk);
  ValueMetadata first;
  ASSERT_EQ(s.Lookup(Key(1), &first), Status::kOk);
  ASSERT_EQ(s.Cache(Key(1), b.data(), b.size()), Status::kOk);
  std::string out;
  ASSERT_EQ(s.Range(Key(1), 0, 0, &out), Status::kOk);
  EXPECT_EQ(out, a) << "keep-first";
  EXPECT_EQ(s.Remove(Key(1)), Status::kOk);
  EXPECT_EQ(s.Remove(Key(1)), Status::kNotFound);
  EXPECT_EQ(s.Range(Key(1), 0, 0, &out), Status::kNotFound);
  ASSERT_EQ(s.Cache(Key(1), b.data(), b.size()), Status::kOk);
  ASSERT_EQ(s.Range(Key(1), 0, 0, &out), Status::kOk);
  EXPECT_EQ(out, b);
  ValueMetadata second;
  ASSERT_EQ(s.Lookup(Key(1), &second), Status::kOk);
  EXPECT_GT(second.generation, first.generation);
  fs::remove_all(dir);
}

TEST(LogStore, AValueLargerThanASegmentCanHoldIsInvalid) {
  const fs::path dir = TestDir("toolarge");
  bool ok = false;
  LogStore s(Opts(dir, 32ull << 20), &ok);  // 8 segments of 4 MiB
  ASSERT_TRUE(ok) << s.StartupError();
  EXPECT_EQ(s.MaxValueBytes(), (4ull << 20) - 16384);
  std::string big(s.MaxValueBytes() + 1, 'b');
  EXPECT_EQ(s.Cache(Key(1), big.data(), big.size()), Status::kInvalid);
  big.resize(s.MaxValueBytes());
  EXPECT_EQ(s.Cache(Key(2), big.data(), big.size()), Status::kOk);
  std::string out;
  ASSERT_EQ(s.Range(Key(2), 0, 0, &out), Status::kOk);
  EXPECT_EQ(out.size(), big.size());
  fs::remove_all(dir);
}

// A store written far past its capacity evicts its oldest values and keeps
// accepting writes (design §7); what it serves is always right.
TEST(LogStore, AFullStoreEvictsTheOldestAndStaysHealthy) {
  const fs::path dir = TestDir("full");
  bool ok = false;
  LogStore s(Opts(dir, 32ull << 20), &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  const std::string v(1 << 20, 'f');
  size_t stored = 0, full = 0;
  for (uint64_t id = 0; id < 120; ++id) {
    const Status st = s.Cache(Key(id), v.data(), v.size());
    ASSERT_TRUE(st == Status::kOk || st == Status::kCacheFull) << int(st);
    if (st == Status::kOk) ++stored;
    else ++full;
  }
  EXPECT_GT(stored, 100u) << "kCacheFull " << full;
  EXPECT_TRUE(s.Healthy());
  EXPECT_GT(s.Evictions(), 60u);
  EXPECT_LE(s.UsedBytes(), 32ull << 20);
  std::string out;
  EXPECT_EQ(s.Range(Key(0), 0, 0, &out), Status::kNotFound) << "the oldest went";
  ASSERT_EQ(s.Range(Key(119), 0, 0, &out), Status::kOk) << "the newest is there";
  EXPECT_EQ(out, v);
  fs::remove_all(dir);
}

TEST(LogStore, DirectWritesRoundTripAndRestart) {
  const fs::path dir = TestDir("direct");
  auto o = Opts(dir);
  o.direct_writes = true;
  {
    bool ok = false;
    LogStore s(o, &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    for (uint64_t id = 0; id < 12; ++id) {
      const std::string v = Value(id, 0);
      if (id % 2 == 0) {
        void* mem = nullptr;
        ASSERT_EQ(::posix_memalign(&mem, 4096, v.size() + 8192), 0);
        std::memcpy(mem, v.data(), v.size());
        ASSERT_EQ(s.CacheDirect(Key(id), static_cast<char*>(mem), v.size(),
                                v.size() + 8192),
                  Status::kOk);
        std::free(mem);
      } else {
        ASSERT_EQ(s.Cache(Key(id), v.data(), v.size()), Status::kOk);
      }
    }
  }
  bool ok = false;
  LogStore s(o, &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  for (uint64_t id = 0; id < 12; ++id) {
    std::string out;
    ASSERT_EQ(s.Range(Key(id), 0, 0, &out), Status::kOk) << id;
    EXPECT_EQ(out, Value(id, 0)) << id;
  }
  fs::remove_all(dir);
}

// ---- restarts ------------------------------------------------------------------
TEST(LogStore, ARestartAfterACleanCloseKeepsEverything) {
  const fs::path dir = TestDir("reopen");
  std::map<uint64_t, uint64_t> generations;
  {
    bool ok = false;
    LogStore s(Opts(dir), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    for (uint64_t id = 0; id < 30; ++id) {
      const std::string v = Value(id, 0);
      ASSERT_EQ(s.Cache(Key(id), v.data(), v.size()), Status::kOk);
    }
    ASSERT_EQ(s.Remove(Key(5)), Status::kOk);
    for (uint64_t id = 0; id < 30; ++id) {
      ValueMetadata m;
      if (s.Lookup(Key(id), &m) == Status::kOk) generations[id] = m.generation;
    }
    EXPECT_TRUE(s.Close());
  }
  bool ok = false;
  LogStore s(Opts(dir), &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  EXPECT_EQ(s.Count(), 29u);
  for (uint64_t id = 0; id < 30; ++id) {
    std::string out;
    if (id == 5) {
      EXPECT_EQ(s.Range(Key(id), 0, 0, &out), Status::kNotFound);
      continue;
    }
    ASSERT_EQ(s.Range(Key(id), 0, 0, &out), Status::kOk) << id;
    EXPECT_EQ(out, Value(id, 0));
    ValueMetadata m;
    ASSERT_EQ(s.Lookup(Key(id), &m), Status::kOk);
    EXPECT_EQ(m.generation, generations[id]) << "generation survives a restart";
  }
  // New values get new generations, never an old one.
  const std::string v = Value(100, 0);
  ASSERT_EQ(s.Cache(Key(100), v.data(), v.size()), Status::kOk);
  ValueMetadata m;
  ASSERT_EQ(s.Lookup(Key(100), &m), Status::kOk);
  for (const auto& g : generations) EXPECT_NE(m.generation, g.second);
  fs::remove_all(dir);
}

// A process killed at any point: what it acknowledged is there after the
// restart (completed writes survive a kill), and a REMOVE it acknowledged
// stays done.
TEST(LogStore, AKilledProcessLosesNothingItAcknowledged) {
  const fs::path dir = TestDir("killed");
  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    bool ok = false;
    LogStore s(Opts(dir), &ok);
    if (!ok) ::_exit(2);
    for (uint64_t id = 0; id < 25; ++id) {
      const std::string v = Value(id, 0);
      if (s.Cache(Key(id), v.data(), v.size()) != Status::kOk) ::_exit(3);
    }
    if (s.Remove(Key(3)) != Status::kOk) ::_exit(4);
    if (s.Remove(Key(7)) != Status::kOk) ::_exit(4);
    ::_exit(0);  // no Close, no destructors: a kill
  }
  int status = 0;
  ASSERT_EQ(::waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
  bool ok = false;
  LogStore s(Opts(dir), &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  EXPECT_EQ(s.Count(), 23u);
  for (uint64_t id = 0; id < 25; ++id) {
    std::string out;
    if (id == 3 || id == 7) {
      EXPECT_EQ(s.Range(Key(id), 0, 0, &out), Status::kNotFound) << id;
      continue;
    }
    ASSERT_EQ(s.Range(Key(id), 0, 0, &out), Status::kOk) << id;
    EXPECT_EQ(out, Value(id, 0)) << id;
  }
  EXPECT_EQ(s.GetRecoveryStats().records_killed, 2u);
  fs::remove_all(dir);
}

// Crash points by name: the child _exits there, the parent reopens.
TEST(LogStore, CrashPointsDuringInitPutAndRemove) {
  struct Case {
    const char* point;
    int nth;              // exit at this occurrence of the point
    bool expect_value;    // key 1 readable after the restart
    bool expect_removed;  // key 0 gone after the restart
  };
  const std::vector<Case> cases = {
      {"init.after_marker", 1, false, false},
      {"init.after_super", 1, false, false},
      {"put.after_reserve", 2, false, false},  // key 1's PUT: nothing written
      {"put.after_write", 2, true, false},     // written, not acknowledged
      {"remove.after_enqueue", 1, true, false},
      {"remove.after_durable", 1, true, true},
      {"activate.after_clean", 1, false, false},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(c.point);
    const fs::path dir = TestDir(std::string("crash_") + c.point);
    const pid_t child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
      int seen = 0;
      auto o = Opts(dir);
      o.crash_hook = [&](const char* point) {
        if (std::strcmp(point, c.point) == 0 && ++seen == c.nth) ::_exit(0);
      };
      bool ok = false;
      LogStore s(o, &ok);
      if (!ok) ::_exit(2);
      const std::string v0 = Value(0, 0), v1 = Value(1, 0);
      if (s.Cache(Key(0), v0.data(), v0.size()) != Status::kOk) ::_exit(3);
      if (s.Cache(Key(1), v1.data(), v1.size()) != Status::kOk) ::_exit(3);
      s.Remove(Key(0));
      ::_exit(9);  // the point was not reached
    }
    int status = 0;
    ASSERT_EQ(::waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFEXITED(status));
    ASSERT_EQ(WEXITSTATUS(status), 0) << "the crash point was not reached";
    bool ok = false;
    LogStore s(Opts(dir), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    std::string out;
    const Status got1 = s.Range(Key(1), 0, 0, &out);
    if (c.expect_value) {
      ASSERT_EQ(got1, Status::kOk);
      EXPECT_EQ(out, Value(1, 0));
    } else if (got1 == Status::kOk) {
      EXPECT_EQ(out, Value(1, 0)) << "if anything is served, it is right";
    }
    const Status got0 = s.Range(Key(0), 0, 0, &out);
    if (c.expect_removed) EXPECT_EQ(got0, Status::kNotFound);
    else if (got0 == Status::kOk) EXPECT_EQ(out, Value(0, 0));
    fs::remove_all(dir);
  }
}

// ---- what is on disk -----------------------------------------------------------
TEST(LogStore, AnOpenedStoreThatIsNotActivatedChangesNothing) {
  const fs::path dir = TestDir("asfound");
  {
    bool ok = false;
    LogStore s(Opts(dir), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    const std::string v = Value(1, 0);
    ASSERT_EQ(s.Cache(Key(1), v.data(), v.size()), Status::kOk);
  }
  const auto before = Snapshot(dir);
  {
    auto o = Opts(dir);
    o.defer_activation = true;
    bool ok = false;
    LogStore s(o, &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    EXPECT_EQ(s.Count(), 1u);
    std::string out;
    EXPECT_EQ(s.Range(Key(1), 0, 0, &out), Status::kIOError) << "not serving";
    const std::string v = Value(2, 0);
    EXPECT_EQ(s.Cache(Key(2), v.data(), v.size()), Status::kIOError);
  }
  EXPECT_TRUE(Snapshot(dir) == before) << "the open wrote something";
  fs::remove_all(dir);
}

TEST(LogStore, ForeignContentIsRefusedAndLeftAlone) {
  const fs::path dir = TestDir("foreign");
  fs::create_directories(dir);
  std::ofstream(dir / "notes.txt") << "keep me";
  bool ok = true;
  LogStore s(Opts(dir), &ok);
  EXPECT_FALSE(ok);
  EXPECT_NE(s.StartupError().find("notes.txt"), std::string::npos)
      << s.StartupError();
  EXPECT_TRUE(fs::exists(dir / "notes.txt"));
  EXPECT_FALSE(fs::exists(dir / "lse_super"));
  fs::remove_all(dir);

  // Files a file system check left in lost+found: refused too. An empty
  // lost+found is not content.
  fs::create_directories(dir / "lost+found");
  {
    bool fine = false;
    LogStore empty_lf(Opts(dir), &fine);
    EXPECT_TRUE(fine) << empty_lf.StartupError();
  }
  fs::remove_all(dir);
  fs::create_directories(dir / "lost+found");
  std::ofstream(dir / "lost+found" / "#4711") << "x";
  bool opened = true;
  LogStore lf(Opts(dir), &opened);
  EXPECT_FALSE(opened);
  EXPECT_NE(lf.StartupError().find("lost+found"), std::string::npos)
      << lf.StartupError();
  EXPECT_FALSE(fs::exists(dir / "lse_super"));
  EXPECT_TRUE(fs::exists(dir / "lost+found" / "#4711"));
  fs::remove_all(dir);
}

// The downgrade case (design §10.3): the slab engine, unchanged since
// 2.33.3, refuses a log store's directory and changes nothing in it.
TEST(LogStore, TheSlabEngineRefusesALogDirectoryWithoutChangingIt) {
  const fs::path dir = TestDir("downgrade");
  {
    bool ok = false;
    LogStore s(Opts(dir), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    const std::string v = Value(1, 0);
    ASSERT_EQ(s.Cache(Key(1), v.data(), v.size()), Status::kOk);
  }
  const auto before = Snapshot(dir);
  DiskSlabStore::Options so;
  so.dir = dir.string();
  so.capacity_bytes = 1ull << 30;
  bool ok = true;
  DiskSlabStore slab(so, &ok);
  EXPECT_FALSE(ok);
  EXPECT_NE(slab.StartupError().find("different or incomplete store layout"),
            std::string::npos)
      << slab.StartupError();
  EXPECT_TRUE(Snapshot(dir) == before);
  fs::remove_all(dir);
}

// A damaged payload: rejected by the walk after an unclean exit, and a miss
// at read time after a clean close (summaries are trusted; reads verify).
TEST(LogStore, ADamagedPayloadIsAMissAtRecoveryAndAtReadTime) {
  for (const bool clean : {false, true}) {
    SCOPED_TRACE(clean ? "clean close" : "killed");
    const fs::path dir = TestDir(clean ? "corrupt_clean" : "corrupt_kill");
    const std::string v = Value(6, 0);  // several blocks
    ASSERT_GT(v.size(), 4096u);
    const pid_t child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
      bool ok = false;
      LogStore s(Opts(dir), &ok);
      if (!ok) ::_exit(2);
      const std::string w = Value(8, 0);
      if (s.Cache(Key(6), v.data(), v.size()) != Status::kOk ||
          s.Cache(Key(8), w.data(), w.size()) != Status::kOk)
        ::_exit(3);
      if (clean && !s.Close()) ::_exit(4);
      ::_exit(0);
    }
    int status = 0;
    ASSERT_EQ(::waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFEXITED(status));
    ASSERT_EQ(WEXITSTATUS(status), 0);
    // The first record of the first opened segment: its payload starts one
    // block after its header block, which is one block into the segment.
    auto flip = [&](uint64_t offset) {
      for (const auto& e : fs::directory_iterator(dir / "segments")) {
        std::fstream f(e.path(), std::ios::in | std::ios::out | std::ios::binary);
        char head[16];
        f.seekg(4096);
        f.read(head, 4);
        if (std::string(head, 4) != "DLR1") continue;
        char c;
        f.seekg(static_cast<std::streamoff>(offset));
        f.read(&c, 1);
        c ^= 0x40;
        f.seekp(static_cast<std::streamoff>(offset));
        f.write(&c, 1);
        return true;
      }
      return false;
    };
    ASSERT_TRUE(flip(8192 + 100));
    bool ok = false;
    {
      LogStore s(Opts(dir), &ok);
      ASSERT_TRUE(ok) << s.StartupError();
      EXPECT_EQ(s.GetRecoveryStats().records_rejected, clean ? 0u : 1u);
      // Killed: the open segment has no trailer and is walked.
      EXPECT_EQ(s.GetRecoveryStats().segments_walked == 0, clean);
      std::string out;
      EXPECT_EQ(s.Range(Key(6), 0, 0, &out), Status::kNotFound);
      ASSERT_EQ(s.Range(Key(8), 0, 0, &out), Status::kOk)
          << "the walk went on past the damaged record";
      EXPECT_EQ(out, Value(8, 0));
      if (clean) EXPECT_EQ(s.GetStats().read_crc_failures, 1u);
    }
    fs::remove_all(dir);
  }
}

// ---- the journal -----------------------------------------------------------------
namespace {
// Records every I/O call; passes them through to the real system calls.
class RecordingIo : public lse::IoBackend {
 public:
  struct Op {
    enum Kind { kCreate, kWrite, kSync, kTruncate, kFallocate, kRename,
                kMkdir, kUnlink } kind;
    std::string path, to;
    uint64_t offset = 0, len = 0;
    std::string data;
    bool dsync = false;
  };
  RecordingIo() : sys_(lse::MakeSystemIoBackend()) {}

  int Open(const std::string& path, int flags, mode_t mode) override {
    const bool existed = fs::exists(path);
    const int fd = sys_->Open(path, flags, mode);
    if (fd >= 0) {
      std::lock_guard<std::mutex> lk(mu_);
      if ((flags & O_CREAT) && !existed) ops_.push_back({Op::kCreate, path});
      else if (flags & O_TRUNC)
        ops_.push_back({Op::kTruncate, path, "", 0, 0});
    }
    return fd;
  }
  int Close(int fd) override { return sys_->Close(fd); }
  ssize_t Pwritev(int fd, const struct iovec* iov, int n, off_t offset,
                  bool dsync) override {
    const ssize_t w = sys_->Pwritev(fd, iov, n, offset, dsync);
    if (w > 0) {
      Op op{Op::kWrite, PathOf(fd)};
      op.offset = static_cast<uint64_t>(offset);
      op.dsync = dsync;
      size_t left = static_cast<size_t>(w);
      for (int i = 0; i < n && left; ++i) {
        const size_t take = std::min(left, iov[i].iov_len);
        op.data.append(static_cast<const char*>(iov[i].iov_base), take);
        left -= take;
      }
      op.len = op.data.size();
      std::lock_guard<std::mutex> lk(mu_);
      ops_.push_back(std::move(op));
    }
    return w;
  }
  ssize_t Pread(int fd, void* buf, size_t len, off_t offset) override {
    return sys_->Pread(fd, buf, len, offset);
  }
  // A sync covers the writes that completed before it was called: its
  // position is taken before the call.
  int Fdatasync(int fd) override {
    Record({Op::kSync, PathOf(fd)});
    return sys_->Fdatasync(fd);
  }
  int Fsync(int fd) override {
    Record({Op::kSync, PathOf(fd)});
    return sys_->Fsync(fd);
  }
  int Fallocate(int fd, off_t offset, off_t len) override {
    const int rc = sys_->Fallocate(fd, offset, len);
    if (rc == 0) Record({Op::kFallocate, PathOf(fd), "", 0,
                         static_cast<uint64_t>(offset + len)});
    return rc;
  }
  int Ftruncate(int fd, off_t len) override {
    const int rc = sys_->Ftruncate(fd, len);
    if (rc == 0)
      Record({Op::kTruncate, PathOf(fd), "", 0, static_cast<uint64_t>(len)});
    return rc;
  }
  int Rename(const std::string& from, const std::string& to) override {
    const int rc = sys_->Rename(from, to);
    if (rc == 0) Record({Op::kRename, from, to});
    return rc;
  }
  int Mkdir(const std::string& path, mode_t mode) override {
    const int rc = sys_->Mkdir(path, mode);
    if (rc == 0) Record({Op::kMkdir, path});
    return rc;
  }
  int Unlink(const std::string& path) override {
    const int rc = sys_->Unlink(path);
    if (rc == 0) Record({Op::kUnlink, path});
    return rc;
  }

  std::vector<Op> Ops() {
    std::lock_guard<std::mutex> lk(mu_);
    return ops_;
  }
  size_t Size() {
    std::lock_guard<std::mutex> lk(mu_);
    return ops_.size();
  }

 private:
  static std::string PathOf(int fd) {
    char link[64], target[4096];
    std::snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
    const ssize_t n = ::readlink(link, target, sizeof(target) - 1);
    return n > 0 ? std::string(target, static_cast<size_t>(n)) : std::string();
  }
  void Record(Op op) {
    std::lock_guard<std::mutex> lk(mu_);
    ops_.push_back(std::move(op));
  }
  std::unique_ptr<lse::IoBackend> sys_;
  std::mutex mu_;
  std::vector<Op> ops_;
};

// How a crash treats the recorded operations not yet durable.
enum class Crash {
  kKill,     // a process kill: every operation in the prefix is kept
  kPower,    // a power loss: each one dropped, kept, or torn at random
  kPowerAll  // a power loss that keeps none of them
};

// Builds the directory a crash after the first `prefix` recorded operations
// could leave (design §8.2). For a power loss, durable are: a write with
// RWF_DSYNC (that write only), and whatever a later fsync/fdatasync of the
// same file covers; a directory entry only after its directory's sync.
// Everything else is dropped, kept, or torn (an arbitrary subset of its
// 512-byte sectors), the survivors applied in a random order after the
// durable operations.
void BuildImage(const std::vector<RecordingIo::Op>& ops, size_t prefix,
                Crash crash, std::mt19937_64* rng, const fs::path& from,
                const fs::path& to) {
  fs::remove_all(to);
  fs::create_directories(to);
  const std::string src = fs::weakly_canonical(from).string();
  auto map = [&](const std::string& p) {
    const std::string canon = fs::weakly_canonical(p).string();
    return (to / fs::relative(canon, src)).string();
  };
  auto parent = [](const std::string& p) {
    return fs::path(p).parent_path().string();
  };
  auto covered = [&](size_t i) {
    const auto& op = ops[i];
    if (crash == Crash::kKill) return true;
    if (op.kind == RecordingIo::Op::kWrite && op.dsync) return true;
    const bool entry = op.kind == RecordingIo::Op::kCreate ||
                       op.kind == RecordingIo::Op::kRename ||
                       op.kind == RecordingIo::Op::kMkdir ||
                       op.kind == RecordingIo::Op::kUnlink;
    const std::string target = entry ? parent(op.kind == RecordingIo::Op::kRename
                                                  ? op.to
                                                  : op.path)
                                     : op.path;
    for (size_t j = i + 1; j < prefix; ++j)
      if (ops[j].kind == RecordingIo::Op::kSync &&
          fs::weakly_canonical(ops[j].path) == fs::weakly_canonical(target))
        return true;
    return false;
  };
  auto keep_loose = [&] { return crash == Crash::kPower && (*rng)() % 2 == 0; };
  auto write = [&](const RecordingIo::Op& op, size_t from_byte, size_t len) {
    const std::string path = map(op.path);
    if (!fs::exists(path)) return;
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(static_cast<std::streamoff>(op.offset + from_byte));
    f.write(op.data.data() + from_byte, static_cast<std::streamsize>(len));
  };
  std::vector<size_t> loose;
  for (size_t i = 0; i < prefix; ++i) {
    const auto& op = ops[i];
    const bool keep = covered(i) || keep_loose();
    switch (op.kind) {
      case RecordingIo::Op::kWrite:
        if (covered(i)) write(op, 0, op.len);
        else loose.push_back(i);
        break;
      case RecordingIo::Op::kCreate:
        if (keep) std::ofstream(map(op.path), std::ios::app | std::ios::binary);
        break;
      case RecordingIo::Op::kMkdir:
        if (keep) fs::create_directories(map(op.path));
        break;
      case RecordingIo::Op::kUnlink:
        if (keep) fs::remove(map(op.path));
        break;
      case RecordingIo::Op::kRename:
        if (keep && fs::exists(map(op.path)))
          fs::rename(map(op.path), map(op.to));
        break;
      case RecordingIo::Op::kTruncate:
      case RecordingIo::Op::kFallocate:
        if (keep && fs::exists(map(op.path))) {
          const uint64_t size = fs::file_size(map(op.path));
          if (op.kind == RecordingIo::Op::kTruncate || op.len > size)
            fs::resize_file(map(op.path), op.len);
        }
        break;
      case RecordingIo::Op::kSync:
        break;
    }
  }
  if (crash == Crash::kPowerAll) return;
  std::shuffle(loose.begin(), loose.end(), *rng);
  for (size_t i : loose) {
    const auto& op = ops[i];
    switch ((*rng)() % 3) {
      case 0: break;  // lost
      case 1: write(op, 0, op.len); break;
      default:  // torn: any subset of its sectors
        for (size_t at = 0; at < op.len; at += 512)
          if ((*rng)() % 2) write(op, at, std::min<size_t>(512, op.len - at));
    }
  }
}
}  // namespace

TEST(LogStore, EveryJournalWriteIsASynchronousBlock) {
  const fs::path dir = TestDir("journal_sync");
  auto io = std::make_shared<RecordingIo>();
  {
    auto o = Opts(dir);
    o.io = io;
    bool ok = false;
    LogStore s(o, &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    for (uint64_t id = 0; id < 80; ++id) {
      const std::string v = Value(id, 0);
      ASSERT_EQ(s.Cache(Key(id), v.data(), v.size()), Status::kOk);
    }
    for (uint64_t id = 0; id < 80; ++id) ASSERT_EQ(s.Remove(Key(id)), Status::kOk);
  }
  size_t journal_writes = 0;
  for (const auto& op : io->Ops()) {
    if (op.kind != RecordingIo::Op::kWrite ||
        fs::path(op.path).filename() != "lse_journal")
      continue;
    ++journal_writes;
    EXPECT_TRUE(op.dsync) << "journal writes are RWF_DSYNC";
    EXPECT_EQ(op.len, 4096u) << "one block per write";
    EXPECT_EQ(op.offset % 4096, 0u);
  }
  EXPECT_GE(journal_writes, 2u);
  fs::remove_all(dir);
}

// A damaged block inside the journal's durable range: the kills it held are
// unknown, so the store comes up empty rather than serve a removed value.
TEST(LogStore, JournalDamageInsideTheDurableRangeResetsTheStore) {
  const fs::path dir = TestDir("journal_damage");
  {
    bool ok = false;
    LogStore s(Opts(dir), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    for (uint64_t id = 0; id < 4; ++id) {
      const std::string v = Value(id, 0);
      ASSERT_EQ(s.Cache(Key(id), v.data(), v.size()), Status::kOk);
    }
    ASSERT_EQ(s.Remove(Key(0)), Status::kOk);  // block 1
    ASSERT_EQ(s.Remove(Key(1)), Status::kOk);  // block 2
  }
  ASSERT_GE(fs::file_size(dir / "lse_journal"), 8192u);
  {
    std::fstream f(dir / "lse_journal",
                   std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(100);
    f.put('\x55');  // damage block 1, block 2 stays valid
  }
  bool ok = false;
  {
    LogStore s(Opts(dir), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    EXPECT_TRUE(s.GetRecoveryStats().journal_damaged);
    EXPECT_EQ(s.Count(), 0u) << "reset to empty";
    std::string out;
    EXPECT_EQ(s.Range(Key(0), 0, 0, &out), Status::kNotFound);
  }
  LogStore again(Opts(dir), &ok);
  ASSERT_TRUE(ok) << again.StartupError();
  EXPECT_FALSE(again.GetRecoveryStats().journal_damaged) << "reset once";
  EXPECT_EQ(again.Count(), 0u);
  fs::remove_all(dir);
}

// A torn tail is not damage: it is cut off at activation, so two ordinary
// crashes in a row never reset the store (design §4.5, C36).
TEST(LogStore, ATornJournalTailIsCutOffNotTakenForDamage) {
  const fs::path dir = TestDir("journal_tail");
  {
    bool ok = false;
    LogStore s(Opts(dir), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    for (uint64_t id = 0; id < 3; ++id) {
      const std::string v = Value(id, 0);
      ASSERT_EQ(s.Cache(Key(id), v.data(), v.size()), Status::kOk);
    }
    ASSERT_EQ(s.Remove(Key(0)), Status::kOk);
  }
  const uint64_t durable = fs::file_size(dir / "lse_journal");
  {
    std::ofstream f(dir / "lse_journal", std::ios::app | std::ios::binary);
    std::string garbage(4096, '\x7f');
    f.write(garbage.data(), garbage.size());  // a torn, unacknowledged block
  }
  bool ok = false;
  {
    LogStore s(Opts(dir), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    EXPECT_FALSE(s.GetRecoveryStats().journal_damaged);
    EXPECT_EQ(fs::file_size(dir / "lse_journal"), durable) << "cut off";
    ASSERT_EQ(s.Remove(Key(1)), Status::kOk);
  }
  LogStore s(Opts(dir), &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  EXPECT_FALSE(s.GetRecoveryStats().journal_damaged);
  EXPECT_EQ(s.Count(), 1u);
  std::string out;
  EXPECT_EQ(s.Range(Key(0), 0, 0, &out), Status::kNotFound);
  EXPECT_EQ(s.Range(Key(1), 0, 0, &out), Status::kNotFound);
  ASSERT_EQ(s.Range(Key(2), 0, 0, &out), Status::kOk);
  fs::remove_all(dir);
}

// ---- crash images ------------------------------------------------------------------
namespace {
// One client operation of a recorded workload. `start`, `end`, `commit` are
// positions in the recorded I/O: before the call, after it returned, and at
// the moment a PUT stored its value (the on_commit hook, on the PUT's thread).
struct ImageOp {
  bool put = false;
  uint64_t id = 0;
  uint32_t version = 0;  // every PUT attempt has its own bytes
  size_t start = 0, end = 0;
  bool wrote = false;
  uint64_t generation = 0;
  size_t commit = 0;
  size_t reserve = SIZE_MAX;  // when a PUT reserved its space
  size_t locked = SIZE_MAX;   // when a REMOVE took the store's mutex
};
thread_local ImageOp* tl_op = nullptr;

LogStore::Options RecordedOpts(const fs::path& dir,
                               const std::shared_ptr<RecordingIo>& io) {
  auto o = Opts(dir, 64ull << 20);
  o.io = io;
  o.sync_ms = 2;  // values become durable within the workload
  o.on_commit = [io](const BlockKey&, uint64_t generation) {
    if (tl_op == nullptr) return;
    tl_op->wrote = true;
    tl_op->generation = generation;
    tl_op->commit = io->Size();
  };
  o.on_reserve = [io](const BlockKey&) {
    if (tl_op != nullptr) tl_op->reserve = io->Size();
  };
  o.on_remove = [io](const BlockKey&) {
    if (tl_op != nullptr) tl_op->locked = io->Size();
  };
  return o;
}

void RunOp(LogStore& s, ImageOp* op, RecordingIo& io,
           const std::function<std::string(uint64_t, uint32_t)>& value = Value) {
  tl_op = op;
  op->start = io.Size();
  if (op->put) {
    const std::string v = value(op->id, op->version);
    s.Cache(Key(op->id), v.data(), v.size());
  } else {
    s.Remove(Key(op->id));
  }
  op->end = io.Size();
  tl_op = nullptr;
}

// What an image may serve (design §8.1): only bytes some PUT stored (I1);
// never a value stored before a REMOVE that returned (I2); never a value
// older than one a PUT stored and acknowledged (I3). After a kill, a key
// whose operations did not overlap and none of which was in flight is in
// the state the last of them left.
// A value evicted from the index (its generation) at a recorded position.
struct ImageEviction {
  BlockKey key;
  uint64_t generation = 0;
  size_t at = 0;
};
// A RETAIN of the workload and what it answered (2..5: protected).
struct ImageRetain {
  uint64_t id = 0;
  size_t start = 0, end = 0;
  uint8_t result = 0;
};
// How a workload's images are opened and judged.
struct ImageCheck {
  std::function<std::string(uint64_t, uint32_t)> value = Value;
  const std::vector<ImageEviction>* evictions = nullptr;
  std::function<LogStore::Options(const fs::path&)> open = [](const fs::path& p) {
    return Opts(p, 64ull << 20);
  };
  // false: a caller whose coverage depends on thread timing sums the counts
  // over several seeds and checks the minimums itself.
  bool check_minimums = true;
  // I4: a value a RETAIN protected (the workload had no protected eviction)
  // is served by every image once a sync covered its record.
  const std::vector<ImageRetain>* retains = nullptr;
  // Power losses that keep nothing loose, at these prefixes, besides the
  // random ones (the moments right after a victim's new header).
  std::vector<size_t> power_prefixes;
};

// A protected value for the I4 check: its PUT, the recorded write of its
// record, and the first sync of that file after it.
struct ProtectedValue {
  const ImageOp* put = nullptr;
  size_t durable_at = SIZE_MAX;
};

// How much the image checks actually looked at.
struct ImageStats {
  int images = 0;
  int strict_keys = 0;       // kill images: keys held to the last state
  int holes = 0;             // images recovered past rejected records
  int uncommitted_served = 0;
  int power_served = 0;      // values served by power-loss images
  int protected_checked = 0; // protected values held to I4
  int protected_power = 0;   // of those, in power-loss images (kills keep all)
};

void CheckImage(LogStore& s, const std::vector<ImageOp>& ops, uint64_t keys,
                size_t prefix, Crash crash, ImageStats* stats,
                const ImageCheck& check,
                const std::vector<ProtectedValue>& protected_values) {
  const auto& Value = check.value;
  for (const auto& pv : protected_values) {
    if (pv.durable_at > prefix) continue;
    // Held to it only while nothing else touched the key up to the crash.
    bool alone = true;
    for (const auto& e : ops)
      if (&e != pv.put && e.id == pv.put->id && e.start < prefix &&
          e.end > pv.put->start)
        alone = false;
    if (!alone) continue;
    std::string out;
    const Status st = s.Range(Key(pv.put->id), 0, 0, &out);
    EXPECT_TRUE(st == Status::kOk && out == Value(pv.put->id, pv.put->version))
        << "key " << pv.put->id << ": a protected value whose record was "
           "synced is gone (I4)";
    ++stats->protected_checked;
    if (crash != Crash::kKill) ++stats->protected_power;
  }
  for (uint64_t id = 0; id < keys; ++id) {
    std::string out;
    const Status st = s.Range(Key(id), 0, 0, &out);
    const ImageOp* served = nullptr;
    if (st == Status::kOk && crash != Crash::kKill) ++stats->power_served;
    if (st == Status::kOk) {
      const ImageOp* written = nullptr;  // written, not (yet) committed
      for (const auto& e : ops) {
        if (!e.put || e.id != id || out != Value(id, e.version)) continue;
        if (e.wrote && e.commit <= prefix) served = &e;
        else if (e.start < prefix) written = &e;
      }
      if (served == nullptr) {
        // A PUT that wrote its record and had not committed at the crash
        // (C5), or one a REMOVE cancelled while it wrote. Its value may come
        // back only if no REMOVE that started after its reservation had
        // returned (that REMOVE killed its location, I2), and no PUT that
        // reserved after it had stored and been acknowledged (that PUT
        // waited for the kill first, I3). Positions are strict: equal ones
        // say nothing about the order.
        ASSERT_NE(written, nullptr)
            << "key " << id << " served bytes no PUT wrote (I1)";
        ASSERT_NE(written->reserve, SIZE_MAX);
        for (const auto& e : ops) {
          if (e.id != id || e.end > prefix) continue;
          if (!e.put)
            EXPECT_FALSE(e.locked != SIZE_MAX && e.locked > written->reserve)
                << "key " << id << ": an uncommitted PUT's value came back "
                   "after a later REMOVE returned (I2)";
          else if (e.wrote)
            EXPECT_FALSE(e.reserve != SIZE_MAX && e.reserve > written->reserve)
                << "key " << id << ": an uncommitted PUT's value came back "
                   "after a later PUT was acknowledged (I3)";
        }
        ++stats->uncommitted_served;
        continue;
      }
      for (const auto& e : ops) {
        if (e.id != id || e.end > prefix || &e == served) continue;
        if (!e.put)
          EXPECT_FALSE(e.locked != SIZE_MAX && served->commit < e.locked)
              << "key " << id << ": a value stored before a REMOVE that "
                 "returned came back (I2)";
        if (e.put && e.wrote)
          EXPECT_FALSE(e.generation > served->generation)
              << "key " << id << ": an older value than an acknowledged "
                 "later one (I3)";
      }
    }
    if (crash != Crash::kKill) continue;
    std::vector<const ImageOp*> mine;
    for (const auto& e : ops)
      if (e.id == id && e.start < prefix) mine.push_back(&e);
    bool clean = true;
    for (size_t k = 0; k < mine.size(); ++k) {
      if (mine[k]->end > prefix) clean = false;
      if (k > 0 && mine[k - 1]->end > mine[k]->start) clean = false;
    }
    if (!clean) continue;
    const ImageOp* expect = nullptr;
    for (const ImageOp* e : mine) {
      if (e->put && e->wrote) expect = e;
      if (!e->put) expect = nullptr;
    }
    // An evicted value may come back, or not (§8.1).
    bool evicted = false;
    if (expect != nullptr && check.evictions != nullptr)
      for (const auto& ev : *check.evictions)
        if (ev.key == Key(id) && ev.generation == expect->generation &&
            ev.at <= prefix)
          evicted = true;
    if (!evicted || served != nullptr) EXPECT_EQ(served, expect) << "key " << id;
    ++stats->strict_keys;
  }
}

// The values the workload's RETAINs protected: the key's last PUT that
// returned before the RETAIN began, with nothing else of the key in between,
// and the first sync of the file its record was written to.
std::vector<ProtectedValue> ProtectedValues(
    const std::vector<RecordingIo::Op>& recorded,
    const std::vector<ImageOp>& ops, const ImageCheck& check) {
  std::vector<ProtectedValue> out;
  if (check.retains == nullptr) return out;
  std::set<const ImageOp*> seen;
  for (const auto& r : *check.retains) {
    if (r.result < 2 || r.result > 5) continue;
    const ImageOp* p = nullptr;
    for (const auto& e : ops)
      if (e.id == r.id && e.put && e.wrote && e.end <= r.start &&
          (p == nullptr || e.start > p->start))
        p = &e;
    if (p == nullptr || !seen.insert(p).second) continue;
    bool alone = true;
    for (const auto& e : ops)
      if (&e != p && e.id == r.id && e.start < r.end && e.end > p->start)
        alone = false;
    const std::string v = check.value(p->id, p->version);
    if (!alone || v.size() < 3000) continue;  // a size the search can find
    ProtectedValue pv;
    pv.put = p;
    for (size_t i = p->start; i < p->commit && i < recorded.size(); ++i) {
      if (recorded[i].kind != RecordingIo::Op::kWrite ||
          recorded[i].data.find(v) == std::string::npos)
        continue;
      for (size_t j = i + 1; j < recorded.size(); ++j)
        if (recorded[j].kind == RecordingIo::Op::kSync &&
            recorded[j].path == recorded[i].path) {
          pv.durable_at = j + 1;
          break;
        }
      break;
    }
    if (pv.durable_at != SIZE_MAX) out.push_back(pv);
  }
  return out;
}

ImageStats CheckImages(const std::vector<RecordingIo::Op>& recorded,
                       const std::vector<ImageOp>& ops, uint64_t keys,
                       const fs::path& dir, const fs::path& image, int images,
                       uint64_t seed, const ImageCheck& check = ImageCheck()) {
  ImageStats stats;
  std::mt19937_64 rng(seed);
  const std::vector<ProtectedValue> protected_values =
      ProtectedValues(recorded, ops, check);
  const int rounds = images + static_cast<int>(check.power_prefixes.size());
  for (int round = 0; round < rounds; ++round) {
    const bool chosen = round >= images;
    const size_t prefix = chosen ? check.power_prefixes[round - images]
                                 : 1 + rng() % recorded.size();
    const Crash crash = chosen || round % 3 == 2 ? Crash::kPowerAll
                        : round % 3 == 0         ? Crash::kKill
                                                 : Crash::kPower;
    SCOPED_TRACE("prefix " + std::to_string(prefix) + " crash " +
                 std::to_string(static_cast<int>(crash)));
    BuildImage(recorded, prefix, crash, &rng, dir, image);
    // A crash during the first initialization leaves the marker (or not even
    // the superblock): the open starts over, and nothing was acknowledged.
    const bool initialized = fs::exists(image / "lse_super") &&
                             !fs::exists(image / "lse_initializing");
    bool ok = false;
    LogStore s(check.open(image), &ok);
    EXPECT_TRUE(ok) << s.StartupError();
    if (!ok) return stats;
    if (!initialized) {
      EXPECT_EQ(s.Count(), 0u);
      continue;
    }
    ++stats.images;
    if (s.GetRecoveryStats().records_rejected > 0) ++stats.holes;
    CheckImage(s, ops, keys, prefix, crash, &stats, check, protected_values);
    if (::testing::Test::HasFatalFailure()) return stats;
  }
  // The checks did look at something (a change that silently emptied them
  // fails here). A caller whose coverage depends on thread timing sums the
  // counts over several seeds and checks them itself.
  EXPECT_GT(stats.images, images / 2);
  if (check.check_minimums) {
    EXPECT_GT(stats.strict_keys, 0);
    EXPECT_GT(stats.holes, 0);
    EXPECT_GT(stats.power_served, 0) << "the power images' I2/I3 checks saw values";
  }
  if (check.retains != nullptr)
    EXPECT_GT(stats.protected_power, 0)
        << "the I4 check saw protected values in power-loss images";
  ::testing::Test::RecordProperty("images", stats.images);
  ::testing::Test::RecordProperty("strict_keys", stats.strict_keys);
  ::testing::Test::RecordProperty("holes", stats.holes);
  ::testing::Test::RecordProperty("uncommitted_served", stats.uncommitted_served);
  ::testing::Test::RecordProperty("power_served", stats.power_served);
  ::testing::Test::RecordProperty("protected_checked", stats.protected_checked);
  ::testing::Test::RecordProperty("protected_power", stats.protected_power);
  return stats;
}
}  // namespace

// One writer: PUTs and REMOVEs, the I/O recorded; then many crash points.
TEST(LogStore, CrashImagesServeOnlyRightValues) {
  const fs::path dir = TestDir("images");
  const fs::path image = TestDir("images_out");
  auto io = std::make_shared<RecordingIo>();
  constexpr uint64_t kKeys = 24;
  std::vector<ImageOp> ops(300);
  {
    bool ok = false;
    LogStore s(RecordedOpts(dir, io), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    std::mt19937_64 rng(3);
    std::map<uint64_t, uint32_t> next_version;
    for (auto& op : ops) {
      op.id = rng() % kKeys;
      op.put = rng() % 3 != 0;
      op.version = next_version[op.id]++;
      RunOp(s, &op, *io);
    }
  }
  CheckImages(io->Ops(), ops, kKeys, dir, image, 150, 5);
  fs::remove_all(dir);
  fs::remove_all(image);
}

// Several writers on the same keys: concurrent reservations leave holes
// that recovery resynchronizes past (C6), REMOVEs and PUTs of one key race
// (C34, and the PUT that must wait for a REMOVE's kill, I3).
TEST(LogStore, ConcurrentCrashImagesServeOnlyRightValues) {
  const fs::path dir = TestDir("cimages");
  const fs::path image = TestDir("cimages_out");
  auto io = std::make_shared<RecordingIo>();
  constexpr uint64_t kKeys = 4;
  constexpr int kThreads = 4, kPerThread = 120;
  std::vector<std::vector<ImageOp>> per(kThreads, std::vector<ImageOp>(kPerThread));
  {
    // Slow journal blocks and PUTs that pause between reservation and write
    // widen the races: REMOVEs cancel PUTs mid-write, and later PUTs of the
    // key arrive while those KILLs are queued.
    auto o = RecordedOpts(dir, io);
    o.crash_hook = [](const char* p) {
      thread_local std::mt19937_64 rng(std::hash<std::thread::id>()(
          std::this_thread::get_id()));
      if (std::strcmp(p, "journal.before_write") == 0)
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
      else if (std::strcmp(p, "put.after_reserve") == 0)
        std::this_thread::sleep_for(std::chrono::microseconds(rng() % 1500));
    };
    bool ok = false;
    LogStore s(o, &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t)
      threads.emplace_back([&, t] {
        std::mt19937_64 rng(100 + t);
        for (int i = 0; i < kPerThread; ++i) {
          ImageOp& op = per[t][i];
          op.id = rng() % kKeys;
          op.put = rng() % 2 == 0;
          op.version = static_cast<uint32_t>(t * 100000 + i);  // unique bytes
          RunOp(s, &op, *io);
        }
      });
    for (auto& th : threads) th.join();
  }
  std::vector<ImageOp> ops;
  for (const auto& v : per) ops.insert(ops.end(), v.begin(), v.end());
  // Which branches a crash point exercises depends on how the threads
  // interleaved: more crash points of the same recording until each one has
  // been (kill images of keys at rest, holes, the power images' checks, and
  // the T1 branch: an uncommitted PUT's value served).
  ImageStats total;
  auto covered = [&] {
    return total.strict_keys > 0 && total.holes > 0 && total.power_served > 0 &&
           total.uncommitted_served > 0;
  };
  ImageCheck check;
  check.check_minimums = false;
  int seeds = 0;
  for (uint64_t seed = 7; seed < 47 && !covered(); ++seed, ++seeds) {
    const ImageStats st =
        CheckImages(io->Ops(), ops, kKeys, dir, image, 300, seed, check);
    total.strict_keys += st.strict_keys;
    total.holes += st.holes;
    total.power_served += st.power_served;
    total.uncommitted_served += st.uncommitted_served;
    if (HasFailure()) break;
  }
  RecordProperty("seeds", seeds);
  EXPECT_GT(total.strict_keys, 0) << "no kill image held a key at rest";
  EXPECT_GT(total.holes, 0) << "no image recovered past a rejected record";
  EXPECT_GT(total.power_served, 0) << "the power images' I2/I3 checks saw no values";
  EXPECT_GT(total.uncommitted_served, 0) << "no image served an uncommitted PUT's value";
  fs::remove_all(dir);
  fs::remove_all(image);
}

// M1 of the stage-2 review, deterministically: a PUT that starts while a
// REMOVE's KILL of the key is still queued is not acknowledged before that
// KILL is durable. Otherwise a power loss right after the acknowledgement
// (the new value not yet synced, the KILL not yet written) brings the
// removed value back, older than the acknowledged one (I3).
TEST(LogStore, APutAfterARemoveWaitsForTheRemovesKill) {
  const fs::path dir = TestDir("putwaitskill");
  const fs::path image = TestDir("putwaitskill_out");
  auto io = std::make_shared<RecordingIo>();
  const std::string g1 = Value(1, 0), g2 = Value(1, 1);
  {
    bool ok = false;
    LogStore s(RecordedOpts(dir, io), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    ASSERT_EQ(s.Cache(Key(1), g1.data(), g1.size()), Status::kOk);
  }  // closed: g1 is synced
  std::mutex mu;
  std::condition_variable cv;
  bool hold = true, held = false;
  auto o = RecordedOpts(dir, io);
  o.sync_ms = 3600000;  // nothing synced behind the test's back
  o.crash_hook = [&](const char* p) {
    if (std::strcmp(p, "journal.before_write") != 0) return;
    std::unique_lock<std::mutex> lk(mu);
    held = true;
    cv.notify_all();
    cv.wait(lk, [&] { return !hold; });
  };
  size_t acked_at = 0;
  std::atomic<bool> acked{false};
  {
    bool ok = false;
    LogStore s(o, &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    std::thread remover([&] { s.Remove(Key(1)); });
    {
      std::unique_lock<std::mutex> lk(mu);
      cv.wait(lk, [&] { return held; });  // the KILL's block is not written
    }
    std::thread putter([&] {
      EXPECT_EQ(s.Cache(Key(1), g2.data(), g2.size()), Status::kOk);
      acked_at = io->Size();
      acked = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    const bool early = acked.load();
    {
      std::lock_guard<std::mutex> lk(mu);
      hold = false;
      cv.notify_all();
    }
    putter.join();
    remover.join();
    EXPECT_FALSE(early) << "acknowledged while the REMOVE's KILL was queued";
    if (early) {
      // The power loss right after that acknowledgement.
      std::mt19937_64 rng(1);
      BuildImage(io->Ops(), acked_at, Crash::kPowerAll, &rng, dir, image);
      bool reopened = false;
      LogStore img(Opts(image, 64ull << 20), &reopened);
      ASSERT_TRUE(reopened) << img.StartupError();
      std::string out;
      EXPECT_FALSE(img.Range(Key(1), 0, 0, &out) == Status::kOk && out == g1)
          << "the removed value came back after a later PUT was acknowledged";
    }
  }
  fs::remove_all(dir);
  fs::remove_all(image);
}

// S1 of the stage-2 review: a segment whose last value goes while a read
// still pins it becomes free when the pin drops, not lost.
TEST(LogStore, ASegmentFreedWhilePinnedIsReusedOnceUnpinned) {
  const fs::path dir = TestDir("pinnedfree");
  auto o = Opts(dir, 32ull << 20);  // 8 segments of three 1 MiB values
  o.cleaner = false;
  bool ok = false;
  LogStore s(o, &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  const std::string v(1 << 20, 'p');
  for (uint64_t id = 0; id < 7; ++id)  // two segments full, a third open
    ASSERT_EQ(s.Cache(Key(id), v.data(), v.size()), Status::kOk);
  for (int i = 0; i < 100 && s.GetStats().segments_sealed_now < 2; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  auto lease = std::make_unique<ReadLease>();
  ASSERT_EQ(s.RangeDirectPrep(Key(0), 0, 0, 2 << 20, lease.get()), Status::kOk);
  const uint32_t free_before = s.GetStats().segments_free;
  for (uint64_t id = 0; id < 3; ++id)  // the first segment's values
    ASSERT_EQ(s.Remove(Key(id)), Status::kOk);
  EXPECT_EQ(s.GetStats().segments_free, free_before) << "pinned: not free yet";
  lease.reset();
  EXPECT_EQ(s.GetStats().segments_free, free_before + 1)
      << "free once the pin is gone";
  fs::remove_all(dir);
}

// The read-time check (§5.2): a whole-value read that fails the payload CRC
// is a miss, and its location a shadow that a REMOVE, or a new PUT of the
// key, kills before returning, so the old bytes cannot come back even if the
// damage was only on the way.
TEST(LogStore, AReadThatFailsItsCheckLeavesAShadowThatIsKilled) {
  for (const bool by_put : {false, true}) {
    SCOPED_TRACE(by_put ? "killed by a PUT" : "killed by a REMOVE");
    const fs::path dir = TestDir(by_put ? "readcrc_put" : "readcrc_remove");
    const std::string v = Value(6, 0), w = Value(6, 1);
    ASSERT_GT(v.size(), 4096u);
    const pid_t child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
      bool ok = false;
      LogStore s(Opts(dir), &ok);
      if (!ok) ::_exit(2);
      if (s.Cache(Key(6), v.data(), v.size()) != Status::kOk) ::_exit(3);
      // Damage the payload under the running store, read, then repair it:
      // the record on disk is whole again.
      const fs::path seg = dir / "segments" / "G000000";
      char c;
      {
        std::fstream f(seg, std::ios::in | std::ios::out | std::ios::binary);
        f.seekg(8192 + 10);
        f.read(&c, 1);
        f.seekp(8192 + 10);
        f.put(static_cast<char>(c ^ 0x20));
      }
      std::string out;
      if (s.Range(Key(6), 0, 0, &out) != Status::kNotFound) ::_exit(4);
      {
        std::fstream f(seg, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(8192 + 10);
        f.put(c);
      }
      if (by_put) {
        if (s.Cache(Key(6), w.data(), w.size()) != Status::kOk) ::_exit(5);
      } else if (s.Remove(Key(6)) != Status::kNotFound) {
        ::_exit(5);
      }
      ::_exit(0);  // killed
    }
    int status = 0;
    ASSERT_EQ(::waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFEXITED(status));
    ASSERT_EQ(WEXITSTATUS(status), 0);
    bool ok = false;
    LogStore s(Opts(dir), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    std::string out;
    if (by_put) {
      ASSERT_EQ(s.Range(Key(6), 0, 0, &out), Status::kOk);
      EXPECT_EQ(out, w) << "the new value, not the old one";
    } else {
      EXPECT_EQ(s.Range(Key(6), 0, 0, &out), Status::kNotFound);
    }
    fs::remove_all(dir);
  }
}

// T1 of the stage-2 review, the cancelled-flight half of M1: a PUT P that a
// REMOVE cancels while it writes leaves its record on disk, killed only
// once the REMOVE's KILL is durable. A later PUT Q of the key must not be
// acknowledged before that, or a power loss (P's record synced, Q's not, the
// KILL not written) brings P's value back after Q was acknowledged.
TEST(LogStore, APutAfterACancelledPutWaitsForItsKill) {
  const fs::path dir = TestDir("cancelledput");
  const fs::path image = TestDir("cancelledput_out");
  auto io = std::make_shared<RecordingIo>();
  const std::string gp = Value(1, 0), gq = Value(1, 1);
  std::mutex mu;
  std::condition_variable cv;
  bool hold_put = true, put_held = false, hold_journal = false, journal_held = false;
  auto o = RecordedOpts(dir, io);
  o.crash_hook = [&](const char* p) {
    std::unique_lock<std::mutex> lk(mu);
    if (std::strcmp(p, "put.after_reserve") == 0 && hold_put) {
      put_held = true;
      cv.notify_all();
      cv.wait(lk, [&] { return !hold_put; });
    } else if (std::strcmp(p, "journal.before_write") == 0 && hold_journal) {
      journal_held = true;
      cv.notify_all();
      cv.wait(lk, [&] { return !hold_journal; });
    }
  };
  size_t acked_at = 0;
  std::atomic<bool> acked{false};
  {
    bool ok = false;
    LogStore s(o, &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    std::thread p([&] { EXPECT_EQ(s.Cache(Key(1), gp.data(), gp.size()), Status::kOk); });
    {
      std::unique_lock<std::mutex> lk(mu);
      cv.wait(lk, [&] { return put_held; });  // P reserved, not written
      hold_journal = true;
    }
    std::thread r([&] { EXPECT_EQ(s.Remove(Key(1)), Status::kOk); });  // cancels P
    {
      std::unique_lock<std::mutex> lk(mu);
      cv.wait(lk, [&] { return journal_held; });  // R's KILL not written
      hold_put = false;
      cv.notify_all();
    }
    p.join();  // P wrote its record and was not published
    std::this_thread::sleep_for(std::chrono::milliseconds(50));  // P synced
    std::thread q([&] {
      EXPECT_EQ(s.Cache(Key(1), gq.data(), gq.size()), Status::kOk);
      acked_at = io->Size();
      acked = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    const bool early = acked.load();
    {
      std::lock_guard<std::mutex> lk(mu);
      hold_journal = false;
      cv.notify_all();
    }
    q.join();
    r.join();
    EXPECT_FALSE(early) << "acknowledged while the cancelled PUT's KILL was queued";
    if (early) {
      std::mt19937_64 rng(1);
      BuildImage(io->Ops(), acked_at, Crash::kPowerAll, &rng, dir, image);
      bool reopened = false;
      LogStore img(Opts(image, 64ull << 20), &reopened);
      ASSERT_TRUE(reopened) << img.StartupError();
      std::string out;
      EXPECT_FALSE(img.Range(Key(1), 0, 0, &out) == Status::kOk && out == gp)
          << "the cancelled PUT's value came back after a later PUT was acknowledged";
    }
  }
  fs::remove_all(dir);
  fs::remove_all(image);
}

// S1 of the round-2 review: when the PUT that waited for a REMOVE's KILL
// drops the last shadow of a segment (before the REMOVE wakes), the segment
// is freed then, not stranded until a restart.
TEST(LogStore, ASegmentWhoseLastShadowAPutDropsIsReused) {
  const fs::path dir = TestDir("lastshadow");
  constexpr size_t kLen = 1 << 20;  // three to a 4 MiB segment
  const std::string v(kLen, 'v');
  std::mutex mu;
  std::condition_variable cv;
  bool hold = false, held = false;
  auto o = Opts(dir, 32ull << 20);
  o.cleaner = false;
  o.crash_hook = [&](const char* p) {
    if (std::strcmp(p, "remove.after_wait") != 0) return;
    std::unique_lock<std::mutex> lk(mu);
    if (!hold) return;
    held = true;
    cv.notify_all();
    cv.wait(lk, [&] { return !hold; });
  };
  bool ok = false;
  LogStore s(o, &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  for (uint64_t id = 0; id < 7; ++id)  // two segments full, a third open
    ASSERT_EQ(s.Cache(Key(id), v.data(), v.size()), Status::kOk);
  for (int i = 0; i < 100 && s.GetStats().segments_sealed_now < 2; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  const uint32_t free_before = s.GetStats().segments_free;
  ASSERT_EQ(s.Remove(Key(0)), Status::kOk);
  ASSERT_EQ(s.Remove(Key(1)), Status::kOk);
  {
    std::lock_guard<std::mutex> lk(mu);
    hold = true;
  }
  std::thread r([&] { EXPECT_EQ(s.Remove(Key(2)), Status::kOk); });
  {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&] { return held; });
  }
  ASSERT_EQ(s.Cache(Key(2), v.data(), v.size()), Status::kOk);
  EXPECT_EQ(s.GetStats().segments_free, free_before + 1)
      << "freed by the PUT that dropped its last shadow";
  {
    std::lock_guard<std::mutex> lk(mu);
    hold = false;
    cv.notify_all();
  }
  r.join();
  EXPECT_EQ(s.GetStats().segments_free, free_before + 1);
  fs::remove_all(dir);
}

namespace {
// Fails every fdatasync of a segment file once armed.
class FailingSyncIo : public RecordingIo {
 public:
  int Fdatasync(int fd) override {
    char link[64], target[4096];
    std::snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
    const ssize_t n = ::readlink(link, target, sizeof(target) - 1);
    if (fail.load() && n > 0 &&
        std::string(target, static_cast<size_t>(n)).find("/segments/") !=
            std::string::npos) {
      errno = EIO;
      return -1;
    }
    return RecordingIo::Fdatasync(fd);
  }
  std::atomic<bool> fail{false};
};

uint32_t CleanFlag(const fs::path& dir) {
  std::ifstream f(dir / "lse_super", std::ios::binary);
  std::string bytes((std::istreambuf_iterator<char>(f)),
                    std::istreambuf_iterator<char>());
  lse::Superblock best;
  bool have = false;
  for (int copy = 0; copy < 2 && bytes.size() >= 8192; ++copy) {
    lse::Superblock sb;
    if (lse::DecodeSuperblock(reinterpret_cast<const uint8_t*>(bytes.data()) +
                                  copy * 4096,
                              true, &sb) == lse::DecodeError::kOk &&
        (!have || sb.sequence > best.sequence)) {
      best = sb;
      have = true;
    }
  }
  return have ? best.clean : 2;
}
}  // namespace

// S3 of the round-2 review: a failed periodic sync fails the store; a later
// sync that happens to succeed must not let Close record a clean close.
TEST(LogStore, AFailedPeriodicSyncFailsTheStore) {
  const fs::path dir = TestDir("syncfail");
  auto io = std::make_shared<FailingSyncIo>();
  auto o = Opts(dir);
  o.io = io;
  o.sync_ms = 2;
  {
    bool ok = false;
    LogStore s(o, &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    const std::string v = Value(1, 0);
    io->fail = true;
    ASSERT_EQ(s.Cache(Key(1), v.data(), v.size()), Status::kOk);
    for (int i = 0; i < 200 && s.Healthy(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    EXPECT_FALSE(s.Healthy()) << "the failed sync failed the store";
    io->fail = false;  // the next sync would succeed
    EXPECT_FALSE(s.Close());
  }
  EXPECT_EQ(CleanFlag(dir), 0u) << "not recorded as a clean close";
  fs::remove_all(dir);
}

// S4 of the round-2 review: the periodic sync covers a segment the writes
// just left (switched away from), not only the open one.
TEST(LogStore, ThePeriodicSyncCoversASegmentJustLeft) {
  const fs::path dir = TestDir("syncleft");
  auto io = std::make_shared<RecordingIo>();
  auto o = Opts(dir, 32ull << 20);
  o.io = io;
  o.sync_ms = 300;  // the four writes below land within one period
  bool ok = false;
  LogStore s(o, &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  const std::string v(1 << 20, 'l');
  for (uint64_t id = 0; id < 4; ++id)  // the fourth opens the second segment
    ASSERT_EQ(s.Cache(Key(id), v.data(), v.size()), Status::kOk);
  std::this_thread::sleep_for(std::chrono::milliseconds(800));
  const auto ops = io->Ops();  // before Close, which syncs everything
  size_t last_write = 0;
  for (size_t i = 0; i < ops.size(); ++i)
    if (fs::path(ops[i].path).filename() == "G000000" &&
        ops[i].kind == RecordingIo::Op::kWrite)
      last_write = i;
  bool synced_after = false;
  for (size_t i = last_write + 1; i < ops.size(); ++i)
    if (fs::path(ops[i].path).filename() == "G000000" &&
        ops[i].kind == RecordingIo::Op::kSync)
      synced_after = true;
  ASSERT_GT(last_write, 0u);
  EXPECT_TRUE(synced_after) << "the first segment's last writes were left unsynced";
  fs::remove_all(dir);
}

// Larger values than Value(): the workload must fill the store several times.
std::string ImageValue(uint64_t id, uint32_t version) {
  static const size_t kSizes[] = {1, 3000, 70000, 200001, 300000, 4096, 600000};
  std::string v = Value(id, version);
  v.resize(kSizes[(id + version) % 7]);
  uint64_t x = id * 31 + version;
  for (size_t i = 0; i < v.size(); i += 61) v[i] = static_cast<char>(x++);
  return v;
}

// A small store written several times past what it holds, with reads (the
// cleaner's second chance) and RETAINs: segments are sealed, reclaimed (relocating
// what was read, evicting the rest, and reclaiming inline), and the
// journal compacted; crash images of it, evictions allowed for.
TEST(LogStore, CrashImagesThroughSealingReclaimAndCompaction) {
  const fs::path dir = TestDir("simages");
  const fs::path image = TestDir("simages_out");
  auto io = std::make_shared<RecordingIo>();
  constexpr uint64_t kKeys = 256;
  std::vector<ImageOp> ops(1500);
  std::vector<ImageEviction> evictions;
  std::vector<ImageRetain> retains;
  std::mutex evict_mu;
  LogStore::Stats stats;
  {
    auto o = RecordedOpts(dir, io);
    o.capacity_bytes = 32ull << 20;
    o.journal_compact_bytes = 3 * 4096;
    o.on_evict = [&](const BlockKey& k, uint64_t g) {
      std::lock_guard<std::mutex> lk(evict_mu);
      evictions.push_back({k, g, io->Size()});
    };
    bool ok = false;
    LogStore s(o, &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    std::mt19937_64 rng(3);
    std::map<uint64_t, uint32_t> next_version;
    for (auto& op : ops) {
      op.id = rng() % kKeys;
      const int what = static_cast<int>(rng() % 10);
      if (what == 1) {  // protected: relocated at the next reclaim
        ImageRetain r;
        r.id = op.id;
        r.start = io->Size();
        r.result = s.Retain(Key(op.id), 600, 0,
                            static_cast<uint32_t>(::time(nullptr)));
        r.end = io->Size();
        retains.push_back(r);
        op.start = op.end = io->Size();
        op.id = kKeys;
        continue;
      }
      if (what == 0) {  // a read: referenced, relocated at the next reclaim
        std::string out;
        s.Range(Key(op.id), 0, 0, &out);
        op.start = op.end = io->Size();
        op.id = kKeys;  // not a key operation
        continue;
      }
      op.put = what >= 4;
      op.version = next_version[op.id]++;
      RunOp(s, &op, *io, ImageValue);
    }
    ASSERT_TRUE(s.Close());  // the counters of the asynchronous sealer too
    stats = s.GetStats();
  }
  EXPECT_EQ(stats.protected_evictions, 0u) << "I4 holds for every RETAIN";
  EXPECT_GT(stats.segments_sealed, 3u);
  EXPECT_GT(stats.segments_reclaimed, 2u);
  EXPECT_GT(stats.records_evicted, 10u);
  EXPECT_GT(stats.records_relocated, 0u);
  EXPECT_GT(stats.journal_compactions, 0u);
  ImageCheck check;
  check.value = ImageValue;
  check.evictions = &evictions;
  check.open = [](const fs::path& p) {
    auto o = Opts(p, 32ull << 20);
    o.cleaner = false;  // it would evict at once in a full store
    return o;
  };
  check.retains = &retains;
  CheckImages(io->Ops(), ops, kKeys, dir, image, 80, 11, check);
  fs::remove_all(dir);
  fs::remove_all(image);
}

// N1 of the #512 review: the same with several writers and a reader holding
// short leases, so reclaims give up on pinned victims and race PUTs and
// REMOVEs; the power losses right after each victim's new header are checked
// as well (C20 with copies of an attempt that gave up, MF1).
TEST(LogStore, ConcurrentCrashImagesThroughSealingAndReclaim) {
  const fs::path dir = TestDir("csimages");
  const fs::path image = TestDir("csimages_out");
  auto io = std::make_shared<RecordingIo>();
  constexpr uint64_t kKeys = 96;
  constexpr int kThreads = 3, kPerThread = 700;
  std::vector<std::vector<ImageOp>> per(kThreads, std::vector<ImageOp>(kPerThread));
  std::vector<std::vector<ImageRetain>> per_retains(kThreads);
  std::vector<ImageEviction> evictions;
  std::vector<size_t> headers;
  std::mutex mu;
  LogStore::Stats stats;
  {
    auto o = RecordedOpts(dir, io);
    o.capacity_bytes = 32ull << 20;
    o.sync_ms = 3600000;  // only seals and reclaims make values durable
    o.pin_wait_ms = 1;
    o.on_evict = [&](const BlockKey& k, uint64_t g) {
      std::lock_guard<std::mutex> lk(mu);
      evictions.push_back({k, g, io->Size()});
    };
    o.crash_hook = [&](const char* p) {
      if (std::strcmp(p, "reclaim.after_header") == 0 ||
          std::strcmp(p, "inline_reclaim.after_header") == 0) {
        std::lock_guard<std::mutex> lk(mu);
        headers.push_back(io->Size());
      } else if (std::strcmp(p, "relocate.after_copy") == 0) {
        // Slow relocations: leases taken meanwhile outlast the pin wait.
        std::this_thread::sleep_for(std::chrono::microseconds(2000));
      }
    };
    bool ok = false;
    LogStore s(o, &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    std::atomic<int> running{kThreads};
    std::thread reader([&] {
      std::mt19937_64 rng(99);
      while (running.load() > 0) {
        std::string out;
        s.Range(Key(rng() % kKeys), 0, 0, &out);  // referenced: a second chance
        // A quarter of the keys leased at a time, for some milliseconds.
        std::vector<ReadLease> leases(24);
        for (auto& l : leases) s.RangeDirectPrep(Key(rng() % kKeys), 0, 0, 1 << 20, &l);
        std::this_thread::sleep_for(std::chrono::microseconds(3000 + rng() % 6000));
      }
    });
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t)
      threads.emplace_back([&, t] {
        std::mt19937_64 rng(200 + t);
        for (int i = 0; i < kPerThread; ++i) {
          ImageOp& op = per[t][i];
          op.id = rng() % kKeys;
          const int what = static_cast<int>(rng() % 10);
          if (what < 2) {
            ImageRetain r;
            r.id = op.id;
            r.start = io->Size();
            r.result = s.Retain(Key(op.id), 600, 0,
                                static_cast<uint32_t>(::time(nullptr)));
            r.end = io->Size();
            per_retains[t].push_back(r);
            op.start = op.end = io->Size();
            op.id = kKeys;  // not a key operation
            continue;
          }
          op.put = what >= 4;
          op.version = static_cast<uint32_t>(t * 100000 + i);  // unique bytes
          RunOp(s, &op, *io, ImageValue);
        }
        --running;
      });
    for (auto& th : threads) th.join();
    reader.join();
    ASSERT_TRUE(s.Close());
    stats = s.GetStats();
  }
  EXPECT_EQ(stats.protected_evictions, 0u) << "I4 holds for every RETAIN";
  EXPECT_GT(stats.segments_reclaimed, 5u);
  EXPECT_GT(stats.records_relocated, 0u);
  EXPECT_GT(stats.victims_skipped_pins, 0u) << "no reclaim gave up on a pin";
  RecordProperty("skipped_pins", static_cast<int>(stats.victims_skipped_pins));
  std::vector<ImageOp> ops;
  for (const auto& v : per) ops.insert(ops.end(), v.begin(), v.end());
  std::vector<ImageRetain> retains;
  for (const auto& v : per_retains) retains.insert(retains.end(), v.begin(), v.end());
  ImageCheck check;
  check.value = ImageValue;
  check.evictions = &evictions;
  check.retains = &retains;
  check.open = [](const fs::path& p) {
    auto o = Opts(p, 32ull << 20);
    o.cleaner = false;
    return o;
  };
  // At most 60 of the header moments, spread over the run.
  for (size_t i = 0; i < headers.size(); i += std::max<size_t>(1, headers.size() / 60))
    check.power_prefixes.push_back(headers[i]);
  CheckImages(io->Ops(), ops, kKeys, dir, image, 90, 13, check);
  fs::remove_all(dir);
  fs::remove_all(image);
}

// MF1 of the #512 review (C20): a reclaim that relocated a protected value
// and then gave up on a victim still pinned leaves the copy unsynced. The
// victim's next header, by a later cleaner pass or an inline reclaim, gives
// up its records: the copy must be durable first. A power loss right after
// that header keeps the value.
TEST(LogStore, ACopyOfAReclaimThatGaveUpIsDurableBeforeTheVictimIsReused) {
  const fs::path dir = TestDir("gaveup");
  const fs::path image = TestDir("gaveup_out");
  auto io = std::make_shared<RecordingIo>();
  auto big = [](uint64_t id) {
    std::string v = Value(id, 0);
    v.resize(400000, static_cast<char>('a' + id % 26));
    return v;
  };
  LogStore* store = nullptr;
  std::unique_ptr<ReadLease> lease;
  std::atomic<int> phase{0};  // 1: leased; 2: the attempt gave up
  std::vector<size_t> headers;
  std::mutex mu;
  auto o = RecordedOpts(dir, io);
  o.capacity_bytes = 32ull << 20;
  o.sync_ms = 3600000;  // only seals and reclaims sync
  o.pin_wait_ms = 50;
  o.crash_hook = [&](const char* p) {
    if (std::strcmp(p, "relocate.after_copy") == 0 && phase.load() == 0) {
      // The protected value is copied; a read of its neighbour pins the
      // victim past the wait.
      lease = std::make_unique<ReadLease>();
      if (store->RangeDirectPrep(Key(1), 0, 0, 1 << 20, lease.get()) == Status::kOk)
        phase = 1;
      return;
    }
    if ((std::strcmp(p, "reclaim.after_header") == 0 ||
         std::strcmp(p, "inline_reclaim.after_header") == 0) &&
        phase.load() == 2) {
      std::lock_guard<std::mutex> lk(mu);
      headers.push_back(io->Size());
    }
  };
  LogStore::Stats stats;
  {
    bool ok = false;
    LogStore s(o, &ok);
    store = &s;
    ASSERT_TRUE(ok) << s.StartupError();
    const uint32_t now = static_cast<uint32_t>(::time(nullptr));
    ASSERT_EQ(s.Cache(Key(0), big(0).data(), big(0).size()), Status::kOk);
    ASSERT_EQ(s.Retain(Key(0), 600, 0, now), 2);  // kProtectedNew
    ASSERT_EQ(s.Cache(Key(1), big(1).data(), big(1).size()), Status::kOk);
    uint64_t id = 2;
    for (; id < 200 && phase.load() == 0; ++id)
      s.Cache(Key(id), big(id).data(), big(id).size());
    ASSERT_EQ(phase.load(), 1) << "the first segment was never reclaimed";
    for (int i = 0; i < 500 && s.GetStats().victims_skipped_pins == 0; ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ASSERT_GT(s.GetStats().victims_skipped_pins, 0u);
    phase = 2;
    lease.reset();
    // More writes: the victim is reused.
    for (uint64_t end = id + 60; id < end; ++id)
      s.Cache(Key(id), big(id).data(), big(id).size());
    stats = s.GetStats();
    ASSERT_TRUE(s.Close());
  }
  EXPECT_GT(stats.records_relocated, 0u);
  EXPECT_EQ(stats.protected_evictions, 0u);
  ASSERT_FALSE(headers.empty());
  std::mt19937_64 rng(1);
  for (size_t i = 0; i < headers.size() && i < 4; ++i) {
    SCOPED_TRACE("header " + std::to_string(i));
    BuildImage(io->Ops(), headers[i], Crash::kPowerAll, &rng, dir, image);
    auto io_opts = Opts(image, 32ull << 20);
    io_opts.cleaner = false;
    bool ok = false;
    LogStore img(io_opts, &ok);
    ASSERT_TRUE(ok) << img.StartupError();
    std::string out;
    EXPECT_TRUE(img.Range(Key(0), 0, 0, &out) == Status::kOk && out == big(0))
        << "the protected value is gone after the victim's new header";
  }
  fs::remove_all(dir);
  fs::remove_all(image);
}
