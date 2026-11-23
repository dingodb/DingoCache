// LogStore, the log-structured engine (docs/design/log-structured-engine.md),
// stage 4 (§14): several open client segments, CacheDirectBatch, leases and
// segment pins, and the verification of whole-value prepared reads through
// the node (§5.3, crash point C32).
#include <gtest/gtest.h>
#include <sys/uio.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "cache/kv_node_server.h"
#include "cache/log_store.h"
#include "log_store_test_util.h"
#include "transport/wire.h"

using namespace dfkv;      // NOLINT
using namespace lse_test;  // NOLINT
namespace fs = std::filesystem;

namespace {

struct Aligned {
  explicit Aligned(size_t n) : size(n) {
    void* p = nullptr;
    EXPECT_EQ(::posix_memalign(&p, 4096, n), 0);
    data.reset(static_cast<char*>(p));
  }
  struct Free {
    void operator()(char* p) const { std::free(p); }
  };
  std::unique_ptr<char, Free> data;
  size_t size;
};

// Counts the writes of more than one record (more than header and payload).
class CountingIo : public lse::IoBackend {
 public:
  CountingIo() : sys_(lse::MakeSystemIoBackend()) {}
  int Open(const std::string& p, int f, mode_t m) override { return sys_->Open(p, f, m); }
  int Close(int fd) override { return sys_->Close(fd); }
  ssize_t Pwritev(int fd, const struct iovec* iov, int n, off_t off,
                  bool dsync) override {
    if (n > 3) multi_record_writes.fetch_add(1);
    return sys_->Pwritev(fd, iov, n, off, dsync);
  }
  ssize_t Pread(int fd, void* b, size_t n, off_t off) override {
    return sys_->Pread(fd, b, n, off);
  }
  int Fdatasync(int fd) override { return sys_->Fdatasync(fd); }
  int Fsync(int fd) override { return sys_->Fsync(fd); }
  int Fallocate(int fd, off_t o, off_t l) override { return sys_->Fallocate(fd, o, l); }
  int Ftruncate(int fd, off_t l) override { return sys_->Ftruncate(fd, l); }
  int Rename(const std::string& a, const std::string& b) override {
    return sys_->Rename(a, b);
  }
  int Mkdir(const std::string& p, mode_t m) override { return sys_->Mkdir(p, m); }
  int Unlink(const std::string& p) override { return sys_->Unlink(p); }
  std::atomic<int> multi_record_writes{0};

 private:
  std::unique_ptr<lse::IoBackend> sys_;
};

}  // namespace

// ---- several open segments (§6.1, §4.7) --------------------------------------------
TEST(LogStoreStage4, SeveralOpenSegmentsTakeConcurrentWrites) {
  const fs::path dir = TestDir("kopen");
  constexpr size_t kLen = 300000;
  {
    auto o = Opts(dir, 64ull << 20);
    o.open_segments = 3;
    bool ok = false;
    LogStore s(o, &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    EXPECT_EQ(s.GetStats().open_segments, 3u);
    std::vector<std::thread> writers;
    std::atomic<int> failures{0};
    for (int t = 0; t < 4; ++t)
      writers.emplace_back([&, t] {
        for (uint64_t i = 0; i < 30; ++i) {
          const uint64_t id = t * 1000 + i;
          const std::string v = Val(id, kLen);
          if (s.Cache(Key(id), v.data(), v.size()) != Status::kOk) ++failures;
        }
      });
    for (auto& w : writers) w.join();
    EXPECT_EQ(failures.load(), 0);
    EXPECT_EQ(s.GetStats().open_segments, 3u);
    for (int t = 0; t < 4; ++t)
      for (uint64_t i = 0; i < 30; ++i) EXPECT_TRUE(Has(s, t * 1000 + i, kLen));
  }
  // Every open segment is sealed at the close.
  bool ok = false;
  LogStore s(Quiet(dir, 64ull << 20), &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  EXPECT_EQ(s.GetRecoveryStats().segments_walked, 0u);
  EXPECT_EQ(s.Count(), 120u);
  EXPECT_EQ(s.GetStats().open_segments, 1u) << "derived: 16 segments / 32, at least 1";
  fs::remove_all(dir);
}

// After a kill, the open segments are among the walked ones: K of them, the
// relocation segment and the sealer queue at most (§8.4).
TEST(LogStoreStage4, AKillWalksAtMostTheOpenSegmentsAndTheQueue) {
  const fs::path dir = TestDir("kopenkill");
  constexpr size_t kLen = 300000;
  ASSERT_EQ(InChild([&]() -> int {
              auto o = Opts(dir, 64ull << 20);
              o.open_segments = 3;
              bool ok = false;
              LogStore s(o, &ok);
              if (!ok) return 2;
              for (uint64_t id = 0; id < 100; ++id) {
                const std::string v = Val(id, kLen);
                if (s.Cache(Key(id), v.data(), v.size()) != Status::kOk) return 3;
              }
              Kill(0);
            }),
            0);
  bool ok = false;
  LogStore s(Quiet(dir, 64ull << 20), &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  EXPECT_LE(s.GetRecoveryStats().segments_walked, 3u + 1u + 2u);
  EXPECT_EQ(s.Count(), 100u);
  for (uint64_t id = 0; id < 100; id += 3) EXPECT_TRUE(Has(s, id, kLen)) << id;
  fs::remove_all(dir);
}

// ---- CacheDirectBatch (§6.4) ---------------------------------------------------------
TEST(LogStoreStage4, ABatchIsOneWriteWithCacheDirectsAnswers) {
  const fs::path dir = TestDir("batch");
  auto io = std::make_shared<CountingIo>();
  std::vector<std::unique_ptr<Aligned>> bufs;
  std::vector<StoreEngine::CacheBatchItem> items;
  auto add = [&](uint64_t id, size_t len, uint32_t version = 0) {
    const std::string v = Val(id, len, version);
    bufs.push_back(std::make_unique<Aligned>(len + 8192));
    std::memcpy(bufs.back()->data.get(), v.data(), v.size());
    items.push_back({Key(id), bufs.back()->data.get(), len, len + 8192});
  };
  const size_t sizes[] = {1, 100, 4000, 4096, 5000, 70000, 200001};
  for (uint64_t id = 0; id < 40; ++id) add(id, sizes[id % 7]);
  add(3, 777, 1);            // the same key again: the first one is kept
  add(500, 9000, 0);         // stored before the batch: keep-first
  {
    auto o = Opts(dir);
    o.io = io;
    bool ok = false;
    LogStore s(o, &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    const std::string before = Val(500, 1234);
    ASSERT_EQ(s.Cache(Key(500), before.data(), before.size()), Status::kOk);
    items.push_back({Key(600), nullptr, 10, 10});                    // no data
    items.push_back({Key(601), bufs[0]->data.get(), s.MaxValueBytes() + 1,
                     s.MaxValueBytes() + 1});                        // too large
    const int writes_before = io->multi_record_writes.load();
    const std::vector<Status> got = s.CacheDirectBatch(items);
    ASSERT_EQ(got.size(), items.size());
    for (size_t i = 0; i < 42; ++i) EXPECT_EQ(got[i], Status::kOk) << i;
    EXPECT_EQ(got[42], Status::kInvalid);
    EXPECT_EQ(got[43], Status::kInvalid);
    EXPECT_EQ(io->multi_record_writes.load() - writes_before, 1)
        << "the 40 fresh records in one write";
    for (uint64_t id = 0; id < 40; ++id) EXPECT_TRUE(Has(s, id, sizes[id % 7])) << id;
    EXPECT_TRUE(Has(s, 500, 1234)) << "keep-first";
    ValueMetadata m;
    ASSERT_EQ(s.Lookup(Key(3), &m), Status::kOk);
    EXPECT_EQ(m.value_len, sizes[3]) << "the batch's first value of the key";
  }
  bool ok = false;
  LogStore s(Quiet(dir), &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  for (uint64_t id = 0; id < 40; ++id) EXPECT_TRUE(Has(s, id, sizes[id % 7])) << id;
  fs::remove_all(dir);
}

// A batch larger than the room left in the open segment continues in the
// next one; a batch into a full store gets kCacheFull only for what found no
// room, and never stores an item twice.
TEST(LogStoreStage4, ABatchSpillsIntoTheNextSegment) {
  const fs::path dir = TestDir("batchspill");
  auto o = Opts(dir);
  bool ok = false;
  LogStore s(o, &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  constexpr size_t kLen = 1 << 20;
  std::vector<std::unique_ptr<Aligned>> bufs;
  std::vector<StoreEngine::CacheBatchItem> items;
  for (uint64_t id = 0; id < 10; ++id) {
    const std::string v = Val(id, kLen);
    bufs.push_back(std::make_unique<Aligned>(kLen));
    std::memcpy(bufs.back()->data.get(), v.data(), v.size());
    items.push_back({Key(id), bufs.back()->data.get(), kLen, kLen});
  }
  const auto got = s.CacheDirectBatch(items);
  for (uint64_t id = 0; id < 10; ++id) {
    ASSERT_EQ(got[id], Status::kOk) << id;
    EXPECT_TRUE(Has(s, id, kLen)) << id;
  }
  EXPECT_EQ(s.Count(), 10u) << "4 MiB segments hold three: it spilled";
  fs::remove_all(dir);
}

// A crash around a batch's write: every value served is right (I1), and the
// store opens.
TEST(LogStoreStage4, CrashPointsAroundABatchWrite) {
  for (const char* point : {"batch.before_write", "batch.after_write"}) {
    SCOPED_TRACE(point);
    const fs::path dir = TestDir("batchcrash");
    ASSERT_EQ(InChild([&]() -> int {
                auto o = Opts(dir);
                o.crash_hook = ExitAt(point);
                bool ok = false;
                LogStore s(o, &ok);
                if (!ok) return 2;
                std::vector<std::unique_ptr<Aligned>> bufs;
                std::vector<StoreEngine::CacheBatchItem> items;
                for (uint64_t id = 0; id < 20; ++id) {
                  const std::string v = Val(id, 30000);
                  bufs.push_back(std::make_unique<Aligned>(32768));
                  std::memcpy(bufs.back()->data.get(), v.data(), v.size());
                  items.push_back({Key(id), bufs.back()->data.get(), 30000, 32768});
                }
                s.CacheDirectBatch(items);
                return 9;
              }),
              0);
    bool ok = false;
    LogStore s(Quiet(dir), &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    for (uint64_t id = 0; id < 20; ++id) {
      std::string out;
      if (s.Range(Key(id), 0, 0, &out) == Status::kOk)
        EXPECT_EQ(out, Val(id, 30000)) << id;
    }
    fs::remove_all(dir);
  }
}

// ---- leases and segment pins (§5.3, §7.2) -----------------------------------------------
// A lease pins its segment: the cleaner passes over it while the lease is
// held (after DFKV_LSE_PIN_WAIT_MS), the leased bytes stay readable, and the
// segment is reclaimed once the lease is gone.
TEST(LogStoreStage4, ALeaseHoldsItsSegmentAgainstReclaim) {
  const fs::path dir = TestDir("leasepin");
  auto o = Opts(dir);
  o.pin_wait_ms = 50;
  bool ok = false;
  LogStore s(o, &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  constexpr size_t kLen = 500000;
  const std::string p = Val(9000, kLen);
  ASSERT_EQ(s.Cache(Key(9000), p.data(), p.size()), Status::kOk);
  auto lease = std::make_unique<ReadLease>();
  ASSERT_EQ(s.RangeDirectPrep(Key(9000), 0, 0, 1 << 20, lease.get()), Status::kOk);
  auto fill = [&](uint64_t from, uint64_t n) {
    for (uint64_t id = from; id < from + n; ++id) {
      const std::string v = Val(id, kLen);
      const Status st = s.Cache(Key(id), v.data(), v.size());
      ASSERT_TRUE(st == Status::kOk || st == Status::kCacheFull);
    }
  };
  fill(0, 120);
  EXPECT_GT(s.GetStats().victims_skipped_pins, 0u);
  Aligned buf(1 << 20);
  ASSERT_EQ(::pread(lease->fd(), buf.data.get(), lease->aligned_len,
                    static_cast<off_t>(lease->aligned_off)),
            static_cast<ssize_t>(lease->aligned_len));
  EXPECT_EQ(std::memcmp(buf.data.get() + lease->head, p.data(), p.size()), 0)
      << "the leased bytes are still there";
  EXPECT_TRUE(lease->Verify(buf.data.get() + lease->head));
  const uint64_t reclaimed = s.GetStats().segments_reclaimed;
  lease.reset();
  fill(1000, 120);
  EXPECT_GT(s.GetStats().segments_reclaimed, reclaimed);
  EXPECT_TRUE(s.Healthy());
  fs::remove_all(dir);
}

// A whole-value lease carries the payload CRC: bytes that fail it make the
// value a miss and its location a shadow, which a REMOVE or a new PUT then
// kills (C32). A partial lease carries no check.
TEST(LogStoreStage4, AWholeValueLeaseIsVerified) {
  const fs::path dir = TestDir("leaseverify");
  bool ok = false;
  LogStore s(Opts(dir), &ok);
  ASSERT_TRUE(ok) << s.StartupError();
  const std::string v = Val(1, 20000);
  ASSERT_EQ(s.Cache(Key(1), v.data(), v.size()), Status::kOk);
  Aligned buf(1 << 20);
  {
    ReadLease part;
    ASSERT_EQ(s.RangeDirectPrep(Key(1), 10, 100, 1 << 20, &part), Status::kOk);
    std::memset(buf.data.get(), 0, 4096);
    EXPECT_TRUE(part.Verify(buf.data.get())) << "no check on a slice";
  }
  ReadLease lease;
  ASSERT_EQ(s.RangeDirectPrep(Key(1), 0, 0, 1 << 20, &lease), Status::kOk);
  ASSERT_EQ(::pread(lease.fd(), buf.data.get(), lease.aligned_len,
                    static_cast<off_t>(lease.aligned_off)),
            static_cast<ssize_t>(lease.aligned_len));
  char* payload = buf.data.get() + lease.head;
  EXPECT_TRUE(lease.Verify(payload));
  payload[777] ^= 0x01;  // damaged on the way
  EXPECT_FALSE(lease.Verify(payload));
  std::string out;
  EXPECT_EQ(s.Range(Key(1), 0, 0, &out), Status::kNotFound);
  EXPECT_EQ(s.GetStats().read_crc_failures, 1u);
  EXPECT_EQ(s.GetStats().shadows, 1u);
  EXPECT_EQ(s.Remove(Key(1)), Status::kNotFound);
  EXPECT_EQ(s.GetStats().shadows, 0u) << "killed by the REMOVE";
  const std::string w = Val(1, 20000, 1);
  ASSERT_EQ(s.Cache(Key(1), w.data(), w.size()), Status::kOk);
  EXPECT_TRUE(Has(s, 1, 20000, 1));
  fs::remove_all(dir);
}

// ---- through the node -------------------------------------------------------------------
namespace {
class LogNode : public ::testing::Test {
 protected:
  void SetUp() override {
    ::setenv("DFKV_STORE_ENGINE", "log", 1);
    ::setenv("DFKV_LSE_EXPERIMENTAL", "1", 1);
    ::setenv("DFKV_READ_COALESCE", "1", 1);
    ::setenv("DFKV_RAM_TIER", "1", 1);
    ::setenv("DFKV_RAM_TIER_BYTES", "16777216", 1);
    ::setenv("DFKV_RAM_TIER_EXTENT_BYTES", "1048576", 1);
    ::setenv("DFKV_RAM_WRITE_MODE", "writearound", 1);
    dir_ = TestDir("node");
    fs::create_directories(dir_);
  }
  void TearDown() override {
    for (const char* n : {"DFKV_LSE_EXPERIMENTAL", "DFKV_READ_COALESCE",
                          "DFKV_RAM_TIER", "DFKV_RAM_TIER_BYTES",
                          "DFKV_RAM_TIER_EXTENT_BYTES", "DFKV_RAM_WRITE_MODE"})
      ::unsetenv(n);
    ::setenv("DFKV_STORE_ENGINE", "file", 1);
    fs::remove_all(dir_);
  }
  fs::path dir_;
};
}  // namespace

// A lease taken before its value was relocated reads the old copy: a
// mismatch there drops only that copy (already a shadow), not the value,
// whose new copy was verified when it was made (review S2 of stage 4).
TEST(LogStoreStage4, AMismatchOfARelocatedLeaseKeepsTheNewCopy) {
  const fs::path dir = TestDir("leasereloc");
  LogStore* store = nullptr;
  std::unique_ptr<ReadLease> lease;
  std::atomic<int> phase{0};  // 1: the lease is taken
  auto o = Opts(dir);
  o.pin_wait_ms = 50;
  o.crash_hook = [&](const char* p) {
    // On the cleaner's thread, between the copy and the index switch: the
    // lease reads the source.
    if (std::strcmp(p, "relocate.after_copy") != 0 || phase.load() != 0) return;
    lease = std::make_unique<ReadLease>();
    if (store->RangeDirectPrep(Key(0), 0, 0, 1 << 20, lease.get()) == Status::kOk)
      phase = 1;
  };
  bool ok = false;
  LogStore s(o, &ok);
  store = &s;
  ASSERT_TRUE(ok) << s.StartupError();
  constexpr size_t kLen = 400000;
  const std::string v0 = Val(0, kLen);
  ASSERT_EQ(s.Cache(Key(0), v0.data(), v0.size()), Status::kOk);
  ASSERT_TRUE(Has(s, 0, kLen));  // read: relocated at the first reclaim
  for (uint64_t id = 1; id < 200 && phase.load() == 0; ++id) {
    const std::string v = Val(id, kLen);
    s.Cache(Key(id), v.data(), v.size());
  }
  ASSERT_EQ(phase.load(), 1) << "the value was never relocated";
  for (int i = 0; i < 500 && s.GetStats().records_relocated == 0; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  ASSERT_GT(s.GetStats().records_relocated, 0u);
  // The old copy's bytes fail the check (damaged on the way).
  Aligned buf(1 << 20);
  ASSERT_EQ(::pread(lease->fd(), buf.data.get(), lease->aligned_len,
                    static_cast<off_t>(lease->aligned_off)),
            static_cast<ssize_t>(lease->aligned_len));
  char* payload = buf.data.get() + lease->head;
  payload[123] ^= 0x01;
  EXPECT_FALSE(lease->Verify(payload));
  EXPECT_TRUE(Has(s, 0, kLen)) << "the relocated copy was dropped";
  lease.reset();
  fs::remove_all(dir);
}


// A whole-value read promoted straight into RAM: good bytes are installed
// and served from RAM next time; bytes that fail the check are a miss, the
// reservation is dropped, and the value is gone from the disk's index.
TEST_F(LogNode, PromotedReadsAreVerifiedBeforeTheyAreUsed) {
  KvNodeServer server(dir_.string(), 1ull << 30);
  ASSERT_TRUE(server.Healthy());
  ASSERT_TRUE(server.ram_enabled());
  ASSERT_EQ(server.engine_name(), "log");
  for (uint64_t id : {1, 2}) {
    const std::string v = Val(id, 8192);
    Aligned seed(8192);
    std::memcpy(seed.data.get(), v.data(), v.size());
    ASSERT_EQ(server.CacheDirectForKey(Key(id), seed.data.get(), 8192, 8192),
              Status::kOk);
  }
  Aligned staging(1 << 20);
  auto read_once = [&](uint64_t id, bool damage) {
    PreparedRead read = server.PrepareReadForKey(Key(id), 0, 8192,
                                                 staging.data.get(), staging.size);
    EXPECT_EQ(read.status(), Status::kOk);
    EXPECT_TRUE(read.needs_io());
    const ssize_t got = ::pread(read.fd(), read.staging(), read.aligned_len(),
                                static_cast<off_t>(read.aligned_off()));
    EXPECT_GE(got, static_cast<ssize_t>(read.head() + read.payload_len()));
    if (damage) read.staging()[read.head() + 100] ^= 0x01;
    const bool good = read.Verify();
    EXPECT_EQ(good, !damage);
    read.Commit(good ? Status::kOk : Status::kNotFound,
                good ? read.payload_len() : 0, 0.001);
    read.Commit(Status::kOk, 0, 0.0);  // the SEND completion
  };
  read_once(1, /*damage=*/false);
  {
    PreparedRead again = server.PrepareReadForKey(Key(1), 0, 8192,
                                                  staging.data.get(), staging.size);
    ASSERT_EQ(again.status(), Status::kOk);
    EXPECT_FALSE(again.needs_io()) << "installed in RAM";
    EXPECT_EQ(std::memcmp(again.data(), Val(1, 8192).data(), 8192), 0);
    again.Commit(Status::kOk, again.payload_len(), 0.0);
  }
  read_once(2, /*damage=*/true);
  PreparedRead after = server.PrepareReadForKey(Key(2), 0, 8192,
                                                staging.data.get(), staging.size);
  EXPECT_EQ(after.status(), Status::kNotFound)
      << "neither installed in RAM nor still indexed on the disk";
  std::string out;
  EXPECT_EQ(server.ProcessRequestForKey(static_cast<uint8_t>(WireOp::kRange),
                                        Key(2), 0, 8192, nullptr, 0, &out),
            Status::kNotFound);
}

// The RAM tier's flusher hands batches to the engine (§9.1): values PUT
// through a RAM-acknowledged node reach the log store and survive a
// restart of the node.
TEST_F(LogNode, RamTierFlushesReachTheLogStore) {
  ::setenv("DFKV_RAM_WRITE_MODE", "writeback", 1);
  constexpr int kValues = 40;
  {
    KvNodeServer server(dir_.string(), 1ull << 30);
    ASSERT_TRUE(server.Healthy());
    for (int id = 0; id < kValues; ++id) {
      const std::string v = Val(id, 50000);
      std::string ignored;
      ASSERT_EQ(server.ProcessRequestForKey(static_cast<uint8_t>(WireOp::kCache),
                                            Key(id), 0, 0, v.data(), v.size(),
                                            &ignored),
                Status::kOk);
    }
    server.Shutdown();  // flushes what the RAM tier holds
  }
  ::setenv("DFKV_RAM_WRITE_MODE", "writearound", 1);
  KvNodeServer server(dir_.string(), 1ull << 30);
  ASSERT_TRUE(server.Healthy());
  for (int id = 0; id < kValues; ++id) {
    std::string out;
    ASSERT_EQ(server.ProcessRequestForKey(static_cast<uint8_t>(WireOp::kRange),
                                          Key(id), 0, 50000, nullptr, 0, &out),
              Status::kOk)
        << id;
    EXPECT_EQ(out, Val(id, 50000)) << id;
  }
}
