// DFKV_RAM_PROMOTE: which whole-value cold reads become RAM residents. With
// first (the default) every one, as before; with second only a key's repeat
// read within DFKV_RAM_PROMOTE_WINDOW_MS, and a read that is not promoted
// leaves the RAM tier as it found it on each of the three read paths: the
// synchronous one, the direct read into a reserved RAM slot, and the staged
// read that is promoted when it lands.
#include "cache/kv_node_server.h"
#include "cache/recent_reads.h"

#include <gtest/gtest.h>

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {

TEST(RecentReads, ASecondReadWithinTheWindowIsARepeat) {
  RecentReads recent(/*window_ms=*/1000, /*capacity=*/8);
  const BlockKey a{1, 2, 3};
  EXPECT_FALSE(recent.ReadAgain(a, 100));
  EXPECT_TRUE(recent.ReadAgain(a, 1099));
  EXPECT_TRUE(recent.ReadAgain(a, 2098));  // each read restarts the window
  EXPECT_FALSE(recent.ReadAgain(a, 3098));  // a full window later: a first read
  EXPECT_EQ(recent.size(), 1u);
  // Keys differ by any of digest and tenant.
  EXPECT_FALSE(recent.ReadAgain(BlockKey{1, 2, 4}, 3098));
  EXPECT_FALSE(recent.ReadAgain(BlockKey{1, 3, 3}, 3098));
  EXPECT_FALSE(recent.ReadAgain(BlockKey{2, 2, 3}, 3098));
  EXPECT_EQ(recent.size(), 4u);
  // Expired keys are dropped from the table.
  EXPECT_FALSE(recent.ReadAgain(BlockKey{9, 9, 9}, 5000));
  EXPECT_EQ(recent.size(), 1u);
}

TEST(RecentReads, AskingWithoutNotingChangesNothing) {
  RecentReads recent(1000, 8);
  const BlockKey a{1, 2, 3};
  EXPECT_FALSE(recent.ReadAgain(a, 100, /*note=*/false));
  EXPECT_FALSE(recent.ReadAgain(a, 200, /*note=*/false));
  EXPECT_EQ(recent.size(), 0u);
  EXPECT_FALSE(recent.ReadAgain(a, 300));
  EXPECT_TRUE(recent.ReadAgain(a, 1200, /*note=*/false));
  EXPECT_FALSE(recent.ReadAgain(a, 1300, /*note=*/false));  // not renewed
}

TEST(RecentReads, TheTableIsBoundedAndForgetsTheLeastRecentlyRead) {
  RecentReads recent(/*window_ms=*/1000000, /*capacity=*/3);
  for (uint64_t k = 1; k <= 3; ++k)
    EXPECT_FALSE(recent.ReadAgain(BlockKey{k, 0, 0}, k));
  EXPECT_TRUE(recent.ReadAgain(BlockKey{1, 0, 0}, 10));  // 1 is now newest
  EXPECT_FALSE(recent.ReadAgain(BlockKey{4, 0, 0}, 11));  // pushes out 2
  EXPECT_EQ(recent.size(), 3u);
  EXPECT_FALSE(recent.ReadAgain(BlockKey{2, 0, 0}, 12));  // pushes out 3
  EXPECT_TRUE(recent.ReadAgain(BlockKey{1, 0, 0}, 13));
  EXPECT_TRUE(recent.ReadAgain(BlockKey{4, 0, 0}, 14));
  EXPECT_FALSE(recent.ReadAgain(BlockKey{3, 0, 0}, 15));
  EXPECT_EQ(recent.size(), 3u);
}

class AlignedBuffer {
 public:
  explicit AlignedBuffer(size_t size) : size_(size) {
    void* p = nullptr;
    EXPECT_EQ(::posix_memalign(&p, 4096, size), 0);
    data_.reset(static_cast<char*>(p));
  }
  char* data() const { return data_.get(); }
  size_t size() const { return size_; }

 private:
  struct Free {
    void operator()(char* p) const { std::free(p); }
  };
  std::unique_ptr<char, Free> data_;
  size_t size_;
};

long MetricVal(const std::string& text, const std::string& name) {
  size_t ls = 0;
  while (ls < text.size()) {
    size_t le = text.find('\n', ls);
    if (le == std::string::npos) le = text.size();
    const std::string line = text.substr(ls, le - ls);
    ls = le + 1;
    if (line.empty() || line[0] == '#') continue;
    if (line.compare(0, name.size(), name) != 0) continue;
    const char after = line.size() > name.size() ? line[name.size()] : '\0';
    if (after != '{' && after != ' ') continue;
    return std::stol(line.substr(line.rfind(' ') + 1));
  }
  return -1;
}

// What the RAM tier and the coalescer did, from the server's metrics.
struct Counts {
  long promoted, declined, aborted, deferred, hits, leaders, coalesced;
  explicit Counts(KvNodeServer& s) {
    const std::string m = s.MetricsText();
    promoted = MetricVal(m, "dfkv_ram_promoted_total");
    declined = MetricVal(m, "dfkv_ram_promote_declined_total");
    aborted = MetricVal(m, "dfkv_ram_reservations_aborted_total");
    deferred = MetricVal(m, "dfkv_ram_promote_deferred_total");
    hits = MetricVal(m, "dfkv_ram_hit_total");
    leaders = MetricVal(m, "dfkv_read_coalesce_leaders_total");
    coalesced = MetricVal(m, "dfkv_read_coalesced_total");
  }
};

class RamPromote : public ::testing::Test {
 protected:
  void SetUp() override {
    ::setenv("DFKV_RAM_TIER", "1", 1);
    ::setenv("DFKV_RAM_TIER_BYTES", "16777216", 1);
    ::setenv("DFKV_RAM_TIER_EXTENT_BYTES", "1048576", 1);
    // Values above an extent are promoted only into this budget, and only
    // after a staged read: they cannot be read into a reserved slot.
    ::setenv("DFKV_RAM_TIER_LARGE_RESERVE_BYTES", "4194304", 1);
    ::setenv("DFKV_RAM_WRITE_MODE", "writearound", 1);
    dir_ = fs::temp_directory_path() /
           ("dfkv_ram_promote_" + std::to_string(::getpid()));
    fs::remove_all(dir_);
    fs::create_directories(dir_);
  }
  void TearDown() override {
    server_.reset();
    fs::remove_all(dir_);
    for (const char* name :
         {"DFKV_RAM_TIER", "DFKV_RAM_TIER_BYTES", "DFKV_RAM_TIER_EXTENT_BYTES",
          "DFKV_RAM_TIER_LARGE_RESERVE_BYTES", "DFKV_RAM_WRITE_MODE",
          "DFKV_RAM_RECLAIM_MS", "DFKV_READ_COALESCE", "DFKV_RAM_PROMOTE",
          "DFKV_RAM_PROMOTE_WINDOW_MS"})
      ::unsetenv(name);
  }
  // promote: nullptr leaves DFKV_RAM_PROMOTE unset.
  KvNodeServer& Start(const char* promote, bool coalesce) {
    server_.reset();
    fs::remove_all(dir_);
    fs::create_directories(dir_);
    if (promote)
      ::setenv("DFKV_RAM_PROMOTE", promote, 1);
    else
      ::unsetenv("DFKV_RAM_PROMOTE");
    ::setenv("DFKV_READ_COALESCE", coalesce ? "1" : "0", 1);
    server_ = std::make_unique<KvNodeServer>(dir_.string(), 1ull << 30);
    EXPECT_TRUE(server_->Healthy());
    EXPECT_TRUE(server_->ram_enabled());
    return *server_;
  }
  static std::string Value(size_t size, char seed) {
    std::string v(size, '\0');
    for (size_t i = 0; i < size; ++i)
      v[i] = static_cast<char>(seed + (i % 23));
    return v;
  }
  static void Persist(KvNodeServer& s, const BlockKey& key,
                      const std::string& value) {
    AlignedBuffer seed(((value.size() + 4095) / 4096) * 4096);
    std::memcpy(seed.data(), value.data(), value.size());
    ASSERT_EQ(s.CacheDirectForKey(key, seed.data(), value.size(), seed.size()),
              Status::kOk);
  }
  // The synchronous read (TCP GET, and RDMA without a prepared read).
  static void SyncGet(KvNodeServer& s, const BlockKey& key,
                      const std::string& value) {
    AlignedBuffer io(4 << 20);
    const char* data = nullptr;
    size_t n = 0, value_len = 0;
    ASSERT_EQ(s.RangeDirectForKey(key, 0, value.size(), io.data(), io.size(),
                                  &data, &n, &value_len),
              Status::kOk);
    ASSERT_EQ(n, value.size());
    EXPECT_EQ(std::memcmp(data, value.data(), n), 0);
  }
  struct Prepared {
    bool needs_io = false;
    bool into_ram = false;  // read straight into a reserved RAM slot
  };
  // The RDMA server's prepared read: disk IO, completion, then SEND.
  static Prepared PreparedGet(KvNodeServer& s, const BlockKey& key,
                              const std::string& value) {
    AlignedBuffer staging(4 << 20);
    PreparedRead read = s.PrepareReadForKey(key, 0, value.size(),
                                            staging.data(), staging.size());
    EXPECT_EQ(read.status(), Status::kOk);
    Prepared seen;
    seen.needs_io = read.needs_io();
    seen.into_ram = read.needs_io() && read.staging() != staging.data();
    if (read.needs_io()) {
      const ssize_t got =
          ::pread(read.fd(), read.staging(), read.aligned_len(),
                  static_cast<off_t>(read.aligned_off()));
      const bool ok = got >= 0 && static_cast<size_t>(got) >=
                                      read.head() + read.payload_len();
      EXPECT_TRUE(ok);
      EXPECT_TRUE(read.Commit(ok ? Status::kOk : Status::kIOError,
                              ok ? read.payload_len() : 0, 0.001));
      EXPECT_EQ(std::memcmp(read.data(), value.data(), value.size()), 0);
      read.Commit(Status::kOk, read.payload_len(), 0.0);  // SEND completion
    } else {
      EXPECT_EQ(std::memcmp(read.data(), value.data(), value.size()), 0);
      EXPECT_TRUE(read.Commit(Status::kOk, read.payload_len(), 0.0));
    }
    return seen;
  }

  fs::path dir_;
  std::unique_ptr<KvNodeServer> server_;
};

TEST_F(RamPromote, SecondPromotesOnlyTheRepeatReadOnTheSynchronousPath) {
  for (bool coalesce : {false, true}) {
    SCOPED_TRACE("coalesce=" + std::to_string(coalesce));
    KvNodeServer& s = Start("second", coalesce);
    const BlockKey key{701, 1, 7};
    const std::string value = Value(8192, 'a');
    Persist(s, key, value);
    const Counts before(s);
    EXPECT_EQ(before.deferred, 0);
    SyncGet(s, key, value);
    const Counts first(s);
    EXPECT_EQ(first.promoted, before.promoted);
    EXPECT_EQ(first.declined, before.declined);
    EXPECT_EQ(first.aborted, before.aborted);
    EXPECT_EQ(first.deferred, 1);
    EXPECT_EQ(MetricVal(s.MetricsText(), "dfkv_ram_recent_reads"), 1);
    SyncGet(s, key, value);
    const Counts second(s);
    EXPECT_EQ(second.promoted, before.promoted + 1);
    EXPECT_EQ(second.deferred, 1);
    SyncGet(s, key, value);  // from RAM
    EXPECT_EQ(Counts(s).hits, second.hits + 1);
  }
}

TEST_F(RamPromote, FirstPromotesTheFirstReadAndHasNoNewMetrics) {
  for (const char* promote : {static_cast<const char*>(nullptr), "first"}) {
    SCOPED_TRACE(promote ? promote : "unset");
    KvNodeServer& s = Start(promote, /*coalesce=*/false);
    const BlockKey key{702, 1, 7};
    const std::string value = Value(8192, 'b');
    Persist(s, key, value);
    const Counts before(s);
    SyncGet(s, key, value);
    EXPECT_EQ(Counts(s).promoted, before.promoted + 1);
    const std::string m = s.MetricsText();
    EXPECT_EQ(m.find("dfkv_ram_promote_deferred_total"), std::string::npos);
    EXPECT_EQ(m.find("dfkv_ram_recent_reads"), std::string::npos);
  }
}

// The direct path reserves a RAM slot before the disk read and publishes it
// when the read lands. Under second a first read reserves nothing, so there
// is nothing to release or abort, and it reads into the caller's staging.
TEST_F(RamPromote, SecondReservesNoSlotForAFirstDirectRead) {
  for (bool coalesce : {false, true}) {
    SCOPED_TRACE("coalesce=" + std::to_string(coalesce));
    KvNodeServer& s = Start("second", coalesce);
    const BlockKey key{703, 1, 7};
    const std::string value = Value(8192, 'c');
    Persist(s, key, value);
    const Counts before(s);
    const Prepared first = PreparedGet(s, key, value);
    EXPECT_TRUE(first.needs_io);
    EXPECT_FALSE(first.into_ram);
    const Counts after_first(s);
    EXPECT_EQ(after_first.promoted, before.promoted);
    EXPECT_EQ(after_first.declined, before.declined);
    EXPECT_EQ(after_first.aborted, before.aborted);
    EXPECT_EQ(after_first.deferred, 1);

    const Prepared second = PreparedGet(s, key, value);
    EXPECT_TRUE(second.needs_io);
    EXPECT_TRUE(second.into_ram);
    const Counts after_second(s);
    EXPECT_EQ(after_second.promoted, before.promoted + 1);
    EXPECT_EQ(after_second.aborted, before.aborted);
    EXPECT_EQ(after_second.deferred, 1);
    EXPECT_FALSE(PreparedGet(s, key, value).needs_io);  // from RAM
  }
}

// Under first the same read reserves at once, as before.
TEST_F(RamPromote, FirstReservesASlotForTheFirstDirectRead) {
  KvNodeServer& s = Start(nullptr, /*coalesce=*/true);
  const BlockKey key{704, 1, 7};
  const std::string value = Value(8192, 'd');
  Persist(s, key, value);
  const Counts before(s);
  const Prepared first = PreparedGet(s, key, value);
  EXPECT_TRUE(first.into_ram);
  EXPECT_EQ(Counts(s).promoted, before.promoted + 1);
  EXPECT_FALSE(PreparedGet(s, key, value).needs_io);
}

// A value above an extent cannot be read into a reserved slot (the
// reservation is declined), so a coalesced read of it is staged and promoted
// when it lands. Under second a first read is neither: the RAM tier is not
// asked at all.
TEST_F(RamPromote, SecondDoesNotPromoteAFirstStagedRead) {
  const std::string value = Value((1 << 20) + 4096 * 3 + 17, 'e');
  {
    KvNodeServer& s = Start(nullptr, /*coalesce=*/true);
    const BlockKey key{705, 1, 7};
    Persist(s, key, value);
    const Counts before(s);
    const Prepared first = PreparedGet(s, key, value);
    EXPECT_TRUE(first.needs_io);
    EXPECT_FALSE(first.into_ram);
    const Counts after(s);
    EXPECT_EQ(after.declined, before.declined + 1);  // the reservation
    EXPECT_EQ(after.promoted, before.promoted + 1);  // staged, on landing
    EXPECT_FALSE(PreparedGet(s, key, value).needs_io);
  }
  KvNodeServer& s = Start("second", /*coalesce=*/true);
  const BlockKey key{706, 1, 7};
  Persist(s, key, value);
  const Counts before(s);
  const Prepared first = PreparedGet(s, key, value);
  EXPECT_TRUE(first.needs_io);
  EXPECT_FALSE(first.into_ram);
  const Counts after_first(s);
  EXPECT_EQ(after_first.declined, before.declined);
  EXPECT_EQ(after_first.promoted, before.promoted);
  EXPECT_EQ(after_first.aborted, before.aborted);
  EXPECT_EQ(after_first.deferred, 1);
  const Prepared second = PreparedGet(s, key, value);
  EXPECT_TRUE(second.needs_io);
  EXPECT_FALSE(second.into_ram);
  const Counts after_second(s);
  EXPECT_EQ(after_second.declined, before.declined + 1);
  EXPECT_EQ(after_second.promoted, before.promoted + 1);
  EXPECT_FALSE(PreparedGet(s, key, value).needs_io);
}

// A prepared read the coalescer turns away (an identical read is in flight)
// is redone by the synchronous path; it is one read, not two.
TEST_F(RamPromote, AReadTurnedAwayIsCountedOnce) {
  KvNodeServer& s = Start("second", /*coalesce=*/true);
  const BlockKey key{707, 1, 7};
  const std::string value = Value(8192, 'f');
  Persist(s, key, value);
  AlignedBuffer a(1 << 20), b(1 << 20);
  PreparedRead leader =
      s.PrepareReadForKey(key, 0, value.size(), a.data(), a.size());
  ASSERT_EQ(leader.status(), Status::kOk);
  ASSERT_TRUE(leader.needs_io());
  PreparedRead turned_away =
      s.PrepareReadForKey(key, 0, value.size(), b.data(), b.size());
  EXPECT_EQ(turned_away.status(), Status::kInvalid);
  EXPECT_EQ(Counts(s).deferred, 1);  // the leader only
  leader.Abort();
  SyncGet(s, key, value);  // the turned-away read, redone: a repeat
  const Counts after(s);
  EXPECT_EQ(after.deferred, 1);
  EXPECT_GE(after.promoted, 1);
}

// What the setting is for, in small: a hot set read again and again while a
// one-time scan streams past, and keys read exactly three times. The numbers
// are printed for the record; the test asserts only their direction.
TEST_F(RamPromote, ScanAndReReadReplay) {
  ::setenv("DFKV_RAM_TIER_BYTES", "8388608", 1);  // 32 values of 256 KiB
  // Evictions only for room, when a value is installed: the background
  // reclaim would evict on its own clock and make the counts timing-bound.
  ::setenv("DFKV_RAM_RECLAIM_MS", "0", 1);
  const size_t kValue = 256 * 1024;
  auto scan = [&](const char* promote) {
    KvNodeServer& s = Start(promote, /*coalesce=*/false);
    const std::string value = Value(kValue, 'h');
    const uint64_t kHot = 8, kScan = 256;
    for (uint64_t k = 0; k < kHot; ++k) Persist(s, BlockKey{1000 + k, 0, 7}, value);
    for (uint64_t k = 0; k < kScan; ++k) Persist(s, BlockKey{2000 + k, 0, 7}, value);
    long hot_reads = 0, hot_hits = 0;
    for (uint64_t k = 0; k < kScan; ++k) {
      SyncGet(s, BlockKey{2000 + k, 0, 7}, value);
      if (k % 4 != 3) continue;
      const long before = Counts(s).hits;
      SyncGet(s, BlockKey{1000 + (k / 4) % kHot, 0, 7}, value);
      ++hot_reads;
      hot_hits += Counts(s).hits - before;
    }
    return std::make_pair(hot_hits, hot_reads);
  };
  auto thrice = [&](const char* promote) {
    KvNodeServer& s = Start(promote, /*coalesce=*/false);
    const std::string value = Value(kValue, 'i');
    long hits[3] = {0, 0, 0};
    // 16 values, which the RAM tier holds without evicting any.
    for (uint64_t k = 0; k < 16; ++k) {
      const BlockKey key{3000 + k, 0, 7};
      Persist(s, key, value);
      for (int r = 0; r < 3; ++r) {
        const long before = Counts(s).hits;
        SyncGet(s, key, value);
        hits[r] += Counts(s).hits - before;
      }
    }
    return std::vector<long>{hits[0], hits[1], hits[2]};
  };
  const auto scan_first = scan("first"), scan_second = scan("second");
  const auto thrice_first = thrice("first"), thrice_second = thrice("second");
  std::printf("replay scan: hot hits first %ld/%ld, second %ld/%ld\n",
              scan_first.first, scan_first.second, scan_second.first,
              scan_second.second);
  std::printf("replay thrice: hits by read first %ld/%ld/%ld, second "
              "%ld/%ld/%ld\n",
              thrice_first[0], thrice_first[1], thrice_first[2],
              thrice_second[0], thrice_second[1], thrice_second[2]);
  EXPECT_GT(scan_second.first, scan_first.first);
  // Under second every hot read hits but each hot key's first two.
  EXPECT_EQ(scan_second.first, scan_second.second - 2 * 8);
  EXPECT_EQ(thrice_first, (std::vector<long>{0, 16, 16}));
  EXPECT_EQ(thrice_second, (std::vector<long>{0, 0, 16}));
}

// The coalescer is not changed by DFKV_RAM_PROMOTE=first: the same reads,
// alone and with a concurrent duplicate, take the same leader and follower
// decisions as with the setting unset.
TEST_F(RamPromote, FirstLeavesTheCoalescersFanInAsItWas) {
  auto run = [&](const char* promote) {
    KvNodeServer& s = Start(promote, /*coalesce=*/true);
    std::vector<long> seen;
    const std::string value = Value(8192, 'g');
    for (uint64_t k = 0; k < 4; ++k) {
      const BlockKey key{710 + k, 1, 7};
      Persist(s, key, value);
      AlignedBuffer a(1 << 20), b(1 << 20);
      PreparedRead leader =
          s.PrepareReadForKey(key, 0, value.size(), a.data(), a.size());
      EXPECT_EQ(leader.status(), Status::kOk);
      PreparedRead follower;
      std::atomic<bool> entered{false};
      std::thread t;
      if (k % 2 == 1) {  // odd keys: a concurrent duplicate joins the read
        t = std::thread([&] {
          entered.store(true, std::memory_order_release);
          follower = s.PrepareReadForKey(key, 0, value.size(), b.data(),
                                         b.size());
        });
        while (!entered.load(std::memory_order_acquire))
          std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
      seen.push_back(leader.needs_io());
      seen.push_back(leader.staging() != a.data());
      const ssize_t got = ::pread(leader.fd(), leader.staging(),
                                  leader.aligned_len(),
                                  static_cast<off_t>(leader.aligned_off()));
      EXPECT_GE(got, 0);
      leader.Commit(Status::kOk, leader.payload_len(), 0.001);
      if (t.joinable()) {
        t.join();
        seen.push_back(static_cast<long>(follower.status()));
        seen.push_back(follower.needs_io());
        seen.push_back(follower.source_registered());
        if (follower.status() == Status::kOk)
          follower.Commit(Status::kOk, follower.payload_len(), 0.0);
      }
      leader.Commit(Status::kOk, leader.payload_len(), 0.0);
      // A later lone read: from RAM, or a disk read the coalescer leads.
      seen.push_back(PreparedGet(s, key, value).needs_io);
      // And a synchronous one, through the coalescer's synchronous path.
      Persist(s, BlockKey{720 + k, 1, 7}, value);
      SyncGet(s, BlockKey{720 + k, 1, 7}, value);
    }
    const Counts c(s);
    seen.push_back(c.leaders);
    seen.push_back(c.coalesced);
    seen.push_back(c.promoted);
    seen.push_back(c.declined);
    seen.push_back(c.aborted);
    seen.push_back(MetricVal(s.MetricsText(), "dfkv_read_coalesce_recur_total"));
    return seen;
  };
  EXPECT_EQ(run("first"), run(nullptr));
}

}  // namespace
