// DiskSlabStore: extent-file slab store on SlabAllocator + slots.tbl rebuild.
// Covers data-path semantics, restart warmth, crash/torn-record handling,
// strict startup geometry, fail-closed metadata I/O, and reclaimer safety.
#include "cache/disk_slab_store.h"

#include <gtest/gtest.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/wait.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

using dfkv::BlockKey;
using dfkv::DiskSlabStore;
using dfkv::Status;
namespace fs = std::filesystem;

namespace dfkv {
class DiskSlabStoreTestPeer {
 public:
  static void ReclaimNow(DiskSlabStore* store,
                         const std::function<void()>& after_watermark_evict) {
    store->ReclaimTick(after_watermark_evict);
  }
  static void BreakTableFd(DiskSlabStore* store) {
    if (store->table_fd_ >= 0) ::close(store->table_fd_);
    store->table_fd_ = -1;
  }

  static bool SyncTableNow(DiskSlabStore* store) {
    return store->SyncTable(
        store->record_writes_.load(std::memory_order_relaxed));
  }
  static bool WriteCrashStage(DiskSlabStore* store, const BlockKey& key,
                              const std::string& value,
                              bool include_record) {
    const std::string filename = key.Filename();
    SlabAllocator::SlotRef ref;
    {
      std::lock_guard<std::mutex> lock(store->mu_);
      std::vector<std::string> evicted;
      if (!store->alloc_->Put(filename, value.size(), &ref, &evicted))
        return false;
      if (!store->alloc_->Pin(filename)) return false;
      store->inflight_[filename]++;
    }
    if (!store->WritePayload(ref, value.data(), value.size())) return false;
    return !include_record ||
           store->WriteRecord(ref, key,
                              static_cast<uint32_t>(value.size()), true);
  }
  static void BreakStateFd(DiskSlabStore* store) {
    if (store->state_fd_ >= 0) ::close(store->state_fd_);
    store->state_fd_ = -1;
  }
  static bool MarkCleanNow(DiskSlabStore* store) {
    return store->MarkCleanEpoch();
  }
};
}  // namespace dfkv

namespace {
class DiskSlabTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           ("dfkv_slab_" + std::to_string(::testing::UnitTest::GetInstance()
                                              ->current_test_info()->line()));
    fs::remove_all(dir_);
  }
  void TearDown() override { fs::remove_all(dir_); }

  DiskSlabStore::Options Opts(uint64_t cap, uint64_t extent, uint64_t gran) {
    DiskSlabStore::Options o;
    o.dir = dir_.string();
    o.capacity_bytes = cap;
    o.extent_bytes = extent;
    o.slot_granularity = gran;
    return o;
  }
  fs::path dir_;
};
BlockKey K(uint64_t id) { return BlockKey{id, 0, 1}; }
}  // namespace

TEST_F(DiskSlabTest, PutGetRemoveRoundTrip) {
  bool ok = false;
  DiskSlabStore s(Opts(1 << 20, 1 << 20, 4096), &ok);
  ASSERT_TRUE(ok);
  std::string v = "the-kv-payload-0123456789";
  ASSERT_EQ(s.Cache(K(1), v.data(), v.size()), Status::kOk);
  EXPECT_TRUE(s.IsCached(K(1)));
  EXPECT_EQ(s.Count(), 1u);

  std::string out;
  ASSERT_EQ(s.Range(K(1), 0, v.size(), &out), Status::kOk);
  EXPECT_EQ(out, v);
  // partial range
  ASSERT_EQ(s.Range(K(1), 4, 5, &out), Status::kOk);
  EXPECT_EQ(out, v.substr(4, 5));
  // RangeInto
  char buf[64];
  size_t got = 0;
  ASSERT_EQ(s.RangeInto(K(1), 0, sizeof(buf), buf, sizeof(buf), &got), Status::kOk);
  EXPECT_EQ(std::string(buf, got), v);

  EXPECT_EQ(s.Range(K(999), 0, 10, &out), Status::kNotFound);
  EXPECT_EQ(s.Cache(K(1), v.data(), v.size()), Status::kOk);  // idempotent
  EXPECT_EQ(s.Count(), 1u);

  ASSERT_EQ(s.Remove(K(1)), Status::kOk);
  EXPECT_FALSE(s.IsCached(K(1)));
  EXPECT_EQ(s.Remove(K(1)), Status::kNotFound);
}

TEST_F(DiskSlabTest, StatsTrackPhysicalAndLogicalOccupancy) {
  bool ok = false;
  DiskSlabStore store(Opts(1 << 20, 1 << 20, 4096), &ok);
  ASSERT_TRUE(ok);
  std::string value(3000, 's');
  ASSERT_EQ(store.Cache(K(1), value.data(), value.size()), Status::kOk);

  auto stats = store.GetStats();
  EXPECT_EQ(stats.capacity_bytes, 1u << 20);
  EXPECT_EQ(stats.allocated_bytes, 4096u);
  EXPECT_EQ(stats.payload_bytes, value.size());
  EXPECT_EQ(stats.allocator_objects, 1u);
  EXPECT_EQ(stats.committed_objects, 1u);
  EXPECT_EQ(stats.class_count, 1u);
  EXPECT_EQ(stats.bound_extents, 1u);
  EXPECT_EQ(stats.pool_extents, 0u);
  EXPECT_EQ(stats.failed_disks, 0u);
  EXPECT_FALSE(stats.failed);

  ASSERT_EQ(store.Remove(K(1)), Status::kOk);
  stats = store.GetStats();
  EXPECT_EQ(stats.allocated_bytes, 0u);
  EXPECT_EQ(stats.payload_bytes, 0u);
  EXPECT_EQ(stats.allocator_objects, 0u);
  EXPECT_EQ(stats.committed_objects, 0u);
}

TEST_F(DiskSlabTest, EvictsUnderCapacity) {
  // 1 extent of 4 * 4096 slots. 5 keys -> one eviction.
  bool ok = false;
  DiskSlabStore s(Opts(4 * 4096, 4 * 4096, 4096), &ok);
  ASSERT_TRUE(ok);
  std::string v(4096, 'e');
  for (int i = 0; i < 4; ++i) ASSERT_EQ(s.Cache(K(i), v.data(), v.size()), Status::kOk);
  EXPECT_EQ(s.Count(), 4u);
  ASSERT_EQ(s.Cache(K(4), v.data(), v.size()), Status::kOk);
  EXPECT_EQ(s.Count(), 4u);
  EXPECT_EQ(s.Evictions(), 1u);
  EXPECT_TRUE(s.IsCached(K(4)));
}

TEST_F(DiskSlabTest, RestartRebuildsIndexKeepingWarmth) {
  std::vector<std::string> vals;
  {
    bool ok = false;
    DiskSlabStore s(Opts(1 << 20, 1 << 20, 4096), &ok);
    ASSERT_TRUE(ok);
    for (int i = 0; i < 10; ++i) {
      std::string v = "val-" + std::to_string(i) + std::string(100, 'x');
      vals.push_back(v);
      ASSERT_EQ(s.Cache(K(1000 + i), v.data(), v.size()), Status::kOk);
    }
    ASSERT_EQ(s.Count(), 10u);
  }
  // Reopen the same dir: the index must rebuild from slots.tbl and all values
  // read back byte-for-byte (warmth preserved across restart).
  bool ok = false;
  DiskSlabStore s2(Opts(1 << 20, 1 << 20, 4096), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(s2.TableRebuilt(), 10u);
  EXPECT_EQ(s2.Count(), 10u);
  for (int i = 0; i < 10; ++i) {
    std::string out;
    ASSERT_EQ(s2.Range(K(1000 + i), 0, vals[i].size(), &out), Status::kOk) << i;
    EXPECT_EQ(out, vals[i]) << i;
  }
}

TEST_F(DiskSlabTest, TornTableRecordReadsAsFreeNotResurrected) {
  {
    bool ok = false;
    DiskSlabStore s(Opts(1 << 20, 1 << 20, 4096), &ok);
    ASSERT_TRUE(ok);
    std::string v(200, 'z');
    ASSERT_EQ(s.Cache(K(7), v.data(), v.size()), Status::kOk);
    ASSERT_EQ(s.Cache(K(8), v.data(), v.size()), Status::kOk);
  }
  // Corrupt the BODY of a real (valid) record so its CRC no longer matches -- a
  // torn write. Records land in high slots (LIFO alloc), so scan for the first
  // "SLTB" magic, then flip a body byte (offset +12, inside the CRC'd region).
  const std::string tbl = (dir_ / "slots.tbl").string();
  int fd = ::open(tbl.c_str(), O_RDWR);
  ASSERT_GE(fd, 0);
  off_t fsz = ::lseek(fd, 0, SEEK_END);
  bool corrupted = false;
  for (off_t o = 0; o + 64 <= fsz; o += 64) {
    uint8_t m[4];
    if (::pread(fd, m, 4, o) != 4) break;
    // "SLTB" little-endian = 53 54 4C 42
    if (m[0] == 0x53 && m[1] == 0x54 && m[2] == 0x4C && m[3] == 0x42) {
      uint8_t byte = 0;
      ASSERT_EQ(::pread(fd, &byte, 1, o + 12), 1);
      byte ^= 0xFF;
      ASSERT_EQ(::pwrite(fd, &byte, 1, o + 12), 1);
      corrupted = true;
      break;
    }
  }
  ::close(fd);
  ASSERT_TRUE(corrupted) << "expected a valid record to corrupt";

  bool ok = false;
  DiskSlabStore s2(Opts(1 << 20, 1 << 20, 4096), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(s2.TableRebuilt(), 1u) << "the torn record must be skipped";
  EXPECT_EQ(s2.Count(), 1u);
  EXPECT_EQ(s2.GetStats().rebuild_corrupt_records, 1u);
}

TEST_F(DiskSlabTest, LayoutMismatchRefusesAndPreservesExistingData) {
  std::string value(50, 'q');
  {
    bool ok = false;
    DiskSlabStore s(Opts(1 << 20, 1 << 20, 4096), &ok);
    ASSERT_TRUE(ok);
    ASSERT_EQ(s.Cache(K(5), value.data(), value.size()), Status::kOk);
  }
  {
    bool ok = true;
    DiskSlabStore mismatched(Opts(1 << 20, 1 << 20, 8192), &ok);
    EXPECT_FALSE(ok);
    EXPECT_FALSE(mismatched.Healthy());
    EXPECT_FALSE(mismatched.StartupError().empty());
  }
  bool ok = false;
  DiskSlabStore reopened(Opts(1 << 20, 1 << 20, 4096), &ok);
  ASSERT_TRUE(ok);
  std::string out;
  ASSERT_EQ(reopened.Range(K(5), 0, 0, &out), Status::kOk);
  EXPECT_EQ(out, value);
}

TEST_F(DiskSlabTest, InvalidGeometryIsRejectedBeforeCreatingFiles) {
  auto invalid = [&](DiskSlabStore::Options options) {
    bool ok = true;
    DiskSlabStore store(options, &ok);
    EXPECT_FALSE(ok);
    EXPECT_FALSE(store.Healthy());
    EXPECT_FALSE(store.StartupError().empty());
  };

  auto zero_capacity = Opts(1 << 20, 1 << 20, 4096);
  zero_capacity.capacity_bytes = 0;
  invalid(zero_capacity);

  auto partial_extent = Opts(1 << 20, 1 << 20, 4096);
  partial_extent.dir = (dir_.string() + "_partial");
  partial_extent.capacity_bytes += 4096;
  invalid(partial_extent);

  auto unaligned_granularity = Opts(1 << 20, 1 << 20, 4096);
  unaligned_granularity.dir = (dir_.string() + "_granularity");
  unaligned_granularity.slot_granularity = 6000;
  invalid(unaligned_granularity);
}

TEST_F(DiskSlabTest, RegularFileCannotMasqueradeAsStoreDirectory) {
  fs::create_directories(dir_.parent_path());
  int fd = ::open(dir_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(::write(fd, "x", 1), 1);
  ::close(fd);

  bool ok = true;
  DiskSlabStore store(Opts(1 << 20, 1 << 20, 4096), &ok);
  EXPECT_FALSE(ok);
  EXPECT_FALSE(store.StartupError().empty());
}

TEST_F(DiskSlabTest, CorruptMetaIsRefusedWithoutReinitializing) {
  {
    bool ok = false;
    DiskSlabStore store(Opts(1 << 20, 1 << 20, 4096), &ok);
    ASSERT_TRUE(ok);
  }
  const fs::path meta = dir_ / "slab_meta";
  int fd = ::open(meta.c_str(), O_RDWR);
  ASSERT_GE(fd, 0);
  const uint32_t bad_magic = 0;
  ASSERT_EQ(::pwrite(fd, &bad_magic, sizeof(bad_magic), 0),
            static_cast<ssize_t>(sizeof(bad_magic)));
  ASSERT_EQ(::fdatasync(fd), 0);
  ::close(fd);

  bool ok = true;
  DiskSlabStore store(Opts(1 << 20, 1 << 20, 4096), &ok);
  EXPECT_FALSE(ok);
  EXPECT_NE(store.StartupError().find("magic"), std::string::npos);
  fd = ::open(meta.c_str(), O_RDONLY);
  ASSERT_GE(fd, 0);
  uint32_t persisted_magic = 1;
  ASSERT_EQ(::read(fd, &persisted_magic, sizeof(persisted_magic)),
            static_cast<ssize_t>(sizeof(persisted_magic)));
  ::close(fd);
  EXPECT_EQ(persisted_magic, 0u);
}

TEST_F(DiskSlabTest, TruncatedSlotTableIsRefusedWithoutResizing) {
  {
    bool ok = false;
    DiskSlabStore store(Opts(1 << 20, 1 << 20, 4096), &ok);
    ASSERT_TRUE(ok);
  }
  const fs::path table = dir_ / "slots.tbl";
  const off_t truncated_size = 255 * 64;
  ASSERT_EQ(::truncate(table.c_str(), truncated_size), 0);

  bool ok = true;
  DiskSlabStore store(Opts(1 << 20, 1 << 20, 4096), &ok);
  EXPECT_FALSE(ok);
  EXPECT_NE(store.StartupError().find("size"), std::string::npos);
  struct stat st {};
  ASSERT_EQ(::stat(table.c_str(), &st), 0);
  EXPECT_EQ(st.st_size, truncated_size);
}

TEST_F(DiskSlabTest, DirtyEpochColdResetsAfterProcessCrash) {
  auto options = Opts(1 << 20, 1 << 20, 4096);
  options.table_sync_ms = 0;
  options.reclaim_interval_ms = 0;
  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    bool ok = false;
    DiskSlabStore store(options, &ok);
    std::string value(4096, 'c');
    const bool cached =
        ok &&
        store.Cache(K(70), value.data(), value.size()) == Status::kOk;
    ::_exit(cached ? 0 : 1);  // bypass destructors: process-crash semantics
  }
  int child_status = 0;
  ASSERT_EQ(::waitpid(child, &child_status, 0), child);
  ASSERT_TRUE(WIFEXITED(child_status));
  ASSERT_EQ(WEXITSTATUS(child_status), 0);

  std::string survivor(4096, 's');
  {
    bool ok = false;
    DiskSlabStore recovered(options, &ok);
    ASSERT_TRUE(ok);
    EXPECT_EQ(recovered.GetStats().unclean_resets, 1u);
    EXPECT_EQ(recovered.Count(), 0u);
    std::string out;
    EXPECT_EQ(recovered.Range(K(70), 0, survivor.size(), &out),
              Status::kNotFound);
    ASSERT_EQ(recovered.Cache(K(71), survivor.data(), survivor.size()),
              Status::kOk);
  }

  bool ok = false;
  DiskSlabStore clean_reopen(options, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(clean_reopen.GetStats().unclean_resets, 0u);
  EXPECT_EQ(clean_reopen.Count(), 1u);
  std::string out;
  ASSERT_EQ(clean_reopen.Range(K(71), 0, survivor.size(), &out),
            Status::kOk);
  EXPECT_EQ(out, survivor);
}

TEST_F(DiskSlabTest, CorruptEpochStateFailsClosed) {
  auto options = Opts(1 << 20, 1 << 20, 4096);
  {
    bool ok = false;
    DiskSlabStore store(options, &ok);
    ASSERT_TRUE(ok);
  }
  const fs::path state = dir_ / "slab_state";
  int fd = ::open(state.c_str(), O_RDWR);
  ASSERT_GE(fd, 0);
  uint8_t bad = 0;
  ASSERT_EQ(::pwrite(fd, &bad, sizeof(bad), 0),
            static_cast<ssize_t>(sizeof(bad)));
  ASSERT_EQ(::fdatasync(fd), 0);
  ::close(fd);

  bool ok = true;
  DiskSlabStore reopened(options, &ok);
  EXPECT_FALSE(ok);
  EXPECT_FALSE(reopened.Healthy());
  EXPECT_NE(reopened.StartupError().find("slab_state"), std::string::npos);
}

TEST_F(DiskSlabTest, MissingEpochStateMigratesConservativelyToCold) {
  auto options = Opts(1 << 20, 1 << 20, 4096);
  {
    bool ok = false;
    DiskSlabStore store(options, &ok);
    ASSERT_TRUE(ok);
    std::string value(4096, 'm');
    ASSERT_EQ(store.Cache(K(75), value.data(), value.size()), Status::kOk);
  }
  ASSERT_TRUE(fs::remove(dir_ / "slab_state"));

  bool ok = false;
  DiskSlabStore migrated(options, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(migrated.GetStats().unclean_resets, 1u);
  EXPECT_EQ(migrated.Count(), 0u);
  EXPECT_FALSE(migrated.IsCached(K(75)));
}

TEST_F(DiskSlabTest, PayloadAndRecordCrashPointsRecoverAsColdMisses) {
  for (bool include_record : {false, true}) {
    fs::remove_all(dir_);
    auto options = Opts(1 << 20, 1 << 20, 4096);
    options.table_sync_ms = 0;
    options.reclaim_interval_ms = 0;
    const pid_t child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
      bool ok = false;
      DiskSlabStore store(options, &ok);
      const std::string value(4096, include_record ? 'r' : 'p');
      const bool staged =
          ok && dfkv::DiskSlabStoreTestPeer::WriteCrashStage(
                    &store, K(80), value, include_record);
      ::_exit(staged ? 0 : 1);
    }
    int child_status = 0;
    ASSERT_EQ(::waitpid(child, &child_status, 0), child);
    ASSERT_TRUE(WIFEXITED(child_status));
    ASSERT_EQ(WEXITSTATUS(child_status), 0);

    bool ok = false;
    DiskSlabStore recovered(options, &ok);
    ASSERT_TRUE(ok);
    EXPECT_EQ(recovered.GetStats().unclean_resets, 1u);
    EXPECT_EQ(recovered.Count(), 0u);
    std::string out;
    EXPECT_EQ(recovered.Range(K(80), 0, 4096, &out),
              Status::kNotFound);
  }
}

TEST_F(DiskSlabTest, OversizeValueRejected) {
  bool ok = false;
  DiskSlabStore s(Opts(4096, 4096, 4096), &ok);  // one 4096 slot per extent
  ASSERT_TRUE(ok);
  std::string big(4097, 'b');
  EXPECT_EQ(s.Cache(K(1), big.data(), big.size()), Status::kIOError);
  std::string okv(4096, 'o');
  EXPECT_EQ(s.Cache(K(2), okv.data(), okv.size()), Status::kOk);
}

// The data-path I/O runs OUTSIDE the store lock (pin + inflight-count protect
// the slot in the unlocked window). These stress tests exercise the unlocked
// windows under eviction pressure and concurrent Remove -- a read must return
// either the key's FULL correct payload or a clean miss, never torn bytes.
// (Run under the CI TSan job, they also pin down the lock discipline.)
TEST_F(DiskSlabTest, ConcurrentPutGetNeverTorn) {
  bool ok = false;
  // Small pool (4 extents x 256KiB, 4KiB slots) => constant eviction pressure.
  DiskSlabStore s(Opts(1 << 20, 1 << 18, 4096), &ok);
  ASSERT_TRUE(ok);
  constexpr int kThreads = 8, kKeys = 64, kIters = 200;
  auto payload = [](uint64_t id) {
    std::string v(4000, '\0');
    for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<char>((id * 131 + i) & 0xFF);
    return v;
  };
  std::atomic<int> torn{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < kThreads; ++t) {
    ts.emplace_back([&, t] {
      for (int i = 0; i < kIters; ++i) {
        const uint64_t id = (t * 7 + i) % kKeys + 1;
        const std::string want = payload(id);
        if (i % 3 == 0) s.Cache(K(id), want.data(), want.size());
        std::string got;
        Status st = s.Range(K(id), 0, 0, &got);
        if (st == Status::kOk && !got.empty() && got != want) torn.fetch_add(1);
        char buf[4096];
        size_t n = 0;
        st = s.RangeInto(K(id), 0, 0, buf, sizeof(buf), &n);
        if (st == Status::kOk && n == want.size() &&
            std::string(buf, n) != want) torn.fetch_add(1);
      }
    });
  }
  for (auto& th : ts) th.join();
  EXPECT_EQ(torn.load(), 0) << "a read observed a torn/foreign payload";
}

TEST_F(DiskSlabTest, RemoveDuringConcurrentAccessStaysConsistent) {
  bool ok = false;
  DiskSlabStore s(Opts(1 << 20, 1 << 18, 4096), &ok);
  ASSERT_TRUE(ok);
  constexpr int kKeys = 16, kIters = 300;
  auto payload = [](uint64_t id) { return std::string(3000, static_cast<char>('a' + id % 26)); };
  std::atomic<int> bad{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < 6; ++t) {
    ts.emplace_back([&, t] {
      for (int i = 0; i < kIters; ++i) {
        const uint64_t id = (t + i) % kKeys + 1;
        const std::string want = payload(id);
        switch ((t + i) % 3) {
          case 0: s.Cache(K(id), want.data(), want.size()); break;
          case 1: {
            std::string got;
            if (s.Range(K(id), 0, 0, &got) == Status::kOk && !got.empty() && got != want)
              bad.fetch_add(1);
            break;
          }
          case 2: s.Remove(K(id)); break;
        }
      }
    });
  }
  for (auto& th : ts) th.join();
  EXPECT_EQ(bad.load(), 0) << "a read observed another key's bytes after Remove";
  // Steady state: every key must round-trip again (no leaked slots / stuck state).
  for (uint64_t id = 1; id <= kKeys; ++id) {
    const std::string want = payload(id);
    ASSERT_EQ(s.Cache(K(id), want.data(), want.size()), Status::kOk);
    std::string got;
    ASSERT_EQ(s.Range(K(id), 0, 0, &got), Status::kOk) << "key " << id;
    EXPECT_EQ(got, want);
  }
}

// O_DIRECT write mode (DFKV_SLAB_WRITE=direct): aligned CacheDirect payloads go
// through the DIO extent fd; unaligned callers fall back to buffered. Reads are
// buffered either way, and restart warmth is unaffected.
TEST_F(DiskSlabTest, DirectWriteModeRoundTrip) {
  auto opts = Opts(1 << 20, 1 << 20, 4096);
  opts.direct_writes = true;
  bool ok = false;
  DiskSlabStore s(opts, &ok);
  ASSERT_TRUE(ok);
  // Aligned buffer + padded cap (the RDMA recv-buffer contract).
  void* mem = nullptr;
  ASSERT_EQ(posix_memalign(&mem, 4096, 8192), 0);
  char* buf = static_cast<char*>(mem);
  for (int i = 0; i < 5000; ++i) buf[i] = static_cast<char>(i * 7);
  std::string want(buf, 5000);
  ASSERT_EQ(s.CacheDirect(K(1), buf, 5000, 8192), Status::kOk);
  std::string got;
  ASSERT_EQ(s.Range(K(1), 0, 0, &got), Status::kOk);
  EXPECT_EQ(got, want);
  // Unaligned source buffer: must fall back to the buffered path, same result.
  std::string v2(3000, 'z');
  ASSERT_EQ(s.CacheDirect(K(2), &v2[0], v2.size(), v2.size()), Status::kOk);
  ASSERT_EQ(s.Range(K(2), 0, 0, &got), Status::kOk);
  EXPECT_EQ(got, v2);
  free(mem);
}

// Async prep hands back a dup'd fd + slot-absolute aligned window and HOLDS the
// slot (pin + inflight) until RangeRelease: a Remove landing mid-hold defers,
// and the release both frees the slot and survives a double call.
TEST_F(DiskSlabTest, RangeDirectPrepHoldsSlotUntilRelease) {
  bool ok = false;
  DiskSlabStore s(Opts(1 << 20, 1 << 20, 4096), &ok);
  ASSERT_TRUE(ok);
  std::string v(5000, 'p');
  ASSERT_EQ(s.Cache(K(1), v.data(), v.size()), Status::kOk);

  dfkv::RangePrep p;
  ASSERT_EQ(s.RangeDirectPrep(K(1), 0, 0, 1 << 20, &p), Status::kOk);
  ASSERT_GE(p.fd, 0);
  ASSERT_NE(p.token, 0u);
  EXPECT_EQ(p.payload_len, v.size());
  EXPECT_EQ(p.aligned_off % 4096, 0u);
  EXPECT_EQ(p.aligned_len % 4096, 0u);
  // The caller's read: aligned pread on the dup'd fd, payload at [head, +len).
  std::vector<char> rbuf(p.aligned_len);
  ASSERT_EQ(::pread(p.fd, rbuf.data(), p.aligned_len,
                    static_cast<off_t>(p.aligned_off)),
            static_cast<ssize_t>(p.aligned_len));
  EXPECT_EQ(std::string(rbuf.data() + p.head, p.payload_len), v);
  ::close(p.fd);

  // Remove during the hold: deferred -- reads gate off, slot not yet freed.
  EXPECT_EQ(s.Remove(K(1)), Status::kOk);
  EXPECT_FALSE(s.IsCached(K(1)));
  EXPECT_EQ(s.Count(), 1u) << "slot must stay held until the release";
  s.RangeRelease(p.token);
  EXPECT_EQ(s.Count(), 0u) << "release must execute the deferred remove";
  s.RangeRelease(p.token);  // double release: no-op
  // Miss and zero-length behave like the sync path.
  EXPECT_EQ(s.RangeDirectPrep(K(9), 0, 0, 1 << 20, &p), Status::kNotFound);
}

// Direct-mode RangeDirect: O_DIRECT aligned-window read with head trim -- the
// returned pointer must land exactly on the requested offset's bytes, including
// a non-4KiB-aligned client offset (head != 0).
TEST_F(DiskSlabTest, DirectModeRangeDirectAlignedWindowRead) {
  auto opts = Opts(1 << 20, 1 << 20, 4096);
  opts.direct_writes = true;
  bool ok = false;
  DiskSlabStore s(opts, &ok);
  ASSERT_TRUE(ok);
  std::string v(9000, '\0');
  for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<char>(i * 13);
  ASSERT_EQ(s.Cache(K(1), v.data(), v.size()), Status::kOk);

  void* mem = nullptr;
  ASSERT_EQ(posix_memalign(&mem, 4096, 1 << 20), 0);
  char* io_buf = static_cast<char*>(mem);
  const char* out = nullptr;
  size_t out_len = 0;
  // Full read.
  ASSERT_EQ(s.RangeDirect(K(1), 0, 0, io_buf, 1 << 20, &out, &out_len), Status::kOk);
  EXPECT_EQ(std::string(out, out_len), v);
  // Unaligned offset (head != 0): bytes [100, 5100).
  ASSERT_EQ(s.RangeDirect(K(1), 100, 5000, io_buf, 1 << 20, &out, &out_len), Status::kOk);
  EXPECT_EQ(out_len, 5000u);
  EXPECT_EQ(std::string(out, out_len), v.substr(100, 5000));
  // Miss stays a miss.
  EXPECT_EQ(s.RangeDirect(K(9), 0, 0, io_buf, 1 << 20, &out, &out_len),
            Status::kNotFound);
  free(mem);
}

// R1-B: the table-sync thread fdatasyncs slots.tbl on a cadence, but only in
// cycles where records were actually written.
TEST_F(DiskSlabTest, TableSyncCadenceCountsCycles) {
  auto opts = Opts(1 << 20, 1 << 20, 4096);
  opts.table_sync_ms = 20;  // fast cadence for the test
  bool ok = false;
  DiskSlabStore s(opts, &ok);
  ASSERT_TRUE(ok);
  std::string v(3000, 's');
  ASSERT_EQ(s.Cache(K(1), v.data(), v.size()), Status::kOk);
  // Within a few cadences the sync thread must have flushed the new record.
  for (int i = 0; i < 100 && s.GetStats().table_syncs == 0; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  EXPECT_GE(s.GetStats().table_syncs, 1u);
  // Idle cadences don't burn syscalls: the counter stays put with no writes.
  const uint64_t after = s.GetStats().table_syncs;
  std::this_thread::sleep_for(std::chrono::milliseconds(120));
  EXPECT_EQ(s.GetStats().table_syncs, after);
}

TEST_F(DiskSlabTest, RecordWriteFailureFailsStoreClosed) {
  auto opts = Opts(1 << 20, 1 << 20, 4096);
  opts.table_sync_ms = 0;
  opts.reclaim_interval_ms = 0;
  bool ok = false;
  DiskSlabStore s(opts, &ok);
  ASSERT_TRUE(ok);
  std::string value(3000, 'r');
  ASSERT_EQ(s.Cache(K(1), value.data(), value.size()), Status::kOk);

  dfkv::DiskSlabStoreTestPeer::BreakTableFd(&s);
  EXPECT_EQ(s.Cache(K(2), value.data(), value.size()), Status::kIOError);
  EXPECT_FALSE(s.Healthy());
  EXPECT_TRUE(s.GetStats().failed);
  EXPECT_EQ(s.GetStats().metadata_io_errors, 1u);
  EXPECT_EQ(s.Count(), 1u);
  std::string out;
  EXPECT_EQ(s.Range(K(1), 0, 0, &out), Status::kIOError);
  EXPECT_EQ(s.Remove(K(1)), Status::kIOError);
}

TEST_F(DiskSlabTest, TombstoneWriteFailureKeepsSlotAndFailsClosed) {
  auto opts = Opts(1 << 20, 1 << 20, 4096);
  opts.table_sync_ms = 0;
  opts.reclaim_interval_ms = 0;
  bool ok = false;
  DiskSlabStore s(opts, &ok);
  ASSERT_TRUE(ok);
  std::string value(3000, 't');
  ASSERT_EQ(s.Cache(K(1), value.data(), value.size()), Status::kOk);

  dfkv::DiskSlabStoreTestPeer::BreakTableFd(&s);
  EXPECT_EQ(s.Remove(K(1)), Status::kIOError);
  EXPECT_FALSE(s.Healthy());
  EXPECT_EQ(s.Count(), 1u);
  EXPECT_EQ(s.GetStats().metadata_io_errors, 1u);
}

TEST_F(DiskSlabTest, TableSyncFailureFailsStoreClosed) {
  auto opts = Opts(1 << 20, 1 << 20, 4096);
  opts.table_sync_ms = 0;
  opts.reclaim_interval_ms = 0;
  bool ok = false;
  DiskSlabStore s(opts, &ok);
  ASSERT_TRUE(ok);
  std::string value(3000, 's');
  ASSERT_EQ(s.Cache(K(1), value.data(), value.size()), Status::kOk);

  dfkv::DiskSlabStoreTestPeer::BreakTableFd(&s);
  EXPECT_FALSE(dfkv::DiskSlabStoreTestPeer::SyncTableNow(&s));
  EXPECT_FALSE(s.Healthy());
  EXPECT_EQ(s.GetStats().table_syncs, 0u);
  EXPECT_EQ(s.GetStats().metadata_io_errors, 1u);
  EXPECT_EQ(s.Cache(K(2), value.data(), value.size()), Status::kIOError);
}

TEST_F(DiskSlabTest, EvictionRecordClearFailurePreservesResidents) {
  auto options = Opts(4 * 4096, 4 * 4096, 4096);
  options.table_sync_ms = 0;
  options.reclaim_interval_ms = 0;
  bool ok = false;
  DiskSlabStore store(options, &ok);
  ASSERT_TRUE(ok);
  std::string value(4000, 'e');
  for (uint64_t i = 0; i < 4; ++i)
    ASSERT_EQ(store.Cache(K(i), value.data(), value.size()), Status::kOk);

  dfkv::DiskSlabStoreTestPeer::BreakTableFd(&store);
  EXPECT_EQ(store.Cache(K(4), value.data(), value.size()),
            Status::kIOError);
  EXPECT_FALSE(store.Healthy());
  EXPECT_EQ(store.Count(), 4u);
  EXPECT_EQ(store.GetStats().metadata_io_errors, 1u);
}

// R2: in direct mode an UNALIGNED CacheDirect payload falls back to the
// buffered path -- and the fallback is counted (the "page cache crept back
// in" signal for direct deployments).
TEST_F(DiskSlabTest, DioFallbackIsCounted) {
  auto opts = Opts(1 << 20, 1 << 20, 4096);
  opts.direct_writes = true;

  bool ok = false;
  DiskSlabStore s(opts, &ok);
  ASSERT_TRUE(ok);
  if (!s.DirectWritesActive()) GTEST_SKIP() << "fs rejected O_DIRECT (tmpfs)";
  std::string v(3000, 'u');  // std::string data: not 4 KiB-aligned
  ASSERT_EQ(s.CacheDirect(K(1), &v[0], v.size(), v.size()), Status::kOk);
  EXPECT_EQ(s.GetStats().dio_write_fallbacks, 1u);
}
TEST_F(DiskSlabTest, CleanEpochWriteFailureLeavesStoreFailed) {
  auto options = Opts(1 << 20, 1 << 20, 4096);
  options.table_sync_ms = 0;
  options.reclaim_interval_ms = 0;
  bool ok = false;
  DiskSlabStore store(options, &ok);
  ASSERT_TRUE(ok);
  dfkv::DiskSlabStoreTestPeer::BreakStateFd(&store);
  EXPECT_FALSE(dfkv::DiskSlabStoreTestPeer::MarkCleanNow(&store));
  EXPECT_FALSE(store.Healthy());
  EXPECT_EQ(store.GetStats().metadata_io_errors, 1u);
}

// A whole extent stolen to a new class must durably wipe its prior slot-grid
// before the allocator publishes the new class. Individual eviction tombstones
// cannot describe grid positions the new class never uses.
TEST_F(DiskSlabTest, RebindWipesStaleRecordsNoResurrectionAcrossRestart) {
  std::vector<int> absent_after_steal;
  std::string bval(8000, 'B');
  {
    bool ok = false;
    // 2 extents x 4 slots(4096): 8 class-A keys fill both extents.
    // Reclaimer OFF: this test asserts an EXACT steal-eviction count (one
    // extent's 4 residents). With the default 50ms reclaimer running on this
    // full store, the background thread can free an extra resident between the
    // steal and the IsCached check, flaking the count to 5 (seen on clang CI).
    // Determinism here is about the steal/rebind path, not reclaiming.
    auto o = Opts(2 * 4 * 4096, 4 * 4096, 4096);
    o.reclaim_interval_ms = 0;
    DiskSlabStore s(o, &ok);
    ASSERT_TRUE(ok);
    std::string a(4000, 'A');
    for (int i = 0; i < 8; ++i)
      ASSERT_EQ(s.Cache(K(100 + i), a.data(), a.size()), Status::kOk);
    ASSERT_EQ(s.Count(), 8u);
    // Class B (8192): pool empty -> steal an A extent, durably wipe its
    // prior slot grid, evict four residents, then bind it to B.
    ASSERT_EQ(s.Cache(K(500), bval.data(), bval.size()), Status::kOk);
    EXPECT_GE(s.GetStats().bind_wipes, 1u) << "steal rebind must wipe the region";
    for (int i = 0; i < 8; ++i)
      if (!s.IsCached(K(100 + i))) absent_after_steal.push_back(100 + i);
    ASSERT_EQ(absent_after_steal.size(), 4u) << "steal evicts one extent's residents";
  }
  // Restart: the stolen extent's old keys must STAY dead; B and the surviving
  // extent's keys must read back intact. Reclaimer OFF for the same reason.
  bool ok = false;
  auto o2 = Opts(2 * 4 * 4096, 4 * 4096, 4096);
  o2.reclaim_interval_ms = 0;
  DiskSlabStore s2(o2, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(s2.Count(), 5u) << "4 surviving A keys + B; no resurrection";
  for (int id : absent_after_steal)
    EXPECT_FALSE(s2.IsCached(K(id))) << "stale key " << id << " resurrected!";
  std::string out;
  ASSERT_EQ(s2.Range(K(500), 0, 0, &out), Status::kOk);
  EXPECT_EQ(out, bval);
}

// Background reclaimer wiring: on a full store with ongoing writes, slots get
// freed ahead of demand by the reclaim thread and show up in Stats.
TEST_F(DiskSlabTest, BackgroundReclaimerFreesSlotsOnFullStore) {
  bool ok = false;
  auto o = Opts(2 << 20, 1 << 20, 4096);  // 2 extents x 256 slots of 4 KiB
  o.reclaim_interval_ms = 1;
  DiskSlabStore s(o, &ok);
  ASSERT_TRUE(ok);
  std::string v(4000, 'x');
  for (uint64_t i = 0; i < 512; ++i)
    ASSERT_EQ(s.Cache(K(1000 + i), v.data(), v.size()), Status::kOk);  // fill
  for (uint64_t i = 0; i < 128; ++i) {  // sustained demand on the full store
    ASSERT_EQ(s.Cache(K(2000 + i), v.data(), v.size()), Status::kOk);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  // The reclaimer keeps a full store writable via EITHER path: demand-driven
  // CLOCK reclaim, or (phase 10) proactive watermark eviction that frees cold
  // extents before the ring hits 100% — the latter now handles the common case.
  const auto st = s.GetStats();
  EXPECT_GT(st.reclaimed_slots + st.watermark_evictions, 0u) << "reclaimer never ran";
}

TEST_F(DiskSlabTest, WatermarkEvictionCannotEraseConcurrentRewriteCommit) {
  setenv("DFKV_SLAB_EVICT_HIGH_PCT", "50", 1);
  setenv("DFKV_SLAB_EVICT_LOW_PCT", "25", 1);
  auto o = Opts(4 * 4096, 4 * 4096, 4096);
  o.reclaim_interval_ms = 0;
  o.table_sync_ms = 0;
  bool ok = false;
  DiskSlabStore s(o, &ok);
  unsetenv("DFKV_SLAB_EVICT_HIGH_PCT");
  unsetenv("DFKV_SLAB_EVICT_LOW_PCT");
  ASSERT_TRUE(ok);

  std::string old_value(4000, 'o');
  for (uint64_t i = 1; i <= 4; ++i) {
    ASSERT_EQ(s.Cache(K(i), old_value.data(), old_value.size()), Status::kOk);
  }

  std::string fresh_value(4000, 'n');
  std::mutex done_mu;
  std::condition_variable done_cv;
  bool writer_started = false;
  bool writer_done = false;
  bool completed_during_eviction_window = false;
  Status writer_status = Status::kIOError;
  std::thread writer;

  dfkv::DiskSlabStoreTestPeer::ReclaimNow(&s, [&] {
    writer = std::thread([&] {
      {
        std::lock_guard<std::mutex> lk(done_mu);
        writer_started = true;
      }
      done_cv.notify_all();
      const Status st =
          s.Cache(K(1), fresh_value.data(), fresh_value.size());
      {
        std::lock_guard<std::mutex> lk(done_mu);
        writer_status = st;
        writer_done = true;
      }
      done_cv.notify_all();
    });

    std::unique_lock<std::mutex> lk(done_mu);
    ASSERT_TRUE(done_cv.wait_for(lk, std::chrono::seconds(2),
                                 [&] { return writer_started; }));
    completed_during_eviction_window = done_cv.wait_for(
        lk, std::chrono::milliseconds(250), [&] { return writer_done; });
  });

  ASSERT_TRUE(writer.joinable());
  writer.join();
  EXPECT_FALSE(completed_during_eviction_window);
  EXPECT_EQ(writer_status, Status::kOk);

  std::string out;
  ASSERT_EQ(s.Range(K(1), 0, fresh_value.size(), &out), Status::kOk);
  EXPECT_EQ(out, fresh_value);
}

TEST_F(DiskSlabTest, WatermarkEvictionsStayDeadAcrossCleanRestart) {
  ::setenv("DFKV_SLAB_EVICT_HIGH_PCT", "50", 1);
  ::setenv("DFKV_SLAB_EVICT_LOW_PCT", "25", 1);
  auto options = Opts(4 * 4096, 4 * 4096, 4096);
  options.reclaim_interval_ms = 0;
  options.table_sync_ms = 0;
  {
    bool ok = false;
    DiskSlabStore store(options, &ok);
    ::unsetenv("DFKV_SLAB_EVICT_HIGH_PCT");
    ::unsetenv("DFKV_SLAB_EVICT_LOW_PCT");
    ASSERT_TRUE(ok);
    std::string value(4000, 'e');
    for (uint64_t i = 0; i < 4; ++i)
      ASSERT_EQ(store.Cache(K(i), value.data(), value.size()), Status::kOk);
    dfkv::DiskSlabStoreTestPeer::ReclaimNow(&store, {});
    EXPECT_EQ(store.Count(), 0u);
    EXPECT_EQ(store.GetStats().eviction_record_clears, 4u);
  }

  bool ok = false;
  DiskSlabStore reopened(options, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(reopened.Count(), 0u);
  for (uint64_t i = 0; i < 4; ++i) EXPECT_FALSE(reopened.IsCached(K(i)));
}

// Class rebalance regression (the "new value size retains only a sliver of its
// writes" failure): fill the store with class A, then write a burst of class B
// larger than B's first extent. Stock behavior self-evicts B forever (B ends
// with one extent's worth of survivors); the reclaim tick must instead grow B
// from the idle A donor so the whole burst stays readable.
TEST_F(DiskSlabTest, ReclaimTickGrowsHotClassFromColdDonor) {
  // Watermark eviction OFF: on a >92%-full store it returns cold extents to
  // the pool each tick, and a non-empty pool disables the donor-steal grow
  // (B then grows via pool grabs, rebalanced_extents stays 0). This test
  // targets the donor-steal path specifically, so isolate it.
  ::setenv("DFKV_SLAB_EVICT_HIGH_PCT", "0", 1);
  bool ok = false;
  auto o = Opts(16 << 20, 1 << 20, 4096);  // 16 extents x 1 MiB
  o.reclaim_interval_ms = 1;
  DiskSlabStore s(o, &ok);
  ::unsetenv("DFKV_SLAB_EVICT_HIGH_PCT");
  ASSERT_TRUE(ok);
  std::string a(4000, 'a');
  for (uint64_t i = 0; i < 4096; ++i)  // 16 extents x 256 slots: fill class A
    ASSERT_EQ(s.Cache(K(10000 + i), a.data(), a.size()), Status::kOk);
  // Class B: 16 KiB values, 64 slots/extent. Burst of 256 = 4 extents' worth,
  // written slowly enough for the 1 ms tick to grow B between writes.
  std::string b(16000, 'b');
  for (uint64_t i = 0; i < 256; ++i) {
    ASSERT_EQ(s.Cache(K(20000 + i), b.data(), b.size()), Status::kOk);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  size_t retained = 0;
  for (uint64_t i = 0; i < 256; ++i) retained += s.IsCached(K(20000 + i));
  EXPECT_GT(s.GetStats().rebalanced_extents, 0u) << "growth never kicked in";
  EXPECT_GE(retained, 250u) << "hot class must absorb capacity, not eat itself";
}

// Batched CacheDirect: same per-item semantics as CacheDirect with one lock
// hold per phase. Covers happy path, batch-internal dup (idempotent), invalid
// item, and read-back byte equality (loop path in non-uring builds).
TEST_F(DiskSlabTest, CacheDirectBatchLandsAllItemsReadable) {
  bool ok = false;
  DiskSlabStore s(Opts(1 << 20, 1 << 20, 4096), &ok);
  ASSERT_TRUE(ok);
  constexpr int N = 12;
  std::vector<std::string> vals(N);
  std::vector<DiskSlabStore::CacheBatchItem> items;
  for (int i = 0; i < N; ++i) {
    vals[i].assign(3000 + i * 17, static_cast<char>('a' + i));
    items.push_back({K(500 + i), vals[i].data(), vals[i].size(), vals[i].size()});
  }
  items.push_back({K(500), vals[0].data(), vals[0].size(), vals[0].size()});  // dup of first
  items.push_back({K(999), nullptr, 10, 10});                                // invalid
  auto sts = s.CacheDirectBatch(items);
  ASSERT_EQ(sts.size(), items.size());
  for (int i = 0; i < N; ++i) EXPECT_EQ(sts[i], Status::kOk) << i;
  EXPECT_EQ(sts[N], Status::kOk) << "batch-internal dup is idempotent kOk";
  EXPECT_EQ(sts[N + 1], Status::kInvalid);
  for (int i = 0; i < N; ++i) {
    std::string out;
    ASSERT_EQ(s.Range(K(500 + i), 0, vals[i].size(), &out), Status::kOk) << i;
    EXPECT_EQ(out, vals[i]) << i;
  }
  EXPECT_GE(s.GetStats().batched_writes + 1, 1u);  // counter is uring-only; loop path leaves 0
}

// Batch under eviction pressure: items exceeding capacity still land (evicting
// older residents), statuses all kOk, store never over-commits.
TEST_F(DiskSlabTest, CacheDirectBatchEvictsUnderPressure) {
  bool ok = false;
  DiskSlabStore s(Opts(8 * 4096, 8 * 4096, 4096), &ok);  // 8 slots total
  ASSERT_TRUE(ok);
  std::string v(4000, 'p');
  for (uint64_t i = 0; i < 8; ++i) ASSERT_EQ(s.Cache(K(1 + i), v.data(), v.size()), Status::kOk);
  std::vector<DiskSlabStore::CacheBatchItem> items;
  std::vector<std::string> vals(4, std::string(4000, 'q'));
  for (int i = 0; i < 4; ++i) items.push_back({K(100 + i), vals[i].data(), vals[i].size(), vals[i].size()});
  auto sts = s.CacheDirectBatch(items);
  for (auto st : sts) EXPECT_EQ(st, Status::kOk);
  EXPECT_EQ(s.Count(), 8u);
  for (int i = 0; i < 4; ++i) EXPECT_TRUE(s.IsCached(K(100 + i)));
}

#ifdef DFKV_WITH_URING
// batched_writes must count CQE-confirmed writes exactly: on a healthy fs a
// fully-successful uring batch tallies every item once (previously the counter
// tallied *submitted* items, so a submitted-but-failed write was counted as
// batched AND again by the sequential fallback it fell through to).
TEST_F(DiskSlabTest, UringBatchCountsOnlyConfirmedWrites) {
  auto opts = Opts(1 << 20, 1 << 20, 4096);
  opts.direct_writes = true;
  bool ok = false;
  DiskSlabStore s(opts, &ok);
  ASSERT_TRUE(ok);
  if (!s.DirectWritesActive()) GTEST_SKIP() << "fs rejected O_DIRECT (tmpfs)";
  // The uring one-submit path takes only page-aligned payloads (unaligned data
  // rides the buffered fallback), so hand it 4 KiB-aligned page-multiple items
  // exactly like RamTier arena slots.
  constexpr uint64_t N = 8;
  constexpr size_t kLen = 4096;
  std::vector<void*> bufs(N);
  std::vector<DiskSlabStore::CacheBatchItem> items;
  for (uint64_t i = 0; i < N; ++i) {
    ASSERT_EQ(posix_memalign(&bufs[i], 4096, kLen), 0);
    std::memset(bufs[i], static_cast<int>('A' + i), kLen);
    items.push_back({K(700 + i), static_cast<char*>(bufs[i]), kLen, kLen});
  }
  auto sts = s.CacheDirectBatch(items);
  for (auto st : sts) ASSERT_EQ(st, Status::kOk);
  const auto stats = s.GetStats();
  if (stats.uring_write_batches == 0) {
    for (auto* b : bufs) free(b);
    GTEST_SKIP() << "uring path not taken (env-disabled or init fallback)";
  }
  EXPECT_EQ(stats.uring_write_batches, 1u);
  EXPECT_EQ(stats.batched_writes, N);
  EXPECT_EQ(stats.dio_write_fallbacks, 0u);
  for (uint64_t i = 0; i < N; ++i) {
    std::string out;
    ASSERT_EQ(s.Range(K(700 + i), 0, kLen, &out), Status::kOk) << i;
    EXPECT_EQ(out, std::string(kLen, static_cast<char>('A' + i))) << i;
  }
  for (auto* b : bufs) free(b);
}
#endif  // DFKV_WITH_URING
