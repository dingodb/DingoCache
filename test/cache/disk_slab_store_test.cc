// DiskSlabStore: extent-file slab store on SlabAllocator + slots.tbl rebuild.
// Covers data-path semantics, restart warmth, crash/torn-record handling,
// strict startup geometry, fail-closed metadata I/O, and reclaimer safety.
#include "cache/disk_slab_store.h"
#include "utils/crc32.h"
#include "utils/net_util.h"

#include <gtest/gtest.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/resource.h>

#ifdef DFKV_WITH_URING
#include <liburing.h>
#endif

#include <cerrno>
#include <csignal>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iterator>
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
  // A read or write in flight holds its slot like this.
  static bool PinSlot(DiskSlabStore* store, const BlockKey& key) {
    std::lock_guard<std::mutex> lock(store->mu_);
    return store->alloc_->Pin(key);
  }
  static bool UnpinSlot(DiskSlabStore* store, const BlockKey& key) {
    std::lock_guard<std::mutex> lock(store->mu_);
    return store->alloc_->Unpin(key);
  }
  static void SetDropCorruptHook(DiskSlabStore* store,
                                 std::function<void()> hook) {
    store->drop_corrupt_hook_for_test_ = std::move(hook);
  }
  static SlabAllocator::SlotRef SlotOf(DiskSlabStore* store,
                                       const BlockKey& key) {
    std::lock_guard<std::mutex> lock(store->mu_);
    SlabAllocator::SlotRef ref;
    store->alloc_->Get(key, &ref);
    return ref;
  }
  static void BreakTableFd(DiskSlabStore* store) {
    if (store->table_fd_ >= 0) ::close(store->table_fd_);
    store->table_fd_ = -1;
  }

  static bool SyncTableNow(DiskSlabStore* store) {
    return store->SyncTable(store->TableWriteStamp());
  }
  static bool WriteCrashStage(DiskSlabStore* store, const BlockKey& key,
                              const std::string& value,
                              bool include_record) {
    const std::string filename = key.Filename();
    SlabAllocator::SlotRef ref;
    {
      std::lock_guard<std::mutex> lock(store->mu_);
      std::vector<BlockKey> evicted;
      if (!store->alloc_->Put(key, value.size(), &ref, &evicted))
        return false;
      if (!store->alloc_->Pin(key)) return false;
      store->inflight_[filename]++;
    }
    if (!store->WritePayload(ref, value.data(), value.size())) return false;
    return !include_record ||
           store->WriteRecord(ref, key,
                              static_cast<uint32_t>(value.size()), true);
  }
  static void SetPayloadWriteHook(DiskSlabStore* store,
                                  std::function<bool()> hook) {
    store->payload_write_hook_for_test_ = std::move(hook);
  }
  static void SetRecordWriteHook(DiskSlabStore* store,
                                 std::function<bool()> hook) {
    store->record_write_hook_for_test_ = std::move(hook);
  }
  // Production never sets this; when set it replaces every CQE reap of the
  // batched io_uring write path (blocking wait AND post-failure drain).
  static void SetUringReapHook(DiskSlabStore* store,
                               std::function<int(void*, void*)> hook) {
    store->uring_reap_hook_for_test_ = std::move(hook);
  }
  static void BreakStateFd(DiskSlabStore* store) {
    if (store->state_fd_ >= 0) ::close(store->state_fd_);
    store->state_fd_ = -1;
  }
  static bool MarkCleanNow(DiskSlabStore* store) {
    return store->MarkCleanEpoch();
  }
  static void SetTableSyncHook(DiskSlabStore* store,
                               std::function<void()> hook) {
    store->table_sync_hook_for_test_ = std::move(hook);
  }
  static void SetBindHook(DiskSlabStore* store,
                          std::function<void(uint32_t)> hook) {
    store->bind_hook_for_test_ = std::move(hook);
  }
  // True iff slots.tbl holds a valid record for key right now. Reads the file,
  // not the allocator: this is what a rebuild after a restart would find.
  static bool HasValidRecordOnDisk(DiskSlabStore* store, const BlockKey& key) {
    struct stat st {};
    if (::fstat(store->table_fd_, &st) != 0) return false;
    std::vector<char> table(static_cast<size_t>(st.st_size));
    if (::pread(store->table_fd_, table.data(), table.size(), 0) !=
        static_cast<ssize_t>(table.size()))
      return false;
    for (size_t at = 0; at + DiskSlabStore::kRecBytes <= table.size();
         at += DiskSlabStore::kRecBytes) {
      const char* rec = table.data() + at;
      if (net::GetU32(rec) == 0 || rec[8] != 1) continue;
      if (net::GetU64(rec + 16) == key.digest_hi &&
          net::GetU64(rec + 24) == key.digest_lo &&
          net::GetU64(rec + 36) == key.tenant_hash)
        return true;
    }
    return false;
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
  void TearDown() override {
    fs::remove_all(dir_);
    fs::remove_all(dir_.string() + "_tokens");
  }

  DiskSlabStore::Options Opts(uint64_t cap, uint64_t extent, uint64_t gran) {
    DiskSlabStore::Options o;
    o.dir = dir_.string();
    // Trust tokens in a directory of this test's own, not in /dev/shm.
    o.trust_token_dir = dir_.string() + "_tokens";
    fs::create_directories(o.trust_token_dir);
    o.capacity_bytes = cap;
    o.extent_bytes = extent;
    o.slot_granularity = gran;
    return o;
  }

  // Fork confines RLIMIT_NOFILE to the failed first startup; the parent
  // retries exactly the on-disk layout the child left behind.
  std::string InterruptedFirstStartup(uint64_t extents = 96) {
    int fds[2];
    if (::pipe(fds) != 0) return {};
    const auto options = Opts(extents * 4096, 4096, 4096);
    const pid_t child = ::fork();
    if (child == 0) {
      ::close(fds[0]);
      struct rlimit limit {};
      if (::getrlimit(RLIMIT_NOFILE, &limit) != 0) ::_exit(2);
      limit.rlim_cur = std::min<rlim_t>(limit.rlim_cur, 48);
      if (::setrlimit(RLIMIT_NOFILE, &limit) != 0) ::_exit(3);
      bool ok = true;
      DiskSlabStore store(options, &ok);
      const std::string error = store.StartupError();
      const ssize_t written = ::write(fds[1], error.data(), error.size());
      ::_exit(!ok && written == static_cast<ssize_t>(error.size()) ? 0 : 4);
    }
    ::close(fds[1]);
    std::string error;
    char buf[256];
    ssize_t n;
    while ((n = ::read(fds[0], buf, sizeof(buf))) > 0) error.append(buf, n);
    ::close(fds[0]);
    int status = 0;
    if (child < 0 || ::waitpid(child, &status, 0) != child ||
        !WIFEXITED(status) || WEXITSTATUS(status) != 0)
      return {};
    return error;
  }
  fs::path dir_;
};
BlockKey K(uint64_t id) { return BlockKey{id, 0}; }
struct HookGate {
  std::mutex mu;
  std::condition_variable cv;
  bool entered = false;
  bool open = false;
  bool result = true;
  bool Run() {
    std::unique_lock<std::mutex> lk(mu);
    entered = true;
    cv.notify_all();
    cv.wait(lk, [this] { return open; });
    return result;
  }
  void WaitEntered() {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [this] { return entered; });
  }
  void Release(bool value) {
    {
      std::lock_guard<std::mutex> lk(mu);
      result = value;
      open = true;
    }
    cv.notify_all();
  }
};
}  // namespace

// A background thread that the system refuses to start (thread limit) fails
// the store instead of terminating the process. Whichever of the two threads
// is refused, the open reports a failure, the object can be destroyed (which
// releases the directory), and the next open of the directory works.
TEST_F(DiskSlabTest, RefusedBackgroundThreadFailsTheStoreNotTheProcess) {
  for (const int which : {1, 2}) {
    fs::remove_all(dir_);
    {
      // Activated by the constructor, where a refused thread used to escape
      // as an exception.
      DiskSlabStore::Options o = Opts(1 << 20, 1 << 20, 4096);
      o.fail_thread_start_for_test = which;
      bool ok = true;
      DiskSlabStore s(o, &ok);
      EXPECT_FALSE(ok) << "thread " << which << " refused, yet the store serves";
      EXPECT_FALSE(s.Healthy());
      EXPECT_NE(s.StartupError().find("thread"), std::string::npos)
          << s.StartupError();
    }
    bool ok = false;
    DiskSlabStore again(Opts(1 << 20, 1 << 20, 4096), &ok);
    EXPECT_TRUE(ok) << "the directory is still locked or unusable (" << which
                    << ")";
    const std::string v(4096, 'r');
    EXPECT_EQ(again.Cache(K(static_cast<uint64_t>(which)), v.data(), v.size()),
              Status::kOk);
  }
}

TEST_F(DiskSlabTest, PutGetRemoveRoundTrip) {
  bool ok = false;
  DiskSlabStore s(Opts(1 << 20, 1 << 20, 4096), &ok);
  ASSERT_TRUE(ok);
  std::string v = "the-kv-payload-0123456789";
  ASSERT_EQ(s.Cache(K(1), v.data(), v.size()), Status::kOk);
  EXPECT_TRUE(s.IsCached(K(1)));
  EXPECT_EQ(s.Count(), 1u);
  dfkv::ValueMetadata metadata;
  ASSERT_EQ(s.Lookup(K(1), &metadata), Status::kOk);
  EXPECT_EQ(metadata.value_len, v.size());
  EXPECT_EQ(s.Lookup(K(999), &metadata), Status::kNotFound);

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

TEST_F(DiskSlabTest, TenantUsageSurvivesRestartAndRemove) {
  const BlockKey a{1, 0, 0x1111};
  const BlockKey b{2, 0, 0x1111};
  const BlockKey other{3, 0, 0x2222};
  {
    bool ok = false;
    DiskSlabStore store(Opts(1 << 20, 1 << 20, 4096), &ok);
    ASSERT_TRUE(ok);
    ASSERT_EQ(store.Cache(a, "aaaa", 4), Status::kOk);
    ASSERT_EQ(store.Cache(b, "bbbbbb", 6), Status::kOk);
    ASSERT_EQ(store.Cache(other, "ccc", 3), Status::kOk);
    EXPECT_EQ(store.TenantUsedBytes(0x1111), 10u);
    EXPECT_EQ(store.TenantUsedBytes(0x2222), 3u);
    ASSERT_EQ(store.Cache(a, "ignored", 7), Status::kOk);
    EXPECT_EQ(store.TenantUsedBytes(0x1111), 10u);
  }
  bool ok = false;
  DiskSlabStore reopened(Opts(1 << 20, 1 << 20, 4096), &ok);
  ASSERT_TRUE(ok) << reopened.StartupError();
  EXPECT_EQ(reopened.TenantUsedBytes(0x1111), 10u);
  EXPECT_EQ(reopened.TenantUsedBytes(0x2222), 3u);
  ASSERT_EQ(reopened.Remove(a), Status::kOk);
  EXPECT_EQ(reopened.TenantUsedBytes(0x1111), 6u);
}

TEST_F(DiskSlabTest, StatsTrackPhysicalAndLogicalOccupancy) {
  bool ok = false;
  DiskSlabStore store(Opts(1 << 20, 1 << 20, 4096), &ok);
  ASSERT_TRUE(ok);
  const uint64_t startup_class_count = store.GetStats().class_count;
  ASSERT_GT(startup_class_count, 1u);
  std::string value(3000, 's');
  ASSERT_EQ(store.Cache(K(1), value.data(), value.size()), Status::kOk);

  auto stats = store.GetStats();
  EXPECT_EQ(stats.capacity_bytes, 1u << 20);
  EXPECT_EQ(stats.allocated_bytes, 4096u);
  EXPECT_EQ(stats.payload_bytes, value.size());
  EXPECT_EQ(stats.allocator_objects, 1u);
  EXPECT_EQ(stats.committed_objects, 1u);
  EXPECT_EQ(stats.class_count, startup_class_count);
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

TEST_F(DiskSlabTest, FreshSparseTableAvoidsLogicalCapacityScanWhenSupported) {
  auto options = Opts(1ull << 30, 1ull << 30, 4096);
  options.table_sync_ms = 0;
  options.reclaim_interval_ms = 0;
  bool ok = false;
  DiskSlabStore store(options, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(store.Count(), 0u);
  EXPECT_EQ(store.TableRebuilt(), 0u);

  const fs::path table = dir_ / "slots.tbl";
  struct stat table_stat {};
  ASSERT_EQ(::stat(table.c_str(), &table_stat), 0);
  ASSERT_GT(table_stat.st_size, 1 << 20);
  const auto stats = store.GetStats();
  EXPECT_EQ(stats.rebuild_mmap_scans, 1u);
  EXPECT_LE(stats.rebuild_scanned_bytes,
            static_cast<uint64_t>(table_stat.st_size));

  int fd = ::open(table.c_str(), O_RDONLY);
  ASSERT_GE(fd, 0);
#if defined(SEEK_DATA)
  errno = 0;
  const off_t first_data = ::lseek(fd, 0, SEEK_DATA);
  const bool sparse_seek_reports_empty = first_data < 0 && errno == ENXIO;
  if (sparse_seek_reports_empty) {
    EXPECT_EQ(stats.rebuild_sequential_fallbacks, 0u);
    EXPECT_EQ(stats.rebuild_sparse_ranges, 0u);
    EXPECT_EQ(stats.rebuild_scan_chunks, 0u);
    EXPECT_EQ(stats.rebuild_scanned_bytes, 0u);
  } else if (stats.rebuild_sequential_fallbacks != 0) {
    EXPECT_EQ(stats.rebuild_scanned_bytes,
              static_cast<uint64_t>(table_stat.st_size));
    EXPECT_GT(stats.rebuild_scan_chunks, 1u);
  }
#else
  EXPECT_EQ(stats.rebuild_sequential_fallbacks, 1u);
  EXPECT_EQ(stats.rebuild_scanned_bytes,
            static_cast<uint64_t>(table_stat.st_size));
  EXPECT_GT(stats.rebuild_scan_chunks, 1u);
#endif
  ::close(fd);
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

TEST_F(DiskSlabTest, PopulatedRestartUsesChunkedRebuildScan) {
  auto options = Opts(1 << 20, 1 << 20, 4096);
  options.table_sync_ms = 0;
  options.reclaim_interval_ms = 0;
  const std::string first(3000, 'a');
  const std::string second(3500, 'b');
  {
    bool ok = false;
    DiskSlabStore store(options, &ok);
    ASSERT_TRUE(ok);
    ASSERT_EQ(store.Cache(K(501), first.data(), first.size()), Status::kOk);
    ASSERT_EQ(store.Cache(K(502), second.data(), second.size()), Status::kOk);
  }

  bool ok = false;
  DiskSlabStore reopened(options, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(reopened.TableRebuilt(), 2u);
  EXPECT_EQ(reopened.Count(), 2u);
  const auto stats = reopened.GetStats();
  EXPECT_GT(stats.rebuild_scan_chunks, 0u);
  EXPECT_GT(stats.rebuild_scanned_bytes, 0u);
  struct stat table_stat {};
  ASSERT_EQ(::stat((dir_ / "slots.tbl").c_str(), &table_stat), 0);
  EXPECT_LE(stats.rebuild_scanned_bytes,
            static_cast<uint64_t>(table_stat.st_size));

  std::string out;
  ASSERT_EQ(reopened.Range(K(501), 0, 0, &out), Status::kOk);
  EXPECT_EQ(out, first);
  ASSERT_EQ(reopened.Range(K(502), 0, 0, &out), Status::kOk);
  EXPECT_EQ(out, second);
}

TEST_F(DiskSlabTest, HistoricalRecordIsRejectedOnRestart) {
  const BlockKey key{4242, 9};
  const std::string value(3000, 'n');
  {
    bool ok = false;
    DiskSlabStore store(Opts(1 << 20, 1 << 20, 4096), &ok);
    ASSERT_TRUE(ok);
    ASSERT_EQ(store.Cache(key, value.data(), value.size()), Status::kOk);
  }

  const std::string table = (dir_ / "slots.tbl").string();
  int fd = ::open(table.c_str(), O_RDWR);
  ASSERT_GE(fd, 0);
  const off_t size = ::lseek(fd, 0, SEEK_END);
  bool replaced = false;
  for (off_t offset = 0; offset + 64 <= size; offset += 64) {
    uint32_t magic = 0;
    ASSERT_EQ(::pread(fd, &magic, sizeof(magic), offset),
              static_cast<ssize_t>(sizeof(magic)));
    if (magic == 0x334C5453u) {
      const uint32_t old_magic = 0x324C5453u;
      ASSERT_EQ(::pwrite(fd, &old_magic, sizeof(old_magic), offset),
                static_cast<ssize_t>(sizeof(old_magic)));
      replaced = true;
      break;
    }
  }
  ::close(fd);
  ASSERT_TRUE(replaced);

  bool ok = false;
  DiskSlabStore reopened(Opts(1 << 20, 1 << 20, 4096), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(reopened.TableRebuilt(), 0u);
  EXPECT_EQ(reopened.GetStats().rebuild_rejected_records, 1u);
  std::string out;
  EXPECT_EQ(reopened.Range(key, 0, 0, &out), Status::kNotFound);
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
  // Corrupt the BODY of a real (valid) v3 record so its CRC no longer
  // matches. Records land in high slots (LIFO alloc), so scan for the first
  // "STL3" magic, then flip a body byte (offset +12, inside the CRC'd region).
  const std::string tbl = (dir_ / "slots.tbl").string();
  int fd = ::open(tbl.c_str(), O_RDWR);
  ASSERT_GE(fd, 0);
  off_t fsz = ::lseek(fd, 0, SEEK_END);
  bool corrupted = false;
  for (off_t o = 0; o + 64 <= fsz; o += 64) {
    uint8_t m[4];
    if (::pread(fd, m, 4, o) != 4) break;
    // "STL3" little-endian = 53 54 4C 33.
    if (m[0] == 0x53 && m[1] == 0x54 && m[2] == 0x4C && m[3] == 0x33) {
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
  // The failed mismatched open must be read-only: the original geometry and
  // payload remain available when reopened with the matching contract.
  bool ok = false;
  DiskSlabStore reopened(Opts(1 << 20, 1 << 20, 4096), &ok);
  ASSERT_TRUE(ok);
  std::string out;
  ASSERT_EQ(reopened.Range(K(5), 0, 0, &out), Status::kOk);
  EXPECT_EQ(out, value);
}

TEST_F(DiskSlabTest, InterruptedFreshInitResumesAfterFdLimitRaised) {
  struct rlimit limit {};
  ASSERT_EQ(::getrlimit(RLIMIT_NOFILE, &limit), 0);
  if (limit.rlim_cur < 40) GTEST_SKIP() << "requires at least 40 descriptors";
  const std::string error = InterruptedFirstStartup();
  ASSERT_NE(error.find("extent"), std::string::npos) << error;
  ASSERT_NE(error.find("errno 24"), std::string::npos) << error;
  ASSERT_NE(error.find("LimitNOFILE"), std::string::npos) << error;
  ASSERT_TRUE(fs::exists(dir_ / "slab_initializing"));
  ASSERT_FALSE(fs::exists(dir_ / "slab_state"));
  struct stat before {};
  ASSERT_EQ(::stat((dir_ / "extents/E00000").c_str(), &before), 0);

  const auto options = Opts(96 * 4096, 4096, 4096);
  bool ok = false;
  {
    DiskSlabStore resumed(options, &ok);
    ASSERT_TRUE(ok) << resumed.StartupError();
    EXPECT_FALSE(fs::exists(dir_ / "slab_initializing"));
    struct stat after {};
    ASSERT_EQ(::stat((dir_ / "extents/E00000").c_str(), &after), 0);
    EXPECT_EQ(before.st_ino, after.st_ino);
    ASSERT_TRUE(fs::exists(dir_ / "extents/E00095"));
    ASSERT_EQ(resumed.Cache(K(600), "warm", 4), Status::kOk);
  }
  DiskSlabStore warm(options, &ok);
  ASSERT_TRUE(ok) << warm.StartupError();
  std::string out;
  ASSERT_EQ(warm.Range(K(600), 0, 0, &out), Status::kOk);
  EXPECT_EQ(out, "warm");
}

TEST_F(DiskSlabTest, InterruptedHighIndexExtentKeepsCanonicalName) {
  struct rlimit limit {};
  ASSERT_EQ(::getrlimit(RLIMIT_NOFILE, &limit), 0);
  if (limit.rlim_cur < 40) GTEST_SKIP() << "requires at least 40 descriptors";
  constexpr uint64_t kExtents = 100001;
  ASSERT_NE(InterruptedFirstStartup(kExtents).find("errno 24"),
            std::string::npos);
  const fs::path high_extent = dir_ / "extents/E100000";
  const int fd = ::open(high_extent.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(::ftruncate(fd, 4096), 0);
  ::close(fd);
  struct stat before {};
  ASSERT_EQ(::stat(high_extent.c_str(), &before), 0);

  // The limited child must get past the canonical 6-digit extent, then fail
  // on its own descriptor limit instead of rejecting a valid existing extent.
  const std::string error = InterruptedFirstStartup(kExtents);
  EXPECT_NE(error.find("errno 24"), std::string::npos) << error;
  struct stat after {};
  ASSERT_EQ(::stat(high_extent.c_str(), &after), 0);
  EXPECT_EQ(before.st_ino, after.st_ino);
  EXPECT_FALSE(fs::exists(dir_ / "slab_state"));
}

TEST_F(DiskSlabTest, InterruptedBeforeMetaCreationResumesOnlyWithMarker) {
  struct rlimit limit {};
  ASSERT_EQ(::getrlimit(RLIMIT_NOFILE, &limit), 0);
  if (limit.rlim_cur < 40) GTEST_SKIP() << "requires at least 40 descriptors";
  ASSERT_NE(InterruptedFirstStartup().find("errno 24"), std::string::npos);
  // Metadata is created after the zero table. An interruption at this earlier
  // stage can leave those files but no slab_meta.
  ASSERT_TRUE(fs::remove(dir_ / "slab_meta"));
  const auto options = Opts(96 * 4096, 4096, 4096);
  bool ok = false;
  DiskSlabStore resumed(options, &ok);
  ASSERT_TRUE(ok) << resumed.StartupError();
  EXPECT_TRUE(fs::exists(dir_ / "slab_meta"));
  EXPECT_FALSE(fs::exists(dir_ / "slab_initializing"));
}

TEST_F(DiskSlabTest, InterruptedInitWithDataOrTruncatedTableFailsClosed) {
  struct rlimit limit {};
  ASSERT_EQ(::getrlimit(RLIMIT_NOFILE, &limit), 0);
  if (limit.rlim_cur < 40) GTEST_SKIP() << "requires at least 40 descriptors";
  ASSERT_NE(InterruptedFirstStartup().find("errno 24"), std::string::npos);
  const auto options = Opts(96 * 4096, 4096, 4096);
  const fs::path extent = dir_ / "extents/E00000";
  int fd = ::open(extent.c_str(), O_RDWR);
  ASSERT_GE(fd, 0);
  const char payload = 'p';
  ASSERT_EQ(::pwrite(fd, &payload, 1, 123), 1);
  ::close(fd);
  const fs::path table = dir_ / "slots.tbl";
  fd = ::open(table.c_str(), O_RDWR);
  ASSERT_GE(fd, 0);
  const char record = 'R';
  ASSERT_EQ(::pwrite(fd, &record, 1, 0), 1);
  ::close(fd);
  bool ok = true;
  {
    DiskSlabStore refused(options, &ok);
    EXPECT_FALSE(ok);
    EXPECT_NE(refused.StartupError().find("contains data"), std::string::npos);
  }
  EXPECT_FALSE(fs::exists(dir_ / "slab_state"));
  fd = ::open(extent.c_str(), O_RDONLY);
  ASSERT_GE(fd, 0);
  char persisted = 0;
  ASSERT_EQ(::pread(fd, &persisted, 1, 123), 1);
  ::close(fd);
  EXPECT_EQ(persisted, payload);
  fd = ::open(table.c_str(), O_RDONLY);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(::pread(fd, &persisted, 1, 0), 1);
  ::close(fd);
  EXPECT_EQ(persisted, record);
  ASSERT_EQ(::truncate(table.c_str(), 1), 0);
  DiskSlabStore truncated(options, &ok);
  EXPECT_FALSE(ok);
  EXPECT_NE(truncated.StartupError().find("size/type"), std::string::npos);
  EXPECT_EQ(fs::file_size(table), 1u);
}

TEST_F(DiskSlabTest, UnmarkedOrUnexpectedPartialLayoutIsNeverRecreated) {
  struct rlimit limit {};
  ASSERT_EQ(::getrlimit(RLIMIT_NOFILE, &limit), 0);
  if (limit.rlim_cur < 40) GTEST_SKIP() << "requires at least 40 descriptors";
  ASSERT_NE(InterruptedFirstStartup().find("errno 24"), std::string::npos);
  const auto options = Opts(96 * 4096, 4096, 4096);
  const fs::path first_extent = dir_ / "extents/E00000";
  struct stat before {};
  ASSERT_EQ(::stat(first_extent.c_str(), &before), 0);
  {
    const fs::path unexpected = dir_ / "unknown.data";
    int fd = ::open(unexpected.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    ASSERT_GE(fd, 0);
    ASSERT_EQ(::write(fd, "keep", 4), 4);
    ::close(fd);
    bool ok = true;
    DiskSlabStore refused(options, &ok);
    EXPECT_FALSE(ok);
    EXPECT_NE(refused.StartupError().find("unexpected file"), std::string::npos);
    EXPECT_EQ(fs::file_size(unexpected), 4u);
    ASSERT_TRUE(fs::remove(unexpected));  // This test's own injected file.
  }
  ASSERT_TRUE(fs::remove(dir_ / "slab_initializing"));  // Simulate older partial.
  bool ok = true;
  DiskSlabStore refused(options, &ok);
  EXPECT_FALSE(ok);
  EXPECT_NE(refused.StartupError().find("errno 2"), std::string::npos)
      << refused.StartupError();
  EXPECT_FALSE(fs::exists(dir_ / "slab_state"));
  struct stat after {};
  ASSERT_EQ(::stat(first_extent.c_str(), &after), 0);
  EXPECT_EQ(before.st_ino, after.st_ino);
  EXPECT_EQ(fs::file_size(dir_ / "slots.tbl"), 96u * 64u);
}

TEST_F(DiskSlabTest, PopulatedStoreIgnoresStaleInitMarkerWithoutLosingData) {
  struct rlimit limit {};
  ASSERT_EQ(::getrlimit(RLIMIT_NOFILE, &limit), 0);
  if (limit.rlim_cur < 40) GTEST_SKIP() << "requires at least 40 descriptors";
  ASSERT_NE(InterruptedFirstStartup().find("errno 24"), std::string::npos);
  const fs::path marker = dir_ / "slab_initializing";
  std::string marker_bytes(64, '\0');
  int fd = ::open(marker.c_str(), O_RDONLY);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(::read(fd, marker_bytes.data(), marker_bytes.size()), 64);
  ::close(fd);
  const auto options = Opts(96 * 4096, 4096, 4096);
  bool ok = false;
  {
    DiskSlabStore store(options, &ok);
    ASSERT_TRUE(ok) << store.StartupError();
    ASSERT_EQ(store.Cache(K(700), "keep", 4), Status::kOk);
  }
  fd = ::open(marker.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(::write(fd, marker_bytes.data(), marker_bytes.size()), 64);
  ::close(fd);
  DiskSlabStore store(options, &ok);
  ASSERT_TRUE(ok) << store.StartupError();
  EXPECT_FALSE(fs::exists(marker));
  std::string out;
  ASSERT_EQ(store.Range(K(700), 0, 0, &out), Status::kOk);
  EXPECT_EQ(out, "keep");
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

TEST_F(DiskSlabTest, V1FormatIsRejectedWithoutRewritingMetadata) {
  {
    bool ok = false;
    DiskSlabStore store(Opts(1 << 20, 1 << 20, 4096), &ok);
    ASSERT_TRUE(ok);
  }
  const fs::path meta = dir_ / "slab_meta";
  int fd = ::open(meta.c_str(), O_RDWR);
  ASSERT_GE(fd, 0);
  const uint32_t old_version = 1;
  ASSERT_EQ(::pwrite(fd, &old_version, sizeof(old_version), 4),
            static_cast<ssize_t>(sizeof(old_version)));
  ASSERT_EQ(::fdatasync(fd), 0);
  ::close(fd);

  bool ok = true;
  DiskSlabStore store(Opts(1 << 20, 1 << 20, 4096), &ok);
  EXPECT_FALSE(ok);
  EXPECT_NE(store.StartupError().find("layout"), std::string::npos);
  fd = ::open(meta.c_str(), O_RDONLY);
  ASSERT_GE(fd, 0);
  uint32_t persisted_version = 0;
  ASSERT_EQ(::pread(fd, &persisted_version, sizeof(persisted_version), 4),
            static_cast<ssize_t>(sizeof(persisted_version)));
  ::close(fd);
  EXPECT_EQ(persisted_version, old_version);
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

// Close() is the destructor's work without the destructor: once it has
// returned, the process may die in any way and the next start is a clean one.
// (A server calls it before it releases a RAM tier that takes tens of seconds
// to unmap; a stop timeout that fires meanwhile must not cost the cache.)
TEST_F(DiskSlabTest, CloseMakesTheStoreCleanBeforeItIsDestroyed) {
  auto options = Opts(1 << 20, 1 << 20, 4096);
  const std::string value(4096, 'k');
  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    bool ok = false;
    DiskSlabStore store(options, &ok);
    const bool closed =
        ok &&
        store.Cache(K(72), value.data(), value.size()) == Status::kOk &&
        store.Close() &&
        store.Close();  // idempotent, same answer
    ::_exit(closed ? 0 : 1);  // no destructor: the process is killed here
  }
  int child_status = 0;
  ASSERT_EQ(::waitpid(child, &child_status, 0), child);
  ASSERT_TRUE(WIFEXITED(child_status));
  ASSERT_EQ(WEXITSTATUS(child_status), 0);

  bool ok = false;
  DiskSlabStore reopened(options, &ok);  // warm restart is off by default
  ASSERT_TRUE(ok);
  EXPECT_EQ(reopened.GetStats().unclean_resets, 0u);
  EXPECT_EQ(reopened.Count(), 1u);
  std::string out;
  ASSERT_EQ(reopened.Range(K(72), 0, value.size(), &out), Status::kOk);
  EXPECT_EQ(out, value);
}

TEST_F(DiskSlabTest, DestructionAfterCloseLeavesTheStoreClean) {
  auto options = Opts(1 << 20, 1 << 20, 4096);
  const std::string value(4096, 'm');
  {
    bool ok = false;
    DiskSlabStore store(options, &ok);
    ASSERT_TRUE(ok);
    ASSERT_EQ(store.Cache(K(73), value.data(), value.size()), Status::kOk);
    EXPECT_TRUE(store.Close());
  }
  bool ok = false;
  DiskSlabStore reopened(options, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(reopened.GetStats().unclean_resets, 0u);
  std::string out;
  ASSERT_EQ(reopened.Range(K(73), 0, value.size(), &out), Status::kOk);
  EXPECT_EQ(out, value);
}

// ---- Unclean exit within the same boot: the table may be kept -------------
//
// The crashing process is a forked child that leaves through _exit, i.e.
// without destructors: what a SIGKILLed server leaves behind. Boot and mount
// identity are injected, so a "reboot" or a "remount" is just another value.
namespace {
using WarmRestart = DiskSlabStore::Options::WarmRestart;

DiskSlabStore::Options WithIdentity(DiskSlabStore::Options options,
                                    WarmRestart policy, uint8_t boot,
                                    uint64_t mount) {
  options.table_sync_ms = 0;
  options.reclaim_interval_ms = 0;
  options.warm_restart = policy;
  std::array<uint8_t, 16> boot_id;
  boot_id.fill(boot);
  options.boot_id = boot_id;
  options.mount_id = mount;
  // No opinion about the filesystem the build machine keeps /tmp on. (Opts
  // already put the trust tokens into this test's own directory.)
  options.require_local_filesystem = false;
  return options;
}

// The token a store holds for `dir` while it vouches for its table.
fs::path TrustToken(const fs::path& dir) {
  struct stat st {};
  if (::stat(dir.c_str(), &st) != 0) return {};
  char name[80];
  std::snprintf(name, sizeof(name), "dfkv-slab-trusted.%llx.%llx",
                static_cast<unsigned long long>(st.st_dev),
                static_cast<unsigned long long>(st.st_ino));
  return fs::path(dir.string() + "_tokens") / name;
}

template <typename Body>
::testing::AssertionResult CrashAfter(const DiskSlabStore::Options& options,
                                      Body&& body) {
  const pid_t child = ::fork();
  if (child < 0) return ::testing::AssertionFailure() << "fork failed";
  if (child == 0) {
    bool ok = false;
    DiskSlabStore store(options, &ok);
    ::_exit(ok && body(store) ? 0 : 1);
  }
  int status = 0;
  if (::waitpid(child, &status, 0) != child || !WIFEXITED(status) ||
      WEXITSTATUS(status) != 0)
    return ::testing::AssertionFailure()
           << "the crashing child failed its own steps";
  return ::testing::AssertionSuccess();
}

std::string Payload(char fill) { return std::string(4096, fill); }

bool PutFilled(DiskSlabStore& store, uint64_t id, char fill) {
  const std::string value = Payload(fill);
  return store.Cache(K(id), value.data(), value.size()) == Status::kOk;
}

// Offsets of the valid records in slots.tbl.
std::vector<off_t> ValidRecordOffsets(const fs::path& dir) {
  std::ifstream in(dir / "slots.tbl", std::ios::binary);
  const std::string table((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
  std::vector<off_t> offsets;
  for (size_t at = 0; at + 64 <= table.size(); at += 64)
    if (dfkv::net::GetU32(table.data() + at) != 0 && table[at + 8] == 1)
      offsets.push_back(static_cast<off_t>(at));
  return offsets;
}
}  // namespace

// ---- STO-15: payload checksums ------------------------------------------------
namespace {
// Flips one byte of `value`'s first copy in the extent files (or overwrites
// all of it with zeros), as damage on the disk or a lost write leaves it.
bool DamagePayload(const fs::path& dir, const std::string& value, bool zero_all) {
  for (const auto& e : fs::directory_iterator(dir / "extents")) {
    std::ifstream in(e.path(), std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
    const size_t at = bytes.find(value);
    if (at == std::string::npos) continue;
    const int fd = ::open(e.path().c_str(), O_WRONLY);
    if (fd < 0) return false;
    bool ok;
    if (zero_all) {
      const std::string zeros(value.size(), '\0');
      ok = ::pwrite(fd, zeros.data(), zeros.size(), static_cast<off_t>(at)) ==
           static_cast<ssize_t>(zeros.size());
    } else {
      const char flipped = static_cast<char>(bytes[at + 100] ^ 0x01);
      ok = ::pwrite(fd, &flipped, 1, static_cast<off_t>(at + 100)) == 1;
    }
    ::fsync(fd);
    ::close(fd);
    return ok;
  }
  return false;
}
std::string Patterned(size_t len, int seed) {
  std::string v(len, '\0');
  uint32_t x = 2463534242u + seed;
  for (auto& c : v) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    c = static_cast<char>(x);
  }
  return v;
}
}  // namespace

// A value whose bytes on the disk no longer match the CRC its PUT recorded
// is a miss on every whole-value read path, counted, and removed (record
// invalidated, so a restart does not bring it back); a new PUT of the key is
// stored. A partial read cannot be checked and is served as read.
TEST_F(DiskSlabTest, ADamagedPayloadIsAMissOnEveryWholeValueRead) {
  enum Path { kRange, kRangeInto, kRangeDirect, kLease };
  for (const Path path : {kRange, kRangeInto, kRangeDirect, kLease}) {
    SCOPED_TRACE(static_cast<int>(path));
    fs::remove_all(dir_);
    auto options = Opts(1 << 20, 1 << 20, 4096);
    options.direct_writes = true;
    options.reclaim_interval_ms = 0;
    const std::string v = Patterned(5000, path);
    {
      bool ok = false;
      DiskSlabStore s(options, &ok);
      ASSERT_TRUE(ok) << s.StartupError();
      ASSERT_EQ(s.Cache(K(7), v.data(), v.size()), Status::kOk);
      std::string out;
      ASSERT_EQ(s.Range(K(7), 0, 0, &out), Status::kOk);  // intact: served
      ASSERT_EQ(out, v);
      ASSERT_TRUE(DamagePayload(dir_, v, /*zero_all=*/false));
      // A slice is not checked.
      ASSERT_EQ(s.Range(K(7), 200, 100, &out), Status::kOk);
      void* mem = nullptr;
      ASSERT_EQ(::posix_memalign(&mem, 4096, 64 << 10), 0);
      char* buf = static_cast<char*>(mem);
      Status st = Status::kOk;
      switch (path) {
        case kRange: st = s.Range(K(7), 0, 0, &out); break;
        case kRangeInto: {
          size_t got = 0;
          st = s.RangeInto(K(7), 0, 0, buf, 64 << 10, &got);
          break;
        }
        case kRangeDirect: {
          const char* data = nullptr;
          size_t got = 0;
          st = s.RangeDirect(K(7), 0, 0, buf, 64 << 10, &data, &got);
          break;
        }
        case kLease: {
          dfkv::ReadLease lease;
          ASSERT_EQ(s.RangeDirectPrep(K(7), 0, 0, 64 << 10, &lease), Status::kOk);
          ASSERT_EQ(::pread(lease.fd(), buf, lease.aligned_len,
                            static_cast<off_t>(lease.aligned_off)),
                    static_cast<ssize_t>(lease.aligned_len));
          st = lease.Verify(buf + lease.head) ? Status::kOk : Status::kNotFound;
          break;
        }
      }
      std::free(mem);
      EXPECT_EQ(st, Status::kNotFound);
      EXPECT_EQ(s.GetStats().payload_crc_failures, 1u);
      EXPECT_FALSE(s.IsCached(K(7)));
      EXPECT_EQ(s.Range(K(7), 0, 0, &out), Status::kNotFound);
      const std::string w = Patterned(5000, 100 + path);
      ASSERT_EQ(s.Cache(K(7), w.data(), w.size()), Status::kOk) << "stored again";
      ASSERT_EQ(s.Range(K(7), 0, 0, &out), Status::kOk);
      EXPECT_EQ(out, w);
      ASSERT_EQ(s.Remove(K(7)), Status::kOk);
    }
    // The removal is on the disk: a clean restart does not bring K(7) back.
    bool ok = false;
    DiskSlabStore reopened(options, &ok);
    ASSERT_TRUE(ok);
    EXPECT_FALSE(reopened.IsCached(K(7)));
  }
}

// A record written by an older version has no checksum: its value is served
// unchecked (and survives a restart), so an upgrade keeps the cache.
TEST_F(DiskSlabTest, ARecordWithoutAChecksumIsServedUnchecked) {
  auto options = Opts(1 << 20, 1 << 20, 4096);
  options.reclaim_interval_ms = 0;
  const std::string v = Patterned(5000, 9);
  {
    bool ok = false;
    DiskSlabStore s(options, &ok);
    ASSERT_TRUE(ok);
    ASSERT_EQ(s.Cache(K(8), v.data(), v.size()), Status::kOk);
  }
  // Rewrite the record as an older version writes it: no flag, no CRC.
  {
    const auto offsets = ValidRecordOffsets(dir_);
    ASSERT_EQ(offsets.size(), 1u);
    const int fd = ::open((dir_ / "slots.tbl").c_str(), O_RDWR);
    ASSERT_GE(fd, 0);
    uint8_t rec[64];
    ASSERT_EQ(::pread(fd, rec, 64, offsets[0]), 64);
    ASSERT_EQ(rec[9], 1u) << "this version writes the flag";
    rec[9] = 0;
    std::memset(rec + 52, 0, 4);
    dfkv::net::PutU32(reinterpret_cast<char*>(rec) + 4, dfkv::Crc32(rec + 8, 56));
    ASSERT_EQ(::pwrite(fd, rec, 64, offsets[0]), 64);
    ::close(fd);
  }
  ASSERT_TRUE(DamagePayload(dir_, v, /*zero_all=*/false));
  bool ok = false;
  DiskSlabStore s(options, &ok);
  ASSERT_TRUE(ok);
  std::string out;
  ASSERT_EQ(s.Range(K(8), 0, 0, &out), Status::kOk) << "unchecked";
  EXPECT_EQ(s.GetStats().payload_crc_failures, 0u);
}

// A damaged read decides to drop the key only after it has let go of the
// slot. A REMOVE and a new PUT of the key in between can put a good value in
// the same slot: that value is kept, and nothing is counted. The read itself
// answers a miss (it read the damaged bytes).
TEST_F(DiskSlabTest, AGoodValueWrittenIntoTheSlotDuringADamagedReadIsKept) {
  enum Path { kRange, kRangeInto, kRangeDirect };
  for (const Path path : {kRange, kRangeInto, kRangeDirect}) {
    SCOPED_TRACE(static_cast<int>(path));
    fs::remove_all(dir_);
    auto options = Opts(1 << 20, 1 << 20, 4096);
    options.direct_writes = true;
    options.reclaim_interval_ms = 0;
    bool ok = false;
    DiskSlabStore s(options, &ok);
    ASSERT_TRUE(ok) << s.StartupError();
    const std::string v = Patterned(5000, 20 + path);
    const std::string w = Patterned(5000, 40 + path);
    ASSERT_EQ(s.Cache(K(7), v.data(), v.size()), Status::kOk);
    const auto before = dfkv::DiskSlabStoreTestPeer::SlotOf(&s, K(7));
    ASSERT_TRUE(DamagePayload(dir_, v, /*zero_all=*/false));
    bool replaced = false;
    dfkv::SlabAllocator::SlotRef after;
    dfkv::DiskSlabStoreTestPeer::SetDropCorruptHook(&s, [&] {
      if (replaced) return;
      replaced = true;
      ASSERT_EQ(s.Remove(K(7)), Status::kOk);
      ASSERT_EQ(s.Cache(K(7), w.data(), w.size()), Status::kOk);
      after = dfkv::DiskSlabStoreTestPeer::SlotOf(&s, K(7));
    });
    void* mem = nullptr;
    ASSERT_EQ(::posix_memalign(&mem, 4096, 64 << 10), 0);
    char* buf = static_cast<char*>(mem);
    std::string out;
    Status st = Status::kOk;
    switch (path) {
      case kRange: st = s.Range(K(7), 0, 0, &out); break;
      case kRangeInto: {
        size_t got = 0;
        st = s.RangeInto(K(7), 0, 0, buf, 64 << 10, &got);
        break;
      }
      case kRangeDirect: {
        const char* data = nullptr;
        size_t got = 0;
        st = s.RangeDirect(K(7), 0, 0, buf, 64 << 10, &data, &got);
        break;
      }
    }
    std::free(mem);
    dfkv::DiskSlabStoreTestPeer::SetDropCorruptHook(&s, {});
    ASSERT_TRUE(replaced);
    ASSERT_EQ(after.extent, before.extent);
    ASSERT_EQ(after.slot, before.slot) << "the new value took the same slot";
    EXPECT_EQ(st, Status::kNotFound);
    EXPECT_EQ(s.GetStats().payload_crc_failures, 0u);
    ASSERT_EQ(s.Range(K(7), 0, 0, &out), Status::kOk) << "the good value";
    EXPECT_EQ(out, w);
  }
}

// DFKV_SLAB_PAYLOAD_CRC=0: a PUT records no checksum (the record is written
// as an older version writes it) and no read checks one, also on a record
// written while it was on. Records written while it is off stay unchecked
// after it is switched on again.
TEST_F(DiskSlabTest, WithTheChecksumOffNothingIsRecordedOrChecked) {
  auto on = Opts(1 << 20, 1 << 20, 4096);
  on.reclaim_interval_ms = 0;
  auto off = on;
  off.payload_crc = false;
  const std::string a = Patterned(5000, 61);  // written while on
  const std::string b = Patterned(5000, 62);  // written while off
  {
    bool ok = false;
    DiskSlabStore s(on, &ok);
    ASSERT_TRUE(ok);
    ASSERT_EQ(s.Cache(K(1), a.data(), a.size()), Status::kOk);
  }
  {
    bool ok = false;
    DiskSlabStore s(off, &ok);
    ASSERT_TRUE(ok);
    ASSERT_EQ(s.Cache(K(2), b.data(), b.size()), Status::kOk);
  }
  // Only the record written while on carries the flag.
  {
    const int fd = ::open((dir_ / "slots.tbl").c_str(), O_RDONLY);
    ASSERT_GE(fd, 0);
    size_t flagged = 0, unflagged = 0;
    for (const auto offset : ValidRecordOffsets(dir_)) {
      uint8_t rec[64];
      ASSERT_EQ(::pread(fd, rec, 64, offset), 64);
      (rec[9] & 1 ? flagged : unflagged)++;
    }
    ::close(fd);
    EXPECT_EQ(flagged, 1u);
    EXPECT_EQ(unflagged, 1u);
  }
  ASSERT_TRUE(DamagePayload(dir_, a, /*zero_all=*/false));
  ASSERT_TRUE(DamagePayload(dir_, b, /*zero_all=*/false));
  std::string out;
  {
    bool ok = false;
    DiskSlabStore s(off, &ok);
    ASSERT_TRUE(ok);
    EXPECT_EQ(s.Range(K(1), 0, 0, &out), Status::kOk) << "not checked when off";
    EXPECT_EQ(s.Range(K(2), 0, 0, &out), Status::kOk);
    dfkv::ReadLease lease;
    ASSERT_EQ(s.RangeDirectPrep(K(1), 0, 0, 64 << 10, &lease), Status::kOk);
    void* mem = nullptr;
    ASSERT_EQ(::posix_memalign(&mem, 4096, 64 << 10), 0);
    ASSERT_EQ(::pread(lease.fd(), mem, lease.aligned_len,
                      static_cast<off_t>(lease.aligned_off)),
              static_cast<ssize_t>(lease.aligned_len));
    EXPECT_TRUE(lease.Verify(static_cast<char*>(mem) + lease.head));
    std::free(mem);
    EXPECT_EQ(s.GetStats().payload_crc_failures, 0u);
  }
  bool ok = false;
  DiskSlabStore s(on, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(s.Range(K(2), 0, 0, &out), Status::kOk)
      << "written while off: unchecked for good";
  EXPECT_EQ(s.Range(K(1), 0, 0, &out), Status::kNotFound) << "checked again";
  EXPECT_EQ(s.GetStats().payload_crc_failures, 1u);
}

// A kill after the record of a PUT was written but before its payload
// reached the slot (the slot still holds other bytes): a warm restart keeps
// the table, and the value is dropped at its first read, not served.
TEST_F(DiskSlabTest, AWarmRestartDropsARecordWhosePayloadNeverLanded) {
  const auto options =
      WithIdentity(Opts(1 << 20, 1 << 20, 4096), WarmRestart::kAuto, 1, 7);
  const std::string v = Patterned(5000, 11);
  ASSERT_TRUE(CrashAfter(options, [&](DiskSlabStore& store) {
    if (store.Cache(K(9), v.data(), v.size()) != Status::kOk) return false;
    // The payload write lost: the slot holds zeros.
    if (!DamagePayload(dir_, v, /*zero_all=*/true)) return false;
    ::_exit(0);
  }));
  bool ok = false;
  DiskSlabStore recovered(options, &ok);
  ASSERT_TRUE(ok) << recovered.StartupError();
  EXPECT_EQ(recovered.GetStats().warm_restarts, 1u);
  EXPECT_TRUE(recovered.IsCached(K(9))) << "the record was kept";
  std::string out;
  EXPECT_EQ(recovered.Range(K(9), 0, 0, &out), Status::kNotFound);
  EXPECT_EQ(recovered.GetStats().payload_crc_failures, 1u);
  EXPECT_FALSE(recovered.IsCached(K(9)));
}

TEST_F(DiskSlabTest, UncleanExitInTheSameBootKeepsTheCache) {
  const auto options =
      WithIdentity(Opts(1 << 20, 1 << 20, 4096), WarmRestart::kAuto, 1, 7);
  ASSERT_TRUE(CrashAfter(options, [](DiskSlabStore& store) {
    return PutFilled(store, 90, 'a') && PutFilled(store, 91, 'b') &&
           PutFilled(store, 92, 'c') && store.Remove(K(91)) == Status::kOk;
  }));

  {
    bool ok = false;
    DiskSlabStore recovered(options, &ok);
    ASSERT_TRUE(ok) << recovered.StartupError();
    EXPECT_EQ(recovered.GetStats().warm_restarts, 1u);
    EXPECT_EQ(recovered.GetStats().unclean_resets, 0u);
    EXPECT_EQ(recovered.Count(), 2u);
    std::string out;
    ASSERT_EQ(recovered.Range(K(90), 0, 4096, &out), Status::kOk);
    EXPECT_EQ(out, Payload('a'));
    ASSERT_EQ(recovered.Range(K(92), 0, 4096, &out), Status::kOk);
    EXPECT_EQ(out, Payload('c'));
    EXPECT_EQ(recovered.Range(K(91), 0, 4096, &out), Status::kNotFound)
        << "a removed key came back";
    ASSERT_TRUE(PutFilled(recovered, 91, 'd'));  // and its slot is reusable
  }

  bool ok = false;
  DiskSlabStore clean_reopen(options, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(clean_reopen.GetStats().warm_restarts, 0u);
  EXPECT_EQ(clean_reopen.Count(), 3u);
  std::string out;
  ASSERT_EQ(clean_reopen.Range(K(91), 0, 4096, &out), Status::kOk);
  EXPECT_EQ(out, Payload('d'));
}

// Every condition short of "same boot, same mount, policy allows it" resets.
TEST_F(DiskSlabTest, UncleanExitIsResetWhenTheTableCannotBeTrusted) {
  struct Case {
    const char* name;
    uint64_t crashed_mount;  // the crashed run: kAuto, boot 1, this mount id
    WarmRestart policy;
    uint8_t boot;
    uint64_t mount;
    bool kept;
  };
  const Case cases[] = {
      {"same boot and mount", 7, WarmRestart::kAuto, 1, 7, true},
      {"rebooted", 7, WarmRestart::kAuto, 2, 7, false},
      {"mounted again", 7, WarmRestart::kAuto, 1, 8, false},
      {"mounted again, boot-only policy", 7, WarmRestart::kBootOnly, 1, 8, false},
      {"mount id lost", 7, WarmRestart::kAuto, 1, 0, false},
      {"mount id lost, boot-only policy", 7, WarmRestart::kBootOnly, 1, 0, false},
      {"never had a mount id", 0, WarmRestart::kAuto, 1, 0, false},
      {"never had a mount id, boot-only policy", 0, WarmRestart::kBootOnly, 1, 0,
       true},
      {"no boot id now", 7, WarmRestart::kBootOnly, 0, 7, false},
      {"policy off", 7, WarmRestart::kOff, 1, 7, false},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    fs::remove_all(dir_);
    const auto base = Opts(1 << 20, 1 << 20, 4096);
    ASSERT_TRUE(
        CrashAfter(WithIdentity(base, WarmRestart::kAuto, 1, c.crashed_mount),
                   [](DiskSlabStore& store) {
                     return PutFilled(store, 90, 'a');
                   }));
    bool ok = false;
    DiskSlabStore recovered(WithIdentity(base, c.policy, c.boot, c.mount), &ok);
    ASSERT_TRUE(ok) << recovered.StartupError();
    EXPECT_EQ(recovered.GetStats().warm_restarts, c.kept ? 1u : 0u);
    EXPECT_EQ(recovered.GetStats().unclean_resets, c.kept ? 0u : 1u);
    EXPECT_EQ(recovered.Count(), c.kept ? 1u : 0u);
  }
}

// The policy is applied by the run that finds the dirty state: a run with the
// policy off still records its identity, so the next start may keep its table.
TEST_F(DiskSlabTest, IdentityIsRecordedWhateverThePolicyOfTheCrashedRun) {
  const auto base = Opts(1 << 20, 1 << 20, 4096);
  ASSERT_TRUE(CrashAfter(WithIdentity(base, WarmRestart::kOff, 1, 7),
                         [](DiskSlabStore& store) {
                           return PutFilled(store, 90, 'a');
                         }));
  bool ok = false;
  DiskSlabStore recovered(WithIdentity(base, WarmRestart::kAuto, 1, 7), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(recovered.GetStats().warm_restarts, 1u);
  EXPECT_EQ(recovered.Count(), 1u);
}

// A Remove acknowledged while a reader still pinned the slot must hold across
// the crash: its record was invalidated before the acknowledgement.
TEST_F(DiskSlabTest, RemoveAcknowledgedBeforeTheCrashStaysRemoved) {
  const auto options =
      WithIdentity(Opts(1 << 20, 1 << 20, 4096), WarmRestart::kAuto, 1, 7);
  ASSERT_TRUE(CrashAfter(options, [](DiskSlabStore& store) {
    if (!PutFilled(store, 90, 'a') || !PutFilled(store, 91, 'b')) return false;
    static dfkv::ReadLease reader;  // never released: the process dies first
    return store.RangeDirectPrep(K(90), 0, 0, 1 << 20, &reader) ==
               Status::kOk &&
           store.Remove(K(90)) == Status::kOk;
  }));
  bool ok = false;
  DiskSlabStore recovered(options, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(recovered.GetStats().warm_restarts, 1u);
  std::string out;
  EXPECT_EQ(recovered.Range(K(90), 0, 4096, &out), Status::kNotFound);
  EXPECT_EQ(recovered.Range(K(91), 0, 4096, &out), Status::kOk);
}

// A run that hit a metadata failure can no longer vouch for what the page
// cache holds, and the kernel will not repeat the error to the next process.
// It cannot rely on the failing disk to remember that either: here slab_state
// is unwritable too, so only the token being gone carries it.
TEST_F(DiskSlabTest, StorageFailureInThePreviousRunForcesAReset) {
  const auto options =
      WithIdentity(Opts(1 << 20, 1 << 20, 4096), WarmRestart::kAuto, 1, 7);
  ASSERT_TRUE(CrashAfter(options, [](DiskSlabStore& store) {
    if (!PutFilled(store, 90, 'a')) return false;
    dfkv::DiskSlabStoreTestPeer::BreakStateFd(&store);
    dfkv::DiskSlabStoreTestPeer::BreakTableFd(&store);
    return !dfkv::DiskSlabStoreTestPeer::SyncTableNow(&store) &&
           !store.Healthy();
  }));
  bool ok = false;
  DiskSlabStore recovered(options, &ok);
  ASSERT_TRUE(ok) << recovered.StartupError();
  EXPECT_EQ(recovered.GetStats().warm_restarts, 0u);
  EXPECT_EQ(recovered.GetStats().unclean_resets, 1u);
  EXPECT_EQ(recovered.Count(), 0u);
}

// What the server's fatal-signal handler does before the process dies.
TEST_F(DiskSlabTest, CrashOfTheProcessItselfForcesAReset) {
  const auto options =
      WithIdentity(Opts(1 << 20, 1 << 20, 4096), WarmRestart::kAuto, 1, 7);
  ASSERT_TRUE(CrashAfter(options, [](DiskSlabStore& store) {
    if (!PutFilled(store, 90, 'a')) return false;
    DiskSlabStore::RevokeAllTrustForCrash();
    return true;
  }));
  bool ok = false;
  DiskSlabStore recovered(options, &ok);
  ASSERT_TRUE(ok) << recovered.StartupError();
  EXPECT_EQ(recovered.GetStats().warm_restarts, 0u);
  EXPECT_EQ(recovered.GetStats().unclean_resets, 1u);
  EXPECT_EQ(recovered.Count(), 0u);
}

// A reset that dies half-way leaves a partly zeroed table next to a state that
// still looked eligible. The identity is withdrawn before the first zero is
// written, so the next start finishes the reset instead of keeping the rest.
TEST_F(DiskSlabTest, InterruptedResetIsFinishedNotKept) {
  const auto base = Opts(1 << 20, 1 << 20, 4096);
  const auto options = WithIdentity(base, WarmRestart::kAuto, 1, 7);
  ASSERT_TRUE(CrashAfter(options, [](DiskSlabStore& store) {
    return PutFilled(store, 90, 'a') && PutFilled(store, 91, 'b') &&
           PutFilled(store, 92, 'c');
  }));
  ASSERT_EQ(ValidRecordOffsets(dir_).size(), 3u);

  // "Rebooted" start whose reset cannot get past the first bytes of the table.
  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    ::signal(SIGXFSZ, SIG_IGN);
    struct rlimit limit {};
    limit.rlim_cur = limit.rlim_max = 64;  // slab_state fits, the table does not
    if (::setrlimit(RLIMIT_FSIZE, &limit) != 0) ::_exit(2);
    bool ok = true;
    DiskSlabStore store(WithIdentity(base, WarmRestart::kAuto, 2, 7), &ok);
    ::_exit(ok ? 3 : 0);  // the reset must have failed
  }
  int status = 0;
  ASSERT_EQ(::waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
  ASSERT_FALSE(ValidRecordOffsets(dir_).empty())
      << "the staged reset was meant to stop early";

  bool ok = false;
  DiskSlabStore recovered(options, &ok);  // the original boot and mount again
  ASSERT_TRUE(ok) << recovered.StartupError();
  EXPECT_EQ(recovered.GetStats().warm_restarts, 0u);
  EXPECT_EQ(recovered.GetStats().unclean_resets, 1u);
  EXPECT_EQ(recovered.Count(), 0u);
}

// Whether a table may be kept must not depend on the disk that failed, and the
// kernel reports a writeback error only once. So a run holds a token on a
// tmpfs exactly while it vouches for its table, and keeping requires the token
// of the run that wrote the dirty state.
TEST_F(DiskSlabTest, RunHoldsItsTrustTokenWhileItVouchesForTheTable) {
  const auto options =
      WithIdentity(Opts(1 << 20, 1 << 20, 4096), WarmRestart::kAuto, 1, 7);
  fs::path token;
  {
    bool ok = false;
    DiskSlabStore store(options, &ok);
    ASSERT_TRUE(ok);
    token = TrustToken(dir_);
    ASSERT_FALSE(token.empty());
    EXPECT_TRUE(fs::exists(token));
    EXPECT_EQ(store.GetStats().trust_tokens_held, 1u);
    ASSERT_TRUE(PutFilled(store, 90, 'a'));
    bool present_inside_sync = true;
    dfkv::DiskSlabStoreTestPeer::SetTableSyncHook(
        &store, [&] { present_inside_sync = fs::exists(token); });
    ASSERT_TRUE(dfkv::DiskSlabStoreTestPeer::SyncTableNow(&store));
    dfkv::DiskSlabStoreTestPeer::SetTableSyncHook(&store, {});
    EXPECT_FALSE(present_inside_sync) << "a sync could consume an error unseen";
    EXPECT_TRUE(fs::exists(token)) << "a clean sync gives the token back";
  }
  EXPECT_FALSE(fs::exists(token)) << "a clean shutdown needs no token";
}

// While any table sync is in flight the token stays away: the one that
// finishes first must not hand it back under the other.
TEST_F(DiskSlabTest, OverlappingTableSyncsReturnTheTokenOnlyAfterTheLast) {
  const auto options =
      WithIdentity(Opts(1 << 20, 1 << 20, 4096), WarmRestart::kAuto, 1, 7);
  bool ok = false;
  DiskSlabStore store(options, &ok);
  ASSERT_TRUE(ok);
  const fs::path token = TrustToken(dir_);
  ASSERT_TRUE(PutFilled(store, 90, 'a'));

  HookGate first_sync;
  std::atomic<int> syncs{0};
  dfkv::DiskSlabStoreTestPeer::SetTableSyncHook(&store, [&] {
    if (syncs.fetch_add(1) == 0) first_sync.Run();  // only the first one waits
  });
  std::thread slow([&] { dfkv::DiskSlabStoreTestPeer::SyncTableNow(&store); });
  first_sync.WaitEntered();
  ASSERT_TRUE(dfkv::DiskSlabStoreTestPeer::SyncTableNow(&store));
  EXPECT_FALSE(fs::exists(token))
      << "a finished sync returned the token while another was in flight";
  first_sync.Release(true);
  slow.join();
  dfkv::DiskSlabStoreTestPeer::SetTableSyncHook(&store, {});
  EXPECT_TRUE(fs::exists(token));
}

// Something undeletable at the token path (here: a directory) must not cost
// more than the ability to keep the cache. In particular a clean shutdown
// stays a clean shutdown.
TEST_F(DiskSlabTest, UnremovableThingAtTheTokenPathLeavesCleanShutdownsClean) {
  const auto options =
      WithIdentity(Opts(1 << 20, 1 << 20, 4096), WarmRestart::kAuto, 1, 7);
  fs::create_directories(dir_);
  const fs::path token = TrustToken(dir_);
  ASSERT_FALSE(token.empty());
  fs::create_directory(token);
  {
    bool ok = false;
    DiskSlabStore store(options, &ok);
    ASSERT_TRUE(ok) << store.StartupError();
    EXPECT_EQ(store.GetStats().trust_tokens_held, 0u);
    ASSERT_TRUE(PutFilled(store, 90, 'a'));
    ASSERT_TRUE(dfkv::DiskSlabStoreTestPeer::SyncTableNow(&store));
  }
  {
    bool ok = false;
    DiskSlabStore reopened(options, &ok);
    ASSERT_TRUE(ok) << reopened.StartupError();
    EXPECT_EQ(reopened.GetStats().unclean_resets, 0u)
        << "the clean state was overwritten on the way out";
    EXPECT_EQ(reopened.Count(), 1u);
  }
  // And an unclean exit of such a run resets: nothing vouches for it.
  ASSERT_TRUE(CrashAfter(options, [](DiskSlabStore& store) {
    return store.Count() == 1 && PutFilled(store, 91, 'b');
  }));
  {
    bool ok = false;
    DiskSlabStore recovered(options, &ok);
    ASSERT_TRUE(ok) << recovered.StartupError();
    EXPECT_EQ(recovered.GetStats().warm_restarts, 0u);
    EXPECT_EQ(recovered.GetStats().unclean_resets, 1u);
  }
  fs::remove(token);
}

TEST_F(DiskSlabTest, MissingOrForeignTrustTokenForcesAReset) {
  // The name and the epoch of a token can be worked out by anyone who can read
  // the cache directory, so only a file that nobody else could have written
  // counts.
  enum class Tamper { kRemoved, kOtherEpoch, kWritableByOthers };
  for (const Tamper tamper :
       {Tamper::kRemoved, Tamper::kOtherEpoch, Tamper::kWritableByOthers}) {
    SCOPED_TRACE(tamper == Tamper::kRemoved
                     ? "no token"
                     : tamper == Tamper::kOtherEpoch ? "token of another run"
                                                     : "token others can write");
    fs::remove_all(dir_);
    const auto options =
        WithIdentity(Opts(1 << 20, 1 << 20, 4096), WarmRestart::kAuto, 1, 7);
    ASSERT_TRUE(CrashAfter(options, [](DiskSlabStore& store) {
      return PutFilled(store, 90, 'a');
    }));
    const fs::path token = TrustToken(dir_);
    ASSERT_TRUE(fs::exists(token)) << "the killed run was vouching";
    switch (tamper) {
      case Tamper::kRemoved:
        fs::remove(token);
        break;
      case Tamper::kOtherEpoch: {
        std::ofstream other(token, std::ios::binary | std::ios::trunc);
        other.write("\0\0\0\0\0\0\0\x7f", 8);
        break;
      }
      case Tamper::kWritableByOthers:
        ASSERT_EQ(::chmod(token.c_str(), 0666), 0);
        break;
    }
    bool ok = false;
    DiskSlabStore recovered(options, &ok);
    ASSERT_TRUE(ok) << recovered.StartupError();
    EXPECT_EQ(recovered.GetStats().warm_restarts, 0u);
    EXPECT_EQ(recovered.GetStats().unclean_resets, 1u);
    EXPECT_EQ(recovered.Count(), 0u);
  }
}

// A process killed inside the table's fdatasync has consumed whatever error
// that call was about to return, and never got to act on it.
TEST_F(DiskSlabTest, KilledInsideATableSyncForcesAReset) {
  const auto options =
      WithIdentity(Opts(1 << 20, 1 << 20, 4096), WarmRestart::kAuto, 1, 7);
  ASSERT_TRUE(CrashAfter(options, [](DiskSlabStore& store) {
    if (!PutFilled(store, 90, 'a')) return false;
    dfkv::DiskSlabStoreTestPeer::SetTableSyncHook(&store, [] { ::_exit(0); });
    dfkv::DiskSlabStoreTestPeer::SyncTableNow(&store);
    return false;  // not reached
  }));
  bool ok = false;
  DiskSlabStore recovered(options, &ok);
  ASSERT_TRUE(ok) << recovered.StartupError();
  EXPECT_EQ(recovered.GetStats().warm_restarts, 0u);
  EXPECT_EQ(recovered.GetStats().unclean_resets, 1u);
  EXPECT_EQ(recovered.Count(), 0u);
}

// STO-11: an extent's bind wipe is not synced on its own. A process killed
// right after a steal wrote the wipe (before the extent was used), or after
// the new class's first value, with no table sync since: the warm restart
// reads the table as the page cache holds it, so the stolen extent's old
// keys stay gone, and nothing else is lost.
TEST_F(DiskSlabTest, AKillAfterAnUnsyncedBindWipeLeavesNoStaleKeys) {
  for (const bool after_put : {false, true}) {
    SCOPED_TRACE(after_put ? "killed after the new value" : "killed after the wipe");
    fs::remove_all(dir_);
    fs::remove_all(dir_.string() + "_tokens");
    // 2 extents x 4 slots of 4 KiB: 8 class-A keys fill both; a class-B value
    // (8 KiB) then steals one extent.
    const auto options = WithIdentity(Opts(2 * 4 * 4096, 4 * 4096, 4096),
                                      WarmRestart::kAuto, 1, 7);
    const std::string a(4000, 'A'), b(8000, 'B');
    ASSERT_TRUE(CrashAfter(options, [&](DiskSlabStore& store) {
      for (int i = 0; i < 8; ++i)
        if (store.Cache(K(100 + i), a.data(), a.size()) != Status::kOk) return false;
      if (!after_put)
        dfkv::DiskSlabStoreTestPeer::SetBindHook(&store, [](uint32_t) { ::_exit(0); });
      if (store.Cache(K(500), b.data(), b.size()) != Status::kOk) return false;
      if (store.GetStats().bind_wipes == 0) return false;
      ::_exit(0);  // killed: no table sync since the wipe (table_sync_ms = 0)
    }));
    bool ok = false;
    DiskSlabStore recovered(options, &ok);
    ASSERT_TRUE(ok) << recovered.StartupError();
    EXPECT_EQ(recovered.GetStats().warm_restarts, 1u) << "the table was kept";
    int present = 0;
    std::string out;
    for (int i = 0; i < 8; ++i) {
      if (recovered.Range(K(100 + i), 0, 0, &out) != Status::kOk) continue;
      ++present;
      EXPECT_EQ(out, a) << i;
    }
    EXPECT_EQ(present, 4) << "the stolen extent's keys came back, or others were lost";
    if (after_put) {
      ASSERT_EQ(recovered.Range(K(500), 0, 0, &out), Status::kOk);
      EXPECT_EQ(out, b);
    } else {
      EXPECT_EQ(recovered.Range(K(500), 0, 0, &out), Status::kNotFound);
    }
    EXPECT_EQ(recovered.Count(), after_put ? 5u : 4u);
  }
}

TEST_F(DiskSlabTest, WarmRecoveryThatDiesHalfWayForcesAReset) {
  const auto base = Opts(1 << 20, 1 << 20, 4096);
  const auto options = WithIdentity(base, WarmRestart::kAuto, 1, 7);
  ASSERT_TRUE(CrashAfter(options, [](DiskSlabStore& store) {
    return PutFilled(store, 90, 'a') && PutFilled(store, 91, 'b');
  }));

  // A start that keeps the table but cannot write slab_state at the end.
  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    ::signal(SIGXFSZ, SIG_IGN);
    struct rlimit limit {};
    limit.rlim_cur = limit.rlim_max = 0;  // the empty marker still fits
    if (::setrlimit(RLIMIT_FSIZE, &limit) != 0) ::_exit(2);
    bool ok = true;
    DiskSlabStore store(options, &ok);
    ::_exit(ok ? 3 : 0);
  }
  int status = 0;
  ASSERT_EQ(::waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0) << "the staged recovery was meant to fail";
  ASSERT_EQ(ValidRecordOffsets(dir_).size(), 2u) << "and to leave the table";
  EXPECT_FALSE(fs::exists(TrustToken(dir_)));

  bool ok = false;
  DiskSlabStore recovered(options, &ok);
  ASSERT_TRUE(ok) << recovered.StartupError();
  EXPECT_EQ(recovered.GetStats().warm_restarts, 0u);
  EXPECT_EQ(recovered.GetStats().unclean_resets, 1u);
  EXPECT_EQ(recovered.Count(), 0u);
}

// Kept, killed again, kept again: the second run records its own identity and
// holds its own token, so nothing about the first kill lingers.
TEST_F(DiskSlabTest, TwoUncleanExitsInARowAreBothKept) {
  const auto options =
      WithIdentity(Opts(1 << 20, 1 << 20, 4096), WarmRestart::kAuto, 1, 7);
  ASSERT_TRUE(CrashAfter(options, [](DiskSlabStore& store) {
    return PutFilled(store, 90, 'a');
  }));
  ASSERT_TRUE(CrashAfter(options, [](DiskSlabStore& store) {
    return store.GetStats().warm_restarts == 1 && store.Count() == 1 &&
           PutFilled(store, 91, 'b') && store.Remove(K(90)) == Status::kOk;
  }));
  bool ok = false;
  DiskSlabStore recovered(options, &ok);
  ASSERT_TRUE(ok) << recovered.StartupError();
  EXPECT_EQ(recovered.GetStats().warm_restarts, 1u);
  EXPECT_TRUE(fs::exists(TrustToken(dir_)));
  std::string out;
  EXPECT_EQ(recovered.Range(K(90), 0, 4096, &out), Status::kNotFound);
  ASSERT_EQ(recovered.Range(K(91), 0, 4096, &out), Status::kOk);
  EXPECT_EQ(out, Payload('b'));
}

// A copy of a directory taken while its server ran is not that directory: the
// files were not copied at one instant.
TEST_F(DiskSlabTest, CopyOfADirectoryIsNotKept) {
  const auto options =
      WithIdentity(Opts(1 << 20, 1 << 20, 4096), WarmRestart::kAuto, 1, 7);
  ASSERT_TRUE(CrashAfter(options, [](DiskSlabStore& store) {
    return PutFilled(store, 90, 'a');
  }));
  const fs::path copy = dir_.string() + "_copy";
  fs::remove_all(copy);
  fs::copy(dir_, copy, fs::copy_options::recursive);
  auto copy_options = options;
  copy_options.dir = copy.string();
  {
    bool ok = false;
    DiskSlabStore recovered(copy_options, &ok);
    EXPECT_TRUE(ok) << recovered.StartupError();
    EXPECT_EQ(recovered.GetStats().warm_restarts, 0u);
    EXPECT_EQ(recovered.GetStats().unclean_resets, 1u);
    EXPECT_EQ(recovered.Count(), 0u);
  }
  fs::remove_all(copy);
}

// Records that contradict each other used to be unreachable after an unclean
// exit (the table was reset first) and fail startup after a clean one. On the
// warm path they mean the table cannot be trusted: reset, do not refuse to start.
TEST_F(DiskSlabTest, InconsistentTableOnTheWarmPathFallsBackToAReset) {
  const auto options =
      WithIdentity(Opts(1 << 20, 1 << 20, 4096), WarmRestart::kAuto, 1, 7);
  ASSERT_TRUE(CrashAfter(options, [](DiskSlabStore& store) {
    return PutFilled(store, 90, 'a') && PutFilled(store, 91, 'b');
  }));
  const std::vector<off_t> records = ValidRecordOffsets(dir_);
  ASSERT_EQ(records.size(), 2u);
  {  // the same key in two slots
    const int fd = ::open((dir_ / "slots.tbl").c_str(), O_RDWR);
    ASSERT_GE(fd, 0);
    char record[64];
    ASSERT_EQ(::pread(fd, record, sizeof(record), records[0]), 64);
    ASSERT_EQ(::pwrite(fd, record, sizeof(record), records[1]), 64);
    ::close(fd);
  }
  bool ok = false;
  DiskSlabStore recovered(options, &ok);
  ASSERT_TRUE(ok) << recovered.StartupError();
  EXPECT_EQ(recovered.GetStats().warm_restarts, 0u);
  EXPECT_EQ(recovered.GetStats().unclean_resets, 1u);
  EXPECT_EQ(recovered.UncleanResetReason(), "slots.tbl is not self-consistent");
  EXPECT_EQ(recovered.Count(), 0u);
  EXPECT_TRUE(PutFilled(recovered, 90, 'z'));
}

// With the system's real identity: kept on a kernel that reports a unique
// mount id, reset on one that does not -- and correct either way.
TEST_F(DiskSlabTest, UncleanExitWithTheRealSystemIdentity) {
  auto options = Opts(1 << 20, 1 << 20, 4096);
  options.table_sync_ms = 0;
  options.reclaim_interval_ms = 0;
  options.warm_restart = WarmRestart::kAuto;
  options.trust_token_dir.clear();  // /dev/shm, if it is a tmpfs
  ASSERT_TRUE(CrashAfter(options, [](DiskSlabStore& store) {
    return PutFilled(store, 90, 'a');
  }));
  bool ok = false;
  DiskSlabStore recovered(options, &ok);
  ASSERT_TRUE(ok) << recovered.StartupError();
  const auto stats = recovered.GetStats();
  ASSERT_EQ(stats.warm_restarts + stats.unclean_resets, 1u);
  std::string out;
  if (stats.warm_restarts == 1) {
    ASSERT_EQ(recovered.Range(K(90), 0, 4096, &out), Status::kOk);
    EXPECT_EQ(out, Payload('a'));
  } else {
    EXPECT_EQ(recovered.Count(), 0u);
  }
}

// One process per directory: the state file is locked for the store's life.
TEST_F(DiskSlabTest, SecondOpenOfALiveDirectoryIsRefused) {
  auto options = Opts(1 << 20, 1 << 20, 4096);
  options.table_sync_ms = 0;
  options.reclaim_interval_ms = 0;
  bool ok = false;
  DiskSlabStore first(options, &ok);
  ASSERT_TRUE(ok);
  ASSERT_TRUE(PutFilled(first, 90, 'a'));
  {
    bool second_ok = true;
    DiskSlabStore second(options, &second_ok);
    EXPECT_FALSE(second_ok);
    EXPECT_NE(second.StartupError().find("in use"), std::string::npos)
        << second.StartupError();
  }
  std::string out;  // the refused open left the live store alone
  ASSERT_EQ(first.Range(K(90), 0, 4096, &out), Status::kOk);
  EXPECT_EQ(out, Payload('a'));
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
  // A value that no extent can hold is an invalid request, the same however
  // often it is made: not an I/O error (a client would stop using the node)
  // and not a full cache (it would try again).
  std::string big(4097, 'b');
  EXPECT_EQ(s.Cache(K(1), big.data(), big.size()), Status::kInvalid);
  const auto batch = s.CacheDirectBatch({{K(1), big.data(), big.size(), 8192}});
  ASSERT_EQ(batch.size(), 1u);
  EXPECT_EQ(batch[0], Status::kInvalid);
  EXPECT_TRUE(s.Healthy());
  std::string okv(4096, 'o');
  EXPECT_EQ(s.Cache(K(2), okv.data(), okv.size()), Status::kOk);
}

// Every slot is held by an operation in flight and there is no extent to
// take: the store is healthy and cannot take another value right now. That
// is kCacheFull, which a client counts as "not stored" and goes on using the
// node; kIOError would have it mark the node bad and stay away from it.
TEST_F(DiskSlabTest, APutThatFindsNoSlotIsToldTheCacheIsFull) {
  auto options = Opts(4 * 4096, 4 * 4096, 4096);  // one extent, four slots
  options.table_sync_ms = 0;
  options.reclaim_interval_ms = 0;
  bool ok = false;
  DiskSlabStore store(options, &ok);
  ASSERT_TRUE(ok);
  std::string value(4000, 'p');
  for (uint64_t i = 0; i < 4; ++i) {
    ASSERT_EQ(store.Cache(K(i), value.data(), value.size()), Status::kOk);
    ASSERT_TRUE(dfkv::DiskSlabStoreTestPeer::PinSlot(&store, K(i)));
  }
  EXPECT_EQ(store.Cache(K(10), value.data(), value.size()),
            Status::kCacheFull);
  EXPECT_EQ(store.CacheDirect(K(11), value.data(), value.size(), value.size()),
            Status::kCacheFull);
  const auto batch = store.CacheDirectBatch(
      {{K(12), value.data(), value.size(), value.size()},
       {K(13), value.data(), value.size(), value.size()}});
  ASSERT_EQ(batch.size(), 2u);
  EXPECT_EQ(batch[0], Status::kCacheFull);
  EXPECT_EQ(batch[1], Status::kCacheFull);
  EXPECT_TRUE(store.Healthy()) << "nothing failed: there was no room";
  EXPECT_EQ(store.GetStats().put_no_slot, 4u);
  EXPECT_EQ(store.GetStats().metadata_io_errors, 0u);
  EXPECT_EQ(store.Count(), 4u);
  std::string out;
  for (uint64_t i = 0; i < 4; ++i)
    EXPECT_EQ(store.Range(K(i), 0, 0, &out), Status::kOk);

  // Room again once one of them lets go.
  ASSERT_TRUE(dfkv::DiskSlabStoreTestPeer::UnpinSlot(&store, K(0)));
  EXPECT_EQ(store.Cache(K(10), value.data(), value.size()), Status::kOk);
  EXPECT_EQ(store.GetStats().put_no_slot, 4u);
  for (uint64_t i = 1; i < 4; ++i)
    ASSERT_TRUE(dfkv::DiskSlabStoreTestPeer::UnpinSlot(&store, K(i)));
}

// Protected by retention is the same case: nothing may be evicted, the node
// is fine.
TEST_F(DiskSlabTest, APutIntoAStoreFullOfRetainedValuesIsToldTheCacheIsFull) {
  ::setenv("DFKV_RETAIN_MAX_PCT", "95", 1);
  ::setenv("DFKV_RETAIN_MAX_CLASS_PCT", "100", 1);
  auto options = Opts(64 * 4096, 64 * 4096, 4096);  // one extent, 64 slots
  options.table_sync_ms = 0;
  options.reclaim_interval_ms = 0;
  bool ok = false;
  DiskSlabStore store(options, &ok);
  ::unsetenv("DFKV_RETAIN_MAX_PCT");
  ::unsetenv("DFKV_RETAIN_MAX_CLASS_PCT");
  ASSERT_TRUE(ok);
  std::string value(4000, 'r');
  const uint32_t now = static_cast<uint32_t>(std::time(nullptr));
  size_t protected_count = 0;
  for (uint64_t i = 0; i < 64; ++i) {
    ASSERT_EQ(store.Cache(K(i), value.data(), value.size()), Status::kOk);
    const uint8_t code = store.Retain(K(i), 600, /*mode=*/0, now);
    if (code == static_cast<uint8_t>(
                    dfkv::SlabAllocator::RetainCode::kProtectedNew))
      ++protected_count;
    else  // beyond the budget: held by a reader instead
      ASSERT_TRUE(dfkv::DiskSlabStoreTestPeer::PinSlot(&store, K(i)));
  }
  ASSERT_GE(protected_count, 32u);
  EXPECT_EQ(store.Cache(K(100), value.data(), value.size()),
            Status::kCacheFull);
  EXPECT_TRUE(store.Healthy());
  EXPECT_EQ(store.GetStats().put_no_slot, 1u);
  for (uint64_t i = 0; i < 64; ++i)
    dfkv::DiskSlabStoreTestPeer::UnpinSlot(&store, K(i));
}

// A real failure while room is being made is still an I/O error, and the
// store is unhealthy afterwards: the two are told apart by what happened,
// not by where the refusal came from. (EvictionRecordClearFailure... above
// covers the scalar path; this is the batch.)
TEST_F(DiskSlabTest, AFailedEvictionDuringABatchPutIsStillAnIoError) {
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
  const auto batch = store.CacheDirectBatch(
      {{K(4), value.data(), value.size(), value.size()}});
  ASSERT_EQ(batch.size(), 1u);
  EXPECT_EQ(batch[0], Status::kIOError);
  EXPECT_FALSE(store.Healthy());
  EXPECT_EQ(store.GetStats().put_no_slot, 0u);
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

TEST_F(DiskSlabTest, ReadLeaseMovesAndReleasesOnDestructionAndError) {
  bool ok = false;
  DiskSlabStore s(Opts(1 << 20, 1 << 20, 4096), &ok);
  ASSERT_TRUE(ok);
  std::string v(5000, 'p');
  ASSERT_EQ(s.Cache(K(1), v.data(), v.size()), Status::kOk);

  int owned_fd = -1;
  {
    dfkv::ReadLease lease;
    ASSERT_EQ(s.RangeDirectPrep(K(1), 0, 0, 1 << 20, &lease),
              Status::kOk);
    ASSERT_GE(lease.fd(), 0);
    owned_fd = lease.fd();
    EXPECT_EQ(lease.payload_len, v.size());
    EXPECT_EQ(lease.aligned_off % 4096, 0u);
    EXPECT_EQ(lease.aligned_len % 4096, 0u);

    dfkv::ReadLease moved(std::move(lease));
    EXPECT_EQ(lease.fd(), -1);
    EXPECT_EQ(moved.fd(), owned_fd);
    std::vector<char> rbuf(moved.aligned_len);
    ASSERT_EQ(::pread(moved.fd(), rbuf.data(), moved.aligned_len,
                     static_cast<off_t>(moved.aligned_off)),
              static_cast<ssize_t>(moved.aligned_len));
    EXPECT_EQ(std::string(rbuf.data() + moved.head, moved.payload_len), v);

    EXPECT_EQ(s.Remove(K(1)), Status::kOk);
    EXPECT_FALSE(s.IsCached(K(1)));
    EXPECT_EQ(s.Count(), 1u) << "moved lease must retain the slot pin";
  }
  EXPECT_EQ(s.Count(), 0u) << "destruction executes the deferred remove";
  errno = 0;
  EXPECT_EQ(::fcntl(owned_fd, F_GETFD), -1);
  EXPECT_EQ(errno, EBADF);

  // Reusing an output for an error destroys its prior lease first, balancing
  // both descriptor and storage hold without an explicit release call.
  ASSERT_EQ(s.Cache(K(2), v.data(), v.size()), Status::kOk);
  dfkv::ReadLease lease;
  ASSERT_EQ(s.RangeDirectPrep(K(2), 0, 0, 1 << 20, &lease), Status::kOk);
  EXPECT_EQ(s.Remove(K(2)), Status::kOk);
  EXPECT_EQ(s.Count(), 1u);
  EXPECT_EQ(s.RangeDirectPrep(K(9), 0, 0, 1 << 20, &lease),
            Status::kNotFound);
  EXPECT_EQ(lease.fd(), -1);
  EXPECT_EQ(s.Count(), 0u);

  // Acquire succeeds before the aligned-window cap check fails. That early
  // error must still balance the pin/inflight hold.
  ASSERT_EQ(s.Cache(K(3), v.data(), v.size()), Status::kOk);
  EXPECT_EQ(s.RangeDirectPrep(K(3), 0, 0, 1, &lease), Status::kInvalid);
  const auto stats = s.GetStats();
  EXPECT_EQ(stats.prep_holds, 0u);
  EXPECT_EQ(stats.inflight, 0u);
  EXPECT_EQ(s.Remove(K(3)), Status::kOk);
  EXPECT_EQ(s.Count(), 0u);
}

// slots.tbl is what a restart rebuilds from, so a Remove that has returned must
// leave no valid record for the key there -- also while a reader still pins
// the slot and the physical removal is deferred. The record used to stay valid
// until the last reader finished.
TEST_F(DiskSlabTest, RemoveInvalidatesTheRecordWhileReadersAreInFlight) {
  auto opts = Opts(1 << 20, 1 << 20, 4096);
  opts.table_sync_ms = 0;
  opts.reclaim_interval_ms = 0;
  bool ok = false;
  DiskSlabStore s(opts, &ok);
  ASSERT_TRUE(ok);
  const std::string value(4096, 'a');
  ASSERT_EQ(s.Cache(K(80), value.data(), value.size()), Status::kOk);
  ASSERT_TRUE(dfkv::DiskSlabStoreTestPeer::HasValidRecordOnDisk(&s, K(80)));

  dfkv::ReadLease lease;
  ASSERT_EQ(s.RangeDirectPrep(K(80), 0, 0, 1 << 20, &lease), Status::kOk);
  ASSERT_EQ(s.Remove(K(80)), Status::kOk);
  EXPECT_EQ(s.Count(), 1u);  // the reader still pins the slot
  EXPECT_FALSE(dfkv::DiskSlabStoreTestPeer::HasValidRecordOnDisk(&s, K(80)))
      << "Remove returned while slots.tbl still held a valid record";
  { dfkv::ReadLease drop(std::move(lease)); }
  EXPECT_EQ(s.Count(), 0u);
  EXPECT_FALSE(dfkv::DiskSlabStoreTestPeer::HasValidRecordOnDisk(&s, K(80)));
}

// If that invalidation cannot be written, the Remove must not be acknowledged.
TEST_F(DiskSlabTest, RemoveFailsWhenItCannotInvalidateUnderInflightReaders) {
  auto opts = Opts(1 << 20, 1 << 20, 4096);
  opts.table_sync_ms = 0;
  opts.reclaim_interval_ms = 0;
  bool ok = false;
  DiskSlabStore s(opts, &ok);
  ASSERT_TRUE(ok);
  const std::string value(4096, 'a');
  ASSERT_EQ(s.Cache(K(82), value.data(), value.size()), Status::kOk);
  dfkv::ReadLease lease;
  ASSERT_EQ(s.RangeDirectPrep(K(82), 0, 0, 1 << 20, &lease), Status::kOk);
  dfkv::DiskSlabStoreTestPeer::SetRecordWriteHook(&s, [] { return false; });
  EXPECT_EQ(s.Remove(K(82)), Status::kIOError);
  EXPECT_FALSE(s.Healthy());
  EXPECT_TRUE(dfkv::DiskSlabStoreTestPeer::HasValidRecordOnDisk(&s, K(82)))
      << "the record is untouched, which is why the Remove may not succeed";
  dfkv::DiskSlabStoreTestPeer::SetRecordWriteHook(&s, {});
}

// The same rule for a write in flight: the writer that a Remove cancelled must
// not put a valid record into slots.tbl after that Remove returned. It used to
// write the record before it learned about the Remove and invalidate it a
// moment later. Checked from the record-write hook, which runs right before
// every record write: none of them may find the key valid on disk.
TEST_F(DiskSlabTest, WriteCancelledByRemoveNeverWritesAValidRecord) {
  for (const bool batch : {false, true}) {
    SCOPED_TRACE(batch ? "batch PUT" : "scalar PUT");
    fs::remove_all(dir_);
    auto opts = Opts(1 << 20, 1 << 20, 4096);
    opts.table_sync_ms = 0;
    opts.reclaim_interval_ms = 0;
    bool ok = false;
    DiskSlabStore s(opts, &ok);
    ASSERT_TRUE(ok);

    HookGate payload_gate;
    dfkv::DiskSlabStoreTestPeer::SetPayloadWriteHook(
        &s, [&] { return payload_gate.Run(); });
    std::atomic<bool> remove_returned{false};
    std::atomic<bool> valid_after_remove{false};
    dfkv::DiskSlabStoreTestPeer::SetRecordWriteHook(&s, [&] {
      if (remove_returned.load() &&
          dfkv::DiskSlabStoreTestPeer::HasValidRecordOnDisk(&s, K(81)))
        valid_after_remove.store(true);
      return true;
    });

    std::string value(4096, 'a');
    Status put = Status::kInvalid;
    std::thread writer([&] {
      if (batch) {
        put = s.CacheDirectBatch(
            {{K(81), value.data(), value.size(), value.size()}})[0];
      } else {
        put = s.Cache(K(81), value.data(), value.size());
      }
    });
    payload_gate.WaitEntered();
    EXPECT_EQ(s.Remove(K(81)), Status::kOk);  // deferred behind the write
    remove_returned.store(true);
    payload_gate.Release(true);
    writer.join();
    dfkv::DiskSlabStoreTestPeer::SetPayloadWriteHook(&s, {});
    dfkv::DiskSlabStoreTestPeer::SetRecordWriteHook(&s, {});

    EXPECT_EQ(put, Status::kOk);
    EXPECT_FALSE(valid_after_remove.load())
        << "a valid record for the removed key reached slots.tbl after the "
           "Remove had returned";
    EXPECT_FALSE(dfkv::DiskSlabStoreTestPeer::HasValidRecordOnDisk(&s, K(81)));
    EXPECT_FALSE(s.IsCached(K(81)));
    EXPECT_EQ(s.Count(), 0u);
  }
}

// Deferred-remove re-PUT (#17): Remove(K) while a read lease pins the slot
// defers the physical remove -- commit state is erased, but the allocator
// slot stays resident. A re-PUT arriving in that window used to hard-fail
// kIOError against the still-resident slot; it must instead wait (bounded)
// for the in-flight holder to release and then land normally.
TEST_F(DiskSlabTest, ReputDuringDeferredRemoveWaitsOutWindowThenLands) {
  auto opts = Opts(1 << 20, 1 << 20, 4096);
  opts.table_sync_ms = 0;
  opts.reclaim_interval_ms = 0;
  bool ok = false;
  DiskSlabStore s(opts, &ok);
  ASSERT_TRUE(ok);
  std::string v1(4096, 'a');
  ASSERT_EQ(s.Cache(K(60), v1.data(), v1.size()), Status::kOk);
  dfkv::ReadLease lease;
  ASSERT_EQ(s.RangeDirectPrep(K(60), 0, 0, 1 << 20, &lease), Status::kOk);
  ASSERT_GE(lease.fd(), 0);
  ASSERT_EQ(s.Remove(K(60)), Status::kOk);
  EXPECT_FALSE(s.IsCached(K(60)));  // commit state erased immediately
  EXPECT_EQ(s.Count(), 1u);         // slot still pinned: the window

  std::string v2(4096, 'b');
  Status reput = Status::kInvalid;
  std::atomic<bool> reput_done{false};
  std::thread t([&] {
    reput = s.Cache(K(60), v2.data(), v2.size());
    reput_done.store(true, std::memory_order_release);
  });
  // The window closes only when the lease below is destroyed, so the re-PUT
  // cannot complete yet; the old hard-fail path would already be done.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(reput_done.load(std::memory_order_acquire))
      << "re-PUT in the window must wait for the pin, not fail at once";
  { dfkv::ReadLease drop(std::move(lease)); }  // destruction releases the pin
  t.join();
  EXPECT_EQ(reput, Status::kOk);
  std::string out;
  ASSERT_EQ(s.Range(K(60), 0, 0, &out), Status::kOk);
  EXPECT_EQ(out, v2) << "readers must observe the re-PUT bytes, not stale ones";
  EXPECT_EQ(s.Count(), 1u);

  // Guard rails: outside the window nothing changed -- remove-then-re-PUT was
  // already fine, and a committed key still wins keep-first.
  ASSERT_EQ(s.Remove(K(60)), Status::kOk);  // no pin now: immediate remove
  EXPECT_EQ(s.Count(), 0u);
  std::string v3(4096, 'c');
  EXPECT_EQ(s.Cache(K(60), v3.data(), v3.size()), Status::kOk);
  std::string v4(4096, 'd');
  EXPECT_EQ(s.Cache(K(60), v4.data(), v4.size()), Status::kOk);
  ASSERT_EQ(s.Range(K(60), 0, 0, &out), Status::kOk);
  EXPECT_EQ(out, v3) << "a duplicate PUT must not overwrite committed bytes";
}

// A PUT that arrives after Remove(K) returned, while the write Remove cancelled
// is still in flight, must not join that write: its leader reports kOk without
// publishing K, so a follower would be told kOk for a value that is not stored.
// The late PUT waits out the removal and then stores its own bytes.
TEST_F(DiskSlabTest, PutAfterRemoveOfInflightWriteIsStored) {
  auto opts = Opts(1 << 20, 1 << 20, 4096);
  opts.table_sync_ms = 0;
  opts.reclaim_interval_ms = 0;
  bool ok = false;
  DiskSlabStore s(opts, &ok);
  ASSERT_TRUE(ok);

  std::mutex m;
  std::condition_variable cv;
  bool entered = false, release = false;
  std::atomic<int> payload_writes{0};
  dfkv::DiskSlabStoreTestPeer::SetPayloadWriteHook(&s, [&] {
    if (payload_writes.fetch_add(1, std::memory_order_relaxed) == 0) {
      std::unique_lock<std::mutex> lk(m);
      entered = true;
      cv.notify_all();
      cv.wait(lk, [&] { return release; });
    }
    return true;
  });

  const std::string v1(4096, 'a'), v2(4096, 'b');
  Status first = Status::kInvalid, second = Status::kInvalid;
  std::thread leader([&] { first = s.Cache(K(70), v1.data(), v1.size()); });
  {
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [&] { return entered; });
  }
  EXPECT_EQ(s.Remove(K(70)), Status::kOk);  // deferred behind the write
  std::atomic<bool> second_done{false};
  std::thread late([&] {
    second = s.Cache(K(70), v2.data(), v2.size());
    second_done.store(true, std::memory_order_release);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(second_done.load(std::memory_order_acquire));
  {
    std::lock_guard<std::mutex> lk(m);
    release = true;
  }
  cv.notify_all();
  leader.join();
  late.join();
  dfkv::DiskSlabStoreTestPeer::SetPayloadWriteHook(&s, {});

  EXPECT_EQ(first, Status::kOk);
  EXPECT_EQ(second, Status::kOk);
  std::string out;
  ASSERT_EQ(s.Range(K(70), 0, 0, &out), Status::kOk)
      << "a PUT acknowledged after the Remove must be readable";
  EXPECT_EQ(out, v2);
  EXPECT_EQ(s.Count(), 1u);
}

// Batch entry of the same ordering: the late PUT arrives through
// CacheDirectBatch while the cancelled scalar write is still in flight.
TEST_F(DiskSlabTest, BatchPutAfterRemoveOfInflightWriteIsStored) {
  auto opts = Opts(1 << 20, 1 << 20, 4096);
  opts.table_sync_ms = 0;
  opts.reclaim_interval_ms = 0;
  bool ok = false;
  DiskSlabStore s(opts, &ok);
  ASSERT_TRUE(ok);

  std::mutex m;
  std::condition_variable cv;
  bool entered = false, release = false;
  std::atomic<int> payload_writes{0};
  dfkv::DiskSlabStoreTestPeer::SetPayloadWriteHook(&s, [&] {
    if (payload_writes.fetch_add(1, std::memory_order_relaxed) == 0) {
      std::unique_lock<std::mutex> lk(m);
      entered = true;
      cv.notify_all();
      cv.wait(lk, [&] { return release; });
    }
    return true;
  });

  const std::string v1(4096, 'a');
  std::string v2(4096, 'b');
  Status first = Status::kInvalid;
  std::vector<Status> second;
  std::thread leader([&] { first = s.Cache(K(71), v1.data(), v1.size()); });
  {
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [&] { return entered; });
  }
  EXPECT_EQ(s.Remove(K(71)), Status::kOk);  // deferred behind the write
  std::thread late([&] {
    second = s.CacheDirectBatch(
        {{K(71), v2.data(), v2.size(), v2.size()}});
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  {
    std::lock_guard<std::mutex> lk(m);
    release = true;
  }
  cv.notify_all();
  leader.join();
  late.join();
  dfkv::DiskSlabStoreTestPeer::SetPayloadWriteHook(&s, {});

  EXPECT_EQ(first, Status::kOk);
  ASSERT_EQ(second.size(), 1u);
  EXPECT_EQ(second[0], Status::kOk);
  std::string out;
  ASSERT_EQ(s.Range(K(71), 0, 0, &out), Status::kOk)
      << "a batch PUT acknowledged after the Remove must be readable";
  EXPECT_EQ(out, v2);
}

// Same window through the batched PUT path: CacheDirectBatch items must wait
// out a deferred remove exactly like the scalar Cache path.
TEST_F(DiskSlabTest, BatchReputDuringDeferredRemoveWaitsOutWindowThenLands) {
  auto opts = Opts(1 << 20, 1 << 20, 4096);
  opts.table_sync_ms = 0;
  opts.reclaim_interval_ms = 0;
  bool ok = false;
  DiskSlabStore s(opts, &ok);
  ASSERT_TRUE(ok);
  std::string v1(4096, 'a');
  ASSERT_EQ(s.Cache(K(61), v1.data(), v1.size()), Status::kOk);
  dfkv::ReadLease lease;
  ASSERT_EQ(s.RangeDirectPrep(K(61), 0, 0, 1 << 20, &lease), Status::kOk);
  ASSERT_GE(lease.fd(), 0);
  ASSERT_EQ(s.Remove(K(61)), Status::kOk);
  EXPECT_EQ(s.Count(), 1u);

  std::string v2(4096, 'e');
  std::vector<Status> sts;
  std::atomic<bool> done{false};
  std::thread t([&] {
    sts = s.CacheDirectBatch({{K(61), v2.data(), v2.size(), v2.size()}});
    done.store(true, std::memory_order_release);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(done.load(std::memory_order_acquire));
  { dfkv::ReadLease drop(std::move(lease)); }
  t.join();
  ASSERT_EQ(sts.size(), 1u);
  EXPECT_EQ(sts[0], Status::kOk);
  std::string out;
  ASSERT_EQ(s.Range(K(61), 0, 0, &out), Status::kOk);
  EXPECT_EQ(out, v2);
}

// A healthy store that cannot take a key yet answers kCacheFull, on the
// scalar and on the batched path: when a deferred remove of the key outlasts
// the wait bound, and when the key still holds a slot from an earlier
// operation that has not let go. Clients cool a node down on kIOError; these
// used to be kIOError.
TEST_F(DiskSlabTest, AKeyTheStoreCannotTakeYetIsCacheFullNotAnIoError) {
  auto opts = Opts(1 << 20, 1 << 20, 4096);
  opts.table_sync_ms = 0;
  opts.reclaim_interval_ms = 0;
  bool ok = false;
  DiskSlabStore s(opts, &ok);
  ASSERT_TRUE(ok);
  std::string v(4096, 'a');
  ASSERT_EQ(s.Cache(K(62), v.data(), v.size()), Status::kOk);
  dfkv::ReadLease lease;
  ASSERT_EQ(s.RangeDirectPrep(K(62), 0, 0, 1 << 20, &lease), Status::kOk);
  ASSERT_EQ(s.Remove(K(62)), Status::kOk);  // deferred: the lease pins it
  Status scalar = Status::kOk;
  std::vector<Status> batch;
  std::thread a([&] { scalar = s.Cache(K(62), v.data(), v.size()); });
  std::thread b([&] {
    batch = s.CacheDirectBatch({{K(62), v.data(), v.size(), v.size()}});
  });
  a.join();
  b.join();  // both waited out the 5 s bound
  EXPECT_EQ(scalar, Status::kCacheFull);
  ASSERT_EQ(batch.size(), 1u);
  EXPECT_EQ(batch[0], Status::kCacheFull);
  EXPECT_TRUE(s.Healthy());
  { dfkv::ReadLease drop(std::move(lease)); }
  EXPECT_EQ(s.Cache(K(62), v.data(), v.size()), Status::kOk);

  // A slot taken and written, its operation not finished (no record, still
  // pinned and in flight).
  ASSERT_TRUE(dfkv::DiskSlabStoreTestPeer::WriteCrashStage(&s, K(63), v, false));
  EXPECT_EQ(s.Cache(K(63), v.data(), v.size()), Status::kCacheFull);
  const auto sts = s.CacheDirectBatch({{K(63), v.data(), v.size(), v.size()}});
  ASSERT_EQ(sts.size(), 1u);
  EXPECT_EQ(sts[0], Status::kCacheFull);
  EXPECT_TRUE(s.Healthy());
  EXPECT_EQ(s.GetStats().put_no_slot, 0u) << "neither is a missing slot";
  EXPECT_EQ(s.GetStats().put_busy, 4u);
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

TEST_F(DiskSlabTest, BatchDuplicateWaitsForScalarLeaderCommit) {
  auto opts = Opts(1 << 20, 1 << 20, 4096);
  opts.table_sync_ms = 0;
  opts.reclaim_interval_ms = 0;
  bool ok = false;
  DiskSlabStore s(opts, &ok);
  ASSERT_TRUE(ok);
  HookGate gate;
  dfkv::DiskSlabStoreTestPeer::SetPayloadWriteHook(
      &s, [&gate] { return gate.Run(); });
  std::string leader_value(3000, 'a');
  std::string follower_value(3000, 'b');
  Status leader = Status::kInvalid;
  std::vector<Status> follower;
  std::atomic<bool> follower_done{false};
  std::thread first([&] {
    leader = s.Cache(K(70), leader_value.data(), leader_value.size());
  });
  gate.WaitEntered();
  std::thread second([&] {
    follower = s.CacheDirectBatch(
        {{K(70), follower_value.data(), follower_value.size(),
          follower_value.size()}});
    follower_done.store(true, std::memory_order_release);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_FALSE(follower_done.load(std::memory_order_acquire));
  gate.Release(true);
  first.join();
  second.join();
  ASSERT_EQ(leader, Status::kOk);
  ASSERT_EQ(follower.size(), 1u);
  EXPECT_EQ(follower[0], Status::kOk);
  std::string out;
  ASSERT_EQ(s.Range(K(70), 0, 0, &out), Status::kOk);
  EXPECT_EQ(out, leader_value);
}

TEST_F(DiskSlabTest, DuplicateSharesLeaderPayloadFailure) {
  auto opts = Opts(1 << 20, 1 << 20, 4096);
  opts.table_sync_ms = 0;
  opts.reclaim_interval_ms = 0;
  bool ok = false;
  DiskSlabStore s(opts, &ok);
  ASSERT_TRUE(ok);
  HookGate gate;
  dfkv::DiskSlabStoreTestPeer::SetPayloadWriteHook(
      &s, [&gate] { return gate.Run(); });
  std::string value(3000, 'p');
  Status leader = Status::kInvalid;
  Status follower = Status::kInvalid;
  std::atomic<bool> follower_done{false};
  std::thread first([&] {
    leader = s.Cache(K(71), value.data(), value.size());
  });
  gate.WaitEntered();
  std::thread second([&] {
    follower = s.Cache(K(71), value.data(), value.size());
    follower_done.store(true, std::memory_order_release);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_FALSE(follower_done.load(std::memory_order_acquire));
  gate.Release(false);
  first.join();
  second.join();
  EXPECT_EQ(leader, Status::kIOError);
  EXPECT_EQ(follower, Status::kIOError);
  EXPECT_FALSE(s.IsCached(K(71)));
}

TEST_F(DiskSlabTest, DuplicateSharesLeaderMetadataFailure) {
  auto opts = Opts(1 << 20, 1 << 20, 4096);
  opts.table_sync_ms = 0;
  opts.reclaim_interval_ms = 0;
  bool ok = false;
  DiskSlabStore s(opts, &ok);
  ASSERT_TRUE(ok);
  // The leader is held in its payload write, outside the store mutex, so the
  // second writer can join its flight. The record write then fails without
  // blocking: it runs at commit, under the mutex.
  HookGate gate;
  dfkv::DiskSlabStoreTestPeer::SetPayloadWriteHook(
      &s, [&gate] { return gate.Run(); });
  dfkv::DiskSlabStoreTestPeer::SetRecordWriteHook(&s, [] { return false; });
  std::string value(3000, 'm');
  Status leader = Status::kInvalid;
  Status follower = Status::kInvalid;
  std::atomic<bool> follower_done{false};
  std::thread first([&] {
    leader = s.Cache(K(72), value.data(), value.size());
  });
  gate.WaitEntered();
  std::thread second([&] {
    follower = s.Cache(K(72), value.data(), value.size());
    follower_done.store(true, std::memory_order_release);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_FALSE(follower_done.load(std::memory_order_acquire));
  gate.Release(true);
  first.join();
  second.join();
  EXPECT_EQ(leader, Status::kIOError);
  EXPECT_EQ(follower, Status::kIOError);
  EXPECT_FALSE(s.Healthy());
  EXPECT_FALSE(s.IsCached(K(72)));
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

TEST_F(DiskSlabTest, CloseReportsAStoreThatCouldNotBeMarkedClean) {
  auto options = Opts(1 << 20, 1 << 20, 4096);
  options.table_sync_ms = 0;
  options.reclaim_interval_ms = 0;
  bool ok = false;
  DiskSlabStore store(options, &ok);
  ASSERT_TRUE(ok);
  dfkv::DiskSlabStoreTestPeer::BreakStateFd(&store);
  EXPECT_FALSE(store.Close());
  EXPECT_FALSE(store.Close());  // the answer does not change
  EXPECT_FALSE(store.Healthy());
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
  // Keep one slot below low; capacity / 100 truncates before multiplication.
  ::setenv("DFKV_SLAB_EVICT_LOW_PCT", "26", 1);
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
    EXPECT_EQ(store.Count(), 1u);
    EXPECT_EQ(store.GetStats().eviction_record_clears, 3u);
  }

  bool ok = false;
  DiskSlabStore reopened(options, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(reopened.Count(), 1u);
  for (uint64_t i = 0; i < 3; ++i) EXPECT_FALSE(reopened.IsCached(K(i)));
  std::string out;
  ASSERT_EQ(reopened.Range(K(3), 0, 4000, &out), Status::kOk);
  EXPECT_EQ(out, std::string(4000, 'e'));
}

TEST_F(DiskSlabTest, SlotWatermarkEvictionRemainsColdAfterDirtyRestart) {
  auto options = Opts(4 * 4096, 4 * 4096, 4096);
  options.reclaim_interval_ms = 0;
  options.table_sync_ms = 0;
  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    ::setenv("DFKV_SLAB_EVICT_HIGH_PCT", "50", 1);
    ::setenv("DFKV_SLAB_EVICT_LOW_PCT", "26", 1);
    bool ok = false;
    DiskSlabStore store(options, &ok);
    if (!ok) ::_exit(1);
    const std::string value(4000, 'd');
    for (uint64_t i = 0; i < 4; ++i)
      if (store.Cache(K(i), value.data(), value.size()) != Status::kOk)
        ::_exit(2);
    dfkv::DiskSlabStoreTestPeer::ReclaimNow(&store, {});
    if (store.Count() != 1 ||
        store.GetStats().eviction_record_clears != 3)
      ::_exit(3);
    ::_exit(0);  // no clean epoch: neither victim nor survivor may resurrect
  }
  int status = 0;
  ASSERT_EQ(::waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
  bool ok = false;
  DiskSlabStore reopened(options, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(reopened.Count(), 0u);
  for (uint64_t i = 0; i < 4; ++i) EXPECT_FALSE(reopened.IsCached(K(i)));
}

// STO-13: DFKV_SLAB_EVICT_HEADROOM_BYTES / _BAND_BYTES place the watermarks
// at absolute distances below capacity and take precedence over the
// percentages. Clamps: the band at least one extent, the headroom at least
// two bands. Fallback to the percentages when the resolved high is below the
// percentage high or below the band (a small disk), or nothing is left.
TEST_F(DiskSlabTest, AbsoluteHeadroomResolvesClampsAndFallsBack) {
  // 4 extents of 4 slots of 4 KiB: capacity 64 KiB, extent 16 KiB; capacity
  // / 100 = 655, so 10/5 percent are 6550/3275 and 92/88 are 60260/57640.
  struct Case { const char* high_pct; const char* low_pct; const char* headroom;
                const char* band; uint64_t high, low; };
  for (const Case& c : {
           // applied; the band clamped up to one extent
           Case{"10", "5", "40000", nullptr, 65536 - 40000, 65536 - 40000 - 16384},
           // applied; both clamped up
           Case{"10", "5", "1", nullptr, 32768, 16384},
           // high (15536) below the band (20000): percentages
           Case{"10", "5", "50000", "20000", 6550, 3275},
           // nothing left below capacity: percentages
           Case{"10", "5", "65536", nullptr, 6550, 3275},
           // a small disk: the clamped headroom leaves high (32768) below
           // the default 92% (60260): percentages
           Case{nullptr, nullptr, "1", nullptr, 60260, 57640},
           // a band alone does nothing
           Case{nullptr, nullptr, nullptr, "20000", 60260, 57640},
           // HIGH_PCT=0 (watermark off) is overridden by a set headroom
           Case{"0", nullptr, "1", nullptr, 32768, 16384}}) {
    SCOPED_TRACE(std::string(c.high_pct ? c.high_pct : "-") + "% " +
                 (c.headroom ? c.headroom : "-") + "/" + (c.band ? c.band : "-"));
    fs::remove_all(dir_);
    if (c.high_pct) ::setenv("DFKV_SLAB_EVICT_HIGH_PCT", c.high_pct, 1);
    if (c.low_pct) ::setenv("DFKV_SLAB_EVICT_LOW_PCT", c.low_pct, 1);
    if (c.headroom) ::setenv("DFKV_SLAB_EVICT_HEADROOM_BYTES", c.headroom, 1);
    if (c.band) ::setenv("DFKV_SLAB_EVICT_BAND_BYTES", c.band, 1);
    auto options = Opts(4 * 4 * 4096, 4 * 4096, 4096);
    options.reclaim_interval_ms = 0;
    bool ok = false;
    DiskSlabStore store(options, &ok);
    for (const char* n : {"DFKV_SLAB_EVICT_HIGH_PCT", "DFKV_SLAB_EVICT_LOW_PCT",
                          "DFKV_SLAB_EVICT_HEADROOM_BYTES", "DFKV_SLAB_EVICT_BAND_BYTES"})
      ::unsetenv(n);
    ASSERT_TRUE(ok);
    const auto w = store.GetWatermarkObservation();
    EXPECT_EQ(w.watermark_high_bytes, c.high);
    EXPECT_EQ(w.watermark_low_bytes, c.low);
  }
}

// The drain follows the absolute values: 10 of 16 slots used is above both
// highs; with the headroom applied (it is above the 10% high) the drain
// stops at its low of 16 KiB, with the percentages alone it goes on to 5%.
TEST_F(DiskSlabTest, AbsoluteHeadroomDrivesTheWatermarkDrain) {
  for (const bool with_headroom : {false, true}) {
    SCOPED_TRACE(with_headroom ? "headroom" : "percentages only");
    fs::remove_all(dir_);
    ::setenv("DFKV_SLAB_EVICT_HIGH_PCT", "10", 1);
    ::setenv("DFKV_SLAB_EVICT_LOW_PCT", "5", 1);
    if (with_headroom) ::setenv("DFKV_SLAB_EVICT_HEADROOM_BYTES", "1", 1);
    auto options = Opts(4 * 4 * 4096, 4 * 4096, 4096);
    options.reclaim_interval_ms = 0;
    options.table_sync_ms = 0;
    bool ok = false;
    DiskSlabStore store(options, &ok);
    ::unsetenv("DFKV_SLAB_EVICT_HIGH_PCT");
    ::unsetenv("DFKV_SLAB_EVICT_LOW_PCT");
    ::unsetenv("DFKV_SLAB_EVICT_HEADROOM_BYTES");
    ASSERT_TRUE(ok);
    const std::string value(4000, 'h');
    for (uint64_t i = 0; i < 10; ++i)
      ASSERT_EQ(store.Cache(K(i), value.data(), value.size()), Status::kOk);
    for (int tick = 0; tick < 8; ++tick)
      dfkv::DiskSlabStoreTestPeer::ReclaimNow(&store, {});
    EXPECT_GT(store.GetStats().watermark_evictions, 0u);
    if (with_headroom) {
      EXPECT_LE(store.Count(), 4u) << "drained to the headroom's low (16 KiB)";
      EXPECT_GE(store.Count(), 3u) << "and no further";
    } else {
      EXPECT_LE(store.Count(), 1u) << "drained to the 5% low";
    }
  }
}

TEST_F(DiskSlabTest, WatermarkDrainIsLatchedAndBoundedPerTick) {
  ::setenv("DFKV_SLAB_EVICT_HIGH_PCT", "50", 1);
  ::setenv("DFKV_SLAB_EVICT_LOW_PCT", "26", 1);
  ::setenv("DFKV_SLAB_EVICT_MAX_VICTIMS_PER_TICK", "1", 1);
  auto options = Opts(8 * 4096, 4096, 4096);
  options.reclaim_interval_ms = 0;
  options.table_sync_ms = 0;
  bool ok = false;
  DiskSlabStore store(options, &ok);
  ::unsetenv("DFKV_SLAB_EVICT_HIGH_PCT");
  ::unsetenv("DFKV_SLAB_EVICT_LOW_PCT");
  ::unsetenv("DFKV_SLAB_EVICT_MAX_VICTIMS_PER_TICK");
  ASSERT_TRUE(ok);

  std::string value(4000, 'w');
  for (uint64_t i = 0; i < 8; ++i)
    ASSERT_EQ(store.Cache(K(i), value.data(), value.size()), Status::kOk);

  dfkv::DiskSlabStoreTestPeer::ReclaimNow(&store, {});
  auto stats = store.GetStats();
  EXPECT_EQ(store.Count(), 6u)
      << "one bounded watermark victim plus regular free-slot reclaim";
  EXPECT_EQ(stats.watermark_evictions, 1u);
  EXPECT_EQ(stats.watermark_extent_clears, 1u);
  EXPECT_EQ(stats.watermark_active, 1u);

  for (int i = 0; i < 5; ++i)
    dfkv::DiskSlabStoreTestPeer::ReclaimNow(&store, {});
  stats = store.GetStats();
  EXPECT_EQ(store.Count(), 2u);
  EXPECT_EQ(stats.watermark_evictions, 5u);
  EXPECT_EQ(stats.watermark_extent_clears, 5u);
  EXPECT_EQ(stats.eviction_record_clears, 6u);
  EXPECT_EQ(stats.watermark_ticks, 5u);
  EXPECT_EQ(stats.watermark_active, 0u);
}

// EvictedBytes() backs dfkv_evicted_bytes_total. Watermark reclamation used to
// erase the victims' payload accounting without adding it to the counter, so
// the metric only reflected inline Put eviction, extent rebalance and
// class-local reclaim.
TEST_F(DiskSlabTest, WatermarkEvictionCountsEvictedBytes) {
  ::setenv("DFKV_SLAB_EVICT_HIGH_PCT", "50", 1);
  ::setenv("DFKV_SLAB_EVICT_LOW_PCT", "26", 1);
  ::setenv("DFKV_SLAB_EVICT_MAX_VICTIMS_PER_TICK", "8", 1);
  auto options = Opts(8 * 4096, 4096, 4096);
  options.reclaim_interval_ms = 0;
  options.table_sync_ms = 0;
  bool ok = false;
  DiskSlabStore store(options, &ok);
  ::unsetenv("DFKV_SLAB_EVICT_HIGH_PCT");
  ::unsetenv("DFKV_SLAB_EVICT_LOW_PCT");
  ::unsetenv("DFKV_SLAB_EVICT_MAX_VICTIMS_PER_TICK");
  ASSERT_TRUE(ok);

  const std::string value(4000, 'e');
  for (uint64_t i = 0; i < 8; ++i)
    ASSERT_EQ(store.Cache(K(i), value.data(), value.size()), Status::kOk);
  ASSERT_EQ(store.EvictedBytes(), 0u);

  dfkv::DiskSlabStoreTestPeer::ReclaimNow(&store, {});
  const auto stats = store.GetStats();
  // One tick drains to the low mark; class-local reclaim has nothing left to
  // do, so every eviction here is a watermark eviction.
  ASSERT_EQ(stats.watermark_evictions, 6u);
  ASSERT_EQ(stats.reclaimed_slots, 0u);
  EXPECT_EQ(store.Count(), 2u);
  EXPECT_EQ(store.Evictions(), 6u);
  EXPECT_EQ(store.EvictedBytes(), 6u * value.size());
}

TEST(NewPutDemandLatch, ExpiresOnlyAfter128CompletedInactiveQuietTicks) {
  NewPutDemandLatch latch;
  latch.BeginTick(90, 80, 40, 0);
  latch.EndTick(40, 40);
  ASSERT_FALSE(latch.active);
  for (int tick = 1; tick <= 127; ++tick) {
    latch.BeginTick(40, 80, 40, 0);
    latch.EndTick(40, 40);
  }
  // Still eligible: successful admission below high must resume draining.
  auto renewed = latch;
  renewed.BeginTick(50, 80, 40, 1);
  EXPECT_TRUE(renewed.active);
  latch.BeginTick(40, 80, 40, 0);
  latch.EndTick(40, 40);
  latch.BeginTick(50, 80, 40, 1);
  EXPECT_FALSE(latch.active);  // expired history cannot trigger a drain
}

TEST(NewPutDemandLatch, FreshHighAndNewAdmissionWinAtExpiryBoundary) {
  NewPutDemandLatch latch;
  latch.BeginTick(90, 80, 40, 0);
  latch.EndTick(40, 40);
  for (int tick = 0; tick < 127; ++tick) {
    latch.BeginTick(40, 80, 40, 0);
    latch.EndTick(40, 40);
  }
  auto high = latch;
  high.BeginTick(81, 80, 40, 0);
  high.EndTick(50, 40);
  EXPECT_TRUE(high.active);
  auto admitted = latch;
  admitted.BeginTick(50, 80, 40, 1);
  admitted.EndTick(40, 40);
  EXPECT_TRUE(admitted.active);
  admitted.BeginTick(40, 80, 40, 0);
  admitted.EndTick(40, 40);
  EXPECT_TRUE(admitted.active);  // previous interval still had demand
  admitted.BeginTick(40, 80, 40, 0);
  admitted.EndTick(40, 40);
  EXPECT_FALSE(admitted.active);
}

TEST(NewPutDemandLatch, QuietExpiryNeverCancelsRequiredActiveDrain) {
  NewPutDemandLatch latch;
  latch.BeginTick(90, 80, 40, 0);
  latch.EndTick(50, 40);
  for (int tick = 0; tick < 129; ++tick) {
    latch.BeginTick(50, 80, 40, 0);
    latch.EndTick(50, 40);
    EXPECT_TRUE(latch.active);
  }
  latch.EndTick(40, 40);
  EXPECT_FALSE(latch.active);
  latch.BeginTick(50, 80, 40, 1);
  EXPECT_FALSE(latch.active);
  latch.BeginTick(81, 80, 40, 0);
  EXPECT_TRUE(latch.active);  // a new high crossing works after expiry
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

// Reap-failure drain (#16): a hard io_uring_wait_cqe failure used to retire
// the ring while the kernel's accepted writes were still un-reaped -- and
// io_uring_queue_exit does NOT cancel in-flight requests, so a zombie write
// could land after the flush rolled the failed items' slots back and handed
// them to other keys. The failed batch must best-effort drain every accepted
// CQE BEFORE the ring is retired: then no zombie remains, the store must NOT
// fail closed, every item lands via the sequential rewrite, and the next
// batch re-inits the ring and fast-paths again.
TEST_F(DiskSlabTest, UringReapFailureDrainsBeforeReset) {
  auto opts = Opts(1 << 20, 1 << 20, 4096);
  opts.direct_writes = true;
  bool ok = false;
  DiskSlabStore s(opts, &ok);
  ASSERT_TRUE(ok);
  if (!s.DirectWritesActive()) GTEST_SKIP() << "fs rejected O_DIRECT (tmpfs)";
  // The first reap (the blocking wait) hard-fails; every reap afterwards is
  // the drain's non-blocking peek, behaving exactly like the real one.
  std::atomic<int> reap_calls{0};
  dfkv::DiskSlabStoreTestPeer::SetUringReapHook(
      &s, [&reap_calls](void* r, void* c) -> int {
        if (reap_calls.fetch_add(1) == 0) return -EIO;
        return io_uring_peek_cqe(static_cast<io_uring*>(r),
                                 static_cast<io_uring_cqe**>(c));
      });
  constexpr uint64_t N = 4;
  constexpr size_t kLen = 4096;
  std::vector<void*> bufs(N);
  std::vector<DiskSlabStore::CacheBatchItem> items;
  for (uint64_t i = 0; i < N; ++i) {
    ASSERT_EQ(posix_memalign(&bufs[i], 4096, kLen), 0);
    std::memset(bufs[i], static_cast<int>('A' + i), kLen);
    items.push_back({K(800 + i), static_cast<char*>(bufs[i]), kLen, kLen});
  }
  auto sts = s.CacheDirectBatch(items);
  if (reap_calls.load() == 0) {
    for (auto* b : bufs) free(b);
    GTEST_SKIP() << "uring path not taken (env-disabled or init fallback)";
  }
  for (auto st : sts) ASSERT_EQ(st, Status::kOk);
  // The failed wait left N accepted writes outstanding; the drain must have
  // reaped them (the real peek may also poll on -EAGAIN before completions).
  EXPECT_GE(reap_calls.load(), 1 + static_cast<int>(N))
      << "drain must reap what the failed wait left outstanding";
  EXPECT_TRUE(s.Healthy()) << "a successful drain is not a fail-closed event";
  EXPECT_EQ(s.GetStats().metadata_io_errors, 0u);
  {  // nothing was CQE-confirmed, so the failed batch tallies nothing uring
    const auto stats = s.GetStats();
    EXPECT_EQ(stats.uring_write_batches, 0u);
    EXPECT_EQ(stats.batched_writes, 0u);
  }
  // The sequential rewrite must be what readers observe (dangling-buffer and
  // torn-payload semantics would show up here or in the next store's data).
  for (uint64_t i = 0; i < N; ++i) {
    std::string out;
    ASSERT_EQ(s.Range(K(800 + i), 0, kLen, &out), Status::kOk) << i;
    EXPECT_EQ(out, std::string(kLen, static_cast<char>('A' + i))) << i;
  }

  // The retired ring re-inits on the next batch and fast-paths again.
  dfkv::DiskSlabStoreTestPeer::SetUringReapHook(&s, nullptr);
  std::vector<DiskSlabStore::CacheBatchItem> items2;
  for (uint64_t i = 0; i < N; ++i) {
    std::memset(bufs[i], static_cast<int>('a' + i), kLen);
    items2.push_back({K(900 + i), static_cast<char*>(bufs[i]), kLen, kLen});
  }
  sts = s.CacheDirectBatch(items2);
  for (auto st : sts) ASSERT_EQ(st, Status::kOk);
  {
    const auto stats = s.GetStats();
    EXPECT_EQ(stats.uring_write_batches, 1u) << "ring re-init did not happen";
    EXPECT_EQ(stats.batched_writes, N);
  }
  for (uint64_t i = 0; i < N; ++i) {
    std::string out;
    ASSERT_EQ(s.Range(K(900 + i), 0, kLen, &out), Status::kOk) << i;
    EXPECT_EQ(out, std::string(kLen, static_cast<char>('a' + i))) << i;
    // And the first batch's bytes are untouched by anything that followed.
    ASSERT_EQ(s.Range(K(800 + i), 0, kLen, &out), Status::kOk) << i;
    EXPECT_EQ(out, std::string(kLen, static_cast<char>('A' + i))) << i;
  }
  for (auto* b : bufs) free(b);
}

// Dropped-CQE fail-closed (#16): if the kernel cannot return the accepted
// completions at all, a zombie write may still land after the slot rollback,
// so the drain coming up short must fail the store closed rather than risk
// serving bytes a late landing overwrote.
TEST_F(DiskSlabTest, UringReapDrainMissFailsStoreClosed) {
  auto opts = Opts(1 << 20, 1 << 20, 4096);
  opts.direct_writes = true;
  bool ok = false;
  DiskSlabStore s(opts, &ok);
  ASSERT_TRUE(ok);
  if (!s.DirectWritesActive()) GTEST_SKIP() << "fs rejected O_DIRECT (tmpfs)";
  std::atomic<int> reap_calls{0};
  dfkv::DiskSlabStoreTestPeer::SetUringReapHook(
      &s, [&reap_calls](void*, void*) -> int {
        reap_calls.fetch_add(1);
        return -EIO;  // blocking wait AND drain peeks all fail
      });
  constexpr uint64_t N = 4;
  constexpr size_t kLen = 4096;
  std::vector<void*> bufs(N);
  std::vector<DiskSlabStore::CacheBatchItem> items;
  for (uint64_t i = 0; i < N; ++i) {
    ASSERT_EQ(posix_memalign(&bufs[i], 4096, kLen), 0);
    std::memset(bufs[i], static_cast<int>('C' + i), kLen);
    items.push_back({K(810 + i), static_cast<char*>(bufs[i]), kLen, kLen});
  }
  auto sts = s.CacheDirectBatch(items);
  if (reap_calls.load() == 0) {
    for (auto* b : bufs) free(b);
    GTEST_SKIP() << "uring path not taken (env-disabled or init fallback)";
  }
  for (auto st : sts) EXPECT_EQ(st, Status::kIOError);
  EXPECT_FALSE(s.Healthy());
  EXPECT_GE(s.GetStats().metadata_io_errors, 1u);
  EXPECT_EQ(s.Count(), 0u) << "failed items must roll their slots back";
  for (uint64_t i = 0; i < N; ++i) EXPECT_FALSE(s.IsCached(K(810 + i)));
  for (auto* b : bufs) free(b);
}
#endif  // DFKV_WITH_URING

// Retention persists in record bytes 44..51 and comes back on restart; the
// record stays CRC-valid, so a binary that ignores those bytes still serves it.
TEST_F(DiskSlabTest, RetentionPersistsAcrossRestart) {
  const uint32_t now = static_cast<uint32_t>(
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
  const std::string v(3000, 'r');
  {
    bool ok = false;
    DiskSlabStore s(Opts(1 << 20, 1 << 20, 4096), &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(s.RetainEnabled());
    ASSERT_EQ(s.Cache(K(7), v.data(), v.size()), Status::kOk);
    ASSERT_EQ(s.Cache(K(8), v.data(), v.size()), Status::kOk);
    EXPECT_EQ(s.Retain(K(7), 600, /*mode=*/0, now),
              static_cast<uint8_t>(dfkv::SlabAllocator::RetainCode::kProtectedNew));
    EXPECT_EQ(s.Retain(K(99), 600, 0, now),
              static_cast<uint8_t>(dfkv::SlabAllocator::RetainCode::kMiss));
    EXPECT_EQ(s.RetentionStats().protected_slots, 1u);
  }
  // Bytes 44..51 of K(7)'s record hold the deadline and TTL.
  {
    std::ifstream tbl(dir_ / "slots.tbl", std::ios::binary);
    ASSERT_TRUE(tbl.good());
    std::vector<char> all((std::istreambuf_iterator<char>(tbl)),
                          std::istreambuf_iterator<char>());
    bool found = false;
    for (size_t off = 0; off + 64 <= all.size(); off += 64) {
      const uint32_t ttl = dfkv::net::GetU32(all.data() + off + 48);
      if (ttl == 600) {
        EXPECT_GE(dfkv::net::GetU32(all.data() + off + 44), now + 600);
        found = true;
      }
    }
    EXPECT_TRUE(found);
  }
  bool ok = false;
  DiskSlabStore s2(Opts(1 << 20, 1 << 20, 4096), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(s2.Count(), 2u);
  EXPECT_EQ(s2.RetentionStats().protected_slots, 1u);
  std::string out;
  ASSERT_EQ(s2.Range(K(7), 0, v.size(), &out), Status::kOk);
  EXPECT_EQ(out, v);
  // REFRESH renews with the persisted TTL.
  EXPECT_EQ(s2.Retain(K(7), 0, /*mode=*/1, now + 10),
            static_cast<uint8_t>(dfkv::SlabAllocator::RetainCode::kRefreshed));
}

TEST_F(DiskSlabTest, RetentionDisabledByEnvAnswersUnsupported) {
  ::setenv("DFKV_RETAIN_ENABLE", "0", 1);
  bool ok = false;
  DiskSlabStore s(Opts(1 << 20, 1 << 20, 4096), &ok);
  ::unsetenv("DFKV_RETAIN_ENABLE");
  ASSERT_TRUE(ok);
  EXPECT_FALSE(s.RetainEnabled());
  const std::string v(100, 'x');
  ASSERT_EQ(s.Cache(K(1), v.data(), v.size()), Status::kOk);
  EXPECT_EQ(s.Retain(K(1), 60, 0, 1000), dfkv::StoreEngine::kRetainUnsupported);
}

// ---- Open cost: which extents are fallocated again at open ----------------
//
// In direct mode every open used to fallocate every extent file. That call
// allocates a fresh extent and, after an unclean exit, waits for direct writes
// the previous process left in flight. It may be left out only when neither
// job exists: the previous run closed clean and the file has all its blocks.
namespace {
constexpr uint64_t kOpenCostExtentBytes = 1 << 16;
constexpr uint32_t kOpenCostExtents = 8;

uint64_t AllocatedBytes(const fs::path& dir, uint32_t extent) {
  char name[32];
  std::snprintf(name, sizeof(name), "E%05u", extent);
  struct stat st {};
  if (::stat((dir / "extents" / name).c_str(), &st) != 0) return 0;
  return static_cast<uint64_t>(st.st_blocks) * 512;
}

int OpenExtent(const fs::path& dir, uint32_t extent) {
  char name[32];
  std::snprintf(name, sizeof(name), "E%05u", extent);
  return ::open((dir / "extents" / name).c_str(), O_RDWR);
}

bool EveryExtentAllocated(const fs::path& dir) {
  for (uint32_t e = 0; e < kOpenCostExtents; ++e)
    if (AllocatedBytes(dir, e) < kOpenCostExtentBytes) return false;
  return true;
}
}  // namespace

// Creates the store in direct mode, writes two values and closes it clean.
// Skips the test where the filesystem under the test directory cannot show
// the behaviour: no O_DIRECT, or no preallocation.
#define DFKV_OPEN_COST_PREPARE(options)                                       \
  do {                                                                        \
    bool prepared = false;                                                    \
    {                                                                         \
      DiskSlabStore first(options, &prepared);                                \
      ASSERT_TRUE(prepared) << first.StartupError();                          \
      if (!first.DirectWritesActive())                                        \
        GTEST_SKIP() << "the test directory does not take O_DIRECT";          \
      EXPECT_EQ(first.GetStats().open_fallocate_skipped, 0u)                  \
          << "a new extent has to be allocated";                              \
      ASSERT_TRUE(PutFilled(first, 1, 'a'));                                  \
      ASSERT_TRUE(PutFilled(first, 2, 'b'));                                  \
    }                                                                         \
    if (!EveryExtentAllocated(dir_))                                          \
      GTEST_SKIP() << "the test directory does not preallocate";              \
  } while (0)

TEST_F(DiskSlabTest, CleanReopenLeavesAllocatedExtentsAlone) {
  auto options = Opts(kOpenCostExtents * kOpenCostExtentBytes,
                      kOpenCostExtentBytes, 4096);
  options.direct_writes = true;
  DFKV_OPEN_COST_PREPARE(options);

  for (int round = 0; round < 2; ++round) {
    bool ok = false;
    DiskSlabStore reopened(options, &ok);
    ASSERT_TRUE(ok) << reopened.StartupError();
    EXPECT_EQ(reopened.GetStats().open_fallocate_skipped, kOpenCostExtents);
    EXPECT_EQ(reopened.Count(), 2u + static_cast<size_t>(round));
    std::string out;
    ASSERT_EQ(reopened.Range(K(1), 0, 4096, &out), Status::kOk);
    EXPECT_EQ(out, Payload('a'));
    ASSERT_EQ(reopened.Range(K(2), 0, 4096, &out), Status::kOk);
    EXPECT_EQ(out, Payload('b'));
    ASSERT_TRUE(PutFilled(reopened, 10 + round, 'c'));
  }
  EXPECT_TRUE(EveryExtentAllocated(dir_));
}

// After an unclean exit the fallocate is the barrier against the previous
// process's in-flight direct writes: it runs for every extent, whether the
// table is then kept or reset. The clean start after that skips again.
TEST_F(DiskSlabTest, UncleanReopenAllocatesEveryExtentAgain) {
  for (const WarmRestart policy : {WarmRestart::kAuto, WarmRestart::kOff}) {
    SCOPED_TRACE(policy == WarmRestart::kAuto ? "table kept" : "table reset");
    fs::remove_all(dir_);
    auto options = WithIdentity(Opts(kOpenCostExtents * kOpenCostExtentBytes,
                                     kOpenCostExtentBytes, 4096),
                                policy, 1, 7);
    options.direct_writes = true;
    DFKV_OPEN_COST_PREPARE(options);
    ASSERT_TRUE(CrashAfter(options, [](DiskSlabStore& store) {
      return PutFilled(store, 3, 'c');
    }));
    ASSERT_TRUE(EveryExtentAllocated(dir_));
    {
      bool ok = false;
      DiskSlabStore recovered(options, &ok);
      ASSERT_TRUE(ok) << recovered.StartupError();
      const auto stats = recovered.GetStats();
      EXPECT_EQ(stats.open_fallocate_skipped, 0u);
      if (policy == WarmRestart::kAuto) {
        EXPECT_EQ(stats.warm_restarts, 1u);
        EXPECT_EQ(recovered.Count(), 3u);
      } else {
        EXPECT_EQ(stats.unclean_resets, 1u);
        EXPECT_EQ(recovered.Count(), 0u);
      }
    }
    bool ok = false;
    DiskSlabStore clean_reopen(options, &ok);
    ASSERT_TRUE(ok) << clean_reopen.StartupError();
    EXPECT_EQ(clean_reopen.GetStats().open_fallocate_skipped, kOpenCostExtents);
  }
}

// The clean marker is what lets the next start leave the fallocate out, so
// Close() must not write it while a payload write is still in flight: that
// write could land after the next run has reused the slot. Such a close is
// an unclean exit to the next start, which then allocates (and so waits)
// again. A close with no writer in flight is clean as before.
TEST_F(DiskSlabTest, CloseWithAPayloadWriteInFlightIsNotClean) {
  for (const bool batch : {false, true}) {
    SCOPED_TRACE(batch ? "batch write" : "scalar write");
    fs::remove_all(dir_);
    auto options = WithIdentity(Opts(kOpenCostExtents * kOpenCostExtentBytes,
                                     kOpenCostExtentBytes, 4096),
                                WarmRestart::kAuto, 1, 7);
    options.direct_writes = true;
    DFKV_OPEN_COST_PREPARE(options);
    {
      bool ok = false;
      DiskSlabStore store(options, &ok);
      ASSERT_TRUE(ok) << store.StartupError();
      ASSERT_EQ(store.GetStats().open_fallocate_skipped, kOpenCostExtents);
      HookGate gate;
      dfkv::DiskSlabStoreTestPeer::SetPayloadWriteHook(
          &store, [&gate] { return gate.Run(); });
      std::string value = Payload('w');
      std::thread writer([&] {
        if (batch)
          store.CacheDirectBatch(
              {{K(50), value.data(), value.size(), value.size()}});
        else
          store.Cache(K(50), value.data(), value.size());
      });
      gate.WaitEntered();
      EXPECT_FALSE(store.Close()) << "clean with a write in flight";
      // The store still vouches for its table and its writers still run, so
      // a fatal signal must still take the token back. The handler's step
      // runs in a forked child: it disables tokens for the whole process.
      const fs::path token = TrustToken(dir_);
      ASSERT_TRUE(fs::exists(token)) << "the declined close dropped the token";
      const pid_t crashing = ::fork();
      ASSERT_GE(crashing, 0);
      if (crashing == 0) {
        DiskSlabStore::RevokeAllTrustForCrash();
        ::_exit(0);
      }
      int status = 0;
      ASSERT_EQ(::waitpid(crashing, &status, 0), crashing);
      EXPECT_FALSE(fs::exists(token))
          << "the crash handler was disarmed while writers ran";
      gate.Release(true);
      writer.join();
      dfkv::DiskSlabStoreTestPeer::SetPayloadWriteHook(&store, {});
    }
    {
      bool ok = false;
      DiskSlabStore next(options, &ok);
      ASSERT_TRUE(ok) << next.StartupError();
      const auto stats = next.GetStats();
      EXPECT_EQ(stats.warm_restarts + stats.unclean_resets, 1u)
          << "the next start took the clean path";
      EXPECT_EQ(stats.open_fallocate_skipped, 0u);
      EXPECT_TRUE(next.Close());
    }
    bool ok = false;
    DiskSlabStore after_clean(options, &ok);
    ASSERT_TRUE(ok) << after_clean.StartupError();
    const auto stats = after_clean.GetStats();
    EXPECT_EQ(stats.warm_restarts + stats.unclean_resets, 0u);
    EXPECT_EQ(stats.open_fallocate_skipped, kOpenCostExtents);
  }
}

// A clean close does not vouch for the files: an extent that lost blocks in
// the meantime (a hole punched, or cut to nothing and grown back) is
// allocated again, the others are left alone.
TEST_F(DiskSlabTest, CleanReopenAllocatesASparseExtentAgain) {
  auto options = Opts(kOpenCostExtents * kOpenCostExtentBytes,
                      kOpenCostExtentBytes, 4096);
  options.direct_writes = true;
  DFKV_OPEN_COST_PREPARE(options);

  const int punched = OpenExtent(dir_, 3);
  ASSERT_GE(punched, 0);
  const int punch_rc =
      ::fallocate(punched, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, 4096,
                  8192);
  ::close(punched);
  if (punch_rc != 0) GTEST_SKIP() << "the test directory cannot punch holes";
  const int regrown = OpenExtent(dir_, 5);
  ASSERT_GE(regrown, 0);
  ASSERT_EQ(::ftruncate(regrown, 0), 0);
  ASSERT_EQ(::ftruncate(regrown, static_cast<off_t>(kOpenCostExtentBytes)), 0);
  ::close(regrown);
  ASSERT_LT(AllocatedBytes(dir_, 3), kOpenCostExtentBytes);
  ASSERT_LT(AllocatedBytes(dir_, 5), kOpenCostExtentBytes);

  bool ok = false;
  DiskSlabStore reopened(options, &ok);
  ASSERT_TRUE(ok) << reopened.StartupError();
  EXPECT_EQ(reopened.GetStats().open_fallocate_skipped, kOpenCostExtents - 2);
  EXPECT_TRUE(EveryExtentAllocated(dir_));
  EXPECT_EQ(reopened.Count(), 2u);
}

// An extent cut short is, as before, a startup failure and not something the
// open repairs.
TEST_F(DiskSlabTest, CleanReopenStillRefusesAShortExtent) {
  auto options = Opts(kOpenCostExtents * kOpenCostExtentBytes,
                      kOpenCostExtentBytes, 4096);
  options.direct_writes = true;
  DFKV_OPEN_COST_PREPARE(options);
  const int cut = OpenExtent(dir_, 2);
  ASSERT_GE(cut, 0);
  ASSERT_EQ(::ftruncate(cut, static_cast<off_t>(kOpenCostExtentBytes / 2)), 0);
  ::close(cut);
  bool ok = true;
  DiskSlabStore reopened(options, &ok);
  EXPECT_FALSE(ok);
  EXPECT_NE(reopened.StartupError().find("extent size differs"),
            std::string::npos)
      << reopened.StartupError();
}

namespace {
// Bytes of slots.tbl that have blocks, and its size.
struct TableSpace {
  uint64_t allocated = 0;
  uint64_t size = 0;
  bool full() const { return allocated >= size; }
};
TableSpace TableSpaceOf(const fs::path& dir) {
  struct stat st {};
  if (::stat((dir / "slots.tbl").c_str(), &st) != 0) return {};
  return {static_cast<uint64_t>(st.st_blocks) * 512,
          static_cast<uint64_t>(st.st_size)};
}

// Whether a file under `dir` gets its blocks from fallocate: not every
// filesystem a build machine offers preallocates.
bool FilesystemPreallocates(const fs::path& dir) {
  fs::create_directories(dir.parent_path());
  const std::string path = dir.string() + ".prealloc_probe";
  const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) return false;
  struct stat st {};
  const bool ok = ::fallocate(fd, 0, 0, 1 << 20) == 0 &&
                  ::fstat(fd, &st) == 0 &&
                  static_cast<uint64_t>(st.st_blocks) * 512 >= (1u << 20);
  ::close(fd);
  ::unlink(path.c_str());
  return ok;
}

// A store as releases before the reservation left it: the table sized, its
// blocks allocated only where records were written.
int LeaveTableSparse(int, uint64_t) { return 0; }

// What a reopen restores: every key with its value, in key order.
std::vector<std::pair<uint64_t, std::string>> Restored(DiskSlabStore& store,
                                                       uint64_t first,
                                                       uint64_t last) {
  std::vector<std::pair<uint64_t, std::string>> out;
  for (uint64_t id = first; id <= last; ++id) {
    std::string value;
    if (store.Range(K(id), 0, 0, &value) == Status::kOk)
      out.emplace_back(id, std::move(value));
  }
  return out;
}
}  // namespace

// slots.tbl gets all of its blocks when the store is created, so that a
// record write never has to allocate one on a filesystem that filled up.
TEST_F(DiskSlabTest, SlotTableIsFullyAllocatedWhenCreated) {
  if (!FilesystemPreallocates(dir_))
    GTEST_SKIP() << "the test directory does not preallocate";
  auto options = Opts(1ull << 30, 1ull << 30, 4096);
  bool ok = false;
  DiskSlabStore store(options, &ok);
  ASSERT_TRUE(ok) << store.StartupError();
  const TableSpace space = TableSpaceOf(dir_);
  ASSERT_GT(space.size, 1u << 20);
  EXPECT_TRUE(space.full()) << space.allocated << " of " << space.size;
  // An empty table restores nothing, allocated or not.
  EXPECT_EQ(store.Count(), 0u);
  EXPECT_EQ(store.TableRebuilt(), 0u);
}

// A directory written by a release that did not reserve the table opens, is
// given the missing blocks, and restores exactly what it restored while the
// table was sparse.
TEST_F(DiskSlabTest, SparseSlotTableIsAllocatedOnReopenAndRestoresTheSame) {
  // 64 extents, of which the 39 values need a few: binding an extent writes
  // its whole part of the table, the rest of the table stays a hole.
  auto options = Opts(64ull << 20, 1 << 20, 4096);
  options.table_sync_ms = 0;
  options.reclaim_interval_ms = 0;
  auto earlier = options;
  earlier.table_allocate_for_test = LeaveTableSparse;
  std::vector<std::pair<uint64_t, std::string>> written;
  {
    bool ok = false;
    DiskSlabStore store(earlier, &ok);
    ASSERT_TRUE(ok) << store.StartupError();
    for (uint64_t id = 1; id <= 40; ++id)
      ASSERT_TRUE(PutFilled(store, id, static_cast<char>('a' + id % 26)));
    ASSERT_EQ(store.Remove(K(7)), Status::kOk);
    written = Restored(store, 1, 40);
    ASSERT_EQ(written.size(), 39u);
  }
  if (TableSpaceOf(dir_).full())
    GTEST_SKIP() << "the test directory does not keep a sized file sparse";

  // Reopened sparse, as the earlier release would.
  std::vector<std::pair<uint64_t, std::string>> sparse_restore;
  {
    bool ok = false;
    DiskSlabStore store(earlier, &ok);
    ASSERT_TRUE(ok) << store.StartupError();
    sparse_restore = Restored(store, 1, 40);
  }
  ASSERT_EQ(sparse_restore, written);
  ASSERT_FALSE(TableSpaceOf(dir_).full());

  for (int round = 0; round < 2; ++round) {  // allocating, then allocated
    bool ok = false;
    DiskSlabStore store(options, &ok);
    ASSERT_TRUE(ok) << store.StartupError();
    const TableSpace space = TableSpaceOf(dir_);
    EXPECT_TRUE(space.full()) << space.allocated << " of " << space.size;
    EXPECT_EQ(store.TableRebuilt(), 39u + round);  // round 0 added one
    EXPECT_EQ(store.Count(), 39u + round);
    const auto stats = store.GetStats();
    EXPECT_EQ(stats.rebuild_corrupt_records, 0u);
    EXPECT_EQ(stats.rebuild_rejected_records, 0u);
    EXPECT_EQ(Restored(store, 1, 40), written) << round;
    ASSERT_TRUE(PutFilled(store, 100 + round, 'z'));
  }
  // The size is what it was: an earlier release still opens the directory.
  EXPECT_EQ(TableSpaceOf(dir_).size, (64ull << 20) / 4096 * 64);
  bool ok = false;
  DiskSlabStore back(earlier, &ok);
  ASSERT_TRUE(ok) << back.StartupError();
  EXPECT_EQ(back.Count(), 41u);
}

// No room for the table is a startup failure that says so -- not a store that
// starts and fails for good at the first record it cannot place.
TEST_F(DiskSlabTest, NoRoomForTheSlotTableFailsStartup) {
  auto options = Opts(1 << 20, 1 << 20, 4096);
  auto full = options;
  full.table_allocate_for_test = [](int, uint64_t) {
    errno = ENOSPC;
    return -1;
  };
  {
    bool ok = true;
    DiskSlabStore store(full, &ok);
    EXPECT_FALSE(ok);
    EXPECT_FALSE(store.Healthy());
    const std::string& error = store.StartupError();
    EXPECT_NE(error.find("cannot reserve 16384 bytes for slots.tbl"),
              std::string::npos)
        << error;
    EXPECT_NE(error.find("errno 28"), std::string::npos) << error;
    EXPECT_NE(store.Cache(K(1), "x", 1), Status::kOk);
  }
  // Once there is room the same directory starts.
  {
    bool ok = false;
    DiskSlabStore store(options, &ok);
    ASSERT_TRUE(ok) << store.StartupError();
    ASSERT_TRUE(PutFilled(store, 1, 'a'));
  }
  // Only a table being created is refused: this one either has its blocks
  // (and is not allocated again) or is an existing table that stays as it is.
  bool ok = false;
  DiskSlabStore store(full, &ok);
  ASSERT_TRUE(ok) << store.StartupError();
  EXPECT_EQ(store.Count(), 1u);
}

// An existing directory of an earlier release on a filesystem with no room
// left for the rest of its table keeps starting: it served like that until
// now, and refusing it would take a node with a cache out of service on
// upgrade. The table stays as it was and is completed once there is room.
TEST_F(DiskSlabTest, NoRoomToCompleteASparseSlotTableStillOpensAndKeepsTheCache) {
  auto options = Opts(64ull << 20, 1 << 20, 4096);
  auto earlier = options;
  earlier.table_allocate_for_test = LeaveTableSparse;
  for (const int no_room : {ENOSPC, EDQUOT}) {
    SCOPED_TRACE(no_room);
    auto full = options;
    full.table_allocate_for_test = [no_room](int, uint64_t) {
      errno = no_room;
      return -1;
    };
    fs::remove_all(dir_);
    {
      bool ok = false;
      DiskSlabStore store(earlier, &ok);
      ASSERT_TRUE(ok) << store.StartupError();
      ASSERT_TRUE(PutFilled(store, 1, 'a'));
    }
    const TableSpace sparse = TableSpaceOf(dir_);
    if (sparse.full())
      GTEST_SKIP() << "the test directory does not keep a sized file sparse";
    {
      bool ok = false;
      DiskSlabStore store(full, &ok);
      ASSERT_TRUE(ok) << store.StartupError();
      EXPECT_TRUE(store.Healthy());
      EXPECT_EQ(store.Count(), 1u) << "the cache is kept";
      ASSERT_TRUE(PutFilled(store, 2, 'b')) << "and the store serves";
    }
    EXPECT_FALSE(TableSpaceOf(dir_).full());
    if (!FilesystemPreallocates(dir_)) continue;
    bool ok = false;
    DiskSlabStore store(options, &ok);
    ASSERT_TRUE(ok) << store.StartupError();
    EXPECT_TRUE(TableSpaceOf(dir_).full()) << "completed once there is room";
    EXPECT_EQ(store.Count(), 2u);
  }
}

// A filesystem that cannot preallocate keeps what it had: the store opens and
// the table's blocks are allocated as they are used.
TEST_F(DiskSlabTest, FilesystemWithoutPreallocationStillOpens) {
  auto options = Opts(1 << 20, 1 << 20, 4096);
  options.table_allocate_for_test = [](int, uint64_t) {
    errno = EOPNOTSUPP;
    return -1;
  };
  for (int round = 0; round < 2; ++round) {
    bool ok = false;
    DiskSlabStore store(options, &ok);
    ASSERT_TRUE(ok) << store.StartupError();
    EXPECT_EQ(store.Count(), static_cast<uint64_t>(round));
    ASSERT_TRUE(PutFilled(store, 1 + round, 'a'));
  }
}

// Not a correctness test: prints how long a clean reopen of a populated store
// takes. Set DFKV_SLAB_OPEN_BENCH_OBJECTS (for example 400000) to run it; it
// needs about 5 KiB of disk per object under the temporary directory. The
// store's own "opened in" log line splits each reopen into its phases.
TEST_F(DiskSlabTest, OpenCostBench) {
  const char* requested = std::getenv("DFKV_SLAB_OPEN_BENCH_OBJECTS");
  if (requested == nullptr || *requested == '\0')
    GTEST_SKIP() << "set DFKV_SLAB_OPEN_BENCH_OBJECTS to run";
  const uint64_t objects = std::strtoull(requested, nullptr, 10);
  ASSERT_GT(objects, 0u);
  constexpr uint64_t kExtent = 1 << 20;  // 256 objects of 4 KiB each
  const uint64_t extents = objects / 256 * 5 / 4 + 16;  // stay off the watermark
  struct rlimit limit {};
  ASSERT_EQ(::getrlimit(RLIMIT_NOFILE, &limit), 0);
  limit.rlim_cur = limit.rlim_max;
  ASSERT_EQ(::setrlimit(RLIMIT_NOFILE, &limit), 0);
  if (limit.rlim_cur < 2 * extents + 64)
    GTEST_SKIP() << "needs " << 2 * extents + 64 << " descriptors";
  auto options = Opts(extents * kExtent, kExtent, 4096);
  options.direct_writes = true;
  options.reclaim_interval_ms = 0;
  {
    bool ok = false;
    DiskSlabStore store(options, &ok);
    ASSERT_TRUE(ok) << store.StartupError();
    const std::string value = Payload('v');
    for (uint64_t i = 1; i <= objects; ++i)
      ASSERT_EQ(store.Cache(BlockKey{i * 0x9E3779B97F4A7C15ull, i, i % 7},
                            value.data(), value.size()),
                Status::kOk);
  }
  for (int round = 0; round < 4; ++round) {
    const auto started = std::chrono::steady_clock::now();
    bool ok = false;
    DiskSlabStore store(options, &ok);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - started)
                          .count();
    ASSERT_TRUE(ok) << store.StartupError();
    std::printf("OPEN_BENCH round=%d objects=%llu extents=%llu direct=%d "
                "reopen_ms=%.1f\n",
                round, static_cast<unsigned long long>(store.Count()),
                static_cast<unsigned long long>(extents),
                store.DirectWritesActive() ? 1 : 0, ms);
  }
}

// ---- Open and activate are two steps ---------------------------------------
//
// The open reads; Activate() marks the store in use (dirty state, identity,
// trust token, background threads) and admits operations. A process that is
// killed in between has changed nothing: the next start finds the directory
// as the previous run closed it, whatever DFKV_SLAB_WARM_RESTART says. The
// crashing process is the forked child of CrashAfter, as above.
namespace {
std::string FileBytes(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

// What a store leaves on disk for the next start to judge it by.
struct OnDisk {
  explicit OnDisk(const fs::path& dir)
      : state(FileBytes(dir / "slab_state")),
        table(FileBytes(dir / "slots.tbl")),
        token(fs::exists(TrustToken(dir))) {}
  bool operator==(const OnDisk& other) const {
    return state == other.state && table == other.table && token == other.token;
  }
  std::string state;
  std::string table;
  bool token;
};

DiskSlabStore::Options Deferred(DiskSlabStore::Options options) {
  options.defer_activation = true;
  return options;
}

const WarmRestart kEveryPolicy[] = {WarmRestart::kOff, WarmRestart::kAuto,
                                    WarmRestart::kBootOnly};
const char* PolicyName(WarmRestart policy) {
  return policy == WarmRestart::kOff    ? "warm restart off"
         : policy == WarmRestart::kAuto ? "warm restart auto"
                                        : "warm restart boot";
}
}  // namespace

// Two objects, closed clean: what every kill-point test starts from.
#define DFKV_CLEAN_STORE_WITH_TWO_OBJECTS(options)                            \
  do {                                                                        \
    fs::remove_all(dir_);                                                     \
    fs::remove_all(dir_.string() + "_tokens");                                \
    fs::create_directories(dir_.string() + "_tokens");                        \
    bool created = false;                                                     \
    DiskSlabStore first(options, &created);                                   \
    ASSERT_TRUE(created) << first.StartupError();                             \
    ASSERT_TRUE(PutFilled(first, 1, 'a'));                                    \
    ASSERT_TRUE(PutFilled(first, 2, 'b'));                                    \
  } while (0)

#define DFKV_EXPECT_BOTH_OBJECTS(store)                                       \
  do {                                                                        \
    std::string read_back;                                                    \
    EXPECT_EQ(store.Count(), 2u);                                             \
    ASSERT_EQ(store.Range(K(1), 0, 4096, &read_back), Status::kOk);           \
    EXPECT_EQ(read_back, Payload('a'));                                       \
    ASSERT_EQ(store.Range(K(2), 0, 4096, &read_back), Status::kOk);           \
    EXPECT_EQ(read_back, Payload('b'));                                       \
  } while (0)

// Kill point: after the open, before Activate(). Nothing on disk has changed
// and the next start is a clean one with the data, under every policy.
TEST_F(DiskSlabTest, KilledBetweenOpenAndActivateCostsNothing) {
  for (const WarmRestart policy : kEveryPolicy) {
    SCOPED_TRACE(PolicyName(policy));
    const auto options =
        WithIdentity(Opts(1 << 20, 1 << 20, 4096), policy, 1, 7);
    DFKV_CLEAN_STORE_WITH_TWO_OBJECTS(options);
    const OnDisk before(dir_);
    ASSERT_FALSE(before.token) << "a cleanly closed store keeps no token";

    ASSERT_TRUE(CrashAfter(Deferred(options), [](DiskSlabStore& store) {
      return store.Healthy() && !store.Activated() && store.Count() == 2;
    }));
    EXPECT_TRUE(OnDisk(dir_) == before)
        << "an open without activation changed the directory";

    bool ok = false;
    DiskSlabStore next(options, &ok);
    ASSERT_TRUE(ok) << next.StartupError();
    EXPECT_EQ(next.GetStats().unclean_resets, 0u);
    EXPECT_EQ(next.GetStats().warm_restarts, 0u);
    DFKV_EXPECT_BOTH_OBJECTS(next);
  }
}

// Kill point: after Activate(), with or without a write. This is the unclean
// exit it always was: reset when warm restart is off, kept otherwise.
TEST_F(DiskSlabTest, KilledAfterActivateIsAnUncleanExit) {
  for (const bool wrote : {false, true}) {
    for (const WarmRestart policy : kEveryPolicy) {
      SCOPED_TRACE(std::string(PolicyName(policy)) +
                   (wrote ? ", a write after activation" : ", no write"));
      const auto options =
          WithIdentity(Opts(1 << 20, 1 << 20, 4096), policy, 1, 7);
      DFKV_CLEAN_STORE_WITH_TWO_OBJECTS(options);
      const OnDisk before(dir_);
      ASSERT_TRUE(CrashAfter(Deferred(options), [wrote](DiskSlabStore& store) {
        if (!store.Activate() || !store.Activated()) return false;
        return !wrote || PutFilled(store, 3, 'c');
      }));
      const OnDisk after(dir_);
      EXPECT_NE(after.state, before.state) << "activation left no dirty state";
      EXPECT_TRUE(after.token) << "activation granted no token";

      bool ok = false;
      DiskSlabStore next(options, &ok);
      ASSERT_TRUE(ok) << next.StartupError();
      const auto stats = next.GetStats();
      if (policy == WarmRestart::kOff) {
        EXPECT_EQ(stats.unclean_resets, 1u);
        EXPECT_EQ(next.Count(), 0u);
      } else {
        EXPECT_EQ(stats.warm_restarts, 1u);
        EXPECT_EQ(stats.unclean_resets, 0u);
        EXPECT_EQ(next.Count(), wrote ? 3u : 2u);
      }
    }
  }
}

// A store found dirty is recovered or reset by its open, which rewrites the
// state and takes the previous run's token away. It is activated right there
// even when activation was to be deferred: left alone it would sit without a
// token of its own, and a second kill would cost a cache the first one did
// not.
TEST_F(DiskSlabTest, AStoreFoundDirtyIsActivatedByItsOpen) {
  for (const WarmRestart policy : kEveryPolicy) {
    SCOPED_TRACE(PolicyName(policy));
    const auto options =
        WithIdentity(Opts(1 << 20, 1 << 20, 4096), policy, 1, 7);
    DFKV_CLEAN_STORE_WITH_TWO_OBJECTS(options);
    ASSERT_TRUE(CrashAfter(options, [](DiskSlabStore&) { return true; }));
    // Killed again, in an open that asked for deferral and never activated.
    ASSERT_TRUE(CrashAfter(Deferred(options), [](DiskSlabStore& store) {
      return store.Healthy() && store.Activated();
    }));
    bool ok = false;
    DiskSlabStore next(Deferred(options), &ok);
    ASSERT_TRUE(ok) << next.StartupError();
    EXPECT_TRUE(next.Activated());
    const auto stats = next.GetStats();
    if (policy == WarmRestart::kOff) {
      EXPECT_EQ(stats.unclean_resets, 1u);
      EXPECT_EQ(next.Count(), 0u);
    } else {
      EXPECT_EQ(stats.warm_restarts, 1u) << next.UncleanResetReason();
      DFKV_EXPECT_BOTH_OBJECTS(next);
    }
  }
}

// Between open and activate the store answers questions about itself and
// nothing else: every operation is refused, no thread runs, and neither the
// state file, the table nor a token changes. Closing it leaves the directory
// as found and reports clean; the next start is a clean one.
TEST_F(DiskSlabTest, AnOpenedStoreRefusesEveryOperationAndWritesNothing) {
  auto options = Opts(1 << 20, 1 << 20, 4096);
  options.table_sync_ms = 1;        // the threads would run at once ...
  options.reclaim_interval_ms = 1;  // ... if they were started
  options.require_local_filesystem = false;
  DFKV_CLEAN_STORE_WITH_TWO_OBJECTS(options);
  const OnDisk before(dir_);
  {
    bool ok = false;
    DiskSlabStore store(Deferred(options), &ok);
    ASSERT_TRUE(ok) << store.StartupError();
    EXPECT_FALSE(store.Activated());
    EXPECT_TRUE(store.Healthy());
    EXPECT_EQ(store.Count(), 2u);
    EXPECT_EQ(store.GetStats().committed_objects, 2u);

    std::string value = Payload('x');
    std::string out;
    size_t got = 0;
    EXPECT_EQ(store.Cache(K(9), value.data(), value.size()), Status::kIOError);
    EXPECT_EQ(store.CacheDirect(K(9), value.data(), value.size(), value.size()),
              Status::kIOError);
    const auto batch = store.CacheDirectBatch(
        {{K(9), value.data(), value.size(), value.size()}});
    ASSERT_EQ(batch.size(), 1u);
    EXPECT_NE(batch[0], Status::kOk);
    EXPECT_EQ(store.Remove(K(1)), Status::kIOError);
    EXPECT_EQ(store.Retain(K(1), 60, 0, 1000),
              static_cast<uint8_t>(dfkv::SlabAllocator::RetainCode::kMiss));
    EXPECT_EQ(store.Range(K(1), 0, 4096, &out), Status::kIOError);
    EXPECT_EQ(store.RangeInto(K(1), 0, 4096, value.data(), value.size(), &got),
              Status::kIOError);
    EXPECT_FALSE(store.IsCached(K(1)));
    dfkv::DiskSlabStoreTestPeer::ReclaimNow(&store, {});
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    EXPECT_EQ(store.GetStats().record_writes, 0u);
    EXPECT_EQ(store.GetStats().table_syncs, 0u);
    EXPECT_TRUE(store.Healthy());
    EXPECT_TRUE(OnDisk(dir_) == before);
    EXPECT_TRUE(store.Close()) << "a store that never served is clean";
    EXPECT_TRUE(OnDisk(dir_) == before) << "Close changed what it found";
  }
  EXPECT_TRUE(OnDisk(dir_) == before);
  bool ok = false;
  DiskSlabStore next(options, &ok);
  ASSERT_TRUE(ok) << next.StartupError();
  EXPECT_EQ(next.GetStats().unclean_resets, 0u);
  EXPECT_EQ(next.GetStats().warm_restarts, 0u);
  DFKV_EXPECT_BOTH_OBJECTS(next);
}

// The refusal at every operation's entry is not the only thing between an
// opened store and its files: the functions that write check for themselves,
// and a write that gets that far fails the store for good without writing.
TEST_F(DiskSlabTest, AWriteBeforeActivationFailsTheStoreAndWritesNothing) {
  auto options = Opts(1 << 20, 1 << 20, 4096);
  options.table_sync_ms = 0;
  options.reclaim_interval_ms = 0;
  DFKV_CLEAN_STORE_WITH_TWO_OBJECTS(options);
  const OnDisk before(dir_);
  {
    bool ok = false;
    DiskSlabStore store(Deferred(options), &ok);
    ASSERT_TRUE(ok) << store.StartupError();
    // Past the entry checks, as a defect in a caller would be: allocate a
    // slot, write its payload and its record.
    EXPECT_FALSE(dfkv::DiskSlabStoreTestPeer::WriteCrashStage(
        &store, K(9), Payload('x'), /*include_record=*/true));
    EXPECT_FALSE(store.Healthy());
    EXPECT_FALSE(store.Activate()) << "a failed store was activated";
    EXPECT_FALSE(store.Close());
  }
  EXPECT_TRUE(OnDisk(dir_) == before);
  bool ok = false;
  DiskSlabStore next(options, &ok);
  ASSERT_TRUE(ok) << next.StartupError();
  EXPECT_EQ(next.GetStats().unclean_resets, 0u);
  DFKV_EXPECT_BOTH_OBJECTS(next);
}

// Activation: dirty state and token first, then operations; twice is once;
// the close after it is the ordinary clean close.
TEST_F(DiskSlabTest, ActivateMakesTheStoreServe) {
  const auto options =
      WithIdentity(Opts(1 << 20, 1 << 20, 4096), WarmRestart::kAuto, 1, 7);
  DFKV_CLEAN_STORE_WITH_TWO_OBJECTS(options);
  const OnDisk before(dir_);
  {
    bool ok = false;
    DiskSlabStore store(Deferred(options), &ok);
    ASSERT_TRUE(ok) << store.StartupError();
    ASSERT_TRUE(store.Activate());
    EXPECT_TRUE(store.Activated());
    const OnDisk active(dir_);
    EXPECT_NE(active.state, before.state);
    EXPECT_TRUE(active.token);
    ASSERT_TRUE(store.Activate());
    EXPECT_EQ(OnDisk(dir_).state, active.state) << "activated twice";
    DFKV_EXPECT_BOTH_OBJECTS(store);
    ASSERT_TRUE(PutFilled(store, 3, 'c'));
    EXPECT_TRUE(store.Close());
    EXPECT_FALSE(OnDisk(dir_).token);
  }
  bool ok = false;
  DiskSlabStore next(options, &ok);
  ASSERT_TRUE(ok) << next.StartupError();
  EXPECT_EQ(next.GetStats().unclean_resets, 0u);
  EXPECT_EQ(next.GetStats().warm_restarts, 0u);
  EXPECT_EQ(next.Count(), 3u);
}

// Kill point: inside the activation. The dirty state is one 64-byte record
// written and synced before anything else happens, so the directory is either
// still clean or already dirty. A state file that cannot be written leaves
// the store clean on disk, unhealthy in memory, and never serving.
TEST_F(DiskSlabTest, AnActivationThatCannotRecordTheDirtyStateDoesNotServe) {
  const auto options =
      WithIdentity(Opts(1 << 20, 1 << 20, 4096), WarmRestart::kOff, 1, 7);
  DFKV_CLEAN_STORE_WITH_TWO_OBJECTS(options);
  const OnDisk before(dir_);
  {
    bool ok = false;
    DiskSlabStore store(Deferred(options), &ok);
    ASSERT_TRUE(ok) << store.StartupError();
    dfkv::DiskSlabStoreTestPeer::BreakStateFd(&store);
    EXPECT_FALSE(store.Activate());
    EXPECT_FALSE(store.Activated());
    EXPECT_FALSE(store.Healthy());
    std::string value = Payload('x');
    EXPECT_EQ(store.Cache(K(9), value.data(), value.size()), Status::kIOError);
  }
  EXPECT_TRUE(OnDisk(dir_) == before);
  bool ok = false;
  DiskSlabStore next(options, &ok);
  ASSERT_TRUE(ok) << next.StartupError();
  EXPECT_EQ(next.GetStats().unclean_resets, 0u);
  DFKV_EXPECT_BOTH_OBJECTS(next);
}

// A directory that did not exist: the open initializes it, and without an
// activation it is left as a clean, empty store with no marker of an
// unfinished initialization.
TEST_F(DiskSlabTest, ANewDirectoryThatIsNeverActivatedIsACleanEmptyStore) {
  const auto options =
      WithIdentity(Opts(1 << 20, 1 << 20, 4096), WarmRestart::kOff, 1, 7);
  ASSERT_TRUE(CrashAfter(Deferred(options), [](DiskSlabStore& store) {
    return store.Healthy() && !store.Activated();
  }));
  EXPECT_FALSE(fs::exists(dir_ / "slab_initializing"));
  bool ok = false;
  DiskSlabStore next(options, &ok);
  ASSERT_TRUE(ok) << next.StartupError();
  EXPECT_EQ(next.GetStats().unclean_resets, 0u);
  EXPECT_EQ(next.GetStats().warm_restarts, 0u);
  EXPECT_EQ(next.Count(), 0u);
  ASSERT_TRUE(PutFilled(next, 1, 'a'));
}
