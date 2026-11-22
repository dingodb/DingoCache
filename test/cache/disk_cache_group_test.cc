// TDD R7 — DiskCacheGroup: one cache node spanning multiple NVMe SSDs, mirroring
// dingo-cache's intra-node Ketama across --cache_dir=d1,d2,d3 (each disk an
// independent KVStore with its own LRU; total capacity split across disks).
#include "cache/disk_cache_group.h"
#include "cache/disk_slab_store.h"

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT
namespace {
struct BatchProbe {
  void Enter(size_t disk) {
    std::unique_lock<std::mutex> lock(mu);
    ++calls;
    ++calls_by_disk[disk];
    ++active;
    max_active = std::max(max_active, active);
    threads.push_back(std::this_thread::get_id());
    if (wait_for_overlap) {
      if (active >= 2) {
        release = true;
        cv.notify_all();
      } else {
        cv.wait_for(lock, overlap_wait, [&] { return release; });
      }
    }
    --active;
  }
  std::mutex mu;
  std::condition_variable cv;
  size_t calls = 0;
  size_t active = 0;
  size_t max_active = 0;
  bool wait_for_overlap = true;
  std::chrono::milliseconds overlap_wait{2000};
  bool probe_scalar = false;  // also probe the scalar CacheDirect entry
  bool release = false;
  std::vector<std::thread::id> threads;
  std::vector<size_t> calls_by_disk;
  // Close(): each engine stays inside until `close_overlap` of them are.
  size_t close_calls = 0;
  size_t close_active = 0;
  size_t close_max_active = 0;
  size_t close_overlap = 0;
  std::set<size_t> unclean_disks;
  // Opening: a factory call stays inside until `open_overlap` of them are.
  void EnterOpen() {
    std::unique_lock<std::mutex> lock(mu);
    ++open_active;
    open_max_active = std::max(open_max_active, open_active);
    cv.notify_all();
    cv.wait_for(lock, std::chrono::seconds(10),
                [&] { return open_max_active >= open_overlap; });
    --open_active;
  }
  size_t open_active = 0;
  size_t open_max_active = 0;
  size_t open_overlap = 0;
};

class ScopedEnv {
 public:
  ScopedEnv(const char* name, const std::string& value) : name_(name) {
    if (const char* old = std::getenv(name)) {
      had_old_ = true;
      old_ = old;
    }
    ::setenv(name, value.c_str(), 1);
  }
  ~ScopedEnv() {
    if (had_old_)
      ::setenv(name_.c_str(), old_.c_str(), 1);
    else
      ::unsetenv(name_.c_str());
  }

 private:
  std::string name_;
  std::string old_;
  bool had_old_ = false;
};

class ProbeEngine final : public StoreEngine {
 public:
  ProbeEngine(std::string dir, size_t disk, std::shared_ptr<BatchProbe> probe)
      : dir_(std::move(dir)), disk_(disk), probe_(std::move(probe)) {}
  bool Healthy() const override { return true; }
  const std::string& StartupError() const override { return error_; }
  Status Cache(const BlockKey&, const void*, size_t) override {
    return Status::kOk;
  }
  Status CacheDirect(const BlockKey&, char*, size_t, size_t) override {
    if (probe_->probe_scalar) probe_->Enter(disk_);
    return Status::kOk;
  }
  std::vector<Status> CacheDirectBatch(
      const std::vector<CacheBatchItem>& items) override {
    probe_->Enter(disk_);
    std::vector<Status> out;
    out.reserve(items.size());
    for (const auto& item : items)
      out.push_back((item.key.digest_lo & 1) ? Status::kInvalid : Status::kOk);
    return out;
  }
  Status Range(const BlockKey&, uint64_t, uint64_t, std::string*,
               size_t*) override {
    return Status::kNotFound;
  }
  Status RangeInto(const BlockKey&, uint64_t, uint64_t, char*, size_t,
                   size_t*, size_t*) override {
    return Status::kNotFound;
  }
  Status RangeDirect(const BlockKey&, uint64_t, uint64_t, char*, size_t,
                     const char**, size_t*, size_t*) override {
    return Status::kNotFound;
  }
  Status RangeDirectPrep(const BlockKey&, uint64_t, uint64_t, size_t,
                         ReadLease*) override {
    return Status::kNotFound;
  }
  bool IsCached(const BlockKey&) const override { return false; }
  Status Lookup(const BlockKey&, ValueMetadata*) const override {
    return Status::kNotFound;
  }
  Status Remove(const BlockKey&) override { return Status::kNotFound; }
  uint64_t UsedBytes() const override { return 0; }
  size_t Count() const override { return 0; }
  uint64_t Evictions() const override { return 0; }
  uint64_t EvictedBytes() const override { return 0; }
  const std::string& Dir() const override { return dir_; }
  bool Close() override {
    std::unique_lock<std::mutex> lock(probe_->mu);
    ++probe_->close_calls;
    ++probe_->close_active;
    probe_->close_max_active =
        std::max(probe_->close_max_active, probe_->close_active);
    probe_->cv.notify_all();
    probe_->cv.wait_for(lock, std::chrono::seconds(10), [&] {
      return probe_->close_max_active >= probe_->close_overlap;
    });
    --probe_->close_active;
    return probe_->unclean_disks.count(disk_) == 0;
  }

 private:
  std::string dir_;
  size_t disk_ = 0;
  std::string error_;
  std::shared_ptr<BatchProbe> probe_;
};
}  // namespace

class DiskGroupTest : public ::testing::Test {
 protected:
  void SetUp() override {
    base_ = fs::temp_directory_path() /
            ("dfkv_dg_" + std::to_string(::testing::UnitTest::GetInstance()
                                             ->current_test_info()->line()));
    fs::remove_all(base_);
  }
  void TearDown() override { fs::remove_all(base_); }

  std::vector<std::string> Dirs(int n) {
    std::vector<std::string> v;
    for (int i = 0; i < n; ++i) {
      auto d = (base_ / ("disk" + std::to_string(i))).string();
      fs::create_directories(d);
      v.push_back(d);
    }
    return v;
  }
  size_t CountIn(const std::string& dir) {
    size_t c = 0;
    auto blocks = fs::path(dir) / "blocks";
    if (!fs::exists(blocks)) return 0;
    for (auto it = fs::recursive_directory_iterator(blocks);
         it != fs::recursive_directory_iterator(); ++it)
      if (it->is_regular_file()) ++c;
    return c;
  }
  fs::path base_;
};

TEST_F(DiskGroupTest, SpreadsBlocksAcrossAllDisks) {
  auto dirs = Dirs(3);
  DiskCacheGroup g({dirs, 1ull << 30, "file"});
  EXPECT_EQ(g.DiskCount(), 3u);
  std::string v(500, 'x');
  for (int i = 0; i < 300; ++i)
    ASSERT_EQ(g.Cache(BlockKey{(uint64_t)i, 0}, v.data(), v.size()), Status::kOk);
  for (auto& d : dirs) EXPECT_GT(CountIn(d), 0u) << d;  // every disk used
  EXPECT_EQ(g.Count(), 300u);
}

TEST_F(DiskGroupTest, PutGetRoundTripAcrossDisks) {
  auto dirs = Dirs(3);
  DiskCacheGroup g({dirs, 1ull << 30, "file"});
  for (int i = 0; i < 100; ++i) {
    std::string v = "val_" + std::to_string(i);
    ASSERT_EQ(g.Cache(BlockKey{(uint64_t)i, 0}, v.data(), v.size()), Status::kOk);
  }
  for (int i = 0; i < 100; ++i) {
    std::string out;
    ASSERT_EQ(g.Range(BlockKey{(uint64_t)i, 0}, 0, 64, &out), Status::kOk);
    EXPECT_EQ(out, "val_" + std::to_string(i));
  }
}

TEST_F(DiskGroupTest, DeterministicRoutingSameKeySameDisk) {
  auto dirs = Dirs(3);
  DiskCacheGroup g({dirs, 1ull << 30, "file"});
  std::string v(10, 'a');
  ASSERT_EQ(g.Cache(BlockKey{77, 0}, v.data(), v.size()), Status::kOk);
  ASSERT_EQ(g.Cache(BlockKey{77, 0}, v.data(), v.size()), Status::kOk);  // idempotent, same disk
  size_t total = 0;
  for (auto& d : dirs) total += CountIn(d);
  EXPECT_EQ(total, 1u);  // exactly one disk holds it, not duplicated
}

TEST_F(DiskGroupTest, MissReturnsNotFound) {
  auto dirs = Dirs(2);
  DiskCacheGroup g({dirs, 1ull << 30, "file"});
  std::string out;
  EXPECT_EQ(g.Range(BlockKey{404, 0}, 0, 8, &out), Status::kNotFound);
  EXPECT_FALSE(g.IsCached(BlockKey{404, 0}));
}

TEST_F(DiskGroupTest, PerDiskCapacityKeepsTotalBounded) {
  auto dirs = Dirs(3);
  // total cap split evenly -> ~ (cap/3) per disk; write far more than capacity
  const uint64_t cap = 3 * 20 * 1000;  // ~20 objs/disk of 1000B
  DiskCacheGroup g({dirs, cap, "file"});
  std::string v(1000, 'y');
  for (int i = 0; i < 500; ++i)
    ASSERT_EQ(g.Cache(BlockKey{(uint64_t)i, 0}, v.data(), v.size()), Status::kOk);
  EXPECT_LE(g.UsedBytes(), cap);  // LRU per disk keeps total under the cap
  EXPECT_GT(g.Count(), 0u);
}

TEST_F(DiskGroupTest, ReloadFromDisksRebuilds) {
  auto dirs = Dirs(3);
  {
    DiskCacheGroup g({dirs, 1ull << 30, "file"});
    std::string v = "persisted";
    ASSERT_EQ(g.Cache(BlockKey{55, 0}, v.data(), v.size()), Status::kOk);
  }
  DiskCacheGroup g2({dirs, 1ull << 30, "file"});  // same disks, fresh instance
  EXPECT_TRUE(g2.IsCached(BlockKey{55, 0}));
  std::string out;
  ASSERT_EQ(g2.Range(BlockKey{55, 0}, 0, 9, &out), Status::kOk);
  EXPECT_EQ(out, "persisted");
}

TEST_F(DiskGroupTest, ZeroLengthWritesNeverAllocateObjects) {
  auto dirs = Dirs(1);
  DiskCacheGroup group({dirs, 1ull << 30, "file"});
  ASSERT_TRUE(group.Healthy());
  char marker = 0;

  EXPECT_EQ(group.Cache(BlockKey{1, 0}, &marker, 0), Status::kInvalid);
  EXPECT_EQ(group.CacheDirect(BlockKey{2, 0}, &marker, 0, 1),
            Status::kInvalid);
  std::vector<StoreEngine::CacheBatchItem> items;
  for (uint64_t i = 0; i < 64; ++i)
    items.push_back({BlockKey{3 + i, 0}, &marker, 0, 1});
  const auto statuses = group.CacheDirectBatch(items);
  ASSERT_EQ(statuses.size(), items.size());
  for (Status status : statuses) EXPECT_EQ(status, Status::kInvalid);
  EXPECT_EQ(group.Count(), 0u);
  EXPECT_EQ(group.UsedBytes(), 0u);
}

TEST_F(DiskGroupTest, CacheDirectBatchOverlapsDisksAndPreservesOrder) {
  auto dirs = Dirs(3);
  auto probe = std::make_shared<BatchProbe>();
  probe->calls_by_disk.resize(dirs.size());
  DiskCacheGroup::Options options;
  options.cache_dirs = dirs;
  options.capacity_bytes = 1ull << 30;
  options.engine = "file";
  options.engine_factory =
      [probe, dirs](const std::string& dir, uint64_t) {
        const size_t disk =
            static_cast<size_t>(std::find(dirs.begin(), dirs.end(), dir) -
                                dirs.begin());
        return std::make_unique<ProbeEngine>(dir, disk, probe);
      };
  DiskCacheGroup group(std::move(options));
  ASSERT_TRUE(group.Healthy());

  char value = 'x';
  std::vector<StoreEngine::CacheBatchItem> items;
  items.reserve(256);
  for (uint64_t i = 0; i < 256; ++i)
    items.push_back({BlockKey{0, i}, &value, 1, 1});
  const std::vector<Status> statuses = group.CacheDirectBatch(items);

  ASSERT_EQ(statuses.size(), items.size());
  for (size_t i = 0; i < statuses.size(); ++i) {
    EXPECT_EQ(statuses[i], (i & 1) ? Status::kInvalid : Status::kOk)
        << "result must remain at original input index " << i;
  }
  EXPECT_EQ(probe->calls, group.DiskCount());
  for (size_t disk = 0; disk < probe->calls_by_disk.size(); ++disk)
    EXPECT_EQ(probe->calls_by_disk[disk], 1u) << "disk " << disk;
  EXPECT_GE(probe->max_active, 2u)
      << "at least two disk groups must execute concurrently";
}

TEST_F(DiskGroupTest, SingleDiskBatchRunsInline) {
  auto dirs = Dirs(1);
  auto probe = std::make_shared<BatchProbe>();
  probe->wait_for_overlap = false;
  probe->calls_by_disk.resize(1);
  DiskCacheGroup::Options options;
  options.cache_dirs = dirs;
  options.capacity_bytes = 1ull << 30;
  options.engine = "file";
  options.engine_factory =
      [probe](const std::string& dir, uint64_t) {
        return std::make_unique<ProbeEngine>(dir, 0, probe);
      };
  DiskCacheGroup group(std::move(options));
  char value = 'x';
  std::vector<StoreEngine::CacheBatchItem> items{
      {BlockKey{0, 2}, &value, 1, 1},
      {BlockKey{0, 3}, &value, 1, 1}};
  const std::thread::id caller = std::this_thread::get_id();
  const auto statuses = group.CacheDirectBatch(items);
  ASSERT_EQ(statuses.size(), 2u);
  ASSERT_EQ(probe->threads.size(), 1u);
  EXPECT_EQ(probe->threads[0], caller);
}

namespace {
// Two callers write one same-tenant item each to a single-disk group. The probe
// engine holds the first call inside the engine until a second one overlaps
// (or its wait expires) and records the peak overlap.
size_t SameTenantBatchOverlap(const std::string& dir,
                              std::chrono::milliseconds overlap_wait) {
  auto probe = std::make_shared<BatchProbe>();
  probe->calls_by_disk.resize(1);
  probe->overlap_wait = overlap_wait;
  DiskCacheGroup::Options options;
  options.cache_dirs = {dir};
  options.capacity_bytes = 1ull << 30;
  options.engine = "file";
  options.engine_factory = [probe](const std::string& path, uint64_t) {
    return std::make_unique<ProbeEngine>(path, 0, probe);
  };
  DiskCacheGroup group(std::move(options));
  EXPECT_TRUE(group.Healthy());
  char value = 'x';
  auto write = [&](uint64_t id) {
    std::vector<StoreEngine::CacheBatchItem> items{
        {BlockKey{0, id}, &value, 1, 1}};
    EXPECT_EQ(group.CacheDirectBatch(items)[0], Status::kOk);
  };
  std::thread first(write, 2), second(write, 4);
  first.join();
  second.join();
  EXPECT_EQ(probe->calls, 2u);
  return probe->max_active;
}
}  // namespace

// Without a quota there is no admission state to protect: same-tenant writes
// must overlap in the engine. They used to take the tenant stripe
// unconditionally and hold it across the storage write, so every RAM-tier
// flush batch of a single-tenant deployment ran one at a time.
TEST_F(DiskGroupTest, UnlimitedTenantWritesOverlap) {
  ScopedEnv default_quota("DFKV_TENANT_DEFAULT_QUOTA_BYTES", "0");
  ScopedEnv quota_file("DFKV_TENANT_QUOTAS_FILE", "");
  EXPECT_EQ(SameTenantBatchOverlap(Dirs(1)[0], std::chrono::seconds(2)), 2u);
}

// The scalar direct PUT entry takes the same early return.
TEST_F(DiskGroupTest, UnlimitedTenantScalarWritesOverlap) {
  ScopedEnv default_quota("DFKV_TENANT_DEFAULT_QUOTA_BYTES", "0");
  ScopedEnv quota_file("DFKV_TENANT_QUOTAS_FILE", "");
  auto probe = std::make_shared<BatchProbe>();
  probe->calls_by_disk.resize(1);
  probe->probe_scalar = true;
  DiskCacheGroup::Options options;
  options.cache_dirs = Dirs(1);
  options.capacity_bytes = 1ull << 30;
  options.engine = "file";
  options.engine_factory = [probe](const std::string& path, uint64_t) {
    return std::make_unique<ProbeEngine>(path, 0, probe);
  };
  DiskCacheGroup group(std::move(options));
  ASSERT_TRUE(group.Healthy());
  char value = 'x';
  auto write = [&](uint64_t id) {
    EXPECT_EQ(group.CacheDirect(BlockKey{0, id}, &value, 1, 1), Status::kOk);
  };
  std::thread first(write, 2), second(write, 4);
  first.join();
  second.join();
  EXPECT_EQ(probe->max_active, 2u);
}

// A limited tenant keeps strict serialization: quota admission and the write
// it admits stay under one stripe hold.
TEST_F(DiskGroupTest, LimitedTenantWritesStaySerialized) {
  ScopedEnv default_quota("DFKV_TENANT_DEFAULT_QUOTA_BYTES", "1048576");
  ScopedEnv quota_file("DFKV_TENANT_QUOTAS_FILE", "");
  EXPECT_EQ(
      SameTenantBatchOverlap(Dirs(1)[0], std::chrono::milliseconds(200)), 1u);
}

TEST_F(DiskGroupTest, QuotaAdmissionSupportsBatchAndRemovalReuse) {
  const auto dirs = Dirs(1);
  const fs::path quotas = base_ / "quotas";
  std::ofstream(quotas) << "# explicit tenant\n"
                        << "0000000000001111 10\n";
  ScopedEnv quota_file("DFKV_TENANT_QUOTAS_FILE", quotas.string());
  ScopedEnv default_quota("DFKV_TENANT_DEFAULT_QUOTA_BYTES", "10");
  DiskCacheGroup group({dirs, 1ull << 30, "file"});
  ASSERT_TRUE(group.Healthy()) << group.StartupError();

  void* raw = std::aligned_alloc(4096, 4096);
  ASSERT_NE(raw, nullptr);
  std::unique_ptr<void, decltype(&std::free)> buffer(raw, &std::free);
  std::memset(raw, 'q', 4096);
  std::vector<StoreEngine::CacheBatchItem> items{
      {BlockKey{1, 0, 0x1111}, static_cast<char*>(raw), 6, 4096},
      {BlockKey{2, 0, 0x1111}, static_cast<char*>(raw), 6, 4096},
      {BlockKey{3, 0, 0x2222}, static_cast<char*>(raw), 6, 4096}};
  const auto statuses = group.CacheDirectBatch(items);
  ASSERT_EQ(statuses.size(), 3u);
  EXPECT_EQ(statuses[0], Status::kOk);
  EXPECT_EQ(statuses[1], Status::kQuotaExceeded);
  EXPECT_EQ(statuses[2], Status::kOk);
  EXPECT_EQ(group.TenantUsedBytes(0x1111), 6u);
  EXPECT_EQ(group.TenantQuotaRejections(), 1u);

  // Existing keys never double-charge, and Remove makes capacity reusable
  // before it releases the same tenant stripe.
  EXPECT_EQ(group.Cache(BlockKey{1, 0, 0x1111}, "ignored", 7), Status::kOk);
  EXPECT_EQ(group.TenantUsedBytes(0x1111), 6u);
  ASSERT_EQ(group.Remove(BlockKey{1, 0, 0x1111}), Status::kOk);
  EXPECT_EQ(group.Cache(BlockKey{2, 0, 0x1111}, "1234567890", 10),
            Status::kOk);
  EXPECT_EQ(group.TenantUsedBytes(0x1111), 10u);

  const auto metrics = group.ConfiguredTenantQuotaMetrics();
  ASSERT_EQ(metrics.size(), 1u);
  EXPECT_EQ(metrics[0].tenant_hash, 0x1111u);
  EXPECT_EQ(metrics[0].limit_bytes, 10u);
  EXPECT_EQ(metrics[0].used_bytes, 10u);
  EXPECT_EQ(metrics[0].rejections, 1u);
}

TEST_F(DiskGroupTest, ConcurrentQuotaAdmissionCannotOversubscribe) {
  ScopedEnv default_quota("DFKV_TENANT_DEFAULT_QUOTA_BYTES", "10");
  ScopedEnv quota_file("DFKV_TENANT_QUOTAS_FILE", "");
  DiskCacheGroup group({Dirs(1), 1ull << 30, "file"});
  ASSERT_TRUE(group.Healthy()) << group.StartupError();
  std::array<Status, 2> statuses{Status::kInvalid, Status::kInvalid};
  std::thread first([&] {

    statuses[0] =
        group.Cache(BlockKey{1, 0, 0x3333}, "123456", 6);
  });
  std::thread second([&] {
    statuses[1] =
        group.Cache(BlockKey{2, 0, 0x3333}, "123456", 6);
  });
  first.join();
  second.join();
  EXPECT_EQ(std::count(statuses.begin(), statuses.end(), Status::kOk), 1);
  EXPECT_EQ(std::count(statuses.begin(), statuses.end(),
                       Status::kQuotaExceeded), 1);
  EXPECT_EQ(group.TenantUsedBytes(0x3333), 6u);
}
TEST_F(DiskGroupTest, StoreEvictionReleasesTenantQuotaImmediately) {
  ScopedEnv default_quota("DFKV_TENANT_DEFAULT_QUOTA_BYTES", "10");
  ScopedEnv quota_file("DFKV_TENANT_QUOTAS_FILE", "");
  DiskCacheGroup group({Dirs(1), 10, "file"});
  ASSERT_TRUE(group.Healthy()) << group.StartupError();
  ASSERT_EQ(group.Cache(BlockKey{1, 0, 0x1111}, "123456", 6),
            Status::kOk);
  ASSERT_EQ(group.Cache(BlockKey{2, 0, 0x2222}, "123456", 6),
            Status::kOk);
  EXPECT_EQ(group.TenantUsedBytes(0x1111), 0u);
  EXPECT_EQ(group.Cache(BlockKey{3, 0, 0x1111}, "1234567890", 10),
            Status::kOk);
  EXPECT_EQ(group.TenantUsedBytes(0x1111), 10u);
}

TEST_F(DiskGroupTest, ConfiguredQuotaFileFailsClosed) {
  const fs::path missing = base_ / "missing";
  {
    ScopedEnv quota_file("DFKV_TENANT_QUOTAS_FILE", missing.string());
    DiskCacheGroup group({Dirs(1), 1ull << 30, "file"});
    EXPECT_FALSE(group.Healthy());
    EXPECT_NE(group.StartupError().find("cannot open"), std::string::npos);
  }
  const fs::path malformed = base_ / "malformed";
  std::ofstream(malformed) << "ABCDEF0123456789 10\n";
  {
    ScopedEnv quota_file("DFKV_TENANT_QUOTAS_FILE", malformed.string());
    DiskCacheGroup group({Dirs(1), 1ull << 30, "file"});
    EXPECT_FALSE(group.Healthy());
    EXPECT_NE(group.StartupError().find("malformed"), std::string::npos);
  }
}

// Close() is what a stopping server waits for before it releases memory: the
// disks must not queue behind one another, and the caller has to learn which
// of them did not get clean.
TEST_F(DiskGroupTest, CloseRunsOnAllDisksAtOnceAndReportsEachOne) {
  const auto dirs = Dirs(4);
  auto probe = std::make_shared<BatchProbe>();
  probe->close_overlap = dirs.size();
  probe->unclean_disks = {2};
  DiskCacheGroup::Options options;
  options.cache_dirs = dirs;
  options.capacity_bytes = 1ull << 30;
  options.engine = "file";
  options.engine_factory = [probe, dirs](const std::string& dir, uint64_t) {
    const size_t disk = static_cast<size_t>(
        std::find(dirs.begin(), dirs.end(), dir) - dirs.begin());
    return std::make_unique<ProbeEngine>(dir, disk, probe);
  };
  {
    DiskCacheGroup group(std::move(options));
    ASSERT_TRUE(group.Healthy());
    const auto report = group.Close();
    ASSERT_EQ(report.size(), dirs.size());
    for (size_t i = 0; i < dirs.size(); ++i) {
      EXPECT_EQ(report[i].dir, dirs[i]);
      EXPECT_EQ(report[i].clean, i != 2) << i;
      // An engine leaves Close() only once all four are inside, or after its
      // 10 s wait: disks closed one after the other would each take that long.
      EXPECT_LT(report[i].seconds, 5.0) << i;
    }
    {
      std::lock_guard<std::mutex> lock(probe->mu);
      EXPECT_EQ(probe->close_max_active, dirs.size());
      EXPECT_EQ(probe->close_calls, dirs.size());
    }
    EXPECT_TRUE(group.Close().empty());  // nothing left to close
  }
  std::lock_guard<std::mutex> lock(probe->mu);
  EXPECT_EQ(probe->close_calls, dirs.size());  // the destructor did not repeat it
}

TEST_F(DiskGroupTest, DestructionClosesDisksThatNobodyClosed) {
  const auto dirs = Dirs(3);
  auto probe = std::make_shared<BatchProbe>();
  probe->close_overlap = dirs.size();
  DiskCacheGroup::Options options;
  options.cache_dirs = dirs;
  options.capacity_bytes = 1ull << 30;
  options.engine = "file";
  options.engine_factory = [probe](const std::string& dir, uint64_t) {
    return std::make_unique<ProbeEngine>(dir, 0, probe);
  };
  { DiskCacheGroup group(std::move(options)); }
  std::lock_guard<std::mutex> lock(probe->mu);
  EXPECT_EQ(probe->close_calls, dirs.size());
  EXPECT_EQ(probe->close_max_active, dirs.size());
}


// A slab store reads its whole table back when it is opened. With the disks
// opened one after the other, startup grew by seconds per disk.
TEST_F(DiskGroupTest, OpensAllDisksAtOnceAndKeepsTheirOrder) {
  const auto dirs = Dirs(4);
  auto probe = std::make_shared<BatchProbe>();
  probe->open_overlap = dirs.size();
  DiskCacheGroup::Options options;
  options.cache_dirs = dirs;
  options.capacity_bytes = 1ull << 30;
  options.engine = "file";
  options.engine_factory = [probe, dirs](const std::string& dir, uint64_t) {
    const size_t disk = static_cast<size_t>(
        std::find(dirs.begin(), dirs.end(), dir) - dirs.begin());
    // A factory call returns only once all four are inside, or after 10 s.
    probe->EnterOpen();
    // The last directory finishes first: the group must not be assembled in
    // completion order.
    std::this_thread::sleep_for(
        std::chrono::milliseconds(30 * (dirs.size() - 1 - disk)));
    return std::make_unique<ProbeEngine>(dir, disk, probe);
  };
  DiskCacheGroup group(std::move(options));
  ASSERT_TRUE(group.Healthy());
  ASSERT_EQ(group.DiskCount(), dirs.size());
  for (size_t i = 0; i < dirs.size(); ++i) EXPECT_EQ(group.DiskPath(i), dirs[i]);
  std::lock_guard<std::mutex> lock(probe->mu);
  EXPECT_EQ(probe->open_max_active, dirs.size());
}

// What an engine throws while it opens is a startup failure of that disk, as
// if the store had come up unhealthy. It is not rethrown: the other disks are
// open (and marked dirty) by then, and only an orderly teardown leaves them
// clean.
TEST_F(DiskGroupTest, AnEngineThatThrowsWhileOpeningIsAStartupFailure) {
  const auto dirs = Dirs(3);
  for (const std::vector<size_t>& throwers :
       {std::vector<size_t>{1}, std::vector<size_t>{0},
        std::vector<size_t>{2, 1}}) {
    auto probe = std::make_shared<BatchProbe>();
    probe->open_overlap = dirs.size();
    DiskCacheGroup::Options options;
    options.cache_dirs = dirs;
    options.capacity_bytes = 1ull << 30;
    options.engine = "file";
    options.engine_factory =
        [probe, dirs, throwers](const std::string& dir,
                                uint64_t) -> std::unique_ptr<StoreEngine> {
      const size_t disk = static_cast<size_t>(
          std::find(dirs.begin(), dirs.end(), dir) - dirs.begin());
      probe->EnterOpen();  // the throw happens while the others are opening
      if (std::find(throwers.begin(), throwers.end(), disk) != throwers.end())
        throw std::runtime_error("cannot open disk " + std::to_string(disk));
      return std::make_unique<ProbeEngine>(dir, disk, probe);
    };
    DiskCacheGroup group(std::move(options));  // does not throw
    EXPECT_FALSE(group.Healthy());
    // The first failing directory in configured order is the one reported.
    const size_t first = *std::min_element(throwers.begin(), throwers.end());
    EXPECT_EQ(group.StartupError(),
              dirs[first] + ": cannot open disk " + std::to_string(first));
  }
}

// The reason a throw is not rethrown: the stores that did open must be closed
// in order. Real slab stores with data; one directory's engine throws; the
// next start must find the others clean and complete, whichever disk failed.
TEST_F(DiskGroupTest, ADiskThatThrowsDoesNotCostTheOthersTheirCache) {
  const auto dirs = Dirs(3);
  const uint64_t kCapacity = 3ull << 30;
  const std::string value(4096, 't');
  {
    DiskCacheGroup group({dirs, kCapacity, "slab"});
    ASSERT_TRUE(group.Healthy()) << group.StartupError();
    for (uint64_t i = 0; i < 64; ++i)
      ASSERT_EQ(group.Cache(BlockKey{9100 + i, 0}, value.data(), value.size()),
                Status::kOk);
  }
  for (size_t thrower = 0; thrower < dirs.size(); ++thrower) {
    {
      DiskCacheGroup::Options options;
      options.cache_dirs = dirs;
      options.capacity_bytes = kCapacity;
      options.engine = "slab";
      options.engine_factory =
          [&dirs, thrower](const std::string& dir,
                           uint64_t capacity) -> std::unique_ptr<StoreEngine> {
        if (dir == dirs[thrower]) {
          // Give the other disks time to open and mark themselves dirty.
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
          throw std::runtime_error("cannot open");
        }
        DiskSlabStore::Options so;
        so.dir = dir;
        so.capacity_bytes = capacity;
        return std::make_unique<DiskSlabStore>(so);
      };
      DiskCacheGroup group(std::move(options));
      EXPECT_FALSE(group.Healthy());
    }
    DiskCacheGroup reopened({dirs, kCapacity, "slab"});
    ASSERT_TRUE(reopened.Healthy()) << reopened.StartupError();
    EXPECT_EQ(reopened.SlabStats().unclean_resets, 0u) << "thrower " << thrower;
    EXPECT_EQ(reopened.Count(), 64u) << "thrower " << thrower;
  }
}

// Two names for one directory: whichever store gets the directory lock, the
// other fails, and the node refuses to start. Decided by a race now, not by
// order, so only the outcome is asserted.
TEST_F(DiskGroupTest, OneDirectoryUnderTwoNamesStillRefusesToStart) {
  const auto dirs = Dirs(1);
  const std::string alias = (base_ / "alias").string();
  fs::create_directory_symlink(dirs[0], alias);
  DiskCacheGroup group({{dirs[0], alias}, 2ull << 30, "slab"});
  EXPECT_FALSE(group.Healthy());
  EXPECT_NE(group.StartupError().find("in use"), std::string::npos)
      << group.StartupError();
}

TEST_F(DiskGroupTest, RealSlabDisksOpenedTogetherKeepTheirData) {
  const auto dirs = Dirs(3);
  const std::string value(4096, 'p');
  {
    DiskCacheGroup group({dirs, 3ull << 30, "slab"});
    ASSERT_TRUE(group.Healthy()) << group.StartupError();
    for (uint64_t i = 0; i < 64; ++i)
      ASSERT_EQ(group.Cache(BlockKey{9000 + i, 0}, value.data(), value.size()),
                Status::kOk);
  }
  DiskCacheGroup reopened({dirs, 3ull << 30, "slab"});
  ASSERT_TRUE(reopened.Healthy()) << reopened.StartupError();
  EXPECT_EQ(reopened.Count(), 64u);
  for (size_t i = 0; i < dirs.size(); ++i) EXPECT_EQ(reopened.DiskPath(i), dirs[i]);
  for (uint64_t i = 0; i < 64; ++i) {
    std::string out;
    ASSERT_EQ(reopened.Range(BlockKey{9000 + i, 0}, 0, value.size(), &out),
              Status::kOk)
        << i;
    EXPECT_EQ(out, value);
  }
}

// ---- Disks are opened first and activated together -------------------------
namespace {
std::string StateFile(const std::string& dir) {
  std::ifstream in(fs::path(dir) / "slab_state", std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}
}  // namespace

// A group told to defer activation opens its disks and changes none of them:
// every state file is as the previous run left it and no operation is taken.
// Activate() marks all of them in use; after that the group serves.
TEST_F(DiskGroupTest, DisksAreOpenedFirstAndActivatedTogether) {
  const auto dirs = Dirs(2);
  const uint64_t kCapacity = 2ull << 30;
  const std::string value(4096, 'a');
  {
    DiskCacheGroup group({dirs, kCapacity, "slab"});
    ASSERT_TRUE(group.Healthy()) << group.StartupError();
    for (uint64_t i = 0; i < 32; ++i)
      ASSERT_EQ(group.Cache(BlockKey{9300 + i, 0}, value.data(), value.size()),
                Status::kOk);
  }
  const std::vector<std::string> clean{StateFile(dirs[0]), StateFile(dirs[1])};
  {
    DiskCacheGroup::Options options{dirs, kCapacity, "slab"};
    options.defer_activation = true;
    DiskCacheGroup group(std::move(options));
    ASSERT_TRUE(group.Healthy()) << group.StartupError();
    EXPECT_EQ(group.Count(), 32u);
    EXPECT_EQ(StateFile(dirs[0]), clean[0]);
    EXPECT_EQ(StateFile(dirs[1]), clean[1]);
    EXPECT_EQ(group.Cache(BlockKey{9400, 0}, value.data(), value.size()),
              Status::kIOError)
        << "a disk that is not activated took a write";
    EXPECT_EQ(StateFile(dirs[0]), clean[0]);
    EXPECT_EQ(StateFile(dirs[1]), clean[1]);

    ASSERT_TRUE(group.Activate());
    EXPECT_NE(StateFile(dirs[0]), clean[0]);
    EXPECT_NE(StateFile(dirs[1]), clean[1]);
    EXPECT_TRUE(group.Activate());
    for (uint64_t i = 0; i < 8; ++i)
      ASSERT_EQ(group.Cache(BlockKey{9400 + i, 0}, value.data(), value.size()),
                Status::kOk);
  }
  DiskCacheGroup reopened({dirs, kCapacity, "slab"});
  ASSERT_TRUE(reopened.Healthy()) << reopened.StartupError();
  EXPECT_EQ(reopened.SlabStats().unclean_resets, 0u);
  EXPECT_EQ(reopened.Count(), 40u);
}

// A group that defers and is never activated (a cancelled or failed start)
// closes its disks as it found them, and says they are clean.
TEST_F(DiskGroupTest, AGroupThatIsNeverActivatedLeavesItsDisksAsFound) {
  const auto dirs = Dirs(2);
  const uint64_t kCapacity = 2ull << 30;
  const std::string value(4096, 'b');
  {
    DiskCacheGroup group({dirs, kCapacity, "slab"});
    ASSERT_TRUE(group.Healthy()) << group.StartupError();
    for (uint64_t i = 0; i < 32; ++i)
      ASSERT_EQ(group.Cache(BlockKey{9500 + i, 0}, value.data(), value.size()),
                Status::kOk);
  }
  const std::vector<std::string> clean{StateFile(dirs[0]), StateFile(dirs[1])};
  {
    DiskCacheGroup::Options options{dirs, kCapacity, "slab"};
    options.defer_activation = true;
    DiskCacheGroup group(std::move(options));
    ASSERT_TRUE(group.Healthy()) << group.StartupError();
    const auto report = group.Close();
    ASSERT_EQ(report.size(), 2u);
    EXPECT_TRUE(report[0].clean);
    EXPECT_TRUE(report[1].clean);
    EXPECT_FALSE(group.Activate()) << "a closed group was activated";
  }
  EXPECT_EQ(StateFile(dirs[0]), clean[0]);
  EXPECT_EQ(StateFile(dirs[1]), clean[1]);
}

// With one disk that fails to open, a group does not activate the others,
// deferred or not: their state files stay as found.
TEST_F(DiskGroupTest, ADiskThatFailsToOpenKeepsTheOthersFromBeingActivated) {
  const auto dirs = Dirs(3);
  const uint64_t kCapacity = 3ull << 30;
  const std::string value(4096, 'c');
  {
    DiskCacheGroup group({dirs, kCapacity, "slab"});
    ASSERT_TRUE(group.Healthy()) << group.StartupError();
    for (uint64_t i = 0; i < 48; ++i)
      ASSERT_EQ(group.Cache(BlockKey{9600 + i, 0}, value.data(), value.size()),
                Status::kOk);
  }
  const std::vector<std::string> clean{StateFile(dirs[0]), StateFile(dirs[1]),
                                       StateFile(dirs[2])};
  // The third directory no longer holds what its meta file says: the open
  // of that disk fails, the group is unhealthy.
  fs::remove(fs::path(dirs[2]) / "slots.tbl");
  for (const bool defer : {false, true}) {
    SCOPED_TRACE(defer ? "deferred" : "activated by the constructor");
    DiskCacheGroup::Options options{dirs, kCapacity, "slab"};
    options.defer_activation = defer;
    DiskCacheGroup group(std::move(options));
    EXPECT_FALSE(group.Healthy());
    EXPECT_FALSE(group.Activate());
    EXPECT_EQ(StateFile(dirs[0]), clean[0]);
    EXPECT_EQ(StateFile(dirs[1]), clean[1]);
    EXPECT_EQ(StateFile(dirs[2]), clean[2]);
  }
}

namespace dfkv {
// Named like the store's test peer (its friend): breaks the state file
// descriptor, so the next write of slab_state fails.
class DiskSlabStoreTestPeer {
 public:
  static void BreakStateFd(DiskSlabStore* store) {
    if (store->state_fd_ >= 0) ::close(store->state_fd_);
    store->state_fd_ = -1;
  }
};
}  // namespace dfkv

namespace {
// A slab store that opens like any other and whose activation fails where a
// failing disk would make it fail: writing the dirty state.
class CannotBeActivated : public dfkv::DiskSlabStore {
 public:
  using dfkv::DiskSlabStore::DiskSlabStore;
  bool Activate() override {
    dfkv::DiskSlabStoreTestPeer::BreakStateFd(this);
    return dfkv::DiskSlabStore::Activate();
  }
};
}  // namespace

// The activation of the second of three disks fails. The first is in use by
// then, the third is never reached; the group is unhealthy. Closing it closes
// the first the ordinary way (clean), and leaves the other two with nothing
// changed. The next start finds all three clean with their data.
TEST_F(DiskGroupTest, AFailedActivationStopsAtThatDisk) {
  const auto dirs = Dirs(3);
  const uint64_t kCapacity = 3ull << 30;
  const std::string value(4096, 'd');
  auto slab_options = [](const std::string& dir, uint64_t capacity) {
    DiskSlabStore::Options so;
    so.dir = dir;
    so.capacity_bytes = capacity;
    so.defer_activation = true;
    return so;
  };
  {
    DiskCacheGroup::Options options{dirs, kCapacity, "slab"};
    options.engine_factory =
        [&](const std::string& dir,
            uint64_t capacity) -> std::unique_ptr<StoreEngine> {
      return std::make_unique<DiskSlabStore>(slab_options(dir, capacity));
    };
    DiskCacheGroup group(std::move(options));
    ASSERT_TRUE(group.Healthy()) << group.StartupError();
    for (uint64_t i = 0; i < 48; ++i)
      ASSERT_EQ(group.Cache(BlockKey{9700 + i, 0}, value.data(), value.size()),
                Status::kOk);
  }
  const std::vector<std::string> clean{StateFile(dirs[0]), StateFile(dirs[1]),
                                       StateFile(dirs[2])};
  {
    DiskCacheGroup::Options options{dirs, kCapacity, "slab"};
    options.defer_activation = true;
    options.engine_factory =
        [&](const std::string& dir,
            uint64_t capacity) -> std::unique_ptr<StoreEngine> {
      if (dir == dirs[1])
        return std::make_unique<CannotBeActivated>(slab_options(dir, capacity));
      return std::make_unique<DiskSlabStore>(slab_options(dir, capacity));
    };
    DiskCacheGroup group(std::move(options));
    ASSERT_TRUE(group.Healthy()) << group.StartupError();
    EXPECT_FALSE(group.Activate());
    EXPECT_FALSE(group.Healthy());
    EXPECT_NE(group.StartupError().find(dirs[1]), std::string::npos)
        << group.StartupError();
    EXPECT_FALSE(group.Activate()) << "an unhealthy group was activated";
    EXPECT_NE(StateFile(dirs[0]), clean[0]) << "the first disk is in use";
    EXPECT_EQ(StateFile(dirs[1]), clean[1]) << "the failed disk was changed";
    EXPECT_EQ(StateFile(dirs[2]), clean[2]) << "the third disk was reached";

    const auto report = group.Close();
    ASSERT_EQ(report.size(), 3u);
    EXPECT_TRUE(report[0].clean) << "the activated disk closes clean";
    EXPECT_FALSE(report[1].clean) << "the failed disk does not claim to be";
    EXPECT_TRUE(report[2].clean) << "the untouched disk is as it was";
  }
  EXPECT_NE(StateFile(dirs[0]), clean[0]) << "a clean close of a new epoch";
  EXPECT_EQ(StateFile(dirs[1]), clean[1]);
  EXPECT_EQ(StateFile(dirs[2]), clean[2]);
  DiskCacheGroup reopened({dirs, kCapacity, "slab"});
  ASSERT_TRUE(reopened.Healthy()) << reopened.StartupError();
  EXPECT_EQ(reopened.SlabStats().unclean_resets, 0u);
  EXPECT_EQ(reopened.SlabStats().warm_restarts, 0u);
  EXPECT_EQ(reopened.Count(), 48u);
}
