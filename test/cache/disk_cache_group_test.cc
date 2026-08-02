// TDD R7 — DiskCacheGroup: one cache node spanning multiple NVMe SSDs, mirroring
// dingo-cache's intra-node Ketama across --cache_dir=d1,d2,d3 (each disk an
// independent KVStore with its own LRU; total capacity split across disks).
#include "cache/disk_cache_group.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
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
        cv.wait_for(lock, std::chrono::seconds(2), [&] { return release; });
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
  bool release = false;
  std::vector<std::thread::id> threads;
  std::vector<size_t> calls_by_disk;
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

  std::vector<StoreEngine::CacheBatchItem> items;
  items.reserve(256);
  for (uint64_t i = 0; i < 256; ++i)
    items.push_back({BlockKey{0, i}, nullptr, 0, 0});
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
  std::vector<StoreEngine::CacheBatchItem> items{
      {BlockKey{0, 2}, nullptr, 0, 0},
      {BlockKey{0, 3}, nullptr, 0, 0}};
  const std::thread::id caller = std::this_thread::get_id();
  const auto statuses = group.CacheDirectBatch(items);
  ASSERT_EQ(statuses.size(), 2u);
  ASSERT_EQ(probe->threads.size(), 1u);
  EXPECT_EQ(probe->threads[0], caller);
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
