// TDD R7 — DiskCacheGroup: one cache node spanning multiple NVMe SSDs, mirroring
// dingo-cache's intra-node Ketama across --cache_dir=d1,d2,d3 (each disk an
// independent KVStore with its own LRU; total capacity split across disks).
#include "cache/disk_cache_group.h"
#include "cache/disk_slab_store.h"
#include "cache/log_store.h"
#include "cache/lse_io.h"

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
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
  void ForEachKey(const std::function<void(const BlockKey&)>&) const override {}
  bool ResetBeforeActivation() override { return true; }
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
  // `false` fails the next slots.tbl write: the store fails for good.
  static void SetRecordWriteHook(DiskSlabStore* store,
                                 std::function<bool()> hook) {
    store->record_write_hook_for_test_ = std::move(hook);
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

// The periodic table sync runs by default only where a recovery can depend
// on it: DFKV_SLAB_WARM_RESTART=boot. Elsewhere the table is synced by a
// clean stop (and extent rebinds) only.
TEST_F(DiskGroupTest, PeriodicTableSyncRunsOnlyWhereARecoveryUsesIt) {
  const std::string value(4096, 's');
  for (const char* mode : {"off", "auto", "boot"}) {
    fs::remove_all(base_);
    const auto dirs = Dirs(1);
    ::setenv("DFKV_SLAB_WARM_RESTART", mode, 1);
    ::unsetenv("DFKV_SLAB_TABLE_SYNC_MS");
    const auto opened = std::chrono::steady_clock::now();
    DiskCacheGroup group({dirs, 1ull << 30, "slab"});
    ::unsetenv("DFKV_SLAB_WARM_RESTART");
    ASSERT_TRUE(group.Healthy()) << group.StartupError();
    for (int round = 0; round < 5; ++round) {
      for (uint64_t i = 0; i < 8; ++i)
        ASSERT_EQ(group.Cache(BlockKey{9500 + round * 8 + i, 0}, value.data(),
                              value.size()),
                  Status::kOk);
      std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
    const std::string m(mode);
    const uint64_t syncs = group.SlabStats().table_syncs;
    if (m == "boot") {
      EXPECT_GT(syncs, 0u) << "boot syncs every 100 ms";
    } else {
      EXPECT_EQ(syncs, 0u) << mode << ": synced at the 100 ms cadence";
    }
    if (m == "auto") {
      // A slow sync (5 s) to find a writeback error during the run. Its
      // clock starts when the disk is activated, which can be well after
      // `opened` on a slow host (738 ms under TSan): wait for the first
      // sync instead of sleeping a fixed time. The cadence check above
      // already showed it is not the 100 ms one.
      const auto deadline = opened + std::chrono::seconds(15);
      while (group.SlabStats().table_syncs == 0 &&
             std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      const double waited_s =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        opened)
              .count();
      EXPECT_GT(group.SlabStats().table_syncs, 0u)
          << "auto: no slow sync within 15 s";
      EXPECT_GE(waited_s, 4.9) << "auto: the first sync came before 5 s";
    }
  }
}

// The quotas file is read again on request (SIGHUP in the server): new
// limits apply to the next PUT, a file that does not parse leaves the old
// ones, and the rejection count of a tenant kept stays.
TEST_F(DiskGroupTest, TenantQuotasReloadFromTheirFile) {
  const auto dirs = Dirs(1);
  const fs::path quotas = base_ / "quotas-reload";
  std::ofstream(quotas) << "0000000000003333 10\n";
  ScopedEnv quota_file("DFKV_TENANT_QUOTAS_FILE", quotas.string());
  ScopedEnv default_quota("DFKV_TENANT_DEFAULT_QUOTA_BYTES", "0");
  DiskCacheGroup group({dirs, 1ull << 30, "file"});
  ASSERT_TRUE(group.Healthy()) << group.StartupError();
  EXPECT_EQ(group.TenantQuotaBytes(0x3333), 10u);
  EXPECT_EQ(group.TenantQuotaBytes(0x4444), 0u);
  const std::string value(8, 'v');
  ASSERT_EQ(group.Cache(BlockKey{1, 0, 0x3333}, value.data(), value.size()),
            Status::kOk);
  EXPECT_EQ(group.Cache(BlockKey{2, 0, 0x3333}, value.data(), value.size()),
            Status::kQuotaExceeded);

  std::ofstream(quotas, std::ios::trunc) << "0000000000003333 100\n"
                                         << "0000000000004444 5\n";
  std::string error;
  ASSERT_TRUE(group.ReloadTenantQuotas(&error)) << error;
  EXPECT_EQ(group.TenantQuotaBytes(0x3333), 100u);
  EXPECT_EQ(group.TenantQuotaBytes(0x4444), 5u);
  EXPECT_EQ(group.Cache(BlockKey{2, 0, 0x3333}, value.data(), value.size()),
            Status::kOk);
  EXPECT_EQ(group.Cache(BlockKey{3, 0, 0x4444}, value.data(), value.size()),
            Status::kQuotaExceeded);
  const auto metrics = group.ConfiguredTenantQuotaMetrics();
  ASSERT_EQ(metrics.size(), 2u);
  EXPECT_EQ(metrics[0].tenant_hash, 0x3333u);
  EXPECT_EQ(metrics[0].rejections, 1u) << "kept across the reload";
  EXPECT_EQ(group.TenantQuotaReloads(), 1u);

  std::ofstream(quotas, std::ios::trunc) << "not a quota line\n";
  EXPECT_FALSE(group.ReloadTenantQuotas(&error));
  EXPECT_FALSE(error.empty());
  EXPECT_EQ(group.TenantQuotaBytes(0x3333), 100u) << "a bad file changes nothing";
  EXPECT_EQ(group.TenantQuotaReloadFailures(), 1u);
}

// A reload takes the quotas lock alone, with writers running: no rejection
// count is lost, and none is lost across the reloads either.
TEST_F(DiskGroupTest, TenantQuotasReloadWhileWritersAreRejected) {
  const auto dirs = Dirs(1);
  const fs::path quotas = base_ / "quotas-reload-busy";
  std::ofstream(quotas) << "0000000000005555 1\n";
  ScopedEnv quota_file("DFKV_TENANT_QUOTAS_FILE", quotas.string());
  ScopedEnv default_quota("DFKV_TENANT_DEFAULT_QUOTA_BYTES", "0");
  DiskCacheGroup group({dirs, 1ull << 30, "file"});
  ASSERT_TRUE(group.Healthy()) << group.StartupError();
  constexpr int kThreads = 4, kWrites = 200;
  const std::string value(8, 'v');
  std::atomic<int> done{0};
  std::vector<std::thread> writers;
  for (int t = 0; t < kThreads; ++t)
    writers.emplace_back([&, t] {
      for (int i = 0; i < kWrites; ++i)
        EXPECT_EQ(group.Cache(BlockKey{static_cast<uint64_t>(t * kWrites + i),
                                       0, 0x5555},
                              value.data(), value.size()),
                  Status::kQuotaExceeded);
      ++done;
    });
  std::string error;
  int reloads = 0;
  while (done.load() < kThreads || reloads < 5) {
    ASSERT_TRUE(group.ReloadTenantQuotas(&error)) << error;
    ++reloads;
  }
  for (auto& writer : writers) writer.join();
  const auto metrics = group.ConfiguredTenantQuotaMetrics();
  ASSERT_EQ(metrics.size(), 1u);
  EXPECT_EQ(metrics[0].rejections,
            static_cast<uint64_t>(kThreads * kWrites));
  EXPECT_EQ(group.TenantQuotaReloads(), static_cast<uint64_t>(reloads));
}

// A batch over more tenant stripes than are held at once is run in parts;
// each tenant is admitted as in one batch.
TEST_F(DiskGroupTest, QuotaBatchOverManyStripesIsAdmittedPerTenant) {
  const auto dirs = Dirs(1);
  constexpr uint64_t kTenants = 40;  // 40 stripes: more than one part
  const fs::path quotas = base_ / "quotas-many";
  {
    std::ofstream file(quotas);
    char line[64];
    for (uint64_t tenant = 1; tenant <= kTenants; ++tenant) {
      std::snprintf(line, sizeof(line), "%016llx 10\n",
                    static_cast<unsigned long long>(tenant));
      file << line;
    }
  }
  ScopedEnv quota_file("DFKV_TENANT_QUOTAS_FILE", quotas.string());
  ScopedEnv default_quota("DFKV_TENANT_DEFAULT_QUOTA_BYTES", "0");
  DiskCacheGroup group({dirs, 1ull << 30, "file"});
  ASSERT_TRUE(group.Healthy()) << group.StartupError();
  void* raw = std::aligned_alloc(4096, 4096);
  ASSERT_NE(raw, nullptr);
  std::unique_ptr<void, decltype(&std::free)> buffer(raw, &std::free);
  std::memset(raw, 'm', 4096);
  std::vector<StoreEngine::CacheBatchItem> items;
  for (uint64_t round = 0; round < 2; ++round)
    for (uint64_t tenant = 1; tenant <= kTenants; ++tenant)
      items.push_back({BlockKey{round * 1000 + tenant, 0, tenant},
                       static_cast<char*>(raw), 6, 4096});
  // A tenant without a quota rides along (with the first part).
  const BlockKey unlimited{5000, 0, 0x9999};
  items.push_back({unlimited, static_cast<char*>(raw), 6, 4096});
  const auto statuses = group.CacheDirectBatch(items);
  ASSERT_EQ(statuses.size(), items.size());
  for (size_t i = 0; i < 2 * kTenants; ++i)
    EXPECT_EQ(statuses[i],
              i < kTenants ? Status::kOk : Status::kQuotaExceeded)
        << i;
  EXPECT_EQ(statuses.back(), Status::kOk);
  EXPECT_TRUE(group.IsCached(unlimited));
  for (uint64_t tenant = 1; tenant <= kTenants; ++tenant)
    EXPECT_EQ(group.TenantUsedBytes(tenant), 6u) << tenant;
  EXPECT_EQ(group.TenantQuotaRejections(), kTenants);
}

namespace {
// Small slab disks; `stores` by directory order.
struct SmallSlabGroup {
  std::vector<DiskSlabStore*> stores;
  std::unique_ptr<DiskCacheGroup> group;

  explicit SmallSlabGroup(const std::vector<std::string>& dirs) {
    stores.assign(dirs.size(), nullptr);
    DiskCacheGroup::Options options{dirs, dirs.size() * (4ull << 20), "slab"};
    options.engine_factory =
        [this, dirs](const std::string& dir,
                     uint64_t capacity) -> std::unique_ptr<StoreEngine> {
      DiskSlabStore::Options o;
      o.dir = dir;
      o.capacity_bytes = capacity;
      o.extent_bytes = 1 << 20;
      o.slot_granularity = 4096;
      o.table_sync_ms = 0;
      o.reclaim_interval_ms = 0;
      o.defer_activation = true;
      auto store = std::make_unique<DiskSlabStore>(o);
      for (size_t i = 0; i < dirs.size(); ++i)
        if (dirs[i] == dir) stores[i] = store.get();
      return store;
    };
    group = std::make_unique<DiskCacheGroup>(std::move(options));
  }

  // Fails disk `i` for good with a PUT of a new key routed to it.
  void Fail(size_t i, uint64_t first_id) {
    const std::string value(4096, 'f');
    uint64_t id = first_id;
    while (group->DiskIndexOf(BlockKey{id, 0}) != i) ++id;
    DiskSlabStoreTestPeer::SetRecordWriteHook(stores[i], [] { return false; });
    EXPECT_EQ(group->Cache(BlockKey{id, 0}, value.data(), value.size()),
              Status::kIOError);
    DiskSlabStoreTestPeer::SetRecordWriteHook(stores[i], {});
    EXPECT_FALSE(stores[i]->Healthy());
  }
};
}  // namespace

// A disk that fails while serving hands its keys to the healthy disks: they
// miss there until written again, and are then served. Keys of healthy disks
// stay where they are. Each further failure moves only the keys of the disk
// that failed. Once every disk has failed, the group is unhealthy.
TEST_F(DiskGroupTest, AFailedDiskHandsItsKeysToTheOthers) {
  const auto dirs = Dirs(4);
  SmallSlabGroup slab(dirs);
  DiskCacheGroup& group = *slab.group;
  ASSERT_TRUE(group.Healthy()) << group.StartupError();
  EXPECT_EQ(group.MinHealthyDisks(), 1u);
  const std::string value(4096, 'v');
  constexpr uint64_t kKeys = 150;
  std::vector<size_t> home(kKeys);
  for (uint64_t id = 0; id < kKeys; ++id) {
    const BlockKey key{id, 0, 0x77};
    home[id] = group.DiskIndexOf(key);
    ASSERT_EQ(group.Cache(key, value.data(), value.size()), Status::kOk);
  }
  const uint64_t tenant_before = group.TenantUsedBytes(0x77);
  const uint64_t on_disk1 = slab.stores[1]->TenantUsedBytes(0x77);
  ASSERT_GT(on_disk1, 0u);

  slab.Fail(1, 10000);
  EXPECT_EQ(group.FailedDisks(), 1u);
  EXPECT_TRUE(group.Healthy());
  EXPECT_EQ(group.TenantUsedBytes(0x77), tenant_before - on_disk1)
      << "what the failed disk held is not charged to its tenant";
  std::set<size_t> moved_to;
  std::string out;
  for (uint64_t id = 0; id < kKeys; ++id) {
    const BlockKey key{id, 0, 0x77};
    const size_t now = group.DiskIndexOf(key);
    EXPECT_FALSE(group.DiskFailedFor(key));
    if (home[id] != 1) {
      EXPECT_EQ(now, home[id]) << id << ": a healthy disk's key moved";
      EXPECT_EQ(group.Range(key, 0, 4096, &out), Status::kOk) << id;
      continue;
    }
    ASSERT_NE(now, 1u);
    moved_to.insert(now);
    EXPECT_FALSE(group.IsCached(key));
    EXPECT_EQ(group.Range(key, 0, 4096, &out), Status::kNotFound) << id;
    ASSERT_EQ(group.Cache(key, value.data(), value.size()), Status::kOk);
    EXPECT_EQ(group.Range(key, 0, 4096, &out), Status::kOk) << id;
  }
  EXPECT_EQ(moved_to, (std::set<size_t>{0, 2, 3})) << "spread over all";

  // A second failure moves only what is on that disk, the first disk's keys
  // that went there included; nothing else moves.
  std::vector<size_t> before(kKeys);
  for (uint64_t id = 0; id < kKeys; ++id)
    before[id] = group.DiskIndexOf(BlockKey{id, 0, 0x77});
  slab.Fail(0, 20000);
  EXPECT_EQ(group.FailedDisks(), 2u);
  EXPECT_TRUE(group.Healthy());
  for (uint64_t id = 0; id < kKeys; ++id) {
    const size_t now = group.DiskIndexOf(BlockKey{id, 0, 0x77});
    if (before[id] == 0)
      EXPECT_TRUE(now == 2 || now == 3) << id << " went to " << now;
    else
      EXPECT_EQ(now, before[id]) << id;
  }

  slab.Fail(2, 30000);
  slab.Fail(3, 40000);
  EXPECT_EQ(group.FailedDisks(), 4u);
  EXPECT_FALSE(group.Healthy());
  EXPECT_TRUE(group.DiskFailedFor(BlockKey{1, 0, 0x77}));
  EXPECT_EQ(group.Cache(BlockKey{50000, 0}, value.data(), value.size()),
            Status::kIOError);
}

// DFKV_DEGRADED_MIN_DISKS: how many disks must be left for the group to stay
// healthy. Above the disk count it means all of them; 0 and text refuse to
// start.
TEST_F(DiskGroupTest, DegradedMinDisksSetsHowManyMustBeLeft) {
  const auto dirs = Dirs(3);
  {
    ScopedEnv min_disks("DFKV_DEGRADED_MIN_DISKS", "2");
    SmallSlabGroup slab(dirs);
    ASSERT_TRUE(slab.group->Healthy()) << slab.group->StartupError();
    slab.Fail(0, 1000);
    EXPECT_TRUE(slab.group->Healthy());
    slab.Fail(1, 2000);
    EXPECT_FALSE(slab.group->Healthy());
  }
  {
    ScopedEnv min_disks("DFKV_DEGRADED_MIN_DISKS", "8");
    SmallSlabGroup slab(dirs);
    ASSERT_TRUE(slab.group->Healthy()) << slab.group->StartupError();
    EXPECT_EQ(slab.group->MinHealthyDisks(), 3u);
    slab.Fail(2, 3000);
    EXPECT_FALSE(slab.group->Healthy());
  }
  for (const char* bad : {"0", "two", "-1"}) {
    ScopedEnv min_disks("DFKV_DEGRADED_MIN_DISKS", bad);
    SmallSlabGroup slab(dirs);
    EXPECT_FALSE(slab.group->Healthy()) << bad;
    EXPECT_NE(slab.group->StartupError().find("DFKV_DEGRADED_MIN_DISKS"),
              std::string::npos)
        << slab.group->StartupError();
  }
}

namespace {
// The real system calls; writes fail with EIO while `armed`.
class FailingIo : public lse::IoBackend {
 public:
  std::atomic<bool> armed{false};
  int Open(const std::string& path, int flags, mode_t mode) override {
    return sys_->Open(path, flags, mode);
  }
  int Close(int fd) override { return sys_->Close(fd); }
  ssize_t Pwritev(int fd, const struct iovec* iov, int n, off_t offset,
                  bool dsync) override {
    if (armed.load()) {
      errno = EIO;
      return -1;
    }
    return sys_->Pwritev(fd, iov, n, offset, dsync);
  }
  ssize_t Pread(int fd, void* buf, size_t len, off_t offset) override {
    return sys_->Pread(fd, buf, len, offset);
  }
  int Fdatasync(int fd) override { return sys_->Fdatasync(fd); }
  int Fsync(int fd) override { return sys_->Fsync(fd); }
  int Fallocate(int fd, off_t offset, off_t len) override {
    return sys_->Fallocate(fd, offset, len);
  }
  int Ftruncate(int fd, off_t len) override { return sys_->Ftruncate(fd, len); }
  int Rename(const std::string& from, const std::string& to) override {
    return sys_->Rename(from, to);
  }
  int Mkdir(const std::string& path, mode_t mode) override {
    return sys_->Mkdir(path, mode);
  }
  int Unlink(const std::string& path) override { return sys_->Unlink(path); }

 private:
  std::unique_ptr<lse::IoBackend> sys_ = lse::MakeSystemIoBackend();
};

// A group of two disks of one engine that can be stopped and started again,
// and whose disks can be failed for good.
struct RestartableGroup {
  enum Engine { kSlab, kLog };
  Engine engine;
  std::vector<std::string> dirs;
  std::vector<StoreEngine*> stores;
  std::vector<std::shared_ptr<FailingIo>> ios;
  std::unique_ptr<DiskCacheGroup> group;
  // How the directories are spelled on the command line (empty: dirs).
  std::vector<std::string> spelled;
  std::function<bool(const std::string&)> record_location;
  std::function<Status(size_t, const BlockKey&)> misplaced_remove;
  std::function<bool(size_t)> sync_removals;

  void Open() {
    const std::vector<std::string> names = spelled.empty() ? dirs : spelled;
    stores.assign(dirs.size(), nullptr);
    ios.clear();
    for (size_t i = 0; i < dirs.size(); ++i)
      ios.push_back(std::make_shared<FailingIo>());
    DiskCacheGroup::Options options{names, dirs.size() * (64ull << 20), "slab"};
    options.record_location_for_test = record_location;
    options.misplaced_remove_for_test = misplaced_remove;
    options.sync_removals_for_test = sync_removals;
    options.engine_factory =
        [this, names](const std::string& dir,
                      uint64_t capacity) -> std::unique_ptr<StoreEngine> {
      size_t i = 0;
      while (names[i] != dir) ++i;
      std::unique_ptr<StoreEngine> store;
      if (engine == kSlab) {
        DiskSlabStore::Options o;
        o.dir = dir;
        o.capacity_bytes = capacity;
        o.extent_bytes = 1 << 20;
        o.slot_granularity = 4096;
        o.table_sync_ms = 0;
        o.reclaim_interval_ms = 0;
        o.defer_activation = true;
        store = std::make_unique<DiskSlabStore>(o);
      } else {
        LogStore::Options o;
        o.dir = dir;
        o.capacity_bytes = capacity;
        o.defer_activation = true;
        o.cleaner = false;
        o.io = ios[i];
        store = std::make_unique<LogStore>(o);
      }
      stores[i] = store.get();
      return store;
    };
    group = std::make_unique<DiskCacheGroup>(std::move(options));
    ASSERT_TRUE(group->Healthy()) << group->StartupError();
  }
  void Restart() {
    group.reset();
    Open();
  }
  // Fails disk `i` for good with a PUT of a new key homed there.
  void Fail(size_t i) {
    uint64_t id = 900000;
    while (group->DiskIndexOf(BlockKey{id, 0}) != i) ++id;
    if (engine == kSlab)
      DiskSlabStoreTestPeer::SetRecordWriteHook(
          static_cast<DiskSlabStore*>(stores[i]), [] { return false; });
    else
      ios[i]->armed = true;
    const std::string value(4096, 'f');
    EXPECT_NE(group->Cache(BlockKey{id, 0}, value.data(), value.size()),
              Status::kOk);
    if (engine == kSlab)
      DiskSlabStoreTestPeer::SetRecordWriteHook(
          static_cast<DiskSlabStore*>(stores[i]), {});
    else
      ios[i]->armed = false;
    ASSERT_FALSE(stores[i]->Healthy());
  }
  // A key homed on disk `i`.
  BlockKey HomedOn(size_t i) {
    uint64_t id = 1;
    while (group->DiskIndexOf(BlockKey{id, 0}) != i) ++id;
    return BlockKey{id, 0};
  }
  Status Put(const BlockKey& key, char fill) {
    const std::string value(4096, fill);
    return group->Cache(key, value.data(), value.size());
  }
  // The fill byte of the value served, or 0 for a miss.
  char Get(const BlockKey& key) {
    std::string out;
    const Status st = group->Range(key, 0, 4096, &out);
    if (st == Status::kNotFound) return 0;
    EXPECT_EQ(st, Status::kOk);
    return out.empty() ? '?' : out[0];
  }
};

std::string dirs_record(const std::string& dir) {
  return DiskCacheGroup::CanonicalDiskId(dir) + ".dfkv-failed-disks";
}

const char* EngineName(RestartableGroup::Engine e) {
  return e == RestartableGroup::kSlab ? "slab" : "log";
}
}  // namespace

// (a) A disk that failed comes back empty: a REMOVE or an overwrite done on
// the disk that took its keys over is not undone by its old value.
TEST_F(DiskGroupTest, AFailedDiskComesBackEmpty) {
  for (const auto engine : {RestartableGroup::kSlab, RestartableGroup::kLog}) {
    for (const bool overwrite : {false, true}) {
      SCOPED_TRACE(std::string(EngineName(engine)) +
                   (overwrite ? " overwrite" : " remove"));
      fs::remove_all(base_);
      RestartableGroup g{engine, Dirs(2)};
      ASSERT_NO_FATAL_FAILURE(g.Open());
      const BlockKey key = g.HomedOn(0);
      ASSERT_EQ(g.Put(key, '1'), Status::kOk);
      ASSERT_NO_FATAL_FAILURE(g.Fail(0));
      ASSERT_EQ(g.group->DiskIndexOf(key), 1u) << "taken over";
      if (overwrite) {
        ASSERT_EQ(g.Put(key, '2'), Status::kOk);
        ASSERT_EQ(g.Get(key), '2');
      } else {
        ASSERT_NE(g.group->Remove(key), Status::kIOError);
        ASSERT_EQ(g.Get(key), 0);
      }
      ASSERT_TRUE(fs::exists(dirs_record(g.dirs[1])));

      ASSERT_NO_FATAL_FAILURE(g.Restart());
      EXPECT_EQ(g.group->DiskIndexOf(key), 0u) << "home again";
      EXPECT_EQ(g.Get(key), 0) << "neither the old value nor the takeover's";
      EXPECT_EQ(g.group->OpenMisplacedKeys(), overwrite ? 1u : 0u);
      EXPECT_EQ(g.group->Count(), 0u);

      // A clean stop of a run with no failure clears the record.
      ASSERT_EQ(g.Put(key, '3'), Status::kOk);
      ASSERT_NO_FATAL_FAILURE(g.Restart());
      EXPECT_FALSE(fs::exists(dirs_record(g.dirs[1])));
      EXPECT_EQ(g.group->OpenEmptiedKeys() + g.group->OpenMisplacedKeys(), 0u);
      EXPECT_EQ(g.Get(key), '3') << "an ordinary restart keeps the cache";
    }
  }
}

// (b) What a failed run wrote in place of a failed disk is gone after the
// restart: when the same disk fails again later, the key goes to the same
// disk as before, which must not serve that old value after a REMOVE or an
// overwrite done at home in between.
TEST_F(DiskGroupTest, TakeoverLeftoversDoNotOutliveTheRun) {
  for (const auto engine : {RestartableGroup::kSlab, RestartableGroup::kLog}) {
    for (const bool overwrite : {false, true}) {
      SCOPED_TRACE(std::string(EngineName(engine)) +
                   (overwrite ? " overwrite" : " remove"));
      fs::remove_all(base_);
      RestartableGroup g{engine, Dirs(2)};
      ASSERT_NO_FATAL_FAILURE(g.Open());
      const BlockKey key = g.HomedOn(0);
      ASSERT_NO_FATAL_FAILURE(g.Fail(0));
      ASSERT_EQ(g.Put(key, '2'), Status::kOk);  // on disk 1
      ASSERT_EQ(g.Get(key), '2');

      ASSERT_NO_FATAL_FAILURE(g.Restart());
      EXPECT_EQ(g.group->OpenMisplacedKeys(), 1u);
      ASSERT_EQ(g.Put(key, '3'), Status::kOk);  // at home
      ASSERT_EQ(g.group->Remove(key), Status::kOk);
      if (overwrite) ASSERT_EQ(g.Put(key, '4'), Status::kOk);

      ASSERT_NO_FATAL_FAILURE(g.Fail(0));
      ASSERT_EQ(g.group->DiskIndexOf(key), 1u);
      EXPECT_EQ(g.Get(key), 0) << "the first run's takeover value came back";
    }
  }
}

// Below DFKV_DEGRADED_MIN_DISKS nothing moves: a failed disk's keys stay on
// it and fail, as before degradation existed.
TEST_F(DiskGroupTest, BelowTheMinimumAFailedDisksKeysDoNotMove) {
  ScopedEnv min_disks("DFKV_DEGRADED_MIN_DISKS", "2");
  RestartableGroup g{RestartableGroup::kSlab, Dirs(2)};
  ASSERT_NO_FATAL_FAILURE(g.Open());
  const BlockKey key = g.HomedOn(0);
  ASSERT_NO_FATAL_FAILURE(g.Fail(0));
  EXPECT_FALSE(g.group->Healthy());
  EXPECT_EQ(g.group->DiskIndexOf(key), 0u);
  EXPECT_TRUE(g.group->DiskFailedFor(key));
  EXPECT_EQ(g.Put(key, '1'), Status::kIOError);
  EXPECT_FALSE(fs::exists(dirs_record(g.dirs[1]))) << "nothing moved";
}

// The record is written only next to a cache directory whose parent is on
// the same filesystem (else it could outlive less than the data, e.g. on a
// mount root or a container volume). With no such place, nothing moves.
TEST_F(DiskGroupTest, AFailureRecordLivesOnlyOnTheDisksOwnFilesystem) {
  {
    const auto dirs = Dirs(3);
    const std::string not_here = DiskCacheGroup::CanonicalDiskId(dirs[2]);
    RestartableGroup g{RestartableGroup::kSlab, dirs};
    g.record_location = [&](const std::string& dir) { return dir != not_here; };
    ASSERT_NO_FATAL_FAILURE(g.Open());
    const BlockKey key = g.HomedOn(0);
    ASSERT_NO_FATAL_FAILURE(g.Fail(0));
    EXPECT_NE(g.group->DiskIndexOf(key), 0u) << "moved";
    EXPECT_TRUE(g.group->Healthy());
    EXPECT_TRUE(fs::exists(dirs_record(dirs[1])));
    EXPECT_FALSE(fs::exists(dirs_record(dirs[2])));
  }
  fs::remove_all(base_);
  {
    RestartableGroup g{RestartableGroup::kSlab, Dirs(2)};
    g.record_location = [](const std::string&) { return false; };
    ASSERT_NO_FATAL_FAILURE(g.Open());
    const BlockKey key = g.HomedOn(0);
    ASSERT_NO_FATAL_FAILURE(g.Fail(0));
    EXPECT_EQ(g.group->DiskIndexOf(key), 0u) << "nothing moves";
    EXPECT_FALSE(g.group->Healthy());
    EXPECT_EQ(g.Put(key, '1'), Status::kIOError);
    EXPECT_FALSE(fs::exists(dirs_record(g.dirs[1])));
  }
}

// A record that is there but cannot be used empties every disk: a copy that
// cannot be read (not a file), an empty one (the writer never writes one),
// or one naming a directory this node does not have. What sits on the
// record path and is not a file is never removed by the node.
TEST_F(DiskGroupTest, AnUnusableFailureRecordEmptiesEveryDisk) {
  enum Kind { kNotAFile, kEmpty, kUnknownId };
  for (const auto engine : {RestartableGroup::kSlab, RestartableGroup::kLog}) {
    for (const Kind kind : {kNotAFile, kEmpty, kUnknownId}) {
      SCOPED_TRACE(std::string(EngineName(engine)) + " " +
                   std::to_string(static_cast<int>(kind)));
      fs::remove_all(base_);
      RestartableGroup g{engine, Dirs(2)};
      ASSERT_NO_FATAL_FAILURE(g.Open());
      const BlockKey a = g.HomedOn(0), b = g.HomedOn(1);
      ASSERT_EQ(g.Put(a, 'a'), Status::kOk);
      ASSERT_EQ(g.Put(b, 'b'), Status::kOk);
      g.group.reset();
      const std::string record = dirs_record(g.dirs[1]);
      if (kind == kNotAFile) {
        fs::create_directories(record);
        std::ofstream(record + "/keep") << "x";
      } else if (kind == kEmpty) {
        std::ofstream{record};
      } else {
        std::ofstream(record) << (base_ / "gone").string() << "\n";
      }
      ASSERT_NO_FATAL_FAILURE(g.Open());
      EXPECT_EQ(g.group->OpenUnusableKeys(), 2u);
      EXPECT_EQ(g.group->Count(), 0u);
      EXPECT_EQ(g.Get(a), 0);
      EXPECT_EQ(g.Get(b), 0);
      ASSERT_EQ(g.Put(a, 'c'), Status::kOk);
      ASSERT_NO_FATAL_FAILURE(g.Restart());
      if (kind == kNotAFile) {
        EXPECT_TRUE(fs::exists(record + "/keep")) << "not ours: left alone";
        EXPECT_EQ(g.Get(a), 0) << "every start empties all disks meanwhile";
        fs::remove_all(record);  // the operator moves it away
        ASSERT_EQ(g.Put(a, 'c'), Status::kOk);
        ASSERT_NO_FATAL_FAILURE(g.Restart());
      } else {
        EXPECT_FALSE(fs::exists(record)) << "a clean stop removes the file";
      }
      EXPECT_EQ(g.group->OpenUnusableKeys(), 0u);
      EXPECT_EQ(g.Get(a), 'c') << "an ordinary restart keeps the cache";
    }
  }
}

// A disk whose not-home keys could not all be removed at the start keeps a
// value it must not serve for another disk: it takes no failed disk's keys
// in this run, so a second failure of that key's home cannot route the key
// to the stale copy. The record stays for the next start.
TEST_F(DiskGroupTest, ADiskThatKeptAStaleTakeoverValueTakesNoKeysThisRun) {
  RestartableGroup g{RestartableGroup::kSlab, Dirs(3)};
  ASSERT_NO_FATAL_FAILURE(g.Open());
  const BlockKey key = g.HomedOn(0);
  ASSERT_NO_FATAL_FAILURE(g.Fail(0));
  ASSERT_EQ(g.Put(key, '2'), Status::kOk);
  const size_t holder = g.group->DiskIndexOf(key);
  ASSERT_NE(holder, 0u);
  g.group.reset();
  g.misplaced_remove = [&g, holder](size_t disk, const BlockKey& k) {
    return disk == holder ? Status::kIOError : g.stores[disk]->Remove(k);
  };
  ASSERT_NO_FATAL_FAILURE(g.Open());
  EXPECT_TRUE(g.group->TakesNoOtherDisksKeys(holder));
  EXPECT_FALSE(g.group->TakesNoOtherDisksKeys(3 - holder));
  ASSERT_EQ(g.group->DiskIndexOf(key), 0u);
  ASSERT_NE(g.group->Remove(key), Status::kIOError);  // at home
  ASSERT_NO_FATAL_FAILURE(g.Fail(0));
  EXPECT_EQ(g.group->DiskIndexOf(key), 3 - holder) << "not to the stale copy";
  EXPECT_EQ(g.Get(key), 0);
  g.group.reset();
  EXPECT_TRUE(fs::exists(dirs_record(g.dirs[3 - holder]))) << "kept";
}

// Disks excluded from takeover do not count toward DFKV_DEGRADED_MIN_DISKS:
// with every healthy disk excluded, a failed disk's keys would fail while the
// node looked healthy. It leaves placement instead.
TEST_F(DiskGroupTest, DisksExcludedFromTakeoverDoNotCountTowardTheMinimum) {
  RestartableGroup g{RestartableGroup::kSlab, Dirs(3)};
  ASSERT_NO_FATAL_FAILURE(g.Open());
  std::vector<BlockKey> homed_on_0;
  for (uint64_t id = 1; homed_on_0.size() < 64; ++id)
    if (g.group->DiskIndexOf(BlockKey{id, 0}) == 0)
      homed_on_0.push_back(BlockKey{id, 0});
  ASSERT_NO_FATAL_FAILURE(g.Fail(0));
  std::vector<int> held(3, 0);  // disk 0's keys each other disk took over
  for (const BlockKey& key : homed_on_0) {
    ASSERT_EQ(g.Put(key, '2'), Status::kOk);
    ++held[g.group->DiskIndexOf(key)];
  }
  ASSERT_GT(held[1], 0);
  ASSERT_GT(held[2], 0);
  g.group.reset();
  g.misplaced_remove = [](size_t, const BlockKey&) { return Status::kIOError; };
  ASSERT_NO_FATAL_FAILURE(g.Open());
  ASSERT_TRUE(g.group->TakesNoOtherDisksKeys(1));
  ASSERT_TRUE(g.group->TakesNoOtherDisksKeys(2));
  EXPECT_TRUE(g.group->Healthy()) << "each disk serves its own keys";
  ASSERT_NO_FATAL_FAILURE(g.Fail(0));
  EXPECT_FALSE(g.group->Healthy())
      << "two healthy disks, but none takes the failed disk's keys";
  EXPECT_EQ(g.group->DiskIndexOf(homed_on_0[0]), 0u);
  EXPECT_NE(g.Put(homed_on_0[0], '3'), Status::kOk);
  g.group.reset();
  g.misplaced_remove = nullptr;
}

// The keys the start after a failed run drops from a disk that is not their
// home stay dropped after a power loss only once the engine synced the
// removal (the file engine's REMOVE renames without a directory sync). Until
// it did, the record stays: a clean close does not clear it.
TEST_F(DiskGroupTest, AnUnsyncedNotHomeRemovalKeepsTheRecord) {
  for (const bool synced : {true, false}) {
    SCOPED_TRACE(synced ? "synced" : "not synced");
    fs::remove_all(base_);
    RestartableGroup g{RestartableGroup::kSlab, Dirs(2)};
    ASSERT_NO_FATAL_FAILURE(g.Open());
    const BlockKey key = g.HomedOn(0);
    ASSERT_NO_FATAL_FAILURE(g.Fail(0));
    ASSERT_EQ(g.Put(key, '2'), Status::kOk);  // on disk 1
    g.group.reset();
    std::vector<int> syncs(2, 0);
    g.sync_removals = [&](size_t disk) {
      ++syncs[disk];
      return synced;
    };
    ASSERT_NO_FATAL_FAILURE(g.Open());
    EXPECT_EQ(g.group->OpenMisplacedKeys(), 1u);
    EXPECT_EQ(syncs, (std::vector<int>{0, 1})) << "the disk that dropped a key";
    EXPECT_EQ(g.Get(key), 0);
    g.group.reset();  // a clean close, no disk failed in this run
    EXPECT_EQ(fs::exists(dirs_record(g.dirs[1])), !synced);
    g.sync_removals = nullptr;
  }
}

// Ids are canonical: the same directories spelled with a trailing slash in
// the failed run are recognized, and the failed disk is emptied.
TEST_F(DiskGroupTest, AFailureRecordMatchesTheSameDirectorySpelledOtherwise) {
  for (const auto engine : {RestartableGroup::kSlab, RestartableGroup::kLog}) {
    SCOPED_TRACE(EngineName(engine));
    fs::remove_all(base_);
    RestartableGroup g{engine, Dirs(2)};
    // A key homed on disk 0 under both spellings (the ring hashes the
    // names as given).
    ASSERT_NO_FATAL_FAILURE(g.Open());
    std::set<uint64_t> home0;
    for (uint64_t id = 1; id < 200; ++id)
      if (g.group->DiskIndexOf(BlockKey{id, 0}) == 0) home0.insert(id);
    g.group.reset();
    for (const auto& dir : g.dirs) g.spelled.push_back(dir + "/");
    ASSERT_NO_FATAL_FAILURE(g.Open());
    uint64_t id = 1;
    while (id < 200 && (!home0.count(id) ||
                        g.group->DiskIndexOf(BlockKey{id, 0}) != 0))
      ++id;
    ASSERT_LT(id, 200u);
    const BlockKey key{id, 0};
    ASSERT_EQ(g.Put(key, '1'), Status::kOk);
    ASSERT_NO_FATAL_FAILURE(g.Fail(0));
    ASSERT_NE(g.group->Remove(key), Status::kIOError);  // on the takeover
    g.group.reset();
    g.spelled.clear();
    ASSERT_NO_FATAL_FAILURE(g.Open());
    EXPECT_EQ(g.group->OpenUnusableKeys(), 0u) << "recognized";
    EXPECT_EQ(g.group->DiskIndexOf(key), 0u);
    EXPECT_EQ(g.Get(key), 0) << "the failed disk's old value is gone";
    if (engine == RestartableGroup::kLog)
      EXPECT_GE(g.group->OpenEmptiedKeys(), 1u);
  }
}

// A record written later in a run names only that run's failures: a disk the
// start already emptied is not emptied again.
TEST_F(DiskGroupTest, ALaterRecordNamesOnlyThisRunsFailures) {
  RestartableGroup g{RestartableGroup::kLog, Dirs(3)};
  ASSERT_NO_FATAL_FAILURE(g.Open());
  const BlockKey on0 = g.HomedOn(0), on1 = g.HomedOn(1);
  ASSERT_NO_FATAL_FAILURE(g.Fail(0));
  EXPECT_NE(g.group->DiskIndexOf(on0), 0u);  // puts it on record
  ASSERT_NO_FATAL_FAILURE(g.Restart());
  ASSERT_NO_FATAL_FAILURE(g.Fail(1));
  EXPECT_NE(g.group->DiskIndexOf(on1), 1u);
  std::ifstream in(dirs_record(g.dirs[2]));
  std::string text((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  EXPECT_EQ(text, DiskCacheGroup::CanonicalDiskId(g.dirs[1]) + "\n");
}

// DFKV_SLAB_PAYLOAD_CRC takes 0 or 1; anything else refuses to start.
TEST_F(DiskGroupTest, SlabPayloadCrcSettingIsZeroOrOne) {
  const auto dirs = Dirs(1);
  for (const char* good : {"0", "1"}) {
    fs::remove_all(dirs[0]);
    fs::create_directories(dirs[0]);
    ScopedEnv crc("DFKV_SLAB_PAYLOAD_CRC", good);
    DiskCacheGroup group({dirs, 1ull << 30, "slab"});
    EXPECT_TRUE(group.Healthy()) << good << ": " << group.StartupError();
  }
  for (const char* bad : {"2", "on", "off"}) {
    ScopedEnv crc("DFKV_SLAB_PAYLOAD_CRC", bad);
    DiskCacheGroup group({dirs, 1ull << 30, "slab"});
    EXPECT_FALSE(group.Healthy()) << bad;
    EXPECT_NE(group.StartupError().find("DFKV_SLAB_PAYLOAD_CRC"),
              std::string::npos)
        << group.StartupError();
  }
}

// DFKV_SLAB_BALANCE takes 0 or 1; anything else refuses to start.
TEST_F(DiskGroupTest, SlabBalanceSettingIsZeroOrOne) {
  const auto dirs = Dirs(1);
  for (const char* good : {"0", "1"}) {
    fs::remove_all(dirs[0]);
    fs::create_directories(dirs[0]);
    ScopedEnv balance("DFKV_SLAB_BALANCE", good);
    DiskCacheGroup group({dirs, 1ull << 30, "slab"});
    EXPECT_TRUE(group.Healthy()) << good << ": " << group.StartupError();
  }
  for (const char* bad : {"2", "on", "true"}) {
    ScopedEnv balance("DFKV_SLAB_BALANCE", bad);
    DiskCacheGroup group({dirs, 1ull << 30, "slab"});
    EXPECT_FALSE(group.Healthy()) << bad;
    EXPECT_NE(group.StartupError().find("DFKV_SLAB_BALANCE"), std::string::npos)
        << group.StartupError();
  }
}
