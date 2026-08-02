// StoreEngine wiring: DiskCacheGroup runs either the file (KVStore, default) or
// slab (DiskSlabStore) backend, selected by Options.engine / DFKV_STORE_ENGINE.
// The refactor must keep the file path a byte-for-byte drop-in and make slab a
// working alternative; both must round-trip and route across disks identically.
#include "cache/disk_cache_group.h"
#include "cache/kv_node_server.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {
class EngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    base_ = fs::temp_directory_path() /
            ("dfkv_eng_" + std::to_string(::testing::UnitTest::GetInstance()
                                              ->current_test_info()->line()));
    fs::remove_all(base_);
    ClearEnv();
  }
  void TearDown() override {
    fs::remove_all(base_);
    ClearEnv();
  }
  void ClearEnv() {
    for (const char* name :
         {"DFKV_STORE_ENGINE", "DFKV_DISK_HASH_WEIGHT", "DFKV_SLAB_WRITE",
          "DFKV_SLAB_GRANULARITY", "DFKV_SLAB_TABLE_SYNC_MS",
          "DFKV_SLAB_RECLAIM_MS"})
      ::unsetenv(name);
  }
  std::vector<std::string> Dirs(int n) {
    std::vector<std::string> v;
    for (int i = 0; i < n; ++i) {
      auto d = (base_ / ("d" + std::to_string(i))).string();
      fs::create_directories(d);
      v.push_back(d);
    }
    return v;
  }
  fs::path base_;
};

// Round-trip Cache/Range/RangeInto/Remove for N keys spread across a group.
void ExerciseGroup(DiskCacheGroup& g) {
  for (uint64_t i = 0; i < 40; ++i) {
    std::string v = "val-" + std::to_string(i) + std::string(50, 'x');
    ASSERT_EQ(g.Cache(BlockKey{i, 0, 1}, v.data(), v.size()), Status::kOk) << i;
  }
  EXPECT_EQ(g.Count(), 40u);
  for (uint64_t i = 0; i < 40; ++i) {
    std::string v = "val-" + std::to_string(i) + std::string(50, 'x');
    std::string out;
    ASSERT_EQ(g.Range(BlockKey{i, 0, 1}, 0, v.size(), &out), Status::kOk) << i;
    EXPECT_EQ(out, v) << i;
    EXPECT_TRUE(g.IsCached(BlockKey{i, 0, 1}));
    char buf[128];
    size_t got = 0;
    ASSERT_EQ(g.RangeInto(BlockKey{i, 0, 1}, 0, sizeof(buf), buf, sizeof(buf), &got),
              Status::kOk) << i;
    EXPECT_EQ(std::string(buf, got), v) << i;
  }
  ASSERT_EQ(g.Remove(BlockKey{0, 0, 1}), Status::kOk);
  EXPECT_FALSE(g.IsCached(BlockKey{0, 0, 1}));
  std::string miss;
  EXPECT_EQ(g.Range(BlockKey{9999, 0, 1}, 0, 10, &miss), Status::kNotFound);
}
}  // namespace

TEST_F(EngineTest, FileEngineRoundTripsAcrossDisks) {
  DiskCacheGroup::Options o;
  o.cache_dirs = Dirs(3);
  o.capacity_bytes = 3ull << 30;
  o.engine = "file";
  DiskCacheGroup g(o);
  ExerciseGroup(g);
}

TEST_F(EngineTest, SlabEngineRoundTripsAcrossDisks) {
  DiskCacheGroup::Options o;
  o.cache_dirs = Dirs(3);
  o.capacity_bytes = 3ull << 30;  // one 1 GiB extent per disk
  o.engine = "slab";
  DiskCacheGroup g(o);
  ExerciseGroup(g);
  EXPECT_GT(g.UsedBytes(), 0u);
}

TEST_F(EngineTest, DefaultIsFileEngine) {
  // No Options.engine and no env -> file backend, which lays out blocks/ dirs.
  DiskCacheGroup::Options o;
  o.cache_dirs = Dirs(1);
  o.capacity_bytes = 1ull << 30;
  DiskCacheGroup g(o);
  std::string v(64, 'f');
  ASSERT_EQ(g.Cache(BlockKey{7, 0, 1}, v.data(), v.size()), Status::kOk);
  // The file engine writes blocks/<bucket>/... ; the slab engine writes
  // extents/ + slots.tbl. Presence of "blocks" proves the default is file.
  EXPECT_TRUE(fs::exists(fs::path(o.cache_dirs[0]) / "blocks"));
  EXPECT_FALSE(fs::exists(fs::path(o.cache_dirs[0]) / "slots.tbl"));
}

TEST_F(EngineTest, EnvSelectsSlabWhenOptionEmpty) {
  ::setenv("DFKV_STORE_ENGINE", "slab", 1);
  DiskCacheGroup::Options o;
  o.cache_dirs = Dirs(1);
  o.capacity_bytes = 1ull << 30;
  // Options.engine empty -> read the env.
  DiskCacheGroup g(o);
  std::string v(64, 's');
  ASSERT_EQ(g.Cache(BlockKey{8, 0, 1}, v.data(), v.size()), Status::kOk);
  EXPECT_TRUE(fs::exists(fs::path(o.cache_dirs[0]) / "slots.tbl"));  // slab layout
  EXPECT_FALSE(fs::exists(fs::path(o.cache_dirs[0]) / "blocks"));
}

TEST_F(EngineTest, InvalidEngineAndSlabConfigurationFailClosed) {
  DiskCacheGroup::Options options;
  options.cache_dirs = Dirs(1);
  options.capacity_bytes = 1ull << 30;
  options.engine = "typo";
  DiskCacheGroup bad_engine(options);
  EXPECT_FALSE(bad_engine.Healthy());
  EXPECT_FALSE(bad_engine.StartupError().empty());

  options.engine = "slab";
  ::setenv("DFKV_SLAB_GRANULARITY", "1048576garbage", 1);
  DiskCacheGroup bad_number(options);
  EXPECT_FALSE(bad_number.Healthy());
  EXPECT_NE(bad_number.StartupError().find("GRANULARITY"),
            std::string::npos);
}

TEST_F(EngineTest, SlabCapacityMustResolveToWholeExtentsPerDisk) {
  DiskCacheGroup::Options options;
  options.cache_dirs = Dirs(1);
  options.capacity_bytes = (1ull << 30) + 1;
  options.engine = "slab";
  DiskCacheGroup group(options);
  EXPECT_FALSE(group.Healthy());
  EXPECT_NE(group.StartupError().find("multiple"), std::string::npos);
}

TEST_F(EngineTest, BackendSwitchRequiresExplicitMigration) {
  DiskCacheGroup::Options options;
  options.cache_dirs = Dirs(1);
  options.capacity_bytes = 1ull << 30;
  options.engine = "file";
  std::string value(64, 'f');
  {
    DiskCacheGroup file(options);
    ASSERT_TRUE(file.Healthy());
    ASSERT_EQ(file.Cache(BlockKey{9, 0, 1}, value.data(), value.size()),
              Status::kOk);
  }
  options.engine = "slab";
  {
    DiskCacheGroup slab(options);
    EXPECT_FALSE(slab.Healthy());
    EXPECT_FALSE(slab.StartupError().empty());
  }
  options.engine = "file";
  DiskCacheGroup reopened(options);
  ASSERT_TRUE(reopened.Healthy());
  std::string out;
  ASSERT_EQ(reopened.Range(BlockKey{9, 0, 1}, 0, value.size(), &out),
            Status::kOk);
  EXPECT_EQ(out, value);
}

TEST_F(EngineTest, EmptyAndDuplicateDiskListsAreRejected) {
  DiskCacheGroup::Options options;
  options.capacity_bytes = 1ull << 30;
  DiskCacheGroup empty(options);
  EXPECT_FALSE(empty.Healthy());

  const auto dirs = Dirs(1);
  options.cache_dirs = {dirs[0], dirs[0]};
  DiskCacheGroup duplicate(options);
  EXPECT_FALSE(duplicate.Healthy());
  EXPECT_NE(duplicate.StartupError().find("duplicate"), std::string::npos);
}

TEST_F(EngineTest, NodeRefusesToListenWhenDiskStoreIsUnhealthy) {
  const auto dirs = Dirs(1);
  ::setenv("DFKV_STORE_ENGINE", "not-an-engine", 1);
  KvNodeServer server(dirs[0], 1ull << 30);
  EXPECT_EQ(server.Start(0), Status::kIOError);
  EXPECT_EQ(server.port(), 0);
}
