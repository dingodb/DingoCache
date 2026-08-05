// StoreEngine wiring: DiskCacheGroup defaults to the production slab backend;
// file-per-block KVStore remains an explicit diagnostic choice. Options.engine
// overrides DFKV_STORE_ENGINE, which in turn overrides the slab default. Both
// engines must round-trip and route across disks identically.
#include "cache/disk_cache_group.h"
#include "cache/ram_tier.h"
#include "cache/kv_node_server.h"

#include <gtest/gtest.h>

#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <functional>
#include <memory>
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
          "DFKV_SLAB_RECLAIM_MS", "DFKV_RAM_TIER",
          "DFKV_RAM_TIER_BYTES", "DFKV_RAM_FLUSH_THREADS",
          "DFKV_RAM_TIER_EXTENT_BYTES",
          "DFKV_RAM_TIER_LARGE_RESERVE_BYTES", "DFKV_RAM_TIER_NUMA",
          "DFKV_RAM_TIER_SHARDS"})
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
    ASSERT_EQ(g.Cache(BlockKey{i, 0}, v.data(), v.size()), Status::kOk) << i;
  }
  EXPECT_EQ(g.Count(), 40u);
  for (uint64_t i = 0; i < 40; ++i) {
    std::string v = "val-" + std::to_string(i) + std::string(50, 'x');
    std::string out;
    ASSERT_EQ(g.Range(BlockKey{i, 0}, 0, v.size(), &out), Status::kOk) << i;
    EXPECT_EQ(out, v) << i;
    EXPECT_TRUE(g.IsCached(BlockKey{i, 0}));
    char buf[128];
    size_t got = 0;
    ASSERT_EQ(g.RangeInto(BlockKey{i, 0}, 0, sizeof(buf), buf, sizeof(buf), &got),
              Status::kOk) << i;
    EXPECT_EQ(std::string(buf, got), v) << i;
  }
  ASSERT_EQ(g.Remove(BlockKey{0, 0}), Status::kOk);
  EXPECT_FALSE(g.IsCached(BlockKey{0, 0}));
  std::string miss;
  EXPECT_EQ(g.Range(BlockKey{9999, 0}, 0, 10, &miss), Status::kNotFound);
}

struct RangeAnswer {
  Status status = Status::kInvalid;
  std::string bytes;
  size_t value_len = 0;
  bool exact_bytes = true;
};
using RangeCall = std::function<RangeAnswer(uint64_t, uint64_t)>;

void ExpectSharedRangeContract(const std::string& backend,
                               const RangeCall& call) {
  struct Example {
    uint64_t offset;
    uint64_t length;
    Status status;
    const char* bytes;
  };
  const Example examples[] = {
      {3, 0, Status::kOk, "3456789"},
      {10, 4, Status::kOk, ""},
      {11, 4, Status::kInvalid, ""},
      {3, 100, Status::kOk, "3456789"},
  };
  for (const auto& example : examples) {
    const RangeAnswer answer = call(example.offset, example.length);
    EXPECT_EQ(answer.status, example.status)
        << backend << " offset=" << example.offset
        << " length=" << example.length;
    if (answer.status == Status::kOk) {
      EXPECT_EQ(answer.bytes.size(), std::strlen(example.bytes)) << backend;
      if (answer.exact_bytes) {
        EXPECT_EQ(answer.bytes, example.bytes) << backend;
      }
      EXPECT_EQ(answer.value_len, 10u) << backend;
    }
  }
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

TEST_F(EngineTest, BackendRangeConformanceTable) {
  const BlockKey key{0x1234, 0x5678, 0x9abc};
  const std::string value = "0123456789";
  std::vector<std::unique_ptr<DiskCacheGroup>> groups;
  std::vector<std::pair<std::string, RangeCall>> table;

  auto add_disk_backend = [&](const std::string& engine,
                              const fs::path& path) {
    fs::create_directories(path);
    DiskCacheGroup::Options options;
    options.cache_dirs = {path.string()};
    options.capacity_bytes = 1ull << 30;
    options.engine = engine;
    auto group = std::make_unique<DiskCacheGroup>(options);
    ASSERT_TRUE(group->Healthy()) << engine;
    ASSERT_EQ(group->Cache(key, value.data(), value.size()), Status::kOk);
    DiskCacheGroup* store = group.get();

    table.push_back({engine + "/Range", [store, key](uint64_t off,
                                                     uint64_t len) {
      RangeAnswer answer;
      answer.status =
          store->Range(key, off, len, &answer.bytes, &answer.value_len);
      return answer;
    }});
    table.push_back({engine + "/RangeInto",
                     [store, key](uint64_t off, uint64_t len) {
      RangeAnswer answer;
      char buffer[32] = {};
      size_t out_len = 0;
      answer.status = store->RangeInto(key, off, len, buffer, sizeof(buffer),
                                       &out_len, &answer.value_len);
      if (answer.status == Status::kOk)
        answer.bytes.assign(buffer, out_len);
      return answer;
    }});
    table.push_back({engine + "/RangeDirect",
                     [store, key](uint64_t off, uint64_t len) {
      RangeAnswer answer;
      void* allocation = nullptr;
      if (::posix_memalign(&allocation, 4096, 8192) != 0)
        return answer;
      const char* data = nullptr;
      size_t out_len = 0;
      answer.status =
          store->RangeDirect(key, off, len, static_cast<char*>(allocation),
                             8192, &data, &out_len, &answer.value_len);
      if (answer.status == Status::kOk && out_len != 0)
        answer.bytes.assign(data, out_len);
      std::free(allocation);
      return answer;
    }});
    table.push_back({engine + "/RangeDirectPrep",
                     [store, key](uint64_t off, uint64_t len) {
      RangeAnswer answer;
      ReadLease lease;
      answer.status =
          store->RangeDirectPrep(key, off, len, 8192, &lease);
      if (answer.status == Status::kOk) {
        answer.bytes.assign(lease.payload_len, '?');
        answer.value_len = lease.value_len;
        answer.exact_bytes = false;
      }
      return answer;
    }});
    groups.push_back(std::move(group));
  };

  add_disk_backend("file", base_ / "file");
  add_disk_backend("slab", base_ / "slab");

  RamTier::Options ram_options;
  ram_options.bytes = 8ull << 20;
  ram_options.large_reserve_bytes = 0;
  auto ram = std::make_unique<RamTier>(
      ram_options, [](const BlockKey&, char*, size_t, size_t) { return true; });
  ASSERT_TRUE(ram->ok());
  ASSERT_EQ(ram->PutCommitted(key, value.data(), value.size()), Status::kOk);
  RamTier* ram_store = ram.get();
  table.push_back({"ram/GetPrep", [ram_store, key](uint64_t off, uint64_t len) {
    RangeAnswer answer;
    RamTier::Hit hit;
    if (ram_store->GetPrep(key, off, len, &hit)) {
      answer.status = Status::kOk;
      answer.bytes.assign(hit.ptr, hit.len);
      answer.value_len = hit.value_len;
    } else {
      answer.status =
          ram_store->Contains(key) ? Status::kInvalid : Status::kNotFound;
    }
    return answer;
  }});

  for (const auto& backend : table)
    ExpectSharedRangeContract(backend.first, backend.second);
}

TEST_F(EngineTest, NoOptionOrEnvironmentDefaultsToSlab) {
  DiskCacheGroup::Options o;
  o.cache_dirs = Dirs(1);
  o.capacity_bytes = 1ull << 30;
  DiskCacheGroup g(o);
  ASSERT_TRUE(g.Healthy()) << g.StartupError();
  EXPECT_EQ(g.EngineName(), "slab");
  std::string v(64, 's');
  ASSERT_EQ(g.Cache(BlockKey{7, 0}, v.data(), v.size()), Status::kOk);
  EXPECT_TRUE(fs::exists(fs::path(o.cache_dirs[0]) / "slots.tbl"));
  EXPECT_FALSE(fs::exists(fs::path(o.cache_dirs[0]) / "blocks"));
}

TEST_F(EngineTest, EnvironmentOverridesSlabDefaultWhenOptionEmpty) {
  ::setenv("DFKV_STORE_ENGINE", "file", 1);
  DiskCacheGroup::Options o;
  o.cache_dirs = Dirs(1);
  o.capacity_bytes = 1ull << 30;
  DiskCacheGroup g(o);
  ASSERT_TRUE(g.Healthy()) << g.StartupError();
  EXPECT_EQ(g.EngineName(), "file");
  std::string v(64, 'f');
  ASSERT_EQ(g.Cache(BlockKey{8, 0}, v.data(), v.size()), Status::kOk);
  EXPECT_TRUE(fs::exists(fs::path(o.cache_dirs[0]) / "blocks"));
  EXPECT_FALSE(fs::exists(fs::path(o.cache_dirs[0]) / "slots.tbl"));
}

TEST_F(EngineTest, ExplicitFileOptionOverridesSlabEnvironment) {
  ::setenv("DFKV_STORE_ENGINE", "slab", 1);
  DiskCacheGroup::Options o;
  o.cache_dirs = Dirs(1);
  o.capacity_bytes = 1ull << 30;
  o.engine = "file";
  DiskCacheGroup g(o);
  ASSERT_TRUE(g.Healthy()) << g.StartupError();
  EXPECT_EQ(g.EngineName(), "file");
  std::string v(64, 'f');
  ASSERT_EQ(g.Cache(BlockKey{9, 0}, v.data(), v.size()), Status::kOk);
  EXPECT_TRUE(fs::exists(fs::path(o.cache_dirs[0]) / "blocks"));
  EXPECT_FALSE(fs::exists(fs::path(o.cache_dirs[0]) / "slots.tbl"));
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

TEST_F(EngineTest, InvalidDefaultSlabCapacityRefusesStartup) {
  const auto dirs = Dirs(1);
  DiskCacheGroup::Options options;
  options.cache_dirs = dirs;
  options.capacity_bytes = (1ull << 30) + 1;
  DiskCacheGroup group(options);
  EXPECT_EQ(group.EngineName(), "slab");
  EXPECT_FALSE(group.Healthy());
  EXPECT_NE(group.StartupError().find("multiple"), std::string::npos);
  EXPECT_FALSE(fs::exists(fs::path(dirs[0]) / "blocks"));

  KvNodeServer server(dirs[0], options.capacity_bytes);
  EXPECT_EQ(server.engine_name(), "slab");
  EXPECT_FALSE(server.Healthy());
  EXPECT_EQ(server.Start(0), Status::kIOError);
  EXPECT_EQ(server.port(), 0);
  EXPECT_FALSE(fs::exists(fs::path(dirs[0]) / "blocks"));
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
    ASSERT_EQ(file.Cache(BlockKey{9, 0}, value.data(), value.size()),
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
  ASSERT_EQ(reopened.Range(BlockKey{9, 0}, 0, value.size(), &out),
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

TEST_F(EngineTest, RequestedRamTierFailureRefusesDiskOnlyStartup) {
  const auto dirs = Dirs(1);
  ::setenv("DFKV_STORE_ENGINE", "file", 1);
  ::setenv("DFKV_RAM_TIER", "1", 1);
  // Larger than the userspace virtual address range: posix_memalign must fail
  // before any page touch, independent of host free memory or overcommit.
  ::setenv("DFKV_RAM_TIER_BYTES", "9223372036854775808", 1);
  ::setenv("DFKV_RAM_TIER_NUMA", "off", 1);
  KvNodeServer server(dirs[0], 1ull << 30);
  EXPECT_FALSE(server.ram_enabled());
  EXPECT_FALSE(server.Healthy());
  EXPECT_EQ(server.Start(0), Status::kIOError);
  EXPECT_EQ(server.port(), 0);
}
