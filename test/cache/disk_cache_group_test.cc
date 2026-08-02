// TDD R7 — DiskCacheGroup: one cache node spanning multiple NVMe SSDs, mirroring
// dingo-cache's intra-node Ketama across --cache_dir=d1,d2,d3 (each disk an
// independent KVStore with its own LRU; total capacity split across disks).
#include "cache/disk_cache_group.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

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
  DiskCacheGroup g({dirs, 1ull << 30});
  EXPECT_EQ(g.DiskCount(), 3u);
  std::string v(500, 'x');
  for (int i = 0; i < 300; ++i)
    ASSERT_EQ(g.Cache(BlockKey{(uint64_t)i, 0}, v.data(), v.size()), Status::kOk);
  for (auto& d : dirs) EXPECT_GT(CountIn(d), 0u) << d;  // every disk used
  EXPECT_EQ(g.Count(), 300u);
}

TEST_F(DiskGroupTest, PutGetRoundTripAcrossDisks) {
  auto dirs = Dirs(3);
  DiskCacheGroup g({dirs, 1ull << 30});
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
  DiskCacheGroup g({dirs, 1ull << 30});
  std::string v(10, 'a');
  ASSERT_EQ(g.Cache(BlockKey{77, 0}, v.data(), v.size()), Status::kOk);
  ASSERT_EQ(g.Cache(BlockKey{77, 0}, v.data(), v.size()), Status::kOk);  // idempotent, same disk
  size_t total = 0;
  for (auto& d : dirs) total += CountIn(d);
  EXPECT_EQ(total, 1u);  // exactly one disk holds it, not duplicated
}

TEST_F(DiskGroupTest, MissReturnsNotFound) {
  auto dirs = Dirs(2);
  DiskCacheGroup g({dirs, 1ull << 30});
  std::string out;
  EXPECT_EQ(g.Range(BlockKey{404, 0}, 0, 8, &out), Status::kNotFound);
  EXPECT_FALSE(g.IsCached(BlockKey{404, 0}));
}

TEST_F(DiskGroupTest, PerDiskCapacityKeepsTotalBounded) {
  auto dirs = Dirs(3);
  // total cap split evenly -> ~ (cap/3) per disk; write far more than capacity
  const uint64_t cap = 3 * 20 * 1000;  // ~20 objs/disk of 1000B
  DiskCacheGroup g({dirs, cap});
  std::string v(1000, 'y');
  for (int i = 0; i < 500; ++i)
    ASSERT_EQ(g.Cache(BlockKey{(uint64_t)i, 0}, v.data(), v.size()), Status::kOk);
  EXPECT_LE(g.UsedBytes(), cap);  // LRU per disk keeps total under the cap
  EXPECT_GT(g.Count(), 0u);
}

TEST_F(DiskGroupTest, ReloadFromDisksRebuilds) {
  auto dirs = Dirs(3);
  {
    DiskCacheGroup g({dirs, 1ull << 30});
    std::string v = "persisted";
    ASSERT_EQ(g.Cache(BlockKey{55, 0}, v.data(), v.size()), Status::kOk);
  }
  DiskCacheGroup g2({dirs, 1ull << 30});  // same disks, fresh instance
  EXPECT_TRUE(g2.IsCached(BlockKey{55, 0}));
  std::string out;
  ASSERT_EQ(g2.Range(BlockKey{55, 0}, 0, 9, &out), Status::kOk);
  EXPECT_EQ(out, "persisted");
}
