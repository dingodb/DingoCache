// Fail-closed startup (v1.40.3 hotfix, backport of the v2 semantics).
//
// Incident regression (hd04 glm, 2026-07-31): two members restarted with an
// unusable slab --dir. DiskSlabStore swallowed the init failure (ok_=false, no
// log), DiskCacheGroup installed the dead store anyway, KvNodeServer::Start()
// only bound TCP, and the node registered with the MDS and kept its ring
// shares while every op routed to it returned kIOError — a zombie member the
// lease-only liveness can never evict. Start() must refuse instead.
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "cache/disk_cache_group.h"
#include "cache/kv_node_server.h"

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {

// RAII env override that restores the previous value on scope exit.
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    const char* old = std::getenv(name);
    if (old) { had_ = true; old_ = old; }
    if (value) ::setenv(name, value, 1); else ::unsetenv(name);
  }
  ~ScopedEnv() {
    if (had_) ::setenv(name_.c_str(), old_.c_str(), 1);
    else ::unsetenv(name_.c_str());
  }
 private:
  std::string name_, old_;
  bool had_ = false;
};

class FailClosedStartupTest : public ::testing::Test {
 protected:
  void SetUp() override {
    base_ = fs::temp_directory_path() /
            ("dfkv_fc_" + std::to_string(::testing::UnitTest::GetInstance()
                                             ->current_test_info()->line()));
    fs::remove_all(base_);
    fs::create_directories(base_);
  }
  void TearDown() override { fs::remove_all(base_); }

  std::string GoodDir(const std::string& name) {
    auto d = (base_ / name).string();
    fs::create_directories(d);
    return d;
  }
  // A path that can never become a directory: a component of it is a regular
  // file, so create_directories fails with ENOTDIR for root and non-root alike
  // (chmod tricks don't survive CI containers running as root).
  std::string UnusableDir() {
    auto f = base_ / "notadir";
    std::FILE* fp = std::fopen(f.string().c_str(), "w");
    if (fp) std::fclose(fp);
    return (f / "sub").string();
  }
  fs::path base_;
};

TEST_F(FailClosedStartupTest, GroupReportsFailedSlabDir) {
  ScopedEnv engine("DFKV_STORE_ENGINE", "slab");
  auto good = GoodDir("disk0");
  auto bad = UnusableDir();
  DiskCacheGroup g({{good, bad}, 1ull << 30, "slab"});
  ASSERT_EQ(g.FailedDirs().size(), 1u);
  EXPECT_EQ(g.FailedDirs()[0], bad);
  EXPECT_EQ(g.DiskCount(), 2u);  // still constructed; refusal is Start()'s job
}

TEST_F(FailClosedStartupTest, HealthyGroupHasNoFailedDirs) {
  ScopedEnv engine("DFKV_STORE_ENGINE", "slab");
  DiskCacheGroup g({{GoodDir("disk0"), GoodDir("disk1")}, 1ull << 30, "slab"});
  EXPECT_TRUE(g.FailedDirs().empty());
}

TEST_F(FailClosedStartupTest, StartRefusesWhenAnySlabDirFailed) {
  ScopedEnv engine("DFKV_STORE_ENGINE", "slab");
  ScopedEnv ram("DFKV_RAM_TIER", nullptr);  // isolate the disk condition
  KvNodeServer srv({GoodDir("disk0"), UnusableDir()}, 1ull << 30);
  EXPECT_FALSE(srv.Healthy());
  EXPECT_EQ(srv.Start(0), Status::kIOError);
}

TEST_F(FailClosedStartupTest, StartSucceedsOnHealthyDirs) {
  ScopedEnv engine("DFKV_STORE_ENGINE", "slab");
  ScopedEnv ram("DFKV_RAM_TIER", nullptr);
  KvNodeServer srv({GoodDir("disk0")}, 1ull << 30);
  EXPECT_TRUE(srv.Healthy());
  ASSERT_EQ(srv.Start(0), Status::kOk);
  EXPECT_GT(srv.port(), 0);
  srv.Stop();
}

TEST_F(FailClosedStartupTest, StartRefusesWhenRequestedRamTierCannotAllocate) {
  ScopedEnv engine("DFKV_STORE_ENGINE", "slab");
  ScopedEnv ram("DFKV_RAM_TIER", "1");
  // 4 EiB: beyond any userspace VA range, so posix_memalign fails
  // deterministically regardless of overcommit settings.
  ScopedEnv bytes("DFKV_RAM_TIER_BYTES", "4611686018427387904");
  KvNodeServer srv({GoodDir("disk0")}, 1ull << 30);
  EXPECT_FALSE(srv.Healthy());
  EXPECT_EQ(srv.Start(0), Status::kIOError);
}

}  // namespace
