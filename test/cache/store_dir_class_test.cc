// Which engine a cache directory belongs to (log-structured engine design
// §10.1): markers decide, lost+found is ignored, slab and log markers
// together or unknown content are refused by the caller.
#include "cache/store_dir_class.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {
class StoreDirClassTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           ("dfkv_dirclass_" + std::to_string(::testing::UnitTest::GetInstance()
                                                  ->current_test_info()->line()));
    fs::remove_all(dir_);
    fs::create_directories(dir_);
  }
  void TearDown() override { fs::remove_all(dir_); }
  void File(const std::string& name) { std::ofstream(dir_ / name) << "x"; }
  void Dir(const std::string& name) { fs::create_directories(dir_ / name); }
  StoreDirClassification Classify() {
    StoreDirClassification out;
    std::string error;
    EXPECT_TRUE(ClassifyStoreDir(dir_.string(), &out, &error)) << error;
    return out;
  }
  fs::path dir_;
};
}  // namespace

TEST_F(StoreDirClassTest, EmptyAndLostAndFoundAreEmpty) {
  EXPECT_EQ(Classify().cls, StoreDirClass::kEmpty);
  Dir("lost+found");
  EXPECT_EQ(Classify().cls, StoreDirClass::kEmpty);
  EXPECT_FALSE(Classify().lost_found_in_use);
  // Files in lost+found do not change the class, but are reported (the log
  // engine refuses such a directory; slab and file ignore it).
  File("lost+found/#4711");
  EXPECT_EQ(Classify().cls, StoreDirClass::kEmpty);
  EXPECT_TRUE(Classify().lost_found_in_use);
  File("slab_meta");
  EXPECT_EQ(Classify().cls, StoreDirClass::kSlab);
  EXPECT_TRUE(Classify().lost_found_in_use);
  StoreDirClassification out;
  std::string error;
  ASSERT_TRUE(ClassifyStoreDir((dir_ / "not-yet").string(), &out, &error));
  EXPECT_EQ(out.cls, StoreDirClass::kEmpty) << "a directory not created yet";
}

TEST_F(StoreDirClassTest, SlabMarkersWinWhateverElseIsPresent) {
  for (const char* marker :
       {"slab_meta", "slots.tbl", "slab_state", "slab_initializing"}) {
    SCOPED_TRACE(marker);
    fs::remove_all(dir_);
    fs::create_directories(dir_);
    File(marker);
    Dir("lost+found");
    File("operator-notes.txt");
    const auto c = Classify();
    EXPECT_EQ(c.cls, StoreDirClass::kSlab);
    ASSERT_EQ(c.deciding.size(), 1u);
    EXPECT_EQ(c.deciding[0], marker);
  }
  fs::remove_all(dir_);
  fs::create_directories(dir_);
  Dir("extents");
  EXPECT_EQ(Classify().cls, StoreDirClass::kSlab);
}

TEST_F(StoreDirClassTest, LogMarkersWinWhateverElseIsPresent) {
  for (const char* marker : {"lse_super", "lse_journal", "lse_initializing"}) {
    SCOPED_TRACE(marker);
    fs::remove_all(dir_);
    fs::create_directories(dir_);
    File(marker);
    File("junk");
    EXPECT_EQ(Classify().cls, StoreDirClass::kLog);
  }
  fs::remove_all(dir_);
  fs::create_directories(dir_);
  Dir("segments");
  EXPECT_EQ(Classify().cls, StoreDirClass::kLog);
}

TEST_F(StoreDirClassTest, BlocksMarkTheFileEngine) {
  Dir("blocks");
  EXPECT_EQ(Classify().cls, StoreDirClass::kFile);
  File("slab_state");
  EXPECT_EQ(Classify().cls, StoreDirClass::kSlab) << "slab markers win";
}

TEST_F(StoreDirClassTest, BothKindsConflictAndUnknownIsForeign) {
  File("slab_meta");
  File("lse_super");
  auto c = Classify();
  EXPECT_EQ(c.cls, StoreDirClass::kConflict);
  EXPECT_EQ(c.deciding.size(), 2u);
  fs::remove_all(dir_);
  fs::create_directories(dir_);
  File("something");
  c = Classify();
  EXPECT_EQ(c.cls, StoreDirClass::kForeign);
  ASSERT_EQ(c.deciding.size(), 1u);
  EXPECT_EQ(c.deciding[0], "something");
}

TEST_F(StoreDirClassTest, APathThatIsNotADirectoryIsAnError) {
  File("plain");
  StoreDirClassification out;
  std::string error;
  EXPECT_FALSE(ClassifyStoreDir((dir_ / "plain").string(), &out, &error));
  EXPECT_FALSE(error.empty());
}
