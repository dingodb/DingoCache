#include "utils/numa_util.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <vector>

using dfkv::numa::CpuInList;
using dfkv::numa::CurrentNode;

TEST(CpuInList, Ranges) {
  EXPECT_TRUE(CpuInList("0-3,8-11", 2));
  EXPECT_TRUE(CpuInList("0-3,8-11", 0));
  EXPECT_TRUE(CpuInList("0-3,8-11", 3));
  EXPECT_TRUE(CpuInList("0-3,8-11", 8));
  EXPECT_TRUE(CpuInList("0-3,8-11", 11));
  EXPECT_TRUE(CpuInList("0-3,8-11", 9));
  EXPECT_FALSE(CpuInList("0-3,8-11", 5));
  EXPECT_FALSE(CpuInList("0-3,8-11", 12));
}

TEST(CpuInList, SingletonsAndEdges) {
  EXPECT_TRUE(CpuInList("5", 5));
  EXPECT_FALSE(CpuInList("5", 6));
  EXPECT_TRUE(CpuInList("48-95,144-191", 144));
  EXPECT_FALSE(CpuInList("48-95,144-191", 96));
  EXPECT_FALSE(CpuInList("", 0));
  EXPECT_FALSE(CpuInList(nullptr, 0));
}

TEST(CurrentNode, BestEffortStableAndValid) {
  int a = CurrentNode();
  EXPECT_GE(a, -1);          // -1 (unknown/single) or a real node id
  EXPECT_EQ(a, CurrentNode());  // stable across calls on a pinned/idle thread
}

// The cached table agrees with each node's sysfs cpulist.
TEST(CurrentNode, TableMatchesSysfsLists) {
  const std::vector<int>& table = dfkv::numa::CpuNodeTable();
  for (int node = 0; node < 256; ++node) {
    char path[256];
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/node/node%d/cpulist", node);
    FILE* f = std::fopen(path, "r");
    if (!f) continue;
    char buf[8192] = {0};
    const size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
    std::fclose(f);
    if (n == 0 || n == sizeof(buf) - 1) continue;
    for (int cpu = 0; cpu < 4096; ++cpu) {
      if (!dfkv::numa::CpuInList(buf, cpu)) continue;
      ASSERT_LT(static_cast<size_t>(cpu), table.size()) << cpu;
      EXPECT_EQ(table[static_cast<size_t>(cpu)], node) << cpu;
    }
  }
}

TEST(ForEachCpuInList, ListsRangesAndSingletons) {
  std::vector<int> cpus;
  dfkv::numa::ForEachCpuInList("0-2,5, 7-8\n", [&](int cpu) { cpus.push_back(cpu); });
  EXPECT_EQ(cpus, (std::vector<int>{0, 1, 2, 5, 7, 8}));
}
