#include "transport/rdma_topology.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

using dfkv::rdma::RdmaDevInfo;
using dfkv::rdma::RdmaTopology;

RdmaDevInfo Device(std::string name, bool active, int numa_node = -1) {
  RdmaDevInfo info;
  info.name = std::move(name);
  info.active = active;
  info.numa_node = numa_node;
  return info;
}

TEST(RdmaTopology, DiscoveryDropsDownPortsWithoutAFilter) {
  const std::vector<RdmaDevInfo> candidates{
      Device("ib7s400p0", true, 0), Device("ib9s400p0", false, 1),
      Device("ib7s400p1", true, 1)};

  const auto selected = RdmaTopology::FilterActive(candidates);

  ASSERT_EQ(selected.size(), 2u);
  EXPECT_EQ(selected[0].name, "ib7s400p0");
  EXPECT_EQ(selected[1].name, "ib7s400p1");
}

TEST(RdmaTopology, WhitelistAndActiveChecksAreBothApplied) {
  const std::vector<RdmaDevInfo> candidates{
      Device("ib7s400p0", true), Device("ib7s400p1", false),
      Device("ib6s200p0", true)};
  const std::vector<std::string> filter{"ib7s400p0", "ib7s400p1"};

  const auto selected = RdmaTopology::FilterActive(candidates, filter);

  ASSERT_EQ(selected.size(), 1u);
  EXPECT_EQ(selected[0].name, "ib7s400p0");
}

TEST(RdmaTopology, AllFilteredPortsDownReturnsEmpty) {
  const std::vector<RdmaDevInfo> candidates{
      Device("ib9s400p0", false), Device("ib9s400p1", false)};
  const std::vector<std::string> filter{"ib9s400p0", "ib9s400p1"};

  EXPECT_TRUE(RdmaTopology::FilterActive(candidates, filter).empty());
}

TEST(RdmaTopology, DuplicateCandidatesRemainOneRail) {
  const std::vector<RdmaDevInfo> candidates{
      Device("ib7s400p0", true), Device("ib7s400p0", true)};

  const auto selected = RdmaTopology::FilterActive(candidates);

  ASSERT_EQ(selected.size(), 1u);
  EXPECT_EQ(selected[0].name, "ib7s400p0");
}

TEST(RdmaTopology, SelectsLocalRailsRoundRobin) {
  RdmaTopology topology({Device("ib0", true, 0), Device("ib1", true, 1),
                         Device("ib2", true, 1)});

  const int first = topology.SelectDevice(1, true);
  const int second = topology.SelectDevice(1, true);
  const int third = topology.SelectDevice(1, true);

  EXPECT_EQ(first, 1);
  EXPECT_EQ(second, 2);
  EXPECT_EQ(third, 1);
}

TEST(RdmaTopology, DisabledRailIsNotSelectedAgain) {
  RdmaTopology topology(
      {Device("ib0", true, 0), Device("ib1", true, 0)});
  topology.DisableDevice("ib0");

  for (int i = 0; i < 4; ++i) EXPECT_EQ(topology.SelectDevice(0, true), 1);
  topology.DisableDevice("ib1");
  EXPECT_EQ(topology.SelectDevice(0, true), -1);
}

}  // namespace
