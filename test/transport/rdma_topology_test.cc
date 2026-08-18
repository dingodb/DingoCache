#include "transport/rdma_topology.h"
#include "transport/rail_select.h"

#include <gtest/gtest.h>

#include <array>
#include <string>
#include <vector>

namespace {

using dfkv::rdma::RailLocality;
using dfkv::rdma::PreferPrimaryRail;
using dfkv::rdma::RailCandidates;
using dfkv::rdma::RdmaDevInfo;
using dfkv::rdma::RdmaDiscoveryPolicy;
using dfkv::rdma::RdmaDiscoveryProbe;
using dfkv::rdma::RdmaDiscoveryStatus;
using dfkv::rdma::RdmaTopology;
using dfkv::rdma::PeerTopologyStore;

RdmaDevInfo Device(std::string name, bool active, int numa_node = -1) {
  RdmaDevInfo info;
  info.name = std::move(name);
  info.active = active;
  info.numa_node = numa_node;
  return info;
}

RdmaDiscoveryProbe Probe(
    std::string name, bool active,
    RdmaDiscoveryStatus status = RdmaDiscoveryStatus::kOk) {
  RdmaDiscoveryProbe probe;
  probe.device = Device(std::move(name), active);
  probe.status = status;
  return probe;
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

TEST(RdmaTopology, TypedActiveOnlyResolutionFiltersInactiveAndFailedProbes) {
  const std::vector<RdmaDiscoveryProbe> probes{
      Probe("ib0", false), Probe("ib1", true),
      Probe("ib2", false, RdmaDiscoveryStatus::kDeviceOpenFailed),
      Probe("ib3", true)};

  const auto result = RdmaTopology::ResolveDiscovery(
      probes, {}, RdmaDiscoveryPolicy::kActiveOnly);

  ASSERT_TRUE(result.ok());
  ASSERT_EQ(result.devices.size(), 2u);
  EXPECT_EQ(result.devices[0].name, "ib1");
  EXPECT_EQ(result.devices[1].name, "ib3");
}

TEST(RdmaTopology, ActiveOnlyPreservesLegacyGidQueryFailureBehavior) {
  const std::vector<RdmaDiscoveryProbe> probes{
      Probe("ib0", true, RdmaDiscoveryStatus::kGidQueryFailed),
      Probe("ib1", true)};

  const auto result = RdmaTopology::ResolveDiscovery(
      probes, {}, RdmaDiscoveryPolicy::kActiveOnly);

  ASSERT_TRUE(result.ok());
  ASSERT_EQ(result.devices.size(), 2u);
  EXPECT_EQ(result.devices[0].name, "ib0");
  EXPECT_EQ(result.devices[0].gid, (std::array<uint8_t, 16>{}));
  EXPECT_EQ(result.devices[1].name, "ib1");
}

TEST(RdmaTopology, AllowInactiveRetainsStateInConfiguredOrder) {
  const std::vector<RdmaDiscoveryProbe> probes{
      Probe("ib0", true), Probe("ib1", false), Probe("ib2", true)};
  const std::vector<std::string> configured{"ib2", "ib1", "ib0"};

  const auto result = RdmaTopology::ResolveDiscovery(
      probes, configured, RdmaDiscoveryPolicy::kAllowInactive);

  ASSERT_TRUE(result.ok());
  ASSERT_EQ(result.devices.size(), 3u);
  EXPECT_EQ(result.devices[0].name, "ib2");
  EXPECT_TRUE(result.devices[0].active);
  EXPECT_EQ(result.devices[1].name, "ib1");
  EXPECT_FALSE(result.devices[1].active);
  EXPECT_EQ(result.devices[2].name, "ib0");
  EXPECT_TRUE(result.devices[2].active);
}

TEST(RdmaTopology, AllowInactiveUsesFirstConfiguredOccurrence) {
  const std::vector<RdmaDiscoveryProbe> probes{
      Probe("ib0", true), Probe("ib1", false)};
  const std::vector<std::string> configured{"ib1", "ib0", "ib1", "ib0"};

  const auto result = RdmaTopology::ResolveDiscovery(
      probes, configured, RdmaDiscoveryPolicy::kAllowInactive);

  ASSERT_TRUE(result.ok());
  ASSERT_EQ(result.devices.size(), 2u);
  EXPECT_EQ(result.devices[0].name, "ib1");
  EXPECT_EQ(result.devices[1].name, "ib0");
}

TEST(RdmaTopology, AllowInactiveFailsClosedForMissingConfiguredDevice) {
  const std::vector<RdmaDiscoveryProbe> probes{Probe("ib0", true)};

  const auto result = RdmaTopology::ResolveDiscovery(
      probes, {"ib0", "ib-missing"}, RdmaDiscoveryPolicy::kAllowInactive);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status, RdmaDiscoveryStatus::kConfiguredDeviceMissing);
  EXPECT_EQ(result.failed_device, "ib-missing");
  EXPECT_TRUE(result.devices.empty());
}

TEST(RdmaTopology, AllowInactiveFailsClosedForDeviceOpenFailure) {
  const std::vector<RdmaDiscoveryProbe> probes{
      Probe("ib0", true),
      Probe("ib1", false, RdmaDiscoveryStatus::kDeviceOpenFailed)};

  const auto result = RdmaTopology::ResolveDiscovery(
      probes, {"ib0", "ib1"}, RdmaDiscoveryPolicy::kAllowInactive);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status, RdmaDiscoveryStatus::kDeviceOpenFailed);
  EXPECT_EQ(result.failed_device, "ib1");
  EXPECT_TRUE(result.devices.empty());
}

TEST(RdmaTopology, AllowInactiveFailsClosedForPortQueryFailure) {
  const std::vector<RdmaDiscoveryProbe> probes{
      Probe("ib0", false, RdmaDiscoveryStatus::kPortQueryFailed),
      Probe("ib1", true)};

  const auto result = RdmaTopology::ResolveDiscovery(
      probes, {"ib0", "ib1"}, RdmaDiscoveryPolicy::kAllowInactive);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status, RdmaDiscoveryStatus::kPortQueryFailed);
  EXPECT_EQ(result.failed_device, "ib0");
  EXPECT_TRUE(result.devices.empty());
}

TEST(RdmaTopology, AllowInactiveFailsClosedForGidQueryFailure) {
  const std::vector<RdmaDiscoveryProbe> probes{
      Probe("ib0", true),
      Probe("ib1", true, RdmaDiscoveryStatus::kGidQueryFailed)};

  const auto result = RdmaTopology::ResolveDiscovery(
      probes, {"ib0", "ib1"}, RdmaDiscoveryPolicy::kAllowInactive);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status, RdmaDiscoveryStatus::kGidQueryFailed);
  EXPECT_EQ(result.failed_device, "ib1");
  EXPECT_TRUE(result.devices.empty());
  ASSERT_EQ(result.observed_devices.size(), 1u);
  EXPECT_EQ(result.observed_devices[0].name, "ib0");
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

TEST(RdmaTopology, CandidateMaskContainsOnlyCallerLocalRails) {
  RdmaTopology topology({Device("ib0", true, 0), Device("ib1", true, 1),
                         Device("ib2", true, 1)});

  const auto candidates = topology.CandidatesFor(1, true);

  EXPECT_EQ(candidates.locality, RailLocality::kLocal);
  EXPECT_EQ(candidates.allowed, (std::vector<uint8_t>{0, 1, 1}));
}

TEST(RdmaTopology, UnknownCallerFallsBackToAllEnabledRails) {
  RdmaTopology topology({Device("ib0", true, 0), Device("ib1", true, 1)});
  topology.DisableDevice("ib1");

  const auto candidates = topology.CandidatesFor(-1, true);

  EXPECT_EQ(candidates.locality, RailLocality::kCallerUnknown);
  EXPECT_EQ(candidates.allowed, (std::vector<uint8_t>{1, 0}));
}

TEST(RdmaTopology, NoLocalRailFallsBackToAllEnabledRails) {
  RdmaTopology topology(
      {Device("ib0", true, 0), Device("ib1", true, -1)});

  const auto candidates = topology.CandidatesFor(1, true);

  EXPECT_EQ(candidates.locality, RailLocality::kNoLocal);
  EXPECT_EQ(candidates.allowed, (std::vector<uint8_t>{1, 1}));
}

TEST(RdmaTopology, LocalMaskCarriesAllEnabledRailsAsFallback) {
  RdmaTopology topology({Device("ib0", true, 0), Device("ib1", true, 0),
                         Device("ib2", true, 1)});

  const auto candidates = topology.CandidatesFor(0, true);

  EXPECT_EQ(candidates.locality, RailLocality::kLocal);
  EXPECT_EQ(candidates.allowed, (std::vector<uint8_t>{1, 1, 0}));
  // The backstop spans every enabled rail, NUMA locality aside.
  EXPECT_EQ(candidates.fallback, (std::vector<uint8_t>{1, 1, 1}));
}

TEST(RdmaTopology, FallbackMaskStillExcludesRuntimeDisabledRails) {
  RdmaTopology topology({Device("ib0", true, 0), Device("ib1", true, 1)});
  topology.DisableDevice("ib1");

  const auto candidates = topology.CandidatesFor(0, true);

  EXPECT_EQ(candidates.locality, RailLocality::kLocal);
  EXPECT_EQ(candidates.allowed, (std::vector<uint8_t>{1, 0}));
  EXPECT_EQ(candidates.fallback, (std::vector<uint8_t>{1, 0}));
}

TEST(RdmaTopology, FallbackMaskIsEmptyWithoutLocalPreference) {
  RdmaTopology topology({Device("ib0", true, 0), Device("ib1", true, 1)});

  const auto disabled = topology.CandidatesFor(0, false);
  EXPECT_EQ(disabled.locality, RailLocality::kDisabled);
  EXPECT_TRUE(disabled.fallback.empty());

  const auto unknown = topology.CandidatesFor(-1, true);
  EXPECT_EQ(unknown.locality, RailLocality::kCallerUnknown);
  EXPECT_TRUE(unknown.fallback.empty());
}

TEST(RdmaTopology, PreferredRailKeepsOnePrimaryAndBoundedFallbacks) {
  RailCandidates candidates;
  candidates.allowed = {1, 1, 0, 1};
  const auto preferred = PreferPrimaryRail(candidates, 1);
  EXPECT_EQ(preferred.allowed, (std::vector<uint8_t>{0, 1, 0, 0}));
  EXPECT_EQ(preferred.fallback, (std::vector<uint8_t>{1, 0, 0, 1}));

  const auto disabled_primary = PreferPrimaryRail(candidates, 2);
  EXPECT_EQ(disabled_primary.allowed, (std::vector<uint8_t>{0, 0, 0, 0}));
  EXPECT_EQ(disabled_primary.fallback,
            (std::vector<uint8_t>{1, 1, 0, 1}));
}


TEST(RdmaTopology, StableIdentityMovesAddressAndGuardsRetirement) {
  PeerTopologyStore store({"ib0"}, {{1}}, /*require_complete=*/true);
  dfkv::PeerTopology first;
  first.peer_addr = "10.0.0.1:1000";
  first.peer_id = "stable-peer";
  first.generation = 11;
  first.complete = true;
  first.rails.push_back(dfkv::PeerRailTopology{"ib0", true});
  ASSERT_TRUE(store.Update(first));
  auto snapshot = store.Snapshot(first.peer_addr);
  EXPECT_EQ(snapshot->peer_id, first.peer_id);
  EXPECT_EQ(snapshot->generation, 11u);
  EXPECT_EQ(snapshot->compatible, (std::vector<uint8_t>{1}));

  dfkv::PeerTopology moved = first;
  moved.peer_addr = "10.0.0.2:2000";
  moved.generation = 3;
  ASSERT_TRUE(store.Update(moved));
  snapshot = store.Snapshot(moved.peer_addr);
  EXPECT_EQ(snapshot->peer_id, first.peer_id);
  EXPECT_EQ(snapshot->generation, 3u);
  EXPECT_EQ(snapshot->compatible, (std::vector<uint8_t>{1}));
  EXPECT_FALSE(store.Snapshot(first.peer_addr)->complete)
      << "the superseded address must not retain a second live binding";

  dfkv::PeerTopology stale_removal = moved;
  stale_removal.present = false;
  stale_removal.peer_id = "replaced-peer";
  EXPECT_FALSE(store.Update(stale_removal));
  EXPECT_EQ(store.Snapshot(moved.peer_addr)->peer_id, "stable-peer")
      << "late retirement for another identity must not remove the replacement";

  stale_removal.peer_id = "stable-peer";
  EXPECT_TRUE(store.Update(stale_removal));
  EXPECT_FALSE(store.Snapshot(moved.peer_addr)->complete);
}

TEST(RdmaTopology,
     RemoveAndIdenticalReaddPublishesANewTransportIncarnation) {
  PeerTopologyStore store({"ib0"}, {{1}}, /*require_complete=*/true);
  dfkv::PeerTopology topology;
  topology.peer_addr = "10.0.0.1:1000";
  topology.peer_id = "stable-peer";
  topology.generation = 77;
  topology.complete = true;
  topology.rails.push_back(dfkv::PeerRailTopology{"ib0", true});

  ASSERT_TRUE(store.Update(topology));
  const auto first = store.Snapshot(topology.peer_addr);
  ASSERT_EQ(first->generation, 77u);
  ASSERT_NE(first->publication, 0u);
  EXPECT_TRUE(store.IsCurrent(topology.peer_addr, topology.peer_id,
                              first->publication));
  EXPECT_FALSE(store.Update(topology))
      << "the opaque content generation remains equality-only";

  topology.present = false;
  ASSERT_TRUE(store.Update(topology));
  EXPECT_FALSE(store.IsCurrent(topology.peer_addr, topology.peer_id,
                               first->publication));

  topology.present = true;
  ASSERT_TRUE(store.Update(topology));
  const auto reincarnated = store.Snapshot(topology.peer_addr);
  EXPECT_EQ(reincarnated->generation, first->generation);
  EXPECT_GT(reincarnated->publication, first->publication);
  EXPECT_FALSE(store.IsCurrent(topology.peer_addr, topology.peer_id,
                               first->publication));
  EXPECT_TRUE(store.IsCurrent(topology.peer_addr, topology.peer_id,
                              reincarnated->publication));
}

}  // namespace
