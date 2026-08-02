#include "transport/endpoint_schedule.h"

#include <gtest/gtest.h>

#include <vector>

namespace dfkv {
namespace {

TEST(EndpointSchedule, LeastInflightWinsBeforeLatency) {
  const std::vector<EndpointConnectionLoad> loads = {
      {1, 2, 10, true}, {2, 0, 900, true}, {3, 1, 1, true}};
  EXPECT_EQ(SelectEndpointConnection(loads), 1u);
}

TEST(EndpointSchedule, SamplesNewConnectionThenUsesLowestLatency) {
  std::vector<EndpointConnectionLoad> loads = {
      {10, 0, 500, true}, {11, 0, 0, true}, {12, 0, 200, true}};
  EXPECT_EQ(SelectEndpointConnection(loads), 1u);
  loads[1].latency_ns = 700;
  EXPECT_EQ(SelectEndpointConnection(loads), 2u);
}

TEST(EndpointSchedule, StableIdBreaksExactTieDeterministically) {
  const std::vector<EndpointConnectionLoad> loads = {
      {90, 0, 100, true}, {7, 0, 100, true}, {42, 0, 100, true}};
  EXPECT_EQ(SelectEndpointConnection(loads), 1u);
}

TEST(EndpointSchedule, UnhealthyConnectionsAreNeverSelected) {
  const std::vector<EndpointConnectionLoad> loads = {
      {1, 0, 1, false}, {2, 3, 100, true}};
  EXPECT_EQ(SelectEndpointConnection(loads), 1u);
  EXPECT_EQ(SelectEndpointConnection({}), 0u);
}

}  // namespace
}  // namespace dfkv
