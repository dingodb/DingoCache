#include "transport/rdma_health.h"

#include <gtest/gtest.h>

#include <deque>
#include <vector>

using namespace dfkv;  // NOLINT

namespace {

IbDeviceHealth Device(const char* name, uint8_t port, uint8_t phys,
                      bool query_ok = true) {
  return IbDeviceHealth{name, port, phys, query_ok};
}

}  // namespace

TEST(RdmaHealth, AnyHealthyDeviceKeepsNodeEligible) {
  std::deque<std::vector<IbDeviceHealth>> samples{
      {Device("ib0", 4, 5), Device("ib1", 4, 5)},
      {Device("ib0", 4, 5), Device("ib1", 2, 2)}};
  rdma::RdmaHealthMonitor monitor([&] {
    auto sample = samples.front();
    samples.pop_front();
    return sample;
  });
  EXPECT_TRUE(monitor.Sample().ring_eligible);
  const MemberHealth partially_degraded = monitor.Sample();
  EXPECT_TRUE(partially_degraded.ring_eligible);
  ASSERT_EQ(partially_degraded.ib_devices.size(), 2u);
  EXPECT_TRUE(partially_degraded.ib_devices[0].healthy());
  EXPECT_FALSE(partially_degraded.ib_devices[1].healthy());
  const std::string metrics = monitor.MetricsText();
  EXPECT_NE(metrics.find("dfkv_server_rdma_rails_configured 2"),
            std::string::npos);
  EXPECT_NE(metrics.find("dfkv_server_rdma_rails_active 1"),
            std::string::npos);
}

TEST(RdmaHealth, ZeroActiveDevicesRemovesNodeImmediately) {
  std::deque<std::vector<IbDeviceHealth>> samples{
      {Device("ib0", 4, 5), Device("ib1", 2, 2)},
      {Device("ib0", 1, 2), Device("ib1", 2, 2)}};
  rdma::RdmaHealthMonitor monitor([&] {
    auto sample = samples.front();
    samples.pop_front();
    return sample;
  });
  EXPECT_TRUE(monitor.Sample().ring_eligible);
  EXPECT_FALSE(monitor.Sample().ring_eligible);
}

TEST(RdmaHealth, RecoveryRequiresConsecutiveHealthySamples) {
  std::deque<std::vector<IbDeviceHealth>> samples{
      {Device("ib0", 1, 2)}, {Device("ib0", 4, 5)},
      {Device("ib0", 1, 2)}, {Device("ib0", 4, 5)},
      {Device("ib0", 4, 5)}, {Device("ib0", 4, 5)}};
  rdma::RdmaHealthMonitor monitor([&] {
    auto sample = samples.front();
    samples.pop_front();
    return sample;
  }, 3);
  EXPECT_FALSE(monitor.Sample().ring_eligible);
  EXPECT_FALSE(monitor.Sample().ring_eligible);
  EXPECT_FALSE(monitor.Sample().ring_eligible);
  EXPECT_FALSE(monitor.Sample().ring_eligible);
  EXPECT_FALSE(monitor.Sample().ring_eligible);
  EXPECT_TRUE(monitor.Sample().ring_eligible);
}

TEST(RdmaHealth, SingleRailUsesTheSameRemovalAndRecoveryLifecycle) {
  std::deque<std::vector<IbDeviceHealth>> samples{
      {Device("ib0", 4, 5)}, {Device("ib0", 1, 2)},
      {Device("ib0", 4, 5)}, {Device("ib0", 4, 5)}};
  rdma::RdmaHealthMonitor monitor([&] {
    auto sample = samples.front();
    samples.pop_front();
    return sample;
  }, 2);
  EXPECT_TRUE(monitor.Sample().ring_eligible);
  EXPECT_FALSE(monitor.Sample().ring_eligible);
  EXPECT_FALSE(monitor.Sample().ring_eligible);
  EXPECT_TRUE(monitor.Sample().ring_eligible);
}

TEST(RdmaHealth, QueryFailureFailsClosed) {
  rdma::RdmaHealthMonitor monitor(
      [] { return std::vector<IbDeviceHealth>{Device("ib0", 0, 0, false)}; });
  const MemberHealth health = monitor.Sample();
  EXPECT_FALSE(health.ring_eligible);
  EXPECT_NE(monitor.MetricsText().find("dfkv_server_ring_eligible 0"),
            std::string::npos);
  EXPECT_NE(monitor.MetricsText().find("device=\"ib0\"} 0"),
            std::string::npos);
}
