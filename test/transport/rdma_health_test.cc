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

TEST(RdmaHealth, AnyUnhealthyDeviceRemovesNodeImmediately) {
  std::deque<std::vector<IbDeviceHealth>> samples{
      {Device("ib0", 4, 5), Device("ib1", 4, 5)},
      {Device("ib0", 4, 5), Device("ib1", 2, 2)}};
  rdma::RdmaHealthMonitor monitor([&] {
    auto sample = samples.front();
    samples.pop_front();
    return sample;
  });
  EXPECT_TRUE(monitor.Sample().ring_eligible);
  const MemberHealth degraded = monitor.Sample();
  EXPECT_FALSE(degraded.ring_eligible);
  ASSERT_EQ(degraded.ib_devices.size(), 2u);
  EXPECT_FALSE(degraded.ib_devices[1].healthy());
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
