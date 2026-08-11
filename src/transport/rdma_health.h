#ifndef DFKV_TRANSPORT_RDMA_HEALTH_H_
#define DFKV_TRANSPORT_RDMA_HEALTH_H_

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "common/membership.h"

namespace dfkv::rdma {

std::vector<IbDeviceHealth> ProbeIbDeviceHealth(
    const std::vector<std::string>& devices,
    const std::string& override_file = {});

class RdmaHealthMonitor {
 public:
  using ProbeFn = std::function<std::vector<IbDeviceHealth>()>;

  explicit RdmaHealthMonitor(ProbeFn probe, uint32_t recovery_samples = 3);

  MemberHealth Sample();
  MemberHealth Last() const;
  std::string MetricsText() const;

 private:
  ProbeFn probe_;
  uint32_t recovery_samples_;
  mutable std::mutex mu_;
  MemberHealth last_;
  uint32_t healthy_streak_ = 0;
  bool sampled_ = false;
};

}  // namespace dfkv::rdma

#endif  // DFKV_TRANSPORT_RDMA_HEALTH_H_
