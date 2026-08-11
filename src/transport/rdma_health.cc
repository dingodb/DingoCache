#include "transport/rdma_health.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "utils/log.h"
#include "utils/prom_escape.h"

namespace dfkv::rdma {
namespace {

bool ReadState(const std::string& path, uint8_t* value) {
  std::ifstream input(path);
  unsigned int parsed = 0;
  if (!(input >> parsed) || parsed > 255) return false;
  *value = static_cast<uint8_t>(parsed);
  return true;
}

std::vector<IbDeviceHealth> ProbeOverride(
    const std::vector<std::string>& devices, const std::string& path) {
  std::ifstream input(path);
  std::unordered_map<std::string, IbDeviceHealth> reported;
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream fields(line);
    IbDeviceHealth health;
    unsigned int port = 0;
    unsigned int phys = 0;
    if (!(fields >> health.name >> port >> phys) || port > 255 || phys > 255)
      continue;
    health.port_state = static_cast<uint8_t>(port);
    health.phys_state = static_cast<uint8_t>(phys);
    health.query_ok = true;
    reported[health.name] = health;
  }
  std::vector<IbDeviceHealth> out;
  out.reserve(devices.size());
  for (const auto& name : devices) {
    auto it = reported.find(name);
    if (it != reported.end()) out.push_back(it->second);
    else out.push_back(IbDeviceHealth{name, 0, 0, false});
  }
  return out;
}

}  // namespace

std::vector<IbDeviceHealth> ProbeIbDeviceHealth(
    const std::vector<std::string>& devices,
    const std::string& override_file) {
  if (!override_file.empty()) return ProbeOverride(devices, override_file);
  std::vector<IbDeviceHealth> out;
  out.reserve(devices.size());
  for (const auto& name : devices) {
    IbDeviceHealth health;
    health.name = name;
    const std::string base = "/sys/class/infiniband/" + name + "/ports/1/";
    health.query_ok = ReadState(base + "state", &health.port_state) &&
                      ReadState(base + "phys_state", &health.phys_state);
    out.push_back(std::move(health));
  }
  return out;
}

RdmaHealthMonitor::RdmaHealthMonitor(ProbeFn probe,
                                     uint32_t recovery_samples)
    : probe_(std::move(probe)),
      recovery_samples_(std::max<uint32_t>(1, recovery_samples)) {}

MemberHealth RdmaHealthMonitor::Sample() {
  std::vector<IbDeviceHealth> devices = probe_ ? probe_() : std::vector<IbDeviceHealth>{};
  const bool all_healthy = !devices.empty() &&
      std::all_of(devices.begin(), devices.end(),
                  [](const IbDeviceHealth& d) { return d.healthy(); });
  std::lock_guard<std::mutex> lock(mu_);
  const bool was_eligible = last_.ring_eligible;
  if (!all_healthy) {
    healthy_streak_ = 0;
    last_.ring_eligible = false;
  } else if (!sampled_ || last_.ring_eligible) {
    healthy_streak_ = recovery_samples_;
    last_.ring_eligible = true;
  } else if (++healthy_streak_ >= recovery_samples_) {
    last_.ring_eligible = true;
  }
  last_.ib_devices = std::move(devices);
  sampled_ = true;
  if (last_.ring_eligible != was_eligible) {
    DFKV_LOG_WARN(std::string("rdma-health: node ") +
                  (last_.ring_eligible ? "rejoined" : "left") +
                  " placement ring after IB health transition");
  }
  return last_;
}

MemberHealth RdmaHealthMonitor::Last() const {
  std::lock_guard<std::mutex> lock(mu_);
  return last_;
}

std::string RdmaHealthMonitor::MetricsText() const {
  const MemberHealth health = Last();
  std::string out =
      "# HELP dfkv_server_ring_eligible Whether IB health permits placement ring membership\n"
      "# TYPE dfkv_server_ring_eligible gauge\n"
      "dfkv_server_ring_eligible " +
      std::string(health.ring_eligible ? "1\n" : "0\n") +
      "# HELP dfkv_server_ib_device_healthy Whether an IB device port is ACTIVE and LinkUp\n"
      "# TYPE dfkv_server_ib_device_healthy gauge\n";
  for (const auto& device : health.ib_devices) {
    out += "dfkv_server_ib_device_healthy{device=\"" +
           PromLabelEscape(device.name) + "\"} " +
           (device.healthy() ? "1\n" : "0\n");
  }
  return out;
}

}  // namespace dfkv::rdma
