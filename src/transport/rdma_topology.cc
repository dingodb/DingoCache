#include "transport/rdma_topology.h"

#include <infiniband/verbs.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <utility>

#include "utils/log.h"
#include "utils/numa_util.h"

namespace dfkv::rdma {
namespace {

bool Contains(const std::vector<std::string>& names, const std::string& name) {
  return names.empty() ||
         std::find(names.begin(), names.end(), name) != names.end();
}

}  // namespace

RdmaTopology::RdmaTopology(std::vector<RdmaDevInfo> devices)
    : devices_(std::move(devices)), enabled_(devices_.size(), 1) {}

std::vector<RdmaDevInfo> RdmaTopology::FilterActive(
    const std::vector<RdmaDevInfo>& candidates,
    const std::vector<std::string>& filter) {
  std::vector<RdmaDevInfo> out;
  out.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    if (!candidate.active || !Contains(filter, candidate.name)) continue;
    const bool duplicate = std::any_of(
        out.begin(), out.end(), [&](const RdmaDevInfo& selected) {
          return selected.name == candidate.name;
        });
    if (!duplicate) out.push_back(candidate);
  }
  return out;
}

RdmaDiscoveryResult RdmaTopology::ResolveDiscovery(
    const std::vector<RdmaDiscoveryProbe>& probes,
    const std::vector<std::string>& filter, RdmaDiscoveryPolicy policy) {
  RdmaDiscoveryResult result;
  if (policy == RdmaDiscoveryPolicy::kActiveOnly) {
    std::vector<RdmaDevInfo> candidates;
    candidates.reserve(probes.size());
    for (const auto& probe : probes) {
      if (probe.status == RdmaDiscoveryStatus::kOk ||
          probe.status == RdmaDiscoveryStatus::kGidQueryFailed) {
        candidates.push_back(probe.device);
      }
    }
    result.devices = FilterActive(candidates, filter);
    return result;
  }

  const auto fail = [&](RdmaDiscoveryStatus status,
                        const std::string& device) {
    result.observed_devices = std::move(result.devices);
    result.devices.clear();
    result.status = status;
    result.failed_device = device;
  };
  if (filter.empty()) {
    result.devices.reserve(probes.size());
    for (const auto& probe : probes) {
      if (probe.status != RdmaDiscoveryStatus::kOk) {
        fail(probe.status, probe.device.name);
        return result;
      }
      const bool duplicate = std::any_of(
          result.devices.begin(), result.devices.end(),
          [&](const RdmaDevInfo& selected) {
            return selected.name == probe.device.name;
          });
      if (!duplicate) result.devices.push_back(probe.device);
    }
    return result;
  }

  result.devices.reserve(filter.size());
  for (const auto& requested : filter) {
    const bool duplicate = std::any_of(
        result.devices.begin(), result.devices.end(),
        [&](const RdmaDevInfo& selected) {
          return selected.name == requested;
        });
    if (duplicate) continue;

    const auto probe = std::find_if(
        probes.begin(), probes.end(), [&](const RdmaDiscoveryProbe& candidate) {
          return candidate.device.name == requested;
        });
    if (probe == probes.end()) {
      fail(RdmaDiscoveryStatus::kConfiguredDeviceMissing, requested);
      return result;
    }
    if (probe->status != RdmaDiscoveryStatus::kOk) {
      fail(probe->status, requested);
      return result;
    }
    result.devices.push_back(probe->device);
  }
  return result;
}

std::vector<RdmaDevInfo> RdmaTopology::Discover(
    const std::vector<std::string>& filter) {
  return Discover(filter, RdmaDiscoveryPolicy::kActiveOnly).devices;
}

RdmaDiscoveryResult RdmaTopology::Discover(
    const std::vector<std::string>& filter, RdmaDiscoveryPolicy policy) {
  int count = 0;
  ibv_device** list = ibv_get_device_list(&count);
  if (!list) {
    DFKV_LOG_WARN("rdma: ibv_get_device_list failed");
    RdmaDiscoveryResult result;
    result.status = RdmaDiscoveryStatus::kDeviceListFailed;
    return result;
  }
  if (count == 0) DFKV_LOG_WARN("rdma: no verbs devices found");

  std::vector<RdmaDiscoveryProbe> probes;
  probes.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    const char* raw_name = ibv_get_device_name(list[i]);
    if (!raw_name) continue;
    const std::string name(raw_name);
    if (!Contains(filter, name)) continue;

    RdmaDiscoveryProbe probe;
    probe.device.name = name;
    std::unique_ptr<ibv_context, decltype(&ibv_close_device)> context(
        ibv_open_device(list[i]), &ibv_close_device);
    if (!context) {
      DFKV_LOG_WARN("rdma: skipping device " + name +
                    ": ibv_open_device failed");
      probe.status = RdmaDiscoveryStatus::kDeviceOpenFailed;
      probes.push_back(std::move(probe));
      continue;
    }

    ibv_port_attr port{};
    if (ibv_query_port(context.get(), 1, &port) != 0) {
      DFKV_LOG_WARN("rdma: skipping device " + name +
                    ": ibv_query_port(1) failed");
      probe.status = RdmaDiscoveryStatus::kPortQueryFailed;
      probes.push_back(std::move(probe));
      continue;
    }

    probe.device.numa_node = numa::DeviceNode(name.c_str());
    probe.device.lid = port.lid;
    probe.device.active = port.state == IBV_PORT_ACTIVE;
    union ibv_gid gid {};
    if (ibv_query_gid(context.get(), 1, 0, &gid) != 0) {
      DFKV_LOG_WARN("rdma: device " + name +
                    ": ibv_query_gid(1, 0) failed");
      probe.status = RdmaDiscoveryStatus::kGidQueryFailed;
    } else {
      std::memcpy(probe.device.gid.data(), gid.raw, probe.device.gid.size());
    }

    if (!probe.device.active) {
      DFKV_LOG_WARN("rdma: device " + name + ": port 1 state=" +
                    std::to_string(static_cast<int>(port.state)) +
                    " (ACTIVE=" +
                    std::to_string(static_cast<int>(IBV_PORT_ACTIVE)) + ")");
    }
    probes.push_back(std::move(probe));
  }
  ibv_free_device_list(list);

  auto result = ResolveDiscovery(probes, filter, policy);
  if (policy == RdmaDiscoveryPolicy::kActiveOnly) {
    for (const auto& requested : filter) {
      const bool found = std::any_of(
          result.devices.begin(), result.devices.end(),
          [&](const RdmaDevInfo& device) {
            return device.name == requested;
          });
      if (!found)
        DFKV_LOG_WARN("rdma: configured device " + requested +
                      " is absent or not ACTIVE");
    }
  }
  return result;
}

RailCandidates RdmaTopology::CandidatesFor(int numa_node,
                                           bool numa_aware) const {
  std::lock_guard<std::mutex> lock(mu_);
  RailCandidates out;
  out.allowed = enabled_;
  if (!numa_aware) return out;
  if (numa_node < 0) {
    out.locality = RailLocality::kCallerUnknown;
    return out;
  }

  bool found_local = false;
  for (size_t i = 0; i < devices_.size(); ++i) {
    if (enabled_[i] && devices_[i].numa_node == numa_node) {
      found_local = true;
      break;
    }
  }
  if (!found_local) {
    out.locality = RailLocality::kNoLocal;
    return out;
  }

  out.locality = RailLocality::kLocal;
  out.fallback = enabled_;
  for (size_t i = 0; i < devices_.size(); ++i) {
    if (devices_[i].numa_node != numa_node) out.allowed[i] = 0;
  }
  return out;
}

int RdmaTopology::SelectDevice(int numa_node, bool numa_aware,
                               size_t retry_count) const {
  std::lock_guard<std::mutex> lock(mu_);
  size_t local_count = 0;
  size_t enabled_count = 0;
  for (size_t i = 0; i < devices_.size(); ++i) {
    if (!enabled_[i]) continue;
    ++enabled_count;
    if (numa_aware && numa_node >= 0 && devices_[i].numa_node == numa_node)
      ++local_count;
  }
  if (enabled_count == 0) return -1;

  const bool use_local = local_count != 0;
  const size_t choice_count = use_local ? local_count : enabled_count;
  const size_t choice =
      (rr_.fetch_add(1, std::memory_order_relaxed) + retry_count) % choice_count;
  size_t seen = 0;
  for (size_t i = 0; i < devices_.size(); ++i) {
    if (!enabled_[i]) continue;
    if (use_local && devices_[i].numa_node != numa_node) continue;
    if (seen++ == choice) return static_cast<int>(i);
  }
  return -1;
}

void RdmaTopology::DisableDevice(const std::string& name) {
  std::lock_guard<std::mutex> lock(mu_);
  for (size_t i = 0; i < devices_.size(); ++i) {
    if (devices_[i].name == name) enabled_[i] = 0;
  }
}

}  // namespace dfkv::rdma
