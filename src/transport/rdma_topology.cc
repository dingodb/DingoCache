#include "transport/rdma_topology.h"

#include <infiniband/verbs.h>

#include <algorithm>
#include <cstring>
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

std::vector<RdmaDevInfo> RdmaTopology::Discover(
    const std::vector<std::string>& filter) {
  int count = 0;
  ibv_device** list = ibv_get_device_list(&count);
  if (!list || count == 0) {
    if (list) ibv_free_device_list(list);
    DFKV_LOG_WARN("rdma: no verbs devices found");
    return {};
  }

  std::vector<RdmaDevInfo> candidates;
  candidates.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    const char* raw_name = ibv_get_device_name(list[i]);
    if (!raw_name) continue;
    const std::string name(raw_name);
    if (!Contains(filter, name)) continue;

    ibv_context* context = ibv_open_device(list[i]);
    if (!context) {
      DFKV_LOG_WARN("rdma: skipping device " + name +
                    ": ibv_open_device failed");
      continue;
    }

    ibv_port_attr port{};
    if (ibv_query_port(context, 1, &port) != 0) {
      DFKV_LOG_WARN("rdma: skipping device " + name +
                    ": ibv_query_port(1) failed");
      ibv_close_device(context);
      continue;
    }

    RdmaDevInfo info;
    info.name = name;
    info.numa_node = numa::DeviceNode(name.c_str());
    info.lid = port.lid;
    info.active = port.state == IBV_PORT_ACTIVE;
    union ibv_gid gid{};
    if (ibv_query_gid(context, 1, 0, &gid) == 0)
      std::memcpy(info.gid.data(), gid.raw, info.gid.size());
    ibv_close_device(context);

    if (!info.active) {
      DFKV_LOG_WARN("rdma: skipping device " + name + ": port 1 state=" +
                    std::to_string(static_cast<int>(port.state)) +
                    " (ACTIVE=" +
                    std::to_string(static_cast<int>(IBV_PORT_ACTIVE)) + ")");
    }
    candidates.push_back(std::move(info));
  }
  ibv_free_device_list(list);

  std::vector<RdmaDevInfo> active = FilterActive(candidates, filter);
  for (const auto& requested : filter) {
    const bool found = std::any_of(
        active.begin(), active.end(), [&](const RdmaDevInfo& device) {
          return device.name == requested;
        });
    if (!found)
      DFKV_LOG_WARN("rdma: configured device " + requested +
                    " is absent or not ACTIVE");
  }
  return active;
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
