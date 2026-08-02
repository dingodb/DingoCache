/* RDMA device discovery and runtime rail selection.
 *
 * Device names supplied through DFKV_RDMA_DEV are a whitelist, never a bypass
 * around link-state checks. Only port-1 devices in IBV_PORT_ACTIVE state are
 * returned, so an unconfigured client automatically uses every healthy rail
 * and an explicit list safely degrades when one member is down. */
#ifndef DFKV_RDMA_TOPOLOGY_H_
#define DFKV_RDMA_TOPOLOGY_H_

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace dfkv::rdma {

struct RdmaDevInfo {
  std::string name;
  int numa_node = -1;
  uint16_t lid = 0;
  std::array<uint8_t, 16> gid{};
  bool active = false;
};

enum class RailLocality : uint8_t {
  kDisabled,
  kLocal,
  kCallerUnknown,
  kNoLocal,
};

struct RailCandidates {
  // Stable device-index mask. A non-empty mask prevents the admission policy's
  // empty-mask "all rails" convention from reviving a disabled topology rail.
  std::vector<uint8_t> allowed;
  RailLocality locality = RailLocality::kDisabled;
};

class RdmaTopology {
 public:
  explicit RdmaTopology(std::vector<RdmaDevInfo> devices);

  // Enumerate port 1 on every verbs device, apply the optional name whitelist,
  // and return only ports whose state is IBV_PORT_ACTIVE.
  static std::vector<RdmaDevInfo> Discover(
      const std::vector<std::string>& filter = {});

  // Pure filtering seam shared by Discover and the hardware-free unit tests.
  // Preserves enumeration order and removes duplicate names.
  static std::vector<RdmaDevInfo> FilterActive(
      const std::vector<RdmaDevInfo>& candidates,
      const std::vector<std::string>& filter = {});

  const std::vector<RdmaDevInfo>& devices() const { return devices_; }

  // Return the stable index into devices(). Prefer enabled rails local to
  // numa_node when requested, otherwise round-robin all enabled rails.
  // retry_count rotates the choice further for callers retrying a failed open.
  int SelectDevice(int numa_node, bool numa_aware,
                   size_t retry_count = 0) const;

  // Build the production admission mask from discovered device NUMA metadata.
  // With NUMA disabled every enabled rail is allowed. With it enabled, local
  // rails are exclusive when known; unknown caller/no-local topology falls back
  // to every enabled rail so locality can never prevent progress.
  RailCandidates CandidatesFor(int numa_node, bool numa_aware) const;

  // Exclude a rail after a runtime local-device failure. Existing connections
  // remain valid; only subsequent selections are affected.
  void DisableDevice(const std::string& name);

 private:
  std::vector<RdmaDevInfo> devices_;
  mutable std::mutex mu_;
  std::vector<uint8_t> enabled_;
  mutable std::atomic<size_t> rr_{0};
};

}  // namespace dfkv::rdma

#endif  // DFKV_RDMA_TOPOLOGY_H_
