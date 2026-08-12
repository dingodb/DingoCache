/* RDMA device discovery and runtime rail selection.
 *
 * Automatic discovery returns only port-1 devices in IBV_PORT_ACTIVE state.
 * Explicitly configured topologies may opt into retaining inactive ports so
 * their stable rail indexes survive a boot-time hardware degradation. */
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

enum class RdmaDiscoveryPolicy : uint8_t {
  kActiveOnly,
  kAllowInactive,
};

enum class RdmaDiscoveryStatus : uint8_t {
  kOk,
  kDeviceListFailed,
  kConfiguredDeviceMissing,
  kDeviceOpenFailed,
  kPortQueryFailed,
  kGidQueryFailed,
};

// A non-kOk result is fail-closed: devices is empty and failed_device names
// the configured rail when the failure is device-specific.
struct RdmaDiscoveryResult {
  std::vector<RdmaDevInfo> devices;
  RdmaDiscoveryStatus status = RdmaDiscoveryStatus::kOk;
  std::string failed_device;

  // Provider-complete rails observed before an explicit discovery failure.
  // Unlike devices, this evidence is preserved for truthful diagnostics when
  // fail-closed resolution rejects the topology.
  std::vector<RdmaDevInfo> observed_devices;

  bool ok() const { return status == RdmaDiscoveryStatus::kOk; }
};

// One deterministic device-enumeration outcome. Failed probes carry only the
// device name, except kGidQueryFailed retains metadata for legacy ACTIVE-only
// resolution; successful probes carry the complete RdmaDevInfo.
struct RdmaDiscoveryProbe {
  RdmaDevInfo device;
  RdmaDiscoveryStatus status = RdmaDiscoveryStatus::kOk;
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
  // Degraded backstop mask, populated only when locality == kLocal: every
  // currently enabled rail, locality aside. Callers retry admission against it
  // once when no allowed (NUMA-local) rail can serve, so locality prefers but
  // never prevents progress. Empty for every other locality, where allowed
  // already spans all enabled rails.
  std::vector<uint8_t> fallback;
  RailLocality locality = RailLocality::kDisabled;
};

class RdmaTopology {
 public:
  explicit RdmaTopology(std::vector<RdmaDevInfo> devices);

  // Default discovery remains ACTIVE-only.
  static std::vector<RdmaDevInfo> Discover(
      const std::vector<std::string>& filter = {});

  // Typed discovery for explicit topology construction. kAllowInactive
  // preserves configured first-occurrence order and fails closed unless every
  // configured device has complete provider metadata, including a queried GID.
  static RdmaDiscoveryResult Discover(
      const std::vector<std::string>& filter, RdmaDiscoveryPolicy policy);

  // Pure resolution seam used by Discover and hardware-free unit tests.
  static RdmaDiscoveryResult ResolveDiscovery(
      const std::vector<RdmaDiscoveryProbe>& probes,
      const std::vector<std::string>& filter, RdmaDiscoveryPolicy policy);

  // Pure ACTIVE-only filtering seam. Preserves enumeration order and removes
  // duplicate names.
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
  // rails are exclusive when known and `fallback` carries every enabled rail
  // as the backstop for when no local rail is admissible (quarantined or
  // credit-bound); unknown caller/no-local topology falls back to every
  // enabled rail in `allowed` itself. Locality can never prevent progress.
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
