#ifndef DFKV_CACHE_NODE_MEMBER_REPORT_H_
#define DFKV_CACHE_NODE_MEMBER_REPORT_H_

#include <functional>
#include <optional>
#include <string>
#include <utility>

#include "cache/kv_node_server.h"
#include "common/membership.h"
#include "mds/mds_registrar.h"
#include "utils/log.h"

namespace dfkv {

// What a cache node says about its own placement eligibility on every MDS
// register/heartbeat.
//
// A disk store that failed for good (see KvNodeServer::DisksHealthy) stores
// and reads nothing more until the process is restarted: PUT and GET fail
// with an I/O error, Exist and Retain answer a miss, and only what a RAM tier
// in front of it still holds is served. While such a node keeps renewing its
// lease as eligible, clients keep routing its share of the keys to it. So the
// node reports itself out of placement: the MDS
// leaves it out of the member list it gives to clients, and it stays
// registered, visible in `dfkvctl ring` as DEGRADED with `store=failed` in
// its info, until it is restarted or replaced.
//
// The signal is a latch that only failed local metadata I/O sets and nothing
// clears, so it cannot toggle: one node leaves placement at most once per
// process. It is carried by the existing HLT1 extension:
//   - with RDMA rails: the rail report, with ring_eligible cleared;
//   - without rails (`rails` empty): no HLT1 while the disks are healthy,
//     exactly as before, and an HLT1 with no devices once they are not.
inline std::optional<MemberHealth> NodeMemberHealth(
    bool disks_healthy, const std::function<MemberHealth()>& rails) {
  if (rails) {
    MemberHealth health = rails();
    if (!disks_healthy) health.ring_eligible = false;
    return health;
  }
  if (disks_healthy) return std::nullopt;
  MemberHealth health;
  health.ring_eligible = false;
  return health;
}

inline std::string NodeMemberInfo(const std::string& info, bool disks_healthy) {
  return disks_healthy ? info : info + ",store=failed";
}

// Wires both into the registrar. `srv` must outlive the registrar's thread.
inline void ReportNodeHealth(MdsRegistrar* registrar, const KvNodeServer* srv,
                             std::string info,
                             std::function<MemberHealth()> rails) {
  registrar->set_health_provider(
      [srv, rails = std::move(rails), logged = false]() mutable {
    const bool disks_healthy = srv->DisksHealthy();
    if (!disks_healthy && !logged) {
      logged = true;
      DFKV_LOG_ERROR(
          "a disk store has failed (see the \"disk store failed for good\" "
          "line above for the directory and the cause): reporting this node "
          "out of the placement ring. It stays registered and out of "
          "placement until it is restarted or replaced.");
    }
    return NodeMemberHealth(disks_healthy, rails);
  });
  registrar->set_info_provider([srv, info = std::move(info)] {
    return NodeMemberInfo(info, srv->DisksHealthy());
  });
}

}  // namespace dfkv

#endif  // DFKV_CACHE_NODE_MEMBER_REPORT_H_
