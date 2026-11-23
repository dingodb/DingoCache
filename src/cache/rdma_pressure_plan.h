/* Which idle RDMA connections to evict so that a receive-pool request fits.
 * A pure function of the pool's geometry and the candidates, so that every
 * branch can be tested without RDMA. */
#ifndef DFKV_CACHE_RDMA_PRESSURE_PLAN_H_
#define DFKV_CACHE_RDMA_PRESSURE_PLAN_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "transport/rdma_recv_segment.h"

namespace dfkv::rdma {

// An idle connection that may be evicted: where its receive lease lies and
// when it was last active (steady microseconds; older is preferred).
struct PressureCandidate {
  const RecvSegment* segment = nullptr;
  size_t offset = 0;
  size_t size = 0;
  uint64_t last_active = 0;
};

// The candidates (indices into `candidates`) to evict so that a request of
// `bytes` for `rail` fits, or none if no set of them would make room.
// Evicting any other only costs its client the connection without making
// room. Two ways, the first preferred:
// - A run in a chunk the request may use (its rail's, or one of none; never
//   the staging-only chunk, which holds no connection lease): free ranges
//   and candidate leases that touch end to end and together hold `bytes`.
//   Leases and free ranges begin and end on the pool's alignment, which is
//   the request's, so such a run is allocatable whole. The run with the
//   fewest candidates, then the least recent activity among them.
// - Budget: evicting every lease of a chunk other than the initial one
//   empties it, and a growth refused by the budget releases empty chunks
//   and grows one for the request. Chunks holding only candidate leases are
//   emptied oldest activity first until the uncommitted budget holds
//   `bytes`; empty chunks count first, at no eviction.
std::vector<size_t> PlanPressureVictims(
    const RecvSegmentPool::Geometry& pool,
    const std::vector<PressureCandidate>& candidates, size_t bytes, int rail);

}  // namespace dfkv::rdma

#endif  // DFKV_CACHE_RDMA_PRESSURE_PLAN_H_
