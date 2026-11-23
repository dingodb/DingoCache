#include "cache/rdma_pressure_plan.h"

#include <algorithm>
#include <limits>
#include <map>
#include <utility>

namespace dfkv::rdma {

std::vector<size_t> PlanPressureVictims(
    const RecvSegmentPool::Geometry& pool,
    const std::vector<PressureCandidate>& candidates, size_t bytes, int rail) {
  std::map<const RecvSegment*, std::vector<size_t>> held;
  for (size_t i = 0; i < candidates.size(); ++i)
    if (candidates[i].segment != nullptr)
      held[candidates[i].segment].push_back(i);
  if (held.empty()) return {};

  // Runs.
  std::vector<size_t> best;
  uint64_t best_newest = std::numeric_limits<uint64_t>::max();
  for (const auto& chunk : pool.chunks) {
    if (chunk.staging_only || (chunk.affinity != -1 && chunk.affinity != rail))
      continue;
    const auto found = held.find(chunk.segment);
    if (found == held.end()) continue;
    struct Piece {
      size_t offset;
      size_t size;
      const PressureCandidate* owner;  // null: free
      size_t index;
    };
    std::vector<Piece> pieces;
    for (const auto& [offset, size] : chunk.free)
      pieces.push_back(Piece{offset, size, nullptr, 0});
    for (const size_t i : found->second)
      pieces.push_back(Piece{candidates[i].offset, candidates[i].size,
                             &candidates[i], i});
    std::sort(pieces.begin(), pieces.end(),
              [](const Piece& a, const Piece& b) { return a.offset < b.offset; });
    // Sliding window over each run of touching pieces: the shortest window
    // ending at each piece that holds `bytes`.
    size_t first = 0, sum = 0;
    for (size_t last = 0; last < pieces.size(); ++last) {
      if (last > first &&
          pieces[last - 1].offset + pieces[last - 1].size != pieces[last].offset) {
        first = last;  // a gap: a new run
        sum = 0;
      }
      sum += pieces[last].size;
      while (first < last && sum - pieces[first].size >= bytes)
        sum -= pieces[first++].size;
      if (sum < bytes) continue;
      std::vector<size_t> victims;
      uint64_t newest = 0;
      for (size_t i = first; i <= last; ++i) {
        if (pieces[i].owner == nullptr) continue;
        victims.push_back(pieces[i].index);
        newest = std::max(newest, pieces[i].owner->last_active);
      }
      if (victims.empty()) continue;  // room already: the allocation raced
      if (best.empty() || victims.size() < best.size() ||
          (victims.size() == best.size() && newest < best_newest)) {
        best = std::move(victims);
        best_newest = newest;
      }
    }
  }
  if (!best.empty()) return best;

  // Budget: chunks whose every lease is a candidate's.
  struct Emptiable {
    size_t size;
    uint64_t newest;
    std::vector<size_t> owners;
  };
  std::vector<Emptiable> emptiable;
  size_t uncommitted = pool.committed_bytes <= pool.max_bytes
                           ? pool.max_bytes - pool.committed_bytes
                           : 0;
  for (const auto& chunk : pool.chunks) {
    if (chunk.initial || chunk.staging_only) continue;
    size_t free_bytes = 0;
    for (const auto& range : chunk.free) free_bytes += range.second;
    size_t evictable = 0;
    Emptiable e{chunk.size, 0, {}};
    const auto found = held.find(chunk.segment);
    if (found != held.end()) {
      for (const size_t i : found->second) {
        evictable += candidates[i].size;
        e.newest = std::max(e.newest, candidates[i].last_active);
        e.owners.push_back(i);
      }
    }
    if (free_bytes + evictable != chunk.size) continue;  // a lease stays
    emptiable.push_back(std::move(e));
  }
  std::sort(emptiable.begin(), emptiable.end(),
            [](const Emptiable& a, const Emptiable& b) { return a.newest < b.newest; });
  std::vector<size_t> owners;
  for (const Emptiable& e : emptiable) {
    if (uncommitted >= bytes) break;
    uncommitted += e.size;
    owners.insert(owners.end(), e.owners.begin(), e.owners.end());
  }
  if (uncommitted < bytes || owners.empty()) return {};
  return owners;
}

}  // namespace dfkv::rdma
