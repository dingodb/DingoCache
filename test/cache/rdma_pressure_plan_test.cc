// RDS-07: the pressure planner on synthetic pool geometry. Segments are only
// identities here, never touched.
#include "cache/rdma_pressure_plan.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace dfkv::rdma {
namespace {

constexpr size_t K = 4096;

const RecvSegment* Seg(uintptr_t n) {
  return reinterpret_cast<const RecvSegment*>(n * 64);
}

RecvSegmentPool::ChunkView Chunk(uintptr_t id, size_t size, int affinity,
                                 std::vector<std::pair<size_t, size_t>> free,
                                 bool initial = false,
                                 bool staging_only = false) {
  RecvSegmentPool::ChunkView view;
  view.segment = Seg(id);
  view.size = size;
  view.affinity = affinity;
  view.staging_only = staging_only;
  view.initial = initial;
  view.free = std::move(free);
  return view;
}

RecvSegmentPool::Geometry Pool(std::vector<RecvSegmentPool::ChunkView> chunks,
                               size_t max_bytes) {
  RecvSegmentPool::Geometry pool;
  pool.max_bytes = max_bytes;
  for (const auto& chunk : chunks) pool.committed_bytes += chunk.size;
  pool.chunks = std::move(chunks);
  return pool;
}

std::vector<size_t> Sorted(std::vector<size_t> v) {
  std::sort(v.begin(), v.end());
  return v;
}

// A run: two touching idle leases and a free range after them hold the
// request; one lease alone does not.
TEST(PressurePlan, ARunOfIdleLeasesAndFreeRanges) {
  // [0,8K) busy | [8K,16K) a | [16K,24K) b | [24K,28K) free | [28K,32K) busy
  const auto pool = Pool({Chunk(1, 32 * K, -1, {{24 * K, 4 * K}}, true)},
                         32 * K);
  const std::vector<PressureCandidate> candidates = {
      {Seg(1), 8 * K, 8 * K, 100}, {Seg(1), 16 * K, 8 * K, 200}};
  EXPECT_EQ(Sorted(PlanPressureVictims(pool, candidates, 12 * K, 0)),
            (std::vector<size_t>{1})) << "b and the free range are enough";
  EXPECT_EQ(Sorted(PlanPressureVictims(pool, candidates, 20 * K, 0)),
            (std::vector<size_t>{0, 1})) << "the whole run";
  EXPECT_TRUE(PlanPressureVictims(pool, candidates, 24 * K, 0).empty())
      << "more than the run holds, and the initial chunk is not emptied";
}

// Between equal runs, the one whose newest lease is least recent.
TEST(PressurePlan, TheLeastRecentlyActiveOfEqualRuns) {
  const auto pool = Pool({Chunk(1, 32 * K, -1, {}, true)}, 32 * K);
  // Two leases of 8K separated by a busy range: two runs of one each.
  const std::vector<PressureCandidate> candidates = {
      {Seg(1), 0, 8 * K, 500}, {Seg(1), 16 * K, 8 * K, 100}};
  EXPECT_EQ(PlanPressureVictims(pool, candidates, 8 * K, 0),
            (std::vector<size_t>{1}));
}

// Another rail's chunk and the staging-only chunk are never used for runs.
TEST(PressurePlan, OtherRailAndStagingOnlyChunksAreExcluded) {
  const auto pool =
      Pool({Chunk(1, 16 * K, -1, {}, true, /*staging_only=*/true),
            Chunk(2, 16 * K, 1, {}, false)},
           32 * K);
  const std::vector<PressureCandidate> in_staging = {{Seg(1), 0, 16 * K, 1}};
  EXPECT_TRUE(PlanPressureVictims(pool, in_staging, 8 * K, 0).empty());
  const std::vector<PressureCandidate> other_rail = {{Seg(2), 0, 16 * K, 1}};
  // On rail 0 that run is useless...
  // ... but the chunk can be emptied for the budget (see below): it is not
  // the initial one. Make it hold a non-candidate lease to see the run rule
  // alone.
  const auto busy = Pool({Chunk(1, 16 * K, -1, {}, true, true),
                          Chunk(2, 32 * K, 1, {})},
                         48 * K);
  EXPECT_TRUE(PlanPressureVictims(busy, other_rail, 8 * K, 0).empty());
  EXPECT_EQ(PlanPressureVictims(busy, other_rail, 8 * K, 1),
            (std::vector<size_t>{0})) << "its own rail may use it";
}

// The budget branch: a chunk whose every lease is a candidate's can be
// emptied (and its budget regrown for the request); one that also holds a
// lease of a busy connection cannot.
TEST(PressurePlan, AnEmptiableChunkVersusOneWithABusyLease) {
  // A full budget: initial chunk busy, chunk 2 held only by candidate 0,
  // chunk 3 held by candidate 1 and a busy lease.
  const auto pool = Pool({Chunk(1, 16 * K, -1, {}, true),
                          Chunk(2, 16 * K, 1, {}),
                          Chunk(3, 16 * K, 2, {})},
                         48 * K);
  const std::vector<PressureCandidate> candidates = {
      {Seg(2), 0, 16 * K, 100}, {Seg(3), 0, 8 * K, 50}};
  // 16K for rail 0: no run on rail 0's chunks; chunk 2 can be emptied.
  EXPECT_EQ(PlanPressureVictims(pool, candidates, 16 * K, 0),
            (std::vector<size_t>{0}));
  // 32K: emptying chunk 2 frees 16K only; chunk 3 cannot be emptied.
  EXPECT_TRUE(PlanPressureVictims(pool, candidates, 32 * K, 0).empty());
}

// The budget is reached after one chunk, or only after two, oldest first;
// already uncommitted budget and empty chunks count first.
TEST(PressurePlan, TheBudgetAfterOneChunkOrTwo) {
  const auto pool = Pool({Chunk(1, 16 * K, -1, {}, true),
                          Chunk(2, 16 * K, 1, {}),
                          Chunk(3, 16 * K, 2, {}),
                          Chunk(4, 16 * K, 3, {{0, 16 * K}})},  // empty
                         72 * K);  // 8K uncommitted
  const std::vector<PressureCandidate> candidates = {
      {Seg(2), 0, 16 * K, 300}, {Seg(3), 0, 8 * K, 100},
      {Seg(3), 8 * K, 8 * K, 200}};
  // 8K uncommitted + 16K empty chunk 4 = 24K: no eviction needed, nothing
  // to do (the refused growth trims chunk 4 itself).
  EXPECT_TRUE(PlanPressureVictims(pool, candidates, 24 * K, 0).empty());
  // 40K: chunk 3 (newest 200) before chunk 2 (300).
  EXPECT_EQ(Sorted(PlanPressureVictims(pool, candidates, 40 * K, 0)),
            (std::vector<size_t>{1, 2}));
  // 56K: both.
  EXPECT_EQ(Sorted(PlanPressureVictims(pool, candidates, 56 * K, 0)),
            (std::vector<size_t>{0, 1, 2}));
  // 64K: even both are not enough.
  EXPECT_TRUE(PlanPressureVictims(pool, candidates, 64 * K, 0).empty());
}

TEST(PressurePlan, NoCandidatesNoVictims) {
  const auto pool = Pool({Chunk(1, 16 * K, -1, {}, true)}, 16 * K);
  EXPECT_TRUE(PlanPressureVictims(pool, {}, 4 * K, 0).empty());
}

}  // namespace
}  // namespace dfkv::rdma
