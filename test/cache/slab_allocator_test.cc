// SlabAllocator: media-agnostic slot lifecycle. Pure logic, hermetic tests.
// Covers: alloc/get/remove, idempotent put, size-class reuse vs new, per-class
// CLOCK eviction, extent binding + cross-class steal, pin-blocks-eviction, byte
// accounting, oversize rejection, and a concurrent (TSan) stress.
#include "cache/slab_allocator.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <set>
#include <string>
#include <thread>
#include <vector>

using dfkv::SlabAllocator;
using SlotRef = SlabAllocator::SlotRef;

namespace {
SlabAllocator::Options Opts(uint64_t extent_bytes, uint32_t num_extents,
                            uint32_t align = 4096, double waste = 0.25) {
  SlabAllocator::Options o;
  o.extent_bytes = extent_bytes;
  o.num_extents = num_extents;
  o.align = align;
  o.max_waste = waste;
  return o;
}
}  // namespace

TEST(SlabAllocator, PutGetRemoveRoundTrip) {
  SlabAllocator a(Opts(64 * 1024, 4));
  std::vector<std::string> ev;
  SlotRef r;
  ASSERT_TRUE(a.Put("k", 4096, &r, &ev));
  EXPECT_TRUE(r.valid());
  EXPECT_EQ(r.slot_size, 4096u);
  EXPECT_EQ(r.offset, static_cast<uint64_t>(r.slot) * r.slot_size);
  EXPECT_TRUE(ev.empty());
  EXPECT_EQ(a.Count(), 1u);
  EXPECT_EQ(a.UsedBytes(), 4096u);

  SlotRef g;
  ASSERT_TRUE(a.Get("k", &g));
  EXPECT_EQ(g.extent, r.extent);
  EXPECT_EQ(g.slot, r.slot);
  EXPECT_FALSE(a.Get("absent", &g));

  EXPECT_EQ(a.Remove("k"), SlabAllocator::RemoveResult::kRemoved);
  EXPECT_FALSE(a.Contains("k"));
  EXPECT_EQ(a.Count(), 0u);
  EXPECT_EQ(a.UsedBytes(), 0u);
  EXPECT_EQ(a.Remove("k"), SlabAllocator::RemoveResult::kNotFound);
}

TEST(SlabAllocator, PutIsIdempotentKeepsSameSlot) {
  SlabAllocator a(Opts(64 * 1024, 4));
  std::vector<std::string> ev;
  SlotRef r1, r2;
  ASSERT_TRUE(a.Put("k", 4096, &r1, &ev));
  ASSERT_TRUE(a.Put("k", 4096, &r2, &ev));  // second put: same slot, no evict
  EXPECT_EQ(r1.extent, r2.extent);
  EXPECT_EQ(r1.slot, r2.slot);
  EXPECT_EQ(a.Count(), 1u);
  EXPECT_TRUE(ev.empty());
}

TEST(SlabAllocator, SizeClassReuseVsNew) {
  SlabAllocator a(Opts(1 << 20, 4, /*align=*/4096, /*waste=*/0.25));
  std::vector<std::string> ev;
  SlotRef r;
  ASSERT_TRUE(a.Put("a", 4096, &r, &ev));       // class 4096
  EXPECT_EQ(a.ClassCount(), 1u);
  ASSERT_TRUE(a.Put("b", 4000, &r, &ev));       // fits 4096 within 25% waste -> reuse
  EXPECT_EQ(a.ClassCount(), 1u);
  EXPECT_EQ(r.slot_size, 4096u);
  ASSERT_TRUE(a.Put("c", 8192, &r, &ev));       // needs a bigger class -> new
  EXPECT_EQ(a.ClassCount(), 2u);
  EXPECT_EQ(r.slot_size, 8192u);
}

TEST(SlabAllocator, EvictsWithinClassUnderPressure) {
  // 1 extent of 4 slots (4 * 4096). A 5th 4096 key must evict one.
  SlabAllocator a(Opts(4 * 4096, 1));
  std::vector<std::string> ev;
  SlotRef r;
  for (int i = 0; i < 4; ++i)
    ASSERT_TRUE(a.Put("k" + std::to_string(i), 4096, &r, &ev));
  EXPECT_EQ(a.Count(), 4u);
  EXPECT_TRUE(ev.empty());
  ASSERT_TRUE(a.Put("k4", 4096, &r, &ev));   // full -> evict one
  EXPECT_EQ(a.Count(), 4u);
  EXPECT_EQ(ev.size(), 1u);
  EXPECT_EQ(a.Evictions(), 1u);
  EXPECT_TRUE(a.Contains("k4"));
}

TEST(SlabAllocator, GetGivesSecondChanceInClock) {
  // 4 slots; touch k0 so it survives the first eviction (referenced bit).
  SlabAllocator a(Opts(4 * 4096, 1));
  std::vector<std::string> ev;
  SlotRef r;
  for (int i = 0; i < 4; ++i)
    ASSERT_TRUE(a.Put("k" + std::to_string(i), 4096, &r, &ev));
  ASSERT_TRUE(a.Get("k0", &r));  // referenced -> should be spared once
  ASSERT_TRUE(a.Put("k4", 4096, &r, &ev));
  EXPECT_EQ(ev.size(), 1u);
  EXPECT_NE(ev[0], "k0") << "a referenced entry gets a second chance";
  EXPECT_TRUE(a.Contains("k0"));
}

TEST(SlabAllocator, PinBlocksEviction) {
  SlabAllocator a(Opts(2 * 4096, 1));  // 2 slots
  std::vector<std::string> ev;
  SlotRef r;
  ASSERT_TRUE(a.Put("pinned", 4096, &r, &ev));
  ASSERT_TRUE(a.Put("other", 4096, &r, &ev));
  ASSERT_TRUE(a.Pin("pinned"));
  ASSERT_TRUE(a.Put("new", 4096, &r, &ev));  // must evict "other", never "pinned"
  EXPECT_EQ(ev.size(), 1u);
  EXPECT_EQ(ev[0], "other");
  EXPECT_TRUE(a.Contains("pinned"));

  // With BOTH slots pinned, a further Put has nothing to evict -> fails.
  ASSERT_TRUE(a.Pin("new"));
  ev.clear();
  EXPECT_FALSE(a.Put("nope", 4096, &r, &ev)) << "all slots pinned -> no room";
  EXPECT_TRUE(ev.empty());
  // Unpin frees the path again.
  ASSERT_TRUE(a.Unpin("new"));
  EXPECT_TRUE(a.Put("nope", 4096, &r, &ev));
}

TEST(SlabAllocator, CrossClassStealWhenPoolEmpty) {
  // 2 extents. Fill both with class-A (4096) keys. Then a class-B (8192) key
  // with the pool empty must STEAL a fully-unpinned A extent and rebind it.
  SlabAllocator a(Opts(4 * 4096, 2));  // each extent: 4 A-slots or 2 B-slots
  std::vector<std::string> ev;
  SlotRef r;
  for (int i = 0; i < 8; ++i)  // 8 A-slots = both extents bound to class A
    ASSERT_TRUE(a.Put("a" + std::to_string(i), 4096, &r, &ev));
  EXPECT_EQ(a.BoundExtents(), 2u);
  ev.clear();
  ASSERT_TRUE(a.Put("b0", 8192, &r, &ev));  // needs class B -> steal an A extent
  EXPECT_EQ(r.slot_size, 8192u);
  EXPECT_FALSE(ev.empty()) << "steal evicts the stolen extent's residents";
  EXPECT_TRUE(a.Contains("b0"));
  EXPECT_GE(a.ClassCount(), 2u);
}

// Fill both size classes to a stable steady state where every extent is full,
// so growth-first striping (grabs up to kStripeWays=8) can't monopolize the
// pool: 16 extents, A(8192) takes 8 (16 keys), B(4096) takes 8 (32 keys).
// Returns after both classes are full; put_seq = 48, A's youngest = 16.
namespace {
void FillTwoClassesFull(SlabAllocator& a, std::vector<std::string>& ev) {
  SlotRef r;
  for (int i = 0; i < 16; ++i)  // stale A: 8192, 2/extent -> 8 extents
    ASSERT_TRUE(a.Put("stale" + std::to_string(i), 8192, &r, &ev));
  for (int i = 0; i < 32; ++i)  // hot B: 4096, 4/extent -> 8 extents
    ASSERT_TRUE(a.Put("hot" + std::to_string(i), 4096, &r, &ev));
}
}  // namespace

TEST(SlabAllocator, ColdDonorStolenBeforeSelfEviction) {
  // Phase 9: once the pool is full, a busy class must reclaim a globally-cold
  // CROSS-class extent instead of self-evicting its own just-written pages.
  ::setenv("DFKV_SLAB_COLD_STEAL_WINDOW", "6", 1);  // small window, determinism
  SlabAllocator a(Opts(4 * 4096, 16));
  std::vector<std::string> ev;
  SlotRef r;
  FillTwoClassesFull(a, ev);
  // Keep writing B (self-eviction would succeed). put_seq climbs past
  // stale's youngest (16) + window (6): the stale A extent becomes cold and is
  // stolen in preference to evicting a hot page.
  bool stole_stale = false;
  for (int i = 32; i < 80 && !stole_stale; ++i) {
    ev.clear();
    ASSERT_TRUE(a.Put("hot" + std::to_string(i), 4096, &r, &ev));
    for (const auto& k : ev)
      if (k.rfind("stale", 0) == 0) stole_stale = true;
  }
  EXPECT_TRUE(stole_stale) << "a globally-cold cross-class extent was reclaimed";
  EXPECT_GE(a.ColdSteals(), 1u);
  ::unsetenv("DFKV_SLAB_COLD_STEAL_WINDOW");
}

TEST(SlabAllocator, ColdStealDisabledFallsBackToSelfEvict) {
  ::setenv("DFKV_SLAB_COLD_STEAL_WINDOW", "0", 1);  // disable
  SlabAllocator a(Opts(4 * 4096, 16));
  std::vector<std::string> ev;
  SlotRef r;
  FillTwoClassesFull(a, ev);
  for (int i = 32; i < 96; ++i)  // busy class self-evicts its own only
    ASSERT_TRUE(a.Put("hot" + std::to_string(i), 4096, &r, &ev));
  EXPECT_EQ(a.ColdSteals(), 0u) << "disabled: never cold-steals";
  EXPECT_TRUE(a.Contains("stale0")) << "stale cross-class data survives";
  ::unsetenv("DFKV_SLAB_COLD_STEAL_WINDOW");
}

TEST(SlabAllocator, EvictColdToTargetFreesGloballyColdestFirst) {
  // Phase 10: proactive watermark eviction. Fill the pool, then evict down to
  // a target — the OLDEST data must go first, newest survives.
  SlabAllocator a(Opts(4 * 4096, 8));  // 8 extents, 32 4096-slots
  std::vector<std::string> ev;
  SlotRef r;
  for (int i = 0; i < 32; ++i)  // fill: k0 oldest .. k31 newest
    ASSERT_TRUE(a.Put("k" + std::to_string(i), 4096, &r, &ev));
  const uint64_t full = a.UsedBytes();
  EXPECT_EQ(full, 32u * 4096);
  ev.clear();
  // Evict down to ~half. Coldest-first: the low-numbered (oldest) keys go.
  size_t freed = a.EvictColdToTarget(full / 2, /*max_extents=*/8, &ev);
  EXPECT_GT(freed, 0u);
  EXPECT_LE(a.UsedBytes(), full / 2 + 4 * 4096);  // within one extent of target
  EXPECT_GE(a.WatermarkEvictions(), 1u);
  ASSERT_FALSE(ev.empty());
  // Everything evicted is from the older half; the newest key survived.
  for (const auto& k : ev) {
    int n = std::stoi(k.substr(1));
    EXPECT_LT(n, 32) << k;
  }
  EXPECT_TRUE(a.Contains("k31")) << "newest data must survive proactive eviction";
}

TEST(SlabAllocator, EvictColdToTargetRespectsPinsAndTarget) {
  SlabAllocator a(Opts(4 * 4096, 4));  // 4 extents, 16 slots
  std::vector<std::string> ev;
  SlotRef r;
  for (int i = 0; i < 16; ++i) ASSERT_TRUE(a.Put("k" + std::to_string(i), 4096, &r, &ev));
  ASSERT_TRUE(a.Pin("k0"));  // pins k0's whole extent against eviction
  ev.clear();
  const uint64_t used0 = a.UsedBytes();
  // Target 0 would want to evict everything, but pinned extents can't be freed.
  a.EvictColdToTarget(0, /*max_extents=*/4, &ev);
  EXPECT_TRUE(a.Contains("k0")) << "pinned key never evicted";
  EXPECT_LT(a.UsedBytes(), used0) << "some cold extents freed";
}

TEST(SlabAllocator, OversizeValueRejected) {
  SlabAllocator a(Opts(4096, 2));  // extent holds one 4096 slot
  std::vector<std::string> ev;
  SlotRef r;
  EXPECT_FALSE(a.Put("big", 4097, &r, &ev)) << "value larger than an extent";
  EXPECT_TRUE(a.Put("ok", 4096, &r, &ev));
}

TEST(SlabAllocator, ZeroLenGetsAMinimalSlot) {
  SlabAllocator a(Opts(64 * 1024, 2));
  std::vector<std::string> ev;
  SlotRef r;
  ASSERT_TRUE(a.Put("z", 0, &r, &ev));  // 0-byte anchor -> one aligned slot
  EXPECT_EQ(r.slot_size, 4096u);
  EXPECT_TRUE(a.Contains("z"));
}

TEST(SlabAllocator, ConcurrentPutGetRemoveIsRaceFree) {
  // TSan target: many threads hammer distinct + shared keys. Correctness beyond
  // "no crash / no race" is loose here; the single mutex must serialize cleanly.
  SlabAllocator a(Opts(256 * 4096, 8));
  constexpr int T = 8, N = 2000;
  std::atomic<int> ok_puts{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < T; ++t) {
    ts.emplace_back([&, t] {
      std::vector<std::string> ev;
      SlotRef r;
      for (int i = 0; i < N; ++i) {
        std::string k = "k" + std::to_string((t * N + i) % 5000);
        if (a.Put(k, 4096, &r, &ev)) ok_puts.fetch_add(1);
        a.Get(k, &r);
        if ((i & 3) == 0) a.Pin(k);
        if ((i & 3) == 0) a.Unpin(k);
        if ((i & 7) == 0) a.Remove(k);
        ev.clear();
      }
    });
  }
  for (auto& th : ts) th.join();
  EXPECT_GT(ok_puts.load(), 0);
  EXPECT_LE(a.UsedBytes(), a.Capacity());  // never over-commit the pool
}

TEST(SlabAllocator, RestoreInstallsKeyAtKnownSlot) {
  SlabAllocator a(Opts(4 * 4096, 2));
  // Restore two keys at known slots (as a rebuild would from persistence).
  EXPECT_TRUE(a.Restore("ka", 4096, /*extent=*/0, /*slot=*/1));
  EXPECT_TRUE(a.Restore("kb", 4096, /*extent=*/0, /*slot=*/2));
  EXPECT_EQ(a.Count(), 2u);
  EXPECT_EQ(a.UsedBytes(), 2u * 4096u);
  SlotRef r;
  ASSERT_TRUE(a.Get("ka", &r));
  EXPECT_EQ(r.extent, 0u);
  EXPECT_EQ(r.slot, 1u);
  EXPECT_EQ(r.offset, 1u * 4096u);
  // A subsequent Put on the same extent must use a still-free slot, not clobber
  // the restored ones.
  std::vector<std::string> ev;
  ASSERT_TRUE(a.Put("kc", 4096, &r, &ev));
  EXPECT_TRUE(ev.empty());
  EXPECT_NE(r.slot, 1u);
  EXPECT_NE(r.slot, 2u);
}

TEST(SlabAllocator, RestoreRejectsInconsistentInput) {
  SlabAllocator a(Opts(4 * 4096, 1));
  EXPECT_FALSE(a.Restore("bad_extent", 4096, /*extent=*/9, 0));   // extent out of range
  EXPECT_FALSE(a.Restore("bad_slot", 4096, 0, /*slot=*/99));      // slot out of range
  EXPECT_TRUE(a.Restore("k", 4096, 0, 0));
  EXPECT_FALSE(a.Restore("k", 4096, 0, 1));                       // duplicate key
  EXPECT_FALSE(a.Restore("k2", 4096, 0, 0));                      // slot already taken
  // A second class on the same extent is a persistence inconsistency.
  EXPECT_FALSE(a.Restore("k3", 8192, 0, 0));
}

// Consecutive Puts must STRIPE across extents (different backing files): a
// single-stack free list hands out one extent's slots back-to-back, funneling
// every concurrent writer into one inode, and buffered writes to one file
// serialize on the kernel's per-inode lock.
TEST(SlabAllocator, ConsecutivePutsStripeAcrossExtents) {
  SlabAllocator a(Opts(4 * 4096, 4));  // 4 extents x 4 slots of one class
  std::vector<std::string> ev;
  SlotRef r;
  std::vector<uint32_t> extents;
  for (int i = 0; i < 4; ++i) {
    ASSERT_TRUE(a.Put("k" + std::to_string(i), 4096, &r, &ev));
    extents.push_back(r.extent);
  }
  // First 4 Puts land on 4 DISTINCT extents (pool has 4, stripe width >= 4).
  std::sort(extents.begin(), extents.end());
  extents.erase(std::unique(extents.begin(), extents.end()), extents.end());
  EXPECT_EQ(extents.size(), 4u) << "puts funneled into fewer inodes than available";
  // The rotation keeps cycling once all extents are bound.
  ASSERT_TRUE(a.Put("k4", 4096, &r, &ev));
  uint32_t e4 = r.extent;
  ASSERT_TRUE(a.Put("k5", 4096, &r, &ev));
  EXPECT_NE(r.extent, e4) << "back-to-back puts hit the same extent";
}

// A fully-freed extent flows back to the shared pool (unbound) once its class
// keeps more than kStripeWays extents in rotation -- a later class of a new
// size can then bind it instead of stealing (which evicts residents).
TEST(SlabAllocator, FullyFreeExtentReturnsToPool) {
  // 12 extents of one 4096-slot each: 12 puts bind all 12 (stripe top-up), and
  // the rotation holds 12 > kStripeWays=8 entries only while slots are free --
  // after the puts every extent is FULL, so frees make extents fully-free again.
  SlabAllocator a(Opts(4096, 12));
  std::vector<std::string> ev;
  SlotRef r;
  for (int i = 0; i < 12; ++i)
    ASSERT_TRUE(a.Put("k" + std::to_string(i), 4096, &r, &ev));
  EXPECT_EQ(a.BoundExtents(), 12u);
  EXPECT_EQ(a.ExtentReturns(), 0u);
  // Free them all: each Remove makes that extent fully free; once the rotation
  // exceeds 8, the surplus fully-free extents unbind back to the pool.
  for (int i = 0; i < 12; ++i) a.Remove("k" + std::to_string(i));
  EXPECT_GT(a.ExtentReturns(), 0u);
  EXPECT_LT(a.BoundExtents(), 12u);
  // A NEW size class (smaller slot => different class) binds a returned pool
  // extent without stealing/evicting.
  SlabAllocator b(Opts(8 * 1024, 12, 1024));
  for (int i = 0; i < 12; ++i)
    ASSERT_TRUE(b.Put("k" + std::to_string(i), 8 * 1024, &r, &ev));  // fill: 1 slot/extent
  for (int i = 0; i < 12; ++i) b.Remove("k" + std::to_string(i));
  EXPECT_GT(b.ExtentReturns(), 0u);
  ASSERT_TRUE(b.Put("small", 1024, &r, &ev));  // new class, binds from pool
  EXPECT_TRUE(ev.empty()) << "returned-pool bind must not evict";
  EXPECT_EQ(b.Steals(), 0u);
}

// Explicit Remove is logically immediate but physical reuse waits for the last
// pin: an unlocked reader/writer can never observe a recycled slot.
TEST(SlabAllocator, RemoveDefersPinnedSlotUntilLastUnpin) {
  SlabAllocator a(Opts(4096, 1));  // one slot total
  std::vector<std::string> ev;
  SlotRef r1, r2;
  ASSERT_TRUE(a.Put("k", 4096, &r1, &ev));
  ASSERT_TRUE(a.Pin("k"));
  ASSERT_TRUE(a.Pin("k"));
  EXPECT_EQ(a.Remove("k"), SlabAllocator::RemoveResult::kDeferred);
  EXPECT_FALSE(a.Contains("k"));
  EXPECT_FALSE(a.Get("k", &r2));
  EXPECT_FALSE(a.Pin("k"));
  EXPECT_EQ(a.Count(), 1u);
  EXPECT_EQ(a.UsedBytes(), 4096u);
  EXPECT_FALSE(a.Put("k2", 4096, &r2, &ev));
  ASSERT_TRUE(a.Unpin("k"));
  EXPECT_EQ(a.Count(), 1u);
  EXPECT_FALSE(a.Put("k2", 4096, &r2, &ev));
  ASSERT_TRUE(a.Unpin("k"));
  EXPECT_EQ(a.Count(), 0u);
  EXPECT_EQ(a.UsedBytes(), 0u);
  ASSERT_TRUE(a.Put("k2", 4096, &r2, &ev));
  EXPECT_TRUE(ev.empty());
  EXPECT_EQ(r2.extent, r1.extent);
  EXPECT_EQ(r2.slot, r1.slot);
}

// ---- background-reclaimer + resident-list additions (put-path deserialization) ----

// Classes() reports per-class free/resident/puts truthfully across put, evict,
// remove, and extent hand-offs (the reclaimer's decisions ride on these counts).
TEST(SlabAllocator, ClassStatsTrackFreeResidentAndPuts) {
  SlabAllocator a(Opts(4 * 4096, 4));  // 16 slots of one 4096 class
  std::vector<std::string> ev;
  SlotRef r;
  for (int i = 0; i < 10; ++i)
    ASSERT_TRUE(a.Put("k" + std::to_string(i), 4096, &r, &ev));
  auto cs = a.Classes();
  ASSERT_EQ(cs.size(), 1u);
  EXPECT_EQ(cs[0].slot_size, 4096u);
  EXPECT_EQ(cs[0].resident, 10u);
  EXPECT_EQ(cs[0].free_slots, 6u);
  EXPECT_EQ(cs[0].puts, 10u);
  ASSERT_TRUE(a.Put("k0", 4096, &r, &ev));  // idempotent hit: NOT a new insert
  EXPECT_EQ(a.Classes()[0].puts, 10u);
  a.Remove("k9");
  cs = a.Classes();
  EXPECT_EQ(cs[0].resident, 9u);
  EXPECT_EQ(cs[0].free_slots, 7u);
}

// ReclaimClass evicts ahead of demand: it frees slots up to the target, in
// CLOCK order, without touching pinned entries, and reports the victims so the
// caller can drop them from its own index.
TEST(SlabAllocator, ReclaimClassCreatesHeadroomSkippingPinned) {
  SlabAllocator a(Opts(4 * 4096, 2));  // 8 slots, one class
  std::vector<std::string> ev;
  SlotRef r;
  for (int i = 0; i < 8; ++i)
    ASSERT_TRUE(a.Put("k" + std::to_string(i), 4096, &r, &ev));
  ASSERT_TRUE(a.Pin("k0"));
  ASSERT_TRUE(a.Pin("k1"));
  std::vector<std::string> victims;
  const size_t got = a.ReclaimClass(0, /*target_free=*/3, /*max_victims=*/8, &victims);
  EXPECT_EQ(got, 3u);
  EXPECT_EQ(victims.size(), 3u);
  for (const auto& v : victims) {
    EXPECT_NE(v, "k0");
    EXPECT_NE(v, "k1");
    EXPECT_FALSE(a.Contains(v));
  }
  EXPECT_EQ(a.Classes()[0].free_slots, 3u);
  // A follow-up Put takes a reclaimed slot WITHOUT evicting inline.
  ev.clear();
  ASSERT_TRUE(a.Put("fresh", 4096, &r, &ev));
  EXPECT_TRUE(ev.empty()) << "put should ride the reclaimed headroom";
}

// ReclaimClass respects max_victims (bounded work per lock hold) and is a no-op
// when the class already holds the target headroom.
TEST(SlabAllocator, ReclaimClassBoundedAndIdempotent) {
  SlabAllocator a(Opts(4 * 4096, 2));
  std::vector<std::string> ev;
  SlotRef r;
  for (int i = 0; i < 8; ++i)
    ASSERT_TRUE(a.Put("k" + std::to_string(i), 4096, &r, &ev));
  std::vector<std::string> victims;
  EXPECT_EQ(a.ReclaimClass(0, /*target_free=*/4, /*max_victims=*/2, &victims), 2u);
  EXPECT_EQ(victims.size(), 2u);
  victims.clear();
  EXPECT_EQ(a.ReclaimClass(0, /*target_free=*/2, /*max_victims=*/8, &victims), 0u);
  EXPECT_TRUE(victims.empty());
  // Out-of-range class index: harmless no-op.
  EXPECT_EQ(a.ReclaimClass(7, 4, 8, &victims), 0u);
}

// The cascade-shrink guard: when an eviction returns a fully-free extent to the
// shared pool (free count goes DOWN, not up), ReclaimClass stops instead of
// hollowing the class out chasing a target it can no longer reach.
TEST(SlabAllocator, ReclaimClassStopsOnExtentReturn) {
  // 12 extents x 1 slot each. Fill all 12, then Remove 8: the rotation now
  // holds exactly kStripeWays(8) fully-free extents (one more and they start
  // unbinding). The next eviction pushes the rotation past 8, so ITS extent
  // returns to the pool -- free count nets zero, and the guard must stop the
  // pass instead of hollowing out the remaining residents.
  SlabAllocator a(Opts(4096, 12));
  std::vector<std::string> ev;
  SlotRef r;
  for (int i = 0; i < 12; ++i)
    ASSERT_TRUE(a.Put("k" + std::to_string(i), 4096, &r, &ev));
  for (int i = 0; i < 8; ++i) a.Remove("k" + std::to_string(i));
  ASSERT_EQ(a.ExtentReturns(), 0u);
  ASSERT_EQ(a.Classes()[0].free_slots, 8u);
  std::vector<std::string> victims;
  const size_t got = a.ReclaimClass(0, /*target_free=*/12, /*max_victims=*/12, &victims);
  EXPECT_EQ(got, 1u) << "must stop after the eviction that returned an extent";
  EXPECT_EQ(a.ExtentReturns(), 1u);
  EXPECT_EQ(a.Classes()[0].resident, 3u);
}

// Steal now walks the stolen extent's resident list instead of scanning the
// whole index: behavior must be identical -- exactly that extent's residents
// are evicted, everything else survives, and the needy class gets the extent.
TEST(SlabAllocator, StealEvictsExactlyTheStolenExtentsResidents) {
  SlabAllocator a(Opts(4 * 4096, 2));  // 2 extents x 4 slots
  std::vector<std::string> ev;
  SlotRef r;
  std::vector<std::string> ext_keys[2];
  for (int i = 0; i < 8; ++i) {
    const std::string k = "k" + std::to_string(i);
    ASSERT_TRUE(a.Put(k, 4096, &r, &ev));
    ext_keys[r.extent].push_back(k);
  }
  // A new class (16384 > 4096/0.25 waste bound) finds no pool extent -> steal.
  ev.clear();
  ASSERT_TRUE(a.Put("big", 16 * 1024, &r, &ev));
  EXPECT_EQ(a.Steals(), 1u);
  EXPECT_EQ(ev.size(), 4u) << "exactly one extent's residents evicted";
  const uint32_t stolen = r.extent;
  std::set<std::string> gone(ev.begin(), ev.end());
  for (const auto& k : ext_keys[stolen]) EXPECT_TRUE(gone.count(k)) << k;
  for (const auto& k : ext_keys[1 - stolen]) {
    EXPECT_FALSE(gone.count(k)) << k;
    EXPECT_TRUE(a.Contains(k)) << k;
  }
}

// Restore populates the resident list too: a steal after a rebuild must evict
// the restored keys of the stolen extent (they live only in the extent list --
// a regression here silently leaks slots).
TEST(SlabAllocator, StealAfterRestoreEvictsRestoredResidents) {
  SlabAllocator a(Opts(4 * 4096, 1));  // one extent x 4 slots
  for (int i = 0; i < 4; ++i)
    ASSERT_TRUE(a.Restore("k" + std::to_string(i), 4096, /*extent=*/0,
                          /*slot=*/static_cast<uint32_t>(i)));
  EXPECT_EQ(a.Count(), 4u);
  std::vector<std::string> ev;
  SlotRef r;
  ASSERT_TRUE(a.Put("big", 16 * 1024, &r, &ev));  // must steal extent 0
  EXPECT_EQ(a.Steals(), 1u);
  EXPECT_EQ(ev.size(), 4u);
  EXPECT_EQ(a.Count(), 1u);
  for (int i = 0; i < 4; ++i)
    EXPECT_FALSE(a.Contains("k" + std::to_string(i)));
}

// ---- class rebalance additions (StealFrom mechanism) ----

// StealFrom moves exactly one extent: the donor's residents on that extent are
// evicted, the target class gains its capacity, everything else survives.
TEST(SlabAllocator, StealFromMovesOneExtentDonorToTarget) {
  SlabAllocator a(Opts(4 * 4096, 3));  // 3 extents x 4 slots
  std::vector<std::string> ev;
  SlotRef r;
  for (int i = 0; i < 12; ++i)
    ASSERT_TRUE(a.Put("k" + std::to_string(i), 4096, &r, &ev));  // class 0 owns all
  // Make a second class by exact size via Restore? No pool extent left, so
  // create it through StealFrom itself: class must pre-exist -> use Put of a
  // 16 KiB value, which steals inline and creates class 1.
  ev.clear();
  ASSERT_TRUE(a.Put("big0", 16 * 1024, &r, &ev));  // class 1, inline steal
  ASSERT_EQ(a.ClassCount(), 2u);
  const size_t evicted_inline = ev.size();
  ASSERT_EQ(evicted_inline, 4u);
  auto cs = a.Classes();
  ASSERT_EQ(cs[0].extents, 2u);
  ASSERT_EQ(cs[1].extents, 1u);
  // Now the mechanism under test: move one more extent 0 -> 1.
  ev.clear();
  ASSERT_TRUE(a.StealFrom(0, 1, &ev));
  EXPECT_EQ(ev.size(), 4u) << "exactly the donor extent's residents";
  cs = a.Classes();
  EXPECT_EQ(cs[0].extents, 1u);
  EXPECT_EQ(cs[1].extents, 2u);
  EXPECT_EQ(cs[0].resident, 4u);
  EXPECT_EQ(cs[1].free_slots, 1u);  // class 1: 2 extents x 1 slot, 1 resident
  // Survivors intact.
  size_t alive = 0;
  for (int i = 0; i < 12; ++i) alive += a.Contains("k" + std::to_string(i));
  EXPECT_EQ(alive, 4u);
  EXPECT_TRUE(a.Contains("big0"));
}

TEST(SlabAllocator, BindPersistenceFailurePublishesNoSlots) {
  bool allow_bind = false;
  auto o = Opts(4 * 4096, 2);
  o.on_extent_bind = [&](uint32_t) { return allow_bind; };
  SlabAllocator a(o);
  std::vector<std::string> ev;
  SlotRef ref;

  EXPECT_FALSE(a.Put("k", 4096, &ref, &ev));
  EXPECT_TRUE(ev.empty());
  EXPECT_EQ(a.Count(), 0u);
  EXPECT_EQ(a.UsedBytes(), 0u);
  EXPECT_EQ(a.BoundExtents(), 0u);
  EXPECT_EQ(a.PoolExtents(), 2u);

  allow_bind = true;
  ASSERT_TRUE(a.Put("k", 4096, &ref, &ev));
  EXPECT_EQ(a.Count(), 1u);
  EXPECT_EQ(a.BoundExtents(), 2u);
}

TEST(SlabAllocator, EvictionPersistenceFailurePreservesResidentSlot) {
  bool allow_evict = false;
  auto options = Opts(4 * 4096, 1);
  options.on_slot_evict =
      [&](const SlotRef&) { return allow_evict; };
  SlabAllocator allocator(options);
  std::vector<std::string> evicted;
  SlotRef ref;
  for (int i = 0; i < 4; ++i)
    ASSERT_TRUE(allocator.Put("k" + std::to_string(i), 4096, &ref,
                              &evicted));

  evicted.clear();
  EXPECT_FALSE(allocator.Put("blocked", 4096, &ref, &evicted));
  EXPECT_TRUE(evicted.empty());
  EXPECT_EQ(allocator.Count(), 4u);
  for (int i = 0; i < 4; ++i)
    EXPECT_TRUE(allocator.Contains("k" + std::to_string(i)));

  allow_evict = true;
  ASSERT_TRUE(allocator.Put("landed", 4096, &ref, &evicted));
  EXPECT_EQ(evicted.size(), 1u);
  EXPECT_TRUE(allocator.Contains("landed"));
  EXPECT_EQ(allocator.Count(), 4u);
}

TEST(SlabAllocator, RebindPersistenceFailurePreservesDonorExtent) {
  bool allow_bind = true;
  auto o = Opts(4 * 4096, 2);
  o.on_extent_bind = [&](uint32_t) { return allow_bind; };
  SlabAllocator a(o);
  std::vector<std::string> ev;
  SlotRef ref;
  for (int i = 0; i < 8; ++i)
    ASSERT_TRUE(a.Put("a" + std::to_string(i), 4096, &ref, &ev));

  allow_bind = false;
  ev.clear();
  EXPECT_FALSE(a.Put("big", 8192, &ref, &ev));
  EXPECT_TRUE(ev.empty());
  EXPECT_EQ(a.Count(), 8u);
  EXPECT_EQ(a.BoundExtents(), 2u);
  for (int i = 0; i < 8; ++i)
    EXPECT_TRUE(a.Contains("a" + std::to_string(i)));

  allow_bind = true;
  ASSERT_TRUE(a.Put("big", 8192, &ref, &ev));
  EXPECT_EQ(ev.size(), 4u);
  EXPECT_TRUE(a.Contains("big"));
}

// StealFrom refuses: donor == target, bad indices, oversized target class, and
// a donor whose every extent holds a pin.
TEST(SlabAllocator, StealFromRefusalCases) {
  SlabAllocator a(Opts(4 * 4096, 2));
  std::vector<std::string> ev;
  SlotRef r;
  for (int i = 0; i < 8; ++i)
    ASSERT_TRUE(a.Put("k" + std::to_string(i), 4096, &r, &ev));
  ASSERT_TRUE(a.Put("big", 16 * 1024, &r, &ev));  // class 1 via inline steal
  EXPECT_FALSE(a.StealFrom(0, 0, &ev));
  EXPECT_FALSE(a.StealFrom(7, 1, &ev));
  EXPECT_FALSE(a.StealFrom(0, 7, &ev));
  // Pin one resident on class 0's remaining extent: no eligible donor extent.
  for (int i = 0; i < 8; ++i)
    if (a.Contains("k" + std::to_string(i))) { ASSERT_TRUE(a.Pin("k" + std::to_string(i))); break; }
  ev.clear();
  EXPECT_FALSE(a.StealFrom(0, 1, &ev));
  EXPECT_TRUE(ev.empty());
}

// ClassStat.extents stays truthful across bind, inline steal, StealFrom, and
// fully-free extent returns (the rebalance policy reads it every tick).
TEST(SlabAllocator, ClassStatsExtentsTracksHandoffs) {
  SlabAllocator a(Opts(4096, 12));  // 12 extents x 1 slot
  std::vector<std::string> ev;
  SlotRef r;
  for (int i = 0; i < 12; ++i)
    ASSERT_TRUE(a.Put("k" + std::to_string(i), 4096, &r, &ev));
  EXPECT_EQ(a.Classes()[0].extents, 12u);
  EXPECT_EQ(a.PoolExtents(), 0u);
  for (int i = 0; i < 9; ++i) a.Remove("k" + std::to_string(i));  // returns fire past 8
  const uint32_t bound_after = a.Classes()[0].extents;
  EXPECT_EQ(bound_after + a.PoolExtents(), 12u) << "bind accounting must balance";
  EXPECT_GT(a.PoolExtents(), 0u);
}

// ---- inline growth-first additions (close the intra-tick starvation window) ----

// A bootstrapping class (< kStripeWays extents) on a full store must GROW by
// stealing from a big donor instead of eating itself: without growth-first,
// self-eviction succeeds as soon as the class has one unpinned resident and
// pins it at birth size until the background rebalance tick.
TEST(SlabAllocator, PutGrowsBootstrappingClassBeforeSelfEvicting) {
  // 16 extents x 4 slots of class A (4096); donor stays above the 8-extent floor.
  SlabAllocator a(Opts(4 * 4096, 16));
  std::vector<std::string> ev;
  SlotRef r;
  for (int i = 0; i < 64; ++i)
    ASSERT_TRUE(a.Put("a" + std::to_string(i), 4096, &r, &ev));
  ASSERT_EQ(a.Classes()[0].extents, 16u);
  // Class B (16 KiB, 1 slot/extent): burst of 6. Every put after the first
  // extent fills must STEAL (donor A: 16 > 8), never evict B's own residents.
  ev.clear();
  for (int i = 0; i < 6; ++i)
    ASSERT_TRUE(a.Put("b" + std::to_string(i), 16 * 1024, &r, &ev));
  for (int i = 0; i < 6; ++i)
    EXPECT_TRUE(a.Contains("b" + std::to_string(i))) << "b" << i << " self-evicted";
  auto cs = a.Classes();
  EXPECT_EQ(cs[1].resident, 6u);
  EXPECT_EQ(cs[1].extents, 6u);
  EXPECT_EQ(a.Steals(), 6u);
}

// Donor floor on the growth-first path: when every other class is at or below
// kStripeWays extents, a bootstrapping class must NOT steal (no ping-pong
// between two under-provisioned classes) -- it falls back to self-eviction.
TEST(SlabAllocator, GrowthFirstRespectsDonorFloor) {
  // 8 extents x 4 slots, all bound to class A (exactly kStripeWays -> not a donor).
  SlabAllocator a(Opts(4 * 4096, 8));
  std::vector<std::string> ev;
  SlotRef r;
  for (int i = 0; i < 32; ++i)
    ASSERT_TRUE(a.Put("a" + std::to_string(i), 4096, &r, &ev));
  ASSERT_EQ(a.Classes()[0].extents, 8u);
  // Class B: first put has no self to evict -> LAST-RESORT steal (floor 0) is
  // allowed and takes one A extent (A drops to 7).
  ev.clear();
  ASSERT_TRUE(a.Put("b0", 16 * 1024, &r, &ev));
  EXPECT_EQ(a.Steals(), 1u);
  // Second put: B(1 extent, full) is bootstrapping, but A(7) is at/below the
  // floor -> growth-first refuses; self-eviction evicts b0.
  ev.clear();
  ASSERT_TRUE(a.Put("b1", 16 * 1024, &r, &ev));
  EXPECT_EQ(a.Steals(), 1u) << "must not steal from a donor at/below the floor";
  ASSERT_EQ(ev.size(), 1u);
  EXPECT_EQ(ev[0], "b0");
  EXPECT_EQ(a.Classes()[0].extents, 7u) << "A must not shrink further";
}

// A class at or above kStripeWays extents behaves exactly as before: steady-
// state churn is self-eviction, not stealing (growth-first is bootstrap-only).
TEST(SlabAllocator, MatureClassStillSelfEvicts) {
  SlabAllocator a(Opts(4096, 16));  // 16 extents x 1 slot, one class
  std::vector<std::string> ev;
  SlotRef r;
  for (int i = 0; i < 16; ++i)
    ASSERT_TRUE(a.Put("k" + std::to_string(i), 4096, &r, &ev));
  ASSERT_EQ(a.Classes()[0].extents, 16u);
  ev.clear();
  ASSERT_TRUE(a.Put("k16", 4096, &r, &ev));  // full + mature -> CLOCK evict
  EXPECT_EQ(ev.size(), 1u);
  EXPECT_EQ(a.Steals(), 0u);
}
