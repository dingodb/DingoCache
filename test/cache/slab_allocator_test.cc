// SlabAllocator: media-agnostic slot lifecycle. Pure logic, hermetic tests.
// Covers: alloc/get/remove, idempotent put, size-class reuse vs new, per-class
// CLOCK eviction, extent binding + cross-class steal, pin-blocks-eviction, byte
// accounting, oversize rejection, and a concurrent (TSan) stress.
#include "cache/slab_allocator.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using dfkv::BlockKey;
using dfkv::SlabAllocator;
using SlotRef = SlabAllocator::SlotRef;

namespace {
BlockKey Key(std::string_view text) {
  auto word = [text](size_t offset) {
    uint64_t value = 0;
    const size_t count =
        offset < text.size() ? std::min<size_t>(8, text.size() - offset) : 0;
    for (size_t i = 0; i < count; ++i)
      value |= static_cast<uint64_t>(
                   static_cast<unsigned char>(text[offset + i]))
               << (i * 8);
    return value;
  };
  return BlockKey{word(0), word(8), word(16)};
}

BlockKey Key(const BlockKey& key) { return key; }

std::string KeyName(const BlockKey& key) {
  std::string text;
  text.reserve(24);
  for (uint64_t word : {key.digest_hi, key.digest_lo, key.tenant_hash}) {
    for (size_t i = 0; i < 8; ++i) {
      const char c = static_cast<char>((word >> (i * 8)) & 0xff);
      if (c == '\0') return text;
      text.push_back(c);
    }
  }
  return text;
}
SlabAllocator::Options Opts(uint64_t extent_bytes, uint32_t num_extents,
                            uint32_t align = 4096, double waste = 0.25) {
  SlabAllocator::Options o;
  o.extent_bytes = extent_bytes;
  o.num_extents = num_extents;
  o.align = align;
  o.max_waste = waste;
  return o;
}

size_t ClassIndex(const SlabAllocator& allocator, uint32_t slot_size) {
  const auto stats = allocator.Classes();
  for (size_t i = 0; i < stats.size(); ++i)
    if (stats[i].slot_size == slot_size) return i;
  return stats.size();
}
}  // namespace

TEST(SlabAllocator, PutGetRemoveRoundTrip) {
  SlabAllocator a(Opts(64 * 1024, 4));
  std::vector<BlockKey> ev;
  SlotRef r;
  ASSERT_TRUE(a.Put(Key("k"), 4096, &r, &ev));
  EXPECT_TRUE(r.valid());
  EXPECT_EQ(r.slot_size, 4096u);
  EXPECT_EQ(r.offset, static_cast<uint64_t>(r.slot) * r.slot_size);
  EXPECT_TRUE(ev.empty());
  EXPECT_EQ(a.Count(), 1u);
  EXPECT_EQ(a.UsedBytes(), 4096u);

  SlotRef g;
  ASSERT_TRUE(a.Get(Key("k"), &g));
  EXPECT_EQ(g.extent, r.extent);
  EXPECT_EQ(g.slot, r.slot);
  EXPECT_FALSE(a.Get(Key("absent"), &g));

  EXPECT_EQ(a.Remove(Key("k")), SlabAllocator::RemoveResult::kRemoved);
  EXPECT_FALSE(a.Contains(Key("k")));
  EXPECT_EQ(a.Count(), 0u);
  EXPECT_EQ(a.UsedBytes(), 0u);
  EXPECT_EQ(a.Remove(Key("k")), SlabAllocator::RemoveResult::kNotFound);
}

TEST(SlabAllocator, PutIsIdempotentKeepsSameSlot) {
  SlabAllocator a(Opts(64 * 1024, 4));
  std::vector<BlockKey> ev;
  SlotRef r1, r2;
  ASSERT_TRUE(a.Put(Key("k"), 4096, &r1, &ev));
  ASSERT_TRUE(a.Put(Key("k"), 4096, &r2, &ev));  // second put: same slot, no evict
  EXPECT_EQ(r1.extent, r2.extent);
  EXPECT_EQ(r1.slot, r2.slot);
  EXPECT_EQ(a.Count(), 1u);
  EXPECT_TRUE(ev.empty());
}

TEST(SlabAllocator, SizeClassReuseVsNew) {
  SlabAllocator a(Opts(1 << 20, 4, /*align=*/4096, /*waste=*/0.25));
  const size_t fixed_classes = a.ClassCount();
  std::vector<BlockKey> ev;
  SlotRef r;
  ASSERT_TRUE(a.Put(Key("a"), 4096, &r, &ev));
  ASSERT_TRUE(a.Put(Key("b"), 4000, &r, &ev));
  EXPECT_EQ(r.slot_size, 4096u);
  ASSERT_TRUE(a.Put(Key("c"), 8192, &r, &ev));
  EXPECT_EQ(r.slot_size, 8192u);
  EXPECT_EQ(a.ClassCount(), fixed_classes);
}

TEST(SlabAllocator, SizeClassesAreEstablishedBeforeInsertion) {
  SlabAllocator first(Opts(1 << 20, 4, /*align=*/4096, /*waste=*/0.25));
  SlabAllocator second(Opts(1 << 20, 4, /*align=*/4096, /*waste=*/0.25));
  const size_t startup_count = first.ClassCount();
  ASSERT_EQ(second.ClassCount(), startup_count);
  ASSERT_GT(startup_count, 2u);
  std::vector<BlockKey> evicted;
  SlotRef a_small, a_large, b_large, b_small;
  ASSERT_TRUE(first.Put(Key("small"), 12 * 1024, &a_small, &evicted));
  ASSERT_TRUE(first.Put(Key("large"), 16 * 1024, &a_large, &evicted));
  ASSERT_TRUE(second.Put(Key("large"), 16 * 1024, &b_large, &evicted));
  ASSERT_TRUE(second.Put(Key("small"), 12 * 1024, &b_small, &evicted));
  EXPECT_EQ(a_small.slot_size, b_small.slot_size);
  EXPECT_EQ(a_large.slot_size, b_large.slot_size);
  EXPECT_EQ(a_small.slot_size, 16u * 1024);
  EXPECT_EQ(first.ClassCount(), startup_count);
  EXPECT_EQ(second.ClassCount(), startup_count);
}

TEST(SlabAllocator, DenseSlotsAreReusedAfterRemoval) {
  SlabAllocator allocator(Opts(2 * 4096, 1));
  std::vector<BlockKey> evicted;
  std::set<std::pair<uint32_t, uint32_t>> placements;
  SlotRef ref;
  for (int i = 0; i < 100; ++i) {
    const std::string key = "k" + std::to_string(i);
    ASSERT_TRUE(allocator.Put(Key(key), 4096, &ref, &evicted));
    placements.emplace(ref.extent, ref.slot);
    EXPECT_EQ(allocator.Remove(Key(key)), SlabAllocator::RemoveResult::kRemoved);
  }
  EXPECT_LE(placements.size(), 2u);
  EXPECT_EQ(allocator.Count(), 0u);
  EXPECT_EQ(allocator.UsedBytes(), 0u);
}

TEST(SlabAllocator, EvictsWithinClassUnderPressure) {
  // 1 extent of 4 slots (4 * 4096). A 5th 4096 key must evict one.
  SlabAllocator a(Opts(4 * 4096, 1));
  std::vector<BlockKey> ev;
  SlotRef r;
  for (int i = 0; i < 4; ++i)
    ASSERT_TRUE(a.Put(Key("k" + std::to_string(i)), 4096, &r, &ev));
  EXPECT_EQ(a.Count(), 4u);
  EXPECT_TRUE(ev.empty());
  ASSERT_TRUE(a.Put(Key("k4"), 4096, &r, &ev));   // full -> evict one
  EXPECT_EQ(a.Count(), 4u);
  EXPECT_EQ(ev.size(), 1u);
  EXPECT_EQ(a.Evictions(), 1u);
  EXPECT_TRUE(a.Contains(Key("k4")));
}

TEST(SlabAllocator, GetGivesSecondChanceInClock) {
  // 4 slots; touch k0 so it survives the first eviction (referenced bit).
  SlabAllocator a(Opts(4 * 4096, 1));
  std::vector<BlockKey> ev;
  SlotRef r;
  for (int i = 0; i < 4; ++i)
    ASSERT_TRUE(a.Put(Key("k" + std::to_string(i)), 4096, &r, &ev));
  ASSERT_TRUE(a.Get(Key("k0"), &r));  // referenced -> should be spared once
  ASSERT_TRUE(a.Put(Key("k4"), 4096, &r, &ev));
  EXPECT_EQ(ev.size(), 1u);
  EXPECT_FALSE(ev[0] == Key("k0")) << "a referenced entry gets a second chance";
  EXPECT_TRUE(a.Contains(Key("k0")));
}

TEST(SlabAllocator, PinBlocksEviction) {
  SlabAllocator a(Opts(2 * 4096, 1));  // 2 slots
  std::vector<BlockKey> ev;
  SlotRef r;
  ASSERT_TRUE(a.Put(Key("pinned"), 4096, &r, &ev));
  ASSERT_TRUE(a.Put(Key("other"), 4096, &r, &ev));
  ASSERT_TRUE(a.Pin(Key("pinned")));
  ASSERT_TRUE(a.Put(Key("new"), 4096, &r, &ev));  // must evict "other", never "pinned"
  EXPECT_EQ(ev.size(), 1u);
  EXPECT_EQ(ev[0], Key("other"));
  EXPECT_TRUE(a.Contains(Key("pinned")));

  // With BOTH slots pinned, a further Put has nothing to evict -> fails.
  ASSERT_TRUE(a.Pin(Key("new")));
  ev.clear();
  EXPECT_FALSE(a.Put(Key("nope"), 4096, &r, &ev)) << "all slots pinned -> no room";
  EXPECT_TRUE(ev.empty());
  // Unpin frees the path again.
  ASSERT_TRUE(a.Unpin(Key("new")));
  EXPECT_TRUE(a.Put(Key("nope"), 4096, &r, &ev));
}

TEST(SlabAllocator, SlotHandlePinsOnlyItsLiveKey) {
  SlabAllocator a(Opts(2 * 4096, 1));
  std::vector<BlockKey> evicted;
  SlotRef ref;
  SlabAllocator::SlotHandle first_handle;
  const BlockKey first = Key("first");
  const BlockKey second = Key("second");

  ASSERT_TRUE(a.Put(first, 4096, &ref, &evicted, &first_handle));
  ASSERT_TRUE(first_handle.valid());
  EXPECT_FALSE(a.Pin(second, first_handle));
  ASSERT_TRUE(a.Pin(first, first_handle));
  EXPECT_EQ(a.Remove(first), SlabAllocator::RemoveResult::kDeferred);
  EXPECT_FALSE(a.Pin(first, first_handle));
  ASSERT_TRUE(a.Unpin(first, first_handle));
  EXPECT_FALSE(a.Contains(first));

  SlabAllocator::SlotHandle second_handle;
  ASSERT_TRUE(a.Put(second, 4096, &ref, &evicted, &second_handle));
  EXPECT_FALSE(a.Pin(first, first_handle));
  ASSERT_TRUE(a.Pin(second, second_handle));
  EXPECT_TRUE(a.Unpin(second, second_handle));
}

TEST(SlabAllocator, CrossClassStealWhenPoolEmpty) {
  // 2 extents. Fill both with class-A (4096) keys. Then a class-B (8192) key
  // with the pool empty must STEAL a fully-unpinned A extent and rebind it.
  SlabAllocator a(Opts(4 * 4096, 2));  // each extent: 4 A-slots or 2 B-slots
  std::vector<BlockKey> ev;
  SlotRef r;
  for (int i = 0; i < 8; ++i)  // 8 A-slots = both extents bound to class A
    ASSERT_TRUE(a.Put(Key("a" + std::to_string(i)), 4096, &r, &ev));
  EXPECT_EQ(a.BoundExtents(), 2u);
  ev.clear();
  ASSERT_TRUE(a.Put(Key("b0"), 8192, &r, &ev));  // needs class B -> steal an A extent
  EXPECT_EQ(r.slot_size, 8192u);
  EXPECT_FALSE(ev.empty()) << "steal evicts the stolen extent's residents";
  EXPECT_TRUE(a.Contains(Key("b0")));
  EXPECT_GE(a.ClassCount(), 2u);
}

// Fill both size classes to a stable steady state where every extent is full,
// so growth-first striping (grabs up to kStripeWays=8) can't monopolize the
// pool: 16 extents, A(8192) takes 8 (16 keys), B(4096) takes 8 (32 keys).
// Returns after both classes are full; put_seq = 48, A's youngest = 16.
namespace {
void FillTwoClassesFull(SlabAllocator& a, std::vector<BlockKey>& ev) {
  SlotRef r;
  for (int i = 0; i < 16; ++i)  // stale A: 8192, 2/extent -> 8 extents
    ASSERT_TRUE(a.Put(Key("stale" + std::to_string(i)), 8192, &r, &ev));
  for (int i = 0; i < 32; ++i)  // hot B: 4096, 4/extent -> 8 extents
    ASSERT_TRUE(a.Put(Key("hot" + std::to_string(i)), 4096, &r, &ev));
}
}  // namespace

TEST(SlabAllocator, ColdDonorStolenBeforeSelfEviction) {
  // Phase 9: once the pool is full, a busy class must reclaim a globally-cold
  // CROSS-class extent instead of self-evicting its own just-written pages.
  ::setenv("DFKV_SLAB_COLD_STEAL_WINDOW", "6", 1);  // small window, determinism
  SlabAllocator a(Opts(4 * 4096, 16));
  std::vector<BlockKey> ev;
  SlotRef r;
  FillTwoClassesFull(a, ev);
  // Keep writing B (self-eviction would succeed). put_seq climbs past
  // stale's youngest (16) + window (6): the stale A extent becomes cold and is
  // stolen in preference to evicting a hot page.
  bool stole_stale = false;
  for (int i = 32; i < 80 && !stole_stale; ++i) {
    ev.clear();
    ASSERT_TRUE(a.Put(Key("hot" + std::to_string(i)), 4096, &r, &ev));
    for (const auto& k : ev)
      if (KeyName(k).rfind("stale", 0) == 0) stole_stale = true;
  }
  EXPECT_TRUE(stole_stale) << "a globally-cold cross-class extent was reclaimed";
  EXPECT_GE(a.ColdSteals(), 1u);
  ::unsetenv("DFKV_SLAB_COLD_STEAL_WINDOW");
}

TEST(SlabAllocator, ColdStealDisabledFallsBackToSelfEvict) {
  ::setenv("DFKV_SLAB_COLD_STEAL_WINDOW", "0", 1);  // disable
  SlabAllocator a(Opts(4 * 4096, 16));
  std::vector<BlockKey> ev;
  SlotRef r;
  FillTwoClassesFull(a, ev);
  for (int i = 32; i < 96; ++i)  // busy class self-evicts its own only
    ASSERT_TRUE(a.Put(Key("hot" + std::to_string(i)), 4096, &r, &ev));
  EXPECT_EQ(a.ColdSteals(), 0u) << "disabled: never cold-steals";
  EXPECT_TRUE(a.Contains(Key("stale0"))) << "stale cross-class data survives";
  ::unsetenv("DFKV_SLAB_COLD_STEAL_WINDOW");
}

TEST(SlabAllocator, EvictColdToTargetFreesGloballyColdestFirst) {
  // Phase 10: proactive watermark eviction. Fill the pool, then evict down to
  // a target — the OLDEST data must go first, newest survives.
  SlabAllocator a(Opts(4 * 4096, 8));  // 8 extents, 32 4096-slots
  std::vector<BlockKey> ev;
  SlotRef r;
  for (int i = 0; i < 32; ++i)  // fill: k0 oldest .. k31 newest
    ASSERT_TRUE(a.Put(Key("k" + std::to_string(i)), 4096, &r, &ev));
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
    int n = std::stoi(KeyName(k).substr(1));
    EXPECT_LT(n, 32) << KeyName(k);
  }
  EXPECT_TRUE(a.Contains(Key("k31"))) << "newest data must survive proactive eviction";
}

TEST(SlabAllocator, EvictColdToTargetRespectsPinsAndTarget) {
  SlabAllocator a(Opts(4 * 4096, 4));  // 4 extents, 16 slots
  std::vector<BlockKey> ev;
  SlotRef r;
  for (int i = 0; i < 16; ++i) ASSERT_TRUE(a.Put(Key("k" + std::to_string(i)), 4096, &r, &ev));
  ASSERT_TRUE(a.Pin(Key("k0")));  // pins k0's whole extent against eviction
  ev.clear();
  const uint64_t used0 = a.UsedBytes();
  // Target 0 would want to evict everything, but pinned extents can't be freed.
  a.EvictColdToTarget(0, /*max_extents=*/4, &ev);
  EXPECT_TRUE(a.Contains(Key("k0"))) << "pinned key never evicted";
  EXPECT_LT(a.UsedBytes(), used0) << "some cold extents freed";
}

TEST(SlabAllocator, OversizeValueRejected) {
  SlabAllocator a(Opts(4096, 2));  // extent holds one 4096 slot
  std::vector<BlockKey> ev;
  SlotRef r;
  EXPECT_FALSE(a.Put(Key("big"), 4097, &r, &ev)) << "value larger than an extent";
  EXPECT_TRUE(a.Put(Key("ok"), 4096, &r, &ev));
  EXPECT_FALSE(a.Put(Key("overflow"), std::numeric_limits<size_t>::max(), &r, &ev));
}

TEST(SlabAllocator, ZeroLenIsRejectedWithoutAllocatingSlot) {
  SlabAllocator a(Opts(64 * 1024, 2));
  std::vector<BlockKey> ev;
  SlotRef r;
  EXPECT_FALSE(a.Put(Key("z"), 0, &r, &ev));
  EXPECT_FALSE(a.Contains(Key("z")));
  EXPECT_EQ(a.Count(), 0u);
  EXPECT_EQ(a.UsedBytes(), 0u);
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
      std::vector<BlockKey> ev;
      SlotRef r;
      for (int i = 0; i < N; ++i) {
        std::string k = "k" + std::to_string((t * N + i) % 5000);
        if (a.Put(Key(k), 4096, &r, &ev)) ok_puts.fetch_add(1);
        a.Get(Key(k), &r);
        if ((i & 3) == 0) a.Pin(Key(k));
        if ((i & 3) == 0) a.Unpin(Key(k));
        if ((i & 7) == 0) a.Remove(Key(k));
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
  EXPECT_TRUE(a.RestoreBulk({{Key("ka"), 4096, 0, 1, 4096}}));
  EXPECT_TRUE(a.RestoreBulk({{Key("kb"), 4096, 0, 2, 4096}}));
  EXPECT_EQ(a.Count(), 2u);
  EXPECT_EQ(a.UsedBytes(), 2u * 4096u);
  SlotRef r;
  ASSERT_TRUE(a.Get(Key("ka"), &r));
  EXPECT_EQ(r.extent, 0u);
  EXPECT_EQ(r.slot, 1u);
  EXPECT_EQ(r.offset, 1u * 4096u);
  // A subsequent Put on the same extent must use a still-free slot, not clobber
  // the restored ones.
  std::vector<BlockKey> ev;
  ASSERT_TRUE(a.Put(Key("kc"), 4096, &r, &ev));
  EXPECT_TRUE(ev.empty());
  EXPECT_NE(r.slot, 1u);
  EXPECT_NE(r.slot, 2u);
}

TEST(SlabAllocator, RestoreRejectsInconsistentInput) {
  SlabAllocator a(Opts(4 * 4096, 1));
  const size_t startup_classes = a.ClassCount();
  EXPECT_FALSE(a.RestoreBulk(
      {{Key("noncanonical"), 12 * 1024, 0, 0, 12 * 1024}}));
  EXPECT_EQ(a.ClassCount(), startup_classes);
  EXPECT_FALSE(a.RestoreBulk({{Key("bad_extent"), 4096, 9, 0, 4096}}));
  EXPECT_FALSE(a.RestoreBulk({{Key("bad_slot"), 4096, 0, 99, 4096}}));
  EXPECT_TRUE(a.RestoreBulk({{Key("k"), 4096, 0, 0, 4096}}));
  EXPECT_FALSE(a.RestoreBulk({{Key("k"), 4096, 0, 1, 4096}}));   // duplicate key
  EXPECT_FALSE(a.RestoreBulk({{Key("k2"), 4096, 0, 0, 4096}}));  // slot already taken
  // A second class on the same extent is a persistence inconsistency.
  EXPECT_FALSE(a.RestoreBulk({{Key("k3"), 8192, 0, 0, 8192}}));
}

TEST(SlabAllocator, BulkRestoreBuildsExactSlotsAndUsefulByteStats) {
  SlabAllocator allocator(Opts(4 * 4096, 2));
  std::vector<SlabAllocator::RestoreEntry> records{
      {Key("a"), 4096, 0, 1, 101},
      {Key("b"), 4096, 0, 3, 2048},
      {Key("c"), 8192, 1, 0, 7000},
  };
  ASSERT_TRUE(allocator.RestoreBulk(records));
  EXPECT_EQ(allocator.Count(), 3u);
  EXPECT_EQ(allocator.UsedBytes(), 2u * 4096 + 8192);

  SlotRef ref;
  ASSERT_TRUE(allocator.Get(Key("b"), &ref));
  EXPECT_EQ(ref.extent, 0u);
  EXPECT_EQ(ref.slot, 3u);
  auto stats = allocator.Classes();
  uint64_t useful = 0;
  uint64_t touches = 0;
  for (const auto& stat : stats) {
    useful += stat.useful_bytes;
    touches += stat.read_touches;
  }
  EXPECT_EQ(useful, 101u + 2048u + 7000u);
  EXPECT_EQ(touches, 1u);

  // A bad batch is rejected before any of it becomes resident.
  std::vector<SlabAllocator::RestoreEntry> bad{
      {Key("new"), 4096, 0, 0, 1},
      {Key("collision"), 4096, 0, 1, 1},
  };
  EXPECT_FALSE(allocator.RestoreBulk(bad));
  EXPECT_FALSE(allocator.Contains(Key("new")));
  EXPECT_EQ(allocator.Count(), 3u);
}

// Consecutive Puts must STRIPE across extents (different backing files): a
// single-stack free list hands out one extent's slots back-to-back, funneling
// every concurrent writer into one inode, and buffered writes to one file
// serialize on the kernel's per-inode lock.
TEST(SlabAllocator, ConsecutivePutsStripeAcrossExtents) {
  SlabAllocator a(Opts(4 * 4096, 4));  // 4 extents x 4 slots of one class
  std::vector<BlockKey> ev;
  SlotRef r;
  std::vector<uint32_t> extents;
  for (int i = 0; i < 4; ++i) {
    ASSERT_TRUE(a.Put(Key("k" + std::to_string(i)), 4096, &r, &ev));
    extents.push_back(r.extent);
  }
  // First 4 Puts land on 4 DISTINCT extents (pool has 4, stripe width >= 4).
  std::sort(extents.begin(), extents.end());
  extents.erase(std::unique(extents.begin(), extents.end()), extents.end());
  EXPECT_EQ(extents.size(), 4u) << "puts funneled into fewer inodes than available";
  // The rotation keeps cycling once all extents are bound.
  ASSERT_TRUE(a.Put(Key("k4"), 4096, &r, &ev));
  uint32_t e4 = r.extent;
  ASSERT_TRUE(a.Put(Key("k5"), 4096, &r, &ev));
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
  std::vector<BlockKey> ev;
  SlotRef r;
  for (int i = 0; i < 12; ++i)
    ASSERT_TRUE(a.Put(Key("k" + std::to_string(i)), 4096, &r, &ev));
  EXPECT_EQ(a.BoundExtents(), 12u);
  EXPECT_EQ(a.ExtentReturns(), 0u);
  // Free them all: each Remove makes that extent fully free; once the rotation
  // exceeds 8, the surplus fully-free extents unbind back to the pool.
  for (int i = 0; i < 12; ++i) a.Remove(Key("k" + std::to_string(i)));
  EXPECT_GT(a.ExtentReturns(), 0u);
  EXPECT_LT(a.BoundExtents(), 12u);
  // A NEW size class (smaller slot => different class) binds a returned pool
  // extent without stealing/evicting.
  SlabAllocator b(Opts(8 * 1024, 12, 1024));
  for (int i = 0; i < 12; ++i)
    ASSERT_TRUE(b.Put(Key("k" + std::to_string(i)), 8 * 1024, &r, &ev));  // fill: 1 slot/extent
  for (int i = 0; i < 12; ++i) b.Remove(Key("k" + std::to_string(i)));
  EXPECT_GT(b.ExtentReturns(), 0u);
  ASSERT_TRUE(b.Put(Key("small"), 1024, &r, &ev));  // new class, binds from pool
  EXPECT_TRUE(ev.empty()) << "returned-pool bind must not evict";
  EXPECT_EQ(b.Steals(), 0u);
}

// Explicit Remove is logically immediate but physical reuse waits for the last
// pin: an unlocked reader/writer can never observe a recycled slot.
TEST(SlabAllocator, RemoveDefersPinnedSlotUntilLastUnpin) {
  SlabAllocator a(Opts(4096, 1));  // one slot total
  std::vector<BlockKey> ev;
  SlotRef r1, r2;
  ASSERT_TRUE(a.Put(Key("k"), 4096, &r1, &ev));
  ASSERT_TRUE(a.Pin(Key("k")));
  ASSERT_TRUE(a.Pin(Key("k")));
  EXPECT_EQ(a.Remove(Key("k")), SlabAllocator::RemoveResult::kDeferred);
  EXPECT_FALSE(a.Contains(Key("k")));
  EXPECT_FALSE(a.Get(Key("k"), &r2));
  EXPECT_FALSE(a.Pin(Key("k")));
  EXPECT_EQ(a.Count(), 1u);
  EXPECT_EQ(a.UsedBytes(), 4096u);
  EXPECT_FALSE(a.Put(Key("k2"), 4096, &r2, &ev));
  ASSERT_TRUE(a.Unpin(Key("k")));
  EXPECT_EQ(a.Count(), 1u);
  EXPECT_FALSE(a.Put(Key("k2"), 4096, &r2, &ev));
  ASSERT_TRUE(a.Unpin(Key("k")));
  EXPECT_EQ(a.Count(), 0u);
  EXPECT_EQ(a.UsedBytes(), 0u);
  ASSERT_TRUE(a.Put(Key("k2"), 4096, &r2, &ev));
  EXPECT_TRUE(ev.empty());
  EXPECT_EQ(r2.extent, r1.extent);
  EXPECT_EQ(r2.slot, r1.slot);
}

// ---- background-reclaimer + resident-list additions (put-path deserialization) ----

// Classes() reports per-class free/resident/puts truthfully across put, evict,
// remove, and extent hand-offs (the reclaimer's decisions ride on these counts).
TEST(SlabAllocator, ClassStatsTrackFreeResidentAndPuts) {
  SlabAllocator a(Opts(4 * 4096, 4));  // 16 slots of one 4096 class
  std::vector<BlockKey> ev;
  SlotRef r;
  for (int i = 0; i < 10; ++i)
    ASSERT_TRUE(a.Put(Key("k" + std::to_string(i)), 4096, &r, &ev));
  auto cs = a.Classes();
  ASSERT_FALSE(cs.empty());
  EXPECT_EQ(cs[0].slot_size, 4096u);
  EXPECT_EQ(cs[0].resident, 10u);
  EXPECT_EQ(cs[0].free_slots, 6u);
  EXPECT_EQ(cs[0].puts, 10u);
  ASSERT_TRUE(a.Put(Key("k0"), 4096, &r, &ev));  // idempotent hit: NOT a new insert
  EXPECT_EQ(a.Classes()[0].puts, 10u);
  a.Remove(Key("k9"));
  cs = a.Classes();
  EXPECT_EQ(cs[0].resident, 9u);
  EXPECT_EQ(cs[0].free_slots, 7u);
}

TEST(SlabAllocator, ReadHotByteDonorIsNotStolen) {
  SlabAllocator allocator(Opts(16 * 1024, 2));
  std::vector<SlabAllocator::RestoreEntry> records{
      {Key("hot-small"), 4096, 0, 0, 512},
      {Key("cold-large"), 8192, 1, 0, 7000},
  };
  ASSERT_TRUE(allocator.RestoreBulk(records));
  SlotRef ref;
  for (int i = 0; i < 16; ++i) ASSERT_TRUE(allocator.Get(Key("hot-small"), &ref));

  std::vector<BlockKey> evicted;
  ASSERT_TRUE(allocator.Put(Key("target"), 16 * 1024, &ref, &evicted));
  EXPECT_TRUE(allocator.Contains(Key("hot-small")));
  EXPECT_FALSE(allocator.Contains(Key("cold-large")));
  EXPECT_EQ(evicted, std::vector<BlockKey>({Key("cold-large")}));

  const auto stats = allocator.Classes();
  auto hot = std::find_if(stats.begin(), stats.end(),
                          [](const SlabAllocator::ClassStat& stat) {
                            return stat.slot_size == 4096;
                          });
  ASSERT_NE(hot, stats.end());
  EXPECT_EQ(hot->useful_bytes, 512u);
  EXPECT_EQ(hot->read_touches, 16u);
  EXPECT_GT(hot->read_heat, 0u);
}

// ReclaimClass evicts ahead of demand: it frees slots up to the target, in
// CLOCK order, without touching pinned entries, and reports the victims so the
// caller can drop them from its own index.
TEST(SlabAllocator, ReclaimClassCreatesHeadroomSkippingPinned) {
  SlabAllocator a(Opts(4 * 4096, 2));  // 8 slots, one class
  std::vector<BlockKey> ev;
  SlotRef r;
  for (int i = 0; i < 8; ++i)
    ASSERT_TRUE(a.Put(Key("k" + std::to_string(i)), 4096, &r, &ev));
  ASSERT_TRUE(a.Pin(Key("k0")));
  ASSERT_TRUE(a.Pin(Key("k1")));
  std::vector<BlockKey> victims;
  const size_t got = a.ReclaimClass(0, /*target_free=*/3, /*max_victims=*/8, &victims);
  EXPECT_EQ(got, 3u);
  EXPECT_EQ(victims.size(), 3u);
  for (const auto& v : victims) {
    EXPECT_FALSE(v == Key("k0"));
    EXPECT_FALSE(v == Key("k1"));
    EXPECT_FALSE(a.Contains(Key(v)));
  }
  EXPECT_EQ(a.Classes()[0].free_slots, 3u);
  // A follow-up Put takes a reclaimed slot WITHOUT evicting inline.
  ev.clear();
  ASSERT_TRUE(a.Put(Key("fresh"), 4096, &r, &ev));
  EXPECT_TRUE(ev.empty()) << "put should ride the reclaimed headroom";
}

// ReclaimClass respects max_victims (bounded work per lock hold) and is a no-op
// when the class already holds the target headroom.
TEST(SlabAllocator, ReclaimClassBoundedAndIdempotent) {
  SlabAllocator a(Opts(4 * 4096, 2));
  std::vector<BlockKey> ev;
  SlotRef r;
  for (int i = 0; i < 8; ++i)
    ASSERT_TRUE(a.Put(Key("k" + std::to_string(i)), 4096, &r, &ev));
  std::vector<BlockKey> victims;
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
  std::vector<BlockKey> ev;
  SlotRef r;
  for (int i = 0; i < 12; ++i)
    ASSERT_TRUE(a.Put(Key("k" + std::to_string(i)), 4096, &r, &ev));
  for (int i = 0; i < 8; ++i) a.Remove(Key("k" + std::to_string(i)));
  ASSERT_EQ(a.ExtentReturns(), 0u);
  ASSERT_EQ(a.Classes()[0].free_slots, 8u);
  std::vector<BlockKey> victims;
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
  std::vector<BlockKey> ev;
  SlotRef r;
  std::vector<std::string> ext_keys[2];
  for (int i = 0; i < 8; ++i) {
    const std::string k = "k" + std::to_string(i);
    ASSERT_TRUE(a.Put(Key(k), 4096, &r, &ev));
    ext_keys[r.extent].push_back(k);
  }
  // A new class (16384 > 4096/0.25 waste bound) finds no pool extent -> steal.
  ev.clear();
  ASSERT_TRUE(a.Put(Key("big"), 16 * 1024, &r, &ev));
  EXPECT_EQ(a.Steals(), 1u);
  EXPECT_EQ(ev.size(), 4u) << "exactly one extent's residents evicted";
  const uint32_t stolen = r.extent;
  std::set<std::string> gone;
  for (const BlockKey& key : ev) gone.insert(KeyName(key));
  for (const auto& k : ext_keys[stolen]) EXPECT_TRUE(gone.count(k)) << k;
  for (const auto& k : ext_keys[1 - stolen]) {
    EXPECT_FALSE(gone.count(k)) << k;
    EXPECT_TRUE(a.Contains(Key(k))) << k;
  }
}

// Restore populates the resident list too: a steal after a rebuild must evict
// the restored keys of the stolen extent (they live only in the extent list --
// a regression here silently leaks slots).
TEST(SlabAllocator, StealAfterRestoreEvictsRestoredResidents) {
  SlabAllocator a(Opts(4 * 4096, 1));  // one extent x 4 slots
  for (int i = 0; i < 4; ++i)
    ASSERT_TRUE(a.RestoreBulk(
        {{Key("k" + std::to_string(i)), 4096, 0,
          static_cast<uint32_t>(i), 4096}}));
  EXPECT_EQ(a.Count(), 4u);
  std::vector<BlockKey> ev;
  SlotRef r;
  ASSERT_TRUE(a.Put(Key("big"), 16 * 1024, &r, &ev));  // must steal extent 0
  EXPECT_EQ(a.Steals(), 1u);
  EXPECT_EQ(ev.size(), 4u);
  EXPECT_EQ(a.Count(), 1u);
  for (int i = 0; i < 4; ++i)
    EXPECT_FALSE(a.Contains(Key("k" + std::to_string(i))));
}

// ---- class rebalance additions (StealFrom mechanism) ----

// StealFrom moves exactly one extent: the donor's residents on that extent are
// evicted, the target class gains its capacity, everything else survives.
TEST(SlabAllocator, StealFromMovesOneExtentDonorToTarget) {
  SlabAllocator a(Opts(4 * 4096, 3));  // 3 extents x 4 slots
  std::vector<BlockKey> ev;
  SlotRef r;
  for (int i = 0; i < 12; ++i)
    ASSERT_TRUE(a.Put(Key("k" + std::to_string(i)), 4096, &r, &ev));  // class 0 owns all
  const size_t donor = ClassIndex(a, 4096);
  const size_t target = ClassIndex(a, 16 * 1024);
  ASSERT_LT(donor, a.ClassCount());
  ASSERT_LT(target, a.ClassCount());
  // The target class already exists in the startup lattice. A 16 KiB Put
  // steals inline because the shared pool is full.
  ev.clear();
  ASSERT_TRUE(a.Put(Key("big0"), 16 * 1024, &r, &ev));
  const size_t evicted_inline = ev.size();
  ASSERT_EQ(evicted_inline, 4u);
  auto cs = a.Classes();
  ASSERT_EQ(cs[donor].extents, 2u);
  ASSERT_EQ(cs[target].extents, 1u);
  ev.clear();
  ASSERT_TRUE(a.StealFrom(donor, target, &ev));
  EXPECT_EQ(ev.size(), 4u) << "exactly the donor extent's residents";
  cs = a.Classes();
  EXPECT_EQ(cs[donor].extents, 1u);
  EXPECT_EQ(cs[target].extents, 2u);
  EXPECT_EQ(cs[donor].resident, 4u);
  EXPECT_EQ(cs[target].free_slots, 1u);
  // Survivors intact.
  size_t alive = 0;
  for (int i = 0; i < 12; ++i) alive += a.Contains(Key("k" + std::to_string(i)));
  EXPECT_EQ(alive, 4u);
  EXPECT_TRUE(a.Contains(Key("big0")));
}

TEST(SlabAllocator, BindPersistenceFailurePublishesNoSlots) {
  bool allow_bind = false;
  auto o = Opts(4 * 4096, 2);
  o.on_extent_bind = [&](uint32_t) { return allow_bind; };
  SlabAllocator a(o);
  std::vector<BlockKey> ev;
  SlotRef ref;

  EXPECT_FALSE(a.Put(Key("k"), 4096, &ref, &ev));
  EXPECT_TRUE(ev.empty());
  EXPECT_EQ(a.Count(), 0u);
  EXPECT_EQ(a.UsedBytes(), 0u);
  EXPECT_EQ(a.BoundExtents(), 0u);
  EXPECT_EQ(a.PoolExtents(), 2u);

  allow_bind = true;
  ASSERT_TRUE(a.Put(Key("k"), 4096, &ref, &ev));
  EXPECT_EQ(a.Count(), 1u);
  EXPECT_EQ(a.BoundExtents(), 2u);
}

TEST(SlabAllocator, EvictionPersistenceFailurePreservesResidentSlot) {
  bool allow_evict = false;
  auto options = Opts(4 * 4096, 1);
  options.on_slot_evict =
      [&](const SlotRef&) { return allow_evict; };
  SlabAllocator allocator(options);
  std::vector<BlockKey> evicted;
  SlotRef ref;
  for (int i = 0; i < 4; ++i)
    ASSERT_TRUE(allocator.Put(Key("k" + std::to_string(i)), 4096, &ref,
                              &evicted));

  evicted.clear();
  EXPECT_FALSE(allocator.Put(Key("blocked"), 4096, &ref, &evicted));
  EXPECT_TRUE(evicted.empty());
  EXPECT_EQ(allocator.Count(), 4u);
  for (int i = 0; i < 4; ++i)
    EXPECT_TRUE(allocator.Contains(Key("k" + std::to_string(i))));

  allow_evict = true;
  ASSERT_TRUE(allocator.Put(Key("landed"), 4096, &ref, &evicted));
  EXPECT_EQ(evicted.size(), 1u);
  EXPECT_TRUE(allocator.Contains(Key("landed")));
  EXPECT_EQ(allocator.Count(), 4u);
}

TEST(SlabAllocator, RebindPersistenceFailurePreservesDonorExtent) {
  bool allow_bind = true;
  auto o = Opts(4 * 4096, 2);
  o.on_extent_bind = [&](uint32_t) { return allow_bind; };
  SlabAllocator a(o);
  std::vector<BlockKey> ev;
  SlotRef ref;
  for (int i = 0; i < 8; ++i)
    ASSERT_TRUE(a.Put(Key("a" + std::to_string(i)), 4096, &ref, &ev));

  allow_bind = false;
  ev.clear();
  EXPECT_FALSE(a.Put(Key("big"), 8192, &ref, &ev));
  EXPECT_TRUE(ev.empty());
  EXPECT_EQ(a.Count(), 8u);
  EXPECT_EQ(a.BoundExtents(), 2u);
  for (int i = 0; i < 8; ++i)
    EXPECT_TRUE(a.Contains(Key("a" + std::to_string(i))));

  allow_bind = true;
  ASSERT_TRUE(a.Put(Key("big"), 8192, &ref, &ev));
  EXPECT_EQ(ev.size(), 4u);
  EXPECT_TRUE(a.Contains(Key("big")));
}

// StealFrom refuses: donor == target, bad indices, oversized target class, and
// a donor whose every extent holds a pin.
TEST(SlabAllocator, StealFromRefusalCases) {
  SlabAllocator a(Opts(4 * 4096, 2));
  std::vector<BlockKey> ev;
  SlotRef r;
  for (int i = 0; i < 8; ++i)
    ASSERT_TRUE(a.Put(Key("k" + std::to_string(i)), 4096, &r, &ev));
  ASSERT_TRUE(a.Put(Key("big"), 16 * 1024, &r, &ev));
  const size_t donor = ClassIndex(a, 4096);
  const size_t target = ClassIndex(a, 16 * 1024);
  const size_t invalid = a.ClassCount() + 1;
  EXPECT_FALSE(a.StealFrom(donor, donor, &ev));
  EXPECT_FALSE(a.StealFrom(invalid, target, &ev));
  EXPECT_FALSE(a.StealFrom(donor, invalid, &ev));
  // Pin one resident on the donor's remaining extent.
  for (int i = 0; i < 8; ++i)
    if (a.Contains(Key("k" + std::to_string(i)))) { ASSERT_TRUE(a.Pin(Key("k" + std::to_string(i)))); break; }
  ev.clear();
  EXPECT_FALSE(a.StealFrom(donor, target, &ev));
  EXPECT_TRUE(ev.empty());
}

// ClassStat.extents stays truthful across bind, inline steal, StealFrom, and
// fully-free extent returns (the rebalance policy reads it every tick).
TEST(SlabAllocator, ClassStatsExtentsTracksHandoffs) {
  SlabAllocator a(Opts(4096, 12));  // 12 extents x 1 slot
  std::vector<BlockKey> ev;
  SlotRef r;
  for (int i = 0; i < 12; ++i)
    ASSERT_TRUE(a.Put(Key("k" + std::to_string(i)), 4096, &r, &ev));
  EXPECT_EQ(a.Classes()[0].extents, 12u);
  EXPECT_EQ(a.PoolExtents(), 0u);
  for (int i = 0; i < 9; ++i) a.Remove(Key("k" + std::to_string(i)));  // returns fire past 8
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
  std::vector<BlockKey> ev;
  SlotRef r;
  for (int i = 0; i < 64; ++i)
    ASSERT_TRUE(a.Put(Key("a" + std::to_string(i)), 4096, &r, &ev));
  ASSERT_EQ(a.Classes()[0].extents, 16u);
  // Class B (16 KiB, 1 slot/extent): burst of 6. Every put after the first
  // extent fills must STEAL (donor A: 16 > 8), never evict B's own residents.
  ev.clear();
  for (int i = 0; i < 6; ++i)
    ASSERT_TRUE(a.Put(Key("b" + std::to_string(i)), 16 * 1024, &r, &ev));
  for (int i = 0; i < 6; ++i)
    EXPECT_TRUE(a.Contains(Key("b" + std::to_string(i)))) << "b" << i << " self-evicted";
  auto cs = a.Classes();
  const size_t target = ClassIndex(a, 16 * 1024);
  ASSERT_LT(target, cs.size());
  EXPECT_EQ(cs[target].resident, 6u);
  EXPECT_EQ(cs[target].extents, 6u);
  EXPECT_EQ(a.Steals(), 6u);
}

// Donor floor on the growth-first path: when every other class is at or below
// kStripeWays extents, a bootstrapping class must NOT steal (no ping-pong
// between two under-provisioned classes) -- it falls back to self-eviction.
TEST(SlabAllocator, GrowthFirstRespectsDonorFloor) {
  // 8 extents x 4 slots, all bound to class A (exactly kStripeWays -> not a donor).
  SlabAllocator a(Opts(4 * 4096, 8));
  std::vector<BlockKey> ev;
  SlotRef r;
  for (int i = 0; i < 32; ++i)
    ASSERT_TRUE(a.Put(Key("a" + std::to_string(i)), 4096, &r, &ev));
  ASSERT_EQ(a.Classes()[0].extents, 8u);
  // Class B: first put has no self to evict -> LAST-RESORT steal (floor 0) is
  // allowed and takes one A extent (A drops to 7).
  ev.clear();
  ASSERT_TRUE(a.Put(Key("b0"), 16 * 1024, &r, &ev));
  EXPECT_EQ(a.Steals(), 1u);
  // Second put: B(1 extent, full) is bootstrapping, but A(7) is at/below the
  // floor -> growth-first refuses; self-eviction evicts b0.
  ev.clear();
  ASSERT_TRUE(a.Put(Key("b1"), 16 * 1024, &r, &ev));
  EXPECT_EQ(a.Steals(), 1u) << "must not steal from a donor at/below the floor";
  ASSERT_EQ(ev.size(), 1u);
  EXPECT_EQ(ev[0], Key("b0"));
  EXPECT_EQ(a.Classes()[0].extents, 7u) << "A must not shrink further";
}

// A class at or above kStripeWays extents behaves exactly as before: steady-
// state churn is self-eviction, not stealing (growth-first is bootstrap-only).
TEST(SlabAllocator, MatureClassStillSelfEvicts) {
  SlabAllocator a(Opts(4096, 16));  // 16 extents x 1 slot, one class
  std::vector<BlockKey> ev;
  SlotRef r;
  for (int i = 0; i < 16; ++i)
    ASSERT_TRUE(a.Put(Key("k" + std::to_string(i)), 4096, &r, &ev));
  ASSERT_EQ(a.Classes()[0].extents, 16u);
  ev.clear();
  ASSERT_TRUE(a.Put(Key("k16"), 4096, &r, &ev));  // full + mature -> CLOCK evict
  EXPECT_EQ(ev.size(), 1u);
  EXPECT_EQ(a.Steals(), 0u);
}

TEST(SlabAllocator, FlatIndexCollisionEraseAndTombstoneReuse) {
  SlabAllocator allocator(Opts(64 * 4096, 1));
  std::vector<BlockKey> evicted;
  SlotRef ref;
  const BlockKey anchor = Key("anchor");
  ASSERT_TRUE(allocator.Put(anchor, 4096, &ref, &evicted));
  const size_t bucket = allocator.IndexBucketForTest(anchor);

  BlockKey first_collision;
  BlockKey second_collision;
  for (uint64_t i = 0; i < 10000 && second_collision == BlockKey{}; ++i) {
    const BlockKey candidate = Key("collision-" + std::to_string(i));
    if (candidate == anchor ||
        allocator.IndexBucketForTest(candidate) != bucket)
      continue;
    if (first_collision == BlockKey{})
      first_collision = candidate;
    else
      second_collision = candidate;
  }
  ASSERT_FALSE(first_collision == BlockKey{});
  ASSERT_FALSE(second_collision == BlockKey{});
  ASSERT_TRUE(allocator.Put(first_collision, 4096, &ref, &evicted));
  EXPECT_TRUE(allocator.Get(anchor, &ref));
  EXPECT_TRUE(allocator.Get(first_collision, &ref));

  ASSERT_EQ(allocator.Remove(anchor), SlabAllocator::RemoveResult::kRemoved);
  ASSERT_EQ(allocator.IndexStatsForTest().tombstones, 1u);
  ASSERT_TRUE(allocator.Put(second_collision, 4096, &ref, &evicted));
  EXPECT_EQ(allocator.IndexStatsForTest().tombstones, 0u);
  EXPECT_TRUE(allocator.Contains(first_collision));
  EXPECT_TRUE(allocator.Contains(second_collision));
}

TEST(SlabAllocator, FlatIndexGrowsAndCompactsTombstones) {
  SlabAllocator allocator(Opts(256 * 4096, 1));
  std::vector<BlockKey> evicted;
  SlotRef ref;
  for (int i = 0; i < 11; ++i)
    ASSERT_TRUE(allocator.Put(Key("key-" + std::to_string(i)), 4096, &ref,
                              &evicted));
  const auto before_remove = allocator.IndexStatsForTest();
  ASSERT_EQ(before_remove.capacity, 16u);
  for (int i = 0; i < 5; ++i)
    ASSERT_EQ(allocator.Remove(Key("key-" + std::to_string(i))),
              SlabAllocator::RemoveResult::kRemoved);
  const auto tombstoned = allocator.IndexStatsForTest();
  ASSERT_EQ(tombstoned.tombstones, 5u);
  ASSERT_TRUE(
      allocator.Put(Key("replacement"), 4096, &ref, &evicted));
  const auto compacted = allocator.IndexStatsForTest();
  EXPECT_EQ(compacted.capacity, tombstoned.capacity);
  EXPECT_EQ(compacted.tombstones, 0u);
  EXPECT_EQ(compacted.rehashes, tombstoned.rehashes + 1);

  for (int i = 11; i < 200; ++i)
    ASSERT_TRUE(allocator.Put(Key("key-" + std::to_string(i)), 4096, &ref,
                              &evicted));
  const auto grown = allocator.IndexStatsForTest();
  EXPECT_GT(grown.capacity, compacted.capacity);
  EXPECT_GT(grown.rehashes, compacted.rehashes);
  for (int i = 5; i < 200; ++i)
    EXPECT_TRUE(allocator.Contains(Key("key-" + std::to_string(i))));
}

TEST(SlabAllocator, BulkRestoreBuildsFlatIndexInOneSetupRehash) {
  constexpr uint32_t kRecords = 300;
  SlabAllocator allocator(Opts(512 * 4096, 1));
  std::vector<SlabAllocator::RestoreEntry> records;
  records.reserve(kRecords);
  for (uint32_t i = 0; i < kRecords; ++i)
    records.push_back(
        {Key("restore-" + std::to_string(i)), 4096, 0, i, 100 + i});

  ASSERT_TRUE(allocator.RestoreBulk(records));
  const auto stats = allocator.IndexStatsForTest();
  EXPECT_EQ(stats.size, kRecords);
  EXPECT_EQ(stats.rehashes, 1u);
  EXPECT_EQ(stats.table_allocations, 1u);
  SlotRef ref;
  for (uint32_t i = 0; i < kRecords; ++i) {
    ASSERT_TRUE(allocator.Get(Key("restore-" + std::to_string(i)), &ref));
    EXPECT_EQ(ref.extent, 0u);
    EXPECT_EQ(ref.slot, i);
  }
}

TEST(SlabAllocator, SteadyLookupUpdateAndRemoveDoNotAllocateIndexTables) {
  SlabAllocator allocator(Opts(128 * 4096, 1));
  std::vector<BlockKey> evicted;
  SlotRef ref;
  for (int i = 0; i < 64; ++i)
    ASSERT_TRUE(allocator.Put(Key("steady-" + std::to_string(i)), 4096, &ref,
                              &evicted));
  const auto steady = allocator.IndexStatsForTest();
  ASSERT_GT(steady.table_allocations, 0u);

  for (int pass = 0; pass < 100; ++pass) {
    for (int i = 0; i < 64; ++i) {
      const BlockKey key = Key("steady-" + std::to_string(i));
      ASSERT_TRUE(allocator.Get(key, &ref));
      ASSERT_TRUE(allocator.Contains(key));
      ASSERT_TRUE(allocator.Put(key, 4096, &ref, &evicted));
    }
  }
  ASSERT_EQ(allocator.Remove(Key("steady-0")),
            SlabAllocator::RemoveResult::kRemoved);
  const auto after = allocator.IndexStatsForTest();
  EXPECT_EQ(after.table_allocations, steady.table_allocations);
  EXPECT_EQ(after.rehashes, steady.rehashes);
}
