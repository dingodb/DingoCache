// SlabAllocator: media-agnostic slot lifecycle. Pure logic, hermetic tests.
// Covers: alloc/get/remove, idempotent put, size-class reuse vs new, per-class
// CLOCK eviction, extent binding + cross-class steal, pin-blocks-eviction, byte
// accounting, oversize rejection, and a concurrent (TSan) stress.
#include "cache/slab_allocator.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using dfkv::BlockKey;
using dfkv::SlabAllocator;
using SlotRef = SlabAllocator::SlotRef;

namespace dfkv {
// Reads the allocator's private state for the bulk-restore equivalence tests
// and carries the reference they compare against.
class SlabAllocatorTestPeer {
 public:
  // RestoreBulk as it was through release 2.37: per-entry hash set of
  // placements, sorted keys, and a linear search of the extent's free list
  // for every slot taken. The production function must leave the allocator
  // in exactly the state this one does.
  // Once a release has shipped with the new restore this copy can go, and
  // the comparison with it be replaced by a test of invariants (per-class
  // resident counts, free lists, hand positions, per-key lookups).
  static bool ReferenceRestore(
      SlabAllocator* a, const std::vector<SlabAllocator::RestoreEntry>& entries) {
    using RestoreEntry = SlabAllocator::RestoreEntry;
    if (entries.empty()) return true;
    std::vector<BlockKey> keys;
    std::set<uint64_t> placements;
    std::map<uint32_t, uint32_t> batch_extent_sizes;
    for (const RestoreEntry& entry : entries) {
      if (entry.slot_size == 0 || entry.extent >= a->extents_.size() ||
          entry.slot_size > a->opt_.extent_bytes ||
          entry.value_bytes > entry.slot_size ||
          a->CanonicalSlotSize(entry.slot_size) != entry.slot_size ||
          a->ClassForExactSize(entry.slot_size) == a->classes_.size() ||
          entry.slot >= a->opt_.extent_bytes / entry.slot_size ||
          a->index_.Find(entry.key) != nullptr)
        return false;
      keys.push_back(entry.key);
      const uint64_t placement =
          (static_cast<uint64_t>(entry.extent) << 32) | entry.slot;
      if (!placements.emplace(placement).second) return false;
      auto batch = batch_extent_sizes.emplace(entry.extent, entry.slot_size);
      if (!batch.second && batch.first->second != entry.slot_size) return false;
      const auto& extent = a->extents_[entry.extent];
      if (extent.cls != SlabAllocator::kUnbound) {
        const auto& cls = a->classes_[static_cast<size_t>(extent.cls)];
        if (cls.slot_size != entry.slot_size) return false;
        auto free_it = cls.free_by_ext.find(entry.extent);
        if (free_it == cls.free_by_ext.end() ||
            std::find(free_it->second.begin(), free_it->second.end(),
                      entry.slot) == free_it->second.end())
          return false;
      }
    }
    std::sort(keys.begin(), keys.end(),
              [](const BlockKey& x, const BlockKey& y) {
                if (x.tenant_hash != y.tenant_hash)
                  return x.tenant_hash < y.tenant_hash;
                if (x.digest_hi != y.digest_hi) return x.digest_hi < y.digest_hi;
                return x.digest_lo < y.digest_lo;
              });
    if (std::adjacent_find(keys.begin(), keys.end()) != keys.end()) return false;

    a->slots_.reserve(a->slots_.size() + entries.size());
    a->index_.Reserve(a->index_.size() + entries.size());
    for (const RestoreEntry& entry : entries) {
      const size_t cls_index = a->ClassForExactSize(entry.slot_size);
      if (a->extents_[entry.extent].cls == SlabAllocator::kUnbound) {
        if (a->BindExtentLocked(cls_index, entry.extent, true) !=
            SlabAllocator::ExtentBindResult::kBound)
          return false;
      }
      auto& cls = a->classes_[cls_index];
      {
        auto it = cls.free_by_ext.find(entry.extent);
        if (it == cls.free_by_ext.end()) return false;
        std::vector<uint32_t>& slots = it->second;
        auto found = std::find(slots.begin(), slots.end(), entry.slot);
        if (found == slots.end()) return false;
        *found = slots.back();
        slots.pop_back();
        --cls.free_count;
        if (slots.empty()) a->DropFromRotationLocked(cls, entry.extent);
      }
      const uint32_t meta_index = a->AcquireSlotMetaLocked();
      auto& meta = a->slots_[meta_index];
      meta.ref.cls = static_cast<uint32_t>(cls_index);
      meta.ref.extent = entry.extent;
      meta.ref.slot = entry.slot;
      meta.ref.slot_size = entry.slot_size;
      meta.ref.offset = static_cast<uint64_t>(entry.slot) * entry.slot_size;
      meta.useful_bytes = entry.value_bytes;
      meta.key = entry.key;
      meta.live = true;
      if (!a->index_.Insert(entry.key, meta_index)) return false;
      a->LinkResidentLocked(meta_index);
      auto& extent = a->extents_[entry.extent];
      --extent.free_slots;
      extent.useful_bytes += meta.useful_bytes;
      cls.useful_bytes += meta.useful_bytes;
      a->used_bytes_ += entry.slot_size;
      if (a->retain_.enabled && entry.retain_until > a->retain_now_s_ &&
          entry.retain_ttl_s != 0) {
        const uint32_t ttl = std::min(entry.retain_ttl_s, a->retain_.max_ttl_s);
        const uint64_t cap = static_cast<uint64_t>(a->retain_now_s_) + ttl;
        meta.retain_ttl_s = ttl;
        meta.retain_until = static_cast<uint32_t>(
            std::min<uint64_t>(entry.retain_until, cap));
        meta.retain_persisted_until = meta.retain_until;
        a->ProtectLocked(meta_index);
        a->PushWheelLocked(meta_index);
      } else if (a->retain_.enabled) {
        meta.retain_ttl_s = std::min(entry.retain_ttl_s, a->retain_.max_ttl_s);
      }
    }
    return true;
  }

  static std::vector<uint32_t> UnboundExtents(const SlabAllocator& a) {
    std::vector<uint32_t> out;
    for (size_t e = 0; e < a.extents_.size(); ++e)
      if (a.extents_[e].cls == SlabAllocator::kUnbound)
        out.push_back(static_cast<uint32_t>(e));
    return out;
  }

  static uint64_t IndexHash(const BlockKey& key) {
    return SlabAllocator::FlatKeyIndex::Hash(key);
  }

  // Free slots of bound extents as (extent, slot, slot size), in a fixed order.
  static std::vector<std::array<uint32_t, 3>> FreeSlots(const SlabAllocator& a) {
    std::vector<std::array<uint32_t, 3>> out;
    for (const auto& cls : a.classes_)
      for (const auto& [extent, slots] : cls.free_by_ext)
        for (uint32_t slot : slots) out.push_back({extent, slot, cls.slot_size});
    std::sort(out.begin(), out.end());
    return out;
  }

  // Every piece of state that decides what the allocator does next: free
  // lists in their order, the rotation, every ring and hand, every slot
  // record, the index layout, the retention wheel and the counters.
  static std::string Dump(const SlabAllocator& a) {
    std::ostringstream o;
    auto key = [&o](const BlockKey& k) {
      o << k.digest_hi << ':' << k.digest_lo << ':' << k.tenant_hash;
    };
    for (size_t c = 0; c < a.classes_.size(); ++c) {
      const auto& cls = a.classes_[c];
      o << "class " << c << ' ' << cls.slot_size << ' ' << cls.slots_per_extent
        << " rr_next=" << cls.rr_next << " free=" << cls.free_count
        << " resident=" << cls.resident_count << " bound=" << cls.bound_extents
        << " puts=" << cls.puts << " useful=" << cls.useful_bytes
        << " touches=" << cls.read_touches << " heat=" << cls.read_heat << '@'
        << cls.heat_seq << " head=" << cls.ring_head << " hand=" << cls.hand
        << " protected=" << cls.protected_slots << " clear_limited="
        << cls.reclaim_clear_limited << "\n rotation";
      for (uint32_t extent : cls.ext_rr) o << ' ' << extent;
      std::map<uint32_t, const std::vector<uint32_t>*> lists;
      for (const auto& [extent, slots] : cls.free_by_ext) lists[extent] = &slots;
      for (const auto& [extent, slots] : lists) {
        o << "\n free[" << extent << "]";
        for (uint32_t slot : *slots) o << ' ' << slot;
      }
      o << '\n';
    }
    for (size_t e = 0; e < a.extents_.size(); ++e) {
      const auto& x = a.extents_[e];
      o << "extent " << e << ' ' << x.cls << ' ' << x.free_slots << ' '
        << x.total_slots << ' ' << x.pinned << ' ' << x.youngest_seq << ' '
        << x.access_seq_sum << ' ' << x.useful_bytes << ' ' << x.read_heat
        << '@' << x.heat_seq << ' ' << x.resident_head << ' '
        << x.resident_count << ' ' << x.protected_slots << '\n';
    }
    for (size_t i = 0; i < a.slots_.size(); ++i) {
      const auto& m = a.slots_[i];
      o << "slot " << i << ' ';
      key(m.key);
      o << ' ' << m.ref.cls << ' ' << m.ref.extent << ' ' << m.ref.slot << ' '
        << m.ref.slot_size << ' ' << m.ref.offset << ' ' << m.useful_bytes
        << " g=" << m.global_prev << ',' << m.global_next << " f="
        << m.free_next << " t=" << m.read_touches << " a=" << m.access_seq
        << " r=" << m.refs << " c=" << m.class_prev << ',' << m.class_next
        << " e=" << m.extent_prev << ',' << m.extent_next << " ret="
        << m.retain_until << ',' << m.retain_ttl_s << ','
        << m.retain_persisted_until << ',' << m.retain_wheel_until << ' '
        << m.referenced << m.remove_pending
        << m.live;
      if (m.live) {
        const uint32_t* found = a.index_.Find(m.key);
        o << " index=" << (found ? static_cast<long long>(*found) : -1)
          << " bucket=" << a.index_.BucketForTest(m.key);
      }
      o << '\n';
    }
    const auto index = a.index_.stats();
    o << "index " << index.size << ' ' << index.capacity << ' '
      << index.tombstones << ' ' << index.table_allocations << ' '
      << index.rehashes << '\n';
    o << "heads " << a.free_slot_meta_head_ << ' ' << a.global_ring_head_ << ' '
      << a.global_hand_ << " unbound=" << a.unbound_ << " used="
      << a.used_bytes_ << " seq=" << a.put_seq_ << ',' << a.activity_seq_
      << " counters=" << a.evictions_ << ',' << a.steals_ << ','
      << a.cold_steals_ << ',' << a.watermark_evictions_ << ','
      << a.extent_returns_ << " retain=" << a.protected_bytes_ << ','
      << a.protected_slots_ << ',' << a.wheel_entries_ << ',' << a.wheel_tick_
      << '\n';
    for (size_t t = 0; t < a.wheel_.size(); ++t) {
      if (a.wheel_[t].empty()) continue;
      o << "wheel " << t;
      for (const auto& entry : a.wheel_[t]) {
        o << ' ';
        key(entry.key);
        o << '@' << entry.until;
      }
      o << '\n';
    }
    return o.str();
  }
};
}  // namespace dfkv

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

// A read pin by handle (the RAM tier's read path) is a CLOCK reference: the
// hand spares the resident once, then treats it like any other.
TEST(SlabAllocator, PinForReadGivesOneSecondChance) {
  SlabAllocator a(Opts(4 * 4096, 1));
  std::vector<BlockKey> ev;
  SlotRef r;
  SlabAllocator::SlotHandle h0;
  ASSERT_TRUE(a.Put(Key("k0"), 4096, &r, &ev, &h0));
  for (int i = 1; i < 4; ++i)
    ASSERT_TRUE(a.Put(Key("k" + std::to_string(i)), 4096, &r, &ev));
  ASSERT_TRUE(a.PinForRead(Key("k0"), h0));
  ASSERT_TRUE(a.PinForRead(Key("k0"), h0));  // a second read adds no credit
  ASSERT_TRUE(a.Unpin(Key("k0"), h0));
  ASSERT_TRUE(a.Unpin(Key("k0"), h0));

  ASSERT_TRUE(a.Put(Key("n0"), 4096, &r, &ev));
  ASSERT_EQ(ev.size(), 1u);
  EXPECT_TRUE(ev[0] == Key("k1")) << "oldest unread resident goes first";
  EXPECT_TRUE(a.Contains(Key("k0")));

  // No further read: k0 is the victim once the hand comes round again, after
  // the residents that were already ahead of it and before any newer one.
  ASSERT_TRUE(a.Put(Key("n1"), 4096, &r, &ev));
  ASSERT_TRUE(a.Put(Key("n2"), 4096, &r, &ev));
  EXPECT_TRUE(a.Contains(Key("k0")));
  ASSERT_TRUE(a.Put(Key("n3"), 4096, &r, &ev));
  ASSERT_EQ(ev.size(), 4u);
  EXPECT_TRUE(ev[3] == Key("k0"));
  EXPECT_TRUE(a.Contains(Key("n0")));
}

// Pins taken for writes (copy window, flush) are not reads: they must not
// hand every new resident a second chance.
TEST(SlabAllocator, PlainHandlePinIsNotAClockReference) {
  SlabAllocator a(Opts(4 * 4096, 1));
  std::vector<BlockKey> ev;
  SlotRef r;
  SlabAllocator::SlotHandle h0;
  ASSERT_TRUE(a.Put(Key("k0"), 4096, &r, &ev, &h0));
  for (int i = 1; i < 4; ++i)
    ASSERT_TRUE(a.Put(Key("k" + std::to_string(i)), 4096, &r, &ev));
  ASSERT_TRUE(a.Pin(Key("k0"), h0));
  ASSERT_TRUE(a.Unpin(Key("k0"), h0));
  ASSERT_TRUE(a.Put(Key("n0"), 4096, &r, &ev));
  ASSERT_EQ(ev.size(), 1u);
  EXPECT_TRUE(ev[0] == Key("k0"));
}

// Every resident referenced: one admission still evicts exactly one victim
// (the sweep clears the bits in one lap and takes the first resident it
// reaches again), and the cleared residents carry no leftover credit.
TEST(SlabAllocator, FullyReferencedClassStillEvictsOnePerAdmission) {
  SlabAllocator a(Opts(4 * 4096, 1));
  std::vector<BlockKey> ev;
  SlotRef r;
  SlabAllocator::SlotHandle h[4];
  for (int i = 0; i < 4; ++i)
    ASSERT_TRUE(a.Put(Key("k" + std::to_string(i)), 4096, &r, &ev, &h[i]));
  for (int i = 0; i < 4; ++i) {
    const BlockKey key = Key("k" + std::to_string(i));
    ASSERT_TRUE(a.PinForRead(key, h[i]));
    ASSERT_TRUE(a.Unpin(key, h[i]));
  }
  ASSERT_TRUE(a.Put(Key("n0"), 4096, &r, &ev));
  ASSERT_EQ(ev.size(), 1u);
  EXPECT_TRUE(ev[0] == Key("k0"));
  EXPECT_EQ(a.Count(), 4u);
  ASSERT_TRUE(a.Put(Key("n1"), 4096, &r, &ev));
  ASSERT_EQ(ev.size(), 2u);
  EXPECT_TRUE(ev[1] == Key("k1"));
}

namespace {
// 16 one-slot residents k0..k15 of one class, admitted in that order (which
// is also the order the hand visits them), all read once.
void FillAndReadSixteen(SlabAllocator& a) {
  std::vector<BlockKey> ev;
  SlotRef r;
  for (int i = 0; i < 16; ++i) {
    const BlockKey key = Key("k" + std::to_string(i));
    SlabAllocator::SlotHandle h;
    ASSERT_TRUE(a.Put(key, 4096, &r, &ev, &h));
    ASSERT_TRUE(a.PinForRead(key, h));
    ASSERT_TRUE(a.Unpin(key, h));
  }
  ASSERT_TRUE(ev.empty());
}
}  // namespace

// evict_clear_limit: with every resident referenced, one admission clears
// only `limit` bits and evicts the first resident it met. The bits beyond
// stay set: the next admission has to clear another `limit` of them, so its
// victim is again the resident at the hand, not one that was already passed.
TEST(SlabAllocator, EvictClearLimitBoundsTheSweepAndKeepsLaterBits) {
  auto o = Opts(16 * 4096, 1);
  o.evict_clear_limit = 4;
  SlabAllocator a(o);
  FillAndReadSixteen(a);
  std::vector<BlockKey> ev;
  SlotRef r;
  for (int i = 0; i < 4; ++i) {
    ASSERT_TRUE(a.Put(Key("n" + std::to_string(i)), 4096, &r, &ev));
    ASSERT_EQ(ev.size(), static_cast<size_t>(i + 1)) << "one victim each";
    EXPECT_EQ(KeyName(ev.back()), "k" + std::to_string(4 * i));
    EXPECT_EQ(a.Count(), 16u);
  }
  // The lap is complete: k1 was cleared by the first sweep and not read since.
  ASSERT_TRUE(a.Put(Key("n4"), 4096, &r, &ev));
  EXPECT_EQ(KeyName(ev.back()), "k1");
}

// Pinned residents are skipped without using up the bound: a pinned run
// longer than the bound at the hand must not make the eviction fail.
TEST(SlabAllocator, EvictClearLimitDoesNotCountPinnedSkips) {
  auto o = Opts(16 * 4096, 1);
  o.evict_clear_limit = 2;
  SlabAllocator a(o);
  FillAndReadSixteen(a);
  for (int i = 0; i < 6; ++i) ASSERT_TRUE(a.Pin(Key("k" + std::to_string(i))));
  std::vector<BlockKey> ev;
  SlotRef r;
  ASSERT_TRUE(a.Put(Key("n0"), 4096, &r, &ev));
  ASSERT_EQ(ev.size(), 1u);
  EXPECT_EQ(KeyName(ev[0]), "k6");
  for (int i = 0; i < 6; ++i)
    EXPECT_TRUE(a.Contains(Key("k" + std::to_string(i))));
}

// A background reclaim call ends once it has cleared `limit` bits, however
// many victims it was allowed.
TEST(SlabAllocator, ReclaimClassStopsAtEvictClearLimit) {
  auto o = Opts(16 * 4096, 1);
  o.evict_clear_limit = 4;
  SlabAllocator a(o);
  FillAndReadSixteen(a);
  std::vector<BlockKey> ev;
  bool clear_limited = false;
  EXPECT_EQ(a.ReclaimClass(0, 8, 8, &ev, &clear_limited), 1u);
  EXPECT_TRUE(clear_limited) << "target unmet: the caller may continue";
  ASSERT_EQ(ev.size(), 1u);
  EXPECT_EQ(KeyName(ev[0]), "k0");
  // The hand stands at k4, whose bit and those after it are still set.
  EXPECT_EQ(a.ReclaimClass(0, 8, 8, &ev), 1u);
  ASSERT_EQ(ev.size(), 2u);
  EXPECT_EQ(KeyName(ev[1]), "k4");
}

// While the background sweep of a class is behind (its last ReclaimClass call
// ended on evict_clear_limit with the target unmet), an admission without a
// free slot evicts from that class and leaves cold extents of other classes
// alone: one eviction per admission. Once a reclaim call ends for any other
// reason, the cold steal is back.
TEST(SlabAllocator, ClearLimitedClassEvictsItselfInsteadOfStealingCold) {
  auto o = Opts(16 * 4096, 10);
  o.evict_clear_limit = 4;
  SlabAllocator a(o);
  std::vector<BlockKey> ev;
  SlotRef r;
  // Cold class: one resident in each of its 8 extents.
  for (int i = 0; i < 8; ++i)
    ASSERT_TRUE(a.Put(Key("b" + std::to_string(i)), 8192, &r, &ev));
  // Read class: the 2 remaining extents, 32 slots, every resident read.
  size_t read_cls = 0;
  for (int i = 0; i < 32; ++i) {
    const BlockKey key = Key("a" + std::to_string(i));
    SlabAllocator::SlotHandle h;
    ASSERT_TRUE(a.Put(key, 4096, &r, &ev, &h));
    read_cls = r.cls;
    ASSERT_TRUE(a.PinForRead(key, h));
    ASSERT_TRUE(a.Unpin(key, h));
  }
  ASSERT_TRUE(ev.empty());
  ASSERT_EQ(a.PoolExtents(), 0u);

  bool clear_limited = false;
  ASSERT_EQ(a.ReclaimClass(read_cls, 8, 8, &ev, &clear_limited), 1u);
  ASSERT_TRUE(clear_limited);
  ASSERT_TRUE(a.Put(Key("x0"), 4096, &r, &ev));  // takes the one free slot
  ASSERT_EQ(ev.size(), 1u);
  for (int i = 1; i <= 4; ++i) {
    ASSERT_TRUE(a.Put(Key("x" + std::to_string(i)), 4096, &r, &ev));
    ASSERT_EQ(ev.size(), static_cast<size_t>(i + 1)) << "one victim each";
    EXPECT_EQ(KeyName(ev.back()), "a" + std::to_string(4 * i));
  }
  for (int i = 0; i < 8; ++i)
    EXPECT_TRUE(a.Contains(Key("b" + std::to_string(i))));
  EXPECT_EQ(a.ColdSteals(), 0u);

  // A reclaim call that ends for another reason (here: nothing to do) lifts
  // the exemption, and the cold extents were eligible all along.
  EXPECT_EQ(a.ReclaimClass(read_cls, 0, 8, &ev, &clear_limited), 0u);
  EXPECT_FALSE(clear_limited);
  ASSERT_TRUE(a.Put(Key("x5"), 4096, &r, &ev));
  EXPECT_EQ(a.ColdSteals(), 1u);
  EXPECT_EQ(KeyName(ev.back()), "b0");
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

// The cold steal takes the qualifying extent whose residents are least
// recently used, like every other steal. Read heat decays within a few
// hundred operations, and ordering by it and then by fewest useful bytes
// took the emptiest qualifying extent, here one read more recently than all
// the others.
TEST(SlabAllocator, ColdStealTakesTheLeastRecentlyUsedExtent) {
  ::setenv("DFKV_SLAB_COLD_STEAL_WINDOW", "6", 1);
  SlabAllocator a(Opts(4 * 4096, 16));
  ::unsetenv("DFKV_SLAB_COLD_STEAL_WINDOW");
  std::vector<BlockKey> ev;
  SlotRef r;
  std::vector<uint32_t> extent_of(16);
  for (int i = 0; i < 16; ++i) {  // A: 8192-byte slots, 2 per extent
    ASSERT_TRUE(a.Put(Key("stale" + std::to_string(i)), 8192, &r, &ev));
    extent_of[i] = r.extent;
  }
  for (int i = 0; i < 32; ++i)  // B: 4096-byte slots, 4 per extent
    ASSERT_TRUE(a.Put(Key("hot" + std::to_string(i)), 4096, &r, &ev));
  // One A extent loses a resident and has its other one read: it holds the
  // fewest useful bytes, and it is the most recently used of A. (Filling B
  // took some A extents; pick one whose two residents are both still there.)
  auto present = [&](int i) {
    return a.Contains(Key("stale" + std::to_string(i)));
  };
  int lost = -1, partner = -1;
  for (int i = 0; i < 16 && partner < 0; ++i)
    for (int j = i + 1; j < 16 && partner < 0; ++j)
      if (extent_of[i] == extent_of[j] && present(i) && present(j)) {
        lost = i;
        partner = j;
      }
  ASSERT_GE(partner, 0);
  // Every other A extent left with one resident gives it up, so that the
  // extent of `partner` is the only one of A with fewer useful bytes.
  for (int i = 0; i < 16; ++i) {
    if (!present(i)) continue;
    bool alone = true;
    for (int j = 0; j < 16; ++j)
      if (j != i && extent_of[j] == extent_of[i] && present(j)) alone = false;
    if (alone)
      ASSERT_EQ(a.Remove(Key("stale" + std::to_string(i))),
                SlabAllocator::RemoveResult::kRemoved);
  }
  ASSERT_EQ(a.Remove(Key("stale" + std::to_string(lost))),
            SlabAllocator::RemoveResult::kRemoved);
  ASSERT_TRUE(a.Get(Key("stale" + std::to_string(partner)), &r));
  // Long enough for read heat to decay away; B stays the warmest class.
  for (int round = 0; round < 40; ++round)
    for (int i = 0; i < 32; ++i)
      ASSERT_TRUE(a.Get(Key("hot" + std::to_string(i)), &r));
  // B writes on with no free slot: one A extent is stolen.
  const uint64_t cold_steals = a.ColdSteals();
  bool stolen = false;
  for (int i = 32; i < 96 && !stolen; ++i) {
    ev.clear();
    ASSERT_TRUE(a.Put(Key("hot" + std::to_string(i)), 4096, &r, &ev));
    for (const auto& k : ev)
      if (KeyName(k).rfind("stale", 0) == 0) stolen = true;
  }
  ASSERT_TRUE(stolen);
  EXPECT_EQ(a.ColdSteals(), cold_steals + 1);
  EXPECT_TRUE(a.Contains(Key("stale" + std::to_string(partner))))
      << "the most recently read A extent went first (fewest useful bytes)";
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
  // Growth back to the striping floor may still take A's least recently used
  // extents (stale0 is the oldest key), but the busy class must not wipe A.
  size_t stale_alive = 0;
  for (int i = 0; i < 16; ++i) stale_alive += a.Contains(Key("stale" + std::to_string(i)));
  EXPECT_GT(stale_alive, 0u) << "stale cross-class data survives";
  ::unsetenv("DFKV_SLAB_COLD_STEAL_WINDOW");
}

TEST(SlabAllocator, WatermarkUsesGetRecencyAcrossClasses) {
  SlabAllocator a(Opts(4 * 4096, 16));
  std::vector<BlockKey> evicted;
  SlotRef ref;
  ASSERT_TRUE(a.Put(Key("read"), 4096, &ref, &evicted));
  ASSERT_TRUE(a.Put(Key("cold"), 8192, &ref, &evicted));
  ASSERT_TRUE(a.Get(Key("read"), &ref));
  ASSERT_TRUE(a.Put(Key("new"), 4096, &ref, &evicted));
  // EXISTS and duplicate PUT are not actual consumption or new admission.
  EXPECT_TRUE(a.Contains(Key("cold")));
  ASSERT_TRUE(a.Put(Key("cold"), 8192, &ref, &evicted));
  EXPECT_EQ(a.EvictColdToTarget(0, 1, &evicted), 1u);
  ASSERT_EQ(evicted.size(), 1u);
  EXPECT_EQ(evicted[0], Key("cold"));
  EXPECT_TRUE(a.Contains(Key("read")));
  EXPECT_TRUE(a.Contains(Key("new")));
  EXPECT_EQ(a.UsedBytes(), 2u * 4096);
}

TEST(SlabAllocator, WatermarkStopsAtSlotTargetAndVictimBudget) {
  SlabAllocator a(Opts(4 * 4096, 1));
  std::vector<BlockKey> evicted;
  SlotRef ref;
  for (int i = 0; i < 4; ++i)
    ASSERT_TRUE(a.Put(Key("k" + std::to_string(i)), 4096, &ref, &evicted));
  EXPECT_EQ(a.EvictColdToTarget(0, 0, &evicted), 0u);
  EXPECT_EQ(a.EvictColdToTarget(0, 1, &evicted), 1u);
  EXPECT_EQ(a.Count(), 3u);
  EXPECT_EQ(a.EvictColdToTarget(2 * 4096, 4, &evicted), 1u);
  ASSERT_EQ(evicted.size(), 2u);
  EXPECT_EQ(evicted[0], Key("k0"));
  EXPECT_EQ(evicted[1], Key("k1"));
  EXPECT_EQ(a.UsedBytes(), 2u * 4096);
  EXPECT_EQ(a.EvictColdToTarget(2 * 4096, 4, &evicted), 0u);
}

TEST(SlabAllocator, WatermarkSkipsPinnedSlotsWithoutProtectingTheirExtent) {
  SlabAllocator a(Opts(4 * 4096, 1));
  std::vector<BlockKey> evicted;
  SlotRef ref;
  for (int i = 0; i < 4; ++i)
    ASSERT_TRUE(a.Put(Key("k" + std::to_string(i)), 4096, &ref, &evicted));
  ASSERT_TRUE(a.Pin(Key("k0")));
  EXPECT_EQ(a.EvictColdToTarget(0, 4, &evicted), 3u);
  EXPECT_TRUE(a.Contains(Key("k0")));
  EXPECT_EQ(a.Count(), 1u);
  EXPECT_EQ(a.UsedBytes(), 4096u);
  ASSERT_TRUE(a.Unpin(Key("k0")));
  EXPECT_EQ(a.EvictColdToTarget(0, 1, &evicted), 1u);
  EXPECT_EQ(a.Count(), 0u);
}

TEST(SlabAllocator, WatermarkBoundedScanContinuesPastPinnedPrefixNextCall) {
  SlabAllocator a(Opts(8 * 4096, 1));
  std::vector<BlockKey> evicted;
  SlotRef ref;
  for (int i = 0; i < 6; ++i) {
    ASSERT_TRUE(a.Put(Key("k" + std::to_string(i)), 4096, &ref, &evicted));
    if (i < 5) ASSERT_TRUE(a.Pin(Key("k" + std::to_string(i))));
  }
  // A one-victim call visits at most four slots, even when none are eligible.
  EXPECT_EQ(a.EvictColdToTarget(0, 1, &evicted), 0u);
  EXPECT_TRUE(evicted.empty());
  EXPECT_EQ(a.EvictColdToTarget(0, 1, &evicted), 1u);
  ASSERT_EQ(evicted.size(), 1u);
  EXPECT_EQ(evicted[0], Key("k5"));
  EXPECT_EQ(a.Count(), 5u);
}

TEST(SlabAllocator, WatermarkClearsIndividualRecordsUntilLastResident) {
  auto options = Opts(4 * 4096, 1);
  uint32_t extent_calls = 0;
  uint32_t slot_calls = 0;
  bool allow_clear = false;
  options.on_extent_evict = [&](uint32_t, uint32_t total, uint32_t residents) {
    ++extent_calls;
    EXPECT_EQ(total, 4u);
    EXPECT_EQ(residents, 1u);
    return allow_clear;
  };
  options.on_slot_evict = [&](const SlotRef&) {
    ++slot_calls;
    return allow_clear;
  };
  SlabAllocator a(options);
  std::vector<BlockKey> evicted;
  SlotRef ref;
  for (int i = 0; i < 4; ++i)
    ASSERT_TRUE(a.Put(Key("k" + std::to_string(i)), 4096, &ref, &evicted));
  EXPECT_EQ(a.EvictColdToTarget(0, 4, &evicted), 0u);
  EXPECT_TRUE(evicted.empty());
  EXPECT_EQ(a.Count(), 4u);  // failed persistence must not forget a resident
  allow_clear = true;
  EXPECT_EQ(a.EvictColdToTarget(4096, 4, &evicted), 3u);
  EXPECT_EQ(slot_calls, 4u);
  EXPECT_EQ(extent_calls, 0u);
  allow_clear = false;
  EXPECT_EQ(a.EvictColdToTarget(0, 1, &evicted), 0u);
  EXPECT_TRUE(a.Contains(Key("k3")));
  allow_clear = true;
  EXPECT_EQ(a.EvictColdToTarget(0, 1, &evicted), 1u);
  EXPECT_EQ(extent_calls, 2u);
  EXPECT_EQ(a.Count(), 0u);
}

TEST(SlabAllocator, WatermarkCannotWipeLivePeersWithOnlyExtentCallback) {
  auto options = Opts(2 * 4096, 1);
  uint32_t extent_calls = 0;
  options.on_extent_evict = [&](uint32_t, uint32_t, uint32_t) {
    ++extent_calls;
    return true;
  };
  SlabAllocator a(options);
  SlotRef ref;
  std::vector<BlockKey> evicted;
  ASSERT_TRUE(a.Put(Key("first"), 4096, &ref, &evicted));
  ASSERT_TRUE(a.Put(Key("peer"), 4096, &ref, &evicted));
  EXPECT_EQ(a.EvictColdToTarget(4096, 1, &evicted), 0u);
  EXPECT_EQ(extent_calls, 0u);
  EXPECT_TRUE(evicted.empty());
  EXPECT_TRUE(a.Contains(Key("first")));
  EXPECT_TRUE(a.Contains(Key("peer")));
  EXPECT_EQ(a.UsedBytes(), 2u * 4096);
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

// A steal must not pick the extent that new writes are still filling just
// because it holds fewer bytes: that wipes the newest residents while older
// full extents survive.
TEST(SlabAllocator, StealFromSparesTheExtentBeingFilled) {
  SlabAllocator a(Opts(4 * 4096, 2));  // 2 extents x 4 slots
  std::vector<SlabAllocator::RestoreEntry> records;
  for (uint32_t s = 0; s < 4; ++s)
    records.push_back({Key("old" + std::to_string(s)), 4096, 0, s, 4096});
  records.push_back({Key("old4"), 4096, 1, 0, 4096});
  ASSERT_TRUE(a.RestoreBulk(records));
  std::vector<BlockKey> ev;
  SlotRef r;
  ASSERT_TRUE(a.Put(Key("new0"), 4096, &r, &ev));
  ASSERT_TRUE(a.Put(Key("new1"), 4096, &r, &ev));
  ASSERT_TRUE(ev.empty());
  ASSERT_EQ(r.extent, 1u);

  ASSERT_TRUE(a.StealFrom(ClassIndex(a, 4096), ClassIndex(a, 16 * 1024), &ev));
  EXPECT_EQ(ev.size(), 4u);
  EXPECT_TRUE(a.Contains(Key("new0")));
  EXPECT_TRUE(a.Contains(Key("new1")));
  EXPECT_TRUE(a.Contains(Key("old4")));
  EXPECT_FALSE(a.Contains(Key("old0")));
}

// The victim is the extent whose residents are least recently used on
// average, not the one touched least recently: one fresh read must not shield
// an extent of otherwise old residents.
TEST(SlabAllocator, StealFromUsesMeanResidentRecency) {
  SlabAllocator a(Opts(4 * 4096, 2));
  std::vector<SlabAllocator::RestoreEntry> records;
  for (uint32_t s = 0; s < 4; ++s) {
    records.push_back({Key("old" + std::to_string(s)), 4096, 0, s, 4096});
    records.push_back({Key("mid" + std::to_string(s)), 4096, 1, s, 4096});
  }
  ASSERT_TRUE(a.RestoreBulk(records));
  SlotRef r;
  for (int s = 0; s < 4; ++s) ASSERT_TRUE(a.Get(Key("mid" + std::to_string(s)), &r));
  ASSERT_TRUE(a.Get(Key("old0"), &r));  // newest single access, extent 0

  std::vector<BlockKey> ev;
  ASSERT_TRUE(a.StealFrom(ClassIndex(a, 4096), ClassIndex(a, 16 * 1024), &ev));
  EXPECT_EQ(ev.size(), 4u);
  for (int s = 0; s < 4; ++s) {
    EXPECT_FALSE(a.Contains(Key("old" + std::to_string(s))));
    EXPECT_TRUE(a.Contains(Key("mid" + std::to_string(s))));
  }
}

// GROW donors are judged by bytes written, not object counts: a large-slot
// class with fewer but bigger writes is not cold.
TEST(SlabAllocator, GrowDonorComparesDemandInBytes) {
  std::vector<SlabAllocator::ClassStat> classes(3);
  classes[0].slot_size = 1u << 20;   // hot by count
  classes[1].slot_size = 64u << 20;  // fewer, much larger writes
  classes[2].slot_size = 4u << 20;   // idle
  const std::vector<uint32_t> extents{100, 100, 100};
  // 0 wants to grow: 13 MiB/pass vs 64 MiB/pass for class 1 -> only 2 is cold.
  EXPECT_EQ(SlabAllocator::ColdestGrowDonor(classes, {13, 1, 0}, extents, 0, 8),
            2u);
  // Without an idle class there is no donor for 0.
  EXPECT_EQ(
      SlabAllocator::ColdestGrowDonor(classes, {13, 1, 1000}, extents, 0, 8),
      classes.size());
  // Class 1 growing may take from the byte-colder class 0.
  EXPECT_EQ(
      SlabAllocator::ColdestGrowDonor(classes, {13, 1, 1000}, extents, 1, 8),
      0u);
  // A donor that is being written keeps its striping floor.
  EXPECT_EQ(
      SlabAllocator::ColdestGrowDonor(classes, {13, 1, 1000}, {8, 8, 100}, 2,
                                      8),
      classes.size());
  // An idle one has no floor: it gives down to its last extent.
  EXPECT_EQ(SlabAllocator::ColdestGrowDonor(classes, {13, 1, 0}, {100, 100, 1}, 0,
                                            8),
            2u);
  EXPECT_EQ(SlabAllocator::ColdestGrowDonor(classes, {13, 1, 0}, {100, 100, 0}, 0,
                                            8),
            classes.size());
}

// The free slots a reclaim pass keeps ahead of a class's writers are bounded
// in bytes: two passes of admissions, at least 64 MiB and at most 1 GiB of
// slots (and at most a quarter of the class). A fixed floor of 64 slots was
// 2 GiB of 32 MiB slots evicted ahead of need, and twice a pass of small
// admissions could hold a quarter of a large class free.
TEST(SlabAllocator, FreeSlotTargetIsBoundedInBytes) {
  const size_t big = std::numeric_limits<size_t>::max();
  constexpr uint32_t kSmall = 4096, kLarge = 32u << 20;
  // Small slots: the floor stays 64 slots; the cap is 1 GiB of them.
  EXPECT_EQ(SlabAllocator::FreeSlotTarget(10, kSmall, big), 64u);
  EXPECT_EQ(SlabAllocator::FreeSlotTarget(1000, kSmall, big), 2000u);
  EXPECT_EQ(SlabAllocator::FreeSlotTarget(1000000, kSmall, big),
            (1u << 30) / kSmall);
  // Large slots: a floor of 64 MiB (2 slots), a cap of 1 GiB (32 slots).
  EXPECT_EQ(SlabAllocator::FreeSlotTarget(0, kLarge, big), 2u);
  EXPECT_EQ(SlabAllocator::FreeSlotTarget(1, kLarge, big), 2u);
  EXPECT_EQ(SlabAllocator::FreeSlotTarget(10, kLarge, big), 20u);
  EXPECT_EQ(SlabAllocator::FreeSlotTarget(100, kLarge, big), 32u);
  // Never more than a quarter of the class.
  EXPECT_EQ(SlabAllocator::FreeSlotTarget(10, kSmall, 100), 25u);
  EXPECT_EQ(SlabAllocator::FreeSlotTarget(100, kLarge, 40), 10u);
  EXPECT_EQ(SlabAllocator::FreeSlotTarget(10, 0, big), 0u);
}

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
// A class without a free slot takes an extent another class holds empty
// before it evicts anything, even when the other class is down to its
// striping floor, if that class has written nothing for a cold window:
// moving an empty extent costs no value.
TEST(SlabAllocator, PutTakesAnEmptyExtentOfAnotherClassBeforeEvicting) {
  ::setenv("DFKV_SLAB_COLD_STEAL_WINDOW", "4", 1);
  SlabAllocator a(Opts(64 * 1024, 16));  // 16 extents
  ::unsetenv("DFKV_SLAB_COLD_STEAL_WINDOW");
  std::vector<uint32_t> one_slot;        // slot sizes with one slot per extent
  for (const auto& c : a.Classes())
    if (c.slots_per_extent == 1) one_slot.push_back(c.slot_size);
  ASSERT_GE(one_slot.size(), 3u);
  const uint32_t sa = one_slot[0], sb = one_slot[1], sc = one_slot[2];
  std::vector<BlockKey> ev;
  SlotRef r;
  // A binds its striping width of empty extents beyond what it fills.
  for (int i = 0; i < 6; ++i)
    ASSERT_TRUE(a.Put(Key("a" + std::to_string(i)), sa, &r, &ev));
  for (int i = 0; i < 6; ++i)
    ASSERT_TRUE(a.Put(Key("b" + std::to_string(i)), sb, &r, &ev));
  // A holds 10 extents now, 4 of them empty, and 15 values need 15 of the
  // 16 extents. Taking two of A's empty extents brings A to its floor of 8
  // (2 empty); the third C value used to evict one of C's own.
  ASSERT_EQ(a.Classes()[ClassIndex(a, sa)].extents, 10u);
  for (int i = 0; i < 3; ++i)
    ASSERT_TRUE(a.Put(Key("c" + std::to_string(i)), sc, &r, &ev));
  EXPECT_TRUE(ev.empty()) << "a value was evicted with an empty extent left";
  EXPECT_EQ(a.Evictions(), 0u);
  for (int i = 0; i < 3; ++i)
    EXPECT_TRUE(a.Contains(Key("c" + std::to_string(i)))) << i;
}

// ...but a class that is being written keeps its striping width of extents
// with free slots, empty or not: the writer evicts from its own class.
TEST(SlabAllocator, AClassBeingWrittenKeepsItsEmptyStripes) {
  SlabAllocator a(Opts(64 * 1024, 16));  // 16 extents
  std::vector<uint32_t> one_slot;
  for (const auto& c : a.Classes())
    if (c.slots_per_extent == 1) one_slot.push_back(c.slot_size);
  ASSERT_GE(one_slot.size(), 2u);
  const uint32_t sa = one_slot[0], sb = one_slot[1];
  std::vector<BlockKey> ev;
  SlotRef r;
  // A: 2 values, and 7 empty extents of striping headroom.
  for (int i = 0; i < 2; ++i)
    ASSERT_TRUE(a.Put(Key("a" + std::to_string(i)), sa, &r, &ev));
  ASSERT_EQ(a.Classes()[ClassIndex(a, sa)].extents, 9u);
  // B takes the other 7, and one of A's through its floor steal (A has more
  // than 8).
  for (int i = 0; i < 8; ++i)
    ASSERT_TRUE(a.Put(Key("b" + std::to_string(i)), sb, &r, &ev));
  ASSERT_EQ(a.Classes()[ClassIndex(a, sa)].extents, 8u);
  ASSERT_TRUE(ev.empty());
  // A writes on. B, at its floor, evicts its own value: A keeps 8.
  ASSERT_TRUE(a.Put(Key("a2"), sa, &r, &ev));
  ASSERT_TRUE(a.Put(Key("b8"), sb, &r, &ev));
  EXPECT_EQ(a.Classes()[ClassIndex(a, sa)].extents, 8u);
  ASSERT_EQ(ev.size(), 1u);
  EXPECT_EQ(KeyName(ev[0]).rfind("b", 0), 0u);
}

// With Options::stripe_ways 1 (the RAM tier) a class binds the extents it
// fills and one more at most, not eight empty ones besides.
TEST(SlabAllocator, OneStripeWayBindsOnlyWhatTheClassFills) {
  SlabAllocator::Options o = Opts(64 * 1024, 16);
  o.stripe_ways = 1;
  SlabAllocator a(o);
  ASSERT_EQ(a.StripeWays(), 1u);
  uint32_t sa = 0;
  for (const auto& c : a.Classes())
    if (c.slots_per_extent == 1) { sa = c.slot_size; break; }
  ASSERT_NE(sa, 0u);
  std::vector<BlockKey> ev;
  SlotRef r;
  for (int i = 0; i < 6; ++i)
    ASSERT_TRUE(a.Put(Key("a" + std::to_string(i)), sa, &r, &ev));
  EXPECT_LE(a.Classes()[ClassIndex(a, sa)].extents, 7u);
  EXPECT_GE(a.PoolExtents(), 9u);
  std::printf("one way: %u extents for 6 values\n",
              a.Classes()[ClassIndex(a, sa)].extents);
  // The default keeps eight extents with free slots.
  SlabAllocator d(Opts(64 * 1024, 16));
  for (int i = 0; i < 6; ++i)
    ASSERT_TRUE(d.Put(Key("a" + std::to_string(i)), sa, &r, &ev));
  EXPECT_GE(d.Classes()[ClassIndex(d, sa)].extents, 8u);
  std::printf("eight ways: %u extents for 6 values\n",
              d.Classes()[ClassIndex(d, sa)].extents);
}

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

// ------------------------------------------------------------------ retention
namespace {
using RetainCode = SlabAllocator::RetainCode;
using RetainMode = SlabAllocator::RetainMode;

SlabAllocator::RetainOptions RetainOpts(uint64_t budget_bytes,
                                        uint32_t max_ttl_s = 3600,
                                        uint32_t class_pct = 100) {
  SlabAllocator::RetainOptions o;
  o.enabled = true;
  o.budget_bytes = budget_bytes;
  o.max_ttl_s = max_ttl_s;
  o.class_pct = class_pct;
  o.persist_min_delta_s = 60;
  return o;
}
constexpr uint32_t kT0 = 1'000'000;  // arbitrary realtime origin
}  // namespace

TEST(SlabAllocatorRetention, ProtectedSlotSurvivesClassEviction) {
  SlabAllocator a(Opts(2 * 4096, 1));  // 2 slots
  a.ConfigureRetention(RetainOpts(1 << 20), kT0);
  std::vector<BlockKey> ev;
  SlotRef r;
  ASSERT_TRUE(a.Put(Key("keep"), 4096, &r, &ev));
  ASSERT_TRUE(a.Put(Key("other"), 4096, &r, &ev));
  EXPECT_EQ(a.Retain(Key("keep"), 300, RetainMode::kSet, kT0).code,
            RetainCode::kProtectedNew);
  for (int i = 0; i < 8; ++i) {
    ev.clear();
    ASSERT_TRUE(a.Put(Key("churn" + std::to_string(i)), 4096, &r, &ev));
    for (const BlockKey& k : ev) EXPECT_FALSE(k == Key("keep"));
  }
  EXPECT_TRUE(a.Contains(Key("keep")));
  // With every unpinned slot protected, a Put has nothing to evict.
  SlabAllocator b(Opts(2 * 4096, 1));
  b.ConfigureRetention(RetainOpts(1 << 20), kT0);
  ASSERT_TRUE(b.Put(Key("p1"), 4096, &r, &ev));
  ASSERT_TRUE(b.Put(Key("p2"), 4096, &r, &ev));
  EXPECT_EQ(b.Retain(Key("p1"), 60, RetainMode::kSet, kT0).code,
            RetainCode::kProtectedNew);
  EXPECT_EQ(b.Retain(Key("p2"), 60, RetainMode::kSet, kT0).code,
            RetainCode::kProtectedNew);
  ev.clear();
  EXPECT_FALSE(b.Put(Key("p3"), 4096, &r, &ev));
  EXPECT_TRUE(ev.empty());
}

TEST(SlabAllocatorRetention, ExpiryReturnsSlotToOrdinaryEviction) {
  SlabAllocator a(Opts(2 * 4096, 1));
  a.ConfigureRetention(RetainOpts(1 << 20), kT0);
  std::vector<BlockKey> ev;
  SlotRef r;
  ASSERT_TRUE(a.Put(Key("p1"), 4096, &r, &ev));
  ASSERT_TRUE(a.Put(Key("p2"), 4096, &r, &ev));
  ASSERT_EQ(a.Retain(Key("p1"), 60, RetainMode::kSet, kT0).code,
            RetainCode::kProtectedNew);
  ASSERT_EQ(a.Retain(Key("p2"), 60, RetainMode::kSet, kT0).code,
            RetainCode::kProtectedNew);
  // Before the deadline (and within the 10 s bucket slack) nothing expires.
  EXPECT_EQ(a.ExpireRetention(kT0 + 59), 0u);
  EXPECT_EQ(a.ExpireRetention(kT0 + 65), 0u);
  EXPECT_FALSE(a.Put(Key("p3"), 4096, &r, &ev));
  // Strictly after the deadline's bucket both are released.
  EXPECT_EQ(a.ExpireRetention(kT0 + 80), 2u);
  EXPECT_EQ(a.GetRetentionStats().protected_slots, 0u);
  EXPECT_EQ(a.GetRetentionStats().protected_bytes, 0u);
  ev.clear();
  EXPECT_TRUE(a.Put(Key("p3"), 4096, &r, &ev));
  EXPECT_EQ(ev.size(), 1u);
}

TEST(SlabAllocatorRetention, RefreshRenewsWithStoredTtlAndNeverShortens) {
  SlabAllocator a(Opts(4 * 4096, 1));
  a.ConfigureRetention(RetainOpts(1 << 20), kT0);
  std::vector<BlockKey> ev;
  SlotRef r;
  ASSERT_TRUE(a.Put(Key("k"), 4096, &r, &ev));
  auto set = a.Retain(Key("k"), 300, RetainMode::kSet, kT0);
  ASSERT_EQ(set.code, RetainCode::kProtectedNew);
  EXPECT_EQ(set.until, kT0 + 300);
  EXPECT_TRUE(set.persist);
  // REFRESH ignores the request TTL and renews with the stored one.
  auto ref = a.Retain(Key("k"), 3600, RetainMode::kRefresh, kT0 + 240);
  EXPECT_EQ(ref.code, RetainCode::kRefreshed);
  EXPECT_EQ(ref.until, kT0 + 540);
  EXPECT_EQ(ref.ttl_s, 300u);
  // A shorter SET never lowers the stored TTL or the deadline.
  auto shorter = a.Retain(Key("k"), 60, RetainMode::kSet, kT0 + 240);
  EXPECT_EQ(shorter.code, RetainCode::kUnchanged);
  EXPECT_EQ(shorter.until, kT0 + 540);
  // A longer SET upgrades and is always persisted.
  auto up = a.Retain(Key("k"), 3600, RetainMode::kSet, kT0 + 242);
  EXPECT_EQ(up.code, RetainCode::kUpgraded);
  EXPECT_EQ(up.until, kT0 + 242 + 3600);
  EXPECT_TRUE(up.persist);
  // The superseded wheel entries do not release the later deadline.
  EXPECT_EQ(a.ExpireRetention(kT0 + 600), 0u);
  EXPECT_EQ(a.GetRetentionStats().protected_slots, 1u);
  EXPECT_EQ(a.ExpireRetention(kT0 + 242 + 3600 + 20), 1u);
  // After expiry a REFRESH re-protects with the stored TTL.
  auto again = a.Retain(Key("k"), 0, RetainMode::kRefresh, kT0 + 5000);
  EXPECT_EQ(again.code, RetainCode::kProtectedNew);
  EXPECT_EQ(again.until, kT0 + 5000 + 3600);
}

TEST(SlabAllocatorRetention, TtlClampedToCapAndMissOnAbsentKey) {
  SlabAllocator a(Opts(4 * 4096, 1));
  a.ConfigureRetention(RetainOpts(1 << 20, /*max_ttl_s=*/600), kT0);
  std::vector<BlockKey> ev;
  SlotRef r;
  ASSERT_TRUE(a.Put(Key("k"), 4096, &r, &ev));
  auto out = a.Retain(Key("k"), 3600, RetainMode::kSet, kT0);
  EXPECT_EQ(out.ttl_s, 600u);
  EXPECT_EQ(out.until, kT0 + 600);
  EXPECT_EQ(a.Retain(Key("absent"), 60, RetainMode::kSet, kT0).code,
            RetainCode::kMiss);
  // REFRESH of a never-protected key is a plain hit.
  ASSERT_TRUE(a.Put(Key("plain"), 4096, &r, &ev));
  EXPECT_EQ(a.Retain(Key("plain"), 60, RetainMode::kRefresh, kT0).code,
            RetainCode::kHitUnprotected);
}

TEST(SlabAllocatorRetention, BudgetAndClassCapDeny) {
  SlabAllocator a(Opts(4 * 4096, 1));
  // Budget for exactly one 4 KiB slot.
  a.ConfigureRetention(RetainOpts(4096), kT0);
  std::vector<BlockKey> ev;
  SlotRef r;
  ASSERT_TRUE(a.Put(Key("a"), 4096, &r, &ev));
  ASSERT_TRUE(a.Put(Key("b"), 4096, &r, &ev));
  EXPECT_EQ(a.Retain(Key("a"), 60, RetainMode::kSet, kT0).code,
            RetainCode::kProtectedNew);
  EXPECT_EQ(a.Retain(Key("b"), 60, RetainMode::kSet, kT0).code,
            RetainCode::kDenied);
  // Renewing already-protected bytes is always allowed.
  EXPECT_EQ(a.Retain(Key("a"), 0, RetainMode::kRefresh, kT0 + 30).code,
            RetainCode::kRefreshed);
  EXPECT_EQ(a.RetentionHeadroomPct(), 0u);

  SlabAllocator c(Opts(4 * 4096, 1));
  // Ample bytes, but at most 50% of the class's 4 slots.
  c.ConfigureRetention(RetainOpts(1 << 20, 3600, /*class_pct=*/50), kT0);
  for (const char* k : {"x", "y", "z"}) ASSERT_TRUE(c.Put(Key(k), 4096, &r, &ev));
  EXPECT_EQ(c.Retain(Key("x"), 60, RetainMode::kSet, kT0).code,
            RetainCode::kProtectedNew);
  EXPECT_EQ(c.Retain(Key("y"), 60, RetainMode::kSet, kT0).code,
            RetainCode::kProtectedNew);
  EXPECT_EQ(c.Retain(Key("z"), 60, RetainMode::kSet, kT0).code,
            RetainCode::kDenied);
}

TEST(SlabAllocatorRetention, RemoveIgnoresProtectionAndReleasesBudget) {
  SlabAllocator a(Opts(4 * 4096, 1));
  a.ConfigureRetention(RetainOpts(4096), kT0);
  std::vector<BlockKey> ev;
  SlotRef r;
  ASSERT_TRUE(a.Put(Key("a"), 4096, &r, &ev));
  ASSERT_TRUE(a.Put(Key("b"), 4096, &r, &ev));
  ASSERT_EQ(a.Retain(Key("a"), 60, RetainMode::kSet, kT0).code,
            RetainCode::kProtectedNew);
  EXPECT_EQ(a.Remove(Key("a")), SlabAllocator::RemoveResult::kRemoved);
  EXPECT_EQ(a.GetRetentionStats().protected_bytes, 0u);
  EXPECT_EQ(a.Retain(Key("b"), 60, RetainMode::kSet, kT0).code,
            RetainCode::kProtectedNew);
  // The stale wheel entry of the removed key is harmless.
  EXPECT_EQ(a.ExpireRetention(kT0 + 100), 1u);
}

TEST(SlabAllocatorRetention, WatermarkAndStealsSkipProtected) {
  SlabAllocator a(Opts(2 * 4096, 2));  // 2 extents x 2 slots
  a.ConfigureRetention(RetainOpts(1 << 20), kT0);
  std::vector<BlockKey> ev;
  SlotRef r;
  for (const char* k : {"a", "b", "c", "d"}) ASSERT_TRUE(a.Put(Key(k), 4096, &r, &ev));
  ASSERT_EQ(a.Retain(Key("a"), 300, RetainMode::kSet, kT0).code,
            RetainCode::kProtectedNew);
  ev.clear();
  a.EvictColdToTarget(0, 16, &ev);
  for (const BlockKey& k : ev) EXPECT_FALSE(k == Key("a"));
  EXPECT_TRUE(a.Contains(Key("a")));
  EXPECT_GE(a.GetRetentionStats().evict_skipped, 1u);
  EXPECT_EQ(a.GetRetentionStats().extents_pinned, 1u);
  // A different size class must steal around the protected extent.
  ev.clear();
  ASSERT_TRUE(a.Put(Key("big"), 8192, &r, &ev));
  EXPECT_TRUE(a.Contains(Key("a")));
}

TEST(SlabAllocatorRetention, RestoreReappliesAndReclampsProtection) {
  SlabAllocator a(Opts(2 * 4096, 1));
  a.ConfigureRetention(RetainOpts(1 << 20, /*max_ttl_s=*/600), kT0);
  std::vector<SlabAllocator::RestoreEntry> entries;
  SlabAllocator::RestoreEntry live{Key("live"), 4096, 0, 0, 4096, kT0 + 3000, 3600};
  SlabAllocator::RestoreEntry gone{Key("gone"), 4096, 0, 1, 4096, kT0 - 1, 300};
  entries.push_back(live);
  entries.push_back(gone);
  ASSERT_TRUE(a.RestoreBulk(entries));
  const auto st = a.GetRetentionStats();
  EXPECT_EQ(st.protected_slots, 1u);
  // Re-clamped to now + new cap; the stored TTL also drops to the cap.
  auto out = a.Retain(Key("live"), 0, RetainMode::kRefresh, kT0);
  EXPECT_EQ(out.ttl_s, 600u);
  EXPECT_EQ(out.until, kT0 + 600);
  // The expired record restores only its TTL, ready for a REFRESH.
  EXPECT_EQ(a.Retain(Key("gone"), 0, RetainMode::kRefresh, kT0).code,
            RetainCode::kProtectedNew);
}

TEST(SlabAllocatorRetention, DisabledRetentionIsInert) {
  SlabAllocator a(Opts(2 * 4096, 1));
  std::vector<BlockKey> ev;
  SlotRef r;
  ASSERT_TRUE(a.Put(Key("k"), 4096, &r, &ev));
  EXPECT_EQ(a.Retain(Key("k"), 300, RetainMode::kSet, kT0).code,
            RetainCode::kHitUnprotected);
  EXPECT_EQ(a.ExpireRetention(kT0 + 1000), 0u);
  EXPECT_EQ(a.RetentionHeadroomPct(), 0u);
}

// Not a correctness test: prints how long RestoreBulk takes for a store of
// production shape (1 GiB extents of 1 MiB slots, four in five slots
// resident, records in table order). Set DFKV_SLAB_RESTORE_BENCH_OBJECTS (for
// example 2300000) to run it; it needs no disk.
TEST(SlabAllocator, RestoreBulkBench) {
  const char* requested = std::getenv("DFKV_SLAB_RESTORE_BENCH_OBJECTS");
  if (requested == nullptr || *requested == '\0')
    GTEST_SKIP() << "set DFKV_SLAB_RESTORE_BENCH_OBJECTS to run";
  const uint64_t objects = std::strtoull(requested, nullptr, 10);
  ASSERT_GT(objects, 0u);
  constexpr uint32_t kSlot = 1u << 20;
  constexpr uint32_t kSlotsPerExtent = 1024;
  const uint32_t extents =
      static_cast<uint32_t>(objects * 5 / 4 / kSlotsPerExtent + 1);
  std::mt19937_64 rng(25);
  std::vector<SlabAllocator::RestoreEntry> entries;
  entries.reserve(objects);
  for (uint32_t e = 0; e < extents && entries.size() < objects; ++e)
    for (uint32_t s = 0; s < kSlotsPerExtent && entries.size() < objects; ++s)
      if (rng() % 5 != 0)
        entries.push_back({BlockKey{rng(), rng(), rng() % 7}, kSlot, e, s,
                           kSlot - 100});
  for (int round = 0; round < 3; ++round) {
    SlabAllocator allocator(
        Opts(uint64_t{kSlot} * kSlotsPerExtent, extents, kSlot));
    const auto started = std::chrono::steady_clock::now();
    ASSERT_TRUE(allocator.RestoreBulk(entries));
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - started)
                          .count();
    ASSERT_EQ(allocator.Count(), entries.size());
    std::printf("RESTORE_BENCH round=%d objects=%zu extents=%u restore_ms=%.1f\n",
                round, entries.size(), extents, ms);
  }
}

// Not a correctness test: prints the time one Put holds the allocator (the
// RAM tier holds its shard lock for exactly that long) on a full allocator of
// RAM tier shape: one stripe way, 32 MiB extents, two size classes written
// three to one, every admission evicting. Set DFKV_SLAB_PUT_BENCH_EXTENTS (for
// example 1024) to run it.
TEST(SlabAllocator, FullPutBench) {
  const char* requested = std::getenv("DFKV_SLAB_PUT_BENCH_EXTENTS");
  if (requested == nullptr || *requested == '\0')
    GTEST_SKIP() << "set DFKV_SLAB_PUT_BENCH_EXTENTS to run";
  const uint32_t extents =
      static_cast<uint32_t>(std::strtoull(requested, nullptr, 10));
  ASSERT_GT(extents, 0u);
  SlabAllocator::Options o = Opts(32ull << 20, extents);
  o.stripe_ways = 1;
  SlabAllocator a(o);
  constexpr size_t kBig = 2873360, kSmall = 1u << 20;
  std::vector<BlockKey> ev;
  SlotRef r;
  uint64_t next = 0;
  auto put = [&] {
    ev.clear();
    const size_t len = next % 4 == 3 ? kSmall : kBig;
    return a.Put(BlockKey{next++, 0x5bd1e995, 1}, len, &r, &ev);
  };
  while (a.Evictions() == 0) ASSERT_TRUE(put());
  for (int i = 0; i < 100000; ++i) ASSERT_TRUE(put());  // settle
  constexpr int kMeasured = 400000;
  std::vector<uint64_t> ns(kMeasured);
  for (int i = 0; i < kMeasured; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = put();
    ns[i] = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - t0)
                                      .count());
    ASSERT_TRUE(ok);
  }
  std::sort(ns.begin(), ns.end());
  uint64_t sum = 0;
  for (const uint64_t v : ns) sum += v;
  std::printf("PUT_BENCH extents=%u residents=%zu p50_ns=%llu p99_ns=%llu "
              "p999_ns=%llu mean_ns=%llu steals=%llu\n",
              extents, a.Count(),
              static_cast<unsigned long long>(ns[kMeasured / 2]),
              static_cast<unsigned long long>(ns[kMeasured * 99 / 100]),
              static_cast<unsigned long long>(ns[kMeasured * 999 / 1000]),
              static_cast<unsigned long long>(sum / kMeasured),
              static_cast<unsigned long long>(a.Steals()));
}

// ---- Bulk restore leaves exactly the state it always did ------------------
//
// RestoreBulk was rewritten for speed. Eviction order, the order free slots
// are handed out in and the index layout after a restart all follow from the
// state it leaves, so that state is compared, field by field, with the one
// the previous implementation (kept in SlabAllocatorTestPeer) produces from
// the same records; then both allocators are driven through the same
// operations and compared again.
namespace {
using dfkv::SlabAllocatorTestPeer;
using RestoreEntry = SlabAllocator::RestoreEntry;

constexpr uint64_t kEqExtentBytes = 64 << 10;
constexpr uint32_t kEqExtents = 24;
constexpr uint32_t kEqNow = 1'000'000;

std::string FirstDifference(const std::string& a, const std::string& b) {
  std::istringstream left(a), right(b);
  std::string x, y;
  for (size_t line = 1;; ++line) {
    const bool has_x = static_cast<bool>(std::getline(left, x));
    const bool has_y = static_cast<bool>(std::getline(right, y));
    if (!has_x && !has_y) return "identical";
    if (!has_x || !has_y || x != y)
      return "line " + std::to_string(line) + "\n  restore:   " +
             (has_x ? x : "<end>") + "\n  reference: " + (has_y ? y : "<end>");
  }
}

// The slot sizes this geometry really has (requests are rounded up to them).
std::vector<uint32_t> EqSlotSizes() {
  SlabAllocator probe(Opts(kEqExtentBytes, 8));
  std::vector<uint32_t> sizes;
  for (size_t len : {4096u, 8192u, 16384u, 30000u}) {
    SlotRef ref;
    std::vector<BlockKey> evicted;
    if (probe.Put(BlockKey{len, 1, 0}, len, &ref, &evicted))
      sizes.push_back(ref.slot_size);
  }
  std::sort(sizes.begin(), sizes.end());
  sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
  return sizes;
}

// What both allocators went through before the restore: residents, reads,
// removals, so extents are bound and their free lists are not in slot order.
void EqPriorOps(SlabAllocator& a, uint64_t seed, int puts) {
  std::mt19937_64 rng(seed);
  const auto sizes = EqSlotSizes();
  std::vector<BlockKey> keys;
  for (int i = 0; i < puts; ++i) {
    const BlockKey key{rng(), rng(), rng() % 3};
    SlotRef ref;
    std::vector<BlockKey> evicted;
    if (a.Put(key, sizes[rng() % sizes.size()] - rng() % 100, &ref, &evicted))
      keys.push_back(key);
  }
  for (size_t i = 0; i < keys.size(); ++i) {
    SlotRef ref;
    if (rng() % 4 == 0) a.Get(keys[i], &ref);
    if (rng() % 3 == 0) a.Remove(keys[i]);
  }
}

// Records for free slots of bound extents and for extents not bound yet.
std::vector<RestoreEntry> EqEntries(const SlabAllocator& a, uint64_t seed,
                                    bool retention, bool shuffled) {
  std::mt19937_64 rng(seed);
  const auto sizes = EqSlotSizes();
  std::vector<RestoreEntry> entries;
  auto add = [&](uint32_t extent, uint32_t slot, uint32_t slot_size) {
    RestoreEntry entry;
    entry.key = BlockKey{rng(), rng(), rng() % 3};
    entry.slot_size = slot_size;
    entry.extent = extent;
    entry.slot = slot;
    entry.value_bytes = slot_size - rng() % 200;
    if (retention && rng() % 3 == 0) {
      // Past, near and beyond-the-cap deadlines, and a zero TTL.
      entry.retain_until = kEqNow - 50 + static_cast<uint32_t>(rng() % 5000);
      entry.retain_ttl_s = static_cast<uint32_t>(rng() % 4000);
    }
    entries.push_back(entry);
  };
  for (const auto& free_slot : SlabAllocatorTestPeer::FreeSlots(a))
    if (rng() % 2 == 0) add(free_slot[0], free_slot[1], free_slot[2]);
  for (uint32_t extent : SlabAllocatorTestPeer::UnboundExtents(a)) {
    if (rng() % 5 == 0) continue;  // stays unbound
    const uint32_t slot_size = sizes[rng() % sizes.size()];
    const uint32_t slots = static_cast<uint32_t>(kEqExtentBytes / slot_size);
    // Some extents filled completely, some sparsely.
    const uint64_t keep_one_in = 1 + rng() % 4;
    for (uint32_t slot = 0; slot < slots; ++slot)
      if (rng() % keep_one_in == 0) add(extent, slot, slot_size);
  }
  if (shuffled) std::shuffle(entries.begin(), entries.end(), rng);
  return entries;
}

// The same operations on both allocators afterwards; returns what they
// answered, so that the answers are compared as well as the final state.
std::string EqLaterOps(SlabAllocator& a, const std::vector<RestoreEntry>& entries,
                       uint64_t seed) {
  std::mt19937_64 rng(seed);
  const auto sizes = EqSlotSizes();
  std::ostringstream out;
  for (size_t i = 0; i < entries.size(); i += 3) {
    SlotRef ref;
    out << a.Get(entries[i].key, &ref) << ref.extent << '/' << ref.slot << ' ';
  }
  for (size_t i = 1; i < entries.size(); i += 7)
    out << static_cast<int>(a.Remove(entries[i].key)) << ' ';
  for (int i = 0; i < 600; ++i) {
    SlotRef ref;
    std::vector<BlockKey> evicted;
    const bool ok = a.Put(BlockKey{rng(), rng(), rng() % 3},
                          sizes[rng() % sizes.size()] - rng() % 100, &ref,
                          &evicted);
    out << ok << ref.extent << '/' << ref.slot;
    for (const BlockKey& key : evicted) out << " -" << key.digest_hi;
    out << '\n';
  }
  for (size_t cls = 0; cls < a.ClassCount(); ++cls) {
    std::vector<BlockKey> evicted;
    out << a.ReclaimClass(cls, 3, 5, &evicted);
    for (const BlockKey& key : evicted) out << " -" << key.digest_hi;
    out << '\n';
  }
  std::vector<BlockKey> evicted;
  out << a.EvictColdToTarget(a.UsedBytes() / 2, 40, &evicted);
  for (const BlockKey& key : evicted) out << " -" << key.digest_hi;
  out << ' ' << a.ExpireRetention(kEqNow + 100000) << '\n';
  return out.str();
}

struct EqPair {
  explicit EqPair(bool retention)
      : restore(Opts(kEqExtentBytes, kEqExtents)),
        reference(Opts(kEqExtentBytes, kEqExtents)) {
    if (!retention) return;
    for (SlabAllocator* a : {&restore, &reference})
      a->ConfigureRetention(RetainOpts(1 << 20, 3600), kEqNow);
  }
  ::testing::AssertionResult Same(const char* when) {
    const std::string difference =
        FirstDifference(SlabAllocatorTestPeer::Dump(restore),
                        SlabAllocatorTestPeer::Dump(reference));
    if (difference == "identical") return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
           << "state differs " << when << " at " << difference;
  }
  SlabAllocator restore;    // goes through RestoreBulk
  SlabAllocator reference;  // goes through the previous implementation
};
}  // namespace

TEST(SlabAllocatorRestoreEquivalence, SameStateAndSameBehaviourAfterwards) {
  size_t restored = 0;
  for (uint64_t seed = 1; seed <= 40; ++seed) {
    const bool retention = seed % 2 == 0;
    const bool shuffled = seed % 3 == 0;
    const int prior_puts = seed % 4 == 0 ? 0 : static_cast<int>(seed * 7 % 90);
    SCOPED_TRACE("seed " + std::to_string(seed));
    EqPair pair(retention);
    EqPriorOps(pair.restore, seed, prior_puts);
    EqPriorOps(pair.reference, seed, prior_puts);
    ASSERT_TRUE(pair.Same("before the restore"));

    const auto entries =
        EqEntries(pair.restore, seed + 1000, retention, shuffled);
    ASSERT_TRUE(pair.restore.RestoreBulk(entries));
    ASSERT_TRUE(SlabAllocatorTestPeer::ReferenceRestore(&pair.reference, entries));
    ASSERT_TRUE(pair.Same("after the restore"));
    restored += entries.size();

    // A second batch into the allocator the first one left.
    const auto more = EqEntries(pair.restore, seed + 2000, retention, !shuffled);
    ASSERT_TRUE(pair.restore.RestoreBulk(more));
    ASSERT_TRUE(SlabAllocatorTestPeer::ReferenceRestore(&pair.reference, more));
    ASSERT_TRUE(pair.Same("after the second restore"));

    EXPECT_EQ(EqLaterOps(pair.restore, entries, seed + 3000),
              EqLaterOps(pair.reference, entries, seed + 3000));
    ASSERT_TRUE(pair.Same("after the later operations"));
  }
  EXPECT_GT(restored, 2000u) << "the cases restored next to nothing";
}

// A batch that is refused is refused by both, and neither has touched the
// allocator: one case per reason, each planted at the front, in the middle
// and at the end of an otherwise valid batch.
TEST(SlabAllocatorRestoreEquivalence, RefusedBatchesLeaveTheStateAlone) {
  enum Fault { kSameKey, kSamePlacement, kSecondSlotSize, kSlotTaken,
               kSlotOutOfRange, kKeyResident, kFaults };
  size_t refused = 0;
  for (uint64_t seed = 1; seed <= 12; ++seed) {
    for (int fault = 0; fault < kFaults; ++fault) {
      for (int where = 0; where < 3; ++where) {
        SCOPED_TRACE("seed " + std::to_string(seed) + " fault " +
                     std::to_string(fault) + " where " + std::to_string(where));
        EqPair pair(seed % 2 == 0);
        EqPriorOps(pair.restore, seed, 60);
        EqPriorOps(pair.reference, seed, 60);
        auto entries = EqEntries(pair.restore, seed + 1000, seed % 2 == 0,
                                 seed % 3 == 0);
        ASSERT_GT(entries.size(), 8u);
        const std::string before = SlabAllocatorTestPeer::Dump(pair.restore);
        const size_t at = where == 0 ? 0
                          : where == 1 ? entries.size() / 2
                                       : entries.size() - 1;
        const size_t other = at == 0 ? entries.size() - 1 : at - 1;
        const auto sizes = EqSlotSizes();
        switch (fault) {
          case kSameKey:
            entries[at].key = entries[other].key;
            break;
          case kSamePlacement:
            entries[at].extent = entries[other].extent;
            entries[at].slot = entries[other].slot;
            entries[at].slot_size = entries[other].slot_size;
            entries[at].value_bytes = 1;
            break;
          case kSecondSlotSize:
            entries[at].slot_size = entries[at].slot_size == sizes[0]
                                        ? sizes[1] : sizes[0];
            entries[at].slot = 0;
            entries[at].value_bytes = 1;
            break;
          case kSlotTaken: {
            // The slot of something that is already resident.
            SlotRef ref;
            std::vector<BlockKey> evicted;
            const BlockKey resident{seed, 77, 0};
            ASSERT_TRUE(pair.restore.Put(resident, sizes[0], &ref, &evicted));
            ASSERT_TRUE(pair.reference.Put(resident, sizes[0], &ref, &evicted));
            entries = EqEntries(pair.restore, seed + 1000, false, false);
            entries.insert(entries.begin() +
                               static_cast<long>(std::min(at, entries.size())),
                           RestoreEntry{BlockKey{seed, 78, 0}, ref.slot_size,
                                        ref.extent, ref.slot, 1});
            break;
          }
          case kSlotOutOfRange:
            entries[at].slot =
                static_cast<uint32_t>(kEqExtentBytes / entries[at].slot_size);
            break;
          case kKeyResident: {
            SlotRef ref;
            std::vector<BlockKey> evicted;
            const BlockKey resident{seed, 79, 0};
            ASSERT_TRUE(pair.restore.Put(resident, sizes[0], &ref, &evicted));
            ASSERT_TRUE(pair.reference.Put(resident, sizes[0], &ref, &evicted));
            entries = EqEntries(pair.restore, seed + 1000, false, false);
            ASSERT_GT(entries.size(), 2u);
            entries[std::min(at, entries.size() - 1)].key = resident;
            break;
          }
        }
        const std::string untouched = fault == kSlotTaken || fault == kKeyResident
                                          ? SlabAllocatorTestPeer::Dump(pair.restore)
                                          : before;
        const bool restore_ok = pair.restore.RestoreBulk(entries);
        const bool reference_ok =
            SlabAllocatorTestPeer::ReferenceRestore(&pair.reference, entries);
        EXPECT_EQ(restore_ok, reference_ok);
        ASSERT_TRUE(pair.Same("after the refused batch"));
        if (!restore_ok) {
          ++refused;
          EXPECT_EQ(SlabAllocatorTestPeer::Dump(pair.restore), untouched)
              << "a refused batch changed the allocator";
        }
      }
    }
  }
  // Nearly every planted fault must really be one (a second slot size on an
  // extent with a single record, for example, is not).
  EXPECT_GT(refused, 12u * kFaults * 3 * 8 / 10);
}

// Duplicate keys are found through equal 64-bit hashes, and equal hashes
// alone prove nothing. The index hash is a chain of invertible steps, so a
// second key with the hash of a given one can be computed; such a pair must
// be restored like any two keys, and a real duplicate beside it still be
// refused.
namespace {
uint64_t InverseOdd(uint64_t x) {  // x * InverseOdd(x) == 1 (mod 2^64)
  uint64_t inverse = x;
  for (int i = 0; i < 6; ++i) inverse *= 2 - x * inverse;
  return inverse;
}
uint64_t UndoXorShift(uint64_t value, int shift) {
  uint64_t out = value;
  for (int i = shift; i < 64; i += shift) out = value ^ (out >> shift);
  return out;
}
// The index hash's mixing step and its inverse.
uint64_t IndexMix(uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}
uint64_t IndexUnmix(uint64_t value) {
  value = UndoXorShift(value, 31);
  value *= InverseOdd(0x94d049bb133111ebULL);
  value = UndoXorShift(value, 27);
  value *= InverseOdd(0xbf58476d1ce4e5b9ULL);
  return UndoXorShift(value, 30);
}
// Another key with the index hash of `key`: digest_hi is chosen freely and
// digest_lo solved for.
BlockKey SameIndexHash(const BlockKey& key, uint64_t other_digest_hi) {
  const uint64_t lo_term = IndexMix(key.digest_hi + 0x9e3779b97f4a7c15ULL) ^
                           IndexMix(key.digest_lo + 0x3c6ef372fe94f82aULL) ^
                           IndexMix(other_digest_hi + 0x9e3779b97f4a7c15ULL);
  return BlockKey{other_digest_hi, IndexUnmix(lo_term) - 0x3c6ef372fe94f82aULL,
                  key.tenant_hash};
}
}  // namespace

TEST(SlabAllocatorRestoreEquivalence, TwoKeysWithOneHashAreTwoKeys) {
  const BlockKey first{0x1111, 0x2222, 3};
  const BlockKey second = SameIndexHash(first, 0x9999);
  ASSERT_FALSE(first == second);
  ASSERT_EQ(SlabAllocatorTestPeer::IndexHash(first),
            SlabAllocatorTestPeer::IndexHash(second))
      << "the index hash changed; this test's inverse has to follow it";
  const uint32_t slot_size = EqSlotSizes().front();

  EqPair pair(false);
  std::vector<RestoreEntry> entries{
      {BlockKey{1, 1, 0}, slot_size, 0, 0, 100},
      {first, slot_size, 0, 1, 200},
      {BlockKey{2, 2, 0}, slot_size, 1, 0, 300},
      {second, slot_size, 1, 3, 400},
  };
  ASSERT_TRUE(pair.restore.RestoreBulk(entries));
  ASSERT_TRUE(SlabAllocatorTestPeer::ReferenceRestore(&pair.reference, entries));
  ASSERT_TRUE(pair.Same("after restoring two keys with one hash"));
  SlotRef ref;
  ASSERT_TRUE(pair.restore.Get(first, &ref));
  EXPECT_EQ(ref.extent, 0u);
  EXPECT_EQ(ref.slot, 1u);
  ASSERT_TRUE(pair.restore.Get(second, &ref));
  EXPECT_EQ(ref.extent, 1u);
  EXPECT_EQ(ref.slot, 3u);

  // The same pair with a key that really occurs twice: refused, untouched.
  EqPair refused(false);
  const std::string before = SlabAllocatorTestPeer::Dump(refused.restore);
  entries.push_back({BlockKey{1, 1, 0}, slot_size, 2, 0, 500});
  EXPECT_FALSE(refused.restore.RestoreBulk(entries));
  EXPECT_FALSE(
      SlabAllocatorTestPeer::ReferenceRestore(&refused.reference, entries));
  EXPECT_EQ(SlabAllocatorTestPeer::Dump(refused.restore), before);
}

// ---- Whole extents follow the data that is used ----------------------------
//
// Two paths move a whole extent (and evict everything on it) from one size
// class to another that is being written: the background GROW of the tiers'
// reclaim tick (ColdestGrowDonor names a class by write demand, StealFrom
// takes an extent from it) and the cold steal inside Put. Both used to judge
// "cold" by writes alone, so a class that everybody reads and nobody writes
// lost its extents. Now an extent goes only if its residents are much colder
// than what the growing class would evict of its own.
//
// The scenarios run with two classes on 24 extents: A (4 KiB slots, 16 per
// extent) starts with 16 extents, B (16 KiB slots, 4 per extent) with its
// floor of 8. The disk tier reads through Get, the RAM tier through
// PinForRead; both are run.
namespace {
enum class ReadStyle { kDiskGet, kRamPinForRead };
const ReadStyle kBothStyles[] = {ReadStyle::kDiskGet,
                                 ReadStyle::kRamPinForRead};
const char* StyleName(ReadStyle style) {
  return style == ReadStyle::kDiskGet ? "disk-tier reads (Get)"
                                      : "RAM-tier reads (PinForRead)";
}

constexpr uint32_t kHeatExtents = 24;

struct HeatClass {
  size_t slot_size = 0;
  uint64_t tag = 0;  // digest_lo of this class's keys
  uint64_t next = 0;
  std::map<uint64_t, SlabAllocator::SlotHandle> live;
  size_t evicted = 0;
};

struct HeatScenario {
  explicit HeatScenario(ReadStyle read_style)
      : style(read_style), a(Opts(64 << 10, kHeatExtents)) {
    a.ConfigureRetention(RetainOpts(1 << 20, 3600), kT0);
    A.slot_size = 4096;
    A.tag = 1;
    B.slot_size = 16384;
    B.tag = 2;
    // A takes every extent, then B takes its floor of kStripeWays extents
    // from A's oldest (that part is by design and not what is measured).
    for (uint64_t i = 0; i < kHeatExtents * 16; ++i) Put(A);
    for (int i = 0; i < 32; ++i) Put(B);
    const auto classes = a.Classes();
    last_puts.assign(classes.size(), 0);
    for (size_t c = 0; c < classes.size(); ++c) last_puts[c] = classes[c].puts;
    A.evicted = B.evicted = 0;
    moved = 0;
    a_extents_at_start = Extents(A);
    b_extents_at_start = Extents(B);
  }
  void Forget(const std::vector<BlockKey>& evicted) {
    for (const BlockKey& key : evicted) {
      HeatClass& cls = key.digest_lo == A.tag ? A : B;
      if (cls.live.erase(key.digest_hi) != 0) ++cls.evicted;
    }
  }
  void Put(HeatClass& cls) {
    const BlockKey key{cls.next, cls.tag, 0};
    SlotRef ref;
    SlabAllocator::SlotHandle handle;
    std::vector<BlockKey> evicted;
    ASSERT_TRUE(a.Put(key, cls.slot_size, &ref, &evicted, &handle));
    cls.live[cls.next++] = handle;
    Forget(evicted);
  }
  void Read(const HeatClass& cls, uint64_t id) {
    const BlockKey key{id, cls.tag, 0};
    if (style == ReadStyle::kDiskGet) {
      SlotRef ref;
      ASSERT_TRUE(a.Get(key, &ref));
    } else {
      const auto handle = cls.live.at(id);
      ASSERT_TRUE(a.PinForRead(key, handle));
      ASSERT_TRUE(a.Unpin(key, handle));
    }
  }
  // One round of traffic, in an order drawn from a fixed seed so that reads
  // of the two classes and the writes are mixed as real traffic mixes them
  // (a class read wholly after the other would look younger than it is):
  // every `a_one_in`-th resident of A and every `b_one_in`-th of B is read
  // once (0 = none; the part read rotates with `round`), and `writes` new
  // values are admitted to `writer`.
  void Round(int round, size_t a_one_in, size_t b_one_in, HeatClass* writer,
             int writes) {
    struct Op {
      HeatClass* cls;
      uint64_t id;
      bool write;
    };
    std::vector<Op> ops;
    auto reads = [&](HeatClass& cls, size_t one_in) {
      if (one_in == 0) return;
      size_t n = 0;
      for (const auto& [id, handle] : cls.live) {
        (void)handle;
        if (n++ % one_in == static_cast<size_t>(round) % one_in)
          ops.push_back({&cls, id, false});
      }
    };
    reads(A, a_one_in);
    reads(B, b_one_in);
    for (int i = 0; i < writes; ++i) ops.push_back({writer, 0, true});
    // Shuffled by hand: std::shuffle draws its numbers in a way that is left
    // to the standard library, and the assertions below are exact.
    for (size_t i = ops.size(); i > 1; --i)
      std::swap(ops[i - 1], ops[static_cast<size_t>(rng() % i)]);
    for (const Op& op : ops) {
      if (op.write)
        Put(*op.cls);
      else if (op.cls->live.count(op.id) != 0)  // not evicted meanwhile
        Read(*op.cls, op.id);
    }
  }
  // What a tier's reclaim tick does for every class that was written since
  // the last tick and is short of free slots: up to `max_extents` whole
  // extents from the donor ColdestGrowDonor names.
  void GrowTick() {
    const size_t max_extents = style == ReadStyle::kDiskGet ? 2 : 8;
    const auto classes = a.Classes();
    std::vector<uint64_t> delta(classes.size(), 0);
    std::vector<uint32_t> extents(classes.size(), 0);
    for (size_t c = 0; c < classes.size(); ++c) {
      delta[c] = classes[c].puts - last_puts[c];
      last_puts[c] = classes[c].puts;
      extents[c] = classes[c].extents;
    }
    for (size_t hot = 0; hot < classes.size(); ++hot) {
      if (delta[hot] == 0) continue;
      const size_t want = SlabAllocator::FreeSlotTarget(
          delta[hot], classes[hot].slot_size,
          std::numeric_limits<size_t>::max());
      if (classes[hot].free_slots >= want) continue;
      size_t taken = 0;
      while (taken < max_extents) {
        const size_t donor =
            SlabAllocator::ColdestGrowDonor(classes, delta, extents, hot,
                                            a.StripeWays());
        if (donor == classes.size()) break;
        std::vector<BlockKey> evicted;
        const bool ok = a.StealFrom(donor, hot, &evicted);
        Forget(evicted);
        if (!ok) {
          extents[donor] = 0;
          continue;
        }
        --extents[donor];
        ++taken;
        ++moved;
      }
    }
  }
  uint32_t Extents(const HeatClass& cls) const {
    for (const auto& stat : a.Classes())
      if (stat.slot_size == cls.slot_size) return stat.extents;
    return 0;
  }
  // Extents that changed class since the scenario was set up, by either path.
  uint64_t Moves() const { return moved + a.ColdSteals(); }
  void Report(const char* what) const {
    std::printf(
        "EXTENT_MOVES %s, %s: A %u -> %u extents (%zu residents evicted), "
        "B %u -> %u (%zu evicted); %llu by GROW, %llu cold steals\n",
        what, StyleName(style), a_extents_at_start, Extents(A), A.evicted,
        b_extents_at_start, Extents(B), B.evicted,
        static_cast<unsigned long long>(moved),
        static_cast<unsigned long long>(a.ColdSteals()));
  }

  ReadStyle style;
  SlabAllocator a;
  HeatClass A, B;
  std::mt19937_64 rng{20261122};
  std::vector<uint64_t> last_puts;
  uint64_t moved = 0;
  uint32_t a_extents_at_start = 0, b_extents_at_start = 0;
};

// The two paths are exercised separately: the GROW variant switches the
// cold steal off, the other never runs a tick.
enum class StealPath { kInlineColdSteal, kBackgroundGrow };
const StealPath kBothPaths[] = {StealPath::kInlineColdSteal,
                                StealPath::kBackgroundGrow};
const char* PathName(StealPath path) {
  return path == StealPath::kInlineColdSteal ? "inline cold steal"
                                             : "background GROW";
}
struct ScopedColdSteal {
  explicit ScopedColdSteal(StealPath path) {
    const char* before = std::getenv("DFKV_SLAB_COLD_STEAL_WINDOW");
    was_set = before != nullptr;
    if (was_set) previous = before;
    // Each path on its own, whatever the environment says: the cold steal
    // off for the GROW variant, its window derived for the other.
    if (path == StealPath::kBackgroundGrow)
      ::setenv("DFKV_SLAB_COLD_STEAL_WINDOW", "0", 1);
    else
      ::unsetenv("DFKV_SLAB_COLD_STEAL_WINDOW");
  }
  ~ScopedColdSteal() {
    if (was_set)
      ::setenv("DFKV_SLAB_COLD_STEAL_WINDOW", previous.c_str(), 1);
    else
      ::unsetenv("DFKV_SLAB_COLD_STEAL_WINDOW");
  }
  bool was_set = false;
  std::string previous;
};
}  // namespace

// The finding. A is read all the time and never written; B is written and
// never read. A keeps every extent: B's writes replace B's own residents.
TEST(SlabAllocatorExtentMoves, AClassThatIsReadIsNotDrainedByOneThatIsWritten) {
  for (const StealPath path : kBothPaths) {
    for (const ReadStyle style : kBothStyles) {
      SCOPED_TRACE(std::string(PathName(path)) + ", " + StyleName(style));
      ScopedColdSteal cold_steal(path);
      HeatScenario s(style);
      ASSERT_EQ(s.a_extents_at_start, kHeatExtents - SlabAllocator::kStripeWays);
      for (int round = 0; round < 64; ++round) {
        s.Round(round, /*A*/ 1, /*B*/ 0, &s.B, 8);
        if (path == StealPath::kBackgroundGrow) s.GrowTick();
      }
      s.Report("A read, B written");
      EXPECT_EQ(s.Extents(s.A), s.a_extents_at_start);
      EXPECT_EQ(s.A.evicted, 0u);
      EXPECT_EQ(s.Moves(), 0u);
    }
  }
}

// The other failure a rule could have: freezing the layout. A is established
// and read at a moderate rate (an eighth of it per round); B is new, written,
// and all of it is read every round. B's data is used far more recently than
// A's least used extent, so B grows beyond its floor at A's expense.
TEST(SlabAllocatorExtentMoves, AHotterClassGrowsAtTheExpenseOfOneReadLess) {
  for (const StealPath path : kBothPaths) {
    for (const ReadStyle style : kBothStyles) {
      SCOPED_TRACE(std::string(PathName(path)) + ", " + StyleName(style));
      ScopedColdSteal cold_steal(path);
      HeatScenario s(style);
      for (int round = 0; round < 64; ++round) {
        s.Round(round, /*A*/ 8, /*B*/ 1, &s.B, 8);
        if (path == StealPath::kBackgroundGrow) s.GrowTick();
      }
      s.Report("A read a little, B written and read");
      EXPECT_GT(s.Extents(s.B), s.b_extents_at_start);
      EXPECT_LT(s.Extents(s.A), s.a_extents_at_start);
      EXPECT_GT(s.Moves(), 0u);
    }
  }
}

// No ping-pong. Both classes are read in full every round and take turns
// being written, eight rounds each. Neither is colder than the other by the
// margin, so no extent changes hands, however often the writer changes.
TEST(SlabAllocatorExtentMoves, EquallyUsedClassesDoNotTradeExtents) {
  for (const StealPath path : kBothPaths) {
    for (const ReadStyle style : kBothStyles) {
      SCOPED_TRACE(std::string(PathName(path)) + ", " + StyleName(style));
      ScopedColdSteal cold_steal(path);
      HeatScenario s(style);
      for (int round = 0; round < 128; ++round) {
        HeatClass& writer = (round / 8) % 2 == 0 ? s.B : s.A;
        s.Round(round, /*A*/ 1, /*B*/ 1, &writer, 8);
        if (path == StealPath::kBackgroundGrow) s.GrowTick();
      }
      s.Report("both read, written in turns");
      EXPECT_EQ(s.Moves(), 0u);
      EXPECT_EQ(s.Extents(s.A), s.a_extents_at_start);
      EXPECT_EQ(s.Extents(s.B), s.b_extents_at_start);
    }
  }
}

// What the growing class is measured against is an extent it could lose. B
// holds one extent whose four residents are all under retention: nothing on
// it can be evicted or stolen, and it only grows older. Compared with that
// one, the unused class A would never be "much colder" and B would be left
// evicting its own values for as long as the retention lasts. B's least
// recently used extent among those it can lose is the comparison, and A
// gives its extents as any unused class does.
TEST(SlabAllocatorExtentMoves, AnExtentHeldByRetentionIsNotTheMeasure) {
  for (const StealPath path : kBothPaths) {
    for (const ReadStyle style : kBothStyles) {
      SCOPED_TRACE(std::string(PathName(path)) + ", " + StyleName(style));
      ScopedColdSteal cold_steal(path);
      HeatScenario s(style);
      // All residents of one of B's extents.
      std::map<uint32_t, std::vector<uint64_t>> by_extent;
      for (const auto& [id, handle] : s.B.live) {
        (void)handle;
        SlotRef ref;
        ASSERT_TRUE(s.a.Get(BlockKey{id, s.B.tag, 0}, &ref));
        by_extent[ref.extent].push_back(id);
      }
      const auto& held = by_extent.begin()->second;
      ASSERT_EQ(held.size(), 4u);
      for (uint64_t id : held)
        ASSERT_EQ(s.a.Retain(BlockKey{id, s.B.tag, 0}, 3000, RetainMode::kSet,
                             kT0)
                      .code,
                  RetainCode::kProtectedNew);
      // A is read once, now, and never again: it and the retained extent
      // start level and age together.
      s.Round(0, /*A*/ 1, /*B*/ 0, &s.B, 0);
      for (int round = 0; round < 128; ++round) {
        s.Round(round, /*A*/ 0, /*B*/ 0, &s.B, 8);
        if (path == StealPath::kBackgroundGrow) s.GrowTick();
      }
      s.Report("A unused, B written, one extent of B retained");
      EXPECT_GT(s.Extents(s.B), s.b_extents_at_start);
      EXPECT_LT(s.Extents(s.A), s.a_extents_at_start);
      for (uint64_t id : held)
        EXPECT_TRUE(s.a.Contains(BlockKey{id, s.B.tag, 0})) << "retained";
    }
  }
}

// A class that is neither written nor read is cold, and still gives its
// extents to the one that is written: by GROW down to its floor, by the cold
// steal without one, as before.
TEST(SlabAllocatorExtentMoves, AnUnusedClassStillGivesItsExtents) {
  for (const StealPath path : kBothPaths) {
    for (const ReadStyle style : kBothStyles) {
      SCOPED_TRACE(std::string(PathName(path)) + ", " + StyleName(style));
      ScopedColdSteal cold_steal(path);
      HeatScenario s(style);
      for (int round = 0; round < 128; ++round) {
        s.Round(round, /*A*/ 0, /*B*/ 0, &s.B, 8);
        if (path == StealPath::kBackgroundGrow) s.GrowTick();
      }
      s.Report("A unused, B written");
      // Both ways below the striping floor: A writes nothing, so it has
      // nothing to stripe.
      EXPECT_LT(s.Extents(s.A), SlabAllocator::kStripeWays);
    }
  }
}

// ---- The retention wheel: one entry per protected resident, bounded expiry --
namespace {
// `count` residents of 4 KiB in an allocator large enough for them, with a
// retention budget that covers them all.
struct RetainedStore {
  explicit RetainedStore(uint64_t count, uint32_t max_ttl_s = 3600)
      : a(Opts(64 << 10, static_cast<uint32_t>(count / 16 + 9))) {
    a.ConfigureRetention(RetainOpts(count * 4096 * 2, max_ttl_s), kT0);
    for (uint64_t i = 0; i < count; ++i) {
      SlotRef ref;
      std::vector<BlockKey> evicted;
      EXPECT_TRUE(a.Put(BlockKey{i, 9, 0}, 4096, &ref, &evicted));
    }
  }
  SlabAllocator a;
};
}  // namespace

// A resident that is renewed over and over has one entry on the wheel, not
// one per renewal. Before, every renewal that moved the deadline (one per
// second at most) left an entry behind until its bucket expired.
TEST(SlabAllocatorRetainWheel, RenewalsDoNotPileUpEntries) {
  constexpr uint64_t kCount = 500;
  RetainedStore store(kCount, 600);
  for (uint64_t i = 0; i < kCount; ++i)
    ASSERT_EQ(store.a.Retain(BlockKey{i, 9, 0}, 600, RetainMode::kSet, kT0).code,
              RetainCode::kProtectedNew);
  EXPECT_EQ(store.a.GetRetentionStats().wheel_entries, kCount);
  // Ten minutes of one renewal per second and resident, with the expiry
  // running every second as the store's reclaim tick does.
  uint64_t most = 0;
  for (uint32_t second = 1; second <= 600; ++second) {
    for (uint64_t i = 0; i < kCount; ++i)
      store.a.Retain(BlockKey{i, 9, 0}, 0, RetainMode::kRefresh, kT0 + second);
    EXPECT_EQ(store.a.ExpireRetention(kT0 + second), 0u) << "second " << second;
    most = std::max(most, store.a.GetRetentionStats().wheel_entries);
  }
  std::printf("RETAIN_WHEEL %llu residents renewed once a second for 600 s: "
              "at most %llu wheel entries\n",
              static_cast<unsigned long long>(kCount),
              static_cast<unsigned long long>(most));
  EXPECT_EQ(most, kCount);
  EXPECT_EQ(store.a.GetRetentionStats().protected_slots, kCount);
  // Renewals stop: everything expires once its last deadline has passed, and
  // the wheel is empty.
  EXPECT_EQ(store.a.ExpireRetention(kT0 + 600 + 600 + 20), kCount);
  EXPECT_EQ(store.a.GetRetentionStats().protected_slots, 0u);
  EXPECT_EQ(store.a.GetRetentionStats().wheel_entries, 0u);
}

// The deadline that counts is the latest one. A resident renewed after its
// entry was pushed stays protected when the bucket of its first deadline is
// drained, and is released when the bucket of its last one is: never early,
// never kept past it.
TEST(SlabAllocatorRetainWheel, ARenewedResidentIsHeldUntilItsLastDeadline) {
  RetainedStore store(4, 600);
  const BlockKey renewed{0, 9, 0}, plain{1, 9, 0};
  ASSERT_EQ(store.a.Retain(renewed, 60, RetainMode::kSet, kT0).code,
            RetainCode::kProtectedNew);
  ASSERT_EQ(store.a.Retain(plain, 60, RetainMode::kSet, kT0).code,
            RetainCode::kProtectedNew);
  // Renewed at +50: the deadline moves from +60 to +110.
  ASSERT_EQ(store.a.Retain(renewed, 0, RetainMode::kRefresh, kT0 + 50).code,
            RetainCode::kRefreshed);
  // The bucket of +60 is due at +80: `plain` goes, `renewed` stays.
  EXPECT_EQ(store.a.ExpireRetention(kT0 + 80), 1u);
  EXPECT_EQ(store.a.GetRetentionStats().protected_slots, 1u);
  EXPECT_EQ(store.a.GetRetentionStats().wheel_entries, 1u);
  EXPECT_EQ(store.a.ExpireRetention(kT0 + 109), 0u);
  EXPECT_EQ(store.a.ExpireRetention(kT0 + 119), 0u) << "its bucket is not past";
  EXPECT_EQ(store.a.ExpireRetention(kT0 + 130), 1u);
  EXPECT_EQ(store.a.GetRetentionStats().protected_slots, 0u);
  EXPECT_EQ(store.a.GetRetentionStats().wheel_entries, 0u);
}

// The entry of a protection that is gone (the key was removed, then put and
// protected again) neither releases the new protection early nor stays on
// the wheel: it is dropped when its bucket comes up.
TEST(SlabAllocatorRetainWheel, AnEntryOfARemovedResidentIsDroppedAndHarmless) {
  RetainedStore store(4, 600);
  const BlockKey key{0, 9, 0};
  ASSERT_EQ(store.a.Retain(key, 60, RetainMode::kSet, kT0).code,
            RetainCode::kProtectedNew);
  ASSERT_EQ(store.a.Remove(key), SlabAllocator::RemoveResult::kRemoved);
  EXPECT_EQ(store.a.GetRetentionStats().protected_slots, 0u);
  EXPECT_EQ(store.a.GetRetentionStats().wheel_entries, 1u) << "left behind";
  SlotRef ref;
  std::vector<BlockKey> evicted;
  ASSERT_TRUE(store.a.Put(key, 4096, &ref, &evicted));
  ASSERT_EQ(store.a.Retain(key, 300, RetainMode::kSet, kT0 + 10).code,
            RetainCode::kProtectedNew);
  EXPECT_EQ(store.a.GetRetentionStats().wheel_entries, 2u);
  // The old entry's bucket (+60) is due: nothing is released, it is gone.
  EXPECT_EQ(store.a.ExpireRetention(kT0 + 80), 0u);
  EXPECT_EQ(store.a.GetRetentionStats().protected_slots, 1u);
  EXPECT_EQ(store.a.GetRetentionStats().wheel_entries, 1u);
  // The new protection ends at +310.
  EXPECT_EQ(store.a.ExpireRetention(kT0 + 300), 0u);
  EXPECT_EQ(store.a.ExpireRetention(kT0 + 330), 1u);
  EXPECT_EQ(store.a.GetRetentionStats().wheel_entries, 0u);
}

// Many residents protected with one deadline expire in one bucket. An expiry
// looks at no more than it is told to under one hold of the lock, says that
// it has more to do, and continues where it stopped; the sum is the same as
// in one go, and operations get the lock in between.
TEST(SlabAllocatorRetainWheel, ExpiryWorksInBoundedBatches) {
  constexpr uint64_t kCount = 20000;
  constexpr size_t kBatch = 1024;
  RetainedStore store(kCount, 600);
  for (uint64_t i = 0; i < kCount; ++i)
    ASSERT_EQ(store.a.Retain(BlockKey{i, 9, 0}, 60, RetainMode::kSet, kT0).code,
              RetainCode::kProtectedNew);
  SlotRef ref;
  bool more = true;
  size_t calls = 0, released = 0;
  while (more) {
    const size_t got = store.a.ExpireRetention(kT0 + 80, kBatch, &more);
    EXPECT_LE(got, kBatch);
    released += got;
    ++calls;
    // Between two batches the store is usable and consistent: a read and an
    // admission go through, and what is still protected is still counted.
    ASSERT_TRUE(store.a.Get(BlockKey{0, 9, 0}, &ref));
    EXPECT_EQ(store.a.GetRetentionStats().protected_slots, kCount - released);
    ASSERT_LT(calls, 1000u);
  }
  EXPECT_EQ(released, kCount);
  EXPECT_GE(calls, kCount / kBatch);
  EXPECT_EQ(store.a.GetRetentionStats().protected_slots, 0u);
  EXPECT_EQ(store.a.GetRetentionStats().wheel_entries, 0u);
  EXPECT_EQ(store.a.GetRetentionStats().expired, kCount);
  // Nothing is left to do, and without a limit it is one call as before.
  EXPECT_EQ(store.a.ExpireRetention(kT0 + 80, kBatch, &more), 0u);
  EXPECT_FALSE(more);
}

// Not a correctness test: how long the lock is held to expire one bucket,
// in one go and in batches, and how long a reader waits meanwhile. Set
// DFKV_RETAIN_WHEEL_BENCH_OBJECTS (for example 1000000) to run it.
TEST(SlabAllocatorRetainWheel, ExpiryLockHoldBench) {
  const char* requested = std::getenv("DFKV_RETAIN_WHEEL_BENCH_OBJECTS");
  if (requested == nullptr || *requested == '\0')
    GTEST_SKIP() << "set DFKV_RETAIN_WHEEL_BENCH_OBJECTS to run";
  const uint64_t count = std::strtoull(requested, nullptr, 10);
  ASSERT_GT(count, 0u);
  for (const size_t batch : {size_t{0}, size_t{1024}}) {
    RetainedStore store(count, 600);
    for (uint64_t i = 0; i < count; ++i)
      store.a.Retain(BlockKey{i, 9, 0}, 60, RetainMode::kSet, kT0);
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> worst_ns{0}, reads{0};
    std::thread reader([&] {
      std::mt19937_64 rng(3);
      while (!stop.load(std::memory_order_relaxed)) {
        SlotRef ref;
        const auto t0 = std::chrono::steady_clock::now();
        store.a.Get(BlockKey{rng() % count, 9, 0}, &ref);
        const uint64_t ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0)
                .count());
        if (ns > worst_ns.load(std::memory_order_relaxed)) worst_ns = ns;
        reads.fetch_add(1, std::memory_order_relaxed);
      }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    worst_ns = 0;
    const auto started = std::chrono::steady_clock::now();
    double longest_hold_ms = 0;
    size_t released = 0;
    bool more = true;
    while (more) {
      const auto t0 = std::chrono::steady_clock::now();
      released += store.a.ExpireRetention(kT0 + 80, batch ? batch : SIZE_MAX,
                                          &more);
      longest_hold_ms = std::max(
          longest_hold_ms, std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t0)
                               .count());
      // As the store does between batches; see DiskSlabStore::ReclaimTick.
      if (more) std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    const double total_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - started)
                                .count();
    stop = true;
    reader.join();
    ASSERT_EQ(released, count);
    std::printf("RETAIN_WHEEL_BENCH %llu residents in one bucket, %s: expiry "
                "%.1f ms in all, longest hold %.3f ms, a reader waited at "
                "most %.3f ms\n",
                static_cast<unsigned long long>(count),
                batch ? "batches of 1024" : "one hold", total_ms,
                longest_hold_ms, worst_ns.load() / 1e6);
  }
}

// ---- STO-35: BalanceTailAges ----
namespace {
// 8 extents of 64 KiB. Class A (4 KiB slots, 16 per extent) is written first
// and then left alone; class B (8 KiB, 8 per extent) takes the other four
// extents and then evicts its own values. A's next victim is much older.
// A cold window longer than any of these tests: Put's cold steal (which
// takes an idle class's extents on its own) stays out of the way, and the
// balancer's hysteresis lasts throughout.
std::unique_ptr<SlabAllocator> BalanceAllocator() {
  auto o = Opts(64 * 1024, 8);
  o.stripe_ways = 1;
  ::setenv("DFKV_SLAB_COLD_STEAL_WINDOW", "100000", 1);
  auto a = std::make_unique<SlabAllocator>(o);
  ::unsetenv("DFKV_SLAB_COLD_STEAL_WINDOW");
  return a;
}
void FillTwoClasses(SlabAllocator& a) {
  std::vector<BlockKey> ev;
  SlotRef r;
  for (int i = 0; i < 64; ++i)
    ASSERT_TRUE(a.Put(Key("a" + std::to_string(i)), 4096, &r, &ev));
  for (int i = 0; i < 64; ++i)
    ASSERT_TRUE(a.Put(Key("b" + std::to_string(i)), 8192, &r, &ev));
}
}  // namespace

TEST(SlabAllocator, BalanceMovesAnExtentToTheClassWithTheYoungerTail) {
  auto owned = BalanceAllocator();
  SlabAllocator& a = *owned;
  ASSERT_NO_FATAL_FAILURE(FillTwoClasses(a));
  const size_t ca = ClassIndex(a, 4096), cb = ClassIndex(a, 8192);
  ASSERT_EQ(a.Classes()[ca].extents, 4u);
  ASSERT_EQ(a.Classes()[cb].extents, 4u);
  EXPECT_GT(a.Classes()[ca].tail_age, 2 * a.Classes()[cb].tail_age);
  std::vector<BlockKey> ev;
  EXPECT_EQ(a.BalanceTailAges(150, 1, &ev), 1u);
  EXPECT_EQ(a.Classes()[ca].extents, 3u);
  EXPECT_EQ(a.Classes()[cb].extents, 5u);
  EXPECT_EQ(ev.size(), 16u) << "one of A's extents, its residents evicted";
  for (const auto& key : ev) EXPECT_FALSE(a.Contains(key));
  EXPECT_EQ(a.BalanceMoves(), 1u);
  // Nothing has evicted since: no class is short of space.
  EXPECT_EQ(a.BalanceTailAges(150, 1, &ev), 0u);
}

TEST(SlabAllocator, BalanceNeedsTheRatio) {
  auto owned = BalanceAllocator();
  SlabAllocator& a = *owned;
  ASSERT_NO_FATAL_FAILURE(FillTwoClasses(a));
  std::vector<BlockKey> ev;
  EXPECT_EQ(a.BalanceTailAges(100000, 1, &ev), 0u);
  EXPECT_TRUE(ev.empty());
}

// A read or write in flight pins its slot; an extent holding one stays.
TEST(SlabAllocator, BalanceNeverMovesAnExtentWithAPinnedSlot) {
  auto owned = BalanceAllocator();
  SlabAllocator& a = *owned;
  ASSERT_NO_FATAL_FAILURE(FillTwoClasses(a));
  for (int i = 0; i < 64; i += 16) ASSERT_TRUE(a.Pin(Key("a" + std::to_string(i))));
  std::vector<BlockKey> ev;
  EXPECT_EQ(a.BalanceTailAges(150, 1, &ev), 0u);
  EXPECT_TRUE(ev.empty());
  for (int i = 0; i < 64; ++i) EXPECT_TRUE(a.Contains(Key("a" + std::to_string(i))));
}

// A protected (RETAIN) value keeps its extent where it is.
TEST(SlabAllocator, BalanceNeverMovesAnExtentWithAProtectedValue) {
  auto owned = BalanceAllocator();
  SlabAllocator& a = *owned;
  a.ConfigureRetention(RetainOpts(1 << 20), kT0);
  ASSERT_NO_FATAL_FAILURE(FillTwoClasses(a));
  for (int i = 0; i < 64; i += 16)
    ASSERT_EQ(a.Retain(Key("a" + std::to_string(i)), 600,
                       SlabAllocator::RetainMode::kSet, kT0)
                  .code,
              SlabAllocator::RetainCode::kProtectedNew);
  std::vector<BlockKey> ev;
  EXPECT_EQ(a.BalanceTailAges(150, 1, &ev), 0u);
  for (int i = 0; i < 64; ++i) EXPECT_TRUE(a.Contains(Key("a" + std::to_string(i))));
}

// An extent the balancer moved does not move again within a cold window,
// even when the pressure turns round at once.
TEST(SlabAllocator, AMovedExtentDoesNotMoveStraightBack) {
  auto owned = BalanceAllocator();
  SlabAllocator& a = *owned;
  ASSERT_NO_FATAL_FAILURE(FillTwoClasses(a));
  std::vector<BlockKey> ev;
  ASSERT_EQ(a.BalanceTailAges(150, 1, &ev), 1u);
  const size_t ca = ClassIndex(a, 4096), cb = ClassIndex(a, 8192);
  // B now leaves its values alone and A evicts: B's tail grows old.
  SlotRef r;
  uint32_t moved_extent = UINT32_MAX;
  {
    // The extent B took holds B's newest value.
    ASSERT_TRUE(a.Put(Key("b-new"), 8192, &r, &ev));
    moved_extent = r.extent;
  }
  for (int i = 64; i < 200; ++i)
    ASSERT_TRUE(a.Put(Key("a" + std::to_string(i)), 4096, &r, &ev));
  ev.clear();
  const uint32_t before = a.Classes()[cb].extents;
  const size_t moves = a.BalanceTailAges(150, 8, &ev);
  EXPECT_GT(moves, 0u) << "the pressure turned round: A should grow back";
  // B's other extents may go back to A; the one it just took stays.
  ASSERT_TRUE(a.Contains(Key("b-new")))
      << "the extent moved in moments ago was moved again (extent "
      << moved_extent << ")";
  EXPECT_EQ(a.Classes()[cb].extents, before - moves);
  EXPECT_GE(a.Classes()[ca].extents, 3u);
}
