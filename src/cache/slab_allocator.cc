#include "utils/env_parse.h"
#include "cache/slab_allocator.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <unordered_map>
#include <utility>

#include "common/config_dump.h"

namespace dfkv {
namespace {

uint64_t AlignUp(uint64_t x, uint64_t a) {
  if (a <= 1) return x;
  if (x > std::numeric_limits<uint64_t>::max() - (a - 1)) return 0;
  return ((x + (a - 1)) / a) * a;
}

}  // namespace

uint64_t SlabAllocator::FlatKeyIndex::Hash(const BlockKey& key) {
  auto mix = [](uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  };
  uint64_t hash = mix(key.digest_hi + 0x9e3779b97f4a7c15ULL);
  hash ^= mix(key.digest_lo + 0x3c6ef372fe94f82aULL);
  hash ^= mix(key.tenant_hash + 0xdaa66d2c7ddef743ULL);
  return mix(hash);
}

void SlabAllocator::FlatKeyIndex::Prefetch(const BlockKey& key) const {
  if (entries_.empty()) return;
  __builtin_prefetch(
      &entries_[static_cast<size_t>(Hash(key)) & (entries_.size() - 1)], 1);
}

uint32_t* SlabAllocator::FlatKeyIndex::Find(const BlockKey& key) {
  if (entries_.empty()) return nullptr;
  const size_t mask = entries_.size() - 1;
  size_t bucket = static_cast<size_t>(Hash(key)) & mask;
  for (size_t probed = 0; probed < entries_.size(); ++probed) {
    Entry& entry = entries_[bucket];
    if (entry.state == 0) return nullptr;
    if (entry.state == 1 && entry.key == key) return &entry.value;
    bucket = (bucket + 1) & mask;
  }
  return nullptr;
}

const uint32_t* SlabAllocator::FlatKeyIndex::Find(
    const BlockKey& key) const {
  return const_cast<FlatKeyIndex*>(this)->Find(key);
}

void SlabAllocator::FlatKeyIndex::Rehash(size_t requested_capacity) {
  size_t capacity = 16;
  while (capacity < requested_capacity) capacity *= 2;
  std::vector<Entry> replacement(capacity);
  const size_t mask = capacity - 1;
  for (const Entry& old : entries_) {
    if (old.state != 1) continue;
    size_t bucket = static_cast<size_t>(Hash(old.key)) & mask;
    while (replacement[bucket].state == 1) bucket = (bucket + 1) & mask;
    replacement[bucket] = old;
  }
  entries_.swap(replacement);
  tombstones_ = 0;
  ++table_allocations_;
  ++rehashes_;
}

void SlabAllocator::FlatKeyIndex::Reserve(size_t expected_size) {
  size_t capacity = entries_.empty() ? 16 : entries_.size();
  while (expected_size > (capacity * 7) / 10) capacity *= 2;
  if (entries_.empty() || capacity != entries_.size() ||
      (tombstones_ != 0 &&
       (expected_size + tombstones_ > (capacity * 7) / 10 ||
        tombstones_ > capacity / 4))) {
    Rehash(capacity);
  }
}

bool SlabAllocator::FlatKeyIndex::Insert(const BlockKey& key,
                                         uint32_t value) {
  if (uint32_t* existing = Find(key)) {
    *existing = value;
    return false;
  }
  Reserve(size_ + 1);
  const size_t mask = entries_.size() - 1;
  size_t bucket = static_cast<size_t>(Hash(key)) & mask;
  size_t tombstone = entries_.size();
  for (size_t probed = 0; probed < entries_.size(); ++probed) {
    Entry& entry = entries_[bucket];
    if (entry.state == 0) {
      const size_t target = tombstone == entries_.size() ? bucket : tombstone;
      Entry& inserted = entries_[target];
      if (inserted.state == 2) --tombstones_;
      inserted.key = key;
      inserted.value = value;
      inserted.state = 1;
      ++size_;
      return true;
    }
    if (entry.state == 2 && tombstone == entries_.size()) tombstone = bucket;
    bucket = (bucket + 1) & mask;
  }
  return false;
}

bool SlabAllocator::FlatKeyIndex::Erase(const BlockKey& key) {
  if (entries_.empty()) return false;
  const size_t mask = entries_.size() - 1;
  size_t bucket = static_cast<size_t>(Hash(key)) & mask;
  for (size_t probed = 0; probed < entries_.size(); ++probed) {
    Entry& entry = entries_[bucket];
    if (entry.state == 0) return false;
    if (entry.state == 1 && entry.key == key) {
      entry.state = 2;
      entry.value = kNoSlot;
      --size_;
      ++tombstones_;
      return true;
    }
    bucket = (bucket + 1) & mask;
  }
  return false;
}

SlabAllocator::IndexStats SlabAllocator::FlatKeyIndex::stats() const {
  return IndexStats{size_, entries_.size(), tombstones_, table_allocations_,
                    rehashes_};
}

size_t SlabAllocator::FlatKeyIndex::BucketForTest(const BlockKey& key) const {
  return entries_.empty()
             ? 0
             : static_cast<size_t>(Hash(key)) & (entries_.size() - 1);
}

SlabAllocator::SlabAllocator(Options opt) : opt_(std::move(opt)) {
  if (opt_.align == 0) opt_.align = 1;
  if (opt_.num_extents == 0) opt_.num_extents = 1;
  if (opt_.max_waste < 0) opt_.max_waste = 0;
  if (opt_.stripe_ways == 0) opt_.stripe_ways = 1;
  extents_.assign(opt_.num_extents, ExtentMeta{});
  unbound_ = opt_.num_extents;

  // Establish the complete canonical class lattice once. Class indices and
  // counts are thereafter immutable and independent of request/restore order.
  const uint64_t uint32_aligned =
      (static_cast<uint64_t>(UINT32_MAX) / opt_.align) * opt_.align;
  const uint64_t max_aligned = std::min(
      (opt_.extent_bytes / opt_.align) * static_cast<uint64_t>(opt_.align),
      uint32_aligned);
  for (uint64_t slot_size = opt_.align; slot_size <= max_aligned;) {
    Class cls;
    cls.slot_size = static_cast<uint32_t>(slot_size);
    cls.slots_per_extent =
        static_cast<uint32_t>(opt_.extent_bytes / slot_size);
    classes_.push_back(std::move(cls));
    if (slot_size == max_aligned) break;
    const uint64_t next = CanonicalSlotSize(slot_size + opt_.align);
    if (next <= slot_size) break;
    slot_size = next;
  }

  {
    // 0 = cold steal off; unset or invalid leaves the window to be derived.
    envparse::U64Spec spec;
    spec.legacy = envparse::Legacy::kPrefix;
    spec.negative = envparse::Negative::kWraps;
    spec.above = envparse::Above::kClamp;  // strtoull saturates
    bool given = false;
    const uint64_t v =
        envparse::EnvU64("DFKV_SLAB_COLD_STEAL_WINDOW", spec, &given);
    if (given) {
      if (v == 0) cold_steal_enabled_ = false;
      else cold_window_ = v;
    }
  }
  config_dump::RecordResolved(
      "DFKV_SLAB_COLD_STEAL_WINDOW",
      !cold_steal_enabled_ ? std::string("disabled")
          : (cold_window_ ? std::to_string(cold_window_) : std::string("auto")));
}

void SlabAllocator::PushFreeLocked(Class& cls, uint32_t extent,
                                   uint32_t slot) {
  auto& slots = cls.free_by_ext[extent];
  if (slots.empty()) cls.ext_rr.push_back(extent);
  slots.push_back(slot);
  ++cls.free_count;
}

bool SlabAllocator::PopFreeLocked(Class& cls, Slot* out) {
  if (cls.ext_rr.empty()) return false;
  if (cls.rr_next >= cls.ext_rr.size()) cls.rr_next = 0;
  const uint32_t extent = cls.ext_rr[cls.rr_next];
  auto it = cls.free_by_ext.find(extent);
  std::vector<uint32_t>& slots = it->second;
  const uint32_t slot = slots.back();
  slots.pop_back();
  --cls.free_count;
  if (slots.empty()) {
    cls.ext_rr[cls.rr_next] = cls.ext_rr.back();
    cls.ext_rr.pop_back();
  } else {
    ++cls.rr_next;
  }
  if (out) *out = Slot{extent, slot};
  return true;
}

void SlabAllocator::DropFromRotationLocked(Class& cls, uint32_t extent) {
  for (size_t i = 0; i < cls.ext_rr.size(); ++i) {
    if (cls.ext_rr[i] != extent) continue;
    cls.ext_rr[i] = cls.ext_rr.back();
    cls.ext_rr.pop_back();
    if (cls.rr_next > i) --cls.rr_next;
    if (cls.rr_next >= cls.ext_rr.size() && !cls.ext_rr.empty()) cls.rr_next = 0;
    return;
  }
}

void SlabAllocator::DropExtentFreeLocked(Class& cls, uint32_t extent) {
  auto it = cls.free_by_ext.find(extent);
  if (it == cls.free_by_ext.end()) return;
  cls.free_count -= it->second.size();
  cls.free_by_ext.erase(it);
  DropFromRotationLocked(cls, extent);
}

uint32_t SlabAllocator::CanonicalSlotSize(size_t aligned_len) const {
  // Build a fixed geometric lattice from alignment and extent geometry. The
  // next bucket is the largest aligned slot for which the smallest request not
  // served by the preceding bucket still satisfies max_waste. Consequently a
  // request's class never depends on which other request arrived first.
  const uint64_t uint32_aligned =
      (static_cast<uint64_t>(UINT32_MAX) / opt_.align) * opt_.align;
  const uint64_t max_aligned = std::min(
      (opt_.extent_bytes / opt_.align) * static_cast<uint64_t>(opt_.align),
      uint32_aligned);
  uint64_t current = opt_.align;
  const double waste = std::min(opt_.max_waste, 0.999999);
  while (current < aligned_len) {
    const uint64_t minimum = current + opt_.align;
    uint64_t candidate = minimum;
    if (waste > 0) {
      const double raw = static_cast<double>(minimum) / (1.0 - waste);
      const uint64_t bounded = static_cast<uint64_t>(raw);
      candidate = (bounded / opt_.align) * static_cast<uint64_t>(opt_.align);
      if (candidate < minimum) candidate = minimum;
    }
    if (candidate > max_aligned) candidate = max_aligned;
    if (candidate <= current) return static_cast<uint32_t>(aligned_len);
    current = candidate;
  }
  return static_cast<uint32_t>(current);
}

size_t SlabAllocator::ClassForLen(size_t aligned_len) {
  return ClassForExactSize(CanonicalSlotSize(aligned_len));
}

size_t SlabAllocator::ClassForExactSize(uint32_t slot_size) {
  for (size_t i = 0; i < classes_.size(); ++i) {
    if (classes_[i].slot_size == slot_size) return i;
  }
  return classes_.size();
}

uint32_t SlabAllocator::AcquireSlotMetaLocked() {
  if (free_slot_meta_head_ != kNoSlot) {
    const uint32_t index = free_slot_meta_head_;
    free_slot_meta_head_ = slots_[index].free_next;
    slots_[index] = SlotMeta{};
    return index;
  }
  slots_.emplace_back();
  return static_cast<uint32_t>(slots_.size() - 1);
}

void SlabAllocator::LinkBeforeHandLocked(uint32_t index) {
  SlotMeta& meta = slots_[index];
  Class& cls = classes_[meta.ref.cls];

  // Behind the hand in forward sweep order: visit older residents first.
  if (cls.hand == kNoSlot) {
    cls.ring_head = cls.hand = index;
  } else {
    meta.class_next = cls.hand;
    meta.class_prev = slots_[cls.hand].class_prev;
    if (meta.class_prev != kNoSlot)
      slots_[meta.class_prev].class_next = index;
    else
      cls.ring_head = index;
    slots_[cls.hand].class_prev = index;
  }
}

void SlabAllocator::LinkBeforeGlobalHandLocked(uint32_t index) {
  SlotMeta& meta = slots_[index];
  if (global_hand_ == kNoSlot) {
    global_ring_head_ = global_hand_ = index;
  } else {
    meta.global_next = global_hand_;
    meta.global_prev = slots_[global_hand_].global_prev;
    if (meta.global_prev != kNoSlot)
      slots_[meta.global_prev].global_next = index;
    else
      global_ring_head_ = index;
    slots_[global_hand_].global_prev = index;
  }
}

void SlabAllocator::LinkResidentLocked(uint32_t index) {
  SlotMeta& meta = slots_[index];
  Class& cls = classes_[meta.ref.cls];
  ExtentMeta& extent = extents_[meta.ref.extent];

  LinkBeforeHandLocked(index);
  LinkBeforeGlobalHandLocked(index);
  ++cls.resident_count;

  meta.extent_next = extent.resident_head;
  if (extent.resident_head != kNoSlot)
    slots_[extent.resident_head].extent_prev = index;
  extent.resident_head = index;
  if (extent.resident_count++ == 0) {
    --cls.empty_extents;
    --empty_extents_;
  }
}

uint64_t SlabAllocator::DecayedHeat(uint64_t heat, uint64_t heat_seq) const {
  if (heat == 0) return 0;
  const uint64_t elapsed = activity_seq_ >= heat_seq ? activity_seq_ - heat_seq : 0;
  const uint64_t shifts = elapsed / kReadDecayOps;
  return shifts >= 63 ? 0 : heat >> shifts;
}

uint64_t SlabAllocator::ExtentHeatLocked(const ExtentMeta& extent) const {
  return DecayedHeat(extent.read_heat, extent.heat_seq);
}

bool SlabAllocator::StealsBeforeLocked(const ExtentMeta& a,
                                       const ExtentMeta& b) const {
  if ((a.resident_count == 0) != (b.resident_count == 0))
    return a.resident_count == 0;
  // Compare mean access_seq without division: sum_a / n_a < sum_b / n_b.
  const unsigned __int128 lhs =
      static_cast<unsigned __int128>(a.access_seq_sum) * b.resident_count;
  const unsigned __int128 rhs =
      static_cast<unsigned __int128>(b.access_seq_sum) * a.resident_count;
  if (lhs != rhs) return lhs < rhs;
  const uint64_t heat_a = ExtentHeatLocked(a), heat_b = ExtentHeatLocked(b);
  if (heat_a != heat_b) return heat_a < heat_b;
  return a.useful_bytes < b.useful_bytes;
}

// Whole-extent moves between size classes follow the data that is used: an
// extent goes to a class that wants to grow only if its residents are much
// colder than what that class would otherwise evict from its own slots.
//
// "Cold" is the mean time since the residents were last read or written, on
// the allocator's clock, which ticks with every admission (and, where reads
// go through Get/GetAndPin, with every read), so the times of all classes are
// on one scale whatever their slot sizes. The donor's extent must be older
// than kStealAgeMargin times the growing class's own least recently used
// extent. Without a margin two classes of about the same age would hand one
// extent back and forth; with it an extent that has moved comes back only
// after the ages have swung by the margin squared.
//
// The comparison is relative on purpose. A class that is read constantly
// and never written keeps its extents against a class that is merely
// written; a class that is read less than a newer one is used gives way to
// it. A rule with a fixed horizon ("used within the last N ticks is never
// cold") would do the first and forbid the second.
bool SlabAllocator::MuchColderLocked(const ExtentMeta& donor,
                                     const ExtentMeta& own) const {
  if (donor.resident_count == 0) return true;   // nothing to lose
  if (own.resident_count == 0) return true;     // nothing to compare with
  // Sum of the residents' ages; activity_seq_ >= every access_seq.
  const unsigned __int128 donor_age =
      static_cast<unsigned __int128>(activity_seq_) * donor.resident_count -
      donor.access_seq_sum;
  const unsigned __int128 own_age =
      static_cast<unsigned __int128>(activity_seq_) * own.resident_count -
      own.access_seq_sum;
  // mean(donor) > kStealAgeMargin * mean(own), without the divisions.
  return donor_age * own.resident_count >
         kStealAgeMargin * own_age * donor.resident_count;
}

// The extent of `cls` whose residents are least recently used on average:
// what the class would lose first if it had to give or evict. -1 if the
// class holds no resident.
//
// An extent the class cannot lose is not that: one with a pinned slot or
// with a resident under retention is passed over by every steal and its
// protected residents by every eviction, so its age says nothing about what
// the class would give up. Measured against an extent that holds a value
// retained for an hour, no donor would be old enough for an hour, and the
// class would go on evicting its own much younger values. Such extents
// count only when the class has no other.
int SlabAllocator::LeastRecentlyUsedExtentLocked(size_t cls) const {
  int best = -1;
  int best_held = -1;
  for (uint32_t e = 0; e < extents_.size(); ++e) {
    const ExtentMeta& extent = extents_[e];
    if (extent.cls != static_cast<int>(cls) || extent.resident_count == 0)
      continue;
    int& candidate =
        extent.pinned != 0 || extent.protected_slots != 0 ? best_held : best;
    if (candidate < 0 ||
        StealsBeforeLocked(extent, extents_[static_cast<size_t>(candidate)]))
      candidate = static_cast<int>(e);
  }
  return best >= 0 ? best : best_held;
}

void SlabAllocator::TouchLocked(uint32_t index) {
  SlotMeta& meta = slots_[index];
  ++activity_seq_;
  meta.referenced = true;
  ++meta.read_touches;

  Class& cls = classes_[meta.ref.cls];
  // A reference bit alone forgets actual read order after one sweep. Refresh
  // the same position used for new writes, without touching extent membership.
  // The predecessor of the hand is already last in sweep order (singleton
  // included), so avoid unlinking it.
  const bool behind_hand =
      meta.class_next == cls.hand ||
      (meta.class_next == kNoSlot && cls.hand == cls.ring_head);
  if (!behind_hand) {
    const uint32_t next = meta.class_next;
    if (meta.class_prev != kNoSlot)
      slots_[meta.class_prev].class_next = next;
    else
      cls.ring_head = next;
    if (next != kNoSlot)
      slots_[next].class_prev = meta.class_prev;
    if (cls.hand == index)
      cls.hand = next != kNoSlot ? next : cls.ring_head;
    LinkBeforeHandLocked(index);
  }

  // Global reclaim uses the same real-read/new-admission order across sizes.
  const bool behind_global_hand =
      meta.global_next == global_hand_ ||
      (meta.global_next == kNoSlot && global_hand_ == global_ring_head_);
  if (!behind_global_hand) {
    const uint32_t next = meta.global_next;
    if (meta.global_prev != kNoSlot)
      slots_[meta.global_prev].global_next = next;
    else
      global_ring_head_ = next;
    if (next != kNoSlot)
      slots_[next].global_prev = meta.global_prev;
    if (global_hand_ == index)
      global_hand_ = next != kNoSlot ? next : global_ring_head_;
    LinkBeforeGlobalHandLocked(index);
  }

  cls.read_heat = DecayedHeat(cls.read_heat, cls.heat_seq) + kReadHeatUnit;
  cls.heat_seq = activity_seq_;
  ++cls.read_touches;

  ExtentMeta& extent = extents_[meta.ref.extent];
  extent.read_heat = ExtentHeatLocked(extent) + kReadHeatUnit;
  extent.heat_seq = activity_seq_;
  extent.access_seq_sum += activity_seq_ - meta.access_seq;
  meta.access_seq = activity_seq_;
}

SlabAllocator::ExtentBindResult SlabAllocator::BindExtentLocked(
    size_t cls_index, uint32_t extent_index, bool metadata_prepared) {
  if (cls_index >= classes_.size() || extent_index >= extents_.size() ||
      extents_[extent_index].cls != kUnbound)
    return ExtentBindResult::kUnavailable;
  Class& cls = classes_[cls_index];
  if (cls.slots_per_extent == 0) return ExtentBindResult::kUnavailable;
  if (!metadata_prepared && opt_.on_extent_bind &&
      !opt_.on_extent_bind(extent_index))
    return ExtentBindResult::kPersistenceError;

  ExtentMeta& extent = extents_[extent_index];
  extent = ExtentMeta{};
  extent.cls = static_cast<int>(cls_index);
  extent.total_slots = cls.slots_per_extent;
  extent.free_slots = cls.slots_per_extent;
  ++cls.bound_extents;
  ++cls.empty_extents;
  ++empty_extents_;
  --unbound_;
  for (uint32_t slot = 0; slot < cls.slots_per_extent; ++slot)
    PushFreeLocked(cls, extent_index, slot);
  return ExtentBindResult::kBound;
}

SlabAllocator::ExtentBindResult SlabAllocator::BindFreeExtent(size_t cls) {
  if (unbound_ == 0) return ExtentBindResult::kUnavailable;
  for (uint32_t extent = 0; extent < extents_.size(); ++extent) {
    if (extents_[extent].cls == kUnbound)
      return BindExtentLocked(cls, extent, false);
  }
  return ExtentBindResult::kUnavailable;
}


bool SlabAllocator::RestoreBulk(const std::vector<RestoreEntry>& entries) {
  std::lock_guard<std::mutex> lock(mu_);
  if (entries.empty()) return true;

  // Everything this pass keeps is flat and sized by the batch or by the
  // geometry, never one heap node per entry: a restart restores millions.
  // fingerprints: one hash per key, to find a key that occurs twice.
  // batch_slot_size: the slot size this batch gave each extent (0 = none yet).
  // placed: per extent, one bit per slot this batch has already placed.
  std::vector<uint64_t> fingerprints;
  std::vector<uint32_t> batch_slot_size(extents_.size(), 0);
  std::vector<std::vector<uint64_t>> placed(extents_.size());
  fingerprints.reserve(entries.size());

  for (const RestoreEntry& entry : entries) {
    if (entry.slot_size == 0 || entry.extent >= extents_.size() ||
        entry.slot_size > opt_.extent_bytes ||
        entry.value_bytes > entry.slot_size ||
        CanonicalSlotSize(entry.slot_size) != entry.slot_size ||
        ClassForExactSize(entry.slot_size) == classes_.size() ||
        entry.slot >= opt_.extent_bytes / entry.slot_size ||
        index_.Find(entry.key) != nullptr)
      return false;
    fingerprints.push_back(FlatKeyIndex::Hash(entry.key));
    std::vector<uint64_t>& bits = placed[entry.extent];
    if (bits.empty())
      bits.assign(opt_.extent_bytes / entry.slot_size / 64 + 1, 0);
    // A second slot size on the extent is refused just below, so the bitmap
    // sized by the first one fits every slot that gets this far.
    uint32_t& extent_slot_size = batch_slot_size[entry.extent];
    if (extent_slot_size == 0) extent_slot_size = entry.slot_size;
    if (extent_slot_size != entry.slot_size) return false;
    uint64_t& word = bits[entry.slot / 64];
    const uint64_t bit = uint64_t{1} << (entry.slot % 64);
    if ((word & bit) != 0) return false;
    word |= bit;
    const ExtentMeta& extent = extents_[entry.extent];
    if (extent.cls != kUnbound) {
      const Class& cls = classes_[static_cast<size_t>(extent.cls)];
      if (cls.slot_size != entry.slot_size) return false;
      auto free_it = cls.free_by_ext.find(entry.extent);
      if (free_it == cls.free_by_ext.end() ||
          std::find(free_it->second.begin(), free_it->second.end(), entry.slot) ==
              free_it->second.end())
        return false;
    }
  }
  // A key that occurs twice has two equal fingerprints; sorting 8-byte
  // hashes finds that much faster than sorting the 24-byte keys. Two equal
  // fingerprints are almost always that, but they prove nothing by
  // themselves, so only then are the keys themselves sorted and compared.
  std::sort(fingerprints.begin(), fingerprints.end());
  if (std::adjacent_find(fingerprints.begin(), fingerprints.end()) !=
      fingerprints.end()) {
    std::vector<BlockKey> keys;
    keys.reserve(entries.size());
    for (const RestoreEntry& entry : entries) keys.push_back(entry.key);
    std::sort(keys.begin(), keys.end(),
              [](const BlockKey& a, const BlockKey& b) {
                if (a.tenant_hash != b.tenant_hash)
                  return a.tenant_hash < b.tenant_hash;
                if (a.digest_hi != b.digest_hi) return a.digest_hi < b.digest_hi;
                return a.digest_lo < b.digest_lo;
              });
    if (std::adjacent_find(keys.begin(), keys.end()) != keys.end())
      return false;
  }
  fingerprints = {};
  placed = {};

  slots_.reserve(slots_.size() + entries.size());
  index_.Reserve(index_.size() + entries.size());

  // Taking a given slot out of an extent's free list: where each free slot
  // sits in the list is kept beside it, so the take is one swap instead of a
  // search of the list. The list ends up exactly as the search-and-swap left
  // it. `free` stays valid while extents are bound below: free_by_ext only
  // gains keys here, and an unordered_map never moves its values.
  struct ExtentFree {
    std::vector<uint32_t>* free = nullptr;
    std::vector<uint32_t> position;  // slot -> index in *free, or kNoSlot
  };
  std::vector<ExtentFree> extent_free(extents_.size());
  // The index is far larger than the CPU caches and keys land in it at
  // random; asking for a later entry's bucket now has it there when its turn
  // comes.
  constexpr size_t kPrefetchAhead = 16;


  for (size_t at = 0; at < entries.size(); ++at) {
    const RestoreEntry& entry = entries[at];
    if (at + kPrefetchAhead < entries.size())
      index_.Prefetch(entries[at + kPrefetchAhead].key);
    const size_t cls_index = ClassForExactSize(entry.slot_size);
    if (extents_[entry.extent].cls == kUnbound) {
      if (BindExtentLocked(cls_index, entry.extent, true) !=
          ExtentBindResult::kBound)
        return false;
    }
    Class& cls = classes_[cls_index];
    ExtentFree& from = extent_free[entry.extent];
    if (from.free == nullptr) {
      auto it = cls.free_by_ext.find(entry.extent);
      if (it == cls.free_by_ext.end()) return false;
      from.free = &it->second;
      from.position.assign(cls.slots_per_extent, kNoSlot);
      for (size_t i = 0; i < from.free->size(); ++i)
        from.position[(*from.free)[i]] = static_cast<uint32_t>(i);
    }
    if (entry.slot >= from.position.size()) return false;
    const uint32_t free_at = from.position[entry.slot];
    if (free_at == kNoSlot) return false;
    const uint32_t moved = from.free->back();
    (*from.free)[free_at] = moved;
    from.position[moved] = free_at;
    from.position[entry.slot] = kNoSlot;
    from.free->pop_back();
    --cls.free_count;
    if (from.free->empty()) DropFromRotationLocked(cls, entry.extent);

    const uint32_t meta_index = AcquireSlotMetaLocked();
    SlotMeta& meta = slots_[meta_index];
    meta.ref.cls = static_cast<uint32_t>(cls_index);
    meta.ref.extent = entry.extent;
    meta.ref.slot = entry.slot;
    meta.ref.slot_size = entry.slot_size;
    meta.ref.offset = static_cast<uint64_t>(entry.slot) * entry.slot_size;
    meta.useful_bytes = entry.value_bytes;
    meta.key = entry.key;
    meta.generation = ++next_generation_;
    meta.live = true;
    if (!index_.Insert(entry.key, meta_index)) return false;
    LinkResidentLocked(meta_index);

    ExtentMeta& extent = extents_[entry.extent];
    --extent.free_slots;
    extent.useful_bytes += meta.useful_bytes;
    cls.useful_bytes += meta.useful_bytes;
    used_bytes_ += entry.slot_size;

    // A persisted protection still in the future comes back, re-clamped to
    // the current cap (which may have been lowered across the restart). The
    // budget is not re-checked: these bytes were admitted under it before.
    if (retain_.enabled && entry.retain_until > retain_now_s_ &&
        entry.retain_ttl_s != 0) {
      const uint32_t ttl = std::min(entry.retain_ttl_s, retain_.max_ttl_s);
      const uint64_t cap = static_cast<uint64_t>(retain_now_s_) + ttl;
      meta.retain_ttl_s = ttl;
      meta.retain_until = static_cast<uint32_t>(
          std::min<uint64_t>(entry.retain_until, cap));
      meta.retain_persisted_until = meta.retain_until;
      ProtectLocked(meta_index);
      PushWheelLocked(meta_index);
    } else if (retain_.enabled) {
      meta.retain_ttl_s = std::min(entry.retain_ttl_s, retain_.max_ttl_s);
    }
  }
  return true;
}

void SlabAllocator::FreeSlotLocked(uint32_t index) {
  if (slots_[index].retain_until != 0) UnprotectLocked(index);
  SlotMeta& meta = slots_[index];
  const uint32_t cls_index = meta.ref.cls;
  const uint32_t extent_index = meta.ref.extent;
  const uint32_t physical_slot = meta.ref.slot;
  const uint32_t slot_size = meta.ref.slot_size;
  const uint64_t useful_bytes = meta.useful_bytes;
  Class& cls = classes_[cls_index];
  ExtentMeta& extent = extents_[extent_index];

  const uint32_t class_next = meta.class_next;
  if (meta.class_prev != kNoSlot)
    slots_[meta.class_prev].class_next = meta.class_next;
  else
    cls.ring_head = meta.class_next;
  if (meta.class_next != kNoSlot)
    slots_[meta.class_next].class_prev = meta.class_prev;
  --cls.resident_count;
  if (cls.hand == index) {
    cls.hand = class_next != kNoSlot ? class_next : cls.ring_head;
  }
  if (cls.resident_count == 0) cls.hand = kNoSlot;

  const uint32_t global_next = meta.global_next;
  if (meta.global_prev != kNoSlot)
    slots_[meta.global_prev].global_next = global_next;
  else
    global_ring_head_ = global_next;
  if (global_next != kNoSlot)
    slots_[global_next].global_prev = meta.global_prev;
  if (global_hand_ == index)
    global_hand_ = global_next != kNoSlot ? global_next : global_ring_head_;

  if (meta.extent_prev != kNoSlot)
    slots_[meta.extent_prev].extent_next = meta.extent_next;
  else
    extent.resident_head = meta.extent_next;
  if (meta.extent_next != kNoSlot)
    slots_[meta.extent_next].extent_prev = meta.extent_prev;
  if (--extent.resident_count == 0) {
    ++cls.empty_extents;
    ++empty_extents_;
  }
  extent.access_seq_sum -= meta.access_seq;

  if (meta.refs > 0 && extent.pinned > 0) --extent.pinned;
  index_.Erase(meta.key);
  PushFreeLocked(cls, extent_index, physical_slot);
  ++extent.free_slots;
  extent.useful_bytes -= useful_bytes;
  cls.useful_bytes -= useful_bytes;
  used_bytes_ -= slot_size;
  meta = SlotMeta{};
  meta.free_next = free_slot_meta_head_;
  free_slot_meta_head_ = index;

  if (extent.free_slots == extent.total_slots &&
      cls.ext_rr.size() > opt_.stripe_ways) {
    DropExtentFreeLocked(cls, extent_index);
    --cls.bound_extents;
    --cls.empty_extents;
    --empty_extents_;
    extent = ExtentMeta{};
    ++unbound_;
    ++extent_returns_;
  }
}

bool SlabAllocator::EvictOneFrom(size_t cls_index,
                                 std::vector<BlockKey>* evicted,
                                 size_t* cleared) {
  Class& cls = classes_[cls_index];
  if (cls.resident_count == 0) return false;
  const size_t limit = 2 * cls.resident_count + 2;
  // Only cleared bits count against evict_clear_limit: pinned and retained
  // residents are skipped under `limit` alone, so a run of them at the hand
  // cannot hide an evictable resident behind it.
  size_t clears = 0;
  uint32_t fallback = kNoSlot;
  for (size_t steps = 0; steps < limit && cls.resident_count != 0; ++steps) {
    if (cls.hand == kNoSlot) cls.hand = cls.ring_head;
    const uint32_t current = cls.hand;
    SlotMeta& meta = slots_[current];
    cls.hand = meta.class_next != kNoSlot ? meta.class_next : cls.ring_head;
    if (meta.refs > 0) continue;
    if (meta.retain_until != 0) {
      ++retain_evict_skipped_;
      continue;
    }
    if (fallback == kNoSlot) fallback = current;
    if (meta.referenced) {
      meta.referenced = false;
      if (cleared) ++*cleared;
      if (++clears == opt_.evict_clear_limit) break;  // never true for 0
      continue;
    }
    if (opt_.on_slot_evict && !opt_.on_slot_evict(meta.ref)) return false;
    BlockKey victim = meta.key;
    FreeSlotLocked(current);
    if (evicted) evicted->push_back(std::move(victim));
    ++evictions_;
    ++classes_[cls_index].evictions_since_balance;
    return true;
  }
  if (fallback != kNoSlot && slots_[fallback].live) {
    SlotMeta& meta = slots_[fallback];
    if (opt_.on_slot_evict && !opt_.on_slot_evict(meta.ref)) return false;
    BlockKey victim = meta.key;
    FreeSlotLocked(fallback);
    if (evicted) evicted->push_back(std::move(victim));
    ++evictions_;
    ++classes_[cls_index].evictions_since_balance;
    return true;
  }
  return false;
}

SlabAllocator::ExtentBindResult SlabAllocator::StealExtentFor(
    size_t target_cls, std::vector<BlockKey>* evicted,
    size_t min_donor_extents) {
  int best = -1;
  for (uint32_t e = 0; e < extents_.size(); ++e) {
    const ExtentMeta& extent = extents_[e];
    if (extent.cls == kUnbound || extent.cls == static_cast<int>(target_cls) ||
        extent.pinned != 0 || extent.protected_slots != 0)
      continue;
    const Class& donor = classes_[static_cast<size_t>(extent.cls)];
    if (min_donor_extents > 0 && donor.bound_extents <= min_donor_extents)
      continue;
    if (best < 0 ||
        StealsBeforeLocked(extent, extents_[static_cast<size_t>(best)]))
      best = static_cast<int>(e);
  }
  if (best < 0) return ExtentBindResult::kUnavailable;
  return StealExtentLocked(static_cast<uint32_t>(best), target_cls, evicted);
}

// An extent of another class with no resident (hence no pin and no
// protection): moving it loses no value. A class that is being written keeps
// its striping width of extents with free slots (Options::stripe_ways); an
// idle one, without a Put for one cold window, gives every empty extent.
// The counts keep the common case, no candidate at all, off the extents.
SlabAllocator::ExtentBindResult SlabAllocator::TakeEmptyExtentFor(
    size_t target_cls) {
  if (empty_extents_ == classes_[target_cls].empty_extents)
    return ExtentBindResult::kUnavailable;
  const uint64_t window = cold_window_ ? cold_window_ : index_.size();
  int donor = -1;
  for (size_t c = 0; c < classes_.size() && donor < 0; ++c) {
    const Class& cls = classes_[c];
    if (c == target_cls || cls.empty_extents == 0) continue;
    const bool idle = put_seq_ - cls.last_put_seq > window;
    if (idle || cls.ext_rr.size() > opt_.stripe_ways) donor = static_cast<int>(c);
  }
  if (donor < 0) return ExtentBindResult::kUnavailable;
  for (uint32_t e = 0; e < extents_.size(); ++e) {
    const ExtentMeta& extent = extents_[e];
    if (extent.cls == donor && extent.resident_count == 0)
      return StealExtentLocked(e, target_cls, nullptr);
  }
  return ExtentBindResult::kUnavailable;
}

SlabAllocator::ExtentBindResult SlabAllocator::StealColdExtentFor(
    size_t target_cls, std::vector<BlockKey>* evicted) {
  if (!cold_steal_enabled_) return ExtentBindResult::kUnavailable;
  const uint64_t window = cold_window_ ? cold_window_ : index_.size();
  if (window == 0 || put_seq_ <= window) return ExtentBindResult::kUnavailable;
  const uint64_t cold_before = put_seq_ - window;
  // What this class would evict instead: a candidate has to be much colder.
  const int own = LeastRecentlyUsedExtentLocked(target_cls);
  // Of the extents that qualify, the least recently used on average, as the
  // other steals choose. Read heat decays to nothing within about 700
  // operations, so ordering by it (and then by fewest useful bytes) took, at
  // any realistic operation rate, the emptiest qualifying extent however
  // recently it had been read.
  int best = -1;
  for (uint32_t e = 0; e < extents_.size(); ++e) {
    const ExtentMeta& extent = extents_[e];
    if (extent.cls == kUnbound || extent.cls == static_cast<int>(target_cls) ||
        extent.pinned != 0 || extent.protected_slots != 0 ||
        extent.resident_count == 0 || extent.youngest_seq >= cold_before ||
        (own >= 0 &&
         !MuchColderLocked(extent, extents_[static_cast<size_t>(own)])))
      continue;
    if (best < 0 ||
        StealsBeforeLocked(extent, extents_[static_cast<size_t>(best)]))
      best = static_cast<int>(e);
  }
  if (best < 0) return ExtentBindResult::kUnavailable;
  const ExtentBindResult result =
      StealExtentLocked(static_cast<uint32_t>(best), target_cls, evicted);
  if (result == ExtentBindResult::kBound) ++cold_steals_;
  return result;
}

SlabAllocator::ExtentBindResult SlabAllocator::StealExtentLocked(
    uint32_t extent_index, size_t target_cls,
    std::vector<BlockKey>* evicted) {
  ExtentMeta& extent = extents_[extent_index];
  const int old_cls = extent.cls;
  if (old_cls == kUnbound || target_cls >= classes_.size() ||
      classes_[target_cls].slots_per_extent == 0 || extent.pinned != 0 ||
      extent.protected_slots != 0)
    return ExtentBindResult::kUnavailable;

  // The durable extent wipe happens before any resident is forgotten.
  if (opt_.on_extent_bind && !opt_.on_extent_bind(extent_index))
    return ExtentBindResult::kPersistenceError;
  while (extent.resident_head != kNoSlot) {
    const uint32_t index = extent.resident_head;
    BlockKey victim = slots_[index].key;
    FreeSlotLocked(index);
    if (evicted) evicted->push_back(std::move(victim));
    ++evictions_;
  }
  if (extents_[extent_index].cls != kUnbound) {
    Class& old_class = classes_[static_cast<size_t>(old_cls)];
    DropExtentFreeLocked(old_class, extent_index);
    --old_class.bound_extents;
    --old_class.empty_extents;  // its residents were all freed above
    --empty_extents_;
    extents_[extent_index] = ExtentMeta{};
    ++unbound_;
  }
  const ExtentBindResult result =
      BindExtentLocked(target_cls, extent_index, true);
  if (result == ExtentBindResult::kBound) ++steals_;
  return result;
}

bool SlabAllocator::StealFrom(size_t donor_cls, size_t target_cls,
                              std::vector<BlockKey>* evicted) {
  std::lock_guard<std::mutex> lock(mu_);
  if (donor_cls == target_cls || donor_cls >= classes_.size() ||
      target_cls >= classes_.size() || classes_[target_cls].slots_per_extent == 0)
    return false;
  int best = -1;
  for (uint32_t e = 0; e < extents_.size(); ++e) {
    const ExtentMeta& extent = extents_[e];
    if (extent.cls != static_cast<int>(donor_cls) || extent.pinned != 0 ||
        extent.protected_slots != 0)
      continue;
    if (best < 0 ||
        StealsBeforeLocked(extent, extents_[static_cast<size_t>(best)]))
      best = static_cast<int>(e);
  }
  if (best < 0) return false;
  // `best` is the donor's least recently used extent. If even that one is not
  // much colder than what the target class would evict from its own slots,
  // the donor has nothing to give it: the target keeps to its own.
  const int own = LeastRecentlyUsedExtentLocked(target_cls);
  if (own >= 0 && !MuchColderLocked(extents_[static_cast<size_t>(best)],
                                    extents_[static_cast<size_t>(own)]))
    return false;
  return StealExtentLocked(static_cast<uint32_t>(best), target_cls, evicted) ==
         ExtentBindResult::kBound;
}

bool SlabAllocator::Put(const BlockKey& key, size_t len, SlotRef* out,
                        std::vector<BlockKey>* evicted,
                        SlotHandle* out_handle) {
  std::lock_guard<std::mutex> lock(mu_);
  if (out_handle) *out_handle = SlotHandle{};
  ++activity_seq_;
  if (uint32_t* existing = index_.Find(key)) {
    SlotMeta& meta = slots_[*existing];
    if (meta.remove_pending) return false;
    meta.referenced = true;
    if (out) *out = meta.ref;
    if (out_handle) out_handle->index = *existing;
    return true;
  }
  if (len == 0 || len > opt_.extent_bytes || len > UINT32_MAX) return false;
  uint64_t aligned = AlignUp(len, opt_.align);
  if (aligned == 0 || aligned > opt_.extent_bytes) return false;

  const size_t cls_index = ClassForLen(static_cast<size_t>(aligned));
  if (cls_index >= classes_.size()) return false;
  Slot placement{0, 0};
  for (;;) {
    Class& cls = classes_[cls_index];
    while (cls.ext_rr.size() < opt_.stripe_ways) {
      const ExtentBindResult bind = BindFreeExtent(cls_index);
      if (bind == ExtentBindResult::kBound) continue;
      if (bind == ExtentBindResult::kPersistenceError) return false;
      break;
    }
    if (PopFreeLocked(cls, &placement)) break;
    // No free slot of its own and none unbound: first an extent another
    // class holds empty, which costs nobody a value; then (below) values,
    // the oldest of this class unless another class's are much older.
    {
      const ExtentBindResult empty = TakeEmptyExtentFor(cls_index);
      if (empty == ExtentBindResult::kBound) continue;
      if (empty == ExtentBindResult::kPersistenceError) return false;
    }
    if (cls.bound_extents < opt_.stripe_ways) {
      const ExtentBindResult grow =
          StealExtentFor(cls_index, evicted, opt_.stripe_ways);
      if (grow == ExtentBindResult::kBound) continue;
      if (grow == ExtentBindResult::kPersistenceError) return false;
    }
    if (!cls.reclaim_clear_limited) {
      const ExtentBindResult cold = StealColdExtentFor(cls_index, evicted);
      if (cold == ExtentBindResult::kBound) continue;
      if (cold == ExtentBindResult::kPersistenceError) return false;
    }
    if (EvictOneFrom(cls_index, evicted)) continue;
    const ExtentBindResult steal = StealExtentFor(cls_index, evicted);
    if (steal == ExtentBindResult::kBound) continue;
    if (steal == ExtentBindResult::kPersistenceError) return false;
    return false;
  }

  const uint32_t meta_index = AcquireSlotMetaLocked();
  SlotMeta& meta = slots_[meta_index];
  meta.ref.cls = static_cast<uint32_t>(cls_index);
  meta.ref.extent = placement.extent;
  meta.ref.slot = placement.slot;
  meta.ref.slot_size = classes_[cls_index].slot_size;
  meta.ref.offset = static_cast<uint64_t>(placement.slot) * meta.ref.slot_size;
  meta.useful_bytes = len;
  const uint64_t insert_seq = ++put_seq_;
  meta.key = key;
  meta.generation = ++next_generation_;
  meta.live = true;
  if (!index_.Insert(key, meta_index)) {
    meta = SlotMeta{};
    meta.free_next = free_slot_meta_head_;
    free_slot_meta_head_ = meta_index;
    PushFreeLocked(classes_[cls_index], placement.extent, placement.slot);
    return false;
  }
  LinkResidentLocked(meta_index);

  Class& cls = classes_[cls_index];
  ExtentMeta& extent = extents_[placement.extent];
  ++cls.puts;
  cls.last_put_seq = insert_seq;
  cls.useful_bytes += len;
  --extent.free_slots;
  extent.useful_bytes += len;
  extent.youngest_seq = std::max(extent.youngest_seq, insert_seq);
  meta.access_seq = activity_seq_;
  extent.access_seq_sum += meta.access_seq;
  used_bytes_ += meta.ref.slot_size;
  if (out) *out = meta.ref;
  if (out_handle) out_handle->index = meta_index;
  return true;
}

bool SlabAllocator::Get(const BlockKey& key, SlotRef* out) {
  std::lock_guard<std::mutex> lock(mu_);
  const uint32_t* index = index_.Find(key);
  if (index == nullptr || slots_[*index].remove_pending) return false;
  TouchLocked(*index);
  if (out) *out = slots_[*index].ref;
  return true;
}

bool SlabAllocator::GetAndPin(const BlockKey& key, SlotRef* out) {
  std::lock_guard<std::mutex> lock(mu_);
  const uint32_t* found = index_.Find(key);
  if (found == nullptr || slots_[*found].remove_pending) return false;
  const uint32_t index = *found;
  TouchLocked(index);
  SlotMeta& meta = slots_[index];
  if (meta.refs == 0) ++extents_[meta.ref.extent].pinned;
  ++meta.refs;
  if (out) *out = meta.ref;
  return true;
}

uint64_t SlabAllocator::Generation(const BlockKey& key) const {
  std::lock_guard<std::mutex> lock(mu_);
  const uint32_t* index = index_.Find(key);
  return index != nullptr && !slots_[*index].remove_pending
             ? slots_[*index].generation
             : 0;
}

bool SlabAllocator::Contains(const BlockKey& key) const {
  std::lock_guard<std::mutex> lock(mu_);
  const uint32_t* index = index_.Find(key);
  return index != nullptr && !slots_[*index].remove_pending;
}

SlabAllocator::RemoveResult SlabAllocator::Remove(const BlockKey& key) {
  std::lock_guard<std::mutex> lock(mu_);
  const uint32_t* found = index_.Find(key);
  if (found == nullptr) return RemoveResult::kNotFound;
  const uint32_t index = *found;
  SlotMeta& meta = slots_[index];
  if (meta.refs > 0) {
    meta.remove_pending = true;
    meta.referenced = false;
    return RemoveResult::kDeferred;
  }
  FreeSlotLocked(index);
  return RemoveResult::kRemoved;
}

bool SlabAllocator::Pin(const BlockKey& key) {
  std::lock_guard<std::mutex> lock(mu_);
  const uint32_t* index = index_.Find(key);
  if (index == nullptr) return false;
  SlotMeta& meta = slots_[*index];
  if (meta.remove_pending) return false;
  if (meta.refs == 0) ++extents_[meta.ref.extent].pinned;
  ++meta.refs;
  return true;
}

bool SlabAllocator::Unpin(const BlockKey& key) {
  std::lock_guard<std::mutex> lock(mu_);
  const uint32_t* found = index_.Find(key);
  if (found == nullptr) return false;
  const uint32_t index = *found;
  SlotMeta& meta = slots_[index];
  if (meta.refs == 0) return false;
  --meta.refs;
  if (meta.refs == 0) {
    if (extents_[meta.ref.extent].pinned > 0) --extents_[meta.ref.extent].pinned;
    if (meta.remove_pending) FreeSlotLocked(index);
  }
  return true;
}

bool SlabAllocator::Pin(const BlockKey& key, SlotHandle handle) {
  std::lock_guard<std::mutex> lock(mu_);
  if (!handle.valid() || handle.index >= slots_.size()) return false;
  SlotMeta& meta = slots_[handle.index];
  if (!meta.live || !(meta.key == key) || meta.remove_pending) return false;
  if (meta.refs == 0) ++extents_[meta.ref.extent].pinned;
  ++meta.refs;
  return true;
}

bool SlabAllocator::PinForRead(const BlockKey& key, SlotHandle handle) {
  std::lock_guard<std::mutex> lock(mu_);
  if (!handle.valid() || handle.index >= slots_.size()) return false;
  SlotMeta& meta = slots_[handle.index];
  if (!meta.live || !(meta.key == key) || meta.remove_pending) return false;
  if (meta.refs == 0) ++extents_[meta.ref.extent].pinned;
  ++meta.refs;
  // Repeated reads of a hot resident find the bit set and store nothing.
  if (!meta.referenced) meta.referenced = true;
  // The read also counts where whole extents are chosen for another class
  // (MuchColderLocked, StealsBeforeLocked): the resident's access time, and
  // with it its extent's mean. The clock is read, not advanced, so reads
  // between two admissions find the time already there and store nothing
  // either; what is stored goes to the slot record and the extent record
  // that this pin and its unpin write on every read anyway (refs, pinned).
  if (meta.access_seq != activity_seq_) {
    extents_[meta.ref.extent].access_seq_sum +=
        activity_seq_ - meta.access_seq;
    meta.access_seq = activity_seq_;
  }
  return true;
}

bool SlabAllocator::Unpin(const BlockKey& key, SlotHandle handle) {
  std::lock_guard<std::mutex> lock(mu_);
  if (!handle.valid() || handle.index >= slots_.size()) return false;
  SlotMeta& meta = slots_[handle.index];
  if (!meta.live || !(meta.key == key) || meta.refs == 0) return false;
  --meta.refs;
  if (meta.refs == 0) {
    if (extents_[meta.ref.extent].pinned > 0) --extents_[meta.ref.extent].pinned;
    if (meta.remove_pending) FreeSlotLocked(handle.index);
  }
  return true;
}

std::vector<SlabAllocator::ClassStat> SlabAllocator::Classes() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<ClassStat> result;
  result.reserve(classes_.size());
  for (const Class& cls : classes_) {
    ClassStat stat;
    stat.slot_size = cls.slot_size;
    stat.free_slots = cls.free_count;
    stat.resident = cls.resident_count;
    stat.puts = cls.puts;
    stat.extents = cls.bound_extents;
    stat.slots_per_extent = cls.slots_per_extent;
    stat.useful_bytes = cls.useful_bytes;
    stat.allocated_bytes = static_cast<uint64_t>(cls.resident_count) * cls.slot_size;
    stat.read_touches = cls.read_touches;
    stat.read_heat = DecayedHeat(cls.read_heat, cls.heat_seq);
    stat.tail_age = TailAgeLocked(cls);
    result.push_back(stat);
  }
  return result;
}

// The class ring is in admission and actual-read order and the hand stands at
// its oldest member: the resident the class evicts next (or near it: pinned
// and protected residents are passed over).
uint64_t SlabAllocator::TailAgeLocked(const Class& cls) const {
  if (cls.hand == kNoSlot || cls.resident_count == 0) return 0;
  const uint64_t seq = slots_[cls.hand].access_seq;
  return activity_seq_ > seq ? activity_seq_ - seq : 0;
}

size_t SlabAllocator::BalanceTailAges(uint32_t ratio_pct, size_t max_moves,
                                      std::vector<BlockKey>* evicted,
                                      bool* ok) {
  std::lock_guard<std::mutex> lock(mu_);
  if (ok) *ok = true;
  const uint64_t window = cold_window_ ? cold_window_ : index_.size();
  size_t evicting = 0;
  for (const Class& cls : classes_)
    if (cls.evictions_since_balance > 0) ++evicting;
  size_t moved = 0;
  while (moved < max_moves && evicting > 0) {
    // The receiver: of the classes evicting for their own writes, the one
    // whose next victim is youngest.
    size_t young = classes_.size();
    uint64_t young_age = 0;
    for (size_t c = 0; c < classes_.size(); ++c) {
      const Class& cls = classes_[c];
      if (cls.evictions_since_balance == 0 || cls.slots_per_extent == 0)
        continue;
      const uint64_t age = TailAgeLocked(cls);
      if (young == classes_.size() || age < young_age) {
        young = c;
        young_age = age;
      }
    }
    if (young == classes_.size()) break;
    // The donor: of the other classes with residents, the one whose next
    // victim is oldest, if it is older by the ratio and above its floor.
    const uint32_t bound = static_cast<uint32_t>(extents_.size()) - unbound_;
    const size_t floor = std::min<size_t>(
        opt_.stripe_ways,
        std::max<size_t>(1, bound / (4 * std::max<size_t>(evicting, 1))));
    size_t old = classes_.size();
    uint64_t old_age = 0;
    for (size_t c = 0; c < classes_.size(); ++c) {
      const Class& cls = classes_[c];
      if (c == young || cls.resident_count == 0 || cls.bound_extents <= floor)
        continue;
      const uint64_t age = TailAgeLocked(cls);
      if (old == classes_.size() || age > old_age) {
        old = c;
        old_age = age;
      }
    }
    if (old == classes_.size() ||
        static_cast<unsigned __int128>(old_age) * 100 <=
            static_cast<unsigned __int128>(young_age) * ratio_pct)
      break;
    // The donor's least recently used extent that may go.
    int best = -1;
    for (uint32_t e = 0; e < extents_.size(); ++e) {
      const ExtentMeta& extent = extents_[e];
      if (extent.cls != static_cast<int>(old) || extent.pinned != 0 ||
          extent.protected_slots != 0 ||
          (extent.balanced_at != 0 && put_seq_ - extent.balanced_at < window))
        continue;
      if (best < 0 ||
          StealsBeforeLocked(extent, extents_[static_cast<size_t>(best)]))
        best = static_cast<int>(e);
    }
    if (best < 0) break;
    const ExtentBindResult result =
        StealExtentLocked(static_cast<uint32_t>(best), young, evicted);
    if (result == ExtentBindResult::kPersistenceError) {
      if (ok) *ok = false;
      break;
    }
    if (result != ExtentBindResult::kBound) break;
    extents_[static_cast<size_t>(best)].balanced_at = std::max<uint64_t>(put_seq_, 1);
    ++balance_moves_;
    ++moved;
  }
  for (Class& cls : classes_) cls.evictions_since_balance = 0;
  return moved;
}

uint64_t SlabAllocator::BalanceMoves() const {
  std::lock_guard<std::mutex> lock(mu_);
  return balance_moves_;
}

size_t SlabAllocator::ColdestGrowDonor(const std::vector<ClassStat>& classes,
                                       const std::vector<uint64_t>& delta,
                                       const std::vector<uint32_t>& extents,
                                       size_t hot, size_t floor) {
  const uint64_t hot_demand = delta[hot] * classes[hot].slot_size;
  size_t donor = classes.size();
  uint64_t donor_demand = 0;
  for (size_t d = 0; d < classes.size(); ++d) {
    if (d == hot || extents[d] == 0) continue;
    const uint64_t demand = delta[d] * classes[d].slot_size;
    if (demand != 0 && extents[d] <= floor) continue;  // keeps its stripes
    if (demand != 0 && demand * 4 > hot_demand) continue;  // not cold enough
    if (donor == classes.size() ||
        std::make_pair(demand, ~uint64_t(extents[d])) <
            std::make_pair(donor_demand, ~uint64_t(extents[donor]))) {
      donor = d;
      donor_demand = demand;
    }
  }
  return donor;
}

size_t SlabAllocator::FreeSlotTarget(uint64_t puts, uint32_t slot_size,
                                     size_t capacity) {
  if (slot_size == 0) return 0;
  const uint64_t floor = std::clamp<uint64_t>(
      (kFreeTargetFloorBytes + slot_size - 1) / slot_size, 2, 64);
  const uint64_t cap = std::max<uint64_t>(floor, kFreeTargetCapBytes / slot_size);
  const uint64_t want = std::min(std::max(2 * puts, floor), cap);
  return static_cast<size_t>(std::min<uint64_t>(want, capacity / 4));
}

size_t SlabAllocator::ReclaimClass(size_t cls_index, size_t target_free,
                                   size_t max_victims,
                                   std::vector<BlockKey>* evicted,
                                   bool* clear_limited) {
  std::lock_guard<std::mutex> lock(mu_);
  if (clear_limited) *clear_limited = false;
  if (cls_index >= classes_.size()) return 0;
  Class& cls = classes_[cls_index];
  cls.reclaim_clear_limited = false;
  size_t count = 0;
  size_t cleared = 0;
  while (count < max_victims && cls.free_count < target_free) {
    if (unbound_ > 0) break;
    const size_t before = cls.free_count;
    if (!EvictOneFrom(cls_index, evicted, &cleared)) break;
    ++count;
    if (cls.free_count <= before) break;
    if (opt_.evict_clear_limit != 0 && cleared >= opt_.evict_clear_limit) {
      cls.reclaim_clear_limited = cls.free_count < target_free;
      if (clear_limited) *clear_limited = cls.reclaim_clear_limited;
      break;
    }
  }
  return count;
}

size_t SlabAllocator::EvictColdToTarget(uint64_t target_bytes,
                                        size_t max_victims,
                                        std::vector<BlockKey>* evicted) {
  std::lock_guard<std::mutex> lock(mu_);
  const size_t limit =
      max_victims > std::numeric_limits<size_t>::max() / 4
          ? std::numeric_limits<size_t>::max()
          : 4 * max_victims;
  size_t freed = 0;
  for (size_t steps = 0;
       steps < limit && freed < max_victims && used_bytes_ > target_bytes &&
       global_hand_ != kNoSlot;
       ++steps) {
    ++watermark_scan_steps_;
    const uint32_t index = global_hand_;
    SlotMeta& meta = slots_[index];
    const uint32_t next =
        meta.global_next != kNoSlot ? meta.global_next : global_ring_head_;
    if (meta.refs > 0) {
      global_hand_ = next;
      continue;
    }
    if (meta.retain_until != 0) {
      ++retain_evict_skipped_;
      global_hand_ = next;
      continue;
    }
    // Actual GETs and new admissions already refresh global sweep order.
    // Leave reference credit to class-local CLOCK; it must not postpone an
    // older resident past more recent reads or writes here.

    const ExtentMeta& extent = extents_[meta.ref.extent];
    if (extent.resident_count == 1 && opt_.on_extent_evict) {
      if (!opt_.on_extent_evict(meta.ref.extent, extent.total_slots, 1))
        break;
    } else if (opt_.on_slot_evict) {
      if (!opt_.on_slot_evict(meta.ref)) break;
    } else if (opt_.on_extent_evict) {
      // A whole-extent callback cannot invalidate only this resident safely.
      break;
    }
    const BlockKey victim = meta.key;
    FreeSlotLocked(index);
    if (evicted) evicted->push_back(victim);
    ++evictions_;
    ++watermark_evictions_;
    ++freed;
  }
  return freed;
}

size_t SlabAllocator::Count() const {
  std::lock_guard<std::mutex> lock(mu_);
  return index_.size();
}
uint64_t SlabAllocator::UsedBytes() const {
  std::lock_guard<std::mutex> lock(mu_);
  return used_bytes_;
}
uint64_t SlabAllocator::Capacity() const {
  return static_cast<uint64_t>(opt_.num_extents) * opt_.extent_bytes;
}
uint64_t SlabAllocator::Evictions() const {
  std::lock_guard<std::mutex> lock(mu_);
  return evictions_;
}
uint64_t SlabAllocator::Steals() const {
  std::lock_guard<std::mutex> lock(mu_);
  return steals_;
}
uint64_t SlabAllocator::ColdSteals() const {
  std::lock_guard<std::mutex> lock(mu_);
  return cold_steals_;
}
uint64_t SlabAllocator::WatermarkEvictions() const {
  std::lock_guard<std::mutex> lock(mu_);
  return watermark_evictions_;
}
uint64_t SlabAllocator::WatermarkScanVisits() const {
  std::lock_guard<std::mutex> lock(mu_);
  return watermark_scan_steps_;
}
uint64_t SlabAllocator::ExtentReturns() const {
  std::lock_guard<std::mutex> lock(mu_);
  return extent_returns_;
}
size_t SlabAllocator::ClassCount() const {
  std::lock_guard<std::mutex> lock(mu_);
  return classes_.size();
}
uint32_t SlabAllocator::PoolExtents() const {
  std::lock_guard<std::mutex> lock(mu_);
  return unbound_;
}
uint32_t SlabAllocator::BoundExtents() const {
  std::lock_guard<std::mutex> lock(mu_);
  uint32_t count = 0;
  for (const ExtentMeta& extent : extents_)
    if (extent.cls != kUnbound) ++count;
  return count;
}

SlabAllocator::IndexStats SlabAllocator::IndexStatsForTest() const {
  std::lock_guard<std::mutex> lock(mu_);
  return index_.stats();
}

size_t SlabAllocator::IndexBucketForTest(const BlockKey& key) const {
  std::lock_guard<std::mutex> lock(mu_);
  return index_.BucketForTest(key);
}

// ---------------------------------------------------------------- retention

void SlabAllocator::ConfigureRetention(const RetainOptions& options,
                                       uint32_t now_s) {
  std::lock_guard<std::mutex> lock(mu_);
  retain_ = options;
  if (retain_.max_ttl_s == 0) retain_.enabled = false;
  if (retain_.class_pct > 100) retain_.class_pct = 100;
  retain_now_s_ = now_s;
  // Every deadline is <= now + max_ttl, so a ring of max_ttl/10 + 2 buckets
  // never holds two laps in one bucket.
  wheel_.assign(retain_.enabled ? retain_.max_ttl_s / kWheelTickS + 2 : 0, {});
  wheel_entries_ = 0;
  wheel_cursor_ = wheel_keep_ = 0;
  wheel_tick_ = now_s / kWheelTickS == 0 ? 0 : now_s / kWheelTickS - 1;
}

uint32_t SlabAllocator::ClassProtectCapLocked(const Class& cls) const {
  const uint64_t slots =
      static_cast<uint64_t>(cls.bound_extents) * cls.slots_per_extent;
  return static_cast<uint32_t>(std::min<uint64_t>(
      slots * retain_.class_pct / 100, std::numeric_limits<uint32_t>::max()));
}

void SlabAllocator::ProtectLocked(uint32_t meta_index) {
  SlotMeta& meta = slots_[meta_index];
  ++extents_[meta.ref.extent].protected_slots;
  ++classes_[meta.ref.cls].protected_slots;
  protected_bytes_ += meta.ref.slot_size;
  ++protected_slots_;
}

void SlabAllocator::UnprotectLocked(uint32_t meta_index) {
  SlotMeta& meta = slots_[meta_index];
  ExtentMeta& extent = extents_[meta.ref.extent];
  Class& cls = classes_[meta.ref.cls];
  if (extent.protected_slots > 0) --extent.protected_slots;
  if (cls.protected_slots > 0) --cls.protected_slots;
  protected_bytes_ -= std::min<uint64_t>(protected_bytes_, meta.ref.slot_size);
  if (protected_slots_ > 0) --protected_slots_;
  meta.retain_until = 0;
}

// A protected resident has exactly one entry on the wheel: in the bucket of
// the deadline it had when the entry was pushed, which the slot remembers
// (retain_wheel_until). A later deadline (refresh, raised TTL) does not push
// another one; the expiry finds the entry, sees that the deadline has moved
// on, and pushes it forward then. An entry whose `until` is not the one its
// slot remembers belongs to a protection that is gone (the key was removed,
// perhaps put and protected again) and is dropped when its bucket comes up.
void SlabAllocator::PushWheelLocked(uint32_t meta_index) {
  if (wheel_.empty()) return;
  SlotMeta& meta = slots_[meta_index];
  meta.retain_wheel_until = meta.retain_until;
  wheel_[(meta.retain_until / kWheelTickS) % wheel_.size()].push_back(
      {meta.key, meta.retain_until});
  ++wheel_entries_;
}

SlabAllocator::RetainOutcome SlabAllocator::Retain(const BlockKey& key,
                                                   uint32_t ttl_s,
                                                   RetainMode mode,
                                                   uint32_t now_s) {
  std::lock_guard<std::mutex> lock(mu_);
  RetainOutcome out;
  // A stepped-back realtime clock must not shorten anything already granted.
  if (now_s > retain_now_s_) retain_now_s_ = now_s;
  const uint32_t now = retain_now_s_;
  const uint32_t* found = index_.Find(key);
  if (found == nullptr || slots_[*found].remove_pending || !retain_.enabled) {
    out.code = found == nullptr || slots_[*found].remove_pending
                   ? RetainCode::kMiss
                   : RetainCode::kHitUnprotected;
    ++retain_results_[static_cast<size_t>(out.code)];
    return out;
  }
  const uint32_t index = *found;
  // A REFRESH is an engine-reported hit served from a higher tier: it counts
  // as an actual read for CLOCK and extent recency. SET does not.
  if (mode == RetainMode::kRefresh) TouchLocked(index);
  SlotMeta& meta = slots_[index];
  out.ref = meta.ref;
  const uint32_t stored = std::min(meta.retain_ttl_s, retain_.max_ttl_s);
  const uint32_t requested = std::min(ttl_s, retain_.max_ttl_s);
  const uint32_t eff =
      mode == RetainMode::kRefresh ? stored : std::max(stored, requested);
  if (eff == 0) {
    out.code = RetainCode::kHitUnprotected;
    ++retain_results_[static_cast<size_t>(out.code)];
    return out;
  }
  const bool was_protected = meta.retain_until != 0;
  if (!was_protected) {
    const Class& cls = classes_[meta.ref.cls];
    if (protected_bytes_ + meta.ref.slot_size > retain_.budget_bytes ||
        cls.protected_slots + 1 > ClassProtectCapLocked(cls)) {
      out.code = RetainCode::kDenied;
      out.ttl_s = meta.retain_ttl_s;
      ++retain_results_[static_cast<size_t>(out.code)];
      return out;
    }
  }
  const uint64_t candidate = static_cast<uint64_t>(now) + eff;
  const uint32_t new_until = static_cast<uint32_t>(std::max<uint64_t>(
      meta.retain_until, std::min<uint64_t>(candidate, UINT32_MAX)));
  if (!was_protected) {
    out.code = RetainCode::kProtectedNew;
  } else if (eff > stored) {
    out.code = RetainCode::kUpgraded;
  } else if (new_until > meta.retain_until) {
    out.code = RetainCode::kRefreshed;
  } else {
    out.code = RetainCode::kUnchanged;
  }
  if (!was_protected) ProtectLocked(index);
  meta.retain_ttl_s = eff;
  meta.retain_until = new_until;
  // Only a new protection goes on the wheel. Pushing an entry for every
  // deadline that moved left one behind per refresh, each to be looked at
  // when its bucket expired: an object renewed once a second for an hour had
  // thousands.
  if (!was_protected) PushWheelLocked(index);
  // Durable record: always for a new or raised TTL; for renewals only once
  // the deadline has moved far enough that losing it on restart matters.
  const uint32_t delta_floor = std::max(retain_.persist_min_delta_s, eff / 4);
  if (out.code == RetainCode::kProtectedNew ||
      out.code == RetainCode::kUpgraded ||
      meta.retain_until >= meta.retain_persisted_until + delta_floor) {
    out.persist = true;
    meta.retain_persisted_until = meta.retain_until;
  }
  out.until = meta.retain_until;
  out.ttl_s = meta.retain_ttl_s;
  ++retain_results_[static_cast<size_t>(out.code)];
  return out;
}

size_t SlabAllocator::ExpireRetention(uint32_t now_s, size_t max_entries,
                                      bool* more) {
  std::lock_guard<std::mutex> lock(mu_);
  if (more) *more = false;
  if (now_s > retain_now_s_) retain_now_s_ = now_s;
  if (wheel_.empty()) return 0;
  const uint32_t now = retain_now_s_;
  // A bucket is drained only once its whole 10 s window lies in the past, so
  // protection may outlive its deadline by < 10 s, never end early.
  const uint64_t target = now / kWheelTickS == 0 ? 0 : now / kWheelTickS - 1;
  if (target <= wheel_tick_) return 0;
  // After a long stall one full lap visits every bucket; skip the rest. Not
  // in the middle of a bucket: that one is finished first.
  if (wheel_cursor_ == 0 && target - wheel_tick_ > wheel_.size())
    wheel_tick_ = target - wheel_.size();
  size_t cleared = 0;
  size_t visited = 0;
  // wheel_tick_ is the last bucket that is done. The one after it may be
  // half done from an earlier call: entries before wheel_keep_ stay, those
  // from wheel_cursor_ on are still to be looked at.
  while (wheel_tick_ < target) {
    const uint64_t tick = wheel_tick_ + 1;
    std::vector<WheelEntry>& bucket = wheel_[tick % wheel_.size()];
    while (wheel_cursor_ < bucket.size()) {
      if (visited == max_entries) {
        // Enough for one hold of the lock; the caller comes back.
        if (more) *more = true;
        return cleared;
      }
      ++visited;
      const WheelEntry entry = bucket[wheel_cursor_++];
      if (entry.until > now) {
        bucket[wheel_keep_++] = entry;  // a later lap (only after a clock step)
        continue;
      }
      --wheel_entries_;
      const uint32_t* found = index_.Find(entry.key);
      if (found == nullptr) continue;
      SlotMeta& meta = slots_[*found];
      // Not this slot's entry: the protection it was pushed for is gone.
      if (!meta.live || meta.retain_until == 0 ||
          meta.retain_wheel_until != entry.until)
        continue;
      const uint64_t due_tick = meta.retain_until / kWheelTickS;
      if (due_tick > tick) {
        // Renewed since: the deadline lies in a bucket that is not due yet.
        // The entry moves there; the resident stays protected. If that is
        // this very bucket a lap later, it stays where it is (appended, it
        // would be looked at again in this pass).
        if (due_tick % wheel_.size() == tick % wheel_.size()) {
          meta.retain_wheel_until = meta.retain_until;
          bucket[wheel_keep_++] = {entry.key, meta.retain_until};
          ++wheel_entries_;
        } else {
          PushWheelLocked(*found);
        }
        continue;
      }
      UnprotectLocked(*found);
      // Rejoin ordinary eviction without leftover reference credit.
      meta.referenced = false;
      ++retain_expired_;
      ++cleared;
    }
    bucket.resize(wheel_keep_);
    wheel_cursor_ = wheel_keep_ = 0;
    wheel_tick_ = tick;
  }
  return cleared;
}

uint32_t SlabAllocator::RetentionHeadroomPct() const {
  std::lock_guard<std::mutex> lock(mu_);
  if (!retain_.enabled || retain_.budget_bytes == 0) return 0;
  const uint64_t used = std::min(protected_bytes_, retain_.budget_bytes);
  return static_cast<uint32_t>((retain_.budget_bytes - used) * 100 /
                               retain_.budget_bytes);
}

SlabAllocator::RetentionStats SlabAllocator::GetRetentionStats() const {
  std::lock_guard<std::mutex> lock(mu_);
  RetentionStats st;
  st.enabled = retain_.enabled;
  st.budget_bytes = retain_.budget_bytes;
  st.protected_bytes = protected_bytes_;
  st.protected_slots = protected_slots_;
  for (const ExtentMeta& extent : extents_)
    if (extent.protected_slots != 0) ++st.extents_pinned;
  st.expired = retain_expired_;
  st.evict_skipped = retain_evict_skipped_;
  st.wheel_entries = wheel_entries_;
  st.max_ttl_s = retain_.max_ttl_s;
  if (retain_.enabled && retain_.budget_bytes != 0) {
    const uint64_t used = std::min(protected_bytes_, retain_.budget_bytes);
    st.headroom_pct = static_cast<uint32_t>(
        (retain_.budget_bytes - used) * 100 / retain_.budget_bytes);
  }
  for (size_t i = 0; i < 7; ++i) st.results[i] = retain_results_[i];
  return st;
}

}  // namespace dfkv
