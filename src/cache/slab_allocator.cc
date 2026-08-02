#include "cache/slab_allocator.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <unordered_map>
#include <unordered_set>
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

  if (const char* w = std::getenv("DFKV_SLAB_COLD_STEAL_WINDOW")) {
    char* end = nullptr;
    const unsigned long long v = std::strtoull(w, &end, 10);
    if (end != w) {
      if (v == 0) cold_steal_enabled_ = false;
      else cold_window_ = static_cast<uint64_t>(v);
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

bool SlabAllocator::TakeFreeSlotLocked(Class& cls, uint32_t extent,
                                       uint32_t slot) {
  auto it = cls.free_by_ext.find(extent);
  if (it == cls.free_by_ext.end()) return false;
  std::vector<uint32_t>& slots = it->second;
  auto found = std::find(slots.begin(), slots.end(), slot);
  if (found == slots.end()) return false;
  *found = slots.back();
  slots.pop_back();
  --cls.free_count;
  if (slots.empty()) {
    DropFromRotationLocked(cls, extent);
  }
  return true;
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

void SlabAllocator::LinkResidentLocked(uint32_t index) {
  SlotMeta& meta = slots_[index];
  Class& cls = classes_[meta.ref.cls];
  ExtentMeta& extent = extents_[meta.ref.extent];

  meta.class_next = cls.ring_head;
  if (cls.ring_head != kNoSlot) slots_[cls.ring_head].class_prev = index;
  cls.ring_head = index;
  if (cls.hand == kNoSlot) cls.hand = index;
  ++cls.resident_count;

  meta.extent_next = extent.resident_head;
  if (extent.resident_head != kNoSlot)
    slots_[extent.resident_head].extent_prev = index;
  extent.resident_head = index;
  ++extent.resident_count;
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

void SlabAllocator::TouchLocked(uint32_t index) {
  SlotMeta& meta = slots_[index];
  ++activity_seq_;
  meta.referenced = true;
  ++meta.read_touches;

  Class& cls = classes_[meta.ref.cls];
  cls.read_heat = DecayedHeat(cls.read_heat, cls.heat_seq) + kReadHeatUnit;
  cls.heat_seq = activity_seq_;
  ++cls.read_touches;

  ExtentMeta& extent = extents_[meta.ref.extent];
  extent.read_heat = ExtentHeatLocked(extent) + kReadHeatUnit;
  extent.heat_seq = activity_seq_;
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

  std::vector<BlockKey> keys;
  std::unordered_set<uint64_t> placements;
  std::unordered_map<uint32_t, uint32_t> batch_extent_sizes;
  keys.reserve(entries.size());
  placements.reserve(entries.size());
  batch_extent_sizes.reserve(entries.size());

  for (const RestoreEntry& entry : entries) {
    if (entry.slot_size == 0 || entry.extent >= extents_.size() ||
        entry.slot_size > opt_.extent_bytes ||
        entry.value_bytes > entry.slot_size ||
        CanonicalSlotSize(entry.slot_size) != entry.slot_size ||
        ClassForExactSize(entry.slot_size) == classes_.size() ||
        entry.slot >= opt_.extent_bytes / entry.slot_size ||
        index_.Find(entry.key) != nullptr)
      return false;
    keys.push_back(entry.key);
    const uint64_t placement =
        (static_cast<uint64_t>(entry.extent) << 32) | entry.slot;
    if (!placements.emplace(placement).second) return false;

    auto batch = batch_extent_sizes.emplace(entry.extent, entry.slot_size);
    if (!batch.second && batch.first->second != entry.slot_size) return false;
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
  std::sort(keys.begin(), keys.end(), [](const BlockKey& a, const BlockKey& b) {
    if (a.tenant_hash != b.tenant_hash) return a.tenant_hash < b.tenant_hash;
    if (a.digest_hi != b.digest_hi) return a.digest_hi < b.digest_hi;
    return a.digest_lo < b.digest_lo;
  });
  if (std::adjacent_find(keys.begin(), keys.end()) != keys.end()) return false;

  slots_.reserve(slots_.size() + entries.size());
  index_.Reserve(index_.size() + entries.size());


  for (const RestoreEntry& entry : entries) {
    const size_t cls_index = ClassForExactSize(entry.slot_size);
    if (extents_[entry.extent].cls == kUnbound) {
      if (BindExtentLocked(cls_index, entry.extent, true) !=
          ExtentBindResult::kBound)
        return false;
    }
    Class& cls = classes_[cls_index];
    if (!TakeFreeSlotLocked(cls, entry.extent, entry.slot)) return false;

    const uint32_t meta_index = AcquireSlotMetaLocked();
    SlotMeta& meta = slots_[meta_index];
    meta.ref.cls = static_cast<uint32_t>(cls_index);
    meta.ref.extent = entry.extent;
    meta.ref.slot = entry.slot;
    meta.ref.slot_size = entry.slot_size;
    meta.ref.offset = static_cast<uint64_t>(entry.slot) * entry.slot_size;
    meta.useful_bytes = entry.value_bytes;
    meta.key = entry.key;
    meta.live = true;
    if (!index_.Insert(entry.key, meta_index)) return false;
    LinkResidentLocked(meta_index);

    ExtentMeta& extent = extents_[entry.extent];
    --extent.free_slots;
    extent.useful_bytes += meta.useful_bytes;
    cls.useful_bytes += meta.useful_bytes;
    used_bytes_ += entry.slot_size;
  }
  return true;
}

void SlabAllocator::FreeSlotLocked(uint32_t index) {
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

  if (meta.extent_prev != kNoSlot)
    slots_[meta.extent_prev].extent_next = meta.extent_next;
  else
    extent.resident_head = meta.extent_next;
  if (meta.extent_next != kNoSlot)
    slots_[meta.extent_next].extent_prev = meta.extent_prev;
  --extent.resident_count;

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
      cls.ext_rr.size() > kStripeWays) {
    DropExtentFreeLocked(cls, extent_index);
    --cls.bound_extents;
    extent = ExtentMeta{};
    ++unbound_;
    ++extent_returns_;
  }
}

bool SlabAllocator::EvictOneFrom(size_t cls_index,
                                 std::vector<BlockKey>* evicted) {
  Class& cls = classes_[cls_index];
  if (cls.resident_count == 0) return false;
  const size_t limit = 2 * cls.resident_count + 2;
  uint32_t fallback = kNoSlot;
  for (size_t steps = 0; steps < limit && cls.resident_count != 0; ++steps) {
    if (cls.hand == kNoSlot) cls.hand = cls.ring_head;
    const uint32_t current = cls.hand;
    SlotMeta& meta = slots_[current];
    cls.hand = meta.class_next != kNoSlot ? meta.class_next : cls.ring_head;
    if (meta.refs > 0) continue;
    if (fallback == kNoSlot) fallback = current;
    if (meta.referenced) {
      meta.referenced = false;
      continue;
    }
    if (opt_.on_slot_evict && !opt_.on_slot_evict(meta.ref)) return false;
    BlockKey victim = meta.key;
    FreeSlotLocked(current);
    if (evicted) evicted->push_back(std::move(victim));
    ++evictions_;
    return true;
  }
  if (fallback != kNoSlot && slots_[fallback].live) {
    SlotMeta& meta = slots_[fallback];
    if (opt_.on_slot_evict && !opt_.on_slot_evict(meta.ref)) return false;
    BlockKey victim = meta.key;
    FreeSlotLocked(fallback);
    if (evicted) evicted->push_back(std::move(victim));
    ++evictions_;
    return true;
  }
  return false;
}

SlabAllocator::ExtentBindResult SlabAllocator::StealExtentFor(
    size_t target_cls, std::vector<BlockKey>* evicted,
    size_t min_donor_extents) {
  int best = -1;
  uint64_t best_heat = std::numeric_limits<uint64_t>::max();
  uint64_t best_useful = std::numeric_limits<uint64_t>::max();
  uint32_t best_free = 0;
  for (uint32_t e = 0; e < extents_.size(); ++e) {
    const ExtentMeta& extent = extents_[e];
    if (extent.cls == kUnbound || extent.cls == static_cast<int>(target_cls) ||
        extent.pinned != 0)
      continue;
    const Class& donor = classes_[static_cast<size_t>(extent.cls)];
    if (min_donor_extents > 0 && donor.bound_extents <= min_donor_extents)
      continue;
    const uint64_t heat = ExtentHeatLocked(extent);
    if (best < 0 || heat < best_heat ||
        (heat == best_heat && extent.useful_bytes < best_useful) ||
        (heat == best_heat && extent.useful_bytes == best_useful &&
         extent.free_slots > best_free)) {
      best = static_cast<int>(e);
      best_heat = heat;
      best_useful = extent.useful_bytes;
      best_free = extent.free_slots;
    }
  }
  if (best < 0) return ExtentBindResult::kUnavailable;
  return StealExtentLocked(static_cast<uint32_t>(best), target_cls, evicted);
}

SlabAllocator::ExtentBindResult SlabAllocator::StealColdExtentFor(
    size_t target_cls, std::vector<BlockKey>* evicted) {
  if (!cold_steal_enabled_) return ExtentBindResult::kUnavailable;
  const uint64_t window = cold_window_ ? cold_window_ : index_.size();
  if (window == 0 || put_seq_ <= window) return ExtentBindResult::kUnavailable;
  const uint64_t cold_before = put_seq_ - window;
  int best = -1;
  uint64_t best_heat = std::numeric_limits<uint64_t>::max();
  uint64_t best_useful = std::numeric_limits<uint64_t>::max();
  uint64_t best_seq = std::numeric_limits<uint64_t>::max();
  for (uint32_t e = 0; e < extents_.size(); ++e) {
    const ExtentMeta& extent = extents_[e];
    if (extent.cls == kUnbound || extent.cls == static_cast<int>(target_cls) ||
        extent.pinned != 0 || extent.resident_count == 0 ||
        extent.youngest_seq >= cold_before)
      continue;
    const uint64_t heat = ExtentHeatLocked(extent);
    if (best < 0 || heat < best_heat ||
        (heat == best_heat && extent.useful_bytes < best_useful) ||
        (heat == best_heat && extent.useful_bytes == best_useful &&
         extent.youngest_seq < best_seq)) {
      best = static_cast<int>(e);
      best_heat = heat;
      best_useful = extent.useful_bytes;
      best_seq = extent.youngest_seq;
    }
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
      classes_[target_cls].slots_per_extent == 0 || extent.pinned != 0)
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
    DropExtentFreeLocked(classes_[static_cast<size_t>(old_cls)], extent_index);
    --classes_[static_cast<size_t>(old_cls)].bound_extents;
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
  uint64_t best_heat = std::numeric_limits<uint64_t>::max();
  uint64_t best_useful = std::numeric_limits<uint64_t>::max();
  uint32_t best_free = 0;
  for (uint32_t e = 0; e < extents_.size(); ++e) {
    const ExtentMeta& extent = extents_[e];
    if (extent.cls != static_cast<int>(donor_cls) || extent.pinned != 0) continue;
    const uint64_t heat = ExtentHeatLocked(extent);
    if (best < 0 || heat < best_heat ||
        (heat == best_heat && extent.useful_bytes < best_useful) ||
        (heat == best_heat && extent.useful_bytes == best_useful &&
         extent.free_slots > best_free)) {
      best = static_cast<int>(e);
      best_heat = heat;
      best_useful = extent.useful_bytes;
      best_free = extent.free_slots;
    }
  }
  return best >= 0 &&
         StealExtentLocked(static_cast<uint32_t>(best), target_cls, evicted) ==
             ExtentBindResult::kBound;
}

bool SlabAllocator::Put(const BlockKey& key, size_t len, SlotRef* out,
                        std::vector<BlockKey>* evicted) {
  std::lock_guard<std::mutex> lock(mu_);
  ++activity_seq_;
  if (uint32_t* existing = index_.Find(key)) {
    SlotMeta& meta = slots_[*existing];
    if (meta.remove_pending) return false;
    meta.referenced = true;
    if (out) *out = meta.ref;
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
    while (cls.ext_rr.size() < kStripeWays) {
      const ExtentBindResult bind = BindFreeExtent(cls_index);
      if (bind == ExtentBindResult::kBound) continue;
      if (bind == ExtentBindResult::kPersistenceError) return false;
      break;
    }
    if (PopFreeLocked(cls, &placement)) break;
    if (cls.bound_extents < kStripeWays) {
      const ExtentBindResult grow =
          StealExtentFor(cls_index, evicted, kStripeWays);
      if (grow == ExtentBindResult::kBound) continue;
      if (grow == ExtentBindResult::kPersistenceError) return false;
    }
    const ExtentBindResult cold = StealColdExtentFor(cls_index, evicted);
    if (cold == ExtentBindResult::kBound) continue;
    if (cold == ExtentBindResult::kPersistenceError) return false;
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
  meta.put_seq = ++put_seq_;
  meta.key = key;
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
  cls.useful_bytes += len;
  --extent.free_slots;
  extent.useful_bytes += len;
  extent.youngest_seq = std::max(extent.youngest_seq, meta.put_seq);
  used_bytes_ += meta.ref.slot_size;
  if (out) *out = meta.ref;
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
    result.push_back(stat);
  }
  return result;
}

size_t SlabAllocator::ReclaimClass(size_t cls_index, size_t target_free,
                                   size_t max_victims,
                                   std::vector<BlockKey>* evicted) {
  std::lock_guard<std::mutex> lock(mu_);
  if (cls_index >= classes_.size()) return 0;
  Class& cls = classes_[cls_index];
  size_t count = 0;
  while (count < max_victims && cls.free_count < target_free) {
    if (unbound_ > 0) break;
    const size_t before = cls.free_count;
    if (!EvictOneFrom(cls_index, evicted)) break;
    ++count;
    if (cls.free_count <= before) break;
  }
  return count;
}

size_t SlabAllocator::EvictColdToTarget(uint64_t target_bytes,
                                        size_t max_extents,
                                        std::vector<BlockKey>* evicted) {
  std::lock_guard<std::mutex> lock(mu_);
  size_t freed = 0;
  while (used_bytes_ > target_bytes && freed < max_extents) {
    // Select the globally coldest fully unpinned extent across all classes.
    // Read heat wins, then insertion recency and useful bytes break ties. This
    // class-agnostic choice keeps headroom ahead of demand without evicting a
    // hotter donor merely because its class was examined first.
    int best = -1;
    uint64_t best_heat = std::numeric_limits<uint64_t>::max();
    uint64_t best_seq = std::numeric_limits<uint64_t>::max();
    uint64_t best_useful = std::numeric_limits<uint64_t>::max();
    for (uint32_t e = 0; e < extents_.size(); ++e) {
      const ExtentMeta& extent = extents_[e];
      if (extent.cls == kUnbound || extent.pinned != 0 ||
          extent.resident_count == 0)
        continue;
      const uint64_t heat = ExtentHeatLocked(extent);
      if (best < 0 || heat < best_heat ||
          (heat == best_heat && extent.youngest_seq < best_seq) ||
          (heat == best_heat && extent.youngest_seq == best_seq &&
           extent.useful_bytes < best_useful)) {
        best = static_cast<int>(e);
        best_heat = heat;
        best_seq = extent.youngest_seq;
        best_useful = extent.useful_bytes;
      }
    }
    if (best < 0) break;
    ExtentMeta& extent = extents_[static_cast<uint32_t>(best)];
    while (extent.resident_head != kNoSlot) {
      const uint32_t index = extent.resident_head;
      SlotMeta& meta = slots_[index];
      if (opt_.on_slot_evict && !opt_.on_slot_evict(meta.ref)) return freed;
      BlockKey victim = meta.key;
      FreeSlotLocked(index);
      if (evicted) evicted->push_back(std::move(victim));
      ++evictions_;
    }
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

}  // namespace dfkv
