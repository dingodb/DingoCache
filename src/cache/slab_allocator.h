/* SlabAllocator — media-agnostic slot lifecycle manager for the slab store. */
#ifndef DFKV_SLAB_ALLOCATOR_H_
#define DFKV_SLAB_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "common/kv_types.h"

namespace dfkv {

class SlabAllocator {
 public:
  struct SlotRef {
    uint32_t cls = 0;
    uint32_t extent = 0;
    uint32_t slot = 0;
    uint32_t slot_size = 0;
    uint64_t offset = 0;
    bool valid() const { return slot_size != 0; }
  };

  enum class RemoveResult { kRemoved, kDeferred, kNotFound };

  struct Options {
    uint64_t extent_bytes = (1ull << 30);
    uint32_t num_extents = 8;
    uint32_t align = 4096;
    double max_waste = 0.25;
    std::function<bool(uint32_t extent)> on_extent_bind;
    std::function<bool(const SlotRef&)> on_slot_evict;
  };

  static constexpr size_t kStripeWays = 8;

  explicit SlabAllocator(Options opt);

  bool Put(const BlockKey& key, size_t len, SlotRef* out,
           std::vector<BlockKey>* evicted);
  bool Get(const BlockKey& key, SlotRef* out);
  bool GetAndPin(const BlockKey& key, SlotRef* out);
  bool Contains(const BlockKey& key) const;

  // A rebuild record carrying the authoritative caller-visible stored length.
  struct RestoreEntry {
    BlockKey key;
    uint32_t slot_size = 0;
    uint32_t extent = 0;
    uint32_t slot = 0;
    uint64_t value_bytes = 0;
  };

  // Validate the whole batch, reserve the dense metadata/index once, then bind
  // and install it under one lock. On invalid input no resident is installed.
  bool RestoreBulk(const std::vector<RestoreEntry>& entries);

  RemoveResult Remove(const BlockKey& key);
  bool Pin(const BlockKey& key);
  bool Unpin(const BlockKey& key);

  struct ClassStat {
    uint32_t slot_size = 0;
    size_t free_slots = 0;
    size_t resident = 0;
    uint64_t puts = 0;
    uint32_t extents = 0;
    uint32_t slots_per_extent = 0;
    uint64_t useful_bytes = 0;    // exact live caller bytes
    uint64_t allocated_bytes = 0; // slot bytes occupied by residents
    uint64_t read_touches = 0;    // cumulative Get/GetAndPin hits
    uint64_t read_heat = 0;       // recent, operation-decayed read evidence
  };
  std::vector<ClassStat> Classes() const;
  uint32_t PoolExtents() const;

  size_t ReclaimClass(size_t cls_index, size_t target_free, size_t max_victims,
                      std::vector<BlockKey>* evicted);
  bool StealFrom(size_t donor_cls, size_t target_cls,
                 std::vector<BlockKey>* evicted);

  size_t Count() const;
  uint64_t UsedBytes() const;
  uint64_t Capacity() const;
  uint64_t Evictions() const;
  uint64_t Steals() const;
  uint64_t ColdSteals() const;
  uint64_t WatermarkEvictions() const;
  size_t EvictColdToTarget(uint64_t target_bytes, size_t max_extents,
                           std::vector<BlockKey>* evicted);
  uint64_t ExtentReturns() const;
  size_t ClassCount() const;
  uint32_t BoundExtents() const;

  struct IndexStats {
    size_t size = 0;
    size_t capacity = 0;
    size_t tombstones = 0;
    uint64_t table_allocations = 0;
    uint64_t rehashes = 0;
  };
  // Test-only observability for the no-allocation hot-path contract.
  IndexStats IndexStatsForTest() const;
  size_t IndexBucketForTest(const BlockKey& key) const;

 private:
  static constexpr int kUnbound = -1;
  static constexpr uint32_t kNoSlot = UINT32_MAX;
  static constexpr uint64_t kReadHeatUnit = 1024;
  static constexpr uint64_t kReadDecayOps = 64;

  enum class ExtentBindResult { kBound, kUnavailable, kPersistenceError };
  struct Slot { uint32_t extent; uint32_t slot; };

  // Exactly one live SlotMeta is authoritative for each resident. Both the
  // index and reverse metadata hold the canonical 24-byte BlockKey by value;
  // neither owns heap-backed key storage. All rings are integer intrusive links
  // into this dense table, whose free entries are reused.
  struct SlotMeta {
    BlockKey key;
    SlotRef ref;
    uint64_t useful_bytes = 0;
    uint64_t put_seq = 0;
    uint32_t free_next = kNoSlot;
    uint64_t read_touches = 0;
    uint32_t refs = 0;
    uint32_t class_prev = kNoSlot;
    uint32_t class_next = kNoSlot;
    uint32_t extent_prev = kNoSlot;
    uint32_t extent_next = kNoSlot;
    bool referenced = false;
    bool remove_pending = false;
    bool live = false;
  };

  struct Class {
    uint32_t slot_size = 0;
    uint32_t slots_per_extent = 0;
    std::unordered_map<uint32_t, std::vector<uint32_t>> free_by_ext;
    std::vector<uint32_t> ext_rr;
    size_t rr_next = 0;
    size_t free_count = 0;
    size_t resident_count = 0;
    uint32_t bound_extents = 0;
    uint64_t puts = 0;
    uint64_t useful_bytes = 0;
    uint64_t read_touches = 0;
    uint64_t read_heat = 0;
    uint64_t heat_seq = 0;
    uint32_t ring_head = kNoSlot;
    uint32_t hand = kNoSlot;
  };

  struct ExtentMeta {
    int cls = kUnbound;
    uint32_t free_slots = 0;
    uint32_t total_slots = 0;
    uint32_t pinned = 0;
    uint64_t youngest_seq = 0;
    uint64_t useful_bytes = 0;
    uint64_t read_heat = 0;
    uint64_t heat_seq = 0;
    uint32_t resident_head = kNoSlot;
    uint32_t resident_count = 0;
  };

  uint32_t CanonicalSlotSize(size_t aligned_len) const;
  size_t ClassForLen(size_t aligned_len);
  size_t ClassForExactSize(uint32_t slot_size);
  ExtentBindResult BindExtentLocked(size_t cls, uint32_t extent,
                                    bool metadata_prepared);
  ExtentBindResult BindFreeExtent(size_t cls);
  bool EvictOneFrom(size_t cls, std::vector<BlockKey>* evicted);
  ExtentBindResult StealExtentFor(size_t cls,
                                  std::vector<BlockKey>* evicted,
                                  size_t min_donor_extents = 0);
  ExtentBindResult StealColdExtentFor(size_t cls,
                                      std::vector<BlockKey>* evicted);
  ExtentBindResult StealExtentLocked(uint32_t extent, size_t target_cls,
                                      std::vector<BlockKey>* evicted);

  uint32_t AcquireSlotMetaLocked();
  void LinkResidentLocked(uint32_t meta_index);
  void FreeSlotLocked(uint32_t meta_index);
  void TouchLocked(uint32_t meta_index);
  uint64_t DecayedHeat(uint64_t heat, uint64_t heat_seq) const;
  uint64_t ExtentHeatLocked(const ExtentMeta& extent) const;

  void PushFreeLocked(Class& cls, uint32_t extent, uint32_t slot);
  bool PopFreeLocked(Class& cls, Slot* out);
  bool TakeFreeSlotLocked(Class& cls, uint32_t extent, uint32_t slot);
  void DropExtentFreeLocked(Class& cls, uint32_t extent);
  void DropFromRotationLocked(Class& cls, uint32_t extent);

  // Power-of-two linear probing. Erase only installs a tombstone; compaction
  // and growth happen on Reserve/Insert so Find, existing-key Put, and Remove
  // never allocate.
  class FlatKeyIndex {
   public:
    struct Entry {
      BlockKey key;
      uint32_t value = kNoSlot;
      uint8_t state = 0;  // 0 empty, 1 occupied, 2 tombstone
    };

    uint32_t* Find(const BlockKey& key);
    const uint32_t* Find(const BlockKey& key) const;
    bool Insert(const BlockKey& key, uint32_t value);
    bool Erase(const BlockKey& key);
    void Reserve(size_t expected_size);
    size_t size() const { return size_; }
    IndexStats stats() const;
    size_t BucketForTest(const BlockKey& key) const;

   private:
    static uint64_t Hash(const BlockKey& key);
    void Rehash(size_t capacity);

    std::vector<Entry> entries_;
    size_t size_ = 0;
    size_t tombstones_ = 0;
    uint64_t table_allocations_ = 0;
    uint64_t rehashes_ = 0;
  };

  Options opt_;
  mutable std::mutex mu_;
  std::vector<Class> classes_;
  std::vector<ExtentMeta> extents_;
  std::vector<SlotMeta> slots_;
  uint32_t free_slot_meta_head_ = kNoSlot;
  FlatKeyIndex index_;
  uint32_t unbound_ = 0;
  uint64_t used_bytes_ = 0;
  uint64_t evictions_ = 0;
  uint64_t steals_ = 0;
  uint64_t cold_steals_ = 0;
  uint64_t watermark_evictions_ = 0;
  uint64_t extent_returns_ = 0;
  uint64_t put_seq_ = 0;
  uint64_t activity_seq_ = 0;
  uint64_t cold_window_ = 0;
  bool cold_steal_enabled_ = true;
};

}  // namespace dfkv

#endif  // DFKV_SLAB_ALLOCATOR_H_
