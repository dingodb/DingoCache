// Recent-REMOVE memory (CLI-02 stage M; docs/design/ring-unavailable-owner.md
// §3.3). A node remembers the keys removed on it recently and answers "was
// this key removed here within the last `age` milliseconds?" (kRemovedWithin).
// The read-back of stage B asks it before reading a previous owner; nothing
// else consults it, so no other op's answer changes.
//
// RAM only: after a start the node knows the REMOVEs since the start, and a
// question reaching further back is answered "unknown". When full, the oldest
// record is dropped and what the node knows starts later.
#ifndef DFKV_REMOVE_MEMORY_H_
#define DFKV_REMOVE_MEMORY_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "common/block_key_hash.h"
#include "common/kv_types.h"

namespace dfkv {

class RemoveMemory {
 public:
  // Per-key answers of a query.
  static constexpr uint8_t kNotRemoved = 0;
  static constexpr uint8_t kRemoved = 1;
  static constexpr uint8_t kUnknown = 2;

  // Above this capacity the ring and index are allocated whole at start:
  // growing them copies under the table's lock, a REMOVE stall of about a
  // second at the largest capacity. At or below it they grow by doubling
  // (a few ms each time).
  static constexpr size_t kPresizeAbove = size_t{4} << 20;

  // window_ms: how long a record is kept; capacity: records held at most;
  // presize_above: see kPresizeAbove (a parameter for tests).
  RemoveMemory(uint64_t window_ms, size_t capacity, uint64_t now_ms,
               size_t presize_above = kPresizeAbove);

  // DFKV_REMOVE_MEMORY_MS (0 = off: nullptr) and DFKV_REMOVE_MEMORY_ENTRIES.
  static std::unique_ptr<RemoveMemory> FromEnv(uint64_t now_ms);

  // Times passed in may be read before the caller's turn at the table's lock
  // (so, slightly older than one already used); the table keeps its time
  // monotonic.
  // A REMOVE of `key` this node is about to acknowledge (removed or absent).
  void Record(const BlockKey& key, uint64_t now_ms);

  // How far back, in ms, this node knows every REMOVE (at most the window).
  uint64_t CoveredMs(uint64_t now_ms);

  // Answers for `count` keys of one tenant (digests at `digests`, 16 bytes
  // each, hi then lo as net::PutU64 writes them): kRemoved if removed within
  // the last age_ms, kNotRemoved if not, kUnknown (every key) if age_ms
  // reaches past what this node knows. Returns CoveredMs.
  uint64_t Query(uint64_t tenant_hash, const char* digests, size_t count,
                 uint64_t age_ms, uint64_t now_ms, uint8_t* out);

  // The same question answered with ages (kRemovedWithinAges, op 26): for
  // each key, how many ms ago its latest record was made, kAgeNone if not
  // within age_ms, kAgeUnknown (every key) if age_ms reaches past what is
  // known. Expires first, as Query does. Returns CoveredMs.
  static constexpr uint32_t kAgeNone = 0xFFFFFFFFu;
  static constexpr uint32_t kAgeUnknown = 0xFFFFFFFEu;
  uint64_t QueryAges(uint64_t tenant_hash, const char* digests, size_t count,
                     uint64_t age_ms, uint64_t now_ms, uint32_t* out);

  // Forgets every record; what the table knows starts at now_ms.
  void Clear(uint64_t now_ms);

  // Moves the table's time to at least now_ms: a record made after this
  // call is stamped now_ms or later, whenever its caller read the clock.
  void AdvanceTo(uint64_t now_ms);

  uint64_t WindowMs() const { return window_ms_; }

  // Metrics.
  size_t Entries();
  uint64_t Dropped() const;
  uint64_t QueriedKeys(uint8_t answer) const;

  // Allocated ring records and index slots (tests).
  size_t RingSlotsForTest();
  size_t IndexSlotsForTest();

 private:
  friend class WriteMemory;
  // Caller holds mu_ and has selected an exact joint M/W snapshot time.
  uint64_t QueryAgesLocked(uint64_t tenant_hash, const char* digests,
                           size_t count, uint64_t age_ms, uint64_t now_ms,
                           uint32_t* out);
  // One REMOVE: 32 bytes.
  struct Rec {
    BlockKey key;
    uint64_t at;
  };
  static constexpr uint32_t kEmpty = UINT32_MAX;

  uint64_t NowLocked(uint64_t now_ms);
  void ExpireLocked(uint64_t now_ms);
  void PopFrontLocked();
  void PushBackLocked(const BlockKey& key, uint64_t now_ms);
  // The index slot holding `key`'s latest record, or the empty slot where it
  // would go (index_[slot] == kEmpty).
  size_t SlotLocked(const BlockKey& key) const;
  void EraseSlotLocked(size_t slot);
  void RebuildIndexLocked(size_t slots);

  const uint64_t window_ms_;
  const size_t capacity_;
  std::mutex mu_;
  // Records in the order they were made (time ascending), a ring of
  // size_ records from head_ that grows up to capacity_; a key removed
  // again has an older record here that the index no longer points at.
  std::vector<Rec> ring_;
  size_t head_ = 0, size_ = 0;
  // Open addressing (linear probing, at most 70 % full): the ring position
  // of each key's latest record, kEmpty for a free slot. 4 bytes a slot.
  std::vector<uint32_t> index_;
  size_t live_ = 0;  // keys in the index
  // Start, or just after the newest record dropped for room.
  uint64_t covered_since_ms_;
  uint64_t last_now_ms_ = 0;  // the latest time seen: never goes back
  std::atomic<uint64_t> dropped_{0};
  std::atomic<uint64_t> answers_[3] = {};
};

}  // namespace dfkv

#endif  // DFKV_REMOVE_MEMORY_H_
