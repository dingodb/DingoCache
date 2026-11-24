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
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "common/block_key_hash.h"
#include "common/kv_types.h"

namespace dfkv {

class RemoveMemory {
 public:
  // Per-key answers of a query.
  static constexpr uint8_t kNotRemoved = 0;
  static constexpr uint8_t kRemoved = 1;
  static constexpr uint8_t kUnknown = 2;

  // window_ms: how long a record is kept; capacity: records held at most.
  RemoveMemory(uint64_t window_ms, size_t capacity, uint64_t now_ms);

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

  // Metrics.
  size_t Entries();
  uint64_t Dropped() const;
  uint64_t QueriedKeys(uint8_t answer) const;

 private:
  uint64_t NowLocked(uint64_t now_ms);
  void ExpireLocked(uint64_t now_ms);

  const uint64_t window_ms_;
  const size_t capacity_;
  std::mutex mu_;
  // Records in the order they were made (time ascending); a key removed
  // again has an older record here that `last_` no longer points at.
  std::deque<std::pair<BlockKey, uint64_t>> order_;
  std::unordered_map<BlockKey, uint64_t, BlockKeyMixHash> last_;  // key -> last REMOVE
  // Start, or just after the newest record dropped for room.
  uint64_t covered_since_ms_;
  uint64_t last_now_ms_ = 0;  // the latest time seen: never goes back
  std::atomic<uint64_t> dropped_{0};
  std::atomic<uint64_t> answers_[3] = {};
};

}  // namespace dfkv

#endif  // DFKV_REMOVE_MEMORY_H_
