// Recent-PUT memory W (CLI-35; docs/design/ring-unavailable-owner.md §3.5).
// A node remembers how long ago it last acknowledged a PUT of each key, for
// the read-back's rewrite check: a key removed elsewhere and written again
// here afterwards may be read back. Ages are on CLOCK_MONOTONIC_RAW.
//
// RAM only, bounded (DFKV_WRITE_MEMORY_ENTRIES), window the node's
// DFKV_REMOVE_MEMORY_MS. A key it does not hold has no recent write, which
// only costs a read-back. The node clears it whenever its raw clock may
// have stopped while time passed: the spread of (CLOCK_REALTIME - RAW) or
// (CLOCK_BOOTTIME - RAW) over the samples of the window exceeding G.
#ifndef DFKV_WRITE_MEMORY_H_
#define DFKV_WRITE_MEMORY_H_

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "cache/remove_memory.h"
#include "common/kv_types.h"

namespace dfkv {

class WriteMemory {
 public:
  static constexpr size_t kShards = 16;
  // G: the clock spread that clears the table (and the client's guard).
  static constexpr int64_t kPauseGuardMs = 1000;
  // Clear reasons (metrics).
  enum ClearReason { kRealtime = 0, kBoottime = 1, kClearReasons = 2 };

  // Offsets of CLOCK_REALTIME and CLOCK_BOOTTIME from RAW, in ms.
  struct Offsets {
    int64_t realtime = 0;
    int64_t boottime = 0;
  };
  using OffsetsFn = std::function<Offsets()>;
  static Offsets SystemOffsets();

  // tick_ms 0: no background tick (tests call Check).
  WriteMemory(uint64_t window_ms, size_t capacity, uint64_t now_ms,
              OffsetsFn offsets = SystemOffsets, uint64_t tick_ms = 1000);
  ~WriteMemory();

  // DFKV_WRITE_MEMORY_ENTRIES (0 = off: nullptr), with M's window
  // (window_ms 0, M off: nullptr).
  static std::unique_ptr<WriteMemory> FromEnv(uint64_t window_ms,
                                              uint64_t now_ms);

  // A PUT this node is about to acknowledge.
  void Record(const BlockKey& key, uint64_t now_ms);

  // The server serializes one key with M and op 26, independently of the
  // admission lock the RAM tier may release during a flush wait. Op 26 holds
  // all its hashed logical-key locks in bucket order, each once.
  // The pause check (the tick runs it every tick_ms): clears every shard
  // when the clock spread over the window exceeds G.
  void Check(uint64_t now_ms);

  // Ages (ms) of each key's latest PUT within age_ms, kAgeNone if none
  // (RemoveMemory's constants). Runs the pause check, then each shard's
  // expiry, first.
  void Ages(uint64_t tenant_hash, const char* digests, size_t count,
            uint64_t age_ms, uint64_t now_ms, uint32_t* out);

  // Joint op-26 M/W answer. Holds M then W throughout, selects one effective
  // time at or after both tables' watermarks, and evaluates both at that time.
  // Returns M's coverage. Other-key PUT/REMOVE calls may advance their tables
  // only before or after this snapshot; the queried publication buckets guard
  // the requested keys themselves.
  uint64_t AgesWithRemovals(RemoveMemory& removes, uint64_t tenant_hash,
                            const char* digests, size_t count, uint64_t age_ms,
                            uint64_t now_ms, uint32_t* removed,
                            uint32_t* written);

  size_t Entries();
  uint64_t Dropped() const;
  uint64_t Cleared(ClearReason reason) const;

 private:
  struct Sample {
    uint64_t at;
    Offsets off;
  };
  RemoveMemory& ShardOf(const BlockKey& key);
  void AgesLocked(uint64_t tenant_hash, const char* digests, size_t count,
                  uint64_t age_ms, uint64_t now_ms, uint32_t* out);
  void CheckLocked(uint64_t now_ms);
  void RestartLocked(uint64_t now_ms, const Offsets& cur);
  void TickLoop(uint64_t tick_ms);

  const uint64_t window_ms_;
  const OffsetsFn offsets_;
  // Under check_mu_: nondecreasing W domain time, independent of the request
  // times a caller sampled before waiting for this lock.
  uint64_t last_now_ms_ = 0;
  std::vector<std::unique_ptr<RemoveMemory>> shards_;
  std::mutex check_mu_;
  // Samples of the window (one older one kept as its start), at most one
  // a second, and their ranges.
  std::deque<Sample> samples_;
  Offsets min_, max_;
  std::atomic<uint64_t> cleared_[kClearReasons] = {};
  std::mutex tick_mu_;
  std::condition_variable tick_cv_;
  bool stop_ = false;
  std::thread tick_;
};

}  // namespace dfkv

#endif  // DFKV_WRITE_MEMORY_H_
