/* Ownership token of a dedup fetch claim (NodeDedup / GpuNodeDedup).
 *
 * A claim that returns kFetch hands the caller a token, and Publish/Abort
 * act only for the holder of the matching token. The token names the slot
 * that was claimed and the generation it had at that moment:
 *
 *   bit 63      always set (0 stays "no claim")
 *   bits 62..39 slot index (tables hold at most 2^24 slots)
 *   bits 38..0  low bits of the slot generation
 *
 * Naming the slot matters: a generation is a small per-slot counter, so the
 * same value turns up in other slots, and a key can move to another slot of
 * its probe window between a claim and its late publish. The token never
 * leaves the process that got it; it is not part of the shared layout. */
#ifndef DFKV_DEDUP_TOKEN_H_
#define DFKV_DEDUP_TOKEN_H_

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace dfkv {
namespace dedup_token {

constexpr uint64_t kValid = 1ull << 63;
constexpr int kGenBits = 39;
constexpr uint64_t kGenMask = (1ull << kGenBits) - 1;

inline uint64_t Make(uint32_t slot_index, uint64_t gen) {
  return kValid | (static_cast<uint64_t>(slot_index) << kGenBits) |
         (gen & kGenMask);
}
inline bool Valid(uint64_t token) { return (token & kValid) != 0; }
inline uint32_t SlotIndex(uint64_t token) {
  return static_cast<uint32_t>((token & ~kValid) >> kGenBits);
}
// True when `gen` is the generation the token was issued at.
inline bool SameGen(uint64_t token, uint64_t gen) {
  return (token & kGenMask) == (gen & kGenMask);
}

}  // namespace dedup_token

// Counts a lookup that found its key published with a length it does not
// expect, and says so once per process: dedup assumes that one key holds the
// same bytes in every process of the host. Defined in node_dedup.cc.
void NoteLengthMismatch(std::atomic<uint64_t>* counter, const char* tag,
                        size_t published, size_t expected);

// Longest a Publish/Abort waits for its slot while another process's stale
// Publish/Abort holds it for a re-check (a few instructions, unless that
// process is preempted or killed right there).
constexpr int kDedupLockWaitMs = 50;
}  // namespace dfkv

#endif  // DFKV_DEDUP_TOKEN_H_
