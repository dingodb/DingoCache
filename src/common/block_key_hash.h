/* block_key_hash — a hash of every BlockKey field for in-memory tables. */
#ifndef DFKV_COMMON_BLOCK_KEY_HASH_H_
#define DFKV_COMMON_BLOCK_KEY_HASH_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <random>

#include <unistd.h>

#include "common/kv_types.h"

namespace dfkv {

// Every field mixed, seeded per process, as the log store's and slab's
// indexes do: keys whose fields are related (digest_hi a multiple of
// digest_lo, as a client's counters can be) must not collide, and keys
// chosen to collide cannot be computed from outside the process. The former
// digest_hi ^ digest_lo * C put every key {id * C + 1, id} into a handful of
// buckets.
struct BlockKeyMixHash {
  static uint64_t Mix(uint64_t h) {
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDull;
    h ^= h >> 33;
    h *= 0xC4CEB9FE1A85EC53ull;
    h ^= h >> 33;
    return h;
  }
  // The process's seed, taken on the first use. std::random_device can
  // throw (no entropy source in a restricted container); this runs inside
  // the noexcept operator(), where a throw would end the process, so the
  // seed then falls back to the clock, an address and the pid.
  static uint64_t Seed() noexcept {
    static const uint64_t seed = MakeSeed(&DeviceEntropy);
    return seed;
  }
  static uint64_t MakeSeed(uint64_t (*entropy)()) noexcept {
    try {
      return entropy();
    } catch (...) {
      static const char here = 0;
      return Mix(static_cast<uint64_t>(
                     std::chrono::steady_clock::now().time_since_epoch().count()) ^
                 reinterpret_cast<uintptr_t>(&here)) ^
             static_cast<uint64_t>(::getpid());
    }
  }
  static uint64_t DeviceEntropy() {
    std::random_device rd;
    return (uint64_t(rd()) << 32) ^ rd();
  }
  size_t operator()(const BlockKey& k) const noexcept {
    return static_cast<size_t>(
        Mix(Mix(Mix(k.digest_lo ^ Seed()) ^ k.digest_hi) ^ k.tenant_hash));
  }
};

}  // namespace dfkv

#endif  // DFKV_COMMON_BLOCK_KEY_HASH_H_
