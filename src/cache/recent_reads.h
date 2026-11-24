/* recent_reads — which keys were read lately, for RAM-tier promotion. */
#ifndef DFKV_CACHE_RECENT_READS_H_
#define DFKV_CACHE_RECENT_READS_H_

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <list>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "common/block_key_hash.h"
#include "common/kv_types.h"

namespace dfkv {

// Keys (fingerprints, not payloads) read within the last window_ms, so that
// a whole-value cold read is promoted into RAM only when the key was read
// before (DFKV_RAM_PROMOTE=second): a one-time scan then does not push the
// re-read keys out. Bounded to `capacity` keys; the oldest go first.
class RecentReads {
 public:
  using Hasher = BlockKeyMixHash;
  RecentReads(uint64_t window_ms, size_t capacity)
      : window_ms_(window_ms), capacity_(capacity) {}

  // True if `key` was read within the window before now. With `note` it is
  // also noted as read now; without, nothing changes but expiry.
  bool ReadAgain(const BlockKey& key, uint64_t now_ms, bool note = true) {
    std::lock_guard<std::mutex> lock(mu_);
    Expire(now_ms);
    auto found = index_.find(key);
    if (!note) return found != index_.end();
    if (found != index_.end()) {
      // Moved to the back: order_ stays oldest first, one entry per key.
      order_.splice(order_.end(), order_, found->second);
      found->second->second = now_ms;
      return true;  // anything older than the window was expired above
    }
    order_.emplace_back(key, now_ms);
    index_.emplace(key, std::prev(order_.end()));
    if (index_.size() > capacity_) {
      index_.erase(order_.front().first);
      order_.pop_front();
    }
    return false;
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return index_.size();
  }

 private:
  using Order = std::list<std::pair<BlockKey, uint64_t>>;

  void Expire(uint64_t now_ms) {
    while (!order_.empty() && now_ms >= order_.front().second &&
           now_ms - order_.front().second >= window_ms_) {
      index_.erase(order_.front().first);
      order_.pop_front();
    }
  }

  const uint64_t window_ms_;
  const size_t capacity_;
  mutable std::mutex mu_;
  Order order_;  // oldest first
  std::unordered_map<BlockKey, Order::iterator, Hasher> index_;
};

}  // namespace dfkv

#endif  // DFKV_CACHE_RECENT_READS_H_
