// Retention (RETAIN) statistics of one storage engine, in a form every
// engine can fill; the group sums them over its disks.
#ifndef DFKV_RETENTION_STATS_H_
#define DFKV_RETENTION_STATS_H_

#include <cstdint>

namespace dfkv {

struct RetentionStats {
  bool enabled = false;
  uint64_t budget_bytes = 0;
  uint64_t protected_bytes = 0;
  uint64_t protected_slots = 0;
  uint64_t extents_pinned = 0;   // slab: extents holding >= 1 protected resident
  uint64_t expired = 0;
  uint64_t evict_skipped = 0;
  uint64_t wheel_entries = 0;
  uint32_t max_ttl_s = 0;
  uint32_t headroom_pct = 0;
  uint64_t results[7] = {0, 0, 0, 0, 0, 0, 0};  // by RetainCode
};

}  // namespace dfkv

#endif  // DFKV_RETENTION_STATS_H_
