#ifndef DFKV_CACHE_NEW_PUT_DEMAND_LATCH_H_
#define DFKV_CACHE_NEW_PUT_DEMAND_LATCH_H_

#include <cstdint>

// Successful NEW commits renew pressure eligibility,
// but 128 completed quiet observations expire it only after active work stops.
// The limit counts reclaimer observations, not a wall-time or per-key TTL.
struct NewPutDemandLatch {
  bool active = false;
  bool pressure_seen = false;
  static constexpr uint32_t kQuietTickLimit = 128;
  uint32_t quiet_ticks = 0;
  uint64_t current_interval_bytes = 0;
  uint64_t previous_interval_bytes = 0;

  void BeginTick(uint64_t used_bytes, uint64_t high_bytes, uint64_t low_bytes,
                 uint64_t newly_admitted_bytes) {
    previous_interval_bytes = current_interval_bytes;
    current_interval_bytes = newly_admitted_bytes;
    if (used_bytes == 0) {
      pressure_seen = false;
      quiet_ticks = 0;
    }
    if (used_bytes > high_bytes) {
      pressure_seen = true;
      active = true;
      quiet_ticks = 0;
    } else if (newly_admitted_bytes != 0) {
      quiet_ticks = 0;
      if (pressure_seen && used_bytes > low_bytes) active = true;
    } else if (pressure_seen) {
      if (quiet_ticks < kQuietTickLimit) ++quiet_ticks;
    } else {
      quiet_ticks = 0;
    }
  }

  void EndTick(uint64_t used_bytes, uint64_t low_bytes) {
    if (used_bytes <= low_bytes && current_interval_bytes == 0 &&
        previous_interval_bytes == 0)
      active = false;
    if (pressure_seen && !active && quiet_ticks == kQuietTickLimit) {
      pressure_seen = false;
      quiet_ticks = 0;
    }
  }
};

#endif
