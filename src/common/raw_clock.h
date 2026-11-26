/* raw_clock — CLOCK_MONOTONIC_RAW readings for ages compared across nodes.
 *
 * The rewrite check of the read-back (docs/design/ring-unavailable-owner.md
 * §3.5) compares ages measured on different nodes, which is sound only on a
 * clock that runs at the oscillator's rate: CLOCK_MONOTONIC and
 * CLOCK_BOOTTIME are slewed by NTP daemons (chrony up to one twelfth by
 * default), CLOCK_MONOTONIC_RAW is not. */
#ifndef DFKV_COMMON_RAW_CLOCK_H_
#define DFKV_COMMON_RAW_CLOCK_H_

#include <cstdint>
#include <ctime>

namespace dfkv {

inline int64_t ClockNs(clockid_t id) {
  timespec ts{};
  clock_gettime(id, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000000 + ts.tv_nsec;
}

// Milliseconds of CLOCK_MONOTONIC_RAW.
inline uint64_t RawMs() {
  return static_cast<uint64_t>(ClockNs(CLOCK_MONOTONIC_RAW) / 1000000);
}

// The same, with the fraction kept (the client's request times).
inline double RawMsExact() {
  return static_cast<double>(ClockNs(CLOCK_MONOTONIC_RAW)) / 1e6;
}

}  // namespace dfkv

#endif  // DFKV_COMMON_RAW_CLOCK_H_
