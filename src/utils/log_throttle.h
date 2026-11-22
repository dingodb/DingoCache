/* Bounded throttle for a log site that failures can drive faster than a log
 * should grow.
 *
 * What it guarantees:
 *  - A window opens with an admitted line and ends with the first event at
 *    least `window_us` later. Within a window a key is admitted at most once,
 *    and at most kMaxKeys distinct keys are admitted, so the line rate is
 *    bounded by kMaxKeys per window however many sources fail.
 *  - Memory is fixed, whatever the key space.
 *  - Every event that is not admitted is counted. The count is a single total
 *    over all keys, handed to the next admitted line as `suppressed` and then
 *    reset.
 *
 * What it does not: `suppressed` says how many events were dropped since the
 * previous line, not which keys they belonged to, and it surfaces only when
 * some later line is admitted, which may be much later and about a different
 * key; drops after the last line stay unreported. Once kMaxKeys keys have been
 * admitted, every other key is dropped for the rest of that window, so a
 * source can go unnamed while more than kMaxKeys sources are failing. Use it
 * next to counters that see every event, not instead of them.
 *
 * Keying by what failed (for example peer, device and status) keeps one noisy
 * source from using up the lines of the others.
 *
 * Takes a mutex: call it only once a failure has already happened, never on a
 * success path. */
#ifndef DFKV_LOG_THROTTLE_H_
#define DFKV_LOG_THROTTLE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

namespace dfkv {

class LogThrottle {
 public:
  static constexpr size_t kMaxKeys = 32;
  static constexpr uint64_t kDefaultWindowUs = 10'000'000;

  explicit LogThrottle(uint64_t window_us = kDefaultWindowUs)
      : window_us_(window_us) {}

  // FNV-1a over the textual part of a key, then the two numeric parts.
  static uint64_t Key(const std::string& text, uint64_t a, uint64_t b) {
    uint64_t hash = 1469598103934665603ull;
    const auto mix = [&hash](uint64_t byte) {
      hash = (hash ^ byte) * 1099511628211ull;
    };
    for (unsigned char c : text) mix(c);
    for (int shift = 0; shift < 64; shift += 8) mix((a >> shift) & 0xff);
    for (int shift = 0; shift < 64; shift += 8) mix((b >> shift) & 0xff);
    return hash;
  }

  // True when the caller should log now. `now_us` is any monotonic clock.
  // On true, *suppressed is the number of events (of any key) swallowed since
  // the previous admitted line.
  bool Admit(uint64_t key, uint64_t now_us, uint64_t* suppressed) {
    std::lock_guard<std::mutex> lock(mu_);
    // Callers read the clock before taking the lock, so `now_us` may be
    // slightly older than the window start; only a later time ends a window.
    if (count_ != 0 && now_us >= window_start_us_ &&
        now_us - window_start_us_ >= window_us_)
      count_ = 0;
    bool seen = false;
    for (size_t i = 0; i < count_ && !seen; ++i) seen = keys_[i] == key;
    if (seen || count_ == kMaxKeys) {
      ++suppressed_;
      return false;
    }
    if (count_ == 0) window_start_us_ = now_us;
    keys_[count_++] = key;
    *suppressed = suppressed_;
    suppressed_ = 0;
    return true;
  }

 private:
  const uint64_t window_us_;
  std::mutex mu_;
  std::array<uint64_t, kMaxKeys> keys_{};
  size_t count_ = 0;
  uint64_t window_start_us_ = 0;
  uint64_t suppressed_ = 0;
};

}  // namespace dfkv

#endif  // DFKV_LOG_THROTTLE_H_
