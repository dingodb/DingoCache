#ifndef DFKV_MDS_ENDPOINTS_H_
#define DFKV_MDS_ENDPOINTS_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dfkv {

// Round-robin selection over a static MDS endpoint list with per-endpoint
// failure backoff. There is NO leader concept — the MDS tier is stateless, so
// any reachable MDS is correct (see design doc decision 3). The clock is
// injected (now_ms) so the policy is unit-testable without real time; callers
// pass a monotonic millisecond clock at runtime.
//
// Sticky selection (the registrar): Pick keeps returning the endpoint it
// returned last until that one is marked failed, so one MDS keeps seeing a
// member's heartbeats; only then does it move on.
class MdsEndpoints {
 public:
  explicit MdsEndpoints(std::vector<std::string> eps, uint64_t backoff_ms = 2000,
                        bool sticky = false)
      : eps_(std::move(eps)), backoff_ms_(backoff_ms), sticky_(sticky),
        until_(eps_.size(), 0) {}

  size_t size() const { return eps_.size(); }

  // Comma-joined endpoint list, for diagnostics/logging only.
  std::string Join() const {
    std::string s;
    for (const auto& e : eps_) { if (!s.empty()) s += ","; s += e; }
    return s;
  }

  // Returns the next endpoint whose backoff has expired, advancing the cursor.
  // If ALL are in backoff, returns the one closest to recovery (degrade, never
  // fail). Empty string only when the list itself is empty.
  std::string Pick(uint64_t now_ms) {
    if (eps_.empty()) return {};
    size_t best = 0;
    uint64_t best_until = UINT64_MAX;
    for (size_t i = 0; i < eps_.size(); ++i) {
      size_t idx = (cursor_ + i) % eps_.size();
      if (until_[idx] <= now_ms) {
        cursor_ = sticky_ ? idx : (idx + 1) % eps_.size();
        return eps_[idx];
      }
      if (until_[idx] < best_until) { best_until = until_[idx]; best = idx; }
    }
    cursor_ = sticky_ ? best : (best + 1) % eps_.size();
    return eps_[best];
  }

  // Sticky: where the first Pick starts. Registrars seed it from their member
  // id, so that members spread over the MDS instead of all starting on the
  // first one and failing over together.
  void SeedCursor(uint64_t seed) {
    if (!eps_.empty()) cursor_ = static_cast<size_t>(seed % eps_.size());
  }

  void MarkFailed(const std::string& ep, uint64_t now_ms) {
    for (size_t i = 0; i < eps_.size(); ++i)
      if (eps_[i] == ep) {
        until_[i] = now_ms + backoff_ms_;
        // Sticky: the next Pick starts after the failed endpoint.
        if (sticky_ && i == cursor_) cursor_ = (i + 1) % eps_.size();
      }
  }

  void MarkOk(const std::string& ep) {
    for (size_t i = 0; i < eps_.size(); ++i)
      if (eps_[i] == ep) until_[i] = 0;
  }

 private:
  std::vector<std::string> eps_;
  uint64_t backoff_ms_;
  bool sticky_;
  std::vector<uint64_t> until_;  // backoff-until (ms) per endpoint; 0 = eligible
  size_t cursor_ = 0;
};

}  // namespace dfkv

#endif  // DFKV_MDS_ENDPOINTS_H_
