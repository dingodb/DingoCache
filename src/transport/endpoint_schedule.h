#ifndef DFKV_ENDPOINT_SCHEDULE_H_
#define DFKV_ENDPOINT_SCHEDULE_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dfkv {

// Load sample for one connection belonging to an already-routed endpoint.
// `id` is stable for the connection lifetime and provides deterministic ties.
struct EndpointConnectionLoad {
  uint64_t id = 0;
  size_t inflight = 0;
  uint64_t latency_ns = 0;  // EWMA; zero means not sampled yet
  bool healthy = true;
};

// Pick the least-loaded healthy connection. In-flight work is the primary
// signal, then latency, then stable connection id. Unsampled connections are
// preferred over sampled peers at the same load so every new connection gets
// one latency sample instead of remaining permanently unused.
inline size_t SelectEndpointConnection(
    const std::vector<EndpointConnectionLoad>& candidates) {
  size_t best = candidates.size();
  for (size_t i = 0; i < candidates.size(); ++i) {
    const auto& c = candidates[i];
    if (!c.healthy) continue;
    if (best == candidates.size()) {
      best = i;
      continue;
    }
    const auto& b = candidates[best];
    if (c.inflight != b.inflight) {
      if (c.inflight < b.inflight) best = i;
      continue;
    }
    const bool c_unsampled = c.latency_ns == 0;
    const bool b_unsampled = b.latency_ns == 0;
    if (c_unsampled != b_unsampled) {
      if (c_unsampled) best = i;
      continue;
    }
    if (c.latency_ns != b.latency_ns) {
      if (c.latency_ns < b.latency_ns) best = i;
      continue;
    }
    if (c.id < b.id) best = i;
  }
  return best;
}

}  // namespace dfkv

#endif  // DFKV_ENDPOINT_SCHEDULE_H_
