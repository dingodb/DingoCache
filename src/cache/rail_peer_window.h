/* rail_peer_window — distinct peer hosts seen within a time window. */
#ifndef DFKV_CACHE_RAIL_PEER_WINDOW_H_
#define DFKV_CACHE_RAIL_PEER_WINDOW_H_

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace dfkv {

// The peer hosts that something happened with on one rail in the last
// window_ms, for a gauge. Bounded: at most `capacity` hosts are held; a new
// one replaces the least recent. Noted only on rare events (a connection's
// failed completion), so a mutex and a linear scan are enough.
class PeerWindow {
 public:
  explicit PeerWindow(size_t capacity = 64, uint64_t window_ms = 30000)
      : capacity_(capacity), window_ms_(window_ms) {}

  void Note(const std::string& host, uint64_t now_ms) {
    if (host.empty() || capacity_ == 0) return;
    std::lock_guard<std::mutex> lock(mu_);
    size_t oldest = 0;
    for (size_t i = 0; i < seen_.size(); ++i) {
      if (seen_[i].first == host) {
        seen_[i].second = now_ms;
        return;
      }
      if (seen_[i].second < seen_[oldest].second) oldest = i;
    }
    if (seen_.size() < capacity_)
      seen_.emplace_back(host, now_ms);
    else
      seen_[oldest] = {host, now_ms};
  }

  // Hosts noted within the window ending at now_ms.
  size_t Count(uint64_t now_ms) const {
    std::lock_guard<std::mutex> lock(mu_);
    size_t n = 0;
    for (const auto& [host, at] : seen_)
      n += at <= now_ms && now_ms - at <= window_ms_;
    return n;
  }

 private:
  const size_t capacity_;
  const uint64_t window_ms_;
  mutable std::mutex mu_;
  std::vector<std::pair<std::string, uint64_t>> seen_;
};

// The host part of "ip:port" or "[ip]:port"; empty for anything else
// (net::PeerAddress answers "unknown" without an IP peer).
inline std::string PeerHost(const std::string& address) {
  if (!address.empty() && address.front() == '[') {
    const size_t close = address.find(']');
    return close == std::string::npos ? std::string()
                                      : address.substr(1, close - 1);
  }
  const size_t colon = address.rfind(':');
  return colon == std::string::npos || colon == 0 ? std::string()
                                                  : address.substr(0, colon);
}

}  // namespace dfkv

#endif  // DFKV_CACHE_RAIL_PEER_WINDOW_H_
