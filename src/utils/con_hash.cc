#include "utils/con_hash.h"

#include "utils/md5.h"

#include <algorithm>

namespace dfkv {

void ConHash::AddNode(const std::string& name, int weight) {
  if (weight < 1) weight = 1;
  nodes_.emplace_back(name, weight);
}

void ConHash::Build() {
  // Ketama: 40 keys * 4 points = 160 vnodes per weight unit. Clamp the weight
  // to [1, 1024]: 40 * weight overflows int for huge weights (keys goes
  // negative and the node silently gets ZERO vnodes — dropped from the data
  // plane while still listed as a member), and weight <= 0 had the same
  // silent-drop effect. 1024 (163 840 vnodes) is far beyond any real skew.
  std::map<uint32_t, uint32_t> ring;  // point -> node index
  collisions_ = 0;
  for (size_t index = 0; index < nodes_.size(); ++index) {
    const auto& [name, weight] = nodes_[index];
    const int w = weight < 1 ? 1 : (weight > 1024 ? 1024 : weight);
    const int keys = 40 * w;
    for (int k = 0; k < keys; ++k) {
      uint8_t d[16];
      std::string s = name + "-" + std::to_string(k);
      Md5(s.data(), s.size(), d);
      for (int h = 0; h < 4; ++h) {
        uint32_t point = uint32_t(d[h * 4]) | (uint32_t(d[h * 4 + 1]) << 8) |
                         (uint32_t(d[h * 4 + 2]) << 16) |
                         (uint32_t(d[h * 4 + 3]) << 24);
        // last-writer-wins on the rare collision
        auto [it, inserted] =
            ring.emplace(point, static_cast<uint32_t>(index));
        if (!inserted) {
          if (nodes_[it->second].first != name) ++collisions_;
          it->second = static_cast<uint32_t>(index);
        }
      }
    }
  }
  points_.assign(ring.begin(), ring.end());
}

std::map<std::string, size_t> ConHash::NodePointCounts() const {
  std::map<std::string, size_t> counts;
  for (const auto& [point, index] : points_) counts[nodes_[index].first]++;
  return counts;
}

uint32_t ConHash::KeyHash(std::string_view key) {
  uint8_t digest[16];
  Md5(key.data(), key.size(), digest);
  return uint32_t(digest[0]) | (uint32_t(digest[1]) << 8) |
         (uint32_t(digest[2]) << 16) | (uint32_t(digest[3]) << 24);
}

uint32_t ConHash::KeyHash(const std::string& key) {
  return KeyHash(std::string_view(key));
}

bool ConHash::Lookup(const std::string& key, std::string* node) const {
  return LookupHash(KeyHash(key), node);
}

bool ConHash::LookupHash(uint32_t h, std::string* node) const {
  if (points_.empty()) return false;
  auto it = std::lower_bound(
      points_.begin(), points_.end(), h,
      [](const std::pair<uint32_t, uint32_t>& point, uint32_t value) {
        return point.first < value;
      });
  if (it == points_.end()) it = points_.begin();  // wrap around the ring
  *node = nodes_[it->second].first;
  return true;
}

}  // namespace dfkv
