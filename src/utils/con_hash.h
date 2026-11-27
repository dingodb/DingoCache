/* Ketama consistent hash for client-side cache-node routing.
 * Mirrors dingofs's MD5-based KetamaConHash behaviour (the real SDK reuses
 * dingofs PeerGroup; this portable copy powers the standalone multi-node test). */
#ifndef DFKV_CON_HASH_H_
#define DFKV_CON_HASH_H_

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace dfkv {

class ConHash {
 public:
  void AddNode(const std::string& name, int weight = 1);
  void Build();
  // Returns false when the ring is empty.
  bool Lookup(const std::string& key, std::string* node) const;
  // The same with the key's hash, KeyHash(key), computed by the caller (for
  // example outside a lock).
  static uint32_t KeyHash(const std::string& key);
  static uint32_t KeyHash(std::string_view key);
  bool LookupHash(uint32_t hash, std::string* node) const;
  // The distinct nodes clockwise from `hash`, the owner first: `visit` is
  // called for each until it returns false or every node was visited.
  template <typename Visit>
  void Walk(uint32_t hash, Visit visit) const;
  size_t NodeCount() const { return nodes_.size(); }
  // The nodes as added: (name, weight).
  const std::vector<std::pair<std::string, int>>& Nodes() const {
    return nodes_;
  }
  size_t RingSize() const { return points_.size(); }
  // Points two vnodes hashed to: the node added later owns them (so the
  // order members are added in decides those points).
  size_t Collisions() const { return collisions_; }
  // Ring points owned per node (the realized vnode distribution after hash
  // collisions). Used by `dfkvctl ring` to show routing share. Off the datapath.
  std::map<std::string, size_t> NodePointCounts() const;

 private:
  std::vector<std::pair<std::string, int>> nodes_;  // (name, weight)
  // The ring, sorted by point: (point, index into nodes_). A sorted vector
  // rather than a map: a lookup is a binary search over contiguous memory.
  std::vector<std::pair<uint32_t, uint32_t>> points_;
  size_t collisions_ = 0;
};

template <typename Visit>
void ConHash::Walk(uint32_t hash, Visit visit) const {
  if (points_.empty()) return;
  size_t at = static_cast<size_t>(
      std::lower_bound(points_.begin(), points_.end(), hash,
                       [](const std::pair<uint32_t, uint32_t>& point,
                          uint32_t value) { return point.first < value; }) -
      points_.begin());
  std::vector<bool> seen(nodes_.size(), false);
  size_t distinct = 0;
  for (size_t step = 0; step < points_.size() && distinct < nodes_.size();
       ++step, ++at) {
    if (at == points_.size()) at = 0;
    const uint32_t index = points_[at].second;
    if (seen[index]) continue;
    seen[index] = true;
    ++distinct;
    if (!visit(nodes_[index].first)) return;
  }
}

}  // namespace dfkv

#endif  // DFKV_CON_HASH_H_
