/* Ketama consistent hash for client-side cache-node routing.
 * Mirrors dingofs's MD5-based KetamaConHash behaviour (the real SDK reuses
 * dingofs PeerGroup; this portable copy powers the standalone multi-node test). */
#ifndef DFKV_CON_HASH_H_
#define DFKV_CON_HASH_H_

#include <cstdint>
#include <map>
#include <string>
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
  bool LookupHash(uint32_t hash, std::string* node) const;
  size_t NodeCount() const { return nodes_.size(); }
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

}  // namespace dfkv

#endif  // DFKV_CON_HASH_H_
