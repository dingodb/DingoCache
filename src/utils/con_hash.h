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
  size_t RingSize() const { return ring_.size(); }
  // Ring points owned per node (the realized vnode distribution after hash
  // collisions). Used by `dfkvctl ring` to show routing share. Off the datapath.
  std::map<std::string, size_t> NodePointCounts() const;

 private:
  std::vector<std::pair<std::string, int>> nodes_;  // (name, weight)
  std::map<uint32_t, std::string> ring_;            // point -> node
};

}  // namespace dfkv

#endif  // DFKV_CON_HASH_H_
