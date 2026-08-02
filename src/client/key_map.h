/* String key -> deterministic native BlockKey.
 *
 * Identity is SHA-256 truncated to 128 bits. The hash input is domain-tagged
 * and length-framed, so model/key boundaries are unambiguous and native v2
 * identities cannot alias earlier MD5-derived keys.
 */
#ifndef DFKV_KEY_MAP_H_
#define DFKV_KEY_MAP_H_

#include <cstdint>
#include <string>

#include "common/kv_types.h"
#include "utils/sha256.h"

namespace dfkv {

inline void AppendKeyU64(std::string* out, uint64_t value) {
  for (size_t i = 0; i < sizeof(value); ++i)
    out->push_back(static_cast<char>(value >> (8 * i)));
}

inline uint64_t KeyDigestWord(const uint8_t* bytes) {
  uint64_t value = 0;
  for (size_t i = 0; i < 8; ++i) value = (value << 8) | bytes[i];
  return value;
}

// model_hash remains part of the transitional C ABI. The v2 ABI supplies the
// complete canonical namespace/object bytes as key and fixes model_hash at zero.
inline BlockKey ToBlockKey(const std::string& key, uint64_t model_hash = 0) {
  static constexpr char kDomain[] = "DFKVKEY2";
  std::string input;
  input.reserve(sizeof(kDomain) - 1 + 2 * sizeof(uint64_t) + key.size());
  input.append(kDomain, sizeof(kDomain) - 1);
  AppendKeyU64(&input, model_hash);
  AppendKeyU64(&input, static_cast<uint64_t>(key.size()));
  input.append(key);

  uint8_t digest[32];
  Sha256(input.data(), input.size(), digest);
  return BlockKey{KeyDigestWord(digest), KeyDigestWord(digest + 8)};
}

}  // namespace dfkv

#endif  // DFKV_KEY_MAP_H_
