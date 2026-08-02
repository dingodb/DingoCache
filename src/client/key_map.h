/* Canonical namespace/object bytes -> deterministic tenant-scoped BlockKey.
 *
 * Object identity is SHA-256 truncated to 128 bits. Tenant identity is a
 * separately domain-separated SHA-256 truncated to 64 bits: canonical
 * DFKVNS-v2 namespaces use their first length-framed tenant field; malformed
 * or noncanonical namespaces use the complete namespace bytes.
 */
#ifndef DFKV_KEY_MAP_H_
#define DFKV_KEY_MAP_H_

#include <limits>
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
inline bool CanonicalTenantIdentity(const std::string& key_namespace,
                                    size_t* tenant_offset,
                                    size_t* tenant_length) {
  static constexpr char kMagic[8] = {'D', 'F', 'K', 'V', 'N', 'S', '\0', '\2'};
  if (key_namespace.size() < sizeof(kMagic) + 4 ||
      key_namespace.compare(0, sizeof(kMagic), kMagic, sizeof(kMagic)) != 0) {
    return false;
  }
  size_t pos = sizeof(kMagic);
  auto u32 = [&](uint32_t* value) {
    if (pos > key_namespace.size() ||
        key_namespace.size() - pos < sizeof(uint32_t)) {
      return false;
    }
    const auto* p =
        reinterpret_cast<const unsigned char*>(key_namespace.data() + pos);
    *value = static_cast<uint32_t>(p[0]) |
             (static_cast<uint32_t>(p[1]) << 8) |
             (static_cast<uint32_t>(p[2]) << 16) |
             (static_cast<uint32_t>(p[3]) << 24);
    pos += sizeof(uint32_t);
    return true;
  };
  auto u16 = [&](uint16_t* value) {
    if (pos > key_namespace.size() || key_namespace.size() - pos < 2)
      return false;
    const auto* p =
        reinterpret_cast<const unsigned char*>(key_namespace.data() + pos);
    *value = static_cast<uint16_t>(p[0]) |
             static_cast<uint16_t>(static_cast<uint16_t>(p[1]) << 8);
    pos += 2;
    return true;
  };
  auto u64 = [&](uint64_t* value) {
    if (pos > key_namespace.size() || key_namespace.size() - pos < 8)
      return false;
    const auto* p =
        reinterpret_cast<const unsigned char*>(key_namespace.data() + pos);
    uint64_t parsed = 0;
    for (size_t i = 0; i < 8; ++i)
      parsed |= static_cast<uint64_t>(p[i]) << (8 * i);
    *value = parsed;
    pos += 8;
    return true;
  };
  auto field = [&](size_t* offset, size_t* length) {
    uint32_t n = 0;
    if (!u32(&n) || n == 0 || n > (1u << 20) ||
        n > key_namespace.size() - pos) {
      return false;
    }
    if (offset) *offset = pos;
    if (length) *length = n;
    pos += n;
    return true;
  };

  size_t first_offset = 0;
  size_t first_length = 0;
  if (!field(&first_offset, &first_length) || !field(nullptr, nullptr) ||
      !field(nullptr, nullptr)) {
    return false;
  }
  uint16_t pool_kind = 0;
  if (!u16(&pool_kind) ||
      !((pool_kind >= 1 && pool_kind <= 8) || pool_kind == 65535) ||
      !field(nullptr, nullptr) || !field(nullptr, nullptr)) {
    return false;
  }
  uint64_t layout_fingerprint = 0;
  uint32_t block_tokens = 0;
  uint32_t group_id = 0;
  uint32_t layer_begin = 0;
  uint32_t layer_count = 0;
  if (!u64(&layout_fingerprint) || !u32(&block_tokens) || !u32(&group_id) ||
      !u32(&layer_begin) || !u32(&layer_count) ||
      layout_fingerprint == 0 || block_tokens == 0 || layer_count == 0 ||
      static_cast<uint64_t>(layer_begin) + layer_count >
          std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  (void)group_id;
  for (size_t axis = 0; axis < 3; ++axis) {
    uint32_t world_size = 0;
    uint32_t rank = 0;
    if (!u32(&world_size) || !u32(&rank) || world_size == 0 ||
        (rank != std::numeric_limits<uint32_t>::max() &&
         rank >= world_size)) {
      return false;
    }
  }
  if (pos != key_namespace.size()) return false;
  *tenant_offset = first_offset;
  *tenant_length = first_length;
  return true;
}

inline uint64_t TenantHash(const std::string& key_namespace) {
  static constexpr char kDomain[] = "DFKVTENANT1";
  size_t offset = 0;
  size_t length = 0;
  const bool canonical =
      CanonicalTenantIdentity(key_namespace, &offset, &length);
  const char* identity =
      canonical ? key_namespace.data() + offset : key_namespace.data();
  const size_t identity_length = canonical ? length : key_namespace.size();
  std::string input;
  input.reserve(sizeof(kDomain) - 1 + sizeof(uint64_t) + identity_length);
  input.append(kDomain, sizeof(kDomain) - 1);
  AppendKeyU64(&input, static_cast<uint64_t>(identity_length));
  input.append(identity, identity_length);
  uint8_t digest[32];
  Sha256(input.data(), input.size(), digest);
  return KeyDigestWord(digest);
}


inline BlockKey ToBlockKey(const std::string& key_namespace,
                           const std::string& object_key) {
  static constexpr char kDomain[] = "DFKVKEY2";
  std::string input;
  input.reserve(sizeof(kDomain) - 1 + 2 * sizeof(uint64_t) +
                key_namespace.size() + object_key.size());
  input.append(kDomain, sizeof(kDomain) - 1);
  AppendKeyU64(&input, static_cast<uint64_t>(key_namespace.size()));
  input.append(key_namespace);
  AppendKeyU64(&input, static_cast<uint64_t>(object_key.size()));
  input.append(object_key);

  uint8_t digest[32];
  Sha256(input.data(), input.size(), digest);
  return BlockKey{KeyDigestWord(digest), KeyDigestWord(digest + 8),
                  TenantHash(key_namespace)};
}

}  // namespace dfkv

#endif  // DFKV_KEY_MAP_H_
