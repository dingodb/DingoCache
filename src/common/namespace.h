/* Canonical cache namespace and object identity.
 *
 * Every native connector serializes these descriptors byte-for-byte before a
 * key is hashed. The format is fixed-width little-endian plus length-prefixed
 * byte strings: delimiters, locale, and Python/C++ integer formatting can never
 * change identity. Rank == -1 means that axis is replicated/shared; otherwise
 * the rank is part of the namespace and isolates the shard.
 */
#ifndef DFKV_NAMESPACE_H_
#define DFKV_NAMESPACE_H_

#include <cstdint>
#include <string>

#include "common/status.h"

namespace dfkv {

enum class PoolKind : uint16_t {
  kAttention = 1,
  kMla = 2,
  kMamba = 3,
  kSwa = 4,
  kIndexer = 5,
  kDraftAttention = 6,
  kDraftMla = 7,
  kDraftIndexer = 8,
  kCustom = 65535,
};

struct RankCoordinate {
  uint32_t world_size = 1;
  int32_t rank = -1;  // -1 = replicated/shared on this axis.

  Status Validate(const char* axis, std::string* error = nullptr) const;
};

struct NamespaceDescriptor {
  std::string tenant_id;
  std::string model_id;
  std::string model_revision;
  PoolKind pool_kind = PoolKind::kCustom;
  std::string pool_name;
  std::string dtype;
  uint64_t layout_fingerprint = 0;
  uint32_t block_tokens = 0;
  uint32_t group_id = 0;
  uint32_t layer_begin = 0;
  uint32_t layer_count = 0;
  RankCoordinate tp;
  RankCoordinate dp;
  RankCoordinate pp;

  Status Validate(std::string* error = nullptr) const;

  // DFKVNS\0\2 followed by canonical fields. Call Validate() before using a
  // descriptor supplied by an untrusted caller.
  std::string Serialize() const;
};

struct ObjectDescriptor {
  // Framework-independent logical page/block hash. It is an opaque byte string,
  // not a C string; embedded NULs are preserved by serialization.
  std::string logical_key;
  // Pool-local component, e.g. "kv", "k", "v", "temporal", "conv:0".
  std::string component;
  uint32_t chunk_index = 0;
  uint32_t chunk_count = 1;

  Status Validate(std::string* error = nullptr) const;

  // DFKVOBJ2 + length-prefixed namespace + object fields.
  std::string Serialize(const NamespaceDescriptor& ns) const;
};

}  // namespace dfkv

#endif  // DFKV_NAMESPACE_H_
