/* Portable KV identity shared by clients, transports, RAM, and disk stores.
 * Native identity is a full 128-bit digest. Compatibility frontends map their
 * legacy tuples into the same width but retain an isolated KeyDomain.
 */
#ifndef DFKV_KV_TYPES_H_
#define DFKV_KV_TYPES_H_

#include <cstdint>
#include <cstdio>
#include <string>

namespace dfkv {

enum class KeyDomain : uint32_t {
  kNative = 0,
  kSgEngineV1 = 1,
};

struct BlockKey {
  uint64_t digest_hi = 0;
  uint64_t digest_lo = 0;
  KeyDomain domain = KeyDomain::kNative;

  static std::string Hex64(uint64_t value) {
    char out[17];
    std::snprintf(out, sizeof(out), "%016llx",
                  static_cast<unsigned long long>(value));
    return std::string(out, 16);
  }

  std::string DigestHex() const {
    return Hex64(digest_hi) + Hex64(digest_lo);
  }

  std::string Filename() const {
    const std::string native = DigestHex();
    if (domain == KeyDomain::kNative) return native;
    if (domain == KeyDomain::kSgEngineV1) return "sgengine-v1_" + native;
    return "domain-" + std::to_string(static_cast<uint32_t>(domain)) + "_" +
           native;
  }

  std::string StoreKey() const {
    const std::string digest = DigestHex();
    const std::string buckets = digest.substr(0, 2) + "/" +
                                digest.substr(0, 4) + "/";
    if (domain == KeyDomain::kNative)
      return "blocks/" + buckets + Filename();
    if (domain == KeyDomain::kSgEngineV1)
      return "compat/sgengine-v1/blocks/" + buckets + Filename();
    return "compat/domain-" +
           std::to_string(static_cast<uint32_t>(domain)) + "/blocks/" +
           buckets + Filename();
  }

  bool operator==(const BlockKey& other) const {
    return digest_hi == other.digest_hi && digest_lo == other.digest_lo &&
           domain == other.domain;
  }
};

}  // namespace dfkv

#endif  // DFKV_KV_TYPES_H_
