/* Portable tenant-scoped KV identity shared by clients, transports, and stores. */
#ifndef DFKV_KV_TYPES_H_
#define DFKV_KV_TYPES_H_

#include <cstdint>
#include <cstdio>
#include <string>

namespace dfkv {

struct BlockKey {
  uint64_t digest_hi = 0;
  uint64_t digest_lo = 0;
  // Stable hash of the tenant identity. Kept last so aggregate BlockKey{hi, lo}
  // remains the tenant-neutral administrative/test form (tenant_hash == 0).
  uint64_t tenant_hash = 0;

  static std::string Hex64(uint64_t value) {
    char out[17];
    std::snprintf(out, sizeof(out), "%016llx",
                  static_cast<unsigned long long>(value));
    return std::string(out, 16);
  }

  std::string DigestHex() const {
    return Hex64(digest_hi) + Hex64(digest_lo);
  }

  // The on-disk name is a strict 48-character lower-case hexadecimal identity:
  // tenant hash first, followed by the 128-bit object digest.
  std::string Filename() const { return Hex64(tenant_hash) + DigestHex(); }

  static bool ParseFilename(const std::string& filename, BlockKey* out) {
    if (out == nullptr || filename.size() != 48) return false;
    auto word = [&](size_t offset, uint64_t* value) {
      uint64_t parsed = 0;
      for (size_t i = 0; i < 16; ++i) {
        const unsigned char c =
            static_cast<unsigned char>(filename[offset + i]);
        uint64_t nibble = 0;
        if (c >= '0' && c <= '9') {
          nibble = c - '0';
        } else if (c >= 'a' && c <= 'f') {
          nibble = c - 'a' + 10;
        } else {
          return false;
        }
        parsed = (parsed << 4) | nibble;
      }
      *value = parsed;
      return true;
    };
    BlockKey parsed;
    if (!word(0, &parsed.tenant_hash) || !word(16, &parsed.digest_hi) ||
        !word(32, &parsed.digest_lo)) {
      return false;
    }
    *out = parsed;
    return true;
  }

  // Hierarchical object path: blocks/{first byte}/{first two bytes}/{filename}.
  // The deterministic prefix buckets match the production object-store layout.
  std::string StoreKey() const {
    const std::string filename = Filename();
    return "blocks/" + filename.substr(0, 2) + "/" +
           filename.substr(0, 4) + "/" + filename;
  }

  bool operator==(const BlockKey& other) const {
    return digest_hi == other.digest_hi && digest_lo == other.digest_lo &&
           tenant_hash == other.tenant_hash;
  }
};

}  // namespace dfkv

#endif  // DFKV_KV_TYPES_H_
