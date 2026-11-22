// CRC-32 (IEEE 802.3, reflected polynomial 0xEDB88320, initial value and final
// xor 0xFFFFFFFF): the checksum of zlib, PNG and Ethernet. Crc32("123456789")
// is 0xCBF43926.
//
// Computed eight input bytes per step from eight 256-entry tables built at
// compile time ("slicing-by-8"). The values are those of the one-bit-at-a-time
// loop; only the cost differs. Input bytes are assembled explicitly, so the
// result does not depend on host byte order or alignment.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace dfkv {
namespace crc32_detail {

constexpr std::array<std::array<uint32_t, 256>, 8> MakeTables() {
  std::array<std::array<uint32_t, 256>, 8> t{};
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t crc = i;
    for (int k = 0; k < 8; ++k)
      crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    t[0][i] = crc;
  }
  for (uint32_t i = 0; i < 256; ++i)
    for (size_t s = 1; s < 8; ++s)
      t[s][i] = (t[s - 1][i] >> 8) ^ t[0][t[s - 1][i] & 0xFFu];
  return t;
}

inline constexpr std::array<std::array<uint32_t, 256>, 8> kTables = MakeTables();

}  // namespace crc32_detail

inline uint32_t Crc32(const uint8_t* p, size_t n) {
  const auto& t = crc32_detail::kTables;
  uint32_t crc = 0xFFFFFFFFu;
  while (n >= 8) {
    const uint32_t lo = crc ^ (static_cast<uint32_t>(p[0]) |
                               static_cast<uint32_t>(p[1]) << 8 |
                               static_cast<uint32_t>(p[2]) << 16 |
                               static_cast<uint32_t>(p[3]) << 24);
    crc = t[7][lo & 0xFFu] ^ t[6][(lo >> 8) & 0xFFu] ^
          t[5][(lo >> 16) & 0xFFu] ^ t[4][lo >> 24] ^ t[3][p[4]] ^
          t[2][p[5]] ^ t[1][p[6]] ^ t[0][p[7]];
    p += 8;
    n -= 8;
  }
  for (; n > 0; --n, ++p) crc = (crc >> 8) ^ t[0][(crc ^ *p) & 0xFFu];
  return crc ^ 0xFFFFFFFFu;
}

}  // namespace dfkv
