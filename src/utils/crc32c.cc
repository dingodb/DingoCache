#include "utils/crc32c.h"

#include <array>
#include <cstring>

#if defined(__x86_64__)
#include <nmmintrin.h>
#elif defined(__aarch64__) && defined(__linux__)
// GCC before 10 declares the CRC intrinsics only when the CRC extension is
// enabled for the whole translation unit; enable it for the header alone.
// HardwareRaw carries its own target attribute and runs only when the CPU
// reports the extension.
#if defined(__GNUC__) && !defined(__clang__) && !defined(__ARM_FEATURE_CRC32)
#pragma GCC push_options
#pragma GCC target("+crc")
#include <arm_acle.h>
#pragma GCC pop_options
#else
#include <arm_acle.h>
#endif
#include <sys/auxv.h>
#ifndef HWCAP_CRC32
#define HWCAP_CRC32 (1 << 7)
#endif
#endif

namespace dfkv {
namespace {

constexpr uint32_t kPoly = 0x82F63B78u;

constexpr std::array<std::array<uint32_t, 256>, 8> MakeTables() {
  std::array<std::array<uint32_t, 256>, 8> t{};
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t c = i;
    for (int k = 0; k < 8; ++k) c = (c & 1) ? (c >> 1) ^ kPoly : c >> 1;
    t[0][i] = c;
  }
  for (uint32_t i = 0; i < 256; ++i)
    for (int s = 1; s < 8; ++s)
      t[s][i] = (t[s - 1][i] >> 8) ^ t[0][t[s - 1][i] & 0xFF];
  return t;
}
constexpr auto kTables = MakeTables();

uint32_t SoftwareRaw(const unsigned char* p, size_t len, uint32_t c) {
  while (len >= 8) {
    const uint32_t lo = c ^ (uint32_t(p[0]) | uint32_t(p[1]) << 8 |
                             uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24);
    const uint32_t hi = uint32_t(p[4]) | uint32_t(p[5]) << 8 |
                        uint32_t(p[6]) << 16 | uint32_t(p[7]) << 24;
    c = kTables[7][lo & 0xFF] ^ kTables[6][(lo >> 8) & 0xFF] ^
        kTables[5][(lo >> 16) & 0xFF] ^ kTables[4][lo >> 24] ^
        kTables[3][hi & 0xFF] ^ kTables[2][(hi >> 8) & 0xFF] ^
        kTables[1][(hi >> 16) & 0xFF] ^ kTables[0][hi >> 24];
    p += 8;
    len -= 8;
  }
  while (len--) c = (c >> 8) ^ kTables[0][(c ^ *p++) & 0xFF];
  return c;
}

#if defined(__x86_64__)
__attribute__((target("sse4.2"))) uint32_t HardwareRaw(const unsigned char* p,
                                                        size_t len,
                                                        uint32_t c) {
  uint64_t c64 = c;
  while (len >= 8) {
    uint64_t v;
    std::memcpy(&v, p, 8);  // the instruction takes the bytes in memory order
    c64 = _mm_crc32_u64(c64, v);
    p += 8;
    len -= 8;
  }
  uint32_t c32 = static_cast<uint32_t>(c64);
  while (len--) c32 = _mm_crc32_u8(c32, *p++);
  return c32;
}
bool DetectHardware() {
  __builtin_cpu_init();  // may run before the runtime's own initialization
  return __builtin_cpu_supports("sse4.2");
}
#elif defined(__aarch64__) && defined(__linux__)
__attribute__((target("+crc"))) uint32_t HardwareRaw(const unsigned char* p,
                                                      size_t len, uint32_t c) {
  while (len >= 8) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    c = __crc32cd(c, v);
    p += 8;
    len -= 8;
  }
  while (len--) c = __crc32cb(c, *p++);
  return c;
}
bool DetectHardware() { return (getauxval(AT_HWCAP) & HWCAP_CRC32) != 0; }
#else
uint32_t HardwareRaw(const unsigned char* p, size_t len, uint32_t c) {
  return SoftwareRaw(p, len, c);
}
bool DetectHardware() { return false; }
#endif

const bool kHardware = DetectHardware();

}  // namespace

uint32_t Crc32cSoftware(const void* data, size_t len, uint32_t crc) {
  return ~SoftwareRaw(static_cast<const unsigned char*>(data), len, ~crc);
}

uint32_t Crc32c(const void* data, size_t len, uint32_t crc) {
  const auto* p = static_cast<const unsigned char*>(data);
  return ~(kHardware ? HardwareRaw(p, len, ~crc) : SoftwareRaw(p, len, ~crc));
}

bool Crc32cHardware() { return kHardware; }

}  // namespace dfkv
