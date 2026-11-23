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

#if defined(__x86_64__) || (defined(__aarch64__) && defined(__linux__))
// The CRC instruction has a latency of several cycles but can start one per
// cycle: one dependent chain runs at a third of its throughput. Long inputs
// are therefore cut into three blocks of the same length, each with its own
// chain, and the three are joined. The register is linear in the bytes:
// running it over A then B gives Shift_|B|(raw(c, A)) ^ raw(0, B), where
// Shift_n multiplies by x^(8n) modulo the polynomial, which is what running
// it over n zero bytes does. Shift_n is applied with four 256-entry tables
// (one per byte of the register), for the two block lengths used.
constexpr size_t kLongBlock = 8192;
constexpr size_t kShortBlock = 256;

struct ShiftTable {
  uint32_t t[4][256];
  uint32_t Apply(uint32_t c) const {
    return t[0][c & 0xFF] ^ t[1][(c >> 8) & 0xFF] ^ t[2][(c >> 16) & 0xFF] ^
           t[3][c >> 24];
  }
};

// Shift_n of every register value, built from the 32 single-bit values:
// each is the software loop run over n zero bytes.
ShiftTable MakeShift(size_t n) {
  static const unsigned char kZeros[kLongBlock] = {};
  uint32_t bit[32];
  for (int i = 0; i < 32; ++i) bit[i] = SoftwareRaw(kZeros, n, 1u << i);
  ShiftTable s{};
  for (int k = 0; k < 4; ++k)
    for (uint32_t b = 0; b < 256; ++b) {
      uint32_t v = 0;
      for (int i = 0; i < 8; ++i)
        if (b & (1u << i)) v ^= bit[8 * k + i];
      s.t[k][b] = v;
    }
  return s;
}

const ShiftTable& LongShift() {
  static const ShiftTable s = MakeShift(kLongBlock);
  return s;
}
const ShiftTable& ShortShift() {
  static const ShiftTable s = MakeShift(kShortBlock);
  return s;
}

// The three-chain loop over whole groups of three blocks of BLOCK bytes,
// with STEP(crc, 8 bytes) the instruction. A macro, not a function: it must
// be expanded inside the function compiled for the instruction set.
#define DFKV_CRC32C_THREE_CHAINS(STEP, BLOCK, SHIFT)                        \
  while (len >= 3 * (BLOCK)) {                                              \
    uint32_t c1 = 0, c2 = 0;                                                \
    const unsigned char* const end = p + (BLOCK);                           \
    do {                                                                    \
      uint64_t v0, v1, v2;                                                  \
      std::memcpy(&v0, p, 8);                                               \
      std::memcpy(&v1, p + (BLOCK), 8);                                     \
      std::memcpy(&v2, p + 2 * (BLOCK), 8);                                 \
      c = STEP(c, v0);                                                      \
      c1 = STEP(c1, v1);                                                    \
      c2 = STEP(c2, v2);                                                    \
      p += 8;                                                               \
    } while (p < end);                                                      \
    c = (SHIFT).Apply(c) ^ c1;                                              \
    c = (SHIFT).Apply(c) ^ c2;                                              \
    p += 2 * (BLOCK);                                                       \
    len -= 3 * (BLOCK);                                                     \
  }
#endif

#if defined(__x86_64__)
__attribute__((target("sse4.2"))) uint32_t HardwareRaw(const unsigned char* p,
                                                        size_t len,
                                                        uint32_t c) {
#define DFKV_CRC32C_STEP(crc, v) static_cast<uint32_t>(_mm_crc32_u64((crc), (v)))
  if (len >= 3 * kShortBlock) {
    const ShiftTable& long_shift = LongShift();
    const ShiftTable& short_shift = ShortShift();
    DFKV_CRC32C_THREE_CHAINS(DFKV_CRC32C_STEP, kLongBlock, long_shift)
    DFKV_CRC32C_THREE_CHAINS(DFKV_CRC32C_STEP, kShortBlock, short_shift)
  }
#undef DFKV_CRC32C_STEP
  uint64_t c64 = c;
  while (len >= 8) {
    uint64_t v;
    std::memcpy(&v, p, 8);
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
#define DFKV_CRC32C_STEP(crc, v) __crc32cd((crc), (v))
  if (len >= 3 * kShortBlock) {
    const ShiftTable& long_shift = LongShift();
    const ShiftTable& short_shift = ShortShift();
    DFKV_CRC32C_THREE_CHAINS(DFKV_CRC32C_STEP, kLongBlock, long_shift)
    DFKV_CRC32C_THREE_CHAINS(DFKV_CRC32C_STEP, kShortBlock, short_shift)
  }
#undef DFKV_CRC32C_STEP
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
