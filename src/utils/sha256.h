/* Small dependency-free SHA-256 used for cache identities.
 * The API is deliberately one-shot: keys are short and callers already own a
 * contiguous canonical descriptor. Digest words are returned as standard
 * big-endian SHA-256 bytes; BlockKey decoding defines its own byte order.
 */
#ifndef DFKV_SHA256_H_
#define DFKV_SHA256_H_

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace dfkv {
namespace sha256_detail {

inline uint32_t RotR(uint32_t value, unsigned bits) {
  return (value >> bits) | (value << (32 - bits));
}

inline uint32_t LoadBe32(const uint8_t* p) {
  return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
         (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

inline void StoreBe32(uint8_t* p, uint32_t value) {
  p[0] = static_cast<uint8_t>(value >> 24);
  p[1] = static_cast<uint8_t>(value >> 16);
  p[2] = static_cast<uint8_t>(value >> 8);
  p[3] = static_cast<uint8_t>(value);
}

inline void Compress(uint32_t state[8], const uint8_t block[64]) {
  static constexpr uint32_t k[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
      0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
      0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
      0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
      0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
      0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
      0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
      0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
      0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
      0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
      0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
      0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
      0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
  uint32_t w[64];
  for (size_t i = 0; i < 16; ++i) w[i] = LoadBe32(block + i * 4);
  for (size_t i = 16; i < 64; ++i) {
    const uint32_t s0 = RotR(w[i - 15], 7) ^ RotR(w[i - 15], 18) ^
                        (w[i - 15] >> 3);
    const uint32_t s1 = RotR(w[i - 2], 17) ^ RotR(w[i - 2], 19) ^
                        (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }

  uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
  uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
  for (size_t i = 0; i < 64; ++i) {
    const uint32_t s1 = RotR(e, 6) ^ RotR(e, 11) ^ RotR(e, 25);
    const uint32_t ch = (e & f) ^ (~e & g);
    const uint32_t t1 = h + s1 + ch + k[i] + w[i];
    const uint32_t s0 = RotR(a, 2) ^ RotR(a, 13) ^ RotR(a, 22);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t t2 = s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}

}  // namespace sha256_detail

inline void Sha256(const void* data, size_t size, uint8_t out[32]) {
  uint32_t state[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                       0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                       0x1f83d9abu, 0x5be0cd19u};
  const auto* bytes = static_cast<const uint8_t*>(data);
  size_t offset = 0;
  while (size - offset >= 64) {
    sha256_detail::Compress(state, bytes + offset);
    offset += 64;
  }

  uint8_t final_blocks[128]{};
  const size_t tail = size - offset;
  if (tail != 0) std::memcpy(final_blocks, bytes + offset, tail);
  final_blocks[tail] = 0x80;
  const size_t padded = tail < 56 ? 64 : 128;
  const uint64_t bit_size = static_cast<uint64_t>(size) * 8;
  for (size_t i = 0; i < 8; ++i) {
    final_blocks[padded - 1 - i] = static_cast<uint8_t>(bit_size >> (8 * i));
  }
  sha256_detail::Compress(state, final_blocks);
  if (padded == 128) sha256_detail::Compress(state, final_blocks + 64);
  for (size_t i = 0; i < 8; ++i) sha256_detail::StoreBe32(out + 4 * i, state[i]);
}

}  // namespace dfkv

#endif  // DFKV_SHA256_H_
