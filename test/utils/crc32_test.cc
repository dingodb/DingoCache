// The table-driven CRC-32 must produce exactly the values of the bit-at-a-time
// loop it replaced: slab table records and slab_state written by any release
// are verified with it.
#include "utils/crc32.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace {

// The loop DiskSlabStore used through release 2.37.0.
uint32_t BitwiseCrc32(const uint8_t* p, size_t n) {
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < n; ++i) {
    crc ^= p[i];
    for (int k = 0; k < 8; ++k)
      crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
  }
  return crc ^ 0xFFFFFFFFu;
}

TEST(Crc32Test, KnownVectors) {
  const char* check = "123456789";
  EXPECT_EQ(dfkv::Crc32(reinterpret_cast<const uint8_t*>(check), 9),
            0xCBF43926u);
  EXPECT_EQ(dfkv::Crc32(nullptr, 0), 0u);
  const uint8_t zero = 0;
  EXPECT_EQ(dfkv::Crc32(&zero, 1), 0xD202EF8Du);
}

TEST(Crc32Test, EqualsTheBitwiseLoopOnRandomInputs) {
  std::mt19937_64 rng(20261122);
  for (size_t len = 0; len <= 200; ++len) {
    for (int round = 0; round < 64; ++round) {
      // One leading byte of slack so every alignment of the data is covered.
      std::vector<uint8_t> buf(len + 8);
      for (auto& b : buf) b = static_cast<uint8_t>(rng());
      const size_t shift = static_cast<size_t>(round) % 8;
      ASSERT_EQ(dfkv::Crc32(buf.data() + shift, len),
                BitwiseCrc32(buf.data() + shift, len))
          << "len " << len << " shift " << shift;
    }
  }
}

// The two record shapes the slab store checksums: 56 bytes of a 64-byte table
// record and 60 bytes of slab_state, including the all-zero ones.
TEST(Crc32Test, EqualsTheBitwiseLoopOnRecordSizedInputs) {
  std::mt19937_64 rng(7);
  for (size_t len : {size_t{56}, size_t{60}, size_t{64}, size_t{4096}}) {
    std::vector<uint8_t> buf(len, 0);
    EXPECT_EQ(dfkv::Crc32(buf.data(), len), BitwiseCrc32(buf.data(), len));
    std::memset(buf.data(), 0xFF, len);
    EXPECT_EQ(dfkv::Crc32(buf.data(), len), BitwiseCrc32(buf.data(), len));
    for (int round = 0; round < 1000; ++round) {
      for (auto& b : buf) b = static_cast<uint8_t>(rng());
      ASSERT_EQ(dfkv::Crc32(buf.data(), len), BitwiseCrc32(buf.data(), len));
    }
  }
}

}  // namespace
