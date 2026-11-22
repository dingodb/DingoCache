// CRC-32C: published check values, and the hardware and software paths agree.
#include "utils/crc32c.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using dfkv::Crc32c;
using dfkv::Crc32cSoftware;

TEST(Crc32c, PublishedCheckValues) {
  EXPECT_EQ(Crc32c("123456789", 9), 0xE3069283u);
  // iSCSI test vectors (RFC 3720, B.4).
  std::vector<uint8_t> zeros(32, 0x00), ones(32, 0xFF), up(32), down(32);
  for (int i = 0; i < 32; ++i) {
    up[i] = static_cast<uint8_t>(i);
    down[i] = static_cast<uint8_t>(31 - i);
  }
  EXPECT_EQ(Crc32c(zeros.data(), 32), 0x8A9136AAu);
  EXPECT_EQ(Crc32c(ones.data(), 32), 0x62A8AB43u);
  EXPECT_EQ(Crc32c(up.data(), 32), 0x46DD794Eu);
  EXPECT_EQ(Crc32c(down.data(), 32), 0x113FDB5Cu);
  EXPECT_EQ(Crc32c(nullptr, 0), 0u);
}

TEST(Crc32c, HardwareAndSoftwareAgreeAndContinue) {
  std::mt19937_64 rng(7);
  std::vector<uint8_t> buf(70000);
  for (auto& b : buf) b = static_cast<uint8_t>(rng());
  for (size_t len : {0ul, 1ul, 7ul, 8ul, 9ul, 63ul, 4096ul, 4099ul, 70000ul}) {
    for (size_t off : {0ul, 1ul, 3ul}) {
      if (off + len > buf.size()) continue;
      SCOPED_TRACE(std::to_string(len) + "@" + std::to_string(off));
      EXPECT_EQ(Crc32c(buf.data() + off, len),
                Crc32cSoftware(buf.data() + off, len));
      const size_t half = len / 2;
      EXPECT_EQ(Crc32c(buf.data() + off + half, len - half,
                       Crc32c(buf.data() + off, half)),
                Crc32c(buf.data() + off, len));
    }
  }
  // Which path ran is reported, not asserted: both must be right.
  RecordProperty("hardware", dfkv::Crc32cHardware() ? "yes" : "no");
}

// CI sets this on machines known to have the instruction (the arm64 job), so
// the hardware path is compiled and run there, not only the fallback.
TEST(Crc32c, HardwarePathRunsWhereExpected) {
  const char* expect = std::getenv("DFKV_TEST_EXPECT_CRC32C_HARDWARE");
  if (expect == nullptr || std::string(expect) != "1")
    GTEST_SKIP() << "DFKV_TEST_EXPECT_CRC32C_HARDWARE is not 1";
  EXPECT_TRUE(dfkv::Crc32cHardware());
}
