// CRC-32C: published check values, and the hardware and software paths agree.
#include "utils/crc32c.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
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
  for (size_t len : {size_t{0}, size_t{1}, size_t{7}, size_t{8}, size_t{9},
                     size_t{63}, size_t{4096}, size_t{4099}, size_t{70000}}) {
    for (size_t off : {size_t{0}, size_t{1}, size_t{3}}) {
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

// The three-chain path (inputs of 768 bytes and more): bit-identical to the
// software loop at every start offset 0..7, for every length up to one
// group of long blocks and a little more (3 * 8192 + 64), then every 61st
// length and the block boundaries up to 64 KiB. The reference is computed
// incrementally (continued over the bytes added since the last length).
TEST(Crc32c, EveryLengthAndOffsetMatchesTheSoftwareLoop) {
  constexpr size_t kMax = 64 * 1024;
  constexpr size_t kDense = 3 * 8192 + 64;
  std::vector<size_t> lengths;
  for (size_t len = 0; len <= kDense; ++len) lengths.push_back(len);
  for (size_t len = kDense + 61; len <= kMax; len += 61) lengths.push_back(len);
  for (size_t m = 2; m * 3 * 8192 <= kMax + 3 * 8192; ++m)
    for (size_t d : {size_t{0}, size_t{1}, size_t{7}, size_t{8}})
      for (size_t base : {m * 3 * 8192 - d, m * 3 * 8192 + d})
        if (base > kDense && base <= kMax) lengths.push_back(base);
  lengths.push_back(kMax);
  std::sort(lengths.begin(), lengths.end());
  lengths.erase(std::unique(lengths.begin(), lengths.end()), lengths.end());
  std::mt19937_64 rng(11);
  std::vector<uint8_t> buf(kMax + 8);
  for (auto& b : buf) b = static_cast<uint8_t>(rng());
  for (size_t off = 0; off < 8; ++off) {
    uint32_t want = 0;  // Crc32cSoftware(buf + off, done)
    size_t done = 0;
    for (size_t len : lengths) {
      want = Crc32cSoftware(&buf[off + done], len - done, want);
      done = len;
      const uint32_t got = Crc32c(&buf[off], len);
      if (got != want) {
        ADD_FAILURE() << "length " << len << " at offset " << off;
        return;
      }
    }
  }
}

// Large and odd sizes, the value sizes of the log engine among them, and
// continued CRCs split anywhere (inside a block, at a block boundary).
TEST(Crc32c, LargeBuffersAndContinuedCrcsMatchTheSoftwareLoop) {
  std::mt19937_64 rng(13);
  std::vector<uint8_t> buf(27158118 + 16);
  for (auto& b : buf) b = static_cast<uint8_t>(rng());
  for (size_t len : {size_t{1} << 20, size_t{1069548}, size_t{2873098},
                     size_t{27158118}, size_t{3 * 8192}, size_t{3 * 8192 + 1},
                     size_t{3 * 8192 - 1}, size_t{3 * 256 + 7}, size_t{999983}}) {
    for (size_t off : {size_t{0}, size_t{1}, size_t{5}, size_t{7}}) {
      SCOPED_TRACE(std::to_string(len) + "@" + std::to_string(off));
      const uint32_t want = Crc32cSoftware(&buf[off], len);
      EXPECT_EQ(Crc32c(&buf[off], len), want);
      for (size_t cut : {size_t{0}, size_t{1}, size_t{8191}, size_t{8192},
                         size_t{3 * 8192}, len / 3, len / 2, len - 1, len}) {
        if (cut > len) continue;
        EXPECT_EQ(Crc32c(&buf[off + cut], len - cut, Crc32c(&buf[off], cut)), want)
            << "split at " << cut;
      }
      // A seeded start (a previous result as the initial value).
      const uint32_t seed = static_cast<uint32_t>(rng());
      EXPECT_EQ(Crc32c(&buf[off], len, seed), Crc32cSoftware(&buf[off], len, seed));
    }
  }
}

// Throughput over a 1 MiB value, reported only (no timing assertion).
TEST(Crc32c, Throughput) {
  std::vector<uint8_t> buf(1069548, 0x5a);
  uint32_t acc = 0;
  const auto t0 = std::chrono::steady_clock::now();
  constexpr int kRounds = 500;
  for (int i = 0; i < kRounds; ++i) acc ^= Crc32c(buf.data(), buf.size(), acc);
  const double s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  RecordProperty("us_per_mib_value", static_cast<int>(s / kRounds * 1e6));
  RecordProperty("gbps_x100", static_cast<int>(kRounds * buf.size() / s / 1e7));
  volatile uint32_t sink = acc;  // keeps the loop
  (void)sink;
}
