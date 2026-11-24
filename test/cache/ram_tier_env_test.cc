// The RAM tier's settings as the server reads them from the environment: a
// value that is empty or not a number must not be read as 0 where 0 has a
// meaning of its own.
#include "cache/kv_node_server.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "utils/env_parse.h"

namespace dfkv {
class KvNodeServerWiringTestPeer {
 public:
  struct Plan {
    bool enabled;
    uint32_t ack_high_watermark_pct;
    uint32_t reclaim_interval_ms;
    uint32_t flush_threads;
    uint64_t bytes;
    uint64_t large_reserve_bytes;
    bool promote_second;
    uint64_t promote_window_ms;
  };
  static size_t FlushLanes(size_t disk_count) {
    return KvNodeServer::PlanRamTier(disk_count).options.flush_lanes;
  }
  static Plan Read() {
    const auto plan = KvNodeServer::PlanRamTier(/*disk_count=*/1);
    return {plan.enabled, plan.options.ack_high_watermark_pct,
            plan.options.reclaim_interval_ms, plan.options.flush_threads,
            plan.options.bytes, plan.options.large_reserve_bytes,
            plan.promote_second, plan.promote_window_ms};
  }
};
}  // namespace dfkv

using namespace dfkv;  // NOLINT
using Peer = KvNodeServerWiringTestPeer;

namespace {

class RamTierEnv : public ::testing::Test {
 protected:
  void SetUp() override {
    Clear();
    ::setenv("DFKV_RAM_TIER", "on", 1);
    defaults_ = Peer::Read();
    ASSERT_TRUE(defaults_.enabled);
    // The cases below only tell something if 0 is not the default.
    ASSERT_NE(defaults_.ack_high_watermark_pct, 0u);
    ASSERT_NE(defaults_.reclaim_interval_ms, 0u);
  }
  void TearDown() override { Clear(); }
  static void Clear() {
    for (const char* name :
         {"DFKV_RAM_TIER", "DFKV_RAM_TIER_BYTES", "DFKV_RAM_FLUSH_THREADS",
          "DFKV_RAM_RECLAIM_MS", "DFKV_RAM_ACK_HIGH_WATERMARK_PCT",
          "DFKV_RAM_WRITE_MODE", "DFKV_PUT_ACK_MODE", "DFKV_RAM_PROMOTE",
          "DFKV_RAM_PROMOTE_WINDOW_MS"})
      ::unsetenv(name);
  }
  Peer::Plan defaults_{};
};

// Reads the plan with one variable set and returns what was logged for it.
struct Seen {
  Peer::Plan plan;
  std::string log;
};
Seen With(const char* name, const char* value) {
  envparse::ForgetWarningsForTest();
  ::setenv(name, value, 1);
  testing::internal::CaptureStderr();
  Seen seen;
  seen.plan = Peer::Read();
  seen.log = testing::internal::GetCapturedStderr();
  ::unsetenv(name);
  return seen;
}
bool Warned(const Seen& seen, const char* name, const char* value) {
  return seen.log.find(std::string("invalid ") + name + "='" + value + "'") !=
         std::string::npos;
}

// 0 = "acknowledge from disk". An empty or non-numeric value has always read
// as 0, so a unit file with an empty assignment runs in disk-ACK mode. It
// keeps doing so -- an upgrade must not switch the ACK mode -- and says it.
TEST_F(RamTierEnv, EmptyOrNonNumericWatermarkStaysZeroAndIsReported) {
  const char* const name = "DFKV_RAM_ACK_HIGH_WATERMARK_PCT";
  for (const char* bad : {"", "abc", "pct", " "}) {
    const Seen seen = With(name, bad);
    EXPECT_EQ(seen.plan.ack_high_watermark_pct, 0u) << "'" << bad << "'";
    EXPECT_TRUE(Warned(seen, name, bad)) << seen.log;
    EXPECT_NE(seen.log.find("using 0 as before"), std::string::npos) << seen.log;
    EXPECT_NE(seen.log.find("every PUT is acknowledged from disk"),
              std::string::npos)
        << seen.log;
    EXPECT_NE(seen.log.find("unset for the default " +
                            std::to_string(defaults_.ack_high_watermark_pct)),
              std::string::npos)
        << seen.log;
  }
  // What the reader's own range check threw away stays the default.
  for (const char* bad : {"-1", "101", "250%"}) {
    const Seen seen = With(name, bad);
    EXPECT_EQ(seen.plan.ack_high_watermark_pct,
              defaults_.ack_high_watermark_pct)
        << bad;
    EXPECT_TRUE(Warned(seen, name, bad)) << seen.log;
  }
  Seen seen = With(name, "60%");
  EXPECT_EQ(seen.plan.ack_high_watermark_pct, 60u);
  EXPECT_TRUE(Warned(seen, name, "60%")) << seen.log;
  // strtoul wraps a negative; one that lands in range was used.
  seen = With(name, "-18446744073709551561");  // 2^64 - 55
  EXPECT_EQ(seen.plan.ack_high_watermark_pct, 55u);
  EXPECT_TRUE(Warned(seen, name, "-18446744073709551561")) << seen.log;
  for (const char* clean : {"0", "55", "100"}) {
    seen = With(name, clean);
    EXPECT_EQ(seen.plan.ack_high_watermark_pct,
              static_cast<uint32_t>(std::atoi(clean)));
    EXPECT_EQ(seen.log.find(name), std::string::npos) << seen.log;
  }
}

// 0 = "no background reclaim": same story.
TEST_F(RamTierEnv, EmptyOrNonNumericReclaimIntervalStaysZeroAndIsReported) {
  const char* const name = "DFKV_RAM_RECLAIM_MS";
  for (const char* bad : {"", "abc", "ms"}) {
    const Seen seen = With(name, bad);
    EXPECT_EQ(seen.plan.reclaim_interval_ms, 0u) << "'" << bad << "'";
    EXPECT_TRUE(Warned(seen, name, bad)) << seen.log;
    EXPECT_NE(seen.log.find("no background reclaim"), std::string::npos)
        << seen.log;
  }
  // strtoul("-1") cut to 32 bits: a reclaim every 49 days. Kept, and the line
  // names the number that is used.
  Seen seen = With(name, "-1");
  EXPECT_EQ(seen.plan.reclaim_interval_ms, 4294967295u);
  EXPECT_TRUE(Warned(seen, name, "-1")) << seen.log;
  EXPECT_NE(seen.log.find("using 4294967295 as before"), std::string::npos)
      << seen.log;
  EXPECT_EQ(seen.log.find("18446744073709551615"), std::string::npos)
      << seen.log;
  // 2^32 is 0 in 32 bits: reclaim off, which nobody meant.
  seen = With(name, "4294967296");
  EXPECT_EQ(seen.plan.reclaim_interval_ms, 0u);
  EXPECT_TRUE(Warned(seen, name, "4294967296")) << seen.log;
  EXPECT_NE(seen.log.find("does not fit 32 bits; using 0 as before"),
            std::string::npos)
      << seen.log;
  EXPECT_NE(seen.log.find("no background reclaim"), std::string::npos)
      << seen.log;
  seen = With(name, "4294967295");
  EXPECT_EQ(seen.plan.reclaim_interval_ms, 4294967295u);
  EXPECT_EQ(seen.log.find(name), std::string::npos) << seen.log;
  seen = With(name, "25ms");
  EXPECT_EQ(seen.plan.reclaim_interval_ms, 25u);
  EXPECT_TRUE(Warned(seen, name, "25ms")) << seen.log;
  for (const char* clean : {"0", "25"}) {
    seen = With(name, clean);
    EXPECT_EQ(seen.plan.reclaim_interval_ms,
              static_cast<uint32_t>(std::atoi(clean)));
    EXPECT_EQ(seen.log.find(name), std::string::npos) << seen.log;
  }
}

// Here 0 was thrown away by the reader, so every non-number was the default.
TEST_F(RamTierEnv, SizeAndThreadCountIgnoreWhatIsNotANumber) {
  for (const char* bad : {"abc", "0", ""}) {
    Seen seen = With("DFKV_RAM_TIER_BYTES", bad);
    EXPECT_EQ(seen.plan.bytes, defaults_.bytes) << bad;
    EXPECT_EQ(Warned(seen, "DFKV_RAM_TIER_BYTES", bad), *bad != '\0')
        << seen.log;
    seen = With("DFKV_RAM_FLUSH_THREADS", bad);
    EXPECT_EQ(seen.plan.flush_threads, defaults_.flush_threads) << bad;
    EXPECT_EQ(Warned(seen, "DFKV_RAM_FLUSH_THREADS", bad), *bad != '\0')
        << seen.log;
  }
  EXPECT_EQ(With("DFKV_RAM_TIER_BYTES", "1073741824").plan.bytes, 1073741824u);
  EXPECT_EQ(With("DFKV_RAM_FLUSH_THREADS", "8").plan.flush_threads, 8u);
  EXPECT_EQ(With("DFKV_RAM_FLUSH_THREADS", "65").plan.flush_threads,
            defaults_.flush_threads);
  // "64G" has been 64 bytes; it still is, and now it is visible.
  Seen seen = With("DFKV_RAM_TIER_BYTES", "64G");
  EXPECT_EQ(seen.plan.bytes, 64u);
  EXPECT_TRUE(Warned(seen, "DFKV_RAM_TIER_BYTES", "64G")) << seen.log;
  // strtoull wraps a negative: -1 has been an arena of 2^64-1 bytes, and a
  // negative that lands in range has been that number.
  seen = With("DFKV_RAM_TIER_BYTES", "-1");
  EXPECT_EQ(seen.plan.bytes, 18446744073709551615ull);
  EXPECT_TRUE(Warned(seen, "DFKV_RAM_TIER_BYTES", "-1")) << seen.log;
  seen = With("DFKV_RAM_FLUSH_THREADS", "-1");
  EXPECT_EQ(seen.plan.flush_threads, defaults_.flush_threads);
  EXPECT_TRUE(Warned(seen, "DFKV_RAM_FLUSH_THREADS", "-1")) << seen.log;
  seen = With("DFKV_RAM_FLUSH_THREADS", "-18446744073709551611");  // 2^64 - 5
  EXPECT_EQ(seen.plan.flush_threads, 5u);
  EXPECT_TRUE(Warned(seen, "DFKV_RAM_FLUSH_THREADS", "-18446744073709551611"))
      << seen.log;
}

// The words that turn the tier on are lower case, as they always were: a
// start that runs without the tier keeps running without it, and says why.
TEST_F(RamTierEnv, TierSwitchKeepsItsLowerCaseWords) {
  for (const char* on : {"1", "on", "true", "yes"}) {
    const Seen seen = With("DFKV_RAM_TIER", on);
    EXPECT_TRUE(seen.plan.enabled) << on;
    EXPECT_EQ(seen.log.find("DFKV_RAM_TIER="), std::string::npos) << seen.log;
  }
  for (const char* off : {"0", "off", "false", "no", ""}) {
    const Seen seen = With("DFKV_RAM_TIER", off);
    EXPECT_FALSE(seen.plan.enabled) << off;
    EXPECT_EQ(seen.log.find("DFKV_RAM_TIER="), std::string::npos) << seen.log;
  }
  for (const char* other : {"ON", "True", "2"}) {
    const Seen seen = With("DFKV_RAM_TIER", other);
    EXPECT_FALSE(seen.plan.enabled) << other;
    EXPECT_TRUE(Warned(seen, "DFKV_RAM_TIER", other)) << seen.log;
    EXPECT_NE(seen.log.find("stays off"), std::string::npos) << seen.log;
  }
}

}  // namespace

// The flush lanes are the configured directories, as the flush threads: the
// RAM tier is built while the disk group is still being opened, so its
// size must not be read from the group (it read an unconstructed one).
TEST_F(RamTierEnv, FlushLanesAreTheConfiguredDisks) {
  EXPECT_EQ(Peer::FlushLanes(3), 3u);
  EXPECT_EQ(Peer::FlushLanes(1), 1u);
}

// The server keeps no dedicated budget for values above an extent unless
// asked: with the default --max-msg no value can use it.
TEST_F(RamTierEnv, NoDedicatedBudgetByDefault) {
  EXPECT_EQ(defaults_.large_reserve_bytes, 0u);
}

// DFKV_RAM_PROMOTE: first (every whole cold read, as before) unless the
// value is exactly "second"; the window is read only for second.
TEST_F(RamTierEnv, PromoteIsFirstUnlessSecondIsAskedFor) {
  EXPECT_FALSE(defaults_.promote_second);
  EXPECT_EQ(defaults_.promote_window_ms, 30000u);
  Seen seen = With("DFKV_RAM_PROMOTE", "second");
  EXPECT_TRUE(seen.plan.promote_second);
  EXPECT_EQ(seen.plan.promote_window_ms, 30000u);
  EXPECT_EQ(seen.log.find("DFKV_RAM_PROMOTE"), std::string::npos) << seen.log;
  seen = With("DFKV_RAM_PROMOTE", "first");
  EXPECT_FALSE(seen.plan.promote_second);
  EXPECT_EQ(seen.log.find("DFKV_RAM_PROMOTE"), std::string::npos) << seen.log;
  for (const char* bad : {"Second", "2", "twice", " second"}) {
    seen = With("DFKV_RAM_PROMOTE", bad);
    EXPECT_FALSE(seen.plan.promote_second) << bad;
    EXPECT_TRUE(Warned(seen, "DFKV_RAM_PROMOTE", bad)) << seen.log;
  }
}

TEST_F(RamTierEnv, PromoteWindowKeepsItsDefaultForWhatIsNotAWindow) {
  ::setenv("DFKV_RAM_PROMOTE", "second", 1);
  Seen seen = With("DFKV_RAM_PROMOTE_WINDOW_MS", "5000");
  EXPECT_EQ(seen.plan.promote_window_ms, 5000u);
  EXPECT_EQ(seen.log.find("DFKV_RAM_PROMOTE_WINDOW_MS"), std::string::npos)
      << seen.log;
  seen = With("DFKV_RAM_PROMOTE_WINDOW_MS", "");  // empty counts as unset
  EXPECT_EQ(seen.plan.promote_window_ms, 30000u);
  for (const char* bad : {"0", "abc", "-1", "5s"}) {
    seen = With("DFKV_RAM_PROMOTE_WINDOW_MS", bad);
    EXPECT_EQ(seen.plan.promote_window_ms, 30000u) << "'" << bad << "'";
    EXPECT_NE(seen.log.find("DFKV_RAM_PROMOTE_WINDOW_MS"), std::string::npos)
        << "'" << bad << "' " << seen.log;
  }
  ::unsetenv("DFKV_RAM_PROMOTE");
}
