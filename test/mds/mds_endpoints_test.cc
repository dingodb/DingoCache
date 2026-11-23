#include "mds/mds_endpoints.h"
#include <gtest/gtest.h>
#include <set>
using namespace dfkv;  // NOLINT

TEST(MdsEndpoints, RoundRobinSpreadsAcrossAll) {
  MdsEndpoints ep({"m1", "m2", "m3"});
  std::set<std::string> seen;
  for (int i = 0; i < 3; ++i) seen.insert(ep.Pick(/*now_ms=*/0));
  EXPECT_EQ(seen.size(), 3u);  // all three handed out, no stickiness
}

TEST(MdsEndpoints, FailedEndpointSkippedDuringBackoff) {
  MdsEndpoints ep({"m1", "m2"}, /*backoff_ms=*/2000);
  ep.MarkFailed("m1", /*now_ms=*/1000);
  // within backoff window: only m2 is handed out
  for (int i = 0; i < 5; ++i) EXPECT_EQ(ep.Pick(/*now_ms=*/1500), "m2");
  // after backoff expires: m1 eligible again
  std::set<std::string> seen;
  for (int i = 0; i < 2; ++i) seen.insert(ep.Pick(/*now_ms=*/3001));
  EXPECT_EQ(seen.count("m1"), 1u);
}

TEST(MdsEndpoints, AllInBackoffDegradesToSoonestExpiry) {
  MdsEndpoints ep({"m1", "m2"}, 2000);
  ep.MarkFailed("m1", 1000);  // until 3000
  ep.MarkFailed("m2", 1500);  // until 3500
  EXPECT_EQ(ep.Pick(/*now_ms=*/2000), "m1");  // soonest to recover, never empty
}

TEST(MdsEndpoints, MarkOkClearsBackoff) {
  MdsEndpoints ep({"m1", "m2"}, 2000);
  ep.MarkFailed("m1", 1000);
  ep.MarkOk("m1");
  std::set<std::string> seen;
  for (int i = 0; i < 2; ++i) seen.insert(ep.Pick(/*now_ms=*/1100));
  EXPECT_EQ(seen.count("m1"), 1u);
}

TEST(MdsEndpoints, EmptyListReturnsEmpty) {
  MdsEndpoints ep({});
  EXPECT_TRUE(ep.Pick(0).empty());
}

TEST(MdsEndpoints, StickySelectionStartsWhereSeeded) {
  MdsEndpoints eps({"a", "b", "c"}, 30000, /*sticky=*/true);
  eps.SeedCursor(4);  // 4 % 3 == 1
  EXPECT_EQ(eps.Pick(0), "b");
  EXPECT_EQ(eps.Pick(0), "b");
  eps.MarkFailed("b", 0);
  EXPECT_EQ(eps.Pick(0), "c");
}
