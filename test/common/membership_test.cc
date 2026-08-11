#include "common/membership.h"
#include <gtest/gtest.h>
using namespace dfkv;  // NOLINT

TEST(MembershipCodec, RoundTripMultiMember) {
  std::vector<MemberInfo> ms = {
      {"n1", "10.0.0.1", 28000, 1},
      {"n2", "10.0.0.2", 28000, 3},
      {"node-with-long-id", "192.168.123.234", 65535, 100},
  };
  std::string buf = EncodeMembers(ms, /*epoch=*/0xDEADBEEF12345678ULL);
  std::vector<MemberInfo> got;
  uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(buf.data(), buf.size(), &got, &epoch));
  EXPECT_EQ(epoch, 0xDEADBEEF12345678ULL);
  EXPECT_EQ(got, ms);
}

TEST(MembershipCodec, RoundTripEmpty) {
  std::string buf = EncodeMembers({}, 42);
  std::vector<MemberInfo> got{{"stale", "x", 1, 1}};
  uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(buf.data(), buf.size(), &got, &epoch));
  EXPECT_EQ(epoch, 42u);
  EXPECT_TRUE(got.empty());
}

TEST(MembershipCodec, RejectsTruncated) {
  std::vector<MemberInfo> ms = {{"n1", "10.0.0.1", 28000, 1, 28100}};  // tcp_port=28100
  std::string buf = EncodeMembers(ms, 1);
  std::vector<MemberInfo> got;
  uint64_t epoch = 0;
  // The mandatory member region must be complete to decode; the OPTIONAL tcp_port
  // tail may be short/absent (older peers send none) -> tolerated, not an error.
  // Discover the region boundary, then require failure below it and success at/above.
  size_t lo = 0;
  while (lo <= buf.size() && !DecodeMembers(buf.data(), lo, &got, &epoch)) ++lo;
  ASSERT_LE(lo, buf.size());
  for (size_t cut = 0; cut < lo; ++cut)
    EXPECT_FALSE(DecodeMembers(buf.data(), cut, &got, &epoch)) << "cut=" << cut;
  for (size_t cut = lo; cut <= buf.size(); ++cut)
    EXPECT_TRUE(DecodeMembers(buf.data(), cut, &got, &epoch)) << "cut=" << cut;
  // Tail absent at the boundary -> tcp_port defaults to 0; fully present -> read back.
  ASSERT_TRUE(DecodeMembers(buf.data(), lo, &got, &epoch));
  EXPECT_EQ(got[0].tcp_port, 0u);
  ASSERT_TRUE(DecodeMembers(buf.data(), buf.size(), &got, &epoch));
  EXPECT_EQ(got[0].tcp_port, 28100u);
}

TEST(MembersEpoch, OrderIndependentAndContentSensitive) {
  std::vector<MemberInfo> a = {
      {"n1", "10.0.0.1", 28000, 1},
      {"n2", "10.0.0.2", 28000, 3},
      {"n3", "10.0.0.3", 28000, 1},
  };
  std::vector<MemberInfo> reordered = {a[2], a[0], a[1]};
  EXPECT_EQ(MembersEpoch(a), MembersEpoch(reordered)) << "order must not matter";

  // Removing a member changes the epoch.
  std::vector<MemberInfo> fewer = {a[0], a[1]};
  EXPECT_NE(MembersEpoch(a), MembersEpoch(fewer));

  // A content-only change (same ids) changes the epoch.
  std::vector<MemberInfo> b = a;
  b[1].weight = 7;  // was 3
  EXPECT_NE(MembersEpoch(a), MembersEpoch(b));
  std::vector<MemberInfo> c = a;
  c[0].ip = "10.0.0.9";
  EXPECT_NE(MembersEpoch(a), MembersEpoch(c));
  std::vector<MemberInfo> d = a;
  d[2].port = 28001;
  EXPECT_NE(MembersEpoch(a), MembersEpoch(d));

  // Stable across calls; empty set is deterministic.
  EXPECT_EQ(MembersEpoch(a), MembersEpoch(a));
  EXPECT_EQ(MembersEpoch({}), MembersEpoch({}));
}

TEST(MembersEpoch, NoFieldBoundaryCollision) {
  // Length-prefixing must prevent "ab"+"c" hashing the same as "a"+"bc".
  std::vector<MemberInfo> x = {{"ab", "c", 0, 0}};
  std::vector<MemberInfo> y = {{"a", "bc", 0, 0}};
  EXPECT_NE(MembersEpoch(x), MembersEpoch(y));
}

TEST(Membership, ValidGroupOrIdAlphabet) {
  // Accepted: the tokens real deployments use.
  EXPECT_TRUE(IsValidGroupOrId("glm"));
  EXPECT_TRUE(IsValidGroupOrId("glm-a"));
  EXPECT_TRUE(IsValidGroupOrId("gpu2-0045"));
  EXPECT_TRUE(IsValidGroupOrId("itest-grp-12345"));
  EXPECT_TRUE(IsValidGroupOrId("a.b_c-1"));
  EXPECT_TRUE(IsValidGroupOrId(std::string(128, 'a')));  // max length

  // Rejected: empty, overlong, and anything that could escape the key subtree.
  EXPECT_FALSE(IsValidGroupOrId(""));
  EXPECT_FALSE(IsValidGroupOrId(std::string(129, 'a')));
  EXPECT_FALSE(IsValidGroupOrId("a/members/ghost"));  // path traversal
  EXPECT_FALSE(IsValidGroupOrId("a/b"));
  EXPECT_FALSE(IsValidGroupOrId("a b"));   // space
  EXPECT_FALSE(IsValidGroupOrId("a\tb"));
  EXPECT_FALSE(IsValidGroupOrId("a:b"));
  EXPECT_FALSE(IsValidGroupOrId(std::string("a\0b", 3)));  // embedded NUL
}

// --- info extension (NFO1): node self-description on register/heartbeat -------

TEST(MembershipCodec, InfoRoundTripsAlongsideTcpPort) {
  std::vector<MemberInfo> ms = {
      {"n1", "10.0.0.1", 28001, 1, 28100, "ver=1.8.0,engine=slab,cap=5497558138880"},
      {"n2", "10.0.0.2", 28001, 3, 28100, ""},  // one member may have no info
  };
  std::string buf = EncodeMembers(ms, 7);
  std::vector<MemberInfo> got;
  uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(buf.data(), buf.size(), &got, &epoch));
  ASSERT_EQ(got.size(), 2u);
  EXPECT_EQ(got[0].info, "ver=1.8.0,engine=slab,cap=5497558138880");
  EXPECT_EQ(got[0].tcp_port, 28100u);  // TCP1 ext still decoded before NFO1
  EXPECT_EQ(got[1].info, "");
}

TEST(MembershipCodec, OldEncodingWithoutExtsDecodesWithEmptyInfo) {
  // Hand-build a pre-extension encoding (fixed part only): older peers send this.
  MemberInfo m{"n1", "10.0.0.9", 28001, 2};
  std::string full = EncodeMembers({m}, 3);
  size_t fixed = 12 + (4 + m.id.size()) + (4 + m.ip.size()) + 8;  // header + one member
  std::string oldbuf = full.substr(0, fixed);
  std::vector<MemberInfo> got;
  uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(oldbuf.data(), oldbuf.size(), &got, &epoch));
  ASSERT_EQ(got.size(), 1u);
  EXPECT_EQ(got[0].tcp_port, 0u);
  EXPECT_EQ(got[0].info, "");
}

TEST(MembershipCodec, UnknownFutureExtTagIsIgnored) {
  MemberInfo m{"n1", "10.0.0.9", 28001, 2, 28100, "ver=1.8.0"};
  std::string buf = EncodeMembers({m}, 3);
  buf += std::string("\xAA\xBB\xCC\xDD garbage-future-extension", 30);
  std::vector<MemberInfo> got;
  uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(buf.data(), buf.size(), &got, &epoch));
  EXPECT_EQ(got[0].info, "ver=1.8.0");  // known exts read, future tail ignored
}

TEST(MembershipCodec, TruncatedInfoExtIsBestEffortNotFatal) {
  MemberInfo m{"n1", "10.0.0.9", 28001, 2, 28100, "ver=1.8.0,engine=slab"};
  std::string buf = EncodeMembers({m}, 3);
  // Cut inside the NFO1 payload: members + tcp_port must survive, decode still ok.
  std::string cut = buf.substr(0, buf.size() - 5);
  std::vector<MemberInfo> got;
  uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(cut.data(), cut.size(), &got, &epoch));
  ASSERT_EQ(got.size(), 1u);
  EXPECT_EQ(got[0].id, "n1");
  EXPECT_EQ(got[0].tcp_port, 28100u);
}

TEST(MembersEpoch, InfoIsExcludedFromEpochAndEquality) {
  // info (like tcp_port) is orthogonal metadata: a change must NOT bump the
  // epoch (no needless client ring rebuilds) and must not affect operator==.
  std::vector<MemberInfo> a = {{"n1", "10.0.0.1", 28001, 1, 28100, "ver=1.7.2"}};
  std::vector<MemberInfo> b = {{"n1", "10.0.0.1", 28001, 1, 28100, "ver=1.8.0,engine=slab"}};
  EXPECT_EQ(MembersEpoch(a), MembersEpoch(b));
  EXPECT_EQ(a[0], b[0]);
}

// STA1 dynamic-stats extension: round-trip, per-member presence, legacy
// absence, forward-skip of unknown appended fields, and the epoch/equality
// exclusion that keeps heartbeat churn from rebuilding client rings.
TEST(Membership, StatsRoundTripWithMixedPresence) {
  MemberInfo a{"n1", "10.0.0.1", 28001, 1, 28100, "ver=1.10.0"};
  a.stats.capacity_bytes = 5ull << 40;
  a.stats.used_bytes = 123456789;
  a.stats.objects = 42;
  a.stats.hits_total = 1000;
  a.stats.misses_total = 50;
  a.stats.evictions_total = 7;
  a.stats.puts_total = 1100;
  a.stats.uptime_seconds = 3600;
  a.stats.put_busy_total = 3;
  a.stats.dio_write_fallbacks = 1;
  a.stats.ram_used_bytes = 1 << 30;
  a.stats.ram_hits_total = 900;
  a.has_stats = true;
  MemberInfo b{"n2", "10.0.0.2", 28001, 3, 28100, ""};  // no stats (pre-upgrade)
  std::string enc = EncodeMembers({a, b}, 5);
  std::vector<MemberInfo> out;
  uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(enc.data(), enc.size(), &out, &epoch));
  ASSERT_EQ(out.size(), 2u);
  ASSERT_TRUE(out[0].has_stats);
  EXPECT_EQ(out[0].stats.capacity_bytes, 5ull << 40);
  EXPECT_EQ(out[0].stats.used_bytes, 123456789u);
  EXPECT_EQ(out[0].stats.objects, 42u);
  EXPECT_EQ(out[0].stats.hits_total, 1000u);
  EXPECT_EQ(out[0].stats.ram_hits_total, 900u);
  EXPECT_EQ(out[0].stats.uptime_seconds, 3600u);
  EXPECT_FALSE(out[1].has_stats);
}

TEST(Membership, StatsForwardSkipUnknownAppendedFields) {
  // A FUTURE encoder appends fields: emulate by patching nfields up and
  // appending extra u64s per present member. Today's decoder must read its
  // known fields and skip the tail cleanly.
  MemberInfo a{"n1", "10.0.0.1", 28001, 1, 28100, ""};
  a.stats.capacity_bytes = 77;
  a.has_stats = true;
  std::string enc = EncodeMembers({a}, 1);
  // Locate STA1: it is the LAST extension; nfields byte follows the tag.
  const uint32_t tag = 0x31415453u;
  size_t pos = std::string::npos;
  for (size_t i = 0; i + 4 <= enc.size(); ++i) {
    uint32_t v;
    std::memcpy(&v, enc.data() + i, 4);
    if (v == tag) pos = i;
  }
  ASSERT_NE(pos, std::string::npos);
  enc[pos + 4] = static_cast<char>(kMemberStatsFields + 2);  // future: 2 extra fields
  char extra[16] = {0};
  enc.append(extra, 16);  // the 2 extra u64s for the one present member
  std::vector<MemberInfo> out;
  uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(enc.data(), enc.size(), &out, &epoch));
  ASSERT_EQ(out.size(), 1u);
  ASSERT_TRUE(out[0].has_stats);
  EXPECT_EQ(out[0].stats.capacity_bytes, 77u);
}

TEST(Membership, StatsExcludedFromEpochAndEquality) {
  MemberInfo a{"n1", "10.0.0.1", 28001, 1, 28100, "ver=x"};
  MemberInfo b = a;
  b.stats.used_bytes = 999;  // heartbeat churn
  b.has_stats = true;
  EXPECT_TRUE(a == b) << "stats must not affect member equality";
  EXPECT_EQ(MembersEpoch({a}), MembersEpoch({b}))
      << "stats churn must never rebuild client rings";
}

TEST(Membership, HealthRoundTripAndPlacementEpoch) {
  MemberInfo active{"n1", "10.0.0.1", 28001, 1};
  active.has_health = true;
  active.health.ring_eligible = true;
  active.health.ib_devices = {
      {"ib0", 4, 5, true}, {"ib1", 4, 5, true}};
  MemberInfo degraded{"n2", "10.0.0.2", 28001, 1};
  degraded.has_health = true;
  degraded.health.ring_eligible = false;
  degraded.health.ib_devices = {{"ib2", 2, 2, true}};

  std::vector<MemberInfo> out;
  uint64_t epoch = 0;
  const std::string encoded = EncodeMembers({active, degraded}, 7);
  ASSERT_TRUE(DecodeMembers(encoded.data(), encoded.size(), &out, &epoch));
  ASSERT_EQ(out.size(), 2u);
  EXPECT_TRUE(out[0].RingEligible());
  EXPECT_EQ(out[0].health.ib_devices[0].name, "ib0");
  EXPECT_TRUE(out[0].health.ib_devices[0].healthy());
  EXPECT_FALSE(out[1].RingEligible());
  EXPECT_EQ(IbPortStateName(out[1].health.ib_devices[0].port_state), "INIT");
  EXPECT_EQ(IbPhysStateName(out[1].health.ib_devices[0].phys_state),
            "POLLING");

  MemberInfo changed = active;
  changed.health.ib_devices[0].port_state = 3;
  EXPECT_EQ(MembersEpoch({active}), MembersEpoch({changed}))
      << "raw telemetry changes must not rebuild the ring";
  changed.health.ring_eligible = false;
  EXPECT_NE(MembersEpoch({active}), MembersEpoch({changed}))
      << "eligibility changes must rebuild the ring";
}

TEST(Membership, MissingHealthExtensionDefaultsEligible) {
  MemberInfo legacy{"legacy", "10.0.0.1", 28001, 1};
  const std::string encoded = EncodeMembers({legacy}, 1);
  std::vector<MemberInfo> out;
  uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(encoded.data(), encoded.size(), &out, &epoch));
  ASSERT_EQ(out.size(), 1u);
  EXPECT_FALSE(out[0].has_health);
  EXPECT_TRUE(out[0].RingEligible());
}
