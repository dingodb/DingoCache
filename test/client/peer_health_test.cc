#include "client/peer_health.h"
#include <gtest/gtest.h>
using namespace dfkv;  // NOLINT

TEST(PeerHealth, UnknownPeerIsHealthy) {
  PeerHealth h(1000);
  EXPECT_TRUE(h.Healthy("a", 0));
}

TEST(PeerHealth, MarkBadCoolsDownThenRecovers) {
  PeerHealth h(/*cooldown_ms=*/1000);
  h.MarkBad("a", 5000);
  EXPECT_FALSE(h.Healthy("a", 5000));
  EXPECT_FALSE(h.Healthy("a", 5999));
  EXPECT_TRUE(h.Healthy("a", 6000));
}

TEST(PeerHealth, MarkGoodClearsImmediately) {
  PeerHealth h(1000);
  h.MarkBad("a", 1000);
  EXPECT_FALSE(h.Healthy("a", 1200));
  h.MarkGood("a", 0);
  EXPECT_TRUE(h.Healthy("a", 1200));
}

TEST(PeerHealth, PeersAreIndependent) {
  PeerHealth h(1000);
  h.MarkBad("a", 1000);
  EXPECT_FALSE(h.Healthy("a", 1500));
  EXPECT_TRUE(h.Healthy("b", 1500));
}

TEST(PeerHealth, CooldownBacksOffExponentiallyAndCapsAtMax) {
  PeerHealth h(/*base_cooldown_ms=*/100, /*max_cooldown_ms=*/800);
  // 1st failure -> 100ms cooldown: unhealthy at now, healthy at now+100.
  h.MarkBad("p", 1000);
  EXPECT_FALSE(h.Healthy("p", 1050));
  EXPECT_TRUE(h.Healthy("p", 1100));
  // Consecutive failures double the cooldown: 200, 400, 800, then capped at 800.
  h.MarkBad("p", 2000); EXPECT_FALSE(h.Healthy("p", 2199)); EXPECT_TRUE(h.Healthy("p", 2200));
  h.MarkBad("p", 3000); EXPECT_FALSE(h.Healthy("p", 3399)); EXPECT_TRUE(h.Healthy("p", 3400));
  h.MarkBad("p", 4000); EXPECT_FALSE(h.Healthy("p", 4799)); EXPECT_TRUE(h.Healthy("p", 4800));
  h.MarkBad("p", 5000); EXPECT_FALSE(h.Healthy("p", 5799)); EXPECT_TRUE(h.Healthy("p", 5800));  // capped
}

TEST(PeerHealth, MarkGoodResetsBackoffStreak) {
  PeerHealth h(/*base=*/100, /*max=*/10000);
  h.MarkBad("p", 0); h.MarkBad("p", 0); h.MarkBad("p", 0);  // streak -> 400ms
  h.MarkGood("p", 0);                                          // reset
  h.MarkBad("p", 1000);                                     // back to base 100ms
  EXPECT_FALSE(h.Healthy("p", 1099));
  EXPECT_TRUE(h.Healthy("p", 1100));
}

TEST(PeerHealth, BusyPeerFailureDoesNotCoolDown) {
  // R2-B: a peer that served a response within the grace window is alive-busy;
  // one failed op on it must not instant-fail everything else for the cooldown
  // (that turned a single write-burst stall into a 2-30s artificial outage and
  // defeated the hicache put retry).
  PeerHealth h(/*base=*/1000, /*max=*/30000, /*busy_grace_ms=*/1000);
  h.MarkGood("p", 5000);
  h.MarkBad("p", 5500);                        // 500ms after a served op
  EXPECT_TRUE(h.Healthy("p", 5500));           // no cooldown: retry can proceed
  EXPECT_EQ(h.busy_suppressed(), 1u);
  EXPECT_EQ(h.errors(), 1u);                   // the failure is still counted
  EXPECT_EQ(h.marked_bad(), 0u);               // no healthy->bad transition
}

TEST(PeerHealth, StaleGoodPeerFailureCoolsDown) {
  // Last served response is OLDER than the grace window: the peer may be dead,
  // cooldown engages exactly as before.
  PeerHealth h(/*base=*/1000, /*max=*/30000, /*busy_grace_ms=*/1000);
  h.MarkGood("p", 5000);
  h.MarkBad("p", 6500);                        // 1.5s silence > 1s grace
  EXPECT_FALSE(h.Healthy("p", 6500));
  EXPECT_EQ(h.busy_suppressed(), 0u);
}

TEST(PeerHealth, NeverServedPeerFailureCoolsDown) {
  // First contact with a dead peer: no served history, first failure cools
  // down immediately (the original connect-storm protection is preserved).
  PeerHealth h(/*base=*/1000, /*max=*/30000, /*busy_grace_ms=*/1000);
  h.MarkBad("p", 100);
  EXPECT_FALSE(h.Healthy("p", 100));
  EXPECT_EQ(h.busy_suppressed(), 0u);
}

TEST(PeerHealth, MarkProbeAliveRecoversWithoutCountingServed) {
  PeerHealth h(/*base=*/1000, /*max=*/30000);
  h.MarkBad("p", 0);
  EXPECT_FALSE(h.Healthy("p", 500));
  uint64_t served_before = h.served();
  h.MarkProbeAlive("p");                       // off-data-path recovery
  EXPECT_TRUE(h.Healthy("p", 500));            // cooldown cleared
  EXPECT_EQ(h.served(), served_before);        // served counter untouched
  EXPECT_EQ(h.recovered(), 1u);                // bad->good edge counted
  // And the streak was reset: next failure starts at base again.
  h.MarkBad("p", 1000);
  EXPECT_TRUE(h.Healthy("p", 2000));
}

TEST(PeerHealth, UniqueFailingAddressFloodBoundsEveryMapAndEvictsLru) {
  PeerHealth h(/*base=*/1000, /*max=*/30000);
  for (int i = 0; i < 5000; ++i)
    h.MarkBad("peer-" + std::to_string(i), 0);

  EXPECT_EQ(h.tracked_peers_for_test(), 4096u);
  EXPECT_EQ(h.cooldown_peers_for_test(), 4096u);
  EXPECT_EQ(h.failure_streak_peers_for_test(), 4096u);
  EXPECT_EQ(h.error_peers_for_test(), 4096u);
  EXPECT_LE(h.last_good_peers_for_test(), 4096u);
  EXPECT_EQ(h.errors(), 5000u);

  // Refresh one retained peer, then insert a new address. The oldest retained
  // peer is evicted from cooldown and streak state; the refreshed peer stays.
  h.MarkBad("peer-1000", 0);
  h.MarkBad("peer-new", 0);
  EXPECT_TRUE(h.Healthy("peer-904", 0));
  EXPECT_FALSE(h.Healthy("peer-1000", 0));
  EXPECT_FALSE(h.Healthy("peer-new", 0));
  EXPECT_EQ(h.tracked_peers_for_test(), 4096u);
  EXPECT_EQ(h.cooldown_peers_for_test(), 4096u);
  EXPECT_EQ(h.failure_streak_peers_for_test(), 4096u);
  EXPECT_EQ(h.error_peers_for_test(), 4096u);
}
