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

TEST(PeerHealth, FailuresDuringACooldownAreTheSameIncident) {
  PeerHealth h(/*base=*/100, /*max=*/3000);
  // Seven shards of one batch fail together, and two stragglers that were in
  // flight report while the peer is already avoided.
  for (int i = 0; i < 7; ++i) h.MarkBad("p", 1000);
  h.MarkBad("p", 1040);
  h.MarkBad("p", 1099);
  EXPECT_EQ(h.errors(), 9u);       // every failure is still counted
  EXPECT_EQ(h.marked_bad(), 1u);   // one healthy->bad transition
  EXPECT_FALSE(h.Healthy("p", 1099));
  EXPECT_TRUE(h.Healthy("p", 1100));  // the base cooldown, from the first
}

TEST(PeerHealth, OneTrialAfterACooldown) {
  PeerHealth h(/*base=*/100, /*max=*/3000);
  h.MarkBad("p", 1000);
  // The cooldown is over. Eight shards ask at once: one is let through.
  int admitted = 0;
  for (int i = 0; i < 8; ++i) admitted += h.Healthy("p", 1100) ? 1 : 0;
  EXPECT_EQ(admitted, 1);
  EXPECT_FALSE(h.Healthy("p", 1150));  // the trial has not reported yet
  // It succeeds: the peer is healthy for everyone.
  h.MarkGood("p", 1160);
  for (int i = 0; i < 8; ++i) EXPECT_TRUE(h.Healthy("p", 1160));
  EXPECT_EQ(h.recovered(), 1u);
}

TEST(PeerHealth, FailedTrialStartsTheNextLongerCooldown) {
  PeerHealth h(/*base=*/100, /*max=*/3000);
  h.MarkBad("p", 1000);
  ASSERT_TRUE(h.Healthy("p", 1100));   // the trial
  h.MarkBad("p", 1130);                // it fails: 200 ms from now
  EXPECT_FALSE(h.Healthy("p", 1329));
  ASSERT_TRUE(h.Healthy("p", 1330));   // the next trial
  EXPECT_FALSE(h.Healthy("p", 1331));
  h.MarkBad("p", 1340);                // 400 ms
  EXPECT_FALSE(h.Healthy("p", 1739));
  EXPECT_TRUE(h.Healthy("p", 1740));
  EXPECT_EQ(h.marked_bad(), 3u);
}

TEST(PeerHealth, TrialThatNeverReportsLapsesAfterOneBaseCooldown) {
  // A trial that hangs reports nothing for as long as it hangs.
  PeerHealth h(/*base=*/100, /*max=*/3000);
  h.MarkBad("p", 1000);
  ASSERT_TRUE(h.Healthy("p", 1100));
  EXPECT_FALSE(h.Healthy("p", 1199));
  EXPECT_TRUE(h.Healthy("p", 1200));   // the next caller is the trial
  EXPECT_FALSE(h.Healthy("p", 1201));
}

TEST(PeerHealth, TrialThatHangsDoesNotBlockThePeerAndRearmsOnce) {
  PeerHealth h(/*base=*/100, /*max=*/3000);
  h.MarkBad("p", 1000);
  ASSERT_TRUE(h.Healthy("p", 1100));   // trial 1: hangs in an operation timeout
  EXPECT_FALSE(h.Healthy("p", 1199));
  ASSERT_TRUE(h.Healthy("p", 1200));   // its lease is over: trial 2
  h.MarkBad("p", 1250);                // trial 1 times out: 200 ms, to 1450
  h.MarkBad("p", 1260);                // trial 2 fails too: the same incident
  EXPECT_EQ(h.marked_bad(), 2u);
  EXPECT_FALSE(h.Healthy("p", 1449));
  EXPECT_TRUE(h.Healthy("p", 1450));
}

TEST(PeerHealth, StatusesAPeerSendsAreAnswers) {
  for (Status s : {Status::kOk, Status::kNotFound, Status::kCacheFull,
                   Status::kQuotaExceeded})
    EXPECT_TRUE(PeerAnswered(s)) << static_cast<int>(s);
  for (Status s : {Status::kIOError, Status::kInvalid,
                   Status::kResourceExhausted, Status::kNoCompatibleRail})
    EXPECT_FALSE(PeerAnswered(s)) << static_cast<int>(s);
  PeerOutcome mixed;
  mixed.Add(Status::kIOError);
  mixed.Add(Status::kCacheFull);
  EXPECT_TRUE(mixed.answered);  // one answer in a batch outweighs a failure
}

TEST(PeerHealth, ReportTurnsAnOutcomeIntoOneVerdict) {
  PeerHealth h(/*base=*/100, /*max=*/3000);
  PeerOutcome failed, answered, neutral;
  failed.Add(Status::kIOError);
  answered.Add(Status::kQuotaExceeded);
  neutral.Add(Status::kResourceExhausted);

  h.Report("p", failed, 1000);
  EXPECT_FALSE(h.Healthy("p", 1050));
  ASSERT_TRUE(h.Healthy("p", 1100));    // the trial
  h.Report("p", neutral, 1105);         // no verdict: the trial is given back
  ASSERT_TRUE(h.Healthy("p", 1105));    // the next caller tries at once
  // Refused by the transport without a dial: no verdict either, and no
  // longer cooldown.
  h.Report("p", failed, 1110, /*refused_without_dial=*/true);
  EXPECT_EQ(h.marked_bad(), 1u);
  ASSERT_TRUE(h.Healthy("p", 1110));
  h.Report("p", answered, 1120);        // "quota exceeded" is an answer
  EXPECT_EQ(h.recovered(), 1u);
  EXPECT_TRUE(h.Healthy("p", 1120));
  EXPECT_TRUE(h.Healthy("p", 1120));
}

TEST(PeerHealth, ProbeAnswerBringsTheTrialForwardAndDecidesNothing) {
  PeerHealth h(/*base=*/1000, /*max=*/30000);
  h.MarkBad("p", 0);
  EXPECT_FALSE(h.Healthy("p", 500));
  const uint64_t served_before = h.served();
  h.MarkProbeAlive("p", 500);
  EXPECT_EQ(h.recovered(), 0u);          // nothing is cleared by a probe
  EXPECT_TRUE(h.Healthy("p", 500));      // the next request is the trial
  EXPECT_FALSE(h.Healthy("p", 500));     // and only that one
  EXPECT_EQ(h.served(), served_before);
  h.MarkGood("p", 510);                  // the trial decides
  EXPECT_EQ(h.recovered(), 1u);
  // The record is clean: the next failure starts at the base cooldown.
  h.MarkBad("p", 5000);
  EXPECT_FALSE(h.Healthy("p", 5999));
  EXPECT_TRUE(h.Healthy("p", 6000));
}

TEST(PeerHealth, ProbeAnswerShortensOneCooldownPerIncident) {
  // A node that answers the probe while its reads fail.
  PeerHealth h(/*base=*/1000, /*max=*/30000);
  h.MarkBad("p", 0);
  h.MarkProbeAlive("p", 100);
  ASSERT_TRUE(h.Healthy("p", 100));      // early trial, granted by the probe
  h.MarkBad("p", 110);                   // fails: 2000 ms, to 2110
  h.MarkProbeAlive("p", 200);            // answers again: not believed twice
  EXPECT_FALSE(h.Healthy("p", 2109));
  ASSERT_TRUE(h.Healthy("p", 2110));
  h.MarkBad("p", 2120);                  // 4000 ms, to 6120
  h.MarkProbeAlive("p", 2200);
  EXPECT_FALSE(h.Healthy("p", 6119));
  // Once a request has succeeded, a later incident gets its early trial again.
  ASSERT_TRUE(h.Healthy("p", 6120));
  h.MarkGood("p", 6130);
  h.MarkBad("p", 9000);
  h.MarkProbeAlive("p", 9100);
  EXPECT_TRUE(h.Healthy("p", 9100));
}

TEST(PeerHealth, ProbeAnswerForAPeerThatIsNotAvoidedChangesNothing) {
  PeerHealth h(/*base=*/1000, /*max=*/30000);
  h.MarkProbeAlive("p", 0);
  EXPECT_TRUE(h.Healthy("p", 0));
  EXPECT_EQ(h.recovered(), 0u);
  EXPECT_EQ(h.cooldown_peers_for_test(), 0u);
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

// A failure is reported with a clock read before the lock; another thread
// may have recorded an answer since, a little later than that clock. The
// peer answered within the grace period: no cooldown.
TEST(PeerHealth, AnswerNewerThanTheFailuresClockIsWithinGrace) {
  PeerHealth h(/*cooldown_ms=*/2000, /*max=*/30000, /*grace=*/1000);
  h.MarkGood("a", 5001);
  h.MarkBad("a", 5000);
  EXPECT_TRUE(h.Healthy("a", 5000)) << "a peer that just answered cooled down";
}

// An answer stamped far ahead of the failure's clock is no skew: the grace
// does not apply.
TEST(PeerHealth, AnswerFarAheadOfTheFailuresClockIsNoGrace) {
  PeerHealth h(/*cooldown_ms=*/2000, /*max=*/30000, /*grace=*/1000);
  h.MarkGood("a", 65000);
  h.MarkBad("a", 5000);
  EXPECT_FALSE(h.Healthy("a", 5000));
}
