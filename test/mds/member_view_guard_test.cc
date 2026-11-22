// MemberViewGuard: ring-adoption hysteresis against transient view collapses
// (etcd outage / NOSPACE recovery). Pure logic — no etcd, always runs.
#include "mds/mds_member_poller.h"

#include <gtest/gtest.h>

#include <limits>

using dfkv::MemberViewGuard;

TEST(MemberViewGuard, FirstViewAdoptedAsIs) {
  MemberViewGuard g(3, 50);
  EXPECT_TRUE(g.Admit(0));  // even an empty first view
  MemberViewGuard g2(3, 50);
  EXPECT_TRUE(g2.Admit(64));
}

TEST(MemberViewGuard, GrowthPassesAndASmallShrinkOnceItRepeats) {
  MemberViewGuard g(3, 50);
  EXPECT_TRUE(g.Admit(64));
  EXPECT_TRUE(g.Admit(80));   // growth
  EXPECT_TRUE(g.Admit(79));   // still above the 64 reference
  EXPECT_FALSE(g.Admit(41));  // 36% below the reference: under the bar...
  EXPECT_TRUE(g.Admit(41));   // ...believed on its second poll
  EXPECT_EQ(g.rejected_shrink(), 1u);
}

TEST(MemberViewGuard, MassShrinkNeedsPersistence) {
  MemberViewGuard g(3, 50);
  EXPECT_TRUE(g.Admit(64));
  // etcd NOSPACE signature: most leases expired, a shrunken view appears.
  EXPECT_FALSE(g.Admit(5));   // poll 1: suspect
  EXPECT_FALSE(g.Admit(5));   // poll 2: still suspect
  EXPECT_TRUE(g.Admit(5));    // poll 3: persisted -> believe the drain
  EXPECT_EQ(g.rejected_shrink(), 2u);
  // Reference moved to 5: a further small shrink needs only two polls.
  EXPECT_FALSE(g.Admit(4));
  EXPECT_TRUE(g.Admit(4));
}

TEST(MemberViewGuard, RecoveryDuringHysteresisResetsStreak) {
  MemberViewGuard g(3, 50);
  EXPECT_TRUE(g.Admit(64));
  EXPECT_FALSE(g.Admit(5));   // transient collapse
  EXPECT_TRUE(g.Admit(64));   // etcd recovered: adopt, streak resets
  EXPECT_FALSE(g.Admit(5));   // a NEW collapse restarts its own hysteresis
  EXPECT_FALSE(g.Admit(5));
  EXPECT_TRUE(g.Admit(5));
}

TEST(MemberViewGuard, EmptyArmStillGuards) {
  MemberViewGuard g(3, 50);
  EXPECT_TRUE(g.Admit(64));
  EXPECT_FALSE(g.Admit(0));
  EXPECT_FALSE(g.Admit(0));
  EXPECT_TRUE(g.Admit(0));    // persisted: teardown believed
  EXPECT_EQ(g.rejected_empty(), 2u);
  // After an adopted empty view the next non-empty is a re-registration: adopt.
  EXPECT_TRUE(g.Admit(64));
}

TEST(MemberViewGuard, ShrinkArmDisabledByZeroPct) {
  MemberViewGuard g(3, 0);
  EXPECT_TRUE(g.Admit(64));
  EXPECT_TRUE(g.Admit(1));    // pct=0: legacy behavior, shrink passes
  EXPECT_TRUE(g.Admit(500));  // pct=0 also disables the growth arm
  EXPECT_FALSE(g.Admit(0));   // empty arm is always on
}

TEST(MemberViewGuard, EmptyThenShrunkenRecoveryIsNotDoubleCounted) {
  // Outage recovery often looks like: empty, empty, then a partial table as
  // members trickle back. The partial view vs the OLD baseline is a shrink,
  // but adopting it beats serving the stale full ring for another cycle once
  // it persists.
  MemberViewGuard g(3, 50);
  EXPECT_TRUE(g.Admit(64));
  EXPECT_FALSE(g.Admit(0));
  EXPECT_FALSE(g.Admit(20));  // partial recovery: shrink hysteresis
  EXPECT_FALSE(g.Admit(20));
  EXPECT_TRUE(g.Admit(20));   // persisted partial view adopted
}

TEST(MemberViewGuard, OddAndSmallRingsUseExactPercentage) {
  MemberViewGuard three(3, 50);
  EXPECT_TRUE(three.Admit(3));
  EXPECT_FALSE(three.Admit(1)) << "3->1 drops more than 50%";
  EXPECT_FALSE(three.Admit(1));
  EXPECT_TRUE(three.Admit(1));

  MemberViewGuard five(3, 50);
  EXPECT_TRUE(five.Admit(5));
  EXPECT_FALSE(five.Admit(2)) << "5->2 retains only 40%";

  MemberViewGuard boundary(3, 50);
  EXPECT_TRUE(boundary.Admit(2));
  EXPECT_FALSE(boundary.Admit(1));
  EXPECT_TRUE(boundary.Admit(1)) << "exactly 50% is not a mass shrink";
}

TEST(MemberViewGuard, NonHalfBoundaryIsExact) {
  MemberViewGuard boundary(3, 33);
  EXPECT_TRUE(boundary.Admit(100));
  EXPECT_FALSE(boundary.Admit(67));
  EXPECT_TRUE(boundary.Admit(67)) << "exactly 33% is not a mass shrink";

  MemberViewGuard odd(3, 34);
  EXPECT_TRUE(odd.Admit(3));
  EXPECT_FALSE(odd.Admit(2));
  EXPECT_TRUE(odd.Admit(2));
  EXPECT_FALSE(odd.Admit(1));
  EXPECT_FALSE(odd.Admit(1)) << "below the bar of the 3 reference";
}

TEST(MemberViewGuard, CrossMultiplicationDoesNotOverflowSizeT) {
  const size_t max = std::numeric_limits<size_t>::max();
  MemberViewGuard guarded(3, 50);
  EXPECT_TRUE(guarded.Admit(max));
  EXPECT_FALSE(guarded.Admit(max / 2));

  MemberViewGuard boundary(3, 50);
  EXPECT_TRUE(boundary.Admit(max - 1));
  EXPECT_FALSE(boundary.Admit((max - 1) / 2));
  EXPECT_TRUE(boundary.Admit((max - 1) / 2)) << "not a mass shrink";
}

TEST(MemberViewGuard, DiffuseDecayChainFreezesReference) {
  // Mass lease expiry with heartbeat phases spread over polls: every hop
  // drains well under 50% of its neighbor, so an adjacent-hop guard ratchets
  // the ring down hop by hop. The trusted reference must hold at 64 until a
  // repeated value crosses the bar.
  MemberViewGuard g(3, 50);
  EXPECT_TRUE(g.Admit(64));
  // Every hop is above the bar but none repeats: none is adopted.
  for (size_t v : {60ul, 56ul, 52ul, 48ul, 44ul, 40ul, 36ul, 33ul}) {
    EXPECT_FALSE(g.Admit(v)) << "hop " << v << " adopted without repeating";
  }
  EXPECT_EQ(g.rejected_shrink(), 8u);
  // The reference did NOT slide along: back at it, the view is adopted.
  EXPECT_TRUE(g.Admit(64));
  // 31 first crosses below 50% of the frozen 64 reference: hysteresis until
  // ONE value persists.
  EXPECT_FALSE(g.Admit(31));
  EXPECT_FALSE(g.Admit(31));
  EXPECT_TRUE(g.Admit(31));   // persisted: real mass drain
  EXPECT_EQ(g.rejected_shrink(), 10u);
}

TEST(MemberViewGuard, SuspiciousViewMustRepeatOneValue) {
  // The old guard counted consecutive *suspicious* polls; a still-draining
  // view (20, 10, 10) would trip it and adopt mid-collapse. Only one value
  // repeating views_to_accept times may be believed.
  MemberViewGuard g(3, 50);
  EXPECT_TRUE(g.Admit(64));
  EXPECT_FALSE(g.Admit(20));
  EXPECT_FALSE(g.Admit(10));  // different suspicious value: streak restarts
  EXPECT_FALSE(g.Admit(10));
  EXPECT_TRUE(g.Admit(10));
  EXPECT_EQ(g.rejected_shrink(), 3u);
}

TEST(MemberViewGuard, RecoveryRampAdoptsOnceAtPlateau) {
  // After a believed collapse, members re-register on their own schedule: the
  // view grows epoch by epoch. Rebuilding the ring per epoch is as disruptive
  // as the collapse itself, so a shifting ramp must never be adopted — only
  // the plateau. (pct=20 so every hop of the 33->40->50->64 ramp clears the
  // suspicion bar and exercises the growth arm.)
  MemberViewGuard g(3, 20);
  EXPECT_TRUE(g.Admit(64));
  EXPECT_FALSE(g.Admit(33));  // collapse: shrink hysteresis
  EXPECT_FALSE(g.Admit(33));
  EXPECT_TRUE(g.Admit(33));   // persisted: believed, reference re-anchored
  // Ramp: every hop differs, so nothing is adopted mid-ramp.
  EXPECT_FALSE(g.Admit(40));
  EXPECT_FALSE(g.Admit(50));
  EXPECT_FALSE(g.Admit(64));
  // Plateau: ONE repeated value adopts, exactly once.
  EXPECT_FALSE(g.Admit(64));
  EXPECT_TRUE(g.Admit(64));
  // Afterwards the stable ring is ordinary traffic again.
  EXPECT_TRUE(g.Admit(64));
  EXPECT_EQ(g.rejected_shrink(), 2u);
  EXPECT_EQ(g.rejected_growth(), 4u);
}

TEST(MemberViewGuard, SingleNodeEvents) {
  MemberViewGuard down(3, 50);
  EXPECT_TRUE(down.Admit(64));
  EXPECT_FALSE(down.Admit(63));  // a single-node failure takes two polls
  EXPECT_TRUE(down.Admit(63));

  MemberViewGuard up(3, 50);
  EXPECT_TRUE(up.Admit(64));
  EXPECT_TRUE(up.Admit(65));    // one node joining is below the growth bar
}

TEST(MemberViewGuard, GrowthArmWideMathDoesNotOverflow) {
  const size_t max = std::numeric_limits<size_t>::max();
  MemberViewGuard g(3, 50);
  EXPECT_TRUE(g.Admit(max / 4));
  // max/4 -> max/2 is a 2x jump: growth-suspicious, must persist as one value.
  EXPECT_FALSE(g.Admit(max / 2));
  EXPECT_FALSE(g.Admit(max / 2));
  EXPECT_TRUE(g.Admit(max / 2));
  EXPECT_EQ(g.rejected_growth(), 2u);

  // Exactly +50% is not over the bar (mirrors the shrink boundary).
  MemberViewGuard boundary(3, 50);
  EXPECT_TRUE(boundary.Admit(max / 3));
  EXPECT_TRUE(boundary.Admit(max / 2));

  // Hundreds of nodes: exact percentage math, no float fuzz.
  MemberViewGuard big(3, 50);
  EXPECT_TRUE(big.Admit(300));
  EXPECT_FALSE(big.Admit(299));
  EXPECT_TRUE(big.Admit(299));
  EXPECT_FALSE(big.Admit(100));  // -67% of the 300 reference: hysteresis
}

// A recovery ramp after an outage that the guard did not see as a mass
// shrink (every view above the bar, members re-registering one by one) must
// not be adopted view by view: each adoption rebuilds the ring on an
// incomplete view. A shrink below the reference is believed once it repeats;
// the ramp's views do not, and the full view is ordinary traffic again.
TEST(MemberViewGuard, ARecoveryRampAboveTheShrinkBarIsNotAdoptedViewByView) {
  MemberViewGuard g(3, 50);
  EXPECT_TRUE(g.Admit(64));
  for (size_t v : {40ul, 50ul, 60ul}) {
    EXPECT_FALSE(g.Admit(v)) << "incomplete view " << v << " adopted";
  }
  EXPECT_TRUE(g.Admit(64));
  EXPECT_EQ(g.rejected_shrink(), 3u);
}

// A real member loss still gets through, one poll later than before: the
// smaller view is believed when it repeats.
TEST(MemberViewGuard, AMemberLossIsBelievedWhenItsViewRepeats) {
  MemberViewGuard g(3, 50);
  EXPECT_TRUE(g.Admit(64));
  EXPECT_FALSE(g.Admit(63));
  EXPECT_TRUE(g.Admit(63));
  EXPECT_TRUE(g.Admit(63));
}

// After a believed empty ring the first members to come back must not be
// frozen in as the whole ring: the ring follows them as more register, and
// the reference anchors only on a view that repeats.
TEST(MemberViewGuard, AfterAnEmptyRingTheRingFollowsTheReturningMembers) {
  MemberViewGuard g(3, 50);
  EXPECT_TRUE(g.Admit(64));
  EXPECT_FALSE(g.Admit(0));
  EXPECT_FALSE(g.Admit(0));
  EXPECT_TRUE(g.Admit(0));  // teardown believed
  for (size_t v : {2ul, 5ul, 9ul, 20ul, 64ul}) {
    EXPECT_TRUE(g.Admit(v)) << "view " << v << " held back after an empty ring";
  }
  EXPECT_EQ(g.rejected_growth(), 0u);
  EXPECT_TRUE(g.Admit(64));
  EXPECT_TRUE(g.Admit(64));  // repeated: the reference is 64 now
  EXPECT_FALSE(g.Admit(5));  // so a collapse to 5 is a mass shrink again
}

// While the reference is not anchored yet (members coming back after an
// empty ring), the last adopted view stands in for it on the shrink side: one
// bad poll that shows far fewer members is held back like any mass shrink,
// and the ramp's growth stays free.
TEST(MemberViewGuard, AfterAnEmptyRingOneBadPollDoesNotShrinkTheRing) {
  MemberViewGuard g(3, 50);
  EXPECT_TRUE(g.Admit(64));
  EXPECT_FALSE(g.Admit(0));
  EXPECT_FALSE(g.Admit(0));
  EXPECT_TRUE(g.Admit(0));
  for (size_t v : {2ul, 5ul, 9ul, 20ul, 40ul}) EXPECT_TRUE(g.Admit(v)) << v;
  EXPECT_FALSE(g.Admit(3)) << "one bad poll rebuilt the ring";
  EXPECT_TRUE(g.Admit(40));
  EXPECT_TRUE(g.Admit(50)) << "growth stays free";
}
