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

TEST(MemberViewGuard, GrowthAndSmallShrinkPassImmediately) {
  MemberViewGuard g(3, 50);
  EXPECT_TRUE(g.Admit(64));
  EXPECT_TRUE(g.Admit(80));   // growth
  EXPECT_TRUE(g.Admit(79));   // single-node failure must propagate fast
  EXPECT_TRUE(g.Admit(41));   // 48% drop: below the 50% bar, adopt
  EXPECT_EQ(g.rejected_shrink(), 0u);
}

TEST(MemberViewGuard, MassShrinkNeedsPersistence) {
  MemberViewGuard g(3, 50);
  EXPECT_TRUE(g.Admit(64));
  // etcd NOSPACE signature: most leases expired, a shrunken view appears.
  EXPECT_FALSE(g.Admit(5));   // poll 1: suspect
  EXPECT_FALSE(g.Admit(5));   // poll 2: still suspect
  EXPECT_TRUE(g.Admit(5));    // poll 3: persisted -> believe the drain
  EXPECT_EQ(g.rejected_shrink(), 2u);
  // Baseline moved to 5: a further small change passes.
  EXPECT_TRUE(g.Admit(4));
}

TEST(MemberViewGuard, RecoveryDuringHysteresisResetsStreak) {
  MemberViewGuard g(3, 50);
  EXPECT_TRUE(g.Admit(64));
  EXPECT_FALSE(g.Admit(5));   // transient collapse
  EXPECT_TRUE(g.Admit(63));   // etcd recovered: adopt, streak resets
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
  EXPECT_TRUE(boundary.Admit(1)) << "exactly 50% shrink is not over threshold";
}

TEST(MemberViewGuard, NonHalfBoundaryIsExact) {
  MemberViewGuard boundary(3, 33);
  EXPECT_TRUE(boundary.Admit(100));
  EXPECT_TRUE(boundary.Admit(67)) << "exactly 33% shrink is not over threshold";

  MemberViewGuard odd(3, 34);
  EXPECT_TRUE(odd.Admit(3));
  EXPECT_TRUE(odd.Admit(2));
  EXPECT_FALSE(odd.Admit(1));
}

TEST(MemberViewGuard, CrossMultiplicationDoesNotOverflowSizeT) {
  const size_t max = std::numeric_limits<size_t>::max();
  MemberViewGuard guarded(3, 50);
  EXPECT_TRUE(guarded.Admit(max));
  EXPECT_FALSE(guarded.Admit(max / 2));

  MemberViewGuard boundary(3, 50);
  EXPECT_TRUE(boundary.Admit(max - 1));
  EXPECT_TRUE(boundary.Admit((max - 1) / 2));
}
