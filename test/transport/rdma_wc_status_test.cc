// Completion-status names are an operator contract: the same token appears in
// the failure log lines and as the `status` label of the per-status error
// counters. Verbs-free, so it runs in builds without libibverbs.
#include "transport/rdma_wc_status.h"
#include "utils/log_throttle.h"

#include <gtest/gtest.h>

#include <set>
#include <string>

using namespace dfkv;  // NOLINT

TEST(RdmaWcStatus, NamesFollowTheVerbsNumbering) {
  EXPECT_STREQ(rdma::WcStatusName(0), "success");
  EXPECT_STREQ(rdma::WcStatusName(5), "wr_flush_err");
  EXPECT_STREQ(rdma::WcStatusName(10), "rem_access_err");
  EXPECT_STREQ(rdma::WcStatusName(12), "retry_exc_err");
  EXPECT_STREQ(rdma::WcStatusName(13), "rnr_retry_exc_err");
  EXPECT_STREQ(rdma::WcStatusName(21), "general_err");
  EXPECT_STREQ(rdma::WcStatusName(23), "tm_rndv_incomplete");
}

TEST(RdmaWcStatus, UnknownStatusesShareOneSlotSoLabelsStayBounded) {
  EXPECT_EQ(rdma::WcStatusSlot(12), 12u);
  for (int status : {-1, 24, 25, 1000, 1 << 30}) {
    EXPECT_EQ(rdma::WcStatusSlot(status), rdma::kWcStatusKnown) << status;
    EXPECT_STREQ(rdma::WcStatusName(status), "other") << status;
  }
  std::set<std::string> names;
  for (size_t slot = 0; slot < rdma::kWcStatusSlots; ++slot) {
    const std::string name = rdma::WcStatusName(static_cast<int>(slot));
    EXPECT_FALSE(name.empty());
    EXPECT_EQ(name.find_first_not_of("abcdefghijklmnopqrstuvwxyz_"),
              std::string::npos)
        << name << " must be usable as a label value and a log token";
    names.insert(name);
  }
  EXPECT_EQ(names.size(), rdma::kWcStatusSlots) << "one name per slot";
}

TEST(RdmaWcStatus, LogFieldsCarryNumberNameAndVendorSyndrome) {
  EXPECT_EQ(rdma::WcStatusFields(12, 129),
            "status=12 status_name=retry_exc_err vendor=129");
  EXPECT_EQ(rdma::WcStatusFields(99, 0),
            "status=99 status_name=other vendor=0");
}

TEST(RdmaWcStatus, FailureKindSeparatesStatusAndVendorSyndrome) {
  EXPECT_EQ(rdma::WcFailureKind(12, 129), rdma::WcFailureKind(12, 129));
  EXPECT_NE(rdma::WcFailureKind(12, 129), rdma::WcFailureKind(12, 130));
  EXPECT_NE(rdma::WcFailureKind(12, 129), rdma::WcFailureKind(5, 129));
  EXPECT_NE(rdma::WcFailureKind(12, 0), rdma::WcFailureKind(0, 12));
  EXPECT_NE(rdma::WcFailureKind(1, 0), rdma::WcFailureKind(0, 0xffffffffu));
}

// The client failure line is throttled per (peer, rail, failure kind): a
// repeat is swallowed, but the same status with another vendor syndrome is a
// different failure and gets its own line in the same window.
TEST(RdmaWcStatus, ThrottleKeyAdmitsANewVendorSyndromeUnderTheSameStatus) {
  LogThrottle throttle;
  const std::string peer = "192.0.2.7:7000";
  const auto key = [&](int status, unsigned vendor) {
    return LogThrottle::Key(peer, /*rail=*/1,
                            rdma::WcFailureKind(status, vendor));
  };
  uint64_t suppressed = 0;
  EXPECT_TRUE(throttle.Admit(key(12, 129), 1000, &suppressed));
  EXPECT_FALSE(throttle.Admit(key(12, 129), 1001, &suppressed));
  EXPECT_TRUE(throttle.Admit(key(12, 130), 1002, &suppressed));
  EXPECT_EQ(suppressed, 1u);
  EXPECT_FALSE(throttle.Admit(key(12, 130), 1003, &suppressed));
  EXPECT_TRUE(throttle.Admit(key(5, 129), 1004, &suppressed));
}
