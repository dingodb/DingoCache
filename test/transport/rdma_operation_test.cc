#include "transport/rdma_operation.h"

#include <atomic>
#include <thread>

#include <gtest/gtest.h>

namespace dfkv::rdma {
namespace {

TEST(RdmaOperation, ExactlyOnePollerOwnsCompletion) {
  OperationContext operation;
  ASSERT_TRUE(operation.Submit());
  ASSERT_TRUE(operation.ClaimPoller());

  std::atomic<bool> foreign_completed{true};
  std::thread foreign([&] {
    EXPECT_FALSE(operation.ClaimPoller());
    foreign_completed.store(operation.Complete(true),
                            std::memory_order_release);
  });
  foreign.join();

  EXPECT_FALSE(foreign_completed.load(std::memory_order_acquire));
  EXPECT_TRUE(operation.Complete(true));
  EXPECT_EQ(operation.state(), OperationState::kCompleted);
  EXPECT_TRUE(operation.terminal());
  EXPECT_FALSE(operation.Complete(true));
}

TEST(RdmaOperation, CancellationIsTerminalOnlyAfterOwnerDrains) {
  OperationContext operation;
  ASSERT_TRUE(operation.Submit());
  ASSERT_TRUE(operation.ClaimPoller());
  operation.RequestCancel();

  EXPECT_FALSE(operation.terminal());
  EXPECT_TRUE(operation.cancel_requested());
  EXPECT_TRUE(operation.Complete(false));
  EXPECT_EQ(operation.state(), OperationState::kCancelled);
  EXPECT_TRUE(operation.terminal());
}

TEST(RdmaOperation, FailedOperationCompletesExactlyOnce) {
  OperationContext operation;
  ASSERT_TRUE(operation.Submit());
  ASSERT_TRUE(operation.ClaimPoller());
  ASSERT_TRUE(operation.Complete(false));
  EXPECT_EQ(operation.state(), OperationState::kFailed);
  EXPECT_FALSE(operation.Complete(false));
}

}  // namespace
}  // namespace dfkv::rdma
