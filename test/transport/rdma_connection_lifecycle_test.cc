#include "transport/rdma_connection_lifecycle.h"

#include <atomic>
#include <thread>

#include <gtest/gtest.h>

namespace dfkv::rdma {
namespace {

TEST(RdmaConnectionLifecycle, IdleEndpointCanActivateOrRetireExactlyOnce) {
  ConnectionLifecycle lifecycle;
  ASSERT_TRUE(lifecycle.MakeIdle());

  std::atomic<int> winners{0};
  std::thread activate([&] {
    if (lifecycle.Activate()) winners.fetch_add(1, std::memory_order_relaxed);
  });
  std::thread retire([&] {
    if (lifecycle.RequestRetire())
      winners.fetch_add(1, std::memory_order_relaxed);
  });
  activate.join();
  retire.join();

  EXPECT_EQ(winners.load(), 1);
  EXPECT_TRUE(lifecycle.state() == ConnectionState::kActive ||
              lifecycle.state() == ConnectionState::kRetireRequested);
}

TEST(RdmaConnectionLifecycle, RetiredEndpointCannotReactivate) {
  ConnectionLifecycle lifecycle;
  ASSERT_TRUE(lifecycle.MakeIdle());
  ASSERT_TRUE(lifecycle.RequestRetire());
  EXPECT_FALSE(lifecycle.Activate());
  EXPECT_TRUE(lifecycle.BeginDrain());
  EXPECT_EQ(lifecycle.state(), ConnectionState::kDraining);
  EXPECT_FALSE(lifecycle.BeginDrain());
}

TEST(RdmaConnectionLifecycle, ActiveFailureTransitionsDirectlyToDrain) {
  ConnectionLifecycle lifecycle;
  ASSERT_TRUE(lifecycle.BeginDrain());
  EXPECT_EQ(lifecycle.state(), ConnectionState::kDraining);
  EXPECT_FALSE(lifecycle.MakeIdle());
  EXPECT_FALSE(lifecycle.RequestRetire());
}

}  // namespace
}  // namespace dfkv::rdma
