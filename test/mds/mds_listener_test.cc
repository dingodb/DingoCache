// The MDS accept loop when the process runs short of descriptors or threads,
// and the fit between DFKV_MDS_MAX_CONNS and the open-file limit.
//
// No etcd is needed: the listener is what is under test. The etcd endpoint
// points at a port nothing listens on, so a request that reaches a handler is
// answered with kIOError, which is enough to tell that it was served.
#include "mds/mds_server.h"

#include <gtest/gtest.h>

#include <netinet/in.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <thread>

#include "../utils/fd_exhaustion.h"
#include "transport/wire.h"
#include "utils/accept_guard.h"
#include "utils/net_util.h"

using namespace dfkv;  // NOLINT

namespace {

// A loopback endpoint nothing listens on: a port the kernel handed out to a
// socket that is bound but never listens. The socket is kept for the life of
// the process so that nothing else, the MDS under test included, gets the
// port; connecting to it is refused at once.
std::string DeadEtcd() {
  static const int port = [] {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t length = sizeof(address);
    ::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address));
    ::getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length);
    return static_cast<int>(ntohs(address.sin_port));
  }();
  return "127.0.0.1:" + std::to_string(port);
}

// One ListGroups round trip on `fd`; true when the MDS answered at all.
bool AnsweredOn(int fd) {
  char request[kReqPrefix];
  EncodeReq(request, WireOp::kListGroups, BlockKey{}, 0, 0, 0);
  char response[kRespPrefix];
  Status status = Status::kOk;
  uint64_t length = 0;
  return net::WriteAll(fd, request, sizeof(request)) &&
         net::ReadAll(fd, response, sizeof(response)) &&
         DecodeResp(response, &status, &length);
}

bool Answered(int port) {
  const int fd = net::Dial("127.0.0.1:" + std::to_string(port), 2000, 5000);
  if (fd < 0) return false;
  const bool answered = AnsweredOn(fd);
  ::close(fd);
  return answered;
}

bool ConnectTo(int fd, int port) {
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = htons(static_cast<uint16_t>(port));
  return ::connect(fd, reinterpret_cast<sockaddr*>(&address),
                   sizeof(address)) == 0;
}

template <class Predicate>
bool EventuallyTrue(Predicate predicate, int timeout_ms = 3000) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return true;
}

}  // namespace

TEST(MdsDescriptorPlan, RaisesTheSoftLimitWhenTheHardLimitAllows) {
  // The usual service defaults: soft 1024, a large hard limit.
  const MdsDescriptorPlan plan = PlanMdsDescriptors(4096, 1024, 524288);
  EXPECT_EQ(plan.soft_limit, MdsDescriptorsFor(4096));
  EXPECT_EQ(plan.soft_limit, 8320u);
  EXPECT_EQ(plan.max_conns, 4096u);
}

TEST(MdsDescriptorPlan, LeavesASufficientLimitAlone) {
  const MdsDescriptorPlan plan = PlanMdsDescriptors(4096, 65536, 1048576);
  EXPECT_EQ(plan.soft_limit, 65536u);
  EXPECT_EQ(plan.max_conns, 4096u);
  // Nor is a limit ever lowered to what a small cap needs.
  EXPECT_EQ(PlanMdsDescriptors(10, 1024, 4096).soft_limit, 1024u);
}

TEST(MdsDescriptorPlan, ShrinksTheCapWhenTheHardLimitIsShort) {
  // Raised as far as it goes; the cap keeps one etcd socket per connection
  // and the reserve under it.
  MdsDescriptorPlan plan = PlanMdsDescriptors(4096, 1024, 4096);
  EXPECT_EQ(plan.soft_limit, 4096u);
  EXPECT_EQ(plan.max_conns, (4096u - kMdsReservedDescriptors) / 2);
  EXPECT_LE(MdsDescriptorsFor(plan.max_conns), plan.soft_limit);

  plan = PlanMdsDescriptors(4096, 1024, 1024);
  EXPECT_EQ(plan.soft_limit, 1024u);
  EXPECT_EQ(plan.max_conns, 448u);

  // A limit below the reserve still serves one connection at a time.
  EXPECT_EQ(PlanMdsDescriptors(4096, 100, 100).max_conns, 1u);
}

TEST(MdsListener, StartFitsTheConnectionCapToTheOpenFileLimit) {
  const int result = testing_fd::RunInChild([] {
    ::unsetenv("DFKV_MDS_MAX_CONNS");
    rlimit limit{};
    if (::getrlimit(RLIMIT_NOFILE, &limit) != 0) return 10;
    // A hard limit below what 4096 connections need (lowering it is allowed
    // and stays inside this child).
    if (limit.rlim_max < 1024) return 0;  // nothing to lower from
    limit.rlim_cur = 256;
    limit.rlim_max = 1024;
    if (::setrlimit(RLIMIT_NOFILE, &limit) != 0) return 11;

    MdsServer mds(DeadEtcd());
    if (mds.Start(0) != Status::kOk) return 12;
    if (::getrlimit(RLIMIT_NOFILE, &limit) != 0) return 13;
    if (limit.rlim_cur != 1024) return 14;  // raised to the hard limit
    if (mds.max_conns() != 448) return 15;  // and the cap shrunk to fit
    if (mds.ListenerMetricsText().find("dfkv_mds_max_conns 448\n") ==
        std::string::npos)
      return 16;
    if (!Answered(mds.port())) return 17;
    mds.Stop();
    return 0;
  });
  EXPECT_EQ(result, 0) << "step code from the child, see the test body";
}

TEST(MdsListener, StartRaisesTheSoftLimitForTheDefaultCap) {
  const int result = testing_fd::RunInChild([] {
    ::unsetenv("DFKV_MDS_MAX_CONNS");
    rlimit limit{};
    if (::getrlimit(RLIMIT_NOFILE, &limit) != 0) return 10;
    const MdsDescriptorPlan expected =
        PlanMdsDescriptors(4096, 256, limit.rlim_max);
    limit.rlim_cur = 256;
    if (::setrlimit(RLIMIT_NOFILE, &limit) != 0) return 11;

    MdsServer mds(DeadEtcd());
    if (mds.Start(0) != Status::kOk) return 12;
    if (::getrlimit(RLIMIT_NOFILE, &limit) != 0) return 13;
    if (limit.rlim_cur != expected.soft_limit) return 14;
    if (mds.max_conns() != expected.max_conns) return 15;
    mds.Stop();
    return 0;
  });
  EXPECT_EQ(result, 0) << "step code from the child, see the test body";
}

// With no descriptor left accept() fails at once, every time, while the peer
// stays queued. The loop has to wait between attempts instead of spinning,
// serve the queued peer once descriptors are back, and still stop promptly.
TEST(MdsListener, DescriptorExhaustionBacksOffAndRecovers) {
  const int result = testing_fd::RunInChild([] {
    MdsServer mds(DeadEtcd());
    if (mds.Start(0) != Status::kOk) return 10;
    if (!Answered(mds.port())) return 11;
    if (!EventuallyTrue([&] { return mds.live_conn_count() == 0; })) return 12;

    // socket() needs a descriptor, so open them before taking the rest away.
    const int queued = ::socket(AF_INET, SOCK_STREAM, 0);
    const int later = ::socket(AF_INET, SOCK_STREAM, 0);
    if (queued < 0 || later < 0) return 13;

    testing_fd::NoFreeDescriptors exhausted;
    if (!exhausted.active()) return 14;
    if (!ConnectTo(queued, mds.port())) return 15;

    const double cpu_before = testing_fd::ProcessCpuSeconds();
    const auto wall_before = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    const double cpu_used = testing_fd::ProcessCpuSeconds() - cpu_before;
    const uint64_t failures = mds.AcceptFailures();
    if (failures == 0) return 16;   // the failure was not seen or not counted
    // One attempt per backoff, however long this thread actually slept.
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - wall_before)
            .count();
    if (failures > static_cast<uint64_t>(elapsed_ms) /
                           net::AcceptGuard::kBackoffMs + 2)
      return 17;  // retried without waiting
    if (cpu_used > 0.2) return 18;  // spun instead of sleeping

    exhausted.Restore();
    if (!AnsweredOn(queued)) return 19;  // the queued peer is served afterwards
    if (mds.ListenerMetricsText().find("dfkv_mds_accept_failures_total " +
                                       std::to_string(failures) + "\n") ==
        std::string::npos)
      return 20;

    // Stopping while the loop is backing off must not wait out anything long.
    testing_fd::NoFreeDescriptors exhausted_again;
    if (!exhausted_again.active()) return 21;
    if (!ConnectTo(later, mds.port())) return 22;
    if (!EventuallyTrue([&] { return mds.AcceptFailures() > failures; }))
      return 23;
    const auto stop_started = std::chrono::steady_clock::now();
    mds.Stop();
    if (std::chrono::steady_clock::now() - stop_started >
        std::chrono::seconds(2))
      return 24;
    return 0;
  });
  EXPECT_EQ(result, 0) << "step code from the child, see the test body";
}

TEST(MdsListener, ThreadThatCannotStartRefusesOnlyThatConnection) {
  MdsServer mds(DeadEtcd());
  ASSERT_EQ(mds.Start(0), Status::kOk);
  ASSERT_TRUE(Answered(mds.port()));

  net::g_fail_thread_starts_for_test.store(1);
  EXPECT_FALSE(Answered(mds.port())) << "the connection without a thread is closed";
  net::g_fail_thread_starts_for_test.store(0);
  EXPECT_EQ(mds.ThreadStartFailures(), 1u);

  EXPECT_TRUE(Answered(mds.port())) << "the MDS keeps serving";
  EXPECT_NE(mds.ListenerMetricsText().find(
                "dfkv_mds_thread_start_failures_total 1\n"),
            std::string::npos);
  mds.Stop();
}
