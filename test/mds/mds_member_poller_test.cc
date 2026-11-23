#include "mds/mds_member_poller.h"
#include "mds/mds_server.h"
#include "mds/mds_registrar.h"
#include "common/membership.h"
#include "transport/wire.h"
#include "utils/net_util.h"
#include "utils/wire_limits.h"
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
using namespace dfkv;  // NOLINT

namespace {
const char* EtcdEp() { return std::getenv("DFKV_TEST_ETCD"); }
}  // namespace

TEST(MdsMemberPoller, PollFiresOnChangeAndDedupsByEpoch) {
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  std::string mds_ep = "127.0.0.1:" + std::to_string(mds.port());
  std::string group = "m3-poll-" + std::to_string(mds.port());

  MemberInfo a{"pa", "10.2.2.1", 28000, 1}, b{"pb", "10.2.2.2", 28000, 2};
  MdsRegistrar ra({mds_ep}, group, a); ASSERT_TRUE(ra.RegisterOnce());
  MdsRegistrar rb({mds_ep}, group, b); ASSERT_TRUE(rb.RegisterOnce());

  std::mutex mu; std::vector<MemberInfo> last; int fires = 0;
  MdsMemberPoller poller({mds_ep}, group,
      [&](const std::vector<MemberInfo>& ms) {
        std::lock_guard<std::mutex> lk(mu); last = ms; ++fires;
      });
  ASSERT_TRUE(poller.PollOnce());
  { std::lock_guard<std::mutex> lk(mu); EXPECT_EQ(fires, 1); EXPECT_EQ(last.size(), 2u); }
  ASSERT_TRUE(poller.PollOnce());
  { std::lock_guard<std::mutex> lk(mu); EXPECT_EQ(fires, 1); }

  MemberInfo c{"pc", "10.2.2.3", 28000, 1};
  MdsRegistrar rc({mds_ep}, group, c); ASSERT_TRUE(rc.RegisterOnce());
  ASSERT_TRUE(poller.PollOnce());
  { std::lock_guard<std::mutex> lk(mu); EXPECT_EQ(fires, 2); EXPECT_EQ(last.size(), 3u); }
  mds.Stop();
}

TEST(MdsMemberPoller, BackgroundLoopPicksUpMembers) {
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  std::string mds_ep = "127.0.0.1:" + std::to_string(mds.port());
  std::string group = "m3-bg-" + std::to_string(mds.port());
  MemberInfo a{"qa", "10.2.3.1", 28000, 1};
  MdsRegistrar ra({mds_ep}, group, a); ASSERT_TRUE(ra.RegisterOnce());

  std::atomic<int> seen{0};
  MdsMemberPoller poller({mds_ep}, group,
      [&](const std::vector<MemberInfo>& ms) { seen.store((int)ms.size()); },
      /*poll_ms=*/100);
  poller.Start();
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (seen.load() < 1 && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(seen.load(), 1);
  poller.Stop();
  mds.Stop();
}

namespace {
// In-process fake MDS that answers kListMembers with a scripted sequence of
// member counts (one entry consumed per poll; the last entry repeats). Lets us
// drive the empty-view guard deterministically without waiting on lease expiry.
class ScriptedMds {
 public:
  explicit ScriptedMds(std::vector<int> counts) : counts_(std::move(counts)) {
    lfd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ::bind(lfd_, reinterpret_cast<sockaddr*>(&sa), sizeof(sa));
    ::listen(lfd_, 8);
    socklen_t sl = sizeof(sa);
    ::getsockname(lfd_, reinterpret_cast<sockaddr*>(&sa), &sl);
    port_ = ntohs(sa.sin_port);
    th_ = std::thread([this] { Serve(); });
  }
  ~ScriptedMds() {
    ::shutdown(lfd_, SHUT_RDWR);
    ::close(lfd_);
    if (th_.joinable()) th_.join();
  }
  int port() const { return port_; }
  std::string ep() const { return "127.0.0.1:" + std::to_string(port_); }

 private:
  void Serve() {
    size_t idx = 0;
    for (;;) {
      int fd = ::accept(lfd_, nullptr, nullptr);
      if (fd < 0) return;
      char pre[kReqPrefix];
      if (net::ReadAll(fd, pre, kReqPrefix)) {
        ReqFields rq;
        if (DecodeReq(pre, &rq, wire_limits::kMdsMaxReqPayload) && rq.payload_len) {
          std::string sink(rq.payload_len, '\0');
          net::ReadAll(fd, &sink[0], rq.payload_len);
        }
        int n = counts_[std::min(idx, counts_.size() - 1)];
        ++idx;
        std::vector<MemberInfo> ms;
        for (int i = 0; i < n; ++i)
          ms.push_back({"n" + std::to_string(i), "10.0.0." + std::to_string(i + 1),
                        28000, 1});
        // Epoch varies with content so the poller's dedup doesn't mask changes.
        std::string data = EncodeMembers(ms, MembersEpoch(ms));
        char rp[kRespPrefix];
        EncodeResp(rp, Status::kOk, data.size());
        net::WriteAll(fd, rp, kRespPrefix);
        if (!data.empty()) net::WriteAll(fd, data.data(), data.size());
      }
      ::close(fd);
    }
  }
  std::vector<int> counts_;
  int lfd_ = -1, port_ = 0;
  std::thread th_;
};
}  // namespace

TEST(MdsMemberPoller, EmptyViewGuardKeepsRingUntilPersistent) {
  // Sequence: 2 members, then empty x4. The first two empties are suppressed;
  // the third empty is accepted (kEmptyViewsToAccept=3).
  ScriptedMds mds({2, 0, 0, 0, 0});
  std::vector<size_t> fired_sizes;
  MdsMemberPoller poller({mds.ep()}, "g",
      [&](const std::vector<MemberInfo>& ms) { fired_sizes.push_back(ms.size()); });

  ASSERT_TRUE(poller.PollOnce());               // 2 members -> fire(2)
  ASSERT_EQ(fired_sizes.size(), 1u);
  EXPECT_EQ(fired_sizes[0], 2u);

  ASSERT_TRUE(poller.PollOnce());               // empty #1 -> suppressed
  ASSERT_TRUE(poller.PollOnce());               // empty #2 -> suppressed
  EXPECT_EQ(fired_sizes.size(), 1u) << "ring must be kept while empties are transient";
  EXPECT_EQ(poller.empty_view_rejected(), 2u);

  ASSERT_TRUE(poller.PollOnce());               // empty #3 -> accepted -> fire(0)
  ASSERT_EQ(fired_sizes.size(), 2u);
  EXPECT_EQ(fired_sizes[1], 0u);
}

TEST(MdsMemberPoller, EmptyViewGuardResetsOnRecovery) {
  // Sequence: 2, empty, empty, then members return before the 3rd empty.
  ScriptedMds mds({2, 0, 0, 3, 3});
  std::vector<size_t> fired_sizes;
  MdsMemberPoller poller({mds.ep()}, "g",
      [&](const std::vector<MemberInfo>& ms) { fired_sizes.push_back(ms.size()); });

  ASSERT_TRUE(poller.PollOnce());  // fire(2)
  ASSERT_TRUE(poller.PollOnce());  // empty #1 suppressed
  ASSERT_TRUE(poller.PollOnce());  // empty #2 suppressed
  ASSERT_TRUE(poller.PollOnce());  // 3 members -> fire(3), counter reset
  ASSERT_EQ(fired_sizes.size(), 2u) << "ring never went empty";
  EXPECT_EQ(fired_sizes[0], 2u);
  EXPECT_EQ(fired_sizes[1], 3u);
  EXPECT_EQ(poller.empty_view_rejected(), 2u);
}

TEST(MdsMemberPoller, FirstViewEmptyIsAdopted) {
  // No prior non-empty view -> an initial empty result is adopted as-is
  // (preserves the existing greenfield behavior).
  ScriptedMds mds({0});
  int fires = 0;
  MdsMemberPoller poller({mds.ep()}, "g",
      [&](const std::vector<MemberInfo>&) { ++fires; });
  ASSERT_TRUE(poller.PollOnce());
  EXPECT_EQ(fires, 1);
  EXPECT_EQ(poller.empty_view_rejected(), 0u);
}

TEST(MdsMemberPoller, FailsOverPastDeadEndpointAcrossPolls) {
  // Mirrors dfkvctl's QueryMembers: a dead first endpoint must not fail the
  // query while a healthy MDS is listed. One PollOnce picks one endpoint;
  // MarkFailed's backoff steers the next Pick to the live one, so a loop of
  // eps.size() PollOnce calls succeeds.
  ScriptedMds live({2});                       // healthy MDS, 2 members
  const std::string dead = "127.0.0.1:9";      // discard port: connect refused fast
  std::vector<std::string> eps = {dead, live.ep()};
  std::vector<MemberInfo> got;
  MdsMemberPoller poller(eps, "g",
      [&](const std::vector<MemberInfo>& ms) { got = ms; });

  bool ok = false;
  for (size_t i = 0; i < eps.size() && !ok; ++i) ok = poller.PollOnce();
  EXPECT_TRUE(ok);
  EXPECT_EQ(got.size(), 2u) << "failover to the live MDS must return its members";
}

// ReportHealth() is Loop()'s per-poll health helper (query_ok = PollOnce()'s
// result). It is what turns a silently-unreachable MDS into a visible signal,
// so unit-test its state machine directly — no etcd/MDS needed.
TEST(MdsMemberPoller, ReportHealthTracksReachabilityAndCounts) {
  MdsMemberPoller poller({"127.0.0.1:9"}, "g",
                         [](const std::vector<MemberInfo>&) {});
  EXPECT_TRUE(poller.reachable());
  EXPECT_EQ(poller.unreachable_polls_total(), 0u);

  // Each failed poll marks the client unreachable and bumps the counter.
  poller.ReportHealth(false);
  EXPECT_FALSE(poller.reachable());
  EXPECT_EQ(poller.unreachable_polls_total(), 1u);
  poller.ReportHealth(false);
  EXPECT_FALSE(poller.reachable());
  EXPECT_EQ(poller.unreachable_polls_total(), 2u);

  // A successful poll flips back to reachable; the counter is monotonic.
  poller.ReportHealth(true);
  EXPECT_TRUE(poller.reachable());
  EXPECT_EQ(poller.unreachable_polls_total(), 2u);

  // A subsequent outage keeps counting from where it left off.
  poller.ReportHealth(false);
  EXPECT_FALSE(poller.reachable());
  EXPECT_EQ(poller.unreachable_polls_total(), 3u);
}

// End-to-end: a poll against an unreachable endpoint fails, never populates the
// ring, and (via ReportHealth, as Loop() calls it) is marked unreachable — the
// exact scenario a mistyped mds_endpoint used to hit 100% silently.
TEST(MdsMemberPoller, UnreachableEndpointIsSurfacedNotSilent) {
  int fires = 0;
  MdsMemberPoller poller({"127.0.0.1:9"}, "g",  // discard port: connect refused fast
                         [&](const std::vector<MemberInfo>&) { ++fires; });
  EXPECT_FALSE(poller.PollOnce()) << "unreachable MDS must fail the poll";
  EXPECT_EQ(fires, 0) << "the ring must never be populated";
  poller.ReportHealth(false);  // Loop() feeds PollOnce()'s result here
  EXPECT_FALSE(poller.reachable());
  EXPECT_GE(poller.unreachable_polls_total(), 1u);
}

// Stop() must not lose its wakeup: a flag flipped without the mutex can land
// between the loop's check of it and its wait, and Stop then waits a whole
// poll period. Many Start/Stop cycles against an MDS that refuses at once;
// none may take near a poll period.
TEST(MdsMemberPoller, StopDoesNotWaitAPollPeriod) {
  MdsMemberPoller poller({"127.0.0.1:1"}, "stop-wakeup",
                         [](const std::vector<MemberInfo>&) {},
                         /*poll_ms=*/2000, /*io_timeout_ms=*/50);
  double worst_ms = 0;
  for (int i = 0; i < 3000; ++i) {
    poller.Start();
    if (i % 3) std::this_thread::sleep_for(std::chrono::microseconds(i % 50));
    const auto t0 = std::chrono::steady_clock::now();
    poller.Stop();
    worst_ms = std::max(worst_ms, std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - t0)
                                      .count());
  }
  EXPECT_LT(worst_ms, 1000.0);
  std::printf("StopDoesNotWaitAPollPeriod worst stop %.1f ms\n", worst_ms);
}
