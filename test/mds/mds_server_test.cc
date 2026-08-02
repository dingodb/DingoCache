#include "mds/mds_server.h"
#include "mds/mds_proto.h"
#include "common/membership.h"
#include "transport/wire.h"
#include "utils/net_util.h"
#include <gtest/gtest.h>
#include <unistd.h>
#include <cstdlib>
#include <string>
#include <vector>
using namespace dfkv;  // NOLINT

namespace {
const char* EtcdEp() { return std::getenv("DFKV_TEST_ETCD"); }

bool DoReq(int port, WireOp op, const std::string& payload, Status* st, std::string* data) {
  int fd = net::Dial("127.0.0.1:" + std::to_string(port), 2000, 2000);
  if (fd < 0) return false;
  char pre[kReqPrefix];
  EncodeReq(pre, op, BlockKey{}, 0, 0, payload.size());
  if (!net::WriteAll(fd, pre, kReqPrefix) ||
      (!payload.empty() && !net::WriteAll(fd, payload.data(), payload.size()))) { ::close(fd); return false; }
  char rp[kRespPrefix];
  if (!net::ReadAll(fd, rp, kRespPrefix)) { ::close(fd); return false; }
  Status s; uint64_t dlen = 0;
  if (!DecodeResp(rp, &s, &dlen)) { ::close(fd); return false; }
  data->resize(dlen);
  if (dlen && !net::ReadAll(fd, &(*data)[0], dlen)) { ::close(fd); return false; }
  *st = s; ::close(fd); return true;
}
}  // namespace

TEST(MdsServer, RegisterThenListRoundTripsThroughEtcd) {
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD=host:port";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  int port = mds.port();
  std::string group = "itest-grp-" + std::to_string(port);

  MemberInfo a{"na", "10.1.1.1", 28000, 1};
  MemberInfo b{"nb", "10.1.1.2", 28000, 3};
  Status st; std::string data;
  ASSERT_TRUE(DoReq(port, WireOp::kRegister, EncodeMemberReq(group, a), &st, &data));
  EXPECT_EQ(st, Status::kOk);
  ASSERT_TRUE(DoReq(port, WireOp::kRegister, EncodeMemberReq(group, b), &st, &data));
  EXPECT_EQ(st, Status::kOk);
  ASSERT_TRUE(DoReq(port, WireOp::kHeartbeat, EncodeMemberReq(group, a), &st, &data));
  EXPECT_EQ(st, Status::kOk);

  ASSERT_TRUE(DoReq(port, WireOp::kListMembers, group, &st, &data));
  ASSERT_EQ(st, Status::kOk);
  std::vector<MemberInfo> got; uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(data.data(), data.size(), &got, &epoch));
  EXPECT_GT(epoch, 0u);
  ASSERT_EQ(got.size(), 2u);
  bool sawA = false, sawB = false;
  for (auto& m : got) { if (m == a) sawA = true; if (m == b) sawB = true; }
  EXPECT_TRUE(sawA); EXPECT_TRUE(sawB);
  mds.Stop();
}

TEST(MdsServer, ListEmptyGroupOkEmpty) {
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD=host:port";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  Status st; std::string data;
  ASSERT_TRUE(DoReq(mds.port(), WireOp::kListMembers,
                    "empty-grp-" + std::to_string(mds.port()), &st, &data));
  ASSERT_EQ(st, Status::kOk);
  std::vector<MemberInfo> got; uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(data.data(), data.size(), &got, &epoch));
  EXPECT_TRUE(got.empty());
  mds.Stop();
}

TEST(MdsServer, HeartbeatRewritesMemberValueUnderOwnLease) {
  // Regression for the multi-MDS lease-drift bug: the fast path used to only
  // LeaseKeepAlive and never re-Put the key, so (a) a MemberInfo change within
  // the TTL was silently dropped, and (b) with several rotating MDS instances
  // the key stayed attached to another MDS's decaying lease while every
  // heartbeat still returned kOk. Both reduce to the same observable pinned
  // here: a heartbeat must re-write the key's current value.
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD=host:port";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  int port = mds.port();
  std::string group = "itest-reput-" + std::to_string(port);

  MemberInfo m{"n1", "10.1.1.1", 28000, 1};
  Status st; std::string data;
  ASSERT_TRUE(DoReq(port, WireOp::kRegister, EncodeMemberReq(group, m), &st, &data));
  ASSERT_EQ(st, Status::kOk);

  m.ip = "10.9.9.9";  // node reconfigured within the TTL
  m.weight = 7;
  ASSERT_TRUE(DoReq(port, WireOp::kHeartbeat, EncodeMemberReq(group, m), &st, &data));
  ASSERT_EQ(st, Status::kOk);

  ASSERT_TRUE(DoReq(port, WireOp::kListMembers, group, &st, &data));
  ASSERT_EQ(st, Status::kOk);
  std::vector<MemberInfo> got; uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(data.data(), data.size(), &got, &epoch));
  ASSERT_EQ(got.size(), 1u);
  EXPECT_EQ(got[0].ip, "10.9.9.9");
  EXPECT_EQ(got[0].weight, 7u);
  mds.Stop();
}

TEST(MdsServer, RejectsPathTraversalGroupAndId) {
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD=host:port";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  int port = mds.port();
  Status st; std::string data;

  // A group containing "/members/" would land the key inside group "victim".
  std::string victim = "victim-" + std::to_string(port);
  MemberInfo ghost{"ghost", "6.6.6.6", 28000, 1};
  std::string evil_group = victim + "/members/x/members";
  ASSERT_TRUE(DoReq(port, WireOp::kRegister, EncodeMemberReq(evil_group, ghost),
                    &st, &data));
  EXPECT_EQ(st, Status::kInvalid) << "path-traversal group must be rejected";

  // An id with a slash is equally dangerous.
  MemberInfo m{"a/members/x", "1.1.1.1", 28000, 1};
  ASSERT_TRUE(DoReq(port, WireOp::kRegister, EncodeMemberReq(victim, m), &st, &data));
  EXPECT_EQ(st, Status::kInvalid) << "path-traversal id must be rejected";

  // The victim group must contain no injected phantom member.
  ASSERT_TRUE(DoReq(port, WireOp::kListMembers, victim, &st, &data));
  ASSERT_EQ(st, Status::kOk);
  std::vector<MemberInfo> got; uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(data.data(), data.size(), &got, &epoch));
  EXPECT_TRUE(got.empty()) << "no phantom member should have been injected";

  // A malformed group on ListMembers is rejected outright.
  ASSERT_TRUE(DoReq(port, WireOp::kListMembers, "a/b", &st, &data));
  EXPECT_EQ(st, Status::kInvalid);
  mds.Stop();
}

TEST(MdsServer, ProbeEtcdReflectsReachability) {
  // Dead endpoint: probe must fail fast (used at startup to exit non-zero on a
  // misconfigured --etcd instead of running "up" with silent write failures).
  { MdsServer dead("127.0.0.1:9", /*timeout_ms=*/500);
    EXPECT_FALSE(dead.ProbeEtcd()); }

  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD for the reachable case";
  MdsServer live(ep);
  EXPECT_TRUE(live.ProbeEtcd());
}

TEST(MdsServer, InfoFlowsRegisterThroughEtcdToList) {
  // The node self-description (MemberInfo.info) must survive the FULL path:
  // register payload -> MDS decode -> etcd member value -> ListMembers decode
  // -> response encode. A heartbeat carrying updated info must also refresh it
  // (the heartbeat fast path re-puts the value under the lease).
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD=host:port";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  int port = mds.port();
  std::string group = "itest-info-" + std::to_string(port);

  MemberInfo m{"ni", "10.1.1.9", 28001, 2, 28100,
               "ver=1.8.0,engine=slab,disks=3,cap=5497558138880,ram=0,rdma=ib7s400p0"};
  Status st; std::string data;
  ASSERT_TRUE(DoReq(port, WireOp::kRegister, EncodeMemberReq(group, m), &st, &data));
  EXPECT_EQ(st, Status::kOk);

  ASSERT_TRUE(DoReq(port, WireOp::kListMembers, group, &st, &data));
  ASSERT_EQ(st, Status::kOk);
  std::vector<MemberInfo> got; uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(data.data(), data.size(), &got, &epoch));
  ASSERT_EQ(got.size(), 1u);
  EXPECT_EQ(got[0].info, m.info) << "info must round-trip through etcd";
  EXPECT_EQ(got[0].tcp_port, 28100u);

  // Heartbeat with CHANGED info (e.g. node restarted with a new version):
  // the re-put must propagate it, and the epoch must NOT change (info is
  // excluded from MembersEpoch -- no needless client ring rebuilds).
  uint64_t epoch_before = epoch;
  m.info = "ver=1.9.0,engine=slab,disks=3,cap=5497558138880,ram=0,rdma=ib7s400p0";
  ASSERT_TRUE(DoReq(port, WireOp::kHeartbeat, EncodeMemberReq(group, m), &st, &data));
  EXPECT_EQ(st, Status::kOk);
  ASSERT_TRUE(DoReq(port, WireOp::kListMembers, group, &st, &data));
  ASSERT_EQ(st, Status::kOk);
  ASSERT_TRUE(DecodeMembers(data.data(), data.size(), &got, &epoch));
  ASSERT_EQ(got.size(), 1u);
  EXPECT_EQ(got[0].info, m.info) << "heartbeat must refresh info";
  EXPECT_EQ(epoch, epoch_before) << "info change must not bump the ring epoch";
  mds.Stop();
}

TEST(MdsServer, StatsFlowThroughEtcdAndAggregateInMetrics) {
  // STA1 stats: register -> etcd -> ListMembers round-trip; heartbeat refresh;
  // GroupMetricsText aggregates per group; ListGroups enumerates the group.
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD=host:port";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  int port = mds.port();
  std::string group = "itest-stats-" + std::to_string(port);

  MemberInfo m{"ns", "10.1.1.10", 28001, 1, 28100, "ver=1.10.0"};
  m.stats.capacity_bytes = 1000;
  m.stats.used_bytes = 250;
  m.stats.objects = 5;
  m.stats.hits_total = 90;
  m.stats.misses_total = 10;
  m.has_stats = true;
  Status st; std::string data;
  ASSERT_TRUE(DoReq(port, WireOp::kRegister, EncodeMemberReq(group, m), &st, &data));
  EXPECT_EQ(st, Status::kOk);

  // Round-trip via kListMembers
  ASSERT_TRUE(DoReq(port, WireOp::kListMembers, group, &st, &data));
  ASSERT_EQ(st, Status::kOk);
  std::vector<MemberInfo> got; uint64_t epoch1 = 0;
  ASSERT_TRUE(DecodeMembers(data.data(), data.size(), &got, &epoch1));
  ASSERT_EQ(got.size(), 1u);
  ASSERT_TRUE(got[0].has_stats);
  EXPECT_EQ(got[0].stats.used_bytes, 250u);
  EXPECT_EQ(got[0].stats.hits_total, 90u);

  // Heartbeat with UPDATED stats refreshes the value; epoch must NOT move.
  m.stats.used_bytes = 400;
  ASSERT_TRUE(DoReq(port, WireOp::kHeartbeat, EncodeMemberReq(group, m), &st, &data));
  EXPECT_EQ(st, Status::kOk);
  ASSERT_TRUE(DoReq(port, WireOp::kListMembers, group, &st, &data));
  std::vector<MemberInfo> got2; uint64_t epoch2 = 0;
  ASSERT_TRUE(DecodeMembers(data.data(), data.size(), &got2, &epoch2));
  ASSERT_EQ(got2.size(), 1u);
  EXPECT_EQ(got2[0].stats.used_bytes, 400u);
  EXPECT_EQ(epoch1, epoch2) << "stats churn must not move the ring epoch";

  // Group aggregates appear in the MDS metrics text.
  const std::string mt = mds.MetricsText();
  EXPECT_NE(mt.find("dfkv_mds_group_used_bytes{group=\"" + group + "\"} 400"),
            std::string::npos) << mt;
  EXPECT_NE(mt.find("dfkv_mds_group_capacity_bytes{group=\"" + group + "\"} 1000"),
            std::string::npos);
  EXPECT_NE(mt.find("dfkv_mds_group_nodes{group=\"" + group + "\"} 1"),
            std::string::npos);
  EXPECT_NE(mt.find("dfkv_mds_group_stats_missing{group=\"" + group + "\"} 0"),
            std::string::npos);

  // kListGroups enumerates it.
  ASSERT_TRUE(DoReq(port, WireOp::kListGroups, "", &st, &data));
  ASSERT_EQ(st, Status::kOk);
  EXPECT_NE(data.find(group), std::string::npos);
}

// ---- Client (consumer) registration -----------------------------------------
// Clients register under /clients/<id> with the SAME payload (group + MemberInfo)
// as members, but the MDS writes a disjoint etcd prefix so they never enter the
// placement ring. These tests pin that contract: register/list round-trip,
// heartbeat refresh, isolation from ListMembers, path-traversal rejection, and
// the per-group clients gauge.

TEST(MdsServer, ClientRegisterThenListRoundTripsThroughEtcd) {
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD=host:port";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  int port = mds.port();
  std::string group = "itest-cli-" + std::to_string(port);

  // A consumer's ip/port/weight carry no placement meaning; only id + info
  // (identity) are load-bearing.
  MemberInfo a{"ca", "0.0.0.0", 0, 0, 0, "type=vllm,model=glm51,role=kv_producer,tp_size=8"};
  MemberInfo b{"cb", "0.0.0.0", 0, 0, 0, "type=lmcache,model=glm51,role=kv_consumer,tp_size=8"};
  Status st; std::string data;
  ASSERT_TRUE(DoReq(port, WireOp::kClientRegister, EncodeMemberReq(group, a), &st, &data));
  EXPECT_EQ(st, Status::kOk);
  ASSERT_TRUE(DoReq(port, WireOp::kClientRegister, EncodeMemberReq(group, b), &st, &data));
  EXPECT_EQ(st, Status::kOk);

  ASSERT_TRUE(DoReq(port, WireOp::kListClients, group, &st, &data));
  ASSERT_EQ(st, Status::kOk);
  std::vector<MemberInfo> got; uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(data.data(), data.size(), &got, &epoch));
  EXPECT_GT(epoch, 0u);
  ASSERT_EQ(got.size(), 2u);
  bool sawA = false, sawB = false;
  for (auto& c : got) { if (c == a) sawA = true; if (c == b) sawB = true; }
  EXPECT_TRUE(sawA); EXPECT_TRUE(sawB);
  // info (identity) must round-trip — `dfkvctl clients` parses it.
  for (auto& c : got) EXPECT_FALSE(c.info.empty());
  mds.Stop();
}

TEST(MdsServer, ClientKeyDoesNotPolluteMembers) {
  // The whole point of the separate prefix: registering a client must NOT make
  // it appear in ListMembers (the placement ring), nor inflate group_nodes.
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD=host:port";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  int port = mds.port();
  std::string group = "itest-iso-" + std::to_string(port);

  MemberInfo c{"c1", "0.0.0.0", 0, 0, 0, "type=vllm"};
  Status st; std::string data;
  ASSERT_TRUE(DoReq(port, WireOp::kClientRegister, EncodeMemberReq(group, c), &st, &data));
  ASSERT_EQ(st, Status::kOk);

  ASSERT_TRUE(DoReq(port, WireOp::kListMembers, group, &st, &data));
  ASSERT_EQ(st, Status::kOk);
  std::vector<MemberInfo> members; uint64_t e = 0;
  ASSERT_TRUE(DecodeMembers(data.data(), data.size(), &members, &e));
  EXPECT_TRUE(members.empty()) << "client must not appear in the placement ring";

  // The clients gauge counts it, but group_nodes stays 0.
  const std::string mt = mds.MetricsText();
  EXPECT_NE(mt.find("dfkv_mds_group_clients{group=\"" + group + "\"} 1"),
            std::string::npos) << mt;
  EXPECT_NE(mt.find("dfkv_mds_group_nodes{group=\"" + group + "\"} 0"),
            std::string::npos) << mt;
  mds.Stop();
}

TEST(MdsServer, ClientHeartbeatRewritesInfo) {
  // Mirrors HeartbeatRewritesMemberValueUnderOwnLease: a heartbeat with changed
  // identity info must re-put the value (the lease-drift fix applies to clients
  // too, since UpsertClient shares UpsertLeased).
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD=host:port";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  int port = mds.port();
  std::string group = "itest-cli-hb-" + std::to_string(port);

  MemberInfo c{"ch", "0.0.0.0", 0, 0, 0, "type=vllm,ver=1.0"};
  Status st; std::string data;
  ASSERT_TRUE(DoReq(port, WireOp::kClientRegister, EncodeMemberReq(group, c), &st, &data));
  ASSERT_EQ(st, Status::kOk);

  c.info = "type=vllm,ver=2.0";  // connector upgraded within the TTL
  ASSERT_TRUE(DoReq(port, WireOp::kClientHeartbeat, EncodeMemberReq(group, c), &st, &data));
  ASSERT_EQ(st, Status::kOk);

  ASSERT_TRUE(DoReq(port, WireOp::kListClients, group, &st, &data));
  ASSERT_EQ(st, Status::kOk);
  std::vector<MemberInfo> got; uint64_t e = 0;
  ASSERT_TRUE(DecodeMembers(data.data(), data.size(), &got, &e));
  ASSERT_EQ(got.size(), 1u);
  EXPECT_EQ(got[0].info, "type=vllm,ver=2.0") << "heartbeat must refresh client info";
  mds.Stop();
}

TEST(MdsServer, ClientRejectsPathTraversalGroupAndId) {
  // A malicious client must not inject keys into another group's /clients/ (or
  // /members/) subtree via a '/' in group or id. Same guard as member registration.
  const char* ep = EtcdEp();
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD=host:port";
  MdsServer mds(ep);
  ASSERT_EQ(mds.Start(0), Status::kOk);
  int port = mds.port();
  Status st; std::string data;
  std::string victim = "vcli-" + std::to_string(port);

  MemberInfo ghost{"ghost", "0.0.0.0", 0, 0, 0, "type=vllm"};
  std::string evil_group = victim + "/clients/x/clients";
  ASSERT_TRUE(DoReq(port, WireOp::kClientRegister, EncodeMemberReq(evil_group, ghost),
                    &st, &data));
  EXPECT_EQ(st, Status::kInvalid);

  MemberInfo m{"a/clients/x", "0.0.0.0", 0, 0, 0, "type=vllm"};
  ASSERT_TRUE(DoReq(port, WireOp::kClientRegister, EncodeMemberReq(victim, m), &st, &data));
  EXPECT_EQ(st, Status::kInvalid);

  ASSERT_TRUE(DoReq(port, WireOp::kListClients, victim, &st, &data));
  ASSERT_EQ(st, Status::kOk);
  std::vector<MemberInfo> got; uint64_t e = 0;
  ASSERT_TRUE(DecodeMembers(data.data(), data.size(), &got, &e));
  EXPECT_TRUE(got.empty()) << "no phantom client injected";

  ASSERT_TRUE(DoReq(port, WireOp::kListClients, "a/b", &st, &data));
  EXPECT_EQ(st, Status::kInvalid);
  mds.Stop();
}

// No etcd needed: the listener + handler lifecycle is all local. A peer that
// connects and then goes silent must be released by the I/O timeout instead of
// pinning its handler thread forever (thread-per-conn with no bound).
TEST(MdsServer, SilentPeerReleasedByIoTimeout) {
  ::setenv("DFKV_MDS_IO_TIMEOUT_S", "1", 1);
  MdsServer srv("127.0.0.1:1");  // etcd never contacted: no requests sent
  ASSERT_EQ(srv.Start(0), Status::kOk);
  int fd = net::Dial("127.0.0.1:" + std::to_string(srv.port()), 2000, 2000);
  ASSERT_GE(fd, 0);
  // Handler picks the conn up and blocks in ReadAll until the 1 s timeout.
  for (int i = 0; i < 50 && srv.live_conn_count() == 0; ++i) usleep(20 * 1000);
  EXPECT_EQ(srv.live_conn_count(), 1u);
  for (int i = 0; i < 200 && srv.live_conn_count() > 0; ++i) usleep(20 * 1000);
  EXPECT_EQ(srv.live_conn_count(), 0u) << "silent conn not reaped by timeout";
  ::close(fd);
  srv.Stop();
  ::unsetenv("DFKV_MDS_IO_TIMEOUT_S");
}

TEST(LocalLeaseMap, ChurnIsBoundedByIdleWindow) {
  LocalLeaseMap leases;
  constexpr uint64_t kStaleMs = 120000;
  size_t pruned = 0;
  for (uint64_t i = 0; i < 10000; ++i) {
    const uint64_t now_ms = i * 1000;
    pruned += leases.MaybePrune(now_ms, kStaleMs);
    leases.Store("member-" + std::to_string(i), static_cast<int64_t>(i + 1),
                 now_ms);
  }
  EXPECT_LE(leases.Size(), 150u);
  EXPECT_GT(pruned, 9800u);
}

TEST(LocalLeaseMap, RecentUseSurvivesWhileStaleEntryIsForgotten) {
  LocalLeaseMap leases;
  leases.Store("cold", 1, 1000);
  leases.Store("hot", 2, 1000);
  int64_t lease_id = 0;
  ASSERT_TRUE(leases.LookupAndTouch("hot", 100000, &lease_id));
  EXPECT_EQ(lease_id, 2);

  EXPECT_EQ(leases.Prune(122000, 120000), 1u);
  EXPECT_EQ(leases.Size(), 1u);
  EXPECT_FALSE(leases.LookupAndTouch("cold", 122000, &lease_id));
  EXPECT_TRUE(leases.LookupAndTouch("hot", 122000, &lease_id));
}
