// What the MDS writes to etcd for heartbeats, and how the registrar picks an
// MDS. Runs against the
// in-process fake of the etcd JSON gateway (fake_etcd.h): no etcd needed.
#include "mds/mds_server.h"

#include <gtest/gtest.h>

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "common/membership.h"
#include "fake_etcd.h"
#include "mds/mds_proto.h"
#include "mds/mds_registrar.h"
#include "transport/wire.h"
#include "utils/net_util.h"

using namespace dfkv;  // NOLINT

namespace {

class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) {
      had_ = true;
      old_ = old;
    }
    if (value)
      ::setenv(name, value, 1);
    else
      ::unsetenv(name);
  }
  ~ScopedEnv() {
    if (had_)
      ::setenv(name_, old_.c_str(), 1);
    else
      ::unsetenv(name_);
  }

 private:
  const char* name_;
  bool had_ = false;
  std::string old_;
};

bool DoReq(int port, WireOp op, const std::string& payload, Status* st,
           std::string* data) {
  const int fd = net::Dial("127.0.0.1:" + std::to_string(port), 2000, 5000);
  if (fd < 0) return false;
  char pre[kReqPrefix];
  EncodeReq(pre, op, BlockKey{}, 0, 0, payload.size());
  bool ok = net::WriteAll(fd, pre, kReqPrefix) &&
            (payload.empty() ||
             net::WriteAll(fd, payload.data(), payload.size()));
  char rp[kRespPrefix];
  uint64_t dlen = 0;
  ok = ok && net::ReadAll(fd, rp, kRespPrefix) && DecodeResp(rp, st, &dlen);
  if (ok) {
    data->resize(dlen);
    ok = dlen == 0 || net::ReadAll(fd, &(*data)[0], dlen);
  }
  ::close(fd);
  return ok;
}

Status Beat(const MdsServer& mds, WireOp op, const std::string& group,
            const MemberInfo& m) {
  Status st = Status::kInvalid;
  std::string data;
  EXPECT_TRUE(DoReq(mds.port(), op, EncodeMemberReq(group, m), &st, &data));
  return st;
}

std::set<std::string> ListIds(const MdsServer& mds, const std::string& group) {
  Status st = Status::kInvalid;
  std::string data;
  std::set<std::string> ids;
  EXPECT_TRUE(DoReq(mds.port(), WireOp::kListMembers, group, &st, &data));
  EXPECT_EQ(st, Status::kOk);
  std::vector<MemberInfo> members;
  uint64_t epoch = 0;
  if (st == Status::kOk && DecodeMembers(data.data(), data.size(), &members,
                                         &epoch))
    for (const auto& m : members) ids.insert(m.id);
  return ids;
}

// A counter of the MDS's metrics; for etcd operations pass
// `dfkv_mds_etcd_requests_total{op="put"}` and the like.
double Metric(MdsServer& mds, const std::string& series) {
  const std::string text = mds.MetricsText();
  const std::string line = "\n" + series + " ";
  const size_t at = text.find(line);
  if (at == std::string::npos) return -1;
  return std::strtod(text.c_str() + at + line.size(), nullptr);
}
double EtcdOps(MdsServer& mds, const char* op) {
  return Metric(mds, std::string("dfkv_mds_etcd_requests_total{op=\"") + op +
                         "\"}");
}

MemberInfo WithStats(MemberInfo m, uint64_t puts) {
  m.has_stats = true;
  m.stats.capacity_bytes = 1ull << 40;
  m.stats.puts_total = puts;
  return m;
}

const MemberInfo kA{"na", "10.1.1.1", 28000, 1};
const MemberInfo kB{"nb", "10.1.1.2", 28000, 1};

int ClosedPort() {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  ::bind(fd, reinterpret_cast<sockaddr*>(&sa), sizeof(sa));
  socklen_t sl = sizeof(sa);
  ::getsockname(fd, reinterpret_cast<sockaddr*>(&sa), &sl);
  ::close(fd);  // nothing listens there now
  return ntohs(sa.sin_port);
}

}  // namespace

// A heartbeat renews the lease the record carries and writes nothing when
// the record is unchanged. It used to Put the record on every heartbeat.
TEST(MdsEtcdWrites, HeartbeatOfAnUnchangedMemberWritesNothing) {
  FakeEtcd etcd;
  MdsServer mds(etcd.ep());
  ASSERT_EQ(mds.Start(0), Status::kOk);
  ASSERT_EQ(Beat(mds, WireOp::kRegister, "g", kA), Status::kOk);
  const double puts = EtcdOps(mds, "put");
  const double grants = EtcdOps(mds, "lease_grant");
  EXPECT_EQ(puts, 1);
  EXPECT_EQ(grants, 1);
  for (int i = 0; i < 3; ++i)
    ASSERT_EQ(Beat(mds, WireOp::kHeartbeat, "g", kA), Status::kOk);
  EXPECT_EQ(EtcdOps(mds, "put"), puts) << "heartbeats rewrote the record";
  EXPECT_EQ(EtcdOps(mds, "lease_grant"), grants);
  EXPECT_EQ(EtcdOps(mds, "lease_keepalive"), 3);
  EXPECT_EQ(Metric(mds, "dfkv_mds_record_puts_skipped_total"), 3);
  EXPECT_EQ(ListIds(mds, "g"), (std::set<std::string>{"na"}));
  mds.Stop();
}

// Stats change on every beat of a serving node. Alone they rewrite the
// record at most every DFKV_MDS_STATS_PUT_MS; any other change at once.
TEST(MdsEtcdWrites, StatsAloneAreRewrittenAtMostEveryInterval) {
  ScopedEnv interval("DFKV_MDS_STATS_PUT_MS", "400");
  FakeEtcd etcd;
  MdsServer mds(etcd.ep());
  ASSERT_EQ(mds.Start(0), Status::kOk);
  ASSERT_EQ(Beat(mds, WireOp::kRegister, "g", WithStats(kA, 1)), Status::kOk);
  ASSERT_EQ(EtcdOps(mds, "put"), 1);
  ASSERT_EQ(Beat(mds, WireOp::kHeartbeat, "g", WithStats(kA, 2)), Status::kOk);
  EXPECT_EQ(EtcdOps(mds, "put"), 1) << "stats alone rewrote the record early";
  std::this_thread::sleep_for(std::chrono::milliseconds(450));
  ASSERT_EQ(Beat(mds, WireOp::kHeartbeat, "g", WithStats(kA, 3)), Status::kOk);
  EXPECT_EQ(EtcdOps(mds, "put"), 2) << "stats not rewritten after the interval";
  MemberInfo moved = WithStats(kA, 3);
  moved.weight = 2;
  ASSERT_EQ(Beat(mds, WireOp::kHeartbeat, "g", moved), Status::kOk);
  EXPECT_EQ(EtcdOps(mds, "put"), 3) << "a placement change waited";
  mds.Stop();
}

// Any MDS renews the lease the record carries. A registrar that moved to
// another MDS cost a lease grant and a Put there, and left the key on a
// lease that only the MDS before kept alive.
TEST(MdsEtcdWrites, AnotherMdsRenewsTheLeaseTheRecordCarries) {
  FakeEtcd etcd;
  MdsServer first(etcd.ep()), second(etcd.ep());
  ASSERT_EQ(first.Start(0), Status::kOk);
  ASSERT_EQ(second.Start(0), Status::kOk);
  ASSERT_EQ(Beat(first, WireOp::kRegister, "g", kA), Status::kOk);
  for (int i = 0; i < 2; ++i)
    ASSERT_EQ(Beat(second, WireOp::kHeartbeat, "g", kA), Status::kOk);
  EXPECT_EQ(EtcdOps(second, "lease_grant"), 0);
  EXPECT_EQ(EtcdOps(second, "put"), 0);
  EXPECT_EQ(EtcdOps(second, "lease_keepalive"), 2);
  // Back on the first MDS: still nothing to write.
  ASSERT_EQ(Beat(first, WireOp::kHeartbeat, "g", kA), Status::kOk);
  EXPECT_EQ(EtcdOps(first, "put"), 1);
  first.Stop();
  second.Stop();
}

// A record whose lease has run out is written again under a new lease.
TEST(MdsEtcdWrites, AnExpiredLeaseIsReplaced) {
  FakeEtcd etcd;
  MdsServer mds(etcd.ep());
  ASSERT_EQ(mds.Start(0), Status::kOk);
  ASSERT_EQ(Beat(mds, WireOp::kRegister, "g", kA), Status::kOk);
  etcd.ExpireKey("/dfkv/v1/groups/g/members/na");
  ASSERT_EQ(Beat(mds, WireOp::kHeartbeat, "g", kA), Status::kOk);
  EXPECT_EQ(EtcdOps(mds, "lease_grant"), 2);
  EXPECT_EQ(EtcdOps(mds, "put"), 2);
  EXPECT_EQ(ListIds(mds, "g"), (std::set<std::string>{"na"}));
  mds.Stop();
}

// The registrar stays on one MDS while it answers, and when it fails tries
// the next at once, in the same heartbeat: an MDS that is down costs no
// failed heartbeat. It used to rotate, one MDS per heartbeat, and a dead one
// failed every heartbeat that came to it.
TEST(MdsRegistrarFailover, StaysOnOneMdsAndFailsOverWithinTheHeartbeat) {
  FakeEtcd etcd;
  MdsServer first(etcd.ep()), second(etcd.ep());
  ASSERT_EQ(first.Start(0), Status::kOk);
  ASSERT_EQ(second.Start(0), Status::kOk);
  const std::string dead = "127.0.0.1:" + std::to_string(ClosedPort());
  MdsRegistrar registrar(
      {dead, "127.0.0.1:" + std::to_string(first.port()),
       "127.0.0.1:" + std::to_string(second.port())},
      "g", kA, /*heartbeat_ms=*/150);
  registrar.Start();
  auto beats = [](MdsServer& mds) {
    return Metric(mds, "dfkv_mds_register_requests_total") +
           Metric(mds, "dfkv_mds_keepalives_total");
  };
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (beats(first) + beats(second) < 6 &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  ASSERT_GE(beats(first) + beats(second), 6);
  // All on one of the two (whichever the member id starts it on, or the
  // one after the dead endpoint).
  MdsServer& used = beats(first) > 0 ? first : second;
  MdsServer& other = &used == &first ? second : first;
  EXPECT_EQ(beats(other), 0) << "rotated instead of staying on one MDS";
  EXPECT_EQ(registrar.heartbeat_failures_total(), 0u)
      << "the dead MDS failed a heartbeat";
  used.Stop();
  const double at_other = beats(other);
  while (beats(other) < at_other + 4 &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_GE(beats(other), at_other + 4);
  EXPECT_EQ(registrar.heartbeat_failures_total(), 0u)
      << "a heartbeat failed while another MDS answered";
  registrar.Stop();
  other.Stop();
}

// Registrars start on an MDS chosen by their member id: members spread over
// the MDS instead of all loading the first and failing over together.
TEST(MdsRegistrarFailover, MembersSpreadOverTheMds) {
  FakeEtcd etcd;
  std::vector<std::unique_ptr<MdsServer>> servers;
  std::vector<std::string> eps;
  for (int i = 0; i < 3; ++i) {
    servers.push_back(std::make_unique<MdsServer>(etcd.ep()));
    ASSERT_EQ(servers.back()->Start(0), Status::kOk);
    eps.push_back("127.0.0.1:" + std::to_string(servers.back()->port()));
  }
  std::vector<std::unique_ptr<MdsRegistrar>> registrars;
  for (int i = 0; i < 12; ++i) {
    MemberInfo m{"node-" + std::to_string(i), "10.3.3." + std::to_string(i),
                 28000, 1};
    registrars.push_back(std::make_unique<MdsRegistrar>(eps, "g", m));
    ASSERT_TRUE(registrars.back()->RegisterOnce());
  }
  int used = 0;
  for (auto& s : servers)
    if (Metric(*s, "dfkv_mds_register_requests_total") > 0) ++used;
  EXPECT_EQ(used, 3) << "members did not spread over the MDS";
  for (auto& s : servers) s->Stop();
}
