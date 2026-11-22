// The member list while etcd answers reads but takes no writes.
//
// An etcd that is out of space refuses Put and LeaseGrant, keeps answering
// Range and LeaseKeepAlive, and keeps expiring leases. Members whose lease
// cannot be renewed then drop out of etcd although the nodes are healthy. The
// MDS must not hand that view to clients as the member list.
//
// Runs against an in-process fake of the etcd JSON gateway, so no etcd is
// needed: the fake can be told to refuse writes and to expire leases.
#include "mds/mds_server.h"

#include <gtest/gtest.h>

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "common/membership.h"
#include "mds/mds_member_poller.h"
#include "mds/mds_proto.h"
#include "transport/wire.h"
#include "transport/wire_legacy.h"
#include "utils/base64.h"
#include "utils/json_min.h"
#include "utils/net_util.h"

using namespace dfkv;  // NOLINT

namespace {

// The four gateway calls the MDS makes, over HTTP/1.1 with Connection: close.
class FakeEtcd {
 public:
  FakeEtcd() {
    lfd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ::bind(lfd_, reinterpret_cast<sockaddr*>(&sa), sizeof(sa));
    ::listen(lfd_, 64);
    socklen_t sl = sizeof(sa);
    ::getsockname(lfd_, reinterpret_cast<sockaddr*>(&sa), &sl);
    port_ = ntohs(sa.sin_port);
    th_ = std::thread([this] { Serve(); });
  }
  ~FakeEtcd() {
    ::shutdown(lfd_, SHUT_RDWR);
    ::close(lfd_);
    if (th_.joinable()) th_.join();
  }
  std::string ep() const { return "127.0.0.1:" + std::to_string(port_); }

  // Out of space: Put and LeaseGrant are refused; KeepAlive and Range answer.
  void RejectWrites(bool on) {
    std::lock_guard<std::mutex> lk(mu_);
    reject_writes_ = on;
  }
  // Every lease runs out: the leases and the keys attached to them are gone.
  void ExpireAllLeases() {
    std::lock_guard<std::mutex> lk(mu_);
    leases_.clear();
    kvs_.clear();
  }
  // The lease holding this one key runs out.
  void ExpireKey(const std::string& key) {
    std::lock_guard<std::mutex> lk(mu_);
    const auto found = kvs_.find(key);
    if (found == kvs_.end()) return;
    leases_.erase(found->second.second);
    kvs_.erase(found);
  }

 private:
  struct Reply {
    int status;
    std::string body;
  };

  Reply Handle(const std::string& path, const std::string& body) {
    static const char kNoSpace[] =
        "{\"error\":\"etcdserver: mvcc: database space exceeded\",\"code\":8}";
    static const char kHeader[] = "\"header\":{\"revision\":\"1\"}";
    JsonValue req;
    if (!JsonParser::Parse(body, &req)) return {400, "{}"};
    std::lock_guard<std::mutex> lk(mu_);
    if (path == "/v3/lease/grant") {
      if (reject_writes_) return {429, kNoSpace};
      const int64_t id = next_lease_++;
      leases_.insert(id);
      return {200, std::string("{") + kHeader + ",\"ID\":\"" +
                       std::to_string(id) + "\",\"TTL\":\"30\"}"};
    }
    if (path == "/v3/lease/keepalive") {
      // An expired lease is answered with 200 and no TTL.
      const std::string id = req.get_str("ID");
      const bool alive = leases_.count(std::strtoll(id.c_str(), nullptr, 10));
      return {200, "{\"result\":{\"ID\":\"" + id + "\"" +
                       (alive ? ",\"TTL\":\"30\"" : "") + "}}"};
    }
    if (path == "/v3/kv/put") {
      if (reject_writes_) return {429, kNoSpace};
      const int64_t lease =
          std::strtoll(req.get_str("lease").c_str(), nullptr, 10);
      if (!leases_.count(lease))
        return {404, "{\"error\":\"etcdserver: requested lease not found\","
                     "\"code\":5}"};
      std::string key;
      if (!Base64Decode(req.get_str("key"), &key)) return {400, "{}"};
      kvs_[key] = {req.get_str("value"), lease};
      return {200, std::string("{") + kHeader + "}"};
    }
    if (path == "/v3/kv/range") {
      std::string begin, end;
      if (!Base64Decode(req.get_str("key"), &begin) ||
          !Base64Decode(req.get_str("range_end"), &end))
        return {400, "{}"};
      std::string kvs;
      for (auto it = kvs_.lower_bound(begin);
           it != kvs_.end() && it->first < end; ++it) {
        if (!kvs.empty()) kvs += ",";
        kvs += "{\"key\":\"" + Base64Encode(it->first) + "\",\"value\":\"" +
               it->second.first + "\"}";
      }
      return {200, std::string("{") + kHeader +
                       (kvs.empty() ? "" : ",\"kvs\":[" + kvs + "]") + "}"};
    }
    return {404, "{}"};
  }

  void Serve() {
    for (;;) {
      const int fd = ::accept(lfd_, nullptr, nullptr);
      if (fd < 0) return;
      std::string raw;
      char chunk[4096];
      size_t head_end = std::string::npos;
      size_t body_len = 0;
      for (;;) {
        if (head_end == std::string::npos) {
          const size_t at = raw.find("\r\n\r\n");
          if (at != std::string::npos) {
            head_end = at + 4;
            const size_t cl = raw.find("Content-Length: ");
            if (cl != std::string::npos && cl < at)
              body_len = std::strtoul(raw.c_str() + cl + 16, nullptr, 10);
          }
        }
        if (head_end != std::string::npos &&
            raw.size() >= head_end + body_len)
          break;
        const ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) break;
        raw.append(chunk, static_cast<size_t>(n));
      }
      if (head_end != std::string::npos && raw.size() >= head_end + body_len) {
        const size_t path_at = raw.find(' ') + 1;
        const std::string path =
            raw.substr(path_at, raw.find(' ', path_at) - path_at);
        const Reply reply = Handle(path, raw.substr(head_end, body_len));
        const std::string out =
            "HTTP/1.1 " + std::to_string(reply.status) +
            " X\r\nContent-Type: application/json\r\nContent-Length: " +
            std::to_string(reply.body.size()) + "\r\n\r\n" + reply.body;
        net::WriteAll(fd, out.data(), out.size());
      }
      ::close(fd);
    }
  }

  int lfd_ = -1, port_ = 0;
  std::thread th_;
  std::mutex mu_;
  bool reject_writes_ = false;
  int64_t next_lease_ = 1000;
  std::set<int64_t> leases_;
  // key -> (base64 value as received, lease)
  std::map<std::string, std::pair<std::string, int64_t>> kvs_;
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

// The same exchange as a v1.x peer frames it: 42-byte request prefix, 10-byte
// response prefix, status in the v1 byte order (0 = ok, 3 = I/O error).
bool DoLegacyReq(int port, WireOp op, const std::string& payload,
                 uint8_t* status, std::string* data) {
  const int fd = net::Dial("127.0.0.1:" + std::to_string(port), 2000, 5000);
  if (fd < 0) return false;
  char pre[kLegacyReqPrefix];
  EncodeLegacyReq(pre, op, payload.size());
  bool ok = net::WriteAll(fd, pre, kLegacyReqPrefix) &&
            (payload.empty() ||
             net::WriteAll(fd, payload.data(), payload.size()));
  char rp[kLegacyRespPrefix];
  ok = ok && net::ReadAll(fd, rp, kLegacyRespPrefix);
  if (ok) {
    *status = static_cast<uint8_t>(rp[1]);
    data->resize(net::GetU64(rp + 2));
    ok = data->empty() || net::ReadAll(fd, &(*data)[0], data->size());
  }
  ::close(fd);
  return ok;
}

// Status of a register/heartbeat; kInvalid if the MDS did not answer at all.
Status Beat(const MdsServer& mds, WireOp op, const std::string& group,
            const MemberInfo& m) {
  Status st = Status::kInvalid;
  std::string data;
  EXPECT_TRUE(DoReq(mds.port(), op, EncodeMemberReq(group, m), &st, &data));
  return st;
}

// Status of a list op; fills the ids it returned when it was served.
Status List(const MdsServer& mds, WireOp op, const std::string& group,
            std::set<std::string>* ids) {
  Status st = Status::kInvalid;
  std::string data;
  EXPECT_TRUE(DoReq(mds.port(), op, group, &st, &data));
  ids->clear();
  if (st == Status::kOk) {
    std::vector<MemberInfo> members;
    uint64_t epoch = 0;
    EXPECT_TRUE(DecodeMembers(data.data(), data.size(), &members, &epoch));
    for (const auto& m : members) ids->insert(m.id);
  }
  return st;
}

std::string MemberKey(const std::string& group, const std::string& id) {
  return "/dfkv/v1/groups/" + group + "/members/" + id;
}

using Ids = std::set<std::string>;

const MemberInfo kA{"na", "10.1.1.1", 28000, 1};
const MemberInfo kB{"nb", "10.1.1.2", 28000, 1};
const MemberInfo kC{"nc", "10.1.1.3", 28000, 1};

}  // namespace

// ---- UnwrittenMembers: the record behind the check (clock injected) ---------

TEST(UnwrittenMembers, RecordsPerPrefixUntilWritten) {
  uint64_t now_ms = 1000;
  UnwrittenMembers unwritten(120000, [&] { return now_ms; });
  unwritten.Add("/g/a/members/n2");
  unwritten.Add("/g/a/members/n1");
  unwritten.Add("/g/ab/members/n9");  // "/g/a" is a prefix of "/g/ab"
  EXPECT_EQ(unwritten.Under("/g/a/members/"),
            (std::vector<std::string>{"/g/a/members/n1", "/g/a/members/n2"}));
  EXPECT_EQ(unwritten.Under("/g/ab/members/"),
            (std::vector<std::string>{"/g/ab/members/n9"}));
  EXPECT_TRUE(unwritten.Under("/g/b/members/").empty());
  EXPECT_EQ(unwritten.Count(), 3u);

  unwritten.Remove("/g/a/members/n1");
  unwritten.Remove("/g/a/members/never-recorded");
  EXPECT_EQ(unwritten.Under("/g/a/members/"),
            (std::vector<std::string>{"/g/a/members/n2"}));
  EXPECT_EQ(unwritten.Count(), 2u);
}

TEST(UnwrittenMembers, LetsGoAfterTheHoldUnlessTheMemberComesBack) {
  uint64_t now_ms = 1000;
  UnwrittenMembers unwritten(120000, [&] { return now_ms; });
  unwritten.Add("/m/gone");
  unwritten.Add("/m/alive");
  now_ms += 90000;
  unwritten.Add("/m/alive");  // still heartbeating here, still not written
  now_ms += 30000;            // exactly the hold since "gone" was last heard
  EXPECT_EQ(unwritten.Count(), 2u);
  now_ms += 1;
  EXPECT_EQ(unwritten.Under("/m/"), (std::vector<std::string>{"/m/alive"}));
  EXPECT_EQ(unwritten.Count(), 1u);
  now_ms += 90000;
  EXPECT_TRUE(unwritten.Under("/m/").empty());
  EXPECT_EQ(unwritten.Count(), 0u);
}

TEST(UnwrittenMembers, ZeroHoldRecordsNothing) {
  UnwrittenMembers unwritten(0, [] { return uint64_t{1000}; });
  unwritten.Add("/m/a");
  EXPECT_TRUE(unwritten.Under("/m/").empty());
  EXPECT_EQ(unwritten.Count(), 0u);
}

TEST(UnwrittenMembers, IdChurnIsBoundedByTheHold) {
  // Ids that fail once and never return are dropped as later failures arrive,
  // without anyone listing or scraping: about two holds' worth is kept.
  uint64_t now_ms = 1000;
  UnwrittenMembers unwritten(120000, [&] { return now_ms; });
  size_t most_kept = 0;
  for (int i = 0; i < 10000; ++i) {
    unwritten.Add("/m/" + std::to_string(i));
    most_kept = std::max(most_kept, unwritten.Size());
    now_ms += 1000;
  }
  EXPECT_LE(most_kept, 250u);
  EXPECT_EQ(unwritten.Count(), 120u);  // one failure a second, hold of 120 s
}

// ---- MdsServer against the fake etcd -----------------------------------------

TEST(MdsEtcdWriteFailure, ListWithheldOnceAHeartbeatingMemberHasDroppedOut) {
  FakeEtcd etcd;
  MdsServer mds(etcd.ep());
  ASSERT_EQ(mds.Start(0), Status::kOk);
  const std::string g = "g";
  for (const auto& m : {kA, kB, kC})
    ASSERT_EQ(Beat(mds, WireOp::kRegister, g, m), Status::kOk);
  Ids ids;
  ASSERT_EQ(List(mds, WireOp::kListMembers, g, &ids), Status::kOk);
  EXPECT_EQ(ids, (Ids{"na", "nb", "nc"}));

  etcd.RejectWrites(true);
  // The lease is still alive, so the keepalive goes through; the Put and the
  // fallback lease grant do not. The node is told its heartbeat failed.
  EXPECT_EQ(Beat(mds, WireOp::kHeartbeat, g, kA), Status::kIOError);
  // Nothing has expired yet: etcd still lists every member, so the list is
  // complete and is served. A failed heartbeat alone withholds nothing.
  ASSERT_EQ(List(mds, WireOp::kListMembers, g, &ids), Status::kOk);
  EXPECT_EQ(ids, (Ids{"na", "nb", "nc"}));
  ASSERT_EQ(List(mds, WireOp::kListTopology, g, &ids), Status::kOk);
  EXPECT_EQ(ids, (Ids{"na", "nb", "nc"}));

  etcd.ExpireAllLeases();
  // etcd now lists nobody, yet na heartbeat a moment ago. An empty member list
  // would be a lie; the MDS answers with an error instead, which clients treat
  // as "keep the ring you have".
  EXPECT_EQ(List(mds, WireOp::kListMembers, g, &ids), Status::kIOError);
  EXPECT_EQ(List(mds, WireOp::kListTopology, g, &ids), Status::kIOError);

  const std::string metrics = mds.MetricsText();
  EXPECT_NE(metrics.find("dfkv_mds_list_withheld_total 2\n"),
            std::string::npos) << metrics;
  EXPECT_NE(metrics.find("dfkv_mds_unwritten_members 1\n"),
            std::string::npos) << metrics;
  mds.Stop();
}

TEST(MdsEtcdWriteFailure, ListServedAgainOnlyWhenEveryUnwrittenMemberIsBack) {
  FakeEtcd etcd;
  MdsServer mds(etcd.ep());
  ASSERT_EQ(mds.Start(0), Status::kOk);
  const std::string g = "g";
  for (const auto& m : {kA, kB, kC})
    ASSERT_EQ(Beat(mds, WireOp::kRegister, g, m), Status::kOk);

  etcd.RejectWrites(true);
  for (const auto& m : {kA, kB, kC})
    EXPECT_EQ(Beat(mds, WireOp::kHeartbeat, g, m), Status::kIOError);
  etcd.ExpireAllLeases();
  Ids ids;
  EXPECT_EQ(List(mds, WireOp::kListTopology, g, &ids), Status::kIOError);

  // etcd takes writes again. Members come back one heartbeat at a time; the
  // half-restored views in between are not member lists either.
  etcd.RejectWrites(false);
  EXPECT_EQ(Beat(mds, WireOp::kHeartbeat, g, kA), Status::kOk);
  EXPECT_EQ(List(mds, WireOp::kListTopology, g, &ids), Status::kIOError);
  EXPECT_EQ(Beat(mds, WireOp::kHeartbeat, g, kB), Status::kOk);
  EXPECT_EQ(List(mds, WireOp::kListMembers, g, &ids), Status::kIOError);
  EXPECT_EQ(Beat(mds, WireOp::kHeartbeat, g, kC), Status::kOk);
  ASSERT_EQ(List(mds, WireOp::kListTopology, g, &ids), Status::kOk);
  EXPECT_EQ(ids, (Ids{"na", "nb", "nc"}));
  ASSERT_EQ(List(mds, WireOp::kListMembers, g, &ids), Status::kOk);
  EXPECT_EQ(ids, (Ids{"na", "nb", "nc"}));

  const std::string metrics = mds.MetricsText();
  EXPECT_NE(metrics.find("dfkv_mds_unwritten_members 0\n"),
            std::string::npos) << metrics;
  mds.Stop();
}

TEST(MdsEtcdWriteFailure, MemberWrittenThroughAnotherMdsCountsAsBack) {
  // Registrars rotate over the MDS instances: the heartbeat that fails here is
  // followed by one that another instance writes. The first instance sees the
  // member in etcd again and has nothing to withhold.
  FakeEtcd etcd;
  MdsServer first(etcd.ep());
  MdsServer second(etcd.ep());
  ASSERT_EQ(first.Start(0), Status::kOk);
  ASSERT_EQ(second.Start(0), Status::kOk);
  const std::string g = "g";
  ASSERT_EQ(Beat(first, WireOp::kRegister, g, kA), Status::kOk);

  etcd.RejectWrites(true);
  EXPECT_EQ(Beat(first, WireOp::kHeartbeat, g, kA), Status::kIOError);
  etcd.ExpireAllLeases();
  Ids ids;
  EXPECT_EQ(List(first, WireOp::kListTopology, g, &ids), Status::kIOError);
  // The second instance has not heard from na: it has no reason to doubt etcd.
  ASSERT_EQ(List(second, WireOp::kListTopology, g, &ids), Status::kOk);
  EXPECT_TRUE(ids.empty());

  etcd.RejectWrites(false);
  ASSERT_EQ(Beat(second, WireOp::kHeartbeat, g, kA), Status::kOk);
  ASSERT_EQ(List(first, WireOp::kListTopology, g, &ids), Status::kOk);
  EXPECT_EQ(ids, (Ids{"na"}));
  first.Stop();
  second.Stop();
}

TEST(MdsEtcdWriteFailure, MembersThatReallyLeaveStillShrinkTheList) {
  FakeEtcd etcd;
  MdsServer mds(etcd.ep());
  ASSERT_EQ(mds.Start(0), Status::kOk);
  const std::string g = "g";
  for (const auto& m : {kA, kB, kC})
    ASSERT_EQ(Beat(mds, WireOp::kRegister, g, m), Status::kOk);

  // nb stops heartbeating and its lease runs out while etcd is healthy.
  etcd.ExpireKey(MemberKey(g, "nb"));
  Ids ids;
  ASSERT_EQ(List(mds, WireOp::kListMembers, g, &ids), Status::kOk);
  EXPECT_EQ(ids, (Ids{"na", "nc"}));

  // The whole group is torn down: an empty list is served as before.
  etcd.ExpireAllLeases();
  ASSERT_EQ(List(mds, WireOp::kListTopology, g, &ids), Status::kOk);
  EXPECT_TRUE(ids.empty());
  mds.Stop();
}

TEST(MdsEtcdWriteFailure, FailedRegistrationIsNoEvidenceOfAMember) {
  // The state with one or two MDS endpoints: etcd refuses writes, keepalives
  // still hold the members' leases, and the lists are complete.
  FakeEtcd etcd;
  MdsServer mds(etcd.ep());
  ASSERT_EQ(mds.Start(0), Status::kOk);
  const std::string g = "g";
  for (const auto& m : {kA, kB})
    ASSERT_EQ(Beat(mds, WireOp::kRegister, g, m), Status::kOk);
  etcd.RejectWrites(true);
  EXPECT_EQ(Beat(mds, WireOp::kHeartbeat, g, kA), Status::kIOError);

  // A node that is not listed keeps trying to get in. It never was a member;
  // the list without it is the member list and must keep being served.
  for (int i = 0; i < 3; ++i)
    EXPECT_EQ(Beat(mds, WireOp::kRegister, g, kC), Status::kIOError);
  Ids ids;
  ASSERT_EQ(List(mds, WireOp::kListMembers, g, &ids), Status::kOk);
  EXPECT_EQ(ids, (Ids{"na", "nb"}));
  ASSERT_EQ(List(mds, WireOp::kListTopology, g, &ids), Status::kOk);
  EXPECT_EQ(ids, (Ids{"na", "nb"}));

  // The price: a member whose process restarted registers again, which says
  // nothing about its past. If its key then runs out, the list goes on
  // without it.
  EXPECT_EQ(Beat(mds, WireOp::kRegister, g, kB), Status::kIOError);
  etcd.ExpireKey(MemberKey(g, "nb"));
  ASSERT_EQ(List(mds, WireOp::kListTopology, g, &ids), Status::kOk);
  EXPECT_EQ(ids, (Ids{"na"}));

  const std::string metrics = mds.MetricsText();
  EXPECT_NE(metrics.find("dfkv_mds_list_withheld_total 0\n"),
            std::string::npos) << metrics;
  EXPECT_NE(metrics.find("dfkv_mds_unwritten_members 1\n"),
            std::string::npos) << metrics;
  mds.Stop();
}

TEST(MdsEtcdWriteFailure, RegistrationThatGoesThroughClearsTheRecord) {
  // A recorded member comes back as a new process: its register is a write
  // like any other and ends the record.
  FakeEtcd etcd;
  MdsServer mds(etcd.ep());
  ASSERT_EQ(mds.Start(0), Status::kOk);
  const std::string g = "g";
  ASSERT_EQ(Beat(mds, WireOp::kRegister, g, kA), Status::kOk);
  etcd.RejectWrites(true);
  EXPECT_EQ(Beat(mds, WireOp::kHeartbeat, g, kA), Status::kIOError);
  etcd.ExpireAllLeases();
  Ids ids;
  EXPECT_EQ(List(mds, WireOp::kListMembers, g, &ids), Status::kIOError);

  etcd.RejectWrites(false);
  ASSERT_EQ(Beat(mds, WireOp::kRegister, g, kA), Status::kOk);
  etcd.ExpireKey(MemberKey(g, "na"));  // and now it leaves for real
  ASSERT_EQ(List(mds, WireOp::kListMembers, g, &ids), Status::kOk);
  EXPECT_TRUE(ids.empty());
  mds.Stop();
}

TEST(MdsEtcdWriteFailure, MemberThatStoppedHeartbeatingIsLetGoAfterTheHold) {
  // A member whose last heartbeat here failed and who then went away for real
  // must not block the list for good. Default hold, clock supplied.
  std::atomic<uint64_t> now_ms{1000};
  FakeEtcd etcd;
  MdsServer mds(etcd.ep(), 2000, [&] { return now_ms.load(); });
  ASSERT_EQ(mds.Start(0), Status::kOk);
  const std::string g = "g";
  for (const auto& m : {kA, kB})
    ASSERT_EQ(Beat(mds, WireOp::kRegister, g, m), Status::kOk);

  etcd.RejectWrites(true);
  EXPECT_EQ(Beat(mds, WireOp::kHeartbeat, g, kB), Status::kIOError);
  etcd.ExpireKey(MemberKey(g, "nb"));
  Ids ids;
  EXPECT_EQ(List(mds, WireOp::kListMembers, g, &ids), Status::kIOError);
  now_ms += MdsServer::kUnwrittenHoldMs;  // silent for exactly the hold
  EXPECT_EQ(List(mds, WireOp::kListMembers, g, &ids), Status::kIOError);
  now_ms += 1;
  ASSERT_EQ(List(mds, WireOp::kListMembers, g, &ids), Status::kOk);
  EXPECT_EQ(ids, (Ids{"na"}));
  mds.Stop();
}

TEST(MdsEtcdWriteFailure, HoldIsTakenFromTheEnvironmentAndClamped) {
  const std::string g = "g";
  const auto withheld_for = [&](const char* env) -> uint64_t {
    // How long after one failed heartbeat the list stays withheld, to the
    // millisecond: the largest age at which it still is.
    ::setenv("DFKV_MDS_UNWRITTEN_HOLD_MS", env, 1);
    std::atomic<uint64_t> now_ms{1000};
    FakeEtcd etcd;
    MdsServer mds(etcd.ep(), 2000, [&] { return now_ms.load(); });
    ::unsetenv("DFKV_MDS_UNWRITTEN_HOLD_MS");
    EXPECT_EQ(mds.Start(0), Status::kOk);
    EXPECT_EQ(Beat(mds, WireOp::kRegister, g, kA), Status::kOk);
    etcd.RejectWrites(true);
    EXPECT_EQ(Beat(mds, WireOp::kHeartbeat, g, kA), Status::kIOError);
    etcd.ExpireAllLeases();
    Ids ids;
    uint64_t lo = 0, hi = 3600000;  // withheld at age lo, served at age hi
    EXPECT_EQ(List(mds, WireOp::kListMembers, g, &ids), Status::kIOError);
    while (hi - lo > 1) {
      const uint64_t mid = lo + (hi - lo) / 2;
      now_ms = 1000 + mid;
      if (List(mds, WireOp::kListMembers, g, &ids) == Status::kIOError)
        lo = mid;
      else
        hi = mid;
    }
    mds.Stop();
    return lo;
  };
  EXPECT_EQ(withheld_for("45000"), 45000u);
  EXPECT_EQ(withheld_for("3000000"), 600000u);  // clamped to ten minutes
  // A negative value parses to a huge unsigned one and is clamped as well; it
  // must not hold a list for the life of the process.
  EXPECT_EQ(withheld_for("-1"), 600000u);
}

TEST(MdsEtcdWriteFailure, LegacyFramedPeersAreTreatedTheSame) {
  // v1.x nodes send the same register and heartbeat ops in the old framing;
  // a v1.x client gets the withheld list as its own I/O error byte.
  ::setenv("DFKV_MDS_ACCEPT_LEGACY", "1", 1);
  FakeEtcd etcd;
  MdsServer mds(etcd.ep());
  ::unsetenv("DFKV_MDS_ACCEPT_LEGACY");
  ASSERT_EQ(mds.Start(0), Status::kOk);
  const std::string g = "g";
  constexpr uint8_t kLegacyOk = 0, kLegacyIOError = 3;
  uint8_t status = 0xFF;
  std::string data;
  ASSERT_TRUE(DoLegacyReq(mds.port(), WireOp::kRegister,
                          EncodeMemberReq(g, kA), &status, &data));
  ASSERT_EQ(status, kLegacyOk);

  etcd.RejectWrites(true);
  ASSERT_TRUE(DoLegacyReq(mds.port(), WireOp::kRegister,
                          EncodeMemberReq(g, kB), &status, &data));
  EXPECT_EQ(status, kLegacyIOError);  // never listed: not recorded
  ASSERT_TRUE(DoLegacyReq(mds.port(), WireOp::kListMembers, g, &status, &data));
  EXPECT_EQ(status, kLegacyOk);
  ASSERT_TRUE(DoLegacyReq(mds.port(), WireOp::kHeartbeat,
                          EncodeMemberReq(g, kA), &status, &data));
  EXPECT_EQ(status, kLegacyIOError);  // was listed: recorded
  etcd.ExpireAllLeases();
  ASSERT_TRUE(DoLegacyReq(mds.port(), WireOp::kListMembers, g, &status, &data));
  EXPECT_EQ(status, kLegacyIOError);
  EXPECT_TRUE(data.empty());

  etcd.RejectWrites(false);
  ASSERT_TRUE(DoLegacyReq(mds.port(), WireOp::kHeartbeat,
                          EncodeMemberReq(g, kA), &status, &data));
  EXPECT_EQ(status, kLegacyOk);
  ASSERT_TRUE(DoLegacyReq(mds.port(), WireOp::kListMembers, g, &status, &data));
  ASSERT_EQ(status, kLegacyOk);
  std::vector<MemberInfo> members;
  uint64_t epoch = 0;
  ASSERT_TRUE(DecodeMembers(data.data(), data.size(), &members, &epoch));
  ASSERT_EQ(members.size(), 1u);
  EXPECT_EQ(members[0].id, "na");
  mds.Stop();
}

TEST(MdsEtcdWriteFailure, OtherGroupsAndClientListsAreNotWithheld) {
  FakeEtcd etcd;
  MdsServer mds(etcd.ep());
  ASSERT_EQ(mds.Start(0), Status::kOk);
  ASSERT_EQ(Beat(mds, WireOp::kRegister, "g1", kA), Status::kOk);
  ASSERT_EQ(Beat(mds, WireOp::kRegister, "g2", kB), Status::kOk);
  const MemberInfo consumer{"c1", "10.2.2.2", 0, 1};
  ASSERT_EQ(Beat(mds, WireOp::kClientRegister, "g2", consumer), Status::kOk);

  etcd.RejectWrites(true);
  EXPECT_EQ(Beat(mds, WireOp::kHeartbeat, "g1", kA), Status::kIOError);
  EXPECT_EQ(Beat(mds, WireOp::kClientHeartbeat, "g2", consumer),
            Status::kIOError);
  etcd.ExpireKey(MemberKey("g1", "na"));
  etcd.ExpireKey("/dfkv/v1/groups/g2/clients/c1");

  Ids ids;
  EXPECT_EQ(List(mds, WireOp::kListTopology, "g1", &ids), Status::kIOError);
  // g2 lost no member; a consumer that dropped out is not part of any ring.
  ASSERT_EQ(List(mds, WireOp::kListTopology, "g2", &ids), Status::kOk);
  EXPECT_EQ(ids, (Ids{"nb"}));
  ASSERT_EQ(List(mds, WireOp::kListClients, "g2", &ids), Status::kOk);
  EXPECT_TRUE(ids.empty());
  mds.Stop();
}

TEST(MdsEtcdWriteFailure, HoldZeroServesTheEtcdViewAsBefore) {
  ::setenv("DFKV_MDS_UNWRITTEN_HOLD_MS", "0", 1);
  FakeEtcd etcd;
  MdsServer mds(etcd.ep());
  ::unsetenv("DFKV_MDS_UNWRITTEN_HOLD_MS");
  ASSERT_EQ(mds.Start(0), Status::kOk);
  const std::string g = "g";
  ASSERT_EQ(Beat(mds, WireOp::kRegister, g, kA), Status::kOk);

  etcd.RejectWrites(true);
  EXPECT_EQ(Beat(mds, WireOp::kHeartbeat, g, kA), Status::kIOError);
  etcd.ExpireAllLeases();
  Ids ids;
  ASSERT_EQ(List(mds, WireOp::kListTopology, g, &ids), Status::kOk);
  EXPECT_TRUE(ids.empty());
  mds.Stop();
}

TEST(MdsEtcdWriteFailure, ClientKeepsItsRingThroughTheOutage) {
  // End to end with the client's poller: every member lease expires while etcd
  // refuses writes. The client must still be on its three-member ring after
  // more polls than its empty-view guard would sit out.
  FakeEtcd etcd;
  MdsServer mds(etcd.ep());
  ASSERT_EQ(mds.Start(0), Status::kOk);
  const std::string g = "g";
  for (const auto& m : {kA, kB, kC})
    ASSERT_EQ(Beat(mds, WireOp::kRegister, g, m), Status::kOk);

  std::vector<size_t> rings;       // size of every ring the client adopted
  std::vector<size_t> topologies;  // size of every topology it was handed
  MdsMemberPoller poller(
      {"127.0.0.1:" + std::to_string(mds.port())}, g,
      [&](const std::vector<MemberInfo>& topology, uint64_t, bool admitted) {
        topologies.push_back(topology.size());
        if (admitted) rings.push_back(topology.size());
      });
  ASSERT_TRUE(poller.PollOnce());
  ASSERT_EQ(rings, (std::vector<size_t>{3}));

  etcd.RejectWrites(true);
  for (const auto& m : {kA, kB, kC})
    EXPECT_EQ(Beat(mds, WireOp::kHeartbeat, g, m), Status::kIOError);
  etcd.ExpireAllLeases();
  for (int i = 0; i < 6; ++i) EXPECT_FALSE(poller.PollOnce()) << "poll " << i;
  EXPECT_EQ(rings, (std::vector<size_t>{3}));
  // The topology stream is not held back by the client's guard: one shrunken
  // answer would already have retired the missing peers' connections.
  EXPECT_EQ(topologies, (std::vector<size_t>{3}));

  etcd.RejectWrites(false);
  for (const auto& m : {kA, kB, kC})
    EXPECT_EQ(Beat(mds, WireOp::kHeartbeat, g, m), Status::kOk);
  EXPECT_TRUE(poller.PollOnce());
  EXPECT_EQ(rings, (std::vector<size_t>{3}));  // same members: nothing to rebuild
  mds.Stop();
}
