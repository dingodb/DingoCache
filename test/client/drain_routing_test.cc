// Drain routing (MDS-04): a draining member keeps its ring position and
// serves reads; clients send it no new writes, read it after a miss at a
// key's new owner, and send REMOVE and RETAIN to both. Run over a scripted
// transport of several nodes, both the TCP form (scalar bodies) and the
// pipelined RDMA form (batch bodies), and once against real nodes over TCP
// for the node's kDrain op.
#include "client/kv_client.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cache/kv_node_server.h"
#include "client/key_map.h"
#include "common/membership.h"
#include "scripted_topology_mds.h"
#include "transport/transport.h"
#include "transport/wire.h"
#include "utils/con_hash.h"
#include "utils/net_util.h"
#include "utils/wire_limits.h"

using namespace dfkv;  // NOLINT

namespace {

// Several nodes, each a map of values; every request names its node.
class NodesTransport final : public Transport {
 public:
  using Key = std::pair<uint64_t, uint64_t>;
  explicit NodesTransport(bool pipelined) : pipelined_(pipelined) {}

  std::mutex mu;
  std::map<std::string, std::map<Key, std::string>> store;
  std::map<std::string, std::set<Key>> retained;
  std::map<std::string, int> reads;  // reads and exists, by node
  std::string failing_removes;        // a node whose REMOVEs fail

  static Key K(const BlockKey& key) { return {key.digest_hi, key.digest_lo}; }

  bool pipelined() const override { return pipelined_; }
  Status Cache(const std::string& node, const BlockKey& key, const void* data,
               size_t len) override {
    std::lock_guard<std::mutex> lk(mu);
    store[node][K(key)].assign(static_cast<const char*>(data), len);
    return Status::kOk;
  }
  Status Range(const std::string& node, const BlockKey& key, uint64_t offset,
               uint64_t length, std::string* out,
               uint64_t* value_len) override {
    std::lock_guard<std::mutex> lk(mu);
    ++reads[node];
    const auto& values = store[node];
    const auto it = values.find(K(key));
    if (it == values.end()) return Status::kNotFound;
    if (value_len) *value_len = it->second.size();
    if (offset > it->second.size()) return Status::kInvalid;
    *out = it->second.substr(offset, length);
    return Status::kOk;
  }
  Status Lookup(const std::string& node, const BlockKey& key,
                uint64_t* value_len) override {
    std::lock_guard<std::mutex> lk(mu);
    const auto& values = store[node];
    const auto it = values.find(K(key));
    if (it == values.end()) return Status::kNotFound;
    *value_len = it->second.size();
    return Status::kOk;
  }
  Status Exist(const std::string& node, const BlockKey& key,
               bool* exist) override {
    std::lock_guard<std::mutex> lk(mu);
    ++reads[node];
    *exist = store[node].count(K(key)) != 0;
    return Status::kOk;
  }
  Status Remove(const std::string& node, const BlockKey& key) override {
    std::lock_guard<std::mutex> lk(mu);
    if (node == failing_removes) return Status::kIOError;
    return store[node].erase(K(key)) ? Status::kOk : Status::kNotFound;
  }
  Status Retain(const std::string& node, uint64_t, uint64_t count,
                const std::string& payload, std::string* out) override {
    std::lock_guard<std::mutex> lk(mu);
    out->assign(count, static_cast<char>(KVClient::kRetMiss));
    for (uint64_t i = 0; i < count; ++i) {
      const char* k = payload.data() + kRetainHeaderBytes + i * kRetainKeyBytes;
      const Key key{net::GetU64(k), net::GetU64(k + 8)};
      retained[node].insert(key);
      if (store[node].count(key))
        (*out)[i] = static_cast<char>(KVClient::kRetProtectedNew);
    }
    out->resize(count + kRetainTrailerBytes);
    RetainTrailer t;
    t.max_ttl_s = 3600;
    t.headroom_pct = 100;
    EncodeRetainTrailer(&(*out)[count], t);
    return Status::kOk;
  }

 private:
  const bool pipelined_;
};

const std::string kNs = "drain/test";
const std::vector<std::string> kIds = {"a", "b", "c"};
std::string Addr(const std::string& id) {
  return "127.0.0.1:" + std::to_string(1 + (id[0] - 'a'));
}
std::vector<MemberInfo> View(const std::set<std::string>& draining) {
  std::vector<MemberInfo> ms;
  for (size_t i = 0; i < kIds.size(); ++i) {
    MemberInfo m{kIds[i], "127.0.0.1", static_cast<uint32_t>(1 + i), 1};
    m.draining = draining.count(kIds[i]) != 0;
    ms.push_back(m);
  }
  return ms;
}
std::vector<std::pair<std::string, std::string>> Pairs() {
  std::vector<std::pair<std::string, std::string>> p;
  for (const auto& id : kIds) p.push_back({id, Addr(id)});
  return p;
}
// The owner of a key on the ring of `ids`, as KVClient builds it.
std::string Owner(const std::string& key, const std::vector<std::string>& ids) {
  ConHash ring;
  for (const auto& id : ids) ring.AddNode(id, 1);
  ring.Build();
  std::string name;
  ring.LookupHash(ConHash::KeyHash(key), &name);
  return name;
}
// Keys whose full-ring owner is `id`.
std::vector<std::string> KeysOwnedBy(const std::string& id, size_t n,
                                     const std::string& prefix) {
  std::vector<std::string> keys;
  for (int i = 0; keys.size() < n; ++i) {
    const std::string key = prefix + std::to_string(i);
    if (Owner(key, kIds) == id) keys.push_back(key);
  }
  return keys;
}
NodesTransport::Key K(const std::string& key) {
  return NodesTransport::K(ToBlockKey(kNs, key));
}
std::string Metric(KVClient& c, const std::string& op,
                   const std::string& result) {
  const std::string text = c.MetricsSnapshot();
  const std::string line = "dfkv_client_drain_requests_total{op=\"" + op +
                           "\",result=\"" + result + "\"} ";
  const size_t at = text.find(line);
  if (at == std::string::npos) return "absent";
  return text.substr(at + line.size(), text.find('\n', at) - at - line.size());
}

class DrainRouting : public ::testing::TestWithParam<bool> {
 protected:
  DrainRouting()
      : t(GetParam()),
        old_client(Pairs(), kNs, &t),
        client(Pairs(), kNs, &t) {
    client.SetMembers(View({"b"}));  // b drains; old_client does not know
  }
  NodesTransport t;
  KVClient old_client;  // a client that predates drain: one ring, a b c
  KVClient client;
};

INSTANTIATE_TEST_SUITE_P(TcpAndPipelined, DrainRouting, ::testing::Bool());

TEST_P(DrainRouting, WritesAvoidTheDrainerAndReadsFindWhatItHolds) {
  const std::string v(32, 'v');
  const auto on_b = KeysOwnedBy("b", 8, "held-");
  for (const auto& key : on_b)
    ASSERT_TRUE(old_client.Put(key, v.data(), v.size()));
  ASSERT_EQ(t.store[Addr("b")].size(), on_b.size());

  // Every read form finds the drainer's keys after a miss at the new owner.
  std::string out(v.size(), '\0');
  EXPECT_TRUE(client.Get(on_b[0], &out[0], out.size()));
  EXPECT_EQ(out, v);
  std::string auto_out;
  EXPECT_TRUE(client.GetAuto(on_b[1], &auto_out));
  EXPECT_EQ(auto_out, v);
  EXPECT_TRUE(client.Exist(on_b[2]));
  std::vector<std::string> outs(on_b.size(), std::string(v.size(), '\0'));
  std::vector<KvGetItem> items;
  for (size_t i = 0; i < on_b.size(); ++i)
    items.push_back({on_b[i], &outs[i][0], outs[i].size()});
  std::vector<ItemStatus> status;
  const auto got = client.BatchGet(items, &status);
  for (size_t i = 0; i < on_b.size(); ++i) {
    EXPECT_TRUE(got[i]) << on_b[i];
    EXPECT_EQ(status[i], ItemStatus::kOk);
    EXPECT_EQ(outs[i], v);
  }
  std::vector<size_t> lens;
  const auto got_auto = client.BatchGetAuto(items, &lens);
  for (size_t i = 0; i < on_b.size(); ++i) {
    EXPECT_TRUE(got_auto[i]);
    EXPECT_EQ(lens[i], v.size());
  }
  const auto exist = client.BatchExist(on_b);
  for (size_t i = 0; i < on_b.size(); ++i) EXPECT_TRUE(exist[i]);
  std::vector<KvGetItemSg> sg;
  for (size_t i = 0; i < on_b.size(); ++i)
    sg.push_back({on_b[i], {&outs[i][0]}, {outs[i].size()}});
  std::vector<size_t> sg_lens;
  const auto got_sg = client.BatchGetAutoSg(sg, &sg_lens);
  for (size_t i = 0; i < on_b.size(); ++i) {
    EXPECT_TRUE(got_sg[i]);
    EXPECT_EQ(sg_lens[i], v.size());
  }
  EXPECT_NE(Metric(client, "read", "served"), "0");
  EXPECT_NE(Metric(client, "read", "served"), "absent");

  // New writes of b's keys go to their owner on the ring without b.
  const auto fresh = KeysOwnedBy("b", 8, "fresh-");
  for (const auto& key : fresh) {
    ASSERT_TRUE(client.Put(key, v.data(), v.size()));
    const std::string owner = Owner(key, {"a", "c"});
    EXPECT_EQ(t.store[Addr(owner)].count(K(key)), 1u) << key;
  }
  std::vector<KvPutItem> puts;
  const auto batch_fresh = KeysOwnedBy("b", 8, "batch-fresh-");
  for (const auto& key : batch_fresh) puts.push_back({key, v.data(), v.size()});
  for (bool ok : client.BatchPut(puts)) EXPECT_TRUE(ok);
  EXPECT_EQ(t.store[Addr("b")].size(), on_b.size()) << "b took a new write";
}

TEST_P(DrainRouting, AMissOfAKeyTheDrainerDoesNotOwnDoesNotAskIt) {
  const auto on_a = KeysOwnedBy("a", 8, "elsewhere-");
  std::string out(8, '\0');
  for (const auto& key : on_a) EXPECT_FALSE(client.Get(key, &out[0], out.size()));
  EXPECT_FALSE(client.BatchExist(on_a)[0]);
  EXPECT_EQ(t.reads[Addr("b")], 0);
  // A miss of b's own key asks b once.
  const auto on_b = KeysOwnedBy("b", 1, "absent-");
  EXPECT_FALSE(client.Get(on_b[0], &out[0], out.size()));
  EXPECT_EQ(t.reads[Addr("b")], 1);
  EXPECT_EQ(Metric(client, "read", "not_served"), "1");
}

// A key written before the drain (on b) and again during it (on its new
// owner): a REMOVE must take both, or a read would find b's copy.
TEST_P(DrainRouting, RemoveTakesTheKeyFromBothOwners) {
  const std::string old_value(16, 'o'), new_value(16, 'n');
  const auto keys = KeysOwnedBy("b", 6, "removed-");
  for (const auto& key : keys) {
    ASSERT_TRUE(old_client.Put(key, old_value.data(), old_value.size()));
    ASSERT_TRUE(client.Put(key, new_value.data(), new_value.size()));
  }
  EXPECT_TRUE(client.Remove(keys[0]));
  std::vector<std::string> rest(keys.begin() + 1, keys.end());
  for (bool ok : client.BatchRemove(rest)) EXPECT_TRUE(ok);
  std::string out(16, '\0');
  for (const auto& key : keys) {
    EXPECT_FALSE(client.Get(key, &out[0], out.size())) << key;
    EXPECT_EQ(t.store[Addr("b")].count(K(key)), 0u);
    EXPECT_EQ(t.store[Addr(Owner(key, {"a", "c"}))].count(K(key)), 0u);
  }
  EXPECT_EQ(Metric(client, "remove", "served"), std::to_string(keys.size()));
}

TEST_P(DrainRouting, ARemoveTheDrainerDidNotConfirmFails) {
  const std::string v(16, 'v');
  const auto keys = KeysOwnedBy("b", 2, "stuck-");
  for (const auto& key : keys) ASSERT_TRUE(old_client.Put(key, v.data(), v.size()));
  t.failing_removes = Addr("b");
  EXPECT_FALSE(client.Remove(keys[0]));
  EXPECT_FALSE(client.BatchRemove({keys[1]})[0]);
  EXPECT_EQ(Metric(client, "remove", "not_served"), "2");
}

// RETAIN goes to both owners, and a key only the drainer holds is answered
// with the drainer's code.
TEST_P(DrainRouting, RetainProtectsTheDrainersCopyToo) {
  const std::string v(16, 'v');
  const auto keys = KeysOwnedBy("b", 4, "kept-");
  ASSERT_TRUE(old_client.Put(keys[0], v.data(), v.size()));  // only on b
  ASSERT_TRUE(client.Put(keys[1], v.data(), v.size()));      // only on the new owner
  std::vector<uint8_t> out(keys.size(), 0);
  KVClient::RetainRequest req;
  req.ttl_s = 60;
  req.mode = kRetainModeSet;
  client.BatchRetain(keys, req, out.data());
  EXPECT_EQ(out[0], KVClient::kRetProtectedNew) << "only b holds it";
  EXPECT_EQ(out[1], KVClient::kRetProtectedNew) << "the new owner holds it";
  EXPECT_EQ(out[2], KVClient::kRetMiss);
  for (const auto& key : keys) {
    EXPECT_EQ(t.retained[Addr("b")].count(K(key)), 1u) << key;
    EXPECT_EQ(t.retained[Addr(Owner(key, {"a", "c"}))].count(K(key)), 1u) << key;
  }
  EXPECT_EQ(Metric(client, "retain", "served"), "1");
  EXPECT_EQ(Metric(client, "retain", "not_served"), "3");
}

// Mixed versions, a known limit: a client that predates drain still routes
// b's keys to b. Its REMOVE of a key a new client wrote during the drain
// reaches only b, so the new owner's copy stays readable until it is evicted
// or removed by a new client. docs/DEPLOY.md says to upgrade every client
// that REMOVEs before draining.
TEST_P(DrainRouting, AnOldClientsRemoveDoesNotReachTheNewOwner) {
  const std::string v(16, 'v');
  const std::string key = KeysOwnedBy("b", 1, "mixed-")[0];
  ASSERT_TRUE(client.Put(key, v.data(), v.size()));
  EXPECT_TRUE(old_client.Remove(key));  // kNotFound on b counts as removed
  std::string out(16, '\0');
  EXPECT_TRUE(client.Get(key, &out[0], out.size()))
      << "the documented limit no longer holds: update docs/DEPLOY.md";
  // A new client's REMOVE does reach both.
  EXPECT_TRUE(client.Remove(key));
  EXPECT_FALSE(client.Get(key, &out[0], out.size()));
  // The other way round needs no help: an old client's PUT lands on b,
  // where a new client's read finds it after the miss.
  ASSERT_TRUE(old_client.Put(key, v.data(), v.size()));
  EXPECT_TRUE(client.Get(key, &out[0], out.size()));
}

TEST_P(DrainRouting, WithEveryMemberDrainingWritesGoWhereTheyAlwaysWent) {
  client.SetMembers(View({"a", "b", "c"}));
  const std::string v(16, 'v');
  for (const auto& key : KeysOwnedBy("b", 4, "all-")) {
    ASSERT_TRUE(client.Put(key, v.data(), v.size()));
    EXPECT_EQ(t.store[Addr("b")].count(K(key)), 1u);
  }
  EXPECT_NE(client.MetricsSnapshot().find("dfkv_client_ring_draining_members 0"),
            std::string::npos);
}

TEST_P(DrainRouting, AnUndrainedMemberTakesWritesAgain) {
  client.SetMembers(View({}));
  const std::string v(16, 'v');
  for (const auto& key : KeysOwnedBy("b", 4, "back-")) {
    ASSERT_TRUE(client.Put(key, v.data(), v.size()));
    EXPECT_EQ(t.store[Addr("b")].count(K(key)), 1u);
  }
  EXPECT_EQ(Metric(client, "read", "served"), "0");
}

// A drain-only change reaches a client through MDS discovery: the view epoch
// changes while placement does not.
TEST(DrainDiscovery, ADrainChangeIsAdoptedFromTheMds) {
  ScriptedTopologyMds mds(View({}));
  NodesTransport t(true);
  KVClient client({}, kNs, &t);
  client.StartMdsDiscovery({mds.endpoint()}, "drain", 10);
  ASSERT_TRUE(client.WaitForRing(5000));
  auto draining = [&] {
    return client.MetricsSnapshot().find("dfkv_client_ring_draining_members 1") !=
           std::string::npos;
  };
  EXPECT_FALSE(draining());
  mds.SetMembers(View({"b"}));
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!draining() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  ASSERT_TRUE(draining());
  const std::string v(16, 'v');
  for (const auto& key : KeysOwnedBy("b", 4, "polled-")) {
    ASSERT_TRUE(client.Put(key, v.data(), v.size()));
    EXPECT_EQ(t.store[Addr("b")].count(K(key)), 0u);
  }
  client.StopMdsDiscovery();
}

// The node's side: kDrain is answered by the handler dfkv_server installs
// with DFKV_DRAIN on, and kInvalid without one (as by a node that predates
// the op).
Status SendDrain(const std::string& addr, const std::string& payload) {
  const int fd = net::Dial(addr, 2000, 2000);
  if (fd < 0) return Status::kIOError;
  char pre[kReqPrefix];
  EncodeReq(pre, WireOp::kDrain, BlockKey{}, 0, 0, payload.size());
  Status st = Status::kIOError;
  if (net::WriteAll(fd, pre, kReqPrefix) &&
      (payload.empty() || net::WriteAll(fd, payload.data(), payload.size()))) {
    char rp[kRespPrefix];
    uint64_t dlen = 0;
    if (!net::ReadAll(fd, rp, kRespPrefix) ||
        !DecodeResp(rp, &st, &dlen, wire_limits::kMdsMaxRespData))
      st = Status::kIOError;
  }
  ::close(fd);
  return st;
}

TEST(DrainNode, KDrainReachesTheHandlerAndIsRefusedWithoutOne) {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "dfkv_drain_node";
  fs::remove_all(dir);
  fs::create_directories(dir);
  KvNodeServer srv(dir.string(), 1ull << 26);
  ASSERT_EQ(srv.Start(0), Status::kOk);
  const std::string addr = "127.0.0.1:" + std::to_string(srv.port());
  EXPECT_EQ(SendDrain(addr, std::string(1, '\1')), Status::kInvalid);
  std::vector<bool> calls;
  srv.set_drain_handler([&](bool on) {
    calls.push_back(on);
    return on ? Status::kOk : Status::kIOError;
  });
  EXPECT_EQ(SendDrain(addr, std::string(1, '\1')), Status::kOk);
  EXPECT_EQ(SendDrain(addr, std::string(1, '\0')), Status::kIOError);
  EXPECT_EQ(SendDrain(addr, ""), Status::kInvalid);
  EXPECT_EQ(SendDrain(addr, "xy"), Status::kInvalid);
  EXPECT_EQ(calls, (std::vector<bool>{true, false}));
  srv.set_drain_handler(nullptr);
  EXPECT_EQ(SendDrain(addr, std::string(1, '\1')), Status::kInvalid);
  srv.Stop();
  fs::remove_all(dir);
}

}  // namespace
