// TDD R12 — dynamic membership: KVClient.SetMembers() rebuilds the ring at
// runtime so adding a node re-routes new keys without recreating the client.
#include "client/kv_client.h"
#include "cache/kv_node_server.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {
std::string Hdr() { return "test/model"; }
struct Node { fs::path dir; std::unique_ptr<KvNodeServer> srv; std::string addr; };
std::unique_ptr<Node> Start(const std::string& tag) {
  auto n = std::make_unique<Node>();
  n->dir = fs::temp_directory_path() / ("dfkv_dyn_" + tag);
  fs::remove_all(n->dir); fs::create_directories(n->dir);
  n->srv = std::make_unique<KvNodeServer>(n->dir.string(), 1ull << 30);
  EXPECT_EQ(n->srv->Start(0), Status::kOk);
  n->addr = "127.0.0.1:" + std::to_string(n->srv->port());
  return n;
}
}  // namespace

TEST(DynamicMembers, AddingNodeReroutesNewKeys) {
  auto a = Start("a"); auto b = Start("b");
  KVClient c({{"a", a->addr}}, Hdr());  // start with only node a
  std::string v(64, 'v');
  for (int i = 0; i < 60; ++i)
    ASSERT_TRUE(c.Put("p1_" + std::to_string(i), v.data(), v.size()));
  EXPECT_EQ(b->srv->Count(), 0u);  // b not in the ring yet

  c.SetMembers(std::vector<std::pair<std::string,std::string>>{{"a", a->addr}, {"b", b->addr}});  // hot add node b
  for (int i = 0; i < 60; ++i)
    ASSERT_TRUE(c.Put("p2_" + std::to_string(i), v.data(), v.size()));
  EXPECT_GT(b->srv->Count(), 0u);  // some new keys now land on b

  // keys still readable (those routed to their current owner)
  std::string out(v.size(), '\0');
  EXPECT_TRUE(c.Get("p2_0", &out[0], out.size()));
  a->srv->Stop(); b->srv->Stop();
}

// AdoptRing()/MembershipDelta: a membership change must log WHICH node ids
// joined/left, so a scale-up or a node loss is obvious at a glance. The logger
// writes to stderr; capture it around each SetMembers. No servers/ops are
// needed — only the ring rebuild runs (the probe + MDS discovery threads stay
// off unless their env knobs are set), so nothing else pollutes the capture.
TEST(DynamicMembers, SetMembersLogsAddRemoveDelta) {
  using P = std::vector<std::pair<std::string, std::string>>;
  KVClient c(P{{"n1", "127.0.0.1:1"}, {"n2", "127.0.0.1:2"}}, Hdr());

  // add n3
  testing::internal::CaptureStderr();
  c.SetMembers(P{{"n1", "127.0.0.1:1"}, {"n2", "127.0.0.1:2"}, {"n3", "127.0.0.1:3"}});
  std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("3 member(s) +1 -0"), std::string::npos) << log;
  EXPECT_NE(log.find("added: n3"), std::string::npos) << log;
  EXPECT_EQ(log.find("removed:"), std::string::npos) << log;

  // remove n1
  testing::internal::CaptureStderr();
  c.SetMembers(P{{"n2", "127.0.0.1:2"}, {"n3", "127.0.0.1:3"}});
  log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("2 member(s) +0 -1"), std::string::npos) << log;
  EXPECT_NE(log.find("removed: n1"), std::string::npos) << log;

  // rolling replace: n2 out, n4 in
  testing::internal::CaptureStderr();
  c.SetMembers(P{{"n3", "127.0.0.1:3"}, {"n4", "127.0.0.1:4"}});
  log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("+1 -1"), std::string::npos) << log;
  EXPECT_NE(log.find("added: n4"), std::string::npos) << log;
  EXPECT_NE(log.find("removed: n2"), std::string::npos) << log;

  // same ids, only the address changed -> no id churn -> "(unchanged)"
  testing::internal::CaptureStderr();
  c.SetMembers(P{{"n3", "127.0.0.1:33"}, {"n4", "127.0.0.1:44"}});
  log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("(unchanged)"), std::string::npos) << log;

  // empty membership -> WARN (the "ring is empty, ops report ok=0" case)
  testing::internal::CaptureStderr();
  c.SetMembers(P{});
  log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("EMPTY membership"), std::string::npos) << log;
}

// Ring adoptions must feed the transport's topology hint: connection budgets
// scale with nodes x rails, so a growing ring has to reach OnTopologyHint
// both at construction (static member lists never re-adopt) and on every
// later SetMembers (0812-004: a fixed budget starved a 55-node ring).
namespace {
struct HintRecordingTransport : dfkv::Transport {
  std::vector<size_t> hints;
  void OnTopologyHint(size_t nodes) override { hints.push_back(nodes); }
  Status Cache(const std::string&, const dfkv::BlockKey&, const void*,
               size_t) override {
    return Status::kOk;
  }
  Status Range(const std::string&, const dfkv::BlockKey&, uint64_t, uint64_t,
               std::string*, uint64_t*) override {
    return Status::kNotFound;
  }
  Status Lookup(const std::string&, const dfkv::BlockKey&,
                uint64_t* value_len) override {
    if (value_len) *value_len = 0;
    return Status::kNotFound;
  }
  Status Exist(const std::string&, const dfkv::BlockKey&,
               bool* exist) override {
    if (exist) *exist = false;
    return Status::kOk;
  }
};
}  // namespace

TEST(DynamicMembers, AdoptionsFeedTransportTopologyHint) {
  using P = std::vector<std::pair<std::string, std::string>>;
  HintRecordingTransport t;
  KVClient c(P{{"n1", "127.0.0.1:1"}, {"n2", "127.0.0.1:2"}}, Hdr(), &t);
  // The constructor replays the initial adoption once the transport exists.
  ASSERT_EQ(t.hints.size(), 1u);
  EXPECT_EQ(t.hints[0], 2u);

  c.SetMembers(P{{"n1", "127.0.0.1:1"},
                 {"n2", "127.0.0.1:2"},
                 {"n3", "127.0.0.1:3"}});
  ASSERT_EQ(t.hints.size(), 2u);
  EXPECT_EQ(t.hints[1], 3u);

  // An empty adoption routes nowhere and must not hint a zero-sized ring.
  c.SetMembers(P{});
  EXPECT_EQ(t.hints.size(), 2u);
}
