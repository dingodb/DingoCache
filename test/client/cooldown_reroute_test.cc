// Cooldown reroute (CLI-02 stage C, DFKV_RING_COOLDOWN_REROUTE; §3.2 of
// docs/design/ring-unavailable-owner.md): a key whose owner cools in this
// client goes to a substitute while the registry gate is open; a REMOVE fans
// out to the key's successors on every recent ring and is ok only if every
// target confirms; the keys written to a substitute are removed from it when
// the episode ends. Run over a scripted transport of several nodes, in the
// TCP form (scalar bodies) and the pipelined RDMA form (batch bodies).
#include "client/kv_client.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "client/key_map.h"
#include "common/membership.h"
#include "scripted_topology_mds.h"
#include "transport/transport.h"
#include "transport/wire.h"
#include "utils/con_hash.h"
#include "utils/net_util.h"

using namespace dfkv;  // NOLINT

namespace {

class NodesTransport final : public Transport {
 public:
  using Key = std::pair<uint64_t, uint64_t>;
  explicit NodesTransport(bool pipelined) : pipelined_(pipelined) {}

  std::mutex mu;
  std::map<std::string, std::map<Key, std::string>> store;
  std::map<std::string, int> reads, removes, writes;
  std::set<std::string> down;             // every op fails
  std::set<std::string> failing_removes;  // REMOVEs fail, the rest serves

  static Key K(const BlockKey& key) { return {key.digest_hi, key.digest_lo}; }

  bool pipelined() const override { return pipelined_; }
  Status Cache(const std::string& node, const BlockKey& key, const void* data,
               size_t len) override {
    std::lock_guard<std::mutex> lk(mu);
    ++writes[node];
    if (down.count(node)) return Status::kIOError;
    // Write-once per node, as the servers.
    store[node].emplace(K(key), std::string(static_cast<const char*>(data), len));
    return Status::kOk;
  }
  Status Range(const std::string& node, const BlockKey& key, uint64_t offset,
               uint64_t length, std::string* out,
               uint64_t* value_len) override {
    std::lock_guard<std::mutex> lk(mu);
    ++reads[node];
    if (down.count(node)) return Status::kIOError;
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
    if (down.count(node)) return Status::kIOError;
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
    if (down.count(node)) return Status::kIOError;
    *exist = store[node].count(K(key)) != 0;
    return Status::kOk;
  }
  Status Remove(const std::string& node, const BlockKey& key) override {
    std::lock_guard<std::mutex> lk(mu);
    ++removes[node];
    if (down.count(node) || failing_removes.count(node)) return Status::kIOError;
    return store[node].erase(K(key)) ? Status::kOk : Status::kNotFound;
  }
  Status Retain(const std::string&, uint64_t, uint64_t count,
                const std::string&, std::string* out) override {
    out->assign(count, static_cast<char>(KVClient::kRetMiss));
    return Status::kOk;
  }
  bool Holds(const std::string& node, const Key& key) {
    std::lock_guard<std::mutex> lk(mu);
    return store[node].count(key) != 0;
  }

 private:
  const bool pipelined_;
};

class ScopedEnv {
 public:
  ScopedEnv(const char* name, const std::string& value) : name_(name) {
    ::setenv(name, value.c_str(), 1);
  }
  ~ScopedEnv() { ::unsetenv(name_); }

 private:
  const char* name_;
};

const std::string kNs = "reroute/test";
const std::vector<std::string> kIds = {"a", "b", "c", "d", "e", "f"};
std::string Addr(const std::string& id) {
  return "127.0.0.1:" + std::to_string(1 + (id[0] - 'a'));
}
std::vector<MemberInfo> View(const std::vector<std::string>& ids,
                             const std::set<std::string>& draining = {}) {
  std::vector<MemberInfo> ms;
  for (const auto& id : ids) {
    MemberInfo m{id, "127.0.0.1", static_cast<uint32_t>(1 + (id[0] - 'a')), 1};
    m.draining = draining.count(id) != 0;
    ms.push_back(m);
  }
  return ms;
}
// The members of `ids` clockwise from the key, the owner first.
std::vector<std::string> Successors(const std::string& key,
                                    const std::vector<std::string>& ids) {
  ConHash ring;
  for (const auto& id : ids) ring.AddNode(id, 1);
  ring.Build();
  std::vector<std::string> out;
  ring.Walk(ConHash::KeyHash(key), [&](const std::string& id) {
    out.push_back(id);
    return true;
  });
  return out;
}
std::vector<std::string> KeysOwnedBy(const std::string& id, size_t n,
                                     const std::string& prefix,
                                     const std::vector<std::string>& ids = kIds) {
  std::vector<std::string> keys;
  for (int i = 0; keys.size() < n; ++i) {
    const std::string key = prefix + std::to_string(i);
    if (Successors(key, ids)[0] == id) keys.push_back(key);
  }
  return keys;
}
NodesTransport::Key K(const std::string& key) {
  return NodesTransport::K(ToBlockKey(kNs, key));
}
std::string Metric(KVClient& c, const std::string& series) {
  const std::string text = c.MetricsSnapshot();
  const std::string line = "\n" + series + " ";
  const size_t at = text.find(line);
  if (at == std::string::npos) return "absent";
  const size_t from = at + line.size();
  return text.substr(from, text.find('\n', from) - from);
}
MemberInfo MarkedClient(const std::string& id, int poll_ms) {
  MemberInfo m{id, "0.0.0.0", 0, 0};
  m.info = "connector=x,dfkv-remove-fanout=1;poll_ms=" + std::to_string(poll_ms);
  return m;
}

// Switches every client of a test shares: C on, short busy grace (a first
// failure starts the cooldown), long cooldown unless a test shortens it.
struct Env {
  ScopedEnv reroute{"DFKV_RING_COOLDOWN_REROUTE", "1"};
  ScopedEnv busy{"DFKV_PEER_BAD_GRACE_MS", "1"};
  std::unique_ptr<ScopedEnv> cooldown, cooldown_max;
  explicit Env(const std::string& cooldown_ms = "60000")
      : cooldown(std::make_unique<ScopedEnv>("DFKV_PEER_COOLDOWN_MS",
                                             cooldown_ms)),
        cooldown_max(std::make_unique<ScopedEnv>("DFKV_PEER_COOLDOWN_MAX_MS",
                                                 cooldown_ms)) {}
};

// A client on `view` with the gate open (registered, polling every second,
// every registered client marked).
std::unique_ptr<KVClient> OpenClient(NodesTransport* t,
                                     const std::vector<MemberInfo>& view) {
  auto c = std::make_unique<KVClient>(
      std::vector<std::pair<std::string, std::string>>{}, kNs, t);
  c->SetMembers(view);
  c->UseRegistryGateForTest(1000);
  c->OnRegistry(true, {MarkedClient("self", 1000)});
  return c;
}

// Cools member `id` in `client`: a read of its key fails there.
void Cool(KVClient* client, NodesTransport* t, const std::string& id,
          const std::vector<std::string>& ids = kIds) {
  std::this_thread::sleep_for(std::chrono::milliseconds(3));  // past the busy grace
  {
    std::lock_guard<std::mutex> lk(t->mu);
    t->down.insert(Addr(id));
  }
  std::string out(8, '\0');
  client->Get(KeysOwnedBy(id, 1, "cool-" + id + "-", ids)[0], &out[0],
              out.size());
  std::lock_guard<std::mutex> lk(t->mu);
  t->down.erase(Addr(id));
}

class CooldownReroute : public ::testing::TestWithParam<bool> {
 protected:
  CooldownReroute() : t(GetParam()) {}
  NodesTransport t;
};

INSTANTIATE_TEST_SUITE_P(TcpAndPipelined, CooldownReroute, ::testing::Bool());

// Off (the default): with an owner cooling, keys route as on main and a
// REMOVE goes to the owner only; close sends nothing.
TEST_P(CooldownReroute, OffRoutesAndRemovesAsToday) {
  ScopedEnv busy("DFKV_PEER_BAD_GRACE_MS", "1");
  ScopedEnv cooldown("DFKV_PEER_COOLDOWN_MS", "60000");
  {
    KVClient client({}, kNs, &t);
    KVClient plain({}, kNs, &t);
    client.SetMembers(View(kIds));
    plain.SetMembers(View(kIds));
    Cool(&client, &t, "b");
    for (int i = 0; i < 2000; ++i) {
      const std::string key = "route-" + std::to_string(i);
      ASSERT_EQ(client.OwnerOf(key), plain.OwnerOf(key)) << key;
    }
    const auto keys = KeysOwnedBy("c", 4, "off-");
    std::map<std::string, int> before = t.removes;
    for (const auto& key : keys) client.Remove(key);
    for (const auto& id : kIds)
      EXPECT_EQ(t.removes[Addr(id)] - before[Addr(id)],
                id == "c" ? static_cast<int>(keys.size()) : 0)
          << id;
    EXPECT_EQ(Metric(client, "dfkv_client_registry_gate_open"), "0");
    t.removes.clear();
  }
  EXPECT_TRUE(t.removes.empty()) << "close sent REMOVEs with C off";
}

// Owner cooling, gate open: every op goes to the first stable, non-cooling
// successor; a second cooling successor is skipped.
TEST_P(CooldownReroute, ACoolingOwnersKeysGoToTheSubstitute) {
  Env env;
  auto client = OpenClient(&t, View(kIds));
  Cool(client.get(), &t, "b");
  const std::string v(16, 'v');
  const auto keys = KeysOwnedBy("b", 6, "sub-");
  for (const auto& key : keys) {
    const std::string sub = Successors(key, kIds)[1];
    ASSERT_TRUE(client->Put(key, v.data(), v.size())) << key;
    EXPECT_TRUE(t.Holds(Addr(sub), K(key))) << key;
    std::string out(v.size(), '\0');
    EXPECT_TRUE(client->Get(key, &out[0], out.size())) << key;
    EXPECT_TRUE(client->Exist(key)) << key;
  }
  std::vector<std::string> outs(keys.size(), std::string(v.size(), '\0'));
  std::vector<KvGetItem> items;
  for (size_t i = 0; i < keys.size(); ++i)
    items.push_back({keys[i], &outs[i][0], outs[i].size()});
  for (bool hit : client->BatchGet(items)) EXPECT_TRUE(hit);
  for (bool e : client->BatchExist(keys)) EXPECT_TRUE(e);
  EXPECT_NE(Metric(*client, "dfkv_client_cooldown_reroutes_total{op=\"put\"}"), "0");

  // A key whose second successor cools too goes to the third.
  std::string key;
  for (int i = 0;; ++i) {
    key = "third-" + std::to_string(i);
    const auto s = Successors(key, kIds);
    if (s[0] == "b" && s[1] == "c") break;
  }
  Cool(client.get(), &t, "c");
  ASSERT_TRUE(client->Put(key, v.data(), v.size()));
  EXPECT_TRUE(t.Holds(Addr(Successors(key, kIds)[2]), K(key)));
}

// No usable candidate among the first DFKV_RING_REMOVE_FANOUT: as today.
TEST_P(CooldownReroute, NoUsableCandidateBehavesAsToday) {
  Env env;
  ScopedEnv fanout("DFKV_RING_REMOVE_FANOUT", "2");
  auto client = OpenClient(&t, View(kIds));
  std::string key;
  for (int i = 0;; ++i) {
    key = "none-" + std::to_string(i);
    const auto s = Successors(key, kIds);
    if (s[0] == "b" && s[1] == "c") break;
  }
  Cool(client.get(), &t, "b");
  Cool(client.get(), &t, "c");
  const std::string v(16, 'v');
  EXPECT_FALSE(client->Put(key, v.data(), v.size()));
  EXPECT_NE(Metric(*client, "dfkv_client_cooldown_reroutes_total{op=\"none\"}"), "0");
}

// The gate closed (an unmarked client registered, a stale answer, no
// answer): routing is today's, at once, and the episode is cleaned up.
TEST_P(CooldownReroute, AClosedGateRoutesAsTodayAndEndsTheEpisode) {
  Env env;
  auto client = OpenClient(&t, View(kIds));
  Cool(client.get(), &t, "b");
  const std::string v(16, 'v');
  const std::string key = KeysOwnedBy("b", 1, "gate-")[0];
  const std::string sub = Successors(key, kIds)[1];
  ASSERT_TRUE(client->Put(key, v.data(), v.size()));
  ASSERT_TRUE(t.Holds(Addr(sub), K(key)));
  MemberInfo old{"old-reader", "0.0.0.0", 0, 0};
  old.info = "connector=sglang";  // an older release: no marker
  client->OnRegistry(true, {MarkedClient("self", 1000), old});
  EXPECT_EQ(Metric(*client, "dfkv_client_registry_gate_open"), "0");
  EXPECT_EQ(Metric(*client, "dfkv_client_registry_unmarked_clients"), "1");
  EXPECT_EQ(Metric(*client,
                   "dfkv_client_registry_gate_failures_total{reason=\"unmarked_client\"}"),
            "1");
  std::string out(v.size(), '\0');
  EXPECT_FALSE(client->Get(key, &out[0], out.size())) << "read the substitute";
  client->TickReroute();
  EXPECT_FALSE(t.Holds(Addr(sub), K(key))) << "the episode was not cleaned up";
  // A marked client without poll_ms keeps it closed too.
  MemberInfo half{"half", "0.0.0.0", 0, 0};
  half.info = "dfkv-remove-fanout=1";
  client->OnRegistry(true, {MarkedClient("self", 1000), half});
  EXPECT_EQ(Metric(*client, "dfkv_client_registry_gate_open"), "0");
  client->OnRegistry(true, {MarkedClient("self", 1000)});
  EXPECT_EQ(Metric(*client, "dfkv_client_registry_gate_open"), "1");
}

// The episode ends when the owner's cooldown expires: routing returns to the
// owner, and the keys written to the substitute are removed from it.
TEST_P(CooldownReroute, TheEpisodeEndRemovesTheSubstitutesCopies) {
  Env env("300");
  auto client = OpenClient(&t, View(kIds));
  Cool(client.get(), &t, "b");
  const std::string v(16, 'v');
  const auto keys = KeysOwnedBy("b", 5, "end-");
  std::vector<KvPutItem> puts;
  for (const auto& key : keys) puts.push_back({key, v.data(), v.size()});
  for (bool ok : client->BatchPut(puts)) ASSERT_TRUE(ok);
  client->TickReroute();  // still cooling: nothing removed
  for (const auto& key : keys)
    EXPECT_TRUE(t.Holds(Addr(Successors(key, kIds)[1]), K(key)));
  std::this_thread::sleep_for(std::chrono::milliseconds(350));
  client->TickReroute();
  for (const auto& key : keys)
    EXPECT_FALSE(t.Holds(Addr(Successors(key, kIds)[1]), K(key))) << key;
  EXPECT_EQ(Metric(*client, "dfkv_client_reroute_cleanup_total{result=\"ok\"}"),
            std::to_string(keys.size()));
  ASSERT_TRUE(client->Put(keys[0], v.data(), v.size()));
  EXPECT_TRUE(t.Holds(Addr("b"), K(keys[0]))) << "not back at the owner";
}

// A cleanup the substitute refuses is queued and sent from the tick; until
// it lands, a later episode does not read that copy.
TEST_P(CooldownReroute, AFailedCleanupIsQueuedAndItsCopyIsNotRead) {
  Env env("300");
  // The substitute answered the write: its failed REMOVE does not cool it.
  ScopedEnv busy("DFKV_PEER_BAD_GRACE_MS", "5000");
  auto client = OpenClient(&t, View(kIds));
  Cool(client.get(), &t, "b");
  const std::string v(16, 'v');
  const std::string key = KeysOwnedBy("b", 1, "refused-")[0];
  const std::string sub = Successors(key, kIds)[1];
  ASSERT_TRUE(client->Put(key, v.data(), v.size()));
  std::this_thread::sleep_for(std::chrono::milliseconds(350));
  {
    std::lock_guard<std::mutex> lk(t.mu);
    t.failing_removes.insert(Addr(sub));
  }
  client->TickReroute();  // the episode ends; the cleanup fails
  EXPECT_TRUE(t.Holds(Addr(sub), K(key)));
  EXPECT_EQ(Metric(*client, "dfkv_client_ring_reroute_episode_pending"), "1");
  EXPECT_EQ(Metric(*client, "dfkv_client_reroute_cleanup_total{result=\"queued\"}"), "1");
  Cool(client.get(), &t, "b");  // a later episode
  std::string out(v.size(), '\0');
  EXPECT_FALSE(client->Get(key, &out[0], out.size()))
      << "a copy waiting for its cleanup was read";
  {
    std::lock_guard<std::mutex> lk(t.mu);
    t.failing_removes.clear();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  client->TickReroute();
  EXPECT_FALSE(t.Holds(Addr(sub), K(key)));
  EXPECT_EQ(Metric(*client, "dfkv_client_ring_reroute_episode_pending"), "0");
}

// An episode is one cooldown: a trial that fails at once after the expiry
// starts the next one before any tick sees the owner not cooling, and the
// first episode's keys are still removed from their substitutes.
TEST_P(CooldownReroute, AFailedTrialEndsTheEpisodeBeforeIt) {
  Env env("300");
  auto client = OpenClient(&t, View(kIds));
  Cool(client.get(), &t, "b");
  const std::string v(16, 'v');
  const auto keys = KeysOwnedBy("b", 3, "chain-");
  for (const auto& key : keys) ASSERT_TRUE(client->Put(key, v.data(), v.size()));
  std::this_thread::sleep_for(std::chrono::milliseconds(350));
  Cool(client.get(), &t, "b");  // the trial fails: the next cooldown
  client->TickReroute();
  for (const auto& key : keys)
    EXPECT_FALSE(t.Holds(Addr(Successors(key, kIds)[1]), K(key)))
        << key << ": the first episode was not cleaned up";
}

// Episode cleanups are never given up outside close, however many fan-out
// parts wait: those have their own bound.
TEST_P(CooldownReroute, FanOutPartsDoNotCrowdOutAnEpisodeCleanup) {
  Env env("1000");
  ScopedEnv busy("DFKV_PEER_BAD_GRACE_MS", "5000");  // s1 stays usable
  ScopedEnv cap("DFKV_RING_REROUTE_KEYS", "1024");
  auto client = OpenClient(&t, View(kIds));
  const std::string v(16, 'v');
  std::string key;
  for (int i = 0;; ++i) {
    key = "crowd-" + std::to_string(i);
    const auto s = Successors(key, kIds);
    if (s[0] == "b" && s[1] != "c") break;
  }
  const std::string s1 = Successors(key, kIds)[1];
  const auto parts = KeysOwnedBy("c", 1100, "part-");  // (slow: first)
  const auto t0 = std::chrono::steady_clock::now();
  Cool(client.get(), &t, "b");  // b cools until t0 + 1 s
  ASSERT_TRUE(client->Put(key, v.data(), v.size()));
  {
    std::lock_guard<std::mutex> lk(t.mu);
    t.failing_removes.insert(Addr(s1));
    t.down.insert(Addr("b"));  // b stays down: it never answers
  }
  // An outage of c, later than b's: more REMOVE parts for c than the bound.
  std::this_thread::sleep_until(t0 + std::chrono::milliseconds(700));
  Cool(client.get(), &t, "c");  // c cools until t0 + 1.7 s
  for (bool ok : client->BatchRemove(parts)) EXPECT_FALSE(ok);
  EXPECT_NE(Metric(*client, "dfkv_client_reroute_cleanup_total{result=\"abandoned\"}"), "0");
  // b's episode ends while c still cools; its cleanup fails and waits.
  std::this_thread::sleep_until(t0 + std::chrono::milliseconds(1100));
  ASSERT_LT(std::chrono::steady_clock::now(), t0 + std::chrono::milliseconds(1600));
  client->OnRegistry(true, {MarkedClient("self", 1000)});  // a fresh answer
  client->TickReroute();
  ASSERT_TRUE(t.Holds(Addr(s1), K(key)));
  // The episode cleanup is still pending: a new episode (b's trial fails)
  // does not read it.
  std::string probe(8, '\0');
  client->Get(KeysOwnedBy("b", 1, "trial-")[0], &probe[0], probe.size());
  ASSERT_EQ(Metric(*client, "dfkv_client_registry_gate_open"), "1");
  const std::string reads =
      Metric(*client, "dfkv_client_cooldown_reroutes_total{op=\"get\"}");
  std::string out(v.size(), '\0');
  EXPECT_FALSE(client->Get(key, &out[0], out.size()))
      << "an episode cleanup was given up for the fan-out parts";
  EXPECT_EQ(Metric(*client, "dfkv_client_cooldown_reroutes_total{op=\"get\"}"),
            reads) << "the read went to the substitute";
  {
    std::lock_guard<std::mutex> lk(t.mu);
    t.failing_removes.clear();
  }
  client->TickReroute();
  EXPECT_FALSE(t.Holds(Addr(s1), K(key))) << "the episode cleanup was given up";
}

// A registry answer older than two polls closes the gate.
TEST_P(CooldownReroute, AStaleRegistryClosesTheGate) {
  if (GetParam()) GTEST_SKIP() << "timing only; once is enough";
  Env env;
  auto client = OpenClient(&t, View(kIds));
  ASSERT_EQ(Metric(*client, "dfkv_client_registry_gate_open"), "1");
  client->TickReroute();
  EXPECT_EQ(Metric(*client, "dfkv_client_registry_gate_open"), "1");
  std::this_thread::sleep_for(std::chrono::milliseconds(2100));
  client->TickReroute();
  EXPECT_EQ(Metric(*client, "dfkv_client_registry_gate_open"), "0");
  EXPECT_EQ(Metric(*client,
                   "dfkv_client_registry_gate_failures_total{reason=\"stale_registry\"}"),
            "1");
}

// At DFKV_RING_REROUTE_KEYS tracked keys, a rerouted write is refused:
// nothing is written to a substitute untracked.
TEST_P(CooldownReroute, RerouteWritesPastTheCapAreRefused) {
  Env env;
  ScopedEnv cap("DFKV_RING_REROUTE_KEYS", "1024");
  auto client = OpenClient(&t, View(kIds));
  Cool(client.get(), &t, "b");
  const std::string v(4, 'v');
  const auto keys = KeysOwnedBy("b", 1030, "cap-");
  size_t accepted = 0;
  for (const auto& key : keys) accepted += client->Put(key, v.data(), v.size());
  EXPECT_EQ(accepted, 1024u);
  EXPECT_EQ(Metric(*client, "dfkv_client_reroute_refused_total"), "6");
}

// REMOVE fans out to the first three successors whatever this client's
// cooldown state, and is ok when every one confirms.
TEST_P(CooldownReroute, ARemoveReachesTheFirstThreeSuccessors) {
  Env env;
  auto client = OpenClient(&t, View(kIds));
  const auto keys = KeysOwnedBy("a", 4, "fan-");
  for (const auto& key : keys) {
    std::map<std::string, int> before = t.removes;
    EXPECT_TRUE(client->Remove(key));
    const auto s = Successors(key, kIds);
    for (size_t i = 0; i < s.size(); ++i)
      EXPECT_EQ(t.removes[Addr(s[i])] - before[Addr(s[i])], i < 3 ? 1 : 0)
          << key << " " << s[i];
  }
  for (bool ok : client->BatchRemove(keys)) EXPECT_TRUE(ok);
}

// A target that fails without cooling (busy, not down) fails the REMOVE.
TEST_P(CooldownReroute, ATargetFailingWithoutCoolingFailsTheRemove) {
  Env env;
  auto client = OpenClient(&t, View(kIds));
  const std::string key = KeysOwnedBy("a", 1, "busy-")[0];
  const std::string second = Successors(key, kIds)[1];
  EXPECT_FALSE(client->Exist(KeysOwnedBy(second, 1, "alive-")[0]));  // answered
  {
    std::lock_guard<std::mutex> lk(t.mu);
    t.failing_removes.insert(Addr(second));
  }
  EXPECT_FALSE(client->Remove(key));
  EXPECT_EQ(Metric(*client, "dfkv_client_remove_unconfirmed_total{reason=\"failed\"}"), "1");
}

// A successor that cools only in the remover: the REMOVE is not ok, the
// part is queued, and once the cleanup lands the reader (which uses that
// successor as its substitute) misses.
TEST_P(CooldownReroute, ASuccessorCoolingOnlyForTheRemover) {
  auto env = std::make_unique<Env>("60000");
  auto reader = OpenClient(&t, View(kIds));  // cools for a minute
  env.reset();  // (its destructor unsets what the next one sets)
  env = std::make_unique<Env>("300");
  auto remover = OpenClient(&t, View(kIds));
  const std::string v(16, 'v');
  const std::string key = KeysOwnedBy("b", 1, "succ-")[0];
  const std::string sub = Successors(key, kIds)[1];
  Cool(reader.get(), &t, "b");
  ASSERT_TRUE(reader->Put(key, v.data(), v.size()));
  ASSERT_TRUE(t.Holds(Addr(sub), K(key)));
  Cool(remover.get(), &t, sub);
  EXPECT_FALSE(remover->Remove(key)) << "ok with a target not confirming";
  EXPECT_EQ(Metric(*remover, "dfkv_client_remove_unconfirmed_total{reason=\"cooling\"}"), "1");
  std::this_thread::sleep_for(std::chrono::milliseconds(350));
  remover->TickReroute();  // the cleanup lands
  std::string out(v.size(), '\0');
  EXPECT_FALSE(reader->Get(key, &out[0], out.size()));
}

// The owner cooling only for the remover: the same.
TEST_P(CooldownReroute, AnOwnerCoolingOnlyForTheRemover) {
  auto env = std::make_unique<Env>("60000");
  auto reader = OpenClient(&t, View(kIds));
  env.reset();  // (its destructor unsets what the next one sets)
  env = std::make_unique<Env>("300");
  auto remover = OpenClient(&t, View(kIds));
  const std::string v(16, 'v');
  const std::string key = KeysOwnedBy("b", 1, "own-")[0];
  ASSERT_TRUE(reader->Put(key, v.data(), v.size()));
  ASSERT_TRUE(t.Holds(Addr("b"), K(key)));
  Cool(remover.get(), &t, "b");
  EXPECT_FALSE(remover->Remove(key));
  std::this_thread::sleep_for(std::chrono::milliseconds(350));
  remover->TickReroute();
  std::string out(v.size(), '\0');
  EXPECT_FALSE(reader->Get(key, &out[0], out.size()));
}

// Two clients on different views (skew): for every key of the owner the
// reader cools, the reader writes to its substitute, the remover REMOVEs
// it, and the reader must miss.
void ExpectNoResurrection(KVClient* reader, KVClient* remover,
                          const std::string& cooled,
                          const std::vector<std::string>& reader_ids,
                          size_t keys_to_try) {
  const std::string v(16, 'v');
  size_t rerouted = 0;
  for (const auto& key : KeysOwnedBy(cooled, keys_to_try, "skew-", reader_ids)) {
    if (!reader->Put(key, v.data(), v.size())) continue;
    ++rerouted;
    EXPECT_TRUE(remover->Remove(key)) << key;
    std::string out(v.size(), '\0');
    EXPECT_FALSE(reader->Get(key, &out[0], out.size()))
        << key << ": the remover's fan-out missed the reader's substitute";
  }
  EXPECT_GT(rerouted, 0u);
}

// A member added in the reader's view only.
TEST_P(CooldownReroute, SkewAMemberAddedInTheReadersViewOnly) {
  Env env;
  const std::vector<std::string> v0 = {"a", "b", "c", "d", "e"};
  auto reader = OpenClient(&t, View(v0));
  reader->SetMembers(View(kIds));  // f added
  auto remover = OpenClient(&t, View(v0));
  Cool(reader.get(), &t, "b", kIds);
  Cool(reader.get(), &t, "c", kIds);
  ExpectNoResurrection(reader.get(), remover.get(), "b", kIds, 60);
}

// A drain in the reader's view only.
TEST_P(CooldownReroute, SkewADrainInTheReadersViewOnly) {
  Env env;
  auto reader = OpenClient(&t, View(kIds));
  reader->SetMembers(View(kIds, {"d"}));
  auto remover = OpenClient(&t, View(kIds));
  Cool(reader.get(), &t, "b");
  Cool(reader.get(), &t, "c");
  ExpectNoResurrection(reader.get(), remover.get(), "b",
                       {"a", "b", "c", "e", "f"}, 60);
}

// An absence in the reader's view only.
TEST_P(CooldownReroute, SkewAnAbsenceInTheReadersViewOnly) {
  Env env;
  ScopedEnv grace("DFKV_RING_ABSENT_GRACE_MS", "60000");
  auto reader = OpenClient(&t, View(kIds));
  reader->SetMembers(View({"a", "b", "c", "e", "f"}));  // d absent
  auto remover = OpenClient(&t, View(kIds));
  Cool(reader.get(), &t, "b");
  Cool(reader.get(), &t, "c");
  ExpectNoResurrection(reader.get(), remover.get(), "b",
                       {"a", "b", "c", "e", "f"}, 60);
}

// The other way round: the remover is ahead (a member added in its view).
TEST_P(CooldownReroute, SkewTheRemoverAhead) {
  Env env;
  const std::vector<std::string> v0 = {"a", "b", "c", "d", "e"};
  auto reader = OpenClient(&t, View(v0));
  auto remover = OpenClient(&t, View(v0));
  remover->SetMembers(View(kIds));
  Cool(reader.get(), &t, "b", v0);
  Cool(reader.get(), &t, "c", v0);
  ExpectNoResurrection(reader.get(), remover.get(), "b", v0, 60);
}

// Small rings (DFKV_RING_REMOVE_FANOUT above N): every member of the write
// ring and the drainer, each once.
TEST_P(CooldownReroute, SmallRingsSendEachTargetOnce) {
  Env env;
  for (const auto& ids : std::vector<std::vector<std::string>>{
           {"a", "b"}, {"a", "b", "c"}}) {
    auto client = OpenClient(&t, View(ids, {"b"}));
    const std::string key = KeysOwnedBy("b", 1, "small-", ids)[0];
    t.removes.clear();
    EXPECT_TRUE(client->Remove(key));
    for (const auto& id : ids) EXPECT_EQ(t.removes[Addr(id)], 1) << id;
  }
}

// A slower client registering grows S: the gate stays closed until the ring
// history covers the new S. When it leaves, the old S is kept for one old S.
TEST_P(CooldownReroute, AGrowingSHoldsTheGateUntilTheHistoryCoversIt) {
  if (GetParam()) GTEST_SKIP() << "timing only; once is enough";
  Env env;
  auto client = OpenClient(&t, View(kIds));
  ASSERT_EQ(Metric(*client, "dfkv_client_registry_gate_open"), "1");
  client->OnRegistry(true, {MarkedClient("self", 1000), MarkedClient("slow", 1100)});
  EXPECT_EQ(Metric(*client, "dfkv_client_registry_gate_open"), "0");
  EXPECT_EQ(Metric(*client,
                   "dfkv_client_registry_gate_failures_total{reason=\"s_growing\"}"),
            "1");
  std::this_thread::sleep_for(std::chrono::milliseconds(11000));
  client->OnRegistry(true, {MarkedClient("self", 1000), MarkedClient("slow", 1100)});
  EXPECT_EQ(Metric(*client, "dfkv_client_registry_gate_open"), "0") << "11 s < 12.1 s";
  std::this_thread::sleep_for(std::chrono::milliseconds(1200));
  client->OnRegistry(true, {MarkedClient("self", 1000), MarkedClient("slow", 1100)});
  EXPECT_EQ(Metric(*client, "dfkv_client_registry_gate_open"), "1");
  EXPECT_EQ(client->RecentKeepMsForTest(), 12100u);
  // The slow client leaves: the old S is kept for one old S, the gate open.
  client->OnRegistry(true, {MarkedClient("self", 1000)});
  EXPECT_EQ(Metric(*client, "dfkv_client_registry_gate_open"), "1");
  EXPECT_EQ(client->RecentKeepMsForTest(), 12100u);
  std::this_thread::sleep_for(std::chrono::milliseconds(11000));
  client->OnRegistry(true, {MarkedClient("self", 1000)});
  EXPECT_EQ(client->RecentKeepMsForTest(), 12100u) << "11 s < the old S";
  std::this_thread::sleep_for(std::chrono::milliseconds(1200));
  client->OnRegistry(true, {MarkedClient("self", 1000)});
  EXPECT_EQ(client->RecentKeepMsForTest(), 11000u);
}

// A clean close mid-episode removes the keys written to substitutes.
TEST_P(CooldownReroute, ACloseMidEpisodeLeavesNoTrackedKey) {
  Env env;
  const std::string v(16, 'v');
  const auto keys = KeysOwnedBy("b", 4, "close-");
  {
    auto client = OpenClient(&t, View(kIds));
    Cool(client.get(), &t, "b");
    for (const auto& key : keys) ASSERT_TRUE(client->Put(key, v.data(), v.size()));
    for (const auto& key : keys)
      ASSERT_TRUE(t.Holds(Addr(Successors(key, kIds)[1]), K(key)));
  }
  for (const auto& key : keys)
    EXPECT_FALSE(t.Holds(Addr(Successors(key, kIds)[1]), K(key))) << key;
}

// Through MDS discovery: the poller reads the registry, and this client's
// registration carries the marker with its poll interval.
TEST(CooldownRerouteDiscovery, TheRegistryOpensTheGateAndTheMarkerIsSent) {
  Env env;
  ScriptedTopologyMds mds(View(kIds));
  mds.SetClients({MarkedClient("self", 1000)});
  NodesTransport t(true);
  KVClient client({}, kNs, &t);
  client.StartMdsDiscovery({mds.endpoint()}, "reroute", 1000);
  ASSERT_TRUE(client.WaitForRing(5000));
  client.StartClientRegistration({mds.endpoint()}, "reroute",
                                 MemberInfo{"self", "0.0.0.0", 0, 0}, 200);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (Metric(client, "dfkv_client_registry_gate_open") != "1" &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_EQ(Metric(client, "dfkv_client_registry_gate_open"), "1");
  bool marked = false;
  while (!marked && std::chrono::steady_clock::now() < deadline) {
    for (const auto& payload : mds.ClientRegistrations())
      marked |= payload.find("dfkv-remove-fanout=1;poll_ms=1000") !=
                std::string::npos;
    if (!marked) std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  EXPECT_TRUE(marked);
  // A marked client polling more slowly: S grows, the gate closes at a poll.
  mds.SetClients({MarkedClient("self", 1000), MarkedClient("slow", 1200)});
  while (Metric(client, "dfkv_client_registry_gate_open") != "0" &&
         std::chrono::steady_clock::now() < deadline + std::chrono::seconds(5))
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_EQ(Metric(client,
                   "dfkv_client_registry_gate_failures_total{reason=\"s_growing\"}"),
            "1");
  EXPECT_EQ(Metric(client, "dfkv_client_registry_unmarked_clients"), "0");
  // An unmarked client appears in the registry: the gate stays closed.
  MemberInfo old{"old", "0.0.0.0", 0, 0};
  mds.SetClients({MarkedClient("self", 1000), old});
  while (Metric(client, "dfkv_client_registry_gate_open") != "0" &&
         std::chrono::steady_clock::now() < deadline + std::chrono::seconds(5))
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_EQ(Metric(client, "dfkv_client_registry_gate_open"), "0");
  client.StopClientRegistration();
  client.StopMdsDiscovery();
}

}  // namespace
