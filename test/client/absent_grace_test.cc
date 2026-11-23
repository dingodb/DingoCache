// Absent grace (CLI-02 stage A, DFKV_RING_ABSENT_GRACE_MS): a member that
// vanishes from an admitted view stays in the full ring as absent for the
// grace: no writes, REMOVE and RETAIN to it too, no read after a miss. Run over a scripted transport of several nodes, both the TCP form
// (scalar bodies) and the pipelined RDMA form (batch bodies), and through
// MDS discovery for the poller's tick and the transport topology.
#include "client/kv_client.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <functional>
#include <map>
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

// Several nodes, each a map of values; a node in `down` fails everything.
class NodesTransport final : public Transport {
 public:
  using Key = std::pair<uint64_t, uint64_t>;
  explicit NodesTransport(bool pipelined) : pipelined_(pipelined) {}

  std::mutex mu;
  std::map<std::string, std::map<Key, std::string>> store;
  std::map<std::string, int> reads;  // reads and exists, by node
  std::map<std::string, int> removes;
  std::map<std::string, std::set<Key>> retained;
  std::set<std::string> down;
  std::set<std::string> failing_removes;  // REMOVEs fail, reads are served
  // Called with the node before a REMOVE is answered (outside the lock).
  std::function<void(const std::string&)> on_remove;
  std::vector<PeerTopology> topologies;

  static Key K(const BlockKey& key) { return {key.digest_hi, key.digest_lo}; }

  bool pipelined() const override { return pipelined_; }
  Status Cache(const std::string& node, const BlockKey& key, const void* data,
               size_t len) override {
    std::lock_guard<std::mutex> lk(mu);
    if (down.count(node)) return Status::kIOError;
    store[node][K(key)].assign(static_cast<const char*>(data), len);
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
    if (on_remove) on_remove(node);
    std::lock_guard<std::mutex> lk(mu);
    ++removes[node];
    if (down.count(node) || failing_removes.count(node)) return Status::kIOError;
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
    RetainTrailer trailer;
    trailer.max_ttl_s = 3600;
    trailer.headroom_pct = 100;
    EncodeRetainTrailer(&(*out)[count], trailer);
    return Status::kOk;
  }
  void OnPeerTopology(const PeerTopology& topology) override {
    std::lock_guard<std::mutex> lk(mu);
    topologies.push_back(topology);
  }
  // The last publication of `id`: 1 present, 0 retired, -1 none.
  int Published(const std::string& id) {
    std::lock_guard<std::mutex> lk(mu);
    for (auto it = topologies.rbegin(); it != topologies.rend(); ++it)
      if (it->peer_id == id) return it->present ? 1 : 0;
    return -1;
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

const std::string kNs = "absent/test";
const std::vector<std::string> kIds = {"a", "b", "c"};
std::string Addr(const std::string& id) {
  return "127.0.0.1:" + std::to_string(1 + (id[0] - 'a'));
}
MemberInfo Member(const std::string& id, bool draining = false) {
  MemberInfo m{id, "127.0.0.1", static_cast<uint32_t>(1 + (id[0] - 'a')), 1};
  m.draining = draining;
  return m;
}
std::vector<MemberInfo> View(const std::vector<std::string>& ids,
                             const std::set<std::string>& draining = {}) {
  std::vector<MemberInfo> ms;
  for (const auto& id : ids) ms.push_back(Member(id, draining.count(id) != 0));
  return ms;
}
std::string Owner(const std::string& key, const std::vector<std::string>& ids) {
  ConHash ring;
  for (const auto& id : ids) ring.AddNode(id, 1);
  ring.Build();
  std::string name;
  ring.LookupHash(ConHash::KeyHash(key), &name);
  return name;
}
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
// The value of a metric line, "absent" when there is none.
std::string Metric(KVClient& c, const std::string& series) {
  const std::string text = c.MetricsSnapshot();
  const std::string line = "\n" + series + " ";
  const size_t at = text.find(line);
  if (at == std::string::npos) return "absent";
  const size_t from = at + line.size();
  return text.substr(from, text.find('\n', from) - from);
}

class AbsentGrace : public ::testing::TestWithParam<bool> {
 protected:
  AbsentGrace() : t(GetParam()) {}
  NodesTransport t;
};

INSTANTIATE_TEST_SUITE_P(TcpAndPipelined, AbsentGrace, ::testing::Bool());

// Off (the default), a vanished member is gone at once, as before: its keys
// are not asked after a miss, and every key routes as on the smaller ring.
TEST_P(AbsentGrace, OffAVanishedMemberIsGoneAtOnce) {
  KVClient client({}, kNs, &t);
  KVClient plain({}, kNs, &t);
  client.SetMembers(View(kIds));
  client.SetMembers(View({"a", "c"}));
  plain.SetMembers(View({"a", "c"}));
  for (int i = 0; i < 2000; ++i) {
    const std::string key = "route-" + std::to_string(i);
    ASSERT_EQ(client.OwnerOf(key), plain.OwnerOf(key)) << key;
  }
  std::string out(8, '\0');
  for (const auto& key : KeysOwnedBy("b", 8, "gone-"))
    EXPECT_FALSE(client.Get(key, &out[0], out.size()));
  EXPECT_EQ(t.reads[Addr("b")], 0);
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_members"), "0");
}

// A member whose record vanishes while it still serves: no new write goes
// to it, and its keys are not read after a miss. A copy there may be one a
// REMOVE did not reach (a remover that never saw the member sends it to the
// new owner only); reading it back is stage B's, under its rules.
TEST_P(AbsentGrace, AnAbsentMembersKeysAreNotReadAfterAMiss) {
  ScopedEnv grace("DFKV_RING_ABSENT_GRACE_MS", "60000");
  KVClient client({}, kNs, &t);
  client.SetMembers(View(kIds));
  const std::string v(32, 'v');
  const auto on_b = KeysOwnedBy("b", 8, "held-");
  for (const auto& key : on_b) ASSERT_TRUE(client.Put(key, v.data(), v.size()));
  ASSERT_EQ(t.store[Addr("b")].size(), on_b.size());

  client.SetMembers(View({"a", "c"}));
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_members"), "1");
  std::string out(v.size(), '\0');
  EXPECT_FALSE(client.Get(on_b[0], &out[0], out.size()));
  std::string auto_out;
  EXPECT_FALSE(client.GetAuto(on_b[1], &auto_out));
  EXPECT_FALSE(client.Exist(on_b[2]));
  std::vector<std::string> outs(on_b.size(), std::string(v.size(), '\0'));
  std::vector<KvGetItem> items;
  for (size_t i = 0; i < on_b.size(); ++i)
    items.push_back({on_b[i], &outs[i][0], outs[i].size()});
  for (bool hit : client.BatchGet(items)) EXPECT_FALSE(hit);
  std::vector<size_t> lens;
  for (bool hit : client.BatchGetAuto(items, &lens)) EXPECT_FALSE(hit);
  for (bool e : client.BatchExist(on_b)) EXPECT_FALSE(e);
  EXPECT_EQ(t.reads[Addr("b")], 0) << "an absent member was read";

  // New writes of b's keys go to their owner on the ring without b.
  for (const auto& key : KeysOwnedBy("b", 8, "fresh-")) {
    ASSERT_TRUE(client.Put(key, v.data(), v.size()));
    EXPECT_EQ(t.store[Addr(Owner(key, {"a", "c"}))].count(K(key)), 1u) << key;
  }
  EXPECT_EQ(t.store[Addr("b")].size(), on_b.size()) << "b took a new write";
}

// RETAIN goes to the absent member too (its copies are kept for its return),
// but its answer does not stand for the key: no read finds that copy now.
TEST_P(AbsentGrace, RetainReachesTheAbsentMemberButItsAnswerDoesNotCount) {
  ScopedEnv grace("DFKV_RING_ABSENT_GRACE_MS", "60000");
  KVClient client({}, kNs, &t);
  client.SetMembers(View(kIds));
  const std::string v(16, 'v');
  const auto keys = KeysOwnedBy("b", 2, "kept-");
  ASSERT_TRUE(client.Put(keys[0], v.data(), v.size()));  // only on b
  client.SetMembers(View({"a", "c"}));
  ASSERT_TRUE(client.Put(keys[1], v.data(), v.size()));  // only on the new owner
  std::vector<uint8_t> out(keys.size(), 0);
  KVClient::RetainRequest req;
  req.ttl_s = 60;
  req.mode = kRetainModeSet;
  client.BatchRetain(keys, req, out.data());
  EXPECT_EQ(out[0], KVClient::kRetMiss) << "b's copy is not readable now";
  EXPECT_EQ(out[1], KVClient::kRetProtectedNew);
  for (const auto& key : keys)
    EXPECT_EQ(t.retained[Addr("b")].count(K(key)), 1u) << key;
}

// A drain and an absence at once: the drainer stays its keys' second owner.
// Key K is owned by the drainer a on the view (a c) and by the absent b on
// the view with b; a REMOVE of K must reach a, or after b's grace a read
// after a miss finds K's removed value on a.
TEST_P(AbsentGrace, AnAbsenceDoesNotHideADrainerFromARemove) {
  ScopedEnv grace("DFKV_RING_ABSENT_GRACE_MS", "50");
  KVClient client({}, kNs, &t);
  client.SetMembers(View(kIds, {"a"}));  // a drains
  std::vector<std::string> keys;
  for (int i = 0; keys.size() < 4; ++i) {
    const std::string key = "hidden-" + std::to_string(i);
    if (Owner(key, kIds) == "b" && Owner(key, {"a", "c"}) == "a")
      keys.push_back(key);
  }
  const std::string v(16, 'v');
  for (const auto& key : keys) t.store[Addr("a")][K(key)] = v;  // written before the drain
  client.SetMembers(View({"a", "c"}, {"a"}));                  // b vanishes
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_members"), "1");
  EXPECT_TRUE(client.Remove(keys[0]));
  std::vector<std::string> rest(keys.begin() + 1, keys.end());
  for (bool ok : client.BatchRemove(rest)) EXPECT_TRUE(ok);
  std::this_thread::sleep_for(std::chrono::milliseconds(80));
  client.TickAbsent();
  ASSERT_EQ(Metric(client, "dfkv_client_ring_absent_members"), "0");
  std::string out(v.size(), '\0');
  for (const auto& key : keys) {
    EXPECT_FALSE(client.Get(key, &out[0], out.size())) << key << " came back";
    EXPECT_FALSE(client.Exist(key)) << key;
    EXPECT_EQ(t.store[Addr("a")].count(K(key)), 0u) << key;
  }
}

// The cooldown is ended for the first send after a return only. Here the
// member answers in between (clearing PeerHealth's once-per-incident probe
// mark) and then fails again: the next tick must not end that cooldown.
TEST_P(AbsentGrace, AReturnEndsTheCooldownOnce) {
  ScopedEnv grace("DFKV_RING_ABSENT_GRACE_MS", "60000");
  ScopedEnv busy("DFKV_PEER_BAD_GRACE_MS", "1");
  ScopedEnv cooldown("DFKV_PEER_COOLDOWN_MS", "200");
  KVClient client({}, kNs, &t);
  client.SetMembers(View(kIds));
  const std::string v(16, 'v');
  const auto keys = KeysOwnedBy("b", 3, "once-");
  for (const auto& key : keys) ASSERT_TRUE(client.Put(key, v.data(), v.size()));
  client.SetMembers(View({"a", "c"}));
  t.failing_removes.insert(Addr("b"));
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  EXPECT_TRUE(client.Remove(keys[0]));    // b cools down
  client.SetMembers(View(kIds));          // back: one reset, the retry fails
  std::this_thread::sleep_for(std::chrono::milliseconds(450));  // that cooldown passes
  std::string out(v.size(), '\0');
  EXPECT_TRUE(client.Get(keys[1], &out[0], out.size()));  // b answers
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  EXPECT_FALSE(client.Remove(keys[2]));  // and fails again: a new cooldown
  const int removes = t.removes[Addr("b")];
  client.TickAbsent();
  EXPECT_EQ(t.removes[Addr("b")], removes)
      << "a tick ended the cooldown of a member back for a while";
}

// Back within its grace, a member is an ordinary member again: it owns its
// keys as before (the ring is deterministic: as without the switch).
TEST_P(AbsentGrace, AMemberBackWithinItsGraceOwnsItsKeysAgain) {
  ScopedEnv grace("DFKV_RING_ABSENT_GRACE_MS", "60000");
  KVClient client({}, kNs, &t);
  client.SetMembers(View(kIds));
  std::map<std::string, std::string> before;
  for (int i = 0; i < 500; ++i) {
    const std::string key = "k" + std::to_string(i);
    before[key] = client.OwnerOf(key);
  }
  client.SetMembers(View({"a", "c"}));
  client.SetMembers(View(kIds));
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_members"), "0");
  for (const auto& [key, owner] : before) EXPECT_EQ(client.OwnerOf(key), owner);
  const std::string v(8, 'v');
  for (const auto& key : KeysOwnedBy("b", 4, "back-")) {
    ASSERT_TRUE(client.Put(key, v.data(), v.size()));
    EXPECT_EQ(t.store[Addr("b")].count(K(key)), 1u);
  }
}

TEST_P(AbsentGrace, AMemberIsDroppedWhenItsGracePasses) {
  ScopedEnv grace("DFKV_RING_ABSENT_GRACE_MS", "100");
  KVClient client({}, kNs, &t);
  client.SetMembers(View(kIds));
  client.SetMembers(View({"a", "c"}));
  client.TickAbsent();
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_members"), "1");
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  client.TickAbsent();
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_members"), "0");
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_dropped_total{reason=\"grace\"}"), "1");
  std::string out(8, '\0');
  for (const auto& key : KeysOwnedBy("b", 4, "late-"))
    EXPECT_FALSE(client.Get(key, &out[0], out.size()));
  EXPECT_EQ(t.reads[Addr("b")], 0);
}

// At most max(1, N/8) members are absent: on three members, one. The
// second to vanish is dropped at once.
TEST_P(AbsentGrace, AtMostAnEighthOfTheMembersAreAbsent) {
  ScopedEnv grace("DFKV_RING_ABSENT_GRACE_MS", "60000");
  KVClient client({}, kNs, &t);
  client.SetMembers(View(kIds));
  client.SetMembers(View({"a", "c"}));
  client.SetMembers(View({"a"}));
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_members"), "1");
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_dropped_total{reason=\"bound\"}"), "1");
  // b, absent first, is the one kept: a REMOVE of its key reaches it.
  const auto on_b = KeysOwnedBy("b", 1, "first-");
  const auto on_c = KeysOwnedBy("c", 1, "second-");
  client.Remove(on_b[0]);
  client.Remove(on_c[0]);
  EXPECT_EQ(t.removes[Addr("b")], 1);
  EXPECT_EQ(t.removes[Addr("c")], 0);
}

// A member that drained before it left is a planned leave: not kept.
TEST_P(AbsentGrace, AMemberThatDrainedIsNotKept) {
  ScopedEnv grace("DFKV_RING_ABSENT_GRACE_MS", "60000");
  KVClient client({}, kNs, &t);
  client.SetMembers(View(kIds, {"b"}));
  client.SetMembers(View({"a", "c"}));
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_members"), "0");
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_dropped_total{reason=\"planned\"}"), "1");
}

// REMOVE goes to the absent member too. It is usually down: the REMOVE
// succeeds on the write owner's confirmation, and the absent member's part
// is retried once when it returns.
TEST_P(AbsentGrace, ARemoveTheAbsentMemberMissedIsRetriedOnItsReturn) {
  ScopedEnv grace("DFKV_RING_ABSENT_GRACE_MS", "60000");
  KVClient client({}, kNs, &t);
  client.SetMembers(View(kIds));
  const std::string v(16, 'v');
  const auto keys = KeysOwnedBy("b", 5, "removed-");
  for (const auto& key : keys) ASSERT_TRUE(client.Put(key, v.data(), v.size()));
  client.SetMembers(View({"a", "c"}));
  t.down.insert(Addr("b"));
  EXPECT_TRUE(client.Remove(keys[0]));
  std::vector<std::string> rest(keys.begin() + 1, keys.end());
  for (bool ok : client.BatchRemove(rest)) EXPECT_TRUE(ok);
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_removes_unconfirmed_total"),
            std::to_string(keys.size()));
  EXPECT_EQ(t.store[Addr("b")].size(), keys.size());

  t.down.clear();
  client.SetMembers(View(kIds));
  EXPECT_EQ(t.store[Addr("b")].size(), 0u) << "the retry did not reach b";
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_removes_retried_total"),
            std::to_string(keys.size()));
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_removes_dropped_total"), "0");
  std::string out(16, '\0');
  for (const auto& key : keys) EXPECT_FALSE(client.Get(key, &out[0], out.size()));
}

// The REMOVEs that failed during the absence put the member in PeerHealth
// cooldown; its return to the view ends the cooldown, so the retry is sent
// at once instead of being skipped as for a cooling peer.
TEST_P(AbsentGrace, TheRetryIsNotSkippedByTheCooldownTheAbsenceStarted) {
  ScopedEnv grace("DFKV_RING_ABSENT_GRACE_MS", "60000");
  ScopedEnv busy("DFKV_PEER_BAD_GRACE_MS", "1");
  ScopedEnv cooldown("DFKV_PEER_COOLDOWN_MS", "60000");
  KVClient client({}, kNs, &t);
  client.SetMembers(View(kIds));
  const std::string v(16, 'v');
  const auto keys = KeysOwnedBy("b", 3, "cooled-");
  for (const auto& key : keys) ASSERT_TRUE(client.Put(key, v.data(), v.size()));
  client.SetMembers(View({"a", "c"}));
  t.down.insert(Addr("b"));
  std::this_thread::sleep_for(std::chrono::milliseconds(5));  // past the busy grace
  for (const auto& key : keys) EXPECT_TRUE(client.Remove(key));
  EXPECT_NE(client.MetricsSnapshot().find("dfkv_client_peer_marked_bad_total 1"),
            std::string::npos)
      << "the absence did not cool b down; the test proves nothing";
  t.down.clear();
  client.SetMembers(View(kIds));
  EXPECT_EQ(t.store[Addr("b")].size(), 0u) << "the retry was skipped";
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_removes_dropped_total"), "0");
}

// A member back but still failing keeps its REMOVEs: they are sent again at
// each tick until confirmed.
TEST_P(AbsentGrace, AnUnconfirmedRetryIsSentAgainAtTheNextTick) {
  ScopedEnv grace("DFKV_RING_ABSENT_GRACE_MS", "60000");
  KVClient client({}, kNs, &t);
  client.SetMembers(View(kIds));
  const std::string v(16, 'v');
  const auto keys = KeysOwnedBy("b", 2, "again-");
  for (const auto& key : keys) ASSERT_TRUE(client.Put(key, v.data(), v.size()));
  client.SetMembers(View({"a", "c"}));
  t.down.insert(Addr("b"));
  for (const auto& key : keys) EXPECT_TRUE(client.Remove(key));
  client.SetMembers(View(kIds));  // back in the view, still failing
  EXPECT_EQ(t.store[Addr("b")].size(), keys.size());
  t.down.clear();
  client.TickAbsent();
  EXPECT_EQ(t.store[Addr("b")].size(), 0u);
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_removes_dropped_total"), "0");
}

// ... for up to the grace after the return; then they are given up.
TEST_P(AbsentGrace, RetriesStopAGraceAfterTheReturn) {
  ScopedEnv grace("DFKV_RING_ABSENT_GRACE_MS", "50");
  KVClient client({}, kNs, &t);
  client.SetMembers(View(kIds));
  const auto keys = KeysOwnedBy("b", 2, "given-up-");
  client.SetMembers(View({"a", "c"}));
  t.down.insert(Addr("b"));
  for (const auto& key : keys) EXPECT_TRUE(client.Remove(key));
  client.SetMembers(View(kIds));
  std::this_thread::sleep_for(std::chrono::milliseconds(80));
  client.TickAbsent();
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_removes_dropped_total"), "2");
  const int removes = t.removes[Addr("b")];
  client.TickAbsent();
  EXPECT_EQ(t.removes[Addr("b")], removes);
}

// A REMOVE the absent member confirms needs no retry.
TEST_P(AbsentGrace, ARemoveReachesAnAbsentMemberThatServes) {
  ScopedEnv grace("DFKV_RING_ABSENT_GRACE_MS", "60000");
  KVClient client({}, kNs, &t);
  client.SetMembers(View(kIds));
  const std::string v(16, 'v');
  const auto keys = KeysOwnedBy("b", 3, "served-");
  for (const auto& key : keys) ASSERT_TRUE(client.Put(key, v.data(), v.size()));
  client.SetMembers(View({"a", "c"}));
  for (bool ok : client.BatchRemove(keys)) EXPECT_TRUE(ok);
  EXPECT_EQ(t.store[Addr("b")].size(), 0u);
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_removes_unconfirmed_total"), "0");
}

TEST_P(AbsentGrace, TheRetryQueueIsBounded) {
  ScopedEnv grace("DFKV_RING_ABSENT_GRACE_MS", "60000");
  ScopedEnv queue("DFKV_RING_ABSENT_REMOVE_QUEUE", "2");
  KVClient client({}, kNs, &t);
  client.SetMembers(View(kIds));
  const auto keys = KeysOwnedBy("b", 3, "bounded-");
  client.SetMembers(View({"a", "c"}));
  t.down.insert(Addr("b"));
  for (const auto& key : keys) EXPECT_TRUE(client.Remove(key));
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_removes_unconfirmed_total"), "3");
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_removes_dropped_total"), "1");
  t.down.clear();
  client.SetMembers(View(kIds));
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_removes_retried_total"), "2");
}

// A member dropped (grace over) takes its queued REMOVEs with it: they are
// counted as never confirmed, and nothing is sent when it comes back.
TEST_P(AbsentGrace, ADroppedMembersQueuedRemovesAreCounted) {
  ScopedEnv grace("DFKV_RING_ABSENT_GRACE_MS", "50");
  KVClient client({}, kNs, &t);
  client.SetMembers(View(kIds));
  const auto keys = KeysOwnedBy("b", 2, "late-remove-");
  client.SetMembers(View({"a", "c"}));
  t.down.insert(Addr("b"));
  for (const auto& key : keys) EXPECT_TRUE(client.Remove(key));
  std::this_thread::sleep_for(std::chrono::milliseconds(80));
  client.TickAbsent();
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_removes_dropped_total"), "2");
  t.down.clear();
  const int removes_before = t.removes[Addr("b")];
  client.SetMembers(View(kIds));
  EXPECT_EQ(t.removes[Addr("b")], removes_before);
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_removes_retried_total"), "0");
}

// Switch off: a REMOVE the drainer refused fails even if the drain ends
// while it is in flight (the batch keeps what it found before the pass).
TEST(AbsentGraceOff, ARemoveTheDrainerRefusedFailsWhenTheDrainEndsMeanwhile) {
  NodesTransport t(true);
  KVClient client({}, kNs, &t);
  client.SetMembers(View(kIds, {"b"}));
  const std::string key = KeysOwnedBy("b", 1, "refused-")[0];
  const std::string v(16, 'v');
  t.store[Addr("b")][K(key)] = v;
  t.failing_removes.insert(Addr("b"));
  t.on_remove = [&](const std::string& node) {
    if (node == Addr("b")) client.SetMembers(View(kIds));  // the drain ends
  };
  EXPECT_FALSE(client.BatchRemove({key})[0]) << "a refused REMOVE reported ok";
  t.on_remove = nullptr;
}

// A grace above the 600 s cap is clamped (with a WARNING), not refused.
TEST(AbsentGraceConfig, AGraceAboveTheCapIsClampedNotRefused) {
  ScopedEnv grace("DFKV_RING_ABSENT_GRACE_MS", "900000");
  NodesTransport t(false);
  KVClient client({}, kNs, &t);
  client.SetMembers(View(kIds));
  client.SetMembers(View({"a", "c"}));
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_members"), "1");
}

// Through MDS discovery: a deleted member record makes the member absent
// (after the member-view guard admits the smaller view), its transport
// topology stays published while it is absent, and the poller's tick drops
// it when the grace passes.
TEST(AbsentGraceDiscovery, ADeletedRecordIsAbsentForTheGraceThenDropped) {
  ScopedEnv grace("DFKV_RING_ABSENT_GRACE_MS", "1500");
  ScriptedTopologyMds mds(View(kIds));
  NodesTransport t(true);
  KVClient client({}, kNs, &t);
  client.StartMdsDiscovery({mds.endpoint()}, "absent", 20);
  ASSERT_TRUE(client.WaitForRing(5000));
  const std::string v(16, 'v');
  const auto on_b = KeysOwnedBy("b", 4, "polled-");
  for (const auto& key : on_b) ASSERT_TRUE(client.Put(key, v.data(), v.size()));

  auto wait_for = [&](const std::string& series, const std::string& value) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (Metric(client, series) != value &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return Metric(client, series) == value;
  };
  mds.SetMembers(View({"a", "c"}));
  ASSERT_TRUE(wait_for("dfkv_client_ring_absent_members", "1"));
  EXPECT_EQ(t.Published("b"), 1) << "an absent member's topology was retired";
  EXPECT_TRUE(client.Remove(on_b[0]));
  EXPECT_EQ(t.store[Addr("b")].count(K(on_b[0])), 0u) << "the REMOVE missed b";

  ASSERT_TRUE(wait_for("dfkv_client_ring_absent_members", "0"));
  EXPECT_EQ(Metric(client, "dfkv_client_ring_absent_dropped_total{reason=\"grace\"}"), "1");
  EXPECT_EQ(t.Published("b"), 0) << "a dropped member's topology stays";
  EXPECT_TRUE(client.Remove(on_b[1]));
  EXPECT_EQ(t.store[Addr("b")].count(K(on_b[1])), 1u) << "a dropped member was asked";
  client.StopMdsDiscovery();
}

}  // namespace
