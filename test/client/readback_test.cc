// Previous-owner read-back (CLI-02 stage B, DFKV_RING_READBACK_MS; §3.4 of
// docs/design/ring-unavailable-owner.md): within the window after a
// write-ring change, a GET or EXIST that the current owner O answers "not
// found" asks the recent-REMOVE memory of the key's ask set (every recent
// ring's and the held ring's first F members, a drainer, an absent owner,
// O excluded) and reads the previous owner P only if every one answers "not
// removed". Run over a scripted transport of several nodes, in the TCP form
// (scalar bodies) and the pipelined RDMA form (batch bodies).
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
#include "common/raw_clock.h"
#include "common/membership.h"
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
    // Write-once per node, as the servers; a duplicate is a PUT all the same.
    store[node].emplace(K(key), std::string(static_cast<const char*>(data), len));
    written_at[node][K(key)] = RawMsExact();
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
    removed_here[node].insert(K(key));
    removed_at[node][K(key)] = RawMsExact();
    return store[node].erase(K(key)) ? Status::kOk : Status::kNotFound;
  }
  // With retain_answers, RETAIN answers as a node with retention: hit if
  // the node holds the key (and protects nothing); else every key a miss.
  bool retain_answers = false;
  Status Retain(const std::string& node, uint64_t tenant, uint64_t count,
                const std::string& payload, std::string* out) override {
    std::lock_guard<std::mutex> lk(mu);
    if (!retain_answers) {
      out->assign(count, static_cast<char>(KVClient::kRetMiss));
      return Status::kOk;
    }
    if (down.count(node)) return Status::kIOError;
    out->assign(count + kRetainTrailerBytes, '\0');
    for (uint64_t i = 0; i < count; ++i) {
      const char* d = payload.data() + kRetainHeaderBytes + i * kRetainKeyBytes;
      BlockKey k;
      k.digest_hi = net::GetU64(d);
      k.digest_lo = net::GetU64(d + 8);
      k.tenant_hash = tenant;
      (*out)[i] = static_cast<char>(store[node].count(K(k))
                                        ? KVClient::kRetHitUnprotected
                                        : KVClient::kRetMiss);
    }
    RetainTrailer tr;
    tr.server_now_s = 1;
    tr.max_ttl_s = 3600;
    tr.headroom_pct = 100;
    EncodeRetainTrailer(&(*out)[count], tr);
    return Status::kOk;
  }
  // Recent-REMOVE memory, per node: kMemory answers from the REMOVEs the
  // node took; kUnknown answers 2 for every key (a node started less than H
  // ago); kNoMemory answers kInvalid (switch off, older release); kFail
  // fails the request as a whole.
  enum Memory { kMemory, kUnknown, kNoMemory, kFail };
  std::map<std::string, Memory> memory;  // default kMemory
  std::map<std::string, std::set<Key>> removed_here;
  std::map<std::string, int> asked;  // kRemovedWithin requests, by node
  std::map<std::string, int> fail_call;  // fail this node's n-th request
  Status RemovedWithin(const std::string& node, uint64_t, uint64_t count,
                       const std::string& payload, std::string* out) override {
    std::lock_guard<std::mutex> lk(mu);
    ++asked[node];
    const Memory m = memory.count(node) ? memory[node] : kMemory;
    if (m == kNoMemory) return Status::kInvalid;
    if (m == kFail || down.count(node) ||
        (fail_call.count(node) && asked[node] == fail_call[node]))
      return Status::kIOError;
    out->assign(4 + count, '\0');
    net::PutU32(&(*out)[0], m == kUnknown ? 0 : 3600000);
    for (uint64_t i = 0; i < count; ++i) {
      const char* d = payload.data() + kRemovedWithinHeaderBytes + i * 16;
      const Key k{net::GetU64(d), net::GetU64(d + 8)};
      (*out)[4 + i] = static_cast<char>(
          m == kUnknown ? 2 : (removed_here[node].count(k) ? 1 : 0));
    }
    return Status::kOk;
  }
  // Op 26 (CLI-35): ages from the RAW times above. no_ages: the node does
  // not know the op (kInvalid; it then answers op 25); w_off: the node has
  // no recent-PUT memory. ages_asked counts op-26 requests; ages_delay_ms
  // holds each answer that long.
  std::map<std::string, std::map<Key, double>> removed_at, written_at;
  std::set<std::string> no_ages, w_off;
  std::map<std::string, int> ages_asked;
  std::map<std::string, int> ages_delay_ms;
  Status RemovedWithinAges(const std::string& node, uint64_t, uint64_t count,
                           const std::string& payload,
                           std::string* out) override {
    int delay = 0;
    {
      std::lock_guard<std::mutex> lk(mu);
      ++ages_asked[node];
      delay = ages_delay_ms.count(node) ? ages_delay_ms[node] : 0;
    }
    if (delay) std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    std::lock_guard<std::mutex> lk(mu);
    const Memory m = memory.count(node) ? memory[node] : kMemory;
    if (m == kNoMemory || no_ages.count(node)) return Status::kInvalid;
    if (m == kFail || down.count(node)) return Status::kIOError;
    const double now = RawMsExact();
    const uint32_t age_ms = net::GetU32(payload.data() + 4);
    out->assign(4 + count * 8, '\0');
    net::PutU32(&(*out)[0], m == kUnknown ? 0 : 3600000);
    for (uint64_t i = 0; i < count; ++i) {
      const char* d = payload.data() + kRemovedWithinHeaderBytes + i * 16;
      const Key k{net::GetU64(d), net::GetU64(d + 8)};
      uint32_t removed = kRemovedAgeNone, written = kRemovedAgeNone;
      if (m == kUnknown) {
        removed = kRemovedAgeUnknown;
      } else if (removed_at[node].count(k) &&
                 now - removed_at[node][k] <= age_ms) {
        removed = static_cast<uint32_t>(now - removed_at[node][k]);
      }
      if (!w_off.count(node) && written_at[node].count(k))
        written = static_cast<uint32_t>(now - written_at[node][k]);
      net::PutU32(&(*out)[4 + i * 8], removed);
      net::PutU32(&(*out)[8 + i * 8], written);
    }
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
// A registered client with fan-out on, and B configured unless `b` is false.
MemberInfo MarkedClient(const std::string& id, int poll_ms, bool b = true) {
  MemberInfo m{id, "0.0.0.0", 0, 0};
  m.info = "connector=x,dfkv-remove-fanout=1;poll_ms=" + std::to_string(poll_ms) +
           (b ? ";readback=1" : "");
  return m;
}

const std::vector<std::string> kOld = {"a", "b", "c", "d", "e"};  // before f joined

// B on (and so fan-out and the gate), on ring kOld; then f joins.
struct RbEnv {
  ScopedEnv readback{"DFKV_RING_READBACK_MS", "60000"};
};
std::unique_ptr<KVClient> OpenOn(NodesTransport* t,
                                 const std::vector<std::string>& ids) {
  auto c = std::make_unique<KVClient>(
      std::vector<std::pair<std::string, std::string>>{}, kNs, t);
  c->SetMembers(View(ids));
  c->UseRegistryGateForTest(1000);
  c->OnRegistry(true, {MarkedClient("self", 1000)});
  return c;
}
// Keys whose owner moved from `from` (on kOld) to f (on kIds).
std::vector<std::string> MovedToF(size_t n, const std::string& prefix) {
  std::vector<std::string> keys;
  for (int i = 0; keys.size() < n; ++i) {
    const std::string key = prefix + std::to_string(i);
    if (Successors(key, kIds)[0] == "f") keys.push_back(key);
  }
  return keys;
}
std::string PrevOwner(const std::string& key) { return Successors(key, kOld)[0]; }

class ReadBack : public ::testing::TestWithParam<bool> {
 protected:
  ReadBack() : t(GetParam()) {}
  NodesTransport t;
  void Store(const std::string& node_id, const std::string& key,
             const std::string& v) {
    std::lock_guard<std::mutex> lk(t.mu);
    t.store[Addr(node_id)][K(key)] = v;
  }
  int Asked(const std::string& id) {
    std::lock_guard<std::mutex> lk(t.mu);
    return t.asked[Addr(id)];
  }
};

INSTANTIATE_TEST_SUITE_P(TcpAndPipelined, ReadBack, ::testing::Bool());

// The core case: f joins; a key now owned by f misses there and is read back
// from its previous owner, every form.
TEST_P(ReadBack, AMissAfterAnOwnerChangeIsReadBackFromThePreviousOwner) {
  RbEnv env;
  auto c = OpenOn(&t, kOld);
  c->SetMembers(View(kIds));
  const std::string v(16, 'v');
  const auto keys = MovedToF(6, "rb-");
  for (const auto& key : keys) Store(PrevOwner(key), key, v);
  std::string out(v.size(), '\0');
  EXPECT_TRUE(c->Get(keys[0], &out[0], out.size()));
  EXPECT_EQ(out, v);
  EXPECT_TRUE(c->Exist(keys[1]));
  size_t len = 0;
  std::string auto_out(64, '\0');
  EXPECT_TRUE(c->GetAuto(keys[2], &auto_out[0], auto_out.size(), &len));
  EXPECT_EQ(len, v.size());
  std::vector<std::string> outs(3, std::string(v.size(), '\0'));
  std::vector<KvGetItem> items;
  for (size_t i = 0; i < 3; ++i) items.push_back({keys[3 + i], &outs[i][0], v.size()});
  for (bool hit : c->BatchGet(items)) EXPECT_TRUE(hit);
  for (bool e : c->BatchExist({keys[3], keys[4], keys[5]})) EXPECT_TRUE(e);
  EXPECT_NE(Metric(*c, "dfkv_client_readback_reads_total{result=\"hit\"}"), "0");
  EXPECT_EQ(Asked("f"), 0) << "O, which answered not found, was asked";
}

// No read-back where it is not a candidate: the owner did not change, the
// owner had it, the window passed, B off.
TEST_P(ReadBack, OnlyAnOwnersNotFoundInsideTheWindowIsACandidate) {
  RbEnv env;
  auto c = OpenOn(&t, kOld);
  c->SetMembers(View(kIds));
  const std::string v(16, 'v');
  std::string out(v.size(), '\0');
  // An owner that did not change: a miss is a miss, nothing asked.
  std::string same;
  for (int i = 0;; ++i) {
    same = "same-" + std::to_string(i);
    if (Successors(same, kIds)[0] == Successors(same, kOld)[0]) break;
  }
  EXPECT_FALSE(c->Get(same, &out[0], out.size()));
  for (const auto& id : kIds) EXPECT_EQ(Asked(id), 0) << id;
  // The owner holds it: no read-back.
  const std::string held = MovedToF(1, "held-")[0];
  Store("f", held, v);
  EXPECT_TRUE(c->Get(held, &out[0], out.size()));
  for (const auto& id : kIds) EXPECT_EQ(Asked(id), 0) << id;
}

TEST_P(ReadBack, NothingIsAskedWithBOffOrAfterTheWindow) {
  {
    auto c = OpenOn(&t, kOld);  // B off
    c->SetMembers(View(kIds));
    const std::string key = MovedToF(1, "off-")[0];
    Store(PrevOwner(key), key, std::string(8, 'v'));
    std::string out(8, '\0');
    EXPECT_FALSE(c->Get(key, &out[0], out.size()));
  }
  {
    ScopedEnv short_window("DFKV_RING_READBACK_MS", "100");
    auto c = OpenOn(&t, kOld);
    c->SetMembers(View(kIds));
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    const std::string key = MovedToF(1, "late-")[0];
    Store(PrevOwner(key), key, std::string(8, 'v'));
    std::string out(8, '\0');
    EXPECT_FALSE(c->Get(key, &out[0], out.size()));
  }
  for (const auto& id : kIds) EXPECT_EQ(Asked(id), 0) << id;
}

// The ask set leaves out O only. O answering unknown (restarted, empty M)
// does not matter; P and the other first successors are asked.
TEST_P(ReadBack, TheAskSetIsTheFirstSuccessorsWithoutTheOwner) {
  RbEnv env;
  auto c = OpenOn(&t, kOld);
  c->SetMembers(View(kIds));
  {
    std::lock_guard<std::mutex> lk(t.mu);
    t.memory[Addr("f")] = NodesTransport::kUnknown;  // O, restarted
  }
  const std::string key = MovedToF(1, "ask-")[0];
  const std::string v(8, 'v');
  Store(PrevOwner(key), key, v);
  std::string out(v.size(), '\0');
  EXPECT_TRUE(c->Get(key, &out[0], out.size()));
  std::set<std::string> want;
  for (const auto& ids : {kIds, kOld}) {
    const auto s = Successors(key, ids);
    for (size_t i = 0; i < 3 && i < s.size(); ++i) want.insert(s[i]);
  }
  want.erase("f");
  for (const auto& id : kIds)
    EXPECT_EQ(Asked(id), want.count(id) ? 1 : 0) << id;
  EXPECT_TRUE(want.count(PrevOwner(key)));
}

// Any other asked node's 2 (unknown, no memory, a failed request) or 1
// refuses: the miss stands.
TEST_P(ReadBack, AnyRemovedOrUnknownRefusesTheReadBack) {
  RbEnv env;
  const std::string v(8, 'v');
  struct Case {
    NodesTransport::Memory memory;
    const char* counter;
  };
  int n = 0;
  for (const Case& cs : {Case{NodesTransport::kUnknown, "refused_unknown"},
                         Case{NodesTransport::kNoMemory, "refused_unknown"},
                         Case{NodesTransport::kFail, "refused_unknown"}}) {
    NodesTransport tt(GetParam());
    auto c = OpenOn(&tt, kOld);
    c->SetMembers(View(kIds));
    const std::string key = MovedToF(1, "unk-" + std::to_string(n++) + "-")[0];
    const std::string p = PrevOwner(key);
    {
      std::lock_guard<std::mutex> lk(tt.mu);
      tt.store[Addr(p)][K(key)] = v;
      tt.memory[Addr(p)] = cs.memory;  // P itself cannot say
    }
    std::string out(v.size(), '\0');
    EXPECT_FALSE(c->Get(key, &out[0], out.size())) << cs.counter;
    EXPECT_EQ(Metric(*c, std::string("dfkv_client_readback_reads_total{result=\"") +
                             cs.counter + "\"}"),
              "1");
  }
  // Removed: a remover's REMOVE reached P; the reader does not read it back.
  auto c = OpenOn(&t, kOld);
  c->SetMembers(View(kIds));
  const std::string key = MovedToF(1, "rem-")[0];
  const std::string p = PrevOwner(key);
  {
    std::lock_guard<std::mutex> lk(t.mu);
    t.removed_here[Addr(p)].insert(K(key));
    t.store[Addr(p)][K(key)] = v;  // a stale copy P still holds
  }
  std::string out(v.size(), '\0');
  EXPECT_FALSE(c->Get(key, &out[0], out.size()));
  EXPECT_EQ(Metric(*c, "dfkv_client_readback_reads_total{result=\"refused_removed\"}"), "1");
}

// A node's questions go in requests of at most kRemovedWithinMaxKeys keys;
// one failed request makes every key asked there unknown, the keys of its
// answered requests included. Keys not asked there are read back.
TEST_P(ReadBack, AFailedRequestMakesEveryKeyAskedThereUnknown) {
  if (!GetParam()) GTEST_SKIP() << "a non-pipelined batch reads key by key";
  RbEnv env;
  auto c = OpenOn(&t, kOld);
  c->SetMembers(View(kIds));
  auto ring = [](const std::vector<std::string>& ids) {
    auto r = std::make_unique<ConHash>();
    for (const auto& id : ids) r->AddNode(id, 1);
    r->Build();
    return r;
  };
  const auto now_ring = ring(kIds), old_ring = ring(kOld);
  auto first3 = [](const ConHash& r, const std::string& key) {
    std::vector<std::string> out;
    r.Walk(ConHash::KeyHash(key), [&](const std::string& id) {
      out.push_back(id);
      return out.size() < 3;
    });
    return out;
  };
  const std::string v(8, 'v');
  std::vector<std::string> keys;
  size_t at_a = 0;
  for (int i = 0; keys.size() < 2400; ++i) {
    const std::string key = "chunk-" + std::to_string(i);
    const auto now3 = first3(*now_ring, key);
    if (now3[0] != "f") continue;
    const auto old3 = first3(*old_ring, key);
    keys.push_back(key);
    t.store[Addr(old3[0])][K(key)] = v;
    bool asks_a = false;
    for (const auto& id : now3) asks_a |= id == "a";
    for (const auto& id : old3) asks_a |= id == "a";
    at_a += asks_a;
  }
  ASSERT_GT(at_a, kRemovedWithinMaxKeys) << "a must get two requests";
  ASSERT_LE(at_a, 2 * kRemovedWithinMaxKeys);
  ASSERT_LT(at_a, keys.size());
  t.fail_call[Addr("a")] = 2;
  size_t hits = 0;
  for (bool e : c->BatchExist(keys)) hits += e;
  EXPECT_EQ(Asked("a"), 2);
  EXPECT_EQ(hits, keys.size() - at_a);
  EXPECT_EQ(Metric(*c, "dfkv_client_readback_reads_total{result=\"refused_unknown\"}"),
            std::to_string(at_a));
}

// A REMOVE by another client (on the new ring) reaches the asked set: the
// reader does not read the removed key back.
TEST_P(ReadBack, ARemoveByAnotherClientIsSeen) {
  RbEnv env;
  auto reader = OpenOn(&t, kOld);
  reader->SetMembers(View(kIds));
  auto remover = OpenOn(&t, kIds);
  const std::string v(8, 'v');
  size_t tried = 0;
  for (const auto& key : MovedToF(20, "other-")) {
    Store(PrevOwner(key), key, v);
    EXPECT_TRUE(remover->Remove(key));
    Store(PrevOwner(key), key, v);  // a stale copy left at P
    std::string out(v.size(), '\0');
    EXPECT_FALSE(reader->Get(key, &out[0], out.size())) << key;
    ++tried;
  }
  EXPECT_EQ(tried, 20u);
}

// The candidate rule's refusals: the registry gate closed, P cooling, a
// counted length mismatch; and a not found from a substitute (O cooling).
TEST_P(ReadBack, GateCoolingAndLengthRefuse) {
  RbEnv env;
  ScopedEnv busy("DFKV_PEER_BAD_GRACE_MS", "1");
  ScopedEnv cooldown("DFKV_PEER_COOLDOWN_MS", "60000");
  const std::string v(8, 'v');
  {
    auto c = OpenOn(&t, kOld);
    c->SetMembers(View(kIds));
    MemberInfo old{"old", "0.0.0.0", 0, 0};
    c->OnRegistry(true, {MarkedClient("self", 1000), old});  // gate closes
    const std::string key = MovedToF(1, "gate-")[0];
    Store(PrevOwner(key), key, v);
    std::string out(v.size(), '\0');
    EXPECT_FALSE(c->Get(key, &out[0], out.size()));
    EXPECT_EQ(Metric(*c, "dfkv_client_readback_reads_total{result=\"refused_gate\"}"), "1");
  }
  {
    auto c = OpenOn(&t, kOld);
    c->SetMembers(View(kIds));
    const std::string key = MovedToF(1, "pcool-")[0];
    Store(PrevOwner(key), key, v);
    // P cools in this client.
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    { std::lock_guard<std::mutex> lk(t.mu); t.down.insert(Addr(PrevOwner(key))); }
    std::string probe(8, '\0');
    c->Get(KeysOwnedBy(PrevOwner(key), 1, "probe-", kIds)[0], &probe[0], probe.size());
    { std::lock_guard<std::mutex> lk(t.mu); t.down.clear(); }
    std::string out(v.size(), '\0');
    EXPECT_FALSE(c->Get(key, &out[0], out.size()));
    EXPECT_EQ(Metric(*c, "dfkv_client_readback_reads_total{result=\"refused_cooling\"}"), "1");
  }
  {
    auto c = OpenOn(&t, kOld);
    c->SetMembers(View(kIds));
    const auto keys = MovedToF(2, "len-");
    Store(PrevOwner(keys[0]), keys[0], std::string(16, 'v'));  // another length
    Store(PrevOwner(keys[1]), keys[1], v);
    std::string out(v.size(), '\0');
    EXPECT_FALSE(c->Get(keys[0], &out[0], out.size()));  // read back: mismatch
    EXPECT_FALSE(c->Get(keys[1], &out[0], out.size()));  // refused from now on
    EXPECT_EQ(Metric(*c, "dfkv_client_readback_reads_total{result=\"refused_length\"}"), "1");
  }
}

// A second change inside an open window clears it: no read-back after it.
TEST_P(ReadBack, ASecondChangeInsideTheWindowClearsIt) {
  RbEnv env;
  auto c = OpenOn(&t, kOld);
  c->SetMembers(View(kIds));
  c->SetMembers(View({"a", "b", "c", "d", "e", "f", "g"}));  // g joins
  EXPECT_EQ(Metric(*c, "dfkv_client_readback_changes_refused_total{reason=\"chained\"}"), "1");
  EXPECT_EQ(Metric(*c, "dfkv_client_readback_window_active"), "0");
}

// B requires F >= 3: with DFKV_RING_REMOVE_FANOUT=2 it refuses to start.
TEST_P(ReadBack, FanoutBelowThreeRefusesB) {
  RbEnv env;
  ScopedEnv fanout("DFKV_RING_REMOVE_FANOUT", "2");
  auto c = OpenOn(&t, kOld);
  c->SetMembers(View(kIds));
  const std::string key = MovedToF(1, "f2-")[0];
  Store(PrevOwner(key), key, std::string(8, 'v'));
  std::string out(8, '\0');
  EXPECT_FALSE(c->Get(key, &out[0], out.size()));
  for (const auto& id : kIds) EXPECT_EQ(Asked(id), 0) << id;
}

// A cooldown-reroute episode is a change with a window: when O comes back
// empty after it, the keys written to the substitute during the episode are
// read back from it, and their cleanup waits for the window to pass.
TEST_P(ReadBack, AKeyWrittenToASubstituteIsReadBackAfterTheEpisode) {
  ScopedEnv readback("DFKV_RING_READBACK_MS", "400");
  ScopedEnv reroute("DFKV_RING_COOLDOWN_REROUTE", "1");
  ScopedEnv busy("DFKV_PEER_BAD_GRACE_MS", "1");
  ScopedEnv cooldown("DFKV_PEER_COOLDOWN_MS", "200");
  ScopedEnv cooldown_max("DFKV_PEER_COOLDOWN_MAX_MS", "200");
  auto c = OpenOn(&t, kIds);
  const std::string v(8, 'v');
  // b fails once: it cools for 200 ms (an episode).
  std::this_thread::sleep_for(std::chrono::milliseconds(3));
  { std::lock_guard<std::mutex> lk(t.mu); t.down.insert(Addr("b")); }
  std::string probe(8, '\0');
  c->Get(KeysOwnedBy("b", 1, "probe-")[0], &probe[0], probe.size());
  { std::lock_guard<std::mutex> lk(t.mu); t.down.clear(); }
  const std::string key = KeysOwnedBy("b", 1, "episode-")[0];
  const std::string sub = Successors(key, kIds)[1];
  ASSERT_TRUE(c->Put(key, v.data(), v.size()));
  ASSERT_TRUE(t.Holds(Addr(sub), K(key)));
  std::this_thread::sleep_for(std::chrono::milliseconds(250));  // expired
  c->OnRegistry(true, {MarkedClient("self", 1000)});
  c->TickReroute();  // the episode ends; its cleanup waits for the window
  EXPECT_TRUE(t.Holds(Addr(sub), K(key)));
  // b serves again, empty: its not found is read back from the substitute.
  std::string out(v.size(), '\0');
  EXPECT_TRUE(c->Get(key, &out[0], out.size()));
  EXPECT_EQ(out, v);
  std::this_thread::sleep_for(std::chrono::milliseconds(450));  // window over
  c->OnRegistry(true, {MarkedClient("self", 1000)});
  c->TickReroute();
  EXPECT_FALSE(t.Holds(Addr(sub), K(key))) << "cleaned up after the window";
  EXPECT_FALSE(c->Get(key, &out[0], out.size()));
}

// ---- Revision 11: deferred episode cleanup ----

struct DeferEnv {
  explicit DeferEnv(const char* cooldown_ms = "200")
      : cooldown{"DFKV_PEER_COOLDOWN_MS", cooldown_ms},
        cooldown_max{"DFKV_PEER_COOLDOWN_MAX_MS", cooldown_ms} {}
  ScopedEnv readback{"DFKV_RING_READBACK_MS", "400"};
  ScopedEnv reroute{"DFKV_RING_COOLDOWN_REROUTE", "1"};
  ScopedEnv busy{"DFKV_PEER_BAD_GRACE_MS", "1"};
  ScopedEnv cooldown, cooldown_max;
};
// Member `id` fails one read in `c`: it cools there for the cooldown.
void CoolIn(KVClient* c, NodesTransport* t, const std::string& id,
            const std::vector<std::string>& ids = kIds) {
  std::this_thread::sleep_for(std::chrono::milliseconds(3));
  { std::lock_guard<std::mutex> lk(t->mu); t->down.insert(Addr(id)); }
  std::string probe(8, '\0');
  c->Get(KeysOwnedBy(id, 1, "cool-" + id + "-", ids)[0], &probe[0],
         probe.size());
  std::lock_guard<std::mutex> lk(t->mu);
  t->down.erase(Addr(id));
}
const std::vector<std::string> kWithG = {"a", "b", "c", "d", "e", "f", "g"};
// The five-step shape: a key of b whose third successor S on kIds is not
// among the first three on kWithG (g lands before S), so a REMOVE on
// kWithG misses S, which a writer cooling b and c on kIds substitutes.
std::string FiveStepKey(const std::string& prefix) {
  for (int i = 0;; ++i) {
    const std::string key = prefix + std::to_string(i);
    const auto v0 = Successors(key, kIds);
    const auto v1 = Successors(key, kWithG);
    if (v0[0] == "b" && v0[1] == "c" && v1[0] == "b" && v1[1] == "c" &&
        v1[2] == "g")
      return key;
  }
}

// A deferred copy, then a ring change and a REMOVE on the new ring that
// misses S: B's M question sees the REMOVE, the copy is not read back.
TEST_P(ReadBack, ADeferredCopyAfterARemoveThatMissedItIsRefused) {
  DeferEnv env;
  auto y = OpenOn(&t, kIds);
  const std::string v(8, 'v');
  const std::string key = FiveStepKey("defer1-");
  const std::string s = Successors(key, kIds)[2];
  CoolIn(y.get(), &t, "c");
  CoolIn(y.get(), &t, "b");
  ASSERT_TRUE(y->Put(key, v.data(), v.size()));
  ASSERT_TRUE(t.Holds(Addr(s), K(key)));
  std::this_thread::sleep_for(std::chrono::milliseconds(250));  // both expired
  y->OnRegistry(true, {MarkedClient("self", 1000)});
  y->TickReroute();  // the episode ends; its cleanup is deferred
  ASSERT_TRUE(t.Holds(Addr(s), K(key)));
  y->SetMembers(View(kWithG));  // a ring change
  auto z = OpenOn(&t, kWithG);
  z->Remove(key);
  ASSERT_TRUE(t.Holds(Addr(s), K(key))) << "the REMOVE was to miss S";
  std::string out(v.size(), '\0');
  EXPECT_FALSE(y->Get(key, &out[0], out.size())) << "a removed key read back";
  EXPECT_EQ(Metric(*y, "dfkv_client_readback_reads_total{result=\"refused_removed\"}"), "1");
}

// Another client's later episode for the same owner picks the same S: its
// substitute read of a key it did not write is a read-back, refused here
// (the REMOVE missed S). With a registered client without B, the episode
// end cleans up at once instead.
TEST_P(ReadBack, AnotherClientsEpisodeDoesNotReadADeferredCopyPlainly) {
  DeferEnv env;
  auto y = OpenOn(&t, kIds);
  auto w = OpenOn(&t, kIds);
  const std::string v(8, 'v');
  const std::string key = FiveStepKey("defer2-");
  const std::string s = Successors(key, kIds)[2];
  CoolIn(y.get(), &t, "c");
  CoolIn(y.get(), &t, "b");
  ASSERT_TRUE(y->Put(key, v.data(), v.size()));
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  y->OnRegistry(true, {MarkedClient("self", 1000)});
  y->TickReroute();  // deferred
  ASSERT_TRUE(t.Holds(Addr(s), K(key)));
  auto z = OpenOn(&t, kWithG);
  z->Remove(key);  // b, c and g: S missed
  ASSERT_TRUE(t.Holds(Addr(s), K(key)));
  CoolIn(w.get(), &t, "c");
  CoolIn(w.get(), &t, "b");  // w's own episode: b's keys go to S
  std::string out(v.size(), '\0');
  EXPECT_FALSE(w->Get(key, &out[0], out.size()))
      << "another client's deferred copy was read plainly";
  EXPECT_FALSE(w->Exist(key));
  const int refused =
      std::stoi(Metric(*w, "dfkv_client_readback_reads_total{result=\"refused_removed\"}")) +
      std::stoi(Metric(*w, "dfkv_client_readback_reads_total{result=\"refused_unknown\"}"));
  EXPECT_EQ(refused, 2);
  // w's own episode writes stay plain reads.
  const std::string own = KeysOwnedBy("b", 1, "defer2-own-")[0];
  ASSERT_TRUE(w->Put(own, v.data(), v.size()));
  EXPECT_TRUE(w->Get(own, &out[0], out.size()));

  // A registered client without B: an episode end cleans up at once.
  auto y2 = OpenOn(&t, kIds);
  y2->OnRegistry(true, {MarkedClient("self", 1000), MarkedClient("old", 1000, false)});
  const std::string key2 = KeysOwnedBy("b", 1, "defer2-now-")[0];
  const std::string s2 = Successors(key2, kIds)[1];
  CoolIn(y2.get(), &t, "b");
  ASSERT_TRUE(y2->Put(key2, v.data(), v.size()));
  ASSERT_TRUE(t.Holds(Addr(s2), K(key2)));
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  y2->OnRegistry(true, {MarkedClient("self", 1000), MarkedClient("old", 1000, false)});
  y2->TickReroute();
  EXPECT_FALSE(t.Holds(Addr(s2), K(key2))) << "deferred with a client without B";
}

// A clean close during the deferral sends the deferred REMOVEs at once.
TEST_P(ReadBack, ACloseDuringTheDeferralLeavesNoTrackedKey) {
  DeferEnv env;
  const std::string v(8, 'v');
  const auto keys = KeysOwnedBy("b", 4, "defer3-");
  {
    auto y = OpenOn(&t, kIds);
    CoolIn(y.get(), &t, "b");
    for (const auto& key : keys) ASSERT_TRUE(y->Put(key, v.data(), v.size()));
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    y->OnRegistry(true, {MarkedClient("self", 1000)});
    y->TickReroute();  // deferred
    for (const auto& key : keys)
      ASSERT_TRUE(t.Holds(Addr(Successors(key, kIds)[1]), K(key)));
  }
  for (const auto& key : keys)
    EXPECT_FALSE(t.Holds(Addr(Successors(key, kIds)[1]), K(key))) << key;
}

// Deferred keys count toward DFKV_RING_REROUTE_KEYS: a later episode's
// rerouted PUTs are refused until the window has passed.
TEST_P(ReadBack, DeferredKeysFillTheRerouteCap) {
  DeferEnv env("300");
  ScopedEnv cap("DFKV_RING_REROUTE_KEYS", "1024");
  auto y = OpenOn(&t, kIds);
  const std::string v(4, 'v');
  const auto keys = KeysOwnedBy("b", 1030, "defer4-");
  CoolIn(y.get(), &t, "b");
  size_t accepted = 0;
  for (size_t i = 0; i < 1024; ++i) accepted += y->Put(keys[i], v.data(), v.size());
  ASSERT_EQ(accepted, 1024u);
  std::this_thread::sleep_for(std::chrono::milliseconds(350));
  y->OnRegistry(true, {MarkedClient("self", 1000)});
  y->TickReroute();  // all 1024 deferred
  EXPECT_EQ(Metric(*y, "dfkv_client_ring_reroute_episode_pending"), "1024");
  CoolIn(y.get(), &t, "b");  // a later episode
  for (size_t i = 1024; i < keys.size(); ++i)
    EXPECT_FALSE(y->Put(keys[i], v.data(), v.size()));
  EXPECT_EQ(Metric(*y, "dfkv_client_reroute_refused_total"), "6");
  std::this_thread::sleep_for(std::chrono::milliseconds(450));  // window over
  y->OnRegistry(true, {MarkedClient("self", 1000)});
  y->TickReroute();
  EXPECT_EQ(Metric(*y, "dfkv_client_ring_reroute_episode_pending"), "0");
}

// The deferral's condition is re-checked at every registry answer and tick:
// a client without B registering, a client dropping B, or a stale answer
// makes every deferred key due at once; a due key is not deferred again.
TEST_P(ReadBack, TheDeferralEndsWhenItsConditionFails) {
  DeferEnv env;
  const std::string v(8, 'v');
  enum Way { kJoins, kDrops, kStale };
  int n = 0;
  for (Way way : {kJoins, kDrops, kStale}) {
    auto y = OpenOn(&t, kIds);
    const std::vector<MemberInfo> all_b = {MarkedClient("self", 1000),
                                           MarkedClient("w", 1000)};
    y->OnRegistry(true, all_b);
    const auto keys = KeysOwnedBy("b", 3, "defer5-" + std::to_string(n++) + "-");
    CoolIn(y.get(), &t, "b");
    for (const auto& key : keys) ASSERT_TRUE(y->Put(key, v.data(), v.size()));
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    y->OnRegistry(true, all_b);
    y->TickReroute();  // deferred
    for (const auto& key : keys)
      ASSERT_TRUE(t.Holds(Addr(Successors(key, kIds)[1]), K(key))) << way;
    if (way == kJoins) {
      y->OnRegistry(true, {MarkedClient("self", 1000), MarkedClient("w", 1000),
                           MarkedClient("new", 1000, false)});
      y->OnRegistry(true, all_b);  // it left again: the keys stay due
    } else if (way == kDrops) {
      y->OnRegistry(true, {MarkedClient("self", 1000),
                           MarkedClient("w", 1000, false)});
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(2100));  // > 2 polls
    }
    y->TickReroute();
    for (const auto& key : keys)
      EXPECT_FALSE(t.Holds(Addr(Successors(key, kIds)[1]), K(key))) << way;
    EXPECT_EQ(Metric(*y, "dfkv_client_reroute_cleanup_total{result=\"ok\"}"), "3")
        << way;
  }
}

// A client with C but not B, while a registered client advertises B: a
// substitute read of a key it did not write in its current episode is a
// miss (it may be a deferred copy it cannot check); its own writes read.
TEST_P(ReadBack, CWithoutBMissesSubstituteReadsWhileAClientDefers) {
  ScopedEnv reroute("DFKV_RING_COOLDOWN_REROUTE", "1");
  ScopedEnv busy("DFKV_PEER_BAD_GRACE_MS", "1");
  ScopedEnv cooldown("DFKV_PEER_COOLDOWN_MS", "2000");
  auto c = OpenOn(&t, kIds);
  const std::string v(8, 'v');
  const auto keys = KeysOwnedBy("b", 2, "defer6-");
  const std::string sub = Successors(keys[0], kIds)[1];
  Store(sub, keys[0], v);  // a copy another client left there
  CoolIn(c.get(), &t, "b");
  ASSERT_TRUE(c->Put(keys[1], v.data(), v.size()));  // its own
  std::string out(v.size(), '\0');
  c->OnRegistry(true, {MarkedClient("self", 1000, false)});
  EXPECT_TRUE(c->Get(keys[0], &out[0], out.size())) << "no client defers";
  c->OnRegistry(true, {MarkedClient("self", 1000, false), MarkedClient("y", 1000)});
  EXPECT_FALSE(c->Get(keys[0], &out[0], out.size()));
  EXPECT_FALSE(c->Exist(keys[0]));
  EXPECT_TRUE(c->Get(keys[1], &out[0], out.size()));
  EXPECT_TRUE(c->Exist(keys[1]));
}

// RETAIN reports on the copy too (REFRESH is the connectors' existence
// probe; a SET's answer says the key is there): another client's deferred,
// removed copy is checked like a GET, and with C alone it is a miss.
TEST_P(ReadBack, ARetainOfAnotherClientsDeferredCopyIsChecked) {
  DeferEnv env;
  t.retain_answers = true;
  auto y = OpenOn(&t, kIds);
  auto w = OpenOn(&t, kIds);
  const std::string v(8, 'v');
  const std::string key = FiveStepKey("defer7-");
  const std::string s = Successors(key, kIds)[2];
  CoolIn(y.get(), &t, "c");
  CoolIn(y.get(), &t, "b");
  ASSERT_TRUE(y->Put(key, v.data(), v.size()));
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  y->OnRegistry(true, {MarkedClient("self", 1000)});
  y->TickReroute();  // deferred
  auto z = OpenOn(&t, kWithG);
  z->Remove(key);  // b, c and g: S missed
  ASSERT_TRUE(t.Holds(Addr(s), K(key)));
  auto hit = [](uint8_t code) {
    return code != KVClient::kRetMiss && code != KVClient::kRetUnsupported &&
           code != KVClient::kRetError;
  };
  CoolIn(w.get(), &t, "c");
  CoolIn(w.get(), &t, "b");  // w's own episode: b's keys go to S
  for (uint8_t mode : {kRetainModeRefresh, kRetainModeSet}) {
    KVClient::RetainRequest req;
    req.ttl_s = 60;
    req.mode = mode;
    req.no_debounce = true;
    uint8_t code = 0;
    w->BatchRetain({key}, req, &code);
    EXPECT_FALSE(hit(code)) << "mode " << int(mode) << ": code " << int(code);
  }
  // C without B, while a registered client advertises B: a miss too.
  ScopedEnv off("DFKV_RING_READBACK_MS", "0");
  auto c_only = OpenOn(&t, kIds);
  c_only->OnRegistry(true, {MarkedClient("self", 1000, false), MarkedClient("y", 1000)});
  CoolIn(c_only.get(), &t, "c");
  CoolIn(c_only.get(), &t, "b");
  KVClient::RetainRequest req;
  req.ttl_s = 60;
  req.mode = kRetainModeRefresh;
  req.no_debounce = true;
  uint8_t code = 0;
  c_only->BatchRetain({key}, req, &code);
  EXPECT_FALSE(hit(code)) << "C alone: code " << int(code);
}

// A refused RETAIN SET on another client's deferred copy is not tracked:
// the same client's later GET and EXIST of the key are still checked.
TEST_P(ReadBack, ARefusedSetDoesNotMakeAForeignCopyOwn) {
  DeferEnv env;
  t.retain_answers = true;
  auto y = OpenOn(&t, kIds);
  const std::string v(8, 'v');
  const std::string key = FiveStepKey("defer8-");
  const std::string s = Successors(key, kIds)[2];
  CoolIn(y.get(), &t, "c");
  CoolIn(y.get(), &t, "b");
  ASSERT_TRUE(y->Put(key, v.data(), v.size()));
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  y->OnRegistry(true, {MarkedClient("self", 1000)});
  y->TickReroute();  // deferred
  auto z = OpenOn(&t, kWithG);
  z->Remove(key);  // b, c and g: S missed
  ASSERT_TRUE(t.Holds(Addr(s), K(key)));
  KVClient::RetainRequest set;
  set.ttl_s = 60;
  set.mode = kRetainModeSet;
  set.no_debounce = true;
  auto expect_checked = [&](KVClient* c, const char* who) {
    CoolIn(c, &t, "c");
    CoolIn(c, &t, "b");  // its own episode: b's keys go to S
    uint8_t code = 0;
    c->BatchRetain({key}, set, &code);
    EXPECT_EQ(code, KVClient::kRetError) << who << ": the SET was not refused";
    std::string out(v.size(), '\0');
    EXPECT_FALSE(c->Get(key, &out[0], out.size()))
        << who << ": the refused SET made the copy its own";
    EXPECT_FALSE(c->Exist(key)) << who;
  };
  auto w = OpenOn(&t, kIds);  // B
  expect_checked(w.get(), "B");
  ScopedEnv off("DFKV_RING_READBACK_MS", "0");
  auto c_only = OpenOn(&t, kIds);
  c_only->OnRegistry(true, {MarkedClient("self", 1000, false), MarkedClient("y", 1000)});
  expect_checked(c_only.get(), "C alone");
}

// ---- CLI-35: the rewrite check (design §3.5) ----

using Answer = KVClient::MemoryAnswer;
using Rewrite = KVClient::Rewrite;
Answer Ans(uint32_t removed, uint32_t written, double sent, double answered) {
  Answer a;
  a.removed_age = removed;
  a.written_age = written;
  a.sent_ms = sent;
  a.answered_ms = answered;
  return a;
}
constexpr uint32_t kNo = kRemovedAgeNone, kUnk = kRemovedAgeUnknown;
// Node i reports a REMOVE `a` ms old, answered at e_i; P (request sent at
// 0) a write 10 ms old.
Rewrite OneRemove(uint32_t a, double e_i) {
  return KVClient::RewriteVerdict(
      "p", {{"i", Ans(a, kNo, e_i - 1, e_i)}, {"p", Ans(kNo, 10, 0, 1)}});
}

// At the margin, both ways: a/(1+rho) - (b+1)/(1-rho) > gap/(1-rho) + G with
// b = 10 and a gap of 5 ms holds from a = 1018. 1017 passes a rule without
// the rho terms (a - 11 > 1005) or without the gap (a/(1+rho) - 11.01 > 1000).
TEST(ReadBackRewrite, TheMarginHoldsBothWays) {
  EXPECT_EQ(OneRemove(1018, 5), Rewrite::kAfterEvery);
  EXPECT_EQ(OneRemove(1017, 5), Rewrite::kRemoved);
}

// Node i answered before P was asked (sequential chunks, a retry, the
// fallback): the gap counts as 0, never as a credit. With the gap taken as
// is (-50 ms), a = 1012 would pass.
TEST(ReadBackRewrite, ANegativeGapCountsAsZero) {
  EXPECT_EQ(OneRemove(1013, -50), Rewrite::kAfterEvery);
  EXPECT_EQ(OneRemove(1012, -50), Rewrite::kRemoved);
}

// P's own records share one millisecond snapshot, but records carry no
// sub-millisecond ordering. Equal ages must not turn PUT→REMOVE into a
// permissible rewrite; reject such ties conservatively.
TEST(ReadBackRewrite, PsOwnRecordsCompareExactly) {
  const auto own = [](uint32_t removed, uint32_t written) {
    return KVClient::RewriteVerdict("p", {{"p", Ans(removed, written, 0, 1)}});
  };
  EXPECT_EQ(own(100, 99), Rewrite::kAfterEvery);
  EXPECT_EQ(own(100, 100), Rewrite::kRemoved);
  EXPECT_EQ(own(99, 100), Rewrite::kRemoved);
  EXPECT_EQ(own(kNo, 100), Rewrite::kNoRemove);
}

TEST(ReadBackRewrite, UnknownAndMissingWritesRefuse) {
  // Some node could not say.
  EXPECT_EQ(KVClient::RewriteVerdict(
                "p", {{"i", Ans(kUnk, kNo, 0, 1)}, {"p", Ans(kNo, 10, 0, 1)}}),
            Rewrite::kUnknown);
  // A REMOVE not shown followed wins over unknown.
  EXPECT_EQ(KVClient::RewriteVerdict("p", {{"i", Ans(kUnk, kNo, 0, 1)},
                                           {"j", Ans(5, kNo, 0, 1)},
                                           {"p", Ans(kNo, 10, 0, 1)}}),
            Rewrite::kRemoved);
  // No write remembered at P (W off, an older node, a failed request).
  EXPECT_EQ(KVClient::RewriteVerdict(
                "p", {{"i", Ans(50000, kNo, 0, 1)}, {"p", Ans(kNo, kNo, 0, 1)}}),
            Rewrite::kRemoved);
  EXPECT_EQ(KVClient::RewriteVerdict("p", {{"i", Ans(50000, kNo, 0, 1)}}),
            Rewrite::kRemoved);
  // Nothing removed: the old rule's read-back.
  EXPECT_EQ(KVClient::RewriteVerdict(
                "p", {{"i", Ans(kNo, kNo, 0, 1)}, {"p", Ans(kNo, kNo, 0, 1)}}),
            Rewrite::kNoRemove);
}

// Real timelines with every clock within rho (fast, exact, slow), answers at
// either end of their requests and the ages floored: a read-back is only
// ever admitted when the write really followed the REMOVE; and one well
// after it is admitted.
TEST(ReadBackRewrite, ClocksWithinRhoNeverAdmitAWriteBeforeTheRemove) {
  const double rho = 1e-3;
  for (double t_w = -1500; t_w <= 3000; t_w += 7) {  // the REMOVE is at 0
    for (double r_i : {1 - rho, 1.0, 1 + rho})
      for (double r_p : {1 - rho, 1.0, 1 + rho})
        for (double spread : {0.0, 3.0, 50.0})
          for (double at_i : {0.0, 1.0})
            for (double at_p : {0.0, 1.0}) {
              const double s = 60000, e = s + spread;
              const double tau_i = s + at_i * spread, tau_p = s + at_p * spread;
              const auto a = static_cast<uint32_t>((tau_i - 0) * r_i);
              const auto b = static_cast<uint32_t>((tau_p - t_w) * r_p);
              const Rewrite v = KVClient::RewriteVerdict(
                  "p", {{"i", Ans(a, kNo, s, e)}, {"p", Ans(kNo, b, s, e)}});
              if (v == Rewrite::kAfterEvery)
                ASSERT_GT(t_w, 0) << "admitted a write before the REMOVE";
              if (t_w > 1500 && spread < 10)
                EXPECT_EQ(v, Rewrite::kAfterEvery) << "t_w " << t_w;
            }
  }
}

// Why RAW: P's CLOCK_MONOTONIC slewed 8.3 % slow (chrony's default
// maxslewrate) understates the write's age, and a write made 100 ms before
// the REMOVE looks 5 s after it. The node's RAW age refuses it.
TEST(ReadBackRewrite, ASlewedClockWouldAdmitWhatTheRawClockRefuses) {
  const double s = 60000;  // the REMOVE at 0, the write at -100
  const auto a = static_cast<uint32_t>(s);
  const auto raw = static_cast<uint32_t>(s + 100);
  const auto slewed = static_cast<uint32_t>((s + 100) * (1 - 1.0 / 12));
  const auto verdict = [&](uint32_t b) {
    return KVClient::RewriteVerdict(
        "p", {{"i", Ans(a, kNo, s, s + 1)}, {"p", Ans(kNo, b, s, s + 1)}});
  };
  EXPECT_EQ(verdict(raw), Rewrite::kRemoved);
  EXPECT_EQ(verdict(slewed), Rewrite::kAfterEvery)
      << "the hazard the raw clock is for";
}

struct RwEnv {
  ScopedEnv readback{"DFKV_RING_READBACK_MS", "60000"};
  ScopedEnv rewrite{"DFKV_RING_READBACK_REWRITE", "1"};
};

// A question split into chunks: each key carries the times of the request
// that answered it, not those of the call.
TEST_P(ReadBack, EachChunkCarriesItsOwnRequestTimes) {
  RwEnv env;
  auto c = OpenOn(&t, kIds);
  t.ages_delay_ms[Addr("a")] = 30;
  std::vector<std::string> keys;
  for (int i = 0; i < 1500; ++i) keys.push_back("chunk-times-" + std::to_string(i));
  std::vector<Answer> answers;
  ASSERT_EQ(c->RemovedWithinAges(Addr("a"), keys, 1000, &answers), Status::kOk);
  EXPECT_EQ(Asked("a"), 0) << "op 25 asked";
  EXPECT_EQ(t.ages_asked[Addr("a")], 2);
  const Answer& first = answers[0];
  const Answer& second = answers[1499];
  EXPECT_GE(first.answered_ms - first.sent_ms, 30);
  EXPECT_GE(second.sent_ms, first.answered_ms) << "one time for the whole call";
  EXPECT_EQ(answers[1023].sent_ms, first.sent_ms);
  EXPECT_EQ(answers[1024].sent_ms, second.sent_ms);
}

// A node without op 26 is asked op 25: a REMOVE comes back as just now, no
// write; it is not asked op 26 again for a while. A node without M: unknown.
TEST_P(ReadBack, ANodeWithoutTheOpIsAskedTheOldOne) {
  RwEnv env;
  auto c = OpenOn(&t, kIds);
  const std::string key = "old-node";
  t.no_ages.insert(Addr("a"));
  t.removed_here[Addr("a")].insert(K(key));
  std::vector<Answer> answers;
  ASSERT_EQ(c->RemovedWithinAges(Addr("a"), {key, "other"}, 1000, &answers),
            Status::kOk);
  EXPECT_EQ(answers[0].removed_age, 0u);
  EXPECT_EQ(answers[1].removed_age, kNo);
  EXPECT_EQ(answers[0].written_age, kNo);
  ASSERT_EQ(c->RemovedWithinAges(Addr("a"), {key}, 1000, &answers), Status::kOk);
  EXPECT_EQ(t.ages_asked[Addr("a")], 1) << "asked op 26 again at once";
  EXPECT_EQ(Asked("a"), 2);
  t.memory[Addr("b")] = NodesTransport::kNoMemory;
  EXPECT_EQ(c->RemovedWithinAges(Addr("b"), {key}, 1000, &answers),
            Status::kInvalid);
  EXPECT_EQ(answers[0].removed_age, kUnk);
}

// End to end: a key removed at an asked node 5 s ago and written at P 2 s
// ago is read back (hit_rewritten); written 4.5 s ago (inside G), it is
// refused; with the switch off, op 26 is never sent and it is refused.
TEST_P(ReadBack, ARewriteAfterARemoveIsReadBack) {
  const std::string v(8, 'v');
  const auto keys = MovedToF(4, "rewrite-");
  auto setup = [&](const std::string& key, double removed_ago,
                   double written_ago) {
    const std::string p = PrevOwner(key);
    std::string x;  // an asked node other than P and f
    for (const auto& id : Successors(key, kOld))
      if (id != p && id != "f") { x = id; break; }
    std::lock_guard<std::mutex> lk(t.mu);
    const double now = RawMsExact();
    t.store[Addr(p)][K(key)] = v;
    t.removed_here[Addr(x)].insert(K(key));
    t.removed_at[Addr(x)][K(key)] = now - removed_ago;
    t.written_at[Addr(p)][K(key)] = now - written_ago;
  };
  std::string out(v.size(), '\0');
  {
    RwEnv env;
    auto c = OpenOn(&t, kOld);
    c->SetMembers(View(kIds));
    setup(keys[0], 5000, 2000);
    EXPECT_TRUE(c->Get(keys[0], &out[0], out.size()));
    EXPECT_EQ(Metric(*c, "dfkv_client_readback_reads_total{result=\"hit_rewritten\"}"), "1");
    setup(keys[1], 5000, 4500);
    EXPECT_FALSE(c->Get(keys[1], &out[0], out.size()));
    EXPECT_EQ(Metric(*c, "dfkv_client_readback_reads_total{result=\"refused_removed\"}"), "1");
    // P's own REMOVE before its rewrite: exact, no guard.
    setup(keys[2], 5000, 2000);
    {
      std::lock_guard<std::mutex> lk(t.mu);
      const std::string p = Addr(PrevOwner(keys[2]));
      t.removed_at[p][K(keys[2])] = t.written_at[p][K(keys[2])] - 1;
    }
    EXPECT_TRUE(c->Get(keys[2], &out[0], out.size()));
  }
  {
    RbEnv env;  // B without the rewrite switch
    auto c = OpenOn(&t, kOld);
    c->SetMembers(View(kIds));
    int before = 0;
    for (const auto& [node, n] : t.ages_asked) before += n;
    setup(keys[3], 5000, 2000);
    EXPECT_FALSE(c->Get(keys[3], &out[0], out.size()));
    int after = 0;
    for (const auto& [node, n] : t.ages_asked) after += n;
    EXPECT_EQ(after, before) << "op 26 sent with the switch off";
  }
}

// Mixed releases fail closed: an asked node without op 26 (its REMOVE known
// only as a bit, so just now), or P without the recent-PUT memory.
TEST_P(ReadBack, MixedReleasesRefuseAsBefore) {
  const std::string v(8, 'v');
  for (const bool p_without_w : {false, true}) {
    NodesTransport tt(GetParam());
    RwEnv env;
    auto c = OpenOn(&tt, kOld);
    c->SetMembers(View(kIds));
    const std::string key = MovedToF(1, std::string("mixed-") +
                                            (p_without_w ? "w" : "op"))[0];
    const std::string p = PrevOwner(key);
    std::string x;
    for (const auto& id : Successors(key, kOld))
      if (id != p && id != "f") { x = id; break; }
    {
      std::lock_guard<std::mutex> lk(tt.mu);
      const double now = RawMsExact();
      tt.store[Addr(p)][K(key)] = v;
      tt.removed_here[Addr(x)].insert(K(key));
      tt.removed_at[Addr(x)][K(key)] = now - 30000;
      tt.written_at[Addr(p)][K(key)] = now - 1000;  // well after
      if (p_without_w) tt.w_off.insert(Addr(p)); else tt.no_ages.insert(Addr(x));
    }
    std::string out(v.size(), '\0');
    EXPECT_FALSE(c->Get(key, &out[0], out.size())) << p_without_w;
    EXPECT_EQ(Metric(*c, "dfkv_client_readback_reads_total{result=\"refused_removed\"}"), "1");
  }
}

}  // namespace
