// KVClient against a scripted in-process transport: per-key read outcomes,
// a node that fails every read, a node that lacks kRetain and then gains it,
// retention and existence calls that can be held. For behaviour that real
// loopback nodes cannot be made to show on demand.
#include "client/kv_client.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "client/key_map.h"
#include "transport/transport.h"
#include "transport/wire.h"

using namespace dfkv;  // NOLINT

namespace {

class ScriptedTransport final : public Transport {
 public:
  enum class Plan { kHit, kMissOnce, kIOError, kIOErrorOnce, kInvalid };

  std::mutex mu;
  std::map<std::string, std::string> values;  // by BlockKey file name
  std::map<std::string, Plan> plans;
  std::map<std::string, int> reads;
  std::map<std::string, std::string> node_of;  // node of a key's first read
  std::string failing_node;  // every read on it fails with kIOError
  // The first key read on another node than failing_node reads absent once.
  bool miss_once_first_healthy = false;
  std::string miss_once_key;
  bool retain_supported = false;
  uint8_t retain_code = KVClient::kRetMiss;  // every key of a kRetain reply
  int retain_calls = 0;
  // While set, Retain waits until it is cleared.
  bool hold_retain = false;
  std::atomic<int> exist_calls{0};
  // While set, ExistMany waits until it is cleared.
  bool hold_exist = false;
  std::condition_variable cv;

  bool pipelined() const override { return true; }
  Status Cache(const std::string&, const BlockKey&, const void*,
               size_t) override {
    return Status::kInvalid;
  }
  Status Range(const std::string&, const BlockKey&, uint64_t, uint64_t,
               std::string*, uint64_t*) override {
    return Status::kInvalid;
  }
  Status Lookup(const std::string&, const BlockKey&, uint64_t*) override {
    return Status::kInvalid;
  }
  Status Exist(const std::string&, const BlockKey&, bool*) override {
    return Status::kInvalid;
  }
  Status Remove(const std::string&, const BlockKey& key) override {
    std::lock_guard<std::mutex> lk(mu);
    values.erase(key.Filename());
    return Status::kOk;
  }

  // One read of one key on one node, by plan.
  Status ReadOne(const std::string& node, const BlockKey& key,
                 std::string* value) {
    std::lock_guard<std::mutex> lk(mu);
    const std::string name = key.Filename();
    const int n = ++reads[name];
    node_of.emplace(name, node);
    if (node == failing_node) return Status::kIOError;
    if (miss_once_first_healthy && miss_once_key.empty()) miss_once_key = name;
    if (name == miss_once_key && n == 1) return Status::kNotFound;
    const auto plan = plans.count(name) ? plans[name] : Plan::kHit;
    if (plan == Plan::kIOError) return Status::kIOError;
    if (plan == Plan::kIOErrorOnce && n == 1) return Status::kIOError;
    if (plan == Plan::kInvalid) return Status::kInvalid;
    if (plan == Plan::kMissOnce && n == 1) return Status::kNotFound;
    const auto it = values.find(name);
    if (it == values.end()) return Status::kNotFound;
    *value = it->second;
    return Status::kOk;
  }

  std::vector<Status> RangeInto(const std::string& node,
                                const std::vector<BlockKey>& keys,
                                const std::vector<RangeDst>& dsts,
                                std::vector<uint64_t>* value_lens) override {
    std::vector<Status> r(keys.size());
    if (value_lens) value_lens->assign(keys.size(), 0);
    for (size_t i = 0; i < keys.size(); ++i) {
      std::string v;
      r[i] = ReadOne(node, keys[i], &v);
      if (r[i] != Status::kOk) continue;
      std::memcpy(dsts[i].payload, v.data(), std::min(v.size(), dsts[i].n));
      if (value_lens) (*value_lens)[i] = v.size();
    }
    return r;
  }

  std::vector<Status> RangeIntoMulti(const std::string& node,
                                     const std::vector<BlockKey>& keys,
                                     const std::vector<RangeDstMulti>& dsts,
                                     std::vector<size_t>* out_lens,
                                     std::string* out_dev) override {
    if (out_dev) out_dev->clear();
    std::vector<Status> r(keys.size());
    if (out_lens) out_lens->assign(keys.size(), 0);
    for (size_t i = 0; i < keys.size(); ++i) {
      std::string v;
      r[i] = ReadOne(node, keys[i], &v);
      if (r[i] != Status::kOk) continue;
      size_t done = 0;
      for (const auto& seg : dsts[i].payloads) {
        const size_t n = std::min(seg.second, v.size() - done);
        std::memcpy(seg.first, v.data() + done, n);
        done += n;
      }
      if (out_lens) (*out_lens)[i] = v.size();
    }
    return r;
  }

  Status Retain(const std::string&, uint64_t, uint64_t count,
                const std::string&, std::string* out) override {
    std::unique_lock<std::mutex> lk(mu);
    ++retain_calls;
    cv.notify_all();
    cv.wait(lk, [&] { return !hold_retain; });
    if (!retain_supported) return Status::kInvalid;
    out->assign(count, static_cast<char>(retain_code));
    out->resize(count + kRetainTrailerBytes);
    RetainTrailer t;
    t.max_ttl_s = 3600;
    t.headroom_pct = 100;
    EncodeRetainTrailer(&(*out)[count], t);
    return Status::kOk;
  }

  std::vector<Status> ExistMany(const std::string&,
                                const std::vector<BlockKey>& keys,
                                std::vector<char>* exists,
                                std::string* out_dev) override {
    if (out_dev) out_dev->clear();
    std::unique_lock<std::mutex> lk(mu);
    ++exist_calls;
    cv.notify_all();
    cv.wait(lk, [&] { return !hold_exist; });
    exists->assign(keys.size(), 0);
    return std::vector<Status>(keys.size(), Status::kOk);
  }
};

struct Client {
  std::string ns;
  ScriptedTransport t;
  std::unique_ptr<KVClient> c;
  explicit Client(const std::string& tag, int nodes = 1)
      : ns("scripted/" + tag) {
    std::vector<std::pair<std::string, std::string>> members;
    for (int i = 0; i < nodes; ++i)
      members.push_back({"n" + std::to_string(i),
                         "127.0.0.1:" + std::to_string(1 + i)});
    c = std::make_unique<KVClient>(std::move(members), ns, &t);
  }
  // Holds the transport's next kRetain until Release().
  void HoldRetain() {
    std::lock_guard<std::mutex> lk(t.mu);
    t.hold_retain = true;
  }
  void WaitRetainCall(int calls) {
    std::unique_lock<std::mutex> lk(t.mu);
    t.cv.wait(lk, [&] { return t.retain_calls >= calls; });
  }
  void Release() {
    {
      std::lock_guard<std::mutex> lk(t.mu);
      t.hold_retain = false;
      t.hold_exist = false;
    }
    t.cv.notify_all();
  }
  std::string Name(const std::string& key) const {
    return ToBlockKey(ns, key).Filename();
  }
  void Store(const std::string& key, const std::string& v) {
    t.values[Name(key)] = v;
  }
  int Reads(const std::string& key) { return t.reads[Name(key)]; }
};

}  // namespace


// ---- the REFRESH debounce and writes that race a kRetain -------------------

// A kRetain answered for a key's old value must not seed the debounce after
// the key was removed or written while the request was out: the next
// REFRESH of the new value would be answered "protected" from the cache.
TEST(KVClientRetain, AReplyThatRacedARemoveDoesNotSeedTheDebounce) {
  Client cl("ret-race");
  cl.t.retain_supported = true;
  cl.t.retain_code = KVClient::kRetProtectedNew;
  const std::vector<std::string> keys = {"r2"};
  cl.Store("r2", std::string(64, 'v'));
  KVClient::RetainRequest set;
  set.ttl_s = 300;
  cl.HoldRetain();
  auto setting = std::async(std::launch::async, [&] {
    uint8_t o[1] = {99};
    cl.c->BatchRetain(keys, set, o);
    return o[0];
  });
  cl.WaitRetainCall(1);
  ASSERT_TRUE(cl.c->Remove("r2"));  // while the SET is out
  cl.Release();
  EXPECT_EQ(setting.get(), KVClient::kRetProtectedNew);
  cl.t.retain_code = KVClient::kRetMiss;
  KVClient::RetainRequest refresh;
  refresh.mode = 1;
  refresh.ttl_s = 300;
  uint8_t out[1] = {99};
  cl.c->BatchRetain(keys, refresh, out);
  EXPECT_EQ(out[0], KVClient::kRetMiss)
      << "the removed key was reported protected from the debounce";
}

// ---- a node without kRetain -------------------------------------------------

// The existence fallback for such a node is a network round trip. It must
// not hold the lock that every other retention call needs.
TEST(KVClientRetain, FallbackDoesNotHoldTheRetentionLock) {
  Client cl("ret-lock");
  const std::vector<std::string> keys = {"r0"};
  uint8_t out[1] = {99};
  KVClient::RetainRequest refresh;
  refresh.mode = 1;
  cl.c->BatchRetain(keys, refresh, out);  // learns: no kRetain
  ASSERT_EQ(out[0], KVClient::kRetMiss);
  {
    std::lock_guard<std::mutex> lk(cl.t.mu);
    cl.t.hold_exist = true;
  }
  const int before = cl.t.exist_calls.load();
  auto refreshing = std::async(std::launch::async, [&] {
    uint8_t o[1] = {99};
    cl.c->BatchRetain(keys, refresh, o);
  });
  {
    std::unique_lock<std::mutex> lk(cl.t.mu);
    cl.t.cv.wait(lk, [&] { return cl.t.exist_calls.load() > before; });
  }
  auto precheck = std::async(std::launch::async,
                             [&] { return cl.c->RetainPrecheck(keys, 5); });
  const bool answered = precheck.wait_for(std::chrono::seconds(2)) ==
                        std::future_status::ready;
  EXPECT_TRUE(answered) << "waited for another node group's round trip";
  {
    std::lock_guard<std::mutex> lk(cl.t.mu);
    cl.t.hold_exist = false;
  }
  cl.t.cv.notify_all();
  refreshing.get();
  if (answered) EXPECT_FALSE(precheck.get());
}

// "No kRetain" is learned once and then trusted for a while only: a node can
// be upgraded under a long-lived client.
TEST(KVClientRetain, UnsupportedVerdictIsAskedAgain) {
  ::setenv("DFKV_RETAIN_REPROBE_MS", "200", 1);
  Client cl("ret-sticky");
  ::unsetenv("DFKV_RETAIN_REPROBE_MS");
  const std::vector<std::string> keys = {"r1"};
  uint8_t out[1] = {99};
  KVClient::RetainRequest set;
  set.ttl_s = 300;
  cl.c->BatchRetain(keys, set, out);
  ASSERT_EQ(out[0], KVClient::kRetUnsupported);
  int supported = -2;
  cl.c->RetainLimits(&supported, nullptr);
  ASSERT_EQ(supported, 0);
  {
    std::lock_guard<std::mutex> lk(cl.t.mu);
    cl.t.retain_supported = true;  // the node was upgraded
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  // A precheck asks nobody: until a probe has heard otherwise, the node
  // still lacks kRetain as far as it knows.
  EXPECT_FALSE(cl.c->RetainPrecheck(keys, 5))
      << "the verdict ran out without anyone asking the node";
  cl.c->RetainLimits(&supported, nullptr);
  EXPECT_EQ(supported, 1) << "the node is still taken to lack kRetain";
  cl.c->BatchRetain(keys, set, out);
  EXPECT_EQ(out[0], KVClient::kRetMiss) << "SET still reported unsupported";
}

// ---- which misses are read a second time ----------------------------------

// The retry exists for two transients: a block that reads absent for an
// instant while the node moves it between tiers, and a read lost to
// connection turnover (an I/O error). A refused request is not retried.
TEST(KVClientRetry, BatchGetRetriesTransientMissesOnly) {
  Client cl("get");
  std::vector<std::string> keys;
  for (int i = 0; i < 10; ++i) keys.push_back("k" + std::to_string(i));
  for (const auto& k : keys) cl.Store(k, std::string(64, k[1]));
  cl.t.plans[cl.Name("k7")] = ScriptedTransport::Plan::kInvalid;
  cl.t.plans[cl.Name("k8")] = ScriptedTransport::Plan::kMissOnce;
  cl.t.plans[cl.Name("k9")] = ScriptedTransport::Plan::kIOErrorOnce;
  std::vector<std::string> bufs(keys.size(), std::string(64, '\0'));
  std::vector<KvGetItem> items;
  for (size_t i = 0; i < keys.size(); ++i)
    items.push_back(KvGetItem{keys[i], &bufs[i][0], 64});
  const std::vector<bool> got = cl.c->BatchGet(items);
  EXPECT_TRUE(got[8]) << "the transient miss was not retried";
  EXPECT_TRUE(got[9]) << "the read lost to turnover was not retried";
  EXPECT_FALSE(got[7]);
  EXPECT_EQ(cl.Reads("k7"), 1) << "a refused read was sent again";
  EXPECT_EQ(cl.Reads("k8"), 2);
  EXPECT_EQ(cl.Reads("k9"), 2);
}

// A node whose whole shard failed is in cooldown after the pass: its keys
// are not sent to it again, while a transient miss elsewhere is retried.
TEST(KVClientRetry, BatchGetDoesNotResendToANodeInCooldown) {
  Client cl("get-cooldown", 4);
  cl.t.failing_node = "127.0.0.1:4";
  cl.t.miss_once_first_healthy = true;
  std::vector<std::string> keys;
  for (int i = 0; i < 40; ++i) keys.push_back("c" + std::to_string(i));
  for (const auto& k : keys) cl.Store(k, std::string(64, 'c'));
  std::vector<std::string> bufs(keys.size(), std::string(64, '\0'));
  std::vector<KvGetItem> items;
  for (size_t i = 0; i < keys.size(); ++i)
    items.push_back(KvGetItem{keys[i], &bufs[i][0], 64});
  cl.c->BatchGet(items);
  int on_failing = 0;
  for (const auto& k : keys) {
    if (cl.t.node_of[cl.Name(k)] != cl.t.failing_node) continue;
    ++on_failing;
    EXPECT_EQ(cl.Reads(k), 1) << "sent again to a node in cooldown";
  }
  EXPECT_GT(on_failing, 0);
  EXPECT_EQ(cl.t.reads[cl.t.miss_once_key], 2) << "no retry pass ran";
}

// The variable-length batch path retries a transient miss as the others do.
TEST(KVClientRetry, BatchGetAutoRetriesATransientMiss) {
  Client cl("auto");
  std::vector<std::string> keys;
  for (int i = 0; i < 10; ++i) keys.push_back("k" + std::to_string(i));
  for (const auto& k : keys) cl.Store(k, std::string(48, k[1]));
  cl.t.plans[cl.Name("k8")] = ScriptedTransport::Plan::kMissOnce;
  cl.t.plans[cl.Name("k9")] = ScriptedTransport::Plan::kInvalid;
  std::vector<std::string> bufs(keys.size(), std::string(64, '\0'));
  std::vector<KvGetItem> items;
  for (size_t i = 0; i < keys.size(); ++i)
    items.push_back(KvGetItem{keys[i], &bufs[i][0], 64});
  std::vector<size_t> lens;
  const std::vector<bool> got = cl.c->BatchGetAuto(items, &lens);
  EXPECT_TRUE(got[8]) << "the transient miss was not retried";
  EXPECT_EQ(lens[8], 48u);
  EXPECT_FALSE(got[9]);
  EXPECT_EQ(cl.Reads("k9"), 1) << "a refused read was sent again";
}

TEST(KVClientRetry, BatchGetAutoSgRetriesTransientMissesOnly) {
  Client cl("sg");
  std::vector<std::string> keys;
  for (int i = 0; i < 10; ++i) keys.push_back("k" + std::to_string(i));
  for (const auto& k : keys) cl.Store(k, std::string(64, k[1]));
  cl.t.plans[cl.Name("k7")] = ScriptedTransport::Plan::kInvalid;
  cl.t.plans[cl.Name("k8")] = ScriptedTransport::Plan::kMissOnce;
  cl.t.plans[cl.Name("k9")] = ScriptedTransport::Plan::kIOErrorOnce;
  std::vector<std::string> a(keys.size(), std::string(32, '\0')),
      b(keys.size(), std::string(32, '\0'));
  std::vector<KvGetItemSg> items;
  for (size_t i = 0; i < keys.size(); ++i)
    items.push_back(KvGetItemSg{keys[i], {&a[i][0], &b[i][0]}, {32, 32}});
  std::vector<size_t> lens;
  const std::vector<bool> got = cl.c->BatchGetAutoSg(items, &lens);
  EXPECT_TRUE(got[8]);
  EXPECT_TRUE(got[9]);
  EXPECT_FALSE(got[7]);
  EXPECT_EQ(cl.Reads("k7"), 1) << "a refused read was sent again";
  EXPECT_EQ(cl.Reads("k9"), 2);
}
