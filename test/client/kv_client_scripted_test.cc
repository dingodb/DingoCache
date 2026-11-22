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
