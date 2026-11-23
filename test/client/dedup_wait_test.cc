// Host rendezvous wait path, end to end through KVClient: what a batch pays
// when the process that owns the fetch for its keys fails, hangs, or reports
// "does not exist". A second attachment of the same shm segment plays the
// owning process, so every interleaving here is deterministic.
#include "client/key_map.h"
#include "client/kv_client.h"
#include "client/node_dedup.h"

#include <gtest/gtest.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace dfkv {
namespace {

// The abort tests never wait when the code is right, so their wait is long
// and the bound is that wait; only the hung-owner test pays its wait.
constexpr int kLongWaitMs = 5000;
constexpr int kShortWaitMs = 200;
constexpr size_t kValueLen = 64;

struct DedupEnv {
  std::string name;
  DedupEnv(const std::string& key_namespace, int wait_ms) {
    ::setenv("DFKV_CLIENT_NODE_DEDUP", "1", 1);
    ::setenv("DFKV_NODE_DEDUP_ARENA_MB", "1", 1);
    ::setenv("DFKV_NODE_DEDUP_SLOTS", "4096", 1);
    ::setenv("DFKV_NODE_DEDUP_WAIT_MS", std::to_string(wait_ms).c_str(), 1);
    name = NodeDedup::EnvSegmentName(ToBlockKey(key_namespace, "").digest_hi);
    ::shm_unlink(name.c_str());
  }
  ~DedupEnv() {
    ::shm_unlink(name.c_str());
    ::unsetenv("DFKV_CLIENT_NODE_DEDUP");
    ::unsetenv("DFKV_NODE_DEDUP_ARENA_MB");
    ::unsetenv("DFKV_NODE_DEDUP_SLOTS");
    ::unsetenv("DFKV_NODE_DEDUP_WAIT_MS");
  }
  // The other process on this host: same segment, same geometry.
  std::unique_ptr<NodeDedup> Peer() const {
    NodeDedup::Options o;
    o.name = name;
    o.arena_bytes = 1ull << 20;
    o.slots = 4096;
    return NodeDedup::Open(o);
  }
};

// Non-pipelined (TCP-shaped) transport over an in-memory map.
class MapTransport final : public Transport {
 public:
  std::map<std::string, std::string> values;  // set before the client runs
  bool absent_is_ok_false = false;  // scalar TCP convention for "absent"
  std::function<void()> before_first_answer;
  // Runs before each RangeInto answers, outside the map lock.
  std::function<void(const std::vector<BlockKey>&)> before_range;
  std::atomic<size_t> range_calls{0}, exist_calls{0};

  Status Cache(const std::string&, const BlockKey& key, const void* data,
               size_t len) override {
    std::lock_guard<std::mutex> lock(mu_);
    values[key.Filename()] = std::string(static_cast<const char*>(data), len);
    return Status::kOk;
  }
  Status Range(const std::string&, const BlockKey&, uint64_t, uint64_t,
               std::string*, uint64_t*) override {
    return Status::kInvalid;
  }
  Status Lookup(const std::string&, const BlockKey&, uint64_t*) override {
    return Status::kInvalid;
  }
  Status Exist(const std::string&, const BlockKey& key,
               bool* exists) override {
    FirstAnswer();
    ++exist_calls;
    std::lock_guard<std::mutex> lock(mu_);
    *exists = values.count(key.Filename()) != 0;
    if (*exists || absent_is_ok_false) return Status::kOk;
    return Status::kNotFound;
  }
  std::vector<Status> RangeInto(const std::string&,
                                const std::vector<BlockKey>& keys,
                                const std::vector<RangeDst>& dsts,
                                std::vector<uint64_t>* value_lens) override {
    FirstAnswer();
    if (before_range) before_range(keys);
    range_calls += keys.size();
    std::lock_guard<std::mutex> lock(mu_);
    if (value_lens) value_lens->assign(keys.size(), 0);
    std::vector<Status> result(keys.size(), Status::kNotFound);
    for (size_t i = 0; i < keys.size(); ++i) {
      auto found = values.find(keys[i].Filename());
      if (found == values.end()) continue;
      const size_t copied = std::min(dsts[i].n, found->second.size());
      if (copied) std::memcpy(dsts[i].payload, found->second.data(), copied);
      if (value_lens) (*value_lens)[i] = found->second.size();
      result[i] = Status::kOk;
    }
    return result;
  }

 private:
  void FirstAnswer() {
    std::call_once(once_, [this] {
      if (before_first_answer) before_first_answer();
    });
  }
  std::mutex mu_;
  std::once_flag once_;
};

// Per-process namespace, hence a per-process segment: concurrent runs of this
// binary on one host do not meet in shared memory.
std::string Namespace(const std::string& tag) {
  return "dedup-wait/" + tag + "/" + std::to_string(::getpid());
}

std::string ValueOf(size_t i) {
  return std::string(kValueLen, static_cast<char>('a' + i % 26));
}

uint64_t Counter(const std::string& snapshot, const std::string& name) {
  const size_t at = snapshot.find("\n" + name + " ");
  EXPECT_NE(at, std::string::npos) << name;
  if (at == std::string::npos) return 0;
  return std::stoull(snapshot.substr(at + name.size() + 2));
}

int64_t ElapsedMs(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

// The peer owns the fetch for `n` keys ("k<i>"). In the abort tests it gives
// every one of them up while this client is already waiting on them: the
// client's batch leads with one key the peer does not hold, whose remote read
// is the first thing to happen after every key of the batch has been claimed,
// and that is where the peer aborts.
struct OwningPeer {
  std::unique_ptr<NodeDedup> peer;
  std::vector<BlockKey> bks;
  std::vector<uint64_t> tokens;
  char scratch[kValueLen];

  OwningPeer(const DedupEnv& env, const std::string& key_namespace,
               size_t n, NodeDedup::Kind kind)
      : peer(env.Peer()), bks(n), tokens(n, 0) {
    EXPECT_TRUE(peer);
    if (!peer) return;
    for (size_t i = 0; i < n; ++i) {
      bks[i] = ToBlockKey(key_namespace, "k" + std::to_string(i));
      bool val = false;
      const NodeDedup::Role role =
          kind == NodeDedup::Kind::kExist
              ? peer->ClaimExist(bks[i], &val, &tokens[i])
              : peer->Claim(bks[i], kValueLen, scratch, &tokens[i]);
      EXPECT_EQ(role, NodeDedup::Role::kFetch);
    }
  }
  void AbortAll(NodeDedup::Kind kind) {
    for (size_t i = 0; i < bks.size(); ++i)
      peer->Abort(bks[i], kind, tokens[i]);
  }
};

TEST(ClientDedupWait, AbortedOwnerDoesNotCostOneWaitPerKey) {
  const std::string key_namespace = Namespace("abort-get");
  const size_t N = 32;
  DedupEnv env(key_namespace, kLongWaitMs);
  MapTransport transport;
  // Half of the keys exist: the keys given up must come back through the
  // client's own fetch with the right bytes, not as blanket misses.
  for (size_t i = 0; i < N; i += 2)
    transport.values[ToBlockKey(key_namespace, "k" + std::to_string(i))
                         .Filename()] = ValueOf(i);
  KVClient client({{"n", "test:1"}}, key_namespace, &transport);
  OwningPeer owner(env, key_namespace, N, NodeDedup::Kind::kData);
  ASSERT_TRUE(owner.peer);
  transport.before_first_answer = [&] {
    owner.AbortAll(NodeDedup::Kind::kData);
  };

  std::vector<std::string> bufs(N + 1, std::string(kValueLen, '\0'));
  std::vector<KvGetItem> items;
  items.push_back({"own", bufs[N].data(), kValueLen});
  for (size_t i = 0; i < N; ++i)
    items.push_back({"k" + std::to_string(i), bufs[i].data(), kValueLen});

  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<bool> got = client.BatchGet(items);
  const int64_t ms = ElapsedMs(t0);

  ASSERT_EQ(got.size(), N + 1);
  EXPECT_FALSE(got[0]);
  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(got[i + 1], i % 2 == 0) << i;
    if (i % 2 == 0) EXPECT_EQ(bufs[i], ValueOf(i)) << i;
  }
  const std::string snapshot = client.MetricsSnapshot();
  // Every one of the N keys really was waited on and then fetched directly.
  EXPECT_EQ(Counter(snapshot, "dfkv_client_dedup_fetches_total"), 1u);
  EXPECT_EQ(Counter(snapshot, "dfkv_client_dedup_wait_timeouts_total"), N);
  EXPECT_EQ(transport.range_calls.load(), N + 1);
  std::printf("[ timing ] BatchGet, %zu keys given up by their owner: %lld ms "
              "(wait_ms=%d)\n", N, static_cast<long long>(ms), kLongWaitMs);
  EXPECT_LT(ms, kLongWaitMs) << "an aborted fetch must end the wait at once";
}

TEST(ClientDedupWait, AbortedExistOwnerDoesNotCostOneWaitPerKey) {
  const std::string key_namespace = Namespace("abort-exist");
  const size_t N = 32;
  DedupEnv env(key_namespace, kLongWaitMs);
  MapTransport transport;
  for (size_t i = 0; i < N; i += 2)
    transport.values[ToBlockKey(key_namespace, "k" + std::to_string(i))
                         .Filename()] = ValueOf(i);
  KVClient client({{"n", "test:1"}}, key_namespace, &transport);
  OwningPeer owner(env, key_namespace, N, NodeDedup::Kind::kExist);
  ASSERT_TRUE(owner.peer);
  transport.before_first_answer = [&] {
    owner.AbortAll(NodeDedup::Kind::kExist);
  };

  std::vector<std::string> keys{"own"};
  for (size_t i = 0; i < N; ++i) keys.push_back("k" + std::to_string(i));

  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<bool> got = client.BatchExist(keys);
  const int64_t ms = ElapsedMs(t0);

  ASSERT_EQ(got.size(), N + 1);
  EXPECT_FALSE(got[0]);
  for (size_t i = 0; i < N; ++i) EXPECT_EQ(got[i + 1], i % 2 == 0) << i;
  const std::string snapshot = client.MetricsSnapshot();
  EXPECT_EQ(Counter(snapshot, "dfkv_client_dedup_fetches_total"), 1u);
  EXPECT_EQ(Counter(snapshot, "dfkv_client_dedup_wait_timeouts_total"), N);
  EXPECT_EQ(transport.exist_calls.load(), N + 1);
  std::printf("[ timing ] BatchExist, %zu keys given up by their owner: %lld "
              "ms (wait_ms=%d)\n", N, static_cast<long long>(ms), kLongWaitMs);
  EXPECT_LT(ms, kLongWaitMs) << "an aborted probe must end the wait at once";
}

// An owner that neither publishes nor aborts (hung, or killed mid-fetch) is
// waited for once per batch.
TEST(ClientDedupWait, HungOwnerCostsOneWaitPerBatch) {
  const std::string key_namespace = Namespace("hung");
  const size_t N = 16;
  DedupEnv env(key_namespace, kShortWaitMs);
  MapTransport transport;
  for (size_t i = 0; i < N; ++i)
    transport.values[ToBlockKey(key_namespace, "k" + std::to_string(i))
                         .Filename()] = ValueOf(i);
  KVClient client({{"n", "test:1"}}, key_namespace, &transport);
  OwningPeer owner(env, key_namespace, N, NodeDedup::Kind::kData);
  ASSERT_TRUE(owner.peer);

  std::vector<std::string> bufs(N, std::string(2 * kValueLen, '\0'));
  std::vector<KvGetItem> items;
  for (size_t i = 0; i < N; ++i)
    items.push_back({"k" + std::to_string(i), bufs[i].data(), 2 * kValueLen});

  std::vector<size_t> lens;
  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<bool> got = client.BatchGetAuto(items, &lens);
  const int64_t ms = ElapsedMs(t0);

  ASSERT_EQ(got.size(), N);
  ASSERT_EQ(lens.size(), N);
  for (size_t i = 0; i < N; ++i) {
    EXPECT_TRUE(got[i]) << i;
    EXPECT_EQ(lens[i], kValueLen) << i;
    EXPECT_EQ(bufs[i].substr(0, kValueLen), ValueOf(i)) << i;
  }
  EXPECT_EQ(Counter(client.MetricsSnapshot(),
                    "dfkv_client_dedup_wait_timeouts_total"), N);
  std::printf("[ timing ] BatchGetAuto, %zu keys whose owner hangs: %lld ms "
              "(wait_ms=%d)\n", N, static_cast<long long>(ms), kShortWaitMs);
  EXPECT_GE(ms, kShortWaitMs - 20) << "a fetch in flight is still waited for";
  EXPECT_LT(ms, 3 * kShortWaitMs) << "one wait for the batch, not one per key";
}

// TcpTransport::Exist answers "absent" as kOk with exists=false. That is an
// answer like any other: it is shared, not aborted.
TEST(ClientDedupWait, AbsentReportedAsOkFalseIsPublished) {
  const std::string key_namespace = Namespace("absent-ok");
  DedupEnv env(key_namespace, kLongWaitMs);
  MapTransport transport;
  transport.absent_is_ok_false = true;
  KVClient client({{"n", "test:1"}}, key_namespace, &transport);
  const std::vector<std::string> keys{"missing"};

  EXPECT_EQ(client.BatchExist(keys), std::vector<bool>({false}));
  EXPECT_EQ(client.BatchExist(keys), std::vector<bool>({false}));
  EXPECT_EQ(transport.exist_calls.load(), 1u)
      << "the second probe must be served from the shared answer";

  // A PUT still retires the shared absence.
  const std::string value = ValueOf(0);
  ASSERT_TRUE(client.Put("missing", value.data(), value.size()));
  EXPECT_EQ(client.BatchExist(keys), std::vector<bool>({true}));
}

// A batch whose owner is still fetching part of it: the keys the owner has
// already got are handed over at once, not when its whole batch is done. The
// owner here gets the first half of its keys, then hangs on the rest until
// the waiter is through; the waiter's budget runs out on that rest.
template <typename Fetch>
void EarlyPublishCase(const std::string& tag, Fetch fetch) {
  const std::string key_namespace = Namespace(tag);
  const size_t N = 16;
  DedupEnv env(key_namespace, kShortWaitMs);
  MapTransport owner_transport;
  MapTransport waiter_transport;
  std::vector<std::string> filenames(N);
  for (size_t i = 0; i < N; ++i) {
    filenames[i] = ToBlockKey(key_namespace, "k" + std::to_string(i)).Filename();
    owner_transport.values[filenames[i]] = ValueOf(i);
    waiter_transport.values[filenames[i]] = ValueOf(i);
  }
  std::mutex gate_mu;
  std::condition_variable gate_cv;
  bool released = false;
  std::atomic<size_t> answered{0};
  owner_transport.before_range = [&](const std::vector<BlockKey>& keys) {
    for (const auto& key : keys) {
      const size_t index = static_cast<size_t>(
          std::find(filenames.begin(), filenames.end(), key.Filename()) -
          filenames.begin());
      if (index >= N / 2) {
        std::unique_lock<std::mutex> lock(gate_mu);
        gate_cv.wait_for(lock, std::chrono::seconds(10),
                         [&] { return released; });
      } else {
        ++answered;
      }
    }
  };
  KVClient owner({{"n", "test:1"}}, key_namespace, &owner_transport);
  KVClient waiter({{"n", "test:1"}}, key_namespace, &waiter_transport);

  std::vector<std::string> owner_bufs(N, std::string(kValueLen, '\0'));
  std::vector<bool> owner_got;
  std::thread owner_thread([&] { owner_got = fetch(owner, &owner_bufs); });
  const auto first_half = std::chrono::steady_clock::now();
  while (answered.load() < N / 2 &&
         std::chrono::steady_clock::now() - first_half < std::chrono::seconds(5))
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  ASSERT_EQ(answered.load(), N / 2);
  // The owner's workers have returned from the first half.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  std::vector<std::string> bufs(N, std::string(kValueLen, '\0'));
  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<bool> got = fetch(waiter, &bufs);
  const int64_t ms = ElapsedMs(t0);
  {
    std::lock_guard<std::mutex> lock(gate_mu);
    released = true;
  }
  gate_cv.notify_all();
  owner_thread.join();

  for (size_t i = 0; i < N; ++i) {
    EXPECT_TRUE(got[i]) << i;
    EXPECT_EQ(bufs[i], ValueOf(i)) << i;
    EXPECT_TRUE(owner_got[i]) << i;
    EXPECT_EQ(owner_bufs[i], ValueOf(i)) << i;
  }
  std::printf("[ timing ] %s: waiter took %lld ms (wait_ms=%d) and fetched "
              "%zu of %zu keys itself\n", tag.c_str(),
              static_cast<long long>(ms), kShortWaitMs,
              waiter_transport.range_calls.load(), N);
  EXPECT_EQ(waiter_transport.range_calls.load(), N / 2)
      << "the keys the owner already had must not be fetched again";
}

TEST(ClientDedupWait, OwnerPublishesEachKeyWhenItArrives) {
  EarlyPublishCase("early-get", [](KVClient& client,
                                   std::vector<std::string>* bufs) {
    std::vector<KvGetItem> items;
    for (size_t i = 0; i < bufs->size(); ++i)
      items.push_back({"k" + std::to_string(i), (*bufs)[i].data(), kValueLen});
    return client.BatchGet(items);
  });
}

TEST(ClientDedupWait, OwnerPublishesEachAutoKeyWhenItArrives) {
  EarlyPublishCase("early-auto", [](KVClient& client,
                                    std::vector<std::string>* bufs) {
    std::vector<KvGetItem> items;
    for (size_t i = 0; i < bufs->size(); ++i)
      items.push_back({"k" + std::to_string(i), (*bufs)[i].data(), kValueLen});
    std::vector<size_t> lens;
    auto got = client.BatchGetAuto(items, &lens);
    for (size_t i = 0; i < got.size(); ++i)
      if (got[i]) EXPECT_EQ(lens[i], kValueLen) << i;
    return got;
  });
}

}  // namespace
}  // namespace dfkv
