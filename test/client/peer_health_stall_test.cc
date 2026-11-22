// What one failing or slow cache node costs the rest of a client process:
// how long the node is avoided after an incident, what the first requests
// after the avoidance period cost, and what the background probe may decide.
// A scripted transport stands in for the nodes; every test is its own process
// under ctest, so the environment set here does not leak.
#include "client/kv_client.h"
#include "common/membership.h"
#include "scripted_topology_mds.h"
#include "transport/transport.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace dfkv;  // NOLINT
using namespace std::chrono_literals;

namespace {

using Clock = std::chrono::steady_clock;

int64_t MsSince(Clock::time_point t0) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() -
                                                               t0)
      .count();
}

// Pipelined (RDMA-shaped) transport: batch reads go through the process-wide
// read scheduler, as in production.
class ScriptedTransport final : public Transport {
 public:
  struct Node {
    Status read = Status::kNotFound;  // what a batch read returns per key
    Status put = Status::kOk;
    Status exist = Status::kOk;       // kOk + exists=false: an answered miss
    std::chrono::milliseconds delay{0};
    // A read does not return before this many reads of the node are in
    // flight (or half a second has passed): failures that land together.
    int together = 0;
    // What the RDMA transport does with an address whose connect failed:
    // for this long after a failed read, further reads fail at once without
    // the delay (no dial). Zero: every read is a dial.
    std::chrono::milliseconds hold{0};
  };

  void Set(const std::string& node, const Node& behaviour) {
    std::lock_guard<std::mutex> lock(mu_);
    nodes_[node] = behaviour;
  }
  int Reads(const std::string& node) {
    std::lock_guard<std::mutex> lock(mu_);
    return reads_[node];
  }
  int MaxConcurrentReads(const std::string& node) {
    std::lock_guard<std::mutex> lock(mu_);
    return max_inflight_[node];
  }
  // Reads refused at once because the address was held after a failure.
  int HeldReads(const std::string& node) {
    std::lock_guard<std::mutex> lock(mu_);
    return held_[node];
  }
  // Holds the address as if a connect to it had just failed elsewhere (the
  // background probe, another client handle).
  void HoldFor(const std::string& node, std::chrono::milliseconds hold) {
    std::lock_guard<std::mutex> lock(mu_);
    hold_until_[node] = Clock::now() + hold;
  }
  bool TakeUnreachableRefusal() override {
    const bool refused = refused_on_this_thread_;
    refused_on_this_thread_ = false;
    return refused;
  }
  // Blocks until `count` reads of `node` are in flight, or `wait` has passed.
  bool WaitInflight(const std::string& node, int count,
                    std::chrono::milliseconds wait) {
    std::unique_lock<std::mutex> lock(mu_);
    return cv_.wait_for(lock, wait, [&] { return inflight_[node] >= count; });
  }

  bool pipelined() const override { return true; }
  Status Cache(const std::string& node, const BlockKey&, const void*,
               size_t) override {
    return Behaviour(node).put;
  }
  Status Range(const std::string&, const BlockKey&, uint64_t, uint64_t,
               std::string*, uint64_t*) override {
    return Status::kInvalid;
  }
  Status Lookup(const std::string&, const BlockKey&, uint64_t*) override {
    return Status::kInvalid;
  }
  // Membership query, not reported to the node's health record: refused
  // while the address is held, like a read; otherwise the scripted node has
  // no member list to give.
  Status Members(const std::string& node, std::string*) override {
    std::lock_guard<std::mutex> lock(mu_);
    if (Clock::now() < hold_until_[node]) {
      ++held_[node];
      refused_on_this_thread_ = true;
    }
    return Status::kIOError;
  }
  Status Exist(const std::string& node, const BlockKey&,
               bool* exists) override {
    *exists = false;
    return Behaviour(node).exist;
  }
  std::vector<Status> CacheFrom(const std::string& node,
                                const std::vector<CacheSrc>& srcs) override {
    return std::vector<Status>(srcs.size(), Behaviour(node).put);
  }
  std::vector<Status> RangeInto(const std::string& node,
                                const std::vector<BlockKey>& keys,
                                const std::vector<RangeDst>&,
                                std::vector<uint64_t>* value_lens) override {
    Node behaviour;
    {
      std::unique_lock<std::mutex> lock(mu_);
      behaviour = nodes_[node];
      if (Clock::now() < hold_until_[node]) {
        ++held_[node];
        refused_on_this_thread_ = true;
        if (value_lens) value_lens->assign(keys.size(), 0);
        return std::vector<Status>(keys.size(), Status::kIOError);
      }
      ++reads_[node];
      const int now = ++inflight_[node];
      if (now > max_inflight_[node]) max_inflight_[node] = now;
      cv_.notify_all();
      if (behaviour.together > 0)
        cv_.wait_for(lock, 500ms,
                     [&] { return inflight_[node] >= behaviour.together; });
    }
    if (behaviour.delay.count() > 0) std::this_thread::sleep_for(behaviour.delay);
    {
      std::lock_guard<std::mutex> lock(mu_);
      // Leave together: every waiter above has seen the full count by now.
      if (behaviour.together == 0) --inflight_[node];
      if (behaviour.read == Status::kIOError && behaviour.hold.count() > 0)
        hold_until_[node] = Clock::now() + behaviour.hold;
    }
    if (value_lens) value_lens->assign(keys.size(), 0);
    return std::vector<Status>(keys.size(), behaviour.read);
  }

 private:
  Node Behaviour(const std::string& node) {
    std::lock_guard<std::mutex> lock(mu_);
    return nodes_[node];
  }
  std::mutex mu_;
  std::condition_variable cv_;
  std::map<std::string, Node> nodes_;
  std::map<std::string, int> reads_, inflight_, max_inflight_, held_;
  std::map<std::string, Clock::time_point> hold_until_;
  static thread_local bool refused_on_this_thread_;
};

thread_local bool ScriptedTransport::refused_on_this_thread_ = false;

// 500 ms base avoidance, 5 s cap: the production 2 s and 30 s, scaled, and
// long enough for the sleeps below to hold on a loaded machine.
constexpr int kBaseMs = 500;
constexpr int kCapMs = 5000;
constexpr int kSlackMs = 100;

void ScaledHealthEnv() {
  ::setenv("DFKV_PEER_COOLDOWN_MS", std::to_string(kBaseMs).c_str(), 1);
  ::setenv("DFKV_PEER_COOLDOWN_MAX_MS", std::to_string(kCapMs).c_str(), 1);
  ::unsetenv("DFKV_PROBE_INTERVAL_MS");
  ::unsetenv("DFKV_CLIENT_NODE_DEDUP");
}

// One cache node, known to the client either from a static member list or
// from an MDS. The node address is never dialed: the transport is scripted.
struct OneNodeRing {
  std::string addr;
  std::unique_ptr<ScriptedTopologyMds> mds;
  std::unique_ptr<KVClient> client;

  OneNodeRing(bool mds_managed, const std::string& key_namespace,
              Transport* transport, uint32_t port = 29001)
      : addr("127.0.0.1:" + std::to_string(port)) {
    using Members = std::vector<std::pair<std::string, std::string>>;
    if (!mds_managed) {
      client = std::make_unique<KVClient>(Members{{"n1", addr}}, key_namespace,
                                          transport);
      return;
    }
    mds = std::make_unique<ScriptedTopologyMds>(
        std::vector<MemberInfo>{MemberInfo{"n1", "127.0.0.1", port, 1}});
    client = std::make_unique<KVClient>(Members{}, key_namespace, transport);
    client->StartMdsDiscovery({mds->endpoint()}, "stall-test", 1000);
    EXPECT_TRUE(client->WaitForRing(10000));
  }
  ~OneNodeRing() {
    if (mds) client->StopMdsDiscovery();
  }
};

// A batch read of `count` equal-sized keys: with the default 16 keys per
// shard and 8 shards per node, 128 keys to one node are 8 shards.
std::vector<bool> ReadBatch(KVClient* client, const std::string& prefix,
                            size_t count) {
  std::vector<std::string> keys(count);
  std::vector<std::vector<char>> bufs(count, std::vector<char>(8));
  std::vector<KvGetItem> items;
  for (size_t i = 0; i < count; ++i) {
    keys[i] = prefix + std::to_string(i);
    items.push_back({keys[i], bufs[i].data(), bufs[i].size()});
  }
  return client->BatchGet(items);
}

}  // namespace

// Six shards of one batch fail together on a node that has just gone away.
// That is one incident: the node is tried again after the base period.
void FailuresThatLandTogetherAreOneIncident(bool mds_managed) {
  ScaledHealthEnv();
  ScriptedTransport transport;
  OneNodeRing ring(mds_managed, "stall/incident", &transport);
  ScriptedTransport::Node gone;
  gone.read = Status::kIOError;
  gone.together = 6;  // one node's shards run on six of the seven workers
  transport.Set(ring.addr, gone);

  ReadBatch(ring.client.get(), "k", 128);
  const auto failed_at = Clock::now();
  const int first = transport.Reads(ring.addr);
  ASSERT_GE(first, 6);

  gone.together = 0;
  transport.Set(ring.addr, gone);
  while (transport.Reads(ring.addr) == first &&
         MsSince(failed_at) < 2 * kCapMs) {
    std::this_thread::sleep_for(5ms);
    ReadBatch(ring.client.get(), "probe", 1);
  }
  const int64_t avoided_ms = MsSince(failed_at);
  std::printf("[ measured ] node avoided for %lld ms after %d failures that "
              "landed together (base %d ms, cap %d ms)\n",
              static_cast<long long>(avoided_ms), first, kBaseMs, kCapMs);
  EXPECT_GE(avoided_ms, kBaseMs - 10);
  EXPECT_LT(avoided_ms, 4 * kBaseMs)
      << "avoided for " << avoided_ms << " ms: " << first
      << " concurrent failures were counted as " << first
      << " consecutive ones";
}

TEST(PeerHealthStall, FailuresThatLandTogetherAreOneIncident) {
  FailuresThatLandTogetherAreOneIncident(/*mds_managed=*/false);
}
TEST(PeerHealthStall, FailuresThatLandTogetherAreOneIncidentMdsRing) {
  FailuresThatLandTogetherAreOneIncident(/*mds_managed=*/true);
}

// When the avoidance period of a dead node ends, one request finds out
// whether it is back. The others keep skipping it meanwhile and come back as
// misses, exactly as during the cooldown, so a batch for that node does not
// take every read worker of the process into a connect timeout, and a read
// of another node is served at once.
void OneRequestTriesARecoveringNode(bool mds_managed) {
  ScaledHealthEnv();
  ScriptedTransport transport;
  OneNodeRing x(mds_managed, "stall/half-open", &transport, 29001);
  OneNodeRing y(mds_managed, "stall/half-open", &transport, 29002);
  ScriptedTransport::Node dead;
  dead.read = Status::kIOError;
  dead.delay = 600ms;  // a connect timeout, scaled
  transport.Set(x.addr, dead);
  transport.Set(y.addr, ScriptedTransport::Node{});

  ReadBatch(x.client.get(), "first", 1);       // the failure that cools it
  std::this_thread::sleep_for(std::chrono::milliseconds(kBaseMs + kSlackMs));

  std::vector<bool> got;
  std::thread batch([&] { got = ReadBatch(x.client.get(), "k", 128); });
  // Without the trial, six shards of that batch would be inside the dead
  // node now (the per-node share of the read workers).
  transport.WaitInflight(x.addr, 6, 150ms);
  const auto t0 = Clock::now();
  ReadBatch(y.client.get(), "k", 16);
  const int64_t other_ms = MsSince(t0);
  batch.join();

  const int concurrent = transport.MaxConcurrentReads(x.addr);
  std::printf("[ measured ] read of a healthy node took %lld ms while a batch "
              "re-tried a dead one; %d of its shards were inside the dead "
              "node at once (timeout %lld ms)\n",
              static_cast<long long>(other_ms), concurrent,
              static_cast<long long>(dead.delay.count()));
  EXPECT_EQ(concurrent, 1) << "shards of one batch inside the dead node at once";
  EXPECT_LT(other_ms, 200)
      << "a read of another node waited " << other_ms
      << " ms behind the dead node's connect timeouts";
  // The trial shard and the seven skipped ones all come back as misses.
  EXPECT_EQ(got, std::vector<bool>(128, false));
  EXPECT_EQ(transport.Reads(x.addr), 2) << "the first failure and one trial";
}

TEST(PeerHealthStall, OneRequestTriesARecoveringNodeAndOthersAreNotHeldUp) {
  OneRequestTriesARecoveringNode(/*mds_managed=*/false);
}
TEST(PeerHealthStall,
     OneRequestTriesARecoveringNodeAndOthersAreNotHeldUpMdsRing) {
  OneRequestTriesARecoveringNode(/*mds_managed=*/true);
}

// The RDMA transport does not dial an address again for a while after a
// connect to it failed (2 s by default, the same as the base cooldown here),
// and fails requests for it at once meanwhile. Both periods end together.
// With one trial there is then one dial; without it, every shard of the next
// batch passes the transport's check before any of them has failed, and each
// waits for its own connect timeout.
TEST(PeerHealthStall, OneTrialIsOneDialWhenTheTransportHoldEndsToo) {
  ScaledHealthEnv();
  ScriptedTransport transport;
  OneNodeRing ring(/*mds_managed=*/false, "stall/hold", &transport);
  ScriptedTransport::Node dead;
  dead.read = Status::kIOError;
  dead.delay = 300ms;                               // the connect timeout
  dead.hold = std::chrono::milliseconds(kBaseMs);   // the transport's hold
  transport.Set(ring.addr, dead);

  ReadBatch(ring.client.get(), "first", 1);         // dial 1: fails
  std::this_thread::sleep_for(std::chrono::milliseconds(kBaseMs + kSlackMs));
  ReadBatch(ring.client.get(), "k", 128);           // dial 2: the trial
  EXPECT_EQ(transport.Reads(ring.addr), 2);
  EXPECT_EQ(transport.MaxConcurrentReads(ring.addr), 1)
      << "dials of one batch that waited for a connect timeout together";
  // The failed trial started the next, doubled cooldown: nothing reaches the
  // transport, held or not, until it is over.
  const int held = transport.HeldReads(ring.addr);
  ReadBatch(ring.client.get(), "again", 128);
  EXPECT_EQ(transport.Reads(ring.addr), 2);
  EXPECT_EQ(transport.HeldReads(ring.addr), held);
  std::this_thread::sleep_for(std::chrono::milliseconds(2 * kBaseMs + kSlackMs));
  ReadBatch(ring.client.get(), "later", 128);       // dial 3: the next trial
  EXPECT_EQ(transport.Reads(ring.addr), 3);
  EXPECT_EQ(transport.MaxConcurrentReads(ring.addr), 1);
}

// The background probe is an existence query. A node whose store has failed
// answers it (a miss) while every read fails. A probe that succeeds may bring
// the next attempt forward once; it must not wipe the node's record each
// round, or the data path walks into the same failure once per probe period.
TEST(PeerHealthStall, ProbeAnswerDoesNotOutvoteFailingReads) {
  ScaledHealthEnv();
  ::setenv("DFKV_PROBE_INTERVAL_MS", "20", 1);
  ScriptedTransport transport;
  ScriptedTransport::Node failed_store;
  failed_store.read = Status::kIOError;
  failed_store.exist = Status::kOk;  // answered: "not there"
  OneNodeRing ring(/*mds_managed=*/false, "stall/probe", &transport);
  transport.Set(ring.addr, failed_store);
  KVClient& client = *ring.client;

  const auto t0 = Clock::now();
  while (MsSince(t0) < 1000) {
    ReadBatch(&client, "k", 1);
    std::this_thread::sleep_for(5ms);
  }
  const int reads = transport.Reads(ring.addr);
  std::printf("[ measured ] %d reads sent to a node with a failed store in "
              "1 s, probe every 20 ms (base %d ms)\n", reads, kBaseMs);
  // The first read, the trial the first probe answer brought forward, and
  // then the back-off: 1 s from there.
  EXPECT_LE(reads, 4) << reads << " reads walked into the same failure";
}

// The transport's hold on an address and the client's cooldown for the node
// need not start together: here the hold starts a quarter of a base period
// into the cooldown (a connect made by someone else failed then). The trial
// after the cooldown is refused by the hold without a dial. That says nothing
// new about the node: the cooldown is not doubled, and the first request
// after the hold is the one that dials.
TEST(PeerHealthStall, RefusalByTheTransportHoldIsNotAFailedTrial) {
  ScaledHealthEnv();
  ScriptedTransport transport;
  OneNodeRing ring(/*mds_managed=*/false, "stall/misaligned", &transport);
  ScriptedTransport::Node dead;
  dead.read = Status::kIOError;  // fails on an established connection: no hold
  transport.Set(ring.addr, dead);

  ReadBatch(ring.client.get(), "first", 1);            // cooldown: T .. T+base
  const auto cooled_at = Clock::now();
  std::this_thread::sleep_for(std::chrono::milliseconds(kBaseMs / 4));
  transport.HoldFor(ring.addr, std::chrono::milliseconds(2 * kBaseMs));
  // The hold ends at T + 2.25 base.
  std::this_thread::sleep_until(
      cooled_at + std::chrono::milliseconds(kBaseMs + kSlackMs));
  ReadBatch(ring.client.get(), "trial", 1);            // refused by the hold
  EXPECT_EQ(transport.Reads(ring.addr), 1);
  EXPECT_EQ(transport.HeldReads(ring.addr), 1);
  // The hold is over; a doubled cooldown would run until about T + 3.2 base.
  std::this_thread::sleep_until(
      cooled_at +
      std::chrono::milliseconds(2 * kBaseMs + kBaseMs / 4 + kSlackMs));
  ReadBatch(ring.client.get(), "after-hold", 1);
  EXPECT_EQ(transport.Reads(ring.addr), 2)
      << "the refusal was counted as a failed trial: the node is still "
         "avoided after the hold";
}

// The transport's note that it refused an operation describes that operation
// only. A refused membership query reports nothing to the node's health
// record; the next operation on the same thread dials the node and fails.
// That failure is a failure: the node is avoided after it.
TEST(PeerHealthStall, RefusalOfACallThatReportsNothingIsNotKeptForTheNext) {
  ScaledHealthEnv();
  ScriptedTransport transport;
  OneNodeRing ring(/*mds_managed=*/false, "stall/stale", &transport);
  ScriptedTransport::Node dead;
  dead.read = Status::kIOError;  // no hold: every read is a dial
  transport.Set(ring.addr, dead);

  transport.HoldFor(ring.addr, 50ms);
  EXPECT_FALSE(ring.client->RefreshMembers(ring.addr));
  EXPECT_EQ(transport.HeldReads(ring.addr), 1);
  std::this_thread::sleep_for(100ms);

  char buf[8];
  EXPECT_FALSE(ring.client->Get("first", buf, sizeof(buf)));  // dials, fails
  EXPECT_EQ(transport.Reads(ring.addr), 1);
  EXPECT_FALSE(ring.client->Get("next", buf, sizeof(buf)));
  EXPECT_EQ(transport.Reads(ring.addr), 1)
      << "the failed read was taken for the earlier refusal: the node was "
         "not avoided";
}

// A trial whose operation says nothing about the node (here: this process is
// out of transport budget) gives the trial back, so the next request tries at
// once instead of a base period later.
TEST(PeerHealthStall, TrialWithoutAVerdictGivesTheTrialBack) {
  ScaledHealthEnv();
  ScriptedTransport transport;
  OneNodeRing ring(/*mds_managed=*/false, "stall/neutral", &transport);
  ScriptedTransport::Node node;
  node.read = Status::kIOError;
  transport.Set(ring.addr, node);
  ReadBatch(ring.client.get(), "first", 1);
  std::this_thread::sleep_for(std::chrono::milliseconds(kBaseMs + kSlackMs));

  node.read = Status::kResourceExhausted;   // never reached the node
  transport.Set(ring.addr, node);
  ReadBatch(ring.client.get(), "trial", 1);
  EXPECT_EQ(transport.Reads(ring.addr), 2);
  ReadBatch(ring.client.get(), "next", 1);  // at once: the next trial
  EXPECT_EQ(transport.Reads(ring.addr), 3)
      << "the node stayed closed behind a trial that had no verdict";
  node.read = Status::kNotFound;            // the node is back and answers
  transport.Set(ring.addr, node);
  ReadBatch(ring.client.get(), "served", 1);
  const int before = transport.Reads(ring.addr);
  ReadBatch(ring.client.get(), "k", 128);   // open for everyone: 8 shards
  EXPECT_EQ(transport.Reads(ring.addr), before + 8);
}

// "Cache full" is an answer from the node. As the outcome of the trial it
// reopens the node, like a hit or a miss would.
TEST(PeerHealthStall, CacheFullAnswerToTheTrialReopensTheNode) {
  ScaledHealthEnv();
  ScriptedTransport transport;
  OneNodeRing ring(/*mds_managed=*/false, "stall/full-trial", &transport);
  ScriptedTransport::Node node;
  node.read = Status::kIOError;
  transport.Set(ring.addr, node);
  ReadBatch(ring.client.get(), "first", 1);
  std::this_thread::sleep_for(std::chrono::milliseconds(kBaseMs + kSlackMs));

  node.read = Status::kNotFound;
  node.put = Status::kCacheFull;            // back, and full
  transport.Set(ring.addr, node);
  const std::string value(8, 'v');
  std::vector<KvPutItem> put{{"p", value.data(), value.size()}};
  EXPECT_EQ(ring.client->BatchPut(put), std::vector<bool>({false}));
  const int before = transport.Reads(ring.addr);
  ReadBatch(ring.client.get(), "k", 128);
  EXPECT_EQ(transport.Reads(ring.addr), before + 8)
      << "reads were still skipped after the node had answered a write";
}

// A node that answers "cache full" to every write is answering. One failed
// read right after such an answer is a failure on a live node, not a reason
// to avoid it.
TEST(PeerHealthStall, CacheFullAnswersShowTheNodeIsAlive) {
  ScaledHealthEnv();
  ::setenv("DFKV_PEER_BAD_GRACE_MS", "300", 1);
  ScriptedTransport transport;
  OneNodeRing ring(/*mds_managed=*/false, "stall/full", &transport);
  ScriptedTransport::Node node;
  transport.Set(ring.addr, node);
  KVClient& client = *ring.client;
  ReadBatch(&client, "k", 1);  // served: the grace window starts
  node.put = Status::kCacheFull;
  transport.Set(ring.addr, node);
  const std::string value(8, 'v');
  const auto t0 = Clock::now();
  int answered = 0;
  while (MsSince(t0) < 600) {  // twice the grace, answered all the time
    std::vector<KvPutItem> put{{"p", value.data(), value.size()}};
    EXPECT_EQ(client.BatchPut(put), std::vector<bool>({false}));
    ++answered;
    std::this_thread::sleep_for(10ms);
  }
  node.read = Status::kIOError;
  transport.Set(ring.addr, node);
  ReadBatch(&client, "one-failure", 1);
  node.read = Status::kNotFound;
  transport.Set(ring.addr, node);
  const int before = transport.Reads(ring.addr);
  ReadBatch(&client, "next", 1);
  EXPECT_EQ(transport.Reads(ring.addr), before + 1)
      << "after " << answered << " cache-full answers in 600 ms, the last one "
         "about 10 ms earlier, one failed read made the client avoid the node";
}

// A node that is alive but slow: every shard takes 600 ms and succeeds. Its
// batch may not hold every read worker of the process: one stays free, and a
// read of another node is served at once.
TEST(PeerHealthStall, SlowNodeLeavesAWorkerForOtherNodes) {
  ScaledHealthEnv();
  ScriptedTransport transport;
  OneNodeRing x(/*mds_managed=*/false, "stall/slow", &transport, 29001);
  OneNodeRing y(/*mds_managed=*/false, "stall/slow", &transport, 29002);
  ScriptedTransport::Node slow;
  slow.delay = 600ms;
  transport.Set(x.addr, slow);
  transport.Set(y.addr, ScriptedTransport::Node{});

  // Nothing else is being read when the slow node's batch starts.
  std::thread batch([&] { ReadBatch(x.client.get(), "k", 128); });
  if (!transport.WaitInflight(x.addr, 6, 2s)) {
    batch.join();
    FAIL() << "the slow node never had six shards in flight";
  }
  const auto t0 = Clock::now();
  ReadBatch(y.client.get(), "k", 16);
  const int64_t other_ms = MsSince(t0);
  batch.join();

  const int concurrent = transport.MaxConcurrentReads(x.addr);
  // Printed for the record; ctest shows it with --verbose or on failure.
  std::printf("[ measured ] read of another node took %lld ms beside a slow "
              "node; %d shards inside the slow node at once\n",
              static_cast<long long>(other_ms), concurrent);
  EXPECT_EQ(concurrent, 6) << "of the seven read workers";
  EXPECT_LT(other_ms, 200) << "a read of another node waited " << other_ms
                           << " ms behind a slow node";
}
