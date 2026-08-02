// Client metrics and public-operation accounting contracts.
#include "client/key_map.h"
#include "client/kv_client.h"
#include "client/peer_health.h"

#include <gtest/gtest.h>
#include <algorithm>

#include <string>
#include <chrono>
#include <cstdlib>
#include <map>
#include <mutex>
#include <sys/mman.h>
#include <thread>
#include <vector>

using namespace dfkv;  // NOLINT

namespace {
class MetricsTransport final : public Transport {
 public:
  bool pipeline = false;
  std::mutex mu;
  std::map<std::string, std::string> values;
  std::vector<Status> exist_script;
  size_t exist_calls = 0;
  size_t range_calls = 0;

  Status Cache(const std::string&, const BlockKey& key, const void* data,
               size_t len) override {
    std::lock_guard<std::mutex> lock(mu);
    values[key.Filename()] =
        std::string(static_cast<const char*>(data), len);
    return Status::kOk;
  }
  Status Range(const std::string&, const BlockKey& key, uint64_t offset,
               uint64_t length, std::string* out,
               uint64_t* value_len) override {
    std::lock_guard<std::mutex> lock(mu);
    ++range_calls;
    auto found = values.find(key.Filename());
    if (found == values.end()) return Status::kNotFound;
    if (value_len) *value_len = found->second.size();
    const size_t begin =
        std::min<size_t>(static_cast<size_t>(offset), found->second.size());
    *out = found->second.substr(begin, static_cast<size_t>(length));
    return Status::kOk;
  }
  Status Lookup(const std::string&, const BlockKey& key,
                uint64_t* value_len) override {
    std::lock_guard<std::mutex> lock(mu);
    auto found = values.find(key.Filename());
    if (found == values.end()) return Status::kNotFound;
    if (value_len) *value_len = found->second.size();
    return Status::kOk;
  }
  Status Exist(const std::string&, const BlockKey& key, bool* exists) override {
    std::lock_guard<std::mutex> lock(mu);
    ++exist_calls;
    if (!exist_script.empty()) {
      const Status scripted = exist_script.front();
      exist_script.erase(exist_script.begin());
      *exists = false;
      return scripted;
    }
    *exists = values.count(key.Filename()) != 0;
    return *exists ? Status::kOk : Status::kNotFound;
  }
  Status Remove(const std::string&, const BlockKey& key) override {
    std::lock_guard<std::mutex> lock(mu);
    values.erase(key.Filename());
    return Status::kOk;
  }
  bool pipelined() const override { return pipeline; }
};

uint64_t Metric(const std::string& snapshot, const std::string& family,
                const std::string& op) {
  const std::string prefix =
      family + "{op=\"" + op + "\"} ";
  const size_t start = snapshot.find(prefix);
  EXPECT_NE(start, std::string::npos) << snapshot;
  if (start == std::string::npos) return 0;
  return std::stoull(snapshot.substr(start + prefix.size()));
}

struct DedupEnv {
  std::string name;
  explicit DedupEnv(const std::string& key_namespace) {
    ::setenv("DFKV_CLIENT_NODE_DEDUP", "1", 1);
    ::setenv("DFKV_NODE_DEDUP_ARENA_MB", "1", 1);
    ::setenv("DFKV_NODE_DEDUP_SLOTS", "1024", 1);
    ::setenv("DFKV_NODE_DEDUP_WAIT_MS", "10", 1);
    ::setenv("DFKV_PEER_COOLDOWN_MS", "1", 1);
    name = NodeDedup::EnvSegmentName(
        ToBlockKey(key_namespace, "").digest_hi);
    ::shm_unlink(name.c_str());
  }
  ~DedupEnv() {
    ::shm_unlink(name.c_str());
    ::unsetenv("DFKV_CLIENT_NODE_DEDUP");
    ::unsetenv("DFKV_NODE_DEDUP_ARENA_MB");
    ::unsetenv("DFKV_NODE_DEDUP_SLOTS");
    ::unsetenv("DFKV_NODE_DEDUP_WAIT_MS");
    ::unsetenv("DFKV_PEER_COOLDOWN_MS");
  }
};
}  // namespace

TEST(ClientMetrics, CountsServedErrorsTransitionsAndRenders) {
  PeerHealth h(/*cooldown_ms=*/1000);
  const std::string p = "10.0.0.1:12000";
  // a healthy check, then a served response
  EXPECT_TRUE(h.Healthy(p, 1000));
  h.MarkGood(p, 0);
  // an IO error marks it bad (healthy->bad edge), then a skip while in cooldown
  h.MarkBad(p, 1000);
  EXPECT_FALSE(h.Healthy(p, 1500));   // still in cooldown -> skip
  // recovery: a later served response clears it (bad->good edge)
  h.MarkGood(p, 0);

  EXPECT_EQ(h.served(), 2u);
  EXPECT_EQ(h.errors(), 1u);
  EXPECT_EQ(h.marked_bad(), 1u);
  EXPECT_EQ(h.recovered(), 1u);

  std::string t = h.Render();
  EXPECT_NE(t.find("dfkv_client_ops_served_total 2"), std::string::npos) << t;
  EXPECT_NE(t.find("dfkv_client_io_errors_total 1"), std::string::npos) << t;
  EXPECT_NE(t.find("dfkv_client_unhealthy_skips_total 1"), std::string::npos) << t;
  EXPECT_NE(t.find("dfkv_client_peer_marked_bad_total 1"), std::string::npos) << t;
  EXPECT_NE(t.find("dfkv_client_peer_recovered_total 1"), std::string::npos) << t;
  EXPECT_NE(t.find("dfkv_client_peer_errors_total{peer=\"10.0.0.1:12000\"} 1"), std::string::npos) << t;
}

TEST(ClientMetrics, PerPeerMapIsCardinalityBounded) {
  // A client that churns through many distinct peer addresses must not grow the
  // per-peer error map (and its scrape cardinality) without bound.
  PeerHealth h(1000);
  for (int i = 0; i < 6000; ++i) h.MarkBad("10.0.0." + std::to_string(i) + ":1", 1000);
  std::string t = h.Render();
  size_t n = 0, pos = 0;
  const std::string needle = "dfkv_client_peer_errors_total{peer=";
  while ((pos = t.find(needle, pos)) != std::string::npos) { ++n; pos += needle.size(); }
  EXPECT_LE(n, 4096u) << "per-peer series cardinality not bounded: " << n;
  EXPECT_EQ(h.errors(), 6000u);  // aggregate still counts every error
}

TEST(ClientOpMetrics, ScalarCallsCountOnceIncludingEarlyFailure) {
  MetricsTransport transport;
  KVClient client({{"n", "test:1"}}, "metrics/scalar", &transport);
  const std::string value = "abc";
  char out[3] = {};
  size_t got = 0;
  EXPECT_TRUE(client.Put("k", value.data(), value.size()));
  EXPECT_TRUE(client.Get("k", out, sizeof(out)));
  EXPECT_TRUE(client.GetAuto("k", out, sizeof(out), &got));
  EXPECT_TRUE(client.Exist("k"));
  EXPECT_TRUE(client.Remove("k"));
  EXPECT_FALSE(client.Get("missing", out, sizeof(out)));

  const std::string snapshot = client.MetricsSnapshot();
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_requests_total", "put"), 1u);
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_requests_total", "get"), 3u);
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_requests_total", "exist"), 1u);
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_requests_total", "remove"), 1u);
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_keys_total", "get"), 3u);
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_hits_total", "get"), 2u);
}

TEST(ClientOpMetrics, TcpAndPipelinedBatchesCountCallsNotFanoutKeys) {
  for (bool pipeline : {false, true}) {
    MetricsTransport transport;
    transport.pipeline = pipeline;
    KVClient client({{"n", "test:1"}}, pipeline ? "metrics/rdma"
                                                : "metrics/tcp",
                    &transport);
    std::vector<std::string> values = {"aa", "bb", "cc"};
    std::vector<KvPutItem> puts;
    std::vector<KvGetItem> gets;
    std::vector<std::string> outputs(3, std::string(2, '\0'));
    std::vector<std::string> keys;
    for (size_t i = 0; i < 3; ++i) {
      keys.push_back("k" + std::to_string(i));
      puts.push_back({keys.back(), values[i].data(), values[i].size()});
    }
    EXPECT_EQ(client.BatchPut(puts), std::vector<bool>({true, true, true}));
    for (size_t i = 0; i < 3; ++i)
      gets.push_back({keys[i], outputs[i].data(), outputs[i].size()});
    EXPECT_EQ(client.BatchGet(gets), std::vector<bool>({true, true, true}));
    EXPECT_EQ(client.BatchExist(keys), std::vector<bool>({true, true, true}));
    EXPECT_EQ(client.BatchRemove(keys), std::vector<bool>({true, true, true}));

    const std::string snapshot = client.MetricsSnapshot();
    for (const char* op : {"put", "get", "exist", "remove"}) {
      EXPECT_EQ(Metric(snapshot, "dfkv_client_op_requests_total", op), 1u)
          << op << " pipeline=" << pipeline;
      EXPECT_EQ(Metric(snapshot, "dfkv_client_op_keys_total", op), 3u)
          << op << " pipeline=" << pipeline;
      EXPECT_EQ(Metric(snapshot, "dfkv_client_op_hits_total", op), 3u)
          << op << " pipeline=" << pipeline;
    }
  }
}

TEST(ClientOpMetrics, RendezvousHitsAreCountedAndRemoveCannotServeStale) {
  const std::string key_namespace = "metrics/dedup";
  DedupEnv env(key_namespace);
  MetricsTransport transport;
  KVClient client({{"n", "test:1"}}, key_namespace, &transport);
  const std::string value = "payload";
  ASSERT_TRUE(client.Put("k", value.data(), value.size()));

  std::string first(value.size(), '\0'), second(value.size(), '\0');
  std::vector<KvGetItem> get1{{"k", first.data(), first.size()}};
  std::vector<KvGetItem> get2{{"k", second.data(), second.size()}};
  EXPECT_EQ(client.BatchGet(get1), std::vector<bool>({true}));
  EXPECT_EQ(client.BatchGet(get2), std::vector<bool>({true}));
  EXPECT_EQ(transport.range_calls, 1u);  // second call was served from shm
  EXPECT_TRUE(client.Remove("k"));
  EXPECT_EQ(client.BatchGet(get2), std::vector<bool>({false}));
  EXPECT_EQ(transport.range_calls, 2u);  // remove invalidated the READY entry

  const std::string snapshot = client.MetricsSnapshot();
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_requests_total", "get"), 3u);
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_keys_total", "get"), 3u);
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_hits_total", "get"), 2u);
}

TEST(ClientDedup, TransientExistFailureIsNotPublishedAsAbsence) {
  const std::string key_namespace = "metrics/exist-transient";
  DedupEnv env(key_namespace);
  MetricsTransport transport;
  transport.exist_script = {Status::kIOError, Status::kNotFound};
  KVClient client({{"n", "test:1"}}, key_namespace, &transport);
  const std::vector<std::string> keys{"missing"};

  EXPECT_EQ(client.BatchExist(keys), std::vector<bool>({false}));
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  EXPECT_EQ(client.BatchExist(keys), std::vector<bool>({false}));
  EXPECT_EQ(transport.exist_calls, 2u);
  // The second authoritative NotFound is publishable; the third call hits shm.
  EXPECT_EQ(client.BatchExist(keys), std::vector<bool>({false}));
  EXPECT_EQ(transport.exist_calls, 2u);
}

TEST(ClientRetries, ExplicitZeroDisablesBoundedMissRetry) {
  ::setenv("DFKV_GET_MISS_RETRIES", "0", 1);
  MetricsTransport transport;
  transport.pipeline = true;
  KVClient client({{"n", "test:1"}}, "retry/zero", &transport);
  ::unsetenv("DFKV_GET_MISS_RETRIES");
  const std::string value = "aa";
  ASSERT_TRUE(client.Put("hit", value.data(), value.size()));
  transport.range_calls = 0;
  std::string hit(2, '\0'), miss(2, '\0');
  std::vector<KvGetItem> gets{{"hit", hit.data(), hit.size()},
                              {"miss", miss.data(), miss.size()}};
  EXPECT_EQ(client.BatchGet(gets), std::vector<bool>({true, false}));
  EXPECT_EQ(transport.range_calls, 2u);
}

TEST(ClientRetries, InvalidOrOutOfRangeValueUsesBoundedDefault) {
  for (const char* invalid : {"-1", " 0", "2147483648", "1x"}) {
    ::setenv("DFKV_GET_MISS_RETRIES", invalid, 1);
    MetricsTransport transport;
    transport.pipeline = true;
    KVClient client({{"n", "test:1"}},
                    std::string("retry/invalid/") + invalid, &transport);
    ::unsetenv("DFKV_GET_MISS_RETRIES");
    const std::string value = "aa";
    ASSERT_TRUE(client.Put("hit", value.data(), value.size()));
    transport.range_calls = 0;
    std::string hit(2, '\0'), miss(2, '\0');
    std::vector<KvGetItem> gets{{"hit", hit.data(), hit.size()},
                                {"miss", miss.data(), miss.size()}};
    EXPECT_EQ(client.BatchGet(gets), std::vector<bool>({true, false}));
    EXPECT_EQ(transport.range_calls, 3u) << invalid;
  }
}
