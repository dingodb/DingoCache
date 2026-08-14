// Client metrics and public-operation accounting contracts.
#include "client/key_map.h"
#include "client/kv_client.h"
#include "client/peer_health.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>

#include <fstream>
#include <limits>
#include <string>
#include <chrono>
#include <cstdlib>
#include <map>
#include <mutex>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>
#include <type_traits>
#include <vector>
#include <iterator>


namespace dfkv::testing {
void DrainPernodeSink();
size_t PernodeQueueCapacity();
size_t PernodeQueueHighWater();
uint64_t PernodeQueueDropped();
uint64_t PernodeQueuePendingDropped();
void PausePernodeSink(bool paused);
void EnqueuePernodeForTest(const char* fname,
                           std::vector<std::string> lines);
std::string PernodeKeyFingerprintForTest(const std::string& key);
}  // namespace dfkv::testing

using namespace dfkv;  // NOLINT

struct PernodeMetricsEnv {
  std::string dir =
      "/tmp/dfkv-pernode-metrics-" + std::to_string(::getpid());
  std::string suffix =
      "dp0_pp0_tp0_g0_p" + std::to_string(::getpid());

  PernodeMetricsEnv() {
    ::mkdir(dir.c_str(), 0700);
    ::setenv("DFKV_CLIENT_PERNODE_STATS", "1", 1);
    ::setenv("DFKV_ACCESS_DIR", dir.c_str(), 1);
    ::setenv("DFKV_CLIENT_LOG_SUFFIX", suffix.c_str(), 1);
    ::setenv("DFKV_CLIENT_PERNODE_QUEUE_CAPACITY", "4", 1);
    ::setenv("DFKV_CLIENT_PERNODE_ROTATE_BYTES", "512", 1);
  }
  std::string Path(const std::string& file) const {
    return dir + "/" + file + "." + suffix;
  }
};

PernodeMetricsEnv pernode_env;

std::string ReadText(const std::string& path) {
  std::ifstream input(path);
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

std::string CurrentAndPrevious(const std::string& file) {
  const std::string current = pernode_env.Path(file);
  return ReadText(current + ".1") + ReadText(current);
}

bool HasLineWith(const std::string& text,
                 std::initializer_list<std::string> fields) {
  size_t begin = 0;
  while (begin <= text.size()) {
    const size_t end = text.find('\n', begin);
    const std::string line =
        text.substr(begin, end == std::string::npos ? std::string::npos
                                                    : end - begin);
    bool all = true;
    for (const std::string& field : fields)
      all = all && line.find(field) != std::string::npos;
    if (all) return true;
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return false;
}
namespace {
class MetricsTransport final : public Transport {
 public:
  enum class AttemptResult : uint8_t {
    kSuccess,
    kRailFailure,
    kAmbiguousPostSubmitFailure,
  };
  struct ScriptedAttempt {
    size_t rail;
    AttemptResult result;
    std::chrono::milliseconds delay{0};
    size_t completed_items_before_failure = 0;
  };

  bool pipeline = false;
  mutable std::mutex mu;
  std::map<std::string, std::string> values;
  std::vector<Status> exist_script;
  std::vector<ScriptedAttempt> attempt_script;
  std::vector<ScriptedAttempt> attempts;
  std::vector<size_t> replay_item_counts;
  std::vector<std::vector<size_t>> attempt_item_indices;
  std::vector<size_t> range_multi_item_calls;
  size_t remote_put_commits = 0;
  size_t exist_calls = 0;
  size_t range_calls = 0;
  size_t range_multi_calls = 0;
  size_t range_multi_misses = 0;
  size_t cache_status_limit = std::numeric_limits<size_t>::max();
  size_t range_status_limit = std::numeric_limits<size_t>::max();
  size_t exist_status_limit = std::numeric_limits<size_t>::max();
  uint64_t cross_rail_retries = 0;
  uint64_t cross_rail_retry_successes = 0;
  uint64_t cross_rail_retry_exhausted = 0;

  void Script(std::initializer_list<ScriptedAttempt> script) {
    std::lock_guard<std::mutex> lock(mu);
    attempt_script.assign(script);
    attempts.clear();
    replay_item_counts.clear();
    attempt_item_indices.clear();
    range_multi_item_calls.clear();
  }

  Status Cache(const std::string&, const BlockKey& key, const void* data,
               size_t len) override {
    std::lock_guard<std::mutex> lock(mu);
    const bool ok = RunAttemptsLocked(1, [&] {
      ++remote_put_commits;
      values[key.Filename()] =
          std::string(static_cast<const char*>(data), len);
    });
    return ok ? Status::kOk : Status::kIOError;
  }
  Status Range(const std::string&, const BlockKey& key, uint64_t offset,
               uint64_t length, std::string* out,
               uint64_t* value_len) override {
    std::lock_guard<std::mutex> lock(mu);
    ++range_calls;
    auto found = values.find(key.Filename());
    if (found == values.end()) return Status::kNotFound;
    const bool ok = RunAttemptsLocked(1, [&] {
      if (value_len) *value_len = found->second.size();
      const size_t begin =
          std::min<size_t>(static_cast<size_t>(offset), found->second.size());
      *out = found->second.substr(begin, static_cast<size_t>(length));
    });
    return ok ? Status::kOk : Status::kIOError;
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

  std::vector<Status> CacheFrom(
      const std::string&, const std::vector<CacheSrc>& srcs) override {
    std::lock_guard<std::mutex> lock(mu);
    const bool ok = RunAttemptsLocked(srcs.size(), [&] {
      remote_put_commits += srcs.size();
      for (const auto& src : srcs) {
        values[src.key.Filename()] =
            std::string(static_cast<const char*>(src.payload),
                        src.payload_len);
      }
    });
    return std::vector<Status>(
        srcs.size(), ok ? Status::kOk : Status::kIOError);
  }

  std::vector<Status> RangeInto(
      const std::string&, const std::vector<BlockKey>& keys,
      const std::vector<RangeDst>& dsts,
      std::vector<uint64_t>* value_lens) override {
    std::lock_guard<std::mutex> lock(mu);
    range_calls += keys.size();
    if (value_lens) value_lens->assign(keys.size(), 0);
    if (dsts.size() != keys.size()) return InvalidStatuses(keys.size());
    std::vector<Status> result(keys.size(), Status::kNotFound);
    const bool ok = RunAttemptsLocked(keys.size(), [&] {
      for (size_t i = 0; i < keys.size(); ++i) {
        auto found = values.find(keys[i].Filename());
        if (found == values.end()) continue;
        const size_t copied = std::min(dsts[i].n, found->second.size());
        if (copied) std::memcpy(dsts[i].payload, found->second.data(), copied);
        if (value_lens) (*value_lens)[i] = found->second.size();
        result[i] = Status::kOk;
      }
    });
    if (!ok) result.assign(keys.size(), Status::kIOError);
    return result;
  }

  std::vector<Status> CacheFromMulti(
      const std::string&, const std::vector<CacheSrcMulti>& srcs,
      std::string* /*out_dev*/ = nullptr) override {
    std::lock_guard<std::mutex> lock(mu);
    const bool ok = RunAttemptsLocked(srcs.size(), [&] {
      remote_put_commits += srcs.size();
      for (const auto& src : srcs) {
        std::string value;
        for (const auto& segment : src.payloads)
          value.append(static_cast<const char*>(segment.first),
                       segment.second);
        values[src.key.Filename()] = std::move(value);
      }
    });
    std::vector<Status> result(
        srcs.size(), ok ? Status::kOk : Status::kIOError);
    result.resize(std::min(result.size(), cache_status_limit));
    return result;
  }

  std::vector<Status> RangeIntoMulti(
      const std::string&, const std::vector<BlockKey>& keys,
      const std::vector<RangeDstMulti>& dsts,
      std::vector<size_t>* out_lens,
      std::string* /*out_dev*/ = nullptr) override {
    std::lock_guard<std::mutex> lock(mu);
    ++range_multi_calls;
    if (out_lens) out_lens->assign(keys.size(), 0);
    if (dsts.size() != keys.size()) return InvalidStatuses(keys.size());
    std::vector<Status> result(keys.size(), Status::kNotFound);
    range_multi_item_calls.assign(keys.size(), 0);
    const bool ok = RunAttemptsLocked(
        keys.size(), [&](size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
        ++range_multi_item_calls[i];
        auto found = values.find(keys[i].Filename());
        if (found == values.end()) continue;
        size_t offset = 0;
        for (const auto& segment : dsts[i].payloads) {
          const size_t copy =
              std::min(segment.second, found->second.size() - offset);
          if (copy)
            std::memcpy(segment.first, found->second.data() + offset, copy);
          offset += copy;
          if (offset == found->second.size()) break;
        }
        if (out_lens) (*out_lens)[i] = found->second.size();
        result[i] = Status::kOk;
      }
    });
    if (!ok) {
      result.assign(keys.size(), Status::kIOError);
    } else {
      for (size_t i = 0; i < result.size() && range_multi_misses > 0; ++i) {
        if (result[i] != Status::kOk) continue;
        result[i] = Status::kNotFound;
        if (out_lens) (*out_lens)[i] = 0;
        --range_multi_misses;
      }
    }
    result.resize(std::min(result.size(), range_status_limit));
    return result;
  }

  std::vector<Status> ExistMany(
      const std::string& node, const std::vector<BlockKey>& keys,
      std::vector<char>* exists, std::string* out_dev = nullptr) override {
    std::vector<Status> result =
        Transport::ExistMany(node, keys, exists, out_dev);
    result.resize(std::min(result.size(), exist_status_limit));
    return result;
  }

  std::string MetricsText() const override {
    std::lock_guard<std::mutex> lock(mu);
    return
        "# TYPE dfkv_rdma_client_cross_rail_retries_total counter\n"
        "dfkv_rdma_client_cross_rail_retries_total " +
        std::to_string(cross_rail_retries) +
        "\n# TYPE dfkv_rdma_client_cross_rail_retry_successes_total counter\n"
        "dfkv_rdma_client_cross_rail_retry_successes_total " +
        std::to_string(cross_rail_retry_successes) +
        "\n# TYPE dfkv_rdma_client_cross_rail_retry_exhausted_total counter\n"
        "dfkv_rdma_client_cross_rail_retry_exhausted_total " +
        std::to_string(cross_rail_retry_exhausted) + "\n";
  }

 private:
  template <typename Success>
  bool RunAttemptsLocked(size_t item_count, Success&& success) {
    const std::vector<ScriptedAttempt> script =
        attempt_script.empty()
            ? std::vector<ScriptedAttempt>{{0, AttemptResult::kSuccess}}
            : attempt_script;
    size_t completed_items = 0;
    bool cross_rail_retry = false;
    for (size_t i = 0; i < script.size() && i < 2; ++i) {
      if (script[i].delay.count() != 0)
        std::this_thread::sleep_for(script[i].delay);
      attempts.push_back(script[i]);
      replay_item_counts.push_back(item_count - completed_items);
      std::vector<size_t> attempted_items;
      attempted_items.reserve(item_count - completed_items);
      for (size_t item = completed_items; item < item_count; ++item)
        attempted_items.push_back(item);
      attempt_item_indices.push_back(std::move(attempted_items));
      if (script[i].result == AttemptResult::kSuccess) {
        if constexpr (std::is_invocable_v<Success, size_t, size_t>)
          success(completed_items, item_count);
        else
          success();
        if (cross_rail_retry) ++cross_rail_retry_successes;
        return true;
      }
      if (script[i].result == AttemptResult::kAmbiguousPostSubmitFailure) {
        // The peer may already have committed this PUT. The production
        // contract returns the ambiguous failure without replaying it on a
        // different rail.
        if constexpr (std::is_invocable_v<Success, size_t, size_t>)
          success(completed_items, item_count);
        else
          success();
        return false;
      }
      const size_t newly_completed =
          std::min(item_count - completed_items,
                   script[i].completed_items_before_failure);
      if constexpr (std::is_invocable_v<Success, size_t, size_t>) {
        if (newly_completed != 0)
          success(completed_items, completed_items + newly_completed);
      }
      completed_items += newly_completed;
      if (i == 0 && script.size() > 1) {
        cross_rail_retry = script[1].rail != script[0].rail;
        if (cross_rail_retry) ++cross_rail_retries;
        continue;
      }
      if (cross_rail_retry) ++cross_rail_retry_exhausted;
      return false;
    }
    if (cross_rail_retry) ++cross_rail_retry_exhausted;
    return false;
  }
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
uint64_t TransportCounter(const std::string& snapshot,
                          const std::string& family) {
  const std::string prefix = family + " ";
  size_t start = snapshot.find(prefix);
  while (start != std::string::npos &&
         start != 0 && snapshot[start - 1] != '\n') {
    start = snapshot.find(prefix, start + 1);
  }
  EXPECT_NE(start, std::string::npos) << snapshot;
  if (start == std::string::npos) return 0;
  EXPECT_EQ(snapshot.find(family + "{"), std::string::npos)
      << family << " must remain an unlabeled process counter";
  return std::stoull(snapshot.substr(start + prefix.size()));
}
double MetricDouble(const std::string& snapshot, const std::string& family,
                    const std::string& op) {
  const std::string prefix =
      family + "{op=\"" + op + "\"} ";
  const size_t start = snapshot.find(prefix);
  EXPECT_NE(start, std::string::npos) << snapshot;
  if (start == std::string::npos) return 0;
  return std::stod(snapshot.substr(start + prefix.size()));
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
TEST(ClientOpMetrics, CrossRailRetryCountsOneScalarPublicCall) {
  using Result = MetricsTransport::AttemptResult;
  MetricsTransport transport;
  transport.pipeline = true;
  KVClient client({{"n", "test:1"}}, "metrics/cross-rail-scalar",
                  &transport);
  const std::string value = "rail-retry-payload";
  const auto failed_attempt_delay = std::chrono::milliseconds(35);
  const auto successful_attempt_delay = std::chrono::milliseconds(5);

  transport.Script(
      {{0, Result::kRailFailure, failed_attempt_delay},
       {1, Result::kSuccess, successful_attempt_delay}});
  ASSERT_TRUE(client.Put("key", value.data(), value.size()));
  ASSERT_EQ(transport.attempts.size(), 2u);
  EXPECT_EQ(transport.attempts[0].rail, 0u);
  EXPECT_EQ(transport.attempts[1].rail, 1u);
  EXPECT_EQ(transport.replay_item_counts, (std::vector<size_t>{1, 1}));

  transport.Script(
      {{0, Result::kRailFailure, failed_attempt_delay},
       {1, Result::kSuccess, successful_attempt_delay}});
  std::string output(value.size(), '\0');
  ASSERT_TRUE(client.Get("key", output.data(), output.size()));
  EXPECT_EQ(output, value);
  ASSERT_EQ(transport.attempts.size(), 2u);
  EXPECT_EQ(transport.attempts[0].rail, 0u);
  EXPECT_EQ(transport.attempts[1].rail, 1u);
  EXPECT_EQ(transport.replay_item_counts, (std::vector<size_t>{1, 1}));

  const std::string snapshot = client.MetricsSnapshot();
  for (const char* op : {"put", "get"}) {
    EXPECT_EQ(Metric(snapshot, "dfkv_client_op_requests_total", op), 1u);
    EXPECT_EQ(Metric(snapshot, "dfkv_client_op_keys_total", op), 1u);
    EXPECT_EQ(Metric(snapshot, "dfkv_client_op_hits_total", op), 1u);
    EXPECT_EQ(Metric(snapshot, "dfkv_client_op_bytes_total", op),
              value.size());
    EXPECT_EQ(Metric(snapshot, "dfkv_client_op_latency_seconds_count", op),
              1u);
    EXPECT_GE(MetricDouble(
                  snapshot, "dfkv_client_op_latency_seconds_sum", op),
              std::chrono::duration<double>(failed_attempt_delay +
                                            successful_attempt_delay)
                      .count() *
                  0.9)
        << "one public latency sample must enclose both transport attempts";
    EXPECT_LT(MetricDouble(
                  snapshot, "dfkv_client_op_latency_seconds_sum", op),
              2.0)
        << "scripted retry latency unexpectedly exceeded the deadlock guard";
  }
  EXPECT_EQ(TransportCounter(
                snapshot, "dfkv_rdma_client_cross_rail_retries_total"),
            2u);
  EXPECT_EQ(TransportCounter(
                snapshot,
                "dfkv_rdma_client_cross_rail_retry_successes_total"),
            2u);
  EXPECT_EQ(TransportCounter(
                snapshot,
                "dfkv_rdma_client_cross_rail_retry_exhausted_total"),
            0u);
}

TEST(ClientOpMetrics, SameRailFallbackDoesNotCountAsCrossRailRetry) {
  using Result = MetricsTransport::AttemptResult;
  MetricsTransport transport;
  transport.pipeline = true;
  KVClient client({{"n", "test:1"}}, "metrics/same-rail-fallback",
                  &transport);
  const std::string value = "same-rail-payload";

  transport.Script(
      {{0, Result::kRailFailure}, {0, Result::kSuccess}});
  ASSERT_TRUE(client.Put("key", value.data(), value.size()));
  ASSERT_EQ(transport.attempts.size(), 2u);
  EXPECT_EQ(transport.attempts[0].rail, 0u);
  EXPECT_EQ(transport.attempts[1].rail, 0u);

  const std::string snapshot = client.MetricsSnapshot();
  EXPECT_EQ(TransportCounter(
                snapshot, "dfkv_rdma_client_cross_rail_retries_total"),
            0u);
  EXPECT_EQ(TransportCounter(
                snapshot,
                "dfkv_rdma_client_cross_rail_retry_successes_total"),
            0u);
  EXPECT_EQ(TransportCounter(
                snapshot,
                "dfkv_rdma_client_cross_rail_retry_exhausted_total"),
            0u);
}

TEST(ClientOpMetrics, AmbiguousPostSubmitPutIsNotCrossRailReplayed) {
  using Result = MetricsTransport::AttemptResult;
  MetricsTransport transport;
  transport.pipeline = true;
  KVClient client({{"n", "test:1"}}, "metrics/ambiguous-put", &transport);
  const std::string value("committed\0payload", 17);

  transport.Script(
      {{0, Result::kAmbiguousPostSubmitFailure},
       {1, Result::kSuccess}});
  EXPECT_FALSE(client.Put("key", value.data(), value.size()));
  ASSERT_EQ(transport.attempts.size(), 1u);
  EXPECT_EQ(transport.attempts[0].rail, 0u);
  EXPECT_EQ(transport.remote_put_commits, 1u)
      << "the ambiguous first attempt may already have committed exactly once";

  const std::string snapshot = client.MetricsSnapshot();
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_requests_total", "put"), 1u);
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_hits_total", "put"), 0u);
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_bytes_total", "put"), 0u);
  EXPECT_EQ(TransportCounter(
                snapshot, "dfkv_rdma_client_cross_rail_retries_total"),
            0u);
  EXPECT_EQ(TransportCounter(
                snapshot,
                "dfkv_rdma_client_cross_rail_retry_successes_total"),
            0u);
  EXPECT_EQ(TransportCounter(
                snapshot,
                "dfkv_rdma_client_cross_rail_retry_exhausted_total"),
            0u);
}

TEST(ClientOpMetrics, AmbiguousPostSubmitPipelinedBatchPutIsNotReplayed) {
  using Result = MetricsTransport::AttemptResult;
  MetricsTransport transport;
  transport.pipeline = true;
  KVClient client({{"n", "test:1"}}, "metrics/ambiguous-batch-put",
                  &transport);
  const std::vector<std::string> values{
      std::string("first\0value", 11),
      std::string("second\0value", 12),
      std::string("third\0value", 11),
  };
  std::vector<KvPutItem> puts;
  for (size_t i = 0; i < values.size(); ++i) {
    puts.push_back({"key-" + std::to_string(i), values[i].data(),
                    values[i].size()});
  }

  transport.Script(
      {{0, Result::kAmbiguousPostSubmitFailure},
       {1, Result::kSuccess}});
  EXPECT_EQ(client.BatchPut(puts), std::vector<bool>(puts.size(), false));
  ASSERT_EQ(transport.attempts.size(), 1u);
  EXPECT_EQ(transport.attempts[0].rail, 0u);
  EXPECT_EQ(transport.replay_item_counts,
            std::vector<size_t>({puts.size()}));
  EXPECT_EQ(transport.remote_put_commits, puts.size());
  for (size_t i = 0; i < puts.size(); ++i) {
    EXPECT_EQ(transport.values[ToBlockKey(
                                   "metrics/ambiguous-batch-put", puts[i].key)
                                   .Filename()],
              values[i]);
  }

  const std::string snapshot = client.MetricsSnapshot();
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_requests_total", "put"), 1u);
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_keys_total", "put"),
            puts.size());
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_hits_total", "put"), 0u);
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_bytes_total", "put"), 0u);
  EXPECT_EQ(TransportCounter(
                snapshot, "dfkv_rdma_client_cross_rail_retries_total"),
            0u);
}

TEST(ClientOpMetrics, AmbiguousPostSubmitSgBatchPutIsNotReplayed) {
  using Result = MetricsTransport::AttemptResult;
  MetricsTransport transport;
  transport.pipeline = true;
  KVClient client({{"n", "test:1"}}, "metrics/ambiguous-sg-put",
                  &transport);
  const std::array<std::string, 3> segments{
      std::string("first\0", 6),
      std::string("second\0", 7),
      std::string("third\0", 6),
  };
  std::vector<KvPutItemSg> puts{
      {"sg-key",
       {segments[0].data(), segments[1].data(), segments[2].data()},
       {segments[0].size(), segments[1].size(), segments[2].size()}},
  };

  transport.Script(
      {{0, Result::kAmbiguousPostSubmitFailure},
       {1, Result::kSuccess}});
  EXPECT_EQ(client.BatchPutSg(puts), std::vector<bool>({false}));
  ASSERT_EQ(transport.attempts.size(), 1u);
  EXPECT_EQ(transport.attempts[0].rail, 0u);
  EXPECT_EQ(transport.replay_item_counts, std::vector<size_t>({1}));
  EXPECT_EQ(transport.remote_put_commits, 1u);
  EXPECT_EQ(transport.values[ToBlockKey("metrics/ambiguous-sg-put", "sg-key")
                                 .Filename()],
            segments[0] + segments[1] + segments[2]);

  const std::string snapshot = client.MetricsSnapshot();
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_requests_total", "put"), 1u);
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_keys_total", "put"), 1u);
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_hits_total", "put"), 0u);
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_bytes_total", "put"), 0u);
  EXPECT_EQ(TransportCounter(
                snapshot, "dfkv_rdma_client_cross_rail_retries_total"),
            0u);
}

TEST(ClientOpMetrics,
     ByteExactGetUsesAtMostTwoAttemptsAndDoesNotReplayCompletedItems) {
  using Result = MetricsTransport::AttemptResult;
  constexpr size_t kItems = 9;  // 2 * depth(4) + 1
  constexpr size_t kSegmentBytes = 4;
  constexpr size_t kCompletedBeforeFailure = 4;
  MetricsTransport transport;
  transport.pipeline = true;
  KVClient client({{"n", "test:1"}}, "metrics/cross-rail-sg", &transport);

  std::vector<std::array<std::string, 3>> segments(kItems);
  std::vector<std::string> expected(kItems);
  std::vector<KvPutItemSg> puts;
  puts.reserve(kItems);
  for (size_t i = 0; i < kItems; ++i) {
    for (size_t segment = 0; segment < 3; ++segment) {
      segments[i][segment].resize(kSegmentBytes);
      for (size_t byte = 0; byte < kSegmentBytes; ++byte) {
        segments[i][segment][byte] = static_cast<char>(
            (i * 37 + segment * 11 + byte * 53) & 0xff);
      }
      expected[i] += segments[i][segment];
    }
    puts.push_back(
        {"key-" + std::to_string(i),
         {segments[i][0].data(), segments[i][1].data(),
          segments[i][2].data()},
         {kSegmentBytes, kSegmentBytes, kSegmentBytes}});
  }

  transport.Script({{0, Result::kSuccess}});
  EXPECT_EQ(client.BatchPutSg(puts), std::vector<bool>(kItems, true));
  EXPECT_EQ(transport.replay_item_counts, (std::vector<size_t>{kItems}));
  ASSERT_EQ(transport.attempts.size(), 1u);

  std::vector<std::array<std::array<char, kSegmentBytes>, 3>> output(kItems);
  std::vector<KvGetItemSg> gets;
  gets.reserve(kItems);
  for (size_t i = 0; i < kItems; ++i) {
    gets.push_back(
        {puts[i].key,
         {output[i][0].data(), output[i][1].data(), output[i][2].data()},
         {kSegmentBytes, kSegmentBytes, kSegmentBytes}});
  }
  transport.Script(
      {{0, Result::kRailFailure, {}, kCompletedBeforeFailure},
       {1, Result::kSuccess}});
  std::vector<size_t> lengths;
  EXPECT_EQ(client.BatchGetAutoSg(gets, &lengths),
            std::vector<bool>(kItems, true));
  EXPECT_EQ(lengths, std::vector<size_t>(kItems, 3 * kSegmentBytes));
  EXPECT_EQ(transport.replay_item_counts,
            (std::vector<size_t>{kItems,
                                 kItems - kCompletedBeforeFailure}));
  ASSERT_EQ(transport.attempts.size(), 2u);
  ASSERT_EQ(transport.attempt_item_indices.size(), 2u);
  EXPECT_EQ(transport.attempt_item_indices[0],
            (std::vector<size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_EQ(transport.attempt_item_indices[1],
            (std::vector<size_t>{4, 5, 6, 7, 8}))
      << "items completed by attempt one must never be replayed";
  EXPECT_EQ(transport.range_multi_item_calls,
            std::vector<size_t>(kItems, 1u))
      << "every byte-exact item must be fetched exactly once across attempts";
  for (size_t i = 0; i < kItems; ++i) {
    std::string actual;
    for (const auto& segment : output[i])
      actual.append(segment.data(), segment.size());
    EXPECT_EQ(actual, expected[i]) << i;
  }

  const std::string snapshot = client.MetricsSnapshot();
  for (const char* op : {"put", "get"}) {
    EXPECT_EQ(Metric(snapshot, "dfkv_client_op_requests_total", op), 1u);
    EXPECT_EQ(Metric(snapshot, "dfkv_client_op_keys_total", op), kItems);
    EXPECT_EQ(Metric(snapshot, "dfkv_client_op_hits_total", op), kItems);
    EXPECT_EQ(Metric(snapshot, "dfkv_client_op_bytes_total", op),
              kItems * 3 * kSegmentBytes);
    EXPECT_EQ(Metric(snapshot, "dfkv_client_op_latency_seconds_count", op),
              1u);
  }
  EXPECT_EQ(TransportCounter(
                snapshot, "dfkv_rdma_client_cross_rail_retries_total"),
            1u);
  EXPECT_EQ(TransportCounter(
                snapshot,
                "dfkv_rdma_client_cross_rail_retry_successes_total"),
            1u);
  EXPECT_EQ(TransportCounter(
                snapshot,
                "dfkv_rdma_client_cross_rail_retry_exhausted_total"),
            0u);
}

TEST(ClientOpMetrics, ExhaustedCrossRailRetryCountsOneFailure) {
  using Result = MetricsTransport::AttemptResult;
  constexpr size_t kItems = 9;
  constexpr size_t kSegmentBytes = 4;
  MetricsTransport transport;
  transport.pipeline = true;
  KVClient client({{"n", "test:1"}}, "metrics/cross-rail-exhausted",
                  &transport);

  std::vector<std::array<std::array<char, kSegmentBytes>, 3>> output(kItems);
  for (auto& item : output)
    for (auto& segment : item) segment.fill(static_cast<char>(0x5a));
  std::vector<KvGetItemSg> gets;
  gets.reserve(kItems);
  for (size_t i = 0; i < kItems; ++i) {
    gets.push_back(
        {"key-" + std::to_string(i),
         {output[i][0].data(), output[i][1].data(), output[i][2].data()},
         {kSegmentBytes, kSegmentBytes, kSegmentBytes}});
  }

  transport.Script(
      {{0, Result::kRailFailure},
       {1, Result::kRailFailure},
       {2, Result::kSuccess}});
  std::vector<size_t> lengths;
  EXPECT_EQ(client.BatchGetAutoSg(gets, &lengths),
            std::vector<bool>(kItems, false));
  EXPECT_EQ(lengths, std::vector<size_t>(kItems, 0));
  ASSERT_EQ(transport.attempts.size(), 2u);
  EXPECT_EQ(transport.attempts[0].rail, 0u);
  EXPECT_EQ(transport.attempts[1].rail, 1u);
  EXPECT_EQ(transport.replay_item_counts,
            (std::vector<size_t>{kItems, kItems}));
  for (const auto& item : output)
    for (const auto& segment : item)
      for (char byte : segment) EXPECT_EQ(byte, static_cast<char>(0x5a));

  const std::string snapshot = client.MetricsSnapshot();
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_requests_total", "get"), 1u);
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_keys_total", "get"), kItems);
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_hits_total", "get"), 0u);
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_bytes_total", "get"), 0u);
  EXPECT_EQ(Metric(snapshot, "dfkv_client_op_latency_seconds_count", "get"),
            1u);
  EXPECT_EQ(TransportCounter(
                snapshot, "dfkv_rdma_client_cross_rail_retries_total"),
            1u);
  EXPECT_EQ(TransportCounter(
                snapshot,
                "dfkv_rdma_client_cross_rail_retry_successes_total"),
            0u);
  EXPECT_EQ(TransportCounter(
                snapshot,
                "dfkv_rdma_client_cross_rail_retry_exhausted_total"),
            1u);
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

TEST(ClientDedup, SuccessfulPutsInvalidateCachedAbsence) {
  const std::string key_namespace = "metrics/put-invalidates-absence";
  DedupEnv env(key_namespace);
  MetricsTransport transport;
  transport.pipeline = true;
  KVClient client({{"n", "test:1"}}, key_namespace, &transport);
  const std::string value = "payload";

  EXPECT_EQ(client.BatchExist({"scalar"}), std::vector<bool>({false}));
  EXPECT_TRUE(client.Put("scalar", value.data(), value.size()));
  EXPECT_EQ(client.BatchExist({"scalar"}), std::vector<bool>({true}));

  EXPECT_EQ(client.BatchExist({"batch"}), std::vector<bool>({false}));
  std::vector<KvPutItem> batch{
      {"batch", value.data(), value.size()},
  };
  EXPECT_EQ(client.BatchPut(batch), std::vector<bool>({true}));
  EXPECT_EQ(client.BatchExist({"batch"}), std::vector<bool>({true}));

  EXPECT_EQ(client.BatchExist({"sg"}), std::vector<bool>({false}));
  std::vector<KvPutItemSg> sg{
      {"sg", {value.data()}, {value.size()}},
  };
  EXPECT_EQ(client.BatchPutSg(sg), std::vector<bool>({true}));
  EXPECT_EQ(client.BatchExist({"sg"}), std::vector<bool>({true}));

  EXPECT_EQ(transport.exist_calls, 6u)
      << "each successful PUT must invalidate the cached absent verdict";
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

TEST(ClientRetries, SgGetRetriesTransientMinorityMisses) {
  MetricsTransport transport;
  transport.pipeline = true;
  const std::string key_namespace = "retry/sg";
  KVClient client({{"n", "test:1"}}, key_namespace, &transport);
  const std::string value = "abcdef";
  ASSERT_TRUE(client.Put("retry", value.data(), value.size()));
  ASSERT_TRUE(client.Put("stable", value.data(), value.size()));
  transport.range_multi_misses = 1;
  std::array<char, 3> retry_first{}, retry_second{};
  std::array<char, 3> stable_first{}, stable_second{};
  KvGetItemSg retry{"retry",
                    {retry_first.data(), retry_second.data()},
                    {retry_first.size(), retry_second.size()}};
  KvGetItemSg stable{"stable",
                     {stable_first.data(), stable_second.data()},
                     {stable_first.size(), stable_second.size()}};
  std::vector<size_t> lengths;
  EXPECT_EQ(client.BatchGetAutoSg({retry, stable}, &lengths),
            std::vector<bool>({true, true}));
  EXPECT_EQ(lengths,
            std::vector<size_t>({value.size(), value.size()}));
  EXPECT_EQ(std::string(retry_first.data(), retry_first.size()) +
                std::string(retry_second.data(), retry_second.size()),
            value);
  EXPECT_EQ(transport.range_multi_calls, 2u);
}

TEST(ClientPernodeStats, OutputsFailuresRetriesAndBoundedRotation) {
  MetricsTransport transport;
  dfkv::testing::DrainPernodeSink();
  transport.pipeline = true;
  KVClient client({{"node-a", "10.0.0.1:12000"}},
                  "pernode/contract", &transport);
  const std::array<std::string, 3> keys{
      "secret-put-alpha", "secret-put-beta", "secret-put-gamma"};
  const std::string value = "payload";

  transport.cache_status_limit = 1;
  std::vector<KvPutItemSg> puts;
  for (const std::string& key : keys)
    puts.push_back({key, {value.data()}, {value.size()}});
  EXPECT_EQ(client.BatchPutSg(puts),
            std::vector<bool>({true, false, false}));
  dfkv::testing::DrainPernodeSink();
  const std::string put_log =
      CurrentAndPrevious("dfkv_client_put.log");
  EXPECT_NE(put_log.find(
                "logical_keys=3 issued_keys=3"),
            std::string::npos)
      << put_log;
  EXPECT_NE(put_log.find("logical_success=1 logical_fail=2"),
            std::string::npos)
      << put_log;
  EXPECT_NE(put_log.find("failed_issued_keys=2"), std::string::npos)
      << put_log;
  EXPECT_NE(put_log.find("statuses=kOk,kInvalid"), std::string::npos)
      << put_log;
  EXPECT_EQ(put_log.find(keys[1]), std::string::npos)
      << "raw keys must never be logged";
  EXPECT_NE(put_log.find("first_fail=len="), std::string::npos) << put_log;

  std::array<std::array<char, 16>, 3> output{};
  std::vector<KvGetItemSg> gets;
  for (size_t i = 0; i < keys.size(); ++i)
    gets.push_back({keys[i], {output[i].data()}, {output[i].size()}});
  transport.range_status_limit = 1;
  std::vector<size_t> lengths;
  EXPECT_EQ(client.BatchGetAutoSg(gets, &lengths),
            std::vector<bool>({true, false, false}));
  dfkv::testing::DrainPernodeSink();
  std::string get_log = CurrentAndPrevious("dfkv_client_get.log");
  EXPECT_NE(get_log.find("logical_keys=3 issued_keys=3"),
            std::string::npos)
      << get_log;
  EXPECT_NE(get_log.find("failed_issued_keys=2"), std::string::npos)
      << get_log;
  EXPECT_NE(get_log.find("failure_status_mask=32"), std::string::npos)
      << get_log;
  EXPECT_EQ(get_log.find(keys[1]), std::string::npos);

  transport.range_status_limit = std::numeric_limits<size_t>::max();
  transport.range_multi_misses = 1;
  EXPECT_EQ(client.BatchGetAutoSg(gets, &lengths),
            std::vector<bool>({true, true, true}));
  dfkv::testing::DrainPernodeSink();
  get_log = CurrentAndPrevious("dfkv_client_get.log");
  EXPECT_NE(get_log.find(
                "logical_keys=3 issued_keys=4"),
            std::string::npos)
      << get_log;
  EXPECT_NE(get_log.find("logical_hits=3 logical_not_hit=0"),
            std::string::npos)
      << get_log;

  transport.exist_status_limit = 1;
  EXPECT_EQ(
      client.BatchExist(std::vector<std::string>(keys.begin(), keys.end())),
      std::vector<bool>({true, false, false}));
  dfkv::testing::DrainPernodeSink();
  const std::string exist_log =
      CurrentAndPrevious("dfkv_client_exist.log");
  EXPECT_NE(exist_log.find("logical_keys=3 issued_keys=3"),
            std::string::npos)
      << exist_log;
  EXPECT_NE(exist_log.find("failed_issued_keys=2"), std::string::npos)
      << exist_log;
  EXPECT_NE(exist_log.find("failure_status_mask=32"), std::string::npos)
      << exist_log;

  EXPECT_EQ(dfkv::testing::PernodeQueueCapacity(), 4u);
  EXPECT_GT(dfkv::testing::PernodeQueueHighWater(), 0u);
  EXPECT_LE(dfkv::testing::PernodeQueueHighWater(),
            dfkv::testing::PernodeQueueCapacity());
  std::ifstream rotated(
      pernode_env.Path("dfkv_client_get.log") + ".1");
  EXPECT_TRUE(rotated.good()) << "fixed .1 rotation file was not created";
}

TEST(ClientPernodeStats, FingerprintsAreKeyedStableAndNeverExposeRawKeys) {
  const std::string key = "dictionary-guessable-secret-key";
  const std::string first =
      dfkv::testing::PernodeKeyFingerprintForTest(key);
  const std::string second =
      dfkv::testing::PernodeKeyFingerprintForTest(key);

  EXPECT_EQ(first, second);
  EXPECT_NE(first.find("len=" + std::to_string(key.size())),
            std::string::npos);
  EXPECT_NE(first.find("hmac-sha256="), std::string::npos);
  EXPECT_EQ(first.find("fnv"), std::string::npos);
  EXPECT_EQ(first.find(key), std::string::npos);
}

TEST(ClientPernodeStats, RecordsEveryPretransportRoutingFailure) {
  dfkv::testing::DrainPernodeSink();
  MetricsTransport transport;
  transport.pipeline = true;
  KVClient client({}, "pernode/no-route", &transport);
  const std::string value = "payload";
  std::array<char, 16> output{};

  const KvGetItemSg invalid_get{
      "invalid-get-secret", {output.data()}, {}};
  const KvGetItemSg unrouted_get{
      "unrouted-get-secret", {output.data()}, {output.size()}};
  std::vector<size_t> lengths;
  EXPECT_EQ(client.BatchGetAutoSg({invalid_get, unrouted_get}, &lengths),
            std::vector<bool>({false, false}));
  EXPECT_EQ(client.BatchExist({"unrouted-exist-secret"}),
            std::vector<bool>({false}));
  const KvPutItemSg invalid_put{
      "invalid-put-secret", {value.data()}, {}};
  const KvPutItemSg unrouted_put{
      "unrouted-put-secret", {value.data()}, {value.size()}};
  EXPECT_EQ(client.BatchPutSg({invalid_put, unrouted_put}),
            std::vector<bool>({false, false}));
  dfkv::testing::DrainPernodeSink();

  const std::string get_log =
      CurrentAndPrevious("dfkv_client_get.log");
  EXPECT_TRUE(HasLineWith(
      get_log,
      {"pernode-get TOTAL:", "logical_keys=2", "issued_keys=0",
       "nonissued_failures=2", "invalid_input=1", "no_route=1",
       "health_cooldown=0",
       "nonissued_reasons=invalid_input:kInvalid=1,"
       "no_route:not_issued=1"}))
      << get_log;
  EXPECT_TRUE(HasLineWith(
      get_log,
      {"failure_status_mask=0", "logical_failure_status_mask=32",
       "nonissued_status_mask=32", "nonissued_statuses=kInvalid"}))
      << get_log;
  EXPECT_EQ(get_log.find("invalid-get-secret"), std::string::npos);
  EXPECT_EQ(get_log.find("unrouted-get-secret"), std::string::npos);

  const std::string exist_log =
      CurrentAndPrevious("dfkv_client_exist.log");
  EXPECT_TRUE(HasLineWith(
      exist_log,
      {"pernode-exist TOTAL:", "logical_keys=1", "issued_keys=0",
       "nonissued_failures=1", "no_route=1",
       "nonissued_reasons=no_route:not_issued=1"}))
      << exist_log;
  EXPECT_EQ(exist_log.find("unrouted-exist-secret"), std::string::npos);

  const std::string put_log =
      CurrentAndPrevious("dfkv_client_put.log");
  EXPECT_TRUE(HasLineWith(
      put_log,
      {"pernode-put TOTAL:", "logical_keys=2", "issued_keys=0",
       "nonissued_failures=2", "invalid_input=1", "no_route=1",
       "preroute_fail=2"}))
      << put_log;
  EXPECT_EQ(put_log.find("invalid-put-secret"), std::string::npos);
  EXPECT_EQ(put_log.find("unrouted-put-secret"), std::string::npos);
}

TEST(ClientPernodeStats, RecordsGetExistAndPutHealthCooldownSkips) {
  dfkv::testing::DrainPernodeSink();
  const std::string value = "payload";

  MetricsTransport get_transport;
  get_transport.pipeline = true;
  get_transport.Script(
      {{0, MetricsTransport::AttemptResult::kRailFailure}});
  KVClient get_client({{"get-node", "10.0.0.1:12100"}},
                      "pernode/get-cooldown", &get_transport);
  std::array<char, 16> output{};
  KvGetItemSg get{"cooling-get-secret",
                  {output.data()}, {output.size()}};
  std::vector<size_t> lengths;
  EXPECT_EQ(get_client.BatchGetAutoSg({get}, &lengths),
            std::vector<bool>({false}));
  const size_t get_calls = get_transport.range_multi_calls;
  EXPECT_EQ(get_client.BatchGetAutoSg({get}, &lengths),
            std::vector<bool>({false}));
  EXPECT_EQ(get_transport.range_multi_calls, get_calls);
  dfkv::testing::DrainPernodeSink();

  MetricsTransport exist_transport;
  exist_transport.pipeline = true;
  exist_transport.exist_script = {Status::kIOError};
  KVClient exist_client({{"exist-node", "10.0.0.1:12200"}},
                        "pernode/exist-cooldown", &exist_transport);
  EXPECT_EQ(exist_client.BatchExist({"cooling-exist-secret"}),
            std::vector<bool>({false}));
  const size_t exist_calls = exist_transport.exist_calls;
  EXPECT_EQ(exist_client.BatchExist({"cooling-exist-secret"}),
            std::vector<bool>({false}));
  EXPECT_EQ(exist_transport.exist_calls, exist_calls);
  dfkv::testing::DrainPernodeSink();

  MetricsTransport put_transport;
  put_transport.pipeline = true;
  put_transport.Script(
      {{0, MetricsTransport::AttemptResult::kRailFailure}});
  KVClient put_client({{"put-node", "10.0.0.1:12300"}},
                      "pernode/put-cooldown", &put_transport);
  KvPutItemSg put{"cooling-put-secret",
                  {value.data()}, {value.size()}};
  EXPECT_EQ(put_client.BatchPutSg({put}), std::vector<bool>({false}));
  const size_t put_attempts = put_transport.attempts.size();
  EXPECT_EQ(put_client.BatchPutSg({put}), std::vector<bool>({false}));
  EXPECT_EQ(put_transport.attempts.size(), put_attempts);
  dfkv::testing::DrainPernodeSink();

  const std::string get_log =
      CurrentAndPrevious("dfkv_client_get.log");
  EXPECT_TRUE(HasLineWith(
      get_log,
      {"pernode-get TOTAL:", "logical_keys=1", "issued_keys=0",
       "nonissued_failures=1", "health_cooldown=1",
       "failure_status_mask=0", "logical_failure_status_mask=0",
       "nonissued_reasons=health_cooldown:not_issued=1"}))
      << get_log;
  EXPECT_EQ(get_log.find("cooling-get-secret"), std::string::npos);

  const std::string exist_log =
      CurrentAndPrevious("dfkv_client_exist.log");
  EXPECT_TRUE(HasLineWith(
      exist_log,
      {"pernode-exist TOTAL:", "logical_keys=1", "issued_keys=0",
       "nonissued_failures=1", "health_cooldown=1",
       "nonissued_reasons=health_cooldown:not_issued=1"}))
      << exist_log;
  EXPECT_EQ(exist_log.find("cooling-exist-secret"), std::string::npos);

  const std::string put_log =
      CurrentAndPrevious("dfkv_client_put.log");
  EXPECT_TRUE(HasLineWith(
      put_log,
      {"pernode-put TOTAL:", "logical_keys=1", "issued_keys=0",
       "nonissued_failures=1", "health_cooldown=1",
       "nonissued_reasons=health_cooldown:not_issued=1"}))
      << put_log;
  EXPECT_EQ(put_log.find("cooling-put-secret"), std::string::npos);
}

TEST(ClientPernodeStats, QueueDropsAreReportedByNextWrittenBatch) {
  dfkv::testing::DrainPernodeSink();
  EXPECT_EQ(dfkv::testing::PernodeQueuePendingDropped(), 0u);
  const size_t capacity = dfkv::testing::PernodeQueueCapacity();
  const uint64_t dropped_before = dfkv::testing::PernodeQueueDropped();

  dfkv::testing::PausePernodeSink(true);
  for (size_t i = 0; i < capacity + 2; ++i)
    dfkv::testing::EnqueuePernodeForTest(
        "dfkv_client_drop_test.log",
        {"sink-test batch=" + std::to_string(i)});
  EXPECT_EQ(dfkv::testing::PernodeQueueDropped() - dropped_before, 2u);
  EXPECT_EQ(dfkv::testing::PernodeQueuePendingDropped(), 2u);
  dfkv::testing::PausePernodeSink(false);
  dfkv::testing::DrainPernodeSink();

  EXPECT_EQ(dfkv::testing::PernodeQueuePendingDropped(), 0u);
  const std::string log =
      CurrentAndPrevious("dfkv_client_drop_test.log");
  EXPECT_NE(log.find("pernode-sink DROPPED: dropped_batches=2"),
            std::string::npos)
      << log;
}
