#include "client/kv_client.h"
#include "transport/transport.h"
#include <gtest/gtest.h>
#include <map>
#include <mutex>
#include <string>
using namespace dfkv;  // NOLINT

namespace {
std::string Hdr() { return "test/model"; }
struct CountingTransport : Transport {
  std::string dead;
  std::mutex mu;
  std::map<std::string, int> range_calls, cache_calls;
  bool pipelined() const override { return false; }
  Status Range(const std::string& node, const BlockKey&, uint64_t, uint64_t,
               std::string*, uint64_t*) override {
    { std::lock_guard<std::mutex> lk(mu); range_calls[node]++; }
    return node == dead ? Status::kIOError : Status::kNotFound;
  }
  Status Cache(const std::string& node, const BlockKey&, const void*, size_t) override {
    { std::lock_guard<std::mutex> lk(mu); cache_calls[node]++; }
    return node == dead ? Status::kIOError : Status::kOk;
  }
  Status Lookup(const std::string& node, const BlockKey&,
                uint64_t* value_len) override {
    *value_len = 0;
    return node == dead ? Status::kIOError : Status::kNotFound;
  }
  Status Exist(const std::string& node, const BlockKey&, bool* e) override {
    *e = false;
    return node == dead ? Status::kIOError : Status::kOk;
  }
};

uint64_t MetricValue(const std::string& text, const std::string& family) {
  const std::string prefix = family + " ";
  size_t pos = text.find(prefix);
  while (pos != std::string::npos && pos != 0 && text[pos - 1] != '\n')
    pos = text.find(prefix, pos + 1);
  EXPECT_NE(pos, std::string::npos) << text;
  return pos == std::string::npos
             ? 0
             : std::stoull(text.substr(pos + prefix.size()));
}
}  // namespace

TEST(KvClientHealth, DeadPeerShortCircuitsAfterFirstFailure) {
  CountingTransport t; t.dead = "10.0.0.9:1";
  KVClient c({{"n", "10.0.0.9:1"}}, Hdr(), &t);
  char out[64] = {0};
  EXPECT_FALSE(c.Get("k1", out, 64));
  EXPECT_FALSE(c.Get("k2", out, 64));
  EXPECT_FALSE(c.Get("k3", out, 64));
  std::lock_guard<std::mutex> lk(t.mu);
  EXPECT_EQ(t.range_calls["10.0.0.9:1"], 1);
}

TEST(KvClientHealth, HealthyPeerNotShortCircuited) {
  CountingTransport t;
  KVClient c({{"n", "10.0.0.1:1"}}, Hdr(), &t);
  char out[64] = {0};
  EXPECT_FALSE(c.Get("k1", out, 64));
  EXPECT_FALSE(c.Get("k2", out, 64));
  std::lock_guard<std::mutex> lk(t.mu);
  EXPECT_EQ(t.range_calls["10.0.0.1:1"], 2);
}

TEST(KvClientHealth, DeadPeerShortCircuitsPut) {
  CountingTransport t; t.dead = "10.0.0.9:2";
  KVClient c({{"n", "10.0.0.9:2"}}, Hdr(), &t);
  std::string v(64, 'x');
  EXPECT_FALSE(c.Put("k1", v.data(), v.size()));
  EXPECT_FALSE(c.Put("k2", v.data(), v.size()));
  std::lock_guard<std::mutex> lk(t.mu);
  EXPECT_EQ(t.cache_calls["10.0.0.9:2"], 1);
}

namespace {
// Local budget starvation: the peer was never dialed, so unlike kIOError it
// must not engage the cooldown (0812-004: cooling an innocent peer blanket-
// fails every key routed to it and collapses cache hit rates).
struct StarvedTransport : CountingTransport {
  Status Range(const std::string& node, const BlockKey&, uint64_t, uint64_t,
               std::string*, uint64_t*) override {
    { std::lock_guard<std::mutex> lk(mu); range_calls[node]++; }
    return Status::kResourceExhausted;
  }
  Status Cache(const std::string& node, const BlockKey&, const void*,
               size_t) override {
    { std::lock_guard<std::mutex> lk(mu); cache_calls[node]++; }
    return Status::kResourceExhausted;
  }
};

struct NoCompatibleTransport : CountingTransport {
  Status Range(const std::string& node, const BlockKey&, uint64_t, uint64_t,
               std::string*, uint64_t*) override {
    { std::lock_guard<std::mutex> lk(mu); range_calls[node]++; }
    return Status::kNoCompatibleRail;
  }
  Status Cache(const std::string& node, const BlockKey&, const void*,
               size_t) override {
    { std::lock_guard<std::mutex> lk(mu); cache_calls[node]++; }
    return Status::kNoCompatibleRail;
  }
};
}  // namespace

TEST(KvClientHealth, ResourceExhaustedDoesNotCoolThePeer) {
  StarvedTransport t;
  KVClient c({{"n", "10.0.0.9:3"}}, Hdr(), &t);
  char out[64] = {0};
  std::string v(64, 'x');
  // Every op fails honestly, but the peer stays healthy: each retry reaches
  // the transport instead of being short-circuited by a cooldown.
  EXPECT_FALSE(c.Get("k1", out, 64));
  EXPECT_FALSE(c.Get("k2", out, 64));
  EXPECT_FALSE(c.Put("k1", v.data(), v.size()));
  EXPECT_FALSE(c.Put("k2", v.data(), v.size()));
  std::lock_guard<std::mutex> lk(t.mu);
  EXPECT_EQ(t.range_calls["10.0.0.9:3"], 2);
  EXPECT_EQ(t.cache_calls["10.0.0.9:3"], 2);
}

TEST(KvClientHealth, NoCompatibleRailIsPeerNeutral) {
  NoCompatibleTransport t;
  const std::string peer = "10.0.0.9:4";
  KVClient c({{"n", peer}}, Hdr(), &t);
  char out[64] = {};
  const std::string value(64, 'x');

  EXPECT_FALSE(c.Get("k1", out, sizeof(out)));
  EXPECT_FALSE(c.Get("k2", out, sizeof(out)));
  EXPECT_FALSE(c.Put("k1", value.data(), value.size()));
  EXPECT_FALSE(c.Put("k2", value.data(), value.size()));

  {
    std::lock_guard<std::mutex> lk(t.mu);
    EXPECT_EQ(t.range_calls[peer], 2)
        << "a local compatibility miss must not cool the peer";
    EXPECT_EQ(t.cache_calls[peer], 2)
        << "a local compatibility miss must not short-circuit later keys";
  }
  const std::string metrics = c.MetricsSnapshot();
  EXPECT_EQ(MetricValue(metrics, "dfkv_client_ops_served_total"), 0u);
  EXPECT_EQ(MetricValue(metrics, "dfkv_client_io_errors_total"), 0u);
  EXPECT_EQ(MetricValue(metrics, "dfkv_client_unhealthy_skips_total"), 0u);
  EXPECT_EQ(MetricValue(metrics, "dfkv_client_peer_marked_bad_total"), 0u);
  EXPECT_EQ(MetricValue(metrics, "dfkv_client_busy_suppressed_total"), 0u);
}
