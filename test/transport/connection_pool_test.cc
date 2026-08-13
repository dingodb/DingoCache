// TDD R8 — TcpTransport connection pooling + server keep-alive.
// Many sequential requests over one transport must REUSE connections (not dial
// per call), and survive a stale pooled connection.
#include "cache/kv_node_server.h"
#include "client/key_map.h"
#include "transport/tcp_transport.h"

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {
struct Node {
  fs::path dir;
  std::unique_ptr<KvNodeServer> srv;
  std::string addr;
};
std::unique_ptr<Node> Start(const std::string& tag) {
  auto n = std::make_unique<Node>();
  n->dir = fs::temp_directory_path() / ("dfkv_pool_" + tag);
  fs::remove_all(n->dir);
  fs::create_directories(n->dir);
  n->srv = std::make_unique<KvNodeServer>(n->dir.string(), 1ull << 30);
  EXPECT_EQ(n->srv->Start(0), Status::kOk);
  n->addr = "127.0.0.1:" + std::to_string(n->srv->port());
  return n;
}

uint64_t MetricValue(const std::string& text, const std::string& name) {
  const std::string marker = "\n" + name + " ";
  const size_t pos = text.find(marker);
  EXPECT_NE(pos, std::string::npos) << text;
  if (pos == std::string::npos) return 0;
  return std::stoull(text.substr(pos + marker.size()));
}

int UnusedLoopbackPort() {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  EXPECT_GE(fd, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  EXPECT_EQ(::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
  socklen_t len = sizeof(addr);
  EXPECT_EQ(::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len), 0);
  const int port = ntohs(addr.sin_port);
  ::close(fd);
  return port;
}
}  // namespace

TEST(ConnectionPool, SequentialRequestsReuseOneConnection) {
  auto n = Start("reuse");
  TcpTransport t;
  std::string v(256, 'p');
  const int N = 100;
  for (int i = 0; i < N; ++i) {
    BlockKey bk = ToBlockKey("test/model", "k" + std::to_string(i));
    ASSERT_EQ(t.Cache(n->addr, bk, v.data(), v.size()), Status::kOk);
  }
  for (int i = 0; i < N; ++i) {
    BlockKey bk = ToBlockKey("test/model", "k" + std::to_string(i));
    std::string out;
    ASSERT_EQ(t.Range(n->addr, bk, 0, v.size(), &out), Status::kOk);
    ASSERT_EQ(out, v);
  }
  // 200 ops over one transport -> connections reused, far fewer than 200 accepts
  EXPECT_LE(n->srv->AcceptCount(), 4u) << "accepts=" << n->srv->AcceptCount();
  n->srv->Stop();
}

TEST(ConnectionPool, CorrectnessAcrossManyKeys) {
  auto n = Start("correct");
  TcpTransport t;
  for (int i = 0; i < 50; ++i) {
    BlockKey bk = ToBlockKey("test/model", "c" + std::to_string(i));
    std::string v = "val_" + std::to_string(i);
    ASSERT_EQ(t.Cache(n->addr, bk, v.data(), v.size()), Status::kOk);
  }
  for (int i = 0; i < 50; ++i) {
    BlockKey bk = ToBlockKey("test/model", "c" + std::to_string(i));
    bool e = false;
    ASSERT_EQ(t.Exist(n->addr, bk, &e), Status::kOk);
    EXPECT_TRUE(e);
    std::string out;
    ASSERT_EQ(t.Range(n->addr, bk, 0, 64, &out), Status::kOk);
    EXPECT_EQ(out, "val_" + std::to_string(i));
  }
  n->srv->Stop();
}

TEST(ConnectionPool, DeviceDestinationsUseStagedPublication) {
  auto n = Start("device_destination");
  TcpTransport t;
  const BlockKey key = ToBlockKey("test/model", "device_destination");
  const std::string value = "abcdefgh";
  ASSERT_EQ(t.Cache(n->addr, key, value.data(), value.size()), Status::kOk);
  const uint64_t accepts = n->srv->AcceptCount();

  std::string device_destination(value.size(), '!');
  std::vector<uint64_t> value_lengths;
  const auto device_status = t.RangeInto(
      n->addr, {key},
      {RangeDst{device_destination.data(), device_destination.size(),
                DestinationMemoryKind::kDevice}},
      &value_lengths);
  ASSERT_EQ(device_status.size(), 1u);
  EXPECT_EQ(device_status[0], Status::kIOError);
  ASSERT_EQ(value_lengths.size(), 1u);
  EXPECT_EQ(value_lengths[0], value.size());
  EXPECT_EQ(device_destination, std::string(value.size(), '!'));

  std::string host_segment(3, '!');
  std::string device_segment(value.size() - host_segment.size(), '!');
  const RangeDstMulti mixed_destination{{
      RangeDstSegment{host_segment.data(), host_segment.size()},
      RangeDstSegment{device_segment.data(), device_segment.size(),
                      DestinationMemoryKind::kDevice},
  }};
  std::vector<size_t> out_lengths;
  const auto mixed_status =
      t.RangeIntoMulti(n->addr, {key}, {mixed_destination}, &out_lengths);
  ASSERT_EQ(mixed_status.size(), 1u);
  EXPECT_EQ(mixed_status[0], Status::kIOError);
  ASSERT_EQ(out_lengths.size(), 1u);
  EXPECT_EQ(out_lengths[0], value.size());
  EXPECT_EQ(host_segment, value.substr(0, host_segment.size()));
  EXPECT_EQ(device_segment, std::string(device_segment.size(), '!'));

  std::string host_destination(value.size(), '!');
  const auto host_status = t.RangeInto(
      n->addr, {key},
      {RangeDst{host_destination.data(), host_destination.size()}},
      &value_lengths);
  ASSERT_EQ(host_status.size(), 1u);
  EXPECT_EQ(host_status[0], Status::kOk);
  EXPECT_EQ(host_destination, value);
  EXPECT_EQ(n->srv->AcceptCount(), accepts);
  n->srv->Stop();
}

TEST(ConnectionPool, SurvivesStalePooledConnection) {
  // Put via transport, restart the server (old pooled fd is now stale), then a
  // new request must transparently reconnect (retry once) — not hard-fail.
  auto n = Start("stale");
  TcpTransport t;
  BlockKey bk = ToBlockKey("test/model", "warm");
  std::string v(64, 'w');
  ASSERT_EQ(t.Cache(n->addr, bk, v.data(), v.size()), Status::kOk);  // pools a conn
  // bounce the server on the SAME port to invalidate the pooled connection
  int port = n->srv->port();
  n->srv->Stop();
  n->srv = std::make_unique<KvNodeServer>(n->dir.string(), 1ull << 30);
  ASSERT_EQ(n->srv->Start(port), Status::kOk);
  // existing data reloaded from disk; request must reconnect through stale fd
  std::string out;
  EXPECT_EQ(t.Range(n->addr, bk, 0, v.size(), &out), Status::kOk);
  EXPECT_EQ(out, v);
  EXPECT_EQ(
      MetricValue(
          t.MetricsText(),
          "dfkv_transport_pool_retirements_total{reason=\"error\"}"),
      1u);
  n->srv->Stop();
}

TEST(ConnectionPool, ConcurrentLoadGrowsToBoundThenReusesAndShrinks) {
  auto n = Start("adaptive");
  EndpointPoolOptions options;
  options.min_connections = 1;
  options.max_connections = 3;
  options.idle_timeout_ms = 2;
  options.acquire_timeout_ms = 10000;
  TcpTransport t(options);

  constexpr size_t kCalls = 12;
  std::string value(4u << 20, 'a');
  std::vector<Status> statuses(kCalls, Status::kIOError);
  std::mutex start_mu;
  std::condition_variable start_cv;
  size_t ready = 0;
  bool go = false;
  std::vector<std::thread> threads;
  threads.reserve(kCalls);
  for (size_t i = 0; i < kCalls; ++i) {
    threads.emplace_back([&, i] {
      {
        std::unique_lock<std::mutex> lk(start_mu);
        ++ready;
        start_cv.notify_all();
        start_cv.wait(lk, [&] { return go; });
      }
      const BlockKey key =
          ToBlockKey("test/model", "parallel-" + std::to_string(i));
      statuses[i] = t.Cache(n->addr, key, value.data(), value.size());
    });
  }
  {
    std::unique_lock<std::mutex> lk(start_mu);
    start_cv.wait(lk, [&] { return ready == kCalls; });
    go = true;
  }
  start_cv.notify_all();
  for (auto& thread : threads) thread.join();
  for (Status status : statuses) EXPECT_EQ(status, Status::kOk);

  const size_t accepts_after_burst = n->srv->AcceptCount();
  EXPECT_GE(accepts_after_burst, 2u);
  EXPECT_LE(accepts_after_burst, options.max_connections);
  std::string metrics = t.MetricsText();
  EXPECT_GE(MetricValue(metrics, "dfkv_transport_pool_connections"), 2u);
  EXPECT_LE(MetricValue(metrics, "dfkv_transport_pool_connections"),
            options.max_connections);
  EXPECT_EQ(MetricValue(metrics, "dfkv_transport_pool_inflight"), 0u);
  EXPECT_EQ(MetricValue(metrics, "dfkv_transport_pool_selections_total"),
            kCalls);

  // Steady-state calls reuse existing sockets: no connection per operation.
  for (size_t i = 0; i < 20; ++i) {
    bool exists = false;
    EXPECT_EQ(t.Exist(n->addr,
                      ToBlockKey("test/model",
                                 "parallel-" + std::to_string(i % kCalls)),
                      &exists),
              Status::kOk);
    EXPECT_TRUE(exists);
  }
  EXPECT_EQ(n->srv->AcceptCount(), accepts_after_burst);

  // The next acquisition performs opportunistic idle retirement, retaining the
  // configured warm minimum rather than the burst-sized pool.
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  bool exists = false;
  EXPECT_EQ(t.Exist(n->addr, ToBlockKey("test/model", "parallel-0"), &exists),
            Status::kOk);
  EXPECT_TRUE(exists);
  metrics = t.MetricsText();
  EXPECT_EQ(MetricValue(metrics, "dfkv_transport_pool_connections"), 1u);
  EXPECT_EQ(
      MetricValue(
          metrics,
          "dfkv_transport_pool_retirements_total{reason=\"idle\"}"),
      accepts_after_burst - options.min_connections);
  n->srv->Stop();
}

TEST(ConnectionPool, FailedDialEntersBackoffAndSuppressesImmediateRetry) {
  EndpointPoolOptions options;
  options.max_connections = 4;
  options.backoff_base_ms = 5;
  options.backoff_max_ms = 20;
  TcpTransport t(options);
  t.set_timeouts(50, 50);
  const std::string node =
      "127.0.0.1:" + std::to_string(UnusedLoopbackPort());
  bool exists = false;
  EXPECT_EQ(t.Exist(node, ToBlockKey("test/model", "down"), &exists),
            Status::kIOError);
  EXPECT_EQ(t.Exist(node, ToBlockKey("test/model", "down"), &exists),
            Status::kIOError);
  std::this_thread::sleep_for(std::chrono::milliseconds(8));
  EXPECT_EQ(t.Exist(node, ToBlockKey("test/model", "down"), &exists),
            Status::kIOError);
  EXPECT_EQ(t.Exist(node, ToBlockKey("test/model", "down"), &exists),
            Status::kIOError);

  const std::string metrics = t.MetricsText();
  EXPECT_EQ(MetricValue(metrics, "dfkv_transport_pool_connections"), 0u);
  EXPECT_EQ(MetricValue(metrics, "dfkv_transport_pool_backoff_endpoints"), 1u);
  EXPECT_EQ(MetricValue(metrics, "dfkv_transport_pool_backoff_events_total"),
            2u);
  EXPECT_EQ(
      MetricValue(metrics, "dfkv_transport_pool_backoff_suppressed_total"),
      2u);
}
