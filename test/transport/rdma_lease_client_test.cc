#include "cache/kv_node_server.h"
#include "cache/rdma_server.h"
#include "transport/rdma_protocol.h"
#include "transport/rdma_transport.h"
#include "transport/rdma_verbs.h"
#include "utils/net_util.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace dfkv {
namespace {

class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) {
      present_ = true;
      old_ = old;
    }
    if (value) ::setenv(name, value, 1);
    else ::unsetenv(name);
  }
  ~ScopedEnv() {
    if (present_) ::setenv(name_.c_str(), old_.c_str(), 1);
    else ::unsetenv(name_.c_str());
  }
 private:
  std::string name_;
  std::string old_;
  bool present_ = false;
};

long Counter(const std::string& text, const std::string& name) {
  const auto at = text.rfind("\n" + name + " ");
  if (at == std::string::npos) return -1;
  return std::strtol(text.c_str() + at + name.size() + 2, nullptr, 10);
}

// Only bootstrap TCP is proxied; payloads use the actual RC QPs. Capability
// masks exercise optional leased PUT negotiation and mandatory GET contracts.
class CapabilityProxy {
 public:
  explicit CapabilityProxy(std::string backend) : backend_(std::move(backend)) {
    listener_ = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (listener_ < 0 ||
        ::bind(listener_, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) != 0 ||
        ::listen(listener_, 8) != 0) return;
    socklen_t size = sizeof(sa);
    if (::getsockname(listener_, reinterpret_cast<sockaddr*>(&sa), &size) != 0)
      return;
    addr = "127.0.0.1:" + std::to_string(ntohs(sa.sin_port));
    worker_ = std::thread([this] {
      while (!stop_.load()) {
        const int client = ::accept(listener_, nullptr, nullptr);
        if (client < 0) break;
        timeval timeout{3, 0};
        ::setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        ::setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        Handle(client);
        ::close(client);
      }
    });
  }
  ~CapabilityProxy() {
    stop_.store(true);
    if (listener_ >= 0) ::shutdown(listener_, SHUT_RDWR);
    if (worker_.joinable()) worker_.join();
    if (listener_ >= 0) ::close(listener_);
  }
  std::string addr;
  std::atomic<bool> advertise_lease{false};
  std::atomic<bool> advertise_dynamic_only{true};
  std::atomic<unsigned> ordinary_bootstraps{0};
  std::atomic<unsigned> leased_bootstraps{0};
  std::atomic<unsigned> dynamic_bootstraps{0};
  std::atomic<unsigned> ordinary_probes{0};

 private:
  void Handle(int client) {
    char frame[rdma::kDevNameBytes];
    if (!net::ReadAll(client, frame, sizeof(frame))) return;
    const int server = net::Dial(backend_, 3000, 3000);
    if (server < 0) return;
    if (net::WriteAll(server, frame, sizeof(frame))) {
      if (rdma::IsV2Probe(frame)) {
        char reply[rdma::kV2ProbeReplyBytes];
        if (net::ReadAll(server, reply, sizeof(reply))) {
          if (!advertise_lease.load()) {
            reply[5] = static_cast<char>(
                static_cast<uint8_t>(reply[5]) &
                ~rdma::kV2ProbeCapLeasedPut);
            ++ordinary_probes;
          }
          if (!advertise_dynamic_only.load()) {
            reply[5] = static_cast<char>(
                static_cast<uint8_t>(reply[5]) &
                ~rdma::kV2ProbeCapDynamicOnly);
          }
          net::WriteAll(client, reply, sizeof(reply));
        }
      } else {
        if (rdma::DevFrameRequestsLeasedPut(frame)) ++leased_bootstraps;
        else ++ordinary_bootstraps;
        const bool dynamic = rdma::DevFrameRequestsDynamicPull(frame);
        if (dynamic) ++dynamic_bootstraps;
        const size_t readiness_bytes = rdma::kV2DynamicOnlyReadinessBytes;
        char qp[rdma::kQpInfoBytes];
        char readiness[rdma::kV2DynamicOnlyReadinessBytes];
        if (net::ReadAll(client, qp, sizeof(qp)) &&
            net::WriteAll(server, qp, sizeof(qp)) &&
            net::ReadAll(server, qp, sizeof(qp)) &&
            net::WriteAll(client, qp, sizeof(qp)) &&
            net::ReadAll(server, readiness, readiness_bytes)) {
          net::WriteAll(client, readiness, readiness_bytes);
        }
      }
    }
    ::close(server);
  }
  std::string backend_;
  int listener_ = -1;
  std::atomic<bool> stop_{false};
  std::thread worker_;
};

class RdmaLeaseClient : public testing::Test {
 protected:
  static constexpr size_t kMaxMsg = 4u << 20;
  static constexpr size_t kCeiling = 1u << 20;
  static constexpr size_t kLarge = 256u << 10;
  static constexpr size_t kSmall = 4096;
  void SetUp() override {
    if (!RdmaTransport::Available()) GTEST_SKIP() << "no RDMA device";
    char path[] = "/tmp/dfkv_lease_client_XXXXXX";
    const char* created = ::mkdtemp(path);
    ASSERT_NE(created, nullptr);
    dir_ = created;
    store_ = std::make_unique<KvNodeServer>(dir_.string(), 64u << 20);
    ASSERT_EQ(store_->Start(0), Status::kOk);
    server_ = std::make_unique<RdmaServer>(
        [this](uint8_t op, const BlockKey& key, uint64_t off, uint64_t len,
               const char* payload, uint64_t payload_len, std::string* out,
               size_t* value_len) {
          return store_->ProcessRequestForKey(op, key, off, len, payload,
                                              payload_len, out, value_len);
        }, kMaxMsg);
    server_->set_cache_direct_handler(
        [this](const BlockKey& key, char* data, size_t len, size_t capacity) {
          return store_->CacheDirectForKey(key, data, len, capacity);
        });
    server_->set_range_handler(
        [this](const BlockKey& key, uint64_t off, uint64_t len, char* buffer,
               size_t capacity, const char** data, size_t* out_len,
               size_t* value_len) {
          return store_->RangeDirectForKey(key, off, len, buffer, capacity,
                                           data, out_len, value_len);
        });
    ASSERT_EQ(server_->Start(0), Status::kOk);
    addr_ = "127.0.0.1:" + std::to_string(server_->port());
  }
  void TearDown() override {
    if (server_) server_->Stop();
    if (store_) store_->Stop();
    if (!dir_.empty()) std::filesystem::remove_all(dir_);
  }
  void ExpectStored(const BlockKey& key, const std::string& expected) {
    std::string actual;
    size_t value_len = 0;
    ASSERT_EQ(store_->ProcessRequestForKey(
                  static_cast<uint8_t>(WireOp::kRange), key, 0, expected.size(),
                  nullptr, 0, &actual, &value_len), Status::kOk);
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(value_len, expected.size());
  }
  static CacheSrcMulti Segments(BlockKey key, const std::string& value) {
    const size_t half = value.size() / 2;
    return {key, {{value.data(), half}, {nullptr, 0},
                  {value.data() + half, value.size() - half}}};
  }
  ScopedEnv device_{"DFKV_RDMA_DEV", nullptr};
  ScopedEnv tiers_{"DFKV_RDMA_RAIL_TIERS", nullptr};
  ScopedEnv ram_{"DFKV_RAM_TIER", "0"};
  ScopedEnv local_fault_{"DFKV_RDMA_TEST_COMPLETION_FAULT", nullptr};
  ScopedEnv remote_fault_{"DFKV_RDMA_TEST_ENDPOINT_COMPLETION_FAULT", nullptr};
  ScopedEnv read_fault_{"DFKV_RDMA_TEST_PULL_READ_FAILURE", nullptr};
  ScopedEnv ceiling_{"DFKV_RDMA_MAX_BLOCK_BYTES", "1048576"};
  ScopedEnv threshold_{"DFKV_RDMA_INLINE_PUT_MAX_BYTES", "131072"};
  ScopedEnv minimum_{"DFKV_RDMA_CONNECTION_MIN_BLOCK_BYTES", "65536"};
  ScopedEnv segment_{"DFKV_RDMA_RECV_SEGMENT_SIZE", "67108864"};
  ScopedEnv chunk_{"DFKV_RDMA_RECV_CHUNK_BYTES", "4194304"};
  ScopedEnv depth_{"DFKV_RDMA_DEPTH", "2"};
  ScopedEnv keepalive_{"DFKV_RDMA_KEEPALIVE_MS", "0"};
  std::filesystem::path dir_;
  std::unique_ptr<KvNodeServer> store_;
  std::unique_ptr<RdmaServer> server_;
  std::string addr_;
};

TEST_F(RdmaLeaseClient, ExplicitCeilingAppliesToEveryPutAndGetApi) {
  RdmaTransport transport(kMaxMsg);
  const BlockKey key{1, 1};
  const std::string boundary(kCeiling, 'b');
  const std::string oversized(kCeiling + 1, 'x');
  ASSERT_EQ(transport.Cache(addr_, key, boundary.data(), boundary.size()), Status::kOk);
  std::string out;
  ASSERT_EQ(transport.Range(addr_, key, 0, boundary.size(), &out), Status::kOk);
  EXPECT_EQ(out, boundary);
  EXPECT_EQ(transport.Cache(addr_, {1, 2}, oversized.data(), oversized.size()), Status::kInvalid);
  EXPECT_EQ(transport.CacheMany(addr_, {{{1, 3}, oversized.data(), oversized.size()}}),
            std::vector<Status>{Status::kInvalid});
  EXPECT_EQ(transport.CacheFrom(addr_, {{{1, 4}, oversized.data(), oversized.size()}}),
            std::vector<Status>{Status::kInvalid});
  EXPECT_EQ(transport.CacheFromMulti(addr_, {Segments({1, 5}, oversized)}),
            std::vector<Status>{Status::kInvalid});
  EXPECT_EQ(transport.Range(addr_, key, 0, oversized.size(), &out), Status::kInvalid);
  std::vector<std::string> outs;
  EXPECT_EQ(transport.RangeMany(addr_, {key}, 0, oversized.size(), &outs),
            std::vector<Status>{Status::kInvalid});
  out.resize(oversized.size());
  EXPECT_EQ(transport.RangeInto(addr_, {key}, {{out.data(), out.size()}}, nullptr),
            std::vector<Status>{Status::kInvalid});
  EXPECT_EQ(transport.RangeIntoMulti(addr_, {key}, {{{{out.data(), out.size()}}}}, nullptr),
            std::vector<Status>{Status::kInvalid});
}

TEST_F(RdmaLeaseClient, DefaultLeaseSettingDoesNotBypassExplicitCeiling) {
  ScopedEnv default_threshold("DFKV_RDMA_INLINE_PUT_MAX_BYTES", nullptr);
  RdmaTransport transport(kMaxMsg);
  const std::string oversized(kCeiling + 1, 'x');
  const std::string valid(kSmall, 'v');
  const auto result = transport.CacheMany(addr_, {
      {{2, 1}, oversized.data(), oversized.size()},
      {{2, 2}, valid.data(), valid.size()}});
  EXPECT_EQ(result, (std::vector<Status>{Status::kInvalid, Status::kOk}));
  ExpectStored({2, 2}, valid);
}

TEST_F(RdmaLeaseClient, InvalidInlineSlotsDoNotPoisonRepliesOrLaterWindows) {
  RdmaTransport transport(kMaxMsg);
  const std::string small(kSmall, 's');
  const std::string oversized(kCeiling + 1, 'x');
  const auto result = transport.CacheMany(addr_, {
      {{3, 1}, oversized.data(), oversized.size()},
      {{3, 2}, small.data(), small.size()},
      {{3, 3}, oversized.data(), oversized.size()},
      {{3, 4}, small.data(), small.size()},
      {{3, 5}, nullptr, 1},
      {{3, 6}, small.data(), small.size()}});
  EXPECT_EQ(result, (std::vector<Status>{Status::kInvalid, Status::kOk,
      Status::kInvalid, Status::kOk, Status::kInvalid, Status::kOk}));
  for (uint64_t i : {2, 4, 6}) ExpectStored({3, i}, small);
  EXPECT_EQ(Counter(transport.MetricsText(), "dfkv_rdma_client_stale_pool_retries_total"), 0);
}

TEST_F(RdmaLeaseClient, MixedSizeDuplicateKeysKeepFirstWriteWins) {
  RdmaTransport transport(kMaxMsg);
  const std::string large(kLarge, 'L');
  const std::string small(kSmall, 's');
  const auto result = transport.CacheMany(addr_, {
      {{4, 1}, large.data(), large.size()},
      {{4, 1}, small.data(), small.size()},
      {{4, 2}, small.data(), small.size()},
      {{4, 2}, large.data(), large.size()}});
  EXPECT_EQ(result, std::vector<Status>(4, Status::kOk));
  ExpectStored({4, 1}, large);
  ExpectStored({4, 2}, small);
  EXPECT_EQ(Counter(transport.MetricsText(), "dfkv_rdma_client_leaseput_ops_total"), 2);
}

TEST_F(RdmaLeaseClient, ScatterGatherMatchesOrderingValidationAndPayload) {
  RdmaTransport transport(kMaxMsg);
  const std::string large(kLarge, 'L');
  const std::string small(kSmall, 's');
  const std::string oversized(kCeiling + 1, 'x');
  const auto result = transport.CacheFromMulti(addr_, {
      Segments({5, 0}, oversized), Segments({5, 1}, large),
      Segments({5, 1}, small), {{5, 2}, {{nullptr, 1}}},
      Segments({5, 3}, small), Segments({5, 3}, large)});
  EXPECT_EQ(result, (std::vector<Status>{Status::kInvalid, Status::kOk,
      Status::kOk, Status::kInvalid, Status::kOk, Status::kOk}));
  ExpectStored({5, 1}, large);
  ExpectStored({5, 3}, small);
}

TEST_F(RdmaLeaseClient, EmptyAndZeroThresholdDisableLeaseNegotiation) {
  const std::string large(kLarge, 'v');
  uint64_t key = 0;
  for (const char* value : {"", "0"}) {
    ScopedEnv threshold("DFKV_RDMA_INLINE_PUT_MAX_BYTES", value);
    RdmaTransport transport(kMaxMsg);
    EXPECT_EQ(transport.CacheMany(addr_, {{{6, ++key}, large.data(), large.size()}}),
              std::vector<Status>{Status::kOk});
    EXPECT_EQ(transport.CacheFromMulti(addr_, {Segments({6, ++key}, large)}),
              std::vector<Status>{Status::kOk});
    EXPECT_EQ(Counter(transport.MetricsText(), "dfkv_rdma_client_leaseput_ops_total"), 0);
    EXPECT_EQ(Counter(transport.MetricsText(), "dfkv_rdma_client_inline_put_max_bytes"), 0);
  }
}

TEST_F(RdmaLeaseClient, OptionalLeasePeerReusesQpAndRefreshesOnPublication) {
  CapabilityProxy proxy(addr_);
  ASSERT_FALSE(proxy.addr.empty());
  RdmaTransport transport(kMaxMsg);
  PeerTopology topology;
  topology.peer_addr = proxy.addr;
  topology.peer_id = "lease-client-peer";
  topology.generation = 1;
  transport.OnPeerTopology(topology);
  const std::string large(kLarge, 'v');
  const auto put = [&](uint64_t key) {
    EXPECT_EQ(transport.CacheMany(proxy.addr, {{{7, key}, large.data(), large.size()}}),
              std::vector<Status>{Status::kOk});
    EXPECT_EQ(transport.CacheFromMulti(proxy.addr, {Segments({8, key}, large)}),
              std::vector<Status>{Status::kOk});
  };
  put(1);
  ASSERT_EQ(proxy.ordinary_bootstraps.load(), 1u);
  ASSERT_GT(proxy.ordinary_probes.load(), 0u);
  const long probes = Counter(transport.MetricsText(), "dfkv_rdma_client_v2_probe_attempts_total");
  put(2);
  EXPECT_EQ(proxy.ordinary_bootstraps.load(), 1u);
  EXPECT_EQ(proxy.leased_bootstraps.load(), 0u);
  EXPECT_EQ(Counter(transport.MetricsText(), "dfkv_rdma_client_v2_probe_attempts_total"), probes);
  // A repeated publication is not a refresh boundary.
  transport.OnPeerTopology(topology);
  put(3);
  EXPECT_EQ(proxy.ordinary_bootstraps.load(), 1u);
  proxy.advertise_lease.store(true);
  ++topology.generation;
  transport.OnPeerTopology(topology);
  put(4);
  EXPECT_EQ(proxy.leased_bootstraps.load(), 1u);
  EXPECT_EQ(Counter(transport.MetricsText(), "dfkv_rdma_client_leaseput_ops_total"), 2);
  proxy.advertise_lease.store(false);
  ++topology.generation;
  transport.OnPeerTopology(topology);
  put(5);
  put(6);
  EXPECT_EQ(proxy.ordinary_bootstraps.load(), 2u);
  EXPECT_EQ(proxy.leased_bootstraps.load(), 1u);
  EXPECT_EQ(Counter(transport.MetricsText(), "dfkv_rdma_client_stale_pool_retries_total"), 0);
  ExpectStored({7, 6}, large);
  ExpectStored({8, 6}, large);
}

TEST_F(RdmaLeaseClient, PostWriteFailureDoesNotReplayScalarOrScatterGather) {
  const std::string large(kLarge, 'v');
  for (bool sg : {false, true}) {
    RdmaTransport transport(kMaxMsg);
    PeerTopology topology;
    topology.peer_addr = addr_;
    topology.peer_id = "post-write-peer";
    topology.generation = 1;
    transport.OnPeerTopology(topology);
    ScopedEnv fault("DFKV_RDMA_TEST_ENDPOINT_COMPLETION_FAULT", "1:1:2");
    const BlockKey key{9, sg ? 2u : 1u};
    const auto result = sg
        ? transport.CacheFromMulti(addr_, {Segments(key, large)})
        : transport.CacheMany(addr_, {{key, large.data(), large.size()}});
    EXPECT_EQ(result, std::vector<Status>{Status::kIOError});
    // Fault is surfaced after the real commit completion: the write landed,
    // but the caller cannot claim success or replay an ambiguous request.
    ExpectStored(key, large);
    EXPECT_EQ(Counter(transport.MetricsText(), "dfkv_rdma_client_conns_opened_total"), 1);
    EXPECT_EQ(Counter(transport.MetricsText(), "dfkv_rdma_client_leaseput_ops_total"), 1);
    EXPECT_EQ(Counter(transport.MetricsText(), "dfkv_rdma_client_stale_pool_retries_total"), 0);
    EXPECT_EQ(Counter(transport.MetricsText(),
                      "dfkv_rdma_client_remote_rail_failures_total"), 1);
  }
}

TEST_F(RdmaLeaseClient, LeasePrepareFailuresKeepRetrySafetyAndAttribution) {
  const std::string large(kLarge, 'v');
  for (bool sg : {false, true}) {
    for (bool endpoint : {false, true}) {
      RdmaTransport transport(kMaxMsg);
      PeerTopology topology;
      topology.peer_addr = addr_;
      topology.peer_id = "prepare-failure-peer";
      topology.generation = 1;
      transport.OnPeerTopology(topology);
      ScopedEnv fault(endpoint ? "DFKV_RDMA_TEST_ENDPOINT_COMPLETION_FAULT"
                               : "DFKV_RDMA_TEST_COMPLETION_FAULT",
                      "1:1:1");
      const BlockKey key{10, static_cast<uint64_t>(sg * 2 + endpoint)};
      const auto result = sg
          ? transport.CacheFromMulti(addr_, {Segments(key, large)})
          : transport.CacheMany(addr_, {{key, large.data(), large.size()}});
      EXPECT_EQ(result, std::vector<Status>{
                            endpoint ? Status::kIOError : Status::kOk});
      EXPECT_EQ(Counter(transport.MetricsText(),
                        "dfkv_rdma_client_remote_rail_failures_total"),
                endpoint ? 1 : 0);
      // PREPARE cannot commit. Local failure may retry once on a traditional
      // connection; a fresh endpoint failure is not replayed on that rail.
      EXPECT_EQ(Counter(transport.MetricsText(),
                        "dfkv_rdma_client_conns_opened_total"),
                endpoint ? 1 : 2);
      if (!endpoint) ExpectStored(key, large);
    }
  }
}

TEST_F(RdmaLeaseClient, PullCapacityRejectionPreservesConnectionAndGeneration) {
  const std::string value(kSmall, 'v');
  for (bool sg : {false, true}) {
    RdmaTransport transport(kMaxMsg);
    PeerTopology topology;
    topology.peer_addr = addr_;
    topology.peer_id = "capacity-peer";
    topology.generation = 1;
    transport.OnPeerTopology(topology);
    const BlockKey key{11, sg ? 2u : 1u};
    ASSERT_EQ(transport.Cache(addr_, key, value.data(), value.size()), Status::kOk);
    const auto get = [&](std::string* out) {
      if (sg) {
        std::vector<size_t> lengths;
        const auto result = transport.RangeIntoMulti(
            addr_, {key}, {{{{out->data(), out->size()}}}}, &lengths);
        if (result[0] != Status::kOk) EXPECT_EQ(lengths[0], 0u);
        return result[0];
      }
      return transport.RangeInto(
          addr_, {key}, {{out->data(), out->size()}}, nullptr)[0];
    };
    // Success and each capacity rejection must retire their own generation.
    std::string out(value.size(), '?');
    ASSERT_EQ(get(&out), Status::kOk);
    EXPECT_EQ(out, value);
    for (size_t capacity : {value.size() - 1, value.size() - 2}) {
      out.assign(capacity, '?');
      EXPECT_EQ(get(&out), Status::kInvalid);
      EXPECT_EQ(out, std::string(capacity, '?'));
    }
    out.assign(value.size(), '?');
    ASSERT_EQ(get(&out), Status::kOk);
    EXPECT_EQ(out, value);
    EXPECT_EQ(Counter(transport.MetricsText(),
                      "dfkv_rdma_client_conns_opened_total"), 1);
    EXPECT_EQ(Counter(transport.MetricsText(),
                      "dfkv_rdma_client_remote_rail_failures_total"), 0);
    EXPECT_EQ(Counter(transport.MetricsText(),
                      "dfkv_rdma_client_pull_failures_total"), 0);
    EXPECT_EQ(Counter(transport.MetricsText(),
                      "dfkv_rdma_client_pull_releases_total"), 4);

    // In contrast, an actual READ failure still retires the endpoint and
    // attributes remote failure; capacity rejection must not hide that case.
    ScopedEnv fault("DFKV_RDMA_TEST_PULL_READ_FAILURE", "1");
    EXPECT_EQ(get(&out), Status::kIOError);
    EXPECT_EQ(Counter(transport.MetricsText(),
                      "dfkv_rdma_client_remote_rail_failures_total"), 1);
    EXPECT_EQ(Counter(transport.MetricsText(),
                      "dfkv_rdma_client_pull_failures_total"), 1);
  }
}

TEST_F(RdmaLeaseClient, DynamicPullReleasesLargeScalarAndScatterReadsBeforeIdle) {
  std::string value(kLarge, '\0');
  for (size_t i = 0; i < value.size(); ++i)
    value[i] = static_cast<char>(i % 251);
  const BlockKey key{12, 1};
  std::string ignored;
  size_t stored_len = 0;
  ASSERT_EQ(store_->ProcessRequestForKey(
                static_cast<uint8_t>(WireOp::kCache), key, 0, 0,
                value.data(), value.size(), &ignored, &stored_len), Status::kOk);
  RdmaTransport transport(kMaxMsg);
  const long baseline_bytes = Counter(
      transport.MetricsText(),
      "dfkv_rdma_client_registered_slot_bytes_budget{kind=\"used\"}");
  const auto expect_idle = [&] {
    EXPECT_EQ(Counter(server_->MetricsText(),
                      "dfkv_rdma_recv_segment_used_bytes"),
              static_cast<long>(rdma::V2SlotSize(65536)));
    EXPECT_EQ(Counter(transport.MetricsText(),
                      "dfkv_rdma_client_conns_opened_total"), 1);
    EXPECT_EQ(Counter(
                  transport.MetricsText(),
                  "dfkv_rdma_client_registered_slot_bytes_budget{kind=\"used\"}"),
              baseline_bytes + static_cast<long>(rdma::V2SlotSize(65536)));
  };
  for (int round = 0; round < 3; ++round) {
    std::string out(value.size() + 17, '?');
    std::vector<uint64_t> value_lens;
    ASSERT_EQ(transport.RangeInto(
                  addr_, {key}, {{out.data(), out.size()}}, &value_lens),
              std::vector<Status>{Status::kOk});
    EXPECT_EQ(out.substr(0, value.size()), value);
    EXPECT_EQ(out.substr(value.size()), std::string(17, '?'));
    EXPECT_EQ(value_lens, std::vector<uint64_t>{value.size()});
    expect_idle();
    out.assign(out.size(), '?');
    const size_t split = value.size() / 3 + 7;
    std::vector<size_t> lengths;
    ASSERT_EQ(transport.RangeIntoMulti(
                  addr_, {key},
                  {{{{out.data(), split}, {nullptr, 0},
                     {out.data() + split, out.size() - split}}}},
                  &lengths), std::vector<Status>{Status::kOk});
    EXPECT_EQ(out.substr(0, value.size()), value);
    EXPECT_EQ(out.substr(value.size()), std::string(17, '?'));
    EXPECT_EQ(lengths, std::vector<size_t>{value.size()});
    expect_idle();
  }
  EXPECT_EQ(Counter(transport.MetricsText(),
                    "dfkv_rdma_client_pull_releases_total"), 6);
  EXPECT_EQ(Counter(transport.MetricsText(),
                    "dfkv_rdma_client_pull_read_bytes_total"), 6 * kLarge);
  std::string small(value.size() - 1, '?');
  std::vector<uint64_t> rejected_lengths;
  EXPECT_EQ(transport.RangeInto(
                addr_, {key}, {{small.data(), small.size()}}, &rejected_lengths),
            std::vector<Status>{Status::kInvalid});
  EXPECT_EQ(rejected_lengths, std::vector<uint64_t>{value.size()});
  EXPECT_EQ(small, std::string(small.size(), '?'));
  expect_idle();
  std::vector<size_t> rejected_sg_lengths;
  EXPECT_EQ(transport.RangeIntoMulti(
                addr_, {key}, {{{{small.data(), small.size()}}}},
                &rejected_sg_lengths), std::vector<Status>{Status::kInvalid});
  EXPECT_EQ(rejected_sg_lengths, std::vector<size_t>{0});
  EXPECT_EQ(small, std::string(small.size(), '?'));
  expect_idle();
  std::string out(value.size(), '?');
  EXPECT_EQ(transport.RangeInto(
                addr_, {{12, 2}}, {{out.data(), out.size()}}, nullptr),
            std::vector<Status>{Status::kNotFound});
  EXPECT_EQ(out, std::string(value.size(), '?'));
  EXPECT_EQ(transport.RangeIntoMulti(
                addr_, {{12, 2}}, {{{{out.data(), out.size()}}}}, nullptr),
            std::vector<Status>{Status::kNotFound});
  EXPECT_EQ(transport.RangeInto(addr_, {key}, {{nullptr, 0}}, nullptr),
            std::vector<Status>{Status::kInvalid});
  EXPECT_EQ(transport.RangeIntoMulti(addr_, {key}, {{}}, nullptr),
            std::vector<Status>{Status::kInvalid});
  expect_idle();
  EXPECT_EQ(Counter(transport.MetricsText(),
                    "dfkv_rdma_client_pull_releases_total"), 8);
  EXPECT_EQ(Counter(transport.MetricsText(),
                    "dfkv_rdma_client_pull_failures_total"), 0);
}

TEST_F(RdmaLeaseClient, DynamicGetReusesQpAndRefreshesOptionalCapabilities) {
  const BlockKey key{13, 1};
  const std::string value(kLarge, 'g');
  std::string ignored;
  size_t stored_len = 0;
  ASSERT_EQ(store_->ProcessRequestForKey(
                static_cast<uint8_t>(WireOp::kCache), key, 0, 0,
                value.data(), value.size(), &ignored, &stored_len), Status::kOk);
  CapabilityProxy proxy(addr_);
  ASSERT_FALSE(proxy.addr.empty());
  RdmaTransport transport(kMaxMsg);
  PeerTopology topology;
  topology.peer_addr = proxy.addr;
  topology.peer_id = "get-peer";
  topology.generation = 1;
  transport.OnPeerTopology(topology);
  const auto get = [&] {
    std::string out(value.size(), '?');
    EXPECT_EQ(transport.RangeInto(
                  proxy.addr, {key}, {{out.data(), out.size()}}, nullptr),
              std::vector<Status>{Status::kOk});
    EXPECT_EQ(out, value);
    out.assign(out.size(), '?');
    EXPECT_EQ(transport.RangeIntoMulti(
                  proxy.addr, {key}, {{{{out.data(), out.size()}}}}, nullptr),
              std::vector<Status>{Status::kOk});
    EXPECT_EQ(out, value);
  };
  get();
  const long probes = Counter(transport.MetricsText(),
                             "dfkv_rdma_client_v2_probe_attempts_total");
  get();
  EXPECT_EQ(proxy.ordinary_bootstraps.load(), 1u);
  EXPECT_EQ(proxy.dynamic_bootstraps.load(), 1u);
  EXPECT_EQ(Counter(transport.MetricsText(),
                    "dfkv_rdma_client_v2_probe_attempts_total"), probes);
  EXPECT_EQ(Counter(server_->MetricsText(),
                    "dfkv_rdma_recv_segment_used_bytes"),
            static_cast<long>(rdma::V2SlotSize(65536)));
  proxy.advertise_lease.store(true);
  ++topology.generation;
  transport.OnPeerTopology(topology);
  get();
  EXPECT_EQ(proxy.dynamic_bootstraps.load(), 2u);
  proxy.advertise_lease.store(false);
  ++topology.generation;
  transport.OnPeerTopology(topology);
  get();
  get();
  EXPECT_EQ(proxy.ordinary_bootstraps.load(), 3u);
  EXPECT_EQ(proxy.dynamic_bootstraps.load(), 3u);
  // Peer identity changes invalidate optional bits even at the same generation.
  proxy.advertise_lease.store(true);
  topology.peer_id = "replacement-get-peer";
  transport.OnPeerTopology(topology);
  get();
  EXPECT_EQ(proxy.ordinary_bootstraps.load(), 4u);
  EXPECT_EQ(proxy.dynamic_bootstraps.load(), 4u);
  EXPECT_EQ(Counter(transport.MetricsText(),
                    "dfkv_rdma_client_pull_releases_total"), 12);
}

TEST_F(RdmaLeaseClient, DynamicOnlyClientRejectsOldServerBeforeQpBootstrap) {
  CapabilityProxy proxy(addr_);
  ASSERT_FALSE(proxy.addr.empty());
  proxy.advertise_dynamic_only.store(false);
  RdmaTransport transport(kMaxMsg);
  std::string out(4096, '?');
  EXPECT_EQ(transport.RangeInto(
                proxy.addr, {{14, 1}}, {{out.data(), out.size()}}, nullptr),
            std::vector<Status>{Status::kIOError});
  EXPECT_EQ(out, std::string(4096, '?'));
  EXPECT_EQ(proxy.dynamic_bootstraps.load(), 0u);
  EXPECT_EQ(proxy.ordinary_bootstraps.load(), 0u);
  EXPECT_EQ(Counter(transport.MetricsText(),
                    "dfkv_rdma_client_conns_opened_total"), 0);
  EXPECT_GT(Counter(transport.MetricsText(),
                    "dfkv_rdma_client_v2_probe_failures_total"), 0);
}

TEST_F(RdmaLeaseClient, StringRangesPreservePartialOffsetsAndStoredLength) {
  const BlockKey key{15, 1};
  const std::string value = "0123456789abcdefghij";
  RdmaTransport transport(kMaxMsg);
  ASSERT_EQ(transport.Cache(addr_, key, value.data(), value.size()), Status::kOk);
  std::string out;
  uint64_t length = 0;
  ASSERT_EQ(transport.Range(addr_, key, 7, 6, &out, &length), Status::kOk);
  EXPECT_EQ(out, value.substr(7, 6));
  EXPECT_EQ(length, value.size());
  std::vector<std::string> outputs;
  std::vector<uint64_t> lengths;
  EXPECT_EQ(transport.RangeMany(addr_, {key, {15, 2}}, 17, 8, &outputs, &lengths),
            (std::vector<Status>{Status::kOk, Status::kNotFound}));
  ASSERT_EQ(outputs.size(), 2u);
  EXPECT_EQ(outputs[0], value.substr(17));
  EXPECT_TRUE(outputs[1].empty());
  ASSERT_EQ(lengths.size(), 2u);
  EXPECT_EQ(lengths[0], value.size());
}

TEST_F(RdmaLeaseClient, StringSliceBatchKeepsOrderDepthAndFullLengths) {
  RdmaTransport transport(kMaxMsg);
  const std::string first = "0123456789abcdefghij";
  const std::string tail = "xyz";
  const BlockKey a{16, 1}, b{16, 2}, missing{16, 3};
  ASSERT_EQ(transport.CacheMany(
                addr_, {{a, first.data(), first.size()},
                        {b, tail.data(), tail.size()}}),
            (std::vector<Status>{Status::kOk, Status::kOk}));
  const long opened = Counter(transport.MetricsText(),
                              "dfkv_rdma_client_conns_opened_total");
  std::vector<std::string> outs;
  std::vector<uint64_t> lengths;
  EXPECT_EQ(transport.RangeMany(addr_, {a, missing, b, a, b},
                                2, 8, &outs, &lengths),
            (std::vector<Status>{Status::kOk, Status::kNotFound,
                                Status::kOk, Status::kOk, Status::kOk}));
  EXPECT_EQ(outs, (std::vector<std::string>{first.substr(2, 8), "",
                                          tail.substr(2), first.substr(2, 8),
                                          tail.substr(2)}));
  EXPECT_EQ(lengths, (std::vector<uint64_t>{first.size(), 0, tail.size(),
                                          first.size(), tail.size()}));
  EXPECT_EQ(Counter(transport.MetricsText(),
                    "dfkv_rdma_client_conns_opened_total"), opened);
  EXPECT_EQ(Counter(transport.MetricsText(),
                    "dfkv_rdma_client_pull_releases_total"), 4);
  EXPECT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), 0u);
  EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), 0u);
}

TEST_F(RdmaLeaseClient, StringSliceEmptyAndBoundsKeepZeroRequestContract) {
  RdmaTransport transport(kMaxMsg);
  const BlockKey key{17, 1}, missing{17, 2};
  const std::string value = "0123456789";
  ASSERT_EQ(transport.Cache(addr_, key, value.data(), value.size()), Status::kOk);
  std::string out = "unchanged";
  uint64_t full = 99;
  EXPECT_EQ(transport.Range(addr_, key, value.size(), 4, &out, &full),
            Status::kOk);
  EXPECT_TRUE(out.empty());
  EXPECT_EQ(full, value.size());
  EXPECT_EQ(transport.Range(addr_, key, value.size(), 0, &out, &full),
            Status::kOk);
  EXPECT_TRUE(out.empty());
  EXPECT_EQ(full, value.size());
  EXPECT_EQ(transport.Range(addr_, key, 0, 0, &out, &full), Status::kInvalid);
  EXPECT_TRUE(out.empty());
  EXPECT_EQ(full, 0u);
  EXPECT_EQ(transport.Range(addr_, key, 7, 0, &out, &full), Status::kInvalid);
  EXPECT_EQ(full, 0u);
  EXPECT_EQ(transport.Range(addr_, key, value.size() + 1, 0, &out, &full),
            Status::kInvalid);
  EXPECT_EQ(full, 0u);
  EXPECT_EQ(transport.Range(addr_, missing, 0, 0, &out, &full),
            Status::kNotFound);
  EXPECT_TRUE(out.empty());
  EXPECT_EQ(full, 0u);
  EXPECT_EQ(transport.Range(addr_, key, UINT64_MAX, 1, &out, &full),
            Status::kInvalid);
  EXPECT_TRUE(out.empty());
  EXPECT_EQ(transport.Range(addr_, key, 7, kCeiling, &out, &full), Status::kOk);
  EXPECT_EQ(out, value.substr(7));
  EXPECT_EQ(full, value.size());
  out = "retained";
  full = 99;
  EXPECT_EQ(transport.Range(addr_, key, 0, kCeiling + 1, &out, &full),
            Status::kInvalid);
  EXPECT_EQ(out, "retained");
  EXPECT_EQ(full, 99u);
  std::vector<std::string> outs;
  std::vector<uint64_t> lengths;
  EXPECT_EQ(transport.RangeMany(addr_, {key, missing, key},
                                value.size(), 0, &outs, &lengths),
            (std::vector<Status>{Status::kOk, Status::kNotFound, Status::kOk}));
  EXPECT_EQ(outs, (std::vector<std::string>{"", "", ""}));
  EXPECT_EQ(lengths, (std::vector<uint64_t>{value.size(), 0, value.size()}));
  EXPECT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), 0u);
}

TEST_F(RdmaLeaseClient, StringReadFailureDoesNotPublishStagedBytes) {
  RdmaTransport transport(kMaxMsg);
  const BlockKey key{18, 1};
  const std::string value(kSmall, 's');
  ASSERT_EQ(transport.Cache(addr_, key, value.data(), value.size()), Status::kOk);
  ScopedEnv fault("DFKV_RDMA_TEST_PULL_READ_FAILURE", "1");
  std::string out = "old";
  uint64_t full = 99;
  EXPECT_EQ(transport.Range(addr_, key, 7, 32, &out, &full), Status::kIOError);
  EXPECT_TRUE(out.empty());
  EXPECT_EQ(full, 99u);
  std::vector<std::string> outs{"retained"};
  std::vector<uint64_t> lengths{99};
  EXPECT_EQ(transport.RangeMany(addr_, {key, key}, 7, 32, &outs, &lengths),
            std::vector<Status>(2, Status::kIOError));
  EXPECT_EQ(outs, std::vector<std::string>{"retained"});
  EXPECT_EQ(lengths, std::vector<uint64_t>{99});
  EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), 0u);
}

}  // namespace
}  // namespace dfkv
