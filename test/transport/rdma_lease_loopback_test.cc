// Operation-scoped PUT/GET leases over loopback RDMA (Soft-RoCE or real HCA).
// Raw peers retain/replay capabilities so client retry cannot hide ownership
// faults. Skip cleanly when no RDMA device exists.
#include "client/kv_client.h"
#include "client/key_map.h"
#include "cache/kv_node_server.h"
#include "cache/rdma_server.h"
#include "transport/rdma_transport.h"
#include "transport/rdma_protocol.h"

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <optional>
#include <thread>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace dfkv {
class RdmaLeaseServerTestPeer {
 public:
  static size_t Trim(RdmaServer& server) {
    return server.recv_segments_.TrimIdle(1);
  }
  static rdma::RecvSegmentPool::Lease Allocate(RdmaServer& server, size_t bytes) {
    return server.recv_segments_.Allocate(bytes);
  }
  static void BeforeTeardown(RdmaServer& server, std::function<void()> hook) {
    server.before_endpoint_teardown_for_test_ = std::move(hook);
  }
};
}  // namespace dfkv

namespace {

constexpr size_t kMsg = 64ull << 20;  // like production rings' --max-msg

bool HaveRdma() { return RdmaTransport::Available(); }
std::string SelfHdr() { return "test/model"; }

class RdmaLeaseLoopback : public ::testing::Test {
 protected:
  RdmaLeaseLoopback() {
    for (const char* key : {"DFKV_RDMA_MAX_BLOCK_BYTES",
                            "DFKV_RDMA_RECV_SEGMENT_SIZE",
                            "DFKV_RDMA_RECV_CHUNK_BYTES",
                            "DFKV_RDMA_INLINE_PUT_MAX_BYTES",
                            "DFKV_RDMA_RECV_CHUNK_IDLE_MS",
                            "DFKV_RDMA_IDLE_MS",
                            "DFKV_RDMA_DEPTH"}) {
      const char* value = std::getenv(key);
      saved_.emplace_back(key, value ? std::optional<std::string>(value)
                                    : std::nullopt);
    }
    setenv("DFKV_RDMA_RECV_CHUNK_IDLE_MS", "0", 1);
    setenv("DFKV_RDMA_IDLE_MS", "0", 1);
  }
  ~RdmaLeaseLoopback() override {
    for (const auto& [key, value] : saved_) {
      if (value) setenv(key.c_str(), value->c_str(), 1);
      else unsetenv(key.c_str());
    }
  }
  std::vector<std::pair<std::string, std::optional<std::string>>> saved_;
};

// rfind lands on the VALUE line emitted after the # HELP / # TYPE lines,
// exactly like rdma_loopback_test's CounterVal.
long CounterOf(const RdmaServer& rsrv, const std::string& name) {
  const std::string text = rsrv.MetricsText();
  const size_t at = text.rfind(name + " ");
  if (at == std::string::npos) return -1;
  const size_t sp = text.find(' ', at);
  if (sp == std::string::npos) return -1;
  return std::strtol(text.c_str() + sp + 1, nullptr, 10);
}

// Production-equivalent handler wiring. Run this suite with
// DFKV_SERVER_URING=0 and =1 to exercise both serving loops.
struct LeaseNode {
  fs::path dir;
  std::unique_ptr<KvNodeServer> srv;
  std::unique_ptr<RdmaServer> rsrv;
  std::string addr;

  explicit LeaseNode(const std::string& tag) {
    // Production rings raise the declared object ceiling to the business
    // maximum (64MiB today); the 4MiB default would bound GET connections.
    setenv("DFKV_RDMA_MAX_BLOCK_BYTES", "67108864", 1);
    setenv("DFKV_RDMA_RECV_SEGMENT_SIZE", std::to_string(192ull << 20).c_str(),
           1);
    setenv("DFKV_RDMA_RECV_CHUNK_BYTES", std::to_string(32ull << 20).c_str(),
           1);
    dir = fs::temp_directory_path() / ("dfkv_lease_" + tag);
    fs::remove_all(dir);
    fs::create_directories(dir);
    srv = std::make_unique<KvNodeServer>(dir.string(), 1ull << 30);
    EXPECT_EQ(srv->Start(0), Status::kOk);
    rsrv = std::make_unique<RdmaServer>(
        [this](uint8_t op, const BlockKey& key, uint64_t off, uint64_t len,
               const char* pl, uint64_t pll, std::string* out,
               size_t* value_len) {
          return srv->ProcessRequestForKey(op, key, off, len, pl, pll, out,
                                            value_len);
        },
        kMsg);
    rsrv->set_range_handler(
        [this](const BlockKey& key, uint64_t off, uint64_t len, char* io_buf,
               size_t cap, const char** out_data, size_t* out_len,
               size_t* value_len) {
          return srv->RangeDirectForKey(key, off, len, io_buf, cap, out_data,
                                        out_len, value_len);
        });
    rsrv->set_cache_direct_handler(
        [this](const BlockKey& key, char* data, size_t len, size_t cap) {
          return srv->CacheDirectForKey(key, data, len, cap);
        });
    rsrv->set_prepare_read_handler(
        [this](const BlockKey& key, uint64_t off, uint64_t len,
               char* staging, size_t cap) {
          return srv->PrepareReadForKey(key, off, len, staging, cap);
        });
    EXPECT_EQ(rsrv->Start(0), Status::kOk);
    addr = "127.0.0.1:" + std::to_string(rsrv->port());
  }
  ~LeaseNode() {
    if (rsrv) rsrv->Stop();
    if (srv) srv->Stop();
    fs::remove_all(dir);
  }
  LeaseNode(const LeaseNode&) = delete;
  LeaseNode& operator=(const LeaseNode&) = delete;
};

std::string Value(size_t len, uint8_t seed) {
  std::string v(len, '\0');
  for (size_t i = 0; i < len; ++i)
    v[i] = static_cast<char>((i * 31 + seed) & 0xFF);
  return v;
}

// Raw v2 peer keeps one live QP while tests retain or deliberately replay a
// lease capability. No client retries may hide a server ownership violation.
struct LeasePeer {
  std::unique_ptr<rdma::RcEndpoint> endpoint =
      std::make_unique<rdma::RcEndpoint>();
  rdma::RcEndpoint& ep = *endpoint;
  rdma::RecvSegmentInfo resident;
  bool Open(const LeaseNode& node) {
    const auto& dev = node.rsrv->DeviceNames().front();
    if (!ep.Open(dev.c_str(), rdma::kV2ControlCap, 1)) return false;
    int fd = net::Dial(node.addr, 10000, 10000);
    if (fd < 0) return false;
    char frame[rdma::kDevNameBytes], mine[rdma::kQpInfoBytes],
        peer[rdma::kQpInfoBytes], ready[rdma::kV2DynamicOnlyReadinessBytes];
    rdma::EncodeDevFrame(dev, (1u << 20) |
                                 rdma::kDevFrameRequestDynamicOnly |
                                 rdma::kDevFrameRequestPullRead |
                                 rdma::kDevFrameRequestDynamicPull |
                                 rdma::kDevFrameRequestLeasedPut, frame);
    auto info = ep.Local();
    info.depth = 1;
    info.protocol_version = rdma::kDevProtoV2;
    rdma::SerializeQpInfo(info, mine);
    bool ok = net::WriteAll(fd, frame, sizeof(frame)) &&
              net::WriteAll(fd, mine, sizeof(mine)) &&
              net::ReadAll(fd, peer, sizeof(peer)) &&
              ep.Connect(rdma::ParseQpInfo(peer)) &&
              net::ReadAll(fd, ready, sizeof(ready)) &&
              rdma::DecodeV2DynamicOnlyReadiness(ready, sizeof(ready),
                                                 &resident);
    ::close(fd);
    return ok;
  }
  bool Response(Status* status, uint64_t* bytes) {
    bool sent = false, received = false;
    while (!sent || !received) {
      ibv_wc wc{};
      if (ep.WaitComp(&wc, 1, 10000) != 1 || wc.status != IBV_WC_SUCCESS)
        return false;
      if (wc.opcode == IBV_WC_RECV) {
        if (wc.byte_len < kRespPrefix ||
            !DecodeRespVersion(ep.rbuf(0), kNativeProtoRdmaV2,
                               status, bytes) ||
            wc.byte_len != kRespPrefix + *bytes)
          return false;
        received = true;
      } else {
        sent = true;
      }
    }
    return true;
  }
  bool Lease(const BlockKey& key, size_t size, Status* status,
             rdma::LeasePutReady* lease) {
    EncodeReqVersion(ep.sbuf(0), kNativeProtoRdmaV2, WireOp::kLeasePut,
                     key, 0, 0, size);
    uint64_t bytes = 0;
    if (!ep.PostRecv(0) || !ep.PostSend(0, kReqPrefix) ||
        !Response(status, &bytes)) return false;
    return *status != Status::kOk ||
           (bytes == rdma::kLeasePutReadyBytes &&
            rdma::DecodeLeasePutReady(ep.rbuf(0) + kRespPrefix, lease));
  }
  bool Put(const BlockKey& key, const rdma::LeasePutReady& lease,
           const std::string& value, bool multi = false) {
    ibv_mr* mr = ep.RegisterTransient(
        const_cast<char*>(value.data()), value.size(), false);
    if (!mr) return false;
    const size_t windows = multi ? 2 : 1;
    size_t offset = 0;
    for (size_t window = 0; window < windows; ++window) {
      const size_t size = window + 1 == windows
                              ? value.size() - offset : value.size() / 2;
      const size_t header = window == 0 ? kReqPrefix : 0;
      EncodeReqVersion(ep.sbuf(0), kNativeProtoRdmaV2, WireOp::kCache,
                       key, multi ? rdma::kV2MultiWrPutMagic : 0,
                       multi ? windows : 0, value.size());
      std::vector<std::pair<const void*, uint32_t>> segments;
      std::vector<ibv_mr*> mrs;
      const size_t first = multi && ep.max_sge() >= 3 ? size / 2 : size;
      segments.emplace_back(value.data() + offset, first);
      mrs.push_back(mr);
      if (first != size) {
        segments.emplace_back(value.data() + offset + first, size - first);
        mrs.push_back(mr);
      }
      const uint64_t address = lease.write_base + rdma::kV2DataOffset +
                               offset - header;
      Status status = Status::kIOError;
      uint64_t bytes = 0;
      if (!ep.PostRecv(0) ||
          !ep.PostWriteImmScatterMulti(0, header, segments, mrs, address,
                                       lease.rkey, lease.slot) ||
          !Response(&status, &bytes) || status != Status::kOk)
        return false;
      offset += size;
    }
    ep.ReleaseTransient(mr);
    return true;
  }
};

struct PullPeer : LeasePeer {
  bool Open(const LeaseNode& node, bool dynamic = true,
            bool request_pull = true, bool retirement = true,
            bool dynamic_only = true) {
    const auto& dev = node.rsrv->DeviceNames().front();
    if (!ep.Open(dev.c_str(), rdma::kV2ControlCap, 1)) return false;
    int fd = net::Dial(node.addr, 10000, 10000);
    if (fd < 0) return false;
    char frame[rdma::kDevNameBytes], mine[rdma::kQpInfoBytes],
        peer[rdma::kQpInfoBytes], ready[rdma::kV2RetirementReadinessBytes];
    uint64_t declared = 1u << 20;
    if (dynamic) declared |= rdma::kDevFrameRequestDynamicPull;
    if (request_pull) declared |= rdma::kDevFrameRequestPullRead;
    if (dynamic_only) declared |= rdma::kDevFrameRequestDynamicOnly;
    else if (retirement) declared |= rdma::kDevFrameRequestWriterRetirement;
    rdma::EncodeDevFrame(dev, declared, frame);
    auto info = ep.Local();
    info.depth = 1;
    info.protocol_version = rdma::kDevProtoV2;
    rdma::SerializeQpInfo(info, mine);
    const size_t size = dynamic_only ? rdma::kV2DynamicOnlyReadinessBytes
                                    : rdma::kV2RetirementReadinessBytes;
    uint64_t token = 0;
    bool ok = net::WriteAll(fd, frame, sizeof(frame)) &&
              net::WriteAll(fd, mine, sizeof(mine)) &&
              net::ReadAll(fd, peer, sizeof(peer)) &&
              ep.Connect(rdma::ParseQpInfo(peer)) &&
              net::ReadAll(fd, ready, size);
    if (ok) {
      ok = dynamic_only
               ? rdma::DecodeV2DynamicOnlyReadiness(ready, size, &resident)
               : rdma::DecodeV2Readiness(ready, size, &resident, &token);
      char extra = 0;
      ok = ok && ::read(fd, &extra, 1) == 0;  // exact bootstrap size, then EOF
    }
    ::close(fd);
    return ok;
  }
  bool Prepare(const BlockKey& key, size_t size, Status* status,
               rdma::DynamicPullReady* ready, uint64_t release = 0,
               uint32_t slot = 0, uint64_t offset = 0) {
    EncodeReqVersion(ep.sbuf(0), kNativeProtoRdmaV2, WireOp::kPullRange,
                     key, offset, size, rdma::kPullPrepareBytes);
    rdma::EncodePullPrepareControl({slot, release}, ep.sbuf(0) + kReqPrefix);
    uint64_t bytes = 0;
    if (!ep.PostRecv(0) ||
        !ep.PostSend(0, kReqPrefix + rdma::kPullPrepareBytes) ||
        !Response(status, &bytes)) return false;
    return *status != Status::kOk ||
           (bytes == rdma::kDynamicPullReadyBytes &&
            rdma::DecodeDynamicPullReady(ep.rbuf(0) + kRespPrefix, ready));
  }
  bool Release(const rdma::DynamicPullReady& ready, Status* status) {
    EncodeReqVersion(ep.sbuf(0), kNativeProtoRdmaV2, WireOp::kPullRelease,
                     BlockKey{}, 0, ready.slot_generation,
                     static_cast<uint64_t>(ready.slot_index) + 1);
    uint64_t bytes = 0;
    return ep.PostRecv(0) && ep.PostSend(0, kReqPrefix) &&
           Response(status, &bytes) && bytes == 0;
  }
  bool Read(const rdma::DynamicPullReady& ready, std::string* out) {
    out->assign(ready.data_len, '\0');
    if (out->empty()) return true;
    ibv_mr* mr = ep.RegisterTransient(out->data(), out->size());
    if (!mr) return false;
    ibv_wc wc{};
    const bool ok =
        ep.PostRead(0, out->data(), out->size(), mr, ready.address, ready.rkey) &&
        ep.WaitComp(&wc, 1, 10000) == 1 && wc.status == IBV_WC_SUCCESS;
    // Failed/ambiguous DMA must remain pinned until the local QP is destroyed.
    if (ok) ep.ReleaseTransient(mr);
    else endpoint.reset();
    return ok;
  }
};

void StoreForPull(LeaseNode& node, const BlockKey& key,
                  const std::string& value) {
  std::string out;
  size_t value_len = 0;
  ASSERT_EQ(node.srv->ProcessRequestForKey(
                static_cast<uint8_t>(WireOp::kCache), key, 0, 0,
                value.data(), value.size(), &out, &value_len), Status::kOk);
}

}  // namespace

TEST_F(RdmaLeaseLoopback, LargeObjectRoundTripsThroughStagedLease) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  setenv("DFKV_RDMA_INLINE_PUT_MAX_BYTES", std::to_string(1ull << 20).c_str(),
         1);
  LeaseNode node("roundtrip");
  RdmaTransport rt(kMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  const std::string v = Value(8ull << 20, 0x55);  // 8 MiB, well over 1 MiB
  ASSERT_TRUE(c.Put("big", v.data(), v.size()));

  const long lease_ops = CounterOf(*node.rsrv, "dfkv_rdma_leaseput_ops_total");
  EXPECT_GT(lease_ops, 0) << "object must use the staged-lease datapath";
  EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_leaseput_active"), 0)
      << "lease must return at store completion";

  std::string out(v.size(), '\0');
  ASSERT_TRUE(c.Get("big", &out[0], out.size()));
  EXPECT_EQ(out, v);
}

TEST_F(RdmaLeaseLoopback, InlineObjectKeepsResidentPath) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  setenv("DFKV_RDMA_INLINE_PUT_MAX_BYTES", std::to_string(1ull << 20).c_str(),
         1);
  LeaseNode node("inline");
  RdmaTransport rt(kMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  const std::string v = Value(256ull << 10, 0xAA);  // below the threshold
  ASSERT_TRUE(c.Put("small", v.data(), v.size()));
  EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_leaseput_ops_total"), 0)
      << "inline objects must not touch the lease datapath";

  std::string out(v.size(), '\0');
  ASSERT_TRUE(c.Get("small", &out[0], out.size()));
  EXPECT_EQ(out, v);
}

TEST_F(RdmaLeaseLoopback, MixedBatchSplitAcrossBothBuckets) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  setenv("DFKV_RDMA_INLINE_PUT_MAX_BYTES", std::to_string(1ull << 20).c_str(),
         1);
  LeaseNode node("mixed");
  RdmaTransport rt(kMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  const std::string big = Value(4ull << 20, 0x11);
  const std::string small = Value(64ull << 10, 0x22);
  // One batch with one object in each bucket: both must land, and the inline
  // one must not be carried by a 4MiB-class connection.
  const std::vector<dfkv::KvPutItem> batch{
      dfkv::KvPutItem{"kbig", big.data(), big.size()},
      dfkv::KvPutItem{"ksmall", small.data(), small.size()}};
  const auto sts = c.BatchPut(batch);
  ASSERT_EQ(sts.size(), 2u);
  EXPECT_TRUE(sts[0]);
  EXPECT_TRUE(sts[1]);

  std::string out_big(big.size(), '\0');
  ASSERT_TRUE(c.Get("kbig", &out_big[0], out_big.size()));
  EXPECT_EQ(out_big, big);
  std::string out_small(small.size(), '\0');
  ASSERT_TRUE(c.Get("ksmall", &out_small[0], out_small.size()));
  EXPECT_EQ(out_small, small);
  EXPECT_GT(CounterOf(*node.rsrv, "dfkv_rdma_leaseput_ops_total"), 0);
}

TEST_F(RdmaLeaseLoopback, MultipleLargeObjectsConcurrentWindows) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  setenv("DFKV_RDMA_INLINE_PUT_MAX_BYTES", std::to_string(1ull << 20).c_str(),
         1);
  setenv("DFKV_RDMA_DEPTH", "4", 1);
  LeaseNode node("windows");
  RdmaTransport rt(kMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  // Enough over-threshold objects to exercise multi-window lease pipelines
  // and repeated per-slot generation reuse on one connection.
  // The values must outlive BatchPut: KV items hold raw pointers, so the
  // owning strings are built first and the batch references them.
  const size_t kObjects = 12;
  std::map<std::string, std::string> sentinel;
  for (size_t i = 0; i < kObjects; ++i)
    sentinel["obj" + std::to_string(i)] =
        Value(2ull << 20, static_cast<uint8_t>(i));
  std::vector<dfkv::KvPutItem> batch;
  for (const auto& [key, v] : sentinel)
    batch.push_back(dfkv::KvPutItem{key, v.data(), v.size()});
  const auto sts = c.BatchPut(batch);
  for (size_t i = 0; i < sts.size(); ++i)
    EXPECT_TRUE(sts[i]) << "object " << i;
  EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_leaseput_active"), 0);
  const long ops = CounterOf(*node.rsrv, "dfkv_rdma_leaseput_ops_total");
  EXPECT_GE(ops, static_cast<long>(kObjects));

  for (const auto& [key, v] : sentinel) {
    std::string out(v.size(), '\0');
    ASSERT_TRUE(c.Get(key, &out[0], out.size())) << key;
    EXPECT_EQ(out, v) << key;
  }
}

TEST_F(RdmaLeaseLoopback, DisabledThresholdKeepsLegacyBehavior) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  setenv("DFKV_RDMA_INLINE_PUT_MAX_BYTES", "0", 1);
  // Without the lease datapath the object ceiling stays the declared block
  // bound, like every deployed v2.25 client (production raises it to the
  // business maximum). Mirror production so the 8MiB object remains legal.
  setenv("DFKV_RDMA_MAX_BLOCK_BYTES", "67108864", 1);
  LeaseNode node("legacy");
  RdmaTransport rt(kMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  // With the datapath disabled the 8MiB object must still succeed through
  // the traditional connection-resident class — same as a v2.25 client.
  const std::string v = Value(8ull << 20, 0x77);
  {
    // Direct transport-level check first, bypassing routing/health layers.
    std::vector<dfkv::CacheSrc> probe;
    probe.push_back(dfkv::CacheSrc{dfkv::ToBlockKey(SelfHdr(), "legacy-big"),
                                   const_cast<char*>(v.data()),
                                   v.size()});
    ASSERT_EQ(rt.CacheFrom(node.addr, probe)[0], Status::kOk);
  }
  ASSERT_TRUE(c.Put("legacy-big", v.data(), v.size()))
      << "\n[client-metrics]\n" << rt.MetricsText()
      << "\n[server-metrics]\n" << node.rsrv->MetricsText();
  EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_leaseput_ops_total"), 0);
  std::string out(v.size(), '\0');
  ASSERT_TRUE(c.Get("legacy-big", &out[0], out.size()));
  EXPECT_EQ(out, v);
}

TEST_F(RdmaLeaseLoopback, RetiredKeyCannotOverwriteReusedLease) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  LeaseNode node("revoke");
  const std::string value = Value(2u << 20, 0x62);
  std::string late(64, 'X');
  LeasePeer peer;
  ASSERT_TRUE(peer.Open(node));
  const BlockKey key = ToBlockKey(SelfHdr(), "reused");
  rdma::LeasePutReady old;
  Status status = Status::kIOError;
  ASSERT_TRUE(peer.Lease(key, value.size(), &status, &old));
  ASSERT_EQ(status, Status::kOk);
  ASSERT_TRUE(peer.Put(key, old, value));
  // Recycle the storage without granting this peer any new remote access.
  // A subsequent authorized MR may reuse the numeric rkey; this test checks
  // revocation, not indefinite uniqueness of provider-generated keys.
  auto current = RdmaLeaseServerTestPeer::Allocate(*node.rsrv, old.lease_bytes);
  ASSERT_TRUE(current);
  ASSERT_EQ(reinterpret_cast<uint64_t>(current.data()), old.write_base);
  char* target = current.data() + rdma::kV2DataOffset;
  std::memset(target, 0x5a, late.size());
  ibv_mr* mr = peer.ep.RegisterTransient(late.data(), late.size(), false);
  ASSERT_NE(mr, nullptr);
  ASSERT_TRUE(peer.ep.PostWrite(0, late.data(), late.size(), mr,
                                old.write_base + rdma::kV2DataOffset,
                                old.rkey));
  ibv_wc wc{};
  ASSERT_EQ(peer.ep.WaitComp(&wc, 1, 10000), 1);
  EXPECT_NE(wc.status, IBV_WC_SUCCESS)
      << "a completed operation's rkey must no longer authorize DMA";
  EXPECT_EQ(std::string(target, late.size()), std::string(late.size(), '\x5a'));
  peer.ep.ReleaseTransient(mr);
}

TEST_F(RdmaLeaseLoopback, LiveConnectionTrimReallocationBoundsMrs) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  LeaseNode node("trim");
  // Larger than the initial chunk: every cycle grows and then trims a chunk,
  // while the very same QP and resident receive lease remain alive.
  const std::string value = Value(40u << 20, 0x29);
  LeasePeer peer;
  ASSERT_TRUE(peer.Open(node));
  const auto pool_mrs = rdma::RcEndpoint::PoolMrActiveRegistrations();
  const auto write_mrs = rdma::RcEndpoint::LeaseWriteMrActive();
  const long committed = CounterOf(*node.rsrv,
                                    "dfkv_rdma_recv_segment_bytes");
  for (int cycle = 0; cycle != 3; ++cycle) {
    const BlockKey key = ToBlockKey(SelfHdr(), "trim-" + std::to_string(cycle));
    Status status = Status::kIOError;
    rdma::LeasePutReady lease;
    ASSERT_TRUE(peer.Lease(key, value.size(), &status, &lease));
    ASSERT_EQ(status, Status::kOk);
    EXPECT_EQ(rdma::RcEndpoint::LeaseWriteMrActive(), write_mrs + 1);
    ASSERT_TRUE(peer.Put(key, lease, value, true));
    EXPECT_EQ(rdma::RcEndpoint::LeaseWriteMrActive(), write_mrs);
    EXPECT_EQ(rdma::RcEndpoint::PoolMrActiveRegistrations(), pool_mrs)
        << "a live connection must not retain staging chunk registrations";
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    EXPECT_GE(RdmaLeaseServerTestPeer::Trim(*node.rsrv), lease.lease_bytes);
    EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_recv_segment_bytes"),
              committed);
    std::string stored;
    size_t stored_len = 0;
    ASSERT_EQ(node.srv->ProcessRequestForKey(
                  static_cast<uint8_t>(WireOp::kRange), key, 0, value.size(),
                  nullptr, 0, &stored, &stored_len), Status::kOk);
    EXPECT_EQ(stored, value) << "every SG window must survive trim/reallocation";
  }
}

TEST_F(RdmaLeaseLoopback, ExhaustionRetainsLiveLeasesUntilCompletion) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  LeaseNode node("pressure");
  const std::string value = Value(64u << 20, 0x49);
  LeasePeer first, second, third;
  ASSERT_TRUE(first.Open(node));
  ASSERT_TRUE(second.Open(node));
  ASSERT_TRUE(third.Open(node));
  const BlockKey key = ToBlockKey(SelfHdr(), "pressure");
  rdma::LeasePutReady a, b, c;
  Status status = Status::kIOError;
  ASSERT_TRUE(first.Lease(key, value.size(), &status, &a));
  ASSERT_EQ(status, Status::kOk);
  ASSERT_TRUE(second.Lease(key, value.size(), &status, &b));
  ASSERT_EQ(status, Status::kOk);
  ASSERT_TRUE(third.Lease(key, value.size(), &status, &c));
  ASSERT_EQ(status, Status::kCacheFull);
  EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_leaseput_active"), 2);
  EXPECT_LE(CounterOf(*node.rsrv, "dfkv_rdma_recv_segment_bytes"),
            192l << 20);
  ASSERT_TRUE(first.Put(key, a, value, true));
  // The rejected QP stays usable; a genuine completion, not a timeout/retry
  // policy, makes exactly one range available to it.
  ASSERT_TRUE(third.Lease(key, value.size(), &status, &c));
  ASSERT_EQ(status, Status::kOk);
  EXPECT_EQ(c.write_base, a.write_base);
  ASSERT_TRUE(third.Put(key, c, value, true));
  ASSERT_TRUE(second.Put(key, b, value, true));
  EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_leaseput_active"), 0);
}

TEST_F(RdmaLeaseLoopback, TeardownHoldsRangeUntilDelayedWriteIsFenced) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  LeaseNode node("teardown");
  std::string late(64, 'D');
  LeasePeer peer;
  std::promise<void> entered, release;
  auto entered_future = entered.get_future();
  auto release_future = release.get_future().share();
  ASSERT_TRUE(peer.Open(node));
  rdma::LeasePutReady lease;
  Status status = Status::kIOError;
  ASSERT_TRUE(peer.Lease(ToBlockKey(SelfHdr(), "abandoned"), 40u << 20,
                          &status, &lease));
  ASSERT_EQ(status, Status::kOk);
  const long held = CounterOf(*node.rsrv, "dfkv_rdma_recv_segment_used_bytes");
  ibv_mr* mr = peer.ep.RegisterTransient(late.data(), late.size(), false);
  ASSERT_NE(mr, nullptr);
  RdmaLeaseServerTestPeer::BeforeTeardown(
      *node.rsrv, [&] { entered.set_value(); release_future.wait(); });
  auto stopped = std::async(std::launch::async, [&] { node.rsrv->Stop(); });
  const bool at_fence = entered_future.wait_for(std::chrono::seconds(10)) ==
                        std::future_status::ready;
  if (at_fence) {
    // Serve has exited, but destruction is gated before QP teardown. This real
    // delayed WRITE still succeeds and therefore must target owned storage.
    EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_leaseput_active"), 1);
    EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_recv_segment_used_bytes"), held);
    const bool posted = peer.ep.PostWrite(
        0, late.data(), late.size(), mr,
        lease.write_base + rdma::kV2DataOffset, lease.rkey);
    EXPECT_TRUE(posted);
    if (posted) {
      ibv_wc wc{};
      const int got = peer.ep.WaitComp(&wc, 1, 10000);
      EXPECT_EQ(got, 1);
      if (got == 1) EXPECT_EQ(wc.status, IBV_WC_SUCCESS);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    EXPECT_EQ(RdmaLeaseServerTestPeer::Trim(*node.rsrv), 0u)
        << "unfenced inbound DMA must prevent idle trim";
  }
  release.set_value();  // always unblock teardown, including failed expectations
  stopped.get();
  ASSERT_TRUE(at_fence);
  EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_leaseput_active"), 0);
  EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_leaseput_bytes_active"), 0);
  EXPECT_EQ(rdma::RcEndpoint::LeaseWriteMrActive(), 0u);
  peer.ep.ReleaseTransient(mr);
}

TEST_F(RdmaLeaseLoopback, DynamicPullOnlyHoldsInflightMemory) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  LeaseNode node("dynamic-roundtrip");
  const BlockKey key = ToBlockKey(SelfHdr(), "dynamic");
  const std::string value = Value(8u << 20, 0x73);
  StoreForPull(node, key, value);
  const long baseline = CounterOf(*node.rsrv,
                                   "dfkv_rdma_recv_segment_used_bytes");
  PullPeer first, second;
  ASSERT_TRUE(first.Open(node));
  ASSERT_TRUE(second.Open(node));
  const long resident = baseline + first.resident.slot_size +
                        second.resident.slot_size;
  EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_recv_segment_used_bytes"),
            resident) << "dynamic bootstrap must not allocate a pull arena";
  EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_dynamic_get_active"), 0);
  Status status = Status::kIOError;
  rdma::DynamicPullReady ready;
  ASSERT_TRUE(first.Prepare(key, value.size(), &status, &ready));
  ASSERT_EQ(status, Status::kOk);
  EXPECT_EQ(ready.data_len, value.size());
  EXPECT_EQ(ready.value_len, value.size());
  EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_dynamic_get_active"), 1);
  EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_dynamic_get_bytes_active"),
            static_cast<long>(rdma::V2SlotSize(value.size())));
  EXPECT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), 1u);
  std::string out;
  ASSERT_TRUE(first.Read(ready, &out));
  EXPECT_EQ(out, value);
  ASSERT_TRUE(first.Release(ready, &status));
  ASSERT_EQ(status, Status::kOk);
  EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_dynamic_get_active"), 0);
  EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_dynamic_get_bytes_active"), 0);
  EXPECT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), 0u);
  EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_recv_segment_used_bytes"),
            resident) << "release ACK must free bytes on the idle live QP";
}

TEST_F(RdmaLeaseLoopback, DynamicPullSmallDestinationMissAndGeneration) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  LeaseNode node("dynamic-errors");
  const BlockKey key = ToBlockKey(SelfHdr(), "small");
  const BlockKey missing = ToBlockKey(SelfHdr(), "missing");
  const std::string value = Value(2u << 20, 0x61);
  StoreForPull(node, key, value);
  PullPeer peer;
  ASSERT_TRUE(peer.Open(node));
  const long resident = CounterOf(*node.rsrv,
                                   "dfkv_rdma_recv_segment_used_bytes");
  rdma::DynamicPullReady ready, next;
  Status status = Status::kIOError;
  ASSERT_TRUE(peer.Prepare(key, 4096, &status, &ready));
  ASSERT_EQ(status, Status::kOk);
  EXPECT_EQ(ready.data_len, 4096u);
  EXPECT_EQ(ready.value_len, value.size());
  auto wrong = ready;
  ++wrong.slot_generation;
  ASSERT_TRUE(peer.Release(wrong, &status));
  EXPECT_EQ(status, Status::kInvalid);
  EXPECT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), 1u);
  ASSERT_TRUE(peer.Prepare(key, 4096, &status, &next,
                           wrong.slot_generation));
  EXPECT_EQ(status, Status::kInvalid);
  std::string out;
  ASSERT_TRUE(peer.Read(ready, &out));
  EXPECT_EQ(out, value.substr(0, 4096));
  ASSERT_TRUE(peer.Release(ready, &status));
  ASSERT_EQ(status, Status::kOk);
  ASSERT_TRUE(peer.Release(ready, &status));
  EXPECT_EQ(status, Status::kInvalid);
  ASSERT_TRUE(peer.Prepare(key, value.size(), &status, &next));
  ASSERT_EQ(status, Status::kOk);
  EXPECT_NE(next.slot_generation, ready.slot_generation);
  // Even the current generation cannot retire a lease through PREPARE.
  ASSERT_TRUE(peer.Prepare(missing, value.size(), &status, &ready,
                           next.slot_generation));
  EXPECT_EQ(status, Status::kInvalid);
  EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_dynamic_get_active"), 1);
  EXPECT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), 1u);
  ASSERT_TRUE(peer.Read(next, &out));
  EXPECT_EQ(out, value);
  ASSERT_TRUE(peer.Release(next, &status));
  ASSERT_EQ(status, Status::kOk);
  // Explicit RELEASE frees the capability before a later lookup misses.
  ASSERT_TRUE(peer.Prepare(missing, value.size(), &status, &ready));
  EXPECT_EQ(status, Status::kNotFound);
  // EOF is a successful empty range, not a universal zero-request error.
  ASSERT_TRUE(peer.Prepare(key, 4096, &status, &ready, 0, 0, value.size()));
  ASSERT_EQ(status, Status::kOk);
  EXPECT_EQ(ready.data_len, 0u);
  EXPECT_EQ(ready.value_len, value.size());
  EXPECT_NE(ready.address, 0u);
  EXPECT_NE(ready.rkey, 0u);
  EXPECT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), 1u);
  ASSERT_TRUE(peer.Read(ready, &out));
  EXPECT_TRUE(out.empty());
  ASSERT_TRUE(peer.Release(ready, &status));
  ASSERT_EQ(status, Status::kOk);
  ASSERT_TRUE(peer.Prepare(key, kMsg + 1, &status, &ready));
  EXPECT_EQ(status, Status::kInvalid);
  ASSERT_TRUE(peer.Prepare(key, 4096, &status, &ready, 0, 1));
  EXPECT_EQ(status, Status::kInvalid);
  EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_dynamic_get_active"), 0);
  EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_dynamic_get_bytes_active"), 0);
  EXPECT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), 0u);
  EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_recv_segment_used_bytes"),
            resident);
}

TEST_F(RdmaLeaseLoopback, DynamicPullRetiredReadKeyIsRejected) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  LeaseNode node("dynamic-revoke");
  const BlockKey key = ToBlockKey(SelfHdr(), "revoke");
  const std::string value = Value(2u << 20, 0x16);
  StoreForPull(node, key, value);
  std::string out(64, 'X');
  PullPeer peer;
  ASSERT_TRUE(peer.Open(node));
  rdma::DynamicPullReady old, current;
  Status status = Status::kIOError;
  ASSERT_TRUE(peer.Prepare(key, value.size(), &status, &old));
  ASSERT_EQ(status, Status::kOk);
  ASSERT_TRUE(peer.Release(old, &status));
  ASSERT_EQ(status, Status::kOk);
  ibv_mr* mr = peer.ep.RegisterTransient(out.data(), out.size());
  ASSERT_NE(mr, nullptr);
  ASSERT_TRUE(peer.ep.PostRead(0, out.data(), out.size(), mr,
                               old.address, old.rkey));
  ibv_wc wc{};
  ASSERT_EQ(peer.ep.WaitComp(&wc, 1, 10000), 1);
  EXPECT_NE(wc.status, IBV_WC_SUCCESS);
  EXPECT_EQ(out, std::string(64, 'X'));
  peer.endpoint.reset();
  // Providers may recycle an MR rkey after a fresh registration. Revocation
  // is checked BEFORE a new grant; generation controls release-message ABA.
  PullPeer replacement;
  ASSERT_TRUE(replacement.Open(node));
  ASSERT_TRUE(replacement.Prepare(key, value.size(), &status, &current));
  ASSERT_EQ(status, Status::kOk);
  ASSERT_TRUE(replacement.Read(current, &out));
  EXPECT_EQ(out, value);
  ASSERT_TRUE(replacement.Release(current, &status));
  EXPECT_EQ(status, Status::kOk);
}

TEST_F(RdmaLeaseLoopback, DynamicPullExhaustionRecoversAfterRelease) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  LeaseNode node("dynamic-pressure");
  const BlockKey key = ToBlockKey(SelfHdr(), "pressure");
  const std::string value = Value(kMsg, 0x45);
  StoreForPull(node, key, value);
  PullPeer first, second, third;
  ASSERT_TRUE(first.Open(node));
  ASSERT_TRUE(second.Open(node));
  ASSERT_TRUE(third.Open(node));
  rdma::DynamicPullReady a, b, c;
  Status status = Status::kIOError;
  ASSERT_TRUE(first.Prepare(key, value.size(), &status, &a));
  ASSERT_EQ(status, Status::kOk);
  ASSERT_TRUE(second.Prepare(key, value.size(), &status, &b));
  ASSERT_EQ(status, Status::kOk);
  ASSERT_TRUE(third.Prepare(key, value.size(), &status, &c));
  ASSERT_EQ(status, Status::kCacheFull);
  EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_dynamic_get_active"), 2);
  EXPECT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), 2u);
  ASSERT_TRUE(first.Release(a, &status));
  ASSERT_EQ(status, Status::kOk);
  ASSERT_TRUE(third.Prepare(key, value.size(), &status, &c));
  ASSERT_EQ(status, Status::kOk);
  std::string out;
  ASSERT_TRUE(third.Read(c, &out));
  EXPECT_EQ(out, value);
  ASSERT_TRUE(third.Release(c, &status));
  ASSERT_EQ(status, Status::kOk);
  ASSERT_TRUE(second.Release(b, &status));
  ASSERT_EQ(status, Status::kOk);
  EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_dynamic_get_active"), 0);
  EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_dynamic_get_bytes_active"), 0);
  EXPECT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), 0u);
}

TEST_F(RdmaLeaseLoopback, DynamicPullTrimReallocationBoundsMrs) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  LeaseNode node("dynamic-trim");
  const BlockKey key = ToBlockKey(SelfHdr(), "trim");
  const std::string value = Value(40u << 20, 0x32);
  StoreForPull(node, key, value);
  PullPeer peer;
  ASSERT_TRUE(peer.Open(node));
  const auto pool_mrs = rdma::RcEndpoint::PoolMrActiveRegistrations();
  const long committed = CounterOf(*node.rsrv, "dfkv_rdma_recv_segment_bytes");
  for (int cycle = 0; cycle != 3; ++cycle) {
    rdma::DynamicPullReady ready;
    Status status = Status::kIOError;
    ASSERT_TRUE(peer.Prepare(key, value.size(), &status, &ready));
    ASSERT_EQ(status, Status::kOk);
    EXPECT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), 1u);
    std::string out;
    ASSERT_TRUE(peer.Read(ready, &out));
    EXPECT_EQ(out, value);
    ASSERT_TRUE(peer.Release(ready, &status));
    ASSERT_EQ(status, Status::kOk);
    EXPECT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), 0u);
    EXPECT_EQ(rdma::RcEndpoint::PoolMrActiveRegistrations(), pool_mrs);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    EXPECT_GE(RdmaLeaseServerTestPeer::Trim(*node.rsrv),
              rdma::V2SlotSize(value.size()));
    EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_recv_segment_bytes"), committed);
    EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_dynamic_get_bytes_active"), 0);
  }
}

TEST_F(RdmaLeaseLoopback, DynamicPullTeardownFencesDelayedRead) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  LeaseNode node("dynamic-teardown");
  const BlockKey key = ToBlockKey(SelfHdr(), "delayed");
  const std::string value = Value(40u << 20, 0x21);
  StoreForPull(node, key, value);
  PullPeer peer;
  std::promise<void> entered, release;
  auto entered_future = entered.get_future();
  auto release_future = release.get_future().share();
  ASSERT_TRUE(peer.Open(node));
  rdma::DynamicPullReady ready;
  Status status = Status::kIOError;
  ASSERT_TRUE(peer.Prepare(key, value.size(), &status, &ready));
  ASSERT_EQ(status, Status::kOk);
  const long held = CounterOf(*node.rsrv, "dfkv_rdma_recv_segment_used_bytes");
  RdmaLeaseServerTestPeer::BeforeTeardown(
      *node.rsrv, [&] { entered.set_value(); release_future.wait(); });
  auto stopped = std::async(std::launch::async, [&] { node.rsrv->Stop(); });
  const bool at_fence = entered_future.wait_for(std::chrono::seconds(10)) ==
                        std::future_status::ready;
  if (at_fence) {
    EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_dynamic_get_active"), 1);
    EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_recv_segment_used_bytes"), held);
    std::string out;
    // The hook runs before endpoint destruction. A READ can still complete
    // here, but its source must remain held and return the original bytes.
    const bool read_ok = peer.Read(ready, &out);
    if (read_ok) {
      EXPECT_EQ(out, value);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    EXPECT_EQ(RdmaLeaseServerTestPeer::Trim(*node.rsrv), 0u);
  }
  release.set_value();
  stopped.get();
  ASSERT_TRUE(at_fence);
  EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_dynamic_get_active"), 0);
  EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_dynamic_get_bytes_active"), 0);
  EXPECT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), 0u);
}

TEST_F(RdmaLeaseLoopback, DynamicPullNegotiationSupportsNewAndLegacyWire) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  LeaseNode node("dynamic-negotiation");
  PullPeer no_pull, no_dynamic, fixed, unnegotiated, no_retirement;
  EXPECT_FALSE(no_pull.Open(node, true, false));
  EXPECT_FALSE(no_dynamic.Open(node, false));
  EXPECT_FALSE(fixed.Open(node, false, true, true, false));
  EXPECT_FALSE(unnegotiated.Open(node, false, false, false, false));
  EXPECT_FALSE(no_retirement.Open(node, true, true, false, false));

  const BlockKey key = ToBlockKey(SelfHdr(), "negotiated");
  const std::string value = Value(2u << 20, 0x58);
  StoreForPull(node, key, value);
  PullPeer legacy, dynamic_only;
  ASSERT_TRUE(legacy.Open(node, true, true, true, false));  // exact 33 bytes
  ASSERT_TRUE(dynamic_only.Open(node, true, true, false));  // exact 25, no writer
  for (PullPeer* peer : {&legacy, &dynamic_only}) {
    Status status = Status::kIOError;
    rdma::DynamicPullReady ready;
    ASSERT_TRUE(peer->Prepare(key, value.size(), &status, &ready));
    ASSERT_EQ(status, Status::kOk);
    EXPECT_EQ(ready.data_len, value.size());
    EXPECT_EQ(ready.value_len, value.size());
    std::string out;
    ASSERT_TRUE(peer->Read(ready, &out));
    EXPECT_EQ(out, value);
    ASSERT_TRUE(peer->Release(ready, &status));
    ASSERT_EQ(status, Status::kOk);
    EXPECT_EQ(CounterOf(*node.rsrv, "dfkv_rdma_dynamic_get_active"), 0);
    EXPECT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), 0u);
  }
}
