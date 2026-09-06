// Staged-lease PUT datapath over a loopback RDMA device. Runs wherever
// rdma_loopback_test runs (Soft-RoCE in CI, real HCAs on development hosts);
// skips cleanly without an RDMA device. Contracts validated here need real
// verbs: WRITE_WITH_IMM windows into a per-op leased range, pool release at
// store completion, inline/lease bucket split inside one batch, and metric
// truth on both ends.
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
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {

constexpr size_t kMsg = 64ull << 20;  // like production rings' --max-msg

bool HaveRdma() { return RdmaTransport::Available(); }
std::string SelfHdr() { return "test/model"; }

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

// One cache node: KvNodeServer owns the storage; RdmaServer serves the
// bootstrap + datapath. Server-side receive budget is small enough that the
// suite could never pass with connection-resident geometry for large objects —
// the lease datapath is the only way these PUTs fit.
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

}  // namespace

TEST(RdmaLeaseLoopback, LargeObjectRoundTripsThroughStagedLease) {
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

TEST(RdmaLeaseLoopback, InlineObjectKeepsResidentPath) {
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

TEST(RdmaLeaseLoopback, MixedBatchSplitAcrossBothBuckets) {
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

TEST(RdmaLeaseLoopback, MultipleLargeObjectsConcurrentWindows) {
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

TEST(RdmaLeaseLoopback, DisabledThresholdKeepsLegacyBehavior) {
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
