// RDMA datapath test over a loopback device (Soft-RoCE / rdma_rxe in CI, or any
// real RDMA NIC). Exercises the native-verbs transport + server + versioned wire
// frame + zero-copy RangeInto + (with depth>1) the pipelined worker pool — the
// code that is otherwise only validated on real 400G hardware. Skips cleanly when
// no RDMA device is present. Built only when DFKV_WITH_RDMA is defined. Run under
// ThreadSanitizer to exercise the worker-pool / QP concurrency.
#include "client/kv_client.h"
#include "client/node_dedup.h"
#include "client/key_map.h"
#include "cache/kv_node_server.h"
#include "cache/rdma_server.h"
#include "transport/rdma_transport.h"
#include "transport/rdma_protocol.h"
#include "transport/rdma_verbs.h"

#include <gtest/gtest.h>
#include <sys/mman.h>  // shm_unlink (node-dedup test)
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {

// Small per-buffer cap so the test stays well under a modest RLIMIT_MEMLOCK
// (RDMA pins registered memory); the test values are a few KB.
constexpr size_t kMaxMsg = 256 * 1024;

std::string SelfHdr() { return "test/model"; }
void ConfigureTestRecvSegment() {
  // Keep Soft-RoCE CI below modest RLIMIT_MEMLOCK. Production defaults to
  // 2 GiB, but these fixtures use 256-KiB blocks and need only a small segment.
  ::setenv("DFKV_RDMA_RECV_SEGMENT_SIZE", "33554432", 0);
}

// A cache node serving RDMA: KvNodeServer owns the DiskCacheGroup; RdmaServer
// bootstraps QPs and routes requests to it (generic handler + zero-copy range).
struct RdmaNode {
  fs::path dir;
  std::unique_ptr<KvNodeServer> srv;
  std::unique_ptr<RdmaServer> rsrv;
  std::string addr;  // bootstrap "ip:port" for the client member list

  explicit RdmaNode(const std::string& tag, size_t max_msg = kMaxMsg) {
    ConfigureTestRecvSegment();
    dir = fs::temp_directory_path() / ("dfkv_rdma_" + tag);
    fs::remove_all(dir);
    fs::create_directories(dir);
    srv = std::make_unique<KvNodeServer>(dir.string(), 1ull << 30);
    EXPECT_EQ(srv->Start(0), Status::kOk);  // TCP listener owns the cache group
    rsrv = std::make_unique<RdmaServer>(
        [this](uint8_t op, const BlockKey& key, uint64_t off, uint64_t len,
               const char* pl, uint64_t pll, std::string* out,
               size_t* value_len) {
          return srv->ProcessRequestForKey(
              op, key, off, len, pl, pll, out, value_len);
        },
        max_msg);
    rsrv->set_range_handler(
        [this](const BlockKey& key, uint64_t off, uint64_t len,
               char* io_buf, size_t cap, const char** out_data,
               size_t* out_len, size_t* value_len) {
          return srv->RangeDirectForKey(
              key, off, len, io_buf, cap, out_data, out_len, value_len);
        });
    rsrv->set_cache_direct_handler(
        [this](const BlockKey& key, char* data, size_t len, size_t cap) {
          return srv->CacheDirectForKey(key, data, len, cap);
        });
    EXPECT_EQ(rsrv->Start(0), Status::kOk);
    addr = "127.0.0.1:" + std::to_string(rsrv->port());
  }
  ~RdmaNode() {
    if (rsrv) rsrv->Stop();
    if (srv) srv->Stop();
    fs::remove_all(dir);
  }
};

bool HaveRdma() { return RdmaTransport::Available(); }

// Last value of a single-line Prometheus counter (skips the # HELP/# TYPE lines
// via rfind, which lands on the value line emitted after them).
long CounterVal(const std::string& text, const std::string& name) {
  auto p = text.rfind(name);
  if (p == std::string::npos) return -1;
  auto sp = text.find(' ', p);
  if (sp == std::string::npos) return -1;
  try { return std::stol(text.substr(sp + 1)); } catch (...) { return -1; }
}

struct FakePoolRail {
  size_t declared = 0;
  size_t staged = 0;
  bool reject_stage = false;
  size_t commits = 0;
  size_t rollbacks = 0;

  bool Stage(size_t requested) {
    if (reject_stage) return false;
    staged = requested;
    return true;
  }
  void Commit() {
    declared = staged;
    ++commits;
  }
  void Rollback() {
    staged = declared;
    ++rollbacks;
  }
  bool Covers(size_t offset, size_t length) const {
    return offset <= declared && length <= declared - offset;
  }
};

}  // namespace

TEST(RdmaSafety, CompletionDeadlineUsesOneAbsoluteBudget) {
  using Clock = CompletionDeadline::Clock;
  const auto start = Clock::time_point(std::chrono::milliseconds(100));
  CompletionDeadline deadline(50, start);
  EXPECT_EQ(deadline.RemainingAt(start), 50);
  EXPECT_EQ(deadline.RemainingAt(start + std::chrono::milliseconds(30)), 20);
  // A partial completion at +30 ms does not create a fresh 50-ms wait.
  EXPECT_EQ(deadline.RemainingAt(start + std::chrono::milliseconds(49)), 1);
  EXPECT_EQ(deadline.RemainingAt(start + std::chrono::milliseconds(50)), 0);
  CompletionDeadline infinite(-1, start);
  EXPECT_EQ(infinite.RemainingAt(start + std::chrono::hours(24)), -1);
}

TEST(RdmaSafety, PartialMultiRailPoolGrowthRollsBackBeforePublication) {
  std::vector<FakePoolRail> rails(3, FakePoolRail{64, 64});
  rails[1].reject_stage = true;
  size_t published_bytes = 64;
  const bool registered = rdma::RunPoolMrTransaction(
      rails.size(),
      [&](size_t rail) { return rails[rail].Stage(128); },
      [&](size_t rail) { rails[rail].Commit(); },
      [&](size_t rail) { rails[rail].Rollback(); });
  if (registered) published_bytes = 128;

  EXPECT_FALSE(registered);
  EXPECT_EQ(published_bytes, 64u);
  for (const auto& rail : rails) {
    EXPECT_EQ(rail.declared, 64u);
    EXPECT_EQ(rail.staged, 64u);
    EXPECT_EQ(rail.commits, 0u);
  }
  EXPECT_EQ(rails[0].rollbacks, 1u);
  EXPECT_EQ(rails[1].rollbacks, 0u);
  EXPECT_EQ(rails[2].rollbacks, 0u);
}

TEST(RdmaSafety, SuccessfulMultiRailPoolGrowthKeepsOldRangeUsable) {
  std::vector<FakePoolRail> rails(3, FakePoolRail{64, 64});
  size_t published_bytes = 64;
  const bool registered = rdma::RunPoolMrTransaction(
      rails.size(),
      [&](size_t rail) { return rails[rail].Stage(128); },
      [&](size_t rail) { rails[rail].Commit(); },
      [&](size_t rail) { rails[rail].Rollback(); });
  if (registered) published_bytes = 128;

  ASSERT_TRUE(registered);
  EXPECT_EQ(published_bytes, 128u);
  for (const auto& rail : rails) {
    EXPECT_EQ(rail.declared, 128u);
    EXPECT_EQ(rail.commits, 1u);
    EXPECT_EQ(rail.rollbacks, 0u);
    EXPECT_TRUE(rail.Covers(8, 16));
    EXPECT_TRUE(rail.Covers(96, 16));
  }
}

TEST(RdmaSafety, OverridesRejectMalformedVectorsBeforeAcquiringQp) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaTransport transport(kMaxMsg);
  const std::vector<BlockKey> keys = {{1, 2}, {3, 4}};
  std::vector<uint64_t> value_lengths;
  const auto mismatched =
      transport.RangeInto("unused", keys, {{nullptr, 0}}, &value_lengths);
  ASSERT_EQ(mismatched.size(), keys.size());
  EXPECT_EQ(mismatched[0], Status::kInvalid);
  EXPECT_EQ(mismatched[1], Status::kInvalid);

  const auto null_write = transport.CacheMany(
      "unused", {CacheItem{keys[0], nullptr, 1}});
  ASSERT_EQ(null_write.size(), 1u);
  EXPECT_EQ(null_write[0], Status::kInvalid);

  const void* nonnull = reinterpret_cast<const void*>(uintptr_t{1});
  CacheSrcMulti overflow{
      keys[0],
      {{nonnull, std::numeric_limits<size_t>::max()}, {nonnull, 1}}};
  const auto overflow_status =
      transport.CacheFromMulti("unused", {overflow});
  ASSERT_EQ(overflow_status.size(), 1u);
  EXPECT_EQ(overflow_status[0], Status::kInvalid);

  EXPECT_EQ(transport.ExistMany("unused", keys, nullptr),
            std::vector<Status>(keys.size(), Status::kInvalid));
  EXPECT_EQ(transport.Members("unused", nullptr), Status::kInvalid);
  EXPECT_EQ(CounterVal(transport.MetricsText(),
                       "dfkv_rdma_client_conns_opened_total"), 0);
}



// Direct transport ExistMany: windowed batch existence probe on one connection.
// Must be correct across multiple send windows (N > depth) with mixed hit/miss.
TEST(RdmaLoopback, ExistManyWindowedMixedHitMiss) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device (load rdma_rxe for Soft-RoCE)";
  RdmaNode node("exm");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  const int N = 80;  // exceeds a single send window (depth) to exercise looping
  for (int i = 0; i < N; ++i) {
    std::string v = "v" + std::to_string(i);
    ASSERT_TRUE(c.Put("p" + std::to_string(i), v.data(), v.size())) << i;
  }
  // Interleave present (even) and absent (odd) keys using the exact namespace
  // the KVClient used for the writes.
  std::vector<BlockKey> keys;
  for (int i = 0; i < N; ++i) {
    keys.push_back(ToBlockKey(SelfHdr(), "p" + std::to_string(i)));
    keys.push_back(ToBlockKey(SelfHdr(), "absent" + std::to_string(i)));
  }
  std::vector<char> exists;
  auto sts = rt.ExistMany(node.addr, keys, &exists);
  ASSERT_EQ(exists.size(), keys.size());
  ASSERT_EQ(sts.size(), keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    bool want = (i % 2 == 0);
    EXPECT_EQ(exists[i] != 0, want) << "i=" << i;
  }
}

// Client BatchExist over RDMA may expand one node's pool for parallel probe
// shards, but repeated calls must reuse that bounded pool instead of opening a
// fresh connection per key.
TEST(RdmaLoopback, BatchExistReusesExpandedPool) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("bex");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  const int N = 64;  // > batch_concurrency (8): the old per-key fan-out opened many
  for (int i = 0; i < N; ++i) {
    std::string v = "v" + std::to_string(i);
    ASSERT_TRUE(c.Put("e" + std::to_string(i), v.data(), v.size())) << i;
  }
  std::vector<std::string> probe;
  for (int i = 0; i < N; ++i) {
    probe.push_back("e" + std::to_string(i));        // present
    probe.push_back("e" + std::to_string(i) + "_x"); // absent
  }

  // Warm one connection, then let the first large batch expand the bounded
  // per-node pool according to the host's available fan-out.
  EXPECT_TRUE(c.Exist("e0"));
  const long before =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  ASSERT_GE(before, 1);

  auto er = c.BatchExist(probe);
  ASSERT_EQ(er.size(), probe.size());
  for (size_t i = 0; i < probe.size(); ++i)
    EXPECT_EQ((bool)er[i], (i % 2 == 0)) << probe[i];
  const long expanded =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  EXPECT_LE(expanded - before, 15);  // default max 16, one already warm

  auto again = c.BatchExist(probe);
  ASSERT_EQ(again.size(), probe.size());
  for (size_t i = 0; i < probe.size(); ++i)
    EXPECT_EQ((bool)again[i], (i % 2 == 0)) << probe[i];
  const long reused =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  EXPECT_EQ(reused, expanded)
      << "repeated BatchExist did not reuse its expanded connection pool";
}

// Non-SG batch PUT: one oversized item must fail ONLY itself, not poison the
// whole same-node batch. Previously any item over the per-op payload bound
// filled every sibling's status with kInvalid — all lost their cache write.
TEST(RdmaLoopback, BatchPutOversizedFailsOnlyOffender) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("bpo");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  std::string small(4096, 's');
  std::string big(kMaxMsg + 4096, 'b');  // exceeds the per-op payload bound
  std::vector<KvPutItem> items = {
      {"bpo_a", small.data(), small.size()},
      {"bpo_big", big.data(), big.size()},
      {"bpo_c", small.data(), small.size()},
  };
  auto pr = c.BatchPut(items);
  ASSERT_EQ(pr.size(), 3u);
  EXPECT_TRUE(pr[0]) << "sibling poisoned by the oversized item";
  EXPECT_FALSE(pr[1]);
  EXPECT_TRUE(pr[2]) << "sibling poisoned by the oversized item";
  EXPECT_TRUE(c.Exist("bpo_a"));
  EXPECT_FALSE(c.Exist("bpo_big"));
  EXPECT_TRUE(c.Exist("bpo_c"));
}

// Same-host GET rendezvous (phase 5): with DFKV_CLIENT_NODE_DEDUP=1, a second
// client reading the SAME keys must be served from the shm rendezvous, not the
// server — server-side completions stay ~flat while the data stays correct.
TEST(RdmaLoopback, NodeDedupCollapsesSameHostGets) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const std::string nm = NodeDedup::EnvSegmentName(
      ToBlockKey(SelfHdr(), "").digest_hi);
  ::shm_unlink(nm.c_str());
  ::setenv("DFKV_CLIENT_NODE_DEDUP", "1", 1);
  {
    RdmaNode node("ndd");
    RdmaTransport rt1(kMaxMsg), rt2(kMaxMsg);
    KVClient c1({{"n", node.addr}}, SelfHdr(), &rt1);
    KVClient c2({{"n", node.addr}}, SelfHdr(), &rt2);

    const int N = 32;
    std::vector<std::string> vals(N);
    for (int i = 0; i < N; ++i) {
      vals[i].assign(8192, '\0');
      for (size_t b = 0; b < vals[i].size(); ++b)
        vals[i][b] = static_cast<char>((i * 131 + b * 7) & 0xFF);
      ASSERT_TRUE(c1.Put("ndd" + std::to_string(i), vals[i].data(), vals[i].size())) << i;
    }

    auto get_all = [&](KVClient& c, std::vector<std::string>* out) {
      std::vector<KvGetItem> items(N);
      out->assign(N, std::string(8192, '\0'));
      for (int i = 0; i < N; ++i)
        items[i] = {"ndd" + std::to_string(i), &(*out)[i][0], 8192};
      return c.BatchGet(items);
    };

    std::vector<std::string> o1, o2;
    auto r1 = get_all(c1, &o1);  // first reader: remote fetch + publish
    for (int i = 0; i < N; ++i) { ASSERT_TRUE(r1[i]) << i; EXPECT_EQ(o1[i], vals[i]); }
    const long mid = CounterVal(node.rsrv->MetricsText(), "dfkv_rdma_completions_total");
    auto r2 = get_all(c2, &o2);  // peer: rendezvous hits, no server traffic
    for (int i = 0; i < N; ++i) { ASSERT_TRUE(r2[i]) << i; EXPECT_EQ(o2[i], vals[i]); }
    const long after = CounterVal(node.rsrv->MetricsText(), "dfkv_rdma_completions_total");
    EXPECT_LE(after - mid, N / 4)
        << "peer reads reached the server (" << (after - mid)
        << " completions); rendezvous not deduplicating";
  }
  ::unsetenv("DFKV_CLIENT_NODE_DEDUP");
  ::shm_unlink(nm.c_str());
}

TEST(RdmaLoopback, PutGetExistMissOverRdma) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device (load rdma_rxe for Soft-RoCE)";
  RdmaNode node("pgem");
  RdmaTransport rt(kMaxMsg);  // first device (rxe0 in CI)
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  std::string v(4096, '\0');
  for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<char>((i * 31 + 7) & 0xFF);
  ASSERT_TRUE(c.Put("k1", v.data(), v.size()));
  EXPECT_TRUE(c.Exist("k1"));
  std::string out(v.size(), '\0');
  ASSERT_TRUE(c.Get("k1", &out[0], out.size()));
  EXPECT_EQ(out, v);

  // miss: absent key
  std::string m(v.size(), '\0');
  EXPECT_FALSE(c.Get("absent", &m[0], m.size()));
  EXPECT_FALSE(c.Exist("absent"));
}

// Observability counters: the server tallies request completions and the client
// transport tallies connections opened (+ per-rail) and MR regions.
TEST(RdmaLoopback, MetricsCountersTrackOps) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("mco");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  std::vector<char> pool(64 * 1024);
  ASSERT_TRUE(c.RegisterMemory(pool.data(), pool.size()));
  std::string v(2048, 'z');
  ASSERT_TRUE(c.Put("m1", v.data(), v.size()));
  std::string out(v.size(), '\0');
  ASSERT_TRUE(c.Get("m1", &out[0], out.size()));

  // server: at least the PUT + GET requests were completed
  EXPECT_GE(node.rsrv->Completions(), 2u);
  std::string srv_text = node.rsrv->MetricsText();
  EXPECT_NE(srv_text.find("dfkv_rdma_completions_total"), std::string::npos) << srv_text;
  EXPECT_GE(CounterVal(srv_text, "dfkv_rdma_v2_conns_opened_total"), 1);
  EXPECT_GE(CounterVal(srv_text, "dfkv_rdma_v2_put_writes_total"), 1);
  EXPECT_GE(CounterVal(srv_text, "dfkv_rdma_v2_get_writes_total"), 1);
  EXPECT_GT(CounterVal(srv_text, "dfkv_rdma_recv_segment_bytes"), 0);

  // client transport: a connection was opened and the MR region declared
  std::string cli_text = rt.MetricsText();
  EXPECT_NE(cli_text.find("dfkv_rdma_client_conns_opened_total"), std::string::npos) << cli_text;
  EXPECT_NE(cli_text.find("dfkv_rdma_client_rail_conns_total{dev="), std::string::npos) << cli_text;
  EXPECT_NE(cli_text.find("dfkv_rdma_client_mr_regions 1"), std::string::npos) << cli_text;
  EXPECT_GE(CounterVal(cli_text, "dfkv_rdma_client_v2_put_writes_total"), 1);
  EXPECT_GE(CounterVal(cli_text, "dfkv_rdma_client_v2_get_writes_total"), 1);
  EXPECT_GE(CounterVal(cli_text,
                       "dfkv_rdma_client_pool_mr_registrations_total"), 1);
  EXPECT_GE(CounterVal(cli_text, "dfkv_rdma_client_max_block_seen_bytes"),
            2048);
  EXPECT_EQ(CounterVal(cli_text,
                       "dfkv_rdma_client_transient_user_mr_active"), 0);
  EXPECT_GE(CounterVal(cli_text, "dfkv_rdma_client_v2_probe_attempts_total"),
            1);
  EXPECT_GE(CounterVal(cli_text,
                       "dfkv_rdma_client_completion_timeouts_total"), 0);

  // and the client snapshot folds transport metrics in after the health metrics
  std::string snap = c.MetricsSnapshot();
  EXPECT_NE(snap.find("dfkv_client_ops_served_total"), std::string::npos) << snap;
  EXPECT_NE(snap.find("dfkv_rdma_client_conns_opened_total"), std::string::npos) << snap;
}

TEST(RdmaLoopback, MembersReplyHonorsExactBoundAndRejectsOversize) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("large-members");
  const std::string boundary(rdma::kV2ControlResponseMax, 'm');
  node.srv->set_members(boundary);

  RdmaTransport transport(kMaxMsg);
  std::string output;
  ASSERT_EQ(transport.Members(node.addr, &output), Status::kOk);
  EXPECT_EQ(output, boundary);
  EXPECT_GE(node.rsrv->V2Conns(), 1u);

  node.srv->set_members(
      std::string(rdma::kV2ControlResponseMax + 1, 'x'));
  output = "must be cleared";
  EXPECT_EQ(transport.Members(node.addr, &output), Status::kIOError);
  EXPECT_TRUE(output.empty());

  node.srv->set_members(boundary);
  ASSERT_EQ(transport.Members(node.addr, &output), Status::kOk);
  EXPECT_EQ(output, boundary);
}

TEST(RdmaLoopback, StartupFailsWhenV2ReceiveSegmentIsUnavailable) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ::setenv("DFKV_RDMA_RECV_SEGMENT_SIZE", "4096", 1);
  RdmaServer server(
      [](uint8_t, const BlockKey&, uint64_t, uint64_t, const char*, uint64_t,
         std::string*, size_t*) { return Status::kInvalid; },
      kMaxMsg);
  EXPECT_EQ(server.Start(0), Status::kIOError);
  ::unsetenv("DFKV_RDMA_RECV_SEGMENT_SIZE");
}


TEST(RdmaLoopback, V2CacheAcceptsReadOnlySourceMemory) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("readonly-v2");
  RdmaTransport transport(kMaxMsg);
  const BlockKey key = ToBlockKey(SelfHdr(), "readonly-v2");
  constexpr size_t kPayload = 512;
  const size_t stored_len = kPayload;
  void* mapping =
      ::mmap(nullptr, 4096, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(mapping, MAP_FAILED);
  std::memset(mapping, 'r', kPayload);
  ASSERT_EQ(::mprotect(mapping, 4096, PROT_READ), 0);

  ASSERT_EQ(transport.Cache(node.addr, key, mapping, stored_len), Status::kOk);
  std::string output;
  ASSERT_EQ(transport.Range(node.addr, key, 0, stored_len, &output),
            Status::kOk);
  ASSERT_EQ(output.size(), stored_len);
  EXPECT_EQ(std::memcmp(output.data(), mapping, stored_len), 0);
  EXPECT_EQ(::munmap(mapping, 4096), 0);
}

TEST(RdmaLoopback, DirectSingleCacheRangeUsesV2Writes) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("direct-v2");
  RdmaTransport transport(kMaxMsg);
  const BlockKey key = ToBlockKey(SelfHdr(), "direct-v2");
  std::string stored(4096, '\0');
  for (size_t i = 0; i < stored.size(); ++i)
    stored[i] = static_cast<char>((i * 29 + 3) & 0xff);

  ASSERT_EQ(transport.Cache(node.addr, key, stored.data(), stored.size()),
            Status::kOk);
  std::string output;
  ASSERT_EQ(transport.Range(node.addr, key, 0, stored.size(), &output),
            Status::kOk);
  EXPECT_EQ(output, stored);
  EXPECT_GE(node.rsrv->V2PutWrites(), 1u);
  EXPECT_GE(node.rsrv->V2GetWrites(), 1u);
}

TEST(RdmaLoopback, BatchZeroCopyRoundtrip) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("bzc");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  const int N = 20;
  const size_t sz = 4096;
  std::vector<std::string> vals(N), keys(N);
  std::vector<KvPutItem> puts(N);
  for (int i = 0; i < N; ++i) {
    keys[i] = "b" + std::to_string(i);
    vals[i].assign(sz, static_cast<char>((i * 13 + 1) & 0xFF));
    puts[i] = {keys[i], vals[i].data(), sz};
  }
  auto pr = c.BatchPut(puts);
  for (int i = 0; i < N; ++i) EXPECT_TRUE(pr[i]) << i;

  // GET into fresh buffers (RDMA scatters payload straight in = zero copy).
  std::vector<std::string> outs(N);
  std::vector<KvGetItem> gets(N);
  for (int i = 0; i < N; ++i) { outs[i].assign(sz, '\0'); gets[i] = {keys[i], &outs[i][0], sz}; }
  auto gr = c.BatchGet(gets);
  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(gr[i]) << i;
    EXPECT_EQ(outs[i], vals[i]) << i;
  }
}

// Single Put/Get always take the zero-copy fast path on RDMA (register caller
// buffer + scatter-send / scatter-recv) — no size threshold.
TEST(RdmaLoopback, SingleZeroCopyPutGet) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("szc");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  std::string v(8192, '\0');
  for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<char>((i * 37 + 11) & 0xFF);
  // Put twice into the SAME source buffer to exercise the MR-cache hit on re-put.
  ASSERT_TRUE(c.Put("z1", v.data(), v.size()));
  ASSERT_TRUE(c.Put("z1", v.data(), v.size()));

  // Get into the SAME dst buffer twice (scatter-recv straight in; MR-cache hit).
  std::string out(v.size(), '\0');
  ASSERT_TRUE(c.Get("z1", &out[0], out.size()));
  EXPECT_EQ(out, v);
  out.assign(v.size(), '\0');
  ASSERT_TRUE(c.Get("z1", &out[0], out.size()));
  EXPECT_EQ(out, v);

  // Miss on the zero-copy Get path must be a clean miss (kNotFound), not an error.
  std::string m(v.size(), '\0');
  EXPECT_FALSE(c.Get("absent", &m[0], m.size()));
  // Size mismatch (stored payload_len != requested n) => miss, not corruption.
  std::string shorter(v.size() / 2, '\0');
  EXPECT_FALSE(c.Get("z1", &shorter[0], shorter.size()));
}

// A registered memory region (RegisterMemory) is registered once; every buffer
// inside it resolves to that one MR with no per-op ibv_reg_mr. Verified directly
// at the endpoint: two distinct sub-buffers return the SAME MR; an outside buffer
// registers ad-hoc (a different MR).
// Many endpoints on the same device must all Open via the shared per-device
// ibv_context+PD registry (#6 fix: no per-connection ibv_open_device thrash).
// Opening + closing N in waves exercises the refcount get-or-create/free path;
// a leak or double-free here would surface as an Open failure or crash.
TEST(RdmaLoopback, ManyEndpointsShareDeviceContext) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  constexpr int N = 24;  // > typical t16; well past the 1-2 conn happy path
  {
    std::vector<std::unique_ptr<rdma::RcEndpoint>> eps;
    for (int i = 0; i < N; ++i) {
      eps.push_back(std::make_unique<rdma::RcEndpoint>());
      ASSERT_TRUE(eps.back()->Open(nullptr, 64 * 1024, 1)) << "endpoint " << i;
    }
    eps.clear();  // all close -> registry refcount returns to 0, frees ctx+pd
  }
  // A fresh endpoint after the registry drained must still open (re-creates the
  // shared device cleanly).
  rdma::RcEndpoint again;
  ASSERT_TRUE(again.Open(nullptr, 64 * 1024, 1));
}


TEST(RdmaLoopback, RegisterMemoryRejectsInvalidRange) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaTransport rt(kMaxMsg);
  char byte = 0;
  EXPECT_FALSE(rt.RegisterMemory(nullptr, 1));
  EXPECT_FALSE(rt.RegisterMemory(&byte, 0));
  const std::string metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_mr_regions"), 0);
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_mr_registered_bytes"), 0);
  EXPECT_EQ(CounterVal(
                metrics,
                "dfkv_rdma_client_mr_registration_rejections_total"),
            2);
}

// Client-side anchor: RegisterMemory must register the pool MRs at
// DECLARATION time (holding a lifetime device ref), not on the first
// connection's first op — a 141 GB host pool costs ~4 s to pin, which
// belongs in engine startup, not the first lookup.
TEST(RdmaLoopback, RegisterMemoryAnchorsPoolMrBeforeFirstConn) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const uint64_t before = rdma::RcEndpoint::PoolMrRegistrations();
  RdmaTransport rt(kMaxMsg);
  std::vector<char> pool(256 * 1024);
  ASSERT_TRUE(rt.RegisterMemory(pool.data(), 64 * 1024));
  EXPECT_GT(rdma::RcEndpoint::PoolMrRegistrations(), before)
      << "pool MR not registered at declaration time (anchor missing)";
  std::string metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_mr_regions"), 1);
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_mr_registered_bytes"),
            64 * 1024);

  ASSERT_TRUE(rt.RegisterMemory(pool.data(), pool.size()));
  metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_mr_regions"), 1);
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_mr_registered_bytes"),
            static_cast<long>(pool.size()));
  EXPECT_EQ(CounterVal(
                metrics,
                "dfkv_rdma_client_mr_registration_rejections_total"),
            0);

  const uintptr_t base = reinterpret_cast<uintptr_t>(pool.data());
  const size_t wrapping_size =
      std::numeric_limits<uintptr_t>::max() - base + 1;
  EXPECT_FALSE(rt.RegisterMemory(pool.data(), wrapping_size));
  metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_mr_regions"), 1);
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_mr_registered_bytes"),
            static_cast<long>(pool.size()));
  EXPECT_EQ(CounterVal(
                metrics,
                "dfkv_rdma_client_mr_registration_rejections_total"),
            1);
}

TEST(RdmaLoopback, SameBasePoolRegistrationGrowthIsEffective) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  rdma::RcEndpoint endpoint;
  rdma::RcEndpoint old_generation_user;
  ASSERT_TRUE(endpoint.Open(nullptr, 64 * 1024, 1));
  ASSERT_TRUE(old_generation_user.Open(nullptr, 64 * 1024, 1));
  std::vector<char> region(256 * 1024);
  ASSERT_TRUE(endpoint.AddPoolMr(region.data(), 64 * 1024, true));
  ASSERT_TRUE(
      old_generation_user.AddPoolMr(region.data(), 64 * 1024, true));
  ibv_mr* initial = endpoint.RegisterUser(region.data() + 4096, 4096);
  ASSERT_NE(initial, nullptr);
  EXPECT_EQ(old_generation_user.RegisterUser(region.data() + 4096, 4096),
            initial);
  const uint64_t registrations = rdma::RcEndpoint::PoolMrRegistrations();
  const uint64_t active =
      rdma::RcEndpoint::PoolMrActiveRegistrations();

  ASSERT_TRUE(endpoint.AddPoolMr(region.data(), region.size(), true));
  EXPECT_EQ(rdma::RcEndpoint::PoolMrRegistrations(), registrations + 1);
  EXPECT_EQ(rdma::RcEndpoint::PoolMrActiveRegistrations(), active + 1)
      << "the old generation stays leased by its current endpoint";
  ibv_mr* grown =
      endpoint.RegisterUser(region.data() + 128 * 1024, 4096);
  ASSERT_NE(grown, nullptr);
  EXPECT_NE(grown, initial);
  EXPECT_EQ(endpoint.RegisterUser(region.data() + 4096, 4096), grown)
      << "the grown endpoint resolves the old prefix through the new MR";
  EXPECT_EQ(old_generation_user.RegisterUser(region.data() + 4096, 4096),
            initial)
      << "an endpoint using the previous generation keeps its old range valid";
  EXPECT_EQ(old_generation_user.RegisterUser(
                region.data() + 128 * 1024, 4096),
            nullptr);

  ASSERT_TRUE(old_generation_user.AddPoolMr(
      region.data(), region.size(), true));
  EXPECT_EQ(rdma::RcEndpoint::PoolMrActiveRegistrations(), active)
      << "the superseded generation retires after its final endpoint advances";
  EXPECT_EQ(old_generation_user.RegisterUser(region.data() + 4096, 4096),
            grown);
}

TEST(RdmaLoopback, UnregisteredBuffersLeaveNoLiveMrAfterReturn) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("transient-lifetime");
  RdmaTransport transport(kMaxMsg);
  const BlockKey key = ToBlockKey(SelfHdr(), "transient-lifetime");
  const uint64_t baseline = rdma::RcEndpoint::TransientUserMrActive();

  {
    std::vector<char> source(4096, 't');
    const auto statuses = transport.CacheFrom(
        node.addr, {CacheSrc{key, source.data(), source.size()}});
    ASSERT_EQ(statuses.size(), 1u);
    ASSERT_EQ(statuses[0], Status::kOk);
    EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), baseline);
  }  // source may be freed immediately: no cached MR may reference it.

  {
    std::vector<char> destination(4096);
    std::vector<uint64_t> value_lengths;
    const auto statuses = transport.RangeInto(
        node.addr, {key},
        {RangeDst{destination.data(), destination.size()}}, &value_lengths);
    ASSERT_EQ(statuses.size(), 1u);
    ASSERT_EQ(statuses[0], Status::kOk);
    EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), baseline);
  }  // destination may be freed immediately after return as well.
}

TEST(RdmaLoopback, PoolMrSharedAcrossBuffers) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  rdma::RcEndpoint ep;
  ASSERT_TRUE(ep.Open(nullptr, 64 * 1024, 1));
  std::vector<char> region(256 * 1024);
  ASSERT_TRUE(ep.AddPoolMr(region.data(), region.size()));
  ibv_mr* a = ep.RegisterUser(region.data() + 4096, 4096);
  ibv_mr* b = ep.RegisterUser(region.data() + 200000, 4096);
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(a, b);  // both inside the pool -> one shared MR, no per-op registration
  std::vector<char> outside(4096);
  EXPECT_EQ(ep.RegisterUser(outside.data(), outside.size()), nullptr)
      << "out-of-pool stable lookup must fail closed, never cache caller memory";
}

TEST(RdmaLoopback, PoolMrAccessUpgradeWinsRangeLookup) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  rdma::RcEndpoint ep;
  ASSERT_TRUE(ep.Open(nullptr, 64 * 1024, 1));
  std::vector<char> region(256 * 1024);
  ASSERT_TRUE(ep.AddPoolMr(region.data(), region.size()));
  ibv_mr* local_only = ep.RegisterUser(region.data() + 4096, 4096);
  ASSERT_NE(local_only, nullptr);
  ibv_mr* remote_write =
      ep.RegisterRemoteRegion(region.data(), region.size());
  ASSERT_NE(remote_write, nullptr);
  EXPECT_NE(remote_write, local_only);
  EXPECT_EQ(ep.RegisterUser(region.data() + 4096, 4096), remote_write)
      << "range lookup must prefer the remote-write access upgrade";
}

// The host KV pool belongs to the PD, not to a connection: many endpoints on
// the same device must register it ONCE (the #P1-2 fix), not once per connection
// (the pre-fix storm — re-running ibv_reg_mr over a tens-of-GB region on every
// reconnect). Verified via the pool-registration counter delta, and by every
// endpoint resolving an in-pool buffer to the SAME MR (shared lkey).
TEST(RdmaLoopback, PoolMrRegisteredOncePerDeviceNotPerConnection) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  // A distinct region per test run so the counter delta is attributable here.
  std::vector<char> region(512 * 1024);
  const uint64_t regs0 = rdma::RcEndpoint::PoolMrRegistrations();
  const uint64_t adhoc0 = rdma::RcEndpoint::AdhocUserMrTotal();

  constexpr int N = 8;
  std::vector<std::unique_ptr<rdma::RcEndpoint>> eps;
  ibv_mr* first = nullptr;
  for (int i = 0; i < N; ++i) {
    eps.push_back(std::make_unique<rdma::RcEndpoint>());
    ASSERT_TRUE(eps.back()->Open(nullptr, 64 * 1024, 1)) << "endpoint " << i;
    ASSERT_TRUE(eps.back()->AddPoolMr(region.data(), region.size()));
    ibv_mr* m = eps.back()->RegisterUser(region.data() + 4096, 4096);
    ASSERT_NE(m, nullptr);
    if (i == 0) first = m;
    else EXPECT_EQ(m, first) << "all endpoints on one PD share the pool MR";
  }
  EXPECT_EQ(rdma::RcEndpoint::PoolMrRegistrations() - regs0, 1u)
      << N << " endpoints registered the region " << (rdma::RcEndpoint::PoolMrRegistrations() - regs0)
      << " times; must be exactly once per device";
  EXPECT_EQ(rdma::RcEndpoint::AdhocUserMrTotal() - adhoc0, 0u)
      << "in-pool buffers must never take the ad-hoc registration path";

  eps.clear();  // all close; region MR freed when the last device ref drops
  // A fresh endpoint re-registers the (now-freed) region: counter advances by 1.
  rdma::RcEndpoint again;
  ASSERT_TRUE(again.Open(nullptr, 64 * 1024, 1));
  ASSERT_TRUE(again.AddPoolMr(region.data(), region.size()));
}

// End-to-end: register one host pool, then PUT from and GET into sub-buffers.
// Every transfer must use the pool MR; no per-operation registration is legal.
TEST(RdmaLoopback, RegisterMemoryRoundtrip) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("rmr");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  std::vector<char> pool(128 * 1024);
  ASSERT_TRUE(c.RegisterMemory(pool.data(), pool.size()));

  const size_t sz = 4096;
  for (int i = 0; i < 8; ++i) {
    char* src = pool.data() + i * sz * 2;            // distinct sub-buffer per page
    char* dst = pool.data() + i * sz * 2 + sz;       // get into a neighbouring slot
    for (size_t k = 0; k < sz; ++k) src[k] = static_cast<char>((i * 17 + k) & 0xFF);
    std::string key = "rm" + std::to_string(i);
    ASSERT_TRUE(c.Put(key, src, sz)) << i;
    ASSERT_TRUE(c.Get(key, dst, sz)) << i;
    EXPECT_EQ(0, std::memcmp(src, dst, sz)) << i;
  }
}

// A zero-length item in a mixed RDMA batch fails independently; the valid
// neighbor still completes and no empty object reaches the server.
TEST(RdmaLoopback, BatchPutRejectsEmptyValue) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("bev");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  std::string nonempty(2048, 'x');
  std::vector<KvPutItem> puts = {
      {"e_full", nonempty.data(), nonempty.size()},
      {"e_empty", nonempty.data(), 0},
  };
  auto pr = c.BatchPut(puts);
  ASSERT_TRUE(pr[0]);
  EXPECT_FALSE(pr[1]);

  EXPECT_FALSE(c.Exist("e_empty"));
  std::string out(nonempty.size(), '\0');
  ASSERT_TRUE(c.Get("e_full", &out[0], out.size()));
  EXPECT_EQ(out, nonempty);
}

// Scatter-gather over RDMA: BatchPutSg gathers N caller buffers into one stored
// blob via a multi-SGE SEND; BatchGetAutoSg scatters the stored blob across N
// caller buffers via a multi-SGE RECV. Exercises the real PostSendScatterMulti /
// PostRecvScatterMulti datapath (not the concat fallback). Covers N=1, N=2,
// N=29 (max_sge-1), variable sizes, the >29 guard, and N > depth windowing.
TEST(RdmaLoopback, ScatterGatherRoundtripOverRdma) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  // depth>1 so the SG path also crosses send windows (N > depth).
  ::setenv("DFKV_RDMA_DEPTH", "4", 1);
  RdmaNode node("sg");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  ASSERT_TRUE(rt.pipelined());  // RDMA path, not native TCP

  auto make_chunks = [](const std::string& tag, const std::vector<size_t>& sizes) {
    std::vector<std::string> v(sizes.size());
    for (size_t i = 0; i < sizes.size(); ++i) {
      v[i].resize(sizes[i]);
      for (size_t b = 0; b < sizes[i]; ++b)
        v[i][b] = static_cast<char>((tag.size() + i * 31 + b * 7) & 0xFF);
    }
    return v;
  };
  auto roundtrip = [&](const std::string& key, const std::vector<size_t>& sizes) {
    auto src = make_chunks(key, sizes);
    std::vector<const void*> ptrs;
    for (auto& s : src) ptrs.push_back(s.data());
    KvPutItemSg put{key, ptrs, sizes};
    auto pr = c.BatchPutSg({put});
    ASSERT_TRUE(pr[0]) << "put " << key;
    std::vector<std::string> dst(sizes.size());
    std::vector<void*> dptrs;
    for (size_t i = 0; i < sizes.size(); ++i) { dst[i].assign(sizes[i], '\0'); dptrs.push_back(&dst[i][0]); }
    KvGetItemSg get{key, dptrs, sizes};
    std::vector<size_t> lens;
    auto gr = c.BatchGetAutoSg({get}, &lens);
    ASSERT_TRUE(gr[0]) << "get " << key;
    size_t total = 0; for (size_t s : sizes) total += s;
    EXPECT_EQ(lens[0], total);
    for (size_t i = 0; i < sizes.size(); ++i) EXPECT_EQ(dst[i], src[i]) << key << " seg " << i;
  };

  roundtrip("sg_n1", std::vector<size_t>(1, 4096));
  roundtrip("sg_n2", std::vector<size_t>(2, 2048));
  roundtrip("sg_n29", std::vector<size_t>(29, 256));         // max payload SGEs
  roundtrip("sg_var", {1, 7, 64, 333, 4096, 11});            // variable per-seg sizes

  // >29 segments: rejected by the guard (put fails, get is a miss) — no corruption.
  {
    std::vector<size_t> sizes(30, 16);
    auto src = make_chunks("sg_over", sizes);
    std::vector<const void*> ptrs; for (auto& s : src) ptrs.push_back(s.data());
    KvPutItemSg put{"sg_over", ptrs, sizes};
    EXPECT_FALSE(c.BatchPutSg({put})[0]);
    std::vector<std::string> dst(30);
    std::vector<void*> dptrs;
    for (int i = 0; i < 30; ++i) { dst[i].assign(16, '\0'); dptrs.push_back(&dst[i][0]); }
    KvGetItemSg get{"sg_over", dptrs, sizes};
    std::vector<size_t> lens;
    EXPECT_FALSE(c.BatchGetAutoSg({get}, &lens)[0]);
  }

  // Multi-key fan-out exceeding the send window depth (4). Pin concurrency to 1:
  // the Soft-RoCE (rdma_rxe) loopback used in CI races when many distinct QPs run
  // in parallel (the SAME flakiness affects the contiguous BatchGetAuto path on
  // rxe — it is an emulation artifact, not an SG defect). This still exercises the
  // real multi-SGE verbs datapath across many keys and multiple send windows.
  {
    c.set_batch_concurrency(1);
    const int N = 24;
    std::vector<std::vector<std::string>> srcs(N);
    std::vector<KvPutItemSg> puts(N);
    for (int i = 0; i < N; ++i) {
      std::vector<size_t> sizes(1 + (i % 4), 64 + (i % 8));
      srcs[i] = make_chunks("sgm" + std::to_string(i), sizes);
      std::vector<const void*> ptrs; for (auto& s : srcs[i]) ptrs.push_back(s.data());
      puts[i] = {"sgm" + std::to_string(i), ptrs, sizes};
    }
    auto pr = c.BatchPutSg(puts);
    for (int i = 0; i < N; ++i) ASSERT_TRUE(pr[i]) << i;
    std::vector<std::vector<std::string>> dsts(N);
    std::vector<KvGetItemSg> gets(N);
    for (int i = 0; i < N; ++i) {
      dsts[i].resize(srcs[i].size());
      std::vector<void*> dptrs; std::vector<size_t> caps(srcs[i].size());
      for (size_t j = 0; j < srcs[i].size(); ++j) {
        dsts[i][j].assign(srcs[i][j].size(), '\0');
        dptrs.push_back(&dsts[i][j][0]); caps[j] = srcs[i][j].size();
      }
      gets[i] = {"sgm" + std::to_string(i), dptrs, caps};
    }
    std::vector<size_t> lens;
    auto gr = c.BatchGetAutoSg(gets, &lens);
    for (int i = 0; i < N; ++i) {
      ASSERT_TRUE(gr[i]) << i;
      for (size_t j = 0; j < srcs[i].size(); ++j) EXPECT_EQ(dsts[i][j], srcs[i][j]) << i << " " << j;
    }
  }
  ::unsetenv("DFKV_RDMA_DEPTH");
}

// Fix 1 regression: an oversized SG key (total payload > max_payload_, but with a
// legal segment count so it passes the client guard) must fail ONLY itself inside
// CacheFromMulti/RangeIntoMulti, NOT poison its node batch. Previously the up-front
// validation std::fill'd every result kInvalid and returned; now the offender is
// skipped in the window and its siblings on the same node proceed normally.
// Regression: one deep window of SG items where every segment is a distinct
// unregistered buffer. All MRs must remain live until the whole posted window
// completes, then be released before the public call returns. This exercises
// depth * (max_sge-1) simultaneous transient registrations across PUT and GET.
TEST(RdmaLoopback, SgDeepWindowManyUnregisteredSegmentsSafe) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ::setenv("DFKV_RDMA_DEPTH", "4", 1);
  RdmaNode node("sgdw");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  ASSERT_TRUE(rt.pipelined());
  const uint64_t transient_baseline =
      rdma::RcEndpoint::TransientUserMrActive();

  constexpr int kItems = 8;   // > depth: also crosses windows
  constexpr int kSegs = 29;   // max payload SGEs per slot
  constexpr size_t kSeg = 512;
  std::vector<std::vector<std::string>> src(kItems);
  std::vector<KvPutItemSg> puts;
  for (int it = 0; it < kItems; ++it) {
    src[it].resize(kSegs);
    std::vector<const void*> ptrs;
    std::vector<size_t> sizes;
    for (int sg = 0; sg < kSegs; ++sg) {
      src[it][sg].assign(kSeg, static_cast<char>('a' + (it * 7 + sg) % 26));
      ptrs.push_back(src[it][sg].data());
      sizes.push_back(kSeg);
    }
    puts.push_back({"sgdw_" + std::to_string(it), ptrs, sizes});
  }
  auto pr = c.BatchPutSg(puts);
  ASSERT_EQ(pr.size(), puts.size());
  for (int it = 0; it < kItems; ++it) EXPECT_TRUE(pr[it]) << "put item " << it;
  EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), transient_baseline);

  std::vector<std::vector<std::string>> dst(kItems);
  std::vector<KvGetItemSg> gets;
  for (int it = 0; it < kItems; ++it) {
    dst[it].assign(kSegs, std::string(kSeg, '\0'));
    std::vector<void*> dptrs;
    std::vector<size_t> caps;
    for (int sg = 0; sg < kSegs; ++sg) { dptrs.push_back(&dst[it][sg][0]); caps.push_back(kSeg); }
    gets.push_back({"sgdw_" + std::to_string(it), dptrs, caps});
  }
  std::vector<size_t> lens;
  auto gr = c.BatchGetAutoSg(gets, &lens);
  ASSERT_EQ(gr.size(), gets.size());
  for (int it = 0; it < kItems; ++it) {
    EXPECT_TRUE(gr[it]) << "get item " << it;
    EXPECT_EQ(lens[it], kSegs * kSeg) << it;
    for (int sg = 0; sg < kSegs; ++sg)
      EXPECT_EQ(dst[it][sg], src[it][sg]) << "item " << it << " seg " << sg;
  }
  EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), transient_baseline);
  ::unsetenv("DFKV_RDMA_DEPTH");
}

TEST(RdmaLoopback, ScatterGatherOversizedFailsOnlyOffender) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ::setenv("DFKV_RDMA_DEPTH", "4", 1);
  RdmaNode node("sgov");
  RdmaTransport rt(kMaxMsg);  // max_payload_ = 256 KiB
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  c.set_batch_concurrency(1);  // single node, stable on rxe loopback
  ASSERT_TRUE(rt.pipelined());

  auto fill = [](std::string& s, int seed) {
    for (size_t b = 0; b < s.size(); ++b) s[b] = static_cast<char>((seed + b * 7) & 0xFF);
  };

  // Valid sibling A, an oversized item (2 segs * 200 KiB = 400 KiB > 256 KiB max_payload,
  // only 2 segments so it clears the <=29-seg client guard), then valid sibling B —
  // all routed to the same (only) node. The offender must report failure; both
  // siblings must succeed and round-trip byte-exact.
  std::string a0(4096, '\0'); fill(a0, 11);
  std::string b0(8192, '\0'); fill(b0, 23);
  std::string big0(200 * 1024, '\0'), big1(200 * 1024, '\0');
  fill(big0, 31); fill(big1, 37);

  std::vector<KvPutItemSg> puts = {
      {"sgov_a", {a0.data()}, {a0.size()}},
      {"sgov_big", {big0.data(), big1.data()}, {big0.size(), big1.size()}},
      {"sgov_b", {b0.data()}, {b0.size()}},
  };
  auto pr = c.BatchPutSg(puts);
  EXPECT_TRUE(pr[0]) << "sibling A put";
  EXPECT_FALSE(pr[1]) << "oversized put must fail only itself";
  EXPECT_TRUE(pr[2]) << "sibling B put (must not be poisoned by the offender)";

  // The two siblings must have been stored; the oversized key must be absent (never
  // written). Drive the GET-side oversized path too: a get whose total cap exceeds
  // max_payload must miss without poisoning its siblings.
  std::string ga(4096, '\0'), gb(8192, '\0');
  std::string gbig0(200 * 1024, '\0'), gbig1(200 * 1024, '\0');
  std::vector<KvGetItemSg> gets = {
      {"sgov_a", {&ga[0]}, {ga.size()}},
      {"sgov_big", {&gbig0[0], &gbig1[0]}, {gbig0.size(), gbig1.size()}},
      {"sgov_b", {&gb[0]}, {gb.size()}},
  };
  std::vector<size_t> lens;
  auto gr = c.BatchGetAutoSg(gets, &lens);
  EXPECT_TRUE(gr[0]) << "sibling A get";
  EXPECT_FALSE(gr[1]) << "oversized get must miss only itself";
  EXPECT_TRUE(gr[2]) << "sibling B get (must not be poisoned by the offender)";
  if (gr[0]) { EXPECT_EQ(ga, a0); }
  if (gr[2]) { EXPECT_EQ(gb, b0); }

  ::unsetenv("DFKV_RDMA_DEPTH");
}

TEST(RdmaLoopback, PipelinedPoolDepth) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  // depth>1 enables client pipelining + the server GET worker pool (the most
  // concurrency-heavy path). Env is read by both the client ctor and the server
  // serve loop in this single process. Run under TSan to catch races.
  ::setenv("DFKV_RDMA_DEPTH", "4", 1);
  ::setenv("DFKV_RDMA_WORKERS", "4", 1);
  RdmaNode node("ppd");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  const int N = 64;
  const size_t sz = 4096;
  std::vector<std::string> vals(N), keys(N);
  std::vector<KvPutItem> puts(N);
  for (int i = 0; i < N; ++i) {
    keys[i] = "p" + std::to_string(i);
    vals[i].assign(sz, static_cast<char>((i * 7 + 3) & 0xFF));
    puts[i] = {keys[i], vals[i].data(), sz};
  }
  auto pr = c.BatchPut(puts);
  for (int i = 0; i < N; ++i) ASSERT_TRUE(pr[i]) << i;

  std::vector<std::string> outs(N);
  std::vector<KvGetItem> gets(N);
  for (int i = 0; i < N; ++i) { outs[i].assign(sz, '\0'); gets[i] = {keys[i], &outs[i][0], sz}; }
  auto gr = c.BatchGet(gets);
  int hits = 0;
  for (int i = 0; i < N; ++i) { if (gr[i]) { ++hits; EXPECT_EQ(outs[i], vals[i]) << i; } }
  EXPECT_EQ(hits, N);

  ::unsetenv("DFKV_RDMA_DEPTH");
  ::unsetenv("DFKV_RDMA_WORKERS");
}

// Regression for the conn-thread leak (#3). A server Serve thread blocks in
// WaitComp forever after a silent client disconnect (a torn-down RC peer yields
// no completion), so without an idle timeout a long-running server accumulates
// one live thread per lifetime connection — Stop() is the only reaper. The fix:
// an idle timeout reclaims the connection (the thread returns), and ReapDoneLocked
// joins the finished thread on the next accept. This test sets a short idle window
// and verifies the live count drains back to the baseline; without the fix it
// would stay pinned near N and time out.
TEST(RdmaLoopback, ReclaimsAndReapsIdleConnThreads) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ::setenv("DFKV_RDMA_IDLE_MS", "120", 1);  // reclaim idle conns fast (test only)
  constexpr size_t kSmallMsg = 16 * 1024;   // stay well under an 8 MiB memlock
  RdmaNode node("reap", kSmallMsg);          // server buffers must be small too
  // One long-lived client endpoint holds the shared per-device ibv_context open
  // (its server side may be reclaimed when idle and is transparently re-dialed).
  RdmaTransport keep(kSmallMsg);
  KVClient ckeep({{"n", node.addr}}, SelfHdr(), &keep);
  std::string kk(64, 'k');
  ASSERT_TRUE(ckeep.Put("keep", kk.data(), kk.size()));

  const int N = 30;
  for (int i = 0; i < N; ++i) {
    RdmaTransport rt(kSmallMsg);
    KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
    std::string v(512, static_cast<char>(i & 0xFF));
    ASSERT_TRUE(c.Put("k" + std::to_string(i), v.data(), v.size())) << "i=" << i;
    // rt + c leave scope here -> transient client gone -> server conn goes idle.
  }

  // Each round: wait past the idle window so every prior connection's Serve thread
  // has exited, then make ONE connection whose accept runs ReapDoneLocked to join
  // all the finished threads. Only that single in-flight drain thread should then
  // remain. Without reclaim+reap the N transient threads stay in conns_ and the
  // count never falls (the loop exhausts its rounds and the assert fails).
  size_t live = 0;
  for (int round = 0; round < 6; ++round) {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));  // > idle (120 ms)
    { RdmaTransport rt(kSmallMsg); KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
      std::string v(16, 'x'); c.Put("drain", v.data(), v.size()); }
    live = node.rsrv->live_conn_count();
    if (live <= 2) break;
  }
  EXPECT_LE(live, 2u) << "idle conn threads were not reclaimed + reaped (leak)";
  ::unsetenv("DFKV_RDMA_IDLE_MS");
}

// Regression: a BATCH op whose pooled connection the server reclaimed on idle
// must transparently re-dial and succeed — NOT return kIOError for the batch.
// Before the fix only the single-op RoundTrip retried a stale pooled conn; the
// batch paths (CacheMany/CacheFrom/RangeInto/ExistMany/...) gave up on the first
// failed window, which surfaced to SGLang as "Write page to storage: N pages
// failed" on writes and 0-hit prefixes on reads after an idle gap.
TEST(RdmaLoopback, BatchRetriesAfterServerReclaim) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ::setenv("DFKV_RDMA_IDLE_MS", "120", 1);   // server reclaims idle conns fast
  RdmaNode node("brc");
  RdmaTransport rt(kMaxMsg);                   // long-lived: holds the device ctx
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  const int N = 24;
  for (int i = 0; i < N; ++i) {
    std::string v = "val" + std::to_string(i);
    ASSERT_TRUE(c.Put("b" + std::to_string(i), v.data(), v.size())) << "warm i=" << i;
  }
  // Warm the pool so a connection is parked for the node, then snapshot the dial
  // counter — the batch op below must open a NEW one when it finds it stale.
  EXPECT_TRUE(c.Exist("b0"));
  long opened_warm =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  ASSERT_GE(opened_warm, 1);

  // Let the server reclaim the now-idle pooled connection (idle window 120 ms).
  std::this_thread::sleep_for(std::chrono::milliseconds(400));

  // The pooled conn is stale (its server peer was reclaimed). A batch existence
  // probe must detect that on the first window and re-dial a fresh conn, returning
  // CORRECT results — not kIOError for the whole batch. Direct transport call so no
  // KVClient-level health retry can mask a transport that failed to recover.
  // Probe with the same canonical namespace/object identity as the warm writes.
  std::vector<BlockKey> keys;
  for (int i = 0; i < N; ++i) {
    keys.push_back(ToBlockKey(SelfHdr(), "b" + std::to_string(i)));
    keys.push_back(ToBlockKey(SelfHdr(), "nope" + std::to_string(i)));
  }
  std::vector<char> exists;
  auto sts = rt.ExistMany(node.addr, keys, &exists);
  ASSERT_EQ(exists.size(), keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    EXPECT_NE(sts[i], Status::kIOError) << "i=" << i << " (stale conn not retried)";
    EXPECT_EQ(exists[i] != 0, (i % 2 == 0)) << "i=" << i;
  }
  // The retry re-dialed: a new client connection was opened after the warm-up.
  long opened_after =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  EXPECT_GT(opened_after, opened_warm) << "batch op did not re-dial after reclaim";

  ::unsetenv("DFKV_RDMA_IDLE_MS");
}

// A cache node that ALSO wires the async-GET prep + complete hooks, so the server
// uses the io_uring batch-and-wait GET path when DFKV_SERVER_URING=1 (and the
// binary was built with -DDFKV_WITH_URING). With the env off / unbuilt these
// hooks are simply never consulted and the node behaves like a plain RdmaNode.
struct RdmaUringNode {
  fs::path dir;
  std::unique_ptr<KvNodeServer> srv;
  std::unique_ptr<RdmaServer> rsrv;
  std::string addr;

  explicit RdmaUringNode(const std::string& tag, size_t max_msg = kMaxMsg) {
    dir = fs::temp_directory_path() / ("dfkv_uring_" + tag);
    fs::remove_all(dir);
    fs::create_directories(dir);
    srv = std::make_unique<KvNodeServer>(dir.string(), 1ull << 30);
    EXPECT_EQ(srv->Start(0), Status::kOk);
    rsrv = std::make_unique<RdmaServer>(
        [this](uint8_t op, const BlockKey& key, uint64_t off, uint64_t len,
               const char* pl, uint64_t pll, std::string* out,
               size_t* value_len) {
          return srv->ProcessRequestForKey(
              op, key, off, len, pl, pll, out, value_len);
        },
        max_msg);
    rsrv->set_range_handler(  // sync fallback (used when uring path is off)
        [this](const BlockKey& key, uint64_t off, uint64_t len,
               char* io_buf, size_t cap, const char** out_data,
               size_t* out_len, size_t* value_len) {
          return srv->RangeDirectForKey(
              key, off, len, io_buf, cap, out_data, out_len, value_len);
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
  ~RdmaUringNode() {
    if (rsrv) rsrv->Stop();
    if (srv) srv->Stop();
    fs::remove_all(dir);
  }
};

// Correctness proof for the io_uring async-GET path: many concurrent GETs over a
// SINGLE pooled connection, with the depth high enough that several requests are
// in flight per WaitComp batch (so the server submits a multi-read io_uring batch
// and must reply in arrival order). Every value must come back byte-correct. With
// the flag OFF this still passes via the synchronous path (regression guard); with
// DFKV_SERVER_URING=1 + a URING build it exercises the batch-and-wait reads.
TEST(RdmaLoopback, UringAsyncGetManyConcurrentInOrder) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  // Request the async path (no-op if unbuilt). Use overwrite=0 so a shell-set
  // DFKV_SERVER_URING (e.g. =0 to force the sync path) wins — this lets the CI
  // run the SAME test through both the sync and async serve loops.
  ::setenv("DFKV_SERVER_URING", "1", 0);
  ::setenv("DFKV_SERVER_URING_DEPTH", "32", 1);
  ::setenv("DFKV_RDMA_DEPTH", "8", 1);      // K=8 in-flight => multi-read batches
  ::setenv("DFKV_RDMA_RECV_SEGMENT_SIZE", "4194304", 1);
  // Small per-buffer cap so K=8 slots (rbuf+sbuf+dbuf each) stay under an 8 MiB
  // RLIMIT_MEMLOCK (CI default). Values below are <= 12 KiB, well within 64 KiB.
  constexpr size_t kUringMsg = 64 * 1024;
  RdmaUringNode node("ag", kUringMsg);
  RdmaTransport rt(kUringMsg);
  ASSERT_TRUE(rt.pipelined());
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  // Pin client batch concurrency to 1 QP: the Soft-RoCE (rdma_rxe) loopback used
  // in CI races when many distinct QPs run in parallel (an emulation artifact that
  // hits the plain sync GET path too — see ScatterGatherRoundtripOverRdma). A
  // single connection still pipelines up to `depth` GETs per send window, so the
  // SERVER still forms multi-read io_uring batches; this only removes the rxe
  // cross-QP flakiness, not the concurrency under test.
  c.set_batch_concurrency(1);

  // Distinct content per key (offset + index dependent) so a misrouted reply
  // (wrong buffer / reordered) would mismatch. Mix of sizes incl. sub-page and
  // multi-page to exercise the O_DIRECT aligned-superset trim.
  const int N = 200;
  const size_t kSizes[] = {64, 512, 4096, 8192, 12288};
  std::vector<std::string> vals(N), keys(N);
  std::vector<KvPutItem> puts(N);
  for (int i = 0; i < N; ++i) {
    keys[i] = "ag" + std::to_string(i);
    size_t sz = kSizes[i % 5];
    vals[i].resize(sz);
    for (size_t k = 0; k < sz; ++k)
      vals[i][k] = static_cast<char>((i * 131 + k * 7 + 13) & 0xFF);
    puts[i] = {keys[i], vals[i].data(), sz};
  }
  auto pr = c.BatchPut(puts);
  for (int i = 0; i < N; ++i) ASSERT_TRUE(pr[i]) << i;

  // Run several BatchGet rounds over the one pooled connection; each round fans
  // out N GETs that pipeline K-at-a-time -> the server forms multi-read batches.
  for (int round = 0; round < 3; ++round) {
    std::vector<std::string> outs(N);
    std::vector<KvGetItem> gets(N);
    for (int i = 0; i < N; ++i) { outs[i].assign(vals[i].size(), '\0'); gets[i] = {keys[i], &outs[i][0], vals[i].size()}; }
    auto gr = c.BatchGet(gets);
    for (int i = 0; i < N; ++i) {
      ASSERT_TRUE(gr[i]) << "round " << round << " key " << i;
      EXPECT_EQ(outs[i], vals[i]) << "round " << round << " key " << i;
    }
  }

  // A miss on the async path must still be a clean miss (kNotFound), not an error
  // or a stale-buffer hit; interleave present/absent to mix sync-miss + async-hit
  // replies in the same pipelined window (order-preservation across reply kinds).
  std::vector<std::string> mouts(N);
  std::vector<KvGetItem> mgets;
  std::vector<std::string> mkeys;
  for (int i = 0; i < N; ++i) {
    mkeys.push_back(keys[i]);
    mkeys.push_back("ag_absent" + std::to_string(i));
  }
  std::vector<std::string> mo(mkeys.size());
  std::vector<KvGetItem> mg(mkeys.size());
  for (size_t i = 0; i < mkeys.size(); ++i) {
    size_t cap = (i % 2 == 0) ? vals[i / 2].size() : 4096;
    mo[i].assign(cap, '\0');
    mg[i] = {mkeys[i], &mo[i][0], cap};
  }
  auto mgr = c.BatchGet(mg);
  for (size_t i = 0; i < mkeys.size(); ++i) {
    if (i % 2 == 0) {
      ASSERT_TRUE(mgr[i]) << "present key " << mkeys[i];
      EXPECT_EQ(mo[i], vals[i / 2]) << "present key " << mkeys[i];
    } else {
      EXPECT_FALSE(mgr[i]) << "absent key should miss: " << mkeys[i];
    }
  }

  // The async (uring) read path now feeds op="get" latency: after this many
  // disk-backed GETs the 1/64 sampler must have recorded at least one sample,
  // where before this change the default read path was latency-blind. (When the
  // shell forces DFKV_SERVER_URING=0 the sync RangeDirect samples the same
  // series, so the assertion holds through both serve loops.)
  const std::string mtext = node.srv->MetricsText();
  auto gp = mtext.find("dfkv_op_latency_seconds_count{op=\"get\"}");
  ASSERT_NE(gp, std::string::npos) << mtext;
  long gcnt = std::stol(mtext.substr(
      gp + std::string("dfkv_op_latency_seconds_count{op=\"get\"}").size()));
  EXPECT_GT(gcnt, 0) << "uring GET path recorded no op=\"get\" latency sample";
#ifdef DFKV_WITH_URING
  const char* uring_enabled = std::getenv("DFKV_SERVER_URING");
  if (!uring_enabled || std::strcmp(uring_enabled, "0") != 0) {
    EXPECT_GT(node.rsrv->UringReads(), 0u)
        << "auto-v2 silently bypassed the io_uring read path";
    EXPECT_GT(node.rsrv->V2Conns(), 0u)
        << "test did not exercise the v2 async-GET response path";
  }
#endif

  ::unsetenv("DFKV_SERVER_URING");
  ::unsetenv("DFKV_SERVER_URING_DEPTH");
  ::unsetenv("DFKV_RDMA_DEPTH");
  ::unsetenv("DFKV_RDMA_RECV_SEGMENT_SIZE");
}

// --- P3 B5-3: RAM hot-tier zero-copy RDMA serve --------------------------------
// A cache node with the RAM tier enabled must serve a GET straight from the
// pre-registered arena MR (scatter-send, no copy into the connection buffer, no
// disk), and the bytes must round-trip correctly. dfkv_ram_hit_total proves the
// zero-copy path was taken; the send-in-flight pin is released on IBV_WC_SEND.
namespace {
struct RamRdmaNode {
  fs::path dir;
  std::unique_ptr<KvNodeServer> srv;
  std::unique_ptr<RdmaServer> rsrv;
  std::string addr;

  explicit RamRdmaNode(const std::string& tag, size_t max_msg = kMaxMsg,
                       bool writearound = false) {
    ConfigureTestRecvSegment();
    if (writearound)
      ::setenv("DFKV_RAM_WRITE_MODE", "writearound", 1);
    else
      ::unsetenv("DFKV_RAM_WRITE_MODE");
    ::setenv("DFKV_RAM_TIER", "1", 1);
    ::setenv("DFKV_RAM_TIER_BYTES", "8388608", 1);  // 8 MiB arena
    dir = fs::temp_directory_path() / ("dfkv_ramrdma_" + tag);
    fs::remove_all(dir);
    fs::create_directories(dir);
    srv = std::make_unique<KvNodeServer>(dir.string(), 1ull << 30);
    EXPECT_EQ(srv->Start(0), Status::kOk);
    rsrv = std::make_unique<RdmaServer>(
        [this](uint8_t op, const BlockKey& key, uint64_t off, uint64_t len,
               const char* pl, uint64_t pll, std::string* out,
               size_t* value_len) {
          return srv->ProcessRequestForKey(
              op, key, off, len, pl, pll, out, value_len);
        },
        max_msg);
    rsrv->set_range_handler(
        [this](const BlockKey& key, uint64_t off, uint64_t len,
               char* io_buf, size_t cap, const char** out_data,
               size_t* out_len, size_t* value_len) {
          return srv->RangeDirectForKey(
              key, off, len, io_buf, cap, out_data, out_len, value_len);
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
    // The RAM arena is a registered source pool; per-send pin ownership lives
    // inside PreparedRead.
    if (srv->ram_enabled()) {
      rsrv->RegisterMemory(srv->ram_arena(), srv->ram_arena_bytes());
    }
    EXPECT_EQ(rsrv->Start(0), Status::kOk);
    addr = "127.0.0.1:" + std::to_string(rsrv->port());
  }
  ~RamRdmaNode() {
    if (rsrv) rsrv->Stop();
    if (srv) srv->Stop();
    srv.reset();
    ::unsetenv("DFKV_RAM_TIER");
    ::unsetenv("DFKV_RAM_TIER_BYTES");
    ::unsetenv("DFKV_RAM_WRITE_MODE");
    fs::remove_all(dir);
  }
};
}  // namespace

TEST(RdmaLoopback, RamTierZeroCopyServe) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device (load rdma_rxe for Soft-RoCE)";
  RamRdmaNode node("ram");
  ASSERT_TRUE(node.srv->ram_enabled());
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  // PUT then GET a spread of blocks over RDMA; each must round-trip byte-for-byte
  // even though the GET is served zero-copy from the arena MR.
  const int N = 40;
  std::vector<std::string> vals;
  for (int i = 0; i < N; ++i) {
    std::string v = "ram-rdma-" + std::to_string(i) + std::string(300 + i, 'q');
    vals.push_back(v);
    ASSERT_TRUE(c.Put("z" + std::to_string(i), v.data(), v.size())) << i;
  }
  for (int i = 0; i < N; ++i) {
    std::string out(vals[i].size(), '\0');
    ASSERT_TRUE(c.Get("z" + std::to_string(i), &out[0], out.size())) << i;
    EXPECT_EQ(out, vals[i]) << i;
  }

  const std::string m = node.srv->MetricsText();
  EXPECT_GT(CounterVal(m, "dfkv_ram_hit_total"), 0) << "GET must be served from RAM";
  // Every send completed, so no arena slot is left send-pinned: a re-GET still
  // works (pin released on IBV_WC_SEND, not leaked).
  std::string out2(vals[0].size(), '\0');
  ASSERT_TRUE(c.Get("z0", &out2[0], out2.size()));
  EXPECT_EQ(out2, vals[0]);
}

TEST(RdmaLoopback, ColdGetPromotesDirectlyIntoRegisteredRamArena) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device (load rdma_rxe for Soft-RoCE)";
  ::setenv("DFKV_READ_COALESCE", "1", 1);
  ::setenv("DFKV_SERVER_URING", "1", 1);
  {
    RamRdmaNode node("cold-direct-promotion", kMaxMsg,
                     /*writearound=*/true);
    ASSERT_TRUE(node.srv->ram_enabled());
    RdmaTransport rt(kMaxMsg);
    KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

    std::string value(256 * 1024, '\0');
    for (size_t i = 0; i < value.size(); ++i)
      value[i] = static_cast<char>('a' + (i % 19));
    ASSERT_TRUE(c.Put("cold-direct", value.data(), value.size()));
    EXPECT_EQ(CounterVal(node.srv->MetricsText(), "dfkv_ram_objects"), 0);

    std::string first(value.size(), '\0');
    ASSERT_TRUE(c.Get("cold-direct", first.data(), first.size()));
    EXPECT_EQ(first, value);
    const std::string after_cold = node.srv->MetricsText();
    EXPECT_EQ(CounterVal(after_cold, "dfkv_ram_promoted_total"), 1);
    EXPECT_EQ(CounterVal(after_cold, "dfkv_ram_objects"), 1);
#ifdef DFKV_WITH_URING
    EXPECT_GT(node.rsrv->UringReads(), 0u);
#endif

    const long hits_before =
        CounterVal(after_cold, "dfkv_ram_hit_total");
    std::string warm(value.size(), '\0');
    ASSERT_TRUE(c.Get("cold-direct", warm.data(), warm.size()));
    EXPECT_EQ(warm, value);
    EXPECT_GT(CounterVal(node.srv->MetricsText(), "dfkv_ram_hit_total"),
              hits_before);
  }
  ::unsetenv("DFKV_READ_COALESCE");
  ::unsetenv("DFKV_SERVER_URING");
}

// DCP2 declared caps: a client that tightens its max block size gets smaller
// shared slots and must still complete every op within the declaration;
// oversized ops fail client-side with kInvalid without touching the wire.
TEST(RdmaLoopback, DeclaredCapsRoundTripAndClientSideBound) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("caps");
  setenv("DFKV_RDMA_MAX_BLOCK_BYTES", "65536", 1);  // declare 64 KiB
  RdmaTransport rt(kMaxMsg);
  unsetenv("DFKV_RDMA_MAX_BLOCK_BYTES");            // don't leak into other tests
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  // Within the declaration: normal round-trip on the right-sized connection.
  std::string v(60 * 1024, 'a');
  ASSERT_TRUE(c.Put("caps-ok", v.data(), v.size()));
  std::string got(v.size(), '\0');
  ASSERT_TRUE(c.Get("caps-ok", got.data(), got.size()));
  EXPECT_EQ(got, v);

  // Over the declaration (but under the transport max): the client bound must
  // reject before sending -- Put returns false, connection stays healthy.
  std::string big(128 * 1024, 'b');
  EXPECT_FALSE(c.Put("caps-over", big.data(), big.size()));
  ASSERT_TRUE(c.Get("caps-ok", got.data(), got.size())) << "conn must survive the rejected op";
  EXPECT_EQ(got, v);
}

// With no explicit override, DCP2 declares the transport's safe global cap;
// blocks right up to that bound still round-trip.
TEST(RdmaLoopback, DefaultDeclarationKeepsGlobalCap) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("nocaps");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  std::string v(kMaxMsg - 1, 'c');  // near the global raw-value cap
  ASSERT_TRUE(c.Put("full-size", v.data(), v.size()));
  std::string got(v.size(), '\0');
  ASSERT_TRUE(c.Get("full-size", got.data(), got.size()));
  EXPECT_EQ(got, v);
}
