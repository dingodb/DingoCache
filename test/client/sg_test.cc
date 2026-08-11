// Scatter-gather (multi-buffer-per-key) round-trip tests for the additive
// dfkv_batch_put_sg / dfkv_batch_get_auto_sg path. One dfkv key gathers N
// non-contiguous source buffers on put and scatters into N destination buffers
// on get. Exercises small widths, variable segment sizes, logical widths larger
// than one transport window, and multi-key/multi-node fan-out over loopback TCP.
#include "client/kv_client.h"
#include "cache/kv_node_server.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {
std::string Hdr() { return "test/model"; }
struct Node { fs::path dir; std::unique_ptr<KvNodeServer> srv; std::string addr; };
std::unique_ptr<Node> Start(const std::string& tag) {
  auto n = std::make_unique<Node>();
  n->dir = fs::temp_directory_path() / ("dfkv_sg_" + tag);
  fs::remove_all(n->dir); fs::create_directories(n->dir);
  n->srv = std::make_unique<KvNodeServer>(n->dir.string(), 1ull << 30);
  EXPECT_EQ(n->srv->Start(0), Status::kOk);
  n->addr = "127.0.0.1:" + std::to_string(n->srv->port());
  return n;
}

// Make `count` distinct payload chunks of given sizes, contents = key-tagged.
std::vector<std::string> MakeChunks(const std::string& tag,
                                    const std::vector<size_t>& sizes) {
  std::vector<std::string> chunks(sizes.size());
  for (size_t i = 0; i < sizes.size(); ++i) {
    chunks[i].resize(sizes[i]);
    for (size_t b = 0; b < sizes[i]; ++b)
      chunks[i][b] = static_cast<char>((tag.size() + i * 31 + b * 7) & 0xFF);
  }
  return chunks;
}
}  // namespace

// Put a key as N separate buffers, get it back into N separate buffers; assert the
// per-segment bytes match and the reported total length is the sum of the sizes.
static void RoundTrip(KVClient& c, const std::string& key, size_t nsegs,
                      size_t seg_bytes) {
  std::vector<size_t> sizes(nsegs, seg_bytes);
  auto src = MakeChunks(key, sizes);
  std::vector<const void*> ptrs;
  for (auto& s : src) ptrs.push_back(s.data());

  KvPutItemSg put;
  put.key = key; put.ptrs = ptrs; put.sizes = sizes;
  auto pr = c.BatchPutSg({put});
  ASSERT_EQ(pr.size(), 1u);
  ASSERT_TRUE(pr[0]) << "put key=" << key << " nsegs=" << nsegs;

  // Distinct destination buffers, one per segment.
  std::vector<std::string> dst(nsegs);
  std::vector<void*> dptrs;
  for (size_t i = 0; i < nsegs; ++i) { dst[i].assign(seg_bytes, '\0'); dptrs.push_back(&dst[i][0]); }
  KvGetItemSg get;
  get.key = key; get.dsts = dptrs; get.caps = sizes;
  std::vector<size_t> lens;
  auto gr = c.BatchGetAutoSg({get}, &lens);
  ASSERT_EQ(gr.size(), 1u);
  ASSERT_TRUE(gr[0]) << "get key=" << key << " nsegs=" << nsegs;
  EXPECT_EQ(lens[0], std::accumulate(sizes.begin(), sizes.end(), size_t{0}));
  for (size_t i = 0; i < nsegs; ++i)
    EXPECT_EQ(dst[i], src[i]) << "key=" << key << " seg=" << i;
}

TEST(Sg, RoundTripN1) {
  auto a = Start("n1");
  KVClient c({{"a", a->addr}}, Hdr());
  RoundTrip(c, "k_n1", 1, 256);
  a->srv->Stop();
}

TEST(Sg, RoundTripN2) {
  auto a = Start("n2");
  KVClient c({{"a", a->addr}}, Hdr());
  RoundTrip(c, "k_n2", 2, 512);
  a->srv->Stop();
}

TEST(Sg, RoundTripN29) {
  auto a = Start("n29");
  KVClient c({{"a", a->addr}}, Hdr());
  RoundTrip(c, "k_n29", 29, 128);
  a->srv->Stop();
}

// Variable segment sizes within one key (scatter split must honor each size).
TEST(Sg, RoundTripVariableSizes) {
  auto a = Start("var");
  KVClient c({{"a", a->addr}}, Hdr());
  std::vector<size_t> sizes = {1, 7, 64, 333, 4096, 11};
  auto src = MakeChunks("var", sizes);
  std::vector<const void*> ptrs;
  for (auto& s : src) ptrs.push_back(s.data());
  KvPutItemSg put{"k_var", ptrs, sizes};
  ASSERT_TRUE(c.BatchPutSg({put})[0]);

  std::vector<std::string> dst(sizes.size());
  std::vector<void*> dptrs;
  for (size_t i = 0; i < sizes.size(); ++i) { dst[i].assign(sizes[i], '\0'); dptrs.push_back(&dst[i][0]); }
  KvGetItemSg get{"k_var", dptrs, sizes};
  std::vector<size_t> lens;
  ASSERT_TRUE(c.BatchGetAutoSg({get}, &lens)[0]);
  EXPECT_EQ(lens[0], std::accumulate(sizes.begin(), sizes.end(), size_t{0}));
  for (size_t i = 0; i < sizes.size(); ++i) EXPECT_EQ(dst[i], src[i]) << i;
  a->srv->Stop();
}

// Logical SG width is not capped by one transport window. Use enough uniquely
// sized and populated segments to cross two boundaries, so both PUT gathering
// and GET scattering must preserve every byte in order across multiple windows.
TEST(Sg, UnlimitedLogicalWidthRoundTripsByteExact) {
  auto a = Start("unlimited_width");
  KVClient c({{"a", a->addr}}, Hdr());
  const size_t window_segs = c.MaxSgPayloadSegs();
  ASSERT_GT(window_segs, 0u);
  const size_t nsegs = window_segs * 2 + 3;

  std::vector<size_t> sizes;
  sizes.reserve(nsegs);
  for (size_t i = 0; i < nsegs; ++i) sizes.push_back(1 + (i * 17) % 97);
  auto src = MakeChunks("unlimited_width", sizes);
  std::vector<const void*> ptrs;
  ptrs.reserve(nsegs);
  for (auto& segment : src) ptrs.push_back(segment.data());

  const auto put = c.BatchPutSg({{"k_unlimited_width", ptrs, sizes}});
  ASSERT_EQ(put.size(), 1u);
  ASSERT_TRUE(put[0]);

  std::vector<std::string> dst(nsegs);
  std::vector<void*> dptrs;
  dptrs.reserve(nsegs);
  for (size_t i = 0; i < nsegs; ++i) {
    dst[i].assign(sizes[i], '\0');
    dptrs.push_back(dst[i].data());
  }
  std::vector<size_t> lengths;
  const auto get = c.BatchGetAutoSg(
      {{"k_unlimited_width", dptrs, sizes}}, &lengths);
  ASSERT_EQ(get.size(), 1u);
  ASSERT_EQ(lengths.size(), 1u);
  ASSERT_TRUE(get[0]);
  EXPECT_EQ(lengths[0],
            std::accumulate(sizes.begin(), sizes.end(), size_t{0}));
  for (size_t i = 0; i < nsegs; ++i)
    EXPECT_EQ(dst[i], src[i]) << "segment " << i;
  a->srv->Stop();
}

TEST(Sg, SizeMaxSumOverflowFailsOnlyOffendingItem) {
  auto a = Start("sum_overflow");
  KVClient c({{"a", a->addr}}, Hdr());
  char byte = 'v';
  KvPutItemSg overflow_put{
      "overflow", {&byte, &byte},
      {std::numeric_limits<size_t>::max(), size_t{1}}};
  KvPutItemSg valid_put{"valid", {&byte}, {size_t{1}}};
  const auto put = c.BatchPutSg({overflow_put, valid_put});
  ASSERT_EQ(put.size(), 2u);
  EXPECT_FALSE(put[0]);
  EXPECT_TRUE(put[1]);

  char overflow_dst = 'x';
  char valid_dst = '\0';
  KvGetItemSg overflow_get{
      "overflow", {&overflow_dst, &overflow_dst},
      {std::numeric_limits<size_t>::max(), size_t{1}}};
  KvGetItemSg valid_get{"valid", {&valid_dst}, {size_t{1}}};
  std::vector<size_t> lengths = {99, 99};
  const auto get = c.BatchGetAutoSg({overflow_get, valid_get}, &lengths);
  ASSERT_EQ(get.size(), 2u);
  ASSERT_EQ(lengths.size(), 2u);
  EXPECT_FALSE(get[0]);
  EXPECT_EQ(lengths[0], 0u);
  EXPECT_EQ(overflow_dst, 'x');
  EXPECT_TRUE(get[1]);
  EXPECT_EQ(lengths[1], 1u);
  EXPECT_EQ(valid_dst, byte);
  a->srv->Stop();
}

// Missing key => miss, len 0.
TEST(Sg, MissReturnsMiss) {
  auto a = Start("miss");
  KVClient c({{"a", a->addr}}, Hdr());
  std::string d0(64, '\0'), d1(64, '\0');
  KvGetItemSg get{"absent", {&d0[0], &d1[0]}, {64, 64}};
  std::vector<size_t> lens;
  auto gr = c.BatchGetAutoSg({get}, &lens);
  EXPECT_FALSE(gr[0]);
  EXPECT_EQ(lens[0], 0u);
  a->srv->Stop();
}

// Fix 3: an empty/null key in a batch is skipped (no header-only blob written, no
// wasted GET issued) and reported failed/miss, while valid siblings still succeed.
TEST(Sg, EmptyKeySkipped) {
  auto a = Start("emptykey");
  KVClient c({{"a", a->addr}}, Hdr());
  std::string v(128, 'x');
  std::vector<KvPutItemSg> puts = {
      {"", {v.data()}, {v.size()}},        // empty key: must be skipped
      {"ek_ok", {v.data()}, {v.size()}},   // valid sibling
  };
  auto pr = c.BatchPutSg(puts);
  EXPECT_FALSE(pr[0]) << "empty-key put must be skipped/failed";
  EXPECT_TRUE(pr[1]) << "valid sibling must still be stored";

  std::string d0(128, '\0'), d1(128, '\0');
  std::vector<KvGetItemSg> gets = {
      {"", {&d0[0]}, {d0.size()}},         // empty key: must miss (no GET issued)
      {"ek_ok", {&d1[0]}, {d1.size()}},    // valid sibling: hit
  };
  std::vector<size_t> lens;
  auto gr = c.BatchGetAutoSg(gets, &lens);
  EXPECT_FALSE(gr[0]) << "empty-key get must miss";
  EXPECT_EQ(lens[0], 0u);
  EXPECT_TRUE(gr[1]) << "valid sibling get must hit";
  if (gr[1]) { EXPECT_EQ(d1, v); }
  a->srv->Stop();
}

// Multi-key fan-out across two nodes, mixed segment counts.
TEST(Sg, MultiKeyTwoNodes) {
  auto a = Start("mk_a"); auto b = Start("mk_b");
  KVClient c({{"a", a->addr}, {"b", b->addr}}, Hdr());
  const int N = 64;
  std::vector<std::vector<std::string>> srcs(N);
  std::vector<KvPutItemSg> puts(N);
  for (int i = 0; i < N; ++i) {
    size_t nsegs = 1 + (i % 5);
    std::vector<size_t> sizes(nsegs, 32 + (i % 7));
    srcs[i] = MakeChunks("mk" + std::to_string(i), sizes);
    std::vector<const void*> ptrs;
    for (auto& s : srcs[i]) ptrs.push_back(s.data());
    puts[i] = {"mk" + std::to_string(i), ptrs, sizes};
  }
  auto pr = c.BatchPutSg(puts);
  for (int i = 0; i < N; ++i) ASSERT_TRUE(pr[i]) << i;

  std::vector<std::vector<std::string>> dsts(N);
  std::vector<KvGetItemSg> gets(N);
  for (int i = 0; i < N; ++i) {
    size_t nsegs = srcs[i].size();
    dsts[i].resize(nsegs);
    std::vector<void*> dptrs;
    std::vector<size_t> caps(nsegs);
    for (size_t j = 0; j < nsegs; ++j) {
      dsts[i][j].assign(srcs[i][j].size(), '\0');
      dptrs.push_back(&dsts[i][j][0]);
      caps[j] = srcs[i][j].size();
    }
    gets[i] = {"mk" + std::to_string(i), dptrs, caps};
  }
  std::vector<size_t> lens;
  auto gr = c.BatchGetAutoSg(gets, &lens);
  for (int i = 0; i < N; ++i) {
    ASSERT_TRUE(gr[i]) << i;
    for (size_t j = 0; j < srcs[i].size(); ++j)
      EXPECT_EQ(dsts[i][j], srcs[i][j]) << "key " << i << " seg " << j;
  }
  a->srv->Stop(); b->srv->Stop();
}

// Large single-node SG batch: with one node in the ring the whole batch routes to
// one (node[,cap]) group, which BOTH the SG write (BatchPutSg) and the SG read
// (BatchGetAutoSg) now split into up to DFKV_READ_MAX_CONNS parallel connections
// (ShardReadGroups). This asserts the shard fan-out preserves every per-key result
// and every scattered byte -- correctness of the concurrent-connection path the
// benchmark leans on for bandwidth. Uniform segment count + size so all keys share
// one (node,cap) group (the worst case for serial drain, best case to shard).
TEST(Sg, LargeSingleNodeShardedRoundTrip) {
  auto a = Start("shard1");
  KVClient c({{"a", a->addr}}, Hdr());
  const int N = 200;             // >> DFKV_READ_SHARD_KEYS(16) => shards to the cap
  const size_t nsegs = 3, seg = 1024;
  std::vector<std::vector<std::string>> srcs(N);
  std::vector<KvPutItemSg> puts(N);
  for (int i = 0; i < N; ++i) {
    std::vector<size_t> sizes(nsegs, seg);
    srcs[i] = MakeChunks("sh" + std::to_string(i), sizes);
    std::vector<const void*> ptrs;
    for (auto& s : srcs[i]) ptrs.push_back(s.data());
    puts[i] = {"sh" + std::to_string(i), ptrs, sizes};
  }
  auto pr = c.BatchPutSg(puts);
  ASSERT_EQ(pr.size(), size_t(N));
  for (int i = 0; i < N; ++i) ASSERT_TRUE(pr[i]) << "sharded put key " << i;

  std::vector<std::vector<std::string>> dsts(N);
  std::vector<KvGetItemSg> gets(N);
  for (int i = 0; i < N; ++i) {
    dsts[i].resize(nsegs);
    std::vector<void*> dptrs;
    std::vector<size_t> caps(nsegs, seg);
    for (size_t j = 0; j < nsegs; ++j) {
      dsts[i][j].assign(seg, '\0');
      dptrs.push_back(&dsts[i][j][0]);
    }
    gets[i] = {"sh" + std::to_string(i), dptrs, caps};
  }
  std::vector<size_t> lens;
  auto gr = c.BatchGetAutoSg(gets, &lens);
  ASSERT_EQ(gr.size(), size_t(N));
  for (int i = 0; i < N; ++i) {
    ASSERT_TRUE(gr[i]) << "sharded get key " << i;
    EXPECT_EQ(lens[i], nsegs * seg) << i;
    for (size_t j = 0; j < nsegs; ++j)
      EXPECT_EQ(dsts[i][j], srcs[i][j]) << "key " << i << " seg " << j;
  }
  a->srv->Stop();
}
