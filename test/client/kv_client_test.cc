// KVClient end-to-end over real TCP loopback cache nodes. Exercises raw-value
// Put/Get/Exist, immediate visibility, namespace isolation, and two-node
// consistent-hash cross-node reads.
#include "client/kv_client.h"
#include "cache/kv_node_server.h"
#include "client/key_map.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {

std::string SelfHdr(uint32_t page_size = 64) {
  return "test/model/page=" + std::to_string(page_size);
}

struct Node {
  fs::path dir;
  std::unique_ptr<KvNodeServer> srv;
  std::string addr;
};

std::unique_ptr<Node> StartNode(const std::string& tag) {
  auto n = std::make_unique<Node>();
  n->dir = fs::temp_directory_path() / ("dfkv_node_" + tag);
  fs::remove_all(n->dir);
  fs::create_directories(n->dir);
  n->srv = std::make_unique<KvNodeServer>(n->dir.string(), 1ull << 30);
  EXPECT_EQ(n->srv->Start(0), Status::kOk);
  n->addr = "127.0.0.1:" + std::to_string(n->srv->port());
  return n;
}

}  // namespace

class KVClientTest : public ::testing::Test {
 protected:
  void TearDown() override {
    for (auto& n : nodes_) { n->srv->Stop(); fs::remove_all(n->dir); }
    nodes_.clear();
  }
  std::vector<std::unique_ptr<Node>> nodes_;
};

TEST_F(KVClientTest, PutGetExistRoundTripWithImmediateVisibility) {
  nodes_.push_back(StartNode("a1"));
  KVClient c({{ "a", nodes_[0]->addr }}, SelfHdr());
  std::string v(2000, 'q'); v[1000] = 'Z';
  ASSERT_TRUE(c.Put("glm-5.1/page_1_k", v.data(), v.size()));
  EXPECT_TRUE(c.Exist("glm-5.1/page_1_k"));  // sync: visible right away
  std::string out(v.size(), '\0');
  ASSERT_TRUE(c.Get("glm-5.1/page_1_k", &out[0], out.size()));
  EXPECT_EQ(out, v);
}

TEST_F(KVClientTest, RemoveRoundTripThenMiss) {
  nodes_.push_back(StartNode("r1"));
  KVClient c({{ "a", nodes_[0]->addr }}, SelfHdr());
  std::string v(1500, 'r');
  ASSERT_TRUE(c.Put("glm-5.1/rm_k", v.data(), v.size()));
  ASSERT_TRUE(c.Exist("glm-5.1/rm_k"));
  EXPECT_TRUE(c.Remove("glm-5.1/rm_k"));       // dropped
  EXPECT_FALSE(c.Exist("glm-5.1/rm_k"));       // gone now
  std::string out(v.size(), '\0');
  EXPECT_FALSE(c.Get("glm-5.1/rm_k", &out[0], out.size()));  // miss
  // Removing an absent key still "succeeds" (node confirmed: not present).
  EXPECT_TRUE(c.Remove("glm-5.1/rm_k"));
}

TEST_F(KVClientTest, BatchRemoveFansOutAcrossNodes) {
  nodes_.push_back(StartNode("b1"));
  nodes_.push_back(StartNode("b2"));
  std::vector<std::pair<std::string, std::string>> members = {
      {"b1", nodes_[0]->addr}, {"b2", nodes_[1]->addr}};
  KVClient c(members, SelfHdr());
  std::vector<std::string> keys;
  std::string v(256, 'z');
  for (int i = 0; i < 12; ++i) {
    std::string k = "glm-5.1/bk_" + std::to_string(i);
    keys.push_back(k);
    ASSERT_TRUE(c.Put(k, v.data(), v.size()));
  }
  for (const auto& k : keys) ASSERT_TRUE(c.Exist(k));
  std::vector<bool> r = c.BatchRemove(keys);
  ASSERT_EQ(r.size(), keys.size());
  for (bool ok : r) EXPECT_TRUE(ok);
  for (const auto& k : keys) EXPECT_FALSE(c.Exist(k));
}

TEST_F(KVClientTest, RefreshMembersDiscoversClusterFromSeed) {
  nodes_.push_back(StartNode("disc1"));
  nodes_.push_back(StartNode("disc2"));
  // Seed node advertises the full 2-node cluster.
  std::string list = "d1=" + nodes_[0]->addr + ",d2=" + nodes_[1]->addr;
  nodes_[0]->srv->set_members(list);

  // Client seeded with only node 0; discovers the rest.
  KVClient c({{ "d1", nodes_[0]->addr }}, SelfHdr());
  ASSERT_TRUE(c.RefreshMembers(nodes_[0]->addr));
  // Empty seed members => no-op false.
  EXPECT_FALSE(c.RefreshMembers(nodes_[1]->addr));  // node 1 advertises nothing

  // Membership applied: keys now route across both nodes and round-trip.
  for (int i = 0; i < 20; ++i) {
    std::string k = "disc_key_" + std::to_string(i);
    std::string v(256, static_cast<char>(i));
    ASSERT_TRUE(c.Put(k, v.data(), v.size())) << k;
    std::string out(v.size(), '\0');
    ASSERT_TRUE(c.Get(k, &out[0], out.size())) << k;
    EXPECT_EQ(out, v);
  }
}

TEST_F(KVClientTest, ActiveProbePopulatesPerPeerLatency) {
  nodes_.push_back(StartNode("probe1"));
  KVClient c({{ "p", nodes_[0]->addr }}, SelfHdr());
  c.StartProbe(10);  // probe each member every 10ms (off the datapath)
  std::this_thread::sleep_for(std::chrono::milliseconds(150));  // several rounds
  c.StopProbe();
  const std::string snap = c.MetricsSnapshot();
  // The probed node appears with a latency histogram + a max gauge.
  const std::string peer = "peer=\"" + nodes_[0]->addr + "\"";
  EXPECT_NE(snap.find("dfkv_client_peer_latency_seconds_count{" + peer + "}"),
            std::string::npos) << snap;
  EXPECT_NE(snap.find("dfkv_client_peer_latency_max_seconds{" + peer + "}"),
            std::string::npos);
}

TEST_F(KVClientTest, MissReturnsFalseNotError) {
  nodes_.push_back(StartNode("a2"));
  KVClient c({{ "a", nodes_[0]->addr }}, SelfHdr());
  EXPECT_FALSE(c.Exist("never/written_k"));
  std::string out(16, '\0');
  EXPECT_FALSE(c.Get("never/written_k", &out[0], out.size()));
}

TEST_F(KVClientTest, NamespaceMismatchTreatedAsMiss) {
  nodes_.push_back(StartNode("a3"));
  std::string v(512, 'x');
  KVClient writer({{ "a", nodes_[0]->addr }}, SelfHdr(/*page=*/64));
  ASSERT_TRUE(writer.Put("k_shared", v.data(), v.size()));
  // The layout belongs in the canonical namespace; a different page geometry
  // maps to a different native key and safely misses.
  KVClient reader({{ "a", nodes_[0]->addr }}, SelfHdr(/*page=*/32));
  std::string out(v.size(), '\0');
  EXPECT_FALSE(reader.Get("k_shared", &out[0], out.size()));
}

TEST_F(KVClientTest, RawPayloadRoundTripAddsNoEnvelope) {
  nodes_.push_back(StartNode("raw"));
  KVClient c({{"a", nodes_[0]->addr}}, SelfHdr());
  const char bytes[] = {'D', 'F', 'K', 'V', '\0', '\x01',
                        static_cast<char>(0xff), 'x'};
  const std::string value(bytes, sizeof(bytes));
  ASSERT_TRUE(c.Put("raw/object", value.data(), value.size()));

  const BlockKey key = ToBlockKey(SelfHdr(), "raw/object");
  const fs::path stored = nodes_[0]->dir / key.StoreKey();
  ASSERT_TRUE(fs::exists(stored));
  EXPECT_EQ(fs::file_size(stored), value.size());

  std::string output(value.size(), '\0');
  ASSERT_TRUE(c.Get("raw/object", output.data(), output.size()));
  EXPECT_EQ(output, value);
}

TEST_F(KVClientTest, RawPayloadCorruptionIsReturned) {
  // Values are deliberately opaque raw bytes. Transport/disk integrity is below
  // this API; the client does not prepend or validate a checksum envelope.
  nodes_.push_back(StartNode("a4"));
  KVClient c({{ "a", nodes_[0]->addr }}, SelfHdr());
  std::string v(1024, 'y');
  ASSERT_TRUE(c.Put("k_crc", v.data(), v.size()));
  // Corrupt one raw payload byte on disk.
  BlockKey bk = ToBlockKey(SelfHdr(), "k_crc");
  fs::path f = nodes_[0]->dir / bk.StoreKey();
  ASSERT_TRUE(fs::exists(f));
  { std::fstream io(f, std::ios::in | std::ios::out | std::ios::binary);
    io.seekp(100); char b = '!'; io.write(&b, 1); }
  std::string out(v.size(), '\0');
  EXPECT_TRUE(c.Get("k_crc", &out[0], out.size()));  // no checksum -> still a hit
  EXPECT_EQ(out[100], '!');                           // the corrupted byte is returned
}

TEST_F(KVClientTest, ScatterGatherMultiWindowSemanticsMatchTcp) {
  nodes_.push_back(StartNode("sg_multiwr_tcp"));
  KVClient c({{"a", nodes_[0]->addr}}, SelfHdr());

  constexpr size_t kSegments = 67;  // more than two 29-SGE WR windows
  std::vector<std::string> src(kSegments);
  std::vector<const void*> ptrs;
  std::vector<size_t> sizes;
  std::string expected;
  for (size_t i = 0; i < kSegments; ++i) {
    const size_t len = (i == 3 || i == 31) ? 0 : 5 + i % 17;
    src[i].resize(len);
    for (size_t j = 0; j < len; ++j)
      src[i][j] = static_cast<char>((i * 37 + j * 11) & 0xff);
    ptrs.push_back(src[i].data());
    sizes.push_back(len);
    expected.append(static_cast<const char*>(ptrs[i]), sizes[i]);
  }

  const auto put = c.BatchPutSg({{"tcp_multiwr", ptrs, sizes}});
  ASSERT_EQ(put.size(), 1u);
  ASSERT_TRUE(put[0]);
  EXPECT_EQ(nodes_[0]->srv->Count(), 1u);

  std::vector<std::string> dst(kSegments);
  std::vector<void*> dptrs;
  for (size_t i = 0; i < kSegments; ++i) {
    dst[i].assign(sizes[i], '\0');
    dptrs.push_back(dst[i].data());
  }
  std::vector<size_t> lens;
  const auto get = c.BatchGetAutoSg(
      {{"tcp_multiwr", dptrs, sizes}}, &lens);
  ASSERT_EQ(get.size(), 1u);
  ASSERT_TRUE(get[0]);
  ASSERT_EQ(lens.size(), 1u);
  EXPECT_EQ(lens[0], expected.size());
  std::string scattered;
  for (size_t i = 0; i < dst.size(); ++i) {
    EXPECT_EQ(dst[i], src[i]) << "segment " << i;
    scattered += dst[i];
  }
  EXPECT_EQ(scattered, expected);

  std::string ordinary(expected.size(), '\0');
  ASSERT_TRUE(c.Get("tcp_multiwr", ordinary.data(), ordinary.size()));
  EXPECT_EQ(ordinary, expected);
}

TEST_F(KVClientTest, TwoNodeConsistentHashCrossNodeRead) {
  nodes_.push_back(StartNode("n1"));
  nodes_.push_back(StartNode("n2"));
  std::vector<std::pair<std::string, std::string>> members = {
      {"n1", nodes_[0]->addr}, {"n2", nodes_[1]->addr}};
  KVClient writer(members, SelfHdr());
  const int N = 200;
  for (int i = 0; i < N; ++i) {
    std::string v = "val_" + std::to_string(i);
    ASSERT_TRUE(writer.Put("glm-5.1/p_" + std::to_string(i) + "_k",
                           v.data(), v.size()));
  }
  // both nodes hold some data (routing actually distributes)
  EXPECT_GT(nodes_[0]->srv->Count(), 0u);
  EXPECT_GT(nodes_[1]->srv->Count(), 0u);
  EXPECT_EQ(nodes_[0]->srv->Count() + nodes_[1]->srv->Count(), (size_t)N);

  // a SECOND client (same membership) reads every key back -> deterministic
  // routing + cross-node read works.
  KVClient reader(members, SelfHdr());
  for (int i = 0; i < N; ++i) {
    std::string key = "glm-5.1/p_" + std::to_string(i) + "_k";
    std::string want = "val_" + std::to_string(i);
    std::string out(want.size(), '\0');
    ASSERT_TRUE(reader.Get(key, &out[0], out.size())) << key;
    EXPECT_EQ(out, want);
  }
}
