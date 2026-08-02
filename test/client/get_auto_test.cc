// Variable-size get: KVClient::GetAuto(buffer) + BatchGetAuto over a REAL TCP
// loopback transport + cache-node server. These back the LMCache connector,
// which (unlike SGLang HiCache) stores variable-size "unfull" chunks and must
// read each back at its true stored length without knowing the size up front.
#include "client/kv_client.h"
#include "cache/kv_node_server.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {

std::string KeyNamespace(uint32_t page_size = 64) {
  return "test/model/page=" + std::to_string(page_size);
}

struct Node {
  fs::path dir;
  std::unique_ptr<KvNodeServer> srv;
  std::string addr;
};

std::unique_ptr<Node> StartNode(const std::string& tag) {
  auto n = std::make_unique<Node>();
  n->dir = fs::temp_directory_path() / ("dfkv_auto_" + tag);
  fs::remove_all(n->dir);
  fs::create_directories(n->dir);
  n->srv = std::make_unique<KvNodeServer>(n->dir.string(), 1ull << 30);
  EXPECT_EQ(n->srv->Start(0), Status::kOk);
  n->addr = "127.0.0.1:" + std::to_string(n->srv->port());
  return n;
}

}  // namespace

class GetAutoTest : public ::testing::Test {
 protected:
  void TearDown() override {
    for (auto& n : nodes_) { n->srv->Stop(); fs::remove_all(n->dir); }
    nodes_.clear();
  }
  std::vector<std::unique_ptr<Node>> nodes_;
};

TEST_F(GetAutoTest, ReadsBackTrueLengthWhenCapIsLarger) {
  nodes_.push_back(StartNode("a1"));
  KVClient c({{ "a", nodes_[0]->addr }}, KeyNamespace());
  std::string v(3000, 'q'); v[1500] = 'Z';
  ASSERT_TRUE(c.Put("k_partial", v.data(), v.size()));

  // Caller has a 4096-byte buffer but the stored payload is only 3000 bytes;
  // GetAuto reports the true length and fills exactly that many bytes.
  std::string buf(4096, '\0');
  size_t got = 0;
  ASSERT_TRUE(c.GetAuto("k_partial", &buf[0], buf.size(), &got));
  EXPECT_EQ(got, v.size());
  EXPECT_EQ(buf.compare(0, v.size(), v), 0);
}

TEST_F(GetAutoTest, ExactCapWorks) {
  nodes_.push_back(StartNode("a2"));
  KVClient c({{ "a", nodes_[0]->addr }}, KeyNamespace());
  std::string v(2048, 'x');
  ASSERT_TRUE(c.Put("k_exact", v.data(), v.size()));
  std::string buf(v.size(), '\0');
  size_t got = 0;
  ASSERT_TRUE(c.GetAuto("k_exact", &buf[0], buf.size(), &got));
  EXPECT_EQ(got, v.size());
  EXPECT_EQ(buf, v);
}

TEST_F(GetAutoTest, CapSmallerThanPayloadIsMiss) {
  nodes_.push_back(StartNode("a3"));
  KVClient c({{ "a", nodes_[0]->addr }}, KeyNamespace());
  std::string v(4096, 'y');
  ASSERT_TRUE(c.Put("k_big", v.data(), v.size()));
  std::string buf(1024, '\0');  // too small to hold the 4096B payload
  size_t got = 12345;
  EXPECT_FALSE(c.GetAuto("k_big", &buf[0], buf.size(), &got));
}

TEST_F(GetAutoTest, MissAndNamespaceMismatch) {
  nodes_.push_back(StartNode("a4"));
  KVClient writer({{ "a", nodes_[0]->addr }}, KeyNamespace(/*page=*/64));
  std::string v(512, 'z');
  ASSERT_TRUE(writer.Put("k_geo", v.data(), v.size()));

  // unwritten key => miss
  std::string buf(512, '\0');
  size_t got = 0;
  EXPECT_FALSE(writer.GetAuto("never_written", &buf[0], buf.size(), &got));

  // A different canonical namespace maps to a different key and misses.
  KVClient reader({{ "a", nodes_[0]->addr }}, KeyNamespace(/*page=*/32));
  EXPECT_FALSE(reader.GetAuto("k_geo", &buf[0], buf.size(), &got));
}

TEST_F(GetAutoTest, BatchMixedFullAndPartial) {
  nodes_.push_back(StartNode("n1"));
  nodes_.push_back(StartNode("n2"));
  std::vector<std::pair<std::string, std::string>> members = {
      {"n1", nodes_[0]->addr}, {"n2", nodes_[1]->addr}};
  KVClient c(members, KeyNamespace());

  const size_t kCap = 4096;  // uniform buffer capacity (== full chunk size)
  std::vector<std::string> keys;
  std::vector<std::string> vals;
  for (int i = 0; i < 32; ++i) {
    keys.push_back("auto_k_" + std::to_string(i));
    // even keys: full (4096), odd keys: partial (1024) — exercises both paths.
    size_t sz = (i % 2 == 0) ? kCap : 1024;
    std::string v(sz, static_cast<char>('A' + (i % 26)));
    vals.push_back(v);
    ASSERT_TRUE(c.Put(keys.back(), v.data(), v.size())) << keys.back();
  }

  std::vector<std::string> bufs(keys.size(), std::string(kCap, '\0'));
  std::vector<KvGetItem> items;
  for (size_t i = 0; i < keys.size(); ++i)
    items.push_back(KvGetItem{keys[i], &bufs[i][0], kCap});

  std::vector<size_t> lens;
  std::vector<bool> hit = c.BatchGetAuto(items, &lens);

  ASSERT_EQ(hit.size(), keys.size());
  ASSERT_EQ(lens.size(), keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    EXPECT_TRUE(hit[i]) << keys[i];
    EXPECT_EQ(lens[i], vals[i].size()) << keys[i];
    EXPECT_EQ(bufs[i].compare(0, vals[i].size(), vals[i]), 0) << keys[i];
  }

  // a missing key in the batch reports hit=false, len=0
  std::string miss_buf(kCap, '\0');
  std::vector<KvGetItem> mitems = {KvGetItem{"absent_key", &miss_buf[0], kCap}};
  std::vector<size_t> mlens;
  std::vector<bool> mhit = c.BatchGetAuto(mitems, &mlens);
  ASSERT_EQ(mhit.size(), 1u);
  EXPECT_FALSE(mhit[0]);
  EXPECT_EQ(mlens[0], 0u);
}
