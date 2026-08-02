// TDD R9 — KVClient batch APIs with concurrent fan-out across nodes.
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
std::string Hdr() { return "test/model"; }
struct Node { fs::path dir; std::unique_ptr<KvNodeServer> srv; std::string addr; };
std::unique_ptr<Node> Start(const std::string& tag) {
  auto n = std::make_unique<Node>();
  n->dir = fs::temp_directory_path() / ("dfkv_batch_" + tag);
  fs::remove_all(n->dir); fs::create_directories(n->dir);
  n->srv = std::make_unique<KvNodeServer>(n->dir.string(), 1ull << 30);
  EXPECT_EQ(n->srv->Start(0), Status::kOk);
  n->addr = "127.0.0.1:" + std::to_string(n->srv->port());
  return n;
}
}  // namespace

TEST(Batch, BatchPutGetExistAcrossTwoNodes) {
  auto a = Start("a"); auto b = Start("b");
  KVClient c({{"a", a->addr}, {"b", b->addr}}, Hdr());

  const int N = 300;
  std::vector<std::string> vals(N);
  std::vector<KvPutItem> puts;
  for (int i = 0; i < N; ++i) {
    vals[i] = "value_number_" + std::to_string(i);
    puts.push_back({"k" + std::to_string(i), vals[i].data(), vals[i].size()});
  }
  auto pr = c.BatchPut(puts);
  ASSERT_EQ(pr.size(), (size_t)N);
  for (bool ok : pr) EXPECT_TRUE(ok);

  // exist
  std::vector<std::string> keys;
  for (int i = 0; i < N; ++i) keys.push_back("k" + std::to_string(i));
  auto er = c.BatchExist(keys);
  for (bool e : er) EXPECT_TRUE(e);

  // get
  std::vector<std::string> outs(N);
  std::vector<KvGetItem> gets;
  for (int i = 0; i < N; ++i) {
    outs[i].assign(vals[i].size(), '\0');
    gets.push_back({"k" + std::to_string(i), &outs[i][0], outs[i].size()});
  }
  auto gr = c.BatchGet(gets);
  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(gr[i]) << i;
    EXPECT_EQ(outs[i], vals[i]) << i;
  }
  // both nodes actually hold data
  EXPECT_GT(a->srv->Count(), 0u);
  EXPECT_GT(b->srv->Count(), 0u);
  a->srv->Stop(); b->srv->Stop();
}

TEST(Batch, BatchGetMixedHitMiss) {
  auto a = Start("mix");
  KVClient c({{"a", a->addr}}, Hdr());
  std::string v = "hello";
  ASSERT_TRUE(c.Put("present", v.data(), v.size()));
  std::string o1(v.size(), '\0'), o2(v.size(), '\0');
  std::vector<KvGetItem> gets = {{"present", &o1[0], o1.size()},
                                 {"absent", &o2[0], o2.size()}};
  auto gr = c.BatchGet(gets);
  EXPECT_TRUE(gr[0]); EXPECT_EQ(o1, v);
  EXPECT_FALSE(gr[1]);  // miss
  a->srv->Stop();
}
