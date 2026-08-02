// TDD R14 — concurrency stress (race detection under TSan) + mixed-batch with a
// down node (partial results, no crash).
#include "client/kv_client.h"
#include "cache/kv_node_server.h"

#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {
std::string Hdr() { return "test/model"; }
struct Node { fs::path dir; std::unique_ptr<KvNodeServer> srv; std::string addr; };
std::unique_ptr<Node> Start(const std::string& tag) {
  auto n = std::make_unique<Node>();
  n->dir = fs::temp_directory_path() / ("dfkv_stress_" + tag);
  fs::remove_all(n->dir); fs::create_directories(n->dir);
  n->srv = std::make_unique<KvNodeServer>(n->dir.string(), 1ull << 30);
  EXPECT_EQ(n->srv->Start(0), Status::kOk);
  n->addr = "127.0.0.1:" + std::to_string(n->srv->port());
  return n;
}
}  // namespace

TEST(Stress, ConcurrentPutGetExistAndSetMembers) {
  auto a = Start("a"); auto b = Start("b");
  KVClient c({{"a", a->addr}}, Hdr());  // start with a
  std::atomic<bool> go{false};
  std::atomic<int> ok{0};
  std::vector<std::thread> ts;
  const int T = 8, OPS = 200;
  for (int t = 0; t < T; ++t) {
    ts.emplace_back([&, t] {
      while (!go.load()) {}
      std::string v(64, char('a' + t));
      for (int i = 0; i < OPS; ++i) {
        std::string k = "t" + std::to_string(t) + "_" + std::to_string(i);
        if (c.Put(k, v.data(), v.size())) {
          std::string out(v.size(), '\0');
          if (c.Get(k, &out[0], out.size()) && out == v) ok.fetch_add(1);
          c.Exist(k);
        }
      }
    });
  }
  // a separate thread hot-swaps membership concurrently (exercises ring_mu_)
  std::thread mem([&] {
    while (!go.load()) {}
    for (int i = 0; i < 20; ++i) {
      c.SetMembers(std::vector<std::pair<std::string,std::string>>{{"a", a->addr}, {"b", b->addr}});
      c.SetMembers(std::vector<std::pair<std::string,std::string>>{{"a", a->addr}});
    }
  });
  go = true;
  for (auto& th : ts) th.join();
  mem.join();
  EXPECT_GT(ok.load(), 0);  // many ops succeeded; primary check = no race/crash (TSan)
  a->srv->Stop(); b->srv->Stop();
}

TEST(Stress, BatchGetWithOneNodeDownReturnsPartial) {
  auto a = Start("up"); auto b = Start("down");
  std::vector<std::pair<std::string, std::string>> members = {
      {"a", a->addr}, {"b", b->addr}};
  KVClient c(members, Hdr());
  const int N = 100;
  std::vector<std::string> vals(N);
  std::vector<KvPutItem> puts;
  for (int i = 0; i < N; ++i) {
    vals[i] = "v" + std::to_string(i);
    puts.push_back({"k" + std::to_string(i), vals[i].data(), vals[i].size()});
  }
  ASSERT_EQ(c.BatchPut(puts).size(), (size_t)N);

  size_t on_b = b->srv->Count();
  ASSERT_GT(on_b, 0u);
  b->srv->Stop();  // take node b down

  std::vector<std::string> outs(N);
  std::vector<KvGetItem> gets;
  for (int i = 0; i < N; ++i) { outs[i].assign(vals[i].size(), '\0');
    gets.push_back({"k" + std::to_string(i), &outs[i][0], outs[i].size()}); }
  auto gr = c.BatchGet(gets);  // must not hang/crash; partial hits
  int hits = 0, misses = 0;
  for (int i = 0; i < N; ++i) { if (gr[i]) { ++hits; EXPECT_EQ(outs[i], vals[i]); } else ++misses; }
  EXPECT_GT(hits, 0);    // node a's keys still served
  EXPECT_GT(misses, 0);  // node b's keys now miss (no replica) — degrade, not fail
  a->srv->Stop();
}
