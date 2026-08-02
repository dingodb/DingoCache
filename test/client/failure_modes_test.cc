// Boundary and robustness: unreachable node, namespace isolation, real MLA
// page size (2.74 MiB), zero-length values, and capacity-bounded server.
#include "client/kv_client.h"
#include "cache/kv_node_server.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <cstdlib>
#include <string>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {
std::string KeyNamespace(uint16_t version = 1, uint32_t page = 64) {
  return "test/model/version=" + std::to_string(version) +
         "/page=" + std::to_string(page);
}
struct Node {
  fs::path dir; std::unique_ptr<KvNodeServer> srv; std::string addr;
};
std::unique_ptr<Node> Start(const std::string& tag,
                            uint64_t cap = (1ull << 30),
                            const char* engine = nullptr) {
  auto n = std::make_unique<Node>();
  n->dir = fs::temp_directory_path() / ("dfkv_fm_" + tag);
  fs::remove_all(n->dir); fs::create_directories(n->dir);
  const char* prior = ::getenv("DFKV_STORE_ENGINE");
  const bool had_prior = prior != nullptr;
  const std::string prior_value = had_prior ? prior : "";
  if (engine != nullptr) ::setenv("DFKV_STORE_ENGINE", engine, 1);
  n->srv = std::make_unique<KvNodeServer>(n->dir.string(), cap);
  if (engine != nullptr) {
    if (had_prior)
      ::setenv("DFKV_STORE_ENGINE", prior_value.c_str(), 1);
    else
      ::unsetenv("DFKV_STORE_ENGINE");
  }
  EXPECT_EQ(n->srv->Start(0), Status::kOk);
  n->addr = "127.0.0.1:" + std::to_string(n->srv->port());
  return n;
}
}  // namespace

TEST(FailureModes, UnreachableNodeReturnsFalseNoCrash) {
  // route to a closed port; client must degrade gracefully, not throw/crash.
  KVClient c({{"dead", "127.0.0.1:9"}}, KeyNamespace());
  std::string v(64, 'z');
  EXPECT_FALSE(c.Put("k", v.data(), v.size()));
  std::string out(64, '\0');
  EXPECT_FALSE(c.Get("k", &out[0], out.size()));
  EXPECT_FALSE(c.Exist("k"));
}

TEST(FailureModes, EmptyMembershipReturnsFalse) {
  KVClient c({}, KeyNamespace());
  EXPECT_FALSE(c.Exist("k"));
}

TEST(FailureModes, VersionDriftTreatedAsMiss) {
  auto n = Start("ver");
  KVClient writer({{"a", n->addr}}, KeyNamespace(/*version=*/1));
  std::string v(128, 'a');
  ASSERT_TRUE(writer.Put("kv", v.data(), v.size()));
  KVClient reader({{"a", n->addr}}, KeyNamespace(/*version=*/2));
  std::string out(128, '\0');
  EXPECT_FALSE(reader.Get("kv", &out[0], out.size()));
  n->srv->Stop();
}

TEST(FailureModes, RealMlaPageSizeRoundTrip) {
  auto n = Start("mlapage");
  KVClient c({{"a", n->addr}}, KeyNamespace());
  // GLM-5.1 MLA: 78 layers * 64 page * (512+64) * 1B = 2,875,392 bytes/page
  const size_t kPage = 78ull * 64 * (512 + 64);
  std::string v(kPage, '\0');
  for (size_t i = 0; i < kPage; i += 997) v[i] = char(i & 0xFF);
  ASSERT_TRUE(c.Put("page", v.data(), v.size()));
  EXPECT_TRUE(c.Exist("page"));
  std::string out(kPage, '\1');
  ASSERT_TRUE(c.Get("page", &out[0], out.size()));
  EXPECT_EQ(out, v);
  n->srv->Stop();
}

TEST(FailureModes, ZeroLengthValueRoundTrip) {
  auto n = Start("zero");
  KVClient c({{"a", n->addr}}, KeyNamespace());
  ASSERT_TRUE(c.Put("empty", nullptr, 0));
  EXPECT_TRUE(c.Exist("empty"));
  ASSERT_TRUE(c.Get("empty", nullptr, 0));  // hit, no bytes
  n->srv->Stop();
}

TEST(FailureModes, ServerCapacityBoundsUsage) {
  // tiny cap: server evicts; usage stays bounded, recent key survives.
  auto n = Start("cap", /*cap=*/size_t{3000}, /*engine=*/"file");
  KVClient c({{"a", n->addr}}, KeyNamespace());
  std::string v(1000, 'x');
  for (int i = 0; i < 20; ++i) {
    ASSERT_TRUE(c.Put("k" + std::to_string(i), v.data(), v.size()));
  }
  EXPECT_LE(n->srv->UsedBytes(), uint64_t{3000});
  std::string out(1000, '\0');
  EXPECT_TRUE(c.Get("k19", &out[0], out.size()));  // most-recent survives
  n->srv->Stop();
}
