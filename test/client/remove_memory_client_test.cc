// Recent-REMOVE memory (CLI-02 stage M) end to end over TCP: a node with
// DFKV_REMOVE_MEMORY_MS answers kRemovedWithin; one without it answers as an
// older node does, which the client takes as "unknown" for every key, without
// charging the node and without asking it again for a while. No other op
// changes: a PUT after a REMOVE is stored and served.
#include "client/kv_client.h"
#include "cache/kv_node_server.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <thread>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {

struct Node {
  fs::path dir;
  std::unique_ptr<KvNodeServer> srv;
  std::string addr;
  explicit Node(const std::string& tag, const char* memory_ms) {
    if (memory_ms)
      ::setenv("DFKV_REMOVE_MEMORY_MS", memory_ms, 1);
    else
      ::unsetenv("DFKV_REMOVE_MEMORY_MS");
    dir = fs::temp_directory_path() / ("dfkv_rmem_" + tag);
    fs::remove_all(dir);
    fs::create_directories(dir);
    srv = std::make_unique<KvNodeServer>(dir.string(), 1ull << 30);
    ::unsetenv("DFKV_REMOVE_MEMORY_MS");
    EXPECT_EQ(srv->Start(0), Status::kOk);
    addr = "127.0.0.1:" + std::to_string(srv->port());
  }
  ~Node() {
    srv->Stop();
    fs::remove_all(dir);
  }
};

uint64_t Counter(const std::string& text, const std::string& series) {
  const size_t at = text.find("\n" + series + " ");
  if (at == std::string::npos) return 0;
  return std::stoull(text.substr(at + series.size() + 2));
}

}  // namespace

TEST(RemoveMemoryTcp, ANodeRemembersItsRemoves) {
  Node node("on", "600000");
  // The node knows the REMOVEs since its start: let 300 ms pass, so that
  // asking about the last 200 ms is within what it knows.
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  KVClient c({{"n", node.addr}}, "rmem/ns");
  const std::string v(64, 'v');
  ASSERT_TRUE(c.Put("gone", v.data(), v.size()));
  ASSERT_TRUE(c.Remove("gone"));
  ASSERT_TRUE(c.Remove("never-there"));  // a REMOVE of an absent key counts
  ASSERT_TRUE(c.Put("kept", v.data(), v.size()));
  std::vector<uint8_t> answers;
  uint32_t covered = 0;
  ASSERT_EQ(c.RemovedWithin(node.addr, {"gone", "never-there", "kept"}, 200,
                            &answers, &covered),
            Status::kOk);
  EXPECT_EQ(answers, (std::vector<uint8_t>{1, 1, 0}));
  // A PUT after the REMOVE is stored and served, and does not clear it.
  ASSERT_TRUE(c.Put("gone", v.data(), v.size()));
  std::string out(v.size(), '\0');
  EXPECT_TRUE(c.Get("gone", &out[0], out.size()));
  ASSERT_EQ(c.RemovedWithin(node.addr, {"gone"}, 200, &answers), Status::kOk);
  EXPECT_EQ(answers[0], 1);
  // Further back than the node has run: unknown.
  ASSERT_EQ(c.RemovedWithin(node.addr, {"gone"}, 3600000, &answers),
            Status::kOk);
  EXPECT_EQ(answers[0], 2);
  // Another namespace (tenant) did not remove it.
  KVClient other({{"n", node.addr}}, "rmem/other");
  ASSERT_EQ(other.RemovedWithin(node.addr, {"gone"}, 200, &answers),
            Status::kOk);
  EXPECT_EQ(answers[0], 0);
  // More keys than one request holds.
  std::vector<std::string> many;
  for (int i = 0; i < 2500; ++i) many.push_back("many-" + std::to_string(i));
  ASSERT_EQ(c.RemovedWithin(node.addr, many, 200, &answers), Status::kOk);
  ASSERT_EQ(answers.size(), many.size());
  for (uint8_t a : answers) EXPECT_EQ(a, 0);
  EXPECT_NE(node.srv->MetricsText().find("dfkv_remove_memory_entries"),
            std::string::npos);
}

// Off (the default): the node answers kInvalid; the client reads unknown,
// does not charge the node, and does not ask it again for a while.
TEST(RemoveMemoryTcp, ANodeWithoutTheMemoryIsUnknownAndNotCharged) {
  Node node("off", nullptr);
  KVClient c({{"n", node.addr}}, "rmem/ns");
  std::vector<uint8_t> answers;
  EXPECT_EQ(c.RemovedWithin(node.addr, {"a", "b"}, 1000, &answers),
            Status::kInvalid);
  EXPECT_EQ(answers, (std::vector<uint8_t>{2, 2}));
  const std::string before = node.srv->MetricsText();
  EXPECT_EQ(c.RemovedWithin(node.addr, {"a"}, 1000, &answers),
            Status::kInvalid);
  EXPECT_EQ(answers[0], 2);
  EXPECT_EQ(Counter(node.srv->MetricsText(), "dfkv_errors_total{op=\"any\",status=\"invalid\"}"),
            Counter(before, "dfkv_errors_total{op=\"any\",status=\"invalid\"}"))
      << "asked again within 60 s";
  EXPECT_EQ(node.srv->MetricsText().find("dfkv_remove_memory_entries"),
            std::string::npos);
  const std::string m = c.MetricsSnapshot();
  EXPECT_EQ(Counter(m, "dfkv_client_peer_marked_bad_total"), 0u);
  // Every other op is as before.
  const std::string v(8, 'v');
  EXPECT_TRUE(c.Put("x", v.data(), v.size()));
  EXPECT_TRUE(c.Remove("x"));
}
