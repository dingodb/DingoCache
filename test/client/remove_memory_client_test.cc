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
#include <future>
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
  explicit Node(const std::string& tag, const char* memory_ms,
                const char* write_entries = nullptr) {
    if (memory_ms)
      ::setenv("DFKV_REMOVE_MEMORY_MS", memory_ms, 1);
    else
      ::unsetenv("DFKV_REMOVE_MEMORY_MS");
    if (write_entries)
      ::setenv("DFKV_WRITE_MEMORY_ENTRIES", write_entries, 1);
    else
      ::unsetenv("DFKV_WRITE_MEMORY_ENTRIES");
    dir = fs::temp_directory_path() / ("dfkv_rmem_" + tag);
    fs::remove_all(dir);
    fs::create_directories(dir);
    srv = std::make_unique<KvNodeServer>(dir.string(), 1ull << 30);
    ::unsetenv("DFKV_REMOVE_MEMORY_MS");
    ::unsetenv("DFKV_WRITE_MEMORY_ENTRIES");
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

namespace dfkv {
class KvNodeServerTestPeer {
 public:
  static void Pause(KvNodeServer& s, int count = 0) {
    std::lock_guard<std::mutex> lk(s.ages_test_mu_);
    s.pause_ages_for_test_.store(true, std::memory_order_relaxed);
    s.ages_test_state_ = count;
  }
  static void WaitQuery(KvNodeServer& s) {
    std::unique_lock<std::mutex> lk(s.ages_test_mu_);
    s.ages_test_cv_.wait(lk, [&] { return s.ages_test_state_ == 1; });
  }
  static void Release(KvNodeServer& s) {
    std::lock_guard<std::mutex> lk(s.ages_test_mu_);
    s.pause_ages_for_test_.store(false, std::memory_order_relaxed);
    s.ages_test_state_++;
    s.ages_test_cv_.notify_all();
  }
  // Re-stamp only the two records of a real TCP PUT→REMOVE pair at one legal
  // RAW tick. The pair and subsequent op-26 request stay end to end, while
  // the same-tick precondition does not depend on scheduler luck.
  static bool MakeSameMillisecond(KvNodeServer& s, const BlockKey& key) {
    if (!s.remove_memory_ || !s.write_memory_) return false;
    auto publication = s.MemoryKey(key);
    const uint64_t now = RawMs();
    s.remove_memory_->Record(key, now);
    s.write_memory_->Record(key, now);
    return true;
  }
  static void MakePutBeforeRemoveWithNewerM(KvNodeServer& s,
                                            const BlockKey& key,
                                            const BlockKey& other) {
    const uint64_t now = RawMs();
    {
      auto publication = s.MemoryKey(key);
      s.write_memory_->Record(key, now - 20);   // completed PUT first
      s.remove_memory_->Record(key, now - 15);  // completed REMOVE second
    }
    // Models another bucket's caller having sampled a later RAW before this
    // op-26 request acquires M: M's effective watermark is ahead of W's.
    auto publication = s.MemoryKey(other);
    s.remove_memory_->Record(other, now + 30);
  }
};
}  // namespace dfkv

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

// Op 26 (CLI-35) end to end over TCP: the age of each key's latest REMOVE
// and latest PUT (a duplicate PUT included), on the node's raw clock; with
// the recent-PUT memory off, no write ages; without M, kInvalid (the client
// then asks op 25, which fails the same way: unknown).
TEST(RemoveMemoryTcp, OpTwentySixAnswersRemoveAndWriteAges) {
  Node node("ages", "600000", "65536");
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  KVClient c({{"n", node.addr}}, "rmem/ages");
  const std::string v(64, 'v');
  ASSERT_TRUE(c.Remove("rewritten"));
  // This legal REMOVE→PUT rewrite must have a strictly older REMOVE. A tie
  // deliberately has the conservative verdict exercised separately below.
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  ASSERT_TRUE(c.Put("rewritten", v.data(), v.size()));
  ASSERT_TRUE(c.Put("old", v.data(), v.size()));
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  ASSERT_TRUE(c.Put("old", v.data(), v.size()));  // a duplicate: a PUT
  ASSERT_TRUE(c.Remove("gone"));
  std::vector<KVClient::MemoryAnswer> a;
  ASSERT_EQ(c.RemovedWithinAges(node.addr, {"rewritten", "old", "gone", "never"},
                                250, &a),
            Status::kOk);
  ASSERT_EQ(a.size(), 4u);
  EXPECT_GE(a[0].removed_age, 200u);
  EXPECT_LT(a[0].removed_age, 250u);
  EXPECT_LT(a[0].written_age, a[0].removed_age)
      << "the PUT came strictly after the REMOVE";
  EXPECT_EQ(KVClient::RewriteVerdict(node.addr, {{node.addr, a[0]}}),
            KVClient::Rewrite::kAfterEvery);
  std::string served(v.size(), '\0');
  EXPECT_TRUE(c.Get("rewritten", &served[0], served.size()));
  EXPECT_LT(a[1].written_age, 100u) << "the duplicate PUT is the latest";
  EXPECT_EQ(a[1].removed_age, kRemovedAgeNone);
  EXPECT_LT(a[2].removed_age, 100u);
  EXPECT_EQ(a[2].written_age, kRemovedAgeNone);
  EXPECT_EQ(a[3].removed_age, kRemovedAgeNone);
  EXPECT_EQ(a[3].written_age, kRemovedAgeNone);
  EXPECT_LE(a[0].sent_ms, a[0].answered_ms);

  // Exercise real TCP PUT→REMOVE, then pin both records at one legal RAW tick
  // so the required same-ms input is deterministic. A separate result vector
  // preserves the original four-key response and its checks above.
  const std::string tie_key = "same-ms-tie";
  ASSERT_TRUE(c.Put(tie_key, v.data(), v.size()));
  ASSERT_TRUE(c.Remove(tie_key));
  ASSERT_TRUE(KvNodeServerTestPeer::MakeSameMillisecond(
      *node.srv, ToBlockKey("rmem/ages", tie_key)));
  std::vector<KVClient::MemoryAnswer> tie;
  ASSERT_EQ(c.RemovedWithinAges(node.addr, {tie_key}, 250, &tie), Status::kOk);
  ASSERT_EQ(tie.size(), 1u);
  ASSERT_EQ(tie[0].removed_age, tie[0].written_age)
      << "the fixture must actually produce a same-tick M/W sample";
  EXPECT_EQ(KVClient::RewriteVerdict(node.addr, {{node.addr, tie[0]}}),
            KVClient::Rewrite::kRemoved)
      << "PUT→REMOVE at equal ages must not be misread as a rewrite";

  EXPECT_EQ(Counter(node.srv->MetricsText(),
                    "dfkv_removed_within_ages_requests_total"),
            2u);
  EXPECT_NE(node.srv->MetricsText().find("dfkv_write_memory_entries"),
            std::string::npos);
  // Beyond what the node knows: unknown.
  ASSERT_EQ(c.RemovedWithinAges(node.addr, {"gone"}, 3600000, &a), Status::kOk);
  EXPECT_EQ(a[0].removed_age, kRemovedAgeUnknown);
}
TEST(RemoveMemoryTcp, OpTwentySixUsesOneEffectiveSnapshotTime) {
  Node node("ages-watermark", "600000", "65536");
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  KVClient c({{"n", node.addr}}, "rmem/watermark");
  KvNodeServerTestPeer::MakePutBeforeRemoveWithNewerM(
      *node.srv, ToBlockKey("rmem/watermark", "before-remove"),
      ToBlockKey("rmem/watermark", "unrelated-remove"));
  std::vector<KVClient::MemoryAnswer> a;
  ASSERT_EQ(c.RemovedWithinAges(node.addr, {"before-remove"}, 250, &a),
            Status::kOk);
  ASSERT_EQ(a.size(), 1u);
  EXPECT_GT(a[0].written_age, a[0].removed_age)
      << "the older PUT must stay older even when M had a newer watermark";
  EXPECT_EQ(KVClient::RewriteVerdict(node.addr, {{node.addr, a[0]}}),
            KVClient::Rewrite::kRemoved)
        << "PUT before REMOVE must not be admitted as a later rewrite";
}

TEST(RemoveMemoryTcp, OpTwentySixWithoutTheWriteMemoryOrWithoutM) {
  {
    Node node("ages-now", "600000");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    KVClient c({{"n", node.addr}}, "rmem/ages");
    const std::string v(8, 'v');
    ASSERT_TRUE(c.Put("k", v.data(), v.size()));
    std::vector<KVClient::MemoryAnswer> a;
    ASSERT_EQ(c.RemovedWithinAges(node.addr, {"k"}, 200, &a), Status::kOk);
    EXPECT_EQ(a[0].written_age, kRemovedAgeNone);
    EXPECT_EQ(node.srv->MetricsText().find("dfkv_write_memory_entries"),
              std::string::npos);
  }
  {
    Node node("ages-off", nullptr, "65536");  // W needs M
    KVClient c({{"n", node.addr}}, "rmem/ages");
    std::vector<KVClient::MemoryAnswer> a;
    EXPECT_EQ(c.RemovedWithinAges(node.addr, {"k"}, 200, &a), Status::kInvalid);
    EXPECT_EQ(a[0].removed_age, kRemovedAgeUnknown);
  }
}

TEST(RemoveMemoryTcp, OpTwentySixAcquiresEachSharedLockOnlyOnce) {
  Node node("ages-buckets", "600000", "65536");
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  KVClient c({{"n", node.addr}}, "rmem/ages");
  std::vector<std::string> keys;
  for (int i = 0; i < 128; ++i) keys.push_back("bucket-" + std::to_string(i));
  std::vector<KVClient::MemoryAnswer> a;
  // 128 distinct keys must collide in the 64 mutex buckets. Full-key
  // de-duplication alone used to self-deadlock at their first collision.
  ASSERT_EQ(c.RemovedWithinAges(node.addr, keys, 200, &a), Status::kOk);
  EXPECT_EQ(a.size(), keys.size());
}

// Q1 pauses before its joint M/W snapshot. A PUT and another REMOVE then
// complete and the millisecond tables are separated. The original read could
// mistake that history for a PUT after the later REMOVE.
TEST(RemoveMemoryTcp, OpTwentySixDoesNotReverseAConcurrentRewrite) {
  Node node("ages-race", "600000", "65536");
  KVClient c({{"n", node.addr}}, "rmem/ages-race");
  KVClient c_query({{"n", node.addr}}, "rmem/ages-race");
  KVClient c_put({{"n", node.addr}}, "rmem/ages-race");
  KVClient c_remove({{"n", node.addr}}, "rmem/ages-race");
  const std::string v(64, 'v');
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  // Start with no prior PUT: the only write that may appear is the one racing
  // the first query below, which must not leak into its older snapshot.
  ASSERT_TRUE(c.Remove("race"));
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  ASSERT_TRUE(c.Remove("advance"));  // M's table time is ahead of W's.
  KvNodeServerTestPeer::Pause(*node.srv);
  std::vector<KVClient::MemoryAnswer> before, after;
  auto first = std::async(std::launch::async, [&] {
    return c_query.RemovedWithinAges(node.addr, {"race"}, 200, &before);
  });
  KvNodeServerTestPeer::WaitQuery(*node.srv);
  auto put = std::async(std::launch::async, [&] {
    EXPECT_TRUE(c_put.Put("race", v.data(), v.size()));
  });
  auto remove = std::async(std::launch::async, [&] {
    EXPECT_TRUE(c_remove.Remove("race"));
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  // More recent M activity makes the original separate-table age inversion
  // deterministic, even if a W shard shares the writer's millisecond.
  std::vector<uint8_t> m_answer;
  ASSERT_EQ(c.RemovedWithin(node.addr, {"race"}, 200, &m_answer),
            Status::kOk);
  KvNodeServerTestPeer::Release(*node.srv);
  ASSERT_EQ(first.get(), Status::kOk);
  ASSERT_TRUE(before[0].removed_age <= 1000u)
      << "covered REMOVE, or no REMOVE, must be known for this race check";
  put.get();
  remove.get();
  ASSERT_EQ(c.RemovedWithinAges(node.addr, {"race"}, 200, &after),
            Status::kOk);
  EXPECT_EQ(before[0].written_age, kRemovedAgeNone)
      << "the query must not mix a subsequent write with its older REMOVE: "
      << before[0].removed_age << ", " << before[0].written_age;
}
