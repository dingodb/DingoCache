// Small client-library fixes (review item CLI-21): a batch PUT whose
// transport returns fewer statuses than keys, a batch PUT to one node in
// shards like the scatter-gather PUT, and a local CUDA publication failure
// that says nothing about the node.
#include "client/dedup_token.h"
#include "client/kv_client.h"
#include "client/peer_health.h"
#include "transport/transport.h"

#include <gtest/gtest.h>

#include <mutex>
#include <string>
#include <vector>

using namespace dfkv;  // NOLINT

namespace {

class PutTransport final : public Transport {
 public:
  // Statuses CacheFrom returns per call: all kOk, or only the first `keep`.
  size_t keep = SIZE_MAX;
  std::mutex mu;
  std::vector<size_t> call_sizes;

  bool pipelined() const override { return true; }  // the RDMA batch path
  Status Cache(const std::string&, const BlockKey&, const void*,
               size_t) override {
    return Status::kOk;
  }
  Status Range(const std::string&, const BlockKey&, uint64_t, uint64_t,
               std::string* out, uint64_t* value_len) override {
    // A 4-byte value, read whole.
    if (out) *out = "abcd";
    if (value_len) *value_len = 4;
    return Status::kOk;
  }
  Status Lookup(const std::string&, const BlockKey&, uint64_t*) override {
    return Status::kNotFound;
  }
  Status Exist(const std::string&, const BlockKey&, bool*) override {
    return Status::kNotFound;
  }
  std::vector<Status> CacheFrom(const std::string&,
                                const std::vector<CacheSrc>& srcs) override {
    {
      std::lock_guard<std::mutex> lk(mu);
      call_sizes.push_back(srcs.size());
    }
    return std::vector<Status>(std::min(keep, srcs.size()), Status::kOk);
  }
};

std::vector<KvPutItem> Items(size_t n, std::vector<std::string>* keys) {
  static const char value[] = "value";
  keys->clear();
  for (size_t i = 0; i < n; ++i) keys->push_back("k" + std::to_string(i));
  std::vector<KvPutItem> items;
  for (const auto& k : *keys) items.push_back(KvPutItem{k, value, 5});
  return items;
}

}  // namespace

TEST(ClientBatchPut, FewerStatusesThanKeysAreFailures) {
  PutTransport t;
  t.keep = 0;  // a transport that answers with no statuses at all
  KVClient c({{"n", "node:1"}}, "test/ns", &t, 1);
  std::vector<std::string> keys;
  const auto items = Items(4, &keys);
  std::vector<ItemStatus> st;
  const auto ok = c.BatchPut(items, &st);
  EXPECT_EQ(ok, std::vector<bool>(4, false));
  for (const auto s : st) EXPECT_EQ(s, ItemStatus::kInvalid);
}

TEST(ClientBatchPut, OneNodesKeysGoOutInShards) {
  // As BatchPutSg: a large group to one node is split across connections
  // instead of draining key by key on one.
  PutTransport t;
  KVClient c({{"n", "node:1"}}, "test/ns", &t, 1);
  std::vector<std::string> keys;
  const auto items = Items(256, &keys);
  EXPECT_EQ(c.BatchPut(items), std::vector<bool>(256, true));
  size_t total = 0;
  for (size_t n : t.call_sizes) total += n;
  EXPECT_EQ(total, 256u);
  EXPECT_GT(t.call_sizes.size(), 1u);
}

TEST(ClientDevicePublish, ALocalPublicationFailureSaysNothingAboutThePeer) {
  // The node answered; the bytes could not be copied to the (device)
  // destination here. Before, that was kIOError and cooled the node.
  PutTransport t;
  char dst[4] = {'?', '?', '?', '?'};
  std::vector<uint64_t> lens;
  const auto st = t.RangeInto(
      "node:1", {BlockKey{1, 2}},
      {RangeDst{dst, sizeof(dst), DestinationMemoryKind::kDevice}}, &lens);
  ASSERT_EQ(st.size(), 1u);
  EXPECT_NE(st[0], Status::kOk);  // host memory is no device destination
  PeerOutcome outcome;
  outcome.Add(st[0]);
  EXPECT_FALSE(outcome.failed) << "status " << static_cast<int>(st[0]);
}

TEST(DedupPayloadBound, AtMostHalfTheArenaAndWhatTheLengthFieldHolds) {
  // The slot records a payload length in 32 bits: an arena over 8 GiB let
  // a payload of 4 GiB or more be claimed and its length truncated.
  EXPECT_EQ(DedupMaxPayload(512ull << 20), 256ull << 20);
  EXPECT_EQ(DedupMaxPayload(8ull << 30), 4ull << 30 > UINT32_MAX
                                             ? static_cast<size_t>(UINT32_MAX)
                                             : static_cast<size_t>(4ull << 30));
  EXPECT_EQ(DedupMaxPayload(64ull << 30), static_cast<size_t>(UINT32_MAX));
}
