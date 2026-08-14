#include "client/kv_client.h"

#include <gtest/gtest.h>

#include <mutex>
#include <string>
#include <vector>

namespace dfkv {
namespace {

class RecordingTransport final : public Transport {
 public:
  Status Cache(const std::string&, const BlockKey&, const void*, size_t) override {
    return Status::kOk;
  }
  Status Range(const std::string&, const BlockKey&, uint64_t, uint64_t,
               std::string*, uint64_t*) override {
    return Status::kNotFound;
  }
  Status Lookup(const std::string&, const BlockKey&, uint64_t*) override {
    return Status::kNotFound;
  }
  Status Exist(const std::string&, const BlockKey&, bool*) override {
    return Status::kNotFound;
  }
  std::vector<Status> CacheFromMulti(
      const std::string& node,
      const std::vector<CacheSrcMulti>& srcs,
      std::string* /*out_dev*/ = nullptr) override {
    {
      std::lock_guard<std::mutex> lock(mu_);
      selected_nodes_.push_back(node);
    }
    return std::vector<Status>(srcs.size(), Status::kOk);
  }

  std::vector<std::string> SelectedNodes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return selected_nodes_;
  }

 private:
  mutable std::mutex mu_;
  std::vector<std::string> selected_nodes_;
};

TEST(EndpointOwnership, ShardingOneKeyNeverChangesCanonicalNode) {
  RecordingTransport transport;
  KVClient client({{"cache-a", "endpoint-a"}, {"cache-b", "endpoint-b"}},
                  "test/model", &transport);
  constexpr size_t kItems = 80;  // five default per-endpoint shards
  std::vector<char> payload(kItems, 'x');
  std::vector<KvPutItemSg> items(kItems);
  for (size_t i = 0; i < kItems; ++i) {
    items[i].key = "the-same-canonical-key";
    items[i].ptrs = {&payload[i]};
    items[i].sizes = {1};
  }

  const std::vector<bool> result = client.BatchPutSg(items);
  ASSERT_EQ(result.size(), kItems);
  for (bool ok : result) EXPECT_TRUE(ok);

  const std::vector<std::string> selected = transport.SelectedNodes();
  ASSERT_GT(selected.size(), 1u);  // the group was actually split
  for (const std::string& node : selected) EXPECT_EQ(node, selected.front());
  EXPECT_TRUE(selected.front() == "endpoint-a" ||
              selected.front() == "endpoint-b");
}

}  // namespace
}  // namespace dfkv
