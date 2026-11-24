// The node's in-memory key tables (parked RETAIN SETs, the PUT key stripes,
// the recent reads of DFKV_RAM_PROMOTE=second) spread keys whose fields are
// related. Their hash was digest_hi ^ digest_lo * C, which put every key
// {id * C + 1, id} into a handful of buckets: measured with 100000 such keys,
// filling the parked-SET table took 28 s instead of 7 ms (one chain of 50000),
// 8 of the 256 PUT stripes held them all, and two passes of recent reads took
// 114 s instead of 37 ms.
#include "cache/kv_node_server.h"
#include "cache/recent_reads.h"
#include "common/block_key_hash.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace dfkv {
class KvNodeServerTestPeer {
 public:
  using PendingRetain = decltype(KvNodeServer::pending_retain_);
  using PutKeys = decltype(KvNodeServer::PutKeyStripe::keys);
  static size_t Stripes() { return KvNodeServer::kPutKeyStripes; }
};
}  // namespace dfkv

using namespace dfkv;  // NOLINT
using Peer = KvNodeServerTestPeer;

namespace {

constexpr uint64_t kKeys = 100000;

std::vector<BlockKey> RelatedKeys() {
  std::vector<BlockKey> keys;
  for (uint64_t id = 0; id < kKeys; ++id)
    keys.push_back(BlockKey{id * 0x9e3779b97f4a7c15ULL + 1, id, id & 3});
  return keys;
}

// The hash every one of these tables uses spreads related keys as a uniform
// hash would: about 1 - e^-0.76 = 53% of 2^17 buckets.
template <class Hash>
size_t Buckets(const std::vector<BlockKey>& keys) {
  std::set<size_t> buckets;
  for (const auto& k : keys) buckets.insert(Hash{}(k) & ((1u << 17) - 1));
  return buckets.size();
}

TEST(NodeKeyHash, ParkedRetainSetsSpreadRelatedKeys) {
  using Hash = Peer::PendingRetain::hasher;
  EXPECT_GT(Buckets<Hash>(RelatedKeys()), kKeys / 2);
}

TEST(NodeKeyHash, RecentReadsSpreadRelatedKeys) {
  EXPECT_GT(Buckets<RecentReads::Hasher>(RelatedKeys()), kKeys / 2);
}

// A stripe is taken by the hash modulo the stripe count: related keys use
// every stripe, none much more than its share (390 on average).
TEST(NodeKeyHash, PutStripesSpreadRelatedKeys) {
  using Hash = Peer::PutKeys::hasher;
  std::vector<size_t> per_stripe(Peer::Stripes());
  for (const auto& k : RelatedKeys()) ++per_stripe[Hash{}(k) % Peer::Stripes()];
  EXPECT_EQ(std::count(per_stripe.begin(), per_stripe.end(), size_t{0}), 0);
  EXPECT_LT(*std::max_element(per_stripe.begin(), per_stripe.end()),
            2 * kKeys / Peer::Stripes());
}

TEST(NodeKeyHash, EveryFieldCounts) {
  const BlockKeyMixHash hash;
  const BlockKey key{1, 2, 3};
  EXPECT_NE(hash(key), hash(BlockKey{0, 2, 3}));
  EXPECT_NE(hash(key), hash(BlockKey{1, 0, 3}));
  EXPECT_NE(hash(key), hash(BlockKey{1, 2, 0}));
  EXPECT_EQ(hash(key), hash(BlockKey{1, 2, 3}));
}

// std::random_device may throw where there is no entropy source; the seed,
// taken inside the noexcept hash, then falls back instead of ending the
// process.
uint64_t NoEntropy() { throw std::runtime_error("no entropy source"); }
uint64_t FixedEntropy() { return 42; }

TEST(NodeKeyHash, ASeedWithoutEntropyFallsBack) {
  EXPECT_EQ(BlockKeyMixHash::MakeSeed(&FixedEntropy), 42u);
  const uint64_t a = BlockKeyMixHash::MakeSeed(&NoEntropy);
  EXPECT_NE(a, 0u);
  // Not a constant: the clock moves between two fallbacks.
  uint64_t b = a;
  for (int i = 0; i < 1000 && b == a; ++i) b = BlockKeyMixHash::MakeSeed(&NoEntropy);
  EXPECT_NE(a, b);
}

}  // namespace
