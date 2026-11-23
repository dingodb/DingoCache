// TDD R2 — consistent hash (Ketama) for client-side routing across cache nodes.
#include "utils/con_hash.h"
#include "utils/md5.h"

#include <gtest/gtest.h>

#include <map>
#include <string>
#include <vector>

using dfkv::ConHash;

TEST(ConHash, EmptyRingReturnsFalse) {
  ConHash ring;
  ring.Build();
  std::string node;
  EXPECT_FALSE(ring.Lookup("anykey", &node));
}

TEST(ConHash, SameKeyAlwaysSameNode) {
  ConHash ring;
  ring.AddNode("nodeA");
  ring.AddNode("nodeB");
  ring.Build();
  std::string n1, n2;
  ASSERT_TRUE(ring.Lookup("glm-5.1/page_42_k", &n1));
  ASSERT_TRUE(ring.Lookup("glm-5.1/page_42_k", &n2));
  EXPECT_EQ(n1, n2);
}

TEST(ConHash, TwoNodesRoughlyBalanced) {
  ConHash ring;
  ring.AddNode("nodeA");
  ring.AddNode("nodeB");
  ring.Build();
  std::map<std::string, int> counts;
  const int N = 20000;
  for (int i = 0; i < N; ++i) {
    std::string node;
    ASSERT_TRUE(ring.Lookup("key_" + std::to_string(i), &node));
    counts[node]++;
  }
  ASSERT_EQ(counts.size(), 2u);
  // each within 35..65% — Ketama virtual nodes keep it balanced
  for (auto& [n, c] : counts) {
    double frac = double(c) / N;
    EXPECT_GT(frac, 0.35) << n;
    EXPECT_LT(frac, 0.65) << n;
  }
}

TEST(ConHash, AddingNodeRemapsOnlyAFraction) {
  ConHash before;
  before.AddNode("a"); before.AddNode("b"); before.AddNode("c");
  before.Build();
  ConHash after;
  after.AddNode("a"); after.AddNode("b"); after.AddNode("c"); after.AddNode("d");
  after.Build();

  const int N = 20000;
  int moved = 0;
  for (int i = 0; i < N; ++i) {
    std::string k = "key_" + std::to_string(i), n1, n2;
    ASSERT_TRUE(before.Lookup(k, &n1));
    ASSERT_TRUE(after.Lookup(k, &n2));
    if (n1 != n2) moved++;
  }
  double frac = double(moved) / N;
  // theory ~1/4 remap; allow generous band, must be well under "rehash-all"
  EXPECT_LT(frac, 0.45);
  EXPECT_GT(frac, 0.05);
}

TEST(ConHash, NodePointCountsReflectWeightAndSumToRing) {
  ConHash ring;
  ring.AddNode("light", 1);
  ring.AddNode("heavy", 3);
  ring.Build();
  auto pc = ring.NodePointCounts();
  ASSERT_EQ(pc.size(), 2u);
  // points sum to the realized ring size; heavy (weight 3) owns more vnodes
  EXPECT_EQ(pc["light"] + pc["heavy"], ring.RingSize());
  EXPECT_GT(pc["heavy"], pc["light"] * 2);
}

TEST(ConHash, WeightShiftsShareProportionally) {
  ConHash ring;
  ring.AddNode("light", 1);
  ring.AddNode("heavy", 4);
  ring.Build();
  std::map<std::string, int> counts;
  const int N = 20000;
  for (int i = 0; i < N; ++i) {
    std::string node;
    ASSERT_TRUE(ring.Lookup("k" + std::to_string(i), &node));
    counts[node]++;
  }
  EXPECT_GT(counts["heavy"], counts["light"] * 2);  // ~4:1 weight => heavy dominates
}

TEST(ConHash, LookupHashEqualsLookup) {
  ConHash ring;
  for (int i = 0; i < 16; ++i) ring.AddNode("node-" + std::to_string(i));
  ring.Build();
  for (int i = 0; i < 2000; ++i) {
    const std::string key = "key-" + std::to_string(i);
    std::string by_key;
    std::string by_hash;
    ASSERT_TRUE(ring.Lookup(key, &by_key));
    ASSERT_TRUE(ring.LookupHash(ConHash::KeyHash(key), &by_hash));
    ASSERT_EQ(by_key, by_hash) << key;
  }
}

namespace {
// The ring as Build made it before the sorted vector: a map point -> name,
// filled in member order, the last writer owning a shared point.
std::map<uint32_t, std::string> ReferenceRing(
    const std::vector<std::pair<std::string, int>>& nodes) {
  std::map<uint32_t, std::string> ring;
  for (const auto& [name, weight] : nodes) {
    const int w = weight < 1 ? 1 : (weight > 1024 ? 1024 : weight);
    for (int k = 0; k < 40 * w; ++k) {
      uint8_t d[16];
      const std::string s = name + "-" + std::to_string(k);
      dfkv::Md5(s.data(), s.size(), d);
      for (int h = 0; h < 4; ++h)
        ring[uint32_t(d[h * 4]) | (uint32_t(d[h * 4 + 1]) << 8) |
             (uint32_t(d[h * 4 + 2]) << 16) |
             (uint32_t(d[h * 4 + 3]) << 24)] = name;
    }
  }
  return ring;
}
}  // namespace

// Every route is the one the map-based ring gave, for several sizes and
// weights, keys and the wrap past the last point.
TEST(ConHash, RoutesAreThoseOfTheMapRing) {
  for (int n : {1, 3, 16, 64}) {
    std::vector<std::pair<std::string, int>> nodes;
    ConHash ring;
    for (int i = 0; i < n; ++i) {
      nodes.push_back({"node-" + std::to_string(i), 1 + i % 3});
      ring.AddNode(nodes.back().first, nodes.back().second);
    }
    ring.Build();
    const auto reference = ReferenceRing(nodes);
    ASSERT_EQ(ring.RingSize(), reference.size());
    auto route = [&](uint32_t h) {
      auto it = reference.lower_bound(h);
      return it == reference.end() ? reference.begin()->second : it->second;
    };
    std::string node;
    for (int k = 0; k < 20000; ++k) {
      const uint32_t h = ConHash::KeyHash("key-" + std::to_string(k));
      ASSERT_TRUE(ring.LookupHash(h, &node));
      ASSERT_EQ(node, route(h)) << n << " " << k;
    }
    for (uint32_t h : {0u, 0xFFFFFFFFu, reference.rbegin()->first,
                       reference.rbegin()->first + 1, reference.begin()->first}) {
      ASSERT_TRUE(ring.LookupHash(h, &node));
      EXPECT_EQ(node, route(h)) << h;
    }
    size_t vnodes = 0;
    for (const auto& [name, weight] : nodes) vnodes += 160u * weight;
    EXPECT_LE(ring.RingSize() + ring.Collisions(), vnodes);
  }
}
