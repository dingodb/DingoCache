// TDD R2 — consistent hash (Ketama) for client-side routing across cache nodes.
#include "utils/con_hash.h"

#include <gtest/gtest.h>

#include <map>
#include <string>

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
