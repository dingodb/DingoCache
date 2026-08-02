// Native string keys map to deterministic, domain-separated 128-bit identities.
#include "client/key_map.h"

#include <gtest/gtest.h>

#include <set>
#include <string>
#include <utility>

using dfkv::BlockKey;
using dfkv::KeyDomain;
using dfkv::ToBlockKey;

TEST(KeyMap, DeterministicSameKeySameDigest) {
  const BlockKey a = ToBlockKey("glm-5.1/abc123_k");
  const BlockKey b = ToBlockKey("glm-5.1/abc123_k");
  EXPECT_EQ(a, b);
  EXPECT_EQ(a.domain, KeyDomain::kNative);
}

TEST(KeyMap, CanonicalSha256GoldenVector) {
  // SHA-256("DFKVKEY2" || LE64(model_hash=0) || LE64(key_len) || key),
  // truncated to the first 128 bits and decoded as two big-endian words.
  const BlockKey key = ToBlockKey("glm-5.2/page_777_k");
  EXPECT_EQ(key.digest_hi, 0xa6454b4cd763924cULL);
  EXPECT_EQ(key.digest_lo, 0x5fad64e186790fa8ULL);
}

TEST(KeyMap, FullDigestParticipatesInIdentity) {
  std::set<std::pair<uint64_t, uint64_t>> digests;
  size_t nonzero_low_words = 0;
  for (int i = 0; i < 50000; ++i) {
    const BlockKey key =
        ToBlockKey("glm-5.2/pg_" + std::to_string(i) + "_k");
    digests.insert({key.digest_hi, key.digest_lo});
    if (key.digest_lo != 0) ++nonzero_low_words;
  }
  EXPECT_EQ(digests.size(), 50000u);
  EXPECT_GT(nonzero_low_words, 49900u);
}

TEST(KeyMap, DifferentModelHashGivesDifferentDigest) {
  const std::string key = "glm-5.2/pg_42_k";
  const BlockKey a1 = ToBlockKey(key, 111111);
  const BlockKey a2 = ToBlockKey(key, 111111);
  const BlockKey b = ToBlockKey(key, 222222);
  EXPECT_EQ(a1, a2);
  EXPECT_FALSE(a1 == b);
  EXPECT_FALSE(ToBlockKey(key) == b);
}

TEST(KeyMap, NoCrossModelHashCollisionInSample) {
  std::set<std::pair<uint64_t, uint64_t>> digests;
  const std::string key = "glm-5.2/shared_content_k";
  for (uint64_t model_hash = 1; model_hash <= 20000; ++model_hash) {
    const BlockKey mapped = ToBlockKey(key, model_hash);
    digests.insert({mapped.digest_hi, mapped.digest_lo});
  }
  EXPECT_EQ(digests.size(), 20000u);
}

TEST(KeyMap, FilenameAndStoreKeyUseFullDigest) {
  const BlockKey key{0x0123456789abcdefULL, 0xfedcba9876543210ULL};
  EXPECT_EQ(key.Filename(), "0123456789abcdeffedcba9876543210");
  EXPECT_EQ(key.StoreKey(),
            "blocks/01/0123/0123456789abcdeffedcba9876543210");
}

TEST(KeyMap, CompatibilityDomainCannotAliasNativeFilename) {
  const BlockKey native{1, 2, KeyDomain::kNative};
  const BlockKey legacy{1, 2, KeyDomain::kSgEngineV1};
  EXPECT_FALSE(native == legacy);
  EXPECT_NE(native.Filename(), legacy.Filename());
  EXPECT_NE(native.StoreKey(), legacy.StoreKey());
}
