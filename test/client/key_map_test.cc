// Native string keys map to deterministic tenant-scoped 192-bit identities.
#include "client/key_map.h"
#include "common/namespace.h"

#include <gtest/gtest.h>

#include <set>
#include <string>
#include <utility>

using dfkv::BlockKey;
using dfkv::ToBlockKey;
using dfkv::NamespaceDescriptor;

TEST(KeyMap, DeterministicSameKeySameDigest) {
  const BlockKey a = ToBlockKey("test/model", "glm-5.1/abc123_k");
  const BlockKey b = ToBlockKey("test/model", "glm-5.1/abc123_k");
  EXPECT_EQ(a, b);
}

TEST(KeyMap, CanonicalSha256GoldenVector) {
  // SHA-256("DFKVKEY2" || LE64(namespace_len) || namespace ||
  //        LE64(object_len) || object), first 128 bits as big-endian words.
  const BlockKey key = ToBlockKey("test/model", "glm-5.2/page_777_k");
  EXPECT_EQ(key.digest_hi, 0x1dabbca41aba2b02ULL);
  EXPECT_EQ(key.digest_lo, 0xa865be51fa395d70ULL);
  EXPECT_EQ(key.tenant_hash, 0xfa1cde78082f951eULL);
}

TEST(KeyMap, LengthFramingPreservesBinaryNamespaceBoundaries) {
  const std::string binary_namespace("model\0revision", 14);
  EXPECT_FALSE(ToBlockKey(binary_namespace, "page") ==
               ToBlockKey("model", "page"));
  EXPECT_FALSE(ToBlockKey("a", "bc") == ToBlockKey("ab", "c"));
}

TEST(KeyMap, FullDigestParticipatesInIdentity) {
  std::set<std::pair<uint64_t, uint64_t>> digests;
  size_t nonzero_low_words = 0;
  for (int i = 0; i < 50000; ++i) {
    const BlockKey key =
        ToBlockKey("test/model", "glm-5.2/pg_" + std::to_string(i) + "_k");
    digests.insert({key.digest_hi, key.digest_lo});
    if (key.digest_lo != 0) ++nonzero_low_words;
  }
  EXPECT_EQ(digests.size(), 50000u);
  EXPECT_GT(nonzero_low_words, 49900u);
}

TEST(KeyMap, DifferentNamespaceGivesDifferentDigest) {
  const std::string key = "glm-5.2/pg_42_k";
  const BlockKey a1 = ToBlockKey("model/111111", key);
  const BlockKey a2 = ToBlockKey("model/111111", key);
  const BlockKey b = ToBlockKey("model/222222", key);
  EXPECT_EQ(a1, a2);
  EXPECT_FALSE(a1 == b);
  EXPECT_FALSE(ToBlockKey("test/model", key) == b);
}

TEST(KeyMap, NoCrossNamespaceCollisionInSample) {
  std::set<std::pair<uint64_t, uint64_t>> digests;
  const std::string key = "glm-5.2/shared_content_k";
  for (uint64_t id = 1; id <= 20000; ++id) {
    const BlockKey mapped =
        ToBlockKey("model/" + std::to_string(id), key);
    digests.insert({mapped.digest_hi, mapped.digest_lo});
  }
  EXPECT_EQ(digests.size(), 20000u);
}

TEST(KeyMap, FilenameAndStoreKeyUseTenantAndFullDigest) {
  const BlockKey key{0x0123456789abcdefULL, 0xfedcba9876543210ULL,
                     0x1122334455667788ULL};
  EXPECT_EQ(
      key.Filename(),
      "11223344556677880123456789abcdeffedcba9876543210");
  EXPECT_EQ(
      key.StoreKey(),
      "blocks/11/1122/11223344556677880123456789abcdeffedcba9876543210");
  BlockKey parsed;
  ASSERT_TRUE(BlockKey::ParseFilename(key.Filename(), &parsed));
  EXPECT_EQ(parsed, key);
  EXPECT_FALSE(BlockKey::ParseFilename(
      "11223344556677880123456789ABCDEFFEDCBA9876543210", &parsed));
  EXPECT_FALSE(BlockKey::ParseFilename(key.Filename() + "0", &parsed));
}

TEST(KeyMap, CanonicalNamespaceUsesOnlyTenantFieldForTenantHash) {
  NamespaceDescriptor ns;
  ns.tenant_id = "tenant-a";
  ns.model_id = "model";
  ns.model_revision = "rev";
  ns.pool_name = "kv";
  ns.dtype = "bytes";
  ns.layout_fingerprint = 1;
  ns.block_tokens = 16;
  ns.layer_count = 1;
  const std::string bytes = ns.Serialize();
  const BlockKey first = ToBlockKey(bytes, "object-a");
  const BlockKey second = ToBlockKey(bytes, "object-b");
  EXPECT_EQ(first.tenant_hash, 0x6164af84acbc9adeULL);
  EXPECT_EQ(first.tenant_hash, second.tenant_hash);
  ns.model_id = "other-model";
  EXPECT_EQ(ToBlockKey(ns.Serialize(), "object-c").tenant_hash,
            first.tenant_hash);
  ns.tenant_id = "tenant-b";
  EXPECT_EQ(ToBlockKey(ns.Serialize(), "object-c").tenant_hash,
            0x27a61068bf52563bULL);
}

TEST(KeyMap, SemanticallyMalformedCanonicalNamespaceHashesAllBytes) {
  NamespaceDescriptor ns;
  ns.tenant_id = "tenant-a";
  ns.model_id = "model";
  ns.model_revision = "rev";
  ns.pool_name = "kv";
  ns.dtype = "bytes";
  ns.layout_fingerprint = 1;
  ns.block_tokens = 16;
  ns.layer_count = 1;
  std::string malformed = ns.Serialize();
  ASSERT_GE(malformed.size(), 4u);
  // Corrupt the replicated pp_rank sentinel into an out-of-range rank.
  malformed[malformed.size() - 4] = 1;
  size_t offset = 0;
  size_t length = 0;
  EXPECT_FALSE(dfkv::CanonicalTenantIdentity(malformed, &offset, &length));
  EXPECT_NE(ToBlockKey(malformed, "object").tenant_hash,
            dfkv::TenantHash(std::string("tenant-a")));
}
