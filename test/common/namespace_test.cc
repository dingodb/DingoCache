#include "common/namespace.h"

#include <cstdint>
#include <string>

#include <gtest/gtest.h>

namespace dfkv {
namespace {

std::string Hex(const std::string& value) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string out;
  out.reserve(value.size() * 2);
  for (unsigned char byte : value) {
    out.push_back(kDigits[byte >> 4]);
    out.push_back(kDigits[byte & 15]);
  }
  return out;
}

NamespaceDescriptor GoldenNamespace() {
  NamespaceDescriptor ns;
  ns.tenant_id = "tenant/a|b";
  ns.model_id = std::string("m\0x", 3);
  ns.model_revision = "rev:42";
  ns.pool_kind = PoolKind::kMamba;
  ns.pool_name = "mamba.temporal";
  ns.dtype = "bf16";
  ns.layout_fingerprint = UINT64_C(0x1122334455667788);
  ns.block_tokens = 64;
  ns.group_id = 7;
  ns.layer_begin = 10;
  ns.layer_count = 22;
  ns.tp = {8, 3};
  ns.dp = {2, -1};
  ns.pp = {4, 1};
  return ns;
}

TEST(Namespace, RejectsImplicitOrInvalidIdentity) {
  NamespaceDescriptor ns = GoldenNamespace();
  std::string error;
  EXPECT_EQ(ns.Validate(&error), Status::kOk) << error;

  ns.tenant_id.clear();
  EXPECT_EQ(ns.Validate(&error), Status::kInvalid);
  EXPECT_NE(error.find("tenant_id"), std::string::npos);

  ns = GoldenNamespace();
  ns.layout_fingerprint = 0;
  EXPECT_EQ(ns.Validate(&error), Status::kInvalid);

  ns = GoldenNamespace();
  ns.tp = {8, 8};
  EXPECT_EQ(ns.Validate(&error), Status::kInvalid);

  ns = GoldenNamespace();
  ns.pp = {0, -1};
  EXPECT_EQ(ns.Validate(&error), Status::kInvalid);
}

TEST(Namespace, LengthPrefixesMakeFieldBoundariesUnambiguous) {
  NamespaceDescriptor left = GoldenNamespace();
  NamespaceDescriptor right = GoldenNamespace();
  left.tenant_id = "ab";
  left.model_id = "c";
  right.tenant_id = "a";
  right.model_id = "bc";
  ASSERT_NE(left.Serialize(), right.Serialize());
}

TEST(Namespace, ReplicatedAndRankScopedAxesDiffer) {
  NamespaceDescriptor shared = GoldenNamespace();
  NamespaceDescriptor rank = GoldenNamespace();
  shared.tp = {8, -1};
  rank.tp = {8, 0};
  ASSERT_NE(shared.Serialize(), rank.Serialize());
}

TEST(Namespace, ObjectChunksAndComponentsAreIdentity) {
  const NamespaceDescriptor ns = GoldenNamespace();
  ObjectDescriptor object;
  object.logical_key = std::string("abc\0def", 7);
  object.component = "conv:1";
  object.chunk_index = 2;
  object.chunk_count = 4;
  std::string error;
  ASSERT_EQ(object.Validate(&error), Status::kOk) << error;

  const std::string baseline = object.Serialize(ns);
  object.component = "conv:2";
  EXPECT_NE(object.Serialize(ns), baseline);
  object.component = "conv:1";
  object.chunk_index = 3;
  EXPECT_NE(object.Serialize(ns), baseline);
}

TEST(Namespace, GoldenVector) {
  const NamespaceDescriptor ns = GoldenNamespace();
  ObjectDescriptor object;
  object.logical_key = std::string("abc\0def", 7);
  object.component = "conv:1";
  object.chunk_index = 2;
  object.chunk_count = 4;

  EXPECT_EQ(
      Hex(ns.Serialize()),
      "44464b564e5300020a00000074656e616e742f617c62030000006d007806000000"
      "7265763a343203000e0000006d616d62612e74656d706f72616c04000000626631"
      "36887766554433221140000000070000000a000000160000000800000003000000"
      "02000000ffffffff0400000001000000");
  EXPECT_EQ(
      Hex(object.Serialize(ns)),
      "44464b564f424a327300000044464b564e5300020a00000074656e616e742f617c"
      "62030000006d0078060000007265763a343203000e0000006d616d62612e74656d"
      "706f72616c0400000062663136887766554433221140000000070000000a000000"
      "16000000080000000300000002000000ffffffff04000000010000000700000061"
      "62630064656606000000636f6e763a310200000004000000");
}

}  // namespace
}  // namespace dfkv
