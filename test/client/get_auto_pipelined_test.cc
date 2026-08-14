// GetAuto over a PIPELINED (RDMA-like) transport must use the zero-copy
// RangeInto path, not the single-Range path (which on RDMA is capped at the
// 8 MiB control_cap and does a double copy). A fake pipelined transport lets
// us assert the routing hermetically; the real zero-copy bytes are covered by
// rdma_loopback_test.
#include "client/kv_client.h"
#include "transport/transport.h"
#include "client/key_map.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <limits>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace dfkv;  // NOLINT

namespace {

// Pipelined transport that serves stored blobs only through RangeInto and counts
// each entry point, so a test can prove GetAuto never falls back to Range here.
struct FakePipelinedTransport : Transport {
  std::map<std::string, std::string> blobs;  // key.Filename() -> payload
  std::string dead;
  int range_calls = 0, rangeinto_calls = 0;
  std::vector<DestinationMemoryKind> last_range_kinds;
  std::vector<std::vector<DestinationMemoryKind>> last_multi_kinds;
  int register_calls = 0;

  bool RegisterMemory(void*, size_t) override {
    ++register_calls;
    return true;
  }

  bool pipelined() const override { return true; }

  Status Range(const std::string&, const BlockKey&, uint64_t, uint64_t,
               std::string*, uint64_t*) override {
    ++range_calls;
    return Status::kIOError;
  }
  Status Cache(const std::string&, const BlockKey&, const void*, size_t) override {
    return Status::kOk;
  }
  Status Lookup(const std::string&, const BlockKey& key,
                uint64_t* value_len) override {
    auto it = blobs.find(key.Filename());
    if (it == blobs.end()) return Status::kNotFound;
    *value_len = it->second.size();
    return Status::kOk;
  }
  Status Exist(const std::string&, const BlockKey&, bool* e) override {
    *e = false;
    return Status::kOk;
  }
  std::vector<Status> RangeInto(const std::string& node,
                                const std::vector<BlockKey>& keys,
                                const std::vector<RangeDst>& dsts,
                                std::vector<uint64_t>* value_lens) override {
    ++rangeinto_calls;
    last_range_kinds.clear();
    last_range_kinds.reserve(dsts.size());
    for (const auto& dst : dsts)
      last_range_kinds.push_back(dst.memory_kind);
    value_lens->assign(keys.size(), 0);
    std::vector<Status> r(keys.size(), Status::kNotFound);
    for (size_t i = 0; i < keys.size(); ++i) {
      if (node == dead) { r[i] = Status::kIOError; continue; }
      auto it = blobs.find(keys[i].Filename());
      if (it == blobs.end()) continue;
      const std::string& payload = it->second;
      (*value_lens)[i] = payload.size();
      const size_t n = std::min(payload.size(), dsts[i].n);
      if (n && dsts[i].payload)
        std::memcpy(dsts[i].payload, payload.data(), n);
      r[i] = Status::kOk;
    }
    return r;
  }

  std::vector<Status> RangeIntoMulti(
      const std::string&, const std::vector<BlockKey>& keys,
      const std::vector<RangeDstMulti>& dsts,
      std::vector<size_t>* value_lens,
      std::string* /*out_dev*/ = nullptr) override {
    value_lens->assign(keys.size(), 0);
    last_multi_kinds.clear();
    last_multi_kinds.reserve(dsts.size());
    std::vector<Status> result(keys.size(), Status::kNotFound);
    for (size_t i = 0; i < keys.size(); ++i) {
      std::vector<DestinationMemoryKind> kinds;
      kinds.reserve(dsts[i].payloads.size());
      for (const auto& segment : dsts[i].payloads)
        kinds.push_back(segment.memory_kind);
      last_multi_kinds.push_back(std::move(kinds));

      auto it = blobs.find(keys[i].Filename());
      if (it == blobs.end()) continue;
      (*value_lens)[i] = it->second.size();
      size_t offset = 0;
      for (const auto& segment : dsts[i].payloads) {
        const size_t copy =
            std::min(segment.second, it->second.size() - offset);
        if (copy)
          std::memcpy(segment.first, it->second.data() + offset, copy);
        offset += copy;
        if (offset == it->second.size()) break;
      }
      result[i] = Status::kOk;
    }
    return result;
  }
};

KVClient MakeClient(FakePipelinedTransport* t) {
  return KVClient({{"n", "10.0.0.1:1"}}, "test/model", t);
}
}  // namespace

TEST(GetAutoPipelined, VoidBufferUsesRangeIntoNotRange) {
  FakePipelinedTransport t;
  KVClient c = MakeClient(&t);
  std::string v(1234, 'p');
  t.blobs[ToBlockKey("test/model", "kv").Filename()] = v;

  std::vector<char> buf(4096);
  size_t got = 0;
  ASSERT_TRUE(c.GetAuto("kv", buf.data(), buf.size(), &got));
  EXPECT_EQ(got, 1234u);
  EXPECT_EQ(std::memcmp(buf.data(), v.data(), 1234), 0);
  EXPECT_EQ(t.rangeinto_calls, 1);
  EXPECT_EQ(t.range_calls, 0) << "pipelined GetAuto must not fall back to Range";
}

TEST(GetAutoPipelined, StringOverloadTrimsToTrueLength) {
  FakePipelinedTransport t;
  KVClient c = MakeClient(&t);
  std::string v(777, 'q');
  t.blobs[ToBlockKey("test/model", "ks").Filename()] = v;

  std::string out;
  ASSERT_TRUE(c.GetAuto("ks", &out, 8192));
  EXPECT_EQ(out.size(), 777u);
  EXPECT_EQ(out, v);
  EXPECT_EQ(t.rangeinto_calls, 1);
  EXPECT_EQ(t.range_calls, 0);
}

TEST(GetAutoPipelined, ValueLargerThanEightMiBIsServed) {
  // The exact regression: a value above the single-Range control_cap (8 MiB)
  // used to miss on GetAuto but hit on batch read. Through RangeInto it serves.
  FakePipelinedTransport t;
  KVClient c = MakeClient(&t);
  const size_t big = (10u << 20);  // 10 MiB > 8 MiB control_cap
  std::string v(big, 'B');
  t.blobs[ToBlockKey("test/model", "kbig").Filename()] = v;

  std::string out;
  ASSERT_TRUE(c.GetAuto("kbig", &out, 16u << 20));
  EXPECT_EQ(out.size(), big);
  EXPECT_EQ(out.front(), 'B');
  EXPECT_EQ(out.back(), 'B');
  EXPECT_EQ(t.range_calls, 0);
}

TEST(GetAutoPipelined, CapSmallerThanPayloadIsMiss) {
  FakePipelinedTransport t;
  KVClient c = MakeClient(&t);
  t.blobs[ToBlockKey("test/model", "kbig2").Filename()] =
      std::string(5000, 'z');

  std::vector<char> buf(1024);
  size_t got = 0;
  EXPECT_FALSE(c.GetAuto("kbig2", buf.data(), buf.size(), &got));
  std::string s;
  EXPECT_FALSE(c.GetAuto("kbig2", &s, 1024));
  EXPECT_TRUE(s.empty());
}

TEST(GetAutoPipelined, MissReturnsFalse) {
  FakePipelinedTransport t;
  KVClient c = MakeClient(&t);
  std::vector<char> buf(64);
  size_t got = 0;
  EXPECT_FALSE(c.GetAuto("absent", buf.data(), buf.size(), &got));
  std::string s;
  EXPECT_FALSE(c.GetAuto("absent", &s, 64));
}

TEST(GetAutoPipelined, RegisteredHostSubrangesClassifyWithoutCudaOnGet) {
  FakePipelinedTransport t;
  KVClient c = MakeClient(&t);
  std::vector<char> pool(4096, '\0');
  ASSERT_TRUE(c.RegisterMemory(pool.data(), 1024));
  ASSERT_TRUE(c.RegisterMemory(pool.data(), pool.size()));
  ASSERT_EQ(t.register_calls, 2);

  ASSERT_TRUE(c.RegisterMemory(pool.data() + 512, 1024));
  ASSERT_EQ(t.register_calls, 3);
  const uintptr_t address = reinterpret_cast<uintptr_t>(pool.data());
  const size_t wrapping_size =
      std::numeric_limits<uintptr_t>::max() - address + 1;
  EXPECT_FALSE(c.RegisterMemory(pool.data(), wrapping_size));
  EXPECT_EQ(t.register_calls, 3);
  const std::string scalar_value(127, 's');
  t.blobs[ToBlockKey("test/model", "registered-scalar").Filename()] =
      scalar_value;
  size_t got = 0;
  char* const scalar_destination = pool.data() + 777;
  ASSERT_TRUE(c.GetAuto("registered-scalar", scalar_destination,
                        scalar_value.size(), &got));
  EXPECT_EQ(got, scalar_value.size());
  ASSERT_EQ(t.last_range_kinds.size(), 1);
  EXPECT_EQ(t.last_range_kinds[0], DestinationMemoryKind::kHost);

  const std::string sg_value = "registered-host-scatter-gather";
  t.blobs[ToBlockKey("test/model", "registered-sg").Filename()] = sg_value;
  KvGetItemSg item;
  item.key = "registered-sg";
  item.dsts = {pool.data() + 31, pool.data() + 503, pool.data() + 2049};
  item.caps = {5, 9, sg_value.size() - 14};
  std::vector<size_t> lengths;
  ASSERT_EQ(c.BatchGetAutoSg({item}, &lengths), std::vector<bool>({true}));
  ASSERT_EQ(lengths, std::vector<size_t>({sg_value.size()}));
  ASSERT_EQ(t.last_multi_kinds.size(), 1);
  EXPECT_EQ(t.last_multi_kinds[0],
            std::vector<DestinationMemoryKind>(
                3, DestinationMemoryKind::kHost));
}

TEST(GetAutoPipelined,
     DefaultMultiPublicationUsesScratchAcrossOddSgBoundariesAndFailure) {
  FakePipelinedTransport t;
  const BlockKey key = ToBlockKey("test/model", "default-sg");
  std::string expected(16391, '\0');
  for (size_t i = 0; i < expected.size(); ++i)
    expected[i] = static_cast<char>((29u * i + 0x63u) & 0xffu);
  t.blobs[key.Filename()] = expected;

  constexpr size_t kGuard = 19;
  constexpr char kSentinel = static_cast<char>(0xd3);
  const std::vector<size_t> segment_sizes{4093, 4109,
                                          expected.size() - 8202};
  std::vector<std::string> allocations;
  RangeDstMulti destination;
  allocations.reserve(segment_sizes.size());
  for (size_t size : segment_sizes) {
    allocations.emplace_back(size + 2 * kGuard, kSentinel);
    destination.payloads.emplace_back(
        allocations.back().data() + kGuard, size,
        DestinationMemoryKind::kHost);
  }

  std::vector<size_t> lengths;
  const auto statuses = t.Transport::RangeIntoMulti(
      "10.0.0.1:1", {key}, {destination}, &lengths);
  EXPECT_EQ(statuses, std::vector<Status>({Status::kOk}));
  EXPECT_EQ(lengths, std::vector<size_t>({expected.size()}));
  size_t offset = 0;
  for (size_t i = 0; i < segment_sizes.size(); ++i) {
    EXPECT_EQ(allocations[i].substr(0, kGuard),
              std::string(kGuard, kSentinel))
        << "leading guard " << i;
    EXPECT_EQ(allocations[i].substr(kGuard, segment_sizes[i]),
              expected.substr(offset, segment_sizes[i]))
        << "payload " << i;
    EXPECT_EQ(allocations[i].substr(kGuard + segment_sizes[i]),
              std::string(kGuard, kSentinel))
        << "trailing guard " << i;
    offset += segment_sizes[i];
  }

  // The base publication contract reads into operation-owned scratch first.
  // A failed attempt therefore reports its exact status without exposing any
  // partially received bytes to caller SG buffers.
  for (auto& allocation : allocations)
    std::fill(allocation.begin(), allocation.end(), kSentinel);
  t.dead = "failed-rail";
  lengths.clear();
  const auto failed = t.Transport::RangeIntoMulti(
      t.dead, {key}, {destination}, &lengths);
  EXPECT_EQ(failed, std::vector<Status>({Status::kIOError}));
  EXPECT_EQ(lengths, std::vector<size_t>({0}));
  for (size_t i = 0; i < allocations.size(); ++i)
    EXPECT_EQ(allocations[i],
              std::string(allocations[i].size(), kSentinel))
        << "failed destination " << i;
}
