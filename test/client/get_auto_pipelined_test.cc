// GetAuto over a PIPELINED (RDMA-like) transport must use the zero-copy
// RangeInto path, not the single-Range path (which on RDMA is capped at the
// 8 MiB control_cap and does a double copy). A fake pipelined transport lets
// us assert the routing hermetically; the real zero-copy bytes are covered by
// rdma_loopback_test.
#include "client/kv_client.h"
#include "transport/transport.h"
#include "client/key_map.h"

#include <gtest/gtest.h>
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
