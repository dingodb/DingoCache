// Per-key outcomes of the C ABI's *_st batch calls: what the caller needs to
// tell "not there, recompute" from "could not ask, back off" and "no room".
// A scripted transport answers each key; both the TCP-shaped path (scalar
// bodies) and the pipelined path (grouped, the RDMA shape) are run.
#include "client/dfkv_c_api.h"
#include "client/key_map.h"
#include "client/kv_client.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr char kNamespace[] = "test/status";

class ScriptedTransport final : public dfkv::Transport {
 public:
  explicit ScriptedTransport(bool pipelined) : pipelined_(pipelined) {}
  // key -> (status, stored value) for reads; key -> status for writes.
  void Read(const std::string& key, dfkv::Status status,
            std::string value = {}) {
    reads_[Name(key)] = {status, std::move(value)};
  }
  void Write(const std::string& key, dfkv::Status status) {
    writes_[Name(key)] = status;
  }

  bool pipelined() const override { return pipelined_; }
  dfkv::Status Cache(const std::string&, const dfkv::BlockKey& key,
                     const void*, size_t) override {
    return WriteAnswer(key);
  }
  std::vector<dfkv::Status> CacheMany(
      const std::string&, const std::vector<dfkv::CacheItem>& items) override {
    std::vector<dfkv::Status> out;
    for (const auto& item : items) out.push_back(WriteAnswer(item.key));
    return out;
  }
  dfkv::Status Range(const std::string&, const dfkv::BlockKey&, uint64_t,
                     uint64_t, std::string*, uint64_t*) override {
    return dfkv::Status::kInvalid;
  }
  dfkv::Status Lookup(const std::string&, const dfkv::BlockKey&,
                      uint64_t*) override {
    return dfkv::Status::kInvalid;
  }
  dfkv::Status Exist(const std::string&, const dfkv::BlockKey&,
                     bool*) override {
    return dfkv::Status::kInvalid;
  }
  std::vector<dfkv::Status> RangeInto(
      const std::string&, const std::vector<dfkv::BlockKey>& keys,
      const std::vector<dfkv::RangeDst>& dsts,
      std::vector<uint64_t>* value_lens) override {
    std::lock_guard<std::mutex> lock(mu_);
    if (value_lens) value_lens->assign(keys.size(), 0);
    std::vector<dfkv::Status> out(keys.size(), dfkv::Status::kNotFound);
    for (size_t i = 0; i < keys.size(); ++i) {
      const auto found = reads_.find(keys[i].Filename());
      if (found == reads_.end()) continue;
      out[i] = found->second.first;
      if (out[i] != dfkv::Status::kOk) continue;
      const std::string& value = found->second.second;
      // As the TCP and RDMA transports: a value larger than the destination
      // is refused, a shorter one is read with its length.
      if (value.size() > dsts[i].n) {
        out[i] = dfkv::Status::kInvalid;
        continue;
      }
      std::memcpy(dsts[i].payload, value.data(), value.size());
      if (value_lens) (*value_lens)[i] = value.size();
    }
    return out;
  }

 private:
  static std::string Name(const std::string& key) {
    return dfkv::ToBlockKey(kNamespace, key).Filename();
  }
  dfkv::Status WriteAnswer(const dfkv::BlockKey& key) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto found = writes_.find(key.Filename());
    return found == writes_.end() ? dfkv::Status::kOk : found->second;
  }
  const bool pipelined_;
  std::mutex mu_;
  std::map<std::string, std::pair<dfkv::Status, std::string>> reads_;
  std::map<std::string, dfkv::Status> writes_;
};

dfkv_client_t Client(ScriptedTransport* transport) {
  return new dfkv::KVClient({{"node", "test:1"}}, kNamespace, transport, 1);
}

struct Keys {
  explicit Keys(std::vector<std::string> names) : names(std::move(names)) {
    for (const auto& name : this->names) {
      ptrs.push_back(name.data());
      lens.push_back(name.size());
    }
  }
  std::vector<std::string> names;
  std::vector<const void*> ptrs;
  std::vector<uint64_t> lens;
  int n() const { return static_cast<int>(names.size()); }
};

class CApiStatus : public ::testing::TestWithParam<bool> {};

TEST_P(CApiStatus, ReadsTellMissFromLengthFromRefusal) {
  ScriptedTransport transport(GetParam());
  transport.Read("hit", dfkv::Status::kOk, std::string(8, 'h'));
  transport.Read("short", dfkv::Status::kOk, std::string(4, 's'));
  transport.Read("long", dfkv::Status::kOk, std::string(16, 'l'));
  transport.Read("refused", dfkv::Status::kInvalid);
  transport.Read("rail", dfkv::Status::kNoCompatibleRail);
  dfkv_client_t client = Client(&transport);

  const Keys keys({"hit", "missing", "short", "long", "refused", "rail", ""});
  std::vector<std::vector<char>> bufs(keys.names.size(), std::vector<char>(8));
  std::vector<void*> ptrs;
  for (auto& buf : bufs) ptrs.push_back(buf.data());
  std::vector<uint64_t> sizes(keys.names.size(), 8);
  std::vector<int> hit(keys.names.size(), 9);
  std::vector<uint8_t> status(keys.names.size(), 99);

  ASSERT_EQ(dfkv_batch_get_st(client, keys.ptrs.data(), keys.lens.data(),
                              ptrs.data(), sizes.data(), keys.n(), hit.data(),
                              status.data()),
            0);
  EXPECT_EQ(hit, (std::vector<int>{1, 0, 0, 0, 0, 0, 0}));
  EXPECT_EQ(status, (std::vector<uint8_t>{
                        DFKV_ST_OK, DFKV_ST_NOT_FOUND, DFKV_ST_LENGTH_MISMATCH,
                        DFKV_ST_INVALID, DFKV_ST_INVALID,
                        DFKV_ST_NO_COMPATIBLE_RAIL, DFKV_ST_BAD_ARGUMENT}));
  // The plain form gives the same hits.
  std::vector<int> plain(keys.names.size(), 9);
  ASSERT_EQ(dfkv_batch_get(client, keys.ptrs.data(), keys.lens.data(),
                           ptrs.data(), sizes.data(), keys.n(), plain.data()),
            0);
  EXPECT_EQ(plain, hit);

  // Variable size: a shorter value fits; a longer one is refused.
  std::vector<uint64_t> lens(keys.names.size(), 77);
  ASSERT_EQ(dfkv_batch_get_auto_st(client, keys.ptrs.data(), keys.lens.data(),
                                   ptrs.data(), sizes.data(), keys.n(),
                                   hit.data(), lens.data(), status.data()),
            0);
  EXPECT_EQ(hit, (std::vector<int>{1, 0, 1, 0, 0, 0, 0}));
  EXPECT_EQ(lens[2], 4u);
  EXPECT_EQ(status, (std::vector<uint8_t>{
                        DFKV_ST_OK, DFKV_ST_NOT_FOUND, DFKV_ST_OK,
                        DFKV_ST_INVALID, DFKV_ST_INVALID,
                        DFKV_ST_NO_COMPATIBLE_RAIL, DFKV_ST_BAD_ARGUMENT}));

  // Scatter-gather: two 4-byte segments per key.
  std::vector<std::vector<void*>> segs(keys.names.size());
  std::vector<std::vector<uint64_t>> caps(keys.names.size(), {4, 4});
  std::vector<void**> dsts;
  std::vector<const uint64_t*> cap_ptrs;
  std::vector<int> counts(keys.names.size(), 2);
  for (size_t i = 0; i < keys.names.size(); ++i) {
    segs[i] = {bufs[i].data(), bufs[i].data() + 4};
    dsts.push_back(segs[i].data());
    cap_ptrs.push_back(caps[i].data());
  }
  ASSERT_EQ(dfkv_batch_get_auto_sg_st(client, keys.ptrs.data(),
                                      keys.lens.data(), dsts.data(),
                                      cap_ptrs.data(), counts.data(), keys.n(),
                                      hit.data(), lens.data(), status.data()),
            0);
  EXPECT_EQ(hit, (std::vector<int>{1, 0, 1, 0, 0, 0, 0}));
  EXPECT_EQ(status, (std::vector<uint8_t>{
                        DFKV_ST_OK, DFKV_ST_NOT_FOUND, DFKV_ST_OK,
                        DFKV_ST_INVALID, DFKV_ST_INVALID,
                        DFKV_ST_NO_COMPATIBLE_RAIL, DFKV_ST_BAD_ARGUMENT}));
  dfkv_close(client);
}

TEST_P(CApiStatus, PutsTellRoomFromQuota) {
  ScriptedTransport transport(GetParam());
  transport.Write("full", dfkv::Status::kCacheFull);
  transport.Write("quota", dfkv::Status::kQuotaExceeded);
  dfkv_client_t client = Client(&transport);
  const Keys keys({"ok", "full", "quota", "empty"});
  const char value[8] = {};
  std::vector<const void*> ptrs(keys.names.size(), value);
  std::vector<uint64_t> sizes{8, 8, 8, 0};
  std::vector<int> ok(keys.names.size(), 9);
  std::vector<uint8_t> status(keys.names.size(), 99);
  ASSERT_EQ(dfkv_batch_put_st(client, keys.ptrs.data(), keys.lens.data(),
                              ptrs.data(), sizes.data(), keys.n(), ok.data(),
                              status.data()),
            0);
  EXPECT_EQ(ok, (std::vector<int>{1, 0, 0, 0}));
  EXPECT_EQ(status, (std::vector<uint8_t>{DFKV_ST_OK, DFKV_ST_CACHE_FULL,
                                          DFKV_ST_QUOTA_EXCEEDED,
                                          DFKV_ST_BAD_ARGUMENT}));

  std::vector<const void*> one_segment(keys.names.size(), value);
  std::vector<const void**> segments;
  std::vector<const uint64_t*> segment_sizes;
  std::vector<int> counts(keys.names.size(), 1);
  for (size_t i = 0; i < keys.names.size(); ++i) {
    segments.push_back(&one_segment[i]);
    segment_sizes.push_back(&sizes[i]);
  }
  ASSERT_EQ(dfkv_batch_put_sg_st(client, keys.ptrs.data(), keys.lens.data(),
                                 segments.data(), segment_sizes.data(),
                                 counts.data(), keys.n(), ok.data(),
                                 status.data()),
            0);
  EXPECT_EQ(ok, (std::vector<int>{1, 0, 0, 0}));
  EXPECT_EQ(status, (std::vector<uint8_t>{DFKV_ST_OK, DFKV_ST_CACHE_FULL,
                                          DFKV_ST_QUOTA_EXCEEDED,
                                          DFKV_ST_BAD_ARGUMENT}));
  dfkv_close(client);
}

TEST_P(CApiStatus, AFailingNodeIsReportedThenSkipped) {
  ::setenv("DFKV_PEER_BAD_GRACE_MS", "0", 1);
  ScriptedTransport transport(GetParam());
  transport.Read("broken", dfkv::Status::kIOError);
  transport.Read("hit", dfkv::Status::kOk, std::string(8, 'h'));
  dfkv_client_t client = Client(&transport);
  ::unsetenv("DFKV_PEER_BAD_GRACE_MS");
  const Keys first({"broken"});
  const Keys second({"hit"});
  char buf[8];
  void* ptrs[] = {buf};
  const uint64_t sizes[] = {8};
  int hit[] = {9};
  uint8_t status[] = {99};
  ASSERT_EQ(dfkv_batch_get_st(client, first.ptrs.data(), first.lens.data(),
                              ptrs, sizes, 1, hit, status),
            0);
  EXPECT_EQ(status[0], DFKV_ST_IO_ERROR);
  ASSERT_EQ(dfkv_batch_get_st(client, second.ptrs.data(), second.lens.data(),
                              ptrs, sizes, 1, hit, status),
            0);
  EXPECT_EQ(hit[0], 0);
  EXPECT_EQ(status[0], DFKV_ST_PEER_COOLING);
  dfkv_close(client);
}

// Below the C layer: a batch read of an empty key is a bad argument and is
// not sent, as for a PUT.
TEST_P(CApiStatus, AnEmptyKeyIsABadArgumentForReadsAsForPuts) {
  ScriptedTransport transport(GetParam());
  transport.Read("", dfkv::Status::kOk, std::string(8, 'e'));
  dfkv::KVClient client({{"node", "test:1"}}, kNamespace, &transport, 1);
  char buf[8];
  std::vector<dfkv::ItemStatus> status;
  EXPECT_EQ(client.BatchGet({{"", buf, sizeof(buf)}}, &status),
            std::vector<bool>{false});
  EXPECT_EQ(status, std::vector<dfkv::ItemStatus>{
                        dfkv::ItemStatus::kBadArgument});
  std::vector<size_t> lens;
  EXPECT_EQ(client.BatchGetAuto({{"", buf, sizeof(buf)}}, &lens, &status),
            std::vector<bool>{false});
  EXPECT_EQ(status, std::vector<dfkv::ItemStatus>{
                        dfkv::ItemStatus::kBadArgument});
  const char value[8] = {};
  EXPECT_EQ(client.BatchPut({{"", value, sizeof(value)}}, &status),
            std::vector<bool>{false});
  EXPECT_EQ(status, std::vector<dfkv::ItemStatus>{
                        dfkv::ItemStatus::kBadArgument});
}

INSTANTIATE_TEST_SUITE_P(Paths, CApiStatus, ::testing::Values(false, true),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "Pipelined" : "Scalar";
                         });

TEST(CApiStatusGuard, EmptyRingAndMissingArray) {
  dfkv_client_options_v2 options{};
  options.struct_size = sizeof(options);
  options.abi_version = DFKV_CLIENT_ABI_VERSION_V2;
  options.members = "";
  options.key_namespace = kNamespace;
  options.key_namespace_len = sizeof(kNamespace) - 1;
  dfkv_client_t client = dfkv_open_v2(&options);
  ASSERT_NE(client, nullptr);
  const Keys keys({"a", ""});
  char buf[8];
  void* ptrs[] = {buf, buf};
  const uint64_t sizes[] = {8, 8};
  int hit[] = {9, 9};
  uint8_t status[] = {99, 99};
  ASSERT_EQ(dfkv_batch_get_st(client, keys.ptrs.data(), keys.lens.data(), ptrs,
                              sizes, 2, hit, status),
            0);
  EXPECT_EQ(status[0], DFKV_ST_NO_ROUTE);
  EXPECT_EQ(status[1], DFKV_ST_BAD_ARGUMENT);
  hit[0] = hit[1] = 9;
  EXPECT_EQ(dfkv_batch_get_st(client, keys.ptrs.data(), keys.lens.data(), ptrs,
                              sizes, 2, hit, nullptr),
            -1);
  EXPECT_EQ(hit[0], 0);
  EXPECT_EQ(hit[1], 0);
  dfkv_close(client);
}
}  // namespace
