// STO-17: route anchors share an owner without changing actual object identity.
// Both the scalar and pipelined native paths exercise owner, drain, absent,
// previous-owner read-back, substitute M checks and reroute cleanup. The C ABI
// cases cover every additive anchored form, fallback and per-item validation.
#include "client/dfkv_c_api.h"
#include "client/key_map.h"
#include "client/kv_client.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "common/membership.h"
#include "transport/transport.h"
#include "transport/wire.h"
#include "utils/con_hash.h"
#include "utils/net_util.h"

using namespace dfkv;  // NOLINT

namespace {

class NodesTransport final : public Transport {
 public:
  using Key = std::pair<uint64_t, uint64_t>;
  explicit NodesTransport(bool pipelined) : pipelined_(pipelined) {}

  static Key K(const BlockKey& key) { return {key.digest_hi, key.digest_lo}; }
  struct RewriteHistory {
    uint32_t removed_age = kRemovedAgeNone;
    uint32_t written_age = kRemovedAgeNone;
  };

  bool pipelined() const override { return pipelined_; }
  Status Cache(const std::string& node, const BlockKey& key, const void* data,
               size_t len) override {
    std::lock_guard<std::mutex> lk(mu);
    if (down.count(node)) return Status::kIOError;
    store[node][K(key)].assign(static_cast<const char*>(data), len);
    return Status::kOk;
  }
  Status Range(const std::string& node, const BlockKey& key, uint64_t offset,
               uint64_t length, std::string* out,
               uint64_t* value_len) override {
    std::lock_guard<std::mutex> lk(mu);
    if (down.count(node)) return Status::kIOError;
    const auto it = store[node].find(K(key));
    if (it == store[node].end()) return Status::kNotFound;
    if (value_len) *value_len = it->second.size();
    if (offset > it->second.size()) return Status::kInvalid;
    *out = it->second.substr(offset, length);
    return Status::kOk;
  }
  Status Lookup(const std::string& node, const BlockKey& key,
                uint64_t* value_len) override {
    std::lock_guard<std::mutex> lk(mu);
    const auto it = store[node].find(K(key));
    if (it == store[node].end()) return Status::kNotFound;
    *value_len = it->second.size();
    return Status::kOk;
  }
  Status Exist(const std::string& node, const BlockKey& key,
               bool* exist) override {
    std::lock_guard<std::mutex> lk(mu);
    if (down.count(node)) return Status::kIOError;
    *exist = store[node].count(K(key)) != 0;
    return Status::kOk;
  }
  Status Remove(const std::string& node, const BlockKey& key) override {
    std::lock_guard<std::mutex> lk(mu);
    if (down.count(node)) return Status::kIOError;
    removed_here[node].insert(K(key));
    return store[node].erase(K(key)) ? Status::kOk : Status::kNotFound;
  }
  Status Retain(const std::string& node, uint64_t, uint64_t count,
                const std::string& payload, std::string* out) override {
    std::lock_guard<std::mutex> lk(mu);
    out->assign(count, static_cast<char>(KVClient::kRetMiss));
    for (uint64_t i = 0; i < count; ++i) {
      const char* k = payload.data() + kRetainHeaderBytes + i * kRetainKeyBytes;
      const Key key{net::GetU64(k), net::GetU64(k + 8)};
      retained[node].insert(key);
      if (store[node].count(key))
        (*out)[i] = static_cast<char>(KVClient::kRetProtectedNew);
    }
    out->resize(count + kRetainTrailerBytes);
    RetainTrailer trailer;
    trailer.max_ttl_s = 3600;
    trailer.headroom_pct = headroom_pct;
    EncodeRetainTrailer(&(*out)[count], trailer);
    return Status::kOk;
  }
  Status RemovedWithin(const std::string& node, uint64_t, uint64_t count,
                       const std::string& payload,
                       std::string* out) override {
    std::lock_guard<std::mutex> lk(mu);
    out->assign(4 + count, '\0');
    net::PutU32(&(*out)[0], 3600000);
    for (uint64_t i = 0; i < count; ++i) {
      const char* d = payload.data() + kRemovedWithinHeaderBytes + i * 16;
      const Key key{net::GetU64(d), net::GetU64(d + 8)};
      memory_queries[node].insert(key);
      (*out)[4 + i] = static_cast<char>(removed_here[node].count(key) ? 1 : 0);
    }
    return Status::kOk;
  }
  Status RemovedWithinAges(const std::string& node, uint64_t, uint64_t count,
                           const std::string& payload,
                           std::string* out) override {
    std::lock_guard<std::mutex> lk(mu);
    out->assign(4 + count * 8, '\0');
    net::PutU32(&(*out)[0], 3600000);
    for (uint64_t i = 0; i < count; ++i) {
      const char* d = payload.data() + kRemovedWithinHeaderBytes + i * 16;
      const Key key{net::GetU64(d), net::GetU64(d + 8)};
      age_queries[node].insert(key);
      const auto history = rewrite_history[node].find(key);
      const uint32_t removed = history == rewrite_history[node].end()
                                   ? kRemovedAgeNone
                                   : history->second.removed_age;
      const uint32_t written = history == rewrite_history[node].end()
                                   ? kRemovedAgeNone
                                   : history->second.written_age;
      net::PutU32(&(*out)[4 + i * 8], removed);
      net::PutU32(&(*out)[8 + i * 8], written);
    }
    return Status::kOk;
  }

  mutable std::mutex mu;
  std::map<std::string, std::map<Key, std::string>> store;
  std::map<std::string, std::set<Key>> retained, removed_here, memory_queries,
      age_queries;
  std::map<std::string, std::map<Key, RewriteHistory>> rewrite_history;
  std::set<std::string> down;
  uint32_t headroom_pct = 100;

 private:
  const bool pipelined_;
};

class ScopedEnv {
 public:
  ScopedEnv(const char* name, const std::string& value) : name_(name) {
    ::setenv(name, value.c_str(), 1);
  }
  ~ScopedEnv() { ::unsetenv(name_); }

 private:
  const char* name_;
};

constexpr char kNs[] = "sto17/test";
const std::vector<std::string> kIds = {"a", "b", "c"};

std::string Addr(const std::string& id) {
  return "127.0.0.1:" + std::to_string(1 + (id[0] - 'a'));
}
MemberInfo Member(const std::string& id, bool draining = false) {
  MemberInfo member{id, "127.0.0.1",
                    static_cast<uint32_t>(1 + (id[0] - 'a')), 1};
  member.draining = draining;
  return member;
}
std::vector<MemberInfo> View(const std::vector<std::string>& ids,
                             const std::set<std::string>& draining = {}) {
  std::vector<MemberInfo> out;
  for (const auto& id : ids) out.push_back(Member(id, draining.count(id) != 0));
  return out;
}
std::vector<std::pair<std::string, std::string>> Pairs(
    const std::vector<std::string>& ids) {
  std::vector<std::pair<std::string, std::string>> out;
  for (const auto& id : ids) out.push_back({id, Addr(id)});
  return out;
}
std::string Owner(const std::string& key, const std::vector<std::string>& ids) {
  ConHash ring;
  for (const auto& id : ids) ring.AddNode(id, 1);
  ring.Build();
  std::string name;
  EXPECT_TRUE(ring.LookupHash(ConHash::KeyHash(key), &name));
  return name;
}
std::string KeyOwnedBy(const std::string& id,
                       const std::vector<std::string>& ids,
                       const std::string& prefix) {
  for (int i = 0;; ++i) {
    const std::string key = prefix + std::to_string(i);
    if (Owner(key, ids) == id) return key;
  }
}
NodesTransport::Key K(const std::string& key) {
  return NodesTransport::K(ToBlockKey(kNs, key));
}
MemberInfo MarkedClient(bool readback) {
  MemberInfo client{"self", "0.0.0.0", 0, 0};
  client.info = "connector=x,dfkv-remove-fanout=1;poll_ms=1000";
  if (readback) client.info += ";readback=1";
  return client;
}
void OpenRegistry(KVClient* client, bool readback) {
  client->UseRegistryGateForTest(1000);
  client->OnRegistry(true, {MarkedClient(readback)});
}
dfkv_client_t CClient(NodesTransport* transport,
                      const std::vector<std::string>& ids = {"a", "b"}) {
  return new KVClient(Pairs(ids), kNs, transport, 1);
}

template <typename T, size_t N>
std::vector<T> Vec(const T (&values)[N]) {
  return {values, values + N};
}

struct Bytes {
  explicit Bytes(std::vector<std::string> input) : strings(std::move(input)) {
    for (const auto& value : strings) {
      pointers.push_back(value.data());
      lengths.push_back(value.size());
    }
  }
  std::vector<std::string> strings;
  std::vector<const void*> pointers;
  std::vector<uint64_t> lengths;
  int n() const { return static_cast<int>(strings.size()); }
};

class SegmentAffinity : public ::testing::TestWithParam<bool> {
 protected:
  SegmentAffinity() : transport(GetParam()) {}
  NodesTransport transport;
};

INSTANTIATE_TEST_SUITE_P(TcpAndPipelined, SegmentAffinity, ::testing::Bool());

TEST_P(SegmentAffinity, CApiSharesOwnerWithoutCollapsingAndCoversEveryForm) {
  dfkv_client_t client = CClient(&transport);
  const std::string anchor = KeyOwnedBy("a", {"a", "b"}, "anchor-");
  const std::string first = KeyOwnedBy("b", {"a", "b"}, "first-");
  const std::string second = KeyOwnedBy("b", {"a", "b"}, "second-");
  Bytes keys({first, second});
  Bytes anchors({anchor, anchor});

  // Default-off legacy placement still follows the actual key.
  Bytes legacy({KeyOwnedBy("b", {"a", "b"}, "legacy-")});
  const char legacy_value[8] = {'l', 'e', 'g', 'a', 'c', 'y', '0', '0'};
  const void* legacy_ptrs[] = {legacy_value};
  const uint64_t value_sizes[] = {8, 8};
  int legacy_ok[] = {0};
  ASSERT_EQ(dfkv_batch_put(client, legacy.pointers.data(), legacy.lengths.data(),
                           legacy_ptrs, value_sizes, 1, legacy_ok),
            0);
  ASSERT_EQ(legacy_ok[0], 1);
  EXPECT_EQ(transport.store[Addr("b")].count(K(legacy.strings[0])), 1u);
  EXPECT_EQ(transport.store[Addr("a")].count(K(legacy.strings[0])), 0u);

  const char left[8] = {'l', 'e', 'f', 't', '-', '0', '0', '0'};
  const char right[8] = {'r', 'i', 'g', 'h', 't', '-', '1', '1'};
  const void* values[] = {left, right};
  int ok[] = {0, 0};
  uint8_t status[] = {99, 99};
  ASSERT_EQ(dfkv_batch_put_anchored(
                client, keys.pointers.data(), keys.lengths.data(),
                anchors.pointers.data(), anchors.lengths.data(), values,
                value_sizes, keys.n(), ok),
            0);
  ASSERT_EQ(Vec(ok), (std::vector<int>{1, 1}));
  ASSERT_EQ(dfkv_batch_put_st_anchored(
                client, keys.pointers.data(), keys.lengths.data(),
                anchors.pointers.data(), anchors.lengths.data(), values,
                value_sizes, keys.n(), ok, status),
            0);
  EXPECT_EQ(Vec(status), (std::vector<uint8_t>{DFKV_ST_OK, DFKV_ST_OK}));
  ASSERT_EQ(transport.store[Addr("a")].count(K(first)), 1u);
  ASSERT_EQ(transport.store[Addr("a")].count(K(second)), 1u);
  EXPECT_NE(K(first), K(second));
  EXPECT_NE(transport.store[Addr("a")][K(first)],
            transport.store[Addr("a")][K(second)]);
  EXPECT_EQ(transport.store[Addr("b")].count(K(first)), 0u);

  std::vector<std::vector<char>> buffers(2, std::vector<char>(8));
  void* exact_ptrs[] = {buffers[0].data(), buffers[1].data()};
  std::vector<uint64_t> lengths(2, 0);
  ASSERT_EQ(dfkv_batch_get_anchored(
                client, keys.pointers.data(), keys.lengths.data(),
                anchors.pointers.data(), anchors.lengths.data(), exact_ptrs,
                value_sizes, keys.n(), ok),
            0);
  EXPECT_EQ(Vec(ok), (std::vector<int>{1, 1}));
  ASSERT_EQ(dfkv_batch_get_st_anchored(
                client, keys.pointers.data(), keys.lengths.data(),
                anchors.pointers.data(), anchors.lengths.data(), exact_ptrs,
                value_sizes, keys.n(), ok, status),
            0);
  EXPECT_EQ(Vec(status), (std::vector<uint8_t>{DFKV_ST_OK, DFKV_ST_OK}));
  EXPECT_EQ(std::string(buffers[0].data(), 8), std::string(left, 8));
  ASSERT_EQ(dfkv_batch_get_auto_anchored(
                client, keys.pointers.data(), keys.lengths.data(),
                anchors.pointers.data(), anchors.lengths.data(), exact_ptrs,
                value_sizes, keys.n(), ok, lengths.data()),
            0);
  EXPECT_EQ(lengths, (std::vector<uint64_t>{8, 8}));
  ASSERT_EQ(dfkv_batch_get_auto_st_anchored(
                client, keys.pointers.data(), keys.lengths.data(),
                anchors.pointers.data(), anchors.lengths.data(), exact_ptrs,
                value_sizes, keys.n(), ok, lengths.data(), status),
            0);
  EXPECT_EQ(Vec(status), (std::vector<uint8_t>{DFKV_ST_OK, DFKV_ST_OK}));
  EXPECT_EQ(std::string(buffers[1].data(), 8), std::string(right, 8));

  ASSERT_EQ(dfkv_batch_exist_anchored(
                client, keys.pointers.data(), keys.lengths.data(),
                anchors.pointers.data(), anchors.lengths.data(), keys.n(), ok),
            0);
  EXPECT_EQ(Vec(ok), (std::vector<int>{1, 1}));

  // Both SG PUT forms and both SG GET forms use the same two identities.
  const void* left_segments[] = {left};
  const void* right_segments[] = {right};
  const void** put_segments[] = {left_segments, right_segments};
  const uint64_t* segment_sizes[] = {value_sizes, value_sizes + 1};
  const int counts[] = {1, 1};
  ASSERT_EQ(dfkv_batch_put_sg_anchored(
                client, keys.pointers.data(), keys.lengths.data(),
                anchors.pointers.data(), anchors.lengths.data(), put_segments,
                segment_sizes, counts, keys.n(), ok),
            0);
  ASSERT_EQ(dfkv_batch_put_sg_st_anchored(
                client, keys.pointers.data(), keys.lengths.data(),
                anchors.pointers.data(), anchors.lengths.data(), put_segments,
                segment_sizes, counts, keys.n(), ok, status),
            0);
  EXPECT_EQ(Vec(status), (std::vector<uint8_t>{DFKV_ST_OK, DFKV_ST_OK}));
  void* left_dsts[] = {buffers[0].data()};
  void* right_dsts[] = {buffers[1].data()};
  void** get_segments[] = {left_dsts, right_dsts};
  ASSERT_EQ(dfkv_batch_get_auto_sg_anchored(
                client, keys.pointers.data(), keys.lengths.data(),
                anchors.pointers.data(), anchors.lengths.data(), get_segments,
                segment_sizes, counts, keys.n(), ok, lengths.data()),
            0);
  ASSERT_EQ(dfkv_batch_get_auto_sg_st_anchored(
                client, keys.pointers.data(), keys.lengths.data(),
                anchors.pointers.data(), anchors.lengths.data(), get_segments,
                segment_sizes, counts, keys.n(), ok, lengths.data(), status),
            0);
  EXPECT_EQ(Vec(status), (std::vector<uint8_t>{DFKV_ST_OK, DFKV_ST_OK}));

  transport.headroom_pct = 1;
  dfkv_retain_options options{};
  options.struct_size = sizeof(options);
  options.mode = DFKV_RETAIN_REFRESH;
  uint8_t retained[] = {DFKV_RET_ERROR, DFKV_RET_ERROR};
  ASSERT_EQ(dfkv_batch_retain_anchored(
                client, keys.pointers.data(), keys.lengths.data(),
                anchors.pointers.data(), anchors.lengths.data(), keys.n(),
                &options, retained),
            0);
  EXPECT_EQ(transport.retained[Addr("a")].count(K(first)), 1u);
  EXPECT_EQ(transport.retained[Addr("a")].count(K(second)), 1u);
  int precheck = -1;
  ASSERT_EQ(dfkv_retain_precheck_anchored(
                client, keys.pointers.data(), keys.lengths.data(),
                anchors.pointers.data(), anchors.lengths.data(), keys.n(), 5,
                &precheck),
            0);
  EXPECT_EQ(precheck, 0);  // the anchored owner's learned headroom, not key b's

  ASSERT_EQ(dfkv_batch_remove_anchored(
                client, keys.pointers.data(), keys.lengths.data(),
                anchors.pointers.data(), anchors.lengths.data(), keys.n(), ok),
            0);
  EXPECT_EQ(Vec(ok), (std::vector<int>{1, 1}));
  EXPECT_EQ(transport.store[Addr("a")].count(K(first)), 0u);
  EXPECT_EQ(transport.store[Addr("a")].count(K(second)), 0u);
  EXPECT_EQ(transport.removed_here[Addr("a")].count(K(first)), 1u);
  EXPECT_EQ(transport.removed_here[Addr("a")].count(K(second)), 1u);
  dfkv_close(client);
}

TEST_P(SegmentAffinity, NullEmptyAndMalformedAnchorsArePerItem) {
  dfkv_client_t client = CClient(&transport);
  const std::string anchor = KeyOwnedBy("a", {"a", "b"}, "validation-anchor-");
  const std::string actual = KeyOwnedBy("b", {"a", "b"}, "validation-actual-");
  Bytes keys({actual, actual, actual});
  Bytes anchors({anchor, anchor, ""});
  anchors.pointers[1] = nullptr;  // non-empty length is malformed only in slot 1
  const char value[8] = {};
  const void* values[] = {value, value, value};
  const uint64_t sizes[] = {8, 8, 8};
  int ok[] = {9, 9, 9};
  uint8_t status[] = {99, 99, 99};
  ASSERT_EQ(dfkv_batch_put_st_anchored(
                client, keys.pointers.data(), keys.lengths.data(),
                anchors.pointers.data(), anchors.lengths.data(), values, sizes,
                keys.n(), ok, status),
            0);
  EXPECT_EQ(Vec(ok), (std::vector<int>{1, 0, 1}));
  EXPECT_EQ(Vec(status), (std::vector<uint8_t>{DFKV_ST_OK, DFKV_ST_BAD_ARGUMENT,
                                          DFKV_ST_OK}));
  EXPECT_EQ(transport.store[Addr("a")].count(K(actual)), 1u);
  EXPECT_EQ(transport.store[Addr("b")].count(K(actual)), 1u);

  ok[0] = ok[2] = 9;
  ASSERT_EQ(dfkv_batch_put_st_anchored(
                client, keys.pointers.data(), keys.lengths.data(),
                anchors.pointers.data(), nullptr, values, sizes, keys.n(), ok,
                status),
            0);
  EXPECT_EQ(Vec(ok), (std::vector<int>{0, 0, 0}));
  EXPECT_EQ(Vec(status), (std::vector<uint8_t>{DFKV_ST_BAD_ARGUMENT,
                                          DFKV_ST_BAD_ARGUMENT,
                                          DFKV_ST_BAD_ARGUMENT}));

  // A wholly null route vector is the legacy path (route_key_lens is ignored).
  ASSERT_EQ(dfkv_batch_exist_anchored(client, keys.pointers.data(),
                                      keys.lengths.data(), nullptr, nullptr,
                                      keys.n(), ok),
            0);
  EXPECT_EQ(Vec(ok), (std::vector<int>{1, 1, 1}));
  dfkv_close(client);
}

TEST_P(SegmentAffinity, DrainAndAbsentUseAnchorPlacementAndActualRemoves) {
  const std::string anchor = KeyOwnedBy("b", {"a", "b"}, "drain-anchor-");
  const std::string actual = KeyOwnedBy("a", {"a", "b"}, "drain-actual-");
  const std::string value(8, 'd');
  KVClient client({}, kNs, &transport, 1);
  client.SetMembers(View({"a", "b"}));

  {
    ScopedEnv grace("DFKV_RING_ABSENT_GRACE_MS", "0");
    transport.store[Addr("b")][K(actual)] = value;
    client.SetMembers(View({"a", "b"}, {"b"}));
    std::string out(value.size(), '\0');
    std::vector<KvGetItem> items{{actual, out.data(), out.size(), anchor}};
    EXPECT_EQ(client.BatchGet(items), std::vector<bool>{true});
    EXPECT_EQ(out, value);

    const std::vector<std::string> keys{actual};
    const KvRouteKeys routes{anchor};
    transport.store[Addr("a")][K(actual)] = value;
    EXPECT_EQ(client.BatchRemove(keys, &routes), std::vector<bool>{true});
    EXPECT_EQ(transport.store[Addr("a")].count(K(actual)), 0u);
    EXPECT_EQ(transport.store[Addr("b")].count(K(actual)), 0u);
    EXPECT_EQ(transport.removed_here[Addr("b")].count(K(actual)), 1u);
    EXPECT_EQ(transport.removed_here[Addr("b")].count(K(anchor)), 0u);
  }

  {
    ScopedEnv grace("DFKV_RING_ABSENT_GRACE_MS", "60000");
    KVClient absent({}, kNs, &transport, 1);
    absent.SetMembers(View({"a", "b"}));
    transport.store[Addr("a")][K(actual)] = value;
    transport.store[Addr("b")][K(actual)] = value;
    absent.SetMembers(View({"a"}));
    const std::vector<std::string> keys{actual};
    const KvRouteKeys routes{anchor};
    EXPECT_EQ(absent.BatchRemove(keys, &routes), std::vector<bool>{true});
    EXPECT_EQ(transport.store[Addr("a")].count(K(actual)), 0u);
    EXPECT_EQ(transport.store[Addr("b")].count(K(actual)), 0u);
    EXPECT_EQ(transport.removed_here[Addr("b")].count(K(actual)), 1u);
  }
}

TEST_P(SegmentAffinity, HeldReadBackRoutesByAnchorAndChecksActualKeyM) {
  ScopedEnv readback("DFKV_RING_READBACK_MS", "60000");
  const std::vector<std::string> old_ids{"a", "b"};
  const std::vector<std::string> new_ids{"a", "b", "c"};
  std::string anchor;
  for (int i = 0;; ++i) {
    anchor = "readback-anchor-" + std::to_string(i);
    if (Owner(anchor, old_ids) == "b" && Owner(anchor, new_ids) == "c") break;
  }
  const std::string actual = KeyOwnedBy("a", old_ids, "readback-actual-");
  const std::string value(8, 'r');
  KVClient client({}, kNs, &transport, 1);
  client.SetMembers(View(old_ids));
  OpenRegistry(&client, true);
  const KvRouteKeys routes{anchor};
  const char* data = value.data();
  ASSERT_EQ(client.BatchPut({{actual, data, value.size(), anchor}}),
            std::vector<bool>{true});
  ASSERT_EQ(transport.store[Addr("b")].count(K(actual)), 1u);

  client.SetMembers(View(new_ids));
  std::string out(value.size(), '\0');
  std::vector<KvGetItem> items{{actual, out.data(), out.size(), anchor}};
  EXPECT_EQ(client.BatchGet(items), std::vector<bool>{true});
  EXPECT_EQ(out, value);
  EXPECT_EQ(transport.memory_queries[Addr("b")].count(K(actual)), 1u);

  // Once M reports the actual key removed, the same anchored read-back is
  // refused. An anchor-key M answer cannot authorize (or forbid) this object.
  transport.removed_here[Addr("b")].insert(K(actual));
  std::fill(out.begin(), out.end(), '\0');
  EXPECT_EQ(client.BatchGet(items), std::vector<bool>{false});
  EXPECT_EQ(transport.store[Addr("b")][K(actual)], value);
}

TEST_P(SegmentAffinity, HeldReadBackRewriteUsesActualKeyNotAnchorHistory) {
  ScopedEnv readback("DFKV_RING_READBACK_MS", "60000");
  ScopedEnv rewrite("DFKV_RING_READBACK_REWRITE", "1");
  const std::vector<std::string> old_ids{"a", "b"};
  const std::vector<std::string> new_ids{"a", "b", "c"};
  std::string anchor;
  for (int i = 0;; ++i) {
    anchor = "rewrite-anchor-" + std::to_string(i);
    if (Owner(anchor, old_ids) == "b" && Owner(anchor, new_ids) == "c") break;
  }
  const std::string actual = KeyOwnedBy("a", old_ids, "rewrite-actual-");
  const std::string value(8, 'w');
  KVClient client({}, kNs, &transport, 1);
  client.SetMembers(View(old_ids));
  OpenRegistry(&client, true);
  const char* data = value.data();
  ASSERT_EQ(client.BatchPut({{actual, data, value.size(), anchor}}),
            std::vector<bool>{true});
  client.SetMembers(View(new_ids));
  std::string out(value.size(), '\0');
  std::vector<KvGetItem> items{{actual, out.data(), out.size(), anchor}};

  // Ages are deterministic: a newer PUT has the smaller age. Actual-key PUT
  // after REMOVE passes W even though the anchor's opposite history would
  // refuse if it were incorrectly used for the rewrite decision.
  transport.rewrite_history[Addr("b")][K(actual)] = {100, 50};
  transport.rewrite_history[Addr("b")][K(anchor)] = {100, 200};
  EXPECT_EQ(client.BatchGet(items), std::vector<bool>{true});
  EXPECT_EQ(out, value);
  EXPECT_GT(transport.age_queries[Addr("b")].count(K(actual)), 0u);
  EXPECT_EQ(transport.age_queries[Addr("b")].count(K(anchor)), 0u);

  // Actual-key PUT before REMOVE must refuse even though the anchor's PUT is
  // now after its REMOVE and would pass if the histories were swapped.
  transport.rewrite_history[Addr("b")][K(actual)] = {100, 200};
  transport.rewrite_history[Addr("b")][K(anchor)] = {100, 50};
  std::fill(out.begin(), out.end(), '\0');
  EXPECT_EQ(client.BatchGet(items), std::vector<bool>{false});
  EXPECT_GT(transport.age_queries[Addr("b")].count(K(actual)), 0u);
  EXPECT_EQ(transport.age_queries[Addr("b")].count(K(anchor)), 0u);
  EXPECT_EQ(transport.store[Addr("b")][K(actual)], value);
}

TEST_P(SegmentAffinity, SubstituteChecksActualMandCleanupTracksActualKey) {
  ScopedEnv reroute("DFKV_RING_COOLDOWN_REROUTE", "1");
  ScopedEnv readback("DFKV_RING_READBACK_MS", "60000");
  ScopedEnv grace("DFKV_PEER_BAD_GRACE_MS", "1");
  ScopedEnv cooldown("DFKV_PEER_COOLDOWN_MS", "100");
  ScopedEnv cooldown_max("DFKV_PEER_COOLDOWN_MAX_MS", "100");
  ScopedEnv close_ms("DFKV_RING_REROUTE_CLOSE_MS", "0");
  const std::string anchor = KeyOwnedBy("b", kIds, "reroute-anchor-");
  const std::string actual = KeyOwnedBy("a", kIds, "reroute-actual-");
  const std::string value(8, 's');
  KVClient client({}, kNs, &transport, 1);
  client.SetMembers(View(kIds));
  OpenRegistry(&client, false);  // C clients do not defer ended-episode cleanup

  std::this_thread::sleep_for(std::chrono::milliseconds(3));
  transport.down.insert(Addr("b"));
  std::string cool_out(8, '\0');
  const std::string cool_key = KeyOwnedBy("b", kIds, "cool-");
  EXPECT_FALSE(client.Get(cool_key, cool_out.data(), cool_out.size()));
  transport.down.erase(Addr("b"));

  ConHash ring;
  for (const auto& id : kIds) ring.AddNode(id, 1);
  ring.Build();
  std::string substitute_id;
  bool after_owner = false;
  ring.Walk(ConHash::KeyHash(anchor), [&](const std::string& id) {
    if (after_owner && substitute_id.empty()) substitute_id = id;
    if (id == "b") after_owner = true;
    return substitute_id.empty();
  });
  ASSERT_FALSE(substitute_id.empty());
  const std::string substitute = Addr(substitute_id);
  transport.store[substitute][K(actual)] = value;

  // This client did not write the substitute copy. Substitute M therefore
  // refuses it after the actual key's REMOVE, not after an anchor-key REMOVE.
  transport.removed_here[substitute].insert(K(actual));
  std::string out(value.size(), '\0');
  std::vector<KvGetItem> get{{actual, out.data(), out.size(), anchor}};
  EXPECT_EQ(client.BatchGet(get), std::vector<bool>{false});
  EXPECT_EQ(transport.memory_queries[substitute].count(K(actual)), 1u);
  EXPECT_EQ(transport.memory_queries[substitute].count(K(anchor)), 0u);

  transport.removed_here.clear();
  const char* data = value.data();
  ASSERT_EQ(client.BatchPut({{actual, data, value.size(), anchor}}),
            std::vector<bool>{true});
  ASSERT_EQ(transport.store[substitute].count(K(actual)), 1u);
  std::this_thread::sleep_for(std::chrono::milliseconds(110));
  client.TickReroute();
  EXPECT_EQ(transport.store[substitute].count(K(actual)), 0u);
  EXPECT_EQ(transport.removed_here[substitute].count(K(actual)), 1u);
  EXPECT_EQ(transport.removed_here[substitute].count(K(anchor)), 0u);
}

}  // namespace
