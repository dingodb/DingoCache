#include "client/dfkv_c_api.h"
#include "client/key_map.h"
#include "client/kv_client.h"

#include <gtest/gtest.h>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr char kNamespace[] = "test/model";

class CApiTransport final : public dfkv::Transport {
 public:
  bool throwing = false;
  bool registration_ok = true;
  std::vector<dfkv::BlockKey> cached;

  dfkv::Status Cache(const std::string&, const dfkv::BlockKey& key,
                     const void*, size_t) override {
    Throw();
    cached.push_back(key);
    return dfkv::Status::kOk;
  }
  dfkv::Status Range(const std::string&, const dfkv::BlockKey&, uint64_t,
                     uint64_t, std::string*, uint64_t*) override {
    Throw();
    return dfkv::Status::kNotFound;
  }
  dfkv::Status Lookup(const std::string&, const dfkv::BlockKey&,
                      uint64_t*) override {
    Throw();
    return dfkv::Status::kNotFound;
  }
  dfkv::Status Exist(const std::string&, const dfkv::BlockKey&,
                     bool*) override {
    Throw();
    return dfkv::Status::kNotFound;
  }
  dfkv::Status Remove(const std::string&, const dfkv::BlockKey&) override {
    Throw();
    return dfkv::Status::kNotFound;
  }
  bool RegisterMemory(void*, size_t) override {
    Throw();
    return registration_ok;
  }
  size_t MaxSgPayloadSegs() const override {
    Throw();
    return 29;
  }
  std::string MetricsText() const override {
    Throw();
    return {};
  }

 private:
  void Throw() const {
    if (throwing) throw std::runtime_error("injected C ABI exception");
  }
};

dfkv_client_options_v2 Options(const void* key_namespace = kNamespace,
                               uint64_t key_namespace_len =
                                   sizeof(kNamespace) - 1) {
  dfkv_client_options_v2 options{};
  options.struct_size = sizeof(options);
  options.abi_version = DFKV_CLIENT_ABI_VERSION_V2;
  options.members = "";
  options.key_namespace = key_namespace;
  options.key_namespace_len = key_namespace_len;
  return options;
}

dfkv_client_t OpenEmpty() {
  auto options = Options();
  return dfkv_open_v2(&options);
}

dfkv_client_t Injected(CApiTransport* transport) {
  return new dfkv::KVClient({{"node", "test:1"}}, kNamespace, transport, 1);
}
}  // namespace

TEST(CApiGuard, RejectsInvalidV2OptionsAndMdsIdsSynchronously) {
  EXPECT_EQ(dfkv_open_v2(nullptr), nullptr);
  auto options = Options(nullptr, 0);
  EXPECT_EQ(dfkv_open_v2(&options), nullptr);

  options = Options();
  options.struct_size = sizeof(options) - 1;
  EXPECT_EQ(dfkv_open_v2(&options), nullptr);
  options = Options();
  options.abi_version = DFKV_CLIENT_ABI_VERSION_V2 + 1;
  EXPECT_EQ(dfkv_open_v2(&options), nullptr);
  options = Options();
  options.flags = 1u << 31;
  EXPECT_EQ(dfkv_open_v2(&options), nullptr);
  options = Options();
  options.reserved0 = 1;
  EXPECT_EQ(dfkv_open_v2(&options), nullptr);

  options = Options();
  options.mds_endpoints = "127.0.0.1:1";
  options.mds_group = "../escaped";
  EXPECT_EQ(dfkv_open_v2(&options), nullptr);

  options = Options();
  options.mds_endpoints = "127.0.0.1:1";
  options.flags = DFKV_CLIENT_OPT_REGISTER_WITH_MDS;
  options.client_id = "bad/client";
  EXPECT_EQ(dfkv_open_v2(&options), nullptr);
}

TEST(CApiGuard, AcceptsLengthDelimitedBinaryNamespaceAndObjectKeys) {
  const char binary_namespace[] = {'m', '\0', 'x'};
  auto options = Options(binary_namespace, sizeof(binary_namespace));
  dfkv_client_t empty = dfkv_open_v2(&options);
  ASSERT_NE(empty, nullptr);
  dfkv_close(empty);

  CApiTransport transport;
  dfkv_client_t client = Injected(&transport);
  const char binary_key[] = {'a', '\0', 'b'};
  ASSERT_EQ(dfkv_put(client, binary_key, sizeof(binary_key), "v", 1), 0);
  ASSERT_EQ(transport.cached.size(), 1u);
  const dfkv::BlockKey expected = dfkv::ToBlockKey(
      kNamespace, std::string(binary_key, sizeof(binary_key)));
  EXPECT_EQ(transport.cached[0].digest_hi, expected.digest_hi);
  EXPECT_EQ(transport.cached[0].digest_lo, expected.digest_lo);

  const void* keys[] = {binary_key, "a", binary_key};
  const uint64_t key_lens[] = {sizeof(binary_key), 1, 0};
  const void* ptrs[] = {"x", "y", "z"};
  const uint64_t sizes[] = {1, 1, 1};
  int out[] = {9, 9, 9};
  EXPECT_EQ(dfkv_batch_put(client, keys, key_lens, ptrs, sizes, 3, out), 0);
  EXPECT_EQ(out[0], 1);
  EXPECT_EQ(out[1], 1);
  EXPECT_EQ(out[2], 0);  // zero-length key is rejected, never truncated/hashed
  ASSERT_EQ(transport.cached.size(), 3u);
  EXPECT_NE(transport.cached[1].digest_hi, transport.cached[2].digest_hi);
  dfkv_close(client);
}

TEST(CApiGuard, NullAndZeroLengthInputsFailClosedWithSafeOutputs) {
  EXPECT_EQ(dfkv_put(nullptr, "k", 1, "v", 1), -1);
  EXPECT_EQ(dfkv_get(nullptr, "k", 1, nullptr, 0), 0);
  EXPECT_EQ(dfkv_exist(nullptr, "k", 1), 0);
  EXPECT_EQ(dfkv_remove(nullptr, "k", 1), 0);

  dfkv_client_t c = OpenEmpty();
  ASSERT_NE(c, nullptr);
  EXPECT_EQ(dfkv_put(c, nullptr, 1, "v", 1), -1);
  EXPECT_EQ(dfkv_put(c, "", 0, "v", 1), -1);
  EXPECT_EQ(dfkv_get(c, "k", 1, nullptr, 1), 0);
  uint64_t length = 9;
  EXPECT_EQ(dfkv_get_auto(c, "k", 1, nullptr, 1, &length), 0);
  EXPECT_EQ(length, 0u);

  int out[2] = {9, 9};
  EXPECT_EQ(dfkv_batch_put(c, nullptr, nullptr, nullptr, nullptr, 2, out), -1);
  EXPECT_EQ(out[0], 0);
  EXPECT_EQ(out[1], 0);
  out[0] = out[1] = 9;
  EXPECT_EQ(dfkv_batch_get(c, nullptr, nullptr, nullptr, nullptr, 2, out), -1);
  EXPECT_EQ(out[0], 0);
  EXPECT_EQ(out[1], 0);
  out[0] = out[1] = 9;
  EXPECT_EQ(dfkv_batch_exist(c, nullptr, nullptr, 2, out), -1);
  EXPECT_EQ(out[0], 0);
  EXPECT_EQ(out[1], 0);

  EXPECT_EQ(dfkv_batch_put(c, nullptr, nullptr, nullptr, nullptr, 0, nullptr), 0);
  EXPECT_EQ(dfkv_batch_get(c, nullptr, nullptr, nullptr, nullptr, 0, nullptr), 0);
  EXPECT_EQ(dfkv_batch_exist(c, nullptr, nullptr, 0, nullptr), 0);
  dfkv_close(c);
}

TEST(CApiGuard, ScatterGatherGuardsInitializeOutputs) {
  dfkv_client_t c = OpenEmpty();
  ASSERT_NE(c, nullptr);
  int hit[2] = {9, 9};
  uint64_t lengths[2] = {9, 9};
  EXPECT_EQ(dfkv_batch_put_sg(c, nullptr, nullptr, nullptr, nullptr, nullptr, 2,
                              hit),
            -1);
  EXPECT_EQ(hit[0], 0);
  EXPECT_EQ(hit[1], 0);
  hit[0] = hit[1] = 9;
  EXPECT_EQ(dfkv_batch_get_auto_sg(c, nullptr, nullptr, nullptr, nullptr,
                                   nullptr, 2, hit, lengths),
            -1);
  EXPECT_EQ(hit[0], 0);
  EXPECT_EQ(hit[1], 0);
  EXPECT_EQ(lengths[0], 0u);
  EXPECT_EQ(lengths[1], 0u);
  EXPECT_EQ(dfkv_batch_put_sg(c, nullptr, nullptr, nullptr, nullptr, nullptr, 0,
                              nullptr),
            0);
  EXPECT_EQ(dfkv_batch_get_auto_sg(c, nullptr, nullptr, nullptr, nullptr,
                                   nullptr, 0, nullptr, nullptr),
            0);
  dfkv_close(c);
}

TEST(CApiGuard, ScatterGatherSizeMaxOverflowFailsPerItem) {
  CApiTransport transport;
  dfkv_client_t c = Injected(&transport);
  char byte = 'v';
  const void* keys[] = {"overflow", "valid"};
  const uint64_t key_lens[] = {8, 5};
  const void* overflow_src[] = {&byte, &byte};
  const void* valid_src[] = {&byte};
  const void** srcs[] = {overflow_src, valid_src};
  const uint64_t overflow_sizes[] = {
      static_cast<uint64_t>(std::numeric_limits<size_t>::max()), 1};
  const uint64_t valid_sizes[] = {1};
  const uint64_t* sizes[] = {overflow_sizes, valid_sizes};
  const int counts[] = {2, 1};
  int ok[] = {9, 9};
  EXPECT_EQ(dfkv_batch_put_sg(c, keys, key_lens, srcs, sizes, counts, 2, ok),
            0);
  EXPECT_EQ(ok[0], 0);
  EXPECT_EQ(ok[1], 1);
  EXPECT_EQ(transport.cached.size(), 1u);

  void* overflow_dst[] = {&byte, &byte};
  void** dsts[] = {overflow_dst};
  const void* get_keys[] = {"overflow"};
  const uint64_t get_key_lens[] = {8};
  const uint64_t* caps[] = {overflow_sizes};
  const int dst_counts[] = {2};
  int hit[] = {9};
  uint64_t lengths[] = {99};
  EXPECT_EQ(dfkv_batch_get_auto_sg(c, get_keys, get_key_lens, dsts, caps,
                                   dst_counts, 1, hit, lengths),
            0);
  EXPECT_EQ(hit[0], 0);
  EXPECT_EQ(lengths[0], 0u);
  dfkv_close(c);
}

TEST(CApiRegistration, PropagatesNativeRegistrationFailure) {
  CApiTransport transport;
  dfkv_client_t c = Injected(&transport);
  char byte = 0;
  transport.registration_ok = false;
  EXPECT_EQ(dfkv_register_memory(c, &byte, 1), -1);
  transport.registration_ok = true;
  EXPECT_EQ(dfkv_register_memory(c, &byte, 1), 0);
  dfkv_close(c);
}

TEST(CApiNoThrow, EveryOperationFamilyTranslatesInjectedExceptions) {
  CApiTransport transport;
  dfkv_client_t c = Injected(&transport);
  transport.throwing = true;
  const char key[] = "k";

  EXPECT_EQ(dfkv_put(c, key, 1, "v", 1), -1);
  EXPECT_EQ(dfkv_get(c, key, 1, nullptr, 0), 0);
  uint64_t got = 9;
  EXPECT_EQ(dfkv_get_auto(c, key, 1, nullptr, 0, &got), 0);
  EXPECT_EQ(got, 0u);
  EXPECT_EQ(dfkv_exist(c, key, 1), 0);
  EXPECT_EQ(dfkv_remove(c, key, 1), 0);
  char byte = 0;
  EXPECT_EQ(dfkv_register_memory(c, &byte, 1), -1);
  EXPECT_EQ(dfkv_max_sg_segs(c), 0u);

  const void* keys[] = {key};
  const uint64_t key_lens[] = {1};
  const void* srcs[] = {&byte};
  void* dsts[] = {&byte};
  const uint64_t sizes[] = {1};
  int out[] = {9};
  EXPECT_EQ(dfkv_batch_put(c, keys, key_lens, srcs, sizes, 1, out), -1);
  EXPECT_EQ(out[0], 0);
  out[0] = 9;
  EXPECT_EQ(dfkv_batch_get(c, keys, key_lens, dsts, sizes, 1, out), -1);
  EXPECT_EQ(out[0], 0);
  uint64_t lengths[] = {9};
  out[0] = 9;
  EXPECT_EQ(dfkv_batch_get_auto(c, keys, key_lens, dsts, sizes, 1, out,
                                lengths),
            -1);
  EXPECT_EQ(out[0], 0);
  EXPECT_EQ(lengths[0], 0u);
  out[0] = 9;
  EXPECT_EQ(dfkv_batch_exist(c, keys, key_lens, 1, out), -1);
  EXPECT_EQ(out[0], 0);
  out[0] = 9;
  EXPECT_EQ(dfkv_batch_remove(c, keys, key_lens, 1, out), -1);
  EXPECT_EQ(out[0], 0);

  const void* sg_src[] = {&byte};
  const void** sg_srcs[] = {sg_src};
  const uint64_t sg_sizes[] = {1};
  const uint64_t* sg_sizes_array[] = {sg_sizes};
  const int counts[] = {1};
  out[0] = 9;
  EXPECT_EQ(dfkv_batch_put_sg(c, keys, key_lens, sg_srcs, sg_sizes_array,
                              counts, 1, out),
            -1);
  EXPECT_EQ(out[0], 0);
  void* sg_dst[] = {&byte};
  void** sg_dsts[] = {sg_dst};
  out[0] = 9;
  lengths[0] = 9;
  EXPECT_EQ(dfkv_batch_get_auto_sg(c, keys, key_lens, sg_dsts,
                                   sg_sizes_array, counts, 1, out, lengths),
            -1);
  EXPECT_EQ(out[0], 0);
  EXPECT_EQ(lengths[0], 0u);

  char stats[16] = {'X'};
  EXPECT_EQ(dfkv_stats_snapshot(c, stats, sizeof(stats)), 0u);
  EXPECT_EQ(stats[0], '\0');
  EXPECT_STREQ(dfkv_transport_mode(c), "injected");
  EXPECT_NE(dfkv_version(), nullptr);
  dfkv_close(c);
}

TEST(CApiStats, SnapshotSizingAndContent) {
  dfkv_client_t c = OpenEmpty();
  ASSERT_NE(c, nullptr);
  EXPECT_EQ(dfkv_get(c, "k", 1, nullptr, 0), 0);
  const uint64_t needed = dfkv_stats_snapshot(c, nullptr, 0);
  EXPECT_GT(needed, 0u);
  std::vector<char> buffer(needed + 1, 'X');
  EXPECT_EQ(dfkv_stats_snapshot(c, buffer.data(), buffer.size()), needed);
  EXPECT_EQ(buffer[needed], '\0');
  EXPECT_NE(std::string(buffer.data()).find("dfkv_client_op_requests_total"),
            std::string::npos);
  dfkv_close(c);
}

TEST(AutoBatchWorkers, ExplicitWinsAutoScalesCapped) {
  EXPECT_EQ(dfkv::AutoBatchWorkers(8, 47), 8u);
  EXPECT_EQ(dfkv::AutoBatchWorkers(0, 47), 32u);
  EXPECT_EQ(dfkv::AutoBatchWorkers(0, 0), 1u);
}
