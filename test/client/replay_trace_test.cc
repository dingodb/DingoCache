// Real optional replay recorder: native public batch calls, JSON parsing,
// disabled-path work counters, exclusive files, rotation and visible drops.
#include "client/replay_trace.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "client/key_map.h"
#include "client/kv_client.h"
#include "common/namespace.h"
#include "transport/wire.h"
#include "utils/json_min.h"

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {

class Env {
 public:
  Env(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) old_ = old;
    if (value)
      ::setenv(name, value, 1);
    else
      ::unsetenv(name);
  }
  ~Env() {
    if (old_)
      ::setenv(name_.c_str(), old_->c_str(), 1);
    else
      ::unsetenv(name_.c_str());
  }
 private:
  std::string name_;
  std::optional<std::string> old_;
};

struct TempDir {
  fs::path path;
  bool preserve = false;
  explicit TempDir(const std::string& tag) {
    if (const char* artifact_root = std::getenv("DFKV_REPLAY_TRACE_TEST_ARTIFACTS")) {
      path = fs::path(artifact_root) / tag;
      fs::create_directories(path);
      preserve = true;  // Optional real native input for the Python validator.
      return;
    }
    const std::string pattern =
        (fs::temp_directory_path() / ("dfkv_replay_" + tag + "_XXXXXX")).string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* made = ::mkdtemp(writable.data());
    EXPECT_NE(made, nullptr);
    path = made;
  }
  ~TempDir() {
    if (!preserve) fs::remove_all(path);
  }
};

std::vector<JsonValue> Read(const std::vector<std::string>& paths) {
  std::vector<JsonValue> records;
  for (const auto& path : paths) {
    std::ifstream input(path);
    EXPECT_TRUE(input.good()) << path;
    std::string line;
    while (input) {
      std::getline(input, line);
      if (!input && line.empty()) break;
      JsonValue value;
      EXPECT_TRUE(JsonParser::Parse(line, &value)) << line;
      records.push_back(std::move(value));
    }
  }
  return records;
}
std::vector<JsonValue> ReadDir(const fs::path& dir, const std::string& prefix) {
  std::vector<std::string> paths;
  for (const auto& entry : fs::directory_iterator(dir))
    if (entry.path().filename().string().find(prefix) == 0)
      paths.push_back(entry.path().string());
  std::sort(paths.begin(), paths.end());
  return Read(paths);
}
const JsonValue* At(const std::vector<JsonValue>& records, size_t index,
                    const char* type, const char* operation = nullptr) {
  size_t seen = 0;
  for (const auto& record : records) {
    if (record.get_str("record_type") != type) continue;
    if (operation && record.get_str("operation") != operation) continue;
    if (seen++ == index) return &record;
  }
  return nullptr;
}
uint64_t Number(const JsonValue& value, const std::string& field) {
  const JsonValue* found = value.find(field);
  EXPECT_NE(found, nullptr) << field;
  return found ? std::stoull(found->scalar) : 0;
}

class TraceTransport final : public Transport {
 public:
  bool pipelined() const override { return false; }
  Status Cache(const std::string&, const BlockKey& key, const void* data,
               size_t n) override {
    values[key.Filename()] = std::string(static_cast<const char*>(data), n);
    return Status::kOk;
  }
  Status Range(const std::string&, const BlockKey& key, uint64_t,
               uint64_t length, std::string* out, uint64_t* stored) override {
    const auto found = values.find(key.Filename());
    if (found == values.end()) return Status::kNotFound;
    if (stored) *stored = found->second.size();
    if (found->second.size() > length) return Status::kInvalid;
    *out = found->second;
    return Status::kOk;
  }
  Status Lookup(const std::string&, const BlockKey& key,
                uint64_t* size) override {
    const auto found = values.find(key.Filename());
    if (found == values.end()) return Status::kNotFound;
    if (size) *size = found->second.size();
    return Status::kOk;
  }
  Status Exist(const std::string&, const BlockKey& key, bool* exists) override {
    *exists = values.count(key.Filename()) != 0;
    return Status::kOk;
  }
  Status Remove(const std::string&, const BlockKey& key) override {
    values.erase(key.Filename());
    return Status::kOk;
  }
  Status Retain(const std::string&, uint64_t, uint64_t count,
                const std::string&, std::string* out) override {
    out->assign(count, static_cast<char>(KVClient::kRetRefreshed));
    out->resize(count + kRetainTrailerBytes);
    RetainTrailer trailer;
    trailer.max_ttl_s = 3600;
    trailer.headroom_pct = 100;
    EncodeRetainTrailer(&(*out)[count], trailer);
    return Status::kOk;
  }
  std::map<std::string, std::string> values;
};

struct Client {
  std::string ns = "test/replay\x01";
  TraceTransport transport;
  std::unique_ptr<KVClient> client;
  Client() {
    client = std::make_unique<KVClient>(
        std::vector<std::pair<std::string, std::string>>{
            {"n", "127.0.0.1:1"}}, ns, &transport);
  }
};

TEST(ReplayTrace, CanonicalBlockKeyLiteralGoldens) {
  EXPECT_EQ(ToBlockKey("test/model", "glm-5.2/page_777_k").Filename(),
            "fa1cde78082f951e1dabbca41aba2b02a865be51fa395d70");
  NamespaceDescriptor descriptor;
  descriptor.tenant_id = "test";
  descriptor.model_id = "structured/model";
  descriptor.model_revision = "r1";
  descriptor.pool_kind = PoolKind::kCustom;
  descriptor.pool_name = "layout/v1";
  descriptor.dtype = "fp16";
  descriptor.layout_fingerprint = 0x1122334455667788ULL;
  descriptor.block_tokens = 64;
  descriptor.group_id = 2;
  descriptor.layer_begin = 3;
  descriptor.layer_count = 4;
  descriptor.tp = RankCoordinate{4, -1};
  descriptor.dp = RankCoordinate{2, -1};
  descriptor.pp = RankCoordinate{1, -1};
  ASSERT_EQ(descriptor.Validate(), Status::kOk);
  EXPECT_EQ(ToBlockKey(descriptor.Serialize(), "structured/page_7").Filename(),
            "b0ba8aaceeb9ae2a8c6208fe6dbcd93931333b2ea5b21ae6");
}

TEST(ReplayTrace, PublicBatchesEmitAlignedStructuredEvidence) {
  TempDir tmp("batches");
  Env path("DFKV_REPLAY_TRACE_PATH", (tmp.path / "capture.jsonl").c_str());
  Env max_bytes("DFKV_REPLAY_TRACE_MAX_BYTES", "0");
  Env dedup("DFKV_CLIENT_NODE_DEDUP", "0");
  Client cl;
  char abc[] = {'a', 'b', 'c'};
  std::vector<ItemStatus> statuses;

  auto put = cl.client->BatchPut({{"alpha", abc, 3}, {"bad", nullptr, 3}},
                                  &statuses);
  ASSERT_EQ(put.size(), 2u);
  EXPECT_TRUE(put[0]);
  EXPECT_FALSE(put[1]);
  ASSERT_EQ(statuses.size(), 2u);
  EXPECT_EQ(statuses[0], ItemStatus::kOk);
  EXPECT_EQ(statuses[1], ItemStatus::kBadArgument);

  char out[8] = {};
  auto get = cl.client->BatchGet({{"alpha", out, 3}, {"missing", out + 3, 3}},
                                  &statuses);
  ASSERT_EQ(get.size(), 2u);
  EXPECT_TRUE(get[0]);
  EXPECT_FALSE(get[1]);
  EXPECT_EQ(statuses[1], ItemStatus::kNotFound);

  std::vector<size_t> lengths;
  auto auto_get = cl.client->BatchGetAuto(
      {{"alpha", out, 8}, {"alpha", out, 2}, {"missing", out, 8}},
      &lengths, &statuses);
  ASSERT_EQ(auto_get.size(), 3u);
  EXPECT_TRUE(auto_get[0]);
  EXPECT_FALSE(auto_get[1]);
  EXPECT_FALSE(auto_get[2]);
  EXPECT_EQ(lengths[0], 3u);
  EXPECT_EQ(lengths[1], 0u);
  EXPECT_EQ(statuses[1], ItemStatus::kLengthMismatch);

  // Enabled observation must allocate its own aligned result arrays when the
  // caller asks for none, without changing the public return values.
  EXPECT_EQ(cl.client->BatchPut({{"null_status", abc, 3}}),
            (std::vector<bool>{true}));
  EXPECT_EQ(cl.client->BatchGet({{"alpha", out, 3}}),
            (std::vector<bool>{true}));
  EXPECT_EQ(cl.client->BatchGetAuto({{"alpha", out, 8}}, nullptr, nullptr),
            (std::vector<bool>{true}));

  const std::vector<std::string> keys = {"alpha", "missing"};
  EXPECT_EQ(cl.client->BatchExist(keys), (std::vector<bool>{true, false}));
  KVClient::RetainRequest retain;
  retain.mode = 1;
  retain.ttl_s = 0;
  uint8_t retained[2] = {99, 99};
  cl.client->BatchRetain(keys, retain, retained);
  EXPECT_EQ(retained[0], KVClient::kRetRefreshed);
  EXPECT_EQ(retained[1], KVClient::kRetRefreshed);
  EXPECT_EQ(cl.client->BatchRemove(keys),
            (std::vector<bool>{true, true}));

  const std::vector<KvPutItemSg> sg_put = {
      {"sg", {abc, abc + 1}, {1, 2}},
      {"sg_bad", {}, {}}};
  auto sg_put_result = cl.client->BatchPutSg(sg_put, &statuses);
  ASSERT_EQ(sg_put_result.size(), 2u);
  EXPECT_TRUE(sg_put_result[0]);
  EXPECT_FALSE(sg_put_result[1]);
  char sg_out1 = 0;
  char sg_out2[2] = {};
  std::vector<KvGetItemSg> sg_get = {
      {"sg", {&sg_out1, sg_out2}, {1, 2}},
      {"missing_sg", {&sg_out1}, {1}}};
  auto sg_get_result = cl.client->BatchGetAutoSg(sg_get, &lengths, &statuses);
  ASSERT_EQ(sg_get_result.size(), 2u);
  EXPECT_TRUE(sg_get_result[0]);
  EXPECT_FALSE(sg_get_result[1]);
  EXPECT_EQ(lengths[0], 3u);
  char null_sg_value = 'z';
  EXPECT_EQ(cl.client->BatchPutSg({{"null_sg_status", {&null_sg_value}, {1}}}),
            (std::vector<bool>{true}));
  char null_sg_out = 0;
  EXPECT_EQ(cl.client->BatchGetAutoSg(
      {{"null_sg_status", {&null_sg_out}, {1}}}, nullptr, nullptr),
      (std::vector<bool>{true}));
  cl.client.reset();

  const auto records = ReadDir(tmp.path, "capture.jsonl.native.");
  ASSERT_EQ(records.size(), 15u);  // start + thirteen real calls + end
  EXPECT_EQ(records.front().get_str("record_type"), "start");
  EXPECT_EQ(records.front().find("initial_inventory")->get_str("state"),
            "unknown");
  EXPECT_EQ(records.back().get_str("record_type"), "end");
  std::map<std::string, int> operations;
  for (const auto& record : records) {
    EXPECT_EQ(record.get_str("schema_version"), "1");
    EXPECT_TRUE(record.get_str("capture_id") ==
                records.front().get_str("capture_id"));
    if (record.get_str("record_type") != "batch_io") continue;
    ++operations[record.get_str("operation")];
    EXPECT_LE(Number(record, "wall_start_ns"), Number(record, "wall_end_ns"));
    EXPECT_LE(Number(record, "monotonic_start_ns"),
              Number(record, "monotonic_end_ns"));
    EXPECT_EQ(Number(record, "expected_item_count"),
              Number(record, "item_count"));
    const auto& items = record.find("items")->arr;
    EXPECT_EQ(items.size(), Number(record, "item_count"));
    for (size_t i = 0; i < items.size(); ++i) {
      EXPECT_EQ(Number(items[i], "input_index"), i);
      const std::string identity = items[i].get_str("block_key");
      EXPECT_EQ(identity.size(), 48u);
      EXPECT_EQ(identity.find_first_not_of("0123456789abcdef"), std::string::npos);
    }
  }
  EXPECT_EQ(operations["put"], 3);  // contiguous, null-status contiguous, and SG
  EXPECT_EQ(operations["get"], 2);
  EXPECT_EQ(operations["get_auto"], 3);
  EXPECT_EQ(operations["exist"], 1);
  EXPECT_EQ(operations["retain"], 1);
  EXPECT_EQ(operations["remove"], 1);

  const JsonValue* put_record = At(records, 0, "batch_io", "put");
  const auto& put_items = put_record->find("items")->arr;
  EXPECT_EQ(put_items[0].get_str("block_key"),
            ToBlockKey(cl.ns, "alpha").Filename());
  EXPECT_EQ(Number(put_items[0], "requested_length"), 3u);
  EXPECT_EQ(put_items[0].get_str("status"), "DFKV_ST_OK");
  EXPECT_EQ(put_items[1].get_str("status"), "DFKV_ST_BAD_ARGUMENT");

  const JsonValue* get_record = At(records, 0, "batch_io", "get");
  for (const auto& item : get_record->find("items")->arr) {
    EXPECT_EQ(item.find("stored_length")->type, JsonValue::kNull);
    EXPECT_EQ(item.get_str("stored_length_semantics"), "not_returned");
    EXPECT_EQ(Number(item, "capacity"), 3u);
  }
  const JsonValue* auto_record = At(records, 0, "batch_io", "get_auto");
  const auto& auto_items = auto_record->find("items")->arr;
  EXPECT_EQ(Number(auto_items[0], "stored_length"), 3u);
  EXPECT_EQ(auto_items[0].get_str("stored_length_semantics"),
            "actual_stored_length");
  EXPECT_EQ(auto_items[1].find("stored_length")->type, JsonValue::kNull);
  EXPECT_EQ(auto_items[1].get_str("status"), "DFKV_ST_LENGTH_MISMATCH");
  EXPECT_EQ(auto_items[2].find("stored_length")->type, JsonValue::kNull);

  const JsonValue* exist_record = At(records, 0, "batch_io", "exist");
  const auto& exist_items = exist_record->find("items")->arr;
  EXPECT_EQ(exist_items[0].get_str("returned"), "true");
  EXPECT_EQ(exist_items[1].get_str("returned"), "false");
  EXPECT_EQ(exist_items[0].find("status")->type, JsonValue::kNull);
  const JsonValue* retain_record = At(records, 0, "batch_io", "retain");
  EXPECT_EQ(retain_record->find("retention")->get_str("mode"), "refresh");
  EXPECT_EQ(retain_record->find("items")->arr[0].get_str("retain_result"),
            "DFKV_RET_REFRESHED");

  const JsonValue* sg_put_record = At(records, 1, "batch_io", "put");
  EXPECT_EQ(sg_put_record->get_str("payload_form"), "scatter_gather");
  const auto& sg_put_items = sg_put_record->find("items")->arr;
  EXPECT_EQ(sg_put_items[0].find("payload")->get_str("kind"),
            "scatter_gather_put");
  EXPECT_EQ(Number(sg_put_items[0], "requested_length"), 3u);
  EXPECT_EQ(sg_put_items[1].get_str("status"), "DFKV_ST_BAD_ARGUMENT");
  const JsonValue* sg_get_record = At(records, 1, "batch_io", "get_auto");
  const auto& sg_get_items = sg_get_record->find("items")->arr;
  EXPECT_EQ(sg_get_items[0].find("payload")->get_str("kind"),
            "scatter_gather_get");
  EXPECT_EQ(Number(sg_get_items[0], "stored_length"), 3u);
  EXPECT_EQ(sg_get_items[0].find("payload")->find("segment_capacities")->arr.size(),
            2u);
  EXPECT_GT(Number(records.back(), "capture_end_epoch_ns"), 0u);
  EXPECT_GE(Number(records.back(), "capture_end_monotonic_ns"),
            Number(records.back(), "capture_start_monotonic_ns"));
  const JsonValue* null_put = At(records, 2, "batch_io", "put");
  const JsonValue* null_auto = At(records, 2, "batch_io", "get_auto");
  ASSERT_NE(null_put, nullptr);
  ASSERT_NE(null_auto, nullptr);
  EXPECT_EQ(null_put->find("items")->arr[0].get_str("status"), "DFKV_ST_OK");
  EXPECT_EQ(Number(null_auto->find("items")->arr[0], "stored_length"), 3u);

  // Raw namespace/object/payload spelling never enters the observation file.
  std::string all;
  for (const auto& entry : fs::directory_iterator(tmp.path)) {
    if (entry.path().filename().string().find("capture.jsonl.native.") != 0)
      continue;
    std::ifstream input(entry.path());
    all.append(std::istreambuf_iterator<char>(input),
               std::istreambuf_iterator<char>());
  }
  EXPECT_EQ(all.find("test/replay"), std::string::npos);
  EXPECT_EQ(all.find("alpha"), std::string::npos);
  EXPECT_EQ(all.find("route_key"), std::string::npos);
  EXPECT_EQ(all.find("0x"), std::string::npos);
  EXPECT_EQ(all.find("payload_bytes"), std::string::npos);
}

TEST(ReplayTrace, RotationAndExplicitInventoryAreCompleteChains) {
  TempDir tmp("rotation");
  std::vector<std::string> paths;
  {
    ReplayTrace trace((tmp.path / "rotate").string(), "test/rotation", true, 900);
    for (int i = 0; i < 8; ++i) {
      const std::vector<std::string> keys = {"k" + std::to_string(i)};
      trace.RecordKeyResult(trace.Now(), "exist", keys, {true});
    }
    trace.RecordKeyResult(trace.Now(), "remove", {"k0"}, {true});
    const uint8_t retained[] = {KVClient::kRetRefreshed};
    trace.RecordRetain(trace.Now(), {"k0"}, 0, 1, false, 0, retained);
    char source[] = {'a', 'b', 'c'};
    char destination[4] = {};
    const std::vector<ItemStatus> ok = {ItemStatus::kOk};
    const std::vector<ItemStatus> missing = {ItemStatus::kNotFound};
    const std::vector<size_t> stored_lengths = {3};
    trace.RecordPut(trace.Now(), {{"put", source, 3}}, {true}, &ok);
    trace.RecordGet(trace.Now(), "get", {{"absent", destination, 3}},
                    {false}, nullptr, &missing);
    trace.RecordGet(trace.Now(), "get_auto", {{"put", destination, 4}},
                    {true}, &stored_lengths, &ok);
    trace.RecordPutSg(trace.Now(),
                      {{"sg", {source, source + 1}, {1, 2}}}, {true}, &ok);
    trace.RecordGetAutoSg(trace.Now(),
                          {{"sg", {destination, destination + 1}, {1, 2}}},
                          {true}, &stored_lengths, &ok);
    paths = trace.paths();
  }
  ASSERT_GE(paths.size(), 2u);
  const auto records = Read(paths);
  size_t rotations = 0, starts = 0;
  for (size_t i = 0; i < records.size(); ++i) {
    EXPECT_EQ(Number(records[i], "sequence"), i);
    rotations += records[i].get_str("record_type") == "rotation";
    starts += records[i].get_str("record_type") == "start";
  }
  EXPECT_GE(rotations, 1u);
  EXPECT_EQ(starts, rotations + 1);
  EXPECT_EQ(records.front().find("initial_inventory")->get_str("state"), "empty");
  EXPECT_EQ(records.back().get_str("record_type"), "end");
  EXPECT_EQ(Number(records.back(), "expected_record_count"), records.size());
}

TEST(ReplayTrace, WriteFailureProducesDropAndErrorEndMarkers) {
  TempDir tmp("drop");
  std::vector<std::string> paths;
  {
    ReplayTrace trace((tmp.path / "drop").string(), "test/drop", true);
    trace.fail_next_write_for_test();
    trace.RecordKeyResult(trace.Now(), "exist", {"lost"}, {true});
    trace.RecordKeyResult(trace.Now(), "exist", {"kept"}, {false});
    paths = trace.paths();
  }
  const auto records = Read(paths);
  const JsonValue* drop = At(records, 0, "drop");
  ASSERT_NE(drop, nullptr);
  EXPECT_EQ(Number(*drop, "dropped_count"), 1u);
  EXPECT_EQ(Number(*drop, "expected_dropped_count"), 1u);
  EXPECT_EQ(Number(records.back(), "dropped_record_count"), 1u);
  EXPECT_GE(Number(records.back(), "write_error_count"), 1u);
  EXPECT_EQ(Number(records.back(), "expected_record_count"),
            records.size() + Number(records.back(), "dropped_record_count"));
  EXPECT_EQ(At(records, 0, "batch_io")->find("items")->arr[0].get_str("returned"),
            "false");
}

TEST(ReplayTrace, OverlappingCallsPublishInCompletionOrder) {
  TempDir tmp("overlap");
  std::vector<std::string> paths;
  {
    ReplayTrace trace((tmp.path / "overlap").string(), "test/overlap", true);
    const ReplayTrace::Mark early = trace.Now();
    std::atomic<bool> publish_early{false};
    std::thread late_finisher([&] {
      while (!publish_early.load(std::memory_order_acquire))
        std::this_thread::yield();
      trace.RecordKeyResult(early, "exist", {"early"}, {true});
    });
    const ReplayTrace::Mark later = trace.Now();
    trace.RecordKeyResult(later, "exist", {"later"}, {false});
    publish_early.store(true, std::memory_order_release);
    late_finisher.join();
    paths = trace.paths();
  }
  const auto records = Read(paths);
  const JsonValue* first = At(records, 0, "batch_io", "exist");
  const JsonValue* second = At(records, 1, "batch_io", "exist");
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(first->find("items")->arr[0].get_str("returned"), "false");
  EXPECT_EQ(second->find("items")->arr[0].get_str("returned"), "true");
  EXPECT_LT(Number(*second, "monotonic_start_ns"),
            Number(*first, "monotonic_start_ns"));
  EXPECT_LE(Number(*second, "monotonic_start_ns"),
            Number(*first, "monotonic_end_ns"));
  EXPECT_LE(Number(*first, "monotonic_end_ns"),
            Number(*second, "monotonic_end_ns"));
  EXPECT_EQ(Number(*first, "sequence"), 1u);
  EXPECT_EQ(Number(*second, "sequence"), 2u);
}

TEST(ReplayTrace, FailedRotationDoesNotChangeCallsOrClaimCompleteCapture) {
  TempDir tmp("rotation_failure");
  ReplayTrace trace((tmp.path / "broken").string(), "test/rotation-failure", true, 600);
  const std::string first_path = trace.paths().at(0);
  std::string next_path = first_path;
  const size_t generation = next_path.rfind(".g0.jsonl");
  ASSERT_NE(generation, std::string::npos);
  next_path.replace(generation, 9, ".g1.jsonl");
  {
    std::ofstream occupied(next_path);
    occupied << "already-owned-next-generation";
  }
  for (int i = 0; i < 8; ++i) {
    const std::vector<std::string> keys = {"key-" + std::to_string(i)};
    trace.RecordKeyResult(trace.Now(), "exist", keys, {i % 2 == 0});
  }
  std::ifstream input(next_path);
  std::string contents;
  std::getline(input, contents);
  EXPECT_EQ(contents, "already-owned-next-generation");
}

TEST(ReplayTrace, DisabledClientDoesNoObservationMetadataOrIO) {
  TempDir tmp("disabled");
  Env path("DFKV_REPLAY_TRACE_PATH", nullptr);
  Env dedup("DFKV_CLIENT_NODE_DEDUP", "0");
  const uint64_t config_before = ReplayTrace::from_env_calls_for_test();
  const uint64_t record_before = ReplayTrace::record_calls_for_test();
  {
    Client cl;
    EXPECT_EQ(ReplayTrace::from_env_calls_for_test(), config_before + 1);
    auto result = cl.client->BatchPut({{"disabled", nullptr, 1}});
    ASSERT_EQ(result.size(), 1u);
    EXPECT_FALSE(result[0]);
  }
  EXPECT_EQ(ReplayTrace::record_calls_for_test(), record_before);
  EXPECT_TRUE(fs::is_empty(tmp.path));
}

TEST(ReplayTrace, OpenFailureKeepsCacheResultsAndExclusiveBaseUntouched) {
  TempDir tmp("open_failure");
  const fs::path sentinel = tmp.path / "already-owned";
  {
    std::ofstream output(sentinel);
    output << "do-not-overwrite";
  }
  Env path("DFKV_REPLAY_TRACE_PATH", sentinel.c_str());
  Env dedup("DFKV_CLIENT_NODE_DEDUP", "0");
  Client cl;
  char value = 'x';
  EXPECT_TRUE(cl.client->BatchPut({{"alpha", &value, 1}})[0]);
  cl.client.reset();
  std::ifstream original(sentinel);
  std::string contents;
  std::getline(original, contents);
  EXPECT_EQ(contents, "do-not-overwrite");

  TempDir absent("abspath");
  Env missing_path("DFKV_REPLAY_TRACE_PATH",
                   (absent.path / "no-such-parent" / "trace").c_str());
  const uint64_t before = ReplayTrace::record_calls_for_test();
  Client failed_open;
  EXPECT_TRUE(failed_open.client->BatchPut({{"alpha", &value, 1}})[0]);
  EXPECT_EQ(ReplayTrace::record_calls_for_test(), before);
  EXPECT_FALSE(fs::exists(absent.path / "no-such-parent"));
}

}  // namespace
