#include "client/replay_trace.h"

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "client/key_map.h"
#include "client/kv_client.h"
#include "utils/log.h"

namespace dfkv {
namespace {

std::atomic<uint64_t> g_from_env_calls{0};
std::atomic<uint64_t> g_record_calls{0};
std::atomic<uint64_t> g_capture_instances{0};

int64_t WallNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}
int64_t MonotonicNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
void Number(std::string* out, uint64_t value) {
  out->append(std::to_string(value));
}
void SignedNumber(std::string* out, int64_t value) {
  out->append(std::to_string(value));
}
void NullableNumber(std::string* out, bool present, uint64_t value) {
  if (present)
    Number(out, value);
  else
    out->append("null");
}
void Bool(std::string* out, bool value) {
  out->append(value ? "true" : "false");
}
void NullableBool(std::string* out, bool present, bool value) {
  if (present)
    Bool(out, value);
  else
    out->append("null");
}
const char* StatusName(ItemStatus status) {
  switch (status) {
    case ItemStatus::kOk: return "DFKV_ST_OK";
    case ItemStatus::kNotFound: return "DFKV_ST_NOT_FOUND";
    case ItemStatus::kCacheFull: return "DFKV_ST_CACHE_FULL";
    case ItemStatus::kQuotaExceeded: return "DFKV_ST_QUOTA_EXCEEDED";
    case ItemStatus::kIOError: return "DFKV_ST_IO_ERROR";
    case ItemStatus::kInvalid: return "DFKV_ST_INVALID";
    case ItemStatus::kResourceExhausted: return "DFKV_ST_RESOURCE_EXHAUSTED";
    case ItemStatus::kNoCompatibleRail: return "DFKV_ST_NO_COMPATIBLE_RAIL";
    case ItemStatus::kLengthMismatch: return "DFKV_ST_LENGTH_MISMATCH";
    case ItemStatus::kPeerCooling: return "DFKV_ST_PEER_COOLING";
    case ItemStatus::kNoRoute: return "DFKV_ST_NO_ROUTE";
    case ItemStatus::kBadArgument: return "DFKV_ST_BAD_ARGUMENT";
  }
  return "UNKNOWN_ITEM_STATUS";
}
const char* RetainName(uint8_t code) {
  static constexpr const char* names[] = {
      "DFKV_RET_MISS", "DFKV_RET_HIT_UNPROTECTED",
      "DFKV_RET_PROTECTED_NEW", "DFKV_RET_REFRESHED", "DFKV_RET_UPGRADED",
      "DFKV_RET_UNCHANGED", "DFKV_RET_DENIED", "DFKV_RET_UNSUPPORTED",
      "DFKV_RET_SKIPPED_DEBOUNCE", "DFKV_RET_ERROR"};
  return code < sizeof(names) / sizeof(names[0]) ? names[code]
                                                : "UNKNOWN_RETAIN_RESULT";
}
void StatusJson(std::string* out, const std::vector<ItemStatus>* status,
                size_t i) {
  if (status == nullptr || i >= status->size()) {
    out->append("\"status\":null,\"status_code\":null");
    return;
  }
  const uint8_t code = static_cast<uint8_t>((*status)[i]);
  out->append("\"status\":\"");
  out->append(code <= static_cast<uint8_t>(ItemStatus::kBadArgument)
                  ? StatusName((*status)[i])
                  : "UNKNOWN_ITEM_STATUS");
  out->append("\",\"status_code\":");
  Number(out, code);
}
void ResultPrefix(std::string* out, size_t i, size_t expected,
                  const std::vector<bool>& result,
                  const std::vector<ItemStatus>* status) {
  out->append("{\"input_index\":");
  Number(out, i);
  const bool returned = i < result.size();
  out->append(",\"output_available\":");
  Bool(out, returned);
  out->append(",\"returned\":");
  NullableBool(out, returned, returned && result[i]);
  out->append(",");
  StatusJson(out, status, i);
  (void)expected;
}
void StoredLength(std::string* out, size_t i,
                  const std::vector<bool>& result,
                  const std::vector<size_t>* lengths) {
  const bool present = lengths != nullptr && i < result.size() && result[i] &&
                       i < lengths->size();
  out->append(",\"stored_length\":");
  NullableNumber(out, present, present ? (*lengths)[i] : 0);
  out->append(",\"stored_length_semantics\":");
  out->append(lengths == nullptr ? "\"not_returned\""
                                 : (present ? "\"actual_stored_length\""
                                            : "\"not_available\""));
}
void SegmentLengths(std::string* out, const std::vector<size_t>& lengths) {
  for (size_t j = 0; j < lengths.size(); ++j) {
    if (j) out->push_back(',');
    Number(out, lengths[j]);
  }
}
bool CheckedSum(const std::vector<size_t>& lengths, size_t* total) {
  *total = 0;
  for (size_t n : lengths) {
    if (n > std::numeric_limits<size_t>::max() - *total) return false;
    *total += n;
  }
  return true;
}
uint64_t ParseMaxBytes(const char* value) {
  constexpr uint64_t kDefault = 64ull << 20;
  if (value == nullptr || *value == '\0') return kDefault;
  if (*value == '-') return kDefault;
  char* end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0') {
    DFKV_LOG_WARN("DFKV_REPLAY_TRACE_MAX_BYTES is invalid; using 64 MiB");
    return kDefault;
  }
  return static_cast<uint64_t>(parsed);
}

}  // namespace

class ReplayTrace::Impl {
 public:
  Impl(std::string path_prefix, std::string key_namespace,
       bool initial_inventory_empty, uint64_t max_bytes)
      : base_path_(std::move(path_prefix)), block_key_(key_namespace),
        namespace_layout_generation_(block_key_("").Filename()),
        initial_inventory_empty_(initial_inventory_empty),
        max_bytes_(max_bytes),
        capture_start_epoch_ns_(WallNs()),
        capture_start_monotonic_ns_(MonotonicNs()),
        capture_id_(std::to_string(capture_start_epoch_ns_) + "-" +
                    std::to_string(static_cast<uint64_t>(::getpid())) + "-" +
                    std::to_string(g_capture_instances.fetch_add(1))) {
    if (base_path_.empty()) throw std::invalid_argument("empty trace path");
    OpenGenerationLocked(0);
    Append("start", std::string(",\"reason\":\"capture_start\"") +
                        ",\"initial_inventory\":{\"state\":\"" +
                        (initial_inventory_empty_ ? "empty" : "unknown") +
                        "\",\"object_count\":" +
                        (initial_inventory_empty_ ? "0" : "null") + "}");
  }
  ~Impl() { Close(); }

  std::vector<std::string> paths() const {
    std::lock_guard<std::mutex> lock(mu_);
    return paths_;
  }
  void FailNextWrite() {
    std::lock_guard<std::mutex> lock(mu_);
    fail_next_write_ = true;
  }

  void Close() noexcept {
    try {
      std::lock_guard<std::mutex> lock(mu_);
      if (closed_) return;
      closed_ = true;
      if (fd_ < 0) return;
      const Mark capture_end{WallNs(), MonotonicNs()};
      bool sync_failed = false;
      if (::fsync(fd_) != 0) {
        ++write_errors_;
        sync_failed = true;
      }
      FlushDropLocked();
      const uint64_t after_end = sequence_ + 1;
      std::string end_body = ",\"capture_end_epoch_ns\":";
      SignedNumber(&end_body, capture_end.wall_ns);
      end_body.append(",\"capture_end_monotonic_ns\":");
      SignedNumber(&end_body, capture_end.monotonic_ns);
      end_body.append(",\"initial_inventory_state_at_end\":\"" +
                        std::string(initial_inventory_empty_ ? "empty"
                                                             : "unknown") +
                        "\",\"write_error_count\":" +
                        std::to_string(write_errors_) +
                        ",\"final_fsync_error\":" +
                        (sync_failed ? "true" : "false") +
                        ",\"expected_record_count\":" +
                        std::to_string(after_end) +
                        ",\"dropped_record_count\":" +
                        std::to_string(dropped_records_ +
                                       pending_dropped_records_) +
                        ",\"durability\":\"fsync_before_end\"");
      WriteRecordLocked("end", end_body);
      ::close(fd_);
      fd_ = -1;
    } catch (...) {
      // A diagnostic stream cannot make client destruction fail.
    }
  }

  void Append(const std::string& type, const std::string& body) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mu_);
      if (closed_) return;
      AppendLocked(type, body);
    } catch (...) {
      // Allocation failures lose this observation, never the cache operation.
      try {
        std::lock_guard<std::mutex> lock(mu_);
        NoteDroppedLocked(sequence_++);
      } catch (...) {
      }
    }
  }

  void ObservationFailed() noexcept {
    try {
      {
        std::lock_guard<std::mutex> lock(mu_);
        ++dropped_records_;
      }
      // Argument conversion is inside this boundary too. If it throws, the
      // unrepresented increment makes end/drop accounting fail closed.
      Append("drop", ",\"dropped_sequence_begin\":null,"
                     "\"dropped_sequence_end\":null,\"dropped_count\":1,"
                     "\"expected_dropped_count\":1,"
                     "\"reason\":\"record_serialization_failed\"");
    } catch (...) {
      try {
        std::lock_guard<std::mutex> lock(mu_);
        ++write_errors_;
      } catch (...) {
      }
    }
  }

  void AppendBatch(const Mark& start, const Mark& end, const char* operation,
                   size_t count, const std::string& items,
                   const std::string& extra) {
    std::string body = ",\"operation\":\"";
    body.append(operation);
    body.append("\",\"namespace_layout_generation\":\"");
    body.append(namespace_layout_generation_);
    body.append("\",\"wall_start_ns\":");
    SignedNumber(&body, start.wall_ns);
    body.append(",\"wall_end_ns\":");
    SignedNumber(&body, end.wall_ns);
    body.append(",\"monotonic_start_ns\":");
    SignedNumber(&body, start.monotonic_ns);
    body.append(",\"monotonic_end_ns\":");
    SignedNumber(&body, end.monotonic_ns);
    body.append(",\"expected_item_count\":");
    Number(&body, count);
    body.append(",\"item_count\":");
    Number(&body, count);
    body.append(extra);
    body.append(",\"items\":");
    body.append(items);
    Append("batch_io", body);
  }

  std::string Identity(const std::string& key) const {
    return block_key_(key).Filename();
  }

 private:
  std::string PathFor(uint64_t generation) const {
    return base_path_ + ".native." + std::to_string(::getpid()) + "." +
           capture_id_ + ".g" + std::to_string(generation) + ".jsonl";
  }
  void OpenGenerationLocked(uint64_t generation) {
    const std::string path = PathFor(generation);
    int flags = O_WRONLY | O_CREAT | O_EXCL;
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
    const int fd = ::open(path.c_str(), flags, 0600);
    if (fd < 0)
      throw std::runtime_error("replay trace exclusive open failed");
    fd_ = fd;
    generation_ = generation;
    bytes_ = 0;
    paths_.push_back(path);
  }
  void WriteRawLocked(const std::string& line) {
    if (fail_next_write_) {
      fail_next_write_ = false;
      throw std::runtime_error("injected replay trace write failure");
    }
    const char* data = line.data();
    size_t left = line.size();
    while (left) {
      const ssize_t written = ::write(fd_, data, left);
      if (written <= 0) throw std::runtime_error("replay trace write failed");
      data += written;
      left -= static_cast<size_t>(written);
      bytes_ += static_cast<size_t>(written);
    }
  }
  void NoteDroppedLocked(uint64_t sequence) {
    if (pending_dropped_records_ == 0) first_dropped_sequence_ = sequence;
    last_dropped_sequence_ = sequence;
    ++pending_dropped_records_;
    ++write_errors_;
  }
  void WriteRecordLocked(const std::string& type, const std::string& body) {
    const uint64_t sequence = sequence_++;
    try {
      std::string line = "{\"schema_version\":1,\"record_type\":\"" + type +
                         "\",\"capture_id\":\"" + capture_id_ +
                         "\",\"capture_start_epoch_ns\":";
      SignedNumber(&line, capture_start_epoch_ns_);
      line.append(",\"capture_start_monotonic_ns\":");
      SignedNumber(&line, capture_start_monotonic_ns_);
      line.append(",\"sequence\":");
      Number(&line, sequence);
      line.append(",\"generation\":");
      Number(&line, generation_);
      line.append(body);
      line.append("}\n");
      WriteRawLocked(line);
    } catch (...) {
      // Attribute construction and I/O loss to the reserved sequence exactly once.
      NoteDroppedLocked(sequence);
    }
  }

  void FlushDropLocked() {
    if (pending_dropped_records_ == 0) return;
    const uint64_t first = first_dropped_sequence_;
    const uint64_t last = last_dropped_sequence_;
    const uint64_t count = pending_dropped_records_;
    const std::string body =
        ",\"dropped_sequence_begin\":" + std::to_string(first) +
        ",\"dropped_sequence_end\":" + std::to_string(last) +
        ",\"dropped_count\":" + std::to_string(count) +
        ",\"expected_dropped_count\":" + std::to_string(count);
    // Keep the pending range intact until every allocating conversion above
    // has succeeded. WriteRecordLocked attributes any subsequent loss to its
    // own newly reserved sequence, never to this range a second time.
    dropped_records_ += count;
    pending_dropped_records_ = 0;
    WriteRecordLocked("drop", body);
  }
  void RotateLocked() {
    const uint64_t next_generation = generation_ + 1;
    WriteRecordLocked("rotation", ",\"next_generation\":" +
                                      std::to_string(next_generation));
    ::close(fd_);
    fd_ = -1;
    try {
      OpenGenerationLocked(next_generation);
    } catch (...) {
      ++write_errors_;
      return;
    }
    WriteRecordLocked("start", ",\"reason\":\"rotation\",\"initial_inventory\":"
                                  "{\"state\":\"carried_from_same_capture\","
                                  "\"object_count\":null}");
  }
  void AppendLocked(const std::string& type, const std::string& body) {
    FlushDropLocked();
    if (fd_ < 0) {
      NoteDroppedLocked(sequence_++);
      return;
    }
    // Leave room for a rotation marker and the next generation's start.
    if (max_bytes_ != 0 && bytes_ != 0 && bytes_ + body.size() + 512 > max_bytes_)
      RotateLocked();
    if (fd_ < 0) {
      NoteDroppedLocked(sequence_++);
      return;
    }
    FlushDropLocked();
    WriteRecordLocked(type, body);
  }

  mutable std::mutex mu_;
  std::string base_path_;
  BlockKeyHasher block_key_;
  std::string namespace_layout_generation_;
  bool initial_inventory_empty_ = false;
  uint64_t max_bytes_ = 0;
  int64_t capture_start_epoch_ns_ = 0;
  int64_t capture_start_monotonic_ns_ = 0;
  std::string capture_id_;
  int fd_ = -1;
  bool closed_ = false;
  uint64_t generation_ = 0;
  uint64_t sequence_ = 0;
  uint64_t bytes_ = 0;
  uint64_t dropped_records_ = 0;
  uint64_t pending_dropped_records_ = 0;
  uint64_t first_dropped_sequence_ = 0;
  uint64_t last_dropped_sequence_ = 0;
  uint64_t write_errors_ = 0;
  bool fail_next_write_ = false;
  std::vector<std::string> paths_;
};

std::unique_ptr<ReplayTrace> ReplayTrace::FromEnv(
    const std::string& key_namespace) noexcept {
  g_from_env_calls.fetch_add(1, std::memory_order_relaxed);
  const char* path = std::getenv("DFKV_REPLAY_TRACE_PATH");
  if (path == nullptr || *path == '\0') return nullptr;
  try {
    return std::make_unique<ReplayTrace>(
        path, key_namespace, false,
        ParseMaxBytes(std::getenv("DFKV_REPLAY_TRACE_MAX_BYTES")));
  } catch (const std::exception& e) {
    try {
      DFKV_LOG_WARN(std::string("dfkv replay trace disabled: ") + e.what());
    } catch (...) {
    }
    return nullptr;
  }
}

ReplayTrace::ReplayTrace(std::string path_prefix,
                         const std::string& key_namespace,
                         bool initial_inventory_empty, uint64_t max_bytes)
    : impl_(std::make_unique<Impl>(std::move(path_prefix), key_namespace,
                                   initial_inventory_empty, max_bytes)) {}
ReplayTrace::~ReplayTrace() = default;
ReplayTrace::Mark ReplayTrace::Now() const noexcept {
  return Mark{WallNs(), MonotonicNs()};
}
std::vector<std::string> ReplayTrace::paths() const { return impl_->paths(); }
void ReplayTrace::fail_next_write_for_test() { impl_->FailNextWrite(); }
uint64_t ReplayTrace::from_env_calls_for_test() {
  return g_from_env_calls.load(std::memory_order_relaxed);
}
uint64_t ReplayTrace::record_calls_for_test() {
  return g_record_calls.load(std::memory_order_relaxed);
}

void ReplayTrace::RecordPut(const Mark& start,
                            const std::vector<KvPutItem>& items,
                            const std::vector<bool>& result,
                            const std::vector<ItemStatus>* status) noexcept {
  const Mark end = Now();  // Cache work has completed; exclude observation work.
  g_record_calls.fetch_add(1, std::memory_order_relaxed);
  try {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); ++i) {
      if (i) out.push_back(',');
      ResultPrefix(&out, i, items.size(), result, status);
      out.append(",\"block_key\":\"" + impl_->Identity(items[i].key) +
                 "\",\"requested_length\":");
      Number(&out, items[i].n);
      out.append(",\"capacity\":null,\"stored_length\":null,"
                 "\"stored_length_semantics\":\"not_returned\","
                 "\"payload\":{\"kind\":\"contiguous_put\"}}");
    }
    out.push_back(']');
    impl_->AppendBatch(start, end, "put", items.size(), out, "");
  } catch (...) {
    impl_->ObservationFailed();
  }
}
void ReplayTrace::RecordGet(const Mark& start, const char* operation,
                            const std::vector<KvGetItem>& items,
                            const std::vector<bool>& result,
                            const std::vector<size_t>* lengths,
                            const std::vector<ItemStatus>* status) noexcept {
  const Mark end = Now();  // Cache work has completed; exclude observation work.
  g_record_calls.fetch_add(1, std::memory_order_relaxed);
  try {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); ++i) {
      if (i) out.push_back(',');
      ResultPrefix(&out, i, items.size(), result, status);
      out.append(",\"block_key\":\"" + impl_->Identity(items[i].key) +
                 "\",\"requested_length\":null,\"capacity\":");
      Number(&out, items[i].n);
      StoredLength(&out, i, result, lengths);
      out.append(",\"payload\":{\"kind\":\"contiguous_get\"}}");
    }
    out.push_back(']');
    impl_->AppendBatch(start, end, operation, items.size(), out, "");
  } catch (...) {
    impl_->ObservationFailed();
  }
}
void ReplayTrace::RecordKeyResult(const Mark& start, const char* operation,
                                  const std::vector<std::string>& keys,
                                  const std::vector<bool>& result) noexcept {
  const Mark end = Now();  // Cache work has completed; exclude observation work.
  g_record_calls.fetch_add(1, std::memory_order_relaxed);
  try {
    std::string out = "[";
    for (size_t i = 0; i < keys.size(); ++i) {
      if (i) out.push_back(',');
      ResultPrefix(&out, i, keys.size(), result, nullptr);
      out.append(",\"block_key\":\"" + impl_->Identity(keys[i]) +
                 "\",\"requested_length\":null,\"capacity\":null,"
                 "\"stored_length\":null,\"stored_length_semantics\":"
                 "\"not_returned\",\"payload\":{\"kind\":\"key_only\"}}");
    }
    out.push_back(']');
    impl_->AppendBatch(start, end, operation, keys.size(), out, "");
  } catch (...) {
    impl_->ObservationFailed();
  }
}
void ReplayTrace::RecordRetain(const Mark& start,
                               const std::vector<std::string>& keys,
                               uint32_t ttl_s, uint8_t mode, bool no_debounce,
                               uint32_t min_refresh_ms,
                               const uint8_t* result) noexcept {
  const Mark end = Now();  // Cache work has completed; exclude observation work.
  g_record_calls.fetch_add(1, std::memory_order_relaxed);
  try {
    std::string out = "[";
    for (size_t i = 0; i < keys.size(); ++i) {
      if (i) out.push_back(',');
      out.append("{\"input_index\":");
      Number(&out, i);
      out.append(",\"output_available\":");
      Bool(&out, result != nullptr);
      out.append(",\"returned\":");
      const bool known_code = result != nullptr && result[i] <= KVClient::kRetError;
      const bool present = known_code &&
          ((result[i] >= KVClient::kRetHitUnprotected &&
            result[i] <= KVClient::kRetDenied) ||
           result[i] == KVClient::kRetSkippedDebounce);
      NullableBool(&out, known_code, present);
      out.append(",\"status\":null,\"status_code\":null,\"block_key\":\"" +
                 impl_->Identity(keys[i]) +
                 "\",\"requested_length\":null,\"capacity\":null,"
                 "\"stored_length\":null,\"stored_length_semantics\":"
                 "\"not_returned\",\"retain_result\":");
      if (result == nullptr) {
        out.append("null,\"retain_result_code\":null");
      } else {
        out.append("\"" + std::string(RetainName(result[i])) +
                   "\",\"retain_result_code\":" + std::to_string(result[i]));
      }
      out.append(",\"payload\":{\"kind\":\"key_only\"}}");
    }
    out.push_back(']');
    std::string extra = ",\"retention\":{\"ttl_s\":";
    Number(&extra, ttl_s);
    extra.append(",\"mode\":");
    extra.append(mode == 0 ? "\"set\"" : (mode == 1 ? "\"refresh\""
                                                    : "\"unknown\""));
    extra.append(",\"mode_code\":");
    Number(&extra, mode);
    extra.append(",\"no_debounce\":");
    Bool(&extra, no_debounce);
    extra.append(",\"min_refresh_ms\":");
    Number(&extra, min_refresh_ms);
    extra.push_back('}');
    impl_->AppendBatch(start, end, "retain", keys.size(), out, extra);
  } catch (...) {
    impl_->ObservationFailed();
  }
}
void ReplayTrace::RecordPutSg(const Mark& start,
                              const std::vector<KvPutItemSg>& items,
                              const std::vector<bool>& result,
                              const std::vector<ItemStatus>* status) noexcept {
  const Mark end = Now();  // Cache work has completed; exclude observation work.
  g_record_calls.fetch_add(1, std::memory_order_relaxed);
  try {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); ++i) {
      if (i) out.push_back(',');
      ResultPrefix(&out, i, items.size(), result, status);
      size_t total = 0;
      const bool valid = CheckedSum(items[i].sizes, &total);
      out.append(",\"block_key\":\"" + impl_->Identity(items[i].key) +
                 "\",\"requested_length\":");
      NullableNumber(&out, valid, total);
      out.append(",\"capacity\":null,\"stored_length\":null,"
                 "\"stored_length_semantics\":\"not_returned\",\"payload\":{"
                 "\"kind\":\"scatter_gather_put\",\"segment_lengths\":[");
      SegmentLengths(&out, items[i].sizes);
      out.append("],\"pointer_descriptor_count\":");
      Number(&out, items[i].ptrs.size());
      out.append(",\"length_descriptor_count\":");
      Number(&out, items[i].sizes.size());
      out.append(",\"sum_valid\":");
      Bool(&out, valid);
      out.append("}}");
    }
    out.push_back(']');
    impl_->AppendBatch(start, end, "put", items.size(), out,
                       ",\"payload_form\":\"scatter_gather\"");
  } catch (...) {
    impl_->ObservationFailed();
  }
}
void ReplayTrace::RecordGetAutoSg(
    const Mark& start, const std::vector<KvGetItemSg>& items,
    const std::vector<bool>& result, const std::vector<size_t>* lengths,
    const std::vector<ItemStatus>* status) noexcept {
  const Mark end = Now();  // Cache work has completed; exclude observation work.
  g_record_calls.fetch_add(1, std::memory_order_relaxed);
  try {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); ++i) {
      if (i) out.push_back(',');
      ResultPrefix(&out, i, items.size(), result, status);
      size_t total = 0;
      const bool valid = CheckedSum(items[i].caps, &total);
      out.append(",\"block_key\":\"" + impl_->Identity(items[i].key) +
                 "\",\"requested_length\":null,\"capacity\":");
      NullableNumber(&out, valid, total);
      StoredLength(&out, i, result, lengths);
      out.append(",\"payload\":{\"kind\":\"scatter_gather_get\","
                 "\"segment_capacities\":[");
      SegmentLengths(&out, items[i].caps);
      out.append("],\"pointer_descriptor_count\":");
      Number(&out, items[i].dsts.size());
      out.append(",\"capacity_descriptor_count\":");
      Number(&out, items[i].caps.size());
      out.append(",\"sum_valid\":");
      Bool(&out, valid);
      out.append("}}");
    }
    out.push_back(']');
    impl_->AppendBatch(start, end, "get_auto", items.size(), out,
                       ",\"payload_form\":\"scatter_gather\"");
  } catch (...) {
    impl_->ObservationFailed();
  }
}

}  // namespace dfkv
