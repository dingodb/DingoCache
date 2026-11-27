/* Optional ordered batch observation for replay/eviction analysis.
 * Disabled clients hold no ReplayTrace and do no per-item observation work.
 * Stream records contain stable BlockKey pseudonyms, never key/payload bytes.
 */
#ifndef DFKV_CLIENT_REPLAY_TRACE_H_
#define DFKV_CLIENT_REPLAY_TRACE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "client/item_status.h"
#include "common/kv_types.h"

namespace dfkv {

struct KvPutItem;
struct KvGetItem;
struct KvPutItemSg;
struct KvGetItemSg;

class ReplayTrace {
 public:
  struct Mark {
    int64_t wall_ns = 0;
    int64_t monotonic_ns = 0;
  };

  // Reads DFKV_REPLAY_TRACE_PATH and DFKV_REPLAY_TRACE_MAX_BYTES exactly once
  // for this client. Configuration/open failures disable observation and leave
  // cache behavior unchanged. The generated per-process file is O_EXCL.
  static std::unique_ptr<ReplayTrace> FromEnv(
      const std::string& key_namespace) noexcept;

  // Programmatic explicit capture. initial_inventory_empty is a caller assertion
  // that the cache was empty; ordinary construction must leave it false/unknown.
  // max_bytes=0 disables size rotation. Throws on configuration/open failure;
  // data-path Record methods below never throw into their cache caller.
  ReplayTrace(std::string path_prefix, const std::string& key_namespace,
              bool initial_inventory_empty, uint64_t max_bytes = 0);
  ~ReplayTrace();
  ReplayTrace(const ReplayTrace&) = delete;
  ReplayTrace& operator=(const ReplayTrace&) = delete;

  Mark Now() const noexcept;
  std::vector<std::string> paths() const;
  void fail_next_write_for_test();

  void RecordPut(const Mark& start, const std::vector<KvPutItem>& items,
                 const std::vector<bool>& result,
                 const std::vector<ItemStatus>* status) noexcept;
  void RecordGet(const Mark& start, const char* operation,
                 const std::vector<KvGetItem>& items,
                 const std::vector<bool>& result,
                 const std::vector<size_t>* stored_lengths,
                 const std::vector<ItemStatus>* status) noexcept;
  void RecordKeyResult(const Mark& start, const char* operation,
                       const std::vector<std::string>& keys,
                       const std::vector<bool>& result) noexcept;
  void RecordRetain(const Mark& start, const std::vector<std::string>& keys,
                    uint32_t ttl_s, uint8_t mode, bool no_debounce,
                    uint32_t min_refresh_ms, const uint8_t* result) noexcept;
  void RecordPutSg(const Mark& start, const std::vector<KvPutItemSg>& items,
                   const std::vector<bool>& result,
                   const std::vector<ItemStatus>* status) noexcept;
  void RecordGetAutoSg(const Mark& start,
                       const std::vector<KvGetItemSg>& items,
                       const std::vector<bool>& result,
                       const std::vector<size_t>* stored_lengths,
                       const std::vector<ItemStatus>* status) noexcept;

  // Instrumentation for the disabled-path unit test, not read by production.
  static uint64_t from_env_calls_for_test();
  static uint64_t record_calls_for_test();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace dfkv

#endif  // DFKV_CLIENT_REPLAY_TRACE_H_
