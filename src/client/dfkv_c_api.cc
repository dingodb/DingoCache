#include "client/dfkv_c_api.h"

#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "client/kv_client.h"
#include "common/membership.h"
#include "common/version.h"
#include "utils/log.h"

using dfkv::KVClient;

namespace {
template <class Result, class Fn>
Result NoThrow(Result failure, Fn&& fn) noexcept {
  try {
    return fn();
  } catch (...) {
    try {
      DFKV_LOG_ERROR("dfkv C ABI operation failed with an exception");
    } catch (...) {
    }
    return failure;
  }
}

template <class Fn>
void NoThrowVoid(Fn&& fn) noexcept {
  try {
    fn();
  } catch (...) {
    try {
      DFKV_LOG_ERROR("dfkv C ABI operation failed with an exception");
    } catch (...) {
    }
  }
}

std::vector<std::pair<std::string, std::string>> ParseMembers(const char* s) {
  std::vector<std::pair<std::string, std::string>> out;
  std::string in(s ? s : "");
  size_t i = 0;
  while (i < in.size()) {
    size_t comma = in.find(',', i);
    std::string tok = in.substr(
        i, comma == std::string::npos ? std::string::npos : comma - i);
    size_t eq = tok.find('=');
    if (eq != std::string::npos)
      out.emplace_back(tok.substr(0, eq), tok.substr(eq + 1));
    if (comma == std::string::npos) break;
    i = comma + 1;
  }
  return out;
}

std::vector<std::string> ParseEndpoints(const char* endpoints) {
  std::vector<std::string> out;
  const std::string input(endpoints ? endpoints : "");
  for (size_t i = 0; i <= input.size();) {
    size_t comma = input.find(',', i);
    if (comma == std::string::npos) comma = input.size();
    if (comma > i) out.push_back(input.substr(i, comma - i));
    if (comma == input.size()) break;
    i = comma + 1;
  }
  return out;
}

bool HasText(const char* value) {
  return value != nullptr && value[0] != '\0';
}

bool ValidBuffer(const void* ptr, uint64_t size) {
  return size <= std::numeric_limits<size_t>::max() &&
         (size == 0 || ptr != nullptr);
}
bool AddBufferSize(uint64_t size, size_t* total) {
  if (size > std::numeric_limits<size_t>::max()) return false;
  size_t next = 0;
  if (!dfkv::CheckedAddSize(*total, static_cast<size_t>(size), &next))
    return false;
  *total = next;
  return true;
}

bool ValidKey(const void* ptr, uint64_t size) {
  return ptr != nullptr && size != 0 &&
         size <= std::numeric_limits<size_t>::max();
}

std::string CopyKey(const void* ptr, uint64_t size) {
  return std::string(static_cast<const char*>(ptr), static_cast<size_t>(size));
}

template <class T>
void ZeroArray(T* output, int n) {
  if (!output || n <= 0) return;
  for (int i = 0; i < n; ++i) output[i] = 0;
}
}  // namespace

extern "C" {

dfkv_client_t dfkv_open_v2(const dfkv_client_options_v2* options) {
  return NoThrow<dfkv_client_t>(nullptr, [&]() -> dfkv_client_t {
    constexpr uint32_t kKnownFlags = DFKV_CLIENT_OPT_REGISTER_WITH_MDS;
    if (options == nullptr ||
        options->struct_size < sizeof(dfkv_client_options_v2) ||
        options->abi_version != DFKV_CLIENT_ABI_VERSION_V2 ||
        (options->flags & ~kKnownFlags) != 0 || options->reserved0 != 0 ||
        options->key_namespace == nullptr || options->key_namespace_len == 0 ||
        options->key_namespace_len > std::numeric_limits<size_t>::max() ||
        options->batch_concurrency > std::numeric_limits<size_t>::max()) {
      DFKV_LOG_ERROR("dfkv_open_v2 failed: invalid options");
      return nullptr;
    }

    const bool register_with_mds =
        (options->flags & DFKV_CLIENT_OPT_REGISTER_WITH_MDS) != 0;
    const bool has_mds = HasText(options->mds_endpoints);
    if (register_with_mds && (!has_mds || !HasText(options->client_id))) {
      DFKV_LOG_ERROR(
          "dfkv_open_v2 failed: MDS registration needs endpoints and client_id");
      return nullptr;
    }

    std::vector<std::string> endpoints;
    std::string group;
    if (has_mds) {
      endpoints = ParseEndpoints(options->mds_endpoints);
      group = HasText(options->mds_group) ? options->mds_group : "default";
      if (endpoints.empty() || !dfkv::IsValidGroupOrId(group) ||
          (register_with_mds &&
           !dfkv::IsValidGroupOrId(std::string(options->client_id)))) {
        DFKV_LOG_ERROR(
            "dfkv_open_v2 failed: invalid MDS endpoint/group/client ID");
        return nullptr;
      }
    }

    auto members = ParseMembers(options->members);
    std::string key_namespace(
        static_cast<const char*>(options->key_namespace),
        static_cast<size_t>(options->key_namespace_len));
    auto client = std::make_unique<KVClient>(
        std::move(members), std::move(key_namespace), nullptr,
        static_cast<size_t>(options->batch_concurrency));

    if (has_mds) {
      client->StartMdsDiscovery(
          endpoints, group,
          options->mds_poll_ms > 0 ? options->mds_poll_ms : 3000);
      if (register_with_mds) {
        dfkv::MemberInfo self{options->client_id, "0.0.0.0", 0, 0};
        self.info = options->client_info ? std::string(options->client_info)
                                         : std::string();
        client->StartClientRegistration(
            endpoints, group, std::move(self),
            options->client_heartbeat_ms > 0 ? options->client_heartbeat_ms
                                             : 10000);
      }
    }
    return client.release();
  });
}

int dfkv_put(dfkv_client_t c, const void* key, uint64_t key_len,
             const void* ptr, uint64_t n) {
  return NoThrow(-1, [&] {
    if (!c || !ValidKey(key, key_len) || n == 0 ||
        !ValidBuffer(ptr, n))
      return -1;
    return static_cast<KVClient*>(c)->Put(
               CopyKey(key, key_len), ptr, static_cast<size_t>(n))
               ? 0
               : -1;
  });
}

int dfkv_get(dfkv_client_t c, const void* key, uint64_t key_len, void* ptr,
             uint64_t n) {
  return NoThrow(0, [&] {
    if (!c || !ValidKey(key, key_len) || !ValidBuffer(ptr, n)) return 0;
    return static_cast<KVClient*>(c)->Get(
               CopyKey(key, key_len), ptr, static_cast<size_t>(n))
               ? 1
               : 0;
  });
}

int dfkv_get_auto(dfkv_client_t c, const void* key, uint64_t key_len,
                  void* ptr, uint64_t cap, uint64_t* out_len) {
  if (out_len) *out_len = 0;
  return NoThrow(0, [&] {
    if (!c || !ValidKey(key, key_len) || !ValidBuffer(ptr, cap)) return 0;
    size_t got = 0;
    const bool hit = static_cast<KVClient*>(c)->GetAuto(
        CopyKey(key, key_len), ptr, static_cast<size_t>(cap), &got);
    if (hit && out_len) *out_len = static_cast<uint64_t>(got);
    return hit ? 1 : 0;
  });
}

int dfkv_exist(dfkv_client_t c, const void* key, uint64_t key_len) {
  return NoThrow(0, [&] {
    if (!c || !ValidKey(key, key_len)) return 0;
    return static_cast<KVClient*>(c)->Exist(CopyKey(key, key_len)) ? 1 : 0;
  });
}

int dfkv_remove(dfkv_client_t c, const void* key, uint64_t key_len) {
  return NoThrow(0, [&] {
    if (!c || !ValidKey(key, key_len)) return 0;
    return static_cast<KVClient*>(c)->Remove(CopyKey(key, key_len)) ? 1 : 0;
  });
}

int dfkv_register_memory(dfkv_client_t c, const void* base, uint64_t size) {
  return NoThrow(-1, [&] {
    if (!c || !base || size == 0 ||
        size > std::numeric_limits<size_t>::max())
      return -1;
    return static_cast<KVClient*>(c)->RegisterMemory(
               const_cast<void*>(base), static_cast<size_t>(size))
               ? 0
               : -1;
  });
}

uint32_t dfkv_max_sg_segs(dfkv_client_t c) {
  return NoThrow<uint32_t>(0, [&] {
    if (!c) return uint32_t{0};
    return static_cast<uint32_t>(
        static_cast<KVClient*>(c)->MaxSgPayloadSegs());
  });
}

int dfkv_batch_put(dfkv_client_t c, const void* const* keys,
                   const uint64_t* key_lens, const void** ptrs,
                   const uint64_t* sizes, int n, int* out_ok) {
  ZeroArray(out_ok, n);
  return NoThrow(-1, [&] {
    if (!c || n < 0) return -1;
    if (n > 0 && (!keys || !key_lens || !ptrs || !sizes || !out_ok)) return -1;
    std::vector<dfkv::KvPutItem> items;
    std::vector<size_t> positions;
    items.reserve(static_cast<size_t>(n));
    positions.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      if (!ValidKey(keys[i], key_lens[i]) || sizes[i] == 0 ||
          !ValidBuffer(ptrs[i], sizes[i]))
        continue;
      positions.push_back(static_cast<size_t>(i));
      items.push_back({CopyKey(keys[i], key_lens[i]), ptrs[i],
                       static_cast<size_t>(sizes[i])});
    }
    const auto result = static_cast<KVClient*>(c)->BatchPut(items);
    for (size_t i = 0; i < positions.size() && i < result.size(); ++i)
      out_ok[positions[i]] = result[i] ? 1 : 0;
    return 0;
  });
}

int dfkv_batch_get(dfkv_client_t c, const void* const* keys,
                   const uint64_t* key_lens, void** ptrs,
                   const uint64_t* sizes, int n, int* out_hit) {
  ZeroArray(out_hit, n);
  return NoThrow(-1, [&] {
    if (!c || n < 0) return -1;
    if (n > 0 && (!keys || !key_lens || !ptrs || !sizes || !out_hit)) return -1;
    std::vector<dfkv::KvGetItem> items;
    std::vector<size_t> positions;
    items.reserve(static_cast<size_t>(n));
    positions.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      if (!ValidKey(keys[i], key_lens[i]) ||
          !ValidBuffer(ptrs[i], sizes[i]))
        continue;
      positions.push_back(static_cast<size_t>(i));
      items.push_back({CopyKey(keys[i], key_lens[i]), ptrs[i],
                       static_cast<size_t>(sizes[i])});
    }
    const auto result = static_cast<KVClient*>(c)->BatchGet(items);
    for (size_t i = 0; i < positions.size() && i < result.size(); ++i)
      out_hit[positions[i]] = result[i] ? 1 : 0;
    return 0;
  });
}

int dfkv_batch_get_auto(dfkv_client_t c, const void* const* keys,
                        const uint64_t* key_lens, void** ptrs,
                        const uint64_t* caps, int n, int* out_hit,
                        uint64_t* out_len) {
  ZeroArray(out_hit, n);
  ZeroArray(out_len, n);
  return NoThrow(-1, [&] {
    if (!c || n < 0) return -1;
    if (n > 0 &&
        (!keys || !key_lens || !ptrs || !caps || !out_hit || !out_len))
      return -1;
    std::vector<dfkv::KvGetItem> items;
    std::vector<size_t> positions;
    items.reserve(static_cast<size_t>(n));
    positions.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      if (!ValidKey(keys[i], key_lens[i]) ||
          !ValidBuffer(ptrs[i], caps[i]))
        continue;
      positions.push_back(static_cast<size_t>(i));
      items.push_back({CopyKey(keys[i], key_lens[i]), ptrs[i],
                       static_cast<size_t>(caps[i])});
    }
    std::vector<size_t> lengths;
    const auto result = static_cast<KVClient*>(c)->BatchGetAuto(items, &lengths);
    for (size_t i = 0; i < positions.size() && i < result.size(); ++i) {
      out_hit[positions[i]] = result[i] ? 1 : 0;
      if (result[i] && i < lengths.size())
        out_len[positions[i]] = static_cast<uint64_t>(lengths[i]);
    }
    return 0;
  });
}

int dfkv_batch_exist(dfkv_client_t c, const void* const* keys,
                     const uint64_t* key_lens, int n, int* out_exist) {
  ZeroArray(out_exist, n);
  return NoThrow(-1, [&] {
    if (!c || n < 0) return -1;
    if (n > 0 && (!keys || !key_lens || !out_exist)) return -1;
    std::vector<std::string> valid_keys;
    std::vector<size_t> positions;
    valid_keys.reserve(static_cast<size_t>(n));
    positions.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      if (!ValidKey(keys[i], key_lens[i])) continue;
      positions.push_back(static_cast<size_t>(i));
      valid_keys.push_back(CopyKey(keys[i], key_lens[i]));
    }
    const auto result = static_cast<KVClient*>(c)->BatchExist(valid_keys);
    for (size_t i = 0; i < positions.size() && i < result.size(); ++i)
      out_exist[positions[i]] = result[i] ? 1 : 0;
    return 0;
  });
}

int dfkv_batch_remove(dfkv_client_t c, const void* const* keys,
                      const uint64_t* key_lens, int n, int* out_ok) {
  ZeroArray(out_ok, n);
  return NoThrow(-1, [&] {
    if (!c || n < 0) return -1;
    if (n > 0 && (!keys || !key_lens || !out_ok)) return -1;
    std::vector<std::string> valid_keys;
    std::vector<size_t> positions;
    valid_keys.reserve(static_cast<size_t>(n));
    positions.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      if (!ValidKey(keys[i], key_lens[i])) continue;
      positions.push_back(static_cast<size_t>(i));
      valid_keys.push_back(CopyKey(keys[i], key_lens[i]));
    }
    const auto result = static_cast<KVClient*>(c)->BatchRemove(valid_keys);
    for (size_t i = 0; i < positions.size() && i < result.size(); ++i)
      out_ok[positions[i]] = result[i] ? 1 : 0;
    return 0;
  });
}

int dfkv_batch_put_sg(dfkv_client_t c, const void* const* keys,
                      const uint64_t* key_lens, const void*** ptrs,
                      const uint64_t** sizes, const int* num_bufs, int n,
                      int* out_ok) {
  ZeroArray(out_ok, n);
  return NoThrow(-1, [&] {
    if (!c || n < 0) return -1;
    if (n > 0 &&
        (!keys || !key_lens || !ptrs || !sizes || !num_bufs || !out_ok))
      return -1;
    std::vector<dfkv::KvPutItemSg> items;
    std::vector<size_t> positions;
    items.reserve(static_cast<size_t>(n));
    positions.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      const int count = num_bufs[i];
      if (!ValidKey(keys[i], key_lens[i]) || count < 0 ||
          (count > 0 && (!ptrs[i] || !sizes[i])))
        continue;
      bool valid = true;
      size_t total = 0;
      for (int j = 0; j < count; ++j) {
        if (!ValidBuffer(ptrs[i][j], sizes[i][j]) ||
            !AddBufferSize(sizes[i][j], &total)) {
          valid = false;
          break;
        }
      }
      if (!valid || total == 0) continue;
      dfkv::KvPutItemSg item;
      item.key = CopyKey(keys[i], key_lens[i]);
      item.ptrs.reserve(static_cast<size_t>(count));
      item.sizes.reserve(static_cast<size_t>(count));
      for (int j = 0; j < count; ++j) {
        item.ptrs.push_back(ptrs[i][j]);
        item.sizes.push_back(static_cast<size_t>(sizes[i][j]));
      }
      positions.push_back(static_cast<size_t>(i));
      items.push_back(std::move(item));
    }
    const auto result = static_cast<KVClient*>(c)->BatchPutSg(items);
    for (size_t i = 0; i < positions.size() && i < result.size(); ++i)
      out_ok[positions[i]] = result[i] ? 1 : 0;
    return 0;
  });
}

int dfkv_batch_get_auto_sg(dfkv_client_t c, const void* const* keys,
                           const uint64_t* key_lens, void*** dsts,
                           const uint64_t** caps, const int* num_dsts, int n,
                           int* out_hit, uint64_t* out_len) {
  ZeroArray(out_hit, n);
  ZeroArray(out_len, n);
  return NoThrow(-1, [&] {
    if (!c || n < 0) return -1;
    if (n > 0 && (!keys || !key_lens || !dsts || !caps || !num_dsts ||
                  !out_hit || !out_len))
      return -1;
    std::vector<dfkv::KvGetItemSg> items;
    std::vector<size_t> positions;
    items.reserve(static_cast<size_t>(n));
    positions.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      const int count = num_dsts[i];
      if (!ValidKey(keys[i], key_lens[i]) || count < 0 ||
          (count > 0 && (!dsts[i] || !caps[i])))
        continue;
      bool valid = true;
      size_t total = 0;
      for (int j = 0; j < count; ++j) {
        if (!ValidBuffer(dsts[i][j], caps[i][j]) ||
            !AddBufferSize(caps[i][j], &total)) {
          valid = false;
          break;
        }
      }
      if (!valid) continue;
      dfkv::KvGetItemSg item;
      item.key = CopyKey(keys[i], key_lens[i]);
      item.dsts.reserve(static_cast<size_t>(count));
      item.caps.reserve(static_cast<size_t>(count));
      for (int j = 0; j < count; ++j) {
        item.dsts.push_back(dsts[i][j]);
        item.caps.push_back(static_cast<size_t>(caps[i][j]));
      }
      positions.push_back(static_cast<size_t>(i));
      items.push_back(std::move(item));
    }
    std::vector<size_t> lengths;
    const auto result =
        static_cast<KVClient*>(c)->BatchGetAutoSg(items, &lengths);
    for (size_t i = 0; i < positions.size() && i < result.size(); ++i) {
      out_hit[positions[i]] = result[i] ? 1 : 0;
      if (result[i] && i < lengths.size())
        out_len[positions[i]] = static_cast<uint64_t>(lengths[i]);
    }
    return 0;
  });
}

const char* dfkv_transport_mode(dfkv_client_t c) {
  return NoThrow<const char*>("", [&] {
    if (!c) return "";
    return static_cast<KVClient*>(c)->TransportMode().c_str();
  });
}

uint64_t dfkv_stats_snapshot(dfkv_client_t c, char* buf, uint64_t cap) {
  if (buf && cap > 0) buf[0] = '\0';
  return NoThrow<uint64_t>(0, [&] {
    if (!c) return uint64_t{0};
    const std::string text = static_cast<KVClient*>(c)->MetricsSnapshot();
    const uint64_t full = static_cast<uint64_t>(text.size());
    if (buf && cap > 0) {
      const uint64_t n = full < cap ? full : cap - 1;
      std::memcpy(buf, text.data(), static_cast<size_t>(n));
      buf[n] = '\0';
    }
    return full;
  });
}

const char* dfkv_version(void) {
  return NoThrow<const char*>("", [] { return dfkv::Version(); });
}

void dfkv_close(dfkv_client_t c) {
  NoThrowVoid([&] { delete static_cast<KVClient*>(c); });
}

}  // extern "C"
