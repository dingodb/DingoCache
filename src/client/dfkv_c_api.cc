#include "client/dfkv_c_api.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "client/item_status.h"
#include "client/kv_client.h"
#include "common/membership.h"
#include "common/version.h"
#include "utils/env_parse.h"
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

// A null route vector preserves the legacy placement. An empty per-item anchor
// falls back to that item's actual key; any other malformed anchor is rejected
// independently, exactly like a malformed actual key.
bool ValidRouteKey(const void* const* route_keys,
                   const uint64_t* route_key_lens, int i) {
  if (route_keys == nullptr) return true;
  return route_key_lens != nullptr &&
         (route_key_lens[i] == 0 || ValidKey(route_keys[i], route_key_lens[i]));
}

std::string_view RouteKeyView(const void* const* route_keys,
                              const uint64_t* route_key_lens, int i) {
  if (route_keys == nullptr || route_key_lens[i] == 0) return {};
  return {static_cast<const char*>(route_keys[i]),
          static_cast<size_t>(route_key_lens[i])};
}

template <class T>
void ZeroArray(T* output, int n) {
  if (!output || n <= 0) return;
  for (int i = 0; i < n; ++i) output[i] = 0;
}

static_assert(DFKV_ST_OK == static_cast<int>(dfkv::ItemStatus::kOk) &&
                  DFKV_ST_NOT_FOUND ==
                      static_cast<int>(dfkv::ItemStatus::kNotFound) &&
                  DFKV_ST_CACHE_FULL ==
                      static_cast<int>(dfkv::ItemStatus::kCacheFull) &&
                  DFKV_ST_QUOTA_EXCEEDED ==
                      static_cast<int>(dfkv::ItemStatus::kQuotaExceeded) &&
                  DFKV_ST_IO_ERROR ==
                      static_cast<int>(dfkv::ItemStatus::kIOError) &&
                  DFKV_ST_INVALID ==
                      static_cast<int>(dfkv::ItemStatus::kInvalid) &&
                  DFKV_ST_RESOURCE_EXHAUSTED ==
                      static_cast<int>(dfkv::ItemStatus::kResourceExhausted) &&
                  DFKV_ST_NO_COMPATIBLE_RAIL ==
                      static_cast<int>(dfkv::ItemStatus::kNoCompatibleRail) &&
                  DFKV_ST_LENGTH_MISMATCH ==
                      static_cast<int>(dfkv::ItemStatus::kLengthMismatch) &&
                  DFKV_ST_PEER_COOLING ==
                      static_cast<int>(dfkv::ItemStatus::kPeerCooling) &&
                  DFKV_ST_NO_ROUTE ==
                      static_cast<int>(dfkv::ItemStatus::kNoRoute) &&
                  DFKV_ST_BAD_ARGUMENT ==
                      static_cast<int>(dfkv::ItemStatus::kBadArgument),
              "the C codes are the ItemStatus values");

// Slots the C layer rejects before the client sees them keep this code.
void FillBadArgument(uint8_t* out_status, int n) {
  if (!out_status || n <= 0) return;
  for (int i = 0; i < n; ++i)
    out_status[i] = static_cast<uint8_t>(dfkv::ItemStatus::kBadArgument);
}

void CopyStatus(const std::vector<dfkv::ItemStatus>& status,
                const std::vector<size_t>& positions, uint8_t* out_status) {
  if (!out_status) return;
  for (size_t i = 0; i < positions.size() && i < status.size(); ++i)
    out_status[positions[i]] = static_cast<uint8_t>(status[i]);
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
      // Synchronously wait for a non-empty ring before returning. Without this,
      // async MDS discovery races the first PUT/GET: if the ring is still empty
      // (first poll not yet complete), every op silently fails (Route() returns ""
      // → ok=0). The write is lost and never retried — L3 cache appears dead.
      // Env DFKV_OPEN_RING_WAIT_MS overrides the timeout (default 30s, 0=skip).
      uint64_t ring_wait_ms = 30000;
      if (const char* e = std::getenv("DFKV_OPEN_RING_WAIT_MS")) {
        char* end = nullptr;
        const unsigned long long v = std::strtoull(e, &end, 10);
        if (end != e) ring_wait_ms = static_cast<uint64_t>(v);
      }
      // What is waited for is the int below, which a larger number wraps.
      const int ring_wait_arg = static_cast<int>(ring_wait_ms);
      dfkv::envparse::ReportUnlessPlain("DFKV_OPEN_RING_WAIT_MS", 0,
                                        std::numeric_limits<int>::max(),
                                        std::to_string(ring_wait_arg));
      if (ring_wait_ms > 0 && !client->WaitForRing(ring_wait_arg)) {
        DFKV_LOG_WARN("dfkv_open_v2: ring still empty after "
                      + std::to_string(ring_wait_ms)
                      + "ms; ops will silently fail until MDS populates the ring");
      }
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

uint64_t dfkv_max_block_bytes(dfkv_client_t c) {
  return NoThrow<uint64_t>(0, [&] {
    if (!c) return uint64_t{0};
    return static_cast<KVClient*>(c)->MaxBlockBytes();
  });
}

static int BatchPutC(dfkv_client_t c, const void* const* keys,
                     const uint64_t* key_lens, const void* const* route_keys,
                     const uint64_t* route_key_lens, const void** ptrs,
                     const uint64_t* sizes, int n, int* out_ok,
                     uint8_t* out_status) {
  ZeroArray(out_ok, n);
  FillBadArgument(out_status, n);
  return NoThrow(-1, [&] {
    if (!c || n < 0) return -1;
    if (n > 0 && (!keys || !key_lens || !ptrs || !sizes || !out_ok)) return -1;
    std::vector<dfkv::KvPutItem> items;
    std::vector<size_t> positions;
    items.reserve(static_cast<size_t>(n));
    positions.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      if (!ValidKey(keys[i], key_lens[i]) ||
          !ValidRouteKey(route_keys, route_key_lens, i) || sizes[i] == 0 ||
          !ValidBuffer(ptrs[i], sizes[i]))
        continue;
      positions.push_back(static_cast<size_t>(i));
      items.push_back({CopyKey(keys[i], key_lens[i]), ptrs[i],
                       static_cast<size_t>(sizes[i]),
                       RouteKeyView(route_keys, route_key_lens, i)});
    }
    std::vector<dfkv::ItemStatus> status;
    const auto result = static_cast<KVClient*>(c)->BatchPut(
        items, out_status ? &status : nullptr);
    CopyStatus(status, positions, out_status);
    for (size_t i = 0; i < positions.size() && i < result.size(); ++i)
      out_ok[positions[i]] = result[i] ? 1 : 0;
    return 0;
  });
}

static int BatchGetC(dfkv_client_t c, const void* const* keys,
                     const uint64_t* key_lens, const void* const* route_keys,
                     const uint64_t* route_key_lens, void** ptrs,
                     const uint64_t* sizes, int n, int* out_hit,
                     uint8_t* out_status) {
  ZeroArray(out_hit, n);
  FillBadArgument(out_status, n);
  return NoThrow(-1, [&] {
    if (!c || n < 0) return -1;
    if (n > 0 && (!keys || !key_lens || !ptrs || !sizes || !out_hit)) return -1;
    std::vector<dfkv::KvGetItem> items;
    std::vector<size_t> positions;
    items.reserve(static_cast<size_t>(n));
    positions.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      if (!ValidKey(keys[i], key_lens[i]) ||
          !ValidRouteKey(route_keys, route_key_lens, i) ||
          !ValidBuffer(ptrs[i], sizes[i]))
        continue;
      positions.push_back(static_cast<size_t>(i));
      items.push_back({CopyKey(keys[i], key_lens[i]), ptrs[i],
                       static_cast<size_t>(sizes[i]),
                       RouteKeyView(route_keys, route_key_lens, i)});
    }
    std::vector<dfkv::ItemStatus> status;
    const auto result = static_cast<KVClient*>(c)->BatchGet(
        items, out_status ? &status : nullptr);
    CopyStatus(status, positions, out_status);
    for (size_t i = 0; i < positions.size() && i < result.size(); ++i)
      out_hit[positions[i]] = result[i] ? 1 : 0;
    return 0;
  });
}

static int BatchGetAutoC(dfkv_client_t c, const void* const* keys,
                         const uint64_t* key_lens, const void* const* route_keys,
                         const uint64_t* route_key_lens, void** ptrs,
                         const uint64_t* caps, int n, int* out_hit,
                         uint64_t* out_len, uint8_t* out_status) {
  ZeroArray(out_hit, n);
  FillBadArgument(out_status, n);
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
          !ValidRouteKey(route_keys, route_key_lens, i) ||
          !ValidBuffer(ptrs[i], caps[i]))
        continue;
      positions.push_back(static_cast<size_t>(i));
      items.push_back({CopyKey(keys[i], key_lens[i]), ptrs[i],
                       static_cast<size_t>(caps[i]),
                       RouteKeyView(route_keys, route_key_lens, i)});
    }
    std::vector<size_t> lengths;
    std::vector<dfkv::ItemStatus> status;
    const auto result = static_cast<KVClient*>(c)->BatchGetAuto(
        items, &lengths, out_status ? &status : nullptr);
    CopyStatus(status, positions, out_status);
    for (size_t i = 0; i < positions.size() && i < result.size(); ++i) {
      out_hit[positions[i]] = result[i] ? 1 : 0;
      if (result[i] && i < lengths.size())
        out_len[positions[i]] = static_cast<uint64_t>(lengths[i]);
    }
    return 0;
  });
}

static int BatchExistC(dfkv_client_t c, const void* const* keys,
                       const uint64_t* key_lens, const void* const* route_keys,
                       const uint64_t* route_key_lens, int n, int* out_exist) {
  ZeroArray(out_exist, n);
  return NoThrow(-1, [&] {
    if (!c || n < 0) return -1;
    if (n > 0 && (!keys || !key_lens || !out_exist)) return -1;
    std::vector<std::string> valid_keys;
    dfkv::KvRouteKeys valid_routes;
    std::vector<size_t> positions;
    valid_keys.reserve(static_cast<size_t>(n));
    if (route_keys) valid_routes.reserve(static_cast<size_t>(n));
    positions.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      if (!ValidKey(keys[i], key_lens[i]) ||
          !ValidRouteKey(route_keys, route_key_lens, i))
        continue;
      positions.push_back(static_cast<size_t>(i));
      valid_keys.push_back(CopyKey(keys[i], key_lens[i]));
      if (route_keys)
        valid_routes.push_back(RouteKeyView(route_keys, route_key_lens, i));
    }
    const auto result = static_cast<KVClient*>(c)->BatchExist(
        valid_keys, route_keys ? &valid_routes : nullptr);
    for (size_t i = 0; i < positions.size() && i < result.size(); ++i)
      out_exist[positions[i]] = result[i] ? 1 : 0;
    return 0;
  });
}

int dfkv_batch_exist(dfkv_client_t c, const void* const* keys,
                     const uint64_t* key_lens, int n, int* out_exist) {
  return BatchExistC(c, keys, key_lens, nullptr, nullptr, n, out_exist);
}

int dfkv_batch_exist_anchored(dfkv_client_t c, const void* const* keys,
                              const uint64_t* key_lens,
                              const void* const* route_keys,
                              const uint64_t* route_key_lens, int n,
                              int* out_exist) {
  return BatchExistC(c, keys, key_lens, route_keys, route_key_lens, n,
                     out_exist);
}

static int BatchRetainC(dfkv_client_t c, const void* const* keys,
                        const uint64_t* key_lens, const void* const* route_keys,
                        const uint64_t* route_key_lens, int n,
                        const dfkv_retain_options* opt, uint8_t* out_result) {
  if (out_result && n > 0)
    std::fill(out_result, out_result + n, static_cast<uint8_t>(DFKV_RET_ERROR));
  return NoThrow(-1, [&] {
    if (!c || n < 0 || !opt) return -1;
    // A field this library does not know is refused, not ignored (as in
    // dfkv_open_v2): a caller built against a later header learns it here.
    if (opt->struct_size < sizeof(dfkv_retain_options) ||
        opt->mode > DFKV_RETAIN_REFRESH || opt->reserved0 != 0 ||
        (opt->flags & ~DFKV_RETAIN_F_NO_DEBOUNCE) != 0)
      return -1;
    if (n > 0 && (!keys || !key_lens || !out_result)) return -1;
    std::vector<std::string> valid_keys;
    dfkv::KvRouteKeys valid_routes;
    std::vector<size_t> positions;
    valid_keys.reserve(static_cast<size_t>(n));
    if (route_keys) valid_routes.reserve(static_cast<size_t>(n));
    positions.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      if (!ValidKey(keys[i], key_lens[i]) ||
          !ValidRouteKey(route_keys, route_key_lens, i))
        continue;
      positions.push_back(static_cast<size_t>(i));
      valid_keys.push_back(CopyKey(keys[i], key_lens[i]));
      if (route_keys)
        valid_routes.push_back(RouteKeyView(route_keys, route_key_lens, i));
    }
    KVClient::RetainRequest req;
    req.ttl_s = opt->ttl_s;
    req.mode = opt->mode;
    req.no_debounce = (opt->flags & DFKV_RETAIN_F_NO_DEBOUNCE) != 0;
    req.min_refresh_ms = opt->min_refresh_ms;
    std::vector<uint8_t> result(valid_keys.size(), DFKV_RET_ERROR);
    static_cast<KVClient*>(c)->BatchRetain(
        valid_keys, req, result.data(), route_keys ? &valid_routes : nullptr);
    for (size_t i = 0; i < positions.size(); ++i)
      out_result[positions[i]] = result[i];
    return 0;
  });
}

int dfkv_batch_retain(dfkv_client_t c, const void* const* keys,
                      const uint64_t* key_lens, int n,
                      const dfkv_retain_options* opt, uint8_t* out_result) {
  return BatchRetainC(c, keys, key_lens, nullptr, nullptr, n, opt, out_result);
}

int dfkv_batch_retain_anchored(dfkv_client_t c, const void* const* keys,
                               const uint64_t* key_lens,
                               const void* const* route_keys,
                               const uint64_t* route_key_lens, int n,
                               const dfkv_retain_options* opt,
                               uint8_t* out_result) {
  return BatchRetainC(c, keys, key_lens, route_keys, route_key_lens, n, opt,
                      out_result);
}

int dfkv_retain_limits(dfkv_client_t c, int* supported, uint32_t* max_ttl_s) {
  if (supported) *supported = -1;
  if (max_ttl_s) *max_ttl_s = 0;
  return NoThrow(-1, [&] {
    if (!c) return -1;
    static_cast<KVClient*>(c)->RetainLimits(supported, max_ttl_s);
    return 0;
  });
}

static int RetainPrecheckC(dfkv_client_t c, const void* const* keys,
                           const uint64_t* key_lens,
                           const void* const* route_keys,
                           const uint64_t* route_key_lens, int n,
                           uint8_t min_headroom_pct, int* ok) {
  if (ok) *ok = 0;
  return NoThrow(-1, [&] {
    if (!c || n < 0 || !ok) return -1;
    if (n > 0 && (!keys || !key_lens)) return -1;
    if (min_headroom_pct == 0) {
      min_headroom_pct = 5;
      if (const char* v = std::getenv("DFKV_RETAIN_PRECHECK_MIN_HEADROOM_PCT")) {
        const long x = std::strtol(v, nullptr, 10);
        if (x >= 0 && x <= 100) min_headroom_pct = static_cast<uint8_t>(x);
      }
      // This is read on every call; the report is made on the first.
      static const bool reported =
          (dfkv::envparse::ReportUnlessPlain(
               "DFKV_RETAIN_PRECHECK_MIN_HEADROOM_PCT", 0, 100,
               std::to_string(static_cast<unsigned>(min_headroom_pct)),
               /*empty_is_unset=*/false),  // strtol("") is 0, not the default
           true);
      (void)reported;
    }
    std::vector<std::string> valid_keys;
    dfkv::KvRouteKeys valid_routes;
    valid_keys.reserve(static_cast<size_t>(n));
    if (route_keys) valid_routes.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      if (!ValidKey(keys[i], key_lens[i]) ||
          !ValidRouteKey(route_keys, route_key_lens, i))
        continue;
      valid_keys.push_back(CopyKey(keys[i], key_lens[i]));
      if (route_keys)
        valid_routes.push_back(RouteKeyView(route_keys, route_key_lens, i));
    }
    *ok = static_cast<KVClient*>(c)->RetainPrecheck(
              valid_keys, min_headroom_pct,
              route_keys ? &valid_routes : nullptr)
              ? 1
              : 0;
    return 0;
  });
}

int dfkv_retain_precheck(dfkv_client_t c, const void* const* keys,
                         const uint64_t* key_lens, int n,
                         uint8_t min_headroom_pct, int* ok) {
  return RetainPrecheckC(c, keys, key_lens, nullptr, nullptr, n,
                         min_headroom_pct, ok);
}

int dfkv_retain_precheck_anchored(
    dfkv_client_t c, const void* const* keys, const uint64_t* key_lens,
    const void* const* route_keys, const uint64_t* route_key_lens, int n,
    uint8_t min_headroom_pct, int* ok) {
  return RetainPrecheckC(c, keys, key_lens, route_keys, route_key_lens, n,
                         min_headroom_pct, ok);
}

static int BatchRemoveC(dfkv_client_t c, const void* const* keys,
                        const uint64_t* key_lens, const void* const* route_keys,
                        const uint64_t* route_key_lens, int n, int* out_ok) {
  ZeroArray(out_ok, n);
  return NoThrow(-1, [&] {
    if (!c || n < 0) return -1;
    if (n > 0 && (!keys || !key_lens || !out_ok)) return -1;
    std::vector<std::string> valid_keys;
    dfkv::KvRouteKeys valid_routes;
    std::vector<size_t> positions;
    valid_keys.reserve(static_cast<size_t>(n));
    if (route_keys) valid_routes.reserve(static_cast<size_t>(n));
    positions.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      if (!ValidKey(keys[i], key_lens[i]) ||
          !ValidRouteKey(route_keys, route_key_lens, i))
        continue;
      positions.push_back(static_cast<size_t>(i));
      valid_keys.push_back(CopyKey(keys[i], key_lens[i]));
      if (route_keys)
        valid_routes.push_back(RouteKeyView(route_keys, route_key_lens, i));
    }
    const auto result = static_cast<KVClient*>(c)->BatchRemove(
        valid_keys, route_keys ? &valid_routes : nullptr);
    for (size_t i = 0; i < positions.size() && i < result.size(); ++i)
      out_ok[positions[i]] = result[i] ? 1 : 0;
    return 0;
  });
}

int dfkv_batch_remove(dfkv_client_t c, const void* const* keys,
                      const uint64_t* key_lens, int n, int* out_ok) {
  return BatchRemoveC(c, keys, key_lens, nullptr, nullptr, n, out_ok);
}

int dfkv_batch_remove_anchored(dfkv_client_t c, const void* const* keys,
                               const uint64_t* key_lens,
                               const void* const* route_keys,
                               const uint64_t* route_key_lens, int n,
                               int* out_ok) {
  return BatchRemoveC(c, keys, key_lens, route_keys, route_key_lens, n, out_ok);
}

static int BatchPutSgC(dfkv_client_t c, const void* const* keys,
                       const uint64_t* key_lens, const void* const* route_keys,
                       const uint64_t* route_key_lens, const void*** ptrs,
                       const uint64_t** sizes, const int* num_bufs, int n,
                       int* out_ok, uint8_t* out_status) {
  ZeroArray(out_ok, n);
  FillBadArgument(out_status, n);
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
      if (!ValidKey(keys[i], key_lens[i]) ||
          !ValidRouteKey(route_keys, route_key_lens, i) || count < 0 ||
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
      item.route_key = RouteKeyView(route_keys, route_key_lens, i);
      item.ptrs.reserve(static_cast<size_t>(count));
      item.sizes.reserve(static_cast<size_t>(count));
      for (int j = 0; j < count; ++j) {
        item.ptrs.push_back(ptrs[i][j]);
        item.sizes.push_back(static_cast<size_t>(sizes[i][j]));
      }
      positions.push_back(static_cast<size_t>(i));
      items.push_back(std::move(item));
    }
    std::vector<dfkv::ItemStatus> status;
    const auto result = static_cast<KVClient*>(c)->BatchPutSg(
        items, out_status ? &status : nullptr);
    CopyStatus(status, positions, out_status);
    for (size_t i = 0; i < positions.size() && i < result.size(); ++i)
      out_ok[positions[i]] = result[i] ? 1 : 0;
    return 0;
  });
}

static int BatchGetAutoSgC(dfkv_client_t c, const void* const* keys,
                           const uint64_t* key_lens,
                           const void* const* route_keys,
                           const uint64_t* route_key_lens, void*** dsts,
                           const uint64_t** caps, const int* num_dsts, int n,
                           int* out_hit, uint64_t* out_len,
                           uint8_t* out_status) {
  ZeroArray(out_hit, n);
  FillBadArgument(out_status, n);
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
      if (!ValidKey(keys[i], key_lens[i]) ||
          !ValidRouteKey(route_keys, route_key_lens, i) || count < 0 ||
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
      item.route_key = RouteKeyView(route_keys, route_key_lens, i);
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
    std::vector<dfkv::ItemStatus> status;
    const auto result = static_cast<KVClient*>(c)->BatchGetAutoSg(
        items, &lengths, out_status ? &status : nullptr);
    CopyStatus(status, positions, out_status);
    for (size_t i = 0; i < positions.size() && i < result.size(); ++i) {
      out_hit[positions[i]] = result[i] ? 1 : 0;
      if (result[i] && i < lengths.size())
        out_len[positions[i]] = static_cast<uint64_t>(lengths[i]);
    }
    return 0;
  });
}

int dfkv_batch_put(dfkv_client_t c, const void* const* keys,
                   const uint64_t* key_lens, const void** ptrs,
                   const uint64_t* sizes, int n, int* out_ok) {
  return BatchPutC(c, keys, key_lens, nullptr, nullptr, ptrs, sizes, n, out_ok, nullptr);
}
int dfkv_batch_put_st(dfkv_client_t c, const void* const* keys,
                      const uint64_t* key_lens, const void** ptrs,
                      const uint64_t* sizes, int n, int* out_ok,
                      uint8_t* out_status) {
  if (n > 0 && !out_status) {
    ZeroArray(out_ok, n);
    return -1;
  }
  return BatchPutC(c, keys, key_lens, nullptr, nullptr, ptrs, sizes, n, out_ok, out_status);
}
int dfkv_batch_get(dfkv_client_t c, const void* const* keys,
                   const uint64_t* key_lens, void** ptrs,
                   const uint64_t* sizes, int n, int* out_hit) {
  return BatchGetC(c, keys, key_lens, nullptr, nullptr, ptrs, sizes, n, out_hit, nullptr);
}
int dfkv_batch_get_st(dfkv_client_t c, const void* const* keys,
                      const uint64_t* key_lens, void** ptrs,
                      const uint64_t* sizes, int n, int* out_hit,
                      uint8_t* out_status) {
  if (n > 0 && !out_status) {
    ZeroArray(out_hit, n);
    return -1;
  }
  return BatchGetC(c, keys, key_lens, nullptr, nullptr, ptrs, sizes, n, out_hit, out_status);
}
int dfkv_batch_get_auto(dfkv_client_t c, const void* const* keys,
                        const uint64_t* key_lens, void** ptrs,
                        const uint64_t* caps, int n, int* out_hit,
                        uint64_t* out_len) {
  return BatchGetAutoC(c, keys, key_lens, nullptr, nullptr, ptrs, caps, n, out_hit, out_len,
                       nullptr);
}
int dfkv_batch_get_auto_st(dfkv_client_t c, const void* const* keys,
                           const uint64_t* key_lens, void** ptrs,
                           const uint64_t* caps, int n, int* out_hit,
                           uint64_t* out_len, uint8_t* out_status) {
  if (n > 0 && !out_status) {
    ZeroArray(out_hit, n);
    return -1;
  }
  return BatchGetAutoC(c, keys, key_lens, nullptr, nullptr, ptrs, caps, n, out_hit, out_len,
                       out_status);
}
int dfkv_batch_put_sg(dfkv_client_t c, const void* const* keys,
                      const uint64_t* key_lens, const void*** ptrs,
                      const uint64_t** sizes, const int* num_bufs, int n,
                      int* out_ok) {
  return BatchPutSgC(c, keys, key_lens, nullptr, nullptr, ptrs, sizes, num_bufs, n, out_ok,
                     nullptr);
}
int dfkv_batch_put_sg_st(dfkv_client_t c, const void* const* keys,
                         const uint64_t* key_lens, const void*** ptrs,
                         const uint64_t** sizes, const int* num_bufs, int n,
                         int* out_ok, uint8_t* out_status) {
  if (n > 0 && !out_status) {
    ZeroArray(out_ok, n);
    return -1;
  }
  return BatchPutSgC(c, keys, key_lens, nullptr, nullptr, ptrs, sizes, num_bufs, n, out_ok,
                     out_status);
}
int dfkv_batch_get_auto_sg(dfkv_client_t c, const void* const* keys,
                           const uint64_t* key_lens, void*** dsts,
                           const uint64_t** caps, const int* num_dsts, int n,
                           int* out_hit, uint64_t* out_len) {
  return BatchGetAutoSgC(c, keys, key_lens, nullptr, nullptr, dsts, caps, num_dsts, n, out_hit,
                         out_len, nullptr);
}
int dfkv_batch_get_auto_sg_st(dfkv_client_t c, const void* const* keys,
                              const uint64_t* key_lens, void*** dsts,
                              const uint64_t** caps, const int* num_dsts,
                              int n, int* out_hit, uint64_t* out_len,
                              uint8_t* out_status) {
  if (n > 0 && !out_status) {
    ZeroArray(out_hit, n);
    return -1;
  }
  return BatchGetAutoSgC(c, keys, key_lens, nullptr, nullptr, dsts, caps, num_dsts, n, out_hit,
                         out_len, out_status);
}

int dfkv_batch_put_anchored(dfkv_client_t c, const void* const* keys,
                            const uint64_t* key_lens,
                            const void* const* route_keys,
                            const uint64_t* route_key_lens, const void** ptrs,
                            const uint64_t* sizes, int n, int* out_ok) {
  return BatchPutC(c, keys, key_lens, route_keys, route_key_lens, ptrs, sizes,
                   n, out_ok, nullptr);
}

int dfkv_batch_put_st_anchored(dfkv_client_t c, const void* const* keys,
                               const uint64_t* key_lens,
                               const void* const* route_keys,
                               const uint64_t* route_key_lens, const void** ptrs,
                               const uint64_t* sizes, int n, int* out_ok,
                               uint8_t* out_status) {
  if (n > 0 && !out_status) {
    ZeroArray(out_ok, n);
    return -1;
  }
  return BatchPutC(c, keys, key_lens, route_keys, route_key_lens, ptrs, sizes,
                   n, out_ok, out_status);
}

int dfkv_batch_get_anchored(dfkv_client_t c, const void* const* keys,
                            const uint64_t* key_lens,
                            const void* const* route_keys,
                            const uint64_t* route_key_lens, void** ptrs,
                            const uint64_t* sizes, int n, int* out_hit) {
  return BatchGetC(c, keys, key_lens, route_keys, route_key_lens, ptrs, sizes,
                   n, out_hit, nullptr);
}

int dfkv_batch_get_st_anchored(dfkv_client_t c, const void* const* keys,
                               const uint64_t* key_lens,
                               const void* const* route_keys,
                               const uint64_t* route_key_lens, void** ptrs,
                               const uint64_t* sizes, int n, int* out_hit,
                               uint8_t* out_status) {
  if (n > 0 && !out_status) {
    ZeroArray(out_hit, n);
    return -1;
  }
  return BatchGetC(c, keys, key_lens, route_keys, route_key_lens, ptrs, sizes,
                   n, out_hit, out_status);
}

int dfkv_batch_get_auto_anchored(dfkv_client_t c, const void* const* keys,
                                 const uint64_t* key_lens,
                                 const void* const* route_keys,
                                 const uint64_t* route_key_lens, void** ptrs,
                                 const uint64_t* caps, int n, int* out_hit,
                                 uint64_t* out_len) {
  return BatchGetAutoC(c, keys, key_lens, route_keys, route_key_lens, ptrs,
                       caps, n, out_hit, out_len, nullptr);
}

int dfkv_batch_get_auto_st_anchored(
    dfkv_client_t c, const void* const* keys, const uint64_t* key_lens,
    const void* const* route_keys, const uint64_t* route_key_lens, void** ptrs,
    const uint64_t* caps, int n, int* out_hit, uint64_t* out_len,
    uint8_t* out_status) {
  if (n > 0 && !out_status) {
    ZeroArray(out_hit, n);
    return -1;
  }
  return BatchGetAutoC(c, keys, key_lens, route_keys, route_key_lens, ptrs,
                       caps, n, out_hit, out_len, out_status);
}

int dfkv_batch_put_sg_anchored(dfkv_client_t c, const void* const* keys,
                               const uint64_t* key_lens,
                               const void* const* route_keys,
                               const uint64_t* route_key_lens,
                               const void*** ptrs, const uint64_t** sizes,
                               const int* num_bufs, int n, int* out_ok) {
  return BatchPutSgC(c, keys, key_lens, route_keys, route_key_lens, ptrs, sizes,
                     num_bufs, n, out_ok, nullptr);
}

int dfkv_batch_put_sg_st_anchored(
    dfkv_client_t c, const void* const* keys, const uint64_t* key_lens,
    const void* const* route_keys, const uint64_t* route_key_lens,
    const void*** ptrs, const uint64_t** sizes, const int* num_bufs, int n,
    int* out_ok, uint8_t* out_status) {
  if (n > 0 && !out_status) {
    ZeroArray(out_ok, n);
    return -1;
  }
  return BatchPutSgC(c, keys, key_lens, route_keys, route_key_lens, ptrs, sizes,
                     num_bufs, n, out_ok, out_status);
}

int dfkv_batch_get_auto_sg_anchored(
    dfkv_client_t c, const void* const* keys, const uint64_t* key_lens,
    const void* const* route_keys, const uint64_t* route_key_lens, void*** dsts,
    const uint64_t** caps, const int* num_dsts, int n, int* out_hit,
    uint64_t* out_len) {
  return BatchGetAutoSgC(c, keys, key_lens, route_keys, route_key_lens, dsts,
                         caps, num_dsts, n, out_hit, out_len, nullptr);
}

int dfkv_batch_get_auto_sg_st_anchored(
    dfkv_client_t c, const void* const* keys, const uint64_t* key_lens,
    const void* const* route_keys, const uint64_t* route_key_lens, void*** dsts,
    const uint64_t** caps, const int* num_dsts, int n, int* out_hit,
    uint64_t* out_len, uint8_t* out_status) {
  if (n > 0 && !out_status) {
    ZeroArray(out_hit, n);
    return -1;
  }
  return BatchGetAutoSgC(c, keys, key_lens, route_keys, route_key_lens, dsts,
                         caps, num_dsts, n, out_hit, out_len, out_status);
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
