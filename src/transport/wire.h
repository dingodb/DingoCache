/* Versioned native dfkv wire.
 * TCP epoch 6 and RDMA epoch 7 carry a tenant hash plus the full 128-bit
 * object identity and authoritative stored-value length. Other epochs are
 * rejected before field decoding.
 */
#ifndef DFKV_WIRE_H_
#define DFKV_WIRE_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "common/status.h"
#include "common/kv_types.h"   // BlockKey
#include "utils/net_util.h"   // net::PutU32/PutU64/GetU32/GetU64 (host-endian codec)

namespace dfkv {

// Wire op codes shared by transports and servers. Variable request content
// rides the payload/data blob.
enum class WireOp : uint8_t {
  kCache = 1, kRange = 2, kExist = 3, kStats = 4, kMembers = 5,
  kRegister = 6, kHeartbeat = 7, kListMembers = 8, kRemove = 9,
  kListGroups = 10,  // MDS: newline-joined distinct group names (dfkvctl stats --all)
  // Client registration (mirrors kRegister/kHeartbeat/kListMembers but for cache
  // *consumers* — inference instances — so the MDS can surface "who is using dfkv"
  // via `dfkvctl clients` + the dfkv_mds_group_clients gauge). Same payload framing
  // (group + MemberInfo) and lease semantics; only the etcd key prefix differs
  // (/clients/<id> vs /members/<id>), so clients never pollute the placement ring.
  kClientRegister = 11, kClientHeartbeat = 12, kListClients = 13,
  kLookup = 14,  // payload-free stored-value metadata lookup
  kListTopology = 15  // MDS: all members, including health-degraded nodes
};

constexpr uint8_t kNativeProtoTcp = 6;
constexpr uint8_t kNativeProtoRdmaV2 = 7;
constexpr uint8_t kProtoVersion = kNativeProtoTcp;

// Hard ceiling on a single wire frame's variable payload. Decode rejects any
// frame whose declared length exceeds this, so a garbage/hostile 64-bit length
// (a version skew, corruption, or a hostile peer) can't drive a multi-exabyte
// std::vector/std::string allocation -> bad_alloc/OOM that kills the process.
// No real dfkv frame (one KV block value, or a stats/membership blob) comes
// anywhere near 16 GiB; callers that know a tighter bound pass it explicitly.
constexpr uint64_t kMaxFrameLen = 1ull << 34;  // 16 GiB

// Request prefix: ver(1) op(1) tenant_hash(8) digest_hi(8) digest_lo(8)
//                 offset(8) length(8) payload_len(8)
constexpr size_t kReqPrefix = 1 + 1 + 8 + 8 + 8 + 8 + 8 + 8;  // = 50
// Response: ver(1) status(1) data_len(8) stored_value_len(8).
constexpr size_t kRespPrefix = 1 + 1 + 8 + 8;
static_assert(static_cast<uint8_t>(Status::kOk) == 0 &&
                  static_cast<uint8_t>(Status::kNotFound) == 1 &&
                  static_cast<uint8_t>(Status::kCacheFull) == 2 &&
                  static_cast<uint8_t>(Status::kQuotaExceeded) == 3 &&
                  static_cast<uint8_t>(Status::kIOError) == 4 &&
                  static_cast<uint8_t>(Status::kInvalid) == 5,
              "Status wire values must remain contiguous and append-only");

inline void EncodeReqVersion(char* p, uint8_t version, WireOp op,
                             const BlockKey& k, uint64_t offset,
                             uint64_t length, uint64_t payload_len) {
  p[0] = static_cast<char>(version);
  p[1] = static_cast<char>(op);
  net::PutU64(p + 2, k.tenant_hash);
  net::PutU64(p + 10, k.digest_hi);
  net::PutU64(p + 18, k.digest_lo);
  net::PutU64(p + 26, offset);
  net::PutU64(p + 34, length);
  net::PutU64(p + 42, payload_len);
}

inline void EncodeReq(char* p, WireOp op, const BlockKey& k, uint64_t offset,
                      uint64_t length, uint64_t payload_len) {
  EncodeReqVersion(p, kNativeProtoTcp, op, k, offset, length, payload_len);
}

struct ReqFields {
  uint8_t op;
  uint64_t tenant_hash;
  uint64_t digest_hi;
  uint64_t digest_lo;
  uint64_t offset;
  uint64_t length;
  uint64_t payload_len;

  BlockKey Key() const { return BlockKey{digest_hi, digest_lo, tenant_hash}; }
};

// False on a version mismatch or an oversized declared payload (> max_payload)
// — the caller drops the connection.
inline bool DecodeReqVersion(const char* p, uint8_t expected_version,
                             ReqFields* o,
                             uint64_t max_payload = kMaxFrameLen) {
  if (static_cast<uint8_t>(p[0]) != expected_version) return false;
  o->op = static_cast<uint8_t>(p[1]);
  o->tenant_hash = net::GetU64(p + 2);
  o->digest_hi = net::GetU64(p + 10);
  o->digest_lo = net::GetU64(p + 18);
  o->offset = net::GetU64(p + 26);
  o->length = net::GetU64(p + 34);
  o->payload_len = net::GetU64(p + 42);
  return o->payload_len <= max_payload;
}

inline bool DecodeReq(const char* p, ReqFields* o,
                      uint64_t max_payload = kMaxFrameLen) {
  return DecodeReqVersion(p, kNativeProtoTcp, o, max_payload);
}

inline void EncodeRespVersion(char* p, uint8_t version, Status st,
                              uint64_t data_len, uint64_t value_len = 0) {
  p[0] = static_cast<char>(version);
  p[1] = static_cast<char>(st);
  net::PutU64(p + 2, data_len);
  net::PutU64(p + 10, value_len);
}

inline void EncodeResp(char* p, Status st, uint64_t data_len,
                       uint64_t value_len = 0) {
  EncodeRespVersion(p, kNativeProtoTcp, st, data_len, value_len);
}

// False on version mismatch, an unknown Status byte, or an oversized declared
// length. Outputs are assigned only after the complete prefix validates.
inline bool DecodeRespVersion(const char* p, uint8_t expected_version,
                              Status* st, uint64_t* data_len,
                              uint64_t max_data = kMaxFrameLen,
                              uint64_t* value_len = nullptr) {
  if (static_cast<uint8_t>(p[0]) != expected_version) return false;
  const uint8_t raw_status = static_cast<uint8_t>(p[1]);
  if (raw_status > static_cast<uint8_t>(Status::kInvalid)) return false;
  const uint64_t decoded_data_len = net::GetU64(p + 2);
  if (decoded_data_len > max_data) return false;
  *st = static_cast<Status>(raw_status);
  *data_len = decoded_data_len;
  if (value_len != nullptr) *value_len = net::GetU64(p + 10);
  return true;
}

inline bool DecodeResp(const char* p, Status* st, uint64_t* data_len,
                       uint64_t max_data = kMaxFrameLen,
                       uint64_t* value_len = nullptr) {
  return DecodeRespVersion(p, kNativeProtoTcp, st, data_len, max_data,
                           value_len);
}

// RDMA v2 GET carries one or more client destinations after the request prefix.
// The server returns metadata in the response prefix and RDMA-WRITEs raw value
// bytes into these regions in order.
struct RdmaWriteTarget {
  uint64_t addr = 0;
  uint32_t rkey = 0;
  uint32_t length = 0;
};

struct RdmaGetFields {
  std::vector<RdmaWriteTarget> targets;

  bool Capacity(uint64_t* capacity) const {
    if (capacity == nullptr) return false;
    uint64_t out = 0;
    for (const auto& target : targets) {
      if (target.length > std::numeric_limits<uint64_t>::max() - out)
        return false;
      out += target.length;
    }
    *capacity = out;
    return true;
  }

  uint64_t Capacity() const {
    uint64_t out = 0;
    return Capacity(&out) ? out : std::numeric_limits<uint64_t>::max();
  }
};

constexpr size_t kRdmaGetFixed = 4;   // target_count(4)
constexpr size_t kRdmaGetTarget = 16; // addr(8), rkey(4), length(4)

inline size_t RdmaGetFrameSize(size_t target_count) {
  if (target_count >
      (std::numeric_limits<size_t>::max() - kReqPrefix - kRdmaGetFixed) /
          kRdmaGetTarget) {
    return 0;
  }
  return kReqPrefix + kRdmaGetFixed + target_count * kRdmaGetTarget;
}

inline bool EncodeRdmaGetReq(char* p, size_t cap, const BlockKey& key,
                             uint64_t offset, uint64_t length,
                             const std::vector<RdmaWriteTarget>& targets,
                             size_t* encoded_len) {
  const size_t frame_len = RdmaGetFrameSize(targets.size());
  if (p == nullptr || encoded_len == nullptr || frame_len == 0 ||
      frame_len > cap ||
      targets.size() > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  uint64_t capacity = 0;
  for (const auto& target : targets) {
    if ((target.length != 0 && (target.addr == 0 || target.rkey == 0)) ||
        target.length > std::numeric_limits<uint64_t>::max() - capacity) {
      return false;
    }
    capacity += target.length;
  }
  if (capacity < length) return false;
  EncodeReqVersion(p, kNativeProtoRdmaV2, WireOp::kRange, key, offset, length,
                   0);
  net::PutU32(p + kReqPrefix, static_cast<uint32_t>(targets.size()));
  char* out = p + kReqPrefix + kRdmaGetFixed;
  for (const auto& target : targets) {
    net::PutU64(out, target.addr);
    net::PutU32(out + 8, target.rkey);
    net::PutU32(out + 12, target.length);
    out += kRdmaGetTarget;
  }
  *encoded_len = frame_len;
  return true;
}

inline bool DecodeRdmaGetReq(const char* p, size_t frame_len, ReqFields* req,
                             RdmaGetFields* get,
                             uint64_t max_payload = kMaxFrameLen) {
  if (frame_len < kReqPrefix + kRdmaGetFixed ||
      !DecodeReqVersion(p, kNativeProtoRdmaV2, req, max_payload) ||
      req->op != static_cast<uint8_t>(WireOp::kRange) ||
      req->payload_len != 0) {
    return false;
  }
  const uint32_t count = net::GetU32(p + kReqPrefix);
  const size_t expected = RdmaGetFrameSize(count);
  if (expected == 0 || expected != frame_len) return false;
  get->targets.clear();
  get->targets.reserve(count);
  const char* in = p + kReqPrefix + kRdmaGetFixed;
  uint64_t capacity = 0;
  for (uint32_t i = 0; i < count; ++i) {
    RdmaWriteTarget target;
    target.addr = net::GetU64(in);
    target.rkey = net::GetU32(in + 8);
    target.length = net::GetU32(in + 12);
    if (target.length != 0 && (target.addr == 0 || target.rkey == 0)) {
      return false;
    }
    if (capacity > std::numeric_limits<uint64_t>::max() - target.length) {
      return false;
    }
    capacity += target.length;
    get->targets.push_back(target);
    in += kRdmaGetTarget;
  }
  if (capacity < req->length) return false;
  return true;
}

}  // namespace dfkv

#endif  // DFKV_WIRE_H_
