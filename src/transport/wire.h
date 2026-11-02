/* Versioned native dfkv wire.
 * TCP epoch 6 and RDMA epoch 8 carry a tenant hash plus the full 128-bit
 * object identity and authoritative stored-value length. Other epochs are
 * rejected before field decoding.
 */
#ifndef DFKV_WIRE_H_
#define DFKV_WIRE_H_

#include <cstddef>
#include <cstdint>

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
  kListTopology = 15,  // MDS: all members, including health-degraded nodes
  kPullRange = 16,     // RDMA v2: prepare one connection-private pull slot
  kPullRelease = 17,  // RDMA v2: release slot after initiator READ terminal
  kLeasePut = 18,     // RDMA v2: request one per-op staging lease; the reply
                      // carries the leased range, the client then sends the
                      // object as an ordinary v2 multi/single-window PUT WRITE
  kRetain = 19        // batched minimum-lifetime control op (see below)
};

constexpr uint8_t kNativeProtoTcp = 6;
constexpr uint8_t kNativeProtoRdmaV2 = 8;
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
  // kInvalid stays the highest legal WIRE status: values above it (e.g.
  // kResourceExhausted) are client-local admission outcomes that no peer may
  // ever encode; keep rejecting them here.
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

// ---- kRetain (op 19) ----
// A control op on the TCP lane and the RDMA control lane; the data path is
// untouched. The prefix key carries only the tenant hash (digests 0) and
// `length` = key count. A node without retention answers kInvalid, exactly
// like any unknown op, and keeps the connection.
//   payload : u32 ttl_s | u8 mode (0 SET, 1 REFRESH) | u8 flags (0)
//             | u16 reserved (0) | count x {u64 digest_hi, u64 digest_lo}
//   reply   : count x u8 RetainCode | u32 server_now_s | u32 max_ttl_s
//             | u8 headroom_pct | 3 x u8 reserved (0)
// The server clamps ttl_s to max_ttl_s. headroom_pct is the node's remaining
// retention budget after this request (0..100).
constexpr size_t kRetainHeaderBytes = 8;
constexpr size_t kRetainKeyBytes = 16;
constexpr size_t kRetainTrailerBytes = 12;
// Fits one RDMA control frame (32 KiB) in both directions.
constexpr size_t kRetainMaxKeys = 1024;
constexpr uint8_t kRetainModeSet = 0;
constexpr uint8_t kRetainModeRefresh = 1;

struct RetainTrailer {
  uint32_t server_now_s = 0;
  uint32_t max_ttl_s = 0;
  uint8_t headroom_pct = 0;
};

inline size_t RetainPayloadBytes(size_t count) {
  return kRetainHeaderBytes + count * kRetainKeyBytes;
}

inline void EncodeRetainPayload(char* p, uint32_t ttl_s, uint8_t mode,
                                const BlockKey* keys, size_t count) {
  net::PutU32(p, ttl_s);
  p[4] = static_cast<char>(mode);
  p[5] = 0;
  p[6] = 0;
  p[7] = 0;
  char* k = p + kRetainHeaderBytes;
  for (size_t i = 0; i < count; ++i, k += kRetainKeyBytes) {
    net::PutU64(k, keys[i].digest_hi);
    net::PutU64(k + 8, keys[i].digest_lo);
  }
}

inline void EncodeRetainTrailer(char* p, const RetainTrailer& t) {
  net::PutU32(p, t.server_now_s);
  net::PutU32(p + 4, t.max_ttl_s);
  p[8] = static_cast<char>(t.headroom_pct);
  p[9] = 0;
  p[10] = 0;
  p[11] = 0;
}

inline RetainTrailer DecodeRetainTrailer(const char* p) {
  RetainTrailer t;
  t.server_now_s = net::GetU32(p);
  t.max_ttl_s = net::GetU32(p + 4);
  t.headroom_pct = static_cast<uint8_t>(p[8]);
  return t;
}


}  // namespace dfkv

#endif  // DFKV_WIRE_H_
