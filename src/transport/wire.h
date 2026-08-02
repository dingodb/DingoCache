/* dfkv native wire protocol — the single request/response framing contract.
 * Every frame carries a native epoch byte. Epoch 3 introduced full 128-bit
 * BlockKey digests; RDMA scatter/gather uses epoch 4. Older native clients are
 * rejected before any field is decoded. SGEngine v1 uses isolated frontends
 * and never enters this codec.
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

// Wire op codes (shared by TcpTransport, RdmaTransport and the server).
// kMembers is the legacy static-list discovery query, superseded by the MDS (kListMembers).
// kRegister/kHeartbeat/kListMembers are the MDS membership ops (M0+): the op byte
// reuses the existing request framing; variable content rides the payload/data blob.
enum class WireOp : uint8_t {
  kCache = 1, kRange = 2, kExist = 3, kStats = 4, kMembers = 5,
  kRegister = 6, kHeartbeat = 7, kListMembers = 8, kRemove = 9,
  kListGroups = 10,  // MDS: newline-joined distinct group names (dfkvctl stats --all)
  // Client registration (mirrors kRegister/kHeartbeat/kListMembers but for cache
  // *consumers* — inference instances — so the MDS can surface "who is using dfkv"
  // via `dfkvctl clients` + the dfkv_mds_group_clients gauge). Same payload framing
  // (group + MemberInfo) and lease semantics; only the etcd key prefix differs
  // (/clients/<id> vs /members/<id>), so clients never pollute the placement ring.
  kClientRegister = 11, kClientHeartbeat = 12, kListClients = 13
};

constexpr uint8_t kSgEngineProtoV1 = 1;
constexpr uint8_t kNativeProtoBase = 3;
constexpr uint8_t kNativeProtoRdmaV2 = 4;
constexpr uint8_t kProtoVersion = kNativeProtoBase;

// Hard ceiling on a single wire frame's variable payload. Decode rejects any
// frame whose declared length exceeds this, so a garbage/hostile 64-bit length
// (a version skew, corruption, or a hostile peer) can't drive a multi-exabyte
// std::vector/std::string allocation -> bad_alloc/OOM that kills the process.
// No real dfkv frame (one KV block value, or a stats/membership blob) comes
// anywhere near 16 GiB; callers that know a tighter bound pass it explicitly.
constexpr uint64_t kMaxFrameLen = 1ull << 34;  // 16 GiB

// Request prefix: ver(1) op(1) digest_hi(8) digest_lo(8)
//                 offset(8) length(8) payload_len(8)
constexpr size_t kReqPrefix = 1 + 1 + 8 + 8 + 8 + 8 + 8;  // = 42
// Response prefix: ver(1) status(1) data_len(8)
constexpr size_t kRespPrefix = 1 + 1 + 8;  // = 10

inline void EncodeReqVersion(char* p, uint8_t version, WireOp op,
                             const BlockKey& k, uint64_t offset,
                             uint64_t length, uint64_t payload_len) {
  p[0] = static_cast<char>(version);
  p[1] = static_cast<char>(op);
  net::PutU64(p + 2, k.digest_hi);
  net::PutU64(p + 10, k.digest_lo);
  net::PutU64(p + 18, offset);
  net::PutU64(p + 26, length);
  net::PutU64(p + 34, payload_len);
}

inline void EncodeReq(char* p, WireOp op, const BlockKey& k, uint64_t offset,
                      uint64_t length, uint64_t payload_len) {
  EncodeReqVersion(p, kNativeProtoBase, op, k, offset, length, payload_len);
}

struct ReqFields {
  uint8_t op;
  uint64_t digest_hi;
  uint64_t digest_lo;
  uint64_t offset;
  uint64_t length;
  uint64_t payload_len;

  BlockKey Key() const { return BlockKey{digest_hi, digest_lo}; }
};

// False on a version mismatch or an oversized declared payload (> max_payload)
// — the caller drops the connection.
inline bool DecodeReqVersion(const char* p, uint8_t expected_version,
                             ReqFields* o,
                             uint64_t max_payload = kMaxFrameLen) {
  if (static_cast<uint8_t>(p[0]) != expected_version) return false;
  o->op = static_cast<uint8_t>(p[1]);
  o->digest_hi = net::GetU64(p + 2);
  o->digest_lo = net::GetU64(p + 10);
  o->offset = net::GetU64(p + 18);
  o->length = net::GetU64(p + 26);
  o->payload_len = net::GetU64(p + 34);
  return o->payload_len <= max_payload;
}

inline bool DecodeReq(const char* p, ReqFields* o,
                      uint64_t max_payload = kMaxFrameLen) {
  return DecodeReqVersion(p, kNativeProtoBase, o, max_payload);
}

inline void EncodeRespVersion(char* p, uint8_t version, Status st,
                              uint64_t data_len) {
  p[0] = static_cast<char>(version);
  p[1] = static_cast<char>(st);
  net::PutU64(p + 2, data_len);
}

inline void EncodeResp(char* p, Status st, uint64_t data_len) {
  EncodeRespVersion(p, kNativeProtoBase, st, data_len);
}

// False on version mismatch or an oversized declared data_len (> max_data).
inline bool DecodeRespVersion(const char* p, uint8_t expected_version,
                              Status* st, uint64_t* data_len,
                              uint64_t max_data = kMaxFrameLen) {
  if (static_cast<uint8_t>(p[0]) != expected_version) return false;
  *st = static_cast<Status>(static_cast<uint8_t>(p[1]));
  *data_len = net::GetU64(p + 2);
  return *data_len <= max_data;
}

inline bool DecodeResp(const char* p, Status* st, uint64_t* data_len,
                       uint64_t max_data = kMaxFrameLen) {
  return DecodeRespVersion(p, kNativeProtoBase, st, data_len, max_data);
}

// RDMA v2 GET carries one or more client destinations after the unchanged
// request prefix.  The server sends the first header_len result bytes in the
// status SEND and RDMA-WRITEs the remainder into these regions in order.
struct RdmaWriteTarget {
  uint64_t addr = 0;
  uint32_t rkey = 0;
  uint32_t length = 0;
};

struct RdmaGetFields {
  uint32_t header_len = 0;
  std::vector<RdmaWriteTarget> targets;

  uint64_t Capacity() const {
    uint64_t out = 0;
    for (const auto& target : targets) out += target.length;
    return out;
  }
};

constexpr size_t kRdmaGetFixed = 8;   // header_len(4), target_count(4)
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
                             uint32_t header_len,
                             const std::vector<RdmaWriteTarget>& targets,
                             size_t* encoded_len) {
  const size_t frame_len = RdmaGetFrameSize(targets.size());
  if (frame_len == 0 || frame_len > cap ||
      targets.size() > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  EncodeReqVersion(p, kNativeProtoRdmaV2, WireOp::kRange, key, offset, length,
                   0);
  net::PutU32(p + kReqPrefix, header_len);
  net::PutU32(p + kReqPrefix + 4, static_cast<uint32_t>(targets.size()));
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
  const uint32_t count = net::GetU32(p + kReqPrefix + 4);
  const size_t expected = RdmaGetFrameSize(count);
  if (expected == 0 || expected != frame_len) return false;
  get->header_len = net::GetU32(p + kReqPrefix);
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
  if (get->header_len > req->length ||
      capacity < req->length - get->header_len) {
    return false;
  }
  return true;
}

}  // namespace dfkv

#endif  // DFKV_WIRE_H_
