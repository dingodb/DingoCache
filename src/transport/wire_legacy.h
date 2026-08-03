/* Legacy (v1.x) control-plane wire — mixed-generation rollout shim.
 *
 * Every v1.x release (v1.37.0..v1.40.x) speaks protocol epoch 1 on the TCP
 * control plane (v1 wire.h, kProtoVersion = 1): a 42-byte request prefix
 * (ver, op, id:u64, index:u32, size:u32, offset:u64, length:u64,
 * payload_len:u64) and a 10-byte response prefix (ver, status, data_len).
 * The v2 data plane stays strictly epoch-6/7; these helpers exist so the
 * MDS can serve BOTH control-plane generations during a ring-level
 * migration (one group per generation — the shim never lets a v1 peer
 * route data to a v2 node or vice versa).
 *
 * Only the FRAMING differs; op codes, the group+MemberInfo payloads, the
 * etcd key/value encoding and the lease contract are identical across
 * generations (see src/common/membership.h, unchanged).
 *
 * Status byte caveat: v2 inserted kQuotaExceeded at 3, shifting kIOError to
 * 4. All legacy peers (v1.37..v1.40.x, incl. v1.40.x clients) decode
 * {0=kOk, 1=kNotFound, 2=kCacheFull, 3=kIOError, 4=kInvalid}. Legacy
 * responses therefore map v2 statuses back onto the v1 enum; kQuotaExceeded
 * never occurs on MDS op paths and folds to kIOError for byte sanity.
 */
#ifndef DFKV_WIRE_LEGACY_H_
#define DFKV_WIRE_LEGACY_H_

#include <cstddef>
#include <cstdint>

#include "common/status.h"
#include "transport/wire.h"    // ReqFields, kMaxFrameLen
#include "utils/net_util.h"    // net::PutU64/PutU32/GetU64

namespace dfkv {

// Control-plane epoch of every v1.x release.
constexpr uint8_t kNativeProtoLegacyTcp = 1;

// Legacy request prefix: ver(1) op(1) id(8) index(4) size(4) offset(8)
//                        length(8) payload_len(8)
constexpr size_t kLegacyReqPrefix = 1 + 1 + 8 + 4 + 4 + 8 + 8 + 8;  // = 42
// Legacy response prefix: ver(1) status(1) data_len(8)
constexpr size_t kLegacyRespPrefix = 1 + 1 + 8;  // = 10

// Encoder used by tests/benchmarks to synthesize legacy peers; production
// code only ever *decodes* legacy frames.
inline void EncodeLegacyReq(char* p, WireOp op, uint64_t payload_len) {
  p[0] = static_cast<char>(kNativeProtoLegacyTcp);
  p[1] = static_cast<char>(op);
  net::PutU64(p + 2, 0);   // v1 BlockKey.id
  net::PutU32(p + 10, 0);  // v1 BlockKey.index
  net::PutU32(p + 14, 0);  // v1 BlockKey.size
  net::PutU64(p + 18, 0);  // offset
  net::PutU64(p + 26, 0);  // length
  net::PutU64(p + 34, payload_len);
}

// Fills the shared ReqFields shape; false on epoch mismatch or an oversized
// declared payload (caller drops the connection). The v1 id/index/size key
// fields are deliberately NOT decoded: MDS control ops are key-free, and the
// data plane — the only consumer of keys — rejects legacy epochs outright.
inline bool DecodeLegacyReq(const char* p, ReqFields* o,
                            uint64_t max_payload = kMaxFrameLen) {
  if (static_cast<uint8_t>(p[0]) != kNativeProtoLegacyTcp) return false;
  o->op = static_cast<uint8_t>(p[1]);
  o->tenant_hash = 0;
  o->digest_hi = 0;
  o->digest_lo = 0;
  o->offset = 0;
  o->length = 0;
  o->payload_len = net::GetU64(p + 34);
  return o->payload_len <= max_payload;
}

// v2 status -> v1 enum byte (see the header comment for the enum shift).
inline uint8_t LegacyStatusByte(Status st) {
  switch (st) {
    case Status::kOk: return 0;
    case Status::kNotFound: return 1;
    case Status::kCacheFull: return 2;
    case Status::kQuotaExceeded: return 3;  // v2-only; folds into legacy kIOError
    case Status::kIOError: return 3;
    case Status::kInvalid: return 4;
  }
  return 4;
}

inline void EncodeLegacyResp(char* p, Status st, uint64_t data_len) {
  p[0] = static_cast<char>(kNativeProtoLegacyTcp);
  p[1] = static_cast<char>(LegacyStatusByte(st));
  net::PutU64(p + 2, data_len);
}

}  // namespace dfkv

#endif  // DFKV_WIRE_LEGACY_H_
