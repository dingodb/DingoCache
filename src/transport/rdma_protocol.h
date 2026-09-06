/* RDMA v2 bootstrap/data-plane constants and verbs-free codecs. */
#ifndef DFKV_TRANSPORT_RDMA_PROTOCOL_H_
#define DFKV_TRANSPORT_RDMA_PROTOCOL_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#include "transport/dev_frame.h"
#include "transport/wire.h"
#include "utils/net_util.h"

namespace dfkv::rdma {

// v2 endpoints keep bounded two-sided request/response buffers. Large PUT
// payloads land at the aligned data area of a shared server receive segment;
// GET payloads are RDMA-WRITEd into client-owned memory. Members is the only
// variable-size two-sided datapath response and has an explicit 32-KiB contract.
// The prefix is included in the registered control-buffer capacity so the exact
// boundary is representable without truncation or a connection abort.
constexpr size_t kV2ControlResponseMax = 32u << 10;
constexpr size_t kV2ControlCap = kRespPrefix + kV2ControlResponseMax;
constexpr size_t kV2DataOffset = 4096;
constexpr size_t kV2PutPrefixOffset = kV2DataOffset - kReqPrefix;
// Fleet-wide protocol bound. A GET target is one server RDMA WRITE WR, not one
// local QP SGE; tying this to either peer's max_sge breaks heterogeneous HCAs.
// Keep it equal to the public SG layout's maximum payload width.
constexpr size_t kV2MaxGetTargets = 29;

// A kCache request with offset=kV2MultiWrPutMagic and length>1 starts an
// ordered multi-WR PUT. payload_len remains the logical object size; each
// WRITE_WITH_IMM completion contributes one bounded payload window. The server
// publishes the object only after exactly `length` windows and payload_len
// bytes have arrived.
constexpr uint64_t kV2MultiWrPutMagic = 0x325257495455504dull;  // "MPUTIWR2"

constexpr const char* kV2ProbeDevice = "__dfkv_v2__";
constexpr uint32_t kV2ProbeMagic = 0x33564644u;  // ASCII "DFV3" (LE)
constexpr size_t kV2ProbeReplyBytes = 8;
// Probe reply byte 5 was reserved in the original 8-byte reply. Advertising
// here is backward-compatible: old clients validate only magic/version, while
// new clients fail before real QP bootstrap unless this bit is present.
constexpr uint8_t kV2ProbeCapWriterRetirement = 1u << 0;
constexpr uint8_t kV2ProbeCapPullRead = 1u << 1;
// Bit 2 advertises the per-op staging-lease PUT datapath (kLeasePut). Old
// clients ignore it; new clients refuse to send lease requests without it and
// fall back to the connection-resident receive-slot path.
constexpr uint8_t kV2ProbeCapLeasedPut = 1u << 2;

inline bool IsV2Probe(const char frame[kDevNameBytes]) {
  size_t n = 0;
  while (n < kDevNameBytes && frame[n] != '\0') ++n;
  return std::string(frame, n) == kV2ProbeDevice &&
         ParseDevFrameProtocol(frame) == kDevProtoV2;
}

inline void EncodeV2ProbeReply(
    char out[kV2ProbeReplyBytes],
    uint8_t capabilities = kV2ProbeCapWriterRetirement |
                           kV2ProbeCapPullRead | kV2ProbeCapLeasedPut) {
  std::memset(out, 0, kV2ProbeReplyBytes);
  std::memcpy(out, &kV2ProbeMagic, sizeof(kV2ProbeMagic));
  out[4] = static_cast<char>(kDevProtoV2);
  out[5] = static_cast<char>(capabilities);
}

inline bool ParseV2ProbeReply(const char in[kV2ProbeReplyBytes]) {
  uint32_t magic = 0;
  std::memcpy(&magic, in, sizeof(magic));
  return magic == kV2ProbeMagic &&
         static_cast<uint8_t>(in[4]) == kDevProtoV2;
}

inline uint8_t ParseV2ProbeCapabilities(
    const char in[kV2ProbeReplyBytes]) {
  return ParseV2ProbeReply(in) ? static_cast<uint8_t>(in[5]) : 0;
}

inline bool V2ProbeSupportsWriterRetirement(
    const char in[kV2ProbeReplyBytes]) {
  return (ParseV2ProbeCapabilities(in) &
          kV2ProbeCapWriterRetirement) != 0;
}

inline bool V2ProbeSupportsPullRead(
    const char in[kV2ProbeReplyBytes]) {
  return (ParseV2ProbeCapabilities(in) & kV2ProbeCapPullRead) != 0;
}

inline bool V2ProbeSupportsLeasedPut(
    const char in[kV2ProbeReplyBytes]) {
  return (ParseV2ProbeCapabilities(in) & kV2ProbeCapLeasedPut) != 0;
}

// Failure-path writer retirement. The client reconnects with the opaque
// per-QP writer token. The responder cancels later PostWrite calls, transitions
// that QP to ERR, drains every signaled WRITE CQE, and only then echoes proof.
// The proof is the protocol fence that permits retry/return with the same
// caller/CUDA destination; QP destruction and MR teardown are not fences.
constexpr const char* kV2RetireWriterDevice = "__dfkv_retire__";
constexpr uint64_t kV2RetireProofMagic = 0x32524657564b4644ull;  // "DFKVWFR2"
constexpr size_t kV2WriterTokenBytes = 8;
constexpr size_t kV2RetireProofBytes = 16;

inline bool IsV2RetireWriter(const char frame[kDevNameBytes]) {
  size_t n = 0;
  while (n < kDevNameBytes && frame[n] != '\0') ++n;
  return std::string(frame, n) == kV2RetireWriterDevice &&
         ParseDevFrameProtocol(frame) == kDevProtoV2 &&
         ParseDevFrameCaps(frame) != 0;
}

inline void EncodeV2RetireProof(
    uint64_t token, char out[kV2RetireProofBytes]) {
  net::PutU64(out, kV2RetireProofMagic);
  net::PutU64(out + 8, token);
}

inline bool ParseV2RetireProof(
    const char in[kV2RetireProofBytes], uint64_t token) {
  return net::GetU64(in) == kV2RetireProofMagic &&
         net::GetU64(in + 8) == token;
}

// Multi-window GET IDs name one of the negotiated per-connection logical
// request slots. Keeping the namespace bounded by queue depth both caps server
// state and makes duplicate ownership explicit.
inline bool V2GetOperationIdValid(uint32_t operation_id, size_t depth) {
  return depth != 0 && operation_id < depth;
}

inline size_t AlignUp(size_t value, size_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0 ||
      value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
    return 0;
  }
  return (value + alignment - 1) & ~(alignment - 1);
}

inline size_t V2SlotSize(uint64_t max_block_bytes) {
  if (max_block_bytes >
      std::numeric_limits<size_t>::max() - kV2DataOffset) {
    return 0;
  }
  return AlignUp(kV2DataOffset + static_cast<size_t>(max_block_bytes),
                 kV2DataOffset);
}

// A server is not ready unless its process-wide segment can accept at least
// one depth-1 connection at the maximum block size it advertises. Starting
// with less capacity turns every valid data handshake into a late rejection.
inline bool V2RecvSegmentFitsOneSlot(size_t segment_bytes,
                                     uint64_t max_block_bytes) {
  const size_t slot_size = V2SlotSize(max_block_bytes);
  return slot_size != 0 && segment_bytes >= slot_size;
}

// WRITE_WITH_IMM completion length is the only proof that the peer overwrote
// the whole v2 PUT frame in its leased slot. Require an exact frame; accepting a
// short write would expose stale bytes from the slot's previous generation.
inline bool V2PutCompletionCoversFrame(uint32_t completion_bytes,
                                       uint64_t payload_len,
                                       size_t slot_size) {
  if (slot_size < kV2PutPrefixOffset + kReqPrefix)
    return false;
  const size_t frame_cap = slot_size - kV2PutPrefixOffset;
  if (payload_len > frame_cap - kReqPrefix)
    return false;
  return completion_bytes == kReqPrefix + payload_len;
}

// SEND_WITH_IMM also sets IBV_WC_WITH_IMM at the receiver, but its opcode is
// ordinary RECV and its bytes live in the posted receive buffer, not the shared
// segment. Accept only an RDMA_WRITE_WITH_IMM receive completion.
inline bool V2PutCompletionIsValid(bool is_rdma_write_with_imm,
                                   bool has_immediate,
                                   uint32_t completion_bytes,
                                   uint64_t payload_len,
                                   uint64_t logical_payload_cap,
                                   size_t slot_size) {
  return is_rdma_write_with_imm && has_immediate &&
         payload_len <= logical_payload_cap &&
         V2PutCompletionCoversFrame(completion_bytes, payload_len, slot_size);
}

struct RecvSegmentInfo {
  uint64_t base_addr = 0;  // this connection's K-slot lease, not global base
  uint32_t rkey = 0;
  uint64_t slot_size = 0;
};

constexpr uint32_t kRecvSegmentInfoMagic = 0x32475352u;  // "RSG2" (LE)
constexpr size_t kRecvSegmentInfoBytes = 24;

inline void EncodeRecvSegmentInfo(const RecvSegmentInfo& info,
                                  char out[kRecvSegmentInfoBytes]) {
  std::memset(out, 0, kRecvSegmentInfoBytes);
  net::PutU32(out, kRecvSegmentInfoMagic);
  net::PutU32(out + 4, info.rkey);
  net::PutU64(out + 8, info.base_addr);
  net::PutU64(out + 16, info.slot_size);
}

inline bool DecodeRecvSegmentInfo(const char in[kRecvSegmentInfoBytes],
                                  RecvSegmentInfo* info) {
  if (net::GetU32(in) != kRecvSegmentInfoMagic) return false;
  info->rkey = net::GetU32(in + 4);
  info->base_addr = net::GetU64(in + 8);
  info->slot_size = net::GetU64(in + 16);
  return info->rkey != 0 && info->base_addr != 0 &&
         info->slot_size >= kV2DataOffset &&
         info->slot_size % kV2DataOffset == 0 &&
         info->base_addr <=
             std::numeric_limits<uint64_t>::max() - (info->slot_size - 1);
}

struct PullArenaInfo {
  uint64_t base_addr = 0;
  uint64_t arena_bytes = 0;
  uint64_t connection_generation = 0;
  uint32_t rkey = 0;
  uint32_t slot_count = 0;
};

constexpr uint32_t kPullArenaInfoMagic = 0x324c5550u;  // "PUL2" (LE)
constexpr size_t kPullArenaInfoBytes = 40;

inline void EncodePullArenaInfo(const PullArenaInfo& info,
                                char out[kPullArenaInfoBytes]) {
  std::memset(out, 0, kPullArenaInfoBytes);
  net::PutU32(out, kPullArenaInfoMagic);
  net::PutU32(out + 4, info.rkey);
  net::PutU64(out + 8, info.base_addr);
  net::PutU64(out + 16, info.arena_bytes);
  net::PutU64(out + 24, info.connection_generation);
  net::PutU32(out + 32, info.slot_count);
}

inline bool DecodePullArenaInfo(const char in[kPullArenaInfoBytes],
                                PullArenaInfo* info) {
  if (net::GetU32(in) != kPullArenaInfoMagic) return false;
  info->rkey = net::GetU32(in + 4);
  info->base_addr = net::GetU64(in + 8);
  info->arena_bytes = net::GetU64(in + 16);
  info->connection_generation = net::GetU64(in + 24);
  info->slot_count = net::GetU32(in + 32);
  return info->rkey != 0 && info->base_addr != 0 &&
         info->arena_bytes != 0 && info->connection_generation != 0 &&
         info->slot_count != 0;
}

struct PullPrepareControl {
  uint32_t slot_index = 0;
  uint64_t release_generation = 0;
};

constexpr uint32_t kPullPrepareMagic = 0x32525050u;  // "PPR2" (LE)
constexpr size_t kPullPrepareBytes = 16;

inline void EncodePullPrepareControl(
    const PullPrepareControl& control, char out[kPullPrepareBytes]) {
  std::memset(out, 0, kPullPrepareBytes);
  net::PutU32(out, kPullPrepareMagic);
  net::PutU32(out + 4, control.slot_index);
  net::PutU64(out + 8, control.release_generation);
}

inline bool DecodePullPrepareControl(
    const char in[kPullPrepareBytes], PullPrepareControl* control) {
  if (net::GetU32(in) != kPullPrepareMagic) return false;
  control->slot_index = net::GetU32(in + 4);
  control->release_generation = net::GetU64(in + 8);
  return true;
}

struct PullReady {
  uint32_t slot_index = 0;
  uint64_t slot_generation = 0;
  uint64_t data_len = 0;
  uint64_t value_len = 0;
};

constexpr uint32_t kPullReadyMagic = 0x32594452u;  // "RDY2" (LE)
constexpr size_t kPullReadyBytes = 40;

inline void EncodePullReady(const PullReady& ready,
                            char out[kPullReadyBytes]) {
  std::memset(out, 0, kPullReadyBytes);
  net::PutU32(out, kPullReadyMagic);
  net::PutU32(out + 4, ready.slot_index);
  net::PutU64(out + 8, ready.slot_generation);
  net::PutU64(out + 16, ready.data_len);
  net::PutU64(out + 24, ready.value_len);
}

inline bool DecodePullReady(const char in[kPullReadyBytes],
                            PullReady* ready) {
  if (net::GetU32(in) != kPullReadyMagic) return false;
  ready->slot_index = net::GetU32(in + 4);
  ready->slot_generation = net::GetU64(in + 8);
  ready->data_len = net::GetU64(in + 16);
  ready->value_len = net::GetU64(in + 24);
  return ready->slot_generation != 0;
}

// Per-op staging lease for the in-flight leased-PUT datapath. The server
// leases exactly one receive-pool range sized for the whole object; the client
// WRITEs the object into [write_base, write_base+lease_bytes) exactly like an
// ordinary receive-slot window (frame prefix at +kV2PutPrefixOffset, payload
// at +kV2DataOffset). The range is freed by the server as soon as the STORE
// handler returns, so receive memory follows data in flight, not connection
// count. `slot` echoes the request's recv slot (also the WRITE immediate) and
// `generation` is a nonzero per-slot monotonically increasing value reserved
// for future release/retry extensions and log correlation.
struct LeasePutReady {
  uint32_t slot = 0;
  uint64_t generation = 0;
  uint32_t rkey = 0;
  uint64_t write_base = 0;
  uint64_t lease_bytes = 0;
};

constexpr uint32_t kLeasePutReadyMagic = 0x3252534cu;  // "LSR2" (LE)
constexpr size_t kLeasePutReadyBytes = 40;

inline void EncodeLeasePutReady(const LeasePutReady& ready,
                                char out[kLeasePutReadyBytes]) {
  std::memset(out, 0, kLeasePutReadyBytes);
  net::PutU32(out, kLeasePutReadyMagic);
  net::PutU32(out + 4, ready.rkey);
  net::PutU32(out + 8, ready.slot);
  net::PutU32(out + 12, 0);
  net::PutU64(out + 16, ready.write_base);
  net::PutU64(out + 24, ready.lease_bytes);
  net::PutU64(out + 32, ready.generation);
}

inline bool DecodeLeasePutReady(const char in[kLeasePutReadyBytes],
                                LeasePutReady* ready) {
  if (net::GetU32(in) != kLeasePutReadyMagic) return false;
  const uint32_t rkey = net::GetU32(in + 4);
  const uint64_t write_base = net::GetU64(in + 16);
  const uint64_t lease_bytes = net::GetU64(in + 24);
  // A lease always stages at least one payload byte after the frame page, so
  // a single empty page is geometry no server can produce: reject it here.
  if (rkey == 0 || write_base == 0 ||
      lease_bytes < 2 * kV2DataOffset ||
      lease_bytes % kV2DataOffset != 0 ||
      write_base >
          std::numeric_limits<uint64_t>::max() - (lease_bytes - 1))
    return false;
  ready->rkey = rkey;
  ready->slot = net::GetU32(in + 8);
  ready->write_base = write_base;
  ready->lease_bytes = lease_bytes;
  ready->generation = net::GetU64(in + 32);
  return ready->generation != 0;
}

// The original readiness response is exactly one ready byte plus the 24-byte
// receive-segment descriptor. A negotiated writer-retirement connection
// appends its nonzero token; an unnegotiated client must see no extra bytes.
constexpr size_t kV2LegacyReadinessBytes = 1 + kRecvSegmentInfoBytes;
constexpr size_t kV2RetirementReadinessBytes =
    kV2LegacyReadinessBytes + kV2WriterTokenBytes;
static_assert(kV2LegacyReadinessBytes == 25);

inline size_t V2ReadinessBytes(bool writer_retirement_negotiated) {
  return writer_retirement_negotiated ? kV2RetirementReadinessBytes
                                      : kV2LegacyReadinessBytes;
}

inline size_t EncodeV2Readiness(
    const RecvSegmentInfo& info, uint64_t writer_token,
    char out[kV2RetirementReadinessBytes]) {
  std::memset(out, 0, kV2RetirementReadinessBytes);
  out[0] = 1;
  EncodeRecvSegmentInfo(info, out + 1);
  if (writer_token == 0) return kV2LegacyReadinessBytes;
  net::PutU64(out + kV2LegacyReadinessBytes, writer_token);
  return kV2RetirementReadinessBytes;
}

constexpr size_t kV2PullReadinessBytes =
    kV2RetirementReadinessBytes + kPullArenaInfoBytes;

inline size_t EncodeV2PullReadiness(
    const RecvSegmentInfo& recv_info, uint64_t writer_token,
    const PullArenaInfo& pull_info, char out[kV2PullReadinessBytes]) {
  std::memset(out, 0, kV2PullReadinessBytes);
  const size_t base = EncodeV2Readiness(recv_info, writer_token, out);
  if (base != kV2RetirementReadinessBytes) return 0;
  EncodePullArenaInfo(pull_info, out + base);
  return kV2PullReadinessBytes;
}


inline bool DecodeV2Readiness(
    const char* in, size_t bytes, bool writer_retirement_negotiated,
    RecvSegmentInfo* info, uint64_t* writer_token) {
  if (bytes != V2ReadinessBytes(writer_retirement_negotiated) || in[0] != 1 ||
      !DecodeRecvSegmentInfo(in + 1, info)) {
    return false;
  }
  *writer_token =
      writer_retirement_negotiated
          ? net::GetU64(in + kV2LegacyReadinessBytes)
          : 0;
  return !writer_retirement_negotiated || *writer_token != 0;
}
inline bool DecodeV2PullReadiness(
    const char* in, size_t bytes, RecvSegmentInfo* recv_info,
    uint64_t* writer_token, PullArenaInfo* pull_info) {
  return bytes == kV2PullReadinessBytes &&
         DecodeV2Readiness(in, kV2RetirementReadinessBytes, true, recv_info,
                           writer_token) &&
         DecodePullArenaInfo(in + kV2RetirementReadinessBytes, pull_info);
}

}  // namespace dfkv::rdma

#endif  // DFKV_TRANSPORT_RDMA_PROTOCOL_H_
