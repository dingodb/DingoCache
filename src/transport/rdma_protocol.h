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
constexpr uint32_t kV2ProbeMagic = 0x32564644u;  // ASCII "DFV2" (LE)
constexpr size_t kV2ProbeReplyBytes = 8;

inline bool IsV2Probe(const char frame[kDevNameBytes]) {
  size_t n = 0;
  while (n < kDevNameBytes && frame[n] != '\0') ++n;
  return std::string(frame, n) == kV2ProbeDevice &&
         ParseDevFrameProtocol(frame) == kDevProtoV2;
}

inline void EncodeV2ProbeReply(char out[kV2ProbeReplyBytes]) {
  std::memset(out, 0, kV2ProbeReplyBytes);
  std::memcpy(out, &kV2ProbeMagic, sizeof(kV2ProbeMagic));
  out[4] = static_cast<char>(kDevProtoV2);
}

inline bool ParseV2ProbeReply(const char in[kV2ProbeReplyBytes]) {
  uint32_t magic = 0;
  std::memcpy(&magic, in, sizeof(magic));
  return magic == kV2ProbeMagic &&
         static_cast<uint8_t>(in[4]) == kDevProtoV2;
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

}  // namespace dfkv::rdma

#endif  // DFKV_TRANSPORT_RDMA_PROTOCOL_H_
