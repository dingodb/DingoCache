/* RDMA bootstrap device-name frame with a backwards-compatible declaration
 * hidden after the NUL-terminated device name.  Legacy readers stop at the
 * first NUL and therefore ignore the extension:
 *
 *   DCP1: [name | \0 | "DCP1" u32 | max_block_bytes u64]
 *   DCP2: [name | \0 | "DCP2" u32 | max_block_bytes u64 | protocol u8]
 *
 * DCP1 preserves the original SEND/RECV protocol.  DCP2 requests the mixed
 * v2 data plane and declares the block bound needed to size its shared receive
 * segment slots.  A missing/zero declaration is always treated as v1.
 *
 * Header-only and verbs-free so the codec is unit-testable in non-RDMA builds;
 * rdma_verbs.h re-exports it for the transport/server. */
#ifndef DFKV_TRANSPORT_DEV_FRAME_H_
#define DFKV_TRANSPORT_DEV_FRAME_H_

#include <cstdint>
#include <cstring>
#include <string>

namespace dfkv::rdma {

constexpr size_t kDevNameBytes = 32;
constexpr uint32_t kDevCapsMagic = 0x31504344u;  // ASCII "DCP1" (LE)
constexpr uint32_t kDevCapsV2Magic = 0x32504344u;  // ASCII "DCP2" (LE)
constexpr uint8_t kDevProtoV1 = 1;
constexpr uint8_t kDevProtoV2 = 2;

// Writes name + an optional DCP1/DCP2 tail. max_block_bytes==0 or a name too
// long to leave the selected tail bytes produces a plain legacy frame.
inline void EncodeDevFrame(const std::string& dev, uint64_t max_block_bytes,
                           char out[kDevNameBytes],
                           uint8_t protocol_version = kDevProtoV1) {
  std::memset(out, 0, kDevNameBytes);
  const size_t n = dev.size() < kDevNameBytes - 1 ? dev.size() : kDevNameBytes - 1;
  std::memcpy(out, dev.data(), n);
  if (max_block_bytes == 0) return;
  if (protocol_version >= kDevProtoV2) {
    if (n + 1 + 4 + 8 + 1 > kDevNameBytes) return;
    std::memcpy(out + n + 1, &kDevCapsV2Magic, 4);
    std::memcpy(out + n + 5, &max_block_bytes, 8);
    out[n + 13] = static_cast<char>(kDevProtoV2);
    return;
  }
  if (n + 1 + 4 + 8 > kDevNameBytes) return;
  std::memcpy(out + n + 1, &kDevCapsMagic, 4);
  std::memcpy(out + n + 5, &max_block_bytes, 8);
}

// Returns the declared max block bytes, or 0 when absent/legacy/garbled.
inline uint64_t ParseDevFrameCaps(const char in[kDevNameBytes]) {
  size_t nul = 0;
  while (nul < kDevNameBytes && in[nul] != '\0') ++nul;
  if (nul + 1 + 4 + 8 > kDevNameBytes) return 0;
  uint32_t magic = 0;
  std::memcpy(&magic, in + nul + 1, 4);
  if (magic != kDevCapsMagic && magic != kDevCapsV2Magic) return 0;
  uint64_t v = 0;
  std::memcpy(&v, in + nul + 5, 8);
  return v;
}

inline uint8_t ParseDevFrameProtocol(const char in[kDevNameBytes]) {
  size_t nul = 0;
  while (nul < kDevNameBytes && in[nul] != '\0') ++nul;
  if (nul + 1 + 4 + 8 + 1 > kDevNameBytes) return kDevProtoV1;
  uint32_t magic = 0;
  std::memcpy(&magic, in + nul + 1, 4);
  if (magic != kDevCapsV2Magic ||
      static_cast<uint8_t>(in[nul + 13]) != kDevProtoV2 ||
      ParseDevFrameCaps(in) == 0) {
    return kDevProtoV1;
  }
  return kDevProtoV2;
}

}  // namespace dfkv::rdma

#endif  // DFKV_TRANSPORT_DEV_FRAME_H_
