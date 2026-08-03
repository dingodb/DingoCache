/* RDMA v2 bootstrap device frame.
 *
 * [name | NUL | "DCP2" u32 | max_block_bytes u64 | protocol u8]
 *
 * The declaration and protocol marker are mandatory. Header-only and
 * verbs-free so protocol rejection is unit-testable without RDMA hardware.
 */
#ifndef DFKV_TRANSPORT_DEV_FRAME_H_
#define DFKV_TRANSPORT_DEV_FRAME_H_

#include <cstdint>
#include <cstring>
#include <string>

namespace dfkv::rdma {

constexpr size_t kDevNameBytes = 32;
constexpr uint32_t kDevCapsV2Magic = 0x32504344u;  // ASCII "DCP2" (LE)
constexpr uint8_t kDevProtoV2 = 2;

// Room a device name must leave in the fixed 32-byte frame: NUL terminator +
// "DCP2" u32 + max_block_bytes u64 + protocol u8. A name at most this long
// encodes whole; a longer one would silently lose the capability trailer
// (EncodeDevFrame drops it as deep defense), after which the peer rejects
// every real connection as "missing v2 negotiation" while short-named
// capability probes keep passing. Configuration must instead fail fast on
// such names — DeviceNameFitsFrame / OversizedDeviceNameError below are the
// shared checks used by the client whitelist and the server anchors.
constexpr size_t kDevFrameCapsTrailerBytes = 1 + 4 + 8 + 1;
constexpr size_t kMaxDeviceNameBytes = kDevNameBytes - kDevFrameCapsTrailerBytes;

// True when dev can be announced inside the bootstrap frame without losing
// the capability trailer. An empty name is legal: it asks the peer for its
// default device.
inline bool DeviceNameFitsFrame(const std::string& dev) {
  return dev.size() <= kMaxDeviceNameBytes;
}

// Shared fail-fast diagnostic naming both the offending device and the real
// limit — not the misleading "missing v2 negotiation" the peer would report.
inline std::string OversizedDeviceNameError(const std::string& dev) {
  return "rdma: device name \"" + dev + "\" is " + std::to_string(dev.size()) +
         " bytes, over the v2 bootstrap frame limit of " +
         std::to_string(kMaxDeviceNameBytes) +
         " bytes; encode would silently drop the \"DCP2\" capability trailer. "
         "Rename the device (udev/bond alias) or configure a shorter name";
}

inline void EncodeDevFrame(const std::string& dev, uint64_t max_block_bytes,
                           char out[kDevNameBytes],
                           uint8_t protocol_version = kDevProtoV2) {
  std::memset(out, 0, kDevNameBytes);
  const size_t n = dev.size() < kDevNameBytes - 1 ? dev.size()
                                                  : kDevNameBytes - 1;
  std::memcpy(out, dev.data(), n);
  // Deep defense: names over kMaxDeviceNameBytes no longer arrive here —
  // client whitelist and server anchor validation reject them at startup —
  // but keep the silent trailer drop so any future unvalidated caller still
  // fails closed (peer rejects) rather than negotiating phantom capabilities.
  if (protocol_version != kDevProtoV2 || max_block_bytes == 0 ||
      n > kMaxDeviceNameBytes)
    return;
  std::memcpy(out + n + 1, &kDevCapsV2Magic, 4);
  std::memcpy(out + n + 5, &max_block_bytes, 8);
  out[n + 13] = static_cast<char>(kDevProtoV2);
}

inline uint64_t ParseDevFrameCaps(const char in[kDevNameBytes]) {
  size_t nul = 0;
  while (nul < kDevNameBytes && in[nul] != '\0') ++nul;
  if (nul + 1 + 4 + 8 + 1 > kDevNameBytes) return 0;
  uint32_t magic = 0;
  std::memcpy(&magic, in + nul + 1, 4);
  if (magic != kDevCapsV2Magic ||
      static_cast<uint8_t>(in[nul + 13]) != kDevProtoV2)
    return 0;
  uint64_t value = 0;
  std::memcpy(&value, in + nul + 5, 8);
  return value;
}

inline uint8_t ParseDevFrameProtocol(const char in[kDevNameBytes]) {
  return ParseDevFrameCaps(in) == 0 ? 0 : kDevProtoV2;
}

}  // namespace dfkv::rdma

#endif  // DFKV_TRANSPORT_DEV_FRAME_H_
