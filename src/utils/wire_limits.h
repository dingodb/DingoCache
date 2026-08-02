/* Shared wire payload limits. Every socket that reads a length prefix MUST
 * pass a real bound into DecodeReq/DecodeResp instead of the 16 GiB
 * kMaxFrameLen default: the servers allocate payload_len BEFORE reading the
 * payload bytes, so an unbounded declared length is a one-frame OOM DoS.
 *
 * ResolveMaxPayload is the single source of the server's max VALUE size (env
 * overridable), previously private to rdma_server.cc; the TCP request path
 * and the RDMA path must agree on it or a value accepted by one transport
 * would be rejected by the other. */
#ifndef DFKV_WIRE_LIMITS_H_
#define DFKV_WIRE_LIMITS_H_

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>


namespace dfkv {
namespace wire_limits {

// O_DIRECT / registered-buffer alignment (mirrors rdma::kDirectIoAlign; kept
// as a plain constant here so this header stays free of RDMA-gated includes).
constexpr size_t kIoAlign = 4096;

// dbuf/SGE length is uint32, so a payload must stay under
// Keep room for direct-I/O alignment rounding inside uint32_t SGE lengths.
constexpr size_t kPayloadHardCap = static_cast<size_t>(
    std::numeric_limits<uint32_t>::max() - 2 * kIoAlign);

inline size_t EnvBytes(const char* name, size_t dflt) {
  const char* v = std::getenv(name);
  if (!v || !*v) return dflt;
  errno = 0;
  char* end = nullptr;
  unsigned long long x = std::strtoull(v, &end, 10);
  if (errno != 0 || end == v || x == 0) return dflt;
  if (x > kPayloadHardCap) x = kPayloadHardCap;
  return static_cast<size_t>(x);
}

// The server's max value payload: configured (0 = 64 MiB default), then the
// DFKV_RDMA_MAX_PAYLOAD_BYTES override, clamped to the uint32 hard cap. Same
// math the RDMA server applies; a PUT larger than this is rejected on every
// transport.
inline size_t ResolveMaxPayload(size_t configured) {
  size_t n = configured ? configured : (64u << 20);
  n = EnvBytes("DFKV_RDMA_MAX_PAYLOAD_BYTES", n);
  if (n > kPayloadHardCap) n = kPayloadHardCap;
  return n;
}

// Bound for the raw value bytes carried by a PUT request.
inline uint64_t MaxRequestPayload(size_t configured_max_value = 0) {
  return static_cast<uint64_t>(ResolveMaxPayload(configured_max_value));
}

// MDS request frames carry a group name + one MemberInfo (well under 1 KiB);
// list responses carry the member table (a few KiB for a 62-node ring). The
// caps leave two orders of magnitude of headroom.
constexpr uint64_t kMdsMaxReqPayload = 1u << 20;   // 1 MiB
constexpr uint64_t kMdsMaxRespData = 1u << 24;     // 16 MiB

// Stats/Members text responses on the data-path TCP port.
constexpr uint64_t kTextMaxRespData = 1u << 24;    // 16 MiB
// Status-only responses (Cache/Exist/Remove) carry no data; allow a tiny
// slack instead of 0 so a future short diagnostic string can't break us.
constexpr uint64_t kStatusMaxRespData = 4096;

}  // namespace wire_limits
}  // namespace dfkv

#endif  // DFKV_WIRE_LIMITS_H_
