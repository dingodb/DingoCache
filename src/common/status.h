/* Status — the shared result code for every dfkv operation (store, transport,
 * MDS, metrics). Lives in common/ so the transport and MDS layers don't have to
 * depend on the cache-node storage engine just to name an error. */
#ifndef DFKV_STATUS_H_
#define DFKV_STATUS_H_

namespace dfkv {

enum class Status {
  kOk,
  kNotFound,
  kCacheFull,
  kQuotaExceeded,
  kIOError,
  kInvalid,
  // Client-local only, NEVER encoded on the wire: the response decoder
  // (wire.h DecodeRespVersion) pins kInvalid as the highest legal wire byte.
  // kResourceExhausted means THIS process ran out of transport budget
  // (endpoints/QPs/WRs/registered bytes) before the peer was ever dialed.
  // Peer health must treat it as neither a served response nor a peer IO
  // failure: cooling down a peer for a local admission stall poisons every
  // key routed to it (see .issue/0812-004).
  kResourceExhausted,
  // Client-local only and peer-health-neutral. The selected peer topology has
  // no healthy device-name intersection with this client's configured rails.
  // This is decided before taking credits, process budget, or dialing a peer.
  kNoCompatibleRail
};

inline const char* StatusName(Status s) {
  switch (s) {
    case Status::kOk: return "Ok";
    case Status::kNotFound: return "NotFound";
    case Status::kCacheFull: return "CacheFull";
    case Status::kQuotaExceeded: return "QuotaExceeded";
    case Status::kIOError: return "IOError";
    case Status::kInvalid: return "Invalid";
    case Status::kResourceExhausted: return "ResourceExhausted";
    case Status::kNoCompatibleRail: return "NoCompatibleRail";
  }
  return "?";
}

}  // namespace dfkv

#endif  // DFKV_STATUS_H_
