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
  kInvalid
};

inline const char* StatusName(Status s) {
  switch (s) {
    case Status::kOk: return "Ok";
    case Status::kNotFound: return "NotFound";
    case Status::kCacheFull: return "CacheFull";
    case Status::kQuotaExceeded: return "QuotaExceeded";
    case Status::kIOError: return "IOError";
    case Status::kInvalid: return "Invalid";
  }
  return "?";
}

}  // namespace dfkv

#endif  // DFKV_STATUS_H_
