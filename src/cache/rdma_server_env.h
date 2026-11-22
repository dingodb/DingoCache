/* Environment settings of the RDMA server whose reading is worth a test of
 * its own (see utils/env_parse.h for the rule: the value a text resolves to
 * does not change, an unclean text is reported). Header-only and free of
 * verbs, so the tests build without RDMA. */
#ifndef DFKV_CACHE_RDMA_SERVER_ENV_H_
#define DFKV_CACHE_RDMA_SERVER_ENV_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "transport/rdma_protocol.h"
#include "transport/rdma_recv_segment.h"
#include "utils/env_parse.h"

namespace dfkv {

// DFKV_SERVER_URING: on unless the value is exactly "0". Read on every call,
// as it always was.
inline bool ServerUringFromEnv() {
  return envparse::EnvSwitch("DFKV_SERVER_URING", true,
                             envparse::OffWhen::kExactlyZero);
}

// DFKV_RDMA_IDLE_MS: per-connection idle timeout, ms; -1 = no reaper. 0
// disables, and so does everything the old strtol read as <= 0: a negative
// number and a value that is not a number at all. Clamped to 24 h.
inline int RdmaIdleMsFromEnv() {
  envparse::U64Spec spec;
  spec.fallback = 600000;  // 10 min
  spec.max = 86400000;     // 24 h
  spec.above = envparse::Above::kClamp;
  spec.legacy = envparse::Legacy::kPrefixOrZero;
  spec.negative = envparse::Negative::kZero;
  spec.representable_max = envparse::kSignedMax;  // read with strtol
  spec.zero_means = "the idle reaper is off";
  const uint64_t v = envparse::EnvU64("DFKV_RDMA_IDLE_MS", spec);
  return v == 0 ? -1 : static_cast<int>(v);
}

// A receive-pool byte count. rdma::ResolveRecvSegmentBytes does the parsing,
// as before (full-string strtoull; 0, garbage and overflow give `fallback`;
// the result is aligned up, or 0 if that does not fit). This adds the report
// of a value that is not a plain number, with the number really used.
inline size_t RecvBytesFromEnv(const char* name, size_t fallback) {
  const char* text = std::getenv(name);
  const size_t bytes =
      rdma::ResolveRecvSegmentBytes(text, fallback, rdma::kV2DataOffset);
  envparse::U64Spec clean;  // only asked whether the text is clean
  clean.min = 1;
  std::string unclean;
  (void)envparse::ParseU64(name, text, clean, &unclean);
  if (!unclean.empty())
    envparse::WarnOnce(
        name, "invalid " + envparse::detail::Quoted(name, text) +
                  ": only a plain decimal number of bytes is valid; using " +
                  std::to_string(bytes) + ", as before");
  return bytes;
}

}  // namespace dfkv

#endif  // DFKV_CACHE_RDMA_SERVER_ENV_H_
