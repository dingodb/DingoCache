/* Environment settings of the RDMA server whose reading is worth a test of
 * its own (see utils/env_parse.h for the rule: the value a text resolves to
 * does not change, an unclean text is reported). Header-only and free of
 * verbs, so the tests build without RDMA. */
#ifndef DFKV_CACHE_RDMA_SERVER_ENV_H_
#define DFKV_CACHE_RDMA_SERVER_ENV_H_

#include <algorithm>
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

// DFKV_RDMA_GRANT_TIMEOUT_MS: how long a GET grant (a pulled value the
// client has not released) or a PUT lease (a staging range the client has not
// written) may wait on the client before the server tears the connection
// down; -1 = no bound. Default 60 s, far above any client deadline. 0
// disables. Clamped to 24 h.
inline int RdmaGrantTimeoutMsFromEnv() {
  envparse::U64Spec spec;
  spec.fallback = 60000;
  spec.max = 86400000;
  spec.above = envparse::Above::kClamp;
  spec.zero_means = "grants and leases wait for the client without bound";
  const uint64_t v = envparse::EnvU64("DFKV_RDMA_GRANT_TIMEOUT_MS", spec);
  return v == 0 ? -1 : static_cast<int>(v);
}

// DFKV_RDMA_STOP_DRAIN_MS: how long Stop() lets connections that hold a
// grant (a GET's READY, a PUT lease) finish before closing them; -1 = do not
// wait. Default 2 s. 0 disables. Clamped to 60 s.
inline int RdmaStopDrainMsFromEnv() {
  envparse::U64Spec spec;
  spec.fallback = 2000;
  spec.max = 60000;
  spec.above = envparse::Above::kClamp;
  spec.zero_means = "Stop() closes connections at once";
  const uint64_t v = envparse::EnvU64("DFKV_RDMA_STOP_DRAIN_MS", spec);
  return v == 0 ? -1 : static_cast<int>(v);
}

// Server memory windows (DFKV_RDMA_SERVER_MW, off by default; see
// docs/design/server-memory-windows.md). The quarantine floor can only be
// raised: below 5 ms (or not a number) is the default 5 ms. The longest
// quarantine a grant may need before it takes a per-grant MR instead:
// default 50 ms, at least the floor.
struct RdmaServerMwSettings {
  bool enabled = false;
  uint64_t floor_ms = 5;
  uint64_t max_ms = 50;
  uint64_t quarantine_bytes = 0;  // 0: an eighth of the receive pool
  uint64_t arena_bytes = 0;       // per rail; 0: the whole arena
  uint64_t link_gbps = 0;         // lower bound on the link rate; 0: ports
};
inline RdmaServerMwSettings RdmaServerMwFromEnv() {
  RdmaServerMwSettings out;
  out.enabled = envparse::EnvSwitch("DFKV_RDMA_SERVER_MW", false,
                                    envparse::OffWhen::kOffWord);
  envparse::U64Spec floor_spec;
  floor_spec.fallback = 5;
  floor_spec.min = 5;
  floor_spec.max = 1000;
  floor_spec.above = envparse::Above::kClamp;
  out.floor_ms =
      envparse::EnvU64("DFKV_RDMA_SERVER_MW_QUARANTINE_MS", floor_spec);
  envparse::U64Spec max_spec;
  max_spec.fallback = 50;
  max_spec.min = 1;
  max_spec.max = 10000;
  max_spec.above = envparse::Above::kClamp;
  out.max_ms = std::max(
      out.floor_ms,
      envparse::EnvU64("DFKV_RDMA_SERVER_MW_QUARANTINE_MAX_MS", max_spec));
  envparse::U64Spec bytes_spec;  // 0 = default, said by the caller
  out.quarantine_bytes =
      envparse::EnvU64("DFKV_RDMA_SERVER_MW_QUARANTINE_BYTES", bytes_spec);
  out.arena_bytes =
      envparse::EnvU64("DFKV_RDMA_SERVER_MW_ARENA_BYTES", bytes_spec);
  out.link_gbps = envparse::EnvU64("DFKV_RDMA_SERVER_MW_LINK_GBPS", bytes_spec);
  return out;
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
