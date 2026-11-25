/* rdma_settings — how the RDMA client reads its numeric DFKV_RDMA_* settings.
 *
 * Each reader is the envparse::ParseU64 spec of the reader the transport used
 * to have of its own (TRN-30): the value a text resolves to is what it always
 * was, and the warning is the one it always gave (ReportUnlessPlain, from the
 * resolved value). One exception, a fix: ReadInt gives the default for a
 * number above INT_MAX, which the old reader cast to a negative or zero int.
 * test/transport/rdma_settings_test.cc compares them with the old readers.
 * Header-only, no verbs. */
#ifndef DFKV_TRANSPORT_RDMA_SETTINGS_H_
#define DFKV_TRANSPORT_RDMA_SETTINGS_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>

#include "utils/env_parse.h"

namespace dfkv {
namespace rdma_settings {

namespace detail {
inline uint64_t Read(const char* name, const envparse::U64Spec& spec) {
  std::string discarded;  // the warning is the old one, below
  return envparse::ParseU64(name, std::getenv(name), spec, &discarded);
}
}  // namespace detail

// A positive int; anything else, and above INT_MAX, is the default. Read as
// strtol without an end check did: "30s" is 30.
inline int ReadInt(const char* name, int dflt) {
  envparse::U64Spec spec;
  spec.fallback = static_cast<uint64_t>(dflt);
  spec.min = 1;
  spec.max = static_cast<uint64_t>(std::numeric_limits<int>::max());
  spec.above = envparse::Above::kDefault;
  spec.legacy = envparse::Legacy::kPrefix;
  spec.negative = envparse::Negative::kInvalid;
  spec.representable_max = envparse::kSignedMax;
  const int resolved = static_cast<int>(detail::Read(name, spec));
  envparse::ReportUnlessPlain(name, 1, std::numeric_limits<int>::max(),
                              std::to_string(resolved));
  return resolved;
}

// A whole positive number, clamped to `upper`; anything else is the default.
// A default of 0 means "not set" to the caller, so 0 is a clean value then.
inline int ReadBoundedInt(const char* name, int dflt, int upper) {
  envparse::U64Spec spec;
  spec.fallback = static_cast<uint64_t>(dflt);
  spec.min = dflt == 0 ? 0 : 1;
  spec.max = static_cast<uint64_t>(upper);
  spec.above = envparse::Above::kClamp;
  spec.legacy = envparse::Legacy::kFull;
  spec.negative = envparse::Negative::kInvalid;
  spec.representable_max = envparse::kSignedMax;
  spec.overflow_is_default = true;
  const int resolved = static_cast<int>(detail::Read(name, spec));
  envparse::ReportUnlessPlain(name, dflt == 0 ? 0 : 1,
                              static_cast<uint64_t>(upper),
                              std::to_string(resolved));
  return resolved;
}

// A positive byte count, clamped to `max`; read as strtoull without an end
// check did ("64M" is 64; a negative number wraps and is clamped).
inline size_t ReadBytes(const char* name, size_t dflt, uint64_t max) {
  envparse::U64Spec spec;
  spec.fallback = dflt;
  spec.min = 1;
  spec.max = max;
  spec.above = envparse::Above::kClamp;
  spec.legacy = envparse::Legacy::kPrefix;
  spec.negative = envparse::Negative::kWraps;
  spec.overflow_is_default = true;
  const size_t resolved = static_cast<size_t>(detail::Read(name, spec));
  envparse::ReportUnlessPlain(name, 1, max, std::to_string(resolved));
  return resolved;
}

// A whole positive number with no upper bound (a negative one wraps).
inline uint64_t ReadU64(const char* name, uint64_t dflt) {
  envparse::U64Spec spec;
  spec.fallback = dflt;
  spec.min = 1;
  spec.legacy = envparse::Legacy::kFull;
  spec.negative = envparse::Negative::kWraps;
  spec.overflow_is_default = true;
  const uint64_t resolved = detail::Read(name, spec);
  envparse::ReportUnlessPlain(name, 1, std::numeric_limits<uint64_t>::max(),
                              std::to_string(resolved));
  return resolved;
}

}  // namespace rdma_settings
}  // namespace dfkv

#endif  // DFKV_TRANSPORT_RDMA_SETTINGS_H_
