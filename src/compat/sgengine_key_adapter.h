/* SGEngine v1 compatibility key adapter.
 *
 * The legacy wire tuple is preserved exactly, but tagged at the frontend
 * boundary with an internal domain. Native clients never set this domain, so
 * legacy and native requests cannot address the same RAM, file, or slab entry.
 */
#ifndef DFKV_SGENGINE_KEY_ADAPTER_H_
#define DFKV_SGENGINE_KEY_ADAPTER_H_

#include <cstdint>

#include "common/kv_types.h"

namespace dfkv::compat {

inline BlockKey SgEngineV1Key(uint64_t id, uint32_t index, uint32_t size) {
  // This is a bijection over the complete legacy tuple, not a lossy re-hash.
  return BlockKey{id, (static_cast<uint64_t>(size) << 32) | index,
                  KeyDomain::kSgEngineV1};
}

}  // namespace dfkv::compat

#endif  // DFKV_SGENGINE_KEY_ADAPTER_H_
