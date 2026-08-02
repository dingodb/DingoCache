/* Native v2-only C ABI for the DingoFS KV client. Namespaces and object keys
 * are explicit length-delimited binary identities; values are opaque bytes.
 * Every pointed-to key is copied before its call returns. */
#ifndef DFKV_DFKV_C_API_H_
#define DFKV_DFKV_C_API_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* dfkv_client_t;

#define DFKV_CLIENT_ABI_VERSION_V2 2u
#define DFKV_CLIENT_OPT_REGISTER_WITH_MDS (1u << 0)

// Versioned, size-delimited client construction contract. All pointed-to data
// is copied before dfkv_open_v2 returns. A zero/NULL optional field selects the
// documented default. Unknown flags, a short struct, or a non-v2 version fail
// closed and return NULL.
typedef struct dfkv_client_options_v2 {
  uint32_t struct_size;             // set to sizeof(dfkv_client_options_v2)
  uint32_t abi_version;             // DFKV_CLIENT_ABI_VERSION_V2
  const char* members;              // optional static "name=ip:port,..." list
  const void* key_namespace;        // required length-delimited binary identity
  uint64_t key_namespace_len;       // required; namespace may contain NUL bytes
  uint64_t batch_concurrency;       // 0 = automatic
  const char* mds_endpoints;        // optional comma-separated discovery seeds
  const char* mds_group;            // default "default"
  int32_t mds_poll_ms;              // <=0 => 3000
  uint32_t flags;                   // DFKV_CLIENT_OPT_*
  const char* client_id;            // required with REGISTER_WITH_MDS
  const char* client_info;          // optional registration metadata
  int32_t client_heartbeat_ms;      // <=0 => 10000
  uint32_t reserved0;               // must be zero
} dfkv_client_options_v2;

dfkv_client_t dfkv_open_v2(const dfkv_client_options_v2* options);

// PUT values must be non-empty. A zero byte count fails before routing.
int dfkv_put(dfkv_client_t c, const void* key, uint64_t key_len,
             const void* ptr, uint64_t n);  // 0=ok,-1=failure
int dfkv_get(dfkv_client_t c, const void* key, uint64_t key_len,
             void* ptr, uint64_t n);  // 1=hit,0=miss/failure
// Variable-size get: writes the stored payload (whatever length it was put with)
// into ptr (which has capacity `cap`) and reports the actual byte length via
// *out_len. Unlike dfkv_get, the caller does NOT need to know the exact stored
// size; a stored payload larger than cap is a miss. 1=hit, 0=miss/failure.
int dfkv_get_auto(dfkv_client_t c, const void* key, uint64_t key_len,
                  void* ptr, uint64_t cap, uint64_t* out_len);
int dfkv_exist(dfkv_client_t c, const void* key, uint64_t key_len);  // 1/0
// Drop a key from the cache (LMCache L2 eviction). Returns 1 if the owning node
// confirmed the op (block removed OR already absent), 0 on route/health/IO error.
int dfkv_remove(dfkv_client_t c, const void* key, uint64_t key_len);  // 1/0
// Register a large host memory region (e.g. the whole SGLang host KV pool) so
// put/get into any covered buffer uses a pre-registered RDMA MR. Call once at
// startup after allocation and before traffic. Copying transports succeed
// without work. Returns 0 only when the full range is ready, -1 otherwise.
int dfkv_register_memory(dfkv_client_t c, const void* base, uint64_t size);
// Runtime payload-segment limit reported by this client's active transport.
// Size batch_put_sg / batch_get_auto_sg grouping from this value; do not assume
// a fixed HCA or transport limit. Returns 0 on a null client.
uint32_t dfkv_max_sg_segs(dfkv_client_t c);

// Actual client transport selected by dfkv_open_v2(), e.g. "rdma",
// "tcp(rdma-not-requested)", or "injected". Returns "" for null clients.
const char* dfkv_transport_mode(dfkv_client_t c);

// Batched, concurrently fanned out. Keys are binary spans
// (keys[i], key_lens[i]); zero-length/null/overflow keys fail their slot.
// PUT slots with a zero-length value also fail without affecting valid siblings.
// out_* arrays (len n) receive per-item results. Return 0 on call success.
int dfkv_batch_put(dfkv_client_t c, const void* const* keys,
                   const uint64_t* key_lens, const void** ptrs,
                   const uint64_t* sizes, int n, int* out_ok);
int dfkv_batch_get(dfkv_client_t c, const void* const* keys,
                   const uint64_t* key_lens, void** ptrs,
                   const uint64_t* sizes, int n, int* out_hit);
int dfkv_batch_get_auto(dfkv_client_t c, const void* const* keys,
                        const uint64_t* key_lens, void** ptrs,
                        const uint64_t* caps, int n, int* out_hit,
                        uint64_t* out_len);
int dfkv_batch_exist(dfkv_client_t c, const void* const* keys,
                     const uint64_t* key_lens, int n, int* out_exist);
int dfkv_batch_remove(dfkv_client_t c, const void* const* keys,
                      const uint64_t* key_lens, int n, int* out_ok);

// Scatter-gather batch put: each of the n keys gathers num_bufs[i] non-contiguous
// source buffers (ptrs[i][0..num_bufs[i]-1], sizes[i][...]) into one stored blob
// (the in-order concatenation; segment boundaries are client-side only). Coalesces
// many tiny KV chunks into one key + one transport SG op. A key exceeding
// dfkv_max_sg_segs(c), whose total byte length is zero, or whose total overflows
// size_t reports out_ok[i]=0 without affecting valid siblings. Return 0 on call
// success.
int dfkv_batch_put_sg(dfkv_client_t c, const void* const* keys,
                      const uint64_t* key_lens, const void*** ptrs,
                      const uint64_t** sizes, const int* num_bufs, int n,
                      int* out_ok);
// Scatter-gather variable-size batch get: each key's stored blob is scattered
// across num_dsts[i] destination buffers (dsts[i][...], caps[i][...]) in order
// (segment sizes define the split). Accepts any stored size <= sum(caps[i]).
// out_hit[i]=1/0; out_len[i] receives the true stored payload byte length (0 on
// miss). A key exceeding the runtime SG limit, or whose capacity sum overflows
// size_t, is a miss. Return 0 on success.
int dfkv_batch_get_auto_sg(dfkv_client_t c, const void* const* keys,
                           const uint64_t* key_lens, void*** dsts,
                           const uint64_t** caps, const int* num_dsts, int n,
                           int* out_hit, uint64_t* out_len);

// Client-side Prometheus metrics snapshot. Every public scalar/batch call is
// counted exactly once with its submitted key count, including early failures
// and rendezvous-served hits. Writes up to `cap` bytes (NUL-terminated when it
// fits) and returns the full text length excluding NUL. Returns 0 on failure.
uint64_t dfkv_stats_snapshot(dfkv_client_t c, char* buf, uint64_t cap);

// dfkv native library version (compile-time DFKV_VERSION, e.g. "1.6.3" or "dev").
// Process-global (no client handle); never NULL; the returned string is static
// and valid for the process lifetime. Connectors report it as the
// dfkv_native_version metric label so the version of the linked libdfkv.so is
// visible per connector instance (alongside daemons' dfkv_build_info gauge).
const char* dfkv_version(void);

void dfkv_close(dfkv_client_t c);

#ifdef __cplusplus
}
#endif

#endif  // DFKV_DFKV_C_API_H_
