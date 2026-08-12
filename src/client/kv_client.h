/* KVClient — the standalone DingoFS KV cache client used by the SGLang HiCache
 * plugin. Routes canonical SHA-256 identities through consistent hashing and
 * stores caller values as opaque raw bytes. Model metadata belongs in the key;
 * values carry no dfkv-owned envelope. */
#ifndef DFKV_KV_CLIENT_H_
#define DFKV_KV_CLIENT_H_

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "utils/con_hash.h"
#include "client/node_dedup.h"
#include "client/node_dedup_gpu.h"
#include "common/membership.h"
#include "mds/mds_member_poller.h"
#include "mds/mds_registrar.h"
#include "client/op_metrics.h"
#include "client/peer_health.h"
#include "client/peer_latency.h"
#include "transport/transport.h"

namespace dfkv {

// When batch_concurrency is unset (0 = auto), use at least 8 workers so a
// single-node ring (groups=1) still gets 8-way parallel batch_put_sg /
// batch_get_auto_sg. Multi-node rings (groups >= 8) are unaffected: they
// already get one worker per group. The floor of 8 matches the typical
// RDMA connection pool size (DFKV_TCP_POOL_MAX_CONNS), so the extra
// workers do not exceed the pooled connection count.
inline size_t AutoBatchWorkers(size_t bc, size_t groups) {
  constexpr size_t kMaxAutoBatchWorkers = 32;
  constexpr size_t kMinAutoBatchWorkers = 8;
  if (bc) return bc;
  // auto: one worker per group, but at least 8 (for single-node rings),
  // capped at 32. Multi-node rings (groups >= 8) are unaffected.
  return std::min(std::max<size_t>(groups, kMinAutoBatchWorkers),
                  kMaxAutoBatchWorkers);
}

struct KvPutItem { std::string key; const void* value; size_t n; };
struct KvGetItem { std::string key; void* out; size_t n; };

// Scatter-gather batch items: one dfkv key gathers N non-contiguous source
// buffers on put / scatters into N destination buffers on get. The stored value
// is the in-order concatenation of the segments; segment boundaries are purely
// client-side (the server stores one opaque blob). ptrs.size() == sizes.size();
// dsts.size() == caps.size(). Used by the additive C ABI dfkv_batch_put_sg /
// dfkv_batch_get_auto_sg; N may exceed one transport WR's SGE width.
struct KvPutItemSg {
  std::string key;
  std::vector<const void*> ptrs;
  std::vector<size_t> sizes;
};
struct KvGetItemSg {
  std::string key;
  std::vector<void*> dsts;
  std::vector<size_t> caps;
};

class KVClient {
 public:
  // key_namespace is binary model/raw-layout identity. Canonical pool and
  // per-object coordinates live in `key`; both byte strings are length-framed
  // before hashing. Transport defaults when nullptr.
  KVClient(std::vector<std::pair<std::string, std::string>> members,
           std::string key_namespace, Transport* transport = nullptr,
           std::optional<size_t> batch_concurrency_override = std::nullopt);
  ~KVClient();

  // PUT values must be non-empty; zero-length values fail before routing.
  bool Put(const std::string& key, const void* value, size_t n);
  bool Get(const std::string& key, void* out, size_t n);  // true = hit (exact n)
  // Variable-size get. The authoritative full stored size comes from the
  // response prefix, separately from returned range bytes.
  bool GetAuto(const std::string& key, std::string* out,
               size_t max_bytes = (64u << 20));
  // Variable-size get into a caller buffer. Returns false when the stored value
  // does not fit; on success *out_len is the full stored size.
  bool GetAuto(const std::string& key, void* out, size_t cap, size_t* out_len);
  bool Exist(const std::string& key);
  // Drop a key from its owning node. true iff the node confirmed the op (the
  // block was removed OR was already absent); false on route/health/IO failure.
  bool Remove(const std::string& key);

  // Batched, concurrently fanned out across owning nodes. Per-item results;
  // zero-length PUT values fail independently before routing.
  std::vector<bool> BatchPut(const std::vector<KvPutItem>& items);
  std::vector<bool> BatchGet(const std::vector<KvGetItem>& items);  // hit/miss
  // Variable-size batched get. items[i].n is the buffer CAPACITY (not the exact
  // payload size); out_lens (if non-null) receives the actual stored payload
  // length per item. Per-item hit/miss like BatchGet, but it accepts any stored
  // size that fits in the buffer instead of requiring payload_len == n — so it
  // reads back unfull/variable chunks. Keeps the RDMA zero-copy datapath for
  // full-size items (same RangeInto path as BatchGet).
  std::vector<bool> BatchGetAuto(const std::vector<KvGetItem>& items,
                                 std::vector<size_t>* out_lens);
  std::vector<bool> BatchExist(const std::vector<std::string>& keys);
  // Batched remove, fanned out across owning nodes (grouped per node). Per-item
  // result like BatchExist: true iff the owning node confirmed the op.
  std::vector<bool> BatchRemove(const std::vector<std::string>& keys);

  // Scatter-gather batch put: each key gathers its N source segments into one
  // stored blob (sum of sizes). Mirrors BatchPut (consistent-hash routing per
  // key, group by node, zero-copy bounded multi-WR gather where required).
  // Descriptor count is arbitrary; zero total size or a byte-sum overflow fails
  // the item before routing.
  std::vector<bool> BatchPutSg(const std::vector<KvPutItemSg>& items);
  // Scatter-gather variable-size batch get: each key's stored blob is scattered
  // across its N destination segments in order. Mirrors BatchGetAuto and accepts
  // a stored size <= the checked sum(caps). Descriptor count is arbitrary;
  // overflow is a miss. out_lens receives the true length (0 on miss).
  std::vector<bool> BatchGetAutoSg(const std::vector<KvGetItemSg>& items,
                                   std::vector<size_t>* out_lens);

  // Batch fan-out worker count. 0 (the default) = auto: one worker per node
  // group, floored at kMinAutoBatchWorkers (8) so single-node rings get
  // 8-way parallelism, capped at kMaxAutoBatchWorkers (32). Explicit n
  // keeps the old fixed behavior (dfkv_bench passes --bc 1 so its external
  // threads stay the only load).
  void set_batch_concurrency(size_t n) {
    batch_concurrency_.store(n, std::memory_order_relaxed);
  }

  // Register a large caller memory region (e.g. the whole SGLang host KV pool)
  // for zero-copy transfer, so Put/Get resolve every covered buffer to the
  // pre-registered pool MR. Returns false when RDMA cannot register the full
  // range; copying transports such as TCP return true without work.
  bool RegisterMemory(void* base, size_t size) {
    return t_->RegisterMemory(base, size);
  }

  // Runtime payload-segment width of one transport window. Logical SG calls may
  // exceed it: the transport splits them into one ordered operation internally.
  size_t MaxSgPayloadSegs() const { return t_->MaxSgPayloadSegs(); }

  // Hot-swap the cluster membership (rebuilds the consistent-hash ring).
  // Thread-safe vs concurrent Put/Get/Exist.
  void SetMembers(std::vector<std::pair<std::string, std::string>> members);

  // Apply a weighted member view (from MDS discovery); rebuilds the ring with
  // per-node vnode weight and resolves addr = "ip:port".
  void SetMembers(const std::vector<MemberInfo>& members);

  // Start background MDS discovery: poll the MDS endpoints for `group` and rebuild
  // the ring on each epoch change. Replaces static membership.
  void StartMdsDiscovery(std::vector<std::string> mds_eps, const std::string& group,
                         int poll_ms = 3000);
  // Block until MDS discovery populates a non-empty ring, or timeout. Returns
 // true if the ring is non-empty when the call returns. Use after
 // StartMdsDiscovery to avoid silent PUT/GET failures during the async
 // discovery race at startup (ring empty → Route() returns "" → all ops
 // silently fail). Env DFKV_OPEN_RING_WAIT_MS overrides the timeout (default 30s).
  bool WaitForRing(int timeout_ms = 30000);
  void StopMdsDiscovery();
  // Register THIS client (a cache consumer / inference instance) with the MDS so
  // `dfkvctl clients` can surface "who is using dfkv". `self` carries identity
  // only (id + info string; ip/port/weight are ignored for placement). The
  // registrar leases its own etcd key under /clients/<id> and keeps it alive via
  // heartbeats; on process death the lease expires within the MDS TTL — no
  // explicit deregister, no stale keys. Best-effort: registration failure is
  // logged and retried, never blocking the data path.
  void StartClientRegistration(std::vector<std::string> mds_eps, const std::string& group,
                               MemberInfo self, int heartbeat_ms = 10000);
  void StopClientRegistration();

  // Discovery: query a seed node ("ip:port") for the current cluster membership
  // and SetMembers() it. Lets clients learn add/remove without a static list or
  // MDS — point at any live node. Returns true if a non-empty list was applied.
  bool RefreshMembers(const std::string& seed_addr);

  // Client-side Prometheus metrics text (ops served, IO errors, peer health
  // transitions, per-peer errors, per-peer latency) plus transport-level pool,
  // RDMA rail, and memory-registration counters. Surfaced via the C ABI.
  std::string MetricsSnapshot() const;
  const std::string& TransportMode() const { return transport_reason_; }

  // Start/stop the active per-peer latency prober: a background thread that
  // sends a cheap round trip (kExist of a sentinel key) to each known member
  // every `interval_ms`, so per-node avg/max latency is visible even when the
  // client is idle. interval_ms<=0 is a no-op. Also auto-started from the ctor
  // when DFKV_PROBE_INTERVAL_MS>0. Thread-safe vs Stop.
  void StartProbe(int interval_ms);
  void StopProbe();

 private:
  std::string Route(const std::string& key) const;
  uint64_t NowMs() const;
  void ProbeLoop();
  // Swap in a freshly-built ring under ring_mu_ and log the membership delta
  // (added/removed node ids) vs the previously-adopted view. Shared by both
  // SetMembers overloads so static and MDS-discovery paths log identically.
  void AdoptRing(ConHash ring, std::map<std::string, std::string> addr);
  // Publish one replace-all topology response. Peers omitted by the response
  // receive incomplete snapshots at its generation while they are still held
  // by the ring, so a guard-rejected placement shrink cannot retain stale rails.
  void PublishPeerTopologies(const std::vector<MemberInfo>& topology,
                             uint64_t generation);
  // Scalar transport bodies. Public scalar methods add exactly one metric;
  // TCP batch fan-out calls these bodies so it cannot double-count each key.
  bool PutDirect(const std::string& key, const void* value, size_t n);
  bool GetDirect(const std::string& key, void* out, size_t n);
  bool GetAutoDirect(const std::string& key, std::string* out,
                     size_t max_bytes);
  bool GetAutoDirect(const std::string& key, void* out, size_t cap,
                     size_t* out_len);
  bool ExistDirect(const std::string& key, Status* status);
  bool RemoveDirect(const std::string& key);
  void InvalidateRendezvous(const BlockKey& key);

  // Plain batch bodies; public methods own the one metric record across direct
  // and rendezvous paths.
  std::vector<bool> BatchGetDirect(const std::vector<KvGetItem>& items);
  std::vector<bool> BatchGetAutoDirect(const std::vector<KvGetItem>& items,
                                       std::vector<size_t>* out_lens);
  std::vector<bool> BatchGetAutoSgDirect(const std::vector<KvGetItemSg>& items,
                                         std::vector<size_t>* out_lens);
  std::vector<bool> BatchExistDirect(const std::vector<std::string>& keys,
                                     std::vector<Status>* statuses);
  GpuNodeDedup* GpuDedup(const void* device_dst_hint);
  // Record one public batch operation (hits are the true flags) and return the
  // item results. Centralizing this at each public return point prevents direct
  // and same-host-rendezvous paths from double-counting or escaping metrics.
  std::vector<bool> RecordBatch(OpMetrics::Op op,
                                std::chrono::steady_clock::time_point t0,
                                std::vector<bool> flags, uint64_t bytes);

  mutable std::mutex ring_mu_;  // guards ring_ + addr_
  ConHash ring_;
  std::map<std::string, std::string> addr_;  // name -> ip:port
  // Addresses in the preceding replace-all topology response. Together with
  // addr_, this finds omitted published or still-ring-held peers.
  std::set<std::string> published_peer_addrs_;
  std::condition_variable ring_cv_;  // notified when AdoptRing sets a non-empty ring
  std::string key_namespace_;
  uint64_t namespace_hash_ = 0;
  std::unique_ptr<Transport> owned_;
  // Null until the constructor selects a transport; AdoptRing runs earlier
  // (initial SetMembers) and must observe null, not garbage.
  Transport* t_ = nullptr;
  std::string transport_reason_ = "unknown";
  // Atomic: configurable via the C ABI; reads on the batch path are relaxed.
  std::atomic<size_t> batch_concurrency_{0};  // 0 = auto (see set_batch_concurrency)
  size_t BatchWorkers(size_t groups) const {
    return AutoBatchWorkers(batch_concurrency_.load(std::memory_order_relaxed), groups);
  }
  std::unique_ptr<MdsMemberPoller> poller_;
  // Same-host GET rendezvous (phase 5, DFKV_CLIENT_NODE_DEDUP=1; host-memory
  // destinations only). nullptr = feature off.
  std::unique_ptr<NodeDedup> dedup_;
  // Same-host GET rendezvous for GPU destinations (phase 2b; vLLM SG path).
  // Lazy-init via GpuDedup(); raw pointer published for lock-free metrics.
  std::once_flag gpu_dedup_once_;
  std::unique_ptr<GpuNodeDedup> gpu_dedup_;
  std::atomic<GpuNodeDedup*> gpu_dedup_raw_{nullptr};
  std::unique_ptr<MdsRegistrar> client_registrar_;  // consumer identity lease (best-effort)
  PeerHealth health_;
  OpMetrics op_stats_;  // every public put/get/exist/remove call, exactly once
  int get_miss_retries_ = 1;

  // Active per-peer latency prober (off the datapath; own thread).
  PeerLatency peer_lat_;
  std::atomic<bool> probe_running_{false};
  int probe_interval_ms_ = 0;
  std::thread probe_th_;
  std::mutex probe_mu_;
  std::condition_variable probe_cv_;
};

}  // namespace dfkv

#endif  // DFKV_KV_CLIENT_H_
