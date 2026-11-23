/* KVClient — the standalone DingoFS KV cache client used by the SGLang HiCache
 * plugin. Routes canonical SHA-256 identities through consistent hashing and
 * stores caller values as opaque raw bytes. Model metadata belongs in the key;
 * values carry no dfkv-owned envelope. */
#ifndef DFKV_KV_CLIENT_H_
#define DFKV_KV_CLIENT_H_

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <shared_mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "client/item_status.h"
#include "client/key_map.h"
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

// Fan-out width of TCP batches and of RDMA writes. RDMA reads do not use it:
// they run on the process-wide ReadScheduler (DFKV_RDMA_READ_WORKERS).
// When batch_concurrency is unset (0 = auto), use at least 8 workers so a
// single-node ring (groups=1) still gets 8-way parallel batches. Multi-node
// rings (groups >= 8) get one worker per group. The floor of 8 matches the
// default TCP connection pool (DFKV_TCP_POOL_MAX_CONNS).
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
  // zero-length PUT values fail independently before routing. The batch
  // calls with a `status` argument also report each item's outcome there
  // (resized to the item count; see item_status.h).
  std::vector<bool> BatchPut(const std::vector<KvPutItem>& items,
                             std::vector<ItemStatus>* status = nullptr);
  std::vector<bool> BatchGet(const std::vector<KvGetItem>& items,
                             std::vector<ItemStatus>* status = nullptr);
  // Variable-size batched get. items[i].n is the buffer CAPACITY (not the exact
  // payload size); out_lens (if non-null) receives the actual stored payload
  // length per item. Per-item hit/miss like BatchGet, but it accepts any stored
  // size that fits in the buffer instead of requiring payload_len == n — so it
  // reads back unfull/variable chunks. Keeps the RDMA zero-copy datapath for
  // full-size items (same RangeInto path as BatchGet).
  std::vector<bool> BatchGetAuto(const std::vector<KvGetItem>& items,
                                 std::vector<size_t>* out_lens,
                                 std::vector<ItemStatus>* status = nullptr);
  std::vector<bool> BatchExist(const std::vector<std::string>& keys);

  // ---- Retention (kRetain) ----
  // Per-key result codes (mirrors dfkv_c_api.h DFKV_RET_*).
  enum RetainResult : uint8_t {
    kRetMiss = 0, kRetHitUnprotected = 1, kRetProtectedNew = 2,
    kRetRefreshed = 3, kRetUpgraded = 4, kRetUnchanged = 5, kRetDenied = 6,
    kRetUnsupported = 7, kRetSkippedDebounce = 8, kRetError = 9,
  };
  struct RetainRequest {
    uint32_t ttl_s = 0;        // SET: requested TTL. REFRESH: debounce hint
                               // (0 = never debounce; server ignores it)
    uint8_t mode = 0;          // 0 SET, 1 REFRESH
    bool no_debounce = false;
    uint32_t min_refresh_ms = 0;
  };
  // Batched RETAIN fanned out per owning node. out[i] is always written.
  // REFRESH doubles as an existence check: on a node without retention the
  // keys are probed with Exist instead (hit -> kRetHitUnprotected), so callers
  // can use it in place of BatchExist. A key is a hit iff its code is one of
  // 1,2,3,4,5,6,8.
  void BatchRetain(const std::vector<std::string>& keys,
                   const RetainRequest& req, uint8_t* out);
  // Ring-wide limits: *supported 1 = every member answered kRetain, 0 = some
  // member lacks it, -1 = unknown. *max_ttl_s = smallest cap seen (0 if not
  // supported). Unknown members are probed once, synchronously.
  void RetainLimits(int* supported, uint32_t* max_ttl_s);
  // Local, RPC-free: true iff every owner of `keys` supports retention and its
  // last reported headroom is >= min_headroom_pct (never-heard nodes pass).
  bool RetainPrecheck(const std::vector<std::string>& keys,
                      uint8_t min_headroom_pct);
  // Batched remove, fanned out across owning nodes (grouped per node). Per-item
  // result like BatchExist: true iff the owning node confirmed the op.
  std::vector<bool> BatchRemove(const std::vector<std::string>& keys);

  // Scatter-gather batch put: each key gathers its N source segments into one
  // stored blob (sum of sizes). Mirrors BatchPut (consistent-hash routing per
  // key, group by node, zero-copy bounded multi-WR gather where required).
  // Descriptor count is arbitrary; zero total size or a byte-sum overflow fails
  // the item before routing.
  std::vector<bool> BatchPutSg(const std::vector<KvPutItemSg>& items,
                               std::vector<ItemStatus>* status = nullptr);
  // Scatter-gather variable-size batch get: each key's stored blob is scattered
  // across its N destination segments in order. Mirrors BatchGetAuto and accepts
  // a stored size <= the checked sum(caps). Descriptor count is arbitrary;
  // overflow is a miss. out_lens receives the true length (0 on miss).
  std::vector<bool> BatchGetAutoSg(const std::vector<KvGetItemSg>& items,
                                   std::vector<size_t>* out_lens,
                                   std::vector<ItemStatus>* status = nullptr);

  // Batch fan-out worker count of TCP batches and RDMA writes; RDMA reads
  // are bounded by DFKV_RDMA_READ_WORKERS instead. 0 (the default) = auto:
  // one worker per node group, floored at kMinAutoBatchWorkers (8) so
  // single-node rings get 8-way parallelism, capped at kMaxAutoBatchWorkers
  // (32). Explicit n keeps the old fixed behavior (dfkv_bench passes --bc 1
  // so its external threads stay the only load).
  void set_batch_concurrency(size_t n) {
    batch_concurrency_.store(n, std::memory_order_relaxed);
  }

  // Register a large caller memory region (e.g. the whole SGLang host KV pool)
  // for zero-copy transfer, so Put/Get resolve every covered buffer to the
  // pre-registered pool MR. Returns false when RDMA cannot register the full
  // range; copying transports such as TCP return true without work.
  bool RegisterMemory(void* base, size_t size);

  // Runtime payload-segment width of one transport window. Logical SG calls may
  // exceed it: the transport splits them into one ordered operation internally.
  size_t MaxSgPayloadSegs() const { return t_->MaxSgPayloadSegs(); }

  // Effective client-local logical object ceiling, or zero if unsupported.
  uint64_t MaxBlockBytes() const { return t_->MaxBlockBytes(); }

  // Hot-swap the cluster membership (rebuilds the consistent-hash ring).
  // Thread-safe vs concurrent Put/Get/Exist.
  void SetMembers(std::vector<std::pair<std::string, std::string>> members);

  // Apply a weighted member view (from MDS discovery); rebuilds the ring with
  // per-node vnode weight and resolves addr = "ip:port". With absent grace
  // (DFKV_RING_ABSENT_GRACE_MS > 0) a member this view omits is kept absent
  // (see absent_).
  void SetMembers(const std::vector<MemberInfo>& members);
  // Drop the absent members whose grace has passed. MDS discovery calls it
  // once a second; public for tests.
  void TickAbsent();

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
  // The address ("ip:port") the ring routes `key` to; "" with no owner.
  std::string OwnerOf(const std::string& key) const { return Route(key); }
  // The addresses of the ring's members, sorted.
  std::vector<std::string> MemberAddresses() const;
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
  struct AbsentMember {
    MemberInfo info;       // its last record in an adopted view
    std::string addr;      // ip:port
    uint64_t since_ms = 0;  // when it vanished (NowMs)
  };
  using AbsentMap = std::map<std::string, AbsentMember>;
  void AdoptRing(ConHash ring, std::map<std::string, std::string> addr,
                 ConHash full_ring = ConHash(),
                 std::set<std::string> draining = {}, AbsentMap absent = {},
                 std::vector<MemberInfo> placement = {},
                 ConHash absent_ring = ConHash());
  // Absent grace: which members of the previous view stay absent under the
  // view `placement` (ring-eligible members), counting the ones dropped.
  // `returned` receives the absent members the view has again.
  AbsentMap NextAbsent(const std::vector<MemberInfo>& placement, uint64_t now,
                       std::vector<std::string>* returned);
  // Build the write and full rings of `placement` plus `absent` and adopt them.
  void AdoptPlacement(const std::vector<MemberInfo>& placement,
                      AbsentMap absent);
  // A REMOVE the absent member `id` did not confirm: kept for one retry when
  // it returns, within the queue bound.
  void QueueAbsentRemove(const std::string& id, const std::string& key);
  // Member `id` is back in the view (or, `returned` false, gone again).
  void NoteAbsentReturned(const std::string& id, bool returned);
  // Send the queued REMOVEs of the members that are back; the ones still not
  // confirmed are sent again at the next tick, for up to the grace after the
  // return. DropAbsentRemoves gives a dropped member's queue up.
  void RetryAbsentRemoves();
  void DropAbsentRemoves(const std::string& id);
  // Retire the transport topology of absent members that were dropped.
  void RetireDropped(const AbsentMap& dropped);
  // Publish one replace-all topology response. Stable MDS ids are retained
  // separately from routing addresses; superseded addresses are explicitly
  // retired, and the complete live-id set is reconciled after publication.
  // With `ring_held` (the placement guard refused this view, so the ring
  // keeps its members), a ring member the view omits keeps its last published
  // topology and stays live: retiring it would make strict rail selection
  // fail closed for a peer the ring still routes to. Once a view is admitted,
  // omitted peers are retired.
  // Each present peer is published under its own MemberTopologyEpoch, so the
  // transport sees a new generation only for members whose own entry changed;
  // `generation`, the epoch of the whole view, only labels retirements.
  // `keep`: absent members (id -> address) whose topology stays live while
  // the view omits them.
  void PublishPeerTopologies(const std::vector<MemberInfo>& topology,
                             uint64_t generation, bool ring_held,
                             const std::map<std::string, std::string>& keep = {});
  // Scalar transport bodies. Public scalar methods add exactly one metric;
  // TCP batch fan-out calls these bodies so it cannot double-count each key.
  // `status`, when given, receives the item's outcome.
  bool PutDirect(const std::string& key, const void* value, size_t n,
                 ItemStatus* status = nullptr);
  // The read, exist and remove bodies below ask a key's draining owner too
  // (see full_ring_); each *Routed form asks the one node Route() names.
  bool GetDirect(const std::string& key, void* out, size_t n,
                 ItemStatus* status = nullptr);
  bool GetDirectRouted(const std::string& key, void* out, size_t n,
                       ItemStatus* status);
  bool GetAutoDirect(const std::string& key, std::string* out,
                     size_t max_bytes);
  bool GetAutoDirectRouted(const std::string& key, std::string* out,
                           size_t max_bytes);
  bool GetAutoDirect(const std::string& key, void* out, size_t cap,
                     size_t* out_len, ItemStatus* status = nullptr);
  bool GetAutoDirectRouted(const std::string& key, void* out, size_t cap,
                           size_t* out_len, ItemStatus* status);
  bool ExistDirect(const std::string& key, Status* status);
  bool ExistDirectRouted(const std::string& key, Status* status);
  bool RemoveDirect(const std::string& key);
  bool RemoveDirectRouted(const std::string& key);
  void CountDrainRead(bool hit);
  void InvalidateRendezvous(const BlockKey& key);
  DestinationMemoryKind ClassifyDestination(const void* destination,
                                             size_t size) const;

  // Plain batch bodies; public methods own the one metric record across direct
  // and rendezvous paths. on_hit(i), when given, runs as soon as item i is a
  // final hit (its destination holds the value, and the auto form's
  // *out_lens entry is not yet written: the length is passed), possibly on a
  // worker thread and concurrently for other items. `status`, when given,
  // points at one outcome per item.
  std::vector<bool> BatchGetDirect(
      const std::vector<KvGetItem>& items,
      const std::function<void(size_t)>& on_hit = nullptr,
      ItemStatus* status = nullptr);
  std::vector<bool> BatchGetDirectRouted(
      const std::vector<KvGetItem>& items,
      const std::function<void(size_t)>& on_hit, ItemStatus* status);
  std::vector<bool> BatchGetAutoDirect(
      const std::vector<KvGetItem>& items, std::vector<size_t>* out_lens,
      const std::function<void(size_t, size_t)>& on_hit = nullptr,
      ItemStatus* status = nullptr);
  std::vector<bool> BatchGetAutoDirectRouted(
      const std::vector<KvGetItem>& items, std::vector<size_t>* out_lens,
      const std::function<void(size_t, size_t)>& on_hit, ItemStatus* status);
  std::vector<bool> BatchGetAutoSgDirect(const std::vector<KvGetItemSg>& items,
                                         std::vector<size_t>* out_lens,
                                         ItemStatus* status = nullptr);
  std::vector<bool> BatchGetAutoSgDirectRouted(
      const std::vector<KvGetItemSg>& items, std::vector<size_t>* out_lens,
      ItemStatus* status);
  std::vector<bool> BatchExistDirect(const std::vector<std::string>& keys,
                                     std::vector<Status>* statuses);
  std::vector<bool> BatchExistDirectRouted(const std::vector<std::string>& keys,
                                           std::vector<Status>* statuses);
  // The RDMA (pipelined) BatchRemove body for the nodes Route() names.
  std::vector<char> BatchRemoveRouted(const std::vector<std::string>& keys);
  void BatchRetainRouted(const std::vector<std::string>& keys,
                         const RetainRequest& req, uint8_t* out);
  // Indices of the keys owned by a draining member (none unless one drains,
  // and none inside a drain pass).
  template <typename KeyOf>
  std::vector<size_t> DrainerOwned(size_t n, const std::vector<bool>* done,
                                   KeyOf key_of) const;
  GpuNodeDedup* GpuDedup(const void* device_dst_hint);
  // Record one public batch operation (hits are the true flags) and return the
  // item results. Centralizing this at each public return point prevents direct
  // and same-host-rendezvous paths from double-counting or escaping metrics.
  std::vector<bool> RecordBatch(OpMetrics::Op op,
                                std::chrono::steady_clock::time_point t0,
                                std::vector<bool> flags, uint64_t bytes);

  struct RegisteredMemoryRange {
    uintptr_t base;
    uintptr_t end;  // exclusive
    DestinationMemoryKind kind;
  };
  // RegisterMemory is rare; GET classification takes only this shared lock and
  // scans the small declaration list without allocating. Newer overlapping
  // declarations win, matching a successful same-base growth deterministically.
  mutable std::shared_mutex registered_memory_mu_;
  std::vector<RegisteredMemoryRange> registered_memory_;
  // Retention per-node view (from kRetain replies) and refresh debounce.
  struct RetainNodeState {
    int supported = -1;  // -1 unknown, 0 no, 1 yes
    // When "no" was learned. A "no" older than retain_reprobe_ms_ is asked
    // again (the node may have been upgraded or replaced since).
    uint64_t unsupported_at_ms = 0;
    uint32_t max_ttl_s = 0;
    uint8_t headroom_pct = 100;
    bool heard = false;
  };
  mutable std::mutex retain_mu_;
  std::map<std::string, RetainNodeState> retain_nodes_;
  // A key the node answered as protected: when, by which node (a hash of its
  // address, so that a key whose owner changed is asked again), and for how
  // long the protection can at most last (the TTL the SET was granted, or
  // the node's cap when only a REFRESH was seen). A write or remove of the
  // key through this client drops it.
  struct RetainDebounceEntry {
    uint64_t protected_at_ms = 0;
    uint64_t valid_ms = 0;
    uint64_t node_id = 0;
  };
  void ForgetRetained(const BlockKey& key);
  static constexpr size_t kRetainDebounceShards = 16;
  struct RetainDebounceShard {
    std::mutex mu;
    std::map<std::pair<uint64_t, uint64_t>, RetainDebounceEntry> entries;
    // Bumped by every ForgetRetained of a key of this shard. A kRetain reply
    // records an entry only if no forget happened since its request was
    // built: the reply describes the value from before a PUT or remove that
    // raced it.
    uint64_t forget_epoch = 0;
  };
  RetainDebounceShard retain_debounce_[kRetainDebounceShards];
  size_t retain_debounce_cap_ = 0;  // per shard; from DFKV_RETAIN_CACHE_ENTRIES
  void RetainNode(const std::string& node, uint64_t tenant,
                  const std::vector<size_t>& idx,
                  const std::vector<BlockKey>& bkeys,
                  const RetainRequest& req, uint8_t* out);
  mutable std::mutex ring_mu_;  // guards ring_, addr_, full_ring_, draining_, absent_, absent_ring_
  // Placement for writes: the eligible members that are not draining (all of
  // them while none is, or if every one is). Reads ask its owner first.
  ConHash ring_;
  std::map<std::string, std::string> addr_;  // every eligible member: name -> ip:port
  // Drain routing (DFKV_DRAIN on the nodes; MemberInfo::draining). While a
  // member drains, full_ring_ is the ring with it, the one keys were written
  // under; a key whose full-ring owner drains is read from it after a miss
  // at its write owner, and REMOVE and RETAIN go to both. Empty otherwise.
  ConHash full_ring_;
  std::set<std::string> draining_;
  std::atomic<bool> any_draining_{false};
  // Absent grace (DFKV_RING_ABSENT_GRACE_MS; 0 = off). A member that
  // vanishes from an admitted view is kept absent until its grace passes
  // (TickAbsent) or it returns: no writes and no reads; REMOVE and RETAIN
  // reach it; its unconfirmed REMOVEs are retried when it returns.
  // absent_ring_ is the view's members plus the absent ones (empty while
  // none is absent); a key it gives to an absent member has that member as
  // one more owner, beside the write owner and a drainer (full_ring_). At most
  // max(1, N/8) members are absent at once; a member whose last record was
  // draining (a planned leave) is not kept. placement_ is the last adopted
  // view (ring-eligible members), kept to rebuild the rings when an absent
  // member is dropped. Both under ring_mu_.
  AbsentMap absent_;
  ConHash absent_ring_;
  std::atomic<bool> any_absent_{false};
  std::vector<MemberInfo> placement_;
  uint64_t absent_grace_ms_ = 0;
  uint64_t topology_generation_ = 0;  // last published; labels retirements
  // REMOVEs absent members did not confirm, by member id, sent again when
  // the member is back (at once, then at each tick until confirmed, for up
  // to the grace). Bounded in total (DFKV_RING_ABSENT_REMOVE_QUEUE).
  std::mutex absent_remove_mu_;
  std::map<std::string, std::set<std::string>> absent_removes_;
  std::map<std::string, uint64_t> absent_returned_;  // id -> when it came back
  // Returned members whose PeerHealth cooldown the first retry may end.
  std::set<std::string> absent_cooldown_reset_;
  size_t absent_removes_queued_ = 0;
  size_t absent_remove_cap_ = 4096;
  std::atomic<uint64_t> absent_dropped_grace_{0}, absent_dropped_bound_{0},
      absent_dropped_planned_{0};
  std::atomic<uint64_t> absent_removes_unconfirmed_{0},
      absent_removes_retried_{0}, absent_removes_dropped_{0};
  // The drainer of a key: its full-ring owner's address if that member
  // drains, else "".
  std::string DrainRoute(const std::string& key) const;
  // The absent owner of a key: its owner on absent_ring_ if that member is
  // absent (its address; `id` receives its id), else "". Only REMOVE and
  // RETAIN go there: an absent member's copy may be one a REMOVE from a
  // client that never saw the member did not reach, so reading it is a
  // read-back (a later stage, under its rules).
  std::string AbsentRoute(const std::string& key, std::string* id) const;
  struct AbsentPart {
    std::string id, addr;
    std::vector<size_t> index;  // into the keys
  };
  // The keys with an absent owner, by member.
  std::vector<AbsentPart> AbsentOwned(const std::vector<std::string>& keys) const;
  // REMOVE `keys`' absent-owned ones at their absent owner; the ones it
  // does not confirm are queued for its return.
  void RemoveAtAbsent(const std::vector<std::string>& keys);
  // Second requests to a draining owner: reads that found the key there /
  // did not; REMOVEs and RETAINs it confirmed / did not.
  std::atomic<uint64_t> drain_reads_hit_{0}, drain_reads_miss_{0};
  std::atomic<uint64_t> drain_removes_ok_{0}, drain_removes_failed_{0};
  std::atomic<uint64_t> drain_retains_ok_{0}, drain_retains_failed_{0};
  // Stable identity -> routing address published to the transport: the
  // preceding replace-all topology response plus the ring members a refused
  // view omitted. Together with addr_, this finds removed identities and old
  // addresses.
  std::map<std::string, std::string> published_peers_;
  std::condition_variable ring_cv_;  // notified when AdoptRing sets a non-empty ring
  std::string key_namespace_;
  BlockKeyHasher block_key_;  // ToBlockKey(key_namespace_, key)
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
  // Reports an operation's statuses for `node` to health_ (see the .cc).
  void ReportPeer(const std::string& node, const Status* statuses, size_t count);
  void ReportPeer(const std::string& node, Status status) {
    ReportPeer(node, &status, 1);
  }
  void ReportPeer(const std::string& node, const std::vector<Status>& statuses) {
    ReportPeer(node, statuses.data(), statuses.size());
  }
  OpMetrics op_stats_;  // every public put/get/exist/remove call, exactly once
  int get_miss_retries_ = 1;
  uint64_t retain_reprobe_ms_ = 30000;  // DFKV_RETAIN_REPROBE_MS
  // supported == 0, learned less than retain_reprobe_ms_ ago. retain_mu_.
  bool KnownUnsupported(const RetainNodeState& ns, uint64_t now) const;

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
