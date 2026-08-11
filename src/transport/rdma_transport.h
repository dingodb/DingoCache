/* RDMA client transport — native v2 libibverbs RC. Requests and bounded
 * responses use 32,786-byte SEND/RECV control buffers (18-byte prefix plus a
 * 32-KiB Members payload); PUT/GET payloads use one-sided WRITEs.
 * A peer that cannot negotiate v2 is rejected.
 * An empty DFKV_RDMA_DEV discovers every ACTIVE HCA; an explicit comma list is
 * a whitelist. Device names, not IPs, select the data fabric. QPs bootstrap over
 * a small TCP channel to the node's member address, so the RDMA fabric itself
 * needs no IP. */
#ifndef DFKV_RDMA_TRANSPORT_H_
#define DFKV_RDMA_TRANSPORT_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <limits>
#include <mutex>
#include <thread>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "transport/transport.h"
#include "transport/rail_select.h"
#include "transport/rdma_resource_budget.h"

namespace dfkv {
namespace rdma {
class RcEndpoint;
class RdmaTopology;
class RailPolicy;
}

// One absolute budget for all WaitComp calls needed to reap a posted window.
// RemainingAt keeps timeout-budget tests deterministic without RDMA hardware.
class CompletionDeadline {
 public:
  using Clock = std::chrono::steady_clock;
  explicit CompletionDeadline(int timeout_ms,
                              Clock::time_point start = Clock::now())
      : infinite_(timeout_ms < 0),
        deadline_(infinite_ ? Clock::time_point::max()
                            : start + std::chrono::milliseconds(timeout_ms)) {}

  int Remaining() const { return RemainingAt(Clock::now()); }
  int RemainingAt(Clock::time_point now) const {
    if (infinite_) return -1;
    if (now >= deadline_) return 0;
    const auto remaining = std::chrono::duration_cast<std::chrono::microseconds>(
                               deadline_ - now)
                               .count();
    const auto rounded_ms = (remaining + 999) / 1000;
    return static_cast<int>(std::min<int64_t>(
        rounded_ms, std::numeric_limits<int>::max()));
  }

 private:
  bool infinite_;
  Clock::time_point deadline_;
};


class RdmaTransport : public Transport {
 public:
  static bool Available();  // true if at least one ACTIVE RDMA port is present

  // dev_name empty => env DFKV_RDMA_DEV. A comma-separated explicit value is a
  // whitelist and opts into multi-rail. With neither, use the first ACTIVE local
  // HCA and send no device name to the peer, preserving host-local selection.
  explicit RdmaTransport(size_t max_msg = (64u << 20),
                         const std::string& dev_name = "");
  ~RdmaTransport() override;

  Status Cache(const std::string& node, const BlockKey& key, const void* data,
               size_t len) override;
  Status Range(const std::string& node, const BlockKey& key, uint64_t offset,
               uint64_t length, std::string* out,
               uint64_t* value_len = nullptr) override;
  Status Lookup(const std::string& node, const BlockKey& key,
                uint64_t* value_len) override;
  Status Exist(const std::string& node, const BlockKey& key,
               bool* exist) override;
  Status Remove(const std::string& node, const BlockKey& key) override;
  Status Members(const std::string& node, std::string* out) override;

  bool RegisterMemory(void* base, size_t size) override;
  std::string MetricsText() const override;  // dfkv_rdma_client_* (conns, per-rail)

  bool pipelined() const override { return true; }
  size_t MaxSgPayloadSegs() const override { return sg_payload_segs_; }
  // Pipelined: up to `depth_` requests in flight on a single connection (default 4; env DFKV_RDMA_DEPTH).
  std::vector<Status> CacheMany(const std::string& node,
                                const std::vector<CacheItem>& items) override;
  std::vector<Status> CacheFrom(const std::string& node,
                                const std::vector<CacheSrc>& srcs) override;
  std::vector<Status> RangeMany(
      const std::string& node, const std::vector<BlockKey>& keys,
      uint64_t offset, uint64_t length, std::vector<std::string>* outs,
      std::vector<uint64_t>* value_lens = nullptr) override;
  std::vector<Status> ExistMany(const std::string& node,
                                const std::vector<BlockKey>& keys,
                                std::vector<char>* exists) override;
  std::vector<Status> RangeInto(
      const std::string& node, const std::vector<BlockKey>& keys,
      const std::vector<RangeDst>& dsts,
      std::vector<uint64_t>* value_lens) override;
  // Scatter-gather overrides: one wire SEND/RECV gathers/scatters N payload
  // segments per key via multi-SGE work requests (additive zero-copy datapath
  // for dfkv_batch_put_sg / dfkv_batch_get_auto_sg).
  std::vector<Status> CacheFromMulti(
      const std::string& node,
      const std::vector<CacheSrcMulti>& srcs) override;
  std::vector<Status> RangeIntoMulti(
      const std::string& node, const std::vector<BlockKey>& keys,
      const std::vector<RangeDstMulti>& dsts,
      std::vector<size_t>* out_lens) override;

 private:
  struct Conn;
  // force_new skips the idle pool after a stale connection. Lanes separate
  // payload, device-direct SG, and key-sized control traffic; each lane keeps
  // the negotiated depth while owning an independent endpoint pool.
  enum class Lane { kData, kSgData, kControl };
  Conn* Acquire(const std::string& node, Lane lane, bool* from_pool,
                bool force_new = false, size_t requested_credits = 1);
  void Release(const std::string& node, Lane lane, Conn* c);
  void Destroy(Conn* c,
               rdma::RailCompletion completion =
                   rdma::RailCompletion::kRailFailure);
  bool EvictOneIdle();
  // Keep every idle QP alive while its client process is healthy. This lets a
  // short server-side idle reaper reclaim dead clients without forcing the
  // first cache read after an idle gap to discover and rebuild stale QPs.
  void KeepaliveLoop();
  bool KeepaliveConn(Conn* c);
  Status RoundTrip(const std::string& node, WireOp op, const BlockKey& key,
                   uint64_t offset, uint64_t length, const void* payload,
                   uint64_t payload_len, std::string* out,
                   uint64_t* value_len = nullptr);
  bool ProbeV2(const std::string& node) const;
  std::mutex mu_;
  std::unordered_map<std::string, std::vector<Conn*>> pool_;
  std::unordered_map<std::string, std::vector<Conn*>> sg_pool_;
  // Separate idle-connection pool for control-lane ops
  // (Exist/Remove/Members). Responses stay bounded, including the explicit
  // 32-KiB Members capacity, and never share a QP with payload transfers.
  std::unordered_map<std::string, std::vector<Conn*>> control_pool_;
  // Last successfully published caller memory declarations. RegisterMemory
  // holds mu_ through per-rail stage/commit, so Acquire can observe either the
  // complete old generation or the complete new one, never a partial growth.
  std::vector<std::pair<void*, size_t>> pools_;
  // Lifetime endpoint per active rail. Its cache reference anchors the newest
  // successful generation; old generations survive only while an older active
  // connection still owns a cache reference.
  std::vector<std::unique_ptr<rdma::RcEndpoint>> anchors_;
  // min over rails of (negotiated max_sge) - 1; set once in the ctor.
  size_t sg_payload_segs_ = 29;
  size_t max_payload_;
  // Maximum block declared during mandatory v2 negotiation. It is both a
  // protocol bound and receive-segment geometry: each data connection leases
  // queue_depth * aligned_slot_size from one fixed pinned segment. Inflating
  // this value reduces connection capacity; understating it deterministically
  // rejects larger operations. It must therefore be exact and nonzero.
  // Default 4 MiB (env DFKV_RDMA_MAX_BLOCK_BYTES); capped by max_payload_.
  uint64_t declared_ = 0;
  size_t OpBound() const {  // per-op payload bound honoring the declaration
    return declared_ ? static_cast<size_t>(declared_) : max_payload_;
  }
  // Largest block this client has actually handed to the transport. Operators
  // use the high-water mark to choose a tight DCP2 declaration: smaller slots
  // admit more concurrent v2 connections into the fixed shared segment, while
  // oversize operations must remain a deterministic client-side rejection.
  mutable std::atomic<uint64_t> max_block_seen_{0};
  mutable std::atomic<uint64_t> oversize_rejects_{0};
  // Records n as a candidate high-water mark and reports whether it exceeds the
  // declaration. Returns true for an oversized block (caller marks it kInvalid).
  bool NoteBlock(size_t n) const;
  size_t depth_;
  int connect_ms_ = 3000;             // bootstrap TCP connect timeout (DFKV_RDMA_CONNECT_MS)
  int io_ms_ = 10000;                 // bootstrap TCP IO timeout (DFKV_RDMA_IO_MS)
  // Per-window datapath completion deadline (DFKV_RDMA_OP_TIMEOUT_MS).
  // WaitComp otherwise blocks forever when an RC peer disappears without a
  // completion or a QP stalls in retries. On timeout the entire connection is
  // destroyed before transient MRs are released, the failed window is reported,
  // and the public operation may retry once on a fresh connection. An explicit
  // non-positive value restores the unbounded wait as an operator escape hatch.
  int op_timeout_ms_ = 5000;          // datapath completion timeout (DFKV_RDMA_OP_TIMEOUT_MS)
  // Batch-window override (DFKV_RDMA_BATCH_OP_TIMEOUT_MS) used by every
  // multi-item window: CacheMany, RangeMany, ExistMany, CacheFrom, RangeInto,
  // and both SG variants. <=0 follows op_timeout_ms_. One absolute deadline
  // covers the whole completion window; partial CQ drains never reset it.
  int batch_op_timeout_ms_ = 0;
  int BatchTimeout() const {
    return batch_op_timeout_ms_ > 0 ? batch_op_timeout_ms_ : op_timeout_ms_;
  }
  size_t pool_max_ = 16;              // idle conns kept per node/lane
  // Enabled by default below the recommended 30 s server reaper interval.
  // Set DFKV_RDMA_KEEPALIVE_MS=0 to disable.
  int keepalive_ms_ = 15000;
  std::atomic<bool> keepalive_stop_{false};
  std::condition_variable keepalive_cv_;
  std::mutex keepalive_mu_;
  std::thread keepalive_thread_;
  std::shared_ptr<rdma::ResourceBudget> resource_budget_;
  int resource_acquire_ms_ = 10000;
  std::unique_ptr<rdma::RdmaTopology> topology_;
  std::vector<std::string> devs_;  // stable discovered ACTIVE rail order
  bool auto_device_ = true;
  bool numa_aware_ = false;
  // Global least-inflight rail admission with per-rail credits, failure
  // quarantine, recovery probes, and a bounded wait when all credits are busy.
  std::unique_ptr<rdma::RailPolicy> rail_policy_;
  uint64_t rail_backpressure_us_ = 10'000;
  // observability (relaxed): connections opened total + per-rail breakdown.
  std::atomic<uint64_t> conns_opened_{0};
  std::atomic<uint64_t> v2_put_writes_{0}, v2_get_writes_{0};
  std::atomic<uint64_t> mr_regions_{0};
  std::atomic<uint64_t> mr_registered_bytes_{0};
  std::atomic<uint64_t> mr_registration_rejections_{0};
  mutable std::atomic<uint64_t> v2_probe_attempts_{0};
  mutable std::atomic<uint64_t> v2_probe_failures_{0};
  std::atomic<uint64_t> stale_pool_retries_{0};
  std::atomic<uint64_t> completion_timeouts_{0};
  std::atomic<uint64_t> keepalive_attempts_{0};
  std::atomic<uint64_t> keepalive_successes_{0};
  std::atomic<uint64_t> keepalive_failures_{0};
  std::unique_ptr<std::atomic<uint64_t>[]> rail_conns_;  // sized to devs_.size()
  std::atomic<uint64_t> endpoint_cache_hits_{0};
  std::atomic<uint64_t> endpoint_cache_misses_{0};
  std::atomic<uint64_t> endpoint_cache_evictions_{0};
  // Fixed-cardinality locality fallback reasons: caller NUMA unknown / no
  // enabled local rail. Endpoint failures are bounded per configured rail in
  // RailPolicy::Snapshot rather than labeled by arbitrary node addresses.
  std::atomic<uint64_t> numa_caller_unknown_fallbacks_{0};
  std::atomic<uint64_t> numa_no_local_fallbacks_{0};
  // #1: async CQ reaper for high-concurrency batch operations
};

}  // namespace dfkv

#endif  // DFKV_RDMA_TRANSPORT_H_
