/* RDMA client transport — native libibverbs RC, two-sided SEND/RECV.
 * Built only when DFKV_WITH_RDMA is defined. Mirrors the TCP wire frames so the
 * server request logic is shared. Connection-pooled per node.
 *
 * Device is selected BY NAME (env DFKV_RDMA_DEV, e.g. "ib7s400p0"), not by IP,
 * so the data plane can ride a 400G IB fabric that has no IP and is separate
 * from the IP network. The QP is bootstrapped over a small TCP channel to the
 * node's member address (an IP on the shared 200G/bond0 network). See
 * rdma_verbs.h for the control-plane/data-plane split rationale. */
#ifndef DFKV_RDMA_TRANSPORT_H_
#define DFKV_RDMA_TRANSPORT_H_

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "transport/transport.h"

namespace dfkv {
namespace rdma { class RcEndpoint; }

class RdmaTransport : public Transport {
 public:
  static bool Available();  // true if at least one RDMA device is present

  // dev_name empty => env DFKV_RDMA_DEV (comma-separated list = multi-rail; each
  // new connection round-robins across the devices), else first device.
  explicit RdmaTransport(size_t max_msg = (64u << 20), const std::string& dev_name = "");
  ~RdmaTransport() override;

  Status Cache(const std::string& node, const BlockKey& key, const void* data,
               size_t len) override;
  Status Range(const std::string& node, const BlockKey& key, uint64_t offset,
               uint64_t length, std::string* out) override;
  Status Exist(const std::string& node, const BlockKey& key,
               bool* exist) override;
  Status Remove(const std::string& node, const BlockKey& key) override;
  Status Members(const std::string& node, std::string* out) override;

  void RegisterMemory(void* base, size_t size) override;
  std::string MetricsText() const override;  // dfkv_rdma_client_* (conns, per-rail)

  bool pipelined() const override { return true; }
  size_t MaxSgPayloadSegs() const override { return sg_payload_segs_; }
  // Pipelined: up to `depth_` requests in flight on a single connection.
  std::vector<Status> CacheMany(const std::string& node,
                                const std::vector<CacheItem>& items) override;
  std::vector<Status> CacheFrom(const std::string& node,
                                const std::vector<CacheSrc>& srcs) override;
  std::vector<Status> RangeMany(const std::string& node,
                                const std::vector<BlockKey>& keys,
                                uint64_t offset, uint64_t length,
                                std::vector<std::string>* outs) override;
  std::vector<Status> ExistMany(const std::string& node,
                                const std::vector<BlockKey>& keys,
                                std::vector<char>* exists) override;
  std::vector<Status> RangeInto(const std::string& node,
                                const std::vector<BlockKey>& keys,
                                const std::vector<RangeDst>& dsts,
                                size_t header_size,
                                std::vector<std::string>* hdrs) override;
  // Scatter-gather overrides: one wire SEND/RECV gathers/scatters N payload
  // segments per key via multi-SGE work requests (additive zero-copy datapath
  // for dfkv_batch_put_sg / dfkv_batch_get_auto_sg).
  std::vector<Status> CacheFromMulti(
      const std::string& node,
      const std::vector<CacheSrcMulti>& srcs) override;
  std::vector<Status> RangeIntoMulti(
      const std::string& node, const std::vector<BlockKey>& keys,
      const std::vector<RangeDstMulti>& dsts, size_t header_size,
      std::vector<std::string>* hdrs, std::vector<size_t>* out_lens) override;

 private:
  struct Conn;
  // force_new (set on a retry after a stale pooled conn) skips the pool and
  // bootstraps a fresh connection, so a second server-reclaimed conn can't be
  // handed back on the retry. *from_pool reports whether a pooled conn was used.
  // lane selects a POOL: kData carries payloads (Cache/Range, up to MBs),
  // kControl carries only key-sized ops (Exist/Remove/Members). They never
  // share a connection, so a lookup storm can't queue behind in-flight 1 MB
  // GETs on the same QP — the hot-round exist p99 of ~800 s the phase-8
  // hit-rate probe measured was exactly that head-of-line blocking.
  enum class Lane { kData, kControl };
  Conn* Acquire(const std::string& node, Lane lane, bool* from_pool,
                bool force_new = false);
  void Release(const std::string& node, Lane lane, Conn* c);
  void Destroy(Conn* c);
  Status RoundTrip(const std::string& node, WireOp op, const BlockKey& key,
                   uint64_t offset, uint64_t length, const void* payload,
                   uint64_t payload_len, std::string* out);

  std::mutex mu_;
  std::unordered_map<std::string, std::vector<Conn*>> pool_;
  // Separate idle-conn pool for control-lane ops (Exist/Remove/Members), so
  // small key-only round trips never share a QP with payload transfers.
  std::unordered_map<std::string, std::vector<Conn*>> control_pool_;
  // Caller memory regions to register on every connection (the host KV pool).
  // Guarded by mu_; snapshotted in Acquire and registered on the connection.
  std::vector<std::pair<void*, size_t>> pools_;
  // Lifetime per-rail device refs holding the pool MRs registered at
  // RegisterMemory time (client-side anchor; see RegisterMemory). Filled once.
  std::vector<std::unique_ptr<rdma::RcEndpoint>> anchors_;
  // min over rails of (negotiated max_sge) - 1; set once in the ctor.
  size_t sg_payload_segs_ = 29;
  size_t max_payload_;
  // DCP1 declared max block bytes (DFKV_RDMA_MAX_BLOCK_BYTES, clamped to
  // max_payload_). 0 = undeclared: worst-case buffers both sides, old wire
  // behavior. When set it also shrinks OUR control buffers and bounds ops
  // client-side (oversize -> kInvalid before touching the wire, because the
  // server sized this conn's recv buffers to the declaration and a bigger
  // send would hard-break the QP).
  uint64_t declared_ = 0;
  size_t OpBound() const {  // per-op payload bound honoring the declaration
    return declared_ ? static_cast<size_t>(declared_) : max_payload_;
  }
  // Largest block this client has actually handed to the transport. Sizing
  // DFKV_RDMA_MAX_BLOCK_BYTES is otherwise guesswork: the declaration decides
  // how much the SERVER pins per connection (qd x (ValueHeader + declared)),
  // which at qd=32 is ~1 GiB per connection measured on a B200 node -- yet the
  // only figure available to an operator was the average transfer size. This
  // reports the actual high-water mark so the declaration can be set from
  // evidence with a deliberate margin.
  mutable std::atomic<uint64_t> max_block_seen_{0};
  mutable std::atomic<uint64_t> oversize_rejects_{0};
  // Records n as a candidate high-water mark and reports whether it exceeds the
  // declaration. Returns true for an oversized block (caller marks it kInvalid).
  bool NoteBlock(size_t n) const;
  size_t control_cap_;
  size_t depth_;
  int connect_ms_ = 3000;             // bootstrap TCP connect timeout (DFKV_RDMA_CONNECT_MS)
  int io_ms_ = 10000;                 // bootstrap TCP IO timeout (DFKV_RDMA_IO_MS)
  // Per-window completion timeout for the datapath (DFKV_RDMA_OP_TIMEOUT_MS).
  // WaitComp default is -1 (block forever): a connection whose completions are
  // delayed (RC retransmit storm, congested NIC, a peer reclaimed mid-op) would
  // otherwise hang the whole batch for tens of seconds, freezing a wait_complete
  // scheduler until a liveness probe kills it. A finite timeout bounds that wait;
  // the batch then tears the conn down and retries once on a fresh one (see the
  // 2-attempt loop in every batch op, mirroring RoundTrip). 0/negative => -1
  // (legacy block-forever). Normal ops finish in ms, so 5s never false-aborts.
  int op_timeout_ms_ = 5000;          // datapath completion timeout (DFKV_RDMA_OP_TIMEOUT_MS)
  // Batch-window override (DFKV_RDMA_BATCH_OP_TIMEOUT_MS, only the multi-op
  // Batch*/CacheFrom/RangeInto windows). <=0 = follow op_timeout_ms_. Lowering
  // it bounds a batch's tail at the price of tearing the connection (all ops of
  // the window report kIOError = miss, peer enters cooldown) when a server is
  // merely slow -- an explicit latency-vs-hit-rate trade for wait_complete
  // schedulers; leave unset for default behavior.
  int batch_op_timeout_ms_ = 0;
  int BatchTimeout() const { return batch_op_timeout_ms_ > 0 ? batch_op_timeout_ms_ : op_timeout_ms_; }
  size_t pool_max_ = 256;             // idle conns kept per node (DFKV_RDMA_POOL_MAX)
  std::vector<std::string> devs_;     // RDMA devices (multi-rail); "" = first
  std::vector<int> dev_node_;         // NUMA node per devs_ entry (-1 unknown)
  std::atomic<size_t> rr_{0};         // round-robin selector across devs_
  // observability (relaxed): connections opened total + per-rail breakdown +
  // declared MR regions. Connection opens are infrequent (pooled), off the op path.
  std::atomic<uint64_t> conns_opened_{0};
  std::atomic<uint64_t> mr_regions_{0};
  std::unique_ptr<std::atomic<uint64_t>[]> rail_conns_;  // sized to devs_.size()
};

}  // namespace dfkv

#endif  // DFKV_RDMA_TRANSPORT_H_
