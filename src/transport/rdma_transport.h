/* RDMA client transport — native v2 libibverbs RC. Requests and bounded
 * responses use 32,786-byte SEND/RECV control buffers (18-byte prefix plus a
 * 32-KiB Members payload); PUT uses WRITEs and every GET uses dynamic READ grants.
 * A peer that cannot negotiate the dynamic-only v2 contract is rejected.
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
#include <functional>
#include <memory>
#include <map>
#include <optional>
#include <set>
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
#include "transport/rdma_wc_status.h"
#include "transport/remote_rail_health.h"
#include "utils/log_throttle.h"

namespace dfkv {
namespace rdma {
class DeviceRef;
class RcEndpoint;
class RdmaTopology;
class PortWatcher;
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
  // Test binaries only: honour the DFKV_RDMA_TEST_* fault hooks. Without
  // this call they are never read.
  static void EnableTestHooks();

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
  bool TakeUnreachableRefusal() override;
  Status Retain(const std::string& node, uint64_t tenant_hash, uint64_t count,
                const std::string& payload, std::string* out) override;
  Status RemovedWithin(const std::string& node, uint64_t tenant_hash,
                       uint64_t count, const std::string& payload,
                       std::string* out) override;
  Status RemovedWithinAges(const std::string& node, uint64_t tenant_hash,
                           uint64_t count, const std::string& payload,
                           std::string* out) override;
  Status Members(const std::string& node, std::string* out) override;

  bool RegisterMemory(void* base, size_t size) override;
  std::string MetricsText() const override;  // dfkv_rdma_client_* (conns, per-rail)

  // Adaptive resource budget: baseline connection demand is nodes x two pools
  // x max(pool_max_, configured rails). The temporary data burst allowance is
  // opportunistic within that process budget, not a reason to raise it.
  // The budget follows the adopted ring and raises (never shrinks) every
  // derived dimension when the ring outgrows the current limit. Disabled when
  // the operator pins any budget env explicitly.
  void OnTopologyHint(size_t nodes) override;
  void OnPeerTopology(const PeerTopology& topology) override;
  void OnPeerIdentities(
      const std::vector<std::string>& live_peer_ids) override;

  bool pipelined() const override { return true; }
  size_t MaxSgPayloadSegs() const override { return sg_payload_segs_; }
  uint64_t MaxBlockBytes() const override { return OpBound(); }
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
                                std::vector<char>* exists,
                                std::string* out_dev = nullptr) override;
  std::vector<Status> RangeInto(
      const std::string& node, const std::vector<BlockKey>& keys,
      const std::vector<RangeDst>& dsts,
      std::vector<uint64_t>* value_lens) override;
  // Scatter-gather overrides: one wire SEND/RECV gathers/scatters N payload
  // segments per key via multi-SGE work requests (additive zero-copy datapath
  // for dfkv_batch_put_sg / dfkv_batch_get_auto_sg).
  std::vector<Status> CacheFromMulti(
      const std::string& node,
      const std::vector<CacheSrcMulti>& srcs,
      std::string* out_dev = nullptr) override;
  std::vector<Status> RangeIntoMulti(
      const std::string& node, const std::vector<BlockKey>& keys,
      const std::vector<RangeDstMulti>& dsts,
      std::vector<size_t>* out_lens, std::string* out_dev = nullptr) override;

 private:
  friend class RdmaTransportTestPeer;
  struct Conn;
  using RailMask = std::vector<uint8_t>;
  enum class AcquireFailure : uint8_t {
    kNone,
    kNoCompatibleRail,
    kAdmission,
    kLocalRail,
    kEndpoint,
    // The peer's TCP bootstrap address did not accept a connection. That is
    // about the node, not about a rail: no rail is charged and nothing is
    // retried, since every rail dials the same address. The operation fails
    // with kIOError, which is what cools the node in the client above. Also
    // returned without dialing while the address is held as unreachable.
    kPeerUnreachable,
    // The peer accepted the connection and closed it before sending anything,
    // on a first attempt. A server does that when it is at its limit of
    // pending handshakes, and also when it rejects the device. No rail is
    // charged; the operation waits a moment and tries once more, on another
    // compatible rail if there is one and on the same rail otherwise. A
    // close on that second attempt is charged as an endpoint failure.
    kPeerBusy,
  };
  enum class ReplaySafety : uint8_t {
    kReplaySafe,
    kUnsafeAfterPost,
  };
  // State of one logical operation across its attempts. The operation owns
  // it and passes it to every Acquire it makes.
  struct OpState {
    // The rail whose bootstrap the peer closed before replying (device frame
    // sent) on this operation's first attempt.
    std::optional<size_t> busy_closed_rail;
    // Until when the operation may wait for rail credits, all its Acquires
    // together; set by the first one (0 = not yet).
    uint64_t admission_deadline_us = 0;
    // The operation's own deadline (0 = none): connection setup, every wait
    // and a retry get only what is left of it. A batch sets it per attempt,
    // for the attempt's connection; its windows keep their own timeouts.
    uint64_t deadline_us = 0;
  };
  // A deadline timeout_ms from now, or 0 (none) for a timeout below 1.
  static uint64_t DeadlineIn(int timeout_ms);
  // What is left of the operation's deadline, at most cap_ms (cap_ms < 0: no
  // cap). Without a deadline, cap_ms.
  static int RemainingMs(const OpState* op, int cap_ms);
  struct AcquireOptions {
    bool force_new = false;
    OpState* op = nullptr;
    // A dial that only checks whether the peer refuses a rail's device
    // (ConfirmRefusal): its close is not counted as the peer being busy.
    // It also waits at most kRefusalCheckMs (and no longer than the
    // operation has left) to connect and for each reply
    // (a timeout confirms nothing), and charges no remote rail whatever
    // happens: only ChargeRefusal does.
    bool refusal_check = false;
    size_t requested_credits = 1;
    size_t required_data_bytes = 0;
    // Ask for the leased-PUT datapath on this connection: objects above the
    // inline threshold arrive via per-op staging leases, so the connection
    // geometry no longer follows the largest object. Honored only when the
    // peer advertises the capability; the request bit is then echoed on the
    // bootstrap frame.
    bool request_leased_put = false;
    // Dynamic-only GET demand has no connection-resident payload. Logical
    // slice length still travels in the operation request.
    bool request_dynamic_pull = false;
    size_t leased_inline_bytes = 0;
    RailMask excluded;
    std::shared_ptr<const rdma::PeerRailSnapshot> peer;
    // Set by Acquire itself when this process could not open an endpoint on
    // the rail it chose: try the other rails, on an idle pooled connection
    // only (no descriptor to open another), and without trying again.
    bool pooled_only = false;
    bool local_resource_fallback = true;
    // How long to wait for rail credits; unset = what is left of the
    // operation's admission budget (AdmissionWait).
    std::optional<uint64_t> backpressure_us;
  };
  struct AcquireResult {
    Conn* conn = nullptr;
    bool from_pool = false;
    std::optional<size_t> attempted_rail;
    AcquireFailure failure = AcquireFailure::kAdmission;
    Status status = Status::kIOError;
    // The peer closed the bootstrap after our device frame, before replying.
    bool device_frame_closed = false;
  };
  // Data operations share one endpoint pool because Acquire gives one logical
  // operation exclusive connection ownership until Release. The enum retains
  // scalar/SG identity for low-cardinality active/idle metrics. Control traffic
  // keeps an independent bounded-response pool.
  enum class Lane { kData, kSgData, kControl };
  AcquireResult Acquire(const std::string& node, Lane lane,
                        const AcquireOptions& options);
  // Schedules at most one fresh retry. Operations that may commit remotely
  // are retryable only until their first request is posted. cross_rail_retry
  // is set only while the failed rail stays excluded and another currently
  // topology-compatible, local-eligible, remotely healthy rail exists.
  bool PrepareRetry(
      int attempt, bool from_pool, std::optional<size_t> attempted_rail,
      AcquireFailure failure, ReplaySafety replay_safety, bool request_posted,
      const std::shared_ptr<const rdma::PeerRailSnapshot>& peer,
      const std::string& node, RailMask* excluded, bool* cross_rail_retry);
  // A new connection to a peer on `rail`. If this operation's first attempt
  // was closed on another rail, that rail may refuse our device: it is
  // dialed again (ConfirmRefusal) and charged only if closed again.
  void NoteConnected(const AcquireOptions& options, const std::string& node,
                     const std::string& health_id, size_t rail);
  // The wait for rail credits this Acquire may take: what is left of the
  // operation's budget (rail_backpressure_us_ from its first Acquire).
  uint64_t AdmissionWait(const AcquireOptions& options) const;
  void ConfirmRefusal(const std::string& node, const std::string& health_id,
                      size_t rail, uint64_t deadline_us);
  // Cools `rail` of the peer for a confirmed refusal unless it is the
  // peer's last usable rail. True if charged.
  bool ChargeRefusal(const std::string& node, const std::string& health_id,
                     size_t rail);
  static constexpr int kRefusalCheckMs = 300;
  // The identity remote rail health keys a peer by (see Conn::health_id).
  static std::string HealthIdLabel(const std::string& health_id) {
    return RemoteRailHealth::IsAddressIdentity(health_id) ? health_id.substr(1)
                                                         : health_id;
  }
  static std::string HealthId(const std::string& peer_id,
                              const std::string& node) {
    return peer_id.empty() ? RemoteRailHealth::AddressIdentity(node) : peer_id;
  }
  void Release(const std::string& node, Lane lane, Conn* c,
               RemoteRailOutcome remote_outcome =
                   RemoteRailOutcome::kSuccess,
               std::optional<uint64_t> application_use_us =
                   rdma::RailPolicy::NowMicros());
  // Returns true when the connection was a stale pooled one: taken from the
  // pool, failed by its peer before any completion succeeded on it since.
  bool Destroy(Conn* c,
               rdma::RailCompletion completion =
                   rdma::RailCompletion::kAdmission);
  // Counts and logs the failed work completion, if any, that the endpoint of
  // a connection about to be destroyed reaped. Observation only: it reads the
  // verdict the caller already reached and changes no retry or health state.
  // peer_rail_charged: the failure counts against the peer's rail.
  void NoteFailedCompletion(Conn& c, rdma::RailCompletion completion,
                            bool peer_rail_charged);
  void MarkActive(Conn* c, Lane lane);
  void MarkInactive(Conn* c);
  void MarkLive(Conn* c);
  void MarkDead(Conn* c);
  void CompleteRemote(const std::string& peer_id, size_t local_rail,
                      uint64_t generation, RemoteRailOutcome outcome);
  void RetireIdlePeerRail(const std::string& peer_id, size_t local_rail);
  static int QpAccessFlags(const Conn* conn);  // tests
  // Every idle connection on a local rail (its port went down).
  void RetireIdleRail(size_t local_rail);
  void RetireIdleIf(const std::function<bool(const Conn*)>& match);
  // Follows each rail's port state: a rail whose port is not ACTIVE is out
  // of selection until it is again (DFKV_RDMA_PORT_CHECK_MS).
  void CheckPorts();
  // TRN-29 (docs/design/rdma-device-reset.md), on the background thread:
  // reads the devices' async events (counted; with DFKV_RDMA_DEVICE_REBUILD,
  // DEVICE_FATAL retires the device's binding) and, with the switch, takes a
  // rail whose device generation moved out of selection and rebuilds it.
  void CheckDevices();
  void MarkRailStale(size_t rail);
  // One attempt; false if it could not run or failed (retried later).
  bool RebuildRail(size_t rail, uint64_t generation);
  // With the switch: the rail's device generation moved since it was built.
  bool RailStale(size_t rail) const;
  // A recovery probe for each rail whose cooldown has elapsed: a keepalive
  // on that rail to a node this client has connected to, trying up to
  // kProbeNodes of them. Runs on the keepalive thread; a dial there takes up
  // to DFKV_RDMA_CONNECT_MS, during which port checks and keepalives wait.
  void ProbeRails();
  static constexpr size_t kProbeNodes = 4;
  // A latency sample per 64 KiB moved, so a rail is not judged by the size
  // of what it happened to carry. The fixed cost of an operation is divided
  // too, so a rail that carries large values looks somewhat faster than one
  // that carries small ones.
  static uint64_t SampleLatency(uint64_t latency_us, uint64_t bytes) {
    constexpr uint64_t kUnit = 64 * 1024;
    return bytes <= kUnit ? latency_us : latency_us * kUnit / bytes;
  }
  void CompleteRemoteLease(Conn* c, RemoteRailOutcome outcome);
  void RecordRailTransfer(Conn* c, bool put, uint64_t operations,
                          uint64_t bytes);
  bool EvictOneIdle();
  // Trim burst-only idle endpoints even when keepalives are disabled. Only
  // application use renews retention; keepalive traffic cannot preserve a
  // burst's QPs indefinitely.
  void TrimIdle(uint64_t now_us);
  void MaintainIdle(uint64_t now_us, bool send_keepalives);
  void KeepaliveLoop();
  bool KeepaliveConn(Conn* c, rdma::RailCompletion* failure);
  enum class PullMode { kWholeObject, kSlice };
  struct PullRequest {
    const BlockKey* key = nullptr;
    const RangeDstSegment* segments = nullptr;
    size_t segment_count = 0;
    size_t capacity = 0;
    PullMode mode = PullMode::kWholeObject;
    uint64_t offset = 0;
    uint64_t length = 0;
    Status status = Status::kIOError;
    uint64_t value_len = 0;
    uint64_t data_len = 0;
    // kInvalid set here because the stored value is larger than capacity.
    bool too_large = false;
    // Whole object read only at exactly `capacity` bytes: a value of
    // another length is released without a READ (see RangeDst::exact).
    bool exact = false;
  };
  // One exclusively owned endpoint executes prepare, READ and RELEASE windows.
  // Failure leaves transient MRs with the endpoint until the caller fences it.
  // *timed_out (when given) tells a failure that ran out of time.
  bool PullWindow(Conn* conn, PullRequest* requests, size_t width,
                  int timeout_ms, rdma::RailCompletion* failure,
                  bool* timed_out = nullptr);
  // Whole objects of several keys into the callers' destinations, in windows
  // of the connection's depth: one connection, and each window's requests,
  // READs and releases in flight together, instead of one key at a time.
  // items[i] is the index into keys/statuses of the i-th request.
  struct PullItem {
    size_t index = 0;
    const RangeDstSegment* segments = nullptr;
    size_t segment_count = 0;
    size_t capacity = 0;
    bool exact = false;  // a value of another length is not READ (see PullInto)
  };
  void PullIntoMany(const std::string& node, const std::vector<BlockKey>& keys,
                    const std::vector<PullItem>& items, Lane lane,
                    std::vector<Status>* statuses,
                    std::vector<uint64_t>* value_lens, std::string* out_dev);
  Status PullInto(const std::string& node, const BlockKey& key,
                  const RangeDstSegment* segments, size_t segment_count,
                  size_t capacity, Lane lane, uint64_t* value_len,
                  std::string* out_dev = nullptr,
                  PullMode mode = PullMode::kWholeObject,
                  uint64_t offset = 0, uint64_t length = 0,
                  uint64_t* data_len = nullptr, bool exact = false);
  Status RoundTrip(const std::string& node, WireOp op, const BlockKey& key,
                   uint64_t offset, uint64_t length, const void* payload,
                   uint64_t payload_len, std::string* out,
                   uint64_t* value_len = nullptr);
  // Probe required base capabilities and report actual optional peer bits.
  enum class ProbeResult : uint8_t {
    kSupported,
    kUnreachable,  // no TCP connection
    kClosed,       // connected, then closed by the peer without a reply
    kRejected,     // timed out, or answered without the required capabilities
  };
  ProbeResult ProbeV2(const std::string& node,
                      bool* leased_put_supported = nullptr,
                      int connect_ms = 0, int io_ms = 0,  // 0: defaults
                      bool* session_fd_supported = nullptr,
                      bool* read_depth_supported = nullptr) const;
  // A buffer of the caller could not be registered on `c`. Marks the
  // connection so that its teardown charges neither rail nor peer.
  void NoteLocalRegistrationFailure(Conn* c, size_t bytes);
  mutable std::mutex mu_;
  // Scalar and SG operations share data endpoints. An acquired connection is
  // never concurrently reused, while operation framing remains self-describing.
  std::unordered_map<std::string, std::vector<Conn*>> pool_;
  // Only an identified publication can share capability observations across
  // new QPs. Static/unpublished peers probe each new QP; warm QPs carry their
  // own negotiated capabilities. Endpoint failure and forced recovery discard
  // cached observations, even if membership has not published a new epoch.
  struct PeerCapability {
    std::string peer_id;
    uint64_t publication = 0;
    bool leased_put = false;
    bool session_fd = false;
    bool read_depth = false;
  };
  std::unordered_map<std::string, PeerCapability> peer_capabilities_;
  void InvalidateCapabilities(const std::string& node,
                              const std::string& peer_id,
                              uint64_t publication);
  // Exist/Remove/Members remain isolated from payload transfers.
  std::vector<size_t> IdleDataBounds(const std::string& node) const;
  std::vector<size_t> IdleDataDepths(const std::string& node) const;
  std::unordered_map<std::string, std::vector<Conn*>> control_pool_;
  // Last successfully published caller memory declarations (under mu_).
  // RegisterMemory stages every rail without mu_ and commits and publishes
  // under it, so Acquire observes either the complete old generation or the
  // complete new one, never a partial growth.
  std::vector<std::pair<void*, size_t>> pools_;
  // One RegisterMemory at a time, and the destructor waits for it. Taken
  // before mu_, never while holding it. The destructor waits only for a call
  // already in flight: a RegisterMemory that starts once destruction has
  // begun is invalid, as is any other call on a transport being destroyed.
  std::mutex register_mu_;
  // Lifetime endpoint per active rail (under register_mu_). Its cache
  // reference anchors the newest successful generation; old generations
  // survive only while an older active connection still owns a cache
  // reference.
  std::vector<std::unique_ptr<rdma::RcEndpoint>> anchors_;
  // min over rails of (negotiated max_sge) - 1; set once in the ctor.
  size_t sg_payload_segs_ = 29;
  size_t max_payload_;
  // Logical per-object safety bound. Connection receive geometry is selected
  // independently from the current operation's actual largest object.
  // DFKV_RDMA_MAX_BLOCK_BYTES remains the deterministic hard rejection limit.
  uint64_t declared_ = 0;
  // Smallest data connection declaration. Larger requests round up to the next
  // power-of-two class, capped by declared_. Default 256 KiB.
  size_t connection_min_block_bytes_ = 0;
  size_t OpBound() const {
    return declared_ ? static_cast<size_t>(declared_) : max_payload_;
  }
  size_t ConnectionBound(size_t required) const;
  size_t ConnectionDepth(const std::string& node, Lane lane,
                         size_t requested_credits);
  // Largest block this client has actually handed to the transport.
  mutable std::atomic<uint64_t> max_block_seen_{0};
  mutable std::atomic<uint64_t> oversize_rejects_{0};
  // Leased-PUT in-flight datapath. Zero disables the optional capability
  // request and keeps every object on connection-resident receive slots.
  size_t inline_put_max_bytes_ = 4194304;  // DFKV_RDMA_INLINE_PUT_MAX_BYTES
  mutable std::atomic<uint64_t> leaseput_ops_{0};
  mutable std::atomic<uint64_t> leaseput_path_fallbacks_{0};
  // Records n as a candidate high-water mark and reports whether it exceeds the
  // logical bound. Returns true for an oversized block.
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
  // Unset POOL_MAX: retain up to 64 data endpoints during recurring bursts,
  // dropping endpoints unused by applications for 60 s until eight remain.
  // A positive explicit POOL_MAX fixes both data limits (no silent expansion).
  // Control always retains at most pool_max_. All endpoints also consume the
  // process-wide ResourceBudget, which can evict idle entries for admission.
  size_t pool_max_ = 8;
  size_t data_pool_max_ = 64;
  static constexpr uint64_t kBurstIdleUs = 60'000'000;
  // Enabled by default below the recommended 30 s server reaper interval.
  // Set DFKV_RDMA_KEEPALIVE_MS=0 to disable.
  int keepalive_ms_ = 15000;
  // DFKV_RDMA_SESSION_FD (default off): ask servers that offer it to keep
  // each connection's bootstrap socket as a session channel, and check it
  // before reusing a pooled connection. A server that ended the connection
  // (idle reclaim, pool pressure, restart, stop) has closed its end, so the
  // dead connection is dropped instead of costing the next operation RC
  // retry exhaustion. One more descriptor per pooled connection.
  bool session_fd_enabled_ = false;
  bool pipelined_reads_ = false;  // DFKV_RDMA_PIPELINED_READS
  // Pooled connections dropped before reuse because the server had closed
  // their session channel.
  std::atomic<uint64_t> session_closed_reaps_{0};
  std::atomic<uint64_t> pipelined_read_conns_{0};
  // True if conn's session channel says the server has ended it.
  static bool SessionClosed(const Conn& conn);
  bool busy_poll_ = false;  // DFKV_RDMA_BUSY_POLL
  static constexpr int kDefaultSpinUs = 0;  // 50 recommended; see DEPLOY
  uint32_t spin_us_ = kDefaultSpinUs;  // DFKV_RDMA_SPIN_US
  int port_check_ms_ = 1000;  // DFKV_RDMA_PORT_CHECK_MS; 0 = off
  std::unique_ptr<rdma::PortWatcher> port_watcher_;
  std::vector<uint8_t> port_active_;  // keepalive thread only
  std::atomic<uint64_t> port_down_events_{0};
  std::atomic<uint64_t> rails_port_down_{0};
  std::atomic<uint64_t> rail_background_probes_{0};
  std::mutex known_nodes_mu_;
  std::set<std::string> known_nodes_;  // nodes connected to (ProbeRails)
  // TRN-29. device_rebuild_: DFKV_RDMA_DEVICE_REBUILD. Per rail: the device
  // generation its connections and anchor were built on; whether it is out
  // of selection for a rebuild; DEVICE_FATAL / Open ENODEV-EIO signals
  // (suspects, 2 per rail); rebuilds. The rest: background thread only.
  bool device_rebuild_ = false;
  std::unique_ptr<std::atomic<uint64_t>[]> rail_generation_;
  std::unique_ptr<std::atomic<uint8_t>[]> rail_stale_;
  std::unique_ptr<std::atomic<uint64_t>[]> device_suspects_;
  std::unique_ptr<std::atomic<uint64_t>[]> device_rebuilds_;
  std::vector<uint64_t> rebuild_not_before_us_;
  std::vector<uint64_t> rebuild_backoff_ms_;
  mutable std::mutex device_events_mu_;
  std::map<std::pair<std::string, std::string>, uint64_t> device_events_;
  // Tests: a rebuild attempt fails as if the device did not open; the
  // background thread does not look at the devices.
  std::atomic<bool> rebuild_fails_for_test_{false};
  std::atomic<bool> device_checks_paused_for_test_{false};
  std::atomic<bool> keepalive_stop_{false};
  std::condition_variable keepalive_cv_;
  std::mutex keepalive_mu_;
  std::thread keepalive_thread_;
  std::shared_ptr<rdma::ResourceBudget> resource_budget_;
  int resource_acquire_ms_ = 10000;
  // Autoscale state: enabled unless any budget env was pinned explicitly.
  // topology_nodes_ dedups repeated hints so steady-state adoptions are free.
  bool budget_autoscale_enabled_ = true;
  std::atomic<size_t> topology_nodes_{0};
  // Learned per-node negotiated queue depth. The server clamps depth by its
  // receive-segment policy (large data slots often clamp to 1); the client
  // only discovers this after Open() already sized QP/WR/slot resources at
  // depth_. First contact pays full depth_; every later connection to that
  // node opens and budgets at the learned value, so churned reconnects stop
  // reserving depth_ x slot bytes they can never use. Data and SG lanes share
  // slot geometry (declared_); control negotiates its own. Values only learn
  // downward within a process lifetime — a server that raises its depth is
  // picked up by new client processes, not running ones (documented).
  std::mutex depth_mu_;
  std::unordered_map<std::string, uint16_t> learned_data_depth_;
  std::unordered_map<std::string, uint16_t> learned_control_depth_;
  uint16_t LearnedDepth(const std::string& node, Lane lane);
  void NoteNegotiatedDepth(const std::string& node, Lane lane,
                           uint16_t remote_depth);
  // Admission (budget) starvation observability: ops that failed because THIS
  // process ran out of transport budget. Logged with a coarse throttle.
  std::atomic<uint64_t> admission_failures_{0};
  // Connections whose clamped negotiation refunded WR/registered budget.
  std::atomic<uint64_t> depth_refunds_{0};
  std::unique_ptr<rdma::RdmaTopology> topology_;
  using ConnectionClass = std::pair<size_t, size_t>;  // block bytes, QP depth
  struct ConnectionClassStats {
    uint64_t opened = 0;
    uint64_t active = 0;
  };
  void MarkClassOpened(Conn* c);
  mutable std::mutex connection_class_mu_;
  std::map<ConnectionClass, ConnectionClassStats> connection_class_stats_;
  std::vector<std::string> devs_;  // stable discovered ACTIVE rail order
  std::vector<std::vector<uint8_t>> rail_tiers_;
  std::optional<size_t> preferred_rail_;
  bool peer_topology_required_ = false;
  std::unique_ptr<rdma::PeerTopologyStore> peer_topologies_;
  std::unique_ptr<RemoteRailHealth> remote_rail_health_;
  bool auto_device_ = true;
  bool numa_aware_ = false;
  // Global least-inflight rail admission with per-rail credits, failure
  // quarantine, recovery probes, and a bounded wait when all credits are busy.
  std::unique_ptr<rdma::RailPolicy> rail_policy_;
  // Per operation (DFKV_RDMA_RAIL_BACKPRESSURE_MS, default 200 ms, at most
  // the operation timeout).
  uint64_t rail_backpressure_us_ = 200'000;
  // observability (relaxed): connections opened total + per-rail breakdown.
  std::atomic<uint64_t> conns_opened_{0};
  std::atomic<uint64_t> v2_put_writes_{0};
  std::atomic<uint64_t> mr_regions_{0};
  std::atomic<uint64_t> mr_registered_bytes_{0};
  std::atomic<uint64_t> mr_registration_rejections_{0};
  mutable std::atomic<uint64_t> v2_probe_attempts_{0};
  mutable std::atomic<uint64_t> v2_probe_failures_{0};
  std::atomic<uint64_t> stale_pool_retries_{0};
  std::atomic<uint64_t> stale_pool_connections_{0};
  // Pause before the one retry after a kPeerBusy; a random part of up to the
  // same length is added per caller. A member so that tests can change it.
  uint64_t peer_busy_retry_us_ = 20000;
  // Addresses whose last TCP connect failed, and until when they are not
  // dialed again. Without it every operation and every round of the client's
  // background prober would wait out a connect timeout on a node that is
  // down, now that no rail is cooled for it. Operations on connections that
  // are already established do not look here.
  bool RecentlyUnreachable(const std::string& node);
  void NoteUnreachable(const std::string& node);
  std::mutex unreachable_mu_;
  std::unordered_map<std::string, uint64_t> unreachable_until_us_;
  uint64_t peer_unreachable_hold_us_ = 2000000;
  std::atomic<uint64_t> peer_unreachable_skips_{0};
  std::atomic<uint64_t> peer_unreachable_{0};
  std::atomic<uint64_t> peer_busy_{0};
  std::atomic<uint64_t> rail_refusals_{0};  // see NoteConnected
  // Operations that ran out of their deadline before a connection was ready.
  std::atomic<uint64_t> op_deadline_expired_{0};
  std::mutex refusal_mu_;
  // (peer, rail) pairs with a confirming dial in flight: one at a time.
  std::set<std::pair<std::string, size_t>> refusal_checks_;
  std::atomic<uint64_t> local_resource_failures_{0};
  // Endpoint opens refused for lack of descriptors or memory whose operation
  // was then served on another rail.
  std::atomic<uint64_t> local_resource_rail_fallbacks_{0};
  // Operations failed at once: every rail had an ENOMEM endpoint open
  // within kEnomemHoldUs.
  std::atomic<uint64_t> local_resource_held_{0};
  // Per rail: until when (NowMicros) it is not chosen after ENOMEM.
  static constexpr uint64_t kEnomemHoldUs = 2000000;
  std::unique_ptr<std::atomic<uint64_t>[]> enomem_until_us_;
  // Per rail: the device's context and PD, held from the rail's first
  // connection for the transport's lifetime, so that a pool that drains
  // (idle reclaim, failures) does not close the device and the next
  // connection reopen it (TRN-16: 11 ms more per open, 1.5 ms per close).
  // A transport that registered memory also holds them through its anchors.
  // An Open that fails with ENODEV or EIO (device gone or reset) drops the
  // rail's reference.
  std::mutex rail_devices_mu_;
  std::vector<std::unique_ptr<rdma::DeviceRef>> rail_devices_;
  // Tests: the errno an endpoint open on this rail fails with (0 = none).
  std::vector<int> fail_open_errno_for_test_;
  // Test seam: this many more bootstrap dials fail as a timeout would.
  mutable std::atomic<int> fail_dials_for_test_{0};
  int DialNode(const std::string& node, int connect_ms, int io_ms) const;
  std::atomic<uint64_t> cross_rail_retries_{0};
  std::atomic<uint64_t> cross_rail_retry_successes_{0};
  std::atomic<uint64_t> cross_rail_retry_exhausted_{0};
  std::atomic<uint64_t> peer_topology_updates_{0};
  std::atomic<uint64_t> no_compatible_rail_{0};
  std::atomic<uint64_t> stale_publication_reaps_{0};
  // Deterministic test-only completion-fault targeting. Inert unless
  // DFKV_RDMA_TEST_COMPLETION_FAULT is set.
  std::atomic<uint64_t> test_completion_fault_calls_{0};
  std::atomic<uint64_t> completion_timeouts_{0};
  std::atomic<uint64_t> pull_prepares_{0};
  std::atomic<uint64_t> pull_reads_{0};
  std::atomic<uint64_t> pull_read_bytes_{0};
  std::atomic<uint64_t> pull_read_wrs_{0};  // READ work requests posted
  std::atomic<uint64_t> test_pull_reads_{0};  // DFKV_RDMA_TEST_PULL_READ_FAILURE=@N
  // GET destinations outside registered memory: registrations made, merged
  // ranges refused and registered per segment, pulls (rate-limits the WARN).
  std::atomic<uint64_t> dest_registrations_{0};
  std::atomic<uint64_t> dest_span_fallbacks_{0};
  std::atomic<uint64_t> dest_memlock_fallbacks_{0};
  LogThrottle dest_log_;
  std::atomic<uint64_t> pull_failures_{0};
  std::atomic<uint64_t> pull_releases_{0};
  std::atomic<uint64_t> pull_release_failures_{0};
  std::atomic<uint64_t> keepalive_attempts_{0};
  std::atomic<uint64_t> keepalive_successes_{0};
  std::atomic<uint64_t> keepalive_failures_{0};
  std::unique_ptr<std::atomic<uint64_t>[]> active_lane_rail_;
  mutable std::mutex peer_connections_mu_;
  std::map<std::pair<std::string, size_t>, uint64_t> peer_connections_;
  std::unique_ptr<std::atomic<uint64_t>[]> rail_conns_;  // sized to devs_.size()
  std::unique_ptr<std::atomic<uint64_t>[]> rail_put_ops_;
  std::unique_ptr<std::atomic<uint64_t>[]> rail_put_bytes_;
  std::unique_ptr<std::atomic<uint64_t>[]> rail_get_ops_;
  std::unique_ptr<std::atomic<uint64_t>[]> rail_get_bytes_;
  std::atomic<uint64_t> endpoint_cache_hits_{0};
  std::atomic<uint64_t> endpoint_cache_misses_{0};
  std::atomic<uint64_t> endpoint_cache_evictions_{0};
  // Fixed-cardinality locality fallback reasons: caller NUMA unknown / no
  // enabled local rail. Endpoint failures are bounded per configured rail in
  // RailPolicy::Snapshot rather than labeled by arbitrary node addresses.
  std::atomic<uint64_t> numa_caller_unknown_fallbacks_{0};
  std::atomic<uint64_t> numa_no_local_fallbacks_{0};
  // Connections lost to a failed work completion, per local rail and status of
  // the first failed completion (rail * rdma::kWcStatusSlots + status slot).
  // Peers are deliberately not a dimension: their addresses are unbounded and
  // go to the throttled log line instead.
  std::unique_ptr<std::atomic<uint64_t>[]> rail_wc_errors_;
  // Bounds that log line; keyed by peer, rail, status and vendor syndrome.
  // Both are touched only when a connection dies, and are declared last so
  // the counters above, which operations update, keep their layout.
  LogThrottle completion_error_log_;
  LogThrottle local_resource_log_;  // keyed by what failed and its errno
};

}  // namespace dfkv

#endif  // DFKV_RDMA_TRANSPORT_H_
