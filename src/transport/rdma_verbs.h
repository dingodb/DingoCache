/* Native libibverbs RC endpoint shared by the dfkv RDMA client and server.
 * Built only when DFKV_WITH_RDMA is defined.
 *
 * Why native verbs (not librdmacm)? rdma_cm resolves the RDMA device from an IP
 * address (IPoIB), so it can only use a device that has an IP. On these clusters
 * the IP lives on the 200G port while the 8x 400G ports have NO IP and sit on a
 * SEPARATE IB fabric. Native verbs opens a device BY NAME (e.g. mlx5_0) and
 * bootstraps the QP over a tiny out-of-band TCP channel (which rides whatever IP
 * network the nodes share — 200G or bond0). Control plane and data plane are
 * thus decoupled: the QP handshake (LID/GID/QPN/PSN, ~32 B) goes over TCP, the
 * data rides the 400G fabric by LID. This is exactly perftest's non-`-R` mode.
 *
 * A connection uses RC SEND/RECV for v2 control frames and one-sided WRITEs for
 * payloads, with a ring of `depth` slots for pipelining. Completions are reaped
 * via a completion channel (blocking, no busy-spin). */
#ifndef DFKV_RDMA_VERBS_H_
#define DFKV_RDMA_VERBS_H_

#include <infiniband/verbs.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

#include "transport/dev_frame.h"
#include <utility>
#include <atomic>
#include <thread>
#include <functional>
#include <vector>

namespace dfkv {
namespace rdma {

// The QP transport retry settings (see QpRetry in rdma_verbs.cc).
struct QpRetrySettings {
  uint8_t timeout = 14;
  uint8_t retry_cnt = 7;
  uint8_t rnr_retry = 7;
};
// Read once per process; ReadQpRetry reads the environment now (tests).
const QpRetrySettings& QpRetry();
QpRetrySettings ReadQpRetry();

// QP connection info exchanged over the TCP bootstrap channel (fixed 32 bytes,
// little-endian). lid==0 selects GRH/GID routing; otherwise IB LID routing.
//
// Wire layout: qpn u32 | psn u32 | lid u16 | gid[16] | pad[6]. The frame length
// cannot grow without breaking bootstrap framing, so v2 reuses the historically
// zeroed pad: DPQ1 magic at [26,30), then a u16 receive depth whose high bit is
// the mandatory v2 marker. The depth is how many in-flight requests the peer's
// posted receives can absorb. Posting beyond it causes RNR retries and silent
// throughput collapse; zero, absent magic or a missing high bit is therefore a
// protocol mismatch, not a legacy default.
struct QpInfo {
  uint32_t qpn = 0;
  uint32_t psn = 0;
  uint16_t lid = 0;
  uint8_t gid[16] = {0};
  uint8_t pad[6] = {0};
  uint16_t depth = 0;  // not a standalone wire field: rides the pad (see above)
  uint8_t protocol_version = 0;  // v2 is encoded in depth's high bit
};
constexpr size_t kQpInfoBytes = 32;
constexpr uint32_t kQpDepthMagic = 0x31515044u;  // ASCII DPQ1 (LE)
void SerializeQpInfo(const QpInfo& in, char out[kQpInfoBytes]);
QpInfo ParseQpInfo(const char in[kQpInfoBytes]);

// Fixed-size device-name field the client sends first in the bootstrap so the
// server opens its QP on the matching device (same rail) for multi-rail setups.

// Target QP scatter-gather entries. SGE0 carries the request/response wire
// prefix; the remaining kMaxSge-1 carry raw-payload segments, so one
// scatter-group object may hold up to kMaxSge-1 (=29) non-contiguous buffers.
// Open() clamps this to the device's reported max_sge.
constexpr size_t kMaxSge = 30;

// The SG width Open() would negotiate on dev_name (min(kMaxSge, device
// max_sge)) WITHOUT bootstrapping a connection: rides the shared-device cache,
// so callers can size scatter-gather batches before any peer traffic. Empty
// dev_name = first device. Returns kMaxSge when no device answers — matching
// Open()'s "trust the documented cap" fallback keeps the client guard at its
// historical 29-payload-segs behavior instead of silently shrinking it.
size_t QueryMaxSge(const char* dev_name);

// TRN-29: recovering a rail after a device reset
// (docs/design/rdma-device-reset.md). DFKV_RDMA_DEVICE_REBUILD, default off,
// read once.
bool DeviceRebuildEnabled();
// The device name's generation: bumped by each RetireDevice. A shared
// context records the generation it was opened under. "" = first device.
uint64_t DeviceGeneration(const std::string& dev_name);
// Retires the name's binding if its generation is still `generation`: the
// next Open of the name opens a fresh context; the old one is closed with its
// last reference. False if it was retired already.
bool RetireDevice(const std::string& dev_name, uint64_t generation);
struct DeviceEvent {
  std::string device;
  std::string event;
  bool fatal = false;          // IBV_EVENT_DEVICE_FATAL
  uint64_t generation = 0;     // of the context it came from
};
// Drains the async events of every shared context (one reader at a time).
std::vector<DeviceEvent> PollDeviceEvents();
// Releases that failed on a context proven disassociated, and were dropped.
uint64_t DeadReleaseFailures();
// Tests: the next PollDeviceEvents reports DEVICE_FATAL for the device.
void InjectDeviceFatalForTest(const std::string& dev_name);
// Tests: faults for the release decision. `release` overrides the result
// of a release that succeeded (by operation name; 0 = keep), `confirm` what
// the confirming verb returns, `listed` whether the device is still listed.
struct ReleaseFaultsForTest {
  std::function<int(const char* operation)> release;
  std::function<int()> confirm;
  std::function<bool()> listed;
};
void SetReleaseFaultsForTest(ReleaseFaultsForTest faults);
// Applies the same fail-closed release policy as a live failed verb. In
// particular, this function does not discard the caller's MR on a normally
// failed deregistration: its caller owns recovery and retains the registry
// entry. Only a context proven disassociated permits that exceptional drop.
// Internal/test helper: callers supply the same context and error as the verb.
void RequireVerbsReleaseForTest(ibv_context* ctx, int result,
                                const char* operation);
// Tests: overrides DFKV_RDMA_DEVICE_REBUILD (1 on, 0 off, -1 the variable).
// A transport reads the switch when it is constructed.
void SetDeviceRebuildForTest(int enabled);
// Retired contexts still referenced. Diagnostic/test.
size_t RetiredDeviceCount();
// Alignment used for server-side O_DIRECT read buffers. NVMe/xfs are happy with
// 4096, and it is a safe superset for 512-byte logical-sector devices.
constexpr size_t kDirectIoAlign = 4096;
// Run one declaration attempt across every active rail. A failed Stage on rail
// N must not mutate that rail; the helper rolls back rails [0,N) in reverse
// order. Commit is called only after every Stage succeeds.
template <typename Stage, typename Commit, typename Rollback>
bool RunPoolMrTransaction(size_t rail_count, Stage stage, Commit commit,
                          Rollback rollback) {
  size_t staged = 0;
  for (; staged < rail_count; ++staged) {
    if (stage(staged)) continue;
    while (staged != 0) rollback(--staged);
    return false;
  }
  for (size_t rail = 0; rail < rail_count; ++rail) commit(rail);
  return true;
}


// A reference to an opened device's shared context and PD, kept apart from
// any endpoint: while one is held, closing the device's last endpoint does
// not close the context, and the next Open does not reopen it (about 11 ms
// to open, 1.5 ms to close). Move-only; released on destruction.
class DeviceRef {
 public:
  DeviceRef() = default;
  ~DeviceRef();
  DeviceRef(DeviceRef&& other) noexcept : ctx_(other.ctx_) {
    other.ctx_ = nullptr;
  }
  DeviceRef& operator=(DeviceRef&& other) noexcept;
  DeviceRef(const DeviceRef&) = delete;
  DeviceRef& operator=(const DeviceRef&) = delete;
  bool held() const { return ctx_ != nullptr; }

 private:
  friend class RcEndpoint;
  explicit DeviceRef(ibv_context* ctx) : ctx_(ctx) {}
  ibv_context* ctx_ = nullptr;
};

// One RC connection: device context + PD + CQ(+channel) + QP + a ring of
// `depth` send and recv buffers (each `cap` bytes, registered once).
class RcEndpoint {
 public:
  RcEndpoint() = default;
  ~RcEndpoint();
  RcEndpoint(const RcEndpoint&) = delete;
  RcEndpoint& operator=(const RcEndpoint&) = delete;

  // Open device `dev_name` (nullptr/"" = first device), create RC QP in INIT,
  // allocate+register `depth` send and receive buffers of `cap` bytes. When
  // direct_io_buffers is true, also allocate one 4096-aligned registered buffer
  // per slot for O_DIRECT reads/writes that are scatter-transferred without a
  // payload copy. direct_io_cap lets that buffer be larger than the ordinary
  // control buffers.
  bool Open(const char* dev_name, size_t cap, size_t depth, uint8_t ib_port = 1,
            bool direct_io_buffers = false, size_t direct_io_cap = 0);
  // Whether the peer may RDMA READ/WRITE through this QP (set before Open).
  // A server endpoint must: clients READ its staging and WRITE its lease
  // regions. A client endpoint is never the target of one (servers send
  // only, and no client rkey is ever on the wire), so the client clears it.
  void set_remote_access(bool allowed) { remote_access_ = allowed; }
  // RDMA READs this QP may have in flight as initiator (max_rd_atomic) and
  // accepts as responder (max_dest_rd_atomic); both 1 by default, set before
  // Connect and clamped to what the device supports. The initiator depth
  // must not exceed the peer's responder depth.
  void set_read_depth(uint8_t depth) { read_depth_ = depth; }
  void set_responder_read_depth(uint8_t depth) {
    responder_read_depth_ = depth;
  }
  // The depths the connected QP was given (ibv_query_qp); false if the
  // device cannot say.
  bool QueryReadDepths(int* initiator, int* responder) const;

  QpInfo Local() const { return local_; }              // my QP info (after Open)
  bool Connect(const QpInfo& remote);                  // INIT -> RTR -> RTS
  // Connect in two steps: a QP in RTR receives; its initiator settings
  // (READ depth) are fixed by the step to RTS, which may wait for the
  // peer's answer.
  bool ConnectRtr(const QpInfo& remote);
  bool ConnectRts();
  // The device's limits, read by Open (1 if it could not say).
  int device_responder_read_depth() const {
    return device_responder_read_depth_;
  }

  size_t depth() const { return depth_; }
  // The peer's required nonzero advertised depth clamps the local posting
  // window. Never pipeline more requests than the peer's posted receives, or
  // they RNR-retry and silently degrade the whole window.
  void set_remote_depth(uint16_t d) { remote_depth_ = d; }
  uint16_t remote_depth() const { return remote_depth_; }
  size_t window() const {
    return std::min<size_t>(remote_depth_, depth_);
  }
  size_t cap() const { return cap_; }
  int numa_node() const { return numa_node_; }  // device's NUMA node, or -1
  // Last completion timestamp (steady-clock us), updated by the owner Serve
  // loop whenever WaitComp yields completions. Lets the server evict the
  // stalest idle connection when the shared receive segment is exhausted.
  std::atomic<uint64_t> last_active_us_{0};
  char* sbuf(size_t slot) { return sbuf_[slot]; }
  char* rbuf(size_t slot) { return rbuf_[slot]; }
  char* dbuf(size_t slot) { return dbuf_.empty() ? nullptr : dbuf_[slot]; }
  size_t dbuf_cap() const { return dbuf_cap_; }
  ibv_mr* dmr(size_t slot) { return dmr_.empty() ? nullptr : dmr_[slot]; }

  bool PostRecv(size_t slot);                          // arm recv into rbuf_[slot]
  bool PostSend(size_t slot, size_t len);              // SEND sbuf_[slot][0,len)

  // Pool registration has an explicit stage/commit/rollback lifecycle so a
  // multi-rail declaration can be published atomically. A staged registration
  // is immediately usable only by this idle endpoint; Commit retires any
  // same-base generation it supersedes, while Rollback removes exactly the
  // cache/reference created by this attempt.
  struct PoolMrRegistration {
    uintptr_t base = 0;
    size_t size = 0;
    int access = 0;
    ibv_mr* mr = nullptr;
    bool added = false;
  };
  bool StagePoolMr(void* base, size_t size, bool remote_write,
                   PoolMrRegistration* registration);
  // A superseded generation no endpoint references any more is deregistered
  // here, or, with `retired`, handed back for DeregisterRetired once the
  // caller holds no lock.
  void CommitPoolMr(PoolMrRegistration* registration,
                    std::vector<ibv_mr*>* retired = nullptr);
  void RollbackPoolMr(PoolMrRegistration* registration);
  // Single-endpoint convenience wrapper, primarily used by server regions.
  bool AddPoolMr(void* base, size_t size, bool remote_write = false);
  // Atomically ensure all declarations on this endpoint. False leaves its
  // lookup cache exactly as it was before the call.
  bool EnsurePoolMrs(const std::vector<std::pair<void*, size_t>>& regions,
                     bool remote_write,
                     std::vector<ibv_mr*>* retired = nullptr);
  // Deregisters what CommitPoolMr or EnsurePoolMrs handed back. The device
  // must still be open: call it while the endpoint that handed them back
  // (or another on the same device) is.
  static void DeregisterRetired(std::vector<ibv_mr*>* retired);
  // This endpoint's own remote-writable region (a server connection's
  // receive slots): an exact MR, never shared with another endpoint, so its
  // rkey reaches nothing else. Deregistered by Close, after the QP.
  ibv_mr* RegisterOwnRegion(void* base, size_t size);
  // Exact per-operation remote WRITE capability. Never enters the pool cache:
  // a later lease at the same VA must not inherit an earlier operation's rkey.
  // The backing allocation must outlive this endpoint, unless explicitly
  // revoked with ReleaseLeaseWriteRegion before returning it to its pool.
  ibv_mr* RegisterLeaseWriteRegion(void* base, size_t size);
  void ReleaseLeaseWriteRegion(ibv_mr* mr);
  static uint64_t LeaseWriteMrActive();
  // Exact operation-scoped READ capability, revoked before its staging range
  // can return to the receive pool. Never retained in a broad chunk cache.
  ibv_mr* RegisterLeaseReadRegion(void* base, size_t size);
  void ReleaseLeaseReadRegion(ibv_mr* mr);
  static uint64_t LeaseReadMrActive();

  // Cumulative one-shot user MRs registered outside explicit pool regions.
  // These are never cached; TransientUserMrActive is the lifetime invariant.
  static uint64_t AdhocUserMrTotal();
  // Process-wide count of actual ibv_reg_mr calls for pool regions. With the
  // shared per-PD registry this counts distinct (device,region) pairs, not
  // connections (the pre-fix per-connection storm). Diagnostic/test.
  static uint64_t PoolMrRegistrations();
  // ibv_query_port calls made by Open (diagnostic: cached per device).
  static uint64_t PortQueries();
  // Test seam: called before each shared pool registration, outside any
  // lock; returning false fails that registration. Empty = none.
  static void SetPoolRegistrationSeamForTest(std::function<bool()> seam);
  // Test seam: called before a pool MR no endpoint references any more is
  // deregistered. Empty = none.
  static void SetPoolDeregistrationSeamForTest(std::function<void()> seam);
  // Test seam: called right after a pool MR is handed back to `retired`
  // (it may throw, as an allocation there could). Empty = none.
  static void SetPoolRetireSeamForTest(std::function<void()> seam);
  // The device generation the endpoint's context was opened under (TRN-29).
  uint64_t device_generation() const { return device_generation_; }
  // References this open endpoint's device; not held if it is not open.
  DeviceRef ReferenceDevice() const;
  // Process-wide references to opened devices (endpoints and DeviceRefs).
  // Diagnostic/test.
  static long DeviceReferences();
  // Process-wide device contexts opened (ibv_open_device). Diagnostic/test.
  static uint64_t DeviceOpens();
  static uint64_t PoolMrRegistrationFailures();
  // Test seam: consulted before each one-shot registration in
  // RegisterTransient (not a pool hit); false fails it.
  static void SetTransientRegistrationSeamForTest(
      std::function<bool(void* addr, size_t len)> seam);
  // Process-wide number of currently retained shared-PD pool MR generations.
  // Superseded generations remain only while an endpoint still references them.
  static uint64_t PoolMrActiveRegistrations();
  // Currently live one-shot out-of-pool registrations. Must return to its
  // baseline before a public transport call returns.
  static uint64_t TransientUserMrActive();
  // Process-wide CQ observations, incremented at the single WaitComp polling
  // seam. MetricsText exposes these without adding bookkeeping to each posting
  // path; errors include CQ poll/notification failures and failed WC statuses.
  static uint64_t CqCompletions();
  static uint64_t CqErrors();

  // Resolve a buffer only inside an explicitly registered pool region. Public
  // operation-owned buffers use RegisterTransient; an out-of-pool lookup fails
  // closed rather than creating a cached MR that could outlive its allocation.
  ibv_mr* RegisterUser(void* addr, size_t len);
  // One-shot MR for API-owned buffers whose lifetime ends or may move after the
  // current operation. kRemoteWrite is for targets a peer writes into;
  // kLocalWrite for this side's RDMA READ destinations; kSendSource keeps
  // const/read-only send sources valid. `remote_write` true/false means
  // kRemoteWrite/kSendSource. Registered pool MRs with sufficient access are
  // reused. The endpoint owns ad-hoc MRs until ReleaseTransient() or
  // Close(); an abandoned READ keeps its destination MR until QP destruction
  // fences DMA, while successful operations release after local completion.
  enum class TransientAccess { kSendSource, kLocalWrite, kRemoteWrite };
  ibv_mr* RegisterTransient(void* addr, size_t len, TransientAccess access);
  ibv_mr* RegisterTransient(void* addr, size_t len,
                            bool remote_write = true) {
    return RegisterTransient(addr, len,
                             remote_write ? TransientAccess::kRemoteWrite
                                          : TransientAccess::kSendSource);
  }
  void ReleaseTransient(ibv_mr* mr);
  // Scatter recv: first `hdr_bytes` of the message land in rbuf_[slot]
  // (response wire prefix); the raw payload lands directly in `payload`
  // (must belong to `payload_mr` from RegisterUser) — zero copy into the caller.
  bool PostRecvScatter(size_t slot, void* payload, size_t payload_len,
                       ibv_mr* payload_mr, size_t hdr_bytes);
  // Scatter send: SGE0 = wire prefix in sbuf_[slot], SGE1 = raw `payload`
  // (must belong to `payload_mr`). Used by client PUT and server GET without
  // copying the payload; payload_len==0 degrades to a prefix-only 1-SGE send.
  bool PostSendScatter(size_t slot, size_t hdr_len, const void* payload,
                       size_t payload_len, ibv_mr* payload_mr);

  // Multi-segment scatter/gather (one wire SEND/RECV gathers/scatters N payload
  // segments, in addition to the header SGE). Used by the additive scatter-gather
  // API (dfkv_batch_put_sg / dfkv_batch_get_auto_sg) so one dfkv key coalesces N
  // non-contiguous caller buffers without a client concat copy. Each segment must
  // belong to its corresponding MR (from RegisterUser / pool MR). num_sge =
  // 1 + segs.size(), which must be <= the QP's max_send_sge/max_recv_sge.
  // PostSendScatterMulti: SGE0 = sbuf_[slot][0,hdr_len), then one SGE per segment.
  bool PostSendScatterMulti(
      size_t slot, size_t hdr_len,
      const std::vector<std::pair<const void*, uint32_t>>& segs,
      const std::vector<ibv_mr*>& mrs);
  // PostRecvScatterMulti: SGE0 = rbuf_[slot][0,hdr_bytes), then one SGE per
  // destination segment (the NIC scatters the concatenated reply payload across
  // the segments in order, honoring each segment's length).
  bool PostRecvScatterMulti(
      size_t slot, const std::vector<std::pair<void*, uint32_t>>& segs,
      const std::vector<ibv_mr*>& mrs, size_t hdr_bytes);

  // One-sided primitives. PostWrite and WRITE_WITH_IMM are signaled; GET
  // uses PostRead, while PUT retains its WRITE operations.
  // Initiator-side one-sided READ. `destination_mr` belongs to this endpoint's
  // PD; the remote descriptor must remain leased until the signaled CQE is
  // terminal.
  bool PostRead(size_t slot, void* destination, size_t length,
                ibv_mr* destination_mr, uint64_t remote_addr,
                uint32_t remote_rkey);
  // One READ of a contiguous remote range into several local segments
  // (count <= ReadSge()).
  bool PostReadSg(size_t slot, const ibv_sge* sges, size_t count,
                  uint64_t remote_addr, uint32_t remote_rkey);
  // Local segments one READ may scatter into: the device's max_sge_rd, at
  // most the QP's send SGE count.
  size_t ReadSge() const { return std::max<size_t>(1, std::min(read_sge_, max_sge_)); }
  bool PostWrite(size_t slot, const void* source, size_t length,
                 ibv_mr* source_mr, uint64_t remote_addr,
                 uint32_t remote_rkey);
  bool PostWriteImm(size_t slot, size_t length, uint64_t remote_addr,
                    uint32_t remote_rkey, uint32_t immediate);
  bool PostWriteImmScatter(
      size_t slot, size_t header_len, const void* payload,
      size_t payload_len, ibv_mr* payload_mr, uint64_t remote_addr,
      uint32_t remote_rkey, uint32_t immediate);
  bool PostWriteImmScatterMulti(
      size_t slot, size_t header_len,
      const std::vector<std::pair<const void*, uint32_t>>& segs,
      const std::vector<ibv_mr*>& mrs, uint64_t remote_addr,
      uint32_t remote_rkey, uint32_t immediate);
  // Unsignaled scatter WRITE (no IMM): header from sbuf_[slot], payload from
  // caller. A following signaled PostSendNotify fences the source buffer.
  bool PostWriteScatter(
      size_t slot, size_t header_len, const void* payload,
      size_t payload_len, ibv_mr* payload_mr, uint64_t remote_addr,
      uint32_t remote_rkey);

  // QP scatter-gather capability negotiated in Open() = min(kMaxSge, device cap).
  // The SG datapath caps raw-payload segments at max_sge()-1 (SGE0 is the wire prefix).
  size_t max_sge() const { return max_sge_; }

  // One nonblocking CQ snapshot. Unlike WaitComp(..., 0), this never arms or
  // consumes completion-channel notifications; it is safe for ready-only
  // microbatch drains after WaitComp has delivered the first completion.
  int PollComp(ibv_wc* out, int max);

  // Block until at least one completion, drain up to `max` into out[]; returns
  // the count (>0), or <0 on error / when Wake() is called. wr_id of each wc =
  // the slot. Used for single round-trips (max=1) and pipelined batches.
  // timeout_ms < 0 blocks forever (the default, used by the client). A finite
  // timeout returns 0 if no completion arrives in that window — the server uses
  // it to reclaim an idle/abandoned connection instead of blocking indefinitely.
  // also_fd, when >= 0, is waited on too: its becoming readable ends the
  // wait like a timeout (returns the CQ poll, possibly 0). The caller drains
  // it.
  int WaitComp(ibv_wc* out, int max, int timeout_ms = -1, int also_fd = -1);
  // The first failed work completion this endpoint reaped, or nullptr. An RC
  // QP does not survive a failed completion, so any later ones are the flush
  // of what was still posted; the first names the cause. Per the verbs
  // contract only wr_id, status, qp_num and vendor_err are valid in it.
  // Owner-thread only, like the polling calls that record it.
  const ibv_wc* failed_wc() const {
    return has_failed_wc_ ? &failed_wc_ : nullptr;
  }
  // Work completions this endpoint reaped with a success status. On an RC QP
  // each one needed the peer's QP to answer, so a change in this number over
  // an interval shows that the peer had the other end during it. Owner-thread
  // only.
  uint64_t succeeded_completions() const { return succeeded_completions_; }
  // errno of the system or verbs call that made the last Open() fail; 0 when
  // Open() gave up for a reason that has no errno (a geometry the device
  // cannot hold).
  int open_errno() const { return open_errno_; }
  // Unblock a thread sitting in WaitComp (so the server can join its Serve
  // threads at shutdown). Thread-safe vs the waiter.
  void Wake();
  void set_busy_poll(bool v) { busy_poll_ = v; }
  // Event mode: poll the CQ for up to this long before arming the completion
  // channel and sleeping (0: sleep at once). A completion due within a few
  // microseconds then costs no sleep and wake-up.
  void set_spin_us(uint32_t us) { spin_us_ = us; }
  // Times a wait armed the completion channel to sleep (tests).
  uint64_t NotifyArms() const {
    return notify_arms_.load(std::memory_order_relaxed);
  }

  // Memory windows: a server GET grant bound by a type-2 window on a
  // region registered once (DFKV_RDMA_SERVER_MW, see
  // docs/design/server-memory-windows.md). Set before Open: room in the
  // send queue and the CQ for one bind and one invalidation per window.
  void set_window_slots(size_t n) { window_slots_ = n; }
  // The device offers type-2B windows: the capability flag, and one window
  // allocated and freed again.
  bool ProbeType2Windows();
  // The device's max_mw (0 if it cannot say).
  int MaxWindows() const;
  // This port's active rate in Mbit/s (speed x width); 0 if unknown.
  uint64_t PortRateMbps() const;
  // n type-2 windows, deallocated by Close. Close first tries deallocation,
  // moves a QP holding refused windows to ERR, and retries before QP destroy.
  // Any unproven revocation remains fail-closed, never assumed safe by time.
  // False (and none kept) if any allocation fails.
  bool AllocWindows(size_t n);
  size_t window_count() const { return windows_.size(); }
  // Binds window i to [addr, addr+len) of `region` for remote READ under a
  // fresh rkey, returned in *rkey. Posted on this QP ahead of the SEND that
  // carries the rkey, which the send queue orders after it. Unsignalled
  // unless asked: the completion of a later signalled request on the send
  // queue means it completed, and a failure ends the QP (its completion,
  // with wr_id, still comes).
  bool PostBindWindow(size_t i, ibv_mr* region, const void* addr, size_t len,
                      uint64_t wr_id, uint32_t* rkey, bool signaled = false);
  // Invalidates a window's rkey; unsignalled unless asked, as above.
  bool PostLocalInvalidate(uint32_t rkey, uint64_t wr_id,
                           bool signaled = false);
  // A region windows bind to: local write and window binding, no remote
  // access, so its own rkey authorizes nothing. Owned by the caller.
  ibv_mr* RegisterBindRegion(void* base, size_t size);
// 0/null if absent. Otherwise enforce the context-aware release guard:
  // an underegistered live region fails closed rather than letting the RAM
  // allocation be freed while DMA may still address it. A caught ordinary
  // error is returned to a recovery owner (the RAM registry entry must stay);
  // an otherwise unproven failed release aborts before its memory is freed.
  static int DeregisterBindRegion(ibv_mr* mr);
  ibv_context* context_for_test() const { return ctx_; }
  // Test seam: consulted before each window allocation; false fails it.
  static void SetWindowAllocSeamForTest(std::function<bool()> seam);
  // Test seam: replaces Close()'s first deallocation of each window (the
  // retry after the QP's destruction is real); nonzero fails it.
  static void SetWindowDeallocSeamForTest(std::function<int(ibv_mw*)> seam);
  // Test seam: observes the exact window after its bind was posted. No bind
  // semantics change; the callback is absent in production.
  static void SetWindowBindObserverForTest(std::function<void(ibv_mw*)> observer);
  void set_num_qp(size_t n) { num_qp_ = n > 0 ? n : 1; }
  size_t num_qp() const { return num_qp_; }
  // The QP's access flags as the device reports them; -1 if it cannot say.
  int QpAccessFlags() const;

 private:
  int ObserveCompletions(ibv_wc* out, int got);
  void Close();

  ibv_context* ctx_ = nullptr;
  uint64_t device_generation_ = 0;
  ibv_pd* pd_ = nullptr;
  ibv_cq* cq_ = nullptr;
  ibv_comp_channel* chan_ = nullptr;
  ibv_qp* qp_ = nullptr;
  uint8_t ib_port_ = 1;
  bool remote_access_ = true;  // see set_remote_access
  int gid_index_ = 0;       // local source GID index (RoCEv2 on Ethernet link layer)
  int numa_node_ = -1;      // NUMA node of the device (for buffer/thread placement)
  int wake_rfd_ = -1, wake_wfd_ = -1;  // self-pipe to interrupt WaitComp on stop
  ibv_mtu mtu_ = IBV_MTU_4096;
  uint8_t read_depth_ = 1, responder_read_depth_ = 1;
  int device_read_depth_ = 1, device_responder_read_depth_ = 1;
  unsigned cq_armed_unacked_ = 0;
  bool busy_poll_ = false;
  uint32_t spin_us_ = 0;
  std::atomic<uint64_t> notify_arms_{0};
  std::atomic<bool> woken_{false};  // Wake() was called (busy-poll checks this)
  size_t num_qp_ = 1;

  size_t cap_ = 0, depth_ = 0, dbuf_cap_ = 0;
  uint16_t remote_depth_ = 0;  // Set from the required v2 QP advertisement.
  size_t max_sge_ = 2;  // QP max_send_sge/max_recv_sge = min(kMaxSge, device cap)
  size_t read_sge_ = 1;  // device max_sge_rd (ReadSge)
  // sbuf_/rbuf_ point into control_buf_, all under control_mr_.
  char* control_buf_ = nullptr;
  size_t control_bytes_ = 0;
  ibv_mr* control_mr_ = nullptr;
  std::vector<char*> sbuf_, rbuf_, dbuf_;
  std::vector<ibv_mr*> smr_, rmr_, dmr_;
  // Big pre-registered caller regions (the host KV pool). Each cache entry owns
  // one shared-registry reference. Successful growth replaces the idle
  // endpoint's older same-base generation; endpoints with an in-flight user
  // retain their old generation until their next EnsurePoolMrs or Close.
  struct PoolMr {
    uintptr_t base;
    size_t size;
    int access;
    ibv_mr* mr;
  };
  std::vector<PoolMr> pool_mr_;
  // Operation-scoped out-of-pool MRs, released after local completion or
  // after endpoint destruction fences an abandoned operation.
  std::vector<ibv_mr*> transient_mr_;
  std::vector<ibv_mr*> lease_write_mr_;
  std::vector<ibv_mr*> own_mr_;
  std::vector<ibv_mr*> lease_read_mr_;
  size_t window_slots_ = 0;
  std::vector<ibv_mw*> windows_;
  std::vector<uint32_t> window_rkeys_;  // the key of each window's last bind
  QpInfo local_;
  // See failed_wc(). Written once, after a failure; declared last so the
  // members every post and poll reads stay where they were.
  ibv_wc failed_wc_{};
  bool has_failed_wc_ = false;
  uint64_t succeeded_completions_ = 0;
  int open_errno_ = 0;
};

// Pool MRs CommitPoolMr or EnsurePoolMrs handed back, deregistered when the
// holder goes (or at Deregister): declared before the lock a caller commits
// under, it deregisters them after the unlock also when the commit throws.
// The device must still be open then (see DeregisterRetired).
class RetiredPoolMrs {
 public:
  RetiredPoolMrs() = default;
  ~RetiredPoolMrs() { RcEndpoint::DeregisterRetired(&mrs_); }
  RetiredPoolMrs(const RetiredPoolMrs&) = delete;
  RetiredPoolMrs& operator=(const RetiredPoolMrs&) = delete;
  std::vector<ibv_mr*>* get() { return &mrs_; }
  void Deregister() { RcEndpoint::DeregisterRetired(&mrs_); }

 private:
  std::vector<ibv_mr*> mrs_;
};


}  // namespace rdma
}  // namespace dfkv

#endif  // DFKV_RDMA_VERBS_H_
