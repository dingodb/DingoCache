/* Native libibverbs RC endpoint shared by the dfkv RDMA client and server.
 * Built only when DFKV_WITH_RDMA is defined.
 *
 * Why native verbs (not librdmacm)? rdma_cm resolves the RDMA device from an IP
 * address (IPoIB), so it can only use a device that has an IP. On these clusters
 * the IP lives on the 200G port while the 8x 400G ports have NO IP and sit on a
 * SEPARATE IB fabric. Native verbs opens a device BY NAME (e.g. ib7s400p0) and
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
#include <vector>

namespace dfkv {
namespace rdma {

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
  // control buffers. v2_responder reserves enough SQ WRs for a server to chain
  // several one-sided GET writes before its signaled status SEND; initiators
  // need only the depth+1 SQ geometry.
  bool Open(const char* dev_name, size_t cap, size_t depth, uint8_t ib_port = 1,
            bool direct_io_buffers = false, size_t direct_io_cap = 0,
            bool v2_responder = false);

  QpInfo Local() const { return local_; }              // my QP info (after Open)
  bool Connect(const QpInfo& remote);                  // INIT -> RTR -> RTS

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
  void CommitPoolMr(PoolMrRegistration* registration);
  void RollbackPoolMr(PoolMrRegistration* registration);
  // Single-endpoint convenience wrapper, primarily used by server regions.
  bool AddPoolMr(void* base, size_t size, bool remote_write = false);
  // Atomically ensure all declarations on this endpoint. False leaves its
  // lookup cache exactly as it was before the call.
  bool EnsurePoolMrs(const std::vector<std::pair<void*, size_t>>& regions,
                     bool remote_write = false);
  // Register a shared server receive segment for remote PUT writes and return
  // the MR carrying both the local lkey and peer-visible rkey.
  ibv_mr* RegisterRemoteRegion(void* base, size_t size);

  // Cumulative one-shot user MRs registered outside explicit pool regions.
  // These are never cached; TransientUserMrActive is the lifetime invariant.
  static uint64_t AdhocUserMrTotal();
  // Process-wide count of actual ibv_reg_mr calls for pool regions. With the
  // shared per-PD registry this counts distinct (device,region) pairs, not
  // connections (the pre-fix per-connection storm). Diagnostic/test.
  static uint64_t PoolMrRegistrations();
  static uint64_t PoolMrRegistrationFailures();
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
  // current operation. `remote_write=true` is for receive targets; false keeps
  // const/read-only send sources valid. Registered pool MRs with sufficient
  // access are reused. The endpoint owns ad-hoc MRs until ReleaseTransient() or
  // Close(), so a failed/timed-out QP tears down before deregistration.
  ibv_mr* RegisterTransient(void* addr, size_t len,
                            bool remote_write = true);
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

  // RDMA v2 one-sided primitives. PostWrite is intentionally UNSIGNALED: a
  // following signaled status SEND on the same RC QP fences its source buffer
  // lifetime and produces the sole server-side completion. WRITE_WITH_IMM is
  // signaled because it is the client's request operation.
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

  // Block until at least one completion, drain up to `max` into out[]; returns
  // the count (>0), or <0 on error / when Wake() is called. wr_id of each wc =
  // the slot. Used for single round-trips (max=1) and pipelined batches.
  // timeout_ms < 0 blocks forever (the default, used by the client). A finite
  // timeout returns 0 if no completion arrives in that window — the server uses
  // it to reclaim an idle/abandoned connection instead of blocking indefinitely.
  int WaitComp(ibv_wc* out, int max, int timeout_ms = -1);
  // Unblock a thread sitting in WaitComp (so the server can join its Serve
  // threads at shutdown). Thread-safe vs the waiter.
  void Wake();
  void set_busy_poll(bool v) { busy_poll_ = v; }
  void set_num_qp(size_t n) { num_qp_ = n > 0 ? n : 1; }
  size_t num_qp() const { return num_qp_; }
  // #1: Start a background CQ reaper that polls ibv_poll_cq in a tight
  // loop and fills per-slot atomic flags. Eliminates CQ channel
  // syscall and CQ contention at high thread counts.

 private:
  void Close();

  ibv_context* ctx_ = nullptr;
  ibv_pd* pd_ = nullptr;
  ibv_cq* cq_ = nullptr;
  ibv_comp_channel* chan_ = nullptr;
  ibv_qp* qp_ = nullptr;
  uint8_t ib_port_ = 1;
  int gid_index_ = 0;       // local source GID index (RoCEv2 on Ethernet link layer)
  int numa_node_ = -1;      // NUMA node of the device (for buffer/thread placement)
  int wake_rfd_ = -1, wake_wfd_ = -1;  // self-pipe to interrupt WaitComp on stop
  ibv_mtu mtu_ = IBV_MTU_4096;
  unsigned cq_armed_unacked_ = 0;
  bool busy_poll_ = false;
  size_t num_qp_ = 1;

  size_t cap_ = 0, depth_ = 0, dbuf_cap_ = 0;
  uint16_t remote_depth_ = 0;  // Set from the required v2 QP advertisement.
  size_t max_sge_ = 2;  // QP max_send_sge/max_recv_sge = min(kMaxSge, device cap)
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
  // Operation-scoped out-of-pool MRs. These are released after completion, or
  // after QP teardown on every failure/timeout path.
  std::vector<ibv_mr*> transient_mr_;
  QpInfo local_;
};

}  // namespace rdma
}  // namespace dfkv

#endif  // DFKV_RDMA_VERBS_H_
