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
 * A connection uses RC two-sided SEND/RECV with a ring of `depth` slots so the
 * caller can keep multiple requests in flight (pipelining). Completions are
 * reaped via a completion channel (blocking, no busy-spin). */
#ifndef DFKV_RDMA_VERBS_H_
#define DFKV_RDMA_VERBS_H_

#include <infiniband/verbs.h>

#include <cstddef>
#include <cstdint>
#include <list>
#include <string>

#include "transport/dev_frame.h"
#include <unordered_map>
#include <utility>
#include <vector>

namespace dfkv {
namespace rdma {

// QP connection info exchanged over the TCP bootstrap channel (fixed 32 bytes,
// little-endian). lid==0 => use GRH/GID routing (RoCE); else IB LID routing.
//
// Wire layout: qpn u32 | psn u32 | lid u16 | gid[16] | pad[6]. Peers before the
// depth negotiation MEMSET the whole 32 bytes and never read the pad, so the
// pad is a deterministic-zero extension area: a non-zero magic there can ONLY
// come from a peer that wrote it on purpose -- exact detection, no length
// change, and every old/new pairing keeps connecting (a rolling-upgrade
// requirement; contrast the MDS member codec, which was born with tagged
// trailing extensions, while this blob was not).
//
// Extension DPQ1 (pad bytes [26,30) = magic, [30,32) = depth u16): each side
// advertises its pipeline depth -- how many in-flight requests its posted
// receives absorb. The CLIENT acts on it: pipelining past the server's posted
// receives hits RNR retries and degrades silently 3-4x (measured on hd05), so
// the client clamps its batching window to the advertised value
// (RcEndpoint::window()). depth==0 = legacy peer / no advertisement.
struct QpInfo {
  uint32_t qpn = 0;
  uint32_t psn = 0;
  uint16_t lid = 0;
  uint8_t gid[16] = {0};
  uint8_t pad[6] = {0};
  uint16_t depth = 0;  // not a standalone wire field: rides the pad (see above)
  uint8_t protocol_version = 1;  // v2 is encoded in the depth field's high bit
};
constexpr size_t kQpInfoBytes = 32;
constexpr uint32_t kQpDepthMagic = 0x31515044u;  // ASCII DPQ1 (LE)
void SerializeQpInfo(const QpInfo& in, char out[kQpInfoBytes]);
QpInfo ParseQpInfo(const char in[kQpInfoBytes]);

// Fixed-size device-name field the client sends first in the bootstrap so the
// server opens its QP on the matching device (same rail) for multi-rail setups.

// Target QP scatter-gather entries. SGE0 is the header (req/resp prefix + value
// header); the remaining kMaxSge-1 carry payload segments, so a scatter-gather
// key may hold up to kMaxSge-1 (=29) non-contiguous buffers. Open() clamps this
// to the device's reported max_sge (ConnectX-* on hd04 reports 30).
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
  // several one-sided GET writes before its signaled status SEND; client and v1
  // endpoints retain the legacy depth+1 SQ geometry.
  bool Open(const char* dev_name, size_t cap, size_t depth, uint8_t ib_port = 1,
            bool direct_io_buffers = false, size_t direct_io_cap = 0,
            bool v2_responder = false);

  QpInfo Local() const { return local_; }              // my QP info (after Open)
  bool Connect(const QpInfo& remote);                  // INIT -> RTR -> RTS

  size_t depth() const { return depth_; }
  // Depth negotiation (DPQ1): the peer's advertised pipeline depth (0 = legacy
  // peer, unknown). window() is the SAFE batching window -- never pipeline more
  // requests than the peer's posted receives, or they RNR-retry and the whole
  // window degrades silently. Falls back to the local depth against legacy
  // peers (the pre-negotiation behavior: an ops-level contract).
  void set_remote_depth(uint16_t d) { remote_depth_ = d; }
  uint16_t remote_depth() const { return remote_depth_; }
  size_t window() const {
    return (remote_depth_ > 0 && remote_depth_ < depth_) ? remote_depth_ : depth_;
  }
  size_t user_mr_cap() const { return user_mr_cap_; }  // >= depth (test/diagnostic)
  size_t cap() const { return cap_; }
  int numa_node() const { return numa_node_; }  // device's NUMA node, or -1
  char* sbuf(size_t slot) { return sbuf_[slot]; }
  char* rbuf(size_t slot) { return rbuf_[slot]; }
  char* dbuf(size_t slot) { return dbuf_.empty() ? nullptr : dbuf_[slot]; }
  size_t dbuf_cap() const { return dbuf_cap_; }
  ibv_mr* dmr(size_t slot) { return dmr_.empty() ? nullptr : dmr_[slot]; }

  bool PostRecv(size_t slot);                          // arm recv into rbuf_[slot]
  bool PostSend(size_t slot, size_t len);              // SEND sbuf_[slot][0,len)

  // Register a large caller memory region (e.g. the WHOLE SGLang host KV pool)
  // once on this PD. After this, RegisterUser() for any buffer inside the region
  // returns this MR by range lookup with NO per-op ibv_reg_mr — registration is a
  // one-time connection-setup cost, not a per-transfer cost. Client destination
  // pools set remote_write=true so a v2 server can RDMA-WRITE GET data into them.
  bool AddPoolMr(void* base, size_t size, bool remote_write = false);
  // Register every region not yet present as a pool MR. Called when a connection
  // is acquired so regions declared at any time land on every connection's PD.
  void EnsurePoolMrs(const std::vector<std::pair<void*, size_t>>& regions,
                     bool remote_write = false);
  // Register a shared server receive segment for remote PUT writes and return
  // the MR carrying both the local lkey and peer-visible rkey.
  ibv_mr* RegisterRemoteRegion(void* base, size_t size);

  // Process-wide count of ad-hoc user MRs registered OUTSIDE any pool region
  // (RegisterUser slow path). Should stay 0 in correct deployments where every
  // datapath buffer lives inside a RegisterMemory pool; a rising value flags a
  // pool-registration gap. Diagnostic; surfaced in the transport MetricsText.
  static uint64_t AdhocUserMrTotal();
  // Process-wide count of actual ibv_reg_mr calls for pool regions. With the
  // shared per-PD registry this counts distinct (device,region) pairs, not
  // connections (the pre-fix per-connection storm). Diagnostic/test.
  static uint64_t PoolMrRegistrations();

  // Resolve a caller buffer to an MR for zero-copy transfer. Fast path: a buffer
  // inside a registered pool region (AddPoolMr) returns that MR with no syscall.
  // Otherwise it is registered ad-hoc and kept in a small LRU cache (stable repeat
  // buffers stay cached). Valid as a recv target AND a send source (LOCAL_WRITE is
  // a superset of the no-flag local read a SEND does). nullptr on failure.
  ibv_mr* RegisterUser(void* addr, size_t len);
  // One-shot MR for API-owned buffers whose lifetime ends or may move after the
  // current operation. `remote_write=true` is for receive targets; false keeps
  // const/read-only send sources valid. Registered pool MRs with sufficient
  // access are reused. The endpoint owns ad-hoc MRs until ReleaseTransient() or
  // Close(), so a failed/timed-out QP tears down before deregistration.
  ibv_mr* RegisterTransient(void* addr, size_t len,
                            bool remote_write = true);
  void ReleaseTransient(ibv_mr* mr);
  // Scatter recv: first `hdr_bytes` of the message land in rbuf_[slot] (resp
  // prefix + value header), the remaining payload lands directly in `payload`
  // (must belong to `payload_mr` from RegisterUser) — zero copy into the caller.
  bool PostRecvScatter(size_t slot, void* payload, size_t payload_len,
                       ibv_mr* payload_mr, size_t hdr_bytes);
  // Scatter send: SGE0 = sbuf_[slot][0,hdr_len), SGE1 = `payload` (must belong to
  // `payload_mr`). Used by client PUT and server GET to gather a tiny header plus
  // a registered payload into one wire SEND with zero copy of the payload. Degrades
  // to a 1-SGE send when payload_len==0, so payload_mr may be null.
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

  // QP scatter-gather capability negotiated in Open() = min(kMaxSge, device cap).
  // The SG datapath caps payload segments at max_sge()-1 (SGE0 is the header).
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

  size_t cap_ = 0, depth_ = 0, dbuf_cap_ = 0;
  uint16_t remote_depth_ = 0;  // DPQ1 advertisement from the peer (0 = legacy)
  size_t max_sge_ = 2;  // QP max_send_sge/max_recv_sge = min(kMaxSge, device cap)
  std::vector<char*> sbuf_, rbuf_, dbuf_;
  std::vector<ibv_mr*> smr_, rmr_, dmr_;
  // Big pre-registered caller regions (the host KV pool). Registered once at
  // connection setup, never evicted; RegisterUser range-looks-up into these.
  struct PoolMr {
    uintptr_t base;
    size_t size;
    int access;
    ibv_mr* mr;
  };
  std::vector<PoolMr> pool_mr_;
  // LRU-capped cache of caller-buffer MRs (addr -> {mr, lru-iterator}); front of
  // user_lru_ is MRU. Bounded so a workload with many distinct buffers can't leak
  // registrations (a stable HiCache pool stays fully cached and always hits). The
  // cap MUST cover every MR one window can hold in flight before posting: the
  // legacy paths register 1 buffer per slot (depth total), but the SG multi
  // paths register up to max_sge-1 segment buffers PER SLOT and only then post
  // the whole window — a smaller cap could evict (ibv_dereg_mr) an MR still
  // referenced in mrs_per (use-after-dereg). Set to
  // max(kMinUserMr, depth * max_sge) in Open() after max_sge_ is negotiated.
  static constexpr size_t kMinUserMr = 64;
  size_t user_mr_cap_ = kMinUserMr;
  std::list<uintptr_t> user_lru_;
  std::unordered_map<uintptr_t, std::pair<ibv_mr*, std::list<uintptr_t>::iterator>> user_mr_;
  std::vector<ibv_mr*> transient_mr_;
  QpInfo local_;
};

}  // namespace rdma
}  // namespace dfkv

#endif  // DFKV_RDMA_VERBS_H_
