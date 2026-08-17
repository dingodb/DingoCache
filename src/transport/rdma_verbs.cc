#include "transport/rdma_verbs.h"
#include "transport/rdma_protocol.h"
#include <arpa/inet.h>

#include <fcntl.h>
#include <poll.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include "utils/log.h"        // slow pool-MR registration report
#include "utils/net_util.h"   // PutU32/GetU32/PutU64 host-endian codec
#include "utils/numa_util.h"  // best-effort NUMA buffer placement

namespace dfkv {
namespace rdma {

namespace {
// Per-device shared ibv_context + PD, refcounted across all RcEndpoints on the
// same device. WHY: opening one ibv_context per connection (ibv_open_device is a
// heavy uverbs/kernel-context allocation) thrashes and stalls at high fan-out
// (~96 concurrent connections hung bench at t32+). The data path only needs a
// QP/CQ per connection; the context+PD are safely shared (MRs on a shared PD are
// usable by any QP on it). After the first connection opens a device, all others
// reuse it -- no further ibv_open_device. Keyed by device name (multi-rail rails
// are distinct devices -> distinct shared entries, which is correct).
// A pool MR is the big pre-registered host KV region. It belongs to the PD, not
// to any one connection, so it is registered ONCE per device and shared by all
// endpoints — previously each new connection re-ran ibv_reg_mr over the whole
// (tens-to-hundreds-of-GB) region, a registration storm at reconnect time.
struct SharedPoolMr {
  uintptr_t base;
  size_t size;
  int access;
  ibv_mr* mr;
  size_t refs;
};
struct SharedDevice {
  ibv_context* ctx;
  ibv_pd* pd;
  long refs;
  // Registered once per PD. Entries retire at zero endpoint refs; device
  // teardown is the safety net if ibv_dereg_mr previously refused one.
  std::vector<SharedPoolMr> pool_mrs;
};
std::mutex g_dev_mu;
std::unordered_map<std::string, SharedDevice> g_devs;

// Cumulative one-shot user MRs registered outside explicit pool regions.
// Unlike the live gauge below this legitimately grows when callers do not
// pre-register a stable pool; none of these registrations is cached.
std::atomic<uint64_t> g_adhoc_user_mr{0};
// Actual ibv_reg_mr calls for pool regions (once per PD per region). Diagnostic:
// with the shared registry this is ~= number of distinct (device,region) pairs,
// NOT number of connections (the pre-fix behavior).
std::atomic<uint64_t> g_pool_mr_regs{0};
std::atomic<uint64_t> g_pool_mr_reg_failures{0};
std::atomic<uint64_t> g_pool_mr_active{0};
std::atomic<uint64_t> g_transient_user_mr_active{0};
std::atomic<uint64_t> g_cq_completions{0};
std::atomic<uint64_t> g_cq_errors{0};

// The default path is one relaxed load in PostWrite. Tests can arm a bounded
// number of responder WRITEs and hold them before ibv_post_send.
std::atomic<size_t> g_test_block_writes{0};
std::mutex g_test_write_mu;
std::condition_variable g_test_write_cv;
size_t g_test_write_entered = 0;
size_t g_test_write_exited = 0;
size_t g_test_write_faults_consumed = 0;
size_t g_test_write_cancellations = 0;
size_t g_test_write_releases = 0;


int ObserveCqPoll(ibv_wc* out, int got) {
  if (got < 0) {
    g_cq_errors.fetch_add(1, std::memory_order_relaxed);
  } else if (got > 0) {
    g_cq_completions.fetch_add(static_cast<uint64_t>(got),
                               std::memory_order_relaxed);
    uint64_t failed = 0;
    for (int i = 0; i < got; ++i)
      if (out[i].status != IBV_WC_SUCCESS) ++failed;
    if (failed != 0)
      g_cq_errors.fetch_add(failed, std::memory_order_relaxed);
  }
  return got;
}

// Get-or-open the shared {ctx, pd} for `dev_name` (nullptr/"" = first device),
// bumping its refcount. Returns {nullptr,nullptr} on failure (no refcount taken).
std::pair<ibv_context*, ibv_pd*> AcquireSharedDevice(const char* dev_name) {
  const std::string key = (dev_name && *dev_name) ? std::string(dev_name) : std::string("\x01first");
  std::lock_guard<std::mutex> lk(g_dev_mu);
  auto it = g_devs.find(key);
  if (it != g_devs.end()) { ++it->second.refs; return {it->second.ctx, it->second.pd}; }
  int n = 0;
  ibv_device** list = ibv_get_device_list(&n);
  if (!list || n == 0) { if (list) ibv_free_device_list(list); return {nullptr, nullptr}; }
  ibv_device* dev = nullptr;
  if (dev_name && *dev_name) {
    for (int i = 0; i < n; ++i)
      if (std::strcmp(ibv_get_device_name(list[i]), dev_name) == 0) { dev = list[i]; break; }
  } else {
    dev = list[0];
  }
  if (!dev) { ibv_free_device_list(list); return {nullptr, nullptr}; }
  ibv_context* ctx = ibv_open_device(dev);
  ibv_free_device_list(list);
  if (!ctx) return {nullptr, nullptr};
  ibv_pd* pd = ibv_alloc_pd(ctx);
  if (!pd) { ibv_close_device(ctx); return {nullptr, nullptr}; }
  g_devs.emplace(key, SharedDevice{ctx, pd, 1, {}});
  return {ctx, pd};
}

// Drop one reference; when the last RcEndpoint on a device closes, free its PD
// then context. No-op for nullptr (failed Open).
void ReleaseSharedDevice(ibv_context* ctx) {
  if (!ctx) return;
  std::lock_guard<std::mutex> lk(g_dev_mu);
  for (auto it = g_devs.begin(); it != g_devs.end(); ++it) {
    if (it->second.ctx != ctx) continue;
    if (--it->second.refs == 0) {
      for (auto& p : it->second.pool_mrs) {
        if (p.mr) ibv_dereg_mr(p.mr);
        g_pool_mr_active.fetch_sub(1, std::memory_order_relaxed);
      }
      ibv_dealloc_pd(it->second.pd);
      ibv_close_device(it->second.ctx);
      g_devs.erase(it);
    }
    return;
  }
}

// Register `base`/`size` on the device's shared PD exactly once; subsequent
// callers (other endpoints on the same PD) get the existing MR with no
// ibv_reg_mr. Returns the (shared, PD-owned) MR, or nullptr on failure. The MR's
// lkey is valid for any QP on this PD, so callers may cache the pointer locally
// for lock-free lookup. Only called at connection setup (not the datapath), so
// taking g_dev_mu here is fine.
ibv_mr* SharedAddPoolMr(ibv_context* ctx, ibv_pd* pd, void* base, size_t size,
                        int access) {
  auto b = reinterpret_cast<uintptr_t>(base);
  std::lock_guard<std::mutex> lk(g_dev_mu);
  for (auto& d : g_devs) {
    if (d.second.ctx != ctx) continue;
    for (auto& p : d.second.pool_mrs) {
      if (p.base == b && p.size >= size && (p.access & access) == access) {
        ++p.refs;
        return p.mr;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    ibv_mr* mr = ibv_reg_mr(pd, base, size, access);
    if (!mr) {
      g_pool_mr_reg_failures.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0).count();
    // Registering a huge (arena-scale) region pins every page: seconds of
    // wall time. That must happen at startup (anchor endpoint), never on a
    // connection's first op — a slow registration HERE at serve time means
    // the anchor is missing or the region appeared late. Always log slow ones.
    if (ms > 100)
      DFKV_LOG_INFO("pool MR registered: " + std::to_string(size >> 20) +
                    " MiB in " + std::to_string(ms) + " ms");
    g_pool_mr_regs.fetch_add(1, std::memory_order_relaxed);
    g_pool_mr_active.fetch_add(1, std::memory_order_relaxed);
    d.second.pool_mrs.push_back(SharedPoolMr{b, size, access, mr, 1});
    return mr;
  }
  return nullptr;
}

// Release one endpoint cache reference. The MR is deregistered as soon as no
// endpoint can return its lkey, which retires superseded growth generations
// without invalidating in-flight users on older endpoints.
void SharedReleasePoolMr(ibv_context* ctx, ibv_mr* mr) {
  if (!ctx || !mr) return;
  std::lock_guard<std::mutex> lk(g_dev_mu);
  for (auto& [name, device] : g_devs) {
    (void)name;
    if (device.ctx != ctx) continue;
    for (auto it = device.pool_mrs.begin(); it != device.pool_mrs.end(); ++it) {
      if (it->mr != mr) continue;
      if (--it->refs == 0 && ibv_dereg_mr(it->mr) == 0) {
        device.pool_mrs.erase(it);
        g_pool_mr_active.fetch_sub(1, std::memory_order_relaxed);
      }
      return;
    }
    return;
  }
}
}  // namespace

void TestBlockNextResponderWrites(size_t count) {
  std::lock_guard<std::mutex> lock(g_test_write_mu);
  g_test_write_entered = 0;
  g_test_write_exited = 0;
  g_test_write_faults_consumed = 0;
  g_test_write_cancellations = 0;
  g_test_write_releases = count == 0 ? count : 0;
  g_test_block_writes.store(count, std::memory_order_release);
}

bool TestWaitForBlockedResponderWrites(size_t count, int timeout_ms) {
  std::unique_lock<std::mutex> lock(g_test_write_mu);
  return g_test_write_cv.wait_for(
      lock, std::chrono::milliseconds(timeout_ms),
      [&] { return g_test_write_entered >= count; });
}

bool TestWaitForReleasedResponderWrites(size_t count, int timeout_ms) {
  std::unique_lock<std::mutex> lock(g_test_write_mu);
  return g_test_write_cv.wait_for(
      lock, std::chrono::milliseconds(timeout_ms),
      [&] { return g_test_write_exited >= count; });
}

void TestReleaseBlockedResponderWrites() {
  {
    std::lock_guard<std::mutex> lock(g_test_write_mu);
    g_test_write_releases = std::numeric_limits<size_t>::max();
    g_test_block_writes.store(0, std::memory_order_release);
  }
  g_test_write_cv.notify_all();
}

void TestReleaseOneBlockedResponderWrite() {
  {
    std::lock_guard<std::mutex> lock(g_test_write_mu);
    ++g_test_write_releases;
  }
  g_test_write_cv.notify_all();
}

bool TestWaitForResponderWriteCancellations(size_t count, int timeout_ms) {
  std::unique_lock<std::mutex> lock(g_test_write_mu);
  return g_test_write_cv.wait_for(
      lock, std::chrono::milliseconds(timeout_ms),
      [&] { return g_test_write_cancellations >= count; });
}

bool TestConsumeBlockedResponderWriteFault() {
  std::lock_guard<std::mutex> lock(g_test_write_mu);
  if (g_test_write_faults_consumed == g_test_write_entered) return false;
  ++g_test_write_faults_consumed;
  return true;
}

uint64_t RcEndpoint::AdhocUserMrTotal() {
  return g_adhoc_user_mr.load(std::memory_order_relaxed);
}

uint64_t RcEndpoint::PoolMrRegistrations() {
  return g_pool_mr_regs.load(std::memory_order_relaxed);
}
uint64_t RcEndpoint::PoolMrRegistrationFailures() {
  return g_pool_mr_reg_failures.load(std::memory_order_relaxed);
}
uint64_t RcEndpoint::PoolMrActiveRegistrations() {
  return g_pool_mr_active.load(std::memory_order_relaxed);
}


uint64_t RcEndpoint::TransientUserMrActive() {
  return g_transient_user_mr_active.load(std::memory_order_relaxed);
}


uint64_t RcEndpoint::CqCompletions() {
  return g_cq_completions.load(std::memory_order_relaxed);
}

uint64_t RcEndpoint::CqErrors() {
  return g_cq_errors.load(std::memory_order_relaxed);
}

size_t QueryMaxSge(const char* dev_name) {
  auto shared = AcquireSharedDevice(dev_name);
  if (!shared.first) return kMaxSge;
  ibv_device_attr dev_attr{};
  size_t r = kMaxSge;
  if (ibv_query_device(shared.first, &dev_attr) == 0 && dev_attr.max_sge > 0) {
    r = std::min(kMaxSge, static_cast<size_t>(dev_attr.max_sge));
    if (r < 2) r = 2;  // SGE0 is the wire prefix; keep >=1 payload segment
  }
  ReleaseSharedDevice(shared.first);
  return r;
}

void SerializeQpInfo(const QpInfo& in, char out[kQpInfoBytes]) {
  std::memset(out, 0, kQpInfoBytes);
  net::PutU32(out + 0, in.qpn);
  net::PutU32(out + 4, in.psn);
  std::memcpy(out + 8, &in.lid, 2);
  std::memcpy(out + 10, in.gid, 16);
  net::PutU32(out + 26, kQpDepthMagic);
  uint16_t depth = in.depth;
  if (in.protocol_version == kDevProtoV2) depth |= 0x8000u;
  std::memcpy(out + 30, &depth, 2);
}

QpInfo ParseQpInfo(const char in[kQpInfoBytes]) {
  QpInfo q;
  q.qpn = net::GetU32(in + 0);
  q.psn = net::GetU32(in + 4);
  std::memcpy(&q.lid, in + 8, 2);
  std::memcpy(q.gid, in + 10, 16);
  // The six-byte pad was deterministic zero before negotiation metadata
  // existed. DPQ1 plus the depth high bit therefore provides an exact v2
  // signal without changing the fixed bootstrap frame length.
  if (net::GetU32(in + 26) == kQpDepthMagic) {
    uint16_t depth = 0;
    std::memcpy(&depth, in + 30, 2);
    if ((depth & 0x8000u) != 0) {
      depth &= 0x7FFFu;
      if (depth >= 1 && depth <= 256) {
        q.protocol_version = kDevProtoV2;
        q.depth = depth;
      }
    }
  }
  return q;
}

RcEndpoint::~RcEndpoint() { Close(); }

void RcEndpoint::Close() {
  if (qp_) { ibv_destroy_qp(qp_); qp_ = nullptr; }
  for (auto* m : smr_) if (m) ibv_dereg_mr(m);
  for (auto* m : rmr_) if (m) ibv_dereg_mr(m);
  for (auto* m : dmr_) if (m) ibv_dereg_mr(m);
  for (auto* mr : transient_mr_) if (mr) ibv_dereg_mr(mr);
  g_transient_user_mr_active.fetch_sub(transient_mr_.size(),
                                       std::memory_order_relaxed);
  transient_mr_.clear();
  // Each local lookup-cache entry owns one shared-registry reference. Release
  // them before the device reference so an old growth generation is deregistered
  // only after the last endpoint that can still return it has gone idle/closed.
  for (const auto& pool : pool_mr_) SharedReleasePoolMr(ctx_, pool.mr);
  pool_mr_.clear();
  smr_.clear(); rmr_.clear(); dmr_.clear();
  for (auto* b : sbuf_) delete[] b;
  for (auto* b : rbuf_) delete[] b;
  for (auto* b : dbuf_) std::free(b);
  sbuf_.clear(); rbuf_.clear(); dbuf_.clear();
  dbuf_cap_ = 0;
  if (cq_) { ibv_destroy_cq(cq_); cq_ = nullptr; }
  if (chan_) { ibv_destroy_comp_channel(chan_); chan_ = nullptr; }
  // ctx_ + pd_ are borrowed from the shared per-device registry; drop our ref
  // (frees them only when the last connection on this device closes).
  if (ctx_) { ReleaseSharedDevice(ctx_); ctx_ = nullptr; pd_ = nullptr; }
  if (wake_rfd_ >= 0) { ::close(wake_rfd_); wake_rfd_ = -1; }
  if (wake_wfd_ >= 0) { ::close(wake_wfd_); wake_wfd_ = -1; }
}

bool RcEndpoint::Open(const char* dev_name, size_t cap, size_t depth,
                      uint8_t ib_port, bool direct_io_buffers,
                      size_t direct_io_cap, bool v2_responder) {
  cap_ = cap; depth_ = depth; ib_port_ = ib_port;

  int wp[2];
  if (::pipe(wp) != 0) return false;  // self-pipe to interrupt WaitComp
  ::fcntl(wp[0], F_SETFL, O_NONBLOCK);
  wake_rfd_ = wp[0]; wake_wfd_ = wp[1];

  // Shared per-device ibv_context + PD (refcounted): the first connection on a
  // device opens it, the rest reuse it -- avoids the per-connection
  // ibv_open_device thrash that stalled high fan-out. Only the QP/CQ/MRs below
  // are per-connection.
  auto shared = AcquireSharedDevice(dev_name);
  ctx_ = shared.first;
  pd_ = shared.second;
  if (!ctx_ || !pd_) { Close(); return false; }

  chan_ = ibv_create_comp_channel(ctx_);
  if (!chan_) { Close(); return false; }
  const size_t cqe_per_slot =
      v2_responder ? kV2MaxGetTargets + 2 : 2;
  int cqe = static_cast<int>(depth_ * cqe_per_slot + 4);
  cq_ = ibv_create_cq(ctx_, cqe, nullptr, chan_, 0);
  if (!cq_) { Close(); return false; }
  if (ibv_req_notify_cq(cq_, 0) != 0) { Close(); return false; }

  // Negotiate scatter-gather width: min(kMaxSge, device cap). SGE0 carries the
  // wire prefix; the rest carry raw-payload segments. The two-SGE
  // [prefix | payload] path remains valid, so raising the cap is additive.
  max_sge_ = 2;
  uint32_t max_qp_wr = 0;
  {
    ibv_device_attr dev_attr{};
    if (ibv_query_device(ctx_, &dev_attr) == 0 && dev_attr.max_sge > 0) {
      const size_t cap_sge = static_cast<size_t>(dev_attr.max_sge);
      max_sge_ = std::min(kMaxSge, cap_sge);
      if (max_sge_ < 2) max_sge_ = 2;
      max_qp_wr = dev_attr.max_qp_wr;
    } else {
      max_sge_ = kMaxSge;  // query failed: let ibv_create_qp validate the caps
    }
  }

  // A v2 server GET may chain one signaled RDMA WRITE per protocol target plus
  // one signaled status SEND. Target count is fleet-wide and independent of
  // either HCA's max_sge because every WRITE itself has one SGE.
  static_assert(kV2MaxGetTargets == kMaxSge - 1);
  const size_t wr_per_slot =
      v2_responder ? (kV2MaxGetTargets + 1) : 2;
  if (depth_ > (std::numeric_limits<uint32_t>::max() - 1) / wr_per_slot) {
    DFKV_LOG_ERROR("rdma: requested send queue depth overflows uint32");
    Close();
    return false;
  }
  const uint32_t send_wr =
      static_cast<uint32_t>(depth_ * wr_per_slot + 1);
  if (max_qp_wr != 0 && send_wr > max_qp_wr) {
    DFKV_LOG_ERROR("rdma: required send WRs " + std::to_string(send_wr) +
                   " exceed device max_qp_wr " +
                   std::to_string(max_qp_wr));
    Close();
    return false;
  }
  ibv_qp_init_attr qa{};
  qa.send_cq = cq_;
  qa.recv_cq = cq_;
  qa.cap.max_send_wr = send_wr;
  qa.cap.max_recv_wr = static_cast<uint32_t>(depth_ + 1);
  qa.cap.max_send_sge = static_cast<uint32_t>(max_sge_);  // SGE0=hdr, rest=payload segs
  qa.cap.max_recv_sge = static_cast<uint32_t>(max_sge_);
  qa.qp_type = IBV_QPT_RC;
  qa.sq_sig_all = 0;
  qp_ = ibv_create_qp(pd_, &qa);
  if (!qp_) { Close(); return false; }

  // ibv_create_qp rewrites qa.cap with what the driver ACTUALLY granted, which
  // until now was discarded: max_sge_ came from the device attribute we derived
  // the *request* from, not from the answer. Two reasons to read it back:
  //
  //  - Defence. The spec says a grant is >= the request (or the call fails), so
  //    a smaller grant "cannot happen" -- but if it ever did, every scatter-gather
  //    post would build an SGL the QP cannot accept and fail at post time, far
  //    from the cause. Clamping here turns that into one line at Open().
  //  - Visibility. A grant LARGER than the request is real headroom: fewer
  //    scatter groups per page, so bigger blocks and fewer RDMA ops.
  //
  // Deliberately NOT adopted when larger. sg_payload_segs_ (= max_sge_ - 1)
  // selects the ``|sg={width}:{group}`` key coordinates, so width is part of
  // the key layout, not just a local transport detail. Clients that choose
  // different widths cannot share entries. Adopting a driver-specific grant
  // would silently partition a heterogeneous fleet; raising the width is a
  // deliberate fleet-wide schema decision.
  {
    const size_t granted = std::min<size_t>(qa.cap.max_send_sge, qa.cap.max_recv_sge);
    if (granted < max_sge_) {
      DFKV_LOG_WARN("rdma: QP granted max_sge=" + std::to_string(granted) +
                    " below the requested " + std::to_string(max_sge_) +
                    " (device reported more); clamping to the grant");
      max_sge_ = granted < 2 ? 2 : granted;
    } else if (granted > max_sge_) {
      DFKV_LOG_INFO("rdma: QP granted max_sge=" + std::to_string(granted) +
                    ", above the requested " + std::to_string(max_sge_) +
                    "; keeping the requested width (it selects the |sg=W:G key "
                    "coordinate, so it must stay uniform across the fleet). "
                    "Raise kMaxSge to use the headroom.");
    }
  }

  // QP -> INIT
  ibv_qp_attr at{};
  at.qp_state = IBV_QPS_INIT;
  at.pkey_index = 0;
  at.port_num = ib_port_;
  at.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                       IBV_ACCESS_REMOTE_READ;
  if (ibv_modify_qp(qp_, &at,
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
    Close(); return false;
  }

  // buffers + MRs. Bind to the device's NUMA node before registration so the
  // pages fault in locally (mbind sets policy; reg_mr faults/pins them). For our
  // sizes (>=128 KiB) new[] is mmap-backed = page-aligned, which mbind needs.
  numa_node_ = numa::DeviceNode(dev_name);
  sbuf_.resize(depth_, nullptr); rbuf_.resize(depth_, nullptr);
  smr_.resize(depth_, nullptr); rmr_.resize(depth_, nullptr);
  if (direct_io_buffers) {
    const size_t dio_cap = direct_io_cap ? direct_io_cap : cap_;
    dbuf_cap_ = dio_cap + 2 * kDirectIoAlign;
    dbuf_.resize(depth_, nullptr);
    dmr_.resize(depth_, nullptr);
  }
  for (size_t i = 0; i < depth_; ++i) {
    sbuf_[i] = new char[cap_]; rbuf_[i] = new char[cap_];
    numa::BindMemory(sbuf_[i], cap_, numa_node_);
    numa::BindMemory(rbuf_[i], cap_, numa_node_);
    smr_[i] = ibv_reg_mr(pd_, sbuf_[i], cap_, IBV_ACCESS_LOCAL_WRITE);
    rmr_[i] = ibv_reg_mr(pd_, rbuf_[i], cap_, IBV_ACCESS_LOCAL_WRITE);
    if (!smr_[i] || !rmr_[i]) { Close(); return false; }
    if (direct_io_buffers) {
      void* p = nullptr;
      if (posix_memalign(&p, kDirectIoAlign, dbuf_cap_) != 0) { Close(); return false; }
      dbuf_[i] = static_cast<char*>(p);
      numa::BindMemory(dbuf_[i], dbuf_cap_, numa_node_);
      dmr_[i] = ibv_reg_mr(pd_, dbuf_[i], dbuf_cap_,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
      if (!dmr_[i]) { Close(); return false; }
    }
  }
  for (size_t i = 0; i < depth_; ++i) {
  }

  // local addressing info
  ibv_port_attr pa{};
  if (ibv_query_port(ctx_, ib_port_, &pa) != 0) { Close(); return false; }
  mtu_ = pa.active_mtu;
  local_.lid = pa.lid;
  local_.qpn = qp_->qp_num;
  // Random 24-bit starting PSN, not derived from the QPN. The kernel recycles
  // QP numbers, so a QPN-derived PSN repeats whenever a QPN is reused; a stale
  // packet from a torn-down QP could then fall inside the new QP's expected PSN
  // window and be wrongly accepted. A random PSN per Open makes that astronomically
  // unlikely. (PSN is 24 bits on the wire.)
  {
    static thread_local std::mt19937 rng(std::random_device{}());
    local_.psn = rng() & 0xFFFFFF;
  }

  // Pick the source GID. On IB the LID is used for addressing (GID is along for
  // the ride), but on RoCE (Ethernet link layer, lid==0) the QP is addressed by
  // GID, and RoCEv2 GIDs are not at index 0 — pick a RoCEv2 entry so RoCE works.
  gid_index_ = 0;
  union ibv_gid gid{};
  ibv_query_gid(ctx_, ib_port_, 0, &gid);
  if (pa.lid == 0) {  // RoCE: pick a RoCEv2 GID, preferring a routable (non
                      // link-local fe80::) one — link-local GIDs don't resolve
                      // for loopback/cross-host the way IPv4-mapped GIDs do.
    int any_v2 = -1;
    for (int i = 0; i < pa.gid_tbl_len; ++i) {
      ibv_gid_entry e{};
      if (ibv_query_gid_ex(ctx_, ib_port_, i, &e, 0) != 0) continue;
      if (e.gid_type != IBV_GID_TYPE_ROCE_V2) continue;
      bool link_local = e.gid.raw[0] == 0xfe && (e.gid.raw[1] & 0xc0) == 0x80;
      if (any_v2 < 0) any_v2 = i;       // remember first RoCEv2 as fallback
      if (!link_local) { gid_index_ = i; gid = e.gid; any_v2 = -1; break; }
    }
    if (any_v2 >= 0) {                   // only link-local RoCEv2 available
      ibv_gid_entry e{};
      if (ibv_query_gid_ex(ctx_, ib_port_, any_v2, &e, 0) == 0) { gid_index_ = any_v2; gid = e.gid; }
    }
  }
  std::memcpy(local_.gid, gid.raw, 16);
  return true;
}

bool RcEndpoint::Connect(const QpInfo& remote) {
  // INIT -> RTR
  ibv_qp_attr at{};
  at.qp_state = IBV_QPS_RTR;
  at.path_mtu = mtu_;
  at.dest_qp_num = remote.qpn;
  at.rq_psn = remote.psn;
  at.max_dest_rd_atomic = 1;
  at.min_rnr_timer = 12;
  at.ah_attr.port_num = ib_port_;
  at.ah_attr.sl = 0;
  if (remote.lid != 0) {  // IB: LID routing
    at.ah_attr.is_global = 0;
    at.ah_attr.dlid = remote.lid;
  } else {                // RoCE: GRH/GID routing
    at.ah_attr.is_global = 1;
    std::memcpy(at.ah_attr.grh.dgid.raw, remote.gid, 16);
    at.ah_attr.grh.sgid_index = gid_index_;
    at.ah_attr.grh.hop_limit = 1;
  }
  int rtr = ibv_modify_qp(qp_, &at,
        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
        IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
  if (rtr != 0) return false;

  // RTR -> RTS
  ibv_qp_attr ts{};
  ts.qp_state = IBV_QPS_RTS;
  ts.timeout = 14;
  ts.retry_cnt = 7;
  ts.rnr_retry = 7;
  ts.sq_psn = local_.psn;
  ts.max_rd_atomic = 1;
  int rts = ibv_modify_qp(qp_, &ts,
        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
        IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
  if (rts != 0) return false;
  return true;
}

bool RcEndpoint::PostRecv(size_t slot) {
  if (slot >= depth_ || cap_ > std::numeric_limits<uint32_t>::max())
    return false;
  ibv_sge sge{};
  sge.addr = reinterpret_cast<uintptr_t>(rbuf_[slot]);
  sge.length = static_cast<uint32_t>(cap_);
  sge.lkey = rmr_[slot]->lkey;
  ibv_recv_wr wr{}, *bad = nullptr;
  wr.wr_id = slot; wr.sg_list = &sge; wr.num_sge = 1;
  return ibv_post_recv(qp_, &wr, &bad) == 0;
}

bool RcEndpoint::PostSend(size_t slot, size_t len) {
  if (slot >= depth_ || len > cap_ ||
      len > std::numeric_limits<uint32_t>::max())
    return false;
  ibv_sge sge{};
  sge.addr = reinterpret_cast<uintptr_t>(sbuf_[slot]);
  sge.length = static_cast<uint32_t>(len);
  sge.lkey = smr_[slot]->lkey;
  ibv_send_wr wr{}, *bad = nullptr;
  wr.wr_id = slot; wr.sg_list = &sge; wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND; wr.send_flags = IBV_SEND_SIGNALED;
  return ibv_post_send(qp_, &wr, &bad) == 0;
}

bool RcEndpoint::StagePoolMr(void* base, size_t size, bool remote_write,
                             PoolMrRegistration* registration) {
  if (!registration || !ctx_ || !pd_ || !base || size == 0) return false;
  const auto b = reinterpret_cast<uintptr_t>(base);
  if (size > std::numeric_limits<uintptr_t>::max() - b) return false;
  const int access = IBV_ACCESS_LOCAL_WRITE |
                     (remote_write ? IBV_ACCESS_REMOTE_WRITE : 0);
  *registration = PoolMrRegistration{b, size, access, nullptr, false};
  for (const auto& p : pool_mr_) {
    if (p.base == b && p.size >= size && (p.access & access) == access) {
      registration->mr = p.mr;
      return true;
    }
  }
  // Register on the process-wide shared PD, then cache the shared MR in this
  // endpoint for lock-free datapath range lookup. New entries go first because
  // an access upgrade covering the same range must win over an older
  // LOCAL_WRITE-only registration.
  ibv_mr* mr = SharedAddPoolMr(ctx_, pd_, base, size, access);
  if (!mr) return false;
  pool_mr_.insert(pool_mr_.begin(), PoolMr{b, size, access, mr});
  registration->mr = mr;
  registration->added = true;
  return true;
}

void RcEndpoint::RollbackPoolMr(PoolMrRegistration* registration) {
  if (!registration || !registration->added) return;
  const auto it = std::find_if(
      pool_mr_.begin(), pool_mr_.end(), [registration](const PoolMr& pool) {
        return pool.base == registration->base &&
               pool.size == registration->size &&
               pool.access == registration->access &&
               pool.mr == registration->mr;
      });
  if (it != pool_mr_.end()) {
    ibv_mr* mr = it->mr;
    pool_mr_.erase(it);
    SharedReleasePoolMr(ctx_, mr);
  }
  registration->added = false;
}

void RcEndpoint::CommitPoolMr(PoolMrRegistration* registration) {
  if (!registration || !registration->added) return;
  bool kept_generation = false;
  for (auto it = pool_mr_.begin(); it != pool_mr_.end();) {
    const bool staged =
        !kept_generation && it->base == registration->base &&
        it->size == registration->size &&
        it->access == registration->access && it->mr == registration->mr;
    if (staged) {
      kept_generation = true;
      ++it;
      continue;
    }
    const bool superseded =
        it->base == registration->base &&
        it->size <= registration->size &&
        (registration->access & it->access) == it->access;
    if (!superseded) {
      ++it;
      continue;
    }
    ibv_mr* old = it->mr;
    it = pool_mr_.erase(it);
    SharedReleasePoolMr(ctx_, old);
  }
  registration->added = false;
}

bool RcEndpoint::AddPoolMr(void* base, size_t size, bool remote_write) {
  PoolMrRegistration registration;
  if (!StagePoolMr(base, size, remote_write, &registration)) return false;
  CommitPoolMr(&registration);
  return true;
}

bool RcEndpoint::EnsurePoolMrs(
    const std::vector<std::pair<void*, size_t>>& regions, bool remote_write) {
  std::vector<PoolMrRegistration> registrations(regions.size());
  return RunPoolMrTransaction(
      regions.size(),
      [&](size_t index) {
        return StagePoolMr(regions[index].first, regions[index].second,
                           remote_write, &registrations[index]);
      },
      [&](size_t index) { CommitPoolMr(&registrations[index]); },
      [&](size_t index) { RollbackPoolMr(&registrations[index]); });
}

ibv_mr* RcEndpoint::RegisterRemoteRegion(void* base, size_t size) {
  const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
  if (!AddPoolMr(base, size, /*remote_write=*/true)) return nullptr;
  const auto b = reinterpret_cast<uintptr_t>(base);
  for (const auto& p : pool_mr_) {
    if (p.base == b && p.size >= size && (p.access & access) == access)
      return p.mr;
  }
  return nullptr;
}

ibv_mr* RcEndpoint::RegisterUser(void* addr, size_t len) {
  if (!addr || len == 0) return nullptr;
  const auto address = reinterpret_cast<uintptr_t>(addr);
  for (const auto& pool : pool_mr_) {
    if (address >= pool.base && len <= pool.size &&
        address - pool.base <= pool.size - len) {
      return pool.mr;
    }
  }
  return nullptr;
}

ibv_mr* RcEndpoint::RegisterTransient(void* addr, size_t len,
                                      bool remote_write) {
  if (!addr || len == 0) return nullptr;
  const auto a = reinterpret_cast<uintptr_t>(addr);
  const int access = remote_write
                         ? IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
                         : 0;
  for (const auto& p : pool_mr_) {
    if (a >= p.base && len <= p.size && a - p.base <= p.size - len &&
        (p.access & access) == access) {
      return p.mr;
    }
  }
  ibv_mr* mr = ibv_reg_mr(pd_, addr, len, access);
  if (mr) {
    transient_mr_.push_back(mr);
    g_adhoc_user_mr.fetch_add(1, std::memory_order_relaxed);
    g_transient_user_mr_active.fetch_add(1, std::memory_order_relaxed);
  }
  return mr;
}

void RcEndpoint::ReleaseTransient(ibv_mr* mr) {
  if (!mr) return;
  const auto it = std::find(transient_mr_.begin(), transient_mr_.end(), mr);
  if (it == transient_mr_.end()) return;
  ibv_dereg_mr(mr);
  transient_mr_.erase(it);
  g_transient_user_mr_active.fetch_sub(1, std::memory_order_relaxed);
}

void RcEndpoint::CancelResponderWrites() {
  if (!responder_cancelled_.exchange(true, std::memory_order_acq_rel)) {
    std::lock_guard<std::mutex> lock(g_test_write_mu);
    if (g_test_write_entered > g_test_write_exited)
      ++g_test_write_cancellations;
    g_test_write_cv.notify_all();
  }
  Wake();
}

bool RcEndpoint::RetireResponderWrites() {
  CancelResponderWrites();
  if (!qp_) return pending_responder_writes_ == 0;
  ibv_qp_attr attr{};
  ibv_qp_init_attr init{};
  if (ibv_query_qp(qp_, &attr, IBV_QP_STATE, &init) != 0) return false;
  if (attr.qp_state != IBV_QPS_ERR) {
    attr = {};
    attr.qp_state = IBV_QPS_ERR;
    if (ibv_modify_qp(qp_, &attr, IBV_QP_STATE) != 0) return false;
  }
  std::vector<ibv_wc> wcs(std::max<size_t>(1, depth_));
  while (pending_responder_writes_ != 0) {
    const int got = ibv_poll_cq(cq_, static_cast<int>(wcs.size()), wcs.data());
    if (got < 0) return false;
    if (got == 0) {
      std::this_thread::yield();
      continue;
    }
    ObserveCompletions(wcs.data(), got);
  }
  return true;
}

bool RcEndpoint::PostSendScatter(size_t slot, size_t hdr_len, const void* payload,
                                 size_t payload_len, ibv_mr* payload_mr) {
  if (slot >= depth_ || hdr_len > cap_ ||
      hdr_len > std::numeric_limits<uint32_t>::max() ||
      payload_len > std::numeric_limits<uint32_t>::max() ||
      (payload_len != 0 && (!payload || !payload_mr)))
    return false;
  ibv_sge sge[2]{};
  sge[0].addr = reinterpret_cast<uintptr_t>(sbuf_[slot]);  // request wire prefix
  sge[0].length = static_cast<uint32_t>(hdr_len);
  sge[0].lkey = smr_[slot]->lkey;
  sge[1].addr = reinterpret_cast<uintptr_t>(payload);      // payload from caller's MR
  sge[1].length = static_cast<uint32_t>(payload_len);
  sge[1].lkey = payload_len ? payload_mr->lkey : 0;
  ibv_send_wr wr{}, *bad = nullptr;
  wr.wr_id = slot; wr.sg_list = sge;
  wr.num_sge = payload_len ? 2 : 1;  // empty value -> prefix-only 1-SGE send
  wr.opcode = IBV_WR_SEND; wr.send_flags = IBV_SEND_SIGNALED;
  return ibv_post_send(qp_, &wr, &bad) == 0;
}

bool RcEndpoint::PostRecvScatter(size_t slot, void* payload, size_t payload_len,
                                 ibv_mr* payload_mr, size_t hdr_bytes) {
  if (slot >= depth_ || hdr_bytes > cap_ ||
      hdr_bytes > std::numeric_limits<uint32_t>::max() ||
      payload_len > std::numeric_limits<uint32_t>::max() ||
      payload_len == 0 || !payload || !payload_mr)
    return false;
  ibv_sge sge[2]{};
  sge[0].addr = reinterpret_cast<uintptr_t>(rbuf_[slot]);  // response wire prefix
  sge[0].length = static_cast<uint32_t>(hdr_bytes);
  sge[0].lkey = rmr_[slot]->lkey;
  sge[1].addr = reinterpret_cast<uintptr_t>(payload);      // payload -> caller buffer
  sge[1].length = static_cast<uint32_t>(payload_len);
  sge[1].lkey = payload_mr->lkey;
  ibv_recv_wr wr{}, *bad = nullptr;
  wr.wr_id = slot; wr.sg_list = sge; wr.num_sge = 2;
  return ibv_post_recv(qp_, &wr, &bad) == 0;
}

bool RcEndpoint::PostSendScatterMulti(
    size_t slot, size_t hdr_len,
    const std::vector<std::pair<const void*, uint32_t>>& segs,
    const std::vector<ibv_mr*>& mrs) {
  // SGE0 = wire prefix from sbuf_[slot]; SGE[1..] = one raw-payload segment
  // each. Validate every parallel vector before indexing it.
  if (slot >= depth_ || hdr_len > cap_ ||
      hdr_len > std::numeric_limits<uint32_t>::max() ||
      segs.size() != mrs.size() || segs.size() >= max_sge_) {
    return false;
  }
  std::vector<ibv_sge> sge(1 + segs.size());
  sge[0].addr = reinterpret_cast<uintptr_t>(sbuf_[slot]);
  sge[0].length = static_cast<uint32_t>(hdr_len);
  sge[0].lkey = smr_[slot]->lkey;
  for (size_t i = 0; i < segs.size(); ++i) {
    if (segs[i].second && (!segs[i].first || !mrs[i])) return false;
    sge[1 + i].addr = reinterpret_cast<uintptr_t>(segs[i].first);
    sge[1 + i].length = segs[i].second;
    sge[1 + i].lkey = segs[i].second ? mrs[i]->lkey : 0;
  }
  ibv_send_wr wr{}, *bad = nullptr;
  wr.wr_id = slot; wr.sg_list = sge.data();
  wr.num_sge = static_cast<int>(sge.size());
  wr.opcode = IBV_WR_SEND; wr.send_flags = IBV_SEND_SIGNALED;
  return ibv_post_send(qp_, &wr, &bad) == 0;
}

bool RcEndpoint::PostRecvScatterMulti(
    size_t slot, const std::vector<std::pair<void*, uint32_t>>& segs,
    const std::vector<ibv_mr*>& mrs, size_t hdr_bytes) {
  // SGE0 = header into rbuf_[slot]; SGE[1..] = one destination segment.
  // Validate every parallel vector before indexing it.
  if (slot >= depth_ || hdr_bytes > cap_ ||
      hdr_bytes > std::numeric_limits<uint32_t>::max() ||
      segs.size() != mrs.size() || segs.size() >= max_sge_) {
    return false;
  }
  std::vector<ibv_sge> sge(1 + segs.size());
  sge[0].addr = reinterpret_cast<uintptr_t>(rbuf_[slot]);
  sge[0].length = static_cast<uint32_t>(hdr_bytes);
  sge[0].lkey = rmr_[slot]->lkey;
  for (size_t i = 0; i < segs.size(); ++i) {
    if (segs[i].second && (!segs[i].first || !mrs[i])) return false;
    sge[1 + i].addr = reinterpret_cast<uintptr_t>(segs[i].first);
    sge[1 + i].length = segs[i].second;
    sge[1 + i].lkey = segs[i].second ? mrs[i]->lkey : 0;
  }
  ibv_recv_wr wr{}, *bad = nullptr;
  wr.wr_id = slot; wr.sg_list = sge.data();
  wr.num_sge = static_cast<int>(sge.size());
  return ibv_post_recv(qp_, &wr, &bad) == 0;
}

bool RcEndpoint::PostRead(size_t slot, void* destination, size_t length,
                          ibv_mr* destination_mr, uint64_t remote_addr,
                          uint32_t remote_rkey) {
  if (slot >= depth_ || length > std::numeric_limits<uint32_t>::max() ||
      (length != 0 && (!destination || !destination_mr)) ||
      remote_addr == 0 || remote_rkey == 0) {
    return false;
  }
  ibv_sge sge{};
  sge.addr = reinterpret_cast<uintptr_t>(destination);
  sge.length = static_cast<uint32_t>(length);
  sge.lkey = length ? destination_mr->lkey : 0;
  ibv_send_wr wr{}, *bad = nullptr;
  wr.wr_id = slot;
  wr.sg_list = &sge;
  wr.num_sge = length ? 1 : 0;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = remote_rkey;
  return ibv_post_send(qp_, &wr, &bad) == 0;
}

bool RcEndpoint::PostWrite(size_t slot, const void* source, size_t length,
                           ibv_mr* source_mr, uint64_t remote_addr,
                           uint32_t remote_rkey) {
  if (slot >= depth_ || length > std::numeric_limits<uint32_t>::max() ||
      (length != 0 && (!source || !source_mr)) ||
      remote_addr == 0 || remote_rkey == 0) {
    return false;
  }
  if (responder_cancelled_.load(std::memory_order_acquire)) return false;
  bool test_blocked = false;
  size_t remaining = g_test_block_writes.load(std::memory_order_acquire);
  while (remaining != 0 &&
         !g_test_block_writes.compare_exchange_weak(
             remaining, remaining - 1, std::memory_order_acq_rel,
             std::memory_order_acquire)) {
  }
  if (remaining != 0) {
    test_blocked = true;
    std::unique_lock<std::mutex> lock(g_test_write_mu);
    const size_t ordinal = ++g_test_write_entered;
    g_test_write_cv.notify_all();
    g_test_write_cv.wait(
        lock, [ordinal] { return g_test_write_releases >= ordinal; });
  }
  if (responder_cancelled_.load(std::memory_order_acquire)) {
    if (test_blocked) {
      std::lock_guard<std::mutex> lock(g_test_write_mu);
      ++g_test_write_exited;
      g_test_write_cv.notify_all();
    }
    return false;
  }
  ibv_sge sge{};
  sge.addr = reinterpret_cast<uintptr_t>(source);
  sge.length = static_cast<uint32_t>(length);
  sge.lkey = length ? source_mr->lkey : 0;
  ibv_send_wr wr{}, *bad = nullptr;
  wr.wr_id = slot;
  wr.sg_list = &sge;
  wr.num_sge = length ? 1 : 0;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = remote_rkey;
  const bool posted = ibv_post_send(qp_, &wr, &bad) == 0;
  if (posted) ++pending_responder_writes_;
  if (test_blocked) {
    std::lock_guard<std::mutex> lock(g_test_write_mu);
    ++g_test_write_exited;
    g_test_write_cv.notify_all();
  }
  return posted;
}

bool RcEndpoint::PostWriteImm(size_t slot, size_t length,
                              uint64_t remote_addr, uint32_t remote_rkey,
                              uint32_t immediate) {
  if (slot >= depth_ || length > cap_ ||
      length > std::numeric_limits<uint32_t>::max() ||
      remote_addr == 0 || remote_rkey == 0) {
    return false;
  }
  ibv_sge sge{};
  sge.addr = reinterpret_cast<uintptr_t>(sbuf_[slot]);
  sge.length = static_cast<uint32_t>(length);
  sge.lkey = smr_[slot]->lkey;
  ibv_send_wr wr{}, *bad = nullptr;
  wr.wr_id = slot;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = htonl(immediate);
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = remote_rkey;
  return ibv_post_send(qp_, &wr, &bad) == 0;
}

bool RcEndpoint::PostWriteImmScatter(
    size_t slot, size_t header_len, const void* payload, size_t payload_len,
    ibv_mr* payload_mr, uint64_t remote_addr, uint32_t remote_rkey,
    uint32_t immediate) {
  if (slot >= depth_ || header_len > cap_ ||
      header_len > std::numeric_limits<uint32_t>::max() ||
      payload_len > std::numeric_limits<uint32_t>::max() ||
      (payload_len != 0 && (!payload || !payload_mr)) ||
      remote_addr == 0 || remote_rkey == 0) {
    return false;
  }
  ibv_sge sge[2]{};
  sge[0].addr = reinterpret_cast<uintptr_t>(sbuf_[slot]);
  sge[0].length = static_cast<uint32_t>(header_len);
  sge[0].lkey = smr_[slot]->lkey;
  sge[1].addr = reinterpret_cast<uintptr_t>(payload);
  sge[1].length = static_cast<uint32_t>(payload_len);
  sge[1].lkey = payload_len ? payload_mr->lkey : 0;
  ibv_send_wr wr{}, *bad = nullptr;
  wr.wr_id = slot;
  wr.sg_list = sge;
  wr.num_sge = payload_len ? 2 : 1;
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = htonl(immediate);
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = remote_rkey;
  return ibv_post_send(qp_, &wr, &bad) == 0;
}

bool RcEndpoint::PostWriteScatter(
    size_t slot, size_t header_len, const void* payload, size_t payload_len,
    ibv_mr* payload_mr, uint64_t remote_addr, uint32_t remote_rkey) {
  if (slot >= depth_ || header_len > cap_ ||
      header_len > std::numeric_limits<uint32_t>::max() ||
      payload_len > std::numeric_limits<uint32_t>::max() ||
      (payload_len != 0 && (!payload || !payload_mr)) ||
      remote_addr == 0 || remote_rkey == 0) {
    return false;
  }
  ibv_sge sge[2]{};
  sge[0].addr = reinterpret_cast<uintptr_t>(sbuf_[slot]);
  sge[0].length = static_cast<uint32_t>(header_len);
  sge[0].lkey = smr_[slot]->lkey;
  sge[1].addr = reinterpret_cast<uintptr_t>(payload);
  sge[1].length = static_cast<uint32_t>(payload_len);
  sge[1].lkey = payload_len ? payload_mr->lkey : 0;
  ibv_send_wr wr{}, *bad = nullptr;
  wr.wr_id = slot;
  wr.sg_list = sge;
  wr.num_sge = payload_len ? 2 : 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = 0;
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = remote_rkey;
  return ibv_post_send(qp_, &wr, &bad) == 0;
}

bool RcEndpoint::PostWriteImmScatterMulti(
    size_t slot, size_t header_len,
    const std::vector<std::pair<const void*, uint32_t>>& segs,
    const std::vector<ibv_mr*>& mrs, uint64_t remote_addr,
    uint32_t remote_rkey, uint32_t immediate) {
  if (slot >= depth_ || segs.size() != mrs.size() ||
      segs.size() >= max_sge_ || header_len > cap_ ||
      header_len > std::numeric_limits<uint32_t>::max() ||
      remote_addr == 0 || remote_rkey == 0) {
    return false;
  }
  std::vector<ibv_sge> sge(1 + segs.size());
  sge[0].addr = reinterpret_cast<uintptr_t>(sbuf_[slot]);
  sge[0].length = static_cast<uint32_t>(header_len);
  sge[0].lkey = smr_[slot]->lkey;
  for (size_t i = 0; i < segs.size(); ++i) {
    if (segs[i].second != 0 && (!segs[i].first || !mrs[i])) return false;
    sge[i + 1].addr = reinterpret_cast<uintptr_t>(segs[i].first);
    sge[i + 1].length = segs[i].second;
    sge[i + 1].lkey = segs[i].second ? mrs[i]->lkey : 0;
  }
  ibv_send_wr wr{}, *bad = nullptr;
  wr.wr_id = slot;
  wr.sg_list = sge.data();
  wr.num_sge = static_cast<int>(sge.size());
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = htonl(immediate);
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = remote_rkey;
  return ibv_post_send(qp_, &wr, &bad) == 0;
}

int RcEndpoint::ObserveCompletions(ibv_wc* out, int got) {
  got = ObserveCqPoll(out, got);
  for (int i = 0; i < got; ++i) {
    if (out[i].opcode == IBV_WC_RDMA_WRITE &&
        pending_responder_writes_ != 0) {
      --pending_responder_writes_;
    }
  }
  return got;
}

int RcEndpoint::PollComp(ibv_wc* out, int max) {
  return ObserveCompletions(out, ibv_poll_cq(cq_, max, out));
}

int RcEndpoint::WaitComp(ibv_wc* out, int max, int timeout_ms) {
  if (busy_poll_) {
    using Clock = std::chrono::steady_clock;
    auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms < 0 ? 0 : timeout_ms);
    for (;;) {
      int got = ibv_poll_cq(cq_, max, out);
      if (got != 0) return ObserveCompletions(out, got);
      pollfd pfd = {wake_rfd_, POLLIN, 0};
      if (::poll(&pfd, 1, 0) > 0 && (pfd.revents & POLLIN)) return -1;
      if (timeout_ms == 0) return ObserveCompletions(out, 0);
      if (timeout_ms > 0 && Clock::now() >= deadline)
        return ObserveCompletions(out, 0);
    }
  }
  for (;;) {
    int got = ibv_poll_cq(cq_, max, out);
    if (got != 0) return ObserveCompletions(out, got);
    // No completion ready: arm notify, poll once more (close the race where a
    // completion arrived between the empty poll and the notify), then block on
    // the comp-channel fd OR the wake pipe (so Stop() can interrupt us).
    if (ibv_req_notify_cq(cq_, 0) != 0) return ObserveCompletions(out, -1);
    got = ibv_poll_cq(cq_, max, out);
    if (got != 0) return ObserveCompletions(out, got);
    pollfd pfds[2] = {{chan_->fd, POLLIN, 0}, {wake_rfd_, POLLIN, 0}};
    int pr = ::poll(pfds, 2, timeout_ms);
    if (pr < 0) {
      if (errno == EINTR) continue;
      return ObserveCompletions(out, -1);
    }
    if (pr == 0)
      return ObserveCompletions(out, ibv_poll_cq(cq_, max, out));
    if (pfds[1].revents & POLLIN) return -1;  // expected shutdown wake
    if (pfds[0].revents & POLLIN) {
      ibv_cq* ev_cq = nullptr;
      void* ev_ctx = nullptr;
      if (ibv_get_cq_event(chan_, &ev_cq, &ev_ctx) != 0)
        return ObserveCompletions(out, -1);
      ibv_ack_cq_events(ev_cq, 1);
    }
  }
}

void RcEndpoint::Wake() {
  if (wake_wfd_ >= 0) { char b = 1; ssize_t n = ::write(wake_wfd_, &b, 1); (void)n; }
}


}  // namespace rdma
}  // namespace dfkv
