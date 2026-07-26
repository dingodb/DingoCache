#include "common/config_dump.h"
#include "transport/rdma_transport.h"

#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "utils/log.h"
#include "utils/net_util.h"     // Dial / WriteAll / ReadAll / Put*/Get*
#include "transport/rdma_verbs.h"   // RcEndpoint, QpInfo
#include "utils/numa_util.h"     // numa::DeviceNode / CurrentNode / Enabled
#include "transport/rail_select.h"   // rdma::PickRail
#include "common/value_header.h"

namespace dfkv {

namespace {
int EnvInt(const char* name, int dflt) {
  const char* v = std::getenv(name);
  if (!v || !*v) return dflt;
  long x = std::strtol(v, nullptr, 10);
  return x > 0 ? static_cast<int>(x) : dflt;
}

size_t EnvBytes(const char* name, size_t dflt) {
  const char* v = std::getenv(name);
  if (!v || !*v) return dflt;
  errno = 0;
  char* end = nullptr;
  unsigned long long x = std::strtoull(v, &end, 10);
  if (errno != 0 || end == v || x == 0) return dflt;
  constexpr unsigned long long kMaxSge =
      static_cast<unsigned long long>(std::numeric_limits<uint32_t>::max() -
                                      ValueHeader::kSize - 2 * rdma::kDirectIoAlign);
  if (x > kMaxSge) x = kMaxSge;
  return static_cast<size_t>(x);
}

size_t ResolveMaxPayload(size_t configured) {
  size_t n = configured ? configured : (64u << 20);
  n = EnvBytes("DFKV_RDMA_MAX_PAYLOAD_BYTES", n);
  n = EnvBytes("DFKV_RDMA_MAX_MSG_BYTES", n);  // compatibility alias
  // Clamp the constructor-supplied value too (EnvBytes only clamps the env paths):
  // dbuf/SGE length is uint32, so payload must stay under uint32 - header - 2*align
  // or the registered length silently overflows/truncates -> corruption.
  constexpr size_t kMaxSge = static_cast<size_t>(
      std::numeric_limits<uint32_t>::max() - ValueHeader::kSize - 2 * rdma::kDirectIoAlign);
  if (n > kMaxSge) n = kMaxSge;
  return n;
}

size_t ControlCapFor(size_t max_payload) {
  constexpr size_t kDefaultControlCap = 8u << 20;
  constexpr size_t kMinControlCap = kReqPrefix + ValueHeader::kSize;
  size_t cap = std::min(kDefaultControlCap, max_payload);
  return cap < kMinControlCap ? kMinControlCap : cap;
}

// Post w requests (already built in ep.sbuf[j], send length slen[j]) with one
// recv per slot, then reap all 2*w completions. recv wr_id = slot, so the reply
// for request j lands in ep.rbuf[j] (RC in-order delivery). rbytes[j] = reply
// length. Returns false (connection unusable) on any error completion.
bool RunWindow(rdma::RcEndpoint& ep, const std::vector<size_t>& slen,
               std::vector<uint32_t>* rbytes, int timeout_ms) {
  const size_t w = slen.size();
  rbytes->assign(w, 0);
  for (size_t j = 0; j < w; ++j)
    if (!ep.PostRecv(j) || !ep.PostSend(j, slen[j])) return false;
  std::vector<ibv_wc> wcs(2 * w);
  int need = static_cast<int>(2 * w);
  while (need > 0) {
    int g = ep.WaitComp(wcs.data(), static_cast<int>(2 * w), timeout_ms);
    if (g <= 0) return false;  // <0 error, 0 timeout => conn unusable, caller retries
    for (int i = 0; i < g; ++i) {
      if (wcs[i].status != IBV_WC_SUCCESS) return false;
      if (wcs[i].opcode == IBV_WC_RECV)
        (*rbytes)[static_cast<size_t>(wcs[i].wr_id)] = wcs[i].byte_len;
      --need;
    }
  }
  return true;
}
}  // namespace

struct RdmaTransport::Conn {
  rdma::RcEndpoint ep;
};

bool RdmaTransport::Available() {
  int n = 0;
  ibv_device** devs = ibv_get_device_list(&n);
  if (devs) ibv_free_device_list(devs);
  return n > 0;
}

RdmaTransport::RdmaTransport(size_t max_msg, const std::string& dev_name)
    : max_payload_(ResolveMaxPayload(max_msg)),
      declared_(std::min<uint64_t>(EnvBytes("DFKV_RDMA_MAX_BLOCK_BYTES", 0),
                                   ResolveMaxPayload(max_msg))),
      control_cap_(ControlCapFor(declared_ ? static_cast<size_t>(declared_)
                                           : ResolveMaxPayload(max_msg))),
      depth_(1) {
  std::string list = dev_name;
  if (list.empty()) { const char* e = std::getenv("DFKV_RDMA_DEV"); if (e) list = e; }
  for (size_t i = 0; i <= list.size();) {  // split on commas (multi-rail)
    size_t c = list.find(',', i);
    if (c == std::string::npos) c = list.size();
    std::string d = list.substr(i, c - i);
    if (!d.empty()) devs_.push_back(d);
    i = c + 1;
  }
  if (devs_.empty()) devs_.push_back("");  // first available device
  for (const auto& d : devs_)
    dev_node_.push_back(numa::DeviceNode(d.empty() ? nullptr : d.c_str()));
  // Live SG width: the tightest rail decides (connections round-robin rails,
  // and a key must fit whichever rail carries it). Queried once — device caps
  // don't change under a running process.
  {
    size_t msge = rdma::kMaxSge;
    for (const auto& d : devs_)
      msge = std::min(msge, rdma::QueryMaxSge(d.empty() ? nullptr : d.c_str()));
    sg_payload_segs_ = msge - 1;  // SGE0 carries the wire/value header
  }
  const char* d = std::getenv("DFKV_RDMA_DEPTH");  // pipeline depth (must be <= server's)
  if (d && *d) { long v = std::strtol(d, nullptr, 10); if (v >= 1 && v <= 256) depth_ = (size_t)v; }
  config_dump::RecordResolved("DFKV_RDMA_DEV", list.empty() ? "(auto)" : list);
  config_dump::RecordResolved("DFKV_RDMA_DEPTH", std::to_string(depth_));
  connect_ms_ = EnvInt("DFKV_RDMA_CONNECT_MS", 3000);
  io_ms_ = EnvInt("DFKV_RDMA_IO_MS", 10000);
  // Datapath completion timeout. EnvInt maps non-positive => default; treat an
  // explicit "0" as -1 (block forever, the legacy behavior) for an escape hatch.
  {
    const char* v = std::getenv("DFKV_RDMA_OP_TIMEOUT_MS");
    if (v && *v) {
      long x = std::strtol(v, nullptr, 10);
      op_timeout_ms_ = (x == 0) ? -1 : (x > 0 ? static_cast<int>(x) : 5000);
    }
  {
    const char* v = std::getenv("DFKV_RDMA_BATCH_OP_TIMEOUT_MS");
    if (v && *v) { long x = std::strtol(v, nullptr, 10); if (x > 0) batch_op_timeout_ms_ = static_cast<int>(x); }
  }
  config_dump::RecordResolved("DFKV_RDMA_OP_TIMEOUT_MS", std::to_string(op_timeout_ms_));
  config_dump::RecordResolved("DFKV_RDMA_BATCH_OP_TIMEOUT_MS", std::to_string(batch_op_timeout_ms_));
  }
  // Idle-connection pool cap. The pool naturally bounds at peak concurrency
  // (each thread holds <=1 conn); this only guards against a thread-count spike
  // leaving many idle conns. Must be >= peak concurrency or releases churn
  // (destroy+recreate every op), which fails the bootstrap under load. Default
  // 256 covers typical fan-out; raise via DFKV_RDMA_POOL_MAX for more threads.
  pool_max_ = static_cast<size_t>(EnvInt("DFKV_RDMA_POOL_MAX", 256));
  rail_conns_ = std::make_unique<std::atomic<uint64_t>[]>(devs_.size());  // 0-initialized
}

RdmaTransport::~RdmaTransport() {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto& [node, cs] : pool_)
    for (Conn* c : cs) Destroy(c);
  for (auto& [node, cs] : control_pool_)
    for (Conn* c : cs) Destroy(c);
}

void RdmaTransport::Destroy(Conn* c) { delete c; }  // RcEndpoint dtor tears down QP/MRs

RdmaTransport::Conn* RdmaTransport::Acquire(const std::string& node, Lane lane,
                                            bool* from_pool, bool force_new) {
  std::vector<std::pair<void*, size_t>> pools;
  Conn* pooled = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    pools = pools_;  // snapshot (a few regions); register on the conn outside the lock
    // A retry (force_new) skips the pool: the conn we just tore down was reclaimed
    // by the server, and the pool may hold more reclaimed conns — bootstrapping a
    // fresh one guarantees the retry isn't handed another dead conn.
    if (!force_new) {
      auto& p = (lane == Lane::kControl) ? control_pool_ : pool_;
      auto it = p.find(node);
      if (it != p.end() && !it->second.empty()) {
        pooled = it->second.back(); it->second.pop_back();
      }
    }
  }
  if (pooled) { pooled->ep.EnsurePoolMrs(pools); *from_pool = true; return pooled; }
  *from_pool = false;
  // Bootstrap the QP over a short-lived TCP connection to the node's member
  // address (control plane). The data plane then rides the named RDMA device.
  int fd = net::Dial(node, connect_ms_, io_ms_);
  if (fd < 0) return nullptr;

  // Pick a device for this connection: NUMA-aware when DFKV_RDMA_NUMA is on
  // (prefer a rail on the calling thread's NUMA node), else round-robin all.
  size_t tick = rr_.fetch_add(1, std::memory_order_relaxed);
  size_t ridx = rdma::PickRail(dev_node_, numa::CurrentNode(), numa::Enabled(), tick);
  const std::string& dev = devs_[ridx];

  auto* c = new Conn();
  if (!c->ep.Open(dev.empty() ? nullptr : dev.c_str(), control_cap_, depth_)) {
    ::close(fd); delete c; return nullptr;
  }
  // Bootstrap: tell the server which device to open (same rail), then exchange QP.
  char devbuf[rdma::kDevNameBytes];
  // DCP1: declare this process's max block size in the frame's zero tail so
  // the server sizes this connection's buffers to it (issue #110). declared_
  // == 0 -> legacy frame -> server keeps worst-case sizing.
  rdma::EncodeDevFrame(dev, declared_, devbuf);
  char mine[rdma::kQpInfoBytes], peer[rdma::kQpInfoBytes];
  rdma::QpInfo my = c->ep.Local();
  my.depth = static_cast<uint16_t>(std::min<size_t>(depth_, 256));  // DPQ1 advertisement
  rdma::SerializeQpInfo(my, mine);
  if (!net::WriteAll(fd, devbuf, rdma::kDevNameBytes) ||
      !net::WriteAll(fd, mine, rdma::kQpInfoBytes) ||
      !net::ReadAll(fd, peer, rdma::kQpInfoBytes)) {
    ::close(fd); delete c; return nullptr;
  }
  const rdma::QpInfo pq = rdma::ParseQpInfo(peer);
  if (!c->ep.Connect(pq)) { ::close(fd); delete c; return nullptr; }
  // Honor the server's advertised depth: clamp our batching window so we never
  // pipeline past its posted receives (RNR-retry silent degradation otherwise).
  if (pq.depth > 0) {
    c->ep.set_remote_depth(pq.depth);
    if (pq.depth < depth_)
      DFKV_LOG_INFO("rdma: server depth " + std::to_string(pq.depth) +
                    " < client depth " + std::to_string(depth_) +
                    ": batching window clamped to " + std::to_string(pq.depth));
  }
  // Wait for the server's "ready" byte: it has posted its recvs, so our first
  // SEND won't hit RNR.
  char ready = 0;
  if (!net::ReadAll(fd, &ready, 1) || ready != 1) { ::close(fd); delete c; return nullptr; }
  ::close(fd);  // bootstrap done; QP is RTS
  c->ep.EnsurePoolMrs(pools);  // register the declared host-pool regions on this PD
  conns_opened_.fetch_add(1, std::memory_order_relaxed);
  if (ridx < devs_.size()) rail_conns_[ridx].fetch_add(1, std::memory_order_relaxed);
  return c;
}

void RdmaTransport::RegisterMemory(void* base, size_t size) {
  if (!base || size == 0) return;
  std::vector<std::pair<void*, size_t>> pools;
  std::vector<rdma::RcEndpoint*> anchor_ptrs;
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& p : pools_) if (p.first == base) return;  // dedup by base
    pools_.push_back({base, size});
    mr_regions_.fetch_add(1, std::memory_order_relaxed);
    // Anchor each configured rail: hold a lifetime device ref and register the
    // pool MRs NOW, at declaration time (inference engines call this during
    // startup), not on the first connection's first op. Registering a
    // hundred-GB host KV pool pins every page (~4 s measured for 141 GB) —
    // without the anchor that cost sat in the first lookup after every client
    // process start, and the client mirror of the server-side dereg-on-idle
    // cycle (fixed by the server anchor) could re-charge it. Mirrors
    // RdmaServer::Start's anchor.
    if (anchors_.empty()) {
      for (const auto& d : devs_) {
        auto ep = std::make_unique<rdma::RcEndpoint>();
        if (ep->Open(d.empty() ? nullptr : d.c_str(), 4096, 1))
          anchors_.push_back(std::move(ep));
      }
    }
    pools = pools_;
    for (auto& ep : anchors_) anchor_ptrs.push_back(ep.get());
  }
  // The actual (seconds-scale for huge pools) registration runs outside mu_;
  // anchors_ itself is only ever filled once under mu_, so the snapshot of raw
  // pointers stays valid for the transport's lifetime.
  for (auto* ep : anchor_ptrs) ep->EnsurePoolMrs(pools);
  // Connections still EnsurePoolMrs on Acquire (no-op once anchored here).
}

std::string RdmaTransport::MetricsText() const {
  std::string s;
  s += "# HELP dfkv_rdma_client_conns_opened_total RDMA client connections opened\n";
  s += "# TYPE dfkv_rdma_client_conns_opened_total counter\n";
  s += "dfkv_rdma_client_conns_opened_total " +
       std::to_string(conns_opened_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_mr_regions Declared host MR regions\n";
  s += "# TYPE dfkv_rdma_client_mr_regions gauge\n";
  s += "dfkv_rdma_client_mr_regions " +
       std::to_string(mr_regions_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_rail_conns_total Connections opened per rail (device)\n";
  s += "# TYPE dfkv_rdma_client_rail_conns_total counter\n";
  for (size_t i = 0; i < devs_.size(); ++i) {
    const std::string& d = devs_[i].empty() ? std::string("default") : devs_[i];
    s += "dfkv_rdma_client_rail_conns_total{dev=\"" + d + "\"} " +
         std::to_string(rail_conns_[i].load(std::memory_order_relaxed)) + "\n";
  }
  // Effective pipeline depth (DFKV_RDMA_DEPTH; default 1). Per-connection pinned
  // control memory is ~2*control_cap*depth, so raising depth trades memory for
  // per-node request pipelining — surfaced so the value in effect is visible.
  s += "# HELP dfkv_rdma_client_pipeline_depth Effective RDMA pipeline depth (env DFKV_RDMA_DEPTH)\n";
  s += "# TYPE dfkv_rdma_client_pipeline_depth gauge\n";
  s += "dfkv_rdma_client_pipeline_depth " + std::to_string(depth_) + "\n";
  // Ad-hoc (out-of-pool) user MR registrations; should be 0 (see AdhocUserMrTotal).
  s += "# HELP dfkv_rdma_client_adhoc_user_mr_total User MRs registered outside any pool region\n";
  s += "# TYPE dfkv_rdma_client_adhoc_user_mr_total counter\n";
  s += "dfkv_rdma_client_adhoc_user_mr_total " +
       std::to_string(rdma::RcEndpoint::AdhocUserMrTotal()) + "\n";
  return s;
}

void RdmaTransport::Release(const std::string& node, Lane lane, Conn* c) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto& v = (lane == Lane::kControl ? control_pool_ : pool_)[node];
    if (v.size() < pool_max_) { v.push_back(c); return; }
  }
  Destroy(c);  // pool full -> drop (and tear down the QP/MRs) instead of growing
}

Status RdmaTransport::RoundTrip(const std::string& node, WireOp op,
                                const BlockKey& k, uint64_t offset,
                                uint64_t length, const void* payload,
                                uint64_t payload_len, std::string* out) {
  if (payload_len > control_cap_ - kReqPrefix) return Status::kInvalid;
  if (op == WireOp::kRange && length > control_cap_ - kRespPrefix)
    return Status::kInvalid;
  // Key-only ops (Exist/Remove/Members) ride the control lane so a lookup
  // storm never queues behind payload transfers; Cache/Range carry payloads.
  const Lane lane = (op == WireOp::kCache || op == WireOp::kRange)
                        ? Lane::kData : Lane::kControl;
  for (int attempt = 0; attempt < 2; ++attempt) {
    bool from_pool = false;
    Conn* c = Acquire(node, lane, &from_pool, attempt > 0);
    if (!c) return Status::kIOError;
    rdma::RcEndpoint& ep = c->ep;

    EncodeReq(ep.sbuf(0), op, k, offset, length, payload_len);
    if (payload_len) std::memcpy(ep.sbuf(0) + kReqPrefix, payload, payload_len);

    bool ok = ep.PostRecv(0) && ep.PostSend(0, kReqPrefix + payload_len);
    // Reap both the send and the recv completion (shared CQ; either order).
    uint32_t recv_bytes = 0;
    bool need_send = ok, need_recv = ok;
    while (ok && (need_send || need_recv)) {
      ibv_wc wc{};
      int g = ep.WaitComp(&wc, 1, op_timeout_ms_);
      if (g <= 0 || wc.status != IBV_WC_SUCCESS) { ok = false; break; }
      if (wc.opcode == IBV_WC_SEND) need_send = false;
      else if (wc.opcode == IBV_WC_RECV) { need_recv = false; recv_bytes = wc.byte_len; }
    }

    if (!ok) {
      Destroy(c);
      if (!from_pool) return Status::kIOError;
      continue;  // stale pooled conn -> retry fresh
    }
    if (recv_bytes < kRespPrefix) { Destroy(c); return Status::kIOError; }
    Status st; uint64_t dlen = 0;
    if (!DecodeResp(ep.rbuf(0), &st, &dlen)) { Destroy(c); return Status::kIOError; }
    if (out) {
      if (kRespPrefix + dlen > recv_bytes) { Destroy(c); return Status::kIOError; }
      out->assign(ep.rbuf(0) + kRespPrefix, dlen);
    }
    Release(node, lane, c);
    return st;
  }
  return Status::kIOError;
}

Status RdmaTransport::Cache(const std::string& node, const BlockKey& key,
                            const void* data, size_t len) {
  return RoundTrip(node, WireOp::kCache, key, 0, 0, data, len, nullptr);
}
Status RdmaTransport::Range(const std::string& node, const BlockKey& key,
                            uint64_t offset, uint64_t length, std::string* out) {
  return RoundTrip(node, WireOp::kRange, key, offset, length, nullptr, 0, out);
}
Status RdmaTransport::Exist(const std::string& node, const BlockKey& key, bool* exist) {
  Status st = RoundTrip(node, WireOp::kExist, key, 0, 0, nullptr, 0, nullptr);
  if (st == Status::kOk) { *exist = true; return Status::kOk; }
  if (st == Status::kNotFound) { *exist = false; return Status::kOk; }
  return st;
}

Status RdmaTransport::Remove(const std::string& node, const BlockKey& key) {
  // Key-only request, Status-only response (same framing as kExist).
  return RoundTrip(node, WireOp::kRemove, key, 0, 0, nullptr, 0, nullptr);
}

Status RdmaTransport::Members(const std::string& node, std::string* out) {
  return RoundTrip(node, WireOp::kMembers, BlockKey{}, 0, 0, nullptr, 0, out);
}

std::vector<Status> RdmaTransport::CacheMany(const std::string& node,
                                             const std::vector<CacheItem>& items) {
  const size_t n = items.size();
  std::vector<Status> res(n, Status::kIOError);
  if (n == 0) return res;
  for (const auto& it : items) {
    if (it.len > control_cap_ - kReqPrefix) {
      std::fill(res.begin(), res.end(), Status::kInvalid);
      return res;
    }
  }

  // 2-attempt loop: a stale pooled conn (peer reclaimed it after idle) fails fast
  // on the first window; tear it down and retry once on a fresh conn. Cache is
  // idempotent, so replaying the whole batch is harmless. Mirrors RoundTrip.
  for (int attempt = 0; attempt < 2; ++attempt) {
    std::fill(res.begin(), res.end(), Status::kIOError);
    bool from_pool = false;
    Conn* c = Acquire(node, Lane::kData, &from_pool, attempt > 0);
    if (!c) return res;
    rdma::RcEndpoint& ep = c->ep;
    const size_t W = ep.window();  // negotiated: never exceed the server's posted recvs
    bool conn_ok = true;
    for (size_t base = 0; base < n && conn_ok; base += W) {
      const size_t w = std::min(W, n - base);
      std::vector<size_t> slen(w);
      for (size_t j = 0; j < w; ++j) {
        const CacheItem& it = items[base + j];
        EncodeReq(ep.sbuf(j), WireOp::kCache, it.key, 0, 0, it.len);
        if (it.len) std::memcpy(ep.sbuf(j) + kReqPrefix, it.data, it.len);
        slen[j] = kReqPrefix + it.len;
      }
      std::vector<uint32_t> rbytes;
      if (!RunWindow(ep, slen, &rbytes, op_timeout_ms_)) { conn_ok = false; break; }
      for (size_t j = 0; j < w; ++j) {
        Status st; uint64_t dl = 0;
        if (rbytes[j] >= kRespPrefix && DecodeResp(ep.rbuf(j), &st, &dl)) res[base + j] = st;
      }
    }
    if (conn_ok) { Release(node, Lane::kData, c); return res; }
    Destroy(c);
    if (from_pool) continue;  // stale pooled conn -> one fresh retry
    return res;               // fresh conn failed -> terminal
  }
  return res;
}

std::vector<Status> RdmaTransport::RangeMany(const std::string& node,
                                             const std::vector<BlockKey>& keys,
                                             uint64_t offset, uint64_t length,
                                             std::vector<std::string>* outs) {
  const size_t n = keys.size();
  outs->assign(n, std::string());
  std::vector<Status> res(n, Status::kIOError);
  if (n == 0) return res;
  if (length > control_cap_ - kRespPrefix) {  // reply wouldn't fit the recv buffer
    std::fill(res.begin(), res.end(), Status::kInvalid);
    return res;
  }

  // 2-attempt loop: retry a stale pooled conn once on a fresh one. Range is a
  // read (idempotent), so replaying the whole batch is harmless. Mirrors RoundTrip.
  for (int attempt = 0; attempt < 2; ++attempt) {
    std::fill(res.begin(), res.end(), Status::kIOError);
    outs->assign(n, std::string());
    bool from_pool = false;
    Conn* c = Acquire(node, Lane::kData, &from_pool, attempt > 0);
    if (!c) return res;
    rdma::RcEndpoint& ep = c->ep;
    const size_t W = ep.window();  // negotiated: never exceed the server's posted recvs
    bool conn_ok = true;
    for (size_t base = 0; base < n && conn_ok; base += W) {
      const size_t w = std::min(W, n - base);
      std::vector<size_t> slen(w);
      for (size_t j = 0; j < w; ++j) {
        EncodeReq(ep.sbuf(j), WireOp::kRange, keys[base + j], offset, length, 0);
        slen[j] = kReqPrefix;
      }
      std::vector<uint32_t> rbytes;
      if (!RunWindow(ep, slen, &rbytes, op_timeout_ms_)) { conn_ok = false; break; }
      for (size_t j = 0; j < w; ++j) {
        uint32_t rb = rbytes[j];
        if (rb < kRespPrefix) continue;
        Status st; uint64_t dlen = 0;
        if (!DecodeResp(ep.rbuf(j), &st, &dlen)) continue;
        res[base + j] = st;
        if (st == Status::kOk) {
          if (kRespPrefix + dlen <= rb) (*outs)[base + j].assign(ep.rbuf(j) + kRespPrefix, dlen);
          else res[base + j] = Status::kIOError;
        }
      }
    }
    if (conn_ok) { Release(node, Lane::kData, c); return res; }
    Destroy(c);
    if (from_pool) continue;  // stale pooled conn -> one fresh retry
    return res;               // fresh conn failed -> terminal
  }
  return res;
}

std::vector<Status> RdmaTransport::ExistMany(const std::string& node,
                                             const std::vector<BlockKey>& keys,
                                             std::vector<char>* exists) {
  const size_t n = keys.size();
  exists->assign(n, 0);
  std::vector<Status> res(n, Status::kIOError);
  if (n == 0) return res;
  // kExist carries no payload and gets a status-only reply, so each request is
  // exactly kReqPrefix bytes and always fits the control buffer (no size guard).

  // 2-attempt loop: retry a stale pooled conn once on a fresh one. Exist is a
  // read (idempotent), so replaying the whole batch is harmless. Mirrors RoundTrip.
  for (int attempt = 0; attempt < 2; ++attempt) {
    std::fill(res.begin(), res.end(), Status::kIOError);
    std::fill(exists->begin(), exists->end(), 0);
    bool from_pool = false;
    Conn* c = Acquire(node, Lane::kControl, &from_pool, attempt > 0);
    if (!c) return res;
    rdma::RcEndpoint& ep = c->ep;
    const size_t W = ep.window();  // negotiated: never exceed the server's posted recvs
    bool conn_ok = true;
    for (size_t base = 0; base < n && conn_ok; base += W) {
      const size_t w = std::min(W, n - base);
      std::vector<size_t> slen(w);
      for (size_t j = 0; j < w; ++j) {
        EncodeReq(ep.sbuf(j), WireOp::kExist, keys[base + j], 0, 0, 0);
        slen[j] = kReqPrefix;
      }
      std::vector<uint32_t> rbytes;
      if (!RunWindow(ep, slen, &rbytes, op_timeout_ms_)) { conn_ok = false; break; }
      for (size_t j = 0; j < w; ++j) {
        if (rbytes[j] < kRespPrefix) continue;
        Status st; uint64_t dl = 0;
        if (!DecodeResp(ep.rbuf(j), &st, &dl)) continue;
        res[base + j] = st;                              // kOk=present, kNotFound=absent
        (*exists)[base + j] = (st == Status::kOk) ? 1 : 0;
      }
    }
    if (conn_ok) { Release(node, Lane::kControl, c); return res; }
    Destroy(c);
    if (from_pool) continue;  // stale pooled conn -> one fresh retry
    return res;               // fresh conn failed -> terminal
  }
  return res;
}

std::vector<Status> RdmaTransport::RangeInto(const std::string& node,
                                             const std::vector<BlockKey>& keys,
                                             const std::vector<RangeDst>& dsts,
                                             size_t header_size,
                                             std::vector<std::string>* hdrs) {
  const size_t n = keys.size();
  hdrs->assign(n, std::string());
  std::vector<Status> res(n, Status::kIOError);
  if (n == 0) return res;
  const size_t hdr_bytes = kRespPrefix + header_size;  // resp prefix + value header
  if (hdr_bytes > control_cap_) {
    std::fill(res.begin(), res.end(), Status::kInvalid);
    return res;
  }
  // Per-item bound check: an oversized dst fails ONLY itself (kInvalid, which
  // the client's health accounting ignores) — the multi variants already
  // fail-soft per item; poisoning the whole node batch made one oversized key
  // cost every sibling its cache hit.
  std::vector<char> bad(n, 0);
  for (size_t i = 0; i < n; ++i)
    if (NoteBlock(dsts[i].n)) bad[i] = 1;

  // 2-attempt loop: retry a stale pooled conn once on a fresh one. Range is a read
  // (idempotent) and the scatter lands in the caller's buffers, so replaying the
  // whole batch is harmless. Mirrors RoundTrip. (MR-registration failure is a
  // local resource issue, not staleness, so it stays terminal — no retry.)
  for (int attempt = 0; attempt < 2; ++attempt) {
    std::fill(res.begin(), res.end(), Status::kIOError);
    for (size_t i = 0; i < n; ++i) if (bad[i]) res[i] = Status::kInvalid;
    hdrs->assign(n, std::string());
    bool from_pool = false;
    Conn* c = Acquire(node, Lane::kData, &from_pool, attempt > 0);
    if (!c) return res;
    rdma::RcEndpoint& ep = c->ep;
    const size_t W = ep.window();  // negotiated: never exceed the server's posted recvs
    bool conn_ok = true;
    for (size_t base = 0; base < n && conn_ok; base += W) {
      const size_t w = std::min(W, n - base);
      // Register the destination buffers (cached).
      std::vector<ibv_mr*> mrs(w, nullptr);
      bool regok = true;
      for (size_t j = 0; j < w && regok; ++j) {
        if (bad[base + j]) continue;          // oversized: slot never posted
        if (dsts[base + j].n == 0) continue;  // empty: header-only reply into rbuf, no MR
        mrs[j] = ep.RegisterUser(dsts[base + j].payload, dsts[base + j].n);
        if (!mrs[j]) regok = false;
      }
      if (!regok) {
        // LOCAL resource failure (ibv_reg_mr), not the peer's fault: report the
        // unprocessed remainder as kInvalid so the caller's health accounting
        // does not MarkBad (cooldown) a healthy node for our own MR pressure.
        for (size_t i = base; i < n; ++i) if (!bad[i]) res[i] = Status::kInvalid;
        Release(node, Lane::kData, c);
        return res;
      }
      // Scatter recv [hdr -> rbuf | payload -> caller buffer], then send the Range req.
      // Empty (n==0) dsts use a plain recv (the reply is just resp-prefix + header).
      size_t posted = 0;
      for (size_t j = 0; j < w && conn_ok; ++j) {
        if (bad[base + j]) continue;  // stays kInvalid; slot not posted
        bool armed = dsts[base + j].n
                         ? ep.PostRecvScatter(j, dsts[base + j].payload, dsts[base + j].n, mrs[j], hdr_bytes)
                         : ep.PostRecv(j);
        if (!armed) { conn_ok = false; break; }
        EncodeReq(ep.sbuf(j), WireOp::kRange, keys[base + j], 0, header_size + dsts[base + j].n, 0);
        if (!ep.PostSend(j, kReqPrefix)) { conn_ok = false; break; }
        ++posted;
      }
      if (!conn_ok) break;
      if (posted == 0) continue;  // whole window oversized
      std::vector<ibv_wc> wcs(2 * posted);
      std::vector<uint32_t> rbytes(w, 0);
      int need = static_cast<int>(2 * posted);
      while (need > 0) {
        int g = ep.WaitComp(wcs.data(), static_cast<int>(2 * posted), BatchTimeout());
        if (g <= 0) { conn_ok = false; break; }
        for (int i = 0; i < g; ++i) {
          if (wcs[i].status != IBV_WC_SUCCESS) { conn_ok = false; break; }
          if (wcs[i].opcode == IBV_WC_RECV) rbytes[static_cast<size_t>(wcs[i].wr_id)] = wcs[i].byte_len;
          --need;
        }
        if (!conn_ok) break;
      }
      if (!conn_ok) break;
      for (size_t j = 0; j < w; ++j) {
        if (bad[base + j]) continue;
        uint32_t rb = rbytes[j];
        if (rb < kRespPrefix) continue;
        Status st; uint64_t dl = 0;
        if (!DecodeResp(ep.rbuf(j), &st, &dl)) continue;
        res[base + j] = st;            // payload (if any) already in dsts[].payload
        if (st == Status::kOk) {
          if (rb >= hdr_bytes) (*hdrs)[base + j].assign(ep.rbuf(j) + kRespPrefix, header_size);
          else res[base + j] = Status::kIOError;
        }
      }
    }
    if (conn_ok) { Release(node, Lane::kData, c); return res; }
    Destroy(c);
    if (from_pool) continue;  // stale pooled conn -> one fresh retry
    return res;               // fresh conn failed -> terminal
  }
  return res;
}

std::vector<Status> RdmaTransport::CacheFrom(const std::string& node,
                                             const std::vector<CacheSrc>& srcs) {
  const size_t n = srcs.size();
  std::vector<Status> res(n, Status::kIOError);
  if (n == 0) return res;
  // Do not silently fall back to the base copy path: RDMA CacheFrom is the
  // connector's zero-copy PUT route, so oversize/reg failures are explicit.
  // Per-item: an oversized item fails ONLY itself (kInvalid, ignored by the
  // client's health accounting) — same fail-soft the multi variants ship.
  std::vector<char> bad(n, 0);
  for (size_t i = 0; i < n; ++i) {
    const CacheSrc& s = srcs[i];
    if (NoteBlock(s.payload_len) || s.header_len > control_cap_ - kReqPrefix) bad[i] = 1;
  }

  // 2-attempt loop: retry a stale pooled conn once on a fresh one. Cache is
  // idempotent, so replaying the whole batch (re-sending earlier windows) is
  // harmless. Mirrors RoundTrip. (MR-registration failure stays terminal.)
  for (int attempt = 0; attempt < 2; ++attempt) {
    std::fill(res.begin(), res.end(), Status::kIOError);
    for (size_t i = 0; i < n; ++i) if (bad[i]) res[i] = Status::kInvalid;
    bool from_pool = false;
    Conn* c = Acquire(node, Lane::kData, &from_pool, attempt > 0);
    if (!c) return res;
    rdma::RcEndpoint& ep = c->ep;
    const size_t W = ep.window();  // negotiated: never exceed the server's posted recvs
    bool conn_ok = true;
    for (size_t base = 0; base < n && conn_ok; base += W) {
      const size_t w = std::min(W, n - base);
      // Register the payload buffers (cached).
      std::vector<ibv_mr*> mrs(w, nullptr);
      bool regok = true;
      for (size_t j = 0; j < w && regok; ++j) {
        const CacheSrc& s = srcs[base + j];
        if (bad[base + j]) continue;       // oversized: slot never posted
        if (s.payload_len == 0) continue;  // empty value: header-only 1-SGE send, no MR
        // const_cast: a SEND source is only read by the NIC; the LOCAL_WRITE-only
        // MR is never written, and RegisterUser keys its cache by address.
        mrs[j] = ep.RegisterUser(const_cast<void*>(s.payload), s.payload_len);
        if (!mrs[j]) regok = false;
      }
      if (!regok) {
        // LOCAL resource failure (ibv_reg_mr), not the peer's fault: kInvalid
        // for the unprocessed remainder so health accounting does not MarkBad
        // (cooldown) a healthy node for our own MR pressure.
        for (size_t i = base; i < n; ++i) if (!bad[i]) res[i] = Status::kInvalid;
        Release(node, Lane::kData, c);
        return res;
      }
      // Build [req prefix | value header] into sbuf[j], scatter-send with the
      // payload coming straight from the caller's registered buffer. The wire
      // payload_len field is header_len + user payload_len (the full stored blob).
      size_t posted = 0;
      for (size_t j = 0; j < w && conn_ok; ++j) {
        const CacheSrc& s = srcs[base + j];
        if (bad[base + j]) continue;  // stays kInvalid; slot not posted
        EncodeReq(ep.sbuf(j), WireOp::kCache, s.key, 0, 0, s.header_len + s.payload_len);
        if (s.header_len) std::memcpy(ep.sbuf(j) + kReqPrefix, s.header, s.header_len);
        if (!ep.PostRecv(j)) { conn_ok = false; break; }
        if (!ep.PostSendScatter(j, kReqPrefix + s.header_len, s.payload, s.payload_len, mrs[j])) {
          conn_ok = false; break;
        }
        ++posted;
      }
      if (!conn_ok) break;
      if (posted == 0) continue;  // whole window oversized
      // Reap 2*posted completions (send + recv per posted slot) before reusing
      // any slot or returning — the NIC reads the payload until SEND completes.
      std::vector<ibv_wc> wcs(2 * posted);
      std::vector<uint32_t> rbytes(w, 0);
      int need = static_cast<int>(2 * posted);
      while (need > 0) {
        int g = ep.WaitComp(wcs.data(), static_cast<int>(2 * posted), BatchTimeout());
        if (g <= 0) { conn_ok = false; break; }
        for (int i = 0; i < g; ++i) {
          if (wcs[i].status != IBV_WC_SUCCESS) { conn_ok = false; break; }
          if (wcs[i].opcode == IBV_WC_RECV) rbytes[static_cast<size_t>(wcs[i].wr_id)] = wcs[i].byte_len;
          --need;
        }
        if (!conn_ok) break;
      }
      if (!conn_ok) break;
      for (size_t j = 0; j < w; ++j) {
        if (bad[base + j]) continue;
        Status st; uint64_t dl = 0;
        if (rbytes[j] >= kRespPrefix && DecodeResp(ep.rbuf(j), &st, &dl)) res[base + j] = st;
      }
    }
    if (conn_ok) { Release(node, Lane::kData, c); return res; }
    Destroy(c);
    if (from_pool) continue;  // stale pooled conn -> one fresh retry
    return res;               // fresh conn failed -> terminal
  }
  return res;
}

std::vector<Status> RdmaTransport::CacheFromMulti(
    const std::string& node, const std::vector<CacheSrcMulti>& srcs) {
  const size_t n = srcs.size();
  std::vector<Status> res(n, Status::kIOError);
  if (n == 0) return res;
  // Validate: header fits the control buffer, total payload fits one message, and
  // the segment count fits one work request (1 header SGE + segs). Like CacheFrom,
  // do not silently fall back — this is the zero-copy SG PUT route.
  // 2-attempt loop: retry a stale pooled conn once on a fresh one. Cache is
  // idempotent, so replaying the whole batch is harmless. Mirrors RoundTrip.
  for (int attempt = 0; attempt < 2; ++attempt) {
    std::fill(res.begin(), res.end(), Status::kIOError);
    bool from_pool = false;
    Conn* c = Acquire(node, Lane::kData, &from_pool, attempt > 0);
    if (!c) return res;
    rdma::RcEndpoint& ep = c->ep;
    const size_t max_payload_segs = ep.max_sge() - 1;  // SGE0 = header
    // Per-item validation: an oversized key must fail ONLY itself (kInvalid), not
    // poison its node batch. The connector treats per-key kInvalid as a save miss
    // (fail-soft recompute), so we mark offenders and skip them in the window below.
    std::vector<char> bad(n, 0);
    for (size_t i = 0; i < n; ++i) {
      const auto& s = srcs[i];
      size_t total = 0;
      for (const auto& p : s.payloads) total += p.second;
      if (NoteBlock(total) || s.header_len > control_cap_ - kReqPrefix ||
          s.payloads.size() > max_payload_segs) {
        bad[i] = 1;
        res[i] = Status::kInvalid;
      }
    }

    const size_t W = ep.window();  // negotiated: never exceed the server's posted recvs
    bool conn_ok = true;
    for (size_t base = 0; base < n && conn_ok; base += W) {
      const size_t w = std::min(W, n - base);
      // Register every payload segment of every key in this window (cached MR lookup).
      // mrs_per[j] holds the per-segment MRs for slot j. Any failure => give the conn
      // back; Cache is idempotent so re-sending earlier windows on retry is harmless.
      // Oversized slots (bad[base+j]) are skipped: no MR, no recv/send, and they do
      // not consume a completion (need is sized to the posted slots only).
      std::vector<std::vector<ibv_mr*>> mrs_per(w);
      bool regok = true;
      for (size_t j = 0; j < w && regok; ++j) {
        if (bad[base + j]) continue;
        const CacheSrcMulti& s = srcs[base + j];
        mrs_per[j].assign(s.payloads.size(), nullptr);
        for (size_t e = 0; e < s.payloads.size() && regok; ++e) {
          if (s.payloads[e].second == 0) continue;  // empty segment: lkey 0, no MR
          mrs_per[j][e] = ep.RegisterUser(const_cast<void*>(s.payloads[e].first),
                                          s.payloads[e].second);
          if (!mrs_per[j][e]) regok = false;
        }
      }
      if (!regok) { Release(node, Lane::kData, c); return res; }
      size_t posted = 0;
      for (size_t j = 0; j < w && conn_ok; ++j) {
        if (bad[base + j]) continue;
        const CacheSrcMulti& s = srcs[base + j];
        size_t total = 0;
        std::vector<std::pair<const void*, uint32_t>> segs;
        segs.reserve(s.payloads.size());
        for (const auto& p : s.payloads) {
          segs.emplace_back(p.first, static_cast<uint32_t>(p.second));
          total += p.second;
        }
        // Wire payload_len = header_len + total user payload (the full stored blob).
        EncodeReq(ep.sbuf(j), WireOp::kCache, s.key, 0, 0, s.header_len + total);
        if (s.header_len) std::memcpy(ep.sbuf(j) + kReqPrefix, s.header, s.header_len);
        if (!ep.PostRecv(j)) { conn_ok = false; break; }
        if (!ep.PostSendScatterMulti(j, kReqPrefix + s.header_len, segs, mrs_per[j])) {
          conn_ok = false; break;
        }
        ++posted;
      }
      if (!conn_ok) break;
      if (posted == 0) continue;  // whole window was oversized: nothing to reap
      std::vector<ibv_wc> wcs(2 * w);
      std::vector<uint32_t> rbytes(w, 0);
      int need = static_cast<int>(2 * posted);
      while (need > 0) {
        int g = ep.WaitComp(wcs.data(), static_cast<int>(2 * w), BatchTimeout());
        if (g <= 0) { conn_ok = false; break; }
        for (int i = 0; i < g; ++i) {
          if (wcs[i].status != IBV_WC_SUCCESS) { conn_ok = false; break; }
          if (wcs[i].opcode == IBV_WC_RECV) rbytes[static_cast<size_t>(wcs[i].wr_id)] = wcs[i].byte_len;
          --need;
        }
        if (!conn_ok) break;
      }
      if (!conn_ok) break;
      for (size_t j = 0; j < w; ++j) {
        Status st; uint64_t dl = 0;
        if (rbytes[j] >= kRespPrefix && DecodeResp(ep.rbuf(j), &st, &dl)) res[base + j] = st;
      }
    }
    if (conn_ok) { Release(node, Lane::kData, c); return res; }
    Destroy(c);
    if (from_pool) continue;  // stale pooled conn -> one fresh retry
    return res;               // fresh conn failed -> terminal
  }
  return res;
}

// Records n as a high-water candidate and reports whether it exceeds the
// declaration. Two problems this closes, both observed on a B200 node:
//
//  1. Sizing DFKV_RDMA_MAX_BLOCK_BYTES was guesswork. The declaration decides
//     how much the SERVER pins per connection -- qd x (ValueHeader + declared),
//     ~1 GiB per connection at qd=32/16MiB measured -- so an over-generous
//     value is not free: it is server memory multiplied by every connection in
//     the fleet. The only figure an operator could see was the AVERAGE transfer
//     size (437 KiB there), which says nothing about the peak that must fit.
//
//  2. An UNDER-sized declaration failed silently. Oversized blocks become
//     kInvalid, which the client's health accounting deliberately ignores, so
//     upstream they are indistinguishable from an ordinary cache miss: no error,
//     no log, no counter -- just a hit rate quietly capped for large pages.
//     Anyone tuning this down would have had no signal that they went too far.
bool RdmaTransport::NoteBlock(size_t n) const {
  uint64_t prev = max_block_seen_.load(std::memory_order_relaxed);
  while (n > prev &&
         !max_block_seen_.compare_exchange_weak(prev, n, std::memory_order_relaxed)) {
  }
  if (n > prev) {  // new high-water mark: the number needed to size the declaration
    DFKV_LOG_INFO("rdma: max block observed " + std::to_string(n) + "B (" +
                  std::to_string(n / 1024) + " KiB); declared bound " +
                  std::to_string(OpBound()) + "B");
  }
  const size_t bound = OpBound();
  if (n <= bound) return false;
  const uint64_t k = oversize_rejects_.fetch_add(1, std::memory_order_relaxed);
  if (k == 0 || (k & 0x3FFu) == 0)  // first, then every 1024th
    DFKV_LOG_WARN("rdma: block " + std::to_string(n) + "B exceeds the declared bound " +
                  std::to_string(bound) + "B -> kInvalid, which reads as a cache MISS "
                  "upstream (not an error). Raise DFKV_RDMA_MAX_BLOCK_BYTES. rejects=" +
                  std::to_string(k + 1));
  return true;
}

std::vector<Status> RdmaTransport::RangeIntoMulti(
    const std::string& node, const std::vector<BlockKey>& keys,
    const std::vector<RangeDstMulti>& dsts, size_t header_size,
    std::vector<std::string>* hdrs, std::vector<size_t>* out_lens) {
  const size_t n = keys.size();
  hdrs->assign(n, std::string());
  if (out_lens) out_lens->assign(n, 0);
  std::vector<Status> res(n, Status::kIOError);
  if (n == 0) return res;
  const size_t hdr_bytes = kRespPrefix + header_size;  // resp prefix + value header
  if (hdr_bytes > control_cap_) {
    std::fill(res.begin(), res.end(), Status::kInvalid);
    return res;
  }
  // 2-attempt loop: retry a stale pooled conn once on a fresh one. Range is a read
  // (idempotent) and the scatter lands in the caller's buffers, so replaying the
  // whole batch is harmless. Mirrors RoundTrip.
  for (int attempt = 0; attempt < 2; ++attempt) {
    std::fill(res.begin(), res.end(), Status::kIOError);
    hdrs->assign(n, std::string());
    if (out_lens) out_lens->assign(n, 0);
    bool from_pool = false;
    Conn* c = Acquire(node, Lane::kData, &from_pool, attempt > 0);
    if (!c) return res;
    rdma::RcEndpoint& ep = c->ep;
    const size_t max_payload_segs = ep.max_sge() - 1;  // SGE0 = header
    // Per-item validation: an oversized destination must fail ONLY itself (kInvalid),
    // not poison its node batch. Offenders are marked and skipped in the window below.
    std::vector<char> bad(n, 0);
    for (size_t i = 0; i < n; ++i) {
      size_t cap = 0;
      for (const auto& p : dsts[i].payloads) cap += p.second;
      if (NoteBlock(cap) || dsts[i].payloads.size() > max_payload_segs) {
        bad[i] = 1;
        res[i] = Status::kInvalid;
      }
    }

    const size_t W = ep.window();  // negotiated: never exceed the server's posted recvs
    bool conn_ok = true;
    for (size_t base = 0; base < n && conn_ok; base += W) {
      const size_t w = std::min(W, n - base);
      // Register every destination segment of every key in this window. Oversized
      // slots (bad[base+j]) are skipped: no MR, no recv/send, no completion consumed.
      std::vector<std::vector<ibv_mr*>> mrs_per(w);
      std::vector<size_t> caps(w, 0);
      bool regok = true;
      for (size_t j = 0; j < w && regok; ++j) {
        if (bad[base + j]) continue;
        const RangeDstMulti& d = dsts[base + j];
        mrs_per[j].assign(d.payloads.size(), nullptr);
        for (size_t e = 0; e < d.payloads.size() && regok; ++e) {
          caps[j] += d.payloads[e].second;
          if (d.payloads[e].second == 0) continue;
          mrs_per[j][e] = ep.RegisterUser(d.payloads[e].first, d.payloads[e].second);
          if (!mrs_per[j][e]) regok = false;
        }
      }
      if (!regok) { Release(node, Lane::kData, c); return res; }
      size_t posted = 0;
      for (size_t j = 0; j < w && conn_ok; ++j) {
        if (bad[base + j]) continue;
        const RangeDstMulti& d = dsts[base + j];
        bool armed;
        if (caps[j]) {
          std::vector<std::pair<void*, uint32_t>> segs;
          segs.reserve(d.payloads.size());
          for (const auto& p : d.payloads)
            segs.emplace_back(p.first, static_cast<uint32_t>(p.second));
          armed = ep.PostRecvScatterMulti(j, segs, mrs_per[j], hdr_bytes);
        } else {
          armed = ep.PostRecv(j);  // header-only reply
        }
        if (!armed) { conn_ok = false; break; }
        EncodeReq(ep.sbuf(j), WireOp::kRange, keys[base + j], 0, header_size + caps[j], 0);
        if (!ep.PostSend(j, kReqPrefix)) { conn_ok = false; break; }
        ++posted;
      }
      if (!conn_ok) break;
      if (posted == 0) continue;  // whole window was oversized: nothing to reap
      std::vector<ibv_wc> wcs(2 * w);
      std::vector<uint32_t> rbytes(w, 0);
      int need = static_cast<int>(2 * posted);
      while (need > 0) {
        int g = ep.WaitComp(wcs.data(), static_cast<int>(2 * w), BatchTimeout());
        if (g <= 0) { conn_ok = false; break; }
        for (int i = 0; i < g; ++i) {
          if (wcs[i].status != IBV_WC_SUCCESS) { conn_ok = false; break; }
          if (wcs[i].opcode == IBV_WC_RECV) rbytes[static_cast<size_t>(wcs[i].wr_id)] = wcs[i].byte_len;
          --need;
        }
        if (!conn_ok) break;
      }
      if (!conn_ok) break;
      for (size_t j = 0; j < w; ++j) {
        uint32_t rb = rbytes[j];
        if (rb < kRespPrefix) continue;
        Status st; uint64_t dl = 0;
        if (!DecodeResp(ep.rbuf(j), &st, &dl)) continue;
        res[base + j] = st;  // payload (if any) already scattered into dsts[].payloads
        if (st == Status::kOk) {
          if (rb >= hdr_bytes) {
            (*hdrs)[base + j].assign(ep.rbuf(j) + kRespPrefix, header_size);
            // True stored payload bytes received = rb - resp-prefix - value header.
            if (out_lens) (*out_lens)[base + j] = rb - hdr_bytes;
          } else {
            res[base + j] = Status::kIOError;
          }
        }
      }
    }
    if (conn_ok) { Release(node, Lane::kData, c); return res; }
    Destroy(c);
    if (from_pool) continue;  // stale pooled conn -> one fresh retry
    return res;               // fresh conn failed -> terminal
  }
  return res;
}

}  // namespace dfkv
