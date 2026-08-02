#include "common/config_dump.h"
#include "transport/rdma_transport.h"

#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "utils/log.h"
#include "utils/net_util.h"     // Dial / WriteAll / ReadAll / Put*/Get*
#include "transport/rdma_topology.h"
#include "transport/rdma_verbs.h"   // RcEndpoint, QpInfo
#include "transport/rdma_protocol.h"
#include "utils/numa_util.h"     // CurrentNode / Enabled
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

std::vector<std::string> ParseDeviceList(const std::string& list) {
  std::vector<std::string> devices;
  for (size_t i = 0; i <= list.size();) {
    size_t comma = list.find(',', i);
    if (comma == std::string::npos) comma = list.size();
    size_t begin = i;
    size_t end = comma;
    while (begin < end &&
           std::isspace(static_cast<unsigned char>(list[begin])))
      ++begin;
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(list[end - 1])))
      --end;
    if (end > begin) devices.push_back(list.substr(begin, end - begin));
    i = comma + 1;
  }
  return devices;
}

std::string JoinDevices(const std::vector<std::string>& devices) {
  std::string out;
  for (const auto& device : devices) {
    if (!out.empty()) out += ",";
    out += device;
  }
  return out;
}

// Reap one signaled request completion and one status RECV per slot. Requests
// may be SEND or WRITE_WITH_IMM; both surface as a send-side completion.
bool ReapWindow(rdma::RcEndpoint& ep, size_t width,
                std::vector<uint32_t>* rbytes, int timeout_ms) {
  rbytes->assign(width, 0);
  std::vector<ibv_wc> wcs(2 * width);
  int need = static_cast<int>(2 * width);
  while (need > 0) {
    int got = ep.WaitComp(wcs.data(), static_cast<int>(2 * width), timeout_ms);
    if (got <= 0) return false;
    for (int i = 0; i < got; ++i) {
      if (wcs[i].status != IBV_WC_SUCCESS) return false;
      if (wcs[i].opcode == IBV_WC_RECV) {
        const size_t slot = static_cast<size_t>(wcs[i].wr_id);
        if (slot >= width) return false;
        (*rbytes)[slot] = wcs[i].byte_len;
      }
      --need;
    }
  }
  return true;
}

// Post w ordinary SEND requests with one status RECV per slot.
bool RunWindow(rdma::RcEndpoint& ep, const std::vector<size_t>& slen,
               std::vector<uint32_t>* rbytes, int timeout_ms) {
  for (size_t j = 0; j < slen.size(); ++j)
    if (!ep.PostRecv(j) || !ep.PostSend(j, slen[j])) return false;
  return ReapWindow(ep, slen.size(), rbytes, timeout_ms);
}
}  // namespace

struct RdmaTransport::Conn {
  rdma::RcEndpoint ep;
  uint8_t protocol_version = kProtoVersionV1;
  rdma::RecvSegmentInfo recv_segment;

  bool v2() const { return protocol_version == kProtoVersionV2; }
  void Encode(char* out, WireOp op, const BlockKey& key, uint64_t offset,
              uint64_t length, uint64_t payload_len) const {
    EncodeReqVersion(out, protocol_version, op, key, offset, length,
                     payload_len);
  }
  bool Decode(const char* in, Status* status, uint64_t* data_len,
              uint64_t max_data = kMaxFrameLen) const {
    return DecodeRespVersion(in, protocol_version, status, data_len, max_data);
  }
  size_t data_capacity() const {
    return recv_segment.slot_size >= rdma::kV2DataOffset
               ? static_cast<size_t>(recv_segment.slot_size -
                                     rdma::kV2DataOffset)
               : 0;
  }
  uint64_t put_addr(size_t slot) const {
    return recv_segment.base_addr +
           slot * recv_segment.slot_size + rdma::kV2PutPrefixOffset;
  }
};

bool RdmaTransport::Available() {
  const char* configured = std::getenv("DFKV_RDMA_DEV");
  const auto filter =
      ParseDeviceList(configured ? std::string(configured) : std::string());
  return !rdma::RdmaTopology::Discover(filter).empty();
}

RdmaTransport::RdmaTransport(size_t max_msg, const std::string& dev_name)
    : max_payload_(ResolveMaxPayload(max_msg)),
      declared_(std::min<uint64_t>(
          EnvBytes("DFKV_RDMA_MAX_BLOCK_BYTES",
                   ResolveMaxPayload(max_msg)),
          ResolveMaxPayload(max_msg))),
      legacy_declared_(std::min<uint64_t>(
          EnvBytes("DFKV_RDMA_MAX_BLOCK_BYTES", 0),
          ResolveMaxPayload(max_msg))),
      control_cap_(ControlCapFor(declared_)),
      depth_(1) {
  std::string list = dev_name;
  if (list.empty()) {
    const char* configured = std::getenv("DFKV_RDMA_DEV");
    if (configured) list = configured;
  }
  const auto filter = ParseDeviceList(list);
  auto_device_ = filter.empty();
  auto discovered = rdma::RdmaTopology::Discover(filter);
  if (discovered.empty()) {
    throw std::runtime_error(
        filter.empty() ? "no ACTIVE RDMA device found"
                       : "no configured RDMA device is ACTIVE");
  }
  if (auto_device_ && discovered.size() > 1) discovered.resize(1);
  devs_.reserve(discovered.size());
  for (const auto& device : discovered) devs_.push_back(device.name);
  topology_ =
      std::make_unique<rdma::RdmaTopology>(std::move(discovered));
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
  config_dump::RecordResolved("DFKV_RDMA_DEV", JoinDevices(devs_));
  config_dump::RecordResolved("DFKV_RDMA_DEPTH", std::to_string(depth_));
  const char* protocol = std::getenv("DFKV_RDMA_PROTOCOL");
  v2_enabled_ = !(protocol && std::strcmp(protocol, "1") == 0);
  config_dump::RecordResolved("DFKV_RDMA_PROTOCOL",
                              v2_enabled_ ? "auto-v2" : "v1");
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
  for (auto& [node, cs] : legacy_control_pool_)
    for (Conn* c : cs) Destroy(c);
}

void RdmaTransport::Destroy(Conn* c) { delete c; }  // RcEndpoint dtor tears down QP/MRs

bool RdmaTransport::ProbeV2(const std::string& node) const {
  if (!v2_enabled_) return false;
  int fd = net::Dial(node, connect_ms_, io_ms_);
  if (fd < 0) return false;
  char frame[rdma::kDevNameBytes];
  const uint64_t declaration =
      declared_ ? declared_ : static_cast<uint64_t>(rdma::kV2ControlCap);
  rdma::EncodeDevFrame(rdma::kV2ProbeDevice, declaration, frame,
                       rdma::kDevProtoV2);
  char reply[rdma::kV2ProbeReplyBytes];
  const bool ok =
      net::WriteAll(fd, frame, sizeof(frame)) &&
      net::ReadAll(fd, reply, sizeof(reply)) &&
      rdma::ParseV2ProbeReply(reply);
  ::close(fd);
  return ok;
}

RdmaTransport::Conn* RdmaTransport::Acquire(const std::string& node, Lane lane,
                                            bool* from_pool, bool force_new) {
  std::vector<std::pair<void*, size_t>> pools;
  Conn* pooled = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    pools = pools_;
    if (!force_new) {
      auto& pool = lane == Lane::kData
                       ? pool_
                       : (lane == Lane::kControl ? control_pool_
                                                 : legacy_control_pool_);
      auto it = pool.find(node);
      if (it != pool.end() && !it->second.empty()) {
        pooled = it->second.back();
        it->second.pop_back();
      }
    }
  }
  if (pooled) {
    pooled->ep.EnsurePoolMrs(
        pools, pooled->protocol_version == kProtoVersionV2);
    *from_pool = true;
    return pooled;
  }
  *from_pool = false;

  const int selected =
      topology_->SelectDevice(numa::CurrentNode(), numa::Enabled());
  if (selected < 0) return nullptr;
  const size_t ridx = static_cast<size_t>(selected);
  const std::string& dev = devs_[ridx];

  bool local_open_failed = false;
  auto connect_protocol = [&](bool protocol_v2) -> Conn* {
    int fd = net::Dial(node, connect_ms_, io_ms_);
    if (fd < 0) return nullptr;

    auto* conn = new Conn();
    const size_t endpoint_cap =
        protocol_v2 ? rdma::kV2ControlCap : control_cap_;
    if (!conn->ep.Open(dev.c_str(), endpoint_cap, depth_)) {
      local_open_failed = true;
      ::close(fd);
      delete conn;
      return nullptr;
    }

    // V2 control lanes declare a tiny bound so they lease only a small shared
    // segment slot. V1 (including automatic fallback) must preserve legacy DCP1
    // semantics: absent DFKV_RDMA_MAX_BLOCK_BYTES means an undeclared plain
    // frame, not a synthetic max_payload_ declaration.
    const uint64_t conn_declared =
        protocol_v2
            ? (lane == Lane::kControl
                   ? static_cast<uint64_t>(rdma::kV2ControlCap)
                   : declared_)
            : legacy_declared_;
    char devbuf[rdma::kDevNameBytes];
    rdma::EncodeDevFrame(
        auto_device_ ? std::string() : dev, conn_declared, devbuf,
        protocol_v2 ? rdma::kDevProtoV2 : rdma::kDevProtoV1);
    char mine[rdma::kQpInfoBytes], peer[rdma::kQpInfoBytes];
    rdma::QpInfo my = conn->ep.Local();
    my.depth = static_cast<uint16_t>(std::min<size_t>(depth_, 256));
    my.protocol_version =
        protocol_v2 ? rdma::kDevProtoV2 : rdma::kDevProtoV1;
    rdma::SerializeQpInfo(my, mine);
    if (!net::WriteAll(fd, devbuf, rdma::kDevNameBytes) ||
        !net::WriteAll(fd, mine, rdma::kQpInfoBytes) ||
        !net::ReadAll(fd, peer, rdma::kQpInfoBytes)) {
      ::close(fd);
      delete conn;
      return nullptr;
    }
    const rdma::QpInfo remote = rdma::ParseQpInfo(peer);
    const uint8_t expected =
        protocol_v2 ? rdma::kDevProtoV2 : rdma::kDevProtoV1;
    if (remote.protocol_version != expected || !conn->ep.Connect(remote)) {
      ::close(fd);
      delete conn;
      return nullptr;
    }
    if (remote.depth > 0) {
      conn->ep.set_remote_depth(remote.depth);
      if (remote.depth < depth_)
        DFKV_LOG_INFO("rdma: server depth " + std::to_string(remote.depth) +
                      " < client depth " + std::to_string(depth_) +
                      ": batching window clamped to " +
                      std::to_string(remote.depth));
    }
    char ready = 0;
    if (!net::ReadAll(fd, &ready, 1) || ready != 1) {
      ::close(fd);
      delete conn;
      return nullptr;
    }
    if (protocol_v2) {
      char encoded[rdma::kRecvSegmentInfoBytes];
      if (!net::ReadAll(fd, encoded, sizeof(encoded)) ||
          !rdma::DecodeRecvSegmentInfo(encoded, &conn->recv_segment)) {
        ::close(fd);
        delete conn;
        return nullptr;
      }
      const size_t expected_slot = rdma::V2SlotSize(
          static_cast<size_t>(std::max<uint64_t>(
              conn_declared, rdma::kV2DataOffset)));
      if (expected_slot == 0 ||
          conn->recv_segment.slot_size != expected_slot) {
        ::close(fd);
        delete conn;
        return nullptr;
      }
    }
    ::close(fd);
    conn->protocol_version = expected;
    conn->ep.EnsurePoolMrs(pools, protocol_v2);
    return conn;
  };

  const bool want_v2 =
      v2_enabled_ && lane != Lane::kLegacyControl &&
      (lane == Lane::kControl || declared_ != 0) &&
      ProbeV2(node);
  Conn* conn = connect_protocol(want_v2);
  if (!conn && want_v2 && !local_open_failed) {
    DFKV_LOG_WARN("rdma: v2 bootstrap failed; retrying v1 for " + node);
    conn = connect_protocol(false);
  }
  if (!conn) {
    if (local_open_failed) {
      topology_->DisableDevice(dev);
      DFKV_LOG_WARN("rdma: disabling device " + dev +
                    " after a local Open failure");
    }
    return nullptr;
  }

  conns_opened_.fetch_add(1, std::memory_order_relaxed);
  if (conn->protocol_version == kProtoVersionV2)
    v2_conns_opened_.fetch_add(1, std::memory_order_relaxed);
  else
    v1_conns_opened_.fetch_add(1, std::memory_order_relaxed);
  if (ridx < devs_.size())
    rail_conns_[ridx].fetch_add(1, std::memory_order_relaxed);
  return conn;
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
  for (auto* ep : anchor_ptrs) ep->EnsurePoolMrs(pools, true);
  // Connections still EnsurePoolMrs on Acquire (no-op once anchored here).
}

std::string RdmaTransport::MetricsText() const {
  std::string s;
  s += "# HELP dfkv_rdma_client_conns_opened_total RDMA client connections opened\n";
  s += "# TYPE dfkv_rdma_client_conns_opened_total counter\n";
  s += "dfkv_rdma_client_conns_opened_total " +
       std::to_string(conns_opened_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_v1_conns_opened_total RDMA v1 connections opened\n";
  s += "# TYPE dfkv_rdma_client_v1_conns_opened_total counter\n";
  s += "dfkv_rdma_client_v1_conns_opened_total " +
       std::to_string(v1_conns_opened_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_v2_conns_opened_total RDMA v2 connections opened\n";
  s += "# TYPE dfkv_rdma_client_v2_conns_opened_total counter\n";
  s += "dfkv_rdma_client_v2_conns_opened_total " +
       std::to_string(v2_conns_opened_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_v2_put_writes_total PUT requests sent with RDMA WRITE_WITH_IMM\n";
  s += "# TYPE dfkv_rdma_client_v2_put_writes_total counter\n";
  s += "dfkv_rdma_client_v2_put_writes_total " +
       std::to_string(v2_put_writes_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_v2_get_writes_total GET requests posted with RDMA WRITE destinations\n";
  s += "# TYPE dfkv_rdma_client_v2_get_writes_total counter\n";
  s += "dfkv_rdma_client_v2_get_writes_total " +
       std::to_string(v2_get_writes_.load(std::memory_order_relaxed)) + "\n";
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
    auto& pool = lane == Lane::kData
                     ? pool_
                     : (lane == Lane::kControl ? control_pool_
                                               : legacy_control_pool_);
    auto& v = pool[node];
    if (v.size() < pool_max_) { v.push_back(c); return; }
  }
  Destroy(c);  // pool full -> drop (and tear down the QP/MRs) instead of growing
}

Status RdmaTransport::RoundTrip(const std::string& node, WireOp op,
                                const BlockKey& key, uint64_t offset,
                                uint64_t length, const void* payload,
                                uint64_t payload_len, std::string* out) {
  const uint64_t value_bound =
      static_cast<uint64_t>(ValueHeader::kSize) + OpBound();
  if (op == WireOp::kCache && payload_len > value_bound)
    return Status::kInvalid;
  if (op == WireOp::kRange && length > value_bound)
    return Status::kInvalid;

  const Lane lane =
      op == WireOp::kCache || op == WireOp::kRange
          ? Lane::kData
          : (op == WireOp::kMembers ? Lane::kLegacyControl : Lane::kControl);
  for (int attempt = 0; attempt < 2; ++attempt) {
    if (out) out->clear();
    bool from_pool = false;
    Conn* conn = Acquire(node, lane, &from_pool, attempt > 0);
    if (!conn) return Status::kIOError;
    rdma::RcEndpoint& ep = conn->ep;

    bool ok = false;
    ibv_mr* transient_output_mr = nullptr;
    if (!conn->v2()) {
      if (payload_len > ep.cap() - kReqPrefix ||
          (op == WireOp::kRange && length > ep.cap() - kRespPrefix)) {
        Release(node, lane, conn);
        return Status::kInvalid;
      }
      conn->Encode(ep.sbuf(0), op, key, offset, length, payload_len);
      if (payload_len)
        std::memcpy(ep.sbuf(0) + kReqPrefix, payload, payload_len);
      ok = ep.PostRecv(0) &&
           ep.PostSend(0, kReqPrefix + static_cast<size_t>(payload_len));
    } else if (op == WireOp::kCache) {
      if (payload_len > conn->data_capacity()) {
        Release(node, lane, conn);
        return Status::kInvalid;
      }
      ibv_mr* payload_mr =
          payload_len
              ? ep.RegisterTransient(const_cast<void*>(payload),
                                     static_cast<size_t>(payload_len),
                                     /*remote_write=*/false)
              : nullptr;
      transient_output_mr = payload_mr;
      if (payload_len != 0 && !payload_mr) {
        Release(node, lane, conn);
        return Status::kIOError;
      }
      conn->Encode(ep.sbuf(0), op, key, offset, length, payload_len);
      ok = ep.PostRecv(0) &&
           ep.PostWriteImmScatter(
               0, kReqPrefix, payload, static_cast<size_t>(payload_len),
               payload_mr, conn->put_addr(0), conn->recv_segment.rkey, 0);
      if (ok)
        v2_put_writes_.fetch_add(1, std::memory_order_relaxed);
    } else if (op == WireOp::kRange) {
      if (!out || length > conn->data_capacity() ||
          length > std::numeric_limits<uint32_t>::max()) {
        Release(node, lane, conn);
        return Status::kInvalid;
      }
      out->resize(static_cast<size_t>(length));
      std::vector<RdmaWriteTarget> targets;
      if (length != 0) {
        transient_output_mr = ep.RegisterTransient(out->data(), out->size());
        if (!transient_output_mr) {
          Release(node, lane, conn);
          return Status::kIOError;
        }
        targets.push_back(
            {reinterpret_cast<uint64_t>(out->data()),
             transient_output_mr->rkey, static_cast<uint32_t>(length)});
      }
      size_t frame_len = 0;
      ok = EncodeRdmaGetReq(ep.sbuf(0), ep.cap(), key, offset, length,
                            /*header_len=*/0, targets, &frame_len) &&
           ep.PostRecv(0) && ep.PostSend(0, frame_len);
      if (ok)
        v2_get_writes_.fetch_add(1, std::memory_order_relaxed);
    } else {
      conn->Encode(ep.sbuf(0), op, key, offset, length, payload_len);
      ok = ep.PostRecv(0) &&
           ep.PostSend(0, kReqPrefix + static_cast<size_t>(payload_len));
    }

    uint32_t recv_bytes = 0;
    bool need_request = ok;
    bool need_recv = ok;
    while (ok && (need_request || need_recv)) {
      ibv_wc completion{};
      int got = ep.WaitComp(&completion, 1, op_timeout_ms_);
      if (got <= 0 || completion.status != IBV_WC_SUCCESS) {
        ok = false;
        break;
      }
      if (completion.opcode == IBV_WC_RECV) {
        need_recv = false;
        recv_bytes = completion.byte_len;
      } else {
        need_request = false;
      }
    }
    if (ok && transient_output_mr) {
      ep.ReleaseTransient(transient_output_mr);
      transient_output_mr = nullptr;
    }

    if (!ok) {
      Destroy(conn);
      if (!from_pool) return Status::kIOError;
      continue;
    }
    if (recv_bytes < kRespPrefix) {
      Destroy(conn);
      return Status::kIOError;
    }
    Status status;
    uint64_t data_len = 0;
    if (!conn->Decode(ep.rbuf(0), &status, &data_len)) {
      Destroy(conn);
      return Status::kIOError;
    }
    if (out) {
      if (conn->v2() && op == WireOp::kRange) {
        if (status == Status::kOk && data_len <= out->size()) {
          out->resize(static_cast<size_t>(data_len));
        } else if (status != Status::kOk) {
          out->clear();
        } else {
          Destroy(conn);
          return Status::kIOError;
        }
      } else {
        if (kRespPrefix + data_len > recv_bytes) {
          Destroy(conn);
          return Status::kIOError;
        }
        out->assign(ep.rbuf(0) + kRespPrefix,
                    static_cast<size_t>(data_len));
      }
    }
    Release(node, lane, conn);
    return status;
  }
  return Status::kIOError;
}

Status RdmaTransport::Cache(const std::string& node, const BlockKey& key,
                            const void* data, size_t len) {
  const size_t value_bytes =
      len > ValueHeader::kSize ? len - ValueHeader::kSize : len;
  if (NoteBlock(value_bytes)) return Status::kInvalid;
  return RoundTrip(node, WireOp::kCache, key, 0, 0, data, len, nullptr);
}
Status RdmaTransport::Range(const std::string& node, const BlockKey& key,
                            uint64_t offset, uint64_t length,
                            std::string* out) {
  const uint64_t value_bytes =
      length > ValueHeader::kSize ? length - ValueHeader::kSize : length;
  if (value_bytes > std::numeric_limits<size_t>::max() ||
      NoteBlock(static_cast<size_t>(value_bytes)))
    return Status::kInvalid;
  return RoundTrip(node, WireOp::kRange, key, offset, length, nullptr, 0, out);
}
Status RdmaTransport::Exist(const std::string& node, const BlockKey& key, bool* exist) {
  Status st = RoundTrip(node, WireOp::kExist, key, 0, 0, nullptr, 0, nullptr);
  if (st == Status::kOk) { *exist = true; return Status::kOk; }
  if (st == Status::kNotFound) { *exist = false; return Status::kOk; }
  return st;
}

Status RdmaTransport::Remove(const std::string& node, const BlockKey& key) {
  return RoundTrip(node, WireOp::kRemove, key, 0, 0, nullptr, 0, nullptr);
}

std::vector<Status> RdmaTransport::CacheMany(
    const std::string& node, const std::vector<CacheItem>& items) {
  const size_t count = items.size();
  std::vector<Status> result(count, Status::kIOError);
  if (count == 0) return result;
  for (const auto& item : items) {
    const size_t value_bytes =
        item.len > ValueHeader::kSize ? item.len - ValueHeader::kSize
                                     : item.len;
    if (NoteBlock(value_bytes)) {
      std::fill(result.begin(), result.end(), Status::kInvalid);
      return result;
    }
  }

  for (int attempt = 0; attempt < 2; ++attempt) {
    std::fill(result.begin(), result.end(), Status::kIOError);
    bool from_pool = false;
    Conn* conn = Acquire(node, Lane::kData, &from_pool, attempt > 0);
    if (!conn) return result;
    rdma::RcEndpoint& ep = conn->ep;
    const size_t window = ep.window();
    bool conn_ok = true;
    for (size_t base = 0; base < count && conn_ok; base += window) {
      const size_t width = std::min(window, count - base);
      std::vector<uint32_t> reply_bytes;
      if (!conn->v2()) {
        std::vector<size_t> send_lengths(width);
        for (size_t slot = 0; slot < width; ++slot) {
          const CacheItem& item = items[base + slot];
          if (item.len > ep.cap() - kReqPrefix) {
            conn_ok = false;
            break;
          }
          conn->Encode(ep.sbuf(slot), WireOp::kCache, item.key, 0, 0,
                       item.len);
          if (item.len)
            std::memcpy(ep.sbuf(slot) + kReqPrefix, item.data, item.len);
          send_lengths[slot] = kReqPrefix + item.len;
        }
        if (conn_ok &&
            !RunWindow(ep, send_lengths, &reply_bytes, op_timeout_ms_))
          conn_ok = false;
      } else {
        std::vector<ibv_mr*> payload_mrs(width, nullptr);
        for (size_t slot = 0; slot < width; ++slot) {
          const CacheItem& item = items[base + slot];
          if (item.len > conn->data_capacity()) {
            conn_ok = false;
            break;
          }
          ibv_mr* mr =
              item.len
                  ? ep.RegisterTransient(const_cast<void*>(item.data), item.len,
                                         /*remote_write=*/false)
                  : nullptr;
          payload_mrs[slot] = mr;
          if ((item.len != 0 && !mr) ||
              !ep.PostRecv(slot)) {
            conn_ok = false;
            break;
          }
          conn->Encode(ep.sbuf(slot), WireOp::kCache, item.key, 0, 0,
                       item.len);
          if (!ep.PostWriteImmScatter(
                  slot, kReqPrefix, item.data, item.len, mr,
                  conn->put_addr(slot), conn->recv_segment.rkey,
                  static_cast<uint32_t>(slot))) {
            conn_ok = false;
            break;
          }
          v2_put_writes_.fetch_add(1, std::memory_order_relaxed);
        }
        if (conn_ok &&
            !ReapWindow(ep, width, &reply_bytes, op_timeout_ms_))
          conn_ok = false;
        if (conn_ok) {
          for (ibv_mr* mr : payload_mrs) ep.ReleaseTransient(mr);
        }
      }
      if (!conn_ok) break;
      for (size_t slot = 0; slot < width; ++slot) {
        Status status;
        uint64_t data_len = 0;
        if (reply_bytes[slot] >= kRespPrefix &&
            conn->Decode(ep.rbuf(slot), &status, &data_len))
          result[base + slot] = status;
      }
    }
    if (conn_ok) {
      Release(node, Lane::kData, conn);
      return result;
    }
    Destroy(conn);
    if (from_pool) continue;
    return result;
  }
  return result;
}

Status RdmaTransport::Members(const std::string& node, std::string* out) {
  return RoundTrip(node, WireOp::kMembers, BlockKey{}, 0, 0, nullptr, 0, out);
}


std::vector<Status> RdmaTransport::RangeMany(
    const std::string& node, const std::vector<BlockKey>& keys,
    uint64_t offset, uint64_t length, std::vector<std::string>* outputs) {
  const size_t count = keys.size();
  outputs->assign(count, std::string());
  std::vector<Status> result(count, Status::kIOError);
  if (count == 0) return result;
  const uint64_t value_bytes =
      length > ValueHeader::kSize ? length - ValueHeader::kSize : length;
  if (length > std::numeric_limits<uint32_t>::max() ||
      value_bytes > std::numeric_limits<size_t>::max() ||
      NoteBlock(static_cast<size_t>(value_bytes))) {
    std::fill(result.begin(), result.end(), Status::kInvalid);
    return result;
  }

  for (int attempt = 0; attempt < 2; ++attempt) {
    std::fill(result.begin(), result.end(), Status::kIOError);
    outputs->assign(count, std::string());
    bool from_pool = false;
    Conn* conn = Acquire(node, Lane::kData, &from_pool, attempt > 0);
    if (!conn) return result;
    rdma::RcEndpoint& ep = conn->ep;
    const size_t window = ep.window();
    bool conn_ok = true;
    for (size_t base = 0; base < count && conn_ok; base += window) {
      const size_t width = std::min(window, count - base);
      std::vector<uint32_t> reply_bytes;
      if (!conn->v2()) {
        if (length > ep.cap() - kRespPrefix) {
          std::fill(result.begin(), result.end(), Status::kInvalid);
          Release(node, Lane::kData, conn);
          return result;
        }
        std::vector<size_t> send_lengths(width, kReqPrefix);
        for (size_t slot = 0; slot < width; ++slot)
          conn->Encode(ep.sbuf(slot), WireOp::kRange, keys[base + slot],
                       offset, length, 0);
        if (!RunWindow(ep, send_lengths, &reply_bytes, op_timeout_ms_))
          conn_ok = false;
      } else {
        std::vector<ibv_mr*> transient_output_mrs(width, nullptr);
        for (size_t slot = 0; slot < width; ++slot) {
          std::string& output = (*outputs)[base + slot];
          output.resize(static_cast<size_t>(length));
          std::vector<RdmaWriteTarget> targets;
          if (length != 0) {
            ibv_mr* mr = ep.RegisterTransient(output.data(), output.size());
            if (!mr) {
              conn_ok = false;
              break;
            }
            transient_output_mrs[slot] = mr;
            targets.push_back(
                {reinterpret_cast<uint64_t>(output.data()), mr->rkey,
                 static_cast<uint32_t>(length)});
          }
          size_t frame_len = 0;
          if (!EncodeRdmaGetReq(ep.sbuf(slot), ep.cap(),
                                keys[base + slot], offset, length, 0,
                                targets, &frame_len) ||
              !ep.PostRecv(slot) || !ep.PostSend(slot, frame_len)) {
            conn_ok = false;
            break;
          }
          v2_get_writes_.fetch_add(1, std::memory_order_relaxed);
        }
        if (conn_ok &&
            !ReapWindow(ep, width, &reply_bytes, op_timeout_ms_))
          conn_ok = false;
        if (conn_ok) {
          for (ibv_mr* mr : transient_output_mrs)
            ep.ReleaseTransient(mr);
        }
      }
      if (!conn_ok) break;

      for (size_t slot = 0; slot < width; ++slot) {
        const uint32_t received = reply_bytes[slot];
        if (received < kRespPrefix) continue;
        Status status;
        uint64_t data_len = 0;
        if (!conn->Decode(ep.rbuf(slot), &status, &data_len)) continue;
        result[base + slot] = status;
        std::string& output = (*outputs)[base + slot];
        if (status != Status::kOk) {
          output.clear();
        } else if (conn->v2()) {
          if (data_len <= output.size())
            output.resize(static_cast<size_t>(data_len));
          else
            result[base + slot] = Status::kIOError;
        } else if (kRespPrefix + data_len <= received) {
          output.assign(ep.rbuf(slot) + kRespPrefix,
                        static_cast<size_t>(data_len));
        } else {
          result[base + slot] = Status::kIOError;
        }
      }
    }
    if (conn_ok) {
      Release(node, Lane::kData, conn);
      return result;
    }
    Destroy(conn);
    if (from_pool) continue;
    return result;
  }
  return result;
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
        c->Encode(ep.sbuf(j), WireOp::kExist, keys[base + j], 0, 0, 0);
        slen[j] = kReqPrefix;
      }
      std::vector<uint32_t> rbytes;
      if (!RunWindow(ep, slen, &rbytes, op_timeout_ms_)) { conn_ok = false; break; }
      for (size_t j = 0; j < w; ++j) {
        if (rbytes[j] < kRespPrefix) continue;
        Status st; uint64_t dl = 0;
        if (!c->Decode(ep.rbuf(j), &st, &dl)) continue;
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

std::vector<Status> RdmaTransport::RangeInto(
    const std::string& node, const std::vector<BlockKey>& keys,
    const std::vector<RangeDst>& destinations, size_t header_size,
    std::vector<std::string>* headers) {
  const size_t count = keys.size();
  headers->assign(count, std::string());
  std::vector<Status> result(count, Status::kIOError);
  if (count == 0) return result;
  if (header_size > rdma::kV2ControlCap - kRespPrefix) {
    std::fill(result.begin(), result.end(), Status::kInvalid);
    return result;
  }
  const size_t status_bytes = kRespPrefix + header_size;
  std::vector<char> bad(count, 0);
  for (size_t i = 0; i < count; ++i) {
    if (destinations[i].n > std::numeric_limits<uint32_t>::max() ||
        destinations[i].n >
            std::numeric_limits<size_t>::max() - header_size ||
        NoteBlock(destinations[i].n))
      bad[i] = 1;
  }

  for (int attempt = 0; attempt < 2; ++attempt) {
    std::fill(result.begin(), result.end(), Status::kIOError);
    for (size_t i = 0; i < count; ++i)
      if (bad[i]) result[i] = Status::kInvalid;
    headers->assign(count, std::string());
    bool from_pool = false;
    Conn* conn = Acquire(node, Lane::kData, &from_pool, attempt > 0);
    if (!conn) return result;
    rdma::RcEndpoint& ep = conn->ep;
    const size_t window = ep.window();
    bool conn_ok = true;
    for (size_t base = 0; base < count && conn_ok; base += window) {
      const size_t width = std::min(window, count - base);
      std::vector<ibv_mr*> mrs(width, nullptr);
      for (size_t slot = 0; slot < width; ++slot) {
        if (bad[base + slot] || destinations[base + slot].n == 0)
          continue;
        mrs[slot] = ep.RegisterUser(destinations[base + slot].payload,
                                    destinations[base + slot].n);
        if (!mrs[slot]) {
          for (size_t i = base; i < count; ++i)
            if (!bad[i]) result[i] = Status::kInvalid;
          Release(node, Lane::kData, conn);
          return result;
        }
      }

      size_t posted = 0;
      for (size_t slot = 0; slot < width; ++slot) {
        if (bad[base + slot]) continue;
        const RangeDst& destination = destinations[base + slot];
        bool request_ok = false;
        if (!conn->v2()) {
          const bool armed =
              destination.n
                  ? ep.PostRecvScatter(slot, destination.payload,
                                       destination.n, mrs[slot], status_bytes)
                  : ep.PostRecv(slot);
          conn->Encode(ep.sbuf(slot), WireOp::kRange, keys[base + slot],
                       0, header_size + destination.n, 0);
          request_ok = armed && ep.PostSend(slot, kReqPrefix);
        } else {
          std::vector<RdmaWriteTarget> targets;
          if (destination.n != 0)
            targets.push_back(
                {reinterpret_cast<uint64_t>(destination.payload),
                 mrs[slot]->rkey,
                 static_cast<uint32_t>(destination.n)});
          size_t frame_len = 0;
          request_ok =
              EncodeRdmaGetReq(
                  ep.sbuf(slot), ep.cap(), keys[base + slot], 0,
                  header_size + destination.n,
                  static_cast<uint32_t>(header_size), targets, &frame_len) &&
              ep.PostRecv(slot) && ep.PostSend(slot, frame_len);
          if (request_ok)
            v2_get_writes_.fetch_add(1, std::memory_order_relaxed);
        }
        if (!request_ok) {
          conn_ok = false;
          break;
        }
        ++posted;
      }
      if (!conn_ok) break;
      if (posted == 0) continue;

      std::vector<ibv_wc> completions(2 * posted);
      std::vector<uint32_t> reply_bytes(width, 0);
      int needed = static_cast<int>(2 * posted);
      while (needed > 0) {
        int got = ep.WaitComp(completions.data(),
                              static_cast<int>(2 * posted), BatchTimeout());
        if (got <= 0) {
          conn_ok = false;
          break;
        }
        for (int i = 0; i < got; ++i) {
          if (completions[i].status != IBV_WC_SUCCESS) {
            conn_ok = false;
            break;
          }
          if (completions[i].opcode == IBV_WC_RECV) {
            const size_t slot =
                static_cast<size_t>(completions[i].wr_id);
            if (slot >= width) {
              conn_ok = false;
              break;
            }
            reply_bytes[slot] = completions[i].byte_len;
          }
          --needed;
        }
        if (!conn_ok) break;
      }
      if (!conn_ok) break;

      for (size_t slot = 0; slot < width; ++slot) {
        if (bad[base + slot] || reply_bytes[slot] < kRespPrefix) continue;
        Status status;
        uint64_t data_len = 0;
        if (!conn->Decode(ep.rbuf(slot), &status, &data_len)) continue;
        result[base + slot] = status;
        if (status == Status::kOk) {
          if (reply_bytes[slot] >= status_bytes &&
              data_len >= header_size &&
              data_len - header_size <= destinations[base + slot].n) {
            (*headers)[base + slot].assign(
                ep.rbuf(slot) + kRespPrefix, header_size);
          } else {
            result[base + slot] = Status::kIOError;
          }
        }
      }
    }
    if (conn_ok) {
      Release(node, Lane::kData, conn);
      return result;
    }
    Destroy(conn);
    if (from_pool) continue;
    return result;
  }
  return result;
}

std::vector<Status> RdmaTransport::CacheFrom(
    const std::string& node, const std::vector<CacheSrc>& sources) {
  const size_t count = sources.size();
  std::vector<Status> result(count, Status::kIOError);
  if (count == 0) return result;
  std::vector<char> bad(count, 0);
  for (size_t i = 0; i < count; ++i) {
    const CacheSrc& source = sources[i];
    if (source.header_len > rdma::kV2ControlCap - kReqPrefix ||
        source.payload_len >
            std::numeric_limits<size_t>::max() - source.header_len ||
        NoteBlock(source.payload_len))
      bad[i] = 1;
  }

  for (int attempt = 0; attempt < 2; ++attempt) {
    std::fill(result.begin(), result.end(), Status::kIOError);
    for (size_t i = 0; i < count; ++i)
      if (bad[i]) result[i] = Status::kInvalid;
    bool from_pool = false;
    Conn* conn = Acquire(node, Lane::kData, &from_pool, attempt > 0);
    if (!conn) return result;
    rdma::RcEndpoint& ep = conn->ep;
    const size_t window = ep.window();
    bool conn_ok = true;
    for (size_t base = 0; base < count && conn_ok; base += window) {
      const size_t width = std::min(window, count - base);
      std::vector<ibv_mr*> mrs(width, nullptr);
      for (size_t slot = 0; slot < width; ++slot) {
        const CacheSrc& source = sources[base + slot];
        if (bad[base + slot] || source.payload_len == 0) continue;
        mrs[slot] = ep.RegisterUser(
            const_cast<void*>(source.payload), source.payload_len);
        if (!mrs[slot]) {
          for (size_t i = base; i < count; ++i)
            if (!bad[i]) result[i] = Status::kInvalid;
          Release(node, Lane::kData, conn);
          return result;
        }
      }

      size_t posted = 0;
      for (size_t slot = 0; slot < width; ++slot) {
        if (bad[base + slot]) continue;
        const CacheSrc& source = sources[base + slot];
        const size_t stored_len = source.header_len + source.payload_len;
        if (conn->v2() && stored_len > conn->data_capacity()) {
          result[base + slot] = Status::kInvalid;
          continue;
        }
        conn->Encode(ep.sbuf(slot), WireOp::kCache, source.key, 0, 0,
                     stored_len);
        if (source.header_len)
          std::memcpy(ep.sbuf(slot) + kReqPrefix, source.header,
                      source.header_len);
        if (!ep.PostRecv(slot)) {
          conn_ok = false;
          break;
        }
        bool sent = false;
        if (!conn->v2()) {
          sent = ep.PostSendScatter(
              slot, kReqPrefix + source.header_len, source.payload,
              source.payload_len, mrs[slot]);
        } else {
          sent = ep.PostWriteImmScatter(
              slot, kReqPrefix + source.header_len, source.payload,
              source.payload_len, mrs[slot], conn->put_addr(slot),
              conn->recv_segment.rkey, static_cast<uint32_t>(slot));
          if (sent)
            v2_put_writes_.fetch_add(1, std::memory_order_relaxed);
        }
        if (!sent) {
          conn_ok = false;
          break;
        }
        ++posted;
      }
      if (!conn_ok) break;
      if (posted == 0) continue;

      std::vector<ibv_wc> completions(2 * posted);
      std::vector<uint32_t> reply_bytes(width, 0);
      int needed = static_cast<int>(2 * posted);
      while (needed > 0) {
        int got = ep.WaitComp(completions.data(),
                              static_cast<int>(2 * posted), BatchTimeout());
        if (got <= 0) {
          conn_ok = false;
          break;
        }
        for (int i = 0; i < got; ++i) {
          if (completions[i].status != IBV_WC_SUCCESS) {
            conn_ok = false;
            break;
          }
          if (completions[i].opcode == IBV_WC_RECV) {
            const size_t slot =
                static_cast<size_t>(completions[i].wr_id);
            if (slot >= width) {
              conn_ok = false;
              break;
            }
            reply_bytes[slot] = completions[i].byte_len;
          }
          --needed;
        }
        if (!conn_ok) break;
      }
      if (!conn_ok) break;
      for (size_t slot = 0; slot < width; ++slot) {
        if (bad[base + slot] ||
            result[base + slot] == Status::kInvalid)
          continue;
        Status status;
        uint64_t data_len = 0;
        if (reply_bytes[slot] >= kRespPrefix &&
            conn->Decode(ep.rbuf(slot), &status, &data_len))
          result[base + slot] = status;
      }
    }
    if (conn_ok) {
      Release(node, Lane::kData, conn);
      return result;
    }
    Destroy(conn);
    if (from_pool) continue;
    return result;
  }
  return result;
}

std::vector<Status> RdmaTransport::CacheFromMulti(
    const std::string& node, const std::vector<CacheSrcMulti>& sources) {
  const size_t count = sources.size();
  std::vector<Status> result(count, Status::kIOError);
  if (count == 0) return result;

  for (int attempt = 0; attempt < 2; ++attempt) {
    std::fill(result.begin(), result.end(), Status::kIOError);
    bool from_pool = false;
    Conn* conn = Acquire(node, Lane::kData, &from_pool, attempt > 0);
    if (!conn) return result;
    rdma::RcEndpoint& ep = conn->ep;
    const size_t max_payload_segments = ep.max_sge() - 1;
    std::vector<char> bad(count, 0);
    std::vector<size_t> totals(count, 0);
    for (size_t i = 0; i < count; ++i) {
      const CacheSrcMulti& source = sources[i];
      bool invalid =
          source.header_len > rdma::kV2ControlCap - kReqPrefix ||
          source.payloads.size() > max_payload_segments;
      size_t total = 0;
      for (const auto& payload : source.payloads) {
        if (payload.second > std::numeric_limits<uint32_t>::max() ||
            payload.second > std::numeric_limits<size_t>::max() - total) {
          invalid = true;
          break;
        }
        total += payload.second;
      }
      if (!invalid && (total >
                           std::numeric_limits<size_t>::max() -
                               source.header_len ||
                       NoteBlock(total)))
        invalid = true;
      totals[i] = total;
      if (invalid) {
        bad[i] = 1;
        result[i] = Status::kInvalid;
      }
    }

    const size_t window = ep.window();
    bool conn_ok = true;
    for (size_t base = 0; base < count && conn_ok; base += window) {
      const size_t width = std::min(window, count - base);
      std::vector<std::vector<ibv_mr*>> mrs(width);
      for (size_t slot = 0; slot < width; ++slot) {
        if (bad[base + slot]) continue;
        const CacheSrcMulti& source = sources[base + slot];
        mrs[slot].assign(source.payloads.size(), nullptr);
        for (size_t i = 0; i < source.payloads.size(); ++i) {
          if (source.payloads[i].second == 0) continue;
          mrs[slot][i] = ep.RegisterUser(
              const_cast<void*>(source.payloads[i].first),
              source.payloads[i].second);
          if (!mrs[slot][i]) {
            Release(node, Lane::kData, conn);
            return result;
          }
        }
      }

      size_t posted = 0;
      for (size_t slot = 0; slot < width; ++slot) {
        if (bad[base + slot]) continue;
        const CacheSrcMulti& source = sources[base + slot];
        const size_t stored_len = source.header_len + totals[base + slot];
        if (conn->v2() && stored_len > conn->data_capacity()) {
          result[base + slot] = Status::kInvalid;
          continue;
        }
        std::vector<std::pair<const void*, uint32_t>> segments;
        segments.reserve(source.payloads.size());
        for (const auto& payload : source.payloads)
          segments.emplace_back(payload.first,
                                static_cast<uint32_t>(payload.second));

        conn->Encode(ep.sbuf(slot), WireOp::kCache, source.key, 0, 0,
                     stored_len);
        if (source.header_len)
          std::memcpy(ep.sbuf(slot) + kReqPrefix, source.header,
                      source.header_len);
        if (!ep.PostRecv(slot)) {
          conn_ok = false;
          break;
        }
        bool sent = false;
        if (!conn->v2()) {
          sent = ep.PostSendScatterMulti(
              slot, kReqPrefix + source.header_len, segments, mrs[slot]);
        } else {
          sent = ep.PostWriteImmScatterMulti(
              slot, kReqPrefix + source.header_len, segments, mrs[slot],
              conn->put_addr(slot), conn->recv_segment.rkey,
              static_cast<uint32_t>(slot));
          if (sent)
            v2_put_writes_.fetch_add(1, std::memory_order_relaxed);
        }
        if (!sent) {
          conn_ok = false;
          break;
        }
        ++posted;
      }
      if (!conn_ok) break;
      if (posted == 0) continue;

      std::vector<ibv_wc> completions(2 * posted);
      std::vector<uint32_t> reply_bytes(width, 0);
      int needed = static_cast<int>(2 * posted);
      while (needed > 0) {
        int got = ep.WaitComp(completions.data(),
                              static_cast<int>(2 * posted), BatchTimeout());
        if (got <= 0) {
          conn_ok = false;
          break;
        }
        for (int i = 0; i < got; ++i) {
          if (completions[i].status != IBV_WC_SUCCESS) {
            conn_ok = false;
            break;
          }
          if (completions[i].opcode == IBV_WC_RECV) {
            const size_t slot =
                static_cast<size_t>(completions[i].wr_id);
            if (slot >= width) {
              conn_ok = false;
              break;
            }
            reply_bytes[slot] = completions[i].byte_len;
          }
          --needed;
        }
        if (!conn_ok) break;
      }
      if (!conn_ok) break;
      for (size_t slot = 0; slot < width; ++slot) {
        if (bad[base + slot] ||
            result[base + slot] == Status::kInvalid)
          continue;
        Status status;
        uint64_t data_len = 0;
        if (reply_bytes[slot] >= kRespPrefix &&
            conn->Decode(ep.rbuf(slot), &status, &data_len))
          result[base + slot] = status;
      }
    }
    if (conn_ok) {
      Release(node, Lane::kData, conn);
      return result;
    }
    Destroy(conn);
    if (from_pool) continue;
    return result;
  }
  return result;
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
    const std::vector<RangeDstMulti>& destinations, size_t header_size,
    std::vector<std::string>* headers, std::vector<size_t>* out_lengths) {
  const size_t count = keys.size();
  headers->assign(count, std::string());
  if (out_lengths) out_lengths->assign(count, 0);
  std::vector<Status> result(count, Status::kIOError);
  if (count == 0) return result;
  if (header_size > rdma::kV2ControlCap - kRespPrefix) {
    std::fill(result.begin(), result.end(), Status::kInvalid);
    return result;
  }
  const size_t status_bytes = kRespPrefix + header_size;

  for (int attempt = 0; attempt < 2; ++attempt) {
    std::fill(result.begin(), result.end(), Status::kIOError);
    headers->assign(count, std::string());
    if (out_lengths) out_lengths->assign(count, 0);
    bool from_pool = false;
    Conn* conn = Acquire(node, Lane::kData, &from_pool, attempt > 0);
    if (!conn) return result;
    rdma::RcEndpoint& ep = conn->ep;
    const size_t max_payload_segments =
        conn->v2() ? rdma::kV2MaxGetTargets : ep.max_sge() - 1;
    std::vector<char> bad(count, 0);
    std::vector<size_t> capacities(count, 0);
    for (size_t i = 0; i < count; ++i) {
      bool invalid =
          destinations[i].payloads.size() > max_payload_segments;
      size_t capacity = 0;
      for (const auto& destination : destinations[i].payloads) {
        if (destination.second > std::numeric_limits<uint32_t>::max() ||
            destination.second >
                std::numeric_limits<size_t>::max() - capacity) {
          invalid = true;
          break;
        }
        capacity += destination.second;
      }
      if (!invalid &&
          (capacity > std::numeric_limits<size_t>::max() - header_size ||
           NoteBlock(capacity)))
        invalid = true;
      capacities[i] = capacity;
      if (invalid) {
        bad[i] = 1;
        result[i] = Status::kInvalid;
      }
    }

    const size_t window = ep.window();
    bool conn_ok = true;
    for (size_t base = 0; base < count && conn_ok; base += window) {
      const size_t width = std::min(window, count - base);
      std::vector<std::vector<ibv_mr*>> mrs(width);
      for (size_t slot = 0; slot < width; ++slot) {
        if (bad[base + slot]) continue;
        const RangeDstMulti& destination = destinations[base + slot];
        mrs[slot].assign(destination.payloads.size(), nullptr);
        for (size_t i = 0; i < destination.payloads.size(); ++i) {
          if (destination.payloads[i].second == 0) continue;
          mrs[slot][i] = ep.RegisterUser(
              destination.payloads[i].first,
              destination.payloads[i].second);
          if (!mrs[slot][i]) {
            Release(node, Lane::kData, conn);
            return result;
          }
        }
      }

      size_t posted = 0;
      for (size_t slot = 0; slot < width; ++slot) {
        if (bad[base + slot]) continue;
        const RangeDstMulti& destination = destinations[base + slot];
        bool request_ok = false;
        if (!conn->v2()) {
          bool armed = false;
          if (capacities[base + slot] != 0) {
            std::vector<std::pair<void*, uint32_t>> segments;
            segments.reserve(destination.payloads.size());
            for (const auto& payload : destination.payloads)
              segments.emplace_back(
                  payload.first, static_cast<uint32_t>(payload.second));
            armed = ep.PostRecvScatterMulti(
                slot, segments, mrs[slot], status_bytes);
          } else {
            armed = ep.PostRecv(slot);
          }
          conn->Encode(
              ep.sbuf(slot), WireOp::kRange, keys[base + slot], 0,
              header_size + capacities[base + slot], 0);
          request_ok = armed && ep.PostSend(slot, kReqPrefix);
        } else {
          std::vector<RdmaWriteTarget> targets;
          targets.reserve(destination.payloads.size());
          for (size_t i = 0; i < destination.payloads.size(); ++i) {
            const auto& payload = destination.payloads[i];
            if (payload.second == 0) continue;
            targets.push_back(
                {reinterpret_cast<uint64_t>(payload.first),
                 mrs[slot][i]->rkey,
                 static_cast<uint32_t>(payload.second)});
          }
          size_t frame_len = 0;
          request_ok =
              EncodeRdmaGetReq(
                  ep.sbuf(slot), ep.cap(), keys[base + slot], 0,
                  header_size + capacities[base + slot],
                  static_cast<uint32_t>(header_size), targets, &frame_len) &&
              ep.PostRecv(slot) && ep.PostSend(slot, frame_len);
          if (request_ok)
            v2_get_writes_.fetch_add(1, std::memory_order_relaxed);
        }
        if (!request_ok) {
          conn_ok = false;
          break;
        }
        ++posted;
      }
      if (!conn_ok) break;
      if (posted == 0) continue;

      std::vector<ibv_wc> completions(2 * posted);
      std::vector<uint32_t> reply_bytes(width, 0);
      int needed = static_cast<int>(2 * posted);
      while (needed > 0) {
        int got = ep.WaitComp(completions.data(),
                              static_cast<int>(2 * posted), BatchTimeout());
        if (got <= 0) {
          conn_ok = false;
          break;
        }
        for (int i = 0; i < got; ++i) {
          if (completions[i].status != IBV_WC_SUCCESS) {
            conn_ok = false;
            break;
          }
          if (completions[i].opcode == IBV_WC_RECV) {
            const size_t slot =
                static_cast<size_t>(completions[i].wr_id);
            if (slot >= width) {
              conn_ok = false;
              break;
            }
            reply_bytes[slot] = completions[i].byte_len;
          }
          --needed;
        }
        if (!conn_ok) break;
      }
      if (!conn_ok) break;

      for (size_t slot = 0; slot < width; ++slot) {
        if (bad[base + slot] || reply_bytes[slot] < kRespPrefix)
          continue;
        Status status;
        uint64_t data_len = 0;
        if (!conn->Decode(ep.rbuf(slot), &status, &data_len)) continue;
        result[base + slot] = status;
        if (status == Status::kOk) {
          const uint64_t payload_len =
              data_len >= header_size ? data_len - header_size
                                      : std::numeric_limits<uint64_t>::max();
          if (reply_bytes[slot] >= status_bytes &&
              payload_len <= capacities[base + slot]) {
            (*headers)[base + slot].assign(
                ep.rbuf(slot) + kRespPrefix, header_size);
            if (out_lengths)
              (*out_lengths)[base + slot] =
                  static_cast<size_t>(payload_len);
          } else {
            result[base + slot] = Status::kIOError;
          }
        }
      }
    }
    if (conn_ok) {
      Release(node, Lane::kData, conn);
      return result;
    }
    Destroy(conn);
    if (from_pool) continue;
    return result;
  }
  return result;
}

}  // namespace dfkv
