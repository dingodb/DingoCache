#include "common/config_dump.h"
#include "cache/rdma_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "utils/log.h"          // DFKV_LOG_WARN (uring init fallback)
#include "utils/net_util.h"     // ReadAll / WriteAll / Get*/Put*
#include "utils/thread_name.h"
#include "utils/numa_util.h"    // pin serve thread to the device's NUMA node
#include "utils/wire_limits.h"  // ResolveMaxPayload (shared with the TCP path)
#include "transport/rdma_verbs.h"   // RcEndpoint, QpInfo
#include "transport/rdma_protocol.h"
#include "transport/rdma_topology.h"
#include "transport/transport.h"    // kReqPrefix, kRespPrefix
#include "cache/uring_reader.h" // io_uring async-GET path (DFKV_WITH_URING only)
#include "common/value_header.h"

namespace dfkv {

namespace {
// EnvBytes/ResolveMaxPayload live in utils/wire_limits.h so the TCP request
// path (kv_node_server) bounds its frames with the SAME resolved max value
// this RDMA server enforces (wire_limits::kIoAlign == rdma::kDirectIoAlign;
// static_assert below keeps that true).
static_assert(wire_limits::kIoAlign == rdma::kDirectIoAlign,
              "wire_limits must mirror the RDMA direct-IO alignment");
using wire_limits::ResolveMaxPayload;

// Monotonic seconds for the async-read submit->complete latency stamp. Read
// twice per deferred GET (prep + completion), both off the SSD-bound path, so
// the vDSO clock read is amortized away.
inline double NowSteadySec() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

size_t ControlCapFor(size_t max_payload) {
  constexpr size_t kDefaultControlCap = 8u << 20;
  constexpr size_t kMinControlCap = kReqPrefix + ValueHeader::kSize;
  size_t cap = std::min(kDefaultControlCap, max_payload);
  return cap < kMinControlCap ? kMinControlCap : cap;
}

size_t RecvSegmentBytes() {
  constexpr size_t kDefault = 2ull << 30;
  const size_t bytes = rdma::ResolveRecvSegmentBytes(
      std::getenv("DFKV_RDMA_RECV_SEGMENT_SIZE"), kDefault,
      rdma::kV2DataOffset);
  config_dump::RecordResolved("DFKV_RDMA_RECV_SEGMENT_SIZE",
                              std::to_string(bytes));
  return bytes;
}
}  // namespace

RdmaServer::RdmaServer(Handler handler, size_t max_msg,
                       const std::string& dev_name,
                       ProtocolMode protocol_mode)
    : handler_(std::move(handler)),
      max_msg_(ResolveMaxPayload(max_msg)),
      control_cap_(ControlCapFor(max_msg_)),
      dev_name_(dev_name) {
  if (dev_name_.empty()) {
    const char* e = std::getenv("DFKV_RDMA_DEV");
    if (e && *e) dev_name_ = e;
  }
  auto_device_ = dev_name_.empty();
  config_dump::RecordResolved("DFKV_RDMA_DEV",
                              dev_name_.empty() ? "(auto)" : dev_name_);
  if (protocol_mode == ProtocolMode::kSgEngineV1) {
    legacy_wire_ = true;
    v2_enabled_ = false;
    config_dump::RecordResolved("DFKV_RDMA_SERVER_PROTOCOL", "sgengine-v1");
  } else {
    const char* protocol = std::getenv("DFKV_RDMA_SERVER_PROTOCOL");
    v2_enabled_ = !(protocol && std::strcmp(protocol, "1") == 0);
    config_dump::RecordResolved("DFKV_RDMA_SERVER_PROTOCOL",
                                v2_enabled_ ? "auto-v2" : "v1");
  }
  // --rdma-dev accepts a comma list (multi-rail): every listed device gets a
  // lifetime anchor in Start(); the FIRST entry stays the default for legacy
  // clients whose bootstrap dev frame is empty.
  for (size_t i = 0; i <= dev_name_.size();) {
    size_t c = dev_name_.find(',', i);
    if (c == std::string::npos) c = dev_name_.size();
    std::string d = dev_name_.substr(i, c - i);
    if (!d.empty()) anchor_devs_.push_back(d);
    i = c + 1;
  }
  if (anchor_devs_.empty()) anchor_devs_.push_back("");
  dev_name_ = anchor_devs_.front();
}

RdmaServer::~RdmaServer() { Stop(); }

Status RdmaServer::Start(int port) {
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) return Status::kIOError;
  int one = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(INADDR_ANY);  // bootstrap reachable on any IP net
  sa.sin_port = htons(static_cast<uint16_t>(port));
  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) != 0) {
    ::close(listen_fd_); listen_fd_ = -1; return Status::kIOError;
  }
  if (::listen(listen_fd_, 128) != 0) {
    ::close(listen_fd_); listen_fd_ = -1; return Status::kIOError;
  }
  socklen_t sl = sizeof(sa);
  ::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&sa), &sl);
  port_ = ntohs(sa.sin_port);
  // Resolve an explicit comma list as an ACTIVE-device whitelist. With no
  // override, preserve host-local device semantics by selecting only the first
  // ACTIVE HCA; multi-rail is opt-in because names need not match on peers.
  std::vector<std::string> requested_devices;
  for (const auto& device : anchor_devs_) {
    if (!device.empty()) requested_devices.push_back(device);
  }
  auto active_devices = rdma::RdmaTopology::Discover(requested_devices);
  if (auto_device_ && active_devices.size() > 1) active_devices.resize(1);
  if (active_devices.empty()) {
    DFKV_LOG_ERROR(
        requested_devices.empty()
            ? "rdma: no ACTIVE device found"
            : "rdma: none of the configured devices is ACTIVE");
    ::close(listen_fd_);
    listen_fd_ = -1;
    return Status::kIOError;
  }
  anchor_devs_.clear();
  std::string resolved_devices;
  for (const auto& device : active_devices) {
    anchor_devs_.push_back(device.name);
    if (!resolved_devices.empty()) resolved_devices += ",";
    resolved_devices += device.name;
  }
  dev_name_ = anchor_devs_.front();
  config_dump::RecordResolved("DFKV_RDMA_DEV", resolved_devices);
  // Allocate the process-wide v2 receive segment before opening device anchors.
  // Allocation failure in auto-v2 mode preserves service by retaining the v1
  // path; metrics expose that v2 has no registered segment.
  if (v2_enabled_) {
    recv_segment_bytes_ = RecvSegmentBytes();
    if (recv_segment_bytes_ == 0 ||
        !recv_segment_.Init(recv_segment_bytes_, rdma::kV2DataOffset)) {
      DFKV_LOG_WARN("rdma: unable to allocate shared receive segment (" +
                    std::to_string(recv_segment_bytes_) +
                    " bytes); continuing v1-only");
      recv_segment_bytes_ = 0;
      v2_enabled_ = false;
    }
  }
  recv_segment_registered_rails_ = 0;
  for (const auto& d : anchor_devs_) {
    auto anchor = std::make_unique<rdma::RcEndpoint>();
    if (anchor->Open(d.empty() ? nullptr : d.c_str(),
                     rdma::kV2ControlCap, 1)) {
      anchor->EnsurePoolMrs(user_regions_);
      if (v2_enabled_) {
        if (!anchor->RegisterRemoteRegion(recv_segment_.data(),
                                          recv_segment_.size())) {
          DFKV_LOG_ERROR("rdma: failed to register shared receive segment on " +
                         (d.empty() ? std::string("(auto)") : d) +
                         "; v2 connections on this rail will be refused");
        } else {
          ++recv_segment_registered_rails_;
        }
      }
      anchors_.push_back(std::move(anchor));
    }
  }
  if (anchors_.empty()) {
    DFKV_LOG_ERROR("rdma: failed to open every resolved ACTIVE device");
    ::close(listen_fd_);
    listen_fd_ = -1;
    return Status::kIOError;
  }
  if (v2_enabled_ && recv_segment_registered_rails_ == 0) {
    DFKV_LOG_WARN(
        "rdma: shared receive segment registration failed on every rail; "
        "continuing v1-only");
    v2_enabled_ = false;
  }
  if (anchor_devs_.size() > 1)
    DFKV_LOG_INFO("rdma multi-rail anchors: " +
                  std::to_string(anchors_.size()) + "/" +
                  std::to_string(anchor_devs_.size()) +
                  " devices pinned");
  running_ = true;
  accept_thread_ =
      std::thread([this] { NameThisThread("rdma-accept"); AcceptLoop(); });
  return Status::kOk;
}

void RdmaServer::Stop() {
  if (!running_.exchange(false)) return;
  if (listen_fd_ >= 0) ::shutdown(listen_fd_, SHUT_RDWR);  // wake accept()
  if (accept_thread_.joinable()) accept_thread_.join();
  if (listen_fd_ >= 0) { ::close(listen_fd_); listen_fd_ = -1; }
  // Wake every in-flight Serve thread out of WaitComp, then join them all so no
  // handler call can race the owner's destruction after Stop() returns.
  std::vector<Conn> conns;
  {
    std::lock_guard<std::mutex> lk(conn_mu_);
    for (rdma::RcEndpoint* ep : live_eps_) ep->Wake();
    conns.swap(conns_);
  }
  for (auto& c : conns) if (c.th.joinable()) c.th.join();
  anchors_.clear();  // drop the lifetime device refs (frees pool MRs last)
}

// Join and drop any Serve threads that have already finished. Called from
// AcceptLoop under conn_mu_; only threads whose `done` is set are touched, and a
// thread sets `done` only after its final conn_mu_ release, so join() never
// blocks here. This keeps conns_ bounded by the live (not lifetime) conn count.
void RdmaServer::ReapDoneLocked() {
  for (auto it = conns_.begin(); it != conns_.end();) {
    if (it->done->load(std::memory_order_acquire)) {
      if (it->th.joinable()) it->th.join();
      it = conns_.erase(it);
    } else {
      ++it;
    }
  }
}

size_t RdmaServer::live_conn_count() {
  std::lock_guard<std::mutex> lk(conn_mu_);
  return conns_.size();
}

void RdmaServer::RegisterMemory(void* base, size_t size) {
  if (base && size) user_regions_.emplace_back(base, size);  // applied per-conn in Serve
}

void RdmaServer::AcceptLoop() {
  while (running_) {
    int fd = ::accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) { if (!running_) break; continue; }
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    timeval tv{10, 0};  // bound the bootstrap handshake so a stalled client
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));  // can't hang Stop()
    std::lock_guard<std::mutex> lk(conn_mu_);
    if (!running_) { ::close(fd); break; }
    ReapDoneLocked();  // reap connections that finished since the last accept
    auto done = std::make_shared<std::atomic<bool>>(false);
    conns_.push_back({std::thread([this, fd, done] {
                        NameThisThread("rdma-serve");
                        Serve(fd);
                        done->store(true, std::memory_order_release);  // last act
                      }),
                      done});
  }
}

namespace {
size_t ServerDepth() {
  // Pipeline depth (requests in flight per connection). Default 1 keeps per-conn
  // pinned memory low; set DFKV_RDMA_DEPTH>1 to enable pipelining (helps the
  // latency-bound PUT path; GET scales via more conns).
  size_t out = 1;
  const char* e = std::getenv("DFKV_RDMA_DEPTH");
  if (e && *e) { long v = std::strtol(e, nullptr, 10); if (v >= 1 && v <= 256) out = (size_t)v; }
  config_dump::RecordResolved("DFKV_RDMA_DEPTH", std::to_string(out));
  return out;
}

int ServerIdleMs() {
  // Per-connection idle timeout. A connection with no completions for this long
  // is reclaimed: its Serve thread returns (freeing the QP, pinned buffers, and
  // the thread itself, which ReapDoneLocked then joins). Without this, a Serve
  // thread blocks in WaitComp forever after a silent client disconnect (a torn-
  // down RC peer yields no completion), so a long-running server accumulates one
  // live thread per lifetime connection. Reclaiming idle connections is safe:
  // the client re-dials a stale pooled connection via RdmaTransport's 2-attempt
  // retry. Default 10 min keeps active/recently-used pooled conns alive; set
  // DFKV_RDMA_IDLE_MS=0 to disable (block forever, the legacy behavior).
  int out = 600000;  // 10 minutes
  const char* e = std::getenv("DFKV_RDMA_IDLE_MS");
  if (e && *e) {
    long v = std::strtol(e, nullptr, 10);
    if (v <= 0) out = -1;                     // disabled => block forever
    else out = static_cast<int>(v > 86400000 ? 86400000 : v);  // clamp to 24h
  }
  config_dump::RecordResolved("DFKV_RDMA_IDLE_MS", std::to_string(out));
  return out;
}

#ifdef DFKV_WITH_URING
// io_uring async-GET ring depth. Defaults to the pipeline depth K so each in-
// flight request can have one read outstanding; can be raised independently to
// expose more disk queue depth without growing the RDMA pipeline. Clamped to
// [1, 256] and to >= K by the caller.
size_t UringDepth(size_t k) {
  size_t out = k;  // one outstanding read per in-flight request by default
  const char* e = std::getenv("DFKV_SERVER_URING_DEPTH");
  if (e && *e) {
    long v = std::strtol(e, nullptr, 10);
    if (v >= 1 && v <= 256) out = static_cast<size_t>(v);
  }
  config_dump::RecordResolved("DFKV_SERVER_URING_DEPTH", std::to_string(out));
  return out;
}
#endif  // DFKV_WITH_URING

}  // namespace

size_t RdmaServer::PipelineDepth() const { return ServerDepth(); }

bool RdmaServer::UseUringPath() const {
#ifdef DFKV_WITH_URING
  if (!range_prep_handler_ || !range_complete_handler_) return false;
  // Phase 10: default ON when built with io_uring. The batch-read path submits
  // a whole completion batch's GET disk reads at QD>1 and replies in arrival
  // order; phase-6 measured it NEUTRAL for the many-connection case (thread/
  // window parallelism already saturates the disk) and phase-10 measured +6%
  // on the single/few-connection deep-pipeline read-back the L3 hot path hits.
  // Non-negative across cases, with a sync fallback on any ring/batch failure.
  // DFKV_SERVER_URING=0 forces the legacy synchronous read loop.
  const char* e = std::getenv("DFKV_SERVER_URING");
  const bool out = !(e && std::strcmp(e, "0") == 0);
  config_dump::RecordResolved("DFKV_SERVER_URING", out ? "on" : "off");
  return out;
#else
  return false;
#endif
}

void RdmaServer::Serve(int boot_fd) {
  // Bootstrap: client first names the device it wants us to use (same rail for
  // multi-rail); fall back to our configured default if it sends an empty name.
  char devbuf[rdma::kDevNameBytes];
  if (!net::ReadAll(boot_fd, devbuf, rdma::kDevNameBytes)) {
    ::close(boot_fd);
    return;
  }
  // A new client probes support before allocating a small v2 endpoint. Legacy
  // servers try to open this deliberately nonexistent device and close quickly;
  // a v2 server answers without creating a QP.
  if (rdma::IsV2Probe(devbuf)) {
    if (v2_enabled_) {
      char reply[rdma::kV2ProbeReplyBytes];
      rdma::EncodeV2ProbeReply(reply);
      net::WriteAll(boot_fd, reply, sizeof(reply));
    }
    ::close(boot_fd);
    return;
  }

  const uint64_t declared = rdma::ParseDevFrameCaps(devbuf);
  const uint8_t requested_protocol = rdma::ParseDevFrameProtocol(devbuf);
  const bool use_v2 =
      v2_enabled_ && requested_protocol >= rdma::kDevProtoV2 &&
      declared != 0;
  const uint8_t wire_epoch =
      legacy_wire_ ? kSgEngineProtoV1
                   : (use_v2 ? kNativeProtoRdmaV2 : kNativeProtoBase);
  auto request_key = [this](const ReqFields& fields) {
    return BlockKey{fields.digest_hi, fields.digest_lo,
                    legacy_wire_ ? KeyDomain::kSgEngineV1
                                 : KeyDomain::kNative};
  };
  if (declared && declared > static_cast<uint64_t>(max_msg_)) {
    DFKV_LOG_ERROR("rdma: client declared max block " + std::to_string(declared) +
                   "B, above this server's cap " + std::to_string(max_msg_) +
                   "B; refusing the connection (sizing down would let the client "
                   "send past our receives). Raise --max-msg or lower the client's "
                   "DFKV_RDMA_MAX_BLOCK_BYTES.");
    ::close(boot_fd);
    return;
  }
  const size_t conn_max =
      declared ? std::max<size_t>(wire_limits::kIoAlign,
                                  std::min<size_t>(declared, max_msg_))
               : max_msg_;
  devbuf[rdma::kDevNameBytes - 1] = '\0';
  std::string dev = devbuf[0] ? std::string(devbuf) : dev_name_;
  if (std::find(anchor_devs_.begin(), anchor_devs_.end(), dev) ==
      anchor_devs_.end()) {
    DFKV_LOG_ERROR("rdma: client requested device outside the ACTIVE filter: " +
                   dev);
    ::close(boot_fd);
    return;
  }

  // The client writes QpInfo before waiting for ours, so read its negotiated
  // depth before allocating any per-connection buffers. A default-depth=1
  // client must lease one shared data slot, not ServerDepth() slots.
  char peer[rdma::kQpInfoBytes];
  if (!net::ReadAll(boot_fd, peer, sizeof(peer))) {
    ::close(boot_fd);
    return;
  }
  const rdma::QpInfo peer_info = rdma::ParseQpInfo(peer);
  if (use_v2 &&
      peer_info.protocol_version != rdma::kDevProtoV2) {
    ::close(boot_fd);
    return;
  }
  size_t K = ServerDepth();
  if (peer_info.depth != 0)
    K = std::min<size_t>(K, peer_info.depth);
  if (K == 0) {
    ::close(boot_fd);
    return;
  }
  const size_t slot_size = use_v2 ? rdma::V2SlotSize(conn_max) : 0;
  rdma::RecvSegment::Lease recv_lease;
  if (use_v2) {
    if (slot_size == 0 ||
        K > std::numeric_limits<size_t>::max() / slot_size) {
      ::close(boot_fd);
      return;
    }
    recv_lease = recv_segment_.Allocate(K * slot_size,
                                        rdma::kV2DataOffset);
    if (!recv_lease) {
      DFKV_LOG_ERROR(
          "rdma v2: shared receive segment exhausted; refusing connection "
          "(need=" +
          std::to_string(K * slot_size) +
          " free=" + std::to_string(recv_segment_.free_bytes()) + ")");
      ::close(boot_fd);
      return;
    }
  }

  rdma::RcEndpoint ep;
  const size_t conn_control =
      use_v2 ? rdma::kV2ControlCap : ControlCapFor(conn_max);
  const size_t direct_cap = ValueHeader::kSize + conn_max;
  if (!ep.Open(dev.empty() ? nullptr : dev.c_str(), conn_control, K,
               /*ib_port=*/1, /*direct_io_buffers=*/!use_v2, direct_cap,
               /*v2_responder=*/use_v2)) {
    ::close(boot_fd);
    return;
  }
  ibv_mr* recv_segment_mr = nullptr;
  if (use_v2) {
    recv_segment_mr = ep.RegisterRemoteRegion(recv_segment_.data(),
                                               recv_segment_.size());
    if (!recv_segment_mr) {
      DFKV_LOG_ERROR("rdma v2: receive-segment MR unavailable on device " +
                     (dev.empty() ? std::string("(auto)") : dev));
      ::close(boot_fd);
      return;
    }
  }
  DFKV_LOG_INFO(
      "rdma conn: protocol=v" + std::to_string(use_v2 ? 2 : 1) +
      " declared=" + std::to_string(declared) +
      " control=" + std::to_string(conn_control) +
      (use_v2 ? " shared-slot=" + std::to_string(slot_size)
              : " per-slot-dbuf=" + std::to_string(direct_cap)) +
      " qd=" + std::to_string(K));
  numa::PinThreadToNode(ep.numa_node());

  // QP bootstrap: the client's QpInfo was read before allocation above; send
  // the now-right-sized server endpoint and connect.
  char mine[rdma::kQpInfoBytes];
  rdma::QpInfo my = ep.Local();
  my.depth = static_cast<uint16_t>(std::min<size_t>(K, 256));
  my.protocol_version = use_v2 ? rdma::kDevProtoV2 : rdma::kDevProtoV1;
  rdma::SerializeQpInfo(my, mine);
  if (!net::WriteAll(boot_fd, mine, rdma::kQpInfoBytes) ||
      !ep.Connect(peer_info)) {
    ::close(boot_fd);
    return;
  }
  ep.EnsurePoolMrs(user_regions_);
  auto post_request_recv = [&](size_t slot) {
    if (use_v2) return ep.PostRecv(slot);
    if (!ep.dbuf(slot) || !ep.dmr(slot) ||
        ep.dbuf_cap() <= ValueHeader::kSize)
      return ep.PostRecv(slot);
    return ep.PostRecvScatter(slot, ep.dbuf(slot) + ValueHeader::kSize,
                              ep.dbuf_cap() - ValueHeader::kSize, ep.dmr(slot),
                              kReqPrefix + ValueHeader::kSize);
  };

  bool armed = true;
  for (size_t i = 0; i < K; ++i) armed = armed && post_request_recv(i);
  if (!armed) {
    ::close(boot_fd);
    return;
  }
  // Tell the client receives are posted, then publish its v2 receive-segment
  // lease. The extension is read only when the QpInfo v2 bit was negotiated.
  char ready = 1;
  bool ok = net::WriteAll(boot_fd, &ready, 1);
  if (ok && use_v2) {
    const rdma::RecvSegmentInfo info{
        reinterpret_cast<uint64_t>(recv_lease.data()),
        recv_segment_mr->rkey, slot_size};
    char encoded[rdma::kRecvSegmentInfoBytes];
    rdma::EncodeRecvSegmentInfo(info, encoded);
    ok = net::WriteAll(boot_fd, encoded, sizeof(encoded));
  }
  ::close(boot_fd);
  if (!ok) return;
  if (use_v2)
    v2_conns_.fetch_add(1, std::memory_order_relaxed);
  else
    v1_conns_.fetch_add(1, std::memory_order_relaxed);

  // Register this endpoint so Stop() can Wake() us out of WaitComp and join. The
  // running_ check under conn_mu_ closes the race with a concurrent Stop(): either
  // Stop sees us in live_eps_ (and wakes us) or we see running_==false here.
  {
    std::lock_guard<std::mutex> lk(conn_mu_);
    if (!running_) return;
    live_eps_.insert(&ep);
  }

  // Send-slot free list (a reply uses one send slot until its SEND completes).
  std::vector<size_t> free_send;
  free_send.reserve(K);
  for (size_t i = 0; i < K; ++i) free_send.push_back(i);

  auto direct_buffer = [&](size_t slot) -> char* {
    if (use_v2)
      return recv_lease.data() + slot * slot_size + rdma::kV2DataOffset;
    return ep.dbuf(slot);
  };
  auto direct_mr = [&](size_t slot) -> ibv_mr* {
    return use_v2 ? recv_segment_mr : ep.dmr(slot);
  };
  const size_t direct_buffer_cap =
      use_v2 ? slot_size - rdma::kV2DataOffset : ep.dbuf_cap();
  const size_t logical_data_cap = ValueHeader::kSize + conn_max;

  struct Request {
    ReqFields fields{};
    uint32_t recv_bytes = 0;
    size_t recv_slot = 0;  // RQ entry consumed by SEND or WRITE_WITH_IMM
    size_t data_slot = 0;  // shared-segment slot (v2) / dbuf slot (v1)
    const char* contiguous_payload = nullptr;
    RdmaGetFields get;
  };

  auto decode_request = [&](const ibv_wc& completion,
                            Request* request) -> bool {
    const size_t recv_slot = static_cast<size_t>(completion.wr_id);
    const bool normal_recv = completion.opcode == IBV_WC_RECV;
    const bool write_imm_recv =
        completion.opcode == IBV_WC_RECV_RDMA_WITH_IMM;
    const bool has_immediate =
        (completion.wc_flags & IBV_WC_WITH_IMM) != 0;
    if (recv_slot >= K ||
        (!normal_recv && !(use_v2 && write_imm_recv)) ||
        has_immediate != write_imm_recv) {
      return false;
    }
    request->recv_slot = recv_slot;
    request->data_slot = recv_slot;
    request->recv_bytes = completion.byte_len;
    if (use_v2 && write_imm_recv) {
      const size_t data_slot = static_cast<size_t>(ntohl(completion.imm_data));
      if (data_slot >= K || completion.byte_len < kReqPrefix) return false;
      const char* frame = recv_lease.data() + data_slot * slot_size +
                          rdma::kV2PutPrefixOffset;
      if (!DecodeReqVersion(
              frame, kNativeProtoRdmaV2, &request->fields,
              static_cast<uint64_t>(logical_data_cap)) ||
          request->fields.op != static_cast<uint8_t>(WireOp::kCache) ||
          !rdma::V2PutCompletionIsValid(
              write_imm_recv, has_immediate, completion.byte_len,
              request->fields.payload_len, logical_data_cap, slot_size)) {
        return false;
      }
      request->data_slot = data_slot;
      request->contiguous_payload = frame + kReqPrefix;
      v2_put_writes_.fetch_add(1, std::memory_order_relaxed);
      return true;
    }

    const char* frame = ep.rbuf(recv_slot);
    if (completion.byte_len < kReqPrefix ||
        !DecodeReqVersion(frame, wire_epoch, &request->fields,
                          static_cast<uint64_t>(ValueHeader::kSize +
                                                conn_max))) {
      return false;
    }
    if (use_v2 &&
        request->fields.op == static_cast<uint8_t>(WireOp::kRange)) {
      if (!DecodeRdmaGetReq(
              frame, completion.byte_len, &request->fields,
              &request->get,
              static_cast<uint64_t>(ValueHeader::kSize + conn_max))) {
        return false;
      }
      return request->get.targets.size() <= rdma::kV2MaxGetTargets;
    }
    if (completion.byte_len <
        kReqPrefix + request->fields.payload_len) {
      return false;
    }
    if (use_v2 && request->fields.payload_len != 0)
      request->contiguous_payload = frame + kReqPrefix;
    return true;
  };

  struct Reply {
    bool scatter = false;
    bool remote_write = false;
    bool defer_recv_rearm = false;
    size_t recv_slot = 0;
    size_t first_len = 0;
    const char* payload = nullptr;
    size_t payload_len = 0;
    ibv_mr* payload_mr = nullptr;
    std::vector<RdmaWriteTarget> targets;
    uint64_t release_token = 0;
  };

  auto copy_payload = [&](const Request& request,
                          std::string* payload) -> bool {
    payload->clear();
    const ReqFields& fields = request.fields;
    if (fields.payload_len == 0) return true;
    if (fields.payload_len >
        static_cast<uint64_t>(ValueHeader::kSize + conn_max))
      return false;
    if (request.contiguous_payload) {
      payload->assign(request.contiguous_payload,
                      static_cast<size_t>(fields.payload_len));
      return true;
    }
    if (request.recv_bytes < kReqPrefix + fields.payload_len) return false;
    payload->resize(static_cast<size_t>(fields.payload_len));
    const size_t head = static_cast<size_t>(
        std::min<uint64_t>(fields.payload_len, ValueHeader::kSize));
    if (head)
      std::memcpy(payload->data(),
                  ep.rbuf(request.recv_slot) + kReqPrefix, head);
    const size_t rest = static_cast<size_t>(fields.payload_len) - head;
    if (rest != 0) {
      if (!ep.dbuf(request.data_slot) ||
          ep.dbuf_cap() < ValueHeader::kSize + rest)
        return false;
      std::memcpy(payload->data() + head,
                  ep.dbuf(request.data_slot) + ValueHeader::kSize, rest);
    }
    return true;
  };

  auto build_reply = [&](size_t send_slot, const Request& request,
                         Reply* reply) -> bool {
    const ReqFields& fields = request.fields;
    const BlockKey key = request_key(fields);
    char* send_buffer = ep.sbuf(send_slot);
    auto encode_status = [&](Status status, uint64_t data_len) {
      EncodeRespVersion(send_buffer, wire_epoch, status, data_len);
    };
    auto invalid_reply = [&] {
      encode_status(Status::kInvalid, 0);
      reply->first_len = kRespPrefix;
      return true;
    };
    auto data_reply = [&](Status status, const char* data, size_t data_len,
                          ibv_mr* data_mr, bool source_uses_slot,
                          uint64_t release_token) -> bool {
      const size_t successful_len =
          status == Status::kOk ? data_len : 0;
      if (!use_v2) {
        encode_status(status, successful_len);
        reply->first_len = kRespPrefix;
        if (status == Status::kOk && successful_len != 0) {
          if (!data || !data_mr) return false;
          reply->scatter = true;
          reply->defer_recv_rearm = source_uses_slot;
          reply->recv_slot = request.recv_slot;
          reply->payload = data;
          reply->payload_len = successful_len;
          reply->payload_mr = data_mr;
          reply->release_token = release_token;
        } else if (release_token && ram_release_handler_) {
          ram_release_handler_(release_token);
        }
        return true;
      }

      if (status != Status::kOk) {
        encode_status(status, 0);
        reply->first_len = kRespPrefix;
        if (release_token && ram_release_handler_)
          ram_release_handler_(release_token);
        return true;
      }
      const size_t header_len = request.get.header_len;
      if (header_len > successful_len ||
          kRespPrefix + header_len > ep.cap() ||
          successful_len - header_len > request.get.Capacity() ||
          (successful_len != 0 && !data)) {
        if (release_token && ram_release_handler_)
          ram_release_handler_(release_token);
        return invalid_reply();
      }
      encode_status(Status::kOk, successful_len);
      if (header_len)
        std::memcpy(send_buffer + kRespPrefix, data, header_len);
      reply->first_len = kRespPrefix + header_len;
      const size_t remote_len = successful_len - header_len;
      if (remote_len != 0) {
        if (!data_mr) return false;
        reply->remote_write = true;
        reply->defer_recv_rearm = source_uses_slot;
        reply->recv_slot = request.recv_slot;
        reply->payload = data + header_len;
        reply->payload_len = remote_len;
        reply->payload_mr = data_mr;
        reply->targets = request.get.targets;
        reply->release_token = release_token;
      } else if (release_token && ram_release_handler_) {
        ram_release_handler_(release_token);
      }
      return true;
    };

    if (fields.op == static_cast<uint8_t>(WireOp::kRange) &&
        range_handler_) {
      if (!direct_buffer(request.data_slot) ||
          !direct_mr(request.data_slot) ||
          fields.length >
              static_cast<uint64_t>(ValueHeader::kSize + conn_max))
        return invalid_reply();

      if (ram_range_handler_) {
        const char* ram_data = nullptr;
        size_t ram_len = 0;
        uint64_t token = 0;
        if (ram_range_handler_(key, fields.offset, fields.length, &ram_data,
                               &ram_len, &token)) {
          ibv_mr* ram_mr =
              ram_len ? ep.RegisterUser(const_cast<char*>(ram_data), ram_len)
                      : nullptr;
          if (ram_len == 0 || ram_mr)
            return data_reply(Status::kOk, ram_data, ram_len, ram_mr,
                              /*source_uses_slot=*/false, token);
          if (ram_release_handler_) ram_release_handler_(token);
        }
      }

      const char* output = nullptr;
      size_t output_len = 0;
      const Status status = range_handler_(
          key, fields.offset, fields.length, direct_buffer(request.data_slot),
          direct_buffer_cap, &output, &output_len);
      return data_reply(status, output, output_len,
                        direct_mr(request.data_slot),
                        /*source_uses_slot=*/true, 0);
    }

    if (fields.op == static_cast<uint8_t>(WireOp::kCache) &&
        cache_direct_handler_) {
      if (!direct_buffer(request.data_slot) ||
          !direct_mr(request.data_slot) ||
          fields.payload_len < ValueHeader::kSize ||
          fields.payload_len >
              static_cast<uint64_t>(logical_data_cap))
        return invalid_reply();

      char* cache_data = direct_buffer(request.data_slot);
      if (request.contiguous_payload != cache_data) {
        if (use_v2) {
          if (!request.contiguous_payload) return invalid_reply();
          std::memcpy(cache_data, request.contiguous_payload,
                      static_cast<size_t>(fields.payload_len));
        } else {
          if (request.recv_bytes < kReqPrefix + fields.payload_len)
            return invalid_reply();
          std::memcpy(cache_data,
                      ep.rbuf(request.recv_slot) + kReqPrefix,
                      ValueHeader::kSize);
        }
      }
      const Status status = cache_direct_handler_(
          key, cache_data, static_cast<size_t>(fields.payload_len),
          direct_buffer_cap);
      encode_status(status, 0);
      reply->first_len = kRespPrefix;
      return true;
    }

    std::string payload_storage;
    const char* payload = nullptr;
    if (fields.payload_len != 0) {
      if (request.contiguous_payload) {
        payload = request.contiguous_payload;
      } else {
        if (!copy_payload(request, &payload_storage))
          return invalid_reply();
        payload = payload_storage.data();
      }
    }
    std::string data;
    const Status status =
        handler_(fields.op, key, fields.offset, fields.length, payload,
                 fields.payload_len, &data);
    if (use_v2 &&
        fields.op == static_cast<uint8_t>(WireOp::kRange)) {
      if (data.size() > direct_buffer_cap) return invalid_reply();
      if (!data.empty())
        std::memcpy(direct_buffer(request.data_slot), data.data(),
                    data.size());
      return data_reply(status, direct_buffer(request.data_slot), data.size(),
                        direct_mr(request.data_slot),
                        /*source_uses_slot=*/true, 0);
    }
    if (kRespPrefix + data.size() > ep.cap()) return false;
    encode_status(status, data.size());
    if (!data.empty())
      std::memcpy(send_buffer + kRespPrefix, data.data(), data.size());
    reply->first_len = kRespPrefix + data.size();
    return true;
  };

  auto post_reply = [&](size_t send_slot, const Reply& reply) -> bool {
    if (!reply.remote_write)
      return reply.scatter
                 ? ep.PostSendScatter(send_slot, reply.first_len,
                                      reply.payload, reply.payload_len,
                                      reply.payload_mr)
                 : ep.PostSend(send_slot, reply.first_len);

    size_t written = 0;
    for (const auto& target : reply.targets) {
      if (written == reply.payload_len) break;
      const size_t bytes =
          std::min<size_t>(target.length, reply.payload_len - written);
      if (bytes == 0) continue;
      if (!ep.PostWrite(send_slot, reply.payload + written, bytes,
                        reply.payload_mr, target.addr, target.rkey))
        return false;
      written += bytes;
    }
    if (written != reply.payload_len) return false;
    v2_get_writes_.fetch_add(1, std::memory_order_relaxed);
    return ep.PostSend(send_slot, reply.first_len);
  };

  // Single-threaded serve loop: reap completions and process each RECV inline, in
  // arrival (= request) order, replying on a free send slot. Replies MUST go out
  // in request order: the pipelined client binds each reply's destination buffer
  // at recv-post time (zero-copy scatter), so an out-of-order reply would land in
  // the wrong buffer. Depth K still gives K-in-flight pipelining; we just don't
  // reorder. (An earlier parallel GET worker pool was removed for this reason —
  // it broke zero-copy correctness for marginal gain; GET scales via connections.)
  std::vector<ibv_wc> wcs(K);
  constexpr size_t kNoSlot = static_cast<size_t>(-1);
  std::vector<size_t> rearm_on_send(K, kNoSlot);
  // Parallel to rearm_on_send: the RAM-hit pin token to release when this send
  // slot's IBV_WC_SEND fires (the arena bytes were read by the NIC in place).
  // 0 = none (B5-3). All-zero unless a ram_range_handler_ hit went out.
  std::vector<uint64_t> release_on_send(K, 0);
  auto release_completed_send = [&](size_t sid) {
    if (sid < release_on_send.size() && release_on_send[sid] != 0) {
      if (ram_release_handler_) ram_release_handler_(release_on_send[sid]);
      release_on_send[sid] = 0;
    }
  };
  bool fail = false;
  const int idle_ms = ServerIdleMs();
  active_conns_.fetch_add(1, std::memory_order_relaxed);

#ifdef DFKV_WITH_URING
  // -------------------------------------------------------------------------
  // io_uring async-GET serve loop (env-gated; correctness-preserving).
  //
  // Batch-and-wait model (Mooncake's uring_file batch_read adapted to dfkv's
  // per-WaitComp completion batch). For each completion batch returned by
  // WaitComp: handle SEND completions and non-kRange RECVs inline exactly as the
  // sync loop; for kRange GETs, prep an ordered descriptor list, submit all disk
  // reads at once, then emit v1 scatter SENDs or v2 RDMA WRITEs in arrival order.
  // The slot backing each read stays leased until the signaled status SEND
  // completes, so neither a new PUT nor another GET can overwrite bytes still
  // being consumed by the NIC. Anything that cannot go async falls back to
  // build_reply for that request without reordering.
  if (UseUringPath()) {
    const size_t uring_depth = std::max(UringDepth(K), K);
    UringReader ring(static_cast<unsigned>(uring_depth));
    if (!ring.ok()) {
      // Ring init failed: fall through to the sync loop below (correctness-first)
      // -- but say so, and count it: an operator who set DFKV_SERVER_URING=1
      // must be able to tell "active" from "silently degraded".
      uring_init_fallbacks_.fetch_add(1, std::memory_order_relaxed);
      DFKV_LOG_WARN("io_uring ring init failed (depth=" +
                    std::to_string(uring_depth) +
                    "); this connection serves on the SYNC path");
      goto sync_serve_loop;
    }
    {
      // One queued reply for this completion batch, kept in strict arrival order.
      // The client correlates replies to requests purely by SEND order on the
      // wire (RC in-order delivery; recv slot j <-> j-th reply), so EVERY reply —
      // sync-built or read-completed — MUST be SENT in arrival order. We therefore
      // queue all replies first, run the io_uring read batch, then emit the whole
      // queue in order. `read_idx>=0` means this entry's payload comes from
      // descs[read_idx] (a deferred kRange hit); otherwise it is a fully-built
      // synchronous Reply.
      struct Queued {
        size_t send_slot = 0;
        size_t recv_slot = 0;
        size_t data_slot = 0;
        RdmaGetFields get;       // v2 response header/remote destinations
        int read_idx = -1;   // >=0: index into descs (async read); -1: sync reply
        int fd = -1;         // owned (async only); closed after the batch read
        uint64_t prep_token = 0;  // slab slot hold; released where fd is closed
        uint64_t flight = 0;      // coalescer registration; completed or aborted
        size_t head = 0;
        size_t payload_len = 0;
        // Steady-clock seconds at prep (read submit), 0 = unsampled. Only the
        // 1/64-sampled async reads are stamped; completion observes get_lat_ so
        // the default uring read path is no longer latency-blind.
        double submit_sec = 0.0;
        Reply reply;         // used when read_idx < 0
      };
      std::vector<UringReader::ReadDesc> descs;
      std::vector<Queued> queue;
      descs.reserve(K);
      queue.reserve(K);

      while (running_ && !fail) {
        int g = ep.WaitComp(wcs.data(), static_cast<int>(K), idle_ms);
        if (g == 0) { idle_reclaims_.fetch_add(1, std::memory_order_relaxed); break; }
        if (g < 0) break;  // error / Stop()'s Wake()
        descs.clear();
        queue.clear();
        for (int w = 0; w < g && !fail; ++w) {
          const ibv_wc& wc = wcs[w];
          if (wc.status != IBV_WC_SUCCESS) {
            completion_errors_.fetch_add(1, std::memory_order_relaxed);
            fail = true; break;
          }
          if (wc.opcode == IBV_WC_SEND) {
            size_t sid = static_cast<size_t>(wc.wr_id);
            if (sid < rearm_on_send.size() && rearm_on_send[sid] != kNoSlot) {
              if (!post_request_recv(rearm_on_send[sid])) { fail = true; break; }
              rearm_on_send[sid] = kNoSlot;
            }
            release_completed_send(sid);  // release any RAM-hit pin (B5-3)
            free_send.push_back(sid);
            continue;
          }
          Request request;
          if (!decode_request(wc, &request)) { fail = true; break; }
          completions_.fetch_add(1, std::memory_order_relaxed);
          const size_t r = request.recv_slot;
          const ReqFields& rq = request.fields;
          if (free_send.empty()) { fail = true; break; }
          size_t s = free_send.back(); free_send.pop_back();

          Queued qd;
          qd.send_slot = s;
          qd.recv_slot = r;
          qd.data_slot = request.data_slot;
          qd.get = request.get;

          // Defer a kRange GET hit's disk read to the io_uring batch below; reply
          // after the whole batch completes (the emit pass preserves arrival order).
          bool deferred = false;
          if (rq.op == static_cast<uint8_t>(WireOp::kRange) &&
              direct_buffer(request.data_slot) &&
              direct_mr(request.data_slot) &&
              rq.length <= static_cast<uint64_t>(ValueHeader::kSize + conn_max)) {
            RangePrepResult pr;
            Status pst = range_prep_handler_(
                request_key(rq), rq.offset, rq.length, direct_buffer_cap, &pr);
            if (pst == Status::kOk && pr.fd >= 0 && pr.payload_len != 0 &&
                pr.aligned_len <= direct_buffer_cap &&
                pr.aligned_len <= std::numeric_limits<unsigned>::max()) {
              UringReader::ReadDesc d;
              d.fd = pr.fd;
              d.buf = direct_buffer(request.data_slot);
              d.len = static_cast<unsigned>(pr.aligned_len);
              d.off = pr.aligned_off;
              qd.read_idx = static_cast<int>(descs.size());
              descs.push_back(d);
              qd.fd = pr.fd;
              qd.prep_token = pr.release_token;
              qd.flight = pr.flight;
              qd.head = pr.head;
              qd.payload_len = pr.payload_len;
              qd.submit_sec = NowSteadySec();  // read submit -> completion latency
              deferred = true;
            } else if (pst == Status::kOk && pr.fd >= 0) {
              ::close(pr.fd);  // zero-len / oversize: handled by sync build below
              if (pr.release_token && range_release_handler_)
                range_release_handler_(pr.release_token);
              if (pr.flight && range_flight_abort_handler_)
                range_flight_abort_handler_(pr.flight);
            }
          }

          if (!deferred) {
            // Synchronous build for this request (non-range, or range miss / zero /
            // oversize / prep miss). Consumes rbuf/dbuf[r]. Queued, not sent yet.
            if (!build_reply(s, request, &qd.reply)) {
              fail = true;
              break;
            }
          }
          queue.push_back(qd);
        }
        if (fail) break;

        // Submit + wait for ALL deferred reads in this batch (QD>1 concurrency).
        // If the batch infrastructure fails, fall back to a synchronous pread per
        // deferred request in the emit pass below (correctness-first).
        bool batch_ok = true;
        if (!descs.empty()) {
          uring_reads_.fetch_add(descs.size(), std::memory_order_relaxed);
          batch_ok = ring.BatchRead(descs.data(), static_cast<int>(descs.size()));
          // A failed batch may have left async reads in flight against leased
          // v1 dbufs or v2 shared-segment slots. Drain before the sync fallback
          // or receive rearm can reuse those buffers.
          // writes and put mixed-generation bytes on the wire — undetectable by
          // the client (ValueHeader carries no CRC; RDMA ICRC only covers the
          // network). If the drain itself times out the buffers still belong to
          // the kernel: the only safe move is to drop the connection while the
          // endpoint (and its registered buffers) is still alive; the client
          // re-dials. Once poisoned, later BatchRead calls return false without
          // submitting, so the connection continues on the sync fallback.
          if (!batch_ok && !ring.Drain()) {
            fail = true;
            break;  // un-emitted queue entries' fds are closed by the teardown below
          }
        }

        // Emit every reply in STRICT arrival order (every read is now complete,
        // so an async reply never trails a later request's reply).
        for (size_t i = 0; i < queue.size() && !fail; ++i) {
          Queued& qd = queue[i];
          if (qd.read_idx < 0) {
            // Sync-built reply: rearm recv (or defer to SEND), then SEND in order.
            Reply& reply = qd.reply;
            if (reply.defer_recv_rearm) {
              rearm_on_send[qd.send_slot] = reply.recv_slot;
            } else if (!post_request_recv(qd.recv_slot)) {
              fail = true; break;
            }
            release_on_send[qd.send_slot] = reply.release_token;  // RAM-hit pin (B5-3)
            if (!post_reply(qd.send_slot, reply)) {
              fail = true;
              break;
            }
            continue;
          }
          // Deferred async read: validate result (or sync fallback), then SEND.
          UringReader::ReadDesc& d = descs[qd.read_idx];
          bool ok;
          if (batch_ok) {
            long res = d.result;
            ok = res >= 0 && static_cast<size_t>(res) >= qd.head + qd.payload_len;
          } else {
            ssize_t got = ::pread(qd.fd, d.buf, d.len, static_cast<off_t>(d.off));
            ok = got >= 0 && static_cast<size_t>(got) >= qd.head + qd.payload_len;
          }
          if (qd.fd >= 0) { ::close(qd.fd); qd.fd = -1; }
          if (qd.prep_token) {  // read done (async or sync fallback): drop the slot hold
            if (range_release_handler_) range_release_handler_(qd.prep_token);
            qd.prep_token = 0;
          }
          // Runs BEFORE the reply send on purpose: coalesced waiters and RAM
          // promotion copy from the completed read buffer.
          const char* out_data =
              static_cast<const char*>(d.buf) + qd.head;
          if (range_complete_handler_) {
            range_complete_handler_(ok, ok ? qd.payload_len : 0,
                                    NowSteadySec() - qd.submit_sec, qd.flight,
                                    ok ? out_data : nullptr);
            qd.flight = 0;  // completed: the teardown sweep must not abort it
          }
          char* sb = ep.sbuf(qd.send_slot);
          if (ok && use_v2) {
            const size_t header_len = qd.get.header_len;
            const size_t remote_len =
                header_len <= qd.payload_len ? qd.payload_len - header_len : 0;
            if (header_len > qd.payload_len ||
                kRespPrefix + header_len > ep.cap() ||
                remote_len > qd.get.Capacity()) {
              EncodeRespVersion(sb, kNativeProtoRdmaV2, Status::kInvalid, 0);
              if (!post_request_recv(qd.recv_slot) ||
                  !ep.PostSend(qd.send_slot, kRespPrefix)) {
                fail = true;
                break;
              }
              continue;
            }
            EncodeRespVersion(sb, kNativeProtoRdmaV2, Status::kOk,
                              qd.payload_len);
            if (header_len)
              std::memcpy(sb + kRespPrefix, out_data, header_len);
            Reply reply;
            reply.first_len = kRespPrefix + header_len;
            if (remote_len != 0) {
              reply.remote_write = true;
              reply.payload = out_data + header_len;
              reply.payload_len = remote_len;
              reply.payload_mr = direct_mr(qd.data_slot);
              reply.targets = qd.get.targets;
              reply.defer_recv_rearm = true;
              reply.recv_slot = qd.recv_slot;
              rearm_on_send[qd.send_slot] = qd.recv_slot;
            } else if (!post_request_recv(qd.recv_slot)) {
              fail = true;
              break;
            }
            if (!post_reply(qd.send_slot, reply)) {
              fail = true;
              break;
            }
          } else if (ok) {
            EncodeRespVersion(sb, wire_epoch, Status::kOk, qd.payload_len);
            // Defer recv rearm until this scatter SEND completes (read target reuse).
            rearm_on_send[qd.send_slot] = qd.recv_slot;
            if (!ep.PostSendScatter(qd.send_slot, kRespPrefix, out_data,
                                    qd.payload_len,
                                    direct_mr(qd.data_slot))) {
              fail = true;
              break;
            }
          } else {
            EncodeRespVersion(sb, wire_epoch, Status::kIOError, 0);
            if (!post_request_recv(qd.recv_slot) ||
                !ep.PostSend(qd.send_slot, kRespPrefix)) {
              fail = true;
              break;
            }
          }
        }
      }

      // Connection ending: close any fds still owned by un-emitted queue
      // entries, and abort their flights so coalesced waiters on other
      // connections stop waiting NOW instead of eating the full timeout.
      for (auto& qd : queue) {
        if (qd.fd >= 0) ::close(qd.fd);
        if (qd.prep_token && range_release_handler_) range_release_handler_(qd.prep_token);
        if (qd.flight && range_flight_abort_handler_) range_flight_abort_handler_(qd.flight);
      }
    }
    // Release RAM-hit pins for sends that never completed (conn tore down) so the
    // arena slots don't stay pinned forever (B5-3).
    for (size_t i = 0; i < release_on_send.size(); ++i) release_completed_send(i);
    active_conns_.fetch_sub(1, std::memory_order_relaxed);
    { std::lock_guard<std::mutex> lk(conn_mu_); live_eps_.erase(&ep); }
    return;
  }
sync_serve_loop:;
#endif  // DFKV_WITH_URING

  while (running_ && !fail) {
    int g = ep.WaitComp(wcs.data(), static_cast<int>(K), idle_ms);
    if (g == 0) { idle_reclaims_.fetch_add(1, std::memory_order_relaxed); break; }  // idle -> reclaim
    if (g < 0) break;  // error / Stop()'s Wake()
    for (int w = 0; w < g && !fail; ++w) {
      const ibv_wc& wc = wcs[w];
      if (wc.status != IBV_WC_SUCCESS) {
        completion_errors_.fetch_add(1, std::memory_order_relaxed);
        fail = true; break;
      }
      if (wc.opcode == IBV_WC_SEND) {
        size_t sid = static_cast<size_t>(wc.wr_id);
        if (sid < rearm_on_send.size() && rearm_on_send[sid] != kNoSlot) {
          if (!post_request_recv(rearm_on_send[sid])) { fail = true; break; }
          rearm_on_send[sid] = kNoSlot;
        }
        release_completed_send(sid);  // release any RAM-hit pin (B5-3)
        free_send.push_back(sid);
        continue;
      }
      Request request;
      if (!decode_request(wc, &request)) { fail = true; break; }
      completions_.fetch_add(1, std::memory_order_relaxed);
      const size_t r = request.recv_slot;
      if (free_send.empty()) { fail = true; break; }
      size_t s = free_send.back(); free_send.pop_back();
      Reply reply;
      bool built = build_reply(s, request, &reply);
      if (!built) { fail = true; break; }
      if (reply.defer_recv_rearm) {
        rearm_on_send[s] = reply.recv_slot;
      } else if (!post_request_recv(r)) {
        fail = true; break;  // re-arm (request consumed)
      }
      release_on_send[s] = reply.release_token;  // RAM-hit pin (B5-3)
      bool sent = post_reply(s, reply);
      if (!sent) { fail = true; break; }
    }
  }
  // Release RAM-hit pins for sends that never completed (conn tore down, B5-3).
  for (size_t i = 0; i < release_on_send.size(); ++i) release_completed_send(i);
  active_conns_.fetch_sub(1, std::memory_order_relaxed);
  { std::lock_guard<std::mutex> lk(conn_mu_); live_eps_.erase(&ep); }
  // ep dtor tears down the QP; the peer observes the drop as an error completion.
}

std::string RdmaServer::MetricsText() const {
  auto m = [](std::string& s, const char* name, const char* type, const char* help,
              uint64_t v) {
    s += "# HELP "; s += name; s += " "; s += help; s += "\n";
    s += "# TYPE "; s += name; s += " "; s += type; s += "\n";
    s += name; s += " "; s += std::to_string(v); s += "\n";
  };
  std::string s;
  m(s, "dfkv_rdma_completions_total", "counter", "RDMA request completions served",
    Completions());
  m(s, "dfkv_rdma_completion_errors_total", "counter", "RDMA error completions",
    CompletionErrors());
  m(s, "dfkv_rdma_active_conns", "gauge",
    "RDMA connections currently serving", ActiveConns());
  m(s, "dfkv_rdma_v1_conns_opened_total", "counter",
    "RDMA v1 connections opened", V1Conns());
  m(s, "dfkv_rdma_v2_conns_opened_total", "counter",
    "RDMA v2 connections opened", V2Conns());
  m(s, "dfkv_rdma_v2_put_writes_total", "counter",
    "PUT requests received by RDMA WRITE_WITH_IMM", V2PutWrites());
  m(s, "dfkv_rdma_v2_get_writes_total", "counter",
    "GET payloads sent by RDMA WRITE", V2GetWrites());
  m(s, "dfkv_rdma_recv_segment_bytes", "gauge",
    "Process-wide registered receive-segment bytes", recv_segment_.size());
  m(s, "dfkv_rdma_recv_segment_free_bytes", "gauge",
    "Unleased bytes remaining in the process-wide receive segment",
    recv_segment_.free_bytes());
  m(s, "dfkv_rdma_recv_segment_registered_rails", "gauge",
    "RDMA rails with the process-wide receive segment registered",
    recv_segment_registered_rails_);
  m(s, "dfkv_rdma_v2_ready", "gauge",
    "Whether RDMA v2 has a registered shared receive segment",
    v2_enabled_ && recv_segment_registered_rails_ > 0 ? 1 : 0);
  m(s, "dfkv_rdma_idle_reclaims_total", "counter", "RDMA connections reclaimed on idle timeout",
    IdleReclaims());
  m(s, "dfkv_uring_reads_total", "counter",
    "GET disk reads submitted through the io_uring path (>0 = path active)",
    UringReads());
  m(s, "dfkv_uring_init_fallbacks_total", "counter",
    "Connections that wanted io_uring but fell back to the sync path (ring init failed)",
    UringInitFallbacks());
  return s;
}

}  // namespace dfkv
