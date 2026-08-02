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
#include "transport/transport.h"    // wire framing and transport structures
#include "cache/uring_reader.h" // io_uring async-GET path (DFKV_WITH_URING only)

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
                       const std::string& dev_name)
    : handler_(std::move(handler)),
      max_msg_(ResolveMaxPayload(max_msg)),
      dev_name_(dev_name) {
  if (dev_name_.empty()) {
    const char* e = std::getenv("DFKV_RDMA_DEV");
    if (e && *e) dev_name_ = e;
  }
  auto_device_ = dev_name_.empty();
  config_dump::RecordResolved("DFKV_RDMA_DEV",
                              dev_name_.empty() ? "(auto)" : dev_name_);
  // --rdma-dev accepts a comma list; every selected device gets a lifetime
  // anchor in Start().
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
  // Allocate the mandatory process-wide receive segment before opening anchors.
  recv_segment_bytes_ = RecvSegmentBytes();
  const size_t min_slot_bytes = rdma::V2SlotSize(max_msg_);
  if (!rdma::V2RecvSegmentFitsOneSlot(recv_segment_bytes_, max_msg_)) {
    DFKV_LOG_ERROR(
        "rdma: configured v2 receive segment (" +
        std::to_string(recv_segment_bytes_) +
        " bytes) cannot fit one complete slot (" +
        std::to_string(min_slot_bytes) + " bytes) at server max block " +
        std::to_string(max_msg_) +
        " bytes; raise DFKV_RDMA_RECV_SEGMENT_SIZE or lower --max-msg");
    recv_segment_bytes_ = 0;
    ::close(listen_fd_);
    listen_fd_ = -1;
    return Status::kIOError;
  }
  if (!recv_segment_.Init(recv_segment_bytes_, rdma::kV2DataOffset)) {
    DFKV_LOG_ERROR("rdma: unable to allocate required v2 receive segment (" +
                   std::to_string(recv_segment_bytes_) + " bytes)");
    recv_segment_bytes_ = 0;
    ::close(listen_fd_);
    listen_fd_ = -1;
    return Status::kIOError;
  }
  recv_segment_registered_rails_ = 0;
  for (const auto& d : anchor_devs_) {
    auto anchor = std::make_unique<rdma::RcEndpoint>();
    if (!anchor->Open(d.empty() ? nullptr : d.c_str(),
                      rdma::kV2ControlCap, 1)) {
      DFKV_LOG_ERROR("rdma: failed to open required v2 device " +
                     (d.empty() ? std::string("(auto)") : d));
      continue;
    }
    if (!anchor->EnsurePoolMrs(user_regions_)) {
      DFKV_LOG_ERROR("rdma: failed to register explicit user region on " +
                     (d.empty() ? std::string("(auto)") : d));
      continue;
    }
    if (!anchor->RegisterRemoteRegion(recv_segment_.data(),
                                      recv_segment_.size())) {
      DFKV_LOG_ERROR("rdma: failed to register required v2 receive segment on " +
                     (d.empty() ? std::string("(auto)") : d));
      continue;
    }
    ++recv_segment_registered_rails_;
    anchors_.push_back(std::move(anchor));
  }
  if (anchors_.size() != anchor_devs_.size() ||
      recv_segment_registered_rails_ != anchor_devs_.size()) {
    DFKV_LOG_ERROR(
        "rdma: v2 receive segment was not established on every selected rail");
    anchors_.clear();
    ::close(listen_fd_);
    listen_fd_ = -1;
    return Status::kIOError;
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
  if (!base || size == 0) return;
  auto existing =
      std::find_if(user_regions_.begin(), user_regions_.end(),
                   [base](const auto& region) { return region.first == base; });
  if (existing == user_regions_.end()) {
    user_regions_.emplace_back(base, size);
  } else if (size > existing->second) {
    existing->second = size;
  }
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
  // DFKV_RDMA_IDLE_MS=0 disables the reaper and waits indefinitely.
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
  if (!prepare_read_handler_) return false;
  // Phase 10: default ON when built with io_uring. The batch-read path submits
  // a whole completion batch's GET disk reads at QD>1 and replies in arrival
  // order; phase-6 measured it NEUTRAL for the many-connection case (thread/
  // window parallelism already saturates the disk) and phase-10 measured +6%
  // on the single/few-connection deep-pipeline read-back the L3 hot path hits.
  // Non-negative across cases, with a sync fallback on any ring/batch failure.
  // DFKV_SERVER_URING=0 forces the synchronous read loop.
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
  // Capability probes are answered without creating a QP.
  if (rdma::IsV2Probe(devbuf)) {
    char reply[rdma::kV2ProbeReplyBytes];
    rdma::EncodeV2ProbeReply(reply);
    net::WriteAll(boot_fd, reply, sizeof(reply));
    ::close(boot_fd);
    return;
  }

  const uint64_t declared = rdma::ParseDevFrameCaps(devbuf);
  if (rdma::ParseDevFrameProtocol(devbuf) != rdma::kDevProtoV2 ||
      declared == 0) {
    DFKV_LOG_ERROR("rdma: rejecting peer without required v2 negotiation");
    ::close(boot_fd);
    return;
  }
  constexpr uint8_t wire_epoch = kNativeProtoRdmaV2;
  constexpr size_t response_prefix = kRespPrefix;
  if (declared > static_cast<uint64_t>(max_msg_)) {
    DFKV_LOG_ERROR("rdma: client declared max block " + std::to_string(declared) +
                   "B, above this server's cap " + std::to_string(max_msg_) +
                   "B; refusing the connection. Raise --max-msg or lower "
                   "DFKV_RDMA_MAX_BLOCK_BYTES.");
    ::close(boot_fd);
    return;
  }
  const size_t conn_max =
      std::max<size_t>(wire_limits::kIoAlign,
                       std::min<size_t>(declared, max_msg_));
  devbuf[rdma::kDevNameBytes - 1] = '\0';
  std::string dev = devbuf[0] ? std::string(devbuf) : dev_name_;
  if (std::find(anchor_devs_.begin(), anchor_devs_.end(), dev) ==
      anchor_devs_.end()) {
    DFKV_LOG_ERROR("rdma: client requested device outside the ACTIVE filter: " +
                   dev);
    ::close(boot_fd);
    return;
  }

  // The client sends QpInfo first. Read and validate its mandatory v2 depth
  // before allocating per-connection control slots or leasing shared receive
  // space; a client advertising depth 1 must consume one slot, not ServerDepth.
  char peer[rdma::kQpInfoBytes];
  if (!net::ReadAll(boot_fd, peer, sizeof(peer))) {
    ::close(boot_fd);
    return;
  }
  const rdma::QpInfo peer_info = rdma::ParseQpInfo(peer);
  if (peer_info.protocol_version != rdma::kDevProtoV2 ||
      peer_info.depth == 0) {
    DFKV_LOG_ERROR("rdma: rejecting incompatible v2 QP negotiation");
    ::close(boot_fd);
    return;
  }
  const size_t K = std::min<size_t>(ServerDepth(), peer_info.depth);
  const size_t slot_size = rdma::V2SlotSize(conn_max);
  if (K == 0 || slot_size == 0 ||
      K > std::numeric_limits<size_t>::max() / slot_size) {
    ::close(boot_fd);
    return;
  }
  rdma::RecvSegment::Lease recv_lease =
      recv_segment_.Allocate(K * slot_size, rdma::kV2DataOffset);
  if (!recv_lease) {
    DFKV_LOG_ERROR(
        "rdma v2: shared receive segment exhausted; refusing connection "
        "(need=" +
        std::to_string(K * slot_size) +
        " free=" + std::to_string(recv_segment_.free_bytes()) + ")");
    ::close(boot_fd);
    return;
  }

  rdma::RcEndpoint ep;
  constexpr size_t conn_control = rdma::kV2ControlCap;
  if (!ep.Open(dev.empty() ? nullptr : dev.c_str(), conn_control, K,
               /*ib_port=*/1, /*direct_io_buffers=*/false, conn_max,
               /*v2_responder=*/true)) {
    ::close(boot_fd);
    return;
  }
  ibv_mr* recv_segment_mr =
      ep.RegisterRemoteRegion(recv_segment_.data(), recv_segment_.size());
  if (!recv_segment_mr) {
    DFKV_LOG_ERROR("rdma v2: receive-segment MR unavailable on device " +
                   (dev.empty() ? std::string("(auto)") : dev));
    ::close(boot_fd);
    return;
  }
  DFKV_LOG_INFO("rdma conn: protocol=v2 declared=" +
                std::to_string(declared) +
                " control=" + std::to_string(conn_control) +
                " shared-slot=" + std::to_string(slot_size) +
                " qd=" + std::to_string(K));
  numa::PinThreadToNode(ep.numa_node());

  // QP bootstrap: the peer geometry was consumed before allocation above.
  // Advertise this endpoint's clamped depth, then connect the correctly sized
  // QP; neither side may infer legacy defaults.
  char mine[rdma::kQpInfoBytes];
  rdma::QpInfo my = ep.Local();
  my.depth = static_cast<uint16_t>(std::min<size_t>(K, 256));
  my.protocol_version = rdma::kDevProtoV2;
  rdma::SerializeQpInfo(my, mine);
  if (!net::WriteAll(boot_fd, mine, rdma::kQpInfoBytes) ||
      !ep.Connect(peer_info)) {
    ::close(boot_fd);
    return;
  }
  if (!ep.EnsurePoolMrs(user_regions_)) {
    DFKV_LOG_ERROR("rdma: connection could not attach explicit user MRs");
    ::close(boot_fd);
    return;
  }
  auto post_request_recv = [&](size_t slot) { return ep.PostRecv(slot); };

  bool armed = true;
  for (size_t i = 0; i < K; ++i) armed = armed && post_request_recv(i);
  if (!armed) {
    ::close(boot_fd);
    return;
  }
  // Receives must be posted before readiness becomes visible. Publish the
  // leased receive-segment address, rkey and slot geometry only after the QP is
  // armed, so the client cannot issue a one-sided write into an unready slot.
  char ready = 1;
  const rdma::RecvSegmentInfo info{
      reinterpret_cast<uint64_t>(recv_lease.data()),
      recv_segment_mr->rkey, slot_size};
  char encoded[rdma::kRecvSegmentInfoBytes];
  rdma::EncodeRecvSegmentInfo(info, encoded);
  const bool ok = net::WriteAll(boot_fd, &ready, 1) &&
                  net::WriteAll(boot_fd, encoded, sizeof(encoded));
  ::close(boot_fd);
  if (!ok) return;
  v2_conns_.fetch_add(1, std::memory_order_relaxed);

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
    return recv_lease.data() + slot * slot_size + rdma::kV2DataOffset;
  };
  auto direct_mr = [&](size_t) -> ibv_mr* { return recv_segment_mr; };
  const size_t direct_buffer_cap = slot_size - rdma::kV2DataOffset;
  const size_t logical_data_cap = conn_max;

  struct Request {
    ReqFields fields{};
    uint32_t recv_bytes = 0;
    size_t recv_slot = 0;  // RQ entry consumed by SEND or WRITE_WITH_IMM
    size_t data_slot = 0;  // shared receive-segment slot
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
    if (recv_slot >= K || (!normal_recv && !write_imm_recv) ||
        has_immediate != write_imm_recv) {
      return false;
    }
    request->recv_slot = recv_slot;
    request->data_slot = recv_slot;
    request->recv_bytes = completion.byte_len;
    if (write_imm_recv) {
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
                          static_cast<uint64_t>(conn_max))) {
      return false;
    }
    if (request->fields.op == static_cast<uint8_t>(WireOp::kRange)) {
      if (!DecodeRdmaGetReq(frame, completion.byte_len, &request->fields,
                            &request->get,
                            static_cast<uint64_t>(conn_max))) {
        return false;
      }
      return request->get.targets.size() <= rdma::kV2MaxGetTargets;
    }
    if (completion.byte_len <
        kReqPrefix + request->fields.payload_len) {
      return false;
    }
    if (request->fields.payload_len != 0)
      request->contiguous_payload = frame + kReqPrefix;
    return true;
  };

  struct Reply {
    bool remote_write = false;
    bool defer_recv_rearm = false;
    size_t recv_slot = 0;
    size_t first_len = 0;
    const char* payload = nullptr;
    size_t payload_len = 0;
    ibv_mr* payload_mr = nullptr;
    std::vector<RdmaWriteTarget> targets;
    PreparedRead completion;
  };


  auto build_reply = [&](size_t send_slot, const Request& request,
                         Reply* reply, bool try_prepare) -> bool {
    const ReqFields& fields = request.fields;
    const BlockKey key = fields.Key();
    char* send_buffer = ep.sbuf(send_slot);
    auto encode_status = [&](Status status, uint64_t data_len,
                             uint64_t value_len = 0) {
      EncodeRespVersion(send_buffer, wire_epoch, status, data_len, value_len);
    };
    auto invalid_reply = [&] {
      encode_status(Status::kInvalid, 0);
      reply->first_len = response_prefix;
      return true;
    };
    auto data_reply = [&](Status status, const char* data, size_t data_len,
                          size_t value_len, ibv_mr* data_mr,
                          bool source_uses_slot,
                          PreparedRead completion) -> bool {
      const size_t successful_len = status == Status::kOk ? data_len : 0;
      if (status != Status::kOk) {
        encode_status(status, 0);
        reply->first_len = response_prefix;
        return true;
      }
      if (successful_len > request.get.Capacity() ||
          (successful_len != 0 && !data)) {
        return invalid_reply();
      }
      encode_status(Status::kOk, successful_len, value_len);
      reply->first_len = response_prefix;
      if (successful_len != 0) {
        if (!data_mr) return false;
        reply->remote_write = true;
        reply->defer_recv_rearm = source_uses_slot;
        reply->recv_slot = request.recv_slot;
        reply->payload = data;
        reply->payload_len = successful_len;
        reply->payload_mr = data_mr;
        reply->targets = request.get.targets;
        reply->completion = std::move(completion);
      } else {
        completion.Commit(Status::kOk, 0, 0.0);
      }
      return true;
    };

    if (fields.op == static_cast<uint8_t>(WireOp::kRange) &&
        range_handler_) {
      if (!direct_buffer(request.data_slot) ||
          !direct_mr(request.data_slot) ||
          fields.length > static_cast<uint64_t>(conn_max))
        return invalid_reply();

      if (try_prepare && prepare_read_handler_) {
        const double submit_sec = NowSteadySec();
        PreparedRead prepared = prepare_read_handler_(
            key, fields.offset, fields.length,
            direct_buffer(request.data_slot), direct_buffer_cap);
        if (prepared.status() == Status::kOk) {
          if (prepared.needs_io()) {
            const ssize_t got =
                ::pread(prepared.fd(), prepared.staging(),
                        prepared.aligned_len(),
                        static_cast<off_t>(prepared.aligned_off()));
            const bool ok =
                got >= 0 &&
                static_cast<size_t>(got) >=
                    prepared.head() + prepared.payload_len();
            const char* data = prepared.data();
            const size_t payload_len = prepared.payload_len();
            const size_t value_len = prepared.value_len();
            prepared.Commit(ok ? Status::kOk : Status::kIOError,
                            ok ? payload_len : 0,
                            NowSteadySec() - submit_sec);
            return data_reply(ok ? Status::kOk : Status::kIOError,
                              ok ? data : nullptr, payload_len, value_len,
                              direct_mr(request.data_slot),
                              /*source_uses_slot=*/true, PreparedRead{});
          }
          const size_t payload_len = prepared.payload_len();
          ibv_mr* source_mr = direct_mr(request.data_slot);
          if (prepared.source_registered() && payload_len != 0) {
            source_mr = ep.RegisterUser(
                const_cast<char*>(prepared.data()), payload_len);
            if (!source_mr && prepared.Stage())
              source_mr = direct_mr(request.data_slot);
          }
          const char* data = prepared.data();
          return data_reply(Status::kOk, data, payload_len,
                            prepared.value_len(), source_mr,
                            /*source_uses_slot=*/
                            !prepared.source_registered(),
                            std::move(prepared));
        }
        if (prepared.status() != Status::kInvalid) {
          return data_reply(prepared.status(), nullptr, 0, 0, nullptr,
                            /*source_uses_slot=*/false, PreparedRead{});
        }
      }

      const char* output = nullptr;
      size_t output_len = 0;
      size_t value_len = 0;
      const Status status = range_handler_(
          key, fields.offset, fields.length, direct_buffer(request.data_slot),
          direct_buffer_cap, &output, &output_len, &value_len);
      return data_reply(status, output, output_len, value_len,
                        direct_mr(request.data_slot),
                        /*source_uses_slot=*/true, PreparedRead{});
    }

    if (fields.op == static_cast<uint8_t>(WireOp::kCache) &&
        cache_direct_handler_) {
      if (!direct_buffer(request.data_slot) ||
          !direct_mr(request.data_slot) || fields.payload_len == 0 ||
          fields.payload_len > static_cast<uint64_t>(logical_data_cap))
        return invalid_reply();

      char* cache_data = direct_buffer(request.data_slot);
      if (request.contiguous_payload &&
          request.contiguous_payload != cache_data) {
        std::memcpy(cache_data, request.contiguous_payload,
                    static_cast<size_t>(fields.payload_len));
      }
      const Status status = cache_direct_handler_(
          key, cache_data, static_cast<size_t>(fields.payload_len),
          direct_buffer_cap);
      encode_status(status, 0);
      reply->first_len = response_prefix;
      return true;
    }

    const char* payload = nullptr;
    if (fields.payload_len != 0) {
      if (!request.contiguous_payload) return invalid_reply();
      payload = request.contiguous_payload;
    }
    std::string data;
    size_t value_len = 0;
    const Status status =
        handler_(fields.op, key, fields.offset, fields.length, payload,
                 fields.payload_len, &data, &value_len);
    if (fields.op == static_cast<uint8_t>(WireOp::kRange)) {
      if (data.size() > direct_buffer_cap) return invalid_reply();
      if (!data.empty())
        std::memcpy(direct_buffer(request.data_slot), data.data(),
                    data.size());
      return data_reply(status, direct_buffer(request.data_slot), data.size(),
                        value_len,
                        direct_mr(request.data_slot),
                        /*source_uses_slot=*/true, PreparedRead{});
    }
    if (ep.cap() < response_prefix ||
        data.size() > ep.cap() - response_prefix ||
        data.size() > rdma::kV2ControlResponseMax) {
      return false;
    }
    encode_status(status, data.size(), static_cast<uint64_t>(value_len));
    if (!data.empty())
      std::memcpy(send_buffer + response_prefix, data.data(), data.size());
    reply->first_len = response_prefix + data.size();
    return true;
  };

  auto post_reply = [&](size_t send_slot, const Reply& reply) -> bool {
    if (!reply.remote_write) return ep.PostSend(send_slot, reply.first_len);

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
  // A prepared RAM read remains pinned in its move-only transaction until the
  // signaled SEND completion. Uncompleted slots destructor-abort on teardown.
  std::vector<PreparedRead> complete_on_send(K);
  auto complete_send = [&](size_t sid) {
    if (sid < complete_on_send.size()) {
      PreparedRead& read = complete_on_send[sid];
      read.Commit(Status::kOk, read.payload_len(), 0.0);
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
  // reads at once, then emit v2 RDMA WRITEs in arrival order.
  // The slot backing each read stays leased until the signaled status SEND
  // completes, so neither a new PUT nor another GET can overwrite bytes still
  // being consumed by the NIC. Anything that cannot go async falls back to
  // build_reply for that request without reordering.
  if (UseUringPath()) {
    const size_t uring_depth = std::max(UringDepth(K), K);
    // Declared before the ring so destruction order is ring first, prepared
    // reads second. An undrained kernel read therefore cannot outlive its
    // descriptor, slab pin, coalescer flight, or destination staging ownership.
    std::vector<PreparedRead> ring_teardown_reads;
    ring_teardown_reads.reserve(K);
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
        int read_idx = -1;       // >=0: index into descs (async read)
        PreparedRead read;
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
            complete_send(sid);
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

          // Defer a prepared storage read to the io_uring batch. RAM hits are
          // already ready and retain their transaction in the queued reply
          // until SEND completion.
          bool deferred = false;
          bool handled = false;
          if (rq.op == static_cast<uint8_t>(WireOp::kRange) &&
              direct_buffer(request.data_slot) &&
              direct_mr(request.data_slot) &&
              rq.length <= static_cast<uint64_t>(conn_max)) {
            PreparedRead prepared = prepare_read_handler_(
                rq.Key(), rq.offset, rq.length,
                direct_buffer(request.data_slot), direct_buffer_cap);
            if (prepared.status() == Status::kOk && prepared.needs_io() &&
                prepared.fd() >= 0 && prepared.payload_len() != 0 &&
                prepared.aligned_len() <= direct_buffer_cap &&
                prepared.aligned_len() <=
                    std::numeric_limits<unsigned>::max()) {
              UringReader::ReadDesc d;
              d.fd = prepared.fd();
              d.buf = prepared.staging();
              d.len = static_cast<unsigned>(prepared.aligned_len());
              d.off = prepared.aligned_off();
              qd.read_idx = static_cast<int>(descs.size());
              descs.push_back(d);
              qd.read = std::move(prepared);
              qd.submit_sec = NowSteadySec();
              deferred = true;
              handled = true;
            } else if (prepared.status() == Status::kOk &&
                       !prepared.needs_io()) {
              char* sb = ep.sbuf(qd.send_slot);
              const size_t payload_len = prepared.payload_len();
              if (payload_len > qd.get.Capacity() ||
                  (payload_len != 0 && prepared.data() == nullptr)) {
                EncodeRespVersion(sb, wire_epoch, Status::kInvalid, 0);
                qd.reply.first_len = response_prefix;
              } else {
                EncodeRespVersion(sb, wire_epoch, Status::kOk, payload_len,
                                  prepared.value_len());
                qd.reply.first_len = response_prefix;
                if (payload_len != 0) {
                  ibv_mr* source_mr = direct_mr(request.data_slot);
                  if (prepared.source_registered()) {
                    source_mr = ep.RegisterUser(
                        const_cast<char*>(prepared.data()), payload_len);
                    if (!source_mr && prepared.Stage())
                      source_mr = direct_mr(request.data_slot);
                  }
                  if (!source_mr) {
                    fail = true;
                    break;
                  }
                  qd.reply.remote_write = true;
                  qd.reply.defer_recv_rearm =
                      !prepared.source_registered();
                  qd.reply.recv_slot = request.recv_slot;
                  qd.reply.payload = prepared.data();
                  qd.reply.payload_len = payload_len;
                  qd.reply.payload_mr = source_mr;
                  qd.reply.targets = qd.get.targets;
                  qd.reply.completion = std::move(prepared);
                } else {
                  prepared.Commit(Status::kOk, 0, 0.0);
                }
              }
              handled = true;
            } else if (prepared.status() != Status::kOk &&
                       prepared.status() != Status::kInvalid) {
              EncodeRespVersion(ep.sbuf(qd.send_slot), wire_epoch,
                                prepared.status(), 0);
              qd.reply.first_len = response_prefix;
              handled = true;
            }
          }

          if (!deferred && !handled) {
            // An existing coalescer flight or unsupported shape takes the
            // synchronous fallback, without re-running preparation.
            if (!build_reply(s, request, &qd.reply,
                             /*try_prepare=*/false)) {
              fail = true;
              break;
            }
          }
          queue.push_back(std::move(qd));
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
          // v2 shared-segment slots. Drain before the sync fallback
          // or receive rearm can reuse those buffers.
          // writes and put mixed-generation bytes on the wire. The native wire
          // carries a canonical identity but intentionally has no payload CRC;
          // RDMA ICRC covers only transport integrity.
          // the kernel: the only safe move is to drop the connection while the
          // endpoint (and its registered buffers) is still alive; the client
          // re-dials. Once poisoned, later BatchRead calls return false without
          // submitting, so the connection continues on the sync fallback.
          if (!batch_ok && !ring.Drain()) {
            for (auto& qd : queue)
              ring_teardown_reads.push_back(std::move(qd.read));
            fail = true;
            break;
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
            complete_on_send[qd.send_slot] = std::move(reply.completion);
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
            ok = res >= 0 &&
                 static_cast<size_t>(res) >=
                     qd.read.head() + qd.read.payload_len();
          } else {
            ssize_t got =
                ::pread(qd.read.fd(), d.buf, d.len, static_cast<off_t>(d.off));
            ok = got >= 0 &&
                 static_cast<size_t>(got) >=
                     qd.read.head() + qd.read.payload_len();
          }
          // Terminal completion runs before the reply send so coalesced waiters
          // and RAM promotion can copy from the still-owned staging bytes.
          const char* out_data = qd.read.data();
          const size_t payload_len = qd.read.payload_len();
          const size_t value_len = qd.read.value_len();
          qd.read.Commit(ok ? Status::kOk : Status::kIOError,
                         ok ? payload_len : 0,
                         NowSteadySec() - qd.submit_sec);
          char* sb = ep.sbuf(qd.send_slot);
          if (!ok) {
            EncodeRespVersion(sb, wire_epoch, Status::kIOError, 0);
            if (!post_request_recv(qd.recv_slot) ||
                !ep.PostSend(qd.send_slot, response_prefix)) {
              fail = true;
              break;
            }
            continue;
          }
          if (payload_len > qd.get.Capacity()) {
            EncodeRespVersion(sb, wire_epoch, Status::kInvalid, 0);
            if (!post_request_recv(qd.recv_slot) ||
                !ep.PostSend(qd.send_slot, response_prefix)) {
              fail = true;
              break;
            }
            continue;
          }
          EncodeRespVersion(sb, wire_epoch, Status::kOk, payload_len,
                            value_len);
          Reply reply;
          reply.first_len = response_prefix;
          if (payload_len != 0) {
            reply.remote_write = true;
            reply.payload = out_data;
            reply.payload_len = payload_len;
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
        }
      }

      // Queued PreparedRead destructors abort any entries that never reached
      // Commit. Prepared SEND owners in complete_on_send do the same on
      // connection teardown.
    }
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
        complete_send(sid);
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
      bool built = build_reply(s, request, &reply, /*try_prepare=*/true);
      if (!built) { fail = true; break; }
      if (reply.defer_recv_rearm) {
        rearm_on_send[s] = reply.recv_slot;
      } else if (!post_request_recv(r)) {
        fail = true; break;  // re-arm (request consumed)
      }
      complete_on_send[s] = std::move(reply.completion);
      bool sent = post_reply(s, reply);
      if (!sent) { fail = true; break; }
    }
  }
  // Any prepared sends without completions destructor-abort below.
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
    recv_segment_registered_rails_ > 0 ? 1 : 0);
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
