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
}  // namespace

RdmaServer::RdmaServer(Handler handler, size_t max_msg, const std::string& dev_name)
    : handler_(std::move(handler)),
      max_msg_(ResolveMaxPayload(max_msg)),
      control_cap_(ControlCapFor(max_msg_)),
      dev_name_(dev_name) {
  if (dev_name_.empty()) {
    const char* e = std::getenv("DFKV_RDMA_DEV");
    if (e && *e) dev_name_ = e;
  }
  config_dump::RecordResolved("DFKV_RDMA_DEV", dev_name_.empty() ? "(auto)" : dev_name_);
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
  // Anchor the pool regions' MRs for the server's lifetime. The shared-device
  // registry is refcounted by live endpoints: without an anchor, the moment
  // the LAST client connection is torn down (disconnect or idle reclaim) the
  // device closes and the pool MRs — arena-scale, seconds to pin — are
  // deregistered, so the NEXT client's first op stalls ~5 s re-registering
  // 128 GiB (observed on B200). The anchor endpoint holds one device ref and
  // performs the registration once, HERE, where the cost is expected and
  // reported. Only the configured device is anchored; a client requesting a
  // different rail still pays that rail's first registration
  // (SharedAddPoolMr logs slow ones).
  if (!user_regions_.empty()) {
    for (const auto& d : anchor_devs_) {
      auto a = std::make_unique<rdma::RcEndpoint>();
      if (a->Open(d.empty() ? nullptr : d.c_str(), 4096, 1)) {
        a->EnsurePoolMrs(user_regions_);  // arena-scale pin: once, here, reported
        anchors_.push_back(std::move(a));
      }
      // An unusable listed rail is skipped (clients requesting it fail the
      // same way with or without an anchor); the default rail failing is the
      // legacy no-anchor behavior.
    }
    if (anchor_devs_.size() > 1)
      DFKV_LOG_INFO("rdma multi-rail anchors: " + std::to_string(anchors_.size()) +
                    "/" + std::to_string(anchor_devs_.size()) + " devices pinned");
  }
  running_ = true;
  accept_thread_ = std::thread([this] { NameThisThread("rdma-accept"); AcceptLoop(); });
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
  if (!net::ReadAll(boot_fd, devbuf, rdma::kDevNameBytes)) { ::close(boot_fd); return; }
  // DCP1 (issue #110): the frame's zero tail may carry the client's declared
  // max block size; size THIS connection's per-slot buffers to it instead of
  // the global worst case. Undeclared (old client / no room) = max_msg_, the
  // exact old behavior. Never below the control floor so metadata ops and
  // small values always fit.
  const uint64_t declared = rdma::ParseDevFrameCaps(devbuf);
  // A declaration ABOVE our cap is refused rather than quietly sized down.
  // The client bounds its own ops by what it declared (RdmaTransport::OpBound),
  // so sizing this connection smaller than the declaration would leave it
  // sending past our posted receives -- an RNR/QP break far from the cause.
  // Refusing here names both ends of the mismatch in one line instead.
  //
  // This is what makes --max-msg a usable memory guard: the per-connection
  // pin is qd x (ValueHeader + conn_max), so an undeclaring client costs the
  // full worst case (measured on a B200 node: ~2 GB/conn at 64 MiB and qd=32,
  // against 0.34 GB when the client declares 4 MiB). Without a server-side
  // ceiling the server's memory budget is decided entirely by its clients,
  // which are often deployed by someone else and versioned independently.
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

  rdma::RcEndpoint ep;
  const size_t K = ServerDepth();
  const size_t conn_control = ControlCapFor(conn_max);
  const size_t direct_cap = ValueHeader::kSize + conn_max;
  if (!ep.Open(dev.empty() ? nullptr : dev.c_str(), conn_control, K,
               /*ib_port=*/1, /*direct_io_buffers=*/true, direct_cap)) {
    ::close(boot_fd); return;
  }
  if (declared)
    DFKV_LOG_INFO("rdma conn caps: declared=" + std::to_string(declared) +
                  " -> per-slot dbuf=" + std::to_string(direct_cap) +
                  " control=" + std::to_string(conn_control) +
                  " (qd=" + std::to_string(K) + ")");
  numa::PinThreadToNode(ep.numa_node());  // keep this conn's serve thread NUMA-local to its NIC
  // QP bootstrap: read client's info, send ours (symmetric to the client).
  char peer[rdma::kQpInfoBytes], mine[rdma::kQpInfoBytes];
  rdma::QpInfo my = ep.Local();
  my.depth = static_cast<uint16_t>(std::min<size_t>(K, 256));  // DPQ1: advertise our posted-recv depth
  rdma::SerializeQpInfo(my, mine);
  if (!net::ReadAll(boot_fd, peer, rdma::kQpInfoBytes) ||
      !net::WriteAll(boot_fd, mine, rdma::kQpInfoBytes)) {
    ::close(boot_fd); return;
  }
  if (!ep.Connect(rdma::ParseQpInfo(peer))) { ::close(boot_fd); return; }
  // Register the RAM arena (and any other declared region) as a pool MR on this
  // connection's PD so a RAM-hit payload resolves to an MR with no per-op reg_mr
  // (B5-3). No-op when nothing was registered.
  if (!user_regions_.empty()) ep.EnsurePoolMrs(user_regions_);
  auto post_request_recv = [&](size_t slot) {
    if (!ep.dbuf(slot) || !ep.dmr(slot) || ep.dbuf_cap() <= ValueHeader::kSize)
      return ep.PostRecv(slot);
    return ep.PostRecvScatter(slot, ep.dbuf(slot) + ValueHeader::kSize,
                              ep.dbuf_cap() - ValueHeader::kSize, ep.dmr(slot),
                              kReqPrefix + ValueHeader::kSize);
  };

  // Post all K receives so the client may keep up to K requests in flight
  // (pipelining). recv buffers and send buffers are independent slot pools.
  bool armed = true;
  for (size_t i = 0; i < K; ++i) armed = armed && post_request_recv(i);
  if (!armed) { ::close(boot_fd); return; }
  // Tell the client we are ready (recvs posted) so its first SENDs won't hit RNR.
  char ready = 1;
  bool ok = net::WriteAll(boot_fd, &ready, 1);
  ::close(boot_fd);  // bootstrap done
  if (!ok) return;

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

  struct Reply {
    bool scatter = false;
    bool defer_recv_rearm = false;
    size_t recv_slot = 0;
    size_t first_len = 0;  // plain: full sbuf length; scatter: sbuf header length
    const void* payload = nullptr;
    size_t payload_len = 0;
    ibv_mr* payload_mr = nullptr;
    uint64_t release_token = 0;  // RAM-hit send-in-flight pin; 0 = none (B5-3)
  };

  // Build the reply for one request. Generic ops are materialized in sbuf. For
  // kRange with a direct range handler set, O_DIRECT reads into ep.dbuf[recv_slot]
  // (4096-aligned + registered) and the wire reply is scatter-sent as
  // [resp-prefix from sbuf | value bytes from dbuf], with no payload memcpy.
  // For kCache with a direct cache handler set, the RDMA RECV has already placed
  // the user payload into ep.dbuf[recv_slot]+ValueHeader::kSize; only the 48B
  // value header is copied over so disk writes see one contiguous aligned blob.
  auto copy_payload = [&](size_t recv_slot, const ReqFields& rq, uint32_t recv_bytes,
                          std::string* payload) -> bool {
    payload->clear();
    if (rq.payload_len == 0) return true;
    if (recv_bytes < kReqPrefix + rq.payload_len) return false;
    if (rq.payload_len > static_cast<uint64_t>(ValueHeader::kSize + conn_max))
      return false;
    payload->resize(static_cast<size_t>(rq.payload_len));
    const size_t head = static_cast<size_t>(
        std::min<uint64_t>(rq.payload_len, ValueHeader::kSize));
    if (head) std::memcpy(payload->data(), ep.rbuf(recv_slot) + kReqPrefix, head);
    const size_t rest = static_cast<size_t>(rq.payload_len) - head;
    if (rest != 0) {
      if (!ep.dbuf(recv_slot) || ep.dbuf_cap() < ValueHeader::kSize + rest) return false;
      std::memcpy(payload->data() + head, ep.dbuf(recv_slot) + ValueHeader::kSize, rest);
    }
    return true;
  };

  auto build_reply = [&](size_t send_slot, size_t recv_slot, const ReqFields& rq,
                         uint32_t recv_bytes,
                         Reply* reply) -> bool {
    char* sb = ep.sbuf(send_slot);
    auto invalid_reply = [&] {
      EncodeResp(sb, Status::kInvalid, 0);
      reply->first_len = kRespPrefix;
      return true;
    };
    if (rq.op == static_cast<uint8_t>(WireOp::kRange) && range_handler_) {
      if (!ep.dbuf(recv_slot) || !ep.dmr(recv_slot)) return false;
      if (rq.length > static_cast<uint64_t>(ValueHeader::kSize + conn_max))
        return invalid_reply();
      // RAM hot tier (B5-3): if the block is resident in the arena, scatter-send
      // it STRAIGHT from the arena MR -- no copy into dbuf, no disk. The slot is
      // pinned until this send completes (release_token -> ram_release_handler_).
      if (ram_range_handler_) {
        const char* rptr = nullptr; size_t rlen = 0; uint64_t tok = 0;
        if (ram_range_handler_(rq.id, rq.index, rq.size, rq.offset, rq.length,
                               &rptr, &rlen, &tok)) {
          ibv_mr* amr = rlen ? ep.RegisterUser(const_cast<char*>(rptr), rlen) : nullptr;
          if (rlen != 0 && amr == nullptr) {  // MR resolve failed -> release + fall back
            if (ram_release_handler_) ram_release_handler_(tok);
          } else {
            EncodeResp(sb, Status::kOk, rlen);
            reply->first_len = kRespPrefix;
            if (rlen != 0) {
              reply->scatter = true;
              reply->defer_recv_rearm = true;
              reply->recv_slot = recv_slot;
              reply->payload = rptr;
              reply->payload_len = rlen;
              reply->payload_mr = amr;
              reply->release_token = tok;
            } else {
              if (ram_release_handler_) ram_release_handler_(tok);  // 0-len: nothing to send
            }
            return true;
          }
        }
      }
      const char* out_data = nullptr;
      size_t out_len = 0;
      Status st = range_handler_(rq.id, rq.index, rq.size, rq.offset, rq.length,
                                 ep.dbuf(recv_slot), ep.dbuf_cap(), &out_data, &out_len);
      uint64_t dlen = (st == Status::kOk) ? out_len : 0;
      EncodeResp(sb, st, dlen);
      reply->first_len = kRespPrefix;
      if (st == Status::kOk && dlen != 0) {
        if (!out_data) return false;
        reply->scatter = true;
        reply->defer_recv_rearm = true;
        reply->recv_slot = recv_slot;
        reply->payload = out_data;
        reply->payload_len = out_len;
        reply->payload_mr = ep.dmr(recv_slot);
      }
      return true;
    }

    if (rq.op == static_cast<uint8_t>(WireOp::kCache) && cache_direct_handler_) {
      if (!ep.dbuf(recv_slot) || !ep.dmr(recv_slot)) return false;
      if (rq.payload_len < ValueHeader::kSize ||
          rq.payload_len > static_cast<uint64_t>(ValueHeader::kSize + conn_max) ||
          recv_bytes < kReqPrefix + rq.payload_len) {
        return invalid_reply();
      }
      std::memcpy(ep.dbuf(recv_slot), ep.rbuf(recv_slot) + kReqPrefix,
                  ValueHeader::kSize);
      Status st = cache_direct_handler_(
          rq.id, rq.index, rq.size, ep.dbuf(recv_slot),
          static_cast<size_t>(rq.payload_len), ep.dbuf_cap());
      EncodeResp(sb, st, 0);
      reply->first_len = kRespPrefix;
      return true;
    }

    std::string payload_buf;
    const char* payload = nullptr;
    if (rq.payload_len != 0) {
      if (!copy_payload(recv_slot, rq, recv_bytes, &payload_buf)) return invalid_reply();
      payload = payload_buf.data();
    }
    std::string data;
    Status st = handler_(rq.op, rq.id, rq.index, rq.size, rq.offset, rq.length,
                         payload, rq.payload_len, &data);
    if (kRespPrefix + data.size() > ep.cap()) return false;
    EncodeResp(sb, st, data.size());
    if (!data.empty()) std::memcpy(sb + kRespPrefix, data.data(), data.size());
    reply->first_len = kRespPrefix + data.size();
    return true;
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
  // sync loop; for the kRange GET RECVs, do the cheap prep (open O_DIRECT fd +
  // index lookup + alignment) into an ordered descriptor list, submit ALL their
  // reads to io_uring at once (QD>1 in flight => saturates the SSD), WAIT for the
  // whole batch, then PostSendScatter each reply IN ARRIVAL ORDER. Because every
  // read is complete before any reply is sent, in-order replies are preserved
  // trivially — no reorder buffer, no out-of-order completion handling. Anything
  // that can't go async (miss, zero-len, oversize, prep error) falls back to the
  // synchronous build_reply for THAT request, still emitted in arrival order.
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
          if (wc.opcode != IBV_WC_RECV || wc.byte_len < kReqPrefix) { fail = true; break; }
          completions_.fetch_add(1, std::memory_order_relaxed);  // a request RECV
          size_t r = static_cast<size_t>(wc.wr_id);
          ReqFields rq;
          if (!DecodeReq(ep.rbuf(r), &rq)) { fail = true; break; }
          if (free_send.empty()) { fail = true; break; }
          size_t s = free_send.back(); free_send.pop_back();

          Queued qd;
          qd.send_slot = s;
          qd.recv_slot = r;

          // Defer a kRange GET hit's disk read to the io_uring batch below; reply
          // after the whole batch completes (the emit pass preserves arrival order).
          bool deferred = false;
          if (rq.op == static_cast<uint8_t>(WireOp::kRange) &&
              ep.dbuf(r) && ep.dmr(r) &&
              rq.length <= static_cast<uint64_t>(ValueHeader::kSize + conn_max)) {
            RangePrepResult pr;
            Status pst = range_prep_handler_(rq.id, rq.index, rq.size, rq.offset,
                                             rq.length, ep.dbuf_cap(), &pr);
            if (pst == Status::kOk && pr.fd >= 0 && pr.payload_len != 0 &&
                pr.aligned_len <= ep.dbuf_cap() &&
                pr.aligned_len <= std::numeric_limits<unsigned>::max()) {
              UringReader::ReadDesc d;
              d.fd = pr.fd;
              d.buf = ep.dbuf(r);
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
            if (!build_reply(s, r, rq, wc.byte_len, &qd.reply)) { fail = true; break; }
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
          // A failed batch may have left async reads in flight against the
          // recv-slot dbufs. Drain (reap-and-discard) until the kernel is done
          // with them BEFORE the emit pass touches those buffers: the sync
          // pread fallback and the recv rearm would otherwise race the kernel's
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
            bool sent = reply.scatter
                            ? ep.PostSendScatter(qd.send_slot, reply.first_len,
                                                 reply.payload, reply.payload_len,
                                                 reply.payload_mr)
                            : ep.PostSend(qd.send_slot, reply.first_len);
            if (!sent) { fail = true; break; }
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
          // Runs BEFORE the reply send on purpose: coalesced waiters and the
          // RAM promotion copy from the dbuf payload, and the buffer is reused
          // the moment the scatter SEND is posted.
          if (range_complete_handler_) {
            range_complete_handler_(ok, ok ? qd.payload_len : 0,
                                    NowSteadySec() - qd.submit_sec, qd.flight,
                                    ok ? ep.dbuf(qd.recv_slot) + qd.head : nullptr);
            qd.flight = 0;  // completed: the teardown sweep must not abort it
          }
          char* sb = ep.sbuf(qd.send_slot);
          if (ok) {
            EncodeResp(sb, Status::kOk, qd.payload_len);
            const char* out_data = ep.dbuf(qd.recv_slot) + qd.head;
            // Defer recv rearm until this scatter SEND completes (read target reuse).
            rearm_on_send[qd.send_slot] = qd.recv_slot;
            if (!ep.PostSendScatter(qd.send_slot, kRespPrefix, out_data,
                                    qd.payload_len, ep.dmr(qd.recv_slot))) {
              fail = true; break;
            }
          } else {
            EncodeResp(sb, Status::kIOError, 0);
            if (!post_request_recv(qd.recv_slot)) { fail = true; break; }
            if (!ep.PostSend(qd.send_slot, kRespPrefix)) { fail = true; break; }
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
      if (wc.opcode != IBV_WC_RECV || wc.byte_len < kReqPrefix) { fail = true; break; }
      completions_.fetch_add(1, std::memory_order_relaxed);  // a request RECV
      size_t r = static_cast<size_t>(wc.wr_id);
      ReqFields rq;
      if (!DecodeReq(ep.rbuf(r), &rq)) { fail = true; break; }
      if (free_send.empty()) { fail = true; break; }
      size_t s = free_send.back(); free_send.pop_back();
      Reply reply;
      bool built = build_reply(s, r, rq, wc.byte_len, &reply);  // consumes rbuf/dbuf[r]
      if (!built) { fail = true; break; }
      if (reply.defer_recv_rearm) {
        rearm_on_send[s] = reply.recv_slot;
      } else if (!post_request_recv(r)) {
        fail = true; break;  // re-arm (request consumed)
      }
      release_on_send[s] = reply.release_token;  // RAM-hit pin (B5-3)
      bool sent = reply.scatter
                      ? ep.PostSendScatter(s, reply.first_len, reply.payload,
                                           reply.payload_len, reply.payload_mr)
                      : ep.PostSend(s, reply.first_len);
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
  m(s, "dfkv_rdma_active_conns", "gauge", "RDMA connections currently serving",
    ActiveConns());
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
