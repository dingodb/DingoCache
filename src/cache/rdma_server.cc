#include "common/config_dump.h"
#include "cache/rdma_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <deque>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "utils/log.h"          // DFKV_LOG_WARN (uring init fallback)
#include "utils/prom_escape.h"
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
// Draw every token from the kernel CRNG. Unlike a userspace PRNG or counter,
// getrandom has no inherited/reset state that can replay after fork or restart.
uint64_t RandomNonzeroWriterToken() {
  for (;;) {
    uint64_t token = 0;
    size_t filled = 0;
    while (filled < sizeof(token)) {
      const ssize_t n =
          ::getrandom(reinterpret_cast<char*>(&token) + filled,
                      sizeof(token) - filled, 0);
      if (n > 0) {
        filled += static_cast<size_t>(n);
        continue;
      }
      if (n < 0 && errno == EINTR) continue;
      const int error = n < 0 ? errno : 0;
      DFKV_LOG_ERROR("rdma: secure writer-token generation failed errno=" +
                     std::to_string(error));
      std::abort();
    }
    if (token != 0) return token;
  }
}

// Monotonic seconds for the async-read submit->complete latency stamp. Read
// twice per deferred GET (prep + completion), both off the SSD-bound path, so
// the vDSO clock read is amortized away.
inline uint64_t SteadyUs() {
  // steady_clock::count() is nanoseconds on Linux; cast to real microseconds.
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}
inline double NowSteadySec() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}


size_t RecvSegmentBytes() {
  constexpr size_t kDefault = 128ull << 30;
  const size_t bytes = rdma::ResolveRecvSegmentBytes(
      std::getenv("DFKV_RDMA_RECV_SEGMENT_SIZE"), kDefault,
      rdma::kV2DataOffset);
  config_dump::RecordResolved("DFKV_RDMA_RECV_SEGMENT_SIZE",
                              std::to_string(bytes));
  return bytes;
}

size_t RecvChunkBytes(size_t max_bytes) {
  constexpr size_t kDefault = 256ull << 20;
  const size_t fallback = std::min(kDefault, max_bytes);
  const size_t parsed = rdma::ResolveRecvSegmentBytes(
      std::getenv("DFKV_RDMA_RECV_CHUNK_BYTES"), fallback,
      rdma::kV2DataOffset);
  const size_t bytes = parsed == 0 ? 0 : std::min(parsed, max_bytes);
  config_dump::RecordResolved("DFKV_RDMA_RECV_CHUNK_BYTES",
                              std::to_string(bytes));
  return bytes;
}

uint64_t RecvChunkIdleMs() {
  constexpr uint64_t kDefault = 60000;
  const char* value = std::getenv("DFKV_RDMA_RECV_CHUNK_IDLE_MS");
  uint64_t out = kDefault;
  if (value && *value) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno == 0 && end != value && *end == '\0')
      out = std::min<uint64_t>(parsed, 86400000);
  }
  config_dump::RecordResolved("DFKV_RDMA_RECV_CHUNK_IDLE_MS",
                              std::to_string(out));
  return out;
}

const char* DiscoveryStatusName(rdma::RdmaDiscoveryStatus status) {
  switch (status) {
    case rdma::RdmaDiscoveryStatus::kOk:
      return "ok";
    case rdma::RdmaDiscoveryStatus::kDeviceListFailed:
      return "device-list-failed";
    case rdma::RdmaDiscoveryStatus::kConfiguredDeviceMissing:
      return "configured-device-missing";
    case rdma::RdmaDiscoveryStatus::kDeviceOpenFailed:
      return "device-open-failed";
    case rdma::RdmaDiscoveryStatus::kPortQueryFailed:
      return "port-query-failed";
    case rdma::RdmaDiscoveryStatus::kGidQueryFailed:
      return "gid-query-failed";
  }
  return "unknown";
}

void LogTopologySummary(size_t configured, size_t initialized,
                        const std::vector<rdma::RdmaDevInfo>& devices,
                        bool discovery_complete = true,
                        const std::string& unresolved = {}) {
  size_t active = 0;
  std::string inactive;
  for (const auto& device : devices) {
    if (device.active) {
      ++active;
      continue;
    }
    if (!inactive.empty()) inactive += ",";
    inactive += device.name;
  }
  std::string summary =
      "rdma topology startup: configured=" + std::to_string(configured) +
      " initialized=" + std::to_string(initialized);
  if (discovery_complete) {
    summary += " ACTIVE=" + std::to_string(active) +
               " inactive=" + (inactive.empty() ? "(none)" : inactive);
  } else {
    summary += " probed=" + std::to_string(devices.size()) +
               " observed_ACTIVE=" + std::to_string(active) +
               " observed_inactive=" +
               (inactive.empty() ? "(none)" : inactive);
  }
  if (!unresolved.empty()) summary += " unresolved=" + unresolved;
  DFKV_LOG_INFO(summary);
}
}  // namespace
uint64_t RdmaServer::RegisterWriter(
    const std::shared_ptr<WriterState>& writer) {
  for (;;) {
    const uint64_t token = RandomNonzeroWriterToken();
    std::lock_guard<std::mutex> lock(writer_mu_);
    if (writers_.emplace(token, writer).second) return token;
  }
}


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
    if (!d.empty() &&
        std::find(anchor_devs_.begin(), anchor_devs_.end(), d) ==
            anchor_devs_.end())
      anchor_devs_.push_back(d);
    i = c + 1;
  }
  if (anchor_devs_.empty()) anchor_devs_.push_back("");
  dev_name_ = anchor_devs_.front();
}

RdmaServer::~RdmaServer() { Stop(); }

Status RdmaServer::Start(int port) {
  // An explicitly configured anchor whose name overruns the v2 bootstrap
  // frame stays fully usable here — this server only EVER parses peer frames
  // and default-anchor (empty-name) clients fall back to it — but it can
  // never be selected BY NAME: an announceable client name must fit the frame
  // (client-side startup rejects longer ones). Name it at startup with the
  // real limit instead of leaving ops to decode per-connection rejections.
  for (const auto& device : anchor_devs_) {
    if (!device.empty() && !rdma::DeviceNameFitsFrame(device)) {
      DFKV_LOG_WARN(rdma::OversizedDeviceNameError(device) +
                    "; serving anyway, reachable as the default anchor only");
    }
  }
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
  // Automatic discovery remains ACTIVE-only and selects the first verbs HCA.
  // Explicit configuration is different: every named HCA must resolve, but an
  // initially inactive port remains a fixed rail so runtime health can recover
  // it without changing topology indices.
  std::vector<std::string> requested_devices;
  if (!auto_device_) requested_devices = anchor_devs_;
  const rdma::RdmaDiscoveryPolicy discovery_policy =
      auto_device_ ? rdma::RdmaDiscoveryPolicy::kActiveOnly
                   : rdma::RdmaDiscoveryPolicy::kAllowInactive;
  rdma::RdmaDiscoveryResult discovery =
      discover_for_test_
          ? discover_for_test_(requested_devices, discovery_policy)
          : rdma::RdmaTopology::Discover(requested_devices, discovery_policy);
  if (!discovery.ok()) {
    LogTopologySummary(requested_devices.size(), 0,
                       discovery.observed_devices, false,
                       discovery.failed_device);
    DFKV_LOG_ERROR(
        "rdma: device discovery failed: status=" +
        std::string(DiscoveryStatusName(discovery.status)) +
        (discovery.failed_device.empty()
             ? std::string()
             : " device=" + discovery.failed_device));
    ::close(listen_fd_);
    listen_fd_ = -1;
    return Status::kIOError;
  }
  if (auto_device_ && discovery.devices.size() > 1)
    discovery.devices.resize(1);
  if (discovery.devices.empty()) {
    LogTopologySummary(requested_devices.size(), 0, discovery.devices);
    DFKV_LOG_ERROR("rdma: no ACTIVE device found");
    ::close(listen_fd_);
    listen_fd_ = -1;
    return Status::kIOError;
  }
  if (!auto_device_) {
    bool exact = discovery.devices.size() == requested_devices.size();
    for (size_t i = 0; exact && i < requested_devices.size(); ++i)
      exact = discovery.devices[i].name == requested_devices[i];
    if (!exact) {
      LogTopologySummary(requested_devices.size(), 0, discovery.devices);
      DFKV_LOG_ERROR(
          "rdma: explicit discovery returned a partial or reordered topology");
      ::close(listen_fd_);
      listen_fd_ = -1;
      return Status::kIOError;
    }
  }

  std::vector<std::string> resolved_device_names;
  resolved_device_names.reserve(discovery.devices.size());
  std::string resolved_devices;
  for (const auto& device : discovery.devices) {
    resolved_device_names.push_back(device.name);
    if (!resolved_devices.empty()) resolved_devices += ",";
    resolved_devices += device.name;
  }
  anchor_devs_ = std::move(resolved_device_names);
  dev_name_ = anchor_devs_.front();
  config_dump::RecordResolved("DFKV_RDMA_DEV", resolved_devices);

  // Commit one small receive chunk at startup; additional chunks are allocated
  // only when live connection leases need them. The legacy segment setting is
  // now the hard process budget rather than an eager allocation size.
  recv_segment_max_bytes_ = RecvSegmentBytes();
  recv_segment_chunk_bytes_ = RecvChunkBytes(recv_segment_max_bytes_);
  recv_chunk_idle_ms_ = RecvChunkIdleMs();
  const size_t min_slot_bytes = rdma::V2SlotSize(max_msg_);
  if (min_slot_bytes == 0 ||
      recv_segment_max_bytes_ < 2 * min_slot_bytes ||
      recv_segment_chunk_bytes_ == 0 ||
      !recv_segments_.Init(recv_segment_chunk_bytes_,
                           recv_segment_max_bytes_,
                           rdma::kV2DataOffset)) {
    LogTopologySummary(requested_devices.size(), 0, discovery.devices);
    DFKV_LOG_ERROR(
        "rdma: invalid receive-pool geometry max=" +
        std::to_string(recv_segment_max_bytes_) +
        " chunk=" + std::to_string(recv_segment_chunk_bytes_) +
        " minimum-two-slot-bytes=" + std::to_string(2 * min_slot_bytes) +
        "; raise DFKV_RDMA_RECV_SEGMENT_SIZE, adjust "
        "DFKV_RDMA_RECV_CHUNK_BYTES, or lower --max-msg");
    recv_segment_max_bytes_ = 0;
    recv_segment_chunk_bytes_ = 0;
    ::close(listen_fd_);
    listen_fd_ = -1;
    return Status::kIOError;
  }
  rdma::RecvSegment* initial_segment = recv_segments_.initial_segment();

  std::vector<std::unique_ptr<rdma::RcEndpoint>> initialized_anchors;
  initialized_anchors.reserve(anchor_devs_.size());
  for (const auto& device : anchor_devs_) {
    std::unique_ptr<rdma::RcEndpoint> anchor;
    if (initialize_anchor_for_test_) {
      anchor = initialize_anchor_for_test_(
          device, user_regions_, initial_segment->data(),
          initial_segment->size());
    } else {
      anchor = std::make_unique<rdma::RcEndpoint>();
      if (!anchor->Open(device.c_str(), rdma::kV2ControlCap, 1)) {
        DFKV_LOG_ERROR("rdma: failed to open required v2 device " + device);
        anchor.reset();
      } else if (!anchor->EnsurePoolMrs(user_regions_)) {
        DFKV_LOG_ERROR("rdma: failed to register explicit user region on " +
                       device);
        anchor.reset();
      } else if (!anchor->RegisterRemoteRegion(initial_segment->data(),
                                               initial_segment->size())) {
        DFKV_LOG_ERROR(
            "rdma: failed to register initial v2 receive chunk on " +
            device);
        anchor.reset();
      }
    }
    if (!anchor) {
      LogTopologySummary(requested_devices.size(),
                         initialized_anchors.size(), discovery.devices);
      DFKV_LOG_ERROR(
          "rdma: every resolved rail must initialize; startup aborted");
      recv_segment_registered_rails_ = 0;
      ::close(listen_fd_);
      listen_fd_ = -1;
      return Status::kIOError;
    }
    initialized_anchors.push_back(std::move(anchor));
  }

  std::vector<std::unique_ptr<RailStats>> initialized_stats;
  initialized_stats.reserve(anchor_devs_.size());
  for (size_t i = 0; i < anchor_devs_.size(); ++i)
    initialized_stats.push_back(std::make_unique<RailStats>());
  anchors_ = std::move(initialized_anchors);
  rail_stats_ = std::move(initialized_stats);
  recv_segment_registered_rails_ = anchors_.size();
  LogTopologySummary(requested_devices.size(), anchors_.size(),
                     discovery.devices);
  if (anchor_devs_.size() > 1)
    DFKV_LOG_INFO("rdma multi-rail anchors: " +
                  std::to_string(anchors_.size()) + "/" +
                  std::to_string(anchor_devs_.size()) +
                  " devices pinned");
  running_ = true;
  accept_thread_ =
      std::thread([this] { NameThisThread("rdma-accept"); AcceptLoop(); });
  if (recv_chunk_idle_ms_ != 0) {
    recv_trim_thread_ = std::thread([this] {
      NameThisThread("rdma-recv-trim");
      std::unique_lock<std::mutex> lock(recv_trim_mu_);
      while (running_.load(std::memory_order_relaxed)) {
        if (recv_trim_cv_.wait_for(lock, std::chrono::seconds(1), [this] {
              return !running_.load(std::memory_order_relaxed);
            }))
          break;
        lock.unlock();
        recv_segments_.TrimIdle(recv_chunk_idle_ms_);
        lock.lock();
      }
    });
  }
  return Status::kOk;
}

void RdmaServer::Stop() {
  if (!running_.exchange(false)) return;
  recv_trim_cv_.notify_all();
  if (listen_fd_ >= 0) ::shutdown(listen_fd_, SHUT_RDWR);  // wake accept()
  if (accept_thread_.joinable()) accept_thread_.join();
  if (recv_trim_thread_.joinable()) recv_trim_thread_.join();
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
  // Pipeline depth (requests in flight per connection). Default 4 matches the
  // typical client depth (DFKV_RDMA_DEPTH=4 in production deployments). A server
  // depth lower than the client causes the batching window to clamp, which
  // silently degrades throughput 3-4x on pipelined GETs and causes PUT batch
  // failures during burst writes (observed 532 "Write page to storage: 128
  // pages failed" on GLM-5.2-NVFP4 with depth=1 server vs depth=4 client,
  // resulting in hot-round L3 prefetch failures and -29.8% throughput vs cold).
  size_t out = 4;
  const char* e = std::getenv("DFKV_RDMA_DEPTH");
  if (e && *e) { long v = std::strtol(e, nullptr, 10); if (v >= 1 && v <= 256) out = (size_t)v; }
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
  int out = 600000;  // 10 min; K8s launcher exports DFKV_RDMA_IDLE_MS=30000 explicitly
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
// Per-connection disk queue depth. It is independent of negotiated RDMA depth:
// requests beyond this bound remain prepared but unsubmitted until a CQE frees
// capacity. Sixteen is enough to expose the SSD queue on the measured hosts;
// operators may select 32 (or another [1, 256] value) independently.
size_t UringDepth() {
  size_t out = 16;
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
  // Default ON when built with io_uring. The split submit/reap pipeline keeps
  // bounded disk reads outstanding while continuing to service verbs CQEs, and
  // serializes only reply emission. Ring initialization may fall back before
  // the first SQE; an infrastructure error after admission drops the connection
  // after draining rather than exposing mixed async/synchronous staging bytes.
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
  if (rdma::IsV2RetireWriter(devbuf)) {
    const uint64_t token = rdma::ParseDevFrameCaps(devbuf);
    std::shared_ptr<WriterState> writer;
    {
      std::lock_guard<std::mutex> lock(writer_mu_);
      const auto found = writers_.find(token);
      if (found != writers_.end()) writer = found->second;
    }
    if (!writer) {
      // Only a token registered by a capability-negotiated bootstrap may
      // operate the retirement control plane. Never manufacture proof for an
      // arbitrary nonzero value.
      ::close(boot_fd);
      return;
    }
    {
      std::unique_lock<std::mutex> lock(writer->mu);
      if (!writer->retired && writer->endpoint)
        writer->endpoint->CancelResponderWrites();
      writer->retired_cv.wait(lock, [&] { return writer->retired; });
    }
    char proof[rdma::kV2RetireProofBytes];
    rdma::EncodeV2RetireProof(token, proof);
    net::WriteAll(boot_fd, proof, sizeof(proof));
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

  // Parse capability and payload geometry from the same raw u64, but never
  // allow the request bit to leak into max-block checks or slot arithmetic.
  const bool writer_retirement_requested =
      rdma::DevFrameRequestsWriterRetirement(devbuf);
  const bool pull_read_requested = rdma::DevFrameRequestsPullRead(devbuf);
  const uint64_t declared = rdma::ParseDevFrameMaxBlock(devbuf);
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
  const auto rail_it =
      std::find(anchor_devs_.begin(), anchor_devs_.end(), dev);
  if (rail_it == anchor_devs_.end()) {
    DFKV_LOG_ERROR("rdma: client requested device outside the fixed topology: " +
                   dev);
    ::close(boot_fd);
    return;
  }
  const size_t rail_index =
      static_cast<size_t>(std::distance(anchor_devs_.begin(), rail_it));
  RailStats& rail_stats = *rail_stats_[rail_index];
  const int rail_numa = anchors_[rail_index]->numa_node();

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
  rdma::RecvSegmentPool::Lease recv_lease = recv_segments_.Allocate(
      K * slot_size, rdma::kV2DataOffset, static_cast<int>(rail_index),
      rail_numa);
  if (!recv_lease) {
    // Segment exhausted: evict the stalest idle connection(s) to make room
    // before refusing. Client pooled connections re-dial via the stale-retry
    // path, so evicting only connections idle longer than kEvictIdleMinUs
    // (no completions for 2 s -> no in-flight request) is safe.
    const uint64_t now = SteadyUs();
    constexpr uint64_t kEvictIdleMinUs = 2000000;
    const uint64_t evict_started = SteadyUs();
    for (int round = 0; round < 32 && !recv_lease; ++round) {
      if (SteadyUs() - evict_started > 5000000) break;  // bound: <= 5 s total
      rdma::RcEndpoint* victim = nullptr;
      uint64_t victim_active = std::numeric_limits<uint64_t>::max();
      {
        std::lock_guard<std::mutex> lk(conn_mu_);
        for (rdma::RcEndpoint* ep : live_eps_) {
          const uint64_t a =
              ep->last_active_us_.load(std::memory_order_relaxed);
          // a == 0: endpoint inserted but Serve has not stamped it yet.
          if (a != 0 && a <= now && now - a >= kEvictIdleMinUs &&
              a < victim_active) {
            victim = ep;
            victim_active = a;
          }
        }
        if (victim) live_eps_.erase(victim);  // claim it: no other evictor
      }                                        // can Wake a freed stack endpoint
      if (!victim) break;  // every connection is recently active; refuse
      victim->Wake();  // Serve exits; the Lease destructor returns its range
      segment_evictions_.fetch_add(1, std::memory_order_relaxed);
      // Poll for the freed range (bounded). The victim's Serve thread tears
      // down its endpoint and releases the lease asynchronously.
      for (int i = 0; i < 100 && !recv_lease; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        recv_lease = recv_segments_.Allocate(
            K * slot_size, rdma::kV2DataOffset,
            static_cast<int>(rail_index), rail_numa);
      }
    }
    if (!recv_lease) {
      const auto stats = recv_segments_.stats();
      DFKV_LOG_ERROR(
          "rdma v2: receive pool exhausted; refusing connection (need=" +
          std::to_string(K * slot_size) +
          " free=" + std::to_string(stats.free_bytes) +
          " committed=" + std::to_string(stats.committed_bytes) +
          " max=" + std::to_string(stats.max_bytes) + ")");
      ::close(boot_fd);
      return;
    }
  }
  rdma::RecvSegmentPool::Lease pull_lease;
  if (pull_read_requested) {
    pull_lease = recv_segments_.Allocate(
        K * slot_size, rdma::kV2DataOffset,
        static_cast<int>(rail_index), rail_numa);
    if (!pull_lease) {
      const auto stats = recv_segments_.stats();
      DFKV_LOG_ERROR(
          "rdma v2: pull-read arena unavailable; refusing negotiated "
          "connection (need=" + std::to_string(K * slot_size) +
          " free=" + std::to_string(stats.free_bytes) +
          " committed=" + std::to_string(stats.committed_bytes) +
          " max=" + std::to_string(stats.max_bytes) + ")");
      ::close(boot_fd);
      return;
    }
  }
  const bool control_connection = conn_max <= rdma::kV2ControlCap;
  const uint64_t connection_bytes =
      static_cast<uint64_t>(recv_lease.size()) + pull_lease.size();
  std::atomic<uint64_t>* const protocol_connections =
      pull_read_requested ? &pull_connections_ : &legacy_connections_;
  std::atomic<uint64_t>* const class_bytes =
      control_connection ? &control_connection_bytes_ : &data_connection_bytes_;
  protocol_connections->fetch_add(1, std::memory_order_relaxed);
  class_bytes->fetch_add(connection_bytes, std::memory_order_relaxed);
  struct ConnectionSegmentAccounting {
    std::atomic<uint64_t>* connections;
    std::atomic<uint64_t>* bytes;
    uint64_t leased_bytes;
    ~ConnectionSegmentAccounting() {
      bytes->fetch_sub(leased_bytes, std::memory_order_relaxed);
      connections->fetch_sub(1, std::memory_order_relaxed);
    }
  } segment_accounting{protocol_connections, class_bytes, connection_bytes};

  // These operation owners deliberately precede the endpoint. On teardown the
  // endpoint (and therefore its QP) is destroyed first, before any prepared
  // source pin, coalescer flight, or shared receive-slot lease is released.
  struct MultiPutState {
    bool active = false;
    BlockKey key;
    uint64_t total_len = 0;
    uint64_t received = 0;
    uint32_t next_window = 0;
    uint32_t window_count = 0;
  };
  struct MultiGetState {
    bool active = false;
    BlockKey key;
    uint64_t total_capacity = 0;
    uint64_t next_offset = 0;
    uint64_t payload_len = 0;
    uint64_t value_len = 0;
    uint32_t next_window = 0;
    uint32_t window_count = 0;
    size_t source_slot = 0;
    size_t last_recv_slot = 0;
    const char* payload = nullptr;
    ibv_mr* payload_mr = nullptr;
    bool source_uses_slot = false;
    double completion_elapsed_sec = 0.0;
    PreparedRead completion;
  };
  struct PendingCompletion {
    PreparedRead read;
    Status status = Status::kOk;
    size_t bytes = 0;
    double elapsed_sec = 0.0;
  };
  struct PullSlotState {
    bool busy = false;
    uint64_t generation = 1;
    size_t data_len = 0;
    size_t value_len = 0;
  };
  std::vector<PullSlotState> pull_slots(K);
  std::vector<MultiPutState> multi_put(K);
  std::vector<MultiGetState> multi_get(K);
  std::vector<int32_t> multi_get_source_owner(K, -1);
  std::vector<PendingCompletion> complete_on_send(K);
  auto clear_multi_get = [&](size_t operation_id) {
    MultiGetState& state = multi_get[operation_id];
    if (state.active && state.source_uses_slot &&
        state.source_slot < multi_get_source_owner.size() &&
        multi_get_source_owner[state.source_slot] ==
            static_cast<int32_t>(operation_id)) {
      multi_get_source_owner[state.source_slot] = -1;
    }
    state = MultiGetState{};
  };

  rdma::RcEndpoint ep;
  constexpr size_t conn_control = rdma::kV2ControlCap;
  if (!ep.Open(dev.empty() ? nullptr : dev.c_str(), conn_control, K,
               /*ib_port=*/1, /*direct_io_buffers=*/false, conn_max,
               /*v2_responder=*/true)) {
    ::close(boot_fd);
    return;
  }
  ibv_mr* recv_segment_mr = ep.RegisterRemoteRegion(
      recv_lease.segment()->data(), recv_lease.segment()->size());
  if (!recv_segment_mr) {
    DFKV_LOG_ERROR("rdma v2: receive-segment MR unavailable on device " +
                   (dev.empty() ? std::string("(auto)") : dev));
    ::close(boot_fd);
    return;
  }
  ibv_mr* pull_pool_mr = nullptr;
  ibv_mr* pull_segment_mr = nullptr;
  uint32_t pull_rkey = 0;
  if (pull_read_requested) {
    pull_pool_mr = ep.RegisterRemoteReadPool(
        pull_lease.segment()->data(), pull_lease.segment()->size());
    if (!pull_pool_mr) {
      pull_segment_mr =
          ep.RegisterRemoteReadRegion(pull_lease.data(), pull_lease.size());
      if (!pull_segment_mr) {
        DFKV_LOG_ERROR(
            "rdma v2: pull-read MR unavailable on device " +
            (dev.empty() ? std::string("(auto)") : dev));
        ::close(boot_fd);
        return;
      }
      pull_rkey = pull_segment_mr->rkey;
      pull_mr_fallbacks_.fetch_add(1, std::memory_order_relaxed);
    }
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
  if (pull_read_requested && pull_pool_mr) {
    if (ep.BindRemoteReadWindow(pull_pool_mr, pull_lease.data(),
                                pull_lease.size(), &pull_rkey)) {
      pull_memory_windows_.fetch_add(1, std::memory_order_relaxed);
    } else {
      pull_segment_mr =
          ep.RegisterRemoteReadRegion(pull_lease.data(), pull_lease.size());
      if (!pull_segment_mr) {
        DFKV_LOG_ERROR(
            "rdma v2: pull-read Memory Window bind and exact MR fallback "
            "both failed on device " +
            (dev.empty() ? std::string("(auto)") : dev));
        ::close(boot_fd);
        return;
      }
      pull_rkey = pull_segment_mr->rkey;
      pull_mr_fallbacks_.fetch_add(1, std::memory_order_relaxed);
    }
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
  std::shared_ptr<WriterState> writer;
  uint64_t writer_token = 0;
  if (writer_retirement_requested) {
    writer = std::make_shared<WriterState>();
    writer->endpoint = &ep;
    writer_token = RegisterWriter(writer);
  }
  auto retire_writer = [&] {
    // An unnegotiated client keeps the legacy teardown path. Only a bootstrap
    // that requested retirement owns a registered token and the explicit
    // responder-WRITE drain/proof lifecycle.
    if (!writer) return;
    const bool retired = ep.RetireResponderWrites();
    if (!retired)
      DFKV_LOG_ERROR("rdma: responder WRITE CQ drain failed");
    {
      std::lock_guard<std::mutex> lock(writer->mu);
      writer->endpoint = nullptr;
      writer->retired = retired;
    }
    writer->retired_cv.notify_all();
    {
      std::lock_guard<std::mutex> lock(writer_mu_);
      writers_.erase(writer_token);
    }
    if (!retired) std::abort();
  };
  // Receives must be posted before readiness becomes visible. Publish the
  // leased receive-segment address, rkey and slot geometry only after the QP is
  // armed, so the client cannot issue a one-sided write into an unready slot.
  const rdma::RecvSegmentInfo info{
      reinterpret_cast<uint64_t>(recv_lease.data()),
      recv_segment_mr->rkey, slot_size};
  char readiness[rdma::kV2PullReadinessBytes];
  size_t readiness_bytes = 0;
  if (pull_read_requested) {
    const rdma::PullArenaInfo pull_info{
        reinterpret_cast<uint64_t>(pull_lease.data()), pull_lease.size(),
        writer_token, pull_rkey, static_cast<uint32_t>(K)};
    readiness_bytes =
        rdma::EncodeV2PullReadiness(info, writer_token, pull_info, readiness);
  } else {
    readiness_bytes =
        rdma::EncodeV2Readiness(info, writer_token, readiness);
  }
  const bool ok =
      readiness_bytes != 0 &&
      net::WriteAll(boot_fd, readiness, readiness_bytes);
  ::close(boot_fd);
  if (!ok) {
    retire_writer();
    return;
  }
  v2_conns_.fetch_add(1, std::memory_order_relaxed);

  // Register this endpoint so Stop() can Wake() us out of WaitComp and join. The
  // running_ check under conn_mu_ closes the race with a concurrent Stop(): either
  // Stop sees us in live_eps_ (and wakes us) or we see running_==false here.
  {
    std::lock_guard<std::mutex> lk(conn_mu_);
    if (!running_) {
      retire_writer();
      return;
    }
    live_eps_.insert(&ep);
  }
  ep.last_active_us_.store(SteadyUs(), std::memory_order_relaxed);

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
    bool multi_put_window = false;
    bool multi_put_final = false;
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
      if (data_slot >= K || multi_get_source_owner[data_slot] >= 0)
        return false;
      const char* frame = recv_lease.data() + data_slot * slot_size +
                          rdma::kV2PutPrefixOffset;
      MultiPutState& state = multi_put[data_slot];
      uint64_t window_bytes = completion.byte_len;
      if (state.active) {
        if (completion.byte_len == 0 || state.next_window >= state.window_count ||
            window_bytes > state.total_len - state.received) {
          return false;
        }
        request->fields.op = static_cast<uint8_t>(WireOp::kCache);
        request->fields.tenant_hash = state.key.tenant_hash;
        request->fields.digest_hi = state.key.digest_hi;
        request->fields.digest_lo = state.key.digest_lo;
        request->fields.offset = rdma::kV2MultiWrPutMagic;
        request->fields.length = state.window_count;
        request->fields.payload_len = state.total_len;
        state.received += window_bytes;
        ++state.next_window;
        request->multi_put_window = true;
        request->multi_put_final =
            state.next_window == state.window_count;
        if (request->multi_put_final !=
            (state.received == state.total_len)) {
          return false;
        }
        if (request->multi_put_final) state = MultiPutState{};
      } else {
        if (completion.byte_len < kReqPrefix ||
            !DecodeReqVersion(
                frame, kNativeProtoRdmaV2, &request->fields,
                static_cast<uint64_t>(logical_data_cap)) ||
            request->fields.op != static_cast<uint8_t>(WireOp::kCache)) {
          return false;
        }
        if (request->fields.offset == rdma::kV2MultiWrPutMagic) {
          if (request->fields.length <= 1 ||
              request->fields.length >
                  std::numeric_limits<uint32_t>::max() ||
              request->fields.payload_len == 0) {
            return false;
          }
          window_bytes -= kReqPrefix;
          if (window_bytes == 0 ||
              window_bytes >= request->fields.payload_len) {
            return false;
          }
          state.active = true;
          state.key = request->fields.Key();
          state.total_len = request->fields.payload_len;
          state.received = window_bytes;
          state.next_window = 1;
          state.window_count =
              static_cast<uint32_t>(request->fields.length);
          request->multi_put_window = true;
        } else if (!rdma::V2PutCompletionIsValid(
                       write_imm_recv, has_immediate,
                       completion.byte_len, request->fields.payload_len,
                       logical_data_cap, slot_size)) {
          return false;
        } else {
          window_bytes = request->fields.payload_len;
        }
      }
      request->data_slot = data_slot;
      request->contiguous_payload = frame + kReqPrefix;
      v2_put_writes_.fetch_add(1, std::memory_order_relaxed);
      rail_stats.put_writes.fetch_add(1, std::memory_order_relaxed);
      rail_stats.put_bytes.fetch_add(window_bytes,
                                     std::memory_order_relaxed);
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
                            static_cast<uint64_t>(conn_max)) ||
          !rdma::V2GetOperationIdValid(request->get.operation_id, K)) {
        return false;
      }
      MultiGetState& state = multi_get[request->get.operation_id];
      if (request->get.window_index == 0 &&
          (multi_put[recv_slot].active ||
           multi_get_source_owner[recv_slot] >= 0 || state.active)) {
        return false;
      }
      return request->get.targets.size() <= rdma::kV2MaxGetTargets;
    }
    if (request->fields.op == static_cast<uint8_t>(WireOp::kPullRange)) {
      if (request->fields.payload_len != rdma::kPullPrepareBytes ||
          completion.byte_len != kReqPrefix + rdma::kPullPrepareBytes)
        return false;
      request->contiguous_payload = frame + kReqPrefix;
      return true;
    }
    if (request->fields.op == static_cast<uint8_t>(WireOp::kPullRelease)) {
      return completion.byte_len == kReqPrefix;
    }


    if (multi_put[recv_slot].active) return false;

    // Other control ops (Exist/Remove/Members/Lookup) with inline payload
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
    bool release_source_on_send = false;
    size_t recv_slot = 0;
    size_t source_recv_slot = 0;
    size_t first_len = 0;
    const char* payload = nullptr;
    size_t payload_len = 0;
    ibv_mr* payload_mr = nullptr;
    std::vector<RdmaWriteTarget> targets;
    PreparedRead completion;
    double completion_elapsed_sec = 0.0;
  };

  auto build_data_reply =
      [&](size_t send_slot, const Request& request, Status status,
          const char* data, size_t data_len, size_t value_len, ibv_mr* data_mr,
          bool source_uses_slot, PreparedRead completion,
          double completion_elapsed_sec, Reply* reply) -> bool {
    auto encode_status = [&](Status response_status, uint64_t response_len,
                             uint64_t response_value_len = 0) {
      EncodeRespVersion(ep.sbuf(send_slot), wire_epoch, response_status,
                        response_len, response_value_len);
    };
    auto invalid_reply = [&] {
      encode_status(Status::kInvalid, 0);
      reply->first_len = response_prefix;
      return true;
    };
    const size_t successful_len = status == Status::kOk ? data_len : 0;
    if (status != Status::kOk) {
      encode_status(status, 0);
      reply->first_len = response_prefix;
      return true;
    }
    if (successful_len > request.get.total_capacity ||
        value_len > request.get.total_capacity ||
        (request.get.window_count > 1 && successful_len != value_len) ||
        (successful_len != 0 && (!data || !data_mr))) {
      return invalid_reply();
    }
    encode_status(Status::kOk, successful_len, value_len);
    reply->first_len = response_prefix;
    if (request.get.window_count == 1) {
      if (successful_len != 0) {
        reply->remote_write = true;
        reply->defer_recv_rearm = source_uses_slot;
        reply->recv_slot = request.recv_slot;
        reply->payload = data;
        reply->payload_len = successful_len;
        reply->payload_mr = data_mr;
        reply->targets = request.get.targets;
        reply->completion = std::move(completion);
        reply->completion_elapsed_sec = completion_elapsed_sec;
      } else {
        completion.Commit(Status::kOk, 0, completion_elapsed_sec);
      }
      return true;
    }

    const size_t operation_id = request.get.operation_id;
    MultiGetState& state = multi_get[operation_id];
    uint64_t window_capacity = 0;
    if (request.get.window_index != 0 ||
        request.get.logical_offset != 0 || state.active ||
        multi_put[request.data_slot].active ||
        multi_get_source_owner[request.data_slot] >= 0 ||
        !request.get.Capacity(&window_capacity) ||
        window_capacity == 0 ||
        window_capacity >= request.get.total_capacity) {
      return invalid_reply();
    }
    state.active = true;
    state.key = request.fields.Key();
    state.total_capacity = request.get.total_capacity;
    state.next_offset = window_capacity;
    state.payload_len = successful_len;
    state.last_recv_slot = request.recv_slot;
    state.value_len = value_len;
    state.next_window = 1;
    state.window_count = request.get.window_count;
    state.source_slot = request.data_slot;
    state.payload = data;
    state.payload_mr = data_mr;
    state.source_uses_slot = source_uses_slot;
    state.completion_elapsed_sec = completion_elapsed_sec;
    state.completion = std::move(completion);
    if (source_uses_slot) {
      multi_get_source_owner[request.data_slot] =
          static_cast<int32_t>(operation_id);
      // Repost this WQE only after the first write finishes, but keep the
      // source owner until the logical GET's final SEND completion.
      reply->defer_recv_rearm = true;
    }
    reply->recv_slot = request.recv_slot;
    const size_t bytes =
        std::min<size_t>(successful_len, static_cast<size_t>(window_capacity));
    if (bytes != 0) {
      reply->remote_write = true;
      reply->payload = data;
      reply->payload_len = bytes;
      reply->payload_mr = data_mr;
      reply->targets = request.get.targets;
    }
    return true;
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
    if (fields.op == static_cast<uint8_t>(WireOp::kPullRelease)) {
      if (!pull_read_requested || fields.payload_len == 0 ||
          fields.payload_len > K)
        return invalid_reply();
      const size_t slot = static_cast<size_t>(fields.payload_len - 1);
      PullSlotState& state = pull_slots[slot];
      if (!state.busy || state.generation != fields.length)
        return invalid_reply();
      state.busy = false;
      state.data_len = 0;
      state.value_len = 0;
      ++state.generation;
      if (state.generation == 0) ++state.generation;
      encode_status(Status::kOk, 0);
      reply->first_len = response_prefix;
      return true;
    }

    if (fields.op == static_cast<uint8_t>(WireOp::kPullRange)) {
      if (!pull_read_requested || !range_handler_ ||
          fields.payload_len != rdma::kPullPrepareBytes ||
          fields.length > static_cast<uint64_t>(conn_max) ||
          request.contiguous_payload == nullptr)
        return invalid_reply();
      rdma::PullPrepareControl control;
      if (!rdma::DecodePullPrepareControl(request.contiguous_payload,
                                          &control) ||
          control.slot_index >= K)
        return invalid_reply();
      const size_t slot = control.slot_index;
      PullSlotState& state = pull_slots[slot];
      if (control.release_generation != 0) {
        if (!state.busy ||
            state.generation != control.release_generation)
          return invalid_reply();
        state.busy = false;
        state.data_len = 0;
        state.value_len = 0;
        ++state.generation;
        if (state.generation == 0) ++state.generation;
      }
      if (state.busy) {
        encode_status(Status::kCacheFull, 0);
        reply->first_len = response_prefix;
        return true;
      }
      char* target = pull_lease.data() + slot * slot_size;
      const char* output = nullptr;
      size_t output_len = 0;
      size_t value_len = 0;
      const Status status =
          range_handler_(key, fields.offset, fields.length, target, slot_size,
                         &output, &output_len, &value_len);
      if (status != Status::kOk) {
        encode_status(status, 0, value_len);
        reply->first_len = response_prefix;
        return true;
      }
      if (output_len > slot_size || (output_len != 0 && output == nullptr))
        return invalid_reply();
      if (output_len != 0 && output != target)
        std::memcpy(target, output, output_len);
      state.busy = true;
      state.data_len = output_len;
      state.value_len = value_len;
      const rdma::PullReady ready{static_cast<uint32_t>(slot),
                                  state.generation, output_len, value_len};
      encode_status(Status::kOk, rdma::kPullReadyBytes, value_len);
      rdma::EncodePullReady(ready, send_buffer + response_prefix);
      reply->first_len = response_prefix + rdma::kPullReadyBytes;
      return true;
    }

    if (fields.op == static_cast<uint8_t>(WireOp::kRange) &&
        request.get.window_index != 0) {
      const size_t operation_id = request.get.operation_id;
      MultiGetState& state = multi_get[operation_id];
      uint64_t window_capacity = 0;
      if (!request.get.Capacity(&window_capacity) || !state.active ||
          !(state.key == key) ||
          request.get.window_count != state.window_count ||
          request.get.window_index != state.next_window ||
          request.get.total_capacity != state.total_capacity ||
          request.get.logical_offset != state.next_offset ||
          window_capacity == 0 ||
          window_capacity > state.total_capacity - state.next_offset ||
          ((state.next_window + 1 == state.window_count) !=
           (window_capacity == state.total_capacity - state.next_offset))) {
        clear_multi_get(operation_id);
        return false;
      }
      if (request.recv_slot != state.last_recv_slot) {
        v2_get_continuation_slot_changes_.fetch_add(
            1, std::memory_order_relaxed);
      }
      state.last_recv_slot = request.recv_slot;
      encode_status(Status::kOk, state.payload_len, state.value_len);
      reply->first_len = response_prefix;
      const uint64_t remaining =
          state.payload_len > state.next_offset
              ? state.payload_len - state.next_offset
              : 0;
      const size_t bytes = static_cast<size_t>(
          std::min<uint64_t>(remaining, window_capacity));
      if (bytes != 0) {
        reply->remote_write = true;
        reply->payload = state.payload + state.next_offset;
        reply->payload_len = bytes;
        reply->payload_mr = state.payload_mr;
        reply->targets = request.get.targets;
      }
      state.next_offset += window_capacity;
      ++state.next_window;
      if (state.next_window == state.window_count) {
        reply->completion = std::move(state.completion);
        reply->completion_elapsed_sec = state.completion_elapsed_sec;
        reply->release_source_on_send = state.source_uses_slot;
        reply->source_recv_slot = state.source_slot;
        if (state.source_uses_slot &&
            request.recv_slot == state.source_slot) {
          reply->defer_recv_rearm = true;
          reply->recv_slot = request.recv_slot;
        }
        // The source remains protected until this final RDMA WRITE's SEND
        // completion. clear_multi_get must not release its owner early.
        state.source_uses_slot = false;
        clear_multi_get(operation_id);
      }
      return true;
    }


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
            bool ok =
                got >= 0 &&
                static_cast<size_t>(got) >=
                    prepared.head() + prepared.payload_len();
            const size_t payload_len = prepared.payload_len();
            const size_t value_len = prepared.value_len();
            ibv_mr* source_mr = direct_mr(request.data_slot);
            bool source_uses_slot = true;
            if (ok && prepared.source_registered() && payload_len != 0) {
              source_mr = ep.RegisterUser(
                  const_cast<char*>(prepared.data()), payload_len);
              if (!source_mr && prepared.Stage())
                source_mr = direct_mr(request.data_slot);
              if (!source_mr) ok = false;
              source_uses_slot = !prepared.source_registered();
            }
            const char* data = prepared.data();
            const double elapsed_sec = NowSteadySec() - submit_sec;
            // A successful multi-window read remains one live transaction
            // through every continuation and the final signaled SEND. Errors
            // and the established single-window path complete immediately.
            if (!ok || request.get.window_count == 1) {
              prepared.Commit(ok ? Status::kOk : Status::kIOError,
                              ok ? payload_len : 0, elapsed_sec);
            }
            return build_data_reply(
                send_slot, request,
                ok ? Status::kOk : Status::kIOError,
                ok ? data : nullptr, payload_len, value_len, source_mr,
                source_uses_slot, std::move(prepared), elapsed_sec, reply);
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
          return build_data_reply(
              send_slot, request, Status::kOk, data, payload_len,
              prepared.value_len(), source_mr,
              /*source_uses_slot=*/!prepared.source_registered(),
              std::move(prepared), /*completion_elapsed_sec=*/0.0, reply);
        }
        if (prepared.status() != Status::kInvalid) {
          return build_data_reply(
              send_slot, request, prepared.status(), nullptr, 0, 0, nullptr,
              /*source_uses_slot=*/false, PreparedRead{},
              /*completion_elapsed_sec=*/0.0, reply);
        }
      }

      const char* output = nullptr;
      size_t output_len = 0;
      size_t value_len = 0;
      const Status status = range_handler_(
          key, fields.offset, fields.length, direct_buffer(request.data_slot),
          direct_buffer_cap, &output, &output_len, &value_len);
      return build_data_reply(
          send_slot, request, status, output, output_len, value_len,
          direct_mr(request.data_slot),
          /*source_uses_slot=*/true, PreparedRead{},
          /*completion_elapsed_sec=*/0.0, reply);
    }

    if (fields.op == static_cast<uint8_t>(WireOp::kCache) &&
        request.multi_put_window && !request.multi_put_final) {
      encode_status(Status::kOk, 0);
      reply->first_len = response_prefix;
      return true;
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
      return build_data_reply(
          send_slot, request, status, direct_buffer(request.data_slot),
          data.size(), value_len, direct_mr(request.data_slot),
          /*source_uses_slot=*/true, PreparedRead{},
          /*completion_elapsed_sec=*/0.0, reply);
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
    rail_stats.get_writes.fetch_add(1, std::memory_order_relaxed);
    rail_stats.get_bytes.fetch_add(reply.payload_len,
                                   std::memory_order_relaxed);
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
  std::vector<size_t> release_source_on_send(K, kNoSlot);
  auto rearm_request_recv = [&](size_t slot) {
    return post_request_recv(slot);
  };
  // Prepared reads remain in their move-only transaction until the signaled
  // SEND completion. A completion can be the first and only terminal call for
  // a multi-window disk read, so resetting the slot after Commit also releases
  // any post-read source hold at that exact fence.
  auto complete_send = [&](size_t sid) {
    if (sid >= complete_on_send.size()) return;
    PendingCompletion& pending = complete_on_send[sid];
    pending.read.Commit(pending.status, pending.bytes, pending.elapsed_sec);
    pending = PendingCompletion{};
  };
  bool fail = false;
  const int idle_ms = ServerIdleMs();
  active_conns_.fetch_add(1, std::memory_order_relaxed);
  rail_stats.active_conns.fetch_add(1, std::memory_order_relaxed);

#ifdef DFKV_WITH_URING
  // -------------------------------------------------------------------------
  // Nonblocking disk pipeline. RDMA receives, disk CQEs, and SEND completions
  // are advanced independently. Replies alone are serialized by request
  // sequence because the client correlates them by RC SEND order.
  if (UseUringPath()) {
    const size_t uring_depth = UringDepth();
    enum class DiskState : uint8_t {
      kNone,
      kWaiting,
      kInflight,
      kComplete,
    };
    struct Queued {
      uint64_t sequence = 0;
      size_t send_slot = 0;
      size_t recv_slot = 0;
      size_t data_slot = 0;
      Request request;
      DiskState disk_state = DiskState::kNone;
      UringReader::ReadDesc desc;
      UringReader::Token token = UringReader::kInvalidToken;
      long read_result = 0;
      PreparedRead read;
      double submit_sec = 0.0;
      bool ready = false;
      Reply reply;
    };

    // queue owns every PreparedRead and therefore precedes ring: reverse
    // destruction drains/exits the ring before any descriptor owner is freed.
    std::deque<Queued> queue;
    std::vector<UringReader::ReadDesc> submit_descs;
    std::vector<UringReader::Token> submit_tokens;
    std::vector<Queued*> submit_owners;
    UringReader::Backend* uring_backend =
        uring_backend_factory_for_test_
            ? uring_backend_factory_for_test_()
            : nullptr;
    UringReader ring(static_cast<unsigned>(uring_depth), uring_backend);
    if (!ring.ok()) {
      // This is the only synchronous fallback. No async SQE or payload has been
      // exposed, so the original loop remains safe for this connection.
      uring_init_fallbacks_.fetch_add(1, std::memory_order_relaxed);
      DFKV_LOG_WARN("io_uring ring init failed (depth=" +
                    std::to_string(uring_depth) +
                    "); this connection serves on the SYNC path");
      goto sync_serve_loop;
    }
    submit_descs.reserve(uring_depth);
    submit_tokens.resize(uring_depth);
    submit_owners.reserve(uring_depth);

    uint64_t next_sequence = 0;
    uint64_t next_emit_sequence = 0;
    uint64_t metric_inflight = 0;
    // Never fill every logical reply slot in one posting burst. Although the
    // QP requests worst-case WR capacity, real providers can stop accepting the
    // final status SEND after a full window of preceding RDMA WRITEs. Leaving
    // one reply chain of headroom also forces SEND CQ progress and receive
    // rearming before the next client window. Depth one remains functional.
    const size_t send_post_limit = K > 1 ? K - 1 : 1;
    size_t posted_sends = 0;

    auto update_inflight_max = [&](uint64_t current) {
      uint64_t previous =
          uring_inflight_max_.load(std::memory_order_relaxed);
      while (current > previous &&
             !uring_inflight_max_.compare_exchange_weak(
                 previous, current, std::memory_order_relaxed,
                 std::memory_order_relaxed)) {
      }
    };

    auto finish_disk_read = [&](Queued& qd) -> bool {
      const bool read_ok =
          qd.read_result >= 0 &&
          static_cast<size_t>(qd.read_result) >=
              qd.read.head() + qd.read.payload_len();
      const size_t payload_len = qd.read.payload_len();
      const size_t value_len = qd.read.value_len();
      ibv_mr* source_mr = direct_mr(qd.data_slot);
      bool source_uses_slot = true;
      bool ok = read_ok;
      if (ok && qd.read.source_registered() && payload_len != 0) {
        source_mr = ep.RegisterUser(
            const_cast<char*>(qd.read.data()), payload_len);
        if (!source_mr && qd.read.Stage())
          source_mr = direct_mr(qd.data_slot);
        if (!source_mr) ok = false;
        source_uses_slot = !qd.read.source_registered();
      }
      const char* out_data = qd.read.data();
      const double elapsed_sec = NowSteadySec() - qd.submit_sec;
      // Multi-window completion stays live through its final continuation SEND.
      if (!ok || qd.request.get.window_count == 1) {
        qd.read.Commit(ok ? Status::kOk : Status::kIOError,
                       ok ? payload_len : 0, elapsed_sec);
      }
      if (!build_data_reply(
              qd.send_slot, qd.request,
              ok ? Status::kOk : Status::kIOError,
              ok ? out_data : nullptr, payload_len, value_len, source_mr,
              source_uses_slot, std::move(qd.read), elapsed_sec, &qd.reply))
        return false;
      qd.disk_state = DiskState::kComplete;
      qd.ready = true;
      return true;
    };

    auto process_wc = [&](const ibv_wc& wc) -> bool {
      if (wc.status != IBV_WC_SUCCESS) {
        completion_errors_.fetch_add(1, std::memory_order_relaxed);
        rail_stats.completion_errors.fetch_add(1,
                                                std::memory_order_relaxed);
        return false;
      }
      if (wc.opcode == IBV_WC_RDMA_WRITE) return true;
      if (wc.opcode == IBV_WC_SEND) {
        const size_t sid = static_cast<size_t>(wc.wr_id);
        if (sid >= K) return false;
        if (posted_sends == 0) return false;
        --posted_sends;
        uring_send_fences_.fetch_add(1, std::memory_order_relaxed);
        // Source ownership is released only at this signaled SEND fence.
        complete_send(sid);
        if (release_source_on_send[sid] != kNoSlot) {
          multi_get_source_owner[release_source_on_send[sid]] = -1;
          release_source_on_send[sid] = kNoSlot;
        }
        if (rearm_on_send[sid] != kNoSlot) {
          if (!rearm_request_recv(rearm_on_send[sid])) return false;
          rearm_on_send[sid] = kNoSlot;
        }
        free_send.push_back(sid);
        return true;
      }

      Request request;
      if (!decode_request(wc, &request) || free_send.empty() ||
          next_sequence == std::numeric_limits<uint64_t>::max())
        return false;
      completions_.fetch_add(1, std::memory_order_relaxed);
      rail_stats.completions.fetch_add(1, std::memory_order_relaxed);

      Queued qd;
      qd.sequence = next_sequence++;
      qd.send_slot = free_send.back();
      free_send.pop_back();
      qd.recv_slot = request.recv_slot;
      qd.data_slot = request.data_slot;
      const ReqFields& fields = request.fields;

      bool deferred = false;
      bool handled = false;
      if (fields.op == static_cast<uint8_t>(WireOp::kRange) &&
          request.get.window_index == 0 &&
          direct_buffer(request.data_slot) &&
          direct_mr(request.data_slot) &&
          fields.length <= static_cast<uint64_t>(conn_max)) {
        PreparedRead prepared = prepare_read_handler_(
            fields.Key(), fields.offset, fields.length,
            direct_buffer(request.data_slot), direct_buffer_cap);
        if (prepared.status() == Status::kOk && prepared.needs_io() &&
            prepared.fd() >= 0 && prepared.payload_len() != 0 &&
            prepared.aligned_len() <= direct_buffer_cap &&
            prepared.aligned_len() <=
                std::numeric_limits<unsigned>::max()) {
          qd.desc.fd = prepared.fd();
          qd.desc.buf = prepared.staging();
          qd.desc.len = static_cast<unsigned>(prepared.aligned_len());
          qd.desc.off = prepared.aligned_off();
          qd.disk_state = DiskState::kWaiting;
          qd.read = std::move(prepared);
          qd.request = std::move(request);
          qd.submit_sec = NowSteadySec();
          deferred = true;
          handled = true;
        } else if (prepared.status() == Status::kOk &&
                   !prepared.needs_io()) {
          const size_t payload_len = prepared.payload_len();
          ibv_mr* source_mr = direct_mr(request.data_slot);
          if (prepared.source_registered() && payload_len != 0) {
            source_mr = ep.RegisterUser(
                const_cast<char*>(prepared.data()), payload_len);
            if (!source_mr && prepared.Stage())
              source_mr = direct_mr(request.data_slot);
          }
          if (!source_mr ||
              !build_data_reply(
                  qd.send_slot, request, Status::kOk, prepared.data(),
                  payload_len, prepared.value_len(), source_mr,
                  /*source_uses_slot=*/!prepared.source_registered(),
                  std::move(prepared), /*completion_elapsed_sec=*/0.0,
                  &qd.reply))
            return false;
          handled = true;
          qd.ready = true;
        } else if (prepared.status() != Status::kOk &&
                   prepared.status() != Status::kInvalid) {
          if (!build_data_reply(
                  qd.send_slot, request, prepared.status(), nullptr, 0, 0,
                  nullptr, /*source_uses_slot=*/false, PreparedRead{},
                  /*completion_elapsed_sec=*/0.0, &qd.reply))
            return false;
          handled = true;
          qd.ready = true;
        }
      }
      if (!deferred && !handled) {
        // Unsupported shapes and coalescer followers retain established sync
        // semantics, but still wait behind earlier sequence numbers to send.
        if (!build_reply(qd.send_slot, request, &qd.reply,
                         /*try_prepare=*/false))
          return false;
        qd.ready = true;
      }
      queue.push_back(std::move(qd));
      return true;
    };

    auto submit_waiting = [&]() -> bool {
      const size_t count_limit = ring.capacity();
      if (count_limit == 0) return true;
      submit_descs.clear();
      submit_owners.clear();
      for (Queued& qd : queue) {
        if (qd.disk_state != DiskState::kWaiting) continue;
        submit_descs.push_back(qd.desc);
        submit_owners.push_back(&qd);
        if (submit_descs.size() == count_limit) break;
      }
      if (submit_descs.empty()) return true;
      if (!ring.Submit(submit_descs.data(), submit_descs.size(),
                       submit_tokens.data()))
        return false;

      const uint64_t count = static_cast<uint64_t>(submit_descs.size());
      uring_reads_.fetch_add(count, std::memory_order_relaxed);
      uring_read_batches_.fetch_add(1, std::memory_order_relaxed);
      uint64_t previous =
          uring_read_batch_max_.load(std::memory_order_relaxed);
      while (count > previous &&
             !uring_read_batch_max_.compare_exchange_weak(
                 previous, count, std::memory_order_relaxed,
                 std::memory_order_relaxed)) {
      }
      const uint64_t current =
          uring_inflight_.fetch_add(count, std::memory_order_relaxed) + count;
      metric_inflight += count;
      update_inflight_max(current);
      for (size_t i = 0; i < submit_owners.size(); ++i) {
        submit_owners[i]->token = submit_tokens[i];
        submit_owners[i]->disk_state = DiskState::kInflight;
      }
      return true;
    };

    auto reap_event = [&](UringReader::Event* event) -> int {
      UringReader::Completion completion;
      const int reaped = ring.Reap(event, &completion);
      if (reaped <= 0) return reaped;
      Queued* qd = nullptr;
      for (Queued& candidate : queue) {
        if (candidate.disk_state == DiskState::kInflight &&
            candidate.token == completion.token) {
          qd = &candidate;
          break;
        }
      }
      if (!qd) return -1;
      qd->read_result = completion.result;
      uring_completions_.fetch_add(1, std::memory_order_relaxed);
      uring_inflight_.fetch_sub(1, std::memory_order_relaxed);
      --metric_inflight;
      return finish_disk_read(*qd) ? 1 : -1;
    };

    auto emit_ready = [&]() -> bool {
      while (!queue.empty() && queue.front().ready &&
             posted_sends < send_post_limit) {
        Queued& qd = queue.front();
        if (qd.sequence != next_emit_sequence ||
            next_emit_sequence == std::numeric_limits<uint64_t>::max())
          return false;
        Reply& reply = qd.reply;
        if (reply.defer_recv_rearm) {
          rearm_on_send[qd.send_slot] = reply.recv_slot;
        } else if (!rearm_request_recv(qd.recv_slot)) {
          return false;
        }
        if (reply.release_source_on_send) {
          if (release_source_on_send[qd.send_slot] != kNoSlot) return false;
          release_source_on_send[qd.send_slot] = reply.source_recv_slot;
        }
        PendingCompletion& pending = complete_on_send[qd.send_slot];
        pending.read = std::move(reply.completion);
        pending.bytes = pending.read.payload_len();
        pending.elapsed_sec = reply.completion_elapsed_sec;
        if (!post_reply(qd.send_slot, reply)) {
          uring_send_post_errors_.fetch_add(1, std::memory_order_relaxed);
          return false;
        }
        uring_replies_posted_.fetch_add(1, std::memory_order_relaxed);
        ++posted_sends;
        ++next_emit_sequence;
        queue.pop_front();
      }
      return true;
    };

    while (running_ && !fail) {
      bool progressed = false;

      // Always give RDMA ingress first opportunity. This keeps first-window
      // receives and SEND completions moving while disk reads are outstanding.
      int g = ep.PollComp(wcs.data(), static_cast<int>(K));
      if (g < 0) {
        fail = true;
        break;
      }
      if (g > 0) {
        ep.last_active_us_.store(SteadyUs(), std::memory_order_relaxed);
        progressed = true;
        for (int w = 0; w < g; ++w) {
          if (!process_wc(wcs[w])) {
            fail = true;
            break;
          }
        }
      }
      if (fail) break;

      if (!submit_waiting()) {
        fail = true;
        break;
      }

      // Drain every ready disk CQE without blocking. Out-of-order completions
      // only mark their own queue entry ready; emit_ready gates the prefix.
      for (;;) {
        UringReader::Event event;
        const int ready = ring.Peek(&event);
        if (ready < 0) {
          fail = true;
          break;
        }
        if (ready == 0) break;
        progressed = true;
        if (reap_event(&event) < 0) {
          fail = true;
          break;
        }
      }
      if (fail || !emit_ready()) {
        fail = true;
        break;
      }
      if (progressed) continue;

      if (ring.inflight() != 0) {
        // liburing and verbs expose separate wait sources. A one-millisecond
        // bounded ring wait avoids spinning without starving the verbs CQ or
        // Stop()'s Wake notification.
        UringReader::Event event;
        const int ready = ring.Wait(&event, /*timeout_ms=*/1);
        if (ready < 0 ||
            (ready > 0 && reap_event(&event) < 0)) {
          fail = true;
          break;
        }
        if (ready > 0 && !emit_ready()) {
          fail = true;
          break;
        }
        continue;
      }

      // SEND completions release PreparedRead/source ownership and rearm the
      // receive window. Some providers can lose a completion-channel edge
      // after our ready-only PollComp drain, so never put an outstanding SEND
      // fence behind the multi-minute connection-idle wait. A bounded wait
      // blocks (no spin) and its timeout path performs a final CQ poll.
      const int verbs_wait_ms = posted_sends != 0 ? 1 : idle_ms;
      g = ep.WaitComp(wcs.data(), static_cast<int>(K), verbs_wait_ms);
      if (g == 0) {
        if (posted_sends != 0) continue;
        idle_reclaims_.fetch_add(1, std::memory_order_relaxed);
        break;
      }
      if (g < 0) break;  // disconnect, endpoint error, or Stop()'s Wake
      ep.last_active_us_.store(SteadyUs(), std::memory_order_relaxed);
      for (int w = 0; w < g; ++w) {
        if (!process_wc(wcs[w])) {
          fail = true;
          break;
        }
      }
    }

    // No synchronous retry is allowed after successful ring initialization:
    // an accepted SQE may have modified staging. Drain kernel ownership while
    // queue/PreparedRead and the endpoint remain alive, then abort the queue.
    if (ring.inflight() != 0 || ring.poisoned())
      ring.Drain();
    if (metric_inflight != 0)
      uring_inflight_.fetch_sub(metric_inflight, std::memory_order_relaxed);

    rail_stats.active_conns.fetch_sub(1, std::memory_order_relaxed);
    active_conns_.fetch_sub(1, std::memory_order_relaxed);
    { std::lock_guard<std::mutex> lk(conn_mu_); live_eps_.erase(&ep); }
    retire_writer();
    return;
  }
sync_serve_loop:;
#endif  // DFKV_WITH_URING

  while (running_ && !fail) {
    int g = ep.WaitComp(wcs.data(), static_cast<int>(K), idle_ms);
    if (g > 0)
      ep.last_active_us_.store(SteadyUs(), std::memory_order_relaxed);
    if (g == 0) { idle_reclaims_.fetch_add(1, std::memory_order_relaxed); break; }  // idle -> reclaim
    if (g < 0) break;  // error / Stop()'s Wake()
    for (int w = 0; w < g && !fail; ++w) {
      const ibv_wc& wc = wcs[w];
      if (wc.status != IBV_WC_SUCCESS) {
        completion_errors_.fetch_add(1, std::memory_order_relaxed);
        rail_stats.completion_errors.fetch_add(1,
                                               std::memory_order_relaxed);
        fail = true; break;
      }
      if (wc.opcode == IBV_WC_RDMA_WRITE) continue;
      if (wc.opcode == IBV_WC_SEND) {
        size_t sid = static_cast<size_t>(wc.wr_id);
        // Finish any deferred read while its source slot is still owned and
        // before reposting a receive that could overwrite completion data.
        complete_send(sid);
        if (sid < release_source_on_send.size() &&
            release_source_on_send[sid] != kNoSlot) {
          multi_get_source_owner[release_source_on_send[sid]] = -1;
          release_source_on_send[sid] = kNoSlot;
        }
        if (sid < rearm_on_send.size() && rearm_on_send[sid] != kNoSlot) {
          if (!rearm_request_recv(rearm_on_send[sid])) {
            fail = true;
            break;
          }
          rearm_on_send[sid] = kNoSlot;
        }
        free_send.push_back(sid);
        continue;
      }
      Request request;
      if (!decode_request(wc, &request)) { fail = true; break; }
      completions_.fetch_add(1, std::memory_order_relaxed);
      rail_stats.completions.fetch_add(1, std::memory_order_relaxed);
      const size_t r = request.recv_slot;
      if (free_send.empty()) { fail = true; break; }
      size_t s = free_send.back(); free_send.pop_back();
      Reply reply;
      bool built = build_reply(s, request, &reply, /*try_prepare=*/true);
      if (!built) { fail = true; break; }
      if (reply.defer_recv_rearm) {
        rearm_on_send[s] = reply.recv_slot;
      } else if (!rearm_request_recv(r)) {
        fail = true;
        break;  // re-arm (request consumed)
      }
      if (reply.release_source_on_send) {
        if (release_source_on_send[s] != kNoSlot) {
          fail = true;
          break;
        }
        release_source_on_send[s] = reply.source_recv_slot;
      }
      PendingCompletion& pending = complete_on_send[s];
      pending.read = std::move(reply.completion);
      pending.bytes = pending.read.payload_len();
      pending.elapsed_sec = reply.completion_elapsed_sec;
      bool sent = post_reply(s, reply);
      if (!sent) { fail = true; break; }
    }
  }
  // Any prepared sends without completions destructor-abort below.
  rail_stats.active_conns.fetch_sub(1, std::memory_order_relaxed);
  active_conns_.fetch_sub(1, std::memory_order_relaxed);
  { std::lock_guard<std::mutex> lk(conn_mu_); live_eps_.erase(&ep); }
  retire_writer();
  // Writer retirement proof, not ep destruction, fences client destinations.
  // The endpoint destructor below only releases verbs resources.
}


std::string RdmaServer::MetricsText() const {
  auto m = [](std::string& s, const char* name, const char* type, const char* help,
              uint64_t v) {
    s += "# HELP "; s += name; s += " "; s += help; s += "\n";
    s += "# TYPE "; s += name; s += " "; s += type; s += "\n";
    s += name; s += " "; s += std::to_string(v); s += "\n";
  };
  std::string s;
  m(s, "dfkv_rdma_completions_total", "counter",
    "RDMA request completions served", Completions());
  m(s, "dfkv_rdma_completion_errors_total", "counter",
    "RDMA error completions", CompletionErrors());
  m(s, "dfkv_rdma_active_conns", "gauge",
    "RDMA connections currently serving", ActiveConns());
  m(s, "dfkv_rdma_rails_initialized", "gauge",
    "RDMA rails with successfully initialized server anchors",
    InitializedRailCount());
  m(s, "dfkv_rdma_v2_conns_opened_total", "counter",
    "RDMA v2 connections opened", V2Conns());
  m(s, "dfkv_rdma_v2_put_writes_total", "counter",
    "PUT requests received by RDMA WRITE_WITH_IMM", V2PutWrites());
  m(s, "dfkv_rdma_v2_get_writes_total", "counter",
    "GET payloads sent by RDMA WRITE", V2GetWrites());
  m(s, "dfkv_rdma_v2_get_continuation_slot_changes_total", "counter",
    "Multi-window GET continuations received on a different WQE slot",
    V2GetContinuationSlotChanges());
  const rdma::RecvSegmentPool::Stats segment = recv_segments_.stats();
  m(s, "dfkv_rdma_recv_segment_bytes", "gauge",
    "Receive-pool bytes currently committed", segment.committed_bytes);
  m(s, "dfkv_rdma_recv_segment_max_bytes", "gauge",
    "Hard receive-pool commit budget", segment.max_bytes);
  m(s, "dfkv_rdma_recv_segment_chunks", "gauge",
    "Receive-pool chunks currently committed", segment.chunks);
  m(s, "dfkv_rdma_recv_segment_used_bytes", "gauge",
    "Bytes leased from committed receive chunks", segment.used_bytes);
  m(s, "dfkv_rdma_recv_segment_free_bytes", "gauge",
    "Unleased bytes in committed receive chunks", segment.free_bytes);
  m(s, "dfkv_rdma_recv_segment_largest_free_range_bytes", "gauge",
    "Largest contiguous unleased range in any receive chunk",
    segment.largest_free_range);
  m(s, "dfkv_rdma_recv_segment_growths_total", "counter",
    "Receive chunks committed after startup", segment.growths);
  m(s, "dfkv_rdma_recv_segment_shrinks_total", "counter",
    "Receive-pool trim passes that released empty non-initial chunks",
    segment.shrinks);
  m(s, "dfkv_rdma_recv_segment_released_bytes_total", "counter",
    "Receive-pool bytes returned after the idle hold period",
    segment.released_bytes);
  m(s, "dfkv_rdma_recv_segment_chunk_idle_ms", "gauge",
    "Idle hold before an empty non-initial receive chunk is released",
    recv_chunk_idle_ms_);
  m(s, "dfkv_rdma_recv_segment_growth_failures_total", "counter",
    "Receive-pool growth attempts rejected by budget or allocation",
    segment.growth_failures);
  m(s, "dfkv_rdma_recv_segment_allocation_failures_total", "counter",
    "Receive-pool allocations that remained unsatisfied after growth",
    segment.allocation_failures);
  m(s, "dfkv_rdma_recv_segment_registered_rails", "gauge",
    "RDMA rails with the initial receive chunk registered",
    recv_segment_registered_rails_);
  m(s, "dfkv_rdma_pull_connections", "gauge",
    "Connections currently holding negotiated pull-read arenas",
    pull_connections_.load(std::memory_order_relaxed));
  m(s, "dfkv_rdma_pull_memory_windows_total", "counter",
    "Pull-read connections isolated with type-2 Memory Windows",
    pull_memory_windows_.load(std::memory_order_relaxed));
  m(s, "dfkv_rdma_pull_mr_fallbacks_total", "counter",
    "Pull-read connections using exact per-connection MR fallback",
    pull_mr_fallbacks_.load(std::memory_order_relaxed));
  m(s, "dfkv_rdma_legacy_connections", "gauge",
    "Connections currently holding only legacy receive arenas",
    legacy_connections_.load(std::memory_order_relaxed));
  s += "# HELP dfkv_rdma_connection_bytes Receive-segment bytes leased by connection class\n";
  s += "# TYPE dfkv_rdma_connection_bytes gauge\n";
  s += "dfkv_rdma_connection_bytes{class=\"data\"} " +
       std::to_string(data_connection_bytes_.load(std::memory_order_relaxed)) +
       "\n";
  s += "dfkv_rdma_connection_bytes{class=\"control\"} " +
       std::to_string(
           control_connection_bytes_.load(std::memory_order_relaxed)) +
       "\n";
  m(s, "dfkv_rdma_v2_ready", "gauge",
    "Whether RDMA v2 has a registered shared receive segment",
    recv_segment_registered_rails_ > 0 ? 1 : 0);
  m(s, "dfkv_rdma_segment_evictions_total", "counter",
     "Connections evicted to free shared receive segment space",
     segment_evictions_.load(std::memory_order_relaxed));
  m(s, "dfkv_rdma_idle_reclaims_total", "counter", "RDMA connections reclaimed on idle timeout",
    IdleReclaims());
  m(s, "dfkv_uring_reads_total", "counter",
    "GET disk reads submitted through the io_uring path (>0 = path active)",
    UringReads());
  m(s, "dfkv_uring_batch_reads_total", "counter",
    "GET disk-read descriptors admitted to io_uring submit groups",
    UringReads());
  m(s, "dfkv_uring_batches_total", "counter",
    "Non-empty io_uring submit groups", UringReadBatches());
  m(s, "dfkv_uring_submit_batches_total", "counter",
    "Non-empty io_uring submit groups", UringReadBatches());
  m(s, "dfkv_uring_batch_max", "gauge",
    "Largest io_uring submit group observed", UringReadBatchMax());
  m(s, "dfkv_uring_completions_total", "counter",
    "Logical io_uring read descriptors completed", UringCompletions());
  m(s, "dfkv_uring_inflight", "gauge",
    "Logical io_uring reads currently outstanding", UringInflight());
  m(s, "dfkv_uring_inflight_max", "gauge",
    "Process-lifetime high-water mark of outstanding io_uring reads",
    UringInflightMax());
  m(s, "dfkv_uring_replies_posted_total", "counter",
    "Ordered status replies successfully posted by io_uring serve loops",
    UringRepliesPosted());
  m(s, "dfkv_uring_send_fences_total", "counter",
    "Signaled status SEND completions reaped by io_uring serve loops",
    UringSendFences());
  m(s, "dfkv_uring_send_post_errors_total", "counter",
    "Status SEND chains rejected after an io_uring disk completion",
    UringSendPostErrors());
  m(s, "dfkv_uring_init_fallbacks_total", "counter",
    "Connections that wanted io_uring but fell back to the sync path (ring init failed)",
    UringInitFallbacks());
  auto rail_metric = [&](const char* name, const char* type, const char* help,
                         const auto& value) {
    s += "# HELP "; s += name; s += " "; s += help; s += "\n";
    s += "# TYPE "; s += name; s += " "; s += type; s += "\n";
    for (size_t i = 0; i < rail_stats_.size(); ++i) {
      s += name;
      s += "{dev=\"" + PromLabelEscape(anchor_devs_[i]) + "\"} ";
      s += std::to_string(value(*rail_stats_[i]));
      s += "\n";
    }
  };
  rail_metric("dfkv_rdma_rail_active_conns", "gauge",
              "RDMA connections currently serving on each local device",
              [](const RailStats& r) {
                return r.active_conns.load(std::memory_order_relaxed);
              });
  rail_metric("dfkv_rdma_rail_completions_total", "counter",
              "RDMA request completions served on each local device",
              [](const RailStats& r) {
                return r.completions.load(std::memory_order_relaxed);
              });
  rail_metric("dfkv_rdma_rail_completion_errors_total", "counter",
              "RDMA error completions on each local device",
              [](const RailStats& r) {
                return r.completion_errors.load(std::memory_order_relaxed);
              });
  rail_metric("dfkv_rdma_rail_put_writes_total", "counter",
              "PUT requests received by RDMA WRITE_WITH_IMM on each local device",
              [](const RailStats& r) {
                return r.put_writes.load(std::memory_order_relaxed);
              });
  rail_metric("dfkv_rdma_rail_put_bytes_total", "counter",
              "PUT payload bytes received on each local device",
              [](const RailStats& r) {
                return r.put_bytes.load(std::memory_order_relaxed);
              });
  rail_metric("dfkv_rdma_rail_get_writes_total", "counter",
              "GET payloads sent by RDMA WRITE on each local device",
              [](const RailStats& r) {
                return r.get_writes.load(std::memory_order_relaxed);
              });
  rail_metric("dfkv_rdma_rail_get_bytes_total", "counter",
              "GET payload bytes sent on each local device",
              [](const RailStats& r) {
                return r.get_bytes.load(std::memory_order_relaxed);
              });
  return s;
}

}  // namespace dfkv
