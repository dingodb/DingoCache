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
// Kernel-random process identity prevents replay after restart. The bounded
// sequence below prevents token reuse within this server's lifetime.
uint64_t RandomNonzeroConnectionSeed() {
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
      DFKV_LOG_ERROR("rdma: secure connection-token generation failed errno=" +
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
uint64_t RdmaServer::RegisterLegacyConnection() {
  std::lock_guard<std::mutex> lock(legacy_mu_);
  if (legacy_token_seed_ == 0)
    legacy_token_seed_ = RandomNonzeroConnectionSeed();
  for (;;) {
    if (legacy_token_sequence_ == std::numeric_limits<uint64_t>::max())
      std::abort();  // Never wrap the process-lifetime identity namespace.
    const uint64_t token = legacy_token_seed_ ^ ++legacy_token_sequence_;
    if (token == 0) continue;
    legacy_connections_tokens_.insert(token);
    return token;
  }
}

void RdmaServer::ForgetLegacyConnection(uint64_t token) {
  std::lock_guard<std::mutex> lock(legacy_mu_);
  legacy_connections_tokens_.erase(token);
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
  // Reserve one staging chunk only if it can hold the largest legal staged
  // object and the hard budget still fits one resident chunk per configured
  // rail plus a spare for concurrent staging. A smaller reserved chunk would
  // steal capacity without ever serving that object.
  const size_t reserve_chunks = anchor_devs_.size() + 2;
  const size_t staging_reserve_bytes =
      min_slot_bytes <= recv_segment_chunk_bytes_ &&
              recv_segment_chunk_bytes_ <=
                  recv_segment_max_bytes_ / reserve_chunks &&
              min_slot_bytes <=
                  (recv_segment_max_bytes_ - recv_segment_chunk_bytes_) / 2
          ? recv_segment_chunk_bytes_
          : 0;
  if (min_slot_bytes == 0 ||
      recv_segment_max_bytes_ < 2 * min_slot_bytes ||
      recv_segment_chunk_bytes_ == 0 ||
      !recv_segments_.Init(recv_segment_chunk_bytes_,
                           recv_segment_max_bytes_,
                           rdma::kV2DataOffset, staging_reserve_bytes)) {
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
    for (const auto& live : live_eps_) live.first->Wake();
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

rdma::RecvSegmentPool::Lease RdmaServer::AllocateReceiveWithPressure(
    size_t bytes, int rail, int numa_node, uint64_t wait_us,
    rdma::RecvSegmentPool::LeaseClass lease_class) {
  auto lease = recv_segments_.Allocate(
      bytes, rdma::kV2DataOffset, rail, numa_node, lease_class);
  if (lease || wait_us == 0) return lease;

  // A pooled peer may post its next (possibly non-replay-safe PUT) operation
  // immediately after receiving a reply, before we poll its new CQE. Preserve
  // a one-second pressure-only idle grace, the interval that passed the
  // short-process hardware run, in addition to the in-flight ownership gate.
  const uint64_t started = SteadyUs();
  for (int round = 0; round < 32 && !lease; ++round) {
    if (SteadyUs() - started >= wait_us) break;
    {
      std::lock_guard<std::mutex> lock(conn_mu_);
      rdma::RcEndpoint* oldest = nullptr;
      rdma::RcEndpoint* sufficient = nullptr;
      uint64_t oldest_active = std::numeric_limits<uint64_t>::max();
      uint64_t sufficient_active = oldest_active;
      const uint64_t now = SteadyUs();
      constexpr uint64_t kPressureIdleGraceUs = 1000000;
      for (const auto& [ep, live] : live_eps_) {
        if (!live.reclaimable->load(std::memory_order_acquire)) continue;
        const uint64_t active =
            ep->last_active_us_.load(std::memory_order_relaxed);
        // A newly inserted endpoint has not stamped its activity yet.
        if (active == 0 || active > now ||
            now - active < kPressureIdleGraceUs) continue;
        if (active < oldest_active) {
          oldest = ep;
          oldest_active = active;
        }
        if (live.recv_lease_bytes >= bytes && active < sufficient_active) {
          sufficient = ep;
          sufficient_active = active;
        }
      }
      rdma::RcEndpoint* victim = sufficient ? sufficient : oldest;
      if (!victim) break;  // all remaining leases have active owners
      live_eps_.erase(victim);  // claim before Wake so concurrent scans skip it
      // The Serve thread must erase under conn_mu_ before destroying its QP.
      victim->Wake();
    }
    segment_evictions_.fetch_add(1, std::memory_order_relaxed);
    // The victim's QP destruction must fence inbound DMA before its lease is
    // released. Give it bounded time to unwind, then try another idle QP.
    for (int i = 0; i < 100 && !lease; ++i) {
      if (SteadyUs() - started >= wait_us) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      lease = recv_segments_.Allocate(
          bytes, rdma::kV2DataOffset, rail, numa_node, lease_class);
    }
  }
  return lease;
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
    // The adapter is valid only for a real, still-known legacy dynamic
    // connection. Lookup and closure are serialized; no historical tokens
    // survive teardown. There is no cancellation/CQ drain: this server has no
    // GET WRITE submission path at all.
    std::lock_guard<std::mutex> lock(legacy_mu_);
    if (legacy_connections_tokens_.find(token) ==
        legacy_connections_tokens_.end()) {
      ::close(boot_fd);
      return;
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
  const bool leased_put_requested = rdma::DevFrameRequestsLeasedPut(devbuf);
  const bool dynamic_pull_requested =
      rdma::DevFrameRequestsDynamicPull(devbuf);
  const bool dynamic_only_requested =
      rdma::DevFrameRequestsDynamicOnly(devbuf);
  const uint64_t declared = rdma::ParseDevFrameMaxBlock(devbuf);
  if (rdma::ParseDevFrameProtocol(devbuf) != rdma::kDevProtoV2 ||
      declared == 0 || !dynamic_pull_requested || !pull_read_requested ||
      (dynamic_only_requested ? writer_retirement_requested
                              : !writer_retirement_requested)) {
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
  // Leave margin within v2.28's ten-second bootstrap timeout.
  constexpr uint64_t kBootstrapPressureUs = 5000000;
  rdma::RecvSegmentPool::Lease recv_lease = AllocateReceiveWithPressure(
      K * slot_size, static_cast<int>(rail_index), rail_numa,
      kBootstrapPressureUs);
  if (!recv_lease) {
    const auto stats = recv_segments_.stats();
    DFKV_LOG_ERROR(
        "rdma v2: receive pool exhausted; refusing connection (need=" +
        std::to_string(K * slot_size) +
        " free=" + std::to_string(stats.free_bytes) +
        " connection_free=" +
        std::to_string(stats.connection_free_bytes) +
        " committed=" + std::to_string(stats.committed_bytes) +
        " max=" + std::to_string(stats.max_bytes) + ")");
    ::close(boot_fd);
    return;
  }
  const bool control_connection = conn_max <= rdma::kV2ControlCap;
  const uint64_t connection_bytes = recv_lease.size();
  std::atomic<uint64_t>* const protocol_connections = &pull_connections_;
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
  // In-flight leased-PUT staging state, one entry per connection recv slot.
  // State is declared before ep: every exceptional/early/normal exit destroys
  // the QP and revokes its MRs before these ranges and their gauges unwind.
  // Successful operations explicitly revoke the exact MR before Reset().
  struct LeasePutState {
    bool active = false;
    uint64_t generation = 0;
    BlockKey key;
    uint64_t payload_len = 0;
    rdma::RecvSegmentPool::Lease lease;
    ibv_mr* mr = nullptr;  // owned by ep, never by the receive-pool chunk
    std::atomic<uint64_t>* active_count = nullptr;
    std::atomic<uint64_t>* active_bytes = nullptr;
    ~LeasePutState() { Reset(); }
    void Reset() {
      const size_t bytes = lease.size();
      lease.Reset();
      if (active) {
        active_bytes->fetch_sub(bytes, std::memory_order_relaxed);
        active_count->fetch_sub(1, std::memory_order_relaxed);
      }
      active = false;
      key = BlockKey{};
      payload_len = 0;
      mr = nullptr;
    }
  };
  struct PullSlotState {
    bool busy = false;
    uint64_t generation = 1;
    size_t data_len = 0;
    size_t value_len = 0;
    rdma::RecvSegmentPool::Lease lease;
    ibv_mr* mr = nullptr;  // endpoint-owned, revoked before Reset()
    // B5-3 zero-copy pull: pinned arena source handed to the peer instead of
    // a staged copy. Empty for staged pulls; the send pin releases exactly
    // when the slot resets (PullRelease or connection teardown).
    PreparedRead arena_read;
    double read_elapsed_sec = 0.0;
    std::atomic<uint64_t>* active_count = nullptr;
    std::atomic<uint64_t>* active_bytes = nullptr;
    ~PullSlotState() { Reset(); }
    void Reset() {
      arena_read = PreparedRead{};
      if (lease) {
        active_bytes->fetch_sub(lease.size(), std::memory_order_relaxed);
        active_count->fetch_sub(1, std::memory_order_relaxed);
        lease.Reset();
      }
      mr = nullptr;
      busy = false;
      data_len = 0;
      value_len = 0;
    }
  };
  std::vector<PullSlotState> pull_slots(K);
  for (auto& state : pull_slots) {
    state.active_count = &dynamic_get_active_;
    state.active_bytes = &dynamic_get_bytes_active_;
  }
  std::vector<MultiPutState> multi_put(K);
  std::vector<LeasePutState> lease_put(K);
  for (auto& state : lease_put) {
    state.active_count = &lease_put_active_;
    state.active_bytes = &lease_put_bytes_active_;
  }
  // generation is the only field that survives release: it monotones upward
  // so every LeasePutReady names a distinct op for diagnostics and any future
  // release/retry wire extension. Never emit 0 (treated as "absent").
  auto next_lease_generation = [&](size_t slot) -> uint64_t {
    uint64_t generation = lease_put[slot].generation + 1;
    if (generation == 0) ++generation;
    lease_put[slot].generation = generation;
    return generation;
  };

  rdma::RcEndpoint ep;
  struct BeforeEndpointTeardown {
    const std::function<void()>& hook;
    ~BeforeEndpointTeardown() { if (hook) hook(); }
  } before_endpoint_teardown{before_endpoint_teardown_for_test_};
  auto release_lease_put = [&](size_t slot) {
    LeasePutState& state = lease_put[slot];
    ep.ReleaseLeaseWriteRegion(state.mr);
    state.Reset();
  };
  auto release_pull = [&](size_t slot) {
    PullSlotState& state = pull_slots[slot];
    ep.ReleaseLeaseReadRegion(state.mr);
    state.arena_read.Commit(Status::kOk, state.data_len,
                           state.read_elapsed_sec);
    state.Reset();
    ++state.generation;
    if (state.generation == 0) ++state.generation;
  };
  constexpr size_t conn_control = rdma::kV2ControlCap;
  if (!ep.Open(dev.empty() ? nullptr : dev.c_str(), conn_control, K,
               /*ib_port=*/1, /*direct_io_buffers=*/false, conn_max)) {
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
  DFKV_LOG_INFO("rdma conn: protocol=v2 declared=" +
                std::to_string(declared) +
                " control=" + std::to_string(conn_control) +
                " shared-slot=" + std::to_string(slot_size) +
                " qd=" + std::to_string(K) +
                " qp=" + std::to_string(ep.Local().qpn) +
                " peer_qp=" + std::to_string(peer_info.qpn));
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
  auto post_request_recv = [&](size_t slot) {
    if (!ep.PostRecv(slot)) return false;
    if (after_request_rearm_for_test_) after_request_rearm_for_test_(slot);
    return true;
  };

  bool armed = true;
  for (size_t i = 0; i < K; ++i) armed = armed && post_request_recv(i);
  if (!armed) {
    ::close(boot_fd);
    return;
  }
  const uint64_t legacy_token =
      dynamic_only_requested ? 0 : RegisterLegacyConnection();
  struct LegacyConnectionIdentity {
    RdmaServer* server;
    uint64_t token;
    ~LegacyConnectionIdentity() {
      if (token != 0) server->ForgetLegacyConnection(token);
    }
  } legacy_identity{this, legacy_token};
  // Receives must be posted before readiness becomes visible. Publish the
  // leased receive-segment address, rkey and slot geometry only after the QP is
  // armed, so the client cannot issue a one-sided write into an unready slot.
  const rdma::RecvSegmentInfo info{
      reinterpret_cast<uint64_t>(recv_lease.data()),
      recv_segment_mr->rkey, slot_size};
  char readiness[rdma::kV2RetirementReadinessBytes];
  const size_t readiness_bytes =
      dynamic_only_requested
          ? rdma::EncodeV2DynamicOnlyReadiness(info, readiness)
          : rdma::EncodeV2Readiness(info, legacy_token, readiness);
  const bool ok =
      readiness_bytes != 0 &&
      net::WriteAll(boot_fd, readiness, readiness_bytes);
  ::close(boot_fd);
  if (!ok) {
    return;
  }
  v2_conns_.fetch_add(1, std::memory_order_relaxed);

  // Fresh QPs have advertised readiness but may already have a first request
  // in transit. Reclaim only after at least one completed request/reply fence
  // has established a genuine quiescent point.
  std::atomic<bool> reclaimable{false};
  // Register this endpoint so Stop() can Wake() us out of WaitComp and join. The
  // running_ check under conn_mu_ closes the race with a concurrent Stop(): either
  // Stop sees us in live_eps_ (and wakes us) or we see running_==false here.
  {
    std::lock_guard<std::mutex> lk(conn_mu_);
    if (!running_) {
      return;
    }
    live_eps_.emplace(&ep, LiveEndpoint{&reclaimable, recv_lease.size()});
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
    bool multi_put_window = false;
    bool multi_put_final = false;
    bool from_lease_put = false;
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
      if (data_slot >= K)
        return false;
      LeasePutState& lstate = lease_put[data_slot];
      const bool from_lease = lstate.active;
      const char* frame =
          from_lease
              ? lstate.lease.data() + rdma::kV2PutPrefixOffset
              : recv_lease.data() + data_slot * slot_size +
                    rdma::kV2PutPrefixOffset;
      // Destination-region geometry: the leased per-op range when this window
      // belongs to an in-flight leased-PUT object, the resident receive slot
      // otherwise. Both size the frame cover check and the logical payload
      // cap so leased objects are not bounded by the connection class.
      const size_t put_region_bytes =
          from_lease ? lstate.lease.size() : slot_size;
      const uint64_t put_payload_cap =
          from_lease
              ? static_cast<uint64_t>(lstate.lease.size() -
                                      rdma::kV2DataOffset)
              : static_cast<uint64_t>(logical_data_cap);
      request->from_lease_put = from_lease;
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
                put_payload_cap) ||
            request->fields.op != static_cast<uint8_t>(WireOp::kCache)) {
          return false;
        }
        // A leased-PUT window must name the exact object the client asked
        // the server to stage: any mismatch is a protocol fault, not a
        // missized retry, so drop the connection (fail closed).
        if (from_lease &&
            (request->fields.payload_len != lstate.payload_len ||
             !(request->fields.Key() == lstate.key))) {
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
                       put_payload_cap, put_region_bytes)) {
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
    if (completion.byte_len < kReqPrefix) return false;
    // A leased-PUT request names the OBJECT size, not an inline payload: its
    // bound is the process-wide payload ceiling, not this connection's
    // inline class. Every other control op keeps the conn_max bound.
    const uint64_t send_max_payload =
        static_cast<uint8_t>(frame[1]) == static_cast<uint8_t>(WireOp::kLeasePut)
            ? static_cast<uint64_t>(max_msg_)
            : static_cast<uint64_t>(conn_max);
    if (!DecodeReqVersion(frame, wire_epoch, &request->fields,
                          send_max_payload)) {
      return false;
    }
    // kRange remains a TCP opcode, but RDMA responder-WRITE GET is retired.
    // Reject before decoding any remote targets or submitting DMA.
    if (request->fields.op == static_cast<uint8_t>(WireOp::kRange))
      return false;
    if (request->fields.op == static_cast<uint8_t>(WireOp::kPullRange)) {
      if (request->fields.payload_len != rdma::kPullPrepareBytes ||
          completion.byte_len != kReqPrefix + rdma::kPullPrepareBytes)
        return false;
      request->contiguous_payload = frame + kReqPrefix;
      return true;
    }
    if (request->fields.op == static_cast<uint8_t>(WireOp::kLeasePut)) {
      // Leased-PUT staging request. The object itself arrives later as one
      // or more WRITE_WITH_IMM windows into the leased range, so this request
      // only needs to prove the object geometry is legal and the slot has no
      // in-flight lease/multi-window state. A lease larger than the logical
      // --max-msg bound is a permanent fault, not backpressure.
      if (!leased_put_requested ||
          request->fields.payload_len == 0 ||
          request->fields.payload_len > static_cast<uint64_t>(max_msg_) ||
          lease_put[recv_slot].active || multi_put[recv_slot].active)
        return false;
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
    size_t first_len = 0;
  };

  auto publish_pull = [&](size_t send_slot, size_t slot,
                          const Request& request, const char* output,
                          size_t output_len, size_t value_len,
                          PreparedRead prepared, double elapsed_sec,
                          Reply* reply) -> bool {
    PullSlotState& state = pull_slots[slot];
    auto error_reply = [&](Status status) {
      prepared.Abort();
      state.Reset();
      EncodeRespVersion(ep.sbuf(send_slot), wire_epoch, status, 0, value_len);
      reply->first_len = response_prefix;
      return true;
    };
    if (request.fields.offset > value_len ||
        output_len > request.fields.length ||
        output_len > value_len - request.fields.offset ||
        (output_len != 0 && output == nullptr))
      return error_reply(Status::kInvalid);
    // Empty results retain a revocable release identity, but authorize no
    // payload READ. The one registered sentinel byte is not a wire request.
    if (output_len == 0) output = state.lease.data();
    state.mr = ep.RegisterLeaseReadRegion(
        const_cast<char*>(output), std::max<size_t>(output_len, 1));
    if (!state.mr && prepared.source_registered() && prepared.Stage()) {
      output = prepared.data();
      state.mr = ep.RegisterLeaseReadRegion(
          const_cast<char*>(output), std::max<size_t>(output_len, 1));
    }
    if (!state.mr) return error_reply(Status::kIOError);
    // Publish disk/coalescer completion before READY; followers must not wait
    // for the client's RELEASE. Its post-read hold stays owned by the grant.
    if (prepared.needs_io())
      prepared.Commit(Status::kOk, output_len, elapsed_sec);
    state.arena_read = std::move(prepared);
    state.read_elapsed_sec = elapsed_sec;
    state.busy = true;
    state.data_len = output_len;
    state.value_len = value_len;
    const rdma::DynamicPullReady ready{
        static_cast<uint32_t>(slot), state.generation, output_len, value_len,
        reinterpret_cast<uint64_t>(output), state.mr->rkey};
    EncodeRespVersion(ep.sbuf(send_slot), wire_epoch, Status::kOk,
                      rdma::kDynamicPullReadyBytes, value_len);
    rdma::EncodeDynamicPullReady(ready, ep.sbuf(send_slot) + response_prefix);
    reply->first_len = response_prefix + rdma::kDynamicPullReadyBytes;
    return true;
  };

  auto try_pinned_pull = [&](size_t send_slot, const Request& request,
                             size_t slot, Reply* reply) -> bool {
    if (!pinned_ram_handler_ || request.fields.length == 0) return false;
    const double started_sec = NowSteadySec();
    PreparedRead pinned;
    if (!pinned_ram_handler_(request.fields.Key(), request.fields.offset,
                             request.fields.length, &pinned) ||
        request.fields.offset > pinned.value_len() ||
        pinned.payload_len() == 0 ||
        pinned.payload_len() > request.fields.length ||
        pinned.payload_len() > pinned.value_len() - request.fields.offset)
      return false;
    const double read_elapsed_sec = NowSteadySec() - started_sec;
    PullSlotState& state = pull_slots[slot];
    state.mr = fail_pinned_registration_for_test_ ? nullptr :
        ep.RegisterLeaseReadRegion(const_cast<char*>(pinned.data()),
                                   pinned.payload_len());
    if (!state.mr) return false;
    state.arena_read = std::move(pinned);
    state.read_elapsed_sec = read_elapsed_sec;
    state.busy = true;
    state.data_len = state.arena_read.payload_len();
    state.value_len = state.arena_read.value_len();
    const rdma::DynamicPullReady ready{
        static_cast<uint32_t>(slot), state.generation, state.data_len,
        state.value_len, reinterpret_cast<uint64_t>(state.arena_read.data()),
        state.mr->rkey};
    EncodeRespVersion(ep.sbuf(send_slot), wire_epoch, Status::kOk,
                      rdma::kDynamicPullReadyBytes, state.value_len);
    rdma::EncodeDynamicPullReady(ready, ep.sbuf(send_slot) + response_prefix);
    reply->first_len = response_prefix + rdma::kDynamicPullReadyBytes;
    pull_zerocopy_served_.fetch_add(1, std::memory_order_relaxed);
    return true;
  };



  auto build_reply = [&](size_t send_slot, const Request& request,
                         Reply* reply) -> bool {
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
      if (!state.busy || state.mr == nullptr ||
          state.generation != fields.length)
        return invalid_reply();
      release_pull(slot);
      encode_status(Status::kOk, 0);
      reply->first_len = response_prefix;
      return true;
    }

    if (fields.op == static_cast<uint8_t>(WireOp::kPullRange)) {
      if (!range_handler_ ||
          fields.payload_len != rdma::kPullPrepareBytes ||
          fields.length > static_cast<uint64_t>(max_msg_) ||
          request.contiguous_payload == nullptr)
        return invalid_reply();
      rdma::PullPrepareControl control;
      if (!rdma::DecodePullPrepareControl(request.contiguous_payload,
                                          &control) ||
          control.slot_index >= K || control.release_generation != 0)
        return invalid_reply();
      const size_t slot = control.slot_index;
      PullSlotState& state = pull_slots[slot];
      if (state.busy) {
        encode_status(Status::kCacheFull, 0);
        reply->first_len = response_prefix;
        return true;
      }
      // RAM hits own only a pin and exact READ registration, no staging lease.
      if (try_pinned_pull(send_slot, request, slot, reply)) return true;
      // Keep zero-length slice semantics on the real range handler. The old
      // WRITE path supplied a connection-sized aligned scratch region even
      // with no client target; capacity zero would introduce disk I/O errors.
      const size_t target_capacity = rdma::V2SlotSize(
          fields.length == 0 ? conn_max : fields.length);
      if (target_capacity == 0) return invalid_reply();
      state.lease = AllocateReceiveWithPressure(
          target_capacity, static_cast<int>(rail_index), rail_numa,
          1000000, rdma::RecvSegmentPool::LeaseClass::kStaging);
      if (!state.lease) {
        encode_status(Status::kCacheFull, 0);
        reply->first_len = response_prefix;
        return true;
      }
      dynamic_get_active_.fetch_add(1, std::memory_order_relaxed);
      dynamic_get_bytes_active_.fetch_add(state.lease.size(),
                                          std::memory_order_relaxed);
      char* target = state.lease.data();
      const char* output = nullptr;
      size_t output_len = 0;
      size_t value_len = 0;
      const Status status =
          range_handler_(key, fields.offset, fields.length, target,
                         target_capacity, &output, &output_len, &value_len);
      if (status != Status::kOk) {
        state.Reset();
        encode_status(status, 0, fields.length == 0 ? 0 : value_len);
        reply->first_len = response_prefix;
        return true;
      }
      // Old RDMA zero-capacity GET cannot return a nonempty remainder. The
      // full length came from this authoritative read, not a racy lookup.
      if (fields.length == 0 && fields.offset < value_len) {
        state.Reset();
        return invalid_reply();
      }
      if (output_len > target_capacity || output_len > fields.length ||
          output_len > value_len ||
          (output_len != 0 && output == nullptr)) {
        state.Reset();
        return invalid_reply();
      }
      if (output_len != 0 && output != target)
        std::memmove(target, output, output_len);
      return publish_pull(send_slot, slot, request, target, output_len,
                          value_len, PreparedRead{}, 0.0, reply);
    }

    if (fields.op == static_cast<uint8_t>(WireOp::kLeasePut)) {
      // Stage exactly this object from the shared receive pool. The whole
      // lease is released by release_lease_put() as soon as the store holds
      // the bytes, so the hard budget bounds data in flight, not
      // connections. A failed allocation is backpressure, not a fault: the
      // client treats kCacheFull like a saturated pull slot and may retry.
      const size_t slot = request.recv_slot;
      const uint64_t want = fields.payload_len;
      const size_t region = rdma::V2SlotSize(want);
      if (region == 0) return false;
      LeasePutState& state = lease_put[slot];
      state.lease = AllocateReceiveWithPressure(
          region, static_cast<int>(rail_index), rail_numa, 1000000,
          rdma::RecvSegmentPool::LeaseClass::kStaging);
      if (!state.lease) {
        lease_put_busy_rejects_.fetch_add(1, std::memory_order_relaxed);
        encode_status(Status::kCacheFull, 0);
        reply->first_len = response_prefix;
        return true;
      }
      state.active = true;
      state.key = key;
      state.payload_len = want;
      lease_put_ops_.fetch_add(1, std::memory_order_relaxed);
      lease_put_bytes_.fetch_add(want, std::memory_order_relaxed);
      lease_put_active_.fetch_add(1, std::memory_order_relaxed);
      lease_put_bytes_active_.fetch_add(state.lease.size(),
                                        std::memory_order_relaxed);
      // A broad chunk rkey survives operation completion and authorizes stale
      // WRITEs into recycled ranges. An exact, uncached MR grants only this
      // lease; synchronous deregistration revokes it before pool reuse/trim.
      // The setup-only MW helper cannot be used here: it consumes CQEs that
      // belong to other pipelined requests.
      state.mr = ep.RegisterLeaseWriteRegion(state.lease.data(),
                                            state.lease.size());
      if (!state.mr) {
        release_lease_put(slot);
        encode_status(Status::kIOError, 0);
        reply->first_len = response_prefix;
        return true;
      }
      const rdma::LeasePutReady ready{
          static_cast<uint32_t>(slot), next_lease_generation(slot),
          state.mr->rkey,
          reinterpret_cast<uint64_t>(state.lease.data()),
          state.lease.size()};
      encode_status(Status::kOk, rdma::kLeasePutReadyBytes);
      rdma::EncodeLeasePutReady(ready, send_buffer + response_prefix);
      reply->first_len = response_prefix + rdma::kLeasePutReadyBytes;
      return true;
    }


    if (fields.op == static_cast<uint8_t>(WireOp::kCache) &&
        request.multi_put_window && !request.multi_put_final) {
      encode_status(Status::kOk, 0);
      reply->first_len = response_prefix;
      return true;
    }

    if (fields.op == static_cast<uint8_t>(WireOp::kCache) &&
        cache_direct_handler_) {
      const bool from_lease = request.from_lease_put;
      LeasePutState& lstate = lease_put[request.data_slot];
      char* cache_data = nullptr;
      size_t cache_cap = 0;
      if (from_lease) {
        // decode matched this window to the in-flight lease; losing that
        // binding now is an internal fault, so release and drop the
        // connection (fail closed) rather than storing from bad memory.
        if (!lstate.active || !lstate.lease ||
            fields.payload_len == 0 ||
            fields.payload_len > lstate.payload_len ||
            fields.payload_len >
                lstate.lease.size() - rdma::kV2DataOffset) {
          return false;
        }
        cache_data = lstate.lease.data() + rdma::kV2DataOffset;
        cache_cap = lstate.lease.size() - rdma::kV2DataOffset;
      } else {
        if (!direct_buffer(request.data_slot) ||
            !direct_mr(request.data_slot) || fields.payload_len == 0 ||
            fields.payload_len > static_cast<uint64_t>(logical_data_cap))
          return invalid_reply();
        cache_data = direct_buffer(request.data_slot);
        cache_cap = direct_buffer_cap;
      }
      if (request.contiguous_payload &&
          request.contiguous_payload != cache_data) {
        std::memcpy(cache_data, request.contiguous_payload,
                    static_cast<size_t>(fields.payload_len));
      }
      const Status status = cache_direct_handler_(
          key, cache_data, static_cast<size_t>(fields.payload_len),
          cache_cap);
      encode_status(status, 0);
      reply->first_len = response_prefix;
      // The store consumed the bytes synchronously (O_DIRECT write or RAM
      // admit copy). Revoke remote WRITE access before returning the staging
      // range to the pool; the status SEND never touches the lease.
      if (from_lease) release_lease_put(request.data_slot);
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
    // The generic handler copies payload bytes synchronously before
    // returning, so a leased-PUT object is fully consumed here and its
    // staging range returns to the pool before the status is encoded.
    if (fields.op == static_cast<uint8_t>(WireOp::kCache) &&
        request.from_lease_put)
      release_lease_put(request.data_slot);
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
    return ep.PostSend(send_slot, reply.first_len);
  };

  // Replies preserve receive order because the client binds destinations by
  // RC SEND order. A consumed control/PUT receive can be posted before its
  // reply is sent, but the reply buffer remains owned until the SEND CQE.
  // If the next request arrives first, retain its CQE and receive-buffer
  // ownership in a depth-bounded FIFO rather than failing or forcing RNR.
  std::vector<ibv_wc> wcs(2 * K);
  ibv_wc* const pending_recv = wcs.data() + K;
  size_t pending_recv_head = 0;
  size_t pending_recv_count = 0;
  auto dispatch_completions = [&](const ibv_wc* batch, int count,
                                  auto&& process_wc,
                                  auto&& has_async_work) -> bool {
    if (count != 0)
      reclaimable.store(false, std::memory_order_release);
    for (int i = 0; i < count; ++i) {
      const ibv_wc& wc = batch[i];
      const bool received = wc.opcode == IBV_WC_RECV ||
                            wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM;
      if (wc.status == IBV_WC_SUCCESS && received &&
          (pending_recv_count != 0 || free_send.empty())) {
        if (pending_recv_count == K) return false;
        pending_recv[(pending_recv_head + pending_recv_count) % K] = wc;
        ++pending_recv_count;
      } else if (!process_wc(wc)) {
        return false;
      }
    }
    while (pending_recv_count != 0 && !free_send.empty()) {
      if (!process_wc(pending_recv[pending_recv_head])) return false;
      pending_recv_head = (pending_recv_head + 1) % K;
      --pending_recv_count;
    }
    if (count != 0 && pending_recv_count == 0 && free_send.size() == K &&
        !has_async_work()) {
      const bool holding_data =
          std::any_of(pull_slots.begin(), pull_slots.end(),
                      [](const auto& state) { return state.busy; }) ||
          std::any_of(lease_put.begin(), lease_put.end(),
                      [](const auto& state) { return state.active; }) ||
          std::any_of(multi_put.begin(), multi_put.end(),
                      [](const auto& state) { return state.active; });
      if (!holding_data)
        reclaimable.store(true, std::memory_order_release);
    }
    if (after_cq_dispatch_for_test_)
      after_cq_dispatch_for_test_(pending_recv_count, free_send.size(),
                                  pending_recv_head);
    return true;
  };
  bool fail = false;
  const int idle_ms = ServerIdleMs();
  active_conns_.fetch_add(1, std::memory_order_relaxed);
  rail_stats.active_conns.fetch_add(1, std::memory_order_relaxed);
  // Failed WCs do not guarantee a meaningful opcode. Retain the raw provider
  // fields and both QP numbers for correlation with short-lived clients.
  auto log_completion_error = [&](const ibv_wc& wc, const char* loop,
                                  size_t posted_sends) {
    DFKV_LOG_WARN(
        "rdma server CQ failure loop=" + std::string(loop) +
        " status=" + std::to_string(wc.status) +
        " opcode_raw=" + std::to_string(wc.opcode) +
        " vendor=" + std::to_string(wc.vendor_err) +
        " wr_id=" + std::to_string(wc.wr_id) +
        " qp=" + std::to_string(ep.Local().qpn) +
        " peer_qp=" + std::to_string(peer_info.qpn) +
        " depth=" + std::to_string(K) +
        " free_send=" + std::to_string(free_send.size()) +
        " queued_recv=" + std::to_string(pending_recv_count) +
        " posted_sends=" + std::to_string(posted_sends));
  };

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
      kSyncFallback,
    };
    struct Queued {
      uint64_t sequence = 0;
      size_t send_slot = 0;
      size_t recv_slot = 0;
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
    // Bound reply posting while preserving CQ progress and receive rearming.
    const size_t send_post_limit = K;
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
      rdma::PullPrepareControl control;
      if (!rdma::DecodePullPrepareControl(qd.request.contiguous_payload,
                                          &control)) return false;
      const double elapsed_sec = NowSteadySec() - qd.submit_sec;
      const bool read_ok =
          qd.read_result >= 0 &&
          static_cast<size_t>(qd.read_result) >=
              qd.read.head() + qd.read.payload_len();
      if (!read_ok) {
        qd.read.Commit(Status::kIOError, 0, elapsed_sec);
        pull_slots[control.slot_index].Reset();
        EncodeRespVersion(ep.sbuf(qd.send_slot), wire_epoch,
                          Status::kIOError, 0);
        qd.reply.first_len = response_prefix;
      } else {
        const char* data = qd.read.data();
        const size_t bytes = qd.read.payload_len();
        const size_t value_len = qd.read.value_len();
        if (!publish_pull(qd.send_slot, control.slot_index, qd.request,
                          data, bytes, value_len, std::move(qd.read),
                          elapsed_sec, &qd.reply)) return false;
      }
      qd.disk_state = DiskState::kComplete;
      qd.ready = true;
      return true;
    };

    auto process_wc = [&](const ibv_wc& wc) -> bool {
      if (wc.status != IBV_WC_SUCCESS) {
        log_completion_error(wc, "uring", posted_sends);
        completion_errors_.fetch_add(1, std::memory_order_relaxed);
        rail_stats.completion_errors.fetch_add(1,
                                                std::memory_order_relaxed);
        return false;
      }
      if (wc.opcode == IBV_WC_SEND) {
        const size_t sid = static_cast<size_t>(wc.wr_id);
        if (sid >= K) return false;
        if (posted_sends == 0) return false;
        --posted_sends;
        uring_send_fences_.fetch_add(1, std::memory_order_relaxed);
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
      const ReqFields& fields = request.fields;

      bool handled = false;
      rdma::PullPrepareControl control;
      if (fields.op == static_cast<uint8_t>(WireOp::kPullRange) &&
          fields.length != 0 && fields.length <= max_msg_ &&
          request.contiguous_payload != nullptr &&
          rdma::DecodePullPrepareControl(request.contiguous_payload, &control) &&
          control.slot_index < K && control.release_generation == 0 &&
          !pull_slots[control.slot_index].busy) {
        if (try_pinned_pull(qd.send_slot, request, control.slot_index,
                            &qd.reply)) {
          handled = true;
          qd.ready = true;
        } else {
          PullSlotState& state = pull_slots[control.slot_index];
          const size_t capacity = rdma::V2SlotSize(fields.length);
          state.lease = AllocateReceiveWithPressure(
              capacity, static_cast<int>(rail_index), rail_numa, 1000000,
              rdma::RecvSegmentPool::LeaseClass::kStaging);
          if (state.lease) {
            dynamic_get_active_.fetch_add(1, std::memory_order_relaxed);
            dynamic_get_bytes_active_.fetch_add(state.lease.size(),
                                                std::memory_order_relaxed);
            const double started_sec = NowSteadySec();
            PreparedRead prepared = prepare_read_handler_(
                fields.Key(), fields.offset, fields.length, state.lease.data(),
                capacity);
            if (prepared.status() == Status::kOk && prepared.needs_io() &&
                prepared.fd() >= 0 &&
                prepared.aligned_len() <= capacity &&
                prepared.aligned_len() <= std::numeric_limits<unsigned>::max()) {
              // Reserve while the kernel owns staging, before a later receive
              // can prepare or release this slot.
              state.busy = true;
              qd.desc.fd = prepared.fd();
              qd.desc.buf = prepared.staging();
              qd.desc.len = static_cast<unsigned>(prepared.aligned_len());
              qd.desc.off = prepared.aligned_off();
              qd.disk_state = DiskState::kWaiting;
              qd.read = std::move(prepared);
              qd.request = std::move(request);
              qd.submit_sec = started_sec;
              handled = true;
            } else if (prepared.status() == Status::kOk && !prepared.needs_io()) {
              const char* data = prepared.data();
              const size_t bytes = prepared.payload_len();
              const size_t value_len = prepared.value_len();
              if (!publish_pull(qd.send_slot, control.slot_index, request, data,
                                bytes, value_len, std::move(prepared),
                                NowSteadySec() - started_sec, &qd.reply))
                return false;
              handled = true;
              qd.ready = true;
            } else if (prepared.status() != Status::kOk &&
                       prepared.status() != Status::kInvalid) {
              const Status status = prepared.status();
              const size_t value_len = prepared.value_len();
              prepared.Abort();
              state.Reset();
              EncodeRespVersion(ep.sbuf(qd.send_slot), wire_epoch, status, 0,
                                value_len);
              qd.reply.first_len = response_prefix;
              handled = true;
              qd.ready = true;
            } else {
              prepared.Abort();
              state.Reset();
            }
          }
        }
      }
      if (!handled) {
        // A copy-coalescer follower must not synchronously wait on a disk
        // leader whose completion only this Serve thread can advance.
        const bool waiting_for_disk =
            fields.op == static_cast<uint8_t>(WireOp::kPullRange) &&
            std::any_of(queue.begin(), queue.end(), [](const Queued& earlier) {
              return earlier.disk_state == DiskState::kWaiting ||
                     earlier.disk_state == DiskState::kInflight;
            });
        if (waiting_for_disk) {
          qd.disk_state = DiskState::kSyncFallback;
          qd.request = std::move(request);
        } else {
          if (!build_reply(qd.send_slot, request, &qd.reply)) return false;
          qd.ready = true;
        }
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

    auto finish_sync_waiting = [&]() -> bool {
      if (ring.inflight() != 0 ||
          std::any_of(queue.begin(), queue.end(), [](const Queued& qd) {
            return qd.disk_state == DiskState::kWaiting;
          })) return true;
      for (Queued& qd : queue) {
        if (qd.disk_state != DiskState::kSyncFallback) continue;
        if (!build_reply(qd.send_slot, qd.request, &qd.reply)) return false;
        qd.disk_state = DiskState::kComplete;
        qd.ready = true;
      }
      return true;
    };

    auto emit_ready = [&]() -> bool {
      while (!queue.empty() && queue.front().ready &&
             posted_sends < send_post_limit) {
        Queued& qd = queue.front();
        if (qd.sequence != next_emit_sequence ||
            next_emit_sequence == std::numeric_limits<uint64_t>::max())
          return false;
        Reply& reply = qd.reply;
        if (!post_request_recv(qd.recv_slot)) return false;
        if (!post_reply(qd.send_slot, reply)) {
          uring_send_post_errors_.fetch_add(1, std::memory_order_relaxed);
          return false;
        }
        if (after_reply_post_for_test_) after_reply_post_for_test_();
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
      if (g >= 0 && reorder_cq_for_test_)
        reorder_cq_for_test_(wcs.data(), &g, K);
      if (g > 0) {
        ep.last_active_us_.store(SteadyUs(), std::memory_order_relaxed);
        progressed = true;
        if (!dispatch_completions(
                wcs.data(), g, process_wc,
                [&] {
                  // A queued descriptor or unretired SEND still owns data
                  // even if all receive slots appear free.
                  return !queue.empty() || ring.inflight() != 0 ||
                         posted_sends != 0;
                }))
          fail = true;
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
      if (fail || !finish_sync_waiting() || !emit_ready()) {
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

      // SEND completions release control reply buffers. Some providers can
      // lose a completion-channel edge
      // after our ready-only PollComp drain, so never put an outstanding SEND
      // fence behind the multi-minute connection-idle wait. A bounded wait
      // blocks (no spin) and its timeout path performs a final CQ poll.
      const int verbs_wait_ms =
          posted_sends != 0 || reorder_cq_for_test_ ? 1 : idle_ms;
      g = ep.WaitComp(wcs.data(), static_cast<int>(K), verbs_wait_ms);
      if (g >= 0 && reorder_cq_for_test_)
        reorder_cq_for_test_(wcs.data(), &g, K);
      if (g == 0) {
        if (posted_sends != 0) continue;
        idle_reclaims_.fetch_add(1, std::memory_order_relaxed);
        break;
      }
      if (g < 0) break;  // disconnect, endpoint error, or Stop()'s Wake
      ep.last_active_us_.store(SteadyUs(), std::memory_order_relaxed);
      if (!dispatch_completions(
              wcs.data(), g, process_wc,
              [&] {
                return !queue.empty() || ring.inflight() != 0 ||
                       posted_sends != 0;
              }))
        fail = true;
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
    return;
  }
sync_serve_loop:;
#endif  // DFKV_WITH_URING

  auto process_sync_wc = [&](const ibv_wc& wc) -> bool {
    if (wc.status != IBV_WC_SUCCESS) {
      log_completion_error(wc, "sync", K - free_send.size());
      completion_errors_.fetch_add(1, std::memory_order_relaxed);
      rail_stats.completion_errors.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    if (wc.opcode == IBV_WC_SEND) {
      const size_t sid = static_cast<size_t>(wc.wr_id);
      if (sid >= K) return false;
      free_send.push_back(sid);
      return true;
    }
    Request request;
    if (!decode_request(wc, &request) || free_send.empty()) return false;
    completions_.fetch_add(1, std::memory_order_relaxed);
    rail_stats.completions.fetch_add(1, std::memory_order_relaxed);
    const size_t r = request.recv_slot;
    const size_t s = free_send.back();
    free_send.pop_back();
    Reply reply;
    if (!build_reply(s, request, &reply)) return false;
    if (!post_request_recv(r)) return false;
    if (!post_reply(s, reply)) return false;
    if (after_reply_post_for_test_) after_reply_post_for_test_();
    return true;
  };
  while (running_ && !fail) {
    const bool deferred_for_test =
        reorder_cq_for_test_ && free_send.size() != K;
    const int wait_ms =
        pending_recv_count != 0 || deferred_for_test ? 1 : idle_ms;
    int g = ep.WaitComp(wcs.data(), static_cast<int>(K), wait_ms);
    if (g >= 0 && reorder_cq_for_test_)
      reorder_cq_for_test_(wcs.data(), &g, K);
    if (g > 0)
      ep.last_active_us_.store(SteadyUs(), std::memory_order_relaxed);
    if (g == 0) {
      if (pending_recv_count != 0 || deferred_for_test) continue;
      idle_reclaims_.fetch_add(1, std::memory_order_relaxed);
      break;
    }
    if (g < 0) break;  // error / Stop()'s Wake()
    if (!dispatch_completions(wcs.data(), g, process_sync_wc,
                              [] { return false; }))
      fail = true;
  }
  // Any prepared sends without completions destructor-abort below.
  rail_stats.active_conns.fetch_sub(1, std::memory_order_relaxed);
  active_conns_.fetch_sub(1, std::memory_order_relaxed);
  { std::lock_guard<std::mutex> lk(conn_mu_); live_eps_.erase(&ep); }
  // ep destruction fences inbound DMA and revokes exact MRs before operation
  // lease/pin RAII owners unwind.
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
    "Unleased bytes in committed receive chunks (including staging reserve)",
    segment.free_bytes);
  m(s, "dfkv_rdma_recv_segment_largest_free_range_bytes", "gauge",
    "Largest contiguous unleased range in any receive chunk",
    segment.largest_free_range);
  m(s, "dfkv_rdma_recv_segment_connection_free_bytes", "gauge",
    "Unleased receive bytes available to connections (excludes staging reserve)",
    segment.connection_free_bytes);
  m(s, "dfkv_rdma_recv_segment_connection_largest_free_range_bytes", "gauge",
    "Largest contiguous unleased connection range",
    segment.connection_largest_free_range);
  m(s, "dfkv_rdma_recv_segment_staging_reserved_bytes", "gauge",
    "Receive-pool bytes dedicated to in-flight GET/PUT staging",
    segment.staging_reserved_bytes);
  m(s, "dfkv_rdma_recv_segment_staging_reserved_free_bytes", "gauge",
    "Unleased bytes in the staging-only receive chunk",
    segment.staging_reserved_free_bytes);
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
    "Connections currently using negotiated pull-read",
    pull_connections_.load(std::memory_order_relaxed));
  s += "# HELP dfkv_rdma_connection_bytes Receive-segment bytes leased by connection class\n";
  s += "# TYPE dfkv_rdma_connection_bytes gauge\n";
  s += "dfkv_rdma_connection_bytes{class=\"data\"} " +
       std::to_string(data_connection_bytes_.load(std::memory_order_relaxed)) +
       "\n";
  s += "dfkv_rdma_connection_bytes{class=\"control\"} " +
       std::to_string(
           control_connection_bytes_.load(std::memory_order_relaxed)) +
       "\n";
  s += "dfkv_rdma_connection_bytes{class=\"lease_put_inflight\"} " +
       std::to_string(
           lease_put_bytes_active_.load(std::memory_order_relaxed)) +
       "\n";
  m(s, "dfkv_rdma_leaseput_ops_total", "counter",
    "Leased-PUT staging operations served", lease_put_ops_);
  m(s, "dfkv_rdma_leaseput_bytes_total", "counter",
    "Logical object bytes delivered via leased-PUT staging",
    lease_put_bytes_);
  m(s, "dfkv_rdma_leaseput_busy_rejects_total", "counter",
    "Leased-PUT requests refused because the receive pool was exhausted",
    lease_put_busy_rejects_);
  m(s, "dfkv_rdma_leaseput_active", "gauge",
    "In-flight leased-PUT staging leases currently held",
    lease_put_active_);
  m(s, "dfkv_rdma_leaseput_bytes_active", "gauge",
    "Receive-pool bytes held by in-flight leased-PUT staging",
    lease_put_bytes_active_);
  m(s, "dfkv_rdma_pull_zerocopy_served_total", "counter",
    "Dynamic-pull GETs served directly from the pinned RAM arena without a staging copy",
    pull_zerocopy_served_);
  m(s, "dfkv_rdma_dynamic_get_active", "gauge",
    "In-flight dynamic GET staging operations currently held",
    dynamic_get_active_);
  m(s, "dfkv_rdma_dynamic_get_bytes_active", "gauge",
    "Receive-pool bytes held by in-flight dynamic GET staging",
    dynamic_get_bytes_active_);
  m(s, "dfkv_rdma_dynamic_get_mr_active", "gauge",
    "Process-wide exact dynamic GET READ registrations currently held",
    rdma::RcEndpoint::LeaseReadMrActive());
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
  return s;
}

}  // namespace dfkv
