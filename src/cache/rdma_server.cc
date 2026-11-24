#include "cache/rdma_server_env.h"
#include "utils/env_parse.h"
#include "common/config_dump.h"
#include "cache/rdma_server.h"
#include "cache/rdma_pressure_plan.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <sys/epoll.h>
#include <sys/eventfd.h>
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


// Added to twice a revoked GET range's wire time in its quarantine
// (MwQuarantineUs); measured, see docs/design/server-memory-windows.md.
constexpr uint64_t kMwMarginUs = 1000;

uint64_t SteadyNs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

size_t RecvSegmentBytes() {
  constexpr size_t kDefault = 128ull << 30;
  const size_t bytes =
      RecvBytesFromEnv("DFKV_RDMA_RECV_SEGMENT_SIZE", kDefault);
  config_dump::RecordResolved("DFKV_RDMA_RECV_SEGMENT_SIZE",
                              std::to_string(bytes));
  return bytes;
}

size_t RecvChunkBytes(size_t max_bytes) {
  constexpr size_t kDefault = 256ull << 20;
  const size_t fallback = std::min(kDefault, max_bytes);
  const size_t parsed =
      RecvBytesFromEnv("DFKV_RDMA_RECV_CHUNK_BYTES", fallback);
  const size_t bytes = parsed == 0 ? 0 : std::min(parsed, max_bytes);
  config_dump::RecordResolved("DFKV_RDMA_RECV_CHUNK_BYTES",
                              std::to_string(bytes));
  return bytes;
}

// Strict parse: anything but a plain number keeps the default, so a typo can
// neither lift the limit nor set it to something tiny.
uint64_t MaxPendingHandshakes(uint64_t fallback) {
  envparse::U64Spec spec;
  spec.overflow_is_default = true;
  spec.fallback = fallback;
  const uint64_t out =
      envparse::EnvU64("DFKV_RDMA_MAX_PENDING_HANDSHAKES", spec);
  config_dump::RecordResolved("DFKV_RDMA_MAX_PENDING_HANDSHAKES",
                              std::to_string(out));
  return out;
}

uint64_t RecvChunkIdleMs() {
  envparse::U64Spec spec;
  spec.overflow_is_default = true;
  spec.legacy = envparse::Legacy::kFull;
  spec.negative = envparse::Negative::kWraps;
  spec.fallback = 60000;
  spec.max = 86400000;
  spec.above = envparse::Above::kClamp;
  const uint64_t out = envparse::EnvU64("DFKV_RDMA_RECV_CHUNK_IDLE_MS", spec);
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
int StopDrainMs();  // defined with the other server settings below
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
  max_pending_handshakes_ = MaxPendingHandshakes(kDefaultMaxPendingHandshakes);
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
  session_fd_enabled_ =
      config_dump::EnvBool("DFKV_RDMA_SESSION_FD", false);
  const RdmaServerMwSettings mw = RdmaServerMwFromEnv();
  mw_enabled_ = mw.enabled;
  mw_floor_us_ = mw.floor_ms * 1000;
  mw_max_us_ = mw.max_ms * 1000;
  mw_margin_us_ = kMwMarginUs;
  mw_quarantine_cap_ = mw.quarantine_bytes;  // 0: set by Start
  mw_arena_cap_ = mw.arena_bytes;
  mw_link_gbps_ = mw.link_gbps;
  config_dump::RecordResolved("DFKV_RDMA_SERVER_MW", mw_enabled_ ? "1" : "0");
  config_dump::RecordResolved("DFKV_RDMA_SERVER_MW_QUARANTINE_MS",
                              std::to_string(mw.floor_ms));
  config_dump::RecordResolved("DFKV_RDMA_SERVER_MW_QUARANTINE_MAX_MS",
                              std::to_string(mw.max_ms));
}

RdmaServer::~RdmaServer() { Stop(); }

bool RdmaServer::TrialRegistration(rdma::RcEndpoint& anchor,
                                   const std::string& device,
                                   rdma::RecvSegment* segment) {
  if (startup_registration_for_test_)
    return startup_registration_for_test_(device);
  ibv_mr* mr = anchor.RegisterTransient(
      segment->data(), segment->size(),
      rdma::RcEndpoint::TransientAccess::kRemoteWrite);
  if (!mr) return false;
  anchor.ReleaseTransient(mr);
  return true;
}

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
  // Each GET staged from disk and each leased PUT registers its range of a
  // receive chunk for that one operation. On 2 MiB pages that costs a
  // fraction of what 4 KiB pages cost from about 4 MiB up (64 MiB: 0.4 ms
  // instead of 2.4 ms per operation, measured), as for the RAM tier arena.
  const bool recv_huge_pages =
      config_dump::EnvBool("DFKV_RDMA_RECV_HUGEPAGES", true);
  recv_segments_.set_huge_pages(recv_huge_pages);
  // A chunk windows are bound in is deregistered before its memory goes
  // (the trim unlinks it first, under the pool lock).
  if (mw_enabled_)
    recv_segments_.set_release_hook([this](const rdma::RecvSegment* segment) {
      return mw_regions_.Forget(segment->data());
    });
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
      anchor = initialize_anchor_for_test_(device, initial_segment->data(),
                                           initial_segment->size());
    } else {
      // Opens the rail and keeps its device open. No receive chunk stays
      // registered as a whole: each connection registers only its own
      // slots. The initial chunk is registered once and released here, so
      // a rail that cannot pin memory (RLIMIT_MEMLOCK) refuses startup
      // instead of every client later.
      anchor = std::make_unique<rdma::RcEndpoint>();
      if (!anchor->Open(device.c_str(), rdma::kV2ControlCap, 1)) {
        DFKV_LOG_ERROR("rdma: failed to open required v2 device " + device);
        anchor.reset();
      } else if (!TrialRegistration(*anchor, device, initial_segment)) {
        DFKV_LOG_ERROR(
            "rdma: cannot register the initial v2 receive chunk on " +
            device + " (check LimitMEMLOCK)");
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
  // Pipelined READs are offered only if every rail's QPs accept
  // kV2ReadDepth in flight: the probe is not per rail.
  int min_responder_depth = std::numeric_limits<int>::max();
  for (const auto& anchor : anchors_)
    min_responder_depth =
        std::min(min_responder_depth, anchor->device_responder_read_depth());
  if (responder_read_depth_for_test_ != 0)
    min_responder_depth = responder_read_depth_for_test_;
  read_depth_offered_ = !anchors_.empty() &&
                        min_responder_depth >= rdma::kV2ReadDepth;
  rail_stats_ = std::move(initialized_stats);
  rail_retry_peers_.clear();
  for (size_t i = 0; i < anchor_devs_.size(); ++i)
    rail_retry_peers_.push_back(std::make_unique<PeerWindow>());
  recv_segment_registered_rails_ = anchors_.size();
  LogTopologySummary(requested_devices.size(), anchors_.size(),
                     discovery.devices);
  rail_mw_.assign(anchors_.size(), 0);
  rail_grant_bytes_ =
      std::make_unique<std::atomic<uint64_t>[]>(anchors_.size());
  if (mw_enabled_) {
    uint64_t slowest = 0;
    bool any = false;
    // Windows per connection: one per pull slot. The receive pool bounds the
    // connections: at most that many control connections of full depth.
    const uint64_t connections =
        recv_segment_max_bytes_ /
        std::max<uint64_t>(1, PipelineDepth() *
                                  rdma::V2SlotSize(rdma::kV2ControlCap));
    for (size_t rail = 0; rail < anchors_.size(); ++rail) {
      rdma::RcEndpoint& anchor = *anchors_[rail];
      rail_mw_[rail] = anchor.ProbeType2Windows() ? 1 : 0;
      any = any || rail_mw_[rail];
      const uint64_t rate = anchor.PortRateMbps();
      if (rate != 0 && (slowest == 0 || rate < slowest)) slowest = rate;
      DFKV_LOG_INFO("rdma: rail " + anchor_devs_[rail] + ": GET grants use " +
                    (rail_mw_[rail] ? "memory windows (type 2B)"
                                    : "per-grant MRs (no type-2B windows)") +
                    ", link " + std::to_string(rate) + " Mbit/s");
      const int max_mw = anchor.MaxWindows();
      if (rail_mw_[rail] && max_mw > 0 &&
          static_cast<uint64_t>(max_mw) < connections * PipelineDepth())
        DFKV_LOG_WARN("rdma: rail " + anchor_devs_[rail] + " offers " +
                      std::to_string(max_mw) + " memory windows, fewer than "
                      "depth x the connections the receive pool can hold (" +
                      std::to_string(connections * PipelineDepth()) +
                      "); connections past it serve GETs with per-grant MRs");
    }
    if (mw_link_gbps_ != 0 &&
        (slowest == 0 || mw_link_gbps_ * 1000 < slowest))
      slowest = mw_link_gbps_ * 1000;
    mw_rate_mbps_ = slowest;
    if (mw_quarantine_cap_ == 0) mw_quarantine_cap_ = recv_segment_max_bytes_ / 8;
    if (mw_arena_tier_ && mw_arena_tier_->arena_base() &&
        mw_arena_tier_->extent_bytes() != 0) {
      mw_arena_ = mw_arena_tier_->arena_base();
      mw_arena_bytes_ = mw_arena_tier_->arena_bytes();
      mw_extent_bytes_ = mw_arena_tier_->extent_bytes();
      mw_arena_tier_->HoldArenaRegistrations(1);
    }
    if (any)
      mw_regions_.Start(
          anchors_.size(),
          [this](size_t rail, char* base, size_t size) -> ibv_mr* {
            if (mw_register_for_test_ && !mw_register_for_test_(rail, base))
              return nullptr;
            return anchors_[rail]->RegisterBindRegion(base, size);
          },
          [](ibv_mr* mr) { return rdma::RcEndpoint::DeregisterBindRegion(mr); },
          mw_arena_cap_);
    config_dump::RecordResolved("DFKV_RDMA_SERVER_MW_QUARANTINE_BYTES",
                                std::to_string(mw_quarantine_cap_));
  }
  if (anchor_devs_.size() > 1)
    DFKV_LOG_INFO("rdma multi-rail anchors: " +
                  std::to_string(anchors_.size()) + "/" +
                  std::to_string(anchor_devs_.size()) +
                  " devices pinned");
  if (session_fd_enabled_) {
    session_epoll_fd_ = ::epoll_create1(EPOLL_CLOEXEC);
    session_stop_fd_ = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    epoll_event stop{};
    stop.events = EPOLLIN;
    stop.data.u64 = 0;  // session ids start at 1
    if (session_epoll_fd_ < 0 || session_stop_fd_ < 0 ||
        ::epoll_ctl(session_epoll_fd_, EPOLL_CTL_ADD, session_stop_fd_,
                    &stop) != 0) {
      DFKV_LOG_WARN(std::string("rdma: no session channels (") +
                    std::strerror(errno) + "); DFKV_RDMA_SESSION_FD is off");
      if (session_epoll_fd_ >= 0) ::close(session_epoll_fd_);
      if (session_stop_fd_ >= 0) ::close(session_stop_fd_);
      session_epoll_fd_ = session_stop_fd_ = -1;
      session_fd_enabled_ = false;
    } else {
      session_thread_ = std::thread([this] {
        NameThisThread("rdma-session");
        SessionWatchLoop();
      });
    }
  }
  // Watch each rail's port at run time (RDS-02): off unless
  // DFKV_RDMA_PORT_CHECK_MS is set (1000 recommended).
  if (const char* value = std::getenv("DFKV_RDMA_PORT_CHECK_MS");
      value && *value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (errno == 0 && end != value && *end == '\0' && parsed >= 0)
      port_check_ms_ = static_cast<int>(std::min<long>(parsed, 60000));
  }
  config_dump::RecordResolved("DFKV_RDMA_PORT_CHECK_MS",
                              std::to_string(port_check_ms_));
  rail_port_state_ =
      std::make_unique<std::atomic<int>[]>(anchor_devs_.size());
  for (size_t i = 0; i < anchor_devs_.size(); ++i) rail_port_state_[i] = -1;
  rail_last_notice_ms_.assign(anchor_devs_.size(), 0);
  rail_notified_state_.assign(anchor_devs_.size(), -1);
  rail_down_polls_.assign(anchor_devs_.size(), 0);
  rail_down_confirmed_ =
      std::make_unique<std::atomic<bool>[]>(anchor_devs_.size());
  for (size_t i = 0; i < anchor_devs_.size(); ++i)
    rail_down_confirmed_[i] = false;
  if (port_check_ms_ > 0) {
    port_watcher_ = std::make_unique<rdma::PortWatcher>(anchor_devs_);
    CheckRailPorts();  // a rail down at startup is refused from the start
  }
  running_ = true;
  accept_thread_ =
      std::thread([this] { NameThisThread("rdma-accept"); AcceptLoop(); });
  if (port_watcher_)
    port_thread_ = std::thread([this] {
      NameThisThread("rdma-ports");
      PortWatchLoop();
    });
  if (recv_chunk_idle_ms_ != 0 || mw_enabled_) {
    // Also releases quarantined GET ranges (DFKV_RDMA_SERVER_MW), waking
    // for the earliest end.
    recv_trim_thread_ = std::thread([this] {
      NameThisThread("rdma-recv-trim");
      std::unique_lock<std::mutex> lock(recv_trim_mu_);
      uint64_t trimmed_us = SteadyUs();
      while (running_.load(std::memory_order_relaxed)) {
        auto wait = std::chrono::microseconds(1000000);
        if (const uint64_t next = MwQuarantineEndUs(false); next != 0) {
          const uint64_t now = SteadyUs();
          wait = std::min(wait, std::chrono::microseconds(
                                    next > now ? next - now : 0));
        }
        // No predicate: a notify for an earlier quarantine end must end
        // the wait too, not only Stop().
        recv_trim_cv_.wait_for(lock, wait);
        if (!running_.load(std::memory_order_relaxed)) break;
        lock.unlock();
        if (mw_enabled_) MwReleaseQuarantine(false);
        // Once a second, as before, however often a quarantine ends.
        if (recv_chunk_idle_ms_ != 0 && SteadyUs() - trimmed_us >= 1000000) {
          recv_segments_.TrimIdle(recv_chunk_idle_ms_);
          trimmed_us = SteadyUs();
        }
        lock.lock();
      }
    });
  }
  return Status::kOk;
}

size_t RdmaServer::DefaultRail() const {
  for (size_t i = 0; rail_port_state_ && i < anchor_devs_.size(); ++i)
    if (rail_port_state_[i].load(std::memory_order_acquire) != 0) return i;
  return 0;
}

void RdmaServer::CheckRailPorts() {
  const uint64_t now_ms = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
  std::function<int(size_t)> seam;
  {
    std::lock_guard<std::mutex> lock(rail_listener_mu_);
    seam = port_state_for_test_;
  }
  for (size_t rail = 0; rail < anchor_devs_.size(); ++rail) {
    // The query (ibv_query_port, about 1 ms) runs under no lock.
    const int state = seam ? seam(rail) : port_watcher_->State(rail);
    if (state < 0) continue;  // unknown: keep what was last seen
    const bool active = state == 1;
    if (active) {
      // Cleared before ACTIVE is published: a handshake that reads ACTIVE
      // must not find the old confirmation and end itself.
      rail_down_polls_[rail] = 0;
      rail_down_confirmed_[rail].store(false, std::memory_order_release);
    }
    const int before =
        rail_port_state_[rail].exchange(state, std::memory_order_acq_rel);
    if (state != before) {
      rail_port_changes_.fetch_add(1, std::memory_order_relaxed);
      if (before >= 0 || !active)
        DFKV_LOG_WARN("rdma: rail " + anchor_devs_[rail] + " port is " +
                      (active ? "ACTIVE again" : "not ACTIVE; new connections "
                                                 "are refused"));
    }
    // New handshakes are refused from the first poll that sees the port down
    // (rail_port_state_). Existing connections end only on the second poll
    // in a row: a link that retrains within a poll keeps its pooled QPs.
    if (!active && ++rail_down_polls_[rail] == 2) {
      // Set before the wake: a connection registering meanwhile checks it
      // after its registration and ends itself (Serve).
      rail_down_confirmed_[rail].store(true, std::memory_order_release);
      DFKV_LOG_WARN("rdma: rail " + anchor_devs_[rail] +
                    " port still not ACTIVE; its connections are ended");
      // That rail's connections only: wake them, as Stop does, and they end.
      std::lock_guard<std::mutex> lock(conn_mu_);
      for (const auto& [ep, live] : live_eps_) {
        if (live.rail != rail) continue;
        ep->Wake();
        rail_down_connections_ended_.fetch_add(1, std::memory_order_relaxed);
      }
    }
    // The listener (an early heartbeat) hears of a state that differs from
    // the last one it heard, at most once a second per rail; a change within
    // the second is sent on the first poll after it, if it still holds. The
    // first look at startup is what it starts from, not a change.
    if (before < 0) {
      rail_notified_state_[rail] = state;
      continue;
    }
    if (state == rail_notified_state_[rail] ||
        now_ms - rail_last_notice_ms_[rail] < 1000)
      continue;
    rail_last_notice_ms_[rail] = now_ms;
    rail_notified_state_[rail] = state;
    RailStateListener listener;
    {
      std::lock_guard<std::mutex> lock(rail_listener_mu_);
      listener = rail_listener_;
    }
    if (listener) listener(anchor_devs_[rail], active);
  }
}

void RdmaServer::PortWatchLoop() {
  std::unique_lock<std::mutex> lock(port_mu_);
  while (!port_cv_.wait_for(
      lock, std::chrono::milliseconds(port_check_ms_),
      [this] { return !running_.load(std::memory_order_relaxed); })) {
    lock.unlock();
    CheckRailPorts();
    lock.lock();
  }
}

void RdmaServer::Stop() {
  Quiesce();
  // Under the same lock: a second caller must not release the rails while the
  // first is still joining connection threads.
  std::lock_guard<std::mutex> lk(stop_mu_);
  // Every window is gone with its connection: the bind-only regions go
  // before the rails' protection domains, and before the arena.
  mw_regions_.Stop();
  if (mw_arena_) {
    mw_arena_tier_->HoldArenaRegistrations(-1);
    mw_arena_ = nullptr;
  }
  anchors_.clear();  // drop the lifetime device refs (frees pool MRs last)
}

void RdmaServer::Quiesce() {
  std::lock_guard<std::mutex> lk(stop_mu_);
  if (!running_.exchange(false)) return;
  recv_trim_cv_.notify_all();
  {
    std::lock_guard<std::mutex> lock(port_mu_);
    port_cv_.notify_all();
  }
  if (port_thread_.joinable()) port_thread_.join();
  if (listen_fd_ >= 0) ::shutdown(listen_fd_, SHUT_RDWR);  // wake accept()
  if (accept_thread_.joinable()) accept_thread_.join();
  if (recv_trim_thread_.joinable()) recv_trim_thread_.join();
  if (listen_fd_ >= 0) { ::close(listen_fd_); listen_fd_ = -1; }
  // A thread still in its handshake waits for the peer's frames until the
  // bootstrap deadline (10 s): end those reads now. A thread that comes to
  // register its socket after this gives up by itself.
  {
    std::lock_guard<std::mutex> lock(handshake_mu_);
    handshakes_closed_ = true;
    for (const int fd : handshake_fds_) ::shutdown(fd, SHUT_RDWR);
  }
  // Drain: a connection that has sent a GET's READY or a PUT lease and waits
  // for its client goes on serving (running_ is false, so it takes no part
  // in pressure reclaim and ends as soon as it holds no grant) for up to
  // stop_drain_ms, so reads and writes in flight complete instead of ending
  // in a completion error at the client. The rest are woken at once.
  const int drain_ms = StopDrainMs();
  const auto drain_until = std::chrono::steady_clock::now() +
                           std::chrono::milliseconds(drain_ms < 0 ? 0 : drain_ms);
  for (;;) {
    bool holding = false;
    {
      std::lock_guard<std::mutex> lk(conn_mu_);
      for (const auto& live : live_eps_) {
        if (live.second.holding_grants->load(std::memory_order_acquire))
          holding = true;
        else
          live.first->Wake();
      }
    }
    if (!holding) break;
    if (std::chrono::steady_clock::now() >= drain_until) {
      stop_drains_cut_.fetch_add(1, std::memory_order_relaxed);
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  // Wake every Serve thread still in WaitComp, then join them all so no
  // handler call can race the owner's destruction after Stop() returns.
  std::vector<Conn> conns;
  {
    std::lock_guard<std::mutex> lk(conn_mu_);
    for (const auto& live : live_eps_) live.first->Wake();
    conns.swap(conns_);
  }
  for (auto& c : conns) if (c.th.joinable()) c.th.join();
  // Every connection has put its windowed ranges into quarantine: wait it
  // out (each at most DFKV_RDMA_SERVER_MW_QUARANTINE_MAX_MS) and release
  // them, so their pins and leases are back before the store stops.
  if (const uint64_t last = MwQuarantineEndUs(true); last != 0) {
    const uint64_t now = SteadyUs();
    if (last > now)
      std::this_thread::sleep_for(std::chrono::microseconds(last - now));
  }
  MwReleaseQuarantine(true);
  // Every Serve thread has ended and unwatched its session.
  if (session_thread_.joinable()) {
    const uint64_t one = 1;
    ssize_t n = ::write(session_stop_fd_, &one, sizeof(one));
    (void)n;
    session_thread_.join();
    ::close(session_epoll_fd_);
    ::close(session_stop_fd_);
    session_epoll_fd_ = session_stop_fd_ = -1;
  }
}

void RdmaServer::SessionWatchLoop() {
  epoll_event events[64];
  for (;;) {
    const int n = ::epoll_wait(session_epoll_fd_, events, 64, -1);
    if (n < 0) {
      if (errno == EINTR) continue;
      return;
    }
    for (int i = 0; i < n; ++i) {
      const uint64_t id = events[i].data.u64;
      if (id == 0) return;  // stop
      std::lock_guard<std::mutex> lock(session_mu_);
      const auto found = sessions_.find(id);
      if (found == sessions_.end()) continue;  // unwatched meanwhile
      // The client's end closed: it has dropped the connection.
      session_peer_closed_.fetch_add(1, std::memory_order_relaxed);
      found->second->Wake();
      sessions_.erase(found);
    }
  }
}

uint64_t RdmaServer::WatchSession(int fd, rdma::RcEndpoint* ep) {
  std::lock_guard<std::mutex> lock(session_mu_);
  const uint64_t id = next_session_id_++;
  epoll_event event{};
  // Not EPOLLIN: the client sends nothing on it; only its close matters.
  event.events = EPOLLRDHUP | EPOLLONESHOT;
  event.data.u64 = id;
  if (::epoll_ctl(session_epoll_fd_, EPOLL_CTL_ADD, fd, &event) != 0) return 0;
  sessions_.emplace(id, ep);
  return id;
}

void RdmaServer::UnwatchSession(uint64_t id, int fd) {
  std::lock_guard<std::mutex> lock(session_mu_);
  sessions_.erase(id);
  ::epoll_ctl(session_epoll_fd_, EPOLL_CTL_DEL, fd, nullptr);
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

void RdmaServer::AcceptLoop() {
  while (running_) {
    int fd = ::accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) {
      const int err = errno;
      if (!running_) break;
      accept_guard_.OnAcceptError(err, running_);
      continue;
    }
    // Anchored at accept: time spent waiting for the thread counts against the
    // peer's budget.
    const auto bootstrap_deadline =
        std::chrono::steady_clock::now() +
        std::chrono::milliseconds(bootstrap_read_ms_);
    // Refuse before spending a thread. Only connections whose bootstrap frames
    // are still outstanding count, so established connections, however many,
    // never use this limit up.
    if (max_pending_handshakes_ != 0 &&
        pending_handshakes_.load(std::memory_order_relaxed) >=
            max_pending_handshakes_) {
      rejected_connections_.fetch_add(1, std::memory_order_relaxed);
      accept_guard_.LogRejected(
          "refusing connection: " + std::to_string(max_pending_handshakes_) +
          " connections have not sent their bootstrap frames "
          "(DFKV_RDMA_MAX_PENDING_HANDSHAKES)");
      ::close(fd);
      continue;
    }
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    std::lock_guard<std::mutex> lk(conn_mu_);
    if (!running_) { ::close(fd); break; }
    ReapDoneLocked();  // reap connections that finished since the last accept
    auto done = std::make_shared<std::atomic<bool>>(false);
    pending_handshakes_.fetch_add(1, std::memory_order_relaxed);
    std::thread serving;
    if (!net::StartThread(&serving, [this, fd, done, bootstrap_deadline] {
          NameThisThread("rdma-serve");
          Serve(fd, bootstrap_deadline, &pending_handshakes_);
          done->store(true, std::memory_order_release);  // last act
        })) {
      pending_handshakes_.fetch_sub(1, std::memory_order_relaxed);
      accept_guard_.OnThreadStartFailure();
      ::close(fd);
      continue;
    }
    conns_.push_back({std::move(serving), done});
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
  envparse::U64Spec spec;
  spec.representable_max = envparse::kSignedMax;  // read with strtol
  spec.legacy = envparse::Legacy::kPrefixOrZero;
  spec.fallback = out;
  spec.min = 1;
  spec.max = 256;
  return static_cast<size_t>(envparse::EnvU64("DFKV_RDMA_DEPTH", spec));
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
  const int out = RdmaIdleMsFromEnv();  // default 10 min
  config_dump::RecordResolved("DFKV_RDMA_IDLE_MS", std::to_string(out));
  return out;
}

int StopDrainMs() {
  const int out = RdmaStopDrainMsFromEnv();  // default 2 s
  config_dump::RecordResolved("DFKV_RDMA_STOP_DRAIN_MS",
                              std::to_string(out < 0 ? 0 : out));
  return out;
}

int ServerGrantTimeoutMs() {
  const int out = RdmaGrantTimeoutMsFromEnv();  // default 60 s
  config_dump::RecordResolved("DFKV_RDMA_GRANT_TIMEOUT_MS",
                              std::to_string(out < 0 ? 0 : out));
  return out;
}

#ifdef DFKV_WITH_URING
// Per-connection disk queue depth. It is independent of negotiated RDMA depth:
// requests beyond this bound remain prepared but unsubmitted until a CQE frees
// capacity. Sixteen is enough to expose the SSD queue on the measured hosts;
// operators may select 32 (or another [1, 256] value) independently.
size_t UringDepth() {
  size_t out = 16;
  envparse::U64Spec spec;
  spec.representable_max = envparse::kSignedMax;  // read with strtol
  spec.legacy = envparse::Legacy::kPrefixOrZero;
  spec.fallback = out;
  spec.min = 1;
  spec.max = 256;
  out = static_cast<size_t>(envparse::EnvU64("DFKV_SERVER_URING_DEPTH", spec));
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
  const bool out = ServerUringFromEnv();
  config_dump::RecordResolved("DFKV_SERVER_URING", out ? "on" : "off");
  return out;
#else
  return false;
#endif
}

// The idle connections a pressured request may evict, handed to the pure
// planner (rdma::PlanPressureVictims) with the pool as it is now. A
// connection qualifies when it is quiescent and past the one-second grace,
// or never got a request in 10 s (a client that gave up after the
// handshake).
std::vector<rdma::RcEndpoint*> RdmaServer::PlanPressureVictimsLocked(
    size_t bytes, int rail) {
  std::vector<rdma::PressureCandidate> candidates;
  std::vector<rdma::RcEndpoint*> owners;
  const uint64_t now = SteadyUs();
  constexpr uint64_t kPressureIdleGraceUs = 1000000;
  constexpr uint64_t kNeverServedUs = 10000000;
  for (const auto& [ep, live] : live_eps_) {
    const uint64_t active = ep->last_active_us_.load(std::memory_order_relaxed);
    // A newly inserted endpoint has not stamped its activity yet.
    if (active == 0 || active > now || now - active < kPressureIdleGraceUs)
      continue;
    if (!live.reclaimable->load(std::memory_order_acquire) &&
        (live.served_any->load(std::memory_order_acquire) ||
         now - active < kNeverServedUs))
      continue;
    if (live.lease_segment == nullptr) continue;
    candidates.push_back(rdma::PressureCandidate{
        live.lease_segment, live.lease_offset, live.recv_lease_bytes, active});
    owners.push_back(ep);
  }
  if (candidates.empty()) return {};
  std::vector<rdma::RcEndpoint*> victims;
  for (const size_t i : rdma::PlanPressureVictims(recv_segments_.Snapshot(),
                                                  candidates, bytes, rail))
    victims.push_back(owners[i]);
  return victims;
}

ibv_mr* RdmaServer::MwRegionFor(size_t rail, const char* at, size_t len,
                                const rdma::RecvSegmentPool::Lease& lease) {
  const auto inside = [&](const char* base, size_t size) {
    return at >= base && len <= size &&
           static_cast<size_t>(at - base) <= size - len;
  };
  if (mw_arena_ && inside(mw_arena_, mw_arena_bytes_)) {
    // Values up to an extent never cross one.
    const size_t extent = static_cast<size_t>(at - mw_arena_) / mw_extent_bytes_;
    char* base = mw_arena_ + extent * mw_extent_bytes_;
    const size_t size = std::min(mw_extent_bytes_,
                                 mw_arena_bytes_ - extent * mw_extent_bytes_);
    if (!inside(base, size)) return nullptr;
    return mw_regions_.Find(rail, base, size,
                            rdma::BindRegionRegistry::Kind::kArena);
  }
  const rdma::RecvSegment* segment = lease ? lease.segment() : nullptr;
  if (segment && inside(segment->data(), segment->size()))
    return mw_regions_.Find(rail, segment->data(), segment->size(),
                            rdma::BindRegionRegistry::Kind::kChunk);
  return nullptr;
}

uint64_t RdmaServer::MwQuarantineUsFor(uint64_t rail_bytes) const {
  return rdma::MwQuarantineUs(rail_bytes, mw_rate_mbps_, mw_floor_us_,
                              mw_margin_us_);
}

bool RdmaServer::MwQuarantineAllows(uint64_t rail_bytes,
                                    uint64_t held_bytes) const {
  return MwQuarantineUsFor(rail_bytes) <= mw_max_us_ &&
         mw_quarantine_bytes_.load(std::memory_order_relaxed) + held_bytes <=
             mw_quarantine_cap_;
}

void RdmaServer::MwQuarantine(MwQuarantined entry) {
  // The admission (MwQuarantineAllows) took the decision; more grants on
  // the rail since then do not stretch it past the maximum.
  const uint64_t until =
      SteadyUs() + std::min(MwQuarantineUsFor(entry.rail_bytes), mw_max_us_);
  const char* at = entry.at;
  bool earliest = false;
  {
    std::lock_guard<std::mutex> lock(mw_quarantine_mu_);
    mw_quarantine_bytes_.fetch_add(entry.bytes, std::memory_order_relaxed);
    const auto it = mw_quarantine_.emplace(until, std::move(entry));
    earliest = it == mw_quarantine_.begin();
  }
  mw_quarantined_total_.fetch_add(1, std::memory_order_relaxed);
  if (mw_event_for_test_) mw_event_for_test_("quarantined", at);
  if (earliest) {
    // The trim thread may be asleep until a later end.
    std::lock_guard<std::mutex> lock(recv_trim_mu_);
    recv_trim_cv_.notify_all();
  }
}

void RdmaServer::MwReleaseQuarantine(bool all) {
  std::vector<MwQuarantined> ended;
  {
    std::lock_guard<std::mutex> lock(mw_quarantine_mu_);
    const uint64_t now = SteadyUs();
    auto it = mw_quarantine_.begin();
    while (it != mw_quarantine_.end() && (all || it->first <= now)) {
      mw_quarantine_bytes_.fetch_sub(it->second.bytes,
                                     std::memory_order_relaxed);
      ended.push_back(std::move(it->second));
      it = mw_quarantine_.erase(it);
    }
  }
  for (auto& entry : ended) {
    if (mw_event_for_test_) mw_event_for_test_("released", entry.at);
    entry.read.Commit(Status::kOk, entry.data_len, entry.read_elapsed_sec);
  }
}

uint64_t RdmaServer::MwQuarantineEndUs(bool latest) {
  std::lock_guard<std::mutex> lock(mw_quarantine_mu_);
  if (mw_quarantine_.empty()) return 0;
  return latest ? mw_quarantine_.rbegin()->first
                : mw_quarantine_.begin()->first;
}

rdma::RecvSegmentPool::Lease RdmaServer::AllocateReceiveWithPressure(
    size_t bytes, int rail, int numa_node, uint64_t wait_us,
    rdma::RecvSegmentPool::LeaseClass lease_class) {
  auto lease = recv_segments_.Allocate(
      bytes, rdma::kV2DataOffset, rail, numa_node, lease_class);
  if (lease || wait_us == 0) return lease;
  // No eviction can make room for more than the budget.
  if (bytes > recv_segments_.max_bytes()) {
    pressure_unsatisfiable_.fetch_add(1, std::memory_order_relaxed);
    return lease;
  }

  // A pooled peer may post its next (possibly non-replay-safe PUT) operation
  // immediately after receiving a reply, before we poll its new CQE. Preserve
  // a one-second pressure-only idle grace, the interval that passed the
  // short-process hardware run, in addition to the in-flight ownership gate.
  const uint64_t started = SteadyUs();
  for (int round = 0; round < 32 && !lease; ++round) {
    if (SteadyUs() - started >= wait_us || !running_) break;
    size_t evicted = 0;
    {
      std::lock_guard<std::mutex> lock(conn_mu_);
      const std::vector<rdma::RcEndpoint*> victims =
          PlanPressureVictimsLocked(bytes, rail);
      if (victims.empty()) {
        // Nobody to evict that would help: fail now, not after a wait.
        if (round == 0)
          pressure_unsatisfiable_.fetch_add(1, std::memory_order_relaxed);
        break;
      }
      for (rdma::RcEndpoint* victim : victims) {
        live_eps_.erase(victim);  // claim before Wake: other scans skip it
        // The Serve thread must erase under conn_mu_ before destroying its QP.
        victim->Wake();
      }
      evicted = victims.size();
    }
    segment_evictions_.fetch_add(evicted, std::memory_order_relaxed);
    // The victim's QP destruction must fence inbound DMA before its lease is
    // released. Give it bounded time to unwind, then try another idle QP.
    for (int i = 0; i < 100 && !lease; ++i) {
      if (SteadyUs() - started >= wait_us || !running_) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      lease = recv_segments_.Allocate(
          bytes, rdma::kV2DataOffset, rail, numa_node, lease_class);
    }
  }
  return lease;
}

void RdmaServer::Serve(
    int boot_fd, std::chrono::steady_clock::time_point bootstrap_deadline,
    std::atomic<uint64_t>* pending) {
  // Leaves the pending count once the peer's frames are in, or on any earlier
  // return.
  struct PendingHandshake {
    std::atomic<uint64_t>* count;
    void Release() {
      if (count) count->fetch_sub(1, std::memory_order_relaxed);
      count = nullptr;
    }
    ~PendingHandshake() { Release(); }
  } pending_handshake{pending};
  {
    std::lock_guard<std::mutex> lock(handshake_mu_);
    if (handshakes_closed_) {
      ::close(boot_fd);
      return;
    }
    handshake_fds_.insert(boot_fd);
  }
  auto close_boot = [&] {
    {
      std::lock_guard<std::mutex> lock(handshake_mu_);
      handshake_fds_.erase(boot_fd);
    }
    ::close(boot_fd);
  };
  // Both reads below share one absolute deadline: a per-recv timeout would let
  // a peer that sends a byte every few seconds hold this thread for minutes.
  const timeval no_per_recv_cap{0, 0};
  // Bootstrap: client first names the device it wants us to use (same rail for
  // multi-rail); fall back to our configured default if it sends an empty name.
  char devbuf[rdma::kDevNameBytes];
  if (!net::ReadAllWithin(boot_fd, devbuf, rdma::kDevNameBytes,
                          no_per_recv_cap, bootstrap_deadline)) {
    close_boot();
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
      close_boot();
      return;
    }
    char proof[rdma::kV2RetireProofBytes];
    rdma::EncodeV2RetireProof(token, proof);
    net::WriteAll(boot_fd, proof, sizeof(proof));
    close_boot();
    return;
  }
  // Capability probes are answered without creating a QP.
  if (rdma::IsV2Probe(devbuf)) {
    // Its frames are in: give the slot back before answering, so the
    // client's bootstrap, sent as soon as it reads the answer, finds it free.
    pending_handshake.Release();
    char reply[rdma::kV2ProbeReplyBytes];
    rdma::EncodeV2ProbeReply(
        reply, rdma::kV2ProbeCapWriterRetirement | rdma::kV2ProbeCapPullRead |
                   rdma::kV2ProbeCapLeasedPut | rdma::kV2ProbeCapDynamicPull |
                   rdma::kV2ProbeCapDynamicOnly |
                   (read_depth_offered_ ? rdma::kV2ProbeCapReadDepth : 0) |
                   (session_fd_enabled_ ? rdma::kV2ProbeCapSessionFd : 0));
    net::WriteAll(boot_fd, reply, sizeof(reply));
    if (after_probe_reply_for_test_) after_probe_reply_for_test_();
    close_boot();
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
  // Kept only for a client that asked, and only on the dynamic-only path.
  const bool keep_session = session_fd_enabled_ && dynamic_only_requested &&
                            rdma::DevFrameRequestsSessionFd(devbuf);
  // Confirmed only to a client that asked, and only by a server whose every
  // rail accepts that many (the probe advertised it on the same condition).
  const bool ack_read_depth = read_depth_offered_ && dynamic_only_requested &&
                              rdma::DevFrameRequestsReadDepth(devbuf);
  const uint64_t declared = rdma::ParseDevFrameMaxBlock(devbuf);
  if (rdma::ParseDevFrameProtocol(devbuf) != rdma::kDevProtoV2 ||
      declared == 0 || !dynamic_pull_requested || !pull_read_requested ||
      (dynamic_only_requested ? writer_retirement_requested
                              : !writer_retirement_requested)) {
    DFKV_LOG_ERROR("rdma: rejecting peer without required v2 negotiation");
    close_boot();
    return;
  }
  constexpr uint8_t wire_epoch = kNativeProtoRdmaV2;
  constexpr size_t response_prefix = kRespPrefix;
  if (declared > static_cast<uint64_t>(max_msg_)) {
    DFKV_LOG_ERROR("rdma: client declared max block " + std::to_string(declared) +
                   "B, above this server's cap " + std::to_string(max_msg_) +
                   "B; refusing the connection. Raise --max-msg or lower "
                   "DFKV_RDMA_MAX_BLOCK_BYTES.");
    close_boot();
    return;
  }
  const size_t conn_max =
      std::max<size_t>(wire_limits::kIoAlign,
                       std::min<size_t>(declared, max_msg_));
  devbuf[rdma::kDevNameBytes - 1] = '\0';
  std::string dev =
      devbuf[0] ? std::string(devbuf) : anchor_devs_[DefaultRail()];
  const auto rail_it =
      std::find(anchor_devs_.begin(), anchor_devs_.end(), dev);
  if (rail_it == anchor_devs_.end()) {
    DFKV_LOG_ERROR("rdma: client requested device outside the fixed topology: " +
                   dev);
    close_boot();
    return;
  }
  const size_t rail_index =
      static_cast<size_t>(std::distance(anchor_devs_.begin(), rail_it));
  // A rail whose port is not ACTIVE: close before replying, as for a busy
  // server. The client pauses and tries once more, then cools this rail of
  // the peer and goes to another, instead of failing each operation on a QP
  // that cannot carry it after RC retries.
  if (rail_port_state_ &&
      rail_port_state_[rail_index].load(std::memory_order_acquire) == 0) {
    inactive_rail_refusals_.fetch_add(1, std::memory_order_relaxed);
    close_boot();
    return;
  }
  RailStats& rail_stats = *rail_stats_[rail_index];
  const int rail_numa = anchors_[rail_index]->numa_node();
  // Resolved once per real connection, while the bootstrap socket is still
  // open, so a later failed completion can be attributed to its peer.
  const std::string peer_addr = net::PeerAddress(boot_fd);
  const uint64_t accepted_us = SteadyUs();

  // The client sends QpInfo first. Read and validate its mandatory v2 depth
  // before allocating per-connection control slots or leasing shared receive
  // space; a client advertising depth 1 must consume one slot, not ServerDepth.
  char peer[rdma::kQpInfoBytes];
  if (!net::ReadAllWithin(boot_fd, peer, sizeof(peer), no_per_recv_cap,
                          bootstrap_deadline)) {
    close_boot();
    return;
  }
  pending_handshake.Release();
  const rdma::QpInfo peer_info = rdma::ParseQpInfo(peer);
  if (peer_info.protocol_version != rdma::kDevProtoV2 ||
      peer_info.depth == 0) {
    DFKV_LOG_ERROR("rdma: rejecting incompatible v2 QP negotiation");
    close_boot();
    return;
  }
  const size_t K = std::min<size_t>(ServerDepth(), peer_info.depth);
  const size_t slot_size = rdma::V2SlotSize(conn_max);
  if (K == 0 || slot_size == 0 ||
      K > std::numeric_limits<size_t>::max() / slot_size) {
    close_boot();
    return;
  }
  // Declared ahead of every per-connection resource so that it unwinds after
  // all of them: by the time dfkv_rdma_active_conns drops, this connection's
  // endpoint, receive lease, source pins and segment accounting are released.
  // Dropping the gauge first let an observer see "no connections" while the
  // connection still held memory.
  struct ActiveConnection {
    std::atomic<uint64_t>* server = nullptr;
    std::atomic<uint64_t>* rail = nullptr;
    void Enter(std::atomic<uint64_t>* server_gauge,
               std::atomic<uint64_t>* rail_gauge) {
      server = server_gauge;
      rail = rail_gauge;
      server->fetch_add(1, std::memory_order_relaxed);
      rail->fetch_add(1, std::memory_order_relaxed);
    }
    ~ActiveConnection() {
      if (server == nullptr) return;
      rail->fetch_sub(1, std::memory_order_relaxed);
      server->fetch_sub(1, std::memory_order_relaxed);
    }
  } active_connection;
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
    close_boot();
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
    // A memory window, not an MR, authorizes the client's READ (the
    // slot's own window, under rkey). Revoking: its invalidation is posted
    // and not yet complete; the slot stays busy until it is.
    bool windowed = false;
    bool revoking = false;
    uint32_t rkey = 0;
    const char* read_at = nullptr;  // the range the window covers
    RdmaServer* server = nullptr;
    // This grant's bytes in the rail's outstanding GET-grant bytes.
    std::atomic<uint64_t>* rail_grant_bytes = nullptr;
    uint64_t granted_bytes = 0;
    // B5-3 zero-copy pull: pinned arena source handed to the peer instead of
    // a staged copy. Empty for staged pulls; the send pin releases exactly
    // when the slot resets (PullRelease or connection teardown).
    PreparedRead arena_read;
    double read_elapsed_sec = 0.0;
    std::atomic<uint64_t>* active_count = nullptr;
    std::atomic<uint64_t>* active_bytes = nullptr;
    ~PullSlotState() {
      // The endpoint, its QP and its windows are gone by now (it is
      // declared after the slots): what a window covered still waits out
      // its quarantine.
      Quarantine();
      Reset();
    }
    // A windowed range leaves the slot for the quarantine, with what holds
    // it (the staging lease, the arena pin); its window is known dead.
    void Quarantine() {
      if (!windowed) return;
      MwQuarantined entry;
      entry.bytes = lease ? lease.size() : std::max<size_t>(data_len, 1);
      if (lease) {
        active_bytes->fetch_sub(lease.size(), std::memory_order_relaxed);
        active_count->fetch_sub(1, std::memory_order_relaxed);
      }
      entry.lease = std::move(lease);
      entry.read = std::move(arena_read);
      entry.data_len = data_len;
      entry.read_elapsed_sec = read_elapsed_sec;
      entry.at = read_at;
      entry.rail_bytes = rail_grant_bytes->load(std::memory_order_relaxed);
      windowed = false;
      revoking = false;
      server->MwQuarantine(std::move(entry));
    }
    // The staging range alone, when the client reads the value from
    // elsewhere (the RAM arena).
    void ReleaseStaging() {
      if (!lease) return;
      active_bytes->fetch_sub(lease.size(), std::memory_order_relaxed);
      active_count->fetch_sub(1, std::memory_order_relaxed);
      lease.Reset();
    }
    void Reset() {
      arena_read = PreparedRead{};
      if (lease) {
        active_bytes->fetch_sub(lease.size(), std::memory_order_relaxed);
        active_count->fetch_sub(1, std::memory_order_relaxed);
        lease.Reset();
      }
      mr = nullptr;
      windowed = false;
      revoking = false;
      rkey = 0;
      read_at = nullptr;
      if (granted_bytes != 0)
        rail_grant_bytes->fetch_sub(granted_bytes, std::memory_order_relaxed);
      granted_bytes = 0;
      busy = false;
      data_len = 0;
      value_len = 0;
    }
  };
  // When this connection last sent a GET's READY or a PUT lease.
  uint64_t last_grant_sent_us = 0;
  std::vector<PullSlotState> pull_slots(K);
  for (auto& state : pull_slots) {
    state.active_count = &dynamic_get_active_;
    state.active_bytes = &dynamic_get_bytes_active_;
    state.server = this;
    state.rail_grant_bytes = &rail_grant_bytes_[rail_index];
  }
  // Slots whose window's invalidation is posted and not yet complete.
  size_t revoking_slots = 0;
  // Per send slot: the pull slot whose invalidation precedes the reply
  // posted from it (K: none). The invalidation is unsignalled; the reply's
  // successful completion, after it on the send queue, means it completed.
  std::vector<size_t> revoke_on_send(K, K);
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
  // Memory windows, one per pull slot, when the rail offers them.
  const bool want_windows = mw_enabled_ && rail_index < rail_mw_.size() &&
                            rail_mw_[rail_index] != 0;
  if (want_windows) ep.set_window_slots(K);
  if (!ep.Open(dev.empty() ? nullptr : dev.c_str(), conn_control, K,
               /*ib_port=*/1, /*direct_io_buffers=*/false, conn_max)) {
    close_boot();
    return;
  }
  const bool conn_windows = want_windows && ep.AllocWindows(K);
  if (want_windows && !conn_windows)
    mw_fallback_connections_.fetch_add(1, std::memory_order_relaxed);
  // The connection's own receive slots, and nothing else, are writable by
  // its client: an rkey over the whole chunk let one connection write into
  // another's slots. Zeroed first, so no bytes of a previous connection can
  // be read back through a PUT that does not cover what it claims.
  std::memset(recv_lease.data(), 0, recv_lease.size());
  ibv_mr* recv_segment_mr =
      ep.RegisterOwnRegion(recv_lease.data(), recv_lease.size());
  if (!recv_segment_mr) {
    DFKV_LOG_ERROR("rdma v2: receive-segment MR unavailable on device " +
                   (dev.empty() ? std::string("(auto)") : dev));
    close_boot();
    return;
  }
  DFKV_LOG_INFO("rdma conn: protocol=v2 declared=" +
                std::to_string(declared) +
                " control=" + std::to_string(conn_control) +
                " shared-slot=" + std::to_string(slot_size) +
                " qd=" + std::to_string(K) +
                " qp=" + std::to_string(ep.Local().qpn) +
                " peer_qp=" + std::to_string(peer_info.qpn) +
                " dev=" + dev + " peer=" + peer_addr);
  numa::PinThreadToNode(ep.numa_node());

  // QP bootstrap: the peer geometry was consumed before allocation above.
  // Advertise this endpoint's clamped depth, then connect the correctly sized
  // QP; neither side may infer legacy defaults.
  char mine[rdma::kQpInfoBytes];
  rdma::QpInfo my = ep.Local();
  my.depth = static_cast<uint16_t>(std::min<size_t>(K, 256));
  my.protocol_version = rdma::kDevProtoV2;
  rdma::SerializeQpInfo(my, mine);
  // Accept a client's pipelined READs (advertised by kV2ProbeCapReadDepth).
  // Responder resources only: a client that asks for one READ at a time is
  // served as before.
  ep.set_responder_read_depth(rdma::kV2ReadDepth);
  if (const int delay_ms = bootstrap_reply_delay_ms_for_test_.load(
          std::memory_order_relaxed))
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
  if (!net::WriteAll(boot_fd, mine, rdma::kQpInfoBytes) ||
      !ep.Connect(peer_info)) {
    close_boot();
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
    close_boot();
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
          ? rdma::EncodeV2DynamicOnlyReadiness(info, readiness, keep_session,
                                               ack_read_depth)
          : rdma::EncodeV2Readiness(info, legacy_token, readiness);
  const bool ok =
      readiness_bytes != 0 &&
      net::WriteAll(boot_fd, readiness, readiness_bytes);
  // The session channel: the bootstrap socket, kept. Closed when this
  // connection ends, whatever ends it, which tells the client.
  struct SessionChannel {
    RdmaServer* server;
    int fd = -1;
    uint64_t id = 0;
    ~SessionChannel() {
      if (fd < 0) return;
      if (id != 0) server->UnwatchSession(id, fd);
      ::close(fd);
    }
  } session{this};
  if (ok && keep_session) {
    {
      std::lock_guard<std::mutex> lock(handshake_mu_);
      handshake_fds_.erase(boot_fd);
    }
    session.fd = boot_fd;
    sessions_kept_.fetch_add(1, std::memory_order_relaxed);
  } else {
    close_boot();
  }
  if (!ok) {
    return;
  }
  v2_conns_.fetch_add(1, std::memory_order_relaxed);

  // Fresh QPs have advertised readiness but may already have a first request
  // in transit. Reclaim only after at least one completed request/reply fence
  // has established a genuine quiescent point.
  std::atomic<bool> reclaimable{false};
  std::atomic<bool> served_any{false};
  std::atomic<bool> holding_grants{false};
  // Register this endpoint so Stop() can Wake() us out of WaitComp and join. The
  // running_ check under conn_mu_ closes the race with a concurrent Stop(): either
  // Stop sees us in live_eps_ (and wakes us) or we see running_==false here.
  if (before_live_registration_for_test_) before_live_registration_for_test_();
  {
    std::lock_guard<std::mutex> lk(conn_mu_);
    if (!running_) {
      return;
    }
    const rdma::RecvSegment* segment = recv_lease.segment();
    live_eps_.emplace(
        &ep, LiveEndpoint{&reclaimable, recv_lease.size(), segment,
                          static_cast<size_t>(recv_lease.data() -
                                              segment->data()),
                          &served_any,
                          &holding_grants, rail_index});
    // Its rail may have gone down after the check above, and the watcher
    // may have woken the rail's connections before this one was among
    // them: it ends itself. (A later wake finds it registered.)
    if (rail_down_confirmed_ &&
        rail_down_confirmed_[rail_index].load(std::memory_order_acquire)) {
      ep.Wake();
      rail_down_connections_ended_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  // From now on a close of the client's end wakes this thread, which ends
  // the connection (WaitComp returns -1, as for Stop).
  if (session.fd >= 0) session.id = WatchSession(session.fd, &ep);
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
    std::chrono::steady_clock::time_point batch_arrival{};
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
    // A leased-PUT request names the OBJECT size, not an inline payload: it
    // carries no payload of its own, and an object above the process-wide
    // payload ceiling is refused with an answer (build_reply), not by
    // dropping the connection. Every other control op keeps the conn_max
    // bound.
    const uint64_t send_max_payload =
        static_cast<uint8_t>(frame[1]) == static_cast<uint8_t>(WireOp::kLeasePut)
            ? std::numeric_limits<uint64_t>::max()
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
      // only needs to prove the slot has no in-flight lease/multi-window
      // state. An object larger than --max-msg is answered kInvalid.
      // Which lease slot it gets is build_reply's choice, among the slots
      // free in the data-slot (immediate) space the windows will name.
      if (!leased_put_requested || request->fields.payload_len == 0)
        return false;
      return true;
    }
    if (request->fields.op == static_cast<uint8_t>(WireOp::kPullRelease)) {
      return completion.byte_len == kReqPrefix;
    }



    // A control op arrives in a receive buffer of its own: a multi-window
    // PUT in progress (named by data slot, a different index space) does not
    // concern it.
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

  // Authorizes the client's READ of [at, at+len) for a GET grant on `slot`:
  // the slot's window bound to it, when this connection has windows, the
  // range lies in a registered region and its quarantine fits; an exact MR
  // otherwise, as before. False if neither could be had.
  constexpr uint64_t kMwWrTag = uint64_t{1} << 63;  // a bind or invalidation
  constexpr uint64_t kMwInvalidate = uint64_t{1} << 62;
  auto authorize_read = [&](size_t slot, const char* at, size_t len,
                            uint32_t* rkey) -> bool {
    PullSlotState& state = pull_slots[slot];
    const uint64_t started_ns = SteadyNs();
    const size_t bytes = std::max<size_t>(len, 1);
    bool ok = false;
    if (conn_windows) {
      const uint64_t held = state.lease ? state.lease.size() : bytes;
      const uint64_t rail_bytes =
          state.rail_grant_bytes->load(std::memory_order_relaxed) + bytes;
      ibv_mr* region = MwRegionFor(rail_index, at, bytes, state.lease);
      if (region == nullptr) {
        mw_fallback_grants_.fetch_add(1, std::memory_order_relaxed);
      } else if (!MwQuarantineAllows(rail_bytes, held)) {
        mw_quarantine_fallbacks_.fetch_add(1, std::memory_order_relaxed);
      } else if (ep.PostBindWindow(slot, region, at, bytes, kMwWrTag | slot,
                                   rkey)) {
        state.windowed = true;
        state.rkey = *rkey;
        state.read_at = at;
        mw_binds_.fetch_add(1, std::memory_order_relaxed);
        ok = true;
      } else {
        mw_fallback_grants_.fetch_add(1, std::memory_order_relaxed);
      }
    }
    if (!ok) {
      state.mr = ep.RegisterLeaseReadRegion(const_cast<char*>(at), bytes);
      if (state.mr) {
        *rkey = state.mr->rkey;
        ok = true;
      }
    }
    if (ok) {
      state.granted_bytes = bytes;
      state.rail_grant_bytes->fetch_add(bytes, std::memory_order_relaxed);
    }
    grant_register_ns_.fetch_add(SteadyNs() - started_ns,
                                 std::memory_order_relaxed);
    grant_registers_.fetch_add(1, std::memory_order_relaxed);
    return ok;
  };
  // A windowed slot's invalidation completed: its range goes into
  // quarantine and the slot is free for the next request.
  auto retire_windowed = [&](size_t slot) {
    PullSlotState& state = pull_slots[slot];
    if (mw_event_for_test_) mw_event_for_test_("invalidated", state.read_at);
    state.Quarantine();
    state.Reset();
    --revoking_slots;
    ++state.generation;
    if (state.generation == 0) ++state.generation;
  };
  // A reply's SEND completed: an invalidation posted ahead of it has too.
  auto on_send_completion = [&](size_t send_slot) {
    const size_t slot = revoke_on_send[send_slot];
    if (slot == K) return;
    revoke_on_send[send_slot] = K;
    retire_windowed(slot);
  };
  auto note_window_failure = [&](const ibv_wc& wc) {
    if ((wc.wr_id & kMwWrTag) != 0 && (wc.wr_id & kMwInvalidate) != 0)
      mw_invalidation_failures_.fetch_add(1, std::memory_order_relaxed);
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
    // A whole value read from the disk is checked before anything uses it
    // (log engine, design §5.3): on a mismatch it is a miss for the
    // client, the coalescer's followers and the RAM tier alike.
    if (!prepared.Verify()) {
      prepared.Commit(Status::kNotFound, 0, elapsed_sec);
      state.Reset();
      EncodeRespVersion(ep.sbuf(send_slot), wire_epoch, Status::kNotFound, 0, 0);
      reply->first_len = response_prefix;
      return true;
    }
    if (request.fields.offset > value_len ||
        output_len > request.fields.length ||
        output_len > value_len - request.fields.offset ||
        (output_len != 0 && output == nullptr))
      return error_reply(Status::kInvalid);
    // Empty results retain a revocable release identity, but authorize no
    // payload READ. The one registered sentinel byte is not a wire request.
    if (output_len == 0) output = state.lease.data();
    uint32_t rkey = 0;
    bool authorized = authorize_read(slot, output, output_len, &rkey);
    if (!authorized && prepared.source_registered() && prepared.Stage()) {
      output = prepared.data();
      authorized = authorize_read(slot, output, output_len, &rkey);
    }
    if (!authorized) return error_reply(Status::kIOError);
    // Served from outside the staging range (a coalesced follower reading
    // the arena, a RAM hit through the prepared read): the staging lease is
    // not needed until the RELEASE, so it goes back to the pool now.
    if (output_len != 0 && state.lease) {
      const auto at = reinterpret_cast<uintptr_t>(output);
      const auto begin = reinterpret_cast<uintptr_t>(state.lease.data());
      if (at < begin || at - begin >= state.lease.size())
        state.ReleaseStaging();
    }
    // Publish disk/coalescer completion before READY; followers must not wait
    // for the client's RELEASE. Its post-read hold stays owned by the grant.
    if (prepared.needs_io())
      prepared.Commit(Status::kOk, output_len, elapsed_sec);
    state.arena_read = std::move(prepared);
    state.read_elapsed_sec = elapsed_sec;
    state.busy = true;
    last_grant_sent_us = SteadyUs();
    state.data_len = output_len;
    state.value_len = value_len;
    const rdma::DynamicPullReady ready{
        static_cast<uint32_t>(slot), state.generation, output_len, value_len,
        reinterpret_cast<uint64_t>(output), rkey};
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
    uint32_t rkey = 0;
    if (fail_pinned_registration_for_test_ ||
        !authorize_read(slot, pinned.data(), pinned.payload_len(), &rkey)) {
      // Falls back to a staged copy. Worth counting: with the arena no longer
      // registered up front, a memlock limit or a device MR limit shows up
      // here, per read, instead of failing startup.
      pull_zerocopy_register_failures_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    state.arena_read = std::move(pinned);
    state.read_elapsed_sec = read_elapsed_sec;
    state.busy = true;
    last_grant_sent_us = SteadyUs();
    state.data_len = state.arena_read.payload_len();
    state.value_len = state.arena_read.value_len();
    const rdma::DynamicPullReady ready{
        static_cast<uint32_t>(slot), state.generation, state.data_len,
        state.value_len, reinterpret_cast<uint64_t>(state.arena_read.data()),
        rkey};
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
      if (!state.busy || (state.mr == nullptr && !state.windowed) ||
          state.revoking || state.generation != fields.length)
        return invalid_reply();
      if (state.windowed) {
        // Revoked on this QP ahead of the reply, which the send queue
        // orders after it; the range is reused only after the
        // invalidation's completion and a quarantine.
        if (!ep.PostLocalInvalidate(state.rkey,
                                    kMwWrTag | kMwInvalidate | slot)) {
          mw_invalidation_failures_.fetch_add(1, std::memory_order_relaxed);
          return false;  // the connection ends; teardown quarantines it
        }
        mw_invalidations_.fetch_add(1, std::memory_order_relaxed);
        state.revoking = true;
        ++revoking_slots;
        revoke_on_send[send_slot] = slot;
      } else {
        release_pull(slot);
      }
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
      // Stopping: no new grant, so that the drain only waits for the ones in
      // flight. A miss is an answer to every client version.
      if (!running_) {
        stop_grants_refused_.fetch_add(1, std::memory_order_relaxed);
        encode_status(Status::kNotFound, 0);
        reply->first_len = response_prefix;
        return true;
      }
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
      // A key the node does not have needs no staging: under receive-pool
      // pressure a miss used to wait for memory, or take another client's
      // idle connection, only to read nothing.
      if (lookup_handler_ && fields.length != 0 &&
          lookup_handler_(key) == Status::kNotFound) {
        unstaged_misses_.fetch_add(1, std::memory_order_relaxed);
        encode_status(Status::kNotFound, 0);
        reply->first_len = response_prefix;
        return true;
      }
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
      const uint64_t want = fields.payload_len;
      // An object this server will never take is a final answer to the
      // client, not a fault of the connection: dropping the connection
      // without a reply left the client to time out and cool a healthy node.
      // Stopping: no new lease (see the pull above); the client takes
      // kCacheFull as a refusal for want of room and fails this PUT alone.
      if (!running_) {
        stop_grants_refused_.fetch_add(1, std::memory_order_relaxed);
        encode_status(Status::kCacheFull, 0);
        reply->first_len = response_prefix;
        return true;
      }
      if (want > static_cast<uint64_t>(max_msg_)) {
        const uint64_t n =
            lease_put_oversize_rejects_.fetch_add(1, std::memory_order_relaxed);
        if (n == 0 || (n & 0x3FFu) == 0)
          DFKV_LOG_WARN("rdma: PUT of " + std::to_string(want) +
                        "B refused (kInvalid): above this server's --max-msg " +
                        std::to_string(max_msg_) + "B. peer=" + peer_addr +
                        " rejects=" + std::to_string(n + 1));
        return invalid_reply();
      }
      const size_t region = rdma::V2SlotSize(want);
      if (region == 0) return false;
      // The lease's slot is what its windows will name in their immediate,
      // the data-slot space a resident multi-window PUT uses too: one free
      // in both. The receive slot's own number when it is free (as before),
      // else the first free one; none free is backpressure, not a fault.
      auto lease_slot_free = [&](size_t d) {
        return !lease_put[d].active && !multi_put[d].active;
      };
      size_t slot = request.recv_slot;
      if (!lease_slot_free(slot)) {
        slot = K;
        for (size_t d = 0; d < K && slot == K; ++d)
          if (lease_slot_free(d)) slot = d;
      }
      if (slot == K) {
        lease_put_busy_rejects_.fetch_add(1, std::memory_order_relaxed);
        encode_status(Status::kCacheFull, 0);
        reply->first_len = response_prefix;
        return true;
      }
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
      last_grant_sent_us = SteadyUs();
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
          cache_cap, request.batch_arrival);
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

  // A request that arrives while this connection owes no reply (every
  // received request answered and every reply SEND completed) starts a
  // batch; the requests that arrive before that is true again belong to it.
  // The client sends a batch at once under one deadline, and this thread
  // serves it in order, so a PUT parked for a flush counts its wait from the
  // batch's arrival, not from its own turn. A reply's SEND completes before a
  // request the client sends after reading it arrives, so a later batch does
  // not join an earlier one.
  std::chrono::steady_clock::time_point batch_arrival{};
  auto note_arrival = [&](Request* request) {
    if (free_send.size() == K && pending_recv_count == 0)
      batch_arrival = std::chrono::steady_clock::now();
    request->batch_arrival = batch_arrival;
  };
  // A GET for a slot whose window is still being revoked (its RELEASE
  // reply's completion not yet polled: the client's next request and the
  // ACK of that reply are separate packets, and either may come first)
  // waits in the FIFO below until that completion retires the slot.
  auto waits_for_revocation = [&](const ibv_wc& wc) -> bool {
    if (revoking_slots == 0 || wc.opcode != IBV_WC_RECV ||
        wc.wr_id >= K || wc.byte_len != kReqPrefix + rdma::kPullPrepareBytes)
      return false;
    const char* frame = ep.rbuf(static_cast<size_t>(wc.wr_id));
    rdma::PullPrepareControl control;
    return static_cast<uint8_t>(frame[1]) ==
               static_cast<uint8_t>(WireOp::kPullRange) &&
           rdma::DecodePullPrepareControl(frame + kReqPrefix, &control) &&
           control.slot_index < K && pull_slots[control.slot_index].revoking;
  };
  auto dispatch_completions = [&](const ibv_wc* batch, int count,
                                  auto&& process_wc,
                                  auto&& has_async_work) -> bool {
    if (count != 0) {
      reclaimable.store(false, std::memory_order_release);
      served_any.store(true, std::memory_order_release);
    }
    for (int i = 0; i < count; ++i) {
      const ibv_wc& wc = batch[i];
      const bool received = wc.opcode == IBV_WC_RECV ||
                            wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM;
      if (wc.status == IBV_WC_SUCCESS && received &&
          (pending_recv_count != 0 || free_send.empty() ||
           waits_for_revocation(wc))) {
        if (pending_recv_count == K) return false;
        pending_recv[(pending_recv_head + pending_recv_count) % K] = wc;
        ++pending_recv_count;
      } else if (!process_wc(wc)) {
        return false;
      }
    }
    while (pending_recv_count != 0 && !free_send.empty() &&
           !waits_for_revocation(pending_recv[pending_recv_head])) {
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
  // A grant the client has yet to use (a GET's READY before its RELEASE, a
  // PUT lease before its window, a multi-window PUT before its next window)
  // keeps its pin or staging range for as long as the connection lives. A
  // client that crashed or gave up leaves no completion to wake on, so the
  // connection would live until idle_ms (10 min by default, forever with 0).
  // So while a grant is held and nothing is left for the server to do (no
  // disk read or reply of this connection in flight), the connection may
  // stay silent for grant_ms at most, counted from the later of its last
  // completion and the last grant sent: a batch's clock starts with its
  // last READY, whatever its first ones waited for. Past that it is torn
  // down, and the endpoint (QP and exact MRs) goes before any pin or range,
  // so nothing the peer could still read or write is freed under it.
  const int grant_ms = ServerGrantTimeoutMs();
  struct GrantView {
    bool holding = false;
    uint64_t deadline_us = 0;  // 0 = none
  };
  // One pass over the slots per loop iteration.
  auto grant_view = [&](bool server_work) {
    GrantView view;
    for (size_t i = 0; i < K && !view.holding; ++i)
      view.holding = pull_slots[i].busy || lease_put[i].active ||
                     multi_put[i].active;
    holding_grants.store(view.holding, std::memory_order_release);
    if (view.holding && !server_work && grant_ms >= 0) {
      const uint64_t since = std::max<uint64_t>(
          ep.last_active_us_.load(std::memory_order_relaxed),
          last_grant_sent_us);
      view.deadline_us = since + static_cast<uint64_t>(grant_ms) * 1000;
    }
    return view;
  };
  // The wait for completions: idle_ms, or less when a grant expires sooner.
  // *for_grant tells a timeout of the grant bound from an idle connection.
  auto wait_bound_ms = [&](const GrantView& view, bool* for_grant) -> int {
    *for_grant = false;
    if (view.deadline_us == 0) return idle_ms;
    const uint64_t now = SteadyUs();
    const int left = view.deadline_us > now
        ? static_cast<int>(std::min<uint64_t>(
              (view.deadline_us - now + 999) / 1000, 86400000))
        : 0;
    if (idle_ms >= 0 && idle_ms <= left) return idle_ms;
    *for_grant = true;
    return std::max(left, 1);
  };
  // After the wait: still silent past the deadline (a completion that just
  // arrived counts, as it moved last_active_us_).
  auto grant_expired = [&](const GrantView& view) -> bool {
    if (view.deadline_us == 0) return false;
    const uint64_t since = std::max<uint64_t>(
        ep.last_active_us_.load(std::memory_order_relaxed),
        last_grant_sent_us);
    if (SteadyUs() < since + static_cast<uint64_t>(grant_ms) * 1000)
      return false;
    const uint64_t n = grant_expiries_.fetch_add(1, std::memory_order_relaxed);
    if (n == 0 || (n & 0x3FFu) == 0)
      DFKV_LOG_WARN("rdma: a connection holding a grant stayed silent for " +
                    std::to_string(grant_ms) +
                    " ms (DFKV_RDMA_GRANT_TIMEOUT_MS); closing it. peer=" +
                    peer_addr + " dev=" + dev +
                    " expiries=" + std::to_string(n + 1));
    return true;
  };
  active_connection.Enter(&active_conns_, &rail_stats.active_conns);
  // Failed WCs do not guarantee a meaningful opcode. Retain the raw provider
  // fields and both QP numbers for correlation with short-lived clients, and
  // name the peer and the local device so the failure can be attributed. One
  // line per failed connection, never throttled: it fires once in a
  // connection's life and is the only record of how that connection failed
  // (vendor syndrome, failed slot, queue state).
  auto note_completion_error = [&](const ibv_wc& wc, const char* loop,
                                   size_t posted_sends) {
    completion_errors_.fetch_add(1, std::memory_order_relaxed);
    rail_stats.completion_errors.fetch_add(1, std::memory_order_relaxed);
    rail_stats.wc_errors[rdma::WcStatusSlot(wc.status)].fetch_add(
        1, std::memory_order_relaxed);
    // Retries ran out toward this peer: its IP address, once, for the
    // per-rail count of distinct addresses (many on one rail point at the
    // rail's path, one at that peer; pods of one host count separately).
    if (wc.status == IBV_WC_RETRY_EXC_ERR ||
        wc.status == IBV_WC_RNR_RETRY_EXC_ERR)
      rail_retry_peers_[rail_index]->Note(PeerHost(peer_addr), SteadyUs() / 1000);
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
        " posted_sends=" + std::to_string(posted_sends) +
        " status_name=" + rdma::WcStatusName(wc.status) +
        " dev=" + dev + " peer=" + peer_addr +
        " age_ms=" + std::to_string((SteadyUs() - accepted_us) / 1000));
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
    bool has_sync_fallbacks = false;
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
        note_window_failure(wc);
        note_completion_error(wc, "uring", posted_sends);
        return false;
      }
      if ((wc.wr_id & kMwWrTag) != 0) return false;  // never signalled
      if (wc.opcode == IBV_WC_SEND) {
        const size_t sid = static_cast<size_t>(wc.wr_id);
        if (sid >= K) return false;
        if (posted_sends == 0) return false;
        --posted_sends;
        uring_send_fences_.fetch_add(1, std::memory_order_relaxed);
        on_send_completion(sid);
        free_send.push_back(sid);
        return true;
      }

      Request request;
      if (!decode_request(wc, &request) || free_send.empty() ||
          next_sequence == std::numeric_limits<uint64_t>::max())
        return false;
      note_arrival(&request);
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
          !pull_slots[control.slot_index].busy && running_) {
        if (try_pinned_pull(qd.send_slot, request, control.slot_index,
                            &qd.reply)) {
          handled = true;
          qd.ready = true;
        } else if (lookup_handler_ &&
                   lookup_handler_(fields.Key()) == Status::kNotFound) {
          // As in build_reply: a miss needs no staging.
          unstaged_misses_.fetch_add(1, std::memory_order_relaxed);
          EncodeRespVersion(ep.sbuf(qd.send_slot), wire_epoch,
                            Status::kNotFound, 0, 0);
          qd.reply.first_len = response_prefix;
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
          has_sync_fallbacks = true;
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
      if (!has_sync_fallbacks) return true;
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
      has_sync_fallbacks = false;
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

    for (;;) {
      // Serve on while running, and during Stop()'s drain for as long as a
      // grant waits for the client.
      const GrantView grants = grant_view(
          !queue.empty() || ring.inflight() != 0 || posted_sends != 0 ||
          revoking_slots != 0);
      if (fail || (!running_ && !grants.holding)) break;
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
                         posted_sends != 0 || revoking_slots != 0;
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

      if (ring.inflight() != 0 && ring.event_fd() < 0) {
        // No completion event from this ring: wait on the ring for at most a
        // millisecond at a time, so the verbs CQ and Stop()'s Wake are seen
        // that late at worst.
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
      // With disk reads in flight the ring's completion event is waited on
      // too, in the same poll(): a new request is taken as soon as it
      // arrives, and a disk completion as soon as it is posted.
      bool grant_wait = false;
      const int bound_ms = wait_bound_ms(grants, &grant_wait);
      const int verbs_wait_ms =
          posted_sends != 0 || reorder_cq_for_test_ ? 1 : bound_ms;
      const bool disk_wait = ring.inflight() != 0;
      g = ep.WaitComp(wcs.data(), static_cast<int>(K), verbs_wait_ms,
                      disk_wait ? ring.event_fd() : -1);
      if (disk_wait) ring.ClearEvent();
      if (g >= 0 && reorder_cq_for_test_)
        reorder_cq_for_test_(wcs.data(), &g, K);
      if (g > 0)
        ep.last_active_us_.store(SteadyUs(), std::memory_order_relaxed);
      if (g >= 0 && grant_expired(grants)) break;
      if (g == 0) {
        if (posted_sends != 0 || grant_wait || disk_wait) continue;
        idle_reclaims_.fetch_add(1, std::memory_order_relaxed);
        break;
      }
      if (g < 0) break;  // disconnect, endpoint error, or Stop()'s Wake
      if (!dispatch_completions(
              wcs.data(), g, process_wc,
              [&] {
                return !queue.empty() || ring.inflight() != 0 ||
                       posted_sends != 0 || revoking_slots != 0;
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
    { std::lock_guard<std::mutex> lk(conn_mu_); live_eps_.erase(&ep); }
    return;
  }
sync_serve_loop:;
#endif  // DFKV_WITH_URING

  auto process_sync_wc = [&](const ibv_wc& wc) -> bool {
    if (wc.status != IBV_WC_SUCCESS) {
      note_window_failure(wc);
      note_completion_error(wc, "sync", K - free_send.size());
      return false;
    }
    if ((wc.wr_id & kMwWrTag) != 0) return false;  // never signalled
    if (wc.opcode == IBV_WC_SEND) {
      const size_t sid = static_cast<size_t>(wc.wr_id);
      if (sid >= K) return false;
      on_send_completion(sid);
      free_send.push_back(sid);
      return true;
    }
    Request request;
    if (!decode_request(wc, &request) || free_send.empty()) return false;
    note_arrival(&request);
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
  for (;;) {
    // Serve on while running, and during Stop()'s drain for as long as a
    // grant waits for the client. Requests are served synchronously here:
    // nothing of the server's is in flight while it waits.
    const GrantView grants = grant_view(/*server_work=*/false);
    if (fail || (!running_ && !grants.holding)) break;
    const bool deferred_for_test =
        reorder_cq_for_test_ && free_send.size() != K;
    bool grant_wait = false;
    const int bound_ms = wait_bound_ms(grants, &grant_wait);
    const int wait_ms =
        pending_recv_count != 0 || deferred_for_test ? 1 : bound_ms;
    int g = ep.WaitComp(wcs.data(), static_cast<int>(K), wait_ms);
    if (g >= 0 && reorder_cq_for_test_)
      reorder_cq_for_test_(wcs.data(), &g, K);
    if (g > 0)
      ep.last_active_us_.store(SteadyUs(), std::memory_order_relaxed);
    if (g >= 0 && grant_expired(grants)) break;
    if (g == 0) {
      if (pending_recv_count != 0 || deferred_for_test || grant_wait) continue;
      idle_reclaims_.fetch_add(1, std::memory_order_relaxed);
      break;
    }
    if (g < 0) break;  // error / Stop()'s Wake()
    if (!dispatch_completions(wcs.data(), g, process_sync_wc,
                              [&] { return revoking_slots != 0; }))
      fail = true;
  }
  // Any prepared sends without completions destructor-abort below.
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
  m(s, "dfkv_rdma_pending_handshakes", "gauge",
    "Accepted bootstrap connections whose bootstrap frames have not arrived",
    PendingHandshakes());
  m(s, "dfkv_rdma_rejected_connections_total", "counter",
    "Bootstrap connections refused at the pending-handshake limit",
    RejectedConnections());
  m(s, "dfkv_rdma_accept_failures_total", "counter",
    "accept() failures from descriptor or memory exhaustion; the bootstrap "
    "listener backed off after each",
    AcceptFailures());
  m(s, "dfkv_rdma_thread_start_failures_total", "counter",
    "Bootstrap connections refused because their thread could not be created",
    ThreadStartFailures());
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
    "RDMA rails opened and checked by a trial registration at startup "
    "(receive chunks are registered per connection, over its own slots)",
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
  m(s, "dfkv_rdma_leaseput_oversize_rejects_total", "counter",
    "Leased-PUT requests refused (kInvalid) for an object above --max-msg",
    lease_put_oversize_rejects_);
  m(s, "dfkv_rdma_leaseput_active", "gauge",
    "In-flight leased-PUT staging leases currently held",
    lease_put_active_);
  m(s, "dfkv_rdma_leaseput_bytes_active", "gauge",
    "Receive-pool bytes held by in-flight leased-PUT staging",
    lease_put_bytes_active_);
  m(s, "dfkv_rdma_pull_zerocopy_served_total", "counter",
    "Dynamic-pull GETs served directly from the pinned RAM arena without a staging copy",
    pull_zerocopy_served_);
  m(s, "dfkv_rdma_pull_zerocopy_register_failures_total", "counter",
    "RAM arena hits that fell back to a staged copy because the per-read MR could not be registered",
    pull_zerocopy_register_failures_);
  m(s, "dfkv_rdma_dynamic_get_active", "gauge",
    "GET staging leases held now: from the grant until the RELEASE, or "
    "until the READY for a GET the client reads from the RAM arena",
    dynamic_get_active_);
  m(s, "dfkv_rdma_dynamic_get_bytes_active", "gauge",
    "Receive-pool bytes of the GET staging leases held now (from the grant "
    "until the RELEASE, or until the READY for an arena-served GET)",
    dynamic_get_bytes_active_);
  m(s, "dfkv_rdma_dynamic_get_mr_active", "gauge",
    "Process-wide exact dynamic GET READ registrations currently held",
    rdma::RcEndpoint::LeaseReadMrActive());
  m(s, "dfkv_rdma_v2_ready", "gauge",
    "Whether RDMA v2 has a registered shared receive segment",
    recv_segment_registered_rails_ > 0 ? 1 : 0);
  m(s, "dfkv_rdma_sessions_kept_total", "counter",
    "Connections that kept their bootstrap socket as a session channel "
    "(DFKV_RDMA_SESSION_FD)",
    sessions_kept_.load(std::memory_order_relaxed));
  m(s, "dfkv_rdma_session_peer_closed_total", "counter",
    "Connections ended because their client closed its session channel",
    session_peer_closed_.load(std::memory_order_relaxed));
  m(s, "dfkv_rdma_inactive_rail_refusals_total", "counter",
    "Bootstrap connections refused because their rail's port was not ACTIVE",
    inactive_rail_refusals_.load(std::memory_order_relaxed));
  m(s, "dfkv_rdma_rail_down_connections_ended_total", "counter",
    "Connections ended because their rail's port stopped being ACTIVE",
    rail_down_connections_ended_.load(std::memory_order_relaxed));
  m(s, "dfkv_rdma_rail_port_changes_total", "counter",
    "Rail port state changes seen by the port watcher (DFKV_RDMA_PORT_CHECK_MS)",
    rail_port_changes_.load(std::memory_order_relaxed));
  m(s, "dfkv_rdma_segment_evictions_total", "counter",
     "Connections evicted to free shared receive segment space",
     segment_evictions_.load(std::memory_order_relaxed));
  m(s, "dfkv_rdma_unstaged_misses_total", "counter",
    "GETs answered kNotFound by an existence check, without a staging lease",
    unstaged_misses_.load(std::memory_order_relaxed));
  m(s, "dfkv_rdma_pressure_unsatisfiable_total", "counter",
    "Receive-pool requests refused without evicting: no idle connection "
    "whose lease would make room, or more than the budget",
    pressure_unsatisfiable_.load(std::memory_order_relaxed));
  m(s, "dfkv_rdma_stop_grants_refused_total", "counter",
    "GET READYs and PUT leases refused (as a miss, or kCacheFull) while "
    "stopping, so that the drain waits only for the grants in flight",
    stop_grants_refused_);
  m(s, "dfkv_rdma_stop_drains_cut_total", "counter",
    "Stops that closed connections still holding grants when "
    "DFKV_RDMA_STOP_DRAIN_MS ran out",
    stop_drains_cut_);
  m(s, "dfkv_rdma_grant_expiries_total", "counter",
    "RDMA connections closed because a GET grant or PUT lease waited longer "
    "than DFKV_RDMA_GRANT_TIMEOUT_MS for the client",
    grant_expiries_);
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
  s += "# HELP dfkv_rdma_rail_port_active Port state of each rail as the port "
       "watcher last saw it: 1 ACTIVE, 0 not, -1 unknown or not watched\n";
  s += "# TYPE dfkv_rdma_rail_port_active gauge\n";
  for (size_t i = 0; rail_port_state_ && i < anchor_devs_.size(); ++i)
    s += "dfkv_rdma_rail_port_active{dev=\"" +
         PromLabelEscape(anchor_devs_[i]) + "\"} " +
         std::to_string(rail_port_state_[i].load(std::memory_order_relaxed)) +
         "\n";
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
  {
    const uint64_t now_ms = SteadyUs() / 1000;
    s += "# HELP dfkv_rdma_rail_retry_exhausted_peers Distinct peer addresses whose "
         "connection on this device ended with RETRY_EXC or RNR_RETRY_EXC in "
         "the last 30 s\n";
    s += "# TYPE dfkv_rdma_rail_retry_exhausted_peers gauge\n";
    for (size_t i = 0; i < rail_retry_peers_.size(); ++i)
      s += "dfkv_rdma_rail_retry_exhausted_peers{dev=\"" +
           PromLabelEscape(anchor_devs_[i]) + "\"} " +
           std::to_string(rail_retry_peers_[i]->Count(now_ms)) + "\n";
  }
  // Lazily populated: a series appears with its first failure, so the label
  // set stays at (devices x statuses that actually occurred).
  s += "# HELP dfkv_rdma_rail_wc_errors_total RDMA error completions on each "
       "local device by work-completion status\n";
  s += "# TYPE dfkv_rdma_rail_wc_errors_total counter\n";
  for (size_t i = 0; i < rail_stats_.size(); ++i) {
    for (size_t slot = 0; slot < rdma::kWcStatusSlots; ++slot) {
      const uint64_t count =
          rail_stats_[i]->wc_errors[slot].load(std::memory_order_relaxed);
      if (count == 0) continue;
      s += "dfkv_rdma_rail_wc_errors_total{dev=\"" +
           PromLabelEscape(anchor_devs_[i]) + "\",status=\"" +
           rdma::WcStatusName(static_cast<int>(slot)) + "\"} " +
           std::to_string(count) + "\n";
    }
  }
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
  // Server memory windows (DFKV_RDMA_SERVER_MW).
  s += "# HELP dfkv_rdma_server_mw_mode 1 if GET grants on this device are "
       "bound by memory windows, 0 if they use per-grant MRs\n";
  s += "# TYPE dfkv_rdma_server_mw_mode gauge\n";
  for (size_t i = 0; i < anchor_devs_.size(); ++i)
    s += "dfkv_rdma_server_mw_mode{dev=\"" + PromLabelEscape(anchor_devs_[i]) +
         "\"} " +
         std::to_string(mw_enabled_ && i < rail_mw_.size() ? rail_mw_[i] : 0) +
         "\n";
  s += "# HELP dfkv_rdma_server_mw_arena_registered_bytes RAM-arena bytes "
       "registered for memory windows on this device\n";
  s += "# TYPE dfkv_rdma_server_mw_arena_registered_bytes gauge\n";
  for (size_t i = 0; i < anchor_devs_.size(); ++i)
    s += "dfkv_rdma_server_mw_arena_registered_bytes{dev=\"" +
         PromLabelEscape(anchor_devs_[i]) + "\"} " +
         std::to_string(mw_regions_.RegisteredBytes(
             i, rdma::BindRegionRegistry::Kind::kArena)) +
         "\n";
  const auto load = [](const std::atomic<uint64_t>& v) {
    return v.load(std::memory_order_relaxed);
  };
  m(s, "dfkv_rdma_server_mw_binds_total", "counter",
    "GET grants bound by a memory window", load(mw_binds_));
  m(s, "dfkv_rdma_server_mw_invalidations_total", "counter",
    "Memory-window invalidations posted at a GET's RELEASE",
    load(mw_invalidations_));
  m(s, "dfkv_rdma_server_mw_invalidation_failures_total", "counter",
    "Memory-window invalidations that could not be posted or completed in "
    "error (the connection ends)",
    load(mw_invalidation_failures_));
  m(s, "dfkv_rdma_server_mw_quarantine_bytes", "gauge",
    "Revoked GET ranges waiting out their quarantine before reuse",
    load(mw_quarantine_bytes_));
  m(s, "dfkv_rdma_server_mw_quarantined_total", "counter",
    "Revoked GET ranges put into quarantine", load(mw_quarantined_total_));
  m(s, "dfkv_rdma_server_mw_quarantine_fallbacks_total", "counter",
    "GET grants given a per-grant MR because the quarantine was full or "
    "would have been longer than DFKV_RDMA_SERVER_MW_QUARANTINE_MAX_MS",
    load(mw_quarantine_fallbacks_));
  m(s, "dfkv_rdma_server_mw_fallback_grants_total", "counter",
    "GET grants given a per-grant MR on a connection with windows: their "
    "memory not (yet) registered for windows",
    load(mw_fallback_grants_));
  m(s, "dfkv_rdma_server_mw_fallback_connections_total", "counter",
    "Connections that could not allocate their memory windows and serve "
    "GETs with per-grant MRs",
    load(mw_fallback_connections_));
  m(s, "dfkv_rdma_server_mw_region_registrations_total", "counter",
    "Receive chunks and RAM-arena extents registered for memory windows, "
    "per device",
    mw_regions_.Registrations());
  m(s, "dfkv_rdma_server_mw_region_register_failures_total", "counter",
    "Registrations of a receive chunk or RAM-arena extent for memory "
    "windows that failed (tried again after 10 s)",
    mw_regions_.RegisterFailures());
  s += "# HELP dfkv_rdma_server_grant_register_seconds Time GET grants spent "
       "authorizing the client's READ: an exact MR, or a window bind\n";
  s += "# TYPE dfkv_rdma_server_grant_register_seconds summary\n";
  s += "dfkv_rdma_server_grant_register_seconds_sum " +
       std::to_string(static_cast<double>(load(grant_register_ns_)) / 1e9) +
       "\n";
  s += "dfkv_rdma_server_grant_register_seconds_count " +
       std::to_string(load(grant_registers_)) + "\n";
  return s;
}

}  // namespace dfkv
