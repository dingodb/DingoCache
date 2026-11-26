// RDMA datapath test over a loopback device (Soft-RoCE / rdma_rxe in CI, or any
// real RDMA NIC). Exercises the native-verbs transport + server + versioned wire
// frame + zero-copy RangeInto + (with depth>1) the pipelined worker pool — the
// code that is otherwise only validated on real 400G hardware. Skips cleanly when
// no RDMA device is present. Built only when DFKV_WITH_RDMA is defined. Run under
// ThreadSanitizer to exercise the worker-pool / QP concurrency.
#include "client/kv_client.h"
#include "client/dfkv_c_api.h"
#include "client/cuda_ipc.h"
#include "client/node_dedup.h"
#include "client/key_map.h"
#include "cache/kv_node_server.h"
#include "cache/rdma_server.h"
#include "common/config_dump.h"
#include "transport/rdma_transport.h"
#include "transport/rail_select.h"
#include "transport/rdma_topology.h"
#include "transport/rdma_protocol.h"
#include "transport/rdma_verbs.h"
#include "utils/accept_guard.h"
#include "utils/net_util.h"

#include "../client/scripted_topology_mds.h"
#include "../utils/fd_exhaustion.h"

#include <fcntl.h>
#include <gtest/gtest.h>
#include <poll.h>
#include <sys/mman.h>  // shm_unlink (node-dedup test)
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/wait.h>
#include <signal.h>

extern char** environ;
#include <unistd.h>

#include <atomic>
#include <iterator>
#include <fstream>
#include <chrono>
#include <condition_variable>
#include <cerrno>
#include <algorithm>
#include <cctype>
#include <array>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <future>
#include <limits>
#include <map>
#include <list>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace dfkv {
class KvNodeServerTestPeer {
 public:
  static RamTier& Ram(KvNodeServer& server) { return *server.ram_; }
  // A RAM promotion copies what the disk holds, so the disk gets the value
  // first: a copy of a value the disk does not hold is not installed. The
  // result is that of the RAM install alone; the disk write must succeed.
  static bool Promote(KvNodeServer& server, const BlockKey& key,
                      const std::string& value) {
    EXPECT_EQ(server.group_.Cache(key, value.data(), value.size()),
              Status::kOk);
    return server.ram_->PutDurable(key, value.data(), value.size(),
                                   server.DiskGeneration(key));
  }
  static uint64_t GetLatencyCount(const KvNodeServer& server) {
    return server.get_lat_.Count();
  }
};
class RdmaServerTestPeer {
 public:
  using DiscoverFn = std::function<rdma::RdmaDiscoveryResult(
      const std::vector<std::string>&, rdma::RdmaDiscoveryPolicy)>;
  using InitializeAnchorFn = std::function<std::unique_ptr<rdma::RcEndpoint>(
      const std::string&, void*, size_t)>;

  static void SetDiscovery(RdmaServer* server, DiscoverFn discover) {
    server->discover_for_test_ = std::move(discover);
  }
  static void SetBeforeEndpointTeardown(RdmaServer* server,
                                        std::function<void()> hook) {
    server->before_endpoint_teardown_for_test_ = std::move(hook);
  }
  static void SetInitializeAnchor(RdmaServer* server,
                                  InitializeAnchorFn initialize) {
    server->initialize_anchor_for_test_ = std::move(initialize);
  }
  static void SetStartupRegistration(
      RdmaServer* server, std::function<bool(const std::string&)> hook) {
    server->startup_registration_for_test_ = std::move(hook);
  }
  static void SetResponderReadDepth(RdmaServer* server, int depth) {
    server->responder_read_depth_for_test_ = depth;
  }
  static void SetPortState(RdmaServer* server,
                           std::function<int(size_t)> state) {
    std::lock_guard<std::mutex> lock(server->rail_listener_mu_);
    server->port_state_for_test_ = std::move(state);
  }
  static void SetBeforeLiveRegistration(RdmaServer* server,
                                        std::function<void()> hook) {
    server->before_live_registration_for_test_ = std::move(hook);
  }
  static size_t LiveConnectionsOnRail(RdmaServer& server, size_t rail) {
    std::lock_guard<std::mutex> lock(server.conn_mu_);
    size_t n = 0;
    for (const auto& [ep, live] : server.live_eps_) n += live.rail == rail;
    return n;
  }
  static size_t DefaultRail(const RdmaServer& server) {
    return server.DefaultRail();
  }
  static size_t AnchorCount(const RdmaServer& server) {
    return server.anchors_.size();
  }
  static bool MwRegistryStarted(const RdmaServer& server) {
    return server.mw_regions_.started();
  }
  static size_t RailStatsCount(const RdmaServer& server) {
    return server.rail_stats_.size();
  }
  static size_t RegisteredRailCount(const RdmaServer& server) {
    return server.recv_segment_registered_rails_;
  }
  static uint64_t RegisterLegacyConnection(RdmaServer* server) {
    return server->RegisterLegacyConnection();
  }
  static bool HasLegacyConnection(RdmaServer* server, uint64_t token) {
    std::lock_guard<std::mutex> lock(server->legacy_mu_);
    return server->legacy_connections_tokens_.count(token) != 0;
  }
  static void ForgetLegacyConnection(RdmaServer* server, uint64_t token) {
    server->ForgetLegacyConnection(token);
  }
  static void ServeBootstrap(RdmaServer* server, int fd) {
    server->Serve(fd, std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(
                              RdmaServer::kBootstrapReadMs));
  }
  static void SetBootstrapReadMs(RdmaServer* server, int ms) {
    server->bootstrap_read_ms_ = ms;
  }
  static void SetAfterProbeReply(RdmaServer* server,
                                 std::function<void()> hook) {
    server->after_probe_reply_for_test_ = std::move(hook);
  }
  static void SetMaxPendingHandshakes(RdmaServer* server, uint64_t limit) {
    server->max_pending_handshakes_ = limit;
  }
  static uint64_t MaxPendingHandshakes(const RdmaServer& server) {
    return server.max_pending_handshakes_;
  }
  static void FailPinnedRegistration(RdmaServer& server) {
    server.fail_pinned_registration_for_test_ = true;
  }
  static void SetMwRegister(RdmaServer& server,
                            std::function<bool(size_t, char*)> hook) {
    server.mw_register_for_test_ = std::move(hook);
  }
  using MwReservation = RdmaServer::MwReservation;
  static uint64_t MwRailSnapshot(const RdmaServer& server,
                                 const std::atomic<uint64_t>& rail,
                                 uint64_t bytes) {
    return server.MwRailSnapshot(rail, bytes);
  }
  static MwReservation MwReserveQuarantine(RdmaServer* server, uint64_t rail,
                                           uint64_t bytes) {
    return server->MwReserveQuarantine(rail, bytes);
  }
  static uint64_t MwReservedBytes(const RdmaServer& server) {
    return server.mw_reserved_bytes_;
  }
  static uint64_t MwQuarantinedBytes(const RdmaServer& server) {
    return server.mw_quarantine_bytes_.load(std::memory_order_relaxed);
  }
  static void MwQuarantine(RdmaServer* server, MwReservation reservation,
                           uint64_t rail, uint64_t bytes) {
    RdmaServer::MwQuarantined entry;
    entry.reservation = std::move(reservation);
    entry.rail_bytes = rail;
    entry.bytes = bytes;
    server->MwQuarantine(std::move(entry));
  }
  static void MwReleaseQuarantine(RdmaServer* server) {
    server->MwReleaseQuarantine(true);
  }
  static void SetMwEvents(
      RdmaServer& server,
      std::function<void(const char*, const void*)> hook) {
    server.mw_event_for_test_ = std::move(hook);
  }
  // The held-grant gauges, read without building the metrics text: fast
  // enough for a reply hook to act before the client has used its grant.
  static uint64_t LeasePutActive(const RdmaServer& server) {
    return server.lease_put_active_.load(std::memory_order_relaxed);
  }
  static uint64_t DynamicGetActive(const RdmaServer& server) {
    return server.dynamic_get_active_.load(std::memory_order_relaxed);
  }
  static void SetBootstrapReplyDelayMs(RdmaServer& server, int ms) {
    server.bootstrap_reply_delay_ms_for_test_.store(ms);
  }
  static void SetAfterReplyPost(RdmaServer& server, std::function<void()> hook) {
    server.after_reply_post_for_test_ = std::move(hook);
  }
  static void SetCqHooks(
      RdmaServer& server,
      std::function<void(ibv_wc*, int*, size_t)> reorder,
      std::function<void(size_t, size_t, size_t)> observe) {
    server.reorder_cq_for_test_ = std::move(reorder);
    server.after_cq_dispatch_for_test_ = std::move(observe);
  }
  static void SetAfterRequestRearm(
      RdmaServer& server, std::function<void(size_t)> hook) {
    server.after_request_rearm_for_test_ = std::move(hook);
  }
  // Ages every connection, quiescent or not: what a connection that never
  // sent a request looks like after age_ms.
  static size_t AgeAllConnections(RdmaServer& server, uint64_t age_ms) {
    const uint64_t now = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    std::lock_guard<std::mutex> lock(server.conn_mu_);
    for (const auto& [ep, live] : server.live_eps_)
      ep->last_active_us_.store(now - age_ms * 1000,
                                std::memory_order_relaxed);
    return server.live_eps_.size();
  }
  static size_t RefreshQuiescentActivity(RdmaServer& server,
                                         uint64_t age_ms = 0) {
    // Shift only the activity timestamps of real, completed QPs. The 1.5 s
    // pressure case distinguishes the old 2 s grace from the new 1 s grace;
    // the zero-age case keeps immediate pooled reuse protected.
    const uint64_t now = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    std::lock_guard<std::mutex> lock(server.conn_mu_);
    size_t count = 0;
    for (const auto& [ep, live] : server.live_eps_) {
      if (!live.reclaimable->load(std::memory_order_acquire)) continue;
      ep->last_active_us_.store(now - age_ms * 1000,
                                std::memory_order_relaxed);
      ++count;
    }
    return count;
  }
#ifdef DFKV_WITH_URING
  static void SetUringBackendFactory(
      RdmaServer* server,
      std::function<UringReader::Backend*()> factory) {
    server->uring_backend_factory_for_test_ = std::move(factory);
  }
#endif
};

class RdmaTransportTestPeer {
 public:
  using Connection = RdmaTransport::Conn;

  // Ends every idle pooled connection, data and control, as the idle
  // reclaim or a failure would; nothing of the transport is left open.
  static size_t DropAllPooled(RdmaTransport* transport) {
    std::vector<Connection*> idle;
    {
      std::lock_guard<std::mutex> lock(transport->mu_);
      for (auto* pools : {&transport->pool_, &transport->control_pool_}) {
        for (auto& [node, connections] : *pools) {
          (void)node;
          idle.insert(idle.end(), connections.begin(), connections.end());
        }
        pools->clear();
      }
    }
    for (Connection* conn : idle) transport->Destroy(conn);
    return idle.size();
  }
  static long Rails(const RdmaTransport& transport) {
    return static_cast<long>(transport.devs_.size());
  }
  static std::string Rail(const RdmaTransport& transport, size_t rail) {
    return transport.devs_[rail];
  }
  // TRN-29 hooks.
  static void FailRebuilds(RdmaTransport* transport, bool fail) {
    transport->rebuild_fails_for_test_ = fail;
  }
  static void PauseDeviceChecks(RdmaTransport* transport, bool paused) {
    transport->device_checks_paused_for_test_ = paused;
  }

  struct SeededFailure {
    uint64_t now_us = 0;
    RemoteRailCompletion completion;
  };

  struct RetryDecision {
    bool retry = false;
    bool cross_rail = false;
    std::vector<uint8_t> excluded;
  };

  static std::optional<SeededFailure> CoolRemoteRail(
      RdmaTransport* transport, const std::string& peer_id, size_t rail) {
    const uint64_t now = rdma::RailPolicy::NowMicros();
    auto lease = transport->remote_rail_health_->TryAcquire(peer_id, rail, now);
    if (!lease) return std::nullopt;
    return SeededFailure{
        now,
        transport->remote_rail_health_->Complete(
            peer_id, rail, lease->generation,
            RemoteRailOutcome::kEndpointFailure, now)};
  }

  static std::vector<uint8_t> RemoteAllowed(
      const RdmaTransport& transport, const std::string& peer_id,
      const std::vector<uint8_t>& candidates, uint64_t now_us) {
    return transport.remote_rail_health_->AllowedMask(peer_id, candidates,
                                                      now_us);
  }

  static size_t DataPoolSize(RdmaTransport* transport,
                             const std::string& node) {
    std::lock_guard<std::mutex> lock(transport->mu_);
    const auto found = transport->pool_.find(node);
    return found == transport->pool_.end() ? 0 : found->second.size();
  }

  static size_t ConnectionBound(const RdmaTransport& transport,
                                size_t required) {
    return transport.ConnectionBound(required);
  }

  static std::vector<size_t> IdleDataBounds(
      RdmaTransport* transport, const std::string& node) {
    return transport->IdleDataBounds(node);
  }
  static std::vector<size_t> IdleDataDepths(
      RdmaTransport* transport, const std::string& node) {
    return transport->IdleDataDepths(node);
  }

  static size_t ConnectionDepth(RdmaTransport* transport,
                                const std::string& node, size_t requested) {
    return transport->ConnectionDepth(node, RdmaTransport::Lane::kData,
                                      requested);
  }

  static uint64_t PeerPublication(const RdmaTransport& transport,
                                  const std::string& node) {
    return transport.peer_topologies_->Snapshot(node)->publication;
  }

  static RdmaTransport::Conn* AcquireData(RdmaTransport* transport,
                                          const std::string& node,
                                          bool force_new = false) {
    RdmaTransport::AcquireOptions options;
    options.force_new = force_new;
    return transport->Acquire(node, RdmaTransport::Lane::kData, options).conn;
  }

  // A confirming dial (ConfirmRefusal's Acquire) on `rail`.
  static RdmaTransport::Conn* AcquireRefusalCheck(RdmaTransport* transport,
                                                  const std::string& node,
                                                  size_t rail) {
    RdmaTransport::AcquireOptions options;
    options.excluded.assign(transport->devs_.size(), 1);
    options.excluded[rail] = 0;
    options.refusal_check = true;
    return transport->Acquire(node, RdmaTransport::Lane::kControl, options)
        .conn;
  }
  static void FailDials(RdmaTransport* transport, int count) {
    transport->fail_dials_for_test_.store(count);
  }
  static bool ChargeRefusal(RdmaTransport* transport, const std::string& node,
                            const std::string& health_id, size_t rail) {
    return transport->ChargeRefusal(node, health_id, rail);
  }
  static int QpAccessFlags(const RdmaTransport::Conn* conn) {
    return RdmaTransport::QpAccessFlags(conn);
  }
  static void ReleaseData(RdmaTransport* transport, const std::string& node,
                          RdmaTransport::Conn* conn) {
    transport->Release(node, RdmaTransport::Lane::kData, conn);
  }
  static void SetPeerBusyRetryUs(RdmaTransport* transport, uint64_t us) {
    transport->peer_busy_retry_us_ = us;
  }
  static void SetUnreachableHoldUs(RdmaTransport* transport, uint64_t us) {
    transport->peer_unreachable_hold_us_ = us;
  }
  static RdmaTransport::Conn* AcquireControl(RdmaTransport* transport,
                                             const std::string& node) {
    return transport
        ->Acquire(node, RdmaTransport::Lane::kControl,
                  RdmaTransport::AcquireOptions{})
        .conn;
  }
  // A control-lane connection on `rail` only.
  static RdmaTransport::Conn* AcquireControlOn(RdmaTransport* transport,
                                               const std::string& node,
                                               size_t rail) {
    RdmaTransport::AcquireOptions options;
    options.excluded.assign(transport->devs_.size(), 1);
    options.excluded[rail] = 0;
    return transport->Acquire(node, RdmaTransport::Lane::kControl, options)
        .conn;
  }
  // Endpoint opens on `rail` fail with `err`, as the device would.
  static void FailOpenOn(RdmaTransport* transport, size_t rail, int err) {
    transport->fail_open_errno_for_test_.resize(transport->devs_.size(), 0);
    transport->fail_open_errno_for_test_[rail] = err;
  }
  static void ReleaseControl(RdmaTransport* transport,
                             const std::string& node,
                             RdmaTransport::Conn* conn) {
    transport->Release(node, RdmaTransport::Lane::kControl, conn);
  }

  static void ReleaseDataAt(RdmaTransport* transport, const std::string& node,
                            Connection* conn, uint64_t now_us) {
    transport->Release(node, RdmaTransport::Lane::kData, conn,
                       RemoteRailOutcome::kSuccess, now_us);
  }

  static void FailData(RdmaTransport* transport, Connection* conn) {
    transport->Destroy(conn, rdma::RailCompletion::kEndpointFailure);
  }

  static void StopMaintenance(RdmaTransport* transport) {
    transport->keepalive_stop_.store(true, std::memory_order_relaxed);
    transport->keepalive_cv_.notify_all();
    if (transport->keepalive_thread_.joinable())
      transport->keepalive_thread_.join();
    // Tests invoke the same maintenance pass explicitly with a logical clock.
    transport->keepalive_stop_.store(false, std::memory_order_relaxed);
  }

  static void MaintainIdle(RdmaTransport* transport, uint64_t now_us,
                           bool keepalive = false) {
    transport->MaintainIdle(now_us, keepalive);
  }

  static rdma::ResourceRequest UsedResources(const RdmaTransport& transport) {
    return transport.resource_budget_->used();
  }

  static std::pair<size_t, size_t> PoolLimits(const RdmaTransport& transport) {
    return {transport.pool_max_, transport.data_pool_max_};
  }

  static bool HasCapabilities(RdmaTransport* transport,
                              const std::string& node) {
    std::lock_guard<std::mutex> lock(transport->mu_);
    return transport->peer_capabilities_.find(node) !=
           transport->peer_capabilities_.end();
  }

  static RetryDecision PrepareEndpointRetry(
      RdmaTransport* transport, bool from_pool, bool replay_safe,
      bool request_posted, size_t attempted_rail) {
    auto peer = std::make_shared<rdma::PeerRailSnapshot>();
    peer->complete = true;
    peer->peer_healthy.assign(transport->devs_.size(), 1);
    peer->compatible.assign(transport->devs_.size(), 1);
    RdmaTransport::RailMask excluded(transport->devs_.size(), 0);
    bool cross_rail = false;
    const bool retry = transport->PrepareRetry(
        /*attempt=*/0, from_pool, attempted_rail,
        RdmaTransport::AcquireFailure::kEndpoint,
        replay_safe ? RdmaTransport::ReplaySafety::kReplaySafe
                    : RdmaTransport::ReplaySafety::kUnsafeAfterPost,
        request_posted, peer, "test-peer:1", &excluded, &cross_rail);
    return RetryDecision{retry, cross_rail, std::move(excluded)};
  }
};

}  // namespace dfkv

namespace {

// Small per-buffer cap so the test stays well under a modest RLIMIT_MEMLOCK
// (RDMA pins registered memory); the test values are a few KB.
constexpr size_t kMaxMsg = 256 * 1024;

// The DFKV_RDMA_TEST_* fault hooks are read only in a process that asks.
[[maybe_unused]] const bool kRdmaTestHooks = (dfkv::RdmaTransport::EnableTestHooks(), true);

class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) {
      had_old_ = true;
      old_ = old;
    }
    if (value)
      ::setenv(name, value, 1);
    else
      ::unsetenv(name);
  }
  ~ScopedEnv() {
    if (had_old_)
      ::setenv(name_.c_str(), old_.c_str(), 1);
    else
      ::unsetenv(name_.c_str());
  }

  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

 private:
  std::string name_;
  std::string old_;
  bool had_old_ = false;
};

std::string SelfHdr() { return "test/model"; }
void ConfigureTestRecvSegment() {
  // Keep Soft-RoCE CI below modest RLIMIT_MEMLOCK. Production commits a
  // 256-MiB initial chunk under a 128-GiB budget; fixtures need only 32 MiB.
  ::setenv("DFKV_RDMA_RECV_SEGMENT_SIZE", "33554432", 0);
}

struct ChildLegacyTokenResult {
  uint64_t token = 0;
  int status = -1;
  bool transferred = false;
  bool stale_token_found = false;
};

ChildLegacyTokenResult LegacyTokenFromFreshProcess(uint64_t stale_token = 0) {
  int pipe_fds[2];
  if (::pipe(pipe_fds) != 0) return {};
  const pid_t child = ::fork();
  if (child < 0) {
    ::close(pipe_fds[0]);
    ::close(pipe_fds[1]);
    return {};
  }
  if (child == 0) {
    ::close(pipe_fds[0]);
    RdmaServer server(RdmaServer::Handler{}, kMaxMsg);
    const uint64_t token = RdmaServerTestPeer::RegisterLegacyConnection(&server);
    std::array<char, sizeof(token) + 1> wire{};
    net::PutU64(wire.data(), token);
    wire[sizeof(token)] = static_cast<char>(
        RdmaServerTestPeer::HasLegacyConnection(&server, stale_token));
    size_t written = 0;
    while (written < wire.size()) {
      const ssize_t n =
          ::write(pipe_fds[1], wire.data() + written, wire.size() - written);
      if (n > 0) {
        written += static_cast<size_t>(n);
        continue;
      }
      if (n < 0 && errno == EINTR) continue;
      ::_exit(2);
    }
    ::_exit(0);
  }

  ::close(pipe_fds[1]);
  ChildLegacyTokenResult result;
  std::array<char, sizeof(result.token) + 1> wire{};
  size_t received = 0;
  while (received < wire.size()) {
    const ssize_t n =
        ::read(pipe_fds[0], wire.data() + received, wire.size() - received);
    if (n > 0) {
      received += static_cast<size_t>(n);
      continue;
    }
    if (n < 0 && errno == EINTR) continue;
    break;
  }
  ::close(pipe_fds[0]);
  pid_t waited;
  do {
    waited = ::waitpid(child, &result.status, 0);
  } while (waited < 0 && errno == EINTR);
  result.token = net::GetU64(wire.data());
  result.stale_token_found = wire[sizeof(result.token)] != 0;
  result.transferred =
      received == wire.size() && waited == child &&
      WIFEXITED(result.status) && WEXITSTATUS(result.status) == 0;
  return result;
}

bool RequestLegacyProof(RdmaServer* server, uint64_t token) {
  int sockets[2];
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) return false;
  std::thread serving(
      [&] { RdmaServerTestPeer::ServeBootstrap(server, sockets[1]); });
  char request[rdma::kDevNameBytes], proof[rdma::kV2RetireProofBytes];
  rdma::EncodeDevFrame(rdma::kV2RetireWriterDevice, token, request);
  const bool proved = net::WriteAll(sockets[0], request, sizeof(request)) &&
      net::ReadAll(sockets[0], proof, sizeof(proof)) &&
      rdma::ParseV2RetireProof(proof, token);
  ::close(sockets[0]);
  serving.join();
  return proved;
}

// A cache node serving RDMA: KvNodeServer owns the DiskCacheGroup; RdmaServer
// bootstraps QPs and routes requests to it (generic handler + zero-copy range).
struct RdmaNode {
  fs::path dir;
  std::unique_ptr<KvNodeServer> srv;
  std::unique_ptr<RdmaServer> rsrv;
  std::string addr;  // bootstrap "ip:port" for the client member list
  mutable std::mutex observation_mu;
  std::map<std::string, size_t> cache_direct_calls;
  std::map<std::string, size_t> range_direct_calls;
  std::atomic<int> handler_delay_ms{0};

  std::function<void(size_t)> before_range;
  // Runs on the RdmaServer before Start (test seams).
  static inline std::function<void(RdmaServer&)> before_start;
  // Called by the direct PUT handler with the batch arrival it was given.
  std::function<void(std::chrono::steady_clock::time_point)> on_cache_direct;
  explicit RdmaNode(const std::string& tag, size_t max_msg = kMaxMsg,
                    bool fail_pinned_registration = false,
                    bool prepare_reads = false,
                    std::function<void()> after_reply_post = {},
                    std::function<void(ibv_wc*, int*, size_t)> reorder = {},
                    std::function<void(size_t, size_t, size_t)> observe = {},
                    std::function<void(size_t)> after_rearm = {}) {
    ConfigureTestRecvSegment();
    dir = fs::temp_directory_path() / ("dfkv_rdma_" + tag);
    fs::remove_all(dir);
    fs::create_directories(dir);
    srv = std::make_unique<KvNodeServer>(dir.string(), 1ull << 30);
    EXPECT_EQ(srv->Start(0), Status::kOk);  // TCP listener owns the cache group
    rsrv = std::make_unique<RdmaServer>(
        [this](uint8_t op, const BlockKey& key, uint64_t off, uint64_t len,
               const char* pl, uint64_t pll, std::string* out,
               size_t* value_len) {
          const int delay = handler_delay_ms.load(std::memory_order_relaxed);
          if (delay > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(delay));
          return srv->ProcessRequestForKey(
              op, key, off, len, pl, pll, out, value_len);
        },
        max_msg);
    rsrv->set_range_handler(
        [this](const BlockKey& key, uint64_t off, uint64_t len,
               char* io_buf, size_t cap, const char** out_data,
               size_t* out_len, size_t* value_len) {
          size_t call = 0;
          {
            std::lock_guard<std::mutex> lock(observation_mu);
            call = ++range_direct_calls[key.Filename()];
          }
          if (before_range) before_range(call);
          return srv->RangeDirectForKey(
              key, off, len, io_buf, cap, out_data, out_len, value_len);
        });
    rsrv->set_cache_direct_handler(
        [this](const BlockKey& key, char* data, size_t len, size_t cap,
               std::chrono::steady_clock::time_point batch_arrival) {
          {
            std::lock_guard<std::mutex> lock(observation_mu);
            ++cache_direct_calls[key.Filename()];
          }
          if (on_cache_direct) on_cache_direct(batch_arrival);
          return srv->CacheDirectForKey(key, data, len, cap, batch_arrival);
        });
    if (prepare_reads) {
      rsrv->set_prepare_read_handler(
          [this](const BlockKey& key, uint64_t off, uint64_t len,
                 char* staging, size_t cap) {
            size_t call = 0;
            {
              std::lock_guard<std::mutex> lock(observation_mu);
              call = ++range_direct_calls[key.Filename()];
            }
            if (before_range) before_range(call);
            return srv->PrepareReadForKey(key, off, len, staging, cap);
          });
    }
    // As dfkv_server_main.
    rsrv->set_lookup_handler(
        [this](const BlockKey& key) { return srv->LookupForGet(key); });
    RdmaServerTestPeer::SetAfterReplyPost(*rsrv, std::move(after_reply_post));
    RdmaServerTestPeer::SetCqHooks(*rsrv, std::move(reorder),
                                   std::move(observe));
    RdmaServerTestPeer::SetAfterRequestRearm(*rsrv, std::move(after_rearm));
    if (fail_pinned_registration)
      RdmaServerTestPeer::FailPinnedRegistration(*rsrv);
    if (srv->ram_enabled()) {
      // Mirror dfkv_server_main: enable the B5-3 pinned zero-copy pull so
      // RAM-tier fixtures exercise the production GET path instead of always
      // staging. The arena itself is not registered; each pull registers the
      // value it serves.
      rsrv->set_pinned_ram_handler(
          [this](const BlockKey& key, uint64_t off, uint64_t len,
                 PreparedRead* out) {
            return srv->RamPinnedHitForKey(key, off, len, out);
          });
      rsrv->set_mw_arena(srv->ram_tier());  // as dfkv_server_main
    }
    if (before_start) before_start(*rsrv);
    EXPECT_EQ(rsrv->Start(0), Status::kOk);
    addr = "127.0.0.1:" + std::to_string(rsrv->port());
  }
  ~RdmaNode() {
    if (rsrv) rsrv->Stop();
    if (srv) srv->Stop();
    fs::remove_all(dir);
  }
  size_t CacheDirectCalls(const BlockKey& key) const {
    std::lock_guard<std::mutex> lock(observation_mu);
    const auto found = cache_direct_calls.find(key.Filename());
    return found == cache_direct_calls.end() ? 0 : found->second;
  }
  size_t RangeDirectCalls(const BlockKey& key) const {
    std::lock_guard<std::mutex> lock(observation_mu);
    const auto found = range_direct_calls.find(key.Filename());
    return found == range_direct_calls.end() ? 0 : found->second;
  }
};

bool HaveRdma() { return RdmaTransport::Available(); }

// True when every ACTIVE RDMA device is Soft-RoCE (rxe), as in CI. There, a
// SEND to a peer QP that was destroyed never completes with an error: no
// RETRY_EXC within 60 s (CI, rxe over lo), where a ConnectX HCA reports it
// after about 3.6 s. Tests that wait for that error skip on rxe.
bool SoftRoceOnly() {
  const auto discovered = rdma::RdmaTopology::Discover(
      {}, rdma::RdmaDiscoveryPolicy::kActiveOnly);
  if (discovered.status != rdma::RdmaDiscoveryStatus::kOk ||
      discovered.devices.empty())
    return false;
  for (const auto& device : discovered.devices)
    if (device.name.rfind("rxe", 0) != 0) return false;
  return true;
}
constexpr const char* kRxeNoRetryExceeded =
    "rxe: a SEND to a destroyed peer QP never fails with RETRY_EXC";

// Last value of a single-line Prometheus counter (skips the # HELP/# TYPE lines
// via rfind, which lands on the value line emitted after them).
long CounterVal(const std::string& text, const std::string& name) {
  auto p = text.rfind(name);
  if (p == std::string::npos) return -1;
  auto sp = text.find(' ', p);
  if (sp == std::string::npos) return -1;
  try { return std::stol(text.substr(sp + 1)); } catch (...) { return -1; }
}
long DeviceCounterVal(const std::string& text, const std::string& name,
                      const std::string& device) {
  const std::string prefix = name + "{dev=\"" + device + "\"} ";
  const auto pos = text.find(prefix);
  if (pos == std::string::npos) return -1;
  try {
    return std::stol(text.substr(pos + prefix.size()));
  } catch (...) {
    return -1;
  }
}

long MetricSum(const std::string& text, const std::string& prefix) {
  long total = 0;
  size_t begin = 0;
  while (begin < text.size()) {
    const size_t end = text.find('\n', begin);
    const size_t length =
        end == std::string::npos ? text.size() - begin : end - begin;
    if (text.compare(begin, prefix.size(), prefix) == 0) {
      const size_t space = text.rfind(' ', begin + length);
      if (space != std::string::npos && space >= begin) {
        try {
          total += std::stol(text.substr(space + 1, begin + length - space - 1));
        } catch (...) {
          return -1;
        }
      }
    }
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return total;
}
// Every sample of a labeled family as {label block, value}, where the label
// block is the text between the braces (e.g. dev="mlx5_0",status="x").
std::vector<std::pair<std::string, long>> LabeledSeries(
    const std::string& text, const std::string& family) {
  std::vector<std::pair<std::string, long>> series;
  const std::string prefix = family + "{";
  size_t begin = 0;
  while (begin < text.size()) {
    size_t end = text.find('\n', begin);
    if (end == std::string::npos) end = text.size();
    const size_t close = text.find("} ", begin);
    if (text.compare(begin, prefix.size(), prefix) == 0 &&
        close != std::string::npos && close < end) {
      try {
        series.emplace_back(
            text.substr(begin + prefix.size(), close - begin - prefix.size()),
            std::stol(text.substr(close + 2, end - close - 2)));
      } catch (...) {
      }
    }
    begin = end + 1;
  }
  return series;
}

// True when the family has series and every one of them is zero.
bool LabeledSeriesAllZero(const std::string& text, const std::string& family) {
  const auto series = LabeledSeries(text, family);
  if (series.empty()) return false;
  for (const auto& [labels, value] : series) {
    (void)labels;
    if (value != 0) return false;
  }
  return true;
}

// Value of one label inside a label block returned by LabeledSeries.
std::string LabelValue(const std::string& labels, const std::string& name) {
  const std::string key = name + "=\"";
  size_t pos = labels.find(key);
  while (pos != std::string::npos && pos != 0 && labels[pos - 1] != ',')
    pos = labels.find(key, pos + 1);
  if (pos == std::string::npos) return {};
  const size_t begin = pos + key.size();
  const size_t end = labels.find('"', begin);
  return end == std::string::npos ? std::string()
                                  : labels.substr(begin, end - begin);
}

std::vector<std::string> LinesContaining(const std::string& text,
                                         const std::string& needle) {
  std::vector<std::string> lines;
  size_t begin = 0;
  while (begin < text.size()) {
    size_t end = text.find('\n', begin);
    if (end == std::string::npos) end = text.size();
    const std::string line = text.substr(begin, end - begin);
    if (line.find(needle) != std::string::npos) lines.push_back(line);
    begin = end + 1;
  }
  return lines;
}

// Value of a space-delimited `name=value` field of a log line.
std::string LogField(const std::string& line, const std::string& name) {
  const std::string key = " " + name + "=";
  const size_t pos = line.find(key);
  if (pos == std::string::npos) return {};
  const size_t begin = pos + key.size();
  const size_t end = line.find(' ', begin);
  return line.substr(begin, end == std::string::npos ? end : end - begin);
}
// Rail-policy configuration is process-global. Keep every two-HCA test on one
// suite-safe value: 32 concurrent callers may each reserve the depth-four
// maximum, even if locality initially selects the same rail for all of them.
constexpr char kRealHcaRailCredits[] = "128";

std::vector<std::string> ConfiguredTwoTestRails() {
  const char* configured = std::getenv("DFKV_RDMA_DEV");
  if (!configured || !*configured) return {};
  std::vector<std::string> rails;
  std::string value(configured);
  for (size_t start = 0; start <= value.size();) {
    const size_t comma = value.find(',', start);
    const size_t end = comma == std::string::npos ? value.size() : comma;
    std::string rail = value.substr(start, end - start);
    rail.erase(std::remove_if(rail.begin(), rail.end(),
                              [](unsigned char c) { return std::isspace(c); }),
               rail.end());
    if (!rail.empty()) rails.push_back(std::move(rail));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return rails.size() == 2 ? rails : std::vector<std::string>{};
}
bool HaveConfiguredActiveRails(const std::vector<std::string>& rails) {
  if (rails.size() != 2 || rails[0] == rails[1]) return false;
  const auto discovered = rdma::RdmaTopology::Discover(
      rails, rdma::RdmaDiscoveryPolicy::kActiveOnly);
  return discovered.status == rdma::RdmaDiscoveryStatus::kOk &&
         discovered.devices.size() == rails.size();
}

void ExpectReleasedRailResources(const std::string& before,
                                 const std::string& after,
                                 const std::vector<std::string>& rails) {
  for (const auto& rail : rails) {
    EXPECT_EQ(DeviceCounterVal(after, "dfkv_rdma_client_rail_inflight", rail),
              0)
        << rail;
    EXPECT_EQ(
        DeviceCounterVal(after, "dfkv_rdma_client_rail_credits_available",
                         rail),
        DeviceCounterVal(before, "dfkv_rdma_client_rail_credits_available",
                         rail))
        << rail;
  }
  EXPECT_EQ(CounterVal(after, "dfkv_rdma_client_transient_user_mr_active"),
            CounterVal(before,
                       "dfkv_rdma_client_transient_user_mr_active"));
}

struct FakePoolRail {
  size_t declared = 0;
  size_t staged = 0;
  bool reject_stage = false;
  size_t commits = 0;
  size_t rollbacks = 0;

  bool Stage(size_t requested) {
    if (reject_stage) return false;
    staged = requested;
    return true;
  }
  void Commit() {
    declared = staged;
    ++commits;
  }
  void Rollback() {
    staged = declared;
    ++rollbacks;
  }
  bool Covers(size_t offset, size_t length) const {
    return offset <= declared && length <= declared - offset;
  }
};

std::string CudaTestBytes(size_t size, uint8_t seed) {
  std::string bytes(size, '\0');
  for (size_t i = 0; i < size; ++i) {
    bytes[i] = static_cast<char>(
        (static_cast<unsigned>(seed) + 131u * i + 17u * (i >> 3)) & 0xffu);
  }
  return bytes;
}

// CUDA allocation with sentinels on both sides of the caller-visible range.
// Upload/download use the dlopen'd driver surface, so this test binary needs
// neither CUDA headers nor a link-time dependency on libcuda.
class CudaGuardedBuffer {
 public:
  static constexpr size_t kGuardBytes = 37;
  static constexpr uint8_t kGuardByte = 0xa7;

  CudaGuardedBuffer(const CudaLib* cuda, size_t payload_size)
      : cuda_(cuda), payload_size_(payload_size) {}
  ~CudaGuardedBuffer() {
    if (base_) cuda_->MemFree(base_);
  }

  bool Allocate() {
    if (cuda_->MemAlloc(&base_, payload_size_ + 2 * kGuardBytes) !=
        kCudaSuccess) {
      base_ = 0;
      return false;
    }
    const std::string initial(payload_size_ + 2 * kGuardBytes,
                              static_cast<char>(kGuardByte));
    return cuda_->Memcpy(base_,
                         reinterpret_cast<CUdeviceptr>(initial.data()),
                         initial.size()) == kCudaSuccess;
  }

  void* payload() const {
    return reinterpret_cast<void*>(base_ + kGuardBytes);
  }

  bool ReadPayload(std::string* out) const {
    out->assign(payload_size_, '\0');
    return cuda_->Memcpy(reinterpret_cast<CUdeviceptr>(out->data()),
                         base_ + kGuardBytes, payload_size_) == kCudaSuccess;
  }

  bool GuardsIntact() const {
    std::string allocation(payload_size_ + 2 * kGuardBytes, '\0');
    if (cuda_->Memcpy(reinterpret_cast<CUdeviceptr>(allocation.data()), base_,
                      allocation.size()) != kCudaSuccess) {
      return false;
    }
    const std::string guard(kGuardBytes, static_cast<char>(kGuardByte));
    return allocation.compare(0, kGuardBytes, guard) == 0 &&
           allocation.compare(kGuardBytes + payload_size_, kGuardBytes,
                              guard) == 0;
  }

 private:
  const CudaLib* cuda_;
  size_t payload_size_;
  CUdeviceptr base_ = 0;
};

const CudaLib* ActiveCudaForTest() {
  // The test is the CUDA host here; libdfkv only attaches to a driver the
  // host loaded and initialized.
  const CudaLib* cuda = CudaLib::LoadForHost();
  if (!cuda || !cuda->BindPrimaryCtx(0)) return nullptr;
  return cuda;
}

using MemcpyAsyncFn =
    CUresult (*)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
using StreamSynchronizeFn = CUresult (*)(CUstream);

MemcpyAsyncFn cuda_memcpy_async_real = nullptr;
StreamSynchronizeFn cuda_stream_synchronize_real = nullptr;
std::atomic<bool> fail_next_cuda_memcpy_async{false};
std::atomic<bool> fail_next_cuda_stream_synchronize{false};
std::atomic<uint64_t> injected_cuda_memcpy_async_calls{0};
std::atomic<uint64_t> injected_cuda_stream_synchronize_calls{0};
std::atomic<CUdeviceptr> last_cuda_memcpy_async_source{0};

CUresult InjectCudaMemcpyAsync(CUdeviceptr destination, CUdeviceptr source,
                               size_t size, CUstream stream) {
  last_cuda_memcpy_async_source.store(source, std::memory_order_relaxed);
  injected_cuda_memcpy_async_calls.fetch_add(1, std::memory_order_relaxed);
  if (fail_next_cuda_memcpy_async.exchange(false,
                                           std::memory_order_relaxed)) {
    return 999;
  }
  return cuda_memcpy_async_real(destination, source, size, stream);
}

CUresult InjectCudaStreamSynchronize(CUstream stream) {
  injected_cuda_stream_synchronize_calls.fetch_add(
      1, std::memory_order_relaxed);
  if (fail_next_cuda_stream_synchronize.exchange(
          false, std::memory_order_relaxed)) {
    return 999;
  }
  return cuda_stream_synchronize_real(stream);
}

class ScopedCudaAsyncFault {
 public:
  explicit ScopedCudaAsyncFault(const CudaLib* cuda)
      : cuda_(const_cast<CudaLib*>(cuda)),
        memcpy_async_(cuda_->MemcpyAsync),
        stream_synchronize_(cuda_->StreamSynchronize) {
    cuda_memcpy_async_real = memcpy_async_;
    cuda_stream_synchronize_real = stream_synchronize_;
    fail_next_cuda_memcpy_async.store(false, std::memory_order_relaxed);
    fail_next_cuda_stream_synchronize.store(false,
                                            std::memory_order_relaxed);
    injected_cuda_memcpy_async_calls.store(0, std::memory_order_relaxed);
    injected_cuda_stream_synchronize_calls.store(0,
                                                  std::memory_order_relaxed);
    last_cuda_memcpy_async_source.store(0, std::memory_order_relaxed);
    cuda_->MemcpyAsync = InjectCudaMemcpyAsync;
    cuda_->StreamSynchronize = InjectCudaStreamSynchronize;
  }

  ~ScopedCudaAsyncFault() {
    cuda_->MemcpyAsync = memcpy_async_;
    cuda_->StreamSynchronize = stream_synchronize_;
    fail_next_cuda_memcpy_async.store(false, std::memory_order_relaxed);
    fail_next_cuda_stream_synchronize.store(false,
                                            std::memory_order_relaxed);
    cuda_memcpy_async_real = nullptr;
    cuda_stream_synchronize_real = nullptr;
  }

  void FailNextEnqueueAndSynchronize() {
    fail_next_cuda_memcpy_async.store(true, std::memory_order_relaxed);
    fail_next_cuda_stream_synchronize.store(true,
                                            std::memory_order_relaxed);
  }

  uint64_t memcpy_async_calls() const {
    return injected_cuda_memcpy_async_calls.load(std::memory_order_relaxed);
  }

  uint64_t stream_synchronize_calls() const {
    return injected_cuda_stream_synchronize_calls.load(
        std::memory_order_relaxed);
  }

  CUdeviceptr last_source() const {
    return last_cuda_memcpy_async_source.load(std::memory_order_relaxed);
  }

 private:
  CudaLib* cuda_;
  MemcpyAsyncFn memcpy_async_;
  StreamSynchronizeFn stream_synchronize_;
};

class ScopedCudaContextRestore {
 public:
  ScopedCudaContextRestore(const CudaLib* cuda, CUcontext context)
      : cuda_(cuda), context_(context) {}
  ~ScopedCudaContextRestore() { cuda_->SetCurrentCtx(context_); }

 private:
  const CudaLib* cuda_;
  CUcontext context_;
};

}  // namespace

TEST(RdmaLegacyConnectionToken, LiveConnectionsHaveUniqueNonzeroTokens) {
  RdmaServer server(RdmaServer::Handler{}, kMaxMsg);
  std::unordered_set<uint64_t> tokens;
  constexpr size_t kConnectionCount = 1024;
  for (size_t i = 0; i < kConnectionCount; ++i) {
    const uint64_t token = RdmaServerTestPeer::RegisterLegacyConnection(&server);
    EXPECT_NE(token, 0u);
    EXPECT_TRUE(tokens.insert(token).second);
  }
}

TEST(RdmaLegacyConnectionToken, FreshProcessesDoNotReuseStaleTokens) {
  const ChildLegacyTokenResult old_process = LegacyTokenFromFreshProcess();
  ASSERT_TRUE(old_process.transferred) << "child status=" << old_process.status;
  ASSERT_NE(old_process.token, 0u);

  const ChildLegacyTokenResult new_process =
      LegacyTokenFromFreshProcess(old_process.token);
  ASSERT_TRUE(new_process.transferred) << "child status=" << new_process.status;
  ASSERT_NE(new_process.token, 0u);
  EXPECT_FALSE(new_process.stale_token_found);
  EXPECT_NE(new_process.token, old_process.token);
}

TEST(RdmaBootstrapCapability, ServerAdvertisesDynamicOnlyAndLegacyProofInProbe) {
  RdmaServer server(RdmaServer::Handler{}, kMaxMsg);
  int sockets[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  std::thread serving(
      [&] { RdmaServerTestPeer::ServeBootstrap(&server, sockets[1]); });

  char request[rdma::kDevNameBytes];
  rdma::EncodeDevFrame(rdma::kV2ProbeDevice, 4u << 20, request);
  const bool wrote = net::WriteAll(sockets[0], request, sizeof(request));
  char reply[rdma::kV2ProbeReplyBytes];
  const bool read = wrote && net::ReadAll(sockets[0], reply, sizeof(reply));

  ::close(sockets[0]);
  serving.join();
  ASSERT_TRUE(read);
  EXPECT_TRUE(rdma::V2ProbeSupportsWriterRetirement(reply));
  EXPECT_TRUE(rdma::V2ProbeSupportsDynamicOnly(reply));
}

TEST(RdmaBootstrapCapability, KnownLiveLegacyConnectionGetsProof) {
  RdmaServer server(RdmaServer::Handler{}, kMaxMsg);
  const uint64_t token = RdmaServerTestPeer::RegisterLegacyConnection(&server);
  ASSERT_NE(token, 0u);
  ASSERT_TRUE(RequestLegacyProof(&server, token));
  EXPECT_TRUE(RdmaServerTestPeer::HasLegacyConnection(&server, token));
}

TEST(RdmaBootstrapCapability, DuplicateLegacyProofDoesNotConsumeConnection) {
  RdmaServer server(RdmaServer::Handler{}, kMaxMsg);
  const uint64_t token = RdmaServerTestPeer::RegisterLegacyConnection(&server);
  ASSERT_TRUE(RequestLegacyProof(&server, token));
  ASSERT_TRUE(RequestLegacyProof(&server, token));
  EXPECT_TRUE(RdmaServerTestPeer::HasLegacyConnection(&server, token));
}

TEST(RdmaBootstrapCapability, ClosedLegacyConnectionGetsNoProof) {
  RdmaServer server(RdmaServer::Handler{}, kMaxMsg);
  const uint64_t token = RdmaServerTestPeer::RegisterLegacyConnection(&server);
  ASSERT_TRUE(RequestLegacyProof(&server, token));
  RdmaServerTestPeer::ForgetLegacyConnection(&server, token);
  EXPECT_FALSE(RdmaServerTestPeer::HasLegacyConnection(&server, token));
  EXPECT_FALSE(RequestLegacyProof(&server, token));
}

TEST(RdmaBootstrapCapability, UnregisteredTokenGetsNoLegacyProof) {
  RdmaServer server(RdmaServer::Handler{}, kMaxMsg);
  int sockets[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  std::thread serving(
      [&] { RdmaServerTestPeer::ServeBootstrap(&server, sockets[1]); });

  char request[rdma::kDevNameBytes];
  rdma::EncodeDevFrame(rdma::kV2RetireWriterDevice, 0x12345678, request);
  const bool wrote = net::WriteAll(sockets[0], request, sizeof(request));
  char proof[rdma::kV2RetireProofBytes];
  const bool read =
      wrote && net::ReadAll(sockets[0], proof, sizeof(proof));

  ::close(sockets[0]);
  serving.join();
  ASSERT_TRUE(wrote);
  EXPECT_FALSE(read);
}

TEST(RdmaServerStartup, AutoModeKeepsFirstActiveDeviceSemantics) {
  ScopedEnv configured_device("DFKV_RDMA_DEV", nullptr);
  ScopedEnv recv_segment("DFKV_RDMA_RECV_SEGMENT_SIZE", "1048576");
  RdmaServer server(
      [](uint8_t, const BlockKey&, uint64_t, uint64_t, const char*, uint64_t,
         std::string*, size_t*) { return Status::kInvalid; },
      kMaxMsg);
  RdmaServerTestPeer::SetDiscovery(
      &server,
      [](const std::vector<std::string>& requested,
         rdma::RdmaDiscoveryPolicy policy) {
        EXPECT_TRUE(requested.empty());
        EXPECT_EQ(policy, rdma::RdmaDiscoveryPolicy::kActiveOnly);
        rdma::RdmaDevInfo first;
        first.name = "mlx5_first";
        first.active = true;
        rdma::RdmaDevInfo second;
        second.name = "mlx5_second";
        second.active = true;
        rdma::RdmaDiscoveryResult result;
        result.devices = {first, second};
        return result;
      });
  std::vector<std::string> initialized;
  RdmaServerTestPeer::SetInitializeAnchor(
      &server,
      [&initialized](
          const std::string& device, void*, size_t) {
        initialized.push_back(device);
        return std::make_unique<rdma::RcEndpoint>();
      });

  ASSERT_EQ(server.Start(0), Status::kOk);
  EXPECT_EQ(server.DeviceNames(),
            (std::vector<std::string>{"mlx5_first"}));
  EXPECT_EQ(initialized, server.DeviceNames());
  EXPECT_EQ(RdmaServerTestPeer::AnchorCount(server), 1u);
  EXPECT_EQ(RdmaServerTestPeer::RailStatsCount(server), 1u);
  EXPECT_EQ(RdmaServerTestPeer::RegisteredRailCount(server), 1u);
  server.Stop();
}

// RDS-13: a rail that cannot register memory (RLIMIT_MEMLOCK too low, say)
// refuses startup. Each connection registers its own slots, so without a
// trial at Start the server came up ready and then refused every client.
TEST(RdmaServerStartup, ARailThatCannotRegisterMemoryRefusesStartup) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ConfigureTestRecvSegment();
  RdmaServer server(
      [](uint8_t, const BlockKey&, uint64_t, uint64_t, const char*, uint64_t,
         std::string*, size_t*) { return Status::kInvalid; },
      kMaxMsg);
  std::vector<std::string> tried;
  RdmaServerTestPeer::SetStartupRegistration(
      &server, [&tried](const std::string& device) {
        tried.push_back(device);
        return false;
      });
  EXPECT_EQ(server.Start(0), Status::kIOError);
  EXPECT_FALSE(tried.empty()) << "no startup registration was tried";
  EXPECT_EQ(RdmaServerTestPeer::AnchorCount(server), 0u);
  EXPECT_EQ(RdmaServerTestPeer::RegisteredRailCount(server), 0u);
  server.Stop();
}

TEST(RdmaServerStartup, ExplicitInactiveRailRemainsInFixedTopology) {
  ScopedEnv recv_segment("DFKV_RDMA_RECV_SEGMENT_SIZE", "1048576");
  RdmaServer server(
      [](uint8_t, const BlockKey&, uint64_t, uint64_t, const char*, uint64_t,
         std::string*, size_t*) { return Status::kInvalid; },
      kMaxMsg, "mlx5_active,mlx5_down,mlx5_active");
  RdmaServerTestPeer::SetDiscovery(
      &server,
      [](const std::vector<std::string>& requested,
         rdma::RdmaDiscoveryPolicy policy) {
        EXPECT_EQ(requested,
                  (std::vector<std::string>{"mlx5_active", "mlx5_down"}));
        EXPECT_EQ(policy, rdma::RdmaDiscoveryPolicy::kAllowInactive);
        rdma::RdmaDevInfo active;
        active.name = "mlx5_active";
        active.active = true;
        rdma::RdmaDevInfo inactive;
        inactive.name = "mlx5_down";
        inactive.active = false;
        rdma::RdmaDiscoveryResult result;
        result.devices = {active, inactive};
        return result;
      });
  std::vector<std::string> initialized;
  RdmaServerTestPeer::SetInitializeAnchor(
      &server,
      [&initialized](
          const std::string& device, void*, size_t) {
        initialized.push_back(device);
        return std::make_unique<rdma::RcEndpoint>();
      });

  ASSERT_EQ(server.Start(0), Status::kOk);
  EXPECT_EQ(server.DeviceNames(),
            (std::vector<std::string>{"mlx5_active", "mlx5_down"}));
  EXPECT_EQ(initialized, server.DeviceNames());
  EXPECT_EQ(RdmaServerTestPeer::AnchorCount(server), 2u);
  EXPECT_EQ(RdmaServerTestPeer::RailStatsCount(server), 2u);
  EXPECT_EQ(RdmaServerTestPeer::RegisteredRailCount(server), 2u);
  server.Stop();
}

TEST(RdmaServerStartup, PartialAnchorMaterializationFailsAtomically) {
  ScopedEnv recv_segment("DFKV_RDMA_RECV_SEGMENT_SIZE", "1048576");
  RdmaServer server(
      [](uint8_t, const BlockKey&, uint64_t, uint64_t, const char*, uint64_t,
         std::string*, size_t*) { return Status::kInvalid; },
      kMaxMsg, "mlx5_0,mlx5_1");
  RdmaServerTestPeer::SetDiscovery(
      &server,
      [](const std::vector<std::string>&,
         rdma::RdmaDiscoveryPolicy policy) {
        EXPECT_EQ(policy, rdma::RdmaDiscoveryPolicy::kAllowInactive);
        rdma::RdmaDevInfo first;
        first.name = "mlx5_0";
        first.active = true;
        rdma::RdmaDevInfo second;
        second.name = "mlx5_1";
        second.active = false;
        rdma::RdmaDiscoveryResult result;
        result.devices = {first, second};
        return result;
      });
  size_t attempts = 0;
  RdmaServerTestPeer::SetInitializeAnchor(
      &server,
      [&attempts](
          const std::string&, void*, size_t) -> std::unique_ptr<rdma::RcEndpoint> {
        if (++attempts == 2) return nullptr;
        return std::make_unique<rdma::RcEndpoint>();
      });

  EXPECT_EQ(server.Start(0), Status::kIOError);
  EXPECT_EQ(attempts, 2u);
  EXPECT_EQ(server.DeviceNames(),
            (std::vector<std::string>{"mlx5_0", "mlx5_1"}));
  EXPECT_EQ(RdmaServerTestPeer::AnchorCount(server), 0u);
  EXPECT_EQ(RdmaServerTestPeer::RailStatsCount(server), 0u);
  EXPECT_EQ(RdmaServerTestPeer::RegisteredRailCount(server), 0u);
}

TEST(RdmaServerStartup, ExplicitResolutionFailureNeverShrinksTopology) {
  RdmaServer server(
      [](uint8_t, const BlockKey&, uint64_t, uint64_t, const char*, uint64_t,
         std::string*, size_t*) { return Status::kInvalid; },
      kMaxMsg, "mlx5_present,mlx5_missing");
  RdmaServerTestPeer::SetDiscovery(
      &server,
      [](const std::vector<std::string>& requested,
         rdma::RdmaDiscoveryPolicy policy) {
        EXPECT_EQ(requested,
                  (std::vector<std::string>{"mlx5_present", "mlx5_missing"}));
        EXPECT_EQ(policy, rdma::RdmaDiscoveryPolicy::kAllowInactive);
        rdma::RdmaDiscoveryResult result;
        result.status =
            rdma::RdmaDiscoveryStatus::kConfiguredDeviceMissing;
        result.failed_device = "mlx5_missing";
        rdma::RdmaDevInfo observed;
        observed.name = "mlx5_present";
        observed.active = true;
        result.observed_devices = {observed};
        return result;
      });
  size_t anchor_attempts = 0;
  RdmaServerTestPeer::SetInitializeAnchor(
      &server,
      [&anchor_attempts](
          const std::string&, void*, size_t) -> std::unique_ptr<rdma::RcEndpoint> {
        ++anchor_attempts;
        return std::make_unique<rdma::RcEndpoint>();
      });

  testing::internal::CaptureStderr();
  EXPECT_EQ(server.Start(0), Status::kIOError);
  const std::string logs = testing::internal::GetCapturedStderr();
  EXPECT_NE(logs.find("configured=2 initialized=0 probed=1 "
                      "observed_ACTIVE=1 observed_inactive=(none) "
                      "unresolved=mlx5_missing"),
            std::string::npos)
      << logs;
  EXPECT_EQ(logs.find(" ACTIVE=0 inactive=(none)"), std::string::npos)
      << logs;
  EXPECT_EQ(anchor_attempts, 0u);
  EXPECT_EQ(server.DeviceNames(),
            (std::vector<std::string>{"mlx5_present", "mlx5_missing"}));
  EXPECT_EQ(RdmaServerTestPeer::AnchorCount(server), 0u);
  EXPECT_EQ(RdmaServerTestPeer::RailStatsCount(server), 0u);
  EXPECT_EQ(RdmaServerTestPeer::RegisteredRailCount(server), 0u);
}

TEST(RdmaSafety, LocalRailFailureQuarantinesButPeerFailureDoesNot) {
  rdma::RailPolicyConfig config;
  config.credits_per_rail = 2;
  config.error_threshold = 1;
  config.quarantine_us = 1000;
  config.latency_weight = 1;
  config.error_penalty_us = 100;
  rdma::RailPolicy policy(2, config);

  auto local = policy.TryAcquire(2, 100);
  ASSERT_TRUE(local);
  ASSERT_EQ(local->rail, 0u);
  policy.Complete(*local, 10, rdma::RailCompletion::kRailFailure, 200);
  auto stats = policy.Snapshot(200);
  ASSERT_EQ(stats.size(), 2u);
  EXPECT_EQ(stats[0].errors, 1u);
  EXPECT_EQ(stats[0].endpoint_errors, 0u);
  EXPECT_EQ(stats[0].consecutive_errors, 1u);
  EXPECT_EQ(stats[0].quarantines, 1u);
  EXPECT_TRUE(stats[0].quarantined);
  EXPECT_EQ(stats[0].inflight, 0u);
  EXPECT_EQ(stats[0].credits, 2u);

  auto endpoint = policy.TryAcquire(2, 200);
  ASSERT_TRUE(endpoint);
  ASSERT_EQ(endpoint->rail, 1u);
  policy.Complete(*endpoint, 10, rdma::RailCompletion::kEndpointFailure, 210);
  stats = policy.Snapshot(210);
  EXPECT_EQ(stats[1].errors, 0u);
  EXPECT_EQ(stats[1].endpoint_errors, 1u);
  EXPECT_EQ(stats[1].consecutive_errors, 0u);
  EXPECT_EQ(stats[1].quarantines, 0u);
  EXPECT_FALSE(stats[1].quarantined);
  EXPECT_EQ(stats[1].inflight, 0u);
  EXPECT_EQ(stats[1].credits, 2u);
}

TEST(RdmaSafety, OperationExclusionAppliesToPreferredAndFallbackRails) {
  rdma::RailPolicy policy(
      3, rdma::RailPolicyConfig{/*credits_per_rail=*/2,
                                /*error_threshold=*/3,
                                /*quarantine_us=*/1000,
                                /*latency_weight=*/1,
                                /*error_penalty_us=*/100});
  const std::vector<uint8_t> preferred{1, 1, 0};
  const std::vector<uint8_t> fallback{0, 0, 1};

  auto preferred_retry = rdma::AcquireWithFallback(
      policy, 1, 0, preferred, fallback, /*excluded=*/{1, 0, 0});
  ASSERT_TRUE(preferred_retry);
  EXPECT_EQ(preferred_retry->rail, 1u);
  policy.Complete(*preferred_retry, 1, rdma::RailCompletion::kSuccess, 100);

  auto fallback_retry = rdma::AcquireWithFallback(
      policy, 1, 0, preferred, fallback, /*excluded=*/{1, 1, 0});
  ASSERT_TRUE(fallback_retry);
  EXPECT_EQ(fallback_retry->rail, 2u);
  policy.Complete(*fallback_retry, 1, rdma::RailCompletion::kSuccess, 101);

  EXPECT_TRUE(rdma::HasUnexcludedRail(preferred, fallback, {1, 1, 0}));
  EXPECT_FALSE(rdma::HasUnexcludedRail({1}, {}, {1}));
  EXPECT_FALSE(
      rdma::AcquireWithFallback(policy, 1, 0, {1, 0, 0}, {}, {1, 0, 0}))
      << "admission must not silently bypass an operation-local exclusion";
}

TEST(RdmaPeerRails, NineRailGpuPeerUsesOnlyPrimaryTier) {
  const std::vector<std::string> devices{
      "gpu0", "gpu1", "gpu2", "gpu3", "gpu4",
      "gpu5", "gpu6", "gpu7", "cpu0"};
  std::vector<std::vector<uint8_t>> tiers;
  std::string error;
  ASSERT_TRUE(rdma::ParseRailTiers(
      "gpu0|gpu1|gpu2|gpu3|gpu4|gpu5|gpu6|gpu7;cpu0", devices,
      &tiers, &error))
      << error;
  ASSERT_EQ(tiers.size(), 2u);
  EXPECT_EQ(tiers[0],
            (std::vector<uint8_t>{1, 1, 1, 1, 1, 1, 1, 1, 0}));
  EXPECT_EQ(tiers[1],
            (std::vector<uint8_t>{0, 0, 0, 0, 0, 0, 0, 0, 1}));

  rdma::PeerTopologyStore store(devices, tiers, /*require_complete=*/true);
  PeerTopology topology{
      "gpu-peer", 41, true,
      {{"gpu0", true}, {"gpu1", true}, {"gpu2", true},
       {"gpu3", true}, {"gpu4", true}, {"gpu5", true},
       {"gpu6", true}, {"gpu7", true}, {"cpu0", true}}};
  ASSERT_TRUE(store.Update(topology));
  const auto snapshot = store.Snapshot("gpu-peer");
  ASSERT_TRUE(snapshot->complete);
  EXPECT_EQ(snapshot->generation, 41u);
  EXPECT_EQ(snapshot->compatible,
            (std::vector<uint8_t>{1, 1, 1, 1, 1, 1, 1, 1, 0}));

  rdma::RailPolicy policy(9);
  auto lease = rdma::AcquireHighestTier(
      policy, 1, 0, snapshot->compatible, snapshot->compatible, {});
  ASSERT_TRUE(lease);
  EXPECT_LT(lease->rail, 8u);
  policy.Complete(*lease, 1, rdma::RailCompletion::kSuccess, 101);
  EXPECT_EQ(policy.Snapshot(101)[8].selections, 0u);
}

TEST(RdmaPeerRails, CpuPeerUsesItsFirstSharedRailWithoutRejection) {
  const std::vector<std::string> devices{
      "gpu0", "gpu1", "gpu2", "gpu3", "gpu4",
      "gpu5", "gpu6", "gpu7", "cpu0"};
  std::vector<std::vector<uint8_t>> tiers;
  ASSERT_TRUE(rdma::ParseRailTiers(
      "gpu0|gpu1|gpu2|gpu3|gpu4|gpu5|gpu6|gpu7;cpu0", devices,
      &tiers, nullptr));
  rdma::PeerTopologyStore store(devices, tiers, /*require_complete=*/true);
  ASSERT_TRUE(store.Update(
      PeerTopology{"cpu-peer", 7, true, {{"cpu0", true}}}));
  const auto snapshot = store.Snapshot("cpu-peer");
  ASSERT_EQ(snapshot->compatible,
            (std::vector<uint8_t>{0, 0, 0, 0, 0, 0, 0, 0, 1}));

  rdma::RailPolicy policy(9);
  auto lease = rdma::AcquireHighestTier(
      policy, 1, 0, snapshot->compatible,
      /*preferred=*/{1, 0, 0, 0, 0, 0, 0, 0, 0},
      /*fallback=*/{1, 1, 1, 1, 1, 1, 1, 1, 1});
  ASSERT_TRUE(lease);
  EXPECT_EQ(lease->rail, 8u);
  policy.Complete(*lease, 1, rdma::RailCompletion::kSuccess, 101);
  const auto stats = policy.Snapshot(101);
  EXPECT_EQ(stats[8].selections, 1u);
  EXPECT_EQ(stats[8].errors, 0u);
  EXPECT_EQ(stats[8].endpoint_errors, 0u);
}

TEST(RdmaPeerRails, PrimaryFailureUses200GFallbackAndRecoveryIsStable) {
  const std::vector<std::string> devices{"gpu400_0", "gpu400_1", "ib200"};
  std::vector<std::vector<uint8_t>> tiers;
  ASSERT_TRUE(rdma::ParseRailTiers("gpu400_0|gpu400_1;ib200", devices, &tiers,
                                   nullptr));
  rdma::PeerTopologyStore store(devices, tiers, /*require_complete=*/true);

  ASSERT_TRUE(store.Update(PeerTopology{
      "peer", 10, true,
      {{"ib200", true}, {"gpu400_1", true}, {"gpu400_0", false}}}));
  const auto partial = store.Snapshot("peer");
  EXPECT_EQ(partial->compatible, (std::vector<uint8_t>{0, 1, 0}));

  ASSERT_TRUE(store.Update(PeerTopology{
      "peer", 11, true,
      {{"gpu400_0", false}, {"gpu400_1", false}, {"ib200", true}}}));
  const auto fallback = store.Snapshot("peer");
  EXPECT_EQ(fallback->compatible, (std::vector<uint8_t>{0, 0, 1}));
  EXPECT_EQ(partial->compatible, (std::vector<uint8_t>{0, 1, 0}))
      << "published snapshots must remain immutable";

  ASSERT_TRUE(store.Update(PeerTopology{
      "peer", 12, true,
      {{"gpu400_0", true}, {"gpu400_1", false}, {"ib200", true}}}));
  const auto recovered = store.Snapshot("peer");
  EXPECT_EQ(recovered->compatible, (std::vector<uint8_t>{1, 0, 0}));
  EXPECT_FALSE(store.Update(PeerTopology{
      "peer", 12, true,
      {{"gpu400_0", false}, {"gpu400_1", false}, {"ib200", true}}}))
      << "duplicate generations must not perturb stable recovery";
  EXPECT_EQ(store.Snapshot("peer")->compatible,
            (std::vector<uint8_t>{1, 0, 0}));
}

TEST(RdmaPeerRails, NoIntersectionIsPeerNeutralAndLocallyObservable) {
  const std::vector<std::string> devices{"gpu0", "cpu0"};
  std::vector<std::vector<uint8_t>> tiers;
  ASSERT_TRUE(
      rdma::ParseRailTiers("gpu0;cpu0", devices, &tiers, nullptr));
  rdma::PeerTopologyStore store(devices, tiers, /*require_complete=*/true);
  ASSERT_TRUE(store.Update(
      PeerTopology{"peer", 3, true, {{"remote-only", true}}}));
  EXPECT_TRUE(store.Snapshot("peer")->compatible.empty());
  EXPECT_STREQ(StatusName(Status::kNoCompatibleRail), "NoCompatibleRail");

  rdma::RailPolicy policy(2, rdma::RailPolicyConfig{
                                 /*credits_per_rail=*/1,
                                 /*error_threshold=*/1,
                                 /*quarantine_us=*/1000,
                                 /*latency_weight=*/1,
                                 /*error_penalty_us=*/100});
  const auto stats = policy.Snapshot(100);
  ASSERT_EQ(stats.size(), 2u);
  for (const auto& rail : stats) {
    EXPECT_EQ(rail.selections, 0u);
    EXPECT_EQ(rail.errors, 0u);
    EXPECT_EQ(rail.endpoint_errors, 0u);
    EXPECT_EQ(rail.quarantines, 0u);
    EXPECT_FALSE(rail.quarantined);
  }
}

TEST(RdmaPeerRails, OperationPublicationFencesOlderSnapshots) {
  const std::vector<std::string> devices{"gpu0", "cpu0"};
  std::vector<std::vector<uint8_t>> tiers;
  ASSERT_TRUE(
      rdma::ParseRailTiers("gpu0;cpu0", devices, &tiers, nullptr));
  rdma::PeerTopologyStore store(devices, tiers, /*require_complete=*/true);
  ASSERT_TRUE(store.Update(
      PeerTopology{"peer", 100, true, {{"gpu0", true}}}));
  const auto operation = store.Snapshot("peer");
  ASSERT_EQ(operation->generation, 100u);
  ASSERT_TRUE(store.IsCurrent("peer", operation->publication));

  ASSERT_TRUE(store.Update(
      PeerTopology{"peer", 101, true, {{"cpu0", true}}}));
  EXPECT_FALSE(store.IsCurrent("peer", operation->publication));
  EXPECT_EQ(operation->generation, 100u);
  EXPECT_EQ(operation->compatible, (std::vector<uint8_t>{1, 0}));
  const auto current = store.Snapshot("peer");
  EXPECT_EQ(current->generation, 101u);
  EXPECT_EQ(current->compatible, (std::vector<uint8_t>{0, 1}));
}

TEST(RdmaPeerRails, PrimaryCreditPressureNeverOverflowsToLowerTier) {
  const std::vector<std::vector<uint8_t>> tiers{{1, 0}, {0, 1}};
  const auto compatible =
      rdma::HighestCompatibleTier(tiers, /*peer_healthy=*/{1, 1});
  ASSERT_EQ(compatible, (std::vector<uint8_t>{1, 0}));
  rdma::RailPolicy policy(
      2, rdma::RailPolicyConfig{/*credits_per_rail=*/1,
                                /*error_threshold=*/3,
                                /*quarantine_us=*/1000,
                                /*latency_weight=*/1,
                                /*error_penalty_us=*/100});

  auto primary = rdma::AcquireHighestTier(
      policy, 1, 0, compatible, /*preferred=*/{1, 0},
      /*fallback=*/{0, 1});
  ASSERT_TRUE(primary);
  ASSERT_EQ(primary->rail, 0u);
  EXPECT_FALSE(rdma::AcquireHighestTier(
      policy, 1, 0, compatible, /*preferred=*/{1, 0},
      /*fallback=*/{0, 1}))
      << "healthy Tier-0 pressure must wait or time out, never use Tier-1";
  const auto pressured = policy.Snapshot(101);
  EXPECT_GT(pressured[0].credits_exhausted, 0u);
  EXPECT_EQ(pressured[1].selections, 0u);
  policy.Complete(*primary, 1, rdma::RailCompletion::kSuccess, 102);
}

TEST(RdmaPeerRails, MixedVersionPolicyFailsClosedOnlyForExplicitTiers) {
  const std::vector<std::string> devices{"gpu0", "cpu0"};
  std::vector<std::vector<uint8_t>> explicit_tiers;
  ASSERT_TRUE(rdma::ParseRailTiers("gpu0;cpu0", devices, &explicit_tiers,
                                   nullptr));
  rdma::PeerTopologyStore strict_store(
      devices, explicit_tiers, /*require_complete=*/true);
  EXPECT_FALSE(strict_store.Snapshot("legacy-peer")->complete);
  EXPECT_TRUE(strict_store.Snapshot("legacy-peer")->compatible.empty());
  ASSERT_TRUE(strict_store.Update(
      PeerTopology{"legacy-peer", 1, false, {}}));
  EXPECT_TRUE(strict_store.Snapshot("legacy-peer")->compatible.empty());

  std::vector<std::vector<uint8_t>> homogeneous;
  ASSERT_TRUE(rdma::ParseRailTiers("", devices, &homogeneous, nullptr));
  rdma::PeerTopologyStore legacy_store(
      devices, homogeneous, /*require_complete=*/false);
  EXPECT_TRUE(legacy_store.Snapshot("legacy-peer")->complete);
  EXPECT_EQ(legacy_store.Snapshot("legacy-peer")->compatible,
            (std::vector<uint8_t>{1, 1}));
  ASSERT_TRUE(legacy_store.Update(
      PeerTopology{"legacy-peer", 1, false, {}}));
  EXPECT_EQ(legacy_store.Snapshot("legacy-peer")->compatible,
            (std::vector<uint8_t>{1, 1}));

  std::string error;
  EXPECT_FALSE(rdma::ParseRailTiers("gpu0;missing", devices, &homogeneous,
                                    &error));
  EXPECT_FALSE(error.empty());
}

TEST(RdmaPeerRails, ConcurrentPublicationAcquisitionAndSnapshotTeardown) {
  const std::vector<std::string> devices{"gpu0", "gpu1", "cpu0"};
  std::vector<std::vector<uint8_t>> tiers;
  ASSERT_TRUE(rdma::ParseRailTiers("gpu0|gpu1;cpu0", devices, &tiers,
                                   nullptr));
  auto store = std::make_unique<rdma::PeerTopologyStore>(
      devices, tiers, /*require_complete=*/true);
  ASSERT_TRUE(store->Update(PeerTopology{
      "peer", 2, true,
      {{"gpu0", true}, {"gpu1", false}, {"cpu0", true}}}));
  rdma::RailPolicy policy(3);

  std::atomic<bool> start{false};
  std::atomic<uint64_t> violations{0};
  std::vector<std::thread> threads;
  for (uint64_t writer = 0; writer < 2; ++writer) {
    threads.emplace_back([&, writer] {
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      for (uint64_t i = 0; i < 1000; ++i) {
        const uint64_t generation = 10 + writer * 1000 + i;
        const bool even = generation % 2 == 0;
        store->Update(PeerTopology{
            "peer", generation, true,
            {{"gpu0", even}, {"gpu1", !even}, {"cpu0", true}}});
      }
    });
  }
  for (size_t reader = 0; reader < 4; ++reader) {
    threads.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      for (size_t i = 0; i < 2000; ++i) {
        const auto snapshot = store->Snapshot("peer");
        const bool even = snapshot->generation % 2 == 0;
        const std::vector<uint8_t> expected =
            even ? std::vector<uint8_t>{1, 0, 0}
                 : std::vector<uint8_t>{0, 1, 0};
        auto lease =
            policy.TryAcquire(1, snapshot->generation, snapshot->compatible);
        if (!snapshot->complete || snapshot->compatible != expected || !lease ||
            lease->rail >= snapshot->compatible.size() ||
            snapshot->compatible[lease->rail] == 0) {
          violations.fetch_add(1, std::memory_order_relaxed);
        }
        if (lease) {
          policy.Complete(*lease, 1, rdma::RailCompletion::kSuccess,
                          snapshot->generation);
        }
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (auto& thread : threads) thread.join();
  EXPECT_EQ(violations.load(std::memory_order_relaxed), 0u);
  for (const auto& rail : policy.Snapshot(3000))
    EXPECT_EQ(rail.inflight, 0u);

  const auto retained = store->Snapshot("peer");
  std::atomic<bool> inspect{false};
  std::thread reader([retained, &inspect, &violations] {
    while (!inspect.load(std::memory_order_acquire))
      std::this_thread::yield();
    for (size_t i = 0; i < 10000; ++i) {
      if (!retained->complete || retained->compatible.empty())
        violations.fetch_add(1, std::memory_order_relaxed);
    }
  });
  inspect.store(true, std::memory_order_release);
  store.reset();
  reader.join();
  EXPECT_EQ(violations.load(std::memory_order_relaxed), 0u);
}

TEST(RdmaSafety, QuarantinedRailAllowsOneProbeAndRecoversOnlyOnSuccess) {

  rdma::RailPolicyConfig config;
  config.credits_per_rail = 64;
  config.error_threshold = 1;
  config.quarantine_us = 1000;
  config.latency_weight = 1;
  config.error_penalty_us = 100;
  rdma::RailPolicy policy(2, config);

  auto failed = policy.TryAcquire(1, 100);
  ASSERT_TRUE(failed);
  ASSERT_EQ(failed->rail, 0u);
  policy.Complete(*failed, 10, rdma::RailCompletion::kRailFailure, 200);
  auto before_cooldown = policy.TryAcquire(1, 1199);
  ASSERT_TRUE(before_cooldown);
  EXPECT_EQ(before_cooldown->rail, 1u);
  policy.Complete(*before_cooldown, 10, rdma::RailCompletion::kSuccess, 1199);

  constexpr size_t kCallers = 32;
  std::mutex gate_mu;
  std::condition_variable ready_cv;
  std::condition_variable start_cv;
  std::condition_variable completed_cv;
  size_t ready = 0;
  size_t completed = 0;
  bool start = false;
  std::vector<std::optional<rdma::RailLease>> leases(kCallers);
  std::vector<std::thread> callers;
  callers.reserve(kCallers);
  for (size_t i = 0; i < kCallers; ++i) {
    callers.emplace_back([&, i] {
      std::unique_lock<std::mutex> lock(gate_mu);
      ++ready;
      ready_cv.notify_one();
      start_cv.wait(lock, [&] { return start; });
      lock.unlock();
      leases[i] = policy.TryAcquire(1, 1200);
      lock.lock();
      ++completed;
      completed_cv.notify_one();
    });
  }
  bool all_ready = false;
  {
    std::unique_lock<std::mutex> lock(gate_mu);
    all_ready = ready_cv.wait_for(lock, std::chrono::seconds(2),
                                  [&] { return ready == kCallers; });
    start = true;
  }
  start_cv.notify_all();
  EXPECT_TRUE(all_ready) << "callers did not reach the start barrier";
  {
    std::unique_lock<std::mutex> lock(gate_mu);
    EXPECT_TRUE(completed_cv.wait_for(lock, std::chrono::seconds(2),
                                      [&] { return completed == kCallers; }))
        << "rail-policy callers deadlocked";
  }
  for (auto& caller : callers) caller.join();

  size_t probe_index = kCallers;
  size_t rail0_leases = 0;
  for (size_t i = 0; i < leases.size(); ++i) {
    ASSERT_TRUE(leases[i]) << i;
    if (leases[i]->rail == 0) {
      ++rail0_leases;
      probe_index = i;
    } else {
      EXPECT_EQ(leases[i]->rail, 1u);
    }
  }
  ASSERT_EQ(rail0_leases, 1u);
  auto stats = policy.Snapshot(1200);
  EXPECT_TRUE(stats[0].quarantined);
  EXPECT_TRUE(stats[0].recovery_probe);
  EXPECT_EQ(stats[0].recoveries, 0u);

  for (size_t i = 0; i < leases.size(); ++i) {
    if (i == probe_index) continue;
    policy.Complete(*leases[i], 10, rdma::RailCompletion::kSuccess, 1201);
  }
  policy.Complete(*leases[probe_index], 10,
                  rdma::RailCompletion::kEndpointFailure, 1201);
  stats = policy.Snapshot(1201);
  EXPECT_TRUE(stats[0].quarantined);
  EXPECT_FALSE(stats[0].recovery_probe);
  EXPECT_EQ(stats[0].recoveries, 0u);
  EXPECT_EQ(stats[0].endpoint_errors, 1u);

  auto failed_probe = policy.TryAcquire(1, 1202);
  ASSERT_TRUE(failed_probe);
  ASSERT_EQ(failed_probe->rail, 0u);
  policy.Complete(*failed_probe, 10, rdma::RailCompletion::kRailFailure, 1202);
  stats = policy.Snapshot(1202);
  EXPECT_EQ(stats[0].quarantines, 2u);
  EXPECT_TRUE(stats[0].quarantined);
  EXPECT_FALSE(stats[0].recovery_probe);

  auto other_success = policy.TryAcquire(1, 2201);
  ASSERT_TRUE(other_success);
  ASSERT_EQ(other_success->rail, 1u);
  policy.Complete(*other_success, 10, rdma::RailCompletion::kSuccess, 2201);
  stats = policy.Snapshot(2201);
  EXPECT_EQ(stats[0].recoveries, 0u)
      << "success on another rail must not recover the quarantined rail";

  auto successful_probe = policy.TryAcquire(1, 2202);
  ASSERT_TRUE(successful_probe);
  ASSERT_EQ(successful_probe->rail, 0u);
  policy.Complete(*successful_probe, 10, rdma::RailCompletion::kSuccess, 2203);
  stats = policy.Snapshot(2203);
  EXPECT_EQ(stats[0].recoveries, 1u);
  EXPECT_EQ(stats[0].consecutive_errors, 0u);
  EXPECT_FALSE(stats[0].quarantined);
  EXPECT_FALSE(stats[0].recovery_probe);

  auto admitted_after_recovery = policy.TryAcquire(1, 2204);
  ASSERT_TRUE(admitted_after_recovery);
  EXPECT_EQ(admitted_after_recovery->rail, 0u);
  policy.Complete(*admitted_after_recovery, 10,
                  rdma::RailCompletion::kSuccess, 2205);
}

TEST(RdmaSafety, CompletionDeadlineUsesOneAbsoluteBudget) {
  using Clock = CompletionDeadline::Clock;
  const auto start = Clock::time_point(std::chrono::milliseconds(100));
  CompletionDeadline deadline(50, start);
  EXPECT_EQ(deadline.RemainingAt(start), 50);
  EXPECT_EQ(deadline.RemainingAt(start + std::chrono::milliseconds(30)), 20);
  // A partial completion at +30 ms does not create a fresh 50-ms wait.
  EXPECT_EQ(deadline.RemainingAt(start + std::chrono::milliseconds(49)), 1);
  EXPECT_EQ(deadline.RemainingAt(start + std::chrono::milliseconds(50)), 0);
  CompletionDeadline infinite(-1, start);
  EXPECT_EQ(infinite.RemainingAt(start + std::chrono::hours(24)), -1);
}

TEST(RdmaSafety, PartialMultiRailPoolGrowthRollsBackBeforePublication) {
  std::vector<FakePoolRail> rails(3, FakePoolRail{64, 64});
  rails[1].reject_stage = true;
  size_t published_bytes = 64;
  const bool registered = rdma::RunPoolMrTransaction(
      rails.size(),
      [&](size_t rail) { return rails[rail].Stage(128); },
      [&](size_t rail) { rails[rail].Commit(); },
      [&](size_t rail) { rails[rail].Rollback(); });
  if (registered) published_bytes = 128;

  EXPECT_FALSE(registered);
  EXPECT_EQ(published_bytes, 64u);
  for (const auto& rail : rails) {
    EXPECT_EQ(rail.declared, 64u);
    EXPECT_EQ(rail.staged, 64u);
    EXPECT_EQ(rail.commits, 0u);
  }
  EXPECT_EQ(rails[0].rollbacks, 1u);
  EXPECT_EQ(rails[1].rollbacks, 0u);
  EXPECT_EQ(rails[2].rollbacks, 0u);
}

TEST(RdmaSafety, SuccessfulMultiRailPoolGrowthKeepsOldRangeUsable) {
  std::vector<FakePoolRail> rails(3, FakePoolRail{64, 64});
  size_t published_bytes = 64;
  const bool registered = rdma::RunPoolMrTransaction(
      rails.size(),
      [&](size_t rail) { return rails[rail].Stage(128); },
      [&](size_t rail) { rails[rail].Commit(); },
      [&](size_t rail) { rails[rail].Rollback(); });
  if (registered) published_bytes = 128;

  ASSERT_TRUE(registered);
  EXPECT_EQ(published_bytes, 128u);
  for (const auto& rail : rails) {
    EXPECT_EQ(rail.declared, 128u);
    EXPECT_EQ(rail.commits, 1u);
    EXPECT_EQ(rail.rollbacks, 0u);
    EXPECT_TRUE(rail.Covers(8, 16));
    EXPECT_TRUE(rail.Covers(96, 16));
  }
}

TEST(RdmaReadPrimitive, HostDestinationIsByteExact) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  constexpr size_t kBytes = 1024 * 1024;
  const char* dev = std::getenv("DFKV_RDMA_DEV");
  rdma::RcEndpoint source;
  rdma::RcEndpoint initiator;
  if (!source.Open(dev, 4096, 1, 1,
                   /*direct_io_buffers=*/true, kBytes))
    GTEST_SKIP() << "no usable RDMA device";
  if (!initiator.Open(dev, 4096, 1))
    GTEST_SKIP() << "no usable RDMA device";
  const rdma::QpInfo source_info = source.Local();
  const rdma::QpInfo initiator_info = initiator.Local();
  ASSERT_TRUE(source.Connect(initiator_info));
  ASSERT_TRUE(initiator.Connect(source_info));

  for (size_t i = 0; i < kBytes; ++i)
    source.dbuf(0)[i] = static_cast<char>((i * 17 + 31) & 0xff);
  std::vector<char> destination(kBytes, '\0');
  ibv_mr* destination_mr =
      initiator.RegisterTransient(destination.data(), destination.size());
  ASSERT_NE(destination_mr, nullptr);
  ASSERT_TRUE(initiator.PostRead(
      0, destination.data(), destination.size(), destination_mr,
      reinterpret_cast<uint64_t>(source.dbuf(0)), source.dmr(0)->rkey));
  ibv_wc completion{};
  ASSERT_EQ(initiator.WaitComp(&completion, 1, 10000), 1);
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_RDMA_READ);
  EXPECT_EQ(std::memcmp(destination.data(), source.dbuf(0), kBytes), 0);
  initiator.ReleaseTransient(destination_mr);
}
TEST(RdmaReadPrimitive, ConcurrentOneMiBThroughput) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  constexpr size_t kBytes = 1024 * 1024;
  constexpr size_t kThreads = 32;
  constexpr size_t kIterations = 512;
  const char* dev = std::getenv("DFKV_RDMA_DEV");
  struct Pair {
    std::unique_ptr<rdma::RcEndpoint> source;
    std::unique_ptr<rdma::RcEndpoint> initiator;
    std::vector<char> destination;
    ibv_mr* destination_mr = nullptr;
  };
  std::vector<std::unique_ptr<Pair>> pairs;
  pairs.reserve(kThreads);
  for (size_t thread = 0; thread < kThreads; ++thread) {
    auto pair = std::make_unique<Pair>();
    pair->source = std::make_unique<rdma::RcEndpoint>();
    pair->initiator = std::make_unique<rdma::RcEndpoint>();
    if (!pair->source->Open(dev, 4096, 1, 1,
                            /*direct_io_buffers=*/true, kBytes))
      GTEST_SKIP() << "no usable RDMA device";
    if (!pair->initiator->Open(dev, 4096, 1))
      GTEST_SKIP() << "no usable RDMA device";
    const rdma::QpInfo source_info = pair->source->Local();
    const rdma::QpInfo initiator_info = pair->initiator->Local();
    ASSERT_TRUE(pair->source->Connect(initiator_info));
    ASSERT_TRUE(pair->initiator->Connect(source_info));
    std::memset(pair->source->dbuf(0), static_cast<int>(thread + 1), kBytes);
    pair->destination.assign(kBytes, '\0');
    pair->destination_mr = pair->initiator->RegisterTransient(
        pair->destination.data(), pair->destination.size());
    ASSERT_NE(pair->destination_mr, nullptr);
    pairs.push_back(std::move(pair));
  }

  std::atomic<bool> start{false};
  std::atomic<size_t> failures{0};
  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  const auto begin = std::chrono::steady_clock::now();
  for (size_t thread = 0; thread < kThreads; ++thread) {
    workers.emplace_back([&, thread] {
      Pair& pair = *pairs[thread];
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      for (size_t i = 0; i < kIterations; ++i) {
        if (!pair.initiator->PostRead(
                0, pair.destination.data(), pair.destination.size(),
                pair.destination_mr,
                reinterpret_cast<uint64_t>(pair.source->dbuf(0)),
                pair.source->dmr(0)->rkey)) {
          failures.fetch_add(1, std::memory_order_relaxed);
          break;
        }
        ibv_wc completion{};
        if (pair.initiator->WaitComp(&completion, 1, 10000) != 1 ||
            completion.status != IBV_WC_SUCCESS ||
            completion.opcode != IBV_WC_RDMA_READ) {
          failures.fetch_add(1, std::memory_order_relaxed);
          break;
        }
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (auto& worker : workers) worker.join();
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
          .count();
  const double gbps =
      static_cast<double>(kThreads * kIterations * kBytes) / seconds / 1e9;
  std::fprintf(stderr,
               "RDMA_READ_BENCH bytes=%zu threads=%zu iterations=%zu "
               "seconds=%.6f goodput=%.3f_GBps\n",
               kBytes, kThreads, kIterations, seconds, gbps);
  EXPECT_EQ(failures.load(std::memory_order_relaxed), 0u);
  for (size_t thread = 0; thread < kThreads; ++thread) {
    EXPECT_EQ(std::memcmp(pairs[thread]->destination.data(),
                          pairs[thread]->source->dbuf(0), kBytes),
              0);
    pairs[thread]->initiator->ReleaseTransient(
        pairs[thread]->destination_mr);
  }
}

TEST(RdmaSafety, OverridesRejectMalformedVectorsBeforeAcquiringQp) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaTransport transport(kMaxMsg);
  const std::vector<BlockKey> keys = {{1, 2}, {3, 4}};
  std::vector<uint64_t> value_lengths;
  const auto mismatched =
      transport.RangeInto("unused", keys, {{nullptr, 0}}, &value_lengths);
  ASSERT_EQ(mismatched.size(), keys.size());
  EXPECT_EQ(mismatched[0], Status::kInvalid);
  EXPECT_EQ(mismatched[1], Status::kInvalid);

  const auto null_write = transport.CacheMany(
      "unused", {CacheItem{keys[0], nullptr, 1}});
  ASSERT_EQ(null_write.size(), 1u);
  EXPECT_EQ(null_write[0], Status::kInvalid);

  const void* nonnull = reinterpret_cast<const void*>(uintptr_t{1});
  CacheSrcMulti overflow{
      keys[0],
      {{nonnull, std::numeric_limits<size_t>::max()}, {nonnull, 1}}};
  const auto overflow_status =
      transport.CacheFromMulti("unused", {overflow});
  ASSERT_EQ(overflow_status.size(), 1u);
  EXPECT_EQ(overflow_status[0], Status::kInvalid);

  EXPECT_EQ(transport.ExistMany("unused", keys, nullptr),
            std::vector<Status>(keys.size(), Status::kInvalid));
  EXPECT_EQ(transport.Members("unused", nullptr), Status::kInvalid);
  EXPECT_EQ(CounterVal(transport.MetricsText(),
                       "dfkv_rdma_client_conns_opened_total"), 0);
}



// Direct transport ExistMany: windowed batch existence probe on one connection.
// Must be correct across multiple send windows (N > depth) with mixed hit/miss.
// Recent-REMOVE memory (CLI-02 stage M) over the RDMA control lane: a node
// with DFKV_REMOVE_MEMORY_MS answers kRemovedWithin; one without it answers
// kInvalid, which the client takes as unknown without charging the node.
TEST(RdmaLoopback, RemovedWithinIsAnsweredOverRdma) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device (load rdma_rxe for Soft-RoCE)";
  ::setenv("DFKV_REMOVE_MEMORY_MS", "600000", 1);
  RdmaNode node("rmem");
  ::unsetenv("DFKV_REMOVE_MEMORY_MS");
  std::this_thread::sleep_for(std::chrono::milliseconds(300));  // see the TCP test
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  const std::string v(64, 'v');
  ASSERT_TRUE(c.Put("gone", v.data(), v.size()));
  ASSERT_TRUE(c.Remove("gone"));
  ASSERT_TRUE(c.Put("kept", v.data(), v.size()));
  std::vector<std::string> keys = {"gone", "kept"};
  for (int i = 0; i < 1500; ++i) keys.push_back("other-" + std::to_string(i));
  std::vector<uint8_t> answers;
  ASSERT_EQ(c.RemovedWithin(node.addr, keys, 200, &answers), Status::kOk);
  ASSERT_EQ(answers.size(), keys.size());
  EXPECT_EQ(answers[0], 1);
  for (size_t i = 1; i < answers.size(); ++i) EXPECT_EQ(answers[i], 0) << i;
}

// Op 26 (CLI-35) over the RDMA control lane, a direct PUT recorded in the
// recent-PUT memory, and more keys than one request holds.
TEST(RdmaLoopback, RemovedWithinAgesIsAnsweredOverRdma) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device (load rdma_rxe for Soft-RoCE)";
  ::setenv("DFKV_REMOVE_MEMORY_MS", "600000", 1);
  ::setenv("DFKV_WRITE_MEMORY_ENTRIES", "65536", 1);
  RdmaNode node("rmem-ages");
  ::unsetenv("DFKV_REMOVE_MEMORY_MS");
  ::unsetenv("DFKV_WRITE_MEMORY_ENTRIES");
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  const std::string v(64, 'v');
  ASSERT_TRUE(c.Remove("again"));
  ASSERT_TRUE(c.Put("again", v.data(), v.size()));
  std::vector<std::string> keys = {"again"};
  for (int i = 0; i < 1500; ++i) keys.push_back("other-" + std::to_string(i));
  std::vector<KVClient::MemoryAnswer> a;
  ASSERT_EQ(c.RemovedWithinAges(node.addr, keys, 200, &a), Status::kOk);
  ASSERT_EQ(a.size(), keys.size());
  EXPECT_NE(a[0].removed_age, kRemovedAgeNone);
  EXPECT_LE(a[0].written_age, a[0].removed_age);
  for (size_t i = 1; i < a.size(); ++i) {
    EXPECT_EQ(a[i].removed_age, kRemovedAgeNone) << i;
    EXPECT_EQ(a[i].written_age, kRemovedAgeNone) << i;
  }
}

TEST(RdmaLoopback, RemovedWithinOfANodeWithoutTheMemoryIsUnknown) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device (load rdma_rxe for Soft-RoCE)";
  ::unsetenv("DFKV_REMOVE_MEMORY_MS");
  RdmaNode node("rmem-off");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  std::vector<uint8_t> answers;
  EXPECT_EQ(c.RemovedWithin(node.addr, {"a", "b"}, 1000, &answers),
            Status::kInvalid);
  EXPECT_EQ(answers, (std::vector<uint8_t>{2, 2}));
  const std::string v(8, 'v');
  EXPECT_TRUE(c.Put("a", v.data(), v.size())) << "the connection was kept";
  EXPECT_EQ(c.MetricsSnapshot().find("dfkv_client_peer_marked_bad_total 1"),
            std::string::npos);
}

TEST(RdmaLoopback, ExistManyWindowedMixedHitMiss) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device (load rdma_rxe for Soft-RoCE)";
  RdmaNode node("exm");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  const int N = 80;  // exceeds a single send window (depth) to exercise looping
  for (int i = 0; i < N; ++i) {
    std::string v = "v" + std::to_string(i);
    ASSERT_TRUE(c.Put("p" + std::to_string(i), v.data(), v.size())) << i;
  }
  // Interleave present (even) and absent (odd) keys using the exact namespace
  // the KVClient used for the writes.
  std::vector<BlockKey> keys;
  for (int i = 0; i < N; ++i) {
    keys.push_back(ToBlockKey(SelfHdr(), "p" + std::to_string(i)));
    keys.push_back(ToBlockKey(SelfHdr(), "absent" + std::to_string(i)));
  }
  std::vector<char> exists;
  auto sts = rt.ExistMany(node.addr, keys, &exists);
  ASSERT_EQ(exists.size(), keys.size());
  ASSERT_EQ(sts.size(), keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    bool want = (i % 2 == 0);
    EXPECT_EQ(exists[i] != 0, want) << "i=" << i;
  }
}

// out_dev names the rails a call used, each time the rail changes: one rail
// for every key of a batch is one name, not one "->name" per key. The keys
// of a batch share one connection, so on two rails too a batch names one
// rail; a second name only after a failed window was retried on another
// rail, and never the same rail twice in a row.
TEST(RdmaLoopback, RangeIntoMultiNamesEachRailOnce) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("rimdev");
  const char* devs = std::getenv("DFKV_RDMA_DEV");
  const std::string list = devs ? devs : "";
  for (const bool one_rail : {true, false}) {
    if (!one_rail && list.find(',') == std::string::npos) continue;
    std::optional<ScopedEnv> rail;
    if (one_rail && !list.empty())
      rail.emplace("DFKV_RDMA_DEV", list.substr(0, list.find(',')).c_str());
    RdmaTransport rt(kMaxMsg);
    KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
    constexpr int kKeys = 8;
    std::vector<BlockKey> keys;
    std::vector<std::string> bufs(kKeys, std::string(4096, '\0'));
    std::vector<RangeDstMulti> dsts(kKeys);
    const std::string tag = one_rail ? "rim1-" : "rim2-";
    for (int i = 0; i < kKeys; ++i) {
      const std::string v(4096, static_cast<char>('a' + i));
      ASSERT_TRUE(c.Put(tag + std::to_string(i), v.data(), v.size())) << i;
      keys.push_back(ToBlockKey(SelfHdr(), tag + std::to_string(i)));
      dsts[i].payloads.emplace_back(&bufs[i][0], bufs[i].size());
    }
    std::vector<size_t> lens;
    std::string dev;
    const long failures_before =
        CounterVal(rt.MetricsText(), "dfkv_rdma_client_pull_failures_total");
    const auto sts = rt.RangeIntoMulti(node.addr, keys, dsts, &lens, &dev);
    const long failures =
        CounterVal(rt.MetricsText(), "dfkv_rdma_client_pull_failures_total") -
        failures_before;
    for (int i = 0; i < kKeys; ++i) ASSERT_EQ(sts[i], Status::kOk) << i;
    ASSERT_FALSE(dev.empty());
    if (one_rail) {
      EXPECT_EQ(dev.find("->"), std::string::npos)
          << "one rail named more than once: " << dev;
      continue;
    }
    std::vector<std::string> names;
    for (size_t start = 0;;) {
      const size_t arrow = dev.find("->", start);
      names.push_back(dev.substr(start, arrow - start));
      if (arrow == std::string::npos) break;
      start = arrow + 2;
    }
    if (failures == 0) EXPECT_EQ(names.size(), 1u) << dev;
    EXPECT_LE(names.size(), 2u) << dev;
    for (size_t i = 1; i < names.size(); ++i)
      EXPECT_NE(names[i], names[i - 1]) << "a rail named twice in a row: "
                                        << dev;
  }
}

TEST(RdmaLoopback, BatchExistMixedHitMiss) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("bex");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  const int N = 64;
  for (int i = 0; i < N; ++i) {
    std::string v = "v" + std::to_string(i);
    ASSERT_TRUE(c.Put("e" + std::to_string(i), v.data(), v.size())) << i;
  }
  std::vector<std::string> probe;
  for (int i = 0; i < N; ++i) {
    probe.push_back("e" + std::to_string(i));        // present
    probe.push_back("e" + std::to_string(i) + "_x"); // absent
  }

  auto er = c.BatchExist(probe);
  ASSERT_EQ(er.size(), probe.size());
  for (size_t i = 0; i < probe.size(); ++i)
    EXPECT_EQ((bool)er[i], (i % 2 == 0)) << probe[i];
}

namespace {

// Hold real acquired endpoints until every checkout is complete. Sequential
// acquisition while all earlier QPs remain held gives exact concurrency,
// independent of thread scheduling, without generating cache data.
class HeldData {
 public:
  HeldData(RdmaTransport* transport, const std::string& node)
      : transport_(transport), node_(node) {}
  ~HeldData() { ReleaseAt(0); }
  HeldData(const HeldData&) = delete;
  HeldData& operator=(const HeldData&) = delete;

  bool Acquire(size_t count, bool force_new = false) {
    for (size_t i = 0; i < count; ++i) {
      auto* conn =
          RdmaTransportTestPeer::AcquireData(transport_, node_, force_new);
      if (!conn) return false;
      held_.push_back(conn);
    }
    return true;
  }
  void ReleaseAt(uint64_t now_us) {
    for (auto* conn : held_)
      RdmaTransportTestPeer::ReleaseDataAt(transport_, node_, conn, now_us);
    held_.clear();
  }
  void FailOne() {
    auto* conn = held_.back();
    held_.pop_back();
    RdmaTransportTestPeer::FailData(transport_, conn);
  }

 private:
  RdmaTransport* transport_;
  const std::string& node_;
  std::vector<RdmaTransportTestPeer::Connection*> held_;
};

constexpr uint64_t kBurstIdleUs = 60'000'000;

PeerTopology PublishedPeer(const std::string& address,
                           const std::string& id, uint64_t generation = 1) {
  PeerTopology topology;
  topology.peer_addr = address;
  topology.peer_id = id;
  topology.generation = generation;
  return topology;  // homogeneous topology, but an identified publication
}

}  // namespace

TEST(RdmaLoopback, BurstPoolRetainsRecurringConcurrency) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv pool_max("DFKV_RDMA_POOL_MAX", nullptr);
  ScopedEnv tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv idle_reaper("DFKV_RDMA_IDLE_MS", "0");
  RdmaNode node("pool-recurrence");
  ASSERT_FALSE(node.rsrv->DeviceNames().empty());
  RdmaTransport transport(kMaxMsg, node.rsrv->DeviceNames().front());
  RdmaTransportTestPeer::StopMaintenance(&transport);
  EXPECT_EQ(RdmaTransportTestPeer::PoolLimits(transport),
            (std::pair<size_t, size_t>{8, 64}));
  HeldData burst(&transport, node.addr);
  ASSERT_TRUE(burst.Acquire(12));
  burst.ReleaseAt(0);
  const long opened = CounterVal(
      transport.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  uint64_t now = 0;
  for (int recurrence = 0; recurrence < 3; ++recurrence) {
    now += kBurstIdleUs - 1;
    RdmaTransportTestPeer::MaintainIdle(&transport, now);
    ASSERT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 12u);
    ASSERT_TRUE(burst.Acquire(12));
    burst.ReleaseAt(now);
    EXPECT_EQ(CounterVal(transport.MetricsText(),
                         "dfkv_rdma_client_conns_opened_total"), opened);
  }
  const auto before = RdmaTransportTestPeer::UsedResources(transport);
  RdmaTransportTestPeer::MaintainIdle(&transport, now + kBurstIdleUs);
  EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 8u);
  const auto after = RdmaTransportTestPeer::UsedResources(transport);
  EXPECT_EQ(before.endpoints - after.endpoints, 4u);
  EXPECT_EQ(before.qps - after.qps, 4u);
  EXPECT_GT(before.registered_bytes, after.registered_bytes);
}

TEST(RdmaLoopback, BurstPoolKeepaliveDoesNotRenewUnusedEndpoints) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv pool_max("DFKV_RDMA_POOL_MAX", nullptr);
  ScopedEnv tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv idle_reaper("DFKV_RDMA_IDLE_MS", "0");
  RdmaNode node("pool-idle");
  ASSERT_FALSE(node.rsrv->DeviceNames().empty());
  RdmaTransport transport(kMaxMsg, node.rsrv->DeviceNames().front());
  RdmaTransportTestPeer::StopMaintenance(&transport);
  HeldData burst(&transport, node.addr);
  ASSERT_TRUE(burst.Acquire(12));
  burst.ReleaseAt(0);
  HeldData active(&transport, node.addr);
  ASSERT_TRUE(active.Acquire(1));
  const auto before = RdmaTransportTestPeer::UsedResources(transport);
  RdmaTransportTestPeer::MaintainIdle(&transport, kBurstIdleUs - 1, true);
  EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 11u);
  EXPECT_EQ(CounterVal(transport.MetricsText(),
                       "dfkv_rdma_client_keepalive_successes_total"), 11);
  RdmaTransportTestPeer::MaintainIdle(&transport, kBurstIdleUs);
  EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 8u);
  EXPECT_EQ(before.endpoints -
                RdmaTransportTestPeer::UsedResources(transport).endpoints, 3u);
  // Maintenance cannot destroy a checked-out QP, and continued use of this
  // one QP must not preserve the other obsolete burst endpoints.
  active.ReleaseAt(kBurstIdleUs);
  RdmaTransportTestPeer::MaintainIdle(&transport, kBurstIdleUs);
  EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 8u);
  EXPECT_EQ(before.endpoints -
                RdmaTransportTestPeer::UsedResources(transport).endpoints, 4u);
}

TEST(RdmaLoopback, BurstPoolDefaultCeilingReleasesResourcesImmediately) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv pool_max("DFKV_RDMA_POOL_MAX", nullptr);
  ScopedEnv tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv idle_reaper("DFKV_RDMA_IDLE_MS", "0");
  ScopedEnv credits("DFKV_RDMA_RAIL_CREDITS", "128");
  RdmaNode node("pool-ceiling");
  ASSERT_FALSE(node.rsrv->DeviceNames().empty());
  RdmaTransport transport(kMaxMsg, node.rsrv->DeviceNames().front());
  RdmaTransportTestPeer::StopMaintenance(&transport);
  const auto baseline = RdmaTransportTestPeer::UsedResources(transport);
  HeldData burst(&transport, node.addr);
  ASSERT_TRUE(burst.Acquire(65));
  const auto active = RdmaTransportTestPeer::UsedResources(transport);
  EXPECT_EQ(active.endpoints - baseline.endpoints, 65u);
  burst.ReleaseAt(0);
  EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 64u);
  const auto retained = RdmaTransportTestPeer::UsedResources(transport);
  EXPECT_EQ(retained.endpoints - baseline.endpoints, 64u);
  EXPECT_LT(retained.registered_bytes, active.registered_bytes);
  RdmaTransportTestPeer::MaintainIdle(&transport, kBurstIdleUs);
  EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 8u);
  EXPECT_EQ(RdmaTransportTestPeer::UsedResources(transport).endpoints -
                baseline.endpoints, 8u);
}

TEST(RdmaLoopback, BurstPoolExplicitLimitNeverExpands) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv idle_reaper("DFKV_RDMA_IDLE_MS", "0");
  for (const size_t limit : {1u, 8u}) {
    SCOPED_TRACE(limit);
    const std::string value = std::to_string(limit);
    ScopedEnv pool_max("DFKV_RDMA_POOL_MAX", value.c_str());
    RdmaNode node("pool-manual-" + value);
    ASSERT_FALSE(node.rsrv->DeviceNames().empty());
    RdmaTransport transport(kMaxMsg, node.rsrv->DeviceNames().front());
    RdmaTransportTestPeer::StopMaintenance(&transport);
    EXPECT_EQ(RdmaTransportTestPeer::PoolLimits(transport),
              (std::pair<size_t, size_t>{limit, limit}));
    const auto baseline = RdmaTransportTestPeer::UsedResources(transport);
    HeldData burst(&transport, node.addr);
    ASSERT_TRUE(burst.Acquire(limit + 2));
    burst.ReleaseAt(0);
    EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), limit);
    EXPECT_EQ(RdmaTransportTestPeer::UsedResources(transport).endpoints -
                  baseline.endpoints, limit);
    const long opened = CounterVal(
        transport.MetricsText(), "dfkv_rdma_client_conns_opened_total");
    ASSERT_TRUE(burst.Acquire(limit + 2));
    burst.ReleaseAt(0);
    EXPECT_EQ(CounterVal(transport.MetricsText(),
                         "dfkv_rdma_client_conns_opened_total") - opened, 2);
    RdmaTransportTestPeer::MaintainIdle(&transport, 2 * kBurstIdleUs);
    EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), limit);
  }
}

TEST(RdmaLoopback, CapabilityCacheFollowsIdentityPublicationAndAddress) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv idle_reaper("DFKV_RDMA_IDLE_MS", "0");
  ScopedEnv pool_max("DFKV_RDMA_POOL_MAX", "8");
  RdmaNode node("caps-publication");
  RdmaNode moved("caps-moved");
  ASSERT_FALSE(node.rsrv->DeviceNames().empty());
  RdmaTransport transport(kMaxMsg, node.rsrv->DeviceNames().front());
  RdmaTransportTestPeer::StopMaintenance(&transport);
  auto topology = PublishedPeer(node.addr, "first");
  transport.OnPeerTopology(topology);
  HeldData batch(&transport, node.addr);
  ASSERT_TRUE(batch.Acquire(2));
  EXPECT_EQ(CounterVal(transport.MetricsText(),
                       "dfkv_rdma_client_v2_probe_attempts_total"), 1);
  batch.ReleaseAt(0);
  transport.OnPeerTopology(topology);  // identical publication is a no-op
  EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 2u);
  EXPECT_TRUE(RdmaTransportTestPeer::HasCapabilities(&transport, node.addr));
  HeldData old_publication(&transport, node.addr);
  ASSERT_TRUE(old_publication.Acquire(1));

  topology.generation = 9;
  transport.OnPeerTopology(topology);
  EXPECT_FALSE(RdmaTransportTestPeer::HasCapabilities(&transport, node.addr));
  EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 0u);
  ASSERT_TRUE(batch.Acquire(1));
  batch.ReleaseAt(0);
  EXPECT_EQ(CounterVal(transport.MetricsText(),
                       "dfkv_rdma_client_v2_probe_attempts_total"), 2);
  old_publication.FailOne();
  EXPECT_TRUE(RdmaTransportTestPeer::HasCapabilities(&transport, node.addr))
      << "a late failure must not erase a newer publication's observation";

  const auto old_identity = topology;
  topology.peer_id = "replacement";
  transport.OnPeerTopology(topology);  // same address/generation, new identity
  ASSERT_TRUE(batch.Acquire(1));
  batch.ReleaseAt(0);
  auto late_removal = old_identity;
  late_removal.present = false;
  transport.OnPeerTopology(late_removal);
  EXPECT_TRUE(RdmaTransportTestPeer::HasCapabilities(&transport, node.addr));
  EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 1u);
  const uint64_t publication =
      RdmaTransportTestPeer::PeerPublication(transport, node.addr);
  topology.present = false;
  transport.OnPeerTopology(topology);
  topology.present = true;
  transport.OnPeerTopology(topology);
  EXPECT_GT(RdmaTransportTestPeer::PeerPublication(transport, node.addr),
            publication);
  EXPECT_FALSE(RdmaTransportTestPeer::HasCapabilities(&transport, node.addr));
  ASSERT_TRUE(batch.Acquire(1));
  batch.ReleaseAt(0);
  EXPECT_EQ(CounterVal(transport.MetricsText(),
                       "dfkv_rdma_client_v2_probe_attempts_total"), 4);

  topology.peer_addr = moved.addr;
  transport.OnPeerTopology(topology);  // identity moves without old-address hint
  EXPECT_FALSE(RdmaTransportTestPeer::HasCapabilities(&transport, node.addr));
  EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 0u);
  HeldData moved_batch(&transport, moved.addr);
  ASSERT_TRUE(moved_batch.Acquire(2));
  EXPECT_EQ(CounterVal(transport.MetricsText(),
                       "dfkv_rdma_client_v2_probe_attempts_total"), 5);
}

TEST(RdmaLoopback, CapabilityCacheUnrelatedPublicationPreservesKeepaliveReuse) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv idle_reaper("DFKV_RDMA_IDLE_MS", "0");
  RdmaNode first("caps-keepalive-first");
  RdmaNode second("caps-keepalive-second");
  ASSERT_FALSE(first.rsrv->DeviceNames().empty());
  RdmaTransport transport(kMaxMsg, first.rsrv->DeviceNames().front());
  RdmaTransportTestPeer::StopMaintenance(&transport);
  transport.OnPeerTopology(PublishedPeer(first.addr, "first"));
  auto topology = PublishedPeer(second.addr, "second");
  transport.OnPeerTopology(topology);
  HeldData first_batch(&transport, first.addr);
  HeldData second_batch(&transport, second.addr);
  ASSERT_TRUE(first_batch.Acquire(2));
  ASSERT_TRUE(second_batch.Acquire(2));
  first_batch.ReleaseAt(0);
  second_batch.ReleaseAt(0);
  ASSERT_EQ(CounterVal(transport.MetricsText(),
                       "dfkv_rdma_client_v2_probe_attempts_total"), 2);

  ++topology.generation;
  transport.OnPeerTopology(topology);
  EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, first.addr), 2u);
  EXPECT_TRUE(RdmaTransportTestPeer::HasCapabilities(&transport, first.addr));
  EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, second.addr), 0u);
  EXPECT_FALSE(RdmaTransportTestPeer::HasCapabilities(&transport, second.addr));

  RdmaTransportTestPeer::MaintainIdle(&transport, 1000, true);
  EXPECT_EQ(CounterVal(transport.MetricsText(),
                       "dfkv_rdma_client_keepalive_successes_total"), 2);
  ASSERT_TRUE(first_batch.Acquire(2));
  EXPECT_EQ(CounterVal(transport.MetricsText(),
                       "dfkv_rdma_client_v2_probe_attempts_total"), 2)
      << "an unrelated publication and successful keepalive preserve warm QPs";
  ASSERT_TRUE(second_batch.Acquire(1));
  EXPECT_EQ(CounterVal(transport.MetricsText(),
                       "dfkv_rdma_client_v2_probe_attempts_total"), 3)
      << "the replaced peer must negotiate its own fresh capabilities";
}

// End to end, with real queue pairs: the client learns from the MDS that one
// of three members changed (it reports one more rail). Only that member's
// pooled connections are torn down; the other two keep theirs and reuse them.
TEST(RdmaLoopback, OneMemberTopologyChangeKeepsOtherPeersPooledConnections) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv idle_reaper("DFKV_RDMA_IDLE_MS", "0");
  ScopedEnv pool_max("DFKV_RDMA_POOL_MAX", "8");
  RdmaNode first("topology-one-of-n-first");
  RdmaNode second("topology-one-of-n-second");
  RdmaNode changed("topology-one-of-n-changed");
  ASSERT_FALSE(first.rsrv->DeviceNames().empty());
  const std::string device = first.rsrv->DeviceNames().front();
  RdmaTransport transport(kMaxMsg, device);
  RdmaTransportTestPeer::StopMaintenance(&transport);

  const auto member = [&](const std::string& id, const std::string& addr) {
    const size_t colon = addr.rfind(':');
    MemberInfo m{id, addr.substr(0, colon),
                 static_cast<uint32_t>(std::stoul(addr.substr(colon + 1))), 1};
    m.has_health = true;
    m.health.ring_eligible = true;
    m.health.ib_devices = {{device, 4, 5, true}};
    return m;
  };
  // The client publishes members in the order the MDS lists them; the one
  // that changes is last, so once its publication has moved the others have
  // been handled too.
  std::vector<MemberInfo> members = {member("first", first.addr),
                                     member("second", second.addr),
                                     member("changed", changed.addr)};
  ScriptedTopologyMds mds(members);
  KVClient client(std::vector<std::pair<std::string, std::string>>{},
                  SelfHdr(), &transport);
  client.StartMdsDiscovery({mds.endpoint()}, "one-of-n-rdma-test", 10);
  const auto wait_for = [](const std::function<bool()>& done) {
    for (int i = 0; i < 5000 && !done(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return done();
  };
  ASSERT_TRUE(wait_for([&] {
    return RdmaTransportTestPeer::PeerPublication(transport, changed.addr) != 0;
  }));
  const uint64_t first_publication =
      RdmaTransportTestPeer::PeerPublication(transport, first.addr);
  const uint64_t second_publication =
      RdmaTransportTestPeer::PeerPublication(transport, second.addr);
  const uint64_t changed_publication =
      RdmaTransportTestPeer::PeerPublication(transport, changed.addr);
  ASSERT_NE(first_publication, 0u);
  ASSERT_NE(second_publication, 0u);

  HeldData first_batch(&transport, first.addr);
  HeldData second_batch(&transport, second.addr);
  HeldData changed_batch(&transport, changed.addr);
  ASSERT_TRUE(first_batch.Acquire(2));
  ASSERT_TRUE(second_batch.Acquire(2));
  ASSERT_TRUE(changed_batch.Acquire(2));
  first_batch.ReleaseAt(0);
  second_batch.ReleaseAt(0);
  changed_batch.ReleaseAt(0);
  ASSERT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, first.addr), 2u);
  ASSERT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, second.addr), 2u);
  ASSERT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, changed.addr), 2u);
  const long reaped = CounterVal(
      transport.MetricsText(), "dfkv_rdma_client_stale_generation_reaps_total");
  const long opened = CounterVal(
      transport.MetricsText(), "dfkv_rdma_client_conns_opened_total");

  // One more rail on the changed member, down and unknown to this client:
  // its entry differs, its usable rails do not.
  members[2].health.ib_devices.push_back({"rail-not-present-here", 1, 2, true});
  mds.SetMembers(members);
  ASSERT_TRUE(wait_for([&] {
    return RdmaTransportTestPeer::PeerPublication(transport, changed.addr) !=
           changed_publication;
  }));
  client.StopMdsDiscovery();

  EXPECT_EQ(RdmaTransportTestPeer::PeerPublication(transport, first.addr),
            first_publication);
  EXPECT_EQ(RdmaTransportTestPeer::PeerPublication(transport, second.addr),
            second_publication);
  EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, first.addr), 2u);
  EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, second.addr), 2u);
  EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, changed.addr), 0u);
  EXPECT_EQ(CounterVal(transport.MetricsText(),
                       "dfkv_rdma_client_stale_generation_reaps_total") -
                reaped, 2)
      << "connections torn down for a change of one member of three";
  ASSERT_TRUE(first_batch.Acquire(2));
  ASSERT_TRUE(second_batch.Acquire(2));
  EXPECT_EQ(CounterVal(transport.MetricsText(),
                       "dfkv_rdma_client_conns_opened_total") - opened, 0)
      << "the unchanged members are served from their warm connections";
}

TEST(RdmaLoopback, CapabilityCacheDoesNotTrustStaticOrAnonymousPeers) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv idle_reaper("DFKV_RDMA_IDLE_MS", "0");
  ScopedEnv pool_max("DFKV_RDMA_POOL_MAX", "8");
  RdmaNode node("caps-static");
  ASSERT_FALSE(node.rsrv->DeviceNames().empty());
  RdmaTransport transport(kMaxMsg, node.rsrv->DeviceNames().front());
  RdmaTransportTestPeer::StopMaintenance(&transport);
  HeldData batch(&transport, node.addr);
  ASSERT_TRUE(batch.Acquire(2));
  batch.ReleaseAt(0);
  EXPECT_FALSE(RdmaTransportTestPeer::HasCapabilities(&transport, node.addr));
  EXPECT_EQ(CounterVal(transport.MetricsText(),
                       "dfkv_rdma_client_v2_probe_attempts_total"), 2);
  ASSERT_TRUE(batch.Acquire(2));  // warm QPs still need no probe
  batch.ReleaseAt(0);
  EXPECT_EQ(CounterVal(transport.MetricsText(),
                       "dfkv_rdma_client_v2_probe_attempts_total"), 2);
  transport.OnPeerTopology(PublishedPeer(node.addr, ""));
  ASSERT_TRUE(batch.Acquire(2));  // nonzero publication alone is insufficient
  EXPECT_FALSE(RdmaTransportTestPeer::HasCapabilities(&transport, node.addr));
  EXPECT_EQ(CounterVal(transport.MetricsText(),
                       "dfkv_rdma_client_v2_probe_attempts_total"), 4);
}

TEST(RdmaLoopback, CapabilityCacheRecoveryReprobesAndFailureInvalidates) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv idle_reaper("DFKV_RDMA_IDLE_MS", "0");
  ScopedEnv pool_max("DFKV_RDMA_POOL_MAX", "8");
  RdmaNode node("caps-recovery");
  ASSERT_FALSE(node.rsrv->DeviceNames().empty());
  RdmaTransport transport(kMaxMsg, node.rsrv->DeviceNames().front());
  RdmaTransportTestPeer::StopMaintenance(&transport);
  transport.OnPeerTopology(PublishedPeer(node.addr, "recovering"));
  HeldData batch(&transport, node.addr);
  ASSERT_TRUE(batch.Acquire(1));
  batch.ReleaseAt(0);
  ASSERT_TRUE(RdmaTransportTestPeer::HasCapabilities(&transport, node.addr));
  ASSERT_TRUE(batch.Acquire(1, /*force_new=*/true));
  EXPECT_EQ(CounterVal(transport.MetricsText(),
                       "dfkv_rdma_client_v2_probe_attempts_total"), 2);
  batch.FailOne();
  EXPECT_FALSE(RdmaTransportTestPeer::HasCapabilities(&transport, node.addr));
}

TEST(RdmaLoopback, CapabilityCacheFailedRecoveryProbeIsNotCached) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv idle_reaper("DFKV_RDMA_IDLE_MS", "0");
  RdmaNode node("caps-failed-probe");
  ASSERT_FALSE(node.rsrv->DeviceNames().empty());
  RdmaTransport transport(kMaxMsg, node.rsrv->DeviceNames().front());
  RdmaTransportTestPeer::StopMaintenance(&transport);
  transport.OnPeerTopology(PublishedPeer(node.addr, "failed-probe"));
  HeldData batch(&transport, node.addr);
  ASSERT_TRUE(batch.Acquire(1));
  batch.ReleaseAt(0);
  ASSERT_TRUE(RdmaTransportTestPeer::HasCapabilities(&transport, node.addr));
  node.rsrv->Stop();
  EXPECT_FALSE(batch.Acquire(1, /*force_new=*/true));
  EXPECT_FALSE(RdmaTransportTestPeer::HasCapabilities(&transport, node.addr));
  EXPECT_EQ(CounterVal(transport.MetricsText(),
                       "dfkv_rdma_client_v2_probe_failures_total"), 1);
}

std::string PatternValue(size_t size, size_t seed);

// Retain an exact READ capability independently of the production client's
// automatic release, so removal, eviction and teardown are deterministic.
struct PinnedPullPeer {
  std::optional<rdma::RcEndpoint> endpoint{std::in_place};
  rdma::RcEndpoint& ep = *endpoint;
  void Close() { endpoint.reset(); }
  template <typename Node>
  bool Open(const Node& node, size_t depth = 1, bool legacy = false,
            uint64_t declared = kMaxMsg, uint64_t extra_flags = 0) {
    const auto& dev = node.rsrv->DeviceNames().front();
    if (!ep.Open(dev.c_str(), rdma::kV2ControlCap, depth)) return false;
    int fd = net::Dial(node.addr, 10000, 10000);
    if (fd < 0) return false;
    char frame[rdma::kDevNameBytes], mine[rdma::kQpInfoBytes],
        remote[rdma::kQpInfoBytes], ready[rdma::kV2RetirementReadinessBytes];
    const size_t readiness_bytes = legacy ? sizeof(ready) :
        rdma::kV2DynamicOnlyReadinessBytes;
    rdma::EncodeDevFrame(dev, declared |
        (legacy ? rdma::kDevFrameRequestWriterRetirement :
                  rdma::kDevFrameRequestDynamicOnly) |
        rdma::kDevFrameRequestPullRead | rdma::kDevFrameRequestDynamicPull |
        extra_flags,
        frame, rdma::kDevProtoV2);
    auto info = ep.Local();
    info.depth = depth;
    info.protocol_version = rdma::kDevProtoV2;
    rdma::SerializeQpInfo(info, mine);
    resident = {};
    legacy_token = 0;
    const bool ok = net::WriteAll(fd, frame, sizeof(frame)) &&
        net::WriteAll(fd, mine, sizeof(mine)) &&
        net::ReadAll(fd, remote, sizeof(remote)) &&
        ep.Connect(rdma::ParseQpInfo(remote)) &&
        net::ReadAll(fd, ready, readiness_bytes) &&
        (legacy ? rdma::DecodeV2Readiness(
                      ready, readiness_bytes, &resident, &legacy_token) :
                  rdma::DecodeV2DynamicOnlyReadiness(
                      ready, readiness_bytes, &resident));
    ::close(fd);
    return ok;
  }
  uint64_t legacy_token = 0;
  rdma::RecvSegmentInfo resident;  // this connection's slots, as granted
  bool Exchange(size_t request_bytes, Status* status, uint64_t* bytes) {
    if (!ep.PostRecv(0) || !ep.PostSend(0, request_bytes)) return false;
    bool sent = false, received = false;
    while (!sent || !received) {
      ibv_wc wc{};
      if (ep.WaitComp(&wc, 1, 10000) != 1 || wc.status != IBV_WC_SUCCESS)
        return false;
      if (wc.opcode == IBV_WC_RECV) {
        if (wc.byte_len < kRespPrefix ||
            !DecodeRespVersion(ep.rbuf(0), kNativeProtoRdmaV2, status, bytes) ||
            wc.byte_len != kRespPrefix + *bytes) return false;
        received = true;
      } else {
        sent = true;
      }
    }
    return true;
  }
  bool Prepare(const BlockKey& key, size_t offset, size_t length,
               rdma::DynamicPullReady* ready) {
    EncodeReqVersion(ep.sbuf(0), kNativeProtoRdmaV2, WireOp::kPullRange,
                     key, offset, length, rdma::kPullPrepareBytes);
    rdma::EncodePullPrepareControl({}, ep.sbuf(0) + kReqPrefix);
    Status status = Status::kIOError;
    uint64_t bytes = 0;
    return Exchange(kReqPrefix + rdma::kPullPrepareBytes, &status, &bytes) &&
        status == Status::kOk && bytes == rdma::kDynamicPullReadyBytes &&
        rdma::DecodeDynamicPullReady(ep.rbuf(0) + kRespPrefix, ready);
  }
  bool Read(const rdma::DynamicPullReady& ready, std::string* out) {
    out->resize(ready.data_len);
    if (out->empty()) return true;
    ibv_mr* mr = ep.RegisterTransient(out->data(), out->size(), true);
    if (!mr) return false;
    ibv_wc wc{};
    const bool ok = ep.PostRead(0, out->data(), out->size(), mr,
                                ready.address, ready.rkey) &&
        ep.WaitComp(&wc, 1, 10000) == 1 && wc.status == IBV_WC_SUCCESS;
    ep.ReleaseTransient(mr);
    return ok;
  }
  bool Release(const BlockKey& key, const rdma::DynamicPullReady& ready) {
    EncodeReqVersion(ep.sbuf(0), kNativeProtoRdmaV2, WireOp::kPullRelease,
                     key, 0, ready.slot_generation, ready.slot_index + 1);
    Status status = Status::kIOError;
    uint64_t bytes = 0;
    return Exchange(kReqPrefix, &status, &bytes) && status == Status::kOk;
  }
  // Pipelined requests (depth > 1): each request is built in its own send
  // buffer and returns its byte count.
  size_t EncodePull(size_t send_slot, const BlockKey& key, size_t offset,
                    size_t length, uint32_t pull_slot) {
    EncodeReqVersion(ep.sbuf(send_slot), kNativeProtoRdmaV2,
                     WireOp::kPullRange, key, offset, length,
                     rdma::kPullPrepareBytes);
    rdma::PullPrepareControl control;
    control.slot_index = pull_slot;
    rdma::EncodePullPrepareControl(control, ep.sbuf(send_slot) + kReqPrefix);
    return kReqPrefix + rdma::kPullPrepareBytes;
  }
  size_t EncodeRelease(size_t send_slot, const BlockKey& key,
                       const rdma::DynamicPullReady& ready) {
    EncodeReqVersion(ep.sbuf(send_slot), kNativeProtoRdmaV2,
                     WireOp::kPullRelease, key, 0, ready.slot_generation,
                     ready.slot_index + 1);
    return kReqPrefix;
  }
  // Sends the requests built in send buffers 0..n-1 back to back, in that
  // order, before waiting for any reply. Replies arrive in request order:
  // reply i lands in receive buffer i.
  bool Pipeline(const std::vector<size_t>& request_bytes,
                std::vector<Status>* statuses) {
    const size_t n = request_bytes.size();
    for (size_t i = 0; i < n; ++i)
      if (!ep.PostRecv(i)) return false;
    for (size_t i = 0; i < n; ++i)
      if (!ep.PostSend(i, request_bytes[i])) return false;
    statuses->assign(n, Status::kIOError);
    size_t sent = 0, received = 0;
    while (sent < n || received < n) {
      ibv_wc wc{};
      if (ep.WaitComp(&wc, 1, 10000) != 1 || wc.status != IBV_WC_SUCCESS)
        return false;
      if (wc.opcode != IBV_WC_RECV) {
        ++sent;
        continue;
      }
      const size_t slot = static_cast<size_t>(wc.wr_id);
      uint64_t bytes = 0;
      if (slot >= n || wc.byte_len < kRespPrefix ||
          !DecodeRespVersion(ep.rbuf(slot), kNativeProtoRdmaV2,
                             &(*statuses)[slot], &bytes) ||
          wc.byte_len != kRespPrefix + bytes)
        return false;
      ++received;
    }
    return true;
  }
};

bool PinnedExists(PinnedPullPeer& peer, const BlockKey& key) {
  EncodeReqVersion(peer.ep.sbuf(0), kNativeProtoRdmaV2,
                   WireOp::kExist, key, 0, 0, 0);
  Status status = Status::kIOError;
  uint64_t bytes = 0;
  return peer.Exchange(kReqPrefix, &status, &bytes) &&
         status == Status::kOk;
}

// The log engine's read check on the pinned PULL path (RdmaServer::Serve,
// publish_pull): a value whose bytes on disk fail their payload checksum is
// a miss for the client, is not installed in the RAM tier, and is gone from
// the disk's index (design §5.3).
TEST(RdmaLoopback, ADamagedLogValueIsAMissAndIsNotInstalled) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  // The disk read and its check on the io_uring path, where publish_pull
  // verifies (without io_uring the synchronous range handler reads and
  // checks the value itself).
  ScopedEnv uring("DFKV_SERVER_URING", "1");
  ScopedEnv engine("DFKV_STORE_ENGINE", "log");
  ScopedEnv lse("DFKV_LSE_EXPERIMENTAL", "1");
  ScopedEnv ram("DFKV_RAM_TIER", "1");
  ScopedEnv ram_bytes("DFKV_RAM_TIER_BYTES", "16777216");
  ScopedEnv ram_extent("DFKV_RAM_TIER_EXTENT_BYTES", "1048576");
  ScopedEnv ram_mode("DFKV_RAM_WRITE_MODE", "writearound");  // PUTs skip RAM
  RdmaNode node("log-damaged-pull", kMaxMsg, /*fail_pinned_registration=*/false,
                /*prepare_reads=*/true);
  ASSERT_EQ(node.srv->engine_name(), "log");
  ASSERT_TRUE(node.srv->ram_enabled());
  const BlockKey good = ToBlockKey(SelfHdr(), "log-good-pull");
  const BlockKey bad = ToBlockKey(SelfHdr(), "log-damaged-pull");
  const std::string good_value = PatternValue(8192, 43);
  const std::string bad_value = PatternValue(8192, 41);
  std::string ignored;
  for (const auto& kv : {std::make_pair(good, good_value),
                         std::make_pair(bad, bad_value)})
    ASSERT_EQ(node.srv->ProcessRequestForKey(
                  static_cast<uint8_t>(WireOp::kCache), kv.first, 0, 0,
                  kv.second.data(), kv.second.size(), &ignored),
              Status::kOk);
  // Damage one byte of the bad value's payload in its segment file.
  bool damaged = false;
  for (const auto& e : fs::recursive_directory_iterator(node.dir)) {
    if (!e.is_regular_file() || e.path().parent_path().filename() != "segments")
      continue;
    std::ifstream in(e.path(), std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
    const size_t at = bytes.find(bad_value);
    if (at == std::string::npos) continue;
    const int fd = ::open(e.path().c_str(), O_WRONLY);
    ASSERT_GE(fd, 0);
    const char flipped = static_cast<char>(bytes[at + 100] ^ 0x01);
    ASSERT_EQ(::pwrite(fd, &flipped, 1, static_cast<off_t>(at + 100)), 1);
    ASSERT_EQ(::fsync(fd), 0);
    ::close(fd);
    damaged = true;
  }
  ASSERT_TRUE(damaged) << "the value was not found in a segment file";
  PinnedPullPeer peer;
  ASSERT_TRUE(peer.Open(node));
  auto pull = [&](const BlockKey& key, Status* status, std::string* out) {
    EncodeReqVersion(peer.ep.sbuf(0), kNativeProtoRdmaV2, WireOp::kPullRange,
                     key, 0, 8192, rdma::kPullPrepareBytes);
    rdma::EncodePullPrepareControl({}, peer.ep.sbuf(0) + kReqPrefix);
    uint64_t bytes = 0;
    if (!peer.Exchange(kReqPrefix + rdma::kPullPrepareBytes, status, &bytes))
      return false;
    if (*status != Status::kOk) return true;
    rdma::DynamicPullReady ready;
    return rdma::DecodeDynamicPullReady(peer.ep.rbuf(0) + kRespPrefix,
                                        &ready) &&
           peer.Read(ready, out) && peer.Release(key, ready);
  };
  // The good value is served (and so the path is the one under test).
  Status status = Status::kIOError;
  std::string out;
  ASSERT_TRUE(pull(good, &status, &out));
  EXPECT_EQ(status, Status::kOk);
  EXPECT_EQ(out, good_value);
  // The damaged one is a miss, twice: not installed in RAM, not indexed.
  ASSERT_TRUE(pull(bad, &status, &out));
  EXPECT_EQ(status, Status::kNotFound);
  EXPECT_FALSE(KvNodeServerTestPeer::Ram(*node.srv).Contains(bad))
      << "the damaged bytes were installed in the RAM tier";
  ASSERT_TRUE(pull(bad, &status, &out));
  EXPECT_EQ(status, Status::kNotFound);
  EXPECT_EQ(node.srv->ProcessRequestForKey(static_cast<uint8_t>(WireOp::kExist),
                                           bad, 0, 0, nullptr, 0, &ignored),
            Status::kNotFound);
  peer.Close();
}

TEST(RdmaLoopback, LegacyDynamicHandshakeReadReleaseAndLiveProof) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "0");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  RdmaNode node("legacy-dynamic");
  const BlockKey key = ToBlockKey(SelfHdr(), "legacy-dynamic");
  const std::string value = PatternValue(8197, 37);
  std::string ignored;
  ASSERT_EQ(node.srv->ProcessRequestForKey(
                static_cast<uint8_t>(WireOp::kCache), key, 0, 0,
                value.data(), value.size(), &ignored), Status::kOk);
  PinnedPullPeer peer;
  ASSERT_TRUE(peer.Open(node, 1, true));
  ASSERT_NE(peer.legacy_token, 0u);
  const auto request_proof = [&] {
    const int fd = net::Dial(node.addr, 5000, 5000);
    if (fd < 0) return false;
    char request[rdma::kDevNameBytes], proof[rdma::kV2RetireProofBytes];
    rdma::EncodeDevFrame(
        rdma::kV2RetireWriterDevice, peer.legacy_token, request);
    const bool proved = net::WriteAll(fd, request, sizeof(request)) &&
        net::ReadAll(fd, proof, sizeof(proof)) &&
        rdma::ParseV2RetireProof(proof, peer.legacy_token);
    ::close(fd);
    return proved;
  };
  rdma::DynamicPullReady ready;
  const uint64_t baseline = rdma::RcEndpoint::LeaseReadMrActive();
  ASSERT_TRUE(peer.Prepare(key, 97, 4097, &ready));
  EXPECT_EQ(ready.value_len, value.size());
  EXPECT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), baseline + 1);
  ASSERT_TRUE(request_proof());
  ASSERT_TRUE(request_proof());
  // Compatibility proof must neither close the connection nor revoke its READ.
  std::string output;
  ASSERT_TRUE(peer.Read(ready, &output));
  EXPECT_EQ(output, value.substr(97, 4097));
  ASSERT_TRUE(peer.Release(key, ready));
  EXPECT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), baseline);
  EXPECT_TRUE(PinnedExists(peer, key));
  // Force an observable server-side close. Destroying a manually bootstrapped
  // RC peer alone does not notify the remote polling loop.
  EncodeReqVersion(peer.ep.sbuf(0), kNativeProtoRdmaV2, WireOp::kRange,
                   key, 0, 0, 0);
  ASSERT_TRUE(peer.ep.PostSend(0, kReqPrefix));
  ibv_wc sent{};
  ASSERT_EQ(peer.ep.WaitComp(&sent, 1, 5000), 1);
  peer.Close();
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (RdmaServerTestPeer::HasLegacyConnection(
             node.rsrv.get(), peer.legacy_token) &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  ASSERT_FALSE(RdmaServerTestPeer::HasLegacyConnection(
      node.rsrv.get(), peer.legacy_token));
  EXPECT_FALSE(request_proof());
}

TEST(RdmaLoopback, HistoricalResponderWriteRangeIsRejectedBeforeDma) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "0");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  RdmaNode node("reject-old-range");
  const BlockKey key = ToBlockKey(SelfHdr(), "reject-old-range");
  const std::string value = PatternValue(4096, 19);
  std::string ignored;
  ASSERT_EQ(node.srv->ProcessRequestForKey(
                static_cast<uint8_t>(WireOp::kCache), key, 0, 0,
                value.data(), value.size(), &ignored), Status::kOk);
  std::string output(value.size(), '\x5a');
  PinnedPullPeer peer;
  ASSERT_TRUE(peer.Open(node));
  ibv_mr* mr = peer.ep.RegisterTransient(output.data(), output.size(), true);
  ASSERT_NE(mr, nullptr);
  // Historical v2 frame: 50-byte prefix, 40-byte window, one 16-byte target.
  // Keep these bytes local: production must not retain the retired GET codec.
  std::array<char, 50 + 40 + 16> request{};
  EncodeReqVersion(request.data(), kNativeProtoRdmaV2, WireOp::kRange,
                   key, 0, value.size(), 0);
  net::PutU32(request.data() + 50, 1);
  net::PutU32(request.data() + 54, 0x3357474du);
  net::PutU32(request.data() + 66, 1);
  net::PutU64(request.data() + 82, output.size());
  net::PutU64(request.data() + 90,
              reinterpret_cast<uint64_t>(output.data()));
  net::PutU32(request.data() + 98, mr->rkey);
  net::PutU32(request.data() + 102, output.size());
  std::memcpy(peer.ep.sbuf(0), request.data(), request.size());
  ASSERT_TRUE(peer.ep.PostRecv(0));
  ASSERT_TRUE(peer.ep.PostSend(0, request.size()));
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (node.rsrv->ActiveConns() != 0 &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  EXPECT_EQ(node.rsrv->ActiveConns(), 0u);
  EXPECT_EQ(node.RangeDirectCalls(key), 0u);
  EXPECT_EQ(output, std::string(value.size(), '\x5a'));
  peer.Close();  // fence the historical destination before it is freed
}

TEST(RdmaLoopback, EmptyEofGrantStillOwnsExactMrUntilExplicitRelease) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "0");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  RdmaNode node("empty-eof-grant");
  const BlockKey key = ToBlockKey(SelfHdr(), "empty-eof-grant");
  const std::string value = PatternValue(20, 9);
  std::string ignored;
  ASSERT_EQ(node.srv->ProcessRequestForKey(
                static_cast<uint8_t>(WireOp::kCache), key, 0, 0,
                value.data(), value.size(), &ignored), Status::kOk);
  PinnedPullPeer peer;
  ASSERT_TRUE(peer.Open(node));
  EXPECT_EQ(peer.legacy_token, 0u);
  const uint64_t baseline = rdma::RcEndpoint::LeaseReadMrActive();
  const uint64_t destination_mrs = rdma::RcEndpoint::TransientUserMrActive();
  rdma::DynamicPullReady ready;
  ASSERT_TRUE(peer.Prepare(key, value.size(), 0, &ready));
  EXPECT_EQ(ready.data_len, 0u);
  EXPECT_EQ(ready.value_len, value.size());
  EXPECT_NE(ready.rkey, 0u);
  EXPECT_NE(ready.slot_generation, 0u);
  EXPECT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), baseline + 1);
  std::string output = "stale";
  ASSERT_TRUE(peer.Read(ready, &output));
  EXPECT_TRUE(output.empty());
  EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), destination_mrs);
  ASSERT_TRUE(peer.Release(key, ready));
  EXPECT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), baseline);
  EXPECT_TRUE(PinnedExists(peer, key));
}

// Pool pressure may reclaim *idle* QPs, but a client may hold a one-sided READ
// grant long after the server's reply SEND CQE. CQ age alone would revoke that
// in-flight data source. Exercise real short-lived QPs and the hard receive
// budget without fabricating a completion.
void ExercisePressurePreservesInFlightPull(bool use_uring) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
#ifndef DFKV_WITH_URING
  if (use_uring) GTEST_SKIP() << "io_uring not compiled";
#endif
  ScopedEnv uring("DFKV_SERVER_URING", use_uring ? "1" : "0");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "1");
  ScopedEnv idle("DFKV_RDMA_IDLE_MS", "5000");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  const std::string value(4096, 'p');
  const size_t budget =
      2 * rdma::V2SlotSize(kMaxMsg) + rdma::V2SlotSize(value.size());
  const std::string budget_text = std::to_string(budget);
  ScopedEnv segment("DFKV_RDMA_RECV_SEGMENT_SIZE", budget_text.c_str());
  RdmaNode node(use_uring ? "busy-pressure-uring" : "busy-pressure-sync",
                kMaxMsg, false, use_uring);
  ASSERT_EQ(node.rsrv->UseUringPath(), use_uring);
  const BlockKey key{90177, 1};
  std::string ignored;
  ASSERT_EQ(node.srv->ProcessRequestForKey(
                static_cast<uint8_t>(WireOp::kCache), key, 0, 0,
                value.data(), value.size(), &ignored), Status::kOk);

  std::mutex mu;
  std::condition_variable cv;
  bool granted = false;
  bool resume = false;
  {
    PinnedPullPeer busy;
    ASSERT_TRUE(busy.Open(node));
    bool prepared = false, read_ok = false, released = false;
    std::string bytes;
    std::thread reader([&] {
      rdma::DynamicPullReady ready;
      prepared = busy.Prepare(key, 0, value.size(), &ready);
      {
        std::unique_lock<std::mutex> lock(mu);
        granted = true;
        cv.notify_all();
        cv.wait(lock, [&] { return resume; });
      }
      if (prepared) {
        read_ok = busy.Read(ready, &bytes);
        released = busy.Release(key, ready);
      }
    });
    struct UnblockReader {
      std::mutex& mu;
      std::condition_variable& cv;
      bool& resume;
      std::thread& reader;
      ~UnblockReader() {
        {
          std::lock_guard<std::mutex> lock(mu);
          resume = true;
        }
        cv.notify_all();
        if (reader.joinable()) reader.join();
      }
    } unblock{mu, cv, resume, reader};
    {
      std::unique_lock<std::mutex> lock(mu);
      ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(5),
                              [&] { return granted; }));
    }
    ASSERT_TRUE(prepared);
    if (use_uring && CounterVal(node.rsrv->MetricsText(),
                                "dfkv_uring_init_fallbacks_total") != 0)
      GTEST_SKIP() << "io_uring unavailable at runtime";
    std::this_thread::sleep_for(std::chrono::milliseconds(2100));
    {
      PinnedPullPeer recent;
      ASSERT_TRUE(recent.Open(node));
      ASSERT_TRUE(PinnedExists(recent, key));
      size_t quiescent = 0;
      for (int i = 0; i < 1000 && quiescent == 0; ++i) {
        quiescent = RdmaServerTestPeer::RefreshQuiescentActivity(*node.rsrv, 1500);
        if (quiescent == 0)
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      ASSERT_EQ(quiescent, 1u) << "the held READ grant is not reclaimable";
      PinnedPullPeer excess;
      ASSERT_TRUE(excess.Open(node))
          << "recycle the idle QP without revoking the held READ grant";
      const std::string metrics = node.rsrv->MetricsText();
      EXPECT_GT(CounterVal(
                    metrics, "dfkv_rdma_recv_segment_allocation_failures_total"),
                0);
      EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_segment_evictions_total"), 1);
    }
    {
      std::lock_guard<std::mutex> lock(mu);
      resume = true;
    }
    cv.notify_all();
    reader.join();
    // The guard also handles early ASSERT/SKIP exits; a joined thread is inert.
    EXPECT_TRUE(prepared);
    EXPECT_TRUE(read_ok);
    EXPECT_TRUE(released);
    EXPECT_EQ(bytes, value);
    EXPECT_EQ(node.rsrv->CompletionErrors(), 0u);
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  long used = -1;
  while (std::chrono::steady_clock::now() < deadline) {
    used = CounterVal(node.rsrv->MetricsText(),
                      "dfkv_rdma_recv_segment_used_bytes");
    if (node.rsrv->ActiveConns() == 0 && used == 0) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  EXPECT_EQ(node.rsrv->ActiveConns(), 0u);
  EXPECT_EQ(used, 0);
}

TEST(RdmaLoopback, PressurePreservesInFlightPullSync) {
  ExercisePressurePreservesInFlightPull(false);
}

TEST(RdmaLoopback, PressurePreservesInFlightPullUring) {
  ExercisePressurePreservesInFlightPull(true);
}

void ExercisePressureRecyclesRecentIdle(bool use_uring) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
#ifndef DFKV_WITH_URING
  if (use_uring) GTEST_SKIP() << "io_uring not compiled";
#endif
  ScopedEnv uring("DFKV_SERVER_URING", use_uring ? "1" : "0");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "1");
  ScopedEnv idle("DFKV_RDMA_IDLE_MS", "5000");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  const std::string budget = std::to_string(2 * rdma::V2SlotSize(kMaxMsg));
  ScopedEnv segment("DFKV_RDMA_RECV_SEGMENT_SIZE", budget.c_str());
  RdmaNode node(use_uring ? "recent-idle-uring" : "recent-idle-sync",
                kMaxMsg, false, use_uring);
  const BlockKey key{90179, 1};
  std::string ignored;
  ASSERT_EQ(node.srv->ProcessRequestForKey(
                static_cast<uint8_t>(WireOp::kCache), key, 0, 0,
                "persist", 7, &ignored), Status::kOk);
  {
    PinnedPullPeer first, second;
    ASSERT_TRUE(first.Open(node));
    ASSERT_TRUE(second.Open(node));
    ASSERT_TRUE(PinnedExists(first, key));
    ASSERT_TRUE(PinnedExists(second, key));
    size_t quiescent = 0;
    const auto ready_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    do {
      quiescent = RdmaServerTestPeer::RefreshQuiescentActivity(*node.rsrv, 1500);
      if (quiescent == 2) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < ready_deadline);
    ASSERT_EQ(quiescent, 2u);
    if (use_uring && CounterVal(node.rsrv->MetricsText(),
                                "dfkv_uring_init_fallbacks_total") != 0)
      GTEST_SKIP() << "io_uring unavailable at runtime";
    PinnedPullPeer admitted;
    ASSERT_TRUE(admitted.Open(node))
        << "pressure must reclaim a quiescent QP without waiting two seconds";
    EXPECT_TRUE(PinnedExists(admitted, key));
    const std::string metrics = node.rsrv->MetricsText();
    EXPECT_GE(CounterVal(metrics, "dfkv_rdma_segment_evictions_total"), 1);
    EXPECT_GT(CounterVal(
                  metrics, "dfkv_rdma_recv_segment_allocation_failures_total"),
              0);
    EXPECT_EQ(node.rsrv->CompletionErrors(), 0u);
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  long used = -1;
  while (std::chrono::steady_clock::now() < deadline) {
    used = CounterVal(node.rsrv->MetricsText(),
                      "dfkv_rdma_recv_segment_used_bytes");
    if (node.rsrv->ActiveConns() == 0 && used == 0) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  EXPECT_EQ(node.rsrv->ActiveConns(), 0u);
  EXPECT_EQ(used, 0);
}

TEST(RdmaLoopback, PressureRecyclesRecentIdleSync) {
  ExercisePressureRecyclesRecentIdle(false);
}

TEST(RdmaLoopback, PressureRecyclesRecentIdleUring) {
  ExercisePressureRecyclesRecentIdle(true);
}

void ExercisePullStagingReclaimsIdleQp(bool use_uring) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
#ifndef DFKV_WITH_URING
  if (use_uring) GTEST_SKIP() << "io_uring not compiled";
#endif
  ScopedEnv uring("DFKV_SERVER_URING", use_uring ? "1" : "0");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "1");
  ScopedEnv idle("DFKV_RDMA_IDLE_MS", "5000");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  const std::string budget = std::to_string(2 * rdma::V2SlotSize(kMaxMsg));
  ScopedEnv segment("DFKV_RDMA_RECV_SEGMENT_SIZE", budget.c_str());
  RdmaNode node(use_uring ? "pull-pressure-uring" : "pull-pressure-sync",
                kMaxMsg, false, use_uring);
  const BlockKey key{90180, 1};
  const std::string value(4096, 'v');
  std::string ignored;
  ASSERT_EQ(node.srv->ProcessRequestForKey(
                static_cast<uint8_t>(WireOp::kCache), key, 0, 0,
                value.data(), value.size(), &ignored), Status::kOk);
  {
    PinnedPullPeer reader, idle_peer;
    ASSERT_TRUE(reader.Open(node));
    ASSERT_TRUE(idle_peer.Open(node));
    ASSERT_TRUE(PinnedExists(idle_peer, key));
    size_t quiescent = 0;
    const auto ready_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    do {
      quiescent = RdmaServerTestPeer::RefreshQuiescentActivity(*node.rsrv, 1500);
      if (quiescent == 1) break;  // reader has not completed its first request.
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < ready_deadline);
    ASSERT_EQ(quiescent, 1u);
    if (use_uring && CounterVal(node.rsrv->MetricsText(),
                                "dfkv_uring_init_fallbacks_total") != 0)
      GTEST_SKIP() << "io_uring unavailable at runtime";
    rdma::DynamicPullReady ready;
    ASSERT_TRUE(reader.Prepare(key, 0, value.size(), &ready))
        << "GET staging should reclaim an idle QP before returning kCacheFull";
    std::string bytes;
    ASSERT_TRUE(reader.Read(ready, &bytes));
    EXPECT_EQ(bytes, value);
    ASSERT_TRUE(reader.Release(key, ready));
    EXPECT_EQ(node.RangeDirectCalls(key), 1u)
        << "RAM disabled: GET must actually use the disk fallback";
    const std::string metrics = node.rsrv->MetricsText();
    EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_segment_evictions_total"), 1);
    EXPECT_GT(CounterVal(
                  metrics, "dfkv_rdma_recv_segment_allocation_failures_total"),
              0);
    EXPECT_EQ(node.rsrv->CompletionErrors(), 0u);
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  long used = -1;
  while (std::chrono::steady_clock::now() < deadline) {
    used = CounterVal(node.rsrv->MetricsText(),
                      "dfkv_rdma_recv_segment_used_bytes");
    if (node.rsrv->ActiveConns() == 0 && used == 0) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  EXPECT_EQ(node.rsrv->ActiveConns(), 0u);
  EXPECT_EQ(used, 0);
}

TEST(RdmaLoopback, PullStagingReclaimsIdleQpSync) {
  ExercisePullStagingReclaimsIdleQp(false);
}

TEST(RdmaLoopback, PullStagingReclaimsIdleQpUring) {
  ExercisePullStagingReclaimsIdleQp(true);
}

// A real client's pull window (width 2 here) sends its prepares back to back
// and releases only after every READY and READ. With pool room for one
// staging lease and an idle connection of another client, the second grant
// takes that connection's lease, as it always has, at once.
TEST(RdmaLoopback, APullWindowUnderPressureGrantsBothWithoutWaiting) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "0");
  ScopedEnv idle("DFKV_RDMA_IDLE_MS", "5000");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  const std::string value(4096, 'w');
  // The window's two receive slots, one idle depth-1 connection, and room
  // for one GET staging.
  const size_t budget =
      3 * rdma::V2SlotSize(kMaxMsg) + rdma::V2SlotSize(value.size());
  const std::string budget_text = std::to_string(budget);
  ScopedEnv segment("DFKV_RDMA_RECV_SEGMENT_SIZE", budget_text.c_str());
  RdmaNode node("pull-window-pressure");
  const BlockKey first{90221, 1}, second{90222, 1};
  std::string ignored;
  for (const BlockKey& key : {first, second})
    ASSERT_EQ(node.srv->ProcessRequestForKey(
                  static_cast<uint8_t>(WireOp::kCache), key, 0, 0,
                  value.data(), value.size(), &ignored), Status::kOk);
  PinnedPullPeer victim, window;
  ASSERT_TRUE(victim.Open(node));
  ASSERT_TRUE(PinnedExists(victim, first));
  ASSERT_TRUE(window.Open(node, 2));
  size_t quiescent = 0;
  for (int i = 0; i < 1000 && quiescent < 1; ++i) {
    quiescent = RdmaServerTestPeer::RefreshQuiescentActivity(*node.rsrv, 1500);
    if (quiescent < 1) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_GE(quiescent, 1u);
  std::vector<Status> statuses;
  const auto started = std::chrono::steady_clock::now();
  ASSERT_TRUE(window.Pipeline({window.EncodePull(0, first, 0, value.size(), 0),
                               window.EncodePull(1, second, 0, value.size(), 1)},
                              &statuses));
  const double elapsed_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
  EXPECT_EQ(statuses[0], Status::kOk);
  EXPECT_EQ(statuses[1], Status::kOk);
  EXPECT_LT(elapsed_ms, 100.0);
  EXPECT_EQ(CounterVal(node.rsrv->MetricsText(),
                       "dfkv_rdma_segment_evictions_total"), 1);
  for (size_t i = 0; i < 2; ++i) {
    rdma::DynamicPullReady ready;
    ASSERT_TRUE(rdma::DecodeDynamicPullReady(window.ep.rbuf(i) + kRespPrefix,
                                             &ready));
    std::string bytes;
    ASSERT_TRUE(window.Read(ready, &bytes));
    EXPECT_EQ(bytes, value);
    EXPECT_TRUE(window.Release(i == 0 ? first : second, ready));
  }
  EXPECT_EQ(node.rsrv->CompletionErrors(), 0u);
}

// A GET answered from the RAM arena through the prepared read (the pinned
// zero-copy path refused, here by an injected registration failure) is read
// by the client from the arena: its staging lease must go back to the pool
// with the READY, not stay leased until the RELEASE.
TEST(RdmaLoopback, AnArenaServedGetHoldsNoStagingWhileGranted) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
#ifndef DFKV_WITH_URING
  GTEST_SKIP() << "io_uring not compiled";
#endif
  ScopedEnv uring("DFKV_SERVER_URING", "1");
  ScopedEnv ram_on("DFKV_RAM_TIER", "1");
  ScopedEnv ram_bytes("DFKV_RAM_TIER_BYTES", "8388608");
  ScopedEnv extent("DFKV_RAM_TIER_EXTENT_BYTES", "1048576");
  ScopedEnv reserve("DFKV_RAM_TIER_LARGE_RESERVE_BYTES", "4194304");
  ScopedEnv shards("DFKV_RAM_TIER_SHARDS", "1");
  ScopedEnv reclaim("DFKV_RAM_RECLAIM_MS", "0");
  ScopedEnv writearound("DFKV_RAM_WRITE_MODE", "writearound");
  RdmaNode node("arena-no-staging", kMaxMsg, /*fail_pinned_registration=*/true,
                /*prepare_reads=*/true);
  if (CounterVal(node.rsrv->MetricsText(), "dfkv_uring_init_fallbacks_total") > 0)
    GTEST_SKIP() << "io_uring unavailable at runtime";
  const BlockKey key = ToBlockKey(SelfHdr(), "arena-no-staging");
  const std::string value = PatternValue(65537, 41);
  ASSERT_TRUE(KvNodeServerTestPeer::Promote(*node.srv, key, value));
  PinnedPullPeer peer;
  ASSERT_TRUE(peer.Open(node));
  rdma::DynamicPullReady ready;
  ASSERT_TRUE(peer.Prepare(key, 0, value.size(), &ready));
  EXPECT_EQ(CounterVal(node.rsrv->MetricsText(),
                       "dfkv_rdma_dynamic_get_bytes_active"), 0)
      << "the staging lease was kept while the client reads the arena";
  std::string bytes;
  ASSERT_TRUE(peer.Read(ready, &bytes));
  EXPECT_EQ(bytes, value);
  ASSERT_TRUE(peer.Release(key, ready));
  EXPECT_EQ(node.rsrv->CompletionErrors(), 0u);
}

// A GET of a key the node does not have needs no staging: it is answered
// kNotFound at once, even with the receive pool full, and evicts nobody.
void ExerciseMissNeedsNoStaging(bool use_uring) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
#ifndef DFKV_WITH_URING
  if (use_uring) GTEST_SKIP() << "io_uring not compiled";
#endif
  ScopedEnv uring("DFKV_SERVER_URING", use_uring ? "1" : "0");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "1");
  ScopedEnv idle("DFKV_RDMA_IDLE_MS", "5000");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  // Two connections and no room for staging.
  const std::string budget = std::to_string(2 * rdma::V2SlotSize(kMaxMsg));
  ScopedEnv segment("DFKV_RDMA_RECV_SEGMENT_SIZE", budget.c_str());
  RdmaNode node(use_uring ? "miss-no-staging-uring" : "miss-no-staging-sync",
                kMaxMsg, false, use_uring);
  const BlockKey present{90211, 1}, absent{90212, 1};
  std::string ignored;
  ASSERT_EQ(node.srv->ProcessRequestForKey(
                static_cast<uint8_t>(WireOp::kCache), present, 0, 0,
                "here", 4, &ignored), Status::kOk);
  PinnedPullPeer victim, peer;
  ASSERT_TRUE(victim.Open(node));
  ASSERT_TRUE(PinnedExists(victim, present));
  ASSERT_TRUE(peer.Open(node));
  ASSERT_TRUE(PinnedExists(peer, present));
  if (use_uring && CounterVal(node.rsrv->MetricsText(),
                              "dfkv_uring_init_fallbacks_total") != 0)
    GTEST_SKIP() << "io_uring unavailable at runtime";
  size_t quiescent = 0;
  for (int i = 0; i < 1000 && quiescent < 2; ++i) {
    quiescent = RdmaServerTestPeer::RefreshQuiescentActivity(*node.rsrv, 1500);
    if (quiescent < 2) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_EQ(quiescent, 2u);
  std::vector<Status> statuses;
  ASSERT_TRUE(peer.Pipeline({peer.EncodePull(0, absent, 0, 4096, 0)},
                            &statuses));
  EXPECT_EQ(statuses[0], Status::kNotFound);
  const std::string metrics = node.rsrv->MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_segment_evictions_total"), 0)
      << "a miss evicted an idle connection for staging it did not need";
  EXPECT_TRUE(PinnedExists(victim, present));
  EXPECT_EQ(node.RangeDirectCalls(absent), 0u);
  EXPECT_EQ(node.rsrv->CompletionErrors(), 0u);
}

TEST(RdmaLoopback, AMissNeedsNoStagingSync) { ExerciseMissNeedsNoStaging(false); }
TEST(RdmaLoopback, AMissNeedsNoStagingUring) { ExerciseMissNeedsNoStaging(true); }

// RDS-07: pressure reclaim evicts only a connection whose lease, once freed,
// makes room for the request. A small idle control connection between two
// busy ones cannot: evicting it made no room, cost its client the
// connection, and the request still failed after the 1 s wait.
TEST(RdmaLoopback, PressureEvictsNobodyWhoseLeaseCannotHelp) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "0");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "1");
  ScopedEnv idle("DFKV_RDMA_IDLE_MS", "5000");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  const size_t budget = 2 * rdma::V2SlotSize(kMaxMsg);
  const std::string budget_text = std::to_string(budget);
  ScopedEnv segment("DFKV_RDMA_RECV_SEGMENT_SIZE", budget_text.c_str());
  ScopedEnv chunk("DFKV_RDMA_RECV_CHUNK_BYTES", budget_text.c_str());
  RdmaNode node("pressure-useless-victim");
  // Staging for this value does not fit in what is left, and the small
  // connection's lease, even freed, is far too small for it.
  const std::string value(200 * 1024, 'v');
  const BlockKey key{90301, 1};
  std::string ignored;
  ASSERT_EQ(node.srv->ProcessRequestForKey(
                static_cast<uint8_t>(WireOp::kCache), key, 0, 0,
                value.data(), value.size(), &ignored), Status::kOk);
  PinnedPullPeer requester, small, filler;
  ASSERT_TRUE(requester.Open(node));
  ASSERT_TRUE(small.Open(node, 1, false, 4096));
  ASSERT_TRUE(filler.Open(node, 1, false, 64 * 1024));
  ASSERT_TRUE(PinnedExists(small, key));
  ASSERT_TRUE(PinnedExists(filler, key));
  size_t quiescent = 0;  // small and filler; the requester has sent nothing
  for (int i = 0; i < 1000 && quiescent < 2; ++i) {
    quiescent = RdmaServerTestPeer::RefreshQuiescentActivity(*node.rsrv, 1500);
    if (quiescent < 2) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_EQ(quiescent, 2u);
  ASSERT_TRUE(PinnedExists(filler, key));  // filler is busy again
  std::vector<Status> statuses;
  const auto started = std::chrono::steady_clock::now();
  ASSERT_TRUE(requester.Pipeline(
      {requester.EncodePull(0, key, 0, value.size(), 0)}, &statuses));
  const double elapsed_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
  EXPECT_EQ(statuses[0], Status::kCacheFull);
  const std::string metrics = node.rsrv->MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_segment_evictions_total"), 0)
      << "evicted a connection whose lease could not make room";
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_pressure_unsatisfiable_total"), 1);
  EXPECT_LT(elapsed_ms, 500.0) << "waited for an eviction that cannot help";
  EXPECT_TRUE(PinnedExists(small, key)) << "the small connection was lost";
  EXPECT_EQ(node.rsrv->CompletionErrors(), 0u);
}

// RDS-07: room that takes two idle leases. Two adjacent idle connections,
// each lease smaller than the request's staging and both together large
// enough: evicting the pair makes room. Picking one victim at a time, a
// victim had to make room alone, so the request failed with nothing
// evicted until both connections hit the idle timeout.
TEST(RdmaLoopback, PressureEvictsARunOfIdleLeasesTogether) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "0");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "1");
  ScopedEnv idle("DFKV_RDMA_IDLE_MS", "60000");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  const size_t half = rdma::V2SlotSize(128 * 1024);
  const size_t budget = rdma::V2SlotSize(kMaxMsg) + 2 * half;
  const std::string budget_text = std::to_string(budget);
  ScopedEnv segment("DFKV_RDMA_RECV_SEGMENT_SIZE", budget_text.c_str());
  ScopedEnv chunk("DFKV_RDMA_RECV_CHUNK_BYTES", budget_text.c_str());
  RdmaNode node("pressure-run");
  const std::string value(200 * 1024, 'r');
  ASSERT_GT(rdma::V2SlotSize(value.size()), half);
  ASSERT_LE(rdma::V2SlotSize(value.size()), 2 * half);
  const BlockKey key{90303, 1};
  std::string ignored;
  ASSERT_EQ(node.srv->ProcessRequestForKey(
                static_cast<uint8_t>(WireOp::kCache), key, 0, 0,
                value.data(), value.size(), &ignored), Status::kOk);
  PinnedPullPeer requester, first, second;
  ASSERT_TRUE(requester.Open(node));
  ASSERT_TRUE(first.Open(node, 1, false, 128 * 1024));
  ASSERT_TRUE(second.Open(node, 1, false, 128 * 1024));
  ASSERT_TRUE(PinnedExists(first, key));
  ASSERT_TRUE(PinnedExists(second, key));
  ASSERT_TRUE(PinnedExists(requester, key));
  size_t quiescent = 0;
  for (int i = 0; i < 1000 && quiescent < 3; ++i) {
    quiescent = RdmaServerTestPeer::RefreshQuiescentActivity(*node.rsrv, 1500);
    if (quiescent < 3) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_EQ(quiescent, 3u);
  rdma::DynamicPullReady ready;
  ASSERT_TRUE(requester.Prepare(key, 0, value.size(), &ready))
      << "the two idle leases were not given up together";
  const std::string metrics = node.rsrv->MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_segment_evictions_total"), 2);
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_pressure_unsatisfiable_total"), 0);
  std::string bytes;
  ASSERT_TRUE(requester.Read(ready, &bytes));
  EXPECT_EQ(bytes, value);
  EXPECT_TRUE(requester.Release(key, ready));
  EXPECT_EQ(node.rsrv->CompletionErrors(), 0u);
}

// RDS-07: a connection that never sent a request was never quiescent, so
// pressure reclaim could not touch it however long ago its client gave up.
// After 10 s without a first request it may be evicted.
TEST(RdmaLoopback, PressureEvictsAConnectionThatNeverSentARequest) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "0");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "1");
  ScopedEnv idle("DFKV_RDMA_IDLE_MS", "60000");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  const std::string budget = std::to_string(2 * rdma::V2SlotSize(kMaxMsg));
  ScopedEnv segment("DFKV_RDMA_RECV_SEGMENT_SIZE", budget.c_str());
  RdmaNode node("pressure-never-served");
  const std::string value(4096, 'n');
  const BlockKey key{90302, 1};
  std::string ignored;
  ASSERT_EQ(node.srv->ProcessRequestForKey(
                static_cast<uint8_t>(WireOp::kCache), key, 0, 0,
                value.data(), value.size(), &ignored), Status::kOk);
  PinnedPullPeer abandoned, requester;
  ASSERT_TRUE(abandoned.Open(node));  // and never a request
  ASSERT_TRUE(requester.Open(node));
  ASSERT_TRUE(PinnedExists(requester, key));
  ASSERT_EQ(RdmaServerTestPeer::AgeAllConnections(*node.rsrv, 11000), 2u);
  ASSERT_TRUE(PinnedExists(requester, key));
  rdma::DynamicPullReady ready;
  EXPECT_TRUE(requester.Prepare(key, 0, value.size(), &ready))
      << "the abandoned connection's lease was not reclaimed";
  EXPECT_EQ(CounterVal(node.rsrv->MetricsText(),
                       "dfkv_rdma_segment_evictions_total"), 1);
  std::string bytes;
  if (ready.data_len != 0) {
    ASSERT_TRUE(requester.Read(ready, &bytes));
    EXPECT_EQ(bytes, value);
    EXPECT_TRUE(requester.Release(key, ready));
  }
}

// RDS-16: a leased PUT's lease slot is its own index space (the immediate
// its windows carry), not the receive slot its request happened to land in.
// Two leases granted on a depth-2 connection, their windows not yet sent:
// the next request lands in the first receive slot again, which is the first
// lease's slot. That dropped the connection; it is now answered, kCacheFull
// while no lease slot is free.
TEST(RdmaLoopback, ALeaseRequestLandingOnABusyLeaseSlotIsAnswered) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "0");
  RdmaNode node("lease-slot-space");
  PinnedPullPeer peer;
  ASSERT_TRUE(peer.Open(node, 2, false, kMaxMsg,
                        rdma::kDevFrameRequestLeasedPut));
  auto lease_request = [&](size_t send_slot, uint64_t id) {
    EncodeReqVersion(peer.ep.sbuf(send_slot), kNativeProtoRdmaV2,
                     WireOp::kLeasePut, BlockKey{90401, id}, 0, 0, 8192);
    return kReqPrefix;
  };
  std::vector<Status> statuses;
  ASSERT_TRUE(peer.Pipeline({lease_request(0, 1), lease_request(1, 2)},
                            &statuses));
  ASSERT_EQ(statuses[0], Status::kOk);
  ASSERT_EQ(statuses[1], Status::kOk);
  rdma::LeasePutReady first, second;
  ASSERT_TRUE(rdma::DecodeLeasePutReady(peer.ep.rbuf(0) + kRespPrefix, &first));
  ASSERT_TRUE(rdma::DecodeLeasePutReady(peer.ep.rbuf(1) + kRespPrefix, &second));
  EXPECT_NE(first.slot, second.slot);
  ASSERT_TRUE(peer.Pipeline({lease_request(0, 3)}, &statuses))
      << "the connection was dropped";
  EXPECT_EQ(statuses[0], Status::kCacheFull);
  // And it keeps serving.
  EncodeReqVersion(peer.ep.sbuf(0), kNativeProtoRdmaV2, WireOp::kExist,
                   BlockKey{90401, 9}, 0, 0, 0);
  Status status = Status::kIOError;
  uint64_t bytes = 0;
  EXPECT_TRUE(peer.Exchange(kReqPrefix, &status, &bytes));
  EXPECT_EQ(node.rsrv->CompletionErrors(), 0u);
}

void ExercisePressureRefusesOnlyBusyQps(bool use_uring) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
#ifndef DFKV_WITH_URING
  if (use_uring) GTEST_SKIP() << "io_uring not compiled";
#endif
  ScopedEnv uring("DFKV_SERVER_URING", use_uring ? "1" : "0");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "1");
  ScopedEnv idle("DFKV_RDMA_IDLE_MS", "5000");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  const std::string value(4096, 'z');
  const std::string budget = std::to_string(
      2 * (rdma::V2SlotSize(kMaxMsg) + rdma::V2SlotSize(value.size())));
  ScopedEnv segment("DFKV_RDMA_RECV_SEGMENT_SIZE", budget.c_str());
  RdmaNode node(use_uring ? "busy-only-uring" : "busy-only-sync",
                kMaxMsg, false, use_uring);
  const BlockKey key{90181, 1};
  std::string ignored;
  ASSERT_EQ(node.srv->ProcessRequestForKey(
                static_cast<uint8_t>(WireOp::kCache), key, 0, 0,
                value.data(), value.size(), &ignored), Status::kOk);
  {
    PinnedPullPeer first, second;
    ASSERT_TRUE(first.Open(node));
    ASSERT_TRUE(second.Open(node));
    rdma::DynamicPullReady first_ready, second_ready;
    ASSERT_TRUE(first.Prepare(key, 0, value.size(), &first_ready));
    ASSERT_TRUE(second.Prepare(key, 0, value.size(), &second_ready));
    if (use_uring && CounterVal(node.rsrv->MetricsText(),
                                "dfkv_uring_init_fallbacks_total") != 0) {
      first.Release(key, first_ready);
      second.Release(key, second_ready);
      GTEST_SKIP() << "io_uring unavailable at runtime";
    }
    PinnedPullPeer excess;
    EXPECT_FALSE(excess.Open(node))
        << "no quiescent owner: do not reclaim either live READ grant";
    const std::string metrics = node.rsrv->MetricsText();
    EXPECT_GT(CounterVal(
                  metrics, "dfkv_rdma_recv_segment_allocation_failures_total"),
              0);
    EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_segment_evictions_total"), 0);
    for (auto& [peer, ready] :
         {std::pair<PinnedPullPeer*, rdma::DynamicPullReady*>{&first, &first_ready},
          {&second, &second_ready}}) {
      std::string bytes;
      ASSERT_TRUE(peer->Read(*ready, &bytes));
      EXPECT_EQ(bytes, value);
      ASSERT_TRUE(peer->Release(key, *ready));
    }
    EXPECT_EQ(node.rsrv->CompletionErrors(), 0u);
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  long used = -1;
  while (std::chrono::steady_clock::now() < deadline) {
    used = CounterVal(node.rsrv->MetricsText(),
                      "dfkv_rdma_recv_segment_used_bytes");
    if (node.rsrv->ActiveConns() == 0 && used == 0) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  EXPECT_EQ(node.rsrv->ActiveConns(), 0u);
  EXPECT_EQ(used, 0);
}

TEST(RdmaLoopback, PressureRefusesOnlyBusyQpsSync) {
  ExercisePressureRefusesOnlyBusyQps(false);
}

TEST(RdmaLoopback, PressureRefusesOnlyBusyQpsUring) {
  ExercisePressureRefusesOnlyBusyQps(true);
}

void ExerciseReservedStagingUnderConcurrentAdmissions(bool use_uring) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
#ifndef DFKV_WITH_URING
  if (use_uring) GTEST_SKIP() << "io_uring not compiled";
#endif
  ScopedEnv uring("DFKV_SERVER_URING", use_uring ? "1" : "0");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "1");
  ScopedEnv idle("DFKV_RDMA_IDLE_MS", "5000");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  const size_t slot_bytes = rdma::V2SlotSize(kMaxMsg);
  const size_t chunk_bytes = 4 * slot_bytes;
  const std::string chunk = std::to_string(chunk_bytes);
  const std::string budget = std::to_string(4 * chunk_bytes);
  ScopedEnv segment("DFKV_RDMA_RECV_SEGMENT_SIZE", budget.c_str());
  ScopedEnv recv_chunk("DFKV_RDMA_RECV_CHUNK_BYTES", chunk.c_str());
  RdmaNode node(use_uring ? "reserve-uring" : "reserve-sync",
                kMaxMsg, false, use_uring);
  const BlockKey key{90184, 1};
  const std::string value(4096, 'g');
  std::string ignored;
  ASSERT_EQ(node.srv->ProcessRequestForKey(
                static_cast<uint8_t>(WireOp::kCache), key, 0, 0,
                value.data(), value.size(), &ignored), Status::kOk);
  {
    PinnedPullPeer reader;
    ASSERT_TRUE(reader.Open(node));
    std::vector<std::unique_ptr<PinnedPullPeer>> connections;
    for (int i = 0; i < 11; ++i) {
      auto peer = std::make_unique<PinnedPullPeer>();
      ASSERT_TRUE(peer->Open(node));
      connections.push_back(std::move(peer));
    }
    if (use_uring && CounterVal(node.rsrv->MetricsText(),
                                "dfkv_uring_init_fallbacks_total") != 0)
      GTEST_SKIP() << "io_uring unavailable at runtime";
    const std::string full = node.rsrv->MetricsText();
    EXPECT_EQ(CounterVal(
                  full, "dfkv_rdma_recv_segment_connection_free_bytes"), 0);
    EXPECT_EQ(CounterVal(
                  full, "dfkv_rdma_recv_segment_staging_reserved_free_bytes"),
              static_cast<long>(chunk_bytes));

    std::mutex start_mu;
    std::condition_variable start_cv;
    bool start = false;
    std::array<bool, 4> admitted{};
    std::vector<std::thread> arrivals;
    for (size_t i = 0; i < admitted.size(); ++i) {
      arrivals.emplace_back([&, i] {
        {
          std::unique_lock<std::mutex> lock(start_mu);
          start_cv.wait(lock, [&] { return start; });
        }
        PinnedPullPeer arriving;
        admitted[i] = arriving.Open(node);
      });
    }
    struct JoinArrivals {
      std::vector<std::thread>& threads;
      ~JoinArrivals() {
        for (auto& thread : threads)
          if (thread.joinable()) thread.join();
      }
    } join{arrivals};
    {
      std::lock_guard<std::mutex> lock(start_mu);
      start = true;
    }
    start_cv.notify_all();
    rdma::DynamicPullReady ready;
    const bool prepared = reader.Prepare(key, 0, value.size(), &ready);
    std::string bytes;
    if (prepared) {
      EXPECT_TRUE(reader.Read(ready, &bytes));
      EXPECT_EQ(bytes, value);
      EXPECT_TRUE(reader.Release(key, ready));
    }
    for (auto& thread : arrivals) thread.join();
    ASSERT_TRUE(prepared)
        << "GET disk fallback must retain contiguous staging headroom";
    for (bool accepted : admitted) EXPECT_FALSE(accepted);
    EXPECT_EQ(node.RangeDirectCalls(key), 1u);
    const std::string after = node.rsrv->MetricsText();
    EXPECT_EQ(CounterVal(after, "dfkv_rdma_segment_evictions_total"), 0);
    EXPECT_GE(CounterVal(
                  after, "dfkv_rdma_recv_segment_allocation_failures_total"),
              static_cast<long>(admitted.size()));
    EXPECT_EQ(node.rsrv->CompletionErrors(), 0u);
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  long used = -1;
  while (std::chrono::steady_clock::now() < deadline) {
    used = CounterVal(node.rsrv->MetricsText(),
                      "dfkv_rdma_recv_segment_used_bytes");
    if (node.rsrv->ActiveConns() == 0 && used == 0) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  EXPECT_EQ(node.rsrv->ActiveConns(), 0u);
  EXPECT_EQ(used, 0);
  EXPECT_EQ(CounterVal(node.rsrv->MetricsText(),
                       "dfkv_rdma_recv_segment_staging_reserved_free_bytes"),
            static_cast<long>(chunk_bytes));
}

TEST(RdmaLoopback, ReservedStagingSurvivesConcurrentAdmissionsSync) {
  ExerciseReservedStagingUnderConcurrentAdmissions(false);
}

TEST(RdmaLoopback, ReservedStagingSurvivesConcurrentAdmissionsUring) {
  ExerciseReservedStagingUnderConcurrentAdmissions(true);
}

void ExercisePressureProtectsImmediatePooledPut(bool use_uring) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
#ifndef DFKV_WITH_URING
  if (use_uring) GTEST_SKIP() << "io_uring not compiled";
#endif
  ScopedEnv uring("DFKV_SERVER_URING", use_uring ? "1" : "0");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "1");
  ScopedEnv idle("DFKV_RDMA_IDLE_MS", "5000");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  ScopedEnv minimum("DFKV_RDMA_CONNECTION_MIN_BLOCK_BYTES", "262144");
  ScopedEnv maximum("DFKV_RDMA_MAX_BLOCK_BYTES", "262144");
  const std::string budget = std::to_string(2 * rdma::V2SlotSize(kMaxMsg));
  ScopedEnv segment("DFKV_RDMA_RECV_SEGMENT_SIZE", budget.c_str());
  std::mutex mu;
  std::condition_variable cv;
  bool paused = false, resume = false;
  RdmaNode node(use_uring ? "reuse-put-uring" : "reuse-put-sync",
                kMaxMsg, false, use_uring, {}, {},
                [&](size_t queued, size_t free, size_t) {
                  if (queued != 0 || free != 1) return;
                  std::unique_lock<std::mutex> lock(mu);
                  if (paused) return;
                  paused = true;
                  cv.notify_all();
                  cv.wait(lock, [&] { return resume; });
                });
  struct ResumeOnExit {
    std::mutex& mu;
    std::condition_variable& cv;
    bool& resume;
    void Release() {
      {
        std::lock_guard<std::mutex> lock(mu);
        resume = true;
      }
      cv.notify_all();
    }
    ~ResumeOnExit() { Release(); }
  } release{mu, cv, resume};
  RdmaTransport transport(kMaxMsg, node.rsrv->DeviceNames().front());
  const BlockKey first_key{90182, 1}, next_key{90183, 1};
  const std::string first_value(4096, 'a'), next_value(4096, 'b');
  ASSERT_EQ(transport.Cache(node.addr, first_key, first_value.data(),
                            first_value.size()), Status::kOk);
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(5),
                            [&] { return paused; }));
  }
  // The first reply SEND fence is complete, but Serve is paused before its
  // next RECV poll. A pooled client can already post the next one-sided PUT.
  const long opened =
      CounterVal(transport.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  PinnedPullPeer second;
  ASSERT_TRUE(second.Open(node));
  ASSERT_TRUE(PinnedExists(second, first_key));
  size_t quiescent = 0;
  const auto ready_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  do {
    quiescent = RdmaServerTestPeer::RefreshQuiescentActivity(*node.rsrv);
    if (quiescent == 2) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  } while (std::chrono::steady_clock::now() < ready_deadline);
  ASSERT_EQ(quiescent, 2u);
  if (use_uring && CounterVal(node.rsrv->MetricsText(),
                              "dfkv_uring_init_fallbacks_total") != 0)
    GTEST_SKIP() << "io_uring unavailable at runtime";
  Status next_status = Status::kIOError;
  std::thread writer([&] {
    next_status = transport.Cache(node.addr, next_key,
                                  next_value.data(), next_value.size());
  });
  struct FinishWriter {
    ResumeOnExit& release;
    std::thread& writer;
    ~FinishWriter() {
      release.Release();
      if (writer.joinable()) writer.join();
    }
  } finish{release, writer};
  const auto posted_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (CounterVal(transport.MetricsText(),
                    "dfkv_rdma_client_v2_put_writes_total") < 2 &&
         std::chrono::steady_clock::now() < posted_deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  ASSERT_GE(CounterVal(transport.MetricsText(),
                      "dfkv_rdma_client_v2_put_writes_total"), 2);
  RdmaServerTestPeer::RefreshQuiescentActivity(*node.rsrv);
  PinnedPullPeer excess;
  bool admitted = false;
  std::thread admission([&] { admitted = excess.Open(node); });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(CounterVal(node.rsrv->MetricsText(),
                       "dfkv_rdma_segment_evictions_total"), 0)
      << "a just-reused PUT QP must not be pressure-reclaimed";
  release.Release();
  admission.join();
  writer.join();
  EXPECT_FALSE(admitted) << "both QPs are inside the pressure idle grace";
  EXPECT_EQ(next_status, Status::kOk);
  EXPECT_EQ(CounterVal(transport.MetricsText(),
                       "dfkv_rdma_client_conns_opened_total"), opened);
  EXPECT_EQ(node.rsrv->CompletionErrors(), 0u);
}

TEST(RdmaLoopback, PressureProtectsImmediatePooledPutSync) {
  ExercisePressureProtectsImmediatePooledPut(false);
}

TEST(RdmaLoopback, PressureProtectsImmediatePooledPutUring) {
  ExercisePressureProtectsImmediatePooledPut(true);
}

static void ExerciseControlReceiveBeforeSendFence(bool use_uring) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
#ifndef DFKV_WITH_URING
  if (use_uring) GTEST_SKIP() << "io_uring not compiled";
#endif
  ScopedEnv uring("DFKV_SERVER_URING", use_uring ? "1" : "0");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "1");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  std::mutex mutex;
  std::condition_variable cv;
  bool paused = false, resumed = false;
  std::atomic<size_t> posts{0};
  RdmaNode node(use_uring ? "control-credit-uring" : "control-credit-sync",
                kMaxMsg, false, use_uring, [&] {
                  if (posts.fetch_add(1) != 0) return;
                  std::unique_lock<std::mutex> lock(mutex);
                  paused = true;
                  cv.notify_all();
                  cv.wait(lock, [&] { return resumed; });
                });
  ASSERT_EQ(node.rsrv->UseUringPath(), use_uring);
  const BlockKey present{9131, 0}, absent{9132, 0};
  std::string ignored;
  ASSERT_EQ(node.srv->ProcessRequestForKey(
                static_cast<uint8_t>(WireOp::kCache), present, 0, 0,
                "value", 5, &ignored), Status::kOk);
  PinnedPullPeer peer;
  ASSERT_TRUE(peer.Open(node));
  struct ResumeOnExit {
    std::mutex& mutex;
    std::condition_variable& cv;
    bool& resumed;
    void Open() {
      std::lock_guard<std::mutex> lock(mutex);
      resumed = true;
      cv.notify_all();
    }
    ~ResumeOnExit() { Open(); }
  } release{mutex, cv, resumed};
  EncodeReqVersion(peer.ep.sbuf(0), kNativeProtoRdmaV2, WireOp::kExist,
                   present, 0, 0, 0);
  Status status = Status::kIOError;
  uint64_t bytes = 0;
  ASSERT_TRUE(peer.Exchange(kReqPrefix, &status, &bytes));
  ASSERT_EQ(status, Status::kOk);
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(2), [&] { return paused; }));
  }
  if (use_uring && CounterVal(node.rsrv->MetricsText(),
                              "dfkv_uring_init_fallbacks_total") != 0)
    GTEST_SKIP() << "io_uring unavailable at runtime";
  EncodeReqVersion(peer.ep.sbuf(0), kNativeProtoRdmaV2, WireOp::kExist,
                   absent, 0, 0, 0);
  ASSERT_TRUE(peer.ep.PostRecv(0));
  ASSERT_TRUE(peer.ep.PostSend(0, kReqPrefix));
  ibv_wc wc{};
  const int ready = peer.ep.WaitComp(&wc, 1, 1000);
  const bool sent_before_fence = ready == 1 &&
      wc.status == IBV_WC_SUCCESS && wc.opcode == IBV_WC_SEND;
  EXPECT_TRUE(sent_before_fence)
      << "Control ingress must not wait for the prior reply's SEND CQE";
  release.Open();
  bool sent = sent_before_fence, received = false;
  while (!sent || !received) {
    ASSERT_EQ(peer.ep.WaitComp(&wc, 1, 10000), 1);
    ASSERT_EQ(wc.status, IBV_WC_SUCCESS);
    if (wc.opcode == IBV_WC_SEND) {
      sent = true;
    } else {
      ASSERT_EQ(wc.opcode, IBV_WC_RECV);
      ASSERT_EQ(wc.byte_len, kRespPrefix);
      ASSERT_TRUE(DecodeRespVersion(peer.ep.rbuf(0), kNativeProtoRdmaV2,
                                    &status, &bytes));
      EXPECT_EQ(status, Status::kNotFound);
      EXPECT_EQ(bytes, 0u);
      received = true;
    }
  }
  EXPECT_EQ(node.rsrv->CompletionErrors(), 0u);
}

TEST(RdmaLoopback, ControlReceiveDoesNotWaitForReplyFenceSync) {
  ExerciseControlReceiveBeforeSendFence(false);
}

TEST(RdmaLoopback, ControlReceiveDoesNotWaitForReplyFenceUring) {
  ExerciseControlReceiveBeforeSendFence(true);
}

// Only this fixture has a CQ filter. It stores real polled SEND completions
// without delivering them to Serve, then replays each exact WC once released.
// The production dispatcher and actual client QP remain in use.
struct HeldReplyCqe {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<ibv_wc> held;
  bool captured = false;
  bool capture_all = false;
  bool release_all = false;
  size_t release_count = 0;
  size_t queued = 0;
  size_t free_slots = 0;
  size_t depth = 0;
  size_t head = 0;
  bool release_on_next_recv = false;
  bool same_batch_recv_and_send = false;
  void Filter(ibv_wc* batch, int* count, size_t capacity) {
    std::lock_guard<std::mutex> lock(mu);
    depth = capacity;
    int output = 0;
    for (int i = 0; i < *count; ++i) {
      if ((capture_all || !captured) &&
          batch[i].status == IBV_WC_SUCCESS &&
          batch[i].opcode == IBV_WC_SEND) {
        held.push_back(batch[i]);
        captured = true;
      } else {
        batch[output++] = batch[i];
      }
    }
    const bool received = std::any_of(batch, batch + output,
                                      [](const ibv_wc& wc) {
                                        return wc.status == IBV_WC_SUCCESS &&
                                            (wc.opcode == IBV_WC_RECV ||
                                             wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM);
                                      });
    if (release_on_next_recv && received) {
      ++release_count;
      release_on_next_recv = false;
    }
    while (!held.empty() && static_cast<size_t>(output) < capacity &&
           (release_all || release_count != 0)) {
      batch[output++] = held.front();
      held.pop_front();
      if (received) same_batch_recv_and_send = true;
      if (release_count != 0) --release_count;
    }
    *count = output;
    cv.notify_all();
  }
  void Observe(size_t pending, size_t free, size_t next_head) {
    std::lock_guard<std::mutex> lock(mu);
    queued = pending;
    free_slots = free;
    head = next_head;
    cv.notify_all();
  }
  void Release() {
    std::lock_guard<std::mutex> lock(mu);
    release_all = true;
    cv.notify_all();
  }
  void ReleaseOne() {
    std::lock_guard<std::mutex> lock(mu);
    ++release_count;
    cv.notify_all();
  }
};

static void ExerciseRealFifoBehindReplyFence(bool use_uring) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
#ifndef DFKV_WITH_URING
  if (use_uring) GTEST_SKIP() << "io_uring not compiled";
#endif
  ScopedEnv uring("DFKV_SERVER_URING", use_uring ? "1" : "0");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "1");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  HeldReplyCqe reorder;
  RdmaNode node(use_uring ? "fifo-fence-uring" : "fifo-fence-sync",
                kMaxMsg, false, use_uring, {},
                [&](ibv_wc* batch, int* count, size_t capacity) {
                  reorder.Filter(batch, count, capacity);
                },
                [&](size_t pending, size_t free, size_t head) {
                  reorder.Observe(pending, free, head);
                });
  struct ReleaseOnExit {
    HeldReplyCqe& state;
    ~ReleaseOnExit() { state.Release(); }
  } unblock{reorder};
  ASSERT_EQ(node.rsrv->UseUringPath(), use_uring);
  const BlockKey present{9311, 0}, absent{9312, 0};
  std::string ignored;
  ASSERT_EQ(node.srv->ProcessRequestForKey(
                static_cast<uint8_t>(WireOp::kCache), present, 0, 0,
                "A-value", 7, &ignored), Status::kOk);
  PinnedPullPeer peer;
  ASSERT_TRUE(peer.Open(node));
  EncodeReqVersion(peer.ep.sbuf(0), kNativeProtoRdmaV2,
                   WireOp::kExist, present, 0, 0, 0);
  Status status = Status::kIOError;
  uint64_t bytes = 0;
  ASSERT_TRUE(peer.Exchange(kReqPrefix, &status, &bytes));
  ASSERT_EQ(status, Status::kOk);
  {
    std::unique_lock<std::mutex> lock(reorder.mu);
    ASSERT_TRUE(reorder.cv.wait_for(lock, std::chrono::seconds(5),
                                   [&] { return !reorder.held.empty(); }));
  }
  if (use_uring && CounterVal(node.rsrv->MetricsText(),
                              "dfkv_uring_init_fallbacks_total") != 0) {
    reorder.Release();
    GTEST_SKIP() << "io_uring unavailable at runtime";
  }
  EncodeReqVersion(peer.ep.sbuf(0), kNativeProtoRdmaV2,
                   WireOp::kExist, absent, 0, 0, 0);
  ASSERT_TRUE(peer.ep.PostRecv(0));
  ASSERT_TRUE(peer.ep.PostSend(0, kReqPrefix));
  {
    std::unique_lock<std::mutex> lock(reorder.mu);
    ASSERT_TRUE(reorder.cv.wait_for(
        lock, std::chrono::seconds(5),
        [&] { return reorder.queued == 1 && reorder.free_slots == 0; }));
  }
  EXPECT_EQ(node.rsrv->Completions(), 1u)
      << "B must remain a received CQE, not execute before A's SEND fence";
  ibv_wc wc{};
  ASSERT_EQ(peer.ep.WaitComp(&wc, 1, 5000), 1);
  ASSERT_EQ(wc.status, IBV_WC_SUCCESS);
  EXPECT_EQ(wc.opcode, IBV_WC_SEND);
  EXPECT_EQ(peer.ep.PollComp(&wc, 1), 0)
      << "B's reply buffer must not be reused before A's SEND fence";
  reorder.Release();
  ASSERT_EQ(peer.ep.WaitComp(&wc, 1, 5000), 1);
  ASSERT_EQ(wc.status, IBV_WC_SUCCESS);
  ASSERT_EQ(wc.opcode, IBV_WC_RECV);
  ASSERT_EQ(wc.byte_len, kRespPrefix);
  ASSERT_TRUE(DecodeRespVersion(peer.ep.rbuf(0), kNativeProtoRdmaV2,
                                &status, &bytes));
  EXPECT_EQ(status, Status::kNotFound);
  EXPECT_EQ(bytes, 0u);
  EXPECT_EQ(node.rsrv->Completions(), 2u);
  {
    std::unique_lock<std::mutex> lock(reorder.mu);
    ASSERT_TRUE(reorder.cv.wait_for(
        lock, std::chrono::seconds(5),
        [&] { return reorder.queued == 0 && reorder.free_slots == 1; }));
  }
  EXPECT_EQ(node.rsrv->CompletionErrors(), 0u);
  std::cout << "FIFO_REAL_CQE_ORDER mode=" << (use_uring ? "uring" : "sync")
            << " depth=1 A-reply B-queued A-fence B-correct" << '\n';
}

TEST(RdmaLoopback, RealFifoReceiveBeforeSendFenceSync) {
  ExerciseRealFifoBehindReplyFence(false);
}
TEST(RdmaLoopback, RealFifoReceiveBeforeSendFenceUring) {
  ExerciseRealFifoBehindReplyFence(true);
}

static void ExerciseFifoOrderAndWrap() {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "0");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  HeldReplyCqe reorder;
  reorder.capture_all = true;
  RdmaNode node("fifo-wrap-sync", kMaxMsg, false, false, {},
                [&](ibv_wc* batch, int* count, size_t capacity) {
                  reorder.Filter(batch, count, capacity);
                },
                [&](size_t pending, size_t free, size_t head) {
                  reorder.Observe(pending, free, head);
                });
  struct ReleaseOnExit {
    HeldReplyCqe& state;
    ~ReleaseOnExit() { state.Release(); }
  } unblock{reorder};
  ASSERT_FALSE(node.rsrv->UseUringPath());
  const BlockKey keys[] = {
      {9510, 0}, {9511, 0}, {9512, 0}, {9513, 0},
      {9514, 0}, {9515, 0}, {9516, 0}, {9517, 0}, {9518, 0}};
  for (const auto& [index, length] :
       {std::pair<size_t, size_t>{0, 3}, {2, 7}, {5, 11},
        {6, 13}, {8, 17}}) {
    std::string value(length, static_cast<char>('a' + index)), ignored;
    ASSERT_EQ(node.srv->ProcessRequestForKey(
                  static_cast<uint8_t>(WireOp::kCache), keys[index], 0, 0,
                  value.data(), value.size(), &ignored), Status::kOk);
  }
  PinnedPullPeer peer;
  ASSERT_TRUE(peer.Open(node, 4));
  const auto post = [&](size_t slot, size_t key_index) {
    if (!peer.ep.PostRecv(slot)) return false;
    EncodeReqVersion(peer.ep.sbuf(slot), kNativeProtoRdmaV2,
                     WireOp::kExist, keys[key_index], 0, 0, 0);
    return peer.ep.PostSend(slot, kReqPrefix);
  };
  const auto read_one = [&](size_t slot, Status expected,
                            uint64_t expected_length) {
    for (int i = 0; i < 16; ++i) {
      ibv_wc wc{};
      if (peer.ep.WaitComp(&wc, 1, 5000) != 1 ||
          wc.status != IBV_WC_SUCCESS) return false;
      if (wc.opcode == IBV_WC_SEND) continue;
      if (wc.opcode != IBV_WC_RECV || wc.wr_id != slot ||
          wc.byte_len != kRespPrefix) return false;
      Status status = Status::kIOError;
      uint64_t data_length = 0, value_length = 0;
      return DecodeRespVersion(peer.ep.rbuf(slot), kNativeProtoRdmaV2,
                               &status, &data_length, kMaxFrameLen,
                               &value_length) &&
          status == expected && data_length == 0 &&
          value_length == expected_length;
    }
    return false;
  };
  const auto wait_state = [&](auto predicate) {
    std::unique_lock<std::mutex> lock(reorder.mu);
    return reorder.cv.wait_for(lock, std::chrono::seconds(5),
                               [&] { return predicate(reorder); });
  };
  for (size_t slot = 0; slot < 4; ++slot)
    ASSERT_TRUE(post(slot, slot));
  for (size_t slot = 0; slot < 4; ++slot)
    ASSERT_TRUE(read_one(slot, slot % 2 == 0 ? Status::kOk
                                             : Status::kNotFound,
                         slot == 0 ? 3 : (slot == 2 ? 7 : 0)));
  ASSERT_TRUE(wait_state(
      [](const HeldReplyCqe& state) { return state.held.size() == 4; }));
  {
    // The serve thread writes depth on every poll, under the lock.
    std::lock_guard<std::mutex> lock(reorder.mu);
    ASSERT_EQ(reorder.depth, 4u);
  }
  ASSERT_TRUE(post(0, 4));  // B0: absent.
  ASSERT_TRUE(post(1, 5));  // B1: 11-byte present value.
  ASSERT_TRUE(wait_state([](const HeldReplyCqe& state) {
    return state.queued == 2 && state.free_slots == 0;
  }));
  {
    std::lock_guard<std::mutex> lock(reorder.mu);
    reorder.release_on_next_recv = true;
  }
  ASSERT_TRUE(post(2, 6));  // C: 13-byte value arrives behind B0/B1.
  ASSERT_TRUE(wait_state([](const HeldReplyCqe& state) {
    return state.same_batch_recv_and_send && state.queued == 2 &&
           state.head == 1 && state.free_slots == 0;
  }));
  ASSERT_TRUE(read_one(0, Status::kNotFound, 0));
  EXPECT_EQ(node.rsrv->Completions(), 5u);
  reorder.ReleaseOne();
  ASSERT_TRUE(wait_state([](const HeldReplyCqe& state) {
    return state.queued == 1 && state.head == 2;
  }));
  ASSERT_TRUE(read_one(1, Status::kOk, 11));
  reorder.ReleaseOne();
  ASSERT_TRUE(wait_state([](const HeldReplyCqe& state) {
    return state.queued == 0 && state.head == 3;
  }));
  ASSERT_TRUE(read_one(2, Status::kOk, 13));
  ASSERT_TRUE(post(0, 7));  // D0: absent.
  ASSERT_TRUE(post(1, 8));  // D1: 17-byte value.
  ASSERT_TRUE(wait_state([](const HeldReplyCqe& state) {
    return state.queued == 2 && state.head == 3;
  }));
  reorder.ReleaseOne();
  ASSERT_TRUE(wait_state([](const HeldReplyCqe& state) {
    return state.queued == 1 && state.head == 0;
  }));
  ASSERT_TRUE(read_one(0, Status::kNotFound, 0));
  reorder.ReleaseOne();
  ASSERT_TRUE(wait_state([](const HeldReplyCqe& state) {
    return state.queued == 0 && state.head == 1;
  }));
  ASSERT_TRUE(read_one(1, Status::kOk, 17));
  reorder.Release();
  ASSERT_TRUE(wait_state([](const HeldReplyCqe& state) {
    return state.queued == 0 && state.free_slots == 4;
  }));
  EXPECT_EQ(node.rsrv->Completions(), 9u);
  EXPECT_EQ(node.rsrv->CompletionErrors(), 0u);
  std::cout << "FIFO_REAL_CQE_ORDER mode=sync"
               " depth=4 old-pending/new-RECV/SEND same batch, partial drain,"
               " head=3->0->1; all nine responses matched" << '\n';
}

TEST(RdmaLoopback, RealFifoOrderWrapSync) {
  ExerciseFifoOrderAndWrap();
}


static void ExerciseQueuedReceiveConnectionClose(bool use_uring) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  // Waits for the server to notice the destroyed client QP; on rxe it did
  // not within the test's 10 s.
  if (SoftRoceOnly()) GTEST_SKIP() << kRxeNoRetryExceeded;
#ifndef DFKV_WITH_URING
  if (use_uring) GTEST_SKIP() << "io_uring not compiled";
#endif
  ScopedEnv uring("DFKV_SERVER_URING", use_uring ? "1" : "0");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "1");
  ScopedEnv idle("DFKV_RDMA_IDLE_MS", "200");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  HeldReplyCqe reorder;
  RdmaNode node(use_uring ? "fifo-close-uring" : "fifo-close-sync",
                kMaxMsg, false, use_uring, {},
                [&](ibv_wc* batch, int* count, size_t capacity) {
                  reorder.Filter(batch, count, capacity);
                },
                [&](size_t pending, size_t free, size_t head) {
                  reorder.Observe(pending, free, head);
                });
  struct ReleaseOnExit {
    HeldReplyCqe& state;
    ~ReleaseOnExit() { state.Release(); }
  } unblock{reorder};
  ASSERT_EQ(node.rsrv->UseUringPath(), use_uring);
  const BlockKey present{9731, 0}, absent{9732, 0};
  std::string ignored;
  ASSERT_EQ(node.srv->ProcessRequestForKey(
                static_cast<uint8_t>(WireOp::kCache), present, 0, 0,
                "persist", 7, &ignored), Status::kOk);
  {
    PinnedPullPeer peer;
    ASSERT_TRUE(peer.Open(node));
    EncodeReqVersion(peer.ep.sbuf(0), kNativeProtoRdmaV2,
                     WireOp::kExist, present, 0, 0, 0);
    Status status = Status::kIOError;
    uint64_t bytes = 0;
    ASSERT_TRUE(peer.Exchange(kReqPrefix, &status, &bytes));
    ASSERT_EQ(status, Status::kOk);
    {
      std::unique_lock<std::mutex> lock(reorder.mu);
      ASSERT_TRUE(reorder.cv.wait_for(
          lock, std::chrono::seconds(5),
          [&] { return !reorder.held.empty(); }));
    }
    if (use_uring && CounterVal(node.rsrv->MetricsText(),
                                "dfkv_uring_init_fallbacks_total") != 0) {
      reorder.Release();
      GTEST_SKIP() << "io_uring unavailable at runtime";
    }
    EncodeReqVersion(peer.ep.sbuf(0), kNativeProtoRdmaV2,
                     WireOp::kExist, absent, 0, 0, 0);
    ASSERT_TRUE(peer.ep.PostRecv(0));
    ASSERT_TRUE(peer.ep.PostSend(0, kReqPrefix));
    {
      std::unique_lock<std::mutex> lock(reorder.mu);
      ASSERT_TRUE(reorder.cv.wait_for(
          lock, std::chrono::seconds(5),
          [&] { return reorder.queued == 1 && reorder.free_slots == 0; }));
    }
    ibv_wc wc{};
    ASSERT_EQ(peer.ep.WaitComp(&wc, 1, 5000), 1);
    ASSERT_EQ(wc.status, IBV_WC_SUCCESS);
    ASSERT_EQ(wc.opcode, IBV_WC_SEND);
    // Destroy this real QP with B's receive still held by Serve's FIFO.
  }
  reorder.Release();
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (node.rsrv->ActiveConns() != 0 &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  ASSERT_EQ(node.rsrv->ActiveConns(), 0u);
  double receive_used = 0;
  double exact_mrs = 0;
  do {
    const std::string metrics = node.rsrv->MetricsText();
    receive_used =
        CounterVal(metrics, "dfkv_rdma_recv_segment_used_bytes");
    exact_mrs = CounterVal(metrics, "dfkv_rdma_dynamic_get_mr_active");
    if (receive_used == 0 && exact_mrs == 0) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  } while (std::chrono::steady_clock::now() < deadline);
  EXPECT_EQ(receive_used, 0);
  EXPECT_EQ(exact_mrs, 0);
  {
    PinnedPullPeer fresh;
    ASSERT_TRUE(fresh.Open(node));
    EncodeReqVersion(fresh.ep.sbuf(0), kNativeProtoRdmaV2,
                     WireOp::kExist, present, 0, 0, 0);
    Status status = Status::kIOError;
    uint64_t bytes = 0;
    ASSERT_TRUE(fresh.Exchange(kReqPrefix, &status, &bytes));
    EXPECT_EQ(status, Status::kOk);
  }
  std::cout << "FIFO_CLOSE_RECOVER mode=" << (use_uring ? "uring" : "sync")
            << " depth=1 queued=1 old-QP-retired new-QP-responded" << '\n';
}

TEST(RdmaLoopback, RealFifoQueuedConnectionCloseSync) {
  ExerciseQueuedReceiveConnectionClose(false);
}
TEST(RdmaLoopback, RealFifoQueuedConnectionCloseUring) {
  ExerciseQueuedReceiveConnectionClose(true);
}

TEST(RdmaLoopback, DynamicPullPinSurvivesRemoveAndEvictionUntilRelease) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv ram_on("DFKV_RAM_TIER", "1");
  ScopedEnv ram_bytes("DFKV_RAM_TIER_BYTES", "1048576");
  ScopedEnv extent("DFKV_RAM_TIER_EXTENT_BYTES", "1048576");
  ScopedEnv reserve("DFKV_RAM_TIER_LARGE_RESERVE_BYTES", "0");
  ScopedEnv shards("DFKV_RAM_TIER_SHARDS", "1");
  ScopedEnv reclaim("DFKV_RAM_RECLAIM_MS", "0");
  RdmaNode node("pull-pin-remove");
  auto& ram = KvNodeServerTestPeer::Ram(*node.srv);
  const BlockKey key = ToBlockKey(SelfHdr(), "pinned");
  const BlockKey replacement = ToBlockKey(SelfHdr(), "replacement");
  const std::string value = PatternValue(700000, 73);
  const std::string other = PatternValue(value.size(), 74);
  ASSERT_TRUE(KvNodeServerTestPeer::Promote(*node.srv, key, value));
  PinnedPullPeer peer;
  ASSERT_TRUE(peer.Open(node));
  const uint64_t baseline = rdma::RcEndpoint::LeaseReadMrActive();
  const size_t hits = node.srv->m_cache_hit();
  rdma::DynamicPullReady ready;
  ASSERT_TRUE(peer.Prepare(key, 97, 200013, &ready));
  EXPECT_EQ(ready.data_len, 200013u);
  EXPECT_EQ(ready.value_len, value.size());
  EXPECT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), baseline + 1);
  EXPECT_EQ(node.srv->m_cache_hit(), hits);
  EXPECT_EQ(ram.Hits(), 0u);
  EXPECT_EQ(KvNodeServerTestPeer::GetLatencyCount(*node.srv), 0u);
  // One extent, one class slot: pressure cannot evict the pinned source.
  EXPECT_FALSE(KvNodeServerTestPeer::Promote(*node.srv, replacement, other));
  EXPECT_TRUE(ram.Contains(key));
  EXPECT_TRUE(ram.Remove(key));
  EXPECT_FALSE(ram.Contains(key));
  EXPECT_GT(ram.UsedBytes(), 0u);
  EXPECT_FALSE(KvNodeServerTestPeer::Promote(*node.srv, replacement, other));
  std::string out;
  ASSERT_TRUE(peer.Read(ready, &out));
  EXPECT_EQ(out, value.substr(97, 200013));
  ASSERT_TRUE(peer.Release(key, ready));
  EXPECT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), baseline);
  EXPECT_EQ(node.srv->m_cache_hit(), hits + 1);
  EXPECT_EQ(ram.Hits(), 1u);
  EXPECT_EQ(KvNodeServerTestPeer::GetLatencyCount(*node.srv), 1u);
  EXPECT_EQ(CounterVal(node.srv->MetricsText(), "dfkv_bytes_read_total"), 200013);
  EXPECT_EQ(ram.UsedBytes(), 0u);
  ASSERT_TRUE(KvNodeServerTestPeer::Promote(*node.srv, replacement, other));
}

TEST(RdmaLoopback, DynamicPullTeardownAbortsAccountingAndReleasesPin) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv ram_on("DFKV_RAM_TIER", "1");
  ScopedEnv ram_bytes("DFKV_RAM_TIER_BYTES", "268435456");
  RdmaNode node("pull-pin-teardown");
  auto& ram = KvNodeServerTestPeer::Ram(*node.srv);
  const BlockKey key = ToBlockKey(SelfHdr(), "aborted");
  const std::string value = PatternValue(65537, 51);
  ASSERT_TRUE(KvNodeServerTestPeer::Promote(*node.srv, key, value));
  PinnedPullPeer peer;
  ASSERT_TRUE(peer.Open(node));
  const uint64_t baseline = rdma::RcEndpoint::LeaseReadMrActive();
  const size_t hits = node.srv->m_cache_hit();
  rdma::DynamicPullReady ready;
  ASSERT_TRUE(peer.Prepare(key, 0, value.size(), &ready));
  EXPECT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), baseline + 1);
  ASSERT_TRUE(ram.Remove(key));
  node.rsrv->Stop();  // joins the owner after QP destruction/MR revocation
  EXPECT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), baseline);
  EXPECT_EQ(ram.UsedBytes(), 0u);
  EXPECT_EQ(ram.Hits(), 0u);
  EXPECT_EQ(KvNodeServerTestPeer::GetLatencyCount(*node.srv), 0u);
  EXPECT_EQ(node.srv->m_cache_hit(), hits);
}

TEST(RdmaLoopback, DynamicPullExactGrantRejectsReadPastRequestedRange) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv ram_on("DFKV_RAM_TIER", "1");
  ScopedEnv ram_bytes("DFKV_RAM_TIER_BYTES", "268435456");
  RdmaNode node("pull-exact-bounds");
  auto& ram = KvNodeServerTestPeer::Ram(*node.srv);
  const BlockKey key = ToBlockKey(SelfHdr(), "bounded");
  const std::string value = PatternValue(65537, 85);
  ASSERT_TRUE(KvNodeServerTestPeer::Promote(*node.srv, key, value));
  PinnedPullPeer peer;
  ASSERT_TRUE(peer.Open(node));
  const uint64_t baseline = rdma::RcEndpoint::LeaseReadMrActive();
  rdma::DynamicPullReady ready;
  ASSERT_TRUE(peer.Prepare(key, 97, 4097, &ready));
  ASSERT_EQ(ready.data_len, 4097u);
  ASSERT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), baseline + 1);
  std::string out(ready.data_len + 1, '\0');
  ibv_mr* mr = peer.ep.RegisterTransient(out.data(), out.size(), true);
  ASSERT_NE(mr, nullptr);
  const bool posted = peer.ep.PostRead(0, out.data(), out.size(), mr,
                                       ready.address, ready.rkey);
  EXPECT_TRUE(posted);
  if (posted) {
    ibv_wc wc{};
    ASSERT_EQ(peer.ep.WaitComp(&wc, 1, 10000), 1);
    EXPECT_NE(wc.status, IBV_WC_SUCCESS)
        << "the grant exposed bytes beyond the requested range";
  }
  peer.ep.ReleaseTransient(mr);
  node.rsrv->Stop();
  EXPECT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), baseline);
  EXPECT_EQ(ram.Hits(), 0u);
  EXPECT_EQ(node.srv->m_cache_hit(), 0u);
}

TEST(RdmaLoopback, DynamicPullFallbackAccountsOneAccess) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv ram_on("DFKV_RAM_TIER", "1");
  ScopedEnv ram_bytes("DFKV_RAM_TIER_BYTES", "8388608");
  ScopedEnv extent("DFKV_RAM_TIER_EXTENT_BYTES", "1048576");
  ScopedEnv reserve("DFKV_RAM_TIER_LARGE_RESERVE_BYTES", "4194304");
  ScopedEnv shards("DFKV_RAM_TIER_SHARDS", "1");
  ScopedEnv reclaim("DFKV_RAM_RECLAIM_MS", "0");
  ScopedEnv writearound("DFKV_RAM_WRITE_MODE", "writearound");
  RdmaNode node("pull-accounting", 2 * 1024 * 1024, true);
  auto& ram = KvNodeServerTestPeer::Ram(*node.srv);
  RdmaTransport transport(2 * 1024 * 1024);
  for (size_t size : {size_t{65537}, size_t{1048577}}) {
    const BlockKey key = ToBlockKey(SelfHdr(), std::to_string(size));
    const std::string value = PatternValue(size, 29);
    ASSERT_TRUE(KvNodeServerTestPeer::Promote(*node.srv, key, value));
    PreparedRead probe;
    EXPECT_EQ(node.srv->RamPinnedHitForKey(key, 0, size, &probe),
              size < 1048576);
    probe.Abort();
    const uint64_t hits = ram.Hits();
    const size_t served = node.srv->m_cache_hit();
    const long bytes = CounterVal(node.srv->MetricsText(), "dfkv_bytes_read_total");
    std::string out(size, '\0');
    ASSERT_EQ(transport.RangeInto(node.addr, {key}, {{out.data(), size}}, nullptr),
              std::vector<Status>({Status::kOk}));
    EXPECT_EQ(out, value);
    EXPECT_EQ(ram.Hits(), hits + 1);
    EXPECT_EQ(node.srv->m_cache_hit(), served + 1);
    EXPECT_EQ(CounterVal(node.srv->MetricsText(), "dfkv_bytes_read_total"),
              bytes + static_cast<long>(size));
    EXPECT_EQ(node.RangeDirectCalls(key), 1u);
  }
  // The injected registration failure made the arena hit fall back to a
  // staged copy, and that is counted.
  EXPECT_GE(CounterVal(node.rsrv->MetricsText(),
                       "dfkv_rdma_pull_zerocopy_register_failures_total"),
            1);
  const BlockKey cold = ToBlockKey(SelfHdr(), "cold-accounting");
  const std::string cold_value = PatternValue(65537, 31);
  ASSERT_EQ(transport.Cache(node.addr, cold, cold_value.data(), cold_value.size()),
            Status::kOk);
  ASSERT_FALSE(ram.Contains(cold));
  const uint64_t cold_misses = ram.Misses();
  const size_t cold_hits = node.srv->m_cache_hit();
  const long cold_bytes =
      CounterVal(node.srv->MetricsText(), "dfkv_bytes_read_total");
  std::string cold_out(cold_value.size(), '\0');
  ASSERT_EQ(transport.RangeInto(node.addr, {cold},
                                {{cold_out.data(), cold_out.size()}}, nullptr),
            std::vector<Status>({Status::kOk}));
  EXPECT_EQ(cold_out, cold_value);
  EXPECT_EQ(ram.Misses(), cold_misses + 1);
  EXPECT_EQ(node.srv->m_cache_hit(), cold_hits + 1);
  EXPECT_EQ(CounterVal(node.srv->MetricsText(), "dfkv_bytes_read_total"),
            cold_bytes + static_cast<long>(cold_value.size()));
  const uint64_t misses = ram.Misses();
  const BlockKey absent = ToBlockKey(SelfHdr(), "absent-accounting");
  std::string out(65537, '\0');
  EXPECT_EQ(transport.RangeInto(node.addr, {absent},
                                {{out.data(), out.size()}}, nullptr),
            std::vector<Status>({Status::kNotFound}));
  EXPECT_EQ(ram.Misses(), misses + 1);
}

// B5-3 zero-copy pull GET: an arena-resident value must answer a dynamic
// pull straight from the pinned arena — no staged copy through the receive
// pool (RangeDirectForKey would be invoked) and byte-identical round-trip.
TEST(RdmaLoopback, DynamicPullServesArenaHitsZeroCopy) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv ram_on("DFKV_RAM_TIER", "1");
  ScopedEnv ram_bytes("DFKV_RAM_TIER_BYTES", "268435456");
  RdmaNode node("pull-zerocopy");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  const BlockKey key = ToBlockKey(SelfHdr(), "pullz");
  const std::string value = PatternValue(kMaxMsg, 91);
  ASSERT_TRUE(c.Put("pullz", value.data(), value.size()));
  ASSERT_GT(node.srv->RamUsedBytes(), 0)
      << "write-back PUT did not land in the RAM arena";
  const long before = CounterVal(node.rsrv->MetricsText(),
                                 "dfkv_rdma_pull_zerocopy_served_total");
  std::string out(value.size(), '\0');
  std::vector<uint64_t> value_lens;
  EXPECT_EQ(rt.RangeInto(node.addr, {key}, {{out.data(), out.size()}},
                         &value_lens),
            std::vector<Status>({Status::kOk}));
  EXPECT_EQ(value_lens, std::vector<uint64_t>({value.size()}));
  EXPECT_EQ(out, value);
  const long after = CounterVal(node.rsrv->MetricsText(),
                                "dfkv_rdma_pull_zerocopy_served_total");
  EXPECT_GT(after, before)
      << "dynamic pull staged an arena hit instead of serving it zero-copy";
  EXPECT_EQ(node.RangeDirectCalls(key), 0)
      << "zero-copy pull must not enter the staged range handler";
}

TEST(RdmaLoopback, ConfiguredPoolMaxBoundsIdleControlConnections) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv pool_max("DFKV_RDMA_POOL_MAX", "4");
  RdmaNode node("poolmax4");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  constexpr int kWorkers = 7;
  for (int i = 0; i < kWorkers; ++i) {
    const std::string value = "v" + std::to_string(i);
    ASSERT_TRUE(c.Put("pm" + std::to_string(i), value.data(), value.size()));
  }
  node.handler_delay_ms.store(250, std::memory_order_relaxed);
  auto wave = [&] {
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::vector<char> found(kWorkers, 0);
    std::vector<std::thread> workers;
    for (int i = 0; i < kWorkers; ++i) {
      workers.emplace_back([&, i] {
        ready.fetch_add(1, std::memory_order_relaxed);
        while (!start.load(std::memory_order_acquire))
          std::this_thread::yield();
        found[i] = c.Exist("pm" + std::to_string(i));
      });
    }
    while (ready.load(std::memory_order_relaxed) != kWorkers)
      std::this_thread::yield();
    start.store(true, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(
        MetricSum(
            rt.MetricsText(),
            "dfkv_rdma_client_pool_connections{lane=\"control\",state=\"active\""),
        kWorkers);
    for (auto& worker : workers) worker.join();
    for (const char present : found) EXPECT_TRUE(present);
  };
  wave();
  const long opened_after_first =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  wave();
  const long opened_after_second =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  EXPECT_GE(opened_after_second - opened_after_first, 3)
      << "pool max four should reopen the three endpoints it cannot retain";

  const std::string metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_pool_limit"), 4);
  EXPECT_LE(
      MetricSum(
          metrics,
          "dfkv_rdma_client_pool_connections{lane=\"control\",state=\"idle\""),
      4);
}

// Non-SG batch PUT: one oversized item must fail ONLY itself, not poison the
// whole same-node batch. Previously any item over the per-op payload bound
// filled every sibling's status with kInvalid — all lost their cache write.
// The PUTs of one client window share the client's deadline and are served
// one after another by one Serve thread, so the server hands them one batch
// arrival: a PUT parked for a flush counts its wait from there, not from its
// own turn. A window sent after the previous one was answered is a new batch.
static void ExerciseWindowPutsShareOneBatchArrival(bool use_uring) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
#ifndef DFKV_WITH_URING
  if (use_uring) GTEST_SKIP() << "io_uring not compiled";
#endif
  ScopedEnv uring("DFKV_SERVER_URING", use_uring ? "1" : "0");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");
  RdmaNode node(use_uring ? "batch-arrival-uring" : "batch-arrival-sync",
                kMaxMsg, false, use_uring);
  ASSERT_EQ(node.rsrv->UseUringPath(), use_uring);
  using Clock = std::chrono::steady_clock;
  std::mutex mu;
  std::vector<Clock::time_point> arrivals;
  node.on_cache_direct = [&](Clock::time_point arrival) {
    {
      std::lock_guard<std::mutex> lock(mu);
      arrivals.push_back(arrival);
    }
    // Stands in for a PUT parked for a flush.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  };
  RdmaTransport rt(kMaxMsg);
  const std::string value(4096, 'w');
  std::vector<CacheItem> window;
  for (uint64_t i = 0; i < 4; ++i)
    window.push_back({BlockKey{90300 + i, 1}, value.data(), value.size()});
  for (const Status st : rt.CacheMany(node.addr, window))
    ASSERT_EQ(st, Status::kOk);
  if (use_uring && CounterVal(node.rsrv->MetricsText(),
                              "dfkv_uring_init_fallbacks_total") != 0)
    GTEST_SKIP() << "io_uring unavailable at runtime";
  const Clock::time_point second_sent = Clock::now();
  ASSERT_EQ(rt.Cache(node.addr, BlockKey{90310, 1}, value.data(),
                     value.size()),
            Status::kOk);
  std::lock_guard<std::mutex> lock(mu);
  ASSERT_EQ(arrivals.size(), 5u);
  for (size_t i = 1; i < 4; ++i)
    EXPECT_EQ(arrivals[i], arrivals[0]) << "PUT " << i << " of the window";
  EXPECT_NE(arrivals[0], Clock::time_point{});
  EXPECT_GE(arrivals[4], second_sent) << "a later window joined the first";
}

TEST(RdmaLoopback, WindowPutsShareOneBatchArrivalSync) {
  ExerciseWindowPutsShareOneBatchArrival(false);
}

TEST(RdmaLoopback, WindowPutsShareOneBatchArrivalUring) {
  ExerciseWindowPutsShareOneBatchArrival(true);
}

// A client that dies holding a grant: the server has sent a GET's READY and
// waits for its RELEASE, or has staged a PUT lease and waits for its window.
// A dead client sends nothing, so nothing wakes the connection, and the pin
// or staging range was held until the connection's idle timeout
// (DFKV_RDMA_IDLE_MS, 10 min by default, forever with 0). The client is a
// separate process (this binary again, running only the helper below, so it
// opens its own device context), killed by the server's reply hook right
// after the grant went out. With DFKV_RDMA_GRANT_TIMEOUT_MS the connection is
// closed once the grant has waited that long, and the gauge of held grants
// goes back to 0.
TEST(RdmaLoopbackChild, HoldsAGrantUntilKilled) {
  const char* mode = std::getenv("DFKV_TEST_GRANT_CHILD");
  const char* addr = std::getenv("DFKV_TEST_GRANT_ADDR");
  const char* go = std::getenv("DFKV_TEST_GRANT_GO_FD");
  if (mode == nullptr || addr == nullptr || go == nullptr)
    GTEST_SKIP() << "helper process of GrantOfAKilledClient...";
  char byte;
  if (::read(std::atoi(go), &byte, 1) != 1) ::_exit(3);
  RdmaTransport rt(1u << 20);
  KVClient c({{"n", addr}}, SelfHdr(), &rt);
  const std::string big(2 * 65536, 'L');
  const std::string small(4096, 'g');
  if (std::string(mode) == "lease") {
    c.Put("grant-big", big.data(), big.size());
  } else {
    c.Put("grant-small", small.data(), small.size());
    std::string out(small.size(), '\0');
    c.Get("grant-small", out.data(), out.size());
  }
  for (;;) ::pause();  // the server kills us
}

// Starts this binary again, running only RdmaLoopbackChild.HoldsAGrantUntil-
// Killed against `addr` (a fresh process: it opens its own device context).
// It waits for StartGrantHolder before it connects.
std::map<pid_t, int> g_grant_holder_go;
pid_t SpawnGrantHolder(const std::string& addr, bool lease) {
  int go[2];
  if (::pipe(go) != 0) return -1;
  // Everything the child needs is built before fork: it only execs.
  std::vector<std::string> env_text;
  for (char** e = environ; *e != nullptr; ++e) env_text.emplace_back(*e);
  env_text.push_back(std::string("DFKV_TEST_GRANT_CHILD=") +
                     (lease ? "lease" : "pull"));
  env_text.push_back("DFKV_TEST_GRANT_ADDR=" + addr);
  env_text.push_back("DFKV_TEST_GRANT_GO_FD=" + std::to_string(go[0]));
  std::vector<char*> envp;
  for (auto& e : env_text) envp.push_back(e.data());
  envp.push_back(nullptr);
  std::string self = "/proc/self/exe";
  std::string filter = "--gtest_filter=RdmaLoopbackChild.HoldsAGrantUntilKilled";
  char* argv[] = {self.data(), filter.data(), nullptr};
  const pid_t child = ::fork();
  if (child == 0) {
    ::close(go[1]);
    ::execve("/proc/self/exe", argv, envp.data());
    ::_exit(4);
  }
  ::close(go[0]);
  if (child < 0) {
    ::close(go[1]);
    return -1;
  }
  g_grant_holder_go[child] = go[1];
  return child;
}
void StartGrantHolder(pid_t child) {
  const int fd = g_grant_holder_go[child];
  g_grant_holder_go.erase(child);
  if (::write(fd, "x", 1) != 1) ADD_FAILURE() << "cannot start the client";
  ::close(fd);
}

TEST(RdmaLoopback, GrantOfAKilledClientIsReleasedAfterTheGrantTimeout) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv inline_put("DFKV_RDMA_INLINE_PUT_MAX_BYTES", "65536");
  ScopedEnv idle("DFKV_RDMA_IDLE_MS", "6000");
  ScopedEnv grant("DFKV_RDMA_GRANT_TIMEOUT_MS", "500");
  for (const bool lease : {true, false})
  for (int attempt = 0;; ++attempt) {
    SCOPED_TRACE(lease ? "PUT lease" : "GET grant");
    // A GET grant is its exact READ registration: its staging lease (if it
    // has one) may already be back in the pool, for an arena-served GET.
    const char* gauge =
        lease ? "dfkv_rdma_leaseput_active" : "dfkv_rdma_dynamic_get_mr_active";
    std::atomic<pid_t> victim{0};
    std::atomic<bool> killed{false};
    std::atomic<int64_t> killed_at_us{0};
    auto now_us = [] {
      return std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::steady_clock::now().time_since_epoch())
          .count();
    };
    RdmaNode* node_ptr = nullptr;
    RdmaNode node(lease ? "grant-lease" : "grant-pull", kMaxMsg, false, false,
                  [&] {
                    const pid_t pid = victim.load();
                    if (pid <= 0 || killed.load() || node_ptr == nullptr)
                      return;
                    const RdmaServer& server = *node_ptr->rsrv;
                    if ((lease ? RdmaServerTestPeer::LeasePutActive(server)
                               : rdma::RcEndpoint::LeaseReadMrActive()) == 0)
                      return;
                    ::kill(pid, SIGKILL);
                    killed_at_us.store(now_us());
                    killed.store(true);
                  });
    node_ptr = &node;
    const pid_t child = SpawnGrantHolder(node.addr, lease);
    ASSERT_GT(child, 0);
    victim.store(child);
    StartGrantHolder(child);
    const auto give_up =
        std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (!killed.load() && std::chrono::steady_clock::now() < give_up)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    if (!killed.load()) ::kill(child, SIGKILL);
    int status = 0;
    ::waitpid(child, &status, 0);
    ASSERT_TRUE(killed.load()) << "the grant was never observed";
    // How long the server holds the dead client's grant.
    while (CounterVal(node.rsrv->MetricsText(), gauge) > 0 &&
           std::chrono::steady_clock::now() < give_up)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    const int64_t held_ms = (now_us() - killed_at_us.load()) / 1000;
    std::printf("GRANT_HELD %s %lld ms after the client died\n",
                lease ? "lease" : "pull", static_cast<long long>(held_ms));
    // The kill raced the client's own completion: on a fast link (rxe over
    // lo) the client can finish its PUT between the server seeing the grant
    // and the SIGKILL landing, and the grant is then released as used.
    // Nothing was left to expire; try again.
    if (attempt < 4 && held_ms < 100 &&
        CounterVal(node.rsrv->MetricsText(),
                   "dfkv_rdma_grant_expiries_total") == 0) {
      std::printf("GRANT_RACED %s: the client finished first; again\n",
                  lease ? "lease" : "pull");
      continue;
    }
    EXPECT_EQ(CounterVal(node.rsrv->MetricsText(), gauge), 0);
    EXPECT_LT(held_ms, 2500) << "held until the idle timeout, not the grant timeout";
    EXPECT_EQ(CounterVal(node.rsrv->MetricsText(),
                         "dfkv_rdma_grant_expiries_total"),
              1);
    break;
  }
}

// Stop() lets a connection that holds a grant finish, for at most
// DFKV_RDMA_STOP_DRAIN_MS, before it closes it. Here the client is dead, so
// the grant is never used: Stop takes the drain bound and no longer.
TEST(RdmaLoopback, StopDrainOfAnUnreleasedGrantIsBounded) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv inline_put("DFKV_RDMA_INLINE_PUT_MAX_BYTES", "65536");
  ScopedEnv idle("DFKV_RDMA_IDLE_MS", "60000");
  ScopedEnv grant("DFKV_RDMA_GRANT_TIMEOUT_MS", "0");
  ScopedEnv drain("DFKV_RDMA_STOP_DRAIN_MS", "800");
  for (int attempt = 0;; ++attempt) {
  std::atomic<pid_t> victim{0};
  std::atomic<bool> killed{false};
  RdmaNode* node_ptr = nullptr;
  RdmaNode node("stop-drain", kMaxMsg, false, false, [&] {
    const pid_t pid = victim.load();
    if (pid <= 0 || killed.load() || node_ptr == nullptr) return;
    if (RdmaServerTestPeer::LeasePutActive(*node_ptr->rsrv) == 0) return;
    ::kill(pid, SIGKILL);
    killed.store(true);
  });
  node_ptr = &node;
  const pid_t child = SpawnGrantHolder(node.addr, /*lease=*/true);
  ASSERT_GT(child, 0);
  victim.store(child);
  StartGrantHolder(child);
  const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(20);
  while (!killed.load() && std::chrono::steady_clock::now() < give_up)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  if (!killed.load()) ::kill(child, SIGKILL);
  int status = 0;
  ::waitpid(child, &status, 0);
  ASSERT_TRUE(killed.load());
  // The kill raced the client's own completion (see GrantOfAKilledClient...):
  // no grant is left unreleased, so there is nothing to drain; try again.
  if (attempt < 4 && RdmaServerTestPeer::LeasePutActive(*node.rsrv) == 0) {
    std::printf("GRANT_RACED stop-drain: the client finished first; again\n");
    continue;
  }
  const auto started = std::chrono::steady_clock::now();
  node.rsrv->Stop();
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - started);
  std::printf("STOP_WITH_UNRELEASED_GRANT %lld ms\n",
              static_cast<long long>(elapsed.count()));
  EXPECT_GE(elapsed.count(), 700);
  EXPECT_LT(elapsed.count(), 2000);
  EXPECT_EQ(CounterVal(node.rsrv->MetricsText(),
                       "dfkv_rdma_stop_drains_cut_total"),
            1);
  EXPECT_EQ(CounterVal(node.rsrv->MetricsText(), "dfkv_rdma_leaseput_active"), 0);
  break;
  }
}

// A GET whose READY is out when the server starts stopping completes: the
// connection serves its RELEASE during the drain, then closes. New grants are
// refused meanwhile, so the drain ends as soon as the ones in flight do.
TEST(RdmaLoopback, AGrantInFlightCompletesDuringTheStopDrain) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv drain("DFKV_RDMA_STOP_DRAIN_MS", "5000");
  std::atomic<bool> started{false};
  std::thread stopper;
  std::chrono::steady_clock::time_point stop_started{};
  std::atomic<int64_t> stop_ms{-1};
  RdmaNode* node_ptr = nullptr;
  std::atomic<bool> armed{false};
  RdmaNode node("drain-grant", kMaxMsg, false, false, [&] {
    if (!armed.load() || started.load() || node_ptr == nullptr) return;
    if (RdmaServerTestPeer::DynamicGetActive(*node_ptr->rsrv) == 0) return;
    started.store(true);
    // The READY of the GET is out: stop the server now, from elsewhere.
    stopper = std::thread([&] {
      stop_started = std::chrono::steady_clock::now();
      node_ptr->rsrv->Stop();
      stop_ms.store(std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - stop_started)
                        .count());
    });
    // Give Stop() time to begin its drain before the client reads.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  });
  node_ptr = &node;
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  const std::string value(8192, 'd');
  ASSERT_TRUE(c.Put("drain-key", value.data(), value.size()));
  armed.store(true);
  std::string out(value.size(), '\0');
  EXPECT_TRUE(c.Get("drain-key", out.data(), out.size()));
  EXPECT_EQ(out, value);
  ASSERT_TRUE(started.load());
  stopper.join();
  std::printf("STOP_WITH_GRANT_IN_FLIGHT %lld ms\n",
              static_cast<long long>(stop_ms.load()));
  EXPECT_LT(stop_ms.load(), 5000) << "the drain ran out";
  EXPECT_EQ(CounterVal(node.rsrv->MetricsText(),
                       "dfkv_rdma_stop_drains_cut_total"),
            0);
}

// A multi-window PUT whose client stops between windows holds its partial
// state until the grant timeout, not the idle timeout.
TEST(RdmaLoopback, AMultiWindowPutStalledBetweenWindowsExpires) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv idle_reclaim("DFKV_RDMA_IDLE_MS", "60000");
  ScopedEnv grant("DFKV_RDMA_GRANT_TIMEOUT_MS", "500");
  RdmaNode node("grant-multiwr");
  RdmaTransport rt(kMaxMsg);
  const size_t max_sg = rt.MaxSgPayloadSegs();
  ASSERT_GE(max_sg, 2u);
  std::vector<std::string> src(max_sg + 7);
  std::vector<const void*> ptrs;
  std::vector<size_t> sizes;
  for (size_t i = 0; i < src.size(); ++i) {
    src[i].assign(97 + i % 11, static_cast<char>('a' + i % 26));
    ptrs.push_back(src[i].data());
    sizes.push_back(src[i].size());
  }
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  {
    ScopedEnv abort_window("DFKV_RDMA_TEST_ABORT_MULTIWR_WINDOW", "2");
    const auto failed = c.BatchPutSg({{"stalled-multiwr", ptrs, sizes}});
    ASSERT_EQ(failed.size(), 1u);
    EXPECT_FALSE(failed[0]);
  }
  long expiries = 0;
  for (int i = 0; i < 300 && expiries == 0; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    expiries = CounterVal(node.rsrv->MetricsText(),
                          "dfkv_rdma_grant_expiries_total");
  }
  EXPECT_EQ(expiries, 1) << "the stalled PUT waited for the idle timeout";
}

TEST(RdmaLoopback, BatchPutOversizedFailsOnlyOffender) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("bpo");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  std::string small(4096, 's');
  std::string big(kMaxMsg + 4096, 'b');  // exceeds the per-op payload bound
  std::vector<KvPutItem> items = {
      {"bpo_a", small.data(), small.size()},
      {"bpo_big", big.data(), big.size()},
      {"bpo_c", small.data(), small.size()},
  };
  auto pr = c.BatchPut(items);
  ASSERT_EQ(pr.size(), 3u);
  EXPECT_TRUE(pr[0]) << "sibling poisoned by the oversized item";
  EXPECT_FALSE(pr[1]);
  EXPECT_TRUE(pr[2]) << "sibling poisoned by the oversized item";
  EXPECT_TRUE(c.Exist("bpo_a"));
  EXPECT_FALSE(c.Exist("bpo_big"));
  EXPECT_TRUE(c.Exist("bpo_c"));
}

// A PUT larger than the server's payload ceiling (--max-msg) but within the
// client's own bound (DFKV_RDMA_MAX_BLOCK_BYTES) goes the leased path. The
// server refused its lease request by dropping the connection without an
// answer: the client waited out its completion deadline (5 s), counted a
// timeout and cooled a healthy node. It is answered kInvalid now, at once,
// and the node goes on serving.
TEST(RdmaLoopback, LeasedPutAboveServerCeilingIsRefusedNotTimedOut) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv inline_put("DFKV_RDMA_INLINE_PUT_MAX_BYTES", "65536");
  ScopedEnv block("DFKV_RDMA_MAX_BLOCK_BYTES", "1048576");
  RdmaNode node("lease-oversize");  // server ceiling kMaxMsg (256 KiB)
  RdmaTransport rt(1u << 20);
  const std::string big(2 * kMaxMsg, 'B');
  const std::string fits(kMaxMsg / 2, 'f');  // leased too, under the ceiling
  for (const bool batch : {false, true}) {
    SCOPED_TRACE(batch ? "batch" : "single");
    const BlockKey big_key =
        ToBlockKey(SelfHdr(), std::string("oversize-big-") + (batch ? "b" : "s"));
    const BlockKey fits_key =
        ToBlockKey(SelfHdr(), std::string("oversize-fits-") + (batch ? "b" : "s"));
    const auto start = std::chrono::steady_clock::now();
    std::vector<Status> statuses;
    if (batch)
      statuses = rt.CacheMany(node.addr, {{big_key, big.data(), big.size()},
                                          {fits_key, fits.data(), fits.size()}});
    else
      statuses = {rt.Cache(node.addr, big_key, big.data(), big.size())};
    const auto elapsed = std::chrono::steady_clock::now() - start;
    EXPECT_EQ(statuses[0], Status::kInvalid);
    // Per object: the item that fits, in the same batch, is stored.
    if (batch) EXPECT_EQ(statuses[1], Status::kOk);
    EXPECT_LT(elapsed, std::chrono::seconds(1)) << "waited for a deadline";
    // The node was not cooled and the connection still serves.
    EXPECT_EQ(rt.Cache(node.addr, fits_key, fits.data(), fits.size()),
              Status::kOk);
    const std::string client = rt.MetricsText();
    EXPECT_EQ(CounterVal(client, "dfkv_rdma_client_completion_timeouts_total"),
              0)
        << client;
  }
  EXPECT_EQ(CounterVal(node.rsrv->MetricsText(),
                       "dfkv_rdma_leaseput_oversize_rejects_total"),
            2);
}

// Same-host GET rendezvous (phase 5): with DFKV_CLIENT_NODE_DEDUP=1, a second
// client reading the SAME keys must be served from the shm rendezvous, not the
// server — server-side completions stay ~flat while the data stays correct.
TEST(RdmaLoopback, NodeDedupCollapsesSameHostGets) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const std::string nm = NodeDedup::EnvSegmentName(
      ToBlockKey(SelfHdr(), "").digest_hi);
  ::shm_unlink(nm.c_str());
  ::setenv("DFKV_CLIENT_NODE_DEDUP", "1", 1);
  {
    RdmaNode node("ndd");
    RdmaTransport rt1(kMaxMsg), rt2(kMaxMsg);
    KVClient c1({{"n", node.addr}}, SelfHdr(), &rt1);
    KVClient c2({{"n", node.addr}}, SelfHdr(), &rt2);

    const int N = 32;
    std::vector<std::string> vals(N);
    for (int i = 0; i < N; ++i) {
      vals[i].assign(8192, '\0');
      for (size_t b = 0; b < vals[i].size(); ++b)
        vals[i][b] = static_cast<char>((i * 131 + b * 7) & 0xFF);
      ASSERT_TRUE(c1.Put("ndd" + std::to_string(i), vals[i].data(), vals[i].size())) << i;
    }

    auto get_all = [&](KVClient& c, std::vector<std::string>* out) {
      std::vector<KvGetItem> items(N);
      out->assign(N, std::string(8192, '\0'));
      for (int i = 0; i < N; ++i)
        items[i] = {"ndd" + std::to_string(i), &(*out)[i][0], 8192};
      return c.BatchGet(items);
    };

    std::vector<std::string> o1, o2;
    auto r1 = get_all(c1, &o1);  // first reader: remote fetch + publish
    for (int i = 0; i < N; ++i) { ASSERT_TRUE(r1[i]) << i; EXPECT_EQ(o1[i], vals[i]); }
    const long mid = CounterVal(node.rsrv->MetricsText(), "dfkv_rdma_completions_total");
    auto r2 = get_all(c2, &o2);  // peer: rendezvous hits, no server traffic
    for (int i = 0; i < N; ++i) { ASSERT_TRUE(r2[i]) << i; EXPECT_EQ(o2[i], vals[i]); }
    const long after = CounterVal(node.rsrv->MetricsText(), "dfkv_rdma_completions_total");
    EXPECT_LE(after - mid, N / 4)
        << "peer reads reached the server (" << (after - mid)
        << " completions); rendezvous not deduplicating";
  }
  ::unsetenv("DFKV_CLIENT_NODE_DEDUP");
  ::shm_unlink(nm.c_str());
}

TEST(RdmaLoopback, PutGetExistMissOverRdma) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device (load rdma_rxe for Soft-RoCE)";
  RdmaNode node("pgem");
  RdmaTransport rt(kMaxMsg);  // first device (rxe0 in CI)
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  std::string v(4096, '\0');
  for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<char>((i * 31 + 7) & 0xFF);
  ASSERT_TRUE(c.Put("k1", v.data(), v.size()));
  EXPECT_TRUE(c.Exist("k1"));
  std::string out(v.size(), '\0');
  ASSERT_TRUE(c.Get("k1", &out[0], out.size()));
  EXPECT_EQ(out, v);

  // miss: absent key
  std::string m(v.size(), '\0');
  EXPECT_FALSE(c.Get("absent", &m[0], m.size()));
  EXPECT_FALSE(c.Exist("absent"));
}

// Observability counters: the server tallies request completions and the client
// transport tallies connections opened (+ per-rail) and MR regions.
TEST(RdmaLoopback, MetricsCountersTrackOps) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("mco");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  std::vector<char> pool(64 * 1024);
  ASSERT_TRUE(c.RegisterMemory(pool.data(), pool.size()));
  std::string v(2048, 'z');
  ASSERT_TRUE(c.Put("m1", v.data(), v.size()));
  std::string out(v.size(), '\0');
  ASSERT_TRUE(c.Get("m1", &out[0], out.size()));

  // server: at least the PUT + GET requests were completed
  EXPECT_GE(node.rsrv->Completions(), 2u);
  std::string srv_text = node.rsrv->MetricsText();
  EXPECT_NE(srv_text.find("dfkv_rdma_completions_total"), std::string::npos) << srv_text;
  EXPECT_GE(CounterVal(srv_text, "dfkv_rdma_v2_conns_opened_total"), 1);
  EXPECT_GE(CounterVal(srv_text, "dfkv_rdma_v2_put_writes_total"), 1);
  EXPECT_GT(CounterVal(srv_text, "dfkv_rdma_recv_segment_bytes"), 0);
  EXPECT_GT(CounterVal(srv_text, "dfkv_rdma_recv_segment_used_bytes"), 0);
  EXPECT_GT(CounterVal(
                srv_text,
                "dfkv_rdma_recv_segment_largest_free_range_bytes"),
            0);
  EXPECT_GE(CounterVal(srv_text, "dfkv_rdma_pull_connections"), 1);
  EXPECT_GT(CounterVal(
                srv_text,
                "dfkv_rdma_connection_bytes{class=\"data\"}"),
            0);
  EXPECT_NE(srv_text.find("dfkv_rdma_rail_active_conns{dev=\""),
            std::string::npos) << srv_text;
  EXPECT_GE(MetricSum(srv_text, "dfkv_rdma_rail_completions_total{"), 2);
  EXPECT_GE(MetricSum(srv_text, "dfkv_rdma_rail_put_writes_total{"), 1);
  EXPECT_GE(MetricSum(srv_text, "dfkv_rdma_rail_put_bytes_total{"), 2048);

  // client transport: a connection was opened and the MR region declared
  std::string cli_text = rt.MetricsText();
  EXPECT_NE(cli_text.find("dfkv_rdma_client_conns_opened_total"), std::string::npos) << cli_text;
  EXPECT_NE(cli_text.find("dfkv_rdma_client_rail_conns_total{dev="), std::string::npos) << cli_text;
  EXPECT_NE(cli_text.find("dfkv_rdma_client_mr_regions 1"), std::string::npos) << cli_text;
  EXPECT_GE(CounterVal(cli_text, "dfkv_rdma_client_v2_put_writes_total"), 1);
  EXPECT_GE(CounterVal(cli_text, "dfkv_rdma_client_pull_reads_total"), 1);
  EXPECT_GE(CounterVal(cli_text,
                       "dfkv_rdma_client_pull_read_bytes_total"), 2048);
  EXPECT_GE(MetricSum(cli_text, "dfkv_rdma_client_rail_put_ops_total{"), 1);
  EXPECT_GE(MetricSum(cli_text, "dfkv_rdma_client_rail_put_bytes_total{"),
            2048);
  EXPECT_GE(MetricSum(cli_text, "dfkv_rdma_client_rail_get_ops_total{"), 1);
  EXPECT_GE(MetricSum(cli_text, "dfkv_rdma_client_rail_get_bytes_total{"),
            2048);
  EXPECT_GE(CounterVal(cli_text,
                       "dfkv_rdma_client_pool_mr_registrations_total"), 1);
  EXPECT_EQ(CounterVal(cli_text, "dfkv_rdma_client_pool_limit"), 8);
  EXPECT_GE(MetricSum(
                cli_text,
                "dfkv_rdma_client_pool_connections{lane=\"payload\",state=\"idle\""),
            1);
  EXPECT_GE(MetricSum(cli_text, "dfkv_rdma_client_peer_connections{"), 1);

  EXPECT_GE(CounterVal(cli_text, "dfkv_rdma_client_max_block_seen_bytes"),
            2048);
  EXPECT_EQ(CounterVal(cli_text,
                       "dfkv_rdma_client_transient_user_mr_active"), 0);
  EXPECT_GE(CounterVal(cli_text, "dfkv_rdma_client_v2_probe_attempts_total"),
            1);
  EXPECT_GE(CounterVal(cli_text,
                       "dfkv_rdma_client_completion_timeouts_total"), 0);

  // and the client snapshot folds transport metrics in after the health metrics
  std::string snap = c.MetricsSnapshot();
  EXPECT_NE(snap.find("dfkv_client_ops_served_total"), std::string::npos) << snap;
  EXPECT_NE(snap.find("dfkv_rdma_client_conns_opened_total"), std::string::npos) << snap;
}

// dfkv_rdma_active_conns is what an operator (and the test below) waits on to
// know a connection is gone. This pins the first half of that contract: the
// gauge must still count a connection when its endpoint teardown starts. It
// used to drop before. That the receive lease and the connection accounting
// are released by the time it drops is covered by the next test, which reads
// those metrics right after the gauge reaches zero.
TEST(RdmaLoopback, ActiveConnsStaysUpUntilTheEndpointIsTornDown) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv idle_reclaim("DFKV_RDMA_IDLE_MS", "200");
  // The hook also runs for a connection that failed its bootstrap and never
  // served, so count the teardowns that saw a live gauge instead of requiring
  // it of every call.
  std::atomic<uint64_t> teardowns_with_live_gauge{0};
  RdmaNode node("active-conns-order");
  RdmaServer* const server = node.rsrv.get();
  RdmaServerTestPeer::SetBeforeEndpointTeardown(server, [&, server] {
    if (server->ActiveConns() >= 1) teardowns_with_live_gauge.fetch_add(1);
  });
  uint64_t served = 0;
  {
    RdmaTransport transport(kMaxMsg);
    KVClient client({{"n", node.addr}}, SelfHdr(), &transport);
    const std::string value(4096, 'a');
    ASSERT_TRUE(client.Put("active-conns-order", value.data(), value.size()));
    served = server->ActiveConns();
    ASSERT_GE(served, 1u);
  }
  for (int i = 0; i < 1000 && server->ActiveConns() != 0; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  ASSERT_EQ(server->ActiveConns(), 0u);
  EXPECT_GE(teardowns_with_live_gauge.load(), served)
      << "the gauge no longer counted a connection whose endpoint was still "
         "being torn down";
}

TEST(RdmaLoopback, SegmentOwnershipMetricsReturnToZeroAfterClientTeardown) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv idle_reclaim("DFKV_RDMA_IDLE_MS", "200");
  RdmaNode node("segment-metric-teardown");
  {
    RdmaTransport transport(kMaxMsg);
    KVClient client({{"n", node.addr}}, SelfHdr(), &transport);
    const std::string value(4096, 's');
    ASSERT_TRUE(client.Put("segment-owned", value.data(), value.size()));
    const std::string active = node.rsrv->MetricsText();
    EXPECT_GT(CounterVal(active, "dfkv_rdma_recv_segment_used_bytes"), 0);
    EXPECT_GT(CounterVal(active, "dfkv_rdma_pull_connections"), 0);
    EXPECT_GT(CounterVal(
                  active,
                  "dfkv_rdma_connection_bytes{class=\"data\"}"),
              0);
  }
  for (int i = 0; i < 1000 && node.rsrv->ActiveConns() != 0; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  ASSERT_EQ(node.rsrv->ActiveConns(), 0u);
  const std::string released = node.rsrv->MetricsText();
  EXPECT_EQ(CounterVal(released, "dfkv_rdma_recv_segment_used_bytes"), 0);
  EXPECT_EQ(CounterVal(released, "dfkv_rdma_pull_connections"), 0);
  EXPECT_EQ(CounterVal(
                released,
                "dfkv_rdma_connection_bytes{class=\"data\"}"),
            0);
  EXPECT_EQ(CounterVal(
                released,
                "dfkv_rdma_connection_bytes{class=\"control\"}"),
            0);
}

TEST(RdmaLoopback, MembersReplyHonorsExactBoundAndRejectsOversize) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("large-members");
  const std::string boundary(rdma::kV2ControlResponseMax, 'm');
  node.srv->set_members(boundary);

  RdmaTransport transport(kMaxMsg);
  std::string output;
  ASSERT_EQ(transport.Members(node.addr, &output), Status::kOk);
  EXPECT_EQ(output, boundary);
  EXPECT_GE(node.rsrv->V2Conns(), 1u);

  node.srv->set_members(
      std::string(rdma::kV2ControlResponseMax + 1, 'x'));
  output = "must be cleared";
  EXPECT_EQ(transport.Members(node.addr, &output), Status::kIOError);
  EXPECT_TRUE(output.empty());

  node.srv->set_members(boundary);
  ASSERT_EQ(transport.Members(node.addr, &output), Status::kOk);
  EXPECT_EQ(output, boundary);
}

TEST(RdmaLoopback, StartupFailsWhenV2ReceiveSegmentIsUnavailable) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ::setenv("DFKV_RDMA_RECV_SEGMENT_SIZE", "4096", 1);
  RdmaServer server(
      [](uint8_t, const BlockKey&, uint64_t, uint64_t, const char*, uint64_t,
         std::string*, size_t*) { return Status::kInvalid; },
      kMaxMsg);
  EXPECT_EQ(server.Start(0), Status::kIOError);
  ::unsetenv("DFKV_RDMA_RECV_SEGMENT_SIZE");
}


TEST(RdmaLoopback, V2CacheAcceptsReadOnlySourceMemory) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("readonly-v2");
  RdmaTransport transport(kMaxMsg);
  const BlockKey key = ToBlockKey(SelfHdr(), "readonly-v2");
  constexpr size_t kPayload = 512;
  const size_t stored_len = kPayload;
  void* mapping =
      ::mmap(nullptr, 4096, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(mapping, MAP_FAILED);
  std::memset(mapping, 'r', kPayload);
  ASSERT_EQ(::mprotect(mapping, 4096, PROT_READ), 0);

  ASSERT_EQ(transport.Cache(node.addr, key, mapping, stored_len), Status::kOk);
  std::string output;
  ASSERT_EQ(transport.Range(node.addr, key, 0, stored_len, &output),
            Status::kOk);
  ASSERT_EQ(output.size(), stored_len);
  EXPECT_EQ(std::memcmp(output.data(), mapping, stored_len), 0);
  EXPECT_EQ(::munmap(mapping, 4096), 0);
}

TEST(RdmaLoopback, DirectSingleCacheRangeUsesPutWriteAndGetRead) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("direct-v2");
  RdmaTransport transport(kMaxMsg);
  const BlockKey key = ToBlockKey(SelfHdr(), "direct-v2");
  std::string stored(4096, '\0');
  for (size_t i = 0; i < stored.size(); ++i)
    stored[i] = static_cast<char>((i * 29 + 3) & 0xff);

  ASSERT_EQ(transport.Cache(node.addr, key, stored.data(), stored.size()),
            Status::kOk);
  std::string output;
  ASSERT_EQ(transport.Range(node.addr, key, 0, stored.size(), &output),
            Status::kOk);
  EXPECT_EQ(output, stored);
  EXPECT_GE(node.rsrv->V2PutWrites(), 1u);
  EXPECT_GE(CounterVal(transport.MetricsText(),
                       "dfkv_rdma_client_pull_reads_total"), 1);
}

TEST(RdmaLoopback, RangeReadsNonalignedHeadTailAndBoundarySlices) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "0");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  RdmaNode node("range-slices");
  RdmaTransport transport(kMaxMsg);
  const BlockKey key = ToBlockKey(SelfHdr(), "range-slices");
  const std::string value = PatternValue(65537, 41);
  ASSERT_EQ(transport.Cache(node.addr, key, value.data(), value.size()),
            Status::kOk);
  struct Slice { uint64_t offset; uint64_t length; Status status; };
  const std::array<Slice, 6> slices{{
      {97, 4097, Status::kOk},
      {0, 13, Status::kOk},
      {value.size() - 19, 128, Status::kOk},
      {value.size(), 97, Status::kOk},
      {value.size() + 1, 97, Status::kInvalid},
      {std::numeric_limits<uint64_t>::max() - 3, 11, Status::kInvalid}}};
  for (const auto& slice : slices) {
    SCOPED_TRACE(slice.offset);
    std::string output = "stale";
    uint64_t value_len = 999;
    ASSERT_EQ(transport.Range(node.addr, key, slice.offset, slice.length,
                              &output, &value_len), slice.status);
    if (slice.status == Status::kOk) {
      EXPECT_EQ(output, value.substr(slice.offset, slice.length));
      EXPECT_EQ(value_len, value.size());
    } else {
      EXPECT_TRUE(output.empty());
      EXPECT_EQ(value_len, 0u);
    }
    EXPECT_EQ(CounterVal(node.rsrv->MetricsText(),
                         "dfkv_rdma_dynamic_get_mr_active"), 0);
  }
}

TEST(RdmaLoopback, RangeManyKeepsMixedHitMissOrderingAndSliceValueLengths) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "0");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");
  RdmaNode node("range-many-slices");
  RdmaTransport transport(kMaxMsg);
  const BlockKey first = ToBlockKey(SelfHdr(), "first");
  const BlockKey second = ToBlockKey(SelfHdr(), "second");
  const BlockKey missing = ToBlockKey(SelfHdr(), "missing");
  const std::string a = PatternValue(65537, 27);
  const std::string b = PatternValue(4113, 28);
  ASSERT_EQ(transport.Cache(node.addr, first, a.data(), a.size()), Status::kOk);
  ASSERT_EQ(transport.Cache(node.addr, second, b.data(), b.size()), Status::kOk);
  const std::vector<BlockKey> keys{first, missing, second, first, missing};
  const std::array<std::pair<uint64_t, uint64_t>, 6> slices{{
      {97, 4097}, {0, 13}, {a.size() - 19, 128},
      {a.size(), 17}, {a.size() + 1, 19},
      {std::numeric_limits<uint64_t>::max() - 3, 11}}};
  for (const auto& [offset, length] : slices) {
    SCOPED_TRACE(offset);
    std::vector<std::string> outputs(1, "stale");
    std::vector<uint64_t> value_lens(1, 999);
    const auto statuses = transport.RangeMany(
        node.addr, keys, offset, length, &outputs, &value_lens);
    ASSERT_EQ(statuses.size(), keys.size());
    ASSERT_EQ(outputs.size(), keys.size());
    ASSERT_EQ(value_lens.size(), keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      SCOPED_TRACE(i);
      const bool miss = i == 1 || i == 4;
      const std::string& stored = i == 2 ? b : a;
      const Status expected = miss ? Status::kNotFound :
          offset > stored.size() ? Status::kInvalid : Status::kOk;
      EXPECT_EQ(statuses[i], expected);
      if (expected == Status::kOk) {
        EXPECT_EQ(outputs[i], stored.substr(offset, length));
        EXPECT_EQ(value_lens[i], stored.size());
      } else {
        EXPECT_TRUE(outputs[i].empty());
        EXPECT_EQ(value_lens[i], 0u);
      }
    }
    EXPECT_EQ(CounterVal(node.rsrv->MetricsText(),
                         "dfkv_rdma_dynamic_get_mr_active"), 0);
  }
}

TEST(RdmaLoopback, ZeroLengthRangeAndRangeManyPreserveStatusAndStoredLength) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "0");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  RdmaNode node("zero-range");
  RdmaTransport transport(kMaxMsg);
  const BlockKey present = ToBlockKey(SelfHdr(), "present");
  const BlockKey missing = ToBlockKey(SelfHdr(), "missing");
  const std::string value = PatternValue(20, 31);
  ASSERT_EQ(transport.Cache(node.addr, present, value.data(), value.size()),
            Status::kOk);
  const std::array<uint64_t, 5> offsets{
      0, 7, value.size(), value.size() + 1,
      std::numeric_limits<uint64_t>::max()};
  for (uint64_t offset : offsets) {
    SCOPED_TRACE(offset);
    const Status expected = offset == value.size() ?
        Status::kOk : Status::kInvalid;
    const uint64_t expected_len = expected == Status::kOk ? value.size() : 0;
    std::string output = "stale";
    uint64_t value_len = 999;
    EXPECT_EQ(transport.Range(node.addr, present, offset, 0,
                              &output, &value_len), expected);
    EXPECT_TRUE(output.empty());
    EXPECT_EQ(value_len, expected_len);
    output = "stale";
    value_len = 999;
    EXPECT_EQ(transport.Range(node.addr, missing, offset, 0,
                              &output, &value_len), Status::kNotFound);
    EXPECT_TRUE(output.empty());
    EXPECT_EQ(value_len, 0u);
    std::vector<std::string> outputs;
    std::vector<uint64_t> value_lens;
    EXPECT_EQ(transport.RangeMany(node.addr, {present, missing, present},
                                  offset, 0, &outputs, &value_lens),
              std::vector<Status>({expected, Status::kNotFound, expected}));
    EXPECT_EQ(outputs, std::vector<std::string>(3));
    EXPECT_EQ(value_lens,
              std::vector<uint64_t>({expected_len, 0, expected_len}));
  }
  EXPECT_EQ(CounterVal(node.rsrv->MetricsText(),
                       "dfkv_rdma_dynamic_get_mr_active"), 0);
}

TEST(RdmaLoopback, BatchZeroCopyRoundtrip) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("bzc");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  const int N = 20;
  const size_t sz = 4096;
  std::vector<std::string> vals(N), keys(N);
  std::vector<KvPutItem> puts(N);
  for (int i = 0; i < N; ++i) {
    keys[i] = "b" + std::to_string(i);
    vals[i].assign(sz, static_cast<char>((i * 13 + 1) & 0xFF));
    puts[i] = {keys[i], vals[i].data(), sz};
  }
  auto pr = c.BatchPut(puts);
  for (int i = 0; i < N; ++i) EXPECT_TRUE(pr[i]) << i;

  // GET into fresh buffers (RDMA scatters payload straight in = zero copy).
  std::vector<std::string> outs(N);
  std::vector<KvGetItem> gets(N);
  for (int i = 0; i < N; ++i) { outs[i].assign(sz, '\0'); gets[i] = {keys[i], &outs[i][0], sz}; }
  auto gr = c.BatchGet(gets);
  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(gr[i]) << i;
    EXPECT_EQ(outs[i], vals[i]) << i;
  }
}

// Single Put/Get always take the zero-copy fast path on RDMA (register caller
// buffer + scatter-send / scatter-recv) — no size threshold.
TEST(RdmaLoopback, SingleZeroCopyPutGet) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("szc");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  std::string v(8192, '\0');
  for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<char>((i * 37 + 11) & 0xFF);
  // Put twice into the SAME source buffer to exercise the MR-cache hit on re-put.
  ASSERT_TRUE(c.Put("z1", v.data(), v.size()));
  ASSERT_TRUE(c.Put("z1", v.data(), v.size()));

  // Get into the SAME dst buffer twice (scatter-recv straight in; MR-cache hit).
  std::string out(v.size(), '\0');
  ASSERT_TRUE(c.Get("z1", &out[0], out.size()));
  EXPECT_EQ(out, v);
  out.assign(v.size(), '\0');
  ASSERT_TRUE(c.Get("z1", &out[0], out.size()));
  EXPECT_EQ(out, v);

  // Miss on the zero-copy Get path must be a clean miss (kNotFound), not an error.
  std::string m(v.size(), '\0');
  EXPECT_FALSE(c.Get("absent", &m[0], m.size()));
  // Size mismatch (stored payload_len != requested n) => miss, not corruption.
  std::string shorter(v.size() / 2, '\0');
  EXPECT_FALSE(c.Get("z1", &shorter[0], shorter.size()));
}

// A registered memory region (RegisterMemory) is registered once; every buffer
// inside it resolves to that one MR with no per-op ibv_reg_mr. Verified directly
// at the endpoint: two distinct sub-buffers return the SAME MR; an outside buffer
// registers ad-hoc (a different MR).
// Many endpoints on the same device must all Open via the shared per-device
// ibv_context+PD registry (#6 fix: no per-connection ibv_open_device thrash).
// Opening + closing N in waves exercises the refcount get-or-create/free path;
// a leak or double-free here would surface as an Open failure or crash.
TEST(RdmaLoopback, ManyEndpointsShareDeviceContext) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  constexpr int N = 24;  // > typical t16; well past the 1-2 conn happy path
  {
    std::vector<std::unique_ptr<rdma::RcEndpoint>> eps;
    for (int i = 0; i < N; ++i) {
      eps.push_back(std::make_unique<rdma::RcEndpoint>());
      ASSERT_TRUE(eps.back()->Open(nullptr, 64 * 1024, 1)) << "endpoint " << i;
    }
    eps.clear();  // all close -> registry refcount returns to 0, frees ctx+pd
  }
  // A fresh endpoint after the registry drained must still open (re-creates the
  // shared device cleanly).
  rdma::RcEndpoint again;
  ASSERT_TRUE(again.Open(nullptr, 64 * 1024, 1));
}


TEST(RdmaLoopback, RegisterMemoryRejectsInvalidRange) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaTransport rt(kMaxMsg);
  char byte = 0;
  EXPECT_FALSE(rt.RegisterMemory(nullptr, 1));
  EXPECT_FALSE(rt.RegisterMemory(&byte, 0));
  const std::string metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_mr_regions"), 0);
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_mr_registered_bytes"), 0);
  EXPECT_EQ(CounterVal(
                metrics,
                "dfkv_rdma_client_mr_registration_rejections_total"),
            2);
}

// Client-side anchor: RegisterMemory must register the pool MRs at
// DECLARATION time (holding a lifetime device ref), not on the first
// connection's first op — a 141 GB host pool costs ~4 s to pin, which
// belongs in engine startup, not the first lookup.
TEST(RdmaLoopback, RegisterMemoryAnchorsPoolMrBeforeFirstConn) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const uint64_t before = rdma::RcEndpoint::PoolMrRegistrations();
  RdmaTransport rt(kMaxMsg);
  std::vector<char> pool(256 * 1024);
  ASSERT_TRUE(rt.RegisterMemory(pool.data(), 64 * 1024));
  EXPECT_GT(rdma::RcEndpoint::PoolMrRegistrations(), before)
      << "pool MR not registered at declaration time (anchor missing)";
  std::string metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_mr_regions"), 1);
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_mr_registered_bytes"),
            64 * 1024);

  ASSERT_TRUE(rt.RegisterMemory(pool.data(), pool.size()));
  metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_mr_regions"), 1);
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_mr_registered_bytes"),
            static_cast<long>(pool.size()));
  EXPECT_EQ(CounterVal(
                metrics,
                "dfkv_rdma_client_mr_registration_rejections_total"),
            0);

  const uintptr_t base = reinterpret_cast<uintptr_t>(pool.data());
  const size_t wrapping_size =
      std::numeric_limits<uintptr_t>::max() - base + 1;
  EXPECT_FALSE(rt.RegisterMemory(pool.data(), wrapping_size));
  metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_mr_regions"), 1);
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_mr_registered_bytes"),
            static_cast<long>(pool.size()));
  EXPECT_EQ(CounterVal(
                metrics,
                "dfkv_rdma_client_mr_registration_rejections_total"),
            1);
}

TEST(RdmaLoopback, SameBasePoolRegistrationGrowthIsEffective) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  rdma::RcEndpoint endpoint;
  rdma::RcEndpoint old_generation_user;
  ASSERT_TRUE(endpoint.Open(nullptr, 64 * 1024, 1));
  ASSERT_TRUE(old_generation_user.Open(nullptr, 64 * 1024, 1));
  std::vector<char> region(256 * 1024);
  ASSERT_TRUE(endpoint.AddPoolMr(region.data(), 64 * 1024, true));
  ASSERT_TRUE(
      old_generation_user.AddPoolMr(region.data(), 64 * 1024, true));
  ibv_mr* initial = endpoint.RegisterUser(region.data() + 4096, 4096);
  ASSERT_NE(initial, nullptr);
  EXPECT_EQ(old_generation_user.RegisterUser(region.data() + 4096, 4096),
            initial);
  const uint64_t registrations = rdma::RcEndpoint::PoolMrRegistrations();
  const uint64_t active =
      rdma::RcEndpoint::PoolMrActiveRegistrations();

  ASSERT_TRUE(endpoint.AddPoolMr(region.data(), region.size(), true));
  EXPECT_EQ(rdma::RcEndpoint::PoolMrRegistrations(), registrations + 1);
  EXPECT_EQ(rdma::RcEndpoint::PoolMrActiveRegistrations(), active + 1)
      << "the old generation stays leased by its current endpoint";
  ibv_mr* grown =
      endpoint.RegisterUser(region.data() + 128 * 1024, 4096);
  ASSERT_NE(grown, nullptr);
  EXPECT_NE(grown, initial);
  EXPECT_EQ(endpoint.RegisterUser(region.data() + 4096, 4096), grown)
      << "the grown endpoint resolves the old prefix through the new MR";
  EXPECT_EQ(old_generation_user.RegisterUser(region.data() + 4096, 4096),
            initial)
      << "an endpoint using the previous generation keeps its old range valid";
  EXPECT_EQ(old_generation_user.RegisterUser(
                region.data() + 128 * 1024, 4096),
            nullptr);

  ASSERT_TRUE(old_generation_user.AddPoolMr(
      region.data(), region.size(), true));
  EXPECT_EQ(rdma::RcEndpoint::PoolMrActiveRegistrations(), active)
      << "the superseded generation retires after its final endpoint advances";
  EXPECT_EQ(old_generation_user.RegisterUser(region.data() + 4096, 4096),
            grown);
}

TEST(RdmaLoopback, UnregisteredBuffersLeaveNoLiveMrAfterReturn) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("transient-lifetime");
  RdmaTransport transport(kMaxMsg);
  const BlockKey key = ToBlockKey(SelfHdr(), "transient-lifetime");
  const uint64_t baseline = rdma::RcEndpoint::TransientUserMrActive();

  {
    std::vector<char> source(4096, 't');
    const auto statuses = transport.CacheFrom(
        node.addr, {CacheSrc{key, source.data(), source.size()}});
    ASSERT_EQ(statuses.size(), 1u);
    ASSERT_EQ(statuses[0], Status::kOk);
    EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), baseline);
  }  // source may be freed immediately: no cached MR may reference it.

  {
    std::vector<char> destination(4096);
    std::vector<uint64_t> value_lengths;
    const auto statuses = transport.RangeInto(
        node.addr, {key},
        {RangeDst{destination.data(), destination.size()}}, &value_lengths);
    ASSERT_EQ(statuses.size(), 1u);
    ASSERT_EQ(statuses[0], Status::kOk);
    EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), baseline);
  }  // destination may be freed immediately after return as well.
}

TEST(RdmaLoopback, PoolMrSharedAcrossBuffers) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  rdma::RcEndpoint ep;
  ASSERT_TRUE(ep.Open(nullptr, 64 * 1024, 1));
  std::vector<char> region(256 * 1024);
  ASSERT_TRUE(ep.AddPoolMr(region.data(), region.size()));
  ibv_mr* a = ep.RegisterUser(region.data() + 4096, 4096);
  ibv_mr* b = ep.RegisterUser(region.data() + 200000, 4096);
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(a, b);  // both inside the pool -> one shared MR, no per-op registration
  std::vector<char> outside(4096);
  EXPECT_EQ(ep.RegisterUser(outside.data(), outside.size()), nullptr)
      << "out-of-pool stable lookup must fail closed, never cache caller memory";
}

TEST(RdmaLoopback, PoolMrAccessUpgradeWinsRangeLookup) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  rdma::RcEndpoint ep;
  ASSERT_TRUE(ep.Open(nullptr, 64 * 1024, 1));
  std::vector<char> region(256 * 1024);
  ASSERT_TRUE(ep.AddPoolMr(region.data(), region.size()));
  ibv_mr* local_only = ep.RegisterUser(region.data() + 4096, 4096);
  ASSERT_NE(local_only, nullptr);
  ASSERT_TRUE(ep.AddPoolMr(region.data(), region.size(),
                           /*remote_write=*/true));
  ibv_mr* remote_write = ep.RegisterUser(region.data(), region.size());
  ASSERT_NE(remote_write, nullptr);
  EXPECT_NE(remote_write, local_only);
  EXPECT_EQ(ep.RegisterUser(region.data() + 4096, 4096), remote_write)
      << "range lookup must prefer the remote-write access upgrade";
}

// The host KV pool belongs to the PD, not to a connection: many endpoints on
// the same device must register it ONCE (the #P1-2 fix), not once per connection
// (the pre-fix storm — re-running ibv_reg_mr over a tens-of-GB region on every
// reconnect). Verified via the pool-registration counter delta, and by every
// endpoint resolving an in-pool buffer to the SAME MR (shared lkey).
namespace {
std::vector<std::string> ConfiguredRails() {
  const char* devs = std::getenv("DFKV_RDMA_DEV");
  std::vector<std::string> out;
  const std::string list = devs ? devs : "";
  size_t start = 0;
  while (start < list.size()) {
    size_t comma = list.find(',', start);
    if (comma == std::string::npos) comma = list.size();
    if (comma > start) out.push_back(list.substr(start, comma - start));
    start = comma + 1;
  }
  return out;
}
}  // namespace

// RDS-15: registering a large pool region (a host KV pool, the RAM arena)
// held the process-wide device lock for the whole ibv_reg_mr, which pins and
// zero-fills every page: measured 94-103 ms for 256 MiB. Opening or closing an
// endpoint on any other rail waited for it.
TEST(RdmaVerbs, PoolRegistrationDoesNotStallOtherRails) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const auto rails = ConfiguredRails();
  if (rails.size() < 2) GTEST_SKIP() << "needs DFKV_RDMA_DEV with two rails";
  constexpr size_t kRegion = 256ull << 20;
  void* region = ::mmap(nullptr, kRegion, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(region, MAP_FAILED);
  struct Sample { double start_ms, ms; };
  std::vector<Sample> samples;
  std::atomic<bool> stop{false};
  const auto origin = std::chrono::steady_clock::now();
  const auto since = [&](std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(t - origin).count();
  };
  rdma::RcEndpoint keep_open;  // the other rail's device stays open
  ASSERT_TRUE(keep_open.Open(rails[1].c_str(), rdma::kV2ControlCap, 1));
  std::thread other([&] {
    while (!stop.load()) {
      const auto t0 = std::chrono::steady_clock::now();
      {
        rdma::RcEndpoint ep;
        if (!ep.Open(rails[1].c_str(), rdma::kV2ControlCap, 4)) break;
      }
      const auto t1 = std::chrono::steady_clock::now();
      samples.push_back({since(t0), since(t1) - since(t0)});
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  double reg_start = 0, reg_end = 0;
  {
    rdma::RcEndpoint ep;
    ASSERT_TRUE(ep.Open(rails[0].c_str(), rdma::kV2ControlCap, 1));
    reg_start = since(std::chrono::steady_clock::now());
    const bool registered = ep.AddPoolMr(region, kRegion, true);
    reg_end = since(std::chrono::steady_clock::now());
    stop = true;
    other.join();
    if (!registered) {
      ::munmap(region, kRegion);
      GTEST_SKIP() << "cannot register 256 MiB here (RLIMIT_MEMLOCK)";
    }
  }
  ::munmap(region, kRegion);
  const double reg_ms = reg_end - reg_start;
  if (reg_ms < 40) GTEST_SKIP() << "registration took " << reg_ms
                                << " ms, too short to tell";
  double worst = 0;
  size_t during = 0;
  for (const auto& sample : samples) {
    if (sample.start_ms + sample.ms <= reg_start || sample.start_ms >= reg_end)
      continue;
    ++during;
    worst = std::max(worst, sample.ms);
  }
  std::printf("pool registration %.1f ms; open+close on the other rail during"
              " it: %zu, the longest %.1f ms\n", reg_ms, during, worst);
  EXPECT_GE(during, 2u) << "the other rail made no progress during a "
                        << reg_ms << " ms registration";
  EXPECT_LT(worst, reg_ms / 2) << "an open+close on the other rail took "
                               << worst << " ms during a " << reg_ms
                               << " ms registration";
}

// RDS-15: the control buffers were new char[] of 32 KiB, heap memory that is
// not page-aligned, so their NUMA bind (mbind) failed with EINVAL and did
// nothing; and each was registered on its own (2 x depth registrations).
TEST(RdmaVerbs, ControlBuffersArePageAlignedForTheirNumaBind) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  rdma::RcEndpoint ep;
  ASSERT_TRUE(ep.Open(nullptr, rdma::kV2ControlCap, 4));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ep.sbuf(0)) % 4096, 0u);
  for (size_t i = 0; i < 4; ++i) {
    std::memset(ep.sbuf(i), 's', rdma::kV2ControlCap);
    std::memset(ep.rbuf(i), 'r', rdma::kV2ControlCap);
  }
  for (size_t i = 0; i < 4; ++i) {  // no buffer overlaps another
    EXPECT_EQ(ep.sbuf(i)[0], 's');
    EXPECT_EQ(ep.sbuf(i)[rdma::kV2ControlCap - 1], 's');
    EXPECT_EQ(ep.rbuf(i)[0], 'r');
    EXPECT_EQ(ep.rbuf(i)[rdma::kV2ControlCap - 1], 'r');
  }
}

// RDS-15: ibv_query_port went to the device on every endpoint open (about
// 1.5 ms with the GID query, the largest single part of an open). A device's
// port attributes are now queried once and reused while the port is ACTIVE.
TEST(RdmaVerbs, PortAttributesAreQueriedOncePerDevice) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  rdma::RcEndpoint first;
  ASSERT_TRUE(first.Open(nullptr, rdma::kV2ControlCap, 1));
  const uint64_t before = rdma::RcEndpoint::PortQueries();
  for (int i = 0; i < 5; ++i) {
    rdma::RcEndpoint ep;
    ASSERT_TRUE(ep.Open(nullptr, rdma::kV2ControlCap, 1));
    EXPECT_EQ(ep.Local().lid, first.Local().lid);
    EXPECT_EQ(std::memcmp(ep.Local().gid, first.Local().gid, 16), 0);
  }
  // One query, or two if the opens straddle the 1 s reuse bound (slow
  // builds); without the cache it was one per open.
  EXPECT_LE(rdma::RcEndpoint::PortQueries() - before, 2u);
}

// RDS-15: two endpoints register the same pool region at once; the later
// one's ibv_reg_mr fails (memlock sized for one pin, say) after the first has
// published its MR. It must take the published MR, not fail the connection.
TEST(RdmaVerbs, ALosingConcurrentPoolRegistrationTakesTheWinners) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  std::vector<char> region(1 << 20);
  rdma::RcEndpoint a, b;
  ASSERT_TRUE(a.Open(nullptr, rdma::kV2ControlCap, 1));
  ASSERT_TRUE(b.Open(nullptr, rdma::kV2ControlCap, 1));
  std::mutex mu;
  std::condition_variable cv;
  int entered = 0;
  bool winner_done = false;
  rdma::RcEndpoint::SetPoolRegistrationSeamForTest([&] {
    std::unique_lock<std::mutex> lock(mu);
    const int me = ++entered;
    cv.notify_all();
    if (me == 1) {  // the winner: go on once the loser has missed too
      cv.wait(lock, [&] { return entered >= 2; });
      return true;
    }
    cv.wait(lock, [&] { return winner_done; });  // the loser
    return false;
  });
  struct ClearSeam {
    ~ClearSeam() { rdma::RcEndpoint::SetPoolRegistrationSeamForTest({}); }
  } clear_seam;
  const uint64_t failures = rdma::RcEndpoint::PoolMrRegistrationFailures();
  bool a_ok = false, b_ok = false;
  std::thread winner([&] {
    a_ok = a.AddPoolMr(region.data(), region.size(), true);
    std::lock_guard<std::mutex> lock(mu);
    winner_done = true;
    cv.notify_all();
  });
  {  // the winner registers first; only then the loser starts
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] { return entered >= 1; });
  }
  std::thread loser([&] {
    b_ok = b.AddPoolMr(region.data(), region.size(), true);
  });
  winner.join();
  loser.join();
  EXPECT_TRUE(a_ok);
  EXPECT_TRUE(b_ok) << "the loser failed though the region was registered";
  EXPECT_EQ(rdma::RcEndpoint::PoolMrRegistrationFailures(), failures);
  ibv_mr* ma = a.RegisterUser(region.data(), 4096);
  ibv_mr* mb = b.RegisterUser(region.data(), 4096);
  ASSERT_NE(ma, nullptr);
  EXPECT_EQ(ma, mb) << "both endpoints use the one registration";
}

// TRN-17: a region registered or grown with RegisterMemory is pinned on
// every rail without the transport's pool lock; only the commit takes it.
// Growing a region to 2 GiB took 666-850 ms on two rails, and every
// operation of the transport waited for it. A seam holds the registration
// here: an operation on the same transport completes meanwhile, and sees
// the old declaration until the new one is published whole.
// Holds every caller at a point until released.
struct Gate {
  std::mutex mu;
  std::condition_variable cv;
  int entered = 0;
  bool released = false;
  void Pass() {
    std::unique_lock<std::mutex> lock(mu);
    ++entered;
    cv.notify_all();
    cv.wait(lock, [this] { return released; });
  }
  bool AwaitEntry() {
    std::unique_lock<std::mutex> lock(mu);
    return cv.wait_for(lock, std::chrono::seconds(5),
                       [this] { return entered > 0; });
  }
  void Release() {
    std::lock_guard<std::mutex> lock(mu);
    released = true;
    cv.notify_all();
  }
};

// A pool registration held until released.
struct HeldPoolRegistration : Gate {
  HeldPoolRegistration() {
    rdma::RcEndpoint::SetPoolRegistrationSeamForTest([this] {
      Pass();
      return true;
    });
  }
  ~HeldPoolRegistration() {
    Release();
    rdma::RcEndpoint::SetPoolRegistrationSeamForTest({});
  }
};

// The deregistration of a superseded pool generation held until released.
struct HeldPoolDeregistration : Gate {
  HeldPoolDeregistration() {
    rdma::RcEndpoint::SetPoolDeregistrationSeamForTest([this] { Pass(); });
  }
  ~HeldPoolDeregistration() {
    Release();
    rdma::RcEndpoint::SetPoolDeregistrationSeamForTest({});
  }
};

TEST(RdmaRegisterMemory, OperationsGoOnWhileARegionGrows) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("register-grow");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  std::vector<char> region(3 << 20);
  ASSERT_TRUE(rt.RegisterMemory(region.data(), 1 << 20));
  const std::string v(4096, 'g');
  ASSERT_TRUE(c.Put("grow", v.data(), v.size()));
  ASSERT_TRUE(c.Exist("grow"));  // pooled connections from here on
  // Whether an operation on the transport completes within 2 s. Metrics take
  // the pool lock too: read them only if the operation got it.
  const auto goes_on = [&](long* published) {
    auto op = std::async(std::launch::async, [&] { return c.Exist("grow"); });
    const bool done =
        op.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    if (published)
      *published = done ? CounterVal(rt.MetricsText(),
                                     "dfkv_rdma_client_mr_registered_bytes")
                        : -1;
    return std::make_pair(done, std::move(op));
  };
  // Two growth steps: the second supersedes a generation the first made.
  for (const size_t size : {size_t{2} << 20, size_t{3} << 20}) {
    SCOPED_TRACE("growth to " + std::to_string(size >> 20) + " MiB");
    const long before = CounterVal(rt.MetricsText(),
                                   "dfkv_rdma_client_mr_registered_bytes");
    bool grown = false;
    std::thread growing;
    {
      // Both seams stay set until the registration has returned: the other
      // rails' registrations read them after the first is released.
      HeldPoolDeregistration dereg;
      HeldPoolRegistration held;
      growing = std::thread(
          [&] { grown = rt.RegisterMemory(region.data(), size); });
      ASSERT_TRUE(held.AwaitEntry());
      {
        long published = 0;
        auto [went_on, op] = goes_on(&published);
        held.Release();
        EXPECT_TRUE(went_on) << "an operation waited for the registration";
        // Not published until every rail is registered.
        if (went_on) EXPECT_EQ(published, before);
        EXPECT_TRUE(op.get());
      }
      // The superseded generation is deregistered once nothing references
      // it, and not under the pool lock either.
      ASSERT_TRUE(dereg.AwaitEntry());
      auto [went_on, op] = goes_on(nullptr);
      dereg.Release();
      EXPECT_TRUE(went_on) << "an operation waited for a deregistration";
      EXPECT_TRUE(op.get());
      growing.join();
    }
    EXPECT_TRUE(grown);
    EXPECT_EQ(CounterVal(rt.MetricsText(),
                         "dfkv_rdma_client_mr_registered_bytes"),
              static_cast<long>(size));
  }
  std::string out(v.size(), '\0');
  ASSERT_TRUE(c.Get("grow", &out[0], out.size()));
  EXPECT_EQ(out, v);
}

// A superseded generation handed back for deregistration after the pool
// lock is deregistered also when the commit throws after that (an
// allocation failing, here a seam): left registered, the device's PD could
// not be deallocated when the transport goes, and that aborts.
TEST(RdmaRegisterMemory, AThrowInTheCommitStillDeregistersTheOldGeneration) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const uint64_t active = rdma::RcEndpoint::PoolMrActiveRegistrations();
  std::atomic<int> deregistered{0};
  {
    RdmaTransport rt(kMaxMsg);
    std::vector<char> region(2 << 20);
    ASSERT_TRUE(rt.RegisterMemory(region.data(), 1 << 20));
    std::atomic<bool> thrown{false};
    rdma::RcEndpoint::SetPoolRetireSeamForTest([&] {
      if (!thrown.exchange(true)) throw std::bad_alloc();
    });
    rdma::RcEndpoint::SetPoolDeregistrationSeamForTest(
        [&] { deregistered.fetch_add(1); });
    struct ClearSeams {
      ~ClearSeams() {
        rdma::RcEndpoint::SetPoolRetireSeamForTest({});
        rdma::RcEndpoint::SetPoolDeregistrationSeamForTest({});
      }
    } clear_seams;
    EXPECT_THROW(rt.RegisterMemory(region.data(), region.size()),
                 std::bad_alloc);
    EXPECT_TRUE(thrown.load());
    EXPECT_GE(deregistered.load(), 1)
        << "the generation handed back before the throw was not deregistered";
  }  // the transport goes: its PDs deallocate (an MR left would abort)
  EXPECT_EQ(rdma::RcEndpoint::PoolMrActiveRegistrations(), active);
}

// Destroying the transport while a declaration registers waits for it, and
// tears down after it: nothing it registers is left behind.
TEST(RdmaRegisterMemory, DestructionWaitsForARegistrationInFlight) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const uint64_t active = rdma::RcEndpoint::PoolMrActiveRegistrations();
  std::vector<char> region(1 << 20);
  auto rt = std::make_unique<RdmaTransport>(kMaxMsg);
  bool registered = false;
  std::atomic<bool> destroyed{false};
  {
    HeldPoolRegistration held;
    std::thread registering(
        [&] { registered = rt->RegisterMemory(region.data(), region.size()); });
    ASSERT_TRUE(held.AwaitEntry());
    std::thread destroying([&] {
      rt.reset();
      destroyed = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_FALSE(destroyed.load()) << "torn down under a registration";
    held.Release();
    registering.join();
    destroying.join();
  }
  EXPECT_TRUE(registered);
  EXPECT_EQ(rdma::RcEndpoint::PoolMrActiveRegistrations(), active);
}

// A rail whose registration fails undoes the rails already registered; the
// published declaration and the operations on it are as before.
TEST(RdmaRegisterMemory, AFailedRailUndoesTheOthersAndPublishesNothing) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  if (ConfiguredRails().size() < 2)
    GTEST_SKIP() << "needs DFKV_RDMA_DEV with two rails";
  RdmaNode node("register-fail");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  std::vector<char> region(2 << 20);
  ASSERT_TRUE(rt.RegisterMemory(region.data(), 1 << 20));
  ASSERT_TRUE(c.Exist("fail") || true);
  const uint64_t active = rdma::RcEndpoint::PoolMrActiveRegistrations();
  std::atomic<int> calls{0};
  rdma::RcEndpoint::SetPoolRegistrationSeamForTest(
      [&] { return ++calls == 1; });  // the first rail registers, not the next
  struct ClearSeam {
    ~ClearSeam() { rdma::RcEndpoint::SetPoolRegistrationSeamForTest({}); }
  } clear_seam;
  EXPECT_FALSE(rt.RegisterMemory(region.data(), region.size()));
  EXPECT_GE(calls.load(), 2);
  EXPECT_EQ(rdma::RcEndpoint::PoolMrActiveRegistrations(), active)
      << "the first rail's registration was kept";
  EXPECT_EQ(CounterVal(rt.MetricsText(), "dfkv_rdma_client_mr_registered_bytes"),
            1 << 20);
  EXPECT_EQ(CounterVal(rt.MetricsText(),
                       "dfkv_rdma_client_mr_registration_rejections_total"),
            1);
  rdma::RcEndpoint::SetPoolRegistrationSeamForTest({});
  const std::string v(4096, 'f');
  EXPECT_TRUE(c.Put("fail", v.data(), v.size()));
  EXPECT_TRUE(rt.RegisterMemory(region.data(), region.size()));
}

// Opt-in: what an endpoint costs to open, connect and close, by depth.
// DFKV_RDMA_SETUP_BENCH=<rounds>.
TEST(RdmaVerbs, EndpointSetupBench) {
  const char* requested = std::getenv("DFKV_RDMA_SETUP_BENCH");
  if (requested == nullptr || *requested == '\0')
    GTEST_SKIP() << "set DFKV_RDMA_SETUP_BENCH=<rounds> to run";
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const int rounds = std::max(1, std::atoi(requested));
  rdma::RcEndpoint anchor;
  ASSERT_TRUE(anchor.Open(nullptr, rdma::kV2ControlCap, 1));
  const auto median = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  for (const size_t depth : {1u, 4u, 16u, 64u}) {
    std::vector<double> open_ms, connect_ms, close_ms;
    for (int round = 0; round < rounds; ++round) {
      auto t0 = std::chrono::steady_clock::now();
      auto a = std::make_unique<rdma::RcEndpoint>();
      a->set_remote_access(true);
      ASSERT_TRUE(a->Open(nullptr, rdma::kV2ControlCap, depth));
      auto t1 = std::chrono::steady_clock::now();
      rdma::RcEndpoint b;
      ASSERT_TRUE(b.Open(nullptr, rdma::kV2ControlCap, depth));
      auto t2 = std::chrono::steady_clock::now();
      ASSERT_TRUE(a->Connect(b.Local()));
      auto t3 = std::chrono::steady_clock::now();
      a.reset();
      auto t4 = std::chrono::steady_clock::now();
      open_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      connect_ms.push_back(std::chrono::duration<double, std::milli>(t3 - t2).count());
      close_ms.push_back(std::chrono::duration<double, std::milli>(t4 - t3).count());
    }
    std::printf("row setup depth=%zu: open %.2f ms, connect %.2f ms, close %.2f ms"
                " (median of %d)\n", depth, median(open_ms), median(connect_ms),
                median(close_ms), rounds);
  }
}

TEST(RdmaLoopback, PoolMrRegisteredOncePerDeviceNotPerConnection) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  // A distinct region per test run so the counter delta is attributable here.
  std::vector<char> region(512 * 1024);
  const uint64_t regs0 = rdma::RcEndpoint::PoolMrRegistrations();
  const uint64_t adhoc0 = rdma::RcEndpoint::AdhocUserMrTotal();

  constexpr int N = 8;
  std::vector<std::unique_ptr<rdma::RcEndpoint>> eps;
  ibv_mr* first = nullptr;
  for (int i = 0; i < N; ++i) {
    eps.push_back(std::make_unique<rdma::RcEndpoint>());
    ASSERT_TRUE(eps.back()->Open(nullptr, 64 * 1024, 1)) << "endpoint " << i;
    ASSERT_TRUE(eps.back()->AddPoolMr(region.data(), region.size()));
    ibv_mr* m = eps.back()->RegisterUser(region.data() + 4096, 4096);
    ASSERT_NE(m, nullptr);
    if (i == 0) first = m;
    else EXPECT_EQ(m, first) << "all endpoints on one PD share the pool MR";
  }
  EXPECT_EQ(rdma::RcEndpoint::PoolMrRegistrations() - regs0, 1u)
      << N << " endpoints registered the region " << (rdma::RcEndpoint::PoolMrRegistrations() - regs0)
      << " times; must be exactly once per device";
  EXPECT_EQ(rdma::RcEndpoint::AdhocUserMrTotal() - adhoc0, 0u)
      << "in-pool buffers must never take the ad-hoc registration path";

  eps.clear();  // all close; region MR freed when the last device ref drops
  // A fresh endpoint re-registers the (now-freed) region: counter advances by 1.
  rdma::RcEndpoint again;
  ASSERT_TRUE(again.Open(nullptr, 64 * 1024, 1));
  ASSERT_TRUE(again.AddPoolMr(region.data(), region.size()));
}

// End-to-end: register one host pool, then PUT from and GET into sub-buffers.
// Every transfer must use the pool MR; no per-operation registration is legal.
TEST(RdmaLoopback, RegisterMemoryRoundtrip) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("rmr");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  std::vector<char> pool(128 * 1024);
  ASSERT_TRUE(c.RegisterMemory(pool.data(), pool.size()));

  const size_t sz = 4096;
  for (int i = 0; i < 8; ++i) {
    char* src = pool.data() + i * sz * 2;            // distinct sub-buffer per page
    char* dst = pool.data() + i * sz * 2 + sz;       // get into a neighbouring slot
    for (size_t k = 0; k < sz; ++k) src[k] = static_cast<char>((i * 17 + k) & 0xFF);
    std::string key = "rm" + std::to_string(i);
    ASSERT_TRUE(c.Put(key, src, sz)) << i;
    ASSERT_TRUE(c.Get(key, dst, sz)) << i;
    EXPECT_EQ(0, std::memcmp(src, dst, sz)) << i;
  }
}


// A live client must not inherit the server's short dead-client reap interval
// as a first-read reconnect penalty. Idle QPs receive lightweight membership
// probes; a dead process sends none and remains reclaimable by the server.
TEST(RdmaLoopback, KeepalivePreservesIdleDataConnection) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ::setenv("DFKV_RDMA_IDLE_MS", "150", 1);
  ::setenv("DFKV_RDMA_KEEPALIVE_MS", "30", 1);
  {
    RdmaNode node("keepalive");
    RdmaTransport rt(kMaxMsg);
    KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
    std::string value(4096, 'k');
    std::string out(value.size(), '\0');
    EXPECT_TRUE(c.Put("alive", value.data(), value.size()));
    EXPECT_TRUE(c.Get("alive", out.data(), out.size()));
    EXPECT_EQ(out, value);

    const long stale_before = CounterVal(
        rt.MetricsText(), "dfkv_rdma_client_stale_pool_retries_total");
    std::this_thread::sleep_for(std::chrono::milliseconds(450));
    out.assign(value.size(), '\0');
    EXPECT_TRUE(c.Get("alive", out.data(), out.size()));
    EXPECT_EQ(out, value);
    const std::string metrics = rt.MetricsText();
    EXPECT_GT(CounterVal(
                  metrics, "dfkv_rdma_client_keepalive_successes_total"),
              0);
    EXPECT_EQ(CounterVal(
                  metrics, "dfkv_rdma_client_stale_pool_retries_total"),
              stale_before);
    EXPECT_EQ(CounterVal(
                  metrics, "dfkv_rdma_client_keepalive_failures_total"),
              0);
  }
  ::unsetenv("DFKV_RDMA_KEEPALIVE_MS");
  ::unsetenv("DFKV_RDMA_IDLE_MS");
}

// A zero-length item in a mixed RDMA batch fails independently; the valid
// neighbor still completes and no empty object reaches the server.
TEST(RdmaLoopback, BatchPutRejectsEmptyValue) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("bev");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  std::string nonempty(2048, 'x');
  std::vector<KvPutItem> puts = {
      {"e_full", nonempty.data(), nonempty.size()},
      {"e_empty", nonempty.data(), 0},
  };
  auto pr = c.BatchPut(puts);
  ASSERT_TRUE(pr[0]);
  EXPECT_FALSE(pr[1]);

  EXPECT_FALSE(c.Exist("e_empty"));
  std::string out(nonempty.size(), '\0');
  ASSERT_TRUE(c.Get("e_full", &out[0], out.size()));
  EXPECT_EQ(out, nonempty);
}

// Scatter-gather over RDMA: one logical object may span any number of caller
// descriptors. The transport windows that descriptor stream to the negotiated
// HCA SGE limit without changing object identity or byte order.
TEST(RdmaLoopback, ScatterGatherRoundtripOverRdma) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  // The configured ordinary-data depth stays >1; the dedicated SG lane must
  // nevertheless use depth-one windows.
  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");
  RdmaNode node("sg");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  ASSERT_TRUE(rt.pipelined());  // RDMA path, not native TCP

  auto make_chunks = [](const std::string& tag, const std::vector<size_t>& sizes) {
    std::vector<std::string> v(sizes.size());
    for (size_t i = 0; i < sizes.size(); ++i) {
      v[i].resize(sizes[i]);
      for (size_t b = 0; b < sizes[i]; ++b)
        v[i][b] = static_cast<char>((tag.size() + i * 31 + b * 7) & 0xFF);
    }
    return v;
  };
  auto roundtrip = [&](const std::string& key, const std::vector<size_t>& sizes) {
    auto src = make_chunks(key, sizes);
    std::vector<const void*> ptrs;
    for (auto& s : src) ptrs.push_back(s.data());
    KvPutItemSg put{key, ptrs, sizes};
    const auto pr = c.BatchPutSg({put});
    ASSERT_EQ(pr.size(), 1u);
    ASSERT_TRUE(pr[0]) << "put " << key;
    std::vector<std::string> dst(sizes.size());
    std::vector<void*> dptrs;
    for (size_t i = 0; i < sizes.size(); ++i) {
      dst[i].assign(sizes[i], '\0');
      dptrs.push_back(dst[i].data());
    }
    KvGetItemSg get{key, dptrs, sizes};
    std::vector<size_t> lens;
    const auto gr = c.BatchGetAutoSg({get}, &lens);
    ASSERT_EQ(gr.size(), 1u);
    ASSERT_TRUE(gr[0]) << "get " << key;
    size_t total = 0; for (size_t s : sizes) total += s;
    EXPECT_EQ(lens[0], total);
    for (size_t i = 0; i < sizes.size(); ++i) EXPECT_EQ(dst[i], src[i]) << key << " seg " << i;
  };

  const size_t max_sg = rt.MaxSgPayloadSegs();
  ASSERT_GE(max_sg, 2u);
  roundtrip("sg_n1", std::vector<size_t>(1, 4096));
  roundtrip("sg_n2", std::vector<size_t>(2, 2048));
  roundtrip("sg_exact_max", std::vector<size_t>(max_sg, 256));
  roundtrip("sg_max_plus_one", std::vector<size_t>(max_sg + 1, 193));
  roundtrip("sg_multi_window", std::vector<size_t>(max_sg * 3 + 5, 71));
  roundtrip("sg_var", {1, 7, 64, 333, 4096, 11});
  roundtrip("sg_zero_descriptors", {17, 0, 31, 0, 9});

  // Insufficient aggregate capacity is rejected before any destination byte is
  // touched, even when both the value and destination cross WR boundaries.
  {
    std::vector<size_t> sizes(max_sg + 3, 32);
    auto src = make_chunks("sg_small_cap", sizes);
    std::vector<const void*> ptrs;
    for (auto& s : src) ptrs.push_back(s.data());
    const auto put = c.BatchPutSg({{"sg_small_cap", ptrs, sizes}});
    ASSERT_EQ(put.size(), 1u);
    ASSERT_TRUE(put[0]);

    std::vector<size_t> caps = sizes;
    ASSERT_FALSE(caps.empty());
    --caps.back();
    std::vector<std::string> dst(caps.size());
    std::vector<void*> dptrs;
    for (size_t i = 0; i < caps.size(); ++i) {
      dst[i].assign(caps[i], '\x5a');
      dptrs.push_back(dst[i].data());
    }
    std::vector<size_t> lens;
    const auto get = c.BatchGetAutoSg(
        {{"sg_small_cap", dptrs, caps}}, &lens);
    ASSERT_EQ(get.size(), 1u);
    EXPECT_FALSE(get[0]);
    ASSERT_EQ(lens.size(), 1u);
    EXPECT_EQ(lens[0], 0u);
    for (const auto& segment : dst)
      EXPECT_EQ(segment, std::string(segment.size(), '\x5a'));

    std::string contiguous;
    for (const auto& segment : src) contiguous += segment;
    std::string out(contiguous.size(), '\0');
    ASSERT_TRUE(c.Get("sg_small_cap", out.data(), out.size()));
    EXPECT_EQ(out, contiguous);
  }

  // Multi-key fan-out crosses the SG lane's depth-one send window. Pin client
  // concurrency to 1: Soft-RoCE (rdma_rxe) loopback races when many distinct
  // QPs run in parallel (the same flakiness affects contiguous BatchGetAuto).
  // This still exercises the real multi-SGE verbs datapath across many keys.
  {
    c.set_batch_concurrency(1);
    const int N = 24;
    std::vector<std::vector<std::string>> srcs(N);
    std::vector<KvPutItemSg> puts(N);
    for (int i = 0; i < N; ++i) {
      std::vector<size_t> sizes(1 + (i % 4), 64 + (i % 8));
      srcs[i] = make_chunks("sgm" + std::to_string(i), sizes);
      std::vector<const void*> ptrs; for (auto& s : srcs[i]) ptrs.push_back(s.data());
      puts[i] = {"sgm" + std::to_string(i), ptrs, sizes};
    }
    auto pr = c.BatchPutSg(puts);
    ASSERT_EQ(pr.size(), static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) ASSERT_TRUE(pr[i]) << i;
    std::vector<std::vector<std::string>> dsts(N);
    std::vector<KvGetItemSg> gets(N);
    for (int i = 0; i < N; ++i) {
      dsts[i].resize(srcs[i].size());
      std::vector<void*> dptrs; std::vector<size_t> caps(srcs[i].size());
      for (size_t j = 0; j < srcs[i].size(); ++j) {
        dsts[i][j].assign(srcs[i][j].size(), '\0');
        dptrs.push_back(&dsts[i][j][0]); caps[j] = srcs[i][j].size();
      }
      gets[i] = {"sgm" + std::to_string(i), dptrs, caps};
    }
    std::vector<size_t> lens;
    auto gr = c.BatchGetAutoSg(gets, &lens);
    ASSERT_EQ(gr.size(), static_cast<size_t>(N));
    ASSERT_EQ(lens.size(), static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
      ASSERT_TRUE(gr[i]) << i;
      size_t expected_len = 0;
      for (size_t j = 0; j < srcs[i].size(); ++j) {
        expected_len += srcs[i][j].size();
        EXPECT_EQ(dsts[i][j], srcs[i][j]) << i << " " << j;
      }
      EXPECT_EQ(lens[i], expected_len) << i;
    }
  }
}

TEST(RdmaLoopback, MultiWrLaterWindowFailureIsAtomicAndReclaimsState) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv idle_reclaim("DFKV_RDMA_IDLE_MS", "2000");
  RdmaNode node("sgabort");
  const uint64_t transient_baseline =
      rdma::RcEndpoint::TransientUserMrActive();
  const uint64_t active_baseline = node.rsrv->ActiveConns();
  const auto wait_for_active_at_most = [&node](uint64_t limit) {
    for (int i = 0; i < 2000 && node.rsrv->ActiveConns() > limit; ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return node.rsrv->ActiveConns();
  };

  std::vector<std::string> src;
  std::vector<const void*> ptrs;
  std::vector<size_t> sizes;
  {
    RdmaTransport rt(kMaxMsg);
    const size_t max_sg = rt.MaxSgPayloadSegs();
    ASSERT_GE(max_sg, 2u);
    src.resize(max_sg + 7);
    for (size_t i = 0; i < src.size(); ++i) {
      src[i].assign(97 + i % 11, static_cast<char>('a' + i % 26));
      ptrs.push_back(src[i].data());
      sizes.push_back(src[i].size());
    }
    KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
    const uint64_t active_before_fault = node.rsrv->ActiveConns();
    const uint64_t opened_before_fault = node.rsrv->V2Conns();
    std::vector<bool> failed;
    {
      ScopedEnv abort_window("DFKV_RDMA_TEST_ABORT_MULTIWR_WINDOW", "2");
      failed = c.BatchPutSg(
          {{"sg_later_window_abort", ptrs, sizes}});
    }
    ASSERT_EQ(failed.size(), 1u);
    EXPECT_FALSE(failed[0]);
    const uint64_t opened_delta =
        node.rsrv->V2Conns() - opened_before_fault;
    ASSERT_GT(opened_delta, 0u);
    // ActiveConns is a server-side gauge: destroying an RC peer may leave its
    // silent server half alive until the idle reaper runs.  Allow every healthy
    // connection opened by the operation, but require the faulted QP itself to
    // disappear.  One leaked QP therefore exceeds this bound by exactly one.
    const uint64_t max_active_after_fault =
        active_before_fault + opened_delta - 1;
    EXPECT_LE(wait_for_active_at_most(max_active_after_fault),
              max_active_after_fault);
    EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), transient_baseline);
  }

  EXPECT_LE(wait_for_active_at_most(active_baseline), active_baseline);
  EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), transient_baseline);

  EXPECT_EQ(node.srv->Count(), 0u);
  // A failed logical PUT never commits a readable prefix. Once fault injection
  // is removed, the same object succeeds and round-trips on a fresh connection.
  {
    RdmaTransport rt(kMaxMsg);
    KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
    EXPECT_FALSE(c.Exist("sg_later_window_abort"));
    const auto put = c.BatchPutSg(
        {{"sg_later_window_abort", ptrs, sizes}});
    ASSERT_EQ(put.size(), 1u);
    ASSERT_TRUE(put[0]);
    std::vector<std::string> dst(src.size());
    std::vector<void*> dptrs;
    for (size_t i = 0; i < src.size(); ++i) {
      dst[i].assign(src[i].size(), '\0');
      dptrs.push_back(dst[i].data());
    }
    std::vector<size_t> lens;
    const auto get = c.BatchGetAutoSg(
        {{"sg_later_window_abort", dptrs, sizes}}, &lens);
    ASSERT_EQ(get.size(), 1u);
    ASSERT_TRUE(get[0]) << rt.MetricsText();
    ASSERT_EQ(lens.size(), 1u);
    size_t expected_len = 0;
    for (size_t i = 0; i < src.size(); ++i) {
      expected_len += src[i].size();
      EXPECT_EQ(dst[i], src[i]) << i;
    }
    EXPECT_EQ(lens[0], expected_len);
  }
  EXPECT_LE(wait_for_active_at_most(active_baseline), active_baseline);
  EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), transient_baseline);
  node.rsrv->Stop();
  EXPECT_EQ(node.rsrv->ActiveConns(), 0u);
}

TEST(RdmaLoopback, ScatterGatherGetReusesUnifiedDataPool) {
  ScopedEnv configured_device("DFKV_RDMA_DEV", nullptr);
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ::setenv("DFKV_RDMA_DEPTH", "4", 1);
  RdmaNode node("sggetlane");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  std::string value = "scatter-get-payload";
  ASSERT_TRUE(c.Put("sg_get_seed", value.data(), value.size()));
  const long data_opened =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  ASSERT_EQ(data_opened, 1);

  std::string first(7, '\0');
  std::string second(value.size() - first.size(), '\0');
  KvGetItemSg get{"sg_get_seed", {first.data(), second.data()},
                  {first.size(), second.size()}};
  std::vector<size_t> lengths;
  ASSERT_TRUE(c.BatchGetAutoSg({get}, &lengths)[0]);
  ASSERT_EQ(lengths[0], value.size());
  EXPECT_EQ(first + second, value);
  const std::string metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_conns_opened_total"),
            data_opened);
  ASSERT_FALSE(node.rsrv->DeviceNames().empty());
  const std::string idle_sg =
      "dfkv_rdma_client_pool_connections{lane=\"sg\",state=\"idle\",dev=\"" +
      node.rsrv->DeviceNames().front() + "\"}";
  EXPECT_EQ(CounterVal(metrics, idle_sg), 1);

  ::unsetenv("DFKV_RDMA_DEPTH");
}

TEST(RdmaLoopback, ScatterGatherPutReturnsToUnifiedDataPool) {
  ScopedEnv configured_device("DFKV_RDMA_DEV", nullptr);
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ::setenv("DFKV_RDMA_DEPTH", "4", 1);
  RdmaNode node("sgputlane");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  std::string first = "scatter-";
  std::string second = "put-payload";
  KvPutItemSg put{"sg_put_seed", {first.data(), second.data()},
                  {first.size(), second.size()}};
  ASSERT_TRUE(c.BatchPutSg({put})[0]);
  const long sg_opened =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  ASSERT_EQ(sg_opened, 1);

  std::string ordinary = "ordinary-data";
  ASSERT_TRUE(c.Put("ordinary_after_sg", ordinary.data(), ordinary.size()));
  EXPECT_EQ(CounterVal(rt.MetricsText(),
                       "dfkv_rdma_client_conns_opened_total"),
            sg_opened);

  std::string actual(ordinary.size(), '\0');
  ASSERT_TRUE(c.Get("ordinary_after_sg", actual.data(), actual.size()));
  EXPECT_EQ(actual, ordinary);
  ::unsetenv("DFKV_RDMA_DEPTH");
}

// Fix 1 regression: an oversized SG key (total payload > max_payload_, but with a
// legal segment count so it passes the client guard) must fail ONLY itself inside
// CacheFromMulti/RangeIntoMulti, NOT poison its node batch. Previously the up-front
// validation std::fill'd every result kInvalid and returned; now the offender is
// skipped in the window and its siblings on the same node proceed normally.
// Regression: one deep window of SG items where every segment is a distinct
// unregistered buffer. All MRs must remain live until the whole posted window
// completes, then be released before the public call returns. This exercises
// depth * (max_sge-1) simultaneous transient registrations across PUT and GET.
TEST(RdmaLoopback, SgDeepWindowManyUnregisteredSegmentsSafe) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ::setenv("DFKV_RDMA_DEPTH", "4", 1);
  RdmaNode node("sgdw");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  ASSERT_TRUE(rt.pipelined());
  const uint64_t transient_baseline =
      rdma::RcEndpoint::TransientUserMrActive();

  constexpr int kItems = 8;   // > depth: also crosses windows
  constexpr int kSegs = 29;   // max payload SGEs per slot
  constexpr size_t kSeg = 512;
  std::vector<std::vector<std::string>> src(kItems);
  std::vector<KvPutItemSg> puts;
  for (int it = 0; it < kItems; ++it) {
    src[it].resize(kSegs);
    std::vector<const void*> ptrs;
    std::vector<size_t> sizes;
    for (int sg = 0; sg < kSegs; ++sg) {
      src[it][sg].assign(kSeg, static_cast<char>('a' + (it * 7 + sg) % 26));
      ptrs.push_back(src[it][sg].data());
      sizes.push_back(kSeg);
    }
    puts.push_back({"sgdw_" + std::to_string(it), ptrs, sizes});
  }
  auto pr = c.BatchPutSg(puts);
  ASSERT_EQ(pr.size(), puts.size());
  for (int it = 0; it < kItems; ++it) EXPECT_TRUE(pr[it]) << "put item " << it;
  EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), transient_baseline);

  std::vector<std::vector<std::string>> dst(kItems);
  std::vector<KvGetItemSg> gets;
  for (int it = 0; it < kItems; ++it) {
    dst[it].assign(kSegs, std::string(kSeg, '\0'));
    std::vector<void*> dptrs;
    std::vector<size_t> caps;
    for (int sg = 0; sg < kSegs; ++sg) { dptrs.push_back(&dst[it][sg][0]); caps.push_back(kSeg); }
    gets.push_back({"sgdw_" + std::to_string(it), dptrs, caps});
  }
  std::vector<size_t> lens;
  auto gr = c.BatchGetAutoSg(gets, &lens);
  ASSERT_EQ(gr.size(), gets.size());
  for (int it = 0; it < kItems; ++it) {
    EXPECT_TRUE(gr[it]) << "get item " << it;
    EXPECT_EQ(lens[it], kSegs * kSeg) << it;
    for (int sg = 0; sg < kSegs; ++sg)
      EXPECT_EQ(dst[it][sg], src[it][sg]) << "item " << it << " seg " << sg;
  }
  EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), transient_baseline);
  ::unsetenv("DFKV_RDMA_DEPTH");
}

TEST(RdmaLoopback, ScatterGatherOversizedFailsOnlyOffender) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ::setenv("DFKV_RDMA_DEPTH", "4", 1);
  RdmaNode node("sgov");
  RdmaTransport rt(kMaxMsg);  // max_payload_ = 256 KiB
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  c.set_batch_concurrency(1);  // single node, stable on rxe loopback
  ASSERT_TRUE(rt.pipelined());

  auto fill = [](std::string& s, int seed) {
    for (size_t b = 0; b < s.size(); ++b) s[b] = static_cast<char>((seed + b * 7) & 0xFF);
  };

  // Valid sibling A, an oversized item (2 segs * 200 KiB = 400 KiB > 256 KiB max_payload,
  // only 2 segments so it clears the <=29-seg client guard), then valid sibling B —
  // all routed to the same (only) node. The offender must report failure; both
  // siblings must succeed and round-trip byte-exact.
  std::string a0(4096, '\0'); fill(a0, 11);
  std::string b0(8192, '\0'); fill(b0, 23);
  std::string big0(200 * 1024, '\0'), big1(200 * 1024, '\0');
  fill(big0, 31); fill(big1, 37);

  std::vector<KvPutItemSg> puts = {
      {"sgov_a", {a0.data()}, {a0.size()}},
      {"sgov_big", {big0.data(), big1.data()}, {big0.size(), big1.size()}},
      {"sgov_b", {b0.data()}, {b0.size()}},
  };
  auto pr = c.BatchPutSg(puts);
  EXPECT_TRUE(pr[0]) << "sibling A put";
  EXPECT_FALSE(pr[1]) << "oversized put must fail only itself";
  EXPECT_TRUE(pr[2]) << "sibling B put (must not be poisoned by the offender)";

  // The two siblings must have been stored; the oversized key must be absent (never
  // written). Drive the GET-side oversized path too: a get whose total cap exceeds
  // max_payload must miss without poisoning its siblings.
  std::string ga(4096, '\0'), gb(8192, '\0');
  std::string gbig0(200 * 1024, '\0'), gbig1(200 * 1024, '\0');
  std::vector<KvGetItemSg> gets = {
      {"sgov_a", {&ga[0]}, {ga.size()}},
      {"sgov_big", {&gbig0[0], &gbig1[0]}, {gbig0.size(), gbig1.size()}},
      {"sgov_b", {&gb[0]}, {gb.size()}},
  };
  std::vector<size_t> lens;
  auto gr = c.BatchGetAutoSg(gets, &lens);
  EXPECT_TRUE(gr[0]) << "sibling A get";
  EXPECT_FALSE(gr[1]) << "oversized get must miss only itself";
  EXPECT_TRUE(gr[2]) << "sibling B get (must not be poisoned by the offender)";
  if (gr[0]) { EXPECT_EQ(ga, a0); }
  if (gr[2]) { EXPECT_EQ(gb, b0); }

  ::unsetenv("DFKV_RDMA_DEPTH");
}

TEST(RdmaLoopback, PipelinedPoolDepth) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  // depth>1 enables client pipelining + the server GET worker pool (the most
  // concurrency-heavy path). Env is read by both the client ctor and the server
  // serve loop in this single process. Run under TSan to catch races.
  ::setenv("DFKV_RDMA_DEPTH", "4", 1);
  ::setenv("DFKV_RDMA_WORKERS", "4", 1);
  RdmaNode node("ppd");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  const int N = 64;
  const size_t sz = 4096;
  std::vector<std::string> vals(N), keys(N);
  std::vector<KvPutItem> puts(N);
  for (int i = 0; i < N; ++i) {
    keys[i] = "p" + std::to_string(i);
    vals[i].assign(sz, static_cast<char>((i * 7 + 3) & 0xFF));
    puts[i] = {keys[i], vals[i].data(), sz};
  }
  auto pr = c.BatchPut(puts);
  for (int i = 0; i < N; ++i) ASSERT_TRUE(pr[i]) << i;

  std::vector<std::string> outs(N);
  std::vector<KvGetItem> gets(N);
  for (int i = 0; i < N; ++i) { outs[i].assign(sz, '\0'); gets[i] = {keys[i], &outs[i][0], sz}; }
  auto gr = c.BatchGet(gets);
  int hits = 0;
  for (int i = 0; i < N; ++i) { if (gr[i]) { ++hits; EXPECT_EQ(outs[i], vals[i]) << i; } }
  EXPECT_EQ(hits, N);

  ::unsetenv("DFKV_RDMA_DEPTH");
  ::unsetenv("DFKV_RDMA_WORKERS");
}

// Regression for the conn-thread leak (#3). A server Serve thread blocks in
// WaitComp forever after a silent client disconnect (a torn-down RC peer yields
// no completion), so without an idle timeout a long-running server accumulates
// one live thread per lifetime connection — Stop() is the only reaper. The fix:
// an idle timeout reclaims the connection (the thread returns), and ReapDoneLocked
// joins the finished thread on the next accept. This test sets a short idle window
// and verifies the live count drains back to the baseline; without the fix it
// would stay pinned near N and time out.
TEST(RdmaLoopback, ReclaimsAndReapsIdleConnThreads) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ::setenv("DFKV_RDMA_IDLE_MS", "120", 1);  // reclaim idle conns fast (test only)
  constexpr size_t kSmallMsg = 16 * 1024;   // stay well under an 8 MiB memlock
  RdmaNode node("reap", kSmallMsg);          // server buffers must be small too
  // One long-lived client endpoint holds the shared per-device ibv_context open
  // (its server side may be reclaimed when idle and is transparently re-dialed).
  RdmaTransport keep(kSmallMsg);
  KVClient ckeep({{"n", node.addr}}, SelfHdr(), &keep);
  std::string kk(64, 'k');
  ASSERT_TRUE(ckeep.Put("keep", kk.data(), kk.size()));

  const int N = 30;
  for (int i = 0; i < N; ++i) {
    RdmaTransport rt(kSmallMsg);
    KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
    std::string v(512, static_cast<char>(i & 0xFF));
    ASSERT_TRUE(c.Put("k" + std::to_string(i), v.data(), v.size())) << "i=" << i;
    // rt + c leave scope here -> transient client gone -> server conn goes idle.
  }

  // Each round: wait past the idle window so every prior connection's Serve thread
  // has exited, then make ONE connection whose accept runs ReapDoneLocked to join
  // all the finished threads. Only that single in-flight drain thread should then
  // remain. Without reclaim+reap the N transient threads stay in conns_ and the
  // count never falls (the loop exhausts its rounds and the assert fails).
  size_t live = 0;
  for (int round = 0; round < 6; ++round) {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));  // > idle (120 ms)
    { RdmaTransport rt(kSmallMsg); KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
      std::string v(16, 'x'); c.Put("drain", v.data(), v.size()); }
    live = node.rsrv->live_conn_count();
    if (live <= 2) break;
  }
  EXPECT_LE(live, 2u) << "idle conn threads were not reclaimed + reaped (leak)";
  ::unsetenv("DFKV_RDMA_IDLE_MS");
}

// Regression: a BATCH op whose pooled connection the server reclaimed on idle
// must transparently re-dial and succeed — NOT return kIOError for the batch.
// Before the fix only the single-op RoundTrip retried a stale pooled conn; the
// batch paths (CacheMany/CacheFrom/RangeInto/ExistMany/...) gave up on the first
// failed window, which surfaced to SGLang as "Write page to storage: N pages
// failed" on writes and 0-hit prefixes on reads after an idle gap.
TEST(RdmaLoopback, BatchRetriesAfterServerReclaim) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv idle_reclaim("DFKV_RDMA_IDLE_MS", "120");
  RdmaNode node("brc");
  RdmaTransport rt(kMaxMsg);                   // long-lived: holds the device ctx
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  const int N = 24;
  for (int i = 0; i < N; ++i) {
    std::string v = "val" + std::to_string(i);
    ASSERT_TRUE(c.Put("b" + std::to_string(i), v.data(), v.size())) << "warm i=" << i;
  }
  // Warm the pool so a connection is parked for the node, then snapshot the dial
  // counter — the batch op below must open a NEW one when it finds it stale.
  EXPECT_TRUE(c.Exist("b0"));
  long opened_warm =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  ASSERT_GE(opened_warm, 1);

  // Let the server reclaim the now-idle pooled connection (idle window 120 ms).
  std::this_thread::sleep_for(std::chrono::milliseconds(400));

  // The pooled conn is stale (its server peer was reclaimed). A batch existence
  // probe must detect that on the first window and re-dial a fresh conn, returning
  // CORRECT results — not kIOError for the whole batch. Direct transport call so no
  // KVClient-level health retry can mask a transport that failed to recover.
  // Probe with the same canonical namespace/object identity as the warm writes.
  std::vector<BlockKey> keys;
  for (int i = 0; i < N; ++i) {
    keys.push_back(ToBlockKey(SelfHdr(), "b" + std::to_string(i)));
    keys.push_back(ToBlockKey(SelfHdr(), "nope" + std::to_string(i)));
  }
  std::vector<char> exists;
  auto sts = rt.ExistMany(node.addr, keys, &exists);
  ASSERT_EQ(exists.size(), keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    EXPECT_NE(sts[i], Status::kIOError) << "i=" << i << " (stale conn not retried)";
    EXPECT_EQ(exists[i] != 0, (i % 2 == 0)) << "i=" << i;
  }
  // The retry re-dialed: a new client connection was opened after the warm-up.
  long opened_after =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  EXPECT_GT(opened_after, opened_warm) << "batch op did not re-dial after reclaim";

}

// Polls until the server serves no RDMA connection any more: its idle
// reclaim has destroyed its side of every connection the client pooled.
bool ServerHasNoConnections(const RdmaNode& node, int timeout_ms = 5000) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);
  while (node.rsrv->ActiveConns() != 0) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return true;
}

// A peer known by identity, reachable over one rail only, reclaims an idle
// pooled connection. The operation that finds the dead connection must get
// through on a new one over the same rail: the failure says the connection is
// gone, not that the peer is unreachable over the rail. Cooling the rail on
// it refused the retry and every other operation to the peer for the
// cooldown. Static members have no peer identity and never showed this.
TEST(RdmaLoopback, StalePooledConnectionOnTheOnlyRailRecoversOnTheRetry) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv idle_reclaim("DFKV_RDMA_IDLE_MS", "120");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  // The failed completion has to arrive before the window deadline, however
  // long this HCA takes to exhaust its retries.
  ScopedEnv op_timeout("DFKV_RDMA_OP_TIMEOUT_MS", "60000");
  ScopedEnv batch_timeout("DFKV_RDMA_BATCH_OP_TIMEOUT_MS", nullptr);
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  const auto discovered = rdma::RdmaTopology::Discover(
      {}, rdma::RdmaDiscoveryPolicy::kActiveOnly);
  if (discovered.status != rdma::RdmaDiscoveryStatus::kOk ||
      discovered.devices.empty())
    GTEST_SKIP() << "no ACTIVE RDMA device";
  const char* configured = std::getenv("DFKV_RDMA_DEV");
  const std::string rail =
      configured && *configured && std::strchr(configured, ',') == nullptr
          ? configured
          : discovered.devices.front().name;
  ScopedEnv one_rail("DFKV_RDMA_DEV", rail.c_str());
  RdmaNode node("stale-single-rail");
  RdmaTransport rt(kMaxMsg, rail);
  const std::string peer_id = "identified-peer";
  rt.OnPeerTopology(PublishedPeer(node.addr, peer_id));
  const std::vector<BlockKey> keys = {ToBlockKey(SelfHdr(), "absent")};
  std::vector<char> exists;

  // What a served probe for an absent key returns; every later probe must
  // return the same.
  const Status served = rt.ExistMany(node.addr, keys, &exists)[0];
  ASSERT_NE(served, Status::kIOError);
  ASSERT_NE(served, Status::kNoCompatibleRail);
  ASSERT_EQ(
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total"), 1);

  // Wait until the server has destroyed its side of the pooled connection;
  // the client only finds out when it sends.
  ASSERT_TRUE(ServerHasNoConnections(node));
  EXPECT_EQ(rt.ExistMany(node.addr, keys, &exists)[0], served)
      << "the retry on a new connection over the same rail must succeed";

  std::string metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_conns_opened_total"), 2);
  EXPECT_EQ(
      CounterVal(metrics, "dfkv_rdma_client_stale_pool_connections_total"), 1);
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_stale_pool_retries_total"),
            1);
  EXPECT_EQ(
      CounterVal(metrics, "dfkv_rdma_client_remote_rail_cooldowns_total"), 0)
      << "a dead pooled connection is no evidence against the peer's rail";
  EXPECT_EQ(
      CounterVal(metrics, "dfkv_rdma_client_remote_rail_failures_total"), 0);
  EXPECT_EQ(RdmaTransportTestPeer::RemoteAllowed(
                rt, peer_id, {1}, rdma::RailPolicy::NowMicros()),
            (std::vector<uint8_t>{1}));

  // The peer stays usable for the operations that follow.
  EXPECT_EQ(rt.ExistMany(node.addr, keys, &exists)[0], served);
  metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_conns_opened_total"), 2);
  EXPECT_EQ(
      CounterVal(metrics, "dfkv_rdma_client_remote_rail_admissions_denied_total"),
      0);
}

// The other half of the rule above: when the connection made for the retry
// fails too, the failure is about the peer and not about one old connection,
// and the peer rail is cooled, once. Here the server has reclaimed the pooled
// connection and then refuses new ones, being at its limit of connections
// that have not finished their handshake.
TEST(RdmaLoopback, FreshConnectionFailingAfterAStaleOneCoolsThePeerRail) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  if (SoftRoceOnly()) GTEST_SKIP() << kRxeNoRetryExceeded;
  ScopedEnv idle_reclaim("DFKV_RDMA_IDLE_MS", "120");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv op_timeout("DFKV_RDMA_OP_TIMEOUT_MS", "60000");
  ScopedEnv batch_timeout("DFKV_RDMA_BATCH_OP_TIMEOUT_MS", nullptr);
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  const auto discovered = rdma::RdmaTopology::Discover(
      {}, rdma::RdmaDiscoveryPolicy::kActiveOnly);
  if (discovered.status != rdma::RdmaDiscoveryStatus::kOk ||
      discovered.devices.empty())
    GTEST_SKIP() << "no ACTIVE RDMA device";
  const char* configured = std::getenv("DFKV_RDMA_DEV");
  const std::string rail =
      configured && *configured && std::strchr(configured, ',') == nullptr
          ? configured
          : discovered.devices.front().name;
  ScopedEnv one_rail("DFKV_RDMA_DEV", rail.c_str());
  RdmaNode node("stale-then-refused");
  // One handshake at a time is enough for this client and lets a single
  // silent peer close the door later.
  RdmaServerTestPeer::SetMaxPendingHandshakes(node.rsrv.get(), 1);
  RdmaTransport rt(kMaxMsg, rail);
  const std::string peer_id = "identified-peer";
  rt.OnPeerTopology(PublishedPeer(node.addr, peer_id));
  const std::vector<BlockKey> keys = {ToBlockKey(SelfHdr(), "absent")};
  std::vector<char> exists;

  const Status served = rt.ExistMany(node.addr, keys, &exists)[0];
  ASSERT_NE(served, Status::kIOError);
  ASSERT_NE(served, Status::kNoCompatibleRail);
  ASSERT_TRUE(ServerHasNoConnections(node));

  // From now on the server refuses every new bootstrap connection: one peer
  // that connects and sends nothing holds the only handshake slot.
  const int silent = net::Dial(node.addr, 2000, 2000);
  ASSERT_GE(silent, 0);
  for (int i = 0; i < 1000 && node.rsrv->PendingHandshakes() != 1; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  ASSERT_EQ(node.rsrv->PendingHandshakes(), 1u);

  EXPECT_EQ(rt.ExistMany(node.addr, keys, &exists)[0], Status::kIOError);
  ::close(silent);

  const std::string metrics = rt.MetricsText();
  EXPECT_EQ(
      CounterVal(metrics, "dfkv_rdma_client_stale_pool_connections_total"), 1);
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_stale_pool_retries_total"),
            1);
  EXPECT_EQ(
      CounterVal(metrics, "dfkv_rdma_client_remote_rail_failures_total"), 1)
      << "the stale connection is not charged, the fresh one is";
  EXPECT_EQ(
      CounterVal(metrics, "dfkv_rdma_client_remote_rail_cooldowns_total"), 1);
  EXPECT_EQ(RdmaTransportTestPeer::RemoteAllowed(
                rt, peer_id, {1}, rdma::RailPolicy::NowMicros()),
            (std::vector<uint8_t>{0}));
}

// The one ACTIVE device these tests pin the client to: DFKV_RDMA_DEV when it
// names a single device, the first discovered one otherwise. Empty when the
// host has none.
std::string SingleTestRail() {
  const auto discovered = rdma::RdmaTopology::Discover(
      {}, rdma::RdmaDiscoveryPolicy::kActiveOnly);
  if (discovered.status != rdma::RdmaDiscoveryStatus::kOk ||
      discovered.devices.empty())
    return std::string();
  const char* configured = std::getenv("DFKV_RDMA_DEV");
  return configured && *configured && std::strchr(configured, ',') == nullptr
             ? configured
             : discovered.devices.front().name;
}

// Settings for the tests below: the server reclaims a connection after
// 120 ms without traffic, the client sends no keepalive that would prevent
// it, and the failed completion on the dead connection arrives before any
// window deadline, however long this HCA takes to exhaust its retries.
struct ReclaimingServerEnv {
  ScopedEnv idle_reclaim{"DFKV_RDMA_IDLE_MS", "120"};
  ScopedEnv keepalive{"DFKV_RDMA_KEEPALIVE_MS", "0"};
  ScopedEnv op_timeout{"DFKV_RDMA_OP_TIMEOUT_MS", "60000"};
  ScopedEnv batch_timeout{"DFKV_RDMA_BATCH_OP_TIMEOUT_MS", nullptr};
  ScopedEnv rail_tiers{"DFKV_RDMA_RAIL_TIERS", nullptr};
};

// A PUT posted on a pooled connection that the server has reclaimed gets no
// acknowledgement and fails with retry exceeded. Nothing reached a server
// that could store it, so it is sent once more on a new connection, and the
// server sees each PUT exactly once.
TEST(RdmaLoopback, PutsOnAConnectionTheServerReclaimedAreSentOnANewOne) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ReclaimingServerEnv env;
  const std::string rail = SingleTestRail();
  if (rail.empty()) GTEST_SKIP() << "no ACTIVE RDMA device";
  ScopedEnv one_rail("DFKV_RDMA_DEV", rail.c_str());
  RdmaNode node("stale-put");
  RdmaTransport rt(kMaxMsg, rail);
  rt.OnPeerTopology(PublishedPeer(node.addr, "stale-put-peer"));
  const std::string value = PatternValue(64 * 1024 + 7, 3);
  const auto key = [](const std::string& name) {
    return ToBlockKey(SelfHdr(), name);
  };
  ASSERT_EQ(rt.Cache(node.addr, key("warm"), value.data(), value.size()),
            Status::kOk);

  // One PUT.
  ASSERT_TRUE(ServerHasNoConnections(node));
  const auto started = std::chrono::steady_clock::now();
  EXPECT_EQ(rt.Cache(node.addr, key("single"), value.data(), value.size()),
            Status::kOk);
  std::cout << "[ info ] PUT on the reclaimed connection and its retry took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - started)
                   .count()
            << " ms\n";
  EXPECT_EQ(node.CacheDirectCalls(key("single")), 1u);

  // A batch of PUTs.
  ASSERT_TRUE(ServerHasNoConnections(node));
  const std::vector<CacheItem> batch = {
      {key("batch-0"), value.data(), value.size()},
      {key("batch-1"), value.data(), value.size()},
      {key("batch-2"), value.data(), value.size()}};
  EXPECT_EQ(rt.CacheMany(node.addr, batch),
            std::vector<Status>(batch.size(), Status::kOk));
  for (const auto& item : batch)
    EXPECT_EQ(node.CacheDirectCalls(item.key), 1u);

  // A scatter-gather PUT, after a first one that pools its connection.
  const size_t half = value.size() / 2;
  const auto source = [&](const std::string& name) {
    return CacheSrcMulti{key(name),
                         {{value.data(), half},
                          {value.data() + half, value.size() - half}}};
  };
  ASSERT_EQ(rt.CacheFromMulti(node.addr, {source("sg-warm")}),
            std::vector<Status>{Status::kOk});
  ASSERT_TRUE(ServerHasNoConnections(node));
  EXPECT_EQ(rt.CacheFromMulti(node.addr, {source("sg")}),
            std::vector<Status>{Status::kOk});
  EXPECT_EQ(node.CacheDirectCalls(key("sg")), 1u);

  const std::string metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_stale_pool_retries_total"),
            3);
  EXPECT_EQ(
      CounterVal(metrics, "dfkv_rdma_client_remote_rail_cooldowns_total"), 0);
  for (const char* name : {"single", "batch-0", "batch-1", "batch-2", "sg"}) {
    std::string out;
    EXPECT_EQ(rt.Range(node.addr, key(name), 0, value.size(), &out),
              Status::kOk)
        << name;
    EXPECT_EQ(out, value) << name;
  }
}

// A read into the caller's memory that finds a pooled connection the server
// has reclaimed is made again on a new connection, for one destination and
// for a scatter-gather one.
TEST(RdmaLoopback, DirectReadsOnAConnectionTheServerReclaimedSucceed) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ReclaimingServerEnv env;
  const std::string rail = SingleTestRail();
  if (rail.empty()) GTEST_SKIP() << "no ACTIVE RDMA device";
  ScopedEnv one_rail("DFKV_RDMA_DEV", rail.c_str());
  RdmaNode node("stale-direct-read");
  RdmaTransport rt(kMaxMsg, rail);
  rt.OnPeerTopology(PublishedPeer(node.addr, "stale-direct-read-peer"));
  const std::string value = PatternValue(64 * 1024 + 7, 5);
  const BlockKey key = ToBlockKey(SelfHdr(), "direct");
  ASSERT_EQ(rt.Cache(node.addr, key, value.data(), value.size()),
            Status::kOk);

  ASSERT_TRUE(ServerHasNoConnections(node));
  std::string out(value.size(), '\0');
  std::vector<uint64_t> value_lens;
  EXPECT_EQ(rt.RangeInto(node.addr, {key}, {{out.data(), out.size()}},
                         &value_lens),
            std::vector<Status>{Status::kOk});
  EXPECT_EQ(value_lens, std::vector<uint64_t>{value.size()});
  EXPECT_EQ(out, value);

  // The scatter-gather lane has its own pool: read once to pool a
  // connection there, then again after the server has reclaimed it.
  const size_t half = value.size() / 2;
  std::string first(half, '\0'), second(value.size() - half, '\0');
  const auto read_sg = [&] {
    first.assign(half, '\0');
    second.assign(value.size() - half, '\0');
    std::vector<size_t> lengths;
    const auto statuses = rt.RangeIntoMulti(
        node.addr, {key},
        {RangeDstMulti{{{first.data(), first.size()},
                        {second.data(), second.size()}}}},
        &lengths);
    return statuses.size() == 1 && statuses[0] == Status::kOk &&
           lengths == std::vector<size_t>{value.size()} &&
           first + second == value;
  };
  ASSERT_TRUE(read_sg());
  ASSERT_TRUE(ServerHasNoConnections(node));
  EXPECT_TRUE(read_sg());

  const std::string metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_stale_pool_retries_total"),
            2);
  EXPECT_EQ(
      CounterVal(metrics, "dfkv_rdma_client_remote_rail_cooldowns_total"), 0);
}

// The retry after a stale pooled connection gets the whole timeout again.
// With the QP's transport retry slowed down, no retry-exceeded completion
// comes before the operation's timeout (as on Soft-RoCE), so the first
// attempt's wait uses all of it. A PUT and a whole-object read on a
// connection the server reclaimed are still made again on a new one and
// succeed, in about one timeout plus the retry.
TEST(RdmaLoopback, AStalePoolRetryGetsTheWholeTimeoutAgain) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ReclaimingServerEnv env;
  // Each test runs in its own process: read when the first QP is set up.
  ScopedEnv qp_timeout("DFKV_RDMA_QP_TIMEOUT", "20");  // ~4 s per try
  ScopedEnv op_timeout("DFKV_RDMA_OP_TIMEOUT_MS", "2000");
  const std::string rail = SingleTestRail();
  if (rail.empty()) GTEST_SKIP() << "no ACTIVE RDMA device";
  ScopedEnv one_rail("DFKV_RDMA_DEV", rail.c_str());
  RdmaNode node("stale-full-timeout");
  RdmaTransport rt(kMaxMsg, rail);
  rt.OnPeerTopology(PublishedPeer(node.addr, "stale-full-timeout-peer"));
  const std::string value = PatternValue(64 * 1024 + 7, 9);
  const auto key = [](const std::string& name) {
    return ToBlockKey(SelfHdr(), name);
  };
  ASSERT_EQ(rt.Cache(node.addr, key("warm"), value.data(), value.size()),
            Status::kOk);

  ASSERT_TRUE(ServerHasNoConnections(node));
  auto started = std::chrono::steady_clock::now();
  EXPECT_EQ(rt.Cache(node.addr, key("put"), value.data(), value.size()),
            Status::kOk);
  auto took = std::chrono::steady_clock::now() - started;
  EXPECT_GE(took, std::chrono::milliseconds(1900)) << "the first wait ended early";
  EXPECT_EQ(node.CacheDirectCalls(key("put")), 1u);

  ASSERT_TRUE(ServerHasNoConnections(node));
  std::string out(value.size(), '\0');
  started = std::chrono::steady_clock::now();
  std::vector<uint64_t> value_lens;
  EXPECT_EQ(rt.RangeInto(node.addr, {key("put")}, {{out.data(), out.size()}},
                         &value_lens),
            std::vector<Status>{Status::kOk});
  took = std::chrono::steady_clock::now() - started;
  EXPECT_GE(took, std::chrono::milliseconds(1900)) << "the first wait ended early";
  EXPECT_EQ(out, value);

  EXPECT_EQ(CounterVal(rt.MetricsText(),
                       "dfkv_rdma_client_stale_pool_retries_total"),
            2);
}

// The other half of the rule above: a PUT posted on a pooled connection on
// which something has succeeded since it left the pool is not sent again,
// whatever fails afterwards. Here every completion of the request arrives
// and the server stores the value; the failure is injected after that, as a
// remote error completion. The single PUT is the first request on its
// reused connection; in the batch the first window succeeds and the second
// fails.
TEST(RdmaLoopback, PutPostedOnAProvenPooledConnectionIsNotSentAgain) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");
  ScopedEnv local_fault("DFKV_RDMA_TEST_COMPLETION_FAULT", nullptr);
  ScopedEnv endpoint_fault("DFKV_RDMA_TEST_ENDPOINT_COMPLETION_FAULT",
                           nullptr);
  const std::string rail = SingleTestRail();
  if (rail.empty()) GTEST_SKIP() << "no ACTIVE RDMA device";
  ScopedEnv one_rail("DFKV_RDMA_DEV", rail.c_str());
  RdmaNode node("proven-put");
  RdmaTransport rt(kMaxMsg, rail);
  const std::string value = PatternValue(16 * 1024 + 3, 9);
  const auto key = [](const std::string& name) {
    return ToBlockKey(SelfHdr(), name);
  };
  const auto items = [&](const std::string& prefix) {
    std::vector<CacheItem> batch;
    for (int i = 0; i < 8; ++i)
      batch.push_back({key(prefix + std::to_string(i)), value.data(),
                       value.size()});
    return batch;
  };
  const auto counter = [&](const std::string& name) {
    return CounterVal(rt.MetricsText(), name);
  };

  ASSERT_EQ(rt.Cache(node.addr, key("warm"), value.data(), value.size()),
            Status::kOk);
  long hits = counter("dfkv_rdma_endpoint_cache_hits_total");
  long writes = counter("dfkv_rdma_client_v2_put_writes_total");
  {
    ScopedEnv fault("DFKV_RDMA_TEST_ENDPOINT_COMPLETION_FAULT", "1:1:1");
    EXPECT_EQ(rt.Cache(node.addr, key("single"), value.data(), value.size()),
              Status::kIOError);
  }
  EXPECT_EQ(counter("dfkv_rdma_endpoint_cache_hits_total"), hits + 1)
      << "the PUT ran on the pooled connection";
  EXPECT_EQ(counter("dfkv_rdma_client_v2_put_writes_total"), writes + 1)
      << "the PUT was not posted again";
  EXPECT_EQ(node.CacheDirectCalls(key("single")), 1u);

  ASSERT_EQ(rt.CacheMany(node.addr, items("warm-batch-")),
            std::vector<Status>(8, Status::kOk));
  const auto batch = items("batch-");
  hits = counter("dfkv_rdma_endpoint_cache_hits_total");
  writes = counter("dfkv_rdma_client_v2_put_writes_total");
  std::vector<Status> statuses;
  {
    ScopedEnv fault("DFKV_RDMA_TEST_ENDPOINT_COMPLETION_FAULT", "2:1:2");
    statuses = rt.CacheMany(node.addr, batch);
  }
  std::vector<Status> expected(4, Status::kOk);
  expected.resize(8, Status::kIOError);
  EXPECT_EQ(statuses, expected);
  EXPECT_EQ(counter("dfkv_rdma_endpoint_cache_hits_total"), hits + 1);
  EXPECT_EQ(counter("dfkv_rdma_client_v2_put_writes_total"), writes + 8)
      << "no item of the batch was posted twice";
  for (const auto& item : batch) EXPECT_EQ(node.CacheDirectCalls(item.key), 1u);
  EXPECT_EQ(counter("dfkv_rdma_client_stale_pool_retries_total"), 0);
  EXPECT_EQ(counter("dfkv_rdma_client_stale_pool_connections_total"), 0);
}

// Polls until the counter has reached at least `at_least`.
bool WaitForCounter(const RdmaTransport& rt, const std::string& name,
                    long at_least, int timeout_ms = 5000) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);
  while (CounterVal(rt.MetricsText(), name) < at_least) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  return true;
}

// A keepalive that is not answered within its short limit (100 ms; an
// operation waits 5 s) says that this idle connection did not answer in
// time, here because the server is busy. The connection is dropped. The
// peer rail is not cooled: with one usable rail that refused every operation
// to the peer for the cooldown, although the peer was serving.
TEST(RdmaLoopback, LateKeepaliveAnswerDropsTheConnectionNotThePeerRail) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "200");
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  const std::string rail = SingleTestRail();
  if (rail.empty()) GTEST_SKIP() << "no ACTIVE RDMA device";
  ScopedEnv one_rail("DFKV_RDMA_DEV", rail.c_str());
  RdmaNode node("late-keepalive");
  RdmaTransport rt(kMaxMsg, rail);
  const std::string peer_id = "late-keepalive-peer";
  rt.OnPeerTopology(PublishedPeer(node.addr, peer_id));
  const std::vector<BlockKey> keys = {ToBlockKey(SelfHdr(), "absent")};
  std::vector<char> exists;
  const Status served = rt.ExistMany(node.addr, keys, &exists)[0];
  ASSERT_NE(served, Status::kIOError);
  ASSERT_NE(served, Status::kNoCompatibleRail);

  node.handler_delay_ms = 300;
  ASSERT_TRUE(WaitForCounter(rt, "dfkv_rdma_client_keepalive_failures_total",
                             1));
  node.handler_delay_ms = 0;

  EXPECT_EQ(
      CounterVal(rt.MetricsText(),
                 "dfkv_rdma_client_remote_rail_cooldowns_total"),
      0);
  EXPECT_EQ(RdmaTransportTestPeer::RemoteAllowed(
                rt, peer_id, {1}, rdma::RailPolicy::NowMicros()),
            (std::vector<uint8_t>{1}));
  EXPECT_EQ(rt.ExistMany(node.addr, keys, &exists)[0], served);
}

// The same for a keepalive on a connection the server has reclaimed: the
// connection is gone, the peer rail is not to blame.
TEST(RdmaLoopback, KeepaliveOnAConnectionTheServerReclaimedDoesNotCoolTheRail) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv idle_reclaim("DFKV_RDMA_IDLE_MS", "120");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "400");
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  const std::string rail = SingleTestRail();
  if (rail.empty()) GTEST_SKIP() << "no ACTIVE RDMA device";
  ScopedEnv one_rail("DFKV_RDMA_DEV", rail.c_str());
  RdmaNode node("reclaimed-keepalive");
  RdmaTransport rt(kMaxMsg, rail);
  const std::string peer_id = "reclaimed-keepalive-peer";
  rt.OnPeerTopology(PublishedPeer(node.addr, peer_id));
  const std::vector<BlockKey> keys = {ToBlockKey(SelfHdr(), "absent")};
  std::vector<char> exists;
  const Status served = rt.ExistMany(node.addr, keys, &exists)[0];
  ASSERT_NE(served, Status::kIOError);
  ASSERT_NE(served, Status::kNoCompatibleRail);

  ASSERT_TRUE(ServerHasNoConnections(node));
  ASSERT_TRUE(WaitForCounter(rt, "dfkv_rdma_client_keepalive_failures_total",
                             1));
  EXPECT_EQ(
      CounterVal(rt.MetricsText(),
                 "dfkv_rdma_client_remote_rail_cooldowns_total"),
      0);
  EXPECT_EQ(rt.ExistMany(node.addr, keys, &exists)[0], served);
}

// A keepalive round takes an idle connection out of its pool only for its
// own probe. While the round waits for the answer of a busy node, the
// pooled connections to other nodes stay in the pool, and an operation on
// one of them uses it instead of dialing a new connection.
TEST(RdmaLoopback, KeepaliveWaitingOnOneNodeLeavesOtherConnectionsPooled) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  // Rounds a second apart: each starts well after the setup before it.
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "1000");
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  const std::string rail = SingleTestRail();
  if (rail.empty()) GTEST_SKIP() << "no ACTIVE RDMA device";
  ScopedEnv one_rail("DFKV_RDMA_DEV", rail.c_str());
  RdmaNode fast("keepalive-fast");
  RdmaNode slow("keepalive-slow");
  RdmaTransport rt(kMaxMsg, rail);
  const BlockKey key = ToBlockKey(SelfHdr(), "keepalive-round");
  const std::string value(4096, 'r');
  const auto counter = [&](const std::string& name) {
    return CounterVal(rt.MetricsText(), name);
  };
  // A round probes data connections before control connections: the slow
  // node gets a data connection, the fast node only a control connection.
  // The slow node answers its keepalives late (its PUTs are not delayed).
  bool exist = false;
  ASSERT_EQ(rt.Exist(fast.addr, key, &exist), Status::kOk);
  slow.handler_delay_ms = 300;

  // The operation on the fast node must run while the round waits on the
  // slow node's probe (100 ms). On a loaded host it may miss that window;
  // only an observation made inside it counts, and a missed one is retried
  // in the next round.
  bool observed = false;
  for (int round = 0; round < 5 && !observed; ++round) {
    ASSERT_EQ(rt.Cache(slow.addr, key, value.data(), value.size()),
              Status::kOk);
    const long attempts = counter("dfkv_rdma_client_keepalive_attempts_total");
    const long failures = counter("dfkv_rdma_client_keepalive_failures_total");
    const long successes =
        counter("dfkv_rdma_client_keepalive_successes_total");
    const long opened = counter("dfkv_rdma_client_conns_opened_total");

    ASSERT_TRUE(WaitForCounter(
        rt, "dfkv_rdma_client_keepalive_attempts_total", attempts + 1));
    ASSERT_EQ(rt.Exist(fast.addr, key, &exist), Status::kOk);
    const std::string metrics = rt.MetricsText();
    // Still the slow node's probe: the round has not got further.
    observed =
        CounterVal(metrics, "dfkv_rdma_client_keepalive_attempts_total") ==
            attempts + 1 &&
        CounterVal(metrics, "dfkv_rdma_client_keepalive_failures_total") ==
            failures;
    if (observed)
      EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_conns_opened_total"),
                opened)
          << "the fast node's pooled connection was out of the pool";

    // Let the round finish: the slow probe fails, the fast one answers.
    ASSERT_TRUE(WaitForCounter(
        rt, "dfkv_rdma_client_keepalive_failures_total", failures + 1));
    ASSERT_TRUE(WaitForCounter(
        rt, "dfkv_rdma_client_keepalive_successes_total", successes + 1));
  }
  if (!observed)
    GTEST_SKIP() << "no operation ran inside a slow keepalive probe";
}

// A TCP connect that fails says the node is not there. It says nothing about
// a rail: every rail dials the same address. So no peer rail is charged and
// nothing is dialed a second time; the caller gets kIOError, which is what
// cools the node in the client above the transport.
TEST(RdmaLoopback, UnreachablePeerIsANodeFailureNotARailFailure) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  const std::string rail = SingleTestRail();
  if (rail.empty()) GTEST_SKIP() << "no ACTIVE RDMA device";
  ScopedEnv one_rail("DFKV_RDMA_DEV", rail.c_str());

  // Bound, never listening: a connect to it is refused at once.
  const int dead = ::socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(dead, 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  socklen_t length = sizeof(address);
  ASSERT_EQ(::bind(dead, reinterpret_cast<sockaddr*>(&address),
                   sizeof(address)),
            0);
  ASSERT_EQ(
      ::getsockname(dead, reinterpret_cast<sockaddr*>(&address), &length), 0);
  const std::string dead_addr =
      "127.0.0.1:" + std::to_string(ntohs(address.sin_port));

  RdmaTransport rt(kMaxMsg, rail);
  // Long enough that the second operation below falls inside it.
  RdmaTransportTestPeer::SetUnreachableHoldUs(&rt, 60ull * 1000 * 1000);
  const std::string peer_id = "absent-peer";
  rt.OnPeerTopology(PublishedPeer(dead_addr, peer_id));
  const std::vector<BlockKey> keys = {ToBlockKey(SelfHdr(), "absent")};
  std::vector<char> exists;
  EXPECT_EQ(rt.ExistMany(dead_addr, keys, &exists)[0], Status::kIOError);
  // That was a connect that failed, not a refusal.
  EXPECT_FALSE(rt.TakeUnreachableRefusal());
  // While the address is held as unreachable it is not dialed again: the
  // next operation, or the client's background probe of the node, fails at
  // once and does not wait for another connect.
  EXPECT_EQ(rt.ExistMany(dead_addr, keys, &exists)[0], Status::kIOError);
  // The same status, but nothing was tried: the caller can tell, once.
  EXPECT_TRUE(rt.TakeUnreachableRefusal());
  EXPECT_FALSE(rt.TakeUnreachableRefusal());

  std::string metrics = rt.MetricsText();
  EXPECT_EQ(
      CounterVal(metrics, "dfkv_rdma_client_peer_unreachable_skips_total"), 1);
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_peer_unreachable_total"), 1)
      << "one connect, not one per rail or per attempt";
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_v2_probe_attempts_total"),
            1);
  EXPECT_EQ(
      CounterVal(metrics, "dfkv_rdma_client_remote_rail_failures_total"), 0);
  EXPECT_EQ(
      CounterVal(metrics, "dfkv_rdma_client_remote_rail_cooldowns_total"), 0);
  EXPECT_EQ(RdmaTransportTestPeer::RemoteAllowed(
                rt, peer_id, {1}, rdma::RailPolicy::NowMicros()),
            (std::vector<uint8_t>{1}));

  // Once the hold is over the address is dialed again.
  RdmaTransport dialing(kMaxMsg, rail);
  RdmaTransportTestPeer::SetUnreachableHoldUs(&dialing, 0);
  dialing.OnPeerTopology(PublishedPeer(dead_addr, peer_id));
  EXPECT_EQ(dialing.ExistMany(dead_addr, keys, &exists)[0], Status::kIOError);
  EXPECT_EQ(dialing.ExistMany(dead_addr, keys, &exists)[0], Status::kIOError);
  EXPECT_FALSE(dialing.TakeUnreachableRefusal());
  metrics = dialing.MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_peer_unreachable_total"), 2);
  EXPECT_EQ(
      CounterVal(metrics, "dfkv_rdma_client_peer_unreachable_skips_total"), 0);
  ::close(dead);
}

// A capability probe gives its pending-handshake slot back before it
// answers: a client that reads the answer and connects at once must not
// find the slot still taken by its own probe, at a limit of one. It was
// given back only when the probe's thread returned, after the answer, so
// under load the client's bootstrap could be refused by its own probe
// (BootstrapRefusedByABusyPeerIsRetriedNotCharged failed 2 of 200 times).
TEST(RdmaLoopback, AProbeFreesItsHandshakeSlotBeforeItAnswers) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("probe-slot");
  RdmaServerTestPeer::SetMaxPendingHandshakes(node.rsrv.get(), 1);
  std::atomic<bool> hold{true};
  RdmaServerTestPeer::SetAfterProbeReply(node.rsrv.get(), [&hold] {
    while (hold.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  });
  struct Release {
    std::atomic<bool>* hold;
    ~Release() { hold->store(false); }
  } release{&hold};
  const auto probe = [&](char reply[rdma::kV2ProbeReplyBytes]) {
    const int fd = net::Dial(node.addr, 2000, 2000);
    if (fd < 0) return false;
    char frame[rdma::kDevNameBytes];
    rdma::EncodeDevFrame(rdma::kV2ProbeDevice, kMaxMsg, frame,
                         rdma::kDevProtoV2);
    const bool ok = net::WriteAll(fd, frame, sizeof(frame)) &&
                    net::ReadAll(fd, reply, rdma::kV2ProbeReplyBytes);
    ::close(fd);
    return ok;
  };
  char reply[rdma::kV2ProbeReplyBytes];
  ASSERT_TRUE(probe(reply));  // its thread now waits after answering
  EXPECT_TRUE(probe(reply)) << "the next connection was refused while the "
                               "first probe's thread had not returned";
  EXPECT_EQ(node.rsrv->RejectedConnections(), 0u);
}

// A server at its limit of pending handshakes closes a new bootstrap
// connection before sending anything. That is "busy", not a rail fault: the
// client waits a moment and tries the same rail again, and gets through once
// a slot is free.
TEST(RdmaLoopback, BootstrapRefusedByABusyPeerIsRetriedNotCharged) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  const std::string rail = SingleTestRail();
  if (rail.empty()) GTEST_SKIP() << "no ACTIVE RDMA device";
  ScopedEnv one_rail("DFKV_RDMA_DEV", rail.c_str());
  RdmaNode node("busy-then-served");
  RdmaServerTestPeer::SetMaxPendingHandshakes(node.rsrv.get(), 1);
  RdmaTransport rt(kMaxMsg, rail);
  // The slot has to be free again before the retry: give the releasing
  // thread and the server far more time than they need.
  RdmaTransportTestPeer::SetPeerBusyRetryUs(&rt, 500 * 1000);
  const std::string peer_id = "busy-peer";
  rt.OnPeerTopology(PublishedPeer(node.addr, peer_id));

  // A peer that connects and sends nothing takes the only slot.
  const int silent = net::Dial(node.addr, 2000, 2000);
  ASSERT_GE(silent, 0);
  for (int i = 0; i < 500 && node.rsrv->PendingHandshakes() != 1; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  ASSERT_EQ(node.rsrv->PendingHandshakes(), 1u);
  // It lets go as soon as the server has refused the client once.
  std::thread releasing([&] {
    while (node.rsrv->RejectedConnections() == 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ::close(silent);
  });
  // (The server notices the close and frees the slot within the pause.)

  const std::vector<BlockKey> keys = {ToBlockKey(SelfHdr(), "absent")};
  std::vector<char> exists;
  const Status status = rt.ExistMany(node.addr, keys, &exists)[0];
  releasing.join();
  EXPECT_NE(status, Status::kIOError);
  EXPECT_NE(status, Status::kNoCompatibleRail);

  const std::string metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_peer_busy_total"), 1);
  EXPECT_EQ(node.rsrv->RejectedConnections(), 1u)
      << "one refused connection, then one that was served";
  EXPECT_EQ(
      CounterVal(metrics, "dfkv_rdma_client_remote_rail_failures_total"), 0);
  EXPECT_EQ(
      CounterVal(metrics, "dfkv_rdma_client_remote_rail_cooldowns_total"), 0);
}

// A server that does not serve the device the client asks for closes the
// bootstrap connection before replying, exactly like a server that is full.
// The retry therefore goes to another rail when the client has one: the
// operation succeeds there, and no operation fails on the way.
TEST(RdmaLoopback, BootstrapRefusedForOneDeviceIsRetriedOnAnotherRail) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const auto discovered = rdma::RdmaTopology::Discover(
      {}, rdma::RdmaDiscoveryPolicy::kActiveOnly);
  if (discovered.status != rdma::RdmaDiscoveryStatus::kOk ||
      discovered.devices.size() < 2)
    GTEST_SKIP() << "needs two ACTIVE RDMA devices";
  const std::string served_rail = discovered.devices[0].name;
  const std::string refused_rail = discovered.devices[1].name;
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv numa("DFKV_RDMA_NUMA", "0");

  // The server has one of the two devices only.
  std::unique_ptr<RdmaNode> node;
  {
    ScopedEnv server_rail("DFKV_RDMA_DEV", served_rail.c_str());
    node = std::make_unique<RdmaNode>("one-device-server");
  }
  ASSERT_EQ(node->rsrv->DeviceNames(),
            (std::vector<std::string>{served_rail}));

  // The client has both and starts with the one the server lacks.
  ScopedEnv primary("DFKV_RDMA_PRIMARY_DEV", refused_rail.c_str());
  RdmaTransport rt(kMaxMsg, refused_rail + "," + served_rail);
  const std::vector<BlockKey> keys = {ToBlockKey(SelfHdr(), "absent")};
  std::vector<char> exists;
  const Status status = rt.ExistMany(node->addr, keys, &exists)[0];
  EXPECT_NE(status, Status::kIOError);
  EXPECT_NE(status, Status::kNoCompatibleRail);
  EXPECT_NE(status, Status::kResourceExhausted);

  const std::string metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_peer_busy_total"), 1)
      << "the first attempt went to the device the server lacks";
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_cross_rail_retries_total"),
            1);
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_conns_opened_total"), 1);
  EXPECT_EQ(DeviceCounterVal(metrics, "dfkv_rdma_client_rail_conns_total",
                             served_rail),
            1);
}

// Refused again on the retry: now it is charged to the peer rail, and the
// client has made two connections in all, not one per rail and attempt.
TEST(RdmaLoopback, BootstrapRefusedTwiceChargesThePeerRail) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  const std::string rail = SingleTestRail();
  if (rail.empty()) GTEST_SKIP() << "no ACTIVE RDMA device";
  ScopedEnv one_rail("DFKV_RDMA_DEV", rail.c_str());
  RdmaNode node("busy-twice");
  RdmaServerTestPeer::SetMaxPendingHandshakes(node.rsrv.get(), 1);
  RdmaTransport rt(kMaxMsg, rail);
  const std::string peer_id = "refusing-peer";
  rt.OnPeerTopology(PublishedPeer(node.addr, peer_id));

  const int silent = net::Dial(node.addr, 2000, 2000);
  ASSERT_GE(silent, 0);
  for (int i = 0; i < 500 && node.rsrv->PendingHandshakes() != 1; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  ASSERT_EQ(node.rsrv->PendingHandshakes(), 1u);

  const std::vector<BlockKey> keys = {ToBlockKey(SelfHdr(), "absent")};
  std::vector<char> exists;
  EXPECT_EQ(rt.ExistMany(node.addr, keys, &exists)[0], Status::kIOError);
  ::close(silent);

  const std::string metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_peer_busy_total"), 1);
  EXPECT_EQ(node.rsrv->RejectedConnections(), 2u);
  EXPECT_EQ(
      CounterVal(metrics, "dfkv_rdma_client_remote_rail_failures_total"), 1);
  EXPECT_EQ(
      CounterVal(metrics, "dfkv_rdma_client_remote_rail_cooldowns_total"), 1);
}

// A caller buffer that cannot be registered is a problem of this process.
// The rail and the peer had no part in it: neither is charged, no other rail
// is tried, the caller is told it was a local resource, and the next
// operation with a usable buffer goes through.
TEST(RdmaLoopback, UnregistrableCallerBufferIsALocalFailure) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv inline_put("DFKV_RDMA_INLINE_PUT_MAX_BYTES", nullptr);
  RdmaNode node("local-register");
  RdmaTransport rt(kMaxMsg);
  const std::string peer_id = "healthy-peer";
  rt.OnPeerTopology(PublishedPeer(node.addr, peer_id));

  // Mapped without any access: the kernel refuses to pin it.
  const size_t bytes = 2 * 4096;
  void* unpinnable = ::mmap(nullptr, bytes, PROT_NONE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(unpinnable, MAP_FAILED);
  const std::string good(bytes, 'g');
  const BlockKey key = ToBlockKey(SelfHdr(), "local-register");

  for (const bool batch : {false, true}) {
    const long before = CounterVal(
        rt.MetricsText(), "dfkv_rdma_client_local_resource_failures_total");
    const Status status =
        batch ? rt.CacheMany(node.addr, {{key, unpinnable, bytes}})[0]
              : rt.Cache(node.addr, key, unpinnable, bytes);
    EXPECT_EQ(status, Status::kResourceExhausted) << "batch=" << batch;
    const std::string metrics = rt.MetricsText();
    EXPECT_EQ(
        CounterVal(metrics, "dfkv_rdma_client_local_resource_failures_total"),
        before + 1)
        << "batch=" << batch;
    EXPECT_TRUE(LabeledSeriesAllZero(metrics,
                                     "dfkv_rdma_client_rail_errors_total"))
        << metrics;
    EXPECT_EQ(
        CounterVal(metrics, "dfkv_rdma_client_remote_rail_failures_total"), 0);
    EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_cross_rail_retries_total"),
              0);
    EXPECT_EQ(rt.Cache(node.addr, key, good.data(), good.size()), Status::kOk)
        << "batch=" << batch;
  }
  ::munmap(unpinnable, bytes);
}

// Out of descriptors while creating an endpoint: again this process, not the
// device. The limit is set for real, in a forked child, so that one more
// descriptor can be had (the bootstrap socket) and the next one cannot (the
// endpoint's wake pipe).
TEST(RdmaLoopback, EndpointOpenWithoutDescriptorsIsALocalFailure) {
  if (testing_fd::kUnderThreadSanitizer)
    GTEST_SKIP() << testing_fd::kTsanNoThreadsAfterFork;
  constexpr int kNoDevice = 77;
  const int result = testing_fd::RunInChild([] {
    if (!HaveRdma()) return kNoDevice;
    ::unsetenv("DFKV_RDMA_RAIL_TIERS");
    ::setenv("DFKV_RDMA_KEEPALIVE_MS", "0", 1);
    ::setenv("DFKV_RDMA_IDLE_MS", "0", 1);
    RdmaNode node("local-open");
    RdmaTransport rt(kMaxMsg);
    RdmaTransportTestPeer::StopMaintenance(&rt);
    rt.OnPeerTopology(PublishedPeer(node.addr, "healthy-peer"));
    const std::vector<BlockKey> keys = {ToBlockKey(SelfHdr(), "absent")};
    std::vector<char> exists;
    // A served operation leaves the peer's capabilities known, so the next
    // connection needs no probe, and its connection is then kept busy so
    // that the next operation has to open another. With more than one rail
    // an acquisition may choose a rail without an idle connection and open
    // a new one, leaving the served connection in the pool for the next
    // operation: connections are held until the pooled one is among them.
    const Status served = rt.ExistMany(node.addr, keys, &exists)[0];
    if (served == Status::kIOError || served == Status::kResourceExhausted)
      return 10;
    const auto pool_hits = [&] {
      return CounterVal(rt.MetricsText(),
                        "dfkv_rdma_endpoint_cache_hits_total");
    };
    const long hits_before = pool_hits();
    std::vector<decltype(RdmaTransportTestPeer::AcquireControl(
        &rt, node.addr))>
        held;
    while (pool_hits() == hits_before && held.size() < 16) {
      auto* conn = RdmaTransportTestPeer::AcquireControl(&rt, node.addr);
      if (!conn) return 11;
      held.push_back(conn);
    }
    if (pool_hits() == hits_before) return 18;

    Status starved = Status::kOk;
    {
      testing_fd::NoFreeDescriptors exhausted(/*spare=*/1);
      if (!exhausted.active()) return 12;
      starved = rt.ExistMany(node.addr, keys, &exists)[0];
    }
    if (starved != Status::kResourceExhausted) return 13;
    const std::string metrics = rt.MetricsText();
    if (CounterVal(metrics,
                   "dfkv_rdma_client_local_resource_failures_total") != 1)
      return 14;
    if (!LabeledSeriesAllZero(metrics, "dfkv_rdma_client_rail_errors_total"))
      return 15;
    if (CounterVal(metrics, "dfkv_rdma_client_remote_rail_failures_total") !=
        0)
      return 16;
    // Another rail would run out of descriptors the same way: not tried.
    if (CounterVal(metrics, "dfkv_rdma_client_cross_rail_retries_total") != 0)
      return 19;
    // With descriptors back the same operation is served.
    if (rt.ExistMany(node.addr, keys, &exists)[0] != served) return 17;
    for (auto* conn : held)
      RdmaTransportTestPeer::ReleaseControl(&rt, node.addr, conn);
    return 0;
  });
  if (result == kNoDevice) GTEST_SKIP() << "no RDMA device";
  EXPECT_EQ(result, 0) << "step code from the child, see the test body";
}

// Out of descriptors, an operation whose rail has no idle connection cannot
// open one, but another rail's idle connection to the same peer needs no
// descriptor: the operation is served there instead of failing at once.
TEST(RdmaLoopback, DescriptorExhaustionUsesAnIdleConnectionOnAnotherRail) {
  if (testing_fd::kUnderThreadSanitizer)
    GTEST_SKIP() << testing_fd::kTsanNoThreadsAfterFork;
  constexpr int kSkip = 77;
  const int result = testing_fd::RunInChild([] {
    if (!HaveRdma()) return kSkip;
    const auto discovered = rdma::RdmaTopology::Discover(
        {}, rdma::RdmaDiscoveryPolicy::kActiveOnly);
    if (discovered.status != rdma::RdmaDiscoveryStatus::kOk ||
        discovered.devices.size() < 2)
      return kSkip;
    const std::string rails =
        discovered.devices[0].name + "," + discovered.devices[1].name;
    ::unsetenv("DFKV_RDMA_RAIL_TIERS");
    ::setenv("DFKV_RDMA_KEEPALIVE_MS", "0", 1);
    ::setenv("DFKV_RDMA_IDLE_MS", "0", 1);
    ::setenv("DFKV_RDMA_NUMA", "0", 1);
    ::setenv("DFKV_RDMA_DEV", rails.c_str(), 1);
    RdmaNode node("fallback-pooled");
    ::setenv("DFKV_RDMA_PRIMARY_DEV", discovered.devices[0].name.c_str(), 1);
    RdmaTransport rt(kMaxMsg, rails);
    RdmaTransportTestPeer::StopMaintenance(&rt);
    rt.OnPeerTopology(PublishedPeer(node.addr, "healthy-peer"));
    const std::vector<BlockKey> keys = {ToBlockKey(SelfHdr(), "absent")};
    std::vector<char> exists;
    const Status served = rt.ExistMany(node.addr, keys, &exists)[0];
    if (served == Status::kIOError || served == Status::kResourceExhausted)
      return 10;
    // An idle connection on the second rail...
    auto* idle = RdmaTransportTestPeer::AcquireControlOn(&rt, node.addr, 1);
    if (!idle) return 11;
    RdmaTransportTestPeer::ReleaseControl(&rt, node.addr, idle);
    // ...and none on the first, the one the operation prefers: hold its
    // connections until a new one had to be opened.
    const auto misses = [&] {
      return CounterVal(rt.MetricsText(),
                        "dfkv_rdma_endpoint_cache_misses_total");
    };
    const long misses_before = misses();
    std::vector<decltype(RdmaTransportTestPeer::AcquireControlOn(
        &rt, node.addr, 0))>
        held;
    while (misses() == misses_before && held.size() < 16) {
      auto* conn = RdmaTransportTestPeer::AcquireControlOn(&rt, node.addr, 0);
      if (!conn) return 12;
      held.push_back(conn);
    }
    if (misses() == misses_before) return 13;
    Status starved = Status::kOk;
    {
      testing_fd::NoFreeDescriptors exhausted(/*spare=*/1);
      if (!exhausted.active()) return 14;
      starved = rt.ExistMany(node.addr, keys, &exists)[0];
    }
    if (starved != served) return 15;
    const std::string metrics = rt.MetricsText();
    if (CounterVal(metrics,
                   "dfkv_rdma_client_local_resource_failures_total") != 1)
      return 16;
    if (CounterVal(metrics,
                   "dfkv_rdma_client_local_resource_rail_fallbacks_total") !=
        1)
      return 17;
    if (!LabeledSeriesAllZero(metrics, "dfkv_rdma_client_rail_errors_total"))
      return 18;
    for (auto* conn : held)
      RdmaTransportTestPeer::ReleaseControl(&rt, node.addr, conn);
    return 0;
  });
  if (result == kSkip) GTEST_SKIP() << "needs two ACTIVE RDMA devices";
  EXPECT_EQ(result, 0) << "step code from the child, see the test body";
}

// With no descriptor at all, socket() in the dial fails before any endpoint
// is opened. That is this process's want of descriptors too, not a peer that
// cannot be reached: the idle connection on another rail serves the
// operation, and the peer is not held off as unreachable.
TEST(RdmaLoopback, DialWithoutDescriptorsUsesAnIdleConnectionOnAnotherRail) {
  if (testing_fd::kUnderThreadSanitizer)
    GTEST_SKIP() << testing_fd::kTsanNoThreadsAfterFork;
  constexpr int kSkip = 77;
  const int result = testing_fd::RunInChild([] {
    if (!HaveRdma()) return kSkip;
    const auto discovered = rdma::RdmaTopology::Discover(
        {}, rdma::RdmaDiscoveryPolicy::kActiveOnly);
    if (discovered.status != rdma::RdmaDiscoveryStatus::kOk ||
        discovered.devices.size() < 2)
      return kSkip;
    const std::string rails =
        discovered.devices[0].name + "," + discovered.devices[1].name;
    ::unsetenv("DFKV_RDMA_RAIL_TIERS");
    ::setenv("DFKV_RDMA_KEEPALIVE_MS", "0", 1);
    ::setenv("DFKV_RDMA_IDLE_MS", "0", 1);
    ::setenv("DFKV_RDMA_NUMA", "0", 1);
    ::setenv("DFKV_RDMA_DEV", rails.c_str(), 1);
    RdmaNode node("fallback-dial");
    ::setenv("DFKV_RDMA_PRIMARY_DEV", discovered.devices[0].name.c_str(), 1);
    RdmaTransport rt(kMaxMsg, rails);
    RdmaTransportTestPeer::StopMaintenance(&rt);
    rt.OnPeerTopology(PublishedPeer(node.addr, "healthy-peer"));
    const std::vector<BlockKey> keys = {ToBlockKey(SelfHdr(), "absent")};
    std::vector<char> exists;
    const Status served = rt.ExistMany(node.addr, keys, &exists)[0];
    if (served == Status::kIOError || served == Status::kResourceExhausted)
      return 10;
    auto* idle = RdmaTransportTestPeer::AcquireControlOn(&rt, node.addr, 1);
    if (!idle) return 11;
    RdmaTransportTestPeer::ReleaseControl(&rt, node.addr, idle);
    const auto misses = [&] {
      return CounterVal(rt.MetricsText(),
                        "dfkv_rdma_endpoint_cache_misses_total");
    };
    const long misses_before = misses();
    std::vector<decltype(RdmaTransportTestPeer::AcquireControlOn(
        &rt, node.addr, 0))>
        held;
    while (misses() == misses_before && held.size() < 16) {
      auto* conn = RdmaTransportTestPeer::AcquireControlOn(&rt, node.addr, 0);
      if (!conn) return 12;
      held.push_back(conn);
    }
    if (misses() == misses_before) return 13;
    Status starved = Status::kOk;
    {
      testing_fd::NoFreeDescriptors exhausted;  // not even one for socket()
      if (!exhausted.active()) return 14;
      starved = rt.ExistMany(node.addr, keys, &exists)[0];
    }
    if (starved != served) return 15;
    const std::string metrics = rt.MetricsText();
    if (CounterVal(metrics,
                   "dfkv_rdma_client_local_resource_failures_total") != 1)
      return 16;
    if (CounterVal(metrics,
                   "dfkv_rdma_client_local_resource_rail_fallbacks_total") !=
        1)
      return 17;
    // The peer was not held off: the next operation that needs a new
    // connection on the first rail gets one.
    for (auto* conn : held)
      RdmaTransportTestPeer::ReleaseControl(&rt, node.addr, conn);
    std::vector<decltype(RdmaTransportTestPeer::AcquireControlOn(
        &rt, node.addr, 0))>
        again;
    for (int i = 0; i < 18; ++i) {
      auto* conn = RdmaTransportTestPeer::AcquireControlOn(&rt, node.addr, 0);
      if (!conn) return 18;
      again.push_back(conn);
    }
    for (auto* conn : again)
      RdmaTransportTestPeer::ReleaseControl(&rt, node.addr, conn);
    return 0;
  });
  if (result == kSkip) GTEST_SKIP() << "needs two ACTIVE RDMA devices";
  EXPECT_EQ(result, 0) << "step code from the child, see the test body";
}

// ENOMEM from building an endpoint may be the device's own limit on QPs or
// CQs, not the process's memory: another rail gets a new connection.
TEST(RdmaLoopback, EndpointOutOfMemoryOnOneRailIsServedOnAnother) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const auto discovered = rdma::RdmaTopology::Discover(
      {}, rdma::RdmaDiscoveryPolicy::kActiveOnly);
  if (discovered.status != rdma::RdmaDiscoveryStatus::kOk ||
      discovered.devices.size() < 2)
    GTEST_SKIP() << "needs two ACTIVE RDMA devices";
  const std::string rails =
      discovered.devices[0].name + "," + discovered.devices[1].name;
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv numa("DFKV_RDMA_NUMA", "0");
  std::unique_ptr<RdmaNode> node;
  {
    ScopedEnv server_rails("DFKV_RDMA_DEV", rails.c_str());
    node = std::make_unique<RdmaNode>("enomem-rail");
  }
  ScopedEnv primary("DFKV_RDMA_PRIMARY_DEV", discovered.devices[0].name.c_str());
  RdmaTransport rt(kMaxMsg, rails);
  rt.OnPeerTopology(PublishedPeer(node->addr, "healthy-peer"));
  RdmaTransportTestPeer::FailOpenOn(&rt, 0, ENOMEM);
  const std::vector<BlockKey> keys = {ToBlockKey(SelfHdr(), "absent")};
  std::vector<char> exists;
  const Status status = rt.ExistMany(node->addr, keys, &exists)[0];
  EXPECT_NE(status, Status::kResourceExhausted);
  EXPECT_NE(status, Status::kIOError);
  const std::string metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics,
                       "dfkv_rdma_client_local_resource_failures_total"),
            1);
  EXPECT_EQ(CounterVal(metrics,
                       "dfkv_rdma_client_local_resource_rail_fallbacks_total"),
            1);
  EXPECT_TRUE(LabeledSeriesAllZero(metrics, "dfkv_rdma_client_rail_errors_total"))
      << metrics;
  // For a moment the rail is not chosen again: no dial, no handshake the
  // server would see abandoned, no new failure.
  EXPECT_EQ(rt.ExistMany(node->addr, keys, &exists)[0], status);
  EXPECT_EQ(CounterVal(rt.MetricsText(),
                       "dfkv_rdma_client_local_resource_failures_total"),
            1);

  // ENOMEM on every rail: the process is out of locked memory. The first
  // operation tries each rail once; the next ones fail at once.
  RdmaTransport all(kMaxMsg, rails);
  all.OnPeerTopology(PublishedPeer(node->addr, "healthy-peer"));
  RdmaTransportTestPeer::FailOpenOn(&all, 0, ENOMEM);
  RdmaTransportTestPeer::FailOpenOn(&all, 1, ENOMEM);
  EXPECT_EQ(all.ExistMany(node->addr, keys, &exists)[0],
            Status::kResourceExhausted);
  EXPECT_EQ(CounterVal(all.MetricsText(),
                       "dfkv_rdma_client_local_resource_failures_total"),
            2);
  for (int i = 0; i < 3; ++i)
    EXPECT_EQ(all.ExistMany(node->addr, keys, &exists)[0],
              Status::kResourceExhausted);
  EXPECT_EQ(CounterVal(all.MetricsText(),
                       "dfkv_rdma_client_local_resource_failures_total"),
            2)
      << "rails held for ENOMEM were dialed again";
  EXPECT_EQ(CounterVal(all.MetricsText(),
                       "dfkv_rdma_client_local_resource_held_total"),
            3);
}

// A rail held after ENOMEM is held only for new endpoints: its idle pooled
// connections keep serving. The hold used to exclude the rail from selection
// altogether, so a single-rail peer whose opens hit ENOMEM (the device's QP
// limit) was unreachable even through its healthy idle connection.
TEST(RdmaLoopback, AnEnomemHoldStillUsesTheRailsIdleConnections) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const auto discovered = rdma::RdmaTopology::Discover(
      {}, rdma::RdmaDiscoveryPolicy::kActiveOnly);
  if (discovered.status != rdma::RdmaDiscoveryStatus::kOk ||
      discovered.devices.empty())
    GTEST_SKIP() << "needs an ACTIVE RDMA device";
  const std::string rail = discovered.devices[0].name;
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv numa("DFKV_RDMA_NUMA", "0");
  std::unique_ptr<RdmaNode> node;
  {
    ScopedEnv server_rail("DFKV_RDMA_DEV", rail.c_str());
    node = std::make_unique<RdmaNode>("enomem-pooled");
  }
  RdmaTransport rt(kMaxMsg, rail);
  rt.OnPeerTopology(PublishedPeer(node->addr, "healthy-peer"));
  const std::vector<BlockKey> keys = {ToBlockKey(SelfHdr(), "absent")};
  std::vector<char> exists;
  const Status served = rt.ExistMany(node->addr, keys, &exists)[0];
  ASSERT_NE(served, Status::kResourceExhausted);
  ASSERT_NE(served, Status::kIOError);
  // Its pooled connection is busy: the next op opens one and hits ENOMEM.
  auto* busy = RdmaTransportTestPeer::AcquireControlOn(&rt, node->addr, 0);
  ASSERT_NE(busy, nullptr);
  RdmaTransportTestPeer::FailOpenOn(&rt, 0, ENOMEM);
  EXPECT_EQ(rt.ExistMany(node->addr, keys, &exists)[0],
            Status::kResourceExhausted);
  RdmaTransportTestPeer::ReleaseControl(&rt, node->addr, busy);
  // Within the hold, the idle connection serves.
  EXPECT_EQ(rt.ExistMany(node->addr, keys, &exists)[0], served);
  const std::string metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics,
                       "dfkv_rdma_client_local_resource_held_total"),
            0);
  EXPECT_EQ(CounterVal(metrics,
                       "dfkv_rdma_client_local_resource_failures_total"),
            1);
}

// A failed work completion on the client must stay attributable afterwards:
// which peer, which local device, which operation and which completion status.
// The failures are real, not injected: the server reclaims two idle pooled
// connections, so the next request on each client endpoint cannot be
// delivered. Both are counted; a (peer, device, status, vendor syndrome) is
// logged once per throttle window, so the two failures produce one line when
// they report the same syndrome and never two lines that say the same thing.
TEST(RdmaLoopback, ClientFailedCompletionNamesPeerDeviceOperationAndStatus) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  if (SoftRoceOnly()) GTEST_SKIP() << kRxeNoRetryExceeded;
  ScopedEnv idle_reclaim("DFKV_RDMA_IDLE_MS", "120");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  // The failed completion has to arrive before the window deadline, however
  // long this HCA takes to exhaust its retries.
  ScopedEnv op_timeout("DFKV_RDMA_OP_TIMEOUT_MS", "60000");
  ScopedEnv batch_timeout("DFKV_RDMA_BATCH_OP_TIMEOUT_MS", nullptr);
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  // One rail: with several, rail selection could send a probe to a rail that
  // has no pooled connection, and nothing would fail.
  const auto discovered = rdma::RdmaTopology::Discover(
      {}, rdma::RdmaDiscoveryPolicy::kActiveOnly);
  if (discovered.status != rdma::RdmaDiscoveryStatus::kOk ||
      discovered.devices.empty())
    GTEST_SKIP() << "no ACTIVE RDMA device";
  const char* configured = std::getenv("DFKV_RDMA_DEV");
  const std::string rail =
      configured && *configured && std::strchr(configured, ',') == nullptr
          ? configured
          : discovered.devices.front().name;
  ScopedEnv one_rail("DFKV_RDMA_DEV", rail.c_str());
  RdmaNode node("wc-attribution-client");
  RdmaTransport rt(kMaxMsg, rail);
  const std::vector<BlockKey> keys = {ToBlockKey(SelfHdr(), "absent")};

  constexpr int kCallers = 2;
  // Existence probes from kCallers threads released together; returns how many
  // of them failed.
  const auto probe_together = [&] {
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::atomic<int> failed{0};
    std::vector<std::thread> callers;
    for (int i = 0; i < kCallers; ++i) {
      callers.emplace_back([&] {
        ready.fetch_add(1, std::memory_order_relaxed);
        while (!start.load(std::memory_order_acquire))
          std::this_thread::yield();
        std::vector<char> exists;
        if (rt.ExistMany(node.addr, keys, &exists)[0] == Status::kIOError)
          failed.fetch_add(1, std::memory_order_relaxed);
      });
    }
    while (ready.load(std::memory_order_relaxed) != kCallers)
      std::this_thread::yield();
    start.store(true, std::memory_order_release);
    for (auto& caller : callers) caller.join();
    return failed.load();
  };

  // Slow replies make the probes overlap, so each needs its own connection
  // and both end up parked in the pool.
  node.handler_delay_ms.store(200, std::memory_order_relaxed);
  ASSERT_EQ(probe_together(), 0);
  node.handler_delay_ms.store(0, std::memory_order_relaxed);
  ASSERT_EQ(
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total"),
      kCallers);
  EXPECT_TRUE(LabeledSeries(rt.MetricsText(),
                            "dfkv_rdma_client_rail_wc_errors_total")
                  .empty())
      << "no completion has failed yet";

  // Past the 120 ms idle window the server has destroyed its side of both
  // pooled connections; the client only finds out when it sends.
  std::this_thread::sleep_for(std::chrono::milliseconds(400));
  testing::internal::CaptureStderr();
  const int failed_probes = probe_together();
  const std::string logs = testing::internal::GetCapturedStderr();
  EXPECT_EQ(failed_probes, 0)
      << "attribution must not change the stale-pool retry";

  const std::string metrics = rt.MetricsText();
  const auto failures =
      LabeledSeries(metrics, "dfkv_rdma_client_rail_wc_errors_total");
  long counted = 0;
  for (const auto& [labels, count] : failures) {
    counted += count;
    EXPECT_EQ(LabelValue(labels, "dev"), rail) << labels;
    EXPECT_FALSE(LabelValue(labels, "status").empty()) << labels;
    EXPECT_NE(LabelValue(labels, "status"), "success") << labels;
    EXPECT_EQ(labels.find("peer"), std::string::npos)
        << "peer addresses are unbounded and belong in the log only";
  }
  EXPECT_EQ(counted, kCallers) << metrics;

  const auto lines = LinesContaining(logs, "rdma client CQ failure");
  EXPECT_GE(lines.size(), 1u) << logs;
  EXPECT_LE(lines.size(), static_cast<size_t>(kCallers)) << logs;
  std::unordered_set<std::string> logged_kinds;
  for (const std::string& line : lines) {
    EXPECT_TRUE(logged_kinds
                    .insert(LogField(line, "status") + "/" +
                            LogField(line, "vendor"))
                    .second)
        << "one line per (peer, device, status, vendor) per throttle window\n"
        << logs;
    EXPECT_EQ(LogField(line, "peer"), node.addr) << line;
    EXPECT_EQ(LogField(line, "dev"), rail) << line;
    EXPECT_EQ(LogField(line, "op"), "exist") << line;
    EXPECT_NE(LogField(line, "status"), "") << line;
    EXPECT_NE(LogField(line, "status"), "0") << line;
    EXPECT_NE(LogField(line, "vendor"), "") << line;
    EXPECT_NE(LogField(line, "qp"), "") << line;
    EXPECT_EQ(LogField(line, "lane"), "control") << line;
    EXPECT_EQ(LogField(line, "pooled"), "1") << line;
    EXPECT_EQ(LogField(line, "verdict"), "endpoint") << line;
    // A pooled connection that failed on reuse is not held against the
    // peer's rail (and this peer has no identity to hold it against).
    EXPECT_EQ(LogField(line, "peer_rail_charged"), "0") << line;
    bool matches_series = false;
    for (const auto& [labels, count] : failures) {
      (void)count;
      matches_series =
          matches_series ||
          LabelValue(labels, "status") == LogField(line, "status_name");
    }
    EXPECT_TRUE(matches_series) << line;
  }
}

// The server must name the peer and the local device of a failed completion,
// and count it by status. Every failed connection gets its own line: two
// connections from the same client address failing with the same status
// moments apart are both logged, each with its own QP numbers and vendor
// syndrome. The completions are made to fail through the CQ test hook on the
// first request of a connection, on both serve loops.
static void ExerciseServerFailedCompletionAttribution(bool use_uring) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
#ifndef DFKV_WITH_URING
  if (use_uring) GTEST_SKIP() << "io_uring not compiled";
#endif
  ScopedEnv uring("DFKV_SERVER_URING", use_uring ? "1" : "0");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  // The client never gets a reply on a dropped connection; bound its wait.
  ScopedEnv op_timeout("DFKV_RDMA_OP_TIMEOUT_MS", "500");
  ScopedEnv batch_timeout("DFKV_RDMA_BATCH_OP_TIMEOUT_MS", nullptr);
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  // One rail, so both failures land on the same device.
  const auto discovered = rdma::RdmaTopology::Discover(
      {}, rdma::RdmaDiscoveryPolicy::kActiveOnly);
  if (discovered.status != rdma::RdmaDiscoveryStatus::kOk ||
      discovered.devices.empty())
    GTEST_SKIP() << "no ACTIVE RDMA device";
  const char* configured = std::getenv("DFKV_RDMA_DEV");
  const std::string rail =
      configured && *configured && std::strchr(configured, ',') == nullptr
          ? configured
          : discovered.devices.front().name;
  ScopedEnv one_rail("DFKV_RDMA_DEV", rail.c_str());

  // Three failures from one client address with one status: two share a
  // vendor syndrome, so any throttle, whatever it keys on, would swallow one.
  constexpr int kFailedConnections = 3;
  constexpr unsigned kFirstVendorSyndrome = 129;
  // A connection stops at its first failed completion, so each rewritten
  // completion ends a different connection.
  std::atomic<int> rewritten{0};
  testing::internal::CaptureStderr();
  std::string metrics;
  bool used_uring = false;
  bool counted = false;
  {
    RdmaNode node(use_uring ? "wc-attribution-uring" : "wc-attribution-sync",
                  kMaxMsg, false, use_uring, {},
                  [&](ibv_wc* batch, int* count, size_t) {
                    if (*count <= 0) return;
                    int index = rewritten.load();
                    while (index < kFailedConnections &&
                           !rewritten.compare_exchange_weak(index, index + 1)) {
                    }
                    if (index >= kFailedConnections) return;
                    batch[0].status = IBV_WC_RETRY_EXC_ERR;
                    batch[0].vendor_err =
                        kFirstVendorSyndrome + static_cast<unsigned>(index % 2);
                  });
    used_uring = node.rsrv->UseUringPath();
    RdmaTransport rt(kMaxMsg, rail);
    const std::vector<BlockKey> keys = {ToBlockKey(SelfHdr(), "absent")};
    std::vector<char> exists;
    // Each probe opens a connection whose first request the server fails; the
    // probe itself then times out, which is not what is under test. With a CQ
    // hook installed the io_uring loop gives a silent connection 1 ms, so a
    // request can also arrive after its connection is already gone: keep
    // asking until enough requests reached a live connection.
    for (int attempt = 0;
         attempt < 20 &&
         node.rsrv->CompletionErrors() < static_cast<uint64_t>(kFailedConnections);
         ++attempt) {
      const uint64_t before = node.rsrv->CompletionErrors();
      (void)rt.ExistMany(node.addr, keys, &exists);
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
      while (node.rsrv->CompletionErrors() == before &&
             std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    counted = node.rsrv->CompletionErrors() ==
              static_cast<uint64_t>(kFailedConnections);
    metrics = node.rsrv->MetricsText();
  }
  const std::string logs = testing::internal::GetCapturedStderr();
  ASSERT_EQ(used_uring, use_uring);
  if (use_uring &&
      CounterVal(metrics, "dfkv_uring_init_fallbacks_total") != 0)
    GTEST_SKIP() << "io_uring unavailable at runtime";
  ASSERT_TRUE(counted) << metrics;

  EXPECT_EQ(DeviceCounterVal(metrics, "dfkv_rdma_rail_completion_errors_total",
                             rail),
            kFailedConnections);
  const auto by_status =
      LabeledSeries(metrics, "dfkv_rdma_rail_wc_errors_total");
  ASSERT_EQ(by_status.size(), 1u) << metrics;
  EXPECT_EQ(by_status[0].first,
            "dev=\"" + rail + "\",status=\"retry_exc_err\"");
  EXPECT_EQ(by_status[0].second, kFailedConnections);

  const auto lines = LinesContaining(logs, "rdma server CQ failure");
  ASSERT_EQ(lines.size(), static_cast<size_t>(kFailedConnections))
      << "one line per failed connection\n" << logs;
  std::unordered_set<std::string> vendors;
  const auto opened = LinesContaining(logs, "rdma conn:");
  for (const std::string& line : lines) {
    EXPECT_EQ(LogField(line, "loop"), use_uring ? "uring" : "sync") << line;
    EXPECT_EQ(LogField(line, "status"), "12") << line;
    EXPECT_EQ(LogField(line, "status_name"), "retry_exc_err") << line;
    EXPECT_EQ(LogField(line, "dev"), rail) << line;
    EXPECT_NE(LogField(line, "peer_qp"), "") << line;
    EXPECT_NE(LogField(line, "age_ms"), "") << line;
    EXPECT_EQ(LogField(line, "suppressed"), "")
        << "the server line is not throttled\n" << line;
    const std::string peer = LogField(line, "peer");
    EXPECT_EQ(peer.rfind("127.0.0.1:", 0), 0u) << line;
    vendors.insert(LogField(line, "vendor"));
    // The connection-open line carries the same peer and device, so a QP
    // number seen in either line resolves to an address.
    bool opened_by_same_peer = false;
    for (const std::string& open_line : opened) {
      opened_by_same_peer =
          opened_by_same_peer ||
          (LogField(open_line, "peer") == peer &&
           LogField(open_line, "dev") == rail &&
           LogField(open_line, "qp") == LogField(line, "qp"));
    }
    EXPECT_TRUE(opened_by_same_peer) << line << "\n" << logs;
  }
  EXPECT_EQ(vendors, (std::unordered_set<std::string>{
                         std::to_string(kFirstVendorSyndrome),
                         std::to_string(kFirstVendorSyndrome + 1)}))
      << "a failure with a different vendor syndrome is not hidden\n"
      << logs;
}

// A connection ending in RETRY_EXC counts its peer host on its device's
// retry-exhausted gauge, once however many of its connections fail that
// way, and for no other device.
TEST(RdmaLoopback, RetryExhaustedConnectionsCountTheirPeerHostOnce) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "0");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv op_timeout("DFKV_RDMA_OP_TIMEOUT_MS", "500");
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  const std::string rail = SingleTestRail();
  if (rail.empty()) GTEST_SKIP() << "no ACTIVE RDMA device";
  std::atomic<int> rewritten{0};
  RdmaNode node("retry-peers", kMaxMsg, false, false, {},
                [&](ibv_wc* batch, int* count, size_t) {
                  if (*count <= 0 || rewritten.load() >= 3) return;
                  rewritten.fetch_add(1);
                  batch[0].status = IBV_WC_RETRY_EXC_ERR;
                });
  RdmaTransport rt(kMaxMsg, rail);
  const std::vector<BlockKey> keys = {ToBlockKey(SelfHdr(), "absent")};
  std::vector<char> exists;
  for (int attempt = 0; attempt < 20 && node.rsrv->CompletionErrors() < 3;
       ++attempt) {
    const uint64_t before = node.rsrv->CompletionErrors();
    (void)rt.ExistMany(node.addr, keys, &exists);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
    while (node.rsrv->CompletionErrors() == before &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  ASSERT_GE(node.rsrv->CompletionErrors(), 3u);
  const std::string metrics = node.rsrv->MetricsText();
  for (const auto& dev : node.rsrv->DeviceNames()) {
    const std::string line =
        "dfkv_rdma_rail_retry_exhausted_peers{dev=\"" + dev + "\"} ";
    const size_t at = metrics.find(line);
    ASSERT_NE(at, std::string::npos) << dev;
    EXPECT_EQ(std::stol(metrics.substr(at + line.size())), dev == rail ? 1 : 0)
        << "three failed connections from one host count it once: " << dev;
  }
}

TEST(RdmaLoopback, ServerFailedCompletionNamesPeerDeviceAndStatusSync) {
  ExerciseServerFailedCompletionAttribution(false);
}
TEST(RdmaLoopback, ServerFailedCompletionNamesPeerDeviceAndStatusUring) {
  ExerciseServerFailedCompletionAttribution(true);
}

#ifdef DFKV_WITH_URING
class ControlledRdmaUringBackend final : public UringReader::Backend {
 public:
  struct Request {
    int fd = -1;
    void* buf = nullptr;
    unsigned len = 0;
    uint64_t off = 0;
    UringReader::Token token = UringReader::kInvalidToken;
  };

  int QueueInit(unsigned entries, io_uring*) override {
    std::lock_guard<std::mutex> lock(mu_);
    initialized_depth_ = entries;
    return queue_init_result;
  }

  void QueueExit(io_uring*) override {
    std::lock_guard<std::mutex> lock(mu_);
    exited_ = true;
    cv_.notify_all();
  }

  io_uring_sqe* GetSqe(io_uring*) override {
    std::lock_guard<std::mutex> lock(mu_);
    sqes_.emplace_back();
    return &sqes_.back();
  }

  void PrepRead(io_uring_sqe* sqe, int fd, void* buf, unsigned len,
                uint64_t off) override {
    std::lock_guard<std::mutex> lock(mu_);
    prepared_[sqe] =
        Request{fd, buf, len, off, UringReader::kInvalidToken};
  }

  void SetData(io_uring_sqe* sqe, UringReader::Token token) override {
    std::lock_guard<std::mutex> lock(mu_);
    auto found = prepared_.find(sqe);
    if (found == prepared_.end()) {
      bad_state_ = true;
      return;
    }
    found->second.token = token;
    dormant_.push_back(found->second);
    prepared_.erase(found);
  }

  int Submit(io_uring*) override {
    std::lock_guard<std::mutex> lock(mu_);
    ++submit_calls_;
    int result = static_cast<int>(dormant_.size());
    if (!submit_results_.empty()) {
      result = submit_results_.front();
      submit_results_.pop_front();
    }
    if (result > 0 && static_cast<size_t>(result) <= dormant_.size()) {
      for (int i = 0; i < result; ++i) {
        Request request = dormant_.front();
        dormant_.pop_front();
        accepted_[request.token] = request;
        history_.push_back(request);
      }
      max_inflight_ = std::max(max_inflight_, accepted_.size());
    }
    cv_.notify_all();
    return result;
  }

  int PeekCqe(io_uring*, io_uring_cqe** cqe) override {
    std::lock_guard<std::mutex> lock(mu_);
    if (ready_.empty()) {
      *cqe = nullptr;
      return -EAGAIN;
    }
    *cqe = &ready_.front();
    return 0;
  }

  int WaitCqe(io_uring*, io_uring_cqe** cqe, int timeout_ms) override {
    std::unique_lock<std::mutex> lock(mu_);
    const auto ready = [&] { return !ready_.empty() || exited_; };
    if (timeout_ms < 0) {
      cv_.wait(lock, ready);
    } else if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                             ready)) {
      *cqe = nullptr;
      return -ETIME;
    }
    if (ready_.empty()) {
      *cqe = nullptr;
      return -ETIME;
    }
    *cqe = &ready_.front();
    return 0;
  }

  void Seen(io_uring*, io_uring_cqe* cqe) override {
    std::lock_guard<std::mutex> lock(mu_);
    if (ready_.empty() || cqe != &ready_.front()) {
      bad_state_ = true;
      return;
    }
    ready_.pop_front();
    ++seen_;
    cv_.notify_all();
  }

  void SetSubmitResults(std::deque<int> results) {
    std::lock_guard<std::mutex> lock(mu_);
    submit_results_ = std::move(results);
  }

  bool WaitForSubmitted(size_t count, int timeout_ms = 5000) {
    std::unique_lock<std::mutex> lock(mu_);
    return cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                        [&] { return history_.size() >= count; });
  }

  bool WaitForSubmitCalls(size_t count, int timeout_ms = 5000) {
    std::unique_lock<std::mutex> lock(mu_);
    return cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                        [&] { return submit_calls_ >= count; });
  }

  bool WaitForSeen(size_t count, int timeout_ms = 5000) {
    std::unique_lock<std::mutex> lock(mu_);
    return cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                        [&] { return seen_ >= count; });
  }

  std::vector<Request> History() const {
    std::lock_guard<std::mutex> lock(mu_);
    return history_;
  }

  bool CompleteRead(UringReader::Token token) {
    Request request;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto found = accepted_.find(token);
      if (found == accepted_.end()) return false;
      request = found->second;
    }
    const ssize_t result =
        ::pread(request.fd, request.buf, request.len,
                static_cast<off_t>(request.off));
    return Complete(token, result);
  }

  bool Complete(UringReader::Token token, long result) {
    std::lock_guard<std::mutex> lock(mu_);
    auto found = accepted_.find(token);
    if (found == accepted_.end()) return false;
    accepted_.erase(found);
    io_uring_cqe cqe{};
    cqe.user_data = token;
    cqe.res = static_cast<int32_t>(result);
    ready_.push_back(cqe);
    if (event_fd_ >= 0) {  // as io_uring_register_eventfd does
      const uint64_t one = 1;
      (void)!::write(event_fd_, &one, sizeof(one));
    }
    cv_.notify_all();
    return true;
  }

  int RegisterEventFd(io_uring*, int fd) override {
    std::lock_guard<std::mutex> lock(mu_);
    event_fd_ = fd;
    return 0;
  }

  size_t MaxInflight() const {
    std::lock_guard<std::mutex> lock(mu_);
    return max_inflight_;
  }

  size_t SeenCount() const {
    std::lock_guard<std::mutex> lock(mu_);
    return seen_;
  }

  size_t AcceptedCount() const {
    std::lock_guard<std::mutex> lock(mu_);
    return accepted_.size();
  }

  bool Exited() const {
    std::lock_guard<std::mutex> lock(mu_);
    return exited_;
  }

  bool BadState() const {
    std::lock_guard<std::mutex> lock(mu_);
    return bad_state_;
  }

  int queue_init_result = 0;

 private:
  mutable std::mutex mu_;
  std::condition_variable cv_;
  unsigned initialized_depth_ = 0;
  size_t submit_calls_ = 0;
  size_t seen_ = 0;
  size_t max_inflight_ = 0;
  bool exited_ = false;
  bool bad_state_ = false;
  std::list<io_uring_sqe> sqes_;
  std::map<io_uring_sqe*, Request> prepared_;
  std::deque<Request> dormant_;
  std::map<UringReader::Token, Request> accepted_;
  std::vector<Request> history_;
  std::list<io_uring_cqe> ready_;
  std::deque<int> submit_results_;
  int event_fd_ = -1;
};
#endif

// A cache node that ALSO wires the async-GET prep + complete hooks, so the server
// uses the nonblocking io_uring GET pipeline when DFKV_SERVER_URING=1 (and the
// binary was built with -DDFKV_WITH_URING). With the env off / unbuilt these
// hooks are simply never consulted and the node behaves like a plain RdmaNode.
struct RdmaUringNode {
  fs::path dir;
  std::unique_ptr<KvNodeServer> srv;
  std::unique_ptr<RdmaServer> rsrv;
  std::string addr;
  mutable std::mutex observation_mu;
  std::map<std::string, size_t> prepare_calls;
  std::map<std::string, size_t> range_calls;
#ifdef DFKV_WITH_URING
  mutable std::mutex backend_mu;
  std::condition_variable backend_cv;
  std::vector<std::unique_ptr<ControlledRdmaUringBackend>> backends;
#endif


  explicit RdmaUringNode(
      const std::string& tag, size_t max_msg = kMaxMsg
#ifdef DFKV_WITH_URING
      ,
      std::function<void(ControlledRdmaUringBackend*)> backend_configure = {}
#endif
      ) {
    dir = fs::temp_directory_path() / ("dfkv_uring_" + tag);
    fs::remove_all(dir);
    fs::create_directories(dir);
    srv = std::make_unique<KvNodeServer>(dir.string(), 1ull << 30);
    EXPECT_EQ(srv->Start(0), Status::kOk);
    rsrv = std::make_unique<RdmaServer>(
        [this](uint8_t op, const BlockKey& key, uint64_t off, uint64_t len,
               const char* pl, uint64_t pll, std::string* out,
               size_t* value_len) {
          return srv->ProcessRequestForKey(
              op, key, off, len, pl, pll, out, value_len);
        },
        max_msg);
    rsrv->set_range_handler(  // sync fallback (used when uring path is off)
        [this](const BlockKey& key, uint64_t off, uint64_t len,
               char* io_buf, size_t cap, const char** out_data,
               size_t* out_len, size_t* value_len) {
          {
            std::lock_guard<std::mutex> lock(observation_mu);
            ++range_calls[key.Filename()];
          }
          return srv->RangeDirectForKey(
              key, off, len, io_buf, cap, out_data, out_len, value_len);
        });
    rsrv->set_cache_direct_handler(
        [this](const BlockKey& key, char* data, size_t len, size_t cap,
               std::chrono::steady_clock::time_point batch_arrival) {
          return srv->CacheDirectForKey(key, data, len, cap, batch_arrival);
        });
    rsrv->set_prepare_read_handler(
        [this](const BlockKey& key, uint64_t off, uint64_t len,
               char* staging, size_t cap) {
          {
            std::lock_guard<std::mutex> lock(observation_mu);
            ++prepare_calls[key.Filename()];
          }
          return srv->PrepareReadForKey(key, off, len, staging, cap);
        });
#ifdef DFKV_WITH_URING
    if (backend_configure)
      InstallControlledBackendFactory(std::move(backend_configure));
#endif
    EXPECT_EQ(rsrv->Start(0), Status::kOk);
    addr = "127.0.0.1:" + std::to_string(rsrv->port());
  }
#ifdef DFKV_WITH_URING
  void InstallControlledBackendFactory(
      std::function<void(ControlledRdmaUringBackend*)> configure = {}) {
    RdmaServerTestPeer::SetUringBackendFactory(
        rsrv.get(), [this, configure = std::move(configure)] {
          auto backend = std::make_unique<ControlledRdmaUringBackend>();
          if (configure) configure(backend.get());
          ControlledRdmaUringBackend* result = backend.get();
          {
            std::lock_guard<std::mutex> lock(backend_mu);
            backends.push_back(std::move(backend));
          }
          backend_cv.notify_all();
          return result;
        });
  }

  ControlledRdmaUringBackend* WaitForBackend(
      size_t index, int timeout_ms = 5000) {
    std::unique_lock<std::mutex> lock(backend_mu);
    if (!backend_cv.wait_for(
            lock, std::chrono::milliseconds(timeout_ms),
            [&] { return backends.size() > index; }))
      return nullptr;
    return backends[index].get();
  }

  ControlledRdmaUringBackend* WaitForBackendWithSubmitted(
      size_t count, int timeout_ms = 5000) const {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    do {
      for (ControlledRdmaUringBackend* backend : BackendSnapshot()) {
        if (backend->History().size() >= count) return backend;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    return nullptr;
  }

  std::vector<ControlledRdmaUringBackend*> BackendSnapshot() const {
    std::lock_guard<std::mutex> lock(backend_mu);
    std::vector<ControlledRdmaUringBackend*> snapshot;
    snapshot.reserve(backends.size());
    for (const auto& backend : backends) snapshot.push_back(backend.get());
    return snapshot;
  }

  struct ControlledSubmission {
    ControlledRdmaUringBackend* backend = nullptr;
    ControlledRdmaUringBackend::Request request;
  };

  std::vector<ControlledSubmission> AggregateSubmissions() const {
    std::vector<ControlledSubmission> submissions;
    for (ControlledRdmaUringBackend* backend : BackendSnapshot()) {
      for (const auto& request : backend->History())
        submissions.push_back({backend, request});
    }
    return submissions;
  }

  bool WaitForAggregateSubmitted(size_t count, int timeout_ms = 5000) const {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    do {
      if (AggregateSubmissions().size() >= count) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
  }

  size_t AggregateAccepted() const {
    size_t accepted = 0;
    for (ControlledRdmaUringBackend* backend : BackendSnapshot())
      accepted += backend->AcceptedCount();
    return accepted;
  }
#endif
  ~RdmaUringNode() {
    if (rsrv) rsrv->Stop();
    if (srv) srv->Stop();
    fs::remove_all(dir);
  }
  size_t PrepareCalls(const std::string& logical_key) const {
    std::lock_guard<std::mutex> lock(observation_mu);
    const auto found =
        prepare_calls.find(ToBlockKey(SelfHdr(), logical_key).Filename());
    return found == prepare_calls.end() ? 0 : found->second;
  }
  size_t RangeCalls(const std::string& logical_key) const {
    std::lock_guard<std::mutex> lock(observation_mu);
    const auto found =
        range_calls.find(ToBlockKey(SelfHdr(), logical_key).Filename());
    return found == range_calls.end() ? 0 : found->second;
  }
};

#ifdef DFKV_WITH_URING
// RDS-11: with a disk read in flight the io_uring serve loop waited on the
// ring alone, for up to 1 ms at a time, and saw a new request only when that
// wait ended. Time from posting a request to its own disk read being
// submitted, with and without another read in flight on the connection.
TEST(RdmaLoopback, ARequestDuringADiskReadIsTakenAtOnce) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "1");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "2");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  RdmaUringNode node("uring-notice", kMaxMsg,
                     [](ControlledRdmaUringBackend*) {});
  const std::string value(4096, 'n');
  std::string ignored;
  const BlockKey keys[2] = {{92001, 1}, {92002, 1}};
  for (const BlockKey& key : keys)
    ASSERT_EQ(node.srv->ProcessRequestForKey(
                  static_cast<uint8_t>(WireOp::kCache), key, 0, 0,
                  value.data(), value.size(), &ignored), Status::kOk);
  PinnedPullPeer peer;
  ASSERT_TRUE(peer.Open(node, 2));
  ControlledRdmaUringBackend* backend = nullptr;
  struct CompleteHeld {
    ControlledRdmaUringBackend** backend;
    ~CompleteHeld() {
      if (!*backend) return;
      for (const auto& request : (*backend)->History())
        (*backend)->CompleteRead(request.token);
    }
  } complete_held{&backend};
  size_t submitted = 0;
  const auto wait_submitted = [&](size_t count) {
    if (!backend) backend = node.WaitForBackendWithSubmitted(count);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (backend && backend->History().size() < count &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    return backend && backend->History().size() >= count;
  };
  const auto post = [&](size_t slot) {
    peer.EncodePull(slot, keys[slot], 0, value.size(),
                    static_cast<uint32_t>(slot));
    return peer.ep.PostRecv(slot) &&
           peer.ep.PostSend(slot, kReqPrefix + rdma::kPullPrepareBytes);
  };
  std::vector<double> alone_us, behind_us;
  for (int round = 0; round < 60; ++round) {
    SCOPED_TRACE(round);
    const bool behind = round % 2 == 1;
    if (behind) {
      ASSERT_TRUE(post(0));
      ASSERT_TRUE(wait_submitted(++submitted));
      std::this_thread::sleep_for(
          std::chrono::microseconds(1100 + 37 * (round % 23)));
    }
    const auto t0 = std::chrono::steady_clock::now();
    ASSERT_TRUE(post(1));
    ASSERT_TRUE(wait_submitted(++submitted));
    (behind ? behind_us : alone_us).push_back(
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - t0).count());
    const auto history = backend->History();
    if (behind) ASSERT_TRUE(backend->CompleteRead(history[submitted - 2].token));
    ASSERT_TRUE(backend->CompleteRead(history[submitted - 1].token));
    std::vector<rdma::DynamicPullReady> readies;
    for (int got = 0; got < (behind ? 4 : 2); ++got) {
      ibv_wc wc{};
      ASSERT_EQ(peer.ep.WaitComp(&wc, 1, 5000), 1);
      ASSERT_EQ(wc.status, IBV_WC_SUCCESS);
      if (wc.opcode != IBV_WC_RECV) continue;
      rdma::DynamicPullReady ready;
      ASSERT_TRUE(rdma::DecodeDynamicPullReady(
          peer.ep.rbuf(wc.wr_id) + kRespPrefix, &ready));
      readies.push_back(ready);
    }
    for (size_t i = 0; i < readies.size(); ++i)
      ASSERT_TRUE(peer.Release(keys[behind ? i : 1], readies[i]));
  }
  const auto median = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  const double alone = median(alone_us), behind = median(behind_us);
  std::printf("request to disk submission, median: %.0f us alone, %.0f us"
              " behind a read in flight\n", alone, behind);
  EXPECT_LT(behind - alone, 200.0)
      << "a request behind a disk read in flight waited for the ring wait";
}

TEST(RdmaLoopback, PressurePreservesSubmittedUringRead) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "1");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "1");
  ScopedEnv idle("DFKV_RDMA_IDLE_MS", "5000");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  const std::string value(4096, 'q');
  const std::string budget = std::to_string(
      2 * rdma::V2SlotSize(kMaxMsg) + rdma::V2SlotSize(value.size()));
  ScopedEnv segment("DFKV_RDMA_RECV_SEGMENT_SIZE", budget.c_str());
  RdmaUringNode node("pressure-uring-read", kMaxMsg,
                     [](ControlledRdmaUringBackend*) {});
  const BlockKey key{90178, 1};
  std::string ignored;
  ASSERT_EQ(node.srv->ProcessRequestForKey(
                static_cast<uint8_t>(WireOp::kCache), key, 0, 0,
                value.data(), value.size(), &ignored), Status::kOk);
  {
    PinnedPullPeer busy;
    ASSERT_TRUE(busy.Open(node));
    EncodeReqVersion(busy.ep.sbuf(0), kNativeProtoRdmaV2, WireOp::kPullRange,
                     key, 0, value.size(), rdma::kPullPrepareBytes);
    rdma::EncodePullPrepareControl({}, busy.ep.sbuf(0) + kReqPrefix);
    const size_t request_bytes = kReqPrefix + rdma::kPullPrepareBytes;
    ASSERT_TRUE(busy.ep.PostRecv(0));
    ASSERT_TRUE(busy.ep.PostSend(0, request_bytes));
    ControlledRdmaUringBackend* backend =
        node.WaitForBackendWithSubmitted(1);
    ASSERT_NE(backend, nullptr) << "GET must be owned by the async disk queue";
    const auto requests = backend->History();
    ASSERT_EQ(requests.size(), 1u);
    struct CompleteOnExit {
      ControlledRdmaUringBackend* backend;
      UringReader::Token token;
      bool done = false;
      ~CompleteOnExit() {
        if (!done) backend->CompleteRead(token);
      }
    } completion{backend, requests[0].token};
    std::this_thread::sleep_for(std::chrono::milliseconds(2100));
    {
      PinnedPullPeer recent;
      ASSERT_TRUE(recent.Open(node));
      ASSERT_TRUE(PinnedExists(recent, key));
      size_t quiescent = 0;
      for (int i = 0; i < 1000 && quiescent == 0; ++i) {
        quiescent = RdmaServerTestPeer::RefreshQuiescentActivity(*node.rsrv, 1500);
        if (quiescent == 0)
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      ASSERT_EQ(quiescent, 1u) << "the queued disk GET is not reclaimable";
      PinnedPullPeer excess;
      ASSERT_TRUE(excess.Open(node))
          << "recycle the idle QP without aborting a queued disk GET";
      const std::string metrics = node.rsrv->MetricsText();
      EXPECT_GT(CounterVal(
                    metrics, "dfkv_rdma_recv_segment_allocation_failures_total"),
                0);
      EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_segment_evictions_total"), 1);
      EXPECT_EQ(node.rsrv->CompletionErrors(), 0u);
    }
    ASSERT_TRUE(backend->CompleteRead(completion.token));
    completion.done = true;
    bool sent = false, received = false;
    rdma::DynamicPullReady ready;
    while (!sent || !received) {
      ibv_wc wc{};
      ASSERT_EQ(busy.ep.WaitComp(&wc, 1, 5000), 1);
      ASSERT_EQ(wc.status, IBV_WC_SUCCESS);
      if (wc.opcode == IBV_WC_SEND) {
        sent = true;
      } else {
        ASSERT_EQ(wc.opcode, IBV_WC_RECV);
        Status status = Status::kIOError;
        uint64_t bytes = 0;
        ASSERT_TRUE(DecodeRespVersion(busy.ep.rbuf(0), kNativeProtoRdmaV2,
                                      &status, &bytes));
        EXPECT_EQ(status, Status::kOk);
        ASSERT_EQ(bytes, rdma::kDynamicPullReadyBytes);
        ASSERT_EQ(wc.byte_len, kRespPrefix + bytes);
        ASSERT_TRUE(rdma::DecodeDynamicPullReady(
            busy.ep.rbuf(0) + kRespPrefix, &ready));
        received = true;
      }
    }
    EXPECT_EQ(ready.data_len, value.size());
    EXPECT_EQ(ready.value_len, value.size());
    std::string output;
    ASSERT_TRUE(busy.Read(ready, &output));
    EXPECT_EQ(output, value);
    ASSERT_TRUE(busy.Release(key, ready));
    EXPECT_EQ(node.rsrv->CompletionErrors(), 0u);
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  long used = -1;
  while (std::chrono::steady_clock::now() < deadline) {
    used = CounterVal(node.rsrv->MetricsText(),
                      "dfkv_rdma_recv_segment_used_bytes");
    if (node.rsrv->ActiveConns() == 0 && used == 0) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  EXPECT_EQ(node.rsrv->ActiveConns(), 0u);
  EXPECT_EQ(used, 0);
}

TEST(RdmaLoopback, AsyncRangeManyKeepsDiskFollowerAndReadyRepliesOrdered) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "1");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  ScopedEnv coalesce("DFKV_READ_COALESCE", "1");
  RdmaUringNode node("async-range-follower", kMaxMsg,
                     [](ControlledRdmaUringBackend*) {});
  const BlockKey first = ToBlockKey(SelfHdr(), "first");
  const BlockKey second = ToBlockKey(SelfHdr(), "second");
  const BlockKey missing = ToBlockKey(SelfHdr(), "missing");
  const std::string a = PatternValue(8192, 27);
  const std::string b = PatternValue(8192, 28);
  std::string ignored;
  const std::array<std::pair<BlockKey, const std::string*>, 2> stored{{
      {first, &a}, {second, &b}}};
  for (const auto& item : stored) {
    ASSERT_EQ(node.srv->ProcessRequestForKey(
                  static_cast<uint8_t>(WireOp::kCache), item.first, 0, 0,
                  item.second->data(), item.second->size(), &ignored),
              Status::kOk);
  }
  RdmaTransport transport(kMaxMsg);
  const std::vector<BlockKey> keys{first, first, missing, second};
  std::vector<std::string> outputs;
  std::vector<uint64_t> value_lens;
  auto result = std::async(std::launch::async, [&] {
    return transport.RangeMany(node.addr, keys, 97, 4113,
                               &outputs, &value_lens);
  });
  struct DrainOnFailure {
    RdmaUringNode& node;
    bool finished = false;
    ~DrainOnFailure() {
      if (finished) return;
      for (const auto& submission : node.AggregateSubmissions())
        submission.backend->CompleteRead(submission.request.token);
      node.rsrv->Stop();
    }
  } drain{node};
  ASSERT_TRUE(node.WaitForAggregateSubmitted(2));
  const auto submissions = node.AggregateSubmissions();
  ASSERT_EQ(submissions.size(), 2u);
  ASSERT_TRUE(submissions[1].backend->CompleteRead(
      submissions[1].request.token));
  EXPECT_EQ(result.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  ASSERT_TRUE(submissions[0].backend->CompleteRead(
      submissions[0].request.token));
  ASSERT_EQ(result.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  const auto statuses = result.get();
  drain.finished = true;
  EXPECT_EQ(statuses, (std::vector<Status>{
                         Status::kOk, Status::kOk,
                         Status::kNotFound, Status::kOk}));
  EXPECT_EQ(outputs, (std::vector<std::string>{
                        a.substr(97, 4113), a.substr(97, 4113),
                        "", b.substr(97, 4113)}));
  EXPECT_EQ(value_lens, (std::vector<uint64_t>{8192, 8192, 0, 8192}));
  EXPECT_EQ(CounterVal(node.rsrv->MetricsText(),
                       "dfkv_rdma_dynamic_get_mr_active"), 0);
  EXPECT_EQ(node.rsrv->CompletionErrors(), 0u);
}

// A batch GET whose first READY is out while a later read still waits on the
// disk: the client RELEASEs only after the last READY, so the first grant
// waits for as long as the disk does. That is not a silent client, and the
// connection is not cut, however long the disk takes (here twice the grant
// timeout).
TEST(RdmaLoopback, AGrantWaitingForALaterSlowReadIsNotExpired) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "1");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");
  ScopedEnv ram("DFKV_RAM_TIER", "0");
  ScopedEnv grant("DFKV_RDMA_GRANT_TIMEOUT_MS", "300");
  ScopedEnv op_timeout("DFKV_RDMA_OP_TIMEOUT_MS", "20000");
  RdmaUringNode node("grant-slow-disk", kMaxMsg,
                     [](ControlledRdmaUringBackend*) {});
  const BlockKey first = ToBlockKey(SelfHdr(), "slow-first");
  const BlockKey second = ToBlockKey(SelfHdr(), "slow-second");
  const std::string a = PatternValue(8192, 31);
  const std::string b = PatternValue(8192, 32);
  std::string ignored;
  for (const auto& item : {std::make_pair(first, &a), std::make_pair(second, &b)})
    ASSERT_EQ(node.srv->ProcessRequestForKey(
                  static_cast<uint8_t>(WireOp::kCache), item.first, 0, 0,
                  item.second->data(), item.second->size(), &ignored),
              Status::kOk);
  RdmaTransport transport(kMaxMsg);
  const std::vector<BlockKey> keys{first, second};
  std::vector<std::string> outputs;
  std::vector<uint64_t> value_lens;
  auto result = std::async(std::launch::async, [&] {
    return transport.RangeMany(node.addr, keys, 0, 8192, &outputs,
                               &value_lens);
  });
  struct DrainOnFailure {
    RdmaUringNode& node;
    bool finished = false;
    ~DrainOnFailure() {
      if (finished) return;
      for (const auto& submission : node.AggregateSubmissions())
        submission.backend->CompleteRead(submission.request.token);
      node.rsrv->Stop();
    }
  } drain{node};
  ASSERT_TRUE(node.WaitForAggregateSubmitted(2));
  const auto submissions = node.AggregateSubmissions();
  ASSERT_EQ(submissions.size(), 2u);
  // The first read lands: its READY goes out. The second waits.
  ASSERT_TRUE(submissions[0].backend->CompleteRead(
      submissions[0].request.token));
  std::this_thread::sleep_for(std::chrono::milliseconds(700));
  ASSERT_TRUE(submissions[1].backend->CompleteRead(
      submissions[1].request.token));
  ASSERT_EQ(result.wait_for(std::chrono::seconds(10)),
            std::future_status::ready);
  const auto statuses = result.get();
  drain.finished = true;
  EXPECT_EQ(statuses, (std::vector<Status>{Status::kOk, Status::kOk}));
  EXPECT_EQ(outputs, (std::vector<std::string>{a, b}));
  EXPECT_EQ(CounterVal(node.rsrv->MetricsText(),
                       "dfkv_rdma_grant_expiries_total"),
            0);
}
#endif

struct MultiWindowGet {
  explicit MultiWindowGet(size_t targets_per_window, size_t window_count,
                          size_t seed)
      : sizes(targets_per_window * (window_count - 1) + 1),
        destinations(sizes.size()),
        pointers(sizes.size()) {
    for (size_t i = 0; i < sizes.size(); ++i) {
      sizes[i] = 113 + ((seed + i * 17) % 89);
      destinations[i].assign(sizes[i], '\0');
      pointers[i] = destinations[i].data();
    }
  }

  size_t capacity() const {
    size_t total = 0;
    for (size_t size : sizes) total += size;
    return total;
  }
  size_t window_count(size_t targets_per_window) const {
    return 1 + (sizes.size() - 1) / targets_per_window;
  }
  std::string Flatten() const {
    std::string out;
    out.reserve(capacity());
    for (const auto& destination : destinations) out += destination;
    return out;
  }

  std::vector<size_t> sizes;
  std::vector<std::string> destinations;
  std::vector<void*> pointers;
};

std::string PatternValue(size_t size, size_t seed) {
  std::string value(size, '\0');
  for (size_t i = 0; i < size; ++i)
    value[i] = static_cast<char>((seed * 131 + i * 29 + i / 97) & 0xff);
  return value;
}

TEST(RdmaLoopback, MaxBlockBytesReportsResolvedClientLocalCeiling) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv max_block("DFKV_RDMA_MAX_BLOCK_BYTES", nullptr);
  ScopedEnv max_payload("DFKV_RDMA_MAX_PAYLOAD_BYTES", nullptr);
  {
    RdmaTransport transport(8u << 20);
    KVClient client({}, SelfHdr(), &transport);
    EXPECT_EQ(dfkv_max_block_bytes(&client), 4u << 20);
  }
  {
    ScopedEnv explicit_block("DFKV_RDMA_MAX_BLOCK_BYTES", "65536");
    RdmaTransport transport(kMaxMsg);
    KVClient client({}, SelfHdr(), &transport);
    EXPECT_EQ(dfkv_max_block_bytes(&client), 65536u);
    // The getter reports this client's resolved bound, not the current env.
    ScopedEnv changed_block("DFKV_RDMA_MAX_BLOCK_BYTES", "131072");
    EXPECT_EQ(dfkv_max_block_bytes(&client), 65536u);
  }
  {
    ScopedEnv explicit_block("DFKV_RDMA_MAX_BLOCK_BYTES", "8388608");
    ScopedEnv payload_cap("DFKV_RDMA_MAX_PAYLOAD_BYTES", "131072");
    RdmaTransport transport(kMaxMsg);
    KVClient client({}, SelfHdr(), &transport);
    EXPECT_EQ(dfkv_max_block_bytes(&client), 131072u);
  }
}

// DCP2 declared caps: a client that tightens its max block size gets smaller
// shared slots and must still complete every op within the declaration;
// oversized ops fail client-side with kInvalid without touching the wire.
TEST(RdmaLoopback, DeclaredCapsRoundTripAndClientSideBound) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("caps");
  setenv("DFKV_RDMA_MAX_BLOCK_BYTES", "65536", 1);  // declare 64 KiB
  RdmaTransport rt(kMaxMsg);
  unsetenv("DFKV_RDMA_MAX_BLOCK_BYTES");            // don't leak into other tests
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  EXPECT_EQ(dfkv_max_block_bytes(&c), 65536u);

  // Within the declaration: normal round-trip on the right-sized connection.
  std::string v(60 * 1024, 'a');
  ASSERT_TRUE(c.Put("caps-ok", v.data(), v.size()));
  std::string got(v.size(), '\0');
  ASSERT_TRUE(c.Get("caps-ok", got.data(), got.size()));
  EXPECT_EQ(got, v);

  // Over the declaration (but under the transport max): the client bound must
  // reject before sending -- Put returns false, connection stays healthy.
  std::string big(128 * 1024, 'b');
  EXPECT_FALSE(c.Put("caps-over", big.data(), big.size()));
  ASSERT_TRUE(c.Get("caps-ok", got.data(), got.size())) << "conn must survive the rejected op";
  EXPECT_EQ(got, v);
  EXPECT_EQ(dfkv_max_block_bytes(&c), 65536u);
}

TEST(RdmaLoopback, DataConnectionsUseActualBlockSizeClasses) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv max_block("DFKV_RDMA_MAX_BLOCK_BYTES", "262144");
  ScopedEnv min_block("DFKV_RDMA_CONNECTION_MIN_BLOCK_BYTES", "4096");
  RdmaNode node("adaptive-classes");
  RdmaTransport transport(kMaxMsg);
  KVClient client({{"n", node.addr}}, SelfHdr(), &transport);

  std::string small(4000, 's');
  ASSERT_TRUE(client.Put("small", small.data(), small.size()));
  EXPECT_EQ(RdmaTransportTestPeer::IdleDataBounds(&transport, node.addr),
            (std::vector<size_t>{4096}));

  std::string large(60 * 1024, 'l');
  ASSERT_TRUE(client.Put("large", large.data(), large.size()));
  EXPECT_EQ(RdmaTransportTestPeer::IdleDataBounds(&transport, node.addr),
            (std::vector<size_t>{4096, 65536}));

  std::string got(small.size(), '\0');
  ASSERT_TRUE(client.Get("small", got.data(), got.size()));
  EXPECT_EQ(got, small);
  EXPECT_EQ(RdmaTransportTestPeer::IdleDataBounds(&transport, node.addr),
            (std::vector<size_t>{4096, 65536}));
  EXPECT_EQ(RdmaTransportTestPeer::ConnectionBound(transport, 1), 4096u);
  EXPECT_EQ(RdmaTransportTestPeer::ConnectionBound(transport, 5000), 8192u);
  EXPECT_EQ(RdmaTransportTestPeer::ConnectionBound(transport, 60 * 1024),
            65536u);
}

TEST(RdmaLoopback, DataConnectionsUseActualDepthClasses) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");
  ScopedEnv max_block("DFKV_RDMA_MAX_BLOCK_BYTES", "262144");
  ScopedEnv min_block("DFKV_RDMA_CONNECTION_MIN_BLOCK_BYTES", "4096");
  RdmaNode node("adaptive-depth");
  RdmaTransport transport(kMaxMsg);
  KVClient client({{"n", node.addr}}, SelfHdr(), &transport);

  std::string scalar(4000, 's');
  ASSERT_TRUE(client.Put("scalar", scalar.data(), scalar.size()));
  EXPECT_EQ(RdmaTransportTestPeer::IdleDataDepths(&transport, node.addr),
            (std::vector<size_t>{1}));

  std::vector<std::string> values(4, std::string(4000, 'b'));
  std::vector<CacheItem> items;
  for (size_t i = 0; i < values.size(); ++i) {
    items.push_back(CacheItem{
        ToBlockKey(SelfHdr(), "batch-depth-" + std::to_string(i)),
        values[i].data(), values[i].size()});
  }
  const auto statuses = transport.CacheMany(node.addr, items);
  ASSERT_EQ(statuses.size(), items.size());
  for (Status status : statuses) EXPECT_EQ(status, Status::kOk);
  EXPECT_EQ(RdmaTransportTestPeer::IdleDataDepths(&transport, node.addr),
            (std::vector<size_t>{1, 4}));

  EXPECT_EQ(RdmaTransportTestPeer::ConnectionDepth(&transport, node.addr, 1),
            1u);
  EXPECT_EQ(RdmaTransportTestPeer::ConnectionDepth(&transport, node.addr, 3),
            4u);
  const std::string metrics = transport.MetricsText();
  EXPECT_NE(metrics.find(
                "dfkv_rdma_client_connection_class_opened_total"
                "{block_bytes=\"4096\",depth=\"1\"} 1"),
            std::string::npos);
  EXPECT_NE(metrics.find(
                "dfkv_rdma_client_connection_class_opened_total"
                "{block_bytes=\"4096\",depth=\"4\"} 1"),
            std::string::npos);
  EXPECT_NE(metrics.find(
                "dfkv_rdma_client_connection_class_idle"
                "{block_bytes=\"4096\",depth=\"1\"} 1"),
            std::string::npos);
}

TEST(RdmaLoopback, ManyAdaptiveDepthOneConnectionsStayBounded) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  int count = 32;
  if (const char* value = std::getenv("DFKV_RDMA_TEST_CONNECTIONS"))
    count = std::max(1, std::min(2000, std::atoi(value)));
  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");
  ScopedEnv max_block("DFKV_RDMA_MAX_BLOCK_BYTES", "4096");
  ScopedEnv min_block("DFKV_RDMA_CONNECTION_MIN_BLOCK_BYTES", "4096");
  ScopedEnv qp_budget("DFKV_RDMA_QP_BUDGET", "4096");
  ScopedEnv wr_budget("DFKV_RDMA_WR_BUDGET", "16384");
  ScopedEnv registered_budget("DFKV_RDMA_REGISTERED_BYTES_BUDGET",
                              "2147483648");
  ScopedEnv endpoint_budget("DFKV_RDMA_ENDPOINT_CACHE_MAX", "4096");
  ScopedEnv recv_max("DFKV_RDMA_RECV_SEGMENT_SIZE", "1073741824");
  ScopedEnv recv_chunk("DFKV_RDMA_RECV_CHUNK_BYTES", "67108864");
  RdmaNode node("many-adaptive-conns");

  std::vector<std::unique_ptr<RdmaTransport>> transports;
  std::vector<std::unique_ptr<KVClient>> clients;
  transports.reserve(static_cast<size_t>(count));
  clients.reserve(static_cast<size_t>(count));
  std::string value(4000, 'c');
  for (int i = 0; i < count; ++i) {
    auto transport = std::make_unique<RdmaTransport>(kMaxMsg);
    auto client = std::make_unique<KVClient>(
        std::vector<std::pair<std::string, std::string>>{{"n", node.addr}},
        SelfHdr(), transport.get());
    ASSERT_TRUE(client->Put("conn-" + std::to_string(i), value.data(),
                            value.size()))
        << "connection " << i;
    transports.push_back(std::move(transport));
    clients.push_back(std::move(client));
  }
  const std::string metrics = node.rsrv->MetricsText();
  EXPECT_GE(CounterVal(metrics, "dfkv_rdma_active_conns"), count);
  EXPECT_LE(CounterVal(metrics,
                       "dfkv_rdma_connection_bytes{class=\"data\"}"),
            static_cast<long>(count) * 32 * 1024);
}

static void ExerciseDepthOneReplyCredits(bool use_uring) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
#ifndef DFKV_WITH_URING
  if (use_uring) GTEST_SKIP() << "io_uring not compiled";
#endif
  ScopedEnv uring("DFKV_SERVER_URING", use_uring ? "1" : "0");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "1");
  ScopedEnv max_block("DFKV_RDMA_MAX_BLOCK_BYTES", "4096");
  ScopedEnv min_block("DFKV_RDMA_CONNECTION_MIN_BLOCK_BYTES", "4096");
  ScopedEnv ram_on("DFKV_RAM_TIER", "1");
  ScopedEnv ram_bytes("DFKV_RAM_TIER_BYTES", "134217728");
  ScopedEnv extent("DFKV_RAM_TIER_EXTENT_BYTES", "1048576");
  ScopedEnv reserve("DFKV_RAM_TIER_LARGE_RESERVE_BYTES", "0");
  ScopedEnv shards("DFKV_RAM_TIER_SHARDS", "4");
  ScopedEnv ram_reclaim("DFKV_RAM_RECLAIM_MS", "0");
  ScopedEnv granularity("DFKV_SLAB_GRANULARITY", "4096");
  RdmaNode node(use_uring ? "reply-credit-uring" : "reply-credit-sync",
                kMaxMsg, false, use_uring);
  ASSERT_EQ(node.rsrv->UseUringPath(), use_uring);
  constexpr size_t kClients = 4;
  constexpr size_t kThreadsPerClient = 8;
  constexpr size_t kWorkers = kClients * kThreadsPerClient;
  // Registered source storage outlives every client and transport.
  std::vector<std::string> values;
  values.reserve(kClients);
  for (size_t i = 0; i < kClients; ++i)
    values.push_back(PatternValue(4096, 101 + i));
  std::vector<std::unique_ptr<RdmaTransport>> transports;
  std::vector<std::unique_ptr<KVClient>> clients;
  for (size_t i = 0; i < kClients; ++i) {
    auto transport = std::make_unique<RdmaTransport>(
        kMaxMsg, node.rsrv->DeviceNames().front());
    auto client = std::make_unique<KVClient>(
        std::vector<std::pair<std::string, std::string>>{{"n", node.addr}},
        SelfHdr(), transport.get());
    client->set_batch_concurrency(1);
    ASSERT_TRUE(client->RegisterMemory(values[i].data(), values[i].size()));
    ASSERT_TRUE(client->Put("reply-" + std::to_string(i),
                            values[i].data(), values[i].size()));
    transports.push_back(std::move(transport));
    clients.push_back(std::move(client));
  }
  if (use_uring && CounterVal(node.rsrv->MetricsText(),
                              "dfkv_uring_init_fallbacks_total") != 0)
    GTEST_SKIP() << "io_uring unavailable at runtime";

  std::mutex gate_mu;
  std::condition_variable gate_cv;
  bool start = false;
  std::vector<int> failures(kWorkers, 0);
  std::vector<std::thread> workers;
  for (size_t i = 0; i < kWorkers; ++i) {
    workers.emplace_back([&, i] {
      {
        std::unique_lock<std::mutex> lock(gate_mu);
        gate_cv.wait(lock, [&] { return start; });
      }
      const size_t client_index = i / kThreadsPerClient;
      const std::string prefix = "reply-" + std::to_string(i) + "-";
      const auto& value = values[client_index];
      std::vector<KvPutItem> items{{prefix, value.data(), value.size()}};
      items[0].key.reserve(prefix.size() + 3);
      for (size_t round = 0; round < 128; ++round) {
        items[0].key.assign(prefix);
        items[0].key.append(std::to_string(round));
        const auto results = clients[client_index]->BatchPut(items);
        if (results.size() != 1 || !results[0]) {
          ++failures[i];
          return;
        }
      }
      std::string output(value.size(), '\0');
      if (!clients[client_index]->Get(items[0].key, output.data(), output.size()) ||
          output != value)
        ++failures[i];
    });
  }
  {
    std::lock_guard<std::mutex> lock(gate_mu);
    start = true;
  }
  gate_cv.notify_all();
  for (auto& worker : workers) worker.join();
  for (size_t i = 0; i < kWorkers; ++i)
    EXPECT_EQ(failures[i], 0) << "worker " << i;
  for (size_t i = 0; i < kClients; ++i) {
    EXPECT_EQ(CounterVal(transports[i]->MetricsText(),
                         "dfkv_rdma_client_completion_timeouts_total"), 0)
        << "client " << i;
  }
}

TEST(RdmaLoopback, DepthOneReplyCreditsSurviveConcurrentReuseSync) {
  ExerciseDepthOneReplyCredits(false);
}

TEST(RdmaLoopback, DepthOneReplyCreditsSurviveConcurrentReuseUring) {
  ExerciseDepthOneReplyCredits(true);
}

TEST(RdmaLoopback, ServerReceivePoolCommitsChunksOnDemand) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv recv_max("DFKV_RDMA_RECV_SEGMENT_SIZE", "4194304");
  ScopedEnv recv_chunk("DFKV_RDMA_RECV_CHUNK_BYTES", "1048576");
  ScopedEnv max_block("DFKV_RDMA_MAX_BLOCK_BYTES", "262144");
  ScopedEnv min_block("DFKV_RDMA_CONNECTION_MIN_BLOCK_BYTES", "262144");
  RdmaNode node("lazy-recv-pool");
  const long initial_bytes = CounterVal(
      node.rsrv->MetricsText(), "dfkv_rdma_recv_segment_bytes");

  std::vector<std::unique_ptr<RdmaTransport>> transports;
  std::vector<std::unique_ptr<KVClient>> clients;
  std::string value(60 * 1024, 'p');
  for (int i = 0; i < 4; ++i) {
    auto transport = std::make_unique<RdmaTransport>(kMaxMsg);
    auto client = std::make_unique<KVClient>(
        std::vector<std::pair<std::string, std::string>>{
            {"n", node.addr}},
        SelfHdr(), transport.get());
    ASSERT_TRUE(client->Put("pool-" + std::to_string(i),
                            value.data(), value.size()));
    transports.push_back(std::move(transport));
    clients.push_back(std::move(client));
  }

  const std::string metrics = node.rsrv->MetricsText();
  const long committed =
      CounterVal(metrics, "dfkv_rdma_recv_segment_bytes");
  EXPECT_GT(committed, initial_bytes);
  EXPECT_LE(committed, 4 * 1024 * 1024);
  EXPECT_EQ(CounterVal(
                metrics,
                "dfkv_rdma_recv_segment_growth_failures_total"),
            0);
}

// With no explicit override, DCP2 declares the transport's safe global cap;
// blocks right up to that bound still round-trip.
TEST(RdmaLoopback, DefaultDeclarationKeepsGlobalCap) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("nocaps");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  std::string v(kMaxMsg - 1, 'c');  // near the global raw-value cap
  ASSERT_TRUE(c.Put("full-size", v.data(), v.size()));
  std::string got(v.size(), '\0');
  ASSERT_TRUE(c.Get("full-size", got.data(), got.size()));
  EXPECT_EQ(got, v);
}

TEST(RdmaLoopback,
     PinnedBouncePoolQuarantinesSlotAfterEnqueueAndRecoverySyncErrors) {
  const CudaLib* cuda = ActiveCudaForTest();
  if (!cuda) GTEST_SKIP() << "no usable CUDA device";

  // Leave healthy capacity after the deliberately poisoned lease so the test
  // can prove both quarantine and forward progress in the process-wide pool.
  ScopedEnv pool_bytes("DFKV_CUDA_PINNED_POOL_BYTES", "3145728");
  ScopedEnv slot_bytes("DFKV_CUDA_PINNED_SLOT_BYTES", "1048576");
  const PinnedBouncePoolStats before = GetPinnedBouncePoolStatsForTest();
  ASSERT_GT(before.slot_bytes, 0u);

  const std::string value = CudaTestBytes(257, 0x42);
  CudaGuardedBuffer failed_destination(cuda, value.size());
  CudaGuardedBuffer healthy_destination(cuda, value.size());
  ASSERT_TRUE(failed_destination.Allocate());
  ASSERT_TRUE(healthy_destination.Allocate());

  CUcontext destination_context = nullptr;
  ASSERT_TRUE(cuda->GetCurrentCtx(&destination_context));
  ASSERT_NE(destination_context, nullptr);
  ScopedCudaContextRestore restore_context(cuda, destination_context);
  ASSERT_TRUE(cuda->SetCurrentCtx(nullptr));
  CUcontext caller_context = destination_context;
  ASSERT_TRUE(cuda->GetCurrentCtx(&caller_context));
  ASSERT_EQ(caller_context, nullptr);

  ScopedCudaAsyncFault fault(cuda);
  DestinationPublisher failed_publisher;
  fault.FailNextEnqueueAndSynchronize();
  EXPECT_FALSE(failed_publisher.Copy(failed_destination.payload(), value.data(),
                                     value.size(),
                                     DestinationMemoryKind::kDevice));
  const CUdeviceptr quarantined_source = fault.last_source();
  ASSERT_NE(quarantined_source, 0u);
  EXPECT_EQ(fault.memcpy_async_calls(), 1u);
  EXPECT_EQ(fault.stream_synchronize_calls(), 1u)
      << "an enqueue error must fence the stream before releasing the lease";

  CUcontext context_after_failure = nullptr;
  ASSERT_TRUE(cuda->GetCurrentCtx(&context_after_failure));
  EXPECT_EQ(context_after_failure, caller_context);
  EXPECT_EQ(GetPinnedBouncePoolStatsForTest().active_leases, 0u);

  EXPECT_FALSE(failed_publisher.Finish());
  const uint64_t calls_after_finish = fault.stream_synchronize_calls();
  EXPECT_FALSE(failed_publisher.Finish());
  EXPECT_EQ(fault.stream_synchronize_calls(), calls_after_finish)
      << "Finish must not repeat teardown or lease handling";
  EXPECT_EQ(GetPinnedBouncePoolStatsForTest().active_leases, 0u);

  DestinationPublisher healthy_publisher;
  ASSERT_TRUE(healthy_publisher.Copy(healthy_destination.payload(), value.data(),
                                     value.size(),
                                     DestinationMemoryKind::kDevice));
  const CUdeviceptr healthy_source = fault.last_source();
  EXPECT_NE(healthy_source, quarantined_source)
      << "the failed recovery fence must permanently quarantine its slot";
  ASSERT_TRUE(healthy_publisher.Finish());
  EXPECT_EQ(GetPinnedBouncePoolStatsForTest().active_leases, 0u);
  CUcontext context_after_finish = destination_context;
  ASSERT_TRUE(cuda->GetCurrentCtx(&context_after_finish));
  EXPECT_EQ(context_after_finish, caller_context);

  ASSERT_TRUE(cuda->SetCurrentCtx(destination_context));

  std::string actual;
  ASSERT_TRUE(healthy_destination.ReadPayload(&actual));
  EXPECT_EQ(actual, value);
  EXPECT_TRUE(healthy_destination.GuardsIntact());
}

TEST(RdmaLoopback,
     PinnedBouncePoolChunksAndBackpressuresConcurrentCudaPublications) {
  const CudaLib* cuda = ActiveCudaForTest();
  if (!cuda) GTEST_SKIP() << "no usable CUDA device";

  // Keep this regression cheap when it is the first CUDA publication in the
  // process. If another test initialized the process-wide pool first, the
  // assertions below derive the already-fixed slot size and remain valid.
  ScopedEnv pool_bytes("DFKV_CUDA_PINNED_POOL_BYTES", "2097152");
  ScopedEnv slot_bytes("DFKV_CUDA_PINNED_SLOT_BYTES", "1048576");
  const PinnedBouncePoolStats before = GetPinnedBouncePoolStatsForTest();
  ASSERT_GT(before.slot_bytes, 0u);

  constexpr size_t kWorkers = 32;
  const size_t payload_size =
      std::max<size_t>(3052800, before.slot_bytes + 8197);
  auto destinations =
      std::make_shared<std::vector<std::unique_ptr<CudaGuardedBuffer>>>();
  destinations->reserve(kWorkers);
  for (size_t i = 0; i < kWorkers; ++i) {
    auto destination =
        std::make_unique<CudaGuardedBuffer>(cuda, payload_size);
    ASSERT_TRUE(destination->Allocate()) << "worker " << i;
    destinations->push_back(std::move(destination));
  }

  struct WaveState {
    std::mutex mutex;
    std::condition_variable cv;
    bool start = false;
    size_t ready = 0;
    size_t done = 0;
    std::vector<uint8_t> published;
    std::vector<uint8_t> context_restored;
  };

  const auto run_wave = [&](uint8_t seed) {
    auto value =
        std::make_shared<const std::string>(CudaTestBytes(payload_size, seed));
    auto state = std::make_shared<WaveState>();
    state->published.assign(kWorkers, 0);
    state->context_restored.assign(kWorkers, 0);
    std::vector<std::thread> workers;
    workers.reserve(kWorkers);
    for (size_t i = 0; i < kWorkers; ++i) {
      workers.emplace_back(
          [cuda, destinations, state, value, pool_slot_bytes = before.slot_bytes,
           i] {
        CUcontext context_before = nullptr;
        CUcontext context_after = nullptr;
        const bool queried_before = cuda->GetCurrentCtx(&context_before);
        {
          std::unique_lock<std::mutex> lock(state->mutex);
          ++state->ready;
          state->cv.notify_all();
          state->cv.wait(lock, [&] { return state->start; });
        }

        DestinationPublisher publisher;
        // The first logical SG segment is itself larger than one bounce slot;
        // the odd following boundaries catch chunk truncation and offset bugs.
        const std::array<size_t, 3> segments{
            pool_slot_bytes + 17,
            2053,
            value->size() - (pool_slot_bytes + 17 + 2053),
        };
        size_t offset = 0;
        bool published = true;
        for (size_t segment : segments) {
          published &=
              publisher.Copy(static_cast<char*>((*destinations)[i]->payload()) +
                                 offset,
                             value->data() + offset, segment,
                             DestinationMemoryKind::kDevice);
          offset += segment;
        }
        published &= publisher.Finish();
        const bool queried_after = cuda->GetCurrentCtx(&context_after);

        {
          std::lock_guard<std::mutex> lock(state->mutex);
          state->published[i] = published;
          state->context_restored[i] =
              queried_before && queried_after &&
              context_before == context_after;
          ++state->done;
        }
        state->cv.notify_all();
      });
    }

    bool all_ready = false;
    {
      std::unique_lock<std::mutex> lock(state->mutex);
      all_ready = state->cv.wait_for(
          lock, std::chrono::seconds(10),
          [&] { return state->ready == kWorkers; });
      state->start = true;
    }
    state->cv.notify_all();

    bool all_done = false;
    {
      std::unique_lock<std::mutex> lock(state->mutex);
      all_done = state->cv.wait_for(
          lock, std::chrono::seconds(30),
          [&] { return state->done == kWorkers; });
    }
    for (auto& worker : workers) {
      if (all_done)
        worker.join();
      else
        worker.detach();
    }

    EXPECT_TRUE(all_ready) << "publication workers missed start deadline";
    EXPECT_TRUE(all_done) << "bounded pool deadlocked under backpressure";
    if (!all_ready || !all_done) return false;
    EXPECT_EQ(state->published,
              std::vector<uint8_t>(kWorkers, 1));
    EXPECT_EQ(state->context_restored,
              std::vector<uint8_t>(kWorkers, 1));
    for (size_t i = 0; i < kWorkers; ++i) {
      std::string actual;
      const bool read_ok = (*destinations)[i]->ReadPayload(&actual);
      EXPECT_TRUE(read_ok) << "worker " << i;
      if (read_ok) EXPECT_EQ(actual, *value) << "worker " << i;
      EXPECT_TRUE((*destinations)[i]->GuardsIntact()) << "worker " << i;
    }
    return true;
  };

  ASSERT_TRUE(run_wave(0x51));
  const PinnedBouncePoolStats warm = GetPinnedBouncePoolStatsForTest();
  EXPECT_GT(warm.allocated_slots, 0u);
  EXPECT_LT(warm.allocated_slots, kWorkers)
      << "the wave must exceed the bounded slot count";
  EXPECT_EQ(warm.active_leases, 0u);
  EXPECT_GT(warm.peak_active_leases, 0u);
  EXPECT_GT(warm.wait_count, before.wait_count)
      << "oversubscription must take the blocking backpressure path";

  ASSERT_TRUE(run_wave(0xb7));
  const PinnedBouncePoolStats reused = GetPinnedBouncePoolStatsForTest();
  EXPECT_EQ(reused.active_leases, 0u);
  EXPECT_EQ(reused.allocated_slots, warm.allocated_slots);
  EXPECT_EQ(reused.allocation_calls, warm.allocation_calls);
  EXPECT_EQ(reused.registration_calls, warm.registration_calls)
      << "all CUDA pinning must be one-time slot warmup, not per publication";
}

TEST(RdmaLoopback,
     MultiSegmentGetPublishesUnequalExactBytesToGuardedCudaDestinations) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const CudaLib* cuda = ActiveCudaForTest();
  if (!cuda) GTEST_SKIP() << "no usable CUDA device";
  ScopedEnv host_dedup("DFKV_CLIENT_NODE_DEDUP", nullptr);
  ScopedEnv gpu_dedup("DFKV_CLIENT_NODE_DEDUP_GPU", nullptr);
  ScopedEnv completion_fault("DFKV_RDMA_TEST_COMPLETION_FAULT", nullptr);
  ScopedEnv endpoint_fault("DFKV_RDMA_TEST_ENDPOINT_COMPLETION_FAULT",
                           nullptr);
  ScopedEnv local_failure("DFKV_RDMA_TEST_LOCAL_RAIL_FAILURE_ATTEMPTS",
                          nullptr);

  // Host first is deliberate: treating the whole SG item as host from its
  // first segment crashes on the following CUDA pointer. The remaining three
  // CUDA segments have unequal, odd capacities and independent guards.
  constexpr std::array<size_t, 4> kSegmentSizes{613, 997, 2051, 4099};
  const size_t total = kSegmentSizes[0] + kSegmentSizes[1] +
                       kSegmentSizes[2] + kSegmentSizes[3];
  const std::string value = CudaTestBytes(total, 0x6d);

  RdmaNode node("cuda-multisegment-get");
  RdmaTransport transport(kMaxMsg);
  KVClient client({{"n", node.addr}}, SelfHdr(), &transport);
  ASSERT_TRUE(client.Put("cuda-multisegment", value.data(), value.size()));

  const std::string host_guard(
      CudaGuardedBuffer::kGuardBytes,
      static_cast<char>(CudaGuardedBuffer::kGuardByte));
  std::string host_allocation(
      kSegmentSizes[0] + 2 * CudaGuardedBuffer::kGuardBytes,
      static_cast<char>(CudaGuardedBuffer::kGuardByte));
  char* const host_payload =
      host_allocation.data() + CudaGuardedBuffer::kGuardBytes;

  std::array<std::unique_ptr<CudaGuardedBuffer>, 3> device_buffers;
  KvGetItemSg get;
  get.key = "cuda-multisegment";
  get.dsts.push_back(host_payload);
  get.caps.push_back(kSegmentSizes[0]);
  for (size_t i = 0; i < device_buffers.size(); ++i) {
    const size_t segment = i + 1;
    device_buffers[i] =
        std::make_unique<CudaGuardedBuffer>(cuda, kSegmentSizes[segment]);
    ASSERT_TRUE(device_buffers[i]->Allocate()) << "segment " << segment;
    ASSERT_TRUE(cuda->IsDevicePtr(device_buffers[i]->payload()))
        << "segment " << segment;
    get.dsts.push_back(device_buffers[i]->payload());
    get.caps.push_back(kSegmentSizes[segment]);
  }

  std::vector<size_t> lengths;
  ASSERT_EQ(client.BatchGetAutoSg({get}, &lengths),
            std::vector<bool>({true}));
  ASSERT_EQ(lengths, std::vector<size_t>({value.size()}));

  // Every immediate read observes API-return state. Odd, unequal boundaries
  // expose word-sized over-copy, and host-first ordering proves memory kind is
  // classified per segment rather than once per SG item.
  EXPECT_EQ(std::string(host_payload, kSegmentSizes[0]),
            value.substr(0, kSegmentSizes[0]));
  EXPECT_EQ(host_allocation.substr(0, CudaGuardedBuffer::kGuardBytes),
            host_guard);
  EXPECT_EQ(host_allocation.substr(CudaGuardedBuffer::kGuardBytes +
                                   kSegmentSizes[0]),
            host_guard);
  size_t offset = kSegmentSizes[0];
  for (size_t i = 0; i < device_buffers.size(); ++i) {
    const size_t segment = i + 1;
    std::string actual;
    ASSERT_TRUE(device_buffers[i]->ReadPayload(&actual))
        << "segment " << segment;
    EXPECT_EQ(actual, value.substr(offset, kSegmentSizes[segment]))
        << "segment " << segment;
    EXPECT_TRUE(device_buffers[i]->GuardsIntact())
        << "segment " << segment;
    offset += kSegmentSizes[segment];
  }
}

TEST(RdmaLoopback, SingleEnabledRailFallbackIsNotCrossRailRetry) {
  if (!HaveRdma())
    GTEST_SKIP() << "no RDMA device (load two rdma_rxe devices for this test)";
  const auto rails = ConfiguredTwoTestRails();
  if (!HaveConfiguredActiveRails(rails))
    GTEST_SKIP() << "set DFKV_RDMA_DEV to exactly two ACTIVE test devices";
  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");
  ScopedEnv credits("DFKV_RDMA_RAIL_CREDITS", kRealHcaRailCredits);
  ScopedEnv threshold("DFKV_RDMA_RAIL_ERROR_THRESHOLD", "100");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv local_failure_fault(
      "DFKV_RDMA_TEST_LOCAL_RAIL_FAILURE_ATTEMPTS", nullptr);
  RdmaNode node("same-rail-fallback");
  RdmaTransport transport(kMaxMsg, rails[0]);
  KVClient client({{"n", node.addr}}, SelfHdr(), &transport);
  const std::string value(64 * 1024, 's');

  const std::string before = transport.MetricsText();
  {
    ScopedEnv fault("DFKV_RDMA_TEST_LOCAL_RAIL_FAILURE_ATTEMPTS", "1");
    ASSERT_TRUE(client.Put("same-rail", value.data(), value.size()));
  }
  const std::string after = transport.MetricsText();
  EXPECT_EQ(
      DeviceCounterVal(after, "dfkv_rdma_client_rail_selections_total",
                       rails[0]) -
          DeviceCounterVal(before,
                           "dfkv_rdma_client_rail_selections_total",
                           rails[0]),
      2);
  EXPECT_EQ(DeviceCounterVal(after, "dfkv_rdma_client_rail_errors_total",
                             rails[0]) -
                DeviceCounterVal(before,
                                 "dfkv_rdma_client_rail_errors_total",
                                 rails[0]),
            1);
  EXPECT_EQ(
      CounterVal(after, "dfkv_rdma_client_cross_rail_retries_total") -
          CounterVal(before, "dfkv_rdma_client_cross_rail_retries_total"),
      0);
  EXPECT_EQ(
      CounterVal(after,
                 "dfkv_rdma_client_cross_rail_retry_successes_total") -
          CounterVal(before,
                     "dfkv_rdma_client_cross_rail_retry_successes_total"),
      0);
  EXPECT_EQ(
      CounterVal(after,
                 "dfkv_rdma_client_cross_rail_retry_exhausted_total") -
          CounterVal(before,
                     "dfkv_rdma_client_cross_rail_retry_exhausted_total"),
      0);
  ExpectReleasedRailResources(before, after, {rails[0]});
}

TEST(RdmaLoopback,
     PullReadFailureClosesConnectionAndNextGetRecoversWithoutAbort) {
  ScopedEnv configured_device("DFKV_RDMA_DEV", nullptr);
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";

  RdmaNode node("pull-read-failure");
  RdmaTransport transport(kMaxMsg);
  const BlockKey key = ToBlockKey(SelfHdr(), "pull-read-failure");
  const std::string value = PatternValue(64 * 1024 + 13, 57);
  ASSERT_EQ(transport.Cache(node.addr, key, value.data(), value.size()),
            Status::kOk);

  std::string output(value.size(), '\x5a');
  std::vector<uint64_t> value_lens;
  {
    ScopedEnv read_fault("DFKV_RDMA_TEST_PULL_READ_FAILURE", "1");
    const auto statuses = transport.RangeInto(
        node.addr, {key}, {{output.data(), output.size()}}, &value_lens);
    EXPECT_EQ(statuses, std::vector<Status>({Status::kIOError}));
  }
  // A posted READ may already have modified the destination before the
  // injected failure fences its QP. Only a successful GET promises valid
  // payload bytes; the recovery below must replace the invalid contents.

  const auto recovered = transport.RangeInto(
      node.addr, {key}, {{output.data(), output.size()}}, &value_lens);
  EXPECT_EQ(recovered, std::vector<Status>({Status::kOk}));
  EXPECT_EQ(value_lens, std::vector<uint64_t>({value.size()}));
  EXPECT_EQ(output, value);
}

TEST(RdmaLoopback,
     PostSubmitPutFailureCoolsActualRemoteRailWithoutReplay) {
  if (!HaveRdma())
    GTEST_SKIP() << "no RDMA device (load two rdma_rxe devices for this test)";
  const auto rails = ConfiguredTwoTestRails();
  if (!HaveConfiguredActiveRails(rails))
    GTEST_SKIP() << "set DFKV_RDMA_DEV to exactly two ACTIVE test devices";
  ScopedEnv cooldown("DFKV_RDMA_REMOTE_RAIL_COOLDOWN_MS", "30000");
  ScopedEnv max_cooldown("DFKV_RDMA_REMOTE_RAIL_MAX_COOLDOWN_MS", "30000");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv credits("DFKV_RDMA_RAIL_CREDITS", kRealHcaRailCredits);
  ScopedEnv ambient_local_completion_fault(
      "DFKV_RDMA_TEST_COMPLETION_FAULT", nullptr);
  ScopedEnv ambient_endpoint_completion_fault(
      "DFKV_RDMA_TEST_ENDPOINT_COMPLETION_FAULT", nullptr);
  ScopedEnv ambient_local_failure(
      "DFKV_RDMA_TEST_LOCAL_RAIL_FAILURE_ATTEMPTS", nullptr);

  RdmaNode node("remote-post-submit-put");
  RdmaTransport transport(kMaxMsg, rails[0] + "," + rails[1]);
  PeerTopology topology;
  topology.peer_addr = node.addr;
  topology.peer_id = "remote-post-submit-put-peer";
  topology.generation = 1;
  topology.complete = true;
  for (const auto& rail : rails)
    topology.rails.push_back(PeerRailTopology{rail, true});
  transport.OnPeerTopology(topology);

  const std::string key_namespace = "test/remote-post-submit-put";
  KVClient client({{"n", node.addr}}, key_namespace, &transport);
  const std::string value(64 * 1024, 'f');
  const BlockKey failed_key =
      ToBlockKey(key_namespace, "completion-failed-put");
  const std::string before_fault = transport.MetricsText();
  {
    ScopedEnv fault("DFKV_RDMA_TEST_ENDPOINT_COMPLETION_FAULT", "1:1:1");
    EXPECT_FALSE(
        client.Put("completion-failed-put", value.data(), value.size()));
  }
  const std::string after_fault = transport.MetricsText();

  EXPECT_EQ(CounterVal(after_fault, "dfkv_rdma_client_v2_put_writes_total") -
                CounterVal(before_fault,
                           "dfkv_rdma_client_v2_put_writes_total"),
            1)
      << "an unsafe PUT must not be posted again after ambiguous completion";
  EXPECT_EQ(node.CacheDirectCalls(failed_key), 1u)
      << "the server must observe exactly the single posted PUT";
  EXPECT_EQ(
      CounterVal(after_fault, "dfkv_rdma_client_cross_rail_retries_total") -
          CounterVal(before_fault,
                     "dfkv_rdma_client_cross_rail_retries_total"),
      0)
      << "post-submit PUT failure is not replay-safe";

  size_t failed_rail = rails.size();
  long failed_selections = 0;
  for (size_t rail = 0; rail < rails.size(); ++rail) {
    const long selected =
        DeviceCounterVal(after_fault,
                         "dfkv_rdma_client_rail_selections_total",
                         rails[rail]) -
        DeviceCounterVal(before_fault,
                         "dfkv_rdma_client_rail_selections_total",
                         rails[rail]);
    EXPECT_TRUE(selected == 0 || selected == 1) << rails[rail];
    if (selected == 1) failed_rail = rail;
    failed_selections += selected;
  }
  ASSERT_EQ(failed_selections, 1);
  ASSERT_LT(failed_rail, rails.size());
  const size_t sibling_rail = 1 - failed_rail;
  EXPECT_EQ(
      DeviceCounterVal(
          after_fault,
          "dfkv_rdma_client_remote_rail_failures_by_rail_total",
          rails[failed_rail]) -
          DeviceCounterVal(
              before_fault,
              "dfkv_rdma_client_remote_rail_failures_by_rail_total",
              rails[failed_rail]),
      1);
  EXPECT_EQ(
      DeviceCounterVal(
          after_fault,
          "dfkv_rdma_client_remote_rail_cooldowns_by_rail_total",
          rails[failed_rail]) -
          DeviceCounterVal(
              before_fault,
              "dfkv_rdma_client_remote_rail_cooldowns_by_rail_total",
              rails[failed_rail]),
      1)
      << "the endpoint completion must cool the rail that owned its lease";
  EXPECT_EQ(
      DeviceCounterVal(
          after_fault,
          "dfkv_rdma_client_remote_rail_cooldowns_by_rail_total",
          rails[sibling_rail]) -
          DeviceCounterVal(
              before_fault,
              "dfkv_rdma_client_remote_rail_cooldowns_by_rail_total",
              rails[sibling_rail]),
      0);

  KVClient future_client({{"n", node.addr}}, key_namespace, &transport);
  const BlockKey future_key = ToBlockKey(key_namespace, "future-put");
  ASSERT_TRUE(future_client.Put("future-put", value.data(), value.size()));
  const std::string after_future = transport.MetricsText();
  EXPECT_EQ(
      DeviceCounterVal(after_future,
                       "dfkv_rdma_client_rail_selections_total",
                       rails[failed_rail]) -
          DeviceCounterVal(after_fault,
                           "dfkv_rdma_client_rail_selections_total",
                           rails[failed_rail]),
      0)
      << "future admission must bypass the cooled endpoint rail";
  EXPECT_EQ(
      DeviceCounterVal(after_future,
                       "dfkv_rdma_client_rail_selections_total",
                       rails[sibling_rail]) -
          DeviceCounterVal(after_fault,
                           "dfkv_rdma_client_rail_selections_total",
                           rails[sibling_rail]),
      1);
  EXPECT_EQ(
      CounterVal(after_future, "dfkv_rdma_client_v2_put_writes_total") -
          CounterVal(after_fault, "dfkv_rdma_client_v2_put_writes_total"),
      1);
  EXPECT_EQ(node.CacheDirectCalls(future_key), 1u);
}

TEST(RdmaLoopback, RemoteCooldownBypassesIdlePoolAndUsesSiblingRail) {
  if (!HaveRdma())
    GTEST_SKIP() << "no RDMA device (load two rdma_rxe devices for this test)";
  const auto rails = ConfiguredTwoTestRails();
  if (!HaveConfiguredActiveRails(rails))
    GTEST_SKIP() << "set DFKV_RDMA_DEV to exactly two ACTIVE test devices";
  ScopedEnv cooldown("DFKV_RDMA_REMOTE_RAIL_COOLDOWN_MS", "30000");
  ScopedEnv max_cooldown("DFKV_RDMA_REMOTE_RAIL_MAX_COOLDOWN_MS", "30000");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv credits("DFKV_RDMA_RAIL_CREDITS", kRealHcaRailCredits);

  RdmaNode node("remote-idle-bypass");
  RdmaTransport transport(kMaxMsg, rails[0] + "," + rails[1]);
  PeerTopology topology;
  topology.peer_addr = node.addr;
  topology.peer_id = "remote-idle-bypass-peer";
  topology.generation = 1;
  topology.complete = true;
  for (const auto& rail : rails)
    topology.rails.push_back(PeerRailTopology{rail, true});
  transport.OnPeerTopology(topology);

  const BlockKey key = ToBlockKey(SelfHdr(), "remote-idle-bypass");
  const std::string value(64 * 1024, 'r');
  ASSERT_EQ(transport.Cache(node.addr, key, value.data(), value.size()),
            Status::kOk);
  ASSERT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 1u);
  const std::string warm = transport.MetricsText();
  ASSERT_EQ(
      DeviceCounterVal(warm, "dfkv_rdma_client_rail_selections_total",
                       rails[0]),
      1)
      << "the deterministic first admission must warm local rail zero";

  const auto seeded = RdmaTransportTestPeer::CoolRemoteRail(
      &transport, topology.peer_id, 0);
  ASSERT_TRUE(seeded);
  ASSERT_EQ(seeded->completion.transition, RemoteRailTransition::kCooled);
  EXPECT_EQ(RdmaTransportTestPeer::RemoteAllowed(
                transport, topology.peer_id, {1, 1}, seeded->now_us),
            (std::vector<uint8_t>{0, 1}));

  const std::string before = transport.MetricsText();
  std::string output;
  ASSERT_EQ(transport.Range(node.addr, key, 0, value.size(), &output, nullptr),
            Status::kOk);
  EXPECT_EQ(output, value);
  const std::string after = transport.MetricsText();
  EXPECT_EQ(
      DeviceCounterVal(after, "dfkv_rdma_client_rail_selections_total",
                       rails[0]) -
          DeviceCounterVal(before, "dfkv_rdma_client_rail_selections_total",
                           rails[0]),
      0)
      << "the idle QP on the cooled peer/rail must not bypass admission";
  EXPECT_EQ(
      DeviceCounterVal(after, "dfkv_rdma_client_rail_selections_total",
                       rails[1]) -
          DeviceCounterVal(before, "dfkv_rdma_client_rail_selections_total",
                           rails[1]),
      1);
  EXPECT_EQ(CounterVal(after, "dfkv_rdma_endpoint_cache_hits_total") -
                CounterVal(before, "dfkv_rdma_endpoint_cache_hits_total"),
            0);
  EXPECT_EQ(CounterVal(after, "dfkv_rdma_endpoint_cache_misses_total") -
                CounterVal(before, "dfkv_rdma_endpoint_cache_misses_total"),
            1);
  EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 2u)
      << "the cooled idle QP remains present only to prove it was bypassed";
}

TEST(RdmaLoopback, SoleRemoteCooledRailReturnsExplicitFailure) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv cooldown("DFKV_RDMA_REMOTE_RAIL_COOLDOWN_MS", "30000");
  ScopedEnv max_cooldown("DFKV_RDMA_REMOTE_RAIL_MAX_COOLDOWN_MS", "30000");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);

  const auto discovered = rdma::RdmaTopology::Discover(
      {}, rdma::RdmaDiscoveryPolicy::kActiveOnly);
  if (discovered.status != rdma::RdmaDiscoveryStatus::kOk ||
      discovered.devices.empty())
    GTEST_SKIP() << "no ACTIVE RDMA device";
  const char* configured = std::getenv("DFKV_RDMA_DEV");
  const std::string rail =
      configured && *configured && std::strchr(configured, ',') == nullptr
          ? configured
          : discovered.devices.front().name;
  ScopedEnv one_rail("DFKV_RDMA_DEV", rail.c_str());
  RdmaNode node("remote-sole-cooled");
  RdmaTransport transport(kMaxMsg, rail);
  PeerTopology topology;
  topology.peer_addr = node.addr;
  topology.peer_id = "remote-sole-cooled-peer";
  topology.generation = 1;
  topology.complete = true;
  topology.rails.push_back(PeerRailTopology{rail, true});
  transport.OnPeerTopology(topology);

  const BlockKey key = ToBlockKey(SelfHdr(), "remote-sole-cooled");
  const std::string value(4096, 'c');
  ASSERT_EQ(transport.Cache(node.addr, key, value.data(), value.size()),
            Status::kOk);
  ASSERT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 1u);
  const auto seeded = RdmaTransportTestPeer::CoolRemoteRail(
      &transport, topology.peer_id, 0);
  ASSERT_TRUE(seeded);
  ASSERT_EQ(seeded->completion.transition, RemoteRailTransition::kCooled);

  const std::string before = transport.MetricsText();
  std::string output;
  EXPECT_EQ(transport.Range(node.addr, key, 0, value.size(), &output, nullptr),
            Status::kNoCompatibleRail);
  const std::string after = transport.MetricsText();
  EXPECT_TRUE(output.empty());
  EXPECT_EQ(CounterVal(after, "dfkv_rdma_client_no_compatible_rail_total") -
                CounterVal(before,
                           "dfkv_rdma_client_no_compatible_rail_total"),
            1);
  EXPECT_EQ(CounterVal(after, "dfkv_rdma_endpoint_cache_hits_total"),
            CounterVal(before, "dfkv_rdma_endpoint_cache_hits_total"));
  EXPECT_EQ(CounterVal(after, "dfkv_rdma_endpoint_cache_misses_total"),
            CounterVal(before, "dfkv_rdma_endpoint_cache_misses_total"));
  EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 1u);
}

TEST(RdmaLoopback, StableIdentityTopologyFlapRetainsRemoteCooldown) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv cooldown("DFKV_RDMA_REMOTE_RAIL_COOLDOWN_MS", "30000");
  ScopedEnv max_cooldown("DFKV_RDMA_REMOTE_RAIL_MAX_COOLDOWN_MS", "30000");
  const auto discovered = rdma::RdmaTopology::Discover(
      {}, rdma::RdmaDiscoveryPolicy::kActiveOnly);
  if (discovered.status != rdma::RdmaDiscoveryStatus::kOk ||
      discovered.devices.empty())
    GTEST_SKIP() << "no ACTIVE RDMA device";
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  const std::string rail = discovered.devices.front().name;
  RdmaTransport transport(kMaxMsg, rail);
  PeerTopology topology;
  topology.peer_addr = "unused-peer-address";
  topology.peer_id = "stable-flapping-peer";
  topology.generation = 100;
  topology.complete = true;
  topology.rails.push_back(PeerRailTopology{rail, true});
  transport.OnPeerTopology(topology);
  const uint64_t first_publication =
      RdmaTransportTestPeer::PeerPublication(transport, topology.peer_addr);

  const auto seeded = RdmaTransportTestPeer::CoolRemoteRail(
      &transport, topology.peer_id, 0);
  ASSERT_TRUE(seeded);
  topology.generation = 7;
  topology.rails[0].healthy = false;
  transport.OnPeerTopology(topology);
  topology.generation = 999;
  topology.rails[0].healthy = true;
  transport.OnPeerTopology(topology);
  EXPECT_GT(
      RdmaTransportTestPeer::PeerPublication(transport, topology.peer_addr),
      first_publication)
      << "health-only snapshots may republish without resetting rail health";
  EXPECT_EQ(RdmaTransportTestPeer::RemoteAllowed(
                transport, topology.peer_id, {1}, seeded->now_us),
            (std::vector<uint8_t>{0}))
      << "health-only topology generations must not reset stable identity";
}


TEST(RdmaLoopback,
     LateReleaseCannotRepoolAcrossRemoveAndIdenticalReadd) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  const auto discovered = rdma::RdmaTopology::Discover(
      {}, rdma::RdmaDiscoveryPolicy::kActiveOnly);
  if (discovered.status != rdma::RdmaDiscoveryStatus::kOk ||
      discovered.devices.empty())
    GTEST_SKIP() << "no ACTIVE RDMA device";

  const char* configured = std::getenv("DFKV_RDMA_DEV");
  const std::string rail =
      configured && *configured && std::strchr(configured, ',') == nullptr
          ? configured
          : discovered.devices.front().name;
  ScopedEnv one_rail("DFKV_RDMA_DEV", rail.c_str());
  RdmaNode node("peer-reincarnation");
  RdmaTransport transport(kMaxMsg, rail);
  PeerTopology topology;
  topology.peer_addr = node.addr;
  topology.peer_id = "reincarnated-peer";
  topology.generation = 55;
  topology.complete = true;
  topology.rails.push_back(PeerRailTopology{rail, true});
  transport.OnPeerTopology(topology);
  const uint64_t first_publication =
      RdmaTransportTestPeer::PeerPublication(transport, node.addr);

  auto* old_active =
      RdmaTransportTestPeer::AcquireData(&transport, node.addr);
  ASSERT_NE(old_active, nullptr);
  ASSERT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 0u);

  topology.present = false;
  transport.OnPeerTopology(topology);
  topology.present = true;
  transport.OnPeerTopology(topology);
  EXPECT_GT(RdmaTransportTestPeer::PeerPublication(transport, node.addr),
            first_publication);

  const std::string before_release = transport.MetricsText();
  RdmaTransportTestPeer::ReleaseData(&transport, node.addr, old_active);
  const std::string after_release = transport.MetricsText();
  EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 0u);
  EXPECT_EQ(
      CounterVal(after_release,
                 "dfkv_rdma_client_stale_generation_reaps_total") -
          CounterVal(before_release,
                     "dfkv_rdma_client_stale_generation_reaps_total"),
      1);

  const std::string before_reacquire = transport.MetricsText();
  auto* current =
      RdmaTransportTestPeer::AcquireData(&transport, node.addr);
  ASSERT_NE(current, nullptr);
  const std::string after_reacquire = transport.MetricsText();
  EXPECT_EQ(CounterVal(after_reacquire,
                       "dfkv_rdma_endpoint_cache_hits_total"),
            CounterVal(before_reacquire,
                       "dfkv_rdma_endpoint_cache_hits_total"));
  EXPECT_EQ(CounterVal(after_reacquire,
                       "dfkv_rdma_endpoint_cache_misses_total") -
                CounterVal(before_reacquire,
                           "dfkv_rdma_endpoint_cache_misses_total"),
            1);
  RdmaTransportTestPeer::ReleaseData(&transport, node.addr, current);
  EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 1u);
}

TEST(RdmaLoopback,
     ReplaySafeEndpointRetryExcludesRailButPostedPutNeverReplays) {
  if (!HaveRdma())
    GTEST_SKIP() << "no RDMA device (load two rdma_rxe devices for this test)";
  const auto rails = ConfiguredTwoTestRails();
  if (!HaveConfiguredActiveRails(rails))
    GTEST_SKIP() << "set DFKV_RDMA_DEV to exactly two ACTIVE test devices";
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  RdmaTransport transport(kMaxMsg, rails[0] + "," + rails[1]);

  const auto replay_safe = RdmaTransportTestPeer::PrepareEndpointRetry(
      &transport, /*from_pool=*/false, /*replay_safe=*/true,
      /*request_posted=*/true, /*attempted_rail=*/0);
  EXPECT_TRUE(replay_safe.retry);
  EXPECT_TRUE(replay_safe.cross_rail);
  EXPECT_EQ(replay_safe.excluded, (std::vector<uint8_t>{1, 0}))
      << "a fresh replay-safe retry must not revisit the failed endpoint rail";

  const auto posted_put = RdmaTransportTestPeer::PrepareEndpointRetry(
      &transport, /*from_pool=*/true, /*replay_safe=*/false,
      /*request_posted=*/true, /*attempted_rail=*/0);
  EXPECT_FALSE(posted_put.retry);
  EXPECT_FALSE(posted_put.cross_rail);
  EXPECT_EQ(posted_put.excluded, (std::vector<uint8_t>{0, 0}));
}

// The bootstrap listener of a started server, without a verbs device: rail
// discovery and anchors are faked, the TCP accept loop and Serve() are real.
// Capability probes are answered before any QP is created, so they tell
// whether the listener is serving.
struct BootstrapListener {
  ScopedEnv configured_device{"DFKV_RDMA_DEV", nullptr};
  ScopedEnv recv_segment{"DFKV_RDMA_RECV_SEGMENT_SIZE", "1048576"};
  RdmaServer server{RdmaServer::Handler{}, kMaxMsg};

  BootstrapListener() {
    RdmaServerTestPeer::SetDiscovery(
        &server, [](const std::vector<std::string>&, rdma::RdmaDiscoveryPolicy) {
          rdma::RdmaDevInfo device;
          device.name = "fake_rail";
          device.active = true;
          rdma::RdmaDiscoveryResult result;
          result.devices = {device};
          return result;
        });
    RdmaServerTestPeer::SetInitializeAnchor(
        &server, [](const std::string&, void*, size_t) {
          return std::make_unique<rdma::RcEndpoint>();
        });
  }

  int Connect() const {
    return net::Dial("127.0.0.1:" + std::to_string(server.port()), 2000, 2000);
  }
  // Sends a capability probe on `fd`; true when the server answered it.
  static bool ProbeOn(int fd) {
    char request[rdma::kDevNameBytes];
    rdma::EncodeDevFrame(rdma::kV2ProbeDevice, 4u << 20, request);
    char reply[rdma::kV2ProbeReplyBytes];
    return net::WriteAll(fd, request, sizeof(request)) &&
           net::ReadAll(fd, reply, sizeof(reply)) &&
           rdma::ParseV2ProbeReply(reply);
  }
  // Blocks until the server has closed its end of `fd`. The server's
  // descriptor for the connection is free again by then, which matters to a
  // test that counts free descriptors.
  static void WaitServerClosed(int fd) {
    char byte;
    while (::recv(fd, &byte, 1, 0) > 0) {
    }
  }
  bool Probe() const {
    const int fd = Connect();
    if (fd < 0) return false;
    const bool answered = ProbeOn(fd);
    WaitServerClosed(fd);
    ::close(fd);
    return answered;
  }
};

template <class Predicate>
bool EventuallyTrue(Predicate predicate, int timeout_ms = 3000) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return true;
}

TEST(RdmaBootstrapListener, ThreadThatCannotStartRefusesOnlyThatConnection) {
  BootstrapListener listener;
  ASSERT_EQ(listener.server.Start(0), Status::kOk);
  ASSERT_TRUE(listener.Probe());

  net::g_fail_thread_starts_for_test.store(1);
  EXPECT_FALSE(listener.Probe()) << "the connection without a thread is closed";
  net::g_fail_thread_starts_for_test.store(0);
  EXPECT_EQ(listener.server.ThreadStartFailures(), 1u);
  EXPECT_EQ(listener.server.PendingHandshakes(), 0u);

  EXPECT_TRUE(listener.Probe()) << "the server keeps serving";
  EXPECT_NE(listener.server.MetricsText().find(
                "dfkv_rdma_thread_start_failures_total 1\n"),
            std::string::npos);
  listener.server.Stop();
}

TEST(RdmaBootstrapListener, PendingHandshakeLimitRefusesAndRecovers) {
  BootstrapListener listener;
  RdmaServerTestPeer::SetMaxPendingHandshakes(&listener.server, 2);
  ASSERT_EQ(listener.server.Start(0), Status::kOk);

  // Two peers that connect and send nothing use the limit up.
  const int silent_a = listener.Connect();
  const int silent_b = listener.Connect();
  ASSERT_GE(silent_a, 0);
  ASSERT_GE(silent_b, 0);
  ASSERT_TRUE(EventuallyTrue(
      [&] { return listener.server.PendingHandshakes() == 2; }));

  EXPECT_FALSE(listener.Probe());
  EXPECT_EQ(listener.server.RejectedConnections(), 1u);
  EXPECT_EQ(listener.server.PendingHandshakes(), 2u)
      << "a refused connection takes no slot";

  // One of them goes away: its slot is free again.
  ::close(silent_a);
  ASSERT_TRUE(EventuallyTrue(
      [&] { return listener.server.PendingHandshakes() == 1; }));
  EXPECT_TRUE(listener.Probe());
  ASSERT_TRUE(EventuallyTrue(
      [&] { return listener.server.PendingHandshakes() == 1; }))
      << "an answered probe gives its slot back";
  EXPECT_EQ(listener.server.RejectedConnections(), 1u);
  const std::string metrics = listener.server.MetricsText();
  EXPECT_NE(metrics.find("dfkv_rdma_rejected_connections_total 1\n"),
            std::string::npos);
  EXPECT_NE(metrics.find("dfkv_rdma_pending_handshakes 1\n"),
            std::string::npos);
  ::close(silent_b);
  listener.server.Stop();
}

TEST(RdmaBootstrapListener, PendingHandshakeLimitZeroIsUnlimited) {
  BootstrapListener listener;
  RdmaServerTestPeer::SetMaxPendingHandshakes(&listener.server, 0);
  ASSERT_EQ(listener.server.Start(0), Status::kOk);
  std::vector<int> silent;
  for (int i = 0; i < 8; ++i) silent.push_back(listener.Connect());
  ASSERT_TRUE(EventuallyTrue(
      [&] { return listener.server.PendingHandshakes() == 8; }));
  EXPECT_TRUE(listener.Probe());
  EXPECT_EQ(listener.server.RejectedConnections(), 0u);
  for (int fd : silent) ::close(fd);
  listener.server.Stop();
}

TEST(RdmaBootstrapListener, PendingHandshakeLimitComesFromTheEnvironment) {
  {
    ScopedEnv limit("DFKV_RDMA_MAX_PENDING_HANDSHAKES", "7");
    RdmaServer server(RdmaServer::Handler{}, kMaxMsg);
    EXPECT_EQ(RdmaServerTestPeer::MaxPendingHandshakes(server), 7u);
  }
  {
    ScopedEnv limit("DFKV_RDMA_MAX_PENDING_HANDSHAKES", "-1");
    RdmaServer server(RdmaServer::Handler{}, kMaxMsg);
    EXPECT_EQ(RdmaServerTestPeer::MaxPendingHandshakes(server), 4096u)
        << "a value that is not a number keeps the default";
  }
  {
    ScopedEnv limit("DFKV_RDMA_MAX_PENDING_HANDSHAKES", "0");
    RdmaServer server(RdmaServer::Handler{}, kMaxMsg);
    EXPECT_EQ(RdmaServerTestPeer::MaxPendingHandshakes(server), 0u);
  }
}

// A peer that sends one byte at a time stays inside any per-recv timeout. The
// bootstrap frames have to be complete by the deadline however they arrive.
TEST(RdmaBootstrapListener, SlowDripPeerIsDroppedAtTheBootstrapDeadline) {
  BootstrapListener listener;
  RdmaServerTestPeer::SetBootstrapReadMs(&listener.server, 300);
  ASSERT_EQ(listener.server.Start(0), Status::kOk);
  const int fd = listener.Connect();
  ASSERT_GE(fd, 0);

  char frame[rdma::kDevNameBytes];
  rdma::EncodeDevFrame(rdma::kV2ProbeDevice, 4u << 20, frame);
  const auto started = std::chrono::steady_clock::now();
  bool closed = false;
  // At 100 ms a byte the 32-byte frame would take 3.2 s to arrive.
  for (size_t i = 0; i < sizeof(frame) && !closed; ++i) {
    if (!net::WriteAll(fd, frame + i, 1)) {
      closed = true;
      break;
    }
    pollfd waiting{fd, POLLIN, 0};
    if (::poll(&waiting, 1, 100) > 0) {
      char byte;
      closed = ::recv(fd, &byte, 1, 0) <= 0;
    }
  }
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - started);
  ::close(fd);
  EXPECT_TRUE(closed);
  EXPECT_LT(elapsed.count(), 3000);
  EXPECT_TRUE(EventuallyTrue(
      [&] { return listener.server.PendingHandshakes() == 0; }));
  listener.server.Stop();
}

// Stop() joins every connection thread. One still in its handshake waited
// for the peer's frames until the bootstrap deadline (10 s), so a peer that
// had only connected held a server's stop up by that long. Stop now shuts
// those sockets down.
TEST(RdmaBootstrapListener, StopDoesNotWaitForHandshakesInFlight) {
  BootstrapListener listener;
  ASSERT_EQ(listener.server.Start(0), Status::kOk);
  std::vector<int> peers;
  for (int i = 0; i < 3; ++i) {
    peers.push_back(listener.Connect());
    ASSERT_GE(peers.back(), 0);
  }
  ASSERT_TRUE(EventuallyTrue(
      [&] { return listener.server.PendingHandshakes() == 3; }));
  const auto started = std::chrono::steady_clock::now();
  listener.server.Stop();
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - started);
  std::printf("STOP_WITH_HANDSHAKES %lld ms\n",
              static_cast<long long>(elapsed.count()));
  for (const int fd : peers) ::close(fd);
  EXPECT_LT(elapsed.count(), 1000) << "waited for the bootstrap deadline";
}

// With no descriptor left accept() fails at once, every time, while the peer
// stays queued. The listener has to wait between attempts instead of spinning,
// serve the queued peer once descriptors are back, and still stop promptly.
TEST(RdmaBootstrapListener, DescriptorExhaustionBacksOffAndRecovers) {
  if (testing_fd::kUnderThreadSanitizer)
    GTEST_SKIP() << testing_fd::kTsanNoThreadsAfterFork;
  const int result = testing_fd::RunInChild([] {
    BootstrapListener listener;
    if (listener.server.Start(0) != Status::kOk) return 10;
    if (!listener.Probe()) return 11;

    // Dial() needs a descriptor, so open the sockets before taking them away.
    const int queued = listener.Connect();
    const int second = ::socket(AF_INET, SOCK_STREAM, 0);
    if (queued < 0 || second < 0) return 12;
    // `queued` may already have its thread; wait until it is counted so that
    // the accept loop is back in accept() when the limit drops.
    if (!EventuallyTrue(
            [&] { return listener.server.PendingHandshakes() == 1; }))
      return 13;

    testing_fd::NoFreeDescriptors exhausted;
    if (!exhausted.active()) return 14;
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(static_cast<uint16_t>(listener.server.port()));
    if (::connect(second, reinterpret_cast<sockaddr*>(&address),
                  sizeof(address)) != 0)
      return 15;

    const double cpu_before = testing_fd::ProcessCpuSeconds();
    const auto wall_before = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    const double cpu_used = testing_fd::ProcessCpuSeconds() - cpu_before;
    const uint64_t failures = listener.server.AcceptFailures();
    if (failures == 0) return 16;   // the failure was not seen or not counted
    // One attempt per backoff, however long this thread actually slept.
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - wall_before)
            .count();
    if (failures > static_cast<uint64_t>(elapsed_ms) /
                           net::AcceptGuard::kBackoffMs + 2)
      return 17;  // retried without waiting
    if (cpu_used > 0.2) return 18;  // spun instead of sleeping

    exhausted.Restore();
    if (!BootstrapListener::ProbeOn(second)) return 19;  // served afterwards
    if (!BootstrapListener::ProbeOn(queued)) return 20;
    BootstrapListener::WaitServerClosed(second);
    BootstrapListener::WaitServerClosed(queued);

    // Stopping while the loop is backing off must not wait out anything long.
    const int third = ::socket(AF_INET, SOCK_STREAM, 0);
    if (third < 0) return 21;
    testing_fd::NoFreeDescriptors exhausted_again;
    if (!exhausted_again.active()) return 22;
    if (::connect(third, reinterpret_cast<sockaddr*>(&address),
                  sizeof(address)) != 0)
      return 23;
    if (!EventuallyTrue(
            [&] { return listener.server.AcceptFailures() > failures; }))
      return 24;
    const auto stop_started = std::chrono::steady_clock::now();
    listener.server.Stop();
    if (std::chrono::steady_clock::now() - stop_started >
        std::chrono::seconds(2))
      return 25;
    return 0;
  });
  EXPECT_EQ(result, 0) << "step code from the child, see the test body";
}

// The RDMA client's settings set to garbage: the transport is built, a PUT
// and a GET work, nothing is thrown, and each name is reported once at WARN
// however many transports are built and connections opened.
TEST(RdmaLoopback, ClientSettingsSetToGarbageAreReportedOnceAndHarmless) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const std::vector<std::pair<const char*, const char*>> garbage = {
      {"DFKV_RDMA_CONNECT_MS", "soon"},
      {"DFKV_RDMA_IO_MS", "later"},
      {"DFKV_RDMA_OP_TIMEOUT_MS", "-5"},        // read as 5000
      {"DFKV_RDMA_BATCH_OP_TIMEOUT_MS", "long"},
      {"DFKV_RDMA_KEEPALIVE_MS", "15s"},
      {"DFKV_RDMA_DEPTH", "deep"},
      {"DFKV_RDMA_RAIL_CREDITS", "many"},
      {"DFKV_RDMA_RAIL_COOLDOWN_MS", "5s"},
      {"DFKV_RDMA_POOL_MAX", "big"},
      {"DFKV_RDMA_RESOURCE_ACQUIRE_MS", "10s"},
      {"DFKV_RDMA_INLINE_PUT_MAX_BYTES", "4M"},
      {"DFKV_RDMA_MAX_BLOCK_BYTES", "big"},
      {"DFKV_RDMA_BUSY_POLL", "off"},           // read as on
  };
  RdmaNode node("garbage-env");
  for (const auto& [name, value] : garbage) ::setenv(name, value, 1);
  dfkv::envparse::ForgetWarningsForTest();
  testing::internal::CaptureStderr();
  for (int round = 0; round < 2; ++round) {
    std::unique_ptr<RdmaTransport> rt;
    ASSERT_NO_THROW(rt = std::make_unique<RdmaTransport>(kMaxMsg));
    KVClient c({{"n", node.addr}}, SelfHdr(), rt.get());
    const std::string value(8192, 'g');
    for (int i = 0; i < 8; ++i) {
      const std::string key = "garbage-" + std::to_string(round * 100 + i);
      ASSERT_TRUE(c.Put(key, value.data(), value.size())) << key;
      std::string out(value.size(), 0);
      ASSERT_TRUE(c.Get(key, &out[0], out.size())) << key;
      ASSERT_EQ(out, value);
    }
  }
  const std::string log = testing::internal::GetCapturedStderr();
  for (const auto& [name, value] : garbage) ::unsetenv(name);
  for (const auto& [name, value] : garbage) {
    const std::string quoted = std::string(name) + "='" + value + "'";
    size_t lines = 0;
    for (size_t at = log.find(quoted); at != std::string::npos;
         at = log.find(quoted, at + 1))
      ++lines;
    EXPECT_EQ(lines, 1u) << quoted << "\n" << log;
  }
  EXPECT_NE(log.find("DFKV_RDMA_OP_TIMEOUT_MS='-5': not a plain decimal "
                     "number from 0 to 2147483647; using 5000, as before"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("DFKV_RDMA_BUSY_POLL='off': read as on, as before"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("DFKV_RDMA_DEPTH='deep'"), std::string::npos) << log;
  EXPECT_NE(log.find("using 4, as before"), std::string::npos) << log;
}

// The RDMA client's settings whose 0 is documented to mean something are
// clean at 0: no WARN, and the transport works.
TEST(RdmaLoopback, ClientSettingsAtAMeaningfulZeroAreNotGarbage) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const char* const names[] = {
      "DFKV_RDMA_OP_TIMEOUT_MS",         // 0 = no deadline
      "DFKV_RDMA_BATCH_OP_TIMEOUT_MS",   // 0 = follows DFKV_RDMA_OP_TIMEOUT_MS
      "DFKV_RDMA_KEEPALIVE_MS",          // 0 = no keepalive
      "DFKV_RDMA_POOL_MAX",              // 0 = no override
      "DFKV_RDMA_INLINE_PUT_MAX_BYTES",  // 0 = inline PUT off
  };
  RdmaNode node("meaningful-zero-env");
  for (const char* name : names) ::setenv(name, "0", 1);
  dfkv::envparse::ForgetWarningsForTest();
  testing::internal::CaptureStderr();
  {
    std::unique_ptr<RdmaTransport> rt;
    ASSERT_NO_THROW(rt = std::make_unique<RdmaTransport>(kMaxMsg));
    KVClient c({{"n", node.addr}}, SelfHdr(), rt.get());
    const std::string value(8192, 'z');
    ASSERT_TRUE(c.Put("zero-0", value.data(), value.size()));
    std::string out(value.size(), 0);
    ASSERT_TRUE(c.Get("zero-0", &out[0], out.size()));
    ASSERT_EQ(out, value);
  }
  const std::string log = testing::internal::GetCapturedStderr();
  for (const char* name : names) ::unsetenv(name);
  EXPECT_EQ(log.find("invalid DFKV_"), std::string::npos) << log;
}

// The QP transport retry settings come from DFKV_RDMA_QP_* (read once per
// process; each test is its own process), within their ranges.
// QpRetry keeps what it read first in the process; ReadQpRetry is the same
// reading, done now, so the test holds in a process that opened QPs before.
TEST(RdmaQpRetry, SettingsComeFromTheEnvironment) {
  ScopedEnv timeout("DFKV_RDMA_QP_TIMEOUT", "17");
  ScopedEnv retries("DFKV_RDMA_QP_RETRY_CNT", "9");  // above 7: the default
  ScopedEnv rnr("DFKV_RDMA_QP_RNR_RETRY", "3");
  const rdma::QpRetrySettings settings = rdma::ReadQpRetry();
  EXPECT_EQ(settings.timeout, 17);
  EXPECT_EQ(settings.retry_cnt, 7);
  EXPECT_EQ(settings.rnr_retry, 3);
}

// A client endpoint is never the target of an RDMA READ or WRITE, so its QP
// grants neither: a peer that tries one (here even into a buffer
// registered for remote writes) gets a remote access error.
TEST(RdmaRemoteAccess, AClientQpRefusesRemoteWritesAndReads) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const char* devs = std::getenv("DFKV_RDMA_DEV");
  const std::string first =
      devs ? std::string(devs).substr(0, std::string(devs).find(',')) : "";
  const char* dev = first.empty() ? nullptr : first.c_str();
  for (const bool allowed : {true, false}) {
    rdma::RcEndpoint target;
    rdma::RcEndpoint peer;
    target.set_remote_access(allowed);
    ASSERT_TRUE(target.Open(dev, 4096, 1));
    ASSERT_TRUE(peer.Open(dev, 4096, 1));
    const rdma::QpInfo ti = target.Local(), pi = peer.Local();
    ASSERT_TRUE(target.Connect(pi));
    ASSERT_TRUE(peer.Connect(ti));
    std::vector<char> region(4096, 'r');
    ibv_mr* mr = target.RegisterTransient(region.data(), region.size(),
                                          /*remote_write=*/true);
    ASSERT_NE(mr, nullptr);
    std::vector<char> source(64, 'w');
    ibv_mr* source_mr = peer.RegisterTransient(source.data(), source.size());
    ASSERT_NE(source_mr, nullptr);
    ASSERT_TRUE(peer.PostWrite(0, source.data(), source.size(), source_mr,
                               reinterpret_cast<uint64_t>(region.data()),
                               mr->rkey));
    ibv_wc wc{};
    ASSERT_EQ(peer.WaitComp(&wc, 1, 10000), 1);
    if (allowed) {
      EXPECT_EQ(wc.status, IBV_WC_SUCCESS);
      EXPECT_EQ(region[0], 'w');
    } else {
      // Refused: an invalid request (what a QP without the permission
      // answers on the devices measured) or an access error.
      EXPECT_TRUE(wc.status == IBV_WC_REM_INV_REQ_ERR ||
                  wc.status == IBV_WC_REM_ACCESS_ERR)
          << ibv_wc_status_str(wc.status);
      EXPECT_EQ(region[0], 'r');
    }
    peer.ReleaseTransient(source_mr);
    target.ReleaseTransient(mr);
  }
}

// The same on the connections the transport makes: neither data nor control
// lanes grant remote access (devices may not report LOCAL_WRITE back).
TEST(RdmaRemoteAccess, TheTransportsConnectionsGrantNoRemoteAccess) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("remote_access_conn");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  const std::string value = "v";
  ASSERT_TRUE(c.Put("remote-access", value.data(), value.size()));
  for (const bool control : {false, true}) {
    auto* conn =
        control ? RdmaTransportTestPeer::AcquireControl(&rt, node.addr)
                : RdmaTransportTestPeer::AcquireData(&rt, node.addr);
    ASSERT_NE(conn, nullptr);
    const int flags = RdmaTransportTestPeer::QpAccessFlags(conn);
    ASSERT_GE(flags, 0) << "ibv_query_qp failed";
    EXPECT_EQ(flags & (IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                       IBV_ACCESS_REMOTE_ATOMIC),
              0)
        << (control ? "control" : "data") << " flags=" << flags;
    if (control)
      RdmaTransportTestPeer::ReleaseControl(&rt, node.addr, conn);
    else
      RdmaTransportTestPeer::ReleaseData(&rt, node.addr, conn);
  }
}

// Registered memory is pinned once per rail, for local access only: PUTs
// from it and GETs (RDMA READs) into it find that registration, on new and
// pooled connections alike, and register nothing more.
TEST(RdmaRemoteAccess, RegisteredMemoryIsPinnedOncePerRail) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("pool_once");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  size_t rails = 0;
  {
    const std::string text = rt.MetricsText();
    const std::string row = "\ndfkv_rdma_client_rail_selections_total{";
    for (size_t at = text.find(row); at != std::string::npos;
         at = text.find(row, at + 1))
      ++rails;
  }
  ASSERT_GE(rails, 1u);
  std::vector<char> pool(1 << 20);
  const uint64_t regs = rdma::RcEndpoint::PoolMrRegistrations();
  const uint64_t adhoc = rdma::RcEndpoint::AdhocUserMrTotal();
  ASSERT_TRUE(c.RegisterMemory(pool.data(), pool.size()));
  EXPECT_EQ(rdma::RcEndpoint::PoolMrRegistrations() - regs, rails);
  constexpr size_t kValue = 128 * 1024;  // read back by RDMA READ
  for (size_t i = 0; i < 4; ++i) {
    char* slot = pool.data() + i * kValue;
    std::memset(slot, static_cast<int>('a' + i), kValue);
    const std::string key = "pool-once-" + std::to_string(i);
    ASSERT_TRUE(c.Put(key, slot, kValue));
    char* out = pool.data() + (1 << 19) + i * kValue;
    ASSERT_TRUE(c.Get(key, out, kValue));
    EXPECT_EQ(std::memcmp(out, slot, kValue), 0);
  }
  EXPECT_EQ(rdma::RcEndpoint::PoolMrRegistrations() - regs, rails)
      << "a connection registered the pool again";
  EXPECT_EQ(rdma::RcEndpoint::AdhocUserMrTotal(), adhoc)
      << "an in-pool buffer took an ad-hoc registration";
}

// A peer whose rail matching our first one does not answer (here: the
// server serves only our second rail), on a static ring (no MDS identity).
// After the first failure on that rail, later operations go straight to the
// rail that works instead of each trying the broken one first.
namespace {
uint64_t RailCounter(const std::string& text, const std::string& name,
                     const std::string& dev) {
  const std::string prefix = name + "{dev=\"" + dev + "\"} ";
  const size_t at = text.find("\n" + prefix);
  if (at == std::string::npos) return 0;
  return std::stoull(text.substr(at + 1 + prefix.size()));
}
}  // namespace

void RunPeerRailDown(bool named) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const char* devs = std::getenv("DFKV_RDMA_DEV");
  const std::string list = devs ? devs : "";
  const size_t comma = list.find(',');
  if (comma == std::string::npos)
    GTEST_SKIP() << "needs DFKV_RDMA_DEV with two rails";
  const std::string first = list.substr(0, comma);
  const std::string second = list.substr(comma + 1);
  std::unique_ptr<RdmaNode> node;
  {
    ScopedEnv only_second("DFKV_RDMA_DEV", second.c_str());
    node = std::make_unique<RdmaNode>("peer_rail_down");
  }
  RdmaTransport rt(kMaxMsg);
  if (named) rt.OnPeerTopology(PublishedPeer(node->addr, "rail-down-peer"));
  KVClient c({{"n", node->addr}}, SelfHdr(), &rt);
  constexpr int kOps = 20;
  int ok = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kOps; ++i) {
    const std::string value = "v" + std::to_string(i);
    ok += c.Put("rail-down-" + std::to_string(i), value.data(), value.size()) ? 1 : 0;
  }
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
  const std::string text = rt.MetricsText();
  const uint64_t broken_attempts =
      RailCounter(text, "dfkv_rdma_client_rail_selections_total", first);
  std::printf("[ rail-down ] named=%d ok=%d/%d ms=%.1f selections(%s)=%llu\n",
              named ? 1 : 0, ok, kOps, ms, first.c_str(),
              (unsigned long long)broken_attempts);
  EXPECT_EQ(ok, kOps);
  // The first operation's attempt and the confirming dial; no other.
  EXPECT_LE(broken_attempts, 2u) << "every operation tried the broken rail";
  EXPECT_NE(text.find("\ndfkv_rdma_client_rail_refusals_total 1\n"),
            std::string::npos)
      << "the refusal was not recorded";
  EXPECT_EQ(CounterVal(text, "dfkv_rdma_client_peer_busy_total"), 1)
      << "the confirming dial is not a busy peer";
}

// A confirmed refusal never cools a peer's last usable rail: of two rails,
// the second refusal is not charged.
TEST(RdmaPeerRailDown, ARefusalNeverCoolsAPeersLastRail) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const char* devs = std::getenv("DFKV_RDMA_DEV");
  const std::string list = devs ? devs : "";
  if (list.find(',') == std::string::npos)
    GTEST_SKIP() << "needs DFKV_RDMA_DEV with two rails";
  RdmaNode node("last-rail");
  RdmaTransport rt(kMaxMsg);
  rt.OnPeerTopology(PublishedPeer(node.addr, "last-rail-peer"));
  EXPECT_TRUE(RdmaTransportTestPeer::ChargeRefusal(&rt, node.addr,
                                                   "last-rail-peer", 0));
  EXPECT_FALSE(RdmaTransportTestPeer::ChargeRefusal(&rt, node.addr,
                                                    "last-rail-peer", 1));
  EXPECT_EQ(CounterVal(rt.MetricsText(), "dfkv_rdma_client_rail_refusals_total"),
            1);
}

// A confirming dial that does not get through within its short limit says
// nothing about the node: it is not held unreachable, and the next
// operation to it connects.
TEST(RdmaPeerRailDown, ATimedOutRefusalCheckDoesNotHoldTheNode) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("refusal-check-timeout");
  RdmaTransport rt(kMaxMsg);
  for (const bool named : {false, true}) {
    // Static: the capability probe's dial; named (capabilities cached once
    // a connection exists): the bootstrap dial.
    if (named) {
      rt.OnPeerTopology(PublishedPeer(node.addr, "timeout-peer"));
      auto* warm = RdmaTransportTestPeer::AcquireData(&rt, node.addr);
      ASSERT_NE(warm, nullptr);
      RdmaTransportTestPeer::ReleaseData(&rt, node.addr, warm);
    }
    RdmaTransportTestPeer::FailDials(&rt, 1);
    EXPECT_EQ(RdmaTransportTestPeer::AcquireRefusalCheck(&rt, node.addr, 0),
              nullptr);
    const std::string text = rt.MetricsText();
    EXPECT_EQ(CounterVal(text, "dfkv_rdma_client_peer_unreachable_total"), 0)
        << (named ? "named" : "static");
    EXPECT_EQ(CounterVal(text, "dfkv_rdma_client_peer_busy_total"), 0);
    const std::vector<BlockKey> keys = {ToBlockKey(SelfHdr(), "absent")};
    std::vector<char> exists;
    EXPECT_NE(rt.ExistMany(node.addr, keys, &exists)[0], Status::kIOError)
        << (named ? "named" : "static");
  }
}

TEST(RdmaPeerRailDown, AStaticPeersBrokenRailIsAvoidedAfterItFails) {
  RunPeerRailDown(/*named=*/false);
}

TEST(RdmaPeerRailDown, AnMdsNamedPeersBrokenRailIsAvoidedAfterItFails) {
  RunPeerRailDown(/*named=*/true);
}

// A peer out of handshake slots closes a bootstrap before replying, as one
// refusing the device does. The operation goes on on the other rail; the
// confirming dial on the closed rail then gets through, so nothing is
// charged.
TEST(RdmaPeerRailDown, AFullPeerIsNotTakenForARefusal) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const char* devs = std::getenv("DFKV_RDMA_DEV");
  const std::string list = devs ? devs : "";
  const size_t comma = list.find(',');
  if (comma == std::string::npos)
    GTEST_SKIP() << "needs DFKV_RDMA_DEV with two rails";
  const std::string first = list.substr(0, comma);
  ScopedEnv primary("DFKV_RDMA_PRIMARY_DEV", first.c_str());
  ScopedEnv numa("DFKV_RDMA_NUMA", "0");
  RdmaNode node("full-not-refusing");
  RdmaTransport rt(kMaxMsg);
  RdmaTransportTestPeer::SetPeerBusyRetryUs(&rt, 200 * 1000);
  // A named peer (its capabilities are cached) and a data connection first:
  // the control connection below is then a bootstrap with our device frame,
  // not a capability probe.
  rt.OnPeerTopology(PublishedPeer(node.addr, "full-peer"));
  auto* warm = RdmaTransportTestPeer::AcquireData(&rt, node.addr);
  ASSERT_NE(warm, nullptr);
  RdmaTransportTestPeer::ReleaseData(&rt, node.addr, warm);
  for (int i = 0; i < 500 && node.rsrv->PendingHandshakes() != 0; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  ASSERT_EQ(node.rsrv->PendingHandshakes(), 0u);
  RdmaServerTestPeer::SetMaxPendingHandshakes(node.rsrv.get(), 1);
  // A peer that connects and sends nothing takes the only slot, and lets go
  // once the server has refused the client once.
  const int silent = net::Dial(node.addr, 2000, 2000);
  ASSERT_GE(silent, 0);
  for (int i = 0; i < 500 && node.rsrv->PendingHandshakes() != 1; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  ASSERT_EQ(node.rsrv->PendingHandshakes(), 1u);
  std::thread releasing([&] {
    while (node.rsrv->RejectedConnections() == 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ::close(silent);
  });
  const std::vector<BlockKey> keys = {ToBlockKey(SelfHdr(), "absent")};
  std::vector<char> exists;
  const Status status = rt.ExistMany(node.addr, keys, &exists)[0];
  releasing.join();
  EXPECT_NE(status, Status::kIOError);
  const std::string text = rt.MetricsText();
  EXPECT_EQ(CounterVal(text, "dfkv_rdma_client_peer_busy_total"), 1);
  EXPECT_EQ(CounterVal(text, "dfkv_rdma_client_rail_refusals_total"), 0);
  EXPECT_EQ(CounterVal(text, "dfkv_rdma_client_remote_rail_cooldowns_total"),
            0);
}

namespace {
std::atomic<bool> g_port_down{false};
std::string g_port_down_dev;  // set before the override is installed
int FakePortState(const std::string& name) {
  return name == g_port_down_dev && g_port_down.load() ? 0 : 1;
}
struct PortOverride {
  PortOverride() {
    rdma::PortWatcher::OverrideForTest().store(&FakePortState);
  }
  ~PortOverride() { rdma::PortWatcher::OverrideForTest().store(nullptr); }
};
bool WaitForMetric(RdmaTransport& rt, const std::string& line) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (std::chrono::steady_clock::now() < deadline) {
    if (rt.MetricsText().find("\n" + line + "\n") != std::string::npos)
      return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return false;
}
}  // namespace

TEST(RdmaPortState, ARailWhosePortGoesDownLeavesSelectionAndComesBack) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const char* devs = std::getenv("DFKV_RDMA_DEV");
  const std::string list = devs ? devs : "";
  const size_t comma = list.find(',');
  if (comma == std::string::npos)
    GTEST_SKIP() << "needs DFKV_RDMA_DEV with two rails";
  const std::string first = list.substr(0, comma);
  RdmaNode node("port_state");
  g_port_down_dev = first;
  g_port_down = false;
  PortOverride override_ports;
  ScopedEnv primary("DFKV_RDMA_PRIMARY_DEV", first.c_str());
  ScopedEnv check("DFKV_RDMA_PORT_CHECK_MS", "20");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  const auto put_all = [&](const std::string& tag) {
    int ok = 0;
    for (int i = 0; i < 10; ++i) {
      const std::string value = tag + std::to_string(i);
      ok += c.Put("port-" + value, value.data(), value.size()) ? 1 : 0;
    }
    return ok;
  };
  const auto on_first = [&] {
    return RailCounter(rt.MetricsText(),
                       "dfkv_rdma_client_rail_selections_total", first);
  };
  ASSERT_EQ(put_all("up"), 10);
  const uint64_t before = on_first();
  EXPECT_GE(before, 1u) << "the primary rail was not used";

  g_port_down = true;
  ASSERT_TRUE(WaitForMetric(rt, "dfkv_rdma_client_rails_port_down 1"));
  EXPECT_EQ(put_all("down"), 10);
  EXPECT_EQ(on_first(), before) << "a rail whose port is down was selected";
  EXPECT_NE(rt.MetricsText().find(
                "\ndfkv_rdma_client_port_down_events_total 1\n"),
            std::string::npos);

  g_port_down = false;
  ASSERT_TRUE(WaitForMetric(rt, "dfkv_rdma_client_rails_port_down 0"));
  EXPECT_EQ(put_all("again"), 10);
  EXPECT_GT(on_first(), before) << "the rail did not come back";
}

// With one credit per rail and 16 threads, operations used to give up after
// 10 ms of waiting for a credit (25 to 57 of 320 PUTs failed in a run that
// took about 0.1 s). The wait is now one budget per operation; given enough
// of it, none fails.
TEST(RdmaRailAdmission, OperationsWaitForCreditsWithinTheirBudget) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("rail_admission");
  ScopedEnv credits("DFKV_RDMA_RAIL_CREDITS", "1");
  ScopedEnv budget("DFKV_RDMA_RAIL_BACKPRESSURE_MS", "2000");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  std::atomic<int> failed{0};
  const std::string value(200 << 10, 'x');
  std::vector<std::thread> threads;
  for (int t = 0; t < 16; ++t)
    threads.emplace_back([&, t] {
      for (int i = 0; i < 20; ++i)
        if (!c.Put("bp-" + std::to_string(t) + "-" + std::to_string(i),
                   value.data(), value.size()))
          ++failed;
    });
  for (auto& thread : threads) thread.join();
  EXPECT_EQ(failed.load(), 0);
}

// A rail's first latency sample was its connection setup (about 12 ms against
// 0.2 ms for an operation), which kept that rail unused for good.
TEST(RdmaRailAdmission, ARailsLatencyExcludesConnectionSetup) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const char* devs = std::getenv("DFKV_RDMA_DEV");
  const std::string list = devs ? devs : "";
  const size_t comma = list.find(',');
  if (comma == std::string::npos)
    GTEST_SKIP() << "needs DFKV_RDMA_DEV with two rails";
  RdmaNode node("rail_latency");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  for (int i = 0; i < 8; ++i) {
    const std::string value = "l" + std::to_string(i);
    ASSERT_TRUE(c.Put("lat-" + value, value.data(), value.size()));
  }
  const std::string text = rt.MetricsText();
  std::vector<uint64_t> ewma;
  for (const std::string dev : {list.substr(0, comma), list.substr(comma + 1)}) {
    EXPECT_GE(RailCounter(text, "dfkv_rdma_client_rail_selections_total", dev),
              1u) << dev;
    ewma.push_back(
        RailCounter(text, "dfkv_rdma_client_rail_latency_ewma_us", dev));
  }
  // A relative bound, so a slow build (sanitizers) slows both alike: with
  // setup counted the two differed some 70 times.
  const uint64_t low = std::max<uint64_t>(100, std::min(ewma[0], ewma[1]));
  EXPECT_LT(std::max(ewma[0], ewma[1]), 20 * low)
      << ewma[0] << " vs " << ewma[1] << " us";
}

// A rail out of its cooldown is probed by the background thread, not by the
// next application operation, and comes back without traffic.
TEST(RdmaRailAdmission, ACooledRailIsProbedInTheBackground) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const char* devs = std::getenv("DFKV_RDMA_DEV");
  const std::string list = devs ? devs : "";
  const size_t comma = list.find(',');
  if (comma == std::string::npos)
    GTEST_SKIP() << "needs DFKV_RDMA_DEV with two rails";
  const std::string first = list.substr(0, comma);
  RdmaNode node("rail_probe");
  ScopedEnv primary("DFKV_RDMA_PRIMARY_DEV", first.c_str());
  ScopedEnv threshold("DFKV_RDMA_RAIL_ERROR_THRESHOLD", "1");
  ScopedEnv cooldown("DFKV_RDMA_RAIL_COOLDOWN_MS", "300");
  ScopedEnv tick("DFKV_RDMA_PORT_CHECK_MS", "20");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  const auto put = [&](const std::string& key) {
    return c.Put(key, key.data(), key.size());
  };
  ASSERT_TRUE(put("probe-warm"));
  {
    // The first attempt of this PUT fails on the first rail as a local
    // fault: the rail is cooled and the PUT is served by the other one.
    ScopedEnv inject("DFKV_RDMA_TEST_LOCAL_RAIL_FAILURE_ATTEMPTS", "1");
    ASSERT_TRUE(put("probe-fail"));
  }
  const auto selections = [&] {
    return RailCounter(rt.MetricsText(),
                       "dfkv_rdma_client_rail_selections_total", first);
  };
  const uint64_t cooled_at = selections();
  ASSERT_TRUE(WaitForMetric(rt, "dfkv_rdma_client_rail_background_probes_total 1"))
      << "no background probe";
  ASSERT_TRUE(WaitForMetric(
      rt, "dfkv_rdma_client_rail_quarantined{dev=\"" + first + "\"} 0"));
  EXPECT_EQ(RailCounter(rt.MetricsText(),
                        "dfkv_rdma_client_rail_recoveries_total", first),
            1u);
  EXPECT_EQ(selections(), cooled_at + 1) << "only the probe used the rail";
  ASSERT_TRUE(put("probe-after"));
  EXPECT_EQ(selections(), cooled_at + 2);
}

namespace {
// RDS-09: what one GET's exact READ registration costs the server. Every
// pull of a RAM-tier value registers that value (ibv_reg_mr) when the READY
// goes out and deregisters it on RELEASE. Opt-in: set DFKV_RDMA_MR_BENCH to
// the iterations per size.
TEST(RdmaMrCost, ReadRegistrationPerGetBench) {
  const char* requested = std::getenv("DFKV_RDMA_MR_BENCH");
  if (requested == nullptr || *requested == '\0')
    GTEST_SKIP() << "set DFKV_RDMA_MR_BENCH=<iterations> to run";
  const int iterations = std::max(1, std::atoi(requested));
  int devices = 0;
  ibv_device** list = ibv_get_device_list(&devices);
  if (list == nullptr || devices == 0) GTEST_SKIP() << "no RDMA device";
  const char* wanted = std::getenv("DFKV_RDMA_DEV");
  ibv_device* device = list[0];
  for (int i = 0; wanted && i < devices; ++i)
    if (std::strncmp(ibv_get_device_name(list[i]), wanted,
                     std::strlen(ibv_get_device_name(list[i]))) == 0)
      device = list[i];
  ibv_context* context = ibv_open_device(device);
  ASSERT_NE(context, nullptr);
  ibv_pd* pd = ibv_alloc_pd(context);
  ASSERT_NE(pd, nullptr);
  std::printf("device %s, %d iterations per row (microseconds per GET: "
              "register / deregister)\n",
              ibv_get_device_name(device), iterations);
  for (const size_t bytes : {size_t{64} << 10, size_t{1} << 20,
                             size_t{4} << 20, size_t{16} << 20,
                             size_t{64} << 20}) {
    for (const bool huge : {false, true}) {
      void* raw = nullptr;
      const size_t align = huge ? (size_t{2} << 20) : 4096;
      const size_t span = (bytes + align - 1) / align * align;
      ASSERT_EQ(::posix_memalign(&raw, align, span), 0);
      if (huge) ::madvise(raw, span, MADV_HUGEPAGE);
      std::memset(raw, 0x5a, span);  // resident, as an arena value is
      for (const bool local_write : {true, false}) {
        const int access =
            IBV_ACCESS_REMOTE_READ | (local_write ? IBV_ACCESS_LOCAL_WRITE : 0);
        double reg_us = 0, dereg_us = 0;
        for (int i = 0; i < iterations; ++i) {
          const auto t0 = std::chrono::steady_clock::now();
          ibv_mr* mr = ibv_reg_mr(pd, raw, bytes, access);
          const auto t1 = std::chrono::steady_clock::now();
          ASSERT_NE(mr, nullptr) << std::strerror(errno);
          ASSERT_EQ(ibv_dereg_mr(mr), 0);
          const auto t2 = std::chrono::steady_clock::now();
          reg_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
          dereg_us += std::chrono::duration<double, std::micro>(t2 - t1).count();
        }
        // The value's own transfer at 400 Gb/s, for scale.
        const double wire_us = static_cast<double>(bytes) * 8 / 400e3;
        std::printf("%6zu KiB %s %-22s reg %9.1f  dereg %9.1f  wire@400G %8.1f\n",
                    bytes >> 10, huge ? "2M-pages" : "4K-pages",
                    local_write ? "LOCAL_WRITE|REMOTE_READ" : "REMOTE_READ",
                    reg_us / iterations, dereg_us / iterations, wire_us);
      }
      std::free(raw);
    }
  }
  ibv_dealloc_pd(pd);
  ibv_close_device(context);
  ibv_free_device_list(list);
}

// RDS-01: a connection's end is signalled in both directions through the
// bootstrap socket kept as session channel (DFKV_RDMA_SESSION_FD, both
// sides). Without it, the first operation on a pooled connection the server
// had ended waited for RC retry exhaustion (measured 3.7 s here).
double TimedExistAfterReclaim(bool session, long* reaps) {
  ScopedEnv on("DFKV_RDMA_SESSION_FD", session ? "1" : "0");
  ScopedEnv idle_reclaim("DFKV_RDMA_IDLE_MS", "200");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv op_timeout("DFKV_RDMA_OP_TIMEOUT_MS", "60000");
  RdmaNode node(session ? "session-reclaim-on" : "session-reclaim-off");
  // One rail: on two, the operation after the reclaim may pick the other
  // rail, which has no pooled connection to reap, and the test measures
  // nothing (it then saw no reap and failed, though the op took 5 ms).
  RdmaTransport rt(kMaxMsg, SingleTestRail());
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  const std::string v(4096, 's');
  EXPECT_TRUE(c.Put("session", v.data(), v.size()));
  EXPECT_TRUE(c.Exist("session"));
  EXPECT_TRUE(ServerHasNoConnections(node)) << "the server did not reclaim";
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_TRUE(c.Exist("session"));
  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  *reaps = CounterVal(rt.MetricsText(),
                      "dfkv_rdma_client_session_closed_reaps_total");
  if (session)
    EXPECT_GE(CounterVal(node.rsrv->MetricsText(),
                         "dfkv_rdma_sessions_kept_total"), 1);
  return ms;
}

TEST(RdmaSession, AReclaimedPooledConnectionIsDroppedBeforeReuse) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  long reaps = -1;
  const double with_ms = TimedExistAfterReclaim(true, &reaps);
  EXPECT_GE(reaps, 1) << "the closed session channel was not seen";
  EXPECT_LT(with_ms, 500.0) << "the dead connection was used anyway";
  std::printf("session channel: first op after reclaim %.3f ms\n", with_ms);
}

// The keepalive drops a pooled connection whose session the server closed,
// instead of sending into it and waiting out RC retries.
TEST(RdmaSession, TheKeepaliveDropsAConnectionTheServerEnded) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv on("DFKV_RDMA_SESSION_FD", "1");
  ScopedEnv idle_reclaim("DFKV_RDMA_IDLE_MS", "200");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "600");
  RdmaNode node("session-keepalive");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  const std::string v(4096, 'k');
  ASSERT_TRUE(c.Put("keepalive", v.data(), v.size()));
  ASSERT_TRUE(ServerHasNoConnections(node)) << "the server did not reclaim";
  // No operation from here on: only the keepalive can find it.
  ASSERT_TRUE(
      WaitForMetric(rt, "dfkv_rdma_client_session_closed_reaps_total 1"))
      << "the keepalive did not drop the ended connection";
  EXPECT_EQ(CounterVal(rt.MetricsText(),
                       "dfkv_rdma_client_keepalive_failures_total"), 0);
}

// The server ends a connection as soon as its client drops it, instead of
// keeping the thread, the QP and its receive slots until the idle timeout.
TEST(RdmaSession, TheServerSeesTheClientDropTheConnection) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv on("DFKV_RDMA_SESSION_FD", "1");
  ScopedEnv idle_reclaim("DFKV_RDMA_IDLE_MS", "600000");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  RdmaNode node("session-client-drop");
  {
    RdmaTransport rt(kMaxMsg);
    KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
    const std::string v(4096, 'd');
    ASSERT_TRUE(c.Put("drop", v.data(), v.size()));
    ASSERT_GE(node.rsrv->ActiveConns(), 1u);
  }  // the client drops its pooled connections
  EXPECT_TRUE(ServerHasNoConnections(node, 2000))
      << "the server kept the connection after its client closed it";
  EXPECT_GE(CounterVal(node.rsrv->MetricsText(),
                       "dfkv_rdma_session_peer_closed_total"), 1);
  EXPECT_EQ(node.rsrv->CompletionErrors(), 0u);
}

// TRN-16: a transport holds each rail's device from its first connection
// until it is destroyed. Without that, a transport that never registered
// memory closed the device when its last connection went (idle reclaim,
// failures) and reopened it for the next: 12-13 ms to open an endpoint on a
// device not held, against 1.4-1.5 ms, and 3 ms to close against 1.5 ms.
TEST(RdmaTransportDevices, ADrainedPoolKeepsTheDevicesOpen) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  // The server, in this process, references the devices too: it ends its
  // side of a connection as soon as the client drops it (session channel),
  // so that the count below is the client's alone.
  ScopedEnv session("DFKV_RDMA_SESSION_FD", "1");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  RdmaNode node("device-held");
  const long before = rdma::RcEndpoint::DeviceReferences();
  {
    RdmaTransport rt(kMaxMsg);
    KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
    const std::string v(4096, 'd');
    ASSERT_TRUE(c.Put("held", v.data(), v.size()));
    ASSERT_TRUE(c.Exist("held"));
    ASSERT_GE(RdmaTransportTestPeer::DropAllPooled(&rt), 1u);
    ASSERT_TRUE(ServerHasNoConnections(node, 2000));
    // Every connection is gone; the devices they used are still held.
    const long held = rdma::RcEndpoint::DeviceReferences() - before;
    EXPECT_GE(held, 1) << "the drained pool released the devices";
    EXPECT_LE(held, RdmaTransportTestPeer::Rails(rt))
        << "more than one reference per rail";
    std::string out(v.size(), '\0');
    ASSERT_TRUE(c.Get("held", &out[0], out.size()));
    EXPECT_EQ(out, v);
    RdmaTransportTestPeer::DropAllPooled(&rt);
    ASSERT_TRUE(ServerHasNoConnections(node, 2000));
    const long again = rdma::RcEndpoint::DeviceReferences() - before;
    EXPECT_GE(again, held);
    EXPECT_LE(again, RdmaTransportTestPeer::Rails(rt))
        << "a reconnect took another reference";
  }
  EXPECT_EQ(rdma::RcEndpoint::DeviceReferences(), before)
      << "the destroyed transport kept a device";
}

// A device that is gone or was reset (an Open fails with ENODEV or EIO) is
// no longer held: once its endpoints are gone its context can close, and
// the next Open opens the device again instead of the dead context.
TEST(RdmaTransportDevices, ADeviceThatIsGoneIsNoLongerHeld) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv session("DFKV_RDMA_SESSION_FD", "1");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  RdmaNode node("device-gone");
  const long before = rdma::RcEndpoint::DeviceReferences();
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  const std::string v(4096, 'x');
  ASSERT_TRUE(c.Put("gone", v.data(), v.size()));
  ASSERT_TRUE(c.Exist("gone"));
  RdmaTransportTestPeer::DropAllPooled(&rt);
  ASSERT_TRUE(ServerHasNoConnections(node, 2000));
  ASSERT_GE(rdma::RcEndpoint::DeviceReferences() - before, 1);
  for (long rail = 0; rail < RdmaTransportTestPeer::Rails(rt); ++rail)
    RdmaTransportTestPeer::FailOpenOn(&rt, static_cast<size_t>(rail), ENODEV);
  EXPECT_FALSE(c.Exist("gone"));  // every rail's Open fails
  EXPECT_EQ(rdma::RcEndpoint::DeviceReferences(), before)
      << "a rail whose device is gone is still held";
}

// What the transport relies on: a device nothing references any more is
// closed, and the next Open opens it again (a fresh context).
TEST(RdmaVerbs, ADeviceNothingHoldsIsOpenedAgain) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  { rdma::RcEndpoint first;
    ASSERT_TRUE(first.Open(nullptr, rdma::kV2ControlCap, 1)); }
  // Run alone (as ctest does) nothing else in the process holds a device;
  // in a run of the whole binary an earlier test's object may.
  const bool unheld = rdma::RcEndpoint::DeviceReferences() == 0;
  const uint64_t opens = rdma::RcEndpoint::DeviceOpens();
  {
    rdma::RcEndpoint again;
    ASSERT_TRUE(again.Open(nullptr, rdma::kV2ControlCap, 1));
  }
  if (unheld) EXPECT_EQ(rdma::RcEndpoint::DeviceOpens(), opens + 1);
  // Held by a reference after its last endpoint closed: not reopened.
  auto holder = std::make_unique<rdma::RcEndpoint>();
  ASSERT_TRUE(holder->Open(nullptr, rdma::kV2ControlCap, 1));
  rdma::DeviceRef held = holder->ReferenceDevice();
  ASSERT_TRUE(held.held());
  holder.reset();
  const uint64_t held_opens = rdma::RcEndpoint::DeviceOpens();
  {
    rdma::RcEndpoint reused;
    ASSERT_TRUE(reused.Open(nullptr, rdma::kV2ControlCap, 1));
  }
  EXPECT_EQ(rdma::RcEndpoint::DeviceOpens(), held_opens);
}

// Fault-injection tests for guards and device disassociation only. No real
// device is reset or taken down.
namespace {
struct DeviceRebuildOn {
  explicit DeviceRebuildOn(int on = 1) { rdma::SetDeviceRebuildForTest(on); }
  ~DeviceRebuildOn() { rdma::SetDeviceRebuildForTest(-1); }
};
}  // namespace

// A failed verbs release is never an ignored success. Only a disassociated
// context lifts the fail-closed guard, and that exception is accounted for.
TEST(RdmaRelease, OrdinaryErrorsAbortButAProvenGoneDeviceIsCounted) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  DeviceRebuildOn on;
  rdma::RcEndpoint ep;
  ASSERT_TRUE(ep.Open(nullptr, 64 * 1024, 1));
  ibv_context* ctx = ep.context_for_test();
  const uint64_t prior = rdma::DeadReleaseFailures();

  // A still-confirming device makes EIO an ordinary error, so the guard
  // aborts rather than discarding the caller's MR.
  rdma::SetReleaseFaultsForTest({{}, []() { return 0; }, {}});
  ASSERT_EXIT((rdma::RequireVerbsReleaseForTest(ctx, EIO, "ordinary-bug")),
              ::testing::KilledBySignal(SIGABRT), "unsafe resource release");

  rdma::SetReleaseFaultsForTest({{}, []() { return ENODEV; }, {}});
  rdma::RequireVerbsReleaseForTest(ctx, ENODEV,
                                   "ibv_dereg_mr(window region)");
  EXPECT_EQ(rdma::DeadReleaseFailures(), prior + 1);
  rdma::SetReleaseFaultsForTest({});
}

// TRN-29 (docs/design/rdma-device-reset.md): a rail whose device was reset.
// No device is ever reset here: a seam reports DEVICE_FATAL, or fails Opens
// with ENODEV, as a reset would.
namespace {

long LabelledVal(const std::string& text, const std::string& line_prefix) {
  const auto pos = text.find("\n" + line_prefix);
  if (pos == std::string::npos) return -1;
  const auto value = text.find(' ', pos + 1 + line_prefix.size() - 1);
  try {
    return std::stol(text.substr(value + 1));
  } catch (...) {
    return -1;
  }
}

long Rebuilds(RdmaTransport& rt, const std::string& dev) {
  return LabelledVal(rt.MetricsText(),
                     "dfkv_rdma_client_device_rebuilds_total{dev=\"" + dev +
                         "\"} ");
}

long Dead(RdmaTransport& rt, const std::string& dev) {
  return LabelledVal(rt.MetricsText(),
                     "dfkv_rdma_client_device_dead{dev=\"" + dev + "\"} ");
}

template <class F>
bool Eventually(F done, int ms = 5000) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
  while (!done()) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return true;
}

// The background thread ticks every 20 ms (the port check's period).
struct FastTicks {
  ScopedEnv port_check{"DFKV_RDMA_PORT_CHECK_MS", "20"};
};

}  // namespace

// Switch off: a DEVICE_FATAL is counted and changes nothing else.
TEST(RdmaDeviceReset, WithTheSwitchOffAFatalEventIsOnlyCounted) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  DeviceRebuildOn off(0);
  FastTicks ticks;
  RdmaNode node("reset-off");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  std::vector<char> region(1 << 20);
  ASSERT_TRUE(rt.RegisterMemory(region.data(), region.size()));
  const std::string v(4096, 'o');
  ASSERT_TRUE(c.Put("off", v.data(), v.size()));
  const std::string dev = RdmaTransportTestPeer::Rail(rt, 0);
  const uint64_t generation = rdma::DeviceGeneration(dev);
  const uint64_t opens = rdma::RcEndpoint::DeviceOpens();
  rdma::InjectDeviceFatalForTest(dev);
  ASSERT_TRUE(Eventually([&] {
    return LabelledVal(rt.MetricsText(),
                       "dfkv_rdma_client_device_suspect_total{dev=\"" + dev +
                           "\",reason=\"fatal\"} ") == 1;
  }));
  EXPECT_EQ(rdma::DeviceGeneration(dev), generation);
  EXPECT_EQ(Dead(rt, dev), 0);
  std::string out(v.size(), '\0');
  EXPECT_TRUE(c.Get("off", &out[0], out.size()));
  EXPECT_EQ(out, v);
  EXPECT_EQ(rdma::RcEndpoint::DeviceOpens(), opens);
}

// Switch on: a DEVICE_FATAL retires the binding; the transport rebuilds the
// rail on a fresh context, every declared region registered on it, and the
// rail serves again.
TEST(RdmaDeviceReset, AFatalEventRebuildsTheRailOnAFreshContext) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  DeviceRebuildOn on;
  FastTicks ticks;
  RdmaNode node("reset-fatal");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  std::vector<char> region(2 << 20);
  ASSERT_TRUE(rt.RegisterMemory(region.data(), 1 << 20));
  ASSERT_TRUE(rt.RegisterMemory(region.data() + (1 << 20), 1 << 20));
  const std::string v(4096, 'f');
  ASSERT_TRUE(c.Put("fatal", v.data(), v.size()));
  const std::string dev = RdmaTransportTestPeer::Rail(rt, 0);
  const uint64_t generation = rdma::DeviceGeneration(dev);
  const uint64_t opens = rdma::RcEndpoint::DeviceOpens();
  const uint64_t regs = rdma::RcEndpoint::PoolMrRegistrations();
  rdma::InjectDeviceFatalForTest(dev);
  ASSERT_TRUE(Eventually([&] { return Rebuilds(rt, dev) == 1; }));
  EXPECT_EQ(rdma::DeviceGeneration(dev), generation + 1);
  EXPECT_GE(rdma::RcEndpoint::DeviceOpens(), opens + 1);
  // Both regions, once each, on the new context.
  EXPECT_EQ(rdma::RcEndpoint::PoolMrRegistrations(), regs + 2);
  EXPECT_EQ(Dead(rt, dev), 0);
  std::string out(v.size(), '\0');
  for (int i = 0; i < 8; ++i) {  // every rail, the rebuilt one included
    ASSERT_TRUE(c.Exist("fatal"));
    ASSERT_TRUE(c.Get("fatal", &out[0], out.size()));
    EXPECT_EQ(out, v);
  }
}

// Without a server holding the device too, the old context closes once the
// old anchor and connections are gone.
TEST(RdmaDeviceReset, TheOldContextClosesAfterTheRebuild) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  DeviceRebuildOn on;
  FastTicks ticks;
  const size_t retired = rdma::RetiredDeviceCount();
  {
    RdmaTransport rt(kMaxMsg);
    std::vector<char> region(1 << 20);
    ASSERT_TRUE(rt.RegisterMemory(region.data(), region.size()));
    const std::string dev = RdmaTransportTestPeer::Rail(rt, 0);
    rdma::InjectDeviceFatalForTest(dev);
    ASSERT_TRUE(Eventually([&] { return Rebuilds(rt, dev) == 1; }));
    EXPECT_EQ(rdma::RetiredDeviceCount(), retired)
        << "the retired context is still referenced";
  }
  EXPECT_EQ(rdma::RetiredDeviceCount(), retired);
}

// An Open that fails with ENODEV retires the binding the rail was built on;
// the rail is rebuilt and serves again.
TEST(RdmaDeviceReset, AnOpenFailingWithEnodevRebuildsTheRail) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  DeviceRebuildOn on;
  FastTicks ticks;
  RdmaNode node("reset-open");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  std::vector<char> region(1 << 20);
  ASSERT_TRUE(rt.RegisterMemory(region.data(), region.size()));
  const std::string v(4096, 'e');
  ASSERT_TRUE(c.Put("enodev", v.data(), v.size()));
  RdmaTransportTestPeer::DropAllPooled(&rt);
  const long rails = RdmaTransportTestPeer::Rails(rt);
  for (long rail = 0; rail < rails; ++rail)
    RdmaTransportTestPeer::FailOpenOn(&rt, static_cast<size_t>(rail), ENODEV);
  (void)c.Exist("enodev");
  for (long rail = 0; rail < rails; ++rail)
    RdmaTransportTestPeer::FailOpenOn(&rt, static_cast<size_t>(rail), 0);
  const std::string dev = RdmaTransportTestPeer::Rail(rt, 0);
  ASSERT_TRUE(Eventually([&] { return Rebuilds(rt, dev) >= 1; }));
  ASSERT_TRUE(Eventually([&] { return Dead(rt, dev) == 0; }));
  std::string out(v.size(), '\0');
  ASSERT_TRUE(Eventually([&] { return c.Get("enodev", &out[0], out.size()); }));
  EXPECT_EQ(out, v);
}

// While a rail cannot be rebuilt (its device does not open yet), memory can
// still be declared; the rebuild registers it with the rest.
TEST(RdmaDeviceReset, MemoryDeclaredWhileARailIsDeadIsRegisteredByItsRebuild) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  DeviceRebuildOn on;
  FastTicks ticks;
  RdmaTransport rt(kMaxMsg);
  std::vector<char> region(2 << 20);
  ASSERT_TRUE(rt.RegisterMemory(region.data(), 1 << 20));
  const std::string dev = RdmaTransportTestPeer::Rail(rt, 0);
  RdmaTransportTestPeer::FailRebuilds(&rt, true);
  rdma::InjectDeviceFatalForTest(dev);
  ASSERT_TRUE(Eventually([&] { return Dead(rt, dev) == 1; }));
  EXPECT_TRUE(rt.RegisterMemory(region.data() + (1 << 20), 1 << 20));
  const uint64_t regs = rdma::RcEndpoint::PoolMrRegistrations();
  RdmaTransportTestPeer::FailRebuilds(&rt, false);
  // The backoff after the failed attempts is at most a few seconds.
  ASSERT_TRUE(Eventually([&] { return Rebuilds(rt, dev) == 1; }, 20000));
  EXPECT_EQ(rdma::RcEndpoint::PoolMrRegistrations(), regs + 2)
      << "the rebuild did not register both regions";
}

// A rebuild attempt never waits for register_mu_: while a RegisterMemory
// holds it, the attempt is skipped and made on a later tick.
TEST(RdmaDeviceReset, AnAttemptWhileMemoryIsBeingDeclaredIsMadeLater) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  DeviceRebuildOn on;
  FastTicks ticks;
  RdmaTransport rt(kMaxMsg);
  std::vector<char> region(2 << 20);
  ASSERT_TRUE(rt.RegisterMemory(region.data(), 1 << 20));
  const std::string dev = RdmaTransportTestPeer::Rail(rt, 0);
  std::thread declaring;
  {
    HeldPoolRegistration held;
    declaring = std::thread(
        [&] { EXPECT_TRUE(rt.RegisterMemory(region.data(), 2 << 20)); });
    ASSERT_TRUE(held.AwaitEntry());
    rdma::InjectDeviceFatalForTest(dev);
    ASSERT_TRUE(Eventually([&] { return Dead(rt, dev) == 1; }));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(Rebuilds(rt, dev), 0) << "the rebuild waited for the lock";
    held.Release();
    declaring.join();
  }
  ASSERT_TRUE(Eventually([&] { return Rebuilds(rt, dev) == 1; }, 20000));
}

// Destroying the transport while a rebuild registers (held by a seam) or
// waits out its backoff returns: the background thread is joined before
// register_mu_ is taken, and a staging stops between regions.
TEST(RdmaDeviceReset, DestructionDuringARebuildReturns) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  DeviceRebuildOn on;
  FastTicks ticks;
  const uint64_t active = rdma::RcEndpoint::PoolMrActiveRegistrations();
  {  // in the backoff
    auto rt = std::make_unique<RdmaTransport>(kMaxMsg);
    std::vector<char> region(1 << 20);
    ASSERT_TRUE(rt->RegisterMemory(region.data(), region.size()));
    const std::string dev = RdmaTransportTestPeer::Rail(*rt, 0);
    RdmaTransportTestPeer::FailRebuilds(rt.get(), true);
    rdma::InjectDeviceFatalForTest(dev);
    ASSERT_TRUE(Eventually([&] { return Dead(*rt, dev) == 1; }));
    const auto t0 = std::chrono::steady_clock::now();
    rt.reset();
    EXPECT_LT(std::chrono::steady_clock::now() - t0, std::chrono::seconds(1));
  }
  {  // in an attempt, registering the first of two regions
    auto rt = std::make_unique<RdmaTransport>(kMaxMsg);
    std::vector<char> region(2 << 20);
    ASSERT_TRUE(rt->RegisterMemory(region.data(), 1 << 20));
    ASSERT_TRUE(rt->RegisterMemory(region.data() + (1 << 20), 1 << 20));
    const std::string dev = RdmaTransportTestPeer::Rail(*rt, 0);
    std::atomic<bool> destroyed{false};
    std::thread destroying;
    {
      HeldPoolRegistration held;
      rdma::InjectDeviceFatalForTest(dev);
      ASSERT_TRUE(held.AwaitEntry());  // the rebuild's first registration
      destroying = std::thread([&] {
        rt.reset();
        destroyed = true;
      });
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      EXPECT_FALSE(destroyed.load());
      held.Release();
      destroying.join();
    }
    EXPECT_TRUE(destroyed.load());
  }
  EXPECT_EQ(rdma::RcEndpoint::PoolMrActiveRegistrations(), active);
}

// The race the post-Open check closes: transport A sees the reset and
// retires the binding; transport B, whose background thread has not looked
// yet, opens a connection that lands on the new context. B drops it, takes
// the rail out of its selection, and rebuilds when its thread looks.
TEST(RdmaDeviceReset, AnOpenOnANewerContextThanTheRailIsDropped) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  DeviceRebuildOn on;
  FastTicks ticks;
  RdmaNode node("reset-race");
  RdmaTransport a(kMaxMsg), b(kMaxMsg);
  KVClient ca({{"n", node.addr}}, SelfHdr(), &a);
  KVClient cb({{"n", node.addr}}, SelfHdr(), &b);
  std::vector<char> region_a(1 << 20), region_b(1 << 20);
  ASSERT_TRUE(a.RegisterMemory(region_a.data(), region_a.size()));
  ASSERT_TRUE(b.RegisterMemory(region_b.data(), region_b.size()));
  const std::string v(4096, 'r');
  ASSERT_TRUE(ca.Put("race", v.data(), v.size()));
  ASSERT_TRUE(cb.Exist("race"));
  const std::string dev = RdmaTransportTestPeer::Rail(b, 0);
  RdmaTransportTestPeer::PauseDeviceChecks(&b, true);
  RdmaTransportTestPeer::DropAllPooled(&b);
  rdma::InjectDeviceFatalForTest(dev);
  ASSERT_TRUE(Eventually([&] { return Rebuilds(a, dev) == 1; }));
  // B's rail 0 still believes the old context: every Open lands on the new.
  const bool served = cb.Exist("race");
  EXPECT_EQ(Dead(b, dev), 1) << "B kept a rail with no registration";
  if (RdmaTransportTestPeer::Rails(b) > 1)
    EXPECT_TRUE(served) << "the operation did not go to another rail";
  EXPECT_EQ(Rebuilds(b, dev), 0);
  RdmaTransportTestPeer::PauseDeviceChecks(&b, false);
  ASSERT_TRUE(Eventually([&] { return Rebuilds(b, dev) == 1; }));
  std::string out(v.size(), '\0');
  ASSERT_TRUE(Eventually([&] { return cb.Get("race", &out[0], out.size()); }));
  EXPECT_EQ(out, v);
}

// The abort on a failed release is skipped only on proof that the context
// is disassociated. Death tests run the statement in a new process.

TEST(RdmaDeviceReset, AFailedReleaseOnAProvenDeadDeviceIsCounted) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  DeviceRebuildOn on;
  struct Case {
    int release, confirm;
    bool fatal, listed;
  };
  for (const Case c : {Case{EIO, 0, true, true}, Case{EIO, ENODEV, false, true},
                       Case{ENODEV, ENODEV, false, true},
                       Case{EIO, EIO, false, false}}) {
    SCOPED_TRACE("release " + std::to_string(c.release) + " confirm " +
                 std::to_string(c.confirm));
    const uint64_t dropped = rdma::DeadReleaseFailures();
    {
      rdma::RcEndpoint ep;
      ASSERT_TRUE(ep.Open(nullptr, rdma::kV2ControlCap, 1));
      if (c.fatal) {
        rdma::InjectDeviceFatalForTest("");
        (void)rdma::PollDeviceEvents();
      }
      rdma::ReleaseFaultsForTest faults;
      faults.release = [c](const char* operation) {
        return std::strcmp(operation, "ibv_destroy_qp") == 0 ? c.release : 0;
      };
      faults.confirm = [c] { return c.confirm; };
      faults.listed = [c] { return c.listed; };
      rdma::SetReleaseFaultsForTest(faults);
    }  // closes: the destroy "fails"
    rdma::SetReleaseFaultsForTest({});
    EXPECT_EQ(rdma::DeadReleaseFailures(), dropped + 1);
    // A context marked fatal is not reused: retire it.
    rdma::RetireDevice("", rdma::DeviceGeneration(""));
  }
}

TEST(RdmaDeviceResetDeathTest, AFailedReleaseWithoutProofStillAborts) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  struct Case {
    const char* what;
    int on, release, confirm;
    bool listed;
  };
  for (const Case c :
       {Case{"a live device answering EIO once", 1, EIO, 0, true},
        Case{"a confirmed EIO on a device still listed", 1, EIO, EIO, true},
        Case{"EBUSY", 1, EBUSY, ENODEV, false},
        Case{"the switch off", 0, EIO, ENODEV, false}}) {
    SCOPED_TRACE(c.what);
    EXPECT_DEATH(
        {
          rdma::SetDeviceRebuildForTest(c.on);
          rdma::RcEndpoint ep;
          if (!ep.Open(nullptr, rdma::kV2ControlCap, 1)) std::exit(3);
          rdma::ReleaseFaultsForTest faults;
          faults.release = [c](const char* operation) {
            return std::strcmp(operation, "ibv_destroy_qp") == 0 ? c.release
                                                                 : 0;
          };
          faults.confirm = [c] { return c.confirm; };
          faults.listed = [c] { return c.listed; };
          rdma::SetReleaseFaultsForTest(faults);
        },
        "unsafe resource release");
  }
}

// Mixed versions, server side: a client that does not ask (every release
// before this one, here the test peer that closes its bootstrap socket at
// once) keeps its connection on a server with session channels on.
TEST(RdmaSession, AClientThatDoesNotAskIsServedAsBefore) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv on("DFKV_RDMA_SESSION_FD", "1");
  RdmaNode node("session-old-client");
  const BlockKey key{90501, 1};
  std::string ignored;
  ASSERT_EQ(node.srv->ProcessRequestForKey(
                static_cast<uint8_t>(WireOp::kCache), key, 0, 0, "old", 3,
                &ignored), Status::kOk);
  PinnedPullPeer peer;
  ASSERT_TRUE(peer.Open(node));  // closes its end of the bootstrap socket
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  EXPECT_TRUE(PinnedExists(peer, key)) << "the closed bootstrap ended it";
  const std::string metrics = node.rsrv->MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_sessions_kept_total"), 0);
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_session_peer_closed_total"), 0);
}

// Mixed versions, client side: a server without session channels (off, or
// a release that predates them, which does not advertise the capability)
// is asked for none and served as before.
TEST(RdmaSession, AServerWithoutSessionsIsUsedAsBefore) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  std::unique_ptr<RdmaNode> node;
  {
    ScopedEnv off("DFKV_RDMA_SESSION_FD", "0");
    node = std::make_unique<RdmaNode>("session-old-server");
  }
  ScopedEnv on("DFKV_RDMA_SESSION_FD", "1");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node->addr}}, SelfHdr(), &rt);
  const std::string v(4096, 'o');
  ASSERT_TRUE(c.Put("old-server", v.data(), v.size()));
  std::string out(v.size(), '\0');
  ASSERT_TRUE(c.Get("old-server", &out[0], out.size()));
  EXPECT_EQ(out, v);
  EXPECT_EQ(CounterVal(node->rsrv->MetricsText(),
                       "dfkv_rdma_sessions_kept_total"), 0);
  EXPECT_EQ(CounterVal(rt.MetricsText(),
                       "dfkv_rdma_client_session_closed_reaps_total"), 0);
}

// Opt-in measurement, both settings: DFKV_RDMA_STALE_BENCH=<rounds>.
TEST(RdmaSession, FirstOperationAfterReclaimBench) {
  const char* requested = std::getenv("DFKV_RDMA_STALE_BENCH");
  if (requested == nullptr || *requested == '\0')
    GTEST_SKIP() << "set DFKV_RDMA_STALE_BENCH=<rounds> to run";
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const int rounds = std::max(1, std::atoi(requested));
  for (int round = 0; round < rounds; ++round) {
    long reaps = 0;
    const double off_ms = TimedExistAfterReclaim(false, &reaps);
    const double on_ms = TimedExistAfterReclaim(true, &reaps);
    std::printf("row round %d: first op after reclaim, session channel off "
                "%.3f ms, on %.3f ms\n", round, off_ms, on_ms);
  }
}
}  // namespace

// A scatter-gather GET reads several destination segments per READ (up to
// the device's max_sge_rd) instead of one READ per segment. 61 segments of
// unequal, odd sizes, with empty ones in the middle and at the end, some
// outside the registered pool, and a value that ends inside the last
// non-empty segment: every byte lands where it belongs, with
// ceil(non-empty segments / ReadSge) READs, not one per segment. Two keys
// in one call keep their own segment and MR state across READ rounds.
TEST(RdmaSgRead, ManySegmentsPerReadLandExactly) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const char* devs = std::getenv("DFKV_RDMA_DEV");
  const std::string first =
      devs ? std::string(devs).substr(0, std::string(devs).find(',')) : "";
  size_t read_sge = 1;
  {
    rdma::RcEndpoint probe;
    ASSERT_TRUE(probe.Open(first.empty() ? nullptr : first.c_str(), 4096, 1));
    read_sge = probe.ReadSge();
  }
  RdmaNode node("sg-read-many", 1u << 20);
  RdmaTransport rt(1u << 20, first);
  constexpr size_t kSegs = 61;
  constexpr size_t kKeys = 2;
  std::vector<size_t> sizes;
  size_t total = 0, non_empty = 0;
  for (size_t s = 0; s < kSegs; ++s) {
    // Empty segments in the middle and the last one empty too.
    const bool empty = s == 10 || s == 11 || s == 40 || s == kSegs - 1;
    sizes.push_back(empty ? 0 : 97 + 31 * s);
    total += sizes.back();
    non_empty += empty ? 0 : 1;
  }
  const size_t value_len = total - 50;  // ends inside the last non-empty one
  std::vector<std::string> values(kKeys, std::string(value_len, '\0'));
  std::vector<BlockKey> keys;
  for (size_t k = 0; k < kKeys; ++k) {
    for (size_t i = 0; i < value_len; ++i)
      values[k][i] = static_cast<char>((i * 131 + 7 + 59 * k) & 0xff);
    keys.push_back(ToBlockKey(SelfHdr(), "sg-many-" + std::to_string(k)));
    ASSERT_EQ(rt.Cache(node.addr, keys[k], values[k].data(), value_len),
              Status::kOk);
  }
  // Segments spread over a registered region and a loose one, each followed
  // by a guard byte that must stay untouched.
  std::vector<char> pooled(kKeys * (total + kSegs), 'G');
  std::vector<char> loose(kKeys * (total + kSegs), 'G');
  ASSERT_TRUE(rt.RegisterMemory(pooled.data(), pooled.size()));
  std::vector<RangeDstMulti> dsts(kKeys);
  size_t pooled_at = 0, loose_at = 0;
  for (size_t k = 0; k < kKeys; ++k) {
    for (size_t s = 0; s < kSegs; ++s) {
      char* at =
          s % 3 == 2 ? loose.data() + loose_at : pooled.data() + pooled_at;
      (s % 3 == 2 ? loose_at : pooled_at) += sizes[s] + 1;
      dsts[k].payloads.emplace_back(at, sizes[s]);
    }
  }
  const auto reads = [&] {
    return CounterVal(rt.MetricsText(), "dfkv_rdma_client_pull_read_wrs_total");
  };
  const long before = reads();
  std::vector<size_t> lens;
  const auto statuses = rt.RangeIntoMulti(node.addr, keys, dsts, &lens);
  const long posted = reads() - before;
  for (size_t k = 0; k < kKeys; ++k) {
    SCOPED_TRACE(k);
    ASSERT_EQ(statuses[k], Status::kOk);
    EXPECT_EQ(lens[k], value_len);
    size_t offset = 0;
    for (size_t s = 0; s < kSegs; ++s) {
      const char* at = static_cast<const char*>(dsts[k].payloads[s].first);
      const size_t want =
          std::min(sizes[s], value_len - std::min(value_len, offset));
      ASSERT_EQ(std::memcmp(at, values[k].data() + offset, want), 0)
          << "segment " << s;
      EXPECT_EQ(at[sizes[s]], 'G') << "guard after segment " << s;
      offset += sizes[s];
    }
  }
  if (read_sge == 1) GTEST_SKIP() << "the device reads into one segment per READ";
  const long per_key = static_cast<long>((non_empty + read_sge - 1) / read_sge);
  EXPECT_EQ(posted, static_cast<long>(kKeys) * per_key)
      << "READs for " << kKeys << " keys of " << non_empty
      << " non-empty segments with ReadSge " << read_sge;
}

// RDS-13: a connection's rkey covers its own receive slots only. With one
// MR over the whole chunk, a connection could write into another
// connection's slots with its own rkey (reproduced: the WRITE completed and
// the bytes landed), corrupting that connection's PUT in flight.
TEST(RdmaLoopback, AConnectionCannotWriteAnotherConnectionsSlots) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "0");
  RdmaNode node("rkey-isolation");
  const BlockKey key{90601, 1};
  std::string ignored;
  ASSERT_EQ(node.srv->ProcessRequestForKey(
                static_cast<uint8_t>(WireOp::kCache), key, 0, 0, "v", 1,
                &ignored), Status::kOk);
  PinnedPullPeer a, b;
  ASSERT_TRUE(a.Open(node));
  ASSERT_TRUE(b.Open(node));
  ASSERT_NE(a.resident.base_addr, b.resident.base_addr);
  EXPECT_NE(a.resident.rkey, b.resident.rkey);
  const uint64_t target = b.resident.base_addr + rdma::kV2DataOffset;
  const std::string before(reinterpret_cast<const char*>(target), 64);
  std::string marker(64, 'X');
  ibv_mr* mr = a.ep.RegisterTransient(
      marker.data(), marker.size(),
      rdma::RcEndpoint::TransientAccess::kLocalWrite);
  ASSERT_NE(mr, nullptr);
  ASSERT_TRUE(a.ep.PostWrite(0, marker.data(), marker.size(), mr, target,
                             a.resident.rkey));
  ibv_wc wc{};
  ASSERT_EQ(a.ep.WaitComp(&wc, 1, 5000), 1);
  a.ep.ReleaseTransient(mr);
  EXPECT_NE(wc.status, IBV_WC_SUCCESS)
      << "a WRITE into another connection's slot was accepted";
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(target), 64), before);
  EXPECT_TRUE(PinnedExists(b, key)) << "b's connection was affected";
  // A WRITE into its own slots still works.
  PinnedPullPeer c;
  ASSERT_TRUE(c.Open(node));
  std::string mine(64, 'c');
  ibv_mr* own_mr = c.ep.RegisterTransient(
      mine.data(), mine.size(),
      rdma::RcEndpoint::TransientAccess::kLocalWrite);
  ASSERT_NE(own_mr, nullptr);
  ASSERT_TRUE(c.ep.PostWrite(0, mine.data(), mine.size(), own_mr,
                             c.resident.base_addr + rdma::kV2DataOffset,
                             c.resident.rkey));
  ASSERT_EQ(c.ep.WaitComp(&wc, 1, 5000), 1);
  c.ep.ReleaseTransient(own_mr);
  EXPECT_EQ(wc.status, IBV_WC_SUCCESS);
  EXPECT_EQ(std::memcmp(reinterpret_cast<const char*>(
                            c.resident.base_addr + rdma::kV2DataOffset),
                        mine.data(), mine.size()), 0);
}

// RDS-13: receive slots leased to a new connection hold none of the bytes
// a previous connection left in them.
TEST(RdmaLoopback, ANewConnectionsSlotsAreZeroed) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "0");
  ScopedEnv idle("DFKV_RDMA_IDLE_MS", "200");
  RdmaNode node("slot-zeroing");
  uint64_t first_base = 0, first_size = 0;
  {
    PinnedPullPeer first;
    ASSERT_TRUE(first.Open(node));
    first_base = first.resident.base_addr;
    first_size = first.resident.slot_size;
    std::string marker(4096, 'R');
    ibv_mr* mr = first.ep.RegisterTransient(
        marker.data(), marker.size(),
        rdma::RcEndpoint::TransientAccess::kLocalWrite);
    ASSERT_NE(mr, nullptr);
    ASSERT_TRUE(first.ep.PostWrite(0, marker.data(), marker.size(), mr,
                                   first_base + rdma::kV2DataOffset,
                                   first.resident.rkey));
    ibv_wc wc{};
    ASSERT_EQ(first.ep.WaitComp(&wc, 1, 5000), 1);
    ASSERT_EQ(wc.status, IBV_WC_SUCCESS);
    first.ep.ReleaseTransient(mr);
    ASSERT_TRUE(ServerHasNoConnections(node)) << "not reclaimed";
  }
  PinnedPullPeer second;
  ASSERT_TRUE(second.Open(node));
  ASSERT_EQ(second.resident.base_addr, first_base)
      << "the new connection did not get the same slots";
  const char* slots = reinterpret_cast<const char*>(second.resident.base_addr);
  EXPECT_EQ(std::count(slots, slots + first_size, '\0'),
            static_cast<std::ptrdiff_t>(first_size))
      << "the previous connection's bytes are still in the slots";
}

// RDS-06: READ depth. A QP takes the initiator and responder depths it is
// given, clamped to what the device supports, and with both at 16 a window
// of READs runs in flight together. Defaults stay 1.
TEST(RdmaReadDepth, AQpTakesItsReadDepthsClampedToTheDevice) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  for (const bool deep : {false, true}) {
    rdma::RcEndpoint a, b;
    a.set_remote_access(true);
    b.set_remote_access(true);
    if (deep) {
      a.set_read_depth(255);  // more than any device: clamped
      b.set_responder_read_depth(rdma::kV2ReadDepth);
    }
    ASSERT_TRUE(a.Open(nullptr, rdma::kV2ControlCap, 16));
    ASSERT_TRUE(b.Open(nullptr, rdma::kV2ControlCap, 16));
    ASSERT_TRUE(a.Connect(b.Local()));
    ASSERT_TRUE(b.Connect(a.Local()));
    int a_init = 0, a_resp = 0, b_init = 0, b_resp = 0;
    ASSERT_TRUE(a.QueryReadDepths(&a_init, &a_resp));
    ASSERT_TRUE(b.QueryReadDepths(&b_init, &b_resp));
    EXPECT_EQ(b_init, 1);
    EXPECT_EQ(a_resp, 1);
    if (!deep) {
      EXPECT_EQ(a_init, 1);
      EXPECT_EQ(b_resp, 1);
      continue;
    }
    EXPECT_GE(a_init, 1);
    EXPECT_LE(a_init, 255);
    EXPECT_GE(b_resp, std::min(a_init, static_cast<int>(rdma::kV2ReadDepth)))
        << "the responder must accept what the initiator sends";
    // Sixteen READs posted together all complete.
    constexpr size_t kBytes = 65536;
    std::vector<char> src(16 * kBytes), dst(16 * kBytes);
    for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<char>(i * 7);
    ibv_mr* remote = b.RegisterLeaseReadRegion(src.data(), src.size());
    ibv_mr* local = a.RegisterTransient(
        dst.data(), dst.size(),
        rdma::RcEndpoint::TransientAccess::kLocalWrite);
    ASSERT_NE(remote, nullptr);
    ASSERT_NE(local, nullptr);
    for (size_t i = 0; i < 16; ++i)
      ASSERT_TRUE(a.PostRead(i, dst.data() + i * kBytes, kBytes, local,
                             reinterpret_cast<uint64_t>(src.data() + i * kBytes),
                             remote->rkey));
    std::vector<ibv_wc> wcs(16);
    for (int got = 0; got < 16;) {
      const int g = a.WaitComp(wcs.data(), 16, 5000);
      ASSERT_GT(g, 0);
      for (int k = 0; k < g; ++k) ASSERT_EQ(wcs[k].status, IBV_WC_SUCCESS);
      got += g;
    }
    EXPECT_EQ(dst, src);
    a.ReleaseTransient(local);
    b.ReleaseLeaseReadRegion(remote);
  }
}

namespace {
// GETs of four values of 200 KiB (under the tests' block bound) through the
// real client and server.
void GetFourValues(RdmaNode& node, RdmaTransport& rt, const std::string& tag) {
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  std::vector<std::string> values;
  for (int i = 0; i < 4; ++i) {
    values.push_back(PatternValue(200 << 10, static_cast<size_t>(i + 11)));
    ASSERT_TRUE(c.Put(tag + std::to_string(i), values.back().data(),
                      values.back().size()));
  }
  for (int i = 0; i < 4; ++i) {
    std::string out(values[i].size(), '\0');
    ASSERT_TRUE(c.Get(tag + std::to_string(i), &out[0], out.size())) << i;
    EXPECT_EQ(out, values[i]) << i;
  }
}
}  // namespace

// The client pipelines its READs only with DFKV_RDMA_PIPELINED_READS on and
// a server that advertises its READ depth (this one does, always).
TEST(RdmaReadDepth, PipelinedReadsOnlyWhenAskedAndAdvertised) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("read-depth");
  {
    RdmaTransport rt(kMaxMsg);  // switch off: as before
    GetFourValues(node, rt, "rd-off-");
    EXPECT_EQ(CounterVal(rt.MetricsText(),
                         "dfkv_rdma_client_pipelined_read_conns_total"), 0);
  }
  {
    ScopedEnv on("DFKV_RDMA_PIPELINED_READS", "1");
    RdmaTransport rt(kMaxMsg);
    GetFourValues(node, rt, "rd-on-");
    EXPECT_GE(CounterVal(rt.MetricsText(),
                         "dfkv_rdma_client_pipelined_read_conns_total"), 1);
  }
  EXPECT_EQ(node.rsrv->CompletionErrors(), 0u);
}

// RDS-06: pipelined READs are offered (probe bit 6) only if every rail's
// device accepts kV2ReadDepth READs in flight per QP; one weaker rail is
// enough to withhold it, since the probe is not per rail.
TEST(RdmaReadDepth, OfferedOnlyIfEveryRailAcceptsTheDepth) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ConfigureTestRecvSegment();
  for (const int device_depth : {0, rdma::kV2ReadDepth - 1}) {
    RdmaServer server(
        [](uint8_t, const BlockKey&, uint64_t, uint64_t, const char*,
           uint64_t, std::string*, size_t*) { return Status::kInvalid; },
        kMaxMsg);
    RdmaServerTestPeer::SetResponderReadDepth(&server, device_depth);
    ASSERT_EQ(server.Start(0), Status::kOk);
    const int fd =
        net::Dial("127.0.0.1:" + std::to_string(server.port()), 2000, 2000);
    ASSERT_GE(fd, 0);
    char frame[rdma::kDevNameBytes], reply[rdma::kV2ProbeReplyBytes];
    rdma::EncodeDevFrame(rdma::kV2ProbeDevice, kMaxMsg, frame,
                         rdma::kDevProtoV2);
    ASSERT_TRUE(net::WriteAll(fd, frame, sizeof(frame)));
    ASSERT_TRUE(net::ReadAll(fd, reply, sizeof(reply)));
    ::close(fd);
    // 0: the devices' own limit (16 here, 128 on rxe); 15: one rail short.
    EXPECT_EQ(rdma::V2ProbeSupportsReadDepth(reply), device_depth == 0)
        << "device depth " << device_depth;
    server.Stop();
  }
  }
namespace {
// RDS-02: the server's port watcher, driven through its seam (no real port
// is touched). Rails by name; the node serves every configured rail.
struct ServerPorts {
  std::vector<std::string> rails;
  std::unique_ptr<std::atomic<int>[]> state;
  explicit ServerPorts(const RdmaNode& node)
      : rails(node.rsrv->DeviceNames()),
        state(std::make_unique<std::atomic<int>[]>(rails.size())) {
    for (size_t i = 0; i < rails.size(); ++i) state[i] = 1;
  }
};
bool WaitFor(const std::function<bool()>& done, int timeout_ms = 3000) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);
  while (!done()) {
    if (std::chrono::steady_clock::now() > deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return true;
}
// Clears the port seam before the state it reads goes away.
// Stops the server first: the watcher calls a copy of the seam outside the
// lock, so a poll in flight could still read the test's locals after the
// reset. Stop() joins the watcher; the node's own Stop() later is a no-op.
struct ResetPortSeam {
  RdmaServer* server;
  ~ResetPortSeam() {
    server->Stop();
    RdmaServerTestPeer::SetPortState(server, nullptr);
  }
};
std::string RailPortLine(const std::string& rail) {
  return "dfkv_rdma_rail_port_active{dev=\"" + rail + "\"} ";
}
bool RailPortIs(const RdmaNode& node, const std::string& rail, int value) {
  return node.rsrv->MetricsText().find(RailPortLine(rail) +
                                       std::to_string(value) + "\n") !=
         std::string::npos;
}
}  // namespace

// A client on a rail whose server port is not ACTIVE is refused at the
// bootstrap: it fails at once instead of after RC retries on a QP that
// cannot carry it, and is served again once the port is back.
TEST(RdmaServerPorts, ARailWhosePortIsDownIsRefusedUntilItIsBack) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv check("DFKV_RDMA_PORT_CHECK_MS", "20");
  RdmaNode node("port-down-refused");
  ServerPorts ports(node);
  const std::string rail = ports.rails.back();
  ports.state[ports.rails.size() - 1] = 0;
  RdmaServerTestPeer::SetPortState(
      node.rsrv.get(), [&ports](size_t i) { return ports.state[i].load(); });
  ResetPortSeam reset_seam{node.rsrv.get()};
  ASSERT_TRUE(WaitFor([&] { return RailPortIs(node, rail, 0); }));
  ScopedEnv no_client_watch("DFKV_RDMA_PORT_CHECK_MS", "0");
  {
    RdmaTransport rt(kMaxMsg, rail);
    KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
    const std::string v(4096, 'd');
    const auto t0 = std::chrono::steady_clock::now();
    EXPECT_FALSE(c.Put("port-down", v.data(), v.size()));
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    EXPECT_LT(ms, 1000.0) << "refused at the bootstrap, not after RC retries";
    EXPECT_GE(CounterVal(node.rsrv->MetricsText(),
                         "dfkv_rdma_inactive_rail_refusals_total"), 1);
    EXPECT_GE(CounterVal(rt.MetricsText(), "dfkv_rdma_client_peer_busy_total"),
              1);
  }
  ports.state[ports.rails.size() - 1] = 1;
  ASSERT_TRUE(WaitFor([&] { return RailPortIs(node, rail, 1); }));
  RdmaTransport rt(kMaxMsg, rail);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  const std::string v(4096, 'u');
  EXPECT_TRUE(c.Put("port-back", v.data(), v.size()));
}

// A rail whose port stops being ACTIVE has its connections ended at once,
// and only its own: another rail's connections go on.
TEST(RdmaServerPorts, ARailGoingDownEndsOnlyItsConnections) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const char* devs = std::getenv("DFKV_RDMA_DEV");
  if (!devs || !std::strchr(devs, ','))
    GTEST_SKIP() << "needs DFKV_RDMA_DEV with two rails";
  ScopedEnv check("DFKV_RDMA_PORT_CHECK_MS", "20");
  RdmaNode node("port-down-ends");
  ServerPorts ports(node);
  ASSERT_GE(ports.rails.size(), 2u);
  RdmaServerTestPeer::SetPortState(
      node.rsrv.get(), [&ports](size_t i) { return ports.state[i].load(); });
  ResetPortSeam reset_seam{node.rsrv.get()};
  ScopedEnv no_client_watch("DFKV_RDMA_PORT_CHECK_MS", "0");
  RdmaTransport first(kMaxMsg, ports.rails[0]), second(kMaxMsg, ports.rails[1]);
  KVClient c0({{"n", node.addr}}, SelfHdr(), &first);
  KVClient c1({{"n", node.addr}}, SelfHdr(), &second);
  const std::string v(4096, 'e');
  ASSERT_TRUE(c0.Put("ends-0", v.data(), v.size()));
  ASSERT_TRUE(c1.Put("ends-1", v.data(), v.size()));
  ASSERT_GE(RdmaServerTestPeer::LiveConnectionsOnRail(*node.rsrv, 0), 1u);
  ASSERT_GE(RdmaServerTestPeer::LiveConnectionsOnRail(*node.rsrv, 1), 1u);
  const size_t on_first =
      RdmaServerTestPeer::LiveConnectionsOnRail(*node.rsrv, 0);
  ports.state[1] = 0;
  EXPECT_TRUE(WaitFor([&] {
    return RdmaServerTestPeer::LiveConnectionsOnRail(*node.rsrv, 1) == 0;
  })) << "the down rail's connections were not ended";
  EXPECT_EQ(RdmaServerTestPeer::LiveConnectionsOnRail(*node.rsrv, 0),
            on_first) << "another rail's connection was ended";
  EXPECT_GE(CounterVal(node.rsrv->MetricsText(),
                       "dfkv_rdma_rail_down_connections_ended_total"), 1);
  std::string out(v.size(), '\0');
  EXPECT_TRUE(c0.Get("ends-0", &out[0], out.size()));
}

// A port that flaps reaches the listener (which asks the MDS for an early
// heartbeat) at most once a second per rail; the first look at startup is
// not reported.
TEST(RdmaServerPorts, AFlappingPortNoticesTheListenerAtMostOnceASecond) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv check("DFKV_RDMA_PORT_CHECK_MS", "10");
  RdmaNode node("port-flap");
  ServerPorts ports(node);
  std::mutex mu;
  std::map<std::string, int> notices;
  node.rsrv->set_rail_state_listener(
      [&](const std::string& dev, bool) {
        std::lock_guard<std::mutex> lock(mu);
        ++notices[dev];
      });
  std::atomic<uint64_t> calls{0};
  RdmaServerTestPeer::SetPortState(
      node.rsrv.get(), [&ports, &calls](size_t i) {
        if (i + 1 != ports.rails.size()) return 1;
        return static_cast<int>(calls.fetch_add(1) % 2);  // flaps every poll
      });
  ResetPortSeam reset_seam{node.rsrv.get()};
  std::this_thread::sleep_for(std::chrono::milliseconds(2500));
  node.rsrv->set_rail_state_listener(nullptr);
  const long changes = CounterVal(node.rsrv->MetricsText(),
                                  "dfkv_rdma_rail_port_changes_total");
  std::lock_guard<std::mutex> lock(mu);
  EXPECT_GE(changes, 20) << "the port did not flap as set";
  EXPECT_GE(notices[ports.rails.back()], 1);
  EXPECT_LE(notices[ports.rails.back()], 4) << changes << " changes";
  for (size_t i = 0; i + 1 < ports.rails.size(); ++i)
    EXPECT_EQ(notices[ports.rails[i]], 0) << ports.rails[i];
}

// A client that names no device gets the first rail not known to be down.
TEST(RdmaServerPorts, NoDeviceNamedGetsTheFirstActiveRail) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const char* devs = std::getenv("DFKV_RDMA_DEV");
  if (!devs || !std::strchr(devs, ','))
    GTEST_SKIP() << "needs DFKV_RDMA_DEV with two rails";
  ScopedEnv check("DFKV_RDMA_PORT_CHECK_MS", "20");
  RdmaNode node("port-default");
  ServerPorts ports(node);
  RdmaServerTestPeer::SetPortState(
      node.rsrv.get(), [&ports](size_t i) { return ports.state[i].load(); });
  ResetPortSeam reset_seam{node.rsrv.get()};
  EXPECT_EQ(RdmaServerTestPeer::DefaultRail(*node.rsrv), 0u);
  ports.state[0] = 0;
  ASSERT_TRUE(WaitFor([&] { return RailPortIs(node, ports.rails[0], 0); }));
  EXPECT_EQ(RdmaServerTestPeer::DefaultRail(*node.rsrv), 1u);
}

// RDS-02 review: a connection that passed its rail check just before the
// port went down registers after the watcher woke the rail's connections:
// it must end itself rather than live on, unnoticed, on a dead rail.
TEST(RdmaServerPorts, AConnectionRegisteringAsItsRailGoesDownEndsItself) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv check("DFKV_RDMA_PORT_CHECK_MS", "20");
  RdmaNode node("port-down-registering");
  ServerPorts ports(node);
  const size_t rail = ports.rails.size() - 1;
  RdmaServerTestPeer::SetPortState(
      node.rsrv.get(), [&ports](size_t i) { return ports.state[i].load(); });
  ResetPortSeam reset_seam{node.rsrv.get()};
  std::atomic<bool> armed{true};
  RdmaServerTestPeer::SetBeforeLiveRegistration(node.rsrv.get(), [&] {
    if (!armed.exchange(false)) return;
    ports.state[rail] = 0;  // after the rail check, before registration
    WaitFor([&] { return RailPortIs(node, ports.rails[rail], 0); });
    // Two more polls: the watcher has confirmed it and woken the rail's
    // connections, of which this one is not yet.
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
  });
  struct ClearHook {
    RdmaServer* server;
    ~ClearHook() { RdmaServerTestPeer::SetBeforeLiveRegistration(server, {}); }
  } clear_hook{node.rsrv.get()};
  ScopedEnv no_client_watch("DFKV_RDMA_PORT_CHECK_MS", "0");
  RdmaTransport rt(kMaxMsg, ports.rails[rail]);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  const std::string v(4096, 'r');
  (void)c.Put("registering", v.data(), v.size());  // may fail: rail is down
  EXPECT_TRUE(WaitFor([&] {
    return RdmaServerTestPeer::LiveConnectionsOnRail(*node.rsrv, rail) == 0;
  })) << "the connection that registered after the wake lives on";
  EXPECT_GE(CounterVal(node.rsrv->MetricsText(),
                       "dfkv_rdma_rail_down_connections_ended_total"), 1);
}

// RDS-02 review: a change suppressed by the once-a-second limit is sent once
// the second is over, if it still holds: a port down then up again within
// the second ends with "up" notified.
TEST(RdmaServerPorts, TheLastStateIsNotifiedAfterTheRateLimit) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv check("DFKV_RDMA_PORT_CHECK_MS", "20");
  RdmaNode node("port-trailing-notice");
  ServerPorts ports(node);
  const size_t rail = ports.rails.size() - 1;
  std::mutex mu;
  std::vector<bool> heard;
  node.rsrv->set_rail_state_listener([&](const std::string& dev, bool up) {
    if (dev != ports.rails[rail]) return;
    std::lock_guard<std::mutex> lock(mu);
    heard.push_back(up);
  });
  RdmaServerTestPeer::SetPortState(
      node.rsrv.get(), [&ports](size_t i) { return ports.state[i].load(); });
  ResetPortSeam reset_seam{node.rsrv.get()};
  ASSERT_TRUE(WaitFor([&] { return RailPortIs(node, ports.rails[rail], 1); }));
  ports.state[rail] = 0;
  ASSERT_TRUE(WaitFor([&] {
    std::lock_guard<std::mutex> lock(mu);
    return !heard.empty();
  }));
  ports.state[rail] = 1;  // within the second: suppressed for now
  std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  node.rsrv->set_rail_state_listener(nullptr);
  std::lock_guard<std::mutex> lock(mu);
  ASSERT_EQ(heard.size(), 2u);
  EXPECT_FALSE(heard[0]);
  EXPECT_TRUE(heard[1]) << "the port came back and nobody was told";
}

// RDS-02 review: one poll that sees the port down refuses new handshakes
// but keeps the rail's connections (a link retraining within a poll); a
// second poll in a row ends them.
TEST(RdmaServerPorts, AOnePollFlapKeepsTheRailsConnections) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv check("DFKV_RDMA_PORT_CHECK_MS", "20");
  RdmaNode node("port-short-flap");
  ServerPorts ports(node);
  const size_t rail = ports.rails.size() - 1;
  std::atomic<int> down_polls{0};
  RdmaServerTestPeer::SetPortState(
      node.rsrv.get(), [&ports, &down_polls, rail](size_t i) {
        if (i == rail && down_polls.load() > 0) {
          down_polls.fetch_sub(1);
          return 0;
        }
        return ports.state[i].load();
      });
  ResetPortSeam reset_seam{node.rsrv.get()};
  ScopedEnv no_client_watch("DFKV_RDMA_PORT_CHECK_MS", "0");
  RdmaTransport rt(kMaxMsg, ports.rails[rail]);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  const std::string v(4096, 'f');
  ASSERT_TRUE(c.Put("flap", v.data(), v.size()));
  ASSERT_GE(RdmaServerTestPeer::LiveConnectionsOnRail(*node.rsrv, rail), 1u);
  const long changes_before = CounterVal(node.rsrv->MetricsText(),
                                         "dfkv_rdma_rail_port_changes_total");
  down_polls = 1;
  ASSERT_TRUE(WaitFor([&] {
    return CounterVal(node.rsrv->MetricsText(),
                      "dfkv_rdma_rail_port_changes_total") >= changes_before + 2;
  })) << "the one-poll flap was not seen";
  EXPECT_GE(RdmaServerTestPeer::LiveConnectionsOnRail(*node.rsrv, rail), 1u)
      << "a one-poll flap ended the rail's connections";
  EXPECT_EQ(CounterVal(node.rsrv->MetricsText(),
                       "dfkv_rdma_rail_down_connections_ended_total"), 0);
  std::string out(v.size(), '\0');
  EXPECT_TRUE(c.Get("flap", &out[0], out.size()));
  down_polls = 1000000;  // down for good: the second poll ends them
  EXPECT_TRUE(WaitFor([&] {
    return RdmaServerTestPeer::LiveConnectionsOnRail(*node.rsrv, rail) == 0;
  }));
  down_polls = 0;
}

namespace {
// A listener that never accepts: the kernel completes TCP handshakes into its
// backlog and nothing ever answers (a half-dead server).
struct SilentTcp {
  int fd = -1;
  std::string addr;
  SilentTcp() {
    fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0 ||
        ::listen(fd, 64) != 0)
      return;
    socklen_t len = sizeof(a);
    ::getsockname(fd, reinterpret_cast<sockaddr*>(&a), &len);
    addr = "127.0.0.1:" + std::to_string(ntohs(a.sin_port));
  }
  ~SilentTcp() { ::close(fd); }
};
}  // namespace

// A node that takes the TCP connection and never answers fails an operation
// within its timeout. The capability probe's read used to wait its own 10 s
// (DFKV_RDMA_IO_MS) whatever the operation timeout: 10.2 s for a GET with
// the default 5 s. The node is not held unreachable for a dial the deadline
// cut short, so nothing is charged.
TEST(RdmaOpDeadline, ASilentNodeFailsWithinTheOperationTimeout) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv op_timeout("DFKV_RDMA_OP_TIMEOUT_MS", "1000");
  for (const bool named : {false, true}) {
    SCOPED_TRACE(named ? "named" : "static");
    SilentTcp silent;
    ASSERT_FALSE(silent.addr.empty());
    RdmaTransport rt(kMaxMsg);
    if (named) rt.OnPeerTopology(PublishedPeer(silent.addr, "silent-peer"));
    KVClient c({{"n", silent.addr}}, SelfHdr(), &rt);
    std::string out(4096, 0);
    auto t0 = std::chrono::steady_clock::now();
    EXPECT_FALSE(c.Get("silent", &out[0], out.size()));
    double seconds = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0).count();
    EXPECT_LT(seconds, 2.5) << "the GET outlived its 1 s timeout";
    const std::vector<BlockKey> keys = {ToBlockKey(SelfHdr(), "silent")};
    std::vector<char> exists;
    t0 = std::chrono::steady_clock::now();
    EXPECT_EQ(rt.ExistMany(silent.addr, keys, &exists)[0], Status::kIOError);
    seconds = std::chrono::duration<double>(
                  std::chrono::steady_clock::now() - t0).count();
    EXPECT_LT(seconds, 2.5) << "the EXIST outlived its 1 s timeout";
    const std::string text = rt.MetricsText();
    EXPECT_GE(CounterVal(text, "dfkv_rdma_client_op_deadline_expired_total"), 1);
    EXPECT_EQ(CounterVal(text, "dfkv_rdma_client_peer_unreachable_total"), 0);
  }
}

// A server under receive-pool pressure holds a new connection's bootstrap
// reply for up to 5 s. An operation whose deadline runs out meanwhile fails
// on time, and that wait is not taken as evidence against the peer's rail:
// nothing is charged or cooled (the read was cut short, not a full
// DFKV_RDMA_IO_MS of silence).
TEST(RdmaOpDeadline, ABootstrapCutShortByTheDeadlineChargesNothing) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("deadline-pressure", kMaxMsg);
  ScopedEnv op_timeout("DFKV_RDMA_OP_TIMEOUT_MS", "1000");
  RdmaTransport rt(kMaxMsg);
  rt.OnPeerTopology(PublishedPeer(node.addr, "pressured-peer"));
  const BlockKey key = ToBlockKey(SelfHdr(), "pressured");
  // The capabilities are learned on a first, quick connection.
  const std::vector<BlockKey> keys = {key};
  std::vector<char> exists;
  ASSERT_NE(rt.ExistMany(node.addr, keys, &exists)[0], Status::kIOError);
  RdmaServerTestPeer::SetBootstrapReplyDelayMs(*node.rsrv, 3000);
  std::vector<char> out(4096);
  const auto t0 = std::chrono::steady_clock::now();
  const auto statuses = rt.RangeInto(
      node.addr, {key}, {RangeDst{out.data(), out.size()}}, nullptr);
  const double seconds = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0).count();
  RdmaServerTestPeer::SetBootstrapReplyDelayMs(*node.rsrv, 0);
  EXPECT_NE(statuses[0], Status::kOk);
  EXPECT_LT(seconds, 2.5);
  const std::string text = rt.MetricsText();
  EXPECT_GE(CounterVal(text, "dfkv_rdma_client_op_deadline_expired_total"), 1);
  EXPECT_EQ(CounterVal(text, "dfkv_rdma_client_remote_rail_failures_total"), 0)
      << "a bootstrap the deadline cut short was charged to the rail";
  EXPECT_EQ(CounterVal(text, "dfkv_rdma_client_remote_rail_cooldowns_total"), 0);
}

// A confirming dial that is refused outright (ECONNREFUSED, as a backlog
// overflowing in a handshake storm can answer) does not hold the node
// unreachable either.
TEST(RdmaPeerRailDown, ARefusedRefusalCheckDoesNotHoldTheNode) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  // Bound, never listening: a connect to it is refused at once.
  const int dead = ::socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(dead, 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  socklen_t length = sizeof(address);
  ASSERT_EQ(::bind(dead, reinterpret_cast<sockaddr*>(&address),
                   sizeof(address)), 0);
  ASSERT_EQ(::getsockname(dead, reinterpret_cast<sockaddr*>(&address),
                          &length), 0);
  const std::string dead_addr =
      "127.0.0.1:" + std::to_string(ntohs(address.sin_port));
  RdmaTransport rt(kMaxMsg);
  EXPECT_EQ(RdmaTransportTestPeer::AcquireRefusalCheck(&rt, dead_addr, 0),
            nullptr);
  EXPECT_EQ(CounterVal(rt.MetricsText(),
                       "dfkv_rdma_client_peer_unreachable_total"), 0);
  ::close(dead);
}

// Wake() ends a busy-poll wait at once: the loop sees a flag (it used to
// poll the wake pipe with a system call on every spin).
TEST(RdmaEndpointWait, WakeEndsABusyPollWait) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const char* devs = std::getenv("DFKV_RDMA_DEV");
  const std::string first =
      devs ? std::string(devs).substr(0, std::string(devs).find(',')) : "";
  rdma::RcEndpoint busy;
  busy.set_busy_poll(true);
  ASSERT_TRUE(busy.Open(first.empty() ? nullptr : first.c_str(), 4096, 1));
  std::thread waker([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    busy.Wake();
  });
  ibv_wc wc{};
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_EQ(busy.WaitComp(&wc, 1, 5000), -1);
  EXPECT_LT(std::chrono::steady_clock::now() - t0, std::chrono::seconds(2));
  waker.join();
}

// The spin before sleeping: a Wake() ends it at once, it comes out of the
// wait's timeout (not on top of it), and a completion that arrives during it
// is returned without arming the completion channel.
TEST(RdmaEndpointWait, TheSpinBeforeSleepingEndsOnWakeTimeoutOrCompletion) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const char* devs = std::getenv("DFKV_RDMA_DEV");
  const std::string first =
      devs ? std::string(devs).substr(0, std::string(devs).find(',')) : "";
  const char* dev = first.empty() ? nullptr : first.c_str();
  ibv_wc wc{};
  {
    rdma::RcEndpoint spinning;
    spinning.set_spin_us(1000000);  // one second
    ASSERT_TRUE(spinning.Open(dev, 4096, 1));
    std::thread waker([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      spinning.Wake();
    });
    const auto t0 = std::chrono::steady_clock::now();
    EXPECT_EQ(spinning.WaitComp(&wc, 1, -1), -1);
    EXPECT_LT(std::chrono::steady_clock::now() - t0,
              std::chrono::milliseconds(200))
        << "the spin did not see the Wake";
    waker.join();
  }
  {
    rdma::RcEndpoint spinning;
    spinning.set_spin_us(1000000);
    ASSERT_TRUE(spinning.Open(dev, 4096, 1));
    const auto t0 = std::chrono::steady_clock::now();
    EXPECT_EQ(spinning.WaitComp(&wc, 1, 30), 0);
    const auto took = std::chrono::steady_clock::now() - t0;
    EXPECT_GE(took, std::chrono::milliseconds(25));
    EXPECT_LT(took, std::chrono::milliseconds(300))
        << "the spin was added to the 30 ms timeout";
  }
  {
    rdma::RcEndpoint target, peer;
    target.set_spin_us(1000000);
    ASSERT_TRUE(target.Open(dev, 4096, 1));
    ASSERT_TRUE(peer.Open(dev, 4096, 1));
    const rdma::QpInfo ti = target.Local(), pi = peer.Local();
    ASSERT_TRUE(target.Connect(pi));
    ASSERT_TRUE(peer.Connect(ti));
    ASSERT_TRUE(target.PostRecv(0));
    std::thread sender([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      std::memset(peer.sbuf(0), 'x', 64);
      EXPECT_TRUE(peer.PostSend(0, 64));
      ibv_wc sent{};
      EXPECT_EQ(peer.WaitComp(&sent, 1, 5000), 1);
    });
    const uint64_t arms = target.NotifyArms();
    EXPECT_EQ(target.WaitComp(&wc, 1, 5000), 1);
    EXPECT_EQ(wc.status, IBV_WC_SUCCESS);
    EXPECT_EQ(target.NotifyArms(), arms)
        << "the receive arrived during the spin, yet the wait slept";
    sender.join();
  }
}

// A signal that interrupts a completion wait does not start its timeout
// again: under a signal every 20 ms (shorter than the wait), a 200 ms wait
// still ends after about 200 ms. It used to restart on each EINTR and never
// end while the signals kept coming.
TEST(RdmaEndpointWait, ASignalDoesNotRestartTheTimeout) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const char* devs = std::getenv("DFKV_RDMA_DEV");
  const std::string first =
      devs ? std::string(devs).substr(0, std::string(devs).find(',')) : "";
  rdma::RcEndpoint ep;
  ASSERT_TRUE(ep.Open(first.empty() ? nullptr : first.c_str(), 4096, 1));
  struct sigaction action{}, previous{};
  action.sa_handler = [](int) {};
  sigemptyset(&action.sa_mask);
  action.sa_flags = 0;  // no SA_RESTART: poll() returns EINTR
  ASSERT_EQ(::sigaction(SIGUSR2, &action, &previous), 0);
  std::atomic<bool> done{false};
  const pthread_t waiter = ::pthread_self();
  std::thread signaller([&] {
    const auto stop = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!done.load() && std::chrono::steady_clock::now() < stop) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      if (!done.load()) ::pthread_kill(waiter, SIGUSR2);
    }
  });
  ibv_wc wc{};
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_EQ(ep.WaitComp(&wc, 1, 200), 0);
  const auto took = std::chrono::steady_clock::now() - t0;
  done.store(true);
  signaller.join();
  ::sigaction(SIGUSR2, &previous, nullptr);
  EXPECT_GE(took, std::chrono::milliseconds(190));
  EXPECT_LT(took, std::chrono::milliseconds(1000))
      << "each signal restarted the wait's timeout";
}

// A batch GET into the callers' buffers (the connectors' dfkv_batch_get_auto
// and _sg) runs in windows on one connection: one rail selection for the
// whole batch, not one per key. Hits, misses and a value larger than its
// destination keep their own statuses and lengths.
TEST(RdmaBatchGet, KeysOfABatchShareWindowsOnOneConnection) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("batch-get-window");
  RdmaTransport rt(kMaxMsg);
  constexpr size_t kKeys = 16;
  constexpr size_t kSize = 8192;
  std::vector<BlockKey> keys;
  for (size_t i = 0; i < kKeys; ++i) {
    keys.push_back(ToBlockKey(SelfHdr(), "window-" + std::to_string(i)));
    if (i % 5 == 3) continue;  // a miss
    const std::string value(i == 7 ? 2 * kSize : kSize, static_cast<char>('a' + i));
    ASSERT_EQ(rt.Cache(node.addr, keys.back(), value.data(), value.size()),
              Status::kOk);
  }
  std::vector<char> pool(kKeys * kSize);
  std::vector<RangeDst> dsts;
  for (size_t i = 0; i < kKeys; ++i)
    dsts.push_back(RangeDst{pool.data() + i * kSize, kSize});
  const uint64_t before =
      MetricSum(rt.MetricsText(), "dfkv_rdma_client_rail_selections_total{");
  std::vector<uint64_t> lens;
  const auto statuses = rt.RangeInto(node.addr, keys, dsts, &lens);
  const uint64_t selections =
      MetricSum(rt.MetricsText(), "dfkv_rdma_client_rail_selections_total{") -
      before;
  EXPECT_EQ(selections, 1u) << "one connection for the batch";
  for (size_t i = 0; i < kKeys; ++i) {
    SCOPED_TRACE(i);
    if (i % 5 == 3) {
      EXPECT_EQ(statuses[i], Status::kNotFound);
    } else if (i == 7) {
      EXPECT_EQ(statuses[i], Status::kInvalid);
      EXPECT_EQ(lens[i], 2 * kSize) << "a value too large reports its length";
    } else {
      ASSERT_EQ(statuses[i], Status::kOk);
      EXPECT_EQ(lens[i], kSize);
      EXPECT_EQ(pool[i * kSize], static_cast<char>('a' + i));
      EXPECT_EQ(pool[i * kSize + kSize - 1], static_cast<char>('a' + i));
    }
  }
  // The same through the scatter-gather call, two segments per key.
  std::vector<char> sg(kKeys * kSize);
  std::vector<RangeDstMulti> multi(kKeys);
  for (size_t i = 0; i < kKeys; ++i) {
    multi[i].payloads.emplace_back(sg.data() + i * (kSize / 2), kSize / 2);
    multi[i].payloads.emplace_back(
        sg.data() + kKeys * (kSize / 2) + i * (kSize / 2), kSize / 2);
  }
  std::vector<size_t> sg_lens;
  const uint64_t sg_before =
      MetricSum(rt.MetricsText(), "dfkv_rdma_client_rail_selections_total{");
  const auto sg_statuses = rt.RangeIntoMulti(node.addr, keys, multi, &sg_lens);
  EXPECT_EQ(MetricSum(rt.MetricsText(),
                      "dfkv_rdma_client_rail_selections_total{") -
                sg_before,
            1u)
      << "one connection for the scatter-gather batch";
  for (size_t i = 0; i < kKeys; ++i) {
    SCOPED_TRACE(i);
    if (i % 5 == 3) {
      EXPECT_EQ(sg_statuses[i], Status::kNotFound);
    } else if (i == 7) {
      EXPECT_EQ(sg_statuses[i], Status::kInvalid);
      EXPECT_EQ(sg_lens[i], 2 * kSize);
    } else {
      ASSERT_EQ(sg_statuses[i], Status::kOk);
      EXPECT_EQ(sg_lens[i], kSize);
      EXPECT_EQ(sg[i * (kSize / 2)], static_cast<char>('a' + i));
      EXPECT_EQ(sg[kKeys * (kSize / 2) + i * (kSize / 2) + kSize / 2 - 1],
                static_cast<char>('a' + i));
    }
  }
}

namespace {
// 16 keys of 8 KiB on a node; their destinations in `pool`.
std::vector<BlockKey> StoreBatch(RdmaTransport& rt, const RdmaNode& node,
                                 const std::string& tag,
                                 std::vector<char>* pool,
                                 std::vector<RangeDst>* dsts) {
  constexpr size_t kKeys = 16, kSize = 8192;
  std::vector<BlockKey> keys;
  pool->assign(kKeys * kSize, 0);
  dsts->clear();
  for (size_t i = 0; i < kKeys; ++i) {
    keys.push_back(ToBlockKey(SelfHdr(), tag + std::to_string(i)));
    const std::string value(kSize, static_cast<char>('A' + i));
    EXPECT_EQ(rt.Cache(node.addr, keys.back(), value.data(), value.size()),
              Status::kOk);
    dsts->push_back(RangeDst{pool->data() + i * kSize, kSize});
  }
  return keys;
}
}  // namespace

// One window of a batch fails (here its first READ): its keys fail, the
// windows before it keep their results, and the windows after it, which no
// READ had touched, are read on a fresh connection, as each key once had
// its own.
TEST(RdmaBatchGet, AFailedWindowFailsOnlyItsOwnKeys) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");  // windows of 4: keys 4..7 are #2
  RdmaNode node("batch-get-partial");
  RdmaTransport rt(kMaxMsg);
  std::vector<char> pool;
  std::vector<RangeDst> dsts;
  const auto keys = StoreBatch(rt, node, "partial-", &pool, &dsts);
  std::vector<uint64_t> lens;
  std::vector<Status> statuses;
  {
    ScopedEnv fault("DFKV_RDMA_TEST_PULL_READ_FAILURE", "@5");
    statuses = rt.RangeInto(node.addr, keys, dsts, &lens);
  }
  for (size_t i = 0; i < keys.size(); ++i) {
    SCOPED_TRACE(i);
    if (i >= 4 && i < 8) {
      EXPECT_EQ(statuses[i], Status::kIOError);
      continue;
    }
    ASSERT_EQ(statuses[i], Status::kOk);
    EXPECT_EQ(lens[i], 8192u);
    EXPECT_EQ(pool[i * 8192], static_cast<char>('A' + i));
    EXPECT_EQ(pool[i * 8192 + 8191], static_cast<char>('A' + i));
  }
}

// A batch that finds its pooled connection reclaimed by the server reads
// the failed window again on a new connection, once, and every key arrives.
// The client has one rail: with two, the second batch may pick the other
// rail by its latency estimate, find no pooled connection there and never
// meet the stale one (4 of 50 runs on two rails).
TEST(RdmaBatchGet, AStalePooledConnectionIsRetriedOnce) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  if (SoftRoceOnly()) GTEST_SKIP() << kRxeNoRetryExceeded;
  ReclaimingServerEnv env;
  RdmaNode node("batch-get-stale");
  RdmaTransport rt(kMaxMsg, node.rsrv->DeviceNames().front());
  std::vector<char> pool;
  std::vector<RangeDst> dsts;
  const auto keys = StoreBatch(rt, node, "stale-", &pool, &dsts);
  for (Status st : rt.RangeInto(node.addr, keys, dsts, nullptr))
    ASSERT_EQ(st, Status::kOk);  // leaves a pooled data connection
  ASSERT_TRUE(ServerHasNoConnections(node));
  std::fill(pool.begin(), pool.end(), 0);
  const long retries_before = CounterVal(
      rt.MetricsText(), "dfkv_rdma_client_stale_pool_retries_total");
  const auto statuses = rt.RangeInto(node.addr, keys, dsts, nullptr);
  for (size_t i = 0; i < keys.size(); ++i) {
    SCOPED_TRACE(i);
    ASSERT_EQ(statuses[i], Status::kOk);
    EXPECT_EQ(pool[i * 8192], static_cast<char>('A' + i));
  }
  EXPECT_EQ(CounterVal(rt.MetricsText(),
                       "dfkv_rdma_client_stale_pool_retries_total") -
                retries_before,
            1);
}

// A scatter-gather GET into memory that is not registered registers it per
// GET, but segments that touch share one registration: 16 keys of 61
// contiguous 16 KiB segments take 16 registrations, not 976. Segments in
// registered memory use that registration; scattered ones still take one
// each; a merged range the device refuses is registered segment by segment,
// counted. Every byte lands where it belongs in each case.
TEST(RdmaSgRead, TouchingUnregisteredSegmentsShareARegistration) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const char* devs = std::getenv("DFKV_RDMA_DEV");
  const std::string first =
      devs ? std::string(devs).substr(0, std::string(devs).find(',')) : "";
  constexpr size_t kKeys = 16, kSegs = 61, kSeg = 16384;
  RdmaNode node("sg-coalesce", 1u << 20);
  ScopedEnv block("DFKV_RDMA_MAX_BLOCK_BYTES", "1048576");
  std::vector<BlockKey> keys;
  std::vector<std::string> values;
  {
    RdmaTransport writer(1u << 20, first);
    for (size_t k = 0; k < kKeys; ++k) {
      values.push_back(PatternValue(kSegs * kSeg, 60 + k));
      keys.push_back(ToBlockKey(SelfHdr(), "coalesce-" + std::to_string(k)));
      ASSERT_EQ(writer.Cache(node.addr, keys[k], values[k].data(),
                             values[k].size()),
                Status::kOk);
    }
  }
  enum Layout {
    kContiguous, kScattered, kHalfRegistered, kSpanRefused, kMemlock
  };
  for (const Layout layout :
       {kContiguous, kScattered, kHalfRegistered, kSpanRefused, kMemlock}) {
    SCOPED_TRACE(static_cast<int>(layout));
    RdmaTransport rt(1u << 20, first);
    std::vector<char> pool(kKeys * kSegs * kSeg + 4096, 'G');
    std::vector<std::unique_ptr<char[]>> owned;
    std::vector<char> registered(kKeys * kSegs * kSeg, 'G');
    if (layout == kHalfRegistered)
      ASSERT_TRUE(rt.RegisterMemory(registered.data(), registered.size()));
    std::vector<RangeDstMulti> dsts(kKeys);
    size_t expected = 0;
    for (size_t k = 0; k < kKeys; ++k) {
      for (size_t s = 0; s < kSegs; ++s) {
        char* at = pool.data() + (k * kSegs + s) * kSeg;
        if (layout == kScattered) {
          owned.emplace_back(new char[kSeg + 128]);
          at = owned.back().get() + 64;
        } else if (layout == kHalfRegistered && s % 2 == 0) {
          at = registered.data() + (k * kSegs + s) * kSeg;
        }
        dsts[k].payloads.emplace_back(at, kSeg);
      }
      // Registrations expected for this key.
      expected += layout == kContiguous ? 1
                  : layout == kHalfRegistered ? kSegs / 2  // odd ones, apart
                                              : kSegs;
    }
    if (layout == kSpanRefused)
      rdma::RcEndpoint::SetTransientRegistrationSeamForTest(
          [](void*, size_t len) { return len <= kSeg; });
    // A locked-memory limit: no merged range, and at most 40 one-shot
    // registrations alive at a time (ENOMEM past it). A READ round needs at
    // most ReadSge (<= 30) of them.
    if (layout == kMemlock)
      rdma::RcEndpoint::SetTransientRegistrationSeamForTest(
          [](void*, size_t len) {
            return len <= kSeg &&
                   rdma::RcEndpoint::TransientUserMrActive() < 40;
          });
    const std::string before = rt.MetricsText();
    std::vector<size_t> lens;
    const auto statuses = rt.RangeIntoMulti(node.addr, keys, dsts, &lens);
    rdma::RcEndpoint::SetTransientRegistrationSeamForTest(nullptr);
    const std::string after = rt.MetricsText();
    for (size_t k = 0; k < kKeys; ++k) {
      ASSERT_EQ(statuses[k], Status::kOk) << k;
      size_t offset = 0;
      for (const auto& segment : dsts[k].payloads) {
        ASSERT_EQ(std::memcmp(segment.first, values[k].data() + offset, kSeg),
                  0)
            << "key " << k << " offset " << offset;
        offset += kSeg;
      }
    }
    EXPECT_EQ(pool.back(), 'G') << "past the last destination";
    const auto delta = [&](const char* name) {
      return CounterVal(after, name) - CounterVal(before, name);
    };
    if (layout != kMemlock)
      EXPECT_EQ(
          delta("dfkv_rdma_client_unregistered_dest_registrations_total"),
          static_cast<long>(expected));
    EXPECT_EQ(delta("dfkv_rdma_client_unregistered_dest_span_fallbacks_total"),
              layout == kSpanRefused || layout == kMemlock
                  ? static_cast<long>(kKeys) : 0);
    EXPECT_EQ(
        delta("dfkv_rdma_client_unregistered_dest_memlock_fallbacks_total"),
        layout == kMemlock ? static_cast<long>(kKeys) : 0);
    EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), 0u);
  }
}

// ---- Server memory windows (DFKV_RDMA_SERVER_MW) ---------------------------

// ---- Server memory windows (DFKV_RDMA_SERVER_MW) ---------------------------

TEST(RdmaServerMw, ReservationChargesGrantUntilQuarantineExpiry) {
  ScopedEnv mw("DFKV_RDMA_SERVER_MW", "1");
  ScopedEnv cap("DFKV_RDMA_SERVER_MW_QUARANTINE_BYTES", "1024");
  ScopedEnv max_ms("DFKV_RDMA_SERVER_MW_QUARANTINE_MAX_MS", "5000");
  ScopedEnv link("DFKV_RDMA_SERVER_MW_LINK_GBPS", "1");
  RdmaServer server([](uint8_t, const BlockKey&, uint64_t, uint64_t,
                       const char*, uint64_t, std::string*, size_t*) {
    return Status::kOk;
  });
  std::atomic<uint64_t> rail{0};
  // A concurrent admission can alter the live rail counter. The snapshot
  // stays an upper bound and never clips to max at quarantine time.
  const uint64_t admitted =
      RdmaServerTestPeer::MwRailSnapshot(server, rail, 256);
  auto granted = RdmaServerTestPeer::MwReserveQuarantine(&server, admitted, 1024);
  ASSERT_TRUE(granted.held());
  rail.store(1u << 20);
  auto moved = std::move(granted);
  EXPECT_EQ(RdmaServerTestPeer::MwReservedBytes(server), 1024u);
  EXPECT_EQ(RdmaServerTestPeer::MwQuarantinedBytes(server), 0u);
  auto over_cap = RdmaServerTestPeer::MwReserveQuarantine(&server, 1, 1);
  EXPECT_FALSE(over_cap.held());
  RdmaServerTestPeer::MwQuarantine(&server, std::move(moved), admitted, 1024);
  EXPECT_EQ(RdmaServerTestPeer::MwReservedBytes(server), 0u);
  EXPECT_EQ(RdmaServerTestPeer::MwQuarantinedBytes(server), 1024u);
  // Expiry lets a later 512-byte grant convert; another 512-byte reservation
  // completes the cap. No remaining capacity now: the refusals own nothing.
  RdmaServerTestPeer::MwReleaseQuarantine(&server);
  EXPECT_EQ(RdmaServerTestPeer::MwQuarantinedBytes(server), 0u);
  granted = RdmaServerTestPeer::MwReserveQuarantine(&server, admitted, 512);
  ASSERT_TRUE(granted.held());
  const auto max_refused =
      RdmaServerTestPeer::MwReserveQuarantine(&server, admitted, 513);
  EXPECT_FALSE(max_refused.held());
  // A failed second claim has no cost: the first stays fully reserved.
  EXPECT_EQ(RdmaServerTestPeer::MwReservedBytes(server), 512u);
  RdmaServerTestPeer::MwQuarantine(&server, std::move(granted), admitted, 512);
  granted = RdmaServerTestPeer::MwReserveQuarantine(&server, admitted, 512);
  ASSERT_TRUE(granted.held());
  // Actual 512 + reserved 512 fills the cap; conversion is not double charge.
  over_cap = RdmaServerTestPeer::MwReserveQuarantine(&server, admitted, 1);
  EXPECT_FALSE(over_cap.held());
  // Destructor cancellation is idempotent and returns exactly its own bytes.
  granted = {};
  EXPECT_EQ(RdmaServerTestPeer::MwReservedBytes(server), 0u);
  EXPECT_EQ(RdmaServerTestPeer::MwQuarantinedBytes(server), 512u);
  RdmaServerTestPeer::MwReleaseQuarantine(&server);
  EXPECT_EQ(RdmaServerTestPeer::MwQuarantinedBytes(server), 0u);
  EXPECT_EQ(RdmaServerTestPeer::MwReservedBytes(server), 0u);
  server.Stop();
}

TEST(RdmaServerMw, ConcurrentReservationsCannotOvershootTheCap) {
  ScopedEnv mw("DFKV_RDMA_SERVER_MW", "1");
  ScopedEnv cap("DFKV_RDMA_SERVER_MW_QUARANTINE_BYTES", "1024");
  RdmaServer server([](uint8_t, const BlockKey&, uint64_t, uint64_t,
                       const char*, uint64_t, std::string*, size_t*) {
    return Status::kOk;
  });
  constexpr size_t kThreads = 8;
  std::vector<RdmaServerTestPeer::MwReservation> claims(kThreads);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (size_t i = 0; i < kThreads; ++i)
    threads.emplace_back([&, i] {
      claims[i] = RdmaServerTestPeer::MwReserveQuarantine(&server, 1024, 256);
    });
  for (auto& thread : threads) thread.join();
  size_t admitted = 0;
  for (const auto& claim : claims) admitted += claim.held() ? 1 : 0;
  EXPECT_EQ(admitted, 4u);
  EXPECT_EQ(RdmaServerTestPeer::MwReservedBytes(server), 1024u);
  EXPECT_EQ(RdmaServerTestPeer::MwQuarantinedBytes(server), 0u);
  for (auto& claim : claims) claim = {};
  EXPECT_EQ(RdmaServerTestPeer::MwReservedBytes(server), 0u);
  server.Stop();
}

namespace {

// Sets RdmaNode::before_start for one node's construction.
struct BeforeStart {
  explicit BeforeStart(std::function<void(RdmaServer&)> hook) {
    RdmaNode::before_start = std::move(hook);
  }
  ~BeforeStart() { RdmaNode::before_start = nullptr; }
};

// The windowed grant's life as the server reports it, in order.
struct MwEvents {
  struct Event {
    std::string name;
    const void* at;
    std::chrono::steady_clock::time_point when;
  };
  std::mutex mu;
  std::vector<Event> events;
  void Note(const char* name, const void* at) {
    std::lock_guard<std::mutex> lock(mu);
    events.push_back({name, at, std::chrono::steady_clock::now()});
  }
  std::vector<Event> Snapshot() {
    std::lock_guard<std::mutex> lock(mu);
    return events;
  }
  // The events of the range at `at`, in order.
  std::vector<Event> For(const void* at) {
    std::lock_guard<std::mutex> lock(mu);
    std::vector<Event> out;
    for (const auto& event : events)
      if (event.at == at) out.push_back(event);
    return out;
  }
  bool WaitFor(const void* at, const std::string& name, int timeout_ms) {
    const auto until = std::chrono::steady_clock::now() +
                       std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < until) {
      for (const auto& event : For(at))
        if (event.name == name) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
  }
};

// 1 if the server's first rail binds GET grants with windows.
long MwMode(const RdmaServer& server) {
  const std::string text = server.MetricsText();
  const std::string name = "dfkv_rdma_server_mw_mode{";
  const size_t at = text.find("\n" + name);
  if (at == std::string::npos) return -1;
  const size_t value = text.find("} ", at);
  return std::strtol(text.c_str() + value + 2, nullptr, 10);
}
long MwCounter(const RdmaServer& server, const char* name) {
  return CounterVal(server.MetricsText(), name);
}
constexpr const char* kNoWindows =
    "the device offers no type-2B memory windows";

// A small RAM tier of one extent and one shard: a pinned value keeps its
// slot, so whether its range is reused shows in a promotion.
struct SmallRamTier {
  ScopedEnv ram_on{"DFKV_RAM_TIER", "1"};
  ScopedEnv ram_bytes{"DFKV_RAM_TIER_BYTES", "1048576"};
  ScopedEnv extent{"DFKV_RAM_TIER_EXTENT_BYTES", "1048576"};
  ScopedEnv reserve{"DFKV_RAM_TIER_LARGE_RESERVE_BYTES", "0"};
  ScopedEnv shards{"DFKV_RAM_TIER_SHARDS", "1"};
  ScopedEnv reclaim{"DFKV_RAM_RECLAIM_MS", "0"};
};

// Prepares `key` until a grant is bound by a window (the first ones use
// per-grant MRs while the region registers); false if none is within 5 s.
bool PrepareWindowed(RdmaNode& node, PinnedPullPeer& peer, const BlockKey& key,
                     size_t offset, size_t length,
                     rdma::DynamicPullReady* ready) {
  const auto until =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < until) {
    const long binds =
        MwCounter(*node.rsrv, "dfkv_rdma_server_mw_binds_total");
    if (!peer.Prepare(key, offset, length, ready)) {
      ADD_FAILURE() << "the PREPARE failed";
      return false;
    }
    if (MwCounter(*node.rsrv, "dfkv_rdma_server_mw_binds_total") > binds)
      return true;
    if (!peer.Release(key, *ready)) {
      ADD_FAILURE() << "the RELEASE failed";
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return false;
}

}  // namespace

// A GET of a RAM-tier value is read through a memory window. At its
// RELEASE the window is invalidated, and only after the invalidation's
// completion and then the quarantine is the range given back: the value's
// slot stays pinned meanwhile (a promotion into the one-slot extent fails),
// and is reused only after. The rkey the client held reads nothing after
// the RELEASE.
TEST(RdmaServerMw, AGrantIsReadThroughAWindowAndWaitsOutTheQuarantine) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv mw("DFKV_RDMA_SERVER_MW", "1");
  SmallRamTier ram_env;
  MwEvents events;
  BeforeStart seam([&](RdmaServer& server) {
    RdmaServerTestPeer::SetMwEvents(
        server, [&](const char* name, const void* at) { events.Note(name, at); });
  });
  RdmaNode node("mw-arena");
  if (MwMode(*node.rsrv) != 1) GTEST_SKIP() << kNoWindows;
  auto& ram = KvNodeServerTestPeer::Ram(*node.srv);
  const BlockKey key = ToBlockKey(SelfHdr(), "windowed");
  const BlockKey replacement = ToBlockKey(SelfHdr(), "replacement");
  const std::string value = PatternValue(700000, 81);
  const std::string other = PatternValue(value.size(), 82);
  ASSERT_TRUE(KvNodeServerTestPeer::Promote(*node.srv, key, value));
  PinnedPullPeer peer;
  ASSERT_TRUE(peer.Open(node));
  const uint64_t mrs = rdma::RcEndpoint::LeaseReadMrActive();
  rdma::DynamicPullReady ready;
  ASSERT_TRUE(PrepareWindowed(node, peer, key, 97, 200013, &ready));
  EXPECT_EQ(rdma::RcEndpoint::LeaseReadMrActive(), mrs) << "no MR for it";
  std::string out;
  ASSERT_TRUE(peer.Read(ready, &out));
  EXPECT_EQ(out, value.substr(97, 200013));
  const void* at = reinterpret_cast<const void*>(ready.address);
  EXPECT_TRUE(events.For(at).empty());

  ASSERT_TRUE(ram.Remove(key));
  ASSERT_TRUE(peer.Release(key, ready));
  ASSERT_TRUE(events.WaitFor(at, "quarantined", 2000));
  // Released only after the quarantine: the slot is still pinned.
  if (events.For(at).size() == 2)
    EXPECT_FALSE(KvNodeServerTestPeer::Promote(*node.srv, replacement, other));
  ASSERT_TRUE(events.WaitFor(at, "released", 2000));
  const auto life = events.For(at);
  ASSERT_EQ(life.size(), 3u);
  EXPECT_EQ(life[0].name, "invalidated");
  EXPECT_EQ(life[1].name, "quarantined");
  EXPECT_EQ(life[2].name, "released");
  EXPECT_GE(life[2].when - life[1].when, std::chrono::milliseconds(5))
      << "released before the 5 ms floor";
  EXPECT_TRUE(KvNodeServerTestPeer::Promote(*node.srv, replacement, other));
  EXPECT_EQ(MwCounter(*node.rsrv, "dfkv_rdma_server_mw_invalidations_total"),
            MwCounter(*node.rsrv, "dfkv_rdma_server_mw_binds_total"));

  // The key the client was given reads nothing now.
  ibv_mr* mr = peer.ep.RegisterTransient(out.data(), out.size(), true);
  ASSERT_NE(mr, nullptr);
  ibv_wc wc{};
  ASSERT_TRUE(peer.ep.PostRead(0, out.data(), out.size(), mr, ready.address,
                               ready.rkey));
  ASSERT_EQ(peer.ep.WaitComp(&wc, 1, 10000), 1);
  EXPECT_EQ(wc.status, IBV_WC_REM_ACCESS_ERR);
}

// A GET staged from the disk keeps its per-grant MR, in the synchronous and
// the io_uring serve loop: a READ of memory the disk has just written into
// is slow through a registration older than the write (RDS-29). A leased PUT
// keeps its per-grant MR too. No receive chunk is registered for windows.
TEST(RdmaServerMw, StagedGetsAndLeasedPutsKeepTheirMrs) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv mw("DFKV_RDMA_SERVER_MW", "1");
  // Windows need a RAM arena; the value is then dropped from it, and a read
  // promotes it back only if read again within 1 ms.
  SmallRamTier ram_env;
  ScopedEnv promote("DFKV_RAM_PROMOTE", "second");
  ScopedEnv promote_window("DFKV_RAM_PROMOTE_WINDOW_MS", "1");
  for (const bool uring : {false, true}) {
    SCOPED_TRACE(uring ? "uring" : "sync");
    ScopedEnv uring_env("DFKV_SERVER_URING", uring ? "1" : "0");
    RdmaNode node(uring ? "mw-staged-uring" : "mw-staged-sync", kMaxMsg,
                  false, /*prepare_reads=*/uring);
    if (MwMode(*node.rsrv) != 1) GTEST_SKIP() << kNoWindows;
    ScopedEnv inline_max("DFKV_RDMA_INLINE_PUT_MAX_BYTES", "65536");
    RdmaTransport rt(kMaxMsg);
    // Above the inline PUT (a lease), within the client's block bound.
    const std::string value = PatternValue(200 * 1024 + 11, 91);
    const BlockKey key = ToBlockKey(SelfHdr(), "staged");
    const long leases =
        MwCounter(*node.rsrv, "dfkv_rdma_leaseput_ops_total");
    ASSERT_EQ(rt.Cache(node.addr, key, value.data(), value.size()),
              Status::kOk);
    EXPECT_GT(MwCounter(*node.rsrv, "dfkv_rdma_leaseput_ops_total"), leases)
        << "the PUT did not take a lease";
    EXPECT_EQ(MwCounter(*node.rsrv, "dfkv_rdma_server_mw_binds_total"), 0)
        << "a PUT lease was bound by a window";
    // Only on disk from here: the RAM copy goes once it is written.
    RamTier& ram = KvNodeServerTestPeer::Ram(*node.srv);
    const auto until =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!ram.DropDurableCopy(key) &&
           std::chrono::steady_clock::now() < until)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    ASSERT_FALSE(ram.Contains(key)) << "the value stayed in the RAM tier";
    const long grants = MwCounter(
        *node.rsrv, "dfkv_rdma_server_grant_register_seconds_count");
    for (int i = 0; i < 20; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(3));
      std::string out(value.size(), '\0');
      std::vector<uint64_t> lens;
      ASSERT_EQ(rt.RangeInto(node.addr, {key}, {{out.data(), out.size()}},
                             &lens),
                std::vector<Status>{Status::kOk});
      ASSERT_EQ(out, value);
    }
    EXPECT_GE(MwCounter(*node.rsrv,
                        "dfkv_rdma_server_grant_register_seconds_count") -
                  grants,
              20);
    EXPECT_EQ(MwCounter(*node.rsrv, "dfkv_rdma_server_mw_binds_total"), 0);
    EXPECT_EQ(MwCounter(*node.rsrv, "dfkv_rdma_server_mw_fallback_grants_total"),
              0)
        << "a staged GET counted as a window fallback";
    EXPECT_EQ(MwCounter(*node.rsrv,
                        "dfkv_rdma_server_mw_region_registrations_total"),
              0)
        << "a receive chunk was registered for windows";
    EXPECT_FALSE(ram.Contains(key)) << "a GET promoted the value";
  }
}

// Without a RAM arena nothing can be bound: with the switch on, connections
// allocate no windows, no region is registered, and the mode gauge says 0.
TEST(RdmaServerMw, WithoutARamArenaNoWindowsAreAllocated) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv mw("DFKV_RDMA_SERVER_MW", "1");
  {
    SmallRamTier ram_env;
    RdmaNode probe("mw-arena-probe");
    if (MwMode(*probe.rsrv) != 1) GTEST_SKIP() << kNoWindows;
  }
  ScopedEnv ram_off("DFKV_RAM_TIER", "0");
  std::atomic<int> allocations{0};
  rdma::RcEndpoint::SetWindowAllocSeamForTest([&] {
    allocations.fetch_add(1);
    return true;
  });
  struct Restore {
    ~Restore() { rdma::RcEndpoint::SetWindowAllocSeamForTest(nullptr); }
  } restore;
  RdmaNode node("mw-no-arena");
  EXPECT_EQ(MwMode(*node.rsrv), 0);
  EXPECT_FALSE(RdmaServerTestPeer::MwRegistryStarted(*node.rsrv));
  RdmaTransport rt(kMaxMsg);
  const std::string value = PatternValue(200 * 1024, 93);
  const BlockKey key = ToBlockKey(SelfHdr(), "no-arena");
  ASSERT_EQ(rt.Cache(node.addr, key, value.data(), value.size()), Status::kOk);
  for (int i = 0; i < 5; ++i) {
    std::string out(value.size(), '\0');
    std::vector<uint64_t> lens;
    ASSERT_EQ(rt.RangeInto(node.addr, {key}, {{out.data(), out.size()}}, &lens),
              std::vector<Status>{Status::kOk});
    ASSERT_EQ(out, value);
  }
  EXPECT_EQ(allocations.load(), 0) << "a connection allocated windows";
  EXPECT_EQ(MwCounter(*node.rsrv, "dfkv_rdma_server_mw_binds_total"), 0);
  EXPECT_EQ(MwCounter(*node.rsrv,
                      "dfkv_rdma_server_mw_fallback_connections_total"), 0);
}

// A connection that ends holding a windowed grant (the client went away; the
// grant timeout closes it): its endpoint, QP and windows go first, and only
// then does the range enter quarantine, without an invalidation.
TEST(RdmaServerMw, AClosedConnectionsWindowedRangeIsQuarantinedAfterItsQp) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv mw("DFKV_RDMA_SERVER_MW", "1");
  ScopedEnv grant("DFKV_RDMA_GRANT_TIMEOUT_MS", "300");
  SmallRamTier ram_env;
  MwEvents events;
  BeforeStart seam([&](RdmaServer& server) {
    RdmaServerTestPeer::SetMwEvents(
        server, [&](const char* name, const void* at) { events.Note(name, at); });
    RdmaServerTestPeer::SetBeforeEndpointTeardown(
        &server, [&] { events.Note("teardown", nullptr); });
  });
  RdmaNode node("mw-close");
  if (MwMode(*node.rsrv) != 1) GTEST_SKIP() << kNoWindows;
  const BlockKey key = ToBlockKey(SelfHdr(), "abandoned");
  const std::string value = PatternValue(200000, 83);
  ASSERT_TRUE(KvNodeServerTestPeer::Promote(*node.srv, key, value));
  PinnedPullPeer peer;
  ASSERT_TRUE(peer.Open(node));
  rdma::DynamicPullReady ready;
  ASSERT_TRUE(PrepareWindowed(node, peer, key, 0, value.size(), &ready));
  const void* at = reinterpret_cast<const void*>(ready.address);
  peer.Close();  // never released
  ASSERT_TRUE(events.WaitFor(at, "released", 5000));
  std::vector<std::string> order;
  {
    std::lock_guard<std::mutex> lock(events.mu);
    for (const auto& event : events.events)
      if (event.at == at || event.name == "teardown")
        order.push_back(event.name);
  }
  EXPECT_EQ(order, (std::vector<std::string>{"teardown", "quarantined",
                                             "released"}));
}

// Past the quarantine cap a grant takes a per-grant MR; without windows on a
// connection (ibv_alloc_mw failing) every grant does, and the connection is
// served.
TEST(RdmaServerMw, AFullQuarantineOrNoWindowsGivesPerGrantMrs) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv mw("DFKV_RDMA_SERVER_MW", "1");
  SmallRamTier ram_env;
  const BlockKey key = ToBlockKey(SelfHdr(), "fallback");
  const std::string value = PatternValue(200000, 84);
  {
    ScopedEnv cap("DFKV_RDMA_SERVER_MW_QUARANTINE_BYTES", "1");
    RdmaNode node("mw-cap");
    if (MwMode(*node.rsrv) != 1) GTEST_SKIP() << kNoWindows;
    ASSERT_TRUE(KvNodeServerTestPeer::Promote(*node.srv, key, value));
    PinnedPullPeer peer;
    ASSERT_TRUE(peer.Open(node));
    rdma::DynamicPullReady ready;
    std::string out;
    for (int i = 0; i < 200 && MwCounter(*node.rsrv,
             "dfkv_rdma_server_mw_quarantine_fallbacks_total") == 0; ++i) {
      ASSERT_TRUE(peer.Prepare(key, 0, value.size(), &ready));
      ASSERT_TRUE(peer.Read(ready, &out));
      ASSERT_EQ(out, value);
      ASSERT_TRUE(peer.Release(key, ready));
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_GT(MwCounter(*node.rsrv,
                        "dfkv_rdma_server_mw_quarantine_fallbacks_total"), 0);
    EXPECT_EQ(MwCounter(*node.rsrv, "dfkv_rdma_server_mw_binds_total"), 0);
  }
  {
    rdma::RcEndpoint::SetWindowAllocSeamForTest([] { return false; });
    struct Restore {
      ~Restore() { rdma::RcEndpoint::SetWindowAllocSeamForTest(nullptr); }
    } restore;
    RdmaNode node("mw-no-windows");
    if (MwMode(*node.rsrv) != 1) GTEST_SKIP() << kNoWindows;
    ASSERT_TRUE(KvNodeServerTestPeer::Promote(*node.srv, key, value));
    PinnedPullPeer peer;
    ASSERT_TRUE(peer.Open(node));
    rdma::DynamicPullReady ready;
    std::string out;
    for (int i = 0; i < 20; ++i) {
      ASSERT_TRUE(peer.Prepare(key, 0, value.size(), &ready));
      ASSERT_TRUE(peer.Read(ready, &out));
      ASSERT_EQ(out, value);
      ASSERT_TRUE(peer.Release(key, ready));
    }
    EXPECT_EQ(MwCounter(*node.rsrv,
                        "dfkv_rdma_server_mw_fallback_connections_total"), 1);
    EXPECT_EQ(MwCounter(*node.rsrv, "dfkv_rdma_server_mw_binds_total"), 0);
  }
}

// A region whose registration is under way, or failed, serves its grants
// with per-grant MRs; it is registered once, and a failed one is not tried
// again at once.
TEST(RdmaServerMw, ARegionInProgressOrFailedIsServedWithMrsAndRegisteredOnce) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv mw("DFKV_RDMA_SERVER_MW", "1");
  SmallRamTier ram_env;
  const BlockKey key = ToBlockKey(SelfHdr(), "in-progress");
  const std::string value = PatternValue(100000, 85);
  for (const bool fail : {false, true}) {
    SCOPED_TRACE(fail ? "failed" : "in progress");
    std::mutex mu;
    std::condition_variable cv;
    bool open = false;
    std::atomic<int> calls{0};
    BeforeStart seam([&](RdmaServer& server) {
      RdmaServerTestPeer::SetMwRegister(server, [&](size_t, char*) {
        calls.fetch_add(1);
        if (fail) return false;
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] { return open; });
        return true;
      });
    });
    RdmaNode node(fail ? "mw-reg-fail" : "mw-reg-hold");
    // Lets a held registration finish before the node stops, whatever ends
    // the test.
    struct Opener {
      std::mutex& mu;
      std::condition_variable& cv;
      bool& open;
      void Open() {
        {
          std::lock_guard<std::mutex> lock(mu);
          open = true;
        }
        cv.notify_all();
      }
      ~Opener() { Open(); }
    } opener{mu, cv, open};
    if (MwMode(*node.rsrv) != 1) GTEST_SKIP() << kNoWindows;
    ASSERT_TRUE(KvNodeServerTestPeer::Promote(*node.srv, key, value));
    PinnedPullPeer peer;
    ASSERT_TRUE(peer.Open(node));
    rdma::DynamicPullReady ready;
    std::string out;
    for (int i = 0; i < 20; ++i) {
      ASSERT_TRUE(peer.Prepare(key, 0, value.size(), &ready));
      ASSERT_TRUE(peer.Read(ready, &out));
      ASSERT_EQ(out, value);
      ASSERT_TRUE(peer.Release(key, ready));
    }
    EXPECT_EQ(calls.load(), 1);
    EXPECT_EQ(MwCounter(*node.rsrv, "dfkv_rdma_server_mw_binds_total"), 0);
    EXPECT_EQ(MwCounter(*node.rsrv, "dfkv_rdma_server_mw_fallback_grants_total"),
              20);
    EXPECT_EQ(MwCounter(*node.rsrv,
                        "dfkv_rdma_server_mw_region_register_failures_total"),
              fail ? 1 : 0);
    if (!fail) {
      opener.Open();
      ASSERT_TRUE(PrepareWindowed(node, peer, key, 0, value.size(), &ready));
      ASSERT_TRUE(peer.Read(ready, &out));
      EXPECT_EQ(out, value);
      ASSERT_TRUE(peer.Release(key, ready));
      EXPECT_EQ(calls.load(), 1);
    }
  }
}

// The client's next PREPARE on a slot can be polled before the completion
// of the RELEASE reply that retires the slot's window (separate packets).
// It waits for that completion instead of being refused as a busy slot.
void ExercisePrepareAheadOfReleaseCompletion(bool uring) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv mw("DFKV_RDMA_SERVER_MW", "1");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "2");  // a free send slot meanwhile
  SmallRamTier ram_env;
  {
    ScopedEnv uring_env("DFKV_SERVER_URING", uring ? "1" : "0");
    HeldReplyCqe reorder;
    reorder.captured = true;  // nothing held until armed
    RdmaNode node(uring ? "mw-early-uring" : "mw-early-sync", kMaxMsg, false,
                  /*prepare_reads=*/uring, {},
                  [&](ibv_wc* batch, int* count, size_t capacity) {
                    reorder.Filter(batch, count, capacity);
                  },
                  [&](size_t pending, size_t free, size_t head) {
                    reorder.Observe(pending, free, head);
                  });
    struct ReleaseOnExit {
      HeldReplyCqe& state;
      ~ReleaseOnExit() { state.Release(); }
    } unblock{reorder};
    if (MwMode(*node.rsrv) != 1) GTEST_SKIP() << kNoWindows;
    const BlockKey key = ToBlockKey(SelfHdr(), "early");
    const std::string value = PatternValue(100000, 86);
    ASSERT_TRUE(KvNodeServerTestPeer::Promote(*node.srv, key, value));
    PinnedPullPeer peer;
    ASSERT_TRUE(peer.Open(node, 2));
    rdma::DynamicPullReady ready;
    // Without pauses: the CQ seam makes the server poll every 1 ms, and a
    // connection with nothing in flight would be taken for idle.
    bool windowed = false;
    for (int i = 0; i < 5000 && !windowed; ++i) {
      const long binds =
          MwCounter(*node.rsrv, "dfkv_rdma_server_mw_binds_total");
      ASSERT_TRUE(peer.Prepare(key, 0, value.size(), &ready));
      windowed =
          MwCounter(*node.rsrv, "dfkv_rdma_server_mw_binds_total") > binds;
      if (!windowed) ASSERT_TRUE(peer.Release(key, ready));
    }
    ASSERT_TRUE(windowed);
    std::string out;
    ASSERT_TRUE(peer.Read(ready, &out));
    {
      // Every earlier SEND completion polled (both send slots free): hold
      // the RELEASE reply's, and let it go behind the next request's
      // receive, in the same batch.
      std::unique_lock<std::mutex> lock(reorder.mu);
      ASSERT_TRUE(reorder.cv.wait_for(lock, std::chrono::seconds(5),
                                      [&] { return reorder.free_slots == 2; }));
      reorder.captured = false;
      reorder.release_on_next_recv = true;
    }
    ASSERT_TRUE(peer.Release(key, ready));
    rdma::DynamicPullReady again;
    EXPECT_TRUE(peer.Prepare(key, 0, value.size(), &again))
        << "the PREPARE was refused while the slot was revoking";
    {
      // The reply's completion was let go only with or after the next
      // request's receive.
      std::lock_guard<std::mutex> lock(reorder.mu);
      EXPECT_TRUE(reorder.held.empty());
    }
    ASSERT_TRUE(peer.Read(again, &out));
    EXPECT_EQ(out, value);
    ASSERT_TRUE(peer.Release(key, again));
  }
}
TEST(RdmaServerMw, APrepareAheadOfTheReleaseCompletionWaitsForItSync) {
  ExercisePrepareAheadOfReleaseCompletion(false);
}
TEST(RdmaServerMw, APrepareAheadOfTheReleaseCompletionWaitsForItUring) {
  ExercisePrepareAheadOfReleaseCompletion(true);
}

// Every admitted window uses the configured maximum hold. A rail accounting
// snapshot chooses MW rather than an exact MR at admission; later rail
// pressure must not alter the lifetime of this already authorized one.
TEST(RdmaServerMw, ArenaPinBudgetSurvivesEarlyStagingRelease) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv mw("DFKV_RDMA_SERVER_MW", "1");
  ScopedEnv cap("DFKV_RDMA_SERVER_MW_QUARANTINE_BYTES", "8388608");
  // This test concerns the allocation retained by a RAM arena. Do not mistake
  // its absence for the provider lacking type-2B windows and silently skip it.
  SmallRamTier ram_env;
  MwEvents events;
  BeforeStart seam([&](RdmaServer& server) {
    RdmaServerTestPeer::SetMwEvents(
        server, [&](const char* name, const void* at) { events.Note(name, at); });
  });
  // Install the observation hook before RdmaNode::Start: the grant's release
  // may complete during Start, before Promote or PinnedPullPeer::Open.
  RdmaNode node("mw-pin-budget", 1024 * 1024);
  if (MwMode(*node.rsrv) != 1) GTEST_SKIP() << kNoWindows;
  const BlockKey key = ToBlockKey(SelfHdr(), "pin-budget-tiny");
  const std::string value(1, 'v');
  ASSERT_TRUE(KvNodeServerTestPeer::Promote(*node.srv, key, value));
  const auto before = MwCounter(*node.rsrv, "dfkv_rdma_server_mw_quarantine_bytes");
  PinnedPullPeer peer;
  ASSERT_TRUE(peer.Open(node, 1, false, value.size()));
  rdma::DynamicPullReady ready;
  // A one-byte RAM value occupies a full allocation, even though the staging
  // buffer is returned as soon as the direct-to-arena read is authorized.
  // Revocation must transfer the same reserved amount, not this range length.
  ASSERT_TRUE(PrepareWindowed(node, peer, key, 0, value.size(), &ready));
  ASSERT_TRUE(peer.Release(key, ready));
  const void* at = reinterpret_cast<const void*>(ready.address);
  const bool released = events.WaitFor(at, "released", 2000);
  if (!released) {
    ADD_FAILURE() << "no release event, MW metrics:\n" << node.rsrv->MetricsText();
    for (const auto& event : events.Snapshot())
      ADD_FAILURE() << "event=" << event.name << " at=" << event.at
                    << " (range=" << at << ")";
    for (const auto& event : events.For(at))
      ADD_FAILURE() << "range-event=" << event.name;
  }
  ASSERT_TRUE(released);
  EXPECT_EQ(RdmaServerTestPeer::MwReservedBytes(*node.rsrv), 0u);
  const long after = MwCounter(*node.rsrv, "dfkv_rdma_server_mw_quarantine_bytes");
  EXPECT_EQ(after, before) << "before=" << before << " after=" << after;
}


TEST(RdmaServerMw, LaterRailPressureDoesNotShortenAnAdmittedQuarantine) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv mw("DFKV_RDMA_SERVER_MW", "1");
  ScopedEnv max_ms("DFKV_RDMA_SERVER_MW_QUARANTINE_MAX_MS", "40");
  ScopedEnv link("DFKV_RDMA_SERVER_MW_LINK_GBPS", "1");
  ScopedEnv cap("DFKV_RDMA_SERVER_MW_QUARANTINE_BYTES", "33554432");
  ScopedEnv ram_on("DFKV_RAM_TIER", "1");
  ScopedEnv ram_bytes("DFKV_RAM_TIER_BYTES", "16777216");
  ScopedEnv extent("DFKV_RAM_TIER_EXTENT_BYTES", "4194304");
  ScopedEnv reserve("DFKV_RAM_TIER_LARGE_RESERVE_BYTES", "0");
  ScopedEnv shards("DFKV_RAM_TIER_SHARDS", "1");
  MwEvents events;
  BeforeStart seam([&](RdmaServer& server) {
    RdmaServerTestPeer::SetMwEvents(
        server, [&](const char* name, const void* at) { events.Note(name, at); });
  });
  RdmaNode node("mw-frozen-rail", 1024 * 1024);
  if (MwMode(*node.rsrv) != 1) GTEST_SKIP() << kNoWindows;
  const BlockKey small = ToBlockKey(SelfHdr(), "frozen-small");
  const BlockKey big = ToBlockKey(SelfHdr(), "frozen-big");
  const std::string small_value = PatternValue(1024 * 1024, 87);
  const std::string big_value = PatternValue(1024 * 1024, 88);
  ASSERT_TRUE(KvNodeServerTestPeer::Promote(*node.srv, small, small_value));
  ASSERT_TRUE(KvNodeServerTestPeer::Promote(*node.srv, big, big_value));
  PinnedPullPeer peer;
  ASSERT_TRUE(peer.Open(node, 1, false, 1024 * 1024));
  rdma::DynamicPullReady ready;
  ASSERT_TRUE(PrepareWindowed(node, peer, small, 0, small_value.size(), &ready));
  // Additional grants below pressure fallback accounting, not this grant's hold.
  // The quarantine itself is the 40 ms maximum. Four more 1 MiB grants take
  // the live rail backlog to 5 MiB, about 44.9 ms, and later grants over that
  // threshold use exact MRs. Neither changes the first grant's retained range.
  // Cap reservations (32 MiB) do not cause a fallback here.
  std::vector<std::unique_ptr<PinnedPullPeer>> holders;
  std::vector<rdma::DynamicPullReady> held(4);
  const long fallbacks_before =
      MwCounter(*node.rsrv, "dfkv_rdma_server_mw_quarantine_fallbacks_total");
  for (int i = 0; i < 4; ++i) {
    holders.push_back(std::make_unique<PinnedPullPeer>());
    ASSERT_TRUE(holders.back()->Open(node, 1, false, 1024 * 1024));
    ASSERT_TRUE(holders.back()->Prepare(big, 0, big_value.size(), &held[i]));
  }
  EXPECT_GT(MwCounter(*node.rsrv,
                      "dfkv_rdma_server_mw_quarantine_fallbacks_total"),
            fallbacks_before);
  const void* at = reinterpret_cast<const void*>(ready.address);
  ASSERT_TRUE(peer.Release(small, ready));
  ASSERT_TRUE(events.WaitFor(at, "released", 2000));
  const auto life = events.For(at);
  ASSERT_EQ(life.size(), 3u);
  const auto held_us = std::chrono::duration_cast<std::chrono::microseconds>(
      life[2].when - life[1].when).count();
  EXPECT_GE(held_us, 40000);
  for (int i = 0; i < 4; ++i) ASSERT_TRUE(holders[i]->Release(big, held[i]));
}

// A window whose deallocation is refused at close (EBUSY on a live QP)
// does not stop the server: the QP goes to the error state, the
// deallocation is tried again before the QP is destroyed, and the range is
// still quarantined after it, without waiting on the bound window (on
// Soft-RoCE, destroying the QP first held the close for about 50 s).
TEST(RdmaServerMw, ARefusedWindowDeallocationIsRetriedBeforeTheQpGoes) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv mw("DFKV_RDMA_SERVER_MW", "1");
  ScopedEnv grant("DFKV_RDMA_GRANT_TIMEOUT_MS", "300");
  SmallRamTier ram_env;
  // Reject the initial close deallocation and its post-ERR retry for the
  // bound window, then use the real verb. A permanent refusal would destroy
  // the QP while the provider still considers that window bound: on rxe that
  // blocks this bounded test for about 50 seconds. It is not a valid recovery
  // expectation: production's fail-closed rule cannot discard ownership just
  // to satisfy an always-failing injection.
  std::mutex refused_mu;
  std::unordered_map<ibv_mw*, int> refusals;
  ibv_mw* last_bind_mw = nullptr;
  ibv_mw* bound_mw = nullptr;
  rdma::RcEndpoint::SetWindowBindObserverForTest([&](ibv_mw* mw) {
    std::lock_guard<std::mutex> lk(refused_mu);
    last_bind_mw = mw;
  });
  rdma::RcEndpoint::SetWindowDeallocSeamForTest([&](ibv_mw* mw) {
    // The bind observer, not an rkey heuristic, identifies this exact bound
    // grant. Unbound/preallocated MWs must release normally and never consume
    // this grant's transient failures.
    std::lock_guard<std::mutex> lk(refused_mu);
    if (mw != bound_mw) return ibv_dealloc_mw(mw);
    if (refusals[mw]++ < 2) return EBUSY;
    return ibv_dealloc_mw(mw);
  });
  struct Restore {
    ~Restore() {
      rdma::RcEndpoint::SetWindowDeallocSeamForTest(nullptr);
      rdma::RcEndpoint::SetWindowBindObserverForTest(nullptr);
    }
  } restore;
  MwEvents events;
  BeforeStart seam([&](RdmaServer& server) {
    RdmaServerTestPeer::SetMwEvents(
        server, [&](const char* name, const void* at) { events.Note(name, at); });
    RdmaServerTestPeer::SetBeforeEndpointTeardown(
        &server, [&] { events.Note("teardown", nullptr); });
  });
  RdmaNode node("mw-dealloc-retry");
  if (MwMode(*node.rsrv) != 1) GTEST_SKIP() << kNoWindows;
  const BlockKey key = ToBlockKey(SelfHdr(), "dealloc");
  const std::string value = PatternValue(100000, 89);
  ASSERT_TRUE(KvNodeServerTestPeer::Promote(*node.srv, key, value));
  {
    PinnedPullPeer peer;
    ASSERT_TRUE(peer.Open(node));
    rdma::DynamicPullReady ready;
    ASSERT_TRUE(PrepareWindowed(node, peer, key, 0, value.size(), &ready));
    {
      // PrepareWindowed returns only after an actual windowed grant is admitted.
      // Freeze that exact final bind: any earlier setup/candidate bind or other
      // connection must not be mistaken for the live grant being torn down.
      std::lock_guard<std::mutex> lk(refused_mu);
      bound_mw = last_bind_mw;
      rdma::RcEndpoint::SetWindowBindObserverForTest(nullptr);
      ASSERT_NE(bound_mw, nullptr);
    }
    const void* at = reinterpret_cast<const void*>(ready.address);
    const auto closed = std::chrono::steady_clock::now();
    peer.Close();  // never released: the grant timeout closes it
    ASSERT_TRUE(events.WaitFor(nullptr, "teardown", 2000));
    ASSERT_TRUE(events.WaitFor(at, "quarantined", 2000));
    ASSERT_TRUE(events.WaitFor(at, "released", 3000));
    EXPECT_LT(std::chrono::steady_clock::now() - closed,
              std::chrono::seconds(2))
        << "the close waited on the bound window";
    const auto life = events.For(at);
    ASSERT_EQ(life.size(), 2u);
    EXPECT_EQ(life[0].name, "quarantined");
    EXPECT_EQ(life[1].name, "released");
    std::lock_guard<std::mutex> lk(refused_mu);
    // One windowed grant is held by this connection. Two injected refusals
    // precede its real deallocation after QP-to-ERR transition. This counts
    // all attempts by the selected bound MW, not just refused attempts: two
    // EBUSY returns followed by the real verbs invocation.
    ASSERT_EQ(refusals.size(), 1u);
    EXPECT_EQ(refusals.begin()->second, 3);
  }
  // The server goes on serving.
  PinnedPullPeer again;
  ASSERT_TRUE(again.Open(node));
  rdma::DynamicPullReady ready;
  std::string out;
  ASSERT_TRUE(again.Prepare(key, 0, value.size(), &ready));
  ASSERT_TRUE(again.Read(ready, &out));
  EXPECT_EQ(out, value);
  ASSERT_TRUE(again.Release(key, ready));
}

// The window primitives on their own: a READ through a bound window reads
// the range; after its invalidation the same rkey is refused.
TEST(RdmaServerMw, AnInvalidatedWindowsRkeyIsRefused) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const char* devs = std::getenv("DFKV_RDMA_DEV");
  const std::string first =
      devs ? std::string(devs).substr(0, std::string(devs).find(',')) : "";
  const char* dev = first.empty() ? nullptr : first.c_str();
  rdma::RcEndpoint target;
  rdma::RcEndpoint reader;
  target.set_window_slots(1);
  ASSERT_TRUE(target.Open(dev, 4096, 1));
  if (!target.ProbeType2Windows()) GTEST_SKIP() << kNoWindows;
  ASSERT_TRUE(reader.Open(dev, 4096, 1));
  ASSERT_TRUE(target.Connect(reader.Local()));
  ASSERT_TRUE(reader.Connect(target.Local()));
  ASSERT_TRUE(target.AllocWindows(1));
  std::vector<char> region(1 << 20);
  for (size_t i = 0; i < region.size(); ++i)
    region[i] = static_cast<char>((i * 7 + 3) & 0xff);
  ibv_mr* bind_mr = target.RegisterBindRegion(region.data(), region.size());
  ASSERT_NE(bind_mr, nullptr);
  constexpr size_t kOffset = 4096 + 17, kLen = 65536;
  uint32_t rkey = 0;
  ASSERT_TRUE(target.PostBindWindow(0, bind_mr, region.data() + kOffset, kLen,
                                    1, &rkey, /*signaled=*/true));
  ibv_wc wc{};
  ASSERT_EQ(target.WaitComp(&wc, 1, 10000), 1);
  ASSERT_EQ(wc.status, IBV_WC_SUCCESS);
  std::vector<char> out(kLen);
  ibv_mr* out_mr = reader.RegisterTransient(out.data(), out.size(), true);
  ASSERT_NE(out_mr, nullptr);
  ASSERT_TRUE(reader.PostRead(0, out.data(), kLen, out_mr,
                              reinterpret_cast<uint64_t>(region.data() + kOffset),
                              rkey));
  ASSERT_EQ(reader.WaitComp(&wc, 1, 10000), 1);
  ASSERT_EQ(wc.status, IBV_WC_SUCCESS);
  EXPECT_EQ(std::memcmp(out.data(), region.data() + kOffset, kLen), 0);
  ASSERT_TRUE(target.PostLocalInvalidate(rkey, 2, /*signaled=*/true));
  ASSERT_EQ(target.WaitComp(&wc, 1, 10000), 1);
  ASSERT_EQ(wc.status, IBV_WC_SUCCESS);
  // The same window bound again, elsewhere, under a new key, as the next
  // grant on its slot does.
  const uint32_t old_rkey = rkey;
  ASSERT_TRUE(target.PostBindWindow(0, bind_mr, region.data() + 2 * kOffset,
                                    kLen, 3, &rkey, /*signaled=*/true));
  ASSERT_EQ(target.WaitComp(&wc, 1, 10000), 1);
  ASSERT_EQ(wc.status, IBV_WC_SUCCESS) << "a second bind of the window";
  ASSERT_NE(rkey, old_rkey);
  ASSERT_TRUE(reader.PostRead(
      0, out.data(), kLen, out_mr,
      reinterpret_cast<uint64_t>(region.data() + 2 * kOffset), rkey));
  ASSERT_EQ(reader.WaitComp(&wc, 1, 10000), 1);
  ASSERT_EQ(wc.status, IBV_WC_SUCCESS);
  EXPECT_EQ(std::memcmp(out.data(), region.data() + 2 * kOffset, kLen), 0);
  ASSERT_TRUE(target.PostLocalInvalidate(rkey, 4, /*signaled=*/true));
  ASSERT_EQ(target.WaitComp(&wc, 1, 10000), 1);
  ASSERT_EQ(wc.status, IBV_WC_SUCCESS);
  rkey = old_rkey;
  ASSERT_TRUE(reader.PostRead(0, out.data(), kLen, out_mr,
                              reinterpret_cast<uint64_t>(region.data() + kOffset),
                              rkey));
  ASSERT_EQ(reader.WaitComp(&wc, 1, 10000), 1);
  EXPECT_EQ(wc.status, IBV_WC_REM_ACCESS_ERR);
  reader.ReleaseTransient(out_mr);
  // Nothing is bound any more: the region deregisters.
  EXPECT_EQ(rdma::RcEndpoint::DeregisterBindRegion(bind_mr), 0);
}
