/* RDMA cache-node listener — native v2 libibverbs RC. Bounded 32,786-byte
 * per-QP control buffers share a process-wide registered payload segment.
 * Peers that cannot negotiate v2 are rejected.
 * The TCP listener bootstraps QPs and adapts v2.28 no-GET-WRITE proofs.
 * Startup discovers the first ACTIVE HCA automatically, or resolves every
 * explicitly configured HCA into a fixed topology (including initially inactive
 * ports), then anchors shared PD/MRs.
 * shutdown(listen_fd) keeps Stop() interruptible. */
#ifndef DFKV_RDMA_SERVER_H_
#define DFKV_RDMA_SERVER_H_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <unordered_set>
#include <vector>

#ifdef DFKV_WITH_URING
#include "cache/uring_reader.h"
#endif
#include "cache/ram_tier.h"
#include "cache/rail_peer_window.h"
#include "cache/store_engine.h"
#include "common/kv_types.h"
#include "common/status.h"
#include "transport/rdma_mw.h"
#include "transport/rdma_recv_segment.h"
#include "transport/rdma_verbs.h"  // rdma::RcEndpoint
#include "transport/rdma_topology.h"
#include "transport/rdma_wc_status.h"
#include "utils/accept_guard.h"

namespace dfkv {
class RdmaServerTestPeer;

class RdmaServer {
 public:
  using Handler = std::function<Status(
      uint8_t op, const BlockKey& key, uint64_t offset, uint64_t length,
      const char* payload, uint64_t payload_len, std::string* out_data,
      size_t* value_len)>;
  // Optional direct GET handler: reads raw bytes into a registered buffer and
  // reports both returned bytes and the authoritative full stored value size.
  using RangeHandler = std::function<Status(
      const BlockKey& key, uint64_t offset, uint64_t length, char* io_buf,
      size_t io_cap, const char** out_data, size_t* out_len,
      size_t* value_len)>;
  // Optional direct PUT handler: `data` points at an aligned registered buffer
  // containing exactly the raw stored value bytes. `batch_arrival` is when the
  // request batch of this PUT arrived: the requests a connection receives
  // while it still owes a reply. One Serve thread handles them one after
  // another, and the client gives the whole batch one deadline, so a PUT that
  // has to wait counts its time from then.
  using CacheDirectHandler = std::function<Status(
      const BlockKey& key, char* data, size_t len, size_t cap,
      std::chrono::steady_clock::time_point batch_arrival)>;
  // Batched direct PUT: multiple keys in one call for io_uring flush.
  using CacheDirectBatchHandler = std::function<std::vector<Status>(
      const std::vector<StoreEngine::CacheBatchItem>& items)>;

  // One read-preparation entry point for disk, coalescer, and RAM sources. The
  // returned move-only transaction is the sole cleanup authority across
  // queueing, fallback, completion, timeout, and connection teardown.
  using PrepareReadHandler = std::function<PreparedRead(
      const BlockKey& key, uint64_t offset, uint64_t length, char* staging,
      size_t staging_cap)>;

  // dev_name empty => env DFKV_RDMA_DEV; both empty => first ACTIVE local HCA.
  // An explicit comma list enables multi-rail.
  explicit RdmaServer(Handler handler, size_t max_msg = (64u << 20),
                      const std::string& dev_name = "");
  void set_range_handler(RangeHandler h) { range_handler_ = std::move(h); }
  void set_cache_direct_handler(CacheDirectHandler h) {
    cache_direct_handler_ = std::move(h);
  }
  void set_cache_direct_batch_handler(CacheDirectBatchHandler h) {
    cache_direct_batch_handler_ = std::move(h);
  }
  void set_prepare_read_handler(PrepareReadHandler h) {
    prepare_read_handler_ = std::move(h);
  }
  // Pinned arena hit lookup (B5-3 zero-copy pull GET): returns true with a
  // send-pinned arena address the dynamic-pull READY can name directly, or
  // false when the read must use the staging-copy pull path. Optional; when
  // unset or returning false the pull path stages through the receive pool.
  using PinnedRamHandler = std::function<bool(
      const BlockKey& key, uint64_t offset, uint64_t length,
      PreparedRead* out)>;
  void set_pinned_ram_handler(PinnedRamHandler h) {
    pinned_ram_handler_ = std::move(h);
  }
  // Optional existence check of a GET before it is given a staging lease:
  // kNotFound is answered at once, with no lease and no wait for receive-pool
  // memory. Unset: every GET is staged and the read answers the miss.
  using LookupHandler = std::function<Status(const BlockKey& key)>;
  void set_lookup_handler(LookupHandler h) { lookup_handler_ = std::move(h); }
  // The RAM tier whose arena GET grants may bind memory windows on
  // (DFKV_RDMA_SERVER_MW); it must outlive Stop(). Set before Start().
  void set_mw_arena(RamTier* tier) { mw_arena_tier_ = tier; }
  ~RdmaServer();

  Status Start(int port);  // TCP bootstrap port
  // Called from the port watcher, off any serve thread, when a rail's port
  // turns ACTIVE or stops being ACTIVE; at most once a second per rail (a
  // flapping port cannot storm the caller, which asks the MDS for an early
  // heartbeat). Set it before Start, or at any time: it is copied under its
  // own lock.
  using RailStateListener =
      std::function<void(const std::string& device, bool active)>;
  void set_rail_state_listener(RailStateListener listener) {
    std::lock_guard<std::mutex> lock(rail_listener_mu_);
    rail_listener_ = std::move(listener);
  }
  void Stop();
  // First half of Stop(): stops accepting and joins every connection thread,
  // so no handler runs once it returns. The rails and their receive-pool
  // registrations stay until Stop(); a server that is shutting down gets its
  // disks clean in between.
  void Quiesce();
  int port() const { return port_; }
  // Explicit-mode failures retain the complete requested order; diagnostic
  // probe evidence never replaces this with a partially discovered topology.
  const std::vector<std::string>& DeviceNames() const { return anchor_devs_; }
  // Fixed topology and successfully materialized anchors. Explicit startup is
  // fail-closed, so a running server has one initialized anchor per configured
  // rail; keeping both counts exposes startup/lifecycle truth independently.
  size_t ConfiguredRailCount() const { return anchor_devs_.size(); }
  size_t InitializedRailCount() const { return anchors_.size(); }

  // Observability (used by tests): number of Serve threads not yet reaped.
  // Bounded under churn now that finished connections are joined as new ones
  // arrive — previously this grew without bound until Stop().
  size_t live_conn_count();

  // RDMA completion counters (relaxed atomics, incremented in the serve loop).
  uint64_t Completions() const { return completions_.load(std::memory_order_relaxed); }
  uint64_t CompletionErrors() const { return completion_errors_.load(std::memory_order_relaxed); }
  uint64_t ActiveConns() const { return active_conns_.load(std::memory_order_relaxed); }
  uint64_t IdleReclaims() const { return idle_reclaims_.load(std::memory_order_relaxed); }
  // io_uring pipeline observability. A submit batch is one SQ flush group;
  // completions count logical descriptors (short-read residuals stay one).
  // All remain zero when the path is off.
  uint64_t UringReads() const {
    return uring_reads_.load(std::memory_order_relaxed);
  }
  uint64_t UringReadBatches() const {
    return uring_read_batches_.load(std::memory_order_relaxed);
  }
  uint64_t UringReadBatchMax() const {
    return uring_read_batch_max_.load(std::memory_order_relaxed);
  }
  uint64_t UringCompletions() const {
    return uring_completions_.load(std::memory_order_relaxed);
  }
  uint64_t UringInflight() const {
    return uring_inflight_.load(std::memory_order_relaxed);
  }
  uint64_t UringInflightMax() const {
    return uring_inflight_max_.load(std::memory_order_relaxed);
  }
  uint64_t UringSendFences() const {
    return uring_send_fences_.load(std::memory_order_relaxed);
  }
  uint64_t UringRepliesPosted() const {
    return uring_replies_posted_.load(std::memory_order_relaxed);
  }
  uint64_t UringSendPostErrors() const {
    return uring_send_post_errors_.load(std::memory_order_relaxed);
  }
  uint64_t UringInitFallbacks() const {
    return uring_init_fallbacks_.load(std::memory_order_relaxed);
  }
  uint64_t V2Conns() const { return v2_conns_.load(std::memory_order_relaxed); }
  // Bootstrap listener: connections that have not delivered their bootstrap
  // frames yet, connections refused at DFKV_RDMA_MAX_PENDING_HANDSHAKES,
  // accept() failures that made the listener back off, and connections
  // refused because their thread could not be created.
  uint64_t PendingHandshakes() const {
    return pending_handshakes_.load(std::memory_order_relaxed);
  }
  uint64_t RejectedConnections() const {
    return rejected_connections_.load(std::memory_order_relaxed);
  }
  uint64_t AcceptFailures() const { return accept_guard_.accept_failures(); }
  uint64_t ThreadStartFailures() const {
    return accept_guard_.thread_start_failures();
  }
  uint64_t V2PutWrites() const {
    return v2_put_writes_.load(std::memory_order_relaxed);
  }
  // The server-side pipeline depth (env DFKV_RDMA_DEPTH, default 4) -- surfaced
  // in ring INFO because the CLIENT's depth must not exceed it: excess in-flight
  // requests hit receiver-not-ready retries and degrade SILENTLY (measured 3-4x
  // on pipelined GETs) and cause PUT batch failures during burst writes.
  size_t PipelineDepth() const;
  std::string MetricsText() const;  // Prometheus text (dfkv_rdma_*)
  // Whether the io_uring async-GET serve path should be used for new conns.
  // Default ON when built with DFKV_WITH_URING and the prepared-read handler is
  // set; DFKV_SERVER_URING=0 forces sync. Decided once per connection.
  bool UseUringPath() const;

 private:
  void AcceptLoop();
  // `bootstrap_deadline` bounds the whole read of the peer's bootstrap frames.
  // `pending`, when set, was incremented by the caller for this connection and
  // is decremented as soon as those frames are in (or the connection is dropped).
  void Serve(int boot_fd,
             std::chrono::steady_clock::time_point bootstrap_deadline,
             std::atomic<uint64_t>* pending = nullptr);
  void ReapDoneLocked();  // join+erase finished Serve threads; conn_mu_ held
  // The idle connections whose leases to give up so that `bytes` fits for
  // a request on `rail` (rdma::PlanPressureVictims); empty if no set of them
  // would make room. conn_mu_ held.
  std::vector<rdma::RcEndpoint*> PlanPressureVictimsLocked(size_t bytes,
                                                           int rail);
  // Server memory windows (DFKV_RDMA_SERVER_MW; design in
  // docs/design/server-memory-windows.md). A revoked GET range: what held
  // it (a staging lease, an arena pin) until its quarantine ends.
  struct MwQuarantined {
    rdma::RecvSegmentPool::Lease lease;
    PreparedRead read;
    size_t data_len = 0;
    double read_elapsed_sec = 0.0;
    uint64_t bytes = 0;  // counted against the quarantine cap
    // GET-grant bytes outstanding on the rail as the window went: what the
    // port could still have to send ahead of a READ in flight.
    uint64_t rail_bytes = 0;
    const char* at = nullptr;  // the range (test hook)
  };
  // The bind-only region [at, at+len) lies in on `rail` (an arena extent,
  // or the chunk of `lease`), if registered; nullptr otherwise, which
  // starts its registration.
  ibv_mr* MwRegionFor(size_t rail, const char* at, size_t len,
                      const rdma::RecvSegmentPool::Lease& lease);
  // A grant holding held_bytes, with rail_bytes of GET grants outstanding
  // on its rail (its own included), may use a window: its quarantine fits
  // the maximum and the quarantined bytes the cap.
  bool MwQuarantineAllows(uint64_t rail_bytes, uint64_t held_bytes) const;
  // The quarantine of a range revoked with rail_bytes of GET grants
  // outstanding on its rail: twice the time the port needs to send them,
  // plus the margin, never below the floor.
  uint64_t MwQuarantineUsFor(uint64_t rail_bytes) const;
  // Holds the range until now + its quarantine; the recv_trim thread
  // releases it.
  void MwQuarantine(MwQuarantined entry);
  // Releases the entries whose quarantine has ended, or all of them.
  void MwReleaseQuarantine(bool all);
  // The earliest and the latest quarantine end; 0 if none.
  uint64_t MwQuarantineEndUs(bool latest);
  rdma::RecvSegmentPool::Lease AllocateReceiveWithPressure(
      size_t bytes, int rail, int numa_node, uint64_t wait_us,
      rdma::RecvSegmentPool::LeaseClass lease_class =
          rdma::RecvSegmentPool::LeaseClass::kConnection);

  // A live connection: its Serve thread plus a flag the thread sets (last thing
  // it does) so AcceptLoop can tell it has finished and join it without blocking.
  // The flag is a shared_ptr because conns_ reallocates on push_back — the thread
  // captures its own copy, so the atomic outlives any vector move/erase.
  struct Conn {
    std::thread th;
    std::shared_ptr<std::atomic<bool>> done;
  };

  // Server-only v2.28 transition identities. A live token proves an accepted
  // dynamic connection, not a writer. No GET WRITE can be submitted anywhere
  // in this server; close erases the token under the same lock as proof lookup.
  uint64_t RegisterLegacyConnection();
  void ForgetLegacyConnection(uint64_t token);

  // Per-device counters make a partial rail failure or affinity imbalance
  // visible without polling host counters. The rail set is fixed by Start().
  struct RailStats {
    std::atomic<uint64_t> active_conns{0};
    std::atomic<uint64_t> completions{0};
    std::atomic<uint64_t> completion_errors{0};
    std::atomic<uint64_t> put_writes{0};
    std::atomic<uint64_t> put_bytes{0};
    // completion_errors broken down by work-completion status. A connection
    // stops at its first failed completion, so each one is counted once.
    // Kept last: every Serve thread bumps the counters above per request, and
    // this array, written only when a connection dies, must not sit between
    // them and spread them over more cache lines.
    std::atomic<uint64_t> wc_errors[rdma::kWcStatusSlots] = {};
  };

  Handler handler_;
  RangeHandler range_handler_;
  CacheDirectHandler cache_direct_handler_;
  CacheDirectBatchHandler cache_direct_batch_handler_;
  PrepareReadHandler prepare_read_handler_;
  PinnedRamHandler pinned_ram_handler_;
  LookupHandler lookup_handler_;
  size_t max_msg_;
  std::string dev_name_;
  bool auto_device_ = true;
  int listen_fd_ = -1;
  int port_ = 0;
  std::atomic<bool> running_{false};
  std::thread accept_thread_;
  net::AcceptGuard accept_guard_{"rdma bootstrap listener"};
  // A peer must deliver its bootstrap frames within this budget, counted from
  // accept(). A client connects, creates its queue pair and only then writes
  // them, so the budget covers the client's queue pair creation too.
  static constexpr int kBootstrapReadMs = 10000;
  int bootstrap_read_ms_ = kBootstrapReadMs;  // tests shorten it
  // Connections accepted whose bootstrap frames have not arrived: the only
  // phase in which a peer that sends nothing holds a thread. A real client
  // is in it while it creates its queue pair, so a mass reconnect holds about
  // (accept rate x that time) slots; the limit has to stay well above that
  // and only needs to bound silent peers. 0 = unlimited.
  static constexpr uint64_t kDefaultMaxPendingHandshakes = 4096;
  uint64_t max_pending_handshakes_ = kDefaultMaxPendingHandshakes;
  std::atomic<uint64_t> pending_handshakes_{0};
  std::atomic<uint64_t> rejected_connections_{0};
  std::thread recv_trim_thread_;
  // Session channels (DFKV_RDMA_SESSION_FD, off by default): the bootstrap
  // socket of a connection whose client asked for it stays open. The
  // watcher thread wakes the connection's Serve thread when the client's end
  // closes, so it ends at once instead of at the idle timeout; the server's
  // own end is closed when the connection ends, for whatever reason, which
  // tells the client before it reuses the connection.
  bool session_fd_enabled_ = false;
  int session_epoll_fd_ = -1;
  int session_stop_fd_ = -1;  // eventfd that ends the watcher
  std::thread session_thread_;
  std::mutex session_mu_;
  uint64_t next_session_id_ = 1;  // under session_mu_
  std::unordered_map<uint64_t, rdma::RcEndpoint*> sessions_;  // under session_mu_
  std::atomic<uint64_t> session_peer_closed_{0};
  std::atomic<uint64_t> sessions_kept_{0};
  void SessionWatchLoop();
  uint64_t WatchSession(int fd, rdma::RcEndpoint* ep);
  void UnwatchSession(uint64_t id, int fd);
  std::mutex recv_trim_mu_;
  std::condition_variable recv_trim_cv_;
  // Port state of each rail (RDS-02), polled every port_check_ms_ by
  // port_thread_ (DFKV_RDMA_PORT_CHECK_MS; 0 = off): 1 ACTIVE, 0 not,
  // -1 unknown (treated as usable). The poll holds no lock a handshake or a
  // serve loop takes; a rail going down takes conn_mu_ only to wake that
  // rail's connections.
  int port_check_ms_ = 0;  // off unless set (new behaviour defaults off)
  std::unique_ptr<rdma::PortWatcher> port_watcher_;
  std::unique_ptr<std::atomic<int>[]> rail_port_state_;
  std::vector<uint64_t> rail_last_notice_ms_;  // port_thread_ only
  std::vector<int> rail_notified_state_;       // port_thread_ only
  std::vector<int> rail_down_polls_;           // port_thread_ only
  // Down on two polls in a row: the rail's connections are ended.
  std::unique_ptr<std::atomic<bool>[]> rail_down_confirmed_;
  // Runs before a connection registers in live_eps_ (after its rail check).
  std::function<void()> before_live_registration_for_test_;
  std::thread port_thread_;
  std::mutex port_mu_;
  std::condition_variable port_cv_;
  std::mutex rail_listener_mu_;
  RailStateListener rail_listener_;
  std::atomic<uint64_t> inactive_rail_refusals_{0};
  std::atomic<uint64_t> rail_down_connections_ended_{0};
  std::atomic<uint64_t> rail_port_changes_{0};
  void PortWatchLoop();
  void CheckRailPorts();
  // The rail a client that names no device gets: the first not known down.
  size_t DefaultRail() const;
  // Track per-connection Serve threads + their endpoints so Stop() can wake them
  // out of WaitComp and join them before the handler's owner is destroyed, and
  // so finished threads are reaped incrementally (see ReapDoneLocked).
  std::mutex conn_mu_;
  std::vector<Conn> conns_;
  // The Serve thread owns the pointed-to flag until it erases its endpoint
  // under conn_mu_. Pressure reclaim considers only quiescent QPs, preferring
  // leases large enough to satisfy the incoming connection. CQ age alone
  // cannot distinguish idle from an in-flight disk/READ operation.
  struct LiveEndpoint {
    std::atomic<bool>* reclaimable;
    size_t recv_lease_bytes;
    // Where its receive lease lies, for pressure reclaim: the range evicting
    // it would free (PlanPressureVictimsLocked).
    const rdma::RecvSegment* lease_segment;
    size_t lease_offset;
    // Set at its first completion. A connection that never got one is not
    // quiescent (its first request may be in flight), until kNeverServedUs.
    std::atomic<bool>* served_any;
    // The connection holds a grant its client has yet to use or release:
    // Quiesce lets it finish (DFKV_RDMA_STOP_DRAIN_MS) before waking it.
    std::atomic<bool>* holding_grants;
    size_t rail = 0;  // index into anchor_devs_
  };
  // Bootstrap sockets of connections still in their handshake: Quiesce shuts
  // them down so their threads stop waiting for the peer's frames. A thread
  // removes its socket under handshake_mu_ before closing it.
  std::mutex handshake_mu_;
  std::unordered_set<int> handshake_fds_;
  bool handshakes_closed_ = false;  // Quiesce has shut them down
  std::unordered_map<rdma::RcEndpoint*, LiveEndpoint> live_eps_;
  std::mutex legacy_mu_;
  std::unordered_set<uint64_t> legacy_connections_tokens_;
  uint64_t legacy_token_seed_ = 0;
  uint64_t legacy_token_sequence_ = 0;
  // Receive memory is committed in fixed-size chunks on demand. Connections
  // lease only their negotiated slot geometry from any chunk; the old
  // DFKV_RDMA_RECV_SEGMENT_SIZE remains the hard process budget.
  rdma::RecvSegmentPool recv_segments_;
  std::atomic<uint64_t> segment_evictions_{0};
  // GETs answered kNotFound by lookup_handler_ before any staging lease.
  std::atomic<uint64_t> unstaged_misses_{0};
  // Pressure reclaims given up without evicting: no idle connection whose
  // lease would make room for the request, or a request the budget can
  // never hold.
  std::atomic<uint64_t> pressure_unsatisfiable_{0};
  size_t recv_segment_max_bytes_ = 0;
  size_t recv_segment_chunk_bytes_ = 0;
  uint64_t recv_chunk_idle_ms_ = 60000;
  size_t recv_segment_registered_rails_ = 0;
  std::atomic<uint64_t> pull_connections_{0};
  std::atomic<uint64_t> data_connection_bytes_{0};
  std::atomic<uint64_t> control_connection_bytes_{0};
  // In-flight leased-PUT staging: receive memory held only while an
  // object's windows are actually in flight, unlike the connection-lifetime
  // leases above. Gauges make the pool/non-resident split observable.
  std::atomic<uint64_t> lease_put_ops_{0};
  std::atomic<uint64_t> lease_put_bytes_{0};
  std::atomic<uint64_t> lease_put_busy_rejects_{0};
  std::atomic<uint64_t> lease_put_oversize_rejects_{0};
  std::atomic<uint64_t> lease_put_active_{0};
  std::atomic<uint64_t> lease_put_bytes_active_{0};
  std::atomic<uint64_t> dynamic_get_active_{0};
  std::atomic<uint64_t> dynamic_get_bytes_active_{0};
  std::atomic<uint64_t> pull_zerocopy_served_{0};
  std::atomic<uint64_t> pull_zerocopy_register_failures_{0};
  // Server memory windows (DFKV_RDMA_SERVER_MW). Settings resolved by the
  // constructor; rail_mw_ by Start (the rail's device offers type-2B
  // windows). The registry and the quarantine outlive every connection.
  bool mw_enabled_ = false;
  uint64_t mw_floor_us_ = 0;
  uint64_t mw_max_us_ = 0;
  uint64_t mw_margin_us_ = 0;
  uint64_t mw_quarantine_cap_ = 0;
  uint64_t mw_arena_cap_ = 0;
  uint64_t mw_link_gbps_ = 0;
  uint64_t mw_rate_mbps_ = 0;  // the slowest rail's
  std::vector<uint8_t> rail_mw_;
  // Per rail: bytes of GET grants (READY sent, not yet released or revoked).
  std::unique_ptr<std::atomic<uint64_t>[]> rail_grant_bytes_;
  RamTier* mw_arena_tier_ = nullptr;
  char* mw_arena_ = nullptr;
  size_t mw_arena_bytes_ = 0;
  size_t mw_extent_bytes_ = 0;
  rdma::BindRegionRegistry mw_regions_;
  std::mutex mw_quarantine_mu_;
  std::multimap<uint64_t, MwQuarantined> mw_quarantine_;  // by end, us
  std::atomic<uint64_t> mw_quarantine_bytes_{0};
  std::atomic<uint64_t> mw_binds_{0}, mw_invalidations_{0},
      mw_invalidation_failures_{0}, mw_quarantine_fallbacks_{0},
      mw_fallback_grants_{0}, mw_fallback_connections_{0},
      mw_quarantined_total_{0};
  // Every GET grant's READ authorization (an exact MR, or a window bind).
  std::atomic<uint64_t> grant_register_ns_{0}, grant_registers_{0};
  // Test seams: a region registration (false fails it, may block: in
  // progress), and the order of a windowed grant's life ("invalidated",
  // "quarantined", "released", with the range's address).
  std::function<bool(size_t rail, char* base)> mw_register_for_test_;
  std::function<void(const char* event, const void* at)> mw_event_for_test_;
  // One anchor per resolved rail holds a lifetime shared device reference and
  // registers the initial receive chunk on that rail's PD.
  // Later chunks register lazily on the rail of the connection that leases
  // them. Auto mode anchors only the first ACTIVE local rail.
  std::vector<std::unique_ptr<rdma::RcEndpoint>> anchors_;
  std::mutex stop_mu_;  // serializes Quiesce() and the release in Stop()
  std::vector<std::string> anchor_devs_;  // fixed resolved rail names
  std::vector<std::unique_ptr<RailStats>> rail_stats_;  // indexed with anchor_devs_
  // Per rail: peer IP addresses whose connection ended in RETRY_EXC or
  // RNR_RETRY_EXC in the last 30 s (dfkv_rdma_rail_retry_exhausted_peers).
  // Observation only; a design question is whether many of them on one rail
  // should take it out of placement.
  std::vector<std::unique_ptr<PeerWindow>> rail_retry_peers_;
  // Narrow hardware-free startup seams, private and reachable only through the
  // test peer. Production always calls verbs discovery and materializes real
  // anchors.
  std::function<rdma::RdmaDiscoveryResult(
      const std::vector<std::string>&, rdma::RdmaDiscoveryPolicy)>
      discover_for_test_;
  std::function<std::unique_ptr<rdma::RcEndpoint>(const std::string&, void*,
                                                  size_t)>
      initialize_anchor_for_test_;
  // Registers and releases `segment` on a rail's anchor: proof at startup
  // that the rail can register connection memory.
  bool TrialRegistration(rdma::RcEndpoint& anchor, const std::string& device,
                         rdma::RecvSegment* segment);
  // Replaces the startup trial registration on a rail (true = registered).
  // Replaces the port query of a rail (1 ACTIVE, 0 not, -1 unknown).
  std::function<int(size_t rail)> port_state_for_test_;
  std::function<bool(const std::string&)> startup_registration_for_test_;
  // Replaces the smallest responder READ depth of the rails' devices.
  int responder_read_depth_for_test_ = 0;
  // Every rail's device accepts kV2ReadDepth READs in flight per QP: the
  // probe advertises kV2ProbeCapReadDepth and askers get the readiness ack.
  bool read_depth_offered_ = false;
#ifdef DFKV_WITH_URING
  // Externally owned backend seam for deterministic queue/completion tests.
  // Production leaves this empty and UringReader selects liburing directly.
  std::function<UringReader::Backend*()> uring_backend_factory_for_test_;
#endif
  friend class RdmaServerTestPeer;
  friend class RdmaLeaseServerTestPeer;
  bool fail_pinned_registration_for_test_ = false;
  // Test barrier immediately before endpoint destruction; ranges must remain
  // owned while delayed inbound DMA can still reach the QP.
  std::function<void()> before_endpoint_teardown_for_test_;
  // Test barrier after reply posting, before its SEND CQE is processed.
  std::function<void()> after_reply_post_for_test_;
  // Runs after a capability probe's reply is written, before its thread
  // closes the socket and returns.
  std::function<void()> after_probe_reply_for_test_;
  // Test-only: hold each bootstrap's QpInfo reply this long, as a receive
  // pool under pressure does (up to 5 s).
  std::atomic<int> bootstrap_reply_delay_ms_for_test_{0};
  // Test-only CQ filter: may withhold and later replay actual polled CQEs, or
  // rewrite the status of one. It never fabricates completions; production
  // leaves this empty.
  std::function<void(ibv_wc*, int*, size_t)> reorder_cq_for_test_;
  std::function<void(size_t, size_t, size_t)> after_cq_dispatch_for_test_;
  std::function<void(size_t)> after_request_rearm_for_test_;
  std::atomic<uint64_t> uring_reads_{0}, uring_read_batches_{0},
      uring_read_batch_max_{0}, uring_completions_{0}, uring_inflight_{0},
      uring_inflight_max_{0}, uring_replies_posted_{0},
      uring_send_fences_{0}, uring_send_post_errors_{0},
      uring_init_fallbacks_{0};
  std::atomic<uint64_t> v2_conns_{0}, v2_put_writes_{0};
  std::atomic<uint64_t> completions_{0}, completion_errors_{0}, active_conns_{0},
      idle_reclaims_{0}, grant_expiries_{0}, stop_drains_cut_{0},
      stop_grants_refused_{0};
};

}  // namespace dfkv

#endif  // DFKV_RDMA_SERVER_H_
