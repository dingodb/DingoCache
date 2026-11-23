#ifndef DFKV_MDS_SERVER_H_
#define DFKV_MDS_SERVER_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "utils/env_parse.h"
#include "mds/etcd_client.h"
#include "utils/http_client.h"
#include "common/status.h"
#include "common/config_dump.h"
#include "mds/mds_metrics.h"
#include "common/membership.h"
#include "utils/accept_guard.h"


namespace dfkv {

// A bounded, reconstructable optimization over etcd lease truth. Entries that
// have not been used locally for several etcd TTLs may be forgotten: the next
// heartbeat simply takes the normal grant+Put path. All methods hold the mutex
// only for local map work; callers perform etcd I/O after Lookup returns.
class LocalLeaseMap {
 public:
  // *put_ms (if given) receives when Store last recorded a Put of the key.
  bool LookupAndTouch(const std::string& key, uint64_t now_ms,
                      int64_t* lease_id, uint64_t* put_ms = nullptr);
  void Store(const std::string& key, int64_t lease_id, uint64_t now_ms);
  size_t MaybePrune(uint64_t now_ms, uint64_t stale_after_ms);
  size_t Prune(uint64_t now_ms, uint64_t stale_after_ms);
  size_t Size() const;

 private:
  struct Entry {
    int64_t lease_id = 0;
    uint64_t last_use_ms = 0;
    uint64_t put_ms = 0;
  };
  size_t PruneLocked(uint64_t now_ms, uint64_t stale_after_ms);

  mutable std::mutex mu_;
  std::map<std::string, Entry> entries_;
  uint64_t next_prune_ms_ = 0;
  static constexpr uint64_t kPruneIntervalMs = 30000;
};


// TTL-debounced single-flight boolean probe (the etcd reachability check behind
// /readyz). Within the TTL the last result — success OR failure — is replayed,
// so a high kubelet scrape cadence can't amplify into etcd read load during an
// outage, and a flapping etcd costs one probe per TTL instead of one per hit.
// On expiry at most ONE real probe runs; a caller arriving mid-flight waits
// for that result instead of stacking a parallel etcd range. ttl_ms == 0
// disables caching (every call probes live — the pre-knob behavior). Clock and
// prober are injected so tests drive it with a fake clock, deterministically.
class EtcdProbeCache {
 public:
  using Clock = std::function<uint64_t()>;   // steady-clock milliseconds
  using Prober = std::function<bool()>;

  EtcdProbeCache(uint64_t ttl_ms, Clock clock, Prober prober)
      : ttl_ms_(ttl_ms),
        clock_(std::move(clock)),
        prober_(std::move(prober)) {}

  bool Get();

 private:
  const uint64_t ttl_ms_;
  Clock clock_;
  Prober prober_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool have_value_ = false;
  bool refreshing_ = false;
  uint64_t fetched_at_ms_ = 0;
  bool value_ = false;
};


// Members whose heartbeat reached this MDS but could not be written to etcd.
//
// Member-list answers, kept for ttl_ms and computed by one caller at a time
// per key (the others wait for its answer), so clients polling every few
// seconds cost etcd one range per group and ttl instead of one per poll.
// Invalidate drops the answers of a group, and a computation already running
// when it is called does not refresh them. ttl_ms == 0 caches nothing. Clock
// injected for tests.
class ListCache {
 public:
  using Clock = std::function<uint64_t()>;  // steady-clock milliseconds
  using Compute = std::function<Status(std::string*)>;
  ListCache(uint64_t ttl_ms, Clock clock)
      : ttl_ms_(ttl_ms), clock_(std::move(clock)) {}
  // *hit tells whether the answer came from the cache.
  Status Get(const std::string& group, bool degraded, std::string* out,
             const Compute& compute, bool* hit = nullptr);
  void Invalidate(const std::string& group);
  size_t SizeForTest() {
    std::lock_guard<std::mutex> lk(mu_);
    return entries_.size();
  }

 private:
  struct Entry {
    bool valid = false;
    bool loading = false;
    uint64_t at_ms = 0;
    uint64_t generation = 0;
    Status status = Status::kIOError;
    std::string out;
  };
  const uint64_t ttl_ms_;
  Clock clock_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::map<std::pair<std::string, bool>, Entry> entries_;
  // Past this many entries, expired ones are dropped when a new one is made.
  static constexpr size_t kMaxEntries = 1024;
};

// The heartbeat proves the node is alive. When etcd takes no writes (out of
// space it refuses Put and LeaseGrant, yet answers Range and keeps expiring
// leases) the member's key runs out although nothing is wrong with the node,
// and a member list read from etcd then omits nodes this MDS has just heard
// from. ListMembers uses this record to tell such a view from a real removal:
// a recorded member that etcd no longer lists means the view is not the
// member list.
//
// Only heartbeats are recorded, never registrations: see MdsServer::Upsert.
//
// A record goes away when the member's next write here succeeds, or `hold_ms`
// after the failed heartbeat: by then a live member would have come back (its
// registrar rotates over the MDS endpoints), so it has stopped heartbeating
// here and its absence may be real. hold_ms == 0 records nothing. Clock is
// injected for tests. Only local map work happens under the mutex.
class UnwrittenMembers {
 public:
  using Clock = std::function<uint64_t()>;  // steady-clock milliseconds

  UnwrittenMembers(uint64_t hold_ms, Clock clock)
      : hold_ms_(hold_ms), clock_(std::move(clock)) {}

  void Add(const std::string& key);
  void Remove(const std::string& key);
  // Recorded keys that start with `prefix` and are within the hold, sorted.
  std::vector<std::string> Under(const std::string& prefix) const;
  size_t Count() const;  // recorded keys within the hold
  size_t Size() const;   // records kept, lapsed ones included (test/diagnostic)

 private:
  bool Held(uint64_t failed_at_ms, uint64_t now_ms) const {
    return now_ms < failed_at_ms || now_ms - failed_at_ms <= hold_ms_;
  }

  const uint64_t hold_ms_;
  Clock clock_;
  mutable std::mutex mu_;
  std::map<std::string, uint64_t> failed_at_ms_;  // etcd key -> failed heartbeat
  uint64_t next_prune_ms_ = 0;
};


// Stateless MDS: nodes/clients connect over TCP (same wire framing as the cache
// node). The MDS is the only etcd client; it holds each member's lease on the
// node's behalf. The {key->leaseID} map is reconstructable from heartbeats, so
// the MDS keeps no durable state. Liveness = the lease (TTL kTtlSeconds).
// DFKV_MDS_FIRST_REQ_MS: the first-frame deadline of a connection, ms. 0
// disables it; garbage or a negative value falls back to the default (it
// must never DISABLE the safeguard); hard-clamped at 1 h so the accept+ms
// deadline cannot wrap. Read with strtol and an errno check up to 2.38, so a
// number that does not fit a long was the default too.
inline long MdsFirstReqMsFromEnv() {
  envparse::U64Spec spec;
  spec.fallback = 30000;
  spec.max = 3600000;
  spec.above = envparse::Above::kClamp;
  spec.legacy = envparse::Legacy::kFull;
  spec.representable_max = envparse::kSignedMax;
  spec.overflow_is_default = true;
  return static_cast<long>(envparse::EnvU64("DFKV_MDS_FIRST_REQ_MS", spec));
}

// Open files the MDS needs to serve `max_conns` connections: each handler
// holds its client socket and, while it works on a request, one socket to
// etcd. The reserve is an estimate, about 128, for the listeners, the metrics
// port's own connections (64 by default), the startup probe and stdio.
constexpr uint64_t kMdsReservedDescriptors = 128;
constexpr uint64_t MdsDescriptorsFor(uint64_t max_conns) {
  return 2 * max_conns + kMdsReservedDescriptors;
}

struct MdsDescriptorPlan {
  uint64_t soft_limit;  // RLIMIT_NOFILE soft limit to run with
  size_t max_conns;     // connection cap that fits under it, at least 1
};

// Fits the requested connection cap to the open-file limit: raise the soft
// limit toward the hard limit as far as needed (never lower it), and if even
// the hard limit is short, shrink the cap so the etcd sockets and the
// reserve still fit. Pure, so the arithmetic is testable without touching
// the process's limits.
inline MdsDescriptorPlan PlanMdsDescriptors(size_t max_conns, uint64_t soft,
                                            uint64_t hard) {
  const uint64_t need = MdsDescriptorsFor(max_conns);
  MdsDescriptorPlan plan{soft, max_conns};
  if (soft >= need) return plan;
  plan.soft_limit = need < hard ? need : hard;
  if (plan.soft_limit < soft) plan.soft_limit = soft;
  if (plan.soft_limit >= need) return plan;
  const uint64_t fits = plan.soft_limit > kMdsReservedDescriptors
                            ? (plan.soft_limit - kMdsReservedDescriptors) / 2
                            : 0;
  plan.max_conns = fits == 0 ? 1 : static_cast<size_t>(fits);
  return plan;
}

class MdsServer {
 public:
  // DFKV_MDS_ACCEPT_LEGACY (default off) widens the control-plane version
  // gate by exactly one epoch: v1.x peers are served with the legacy
  // 42/10-byte framing via wire_legacy.h. Off = historic strict behavior.
  explicit MdsServer(const std::string& etcd_addr, int etcd_timeout_ms = 2000);
  // Same, with the clock behind the unwritten-member hold supplied: tests let
  // the hold pass without waiting for it.
  MdsServer(const std::string& etcd_addr, int etcd_timeout_ms,
            UnwrittenMembers::Clock hold_clock);
  ~MdsServer();

  Status Start(int port);
  void Stop();
  int port() const { return port_; }
  // Static counters + per-group ring aggregates. One etcd prefix range over
  // /dfkv/v1/groups/ both proves readiness and supplies every group aggregate;
  // a scrape never performs a second sequential health range.
  std::string MetricsText() {
    metrics_.local_member_leases.store(leases_.Size(),
                                       std::memory_order_relaxed);
    metrics_.local_client_leases.store(client_leases_.Size(),
                                       std::memory_order_relaxed);
    metrics_.unwritten_members.store(unwritten_.Count(),
                                     std::memory_order_relaxed);
    bool ready = false;
    const std::string group_metrics = GroupMetricsText(&ready);
    std::string out = metrics_.Render() + ListenerMetricsText() +
                      etcd_.MetricsText() + group_metrics;
    out += "# HELP dfkv_mds_ready Whether the MDS can currently reach etcd\n";
    out += "# TYPE dfkv_mds_ready gauge\n";
    out += std::string("dfkv_mds_ready ") + (ready ? "1\n" : "0\n");
    return out;
  }
  std::string GroupMetricsText(bool* etcd_reachable = nullptr);
  std::string ListenerMetricsText() const;  // dfkv_mds_accept_*, thread starts
  // kListGroups backend: distinct group names under /dfkv/v1/groups/ (newline-
  // joined). Feeds `dfkvctl stats --all`.
  Status ListGroups(std::string* out);
  size_t live_conn_count();  // handler threads not yet reaped (test/diagnostic)
  // DFKV_MDS_MAX_CONNS as in force after Start() fitted it to the open-file
  // limit.
  size_t max_conns() const { return max_conns_; }
  uint64_t AcceptFailures() const { return accept_guard_.accept_failures(); }
  uint64_t ThreadStartFailures() const {
    return accept_guard_.thread_start_failures();
  }
  // One read against etcd (a bounded RangePrefix on a probe key). Returns true
  // iff etcd answered. Used at startup to fail loud on a misconfigured --etcd
  // (a wrong endpoint/scheme otherwise runs "healthy" while every registration
  // silently fails), and by the startup probe loop in dfkv_mds_main.
  bool ProbeEtcd() { return etcd_.RangePrefix("/dfkv/v1/_healthz_probe/").has_value(); }
  // Debounced probe for /readyz: replayed within DFKV_MDS_PROBE_CACHE_MS so
  // kubelet probe cadence can't amplify into etcd load (see EtcdProbeCache).
  bool ProbeEtcdCached() { return probe_cache_.Get(); }

  static constexpr int kTtlSeconds = 30;
  // Default DFKV_MDS_STATS_PUT_MS: a member's record is rewritten for a
  // change of its stats (STA1) alone at most this often.
  static constexpr uint64_t kStatsPutMs = 60000;
  // Default DFKV_MDS_LIST_CACHE_MS.
  static constexpr uint64_t kListCacheMs = 500;
  // Default DFKV_MDS_UNWRITTEN_HOLD_MS: four lease TTLs, the margin the local
  // lease shortcut already gives a registrar rotating over several MDS.
  static constexpr uint64_t kUnwrittenHoldMs =
      static_cast<uint64_t>(kTtlSeconds) * 4 * 1000;

 private:
  void AcceptLoop();
  // first_frame_deadline == time_point::max() means the first-frame deadline
  // knob is off (DFKV_MDS_FIRST_REQ_MS=0): the whole conn uses plain reads.
  void Handle(int fd, long io_timeout_s,
              std::chrono::steady_clock::time_point first_frame_deadline);
  void ReapDoneLocked();  // join+erase finished handler threads; conn_mu_ held
  // heartbeat: the op was kHeartbeat rather than kRegister.
  Status Upsert(const std::string& group, const MemberInfo& m, bool heartbeat);
  // Deletes the member's record (kDeregister): sent by a node that was
  // drained and is stopping.
  Status Deregister(const std::string& group, const MemberInfo& m);
  Status ListMembers(const std::string& group, std::string* out,
                     bool include_degraded = false);
  Status ListMembersFromEtcd(const std::string& group, std::string* out,
                             bool include_degraded);
  // Member fencing (DFKV_MDS_FENCING). A fenced id has a record without a
  // lease under /fenced/<id>: its registers and heartbeats are answered
  // kNotFound and not written, its member record is deleted when it is
  // fenced, and member lists leave it out. One step removes a node that is
  // still running, until it is unfenced.
  Status Fence(const std::string& payload, bool fence);
  Status ListFenced(const std::string& group, std::string* out);
  static std::string FencedKey(const std::string& group, const std::string& id);
  // Whether `id` is fenced; nullopt = etcd did not answer.
  std::optional<bool> IsFenced(const std::string& group, const std::string& id);

  // Client (consumer) registration — same lease/keepalive contract as Upsert/
  // ListMembers but under a disjoint etcd prefix (/clients/<id> vs /members/<id>)
  // so consumers never enter the placement ring. A consumer carries no data-path
  // port or stats; it is pure identity ("who is using dfkv").
  Status UpsertClient(const std::string& group, const MemberInfo& m);
  Status ListClients(const std::string& group, std::string* out);
  // Shared by Upsert/UpsertClient. The local cache is only a bounded shortcut
  // over etcd truth; all blocking etcd operations happen outside its mutex.
  // *wrote: the record was written to etcd (not only its lease renewed).
  Status UpsertLeased(const std::string& key, const MemberInfo& m,
                      LocalLeaseMap& leases, bool client,
                      bool* wrote = nullptr);
  static std::string MemberKey(const std::string& group, const std::string& id);
  static std::string ClientKey(const std::string& group, const std::string& id);

  TcpHttpTransport http_;
  EtcdClient etcd_;
  // Resolved once at ctor; widens Handle()'s control-plane version gate by
  // the single legacy epoch (v1.x peers, wire_legacy.h framing).
  const bool accept_legacy_;
  EtcdProbeCache probe_cache_;  // TTL + prober resolved in the ctor (env knob)
  std::atomic<bool> running_{false};
  int listen_fd_ = -1;
  int port_ = 0;
  std::thread accept_thread_;
  net::AcceptGuard accept_guard_{"mds listener"};
  static constexpr size_t kDefaultMaxConns = 4096;
  size_t max_conns_ = kDefaultMaxConns;  // set by Start(), then read-only
  struct Conn {
    std::thread th;
    std::shared_ptr<std::atomic<bool>> done;
  };
  std::mutex conn_mu_;
  std::vector<int> conn_fds_;
  std::vector<Conn> conns_;
  LocalLeaseMap leases_;
  LocalLeaseMap client_leases_;
  uint64_t stats_put_ms_;  // DFKV_MDS_STATS_PUT_MS, resolved in the ctor
  ListCache list_cache_;   // DFKV_MDS_LIST_CACHE_MS, resolved in the ctor
  bool fencing_;           // DFKV_MDS_FENCING, resolved in the ctor
  // Fenced ids per group, read at most once a second for the heartbeat path
  // (a list reads them afresh). A fence or unfence here drops the group's.
  struct FencedIds {
    uint64_t at_ms = 0;
    std::vector<std::string> ids;
  };
  std::mutex fenced_mu_;
  std::map<std::string, FencedIds> fenced_cache_;
  static constexpr uint64_t kFencedCacheMs = 1000;
  static constexpr size_t kMaxFencedGroups = 1024;
  std::optional<std::vector<std::string>> FencedIdsOf(const std::string& group,
                                                      bool fresh);
  UnwrittenMembers unwritten_;  // hold resolved in the ctor (env knob)
  std::atomic<uint64_t> withheld_log_ms_{0};  // rate limit of the WARN line
  MdsMetrics metrics_;
  void NoteUndecodable(const std::string& key);
  std::atomic<uint64_t> undecodable_logged_ms_{0};
};

}  // namespace dfkv

#endif  // DFKV_MDS_SERVER_H_
