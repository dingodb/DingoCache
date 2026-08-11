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

#include "mds/etcd_client.h"
#include "utils/http_client.h"
#include "common/status.h"
#include "common/config_dump.h"
#include "mds/mds_metrics.h"
#include "common/membership.h"


namespace dfkv {

// A bounded, reconstructable optimization over etcd lease truth. Entries that
// have not been used locally for several etcd TTLs may be forgotten: the next
// heartbeat simply takes the normal grant+Put path. All methods hold the mutex
// only for local map work; callers perform etcd I/O after Lookup returns.
class LocalLeaseMap {
 public:
  bool LookupAndTouch(const std::string& key, uint64_t now_ms,
                      int64_t* lease_id);
  void Store(const std::string& key, int64_t lease_id, uint64_t now_ms);
  size_t MaybePrune(uint64_t now_ms, uint64_t stale_after_ms);
  size_t Prune(uint64_t now_ms, uint64_t stale_after_ms);
  size_t Size() const;

 private:
  struct Entry {
    int64_t lease_id = 0;
    uint64_t last_use_ms = 0;
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


// Stateless MDS: nodes/clients connect over TCP (same wire framing as the cache
// node). The MDS is the only etcd client; it holds each member's lease on the
// node's behalf. The {key->leaseID} map is reconstructable from heartbeats, so
// the MDS keeps no durable state. Liveness = the lease (TTL kTtlSeconds).
class MdsServer {
 public:
  // DFKV_MDS_ACCEPT_LEGACY (default off) widens the control-plane version
  // gate by exactly one epoch: v1.x peers are served with the legacy
  // 42/10-byte framing via wire_legacy.h. Off = historic strict behavior.
  explicit MdsServer(const std::string& etcd_addr, int etcd_timeout_ms = 2000);
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
    bool ready = false;
    const std::string group_metrics = GroupMetricsText(&ready);
    std::string out = metrics_.Render() + etcd_.MetricsText() + group_metrics;
    out += "# HELP dfkv_mds_ready Whether the MDS can currently reach etcd\n";
    out += "# TYPE dfkv_mds_ready gauge\n";
    out += std::string("dfkv_mds_ready ") + (ready ? "1\n" : "0\n");
    return out;
  }
  std::string GroupMetricsText(bool* etcd_reachable = nullptr);
  // kListGroups backend: distinct group names under /dfkv/v1/groups/ (newline-
  // joined). Feeds `dfkvctl stats --all`.
  Status ListGroups(std::string* out);
  size_t live_conn_count();  // handler threads not yet reaped (test/diagnostic)
  // One read against etcd (a bounded RangePrefix on a probe key). Returns true
  // iff etcd answered. Used at startup to fail loud on a misconfigured --etcd
  // (a wrong endpoint/scheme otherwise runs "healthy" while every registration
  // silently fails), and by the startup probe loop in dfkv_mds_main.
  bool ProbeEtcd() { return etcd_.RangePrefix("/dfkv/v1/_healthz_probe/").has_value(); }
  // Debounced probe for /readyz: replayed within DFKV_MDS_PROBE_CACHE_MS so
  // kubelet probe cadence can't amplify into etcd load (see EtcdProbeCache).
  bool ProbeEtcdCached() { return probe_cache_.Get(); }

  static constexpr int kTtlSeconds = 30;

 private:
  void AcceptLoop();
  // first_frame_deadline == time_point::max() means the first-frame deadline
  // knob is off (DFKV_MDS_FIRST_REQ_MS=0): the whole conn uses plain reads.
  void Handle(int fd, long io_timeout_s,
              std::chrono::steady_clock::time_point first_frame_deadline);
  void ReapDoneLocked();  // join+erase finished handler threads; conn_mu_ held
  Status Upsert(const std::string& group, const MemberInfo& m);
  Status ListMembers(const std::string& group, std::string* out,
                     bool include_degraded = false);
  // Client (consumer) registration — same lease/keepalive contract as Upsert/
  // ListMembers but under a disjoint etcd prefix (/clients/<id> vs /members/<id>)
  // so consumers never enter the placement ring. A consumer carries no data-path
  // port or stats; it is pure identity ("who is using dfkv").
  Status UpsertClient(const std::string& group, const MemberInfo& m);
  Status ListClients(const std::string& group, std::string* out);
  // Shared by Upsert/UpsertClient. The local cache is only a bounded shortcut
  // over etcd truth; all blocking etcd operations happen outside its mutex.
  Status UpsertLeased(const std::string& key, const MemberInfo& m,
                      LocalLeaseMap& leases, bool client);
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
  struct Conn {
    std::thread th;
    std::shared_ptr<std::atomic<bool>> done;
  };
  std::mutex conn_mu_;
  std::vector<int> conn_fds_;
  std::vector<Conn> conns_;
  LocalLeaseMap leases_;
  LocalLeaseMap client_leases_;
  MdsMetrics metrics_;
};

}  // namespace dfkv

#endif  // DFKV_MDS_SERVER_H_
