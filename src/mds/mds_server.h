#ifndef DFKV_MDS_SERVER_H_
#define DFKV_MDS_SERVER_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "mds/etcd_client.h"
#include "utils/http_client.h"
#include "common/status.h"
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


// Stateless MDS: nodes/clients connect over TCP (same wire framing as the cache
// node). The MDS is the only etcd client; it holds each member's lease on the
// node's behalf. The {key->leaseID} map is reconstructable from heartbeats, so
// the MDS keeps no durable state. Liveness = the lease (TTL kTtlSeconds).
class MdsServer {
 public:
  explicit MdsServer(const std::string& etcd_addr, int etcd_timeout_ms = 2000)
      : http_(etcd_addr, etcd_timeout_ms), etcd_(&http_) {}
  ~MdsServer();

  Status Start(int port);
  void Stop();
  int port() const { return port_; }
  // Static counters + per-group ring aggregates. The aggregate half does ONE
  // etcd prefix range over /dfkv/v1/groups/ at scrape time (MDS stays
  // stateless; ~30s Prometheus cadence makes this negligible), decodes each
  // member's STA1 stats and sums them per group -- ring capacity / usage /
  // hit-rate / alarm counters become one MDS scrape instead of a fleet sweep.
  std::string MetricsText() {
    metrics_.local_member_leases.store(leases_.Size(),
                                        std::memory_order_relaxed);
    metrics_.local_client_leases.store(client_leases_.Size(),
                                        std::memory_order_relaxed);
    return metrics_.Render() + GroupMetricsText();
  }
  std::string GroupMetricsText();
  // kListGroups backend: distinct group names under /dfkv/v1/groups/ (newline-
  // joined). Feeds `dfkvctl stats --all`.
  Status ListGroups(std::string* out);
  size_t live_conn_count();  // handler threads not yet reaped (test/diagnostic)
  // One read against etcd (a bounded RangePrefix on a probe key). Returns true
  // iff etcd answered. Used at startup to fail loud on a misconfigured --etcd
  // (a wrong endpoint/scheme otherwise runs "healthy" while every registration
  // silently fails), and by /healthz to reflect etcd reachability.
  bool ProbeEtcd() { return etcd_.RangePrefix("/dfkv/v1/_healthz_probe/").has_value(); }

  static constexpr int kTtlSeconds = 30;

 private:
  void AcceptLoop();
  void Handle(int fd);
  void ReapDoneLocked();  // join+erase finished handler threads; conn_mu_ held
  Status Upsert(const std::string& group, const MemberInfo& m);
  Status ListMembers(const std::string& group, std::string* out);
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
