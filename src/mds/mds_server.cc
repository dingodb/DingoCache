#include "utils/env_parse.h"
#include "common/config_dump.h"
#include "mds/mds_server.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <chrono>
#include <map>

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <unistd.h>

#include "common/membership.h"
#include "mds/mds_proto.h"
#include "utils/log.h"
#include "utils/net_util.h"
#include "utils/thread_name.h"
#include "utils/wire_limits.h"
#include "transport/wire.h"
#include "transport/wire_legacy.h"

namespace dfkv {

namespace {
uint64_t SteadyNowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
}  // namespace

bool LocalLeaseMap::LookupAndTouch(const std::string& key, uint64_t now_ms,
                                   int64_t* lease_id, uint64_t* put_ms) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = entries_.find(key);
  if (it == entries_.end()) return false;
  it->second.last_use_ms = now_ms;
  *lease_id = it->second.lease_id;
  if (put_ms) *put_ms = it->second.put_ms;
  return true;
}

void LocalLeaseMap::Store(const std::string& key, int64_t lease_id,
                          uint64_t now_ms) {
  std::lock_guard<std::mutex> lk(mu_);
  entries_[key] = Entry{lease_id, now_ms, now_ms};
}

size_t LocalLeaseMap::PruneLocked(uint64_t now_ms, uint64_t stale_after_ms) {
  size_t pruned = 0;
  for (auto it = entries_.begin(); it != entries_.end();) {
    const uint64_t last = it->second.last_use_ms;
    if (now_ms >= last && now_ms - last > stale_after_ms) {
      it = entries_.erase(it);
      ++pruned;
    } else {
      ++it;
    }
  }
  return pruned;
}

size_t LocalLeaseMap::MaybePrune(uint64_t now_ms, uint64_t stale_after_ms) {
  std::lock_guard<std::mutex> lk(mu_);
  if (now_ms < next_prune_ms_) return 0;
  next_prune_ms_ =
      now_ms > UINT64_MAX - kPruneIntervalMs
          ? UINT64_MAX
          : now_ms + kPruneIntervalMs;
  return PruneLocked(now_ms, stale_after_ms);
}

size_t LocalLeaseMap::Prune(uint64_t now_ms, uint64_t stale_after_ms) {
  std::lock_guard<std::mutex> lk(mu_);
  return PruneLocked(now_ms, stale_after_ms);
}

size_t LocalLeaseMap::Size() const {
  std::lock_guard<std::mutex> lk(mu_);
  return entries_.size();
}

void UnwrittenMembers::Add(const std::string& key) {
  if (hold_ms_ == 0) return;
  const uint64_t now_ms = clock_();
  std::lock_guard<std::mutex> lk(mu_);
  if (now_ms >= next_prune_ms_) {
    // Once per hold: forget members that stopped heartbeating here. Under() and
    // Count() skip them anyway; this only bounds the map under id churn.
    for (auto it = failed_at_ms_.begin(); it != failed_at_ms_.end();) {
      if (Held(it->second, now_ms))
        ++it;
      else
        it = failed_at_ms_.erase(it);
    }
    next_prune_ms_ =
        now_ms > UINT64_MAX - hold_ms_ ? UINT64_MAX : now_ms + hold_ms_;
  }
  failed_at_ms_[key] = now_ms;
}

void UnwrittenMembers::Remove(const std::string& key) {
  std::lock_guard<std::mutex> lk(mu_);
  failed_at_ms_.erase(key);
}

std::vector<std::string> UnwrittenMembers::Under(
    const std::string& prefix) const {
  std::vector<std::string> keys;
  std::lock_guard<std::mutex> lk(mu_);
  if (failed_at_ms_.empty()) return keys;  // the normal case: no clock read
  const uint64_t now_ms = clock_();
  for (auto it = failed_at_ms_.lower_bound(prefix);
       it != failed_at_ms_.end() &&
       it->first.compare(0, prefix.size(), prefix) == 0;
       ++it) {
    if (Held(it->second, now_ms)) keys.push_back(it->first);
  }
  return keys;
}

size_t UnwrittenMembers::Count() const {
  std::lock_guard<std::mutex> lk(mu_);
  if (failed_at_ms_.empty()) return 0;
  const uint64_t now_ms = clock_();
  size_t held = 0;
  for (const auto& entry : failed_at_ms_)
    if (Held(entry.second, now_ms)) ++held;
  return held;
}

size_t UnwrittenMembers::Size() const {
  std::lock_guard<std::mutex> lk(mu_);
  return failed_at_ms_.size();
}

bool EtcdProbeCache::Get() {
  if (ttl_ms_ == 0) return prober_();  // caching off: live probe, old behavior
  std::unique_lock<std::mutex> lk(mu_);
  const uint64_t now = clock_();
  if (have_value_ && now >= fetched_at_ms_ && now - fetched_at_ms_ < ttl_ms_)
    return value_;  // fresh: replay the last result (failures included)
  if (refreshing_) {
    // Single-flight: wait for the in-flight probe instead of stacking a
    // concurrent etcd read; the result it caches is fresh by construction.
    cv_.wait(lk, [this] { return !refreshing_; });
    return value_;
  }
  refreshing_ = true;
  lk.unlock();
  const bool v = prober_();  // the one real probe for this TTL window
  lk.lock();
  value_ = v;
  have_value_ = true;
  fetched_at_ms_ = clock_();
  refreshing_ = false;
  cv_.notify_all();
  return value_;
}

MdsServer::MdsServer(const std::string& etcd_addr, int etcd_timeout_ms)
    : MdsServer(etcd_addr, etcd_timeout_ms, SteadyNowMs) {}

MdsServer::MdsServer(const std::string& etcd_addr, int etcd_timeout_ms,
                     UnwrittenMembers::Clock hold_clock)
    : http_(etcd_addr, etcd_timeout_ms),
      etcd_(&http_),
      accept_legacy_(config_dump::EnvBool("DFKV_MDS_ACCEPT_LEGACY", false)),
      // DFKV_MDS_PROBE_CACHE_MS: /readyz probe debounce, default 2500 ms,
      // 0 disables caching (ProbeEtcdCached then behaves like ProbeEtcd).
      // Clamped at 10 min: a negative env value wraps into ~2^64 and would
      // pin /readyz on its first result for the process lifetime.
      probe_cache_(std::min<uint64_t>(
                       config_dump::EnvU64("DFKV_MDS_PROBE_CACHE_MS", 2500),
                       600000),
                   SteadyNowMs,
                   [this] { return ProbeEtcd(); }),
      // DFKV_MDS_UNWRITTEN_HOLD_MS: how long a member whose heartbeat etcd did
      // not take still counts as alive here (see UnwrittenMembers). 0 = never
      // withhold a member list, the behavior up to 2.36. Clamped like the
      // probe cache above, for the same reason.
      unwritten_(std::min<uint64_t>(
                     config_dump::EnvU64("DFKV_MDS_UNWRITTEN_HOLD_MS",
                                         kUnwrittenHoldMs),
                     600000),
                 std::move(hold_clock)),
      // DFKV_MDS_STATS_PUT_MS: see kStatsPutMs. 0 = rewrite on every
      // heartbeat whose stats changed (every heartbeat of a serving node).
      stats_put_ms_(std::min<uint64_t>(
          config_dump::EnvU64("DFKV_MDS_STATS_PUT_MS", kStatsPutMs),
          600000)),
      // DFKV_MDS_LIST_CACHE_MS: how long a member-list answer is reused (see
      // ListCache); 0 = every poll reads etcd, the behavior up to 2.42.
      // Client lists use it too, at most kMaxClientListCacheMs.
      list_cache_(ListCacheMs(), SteadyNowMs),
      client_list_cache_(ClientListCacheMs(), SteadyNowMs),
      // DFKV_MDS_FENCING: member fencing (dfkvctl member fence). Off by
      // default: it adds records under /fenced/ to the etcd layout and a
      // read per heartbeat, and every MDS of a group has to have it on, or
      // one without it lets a fenced node back in.
      fencing_(config_dump::EnvBool("DFKV_MDS_FENCING", false)) {}

MdsServer::~MdsServer() { Stop(); }

std::string MdsServer::MemberKey(const std::string& group, const std::string& id) {
  return "/dfkv/v1/groups/" + group + "/members/" + id;
}

std::string MdsServer::ClientKey(const std::string& group, const std::string& id) {
  return "/dfkv/v1/groups/" + group + "/clients/" + id;
}

Status MdsServer::Start(int port) {
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) return Status::kIOError;
  int one = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(INADDR_ANY);
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
  // DFKV_MDS_MAX_CONNS caps the handler-thread count so a connect flood
  // degrades to refused conns instead of unbounded threads. It is reconciled
  // with the descriptor limit here: a cap the process has no descriptors for
  // is not a cap, it is accept() and the etcd dials failing together.
  size_t max_conns = kDefaultMaxConns;
  {
    envparse::U64Spec spec;
    spec.legacy = envparse::Legacy::kPrefixOrZero;
    spec.fallback = max_conns;
    spec.min = 1;
    spec.representable_max = envparse::kSignedMax;  // read with atol
    max_conns = static_cast<size_t>(envparse::EnvU64("DFKV_MDS_MAX_CONNS", spec));
  }
  rlimit nofile{};
  if (::getrlimit(RLIMIT_NOFILE, &nofile) == 0) {
    const MdsDescriptorPlan plan =
        PlanMdsDescriptors(max_conns, nofile.rlim_cur, nofile.rlim_max);
    if (plan.soft_limit != nofile.rlim_cur) {
      rlimit raised = nofile;
      raised.rlim_cur = static_cast<rlim_t>(plan.soft_limit);
      if (::setrlimit(RLIMIT_NOFILE, &raised) == 0) {
        DFKV_LOG_INFO("mds: raised the open-file soft limit from " +
                      std::to_string(nofile.rlim_cur) + " to " +
                      std::to_string(plan.soft_limit) + " for " +
                      std::to_string(max_conns) + " connections");
        nofile = raised;
      }
    }
    // Recomputed from the limit actually in force, in case the raise failed.
    const size_t fits =
        PlanMdsDescriptors(max_conns, nofile.rlim_cur, nofile.rlim_cur)
            .max_conns;
    if (fits < max_conns) {
      DFKV_LOG_WARN("mds: DFKV_MDS_MAX_CONNS=" + std::to_string(max_conns) +
                    " needs " + std::to_string(MdsDescriptorsFor(max_conns)) +
                    " open files but the limit is " +
                    std::to_string(nofile.rlim_cur) +
                    "; serving at most " + std::to_string(fits) +
                    " connections. Raise LimitNOFILE (ulimit -n) to lift this.");
      max_conns = fits;
    }
  }
  max_conns_ = max_conns;
  config_dump::RecordResolved("DFKV_MDS_MAX_CONNS", std::to_string(max_conns_));
  running_ = true;
  accept_thread_ = std::thread([this] { NameThisThread("mds-accept"); AcceptLoop(); });
  return Status::kOk;
}

void MdsServer::Stop() {
  if (!running_.exchange(false)) return;
  if (listen_fd_ >= 0) ::shutdown(listen_fd_, SHUT_RDWR);
  if (accept_thread_.joinable()) accept_thread_.join();
  if (listen_fd_ >= 0) { ::close(listen_fd_); listen_fd_ = -1; }
  std::vector<int> fds;
  std::vector<Conn> conns;
  {
    std::lock_guard<std::mutex> lk(conn_mu_);
    fds.assign(conn_fds_.begin(), conn_fds_.end());
    conns.swap(conns_);
  }
  for (int fd : fds) ::shutdown(fd, SHUT_RDWR);
  for (auto& c : conns) if (c.th.joinable()) c.th.join();
}

void MdsServer::ReapDoneLocked() {
  for (auto it = conns_.begin(); it != conns_.end();) {
    if (it->done->load(std::memory_order_acquire)) {
      if (it->th.joinable()) it->th.join();
      it = conns_.erase(it);
    } else {
      ++it;
    }
  }
}

size_t MdsServer::live_conn_count() {
  std::lock_guard<std::mutex> lk(conn_mu_);
  ReapDoneLocked();  // count only genuinely-live handlers, not finished-unreaped ones
  return conns_.size();
}

void MdsServer::AcceptLoop() {
  // Silent/half-open peers must not pin a handler thread forever (Handle is a
  // thread-per-conn long-lived loop): bound both directions, like the RDMA
  // bootstrap does. Legit clients speak every few seconds (member poll /
  // heartbeat), far inside the 60 s default; a timed-out ReadAll/WriteAll
  // returns false and the handler closes the conn (the client re-dials).
  // DFKV_MDS_IO_TIMEOUT_S tunes it (tests use 1).
  long tmo_s = 60;
  {
    envparse::U64Spec spec;
    spec.legacy = envparse::Legacy::kPrefixOrZero;
    spec.fallback = static_cast<uint64_t>(tmo_s);
    spec.min = 1;
    spec.representable_max = envparse::kSignedMax;  // read with atol
    tmo_s = static_cast<long>(envparse::EnvU64("DFKV_MDS_IO_TIMEOUT_S", spec));
  }
  // First-frame ABSOLUTE deadline (DFKV_MDS_FIRST_REQ_MS, default 30000,
  // 0=disabled): the per-syscall SO_RCVTIMEO above only bounds ONE recv, so a
  // peer dribbling one byte per sub-timeout interval would pin a handler
  // thread forever. With the knob on, the first COMPLETE frame must arrive
  // within this budget counted from accept(); steady-state traffic is
  // untouched (plain per-syscall timeout, same as before).
  const long first_req_ms = MdsFirstReqMsFromEnv();
  // Recorded from the accept thread's prologue (runs at Start, before main
  // reaches Emit after its network etcd probe) so the dump shows these defaults.
  config_dump::RecordResolved("DFKV_MDS_IO_TIMEOUT_S", std::to_string(tmo_s));
  config_dump::RecordResolved("DFKV_MDS_FIRST_REQ_MS", std::to_string(first_req_ms));
  while (running_) {
    int fd = ::accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) {
      const int err = errno;
      if (!running_) break;
      accept_guard_.OnAcceptError(err, running_);
      continue;
    }
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    timeval tv{tmo_s, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    std::lock_guard<std::mutex> lk(conn_mu_);
    if (!running_) { ::close(fd); break; }
    ReapDoneLocked();  // join handlers that finished since the last accept
    if (conns_.size() >= max_conns_) { ::close(fd); continue; }
    // The deadline is stamped at ACCEPT time (before the handler thread even
    // starts), so a thread-start delay can never extend it.
    const auto first_req_deadline =
        first_req_ms > 0
            ? std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(first_req_ms)
            : std::chrono::steady_clock::time_point::max();
    auto done = std::make_shared<std::atomic<bool>>(false);
    std::thread handling;
    if (!net::StartThread(&handling, [this, fd, done, tmo_s, first_req_deadline] {
          NameThisThread("mds-conn");
          Handle(fd, tmo_s, first_req_deadline);
          {
            // Taken only after the accept loop, which holds conn_mu_ across
            // the thread start, has recorded this fd.
            std::lock_guard<std::mutex> lk(conn_mu_);
            for (auto it = conn_fds_.begin(); it != conn_fds_.end(); ++it)
              if (*it == fd) { conn_fds_.erase(it); break; }
          }
          ::close(fd);
          done->store(true, std::memory_order_release);  // last act
        })) {
      accept_guard_.OnThreadStartFailure();
      ::close(fd);
      continue;
    }
    conn_fds_.push_back(fd);
    conns_.push_back({std::move(handling), done});
  }
}

Status MdsServer::Upsert(const std::string& group, const MemberInfo& m,
                         bool heartbeat) {
  // Reject before the id/group reach the etcd key: an unrestricted token can
  // contain '/' and escape its own key subtree (phantom-member injection).
  if (!IsValidGroupOrId(group) || !IsValidGroupOrId(m.id)) return Status::kInvalid;
  const std::string key = MemberKey(group, m.id);
  if (fencing_) {
    const auto fenced = IsFenced(group, m.id);
    if (!fenced) {
      metrics_.etcd_errors.fetch_add(1, std::memory_order_relaxed);
      return Status::kIOError;
    }
    if (*fenced) {
      metrics_.fenced_refusals.fetch_add(1, std::memory_order_relaxed);
      return Status::kNotFound;
    }
  }
  bool wrote = false;
  const Status st = UpsertLeased(key, m, leases_, false, &wrote);
  // A record written here changes the list: the next poll reads it.
  if (wrote) list_cache_.Invalidate(group);
  // A heartbeat etcd did not take is recorded for ListMembers (see
  // UnwrittenMembers). A failed register is not. A registrar heartbeats only
  // after a register of the same process went through, so a heartbeat comes
  // from a node that was listed; a register may come from one that never was
  // (a new node, or one back after its key ran out). Counting that one as a
  // missing member would withhold the group's list from everybody for as long
  // as it keeps trying to get in.
  if (st == Status::kOk)
    unwritten_.Remove(key);
  else if (heartbeat)
    unwritten_.Add(key);
  return st;
}

Status MdsServer::UpsertClient(const std::string& group, const MemberInfo& m) {
  // Same path-traversal guard as Upsert; clients live under a parallel prefix
  // (/clients/<id>) so a malformed id can't escape into /members/ either.
  if (!IsValidGroupOrId(group) || !IsValidGroupOrId(m.id)) return Status::kInvalid;
  bool wrote = false;
  const Status st =
      UpsertLeased(ClientKey(group, m.id), m, client_leases_, true, &wrote);
  // A record written here changes the client list: the next poll reads it.
  if (wrote) client_list_cache_.Invalidate(group);
  return st;
}

namespace {
// `stored` and `next` encode the same member except for its stats: rewriting
// the record would only refresh them.
// Whatever else differs, drain included (DRN1 is in the encoding, and a
// view without a draining member has none), is a record change and is
// written at once.
bool OnlyStatsDiffer(const std::string& stored, const MemberInfo& next,
                     const std::string& next_encoded) {
  std::vector<MemberInfo> one;
  uint64_t epoch = 0;
  if (!DecodeMembers(stored.data(), stored.size(), &one, &epoch) ||
      one.size() != 1)
    return false;
  one[0].stats = next.stats;
  one[0].has_stats = next.has_stats;
  return EncodeMembers(one, 0) == next_encoded;
}
}  // namespace

// Shared lease renewal for members and clients. The record carries the lease
// it is attached to, and any MDS may keep any lease alive: a heartbeat reads
// the record, renews that lease, and writes only if the record changed. The
// stats alone (they change on every beat of a serving node) are rewritten at
// most every stats_put_ms_. A record that is gone, or whose lease expired, is
// written under a new lease. So a heartbeat costs etcd a read and a lease
// renewal, neither of which adds a revision, instead of a Put (and, behind
// registrars rotating over several MDS, a lease grant too, as each MDS's own
// lease had lapsed by the time the rotation came back to it). The key expires
// when nobody has renewed its lease for kTtlSeconds, as before.
//
// The local map keeps, per key, when this MDS last wrote it (for the stats
// interval). Entries are discarded after four etcd TTLs without local use.
Status MdsServer::UpsertLeased(const std::string& key, const MemberInfo& m,
                               LocalLeaseMap& leases, bool client,
                               bool* wrote) {
  if (wrote) *wrote = false;
  const std::string val = EncodeMembers({m}, 0);
  constexpr uint64_t kLocalLeaseStaleMs =
      static_cast<uint64_t>(kTtlSeconds) * 4 * 1000;
  const uint64_t now_ms = SteadyNowMs();
  const size_t pruned = leases.MaybePrune(now_ms, kLocalLeaseStaleMs);
  auto& current_metric =
      client ? metrics_.local_client_leases : metrics_.local_member_leases;
  auto& pruned_metric = client ? metrics_.local_client_leases_pruned
                               : metrics_.local_member_leases_pruned;
  if (pruned != 0)
    pruned_metric.fetch_add(pruned, std::memory_order_relaxed);
  current_metric.store(leases.Size(), std::memory_order_relaxed);
  auto io_error = [&] {
    metrics_.etcd_errors.fetch_add(1, std::memory_order_relaxed);
    return Status::kIOError;
  };
  auto put = [&](int64_t lease) {
    if (!etcd_.Put(key, val, lease)) return false;
    metrics_.record_puts.fetch_add(1, std::memory_order_relaxed);
    if (wrote) *wrote = true;
    leases.Store(key, lease, SteadyNowMs());
    current_metric.store(leases.Size(), std::memory_order_relaxed);
    return true;
  };
  const auto stored = etcd_.Get(key);
  if (!stored) return io_error();
  if (stored->exists && stored->lease != 0 &&
      etcd_.LeaseKeepAlive(stored->lease)) {
    if (stored->value == val) {
      metrics_.record_puts_skipped.fetch_add(1, std::memory_order_relaxed);
      return Status::kOk;
    }
    int64_t known = 0;
    uint64_t put_ms = 0;
    const bool recent =
        leases.LookupAndTouch(key, now_ms, &known, &put_ms) &&
        now_ms >= put_ms && now_ms - put_ms < stats_put_ms_;
    const bool stats_only = OnlyStatsDiffer(stored->value, m, val);
    if (recent && stats_only) {
      metrics_.record_puts_skipped.fetch_add(1, std::memory_order_relaxed);
      return Status::kOk;
    }
    if (put(stored->lease)) return Status::kOk;
    // The lease is renewed and the record listed: only its stats are stale.
    if (stats_only) {
      metrics_.etcd_errors.fetch_add(1, std::memory_order_relaxed);
      return Status::kOk;
    }
    return io_error();
  }
  // Absent, without a lease, or its lease expired: a new lease. A same-key
  // race here (two heartbeats both finding no live lease) can briefly orphan
  // one lease, which expires on its own via the TTL.
  auto lid = etcd_.LeaseGrant(kTtlSeconds);
  if (!lid) return io_error();
  metrics_.lease_grants.fetch_add(1, std::memory_order_relaxed);
  return put(*lid) ? Status::kOk : io_error();
}

Status MdsServer::ListMembers(const std::string& group, std::string* out,
                              bool include_degraded) {
  metrics_.list_requests.fetch_add(1, std::memory_order_relaxed);
  // A malformed group would build a RangePrefix straddling other groups'
  // subtrees; reject it the same way Upsert does.
  if (!IsValidGroupOrId(group)) return Status::kInvalid;
  bool hit = false;
  const Status st = list_cache_.Get(
      group, include_degraded, out,
      [&](std::string* fresh) {
        return ListMembersFromEtcd(group, fresh, include_degraded);
      },
      &hit);
  if (hit) metrics_.list_cache_hits.fetch_add(1, std::memory_order_relaxed);
  return st;
}

Status ListCache::Get(const std::string& group, bool degraded,
                      std::string* out, const Compute& compute, bool* hit) {
  if (hit) *hit = false;
  if (ttl_ms_ == 0) return compute(out);
  const auto key = std::make_pair(group, degraded);
  std::unique_lock<std::mutex> lk(mu_);
  if (entries_.size() >= kMaxEntries && entries_.find(key) == entries_.end()) {
    const uint64_t now = clock_();
    for (auto it = entries_.begin(); it != entries_.end();) {
      const Entry& old = it->second;
      if (!old.loading && (!old.valid || now < old.at_ms ||
                           now - old.at_ms >= ttl_ms_))
        it = entries_.erase(it);
      else
        ++it;
    }
  }
  for (;;) {
    Entry& e = entries_[key];
    const uint64_t now = clock_();
    if (e.valid && now >= e.at_ms && now - e.at_ms < ttl_ms_) {
      *out = e.out;
      if (hit) *hit = true;
      return e.status;
    }
    if (!e.loading) break;
    cv_.wait(lk);
  }
  Entry& e = entries_[key];
  e.loading = true;
  const uint64_t generation = e.generation;
  const uint64_t started = clock_();
  lk.unlock();
  std::string fresh;
  const Status st = compute(&fresh);
  lk.lock();
  Entry& done = entries_[key];
  done.loading = false;
  if (done.generation == generation) {
    done.valid = true;
    done.at_ms = started;  // as old as the read it rests on
    done.status = st;
    done.out = fresh;
  }
  cv_.notify_all();
  *out = std::move(fresh);
  return st;
}

void ListCache::Invalidate(const std::string& group) {
  if (ttl_ms_ == 0) return;
  std::lock_guard<std::mutex> lk(mu_);
  for (const bool degraded : {false, true}) {
    const auto found = entries_.find(std::make_pair(group, degraded));
    if (found == entries_.end()) continue;
    found->second.valid = false;
    ++found->second.generation;
  }
}

std::string MdsServer::FencedKey(const std::string& group,
                                 const std::string& id) {
  return "/dfkv/v1/groups/" + group + "/fenced/" + id;
}

std::optional<std::vector<std::string>> MdsServer::FencedIdsOf(
    const std::string& group, bool fresh) {
  const uint64_t now = SteadyNowMs();
  if (!fresh) {
    std::lock_guard<std::mutex> lk(fenced_mu_);
    const auto found = fenced_cache_.find(group);
    if (found != fenced_cache_.end() && now >= found->second.at_ms &&
        now - found->second.at_ms < kFencedCacheMs)
      return found->second.ids;
  }
  const std::string prefix = "/dfkv/v1/groups/" + group + "/fenced/";
  const auto r = etcd_.RangePrefix(prefix);
  if (!r) return std::nullopt;
  std::vector<std::string> ids;
  for (const auto& kv : r->kvs) ids.push_back(kv.first.substr(prefix.size()));
  std::lock_guard<std::mutex> lk(fenced_mu_);
  if (fenced_cache_.size() >= kMaxFencedGroups &&
      fenced_cache_.find(group) == fenced_cache_.end()) {
    for (auto it = fenced_cache_.begin(); it != fenced_cache_.end();) {
      if (now < it->second.at_ms || now - it->second.at_ms >= kFencedCacheMs)
        it = fenced_cache_.erase(it);
      else
        ++it;
    }
  }
  fenced_cache_[group] = FencedIds{now, ids};
  return ids;
}

std::optional<bool> MdsServer::IsFenced(const std::string& group,
                                        const std::string& id) {
  const auto ids = FencedIdsOf(group, /*fresh=*/false);
  if (!ids) return std::nullopt;
  return std::find(ids->begin(), ids->end(), id) != ids->end();
}

Status MdsServer::Fence(const std::string& payload, bool fence) {
  if (!fencing_) return Status::kInvalid;
  const size_t nl = payload.find('\n');
  if (nl == std::string::npos) return Status::kInvalid;
  const std::string group = payload.substr(0, nl);
  const std::string id = payload.substr(nl + 1);
  if (!IsValidGroupOrId(group) || !IsValidGroupOrId(id)) return Status::kInvalid;
  if (fence) {
    // The fence first: a heartbeat arriving between the two writes is then
    // refused instead of writing the record back.
    if (!etcd_.Put(FencedKey(group, id), "fenced", 0) ||
        !etcd_.Delete(MemberKey(group, id))) {
      metrics_.etcd_errors.fetch_add(1, std::memory_order_relaxed);
      return Status::kIOError;
    }
    // Its heartbeats are refused now, not unwritten: nothing to wait for.
    unwritten_.Remove(MemberKey(group, id));
    DFKV_LOG_INFO("mds: fenced member group=" + group + " id=" + id);
  } else {
    if (!etcd_.Delete(FencedKey(group, id))) {
      metrics_.etcd_errors.fetch_add(1, std::memory_order_relaxed);
      return Status::kIOError;
    }
    DFKV_LOG_INFO("mds: unfenced member group=" + group + " id=" + id);
  }
  {
    std::lock_guard<std::mutex> lk(fenced_mu_);
    fenced_cache_.erase(group);
  }
  list_cache_.Invalidate(group);
  return Status::kOk;
}

Status MdsServer::ListFenced(const std::string& group, std::string* out) {
  if (!fencing_) return Status::kInvalid;
  if (!IsValidGroupOrId(group)) return Status::kInvalid;
  const auto ids = FencedIdsOf(group, /*fresh=*/true);
  if (!ids) {
    metrics_.etcd_errors.fetch_add(1, std::memory_order_relaxed);
    return Status::kIOError;
  }
  out->clear();
  for (const auto& id : *ids) {
    if (!out->empty()) *out += "\n";
    *out += id;
  }
  return Status::kOk;
}

Status MdsServer::ListMembersFromEtcd(const std::string& group,
                                      std::string* out,
                                      bool include_degraded) {
  std::string prefix = "/dfkv/v1/groups/" + group + "/members/";
  // Read before the range, so that a member written in between is judged
  // against a view that may predate its write: the list is then withheld for
  // one more poll instead of being served without that member.
  const std::vector<std::string> unwritten = unwritten_.Under(prefix);
  auto r = etcd_.RangePrefix(prefix);
  if (!r) { metrics_.etcd_errors.fetch_add(1, std::memory_order_relaxed); return Status::kIOError; }
  if (!unwritten.empty()) {
    // Members of this group heartbeat here but etcd did not take the write.
    // As long as etcd still lists all of them (their old lease is alive, or
    // another MDS wrote them) the view is complete. Once one is missing, it
    // expired because it could not be renewed, not because the node left: the
    // view is not the member list. Answer as for an unreadable etcd; a client
    // keeps its last ring on any failed list, including clients that predate
    // this check.
    std::vector<char> listed(unwritten.size(), 0);
    for (const auto& kv : r->kvs) {
      const auto it =
          std::lower_bound(unwritten.begin(), unwritten.end(), kv.first);
      if (it != unwritten.end() && *it == kv.first)
        listed[static_cast<size_t>(it - unwritten.begin())] = 1;
    }
    const auto first_missing = std::find(listed.begin(), listed.end(), 0);
    if (first_missing != listed.end()) {
      metrics_.lists_withheld.fetch_add(1, std::memory_order_relaxed);
      const uint64_t now_ms = SteadyNowMs();
      uint64_t last_ms = withheld_log_ms_.load(std::memory_order_relaxed);
      if ((last_ms == 0 || now_ms - last_ms >= 30000) &&
          withheld_log_ms_.compare_exchange_strong(last_ms, now_ms)) {
        DFKV_LOG_WARN(
            "mds: withholding the member list of group=" + group + ": " +
            std::to_string(unwritten.size()) +
            " member(s) heartbeat here but etcd did not take the write, and " +
            std::to_string(std::count(listed.begin(), listed.end(), 0)) +
            " of them (e.g. " +
            unwritten[static_cast<size_t>(first_missing - listed.begin())]
                .substr(prefix.size()) +
            ") are no longer in etcd. Clients keep their last ring until "
            "etcd accepts writes again (check its space quota and alarms).");
      }
      return Status::kIOError;
    }
  }
  // A record written back after its member was fenced (by an MDS without
  // fencing, or one whose fenced ids were a second old) is not listed.
  std::vector<std::string> fenced;
  if (fencing_) {
    const auto ids = FencedIdsOf(group, /*fresh=*/true);
    if (!ids) {
      metrics_.etcd_errors.fetch_add(1, std::memory_order_relaxed);
      return Status::kIOError;
    }
    fenced = *ids;
  }
  std::vector<MemberInfo> members;
  for (const auto& kv : r->kvs) {
    std::vector<MemberInfo> one;
    uint64_t e = 0;
    if (!DecodeMembers(kv.second.data(), kv.second.size(), &one, &e) ||
        one.size() != 1) {
      NoteUndecodable(kv.first);
      continue;
    }
    if (std::find(fenced.begin(), fenced.end(), one[0].id) != fenced.end())
      continue;
    if (include_degraded || one[0].RingEligible()) members.push_back(one[0]);
  }
  // Placement and topology have separate canonical generations. The legacy
  // kListMembers view is filtered to eligible members and keeps its placement
  // epoch. kListTopology includes degraded members and uses the HLT1-sensitive
  // generation, so rail-only changes reach modern clients without inventing a
  // second wire extension.
  metrics_.members_last_list.store(members.size(), std::memory_order_relaxed);
  const uint64_t epoch = include_degraded
                             ? MembersViewEpoch(members)
                             : MembersEpoch(members);
  *out = EncodeMembers(members, epoch);
  return Status::kOk;
}

Status MdsServer::Deregister(const std::string& group, const MemberInfo& m) {
  if (!IsValidGroupOrId(group) || !IsValidGroupOrId(m.id)) return Status::kInvalid;
  const std::string key = MemberKey(group, m.id);
  if (!etcd_.Delete(key)) {
    metrics_.etcd_errors.fetch_add(1, std::memory_order_relaxed);
    return Status::kIOError;
  }
  unwritten_.Remove(key);
  // The next poll reads the list without it.
  list_cache_.Invalidate(group);
  DFKV_LOG_INFO("mds: member " + m.id + " of group " + group +
                " deregistered (drained and stopping)");
  return Status::kOk;
}

// A member or client record that cannot be decoded is skipped; it is counted,
// and logged with its key at most once every 30 s.
void MdsServer::NoteUndecodable(const std::string& key) {
  metrics_.member_decode_errors.fetch_add(1, std::memory_order_relaxed);
  const uint64_t now = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
  uint64_t last = undecodable_logged_ms_.load(std::memory_order_relaxed);
  if (last != 0 && now - last < 30000) return;
  if (!undecodable_logged_ms_.compare_exchange_strong(last, now)) return;
  DFKV_LOG_WARN("mds: skipped a record in etcd that is not a member record: " +
                key + " (dfkv_mds_member_decode_errors_total counts them)");
}

Status MdsServer::ListClients(const std::string& group, std::string* out) {
  metrics_.client_list_requests.fetch_add(1, std::memory_order_relaxed);
  // Same path-traversal guard as ListMembers: a malformed group would build a
  // RangePrefix straddling other groups' client subtrees.
  if (!IsValidGroupOrId(group)) return Status::kInvalid;
  bool hit = false;
  const Status st = client_list_cache_.Get(
      group, false, out,
      [&](std::string* fresh) { return ListClientsFromEtcd(group, fresh); },
      &hit);
  if (hit)
    metrics_.client_list_cache_hits.fetch_add(1, std::memory_order_relaxed);
  return st;
}

uint64_t MdsServer::ListCacheMs() {
  return std::min<uint64_t>(
      config_dump::EnvU64("DFKV_MDS_LIST_CACHE_MS", kListCacheMs), 10000);
}

uint64_t MdsServer::ClientListCacheMs() {
  return std::min<uint64_t>(ListCacheMs(), kMaxClientListCacheMs);
}

Status MdsServer::ListClientsFromEtcd(const std::string& group,
                                      std::string* out) {
  std::string prefix = "/dfkv/v1/groups/" + group + "/clients/";
  auto r = etcd_.RangePrefix(prefix);
  if (!r) { metrics_.etcd_errors.fetch_add(1, std::memory_order_relaxed); return Status::kIOError; }
  std::vector<MemberInfo> clients;
  for (const auto& kv : r->kvs) {
    std::vector<MemberInfo> one;
    uint64_t e = 0;
    if (DecodeMembers(kv.second.data(), kv.second.size(), &one, &e) && one.size() == 1)
      clients.push_back(one[0]);
    else
      NoteUndecodable(kv.first);
  }
  // Clients carry no placement semantics, so the epoch is a content hash over the
  // identity set only — `dfkvctl clients` re-fetches every run, so a changing
  // epoch need not (and cannot) trigger any ring rebuild.
  metrics_.clients_last_list.store(clients.size(), std::memory_order_relaxed);
  *out = EncodeMembers(clients, MembersEpoch(clients));
  return Status::kOk;
}

Status MdsServer::ListGroups(std::string* out) {
  auto r = etcd_.RangePrefix("/dfkv/v1/groups/");
  if (!r) { metrics_.etcd_errors.fetch_add(1, std::memory_order_relaxed); return Status::kIOError; }
  // Key shape: /dfkv/v1/groups/<group>/members/<id> -- collect distinct <group>.
  std::string prev;
  std::string joined;
  for (const auto& kv : r->kvs) {
    const std::string& k = kv.first;
    const size_t base = sizeof("/dfkv/v1/groups/") - 1;
    if (k.size() <= base) continue;
    const size_t slash = k.find('/', base);
    if (slash == std::string::npos) continue;
    // A group of nothing but fence records (they carry no lease) is gone.
    if (k.compare(slash, 8, "/fenced/") == 0) continue;
    std::string g = k.substr(base, slash - base);
    if (g == prev) continue;  // keys arrive sorted: dedup against the previous
    prev = g;
    if (!joined.empty()) joined += '\n';
    joined += g;
  }
  *out = std::move(joined);
  return Status::kOk;
}

std::string MdsServer::ListenerMetricsText() const {
  std::string s;
  auto m = [&](const char* name, const char* type, const char* help,
               uint64_t v) {
    s += "# HELP "; s += name; s += " "; s += help; s += "\n";
    s += "# TYPE "; s += name; s += " "; s += type; s += "\n";
    s += name; s += " "; s += std::to_string(v); s += "\n";
  };
  m("dfkv_mds_max_conns", "gauge",
    "Connection cap in force (DFKV_MDS_MAX_CONNS fitted to the open-file limit)",
    max_conns_);
  m("dfkv_mds_accept_failures_total", "counter",
    "accept() failures from descriptor or memory exhaustion; the listener "
    "backed off after each",
    AcceptFailures());
  m("dfkv_mds_thread_start_failures_total", "counter",
    "Connections refused because their handler thread could not be created",
    ThreadStartFailures());
  return s;
}

std::string MdsServer::GroupMetricsText(bool* etcd_reachable) {
  if (etcd_reachable) *etcd_reachable = false;
  auto r = etcd_.RangePrefixForMetrics("/dfkv/v1/groups/");
  if (!r) {
    metrics_.etcd_errors.fetch_add(1, std::memory_order_relaxed);
    return "";
  }
  if (etcd_reachable) *etcd_reachable = true;
  struct Agg {
    uint64_t nodes = 0, degraded = 0, missing = 0, clients = 0;
    MemberStats sum;
    std::map<std::string, int> vers;  // version skew, from info "ver="
  };
  std::map<std::string, Agg> groups;
  const size_t base = sizeof("/dfkv/v1/groups/") - 1;
  for (const auto& kv : r->kvs) {
    const size_t slash = kv.first.find('/', base);
    if (kv.first.size() <= base || slash == std::string::npos) continue;
    // Fence records (DFKV_MDS_FENCING) are not members; a group of nothing
    // but them is not reported.
    if (kv.first.compare(slash, 8, "/fenced/") == 0) continue;
    Agg& a = groups[kv.first.substr(base, slash - base)];
    // Distinguish /members/<id> (placement ring) from /clients/<id> (consumers).
    // A client key carries identity only (no port/stats); it must NOT inflate the
    // ring node count, the capacity/hit aggregates, or the version-skew set —
    // otherwise a fleet of connectors would silently corrupt every group metric.
    const size_t seg2 = kv.first.find('/', slash + 1);
    const std::string kind = (seg2 == std::string::npos)
        ? std::string() : kv.first.substr(slash + 1, seg2 - slash - 1);
    if (kind == "clients") { a.clients++; continue; }
    std::vector<MemberInfo> one;
    uint64_t e = 0;
    if (!DecodeMembers(kv.second.data(), kv.second.size(), &one, &e) ||
        one.size() != 1)
      continue;
    const MemberInfo& m = one[0];
    if (!m.RingEligible()) a.degraded++;
    a.nodes++;
    // version skew from the static info string ("ver=X,...") -- the one place
    // we parse info at the MDS; everything numeric rides binary STA1.
    size_t vp = m.info.find("ver=");
    if (vp != std::string::npos) {
      size_t ve = m.info.find(',', vp);
      a.vers[m.info.substr(vp + 4, (ve == std::string::npos ? m.info.size() : ve) - vp - 4)]++;
    }
    if (!m.has_stats) { a.missing++; continue; }
    const MemberStats& st = m.stats;
    a.sum.capacity_bytes += st.capacity_bytes;
    a.sum.used_bytes += st.used_bytes;
    a.sum.objects += st.objects;
    a.sum.hits_total += st.hits_total;
    a.sum.misses_total += st.misses_total;
    a.sum.evictions_total += st.evictions_total;
    a.sum.puts_total += st.puts_total;
    a.sum.put_busy_total += st.put_busy_total;
    a.sum.dio_write_fallbacks += st.dio_write_fallbacks;
    a.sum.ram_used_bytes += st.ram_used_bytes;
    a.sum.ram_hits_total += st.ram_hits_total;
  }
  if (groups.empty()) return "";
  std::string s;
  auto emit = [&](const char* name, const char* help) {
    s += "# HELP "; s += name; s += " "; s += help; s += "\n";
    s += "# TYPE "; s += name; s += " gauge\n";
  };
  auto line = [&](const char* name, const std::string& g, uint64_t v) {
    s += name; s += "{group=\""; s += g; s += "\"} "; s += std::to_string(v); s += "\n";
  };
  // All ring aggregates are GAUGES (a node restart resets its counters, so the
  emit("dfkv_mds_group_ring_eligible_nodes",
       "Members currently eligible for placement");
  for (auto& [g, a] : groups)
    line("dfkv_mds_group_ring_eligible_nodes", g, a.nodes - a.degraded);
  emit("dfkv_mds_group_degraded_nodes",
       "Registered members excluded from placement by health");
  for (auto& [g, a] : groups)
    line("dfkv_mds_group_degraded_nodes", g, a.degraded);
  // per-group sum can go down; rate analysis belongs on the node-level series).
  emit("dfkv_mds_group_nodes", "Registered members per group");
  for (auto& [g, a] : groups) line("dfkv_mds_group_nodes", g, a.nodes);
  emit("dfkv_mds_group_capacity_bytes", "Sum of member capacities per group");
  for (auto& [g, a] : groups) line("dfkv_mds_group_capacity_bytes", g, a.sum.capacity_bytes);
  emit("dfkv_mds_group_used_bytes", "Sum of member used bytes per group (ring watermark)");
  for (auto& [g, a] : groups) line("dfkv_mds_group_used_bytes", g, a.sum.used_bytes);
  emit("dfkv_mds_group_objects", "Sum of resident objects per group");
  for (auto& [g, a] : groups) line("dfkv_mds_group_objects", g, a.sum.objects);
  emit("dfkv_mds_group_hits_sum", "Sum of member hit counters (resets with node restarts)");
  for (auto& [g, a] : groups) line("dfkv_mds_group_hits_sum", g, a.sum.hits_total);
  emit("dfkv_mds_group_misses_sum", "Sum of member miss counters");
  for (auto& [g, a] : groups) line("dfkv_mds_group_misses_sum", g, a.sum.misses_total);
  emit("dfkv_mds_group_evictions_sum", "Sum of member eviction counters");
  for (auto& [g, a] : groups) line("dfkv_mds_group_evictions_sum", g, a.sum.evictions_total);
  emit("dfkv_mds_group_puts_sum", "Sum of member put counters");
  for (auto& [g, a] : groups) line("dfkv_mds_group_puts_sum", g, a.sum.puts_total);
  emit("dfkv_mds_group_put_busy_sum", "Sum of admission-gate rejections");
  for (auto& [g, a] : groups) line("dfkv_mds_group_put_busy_sum", g, a.sum.put_busy_total);
  emit("dfkv_mds_group_dio_fallbacks_sum", "Sum of direct-mode buffered fallbacks (page-cache alarm)");
  for (auto& [g, a] : groups) line("dfkv_mds_group_dio_fallbacks_sum", g, a.sum.dio_write_fallbacks);
  emit("dfkv_mds_group_ram_used_bytes", "Sum of RAM-tier resident bytes per group");
  for (auto& [g, a] : groups) line("dfkv_mds_group_ram_used_bytes", g, a.sum.ram_used_bytes);
  emit("dfkv_mds_group_ram_hits_sum", "Sum of RAM-tier hit counters");
  for (auto& [g, a] : groups) line("dfkv_mds_group_ram_hits_sum", g, a.sum.ram_hits_total);
  emit("dfkv_mds_group_stats_missing", "Members without STA1 stats (pre-upgrade nodes)");
  for (auto& [g, a] : groups) line("dfkv_mds_group_stats_missing", g, a.missing);
  emit("dfkv_mds_group_version_skew", "Distinct member versions per group (>1 = drift)");
  for (auto& [g, a] : groups) line("dfkv_mds_group_version_skew", g, a.vers.size());
  // Registered cache consumers (inference connector instances) per group. A
  // gauge under the lease TTL: a connector that dies stops heartbeating and its
  // /clients/<id> key expires out of etcd within ~kTtlSeconds.
  emit("dfkv_mds_group_clients", "Registered client (consumer) instances per group");
  for (auto& [g, a] : groups) line("dfkv_mds_group_clients", g, a.clients);
  return s;
}

void MdsServer::Handle(int fd, long io_timeout_s,
                       std::chrono::steady_clock::time_point first_frame_deadline) {
  const timeval base_rcvtmo{io_timeout_s, 0};
  // While armed, both reads of the FIRST frame (prefix + payload) run through
  // net::ReadAllWithin, which re-arms SO_RCVTIMEO to min(base, remaining)
  // before every recv — so the deadline is absolute and a byte-at-a-time
  // dribble dies on it instead of surviving under a fresh per-syscall budget.
  bool arm_deadline =
      first_frame_deadline != std::chrono::steady_clock::time_point::max();
  while (running_) {
    // Control-plane epoch discrimination for mixed-generation rollouts. The
    // leading byte is the deterministic discriminator: native v2 peers write
    // a 50-byte prefix, legacy v1.x peers a 42-byte one. Unknown epochs still
    // fail closed — the deliberate fast-reject version gate is widened by one
    // known epoch (via DFKV_MDS_ACCEPT_LEGACY), not removed. Data-plane
    // listeners stay strictly epoch-6/7; this shim exists only in the MDS.
    char prefix[kReqPrefix];
    // Epoch discriminator: read the leading byte first (v2 = 50-byte prefix,
    // legacy = 42-byte one). While armed it runs under the same absolute
    // first-frame deadline as every other read of the first frame.
    const bool got_discriminator =
        arm_deadline
            ? net::ReadAllWithin(fd, prefix, 1, base_rcvtmo,
                                 first_frame_deadline)
            : net::ReadAll(fd, prefix, 1);
    if (!got_discriminator) return;
    const uint8_t epoch = static_cast<uint8_t>(prefix[0]);
    bool legacy = false;
    ReqFields rq;
    // MDS frames carry a group name + one MemberInfo (<1 KiB); cap the
    // declared length so a forged prefix can't trigger a giant allocation.
    if (epoch == kNativeProtoTcp) {
      const bool got_prefix =
          arm_deadline
              ? net::ReadAllWithin(fd, prefix + 1, kReqPrefix - 1, base_rcvtmo,
                                   first_frame_deadline)
              : net::ReadAll(fd, prefix + 1, kReqPrefix - 1);
      if (!got_prefix ||
          !DecodeReq(prefix, &rq, wire_limits::kMdsMaxReqPayload)) {
        return;
      }
    } else {
      if (!accept_legacy_ || epoch != kNativeProtoLegacyTcp) return;
      const bool got_prefix =
          arm_deadline
              ? net::ReadAllWithin(fd, prefix + 1, kLegacyReqPrefix - 1,
                                   base_rcvtmo, first_frame_deadline)
              : net::ReadAll(fd, prefix + 1, kLegacyReqPrefix - 1);
      if (!got_prefix ||
          !DecodeLegacyReq(prefix, &rq, wire_limits::kMdsMaxReqPayload)) {
        return;
      }
      legacy = true;
      metrics_.legacy_frames.fetch_add(1, std::memory_order_relaxed);
    }
    std::vector<char> payload(rq.payload_len);
    if (rq.payload_len) {
      const bool got_payload =
          arm_deadline
              ? net::ReadAllWithin(fd, payload.data(), rq.payload_len,
                                   base_rcvtmo, first_frame_deadline)
              : net::ReadAll(fd, payload.data(), rq.payload_len);
      if (!got_payload) return;
    }
    if (arm_deadline) {
      // First complete frame received: end the deadline regime. ReadAllWithin
      // shrank SO_RCVTIMEO toward the deadline, so restore the base per-syscall
      // budget for all steady-state reads.
      arm_deadline = false;
      ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &base_rcvtmo,
                   sizeof(base_rcvtmo));
    }

    std::string data;
    Status st = Status::kInvalid;
    WireOp op = static_cast<WireOp>(rq.op);
    if (op == WireOp::kRegister || op == WireOp::kHeartbeat) {
      if (op == WireOp::kRegister)
        metrics_.register_requests.fetch_add(1, std::memory_order_relaxed);
      else
        metrics_.keepalives.fetch_add(1, std::memory_order_relaxed);
      std::string group;
      MemberInfo m;
      if (DecodeMemberReq(payload.data(), rq.payload_len, &group, &m))
        st = Upsert(group, m, op == WireOp::kHeartbeat);
    } else if (op == WireOp::kDeregister) {
      metrics_.deregister_requests.fetch_add(1, std::memory_order_relaxed);
      std::string group;
      MemberInfo m;
      if (DecodeMemberReq(payload.data(), rq.payload_len, &group, &m))
        st = Deregister(group, m);
    } else if (op == WireOp::kClientRegister || op == WireOp::kClientHeartbeat) {
      if (op == WireOp::kClientRegister)
        metrics_.client_register_requests.fetch_add(1, std::memory_order_relaxed);
      else
        metrics_.client_keepalives.fetch_add(1, std::memory_order_relaxed);
      // Same payload (group + MemberInfo) as member registration; only the etcd
      // prefix differs. A client's ip/port/weight are identity-only and carry no
      // placement meaning (UpsertClient writes /clients/<id>, never the ring).
      std::string group;
      MemberInfo m;
      if (DecodeMemberReq(payload.data(), rq.payload_len, &group, &m))
        st = UpsertClient(group, m);
    } else if (op == WireOp::kListMembers ||
               op == WireOp::kListTopology) {
      std::string group(payload.data(), rq.payload_len);
      st = ListMembers(group, &data, op == WireOp::kListTopology);
    } else if (op == WireOp::kListClients) {
      std::string group(payload.data(), rq.payload_len);
      st = ListClients(group, &data);
    } else if (op == WireOp::kListGroups) {
      st = ListGroups(&data);
    } else if (op == WireOp::kFenceMember || op == WireOp::kUnfenceMember) {
      st = Fence(std::string(payload.data(), rq.payload_len),
                 op == WireOp::kFenceMember);
    } else if (op == WireOp::kListFenced) {
      st = ListFenced(std::string(payload.data(), rq.payload_len), &data);
    }

    char rp[kRespPrefix];
    size_t rp_len = kRespPrefix;
    if (legacy) {
      // v1.x peers expect a 10-byte prefix and the v1 status byte order
      // (kIOError=3); the response data itself is generation-neutral.
      EncodeLegacyResp(rp, st, data.size());
      rp_len = kLegacyRespPrefix;
    } else {
      EncodeResp(rp, st, data.size());
    }
    if (!net::WriteAll(fd, rp, rp_len)) return;
    if (!data.empty() && !net::WriteAll(fd, data.data(), data.size())) return;
  }
}

}  // namespace dfkv
