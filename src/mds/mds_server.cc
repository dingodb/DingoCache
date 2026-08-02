#include "common/config_dump.h"
#include "mds/mds_server.h"

#include <cstdlib>
#include <chrono>
#include <map>

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include "common/membership.h"
#include "mds/mds_proto.h"
#include "utils/net_util.h"
#include "utils/thread_name.h"
#include "utils/wire_limits.h"
#include "transport/wire.h"

namespace dfkv {

namespace {
uint64_t SteadyNowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
}  // namespace

bool LocalLeaseMap::LookupAndTouch(const std::string& key, uint64_t now_ms,
                                   int64_t* lease_id) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = entries_.find(key);
  if (it == entries_.end()) return false;
  it->second.last_use_ms = now_ms;
  *lease_id = it->second.lease_id;
  return true;
}

void LocalLeaseMap::Store(const std::string& key, int64_t lease_id,
                          uint64_t now_ms) {
  std::lock_guard<std::mutex> lk(mu_);
  entries_[key] = Entry{lease_id, now_ms};
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
  // DFKV_MDS_IO_TIMEOUT_S tunes it (tests use 1); DFKV_MDS_MAX_CONNS caps the
  // handler-thread count so a connect flood degrades to refused conns instead
  // of unbounded threads.
  long tmo_s = 60;
  if (const char* v = std::getenv("DFKV_MDS_IO_TIMEOUT_S")) {
    const long t = std::atol(v);
    if (t > 0) tmo_s = t;
  }
  size_t max_conns = 4096;
  if (const char* v = std::getenv("DFKV_MDS_MAX_CONNS")) {
    const long m = std::atol(v);
    if (m > 0) max_conns = static_cast<size_t>(m);
  }
  // Recorded from the accept thread's prologue (runs at Start, before main
  // reaches Emit after its network etcd probe) so the dump shows these defaults.
  config_dump::RecordResolved("DFKV_MDS_IO_TIMEOUT_S", std::to_string(tmo_s));
  config_dump::RecordResolved("DFKV_MDS_MAX_CONNS", std::to_string(max_conns));
  while (running_) {
    int fd = ::accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) { if (!running_) break; continue; }
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    timeval tv{tmo_s, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    std::lock_guard<std::mutex> lk(conn_mu_);
    if (!running_) { ::close(fd); break; }
    ReapDoneLocked();  // join handlers that finished since the last accept
    if (conns_.size() >= max_conns) { ::close(fd); continue; }
    conn_fds_.push_back(fd);
    auto done = std::make_shared<std::atomic<bool>>(false);
    conns_.push_back({std::thread([this, fd, done] {
                        NameThisThread("mds-conn");
                        Handle(fd);
                        {
                          std::lock_guard<std::mutex> lk(conn_mu_);
                          for (auto it = conn_fds_.begin(); it != conn_fds_.end(); ++it)
                            if (*it == fd) { conn_fds_.erase(it); break; }
                        }
                        ::close(fd);
                        done->store(true, std::memory_order_release);  // last act
                      }),
                      done});
  }
}

Status MdsServer::Upsert(const std::string& group, const MemberInfo& m) {
  // Reject before the id/group reach the etcd key: an unrestricted token can
  // contain '/' and escape its own key subtree (phantom-member injection).
  if (!IsValidGroupOrId(group) || !IsValidGroupOrId(m.id)) return Status::kInvalid;
  return UpsertLeased(MemberKey(group, m.id), m, leases_, false);
}

Status MdsServer::UpsertClient(const std::string& group, const MemberInfo& m) {
  // Same path-traversal guard as Upsert; clients live under a parallel prefix
  // (/clients/<id>) so a malformed id can't escape into /members/ either.
  if (!IsValidGroupOrId(group) || !IsValidGroupOrId(m.id)) return Status::kInvalid;
  return UpsertLeased(ClientKey(group, m.id), m, client_leases_, true);
}

// Shared lease keepalive + re-Put for both members and clients. Local entries
// are discarded after four etcd TTLs without local use. That is deliberately
// conservative for rotating multi-MDS registrars, while bounding process memory
// under ID churn. etcd remains authoritative; a cache miss grants and Puts.
Status MdsServer::UpsertLeased(const std::string& key, const MemberInfo& m,
                               LocalLeaseMap& leases, bool client) {
  std::string val = EncodeMembers({m}, 0);
  // Look up this member's known lease under the lock, then do the BLOCKING etcd
  // calls OUTSIDE it. Previously lease_mu_ was held across LeaseKeepAlive/Grant/Put,
  // so every member's heartbeat serialized on one mutex held across network I/O —
  // a slow etcd stalled all members' liveness. Now heartbeats for distinct members
  // run in parallel; the lock only guards the in-memory {key->leaseID} map.
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
  int64_t existing = 0;
  const bool have = leases.LookupAndTouch(key, now_ms, &existing);
  // Fast path: refresh the lease AND re-write the key under it. The Put is
  // load-bearing, not redundant: with several MDS instances behind a rotating
  // registrar, a slow path on ANOTHER MDS re-attaches the key to that MDS's
  // lease. From then on a value-less keepalive here would refresh a lease that
  // no longer holds the key, while the key's actual lease only gets refreshed
  // once per full rotation (N_mds * heartbeat) — at 3 MDS * 10s hb that equals
  // the 30s TTL, so the member can expire out of etcd while every heartbeat
  // still returns kOk. Re-putting under our own live lease restores the
  // invariant "any MDS that answers a heartbeat guarantees the key survives".
  // It also propagates MemberInfo changes (ip/weight after a reconfigure)
  // that the old refresh silently dropped. Cost: one small etcd write per
  // heartbeat; the epoch is a content hash, so unchanged values don't make
  // clients rebuild their ring.
  if (have && etcd_.LeaseKeepAlive(existing) && etcd_.Put(key, val, existing))
    return Status::kOk;
  // Slow path: (re)grant a lease and (re)write the member key. A same-key race
  // here (two heartbeats both missing the lease) can briefly orphan one lease,
  // which expires on its own via the TTL; same-key heartbeats are serial in
  // practice (one node, one connection), so this is rare and harmless.
  auto lid = etcd_.LeaseGrant(kTtlSeconds);
  if (!lid) { metrics_.etcd_errors.fetch_add(1, std::memory_order_relaxed); return Status::kIOError; }
  metrics_.lease_grants.fetch_add(1, std::memory_order_relaxed);
  if (!etcd_.Put(key, val, *lid)) { metrics_.etcd_errors.fetch_add(1, std::memory_order_relaxed); return Status::kIOError; }
  leases.Store(key, *lid, SteadyNowMs());
  current_metric.store(leases.Size(), std::memory_order_relaxed);
  return Status::kOk;
}

Status MdsServer::ListMembers(const std::string& group, std::string* out) {
  metrics_.list_requests.fetch_add(1, std::memory_order_relaxed);
  // A malformed group would build a RangePrefix straddling other groups'
  // subtrees; reject it the same way Upsert does.
  if (!IsValidGroupOrId(group)) return Status::kInvalid;
  std::string prefix = "/dfkv/v1/groups/" + group + "/members/";
  auto r = etcd_.RangePrefix(prefix);
  if (!r) { metrics_.etcd_errors.fetch_add(1, std::memory_order_relaxed); return Status::kIOError; }
  std::vector<MemberInfo> members;
  for (const auto& kv : r->kvs) {
    std::vector<MemberInfo> one;
    uint64_t e = 0;
    if (DecodeMembers(kv.second.data(), kv.second.size(), &one, &e) && one.size() == 1)
      members.push_back(one[0]);
  }
  // Epoch = content hash of the member set, NOT etcd's global revision: the
  // revision bumps on every cluster write (including unrelated groups), which
  // would make clients rebuild their ring needlessly. The hash changes iff THIS
  // group's membership content changes.
  metrics_.members_last_list.store(members.size(), std::memory_order_relaxed);
  *out = EncodeMembers(members, MembersEpoch(members));
  return Status::kOk;
}

Status MdsServer::ListClients(const std::string& group, std::string* out) {
  metrics_.client_list_requests.fetch_add(1, std::memory_order_relaxed);
  // Same path-traversal guard as ListMembers: a malformed group would build a
  // RangePrefix straddling other groups' client subtrees.
  if (!IsValidGroupOrId(group)) return Status::kInvalid;
  std::string prefix = "/dfkv/v1/groups/" + group + "/clients/";
  auto r = etcd_.RangePrefix(prefix);
  if (!r) { metrics_.etcd_errors.fetch_add(1, std::memory_order_relaxed); return Status::kIOError; }
  std::vector<MemberInfo> clients;
  for (const auto& kv : r->kvs) {
    std::vector<MemberInfo> one;
    uint64_t e = 0;
    if (DecodeMembers(kv.second.data(), kv.second.size(), &one, &e) && one.size() == 1)
      clients.push_back(one[0]);
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
    std::string g = k.substr(base, slash - base);
    if (g == prev) continue;  // keys arrive sorted: dedup against the previous
    prev = g;
    if (!joined.empty()) joined += '\n';
    joined += g;
  }
  *out = std::move(joined);
  return Status::kOk;
}

std::string MdsServer::GroupMetricsText() {
  auto r = etcd_.RangePrefix("/dfkv/v1/groups/");
  if (!r) { metrics_.etcd_errors.fetch_add(1, std::memory_order_relaxed); return ""; }
  struct Agg {
    uint64_t nodes = 0, missing = 0, clients = 0;
    MemberStats sum;
    std::map<std::string, int> vers;  // version skew, from info "ver="
  };
  std::map<std::string, Agg> groups;
  const size_t base = sizeof("/dfkv/v1/groups/") - 1;
  for (const auto& kv : r->kvs) {
    const size_t slash = kv.first.find('/', base);
    if (kv.first.size() <= base || slash == std::string::npos) continue;
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
    if (!DecodeMembers(kv.second.data(), kv.second.size(), &one, &e) || one.size() != 1)
      continue;
    const MemberInfo& m = one[0];
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

void MdsServer::Handle(int fd) {
  while (running_) {
    char prefix[kReqPrefix];
    if (!net::ReadAll(fd, prefix, kReqPrefix)) return;
    ReqFields rq;
    // MDS frames carry a group name + one MemberInfo (<1 KiB); cap the
    // declared length so a forged prefix can't trigger a giant allocation.
    if (!DecodeReq(prefix, &rq, wire_limits::kMdsMaxReqPayload)) return;
    std::vector<char> payload(rq.payload_len);
    if (rq.payload_len && !net::ReadAll(fd, payload.data(), rq.payload_len)) return;

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
        st = Upsert(group, m);
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
    } else if (op == WireOp::kListMembers) {
      std::string group(payload.data(), rq.payload_len);
      st = ListMembers(group, &data);
    } else if (op == WireOp::kListClients) {
      std::string group(payload.data(), rq.payload_len);
      st = ListClients(group, &data);
    } else if (op == WireOp::kListGroups) {
      st = ListGroups(&data);
    }

    char rp[kRespPrefix];
    EncodeResp(rp, st, data.size());
    if (!net::WriteAll(fd, rp, kRespPrefix)) return;
    if (!data.empty() && !net::WriteAll(fd, data.data(), data.size())) return;
  }
}

}  // namespace dfkv
