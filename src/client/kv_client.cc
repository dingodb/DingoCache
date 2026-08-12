#include "client/kv_client.h"

#include <array>
#include <atomic>
#include <climits>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include "client/group_shard.h"
#include "client/key_map.h"
#include "client/read_scheduler.h"
#include "utils/log.h"
#include "utils/thread_name.h"
#include "utils/parallel_for.h"
#include "transport/transport_factory.h"
#include "common/config_dump.h"

namespace dfkv {

namespace {
// Summarize a membership change as "+A -R (added: ...; removed: ...)", comparing
// new node ids against the currently-adopted ones. Empty when there is no id
// churn (e.g. only an address/weight changed). Ids only; small lists spelled out
// so an operator sees exactly which node joined or left.
std::string MembershipDelta(const std::map<std::string, std::string>& old_m,
                            const std::map<std::string, std::string>& new_m) {
  std::vector<std::string> added, removed;
  for (const auto& kv : new_m) if (!old_m.count(kv.first)) added.push_back(kv.first);
  for (const auto& kv : old_m) if (!new_m.count(kv.first)) removed.push_back(kv.first);
  if (added.empty() && removed.empty()) return {};
  auto join = [](const std::vector<std::string>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) { if (i) s += ","; s += v[i]; }
    return s;
  };
  std::string s = "+" + std::to_string(added.size()) + " -" + std::to_string(removed.size());
  if (!added.empty()) s += " (added: " + join(added) + ")";
  if (!removed.empty()) s += " (removed: " + join(removed) + ")";
  return s;
}

// Aggregate SG byte counts are checked once before routing. Descriptor count is
// intentionally unbounded at this layer: RDMA windows it to the negotiated HCA
// width, while framed transports preserve their existing logical SG fallback.

bool CheckedSizeSum(const std::vector<size_t>& values, size_t* total) {
  size_t sum = 0;
  for (size_t value : values) {
    size_t next = 0;
    if (!CheckedAddSize(sum, value, &next)) {
      *total = 0;
      return false;
    }
    sum = next;
  }
  *total = sum;
  return true;
}

void AddMetricBytes(size_t value, uint64_t* total) {
  if constexpr (sizeof(size_t) > sizeof(uint64_t)) {
    if (value > std::numeric_limits<uint64_t>::max()) {
      *total = std::numeric_limits<uint64_t>::max();
      return;
    }
  }
  const uint64_t bytes = static_cast<uint64_t>(value);
  if (bytes > std::numeric_limits<uint64_t>::max() - *total)
    *total = std::numeric_limits<uint64_t>::max();
  else
    *total += bytes;
}

size_t EnvSize(const char* name, size_t dflt) {
  const char* v = std::getenv(name);
  size_t out = dflt;
  bool from_env = false;
  if (v && *v) {
    char* end = nullptr;
    unsigned long long x = std::strtoull(v, &end, 10);
    if (end != v && x > 0) { out = static_cast<size_t>(x); from_env = true; }
  }
  config_dump::Record(name, std::to_string(out),
                      from_env ? config_dump::Source::kEnv
                               : config_dump::Source::kDefault);
  return out;
}

int EnvNonnegativeInt(const char* name, int dflt) {
  const char* text = std::getenv(name);
  int out = dflt;
  bool from_env = false;
  if (text && *text) {
    unsigned int value = 0;
    bool valid = true;
    for (const unsigned char* p =
             reinterpret_cast<const unsigned char*>(text);
         *p; ++p) {
      if (*p < '0' || *p > '9') {
        valid = false;
        break;
      }
      const unsigned int digit = *p - '0';
      if (value > (static_cast<unsigned int>(INT_MAX) - digit) / 10) {
        valid = false;
        break;
      }
      value = value * 10 + digit;
    }
    if (valid) {
      out = static_cast<int>(value);
      from_env = true;
    }
  }
  config_dump::Record(name, std::to_string(out),
                      from_env ? config_dump::Source::kEnv
                               : config_dump::Source::kDefault);
  return out;
}

// Env-reading wrapper over ShardGroups (client/group_shard.h): resolve the
// DFKV_READ_SHARD_KEYS / DFKV_READ_MAX_CONNS knobs once (recorded into the config
// dump via EnvSize) and split each per-node group into up to DFKV_READ_MAX_CONNS
// shards so multiple connections drive one node concurrently — the mechanism that
// breaks the single-connection ~166 MB/s/conn key-by-key drain on a single- or
// few-node ring. Used by both the read (BatchGet / SG GET) and write (BatchPutSg)
// batch paths; the same two knobs govern read and write sharding.
template <class GroupVec>
GroupVec ShardReadGroups(GroupVec groups) {
  static const size_t target = EnvSize("DFKV_READ_SHARD_KEYS", 16);
  static const size_t maxc = EnvSize("DFKV_READ_MAX_CONNS", 8);
  return ShardGroups(std::move(groups), target, maxc);
}
}  // namespace

KVClient::KVClient(std::vector<std::pair<std::string, std::string>> members,
                   std::string key_namespace, Transport* transport,
                   std::optional<size_t> batch_concurrency_override)
    : key_namespace_(std::move(key_namespace)),
      namespace_hash_(ToBlockKey(key_namespace_, "").digest_hi),
      health_(EnvSize("DFKV_PEER_COOLDOWN_MS", 2000),
              EnvSize("DFKV_PEER_COOLDOWN_MAX_MS", 30000),
              EnvSize("DFKV_PEER_BAD_GRACE_MS", 1000)) {
  SetMembers(std::move(members));
  if (transport) {
    t_ = transport;
    transport_reason_ = "injected";
  } else {
    std::string reason;
    owned_ = MakeClientTransport(&reason);  // RDMA if available+requested, else TCP
    transport_reason_ = reason.empty() ? std::string("unknown") : reason;
    if (!owned_) {
      DFKV_LOG_ERROR("dfkv client transport unavailable: " + transport_reason_);
      throw std::runtime_error("dfkv client transport unavailable: " + transport_reason_);
    }
    t_ = owned_.get();
    DFKV_LOG_INFO("dfkv client transport=" + transport_reason_);
  }
  // Static member lists adopt their ring above, before t_ exists, and never
  // pass through the MDS poller again — replay the scale hint once so the
  // transport budget is sized for them too.
  {
    size_t adopted = 0;
    {
      std::lock_guard<std::mutex> lk(ring_mu_);
      adopted = addr_.size();
    }
    if (adopted > 0) t_->OnTopologyHint(adopted);
  }
  // Same-host GET rendezvous. The namespace digest prevents independent
  // model/raw-layout domains from sharing process-local entries.
  dedup_ = NodeDedup::FromEnv(namespace_hash_);
  get_miss_retries_ = EnvNonnegativeInt("DFKV_GET_MISS_RETRIES", 1);
  // An ABI descriptor value (including 0=auto) supersedes the environment.
  if (batch_concurrency_override) {
    batch_concurrency_.store(*batch_concurrency_override,
                             std::memory_order_relaxed);
    config_dump::Record("DFKV_BATCH_CONCURRENCY",
                        std::to_string(*batch_concurrency_override),
                        config_dump::Source::kArg);
  } else if (const char* bc = std::getenv("DFKV_BATCH_CONCURRENCY")) {
    char* end = nullptr;
    const unsigned long long x = std::strtoull(bc, &end, 10);
    if (end != bc && *end == '\0' && x > 0 &&
        x <= std::numeric_limits<size_t>::max())
      batch_concurrency_.store(static_cast<size_t>(x),
                               std::memory_order_relaxed);
  }
  // Opt-in active per-peer latency probe. Off unless DFKV_PROBE_INTERVAL_MS>0,
  // so default behavior is unchanged (no background traffic).
  int probe_ms = 0;
  if (const char* pe = std::getenv("DFKV_PROBE_INTERVAL_MS")) probe_ms = std::atoi(pe);
  if (probe_ms > 0) StartProbe(probe_ms);
  // Record the resolved client env knobs (defaults included). FanoutPool's is a
  // lazy static, so force it now rather than wait for the first fan-out.
  {
    namespace cd = dfkv::config_dump;
    (void)ParallelExecutor::MaxThreads();  // records DFKV_FANOUT_THREADS
    if (!batch_concurrency_override)
      cd::RecordResolved(
          "DFKV_BATCH_CONCURRENCY",
          std::to_string(batch_concurrency_.load(std::memory_order_relaxed)));
    cd::RecordResolved("DFKV_PROBE_INTERVAL_MS", std::to_string(probe_ms));
    cd::RecordResolved("DFKV_CLIENT_NODE_DEDUP", dedup_ ? "on" : "off");
  }
  // Log only the namespace digest; raw model/tenant names may be sensitive.
  static std::once_flag dump_once;
  std::call_once(dump_once, [this] {
    namespace cd = dfkv::config_dump;
    cd::Record("key_namespace_digest",
               ToBlockKey(key_namespace_, "").DigestHex(), cd::Source::kArg);
    cd::Record("transport", transport_reason_, cd::Source::kArg);
    cd::Emit("client");
  });
}

void KVClient::AdoptRing(ConHash ring, std::map<std::string, std::string> addr) {
  const size_t count = addr.size();
  std::string delta;
  {
    std::lock_guard<std::mutex> lk(ring_mu_);
    delta = MembershipDelta(addr_, addr);  // vs the previously-adopted view
    ring_ = std::move(ring);
    addr_ = std::move(addr);
  }
  // A membership change is rare (epoch-gated) and worth one line: its presence
  // confirms discovery populated the ring; its ABSENCE at startup is the fastest
  // tell that the ring is empty and every op is silently missing. The delta
  // (added/removed node ids) makes a scale-up or a node loss obvious at a glance.
  if (count == 0)
    DFKV_LOG_WARN("ring: adopted EMPTY membership (0 nodes) — puts/gets route to nowhere (ok=0)");
  else if (delta.empty())
    DFKV_LOG_INFO("ring: " + std::to_string(count) + " member(s) (unchanged)");
  else
    DFKV_LOG_INFO("ring: " + std::to_string(count) + " member(s) " + delta);
  // Budget demand follows the ring (nodes x rails), so every adoption feeds
  // the transport's scale hint. t_ is null for the constructor's initial
  // SetMembers — the ctor replays the hint once the transport exists.
  if (count > 0 && t_ != nullptr) t_->OnTopologyHint(count);
  if (count > 0) ring_cv_.notify_all();
}

void KVClient::SetMembers(std::vector<std::pair<std::string, std::string>> members) {
  ConHash ring;
  std::map<std::string, std::string> addr;
  for (auto& [name, a] : members) {
    ring.AddNode(name);
    addr[name] = a;
  }
  ring.Build();
  AdoptRing(std::move(ring), std::move(addr));
}

void KVClient::SetMembers(const std::vector<MemberInfo>& members) {
  ConHash ring;
  std::map<std::string, std::string> addr;
  for (const auto& m : members) {
    ring.AddNode(m.id, m.weight < 1 ? 1 : static_cast<int>(m.weight));
    addr[m.id] = m.ip + ":" + std::to_string(m.port);
  }
  ring.Build();
  AdoptRing(std::move(ring), std::move(addr));
}

void KVClient::StartMdsDiscovery(std::vector<std::string> mds_eps,
                                 const std::string& group, int poll_ms) {
  StopMdsDiscovery();
  poller_ = std::make_unique<MdsMemberPoller>(
      std::move(mds_eps), group,
      [this](const std::vector<MemberInfo>& ms) { SetMembers(ms); }, poll_ms);
  poller_->Start();
}

bool KVClient::WaitForRing(int timeout_ms) {
  std::unique_lock<std::mutex> lk(ring_mu_);
  if (!addr_.empty()) return true;
  ring_cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                    [this] { return !addr_.empty(); });
  return !addr_.empty();
}

void KVClient::StopMdsDiscovery() {
  if (poller_) { poller_->Stop(); poller_.reset(); }
}

void KVClient::StartClientRegistration(std::vector<std::string> mds_eps,
                                       const std::string& group, MemberInfo self,
                                       int heartbeat_ms) {
  StopClientRegistration();
  // is_client=true routes the registrar to kClientRegister/kClientHeartbeat,
  // which the MDS writes under /clients/<id> (never the placement ring). No
  // stats provider: a consumer has no ring-level stats to report.
  client_registrar_ = std::make_unique<MdsRegistrar>(
      std::move(mds_eps), group, self, heartbeat_ms,
      MdsRegistrar::kDefaultIoTimeoutMs,
      /*is_client=*/true);
  client_registrar_->Start();
}

void KVClient::StopClientRegistration() {
  if (client_registrar_) { client_registrar_->Stop(); client_registrar_.reset(); }
}

KVClient::~KVClient() { StopProbe(); StopClientRegistration(); StopMdsDiscovery(); }

void KVClient::StartProbe(int interval_ms) {
  if (interval_ms <= 0 || probe_running_.load(std::memory_order_relaxed)) return;
  probe_interval_ms_ = interval_ms;
  probe_running_.store(true, std::memory_order_relaxed);
  probe_th_ = std::thread([this] { NameThisThread("kv-probe"); ProbeLoop(); });
}

void KVClient::StopProbe() {
  if (!probe_running_.exchange(false, std::memory_order_relaxed)) return;
  probe_cv_.notify_all();
  if (probe_th_.joinable()) probe_th_.join();
}

void KVClient::ProbeLoop() {
  // A sentinel key that no real workload writes; kExist of it is a cheap round
  // trip that measures the node's responsiveness (kNotFound is a healthy reply).
  const BlockKey probe = ToBlockKey(key_namespace_, "__dfkv_probe__");
  while (probe_running_.load(std::memory_order_relaxed)) {
    std::vector<std::string> addrs;
    {
      std::lock_guard<std::mutex> lk(ring_mu_);
      addrs.reserve(addr_.size());
      for (const auto& [name, a] : addr_) addrs.push_back(a);
    }
    for (const auto& a : addrs) {
      if (!probe_running_.load(std::memory_order_relaxed)) break;
      bool e = false;
      auto t0 = std::chrono::steady_clock::now();
      Status st = t_->Exist(a, probe, &e);
      double sec = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t0).count();
      // Only record real round trips; a transport IO error (e.g. connect refused)
      // is fast and would skew latency, and is already counted by PeerHealth.
      // A non-IO reply also means the peer is reachable: clear its data-path
      // cooldown here (off the data path) so recovery never depends on a data
      // request re-dialing a peer that is still inside its backed-off cooldown.
      // kResourceExhausted never dialed the peer at all: its wall time is a
      // local budget wait (up to RESOURCE_ACQUIRE_MS) that would poison the
      // latency EWMA, and it proves nothing about peer reachability.
      if (st != Status::kIOError && st != Status::kResourceExhausted) {
        peer_lat_.Observe(a, sec);
        health_.MarkProbeAlive(a);
      }
    }
    std::unique_lock<std::mutex> lk(probe_mu_);
    probe_cv_.wait_for(lk, std::chrono::milliseconds(probe_interval_ms_),
                       [this] { return !probe_running_.load(std::memory_order_relaxed); });
  }
}

std::string KVClient::MetricsSnapshot() const {
  std::string s;
  const std::string transport =
      transport_reason_.rfind("rdma", 0) == 0 ? "rdma" :
      transport_reason_.rfind("tcp", 0) == 0 ? "tcp" : transport_reason_;
  s += "# HELP dfkv_client_transport_info Client transport selected by dfkv_open_v2\n";
  s += "# TYPE dfkv_client_transport_info gauge\n";
  s += "dfkv_client_transport_info{transport=\"" + transport + "\"} 1\n";
  s += health_.Render();
  s += peer_lat_.Render();
  s += op_stats_.Render();
  if (dedup_) {
    s += "# HELP dfkv_client_dedup_hits_total Rendezvous results served from shm\n";
    s += "# TYPE dfkv_client_dedup_hits_total counter\n";
    s += "dfkv_client_dedup_hits_total " + std::to_string(dedup_->hits()) + "\n";
    s += "# HELP dfkv_client_dedup_fetches_total Keys this process fetched for the rendezvous\n";
    s += "# TYPE dfkv_client_dedup_fetches_total counter\n";
    s += "dfkv_client_dedup_fetches_total " + std::to_string(dedup_->fetches()) + "\n";
    s += "# HELP dfkv_client_dedup_wait_hits_total Waits resolved by a peer's publish\n";
    s += "# TYPE dfkv_client_dedup_wait_hits_total counter\n";
    s += "dfkv_client_dedup_wait_hits_total " + std::to_string(dedup_->wait_hits()) + "\n";
    s += "# HELP dfkv_client_dedup_wait_timeouts_total Waits that fell back to a direct fetch\n";
    s += "# TYPE dfkv_client_dedup_wait_timeouts_total counter\n";
    s += "dfkv_client_dedup_wait_timeouts_total " + std::to_string(dedup_->wait_timeouts()) + "\n";
  }
  if (const GpuNodeDedup* gd = gpu_dedup_raw_.load(std::memory_order_acquire)) {
    s += "# HELP dfkv_client_gpu_dedup_hits_total GPU rendezvous results served via CUDA IPC\n";
    s += "# TYPE dfkv_client_gpu_dedup_hits_total counter\n";
    s += "dfkv_client_gpu_dedup_hits_total " + std::to_string(gd->hits()) + "\n";
    s += "# HELP dfkv_client_gpu_dedup_fetches_total Keys this process fetched for the GPU rendezvous\n";
    s += "# TYPE dfkv_client_gpu_dedup_fetches_total counter\n";
    s += "dfkv_client_gpu_dedup_fetches_total " + std::to_string(gd->fetches()) + "\n";
    s += "# HELP dfkv_client_gpu_dedup_wait_hits_total Waits resolved by a peer's publish\n";
    s += "# TYPE dfkv_client_gpu_dedup_wait_hits_total counter\n";
    s += "dfkv_client_gpu_dedup_wait_hits_total " + std::to_string(gd->wait_hits()) + "\n";
    s += "# HELP dfkv_client_gpu_dedup_wait_timeouts_total Waits that fell back to a direct fetch\n";
    s += "# TYPE dfkv_client_gpu_dedup_wait_timeouts_total counter\n";
    s += "dfkv_client_gpu_dedup_wait_timeouts_total " + std::to_string(gd->wait_timeouts()) + "\n";
  }
  // MDS discovery / placement-ring health. An empty ring means every put/get
  // routes to nowhere and silently reports ok=0, so surface both the current
  // ring size and MDS reachability — the fastest tell of a bad mds_endpoint.
  {
    size_t members;
    { std::lock_guard<std::mutex> lk(ring_mu_); members = addr_.size(); }
    s += "# HELP dfkv_client_ring_members Placement-ring members currently adopted from MDS discovery\n";
    s += "# TYPE dfkv_client_ring_members gauge\n";
    s += "dfkv_client_ring_members " + std::to_string(members) + "\n";
  }
  if (poller_) {
    s += "# HELP dfkv_client_mds_unreachable_polls_total MDS list-members polls that failed (endpoint unreachable or RPC error)\n";
    s += "# TYPE dfkv_client_mds_unreachable_polls_total counter\n";
    s += "dfkv_client_mds_unreachable_polls_total " +
         std::to_string(poller_->unreachable_polls_total()) + "\n";
    s += "# HELP dfkv_client_mds_reachable 1 if the most recent MDS poll succeeded, 0 during an outage\n";
    s += "# TYPE dfkv_client_mds_reachable gauge\n";
    s += "dfkv_client_mds_reachable " + std::string(poller_->reachable() ? "1" : "0") + "\n";
  }
  if (t_ && t_->pipelined())
    s += ReadScheduler::Instance().MetricsText();
  if (t_) s += t_->MetricsText();
  return s;
}

std::vector<bool> KVClient::RecordBatch(
    OpMetrics::Op op, std::chrono::steady_clock::time_point t0,
    std::vector<bool> flags, uint64_t bytes) {
  uint64_t hits = 0;
  for (bool value : flags) if (value) ++hits;
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  op_stats_.Record(op, flags.size(), hits, bytes, seconds);
  return flags;
}

uint64_t KVClient::NowMs() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool KVClient::RefreshMembers(const std::string& seed_addr) {
  std::string text;
  if (t_->Members(seed_addr, &text) != Status::kOk) return false;
  std::vector<std::pair<std::string, std::string>> members;
  for (size_t i = 0; i <= text.size();) {  // parse "name=ip:port,name=ip:port,..."
    size_t c = text.find(',', i);
    if (c == std::string::npos) c = text.size();
    std::string tok = text.substr(i, c - i);
    size_t eq = tok.find('=');
    if (eq != std::string::npos && eq + 1 < tok.size())
      members.emplace_back(tok.substr(0, eq), tok.substr(eq + 1));
    if (c == text.size()) break;
    i = c + 1;
  }
  if (members.empty()) return false;
  SetMembers(std::move(members));
  return true;
}

std::string KVClient::Route(const std::string& key) const {
  std::lock_guard<std::mutex> lk(ring_mu_);
  std::string name;
  if (!ring_.Lookup(key, &name)) return "";
  auto it = addr_.find(name);
  return it == addr_.end() ? "" : it->second;
}

bool KVClient::Put(const std::string& key, const void* value, size_t n) {
  auto metric = op_stats_.Begin(OpMetrics::kPut);
  const bool ok = PutDirect(key, value, n);
  metric.hits = ok ? 1 : 0;
  metric.bytes = ok ? n : 0;
  return ok;
}

bool KVClient::PutDirect(const std::string& key, const void* value, size_t n) {
  if (key.empty() || n == 0 || value == nullptr) return false;
  std::string node = Route(key);
  if (node.empty()) return false;
  uint64_t now = NowMs();
  if (!health_.Healthy(node, now)) return false;
  const BlockKey bk = ToBlockKey(key_namespace_, key);
  Status st;
  if (t_->pipelined()) {
    std::vector<CacheSrc> srcs{CacheSrc{bk, value, n}};
    st = t_->CacheFrom(node, srcs)[0];
  } else {
    st = t_->Cache(node, bk, value, n);
  }
  if (st == Status::kIOError) {
    health_.MarkBad(node, NowMs());
    return false;
  }
  if (st == Status::kOk || st == Status::kNotFound)
    health_.MarkGood(node, NowMs());
  const bool ok = st == Status::kOk;
  if (ok) InvalidateRendezvous(bk);
  return ok;
}

bool KVClient::Get(const std::string& key, void* out, size_t n) {
  auto metric = op_stats_.Begin(OpMetrics::kGet);
  const bool hit = GetDirect(key, out, n);
  metric.hits = hit ? 1 : 0;
  metric.bytes = hit ? n : 0;
  return hit;
}

bool KVClient::GetDirect(const std::string& key, void* out, size_t n) {
  std::string node = Route(key);
  if (node.empty()) return false;
  uint64_t now = NowMs();
  if (!health_.Healthy(node, now)) return false;
  const BlockKey bk = ToBlockKey(key_namespace_, key);
  uint64_t value_len = 0;
  std::vector<BlockKey> keys{bk};
  std::vector<RangeDst> dsts{RangeDst{n ? out : nullptr, n}};
  std::vector<uint64_t> value_lens;
  const Status st = t_->RangeInto(node, keys, dsts, &value_lens)[0];
  if (!value_lens.empty()) value_len = value_lens[0];
  if (st == Status::kIOError) {
    health_.MarkBad(node, NowMs());
    return false;
  }
  if (st == Status::kOk || st == Status::kNotFound)
    health_.MarkGood(node, NowMs());
  return st == Status::kOk && value_len == n;
}

bool KVClient::GetAuto(const std::string& key, std::string* out,
                       size_t max_bytes) {
  auto metric = op_stats_.Begin(OpMetrics::kGet);
  const bool hit = GetAutoDirect(key, out, max_bytes);
  metric.hits = hit ? 1 : 0;
  metric.bytes = hit ? out->size() : 0;
  return hit;
}

bool KVClient::GetAutoDirect(const std::string& key, std::string* out,
                             size_t max_bytes) {
  std::string node = Route(key);
  if (node.empty()) return false;
  uint64_t now = NowMs();
  if (!health_.Healthy(node, now)) return false;
  const BlockKey bk = ToBlockKey(key_namespace_, key);
  uint64_t value_len = 0;
  out->resize(max_bytes);
  std::vector<BlockKey> keys{bk};
  std::vector<RangeDst> dsts{
      RangeDst{max_bytes ? &(*out)[0] : nullptr, max_bytes}};
  std::vector<uint64_t> value_lens;
  const Status st = t_->RangeInto(node, keys, dsts, &value_lens)[0];
  if (!value_lens.empty()) value_len = value_lens[0];
  if (st == Status::kIOError) {
    health_.MarkBad(node, NowMs());
    out->clear();
    return false;
  }
  if (st == Status::kOk || st == Status::kNotFound)
    health_.MarkGood(node, NowMs());
  if (st != Status::kOk || value_len > max_bytes ||
      out->size() < value_len) {
    out->clear();
    return false;
  }
  out->resize(static_cast<size_t>(value_len));
  return true;
}

bool KVClient::GetAuto(const std::string& key, void* out, size_t cap,
                       size_t* out_len) {
  auto metric = op_stats_.Begin(OpMetrics::kGet);
  size_t got = 0;
  const bool hit = GetAutoDirect(key, out, cap, &got);
  if (out_len) *out_len = hit ? got : 0;
  metric.hits = hit ? 1 : 0;
  metric.bytes = hit ? got : 0;
  return hit;
}

bool KVClient::GetAutoDirect(const std::string& key, void* out, size_t cap,
                             size_t* out_len) {
  if (out_len) *out_len = 0;
  std::string node = Route(key);
  if (node.empty()) return false;
  uint64_t now = NowMs();
  if (!health_.Healthy(node, now)) return false;
  const BlockKey bk = ToBlockKey(key_namespace_, key);
  uint64_t value_len = 0;
  std::vector<BlockKey> keys{bk};
  std::vector<RangeDst> dsts{RangeDst{cap ? out : nullptr, cap}};
  std::vector<uint64_t> value_lens;
  const Status st = t_->RangeInto(node, keys, dsts, &value_lens)[0];
  if (!value_lens.empty()) value_len = value_lens[0];
  if (st == Status::kIOError) {
    health_.MarkBad(node, NowMs());
    return false;
  }
  if (st == Status::kOk || st == Status::kNotFound)
    health_.MarkGood(node, NowMs());
  if (st != Status::kOk || value_len > cap) return false;
  if (out_len) *out_len = static_cast<size_t>(value_len);
  return true;
}

bool KVClient::Exist(const std::string& key) {
  auto metric = op_stats_.Begin(OpMetrics::kExist);
  Status status = Status::kInvalid;
  const bool exists = ExistDirect(key, &status);
  metric.hits = exists ? 1 : 0;
  return exists;
}

bool KVClient::ExistDirect(const std::string& key, Status* status) {
  if (status) *status = Status::kInvalid;
  std::string node = Route(key);
  if (node.empty()) return false;
  uint64_t now = NowMs();
  if (!health_.Healthy(node, now)) return false;
  bool exists = false;
  const Status st = t_->Exist(node, ToBlockKey(key_namespace_, key), &exists);
  if (status) *status = st;
  if (st == Status::kIOError) {
    health_.MarkBad(node, NowMs());
    return false;
  }
  if (st == Status::kOk || st == Status::kNotFound)
    health_.MarkGood(node, NowMs());
  return st == Status::kOk && exists;
}

bool KVClient::Remove(const std::string& key) {
  auto metric = op_stats_.Begin(OpMetrics::kRemove);
  const bool ok = RemoveDirect(key);
  metric.hits = ok ? 1 : 0;
  return ok;
}

void KVClient::InvalidateRendezvous(const BlockKey& key) {
  if (dedup_) dedup_->Invalidate(key);
  if (GpuNodeDedup* gd = gpu_dedup_raw_.load(std::memory_order_acquire))
    gd->Invalidate(key);
}

bool KVClient::RemoveDirect(const std::string& key) {
  std::string node = Route(key);
  if (node.empty()) return false;
  uint64_t now = NowMs();
  if (!health_.Healthy(node, now)) return false;
  const BlockKey block_key = ToBlockKey(key_namespace_, key);
  const Status st = t_->Remove(node, block_key);
  if (st == Status::kIOError) {
    health_.MarkBad(node, NowMs());
    return false;
  }
  if (st == Status::kOk || st == Status::kNotFound)
    health_.MarkGood(node, NowMs());
  const bool ok = st == Status::kOk || st == Status::kNotFound;
  if (ok) InvalidateRendezvous(block_key);
  return ok;
}

std::vector<bool> KVClient::BatchPut(const std::vector<KvPutItem>& items) {
  auto t0 = std::chrono::steady_clock::now();
  const size_t N = items.size();
  std::vector<char> ok(N, 0);  // char (not vector<bool>) for thread-safe writes
  if (!t_->pipelined()) {  // TCP fan-out uses scalar bodies, not scalar metrics.
    RunParallel(N, BatchWorkers(N), [&](size_t i) {
      ok[i] = PutDirect(items[i].key, items[i].value, items[i].n) ? 1 : 0;
    });
  } else {
  // RDMA: group by node and scatter-send raw caller values without copying.
  std::map<std::string, std::vector<size_t>> by_node;
  for (size_t i = 0; i < N; ++i) {
    if (items[i].key.empty() || items[i].n == 0 ||
        items[i].value == nullptr)
      continue;
    std::string node = Route(items[i].key);
    if (node.empty()) continue;
    by_node[node].push_back(i);
  }
  std::vector<std::pair<std::string, std::vector<size_t>>> groups(by_node.begin(), by_node.end());
  RunParallel(groups.size(), BatchWorkers(groups.size()), [&](size_t g) {
    const std::string& node = groups[g].first;
    uint64_t now = NowMs();
    if (!health_.Healthy(node, now)) return;
    const std::vector<size_t>& idx = groups[g].second;
    std::vector<CacheSrc> srcs;
    srcs.reserve(idx.size());
    for (size_t k : idx)
      srcs.push_back(CacheSrc{ToBlockKey(key_namespace_, items[k].key),
                              items[k].value, items[k].n});
    std::vector<Status> sts = t_->CacheFrom(node, srcs);
    for (size_t m = 0; m < idx.size(); ++m) {
      ok[idx[m]] = (sts[m] == Status::kOk) ? 1 : 0;
      if (ok[idx[m]]) InvalidateRendezvous(srcs[m].key);
    }
    bool resp = false, ioerr = false;
    // kInvalid (oversize/per-item guard) is neither: it must not clear the
    // peer cooldown (resp) nor trip MarkBad (ioerr).
    for (Status s : sts) {
      if (s == Status::kIOError) ioerr = true;
      else if (s == Status::kOk || s == Status::kNotFound) resp = true;
    }
    if (resp) health_.MarkGood(node, NowMs()); else if (ioerr) health_.MarkBad(node, NowMs());
  });
  }
  uint64_t bytes = 0;
  for (size_t i = 0; i < N; ++i) if (ok[i]) bytes += items[i].n;
  return RecordBatch(OpMetrics::kPut, t0,
                     std::vector<bool>(ok.begin(), ok.end()), bytes);
}

std::vector<bool> KVClient::BatchGet(const std::vector<KvGetItem>& items) {
  const auto t0 = std::chrono::steady_clock::now();
  if (!dedup_) {
    auto result = BatchGetDirect(items);
    uint64_t bytes = 0;
    for (size_t i = 0; i < result.size(); ++i)
      if (result[i]) bytes += items[i].n;
    return RecordBatch(OpMetrics::kGet, t0, std::move(result), bytes);
  }
  // Same-host rendezvous: metrics wrap the whole public call, including shm
  // hits, waits, direct fetches, and fallback reads.
  const size_t N = items.size();
  std::vector<bool> res(N, false);
  std::vector<BlockKey> bks(N);
  std::vector<KvGetItem> fetch_items;
  std::vector<size_t> fetch_map, wait_list;
  std::vector<uint64_t> tokens(N, 0);
  fetch_items.reserve(N);
  // Host rendezvous can't serve device destinations (memcpy to a device VA);
  // route those straight to the fetch list unclaimed instead of crashing when
  // the env switch is set in a GPUDirect process. cu is null on CPU-only hosts.
  const CudaLib* cu = CudaLib::Get();
  for (size_t i = 0; i < N; ++i) {
    if (cu && cu->IsDevicePtr(items[i].out)) {
      fetch_map.push_back(i);
      fetch_items.push_back(items[i]);
      // token remains zero: this direct fetch has no publish obligation.
      continue;
    }
    bks[i] = ToBlockKey(key_namespace_, items[i].key);
    switch (dedup_->Claim(bks[i], items[i].n,
                          static_cast<char*>(items[i].out), &tokens[i])) {
      case NodeDedup::Role::kHit:
        res[i] = true;
        break;
      case NodeDedup::Role::kFetch:
        fetch_map.push_back(i);
        fetch_items.push_back(items[i]);
        // token identifies the claimed slot (or zero for an unclaimed fetch).
        break;
      case NodeDedup::Role::kWait:
        wait_list.push_back(i);
        break;
    }
  }
  if (!fetch_items.empty()) {
    auto r = BatchGetDirect(fetch_items);
    for (size_t m = 0; m < fetch_map.size(); ++m) {
      const size_t i = fetch_map[m];
      res[i] = r[m];
      if (r[m])
        dedup_->Publish(bks[i], NodeDedup::Kind::kData, tokens[i],
                        static_cast<const char*>(items[i].out), items[i].n);
      else
        dedup_->Abort(bks[i], NodeDedup::Kind::kData, tokens[i]);
    }
  }
  for (size_t i : wait_list) {
    if (dedup_->WaitCopy(bks[i], items[i].n, static_cast<char*>(items[i].out)))
      res[i] = true;
    else  // fetcher failed/slow: bounded fallback to a direct read
      res[i] = GetDirect(items[i].key, items[i].out, items[i].n);
  }
  uint64_t bytes = 0;
  for (size_t i = 0; i < N; ++i) if (res[i]) bytes += items[i].n;
  return RecordBatch(OpMetrics::kGet, t0, std::move(res), bytes);
}

std::vector<bool> KVClient::BatchGetDirect(const std::vector<KvGetItem>& items) {
  const size_t N = items.size();
  std::vector<char> hit(N, 0);
  if (!t_->pipelined()) {  // TCP: parallelize across items with our own threads
    RunParallel(N, BatchWorkers(N), [&](size_t i) {
      hit[i] = GetDirect(items[i].key, items[i].out, items[i].n) ? 1 : 0;
    });
    return std::vector<bool>(hit.begin(), hit.end());
  }
  // RDMA: group by (node, n) so each group shares the Range length, then pipeline.
  // Resolve one set of item indices -> hit[]. Reused for the miss-retry pass.
  auto run_pass = [&](const std::vector<size_t>& todo) {
    std::map<std::pair<std::string, size_t>, std::vector<size_t>> by;
    for (size_t i : todo) {
      std::string node = Route(items[i].key);
      if (node.empty()) continue;
      by[{node, items[i].n}].push_back(i);
    }
    std::vector<std::pair<std::pair<std::string, size_t>, std::vector<size_t>>> groups = ShardReadGroups(std::vector<std::pair<std::pair<std::string, size_t>, std::vector<size_t>>>(by.begin(), by.end()));
    RunReadScheduled(groups.size(), [&](size_t g) {
      const std::string& node = groups[g].first.first;
      uint64_t now = NowMs();
      if (!health_.Healthy(node, now)) return;
      const size_t n = groups[g].first.second;
      const std::vector<size_t>& idx = groups[g].second;
      std::vector<BlockKey> keys;
      std::vector<RangeDst> dsts;
      keys.reserve(idx.size());
      dsts.reserve(idx.size());
      for (size_t k : idx) {
        keys.push_back(ToBlockKey(key_namespace_, items[k].key));
        dsts.push_back(RangeDst{items[k].out, n});
      }
      // Raw payload lands straight in items[].out; authoritative stored sizes
      // return in the response prefix.
      std::vector<uint64_t> value_lens;
      std::vector<Status> sts = t_->RangeInto(node, keys, dsts, &value_lens);
      bool resp = false, ioerr = false;
      // kInvalid (oversize/per-item guard) is neither: it must not clear the
      // peer cooldown (resp) nor trip MarkBad (ioerr).
      for (Status s : sts) {
        if (s == Status::kIOError) ioerr = true;
        else if (s == Status::kOk || s == Status::kNotFound) resp = true;
      }
      if (resp) health_.MarkGood(node, NowMs()); else if (ioerr) health_.MarkBad(node, NowMs());
      for (size_t m = 0; m < idx.size(); ++m) {
        if (sts[m] == Status::kOk && m < value_lens.size() &&
            value_lens[m] == n)
          hit[idx[m]] = 1;
      }
    });
  };
  std::vector<size_t> all(N);
  for (size_t i = 0; i < N; ++i) all[i] = i;
  run_pass(all);
  // Bounded miss-retry (R4): under concurrent reads a resident block can transiently
  // read absent for a few hundred microseconds during the RAM->disk flush/reclaim
  // handoff (a shard-lock op briefly hides it from BOTH tiers; the server serves it
  // on the very next attempt -- server cache_miss stays 0, quiescent reads never
  // miss). One re-resolve of the missed subset recovers it. Guarded to a MINORITY
  // of the batch so a genuinely cold batch (mostly missing) is not re-fetched
  // wholesale -- the transient is a sprinkle in an otherwise-hit batch. Off with
  // DFKV_GET_MISS_RETRIES=0.
  for (int r = 0; r < get_miss_retries_; ++r) {
    std::vector<size_t> missed;
    for (size_t i = 0; i < N; ++i) if (!hit[i]) missed.push_back(i);
    if (missed.empty() || missed.size() > N / 2) break;  // none, or a cold batch
    run_pass(missed);
  }
  return std::vector<bool>(hit.begin(), hit.end());
}

std::vector<bool> KVClient::BatchGetAuto(const std::vector<KvGetItem>& items,
                                         std::vector<size_t>* out_lens) {
  const auto t0 = std::chrono::steady_clock::now();
  if (!dedup_) {
    std::vector<size_t> lengths;
    auto result = BatchGetAutoDirect(items, &lengths);
    uint64_t bytes = 0;
    for (size_t length : lengths) bytes += length;
    if (out_lens) *out_lens = std::move(lengths);
    return RecordBatch(OpMetrics::kGet, t0, std::move(result), bytes);
  }
  const size_t N = items.size();
  std::vector<bool> res(N, false);
  std::vector<size_t> lens(N, 0);
  std::vector<BlockKey> bks(N);
  std::vector<KvGetItem> fetch_items;
  std::vector<size_t> fetch_map, wait_list;
  std::vector<uint64_t> tokens(N, 0);
  fetch_items.reserve(N);
  const CudaLib* cu = CudaLib::Get();  // device destinations bypass (see BatchGet)
  for (size_t i = 0; i < N; ++i) {
    if (cu && cu->IsDevicePtr(items[i].out)) {
      fetch_map.push_back(i);
      fetch_items.push_back(items[i]);
      // Unclaimed device fetch: token remains zero.
      continue;
    }
    bks[i] = ToBlockKey(key_namespace_, items[i].key);
    size_t got = 0;
    switch (dedup_->ClaimAuto(bks[i], items[i].n,
                              static_cast<char*>(items[i].out), &got,
                              &tokens[i])) {
      case NodeDedup::Role::kHit:
        res[i] = true;
        lens[i] = got;
        break;
      case NodeDedup::Role::kFetch:
        fetch_map.push_back(i);
        fetch_items.push_back(items[i]);
        // token identifies the fetch generation (zero means direct-only).
        break;
      case NodeDedup::Role::kWait:
        wait_list.push_back(i);
        break;
    }
  }
  if (!fetch_items.empty()) {
    std::vector<size_t> flens;
    auto r = BatchGetAutoDirect(fetch_items, &flens);
    for (size_t m = 0; m < fetch_map.size(); ++m) {
      const size_t i = fetch_map[m];
      res[i] = r[m];
      if (r[m]) lens[i] = flens[m];
      if (r[m]) {
        dedup_->Publish(bks[i], NodeDedup::Kind::kData, tokens[i],
                        static_cast<const char*>(items[i].out), flens[m]);
      } else {
        dedup_->Abort(bks[i], NodeDedup::Kind::kData, tokens[i]);
      }
    }
  }
  for (size_t i : wait_list) {
    size_t got = 0;
    if (dedup_->WaitCopyAuto(bks[i], items[i].n, static_cast<char*>(items[i].out), &got)) {
      res[i] = true;
      lens[i] = got;
    } else {  // fetcher failed/slow: bounded fallback to a direct read
      size_t got2 = 0;
      res[i] = GetAutoDirect(items[i].key, items[i].out, items[i].n, &got2);
      if (res[i]) lens[i] = got2;
    }
  }
  uint64_t bytes = 0;
  for (size_t length : lens) bytes += length;
  if (out_lens) *out_lens = std::move(lens);
  return RecordBatch(OpMetrics::kGet, t0, std::move(res), bytes);
}

std::vector<bool> KVClient::BatchGetAutoDirect(const std::vector<KvGetItem>& items,
                                         std::vector<size_t>* out_lens) {
  const size_t N = items.size();
  std::vector<char> hit(N, 0);
  std::vector<size_t> lens(N, 0);  // distinct indices => thread-safe writes
  if (!t_->pipelined()) {  // TCP: parallelize per item with our own threads.
    RunParallel(N, BatchWorkers(N), [&](size_t i) {
      size_t got = 0;
      if (GetAutoDirect(items[i].key, items[i].out, items[i].n, &got)) {
        hit[i] = 1;
        lens[i] = got;
      }
    });
    if (out_lens) *out_lens = std::move(lens);
    return std::vector<bool>(hit.begin(), hit.end());
  }
  // RDMA: group by (node, cap) so each group shares the Range length, then
  // pipeline. Same zero-copy datapath as BatchGet; the only difference is we
  // accept any payload_len <= cap (instead of == n) and report the true length.
  std::map<std::pair<std::string, size_t>, std::vector<size_t>> by;
  for (size_t i = 0; i < N; ++i) {
    std::string node = Route(items[i].key);
    if (node.empty()) continue;
    by[{node, items[i].n}].push_back(i);
  }
  std::vector<std::pair<std::pair<std::string, size_t>, std::vector<size_t>>> groups = ShardReadGroups(std::vector<std::pair<std::pair<std::string, size_t>, std::vector<size_t>>>(by.begin(), by.end()));
  RunReadScheduled(groups.size(), [&](size_t g) {
    const std::string& node = groups[g].first.first;
    uint64_t now = NowMs();
    if (!health_.Healthy(node, now)) return;
    const size_t cap = groups[g].first.second;
    const std::vector<size_t>& idx = groups[g].second;
    std::vector<BlockKey> keys;
    std::vector<RangeDst> dsts;
    keys.reserve(idx.size());
    dsts.reserve(idx.size());
    for (size_t k : idx) {
      keys.push_back(ToBlockKey(key_namespace_, items[k].key));
      dsts.push_back(RangeDst{items[k].out, cap});
    }
    std::vector<uint64_t> value_lens;
    std::vector<Status> sts = t_->RangeInto(node, keys, dsts, &value_lens);
    bool resp = false, ioerr = false;
    // kInvalid (oversize/per-item guard) is neither: it must not clear the
    // peer cooldown (resp) nor trip MarkBad (ioerr).
    for (Status s : sts) {
      if (s == Status::kIOError) ioerr = true;
      else if (s == Status::kOk || s == Status::kNotFound) resp = true;
    }
    if (resp) health_.MarkGood(node, NowMs()); else if (ioerr) health_.MarkBad(node, NowMs());
    for (size_t m = 0; m < idx.size(); ++m) {
      if (sts[m] != Status::kOk || m >= value_lens.size() ||
          value_lens[m] > cap)
        continue;
      lens[idx[m]] = static_cast<size_t>(value_lens[m]);
      hit[idx[m]] = 1;
    }
  });
  if (out_lens) *out_lens = std::move(lens);
  return std::vector<bool>(hit.begin(), hit.end());
}

std::vector<bool> KVClient::BatchExist(const std::vector<std::string>& keys) {
  const auto t0 = std::chrono::steady_clock::now();
  if (!dedup_) {
    auto result = BatchExistDirect(keys, nullptr);
    return RecordBatch(OpMetrics::kExist, t0, std::move(result), 0);
  }
  // Only authoritative native outcomes are published. A transient route,
  // cooldown, invalid, or I/O failure aborts the slot and retries directly on
  // the next public call rather than masquerading as cached absence.
  const size_t N = keys.size();
  std::vector<bool> res(N, false);
  std::vector<BlockKey> bks(N);
  std::vector<std::string> fetch_keys;
  std::vector<size_t> fetch_map, wait_list;
  std::vector<uint64_t> tokens(N, 0);
  std::vector<char> roles(N, 0), values(N, 0);
  fetch_keys.reserve(N);
  // Shared-memory rendezvous probes are independent per key. Long-context
  // scheduler lookups contain thousands of keys; serial ClaimExist calls can
  // cost more than the sharded RDMA ExistMany itself.
  RunParallel(N, BatchWorkers(N), [&](size_t i) {
    bks[i] = ToBlockKey(key_namespace_, keys[i]);
    bool value = false;
    const NodeDedup::Role role =
        dedup_->ClaimExist(bks[i], &value, &tokens[i]);
    roles[i] = role == NodeDedup::Role::kHit
                   ? 1
                   : (role == NodeDedup::Role::kFetch ? 2 : 3);
    values[i] = value ? 1 : 0;
  });
  for (size_t i = 0; i < N; ++i) {
    if (roles[i] == 1) {
      res[i] = values[i] != 0;
    } else if (roles[i] == 2) {
      fetch_map.push_back(i);
      fetch_keys.push_back(keys[i]);
    } else {
      wait_list.push_back(i);
    }
  }
  if (!fetch_keys.empty()) {
    std::vector<Status> statuses;
    auto r = BatchExistDirect(fetch_keys, &statuses);
    for (size_t m = 0; m < fetch_map.size(); ++m) {
      const size_t i = fetch_map[m];
      res[i] = r[m];
      if (statuses[m] == Status::kOk && r[m]) {
        const char present = 1;
        dedup_->Publish(bks[i], NodeDedup::Kind::kExist, tokens[i],
                        &present, 1);
      } else if (statuses[m] == Status::kNotFound) {
        const char absent = 0;
        dedup_->Publish(bks[i], NodeDedup::Kind::kExist, tokens[i],
                        &absent, 1);
      } else {
        dedup_->Abort(bks[i], NodeDedup::Kind::kExist, tokens[i]);
      }
    }
  }
  for (size_t i : wait_list) {
    bool val = false;
    if (dedup_->WaitExist(bks[i], &val)) res[i] = val;
    else {
      Status status = Status::kInvalid;
      res[i] = ExistDirect(keys[i], &status);
    }
  }
  return RecordBatch(OpMetrics::kExist, t0, std::move(res), 0);
}

std::vector<bool> KVClient::BatchExistDirect(
    const std::vector<std::string>& keys, std::vector<Status>* statuses) {
  const size_t N = keys.size();
  std::vector<char> e(N, 0);
  std::vector<Status> raw(N, Status::kInvalid);
  if (!t_->pipelined()) {  // TCP: parallelize across items with our own threads
    RunParallel(N, BatchWorkers(N), [&](size_t i) {
      e[i] = ExistDirect(keys[i], &raw[i]) ? 1 : 0;
    });
    if (statuses) *statuses = std::move(raw);
    return std::vector<bool>(e.begin(), e.end());
  }
  // RDMA: shard each node's keys across pooled connections. ExistMany is
  // served key-by-key on one QP; a single-node ring would otherwise serialize
  // thousands of scheduler probes before any KV load can start. Keep original
  // indices in every shard so result ordering is unchanged.
  std::map<std::string, std::vector<size_t>> by_node;
  for (size_t i = 0; i < N; ++i) {
    std::string node = Route(keys[i]);
    if (node.empty()) continue;
    by_node[node].push_back(i);
  }
  std::vector<std::pair<std::string, std::vector<size_t>>> groups =
      ShardReadGroups(std::vector<std::pair<std::string, std::vector<size_t>>>(
          by_node.begin(), by_node.end()));
  RunReadScheduled(groups.size(), [&](size_t g) {
    const std::string& node = groups[g].first;
    uint64_t now = NowMs();
    if (!health_.Healthy(node, now)) return;
    const std::vector<size_t>& idx = groups[g].second;
    std::vector<BlockKey> bkeys;
    bkeys.reserve(idx.size());
    for (size_t k : idx) bkeys.push_back(ToBlockKey(key_namespace_, keys[k]));
    std::vector<char> ex;
    std::vector<Status> sts = t_->ExistMany(node, bkeys, &ex);
    bool resp = false, ioerr = false;
    // kInvalid (oversize/per-item guard) is neither: it must not clear the
    // peer cooldown (resp) nor trip MarkBad (ioerr).
    for (Status s : sts) {
      if (s == Status::kIOError) ioerr = true;
      else if (s == Status::kOk || s == Status::kNotFound) resp = true;
    }
    if (resp) health_.MarkGood(node, NowMs()); else if (ioerr) health_.MarkBad(node, NowMs());
    for (size_t m = 0; m < idx.size(); ++m) {
      if (m < sts.size()) raw[idx[m]] = sts[m];
      e[idx[m]] = (m < ex.size() && ex[m]) ? 1 : 0;
    }
  });
  if (statuses) *statuses = std::move(raw);
  return std::vector<bool>(e.begin(), e.end());
}

std::vector<bool> KVClient::BatchRemove(const std::vector<std::string>& keys) {
  const auto t0 = std::chrono::steady_clock::now();
  const size_t N = keys.size();
  std::vector<char> ok(N, 0);
  if (!t_->pipelined()) {
    RunParallel(N, BatchWorkers(N), [&](size_t i) {
      ok[i] = RemoveDirect(keys[i]) ? 1 : 0;
    });
  } else {
  // RDMA: group by node and RemoveMany per node (sequential within a node since
  // eviction is off the hot path; nodes still fan out in parallel).
  std::map<std::string, std::vector<size_t>> by_node;
  for (size_t i = 0; i < N; ++i) {
    std::string node = Route(keys[i]);
    if (node.empty()) continue;
    by_node[node].push_back(i);
  }
  std::vector<std::pair<std::string, std::vector<size_t>>> groups(by_node.begin(), by_node.end());
  RunParallel(groups.size(), BatchWorkers(groups.size()), [&](size_t g) {
    const std::string& node = groups[g].first;
    uint64_t now = NowMs();
    if (!health_.Healthy(node, now)) return;
    const std::vector<size_t>& idx = groups[g].second;
    std::vector<BlockKey> bkeys;
    bkeys.reserve(idx.size());
    for (size_t k : idx) bkeys.push_back(ToBlockKey(key_namespace_, keys[k]));
    std::vector<Status> sts = t_->RemoveMany(node, bkeys);
    bool resp = false, ioerr = false;
    // kInvalid (oversize/per-item guard) is neither: it must not clear the
    // peer cooldown (resp) nor trip MarkBad (ioerr).
    for (Status s : sts) {
      if (s == Status::kIOError) ioerr = true;
      else if (s == Status::kOk || s == Status::kNotFound) resp = true;
    }
    if (resp) health_.MarkGood(node, NowMs()); else if (ioerr) health_.MarkBad(node, NowMs());
    for (size_t m = 0; m < idx.size(); ++m) {
      const bool removed =
          m < sts.size() &&
          (sts[m] == Status::kOk || sts[m] == Status::kNotFound);
      ok[idx[m]] = removed ? 1 : 0;
      if (removed) InvalidateRendezvous(bkeys[m]);
    }
  });
  }
  return RecordBatch(OpMetrics::kRemove, t0,
                     std::vector<bool>(ok.begin(), ok.end()), 0);
}

std::vector<bool> KVClient::BatchPutSg(const std::vector<KvPutItemSg>& items) {
  auto t0 = std::chrono::steady_clock::now();
  const size_t N = items.size();
  std::vector<char> ok(N, 0);  // char (not vector<bool>) for thread-safe writes

  // Reject malformed writes and zero-total values before routing. Individual
  // empty SG segments are harmless when the complete object is non-empty.
  std::vector<char> over(N, 0);
  std::vector<size_t> total_bytes(N, 0);
  for (size_t i = 0; i < N; ++i) {
    bool invalid_buffer = false;
    for (size_t j = 0; j < items[i].ptrs.size(); ++j) {
      if (j < items[i].sizes.size() && items[i].sizes[j] != 0 &&
          items[i].ptrs[j] == nullptr) {
        invalid_buffer = true;
        break;
      }
    }
    if (items[i].key.empty() ||
        items[i].ptrs.size() != items[i].sizes.size() ||
        !CheckedSizeSum(items[i].sizes, &total_bytes[i]) ||
        total_bytes[i] == 0 || invalid_buffer)
      over[i] = 1;
  }

  std::map<std::string, std::vector<size_t>> by_node;
  for (size_t i = 0; i < N; ++i) {
    if (over[i]) continue;
    std::string node = Route(items[i].key);
    if (node.empty()) continue;
    by_node[node].push_back(i);
  }
  // Shard each per-node put group the same way the read path does (shared knobs
  // DFKV_READ_SHARD_KEYS / DFKV_READ_MAX_CONNS): a single-node ring otherwise
  // gathers the whole SG batch into one group -> one worker -> one QP that the
  // server drains key-by-key. Sharding fans it across up to DFKV_READ_MAX_CONNS
  // connections. Each shard keeps original item indices, so result addressing
  // is unchanged.
  std::vector<std::pair<std::string, std::vector<size_t>>> groups =
      ShardReadGroups(std::vector<std::pair<std::string, std::vector<size_t>>>(
          by_node.begin(), by_node.end()));
  RunParallel(groups.size(), BatchWorkers(groups.size()), [&](size_t g) {
    const std::string& node = groups[g].first;
    uint64_t now = NowMs();
    if (!health_.Healthy(node, now)) return;
    const std::vector<size_t>& idx = groups[g].second;
    std::vector<CacheSrcMulti> srcs;
    srcs.reserve(idx.size());
    for (size_t k : idx) {
      CacheSrcMulti s;
      s.key = ToBlockKey(key_namespace_, items[k].key);
      s.payloads.reserve(items[k].ptrs.size());
      for (size_t j = 0; j < items[k].ptrs.size(); ++j)
        s.payloads.emplace_back(items[k].ptrs[j], items[k].sizes[j]);
      srcs.push_back(std::move(s));
    }
    std::vector<Status> sts = t_->CacheFromMulti(node, srcs);
    for (size_t m = 0; m < idx.size(); ++m) {
      ok[idx[m]] = (sts[m] == Status::kOk) ? 1 : 0;
      if (ok[idx[m]]) InvalidateRendezvous(srcs[m].key);
    }
    bool resp = false, ioerr = false;
    // kInvalid (oversize/per-item guard) is neither: it must not clear the
    // peer cooldown (resp) nor trip MarkBad (ioerr).
    for (Status s : sts) {
      if (s == Status::kIOError) ioerr = true;
      else if (s == Status::kOk || s == Status::kNotFound) resp = true;
    }
    if (resp) health_.MarkGood(node, NowMs()); else if (ioerr) health_.MarkBad(node, NowMs());
  });
  uint64_t bytes = 0;
  for (size_t i = 0; i < N; ++i)
    if (ok[i]) AddMetricBytes(total_bytes[i], &bytes);
  return RecordBatch(OpMetrics::kPut, t0,
                     std::vector<bool>(ok.begin(), ok.end()), bytes);
}

GpuNodeDedup* KVClient::GpuDedup(const void* device_dst_hint) {
  std::call_once(gpu_dedup_once_, [this, device_dst_hint] {
    gpu_dedup_ = GpuNodeDedup::FromEnv(namespace_hash_, device_dst_hint);
    gpu_dedup_raw_.store(gpu_dedup_.get(), std::memory_order_release);
  });
  return gpu_dedup_.get();
}

std::vector<bool> KVClient::BatchGetAutoSg(const std::vector<KvGetItemSg>& items,
                                           std::vector<size_t>* out_lens) {
  const auto t0 = std::chrono::steady_clock::now();
  // Init the GPU rendezvous only once a DEVICE destination shows up: its
  // pointer picks the primary context to bind when this (transfer) thread
  // has none, and host-only callers never pay for an arena.
  GpuNodeDedup* gd = nullptr;
  if (const CudaLib* cu0 = CudaLib::Get()) {
    const void* hint = nullptr;
    for (const auto& it : items)
      if (!it.dsts.empty() && cu0->IsDevicePtr(it.dsts[0])) { hint = it.dsts[0]; break; }
    if (hint) gd = GpuDedup(hint);
  }
  if (!gd || items.empty()) {
    std::vector<size_t> lengths;
    auto result = BatchGetAutoSgDirect(items, &lengths);
    uint64_t bytes = 0;
    for (size_t length : lengths)
      AddMetricBytes(length, &bytes);
    if (out_lens) *out_lens = std::move(lengths);
    return RecordBatch(OpMetrics::kGet, t0, std::move(result), bytes);
  }
  // Same-host rendezvous for GPU destinations (phase 2b, see node_dedup_gpu.h):
  // the vLLM connector's SG gets land in device memory, where lockstep TP
  // ranks re-fetch identical TP-replicated KV. Payloads rendezvous through
  // per-process GPU staging arenas (CUDA IPC + NVLink D2D); every outcome
  // degrades to the plain remote path.
  const CudaLib* cu = CudaLib::Get();  // non-null: gd exists
  const size_t N = items.size();
  std::vector<bool> res(N, false);
  std::vector<size_t> lens(N, 0);
  std::vector<BlockKey> bks(N);
  std::vector<KvGetItemSg> fetch_items;
  std::vector<size_t> fetch_map, wait_list;
  std::vector<uint64_t> tokens(N, 0);
  std::vector<GpuNodeDedup::Seg> segs;
  auto segs_of = [&segs](const KvGetItemSg& it) {
    segs.clear();
    segs.reserve(it.dsts.size());
    for (size_t j = 0; j < it.dsts.size(); ++j)
      segs.push_back(GpuNodeDedup::Seg{it.dsts[j], it.caps[j]});
  };
  std::vector<size_t> total_caps(N, 0);
  std::vector<char> totals_valid(N, 0);
  for (size_t i = 0; i < N; ++i)
    totals_valid[i] = CheckedSizeSum(items[i].caps, &total_caps[i]) ? 1 : 0;
  for (size_t i = 0; i < N; ++i) {
    // Eligible = well-shaped AND device-memory destination. Host-destination
    // SG items (or malformed ones the Direct guard rejects) skip the
    // rendezvous entirely — an unclaimed fetch has no publish obligation.
    const bool eligible = !items[i].key.empty() && !items[i].dsts.empty() &&
                          items[i].dsts.size() == items[i].caps.size() &&
                          totals_valid[i] && cu->IsDevicePtr(items[i].dsts[0]);
    if (!eligible) {
      fetch_map.push_back(i);
      fetch_items.push_back(items[i]);
      // token remains zero: no rendezvous publish obligation.
      continue;
    }
    bks[i] = ToBlockKey(key_namespace_, items[i].key);
    segs_of(items[i]);
    size_t got = 0;
    switch (gd->ClaimSg(bks[i], segs.data(), segs.size(),
                        total_caps[i], &got, &tokens[i])) {
      case GpuNodeDedup::Role::kHit:
        res[i] = true;
        lens[i] = got;
        break;
      case GpuNodeDedup::Role::kFetch:
        fetch_map.push_back(i);
        fetch_items.push_back(items[i]);
        // token identifies this fetch generation.
        break;
      case GpuNodeDedup::Role::kWait:
        wait_list.push_back(i);
        break;
    }
  }
  if (!fetch_items.empty()) {
    std::vector<size_t> flens;
    auto r = BatchGetAutoSgDirect(fetch_items, &flens);
    for (size_t m = 0; m < fetch_map.size(); ++m) {
      const size_t i = fetch_map[m];
      res[i] = r[m];
      if (r[m]) lens[i] = flens[m];
      if (r[m]) {
        segs_of(items[i]);
        gd->PublishSg(bks[i], tokens[i], segs.data(), segs.size(), flens[m]);
      } else {
        gd->Abort(bks[i], tokens[i]);
      }
    }
  }
  // One amortized-sync wait for the WHOLE wait list (WaitManySg): copies are
  // enqueued as keys turn READY and the stream synchronizes per round, not
  // per key — per-key syncs cost ~ms under load and burned any wait budget
  // long before the list was through (the dedup win leaked back out through
  // fallback re-fetches; measured live in the PD A/B). Failures fall back
  // through ONE direct batch.
  std::vector<size_t> fb;
  std::vector<GpuNodeDedup::WaitItem> wits(wait_list.size());
  std::vector<std::vector<GpuNodeDedup::Seg>> wsegs(wait_list.size());
  for (size_t m = 0; m < wait_list.size(); ++m) {
    const size_t i = wait_list[m];
    wsegs[m].reserve(items[i].dsts.size());
    for (size_t j = 0; j < items[i].dsts.size(); ++j)
      wsegs[m].push_back(GpuNodeDedup::Seg{items[i].dsts[j], items[i].caps[j]});
    wits[m] = GpuNodeDedup::WaitItem{bks[i], wsegs[m].data(), wsegs[m].size(),
                                     total_caps[i], false, 0};
  }
  gd->WaitManySg(wits.data(), wits.size());
  for (size_t m = 0; m < wait_list.size(); ++m) {
    const size_t i = wait_list[m];
    if (wits[m].ok) {
      res[i] = true;
      lens[i] = wits[m].got;
    } else {
      fb.push_back(i);
    }
  }
  if (!fb.empty()) {
    std::vector<KvGetItemSg> rest;
    rest.reserve(fb.size());
    for (size_t i : fb) rest.push_back(items[i]);
    std::vector<size_t> rl;
    auto rr = BatchGetAutoSgDirect(rest, &rl);
    for (size_t m = 0; m < fb.size(); ++m) {
      res[fb[m]] = rr[m];
      if (rr[m]) lens[fb[m]] = rl[m];
    }
  }
  // Per-batch split, opt-in (DFKV_CLIENT_NODE_DEDUP_LOG=1): the one line that
  // distinguishes "waits time out" from "claims never dedup" when the server
  // counters look wrong — cheap enough to leave on in a canary.
  static const bool log_split = [] {
    const char* v = std::getenv("DFKV_CLIENT_NODE_DEDUP_LOG");
    return v && std::strcmp(v, "1") == 0;
  }();
  if (log_split) {
    size_t hits0 = 0;
    for (bool b : res) hits0 += b;
    DFKV_LOG_INFO("gpu-dedup batch: n=" + std::to_string(N) +
                  " claim_hit=" + std::to_string(N - fetch_map.size() - wait_list.size()) +
                  " fetched=" + std::to_string(fetch_map.size()) +
                  " waited=" + std::to_string(wait_list.size()) +
                  " fallback=" + std::to_string(fb.size()) +
                  " ok=" + std::to_string(hits0) +
                  " cum(h=" + std::to_string(gd->hits()) +
                  ",f=" + std::to_string(gd->fetches()) +
                  ",wh=" + std::to_string(gd->wait_hits()) +
                  ",wt=" + std::to_string(gd->wait_timeouts()) + ")");
  }
  uint64_t bytes = 0;
  for (size_t length : lens)
    AddMetricBytes(length, &bytes);
  if (out_lens) *out_lens = std::move(lens);
  return RecordBatch(OpMetrics::kGet, t0, std::move(res), bytes);
}

std::vector<bool> KVClient::BatchGetAutoSgDirect(const std::vector<KvGetItemSg>& items,
                                                 std::vector<size_t>* out_lens) {
  const size_t N = items.size();
  std::vector<char> hit(N, 0);
  std::vector<size_t> lens(N, 0);  // distinct indices => thread-safe writes

  // Validate shape and aggregate capacity; the transport windows arbitrary
  // descriptor counts internally.
  std::vector<char> over(N, 0);
  std::vector<size_t> total_caps(N, 0);
  for (size_t i = 0; i < N; ++i) {
    if (items[i].key.empty() ||  // null/empty key: skip (no wasted GET issued)
        items[i].dsts.size() != items[i].caps.size() ||
        !CheckedSizeSum(items[i].caps, &total_caps[i]))
      over[i] = 1;
  }

  // Group by (node, total_cap): a group shares the Range length (= sum of caps)
  // so the existing RangeInto windowing/pipelining applies unchanged.
  auto run_pass = [&](const std::vector<size_t>& todo) {
    std::map<std::pair<std::string, size_t>, std::vector<size_t>> by;
    for (size_t i : todo) {
      if (over[i]) continue;
      std::string node = Route(items[i].key);
      if (node.empty()) continue;
      by[{node, total_caps[i]}].push_back(i);
    }
    std::vector<std::pair<std::pair<std::string, size_t>,
                          std::vector<size_t>>>
        groups = ShardReadGroups(
            std::vector<std::pair<std::pair<std::string, size_t>,
                                  std::vector<size_t>>>(by.begin(), by.end()));
    RunReadScheduled(groups.size(), [&](size_t g) {
      const std::string& node = groups[g].first.first;
      uint64_t now = NowMs();
      if (!health_.Healthy(node, now)) return;
      const std::vector<size_t>& idx = groups[g].second;
      std::vector<BlockKey> keys;
      std::vector<RangeDstMulti> dsts;
      keys.reserve(idx.size());
      dsts.reserve(idx.size());
      for (size_t k : idx) {
        keys.push_back(ToBlockKey(key_namespace_, items[k].key));
        RangeDstMulti d;
        d.payloads.reserve(items[k].dsts.size());
        for (size_t j = 0; j < items[k].dsts.size(); ++j)
          d.payloads.emplace_back(items[k].dsts[j], items[k].caps[j]);
        dsts.push_back(std::move(d));
      }
      std::vector<size_t> value_lens;
      std::vector<Status> sts =
          t_->RangeIntoMulti(node, keys, dsts, &value_lens);
      bool resp = false, ioerr = false;
      // kInvalid (oversize/per-item guard) is neither: it must not clear the
      // peer cooldown (resp) nor trip MarkBad (ioerr).
      for (Status s : sts) {
        if (s == Status::kIOError) ioerr = true;
        else if (s == Status::kOk || s == Status::kNotFound) resp = true;
      }
      if (resp) health_.MarkGood(node, NowMs());
      else if (ioerr) health_.MarkBad(node, NowMs());
      for (size_t m = 0; m < idx.size(); ++m) {
        size_t cap = groups[g].first.second;
        if (sts[m] != Status::kOk || m >= value_lens.size() ||
            value_lens[m] > cap)
          continue;
        lens[idx[m]] = static_cast<size_t>(value_lens[m]);
        hit[idx[m]] = 1;
      }
    });
  };
  std::vector<size_t> all(N);
  for (size_t i = 0; i < N; ++i) all[i] = i;
  run_pass(all);
  // Match scalar BatchGet's bounded transient-miss recovery. Under concurrent
  // RAM flush/reclaim or connection turnover a small subset can return absent
  // for one pass even though the server serves it immediately on retry. Retry
  // only a minority; a genuinely cold SG batch must remain a single probe.
  for (int r = 0; r < get_miss_retries_; ++r) {
    std::vector<size_t> missed;
    for (size_t i = 0; i < N; ++i)
      if (!over[i] && !hit[i]) missed.push_back(i);
    if (missed.empty() || missed.size() > N / 2) break;
    run_pass(missed);
  }
  if (out_lens) *out_lens = std::move(lens);
  return std::vector<bool>(hit.begin(), hit.end());
}

}  // namespace dfkv
