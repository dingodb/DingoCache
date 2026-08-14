#include "client/kv_client.h"

#include <unistd.h>
#include <sys/random.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <atomic>
#include <climits>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <deque>
#include <functional>
#include <memory>
#include <map>
#include <mutex>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include "client/group_shard.h"
#include "client/key_map.h"
#include "client/read_scheduler.h"
#include "utils/log.h"
#include "utils/sha256.h"
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

bool KVClient::RegisterMemory(void* base, size_t size) {
  const uintptr_t address = reinterpret_cast<uintptr_t>(base);
  if (!base || size == 0 ||
      size > std::numeric_limits<uintptr_t>::max() - address) {
    return false;
  }

  // Serialize declarations with classification reads so a GET concurrent with
  // a successful registration cannot slip through the publication gap and
  // initialize/query CUDA on its registered host destination.
  std::unique_lock<std::shared_mutex> lock(registered_memory_mu_);
  const CudaLib* cuda = CudaLib::Get();
  const DestinationMemoryKind kind =
      cuda && cuda->IsDevicePtr(base) ? DestinationMemoryKind::kDevice
                                      : DestinationMemoryKind::kHost;
  if (!t_->RegisterMemory(base, size)) return false;

  const uintptr_t end = address + size;
  for (auto it = registered_memory_.begin();
       it != registered_memory_.end(); ++it) {
    if (it->base != address) continue;
    // Same-base declarations are one transport pool. A smaller redeclaration
    // does not shrink it; growth replaces the older entry and becomes newest.
    RegisteredMemoryRange updated{
        address, std::max(it->end, end), kind};
    registered_memory_.erase(it);
    registered_memory_.push_back(updated);
    return true;
  }
  registered_memory_.push_back({address, end, kind});
  return true;
}

DestinationMemoryKind KVClient::ClassifyDestination(
    const void* destination, size_t size) const {
  if (!destination || size == 0) return DestinationMemoryKind::kHost;
  const uintptr_t address = reinterpret_cast<uintptr_t>(destination);
  const bool valid =
      size <= std::numeric_limits<uintptr_t>::max() - address;
  if (valid) {
    std::shared_lock<std::shared_mutex> lock(registered_memory_mu_);
    // Reverse declaration order gives overlaps an explicit latest-wins rule.
    // Subtraction-based containment cannot wrap at the end of address space.
    for (auto it = registered_memory_.rbegin();
         it != registered_memory_.rend(); ++it) {
      if (address >= it->base && address < it->end &&
          size <= it->end - address) {
        return it->kind;
      }
    }
  }

  // No lifetime is known for an unregistered pointer, so caching a host answer
  // could misclassify a later CUDA allocation that reuses the same VA.
  const CudaLib* cuda = CudaLib::Get();
  return cuda && cuda->IsDevicePtr(destination)
             ? DestinationMemoryKind::kDevice
             : DestinationMemoryKind::kHost;
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

void KVClient::PublishPeerTopologies(
    const std::vector<MemberInfo>& topology, uint64_t generation) {
  std::map<std::string, std::string> current;
  for (const auto& member : topology)
    current[member.id] = member.ip + ":" + std::to_string(member.port);

  std::set<std::pair<std::string, std::string>> retired;
  {
    std::lock_guard<std::mutex> lock(ring_mu_);
    for (const auto& peer : published_peers_) retired.insert(peer);
    for (const auto& peer : addr_) retired.insert(peer);
    for (const auto& peer : current) retired.erase(peer);
    published_peers_ = current;
  }

  // Retire old address/identity bindings before publishing replacements. In
  // particular, replacing an id at the same address must not let the old
  // retirement tear down the newly-published binding.
  for (const auto& [peer_id, address] : retired) {
    PeerTopology peer;
    peer.peer_addr = address;
    peer.generation = generation;
    peer.complete = false;
    peer.peer_id = peer_id;
    peer.present = false;
    t_->OnPeerTopology(peer);
  }

  for (const auto& member : topology) {
    PeerTopology peer;
    peer.peer_addr = member.ip + ":" + std::to_string(member.port);
    peer.generation = generation;
    peer.complete = member.has_health;
    peer.rails.reserve(member.health.ib_devices.size());
    for (const auto& device : member.health.ib_devices)
      peer.rails.push_back(PeerRailTopology{device.name, device.healthy()});
    std::sort(peer.rails.begin(), peer.rails.end(),
              [](const PeerRailTopology& a, const PeerRailTopology& b) {
                return a.name < b.name;
              });
    peer.peer_id = member.id;
    peer.present = true;
    t_->OnPeerTopology(peer);
  }

  std::vector<std::string> live_peer_ids;
  live_peer_ids.reserve(current.size());
  for (const auto& [peer_id, address] : current) {
    (void)address;
    live_peer_ids.push_back(peer_id);
  }
  t_->OnPeerIdentities(live_peer_ids);
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
    if (!m.RingEligible()) continue;
    ring.AddNode(m.id, m.weight < 1 ? 1 : static_cast<int>(m.weight));
    addr[m.id] = m.ip + ":" + std::to_string(m.port);
  }
  ring.Build();
  AdoptRing(std::move(ring), std::move(addr));
}

void KVClient::StartMdsDiscovery(std::vector<std::string> mds_eps,
                                 const std::string& group, int poll_ms) {
  StopMdsDiscovery();
  struct PlacementEpochState {
    bool have_epoch = false;
    uint64_t epoch = 0;
  };
  auto placement_state = std::make_shared<PlacementEpochState>();
  poller_ = std::make_unique<MdsMemberPoller>(
      std::move(mds_eps), group,
      [this, placement_state](const std::vector<MemberInfo>& topology,
                              uint64_t topology_epoch,
                              bool placement_admitted) {
        // Topology is a replace-all stream independent of placement. Degraded
        // members remain here even though SetMembers excludes them. Omitted
        // peers still held by a guard-rejected ring are explicitly invalidated.
        PublishPeerTopologies(topology, topology_epoch);

        if (!placement_admitted) return;
        std::vector<MemberInfo> placement;
        placement.reserve(topology.size());
        for (const auto& member : topology) {
          if (member.RingEligible()) placement.push_back(member);
        }
        const uint64_t placement_epoch = MembersEpoch(placement);
        if (placement_state->have_epoch &&
            placement_state->epoch == placement_epoch)
          return;
        placement_state->epoch = placement_epoch;
        placement_state->have_epoch = true;
        SetMembers(placement);
      },
      poll_ms);
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
      // kResourceExhausted and kNoCompatibleRail never dialed the peer at all:
      // their wall time is local resource/topology selection and would poison
      // the latency EWMA; neither proves anything about peer reachability.
      if (st != Status::kIOError && st != Status::kResourceExhausted &&
          st != Status::kNoCompatibleRail) {
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
  std::vector<RangeDst> dsts{
      RangeDst{n ? out : nullptr, n, ClassifyDestination(out, n)}};
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
  std::vector<RangeDst> dsts{RangeDst{
      max_bytes ? &(*out)[0] : nullptr, max_bytes,
      DestinationMemoryKind::kHost}};
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
  std::vector<RangeDst> dsts{
      RangeDst{cap ? out : nullptr, cap, ClassifyDestination(out, cap)}};
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
  // Host rendezvous cannot serve device destinations (memcpy to a device VA);
  // registered pools use the declaration cache, while unregistered pointers
  // take the conservative CUDA probe.
  for (size_t i = 0; i < N; ++i) {
    if (ClassifyDestination(items[i].out, items[i].n) ==
        DestinationMemoryKind::kDevice) {
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
        dsts.push_back(RangeDst{items[k].out, n,
                                ClassifyDestination(items[k].out, n)});
      }
      // The transport stages retry-owned bytes, then publishes them into each
      // host or CUDA destination before returning the authoritative size.
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
  for (size_t i = 0; i < N; ++i) {
    if (ClassifyDestination(items[i].out, items[i].n) ==
        DestinationMemoryKind::kDevice) {
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
  // pipeline. Same staged-publication datapath as BatchGet; the only
  // difference is accepting payload_len <= cap and reporting the true length.
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
      dsts.push_back(RangeDst{items[k].out, cap,
                              ClassifyDestination(items[k].out, cap)});
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

namespace {

// Shared per-node (ip:port) stats for GET/PUT/EXIST. Gated by
// DFKV_CLIENT_PERNODE_STATS=1 (off by default). The disabled path is this one
// cached branch; no clocks, maps, strings, or sink state are touched.
bool PernodeOn() {
  static const bool on = [] {
    const char* e = std::getenv("DFKV_CLIENT_PERNODE_STATS");
    return e && *e && *e != '0';
  }();
  return on;
}

struct PernodeStat {
  // issued_keys counts transport items, so a retried GET is counted once per
  // attempt. logical_keys exists only on TOTAL and always means caller inputs.
  uint64_t issued_keys = 0, bytes = 0, busy_us = 0, wall_us = 0, shards = 0;
  uint64_t min_b = std::numeric_limits<uint64_t>::max(), max_b = 0;
  uint64_t hit = 0;   // GET found / PUT stored / EXIST present
  uint64_t fail = 0;  // issued responses other than kOk/kNotFound (PUT: non-kOk)
  uint64_t nonissued_fail = 0;
  uint64_t invalid_input = 0, no_route = 0, health_cooldown = 0;
  uint64_t resource_exhausted = 0, no_compatible_rail = 0;
  uint32_t status_mask = 0;       // statuses returned by the transport
  uint32_t fail_status_mask = 0;  // failed statuses returned by the transport
  uint32_t nonissued_status_mask = 0;
  std::string first_fail;         // keyed id of first failed key
  std::string dev;                // slowest shard's IB rail
};

const char* PernodeStatusName(Status s) {
  switch (s) {
    case Status::kOk: return "kOk";
    case Status::kNotFound: return "kNotFound";
    case Status::kCacheFull: return "kCacheFull";
    case Status::kQuotaExceeded: return "kQuotaExceeded";
    case Status::kIOError: return "kIOError";
    case Status::kInvalid: return "kInvalid";
    case Status::kResourceExhausted: return "kResourceExhausted";
    case Status::kNoCompatibleRail: return "kNoCompatibleRail";
  }
  return "kUnknown";
}

uint32_t PernodeStatusBit(Status s) {
  const int value = static_cast<int>(s);
  return value >= 0 && value < 32 ? (1u << value) : 0;
}

std::string PernodeStatusList(uint32_t mask) {
  std::string out;
  for (int value = 0; value < 8; ++value) {
    if ((mask & (1u << value)) == 0) continue;
    if (!out.empty()) out += ",";
    out += PernodeStatusName(static_cast<Status>(value));
  }
  return out.empty() ? "none" : out;
}

// Process-local HMAC-SHA256 makes repeated failures correlatable without
// leaving a dictionary-reversible digest of the raw key in logs. If kernel
// randomness is unavailable, omit the digest rather than weakening privacy.
std::string PernodeKeyFp(const std::string& key) {
  struct Key {
    std::array<uint8_t, 32> bytes{};
    bool ready = false;
  };
  static const Key process_key = [] {
    Key result;
    size_t offset = 0;
    while (offset < result.bytes.size()) {
      const ssize_t n =
          ::getrandom(result.bytes.data() + offset,
                      result.bytes.size() - offset, 0);
      if (n > 0) {
        offset += static_cast<size_t>(n);
        continue;
      }
      if (n < 0 && errno == EINTR) continue;
      return result;
    }
    result.ready = true;
    return result;
  }();
  if (!process_key.ready)
    return "len=" + std::to_string(key.size()) + ":digest=redacted";

  std::array<uint8_t, 64> inner_pad{}, outer_pad{};
  inner_pad.fill(0x36);
  outer_pad.fill(0x5c);
  for (size_t i = 0; i < process_key.bytes.size(); ++i) {
    inner_pad[i] ^= process_key.bytes[i];
    outer_pad[i] ^= process_key.bytes[i];
  }
  std::vector<uint8_t> inner(inner_pad.size() + key.size());
  std::memcpy(inner.data(), inner_pad.data(), inner_pad.size());
  if (!key.empty())
    std::memcpy(inner.data() + inner_pad.size(), key.data(), key.size());
  std::array<uint8_t, 32> inner_digest{};
  Sha256(inner.data(), inner.size(), inner_digest.data());
  std::array<uint8_t, 64 + 32> outer{};
  std::memcpy(outer.data(), outer_pad.data(), outer_pad.size());
  std::memcpy(outer.data() + outer_pad.size(), inner_digest.data(),
              inner_digest.size());
  std::array<uint8_t, 32> digest{};
  Sha256(outer.data(), outer.size(), digest.data());

  static const char* hex = "0123456789abcdef";
  std::string id(32, '0');
  for (size_t i = 0; i < 16; ++i) {
    id[2 * i] = hex[digest[i] >> 4];
    id[2 * i + 1] = hex[digest[i] & 0xf];
  }
  return "len=" + std::to_string(key.size()) + ":hmac-sha256=" + id;
}

enum class PernodeNonIssuedReason {
  kInvalidInput,
  kNoRoute,
  kHealthCooldown,
  kResourceExhausted,
  kNoCompatibleRail,
};

uint32_t PernodeNonIssuedStatusBit(PernodeNonIssuedReason reason) {
  switch (reason) {
    case PernodeNonIssuedReason::kInvalidInput:
      return PernodeStatusBit(Status::kInvalid);
    case PernodeNonIssuedReason::kResourceExhausted:
      return PernodeStatusBit(Status::kResourceExhausted);
    case PernodeNonIssuedReason::kNoCompatibleRail:
      return PernodeStatusBit(Status::kNoCompatibleRail);
    case PernodeNonIssuedReason::kNoRoute:
    case PernodeNonIssuedReason::kHealthCooldown:
      return 0;
  }
  return 0;
}

void RecordPernodeNonIssued(PernodeStat* stat,
                            PernodeNonIssuedReason reason, uint64_t count,
                            const std::string& first_key) {
  stat->nonissued_fail += count;
  switch (reason) {
    case PernodeNonIssuedReason::kInvalidInput:
      stat->invalid_input += count;
      break;
    case PernodeNonIssuedReason::kNoRoute:
      stat->no_route += count;
      break;
    case PernodeNonIssuedReason::kHealthCooldown:
      stat->health_cooldown += count;
      break;
    case PernodeNonIssuedReason::kResourceExhausted:
      stat->resource_exhausted += count;
      break;
    case PernodeNonIssuedReason::kNoCompatibleRail:
      stat->no_compatible_rail += count;
      break;
  }
  stat->nonissued_status_mask |= PernodeNonIssuedStatusBit(reason);
  if (stat->first_fail.empty()) stat->first_fail = PernodeKeyFp(first_key);
}

void MergePernodeNonIssued(PernodeStat* into, const PernodeStat& from) {
  into->nonissued_fail += from.nonissued_fail;
  into->invalid_input += from.invalid_input;
  into->no_route += from.no_route;
  into->health_cooldown += from.health_cooldown;
  into->resource_exhausted += from.resource_exhausted;
  into->no_compatible_rail += from.no_compatible_rail;
  into->nonissued_status_mask |= from.nonissued_status_mask;
  if (into->first_fail.empty() && !from.first_fail.empty())
    into->first_fail = from.first_fail;
}

std::string PernodeNonIssuedReasons(const PernodeStat& stat) {
  std::string out;
  auto append = [&](const char* reason, const char* status, uint64_t count) {
    if (count == 0) return;
    if (!out.empty()) out += ",";
    out += reason;
    out += ":";
    out += status;
    out += "=";
    out += std::to_string(count);
  };
  append("invalid_input", "kInvalid", stat.invalid_input);
  append("no_route", "not_issued", stat.no_route);
  append("health_cooldown", "not_issued", stat.health_cooldown);
  append("resource_exhausted", "kResourceExhausted",
         stat.resource_exhausted);
  append("no_compatible_rail", "kNoCompatibleRail",
         stat.no_compatible_rail);
  return out.empty() ? "none" : out;
}

std::string PernodeFmt1(uint64_t x10) {
  return std::to_string(x10 / 10) + "." + std::to_string(x10 % 10);
}

const std::string& PernodeSuffix() {
  static const std::string suffix = [] {
    const char* env_suffix = std::getenv("DFKV_CLIENT_LOG_SUFFIX");
    const std::string pid_tag = "p" + std::to_string(::getpid());
    std::string value =
        env_suffix && *env_suffix ? env_suffix : pid_tag;
    for (char& c : value)
      if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_')
        c = '_';
    // A launcher-supplied rank identity is attributable but not necessarily
    // process-unique (for example TP rank repeats across DP/PP groups).
    if (value != pid_tag &&
        (value.size() <= pid_tag.size() ||
         value.compare(value.size() - pid_tag.size(), pid_tag.size(),
                       pid_tag) != 0))
      value += "_" + pid_tag;
    return value;
  }();
  return suffix;
}

// File I/O is confined to this worker. Producers only move an already-formatted
// batch into a bounded queue and notify it. Rotation keeps exactly current +
// ".1" per operation file, so both memory and disk file count are bounded.
class PernodeSink {
 public:
  static PernodeSink& Instance() {
    static PernodeSink sink;
    return sink;
  }

  void Enqueue(const char* fname, std::vector<std::string> lines) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (queue_.size() >= capacity_) {
        ++dropped_total_;
        ++dropped_pending_;
        return;
      }
      queue_.push_back({fname, std::move(lines)});
      high_water_ = std::max(high_water_, queue_.size());
    }
    cv_.notify_one();
  }

  void DrainForTest() {
    std::unique_lock<std::mutex> lock(mu_);
    flush_requested_ = true;
    cv_.notify_one();
    drained_.wait(lock, [&] {
      return queue_.empty() && !writing_ && !flush_requested_;
    });
  }

  size_t capacity() const { return capacity_; }
  uint64_t rotate_bytes() const { return rotate_bytes_; }
  size_t high_water() const {
    std::lock_guard<std::mutex> lock(mu_);
    return high_water_;
  }
  uint64_t dropped() const {
    std::lock_guard<std::mutex> lock(mu_);
    return dropped_total_;
  }

  uint64_t pending_dropped() const {
    std::lock_guard<std::mutex> lock(mu_);
    return dropped_pending_;
  }
  void PauseForTest(bool paused) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      paused_for_test_ = paused;
    }
    if (!paused) cv_.notify_one();
  }

 private:
  struct Batch {
    std::string fname;
    std::vector<std::string> lines;
    uint64_t dropped_before = 0;
  };
  struct FileState {
    FILE* file = nullptr;
    std::string path;
    uint64_t bytes = 0;
  };

  PernodeSink()
      : capacity_(std::min<size_t>(
            EnvSize("DFKV_CLIENT_PERNODE_QUEUE_CAPACITY", 256), 65536)),
        rotate_bytes_(std::max<size_t>(
            1, EnvSize("DFKV_CLIENT_PERNODE_ROTATE_BYTES", 64u << 20))),
        worker_([this] { Run(); }) {}

  ~PernodeSink() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stopping_ = true;
    }
    cv_.notify_one();
    worker_.join();
  }

  FileState& Open(const std::string& fname) {
    FileState& state = files_[fname];
    if (state.file || !state.path.empty()) return state;
    const char* dir = std::getenv("DFKV_ACCESS_DIR");
    if (!dir || !*dir) return state;
    state.path = std::string(dir) + "/" + fname + "." + PernodeSuffix();
    state.file = std::fopen(state.path.c_str(), "a");
    if (!state.file) {
      DFKV_LOG_WARN("pernode: cannot open " + state.path + "; using stderr");
      return state;
    }
    if (std::fseek(state.file, 0, SEEK_END) == 0) {
      const long offset = std::ftell(state.file);
      if (offset > 0) state.bytes = static_cast<uint64_t>(offset);
    }
    return state;
  }

  void Rotate(FileState& state) {
    if (state.file) std::fclose(state.file);
    state.file = nullptr;
    const std::string previous = state.path + ".1";
    std::remove(previous.c_str());
    if (std::rename(state.path.c_str(), previous.c_str()) != 0)
      DFKV_LOG_WARN("pernode: cannot rotate " + state.path);
    state.file = std::fopen(state.path.c_str(), "w");
    state.bytes = 0;
    if (!state.file)
      DFKV_LOG_WARN("pernode: cannot reopen " + state.path + "; using stderr");
  }

  void Write(Batch batch) {
    char ts[32];
    const std::time_t tt = std::time(nullptr);
    std::tm tm{};
    localtime_r(&tt, &tm);
    std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm);
    const std::string drop_line =
        batch.dropped_before == 0
            ? std::string()
            : "pernode-sink DROPPED: dropped_batches=" +
                  std::to_string(batch.dropped_before);

    FileState& state = Open(batch.fname);
    auto write_line = [&](const std::string& line) {
      if (!state.file) {
        DFKV_LOG_INFO(line);
        return;
      }

      std::string serialized =
          std::string(ts) + " " + line + "\n";
      if (serialized.size() > rotate_bytes_) {
        const size_t original_bytes = serialized.size();
        serialized = std::string(ts) +
                     " pernode-sink OVERSIZE: original_bytes=" +
                     std::to_string(original_bytes) + "\n";
        if (serialized.size() > rotate_bytes_) {
          serialized = "OVERSIZE\n";
          if (serialized.size() > rotate_bytes_) {
            serialized.assign(static_cast<size_t>(rotate_bytes_), '!');
            serialized.back() = '\n';
          }
        }
      }

      if (state.bytes > 0 &&
          (state.bytes >= rotate_bytes_ ||
           serialized.size() > rotate_bytes_ - state.bytes))
        Rotate(state);
      if (!state.file) {
        DFKV_LOG_INFO(line);
        return;
      }
      if (!serialized.empty())
        std::fwrite(serialized.data(), 1, serialized.size(), state.file);
      state.bytes += serialized.size();
    };

    if (!drop_line.empty()) write_line(drop_line);
    for (const std::string& line : batch.lines) write_line(line);
  }

  void Run() {
    NameThisThread("dfkv-pernode");
    for (;;) {
      Batch batch;
      bool flush = false;
      uint64_t reported_drops = 0;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] {
          return stopping_ || flush_requested_ ||
                 (!paused_for_test_ && !queue_.empty());
        });
        if (queue_.empty()) {
          if (flush_requested_) {
            flush = true;
            writing_ = true;
          } else if (stopping_) {
            break;
          } else {
            continue;
          }
        } else {
          batch = std::move(queue_.front());
          queue_.pop_front();
          batch.dropped_before = dropped_pending_;
          reported_drops = batch.dropped_before;
          writing_ = true;
        }
      }
      if (flush) {
        for (auto& entry : files_)
          if (entry.second.file) std::fflush(entry.second.file);
      } else {
        Write(std::move(batch));
      }
      {
        std::lock_guard<std::mutex> lock(mu_);
        if (!flush && reported_drops != 0)
          dropped_pending_ -= std::min(dropped_pending_, reported_drops);
        writing_ = false;
        if (flush) flush_requested_ = false;
        if (queue_.empty() && !flush_requested_) drained_.notify_all();
      }
    }
    // The writer owns every FILE* through final flush/close; producer and
    // teardown threads only signal and join, never perform sink file I/O.
    for (auto& entry : files_) {
      if (entry.second.file) {
        std::fclose(entry.second.file);
        entry.second.file = nullptr;
      }
    }
  }

  const size_t capacity_;
  const uint64_t rotate_bytes_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable drained_;
  std::deque<Batch> queue_;
  std::map<std::string, FileState> files_;
  bool writing_ = false;
  bool stopping_ = false;
  bool flush_requested_ = false;
  bool paused_for_test_ = false;
  size_t high_water_ = 0;
  uint64_t dropped_total_ = 0;
  uint64_t dropped_pending_ = 0;
  std::thread worker_;
};

void PernodeWrite(const char* fname, std::vector<std::string> lines) {
  PernodeSink::Instance().Enqueue(fname, std::move(lines));
}

// op: 0=get, 1=put, 2=exist. issued_keys counts transport items only.
// Client-local routing, health, resource, and topology failures remain logical
// failures and carry their reason/status separately from transport attempts.
void EmitPernode(int op, const char* label, const char* fname,
                 const std::map<std::string, PernodeStat>& per,
                 const std::map<std::string, std::string>& addr2name,
                 uint64_t total_wall_us, uint64_t logical_keys,
                 uint64_t logical_hits, const PernodeStat& unrouted) {
  std::vector<const std::pair<const std::string, PernodeStat>*> rows;
  rows.reserve(per.size());
  for (const auto& it : per) rows.push_back(&it);
  std::sort(rows.begin(), rows.end(), [](const auto* left, const auto* right) {
    return left->second.wall_us > right->second.wall_us;
  });
  uint64_t issued = 0, bytes = 0, hits = 0, failed = 0;
  uint64_t min_b = std::numeric_limits<uint64_t>::max(), max_b = 0;
  uint32_t issued_status_mask = 0, issued_fail_status_mask = 0;
  PernodeStat nonissued = unrouted;
  for (const auto* row : rows) {
    const PernodeStat& stat = row->second;
    issued += stat.issued_keys;
    bytes += stat.bytes;
    hits += stat.hit;
    failed += stat.fail;
    issued_status_mask |= stat.status_mask;
    issued_fail_status_mask |= stat.fail_status_mask;
    MergePernodeNonIssued(&nonissued, stat);
    if (stat.max_b > 0) {
      min_b = std::min(min_b, stat.min_b);
      max_b = std::max(max_b, stat.max_b);
    }
  }
  if (max_b == 0) min_b = 0;

  std::vector<std::string> lines;
  lines.reserve(rows.size() + 1);
  const uint64_t mbps10 =
      total_wall_us ? (bytes * 10 + total_wall_us / 2) / total_wall_us : 0;
  const uint64_t avg_value = hits ? (bytes + hits / 2) / hits : 0;
  const uint32_t logical_failure_status_mask =
      issued_fail_status_mask | nonissued.nonissued_status_mask;
  std::string total =
      std::string("pernode-") + label + " TOTAL: " + label +
      "_ms=" + PernodeFmt1((total_wall_us + 50) / 100) +
      " logical_keys=" + std::to_string(logical_keys) +
      " issued_keys=" + std::to_string(issued) +
      " nonissued_failures=" + std::to_string(nonissued.nonissued_fail) +
      " servers=" + std::to_string(rows.size()) +
      " bytes=" + std::to_string(bytes) +
      " avg_success_bytes=" + std::to_string(avg_value) +
      " min_bytes=" + std::to_string(min_b) +
      " max_bytes=" + std::to_string(max_b) +
      " MB/s=" + PernodeFmt1(mbps10) +
      " failed_issued_keys=" + std::to_string(failed) +
      " status_mask=" + std::to_string(issued_status_mask) +
      " statuses=" + PernodeStatusList(issued_status_mask) +
      " failure_status_mask=" + std::to_string(issued_fail_status_mask) +
      " logical_failure_status_mask=" +
      std::to_string(logical_failure_status_mask) +
      " nonissued_status_mask=" +
      std::to_string(nonissued.nonissued_status_mask) +
      " nonissued_statuses=" +
      PernodeStatusList(nonissued.nonissued_status_mask) +
      " invalid_input=" + std::to_string(nonissued.invalid_input) +
      " no_route=" + std::to_string(nonissued.no_route) +
      " health_cooldown=" + std::to_string(nonissued.health_cooldown) +
      " resource_exhausted=" +
      std::to_string(nonissued.resource_exhausted) +
      " no_compatible_rail=" +
      std::to_string(nonissued.no_compatible_rail) +
      " nonissued_reasons=" + PernodeNonIssuedReasons(nonissued);
  if (op == 0)
    total += " logical_hits=" + std::to_string(logical_hits) +
             " logical_not_hit=" +
             std::to_string(logical_keys - logical_hits) +
             " missed_issued_keys=" +
             std::to_string(issued - hits - failed);
  else if (op == 2)
    total += " logical_present=" + std::to_string(logical_hits) +
             " logical_not_present=" +
             std::to_string(logical_keys - logical_hits) +
             " absent_issued_keys=" +
             std::to_string(issued - hits - failed);
  else
    total += " logical_success=" + std::to_string(logical_hits) +
             " logical_fail=" +
             std::to_string(logical_keys - logical_hits) +
             " successful_issued_keys=" + std::to_string(hits) +
             " preroute_fail=" +
             std::to_string(nonissued.invalid_input + nonissued.no_route);
  if (nonissued.nonissued_fail > 0)
    total += " first_nonissued_fail=" + nonissued.first_fail;
  lines.push_back(std::move(total));

  for (const auto* row : rows) {
    const PernodeStat& stat = row->second;
    const uint64_t node_mbps10 =
        stat.wall_us
            ? (stat.bytes * 10 + stat.wall_us / 2) / stat.wall_us
            : 0;
    const uint64_t avg_value =
        stat.hit ? (stat.bytes + stat.hit / 2) / stat.hit : 0;
    const uint64_t us_per_key =
        stat.issued_keys
            ? (stat.busy_us + stat.issued_keys / 2) / stat.issued_keys
            : 0;
    const uint64_t node_min = stat.max_b ? stat.min_b : 0;
    auto name = addr2name.find(row->first);
    const uint32_t node_logical_failure_status_mask =
        stat.fail_status_mask | stat.nonissued_status_mask;
    std::string line =
        std::string("pernode-") + label + ": node=" + row->first +
        (name != addr2name.end() ? (" name=" + name->second) : std::string()) +
        (stat.dev.empty() ? std::string() : (" dev=" + stat.dev)) +
        " issued_keys=" + std::to_string(stat.issued_keys) +
        " nonissued_failures=" + std::to_string(stat.nonissued_fail) +
        " bytes=" + std::to_string(stat.bytes) +
        " avg_success_bytes=" + std::to_string(avg_value) +
        " min_bytes=" + std::to_string(node_min) +
        " max_bytes=" + std::to_string(stat.max_b) +
        " shards=" + std::to_string(stat.shards) +
        " wall_us=" + std::to_string(stat.wall_us) +
        " MB/s=" + PernodeFmt1(node_mbps10) +
        " avg_us/issued_key=" + std::to_string(us_per_key) +
        " failed_issued_keys=" + std::to_string(stat.fail) +
        " status_mask=" + std::to_string(stat.status_mask) +
        " statuses=" + PernodeStatusList(stat.status_mask) +
        " failure_status_mask=" +
        std::to_string(stat.fail_status_mask) +
        " logical_failure_status_mask=" +
        std::to_string(node_logical_failure_status_mask) +
        " nonissued_status_mask=" +
        std::to_string(stat.nonissued_status_mask) +
        " nonissued_statuses=" +
        PernodeStatusList(stat.nonissued_status_mask) +
        " invalid_input=" + std::to_string(stat.invalid_input) +
        " no_route=" + std::to_string(stat.no_route) +
        " health_cooldown=" + std::to_string(stat.health_cooldown) +
        " resource_exhausted=" +
        std::to_string(stat.resource_exhausted) +
        " no_compatible_rail=" +
        std::to_string(stat.no_compatible_rail) +
        " nonissued_reasons=" + PernodeNonIssuedReasons(stat);
    if (op == 0)
      line += " hit_issued_keys=" + std::to_string(stat.hit) +
              " missed_issued_keys=" +
              std::to_string(stat.issued_keys - stat.hit - stat.fail);
    else if (op == 2)
      line += " present_issued_keys=" + std::to_string(stat.hit) +
              " absent_issued_keys=" +
              std::to_string(stat.issued_keys - stat.hit - stat.fail);
    else
      line += " successful_issued_keys=" + std::to_string(stat.hit);
    if (stat.fail > 0 || stat.nonissued_fail > 0)
      line += " first_fail=" + stat.first_fail;
    lines.push_back(std::move(line));
  }
  PernodeWrite(fname, std::move(lines));
}

}  // namespace
namespace testing {
void DrainPernodeSink() { PernodeSink::Instance().DrainForTest(); }
size_t PernodeQueueCapacity() { return PernodeSink::Instance().capacity(); }
uint64_t PernodeRotateBytes() {
  return PernodeSink::Instance().rotate_bytes();
}
size_t PernodeQueueHighWater() { return PernodeSink::Instance().high_water(); }
uint64_t PernodeQueueDropped() { return PernodeSink::Instance().dropped(); }
uint64_t PernodeQueuePendingDropped() {
  return PernodeSink::Instance().pending_dropped();
}
void PausePernodeSink(bool paused) {
  PernodeSink::Instance().PauseForTest(paused);
}
void EnqueuePernodeForTest(const char* fname,
                           std::vector<std::string> lines) {
  PernodeSink::Instance().Enqueue(fname, std::move(lines));
}
std::string PernodeKeyFingerprintForTest(const std::string& key) {
  return PernodeKeyFp(key);
}
}  // namespace testing


std::vector<bool> KVClient::BatchExistDirect(
    const std::vector<std::string>& keys, std::vector<Status>* statuses) {
  const size_t N = keys.size();
  std::vector<char> e(N, 0);
  std::vector<Status> raw(N, Status::kInvalid);
  const bool pernode_on = PernodeOn();
  std::chrono::steady_clock::time_point call_t0;
  if (pernode_on) call_t0 = std::chrono::steady_clock::now();
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
  std::unique_ptr<PernodeStat> unrouted;
  if (pernode_on) unrouted = std::make_unique<PernodeStat>();
  std::map<std::string, std::vector<size_t>> by_node;
  for (size_t i = 0; i < N; ++i) {
    std::string node = Route(keys[i]);
    if (node.empty()) {
      if (pernode_on)
        RecordPernodeNonIssued(unrouted.get(),
                               PernodeNonIssuedReason::kNoRoute, 1, keys[i]);
      continue;
    }
    by_node[node].push_back(i);
  }
  std::vector<std::pair<std::string, std::vector<size_t>>> groups =
      ShardReadGroups(std::vector<std::pair<std::string, std::vector<size_t>>>(
          by_node.begin(), by_node.end()));
  std::vector<PernodeStat> gstat(pernode_on ? groups.size() : 0);
  RunReadScheduled(groups.size(), [&](size_t g) {
    const std::string& node = groups[g].first;
    const std::vector<size_t>& idx = groups[g].second;
    uint64_t now = NowMs();
    if (!health_.Healthy(node, now)) {
      if (pernode_on)
        RecordPernodeNonIssued(
            &gstat[g], PernodeNonIssuedReason::kHealthCooldown, idx.size(),
            keys[idx.front()]);
      return;
    }
    std::vector<BlockKey> bkeys;
    bkeys.reserve(idx.size());
    for (size_t k : idx) bkeys.push_back(ToBlockKey(key_namespace_, keys[k]));
    std::vector<char> ex;
    std::string shard_dev;
    std::chrono::steady_clock::time_point pn_t0;
    if (pernode_on) pn_t0 = std::chrono::steady_clock::now();
    std::vector<Status> sts =
        t_->ExistMany(node, bkeys, &ex, pernode_on ? &shard_dev : nullptr);
    if (pernode_on) {
      const uint64_t us = static_cast<uint64_t>(
          std::chrono::duration<double, std::micro>(
              std::chrono::steady_clock::now() - pn_t0).count());
      uint64_t issuedn = 0, present = 0, failn = 0;
      uint32_t status_mask = 0, fail_mask = 0;
      std::string first_fail;
      for (size_t m = 0; m < idx.size(); ++m) {
        const Status status =
            m < sts.size() ? sts[m] : Status::kInvalid;
        if (status == Status::kResourceExhausted ||
            status == Status::kNoCompatibleRail) {
          RecordPernodeNonIssued(
              &gstat[g],
              status == Status::kResourceExhausted
                  ? PernodeNonIssuedReason::kResourceExhausted
                  : PernodeNonIssuedReason::kNoCompatibleRail,
              1, keys[idx[m]]);
          continue;
        }
        ++issuedn;
        status_mask |= PernodeStatusBit(status);
        if (status == Status::kOk) {
          if (m < ex.size() && ex[m]) ++present;
        } else if (status != Status::kNotFound) {
          ++failn;
          fail_mask |= PernodeStatusBit(status);
          if (first_fail.empty())
            first_fail = PernodeKeyFp(keys[idx[m]]);
        }
      }
      gstat[g].issued_keys = issuedn;
      gstat[g].busy_us = issuedn == 0 ? 0 : us;
      gstat[g].hit = present;
      gstat[g].fail = failn;
      gstat[g].status_mask = status_mask;
      gstat[g].fail_status_mask = fail_mask;
      if (gstat[g].first_fail.empty())
        gstat[g].first_fail = std::move(first_fail);
      gstat[g].dev = std::move(shard_dev);
    }
    bool resp = false, ioerr = false;
    // kInvalid (oversize/per-item guard) is neither: it must not clear the
    // peer cooldown (resp) nor trip MarkBad (ioerr).
    for (Status status : sts) {
      if (status == Status::kIOError)
        ioerr = true;
      else if (status == Status::kOk || status == Status::kNotFound)
        resp = true;
    }
    if (resp)
      health_.MarkGood(node, NowMs());
    else if (ioerr)
      health_.MarkBad(node, NowMs());
    for (size_t m = 0; m < idx.size(); ++m) {
      if (m < sts.size()) raw[idx[m]] = sts[m];
      e[idx[m]] =
          (m < sts.size() && sts[m] == Status::kOk &&
           m < ex.size() && ex[m])
              ? 1
              : 0;
    }
  });
  if (pernode_on) {
    std::map<std::string, PernodeStat> per;
    for (size_t g = 0; g < groups.size(); ++g) {
      if (gstat[g].issued_keys == 0 && gstat[g].nonissued_fail == 0) continue;
      PernodeStat& a = per[groups[g].first];
      a.issued_keys += gstat[g].issued_keys;
      a.busy_us += gstat[g].busy_us;
      if (gstat[g].busy_us > a.wall_us) {
        a.wall_us = gstat[g].busy_us;
        a.dev = gstat[g].dev;
      }
      if (gstat[g].issued_keys > 0) a.shards += 1;
      a.hit += gstat[g].hit;
      a.fail += gstat[g].fail;
      a.status_mask |= gstat[g].status_mask;
      a.fail_status_mask |= gstat[g].fail_status_mask;
      MergePernodeNonIssued(&a, gstat[g]);
      if (a.first_fail.empty() && !gstat[g].first_fail.empty())
        a.first_fail = gstat[g].first_fail;
    }
    std::map<std::string, std::string> addr2name;
    {
      std::lock_guard<std::mutex> lk(ring_mu_);
      for (const auto& kv : addr_) addr2name[kv.second] = kv.first;
    }
    const uint64_t total_wall_us = static_cast<uint64_t>(
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - call_t0).count());
    uint64_t logical_present = 0;
    for (char present : e) logical_present += present != 0;
    EmitPernode(2, "exist", "dfkv_client_exist.log", per, addr2name,
                total_wall_us, N, logical_present, *unrouted);
  }
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

  const bool pernode_on = PernodeOn();
  std::unique_ptr<PernodeStat> unrouted;
  if (pernode_on) unrouted = std::make_unique<PernodeStat>();
  std::map<std::string, std::vector<size_t>> by_node;
  for (size_t i = 0; i < N; ++i) {
    if (over[i]) {
      if (pernode_on)
        RecordPernodeNonIssued(
            unrouted.get(), PernodeNonIssuedReason::kInvalidInput, 1,
            items[i].key);
      continue;
    }
    std::string node = Route(items[i].key);
    if (node.empty()) {
      if (pernode_on)
        RecordPernodeNonIssued(unrouted.get(),
                               PernodeNonIssuedReason::kNoRoute, 1,
                               items[i].key);
      continue;
    }
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
  std::vector<PernodeStat> gstat(pernode_on ? groups.size() : 0);
  RunParallel(groups.size(), BatchWorkers(groups.size()), [&](size_t g) {
    const std::string& node = groups[g].first;
    const std::vector<size_t>& idx = groups[g].second;
    uint64_t now = NowMs();
    if (!health_.Healthy(node, now)) {
      if (pernode_on)
        RecordPernodeNonIssued(
            &gstat[g], PernodeNonIssuedReason::kHealthCooldown, idx.size(),
            items[idx.front()].key);
      return;
    }
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
    std::string shard_dev;
    std::chrono::steady_clock::time_point pn_t0;
    if (pernode_on) pn_t0 = std::chrono::steady_clock::now();
    std::vector<Status> sts =
        t_->CacheFromMulti(node, srcs, pernode_on ? &shard_dev : nullptr);
    if (pernode_on) {
      const uint64_t us = static_cast<uint64_t>(
          std::chrono::duration<double, std::micro>(
              std::chrono::steady_clock::now() - pn_t0).count());
      uint64_t issuedn = 0, vb = 0, hits = 0, failn = 0,
               mn = std::numeric_limits<uint64_t>::max(), mx = 0;
      uint32_t status_mask = 0, fail_mask = 0;
      std::string first_fail;
      for (size_t m = 0; m < idx.size(); ++m) {
        const Status status =
            m < sts.size() ? sts[m] : Status::kInvalid;
        if (status == Status::kResourceExhausted ||
            status == Status::kNoCompatibleRail) {
          RecordPernodeNonIssued(
              &gstat[g],
              status == Status::kResourceExhausted
                  ? PernodeNonIssuedReason::kResourceExhausted
                  : PernodeNonIssuedReason::kNoCompatibleRail,
              1, items[idx[m]].key);
          continue;
        }
        ++issuedn;
        status_mask |= PernodeStatusBit(status);
        if (status == Status::kOk) {
          ++hits;
          const uint64_t value_bytes = total_bytes[idx[m]];
          vb += value_bytes;
          mn = std::min(mn, value_bytes);
          mx = std::max(mx, value_bytes);
        } else {
          ++failn;
          fail_mask |= PernodeStatusBit(status);
          if (first_fail.empty())
            first_fail = PernodeKeyFp(items[idx[m]].key);
        }
      }
      gstat[g].issued_keys = issuedn;
      gstat[g].bytes = vb;
      gstat[g].busy_us = issuedn == 0 ? 0 : us;
      gstat[g].min_b = mn;
      gstat[g].max_b = mx;
      gstat[g].hit = hits;
      gstat[g].fail = failn;
      gstat[g].status_mask = status_mask;
      gstat[g].fail_status_mask = fail_mask;
      if (gstat[g].first_fail.empty())
        gstat[g].first_fail = std::move(first_fail);
      gstat[g].dev = std::move(shard_dev);
    }
    for (size_t m = 0; m < idx.size(); ++m) {
      ok[idx[m]] =
          (m < sts.size() && sts[m] == Status::kOk) ? 1 : 0;
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
  if (pernode_on) {
    std::map<std::string, PernodeStat> per;
    for (size_t g = 0; g < groups.size(); ++g) {
      if (gstat[g].issued_keys == 0 && gstat[g].nonissued_fail == 0) continue;
      PernodeStat& a = per[groups[g].first];
      a.issued_keys += gstat[g].issued_keys;
      a.bytes += gstat[g].bytes;
      a.busy_us += gstat[g].busy_us;
      if (gstat[g].busy_us > a.wall_us) {
        a.wall_us = gstat[g].busy_us;
        a.dev = gstat[g].dev;
      }
      if (gstat[g].issued_keys > 0) a.shards += 1;
      a.hit += gstat[g].hit;
      a.fail += gstat[g].fail;
      a.status_mask |= gstat[g].status_mask;
      a.fail_status_mask |= gstat[g].fail_status_mask;
      MergePernodeNonIssued(&a, gstat[g]);
      if (gstat[g].max_b > 0) {
        a.min_b = std::min(a.min_b, gstat[g].min_b);
        a.max_b = std::max(a.max_b, gstat[g].max_b);
      }
      if (a.first_fail.empty() && !gstat[g].first_fail.empty())
        a.first_fail = gstat[g].first_fail;
    }
    std::map<std::string, std::string> addr2name;
    {
      std::lock_guard<std::mutex> lk(ring_mu_);
      for (const auto& kv : addr_) addr2name[kv.second] = kv.first;
    }
    const uint64_t total_wall_us = static_cast<uint64_t>(
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - t0).count());
    uint64_t logical_success = 0;
    for (char success : ok) logical_success += success != 0;
    EmitPernode(1, "put", "dfkv_client_put.log", per, addr2name,
                total_wall_us, N, logical_success, *unrouted);
  }
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
  // Init the GPU rendezvous only once an all-device SG destination shows up:
  // its first pointer selects the primary context for the arena. Mixed and
  // host-only SG items use the ordinary transport publication path.
  std::vector<char> all_device(items.size(), 0);
  const void* hint = nullptr;
  for (size_t i = 0; i < items.size(); ++i) {
    bool all = !items[i].dsts.empty() &&
               items[i].dsts.size() == items[i].caps.size();
    for (size_t j = 0; all && j < items[i].dsts.size(); ++j) {
      if (ClassifyDestination(items[i].dsts[j], items[i].caps[j]) !=
          DestinationMemoryKind::kDevice) {
        all = false;
      }
    }
    all_device[i] = all ? 1 : 0;
    if (all && !hint) hint = items[i].dsts[0];
  }
  GpuNodeDedup* gd = hint ? GpuDedup(hint) : nullptr;
  if (!gd || items.empty()) {
    std::vector<size_t> lengths;
    auto result = BatchGetAutoSgDirect(items, &lengths);
    uint64_t bytes = 0;
    for (size_t length : lengths)
      AddMetricBytes(length, &bytes);
    if (out_lens) *out_lens = std::move(lengths);
    return RecordBatch(OpMetrics::kGet, t0, std::move(result), bytes);
  }
  // Same-host rendezvous for all-device GPU destinations (phase 2b, see
  // node_dedup_gpu.h): the vLLM connector's SG lands in device memory, where
  // lockstep TP ranks re-fetch identical TP-replicated KV. Mixed SG items use
  // the ordinary staged publication path instead.
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
    // Eligible = well-shaped AND entirely device memory. Mixed SG must bypass
    // GPU rendezvous because its CUDA IPC copier cannot publish host segments.
    const bool eligible = !items[i].key.empty() && !items[i].dsts.empty() &&
                          items[i].dsts.size() == items[i].caps.size() &&
                          totals_valid[i] && all_device[i];
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

  // Per-node (ip:port) GET timing (DFKV_CLIENT_PERNODE_STATS=1, off by default):
  // accumulate keys/bytes/time by target server across all passes, emit one line
  // per node at return to spot slow nodes. busy_us sums per-shard service time
  // (=> avg_us/key); wall_us takes the max shard time — shards run in parallel,
  // so it approximates the node's wall time, the honest basis for MB/s (summing
  // parallel shard time would understate throughput).
  const bool pernode_on = PernodeOn();
  std::unique_ptr<std::map<std::string, PernodeStat>> pernode;
  std::unique_ptr<std::vector<char>> nonissued_recorded;
  std::chrono::steady_clock::time_point call_t0;
  if (pernode_on) {
    pernode = std::make_unique<std::map<std::string, PernodeStat>>();
    nonissued_recorded = std::make_unique<std::vector<char>>(N, 0);
    call_t0 = std::chrono::steady_clock::now();
  }

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
  std::unique_ptr<PernodeStat> unrouted;
  if (pernode_on) unrouted = std::make_unique<PernodeStat>();

  // Group by (node, total_cap): a group shares the Range length (= sum of caps)
  // so the existing RangeInto windowing/pipelining applies unchanged.
  auto run_pass = [&](const std::vector<size_t>& todo) {
    std::map<std::pair<std::string, size_t>, std::vector<size_t>> by;
    for (size_t i : todo) {
      if (over[i]) {
        if (pernode_on)
          RecordPernodeNonIssued(
              unrouted.get(), PernodeNonIssuedReason::kInvalidInput, 1,
              items[i].key);
        continue;
      }
      std::string node = Route(items[i].key);
      if (node.empty()) {
        if (pernode_on && !(*nonissued_recorded)[i]) {
          RecordPernodeNonIssued(
              unrouted.get(), PernodeNonIssuedReason::kNoRoute, 1,
              items[i].key);
          (*nonissued_recorded)[i] = 1;
        }
        continue;
      }
      by[{node, total_caps[i]}].push_back(i);
    }
    std::vector<std::pair<std::pair<std::string, size_t>,
                          std::vector<size_t>>>
        groups = ShardReadGroups(
            std::vector<std::pair<std::pair<std::string, size_t>,
                                  std::vector<size_t>>>(by.begin(), by.end()));
    std::vector<PernodeStat> gstat(pernode_on ? groups.size() : 0);
    RunReadScheduled(groups.size(), [&](size_t g) {
      const std::string& node = groups[g].first.first;
      const std::vector<size_t>& idx = groups[g].second;
      uint64_t now = NowMs();
      if (!health_.Healthy(node, now)) {
        if (pernode_on) {
          size_t fresh = 0;
          size_t first_fresh = 0;
          for (size_t item_index : idx) {
            if ((*nonissued_recorded)[item_index]) continue;
            if (fresh == 0) first_fresh = item_index;
            (*nonissued_recorded)[item_index] = 1;
            ++fresh;
          }
          if (fresh != 0)
            RecordPernodeNonIssued(
                &gstat[g], PernodeNonIssuedReason::kHealthCooldown, fresh,
                items[first_fresh].key);
        }
        return;
      }
      std::vector<BlockKey> keys;
      std::vector<RangeDstMulti> dsts;
      keys.reserve(idx.size());
      dsts.reserve(idx.size());
      for (size_t k : idx) {
        keys.push_back(ToBlockKey(key_namespace_, items[k].key));
        RangeDstMulti d;
        d.payloads.reserve(items[k].dsts.size());
        for (size_t j = 0; j < items[k].dsts.size(); ++j)
          d.payloads.emplace_back(
              items[k].dsts[j], items[k].caps[j],
              ClassifyDestination(items[k].dsts[j], items[k].caps[j]));
        dsts.push_back(std::move(d));
      }
      std::vector<size_t> value_lens;
      std::string shard_dev;
      std::chrono::steady_clock::time_point pn_t0;
      if (pernode_on) pn_t0 = std::chrono::steady_clock::now();
      std::vector<Status> sts = t_->RangeIntoMulti(
          node, keys, dsts, &value_lens, pernode_on ? &shard_dev : nullptr);
      if (pernode_on) {
        const uint64_t us = static_cast<uint64_t>(
            std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - pn_t0).count());
        uint64_t issuedn = 0, vb = 0, hits = 0, failn = 0,
                 mn = std::numeric_limits<uint64_t>::max(), mx = 0;
        uint32_t status_mask = 0, fail_mask = 0;
        std::string first_fail;
        const size_t cap = groups[g].first.second;
        for (size_t m = 0; m < idx.size(); ++m) {
          Status status = m < sts.size() ? sts[m] : Status::kInvalid;
          if (status == Status::kOk &&
              (m >= value_lens.size() || value_lens[m] > cap))
            status = Status::kInvalid;
          if (status == Status::kResourceExhausted ||
              status == Status::kNoCompatibleRail) {
            RecordPernodeNonIssued(
                &gstat[g],
                status == Status::kResourceExhausted
                    ? PernodeNonIssuedReason::kResourceExhausted
                    : PernodeNonIssuedReason::kNoCompatibleRail,
                1, items[idx[m]].key);
            continue;
          }
          ++issuedn;
          status_mask |= PernodeStatusBit(status);
          if (status == Status::kOk) {
            ++hits;
            const uint64_t value_bytes = value_lens[m];
            vb += value_bytes;
            mn = std::min(mn, value_bytes);
            mx = std::max(mx, value_bytes);
          } else if (status != Status::kNotFound) {
            ++failn;
            fail_mask |= PernodeStatusBit(status);
            if (first_fail.empty())
              first_fail = PernodeKeyFp(items[idx[m]].key);
          }
        }
        gstat[g].issued_keys = issuedn;
        gstat[g].bytes = vb;
        gstat[g].busy_us = issuedn == 0 ? 0 : us;
        gstat[g].min_b = mn;
        gstat[g].max_b = mx;
        gstat[g].hit = hits;
        gstat[g].fail = failn;
        gstat[g].status_mask = status_mask;
        gstat[g].fail_status_mask = fail_mask;
        if (gstat[g].first_fail.empty())
          gstat[g].first_fail = std::move(first_fail);
        gstat[g].dev = std::move(shard_dev);
      }
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
        const size_t cap = groups[g].first.second;
        if (m >= sts.size() || sts[m] != Status::kOk ||
            m >= value_lens.size() || value_lens[m] > cap)
          continue;
        lens[idx[m]] = value_lens[m];
        hit[idx[m]] = 1;
      }
    });
    if (pernode_on) {
      for (size_t g = 0; g < groups.size(); ++g) {
        if (gstat[g].issued_keys == 0 && gstat[g].nonissued_fail == 0) continue;
        PernodeStat& a = (*pernode)[groups[g].first.first];
        a.issued_keys += gstat[g].issued_keys;
        a.bytes += gstat[g].bytes;
        a.busy_us += gstat[g].busy_us;
        if (gstat[g].busy_us > a.wall_us) {
          a.wall_us = gstat[g].busy_us;
          a.dev = gstat[g].dev;
        }
        if (gstat[g].issued_keys > 0) a.shards += 1;
        a.hit += gstat[g].hit;
        a.fail += gstat[g].fail;
        a.status_mask |= gstat[g].status_mask;
        a.fail_status_mask |= gstat[g].fail_status_mask;
        MergePernodeNonIssued(&a, gstat[g]);
        if (gstat[g].max_b > 0) {
          a.min_b = std::min(a.min_b, gstat[g].min_b);
          a.max_b = std::max(a.max_b, gstat[g].max_b);
        }
        if (a.first_fail.empty() && !gstat[g].first_fail.empty())
          a.first_fail = gstat[g].first_fail;
      }
    }
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
  if (pernode_on) {
    const uint64_t total_wall_us = static_cast<uint64_t>(
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - call_t0).count());
    std::map<std::string, std::string> addr2name;
    {
      std::lock_guard<std::mutex> lk(ring_mu_);
      for (const auto& kv : addr_) addr2name[kv.second] = kv.first;
    }
    uint64_t logical_hits = 0;
    for (char found : hit) logical_hits += found != 0;
    EmitPernode(0, "get", "dfkv_client_get.log", *pernode, addr2name,
                total_wall_us, N, logical_hits, *unrouted);
  }
  if (out_lens) *out_lens = std::move(lens);
  return std::vector<bool>(hit.begin(), hit.end());
}

}  // namespace dfkv
