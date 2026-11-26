#include "client/kv_client.h"

#include "common/raw_clock.h"

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
#include <exception>
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
#include "utils/env_parse.h"
#include "utils/sha256.h"
#include "utils/thread_name.h"
#include "utils/parallel_for.h"
#include "transport/transport_factory.h"
#include "common/config_dump.h"

namespace dfkv {

namespace {
// Set on this thread while a request is sent to the draining owners of its
// keys (see KVClient::full_ring_): Route() then names a key's draining
// full-ring owner, or nothing for a key that has none, and the bodies do not
// look for drainers again. Requests route on the thread that issues them.
thread_local bool t_drain_pass = false;
// Set while requests go to one absent member (its REMOVEs and RETAINs, and
// the retries of the REMOVEs it missed): Route() names it for every key.
thread_local const std::string* t_route_to = nullptr;
struct RouteTo {
  explicit RouteTo(const std::string& addr) { t_route_to = &addr; }
  ~RouteTo() { t_route_to = nullptr; }
  RouteTo(const RouteTo&) = delete;
  RouteTo& operator=(const RouteTo&) = delete;
};
// Runs a log statement once per process. Not std::call_once: an exception
// unwinding through it (pthread_once) aborts the process. One that throws
// (an allocation failing) only loses the line, and it is not tried again.
struct LogOnce {
  std::mutex mu;
  bool done = false;
};
template <typename Fn>
void RunLogOnce(LogOnce& once, Fn fn) noexcept {
  try {
    std::lock_guard<std::mutex> lk(once.mu);
    if (once.done) return;
    once.done = true;
    fn();
  } catch (...) {
  }
}

struct DrainPass {
  DrainPass() { t_drain_pass = true; }
  ~DrainPass() { t_drain_pass = false; }
  DrainPass(const DrainPass&) = delete;
  DrainPass& operator=(const DrainPass&) = delete;
};

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
  envparse::ReportUnlessPlain(name, 1, std::numeric_limits<uint64_t>::max(),
                              std::to_string(out));
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
  envparse::ReportUnlessPlain(name, 0, INT_MAX, std::to_string(out));
  config_dump::Record(name, std::to_string(out),
                      from_env ? config_dump::Source::kEnv
                               : config_dump::Source::kDefault);
  return out;
}

// Claimed rendezvous slots must not outlive an exception (bad_alloc, say)
// thrown before their publish or abort: such a slot stays FETCHING, and every
// process of the host waits on the key until the takeover age. On unwinding,
// each claimed slot is aborted; an abort of a slot already published or
// aborted is a no-op.
class AbortClaimsOnThrow {
 public:
  explicit AbortClaimsOnThrow(std::function<void()> abort_all)
      : abort_all_(std::move(abort_all)),
        uncaught_(std::uncaught_exceptions()) {}
  ~AbortClaimsOnThrow() {
    if (std::uncaught_exceptions() <= uncaught_) return;
    try {
      abort_all_();
    } catch (...) {  // already unwinding: nothing more to do
    }
  }
  AbortClaimsOnThrow(const AbortClaimsOnThrow&) = delete;
  AbortClaimsOnThrow& operator=(const AbortClaimsOnThrow&) = delete;

 private:
  std::function<void()> abort_all_;
  int uncaught_;
};

// Env-reading wrapper over ShardGroups (client/group_shard.h): resolve the
// DFKV_READ_SHARD_KEYS / DFKV_READ_MAX_CONNS knobs once (recorded into the config
// dump via EnvSize) and split each per-node group into up to DFKV_READ_MAX_CONNS
// shards so multiple connections drive one node concurrently — the mechanism that
// breaks the single-connection ~166 MB/s/conn key-by-key drain on a single- or
// few-node ring. Used by the read (BatchGet / SG GET) and write (BatchPut, BatchPutSg)
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
      block_key_(key_namespace_),
      namespace_hash_(block_key_("").digest_hi),
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
  {
    // Absent grace (MDS discovery only): at most 600 s.
    constexpr int kMaxAbsentGraceMs = 600000;
    int grace = EnvNonnegativeInt("DFKV_RING_ABSENT_GRACE_MS", 0);
    if (grace > kMaxAbsentGraceMs) {
      DFKV_LOG_WARN("DFKV_RING_ABSENT_GRACE_MS=" + std::to_string(grace) +
                    " is above the cap; using " +
                    std::to_string(kMaxAbsentGraceMs));
      grace = kMaxAbsentGraceMs;
    }
    absent_grace_ms_ = static_cast<uint64_t>(grace);
    absent_remove_cap_ = EnvSize("DFKV_RING_ABSENT_REMOVE_QUEUE", 4096);
  }
  {
    // Cooldown reroute (C). Fan-out and the registry marker come with it.
    const int reroute = EnvNonnegativeInt("DFKV_RING_COOLDOWN_REROUTE", 0);
    if (reroute > 1)
      DFKV_LOG_WARN("DFKV_RING_COOLDOWN_REROUTE=" + std::to_string(reroute) +
                    " is not 0 or 1; taken as 1");
    reroute_on_ = reroute != 0;
    fanout_on_ = reroute_on_;
    const auto clamp = [](const char* name, int v, int lo, int hi) {
      if (v >= lo && v <= hi) return v;
      const int c = v < lo ? lo : hi;
      DFKV_LOG_WARN(std::string(name) + "=" + std::to_string(v) +
                    " is outside " + std::to_string(lo) + "-" +
                    std::to_string(hi) + "; using " + std::to_string(c));
      return c;
    };
    fanout_ = static_cast<size_t>(clamp(
        "DFKV_RING_REMOVE_FANOUT",
        EnvNonnegativeInt("DFKV_RING_REMOVE_FANOUT", 3), 1, 8));
    reroute_cap_ = static_cast<size_t>(clamp(
        "DFKV_RING_REROUTE_KEYS",
        EnvNonnegativeInt("DFKV_RING_REROUTE_KEYS", 65536), 1024, 16777216));
    reroute_close_ms_ = static_cast<uint64_t>(clamp(
        "DFKV_RING_REROUTE_CLOSE_MS",
        EnvNonnegativeInt("DFKV_RING_REROUTE_CLOSE_MS", 2000), 0, 60000));
    // Previous-owner read-back (B). It needs the fan-out of at least three
    // successors (§3.4: safe for c <= F - 2 view changes).
    readback_ms_ = static_cast<uint64_t>(clamp(
        "DFKV_RING_READBACK_MS",
        EnvNonnegativeInt("DFKV_RING_READBACK_MS", 0), 0, 600000));
    readback_on_ = readback_ms_ > 0;
    if (readback_on_ && fanout_ < 3) {
      DFKV_LOG_WARN("DFKV_RING_READBACK_MS: read-back needs "
                    "DFKV_RING_REMOVE_FANOUT >= 3 (it is " +
                    std::to_string(fanout_) + "); read-back stays off");
      readback_on_ = false;
    }
    rewrite_on_ = EnvNonnegativeInt("DFKV_RING_READBACK_REWRITE", 0) != 0;
    if (rewrite_on_ && !readback_on_) {
      DFKV_LOG_WARN("DFKV_RING_READBACK_REWRITE needs the read-back "
                    "(DFKV_RING_READBACK_MS); it stays off");
      rewrite_on_ = false;
    }
    config_dump::RecordResolved("DFKV_RING_READBACK_REWRITE",
                                rewrite_on_ ? "1" : "0");
    fanout_on_ = reroute_on_ || readback_on_;
    if (fanout_on_) {
      recent_keep_ms_.store(33000, std::memory_order_relaxed);  // 11 polls of 3 s
      // The ring adopted above, before the switch was read, is the first
      // recent ring.
      std::lock_guard<std::mutex> lk(ring_mu_);
      std::map<std::string, int> weight;
      for (const auto& [id, w] : ring_.Nodes()) weight[id] = w;
      NoteRecentLocked(weight, addr_, true, NowMs());
    }
  }
  retain_reprobe_ms_ =
      static_cast<uint64_t>(EnvNonnegativeInt("DFKV_RETAIN_REPROBE_MS", 30000));
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
    // 0 is documented as auto (the default); it leaves the default in place.
    const size_t resolved = batch_concurrency_.load(std::memory_order_relaxed);
    envparse::ReportUnlessPlain(
        "DFKV_BATCH_CONCURRENCY", 0, std::numeric_limits<size_t>::max(),
        resolved == 0 ? std::string("0 (auto)") : std::to_string(resolved));
  }
  // Opt-in active per-peer latency probe. Off unless DFKV_PROBE_INTERVAL_MS>0,
  // so default behavior is unchanged (no background traffic).
  int probe_ms = 0;
  if (const char* pe = std::getenv("DFKV_PROBE_INTERVAL_MS")) probe_ms = std::atoi(pe);
  envparse::ReportUnlessPlain(
      "DFKV_PROBE_INTERVAL_MS", 0, INT_MAX,
      probe_ms > 0 ? std::to_string(probe_ms) : std::string("0 (no probe)"));
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
  static LogOnce dump_once;
  RunLogOnce(dump_once, [this] {
    namespace cd = dfkv::config_dump;
    cd::Record("key_namespace_digest",
               block_key_("").DigestHex(), cd::Source::kArg);
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
  // query CUDA about its registered host destination.
  std::unique_lock<std::shared_mutex> lock(registered_memory_mu_);
  // The kind decided here is remembered for the life of the pool, so it must
  // not be taken from a throttled probe that predates the host's CUDA setup.
  const CudaLib* cuda = CudaLib::Probe();
  const DestinationMemoryKind kind =
      cuda && cuda->IsDevicePtr(base) ? DestinationMemoryKind::kDevice
                                      : DestinationMemoryKind::kHost;
  if (kind == DestinationMemoryKind::kDevice) {
    // GPUDirect RDMA memory-ordering contract (design guide, "Synchronization
    // and Memory Ordering"): without ordering, a CUDA kernel launched right
    // after an RDMA completion may read the destination before its BAR writes
    // land (observed live as exact-hit loads with degraded generation). Arm
    // SYNC_MEMOPS where the driver accepts it (plain cudaMalloc allocations);
    // VMM/cuMemMap pools commonly reject it and use the load-path fence.
    static LogOnce sync_memops_unavailable;
    if (!cuda->SetSyncMemops(base)) {
      RunLogOnce(sync_memops_unavailable, [] {
        DFKV_LOG_WARN(
            "dfkv: cuPointerSetAttribute(SYNC_MEMOPS) rejected for a GPU pool "
            "(VMM/cuMemMap allocation); GPUDirect loads rely on the "
            "load-completion CUDA fence for memory ordering");
      });
    }
  }
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

void KVClient::AdoptRing(ConHash ring, std::map<std::string, std::string> addr,
                         ConHash full_ring, std::set<std::string> draining,
                         AbsentMap absent, std::vector<MemberInfo> placement,
                         ConHash absent_ring) {
  const size_t count = addr.size();
  std::string delta;
  std::string drained;
  std::string absent_ids;
  {
    std::lock_guard<std::mutex> lk(ring_mu_);
    delta = MembershipDelta(addr_, addr);  // vs the previously-adopted view
    if (draining != draining_) {
      drained = draining.empty() ? std::string("none") : std::string();
      for (const auto& id : draining)
        drained += (drained.empty() ? "" : ",") + id;
    }
    bool absent_changed = absent.size() != absent_.size();
    for (const auto& [id, member] : absent) {
      (void)member;
      if (absent_.count(id) == 0) absent_changed = true;
    }
    if (absent_changed) {
      absent_ids = absent.empty() ? std::string("none") : std::string();
      for (const auto& [id, member] : absent) {
        (void)member;
        absent_ids += (absent_ids.empty() ? "" : ",") + id;
      }
    }
    if (readback_on_) {
      bool changed = ring_.Nodes().size() != ring.Nodes().size();
      if (!changed) {
        std::map<std::string, int> a(ring_.Nodes().begin(), ring_.Nodes().end());
        std::map<std::string, int> b(ring.Nodes().begin(), ring.Nodes().end());
        changed = a != b;
      }
      if (changed) NoteWriteRingChangeLocked(ring_, addr_, NowMs());
    }
    ring_ = std::move(ring);
    addr_ = std::move(addr);
    full_ring_ = std::move(full_ring);
    draining_ = std::move(draining);
    absent_ = std::move(absent);
    absent_ring_ = std::move(absent_ring);
    placement_ = std::move(placement);
    if (fanout_on_) {
      std::map<std::string, int> weight;
      for (const auto& [id, w] : ring_.Nodes()) weight[id] = w;
      NoteRecentLocked(weight, addr_, true, NowMs());
    }
    any_draining_.store(!draining_.empty(), std::memory_order_relaxed);
    any_absent_.store(!absent_.empty(), std::memory_order_relaxed);
  }
  if (!drained.empty())
    DFKV_LOG_INFO("ring: draining members: " + drained +
                  " (no new writes; read after a miss at the new owner)");
  if (!absent_ids.empty())
    DFKV_LOG_INFO("ring: absent members: " + absent_ids +
                  " (vanished from the view; no new writes, REMOVE and "
                  "RETAIN to them too, until the grace passes)");
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
    const std::vector<MemberInfo>& topology, uint64_t generation,
    bool ring_held, const std::map<std::string, std::string>& keep) {
  std::map<std::string, std::string> current;
  for (const auto& member : topology)
    current[member.id] = member.ip + ":" + std::to_string(member.port);

  std::set<std::pair<std::string, std::string>> retired;
  std::map<std::string, std::string> live = current;
  {
    std::lock_guard<std::mutex> lock(ring_mu_);
    for (const auto& peer : published_peers_) retired.insert(peer);
    for (const auto& peer : addr_) retired.insert(peer);
    for (const auto& peer : current) retired.erase(peer);
    if (ring_held) {
      // Keep what the transport holds for the ring members this view omits
      // (an identity the view moves to another address is not omitted).
      for (const auto& peer : addr_) {
        if (current.count(peer.first) != 0) continue;
        retired.erase(peer);
        live.insert(peer);
      }
    }
    // Absent members are still asked after a miss (absent_).
    for (const auto& peer : keep) {
      if (current.count(peer.first) != 0) continue;
      retired.erase(peer);
      live.insert(peer);
    }
    published_peers_ = live;
    topology_generation_ = generation;
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
    // Not the epoch of the whole view: that changes when any member does,
    // and the transport would then drop every peer's pooled connections for
    // a change that concerns one of them.
    peer.generation = MemberTopologyEpoch(member);
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
  live_peer_ids.reserve(live.size());
  for (const auto& [peer_id, address] : live) {
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
  std::vector<MemberInfo> placement;
  placement.reserve(members.size());
  for (const auto& m : members)
    if (m.RingEligible()) placement.push_back(m);
  std::vector<std::string> returned;
  AbsentMap absent;
  if (absent_grace_ms_ > 0) absent = NextAbsent(placement, NowMs(), &returned);
  AdoptPlacement(placement, std::move(absent));
  for (const auto& id : returned) NoteAbsentReturned(id, true);
  if (!returned.empty()) RetryAbsentRemoves();
}

void KVClient::AdoptPlacement(const std::vector<MemberInfo>& placement,
                              AbsentMap absent) {
  ConHash ring, full_ring;
  std::map<std::string, std::string> addr;
  std::set<std::string> draining;
  size_t serving = 0;
  for (const auto& m : placement) {
    addr[m.id] = m.ip + ":" + std::to_string(m.port);
    if (m.draining) draining.insert(m.id); else ++serving;
  }
  // With every member draining there is nowhere else to write: no drain.
  if (serving == 0) draining.clear();
  // Nor anything to keep absent members beside.
  if (placement.empty()) absent.clear();
  // The drain ring is the view's (as without absent members): a drainer
  // stays its keys' second owner whatever is absent. The absent ring adds
  // the absent members to it; a key it gives to one of them has that
  // member as one more owner.
  ConHash absent_ring;
  for (const auto& m : placement) {
    const int weight = m.weight < 1 ? 1 : static_cast<int>(m.weight);
    if (!draining.empty()) full_ring.AddNode(m.id, weight);
    if (!absent.empty()) absent_ring.AddNode(m.id, weight);
    if (draining.count(m.id) == 0) ring.AddNode(m.id, weight);
  }
  for (const auto& [id, member] : absent) {
    const uint32_t w = member.info.weight;
    absent_ring.AddNode(id, w < 1 ? 1 : static_cast<int>(w));
  }
  ring.Build();
  if (!draining.empty()) full_ring.Build();
  if (!absent.empty()) absent_ring.Build();
  AdoptRing(std::move(ring), std::move(addr), std::move(full_ring),
            std::move(draining), std::move(absent),
            absent_grace_ms_ > 0 ? placement : std::vector<MemberInfo>(),
            std::move(absent_ring));
}

KVClient::AbsentMap KVClient::NextAbsent(
    const std::vector<MemberInfo>& placement, uint64_t now,
    std::vector<std::string>* returned) {
  std::vector<MemberInfo> previous;
  AbsentMap held;
  {
    std::lock_guard<std::mutex> lk(ring_mu_);
    previous = placement_;
    held = absent_;
  }
  std::set<std::string> present;
  for (const auto& m : placement) present.insert(m.id);
  // At most max(1, N/8) absent, N the members of the previous full ring.
  const size_t bound =
      std::max<size_t>(1, (previous.size() + held.size()) / 8);
  AbsentMap next;
  AbsentMap dropped;
  // Members already absent first, oldest first; then the newly vanished.
  std::vector<std::pair<uint64_t, std::string>> order;
  for (const auto& [id, member] : held) {
    if (present.count(id) != 0) {
      returned->push_back(id);
      continue;
    }
    if (now - member.since_ms >= absent_grace_ms_) {
      absent_dropped_grace_.fetch_add(1, std::memory_order_relaxed);
      dropped.emplace(id, member);
      continue;
    }
    order.push_back({member.since_ms, id});
  }
  std::sort(order.begin(), order.end());
  for (const auto& [since, id] : order) {
    (void)since;
    if (next.size() < bound) {
      next.emplace(id, held[id]);
    } else {
      absent_dropped_bound_.fetch_add(1, std::memory_order_relaxed);
      dropped.emplace(id, held[id]);
    }
  }
  for (const auto& m : previous) {  // sorted by id as the MDS lists them
    if (present.count(m.id) != 0) continue;
    NoteAbsentReturned(m.id, false);
    AbsentMember member{m, m.ip + ":" + std::to_string(m.port), now};
    if (m.draining) {
      // A planned leave: it drained, its keys were written elsewhere.
      absent_dropped_planned_.fetch_add(1, std::memory_order_relaxed);
      DFKV_LOG_INFO("ring: member " + m.id +
                    " left after draining; not kept absent");
      continue;
    }
    if (next.size() < bound) {
      next.emplace(m.id, std::move(member));
    } else {
      absent_dropped_bound_.fetch_add(1, std::memory_order_relaxed);
      DFKV_LOG_WARN("ring: member " + m.id +
                    " vanished with " + std::to_string(bound) +
                    " member(s) absent already; dropped at once");
    }
  }
  for (const auto& [id, member] : dropped) {
    (void)member;
    DropAbsentRemoves(id);
  }
  RetireDropped(dropped);
  return next;
}

void KVClient::TickAbsent() {
  if (absent_grace_ms_ == 0) return;
  RetryAbsentRemoves();
  const uint64_t now = NowMs();
  std::vector<MemberInfo> placement;
  AbsentMap absent;
  {
    std::lock_guard<std::mutex> lk(ring_mu_);
    if (absent_.empty()) return;
    placement = placement_;
    absent = absent_;
  }
  AbsentMap dropped;
  for (auto it = absent.begin(); it != absent.end();) {
    if (now - it->second.since_ms < absent_grace_ms_) {
      ++it;
      continue;
    }
    absent_dropped_grace_.fetch_add(1, std::memory_order_relaxed);
    DFKV_LOG_INFO("ring: absent member " + it->first +
                  " dropped (grace passed)");
    dropped.insert(*it);
    it = absent.erase(it);
  }
  if (dropped.empty()) return;
  for (const auto& [id, member] : dropped) {
    (void)member;
    DropAbsentRemoves(id);
  }
  AdoptPlacement(placement, std::move(absent));
  RetireDropped(dropped);
}

void KVClient::RetireDropped(const AbsentMap& dropped) {
  if (dropped.empty() || t_ == nullptr) return;
  std::vector<std::pair<std::string, std::string>> retired;
  std::vector<std::string> live_ids;
  uint64_t generation = 0;
  {
    std::lock_guard<std::mutex> lk(ring_mu_);
    generation = topology_generation_;
    for (const auto& [id, member] : dropped) {
      const auto it = published_peers_.find(id);
      // Published again under this id (it came back): not ours to retire.
      if (it == published_peers_.end() || it->second != member.addr ||
          addr_.count(id) != 0)
        continue;
      retired.push_back(*it);
      published_peers_.erase(it);
    }
    for (const auto& peer : published_peers_) live_ids.push_back(peer.first);
  }
  if (retired.empty()) return;
  for (const auto& [peer_id, address] : retired) {
    PeerTopology peer;
    peer.peer_addr = address;
    peer.generation = generation;
    peer.complete = false;
    peer.peer_id = peer_id;
    peer.present = false;
    t_->OnPeerTopology(peer);
  }
  t_->OnPeerIdentities(live_ids);
}

void KVClient::QueueAbsentRemove(const std::string& id,
                                 const std::string& key) {
  absent_removes_unconfirmed_.fetch_add(1, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lk(absent_remove_mu_);
  auto& queue = absent_removes_[id];
  if (queue.count(key) != 0) return;  // queued already
  if (absent_removes_queued_ >= absent_remove_cap_) {
    absent_removes_dropped_.fetch_add(1, std::memory_order_relaxed);
    if (queue.empty()) absent_removes_.erase(id);
    return;
  }
  queue.insert(key);
  ++absent_removes_queued_;
}

void KVClient::DropAbsentRemoves(const std::string& id) {
  std::lock_guard<std::mutex> lk(absent_remove_mu_);
  absent_returned_.erase(id);
  const auto it = absent_removes_.find(id);
  if (it == absent_removes_.end()) return;
  absent_removes_dropped_.fetch_add(it->second.size(),
                                    std::memory_order_relaxed);
  absent_removes_queued_ -= it->second.size();
  absent_removes_.erase(it);
}

void KVClient::NoteAbsentReturned(const std::string& id, bool returned) {
  const uint64_t now = NowMs();
  if (returned) {
    // Its return to the view is as good as a probe answer, in every client:
    // end the cooldown its failures during the absence started, so that the
    // next request (a read, or the REMOVE retry below) is the trial instead
    // of waiting out the backoff. Once per incident, as any probe answer.
    std::string addr;
    {
      std::lock_guard<std::mutex> lk(ring_mu_);
      const auto it = addr_.find(id);
      if (it != addr_.end()) addr = it->second;
    }
    if (!addr.empty()) health_.MarkProbeAlive(addr, now);
  }
  std::lock_guard<std::mutex> lk(absent_remove_mu_);
  if (!returned) {
    // Gone again: its REMOVEs wait for the next return.
    absent_returned_.erase(id);
    return;
  }
  if (absent_removes_.count(id) == 0) return;
  absent_returned_[id] = now;
}

void KVClient::RetryAbsentRemoves() {
  const uint64_t now = NowMs();
  struct Work {
    std::string id;
    std::vector<std::string> keys;
  };
  std::vector<Work> work;
  {
    std::lock_guard<std::mutex> lk(absent_remove_mu_);
    for (auto it = absent_returned_.begin(); it != absent_returned_.end();) {
      auto queued = absent_removes_.find(it->first);
      if (queued == absent_removes_.end()) {
        it = absent_returned_.erase(it);
        continue;
      }
      absent_removes_queued_ -= queued->second.size();
      if (now - it->second > absent_grace_ms_) {
        // Back for a whole grace without confirming: given up.
        absent_removes_dropped_.fetch_add(queued->second.size(),
                                          std::memory_order_relaxed);
        absent_removes_.erase(queued);
        it = absent_returned_.erase(it);
        continue;
      }
      work.push_back({it->first,
                      std::vector<std::string>(queued->second.begin(),
                                               queued->second.end())});
      absent_removes_.erase(queued);
      ++it;
    }
  }
  for (auto& [id, keys] : work) {
    std::string addr;
    {
      std::lock_guard<std::mutex> lk(ring_mu_);
      const auto it = addr_.find(id);
      if (it != addr_.end()) addr = it->second;
    }
    std::vector<char> ok(keys.size(), 0);
    if (!addr.empty()) {
      // The return ended its cooldown (NoteAbsentReturned): the first
      // send is the trial; later sends take PeerHealth as it is.
      // To the member itself, wherever the ring now routes the keys.
      RouteTo to(addr);
      if (t_->pipelined()) {
        ok = BatchRemoveRouted(keys);
      } else {
        for (size_t i = 0; i < keys.size(); ++i)
          ok[i] = RemoveDirectRouted(keys[i]) ? 1 : 0;
      }
      absent_removes_retried_.fetch_add(keys.size(),
                                        std::memory_order_relaxed);
    }
    std::vector<std::string> left;
    for (size_t i = 0; i < keys.size(); ++i)
      if (!ok[i]) left.push_back(std::move(keys[i]));
    if (!addr.empty())
      DFKV_LOG_INFO("ring: member " + id + " is back; " +
                    std::to_string(keys.size() - left.size()) + " of " +
                    std::to_string(keys.size()) +
                    " REMOVE(s) it had not confirmed are confirmed now");
    std::lock_guard<std::mutex> lk(absent_remove_mu_);
    if (left.empty()) {
      if (absent_removes_.count(id) == 0) absent_returned_.erase(id);
      continue;
    }
    // Asked again at the next tick, while the member stays; within the
    // queue's bound, as any queued REMOVE.
    auto& queue = absent_removes_[id];
    for (auto& key : left) {
      if (queue.count(key) != 0) continue;
      if (absent_removes_queued_ >= absent_remove_cap_) {
        absent_removes_dropped_.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      queue.insert(std::move(key));
      ++absent_removes_queued_;
    }
    if (queue.empty()) {
      absent_removes_.erase(id);
      absent_returned_.erase(id);
    }
  }
}

// ---- Cooldown reroute and REMOVE fan-out (CLI-02 stage C) ----

void KVClient::NoteRecentLocked(const std::map<std::string, int>& weight,
                                const std::map<std::string, std::string>& addr,
                                bool adopted, uint64_t now) {
  if (weight.empty()) return;
  bool found = false;
  for (RecentRing& r : recent_) {
    bool& flag = adopted ? r.adopted : r.seen;
    const bool match = r.weight == weight;
    if (match) {
      found = true;
      flag = true;
      for (const auto& [id, a] : addr)
        if (weight.count(id) != 0) r.addr[id] = a;
      continue;
    }
    if (flag) {
      flag = false;
      if (!r.adopted && !r.seen) r.last_ms = now;
    }
  }
  if (!found) {
    RecentRing r;
    r.weight = weight;
    for (const auto& [id, a] : addr)
      if (weight.count(id) != 0) r.addr[id] = a;
    for (const auto& [id, w] : weight) r.ring.AddNode(id, w);
    r.ring.Build();
    (adopted ? r.adopted : r.seen) = true;
    r.last_ms = now;
    recent_.push_back(std::move(r));
  }
  RebuildRecentLocked();
}

void KVClient::RebuildRecentLocked() {
  std::map<std::string, int> all;
  std::map<std::string, size_t> count;
  recent_addr_.clear();
  for (const RecentRing& r : recent_) {
    for (const auto& [id, w] : r.weight) {
      all[id] = w;
      ++count[id];
    }
    for (const auto& [id, a] : r.addr) recent_addr_[id] = a;
  }
  // The current adopted ring's addresses win (a member that moved).
  for (const RecentRing& r : recent_)
    if (r.adopted)
      for (const auto& [id, a] : r.addr) recent_addr_[id] = a;
  recent_stable_.clear();
  for (const auto& [id, n] : count)
    if (n == recent_.size()) recent_stable_.insert(id);
  ConHash u;
  for (const auto& [id, w] : all) u.AddNode(id, w);
  u.Build();
  recent_union_ = std::move(u);
}

bool KVClient::ExpireRecentLocked(uint64_t now) {
  const uint64_t keep = recent_keep_ms_.load(std::memory_order_relaxed);
  const size_t before = recent_.size();
  recent_.erase(std::remove_if(recent_.begin(), recent_.end(),
                               [&](const RecentRing& r) {
                                 return !r.adopted && !r.seen &&
                                        now - r.last_ms > keep;
                               }),
                recent_.end());
  if (recent_.size() == before) return false;
  RebuildRecentLocked();
  return true;
}

std::string KVClient::SubstituteLocked(uint32_t hash, const std::string& owner,
                                       uint64_t now) const {
  std::string sub;
  size_t rank = 0;
  recent_union_.Walk(hash, [&](const std::string& id) {
    if (rank++ >= fanout_) return false;
    const auto a = recent_addr_.find(id);
    // Unusable members count toward the rank: the owner, members not in
    // every recent ring, cooling ones.
    if (a == recent_addr_.end() || a->second == owner) return true;
    if (recent_stable_.count(id) == 0) return true;
    if (health_.Cooling(a->second, now)) return true;
    sub = a->second;
    return false;
  });
  return sub;
}

void KVClient::FanoutTargetsLocked(uint32_t hash,
                                   std::set<std::string>* out) const {
  for (const RecentRing& r : recent_) {
    size_t stable = 0;
    r.ring.Walk(hash, [&](const std::string& id) {
      const auto a = r.addr.find(id);
      if (a != r.addr.end()) out->insert(a->second);
      if (recent_stable_.count(id) != 0 && ++stable >= fanout_) return false;
      return true;
    });
  }
}

void KVClient::NoteSeenView(const std::vector<MemberInfo>& placement) {
  if (!fanout_on_) return;
  // Its write ring: the eligible members that do not drain (all of them if
  // every one drains), as AdoptPlacement builds it.
  size_t serving = 0;
  for (const auto& m : placement)
    if (m.RingEligible() && !m.draining) ++serving;
  std::map<std::string, int> weight;
  std::map<std::string, std::string> addr;
  for (const auto& m : placement) {
    if (!m.RingEligible() || (serving != 0 && m.draining)) continue;
    weight[m.id] = m.weight < 1 ? 1 : static_cast<int>(m.weight);
    addr[m.id] = m.ip + ":" + std::to_string(m.port);
  }
  std::lock_guard<std::mutex> lk(ring_mu_);
  const uint64_t now = NowMs();
  if (readback_on_ && first_diverge_ms_ == 0 && !weight.empty()) {
    std::map<std::string, int> adopted(ring_.Nodes().begin(),
                                       ring_.Nodes().end());
    if (adopted != weight) first_diverge_ms_ = now;
  }
  NoteRecentLocked(weight, addr, false, now);
}

std::string KVClient::RouteFor(const std::string& key, RouteOp op) {
  std::string check_owner;
  uint64_t set_episode = 0;
  std::vector<std::string> sub{
      RouteForUnchecked(key, op, &check_owner, &set_episode)};
  if (check_owner.empty()) return sub[0];
  CheckSubstituteReads({key}, {check_owner}, &sub);
  // A SET on another episode's copy is tracked only once its check passed:
  // a refused one is not sent, and must not make the copy this client's own.
  if (set_episode != 0 && !sub[0].empty() &&
      !TrackReroute(check_owner, set_episode, key, sub[0])) {
    reroute_refused_.fetch_add(1, std::memory_order_relaxed);
    return "";
  }
  return sub[0];
}

std::vector<std::string> KVClient::RouteReads(
    size_t n, const std::function<const std::string&(size_t)>& key_of,
    RouteOp op) {
  std::vector<std::string> routes(n);
  std::vector<size_t> at;
  std::vector<std::string> keys, owners, subs;
  for (size_t j = 0; j < n; ++j) {
    std::string check_owner;
    routes[j] = RouteForUnchecked(key_of(j), op, &check_owner, nullptr);
    if (check_owner.empty()) continue;
    at.push_back(j);
    keys.push_back(key_of(j));
    owners.push_back(std::move(check_owner));
    subs.push_back(routes[j]);
  }
  if (at.empty()) return routes;
  CheckSubstituteReads(keys, owners, &subs);
  for (size_t k = 0; k < at.size(); ++k) routes[at[k]] = std::move(subs[k]);
  return routes;
}

std::string KVClient::RouteForUnchecked(const std::string& key, RouteOp op,
                                        std::string* check_owner,
                                        uint64_t* set_episode) {
  if (!reroute_on_ || t_drain_pass || t_route_to != nullptr) return Route(key);
  const uint32_t hash = ConHash::KeyHash(key);
  std::string owner;
  {
    std::lock_guard<std::mutex> lk(ring_mu_);
    std::string name;
    if (!ring_.LookupHash(hash, &name)) return "";
    const auto it = addr_.find(name);
    if (it == addr_.end()) return "";
    owner = it->second;
  }
  if (!health_.AnyCooldown() || !gate_open_.load(std::memory_order_acquire))
    return owner;
  const uint64_t now = NowMs();
  const uint64_t episode = health_.CooldownUntil(owner, now);
  if (episode == 0) return owner;
  std::string sub;
  {
    std::lock_guard<std::mutex> lk(ring_mu_);
    sub = SubstituteLocked(hash, owner, now);
  }
  if (sub.empty()) {
    reroutes_[kRrNone].fetch_add(1, std::memory_order_relaxed);
    return owner;  // as today: the cooling owner is skipped
  }
  // Revision 11: a copy this client did not write in the owner's current
  // episode may be another client's deferred one. Every op that reports on
  // the copy (GET, EXIST, and RETAIN, whose answer is an existence check)
  // is then a read-back with B; with C alone, while some client may defer,
  // a miss.
  const bool checks =
      op != RouteOp::kPut &&
      (readback_on_ || registry_readback_any_.load(std::memory_order_acquire));
  auto own_copy = [&] {  // reroute_mu_ held
    const auto o = reroute_keys_.find(owner);
    if (o == reroute_keys_.end()) return false;
    const auto ep = o->second.find(episode);
    return ep != o->second.end() && ep->second.count({key, sub}) != 0;
  };
  bool own = true;
  if (op == RouteOp::kPut || op == RouteOp::kRetainSet) {
    if (checks) {
      std::lock_guard<std::mutex> lk(reroute_mu_);
      own = own_copy();  // before this SET tracks it
    }
    if (!own && check_owner != nullptr) {
      // Another episode's copy: not tracked here. C alone refuses it; with
      // B, RouteFor tracks it once its check passed.
      if (!readback_on_) return "";
      *check_owner = owner;
      if (set_episode != nullptr) *set_episode = episode;
      reroutes_[kRrRetain].fetch_add(1, std::memory_order_relaxed);
      return sub;
    }
    if (!TrackReroute(owner, episode, key, sub)) {
      reroute_refused_.fetch_add(1, std::memory_order_relaxed);
      return "";  // never written to a substitute untracked
    }
  } else {
    // A copy whose cleanup is still queued is not read: it may be one a
    // REMOVE did not reach.
    std::lock_guard<std::mutex> lk(reroute_mu_);
    const auto pending = episode_cleanup_.find(sub);
    if (pending != episode_cleanup_.end() && pending->second.count(key) != 0)
      return "";
    if (checks) own = own_copy();
  }
  if (!own && check_owner != nullptr) {  // not RetainPrecheck
    if (!readback_on_) return "";
    *check_owner = owner;
  }
  const RerouteOpIndex index =
      op == RouteOp::kGet     ? kRrGet
      : op == RouteOp::kExist ? kRrExist
      : op == RouteOp::kPut   ? kRrPut
                              : kRrRetain;
  reroutes_[index].fetch_add(1, std::memory_order_relaxed);
  return sub;
}

bool KVClient::TrackReroute(const std::string& owner, uint64_t episode,
                            const std::string& key, const std::string& sub) {
  std::lock_guard<std::mutex> lk(reroute_mu_);
  auto& keys = reroute_keys_[owner][episode];
  if (keys.count({key, sub}) != 0) return true;
  // Tracked keys and episode cleanups still to land share the bound: no
  // write goes to a substitute that could not be cleaned up.
  if (reroute_tracked_ + episode_pending_ >= reroute_cap_) {
    if (keys.empty()) {
      reroute_keys_[owner].erase(episode);
      if (reroute_keys_[owner].empty()) reroute_keys_.erase(owner);
    }
    return false;
  }
  keys.insert({key, sub});
  ++reroute_tracked_;
  return true;
}

void KVClient::QueueFanoutCleanupLocked(const std::string& addr,
                                        const std::string& key) {
  auto& keys = fanout_cleanup_[addr];
  if (keys.count(key) != 0) return;
  if (fanout_pending_ >= reroute_cap_) {
    cleanup_abandoned_.fetch_add(1, std::memory_order_relaxed);
    if (keys.empty()) fanout_cleanup_.erase(addr);
    return;
  }
  keys.insert(key);
  ++fanout_pending_;
  cleanup_queued_total_.fetch_add(1, std::memory_order_relaxed);
}

std::vector<std::vector<char>> KVClient::RemoveAtNodes(
    const std::vector<std::pair<std::string, std::vector<std::string>>>&
        groups) {
  std::vector<std::vector<char>> result(groups.size());
  RunParallel(groups.size(), BatchWorkers(groups.size()), [&](size_t g) {
    const std::string& node = groups[g].first;
    const std::vector<std::string>& keys = groups[g].second;
    std::vector<char>& ok = result[g];
    ok.assign(keys.size(), 0);
    const uint64_t now = NowMs();
    if (health_.Cooling(node, now)) {
      ok.assign(keys.size(), 2);
      return;
    }
    if (!health_.Healthy(node, now)) return;
    std::vector<BlockKey> bkeys;
    bkeys.reserve(keys.size());
    for (const auto& key : keys) bkeys.push_back(block_key_(key));
    const std::vector<Status> sts = t_->RemoveMany(node, bkeys);
    ReportPeer(node, sts);
    for (size_t m = 0; m < keys.size(); ++m) {
      const bool removed = m < sts.size() && (sts[m] == Status::kOk ||
                                              sts[m] == Status::kNotFound);
      ok[m] = removed ? 1 : 0;
      if (removed) InvalidateRendezvous(bkeys[m]);
    }
  });
  return result;
}

// Fan-out REMOVE (§3.2): every target of a key must confirm in this call (the
// owner, a drainer, the fan-out set of every recent ring); a target cooling
// in this client, or failing, makes the REMOVE not ok, and a cooling one is
// queued for cleanup. The absent owner keeps stage A's treatment.
std::vector<char> KVClient::RemoveFanout(const std::vector<std::string>& keys) {
  const size_t n = keys.size();
  std::vector<char> ok(n, 1);
  std::vector<char> cooling(n, 0), failed(n, 0);
  std::map<std::string, std::vector<size_t>> by_node;
  for (size_t i = 0; i < n; ++i) {
    std::set<std::string> targets;
    {
      std::lock_guard<std::mutex> lk(ring_mu_);
      FanoutTargetsLocked(ConHash::KeyHash(keys[i]), &targets);
      if (targets.empty()) {
        // No recent ring (C configured without a ring yet): the owner.
        std::string name;
        if (ring_.LookupHash(ConHash::KeyHash(keys[i]), &name)) {
          const auto it = addr_.find(name);
          if (it != addr_.end()) targets.insert(it->second);
        }
      }
    }
    const std::string drainer = DrainRoute(keys[i]);
    if (!drainer.empty()) targets.insert(drainer);
    if (targets.empty()) {
      ok[i] = 0;
      failed[i] = 1;
      continue;
    }
    for (const auto& t : targets) by_node[t].push_back(i);
  }
  std::vector<std::pair<std::string, std::vector<std::string>>> groups;
  std::vector<std::vector<size_t>> index;
  for (auto& [node, idx] : by_node) {
    std::vector<std::string> sub;
    sub.reserve(idx.size());
    for (size_t i : idx) sub.push_back(keys[i]);
    groups.emplace_back(node, std::move(sub));
    index.push_back(std::move(idx));
  }
  const auto results = RemoveAtNodes(groups);
  {
    std::lock_guard<std::mutex> lk(reroute_mu_);
    for (size_t g = 0; g < groups.size(); ++g) {
      for (size_t m = 0; m < index[g].size(); ++m) {
        const size_t i = index[g][m];
        switch (results[g][m]) {
          case 1:
            fanout_ok_.fetch_add(1, std::memory_order_relaxed);
            break;
          case 2:
            ok[i] = 0;
            cooling[i] = 1;
            fanout_queued_cooling_.fetch_add(1, std::memory_order_relaxed);
            QueueFanoutCleanupLocked(groups[g].first, keys[i]);
            break;
          default:
            ok[i] = 0;
            failed[i] = 1;
            fanout_error_.fetch_add(1, std::memory_order_relaxed);
            break;
        }
      }
    }
  }
  for (size_t i = 0; i < n; ++i) {
    if (cooling[i])
      unconfirmed_cooling_.fetch_add(1, std::memory_order_relaxed);
    else if (failed[i])
      unconfirmed_failed_.fetch_add(1, std::memory_order_relaxed);
  }
  if (any_absent_.load(std::memory_order_relaxed)) RemoveAtAbsent(keys);
  return ok;
}

size_t KVClient::RunCleanup(bool all, uint64_t deadline_ms) {
  const uint64_t now = NowMs();
  const bool open = gate_open_.load(std::memory_order_acquire);
  // node -> keys, and whether the keys are episode cleanups (they stay in
  // episode_cleanup_ until confirmed, so that no read finds them meanwhile).
  std::map<std::string, std::set<std::string>> episode_work, fanout_work;
  {
    std::lock_guard<std::mutex> lk(reroute_mu_);
    // Ended episodes' cleanups wait for B's window only while every
    // registered client checks substitute reads (revision 11); otherwise
    // they go now. Read under reroute_mu_, which OnRegistry's clearing takes.
    const bool defer = !all && DeferralHolds();
    // Episodes that ended: any but the owner's running cooldown, all of
    // them once it does not cool or the gate closed.
    for (auto owner = reroute_keys_.begin(); owner != reroute_keys_.end();) {
      const uint64_t current =
          all || !open ? 0 : health_.CooldownUntil(owner->first, now);
      for (auto ep = owner->second.begin(); ep != owner->second.end();) {
        if (ep->first == current) {
          ++ep;
          continue;
        }
        for (const auto& [key, sub] : ep->second) {
          if (episode_cleanup_[sub].insert(key).second) ++episode_pending_;
          // With read-back on, the episode is a change with a window: its
          // keys stay on the substitute, readable through B only, until the
          // window has passed (design §3.4, revision 11).
          if (defer) episode_readback_[key] = {sub, now + readback_ms_};
        }
        reroute_tracked_ -= ep->second.size();
        ep = owner->second.erase(ep);
      }
      owner = owner->second.empty() ? reroute_keys_.erase(owner) : ++owner;
    }
    for (auto it = episode_readback_.begin(); it != episode_readback_.end();)
      it = !defer || now >= it->second.second ? episode_readback_.erase(it)
                                              : ++it;
    for (const auto& [node, keys] : episode_cleanup_) {
      if (!all && health_.Cooling(node, now)) continue;
      for (const auto& key : keys) {
        const auto deferred = episode_readback_.find(key);
        if (deferred != episode_readback_.end() &&
            deferred->second.first == node)
          continue;  // its read-back window is still open
        episode_work[node].insert(key);
      }
    }
    for (auto it = fanout_cleanup_.begin(); it != fanout_cleanup_.end();) {
      if (!all && health_.Cooling(it->first, now)) {
        ++it;
        continue;
      }
      fanout_work[it->first] = std::move(it->second);
      fanout_pending_ -= fanout_work[it->first].size();
      it = fanout_cleanup_.erase(it);
    }
  }
  if (episode_work.empty() && fanout_work.empty()) return 0;
  std::vector<std::pair<std::string, std::vector<std::string>>> groups;
  std::vector<bool> is_episode;
  for (const auto* work : {&episode_work, &fanout_work}) {
    for (const auto& [node, keys] : *work) {
      groups.emplace_back(node,
                          std::vector<std::string>(keys.begin(), keys.end()));
      is_episode.push_back(work == &episode_work);
    }
  }
  std::vector<std::vector<char>> results;
  if (deadline_ms == 0 || NowMs() < deadline_ms) results = RemoveAtNodes(groups);
  size_t left = 0;
  std::lock_guard<std::mutex> lk(reroute_mu_);
  for (size_t g = 0; g < groups.size(); ++g) {
    const std::string& node = groups[g].first;
    for (size_t m = 0; m < groups[g].second.size(); ++m) {
      const std::string& key = groups[g].second[m];
      const bool sent = !results.empty() && results[g][m] == 1;
      if (sent) cleanup_ok_.fetch_add(1, std::memory_order_relaxed);
      if (!sent) ++left;
      if (is_episode[g]) {
        if (sent || all) {
          auto it = episode_cleanup_.find(node);
          if (it != episode_cleanup_.end() && it->second.erase(key) != 0) {
            --episode_pending_;
            if (it->second.empty()) episode_cleanup_.erase(it);
          }
          if (!sent) cleanup_abandoned_.fetch_add(1, std::memory_order_relaxed);
        } else {
          cleanup_queued_total_.fetch_add(1, std::memory_order_relaxed);
        }
      } else if (!sent) {
        if (all)
          cleanup_abandoned_.fetch_add(1, std::memory_order_relaxed);
        else
          QueueFanoutCleanupLocked(node, key);
      }
    }
  }
  return left;
}

void KVClient::OnRegistry(bool ok, const std::vector<MemberInfo>& clients) {
  if (!fanout_on_) return;
  const uint64_t now = NowMs();
  {
  std::lock_guard<std::mutex> lk(gate_mu_);
  if (ok) {
    registry_seen_ = true;
    registry_ok_ms_ = now;
    size_t unmarked = 0;
    bool readback_all = true, readback_any = false;
    uint64_t poll_max = static_cast<uint64_t>(own_poll_ms_);
    static const std::string kMarker = "dfkv-remove-fanout=1;poll_ms=";
    for (const auto& c : clients) {
      const size_t at = c.info.find(kMarker);
      uint64_t poll = 0;
      if (at != std::string::npos) {
        size_t p = at + kMarker.size();
        while (p < c.info.size() && c.info[p] >= '0' && c.info[p] <= '9' &&
               poll < (1ull << 40))
          poll = poll * 10 + static_cast<uint64_t>(c.info[p++] - '0');
      }
      if (poll == 0)
        ++unmarked;
      else
        poll_max = std::max(poll_max, poll);
      // B configured there: it checks every substitute read it did not
      // write in its own current episode (revision 11).
      const size_t end = at == std::string::npos ? at : c.info.find(',', at);
      const bool b =
          at != std::string::npos &&
          c.info.substr(at, end == std::string::npos ? end : end - at)
                  .find(";readback=1") != std::string::npos;
      readback_all = readback_all && b;
      readback_any = readback_any || b;
    }
    registry_readback_all_.store(readback_all, std::memory_order_release);
    registry_readback_any_.store(readback_any, std::memory_order_release);
    unmarked_clients_ = unmarked;
    unmarked_gauge_.store(unmarked, std::memory_order_relaxed);
    poll_max_ms_ = poll_max;
  }
  EvaluateGateLocked(now);
  }
  // Revision 11 (a), re-checked at every answer: once it fails, every
  // deferred key is due (the tick sends it) and is never deferred again.
  if (readback_on_ && !DeferralHolds()) {
    std::lock_guard<std::mutex> lk(reroute_mu_);
    episode_readback_.clear();
  }
}

bool KVClient::DeferralHolds() const {
  return readback_on_ && gate_open_.load(std::memory_order_acquire) &&
         registry_readback_all_.load(std::memory_order_acquire);
}

void KVClient::EvaluateGateLocked(uint64_t now) {
  const uint64_t own = static_cast<uint64_t>(own_poll_ms_);
  // S follows the slowest marked poller: a larger one is used for keeping
  // rings at once but opens the gate only once that much history exists; a
  // smaller one replaces the old after one old S.
  if (poll_max_ms_ != 0) {
    const uint64_t s = 11 * poll_max_ms_;
    if (s_ms_ == 0) {
      s_ms_ = s;
    } else if (s != s_ms_) {
      if (s_next_ms_ != s) {
        s_next_ms_ = s;
        s_change_ms_ = now;
      }
      const uint64_t wait = s > s_ms_ ? s : s_ms_;
      if (now - s_change_ms_ >= wait) {
        s_ms_ = s;
        s_next_ms_ = 0;
      }
    } else {
      s_next_ms_ = 0;
    }
  }
  recent_keep_ms_.store(std::max(s_ms_, s_next_ms_),
                        std::memory_order_relaxed);
  poll_max_now_ms_.store(std::max(poll_max_ms_, own), std::memory_order_relaxed);
  int reason = -1;
  if (!fanout_on_ || own == 0 || !registered_ ||
      !registry_seen_ || own < 1000)
    reason = kGateNoRegistry;
  else if (now - registry_ok_ms_ > 2 * own)
    reason = kGateStale;
  else if (unmarked_clients_ != 0)
    reason = kGateUnmarked;
  else if (s_next_ms_ > s_ms_)
    reason = kGateSGrowing;
  const bool open = reason < 0;
  const bool was = gate_open_.exchange(open, std::memory_order_acq_rel);
  if (was && !open) {
    gate_failures_[reason].fetch_add(1, std::memory_order_relaxed);
    DFKV_LOG_WARN(std::string("ring: cooldown reroute off: ") +
                  (reason == kGateUnmarked
                       ? "a registered client does not fan out its REMOVEs"
                   : reason == kGateStale ? "the client registry answer is stale"
                   : reason == kGateSGrowing
                       ? "a slower client registered; waiting for the ring history"
                       : "the client registry cannot be read"));
  } else if (!was && open) {
    DFKV_LOG_INFO("ring: cooldown reroute on (every registered client fans "
                  "out its REMOVEs)");
  }
}

uint64_t KVClient::RecentKeepMsForTest() const {
  return recent_keep_ms_.load(std::memory_order_relaxed);
}

void KVClient::UseRegistryGateForTest(int poll_ms) {
  std::lock_guard<std::mutex> lk(gate_mu_);
  own_poll_ms_ = poll_ms;
  registered_ = true;
  s_ms_ = 11 * static_cast<uint64_t>(poll_ms);
  recent_keep_ms_.store(s_ms_, std::memory_order_relaxed);
}

void KVClient::TickReroute() {
  if (!fanout_on_) return;
  const uint64_t now = NowMs();
  {
    std::lock_guard<std::mutex> lk(gate_mu_);
    EvaluateGateLocked(now);
  }
  {
    std::lock_guard<std::mutex> lk(ring_mu_);
    ExpireRecentLocked(now);
  }
  RunCleanup(false, 0);
}

void KVClient::CloseReroute() {
  if (!fanout_on_) return;
  {
    std::lock_guard<std::mutex> lk(reroute_mu_);
    if (reroute_keys_.empty() && episode_cleanup_.empty() &&
        fanout_cleanup_.empty())
      return;
  }
  const size_t left =
      RunCleanup(true, reroute_close_ms_ == 0 ? 1 : NowMs() + reroute_close_ms_);
  if (left != 0)
    DFKV_LOG_WARN("ring: close left " + std::to_string(left) +
                  " key(s) on substitutes it could not REMOVE");
}

// ---- Previous-owner read-back (CLI-02 stage B) ----

void KVClient::NoteWriteRingChangeLocked(
    const ConHash& before, const std::map<std::string, std::string>& before_addr,
    uint64_t now) {
  const uint64_t diverged = first_diverge_ms_;
  first_diverge_ms_ = 0;
  if (before.Nodes().empty()) return;  // the first view: nothing was before
  if (now < held_until_ms_) {
    // A second change inside the window: the previous ring is cleared, not
    // chained (§3.4), and this change gets no window either.
    held_until_ms_ = 0;
    held_ring_ = ConHash();
    held_addr_.clear();
    readback_changes_chained_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  const uint64_t poll =
      std::max<uint64_t>(poll_max_now_ms_.load(std::memory_order_relaxed),
                         static_cast<uint64_t>(own_poll_ms_));
  const uint64_t settle = 10 * poll;
  if (diverged != 0 && now > diverged && now - diverged > settle - poll) {
    // Held longer than settle - poll_max: a remover on the newer ring may
    // have acted before H reaches (§3.4).
    readback_changes_held_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  held_ring_ = before;
  held_addr_ = before_addr;
  held_until_ms_ = now + readback_ms_;
}

void KVClient::AddAskSetLocked(uint32_t hash, uint64_t now,
                               std::set<std::string>* ask) const {
  auto walk = [&](const ConHash& ring,
                  const std::map<std::string, std::string>& addr) {
    size_t n = 0;
    ring.Walk(hash, [&](const std::string& id) {
      const auto a = addr.find(id);
      if (a != addr.end()) ask->insert(a->second);
      return ++n < fanout_;
    });
  };
  for (const RecentRing& r : recent_) walk(r.ring, r.addr);
  if (now < held_until_ms_) walk(held_ring_, held_addr_);
}

std::map<std::string, std::vector<size_t>> KVClient::PlanReadBack(
    const std::vector<std::string>& keys) {
  std::map<std::string, std::vector<size_t>> plan;
  if (!readback_on_ || keys.empty()) return plan;
  const uint64_t now = NowMs();
  std::vector<RbCand> cands;
  {
    std::lock_guard<std::mutex> lk(ring_mu_);
    for (size_t i = 0; i < keys.size(); ++i) {
      const uint32_t hash = ConHash::KeyHash(keys[i]);
      std::string o, p;
      if (!ring_.LookupHash(hash, &o)) continue;
      const auto oa = addr_.find(o);
      if (oa == addr_.end()) continue;
      std::string prev;
      if (now < held_until_ms_ && held_ring_.LookupHash(hash, &p)) {
        const auto pa = held_addr_.find(p);
        if (pa != held_addr_.end()) prev = pa->second;
      }
      if (prev.empty() || prev == oa->second) {
        // A cooldown-reroute episode of O that ended within the window: the
        // substitute is the previous owner of the keys written there.
        std::lock_guard<std::mutex> rk(reroute_mu_);
        const auto ep = episode_readback_.find(keys[i]);
        if (ep != episode_readback_.end() && now < ep->second.second)
          prev = ep->second.first;
      }
      if (prev.empty() || prev == oa->second)
        continue;  // the owner did not change
      RbCand c{i, oa->second, prev, {}, -1};
      c.ask.insert(prev);
      AddAskSetLocked(hash, now, &c.ask);
      cands.push_back(std::move(c));
    }
  }
  if (cands.empty()) return plan;
  const bool gate = gate_open_.load(std::memory_order_acquire);
  for (RbCand& cand : cands) {
    // A not found from a substitute (the owner cools here) is not the
    // owner's answer: not a candidate.
    if (gate && health_.Cooling(cand.owner, now))
      cand.refusal = kRbRefusals;  // dropped silently
    else if (!gate)
      cand.refusal = kRbGate;
    else if (health_.Cooling(cand.prev, now))
      cand.refusal = kRbCooling;
  }
  AskRemoved(keys, now, &cands);
  for (const RbCand& cand : cands)
    if (cand.refusal < 0) plan[cand.prev].push_back(cand.i);
  return plan;
}

void KVClient::AskRemoved(const std::vector<std::string>& keys, uint64_t now,
                          std::vector<RbCand>* cands_in) {
  std::vector<RbCand>& cands = *cands_in;
  const bool length = length_mismatch_seen_.load(std::memory_order_relaxed);
  std::map<std::string, std::vector<size_t>> by_node;  // -> cands
  for (size_t c = 0; c < cands.size(); ++c) {
    RbCand& cand = cands[c];
    if (cand.refusal >= 0) continue;
    if (length) { cand.refusal = kRbLength; continue; }
    const std::string& key = keys[cand.i];
    const std::string drainer = DrainRoute(key);
    if (!drainer.empty()) cand.ask.insert(drainer);
    std::string absent_id;
    const std::string absent = AbsentRoute(key, &absent_id);
    if (!absent.empty()) cand.ask.insert(absent);
    cand.ask.erase(cand.owner);  // O answered not found or cools (§3.4)
    for (const auto& node : cand.ask) by_node[node].push_back(c);
  }
  const uint64_t poll =
      std::max<uint64_t>(poll_max_now_ms_.load(std::memory_order_relaxed),
                         static_cast<uint64_t>(own_poll_ms_));
  const uint64_t h64 = readback_ms_ + 10 * poll;
  const uint32_t h = static_cast<uint32_t>(std::min<uint64_t>(h64, UINT32_MAX));
  // With the rewrite check (CLI-35): each candidate's answers, by node.
  const bool rewrite = rewrite_on_ && readback_on_;
  std::vector<std::vector<std::pair<std::string, MemoryAnswer>>> heard(
      rewrite ? cands.size() : 0);
  for (const auto& [node, list] : by_node) {
    std::vector<std::string> ask_keys;
    ask_keys.reserve(list.size());
    for (size_t c : list) ask_keys.push_back(keys[cands[c].i]);
    std::vector<uint8_t> answers;
    uint32_t covered = 0;
    Status st;
    if (rewrite) {
      std::vector<MemoryAnswer> ages;
      st = RemovedWithinAges(node, ask_keys, h, &ages, &covered);
      for (size_t m = 0; m < list.size(); ++m)
        heard[list[m]].emplace_back(node, ages[m]);
    } else {
      st = RemovedWithin(node, ask_keys, h, &answers, &covered);
    }
    if (st == Status::kOk) {
      // A node whose M stays shorter than H past a start: counted once.
      std::lock_guard<std::mutex> lk(memory_short_mu_);
      auto& [since, reported] = memory_short_[node];
      if (covered >= h) {
        since = 0;
      } else if (since == 0) {
        since = now;
      } else if (!reported && now - since > h) {
        reported = true;
        readback_memory_short_.fetch_add(1, std::memory_order_relaxed);
        DFKV_LOG_WARN("read-back: " + node + " remembers REMOVEs for " +
                      std::to_string(covered) + " ms, less than H = " +
                      std::to_string(h) +
                      " ms; its read-backs are refused (raise "
                      "DFKV_REMOVE_MEMORY_MS there)");
      }
    }
    if (rewrite) continue;
    for (size_t m = 0; m < list.size(); ++m) {
      RbCand& cand = cands[list[m]];
      if (answers[m] == 0) continue;
      // Removed wins over unknown in the count; either refuses.
      if (answers[m] == 1)
        cand.refusal = kRbRemoved;
      else if (cand.refusal != kRbRemoved)
        cand.refusal = kRbUnknown;
    }
  }
  for (size_t c = 0; c < heard.size(); ++c) {
    RbCand& cand = cands[c];
    if (cand.refusal >= 0) continue;
    switch (RewriteVerdict(cand.prev, heard[c])) {
      case Rewrite::kNoRemove:
        break;
      case Rewrite::kAfterEvery:
        readback_rewritten_.fetch_add(1, std::memory_order_relaxed);
        break;
      case Rewrite::kRemoved:
        cand.refusal = kRbRemoved;
        break;
      case Rewrite::kUnknown:
        cand.refusal = kRbUnknown;
        break;
    }
  }
  for (const RbCand& cand : cands)
    if (cand.refusal >= 0 && cand.refusal < kRbRefusals)
      readback_refused_[cand.refusal].fetch_add(1, std::memory_order_relaxed);
}

KVClient::Rewrite KVClient::RewriteVerdict(
    const std::string& prev,
    const std::vector<std::pair<std::string, MemoryAnswer>>& heard) {
  // §3.5. Ages on each node's CLOCK_MONOTONIC_RAW, compared through the
  // client's own RAW request times; rho bounds every clock's rate, G covers
  // a raw clock stopped for less than the nodes' pause check sees.
  constexpr double kRho = 1e-3;
  constexpr double kGuardMs = 1000;
  const MemoryAnswer* p = nullptr;
  for (const auto& [node, a] : heard)
    if (node == prev) p = &a;
  bool unknown = false, removed = false, refused = false;
  for (const auto& [node, a] : heard) {
    if (a.removed_age == kRemovedAgeUnknown) {
      unknown = true;
      continue;
    }
    if (a.removed_age == kRemovedAgeNone) continue;
    removed = true;
    if (p == nullptr || p->written_age == kRemovedAgeNone) {
      refused = true;
    } else if (node == prev) {
      // Equal integer ages cannot say whether a same-ms rewrite happened
      // before or after this REMOVE. A joint snapshot orders reads, not those
      // completed operations, so a tie must not amortize a read-back.
      if (a.removed_age <= p->written_age) refused = true;
    } else {
      const double lhs = a.removed_age / (1 + kRho) -
                         (static_cast<double>(p->written_age) + 1) / (1 - kRho);
      const double rhs =
          std::max(0.0, a.answered_ms - p->sent_ms) / (1 - kRho) + kGuardMs;
      if (!(lhs > rhs)) refused = true;
    }
  }
  if (refused) return Rewrite::kRemoved;  // removed wins over unknown
  if (unknown) return Rewrite::kUnknown;
  return removed ? Rewrite::kAfterEvery : Rewrite::kNoRemove;
}

void KVClient::CheckSubstituteReads(const std::vector<std::string>& keys,
                                    const std::vector<std::string>& owners,
                                    std::vector<std::string>* subs) {
  const uint64_t now = NowMs();
  std::vector<RbCand> cands;
  cands.reserve(keys.size());
  {
    std::lock_guard<std::mutex> lk(ring_mu_);
    for (size_t k = 0; k < keys.size(); ++k) {
      RbCand c{k, owners[k], (*subs)[k], {}, -1};
      c.ask.insert(c.prev);
      AddAskSetLocked(ConHash::KeyHash(keys[k]), now, &c.ask);
      cands.push_back(std::move(c));
    }
  }
  AskRemoved(keys, now, &cands);
  for (const RbCand& cand : cands)
    if (cand.refusal >= 0) (*subs)[cand.i].clear();  // a miss
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
        // Every view seen, admitted or not, is a recent ring (stage C).
        NoteSeenView(topology);
        // Topology is a replace-all stream independent of placement. Degraded
        // members remain here even though SetMembers excludes them. Ring
        // members a guard-refused view omits keep their topology until a view
        // without them is admitted.
        // Absent members (absent grace) keep theirs while they are held.
        auto held = [this] {
          std::map<std::string, std::string> keep;
          std::lock_guard<std::mutex> lk(ring_mu_);
          for (const auto& [id, member] : absent_) keep[id] = member.addr;
          return keep;
        };
        if (!placement_admitted) {
          PublishPeerTopologies(topology, topology_epoch, true, held());
          return;
        }
        std::vector<MemberInfo> placement;
        placement.reserve(topology.size());
        for (const auto& member : topology) {
          if (member.RingEligible()) placement.push_back(member);
        }
        // Which members drain is part of the placement here: it splits the
        // write ring from the read ring.
        const uint64_t placement_epoch =
            MembersEpoch(placement) ^ MembersDrainEpoch(placement);
        const bool changed = !placement_state->have_epoch ||
                             placement_state->epoch != placement_epoch;
        if (!changed || absent_grace_ms_ == 0) {
          PublishPeerTopologies(topology, topology_epoch, false, held());
          if (!changed) return;
          placement_state->epoch = placement_epoch;
          placement_state->have_epoch = true;
          SetMembers(placement);
          return;
        }
        // Which members stay absent is known before the topology is
        // published, so that theirs is not retired.
        placement_state->epoch = placement_epoch;
        placement_state->have_epoch = true;
        std::vector<std::string> returned;
        AbsentMap absent = NextAbsent(placement, NowMs(), &returned);
        std::map<std::string, std::string> keep;
        for (const auto& [id, member] : absent) keep[id] = member.addr;
        PublishPeerTopologies(topology, topology_epoch, false, keep);
        AdoptPlacement(placement, std::move(absent));
        for (const auto& id : returned) NoteAbsentReturned(id, true);
        if (!returned.empty()) RetryAbsentRemoves();
      },
      poll_ms);
  if (fanout_on_) {
    own_poll_ms_ = poll_ms;
    {
      std::lock_guard<std::mutex> lk(gate_mu_);
      s_ms_ = 11 * static_cast<uint64_t>(poll_ms > 0 ? poll_ms : 0);
      recent_keep_ms_.store(s_ms_, std::memory_order_relaxed);
    }
    if (poll_ms < 1000)
      DFKV_LOG_WARN("DFKV_RING_COOLDOWN_REROUTE / DFKV_RING_READBACK_MS: the "
                    "member poll interval (" + std::to_string(poll_ms) +
                    " ms) is below 1000 ms; reroute and read-back stay off");
    // H = window + settle must fit M's 600 s default (§3.4).
    const uint64_t settle = 10 * static_cast<uint64_t>(poll_ms > 0 ? poll_ms : 0);
    if (readback_on_ && readback_ms_ + settle > 600000) {
      if (settle >= 600000) {
        DFKV_LOG_WARN("DFKV_RING_READBACK_MS: ten poll intervals leave no "
                      "read-back window under 600 s; read-back stays off");
        readback_on_ = false;
      } else {
        DFKV_LOG_WARN("DFKV_RING_READBACK_MS=" + std::to_string(readback_ms_) +
                      " plus ten poll intervals exceeds 600 s; using " +
                      std::to_string(600000 - settle));
        readback_ms_ = 600000 - settle;
      }
    }
    poller_->SetRegistry([this](bool ok, const std::vector<MemberInfo>& clients) {
      OnRegistry(ok, clients);
    });
  }
  if (absent_grace_ms_ > 0 || fanout_on_)
    poller_->SetTick(
        [this] {
          TickAbsent();
          TickReroute();
        },
        1000);
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
  // With fan-out on, the registry marker the gate of every client reads
  // (§3.2): this client fans out its REMOVEs, and polls every poll_ms.
  if (fanout_on_) {
    std::lock_guard<std::mutex> lk(gate_mu_);
    registered_ = true;
  }
  // Only with the poll interval MDS discovery set: a guess would size the
  // other clients' S wrongly.
  if (fanout_on_ && own_poll_ms_ > 0)
    self.info += (self.info.empty() ? "" : ",") +
                 std::string("dfkv-remove-fanout=1;poll_ms=") +
                 std::to_string(own_poll_ms_) +
                 (readback_on_ ? ";readback=1" : "");
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

KVClient::~KVClient() {
  // Stage C: the episode-end cleanup of a clean close (nothing with C off).
  CloseReroute();
  StopProbe();
  StopClientRegistration();
  StopMdsDiscovery();
}

// Every operation reports here what its statuses say about the node it was
// sent to: PeerHealth::Report is the one place that turns statuses into a
// verdict. A failure that the transport refused without dialing (the address
// was held after an earlier failed connect) is not evidence of anything new.
// The transport's record of this thread's operation is taken on every report,
// whatever the status, so that it never outlives the operation it describes.
Status KVClient::RemovedWithin(const std::string& node,
                               const std::vector<std::string>& keys,
                               uint32_t age_ms, std::vector<uint8_t>* answers,
                               uint32_t* covered_ms) {
  constexpr uint64_t kNoMemoryRecheckMs = 60000;
  answers->assign(keys.size(), 2);
  if (covered_ms) *covered_ms = 0;
  if (keys.empty()) return Status::kOk;
  const uint64_t now = NowMs();
  {
    std::lock_guard<std::mutex> lk(no_memory_mu_);
    const auto it = no_memory_until_.find(node);
    if (it != no_memory_until_.end()) {
      if (now < it->second) return Status::kInvalid;
      no_memory_until_.erase(it);
    }
  }
  if (!health_.Healthy(node, now)) return Status::kIOError;
  uint32_t covered = UINT32_MAX;
  for (size_t begin = 0; begin < keys.size();
       begin += kRemovedWithinMaxKeys) {
    const size_t n = std::min(kRemovedWithinMaxKeys, keys.size() - begin);
    std::vector<BlockKey> bkeys;
    bkeys.reserve(n);
    for (size_t i = 0; i < n; ++i) bkeys.push_back(block_key_(keys[begin + i]));
    const std::string payload =
        EncodeRemovedWithinPayload(age_ms, bkeys.data(), n);
    std::string out;
    const Status st =
        t_->RemovedWithin(node, bkeys[0].tenant_hash, n, payload, &out);
    ReportPeer(node, st);
    if (st == Status::kInvalid) {
      // No memory there: unknown, and not asked again for a while (an
      // older node counts every unknown op it is sent).
      std::lock_guard<std::mutex> lk(no_memory_mu_);
      no_memory_until_[node] = NowMs() + kNoMemoryRecheckMs;
      answers->assign(keys.size(), 2);
      return Status::kInvalid;
    }
    if (st != Status::kOk || out.size() != 4 + n) {
      answers->assign(keys.size(), 2);
      return st == Status::kOk ? Status::kIOError : st;
    }
    covered = std::min(covered, net::GetU32(out.data()));
    for (size_t i = 0; i < n; ++i) {
      const uint8_t a = static_cast<uint8_t>(out[4 + i]);
      (*answers)[begin + i] = a <= 2 ? a : 2;
    }
  }
  if (covered_ms) *covered_ms = covered;
  return Status::kOk;
}

Status KVClient::RemovedWithinAges(const std::string& node,
                                   const std::vector<std::string>& keys,
                                   uint32_t age_ms,
                                   std::vector<MemoryAnswer>* answers,
                                   uint32_t* covered_ms) {
  constexpr uint64_t kRecheckMs = 60000;
  answers->assign(keys.size(), MemoryAnswer{});
  if (covered_ms) *covered_ms = 0;
  if (keys.empty()) return Status::kOk;
  const uint64_t now = NowMs();
  bool ages = true;
  {
    std::lock_guard<std::mutex> lk(no_memory_mu_);
    const auto it = no_memory_until_.find(node);
    if (it != no_memory_until_.end()) {
      if (now < it->second) return Status::kInvalid;
      no_memory_until_.erase(it);
    }
    const auto at = no_ages_until_.find(node);
    if (at != no_ages_until_.end()) {
      if (now < at->second)
        ages = false;
      else
        no_ages_until_.erase(at);
    }
  }
  if (!health_.Healthy(node, now)) return Status::kIOError;
  uint32_t covered = UINT32_MAX;
  for (size_t begin = 0; begin < keys.size();
       begin += kRemovedWithinMaxKeys) {
    const size_t n = std::min(kRemovedWithinMaxKeys, keys.size() - begin);
    std::vector<BlockKey> bkeys;
    bkeys.reserve(n);
    for (size_t i = 0; i < n; ++i) bkeys.push_back(block_key_(keys[begin + i]));
    const std::string payload =
        EncodeRemovedWithinPayload(age_ms, bkeys.data(), n);
    // Times per request actually answered (§3.5): a fallback or a later
    // chunk has its own.
    std::string out;
    double sent = RawMsExact();
    Status st = Status::kInvalid;
    if (ages) {
      st = t_->RemovedWithinAges(node, bkeys[0].tenant_hash, n, payload, &out);
      if (st == Status::kInvalid) {
        // No op 26 there (an older node, or one without M): op 25 instead.
        std::lock_guard<std::mutex> lk(no_memory_mu_);
        no_ages_until_[node] = NowMs() + kRecheckMs;
        ages = false;
      }
    }
    if (!ages) {
      out.clear();
      sent = RawMsExact();
      st = t_->RemovedWithin(node, bkeys[0].tenant_hash, n, payload, &out);
    }
    const double answered = RawMsExact();
    ReportPeer(node, st);
    if (st == Status::kInvalid) {
      std::lock_guard<std::mutex> lk(no_memory_mu_);
      no_memory_until_[node] = NowMs() + kRecheckMs;
      answers->assign(keys.size(), MemoryAnswer{});
      return Status::kInvalid;
    }
    const size_t want = ages ? 4 + n * kRemovedWithinAgesKeyBytes : 4 + n;
    if (st != Status::kOk || out.size() != want) {
      answers->assign(keys.size(), MemoryAnswer{});
      return st == Status::kOk ? Status::kIOError : st;
    }
    covered = std::min(covered, net::GetU32(out.data()));
    for (size_t i = 0; i < n; ++i) {
      MemoryAnswer& a = (*answers)[begin + i];
      a.sent_ms = sent;
      a.answered_ms = answered;
      if (ages) {
        a.removed_age = net::GetU32(&out[4 + i * 8]);
        a.written_age = net::GetU32(&out[8 + i * 8]);
        if (a.written_age >= kRemovedAgeUnknown) a.written_age = kRemovedAgeNone;
      } else {
        // Op 25: a REMOVE counts as just now; no write ages.
        const uint8_t bit = static_cast<uint8_t>(out[4 + i]);
        a.removed_age = bit == 0   ? kRemovedAgeNone
                        : bit == 1 ? 0
                                   : kRemovedAgeUnknown;
        a.written_age = kRemovedAgeNone;
      }
    }
  }
  if (covered_ms) *covered_ms = covered;
  return Status::kOk;
}

void KVClient::ReportPeer(const std::string& node, const Status* statuses,
                          size_t count) {
  PeerOutcome outcome;
  for (size_t i = 0; i < count; ++i) {
    outcome.Add(statuses[i]);
    if (statuses[i] == Status::kCacheFull)
      refused_cache_full_.fetch_add(1, std::memory_order_relaxed);
    else if (statuses[i] == Status::kQuotaExceeded)
      refused_quota_.fetch_add(1, std::memory_order_relaxed);
  }
  const bool refused = t_->TakeUnreachableRefusal();
  health_.Report(node, outcome, NowMs(), outcome.failed && refused);
}

void KVClient::StartProbe(int interval_ms) {
  if (interval_ms <= 0 || probe_running_.load(std::memory_order_relaxed)) return;
  probe_interval_ms_ = interval_ms;
  probe_running_.store(true, std::memory_order_relaxed);
  probe_th_ = std::thread([this] { NameThisThread("kv-probe"); ProbeLoop(); });
}

void KVClient::StopProbe() {
  {
    std::lock_guard<std::mutex> lk(probe_mu_);  // see MdsMemberPoller::Stop
    if (!probe_running_.exchange(false, std::memory_order_relaxed)) return;
  }
  probe_cv_.notify_all();
  if (probe_th_.joinable()) probe_th_.join();
}

void KVClient::ProbeLoop() {
  // A sentinel key that no real workload writes; kExist of it is a cheap round
  // trip that measures the node's responsiveness (kNotFound is a healthy reply).
  const BlockKey probe = block_key_("__dfkv_probe__");
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
      // The probe reports nothing to PeerHealth: drop the transport's record
      // of this operation, as ReportPeer does for the data path.
      t_->TakeUnreachableRefusal();
      double sec = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t0).count();
      // Only record real round trips; a transport IO error (e.g. connect refused)
      // is fast and would skew latency, and is already counted by PeerHealth.
      // A non-IO reply also means the peer is reachable: end its data-path
      // cooldown early, so that the next request tries it instead of waiting
      // out a backed-off cooldown. The probe does not decide that the peer is
      // healthy: a node with a failed store answers this query with a miss.
      // kResourceExhausted and kNoCompatibleRail never dialed the peer at all:
      // their wall time is local resource/topology selection and would poison
      // the latency EWMA; neither proves anything about peer reachability.
      if (st != Status::kIOError && st != Status::kResourceExhausted &&
          st != Status::kNoCompatibleRail) {
        peer_lat_.Observe(a, sec);
        health_.MarkProbeAlive(a, NowMs());
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
  s += "# HELP dfkv_client_node_refusals_total Keys a node refused to store: cache_full (no room, or its RAM tier over the dirty watermark), quota_exceeded\n";
  s += "# TYPE dfkv_client_node_refusals_total counter\n";
  s += "dfkv_client_node_refusals_total{status=\"cache_full\"} " +
       std::to_string(refused_cache_full_.load(std::memory_order_relaxed)) + "\n";
  s += "dfkv_client_node_refusals_total{status=\"quota_exceeded\"} " +
       std::to_string(refused_quota_.load(std::memory_order_relaxed)) + "\n";
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
    s += "# HELP dfkv_client_dedup_length_mismatches_total Lookups that found their key published with a length they do not expect\n";
    s += "# TYPE dfkv_client_dedup_length_mismatches_total counter\n";
    s += "dfkv_client_dedup_length_mismatches_total " + std::to_string(dedup_->length_mismatches()) + "\n";
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
    s += "# HELP dfkv_client_gpu_dedup_length_mismatches_total Lookups that found their key published longer than their segments hold\n";
    s += "# TYPE dfkv_client_gpu_dedup_length_mismatches_total counter\n";
    s += "dfkv_client_gpu_dedup_length_mismatches_total " + std::to_string(gd->length_mismatches()) + "\n";
    s += "# HELP dfkv_client_gpu_dedup_load_fences_total Device synchronizations made before publishing a fetch to the GPU rendezvous\n";
    s += "# TYPE dfkv_client_gpu_dedup_load_fences_total counter\n";
    s += "dfkv_client_gpu_dedup_load_fences_total " + std::to_string(gd->load_fences()) + "\n";
    s += "# HELP dfkv_client_gpu_dedup_load_fence_failures_total Fetches not published to the GPU rendezvous because the device could not be synchronized\n";
    s += "# TYPE dfkv_client_gpu_dedup_load_fence_failures_total counter\n";
    s += "dfkv_client_gpu_dedup_load_fence_failures_total " + std::to_string(gd->load_fence_failures()) + "\n";
    s += "# HELP dfkv_client_gpu_dedup_invalidate_stuck_total Invalidations that waited for one slot longer than the takeover age, or left a slot whose release can only end empty\n";
    s += "# TYPE dfkv_client_gpu_dedup_invalidate_stuck_total counter\n";
    s += "dfkv_client_gpu_dedup_invalidate_stuck_total " + std::to_string(gd->invalidate_stuck()) + "\n";
  }
  // MDS discovery / placement-ring health. An empty ring means every put/get
  // routes to nowhere and silently reports ok=0, so surface both the current
  // ring size and MDS reachability — the fastest tell of a bad mds_endpoint.
  {
    size_t members, draining;
    size_t collisions;
    {
      std::lock_guard<std::mutex> lk(ring_mu_);
      members = addr_.size();
      draining = draining_.size();
      collisions = ring_.Collisions();
    }
    s += "# HELP dfkv_client_ring_members Placement-ring members currently adopted from MDS discovery\n";
    s += "# TYPE dfkv_client_ring_members gauge\n";
    s += "dfkv_client_ring_members " + std::to_string(members) + "\n";
    s += "# HELP dfkv_client_ring_draining_members Ring members that are draining: no new writes, read after a miss at the new owner\n";
    s += "# TYPE dfkv_client_ring_draining_members gauge\n";
    s += "dfkv_client_ring_draining_members " + std::to_string(draining) + "\n";
    auto load = [](const std::atomic<uint64_t>& v) {
      return std::to_string(v.load(std::memory_order_relaxed));
    };
    s += "# HELP dfkv_client_drain_requests_total Second requests to a key's draining owner: reads after a miss at the new owner, and the REMOVEs and RETAINs that go to both; result=served when it had (or removed) the key\n";
    s += "# TYPE dfkv_client_drain_requests_total counter\n";
    s += "dfkv_client_drain_requests_total{op=\"read\",result=\"served\"} " + load(drain_reads_hit_) + "\n";
    s += "dfkv_client_drain_requests_total{op=\"read\",result=\"not_served\"} " + load(drain_reads_miss_) + "\n";
    s += "dfkv_client_drain_requests_total{op=\"remove\",result=\"served\"} " + load(drain_removes_ok_) + "\n";
    s += "dfkv_client_drain_requests_total{op=\"remove\",result=\"not_served\"} " + load(drain_removes_failed_) + "\n";
    s += "dfkv_client_drain_requests_total{op=\"retain\",result=\"served\"} " + load(drain_retains_ok_) + "\n";
    s += "dfkv_client_drain_requests_total{op=\"retain\",result=\"not_served\"} " + load(drain_retains_failed_) + "\n";
    size_t absent = 0;
    {
      std::lock_guard<std::mutex> lk(ring_mu_);
      absent = absent_.size();
    }
    s += "# HELP dfkv_client_ring_absent_members Members kept absent (DFKV_RING_ABSENT_GRACE_MS): vanished from the view; no writes and no reads; REMOVE and RETAIN reach them; their unconfirmed REMOVEs are retried when they return\n";
    s += "# TYPE dfkv_client_ring_absent_members gauge\n";
    s += "dfkv_client_ring_absent_members " + std::to_string(absent) + "\n";
    s += "# HELP dfkv_client_ring_absent_dropped_total Absent members dropped: grace over, over the bound of max(1, N/8), or a planned leave (it drained) that is not kept\n";
    s += "# TYPE dfkv_client_ring_absent_dropped_total counter\n";
    s += "dfkv_client_ring_absent_dropped_total{reason=\"grace\"} " + load(absent_dropped_grace_) + "\n";
    s += "dfkv_client_ring_absent_dropped_total{reason=\"bound\"} " + load(absent_dropped_bound_) + "\n";
    s += "dfkv_client_ring_absent_dropped_total{reason=\"planned\"} " + load(absent_dropped_planned_) + "\n";
    s += "# HELP dfkv_client_ring_absent_removes_unconfirmed_total REMOVEs an absent member did not confirm (queued for one retry on its return)\n";
    s += "# TYPE dfkv_client_ring_absent_removes_unconfirmed_total counter\n";
    s += "dfkv_client_ring_absent_removes_unconfirmed_total " + load(absent_removes_unconfirmed_) + "\n";
    s += "# HELP dfkv_client_ring_absent_removes_retried_total Unconfirmed REMOVEs sent again after the member returned (each attempt)\n";
    s += "# TYPE dfkv_client_ring_absent_removes_retried_total counter\n";
    s += "dfkv_client_ring_absent_removes_retried_total " + load(absent_removes_retried_) + "\n";
    s += "# HELP dfkv_client_ring_absent_removes_dropped_total Unconfirmed REMOVEs never confirmed: over the queue bound, the member dropped, or not confirmed within the grace after its return\n";
    s += "# TYPE dfkv_client_ring_absent_removes_dropped_total counter\n";
    s += "dfkv_client_ring_absent_removes_dropped_total " + load(absent_removes_dropped_) + "\n";
    s += "# HELP dfkv_client_cooldown_reroutes_total Keys routed past an owner cooling in this client (cooldown reroute); op=\"none\": the owner cooled and no candidate was usable\n";
    s += "# TYPE dfkv_client_cooldown_reroutes_total counter\n";
    s += "dfkv_client_cooldown_reroutes_total{op=\"get\"} " + load(reroutes_[kRrGet]) + "\n";
    s += "dfkv_client_cooldown_reroutes_total{op=\"exist\"} " + load(reroutes_[kRrExist]) + "\n";
    s += "dfkv_client_cooldown_reroutes_total{op=\"put\"} " + load(reroutes_[kRrPut]) + "\n";
    s += "dfkv_client_cooldown_reroutes_total{op=\"retain\"} " + load(reroutes_[kRrRetain]) + "\n";
    s += "dfkv_client_cooldown_reroutes_total{op=\"none\"} " + load(reroutes_[kRrNone]) + "\n";
    size_t episode_pending = 0;
    {
      std::lock_guard<std::mutex> lk(reroute_mu_);
      episode_pending = episode_pending_;
    }
    s += "# HELP dfkv_client_readback_reads_total Previous-owner read-backs (DFKV_RING_READBACK_MS): hit, miss, and the candidates refused (removed or unknown in some asked node's M, a counted length mismatch, the registry gate closed, the previous owner cooling); hit_rewritten: read back because the previous owner's write followed every reported REMOVE (DFKV_RING_READBACK_REWRITE; counted in addition to hit or miss)\n";
    s += "# TYPE dfkv_client_readback_reads_total counter\n";
    s += "dfkv_client_readback_reads_total{result=\"hit\"} " + load(readback_hit_) + "\n";
    s += "dfkv_client_readback_reads_total{result=\"miss\"} " + load(readback_miss_) + "\n";
    s += "dfkv_client_readback_reads_total{result=\"refused_removed\"} " + load(readback_refused_[kRbRemoved]) + "\n";
    s += "dfkv_client_readback_reads_total{result=\"refused_unknown\"} " + load(readback_refused_[kRbUnknown]) + "\n";
    s += "dfkv_client_readback_reads_total{result=\"refused_length\"} " + load(readback_refused_[kRbLength]) + "\n";
    s += "dfkv_client_readback_reads_total{result=\"refused_gate\"} " + load(readback_refused_[kRbGate]) + "\n";
    s += "dfkv_client_readback_reads_total{result=\"refused_cooling\"} " + load(readback_refused_[kRbCooling]) + "\n";
    s += "dfkv_client_readback_reads_total{result=\"hit_rewritten\"} " + load(readback_rewritten_) + "\n";
    s += "# HELP dfkv_client_readback_changes_refused_total Adopted write-ring changes given no read-back window: held (the guard held the change longer than settle - poll_max), chained (a second change inside an open window)\n";
    s += "# TYPE dfkv_client_readback_changes_refused_total counter\n";
    s += "dfkv_client_readback_changes_refused_total{reason=\"held\"} " + load(readback_changes_held_) + "\n";
    s += "dfkv_client_readback_changes_refused_total{reason=\"chained\"} " + load(readback_changes_chained_) + "\n";
    s += "# HELP dfkv_client_readback_memory_short_total Nodes whose recent-REMOVE memory stayed shorter than H for longer than H\n";
    s += "# TYPE dfkv_client_readback_memory_short_total counter\n";
    s += "dfkv_client_readback_memory_short_total{reason=\"window\"} " + load(readback_memory_short_) + "\n";
    {
      std::lock_guard<std::mutex> lk(ring_mu_);
      s += "# HELP dfkv_client_readback_window_active 1 while a read-back window is open\n";
      s += "# TYPE dfkv_client_readback_window_active gauge\n";
      s += std::string("dfkv_client_readback_window_active ") + (NowMs() < held_until_ms_ ? "1" : "0") + "\n";
    }
    s += "# HELP dfkv_client_ring_reroute_episode_pending Keys of ended reroute episodes whose REMOVE at the substitute is not confirmed yet; they count toward DFKV_RING_REROUTE_KEYS with the tracked keys\n";
    s += "# TYPE dfkv_client_ring_reroute_episode_pending gauge\n";
    s += "dfkv_client_ring_reroute_episode_pending " + std::to_string(episode_pending) + "\n";
    s += "# HELP dfkv_client_reroute_refused_total Rerouted writes refused because DFKV_RING_REROUTE_KEYS keys are tracked for cleanup\n";
    s += "# TYPE dfkv_client_reroute_refused_total counter\n";
    s += "dfkv_client_reroute_refused_total " + load(reroute_refused_) + "\n";
    s += "# HELP dfkv_client_reroute_cleanup_total REMOVEs of keys written to substitutes (episode end, or a fan-out target that cooled): ok, queued (sent again from the tick), abandoned (over the bound, or not sent before a close)\n";
    s += "# TYPE dfkv_client_reroute_cleanup_total counter\n";
    s += "dfkv_client_reroute_cleanup_total{result=\"ok\"} " + load(cleanup_ok_) + "\n";
    s += "dfkv_client_reroute_cleanup_total{result=\"queued\"} " + load(cleanup_queued_total_) + "\n";
    s += "dfkv_client_reroute_cleanup_total{result=\"abandoned\"} " + load(cleanup_abandoned_) + "\n";
    s += "# HELP dfkv_client_remove_fanout_total REMOVE fan-out messages: ok, error, queued_cooling (the target cools in this client: queued for cleanup)\n";
    s += "# TYPE dfkv_client_remove_fanout_total counter\n";
    s += "dfkv_client_remove_fanout_total{result=\"ok\"} " + load(fanout_ok_) + "\n";
    s += "dfkv_client_remove_fanout_total{result=\"error\"} " + load(fanout_error_) + "\n";
    s += "dfkv_client_remove_fanout_total{result=\"queued_cooling\"} " + load(fanout_queued_cooling_) + "\n";
    s += "# HELP dfkv_client_remove_unconfirmed_total REMOVEs answered not ok because a target did not confirm in the call: cooling (in this client), failed\n";
    s += "# TYPE dfkv_client_remove_unconfirmed_total counter\n";
    s += "dfkv_client_remove_unconfirmed_total{reason=\"cooling\"} " + load(unconfirmed_cooling_) + "\n";
    s += "dfkv_client_remove_unconfirmed_total{reason=\"failed\"} " + load(unconfirmed_failed_) + "\n";
    s += "# HELP dfkv_client_registry_gate_open 1 while the registry gate passes (cooldown reroute may act)\n";
    s += "# TYPE dfkv_client_registry_gate_open gauge\n";
    s += std::string("dfkv_client_registry_gate_open ") + (gate_open_.load(std::memory_order_relaxed) ? "1" : "0") + "\n";
    s += "# HELP dfkv_client_registry_gate_failures_total Registry gate transitions from open to closed, by reason\n";
    s += "# TYPE dfkv_client_registry_gate_failures_total counter\n";
    s += "dfkv_client_registry_gate_failures_total{reason=\"unmarked_client\"} " + load(gate_failures_[kGateUnmarked]) + "\n";
    s += "dfkv_client_registry_gate_failures_total{reason=\"stale_registry\"} " + load(gate_failures_[kGateStale]) + "\n";
    s += "dfkv_client_registry_gate_failures_total{reason=\"no_registry\"} " + load(gate_failures_[kGateNoRegistry]) + "\n";
    s += "dfkv_client_registry_gate_failures_total{reason=\"s_growing\"} " + load(gate_failures_[kGateSGrowing]) + "\n";
    s += "# HELP dfkv_client_registry_unmarked_clients Registered clients without the fan-out marker (or its poll_ms) in the last registry answer\n";
    s += "# TYPE dfkv_client_registry_unmarked_clients gauge\n";
    s += "dfkv_client_registry_unmarked_clients " + std::to_string(unmarked_gauge_.load(std::memory_order_relaxed)) + "\n";
    s += "# HELP dfkv_client_ring_point_collisions Ring points two members hashed to; the member added later owns each\n";
    s += "# TYPE dfkv_client_ring_point_collisions gauge\n";
    s += "dfkv_client_ring_point_collisions " + std::to_string(collisions) + "\n";
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
  const Status st = t_->Members(seed_addr, &text);
  t_->TakeUnreachableRefusal();  // not reported to PeerHealth; see ReportPeer
  if (st != Status::kOk) return false;
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

std::vector<std::string> KVClient::MemberAddresses() const {
  std::vector<std::string> out;
  {
    std::lock_guard<std::mutex> lk(ring_mu_);
    for (const auto& member : addr_) out.push_back(member.second);
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

std::string KVClient::Route(const std::string& key) const {
  if (t_drain_pass) return DrainRoute(key);
  if (t_route_to != nullptr) return *t_route_to;
  // Hashed before the lock: every routed key of every thread takes it.
  const uint32_t hash = ConHash::KeyHash(key);
  std::lock_guard<std::mutex> lk(ring_mu_);
  std::string name;
  if (!ring_.LookupHash(hash, &name)) return "";
  auto it = addr_.find(name);
  return it == addr_.end() ? "" : it->second;
}

std::string KVClient::DrainRoute(const std::string& key) const {
  if (!any_draining_.load(std::memory_order_relaxed)) return "";
  const uint32_t hash = ConHash::KeyHash(key);
  std::lock_guard<std::mutex> lk(ring_mu_);
  std::string name;
  if (!full_ring_.LookupHash(hash, &name) || draining_.count(name) == 0)
    return "";
  auto it = addr_.find(name);
  return it == addr_.end() ? "" : it->second;
}

std::string KVClient::AbsentRoute(const std::string& key,
                                  std::string* id) const {
  if (!any_absent_.load(std::memory_order_relaxed)) return "";
  const uint32_t hash = ConHash::KeyHash(key);
  std::lock_guard<std::mutex> lk(ring_mu_);
  std::string name;
  if (!absent_ring_.LookupHash(hash, &name)) return "";
  const auto it = absent_.find(name);
  if (it == absent_.end()) return "";
  *id = name;
  return it->second.addr;
}

std::vector<KVClient::AbsentPart> KVClient::AbsentOwned(
    const std::vector<std::string>& keys) const {
  std::map<std::string, AbsentPart> by_member;
  if (!any_absent_.load(std::memory_order_relaxed)) return {};
  for (size_t i = 0; i < keys.size(); ++i) {
    if (keys[i].empty()) continue;
    std::string id;
    const std::string addr = AbsentRoute(keys[i], &id);
    if (addr.empty()) continue;
    AbsentPart& part = by_member[id];
    part.id = id;
    part.addr = addr;
    part.index.push_back(i);
  }
  std::vector<AbsentPart> out;
  for (auto& [id, part] : by_member) {
    (void)id;
    out.push_back(std::move(part));
  }
  return out;
}

void KVClient::RemoveAtAbsent(const std::vector<std::string>& keys) {
  for (const AbsentPart& part : AbsentOwned(keys)) {
    std::vector<std::string> sub;
    sub.reserve(part.index.size());
    for (size_t i : part.index) sub.push_back(keys[i]);
    std::vector<char> ok(sub.size(), 0);
    {
      RouteTo to(part.addr);
      if (t_->pipelined()) {
        ok = BatchRemoveRouted(sub);
      } else {
        for (size_t m = 0; m < sub.size(); ++m)
          ok[m] = RemoveDirectRouted(sub[m]) ? 1 : 0;
      }
    }
    for (size_t m = 0; m < sub.size(); ++m) {
      (ok[m] ? drain_removes_ok_ : drain_removes_failed_)
          .fetch_add(1, std::memory_order_relaxed);
      if (!ok[m]) QueueAbsentRemove(part.id, sub[m]);
    }
  }
}

template <typename KeyOf>
std::vector<size_t> KVClient::DrainerOwned(size_t n,
                                           const std::vector<bool>* done,
                                           KeyOf key_of) const {
  std::vector<size_t> out;
  if (t_drain_pass || !any_draining_.load(std::memory_order_relaxed))
    return out;
  for (size_t i = 0; i < n; ++i) {
    if (done != nullptr && (*done)[i]) continue;
    const std::string& key = key_of(i);
    if (!key.empty() && !DrainRoute(key).empty()) out.push_back(i);
  }
  return out;
}

void KVClient::CountDrainRead(bool hit) {
  (hit ? drain_reads_hit_ : drain_reads_miss_)
      .fetch_add(1, std::memory_order_relaxed);
}

// Scalar reads: the write owner first; after a miss, a draining full-ring
// owner (the key may have been written before it drained).
bool KVClient::GetDirect(const std::string& key, void* out, size_t n,
                         ItemStatus* status) {
  ItemStatus first = ItemStatus::kNoRoute;
  if (GetDirectRouted(key, out, n, &first)) {
    if (status) *status = first;
    return true;
  }
  if (status) *status = first;
  if (!t_drain_pass && !DrainRoute(key).empty()) {
    ItemStatus second = ItemStatus::kNoRoute;
    bool hit;
    {
      DrainPass pass;
      hit = GetDirectRouted(key, out, n, &second);
    }
    CountDrainRead(hit);
    if (hit) {
      if (status) *status = second;
      return true;
    }
  }
  // Previous-owner read-back (stage B): only after the owner's not found.
  if (!readback_on_ || first != ItemStatus::kNotFound || t_drain_pass ||
      t_route_to != nullptr)
    return false;
  const auto plan = PlanReadBack({key});
  if (plan.empty()) return false;
  ItemStatus back = ItemStatus::kNoRoute;
  bool hit;
  {
    RouteTo to(plan.begin()->first);
    hit = GetDirectRouted(key, out, n, &back);
  }
  CountReadBack(hit);
  if (back == ItemStatus::kLengthMismatch)
    length_mismatch_seen_.store(true, std::memory_order_relaxed);
  if (hit && status) *status = back;
  return hit;
}

bool KVClient::GetAutoDirect(const std::string& key, std::string* out,
                             size_t max_bytes) {
  if (GetAutoDirectRouted(key, out, max_bytes)) return true;
  if (t_drain_pass || DrainRoute(key).empty()) return false;
  bool hit;
  {
    DrainPass pass;
    hit = GetAutoDirectRouted(key, out, max_bytes);
  }
  CountDrainRead(hit);
  return hit;
}

bool KVClient::GetAutoDirect(const std::string& key, void* out, size_t cap,
                             size_t* out_len, ItemStatus* status) {
  ItemStatus first = ItemStatus::kNoRoute;
  if (GetAutoDirectRouted(key, out, cap, out_len, &first)) {
    if (status) *status = first;
    return true;
  }
  if (status) *status = first;
  if (!t_drain_pass && !DrainRoute(key).empty()) {
    ItemStatus second = ItemStatus::kNoRoute;
    bool hit;
    {
      DrainPass pass;
      hit = GetAutoDirectRouted(key, out, cap, out_len, &second);
    }
    CountDrainRead(hit);
    if (hit) {
      if (status) *status = second;
      return true;
    }
  }
  if (!readback_on_ || first != ItemStatus::kNotFound || t_drain_pass ||
      t_route_to != nullptr)
    return false;
  const auto plan = PlanReadBack({key});
  if (plan.empty()) return false;
  ItemStatus back = ItemStatus::kNoRoute;
  bool hit;
  {
    RouteTo to(plan.begin()->first);
    hit = GetAutoDirectRouted(key, out, cap, out_len, &back);
  }
  CountReadBack(hit);
  if (back == ItemStatus::kLengthMismatch)
    length_mismatch_seen_.store(true, std::memory_order_relaxed);
  if (hit && status) *status = back;
  return hit;
}

bool KVClient::ExistDirect(const std::string& key, Status* status) {
  Status first = Status::kInvalid;
  if (ExistDirectRouted(key, &first)) {
    if (status) *status = first;
    return true;
  }
  if (status) *status = first;
  if (!t_drain_pass && !DrainRoute(key).empty()) {
    Status second = Status::kInvalid;
    bool hit;
    {
      DrainPass pass;
      hit = ExistDirectRouted(key, &second);
    }
    CountDrainRead(hit);
    if (hit) {
      if (status) *status = second;
      return true;
    }
  }
  // The owner answered and does not hold it (EXIST reports an absent key as
  // kOk, not present).
  if (!readback_on_ || first != Status::kOk || t_drain_pass ||
      t_route_to != nullptr)
    return false;
  const auto plan = PlanReadBack({key});
  if (plan.empty()) return false;
  Status back = Status::kInvalid;
  bool hit;
  {
    RouteTo to(plan.begin()->first);
    hit = ExistDirectRouted(key, &back);
  }
  CountReadBack(hit);
  if (hit && status) *status = back;
  return hit;
}

// A key whose full-ring owner drains is removed from both: one left on the
// drainer would be served by the read after a miss. A key whose owner with
// the absent members (absent grace) is absent is removed there too; that
// member (usually down) need not confirm, its part is queued for its return.
bool KVClient::RemoveDirect(const std::string& key) {
  if (fanout_on_ && !t_drain_pass && t_route_to == nullptr)
    return RemoveFanout({key})[0] != 0;
  const bool ok = RemoveDirectRouted(key);
  if (t_drain_pass || t_route_to != nullptr) return ok;
  if (any_absent_.load(std::memory_order_relaxed)) RemoveAtAbsent({key});
  if (DrainRoute(key).empty()) return ok;
  bool drained;
  {
    DrainPass pass;
    drained = RemoveDirectRouted(key);
  }
  (drained ? drain_removes_ok_ : drain_removes_failed_)
      .fetch_add(1, std::memory_order_relaxed);
  return ok && drained;
}

// Batch reads: the TCP bodies call the scalar forms above, which already
// ask drainers; the RDMA (pipelined) bodies route in batches, so their
// misses owned by a draining member go there in one more batch.
std::vector<bool> KVClient::BatchGetDirect(
    const std::vector<KvGetItem>& items,
    const std::function<void(size_t)>& on_hit, ItemStatus* st) {
  // Read-back needs each key's answer, also when the caller wants none.
  std::vector<ItemStatus> own;
  if (!st && readback_on_ && t_->pipelined()) {
    own.assign(items.size(), ItemStatus::kNoRoute);
    st = own.data();
  }
  std::vector<bool> hit = BatchGetDirectRouted(items, on_hit, st);
  if (!t_->pipelined()) return hit;  // the scalar bodies do both passes
  std::vector<ItemStatus> first;
  if (st) first.assign(st, st + items.size());
  const std::vector<size_t> again = DrainerOwned(
      items.size(), &hit,
      [&](size_t i) -> const std::string& { return items[i].key; });
  if (!again.empty()) {
    std::vector<KvGetItem> sub;
    sub.reserve(again.size());
    for (size_t i : again) sub.push_back(items[i]);
    std::vector<ItemStatus> sub_st(sub.size(), ItemStatus::kNoRoute);
    std::vector<bool> sub_hit;
    {
      DrainPass pass;
      sub_hit = BatchGetDirectRouted(
          sub, on_hit ? std::function<void(size_t)>([&](size_t m) { on_hit(again[m]); })
                      : std::function<void(size_t)>(),
          sub_st.data());
    }
    for (size_t m = 0; m < again.size(); ++m) {
      CountDrainRead(sub_hit[m]);
      if (!sub_hit[m]) continue;
      hit[again[m]] = true;
      if (st) st[again[m]] = sub_st[m];
    }
  }
  if (!readback_on_ || t_drain_pass || t_route_to != nullptr || !st) return hit;
  std::vector<size_t> missed;
  std::vector<std::string> missed_keys;
  for (size_t i = 0; i < items.size(); ++i)
    if (!hit[i] && first[i] == ItemStatus::kNotFound) {
      missed.push_back(i);
      missed_keys.push_back(items[i].key);
    }
  for (const auto& [prev, pos] : PlanReadBack(missed_keys)) {
    std::vector<KvGetItem> sub;
    for (size_t m : pos) sub.push_back(items[missed[m]]);
    std::vector<ItemStatus> sub_st(sub.size(), ItemStatus::kNoRoute);
    std::vector<bool> sub_hit;
    {
      RouteTo to(prev);
      sub_hit = BatchGetDirectRouted(
          sub,
          on_hit ? std::function<void(size_t)>(
                       [&](size_t k) { on_hit(missed[pos[k]]); })
                 : std::function<void(size_t)>(),
          sub_st.data());
    }
    for (size_t k = 0; k < pos.size(); ++k) {
      CountReadBack(sub_hit[k]);
      if (sub_st[k] == ItemStatus::kLengthMismatch)
        length_mismatch_seen_.store(true, std::memory_order_relaxed);
      if (!sub_hit[k]) continue;
      hit[missed[pos[k]]] = true;
      st[missed[pos[k]]] = sub_st[k];
    }
  }
  return hit;
}

std::vector<bool> KVClient::BatchGetAutoDirect(
    const std::vector<KvGetItem>& items, std::vector<size_t>* out_lens,
    const std::function<void(size_t, size_t)>& on_hit, ItemStatus* st) {
  std::vector<ItemStatus> own;
  if (!st && readback_on_ && t_->pipelined()) {
    own.assign(items.size(), ItemStatus::kNoRoute);
    st = own.data();
  }
  std::vector<bool> hit = BatchGetAutoDirectRouted(items, out_lens, on_hit, st);
  if (!t_->pipelined()) return hit;  // the scalar bodies do both passes
  std::vector<ItemStatus> first;
  if (st) first.assign(st, st + items.size());
  const std::vector<size_t> again = DrainerOwned(
      items.size(), &hit,
      [&](size_t i) -> const std::string& { return items[i].key; });
  if (!again.empty()) {
    std::vector<KvGetItem> sub;
    sub.reserve(again.size());
    for (size_t i : again) sub.push_back(items[i]);
    std::vector<ItemStatus> sub_st(sub.size(), ItemStatus::kNoRoute);
    std::vector<size_t> sub_lens;
    std::vector<bool> sub_hit;
    {
      DrainPass pass;
      sub_hit = BatchGetAutoDirectRouted(
          sub, &sub_lens,
          on_hit ? std::function<void(size_t, size_t)>(
                       [&](size_t m, size_t len) { on_hit(again[m], len); })
                 : std::function<void(size_t, size_t)>(),
          sub_st.data());
    }
    for (size_t m = 0; m < again.size(); ++m) {
      CountDrainRead(sub_hit[m]);
      if (!sub_hit[m]) continue;
      hit[again[m]] = true;
      if (st) st[again[m]] = sub_st[m];
      if (out_lens && again[m] < out_lens->size() && m < sub_lens.size())
        (*out_lens)[again[m]] = sub_lens[m];
    }
  }
  if (!readback_on_ || t_drain_pass || t_route_to != nullptr || !st) return hit;
  std::vector<size_t> missed;
  std::vector<std::string> missed_keys;
  for (size_t i = 0; i < items.size(); ++i)
    if (!hit[i] && first[i] == ItemStatus::kNotFound) {
      missed.push_back(i);
      missed_keys.push_back(items[i].key);
    }
  for (const auto& [prev, pos] : PlanReadBack(missed_keys)) {
    std::vector<KvGetItem> sub;
    for (size_t m : pos) sub.push_back(items[missed[m]]);
    std::vector<ItemStatus> sub_st(sub.size(), ItemStatus::kNoRoute);
    std::vector<size_t> sub_lens;
    std::vector<bool> sub_hit;
    {
      RouteTo to(prev);
      sub_hit = BatchGetAutoDirectRouted(
          sub, &sub_lens,
          on_hit ? std::function<void(size_t, size_t)>(
                       [&](size_t k, size_t len) { on_hit(missed[pos[k]], len); })
                 : std::function<void(size_t, size_t)>(),
          sub_st.data());
    }
    for (size_t k = 0; k < pos.size(); ++k) {
      CountReadBack(sub_hit[k]);
      if (sub_st[k] == ItemStatus::kLengthMismatch)
        length_mismatch_seen_.store(true, std::memory_order_relaxed);
      if (!sub_hit[k]) continue;
      const size_t i = missed[pos[k]];
      hit[i] = true;
      st[i] = sub_st[k];
      if (out_lens && i < out_lens->size() && k < sub_lens.size())
        (*out_lens)[i] = sub_lens[k];
    }
  }
  return hit;
}

// No TCP body of its own: every transport routes this one in batches.
std::vector<bool> KVClient::BatchGetAutoSgDirect(
    const std::vector<KvGetItemSg>& items, std::vector<size_t>* out_lens,
    ItemStatus* st) {
  std::vector<ItemStatus> own;
  if (!st && readback_on_) {
    own.assign(items.size(), ItemStatus::kNoRoute);
    st = own.data();
  }
  std::vector<bool> hit = BatchGetAutoSgDirectRouted(items, out_lens, st);
  std::vector<ItemStatus> first;
  if (st) first.assign(st, st + items.size());
  const std::vector<size_t> again = DrainerOwned(
      items.size(), &hit,
      [&](size_t i) -> const std::string& { return items[i].key; });
  if (!again.empty()) {
    std::vector<KvGetItemSg> sub;
    sub.reserve(again.size());
    for (size_t i : again) sub.push_back(items[i]);
    std::vector<ItemStatus> sub_st(sub.size(), ItemStatus::kNoRoute);
    std::vector<size_t> sub_lens;
    std::vector<bool> sub_hit;
    {
      DrainPass pass;
      sub_hit = BatchGetAutoSgDirectRouted(sub, &sub_lens, sub_st.data());
    }
    for (size_t m = 0; m < again.size(); ++m) {
      CountDrainRead(sub_hit[m]);
      if (!sub_hit[m]) continue;
      hit[again[m]] = true;
      if (st) st[again[m]] = sub_st[m];
      if (out_lens && again[m] < out_lens->size() && m < sub_lens.size())
        (*out_lens)[again[m]] = sub_lens[m];
    }
  }
  if (!readback_on_ || t_drain_pass || t_route_to != nullptr || !st) return hit;
  std::vector<size_t> missed;
  std::vector<std::string> missed_keys;
  for (size_t i = 0; i < items.size(); ++i)
    if (!hit[i] && first[i] == ItemStatus::kNotFound) {
      missed.push_back(i);
      missed_keys.push_back(items[i].key);
    }
  for (const auto& [prev, pos] : PlanReadBack(missed_keys)) {
    std::vector<KvGetItemSg> sub;
    for (size_t m : pos) sub.push_back(items[missed[m]]);
    std::vector<ItemStatus> sub_st(sub.size(), ItemStatus::kNoRoute);
    std::vector<size_t> sub_lens;
    std::vector<bool> sub_hit;
    {
      RouteTo to(prev);
      sub_hit = BatchGetAutoSgDirectRouted(sub, &sub_lens, sub_st.data());
    }
    for (size_t k = 0; k < pos.size(); ++k) {
      CountReadBack(sub_hit[k]);
      if (sub_st[k] == ItemStatus::kLengthMismatch)
        length_mismatch_seen_.store(true, std::memory_order_relaxed);
      if (!sub_hit[k]) continue;
      const size_t i = missed[pos[k]];
      hit[i] = true;
      st[i] = sub_st[k];
      if (out_lens && i < out_lens->size() && k < sub_lens.size())
        (*out_lens)[i] = sub_lens[k];
    }
  }
  return hit;
}

std::vector<bool> KVClient::BatchExistDirect(
    const std::vector<std::string>& keys, std::vector<Status>* statuses) {
  std::vector<Status> own;
  if (!statuses && readback_on_ && t_->pipelined()) statuses = &own;
  std::vector<bool> hit = BatchExistDirectRouted(keys, statuses);
  if (!t_->pipelined()) return hit;  // the scalar bodies do both passes
  std::vector<Status> first;
  if (statuses) first = *statuses;
  const std::vector<size_t> again = DrainerOwned(
      keys.size(), &hit, [&](size_t i) -> const std::string& { return keys[i]; });
  if (!again.empty()) {
    std::vector<std::string> sub;
    sub.reserve(again.size());
    for (size_t i : again) sub.push_back(keys[i]);
    std::vector<Status> sub_st;
    std::vector<bool> sub_hit;
    {
      DrainPass pass;
      sub_hit = BatchExistDirectRouted(sub, statuses ? &sub_st : nullptr);
    }
    for (size_t m = 0; m < again.size(); ++m) {
      CountDrainRead(sub_hit[m]);
      if (!sub_hit[m]) continue;
      hit[again[m]] = true;
      if (statuses && again[m] < statuses->size() && m < sub_st.size())
        (*statuses)[again[m]] = sub_st[m];
    }
  }
  if (!readback_on_ || t_drain_pass || t_route_to != nullptr || !statuses)
    return hit;
  std::vector<size_t> missed;
  std::vector<std::string> missed_keys;
  for (size_t i = 0; i < keys.size() && i < first.size(); ++i)
    if (!hit[i] && (first[i] == Status::kNotFound || first[i] == Status::kOk)) {
      missed.push_back(i);
      missed_keys.push_back(keys[i]);
    }
  for (const auto& [prev, pos] : PlanReadBack(missed_keys)) {
    std::vector<std::string> sub;
    for (size_t m : pos) sub.push_back(keys[missed[m]]);
    std::vector<Status> sub_st;
    std::vector<bool> sub_hit;
    {
      RouteTo to(prev);
      sub_hit = BatchExistDirectRouted(sub, &sub_st);
    }
    for (size_t k = 0; k < pos.size(); ++k) {
      CountReadBack(sub_hit[k]);
      if (!sub_hit[k]) continue;
      const size_t i = missed[pos[k]];
      hit[i] = true;
      if (i < statuses->size() && k < sub_st.size()) (*statuses)[i] = sub_st[k];
    }
  }
  return hit;
}

bool KVClient::Put(const std::string& key, const void* value, size_t n) {
  auto metric = op_stats_.Begin(OpMetrics::kPut);
  const bool ok = PutDirect(key, value, n);
  metric.hits = ok ? 1 : 0;
  metric.bytes = ok ? n : 0;
  return ok;
}

bool KVClient::PutDirect(const std::string& key, const void* value, size_t n,
                         ItemStatus* status) {
  ItemStatus unused;
  ItemStatus& outcome = status ? *status : unused;
  outcome = ItemStatus::kBadArgument;
  if (key.empty() || n == 0 || value == nullptr) return false;
  outcome = ItemStatus::kNoRoute;
  std::string node = RouteFor(key, RouteOp::kPut);
  if (node.empty()) return false;
  outcome = ItemStatus::kPeerCooling;
  uint64_t now = NowMs();
  if (!health_.Healthy(node, now)) return false;
  const BlockKey bk = block_key_(key);
  Status st;
  if (t_->pipelined()) {
    std::vector<CacheSrc> srcs{CacheSrc{bk, value, n}};
    st = t_->CacheFrom(node, srcs)[0];
  } else {
    st = t_->Cache(node, bk, value, n);
  }
  outcome = ToItemStatus(st);
  ReportPeer(node, st);
  if (st == Status::kIOError) {
    return false;
  }
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

bool KVClient::GetDirectRouted(const std::string& key, void* out, size_t n,
                               ItemStatus* status) {
  ItemStatus unused;
  ItemStatus& outcome = status ? *status : unused;
  outcome = ItemStatus::kBadArgument;
  if (key.empty()) return false;  // as PUT
  outcome = ItemStatus::kNoRoute;
  std::string node = RouteFor(key, RouteOp::kGet);
  if (node.empty()) return false;
  outcome = ItemStatus::kPeerCooling;
  uint64_t now = NowMs();
  if (!health_.Healthy(node, now)) return false;
  const BlockKey bk = block_key_(key);
  uint64_t value_len = 0;
  std::vector<BlockKey> keys{bk};
  std::vector<RangeDst> dsts{
      RangeDst{n ? out : nullptr, n, ClassifyDestination(out, n),
               /*exact=*/true}};
  std::vector<uint64_t> value_lens;
  const Status st = t_->RangeInto(node, keys, dsts, &value_lens)[0];
  if (!value_lens.empty()) value_len = value_lens[0];
  outcome = ReadItemStatus(st, value_len, n, /*exact=*/true);
  ReportPeer(node, st);
  if (st == Status::kIOError) {
    return false;
  }
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

bool KVClient::GetAutoDirectRouted(const std::string& key, std::string* out,
                                   size_t max_bytes) {
  std::string node = RouteFor(key, RouteOp::kGet);
  if (node.empty()) return false;
  uint64_t now = NowMs();
  if (!health_.Healthy(node, now)) return false;
  const BlockKey bk = block_key_(key);
  uint64_t value_len = 0;
  // An uninitialized buffer: std::string::resize zero-filled all max_bytes
  // (64 MiB by default, about 20 ms) on every call, for a value of any size.
  std::unique_ptr<char[]> buffer(max_bytes ? new char[max_bytes] : nullptr);
  std::vector<BlockKey> keys{bk};
  std::vector<RangeDst> dsts{RangeDst{buffer.get(), max_bytes,
                                      DestinationMemoryKind::kHost}};
  std::vector<uint64_t> value_lens;
  const Status st = t_->RangeInto(node, keys, dsts, &value_lens)[0];
  if (!value_lens.empty()) value_len = value_lens[0];
  ReportPeer(node, st);
  out->clear();
  if (st != Status::kOk || value_len > max_bytes) return false;
  out->assign(buffer.get(), static_cast<size_t>(value_len));
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

bool KVClient::GetAutoDirectRouted(const std::string& key, void* out,
                                   size_t cap, size_t* out_len,
                                   ItemStatus* status) {
  ItemStatus unused;
  ItemStatus& outcome = status ? *status : unused;
  if (out_len) *out_len = 0;
  outcome = ItemStatus::kBadArgument;
  if (key.empty()) return false;  // as PUT
  outcome = ItemStatus::kNoRoute;
  std::string node = RouteFor(key, RouteOp::kGet);
  if (node.empty()) return false;
  outcome = ItemStatus::kPeerCooling;
  uint64_t now = NowMs();
  if (!health_.Healthy(node, now)) return false;
  const BlockKey bk = block_key_(key);
  uint64_t value_len = 0;
  std::vector<BlockKey> keys{bk};
  std::vector<RangeDst> dsts{
      RangeDst{cap ? out : nullptr, cap, ClassifyDestination(out, cap)}};
  std::vector<uint64_t> value_lens;
  const Status st = t_->RangeInto(node, keys, dsts, &value_lens)[0];
  if (!value_lens.empty()) value_len = value_lens[0];
  outcome = ReadItemStatus(st, value_len, cap, /*exact=*/false);
  ReportPeer(node, st);
  if (st == Status::kIOError) {
    return false;
  }
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

bool KVClient::ExistDirectRouted(const std::string& key, Status* status) {
  if (status) *status = Status::kInvalid;
  std::string node = RouteFor(key, RouteOp::kExist);
  if (node.empty()) return false;
  uint64_t now = NowMs();
  if (!health_.Healthy(node, now)) return false;
  bool exists = false;
  const Status st = t_->Exist(node, block_key_(key), &exists);
  if (status) *status = st;
  ReportPeer(node, st);
  if (st == Status::kIOError) {
    return false;
  }
  return st == Status::kOk && exists;
}

bool KVClient::Remove(const std::string& key) {
  auto metric = op_stats_.Begin(OpMetrics::kRemove);
  const bool ok = RemoveDirect(key);
  metric.hits = ok ? 1 : 0;
  return ok;
}

void KVClient::InvalidateRendezvous(const BlockKey& key) {
  ForgetRetained(key);
  if (dedup_) dedup_->Invalidate(key);
  if (GpuNodeDedup* gd = gpu_dedup_raw_.load(std::memory_order_acquire))
    gd->Invalidate(key);
}

bool KVClient::RemoveDirectRouted(const std::string& key) {
  std::string node = Route(key);
  if (node.empty()) return false;
  uint64_t now = NowMs();
  if (!health_.Healthy(node, now)) return false;
  const BlockKey block_key = block_key_(key);
  const Status st = t_->Remove(node, block_key);
  ReportPeer(node, st);
  if (st == Status::kIOError) {
    return false;
  }
  const bool ok = st == Status::kOk || st == Status::kNotFound;
  if (ok) InvalidateRendezvous(block_key);
  return ok;
}

std::vector<bool> KVClient::BatchPut(const std::vector<KvPutItem>& items,
                                     std::vector<ItemStatus>* status) {
  auto t0 = std::chrono::steady_clock::now();
  const size_t N = items.size();
  std::vector<char> ok(N, 0);  // char (not vector<bool>) for thread-safe writes
  ItemStatus* st = nullptr;
  if (status) {
    status->assign(N, ItemStatus::kNoRoute);
    st = status->data();
  }
  if (!t_->pipelined()) {  // TCP fan-out uses scalar bodies, not scalar metrics.
    RunParallel(N, BatchWorkers(N), [&](size_t i) {
      ok[i] = PutDirect(items[i].key, items[i].value, items[i].n,
                        st ? st + i : nullptr)
                  ? 1
                  : 0;
    });
  } else {
  // RDMA: group by node and scatter-send raw caller values without copying.
  std::map<std::string, std::vector<size_t>> by_node;
  for (size_t i = 0; i < N; ++i) {
    if (items[i].key.empty() || items[i].n == 0 ||
        items[i].value == nullptr) {
      if (st) st[i] = ItemStatus::kBadArgument;
      continue;
    }
    std::string node = RouteFor(items[i].key, RouteOp::kPut);
    if (node.empty()) continue;
    by_node[node].push_back(i);
  }
  // Shard each node's group as BatchPutSg does: one connection would drain a
  // large group key by key.
  std::vector<std::pair<std::string, std::vector<size_t>>> groups =
      ShardReadGroups(std::vector<std::pair<std::string, std::vector<size_t>>>(
          by_node.begin(), by_node.end()));
  RunParallel(groups.size(), BatchWorkers(groups.size()), [&](size_t g) {
    const std::string& node = groups[g].first;
    const std::vector<size_t>& idx = groups[g].second;
    uint64_t now = NowMs();
    if (!health_.Healthy(node, now)) {
      if (st)
        for (size_t k : idx) st[k] = ItemStatus::kPeerCooling;
      return;
    }
    std::vector<CacheSrc> srcs;
    srcs.reserve(idx.size());
    for (size_t k : idx)
      srcs.push_back(CacheSrc{block_key_(items[k].key),
                              items[k].value, items[k].n});
    std::vector<Status> sts = t_->CacheFrom(node, srcs);
    for (size_t m = 0; m < idx.size(); ++m) {
      // A transport that answered for fewer keys stored none of the rest.
      const Status s = m < sts.size() ? sts[m] : Status::kInvalid;
      ok[idx[m]] = (s == Status::kOk) ? 1 : 0;
      if (st) st[idx[m]] = ToItemStatus(s);
      if (ok[idx[m]]) InvalidateRendezvous(srcs[m].key);
    }
    ReportPeer(node, sts);
  });
  }
  uint64_t bytes = 0;
  for (size_t i = 0; i < N; ++i) if (ok[i]) bytes += items[i].n;
  return RecordBatch(OpMetrics::kPut, t0,
                     std::vector<bool>(ok.begin(), ok.end()), bytes);
}

std::vector<bool> KVClient::BatchGet(const std::vector<KvGetItem>& items,
                                     std::vector<ItemStatus>* status) {
  const auto t0 = std::chrono::steady_clock::now();
  ItemStatus* st = nullptr;
  if (status) {
    status->assign(items.size(), ItemStatus::kNoRoute);
    st = status->data();
  }
  if (!dedup_) {
    auto result = BatchGetDirect(items, nullptr, st);
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
  AbortClaimsOnThrow abort_claims([&] {
    for (size_t i = 0; i < N; ++i)
      if (tokens[i]) dedup_->Abort(bks[i], NodeDedup::Kind::kData, tokens[i]);
  });
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
    bks[i] = block_key_(items[i].key);
    switch (dedup_->Claim(bks[i], items[i].n,
                          static_cast<char*>(items[i].out), &tokens[i])) {
      case NodeDedup::Role::kHit:
        res[i] = true;
        if (st) st[i] = ItemStatus::kOk;
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
    // Each key is published as soon as it has arrived, not when the whole
    // batch has: a waiter's budget covers one batch, and a slow part of
    // this batch must not cost it the keys already here. Publishing is the
    // same lock-free protocol other processes run concurrently; a key that
    // is not a hit in the end is aborted once the batch is done.
    std::vector<ItemStatus> fst(st ? fetch_items.size() : 0);
    auto r = BatchGetDirect(fetch_items, [&](size_t m) {
      const size_t i = fetch_map[m];
      dedup_->Publish(bks[i], NodeDedup::Kind::kData, tokens[i],
                      static_cast<const char*>(items[i].out), items[i].n);
    }, st ? fst.data() : nullptr);
    for (size_t m = 0; m < fetch_map.size(); ++m) {
      const size_t i = fetch_map[m];
      res[i] = r[m];
      if (st) st[i] = fst[m];
      if (!r[m]) dedup_->Abort(bks[i], NodeDedup::Kind::kData, tokens[i]);
    }
  }
  // One deadline for the whole wait list and one direct batch for the keys
  // the owner did not deliver: a failed or slow owner costs this call at most
  // one wait, not one per key.
  if (!wait_list.empty()) {
    const uint64_t deadline = dedup_->WaitDeadline();
    std::vector<size_t> fb;
    std::vector<KvGetItem> rest;
    for (size_t i : wait_list) {
      if (dedup_->WaitCopy(bks[i], items[i].n,
                           static_cast<char*>(items[i].out), deadline)) {
        res[i] = true;
        if (st) st[i] = ItemStatus::kOk;
      } else {
        fb.push_back(i);
        rest.push_back(items[i]);
      }
    }
    if (!rest.empty()) {
      std::vector<ItemStatus> rst(st ? rest.size() : 0);
      auto rr = BatchGetDirect(rest, nullptr, st ? rst.data() : nullptr);
      for (size_t m = 0; m < fb.size(); ++m) {
        res[fb[m]] = rr[m];
        if (st) st[fb[m]] = rst[m];
      }
    }
  }
  uint64_t bytes = 0;
  for (size_t i = 0; i < N; ++i) if (res[i]) bytes += items[i].n;
  return RecordBatch(OpMetrics::kGet, t0, std::move(res), bytes);
}

namespace {
// Outcome of item m of a batched read; a transport that answered kOk without
// a length for it gave nothing usable.
template <typename Length>
ItemStatus ReadItemStatusAt(const std::vector<Status>& sts,
                            const std::vector<Length>& value_lens, size_t m,
                            size_t size, bool exact) {
  if (m >= sts.size()) return ItemStatus::kInvalid;
  if (m >= value_lens.size())
    return sts[m] == Status::kOk ? ItemStatus::kInvalid
                                 : ToItemStatus(sts[m]);
  return ReadItemStatus(sts[m], value_lens[m], size, exact);
}

// A read outcome the bounded miss retry is for: a block that read absent for
// an instant while the node moved it between tiers, or a read lost to
// connection turnover (see BatchGetDirect).
bool TransientReadMiss(Status st) {
  return st == Status::kNotFound || st == Status::kIOError;
}
}  // namespace

std::vector<bool> KVClient::BatchGetDirectRouted(
    const std::vector<KvGetItem>& items,
    const std::function<void(size_t)>& on_hit, ItemStatus* st) {
  const size_t N = items.size();
  std::vector<char> hit(N, 0);
  if (st) std::fill(st, st + N, ItemStatus::kNoRoute);
  if (!t_->pipelined()) {  // TCP: parallelize across items with our own threads
    RunParallel(N, BatchWorkers(N), [&](size_t i) {
      hit[i] = GetDirect(items[i].key, items[i].out, items[i].n,
                         st ? st + i : nullptr)
                   ? 1
                   : 0;
      if (hit[i] && on_hit) on_hit(i);
    });
    return std::vector<bool>(hit.begin(), hit.end());
  }
  // RDMA: group by (node, n) so each group shares the Range length, then pipeline.
  // Resolve one set of item indices -> hit[]. Reused for the miss-retry pass.
  // retry[i]: the last pass may have failed this key transiently (see below).
  std::vector<char> retry(N, 0);
  auto run_pass = [&](const std::vector<size_t>& todo) {
    std::map<std::pair<std::string, size_t>, std::vector<size_t>> by;
    std::vector<std::string> routes = RouteReads(
        todo.size(),
        [&](size_t j) -> const std::string& { return items[todo[j]].key; },
        RouteOp::kGet);
    size_t j = 0;
    for (size_t i : todo) {
      std::string node = std::move(routes[j++]);
      if (items[i].key.empty()) {  // as PUT
        if (st) st[i] = ItemStatus::kBadArgument;
        continue;
      }
      if (node.empty()) continue;
      by[{node, items[i].n}].push_back(i);
    }
    std::vector<std::pair<std::pair<std::string, size_t>, std::vector<size_t>>> groups = ShardReadGroups(std::vector<std::pair<std::pair<std::string, size_t>, std::vector<size_t>>>(by.begin(), by.end()));
    const auto node_of = [&](size_t g) -> const std::string& {
      return groups[g].first.first;
    };
    RunReadScheduled(groups.size(), [&](size_t g) {
      const std::string& node = groups[g].first.first;
      const size_t n = groups[g].first.second;
      const std::vector<size_t>& idx = groups[g].second;
      uint64_t now = NowMs();
      if (!health_.Healthy(node, now)) {
        if (st)
          for (size_t k : idx) st[k] = ItemStatus::kPeerCooling;
        return;
      }
      std::vector<BlockKey> keys;
      std::vector<RangeDst> dsts;
      keys.reserve(idx.size());
      dsts.reserve(idx.size());
      for (size_t k : idx) {
        keys.push_back(block_key_(items[k].key));
        dsts.push_back(RangeDst{items[k].out, n,
                                ClassifyDestination(items[k].out, n),
                                /*exact=*/true});
      }
      // The transport stages retry-owned bytes, then publishes them into each
      // host or CUDA destination before returning the authoritative size.
      std::vector<uint64_t> value_lens;
      std::vector<Status> sts = t_->RangeInto(node, keys, dsts, &value_lens);
      ReportPeer(node, sts);
      for (size_t m = 0; m < idx.size(); ++m) {
        if (st) st[idx[m]] = ReadItemStatusAt(sts, value_lens, m, n, true);
        retry[idx[m]] = TransientReadMiss(sts[m]);
        if (sts[m] == Status::kOk && m < value_lens.size() &&
            value_lens[m] == n) {
          hit[idx[m]] = 1;
          if (on_hit) on_hit(idx[m]);
        }
      }
    }, node_of);
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
  // DFKV_GET_MISS_RETRIES=0. Connection turnover is the other transient: a
  // subset of a batch fails with kIOError on a connection being replaced
  // (Status has no separate timeout, so a timed-out shard reads the same).
  // Only those two outcomes are retried, and an I/O error only while its node
  // is still healthy: the pass checks that before it sends, and a node whose
  // shard failed outright is in cooldown by then. A refused request
  // (kInvalid, resource exhaustion) or a length mismatch is not retried.
  for (int r = 0; r < get_miss_retries_; ++r) {
    std::vector<size_t> missed;
    for (size_t i = 0; i < N; ++i) if (!hit[i] && retry[i]) missed.push_back(i);
    if (missed.empty() || missed.size() > N / 2) break;  // none, or a cold batch
    run_pass(missed);
  }
  return std::vector<bool>(hit.begin(), hit.end());
}

std::vector<bool> KVClient::BatchGetAuto(const std::vector<KvGetItem>& items,
                                         std::vector<size_t>* out_lens,
                                         std::vector<ItemStatus>* status) {
  const auto t0 = std::chrono::steady_clock::now();
  ItemStatus* st = nullptr;
  if (status) {
    status->assign(items.size(), ItemStatus::kNoRoute);
    st = status->data();
  }
  if (!dedup_) {
    std::vector<size_t> lengths;
    auto result = BatchGetAutoDirect(items, &lengths, nullptr, st);
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
  AbortClaimsOnThrow abort_claims([&] {
    for (size_t i = 0; i < N; ++i)
      if (tokens[i]) dedup_->Abort(bks[i], NodeDedup::Kind::kData, tokens[i]);
  });
  fetch_items.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    if (ClassifyDestination(items[i].out, items[i].n) ==
        DestinationMemoryKind::kDevice) {
      fetch_map.push_back(i);
      fetch_items.push_back(items[i]);
      // Unclaimed device fetch: token remains zero.
      continue;
    }
    bks[i] = block_key_(items[i].key);
    size_t got = 0;
    switch (dedup_->ClaimAuto(bks[i], items[i].n,
                              static_cast<char*>(items[i].out), &got,
                              &tokens[i])) {
      case NodeDedup::Role::kHit:
        res[i] = true;
        lens[i] = got;
        if (st) st[i] = ItemStatus::kOk;
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
    // Published key by key as they arrive, as in BatchGet.
    std::vector<size_t> flens;
    std::vector<ItemStatus> fst(st ? fetch_items.size() : 0);
    auto r = BatchGetAutoDirect(
        fetch_items, &flens, [&](size_t m, size_t length) {
          const size_t i = fetch_map[m];
          dedup_->Publish(bks[i], NodeDedup::Kind::kData, tokens[i],
                          static_cast<const char*>(items[i].out), length);
        }, st ? fst.data() : nullptr);
    for (size_t m = 0; m < fetch_map.size(); ++m) {
      const size_t i = fetch_map[m];
      res[i] = r[m];
      if (st) st[i] = fst[m];
      if (r[m])
        lens[i] = flens[m];
      else
        dedup_->Abort(bks[i], NodeDedup::Kind::kData, tokens[i]);
    }
  }
  // One deadline and one direct fallback batch, as in BatchGet.
  if (!wait_list.empty()) {
    const uint64_t deadline = dedup_->WaitDeadline();
    std::vector<size_t> fb;
    std::vector<KvGetItem> rest;
    for (size_t i : wait_list) {
      size_t got = 0;
      if (dedup_->WaitCopyAuto(bks[i], items[i].n,
                               static_cast<char*>(items[i].out), &got,
                               deadline)) {
        res[i] = true;
        lens[i] = got;
        if (st) st[i] = ItemStatus::kOk;
      } else {
        fb.push_back(i);
        rest.push_back(items[i]);
      }
    }
    if (!rest.empty()) {
      std::vector<size_t> rl;
      std::vector<ItemStatus> rst(st ? rest.size() : 0);
      auto rr = BatchGetAutoDirect(rest, &rl, nullptr,
                                   st ? rst.data() : nullptr);
      for (size_t m = 0; m < fb.size(); ++m) {
        res[fb[m]] = rr[m];
        if (rr[m]) lens[fb[m]] = rl[m];
        if (st) st[fb[m]] = rst[m];
      }
    }
  }
  uint64_t bytes = 0;
  for (size_t length : lens) bytes += length;
  if (out_lens) *out_lens = std::move(lens);
  return RecordBatch(OpMetrics::kGet, t0, std::move(res), bytes);
}

std::vector<bool> KVClient::BatchGetAutoDirectRouted(
    const std::vector<KvGetItem>& items, std::vector<size_t>* out_lens,
    const std::function<void(size_t, size_t)>& on_hit, ItemStatus* st) {
  const size_t N = items.size();
  std::vector<char> hit(N, 0);
  std::vector<size_t> lens(N, 0);  // distinct indices => thread-safe writes
  if (st) std::fill(st, st + N, ItemStatus::kNoRoute);
  if (!t_->pipelined()) {  // TCP: parallelize per item with our own threads.
    RunParallel(N, BatchWorkers(N), [&](size_t i) {
      size_t got = 0;
      if (GetAutoDirect(items[i].key, items[i].out, items[i].n, &got,
                        st ? st + i : nullptr)) {
        hit[i] = 1;
        lens[i] = got;
        if (on_hit) on_hit(i, got);
      }
    });
    if (out_lens) *out_lens = std::move(lens);
    return std::vector<bool>(hit.begin(), hit.end());
  }
  // RDMA: group by (node, cap) so each group shares the Range length, then
  // pipeline. Same staged-publication datapath as BatchGet; the only
  // difference is accepting payload_len <= cap and reporting the true length.
  // The same bounded retry of transient misses as BatchGet.
  std::vector<char> retry(N, 0);
  auto run_pass = [&](const std::vector<size_t>& todo) {
  std::map<std::pair<std::string, size_t>, std::vector<size_t>> by;
  std::vector<std::string> routes = RouteReads(
      todo.size(),
      [&](size_t j) -> const std::string& { return items[todo[j]].key; },
      RouteOp::kGet);
  size_t j = 0;
  for (size_t i : todo) {
    std::string node = std::move(routes[j++]);
    if (items[i].key.empty()) {  // as PUT
      if (st) st[i] = ItemStatus::kBadArgument;
      continue;
    }
    if (node.empty()) continue;
    by[{node, items[i].n}].push_back(i);
  }
  std::vector<std::pair<std::pair<std::string, size_t>, std::vector<size_t>>> groups = ShardReadGroups(std::vector<std::pair<std::pair<std::string, size_t>, std::vector<size_t>>>(by.begin(), by.end()));
  const auto node_of = [&](size_t g) -> const std::string& {
    return groups[g].first.first;
  };
  RunReadScheduled(groups.size(), [&](size_t g) {
    const std::string& node = groups[g].first.first;
    const size_t cap = groups[g].first.second;
    const std::vector<size_t>& idx = groups[g].second;
    uint64_t now = NowMs();
    if (!health_.Healthy(node, now)) {
      if (st)
        for (size_t k : idx) st[k] = ItemStatus::kPeerCooling;
      return;
    }
    std::vector<BlockKey> keys;
    std::vector<RangeDst> dsts;
    keys.reserve(idx.size());
    dsts.reserve(idx.size());
    for (size_t k : idx) {
      keys.push_back(block_key_(items[k].key));
      dsts.push_back(RangeDst{items[k].out, cap,
                              ClassifyDestination(items[k].out, cap)});
    }
    std::vector<uint64_t> value_lens;
    std::vector<Status> sts = t_->RangeInto(node, keys, dsts, &value_lens);
    ReportPeer(node, sts);
    for (size_t m = 0; m < idx.size(); ++m) {
      if (st) st[idx[m]] = ReadItemStatusAt(sts, value_lens, m, cap, false);
      retry[idx[m]] = TransientReadMiss(sts[m]);
      if (sts[m] != Status::kOk || m >= value_lens.size() ||
          value_lens[m] > cap)
        continue;
      lens[idx[m]] = static_cast<size_t>(value_lens[m]);
      hit[idx[m]] = 1;
      if (on_hit) on_hit(idx[m], lens[idx[m]]);
    }
  }, node_of);
  };
  std::vector<size_t> all(N);
  for (size_t i = 0; i < N; ++i) all[i] = i;
  run_pass(all);
  for (int r = 0; r < get_miss_retries_; ++r) {
    std::vector<size_t> missed;
    for (size_t i = 0; i < N; ++i) if (!hit[i] && retry[i]) missed.push_back(i);
    if (missed.empty() || missed.size() > N / 2) break;  // none, or a cold batch
    run_pass(missed);
  }
  if (out_lens) *out_lens = std::move(lens);
  return std::vector<bool>(hit.begin(), hit.end());
}

std::vector<bool> KVClient::BatchExist(const std::vector<std::string>& keys) {
  const auto t0 = std::chrono::steady_clock::now();
  if (!dedup_) {
    auto result = BatchExistDirect(keys, nullptr);
    return RecordBatch(OpMetrics::kExist, t0, std::move(result), 0);
  }
  // Only an answer the server gave is published. A transient route,
  // cooldown, invalid, or I/O failure aborts the slot and retries directly on
  // the next public call rather than masquerading as cached absence.
  const size_t N = keys.size();
  std::vector<bool> res(N, false);
  std::vector<BlockKey> bks(N);
  std::vector<std::string> fetch_keys;
  std::vector<size_t> fetch_map, wait_list;
  std::vector<uint64_t> tokens(N, 0);
  AbortClaimsOnThrow abort_claims([&] {
    for (size_t i = 0; i < N; ++i)
      if (tokens[i]) dedup_->Abort(bks[i], NodeDedup::Kind::kExist, tokens[i]);
  });
  std::vector<char> roles(N, 0), values(N, 0);
  fetch_keys.reserve(N);
  // Shared-memory rendezvous probes are independent per key. Long-context
  // scheduler lookups contain thousands of keys; serial ClaimExist calls can
  // cost more than the sharded RDMA ExistMany itself.
  RunParallel(N, BatchWorkers(N), [&](size_t i) {
    bks[i] = block_key_(keys[i]);
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
      // "Does not exist" arrives as kNotFound or as kOk with exists=false,
      // depending on the transport and the probe path; both are answers.
      if (statuses[m] == Status::kOk || statuses[m] == Status::kNotFound) {
        const char answer = r[m] ? 1 : 0;
        dedup_->Publish(bks[i], NodeDedup::Kind::kExist, tokens[i],
                        &answer, 1);
      } else {
        dedup_->Abort(bks[i], NodeDedup::Kind::kExist, tokens[i]);
      }
    }
  }
  // One deadline and one direct fallback batch, as in BatchGet.
  if (!wait_list.empty()) {
    const uint64_t deadline = dedup_->WaitDeadline();
    std::vector<size_t> fb;
    std::vector<std::string> rest;
    for (size_t i : wait_list) {
      bool val = false;
      if (dedup_->WaitExist(bks[i], &val, deadline)) {
        res[i] = val;
      } else {
        fb.push_back(i);
        rest.push_back(keys[i]);
      }
    }
    if (!rest.empty()) {
      auto rr = BatchExistDirect(rest, nullptr);
      for (size_t m = 0; m < fb.size(); ++m) res[fb[m]] = rr[m];
    }
  }
  return RecordBatch(OpMetrics::kExist, t0, std::move(res), 0);
}

// ------------------------------------------------------------- retention

namespace {
size_t RetainDebounceCapPerShard(size_t shards) {
  static const size_t total = [] {
    const char* v = std::getenv("DFKV_RETAIN_CACHE_ENTRIES");
    size_t resolved = static_cast<size_t>(1) << 20;
    if (v && *v) {
      char* end = nullptr;
      const unsigned long long n = std::strtoull(v, &end, 10);
      if (end != v && *end == '\0') resolved = static_cast<size_t>(n);
    }
    envparse::ReportUnlessPlain("DFKV_RETAIN_CACHE_ENTRIES", 0,
                                std::numeric_limits<uint64_t>::max(),
                                std::to_string(resolved));
    return resolved;
  }();
  return total / shards;
}

bool RetainCodeProtected(uint8_t code) {
  return code == KVClient::kRetProtectedNew ||
         code == KVClient::kRetRefreshed || code == KVClient::kRetUpgraded ||
         code == KVClient::kRetUnchanged;
}
}  // namespace

void KVClient::RetainNode(const std::string& node, uint64_t tenant,
                          const std::vector<size_t>& idx,
                          const std::vector<BlockKey>& bkeys,
                          const RetainRequest& req, uint8_t* out) {
  auto fallback = [&](size_t from) {
    // Node without kRetain: REFRESH degrades to an existence probe so callers
    // keep exact hit/miss; SET cannot be honoured.
    if (req.mode != kRetainModeRefresh) {
      for (size_t m = from; m < idx.size(); ++m) out[idx[m]] = kRetUnsupported;
      return;
    }
    std::vector<BlockKey> sub;
    sub.reserve(idx.size() - from);
    for (size_t m = from; m < idx.size(); ++m) sub.push_back(bkeys[idx[m]]);
    std::vector<char> ex;
    const std::vector<Status> sts = t_->ExistMany(node, sub, &ex);
    for (size_t m = 0; m < sub.size(); ++m) {
      const Status st = m < sts.size() ? sts[m] : Status::kInvalid;
      uint8_t code = kRetError;
      if (st == Status::kOk)
        code = (m < ex.size() && ex[m]) ? kRetHitUnprotected : kRetMiss;
      else if (st == Status::kNotFound)
        code = kRetMiss;
      out[idx[from + m]] = code;
    }
    ReportPeer(node, sts);
  };

  if (!health_.Healthy(node, NowMs())) {
    for (size_t k : idx) out[k] = kRetError;
    return;
  }
  // Only the verdict is read under the lock: the fallback is a network
  // round trip, and every other node group of the batch, RetainPrecheck and
  // RetainLimits wait for this lock.
  bool unsupported = false;
  {
    std::lock_guard<std::mutex> lk(retain_mu_);
    unsupported = KnownUnsupported(retain_nodes_[node], NowMs());
  }
  if (unsupported) {
    fallback(0);
    return;
  }
  const size_t cap = RetainDebounceCapPerShard(kRetainDebounceShards);
  for (size_t begin = 0; begin < idx.size(); begin += kRetainMaxKeys) {
    const size_t n = std::min(kRetainMaxKeys, idx.size() - begin);
    std::vector<BlockKey> chunk(n);
    for (size_t m = 0; m < n; ++m) chunk[m] = bkeys[idx[begin + m]];
    std::string payload(RetainPayloadBytes(n), '\0');
    EncodeRetainPayload(&payload[0], req.ttl_s, req.mode, chunk.data(), n);
    uint64_t epochs[kRetainDebounceShards];
    for (size_t sh = 0; sh < kRetainDebounceShards; ++sh) {
      std::lock_guard<std::mutex> lk(retain_debounce_[sh].mu);
      epochs[sh] = retain_debounce_[sh].forget_epoch;
    }
    std::string reply;
    const Status st = t_->Retain(node, tenant, n, payload, &reply);
    if (st == Status::kInvalid) {
      {
        std::lock_guard<std::mutex> lk(retain_mu_);
        RetainNodeState& ns = retain_nodes_[node];
        ns.supported = 0;
        ns.unsupported_at_ms = NowMs();
        ns.max_ttl_s = 0;
      }
      fallback(begin);
      return;
    }
    if (st != Status::kOk || reply.size() != n + kRetainTrailerBytes) {
      ReportPeer(node, st);
      for (size_t m = 0; m < n; ++m) out[idx[begin + m]] = kRetError;
      continue;
    }
    ReportPeer(node, st);
    const RetainTrailer trailer = DecodeRetainTrailer(reply.data() + n);
    {
      std::lock_guard<std::mutex> lk(retain_mu_);
      RetainNodeState& ns = retain_nodes_[node];
      ns.supported = 1;
      ns.max_ttl_s = trailer.max_ttl_s;
      ns.headroom_pct = trailer.headroom_pct;
      ns.heard = true;
    }
    const uint64_t now = NowMs();
    // How long a protection just confirmed can last at most: a SET was
    // granted its TTL clamped to the node's cap; a REFRESH keeps whatever
    // the SET was granted, which the cap bounds.
    const uint64_t cap_s = trailer.max_ttl_s;
    const uint64_t valid_ms =
        1000 * (req.mode == kRetainModeRefresh
                    ? cap_s
                    : std::min<uint64_t>(req.ttl_s, cap_s));
    const uint64_t node_id = std::hash<std::string>{}(node);
    for (size_t m = 0; m < n; ++m) {
      const uint8_t code = static_cast<uint8_t>(reply[m]);
      out[idx[begin + m]] = code <= kRetDenied ? code : kRetError;
      const BlockKey& bk = chunk[m];
      const size_t sh = bk.digest_lo % kRetainDebounceShards;
      RetainDebounceShard& shard = retain_debounce_[sh];
      std::lock_guard<std::mutex> lk(shard.mu);
      const auto key = std::make_pair(bk.digest_hi, bk.digest_lo);
      if (shard.forget_epoch != epochs[sh]) {
        // A PUT or remove of a key of this shard ran while the request was
        // out: the reply may be about the value it replaced.
        shard.entries.erase(key);
      } else if (RetainCodeProtected(code)) {
        if (shard.entries.size() >= cap && shard.entries.count(key) == 0)
          shard.entries.clear();  // bounded: rare full reset
        if (cap != 0) {
          RetainDebounceEntry& e = shard.entries[key];
          e.protected_at_ms = now;
          e.valid_ms = valid_ms;
          e.node_id = node_id;
        }
      } else {
        shard.entries.erase(key);
      }
    }
  }
}

void KVClient::BatchRetain(const std::vector<std::string>& keys,
                           const RetainRequest& req, uint8_t* out) {
  const auto t0 = std::chrono::steady_clock::now();
  const size_t n = keys.size();
  // Counted once like every public call. A REFRESH without a TTL is how the
  // connectors probe existence when the ring has retention, so it counts as
  // exist (op="exist" would otherwise lose those probes); SET and a REFRESH
  // with a TTL count as retain. Hits are the keys that are present.
  const OpMetrics::Op op = req.mode == kRetainModeRefresh && req.ttl_s == 0
                               ? OpMetrics::kExist
                               : OpMetrics::kRetain;
  struct Record {
    OpMetrics* m;
    OpMetrics::Op op;
    std::chrono::steady_clock::time_point t0;
    const uint8_t* out;
    size_t n;
    ~Record() {
      uint64_t hits = 0;
      for (size_t i = 0; out != nullptr && i < n; ++i)
        if (out[i] != kRetMiss && out[i] != kRetUnsupported &&
            out[i] != kRetError)
          ++hits;
      m->Record(op, n, hits, 0,
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count());
    }
  } record{&op_stats_, op, t0, out, n};
  if (out == nullptr || n == 0) return;
  BatchRetainRouted(keys, req, out);
  // A key whose owner with the absent members is absent is retained there
  // too, so that its copy is kept for the member's return; no read finds
  // that copy meanwhile, so its answer does not stand for the key.
  for (const AbsentPart& part : AbsentOwned(keys)) {
    std::vector<std::string> sub;
    sub.reserve(part.index.size());
    for (size_t i : part.index) sub.push_back(keys[i]);
    std::vector<uint8_t> sub_out(sub.size(), kRetError);
    {
      RouteTo to(part.addr);
      BatchRetainRouted(sub, req, sub_out.data());
    }
    for (uint8_t code : sub_out)
      (code != kRetMiss && code != kRetUnsupported && code != kRetError
           ? drain_retains_ok_
           : drain_retains_failed_)
          .fetch_add(1, std::memory_order_relaxed);
  }
  // A key whose full-ring owner drains is retained there too: that copy
  // still serves reads. Its answer counts where the write owner has nothing.
  const std::vector<size_t> drained =
      DrainerOwned(n, nullptr, [&](size_t i) -> const std::string& {
        return keys[i];
      });
  if (drained.empty()) return;
  std::vector<std::string> sub;
  sub.reserve(drained.size());
  for (size_t i : drained) sub.push_back(keys[i]);
  std::vector<uint8_t> sub_out(sub.size(), kRetError);
  {
    DrainPass pass;
    BatchRetainRouted(sub, req, sub_out.data());
  }
  auto present = [](uint8_t code) {
    return code != kRetMiss && code != kRetUnsupported && code != kRetError;
  };
  for (size_t m = 0; m < drained.size(); ++m) {
    (present(sub_out[m]) ? drain_retains_ok_ : drain_retains_failed_)
        .fetch_add(1, std::memory_order_relaxed);
    uint8_t& code = out[drained[m]];
    if (!present(code) && present(sub_out[m])) code = sub_out[m];
  }
}

void KVClient::BatchRetainRouted(const std::vector<std::string>& keys,
                                 const RetainRequest& req, uint8_t* out) {
  const size_t n = keys.size();
  std::fill(out, out + n, static_cast<uint8_t>(kRetError));
  const bool debounce = req.mode == kRetainModeRefresh && req.ttl_s != 0 &&
                        !req.no_debounce;
  const uint64_t interval = std::max<uint64_t>(
      req.min_refresh_ms, static_cast<uint64_t>(req.ttl_s) * 100);
  const uint64_t now = NowMs();
  const RouteOp retain_op = req.mode == kRetainModeSet
                                ? RouteOp::kRetainSet
                                : RouteOp::kRetainRefresh;
  std::vector<BlockKey> bks(n);
  std::map<std::pair<std::string, uint64_t>, std::vector<size_t>> by_node;
  for (size_t i = 0; i < n; ++i) {
    bks[i] = block_key_(keys[i]);
    std::string node = RouteFor(keys[i], retain_op);
    if (node.empty()) continue;  // stays kRetError
    if (debounce) {
      RetainDebounceShard& shard =
          retain_debounce_[bks[i].digest_lo % kRetainDebounceShards];
      std::lock_guard<std::mutex> lk(shard.mu);
      auto it = shard.entries.find(
          std::make_pair(bks[i].digest_hi, bks[i].digest_lo));
      // Protected by its present owner within the last ttl/10, and not for
      // longer than that protection can last: still protected, hence
      // present. (`now` was read first: an entry written since is newer.)
      if (it != shard.entries.end() &&
          it->second.node_id == std::hash<std::string>{}(node) &&
          (it->second.protected_at_ms >= now ||
           now - it->second.protected_at_ms <
               std::min(interval, it->second.valid_ms))) {
        out[i] = kRetSkippedDebounce;
        continue;
      }
    }
    by_node[{node, bks[i].tenant_hash}].push_back(i);
  }
  std::vector<std::pair<std::pair<std::string, uint64_t>, std::vector<size_t>>>
      groups(by_node.begin(), by_node.end());
  RunParallel(groups.size(), BatchWorkers(groups.size()), [&](size_t g) {
    RetainNode(groups[g].first.first, groups[g].first.second, groups[g].second,
               bks, req, out);
  });
}

bool KVClient::KnownUnsupported(const RetainNodeState& ns,
                                uint64_t now) const {
  return ns.supported == 0 && ns.unsupported_at_ms <= now &&
         now - ns.unsupported_at_ms < retain_reprobe_ms_;
}

void KVClient::ForgetRetained(const BlockKey& key) {
  RetainDebounceShard& shard =
      retain_debounce_[key.digest_lo % kRetainDebounceShards];
  std::lock_guard<std::mutex> lk(shard.mu);
  shard.entries.erase(std::make_pair(key.digest_hi, key.digest_lo));
  ++shard.forget_epoch;
}

void KVClient::RetainLimits(int* supported, uint32_t* max_ttl_s) {
  std::vector<std::string> nodes;
  {
    std::lock_guard<std::mutex> lk(ring_mu_);
    for (const auto& kv : addr_) nodes.push_back(kv.second);
  }
  const uint64_t tenant = block_key_("").tenant_hash;
  for (const std::string& node : nodes) {
    {
      std::lock_guard<std::mutex> lk(retain_mu_);
      const RetainNodeState& ns = retain_nodes_[node];
      if (ns.supported == 1 || KnownUnsupported(ns, NowMs())) continue;
    }
    // One-key REFRESH probe of an impossible key: answers MISS plus limits.
    const BlockKey probe{0, 0, tenant};
    std::string payload(RetainPayloadBytes(1), '\0');
    EncodeRetainPayload(&payload[0], 0, kRetainModeRefresh, &probe, 1);
    std::string reply;
    const Status st = t_->Retain(node, tenant, 1, payload, &reply);
    t_->TakeUnreachableRefusal();  // not reported to PeerHealth; see ReportPeer
    std::lock_guard<std::mutex> lk(retain_mu_);
    RetainNodeState& ns = retain_nodes_[node];
    if (st == Status::kInvalid) {
      ns.supported = 0;
      ns.unsupported_at_ms = NowMs();
      ns.max_ttl_s = 0;
    } else if (st == Status::kOk && reply.size() == 1 + kRetainTrailerBytes) {
      const RetainTrailer t = DecodeRetainTrailer(reply.data() + 1);
      ns.supported = 1;
      ns.max_ttl_s = t.max_ttl_s;
      ns.headroom_pct = t.headroom_pct;
      ns.heard = true;
    }
  }
  int sup = nodes.empty() ? -1 : 1;
  uint32_t ttl = UINT32_MAX;
  {
    std::lock_guard<std::mutex> lk(retain_mu_);
    for (const std::string& node : nodes) {
      const RetainNodeState& ns = retain_nodes_[node];
      if (ns.supported == 0) sup = 0;
      else if (ns.supported == -1 && sup == 1) sup = -1;
      if (ns.supported == 1) ttl = std::min(ttl, ns.max_ttl_s);
    }
  }
  if (supported) *supported = sup;
  if (max_ttl_s) *max_ttl_s = sup == 1 ? ttl : 0;
}

bool KVClient::RetainPrecheck(const std::vector<std::string>& keys,
                              uint8_t min_headroom_pct) {
  std::set<std::string> owners;
  for (const std::string& key : keys) {
    std::string node =
        RouteForUnchecked(key, RouteOp::kExist, nullptr, nullptr);
    if (node.empty()) return false;
    owners.insert(std::move(node));
  }
  std::lock_guard<std::mutex> lk(retain_mu_);
  for (const std::string& node : owners) {
    auto it = retain_nodes_.find(node);
    if (it == retain_nodes_.end()) continue;  // never heard: optimistic
    // Any age: the precheck asks nobody, so an expired verdict stands until
    // RetainNode or RetainLimits, which do ask, hear otherwise.
    if (it->second.supported == 0) return false;
    if (it->second.heard && it->second.headroom_pct < min_headroom_pct)
      return false;
  }
  return true;
}

namespace {

// Shared per-node (ip:port) stats for GET/PUT/EXIST. Gated by
// DFKV_CLIENT_PERNODE_STATS=1 (off by default). The disabled path is this one
// cached branch; no clocks, maps, strings, or sink state are touched.
bool PernodeOn() {
  static const bool on = [] {
    const char* e = std::getenv("DFKV_CLIENT_PERNODE_STATS");
    const bool enabled = e && *e && *e != '0';
    envparse::ReportSwitch("DFKV_CLIENT_PERNODE_STATS", enabled,
                           "only a value starting with 0 is off");
    return enabled;
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

std::string PernodeSuffix() {
  const char* env_suffix = std::getenv("DFKV_CLIENT_LOG_SUFFIX");
  const std::string pid_tag = "p" + std::to_string(::getpid());
  std::string value = env_suffix && *env_suffix ? env_suffix : pid_tag;
  for (char& c : value)
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_')
      c = '_';
  // A launcher-supplied rank identity is attributable but not necessarily
  // process-unique (for example TP rank repeats across DP/PP groups).
  if (value != pid_tag &&
      (value.size() <= pid_tag.size() ||
       value.compare(value.size() - pid_tag.size(), pid_tag.size(), pid_tag) !=
           0))
    value += "_" + pid_tag;
  return value;
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


std::vector<bool> KVClient::BatchExistDirectRouted(
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
  std::vector<std::string> routes = RouteReads(
      N, [&](size_t i) -> const std::string& { return keys[i]; },
      RouteOp::kExist);
  for (size_t i = 0; i < N; ++i) {
    std::string node = std::move(routes[i]);
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
  const auto node_of = [&](size_t g) -> const std::string& {
    return groups[g].first;
  };
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
    for (size_t k : idx) bkeys.push_back(block_key_(keys[k]));
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
    ReportPeer(node, sts);
    for (size_t m = 0; m < idx.size(); ++m) {
      if (m < sts.size()) raw[idx[m]] = sts[m];
      e[idx[m]] =
          (m < sts.size() && sts[m] == Status::kOk &&
           m < ex.size() && ex[m])
              ? 1
              : 0;
    }
  }, node_of);
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
      } else if (a.dev.empty()) {
        // A shard timed at 0 us (sub-microsecond) still names its rail.
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
  if (fanout_on_) {
    ok = RemoveFanout(keys);  // owner, drainer and fan-out set, both forms
  } else if (!t_->pipelined()) {
    RunParallel(N, BatchWorkers(N), [&](size_t i) {
      ok[i] = RemoveDirect(keys[i]) ? 1 : 0;  // asks a draining owner too
    });
  } else {
    ok = BatchRemoveRouted(keys);
    RemoveAtAbsent(keys);
    // A key whose full-ring owner drains may still be there: removed only
    // once both have confirmed, or the drainer would serve it on a read.
    const std::vector<size_t> drained =
        DrainerOwned(N, nullptr, [&](size_t i) -> const std::string& {
          return keys[i];
        });
    if (!drained.empty()) {
      std::vector<std::string> sub;
      sub.reserve(drained.size());
      for (size_t i : drained) sub.push_back(keys[i]);
      std::vector<char> sub_ok;
      {
        DrainPass pass;
        sub_ok = BatchRemoveRouted(sub);
      }
      for (size_t m = 0; m < drained.size(); ++m) {
        (sub_ok[m] ? drain_removes_ok_ : drain_removes_failed_)
            .fetch_add(1, std::memory_order_relaxed);
        if (!sub_ok[m]) ok[drained[m]] = 0;
      }
    }
  }
  return RecordBatch(OpMetrics::kRemove, t0,
                     std::vector<bool>(ok.begin(), ok.end()), 0);
}

std::vector<char> KVClient::BatchRemoveRouted(
    const std::vector<std::string>& keys) {
  const size_t N = keys.size();
  std::vector<char> ok(N, 0);
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
    for (size_t k : idx) bkeys.push_back(block_key_(keys[k]));
    std::vector<Status> sts = t_->RemoveMany(node, bkeys);
    ReportPeer(node, sts);
    for (size_t m = 0; m < idx.size(); ++m) {
      const bool removed =
          m < sts.size() &&
          (sts[m] == Status::kOk || sts[m] == Status::kNotFound);
      ok[idx[m]] = removed ? 1 : 0;
      if (removed) InvalidateRendezvous(bkeys[m]);
    }
  });
  return ok;
}

std::vector<bool> KVClient::BatchPutSg(const std::vector<KvPutItemSg>& items,
                                       std::vector<ItemStatus>* status) {
  auto t0 = std::chrono::steady_clock::now();
  const size_t N = items.size();
  std::vector<char> ok(N, 0);  // char (not vector<bool>) for thread-safe writes
  ItemStatus* st = nullptr;
  if (status) {
    status->assign(N, ItemStatus::kNoRoute);
    st = status->data();
  }

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
      if (st) st[i] = ItemStatus::kBadArgument;
      if (pernode_on)
        RecordPernodeNonIssued(
            unrouted.get(), PernodeNonIssuedReason::kInvalidInput, 1,
            items[i].key);
      continue;
    }
    std::string node = RouteFor(items[i].key, RouteOp::kPut);
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
      if (st)
        for (size_t k : idx) st[k] = ItemStatus::kPeerCooling;
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
      s.key = block_key_(items[k].key);
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
      if (st)
        st[idx[m]] = m < sts.size() ? ToItemStatus(sts[m])
                                    : ItemStatus::kInvalid;
      if (ok[idx[m]]) InvalidateRendezvous(srcs[m].key);
    }
    ReportPeer(node, sts);
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
      } else if (a.dev.empty()) {
        // A shard timed at 0 us (sub-microsecond) still names its rail.
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

// Set up once, on the first all-device batch. Not std::call_once: an
// exception unwinding through it (pthread_once) aborts the process. A set-up
// that throws leaves this client without GPU dedup, as one that finds no
// device does.
GpuNodeDedup* KVClient::GpuDedup(const void* device_dst_hint) {
  if (gpu_dedup_tried_.load(std::memory_order_acquire))
    return gpu_dedup_raw_.load(std::memory_order_relaxed);
  // Callers of the first batches wait here for the one set-up, as they did
  // in std::call_once; tried_ is published last.
  std::lock_guard<std::mutex> lk(gpu_dedup_mu_);
  if (gpu_dedup_tried_.load(std::memory_order_relaxed))
    return gpu_dedup_.get();
  const char* failure = nullptr;
  std::string what;
  try {
    gpu_dedup_ = GpuNodeDedup::FromEnv(namespace_hash_, device_dst_hint);
  } catch (const std::exception& e) {
    gpu_dedup_.reset();
    failure = "set-up failed";
    try {
      what = e.what();
    } catch (...) {
    }
  } catch (...) {
    gpu_dedup_.reset();
    failure = "set-up failed";
  }
  gpu_dedup_raw_.store(gpu_dedup_.get(), std::memory_order_release);
  gpu_dedup_tried_.store(true, std::memory_order_release);
  if (failure != nullptr) {
    try {
      DFKV_LOG_WARN(std::string("gpu-dedup: ") + failure +
                    (what.empty() ? std::string() : " (" + what + ")") +
                    "; this client runs without GPU dedup");
    } catch (...) {
    }
  }
  return gpu_dedup_.get();
}

std::vector<bool> KVClient::BatchGetAutoSg(const std::vector<KvGetItemSg>& items,
                                           std::vector<size_t>* out_lens,
                                           std::vector<ItemStatus>* status) {
  const auto t0 = std::chrono::steady_clock::now();
  ItemStatus* st = nullptr;
  if (status) {
    status->assign(items.size(), ItemStatus::kNoRoute);
    st = status->data();
  }
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
    auto result = BatchGetAutoSgDirect(items, &lengths, st);
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
  AbortClaimsOnThrow abort_claims([&] {
    for (size_t i = 0; i < N; ++i)
      if (tokens[i]) gd->Abort(bks[i], tokens[i]);
  });
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
  GpuNodeDedup::HitBatch hit_batch;
  // Room for every hit, so that recording one cannot throw once its copy is
  // enqueued.
  hit_batch.pending.reserve(N);
  // A throw before FinishHits leaves hit copies queued into the caller's
  // destinations: wait for them before the caller gets the buffers back.
  AbortClaimsOnThrow settle_hits([&] { gd->SettleHits(&hit_batch); });
  // Hits whose copy did not validate: no claim is held, fetched directly.
  std::vector<size_t> unhit;
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
    bks[i] = block_key_(items[i].key);
    segs_of(items[i]);
    // A hit's copy is only enqueued here; FinishHits below synchronizes once
    // for all of them instead of once per key.
    switch (gd->ClaimSgDeferred(bks[i], segs.data(), segs.size(),
                                total_caps[i], &tokens[i], &hit_batch, i)) {
      case GpuNodeDedup::Role::kHit:
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
  {
    std::vector<std::pair<size_t, size_t>> hits;
    gd->FinishHits(&hit_batch, &hits, &unhit);
    for (const auto& [i, got] : hits) {
      res[i] = true;
      lens[i] = got;
      if (st) st[i] = ItemStatus::kOk;
    }
  }
  if (!fetch_items.empty()) {
    std::vector<size_t> flens;
    std::vector<ItemStatus> fst(st ? fetch_items.size() : 0);
    auto r = BatchGetAutoSgDirect(fetch_items, &flens,
                                  st ? fst.data() : nullptr);
    // The fetch may have landed in these device destinations over GPUDirect.
    // Publishing gathers them with a CUDA copy, which must not run ahead of
    // those writes: order them first, once for the batch. If that cannot be
    // done nothing is published, and the waiters fetch for themselves. The
    // caller's own result does not depend on it: its connector fences before
    // it uses the destinations.
    bool anything_to_publish = false;
    for (size_t m = 0; m < fetch_map.size(); ++m)
      if (r[m] && tokens[fetch_map[m]] != 0) anything_to_publish = true;
    const bool settled = !anything_to_publish || gd->FenceLoads();
    // Published together: one stream synchronization per chunk, not per key.
    std::vector<std::vector<GpuNodeDedup::Seg>> psegs;
    std::vector<GpuNodeDedup::PublishItem> publish;
    psegs.reserve(fetch_map.size());
    publish.reserve(fetch_map.size());
    for (size_t m = 0; m < fetch_map.size(); ++m) {
      const size_t i = fetch_map[m];
      res[i] = r[m];
      if (r[m]) lens[i] = flens[m];
      if (st) st[i] = fst[m];
      if (r[m] && settled && tokens[i] != 0) {
        segs_of(items[i]);
        psegs.push_back(segs);
        publish.push_back(GpuNodeDedup::PublishItem{
            bks[i], tokens[i], psegs.back().data(), psegs.back().size(),
            flens[m]});
      } else {
        gd->Abort(bks[i], tokens[i]);
      }
    }
    gd->PublishManySg(publish.data(), publish.size());
  }
  // One amortized-sync wait for the WHOLE wait list (WaitManySg): copies are
  // enqueued as keys turn READY and the stream synchronizes per round, not
  // per key — per-key syncs cost ~ms under load and burned any wait budget
  // long before the list was through (the dedup win leaked back out through
  // fallback re-fetches; measured live in the PD A/B). Failures fall back
  // through ONE direct batch.
  std::vector<size_t> fb(unhit);  // hits that did not validate, then waits
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
      if (st) st[i] = ItemStatus::kOk;
    } else {
      fb.push_back(i);
    }
  }
  if (!fb.empty()) {
    std::vector<KvGetItemSg> rest;
    rest.reserve(fb.size());
    for (size_t i : fb) rest.push_back(items[i]);
    std::vector<size_t> rl;
    std::vector<ItemStatus> rst(st ? rest.size() : 0);
    auto rr = BatchGetAutoSgDirect(rest, &rl, st ? rst.data() : nullptr);
    for (size_t m = 0; m < fb.size(); ++m) {
      res[fb[m]] = rr[m];
      if (rr[m]) lens[fb[m]] = rl[m];
      if (st) st[fb[m]] = rst[m];
    }
  }
  // Per-batch split, opt-in (DFKV_CLIENT_NODE_DEDUP_LOG=1): the one line that
  // distinguishes "waits time out" from "claims never dedup" when the server
  // counters look wrong — cheap enough to leave on in a canary.
  static const bool log_split = [] {
    const char* v = std::getenv("DFKV_CLIENT_NODE_DEDUP_LOG");
    const bool enabled = v && std::strcmp(v, "1") == 0;
    envparse::ReportSwitch("DFKV_CLIENT_NODE_DEDUP_LOG", enabled,
                           "only 1 is on");
    return enabled;
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

std::vector<bool> KVClient::BatchGetAutoSgDirectRouted(const std::vector<KvGetItemSg>& items,
                                                 std::vector<size_t>* out_lens,
                                                 ItemStatus* st) {
  const size_t N = items.size();
  std::vector<char> hit(N, 0);
  std::vector<size_t> lens(N, 0);  // distinct indices => thread-safe writes
  if (st) std::fill(st, st + N, ItemStatus::kNoRoute);

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
  std::vector<char> retry(N, 0);  // see BatchGetDirect
  auto run_pass = [&](const std::vector<size_t>& todo) {
    std::map<std::pair<std::string, size_t>, std::vector<size_t>> by;
    std::vector<std::string> routes = RouteReads(
        todo.size(),
        [&](size_t j) -> const std::string& { return items[todo[j]].key; },
        RouteOp::kGet);
    size_t j = 0;
    for (size_t i : todo) {
      std::string node = std::move(routes[j++]);
      if (over[i]) {
        if (st) st[i] = ItemStatus::kBadArgument;
        if (pernode_on)
          RecordPernodeNonIssued(
              unrouted.get(), PernodeNonIssuedReason::kInvalidInput, 1,
              items[i].key);
        continue;
      }
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
    const auto node_of = [&](size_t g) -> const std::string& {
      return groups[g].first.first;
    };
    RunReadScheduled(groups.size(), [&](size_t g) {
      const std::string& node = groups[g].first.first;
      const std::vector<size_t>& idx = groups[g].second;
      uint64_t now = NowMs();
      if (!health_.Healthy(node, now)) {
        if (st)
          for (size_t k : idx) st[k] = ItemStatus::kPeerCooling;
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
        keys.push_back(block_key_(items[k].key));
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
      ReportPeer(node, sts);
      for (size_t m = 0; m < idx.size(); ++m) {
        const size_t cap = groups[g].first.second;
        if (st) st[idx[m]] = ReadItemStatusAt(sts, value_lens, m, cap, false);
        retry[idx[m]] = m < sts.size() && TransientReadMiss(sts[m]);
        if (m >= sts.size() || sts[m] != Status::kOk ||
            m >= value_lens.size() || value_lens[m] > cap)
          continue;
        lens[idx[m]] = value_lens[m];
        hit[idx[m]] = 1;
      }
    }, node_of);
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
        } else if (a.dev.empty()) {
          // A shard timed at 0 us (sub-microsecond) still names its rail.
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
  // Only those two transients (absent, or an I/O error on a node that is
  // still healthy): see BatchGetDirect.
  for (int r = 0; r < get_miss_retries_; ++r) {
    std::vector<size_t> missed;
    for (size_t i = 0; i < N; ++i)
      if (!over[i] && !hit[i] && retry[i]) missed.push_back(i);
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
