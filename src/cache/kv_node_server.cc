#include "cache/kv_node_server.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <string>

#include <vector>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include "common/config_dump.h"
#include "utils/log.h"
#include "utils/env_parse.h"
#include "utils/net_util.h"
#include "utils/thread_name.h"
#include "utils/wire_limits.h"
#include "utils/prom_escape.h"
#include "transport/transport.h"

namespace dfkv {

namespace {
// How long a stopping server lets the RAM tier write its backlog. Parsed
// strictly: the value decides how much is dropped, and "abc" or "5m" must not
// quietly become 0 or 5 seconds.
uint64_t ResolveShutdownDrainTimeoutSeconds() {
  const char* const name = "DFKV_RAM_SHUTDOWN_DRAIN_TIMEOUT_S";
  envparse::U64Spec spec;
  spec.fallback = 30;
  spec.max = 3600;
  spec.above = envparse::Above::kClamp;
  const uint64_t seconds = envparse::EnvU64(name, spec);
  config_dump::RecordResolved(name, std::to_string(seconds));
  return seconds;
}

// Monotonic seconds; only read on the ~1/64 sampled ops, so the cost is amortized
// away on the hot path.
inline double NowSec() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

void KvNodeServer::InitTcpListenerConfig() {
  // allow_zero: knobs where 0 is a documented "feature off" setting (the
  // first-frame deadline) keep 0 instead of falling back with a warning.
  auto bounded = [](const char* name, uint64_t fallback, uint64_t hard_max,
                    bool allow_zero = false) -> uint64_t {
    envparse::U64Spec spec;
    spec.fallback = fallback;
    spec.min = allow_zero ? 0 : 1;
    spec.max = hard_max;
    spec.above = envparse::Above::kClamp;
    spec.overflow_is_default = true;
    return envparse::EnvU64(name, spec);
  };
  tcp_max_connections_ = static_cast<size_t>(
      bounded("DFKV_TCP_MAX_CONNS", kDefaultTcpMaxConnections,
              kHardTcpMaxConnections));
  tcp_io_timeout_seconds_ = static_cast<int>(
      bounded("DFKV_TCP_IO_TIMEOUT_S", kDefaultTcpIoTimeoutSeconds,
              kHardTcpIoTimeoutSeconds));
  tcp_first_req_ms_ =
      bounded("DFKV_TCP_FIRST_REQ_MS", kDefaultTcpFirstReqMs,
              kHardTcpFirstReqMs, /*allow_zero=*/true);
  config_dump::RecordResolved("DFKV_TCP_MAX_CONNS",
                              std::to_string(tcp_max_connections_));
  config_dump::RecordResolved("DFKV_TCP_IO_TIMEOUT_S",
                              std::to_string(tcp_io_timeout_seconds_));
  config_dump::RecordResolved("DFKV_TCP_FIRST_REQ_MS",
                              std::to_string(tcp_first_req_ms_));
}

namespace {
uint64_t SteadyMs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}
}  // namespace

KvNodeServer::KvNodeServer(const std::string& cache_dir,
                           uint64_t capacity_bytes)
    : KvNodeServer(DiskCacheGroup::Options{{cache_dir}, capacity_bytes}) {}

KvNodeServer::KvNodeServer(const std::vector<std::string>& cache_dirs,
                           uint64_t capacity_bytes,
                           const std::atomic<bool>* cancel_start)
    : KvNodeServer(DiskCacheGroup::Options{cache_dirs, capacity_bytes},
                   cancel_start) {}

// The two slow parts of a start are independent of each other: faulting in
// the RAM tier arena and opening the disks (one thread per disk). They run
// side by side. The members are initialized in this order: the plan (the
// environment, read once), the tier under construction on its own thread,
// then group_, whose constructor returns when every disk is open.
KvNodeServer::KvNodeServer(DiskCacheGroup::Options options,
                           const std::atomic<bool>* cancel_start)
    : cancel_start_(cancel_start),
      ram_plan_(PlanRamTier(options.cache_dirs.size())),
      ram_starting_(StartRamTier()),
      group_(OpenedNotActivated(std::move(options))) {
  shutdown_drain_timeout_s_ = ResolveShutdownDrainTimeoutSeconds();
  // Ages on CLOCK_MONOTONIC_RAW: compared across nodes by the read-back's
  // rewrite check (design §3.5), which bounds only the clock's rate.
  remove_memory_ = RemoveMemory::FromEnv(RawMs());
  write_memory_ = WriteMemory::FromEnv(
      remove_memory_ ? remove_memory_->WindowMs() : 0, RawMs());
  // With a disk that failed to open the start is over: the arena need not be
  // faulted in to the end just to be dropped.
  if (!group_.Healthy())
    ram_start_abandoned_.store(true, std::memory_order_relaxed);
  // Always waited for, whatever the disks did: the tier's threads must be
  // gone before this object can be.
  std::unique_ptr<RamTier> tier;
  if (ram_starting_.valid()) tier = ram_starting_.get();
  // A cancelled start keeps what the disks opened, for Shutdown() to close
  // clean, and nothing else. Read once, here: what the tier saw of the flag
  // while it was being built does not matter after this.
  start_cancelled_ = cancel_start_ != nullptr &&
                     cancel_start_->load(std::memory_order_relaxed);
  if (start_cancelled_) return;
  // A disk that failed to open ends the start as before, without a RAM tier;
  // the one built meanwhile holds nothing and is dropped here.
  if (!group_.Healthy()) return;
  if (ram_plan_.enabled) AdoptRamTier(std::move(tier));
  // Everything slow is behind us and nothing has failed: only now are the
  // disks marked in use. Up to here they were open but unchanged, so a start
  // that was killed, cancelled, or failed on a disk or on the RAM tier left
  // every cache exactly as its previous run closed it.
  if (!ram_required_failed_) group_.Activate();
  InitAdmission();
}

DiskCacheGroup::Options KvNodeServer::OpenedNotActivated(
    DiskCacheGroup::Options options) {
  options.defer_activation = true;
  return options;
}

KvNodeServer::~KvNodeServer() {
  // Order matters: Stop() first joins the accept + handler threads so no request
  // is still reading ram_; THEN reset ram_ (stops+joins its flusher, whose
  // callback calls group_.Cache -- group_ is a member, still alive here); group_
  // is destroyed last, after this body.
  Stop();
  ram_.reset();
}

void KvNodeServer::InitAdmission() {
  {
    envparse::U64Spec spec;  // 0 = off, also the default
    spec.above = envparse::Above::kClamp;  // strtoull saturates
    spec.legacy = envparse::Legacy::kPrefixOrZero;
    spec.negative = envparse::Negative::kWraps;
    spec.fallback = put_busy_limit_;
    put_busy_limit_ = static_cast<size_t>(
        envparse::EnvU64("DFKV_PUT_INFLIGHT_LIMIT", spec));
  }
  config_dump::RecordResolved("DFKV_PUT_INFLIGHT_LIMIT",
                              std::to_string(put_busy_limit_));
  // Read-side convoy collapse master switch, plus its timeout/recur sub-knobs
  // (forced now so their defaults show even before the first coalesced read).
  config_dump::RecordResolved("DFKV_READ_COALESCE", coalesce_enabled_ ? "on" : "off");
  if (coalesce_enabled_) ReadCoalescer::RecordConfig();
}

bool KvNodeServer::TryBeginPut() {
  if (put_busy_limit_ == 0) return true;
  size_t inflight = put_inflight_.load(std::memory_order_relaxed);
  while (inflight < put_busy_limit_) {
    if (put_inflight_.compare_exchange_weak(
            inflight, inflight + 1, std::memory_order_relaxed))
      return true;
  }
  put_busy_.fetch_add(1, std::memory_order_relaxed);
  return false;
}

void KvNodeServer::EndPut() {
  if (put_busy_limit_ != 0)
    put_inflight_.fetch_sub(1, std::memory_order_relaxed);
}

KvNodeServer::PutKeyLock KvNodeServer::LockPutKey(const BlockKey& key) {
  if (!ram_write_back_ || !ram_) return {};
  PutKeyStripe& stripe = put_keys_[BlockKeyMixHash{}(key) % kPutKeyStripes];
  PutKeySlot* slot = nullptr;
  {
    std::lock_guard<std::mutex> guard(stripe.mu);
    auto& entry = stripe.keys[key];
    if (!entry && !stripe.free.empty()) {
      entry = std::move(stripe.free.back());
      stripe.free.pop_back();
    }
    if (!entry) entry = std::make_unique<PutKeySlot>();
    slot = entry.get();
    ++slot->users;
  }
  PutKeyLock out(this, key);
  out.lock = std::unique_lock<std::mutex>(slot->mu, std::try_to_lock);
  if (out.lock.owns_lock()) return out;
  const auto started = std::chrono::steady_clock::now();
  out.lock.lock();
  put_key_lock_waits_.fetch_add(1, std::memory_order_relaxed);
  put_key_lock_wait_us_.fetch_add(
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - started)
                                .count()),
      std::memory_order_relaxed);
  return out;
}

void KvNodeServer::ReleasePutKey(const BlockKey& key,
                                 std::unique_lock<std::mutex>* lock) {
  if (lock->owns_lock()) lock->unlock();
  lock->release();  // the slot may go below
  PutKeyStripe& stripe = put_keys_[BlockKeyMixHash{}(key) % kPutKeyStripes];
  std::lock_guard<std::mutex> guard(stripe.mu);
  auto it = stripe.keys.find(key);
  if (it != stripe.keys.end() && --it->second->users == 0) {
    // Kept for the next key of the stripe, a few at most.
    if (stripe.free.size() < kPutKeyFreeSlots)
      stripe.free.push_back(std::move(it->second));
    stripe.keys.erase(it);
  }
}

// DFKV_RAM_RECLAIM_MS: background free-slot reclaimer cadence, ms; 0
// disables. Read with an unchecked strtoul and stored in 32 bits up to 2.38,
// which is kept: an empty or non-numeric value is 0 (reclaim off), and a
// number is cut to its low 32 bits -- so that -1 is 4294967295 and
// 4294967296 is 0. All of those are reported, with the value really used.
static uint32_t RamReclaimMsFromEnv(uint32_t fallback) {
  const char* const name = "DFKV_RAM_RECLAIM_MS";
  const char* text = std::getenv(name);
  envparse::U64Spec spec;
  spec.fallback = fallback;
  spec.legacy = envparse::Legacy::kPrefixOrZero;
  spec.negative = envparse::Negative::kWraps;
  spec.empty_is_zero = true;
  spec.zero_means = "no background reclaim";
  std::string warning;
  const uint64_t read = envparse::ParseU64(name, text, spec, &warning);
  const uint32_t used = static_cast<uint32_t>(read);
  if (used != read) {
    warning = "invalid " + envparse::detail::Quoted(name, text) +
              ": does not fit 32 bits; using " + std::to_string(used) +
              " as before" +
              (used == 0 ? ", which means no background reclaim (write 0 if "
                           "that is intended)"
                         : "") +
              ". A later release will not accept this value.";
  }
  envparse::WarnOnce(name, warning);
  return used;
}

KvNodeServer::RamTierPlan KvNodeServer::PlanRamTier(size_t disk_count) {
  RamTierPlan plan;
  // Off by default. DFKV_RAM_TIER in {1,on,true,yes} enables the RAM hot tier;
  // DFKV_RAM_TIER_BYTES sizes the pre-registered arena (default 16 GiB).
  // The words are matched in lower case only, as they always were: a value
  // in another case used to leave the tier off, and turning a multi-gigabyte
  // locked arena on at an upgrade could fail a start that works. It still
  // leaves the tier off, but no longer without saying so.
  const char* e = std::getenv("DFKV_RAM_TIER");
  const std::string v = (e && *e) ? std::string(e) : "";
  plan.enabled = (v == "1" || v == "on" || v == "true" || v == "yes");
  if (!plan.enabled && !v.empty() && v != "0" && v != "off" && v != "false" &&
      v != "no")
    envparse::WarnOnce(
        "DFKV_RAM_TIER",
        "invalid DFKV_RAM_TIER='" + v +
            "': the RAM tier stays off, as before. Only 1, on, true and yes "
            "in lower case switch it on.");
  config_dump::RecordResolved("DFKV_RAM_TIER", plan.enabled ? "on" : "off");
  if (!plan.enabled) return plan;
  RamTier::Options& o = plan.options;
  {
    envparse::U64Spec spec;
    spec.negative = envparse::Negative::kWraps;
    spec.legacy = envparse::Legacy::kPrefixOrZero;
    spec.fallback = o.bytes;
    spec.min = 1;
    o.bytes = envparse::EnvU64("DFKV_RAM_TIER_BYTES", spec);
  }
  // RAM write policy: writeback (default) admits PUT into the arena and flushes
  // asynchronously; writearound writes directly to disk and fills the arena via
  // GET read-promotion. Any value other than "writearound" keeps write-back.
  const char* wm = std::getenv("DFKV_RAM_WRITE_MODE");
  std::string mode = wm ? std::string(wm) : "";
  if (mode != "writeback" && mode != "writearound") {
    if (!mode.empty())
      DFKV_LOG_WARN("DFKV_RAM_WRITE_MODE='" + mode +
                    "' not recognized; using writeback");
    mode = "writeback";
  }
  plan.write_back = mode == "writeback";
  config_dump::RecordResolved("DFKV_RAM_WRITE_MODE", mode);
  // No dedicated budget for values larger than an arena extent unless
  // DFKV_RAM_TIER_LARGE_RESERVE_BYTES asks for one: with the default
  // --max-msg (one extent) no value can use it, and the two extents it took
  // from the arena were lost; with a larger --max-msg it held two such
  // values at a time. Such values go to the disk and are read from it.
  o.large_reserve_bytes = 0;
  // Flush workers default to 4x the disk count (cap 16). One sync DIO stream
  // per worker leaves NVMe queues nearly idle for small objects; measured on a
  // 3-disk node (64 KiB saturated writes): 3 workers 6.8k ops/s -> 16 workers
  // 11.7k ops/s (+73%), PUT p99 95 -> 37 ms. The old disk-count default dated
  // from when the ingest path's lock contention masked any worker scaling.
  // The count is that of the configured directories: the disks are being
  // opened while this tier is built, and a start with fewer does not go on.
  o.flush_threads = static_cast<uint32_t>(
      std::min<size_t>(4 * (disk_count ? disk_count : 1), 16));
  // Flush queues and dirty accounting by disk (see Options::flush_lane):
  // the count of the configured directories, for the same reason.
  o.flush_lanes = disk_count ? disk_count : 1;
  {
    envparse::U64Spec spec;
    spec.negative = envparse::Negative::kWraps;
    spec.legacy = envparse::Legacy::kPrefixOrZero;
    spec.fallback = o.flush_threads;
    spec.min = 1;
    spec.max = 64;
    o.flush_threads =
        static_cast<uint32_t>(envparse::EnvU64("DFKV_RAM_FLUSH_THREADS", spec));
  }
  o.reclaim_interval_ms = RamReclaimMsFromEnv(o.reclaim_interval_ms);
  const char* ack_env = std::getenv("DFKV_PUT_ACK_MODE");
  const std::string ack = ack_env ? std::string(ack_env) : "";
  if (ack.empty()) {
    plan.ack_enabled = plan.write_back;
  } else if (ack == "ram" || ack == "disk") {
    plan.ack_enabled = ack == "ram";
  } else {
    DFKV_LOG_WARN("DFKV_PUT_ACK_MODE='" + ack +
                  "' not recognized; using RAM-ACK for writeback");
    plan.ack_enabled = plan.write_back;
  }
  // 0 means "acknowledge from disk". An empty or non-numeric value reads as
  // 0 and so changes the ACK mode, as it always did; the parser now says so.
  {
    envparse::U64Spec spec;
    spec.negative = envparse::Negative::kWraps;
    spec.legacy = envparse::Legacy::kPrefixOrZero;
    spec.empty_is_zero = true;
    spec.zero_means = "every PUT is acknowledged from disk";
    spec.fallback = o.ack_high_watermark_pct;
    spec.max = 100;
    o.ack_high_watermark_pct = static_cast<uint32_t>(
        envparse::EnvU64("DFKV_RAM_ACK_HIGH_WATERMARK_PCT", spec));
  }
  // Longest a PUT is parked for the value that owns its key (ms).
  {
    envparse::U64Spec spec;
    spec.zero_means = "a parked PUT waits without limit";
    spec.fallback = o.put_wait_ms;
    spec.max = 3600000;
    o.put_wait_ms =
        static_cast<uint32_t>(envparse::EnvU64("DFKV_RAM_PUT_WAIT_MS", spec));
  }
  config_dump::RecordResolved("DFKV_RAM_PUT_WAIT_MS",
                              std::to_string(o.put_wait_ms));
  // REQ-26: how long a RAM-ACK PUT of a new key waits at the dirty
  // watermark before it is refused (0 = refuse at once, as before).
  {
    envparse::U64Spec spec;
    spec.fallback = o.ack_backpressure_wait_ms;
    spec.max = 60000;
    o.ack_backpressure_wait_ms = static_cast<uint32_t>(
        envparse::EnvU64("DFKV_RAM_ACK_BACKPRESSURE_WAIT_MS", spec));
  }
  config_dump::RecordResolved("DFKV_RAM_ACK_BACKPRESSURE_WAIT_MS",
                              std::to_string(o.ack_backpressure_wait_ms));
  config_dump::RecordResolved("DFKV_PUT_ACK_MODE",
                              plan.ack_enabled ? "ram" : "disk");
  config_dump::RecordResolved("DFKV_RAM_ACK_HIGH_WATERMARK_PCT",
                              std::to_string(o.ack_high_watermark_pct));
  config_dump::RecordResolved("DFKV_RAM_TIER_BYTES", std::to_string(o.bytes));
  config_dump::RecordResolved("DFKV_RAM_RECLAIM_MS", std::to_string(o.reclaim_interval_ms));
  // Which whole-value cold reads are promoted into RAM: every one (first,
  // as before) or only a key's repeat read within the window (second), so a
  // one-time scan does not push out values that are read again.
  const char* pe = std::getenv("DFKV_RAM_PROMOTE");
  std::string promote = pe ? std::string(pe) : "";
  if (promote != "first" && promote != "second") {
    if (!promote.empty())
      envparse::WarnOnce("DFKV_RAM_PROMOTE",
                         "invalid DFKV_RAM_PROMOTE='" + promote +
                             "': only first and second are known; using "
                             "first");
    promote = "first";
  }
  plan.promote_second = promote == "second";
  config_dump::RecordResolved("DFKV_RAM_PROMOTE", promote);
  if (plan.promote_second) {
    envparse::U64Spec spec;
    spec.fallback = plan.promote_window_ms;
    spec.min = 1;
    spec.max = 86400000;
    plan.promote_window_ms =
        envparse::EnvU64("DFKV_RAM_PROMOTE_WINDOW_MS", spec);
    config_dump::RecordResolved("DFKV_RAM_PROMOTE_WINDOW_MS",
                                std::to_string(plan.promote_window_ms));
  }
  return plan;
}

// Runs while group_ is still being constructed. The callbacks handed to the
// tier name group_ and other members of this object, which is safe for one
// reason only: the tier calls them for objects it holds, and it holds none
// until the server takes requests, long after the constructor has returned.
// Nothing else here may touch a member declared after ram_starting_.
std::future<std::unique_ptr<RamTier>> KvNodeServer::StartRamTier() {
  if (!ram_plan_.enabled) return {};
  RamTier::Options o = ram_plan_.options;
  o.is_persisted = [this](const BlockKey& key) {
    return group_.IsCached(key);
  };
  o.disk_generation = [this](const BlockKey& key) {
    return DiskGeneration(key);
  };
  o.cancel_start = [this] { return RamStartUnwanted(); };
  // A flush to a disk store that has failed for good is not tried again.
  o.flush_target_failed = [this](const BlockKey& key) {
    return group_.DiskFailedFor(key);
  };
  // Flush queues and dirty accounting by disk: a slow disk holds up its own
  // values only. The lane count comes from the plan: this runs before group_
  // is constructed (it is declared after ram_starting_), so group_ may only
  // be used by what runs later, as this lambda (PUT admission) is.
  o.flush_lane = [this](const BlockKey& key) { return group_.DiskIndexOf(key); };
  // A value that will not be written takes its parked RETAIN SET with it.
  o.flush_given_up = [this](const BlockKey& key) {
    ApplyPendingRetain(key, /*flushed=*/false);
  };
  return std::async(std::launch::async, [this, o] {
    // Flusher persists a RAM slot to the disk group. CacheDirect (not Cache):
    // the arena slot is 4 KiB-aligned with cap slack, so a direct-mode slab
    // lands it via O_DIRECT -- a buffered flush would route the RAM tier's
    // entire write volume back through the page cache, defeating the tier's
    // purpose of keeping memory use bounded and explicit on GPU nodes.
    return std::make_unique<RamTier>(
        o, [this](const BlockKey& k, char* d, size_t l, size_t cap) {
          // Only a written value: a refused or failed attempt may be followed
          // by another one, and the parked SET is for the value that lands.
          const Status st = group_.CacheDirect(k, d, l, cap);
          if (st == Status::kOk) ApplyPendingRetain(k, /*flushed=*/true);
          return st;
        });
  });
}

void KvNodeServer::AdoptRamTier(std::unique_ptr<RamTier> tier) {
  ram_write_back_ = ram_plan_.write_back;
  ram_ack_enabled_ = ram_plan_.ack_enabled;
  if (ram_plan_.promote_second)
    recent_reads_ = std::make_unique<RecentReads>(ram_plan_.promote_window_ms,
                                                  kRecentReadsMax);
  ram_flush_disks_.reserve(group_.DiskCount());
  for (size_t i = 0; i < group_.DiskCount(); ++i)
    ram_flush_disks_.push_back(std::make_unique<RamFlushDiskMetrics>());
  // Batched sink: a flush worker's whole dequeue rides one lock-amortized
  // (and, with a uring build, one-submit) store visit per disk.
  tier->set_flush_batch([this](const std::vector<RamTier::FlushItem>& items) {
    std::vector<StoreEngine::CacheBatchItem> b;
    b.reserve(items.size());
    for (const auto& it : items)
      b.push_back(StoreEngine::CacheBatchItem{it.key, it.data, it.len, it.cap});
    std::vector<Status> sts = group_.CacheDirectBatch(
        b, [this](size_t disk, size_t objects, uint64_t bytes,
                  size_t failures, double seconds) {
          if (disk >= ram_flush_disks_.size()) return;
          auto& m = *ram_flush_disks_[disk];
          m.batches.fetch_add(1, std::memory_order_relaxed);
          m.objects.fetch_add(objects, std::memory_order_relaxed);
          m.bytes.fetch_add(bytes, std::memory_order_relaxed);
          m.failures.fetch_add(failures, std::memory_order_relaxed);
          m.latency.Observe(seconds);
        });
    for (size_t i = 0; i < sts.size(); ++i)
      if (sts[i] == Status::kOk)  // see the per-item sink above
        ApplyPendingRetain(items[i].key, /*flushed=*/true);
    return sts;
  });
  if (tier->ok()) {
    ram_ = std::move(tier);
  } else {
    ram_required_failed_ = true;
    DFKV_LOG_ERROR(
        "requested RAM tier initialization failed; refusing disk-only startup");
  }
}

Status KvNodeServer::Start(int port) {
  if (!Healthy()) {
    DFKV_LOG_ERROR("refusing startup: local storage initialization failed");
    return Status::kIOError;
  }
  InitTcpListenerConfig();
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) return Status::kIOError;
  int one = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  // Listen on all interfaces so peer nodes can reach this cache node (a
  // distributed cluster needs a routable bind, not loopback). Access control is
  // by network isolation / firewall, per docs/DEPLOY.md.
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
  accept_thread_ = std::thread([this] { NameThisThread("kv-accept"); AcceptLoop(); });
  return Status::kOk;
}

namespace {
std::string Seconds(std::chrono::steady_clock::time_point since) {
  char text[32];
  std::snprintf(text, sizeof(text), "%.2f s",
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - since).count());
  return text;
}
}  // namespace

void KvNodeServer::CancelParkedPuts() {
  if (ram_) ram_->CancelPutWaits();
}

void KvNodeServer::Stop() {
  CancelParkedPuts();
  if (!StopRequests()) return;  // idempotent
  if (ram_) {
    const uint64_t timeout_seconds = shutdown_drain_timeout_s_;
    const bool drained = ram_->WaitForDrain(
        std::chrono::seconds(timeout_seconds));
    if (!drained) {
      DFKV_LOG_WARN("RAM shutdown drain timed out after " +
                    std::to_string(timeout_seconds) + "s with " +
                    std::to_string(ram_->DirtyObjects()) +
                    " dirty objects; destructor will continue safe draining");
    }
  }
}

bool KvNodeServer::Shutdown(std::chrono::steady_clock::time_point started) {
  if (shutdown_done_) return shutdown_clean_;
  shutdown_done_ = true;
  using Clock = std::chrono::steady_clock;
  auto stage = Clock::now();
  CancelParkedPuts();
  StopRequests();
  DFKV_LOG_INFO("shutdown: TCP requests stopped in " + Seconds(stage));

  if (ram_) {
    stage = Clock::now();
    const uint64_t timeout_seconds = shutdown_drain_timeout_s_;
    // The flushers have been writing since `started`: stopping the request
    // paths waits for PUTs that are parked until their value is on disk.
    const auto deadline = started + std::chrono::seconds(timeout_seconds);
    const uint64_t objects_before = ram_->DirtyObjects();
    const uint64_t bytes_before = ram_->DirtyBytes();
    auto next_progress = stage + std::chrono::seconds(5);
    bool drained = true;
    while (ram_->DirtyObjects() != 0) {
      const auto now = Clock::now();
      if (now >= deadline) {
        drained = false;
        break;
      }
      if (now >= next_progress) {
        DFKV_LOG_INFO("shutdown: RAM tier still has " +
                      std::to_string(ram_->DirtyObjects()) + " objects (" +
                      std::to_string(ram_->DirtyBytes()) +
                      " bytes) to write to disk after " + Seconds(stage));
        next_progress += std::chrono::seconds(5);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    // Stops the flushers: no disk write is in flight once this returns.
    ram_->StopBackground(/*abandon_backlog=*/!drained);
    if (drained) {
      DFKV_LOG_INFO("shutdown: RAM tier written to disk (" +
                    std::to_string(objects_before) + " objects, " +
                    std::to_string(bytes_before) + " bytes) in " +
                    Seconds(stage));
    } else {
      DFKV_LOG_WARN(
          "shutdown: RAM tier not written to disk within " +
          std::to_string(timeout_seconds) +
          " s of the stop (DFKV_RAM_SHUTDOWN_DRAIN_TIMEOUT_S); dropping the " +
          std::to_string(ram_->DirtyObjects()) + " objects (" +
          std::to_string(ram_->DirtyBytes()) +
          " bytes) that are only in RAM, of " +
          std::to_string(objects_before) + " (" +
          std::to_string(bytes_before) + " bytes) at the start");
    }
  }

  stage = Clock::now();
  size_t clean = 0;
  const auto report = group_.Close();
  for (const auto& disk : report) {
    char seconds[32];
    std::snprintf(seconds, sizeof(seconds), "%.2f s", disk.seconds);
    if (disk.clean) {
      ++clean;
      DFKV_LOG_INFO("shutdown: disk " + disk.dir + " synced and clean in " +
                    seconds);
    } else {
      DFKV_LOG_WARN("shutdown: disk " + disk.dir +
                    " could not record a clean shutdown (" + seconds +
                    "); the next start treats it as an unclean exit");
    }
  }
  DFKV_LOG_INFO("shutdown: " + std::to_string(clean) + " of " +
                std::to_string(report.size()) + " disks clean in " +
                Seconds(stage));
  // The report covers the disks the group has. One whose open threw is not
  // among them, and a group that lost a disk that way is not "all clean"
  // however the others closed. (A disk that failed later, while serving, is
  // in the report and has said so itself.)
  const bool all_closed_clean = clean == report.size();
  const bool disk_missing = all_closed_clean && !group_.Healthy();
  if (disk_missing)
    DFKV_LOG_WARN("shutdown: a disk that failed to open is not among them");
  shutdown_clean_ = all_closed_clean && !disk_missing;
  return shutdown_clean_;
}

bool KvNodeServer::StopRequests() {
  if (!running_.exchange(false)) return false;  // idempotent
  // shutdown() (a read of listen_fd_) unblocks accept(); join the accept loop so
  // there is no concurrent reader before we mutate/close listen_fd_.
  if (listen_fd_ >= 0) ::shutdown(listen_fd_, SHUT_RDWR);
  if (accept_thread_.joinable()) accept_thread_.join();
  if (listen_fd_ >= 0) { ::close(listen_fd_); listen_fd_ = -1; }
  // accept loop is done; drain in-flight connections: unblock their recv(), then
  // join handler threads so group_ outlives them.
  std::vector<int> fds;
  std::vector<Conn> conns;
  {
    std::lock_guard<std::mutex> lk(conn_mu_);
    fds.assign(conn_fds_.begin(), conn_fds_.end());
    conns.swap(conns_);
  }
  for (int fd : fds) ::shutdown(fd, SHUT_RDWR);
  for (auto& c : conns) if (c.th.joinable()) c.th.join();
  return true;
}

void KvNodeServer::ReapDoneLocked() {
  for (auto it = conns_.begin(); it != conns_.end();) {
    if (it->done->load(std::memory_order_acquire)) {
      if (it->th.joinable()) it->th.join();
      it = conns_.erase(it);
    } else {
      ++it;
    }
  }
}

size_t KvNodeServer::live_conn_count() {
  std::lock_guard<std::mutex> lk(conn_mu_);
  ReapDoneLocked();
  return conns_.size();
}

#ifndef DFKV_VERSION
#define DFKV_VERSION "dev"
#endif
#ifdef DFKV_WITH_RDMA
#define DFKV_BUILD_TRANSPORT "rdma"
#else
#define DFKV_BUILD_TRANSPORT "tcp"
#endif

std::string KvNodeServer::MetricsText() const {
  // Node identity as an inner label set (no braces), empty when unset so a
  // single-node scrape and older tooling keep the bare `metric N` form.
  std::string idlabels;
  if (!node_id_.empty() || !node_group_.empty())
    idlabels = "node=\"" + PromLabelEscape(node_id_) + "\",group=\"" +
               PromLabelEscape(node_group_) + "\"";
  // braces(extra): merge `extra` label set with the identity labels.
  auto braces = [&](const std::string& extra) {
    std::string in = extra;
    if (!idlabels.empty()) in += (in.empty() ? "" : ",") + idlabels;
    return in.empty() ? std::string() : "{" + in + "}";
  };
  const std::string lbl = braces("");
  std::string s;
  auto metric = [&](const char* name, const char* type, const char* help,
                    uint64_t v) {
    s += "# HELP "; s += name; s += " "; s += help; s += "\n";
    s += "# TYPE "; s += name; s += " "; s += type; s += "\n";
    s += name; s += lbl; s += " "; s += std::to_string(v); s += "\n";
  };
  auto rd = [](const std::atomic<size_t>& a) { return a.load(std::memory_order_relaxed); };
  // build/version + uptime
  s += "# HELP dfkv_build_info Build and version info (constant 1)\n";
  s += "# TYPE dfkv_build_info gauge\n";
  s += std::string("dfkv_build_info{version=\"") + DFKV_VERSION +
       "\",transport=\"" DFKV_BUILD_TRANSPORT "\",engine=\"" +
       PromLabelEscape(group_.EngineName()) + "\",write_mode=\"" +
       PromLabelEscape(group_.WriteMode().empty() ? "n/a" :
                                                group_.WriteMode()) + "\"";
  if (!node_id_.empty() || !node_group_.empty())
    s += ",node=\"" + PromLabelEscape(node_id_) + "\",group=\"" +
         PromLabelEscape(node_group_) + "\"";
  s += "} 1\n";
  uint64_t up = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - start_time_).count();
  metric("dfkv_uptime_seconds", "gauge", "Seconds since node start", up);
  // counters
  metric("dfkv_storage_healthy", "gauge",
         "Current local disk and requested RAM-tier health (readiness input)",
         Healthy() ? 1 : 0);
  metric("dfkv_cache_put_total", "counter", "Cache (PUT) ops accepted", m_cache_put());
  metric("dfkv_cache_hit_total", "counter", "Range (GET) ops that hit", m_cache_hit());
  metric("dfkv_cache_miss_total", "counter", "Range (GET) ops that missed", m_cache_miss());
  metric("dfkv_exist_hit_total", "counter", "Exist checks that hit", m_exist_hit());
  metric("dfkv_exist_miss_total", "counter", "Exist checks that missed", m_exist_miss());
  {
    static const char* kModes[2] = {"set", "refresh"};
    static const char* kResults[8] = {"miss",      "hit_unprotected",
                                      "protected", "refreshed",
                                      "upgraded",  "unchanged",
                                      "denied",    "other"};
    s += "# HELP dfkv_retain_keys_total kRetain per-key outcomes by mode and result\n";
    s += "# TYPE dfkv_retain_keys_total counter\n";
    for (int m = 0; m < 2; ++m)
      for (int r = 0; r < 8; ++r)
        s += "dfkv_retain_keys_total" +
             braces(std::string("mode=\"") + kModes[m] + "\",result=\"" +
                    kResults[r] + "\"") +
             " " + std::to_string(rd(retain_keys_[m][r])) + "\n";
    metric("dfkv_retain_requests_total", "counter", "kRetain requests served",
           rd(retain_requests_));
    metric("dfkv_retain_pending_applied_total", "counter",
           "RAM write-back SETs applied when the flush committed",
           rd(pending_retain_applied_));
    metric("dfkv_retain_pending_denied_total", "counter",
           "Parked RAM write-back SETs the disk denied when they were applied "
           "(retention budget full)",
           rd(pending_retain_denied_));
    {
      size_t parked = 0;
      {
        std::lock_guard<std::mutex> lk(pending_retain_mu_);
        parked = pending_retain_.size();
      }
      metric("dfkv_retain_pending_entries", "gauge",
             "SETs parked for RAM values still to be written", parked);
    }
    const auto rs = group_.RetentionStats();
    metric("dfkv_retain_enabled", "gauge", "Retention enabled on every disk",
           group_.RetainEnabled() ? 1 : 0);
    metric("dfkv_retain_max_ttl_seconds", "gauge", "Cap on any applied TTL",
           group_.RetainMaxTtl());
    metric("dfkv_retain_budget_bytes", "gauge", "Protected-bytes budget",
           rs.budget_bytes);
    metric("dfkv_retained_bytes", "gauge", "Slot bytes currently protected",
           rs.protected_bytes);
    metric("dfkv_retained_slots", "gauge", "Residents currently protected",
           rs.protected_slots);
    metric("dfkv_retain_headroom_pct", "gauge",
           "Remaining retention budget, minimum across disks",
           group_.RetainHeadroomPct());
    metric("dfkv_retain_expired_total", "counter",
           "Protections released by the expiry wheel", rs.expired);
    metric("dfkv_evict_skipped_protected_total", "counter",
           "Eviction candidates skipped because they are protected",
           rs.evict_skipped);
    metric("dfkv_slab_extents_pinned_by_retention", "gauge",
           "Extents holding at least one protected resident", rs.extents_pinned);
    metric("dfkv_retain_wheel_entries", "gauge",
           "Expiry-wheel entries: one per protected resident, plus those of "
           "protected residents that were removed, until their bucket expires",
           rs.wheel_entries);
  }
  if (remove_memory_) {
    const uint64_t now = RawMs();
    metric("dfkv_remove_memory_entries", "gauge",
           "Keys the recent-REMOVE memory holds (DFKV_REMOVE_MEMORY_MS)",
           remove_memory_->Entries());
    metric("dfkv_remove_memory_covered_seconds", "gauge",
           "How far back this node knows every REMOVE it acknowledged",
           remove_memory_->CoveredMs(now) / 1000);
    metric("dfkv_remove_memory_dropped_total", "counter",
           "Records dropped for room (what the node knows then starts later)",
           remove_memory_->Dropped());
    metric("dfkv_removed_within_ages_requests_total", "counter",
           "kRemovedWithinAges (op 26) requests answered",
           removed_within_ages_requests_.load(std::memory_order_relaxed));
    if (write_memory_) {
      metric("dfkv_write_memory_entries", "gauge",
             "Keys the recent-PUT memory holds (DFKV_WRITE_MEMORY_ENTRIES)",
             write_memory_->Entries());
      metric("dfkv_write_memory_dropped_total", "counter",
             "Recent-PUT records dropped for room",
             write_memory_->Dropped());
      s += "# HELP dfkv_write_memory_cleared_total Times the recent-PUT "
           "memory was cleared because a clock moved against "
           "CLOCK_MONOTONIC_RAW by more than 1 s within its window, by clock\n"
           "# TYPE dfkv_write_memory_cleared_total counter\n";
      s += "dfkv_write_memory_cleared_total{reason=\"realtime\"} " +
           std::to_string(write_memory_->Cleared(WriteMemory::kRealtime)) + "\n";
      s += "dfkv_write_memory_cleared_total{reason=\"boottime\"} " +
           std::to_string(write_memory_->Cleared(WriteMemory::kBoottime)) + "\n";
    }
    s += "# HELP dfkv_removed_within_queries_total Keys asked about by "
         "kRemovedWithin, by answer\n"
         "# TYPE dfkv_removed_within_queries_total counter\n";
    const char* names[3] = {"not_removed", "removed", "unknown"};
    for (uint8_t a = 0; a < 3; ++a)
      s += "dfkv_removed_within_queries_total" +
           braces(std::string("result=\"") + names[a] + "\"") + " " +
           std::to_string(remove_memory_->QueriedKeys(a)) + "\n";
  }
  metric("dfkv_remove_ok_total", "counter", "Remove ops that dropped a block", m_remove_ok());
  metric("dfkv_remove_miss_total", "counter", "Remove ops for an absent block", m_remove_miss());
  metric("dfkv_bytes_written_total", "counter", "Payload bytes written",
         bytes_written_.load(std::memory_order_relaxed));
  metric("dfkv_bytes_read_total", "counter", "Payload bytes read",
         bytes_read_.load(std::memory_order_relaxed));
  metric("dfkv_read_coalesce_leaders_total", "counter",
         "Disk reads executed on behalf of a convoy", read_coalescer_.leaders());
  metric("dfkv_read_coalesced_total", "counter",
         "GETs served from an in-flight identical read", read_coalescer_.coalesced());
  metric("dfkv_read_coalesce_timeouts_total", "counter",
         "Coalesce waiters that timed out and fell back to their own read",
         read_coalescer_.timeouts());
  metric("dfkv_read_coalesce_recur_total", "counter",
         "Staged reads that hit a recurrence tombstone (drift diagnostic)",
         read_coalescer_.recur_hits());
  metric("dfkv_read_coalesce_held_reads_total", "counter",
         "Coalesce followers that read for themselves because their serve "
         "thread held an unfinished read of its own",
         read_coalescer_.held_fallbacks());
  metric("dfkv_put_key_lock_waits_total", "counter",
         "PUTs that waited for the lock serializing PUTs of their key",
         put_key_lock_waits_.load(std::memory_order_relaxed));
  metric("dfkv_put_key_lock_wait_us_total", "counter",
         "Microseconds PUTs waited for that lock",
         put_key_lock_wait_us_.load(std::memory_order_relaxed));
  metric("dfkv_accepts_total", "counter", "TCP connections accepted", AcceptCount());
  metric("dfkv_evictions_total", "counter", "Objects evicted (capacity pressure)",
         group_.Evictions());
  metric("dfkv_evicted_bytes_total", "counter", "Bytes evicted", group_.EvictedBytes());
  // gauges
  metric("dfkv_objects", "gauge", "Cached objects on this node", group_.Count());
  metric("dfkv_used_bytes", "gauge", "Bytes used on disk", group_.UsedBytes());
  metric("dfkv_disks", "gauge", "Backing NVMe disks", group_.DiskCount());
  metric("dfkv_disks_failed", "gauge",
         "Backing disks whose store has failed for good; their keys go to "
         "the other disks", group_.FailedDisks());
  metric("dfkv_server_degraded", "gauge",
         "1 while some disks have failed and the node still serves on the "
         "others (at least DFKV_DEGRADED_MIN_DISKS healthy)",
         DisksDegraded() ? 1 : 0);
  metric("dfkv_tenant_default_quota_bytes", "gauge",
         "Default per-node tenant quota bytes (0 means unlimited)",
         group_.TenantDefaultQuotaBytes());
  metric("dfkv_tenant_quota_rejections_total", "counter",
         "PUT items rejected by per-node tenant quotas",
         group_.TenantQuotaRejections());
  metric("dfkv_tenant_quota_reloads_total", "counter",
         "Tenant quotas file reloads (SIGHUP) that took effect",
         group_.TenantQuotaReloads());
  metric("dfkv_tenant_quota_reload_failures_total", "counter",
         "Tenant quotas file reloads that failed (the old quotas stay)",
         group_.TenantQuotaReloadFailures());
  const auto tenant_quotas = group_.ConfiguredTenantQuotaMetrics();
  auto tenant_metric = [&](const char* name, const char* type,
                           const char* help, auto value) {
    s += "# HELP "; s += name; s += " "; s += help; s += "\n";
    s += "# TYPE "; s += name; s += " "; s += type; s += "\n";
    for (const auto& tenant : tenant_quotas) {
      const std::string labels =
          braces("tenant_hash=\"" +
                 BlockKey::Hex64(tenant.tenant_hash) + "\"");
      s += name; s += labels; s += " ";
      s += std::to_string(value(tenant)); s += "\n";
    }
  };
  tenant_metric("dfkv_tenant_quota_limit_bytes", "gauge",
                "Configured per-node quota for this tenant hash",
                [](const auto& tenant) { return tenant.limit_bytes; });
  tenant_metric("dfkv_tenant_used_bytes", "gauge",
                "Committed payload bytes for this configured tenant hash",
                [](const auto& tenant) { return tenant.used_bytes; });
  tenant_metric("dfkv_tenant_quota_rejections_by_hash_total", "counter",
                "PUT items rejected for this configured tenant hash",
                [](const auto& tenant) { return tenant.rejections; });
  metric("dfkv_open_connections", "gauge", "Currently open client connections",
         rd(open_connections_));
  metric("dfkv_tcp_max_connections", "gauge",
         "Configured cache TCP handler admission limit",
         tcp_max_connections_);
  metric("dfkv_tcp_io_timeout_seconds", "gauge",
         "Configured cache TCP per-socket receive/send timeout",
         tcp_io_timeout_seconds_);
  metric("dfkv_tcp_first_req_ms", "gauge",
         "Absolute deadline for the first complete request frame after accept "
         "(0 = disabled; anti slow-drip bound, not per-syscall)",
         tcp_first_req_ms_);
  metric("dfkv_tcp_rejected_connections_total", "counter",
         "Cache TCP connections rejected at the handler admission limit",
         rd(tcp_rejected_connections_));
  metric("dfkv_tcp_accept_failures_total", "counter",
         "accept() failures from descriptor or memory exhaustion; the cache "
         "TCP listener backed off after each",
         TcpAcceptFailures());
  metric("dfkv_tcp_thread_start_failures_total", "counter",
         "Cache TCP connections refused because their handler thread could "
         "not be created",
         TcpThreadStartFailures());
  // errors by op+status (one HELP/TYPE, multiple labeled series)
  s += "# HELP dfkv_errors_total Failed ops by op and status\n";
  s += "# TYPE dfkv_errors_total counter\n";
  s += "dfkv_errors_total" + braces("op=\"put\",status=\"io\"") + " " +
       std::to_string(rd(put_io_err_)) + "\n";
  s += "dfkv_errors_total" + braces("op=\"get\",status=\"io\"") + " " +
       std::to_string(rd(get_io_err_)) + "\n";
  s += "dfkv_errors_total" + braces("op=\"any\",status=\"invalid\"") + " " +
       std::to_string(rd(invalid_ops_)) + "\n";
  // per-disk gauges (one HELP/TYPE, one series per disk)
  s += "# HELP dfkv_disk_used_bytes Bytes used per backing disk\n";
  s += "# TYPE dfkv_disk_used_bytes gauge\n";
  for (size_t i = 0; i < group_.DiskCount(); ++i)
    s += "dfkv_disk_used_bytes" + braces("disk=\"" + PromLabelEscape(group_.DiskPath(i)) + "\"") + " " +
         std::to_string(group_.DiskUsedBytes(i)) + "\n";
  s += "# HELP dfkv_disk_objects Objects per backing disk\n";
  s += "# TYPE dfkv_disk_objects gauge\n";
  for (size_t i = 0; i < group_.DiskCount(); ++i)
    s += "dfkv_disk_objects" + braces("disk=\"" + PromLabelEscape(group_.DiskPath(i)) + "\"") + " " +
         std::to_string(group_.DiskObjects(i)) + "\n";
  s += "# HELP dfkv_disks_open_dropped_keys Keys this start dropped after a run with failed disks: every key of a disk that had failed, of every disk when a record could not be used, and keys found on a disk that is not their home\n";
  s += "# TYPE dfkv_disks_open_dropped_keys gauge\n";
  s += "dfkv_disks_open_dropped_keys" + braces("reason=\"failed_disk\"") + " " +
       std::to_string(group_.OpenEmptiedKeys()) + "\n";
  s += "dfkv_disks_open_dropped_keys" + braces("reason=\"not_home\"") + " " +
       std::to_string(group_.OpenMisplacedKeys()) + "\n";
  s += "dfkv_disks_open_dropped_keys" + braces("reason=\"unusable_record\"") + " " +
       std::to_string(group_.OpenUnusableKeys()) + "\n";
  s += "# HELP dfkv_disk_takeover_excluded 1 if a backing disk kept keys of other disks this start could not remove, and so takes no failed disk's keys in this run\n";
  s += "# TYPE dfkv_disk_takeover_excluded gauge\n";
  for (size_t i = 0; i < group_.DiskCount(); ++i)
    s += "dfkv_disk_takeover_excluded" + braces("disk=\"" + PromLabelEscape(group_.DiskPath(i)) + "\"") +
         (group_.TakesNoOtherDisksKeys(i) ? " 1\n" : " 0\n");
  s += "# HELP dfkv_disk_failed Whether a backing disk's store has failed for good\n";
  s += "# TYPE dfkv_disk_failed gauge\n";
  for (size_t i = 0; i < group_.DiskCount(); ++i)
    s += "dfkv_disk_failed" + braces("disk=\"" + PromLabelEscape(group_.DiskPath(i)) + "\"") +
         (group_.DiskHealthy(i) ? " 0\n" : " 1\n");
  s += "# HELP dfkv_disk_engine_info Storage engine of each backing disk (constant 1)\n";
  s += "# TYPE dfkv_disk_engine_info gauge\n";
  for (size_t i = 0; i < group_.DiskCount(); ++i)
    s += "dfkv_disk_engine_info" +
         braces("disk=\"" + PromLabelEscape(group_.DiskPath(i)) +
                "\",engine=\"" + PromLabelEscape(group_.DiskEngineName(i)) +
                "\"") +
         " 1\n";
  // Engine-specific per-disk families (the log engine's dfkv_lse_*): one
  // HELP/TYPE per name, a sample per disk that has it.
  {
    std::vector<std::string> order;
    std::map<std::string, std::pair<StoreEngine::EngineMetric,
                                    std::vector<std::pair<size_t, uint64_t>>>>
        families;
    for (size_t i = 0; i < group_.DiskCount(); ++i) {
      std::vector<StoreEngine::EngineMetric> em;
      group_.DiskEngineMetrics(i, &em);
      for (const auto& m : em) {
        auto it = families.find(m.name);
        if (it == families.end()) {
          order.push_back(m.name);
          it = families.emplace(m.name, std::make_pair(m, std::vector<std::pair<size_t, uint64_t>>{})).first;
        }
        it->second.second.emplace_back(i, m.value);
      }
    }
    for (const auto& name : order) {
      const auto& fam = families[name];
      s += "# HELP " + name + " " + fam.first.help + "\n";
      s += "# TYPE " + name + " " + fam.first.type + "\n";
      for (const auto& sample : fam.second)
        s += name +
             braces("disk=\"" + PromLabelEscape(group_.DiskPath(sample.first)) +
                    "\",disk_index=\"" + std::to_string(sample.first) + "\"") +
             " " + std::to_string(sample.second) + "\n";
    }
  }
  // sampled op latency histograms (op label merged with identity); one HELP/TYPE,
  // two label sets (get/put).
  {
    std::string get_lbl = std::string("op=\"get\"") + (idlabels.empty() ? "" : "," + idlabels);
    std::string put_lbl = std::string("op=\"put\"") + (idlabels.empty() ? "" : "," + idlabels);
    s += "# HELP dfkv_op_latency_seconds Sampled server op latency seconds\n";
    s += "# TYPE dfkv_op_latency_seconds histogram\n";
    std::string exist_lbl = std::string("op=\"exist\"") + (idlabels.empty() ? "" : "," + idlabels);
    s += get_lat_.RenderBody("dfkv_op_latency_seconds", get_lbl);
    s += put_lat_.RenderBody("dfkv_op_latency_seconds", put_lbl);
    s += exist_lat_.RenderBody("dfkv_op_latency_seconds", exist_lbl);
  }
  // Slab engine internals, emitted only for slab so a file-engine scrape is
  // unchanged. The dio_*_fallback counters are the "page cache silently crept
  // back in" alarm for direct-mode deployments.
  if (group_.HasSlabDisks()) {
    // Per-disk observations must not conflate independent high-water crossings.
    auto header = [&](const char* name, const char* type, const char* help) {
      s += "# HELP "; s += name; s += " "; s += help; s += "\n";
      s += "# TYPE "; s += name; s += " "; s += type; s += "\n";
    };
    header("dfkv_slab_disk_run_epoch", "gauge", "Dirty run epoch for this store instance");
    header("dfkv_slab_disk_capacity_bytes", "gauge", "Physical capacity for this store");
    header("dfkv_slab_disk_watermark_high_bytes", "gauge", "Resolved strict high threshold");
    header("dfkv_slab_disk_watermark_low_bytes", "gauge", "Resolved drain threshold");
    header("dfkv_slab_disk_reclaim_interval_ms", "gauge", "Resolved background reclaim cadence");
    header("dfkv_slab_disk_watermark_active", "gauge", "Current high-to-low drain latch");
    header("dfkv_slab_disk_watermark_high_observations_total", "counter", "Reclaimer samples directly satisfying used greater than high; not eviction counts");
    header("dfkv_slab_disk_watermark_last_high_bytes", "gauge", "Latest direct above-high predicate input before eviction");
    header("dfkv_slab_disk_watermark_peak_high_bytes", "gauge", "Lifetime maximum of sampled above-high inputs; never phase qualification alone");
    for (size_t i = 0; i < group_.DiskCount(); ++i) {
      // On a node whose disks run different engines, only slab disks have
      // these: no zero samples for the others.
      if (group_.DiskEngineName(i) != "slab") continue;
      const auto st = group_.DiskWatermarkObservation(i);
      const std::string labels =
          "disk=\"" + PromLabelEscape(group_.DiskPath(i)) +
          "\",disk_index=\"" + std::to_string(i) + "\"";
      const std::string disk_labels = braces(labels);
      auto sample = [&](const char* name, uint64_t value) {
        s += name; s += disk_labels; s += " ";
        s += std::to_string(value); s += "\n";
      };
      sample("dfkv_slab_disk_run_epoch", st.run_epoch);
      sample("dfkv_slab_disk_capacity_bytes", st.capacity_bytes);
      sample("dfkv_slab_disk_watermark_high_bytes", st.watermark_high_bytes);
      sample("dfkv_slab_disk_watermark_low_bytes", st.watermark_low_bytes);
      sample("dfkv_slab_disk_reclaim_interval_ms", st.reclaim_interval_ms);
      sample("dfkv_slab_disk_watermark_active", st.watermark_active);
      sample("dfkv_slab_disk_watermark_high_observations_total", st.watermark_high_observations);
      sample("dfkv_slab_disk_watermark_last_high_bytes", st.watermark_last_high_bytes);
      sample("dfkv_slab_disk_watermark_peak_high_bytes", st.watermark_peak_high_bytes);
    }
    const auto ss = group_.SlabStats();
    metric("dfkv_slab_dio_write_fallback_total", "counter",
           "Direct-mode PUTs that fell back to a buffered write", ss.dio_write_fallbacks);
    metric("dfkv_slab_dio_read_fallback_total", "counter",
           "Direct-mode aligned GETs that fell back to a buffered read", ss.dio_read_fallbacks);
    metric("dfkv_slab_put_no_slot_total", "counter",
           "Disk PUTs (client PUTs and RAM-tier flush attempts, retries "
           "included) refused with kCacheFull for want of a slot",
           ss.put_no_slot);
    metric("dfkv_slab_put_busy_total", "counter",
           "Disk PUTs refused with kCacheFull because the key could not be "
           "taken yet (a slot still held by an unfinished operation, or a "
           "deferred removal past its wait bound)",
           ss.put_busy);
    metric("dfkv_slab_table_sync_total", "counter",
           "slots.tbl fdatasync cycles performed", ss.table_syncs);
    metric("dfkv_slab_extent_steals_total", "counter",
           "Cross-class extent steals (class capacity churn)", ss.steals);
    metric("dfkv_slab_cold_steals_total", "counter",
           "Steals of globally-cold cross-class extents (full-ring self-thrash guard)",
           ss.cold_steals);
    metric("dfkv_slab_watermark_evictions_total", "counter",
           "Objects removed by proactive watermark reclamation",
           ss.watermark_evictions);
    metric("dfkv_slab_watermark_extent_clears_total", "counter",
           "Successful whole-extent metadata pwrite callbacks; NOT fdatasync durability",
           ss.watermark_extent_clears);
    metric("dfkv_slab_watermark_slot_removals_total", "counter",
           "Committed global allocator slot removals including last-resident extent callback cases",
           ss.watermark_slot_removals);
    metric("dfkv_slab_watermark_slot_record_clears_total", "counter",
           "Successful global partial 64-byte metadata pwrite tombstones; NOT fdatasync durability",
           ss.watermark_slot_record_clears);
    metric("dfkv_slab_watermark_record_clear_bytes_total", "counter",
           "Successful global partial and whole-extent metadata pwrite bytes; NOT durability or exclusive recycling cost",
           ss.watermark_record_clear_bytes);
    metric("dfkv_slab_watermark_freed_slot_bytes_total", "counter",
           "Allocated slot bytes actually removed by global selector calls",
           ss.watermark_freed_slot_bytes);
    metric("dfkv_slab_watermark_scan_visits_total", "counter",
           "Allocator global reclamation scan visits",
           ss.watermark_scan_visits);
    metric("dfkv_slab_watermark_max_victims_per_tick", "gauge",
           "Configured per-disk slot-removal cap per tick",
           ss.watermark_max_victims_per_tick);
    metric("dfkv_slab_watermark_max_visits_per_tick", "gauge",
           "Configured per-disk scan-visit cap per tick, four times slot-removal cap",
           ss.watermark_max_visits_per_tick);
    metric("dfkv_slab_watermark_ticks_total", "counter",
           "Bounded watermark eviction ticks", ss.watermark_ticks);
    metric("dfkv_slab_watermark_active", "gauge",
           "Whether high-water eviction remains latched until low watermark",
           ss.watermark_active);
    metric("dfkv_slab_watermark_last_tick_us", "gauge",
           "Duration of the latest watermark tick in microseconds",
           ss.watermark_last_tick_us);
    if (group_.SlabBalance()) {
      metric("dfkv_slab_balance_moves_total", "counter",
             "Extents moved between size classes by their tail ages "
             "(DFKV_SLAB_BALANCE)", ss.balance_moves);
      metric("dfkv_slab_balance_max_hold_us", "gauge",
             "Longest store-lock hold of a tick that moved extents between "
             "size classes", ss.balance_max_hold_us);
    }
    metric("dfkv_slab_watermark_max_tick_us", "gauge",
           "Maximum watermark tick duration in microseconds",
           ss.watermark_max_tick_us);
    metric("dfkv_slab_extent_returns_total", "counter",
           "Fully-free extents returned to the shared pool", ss.extent_returns);
    metric("dfkv_slab_deferred_removes_total", "counter",
           "Removes deferred behind in-flight slot I/O", ss.deferred_removes);
    metric("dfkv_slab_inflight_keys", "gauge",
           "Keys with an unlocked read/write in flight", ss.inflight);
    metric("dfkv_slab_prep_holds", "gauge",
           "Outstanding async-prep slot holds", ss.prep_holds);
    metric("dfkv_slab_reclaimed_total", "counter",
           "Slots freed ahead of demand by the background reclaimer", ss.reclaimed_slots);
    metric("dfkv_slab_batched_writes_total", "counter",
           "Payload writes that rode a batched store visit", ss.batched_writes);
    metric("dfkv_slab_uring_write_batches_total", "counter",
           "io_uring one-submit rounds on the batched write path", ss.uring_write_batches);
    metric("dfkv_slab_rebalanced_total", "counter",
           "Extents moved from cold classes to hot ones by the reclaimer", ss.rebalanced_extents);
    metric("dfkv_slab_bind_wipes_total", "counter",
           "Extent slot-grid wipes completed before publishing a new class",
           ss.bind_wipes);
    metric("dfkv_slab_payload_crc_failures_total", "counter",
           "Whole-value reads whose payload failed its CRC32C (served as a miss)",
           ss.payload_crc_failures);
    metric("dfkv_slab_metadata_io_errors_total", "counter",
           "Failed slots.tbl or epoch-state durability operations",
           ss.metadata_io_errors);
    metric("dfkv_slab_unclean_resets_total", "counter",
           "Dirty prior epochs discarded during cold recovery",
           ss.unclean_resets);
    metric("dfkv_slab_warm_restarts_total", "counter",
           "Dirty prior epochs whose table was kept (DFKV_SLAB_WARM_RESTART)",
           ss.warm_restarts);
    metric("dfkv_slab_trust_tokens_held", "gauge",
           "Slab stores holding their trust token: an unclean exit right now "
           "could keep their cache",
           ss.trust_tokens_held);
    metric("dfkv_slab_eviction_record_clears_total", "counter",
           "Slot records durably cleared before allocator reuse",
           ss.eviction_record_clears);
    metric("dfkv_slab_record_writes_total", "counter",
           "Committed slots.tbl record writes", ss.record_writes);
    metric("dfkv_slab_table_rebuilt_objects", "gauge",
           "Committed objects restored from slots.tbl at startup",
           ss.table_rebuilt);
    metric("dfkv_slab_rebuild_corrupt_records_total", "counter",
           "Non-empty malformed or checksum-invalid startup records",
           ss.rebuild_corrupt_records);
    metric("dfkv_slab_rebuild_rejected_records_total", "counter",
           "Valid records rejected and cleared for unsafe geometry or conflict",
           ss.rebuild_rejected_records);
    metric("dfkv_slab_rebuild_scanned_bytes", "gauge",
           "slots.tbl bytes read during the last startup rebuild",
           ss.rebuild_scanned_bytes);
    metric("dfkv_slab_rebuild_scan_chunks", "gauge",
           "Bounded chunks visited by the last startup rebuild",
           ss.rebuild_scan_chunks);
    metric("dfkv_slab_rebuild_sparse_ranges", "gauge",
           "Allocated sparse ranges visited by the last startup rebuild",
           ss.rebuild_sparse_ranges);
    metric("dfkv_slab_rebuild_sequential_fallbacks_total", "counter",
           "Startup rebuilds that fell back from sparse seek to sequential scan",
           ss.rebuild_sequential_fallbacks);
    metric("dfkv_slab_rebuild_mmap_scans", "gauge",
           "Slab tables mapped for the last startup rebuild",
           ss.rebuild_mmap_scans);
    metric("dfkv_slab_capacity_bytes", "gauge",
           "Configured physical slab payload capacity", ss.capacity_bytes);
    metric("dfkv_slab_allocated_bytes", "gauge",
           "Slot bytes occupied by allocator residents", ss.allocated_bytes);
    metric("dfkv_slab_payload_bytes", "gauge",
           "Logical committed payload bytes", ss.payload_bytes);
    metric("dfkv_slab_internal_fragmentation_bytes", "gauge",
           "Allocated slot bytes not carrying logical payload",
           ss.allocated_bytes >= ss.payload_bytes
               ? ss.allocated_bytes - ss.payload_bytes
               : 0);
    metric("dfkv_slab_allocator_objects", "gauge",
           "Allocator residents including uncommitted writes",
           ss.allocator_objects);
    metric("dfkv_slab_committed_objects", "gauge",
           "Reader-visible committed slab objects", ss.committed_objects);
    metric("dfkv_slab_classes", "gauge", "Active slab size classes",
           ss.class_count);
    metric("dfkv_slab_bound_extents", "gauge",
           "Extents assigned to active slab classes", ss.bound_extents);
    metric("dfkv_slab_pool_extents", "gauge",
           "Unbound extents available to new classes", ss.pool_extents);
    const auto classes = group_.SlabClasses();
    auto class_metric = [&](const char* name, const char* type,
                            const char* help, auto value) {
      s += "# HELP "; s += name; s += " "; s += help; s += "\n";
      s += "# TYPE "; s += name; s += " "; s += type; s += "\n";
      for (const auto& cls : classes) {
        const std::string labels =
            braces("slot_size=\"" + std::to_string(cls.slot_size) + "\"");
        s += name; s += labels; s += " ";
        s += std::to_string(value(cls)); s += "\n";
      }
    };
    class_metric("dfkv_slab_class_extents", "gauge",
                 "Bound extents in this live slab size class",
                 [](const auto& cls) { return cls.extents; });
    class_metric("dfkv_slab_class_resident_objects", "gauge",
                 "Resident objects in this live slab size class",
                 [](const auto& cls) { return cls.resident; });
    class_metric("dfkv_slab_class_free_slots", "gauge",
                 "Immediately allocatable slots in this live slab size class",
                 [](const auto& cls) { return cls.free_slots; });
    class_metric("dfkv_slab_class_allocated_bytes", "gauge",
                 "Occupied slot bytes in this live slab size class",
                 [](const auto& cls) { return cls.allocated_bytes; });
    class_metric("dfkv_slab_class_useful_bytes", "gauge",
                 "Caller payload bytes in this live slab size class",
                 [](const auto& cls) { return cls.useful_bytes; });
    class_metric("dfkv_slab_class_fragmentation_bytes", "gauge",
                 "Occupied slot bytes not carrying caller payload",
                 [](const auto& cls) {
                   return cls.allocated_bytes >= cls.useful_bytes
                              ? cls.allocated_bytes - cls.useful_bytes
                              : 0;
                 });
    if (group_.SlabBalance())
      class_metric("dfkv_slab_class_tail_age", "gauge",
                   "Allocator operations since the value this size class "
                   "would evict next was written or read (the oldest across "
                   "disks)",
                   [](const auto& cls) { return cls.tail_age; });
    class_metric("dfkv_slab_class_read_heat", "gauge",
                 "Operation-decayed read heat for this live slab size class",
                 [](const auto& cls) { return cls.read_heat; });
    class_metric("dfkv_slab_class_put_total", "counter",
                 "Cumulative puts admitted to this slab size class",
                 [](const auto& cls) { return cls.puts; });
    metric("dfkv_slab_failed_disks", "gauge",
           "Slab disks in fail-closed state", ss.failed_disks);
    metric("dfkv_slab_healthy", "gauge",
           "Whether all configured slab disks are healthy",
           ss.failed ? 0 : 1);
  }
  if (put_busy_limit_ > 0)
    metric("dfkv_put_busy_total", "counter",
           "PUTs rejected before RAM or disk admission (kCacheFull)",
           put_busy_.load(std::memory_order_relaxed));
  // RAM hot tier (P3), emitted only when enabled so a disk-only node's scrape is
  // unchanged. ram_hit/ram_miss let you compute the RAM hit rate; ram_put_bypass
  // is the flush-backpressure signal from the design's gap 10.3.
  if (ram_) {
    metric("dfkv_ram_hit_total", "counter", "GETs served from the RAM hot tier", ram_->Hits());
    metric("dfkv_ram_miss_total", "counter", "GETs that missed RAM (fell to disk)", ram_->Misses());
    metric("dfkv_ram_put_total", "counter", "PUTs accepted into the RAM hot tier", ram_->Puts());
    metric("dfkv_ram_put_bypass_total", "counter", "PUTs bypassing RAM (flush backpressure)", ram_->PutBypass());
    metric("dfkv_ram_promoted_total", "counter",
           "Coalesced cold reads promoted into RAM as durable residents", ram_->Promoted());
    metric("dfkv_ram_promote_declined_total", "counter",
           "Promotions the RAM tier did not take (no room, or a value it "
           "keeps out of RAM)", ram_->PromoteDeclined());
    if (recent_reads_) {  // DFKV_RAM_PROMOTE=second only
      metric("dfkv_ram_promote_deferred_total", "counter",
             "Whole cold reads not promoted into RAM because the key was "
             "not read within DFKV_RAM_PROMOTE_WINDOW_MS before",
             promote_deferred_.load(std::memory_order_relaxed));
      metric("dfkv_ram_recent_reads", "gauge",
             "Keys remembered as read within DFKV_RAM_PROMOTE_WINDOW_MS",
             recent_reads_->size());
    }
    metric("dfkv_ram_reservations_aborted_total", "counter",
           "Promotion reservations that ended without a copy installed",
           ram_->ReservationsAborted());
    metric("dfkv_ram_evicted_unread_total", "counter",
           "Values evicted from RAM for room without having been read from it",
           ram_->EvictedUnread());
    metric("dfkv_ram_evicted_age_ms_total", "counter",
           "Milliseconds evicted values had been resident, summed (divide by "
           "dfkv_ram_evictions_total for the mean)", ram_->EvictedAgeMs());
    metric("dfkv_ram_flushed_total", "counter", "RAM slots flushed to disk (now durable)", ram_->Flushed());
    metric("dfkv_ram_flush_dropped_total", "counter", "RAM slots dropped after flush failure", ram_->FlushDropped());
    metric("dfkv_ram_evictions_total", "counter", "RAM slots evicted under capacity pressure", ram_->Evictions());
    metric("dfkv_ram_reclaimed_total", "counter", "RAM slots freed ahead of demand by the background reclaimer", ram_->Reclaimed());
    metric("dfkv_ram_rebalanced_total", "counter", "RAM extents moved from cold classes to hot ones by the reclaimer", ram_->Rebalanced());
    metric("dfkv_ram_objects", "gauge", "Blocks currently resident in the RAM hot tier", ram_->Count());
    metric("dfkv_ram_flush_backlog", "gauge", "RAM slots queued or flushing (not yet durable)", ram_->FlushBacklog());
    metric("dfkv_ram_dirty_objects", "gauge",
           "RAM-resident objects not yet durable on disk", ram_->DirtyObjects());
    metric("dfkv_ram_dirty_bytes", "gauge",
           "Allocated bytes not yet durable on disk", ram_->DirtyBytes());
    metric("dfkv_ram_oldest_dirty_age_seconds", "gauge",
           "Age in seconds of the oldest non-durable RAM object",
           ram_->OldestDirtyAgeSeconds());
    metric("dfkv_ram_ack_total", "counter",
           "PUTs acknowledged from flush-pinned RAM before disk commit",
           ram_->RamAcks());
    metric("dfkv_ram_ack_backpressure_total", "counter",
           "RAM-ACK PUTs rejected at a positive dirty watermark or disk-ACKed at zero",
           ram_->AckBackpressure());
    metric("dfkv_ram_ack_backpressure_waits_total", "counter",
           "RAM-ACK PUTs that waited at the dirty watermark "
           "(DFKV_RAM_ACK_BACKPRESSURE_WAIT_MS)",
           ram_->BackpressureWaits());
    metric("dfkv_ram_ack_backpressure_wait_timeouts_total", "counter",
           "PUTs still refused when their wait at the dirty watermark ended",
           ram_->BackpressureWaitTimeouts());
    metric("dfkv_ram_ack_backpressure_wait_microseconds_total", "counter",
           "Time PUTs spent waiting at the dirty watermark, in microseconds",
           ram_->BackpressureWaitMicros());
    metric("dfkv_ram_parked_puts", "gauge",
           "PUTs waiting for the value that owns their key (disk-ACK PUTs waiting for its flush)",
           ram_->ParkedPuts());
    metric("dfkv_ram_put_wait_timeouts_total", "counter",
           "PUTs answered CacheFull once DFKV_RAM_PUT_WAIT_MS from the arrival of their request batch had passed",
           ram_->PutWaitTimeouts());
    metric("dfkv_ram_put_waits_canceled_total", "counter",
           "PUTs waiting for a flush that were answered CacheFull because the server is stopping",
           ram_->PutWaitsCanceled());
    metric("dfkv_ram_post_ack_flush_failures_total", "counter",
           "Acknowledged RAM PUTs whose value was dropped because its disk flush kept failing (lost)",
           ram_->PostAckFlushFailures());
    metric("dfkv_ram_flush_retries_total", "counter",
           "Failed RAM flush attempts that were followed by another attempt",
           ram_->FlushRetries());
    metric("dfkv_ram_flush_no_room_total", "counter",
           "RAM flush attempts the disk answered CacheFull (no slot for the "
           "value now); not failures",
           ram_->FlushNoRoom());
    metric("dfkv_ram_flush_no_room_dropped_total", "counter",
           "RAM values dropped because the disk had no room: disk-ACK PUTs "
           "answered CacheFull, acknowledged values after the hold time",
           ram_->FlushNoRoomDropped());
    metric("dfkv_ram_flush_refused_total", "counter",
           "RAM values dropped because the disk will never take them "
           "(Invalid: too large for it, or QuotaExceeded); not failures",
           ram_->FlushRefused());
    s += ram_->AckToDurableLatencyMetrics(idlabels);
    metric("dfkv_ram_budget_bytes", "gauge",
           "Hard total RAM-tier budget across arena and dedicated values",
           ram_->budget_bytes());
    metric("dfkv_ram_arena_bytes", "gauge",
           "Fixed arena bytes reserved from the total budget",
           ram_->arena_bytes());
    metric("dfkv_ram_arena_locked", "gauge",
           "1 if the arena is locked in memory and cannot be swapped out",
           ram_->arena_locked() ? 1 : 0);
    metric("dfkv_ram_arena_hugepage_bytes", "gauge",
           "Arena bytes backed by transparent huge pages after the pre-fault",
           ram_->arena_hugepage_bytes());
    metric("dfkv_ram_large_budget_bytes", "gauge",
           "Dedicated oversized-value reserve within the total budget",
           ram_->large_budget_bytes());
    metric("dfkv_ram_used_bytes", "gauge",
           "Resident slot plus dedicated allocation bytes",
           ram_->UsedBytes());
    metric("dfkv_ram_payload_bytes", "gauge",
           "Payload bytes of the resident values; dfkv_ram_used_bytes minus "
           "this is slot rounding",
           ram_->PayloadBytes());
    metric("dfkv_ram_large_used_bytes", "gauge",
           "Resident dedicated oversized allocation bytes",
           ram_->large_used_bytes());
    metric("dfkv_ram_healthy", "gauge",
           "0 from a dropped flush until flushes have succeeded again in a row",
           ram_->healthy() ? 1 : 0);
    metric("dfkv_ram_flush_threads", "gauge",
           "Actual RAM flusher count after the per-shard minimum adjustment",
           ram_->flusher_count());
    metric("dfkv_ram_shutdown_drain_timeouts_total", "counter",
           "Graceful shutdown drain attempts that exceeded their timeout",
           ram_->ShutdownDrainTimeouts());
    s += "# HELP dfkv_ram_flush_disk_batches_total RAM flush batches completed by disk\n";
    s += "# TYPE dfkv_ram_flush_disk_batches_total counter\n";
    s += "# HELP dfkv_ram_flush_disk_objects_total RAM flush objects submitted by disk\n";
    s += "# TYPE dfkv_ram_flush_disk_objects_total counter\n";
    s += "# HELP dfkv_ram_flush_disk_bytes_total RAM flush payload bytes submitted by disk\n";
    s += "# TYPE dfkv_ram_flush_disk_bytes_total counter\n";
    s += "# HELP dfkv_ram_flush_disk_failures_total RAM flush objects failed by disk\n";
    s += "# TYPE dfkv_ram_flush_disk_failures_total counter\n";
    s += "# HELP dfkv_ram_flush_disk_latency_seconds RAM flush engine batch latency seconds\n";
    s += "# TYPE dfkv_ram_flush_disk_latency_seconds histogram\n";
    for (size_t i = 0; i < ram_flush_disks_.size(); ++i) {
      const auto& m = *ram_flush_disks_[i];
      const std::string disk_labels =
          "disk=\"" + PromLabelEscape(group_.DiskPath(i)) + "\",disk_index=\"" +
          std::to_string(i) + "\"" +
          (idlabels.empty() ? "" : "," + idlabels);
      const std::string disk_braces = "{" + disk_labels + "}";
      s += "dfkv_ram_flush_disk_batches_total" + disk_braces + " " +
           std::to_string(m.batches.load(std::memory_order_relaxed)) + "\n";
      s += "dfkv_ram_flush_disk_objects_total" + disk_braces + " " +
           std::to_string(m.objects.load(std::memory_order_relaxed)) + "\n";
      s += "dfkv_ram_flush_disk_bytes_total" + disk_braces + " " +
           std::to_string(m.bytes.load(std::memory_order_relaxed)) + "\n";
      s += "dfkv_ram_flush_disk_failures_total" + disk_braces + " " +
           std::to_string(m.failures.load(std::memory_order_relaxed)) + "\n";
      s += m.latency.RenderBody("dfkv_ram_flush_disk_latency_seconds",
                                disk_labels);
    }
  }
  return s;
}

void KvNodeServer::set_drain_handler(std::function<Status(bool)> handler) {
  std::lock_guard<std::mutex> lk(metrics_extension_mu_);
  drain_handler_ = std::move(handler);
}

void KvNodeServer::set_metrics_extension(
    std::function<std::string()> render) {
  std::lock_guard<std::mutex> lk(metrics_extension_mu_);
  metrics_extension_ = std::move(render);
}

std::string KvNodeServer::FullMetricsText() const {
  std::function<std::string()> render;
  {
    std::lock_guard<std::mutex> lk(metrics_extension_mu_);
    render = metrics_extension_;
  }
  std::string text = MetricsText();
  if (render) text += render();
  return text;
}

void KvNodeServer::AcceptLoop() {
  while (running_) {
    int fd = ::accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) {
      const int err = errno;
      if (!running_) break;
      accept_guard_.OnAcceptError(err, running_);
      continue;
    }
    // The first-frame deadline is anchored at ACCEPT time: queuing delay
    // (handler-thread spawn) counts against the peer's budget, not ours.
    const auto accepted_at = std::chrono::steady_clock::now();
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    timeval timeout{tcp_io_timeout_seconds_, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    accept_count_.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(conn_mu_);
    if (!running_) { ::close(fd); break; }
    ReapDoneLocked();
    if (conns_.size() >= tcp_max_connections_) {
      tcp_rejected_connections_.fetch_add(1, std::memory_order_relaxed);
      ::close(fd);
      continue;
    }
    open_connections_.fetch_add(1, std::memory_order_relaxed);
    auto done = std::make_shared<std::atomic<bool>>(false);
    std::thread handling;
    if (!net::StartThread(&handling, [this, fd, done, accepted_at] {
          NameThisThread("kv-serve");
          Handle(fd, accepted_at);
          open_connections_.fetch_sub(1, std::memory_order_relaxed);
          // Taken only after the accept loop, which holds conn_mu_ across the
          // thread start, has recorded this fd.
          { std::lock_guard<std::mutex> lk(conn_mu_); conn_fds_.erase(fd); }
          ::close(fd);
          done->store(true, std::memory_order_release);  // last act
        })) {
      open_connections_.fetch_sub(1, std::memory_order_relaxed);
      accept_guard_.OnThreadStartFailure();
      ::close(fd);
      continue;
    }
    conn_fds_.insert(fd);
    conns_.push_back({std::move(handling), done});
  }
}

// Transport-agnostic request processing + metrics (shared by TCP and RDMA).
Status KvNodeServer::LookupForKey(const BlockKey& key,
                                  ValueMetadata* out) const {
  if (out == nullptr) return Status::kInvalid;
  size_t ram_len = 0;
  if (ram_ && ram_->Lookup(key, &ram_len)) {
    out->value_len = ram_len;
    return Status::kOk;
  }
  return group_.Lookup(key, out);
}


Status KvNodeServer::LookupForGet(const BlockKey& key) {
  ValueMetadata metadata;
  const Status st = LookupForKey(key, &metadata);
  if (st == Status::kNotFound) {
    cache_miss_.fetch_add(1, std::memory_order_relaxed);
    if (ram_) ram_->NoteMiss();
  }
  return st;
}

Status KvNodeServer::ProcessRequestForKey(
    uint8_t op_raw, const BlockKey& key, uint64_t offset, uint64_t length,
    const char* payload, uint64_t payload_len, std::string* out_data,
    size_t* out_value_len) {
  if (out_value_len) *out_value_len = 0;
  WireOp op = static_cast<WireOp>(op_raw);
  Status st = Status::kInvalid;
  switch (op) {
    case WireOp::kCache: {
      if (payload_len == 0 || payload == nullptr) {
        st = Status::kInvalid;
        break;
      }
      // Reject only before any RAM value or disk write belongs to this PUT.
      // An admitted conflicting duplicate must never fall back on a deadline.
      if (!TryBeginPut()) {
        st = Status::kCacheFull;
        break;
      }
      bool samp = lat_sampler_.ShouldSample();
      double t0 = samp ? NowSec() : 0.0;
      std::unique_lock<std::mutex> publication_lock;
      if (write_memory_) publication_lock = MemoryKey(key);
      auto key_lock = LockPutKey(key);
      // RAM-ACK publishes a flush-pinned copy. A reached positive watermark
      // rejects new keys before admission; only a capacity bypass may take
      // the normal disk path, never a terminal watermark rejection.
      // Nor may RAM acknowledge a value that can never reach its disk: once
      // that disk's store has failed for good the PUT goes the disk path and
      // gets the store's I/O error, as it would without a RAM tier; and a
      // value larger than the disks can store gets their kInvalid.
      bool needs_disk = true;
      if (ram_write_back_ && ram_ &&
          group_.TenantQuotaBytes(key.tenant_hash) == 0 &&
          payload_len <= group_.MaxValueBytes() &&
          !group_.DiskFailedFor(key)) {
        // The RAM tier releases key_lock before it waits for a flush.
        const RamTier::PutResult result =
            ram_ack_enabled_
                ? ram_->PutWriteBack(key, payload, payload_len, &key_lock.lock)
                : ram_->PutCommitted(key, payload, payload_len, &key_lock.lock);
        st = result.status;
        needs_disk = result.disposition == RamTier::PutDisposition::kBypass;
        if (st == Status::kOk) {
          cache_put_.fetch_add(1, std::memory_order_relaxed);
          bytes_written_.fetch_add(payload_len, std::memory_order_relaxed);
        } else if (!needs_disk && st == Status::kIOError) {
          put_io_err_.fetch_add(1, std::memory_order_relaxed);
        }
      }
      if (needs_disk) {
        st = group_.Cache(key, payload, payload_len);
        if (st == Status::kOk) {
          cache_put_.fetch_add(1, std::memory_order_relaxed);
          bytes_written_.fetch_add(payload_len, std::memory_order_relaxed);
        } else if (st == Status::kIOError) {
          put_io_err_.fetch_add(1, std::memory_order_relaxed);
        } else if (st == Status::kInvalid) {
          invalid_ops_.fetch_add(1, std::memory_order_relaxed);
        }
      }
      if (st == Status::kOk && write_memory_) NoteWrite(key);
      if (samp) put_lat_.Observe(NowSec() - t0);
      EndPut();
      break;
    }
    case WireOp::kRange: {
      bool samp = lat_sampler_.ShouldSample();
      double t0 = samp ? NowSec() : 0.0;
      bool ram_served = false;
      if (ram_) {  // RAM hot tier: serve straight from the arena (copy out), no disk
        RamTier::Hit h;
        if (ram_->GetPrep(key, offset, length, &h)) {
          out_data->assign(h.ptr, h.len);
          if (out_value_len) *out_value_len = h.value_len;
          st = Status::kOk;
          cache_hit_.fetch_add(1, std::memory_order_relaxed);
          bytes_read_.fetch_add(out_data->size(), std::memory_order_relaxed);
          ram_served = true;
        }
      }
      if (!ram_served) {
        // length==0 stays on the plain path: its scratch would have to be
        // sized at the max-value worst case (a 64 MiB zeroed allocation per
        // request), which costs more than the duplicate read it would save.
        if (coalesce_enabled_ && length > 0) {
          // TCP convoy collapse: leader reads via group_.Range into a scratch
          // string sized by the store; followers copy the shared bytes.
          //
          // The client-declared length is UNTRUSTED: sizing the scratch
          // straight from it lets one forged kRange frame drive an unbounded
          // allocation (length_error/bad_alloc -> std::terminate, or OOM).
          // Look up the authoritative stored length first -- an unknown key
          // must fail here exactly like the plain path, before ANY sized
          // allocation happens.
          ValueMetadata metadata;
          st = LookupForKey(key, &metadata);
          if (st == Status::kOk) {
            const uint64_t max_payload = max_request_payload_
                                             ? max_request_payload_
                                             : wire_limits::MaxRequestPayload();
            const uint64_t avail = offset < metadata.value_len
                                       ? metadata.value_len - offset
                                       : 0;
            // cap is an ALLOCATION bound only: the fill still issues
            // group_.Range with the original (offset, length), so store-side
            // Range semantics (offset >= value_len, data clamped to the
            // remainder) are unchanged; a concurrent same-key rewrite stays
            // the store-level race it already was.
            const size_t cap =
                static_cast<size_t>(std::min({length, avail, max_payload}));
            out_data->resize(cap);
            size_t n = 0;
            st = read_coalescer_.Read(
                key, offset, length,
                out_data->empty() ? nullptr : &(*out_data)[0], cap, &n,
                out_value_len,
                [&](char* buf, size_t bcap, size_t* on, size_t* value_len) {
                  std::string tmp;
                  Status s =
                      group_.Range(key, offset, length, &tmp, value_len);
                  if (s == Status::kOk) {
                    size_t m = tmp.size() < bcap ? tmp.size() : bcap;
                    // The coalescer hands the caller's own buffer to the
                    // fill, which is null with cap 0 when the value has no
                    // bytes at this offset (memcpy of a null pointer is UB
                    // even for 0 bytes).
                    if (m) std::memcpy(buf, tmp.data(), m);
                    *on = m;
                  }
                  return s;
                });
            out_data->resize(st == Status::kOk ? n : 0);
          }
        } else {
          st = group_.Range(key, offset, length, out_data, out_value_len);
        }
        if (st == Status::kOk) {
          cache_hit_.fetch_add(1, std::memory_order_relaxed);
          bytes_read_.fetch_add(out_data->size(), std::memory_order_relaxed);
        } else if (st == Status::kNotFound) {
          cache_miss_.fetch_add(1, std::memory_order_relaxed);
        } else if (st == Status::kIOError) {
          get_io_err_.fetch_add(1, std::memory_order_relaxed);
        }
      }
      if (samp) get_lat_.Observe(NowSec() - t0);
      break;
    }
    case WireOp::kExist: {
      bool samp = lat_sampler_.ShouldSample();
      double t0 = samp ? NowSec() : 0.0;
      ValueMetadata metadata;
      st = LookupForKey(key, &metadata);
      if (st == Status::kOk) {
        if (out_value_len)
          *out_value_len = static_cast<size_t>(metadata.value_len);
        exist_hit_.fetch_add(1, std::memory_order_relaxed);
      } else if (st == Status::kNotFound) {
        exist_miss_.fetch_add(1, std::memory_order_relaxed);
      }
      // Handler-body latency includes the RAM shard and disk cached-set locks.
      // A reclaim or flush can hold either lock, so this tail is intentionally
      // a lock-contention signal rather than only a transport measurement.
      if (samp) exist_lat_.Observe(NowSec() - t0);
      break;
    }
    case WireOp::kRetain:
      st = ProcessRetain(key, length, payload, payload_len, out_data);
      break;
    case WireOp::kRemovedWithin:
      st = ProcessRemovedWithin(key, length, payload, payload_len, out_data);
      break;
    case WireOp::kRemovedWithinAges:
      st = ProcessRemovedWithinAges(key, length, payload, payload_len,
                                    out_data);
      break;
    case WireOp::kLookup: {
      ValueMetadata metadata;
      st = LookupForKey(key, &metadata);
      if (st == Status::kOk && out_value_len)
        *out_value_len = static_cast<size_t>(metadata.value_len);
      break;
    }
    case WireOp::kRemove: {
      // RamTier hides the key first and waits out/cancels any flush. The disk
      // remove therefore compensates a flush that reached durable storage just
      // before the fence; after it returns no later RAM worker can republish.
      // With W, this per-key lock covers storage as well as publication, so
      // the PUT/REMOVE completion order and the op-26 view agree.
      std::unique_lock<std::mutex> write_key_lock;
      if (write_memory_) write_key_lock = MemoryKey(key);
      // A parked RETAIN SET goes first: forgotten after the RAM remove, it
      // could be one made for the next value of the key.
      if (ram_) ApplyPendingRetain(key, /*flushed=*/false);
      const bool ram_had = ram_ && ram_->Remove(key);
      const Status disk_status = group_.Remove(key);
      // A read of the key that was in flight may have promoted it into RAM
      // between the two removes; a promotion after the disk remove is not
      // installed (it checks the disk), one before it is dropped here.
      if (ram_) ram_->DropDurableCopy(key);
      // Ordered with W by the per-key lock, before the acknowledgement.
      if (remove_memory_ &&
          (disk_status == Status::kOk || disk_status == Status::kNotFound))
        remove_memory_->Record(key, RawMs());
      if (disk_status == Status::kOk ||
          (disk_status == Status::kNotFound && ram_had)) {
        st = Status::kOk;
        remove_ok_.fetch_add(1, std::memory_order_relaxed);
      } else if (disk_status == Status::kNotFound) {
        st = Status::kNotFound;
        remove_miss_.fetch_add(1, std::memory_order_relaxed);
      } else {
        st = disk_status;
      }
      break;
    }
    case WireOp::kStats:
      *out_data = FullMetricsText();
      st = Status::kOk;
      break;
    case WireOp::kMembers: {
      // advertised cluster membership (for client discovery)
      std::lock_guard<std::mutex> lock(members_mu_);
      *out_data = members_;
      st = Status::kOk;
      break;
    }
    case WireOp::kDrain: {
      std::function<Status(bool)> handler;
      {
        std::lock_guard<std::mutex> lk(metrics_extension_mu_);
        handler = drain_handler_;
      }
      if (handler && payload_len == 1 && payload != nullptr)
        st = handler(payload[0] != 0);
      break;
    }
    case WireOp::kDeregister:
    case WireOp::kRegister:
    case WireOp::kHeartbeat:
    case WireOp::kListMembers:
    case WireOp::kListGroups:
    case WireOp::kClientRegister:
    case WireOp::kClientHeartbeat:
    case WireOp::kListClients:
    case WireOp::kListTopology:
    case WireOp::kPullRange:
    case WireOp::kPullRelease:
    case WireOp::kLeasePut:
      st = Status::kInvalid;  // MDS/RDMA-only ops are not served by a cache node
      break;
    }
  if (st == Status::kInvalid) invalid_ops_.fetch_add(1, std::memory_order_relaxed);
  return st;
}


namespace {
uint32_t RetainWallNowS() {
  return static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::seconds>(
                                   std::chrono::system_clock::now()
                                       .time_since_epoch())
                                   .count());
}
}  // namespace

// kRemovedWithin (op 25). A node without the memory answers as an older one
// does (an unknown op).
Status KvNodeServer::ProcessRemovedWithin(const BlockKey& prefix_key,
                                          uint64_t count, const char* payload,
                                          uint64_t payload_len,
                                          std::string* out_data) {
  if (out_data == nullptr || !remove_memory_) return Status::kInvalid;
  if (count == 0 || count > kRemovedWithinMaxKeys || payload == nullptr ||
      payload_len != kRemovedWithinHeaderBytes + count * 16 ||
      net::GetU32(payload) != count)
    return Status::kInvalid;
  const uint64_t age_ms = net::GetU32(payload + 4);
  out_data->assign(4 + count, '\0');
  const uint64_t covered = remove_memory_->Query(
      prefix_key.tenant_hash, payload + kRemovedWithinHeaderBytes,
      static_cast<size_t>(count), age_ms, RawMs(),
      reinterpret_cast<uint8_t*>(&(*out_data)[4]));
  net::PutU32(&(*out_data)[0],
              static_cast<uint32_t>(std::min<uint64_t>(covered, UINT32_MAX)));
  return Status::kOk;
}

// kRemovedWithinAges (op 26, CLI-35). Both ages from one clock reading and
// one publication snapshot, so P's own REMOVE and PUT compare exactly. W's
// pause check and expiry run first; without W every key has no recent write.
void KvNodeServer::AfterRemovedAgesForTest() {
  if (!pause_ages_for_test_.load(std::memory_order_relaxed)) return;
  std::unique_lock<std::mutex> lk(ages_test_mu_);
  ++ages_test_state_;
  ages_test_cv_.notify_all();
  ages_test_cv_.wait(lk, [this] { return ages_test_state_ % 2 == 0; });
}
Status KvNodeServer::ProcessRemovedWithinAges(const BlockKey& prefix_key,
                                              uint64_t count,
                                              const char* payload,
                                              uint64_t payload_len,
                                              std::string* out_data) {
  if (out_data == nullptr || !remove_memory_) return Status::kInvalid;
  if (count == 0 || count > kRemovedWithinMaxKeys || payload == nullptr ||
      payload_len != kRemovedWithinHeaderBytes + count * 16 ||
      net::GetU32(payload) != count)
    return Status::kInvalid;
  removed_within_ages_requests_.fetch_add(1, std::memory_order_relaxed);
  const uint64_t age_ms = net::GetU32(payload + 4);
  const char* digests = payload + kRemovedWithinHeaderBytes;
  const size_t n = static_cast<size_t>(count);
  std::vector<uint32_t> removed(n, kRemovedAgeNone), written(n, kRemovedAgeNone);
  uint64_t covered = 0;
  {
    // With W, reject an unfinished REMOVE as well as an in-flight PUT of the
    // same key (without an unrelated-key lock). Locks are ordered and
    // deduplicated by their hashed mutex bucket, not only by full key: two
    // different keys may share one of the 64 mutexes.
    std::vector<BlockKey> keys;
    keys.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      const char* d = digests + i * 16;
      keys.push_back({net::GetU64(d), net::GetU64(d + 8),
                     prefix_key.tenant_hash});
    }
    std::sort(keys.begin(), keys.end(), [this](const BlockKey& a, const BlockKey& b) {
      return BlockKeyMixHash{}(a) % memory_key_locks_.size() <
             BlockKeyMixHash{}(b) % memory_key_locks_.size();
    });
    keys.erase(std::unique(keys.begin(), keys.end(),
                           [this](const BlockKey& a, const BlockKey& b) {
                             return BlockKeyMixHash{}(a) %
                                        memory_key_locks_.size() ==
                                    BlockKeyMixHash{}(b) %
                                        memory_key_locks_.size();
                           }), keys.end());
    std::vector<std::unique_lock<std::mutex>> held;
    held.reserve(keys.size());
    for (const BlockKey& key : keys)
      if (write_memory_) held.push_back(MemoryKey(key));
    const uint64_t now = RawMs();
    AfterRemovedAgesForTest();
    if (write_memory_) {
      covered = write_memory_->AgesWithRemovals(
          *remove_memory_, prefix_key.tenant_hash, digests, n, age_ms, now,
          removed.data(), written.data());
    } else {
      covered = remove_memory_->QueryAges(prefix_key.tenant_hash, digests, n,
                                          age_ms, now, removed.data());
    }
  }
  out_data->assign(4 + n * kRemovedWithinAgesKeyBytes, '\0');
  net::PutU32(&(*out_data)[0],
              static_cast<uint32_t>(std::min<uint64_t>(covered, UINT32_MAX)));
  for (size_t i = 0; i < n; ++i) {
    net::PutU32(&(*out_data)[4 + i * 8], removed[i]);
    net::PutU32(&(*out_data)[8 + i * 8], written[i]);
  }
  return Status::kOk;
}

Status KvNodeServer::ProcessRetain(const BlockKey& prefix_key, uint64_t count,
                                   const char* payload, uint64_t payload_len,
                                   std::string* out_data) {
  // Behave exactly like an older node (unknown op) when retention is off.
  if (out_data == nullptr || !group_.RetainEnabled()) return Status::kInvalid;
  if (count == 0 || count > kRetainMaxKeys || payload == nullptr ||
      payload_len != RetainPayloadBytes(static_cast<size_t>(count)))
    return Status::kInvalid;
  const uint32_t ttl_s = net::GetU32(payload);
  const uint8_t mode = static_cast<uint8_t>(payload[4]);
  if (mode > kRetainModeRefresh) return Status::kInvalid;
  retain_requests_.fetch_add(1, std::memory_order_relaxed);
  const uint32_t now = RetainWallNowS();
  const uint32_t max_ttl = group_.RetainMaxTtl();
  out_data->assign(static_cast<size_t>(count) + kRetainTrailerBytes, '\0');
  const char* k = payload + kRetainHeaderBytes;
  for (uint64_t i = 0; i < count; ++i, k += kRetainKeyBytes) {
    const BlockKey key{net::GetU64(k), net::GetU64(k + 8),
                       prefix_key.tenant_hash};
    uint8_t code = group_.Retain(key, ttl_s, mode, now);
    if (code == StoreEngine::kRetainUnsupported) return Status::kInvalid;
    // A miss on disk with the value in RAM: a hit. If the value is still to
    // be written, a SET is parked and applied when the flush commits the disk
    // slot. A written RAM copy of a key the disk no longer has can be
    // protected by nothing, and no flush will come for it: unprotected, not
    // parked. Its write may have landed since the disk was asked (it is no
    // longer dirty, or it landed just as the SET was parked): then the disk
    // is asked again.
    size_t ram_len = 0;
    bool dirty = false;
    if (code == static_cast<uint8_t>(SlabAllocator::RetainCode::kMiss) && ram_ &&
        ram_->Lookup(key, &ram_len, &dirty)) {
      code = static_cast<uint8_t>(SlabAllocator::RetainCode::kHitUnprotected);
      if (!dirty) {
        // Asked again only if the write has landed, so a copy the disk does
        // not have is not counted twice in the store's retention results.
        if (group_.IsCached(key)) code = group_.Retain(key, ttl_s, mode, now);
      } else if (mode == kRetainModeSet && ttl_s != 0) {
        {
          std::lock_guard<std::mutex> lk(pending_retain_mu_);
          auto it = pending_retain_.find(key);
          if (it != pending_retain_.end()) {
            it->second = std::max(it->second, std::min(ttl_s, max_ttl));
            code =
                static_cast<uint8_t>(SlabAllocator::RetainCode::kProtectedNew);
          } else if (pending_retain_.size() < kPendingRetainMax) {
            pending_retain_.emplace(key, std::min(ttl_s, max_ttl));
            code =
                static_cast<uint8_t>(SlabAllocator::RetainCode::kProtectedNew);
          } else {
            code = static_cast<uint8_t>(SlabAllocator::RetainCode::kDenied);
          }
        }
        // Landed between the disk's miss and the park: no flush will apply
        // the parked SET any more. Take it back and apply it here, unless the
        // flush took it first (then it was applied there).
        if (code ==
                static_cast<uint8_t>(SlabAllocator::RetainCode::kProtectedNew) &&
            group_.IsCached(key)) {
          const int applied = ApplyPendingRetain(key, /*flushed=*/true);
          if (applied >= 0) code = static_cast<uint8_t>(applied);
        }
      }
    }
    (*out_data)[static_cast<size_t>(i)] = static_cast<char>(code);
    retain_keys_[mode][code < 8 ? code : 7].fetch_add(
        1, std::memory_order_relaxed);
  }
  RetainTrailer trailer;
  trailer.server_now_s = now;
  trailer.max_ttl_s = max_ttl;
  trailer.headroom_pct = static_cast<uint8_t>(
      std::min<uint32_t>(group_.RetainHeadroomPct(), 100));
  EncodeRetainTrailer(&(*out_data)[static_cast<size_t>(count)], trailer);
  return Status::kOk;
}

int KvNodeServer::ApplyPendingRetain(const BlockKey& key, bool flushed) {
  uint32_t ttl_s = 0;
  {
    std::lock_guard<std::mutex> lk(pending_retain_mu_);
    if (pending_retain_.empty()) return -1;
    auto it = pending_retain_.find(key);
    if (it == pending_retain_.end()) return -1;
    ttl_s = it->second;
    pending_retain_.erase(it);
  }
  if (!flushed) return -1;
  const uint8_t code =
      group_.Retain(key, ttl_s, kRetainModeSet, RetainWallNowS());
  pending_retain_applied_.fetch_add(1, std::memory_order_relaxed);
  if (code == static_cast<uint8_t>(SlabAllocator::RetainCode::kDenied))
    pending_retain_denied_.fetch_add(1, std::memory_order_relaxed);
  return code;
}

Status KvNodeServer::RangeIntoForKey(const BlockKey& key, uint64_t offset,
                                     uint64_t length, char* dst,
                                     size_t dst_cap, size_t* out_len,
                                     size_t* value_len) {
  bool samp = lat_sampler_.ShouldSample();
  double t0 = samp ? NowSec() : 0.0;
  if (ram_) {
    RamTier::Hit h;
    if (ram_->GetPrep(key, offset, length, &h)) {  // RAM hit: copy out, no disk
      size_t n = h.len < dst_cap ? h.len : dst_cap;
      std::memcpy(dst, h.ptr, n);
      if (out_len) *out_len = n;
      if (value_len) *value_len = h.value_len;
      cache_hit_.fetch_add(1, std::memory_order_relaxed);
      bytes_read_.fetch_add(n, std::memory_order_relaxed);
      if (samp) get_lat_.Observe(NowSec() - t0);
      return Status::kOk;
    }
  }
  Status st;
  if (coalesce_enabled_) {
    st = read_coalescer_.Read(
        key, offset, length, dst, dst_cap, out_len, value_len,
        [&](char* buf, size_t cap, size_t* n, size_t* stored_len) {
          return group_.RangeInto(key, offset, length, buf, cap, n,
                                  stored_len);
        });
  } else {
    st = group_.RangeInto(key, offset, length, dst, dst_cap, out_len,
                          value_len);
  }
  if (st == Status::kOk) {
    cache_hit_.fetch_add(1, std::memory_order_relaxed);
    bytes_read_.fetch_add(*out_len, std::memory_order_relaxed);
  } else if (st == Status::kNotFound) {
    cache_miss_.fetch_add(1, std::memory_order_relaxed);
  } else if (st == Status::kIOError) {
    get_io_err_.fetch_add(1, std::memory_order_relaxed);
  }
  if (samp) get_lat_.Observe(NowSec() - t0);
  return st;
}


Status KvNodeServer::CacheDirectForKey(
    const BlockKey& key, char* data, size_t len, size_t cap,
    std::chrono::steady_clock::time_point batch_arrival) {
  if (len == 0 || data == nullptr) {
    invalid_ops_.fetch_add(1, std::memory_order_relaxed);
    return Status::kInvalid;
  }
  if (!TryBeginPut()) return Status::kCacheFull;
  bool samp = lat_sampler_.ShouldSample();
  double t0 = samp ? NowSec() : 0.0;
  std::unique_lock<std::mutex> publication_lock;
  if (write_memory_) publication_lock = MemoryKey(key);
  auto key_lock = LockPutKey(key);
  // PUT write-around: write directly to disk, skip the synchronous 1 MiB
  // arena memcpy. The arena is filled lazily by read-promotion on GET miss
  // (RangeDirectForKey), so hot keys still get zero-copy RDMA GET from the
  // arena without paying a per-PUT copy on the write path. The old path
  // (ram_->PutCommitted) copied every PUT into the arena and flushed
  // asynchronously — the 1 MiB memcpy was 44% of serve-thread CPU and
  // capped single-node PUT at 3.4 GB/s vs 4.4 GB/s direct-disk.
  Status st = Status::kCacheFull;
  // A tenant with a node quota never takes RAM write-back admission: its PUT
  // result must reflect the quota-checked durable write. It still has to reach
  // that write -- the disk group owns quota admission for every caller.
  const bool unlimited = group_.TenantQuotaBytes(key.tenant_hash) == 0;
  bool needs_disk = true;
  // Write-back admits to the RAM arena first. RAM-ACK returns after the
  // flush-pinned value is visible; disk-ACK waits for the same flush result.
  // Only capacity bypass falls through; watermark rejection is terminal.
  // A disk whose store has failed for good takes no RAM admission either: the
  // value could never be flushed, so the PUT gets the store's I/O error. Nor
  // does a value larger than the disks can store: it gets their kInvalid.
  if (ram_write_back_ && ram_ && unlimited && len <= group_.MaxValueBytes() &&
      !group_.DiskFailedFor(key)) {
    // The RAM tier releases key_lock before it waits for a flush.
    const RamTier::PutResult result =
        ram_ack_enabled_
            ? ram_->PutWriteBack(key, data, len, &key_lock.lock, batch_arrival)
            : ram_->PutCommitted(key, data, len, &key_lock.lock, batch_arrival);
    st = result.status;
    needs_disk = result.disposition == RamTier::PutDisposition::kBypass;
    if (st == Status::kOk) {
      cache_put_.fetch_add(1, std::memory_order_relaxed);
      bytes_written_.fetch_add(len, std::memory_order_relaxed);
    } else if (!needs_disk && st == Status::kIOError) {
      put_io_err_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  if (needs_disk) {
    st = group_.CacheDirect(key, data, len, cap);
    if (st == Status::kOk) {
      cache_put_.fetch_add(1, std::memory_order_relaxed);
      bytes_written_.fetch_add(len, std::memory_order_relaxed);
    } else if (st == Status::kIOError) {
      put_io_err_.fetch_add(1, std::memory_order_relaxed);
    } else if (st == Status::kInvalid) {
      invalid_ops_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  if (st == Status::kOk) NoteWrite(key);
  if (samp) put_lat_.Observe(NowSec() - t0);
  EndPut();
  return st;
}


bool KvNodeServer::RamPinnedHitForKey(const BlockKey& key, uint64_t offset,
                                      uint64_t length, PreparedRead* out) {
  if (out == nullptr) return false;
  *out = PreparedRead{};
  RamTier::Hit hit;
  if (!ram_ || !ram_->GetPrep(key, offset, length, &hit,
                              /*count_access=*/false))
    return false;
  if (!hit.in_arena || hit.len == 0) return false;
  *out = PreparedRead::Ready(hit.ptr, hit.len, hit.value_len, true, nullptr,
                            0, this, hit.TransferToken(), &FinishPullRamRead,
                            hit.allocated_bytes);
  return true;
}

uint64_t KvNodeServer::DiskGeneration(const BlockKey& key) const {
  ValueMetadata metadata;
  if (group_.Lookup(key, &metadata) != Status::kOk) return 0;
  // An engine that does not tell the generation still tells presence.
  return metadata.generation != 0 ? metadata.generation : 1;
}

Status KvNodeServer::RangeDirectForKey(
    const BlockKey& key, uint64_t offset, uint64_t length, char* io_buf,
    size_t io_cap, const char** out_data, size_t* out_len, size_t* value_len) {
  bool samp = lat_sampler_.ShouldSample();
  double t0 = samp ? NowSec() : 0.0;
  if (ram_) {
    RamTier::Hit h;
    if (ram_->GetPrep(key, offset, length, &h)) {
      // RAM hit: copy the value into the caller's registered io_buf so the RDMA
      // layer scatter-sends it from there (no disk, no open, no pread). True
      // zero-copy (send straight from the arena MR) is B5-3.
      size_t n = h.len < io_cap ? h.len : io_cap;
      std::memcpy(io_buf, h.ptr, n);
      if (out_data) *out_data = io_buf;
      if (out_len) *out_len = n;
      if (value_len) *value_len = h.value_len;
      cache_hit_.fetch_add(1, std::memory_order_relaxed);
      bytes_read_.fetch_add(n, std::memory_order_relaxed);
      if (samp) get_lat_.Observe(NowSec() - t0);
      return Status::kOk;
    }
  }
  // Promotion needs the same read's full length even when the caller omits it,
  // and which disk value it reads, taken before the read.
  size_t stored_len = 0;
  if (value_len == nullptr) value_len = &stored_len;
  const uint64_t generation = ram_ ? DiskGeneration(key) : 0;
  Status st;
  // Only bytes this call read itself may be promoted under its generation: a
  // follower of a coalesced read gets the leader's bytes, which may be of an
  // older value than the one the follower's generation names (the key was
  // removed and written again during the leader's read). The leader promotes.
  bool own_read = true;
  if (coalesce_enabled_) {
    // Leader reads into the shared scratch, every rank's convoy copy lands in
    // its own registered io_buf; *out_data must point at io_buf either way.
    ReadCoalescer::Outcome outcome;
    st = read_coalescer_.Read(
        key, offset, length, io_buf, io_cap, out_len, value_len,
        [&](char* buf, size_t cap, size_t* n, size_t* stored_len) {
          const char* p = nullptr;
          Status s = group_.RangeDirect(key, offset, length, buf, cap, &p, n,
                                        stored_len);
          // A DIO superset read returns p = buf + head with head in (0, 4095]:
          // src points INSIDE dst, so this in-place compaction overlaps
          // (forward) and must be a memmove -- memcpy on an overlap is UB and
          // can hand shifted bytes to the leader AND every follower.
          if (s == Status::kOk && p && p != buf && *n > 0)
            std::memmove(buf, p, *n < cap ? *n : cap);
          return s;
        },
        &outcome);
    own_read = outcome.leader;
    if (st == Status::kOk && out_data) *out_data = io_buf;
  } else {
    st = group_.RangeDirect(key, offset, length, io_buf, io_cap, out_data,
                            out_len, value_len);
  }
  if (st == Status::kOk) {
    cache_hit_.fetch_add(1, std::memory_order_relaxed);
    bytes_read_.fetch_add(*out_len, std::memory_order_relaxed);
    // Only a complete value may become a durable RAM resident. A slice must
    // never replace the key's full contents or reported stored length.
    // Best-effort: the data is already durable, so no flush is needed.
    if (ram_ && own_read && offset == 0 && *out_data && *out_len > 0 &&
        *out_len == *value_len && PromoteRead(key))
      ram_->PutDurable(key, *out_data, *out_len, generation);
  } else if (st == Status::kNotFound) {
    cache_miss_.fetch_add(1, std::memory_order_relaxed);
  } else if (st == Status::kIOError) {
    get_io_err_.fetch_add(1, std::memory_order_relaxed);
  }
  if (samp) get_lat_.Observe(NowSec() - t0);
  return st;
}
PreparedRead KvNodeServer::PrepareReadForKey(
    const BlockKey& key, uint64_t offset, uint64_t length, char* staging,
    size_t staging_cap) {
  if (staging == nullptr) return PreparedRead::Result(Status::kInvalid);

  // RAM preparation is part of the same owner as disk preparation. Arena hits
  // keep their send pin until SEND completion; dedicated allocations copy into
  // the registered staging buffer but retain the pin until the same terminal
  // action, so every exit follows one ownership protocol.
  auto prepare_ram_hit = [&]() -> PreparedRead {
    if (!ram_) return PreparedRead::Result(Status::kInvalid);
    RamTier::Hit hit;
    if (!ram_->GetPrep(key, offset, length, &hit))
      return PreparedRead::Result(Status::kInvalid);
    const uint64_t token = hit.TransferToken();
    PreparedRead read = PreparedRead::Ready(
        hit.in_arena ? hit.ptr : staging, hit.len, hit.value_len,
        hit.in_arena, staging, staging_cap, this, token, &FinishRamRead,
        hit.allocated_bytes);
    if (hit.len > staging_cap || (hit.len != 0 && hit.ptr == nullptr)) {
      read.Abort();
      return PreparedRead::Result(Status::kInvalid);
    }
    if (!hit.in_arena && hit.len != 0)
      std::memcpy(staging, hit.ptr, hit.len);
    cache_hit_.fetch_add(1, std::memory_order_relaxed);
    bytes_read_.fetch_add(hit.len, std::memory_order_relaxed);
    return read;
  };

  PreparedRead resident = prepare_ram_hit();
  if (resident.status() == Status::kOk) return resident;

  // A direct-to-RAM leader publishes before waking its no-copy followers.
  // A successful follower acquires its own send pin; timeout, failure, or a
  // same-thread duplicate must issue an independent read instead of waiting a
  // second time in the synchronous coalescer fallback.
  bool force_independent_read = false;
  if (ram_ && coalesce_enabled_) {
    const ReadCoalescer::PublishedWait waited =
        read_coalescer_.WaitForPublished(key, offset, length);
    if (waited != ReadCoalescer::PublishedWait::kNotEligible) {
      if (waited == ReadCoalescer::PublishedWait::kReady) {
        resident = prepare_ram_hit();
        if (resident.status() == Status::kOk) {
          read_coalescer_.RecordCoalesced();
          return resident;
        }
      }
      force_independent_read = true;
    }
  }

  // A normal async leader is joined by the synchronous copy-follower path.
  // Direct-flight timeout/failure/self-recursion bypasses that second wait.
  if (!force_independent_read && coalesce_enabled_ &&
      read_coalescer_.InFlight(key, offset, length))
    return PreparedRead::Result(Status::kInvalid);

  // Which disk value is read, taken before the read: a promotion is installed
  // only if the disk still holds that value when it lands.
  const uint64_t generation = ram_ ? DiskGeneration(key) : 0;
  ReadLease lease;
  const Status st =
      group_.RangeDirectPrep(key, offset, length, staging_cap, &lease);
  if (st == Status::kNotFound) {
    cache_miss_.fetch_add(1, std::memory_order_relaxed);
  } else if (st == Status::kIOError) {
    get_io_err_.fetch_add(1, std::memory_order_relaxed);
  }
  if (st != Status::kOk) return PreparedRead::Result(st);

  const bool whole = offset == 0 && lease.value_len != 0 &&
                     lease.payload_len == lease.value_len;
  // Decided before reserving: under DFKV_RAM_PROMOTE=second a read that is
  // not promoted neither reserves RAM nor installs a copy when it lands. The
  // read is noted only once it is sure to go ahead, as a read given up below
  // is redone by the synchronous path, which notes it.
  const bool promote = ram_ && whole && PromoteRead(key, /*note=*/false);

  // Reserve before publishing the async flight: followers may skip their
  // payload copy only when the flight is born with an immutable promise that
  // successful completion will publish a durable RAM entry before wakeup.
  RamTier::DurableReservation reservation;
  bool direct_reservation = false;
  if (promote && lease.head == 0 &&
      ram_->ReserveDurable(key, lease.value_len, &reservation, generation)) {
    direct_reservation = lease.aligned_len <= reservation.capacity();
    if (!direct_reservation)
      reservation = RamTier::DurableReservation{};
  }

  uint64_t flight = 0;
  if (!force_independent_read && coalesce_enabled_ && lease.fd() >= 0 &&
      lease.payload_len != 0 && lease.aligned_len <= staging_cap &&
      lease.aligned_len <= std::numeric_limits<unsigned>::max()) {
    flight = read_coalescer_.TryRegisterAsync(
        key, offset, length, whole, lease.value_len, direct_reservation,
        generation);
    if (flight == 0) {
      // The lease and any hidden reservation release on return.
      return PreparedRead::Result(Status::kInvalid);
    }
  }

  if (recent_reads_ && ram_ && whole) NoteRead(key, promote);

  // Whole-value misses can read O_DIRECT straight into their final registered
  // RAM slot. The reservation stays hidden and pinned across disk IO and RDMA
  // SEND; completion publishes the durable entry, then SEND completion drops
  // the final pin. Partial, oversized, and arena-pressure reads retain staging.
  if (direct_reservation) {
    auto promoted = std::make_shared<PromotedRead>();
    promoted->server = this;
    promoted->reservation = std::move(reservation);
    char* destination = promoted->reservation.data();
    const size_t destination_cap = promoted->reservation.capacity();
    return PreparedRead::Storage(
        std::move(lease), destination, destination_cap, promoted.get(),
        flight, &FinishPromotedDiskRead,
        /*source_registered=*/true, staging, staging_cap, promoted,
        destination_cap);
  }

  return PreparedRead::Storage(
      std::move(lease), staging, staging_cap, this, flight,
      recent_reads_ && !promote ? &FinishUnpromotedDiskRead : &FinishDiskRead);
}


bool KvNodeServer::PromoteRead(const BlockKey& key, bool note) {
  if (!recent_reads_) return true;
  const bool again = recent_reads_->ReadAgain(key, SteadyMs(), note);
  if (note && !again) promote_deferred_.fetch_add(1, std::memory_order_relaxed);
  return again;
}

void KvNodeServer::NoteRead(const BlockKey& key, bool promoted) {
  recent_reads_->ReadAgain(key, SteadyMs());
  if (!promoted) promote_deferred_.fetch_add(1, std::memory_order_relaxed);
}

void KvNodeServer::FinishDiskRead(
    void* owner, uint64_t flight, bool committed, Status result,
    size_t bytes_read, double elapsed_sec, const char* data) noexcept {
  KvNodeServer* server = static_cast<KvNodeServer*>(owner);
  if (server == nullptr) return;
  server->FinishStagedDiskRead(flight, committed, result, bytes_read,
                               elapsed_sec, data, /*promote=*/true);
}

void KvNodeServer::FinishUnpromotedDiskRead(
    void* owner, uint64_t flight, bool committed, Status result,
    size_t bytes_read, double elapsed_sec, const char* data) noexcept {
  KvNodeServer* server = static_cast<KvNodeServer*>(owner);
  if (server == nullptr) return;
  server->FinishStagedDiskRead(flight, committed, result, bytes_read,
                               elapsed_sec, data, /*promote=*/false);
}

void KvNodeServer::FinishStagedDiskRead(uint64_t flight, bool committed,
                                        Status result, size_t bytes_read,
                                        double elapsed_sec, const char* data,
                                        bool promote) noexcept {
  if (!committed) {
    if (flight)
      read_coalescer_.CompleteAsync(flight, Status::kIOError, nullptr, 0);
    return;
  }
  const bool ok = result == Status::kOk;
  if (ok) {
    cache_hit_.fetch_add(1, std::memory_order_relaxed);
    bytes_read_.fetch_add(bytes_read, std::memory_order_relaxed);
  } else if (result == Status::kNotFound) {  // failed its check: a miss
    cache_miss_.fetch_add(1, std::memory_order_relaxed);
  } else {
    get_io_err_.fetch_add(1, std::memory_order_relaxed);
  }
  if (flight) {
    BlockKey key;
    bool whole = false;
    bool recurrent = false;
    uint64_t generation = 0;
    (void)read_coalescer_.CompleteAsync(
        flight, ok ? Status::kOk : Status::kIOError, data, bytes_read, &key,
        &whole, &recurrent, &generation);
    if (promote && ok && whole && ram_ && data && bytes_read) {
      ram_->PutDurable(key, data, bytes_read, generation);
    }
  }
  if (lat_sampler_.ShouldSample()) get_lat_.Observe(elapsed_sec);
}

void KvNodeServer::FinishPromotedDiskRead(
    void* owner, uint64_t flight, bool committed, Status result,
    size_t bytes_read, double elapsed_sec, const char* data) noexcept {
  PromotedRead* promoted = static_cast<PromotedRead*>(owner);
  KvNodeServer* server = promoted == nullptr ? nullptr : promoted->server;
  if (server == nullptr) return;
  if (!committed) {
    if (flight)
      server->read_coalescer_.CompleteAsync(flight, Status::kIOError, nullptr,
                                            0);
    return;
  }

  const bool ok = result == Status::kOk;
  if (ok) {
    // Commit keeps the reservation's allocator pin as a send pin. The
    // PreparedRead post-read hold releases it only after SEND completion.
    (void)promoted->reservation.Commit();
    server->cache_hit_.fetch_add(1, std::memory_order_relaxed);
    server->bytes_read_.fetch_add(bytes_read, std::memory_order_relaxed);
  } else if (result == Status::kNotFound) {  // failed its check: a miss
    server->cache_miss_.fetch_add(1, std::memory_order_relaxed);
  } else {
    server->get_io_err_.fetch_add(1, std::memory_order_relaxed);
  }
  if (flight) {
    (void)server->read_coalescer_.CompleteAsync(
        flight, ok ? Status::kOk : Status::kIOError, data, bytes_read);
  }
  if (server->lat_sampler_.ShouldSample())
    server->get_lat_.Observe(elapsed_sec);
}

void KvNodeServer::FinishRamRead(
    void* owner, uint64_t token, bool committed, Status result,
    size_t bytes_read, double elapsed_sec, const char* data) noexcept {
  (void)committed;
  (void)result;
  (void)bytes_read;
  (void)elapsed_sec;
  (void)data;
  KvNodeServer* server = static_cast<KvNodeServer*>(owner);
  if (server != nullptr && server->ram_) server->ram_->ReleaseToken(token);
}

void KvNodeServer::FinishPullRamRead(
    void* owner, uint64_t token, bool committed, Status result,
    size_t bytes_read, double elapsed_sec, const char* data) noexcept {
  auto* server = static_cast<KvNodeServer*>(owner);
  if (server != nullptr && committed && result == Status::kOk) {
    server->ram_->hits_.fetch_add(1, std::memory_order_relaxed);
    server->cache_hit_.fetch_add(1, std::memory_order_relaxed);
    server->bytes_read_.fetch_add(bytes_read, std::memory_order_relaxed);
    if (server->lat_sampler_.ShouldSample())
      server->get_lat_.Observe(elapsed_sec);
  }
  FinishRamRead(owner, token, committed, result, bytes_read, elapsed_sec, data);
}

// Keep-alive: serve requests on this connection until the peer closes it.
void KvNodeServer::Handle(int fd,
                          std::chrono::steady_clock::time_point accepted_at) {
  // Bound the declared payload_len BEFORE the allocation below: without this
  // a forged 42-byte prefix is a 16 GiB allocation. Same bound as the RDMA
  // path (utils/wire_limits.h).
  const uint64_t max_payload = max_request_payload_
                                   ? max_request_payload_
                                   : wire_limits::MaxRequestPayload();
  // Absolute deadline for the FIRST complete frame on the connection
  // (DFKV_TCP_FIRST_REQ_MS). SO_RCVTIMEO is a per-syscall budget, so a drip
  // feeder (1 byte/interval) otherwise pins a connection slot forever; until
  // the first prefix+payload are fully in, each read's per-syscall budget is
  // recomputed as min(base timeout, deadline remaining) and an already-spent
  // budget closes the connection. 0 disables (legacy per-syscall behavior).
  bool first_frame_pending = tcp_first_req_ms_ > 0;
  const auto first_deadline =
      accepted_at + std::chrono::milliseconds(tcp_first_req_ms_);
  const timeval base_timeout{tcp_io_timeout_seconds_, 0};
  // Every later frame has an absolute deadline too, from its first byte:
  // the base timeout plus its payload at kTcpMinFrameBytesPerSecond. A peer
  // that trickles a frame in, each byte inside the per-syscall timeout, no
  // longer holds the connection (and its handler thread) for good.
  constexpr uint64_t kTcpMinFrameBytesPerSecond = 1ull << 20;
  // The payload buffer grows as bytes arrive, by doubling from 1 MiB, rather
  // than being allocated and zeroed at its declared length up front: a frame
  // that declares 32 MiB and sends nothing costs no memory.
  constexpr size_t kPayloadFirstChunk = 1u << 20;
  while (running_) {
    char prefix[kReqPrefix];
    std::chrono::steady_clock::time_point frame_deadline = first_deadline;
    if (first_frame_pending) {
      if (!net::ReadAllWithin(fd, prefix, kReqPrefix, base_timeout,
                              first_deadline))
        return;
    } else {
      if (!net::ReadAll(fd, prefix, 1)) return;  // idle until a frame starts
      frame_deadline = std::chrono::steady_clock::now() +
                       std::chrono::seconds(tcp_io_timeout_seconds_);
      if (!net::ReadAllWithin(fd, prefix + 1, kReqPrefix - 1, base_timeout,
                              frame_deadline))
        return;
    }
    ReqFields rq;
    if (!DecodeReq(prefix, &rq, max_payload)) return;  // bad version / oversize => drop
    // Same-source bound for a kRange's declared range length: a value can
    // never exceed max_payload, so a longer range is always garbage or a
    // hostile pre-allocation attempt. The RDMA frontend rejects this shape at
    // its own decode/serve gates (rdma_server.cc); the TCP frontend drops the
    // connection here so no request handler ever sees it.
    if (rq.op == static_cast<uint8_t>(WireOp::kRange) &&
        rq.length > max_payload)
      return;
    // The payload's allowance, for the first frame as for later ones.
    frame_deadline += std::chrono::microseconds(
        rq.payload_len * 1000000 / kTcpMinFrameBytesPerSecond);
    std::vector<char> payload;
    for (size_t got = 0; got < rq.payload_len;) {
      const size_t want = std::min<size_t>(
          rq.payload_len, std::max(kPayloadFirstChunk, 2 * got));
      payload.resize(want);
      if (!net::ReadAllWithin(fd, payload.data() + got, want - got,
                              base_timeout, frame_deadline))
        return;
      got = want;
    }
    // Restore the plain per-syscall budget the deadline reads shrank, so the
    // wait for the next frame behaves as before.
    first_frame_pending = false;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &base_timeout,
                 sizeof(base_timeout));

    std::string data;
    size_t value_len = 0;
    Status st = ProcessRequestForKey(
        rq.op, rq.Key(), rq.offset, rq.length, payload.data(), rq.payload_len,
        &data, &value_len);
    char rp[kRespPrefix];
    EncodeResp(rp, st, data.size(), static_cast<uint64_t>(value_len));
    if (!net::WriteAll(fd, rp, sizeof(rp))) return;
    if (!data.empty() && !net::WriteAll(fd, data.data(), data.size())) return;
  }
}

}  // namespace dfkv
