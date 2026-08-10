#include "cache/kv_node_server.h"

#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#include <vector>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include "common/config_dump.h"
#include "utils/log.h"
#include "utils/net_util.h"
#include "utils/thread_name.h"
#include "utils/wire_limits.h"
#include "utils/prom_escape.h"
#include "transport/transport.h"

namespace dfkv {

namespace {
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
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return fallback;
    if (*value < '0' || *value > '9') {
      DFKV_LOG_WARN(std::string("invalid ") + name + "='" + value +
                    "'; using " + std::to_string(fallback));
      return fallback;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' ||
        (parsed == 0 && !allow_zero)) {
      DFKV_LOG_WARN(std::string("invalid ") + name + "='" + value +
                    "'; using " + std::to_string(fallback));
      return fallback;
    }
    if (parsed > hard_max) {
      DFKV_LOG_WARN(std::string(name) + " clamped to hard maximum " +
                    std::to_string(hard_max));
      return hard_max;
    }
    return parsed;
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

KvNodeServer::KvNodeServer(const std::string& cache_dir,
                           uint64_t capacity_bytes)
    : group_(DiskCacheGroup::Options{{cache_dir}, capacity_bytes}) {
  if (!group_.Healthy()) return;
  InitRamTier();
  InitAdmission();
}

KvNodeServer::KvNodeServer(const std::vector<std::string>& cache_dirs,
                           uint64_t capacity_bytes)
    : group_(DiskCacheGroup::Options{cache_dirs, capacity_bytes}) {
  if (!group_.Healthy()) return;
  InitRamTier();
  InitAdmission();
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
  if (const char* v = std::getenv("DFKV_PUT_INFLIGHT_LIMIT")) {
    unsigned long long n = std::strtoull(v, nullptr, 10);
    put_busy_limit_ = static_cast<size_t>(n);
  }
  config_dump::RecordResolved("DFKV_PUT_INFLIGHT_LIMIT",
                              std::to_string(put_busy_limit_));
  // Read-side convoy collapse master switch, plus its timeout/recur sub-knobs
  // (forced now so their defaults show even before the first coalesced read).
  config_dump::RecordResolved("DFKV_READ_COALESCE", coalesce_enabled_ ? "on" : "off");
  if (coalesce_enabled_) ReadCoalescer::RecordConfig();
}

void KvNodeServer::InitRamTier() {
  // Off by default. DFKV_RAM_TIER in {1,on,true,yes} enables the RAM hot tier;
  // DFKV_RAM_TIER_BYTES sizes the pre-registered arena (default 16 GiB).
  const char* e = std::getenv("DFKV_RAM_TIER");
  const std::string v = (e && *e) ? std::string(e) : "";
  const bool enabled = (v == "1" || v == "on" || v == "true" || v == "yes");
  config_dump::RecordResolved("DFKV_RAM_TIER", enabled ? "on" : "off");
  if (!enabled) return;
  RamTier::Options o;
  if (const char* b = std::getenv("DFKV_RAM_TIER_BYTES")) {
    unsigned long long n = std::strtoull(b, nullptr, 10);
    if (n > 0) o.bytes = n;
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
  ram_write_back_ = mode == "writeback";
  config_dump::RecordResolved("DFKV_RAM_WRITE_MODE", mode);
  // Flush workers default to 4x the disk count (cap 16). One sync DIO stream
  // per worker leaves NVMe queues nearly idle for small objects; measured on a
  // 3-disk node (64 KiB saturated writes): 3 workers 6.8k ops/s -> 16 workers
  // 11.7k ops/s (+73%), PUT p99 95 -> 37 ms. The old disk-count default dated
  // from when the ingest path's lock contention masked any worker scaling.
  o.flush_threads = static_cast<uint32_t>(
      std::min<size_t>(4 * (group_.DiskCount() ? group_.DiskCount() : 1), 16));
  if (const char* f = std::getenv("DFKV_RAM_FLUSH_THREADS")) {
    unsigned long long n = std::strtoull(f, nullptr, 10);
    if (n > 0 && n <= 64) o.flush_threads = static_cast<uint32_t>(n);
  }
  // Background free-slot reclaimer cadence (ms; 0 disables). Default 10.
  if (const char* r = std::getenv("DFKV_RAM_RECLAIM_MS"))
    o.reclaim_interval_ms = static_cast<uint32_t>(std::strtoul(r, nullptr, 10));
  const char* ack_env = std::getenv("DFKV_PUT_ACK_MODE");
  const std::string ack = ack_env ? std::string(ack_env) : "";
  if (ack.empty()) {
    ram_ack_enabled_ = ram_write_back_;
  } else if (ack == "ram" || ack == "disk") {
    ram_ack_enabled_ = ack == "ram";
  } else {
    DFKV_LOG_WARN("DFKV_PUT_ACK_MODE='" + ack +
                  "' not recognized; using RAM-ACK for writeback");
    ram_ack_enabled_ = ram_write_back_;
  }
  if (const char* p = std::getenv("DFKV_RAM_ACK_HIGH_WATERMARK_PCT")) {
    const unsigned long n = std::strtoul(p, nullptr, 10);
    if (n <= 100) o.ack_high_watermark_pct = static_cast<uint32_t>(n);
  }
  config_dump::RecordResolved("DFKV_PUT_ACK_MODE",
                              ram_ack_enabled_ ? "ram" : "disk");
  config_dump::RecordResolved("DFKV_RAM_ACK_HIGH_WATERMARK_PCT",
                              std::to_string(o.ack_high_watermark_pct));
  config_dump::RecordResolved("DFKV_RAM_TIER_BYTES", std::to_string(o.bytes));
  config_dump::RecordResolved("DFKV_RAM_RECLAIM_MS", std::to_string(o.reclaim_interval_ms));
  // Flusher persists a RAM slot to the disk group. CacheDirect (not Cache): the
  // arena slot is 4 KiB-aligned with cap slack, so a direct-mode slab lands it
  // via O_DIRECT -- a buffered flush would route the RAM tier's entire write
  // volume back through the page cache, defeating the tier's purpose of keeping
  // memory use bounded and explicit on GPU nodes.
  auto tier = std::make_unique<RamTier>(
      o, [this](const BlockKey& k, char* d, size_t l, size_t cap) {
        return group_.CacheDirect(k, d, l, cap) == Status::kOk;
      });
  // Batched sink: a flush worker's whole dequeue rides one lock-amortized
  // (and, with a uring build, one-submit) store visit per disk.
  tier->set_flush_batch([this](const std::vector<RamTier::FlushItem>& items) {
    std::vector<StoreEngine::CacheBatchItem> b;
    b.reserve(items.size());
    for (const auto& it : items)
      b.push_back(StoreEngine::CacheBatchItem{it.key, it.data, it.len, it.cap});
    std::vector<Status> sts = group_.CacheDirectBatch(b);
    std::vector<bool> ok(items.size(), false);
    for (size_t i = 0; i < sts.size(); ++i) ok[i] = (sts[i] == Status::kOk);
    return ok;
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

void KvNodeServer::Stop() {
  if (!running_.exchange(false)) return;  // idempotent
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
  metric("dfkv_accepts_total", "counter", "TCP connections accepted", AcceptCount());
  metric("dfkv_evictions_total", "counter", "Objects evicted (capacity pressure)",
         group_.Evictions());
  metric("dfkv_evicted_bytes_total", "counter", "Bytes evicted", group_.EvictedBytes());
  // gauges
  metric("dfkv_objects", "gauge", "Cached objects on this node", group_.Count());
  metric("dfkv_used_bytes", "gauge", "Bytes used on disk", group_.UsedBytes());
  metric("dfkv_disks", "gauge", "Backing NVMe disks", group_.DiskCount());
  metric("dfkv_tenant_default_quota_bytes", "gauge",
         "Default per-node tenant quota bytes (0 means unlimited)",
         group_.TenantDefaultQuotaBytes());
  metric("dfkv_tenant_quota_rejections_total", "counter",
         "PUT items rejected by per-node tenant quotas",
         group_.TenantQuotaRejections());
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
  if (group_.EngineName() == "slab") {
    const auto ss = group_.SlabStats();
    metric("dfkv_slab_dio_write_fallback_total", "counter",
           "Direct-mode PUTs that fell back to a buffered write", ss.dio_write_fallbacks);
    metric("dfkv_slab_dio_read_fallback_total", "counter",
           "Direct-mode aligned GETs that fell back to a buffered read", ss.dio_read_fallbacks);
    metric("dfkv_slab_table_sync_total", "counter",
           "slots.tbl fdatasync cycles performed", ss.table_syncs);
    metric("dfkv_slab_extent_steals_total", "counter",
           "Cross-class extent steals (class capacity churn)", ss.steals);
    metric("dfkv_slab_cold_steals_total", "counter",
           "Steals of globally-cold cross-class extents (full-ring self-thrash guard)",
           ss.cold_steals);
    metric("dfkv_slab_watermark_evictions_total", "counter",
           "Proactive watermark extent evictions (headroom kept ahead of demand)",
           ss.watermark_evictions);
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
    metric("dfkv_slab_metadata_io_errors_total", "counter",
           "Failed slots.tbl or epoch-state durability operations",
           ss.metadata_io_errors);
    metric("dfkv_slab_unclean_resets_total", "counter",
           "Dirty prior epochs discarded during cold recovery",
           ss.unclean_resets);
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
           "PUTs rejected by the disk-write admission gate (kCacheFull)",
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
           "RAM-ACK PUTs forced to wait for disk at the dirty watermark",
           ram_->AckBackpressure());
    metric("dfkv_ram_post_ack_flush_failures_total", "counter",
           "Acknowledged RAM PUTs whose background disk flush later failed",
           ram_->PostAckFlushFailures());
    metric("dfkv_ram_budget_bytes", "gauge",
           "Hard total RAM-tier budget across arena and dedicated values",
           ram_->budget_bytes());
    metric("dfkv_ram_arena_bytes", "gauge",
           "Fixed registered arena bytes reserved from the total budget",
           ram_->arena_bytes());
    metric("dfkv_ram_large_budget_bytes", "gauge",
           "Dedicated oversized-value reserve within the total budget",
           ram_->large_budget_bytes());
    metric("dfkv_ram_used_bytes", "gauge",
           "Resident slot plus dedicated allocation bytes",
           ram_->UsedBytes());
    metric("dfkv_ram_large_used_bytes", "gauge",
           "Resident dedicated oversized allocation bytes",
           ram_->large_used_bytes());
    metric("dfkv_ram_healthy", "gauge",
           "Whether the RAM tier has avoided a terminal flush failure",
           ram_->healthy() ? 1 : 0);
    metric("dfkv_ram_flush_threads", "gauge",
           "Actual RAM flusher count after the per-shard minimum adjustment",
           ram_->flusher_count());
  }
  return s;
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
    if (fd < 0) { if (!running_) break; continue; }
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
    conn_fds_.insert(fd);
    auto done = std::make_shared<std::atomic<bool>>(false);
    conns_.push_back({std::thread([this, fd, done, accepted_at] {
                        NameThisThread("kv-serve");
                        Handle(fd, accepted_at);
                        open_connections_.fetch_sub(1, std::memory_order_relaxed);
                        { std::lock_guard<std::mutex> lk(conn_mu_); conn_fds_.erase(fd); }
                        ::close(fd);
                        done->store(true, std::memory_order_release);  // last act
                      }),
                      done});
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
      bool samp = lat_sampler_.ShouldSample();
      double t0 = samp ? NowSec() : 0.0;
      // RAM-ACK mode publishes a flush-pinned copy and returns immediately;
      // its dirty-byte watermark switches individual requests back to waiting
      // for disk. Admission failure falls through to the normal disk path.
      bool needs_disk = true;
      if (ram_ && group_.TenantQuotaBytes(key.tenant_hash) == 0) {
        st = ram_ack_enabled_
                 ? ram_->PutWriteBack(key, payload, payload_len)
                 : ram_->PutCommitted(key, payload, payload_len);
        needs_disk = st == Status::kCacheFull;
        if (st == Status::kOk) {
          cache_put_.fetch_add(1, std::memory_order_relaxed);
          bytes_written_.fetch_add(payload_len, std::memory_order_relaxed);
        } else if (!needs_disk && st == Status::kIOError) {
          put_io_err_.fetch_add(1, std::memory_order_relaxed);
        }
      }
      if (needs_disk && put_busy_limit_ > 0 &&
          disk_put_inflight_.load(std::memory_order_relaxed) >=
              put_busy_limit_) {
        // Apply the same admission gate as the RDMA CacheDirect path. TCP is a
        // diagnostic datapath, not an ungated side door around disk-write
        // concurrency and backpressure.
        st = Status::kCacheFull;
        put_busy_.fetch_add(1, std::memory_order_relaxed);
        needs_disk = false;
      }
      if (needs_disk) {
        disk_put_inflight_.fetch_add(1, std::memory_order_relaxed);
        st = group_.Cache(key, payload, payload_len);
        disk_put_inflight_.fetch_sub(1, std::memory_order_relaxed);
        if (st == Status::kOk) {
          cache_put_.fetch_add(1, std::memory_order_relaxed);
          bytes_written_.fetch_add(payload_len, std::memory_order_relaxed);
        } else if (st == Status::kIOError) {
          put_io_err_.fetch_add(1, std::memory_order_relaxed);
        }
      }
      if (samp) put_lat_.Observe(NowSec() - t0);
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
                    std::memcpy(buf, tmp.data(), m);
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
      const bool ram_had = ram_ && ram_->Remove(key);
      const Status disk_status = group_.Remove(key);
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
    case WireOp::kMembers:
      *out_data = members_;  // advertised cluster membership (for client discovery)
      st = Status::kOk;
      break;
    case WireOp::kRegister:
    case WireOp::kHeartbeat:
    case WireOp::kListMembers:
    case WireOp::kListGroups:
    case WireOp::kClientRegister:
    case WireOp::kClientHeartbeat:
    case WireOp::kListClients:
      st = Status::kInvalid;  // MDS ops are not served by a cache node
      break;
    }
  if (st == Status::kInvalid) invalid_ops_.fetch_add(1, std::memory_order_relaxed);
  return st;
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


std::vector<Status> KvNodeServer::CacheDirectBatchForKeys(
    const std::vector<StoreEngine::CacheBatchItem>& items) {
  return group_.CacheDirectBatch(items);
}

Status KvNodeServer::CacheDirectForKey(const BlockKey& key, char* data,
                                       size_t len, size_t cap) {
  if (len == 0 || data == nullptr) {
    invalid_ops_.fetch_add(1, std::memory_order_relaxed);
    return Status::kInvalid;
  }
  bool samp = lat_sampler_.ShouldSample();
  double t0 = samp ? NowSec() : 0.0;
  // PUT write-around: write directly to disk, skip the synchronous 1 MiB
  // arena memcpy. The arena is filled lazily by read-promotion on GET miss
  // (RangeDirectForKey), so hot keys still get zero-copy RDMA GET from the
  // arena without paying a per-PUT copy on the write path. The old path
  // (ram_->PutCommitted) copied every PUT into the arena and flushed
  // asynchronously — the 1 MiB memcpy was 44% of serve-thread CPU and
  // capped single-node PUT at 3.4 GB/s vs 4.4 GB/s direct-disk.
  Status st = Status::kCacheFull;
  const bool quota_ok = !ram_ || group_.TenantQuotaBytes(key.tenant_hash) == 0;
  bool needs_disk = true;
  // Write-back admits to the RAM arena first. RAM-ACK returns after the
  // flush-pinned value is visible; disk-ACK waits for the same flush result.
  // Genuine arena backpressure falls through to direct disk write.
  if (ram_write_back_ && ram_ && quota_ok) {
    st = ram_ack_enabled_ ? ram_->PutWriteBack(key, data, len)
                          : ram_->PutCommitted(key, data, len);
    needs_disk = st == Status::kCacheFull;
    if (st == Status::kOk) {
      cache_put_.fetch_add(1, std::memory_order_relaxed);
      bytes_written_.fetch_add(len, std::memory_order_relaxed);
    } else if (!needs_disk && st == Status::kIOError) {
      put_io_err_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  if (needs_disk && quota_ok && put_busy_limit_ > 0 &&
      disk_put_inflight_.load(std::memory_order_relaxed) >= put_busy_limit_) {
    st = Status::kCacheFull;
    put_busy_.fetch_add(1, std::memory_order_relaxed);
  } else if (needs_disk && quota_ok) {
    disk_put_inflight_.fetch_add(1, std::memory_order_relaxed);
    st = group_.CacheDirect(key, data, len, cap);
    disk_put_inflight_.fetch_sub(1, std::memory_order_relaxed);
    if (st == Status::kOk) {
      cache_put_.fetch_add(1, std::memory_order_relaxed);
      bytes_written_.fetch_add(len, std::memory_order_relaxed);
    } else if (st == Status::kIOError) {
      put_io_err_.fetch_add(1, std::memory_order_relaxed);
    } else if (st == Status::kInvalid) {
      invalid_ops_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  if (samp) put_lat_.Observe(NowSec() - t0);
  return st;
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
  Status st;
  if (coalesce_enabled_) {
    // Leader reads into the shared scratch, every rank's convoy copy lands in
    // its own registered io_buf; *out_data must point at io_buf either way.
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
        });
    if (st == Status::kOk && out_data) *out_data = io_buf;
  } else {
    st = group_.RangeDirect(key, offset, length, io_buf, io_cap, out_data,
                            out_len, value_len);
  }
  if (st == Status::kOk) {
    cache_hit_.fetch_add(1, std::memory_order_relaxed);
    bytes_read_.fetch_add(*out_len, std::memory_order_relaxed);
    // Read promotion: install the just-read value into the RAM arena so
    // subsequent GETs hit the zero-copy RDMA path. Best-effort (a full arena
    // silently skips); the data is already durable on disk so this costs zero
    // flush bandwidth.  With the PUT write-around change above, this is the
    // ONLY way the arena gets populated.
    if (ram_ && *out_data && *out_len > 0)
      ram_->PutDurable(key, *out_data, *out_len);
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
        hit.in_arena, staging, staging_cap, this, token, &FinishRamRead);
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

  // Reserve before publishing the async flight: followers may skip their
  // payload copy only when the flight is born with an immutable promise that
  // successful completion will publish a durable RAM entry before wakeup.
  RamTier::DurableReservation reservation;
  bool direct_reservation = false;
  if (ram_ && whole && lease.head == 0 &&
      ram_->ReserveDurable(key, lease.value_len, &reservation)) {
    direct_reservation = lease.aligned_len <= reservation.capacity();
    if (!direct_reservation)
      reservation = RamTier::DurableReservation{};
  }

  uint64_t flight = 0;
  if (!force_independent_read && coalesce_enabled_ && lease.fd() >= 0 &&
      lease.payload_len != 0 && lease.aligned_len <= staging_cap &&
      lease.aligned_len <= std::numeric_limits<unsigned>::max()) {
    flight = read_coalescer_.TryRegisterAsync(
        key, offset, length, whole, lease.value_len, direct_reservation);
    if (flight == 0) {
      // The lease and any hidden reservation release on return.
      return PreparedRead::Result(Status::kInvalid);
    }
  }

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
        /*source_registered=*/true, staging, staging_cap, promoted);
  }

  return PreparedRead::Storage(std::move(lease), staging, staging_cap, this,
                               flight, &FinishDiskRead);
}

void KvNodeServer::FinishDiskRead(
    void* owner, uint64_t flight, bool committed, Status result,
    size_t bytes_read, double elapsed_sec, const char* data) noexcept {
  KvNodeServer* server = static_cast<KvNodeServer*>(owner);
  if (server == nullptr) return;
  if (!committed) {
    if (flight)
      server->read_coalescer_.CompleteAsync(flight, Status::kIOError, nullptr,
                                            0);
    return;
  }
  const bool ok = result == Status::kOk;
  if (ok) {
    server->cache_hit_.fetch_add(1, std::memory_order_relaxed);
    server->bytes_read_.fetch_add(bytes_read, std::memory_order_relaxed);
  } else {
    server->get_io_err_.fetch_add(1, std::memory_order_relaxed);
  }
  if (flight) {
    BlockKey key;
    bool whole = false;
    bool recurrent = false;
    (void)server->read_coalescer_.CompleteAsync(
        flight, ok ? Status::kOk : Status::kIOError, data, bytes_read, &key,
        &whole, &recurrent);
    if (ok && whole && server->ram_ && data && bytes_read) {
      server->ram_->PutDurable(key, data, bytes_read);
    }
  }
  if (server->lat_sampler_.ShouldSample())
    server->get_lat_.Observe(elapsed_sec);
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
  while (running_) {
    char prefix[kReqPrefix];
    if (first_frame_pending) {
      if (!net::ReadAllWithin(fd, prefix, kReqPrefix, base_timeout,
                              first_deadline))
        return;
    } else if (!net::ReadAll(fd, prefix, kReqPrefix)) {
      return;  // peer closed / error
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
    std::vector<char> payload(rq.payload_len);
    if (rq.payload_len) {
      if (first_frame_pending) {
        if (!net::ReadAllWithin(fd, payload.data(), rq.payload_len,
                                base_timeout, first_deadline))
          return;
      } else if (!net::ReadAll(fd, payload.data(), rq.payload_len)) {
        return;
      }
    }
    if (first_frame_pending) {
      // First frame complete: restore the plain base per-syscall budget the
      // deadline reads shrank, so keep-alive reads behave exactly as before.
      first_frame_pending = false;
      ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &base_timeout,
                   sizeof(base_timeout));
    }

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
