#include "cache/kv_node_server.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#include <vector>

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
  // DFKV_RAM_TIER_BYTES sizes the pre-registered arena (default 4 GiB).
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
  config_dump::RecordResolved("DFKV_RAM_TIER_BYTES", std::to_string(o.bytes));
  config_dump::RecordResolved("DFKV_RAM_FLUSH_THREADS", std::to_string(o.flush_threads));
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
  if (tier->ok()) ram_ = std::move(tier);  // arena alloc failed => stay disk-only
}

Status KvNodeServer::Start(int port) {
  if (!group_.Healthy()) {
    DFKV_LOG_ERROR("refusing startup: " + group_.StartupError());
    return Status::kIOError;
  }
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
       "\",transport=\"" DFKV_BUILD_TRANSPORT "\"";
  if (!node_id_.empty() || !node_group_.empty())
    s += ",node=\"" + PromLabelEscape(node_id_) + "\",group=\"" +
         PromLabelEscape(node_group_) + "\"";
  s += "} 1\n";
  uint64_t up = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - start_time_).count();
  metric("dfkv_uptime_seconds", "gauge", "Seconds since node start", up);
  // counters
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
         "Reads that hit a recurrence tombstone (drift-window promotion evidence)",
         read_coalescer_.recur_hits());
  metric("dfkv_accepts_total", "counter", "TCP connections accepted", AcceptCount());
  metric("dfkv_evictions_total", "counter", "Objects evicted (capacity pressure)",
         group_.Evictions());
  metric("dfkv_evicted_bytes_total", "counter", "Bytes evicted", group_.EvictedBytes());
  // gauges
  metric("dfkv_objects", "gauge", "Cached objects on this node", group_.Count());
  metric("dfkv_used_bytes", "gauge", "Bytes used on disk", group_.UsedBytes());
  metric("dfkv_disks", "gauge", "Backing NVMe disks", group_.DiskCount());
  metric("dfkv_open_connections", "gauge", "Currently open client connections",
         rd(open_connections_));
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
    metric("dfkv_ram_flush_backlog", "gauge", "RAM slots queued for flush (not yet durable)", ram_->FlushBacklog());
  }
  return s;
}

void KvNodeServer::AcceptLoop() {
  while (running_) {
    int fd = ::accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) { if (!running_) break; continue; }
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));  // avoid Nagle stalls
    accept_count_.fetch_add(1, std::memory_order_relaxed);
    open_connections_.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(conn_mu_);
    if (!running_) { open_connections_.fetch_sub(1, std::memory_order_relaxed); ::close(fd); break; }
    ReapDoneLocked();  // join handlers that finished since the last accept
    conn_fds_.insert(fd);
    auto done = std::make_shared<std::atomic<bool>>(false);
    conns_.push_back({std::thread([this, fd, done] {
                        NameThisThread("kv-serve");
                        Handle(fd);
                        open_connections_.fetch_sub(1, std::memory_order_relaxed);
                        { std::lock_guard<std::mutex> lk(conn_mu_); conn_fds_.erase(fd); }
                        ::close(fd);
                        done->store(true, std::memory_order_release);  // last act
                      }),
                      done});
  }
}

// Transport-agnostic request processing + metrics (shared by TCP and RDMA).

Status KvNodeServer::ProcessRequestForKey(
    uint8_t op_raw, const BlockKey& key, uint64_t offset, uint64_t length,
    const char* payload, uint64_t payload_len, std::string* out_data) {
  WireOp op = static_cast<WireOp>(op_raw);
  Status st = Status::kInvalid;
  switch (op) {
    case WireOp::kCache: {
      bool samp = lat_sampler_.ShouldSample();
      double t0 = samp ? NowSec() : 0.0;
      // Write-through RAM tier: on acceptance the value is synchronously visible
      // in RAM (read-after-write) and flushed to disk in the background. On
      // backpressure (arena full of un-flushed slots) Put declines and we take
      // the normal disk path.
      if (ram_ && ram_->Put(key, payload, payload_len)) {
        st = Status::kOk;
        cache_put_.fetch_add(1, std::memory_order_relaxed);
        bytes_written_.fetch_add(payload_len, std::memory_order_relaxed);
      } else if (put_busy_limit_ > 0 &&
                 disk_put_inflight_.load(std::memory_order_relaxed) >=
                     put_busy_limit_) {
        // Same admission gate as the RDMA CacheDirect path (see there): the TCP
        // data path is non-production but must not become an ungated side door
        // around the disk-write cap.
        st = Status::kCacheFull;
        put_busy_.fetch_add(1, std::memory_order_relaxed);
      } else {
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
          ram_->Release(h.token);
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
          const size_t cap = length;
          out_data->resize(cap);
          size_t n = 0;
          st = read_coalescer_.Read(key, offset, length, out_data->empty() ? nullptr : &(*out_data)[0], cap, &n,
              [&](char* buf, size_t bcap, size_t* on) {
                std::string tmp;
                Status s = group_.Range(key, offset, length, &tmp);
                if (s == Status::kOk) {
                  size_t m = tmp.size() < bcap ? tmp.size() : bcap;
                  std::memcpy(buf, tmp.data(), m);
                  *on = m;
                }
                return s;
              });
          out_data->resize(st == Status::kOk ? n : 0);
        } else {
          st = group_.Range(key, offset, length, out_data);
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
      // A RAM-resident block (possibly RAM-only, not yet flushed) must count as
      // present, else Exist would report absent for a just-written key.
      if ((ram_ && ram_->Contains(key)) || group_.IsCached(key)) {
        st = Status::kOk; exist_hit_.fetch_add(1, std::memory_order_relaxed);
      } else {
        st = Status::kNotFound; exist_miss_.fetch_add(1, std::memory_order_relaxed);
      }
      // Handler-body latency: both branches take the RAM shard lock (Contains)
      // and/or the disk group's cached-set lock (IsCached), which a slab reclaim
      // or flush can hold — the tail here is a lock-contention signal.
      if (samp) exist_lat_.Observe(NowSec() - t0);
      break;
    }
    case WireOp::kRemove: {
      // Drop from both tiers. RamTier::Remove is best-effort (a still-flushing or
      // in-flight slot declines and is reclaimed later under pressure); the L2
      // eviction this backs targets durable blocks, which drop cleanly.
      bool ram_had = ram_ && ram_->Remove(key);
      Status ds = group_.Remove(key);
      if (ram_had || ds == Status::kOk) {
        st = Status::kOk; remove_ok_.fetch_add(1, std::memory_order_relaxed);
      } else if (ds == Status::kNotFound) {
        st = Status::kNotFound; remove_miss_.fetch_add(1, std::memory_order_relaxed);
      } else {
        st = ds;
      }
      break;
    }
    case WireOp::kStats:
      *out_data = MetricsText();
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
                                     size_t dst_cap, size_t* out_len) {
  bool samp = lat_sampler_.ShouldSample();
  double t0 = samp ? NowSec() : 0.0;
  if (ram_) {
    RamTier::Hit h;
    if (ram_->GetPrep(key, offset, length, &h)) {  // RAM hit: copy out, no disk
      size_t n = h.len < dst_cap ? h.len : dst_cap;
      std::memcpy(dst, h.ptr, n);
      ram_->Release(h.token);
      if (out_len) *out_len = n;
      cache_hit_.fetch_add(1, std::memory_order_relaxed);
      bytes_read_.fetch_add(n, std::memory_order_relaxed);
      if (samp) get_lat_.Observe(NowSec() - t0);
      return Status::kOk;
    }
  }
  Status st;
  if (coalesce_enabled_) {
    st = read_coalescer_.Read(key, offset, length, dst, dst_cap, out_len,
        [&](char* buf, size_t cap, size_t* n) {
          return group_.RangeInto(key, offset, length, buf, cap, n);
        });
  } else {
    st = group_.RangeInto(key, offset, length, dst, dst_cap, out_len);
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


Status KvNodeServer::CacheDirectForKey(const BlockKey& key, char* data,
                                       size_t len, size_t cap) {
  bool samp = lat_sampler_.ShouldSample();
  double t0 = samp ? NowSec() : 0.0;
  // RAM write-through (same [ValueHeader|payload] blob GET returns); backpressure
  // or disabled falls through to the O_DIRECT disk write.
  Status st;
  if (ram_ && ram_->Put(key, data, len)) {
    st = Status::kOk;
    cache_put_.fetch_add(1, std::memory_order_relaxed);
    bytes_written_.fetch_add(len, std::memory_order_relaxed);
  } else if (put_busy_limit_ > 0 &&
             disk_put_inflight_.load(std::memory_order_relaxed) >=
                 put_busy_limit_) {
    // Admission gate (opt-in, --put-inflight-limit): the disk write queue is
    // already `limit` deep -- fast-fail with kCacheFull instead of joining a
    // 100ms+ device queue. Clients treat kCacheFull as a plain put-failure
    // WITHOUT a peer cooldown (unlike kIOError), so the block is simply
    // recomputed later: a controlled miss instead of a latency tail.
    st = Status::kCacheFull;
    put_busy_.fetch_add(1, std::memory_order_relaxed);
  } else {
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
    size_t io_cap, const char** out_data, size_t* out_len) {
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
      ram_->Release(h.token);
      if (out_data) *out_data = io_buf;
      if (out_len) *out_len = n;
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
    st = read_coalescer_.Read(key, offset, length, io_buf, io_cap, out_len,
        [&](char* buf, size_t cap, size_t* n) {
          const char* p = nullptr;
          Status s = group_.RangeDirect(key, offset, length, buf, cap, &p, n);
          if (s == Status::kOk && p && p != buf && *n > 0)
            std::memcpy(buf, p, *n < cap ? *n : cap);
          return s;
        });
    if (st == Status::kOk && out_data) *out_data = io_buf;
  } else {
    st = group_.RangeDirect(key, offset, length, io_buf, io_cap, out_data, out_len);
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


Status KvNodeServer::RangeDirectPrepForKey(
    const BlockKey& key, uint64_t offset, uint64_t length, size_t io_cap,
    KVStore::RangePrep* out, uint64_t* out_flight) {
  if (out_flight) *out_flight = 0;
  // A RAM-resident key has no fd to hand io_uring (and may be RAM-only, not yet
  // on disk). Decline the async prep (kInvalid) WITHOUT counting a miss so the
  // RDMA serve loop falls back to the synchronous RangeDirect, which serves it
  // from the arena. (Only reached when the RAM tier is enabled.)
  if (ram_ && ram_->Contains(key)) {
    if (out) *out = KVStore::RangePrep{};
    return Status::kInvalid;
  }
  // An identical read is already on the disk: decline the async prep so the
  // serve loop falls back to the synchronous RangeDirect, which joins the
  // in-flight read instead of issuing a duplicate NVMe fetch.
  if (coalesce_enabled_ && read_coalescer_.InFlight(key, offset, length)) {
    if (out) *out = KVStore::RangePrep{};
    return Status::kInvalid;
  }
  // RAM consulted and absent -> this GET is a RAM-tier miss regardless of the
  // disk outcome (mirrors GetPrep's accounting on the sync path). Without this
  // the uring path -- the default since v1.20.0 opt-in / v1.27.0 default-on --
  // reports dfkv_ram_miss_total == 0 forever while hits accumulate, so
  // hit/(hit+miss) reads as a fake 100%.
  if (ram_) ram_->CountMiss();
  Status st = group_.RangeDirectPrep(key, offset, length, io_cap, out);
  // Only miss/io-error are final here; a kOk prep is accounted on read completion
  // (RangeDirectComplete) because the async read can still fail.
  if (st == Status::kNotFound) {
    cache_miss_.fetch_add(1, std::memory_order_relaxed);
  } else if (st == Status::kIOError) {
    get_io_err_.fetch_add(1, std::memory_order_relaxed);
  }
  // Register the flight ONLY under the exact conditions the RDMA serve loop
  // defers on (readable fd, non-zero payload, fits the connection buffer):
  // any other prep outcome is served by the sync fallback, which joins flights
  // itself — registering one here that no async read will ever complete would
  // strand its waiters.
  if (st == Status::kOk && coalesce_enabled_ && out_flight && out->fd >= 0 &&
      out->payload_len != 0 && out->aligned_len <= io_cap &&
      out->aligned_len <= std::numeric_limits<unsigned>::max()) {
    // Whole-value reads only are promotable: RAM-tier entries are full
    // [ValueHeader|payload] blobs, so installing a sub-range would corrupt
    // every later reader of the key.
    const bool whole = offset == 0 && out->value_len != 0 &&
                       out->payload_len == out->value_len;
    uint64_t t = read_coalescer_.TryRegisterAsync(key, offset, length, whole);
    if (t == 0) {
      // Lost the race to an identical read registered since the InFlight()
      // check above: decline the async prep; the sync fallback joins the
      // flight instead of issuing a duplicate NVMe fetch.
      ::close(out->fd);
      if (out->token) group_.RangeRelease(out->token);
      *out = KVStore::RangePrep{};
      return Status::kInvalid;
    }
    *out_flight = t;
  }
  return st;
}

void KvNodeServer::RangeDirectComplete(bool ok, size_t bytes_read,
                                       double elapsed_sec, uint64_t flight,
                                       const char* data) {
  if (ok) {
    cache_hit_.fetch_add(1, std::memory_order_relaxed);
    bytes_read_.fetch_add(bytes_read, std::memory_order_relaxed);
  } else {
    get_io_err_.fetch_add(1, std::memory_order_relaxed);
  }
  if (flight) {
    BlockKey key;
    bool whole = false;
    bool recurrent = false;
    const bool had_waiters = read_coalescer_.CompleteAsync(
        flight, ok ? Status::kOk : Status::kIOError, data, bytes_read, &key,
        &whole, &recurrent);
    // Admission gate: promote ONLY pages with convoy evidence — an in-flight
    // waiter (overlap) or a recurrence-tombstone hit (same range re-read
    // within the drift window; v2). A convoy page's laggard ranks, repeat
    // cold reads and post-restart replays are then served from the arena; a
    // one-off cold read never pollutes the RAM tier (the num_extents churn
    // lesson). Durable install: costs no flush bandwidth, evictable at once.
    if (ok && (had_waiters || recurrent) && whole && ram_ && data && bytes_read)
      ram_->PutDurable(key, data, bytes_read);
  }
  // Sample the async (uring) read latency into the SAME op="get" histogram the
  // synchronous RangeDirect feeds — before this the default read path since
  // v1.27.0 contributed no latency samples at all.
  if (lat_sampler_.ShouldSample()) get_lat_.Observe(elapsed_sec);
}

void KvNodeServer::RangeFlightAbort(uint64_t flight) {
  read_coalescer_.CompleteAsync(flight, Status::kIOError, nullptr, 0);
}


bool KvNodeServer::RamRangePrepForKey(
    const BlockKey& key, uint64_t offset, uint64_t length,
    const char** out_ptr, size_t* out_len, uint64_t* out_token) {
  if (!ram_) return false;
  RamTier::Hit h;
  if (!ram_->GetPrep(key, offset, length, &h)) return false;
  if (out_ptr) *out_ptr = h.ptr;
  if (out_len) *out_len = h.len;
  if (out_token) *out_token = h.token;
  cache_hit_.fetch_add(1, std::memory_order_relaxed);
  bytes_read_.fetch_add(h.len, std::memory_order_relaxed);
  return true;
}

void KvNodeServer::RamRelease(uint64_t token) {
  if (ram_) ram_->Release(token);
}

// Keep-alive: serve requests on this connection until the peer closes it.
void KvNodeServer::Handle(int fd) {
  // Bound the declared payload_len BEFORE the allocation below: without this
  // a forged 42-byte prefix is a 16 GiB allocation. Same bound as the RDMA
  // path (utils/wire_limits.h).
  const uint64_t max_payload = max_request_payload_
                                   ? max_request_payload_
                                   : wire_limits::MaxRequestPayload();
  while (running_) {
    char prefix[kReqPrefix];
    if (!net::ReadAll(fd, prefix, kReqPrefix)) return;  // peer closed / error
    ReqFields rq;
    if (!DecodeReq(prefix, &rq, max_payload)) return;  // bad version / oversize => drop
    std::vector<char> payload(rq.payload_len);
    if (rq.payload_len && !net::ReadAll(fd, payload.data(), rq.payload_len)) return;

    std::string data;
    Status st = ProcessRequestForKey(
        rq.op, rq.Key(), rq.offset, rq.length, payload.data(), rq.payload_len,
        &data);

    char rp[kRespPrefix];
    EncodeResp(rp, st, data.size());
    if (!net::WriteAll(fd, rp, kRespPrefix)) return;
    if (!data.empty() && !net::WriteAll(fd, data.data(), data.size())) return;
  }
}

}  // namespace dfkv
