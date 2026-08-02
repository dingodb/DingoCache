#include "transport/tcp_transport.h"

#include "common/config_dump.h"
#include "transport/endpoint_schedule.h"
#include "utils/wire_limits.h"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <string>

#include "utils/net_util.h"

namespace dfkv {

namespace {

// Do one request/response on an existing fd. Returns false on any TRANSPORT
// failure (caller should drop the connection); on success sets *st to the
// server's response status (which may itself be NotFound etc.).
bool OneShot(int fd, WireOp op, const BlockKey& k, uint64_t offset,
             uint64_t length, const void* payload, uint64_t payload_len,
             Status* st, std::string* out, uint64_t max_data,
             uint64_t* value_len) {
  char prefix[kReqPrefix];
  EncodeReq(prefix, op, k, offset, length, payload_len);
  if (!net::WriteAll(fd, prefix, kReqPrefix)) return false;
  if (payload_len && !net::WriteAll(fd, payload, payload_len)) return false;

  char rp[kRespPrefix];
  if (!net::ReadAll(fd, rp, kRespPrefix)) return false;
  uint64_t dlen = 0;
  // max_data caps the server-declared response length BEFORE the allocation
  // below (the caller knows what it asked for; a corrupt/hostile peer must
  // not be able to declare a 16 GiB body).
  if (!DecodeResp(rp, st, &dlen, max_data, value_len))
    return false;  // bad version / oversize
  if (dlen) {
    std::string data(dlen, '\0');
    if (!net::ReadAll(fd, &data[0], dlen)) return false;
    if (out) *out = std::move(data);
  } else if (out) {
    out->clear();
  }
  return true;
}

// Range variant that receives raw payload bytes straight into caller memory.
// The response prefix is still validated before the first payload byte is read,
// so a peer cannot overrun dst by lying about data_len.
bool OneShotInto(int fd, const BlockKey& key, uint64_t offset, uint64_t length,
                 void* dst, size_t dst_cap, Status* st,
                 uint64_t* value_len) {
  char prefix[kReqPrefix];
  EncodeReq(prefix, WireOp::kRange, key, offset, length, 0);
  if (!net::WriteAll(fd, prefix, kReqPrefix)) return false;

  char response[kRespPrefix];
  if (!net::ReadAll(fd, response, kRespPrefix)) return false;
  uint64_t data_len = 0;
  if (!DecodeResp(response, st, &data_len, dst_cap, value_len)) return false;
  if (data_len == 0) return true;
  if (dst == nullptr) return false;
  return net::ReadAll(fd, dst, static_cast<size_t>(data_len));
}
uint64_t PoolEnv(const char* name, uint64_t dflt, uint64_t lo, uint64_t hi) {
  const char* value = std::getenv(name);
  uint64_t resolved = dflt;
  bool from_env = false;
  if (value && *value) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end != value && *end == '\0' && parsed >= lo && parsed <= hi) {
      resolved = static_cast<uint64_t>(parsed);
      from_env = true;
    }
  }
  config_dump::Record(name, std::to_string(resolved),
                      from_env ? config_dump::Source::kEnv
                               : config_dump::Source::kDefault);
  return resolved;
}
}  // namespace

TcpTransport::Connection::~Connection() {
  if (fd >= 0) ::close(fd);
}

EndpointPoolOptions TcpTransport::NormalizeOptions(EndpointPoolOptions options) {
  options.max_connections =
      std::clamp<size_t>(options.max_connections, 1, 64);
  options.min_connections =
      std::min(options.min_connections, options.max_connections);
  options.idle_timeout_ms =
      std::clamp<uint64_t>(options.idle_timeout_ms, 1, 3600000);
  options.backoff_base_ms =
      std::clamp<uint64_t>(options.backoff_base_ms, 1, 60000);
  options.backoff_max_ms =
      std::clamp<uint64_t>(options.backoff_max_ms,
                           options.backoff_base_ms, 600000);
  options.acquire_timeout_ms =
      std::clamp<uint64_t>(options.acquire_timeout_ms, 1, 600000);
  return options;
}

EndpointPoolOptions TcpTransport::OptionsFromEnv() {
  EndpointPoolOptions options;
  options.max_connections = static_cast<size_t>(
      PoolEnv("DFKV_TCP_POOL_MAX_CONNS", 8, 1, 64));
  options.min_connections = static_cast<size_t>(
      PoolEnv("DFKV_TCP_POOL_MIN_CONNS", 1, 0, options.max_connections));
  options.idle_timeout_ms =
      PoolEnv("DFKV_TCP_POOL_IDLE_MS", 30000, 1, 3600000);
  options.backoff_base_ms =
      PoolEnv("DFKV_TCP_POOL_BACKOFF_BASE_MS", 50, 1, 60000);
  options.backoff_max_ms = PoolEnv(
      "DFKV_TCP_POOL_BACKOFF_MAX_MS",
      std::max<uint64_t>(5000, options.backoff_base_ms),
      options.backoff_base_ms, 600000);
  options.acquire_timeout_ms =
      PoolEnv("DFKV_TCP_POOL_ACQUIRE_TIMEOUT_MS", 10000, 1, 600000);
  return NormalizeOptions(options);
}

TcpTransport::TcpTransport() : TcpTransport(OptionsFromEnv()) {}

TcpTransport::TcpTransport(EndpointPoolOptions options)
    : options_(NormalizeOptions(options)) {}

TcpTransport::~TcpTransport() {
  std::lock_guard<std::mutex> lk(mu_);
  pools_.clear();
  pool_connections_.store(0, std::memory_order_relaxed);
  pool_inflight_.store(0, std::memory_order_relaxed);
}

void TcpTransport::ApplyBackoffLocked(NodePool* pool, Clock::time_point now) {
  const uint32_t streak = ++pool->failure_streak;
  const unsigned shift = std::min<unsigned>(streak - 1, 20);
  uint64_t delay = options_.backoff_base_ms;
  if (delay <= (std::numeric_limits<uint64_t>::max() >> shift))
    delay <<= shift;
  else
    delay = options_.backoff_max_ms;
  delay = std::min(delay, options_.backoff_max_ms);
  pool->backoff_until = now + std::chrono::milliseconds(delay);
  pool_backoff_events_.fetch_add(1, std::memory_order_relaxed);
}

void TcpTransport::RetireIdleLocked(NodePool* pool, Clock::time_point now) {
  while (pool->connections.size() > options_.min_connections) {
    auto victim = pool->connections.end();
    for (auto it = pool->connections.begin(); it != pool->connections.end();
         ++it) {
      const auto& connection = *it;
      if (connection->inflight != 0) continue;
      const auto idle = std::chrono::duration_cast<std::chrono::milliseconds>(
          now - connection->last_used);
      if (idle.count() < static_cast<int64_t>(options_.idle_timeout_ms))
        continue;
      if (victim == pool->connections.end() ||
          connection->last_used < (*victim)->last_used ||
          (connection->last_used == (*victim)->last_used &&
           connection->id < (*victim)->id)) {
        victim = it;
      }
    }
    if (victim == pool->connections.end()) break;
    pool->connections.erase(victim);
    pool_connections_.fetch_sub(1, std::memory_order_relaxed);
    pool_retired_idle_.fetch_add(1, std::memory_order_relaxed);
  }
}

bool TcpTransport::Acquire(const std::string& node, Lease* lease) {
  const auto deadline =
      Clock::now() + std::chrono::milliseconds(options_.acquire_timeout_ms);
  for (;;) {
    bool dial = false;
    {
      std::unique_lock<std::mutex> lk(mu_);
      NodePool& pool = pools_[node];
      const auto now = Clock::now();
      RetireIdleLocked(&pool, now);

      std::vector<EndpointConnectionLoad> loads;
      loads.reserve(pool.connections.size());
      for (const auto& connection : pool.connections) {
        loads.push_back(EndpointConnectionLoad{
            connection->id, connection->inflight,
            connection->latency_ns, true});
      }
      const size_t selected = SelectEndpointConnection(loads);
      if (selected != loads.size() && loads[selected].inflight == 0) {
        auto connection = pool.connections[selected];
        connection->inflight = 1;
        lease->connection = std::move(connection);
        lease->reused = true;
        lease->started = now;
        pool_inflight_.fetch_add(1, std::memory_order_relaxed);
        pool_selections_.fetch_add(1, std::memory_order_relaxed);
        return true;
      }

      if (pool.connections.size() + pool.creating <
              options_.max_connections &&
          now >= pool.backoff_until) {
        ++pool.creating;
        dial = true;
      } else {
        if (pool.connections.empty() && now < pool.backoff_until) {
          pool_backoff_suppressed_.fetch_add(1, std::memory_order_relaxed);
          return false;
        }
        if (now >= deadline) return false;
        auto wake = deadline;
        if (now < pool.backoff_until)
          wake = std::min(wake, pool.backoff_until);
        cv_.wait_until(lk, wake);
      }
    }
    if (!dial) continue;

    const int fd = net::Dial(node, connect_ms_, io_ms_);
    bool empty_after_failure = false;
    {
      std::lock_guard<std::mutex> lk(mu_);
      NodePool& pool = pools_[node];
      --pool.creating;
      const auto now = Clock::now();
      if (fd < 0) {
        ApplyBackoffLocked(&pool, now);
        empty_after_failure = pool.connections.empty();
      } else {
        pool.failure_streak = 0;
        pool.backoff_until = Clock::time_point{};
        auto connection =
            std::make_shared<Connection>(fd, next_connection_id_++, now);
        connection->inflight = 1;
        pool.connections.push_back(connection);
        lease->connection = std::move(connection);
        lease->reused = false;
        lease->started = now;
        pool_connections_.fetch_add(1, std::memory_order_relaxed);
        pool_inflight_.fetch_add(1, std::memory_order_relaxed);
        pool_selections_.fetch_add(1, std::memory_order_relaxed);
      }
    }
    cv_.notify_all();
    if (fd >= 0) return true;
    if (empty_after_failure) return false;
  }
}

void TcpTransport::Complete(const std::string& node, Lease lease, bool healthy,
                            bool apply_backoff) {
  const auto now = Clock::now();
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto pit = pools_.find(node);
    if (pit == pools_.end()) return;
    NodePool& pool = pit->second;
    auto it = std::find(pool.connections.begin(), pool.connections.end(),
                        lease.connection);
    if (it == pool.connections.end()) return;

    lease.connection->inflight = 0;
    pool_inflight_.fetch_sub(1, std::memory_order_relaxed);
    if (healthy) {
      uint64_t sample = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              now - lease.started).count());
      sample = std::max<uint64_t>(sample, 1);
      if (lease.connection->latency_ns == 0)
        lease.connection->latency_ns = sample;
      else
        lease.connection->latency_ns =
            (lease.connection->latency_ns * 7 + sample) / 8;
      lease.connection->last_used = now;
      pool.failure_streak = 0;
      pool.backoff_until = Clock::time_point{};
    } else {
      pool.connections.erase(it);
      pool_connections_.fetch_sub(1, std::memory_order_relaxed);
      pool_retired_error_.fetch_add(1, std::memory_order_relaxed);
      if (apply_backoff) ApplyBackoffLocked(&pool, now);
    }
  }
  cv_.notify_all();
}

Status TcpTransport::RoundTrip(const std::string& node, WireOp op,
                               const BlockKey& k, uint64_t offset,
                               uint64_t length, const void* payload,
                               uint64_t payload_len, std::string* out,
                               uint64_t max_data, uint64_t* value_len) {
  // A stale reused socket is retired and retried once. A fresh socket failure
  // starts endpoint dial backoff and is returned immediately.
  for (int attempt = 0; attempt < 2; ++attempt) {
    Lease lease;
    if (!Acquire(node, &lease)) return Status::kIOError;
    Status st = Status::kIOError;
    if (OneShot(lease.connection->fd, op, k, offset, length, payload,
                payload_len, &st, out, max_data, value_len)) {
      Complete(node, std::move(lease), true, false);
      return st;
    }
    const bool retry = lease.reused;
    Complete(node, std::move(lease), false, !retry);
    if (!retry) return Status::kIOError;
  }
  return Status::kIOError;
}

Status TcpTransport::RoundTripInto(const std::string& node,
                                   const BlockKey& key, uint64_t offset,
                                   uint64_t length, void* dst, size_t dst_cap,
                                   uint64_t* value_len) {
  for (int attempt = 0; attempt < 2; ++attempt) {
    Lease lease;
    if (!Acquire(node, &lease)) return Status::kIOError;
    Status st = Status::kIOError;
    if (OneShotInto(lease.connection->fd, key, offset, length, dst, dst_cap,
                    &st, value_len)) {
      Complete(node, std::move(lease), true, false);
      return st;
    }
    const bool retry = lease.reused;
    Complete(node, std::move(lease), false, !retry);
    if (!retry) return Status::kIOError;
  }
  return Status::kIOError;
}

Status TcpTransport::Cache(const std::string& node, const BlockKey& key,
                           const void* data, size_t len) {
  return RoundTrip(node, WireOp::kCache, key, 0, 0, data, len, nullptr,
                   wire_limits::kStatusMaxRespData);
}

Status TcpTransport::Range(const std::string& node, const BlockKey& key,
                           uint64_t offset, uint64_t length, std::string* out,
                           uint64_t* value_len) {
  // A Range reply carries at most the requested byte count.
  return RoundTrip(node, WireOp::kRange, key, offset, length, nullptr, 0, out,
                   length, value_len);
}

std::vector<Status> TcpTransport::RangeInto(
    const std::string& node, const std::vector<BlockKey>& keys,
    const std::vector<RangeDst>& dsts,
    std::vector<uint64_t>* value_lens) {
  if (keys.size() != dsts.size())
    return std::vector<Status>(keys.size(), Status::kInvalid);
  if (value_lens) value_lens->assign(keys.size(), 0);
  std::vector<Status> statuses;
  statuses.reserve(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    uint64_t stored = 0;
    statuses.push_back(RoundTripInto(node, keys[i], 0, dsts[i].n,
                                     dsts[i].payload, dsts[i].n, &stored));
    if (value_lens) (*value_lens)[i] = stored;
  }
  return statuses;
}

Status TcpTransport::Lookup(const std::string& node, const BlockKey& key,
                            uint64_t* value_len) {
  if (value_len == nullptr) return Status::kInvalid;
  *value_len = 0;
  return RoundTrip(node, WireOp::kLookup, key, 0, 0, nullptr, 0, nullptr,
                   wire_limits::kStatusMaxRespData, value_len);
}

Status TcpTransport::Stats(const std::string& node, std::string* out) {
  return RoundTrip(node, WireOp::kStats, BlockKey{}, 0, 0, nullptr, 0, out,
                   wire_limits::kTextMaxRespData);
}

Status TcpTransport::Members(const std::string& node, std::string* out) {
  return RoundTrip(node, WireOp::kMembers, BlockKey{}, 0, 0, nullptr, 0, out,
                   wire_limits::kTextMaxRespData);
}

Status TcpTransport::Exist(const std::string& node, const BlockKey& key,
                           bool* exist) {
  Status st = RoundTrip(node, WireOp::kExist, key, 0, 0, nullptr, 0, nullptr,
                        wire_limits::kStatusMaxRespData);
  if (st == Status::kOk) { *exist = true; return Status::kOk; }
  if (st == Status::kNotFound) { *exist = false; return Status::kOk; }
  return st;
}

Status TcpTransport::Remove(const std::string& node, const BlockKey& key) {
  // kRemove is a key-only request (no payload), Status-only response: kOk when a
  // block was dropped, kNotFound when it was already absent (both fine for the
  // caller), kIOError on a transport failure.
  return RoundTrip(node, WireOp::kRemove, key, 0, 0, nullptr, 0, nullptr,
                   wire_limits::kStatusMaxRespData);
}

std::string TcpTransport::MetricsText() const {
  uint64_t backing_off = 0;
  {
    std::lock_guard<std::mutex> lk(mu_);
    const auto now = Clock::now();
    for (const auto& entry : pools_)
      if (now < entry.second.backoff_until) ++backing_off;
  }
  std::string out;
  auto metric = [&out](const char* name, const char* type,
                       const char* help, uint64_t value,
                       const char* labels = "") {
    out += "# HELP "; out += name; out += " "; out += help; out += "\n";
    out += "# TYPE "; out += name; out += " "; out += type; out += "\n";
    out += name; out += labels; out += " "; out += std::to_string(value);
    out += "\n";
  };
  metric("dfkv_transport_pool_connections", "gauge",
         "Open pooled transport connections",
         pool_connections_.load(std::memory_order_relaxed));
  metric("dfkv_transport_pool_inflight", "gauge",
         "Transport operations currently holding a pooled connection",
         pool_inflight_.load(std::memory_order_relaxed));
  metric("dfkv_transport_pool_selections_total", "counter",
         "Pooled connection selections for routed endpoints",
         pool_selections_.load(std::memory_order_relaxed));
  out += "# HELP dfkv_transport_pool_retirements_total "
         "Pooled connections retired by reason\n";
  out += "# TYPE dfkv_transport_pool_retirements_total counter\n";
  out += "dfkv_transport_pool_retirements_total{reason=\"idle\"} ";
  out += std::to_string(
      pool_retired_idle_.load(std::memory_order_relaxed));
  out += "\n";
  out += "dfkv_transport_pool_retirements_total{reason=\"error\"} ";
  out += std::to_string(
      pool_retired_error_.load(std::memory_order_relaxed));
  out += "\n";
  metric("dfkv_transport_pool_backoff_endpoints", "gauge",
         "Routed endpoints whose connection growth is in backoff",
         backing_off);
  metric("dfkv_transport_pool_backoff_events_total", "counter",
         "Connection failures that advanced endpoint backoff",
         pool_backoff_events_.load(std::memory_order_relaxed));
  metric("dfkv_transport_pool_backoff_suppressed_total", "counter",
         "Connection acquisitions fast-failed while an endpoint was in backoff",
         pool_backoff_suppressed_.load(std::memory_order_relaxed));
  return out;
}

}  // namespace dfkv
