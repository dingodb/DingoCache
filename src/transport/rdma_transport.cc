#include "common/config_dump.h"
#include "transport/rdma_transport.h"

#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#include "utils/log.h"
#include "utils/net_util.h"     // Dial / WriteAll / ReadAll / Put*/Get*
#include "utils/numa_util.h"
#include "transport/rdma_topology.h"
#include "transport/rail_classify.h"
#include "transport/rail_select.h"
#include "transport/rdma_verbs.h"   // RcEndpoint, QpInfo
#include "transport/rdma_protocol.h"

namespace dfkv {

namespace {
int EnvInt(const char* name, int dflt) {
  const char* v = std::getenv(name);
  if (!v || !*v) return dflt;
  long x = std::strtol(v, nullptr, 10);
  return x > 0 ? static_cast<int>(x) : dflt;
}

int EnvBoundedInt(const char* name, int dflt, int upper) {
  const char* v = std::getenv(name);
  if (!v || !*v) return dflt;
  errno = 0;
  char* end = nullptr;
  long x = std::strtol(v, &end, 10);
  if (errno != 0 || end == v || *end != '\0' || x <= 0) return dflt;
  return static_cast<int>(std::min<long>(x, upper));
}

size_t EnvBytes(const char* name, size_t dflt) {
  const char* v = std::getenv(name);
  if (!v || !*v) return dflt;
  errno = 0;
  char* end = nullptr;
  unsigned long long x = std::strtoull(v, &end, 10);
  if (errno != 0 || end == v || x == 0) return dflt;
  constexpr unsigned long long kMaxSge =
      static_cast<unsigned long long>(std::numeric_limits<uint32_t>::max() -
                                      2 * rdma::kDirectIoAlign);
  if (x > kMaxSge) x = kMaxSge;
  return static_cast<size_t>(x);
}

size_t ResolveMaxPayload(size_t configured) {
  size_t n = configured ? configured : (64u << 20);
  n = EnvBytes("DFKV_RDMA_MAX_PAYLOAD_BYTES", n);
  // dbuf/SGE length is uint32; keep room for direct-I/O alignment rounding.
  constexpr size_t kMaxSge = static_cast<size_t>(
      std::numeric_limits<uint32_t>::max() - 2 * rdma::kDirectIoAlign);
  if (n > kMaxSge) n = kMaxSge;
  return n;
}


std::vector<std::string> ParseDeviceList(const std::string& list) {
  std::vector<std::string> devices;
  for (size_t i = 0; i <= list.size();) {
    size_t comma = list.find(',', i);
    if (comma == std::string::npos) comma = list.size();
    size_t begin = i;
    size_t end = comma;
    while (begin < end &&
           std::isspace(static_cast<unsigned char>(list[begin])))
      ++begin;
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(list[end - 1])))
      --end;
    if (end > begin) devices.push_back(list.substr(begin, end - begin));
    i = comma + 1;
  }
  return devices;
}

std::string JoinDevices(const std::vector<std::string>& devices) {
  std::string out;
  for (const auto& device : devices) {
    if (!out.empty()) out += ",";
    out += device;
  }
  return out;
}

// Local post/CQ/verbs API failures happen below the work-completion layer, so
// no real WC status exists for them. The reap helpers report such failures as
// this local-fault bucket so every datapath teardown can funnel through
// rdma::ClassifyCompletion and attribute them kRailFailure.
constexpr ibv_wc_status kLocalFaultWcStatus = IBV_WC_GENERAL_ERR;

// Reap one signaled request completion and one status RECV per posted request.
// Every poll consumes the same absolute deadline; a partial CQ drain cannot
// restart the timeout budget. slot_count may exceed posted when invalid items
// were skipped inside a mixed window.
// On failure the evidence out-params describe what killed the window:
// worst_status is the first non-SUCCESS reaped status (IBV_WC_SUCCESS when no
// error WC was observed — deadline expiry or a bogus wr_id), and
// had_completions whether ANY completion events arrived. Classify them with
// rdma::ClassifyCompletion before Destroying the connection: a reaped
// remote-side status or a silent deadline is peer evidence, never rail
// evidence. A WaitComp CQ/verbs API failure surfaces as kLocalFaultWcStatus.
bool ReapPosted(rdma::RcEndpoint& ep, size_t posted, size_t slot_count,
                std::vector<uint32_t>* rbytes, int timeout_ms,
                bool* timed_out = nullptr,
                ibv_wc_status* worst_status = nullptr,
                bool* had_completions = nullptr) {
  if (timed_out) *timed_out = false;
  if (worst_status) *worst_status = IBV_WC_SUCCESS;
  if (had_completions) *had_completions = false;
  rbytes->assign(slot_count, 0);
  if (posted == 0) return true;
  std::vector<ibv_wc> wcs(2 * posted);
  int need = static_cast<int>(2 * posted);
  CompletionDeadline deadline(timeout_ms);
  while (need > 0) {
    const int remaining_ms = deadline.Remaining();
    if (remaining_ms == 0) {
      if (timed_out) *timed_out = true;
      return false;
    }
    const int got = ep.WaitComp(wcs.data(), static_cast<int>(wcs.size()),
                                remaining_ms);
    if (got <= 0) {
      if (got == 0 && timed_out) *timed_out = true;
      if (got < 0 && worst_status) {
        // CQ notification/polling itself failed: local rail evidence (the
        // client never Wake()s a live endpoint; Wake is server shutdown only).
        *worst_status = kLocalFaultWcStatus;
      }
      return false;
    }
    if (had_completions) *had_completions = true;
    for (int i = 0; i < got; ++i) {
      if (wcs[i].status != IBV_WC_SUCCESS) {
        if (worst_status) *worst_status = wcs[i].status;
        return false;
      }
      if (wcs[i].opcode == IBV_WC_RECV) {
        const size_t slot = static_cast<size_t>(wcs[i].wr_id);
        if (slot >= slot_count) return false;
        (*rbytes)[slot] = wcs[i].byte_len;
      }
      --need;
    }
  }
  return true;
}

bool ReapWindow(rdma::RcEndpoint& ep, size_t width,
                std::vector<uint32_t>* rbytes, int timeout_ms,
                bool* timed_out = nullptr,
                ibv_wc_status* worst_status = nullptr,
                bool* had_completions = nullptr) {
  return ReapPosted(ep, width, width, rbytes, timeout_ms, timed_out,
                    worst_status, had_completions);
}

// Post ordinary SEND requests with one status RECV per slot.
bool RunWindow(rdma::RcEndpoint& ep, const std::vector<size_t>& slen,
               std::vector<uint32_t>* rbytes, int timeout_ms,
               bool* timed_out = nullptr,
               ibv_wc_status* worst_status = nullptr,
               bool* had_completions = nullptr) {
  for (size_t j = 0; j < slen.size(); ++j) {
    if (!ep.PostRecv(j) || !ep.PostSend(j, slen[j])) {
      // A post failure is local verbs evidence: no WC exists to reap, so
      // report the local-fault bucket for uniform classification.
      if (worst_status) *worst_status = kLocalFaultWcStatus;
      if (had_completions) *had_completions = false;
      return false;
    }
  }
  return ReapWindow(ep, slen.size(), rbytes, timeout_ms, timed_out,
                    worst_status, had_completions);
}
}  // namespace

struct RdmaTransport::Conn {
  rdma::RcEndpoint ep;
  rdma::RecvSegmentInfo recv_segment;
  size_t rail_index = 0;
  rdma::RailLease lease;
  uint64_t lease_started_us = 0;
  bool credit_held = false;

  void Encode(char* out, WireOp op, const BlockKey& key, uint64_t offset,
              uint64_t length, uint64_t payload_len) const {
    EncodeReqVersion(out, kNativeProtoRdmaV2, op, key, offset, length,
                     payload_len);
  }
  bool Decode(const char* in, Status* status, uint64_t* data_len,
              uint64_t max_data = kMaxFrameLen,
              uint64_t* value_len = nullptr) const {
    return DecodeRespVersion(in, kNativeProtoRdmaV2, status, data_len, max_data,
                             value_len);
  }
  size_t data_capacity() const {
    return recv_segment.slot_size >= rdma::kV2DataOffset
               ? static_cast<size_t>(recv_segment.slot_size -
                                     rdma::kV2DataOffset)
               : 0;
  }
  uint64_t put_addr(size_t slot) const {
    return recv_segment.base_addr +
           slot * recv_segment.slot_size + rdma::kV2PutPrefixOffset;
  }
};

bool RdmaTransport::Available() {
  const char* configured = std::getenv("DFKV_RDMA_DEV");
  const auto filter =
      ParseDeviceList(configured ? std::string(configured) : std::string());
  return !rdma::RdmaTopology::Discover(filter).empty();
}

RdmaTransport::RdmaTransport(size_t max_msg, const std::string& dev_name)
    : max_payload_(ResolveMaxPayload(max_msg)),
      declared_(std::min<uint64_t>(
          EnvBytes("DFKV_RDMA_MAX_BLOCK_BYTES", 4ull << 20),
          ResolveMaxPayload(max_msg))),
      depth_(4) {
  std::string list = dev_name;
  if (list.empty()) {
    const char* configured = std::getenv("DFKV_RDMA_DEV");
    if (configured) list = configured;
  }
  const auto filter = ParseDeviceList(list);
  // A configured rail name is announced to the peer inside the fixed-size v2
  // bootstrap frame (EncodeDevFrame at Acquire). One that does not fit would
  // silently lose the "DCP2" capability trailer: capability probes (short
  // probe name) keep succeeding while every real connection is rejected as
  // "missing v2 negotiation". Fail here, at startup, with the actual cause
  // and limit. Auto-discovery needs no check: it announces an empty name.
  for (const auto& device : filter) {
    if (!rdma::DeviceNameFitsFrame(device)) {
      const std::string error = rdma::OversizedDeviceNameError(device);
      DFKV_LOG_ERROR(error);
      throw std::runtime_error(error);
    }
  }
  auto_device_ = filter.empty();
  auto discovered = rdma::RdmaTopology::Discover(filter);
  if (discovered.empty()) {
    throw std::runtime_error(
        filter.empty() ? "no ACTIVE RDMA device found"
                       : "no configured RDMA device is ACTIVE");
  }
  if (auto_device_ && discovered.size() > 1) discovered.resize(1);
  devs_.reserve(discovered.size());
  for (const auto& device : discovered) devs_.push_back(device.name);
  topology_ =
      std::make_unique<rdma::RdmaTopology>(std::move(discovered));
  numa_aware_ = numa::Enabled();
  // Live SG width: the tightest rail decides (connections round-robin rails,
  // and a key must fit whichever rail carries it). Queried once — device caps
  // don't change under a running process.
  {
    size_t msge = rdma::kMaxSge;
    for (const auto& d : devs_)
      msge = std::min(msge, rdma::QueryMaxSge(d.empty() ? nullptr : d.c_str()));
    sg_payload_segs_ = msge - 1;  // SGE0 carries the wire request prefix
  }
  const char* d = std::getenv("DFKV_RDMA_DEPTH");  // pipeline depth (default 4; must be <= server's)
  if (d && *d) { long v = std::strtol(d, nullptr, 10); if (v >= 1 && v <= 256) depth_ = (size_t)v; }
  config_dump::RecordResolved("DFKV_RDMA_DEV", JoinDevices(devs_));
  config_dump::RecordResolved("DFKV_RDMA_DEPTH", std::to_string(depth_));
  config_dump::RecordResolved("DFKV_RDMA_NUMA", numa_aware_ ? "1" : "0");
  rdma::RailPolicyConfig rail_config;
  rail_config.credits_per_rail = static_cast<uint32_t>(
      EnvBoundedInt("DFKV_RDMA_RAIL_CREDITS", 64, 4096));
  rail_config.error_threshold = static_cast<uint32_t>(
      EnvBoundedInt("DFKV_RDMA_RAIL_ERROR_THRESHOLD", 3, 1000));
  const int cooldown_ms =
      EnvBoundedInt("DFKV_RDMA_RAIL_COOLDOWN_MS", 5000, 600000);
  rail_config.quarantine_us = static_cast<uint64_t>(cooldown_ms) * 1000;
  rail_config.latency_weight = static_cast<uint32_t>(
      EnvBoundedInt("DFKV_RDMA_RAIL_LATENCY_WEIGHT", 1, 1000));
  rail_config.error_penalty_us = static_cast<uint64_t>(
      EnvBoundedInt("DFKV_RDMA_RAIL_ERROR_PENALTY_US", 100000, 60000000));
  const int backpressure_ms =
      EnvBoundedInt("DFKV_RDMA_RAIL_BACKPRESSURE_MS", 10, 60000);
  rail_backpressure_us_ = static_cast<uint64_t>(backpressure_ms) * 1000;
  rail_policy_ =
      std::make_unique<rdma::RailPolicy>(devs_.size(), rail_config);
  config_dump::RecordResolved("DFKV_RDMA_RAIL_CREDITS",
                              std::to_string(rail_config.credits_per_rail));
  config_dump::RecordResolved(
      "DFKV_RDMA_RAIL_ERROR_THRESHOLD",
      std::to_string(rail_config.error_threshold));
  config_dump::RecordResolved("DFKV_RDMA_RAIL_COOLDOWN_MS",
                              std::to_string(cooldown_ms));
  config_dump::RecordResolved(
      "DFKV_RDMA_RAIL_LATENCY_WEIGHT",
      std::to_string(rail_config.latency_weight));
  config_dump::RecordResolved(
      "DFKV_RDMA_RAIL_ERROR_PENALTY_US",
      std::to_string(rail_config.error_penalty_us));
  config_dump::RecordResolved("DFKV_RDMA_RAIL_BACKPRESSURE_MS",
                              std::to_string(backpressure_ms));
  connect_ms_ = EnvInt("DFKV_RDMA_CONNECT_MS", 3000);
  io_ms_ = EnvInt("DFKV_RDMA_IO_MS", 10000);
  // An explicit zero disables the single-op datapath completion deadline.
  {
    const char* value = std::getenv("DFKV_RDMA_OP_TIMEOUT_MS");
    if (value && *value) {
      const long parsed = std::strtol(value, nullptr, 10);
      op_timeout_ms_ =
          parsed == 0
              ? -1
              : (parsed > 0
                     ? static_cast<int>(std::min<long>(
                           parsed, std::numeric_limits<int>::max()))
                     : 5000);
    }
  }
  {
    const char* value = std::getenv("DFKV_RDMA_BATCH_OP_TIMEOUT_MS");
    if (value && *value) {
      const long parsed = std::strtol(value, nullptr, 10);
      if (parsed > 0) {
        batch_op_timeout_ms_ = static_cast<int>(std::min<long>(
            parsed, std::numeric_limits<int>::max()));
      }
    }
  }
  config_dump::RecordResolved("DFKV_RDMA_OP_TIMEOUT_MS",
                              std::to_string(op_timeout_ms_));
  config_dump::RecordResolved("DFKV_RDMA_BATCH_OP_TIMEOUT_MS",
                              std::to_string(batch_op_timeout_ms_));
  // Idle-connection pool cap. The pool naturally bounds at peak concurrency
  // (each thread holds <=1 conn); this only guards against a thread-count spike
  // leaving many idle conns. Must be >= peak concurrency or releases churn
  // (destroy+recreate every op), which fails the bootstrap under load. Default
  // 256 covers typical fan-out; raise via DFKV_RDMA_POOL_MAX for more threads.
  pool_max_ = static_cast<size_t>(EnvInt("DFKV_RDMA_POOL_MAX", 256));
  rail_conns_ = std::make_unique<std::atomic<uint64_t>[]>(devs_.size());  // 0-initialized
}

RdmaTransport::~RdmaTransport() {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto& [node, cs] : pool_)
    for (Conn* c : cs) Destroy(c);
  for (auto& [node, cs] : control_pool_)
    for (Conn* c : cs) Destroy(c);
}

void RdmaTransport::Destroy(Conn* c, rdma::RailCompletion completion) {
  if (!c) return;
  if (c->credit_held) {
    const uint64_t now = rdma::RailPolicy::NowMicros();
    rail_policy_->Complete(c->lease, now - c->lease_started_us, completion,
                           now);
    c->credit_held = false;
  }
  delete c;  // RcEndpoint dtor tears down QP/MRs.
}

bool RdmaTransport::ProbeV2(const std::string& node) const {
  v2_probe_attempts_.fetch_add(1, std::memory_order_relaxed);
  int fd = net::Dial(node, connect_ms_, io_ms_);
  if (fd < 0) {
    v2_probe_failures_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  char frame[rdma::kDevNameBytes];
  rdma::EncodeDevFrame(rdma::kV2ProbeDevice, declared_, frame,
                       rdma::kDevProtoV2);
  char reply[rdma::kV2ProbeReplyBytes];
  const bool ok =
      net::WriteAll(fd, frame, sizeof(frame)) &&
      net::ReadAll(fd, reply, sizeof(reply)) &&
      rdma::ParseV2ProbeReply(reply);
  ::close(fd);
  if (!ok) v2_probe_failures_.fetch_add(1, std::memory_order_relaxed);
  return ok;
}

RdmaTransport::Conn* RdmaTransport::Acquire(
    const std::string& node, Lane lane, bool* from_pool, bool force_new,
    size_t requested_credits) {
  *from_pool = false;
  const uint32_t requested = static_cast<uint32_t>(
      std::min<size_t>(requested_credits,
                       std::numeric_limits<uint32_t>::max()));
  // Selection is global across configured rails: first constrain candidates to
  // the caller's local NUMA rails when that topology exists, then choose least
  // normalized inflight with latency/error penalties and stable device index as
  // the final tie-break. Unknown/no-local topology falls back to all rails.
  // When every NUMA-local rail is inadmissible (quarantined or credit-bound),
  // admission retries once across every enabled rail — locality is a
  // preference, never an availability gate.
  const int caller_node = numa_aware_ ? numa::CurrentNode() : -1;
  const auto candidates =
      topology_->CandidatesFor(caller_node, numa_aware_);
  if (candidates.locality == rdma::RailLocality::kCallerUnknown) {
    numa_caller_unknown_fallbacks_.fetch_add(1, std::memory_order_relaxed);
  } else if (candidates.locality == rdma::RailLocality::kNoLocal) {
    numa_no_local_fallbacks_.fetch_add(1, std::memory_order_relaxed);
  }
  auto lease = rdma::AcquireWithFallback(*rail_policy_, requested,
                                         rail_backpressure_us_,
                                         candidates.allowed,
                                         candidates.fallback);
  if (!lease) return nullptr;

  const size_t ridx = lease->rail;
  const uint64_t lease_started = rdma::RailPolicy::NowMicros();
  const auto fail_endpoint_lease = [&] {
    const uint64_t now = rdma::RailPolicy::NowMicros();
    rail_policy_->Complete(*lease, now - lease_started,
                           rdma::RailCompletion::kEndpointFailure, now);
  };

  std::vector<std::pair<void*, size_t>> pools;
  Conn* pooled = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    pools = pools_;
    if (!force_new) {
      auto& idle = lane == Lane::kData ? pool_ : control_pool_;
      auto it = idle.find(node);
      if (it != idle.end()) {
        auto& candidates = it->second;
        for (size_t i = candidates.size(); i > 0; --i) {
          if (candidates[i - 1]->rail_index == ridx) {
            pooled = candidates[i - 1];
            candidates.erase(candidates.begin() +
                             static_cast<std::ptrdiff_t>(i - 1));
            break;
          }
        }
      }
    }
  }
  if (pooled) {
    pooled->lease = *lease;
    pooled->lease_started_us = lease_started;
    pooled->credit_held = true;
    if (!pooled->ep.EnsurePoolMrs(pools, true)) {
      Destroy(pooled, rdma::RailCompletion::kRailFailure);
      return nullptr;
    }
    *from_pool = true;
    return pooled;
  }

  const std::string& dev = devs_[ridx];
  if (!ProbeV2(node)) {
    DFKV_LOG_ERROR("rdma: peer " + node +
                   " does not support the required v2 protocol");
    fail_endpoint_lease();
    return nullptr;
  }

  int fd = net::Dial(node, connect_ms_, io_ms_);
  if (fd < 0) {
    fail_endpoint_lease();
    return nullptr;
  }
  auto* conn = new Conn();
  conn->rail_index = ridx;
  conn->lease = *lease;
  conn->lease_started_us = lease_started;
  conn->credit_held = true;
  if (!conn->ep.Open(dev.c_str(), rdma::kV2ControlCap, depth_)) {
    ::close(fd);
    DFKV_LOG_WARN("rdma: device " + dev +
                  " Open failed; rail failure policy will quarantine it");
    Destroy(conn, rdma::RailCompletion::kRailFailure);
    return nullptr;
  }
  { const char* bp = std::getenv("DFKV_RDMA_BUSY_POLL"); if (bp && *bp && *bp != '0') conn->ep.set_busy_poll(true); }

  const uint64_t conn_declared =
      lane == Lane::kControl
          ? static_cast<uint64_t>(rdma::kV2ControlCap)
          : declared_;
  char devbuf[rdma::kDevNameBytes];
  rdma::EncodeDevFrame(auto_device_ ? std::string() : dev, conn_declared,
                       devbuf, rdma::kDevProtoV2);
  char mine[rdma::kQpInfoBytes], peer[rdma::kQpInfoBytes];
  rdma::QpInfo my = conn->ep.Local();
  my.depth = static_cast<uint16_t>(std::min<size_t>(depth_, 256));
  my.protocol_version = rdma::kDevProtoV2;
  rdma::SerializeQpInfo(my, mine);
  if (!net::WriteAll(fd, devbuf, rdma::kDevNameBytes) ||
      !net::WriteAll(fd, mine, rdma::kQpInfoBytes) ||
      !net::ReadAll(fd, peer, rdma::kQpInfoBytes)) {
    ::close(fd);
    Destroy(conn, rdma::RailCompletion::kEndpointFailure);
    return nullptr;
  }
  const rdma::QpInfo remote = rdma::ParseQpInfo(peer);
  if (remote.protocol_version != rdma::kDevProtoV2 || remote.depth == 0) {
    DFKV_LOG_ERROR("rdma: peer " + node +
                   " returned an invalid v2 QP negotiation");
    ::close(fd);
    Destroy(conn, rdma::RailCompletion::kEndpointFailure);
    return nullptr;
  }
  if (!conn->ep.Connect(remote)) {
    DFKV_LOG_WARN("rdma: device " + dev +
                  " failed local QP transition during v2 negotiation");
    ::close(fd);
    Destroy(conn, rdma::RailCompletion::kRailFailure);
    return nullptr;
  }
  conn->ep.set_remote_depth(remote.depth);
  if (remote.depth < depth_)
    DFKV_LOG_INFO("rdma: server depth " + std::to_string(remote.depth) +
                  " < client depth " + std::to_string(depth_) +
                  ": batching window clamped to " +
                  std::to_string(remote.depth));

  char ready = 0;
  char encoded[rdma::kRecvSegmentInfoBytes];
  if (!net::ReadAll(fd, &ready, 1) || ready != 1 ||
      !net::ReadAll(fd, encoded, sizeof(encoded)) ||
      !rdma::DecodeRecvSegmentInfo(encoded, &conn->recv_segment)) {
    DFKV_LOG_ERROR("rdma: peer " + node +
                   " did not provide a valid v2 receive segment");
    ::close(fd);
    Destroy(conn, rdma::RailCompletion::kEndpointFailure);
    return nullptr;
  }
  const size_t expected_slot = rdma::V2SlotSize(
      static_cast<size_t>(std::max<uint64_t>(
          conn_declared, rdma::kV2DataOffset)));
  if (expected_slot == 0 ||
      conn->recv_segment.slot_size != expected_slot) {
    DFKV_LOG_ERROR("rdma: peer " + node +
                   " returned incompatible v2 receive-segment geometry");
    ::close(fd);
    Destroy(conn, rdma::RailCompletion::kEndpointFailure);
    return nullptr;
  }
  ::close(fd);
  if (!conn->ep.EnsurePoolMrs(pools, true)) {
    Destroy(conn, rdma::RailCompletion::kRailFailure);
    return nullptr;
  }
  conns_opened_.fetch_add(1, std::memory_order_relaxed);
  rail_conns_[ridx].fetch_add(1, std::memory_order_relaxed);
  return conn;
}

bool RdmaTransport::RegisterMemory(void* base, size_t size) {
  const auto address = reinterpret_cast<uintptr_t>(base);
  if (!base || size == 0 ||
      size > std::numeric_limits<uintptr_t>::max() - address) {
    mr_registration_rejections_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  std::lock_guard<std::mutex> lk(mu_);
  std::vector<std::pair<void*, size_t>> candidate = pools_;
  auto existing =
      std::find_if(candidate.begin(), candidate.end(),
                   [base](const auto& pool) { return pool.first == base; });
  const bool grew = existing != candidate.end() && size > existing->second;
  if (existing != candidate.end()) {
    if (size <= existing->second) return true;
    existing->second = size;
  } else {
    candidate.push_back({base, size});
  }

  // The first declaration creates one lifetime anchor per discovered ACTIVE
  // rail. Opening fewer than all rails is a rejected transaction: publishing
  // the declaration would otherwise make the gauge and future connections lie.
  std::vector<std::unique_ptr<rdma::RcEndpoint>> new_anchors;
  std::vector<rdma::RcEndpoint*> rails;
  if (anchors_.empty()) {
    new_anchors.reserve(devs_.size());
    for (const auto& device : devs_) {
      auto endpoint = std::make_unique<rdma::RcEndpoint>();
      if (!endpoint->Open(device.empty() ? nullptr : device.c_str(),
                          rdma::kV2ControlCap, 1)) {
        mr_registration_rejections_.fetch_add(1, std::memory_order_relaxed);
        DFKV_LOG_ERROR(
            "rdma: memory declaration rejected because an active rail "
            "could not be anchored");
        return false;
      }
      rails.push_back(endpoint.get());
      new_anchors.push_back(std::move(endpoint));
    }
  } else {
    if (anchors_.size() != devs_.size()) {
      mr_registration_rejections_.fetch_add(1, std::memory_order_relaxed);
      DFKV_LOG_ERROR(
          "rdma: memory declaration rejected because the active rail anchor "
          "set is incomplete");
      return false;
    }
    rails.reserve(anchors_.size());
    for (auto& endpoint : anchors_) rails.push_back(endpoint.get());
  }

  std::vector<rdma::RcEndpoint::PoolMrRegistration> registrations(rails.size());
  const bool registered = rdma::RunPoolMrTransaction(
      rails.size(),
      [&](size_t rail) {
        return rails[rail]->StagePoolMr(base, size, /*remote_write=*/true,
                                        &registrations[rail]);
      },
      [&](size_t rail) { rails[rail]->CommitPoolMr(&registrations[rail]); },
      [&](size_t rail) { rails[rail]->RollbackPoolMr(&registrations[rail]); });
  if (!registered) {
    mr_registration_rejections_.fetch_add(1, std::memory_order_relaxed);
    DFKV_LOG_ERROR(
        "rdma: explicit memory registration rejected for base=" +
        std::to_string(address) + " size=" + std::to_string(size));
    return false;
  }

  if (!new_anchors.empty()) anchors_ = std::move(new_anchors);
  // Idle pooled endpoints have no in-flight WRs, so they can safely adopt the
  // new generation now and release their superseded same-base cache reference.
  // An endpoint that cannot resolve the already-anchored declaration is
  // discarded; a future Acquire recreates it from the published generation.
  const auto refresh_idle = [&](auto& idle) {
    for (auto& [node, connections] : idle) {
      (void)node;
      for (auto it = connections.begin(); it != connections.end();) {
        Conn* connection = *it;
        if (connection->ep.EnsurePoolMrs(candidate, true)) {
          ++it;
          continue;
        }
        it = connections.erase(it);
        Destroy(connection);
      }
    }
  };
  refresh_idle(pool_);
  refresh_idle(control_pool_);
  pools_ = std::move(candidate);
  uint64_t registered_bytes = 0;
  for (const auto& pool : pools_) {
    if (pool.second >
        std::numeric_limits<uint64_t>::max() - registered_bytes) {
      registered_bytes = std::numeric_limits<uint64_t>::max();
      break;
    }
    registered_bytes += static_cast<uint64_t>(pool.second);
  }
  mr_regions_.store(pools_.size(), std::memory_order_relaxed);
  mr_registered_bytes_.store(registered_bytes, std::memory_order_relaxed);
  if (grew) {
    DFKV_LOG_INFO(
        "rdma: explicit memory registration grew for base=" +
        std::to_string(address) + " size=" + std::to_string(size));
  }
  return true;
}

std::string RdmaTransport::MetricsText() const {
  std::string s;
  s += "# HELP dfkv_rdma_client_conns_opened_total RDMA client connections opened\n";
  s += "# TYPE dfkv_rdma_client_conns_opened_total counter\n";
  s += "dfkv_rdma_client_conns_opened_total " +
       std::to_string(conns_opened_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_v2_put_writes_total PUT requests sent with RDMA WRITE_WITH_IMM\n";
  s += "# TYPE dfkv_rdma_client_v2_put_writes_total counter\n";
  s += "dfkv_rdma_client_v2_put_writes_total " +
       std::to_string(v2_put_writes_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_v2_get_writes_total GET requests posted with RDMA WRITE destinations\n";
  s += "# TYPE dfkv_rdma_client_v2_get_writes_total counter\n";
  s += "dfkv_rdma_client_v2_get_writes_total " +
       std::to_string(v2_get_writes_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_mr_regions Successfully published host MR regions\n";
  s += "# TYPE dfkv_rdma_client_mr_regions gauge\n";
  s += "dfkv_rdma_client_mr_regions " +
       std::to_string(mr_regions_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_mr_registered_bytes Bytes in successfully published host MR declarations\n";
  s += "# TYPE dfkv_rdma_client_mr_registered_bytes gauge\n";
  s += "dfkv_rdma_client_mr_registered_bytes " +
       std::to_string(
           mr_registered_bytes_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_mr_registration_rejections_total Pool MR declarations rejected before publication\n";
  s += "# TYPE dfkv_rdma_client_mr_registration_rejections_total counter\n";
  s += "dfkv_rdma_client_mr_registration_rejections_total " +
       std::to_string(
           mr_registration_rejections_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_rail_conns_total Connections opened per rail (device)\n";
  s += "# TYPE dfkv_rdma_client_rail_conns_total counter\n";
  for (size_t i = 0; i < devs_.size(); ++i) {
    const std::string& d = devs_[i].empty() ? std::string("default") : devs_[i];
    s += "dfkv_rdma_client_rail_conns_total{dev=\"" + d + "\"} " +
         std::to_string(rail_conns_[i].load(std::memory_order_relaxed)) + "\n";
  }
  const uint64_t now = rdma::RailPolicy::NowMicros();
  const auto rail_stats = rail_policy_->Snapshot(now);
  s += "# HELP dfkv_rdma_client_rail_selections_total Operations selected per RDMA rail\n";
  s += "# TYPE dfkv_rdma_client_rail_selections_total counter\n";
  s += "# HELP dfkv_rdma_client_rail_inflight Reserved outstanding request credits per RDMA rail\n";
  s += "# TYPE dfkv_rdma_client_rail_inflight gauge\n";
  s += "# HELP dfkv_rdma_client_rail_credits_available Available outstanding request credits per RDMA rail\n";
  s += "# TYPE dfkv_rdma_client_rail_credits_available gauge\n";
  s += "# HELP dfkv_rdma_client_rail_credits_exhausted_total Selection candidates skipped because request credits were exhausted\n";
  s += "# TYPE dfkv_rdma_client_rail_credits_exhausted_total counter\n";
  s += "# HELP dfkv_rdma_client_rail_errors_total Local verbs, post, CQ, or device failures attributed to an RDMA rail\n";
  s += "# TYPE dfkv_rdma_client_rail_errors_total counter\n";
  s += "# HELP dfkv_rdma_client_endpoint_errors_total Remote endpoint, bootstrap, or protocol failures released without penalizing the rail\n";
  s += "# TYPE dfkv_rdma_client_endpoint_errors_total counter\n";
  s += "# HELP dfkv_rdma_client_rail_consecutive_errors Consecutive local rail failures used by quarantine policy\n";
  s += "# TYPE dfkv_rdma_client_rail_consecutive_errors gauge\n";
  s += "# HELP dfkv_rdma_client_rail_quarantines_total Rail quarantine transitions after consecutive failures\n";
  s += "# TYPE dfkv_rdma_client_rail_quarantines_total counter\n";
  s += "# HELP dfkv_rdma_client_rail_recoveries_total Successful rail recovery probes after cooldown\n";
  s += "# TYPE dfkv_rdma_client_rail_recoveries_total counter\n";
  s += "# HELP dfkv_rdma_client_rail_quarantined Whether a rail is currently quarantined\n";
  s += "# TYPE dfkv_rdma_client_rail_quarantined gauge\n";
  s += "# HELP dfkv_rdma_client_rail_recovery_probe Whether the quarantined rail currently has its single recovery admission in flight\n";
  s += "# TYPE dfkv_rdma_client_rail_recovery_probe gauge\n";
  s += "# HELP dfkv_rdma_client_rail_latency_ewma_us Rail completion latency EWMA used by selection policy\n";
  s += "# TYPE dfkv_rdma_client_rail_latency_ewma_us gauge\n";
  for (size_t i = 0; i < rail_stats.size(); ++i) {
    const std::string& d =
        devs_[i].empty() ? std::string("default") : devs_[i];
    const std::string labels = "{dev=\"" + d + "\"} ";
    const auto& stats = rail_stats[i];
    s += "dfkv_rdma_client_rail_selections_total" + labels +
         std::to_string(stats.selections) + "\n";
    s += "dfkv_rdma_client_rail_inflight" + labels +
         std::to_string(stats.inflight) + "\n";
    s += "dfkv_rdma_client_rail_credits_available" + labels +
         std::to_string(stats.credits - stats.inflight) + "\n";
    s += "dfkv_rdma_client_rail_credits_exhausted_total" + labels +
         std::to_string(stats.credits_exhausted) + "\n";
    s += "dfkv_rdma_client_rail_errors_total" + labels +
         std::to_string(stats.errors) + "\n";
    s += "dfkv_rdma_client_endpoint_errors_total" + labels +
         std::to_string(stats.endpoint_errors) + "\n";
    s += "dfkv_rdma_client_rail_consecutive_errors" + labels +
         std::to_string(stats.consecutive_errors) + "\n";
    s += "dfkv_rdma_client_rail_quarantines_total" + labels +
         std::to_string(stats.quarantines) + "\n";
    s += "dfkv_rdma_client_rail_recoveries_total" + labels +
         std::to_string(stats.recoveries) + "\n";
    s += "dfkv_rdma_client_rail_quarantined" + labels +
         std::to_string(stats.quarantined ? 1 : 0) + "\n";
    s += "dfkv_rdma_client_rail_recovery_probe" + labels +
         std::to_string(stats.recovery_probe ? 1 : 0) + "\n";
    s += "dfkv_rdma_client_rail_latency_ewma_us" + labels +
         std::to_string(stats.latency_ewma_us) + "\n";
  }
  s += "# HELP dfkv_rdma_client_numa_fallbacks_total NUMA-aware admissions that fell back to all enabled rails\n";
  s += "# TYPE dfkv_rdma_client_numa_fallbacks_total counter\n";
  s += "dfkv_rdma_client_numa_fallbacks_total{reason=\"caller_unknown\"} " +
       std::to_string(
           numa_caller_unknown_fallbacks_.load(std::memory_order_relaxed)) +
       "\n";
  s += "dfkv_rdma_client_numa_fallbacks_total{reason=\"no_local_rail\"} " +
       std::to_string(
           numa_no_local_fallbacks_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_cq_completions_total Work completions reaped from RDMA completion queues\n";
  s += "# TYPE dfkv_rdma_cq_completions_total counter\n";
  s += "dfkv_rdma_cq_completions_total " +
       std::to_string(rdma::RcEndpoint::CqCompletions()) + "\n";
  s += "# HELP dfkv_rdma_cq_errors_total Completion queue polling and failed work completion errors\n";
  s += "# TYPE dfkv_rdma_cq_errors_total counter\n";
  s += "dfkv_rdma_cq_errors_total " +
       std::to_string(rdma::RcEndpoint::CqErrors()) + "\n";
  // Effective per-connection posting depth. Aggregate concurrency is bounded
  // independently by the per-rail credit policy above.
  s += "# HELP dfkv_rdma_client_pipeline_depth Effective RDMA pipeline depth (env DFKV_RDMA_DEPTH)\n";
  s += "# TYPE dfkv_rdma_client_pipeline_depth gauge\n";
  s += "dfkv_rdma_client_pipeline_depth " + std::to_string(depth_) + "\n";
  s += "# HELP dfkv_rdma_client_adhoc_user_mr_total One-shot user MRs registered outside explicit pool regions\n";
  s += "# TYPE dfkv_rdma_client_adhoc_user_mr_total counter\n";
  s += "dfkv_rdma_client_adhoc_user_mr_total " +
       std::to_string(rdma::RcEndpoint::AdhocUserMrTotal()) + "\n";
  s += "# HELP dfkv_rdma_client_transient_user_mr_active Live one-shot out-of-pool user MRs\n";
  s += "# TYPE dfkv_rdma_client_transient_user_mr_active gauge\n";
  s += "dfkv_rdma_client_transient_user_mr_active " +
       std::to_string(rdma::RcEndpoint::TransientUserMrActive()) + "\n";
  s += "# HELP dfkv_rdma_client_pool_mr_registrations_total Actual pool-region ibv_reg_mr calls\n";
  s += "# TYPE dfkv_rdma_client_pool_mr_registrations_total counter\n";
  s += "dfkv_rdma_client_pool_mr_registrations_total " +
       std::to_string(rdma::RcEndpoint::PoolMrRegistrations()) + "\n";
  s += "# HELP dfkv_rdma_client_pool_mr_registration_failures_total Failed explicit pool-region registrations\n";
  s += "# TYPE dfkv_rdma_client_pool_mr_registration_failures_total counter\n";
  s += "dfkv_rdma_client_pool_mr_registration_failures_total " +
       std::to_string(rdma::RcEndpoint::PoolMrRegistrationFailures()) + "\n";
  s += "# HELP dfkv_rdma_client_pool_mr_active_registrations Live shared-PD pool MR generations\n";
  s += "# TYPE dfkv_rdma_client_pool_mr_active_registrations gauge\n";
  s += "dfkv_rdma_client_pool_mr_active_registrations " +
       std::to_string(rdma::RcEndpoint::PoolMrActiveRegistrations()) + "\n";
  s += "# HELP dfkv_rdma_client_max_block_seen_bytes Largest requested RDMA block or destination capacity\n";
  s += "# TYPE dfkv_rdma_client_max_block_seen_bytes gauge\n";
  s += "dfkv_rdma_client_max_block_seen_bytes " +
       std::to_string(max_block_seen_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_declared_max_block_bytes Negotiated maximum block declaration\n";
  s += "# TYPE dfkv_rdma_client_declared_max_block_bytes gauge\n";
  s += "dfkv_rdma_client_declared_max_block_bytes " +
       std::to_string(declared_) + "\n";
  s += "# HELP dfkv_rdma_client_oversize_rejects_total Operations rejected above the declared maximum block\n";
  s += "# TYPE dfkv_rdma_client_oversize_rejects_total counter\n";
  s += "dfkv_rdma_client_oversize_rejects_total " +
       std::to_string(oversize_rejects_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_v2_probe_attempts_total Required-protocol bootstrap probes\n";
  s += "# TYPE dfkv_rdma_client_v2_probe_attempts_total counter\n";
  s += "dfkv_rdma_client_v2_probe_attempts_total " +
       std::to_string(v2_probe_attempts_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_v2_probe_failures_total Failed required-protocol bootstrap probes\n";
  s += "# TYPE dfkv_rdma_client_v2_probe_failures_total counter\n";
  s += "dfkv_rdma_client_v2_probe_failures_total " +
       std::to_string(v2_probe_failures_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_stale_pool_retries_total Fresh-connection retries after a pooled QP failure\n";
  s += "# TYPE dfkv_rdma_client_stale_pool_retries_total counter\n";
  s += "dfkv_rdma_client_stale_pool_retries_total " +
       std::to_string(stale_pool_retries_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_completion_timeouts_total Completion windows exhausting their absolute deadline\n";
  s += "# TYPE dfkv_rdma_client_completion_timeouts_total counter\n";
  s += "dfkv_rdma_client_completion_timeouts_total " +
       std::to_string(completion_timeouts_.load(std::memory_order_relaxed)) +
       "\n";
  return s;
}

void RdmaTransport::Release(const std::string& node, Lane lane, Conn* c) {
  if (c->credit_held) {
    const uint64_t now = rdma::RailPolicy::NowMicros();
    rail_policy_->Complete(c->lease, now - c->lease_started_us,
                           rdma::RailCompletion::kSuccess, now);
    c->credit_held = false;
  }
  bool reusable = false;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto& idle = lane == Lane::kData ? pool_ : control_pool_;
    auto& v = idle[node];
    // A connection that was active during successful growth still owns the old
    // generation. Refresh it only after its WRs complete, before it becomes
    // idle; this is the endpoint lease that makes old-generation retirement
    // safe without interrupting the operation that used it.
    if (v.size() < pool_max_ && c->ep.EnsurePoolMrs(pools_, true)) {
      v.push_back(c);
      reusable = true;
    }
  }
  if (!reusable) Destroy(c);
}

Status RdmaTransport::RoundTrip(const std::string& node, WireOp op,
                                const BlockKey& key, uint64_t offset,
                                uint64_t length, const void* payload,
                                uint64_t payload_len, std::string* out,
                                uint64_t* value_len) {
  const uint64_t value_bound = static_cast<uint64_t>(OpBound());
  if ((op == WireOp::kCache &&
       (payload_len == 0 || payload_len > value_bound ||
        payload_len > std::numeric_limits<size_t>::max() ||
        !ValidBuffer(payload, static_cast<size_t>(payload_len)))) ||
      (op == WireOp::kRange && (length > value_bound || out == nullptr)) ||
      (op == WireOp::kMembers && out == nullptr)) {
    return Status::kInvalid;
  }

  const Lane lane =
      op == WireOp::kCache || op == WireOp::kRange ? Lane::kData
                                                   : Lane::kControl;
  for (int attempt = 0; attempt < 2; ++attempt) {
    if (attempt != 0)
      stale_pool_retries_.fetch_add(1, std::memory_order_relaxed);
    if (out) out->clear();
    bool from_pool = false;
    Conn* conn = Acquire(node, lane, &from_pool, attempt > 0);
    if (!conn) return Status::kIOError;
    rdma::RcEndpoint& ep = conn->ep;

    bool ok = false;
    ibv_mr* transient_output_mr = nullptr;
    if (op == WireOp::kCache) {
      if (payload_len > conn->data_capacity()) {
        Release(node, lane, conn);
        return Status::kInvalid;
      }
      ibv_mr* payload_mr =
          payload_len
              ? ep.RegisterTransient(const_cast<void*>(payload),
                                     static_cast<size_t>(payload_len),
                                     /*remote_write=*/false)
              : nullptr;
      transient_output_mr = payload_mr;
      if (payload_len != 0 && !payload_mr) {
        Release(node, lane, conn);
        return Status::kIOError;
      }
      conn->Encode(ep.sbuf(0), op, key, offset, length, payload_len);
      ok = ep.PostRecv(0) &&
           ep.PostWriteImmScatter(
               0, kReqPrefix, payload, static_cast<size_t>(payload_len),
               payload_mr, conn->put_addr(0), conn->recv_segment.rkey, 0);
      if (ok)
        v2_put_writes_.fetch_add(1, std::memory_order_relaxed);
    } else if (op == WireOp::kRange) {
      if (!out || length > conn->data_capacity() ||
          length > std::numeric_limits<uint32_t>::max()) {
        Release(node, lane, conn);
        return Status::kInvalid;
      }
      out->resize(static_cast<size_t>(length));
      std::vector<RdmaWriteTarget> targets;
      if (length != 0) {
        transient_output_mr = ep.RegisterTransient(out->data(), out->size());
        if (!transient_output_mr) {
          Release(node, lane, conn);
          return Status::kIOError;
        }
        targets.push_back(
            {reinterpret_cast<uint64_t>(out->data()),
             transient_output_mr->rkey, static_cast<uint32_t>(length)});
      }
      size_t frame_len = 0;
      ok = EncodeRdmaGetReq(ep.sbuf(0), ep.cap(), key, offset, length,
                            targets, &frame_len) &&
           ep.PostRecv(0) && ep.PostSend(0, frame_len);
      if (ok)
        v2_get_writes_.fetch_add(1, std::memory_order_relaxed);
    } else {
      conn->Encode(ep.sbuf(0), op, key, offset, length, payload_len);
      ok = ep.PostRecv(0) &&
           ep.PostSend(0, kReqPrefix + static_cast<size_t>(payload_len));
    }

    std::vector<uint32_t> reply_bytes;
    bool timed_out = false;
    // Post/encode failures above carry no WC evidence and stay a local rail
    // failure; a failed reap window is classified from its own evidence so a
    // dead peer cannot poison a healthy local rail.
    rdma::RailCompletion completion = rdma::RailCompletion::kRailFailure;
    if (ok) {
      ibv_wc_status wc_status = IBV_WC_SUCCESS;
      bool had_wcs = false;
      ok = ReapWindow(ep, 1, &reply_bytes, op_timeout_ms_, &timed_out,
                      &wc_status, &had_wcs);
      if (!ok) completion = rdma::ClassifyCompletion(wc_status, had_wcs);
    }
    if (timed_out)
      completion_timeouts_.fetch_add(1, std::memory_order_relaxed);
    const uint32_t recv_bytes =
        ok && !reply_bytes.empty() ? reply_bytes[0] : 0;
    if (ok && transient_output_mr) {
      ep.ReleaseTransient(transient_output_mr);
      transient_output_mr = nullptr;
    }

    if (!ok) {
      Destroy(conn, completion);
      if (!from_pool) return Status::kIOError;
      continue;
    }
    if (recv_bytes < kRespPrefix) {
      Destroy(conn, rdma::RailCompletion::kEndpointFailure);
      return Status::kIOError;
    }
    Status status;
    uint64_t data_len = 0;
    const uint64_t response_bound =
        op == WireOp::kMembers ? rdma::kV2ControlResponseMax
                               : (op == WireOp::kRange ? length : 0);
    if (!conn->Decode(ep.rbuf(0), &status, &data_len, response_bound,
                      value_len)) {
      Destroy(conn, rdma::RailCompletion::kEndpointFailure);
      return Status::kIOError;
    }
    if (out) {
      if (op == WireOp::kRange) {
        if (status == Status::kOk && data_len <= out->size()) {
          out->resize(static_cast<size_t>(data_len));
        } else if (status != Status::kOk) {
          out->clear();
        } else {
          Destroy(conn, rdma::RailCompletion::kEndpointFailure);
          return Status::kIOError;
        }
      } else {
        if (data_len > rdma::kV2ControlResponseMax ||
            data_len > recv_bytes - kRespPrefix) {
          Destroy(conn, rdma::RailCompletion::kEndpointFailure);
          return Status::kIOError;
        }
        out->assign(ep.rbuf(0) + kRespPrefix,
                    static_cast<size_t>(data_len));
      }
    }
    Release(node, lane, conn);
    return status;
  }
  return Status::kIOError;
}

Status RdmaTransport::Cache(const std::string& node, const BlockKey& key,
                            const void* data, size_t len) {
  if (NoteBlock(len)) return Status::kInvalid;
  return RoundTrip(node, WireOp::kCache, key, 0, 0, data, len, nullptr);
}
Status RdmaTransport::Range(const std::string& node, const BlockKey& key,
                            uint64_t offset, uint64_t length,
                            std::string* out, uint64_t* value_len) {
  if (length > std::numeric_limits<size_t>::max() ||
      NoteBlock(static_cast<size_t>(length)))
    return Status::kInvalid;
  return RoundTrip(node, WireOp::kRange, key, offset, length, nullptr, 0, out,
                   value_len);
}

Status RdmaTransport::Lookup(const std::string& node, const BlockKey& key,
                             uint64_t* value_len) {
  if (value_len == nullptr) return Status::kInvalid;
  *value_len = 0;
  return RoundTrip(node, WireOp::kLookup, key, 0, 0, nullptr, 0, nullptr,
                   value_len);
}
Status RdmaTransport::Exist(const std::string& node, const BlockKey& key,
                            bool* exist) {
  if (exist == nullptr) return Status::kInvalid;
  Status st =
      RoundTrip(node, WireOp::kExist, key, 0, 0, nullptr, 0, nullptr);
  if (st == Status::kOk) {
    *exist = true;
    return Status::kOk;
  }
  if (st == Status::kNotFound) {
    *exist = false;
    return Status::kOk;
  }
  return st;
}

Status RdmaTransport::Remove(const std::string& node, const BlockKey& key) {
  return RoundTrip(node, WireOp::kRemove, key, 0, 0, nullptr, 0, nullptr);
}

std::vector<Status> RdmaTransport::CacheMany(
    const std::string& node, const std::vector<CacheItem>& items) {
  const size_t count = items.size();
  std::vector<Status> result(count, Status::kIOError);
  if (count == 0) return result;
  std::vector<char> bad(count, 0);
  size_t valid_count = 0;
  for (size_t i = 0; i < count; ++i) {
    if (items[i].len == 0 ||
        !ValidBuffer(items[i].data, items[i].len) ||
        NoteBlock(items[i].len)) {
      bad[i] = 1;
      result[i] = Status::kInvalid;
    } else {
      ++valid_count;
    }
  }
  if (valid_count == 0) return result;

  for (int attempt = 0; attempt < 2; ++attempt) {
    if (attempt != 0)
      stale_pool_retries_.fetch_add(1, std::memory_order_relaxed);
    std::fill(result.begin(), result.end(), Status::kIOError);
    for (size_t i = 0; i < count; ++i)
      if (bad[i]) result[i] = Status::kInvalid;
    bool from_pool = false;
    Conn* conn = Acquire(node, Lane::kData, &from_pool, attempt > 0,
                         std::min(valid_count, depth_));
    if (!conn) return result;
    rdma::RcEndpoint& ep = conn->ep;
    const size_t window = std::min(
        ep.window(), static_cast<size_t>(conn->lease.credits));
    bool conn_ok = true;
    // Failure attribution for Destroy: post/encode/register failures keep the
    // local-rail default; a failed reap window is classified from its WC
    // evidence; wire decode failures blame the peer.
    rdma::RailCompletion completion = rdma::RailCompletion::kRailFailure;
    for (size_t base = 0; base < count && conn_ok; base += window) {
      const size_t width = std::min(window, count - base);
      std::vector<ibv_mr*> payload_mrs(width, nullptr);
      size_t posted = 0;
      for (size_t slot = 0; slot < width; ++slot) {
        const size_t item_index = base + slot;
        if (bad[item_index]) continue;
        const CacheItem& item = items[item_index];
        if (item.len > conn->data_capacity()) {
          bad[item_index] = 1;
          result[item_index] = Status::kInvalid;
          continue;
        }
        ibv_mr* mr =
            item.len
                ? ep.RegisterTransient(const_cast<void*>(item.data), item.len,
                                       /*remote_write=*/false)
                : nullptr;
        payload_mrs[slot] = mr;
        conn->Encode(ep.sbuf(slot), WireOp::kCache, item.key, 0, 0,
                     item.len);
        if ((item.len != 0 && !mr) || !ep.PostRecv(slot) ||
            !ep.PostWriteImmScatter(
                slot, kReqPrefix, item.data, item.len, mr,
                conn->put_addr(slot), conn->recv_segment.rkey,
                static_cast<uint32_t>(slot))) {
          conn_ok = false;
          break;
        }
        ++posted;
        v2_put_writes_.fetch_add(1, std::memory_order_relaxed);
      }
      std::vector<uint32_t> reply_bytes;
      bool timed_out = false;
      ibv_wc_status wc_status = IBV_WC_SUCCESS;
      bool had_wcs = false;
      if (conn_ok &&
          !ReapPosted(ep, posted, width, &reply_bytes, BatchTimeout(),
                      &timed_out, &wc_status, &had_wcs)) {
        conn_ok = false;
        completion = rdma::ClassifyCompletion(wc_status, had_wcs);
      }
      if (timed_out)
        completion_timeouts_.fetch_add(1, std::memory_order_relaxed);
      if (!conn_ok) break;
      for (ibv_mr* mr : payload_mrs) ep.ReleaseTransient(mr);
      for (size_t slot = 0; slot < width; ++slot) {
        const size_t item_index = base + slot;
        if (bad[item_index]) continue;
        Status status;
        uint64_t data_len = 0;
        if (reply_bytes[slot] < kRespPrefix ||
            !conn->Decode(ep.rbuf(slot), &status, &data_len, 0)) {
          conn_ok = false;
          completion = rdma::RailCompletion::kEndpointFailure;
          break;
        }
        result[item_index] = status;
      }
    }
    if (conn_ok) {
      Release(node, Lane::kData, conn);
      return result;
    }
    Destroy(conn, completion);
    if (from_pool) continue;
    return result;
  }
  return result;
}

Status RdmaTransport::Members(const std::string& node, std::string* out) {
  return RoundTrip(node, WireOp::kMembers, BlockKey{}, 0, 0, nullptr, 0, out);
}


std::vector<Status> RdmaTransport::RangeMany(
    const std::string& node, const std::vector<BlockKey>& keys,
    uint64_t offset, uint64_t length, std::vector<std::string>* outputs,
    std::vector<uint64_t>* value_lens) {
  const size_t count = keys.size();
  if (outputs == nullptr) {
    if (value_lens) value_lens->clear();
    return InvalidStatuses(count);
  }
  std::vector<Status> result(count, Status::kIOError);
  if (count == 0) {
    outputs->clear();
    if (value_lens) value_lens->clear();
    return result;
  }
  if (length > std::numeric_limits<uint32_t>::max() ||
      length > std::numeric_limits<size_t>::max() ||
      NoteBlock(static_cast<size_t>(length))) {
    outputs->clear();
    if (value_lens) value_lens->clear();
    return InvalidStatuses(count);
  }
  outputs->assign(count, std::string());
  if (value_lens) value_lens->assign(count, 0);

  for (int attempt = 0; attempt < 2; ++attempt) {
    if (attempt != 0)
      stale_pool_retries_.fetch_add(1, std::memory_order_relaxed);
    std::fill(result.begin(), result.end(), Status::kIOError);
    outputs->assign(count, std::string());
    if (value_lens) value_lens->assign(count, 0);
    bool from_pool = false;
    Conn* conn = Acquire(node, Lane::kData, &from_pool, attempt > 0,
                         std::min(count, depth_));
    if (!conn) return result;
    rdma::RcEndpoint& ep = conn->ep;
    const size_t window = std::min(
        ep.window(), static_cast<size_t>(conn->lease.credits));
    bool conn_ok = true;
    // Failure attribution for Destroy: post/encode/register failures keep the
    // local-rail default; a failed reap window is classified from its WC
    // evidence; wire decode failures blame the peer.
    rdma::RailCompletion completion = rdma::RailCompletion::kRailFailure;
    for (size_t base = 0; base < count && conn_ok; base += window) {
      const size_t width = std::min(window, count - base);
      std::vector<ibv_mr*> output_mrs(width, nullptr);
      for (size_t slot = 0; slot < width; ++slot) {
        std::string& output = (*outputs)[base + slot];
        output.resize(static_cast<size_t>(length));
        std::vector<RdmaWriteTarget> targets;
        if (length != 0) {
          output_mrs[slot] =
              ep.RegisterTransient(output.data(), output.size());
          if (!output_mrs[slot]) {
            conn_ok = false;
            break;
          }
          targets.push_back(
              {reinterpret_cast<uint64_t>(output.data()),
               output_mrs[slot]->rkey, static_cast<uint32_t>(length)});
        }
        size_t frame_len = 0;
        if (!EncodeRdmaGetReq(ep.sbuf(slot), ep.cap(), keys[base + slot],
                              offset, length, targets, &frame_len) ||
            !ep.PostRecv(slot) || !ep.PostSend(slot, frame_len)) {
          conn_ok = false;
          break;
        }
        v2_get_writes_.fetch_add(1, std::memory_order_relaxed);
      }
      std::vector<uint32_t> reply_bytes;
      bool timed_out = false;
      ibv_wc_status wc_status = IBV_WC_SUCCESS;
      bool had_wcs = false;
      if (conn_ok &&
          !ReapWindow(ep, width, &reply_bytes, BatchTimeout(), &timed_out,
                      &wc_status, &had_wcs)) {
        conn_ok = false;
        completion = rdma::ClassifyCompletion(wc_status, had_wcs);
      }
      if (timed_out)
        completion_timeouts_.fetch_add(1, std::memory_order_relaxed);
      if (!conn_ok) break;
      for (ibv_mr* mr : output_mrs) ep.ReleaseTransient(mr);
      for (size_t slot = 0; slot < width; ++slot) {
        if (reply_bytes[slot] < kRespPrefix) {
          conn_ok = false;
          completion = rdma::RailCompletion::kEndpointFailure;
          break;
        }
        Status status;
        uint64_t data_len = 0;
        uint64_t value_len = 0;
        if (!conn->Decode(ep.rbuf(slot), &status, &data_len, length,
                          &value_len)) {
          conn_ok = false;
          completion = rdma::RailCompletion::kEndpointFailure;
          break;
        }
        result[base + slot] = status;
        if (value_lens) (*value_lens)[base + slot] = value_len;
        std::string& output = (*outputs)[base + slot];
        if (status != Status::kOk) {
          output.clear();
        } else if (data_len <= output.size()) {
          output.resize(static_cast<size_t>(data_len));
        } else {
          conn_ok = false;
          completion = rdma::RailCompletion::kEndpointFailure;
          break;
        }
      }
    }
    if (conn_ok) {
      Release(node, Lane::kData, conn);
      return result;
    }
    Destroy(conn, completion);
    if (from_pool) continue;
    return result;
  }
  return result;
}

std::vector<Status> RdmaTransport::ExistMany(
    const std::string& node, const std::vector<BlockKey>& keys,
    std::vector<char>* exists) {
  const size_t count = keys.size();
  if (exists == nullptr) return InvalidStatuses(count);
  exists->assign(count, 0);
  std::vector<Status> result(count, Status::kIOError);
  if (count == 0) return result;

  for (int attempt = 0; attempt < 2; ++attempt) {
    if (attempt != 0)
      stale_pool_retries_.fetch_add(1, std::memory_order_relaxed);
    std::fill(result.begin(), result.end(), Status::kIOError);
    std::fill(exists->begin(), exists->end(), 0);
    bool from_pool = false;
    Conn* conn = Acquire(node, Lane::kControl, &from_pool, attempt > 0,
                         std::min(count, depth_));
    if (!conn) return result;
    rdma::RcEndpoint& ep = conn->ep;
    const size_t window = std::min(
        ep.window(), static_cast<size_t>(conn->lease.credits));
    bool conn_ok = true;
    // Failure attribution for Destroy: post/encode/register failures keep the
    // local-rail default; a failed reap window is classified from its WC
    // evidence; wire decode failures blame the peer.
    rdma::RailCompletion completion = rdma::RailCompletion::kRailFailure;
    for (size_t base = 0; base < count && conn_ok; base += window) {
      const size_t width = std::min(window, count - base);
      std::vector<size_t> send_lengths(width);
      for (size_t slot = 0; slot < width; ++slot) {
        conn->Encode(ep.sbuf(slot), WireOp::kExist, keys[base + slot],
                     0, 0, 0);
        send_lengths[slot] = kReqPrefix;
      }
      std::vector<uint32_t> reply_bytes;
      bool timed_out = false;
      ibv_wc_status wc_status = IBV_WC_SUCCESS;
      bool had_wcs = false;
      if (!RunWindow(ep, send_lengths, &reply_bytes, BatchTimeout(),
                     &timed_out, &wc_status, &had_wcs)) {
        completion = rdma::ClassifyCompletion(wc_status, had_wcs);
        if (timed_out)
          completion_timeouts_.fetch_add(1, std::memory_order_relaxed);
        conn_ok = false;
        break;
      }
      for (size_t slot = 0; slot < width; ++slot) {
        Status status;
        uint64_t data_len = 0;
        if (reply_bytes[slot] < kRespPrefix ||
            !conn->Decode(ep.rbuf(slot), &status, &data_len, 0)) {
          conn_ok = false;
          completion = rdma::RailCompletion::kEndpointFailure;
          break;
        }
        result[base + slot] = status;
        (*exists)[base + slot] = status == Status::kOk ? 1 : 0;
      }
    }
    if (conn_ok) {
      Release(node, Lane::kControl, conn);
      return result;
    }
    Destroy(conn, completion);
    if (from_pool) continue;
    return result;
  }
  return result;
}

std::vector<Status> RdmaTransport::RangeInto(
    const std::string& node, const std::vector<BlockKey>& keys,
    const std::vector<RangeDst>& destinations,
    std::vector<uint64_t>* value_lens) {
  const size_t count = keys.size();
  if (value_lens) value_lens->assign(count, 0);
  if (destinations.size() != count) return InvalidStatuses(count);
  std::vector<Status> result(count, Status::kIOError);
  if (count == 0) return result;
  std::vector<char> bad(count, 0);
  size_t valid_count = 0;
  for (size_t i = 0; i < count; ++i) {
    if (!ValidBuffer(destinations[i].payload, destinations[i].n) ||
        destinations[i].n > std::numeric_limits<uint32_t>::max() ||
        NoteBlock(destinations[i].n)) {
      bad[i] = 1;
      result[i] = Status::kInvalid;
    } else {
      ++valid_count;
    }
  }
  if (valid_count == 0) return result;

  for (int attempt = 0; attempt < 2; ++attempt) {
    if (attempt != 0)
      stale_pool_retries_.fetch_add(1, std::memory_order_relaxed);
    std::fill(result.begin(), result.end(), Status::kIOError);
    for (size_t i = 0; i < count; ++i)
      if (bad[i]) result[i] = Status::kInvalid;
    if (value_lens) value_lens->assign(count, 0);
    bool from_pool = false;
    Conn* conn = Acquire(node, Lane::kData, &from_pool, attempt > 0,
                         std::min(valid_count, depth_));
    if (!conn) return result;
    rdma::RcEndpoint& ep = conn->ep;
    const size_t window = std::min(
        ep.window(), static_cast<size_t>(conn->lease.credits));
    bool conn_ok = true;
    // Failure attribution for Destroy: post/encode/register failures keep the
    // local-rail default; a failed reap window is classified from its WC
    // evidence; wire decode failures blame the peer.
    rdma::RailCompletion completion = rdma::RailCompletion::kRailFailure;
    for (size_t base = 0; base < count && conn_ok; base += window) {
      const size_t width = std::min(window, count - base);
      std::vector<ibv_mr*> output_mrs(width, nullptr);
      size_t posted = 0;
      for (size_t slot = 0; slot < width; ++slot) {
        const size_t item_index = base + slot;
        if (bad[item_index]) continue;
        const RangeDst& destination = destinations[item_index];
        if (destination.n != 0) {
          output_mrs[slot] =
              ep.RegisterTransient(destination.payload, destination.n);
          if (!output_mrs[slot]) {
            conn_ok = false;
            break;
          }
        }
        std::vector<RdmaWriteTarget> targets;
        if (destination.n != 0) {
          targets.push_back(
              {reinterpret_cast<uint64_t>(destination.payload),
               output_mrs[slot]->rkey,
               static_cast<uint32_t>(destination.n)});
        }
        size_t frame_len = 0;
        if (!EncodeRdmaGetReq(ep.sbuf(slot), ep.cap(), keys[item_index],
                              0, destination.n, targets, &frame_len) ||
            !ep.PostRecv(slot) || !ep.PostSend(slot, frame_len)) {
          conn_ok = false;
          break;
        }
        ++posted;
        v2_get_writes_.fetch_add(1, std::memory_order_relaxed);
      }
      std::vector<uint32_t> reply_bytes;
      bool timed_out = false;
      ibv_wc_status wc_status = IBV_WC_SUCCESS;
      bool had_wcs = false;
      if (conn_ok &&
          !ReapPosted(ep, posted, width, &reply_bytes, BatchTimeout(),
                      &timed_out, &wc_status, &had_wcs)) {
        conn_ok = false;
        completion = rdma::ClassifyCompletion(wc_status, had_wcs);
      }
      if (timed_out)
        completion_timeouts_.fetch_add(1, std::memory_order_relaxed);
      if (!conn_ok) break;
      for (ibv_mr* mr : output_mrs) ep.ReleaseTransient(mr);
      for (size_t slot = 0; slot < width; ++slot) {
        const size_t item_index = base + slot;
        if (bad[item_index]) continue;
        Status status;
        uint64_t data_len = 0;
        uint64_t value_len = 0;
        if (reply_bytes[slot] < kRespPrefix ||
            !conn->Decode(ep.rbuf(slot), &status, &data_len,
                          destinations[item_index].n, &value_len)) {
          conn_ok = false;
          completion = rdma::RailCompletion::kEndpointFailure;
          break;
        }
        result[item_index] = status;
        if (value_lens) (*value_lens)[item_index] = value_len;
      }
    }
    if (conn_ok) {
      Release(node, Lane::kData, conn);
      return result;
    }
    Destroy(conn, completion);
    if (from_pool) continue;
    return result;
  }
  return result;
}

std::vector<Status> RdmaTransport::CacheFrom(
    const std::string& node, const std::vector<CacheSrc>& sources) {
  std::vector<CacheItem> items;
  items.reserve(sources.size());
  for (const auto& source : sources) {
    items.push_back(
        CacheItem{source.key, source.payload, source.payload_len});
  }
  return CacheMany(node, items);
}

std::vector<Status> RdmaTransport::CacheFromMulti(
    const std::string& node, const std::vector<CacheSrcMulti>& sources) {
  const size_t count = sources.size();
  std::vector<Status> result(count, Status::kIOError);
  if (count == 0) return result;
  std::vector<char> bad(count, 0);
  std::vector<size_t> totals(count, 0);
  size_t valid_count = 0;
  for (size_t i = 0; i < count; ++i) {
    bool invalid = sources[i].payloads.size() > sg_payload_segs_;
    size_t total = 0;
    for (const auto& payload : sources[i].payloads) {
      size_t next = 0;
      if (!ValidBuffer(payload.first, payload.second) ||
          payload.second > std::numeric_limits<uint32_t>::max() ||
          !CheckedAddSize(total, payload.second, &next)) {
        invalid = true;
        break;
      }
      total = next;
    }
    if (!invalid && (total == 0 || NoteBlock(total))) invalid = true;
    totals[i] = total;
    if (invalid) {
      bad[i] = 1;
      result[i] = Status::kInvalid;
    } else {
      ++valid_count;
    }
  }
  if (valid_count == 0) return result;

  for (int attempt = 0; attempt < 2; ++attempt) {
    if (attempt != 0)
      stale_pool_retries_.fetch_add(1, std::memory_order_relaxed);
    std::fill(result.begin(), result.end(), Status::kIOError);
    for (size_t i = 0; i < count; ++i)
      if (bad[i]) result[i] = Status::kInvalid;
    bool from_pool = false;
    Conn* conn = Acquire(node, Lane::kData, &from_pool, attempt > 0,
                         std::min(valid_count, depth_));
    if (!conn) return result;
    rdma::RcEndpoint& ep = conn->ep;
    const size_t window = std::min(
        ep.window(), static_cast<size_t>(conn->lease.credits));
    bool conn_ok = true;
    // Failure attribution for Destroy: post/encode/register failures keep the
    // local-rail default; a failed reap window is classified from its WC
    // evidence; wire decode failures blame the peer.
    rdma::RailCompletion completion = rdma::RailCompletion::kRailFailure;
    for (size_t base = 0; base < count && conn_ok; base += window) {
      const size_t width = std::min(window, count - base);
      std::vector<std::vector<ibv_mr*>> mrs(width);
      size_t posted = 0;
      for (size_t slot = 0; slot < width; ++slot) {
        const size_t item_index = base + slot;
        if (bad[item_index]) continue;
        const CacheSrcMulti& source = sources[item_index];
        mrs[slot].assign(source.payloads.size(), nullptr);
        for (size_t seg = 0; seg < source.payloads.size(); ++seg) {
          if (source.payloads[seg].second == 0) continue;
          mrs[slot][seg] = ep.RegisterTransient(
              const_cast<void*>(source.payloads[seg].first),
              source.payloads[seg].second, /*remote_write=*/false);
          if (!mrs[slot][seg]) {
            conn_ok = false;
            break;
          }
        }
        if (!conn_ok) break;
        std::vector<std::pair<const void*, uint32_t>> segments;
        segments.reserve(source.payloads.size());
        for (const auto& payload : source.payloads) {
          segments.emplace_back(payload.first,
                                static_cast<uint32_t>(payload.second));
        }
        conn->Encode(ep.sbuf(slot), WireOp::kCache, source.key, 0, 0,
                     totals[item_index]);
        if (!ep.PostRecv(slot) ||
            !ep.PostWriteImmScatterMulti(
                slot, kReqPrefix, segments, mrs[slot],
                conn->put_addr(slot), conn->recv_segment.rkey,
                static_cast<uint32_t>(slot))) {
          conn_ok = false;
          break;
        }
        ++posted;
        v2_put_writes_.fetch_add(1, std::memory_order_relaxed);
      }
      std::vector<uint32_t> reply_bytes;
      bool timed_out = false;
      ibv_wc_status wc_status = IBV_WC_SUCCESS;
      bool had_wcs = false;
      if (conn_ok &&
          !ReapPosted(ep, posted, width, &reply_bytes, BatchTimeout(),
                      &timed_out, &wc_status, &had_wcs)) {
        conn_ok = false;
        completion = rdma::ClassifyCompletion(wc_status, had_wcs);
      }
      if (timed_out)
        completion_timeouts_.fetch_add(1, std::memory_order_relaxed);
      if (!conn_ok) break;
      for (auto& slot_mrs : mrs)
        for (ibv_mr* mr : slot_mrs) ep.ReleaseTransient(mr);
      for (size_t slot = 0; slot < width; ++slot) {
        const size_t item_index = base + slot;
        if (bad[item_index]) continue;
        Status status;
        uint64_t data_len = 0;
        if (reply_bytes[slot] < kRespPrefix ||
            !conn->Decode(ep.rbuf(slot), &status, &data_len, 0)) {
          conn_ok = false;
          completion = rdma::RailCompletion::kEndpointFailure;
          break;
        }
        result[item_index] = status;
      }
    }
    if (conn_ok) {
      Release(node, Lane::kData, conn);
      return result;
    }
    Destroy(conn, completion);
    if (from_pool) continue;
    return result;
  }
  return result;
}

// Records n as a high-water candidate and reports whether it exceeds the
// declaration. Two problems this closes, both observed on a B200 node:
//
//  1. The declaration determines receive-segment capacity reserved for every
//     data connection: queue depth multiplied by the aligned slot size. An
//     over-generous value consumes scarce pinned memory across the fleet. The
//     average transfer size (437 KiB on the incident host) says nothing about
//     the peak block that must fit, so the observed high-water mark is the
//     actionable sizing signal.
//
//  2. An UNDER-sized declaration failed silently. Oversized blocks become
//     kInvalid, which the client's health accounting deliberately ignores, so
//     upstream they are indistinguishable from an ordinary cache miss: no error,
//     no log, no counter -- just a hit rate quietly capped for large pages.
//     Anyone tuning this down would have had no signal that they went too far.
bool RdmaTransport::NoteBlock(size_t n) const {
  uint64_t prev = max_block_seen_.load(std::memory_order_relaxed);
  while (n > prev &&
         !max_block_seen_.compare_exchange_weak(prev, n, std::memory_order_relaxed)) {
  }
  if (n > prev) {  // new high-water mark: the number needed to size the declaration
    DFKV_LOG_INFO("rdma: max block observed " + std::to_string(n) + "B (" +
                  std::to_string(n / 1024) + " KiB); declared bound " +
                  std::to_string(OpBound()) + "B");
  }
  const size_t bound = OpBound();
  if (n <= bound) return false;
  const uint64_t k = oversize_rejects_.fetch_add(1, std::memory_order_relaxed);
  if (k == 0 || (k & 0x3FFu) == 0)  // first, then every 1024th
    DFKV_LOG_WARN("rdma: block " + std::to_string(n) + "B exceeds the declared bound " +
                  std::to_string(bound) + "B -> kInvalid, which reads as a cache MISS "
                  "upstream (not an error). Raise DFKV_RDMA_MAX_BLOCK_BYTES. rejects=" +
                  std::to_string(k + 1));
  return true;
}

std::vector<Status> RdmaTransport::RangeIntoMulti(
    const std::string& node, const std::vector<BlockKey>& keys,
    const std::vector<RangeDstMulti>& destinations,
    std::vector<size_t>* out_lengths) {
  const size_t count = keys.size();
  if (out_lengths) out_lengths->assign(count, 0);
  if (destinations.size() != count) return InvalidStatuses(count);
  std::vector<Status> result(count, Status::kIOError);
  if (count == 0) return result;
  std::vector<char> bad(count, 0);
  std::vector<size_t> capacities(count, 0);
  size_t valid_count = 0;
  for (size_t i = 0; i < count; ++i) {
    bool invalid =
        destinations[i].payloads.size() > rdma::kV2MaxGetTargets;
    size_t capacity = 0;
    for (const auto& destination : destinations[i].payloads) {
      size_t next = 0;
      if (!ValidBuffer(destination.first, destination.second) ||
          destination.second > std::numeric_limits<uint32_t>::max() ||
          !CheckedAddSize(capacity, destination.second, &next)) {
        invalid = true;
        break;
      }
      capacity = next;
    }
    if (!invalid && NoteBlock(capacity)) invalid = true;
    capacities[i] = capacity;
    if (invalid) {
      bad[i] = 1;
      result[i] = Status::kInvalid;
    } else {
      ++valid_count;
    }
  }
  if (valid_count == 0) return result;

  for (int attempt = 0; attempt < 2; ++attempt) {
    if (attempt != 0)
      stale_pool_retries_.fetch_add(1, std::memory_order_relaxed);
    std::fill(result.begin(), result.end(), Status::kIOError);
    for (size_t i = 0; i < count; ++i)
      if (bad[i]) result[i] = Status::kInvalid;
    if (out_lengths) out_lengths->assign(count, 0);
    bool from_pool = false;
    Conn* conn = Acquire(node, Lane::kData, &from_pool, attempt > 0,
                         std::min(valid_count, depth_));
    if (!conn) return result;
    rdma::RcEndpoint& ep = conn->ep;
    const size_t window = std::min(
        ep.window(), static_cast<size_t>(conn->lease.credits));
    bool conn_ok = true;
    // Failure attribution for Destroy: post/encode/register failures keep the
    // local-rail default; a failed reap window is classified from its WC
    // evidence; wire decode failures blame the peer.
    rdma::RailCompletion completion = rdma::RailCompletion::kRailFailure;
    for (size_t base = 0; base < count && conn_ok; base += window) {
      const size_t width = std::min(window, count - base);
      std::vector<std::vector<ibv_mr*>> mrs(width);
      size_t posted = 0;
      for (size_t slot = 0; slot < width; ++slot) {
        const size_t item_index = base + slot;
        if (bad[item_index]) continue;
        const RangeDstMulti& destination = destinations[item_index];
        mrs[slot].assign(destination.payloads.size(), nullptr);
        std::vector<RdmaWriteTarget> targets;
        targets.reserve(destination.payloads.size());
        for (size_t seg = 0; seg < destination.payloads.size(); ++seg) {
          const auto& payload = destination.payloads[seg];
          if (payload.second == 0) continue;
          mrs[slot][seg] =
              ep.RegisterTransient(payload.first, payload.second);
          if (!mrs[slot][seg]) {
            conn_ok = false;
            break;
          }
          targets.push_back(
              {reinterpret_cast<uint64_t>(payload.first),
               mrs[slot][seg]->rkey,
               static_cast<uint32_t>(payload.second)});
        }
        if (!conn_ok) break;
        size_t frame_len = 0;
        if (!EncodeRdmaGetReq(ep.sbuf(slot), ep.cap(), keys[item_index],
                              0, capacities[item_index], targets,
                              &frame_len) ||
            !ep.PostRecv(slot) || !ep.PostSend(slot, frame_len)) {
          conn_ok = false;
          break;
        }
        ++posted;
        v2_get_writes_.fetch_add(1, std::memory_order_relaxed);
      }
      std::vector<uint32_t> reply_bytes;
      bool timed_out = false;
      ibv_wc_status wc_status = IBV_WC_SUCCESS;
      bool had_wcs = false;
      if (conn_ok &&
          !ReapPosted(ep, posted, width, &reply_bytes, BatchTimeout(),
                      &timed_out, &wc_status, &had_wcs)) {
        conn_ok = false;
        completion = rdma::ClassifyCompletion(wc_status, had_wcs);
      }
      if (timed_out)
        completion_timeouts_.fetch_add(1, std::memory_order_relaxed);
      if (!conn_ok) break;
      for (auto& slot_mrs : mrs)
        for (ibv_mr* mr : slot_mrs) ep.ReleaseTransient(mr);
      for (size_t slot = 0; slot < width; ++slot) {
        const size_t item_index = base + slot;
        if (bad[item_index]) continue;
        Status status;
        uint64_t data_len = 0;
        uint64_t value_len = 0;
        if (reply_bytes[slot] < kRespPrefix ||
            !conn->Decode(ep.rbuf(slot), &status, &data_len,
                          capacities[item_index], &value_len) ||
            value_len > std::numeric_limits<size_t>::max()) {
          conn_ok = false;
          completion = rdma::RailCompletion::kEndpointFailure;
          break;
        }
        result[item_index] = status;
        if (status == Status::kOk && out_lengths) {
          (*out_lengths)[item_index] = static_cast<size_t>(value_len);
        }
      }
    }
    if (conn_ok) {
      Release(node, Lane::kData, conn);
      return result;
    }
    Destroy(conn, completion);
    if (from_pool) continue;
    return result;
  }
  return result;
}

}  // namespace dfkv
