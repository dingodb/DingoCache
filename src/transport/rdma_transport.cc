#include "common/config_dump.h"
#include "transport/rdma_transport.h"

#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
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
#include "transport/rdma_operation.h"
#include "transport/rdma_connection_lifecycle.h"

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
uint64_t EnvU64(const char* name, uint64_t dflt) {
  const char* value = std::getenv(name);
  if (!value || !*value) return dflt;
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  return errno == 0 && end != value && *end == '\0' && parsed > 0
             ? static_cast<uint64_t>(parsed)
             : dflt;
}


bool AbortMultiWrWindow(size_t one_based_window, size_t window_count) {
  if (window_count <= 1) return false;
  const char* value = std::getenv("DFKV_RDMA_TEST_ABORT_MULTIWR_WINDOW");
  if (!value || !*value) return false;
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  return errno == 0 && end != value && *end == '\0' &&
         parsed == one_based_window;
}


bool InjectLocalRailFailure(int zero_based_attempt) {
  const char* value =
      std::getenv("DFKV_RDMA_TEST_LOCAL_RAIL_FAILURE_ATTEMPTS");
  if (!value || !*value) return false;
  const unsigned long wanted =
      static_cast<unsigned long>(zero_based_attempt + 1);
  const char* cursor = value;
  while (*cursor) {
    errno = 0;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(cursor, &end, 10);
    if (errno != 0 || end == cursor) return false;
    if (parsed == wanted) return true;
    if (*end == '\0') return false;
    if (*end != ',') return false;
    cursor = end + 1;
  }
  return false;
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

std::string JoinDevices(const std::vector<std::string>& devices,
                        const char* separator = ",") {
  std::string out;
  for (const auto& device : devices) {
    if (!out.empty()) out += separator;
    out += device;
  }
  return out;
}

// Test-only deterministic completion faults for data operations.
// DFKV_RDMA_TEST_COMPLETION_FAULT injects local IBV_WC_GENERAL_ERR evidence;
// DFKV_RDMA_TEST_ENDPOINT_COMPLETION_FAULT injects remote
// IBV_WC_REM_ACCESS_ERR evidence. Their value is a comma-separated list of
// "<logical-call>:<attempt>:<completion-window>" specs, all one-based.
// Eligible logical calls are numbered only while one of the env vars is set.
// Each operation owns its injection state and passes it explicitly to
// data-path reaps, so pooled connection maintenance cannot consume a target
// window. A matching successfully reaped window is surfaced once through the
// production ReapPosted/ClassifyCompletion path.
struct CompletionFaultTarget {
  uint64_t call = 0;
  uint64_t attempt = 0;
  uint64_t window = 0;
};

class CompletionFaultOperation {
 public:
  explicit CompletionFaultOperation(std::atomic<uint64_t>* calls,
                                    bool enabled = true) {
    if (!enabled) return;
    const char* cursor =
        std::getenv("DFKV_RDMA_TEST_ENDPOINT_COMPLETION_FAULT");
    if (cursor && *cursor) {
      status_ = IBV_WC_REM_ACCESS_ERR;
    } else {
      cursor = std::getenv("DFKV_RDMA_TEST_COMPLETION_FAULT");
    }
    if (!cursor || !*cursor) return;
    std::vector<CompletionFaultTarget> parsed;
    for (;;) {
      CompletionFaultTarget target;
      uint64_t* fields[] = {&target.call, &target.attempt, &target.window};
      char* end = nullptr;
      for (size_t i = 0; i < 3; ++i) {
        errno = 0;
        *fields[i] = std::strtoull(cursor, &end, 10);
        if (errno != 0 || end == cursor || *fields[i] == 0 ||
            (i != 2 ? *end != ':' : (*end != '\0' && *end != ','))) {
          return;
        }
        if (i != 2) cursor = end + 1;
      }
      parsed.push_back(target);
      if (*end == '\0') break;
      cursor = end + 1;
      if (!*cursor) return;
    }

    const uint64_t logical_call =
        calls->fetch_add(1, std::memory_order_relaxed) + 1;
    for (const auto& target : parsed) {
      if (target.call == logical_call) {
        targets_.push_back(target);
      }
    }
  }

  void BeginAttempt(int zero_based_attempt) {
    attempt_ = static_cast<uint64_t>(zero_based_attempt) + 1;
    window_ = 0;
  }

  bool InjectAfterCompletedWindow() {
    if (targets_.empty()) return false;
    ++window_;
    for (const auto& target : targets_) {
      if (target.attempt == attempt_ && target.window == window_) return true;
    }
    return false;
  }
  ibv_wc_status status() const { return status_; }

 private:
  std::vector<CompletionFaultTarget> targets_;
  uint64_t attempt_ = 0;
  ibv_wc_status status_ = IBV_WC_GENERAL_ERR;
  uint64_t window_ = 0;
};

// Local post/CQ/verbs API failures happen below the work-completion layer, so
// no real WC status exists for them. The reap helpers report such failures as
// this local-fault bucket so every datapath teardown can funnel through
// rdma::ClassifyCompletion and attribute them kRailFailure.
constexpr ibv_wc_status kLocalFaultWcStatus = IBV_WC_GENERAL_ERR;

// A response DONE is the success fence: the responder posts it after signaled
// WRITEs on the same RC QP. An ambiguous attempt must obtain server proof that
// its writer token is retired and every WRITE CQE is terminal before the same
// caller/CUDA destination can be retried, returned, or deregistered.
bool RetireRemoteWriter(const std::string& node, uint64_t token,
                        int connect_ms, int io_ms) {
  if (token == 0) return false;
  const int fd = net::Dial(node, connect_ms, io_ms);
  if (fd < 0) return false;
  char request[rdma::kDevNameBytes];
  rdma::EncodeDevFrame(rdma::kV2RetireWriterDevice, token, request,
                       rdma::kDevProtoV2);
  char proof[rdma::kV2RetireProofBytes];
  const bool retired =
      net::WriteAll(fd, request, sizeof(request)) &&
      net::ReadAll(fd, proof, sizeof(proof)) &&
      rdma::ParseV2RetireProof(proof, token);
  ::close(fd);
  return retired;
}

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
                bool* had_completions = nullptr,
                CompletionFaultOperation* completion_fault = nullptr) {
  if (timed_out) *timed_out = false;
  if (worst_status) *worst_status = IBV_WC_SUCCESS;
  if (had_completions) *had_completions = false;
  rbytes->assign(slot_count, 0);
  if (posted == 0) return true;
  std::vector<ibv_wc> wcs(2 * posted);
  int need = static_cast<int>(2 * posted);
  CompletionDeadline deadline(timeout_ms);
  while (need > 0) {
    if (rdma::TestConsumeBlockedResponderWriteFault()) {
      if (worst_status) *worst_status = kLocalFaultWcStatus;
      return false;
    }
    const int remaining_ms = deadline.Remaining();
    if (remaining_ms == 0) {
      if (timed_out) *timed_out = true;
      return false;
    }
    const int got = ep.WaitComp(wcs.data(), static_cast<int>(wcs.size()),
                                remaining_ms);
    if (got <= 0) {
      if (rdma::TestConsumeBlockedResponderWriteFault()) {
        if (worst_status) *worst_status = kLocalFaultWcStatus;
        return false;
      }
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
  if (completion_fault && completion_fault->InjectAfterCompletedWindow()) {
    if (worst_status) *worst_status = completion_fault->status();
    if (had_completions) *had_completions = true;
    return false;
  }
  return true;
}

bool ReapWindow(rdma::RcEndpoint& ep, size_t width,
                std::vector<uint32_t>* rbytes, int timeout_ms,
                bool* timed_out = nullptr,
                ibv_wc_status* worst_status = nullptr,
                bool* had_completions = nullptr,
                CompletionFaultOperation* completion_fault = nullptr) {
  return ReapPosted(ep, width, width, rbytes, timeout_ms, timed_out,
                    worst_status, had_completions, completion_fault);
}

// Post ordinary SEND requests with one status RECV per slot.
bool RunWindow(rdma::RcEndpoint& ep, const std::vector<size_t>& slen,
               std::vector<uint32_t>* rbytes, int timeout_ms,
               bool* timed_out = nullptr,
               ibv_wc_status* worst_status = nullptr,
               bool* had_completions = nullptr,
               CompletionFaultOperation* completion_fault = nullptr) {
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
                    worst_status, had_completions, completion_fault);
}
std::shared_ptr<rdma::ResourceBudget> ProcessResourceBudget(
    const rdma::ResourceRequest& limit) {
  static auto budget = std::make_shared<rdma::ResourceBudget>(limit);
  return budget;
}

// Client-local failures that occur before dialing the peer must remain neutral
// to PeerHealth. Flip only slots still carrying the kIOError default;
// deterministic kInvalid verdicts stand.
void MarkClientLocalFailure(std::vector<Status>* result, Status status) {
  for (auto& s : *result)
    if (s == Status::kIOError) s = status;
}

}  // namespace

struct RdmaTransport::Conn {
  rdma::RcEndpoint ep;
  rdma::RecvSegmentInfo recv_segment;
  uint64_t writer_token = 0;
  size_t rail_index = 0;
  uint64_t peer_publication = 0;
  std::string peer_id;
  uint64_t remote_lease_generation = 0;
  bool remote_lease_held = false;
  bool remote_recovery_probe = false;
  rdma::RailLease lease;
  uint64_t lease_started_us = 0;
  bool credit_held = false;
  rdma::ResourceRequest budget_request;
  bool budget_held = false;
  bool visited = true;  // guarded by RdmaTransport::mu_ while idle

  rdma::ConnectionLifecycle lifecycle;
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
  const char* tiers_env = std::getenv("DFKV_RDMA_RAIL_TIERS");
  const std::string tiers_spec = tiers_env ? tiers_env : "";
  peer_topology_required_ = !tiers_spec.empty();
  if (peer_topology_required_ && filter.empty()) {
    const std::string error =
        "DFKV_RDMA_RAIL_TIERS requires an explicit DFKV_RDMA_DEV list";
    DFKV_LOG_ERROR(error);
    throw std::runtime_error(error);
  }
  std::string tiers_error;
  if (!rdma::ParseRailTiers(tiers_spec, devs_, &rail_tiers_, &tiers_error)) {
    DFKV_LOG_ERROR(tiers_error);
    throw std::runtime_error(tiers_error);
  }
  peer_topologies_ = std::make_unique<rdma::PeerTopologyStore>(
      devs_, rail_tiers_, peer_topology_required_);
  const int remote_base_cooldown_ms = EnvBoundedInt(
      "DFKV_RDMA_REMOTE_RAIL_COOLDOWN_MS", 2000, 30000);
  const int remote_max_cooldown_ms = std::max(
      remote_base_cooldown_ms,
      EnvBoundedInt("DFKV_RDMA_REMOTE_RAIL_MAX_COOLDOWN_MS", 30000, 30000));
  remote_rail_health_ = std::make_unique<RemoteRailHealth>(
      RemoteRailHealthConfig{
          static_cast<uint64_t>(remote_base_cooldown_ms) * 1000,
          static_cast<uint64_t>(remote_max_cooldown_ms) * 1000});
  config_dump::RecordResolved("DFKV_RDMA_REMOTE_RAIL_COOLDOWN_MS",
                              std::to_string(remote_base_cooldown_ms));
  config_dump::RecordResolved("DFKV_RDMA_REMOTE_RAIL_MAX_COOLDOWN_MS",
                              std::to_string(remote_max_cooldown_ms));
  config_dump::RecordResolved(
      "DFKV_RDMA_RAIL_TIERS",
      peer_topology_required_ ? tiers_spec : JoinDevices(devs_, "|"));
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
  const size_t endpoint_limit = static_cast<size_t>(
      EnvBoundedInt("DFKV_RDMA_ENDPOINT_CACHE_MAX", 256, 65536));
  const size_t qp_limit = static_cast<size_t>(
      EnvBoundedInt("DFKV_RDMA_QP_BUDGET",
                    static_cast<int>(endpoint_limit), 65536));
  const size_t wr_limit = static_cast<size_t>(EnvU64(
      "DFKV_RDMA_WR_BUDGET",
      static_cast<uint64_t>(endpoint_limit) * depth_));
  const uint64_t slot_bytes =
      static_cast<uint64_t>(rdma::V2SlotSize(static_cast<size_t>(declared_)));
  const uint64_t default_registered_budget =
      slot_bytes != 0 &&
              endpoint_limit <=
                  std::numeric_limits<uint64_t>::max() / slot_bytes / depth_
          ? slot_bytes * depth_ * endpoint_limit
          : std::numeric_limits<uint64_t>::max();
  const uint64_t registered_limit = EnvU64(
      "DFKV_RDMA_REGISTERED_BYTES_BUDGET", default_registered_budget);
  resource_acquire_ms_ =
      EnvBoundedInt("DFKV_RDMA_RESOURCE_ACQUIRE_MS", 10000, 600000);
  resource_budget_ = ProcessResourceBudget(
      {endpoint_limit, qp_limit, wr_limit, registered_limit});
  // Any explicitly pinned budget dimension turns the whole autoscale off:
  // partial-manual sizing recreated exactly the "QP raised but WR/registered
  // still starve" trap of 0812-002 §6.1, so explicit config wins wholesale.
  budget_autoscale_enabled_ =
      !std::getenv("DFKV_RDMA_ENDPOINT_CACHE_MAX") &&
      !std::getenv("DFKV_RDMA_QP_BUDGET") &&
      !std::getenv("DFKV_RDMA_WR_BUDGET") &&
      !std::getenv("DFKV_RDMA_REGISTERED_BYTES_BUDGET");
  config_dump::RecordResolved("budget_autoscale",
                              budget_autoscale_enabled_ ? "on" : "off (explicit budget env)");
  const rdma::ResourceRequest effective_budget = resource_budget_->limit();
  config_dump::RecordResolved("DFKV_RDMA_ENDPOINT_CACHE_MAX",
                              std::to_string(effective_budget.endpoints));
  config_dump::RecordResolved("DFKV_RDMA_QP_BUDGET",
                              std::to_string(effective_budget.qps));
  config_dump::RecordResolved("DFKV_RDMA_WR_BUDGET",
                              std::to_string(effective_budget.wr_slots));
  config_dump::RecordResolved(
      "DFKV_RDMA_REGISTERED_BYTES_BUDGET",
      std::to_string(effective_budget.registered_bytes));
  config_dump::RecordResolved("DFKV_RDMA_RESOURCE_ACQUIRE_MS",
                              std::to_string(resource_acquire_ms_));
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
  // Keep live pooled QPs warm by default. The 15 s interval is below the
  // recommended 30 s server dead-client reaper; set 0 explicitly to disable.
  if (const char* value = std::getenv("DFKV_RDMA_KEEPALIVE_MS");
      value && *value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (errno == 0 && end != value && *end == '\0' && parsed >= 0) {
      keepalive_ms_ = static_cast<int>(
          std::min<long>(parsed, std::numeric_limits<int>::max()));
    }
  }
  config_dump::RecordResolved("DFKV_RDMA_KEEPALIVE_MS",
                              std::to_string(keepalive_ms_));
  // Idle-connection pool cap per node and lane. The pool naturally bounds at
  // peak concurrency; this guards against a transient thread spike retaining
  // enough QPs to consume the server's shared receive segment. Sixteen covers
  // the measured 0064 SGLang C32 fan-out with 2x per-lane headroom; raise it
  // only for a workload that proves repeated connection churn.
  pool_max_ = static_cast<size_t>(EnvInt("DFKV_RDMA_POOL_MAX", 16));
  config_dump::RecordResolved("DFKV_RDMA_POOL_MAX",
                              std::to_string(pool_max_));
  rail_conns_ = std::make_unique<std::atomic<uint64_t>[]>(devs_.size());
  if (keepalive_ms_ > 0)
    keepalive_thread_ = std::thread(&RdmaTransport::KeepaliveLoop, this);
}

RdmaTransport::~RdmaTransport() {
  keepalive_stop_.store(true, std::memory_order_relaxed);
  keepalive_cv_.notify_all();
  if (keepalive_thread_.joinable()) keepalive_thread_.join();

  std::vector<Conn*> idle;
  {
    std::lock_guard<std::mutex> lk(mu_);
    const auto detach = [&](auto& pools) {
      for (auto& [node, connections] : pools) {
        (void)node;
        idle.insert(idle.end(), connections.begin(), connections.end());
        connections.clear();
      }
      pools.clear();
    };
    detach(pool_);
    detach(sg_pool_);
    detach(control_pool_);
  }
  for (Conn* conn : idle) Destroy(conn);
}

bool RdmaTransport::KeepaliveConn(Conn* conn,
                                  rdma::RailCompletion* failure) {
  if (!conn || !failure) return false;
  *failure = rdma::RailCompletion::kAdmission;
  keepalive_attempts_.fetch_add(1, std::memory_order_relaxed);
  rdma::RcEndpoint& ep = conn->ep;
  conn->Encode(ep.sbuf(0), WireOp::kMembers, BlockKey{}, 0, 0, 0);
  if (!ep.PostRecv(0) || !ep.PostSend(0, kReqPrefix)) {
    *failure = rdma::RailCompletion::kRailFailure;
    keepalive_failures_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  std::vector<uint32_t> reply_bytes;
  bool timed_out = false;
  ibv_wc_status wc_status = IBV_WC_SUCCESS;
  bool had_wcs = false;
  const int timeout_ms =
      op_timeout_ms_ < 0 ? 100 : std::min(op_timeout_ms_, 100);
  if (!ReapWindow(ep, 1, &reply_bytes, timeout_ms, &timed_out, &wc_status,
                  &had_wcs)) {
    *failure = rdma::ClassifyCompletion(wc_status, had_wcs);
    keepalive_failures_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (reply_bytes.empty() || reply_bytes[0] < kRespPrefix) {
    *failure = rdma::RailCompletion::kEndpointFailure;
    keepalive_failures_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  Status status = Status::kIOError;
  uint64_t data_len = 0;
  if (!conn->Decode(ep.rbuf(0), &status, &data_len,
                    rdma::kV2ControlResponseMax) ||
      data_len > reply_bytes[0] - kRespPrefix) {
    *failure = rdma::RailCompletion::kEndpointFailure;
    keepalive_failures_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (status != Status::kOk) {
    keepalive_failures_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  keepalive_successes_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void RdmaTransport::KeepaliveLoop() {
  struct IdleConn {
    std::string node;
    Lane lane;
    Conn* conn;
  };
  std::unique_lock<std::mutex> wait_lock(keepalive_mu_);
  while (!keepalive_cv_.wait_for(
      wait_lock, std::chrono::milliseconds(keepalive_ms_),
      [&] { return keepalive_stop_.load(std::memory_order_relaxed); })) {
    wait_lock.unlock();
    std::vector<IdleConn> idle;
    {
      std::lock_guard<std::mutex> lock(mu_);
      const auto collect = [&](auto& pools, Lane lane) {
        for (auto& [node, connections] : pools) {
          for (Conn* conn : connections) {
            if (conn->lifecycle.Activate())
              idle.push_back(IdleConn{node, lane, conn});
          }
          connections.clear();
        }
      };
      collect(pool_, Lane::kData);
      collect(sg_pool_, Lane::kSgData);
      collect(control_pool_, Lane::kControl);
    }
    for (const auto& entry : idle) {
      if (!peer_topologies_->IsCurrent(entry.node, entry.conn->peer_id,
                                       entry.conn->peer_publication)) {
        stale_publication_reaps_.fetch_add(1, std::memory_order_relaxed);
        Destroy(entry.conn, rdma::RailCompletion::kAdmission);
        continue;
      }
      if (keepalive_stop_.load(std::memory_order_relaxed)) {
        Destroy(entry.conn, rdma::RailCompletion::kAdmission);
        continue;
      }
      if (!entry.conn->peer_id.empty()) {
        const auto remote = remote_rail_health_->TryAcquire(
            entry.conn->peer_id, entry.conn->rail_index,
            rdma::RailPolicy::NowMicros());
        if (!remote) {
          Destroy(entry.conn, rdma::RailCompletion::kAdmission);
          continue;
        }
        entry.conn->remote_lease_generation = remote->generation;
        entry.conn->remote_lease_held = true;
        entry.conn->remote_recovery_probe = remote->recovery_probe;
      }
      rdma::RailCompletion failure = rdma::RailCompletion::kAdmission;
      if (!KeepaliveConn(entry.conn, &failure))
        Destroy(entry.conn, failure);
      else
        Release(entry.node, entry.lane, entry.conn);
    }
    wait_lock.lock();
  }
}

void RdmaTransport::RetireIdlePeerRail(const std::string& peer_id,
                                       size_t local_rail) {
  if (peer_id.empty()) return;
  std::vector<Conn*> retired;
  {
    std::lock_guard<std::mutex> lock(mu_);
    const auto reap = [&](auto& pools) {
      for (auto map_it = pools.begin(); map_it != pools.end();) {
        auto& connections = map_it->second;
        for (auto it = connections.begin(); it != connections.end();) {
          Conn* conn = *it;
          if (conn->peer_id != peer_id || conn->rail_index != local_rail ||
              !conn->lifecycle.RequestRetire()) {
            ++it;
            continue;
          }
          retired.push_back(conn);
          it = connections.erase(it);
        }
        if (connections.empty())
          map_it = pools.erase(map_it);
        else
          ++map_it;
      }
    };
    reap(pool_);
    reap(sg_pool_);
    reap(control_pool_);
  }
  for (Conn* conn : retired)
    Destroy(conn, rdma::RailCompletion::kAdmission);
}

void RdmaTransport::CompleteRemote(const std::string& peer_id,
                                   size_t local_rail, uint64_t generation,
                                   RemoteRailOutcome outcome) {
  if (peer_id.empty() || generation == 0) return;
  const uint64_t now = rdma::RailPolicy::NowMicros();
  const RemoteRailCompletion completion = remote_rail_health_->Complete(
      peer_id, local_rail, generation, outcome, now);
  const std::string dev =
      local_rail < devs_.size() && !devs_[local_rail].empty()
          ? devs_[local_rail]
          : "default";
  if (completion.transition == RemoteRailTransition::kCooled) {
    DFKV_LOG_WARN(
        "rdma: remote endpoint rail cooled peer_id=" + peer_id +
        " dev=" + dev + " cooldown_ms=" +
        std::to_string(completion.cooldown_until_us > now
                           ? (completion.cooldown_until_us - now) / 1000
                           : 0));
    RetireIdlePeerRail(peer_id, local_rail);
  } else if (completion.transition == RemoteRailTransition::kRecovered) {
    DFKV_LOG_INFO("rdma: remote endpoint rail recovered peer_id=" + peer_id +
                  " dev=" + dev);
  }
}

void RdmaTransport::CompleteRemoteLease(Conn* c,
                                        RemoteRailOutcome outcome) {
  if (!c || !c->remote_lease_held) return;
  c->remote_lease_held = false;
  CompleteRemote(c->peer_id, c->rail_index, c->remote_lease_generation,
                 outcome);
}

void RdmaTransport::Destroy(Conn* c, rdma::RailCompletion completion) {
  if (!c) return;

  c->lifecycle.BeginDrain();
  const bool credit_held = c->credit_held;
  const rdma::RailLease lease = c->lease;
  const uint64_t lease_started_us = c->lease_started_us;
  const bool release_budget = c->budget_held;
  const rdma::ResourceRequest budget_request = c->budget_request;
  CompleteRemoteLease(
      c, completion == rdma::RailCompletion::kEndpointFailure
             ? RemoteRailOutcome::kEndpointFailure
             : RemoteRailOutcome::kAbandon);

  // Direct GET failure paths have already obtained responder-retirement proof.
  // Deletion releases terminal verbs resources; it is not a DMA fence.
  delete c;
  if (credit_held) {
    const uint64_t now = rdma::RailPolicy::NowMicros();
    rail_policy_->Complete(lease, now - lease_started_us, completion, now);
  }
  if (release_budget) resource_budget_->Release(budget_request);
}

void RdmaTransport::QuarantineAmbiguousGet(
    Conn* c, void* destination_hold, size_t destination_bytes,
    const char* path, rdma::RailCompletion completion) {
  if (!c || !destination_hold) return;
  c->lifecycle.BeginDrain();
  CompleteRemoteLease(c, RemoteRailOutcome::kEndpointFailure);
  if (c->credit_held) {
    const uint64_t now = rdma::RailPolicy::NowMicros();
    rail_policy_->Complete(c->lease, now - c->lease_started_us, completion,
                           now);
    c->credit_held = false;
  }
  const uint64_t count =
      ambiguous_get_quarantines_.fetch_add(1, std::memory_order_relaxed) + 1;
  const uint64_t bytes =
      ambiguous_get_quarantined_bytes_.fetch_add(
          destination_bytes, std::memory_order_relaxed) +
      destination_bytes;
  {
    std::lock_guard<std::mutex> lock(quarantine_mu_);
    quarantined_get_connections_.push_back(c);
    quarantined_get_destinations_.push_back(destination_hold);
  }
  DFKV_LOG_ERROR(
      "rdma: quarantined ambiguous staged GET path=" + std::string(path) +
      " peer_id=" + c->peer_id +
      " rail=" + std::to_string(c->rail_index) +
      " quarantines=" + std::to_string(count) +
      " bytes=" + std::to_string(bytes));
}

bool RdmaTransport::EvictOneIdle() {
  Conn* victim = nullptr;
  {
    std::lock_guard<std::mutex> lock(mu_);
    const auto scan = [&](auto& pools) {
      for (auto& [node, connections] : pools) {
        (void)node;
        for (auto it = connections.begin(); it != connections.end(); ++it) {
          if ((*it)->visited) {
            (*it)->visited = false;
            continue;
          }
          if (!(*it)->lifecycle.RequestRetire()) continue;
          victim = *it;
          connections.erase(it);
          return true;
        }
      }
      return false;
    };
    for (int pass = 0; pass < 2 && victim == nullptr; ++pass) {
      if (scan(pool_) || scan(sg_pool_) || scan(control_pool_)) break;
    }
  }
  if (!victim) return false;
  Destroy(victim);
  endpoint_cache_evictions_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void RdmaTransport::OnTopologyHint(size_t nodes) {
  if (!budget_autoscale_enabled_ || nodes == 0) return;
  // Same-size adoptions are the steady state (MDS re-polls); dedup them so
  // the hint costs one relaxed load. Shrinking rings keep the high-water
  // budget: limits are ceilings, an oversized ceiling costs nothing.
  size_t previous = topology_nodes_.load(std::memory_order_relaxed);
  while (nodes > previous) {
    if (!topology_nodes_.compare_exchange_weak(previous, nodes,
                                               std::memory_order_relaxed))
      continue;
    const uint64_t slot_bytes = static_cast<uint64_t>(
        rdma::V2SlotSize(static_cast<size_t>(declared_)));
    const rdma::ResourceRequest target = rdma::AdaptiveBudgetTarget(
        nodes, devs_.size(), depth_, slot_bytes, /*floor_endpoints=*/256);
    if (resource_budget_->RaiseLimit(target)) {
      const rdma::ResourceRequest limit = resource_budget_->limit();
      DFKV_LOG_INFO(
          "rdma: resource budget autoscaled for ring=" +
          std::to_string(nodes) + " rails=" + std::to_string(devs_.size()) +
          ": endpoints=" + std::to_string(limit.endpoints) +
          " qps=" + std::to_string(limit.qps) +
          " wr=" + std::to_string(limit.wr_slots) +
          " registered_bytes=" + std::to_string(limit.registered_bytes));
    }
    return;
  }
}

void RdmaTransport::OnPeerTopology(const PeerTopology& topology) {
  std::vector<Conn*> stale;
  {
    // Publish and detach under the same pool lock. Acquire's publication check
    // uses this lock too, so no idle endpoint can cross the update
    // linearization point and become active with the retired incarnation.
    std::lock_guard<std::mutex> lock(mu_);
    if (!peer_topologies_->Update(topology)) return;
    const auto current_snapshot =
        peer_topologies_->Snapshot(topology.peer_addr);
    const auto reap = [&](auto& pools) {
      const auto found = pools.find(topology.peer_addr);
      if (found == pools.end()) return;
      auto& connections = found->second;
      for (auto it = connections.begin(); it != connections.end();) {
        Conn* conn = *it;
        const bool current =
            topology.present && conn->peer_id == current_snapshot->peer_id &&
            conn->peer_publication == current_snapshot->publication;
        if (current || !conn->lifecycle.RequestRetire()) {
          ++it;
          continue;
        }
        stale.push_back(conn);
        it = connections.erase(it);
      }
      if (connections.empty()) pools.erase(found);
    };
    reap(pool_);
    reap(sg_pool_);
    reap(control_pool_);
  }
  peer_topology_updates_.fetch_add(1, std::memory_order_relaxed);
  for (Conn* conn : stale)
    Destroy(conn, rdma::RailCompletion::kAdmission);
  stale_publication_reaps_.fetch_add(stale.size(), std::memory_order_relaxed);
}

void RdmaTransport::OnPeerIdentities(
    const std::vector<std::string>& live_peer_ids) {
  remote_rail_health_->Reconcile(live_peer_ids);
}

uint16_t RdmaTransport::LearnedDepth(const std::string& node, Lane lane) {
  const uint16_t full = static_cast<uint16_t>(std::min<size_t>(depth_, 256));
  std::lock_guard<std::mutex> lk(depth_mu_);
  const auto& map =
      lane == Lane::kControl ? learned_control_depth_ : learned_data_depth_;
  const auto it = map.find(node);
  return it == map.end() ? full : std::min(full, it->second);
}

void RdmaTransport::NoteNegotiatedDepth(const std::string& node, Lane lane,
                                        uint16_t remote_depth) {
  if (remote_depth == 0) return;
  std::lock_guard<std::mutex> lk(depth_mu_);
  auto& map =
      lane == Lane::kControl ? learned_control_depth_ : learned_data_depth_;
  const auto it = map.find(node);
  if (it == map.end() || remote_depth < it->second) map[node] = remote_depth;
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
      rdma::V2ProbeSupportsWriterRetirement(reply);
  ::close(fd);
  if (!ok) v2_probe_failures_.fetch_add(1, std::memory_order_relaxed);
  return ok;
}

RdmaTransport::AcquireResult RdmaTransport::Acquire(
    const std::string& node, Lane lane, const AcquireOptions& options) {
  AcquireResult result;
  const auto peer_snapshot =
      options.peer ? options.peer : peer_topologies_->Snapshot(node);
  if (!peer_snapshot || !rdma::HasRail(peer_snapshot->peer_healthy)) {
    no_compatible_rail_.fetch_add(1, std::memory_order_relaxed);
    result.failure = AcquireFailure::kNoCompatibleRail;
    result.status = Status::kNoCompatibleRail;
    return result;
  }
  const uint32_t requested = static_cast<uint32_t>(
      std::min<size_t>(options.requested_credits,
                       std::numeric_limits<uint32_t>::max()));
  // NUMA preference, credits, latency, and local quarantine are applied only
  // inside the highest tier in the captured immutable peer snapshot that
  // intersects runtime-enabled local topology and remote endpoint health.
  const int caller_node = numa_aware_ ? numa::CurrentNode() : -1;
  const auto candidates =
      topology_->CandidatesFor(caller_node, numa_aware_);
  if (candidates.locality == rdma::RailLocality::kCallerUnknown) {
    numa_caller_unknown_fallbacks_.fetch_add(1, std::memory_order_relaxed);
  } else if (candidates.locality == rdma::RailLocality::kNoLocal) {
    numa_no_local_fallbacks_.fetch_add(1, std::memory_order_relaxed);
  }
  const auto local_enabled =
      rdma::UnionRailMasks(candidates.allowed, candidates.fallback);
  const auto topology_compatible =
      rdma::IntersectRailMasks(peer_snapshot->peer_healthy, local_enabled);
  RailMask admission_excluded = options.excluded;
  admission_excluded.resize(devs_.size(), 0);
  std::optional<rdma::RailLease> lease;
  std::optional<RemoteRailLease> remote_lease;
  uint64_t lease_started = 0;
  size_t ridx = 0;
  for (size_t selection_attempt = 0; selection_attempt < devs_.size();
       ++selection_attempt) {
    const uint64_t now = rdma::RailPolicy::NowMicros();
    const auto remote_compatible =
        peer_snapshot->peer_id.empty()
            ? topology_compatible
            : remote_rail_health_->AllowedMask(
                  peer_snapshot->peer_id, topology_compatible, now);
    const auto highest_tier =
        rdma::HighestCompatibleTier(rail_tiers_, remote_compatible);
    if (!rdma::HasRail(highest_tier)) break;
    lease = rdma::AcquireHighestTier(
        *rail_policy_, requested, rail_backpressure_us_, highest_tier,
        candidates.allowed, candidates.fallback, admission_excluded);
    if (!lease) {
      result.failure = AcquireFailure::kAdmission;
      result.status = Status::kResourceExhausted;
      return result;
    }
    ridx = lease->rail;
    result.attempted_rail = ridx;
    lease_started = rdma::RailPolicy::NowMicros();
    if (peer_snapshot->peer_id.empty()) break;
    remote_lease = remote_rail_health_->TryAcquire(
        peer_snapshot->peer_id, ridx, lease_started);
    if (remote_lease) break;
    const uint64_t denied_at = rdma::RailPolicy::NowMicros();
    rail_policy_->Complete(*lease, denied_at - lease_started,
                           rdma::RailCompletion::kAdmission, denied_at);
    admission_excluded[ridx] = 1;
    lease.reset();
  }
  if (!lease) {
    no_compatible_rail_.fetch_add(1, std::memory_order_relaxed);
    result.failure = AcquireFailure::kNoCompatibleRail;
    result.status = Status::kNoCompatibleRail;
    return result;
  }
  const auto complete_unowned_lease =
      [&](rdma::RailCompletion completion) {
        if (remote_lease) {
          CompleteRemote(
              peer_snapshot->peer_id, ridx, remote_lease->generation,
              completion == rdma::RailCompletion::kEndpointFailure
                  ? RemoteRailOutcome::kEndpointFailure
                  : RemoteRailOutcome::kAbandon);
        }
        const uint64_t now = rdma::RailPolicy::NowMicros();
        rail_policy_->Complete(*lease, now - lease_started, completion, now);
      };

  std::vector<std::pair<void*, size_t>> pools;
  std::vector<Conn*> stale;
  Conn* pooled = nullptr;
  bool snapshot_current = false;
  {
    std::lock_guard<std::mutex> lk(mu_);
    pools = pools_;
    const auto current_peer = peer_topologies_->Snapshot(node);
    snapshot_current =
        current_peer && current_peer->peer_id == peer_snapshot->peer_id &&
        current_peer->publication == peer_snapshot->publication;
    if (snapshot_current && !options.force_new) {
      auto& idle = lane == Lane::kData
                       ? pool_
                       : (lane == Lane::kSgData ? sg_pool_ : control_pool_);
      auto it = idle.find(node);
      if (it != idle.end()) {
        auto& pool_candidates = it->second;
        for (size_t i = pool_candidates.size(); i > 0; --i) {
          Conn* candidate = pool_candidates[i - 1];
          if (!current_peer ||
              candidate->peer_id != current_peer->peer_id ||
              candidate->peer_publication != current_peer->publication) {
            if (candidate->lifecycle.RequestRetire()) {
              stale.push_back(candidate);
              pool_candidates.erase(pool_candidates.begin() +
                                    static_cast<std::ptrdiff_t>(i - 1));
            }
            continue;
          }
          if (candidate->peer_id == peer_snapshot->peer_id &&
              candidate->peer_publication == peer_snapshot->publication &&
              candidate->rail_index == ridx &&
              candidate->lifecycle.Activate()) {
            pooled = candidate;
            pool_candidates.erase(
                pool_candidates.begin() + static_cast<std::ptrdiff_t>(i - 1));
            break;
          }
        }
      }
    }
  }
  for (Conn* conn : stale)
    Destroy(conn, rdma::RailCompletion::kAdmission);
  stale_publication_reaps_.fetch_add(stale.size(), std::memory_order_relaxed);
  if (!snapshot_current) {
    complete_unowned_lease(rdma::RailCompletion::kAdmission);
    result.failure = AcquireFailure::kAdmission;
    return result;
  }
  if (pooled) {
    pooled->lease = *lease;
    pooled->lease_started_us = lease_started;
    pooled->credit_held = true;
    pooled->visited = true;
    pooled->remote_lease_generation =
        remote_lease ? remote_lease->generation : 0;
    pooled->remote_lease_held = remote_lease.has_value();
    pooled->remote_recovery_probe =
        remote_lease && remote_lease->recovery_probe;
    result.from_pool = true;
    if (!pooled->ep.EnsurePoolMrs(pools, true)) {
      Destroy(pooled, rdma::RailCompletion::kRailFailure);
      result.failure = AcquireFailure::kLocalRail;
      return result;
    }
    endpoint_cache_hits_.fetch_add(1, std::memory_order_relaxed);
    result.conn = pooled;
    result.failure = AcquireFailure::kNone;
    return result;
  }

  endpoint_cache_misses_.fetch_add(1, std::memory_order_relaxed);
  // Open, announce, and budget at the depth this node actually granted last
  // time: churned reconnects stop reserving WR/slot capacity the server's
  // clamp will never let them use (first contact still probes at depth_).
  const size_t conn_depth = LearnedDepth(node, lane);
  const uint64_t conn_declared =
      lane == Lane::kControl
          ? static_cast<uint64_t>(rdma::kV2ControlCap)
          : declared_;
  const uint64_t conn_slot_bytes = static_cast<uint64_t>(rdma::V2SlotSize(
      static_cast<size_t>(
          std::max<uint64_t>(conn_declared, rdma::kV2DataOffset))));
  const rdma::ResourceRequest budget_request{
      1, 1, conn_depth, conn_slot_bytes * conn_depth};
  bool budget_acquired = resource_budget_->TryAcquire(budget_request);
  if (!budget_acquired && EvictOneIdle())
    budget_acquired = resource_budget_->TryAcquire(budget_request);
  if (!budget_acquired)
    budget_acquired = resource_budget_->Acquire(
        budget_request, std::chrono::milliseconds(resource_acquire_ms_));
  if (!budget_acquired) {
    // Budget starvation is a LOCAL condition: the peer was never dialed. The
    // silent variant of this path cost days of diagnosis (0812-002/004), so
    // say what starved and how big the pool is, throttled to one line per 64
    // occurrences (bursts arrive at batch rate).
    const uint64_t seen =
        admission_failures_.fetch_add(1, std::memory_order_relaxed);
    if ((seen & 63u) == 0) {
      const rdma::ResourceRequest used = resource_budget_->used();
      const rdma::ResourceRequest limit = resource_budget_->limit();
      DFKV_LOG_WARN(
          "rdma: transport budget exhausted acquiring an endpoint for " +
          node + " after " + std::to_string(resource_acquire_ms_) +
          "ms (endpoints " + std::to_string(used.endpoints) + "/" +
          std::to_string(limit.endpoints) + ", wr " +
          std::to_string(used.wr_slots) + "/" +
          std::to_string(limit.wr_slots) +
          "); op fails kResourceExhausted, peer is NOT cooled down; raise "
          "DFKV_RDMA_ENDPOINT_CACHE_MAX or let ring autoscale size it");
    }
    complete_unowned_lease(rdma::RailCompletion::kAdmission);
    result.failure = AcquireFailure::kAdmission;
    result.status = Status::kResourceExhausted;
    return result;
  }

  const std::string& dev = devs_[ridx];
  if (!ProbeV2(node)) {
    DFKV_LOG_ERROR(
        "rdma: peer " + node +
        " does not advertise the required v2 writer-retirement capability");
    complete_unowned_lease(rdma::RailCompletion::kEndpointFailure);
    resource_budget_->Release(budget_request);
    result.failure = AcquireFailure::kEndpoint;
    return result;
  }

  int fd = net::Dial(node, connect_ms_, io_ms_);
  if (fd < 0) {
    complete_unowned_lease(rdma::RailCompletion::kEndpointFailure);
    resource_budget_->Release(budget_request);
    result.failure = AcquireFailure::kEndpoint;
    return result;
  }
  auto* conn = new Conn();
  conn->rail_index = ridx;
  conn->peer_publication = peer_snapshot->publication;
  conn->peer_id = peer_snapshot->peer_id;
  conn->remote_lease_generation =
      remote_lease ? remote_lease->generation : 0;
  conn->remote_lease_held = remote_lease.has_value();
  conn->remote_recovery_probe =
      remote_lease && remote_lease->recovery_probe;
  conn->lease = *lease;
  conn->lease_started_us = lease_started;
  conn->credit_held = true;
  conn->budget_request = budget_request;
  conn->budget_held = true;
  if (!conn->ep.Open(dev.c_str(), rdma::kV2ControlCap, conn_depth)) {
    ::close(fd);
    DFKV_LOG_WARN("rdma: device " + dev +
                  " Open failed; rail failure policy will quarantine it");
    Destroy(conn, rdma::RailCompletion::kRailFailure);
    result.failure = AcquireFailure::kLocalRail;
    return result;
  }
  {
    const char* bp = std::getenv("DFKV_RDMA_BUSY_POLL");
    if (bp && *bp && *bp != '0') conn->ep.set_busy_poll(true);
  }

  char devbuf[rdma::kDevNameBytes];
  // The probe above proves the peer understands bit 63 before a real QP is
  // created. Echo the request on this bootstrap so a rolling-upgraded server
  // sends the token only to a client that will consume it.
  const uint64_t bootstrap_declared =
      conn_declared | rdma::kDevFrameRequestWriterRetirement;
  rdma::EncodeDevFrame(auto_device_ ? std::string() : dev,
                       bootstrap_declared, devbuf, rdma::kDevProtoV2);
  char mine[rdma::kQpInfoBytes], remote_qp_bytes[rdma::kQpInfoBytes];
  rdma::QpInfo my = conn->ep.Local();
  my.depth = static_cast<uint16_t>(std::min<size_t>(conn_depth, 256));
  my.protocol_version = rdma::kDevProtoV2;
  rdma::SerializeQpInfo(my, mine);
  if (!net::WriteAll(fd, devbuf, rdma::kDevNameBytes) ||
      !net::WriteAll(fd, mine, rdma::kQpInfoBytes) ||
      !net::ReadAll(fd, remote_qp_bytes, rdma::kQpInfoBytes)) {
    ::close(fd);
    Destroy(conn, rdma::RailCompletion::kEndpointFailure);
    result.failure = AcquireFailure::kEndpoint;
    return result;
  }
  const rdma::QpInfo remote = rdma::ParseQpInfo(remote_qp_bytes);
  if (remote.protocol_version != rdma::kDevProtoV2 || remote.depth == 0) {
    DFKV_LOG_ERROR("rdma: peer " + node +
                   " returned an invalid v2 QP negotiation");
    ::close(fd);
    Destroy(conn, rdma::RailCompletion::kEndpointFailure);
    result.failure = AcquireFailure::kEndpoint;
    return result;
  }
  if (!conn->ep.Connect(remote)) {
    DFKV_LOG_WARN("rdma: device " + dev +
                  " failed local QP transition during v2 negotiation");
    ::close(fd);
    Destroy(conn, rdma::RailCompletion::kRailFailure);
    result.failure = AcquireFailure::kLocalRail;
    return result;
  }
  conn->ep.set_remote_depth(remote.depth);
  NoteNegotiatedDepth(node, lane, remote.depth);
  if (remote.depth < conn_depth) {
    DFKV_LOG_INFO("rdma: server depth " + std::to_string(remote.depth) +
                  " < client depth " + std::to_string(conn_depth) +
                  ": batching window clamped to " +
                  std::to_string(remote.depth) +
                  " (later connections open at the learned depth)");
    // Refund the budget the clamp made unusable. The server leases its recv
    // segment at the NEGOTIATED depth, so WR/registered accounting above the
    // clamp models capacity that exists nowhere. Shrink the connection's own
    // record first, then release the surplus: every later failure path
    // releases via Destroy(conn->budget_request), and this order keeps
    // acquired == refund + destroy-release exact (Subtract is unchecked).
    const size_t surplus = conn_depth - remote.depth;
    conn->budget_request = rdma::ResourceRequest{
        1, 1, static_cast<size_t>(remote.depth),
        conn_slot_bytes * remote.depth};
    resource_budget_->Release(
        rdma::ResourceRequest{0, 0, surplus, conn_slot_bytes * surplus});
    depth_refunds_.fetch_add(1, std::memory_order_relaxed);
  }

  char readiness[rdma::kV2RetirementReadinessBytes];
  if (!net::ReadAll(fd, readiness, sizeof(readiness)) ||
      !rdma::DecodeV2Readiness(
          readiness, sizeof(readiness),
          /*writer_retirement_negotiated=*/true, &conn->recv_segment,
          &conn->writer_token)) {
    DFKV_LOG_ERROR(
        "rdma: peer " + node +
        " did not provide a valid retirement-capable v2 receive segment");
    ::close(fd);
    Destroy(conn, rdma::RailCompletion::kEndpointFailure);
    result.failure = AcquireFailure::kEndpoint;
    return result;
  }
  const size_t expected_slot = rdma::V2SlotSize(static_cast<size_t>(
      std::max<uint64_t>(conn_declared, rdma::kV2DataOffset)));
  if (expected_slot == 0 ||
      conn->recv_segment.slot_size != expected_slot) {
    DFKV_LOG_ERROR("rdma: peer " + node +
                   " returned incompatible v2 receive-segment geometry");
    ::close(fd);
    Destroy(conn, rdma::RailCompletion::kEndpointFailure);
    result.failure = AcquireFailure::kEndpoint;
    return result;
  }
  ::close(fd);
  if (!conn->ep.EnsurePoolMrs(pools, true)) {
    Destroy(conn, rdma::RailCompletion::kRailFailure);
    result.failure = AcquireFailure::kLocalRail;
    return result;
  }
  bool publication_current = false;
  {
    std::lock_guard<std::mutex> lock(mu_);
    publication_current = peer_topologies_->IsCurrent(
        node, conn->peer_id, conn->peer_publication);
  }
  if (!publication_current) {
    stale_publication_reaps_.fetch_add(1, std::memory_order_relaxed);
    Destroy(conn, rdma::RailCompletion::kAdmission);
    result.failure = AcquireFailure::kAdmission;
    return result;
  }
  conns_opened_.fetch_add(1, std::memory_order_relaxed);
  rail_conns_[ridx].fetch_add(1, std::memory_order_relaxed);
  result.conn = conn;
  result.failure = AcquireFailure::kNone;
  return result;
}

bool RdmaTransport::PrepareRetry(
    int attempt, bool from_pool, std::optional<size_t> attempted_rail,
    AcquireFailure failure, ReplaySafety replay_safety, bool request_posted,
    const std::shared_ptr<const rdma::PeerRailSnapshot>& peer,
    RailMask* excluded, bool* cross_rail_retry) {
  if (attempt != 0 || failure == AcquireFailure::kNoCompatibleRail ||
      (replay_safety == ReplaySafety::kUnsafeAfterPost && request_posted))
    return false;
  if (failure == AcquireFailure::kLocalRail && attempted_rail &&
      *attempted_rail < devs_.size()) {
    excluded->assign(devs_.size(), 0);
    (*excluded)[*attempted_rail] = 1;
    const int caller_node = numa_aware_ ? numa::CurrentNode() : -1;
    const auto candidates =
        topology_->CandidatesFor(caller_node, numa_aware_);
    const auto local_enabled =
        rdma::UnionRailMasks(candidates.allowed, candidates.fallback);
    auto compatible =
        rdma::IntersectRailMasks(peer->peer_healthy, local_enabled);
    if (!peer->peer_id.empty()) {
      compatible = remote_rail_health_->AllowedMask(
          peer->peer_id, compatible, rdma::RailPolicy::NowMicros());
    }
    const auto highest_tier =
        rdma::HighestCompatibleTier(rail_tiers_, compatible);
    const auto preferred =
        rdma::IntersectRailMasks(highest_tier, candidates.allowed);
    const auto fallback =
        rdma::IntersectRailMasks(highest_tier, candidates.fallback);
    if (rdma::HasUnexcludedRail(preferred, fallback, *excluded)) {
      cross_rail_retries_.fetch_add(1, std::memory_order_relaxed);
      *cross_rail_retry = true;
    } else {
      // A one-compatible-rail topology may make one fresh same-rail attempt,
      // while ordinary admission still enforces quarantine and credits.
      excluded->assign(devs_.size(), 0);
    }
    return true;
  }
  if (failure == AcquireFailure::kEndpoint && attempted_rail &&
      *attempted_rail < devs_.size()) {
    excluded->assign(devs_.size(), 0);
    (*excluded)[*attempted_rail] = 1;
    const int caller_node = numa_aware_ ? numa::CurrentNode() : -1;
    const auto candidates =
        topology_->CandidatesFor(caller_node, numa_aware_);
    const auto local_enabled =
        rdma::UnionRailMasks(candidates.allowed, candidates.fallback);
    auto compatible =
        rdma::IntersectRailMasks(peer->peer_healthy, local_enabled);
    if (!peer->peer_id.empty()) {
      compatible = remote_rail_health_->AllowedMask(
          peer->peer_id, compatible, rdma::RailPolicy::NowMicros());
    }
    const auto highest_tier =
        rdma::HighestCompatibleTier(rail_tiers_, compatible);
    const auto preferred =
        rdma::IntersectRailMasks(highest_tier, candidates.allowed);
    const auto fallback =
        rdma::IntersectRailMasks(highest_tier, candidates.fallback);
    if (!rdma::HasUnexcludedRail(preferred, fallback, *excluded))
      return false;
    if (from_pool)
      stale_pool_retries_.fetch_add(1, std::memory_order_relaxed);
    cross_rail_retries_.fetch_add(1, std::memory_order_relaxed);
    *cross_rail_retry = true;
    return true;
  }
  return false;
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
  const rdma::ResourceRequest resource_used = resource_budget_->used();
  const rdma::ResourceRequest resource_limit = resource_budget_->limit();
  const auto resource_gauge = [&](const char* name, const char* help,
                                  uint64_t used, uint64_t limit) {
    s += "# HELP " + std::string(name) + " " + help + "\n";
    s += "# TYPE " + std::string(name) + " gauge\n";
    s += std::string(name) + "{kind=\"used\"} " + std::to_string(used) + "\n";
    s += std::string(name) + "{kind=\"limit\"} " + std::to_string(limit) +
         "\n";
  };
  resource_gauge("dfkv_rdma_client_endpoint_budget",
                 "Process-wide RDMA endpoint budget", resource_used.endpoints,
                 resource_limit.endpoints);
  resource_gauge("dfkv_rdma_client_qp_budget",
                 "Process-wide RDMA QP budget", resource_used.qps,
                 resource_limit.qps);
  resource_gauge("dfkv_rdma_client_wr_slot_budget",
                 "Process-wide RDMA work-request slot budget",
                 resource_used.wr_slots, resource_limit.wr_slots);
  resource_gauge("dfkv_rdma_client_registered_slot_bytes_budget",
                 "Process-wide remote receive-slot byte budget",
                 resource_used.registered_bytes,
                 resource_limit.registered_bytes);
  s += "# HELP dfkv_rdma_client_resource_budget_timeouts_total Connection admissions rejected after bounded resource wait\n";
  s += "# TYPE dfkv_rdma_client_resource_budget_timeouts_total counter\n";
  s += "dfkv_rdma_client_resource_budget_timeouts_total " +
       std::to_string(resource_budget_->timeouts()) + "\n";
  s += "# HELP dfkv_rdma_client_resource_budget_raises_total Autoscale limit raises applied to the process budget\n";
  s += "# TYPE dfkv_rdma_client_resource_budget_raises_total counter\n";
  s += "dfkv_rdma_client_resource_budget_raises_total " +
       std::to_string(resource_budget_->raises()) + "\n";
  s += "# HELP dfkv_rdma_client_admission_failures_total Operations failed kResourceExhausted by local budget starvation (peer never dialed)\n";
  s += "# TYPE dfkv_rdma_client_admission_failures_total counter\n";
  s += "dfkv_rdma_client_admission_failures_total " +
       std::to_string(admission_failures_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_depth_refunds_total Connections whose clamped depth negotiation refunded WR/registered budget\n";
  s += "# TYPE dfkv_rdma_client_depth_refunds_total counter\n";
  s += "dfkv_rdma_client_depth_refunds_total " +
       std::to_string(depth_refunds_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_topology_nodes Ring size last hinted to the transport budget autoscaler (0=never hinted or autoscale off)\n";
  s += "# TYPE dfkv_rdma_client_topology_nodes gauge\n";
  s += "dfkv_rdma_client_topology_nodes " +
       std::to_string(topology_nodes_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_endpoint_cache_hits_total Idle endpoint cache hits\n";
  s += "# TYPE dfkv_rdma_endpoint_cache_hits_total counter\n";
  s += "dfkv_rdma_endpoint_cache_hits_total " +
       std::to_string(endpoint_cache_hits_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_endpoint_cache_misses_total Endpoint acquisitions requiring a new QP\n";
  s += "# TYPE dfkv_rdma_endpoint_cache_misses_total counter\n";
  s += "dfkv_rdma_endpoint_cache_misses_total " +
       std::to_string(endpoint_cache_misses_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_endpoint_cache_evictions_total Idle endpoints evicted under process-wide resource pressure\n";
  s += "# TYPE dfkv_rdma_endpoint_cache_evictions_total counter\n";
  s += "dfkv_rdma_endpoint_cache_evictions_total " +
       std::to_string(
           endpoint_cache_evictions_.load(std::memory_order_relaxed)) +
       "\n";
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
  s += "# HELP dfkv_rdma_client_pipeline_depth Configured RDMA pipeline depth ceiling (env DFKV_RDMA_DEPTH); per-node connections may open at a lower learned negotiated depth\n";
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
  s += "# HELP dfkv_rdma_client_cross_rail_retries_total Logical operations retried with the failed local or remote endpoint rail excluded and another compatible healthy rail available\n";
  s += "# TYPE dfkv_rdma_client_cross_rail_retries_total counter\n";
  s += "dfkv_rdma_client_cross_rail_retries_total " +
       std::to_string(cross_rail_retries_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_cross_rail_retry_successes_total Cross-rail retries completed successfully\n";
  s += "# TYPE dfkv_rdma_client_cross_rail_retry_successes_total counter\n";
  s += "dfkv_rdma_client_cross_rail_retry_successes_total " +
       std::to_string(
           cross_rail_retry_successes_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_cross_rail_retry_exhausted_total Cross-rail retries whose second transport attempt failed\n";
  s += "# TYPE dfkv_rdma_client_cross_rail_retry_exhausted_total counter\n";
  s += "dfkv_rdma_client_cross_rail_retry_exhausted_total " +
       std::to_string(
           cross_rail_retry_exhausted_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_completion_timeouts_total Completion windows exhausting their absolute deadline\n";
  s += "# TYPE dfkv_rdma_client_completion_timeouts_total counter\n";
  s += "dfkv_rdma_client_completion_timeouts_total " +
       std::to_string(completion_timeouts_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_ambiguous_get_quarantines_total Failed staged GETs retained because responder retirement proof was unavailable\n";
  s += "# TYPE dfkv_rdma_client_ambiguous_get_quarantines_total counter\n";
  s += "dfkv_rdma_client_ambiguous_get_quarantines_total " +
       std::to_string(
           ambiguous_get_quarantines_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_ambiguous_get_quarantined_bytes Bytes held with ambiguous staged GET endpoints until process teardown\n";
  s += "# TYPE dfkv_rdma_client_ambiguous_get_quarantined_bytes gauge\n";
  s += "dfkv_rdma_client_ambiguous_get_quarantined_bytes " +
       std::to_string(
           ambiguous_get_quarantined_bytes_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_keepalive_attempts_total Idle pooled QP keepalive probes attempted\n";
  s += "# TYPE dfkv_rdma_client_keepalive_attempts_total counter\n";
  s += "dfkv_rdma_client_keepalive_attempts_total " +
       std::to_string(keepalive_attempts_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_keepalive_successes_total Idle pooled QP keepalive probes completed\n";
  s += "# TYPE dfkv_rdma_client_keepalive_successes_total counter\n";
  s += "dfkv_rdma_client_keepalive_successes_total " +
       std::to_string(keepalive_successes_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_keepalive_failures_total Idle pooled QP keepalive probes that retired a connection\n";
  s += "# TYPE dfkv_rdma_client_keepalive_failures_total counter\n";
  s += "dfkv_rdma_client_keepalive_failures_total " +
       std::to_string(keepalive_failures_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_peer_topology_updates_total Published peer topology snapshots\n";
  s += "# TYPE dfkv_rdma_client_peer_topology_updates_total counter\n";
  s += "dfkv_rdma_client_peer_topology_updates_total " +
       std::to_string(
           peer_topology_updates_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_no_compatible_rail_total Operations rejected before admission because no peer-compatible rail exists\n";
  s += "# TYPE dfkv_rdma_client_no_compatible_rail_total counter\n";
  s += "dfkv_rdma_client_no_compatible_rail_total " +
       std::to_string(no_compatible_rail_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_stale_generation_reaps_total Connections retired instead of reuse after a peer publication/incarnation change\n";
  s += "# TYPE dfkv_rdma_client_stale_generation_reaps_total counter\n";
  s += "dfkv_rdma_client_stale_generation_reaps_total " +
       std::to_string(
           stale_publication_reaps_.load(std::memory_order_relaxed)) + "\n";
  s += remote_rail_health_->MetricsText(devs_);
  return s;
}

void RdmaTransport::Release(const std::string& node, Lane lane, Conn* c,
                            RemoteRailOutcome remote_outcome) {
  CompleteRemoteLease(c, remote_outcome);
  bool refresh_ok = false;
  {
    std::lock_guard<std::mutex> lk(mu_);
    // Refresh while still active and while pools_ is stable. A refresh failure
    // is local MR evidence and must tear down before returning the lease.
    refresh_ok = c->ep.EnsurePoolMrs(pools_, true);
  }
  if (!refresh_ok) {
    Destroy(c, rdma::RailCompletion::kRailFailure);
    return;
  }

  if (c->credit_held) {
    const uint64_t now = rdma::RailPolicy::NowMicros();
    rail_policy_->Complete(c->lease, now - c->lease_started_us,
                           rdma::RailCompletion::kSuccess, now);
    c->credit_held = false;
  }

  bool reusable = false;
  bool publication_current = false;
  {
    std::lock_guard<std::mutex> lk(mu_);
    publication_current = peer_topologies_->IsCurrent(
        node, c->peer_id, c->peer_publication);
    if (publication_current) {
      auto& idle = lane == Lane::kData
                       ? pool_
                       : (lane == Lane::kSgData ? sg_pool_ : control_pool_);
      auto& v = idle[node];
      if (v.size() < pool_max_ && c->lifecycle.MakeIdle()) {
        v.push_back(c);
        reusable = true;
      }
    }
  }
  if (!reusable && !publication_current)
    stale_publication_reaps_.fetch_add(1, std::memory_order_relaxed);
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
  const bool local_rail_replay_safe =
      op != WireOp::kCache && op != WireOp::kRemove;
  const ReplaySafety replay_safety =
      op == WireOp::kCache ? ReplaySafety::kUnsafeAfterPost
                           : ReplaySafety::kReplaySafe;
  const auto peer = peer_topologies_->Snapshot(node);
  RailMask excluded(devs_.size(), 0);
  bool cross_rail_retry = false;
  CompletionFaultOperation completion_fault(
      &test_completion_fault_calls_,
      op == WireOp::kCache || op == WireOp::kRange);
  for (int attempt = 0; attempt < 2; ++attempt) {
    completion_fault.BeginAttempt(attempt);
    std::string attempt_out;
    uint64_t attempt_value_len = 0;
    AcquireOptions options;
    options.force_new = attempt != 0;
    options.excluded = excluded;
    options.peer = peer;
    AcquireResult acquired = Acquire(node, lane, options);
    if (!acquired.conn) {
      if (PrepareRetry(attempt, acquired.from_pool, acquired.attempted_rail,
                       acquired.failure, replay_safety, false, peer, &excluded,
                       &cross_rail_retry)) {
        continue;
      }
      if (cross_rail_retry)
        cross_rail_retry_exhausted_.fetch_add(1,
                                              std::memory_order_relaxed);
      return acquired.status;
    }
    Conn* conn = acquired.conn;
    if (InjectLocalRailFailure(attempt)) {
      const size_t failed_rail = conn->rail_index;
      Destroy(conn, rdma::RailCompletion::kRailFailure);
      if (PrepareRetry(attempt, acquired.from_pool, failed_rail,
                       AcquireFailure::kLocalRail, replay_safety, false, peer,
                       &excluded, &cross_rail_retry))
        continue;
      if (cross_rail_retry)
        cross_rail_retry_exhausted_.fetch_add(1,
                                              std::memory_order_relaxed);
      return Status::kIOError;
    }
    rdma::RcEndpoint& ep = conn->ep;

    bool ok = false;
    ibv_mr* transient_output_mr = nullptr;
    rdma::RailCompletion completion = rdma::RailCompletion::kRailFailure;
    bool request_posted = false;
    if (op == WireOp::kCache) {
      if (payload_len > conn->data_capacity()) {
        Release(node, lane, conn, RemoteRailOutcome::kAbandon);
        return Status::kInvalid;
      }
      ibv_mr* payload_mr =
          payload_len
              ? ep.RegisterTransient(const_cast<void*>(payload),
                                     static_cast<size_t>(payload_len),
                                     /*remote_write=*/false)
              : nullptr;
      transient_output_mr = payload_mr;
      if (payload_len == 0 || payload_mr) {
        conn->Encode(ep.sbuf(0), op, key, offset, length, payload_len);
        ok = ep.PostRecv(0) &&
             ep.PostWriteImmScatter(
                 0, kReqPrefix, payload, static_cast<size_t>(payload_len),
                 payload_mr, conn->put_addr(0), conn->recv_segment.rkey, 0);
        if (ok)
          v2_put_writes_.fetch_add(1, std::memory_order_relaxed);
      }
    } else if (op == WireOp::kRange) {
      if (!out || length > conn->data_capacity() ||
          length > std::numeric_limits<uint32_t>::max()) {
        Release(node, lane, conn, RemoteRailOutcome::kAbandon);
        return Status::kInvalid;
      }
      attempt_out.resize(static_cast<size_t>(length));
      std::vector<RdmaWriteTarget> targets;
      if (length != 0) {
        transient_output_mr =
            ep.RegisterTransient(attempt_out.data(), attempt_out.size());
        if (transient_output_mr) {
          targets.push_back(
              {reinterpret_cast<uint64_t>(attempt_out.data()),
               transient_output_mr->rkey, static_cast<uint32_t>(length)});
        }
      }
      if (length == 0 || transient_output_mr) {
        size_t frame_len = 0;
        ok = EncodeRdmaGetReq(ep.sbuf(0), ep.cap(), key, offset, length,
                              targets, &frame_len) &&
             ep.PostRecv(0) && ep.PostSend(0, frame_len);
        if (ok)
          v2_get_writes_.fetch_add(1, std::memory_order_relaxed);
      }
    } else {
      conn->Encode(ep.sbuf(0), op, key, offset, length, payload_len);
      ok = ep.PostRecv(0) &&
           ep.PostSend(0, kReqPrefix + static_cast<size_t>(payload_len));
    }
    request_posted = ok;

    std::vector<uint32_t> reply_bytes;
    bool timed_out = false;
    if (ok) {
      ibv_wc_status wc_status = IBV_WC_SUCCESS;
      bool had_wcs = false;
      ok = ReapWindow(ep, 1, &reply_bytes, op_timeout_ms_, &timed_out,
                      &wc_status, &had_wcs, &completion_fault);
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

    Status status = Status::kIOError;
    uint64_t data_len = 0;
    if (ok && recv_bytes < kRespPrefix) {
      ok = false;
      completion = rdma::RailCompletion::kEndpointFailure;
    }
    const uint64_t response_bound =
        op == WireOp::kMembers ? rdma::kV2ControlResponseMax
                               : (op == WireOp::kRange ? length : 0);
    if (ok && !conn->Decode(
                  ep.rbuf(0), &status, &data_len, response_bound,
                  value_len ? &attempt_value_len : nullptr)) {
      ok = false;
      completion = rdma::RailCompletion::kEndpointFailure;
    }
    if (ok && out) {
      if (op == WireOp::kRange) {
        if (status == Status::kOk && data_len <= attempt_out.size()) {
          attempt_out.resize(static_cast<size_t>(data_len));
        } else if (status != Status::kOk) {
          attempt_out.clear();
        } else {
          ok = false;
          completion = rdma::RailCompletion::kEndpointFailure;
        }
      } else if (data_len > rdma::kV2ControlResponseMax ||
                 data_len > recv_bytes - kRespPrefix) {
        ok = false;
        completion = rdma::RailCompletion::kEndpointFailure;
      } else {
        attempt_out.assign(ep.rbuf(0) + kRespPrefix,
                           static_cast<size_t>(data_len));
      }
    }
    if (ok) {
      if (out) *out = std::move(attempt_out);
      if (value_len) *value_len = attempt_value_len;
      Release(node, lane, conn);
      if (cross_rail_retry)
        cross_rail_retry_successes_.fetch_add(1,
                                              std::memory_order_relaxed);
      return status;
    }

    const size_t failed_rail = conn->rail_index;
    bool quarantined = false;
    if (op == WireOp::kRange &&
        !RetireRemoteWriter(node, conn->writer_token, connect_ms_, io_ms_)) {
      auto* hold = new std::string();
      hold->swap(attempt_out);
      QuarantineAmbiguousGet(conn, hold, hold->capacity(), "range",
                             completion);
      quarantined = true;
    }
    if (!quarantined) Destroy(conn, completion);
    const AcquireFailure failure =
        completion == rdma::RailCompletion::kRailFailure &&
                (local_rail_replay_safe || !request_posted)
            ? AcquireFailure::kLocalRail
            : AcquireFailure::kEndpoint;
    if (PrepareRetry(attempt, acquired.from_pool, failed_rail, failure,
                     replay_safety, request_posted, peer, &excluded,
                     &cross_rail_retry)) {
      continue;
    }
    if (cross_rail_retry)
      cross_rail_retry_exhausted_.fetch_add(1, std::memory_order_relaxed);
    return Status::kIOError;
  }
  if (cross_rail_retry)
    cross_rail_retry_exhausted_.fetch_add(1, std::memory_order_relaxed);
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

  const auto peer = peer_topologies_->Snapshot(node);
  RailMask excluded(devs_.size(), 0);
  bool cross_rail_retry = false;
  CompletionFaultOperation completion_fault(&test_completion_fault_calls_);
  for (int attempt = 0; attempt < 2; ++attempt) {
    completion_fault.BeginAttempt(attempt);
    std::fill(result.begin(), result.end(), Status::kIOError);
    for (size_t i = 0; i < count; ++i)
      if (bad[i]) result[i] = Status::kInvalid;
    AcquireOptions options;
    options.force_new = attempt != 0;
    options.requested_credits = std::min(valid_count, depth_);
    options.excluded = excluded;
    options.peer = peer;
    AcquireResult acquired = Acquire(node, Lane::kData, options);
    if (!acquired.conn) {
      if (PrepareRetry(attempt, acquired.from_pool, acquired.attempted_rail,
                       acquired.failure, ReplaySafety::kUnsafeAfterPost, false,
                       peer, &excluded, &cross_rail_retry))
        continue;
      if (cross_rail_retry)
        cross_rail_retry_exhausted_.fetch_add(1,
                                              std::memory_order_relaxed);
      if (acquired.failure == AcquireFailure::kAdmission ||
          acquired.failure == AcquireFailure::kNoCompatibleRail)
        MarkClientLocalFailure(&result, acquired.status);
      return result;
    }
    Conn* conn = acquired.conn;
    rdma::RcEndpoint& ep = conn->ep;
    const size_t window = std::min(
        ep.window(), static_cast<size_t>(conn->lease.credits));
    bool conn_ok = !InjectLocalRailFailure(attempt);
    bool request_posted = false;
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
        request_posted = true;
      }
      std::vector<uint32_t> reply_bytes;
      bool timed_out = false;
      ibv_wc_status wc_status = IBV_WC_SUCCESS;
      bool had_wcs = false;
      if (conn_ok &&
          !ReapPosted(ep, posted, width, &reply_bytes, BatchTimeout(),
                      &timed_out, &wc_status, &had_wcs, &completion_fault)) {
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
      if (cross_rail_retry)
        cross_rail_retry_successes_.fetch_add(1,
                                              std::memory_order_relaxed);
      return result;
    }
    const size_t failed_rail = conn->rail_index;
    Destroy(conn, completion);
    const AcquireFailure failure =
        completion == rdma::RailCompletion::kRailFailure && !request_posted
            ? AcquireFailure::kLocalRail
            : AcquireFailure::kEndpoint;
    if (PrepareRetry(attempt, acquired.from_pool, failed_rail, failure,
                     ReplaySafety::kUnsafeAfterPost, request_posted, peer,
                     &excluded, &cross_rail_retry))
      continue;
    if (cross_rail_retry)
      cross_rail_retry_exhausted_.fetch_add(1, std::memory_order_relaxed);
    return result;
  }
  if (cross_rail_retry)
    cross_rail_retry_exhausted_.fetch_add(1, std::memory_order_relaxed);
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
  if (outputs == nullptr) return InvalidStatuses(count);
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

  const auto peer = peer_topologies_->Snapshot(node);
  RailMask excluded(devs_.size(), 0);
  bool cross_rail_retry = false;
  CompletionFaultOperation completion_fault(&test_completion_fault_calls_);
  for (int attempt = 0; attempt < 2; ++attempt) {
    completion_fault.BeginAttempt(attempt);
    std::fill(result.begin(), result.end(), Status::kIOError);
    std::vector<std::string> attempt_outputs(count);
    std::vector<uint64_t> attempt_value_lens(count, 0);
    AcquireOptions options;
    options.force_new = attempt != 0;
    options.requested_credits = std::min(count, depth_);
    options.excluded = excluded;
    options.peer = peer;
    AcquireResult acquired = Acquire(node, Lane::kData, options);
    if (!acquired.conn) {
      if (PrepareRetry(attempt, acquired.from_pool, acquired.attempted_rail,
                       acquired.failure, ReplaySafety::kReplaySafe, false,
                       peer, &excluded, &cross_rail_retry))
        continue;
      if (cross_rail_retry)
        cross_rail_retry_exhausted_.fetch_add(1,
                                              std::memory_order_relaxed);
      if (acquired.failure == AcquireFailure::kAdmission ||
          acquired.failure == AcquireFailure::kNoCompatibleRail)
        MarkClientLocalFailure(&result, acquired.status);
      return result;
    }
    Conn* conn = acquired.conn;
    rdma::RcEndpoint& ep = conn->ep;
    const size_t window = std::min(
        ep.window(), static_cast<size_t>(conn->lease.credits));
    bool conn_ok = !InjectLocalRailFailure(attempt);
    // Failure attribution for Destroy: post/encode/register failures keep the
    // local-rail default; a failed reap window is classified from its WC
    // evidence; wire decode failures blame the peer.
    rdma::RailCompletion completion = rdma::RailCompletion::kRailFailure;
    for (size_t base = 0; base < count && conn_ok; base += window) {
      const size_t width = std::min(window, count - base);
      std::vector<ibv_mr*> output_mrs(width, nullptr);
      for (size_t slot = 0; slot < width; ++slot) {
        std::string& output = attempt_outputs[base + slot];
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
                      &wc_status, &had_wcs, &completion_fault)) {
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
        attempt_value_lens[base + slot] = value_len;
        std::string& output = attempt_outputs[base + slot];
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
      *outputs = std::move(attempt_outputs);
      if (value_lens) *value_lens = std::move(attempt_value_lens);
      Release(node, Lane::kData, conn);
      if (cross_rail_retry)
        cross_rail_retry_successes_.fetch_add(1,
                                              std::memory_order_relaxed);
      return result;
    }
    const size_t failed_rail = conn->rail_index;
    bool quarantined = false;
    if (!RetireRemoteWriter(node, conn->writer_token, connect_ms_, io_ms_)) {
      auto* hold = new std::vector<std::string>();
      hold->swap(attempt_outputs);
      size_t held_bytes = 0;
      for (const auto& value : *hold) {
        if (value.capacity() > std::numeric_limits<size_t>::max() - held_bytes) {
          held_bytes = std::numeric_limits<size_t>::max();
          break;
        }
        held_bytes += value.capacity();
      }
      QuarantineAmbiguousGet(conn, hold, held_bytes, "range_many",
                             completion);
      quarantined = true;
    }
    if (!quarantined) Destroy(conn, completion);
    const AcquireFailure failure =
        completion == rdma::RailCompletion::kRailFailure
            ? AcquireFailure::kLocalRail
            : AcquireFailure::kEndpoint;
    if (PrepareRetry(attempt, acquired.from_pool, failed_rail, failure,
                     ReplaySafety::kReplaySafe, false, peer, &excluded,
                     &cross_rail_retry))
      continue;
    if (cross_rail_retry)
      cross_rail_retry_exhausted_.fetch_add(1, std::memory_order_relaxed);
    return result;
  }
  if (cross_rail_retry)
    cross_rail_retry_exhausted_.fetch_add(1, std::memory_order_relaxed);
  return result;
}

std::vector<Status> RdmaTransport::ExistMany(
    const std::string& node, const std::vector<BlockKey>& keys,
    std::vector<char>* exists, std::string* out_dev) {
  const size_t count = keys.size();
  if (exists == nullptr) return InvalidStatuses(count);
  exists->assign(count, 0);
  std::vector<Status> result(count, Status::kIOError);
  if (count == 0) return result;

  const auto peer = peer_topologies_->Snapshot(node);
  RailMask excluded(devs_.size(), 0);
  bool cross_rail_retry = false;
  bool out_dev_set = false;
  for (int attempt = 0; attempt < 2; ++attempt) {
    std::fill(result.begin(), result.end(), Status::kIOError);
    std::fill(exists->begin(), exists->end(), 0);
    AcquireOptions options;
    options.force_new = attempt != 0;
    options.requested_credits = std::min(count, depth_);
    options.excluded = excluded;
    options.peer = peer;
    AcquireResult acquired = Acquire(node, Lane::kControl, options);
    if (!acquired.conn) {
      if (PrepareRetry(attempt, acquired.from_pool, acquired.attempted_rail,
                       acquired.failure, ReplaySafety::kReplaySafe, false,
                       peer, &excluded, &cross_rail_retry))
        continue;
      if (cross_rail_retry)
        cross_rail_retry_exhausted_.fetch_add(1,
                                              std::memory_order_relaxed);
      if (acquired.failure == AcquireFailure::kAdmission ||
          acquired.failure == AcquireFailure::kNoCompatibleRail)
        MarkClientLocalFailure(&result, acquired.status);
      return result;
    }
    Conn* conn = acquired.conn;
    if (out_dev) {
      const std::string& attempted_dev = devs_[conn->rail_index];
      if (!out_dev_set) {
        *out_dev = attempted_dev;
        out_dev_set = true;
      } else if (*out_dev != attempted_dev) {
        *out_dev += "->";
        *out_dev += attempted_dev;
      }
    }
    rdma::RcEndpoint& ep = conn->ep;
    const size_t window = std::min(
        ep.window(), static_cast<size_t>(conn->lease.credits));
    bool conn_ok = !InjectLocalRailFailure(attempt);
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
      if (cross_rail_retry)
        cross_rail_retry_successes_.fetch_add(1,
                                              std::memory_order_relaxed);
      return result;
    }
    const size_t failed_rail = conn->rail_index;
    Destroy(conn, completion);
    const AcquireFailure failure =
        completion == rdma::RailCompletion::kRailFailure
            ? AcquireFailure::kLocalRail
            : AcquireFailure::kEndpoint;
    if (PrepareRetry(attempt, acquired.from_pool, failed_rail, failure,
                     ReplaySafety::kReplaySafe, false, peer, &excluded,
                     &cross_rail_retry))
      continue;
    if (cross_rail_retry)
      cross_rail_retry_exhausted_.fetch_add(1, std::memory_order_relaxed);
    return result;
  }
  if (cross_rail_retry)
    cross_rail_retry_exhausted_.fetch_add(1, std::memory_order_relaxed);
  return result;
}

std::vector<Status> RdmaTransport::RangeInto(
    const std::string& node, const std::vector<BlockKey>& keys,
    const std::vector<RangeDst>& destinations,
    std::vector<uint64_t>* value_lens) {
  const size_t count = keys.size();
  if (value_lens) value_lens->assign(count, 0);
  if (destinations.size() != count) return InvalidStatuses(count);
  std::vector<Status> result(count, Status::kInvalid);
  if (count == 0) return result;

  std::vector<BlockKey> valid_keys;
  std::vector<size_t> indices;
  valid_keys.reserve(count);
  indices.reserve(count);
  size_t max_capacity = 0;
  for (size_t i = 0; i < count; ++i) {
    if (!ValidBuffer(destinations[i].payload, destinations[i].n)) continue;
    valid_keys.push_back(keys[i]);
    indices.push_back(i);
    max_capacity = std::max(max_capacity, destinations[i].n);
  }
  if (valid_keys.empty()) return result;

  // Responder DMA is confined to these operation-owned strings. Only a
  // successful DONE fence reaches DestinationPublisher and caller memory.
  std::vector<std::string> staging;
  std::vector<uint64_t> stored;
  const auto staged =
      RangeMany(node, valid_keys, 0, max_capacity, &staging, &stored);
  DestinationPublisher publisher;
  for (size_t valid_i = 0;
       valid_i < indices.size() && valid_i < staged.size(); ++valid_i) {
    const size_t i = indices[valid_i];
    result[i] = staged[valid_i];
    if (valid_i < stored.size() && value_lens)
      (*value_lens)[i] = stored[valid_i];
    if (result[i] != Status::kOk || valid_i >= staging.size()) continue;
    const size_t bytes =
        std::min(staging[valid_i].size(), destinations[i].n);
    if (bytes &&
        !publisher.Copy(destinations[i].payload, staging[valid_i].data(),
                        bytes, destinations[i].memory_kind))
      result[i] = Status::kIOError;
  }
  if (!publisher.Finish()) {
    for (size_t i : indices)
      if (result[i] == Status::kOk &&
          destinations[i].memory_kind == DestinationMemoryKind::kDevice)
        result[i] = Status::kIOError;
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
    const std::string& node, const std::vector<CacheSrcMulti>& sources, std::string* out_dev) {
  const size_t count = sources.size();
  std::vector<Status> result(count, Status::kIOError);
  if (count == 0) return result;
  std::vector<char> bad(count, 0);
  std::vector<char> completed(count, 0);
  std::vector<size_t> totals(count, 0);
  size_t valid_count = 0;
  for (size_t i = 0; i < count; ++i) {
    bool invalid = false;
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

  const auto peer = peer_topologies_->Snapshot(node);
  RailMask excluded(devs_.size(), 0);
  bool cross_rail_retry = false;
  bool out_dev_set = false;
  CompletionFaultOperation completion_fault(&test_completion_fault_calls_);
  for (int attempt = 0; attempt < 2; ++attempt) {
    // A fresh attempt is possible only before any request is posted;
    // validation results survive while connection-local progress restarts.
    completion_fault.BeginAttempt(attempt);
    AcquireOptions options;
    options.force_new = attempt != 0;
    options.requested_credits = 1;
    options.excluded = excluded;
    options.peer = peer;
    AcquireResult acquired = Acquire(node, Lane::kSgData, options);
    if (!acquired.conn) {
      if (PrepareRetry(attempt, acquired.from_pool, acquired.attempted_rail,
                       acquired.failure, ReplaySafety::kUnsafeAfterPost, false,
                       peer, &excluded, &cross_rail_retry))
        continue;
      if (cross_rail_retry)
        cross_rail_retry_exhausted_.fetch_add(1,
                                              std::memory_order_relaxed);
      if (acquired.failure == AcquireFailure::kAdmission ||
          acquired.failure == AcquireFailure::kNoCompatibleRail)
        MarkClientLocalFailure(&result, acquired.status);
      return result;
    }
    Conn* conn = acquired.conn;
    if (out_dev) {
      const std::string& attempted_dev = devs_[conn->rail_index];
      if (!out_dev_set) {
        *out_dev = attempted_dev;
        out_dev_set = true;
      } else if (*out_dev != attempted_dev) {
        *out_dev += "->";
        *out_dev += attempted_dev;
      }
    }
    rdma::RcEndpoint& ep = conn->ep;
    const size_t seg_limit = std::max<size_t>(1, ep.max_sge() - 1);
    bool conn_ok = !InjectLocalRailFailure(attempt);
    bool request_posted = false;
    rdma::RailCompletion completion = rdma::RailCompletion::kRailFailure;
    const size_t remote_capacity = conn->data_capacity();

    for (size_t item_index = 0; item_index < count && conn_ok; ++item_index) {
      if (bad[item_index] || completed[item_index]) continue;
      if (totals[item_index] > remote_capacity) {
        result[item_index] = Status::kInvalid;
        completed[item_index] = 1;
        continue;
      }
      const CacheSrcMulti& source = sources[item_index];
      size_t nonempty = 0;
      for (const auto& payload : source.payloads)
        if (payload.second != 0) ++nonempty;
      const size_t window_count = 1 + (nonempty - 1) / seg_limit;
      if (window_count > std::numeric_limits<uint32_t>::max()) {
        result[item_index] = Status::kInvalid;
        completed[item_index] = 1;
        continue;
      }

      conn->Encode(ep.sbuf(0), WireOp::kCache, source.key,
                   window_count > 1 ? rdma::kV2MultiWrPutMagic : 0,
                   window_count > 1 ? window_count : 0,
                   totals[item_index]);
      size_t segment_index = 0;
      size_t logical_offset = 0;
      for (size_t window_index = 0;
           window_index < window_count && conn_ok; ++window_index) {
        std::vector<std::pair<const void*, uint32_t>> segments;
        std::vector<ibv_mr*> mrs;
        segments.reserve(seg_limit);
        mrs.reserve(seg_limit);
        size_t window_bytes = 0;
        while (segment_index < source.payloads.size() &&
               segments.size() < seg_limit) {
          const auto& payload = source.payloads[segment_index++];
          if (payload.second == 0) continue;
          ibv_mr* mr = ep.RegisterTransient(
              const_cast<void*>(payload.first), payload.second,
              /*remote_write=*/false);
          if (!mr) {
            conn_ok = false;
            break;
          }
          segments.emplace_back(payload.first,
                                static_cast<uint32_t>(payload.second));
          mrs.push_back(mr);
          window_bytes += payload.second;
        }
        if (!conn_ok) break;
        if (AbortMultiWrWindow(window_index + 1, window_count)) {
          conn_ok = false;
          break;
        }
        const size_t header_len = window_index == 0 ? kReqPrefix : 0;
        const uint64_t remote_addr =
            conn->put_addr(0) +
            (window_index == 0 ? 0 : kReqPrefix + logical_offset);
        if (!ep.PostRecv(0) ||
            !ep.PostWriteImmScatterMulti(
                0, header_len, segments, mrs, remote_addr,
                conn->recv_segment.rkey, 0)) {
          conn_ok = false;
          break;
        }
        v2_put_writes_.fetch_add(1, std::memory_order_relaxed);
        request_posted = true;
        std::vector<uint32_t> reply_bytes;
        bool timed_out = false;
        ibv_wc_status wc_status = IBV_WC_SUCCESS;
        bool had_wcs = false;
        if (!ReapPosted(ep, 1, 1, &reply_bytes, BatchTimeout(), &timed_out,
                        &wc_status, &had_wcs, &completion_fault)) {
          conn_ok = false;
          completion = rdma::ClassifyCompletion(wc_status, had_wcs);
        }
        if (timed_out)
          completion_timeouts_.fetch_add(1, std::memory_order_relaxed);
        if (!conn_ok) break;
        for (ibv_mr* mr : mrs) ep.ReleaseTransient(mr);
        Status status;
        uint64_t data_len = 0;
        if (reply_bytes[0] < kRespPrefix ||
            !conn->Decode(ep.rbuf(0), &status, &data_len, 0)) {
          conn_ok = false;
          completion = rdma::RailCompletion::kEndpointFailure;
          break;
        }
        result[item_index] = status;
        if (status != Status::kOk) {
          if (window_index + 1 != window_count) {
            conn_ok = false;
            completion = rdma::RailCompletion::kEndpointFailure;
          }
          break;
        }
        logical_offset += window_bytes;
      }
      if (conn_ok && result[item_index] == Status::kOk &&
          logical_offset != totals[item_index]) {
        conn_ok = false;
        completion = rdma::RailCompletion::kEndpointFailure;
      }
      if (conn_ok) completed[item_index] = 1;
    }
    if (conn_ok) {
      Release(node, Lane::kSgData, conn);
      if (cross_rail_retry)
        cross_rail_retry_successes_.fetch_add(1,
                                              std::memory_order_relaxed);
      return result;
    }
    const size_t failed_rail = conn->rail_index;
    Destroy(conn, completion);
    for (size_t i = 0; i < count; ++i) {
      if (!bad[i] && !completed[i]) result[i] = Status::kIOError;
    }
    const AcquireFailure failure =
        completion == rdma::RailCompletion::kRailFailure && !request_posted
            ? AcquireFailure::kLocalRail
            : AcquireFailure::kEndpoint;
    if (PrepareRetry(attempt, acquired.from_pool, failed_rail, failure,
                     ReplaySafety::kUnsafeAfterPost, request_posted, peer,
                     &excluded, &cross_rail_retry))
      continue;
    if (cross_rail_retry)
      cross_rail_retry_exhausted_.fetch_add(1, std::memory_order_relaxed);
    return result;
  }
  if (cross_rail_retry)
    cross_rail_retry_exhausted_.fetch_add(1, std::memory_order_relaxed);
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
    std::vector<size_t>* out_lengths, std::string* out_dev) {
  // Flatten into operation-owned contiguous staging, then publish/scatter only
  // after the staged GET is terminal. This removes remote access to caller
  // segments from the failure path.
  return Transport::RangeIntoMulti(node, keys, destinations, out_lengths,
                                   out_dev);
}

}  // namespace dfkv
