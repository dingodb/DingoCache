#include "common/config_dump.h"
#include "transport/rdma_transport.h"

#include <sys/resource.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>

#include "utils/log.h"
#include "utils/env_parse.h"
#include "utils/net_util.h"     // Dial / WriteAll / ReadAll / Put*/Get*
#include "utils/numa_util.h"
#include "utils/prom_escape.h"
#include "transport/rdma_topology.h"
#include "transport/rail_classify.h"
#include "transport/rail_select.h"
#include "transport/rdma_verbs.h"   // RcEndpoint, QpInfo
#include "transport/rdma_protocol.h"
#include "transport/rdma_operation.h"
#include "transport/rdma_connection_lifecycle.h"
#include "transport/rdma_settings.h"

namespace dfkv {

namespace {
// The byte settings' bound: dbuf/SGE length is uint32, with room for the
// direct-I/O alignment rounding.
constexpr uint64_t kMaxSgeBytes =
    std::numeric_limits<uint32_t>::max() - 2 * rdma::kDirectIoAlign;

// The DFKV_RDMA_TEST_* fault hooks below read the environment only once a
// test binary has turned them on (RdmaTransport::EnableTestHooks). A process
// that never does, every production one, makes no getenv call on the data
// path, which would race a host that calls setenv (Python, for one).
std::atomic<bool> g_test_hooks{false};
bool TestHooks() { return g_test_hooks.load(std::memory_order_relaxed); }

bool AbortMultiWrWindow(size_t one_based_window, size_t window_count) {
  if (window_count <= 1 || !TestHooks()) return false;
  const char* value = std::getenv("DFKV_RDMA_TEST_ABORT_MULTIWR_WINDOW");
  if (!value || !*value) return false;
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  return errno == 0 && end != value && *end == '\0' &&
         parsed == one_based_window;
}


bool InjectLocalRailFailure(int zero_based_attempt) {
  if (!TestHooks()) return false;
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

// "1": every READ fails. "@N": only the transport's N-th READ (one-based).
bool InjectPullReadFailure(std::atomic<uint64_t>* reads) {
  if (!TestHooks()) return false;
  const char* value = std::getenv("DFKV_RDMA_TEST_PULL_READ_FAILURE");
  if (!value) return false;
  if (std::strcmp(value, "1") == 0) return true;
  if (value[0] != '@') return false;
  const uint64_t nth = std::strtoull(value + 1, nullptr, 10);
  return nth != 0 && reads->fetch_add(1, std::memory_order_relaxed) + 1 == nth;
}

size_t ResolveMaxPayload(size_t configured) {
  size_t n = configured ? configured : (64u << 20);
  n = rdma_settings::ReadBytes("DFKV_RDMA_MAX_PAYLOAD_BYTES", n, kMaxSgeBytes);
  // A --max-msg above the bound is clamped too.
  return n > kMaxSgeBytes ? static_cast<size_t>(kMaxSgeBytes) : n;
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
    if (!enabled || !TestHooks()) return;
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
// After a failed send or receive on a socket with a timeout: true when the
// call ran out of time, false when the peer closed or reset the connection.
// The caller clears errno first, since a clean close sets none.
bool SocketTimedOut(int err) { return err == EAGAIN || err == EWOULDBLOCK; }

void MarkClientLocalFailure(std::vector<Status>* result, Status status) {
  for (auto& s : *result)
    if (s == Status::kIOError) s = status;
}

// Operation name for the failed-completion log line.
const char* WireOpLogName(uint8_t op) {
  switch (static_cast<WireOp>(op)) {
    case WireOp::kCache: return "put";
    case WireOp::kExist: return "exist";
    case WireOp::kMembers: return "members";
    case WireOp::kRemove: return "remove";
    case WireOp::kLookup: return "lookup";
    case WireOp::kPullRange: return "pull_range";
    case WireOp::kPullRelease: return "pull_release";
    case WireOp::kLeasePut: return "lease_put";
    case WireOp::kRetain: return "retain";
    case WireOp::kRemovedWithin: return "removed_within";
    case WireOp::kRemovedWithinAges: return "removed_within_ages";
    default: return "unknown";
  }
}

}  // namespace

struct RdmaTransport::Conn {
  rdma::RcEndpoint ep;
  rdma::RecvSegmentInfo recv_segment;
  size_t rail_index = 0;
  uint64_t peer_publication = 0;
  std::string peer_id;
  // The identity remote rail health keys this peer by: peer_id, or for a
  // peer the MDS does not name, its address (RemoteRailHealth::AddressIdentity).
  std::string health_id;
  std::string node;  // assigned once at connection creation, never per acquire
  std::string metric_peer;
  uint64_t remote_lease_generation = 0;
  bool remote_lease_held = false;
  bool remote_recovery_probe = false;
  // This acquisition took the connection from the pool, at the given count
  // of successful completions. Until that count moves nothing says the peer
  // still has the other end.
  bool reused_from_pool = false;
  uint64_t completions_at_reuse = 0;
  // The operation stopped because this process could not register a caller
  // buffer. Neither the rail nor the peer had a part in that.
  bool local_resource_failure = false;
  rdma::RailLease lease;
  uint64_t lease_started_us = 0;
  uint64_t lease_bytes = 0;  // payload moved under the current lease
  bool credit_held = false;
  rdma::ResourceRequest budget_request;
  bool budget_held = false;
  size_t declared_bytes = 0;
  size_t depth = 1;
  Lane lane = Lane::kData;
  // Negotiated staged-lease PUT datapath (probe bit + bootstrap request bit).
  // Off for pooled conns predating the opt-in and for peers without support.
  bool leased_put = false;
  bool leased_put_supported = false;
  bool active_counted = false;
  bool live_counted = false;
  bool visited = true;  // guarded by RdmaTransport::mu_ while idle
  uint64_t last_application_us = 0;  // keepalives do not renew idle retention

  rdma::ConnectionLifecycle lifecycle;
  bool keepalive_probe = false;  // a keepalive, not an application op, is in flight
  // The bootstrap socket kept as session channel (DFKV_RDMA_SESSION_FD); -1
  // without one. The server closes its end when it ends the connection.
  int session_fd = -1;
  ~Conn() {
    if (session_fd >= 0) ::close(session_fd);
  }
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

// Enumerability is not usability: a hosted machine can enumerate an active
// HCA, pass port/GID queries, and still reject the verbs objects the data
// path builds (observed on GitHub-hosted jobs: discovery succeeds, then every
// lease-suite server aborts with "failed to open required v2 device"). The
// availability probe therefore constructs the minimal endpoint the suites
// themselves require, instead of trusting discovery alone.
bool RdmaTransport::Available() {
  const char* configured = std::getenv("DFKV_RDMA_DEV");
  const auto filter =
      ParseDeviceList(configured ? std::string(configured) : std::string());
  for (const rdma::RdmaDevInfo& device : rdma::RdmaTopology::Discover(filter)) {
    rdma::RcEndpoint probe;
    probe.set_remote_access(false);
    if (probe.Open(device.name.c_str(), rdma::kV2ControlCap, 1)) return true;
  }
  return false;
}

RdmaTransport::RdmaTransport(size_t max_msg, const std::string& dev_name)
    : max_payload_(ResolveMaxPayload(max_msg)),
      declared_(std::min<uint64_t>(
          rdma_settings::ReadBytes("DFKV_RDMA_MAX_BLOCK_BYTES", 4ull << 20,
                                   kMaxSgeBytes),
          ResolveMaxPayload(max_msg))),
      connection_min_block_bytes_(std::min<size_t>(
          rdma_settings::ReadBytes("DFKV_RDMA_CONNECTION_MIN_BLOCK_BYTES",
                                   256ull << 10, kMaxSgeBytes),
          static_cast<size_t>(declared_))),
      depth_(4) {
  // Optional staged-lease PUT threshold (env DFKV_RDMA_INLINE_PUT_MAX_BYTES,
  // default 4 MiB; explicit 0 disables). Objects larger than this use the
  // per-op staging-lease datapath on connections whose peer negotiated it,
  // keeping the connection-resident receive-slot class at the threshold.
  // Unlike ReadBytes, an explicit 0 must stay 0 (disabled), not fall back.
  inline_put_max_bytes_ = 4ull << 20;
  size_t inline_resolved_state = 1;  // 1 = default, 0 = disabled, 2 = value
  if (const char* const v = std::getenv("DFKV_RDMA_INLINE_PUT_MAX_BYTES")) {
    if (*v) {
      errno = 0;
      char* end = nullptr;
      const unsigned long long x = std::strtoull(v, &end, 10);
      if (errno == 0 && end != v && *end == '\0') {
        inline_put_max_bytes_ = std::min<size_t>(
            static_cast<size_t>(x), max_payload_);
        inline_resolved_state = x == 0 ? 0 : 2;
      }
    } else {
      inline_put_max_bytes_ = 0;
      inline_resolved_state = 0;  // set to the empty string: disabled
    }
  }
  // 0 disables, and so does an empty value (it does not mean "unset" here).
  envparse::ReportUnlessPlain(
      "DFKV_RDMA_INLINE_PUT_MAX_BYTES", 0, std::numeric_limits<uint64_t>::max(),
      inline_resolved_state == 0 ? std::string("0 (disabled)")
                                 : std::to_string(inline_put_max_bytes_),
      /*empty_is_unset=*/false);
  config_dump::RecordResolved(
      "DFKV_RDMA_INLINE_PUT_MAX_BYTES",
      inline_resolved_state == 0
          ? std::string("0")
          : std::to_string(inline_put_max_bytes_));
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
  enomem_until_us_ =
      std::make_unique<std::atomic<uint64_t>[]>(devs_.size());
  rail_devices_.resize(devs_.size());
  device_rebuild_ = rdma::DeviceRebuildEnabled();
  rail_generation_ = std::make_unique<std::atomic<uint64_t>[]>(devs_.size());
  rail_stale_ = std::make_unique<std::atomic<uint8_t>[]>(devs_.size());
  device_suspects_ =
      std::make_unique<std::atomic<uint64_t>[]>(devs_.size() * 2);
  device_rebuilds_ = std::make_unique<std::atomic<uint64_t>[]>(devs_.size());
  for (size_t rail = 0; rail < devs_.size(); ++rail)
    rail_generation_[rail].store(rdma::DeviceGeneration(devs_[rail]));
  rebuild_not_before_us_.assign(devs_.size(), 0);
  rebuild_backoff_ms_.assign(devs_.size(), 0);
  if (const char* primary = std::getenv("DFKV_RDMA_PRIMARY_DEV");
      primary && *primary) {
    const auto found = std::find(devs_.begin(), devs_.end(), primary);
    if (found == devs_.end()) {
      throw std::runtime_error(
          "DFKV_RDMA_PRIMARY_DEV is not present in DFKV_RDMA_DEV");
    }
    preferred_rail_ =
        static_cast<size_t>(std::distance(devs_.begin(), found));
  }
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
  const int remote_base_cooldown_ms = rdma_settings::ReadBoundedInt(
      "DFKV_RDMA_REMOTE_RAIL_COOLDOWN_MS", 2000, 30000);
  const int remote_max_cooldown_ms = std::max(
      remote_base_cooldown_ms,
      rdma_settings::ReadBoundedInt("DFKV_RDMA_REMOTE_RAIL_MAX_COOLDOWN_MS", 30000, 30000));
  remote_rail_health_ = std::make_unique<RemoteRailHealth>(
      RemoteRailHealthConfig{
          static_cast<uint64_t>(remote_base_cooldown_ms) * 1000,
          static_cast<uint64_t>(remote_max_cooldown_ms) * 1000});
  // A node that refused a connect is not dialed again for as long as its
  // rails used to be cooled for it.
  peer_unreachable_hold_us_ =
      static_cast<uint64_t>(remote_base_cooldown_ms) * 1000;
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
  config_dump::RecordResolved(
      "DFKV_RDMA_PRIMARY_DEV",
      preferred_rail_ ? devs_[*preferred_rail_] : std::string(""));
  envparse::ReportUnlessPlain("DFKV_RDMA_DEPTH", 1, 256,
                              std::to_string(depth_));
  config_dump::RecordResolved("DFKV_RDMA_DEPTH", std::to_string(depth_));
  config_dump::RecordResolved(
      "DFKV_RDMA_CONNECTION_MIN_BLOCK_BYTES",
      std::to_string(connection_min_block_bytes_));
  config_dump::RecordResolved("DFKV_RDMA_NUMA", numa_aware_ ? "1" : "0");
  const size_t endpoint_limit = static_cast<size_t>(
      rdma_settings::ReadBoundedInt("DFKV_RDMA_ENDPOINT_CACHE_MAX", 256, 65536));
  const size_t qp_limit = static_cast<size_t>(
      rdma_settings::ReadBoundedInt("DFKV_RDMA_QP_BUDGET",
                    static_cast<int>(endpoint_limit), 65536));
  const size_t wr_limit = static_cast<size_t>(rdma_settings::ReadU64(
      "DFKV_RDMA_WR_BUDGET",
      static_cast<uint64_t>(endpoint_limit) * depth_));
  const uint64_t slot_bytes =
      static_cast<uint64_t>(rdma::V2SlotSize(static_cast<size_t>(declared_)));
  const uint64_t default_registered_budget =
      slot_bytes != 0 &&
              endpoint_limit <=
                  std::numeric_limits<uint64_t>::max() / slot_bytes / depth_ / 2
          ? 2 * slot_bytes * depth_ * endpoint_limit
          : std::numeric_limits<uint64_t>::max();
  const uint64_t registered_limit = rdma_settings::ReadU64(
      "DFKV_RDMA_REGISTERED_BYTES_BUDGET", default_registered_budget);
  resource_acquire_ms_ =
      rdma_settings::ReadBoundedInt("DFKV_RDMA_RESOURCE_ACQUIRE_MS", 10000, 600000);
  resource_budget_ = ProcessResourceBudget(
      {endpoint_limit, qp_limit, wr_limit, registered_limit});
  // Any explicitly pinned budget dimension turns the whole autoscale off:
  // partial-manual sizing recreated exactly the "QP raised but WR/registered
  // still starve" trap of 0812-002 §6.1, so explicit config wins wholesale.
  // Being set at all is what counts here, also to an empty or invalid value
  // that leaves the budget itself at its default: say so for the empty one
  // (an invalid one is reported by its reader above).
  for (const char* name :
       {"DFKV_RDMA_ENDPOINT_CACHE_MAX", "DFKV_RDMA_QP_BUDGET",
        "DFKV_RDMA_WR_BUDGET", "DFKV_RDMA_REGISTERED_BYTES_BUDGET"}) {
    const char* value = std::getenv(name);
    if (value && !*value)
      envparse::WarnOnce(
          name, std::string("invalid ") + name +
                    "='': set but empty; the budget keeps its default, and "
                    "being set switches the automatic budget scaling off, as "
                    "before. Unset it or give it a number.");
  }
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
      rdma_settings::ReadBoundedInt("DFKV_RDMA_RAIL_CREDITS", 64, 4096));
  rail_config.error_threshold = static_cast<uint32_t>(
      rdma_settings::ReadBoundedInt("DFKV_RDMA_RAIL_ERROR_THRESHOLD", 3, 1000));
  const int cooldown_ms =
      rdma_settings::ReadBoundedInt("DFKV_RDMA_RAIL_COOLDOWN_MS", 5000, 600000);
  rail_config.quarantine_us = static_cast<uint64_t>(cooldown_ms) * 1000;
  rail_config.latency_weight = static_cast<uint32_t>(
      rdma_settings::ReadBoundedInt("DFKV_RDMA_RAIL_LATENCY_WEIGHT", 1, 1000));
  rail_config.error_penalty_us = static_cast<uint64_t>(
      rdma_settings::ReadBoundedInt("DFKV_RDMA_RAIL_ERROR_PENALTY_US", 100000, 60000000));
  const int backpressure_ms =
      rdma_settings::ReadBoundedInt("DFKV_RDMA_RAIL_BACKPRESSURE_MS", 0, 60000);
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
  {
    // Read once here, not per connection: no getenv off the constructor.
    const char* bp = std::getenv("DFKV_RDMA_BUSY_POLL");
    busy_poll_ = bp && *bp && *bp != '0';
    envparse::ReportSwitch("DFKV_RDMA_BUSY_POLL", busy_poll_,
                           "only a value starting with 0 is off");
  }
  // How long a client completion wait polls before it sleeps (event mode).
  // 0 sleeps at once; at most 10000.
  if (const char* value = std::getenv("DFKV_RDMA_SPIN_US"); value && *value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (errno == 0 && end != value && *end == '\0' && parsed >= 0)
      spin_us_ = static_cast<uint32_t>(std::min<long>(parsed, 10000));
  }
  envparse::ReportUnlessPlain("DFKV_RDMA_SPIN_US", 0, 10000,
                              std::to_string(spin_us_));
  config_dump::RecordResolved("DFKV_RDMA_SPIN_US", std::to_string(spin_us_));
  connect_ms_ = rdma_settings::ReadInt("DFKV_RDMA_CONNECT_MS", 3000);
  io_ms_ = rdma_settings::ReadInt("DFKV_RDMA_IO_MS", 10000);
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
  {
    // How long one operation, all its attempts together, may wait for rail
    // credits: 200 ms unless set, and never past the operation timeout.
    int wait_ms = backpressure_ms > 0 ? backpressure_ms : 200;
    if (op_timeout_ms_ > 0) wait_ms = std::min(wait_ms, op_timeout_ms_);
    rail_backpressure_us_ = static_cast<uint64_t>(wait_ms) * 1000;
    config_dump::RecordResolved("DFKV_RDMA_RAIL_BACKPRESSURE_MS",
                                std::to_string(wait_ms));
  }
  // 0 means no deadline, and so does a value that is not a number at all;
  // a negative one means 5000.
  envparse::ReportUnlessPlain(
      "DFKV_RDMA_OP_TIMEOUT_MS", 0, std::numeric_limits<int>::max(),
      op_timeout_ms_ < 0 ? std::string("no deadline")
                         : std::to_string(op_timeout_ms_));
  // 0 (the default) follows DFKV_RDMA_OP_TIMEOUT_MS.
  envparse::ReportUnlessPlain(
      "DFKV_RDMA_BATCH_OP_TIMEOUT_MS", 0, std::numeric_limits<int>::max(),
      batch_op_timeout_ms_ == 0
          ? std::string("0 (follows DFKV_RDMA_OP_TIMEOUT_MS)")
          : std::to_string(batch_op_timeout_ms_));
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
  envparse::ReportUnlessPlain("DFKV_RDMA_KEEPALIVE_MS", 0,
                              std::numeric_limits<int>::max(),
                              std::to_string(keepalive_ms_));
  config_dump::RecordResolved("DFKV_RDMA_KEEPALIVE_MS",
                              std::to_string(keepalive_ms_));
  session_fd_enabled_ = config_dump::EnvBool("DFKV_RDMA_SESSION_FD", false);
  pipelined_reads_ = config_dump::EnvBool("DFKV_RDMA_PIPELINED_READS", false);
  if (session_fd_enabled_) {
    // One more descriptor per pooled connection: raise the soft limit to
    // the hard one, as the MDS does for its connections.
    rlimit nofile{};
    if (::getrlimit(RLIMIT_NOFILE, &nofile) == 0 &&
        nofile.rlim_cur < nofile.rlim_max) {
      rlimit raised = nofile;
      raised.rlim_cur = nofile.rlim_max;
      if (::setrlimit(RLIMIT_NOFILE, &raised) == 0)
        DFKV_LOG_INFO("rdma: DFKV_RDMA_SESSION_FD raised the open-file soft "
                      "limit from " + std::to_string(nofile.rlim_cur) +
                      " to " + std::to_string(raised.rlim_cur));
    }
  }
  // An explicit limit remains a hard per-pool retention limit. With no
  // override, only data endpoints get a bounded, expiring burst allowance.
  const int pool_override = rdma_settings::ReadBoundedInt(
      "DFKV_RDMA_POOL_MAX", 0, std::numeric_limits<int>::max());
  if (pool_override > 0)
    pool_max_ = data_pool_max_ = static_cast<size_t>(pool_override);
  config_dump::RecordResolved("DFKV_RDMA_POOL_MAX",
                              std::to_string(pool_max_));
  active_lane_rail_ =
      std::make_unique<std::atomic<uint64_t>[]>(3 * devs_.size());
  rail_conns_ = std::make_unique<std::atomic<uint64_t>[]>(devs_.size());
  rail_put_ops_ =
      std::make_unique<std::atomic<uint64_t>[]>(devs_.size());
  rail_put_bytes_ =
      std::make_unique<std::atomic<uint64_t>[]>(devs_.size());
  rail_get_ops_ =
      std::make_unique<std::atomic<uint64_t>[]>(devs_.size());
  rail_get_bytes_ =
      std::make_unique<std::atomic<uint64_t>[]>(devs_.size());
  rail_wc_errors_ = std::make_unique<std::atomic<uint64_t>[]>(
      devs_.size() * rdma::kWcStatusSlots);
  // Watch each rail's port at run time; 0 switches it off.
  if (const char* value = std::getenv("DFKV_RDMA_PORT_CHECK_MS");
      value && *value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (errno == 0 && end != value && *end == '\0' && parsed >= 0)
      port_check_ms_ = static_cast<int>(std::min<long>(parsed, 60000));
  }
  envparse::ReportUnlessPlain("DFKV_RDMA_PORT_CHECK_MS", 0, 60000,
                              std::to_string(port_check_ms_));
  config_dump::RecordResolved("DFKV_RDMA_PORT_CHECK_MS",
                              std::to_string(port_check_ms_));
  if (port_check_ms_ > 0) {
    port_watcher_ = std::make_unique<rdma::PortWatcher>(devs_);
    port_active_.assign(devs_.size(), 1);
  }
  if (keepalive_ms_ > 0 || data_pool_max_ > pool_max_ || port_watcher_ ||
      device_rebuild_) {
    // Recovery probes run on this thread, so application operations go to a
    // healthy rail first.
    rail_policy_->SetProbeAfterHealthy(true);
    keepalive_thread_ = std::thread(&RdmaTransport::KeepaliveLoop, this);
  }
}

RdmaTransport::~RdmaTransport() {
  // The background thread first: a rail rebuild on it takes register_mu_
  // (try_lock only), so it is joined before the lock is taken.
  keepalive_stop_.store(true, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lock(keepalive_mu_);
    keepalive_cv_.notify_all();
  }
  if (keepalive_thread_.joinable()) keepalive_thread_.join();
  // A declaration still registering finishes before anything goes away.
  std::lock_guard<std::mutex> declaring(register_mu_);

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
    detach(control_pool_);
  }
  for (Conn* conn : idle) Destroy(conn);
}

bool RdmaTransport::KeepaliveConn(Conn* conn,
                                  rdma::RailCompletion* failure) {
  if (!conn || !failure) return false;
  *failure = rdma::RailCompletion::kAdmission;
  keepalive_attempts_.fetch_add(1, std::memory_order_relaxed);
  conn->keepalive_probe = true;
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
  conn->keepalive_probe = false;
  return true;
}

void RdmaTransport::TrimIdle(uint64_t now_us) {
  std::vector<Conn*> retired;
  {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& [node, connections] : pool_) {
      (void)node;
      for (auto it = connections.begin();
           it != connections.end() && connections.size() > pool_max_;) {
        Conn* conn = *it;
        if (now_us >= conn->last_application_us &&
            now_us - conn->last_application_us >= kBurstIdleUs &&
            conn->lifecycle.RequestRetire()) {
          retired.push_back(conn);
          it = connections.erase(it);
        } else {
          ++it;
        }
      }
    }
  }
  for (Conn* conn : retired) Destroy(conn);
  endpoint_cache_evictions_.fetch_add(retired.size(), std::memory_order_relaxed);
}

void RdmaTransport::MaintainIdle(uint64_t now_us, bool send_keepalives) {
  TrimIdle(now_us);
  if (!send_keepalives) return;
  // The connections idle now are probed one after another, and each is taken
  // out of its pool only for its own probe. The others stay usable meanwhile;
  // a probe that waits for a late answer would otherwise leave operations to
  // every node without a pooled connection. A listed connection that an
  // operation has taken in the meantime is skipped, as is one that is gone:
  // it is looked for in its pool, never dereferenced before.
  struct Listed {
    bool control;
    std::string node;
    Conn* conn;
  };
  std::vector<Listed> listed;
  {
    std::lock_guard<std::mutex> lock(mu_);
    const auto list = [&](bool control) {
      for (const auto& [node, connections] :
           control ? control_pool_ : pool_)
        for (Conn* conn : connections)
          listed.push_back({control, node, conn});
    };
    list(false);
    list(true);
  }
  for (const Listed& entry : listed) {
    if (keepalive_stop_.load(std::memory_order_relaxed)) return;
    Conn* conn = nullptr;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto& pools = entry.control ? control_pool_ : pool_;
      const auto found = pools.find(entry.node);
      if (found == pools.end()) continue;
      auto& connections = found->second;
      const auto it =
          std::find(connections.begin(), connections.end(), entry.conn);
      if (it == connections.end() || !(*it)->lifecycle.Activate()) continue;
      conn = *it;
      connections.erase(it);
      MarkActive(conn, conn->lane);
    }
    if (!peer_topologies_->IsCurrent(conn->node, conn->peer_id,
                                     conn->peer_publication)) {
      stale_publication_reaps_.fetch_add(1, std::memory_order_relaxed);
      Destroy(conn, rdma::RailCompletion::kAdmission);
      continue;
    }
    // Ended by the server: a keepalive would only wait out RC retries.
    if (SessionClosed(*conn)) {
      session_closed_reaps_.fetch_add(1, std::memory_order_relaxed);
      Destroy(conn, rdma::RailCompletion::kAdmission);
      continue;
    }
    if (!conn->peer_id.empty()) {
      const auto remote = remote_rail_health_->TryAcquire(
          conn->health_id, conn->rail_index, rdma::RailPolicy::NowMicros());
      if (!remote) {
        Destroy(conn, rdma::RailCompletion::kAdmission);
        continue;
      }
      conn->remote_lease_generation = remote->generation;
      conn->remote_lease_held = true;
      conn->remote_recovery_probe = remote->recovery_probe;
    }
    conn->reused_from_pool = false;  // a keepalive is judged in Destroy
    rdma::RailCompletion failure = rdma::RailCompletion::kAdmission;
    if (!KeepaliveConn(conn, &failure))
      Destroy(conn, failure);
    else
      Release(conn->node, conn->lane, conn,
              RemoteRailOutcome::kSuccess, std::nullopt);
  }
}

void RdmaTransport::KeepaliveLoop() {
  const uint64_t interval_us = static_cast<uint64_t>(keepalive_ms_) * 1000;
  uint64_t last_keepalive_us = rdma::RailPolicy::NowMicros();
  int tick_ms = keepalive_ms_ > 0 ? std::min(keepalive_ms_, 1000) : 1000;
  if (port_watcher_) tick_ms = std::min(tick_ms, port_check_ms_);
  std::unique_lock<std::mutex> wait_lock(keepalive_mu_);
  while (!keepalive_cv_.wait_for(
      wait_lock, std::chrono::milliseconds(tick_ms),
      [&] { return keepalive_stop_.load(std::memory_order_relaxed); })) {
    wait_lock.unlock();
    CheckDevices();
    if (port_watcher_) CheckPorts();
    ProbeRails();
    const uint64_t now = rdma::RailPolicy::NowMicros();
    const bool send = keepalive_ms_ > 0 &&
                      now - last_keepalive_us >= interval_us;
    MaintainIdle(now, send);
    if (send) last_keepalive_us = now;
    wait_lock.lock();
  }
}

int RdmaTransport::QpAccessFlags(const Conn* conn) {
  return conn ? conn->ep.QpAccessFlags() : -1;
}

void RdmaTransport::CheckPorts() {
  for (size_t rail = 0; rail < devs_.size(); ++rail) {
    const int state = port_watcher_->State(rail);
    if (state < 0 || (state == 1) == (port_active_[rail] != 0)) continue;
    port_active_[rail] = static_cast<uint8_t>(state);
    // A rail out of selection for a rebuild comes back with the rebuild.
    topology_->SetDeviceEnabled(rail, state == 1 && !RailStale(rail));
    if (state == 1) {
      rails_port_down_.fetch_sub(1, std::memory_order_relaxed);
      DFKV_LOG_INFO("rdma: port ACTIVE again, rail back in selection dev=" +
                    devs_[rail]);
    } else {
      port_down_events_.fetch_add(1, std::memory_order_relaxed);
      rails_port_down_.fetch_add(1, std::memory_order_relaxed);
      DFKV_LOG_WARN("rdma: port not ACTIVE, rail out of selection dev=" +
                    devs_[rail]);
      RetireIdleRail(rail);
    }
  }
}

bool RdmaTransport::RailStale(size_t rail) const {
  if (!device_rebuild_ || rail >= devs_.size()) return false;
  return rail_stale_[rail].load(std::memory_order_acquire) != 0 ||
         rdma::DeviceGeneration(devs_[rail]) !=
             rail_generation_[rail].load(std::memory_order_acquire);
}

void RdmaTransport::CheckDevices() {
  if (device_checks_paused_for_test_.load(std::memory_order_relaxed)) return;
  for (const auto& event : rdma::PollDeviceEvents()) {
    {
      std::lock_guard<std::mutex> lock(device_events_mu_);
      ++device_events_[{event.device, event.event}];
    }
    if (!event.fatal) {
      DFKV_LOG_INFO("rdma: device event dev=" + event.device + ": " +
                    event.event);
      continue;
    }
    DFKV_LOG_WARN("rdma: device fatal error dev=" + event.device);
    const auto rail = std::find(devs_.begin(), devs_.end(), event.device);
    if (rail != devs_.end())
      device_suspects_[static_cast<size_t>(rail - devs_.begin()) * 2]
          .fetch_add(1, std::memory_order_relaxed);
    if (device_rebuild_) rdma::RetireDevice(event.device, event.generation);
  }
  if (!device_rebuild_) return;
  const uint64_t now = rdma::RailPolicy::NowMicros();
  for (size_t rail = 0; rail < devs_.size(); ++rail) {
    if (keepalive_stop_.load(std::memory_order_relaxed)) return;
    if (!RailStale(rail)) continue;
    MarkRailStale(rail);
    if (now < rebuild_not_before_us_[rail]) continue;
    if (RebuildRail(rail, rdma::DeviceGeneration(devs_[rail]))) {
      rebuild_backoff_ms_[rail] = 0;
      rebuild_not_before_us_[rail] = 0;
    } else {
      // 1 s, doubling to 30 s; the wait is the background thread's tick.
      rebuild_backoff_ms_[rail] =
          rebuild_backoff_ms_[rail] == 0
              ? 1000
              : std::min<uint64_t>(rebuild_backoff_ms_[rail] * 2, 30000);
      rebuild_not_before_us_[rail] = now + rebuild_backoff_ms_[rail] * 1000;
    }
  }
}

void RdmaTransport::MarkRailStale(size_t rail) {
  if (rail_stale_[rail].exchange(1, std::memory_order_acq_rel) == 0)
    DFKV_LOG_WARN("rdma: device of rail dev=" + devs_[rail] +
                  " was reset; rail out of selection until it is rebuilt");
  // Again on every tick: the port watcher, or a connection taken meanwhile,
  // may have put it back.
  topology_->SetDeviceEnabled(rail, false);
  RetireIdleRail(rail);
  std::lock_guard<std::mutex> lock(rail_devices_mu_);
  rail_devices_[rail].reset();
}

bool RdmaTransport::RebuildRail(size_t rail, uint64_t generation) {
  // One attempt holds register_mu_, never while waiting; a RegisterMemory
  // (or the destructor) holding it makes this attempt wait for a later tick.
  std::unique_lock<std::mutex> declaring(register_mu_, std::try_to_lock);
  if (!declaring.owns_lock()) return false;
  const std::string& dev = devs_[rail];
  std::vector<std::pair<void*, size_t>> regions;
  {
    std::lock_guard<std::mutex> lock(mu_);
    regions = pools_;
  }
  // A transport that declared memory has an anchor per rail: a new one on
  // the fresh context, with every declared region registered on it.
  const bool anchored = !anchors_.empty();
  std::unique_ptr<rdma::RcEndpoint> anchor;
  uint64_t built_on = generation;
  std::vector<rdma::RcEndpoint::PoolMrRegistration> registrations(
      regions.size());
  if (rebuild_fails_for_test_.load(std::memory_order_relaxed)) return false;
  if (anchored) {
    anchor = std::make_unique<rdma::RcEndpoint>();
    anchor->set_remote_access(false);
    if (!anchor->Open(dev.empty() ? nullptr : dev.c_str(), rdma::kV2ControlCap,
                      1)) {
      DFKV_LOG_WARN("rdma: rail dev=" + dev +
                    " cannot be rebuilt yet: its device does not open");
      return false;
    }
    built_on = anchor->device_generation();
    size_t staged = 0;
    for (; staged < regions.size(); ++staged) {
      // A large pool must not hold the destructor's join for seconds.
      if (keepalive_stop_.load(std::memory_order_relaxed)) break;
      if (!anchor->StagePoolMr(regions[staged].first, regions[staged].second,
                               /*remote_write=*/false,
                               &registrations[staged]))
        break;
    }
    if (staged != regions.size()) {
      while (staged != 0) {
        --staged;
        anchor->RollbackPoolMr(&registrations[staged]);
      }
      DFKV_LOG_WARN("rdma: rail dev=" + dev +
                    " not rebuilt: registering its memory failed");
      return false;
    }
  }
  {
    rdma::RetiredPoolMrs retired;
    std::unique_ptr<rdma::RcEndpoint> old;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (anchored) {
        for (auto& registration : registrations)
          anchor->CommitPoolMr(&registration, retired.get());
        old = std::move(anchors_[rail]);
        anchors_[rail] = std::move(anchor);
      }
      rail_generation_[rail].store(built_on, std::memory_order_release);
    }
    retired.Deregister();
    old.reset();  // the old context's last references go with it
  }
  declaring.unlock();
  rail_stale_[rail].store(0, std::memory_order_release);
  if (port_watcher_) port_watcher_->Reopen(rail);
  if (!port_watcher_ || port_active_[rail]) topology_->SetDeviceEnabled(rail, true);
  device_rebuilds_[rail].fetch_add(1, std::memory_order_relaxed);
  DFKV_LOG_INFO("rdma: rail dev=" + dev + " rebuilt on its reset device (" +
                std::to_string(regions.size()) + " regions registered)");
  return true;
}

void RdmaTransport::ProbeRails() {
  const auto due = rail_policy_->ProbesDue(rdma::RailPolicy::NowMicros());
  if (due.empty()) return;
  // Nodes with idle connections first (they answer), then any connected to.
  std::vector<std::string> nodes;
  {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto* pools : {&control_pool_, &pool_})
      for (const auto& [node, connections] : *pools)
        if (!connections.empty()) nodes.push_back(node);
  }
  {
    std::lock_guard<std::mutex> lock(known_nodes_mu_);
    nodes.insert(nodes.end(), known_nodes_.begin(), known_nodes_.end());
  }
  std::vector<std::string> unique;
  for (const auto& node : nodes)
    if (std::find(unique.begin(), unique.end(), node) == unique.end())
      unique.push_back(node);
  if (unique.empty()) return;  // no peer yet; application traffic will probe
  for (size_t rail : due) {
    size_t tried = 0;
    for (const auto& node : unique) {
      if (keepalive_stop_.load(std::memory_order_relaxed)) return;
      if (tried++ == kProbeNodes) break;
      AcquireOptions options;
      options.excluded.assign(devs_.size(), 1);
      options.excluded[rail] = 0;
      options.backpressure_us = 0;
      AcquireResult acquired = Acquire(node, Lane::kControl, options);
      if (!acquired.conn) continue;  // e.g. the peer's rail is cooled
      rail_background_probes_.fetch_add(1, std::memory_order_relaxed);
      Conn* conn = acquired.conn;
      rdma::RailCompletion failure = rdma::RailCompletion::kAdmission;
      if (!KeepaliveConn(conn, &failure))
        Destroy(conn, failure);
      else
        Release(conn->node, conn->lane, conn, RemoteRailOutcome::kSuccess,
                std::nullopt);
      break;
    }
  }
}

void RdmaTransport::RetireIdleRail(size_t local_rail) {
  RetireIdleIf(
      [&](const Conn* conn) { return conn->rail_index == local_rail; });
}

void RdmaTransport::RetireIdlePeerRail(const std::string& peer_id,
                                       size_t local_rail) {
  if (peer_id.empty()) return;
  RetireIdleIf([&](const Conn* conn) {
    return conn->health_id == peer_id && conn->rail_index == local_rail;
  });
}

void RdmaTransport::RetireIdleIf(
    const std::function<bool(const Conn*)>& match) {
  std::vector<Conn*> retired;
  {
    std::lock_guard<std::mutex> lock(mu_);
    const auto reap = [&](auto& pools) {
      for (auto map_it = pools.begin(); map_it != pools.end();) {
        auto& connections = map_it->second;
        for (auto it = connections.begin(); it != connections.end();) {
          Conn* conn = *it;
          if (!match(conn) || !conn->lifecycle.RequestRetire()) {
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
        "rdma: remote endpoint rail cooled peer=" + HealthIdLabel(peer_id) +
        " dev=" + dev + " cooldown_ms=" +
        std::to_string(completion.cooldown_until_us > now
                           ? (completion.cooldown_until_us - now) / 1000
                           : 0));
    RetireIdlePeerRail(peer_id, local_rail);
  } else if (completion.transition == RemoteRailTransition::kRecovered) {
    DFKV_LOG_INFO("rdma: remote endpoint rail recovered peer=" + HealthIdLabel(peer_id) +
                  " dev=" + dev);
  }
}

void RdmaTransport::CompleteRemoteLease(Conn* c,
                                        RemoteRailOutcome outcome) {
  if (!c || !c->remote_lease_held) return;
  c->remote_lease_held = false;
  CompleteRemote(c->health_id, c->rail_index, c->remote_lease_generation,
                 outcome);
}

void RdmaTransport::RecordRailTransfer(Conn* c, bool put,
                                       uint64_t operations, uint64_t bytes) {
  if (!c || c->rail_index >= devs_.size() || operations == 0) return;
  c->lease_bytes += bytes;
  auto& op_counters = put ? rail_put_ops_ : rail_get_ops_;
  auto& byte_counters = put ? rail_put_bytes_ : rail_get_bytes_;
  op_counters[c->rail_index].fetch_add(operations, std::memory_order_relaxed);
  byte_counters[c->rail_index].fetch_add(bytes, std::memory_order_relaxed);
}

void RdmaTransport::MarkClassOpened(Conn* c) {
  if (!c || c->lane == Lane::kControl) return;
  std::lock_guard<std::mutex> lock(connection_class_mu_);
  ++connection_class_stats_[{c->declared_bytes, c->depth}].opened;
}

void RdmaTransport::MarkActive(Conn* c, Lane lane) {
  if (!c || c->active_counted || c->rail_index >= devs_.size()) return;
  c->lane = lane;
  c->active_counted = true;
  const size_t index =
      static_cast<size_t>(lane) * devs_.size() + c->rail_index;
  active_lane_rail_[index].fetch_add(1, std::memory_order_relaxed);
  if (lane != Lane::kControl) {
    std::lock_guard<std::mutex> lock(connection_class_mu_);
    ++connection_class_stats_[{c->declared_bytes, c->depth}].active;
  }
}

void RdmaTransport::MarkInactive(Conn* c) {
  if (!c || !c->active_counted || c->rail_index >= devs_.size()) return;
  c->active_counted = false;
  const size_t index =
      static_cast<size_t>(c->lane) * devs_.size() + c->rail_index;
  active_lane_rail_[index].fetch_sub(1, std::memory_order_relaxed);
  if (c->lane != Lane::kControl) {
    std::lock_guard<std::mutex> lock(connection_class_mu_);
    auto& active =
        connection_class_stats_[{c->declared_bytes, c->depth}].active;
    if (active != 0) --active;
  }
}


void RdmaTransport::MarkLive(Conn* c) {
  if (!c || c->live_counted || c->rail_index >= devs_.size()) return;
  std::lock_guard<std::mutex> lock(peer_connections_mu_);
  ++peer_connections_[{c->metric_peer, c->rail_index}];
  c->live_counted = true;
}

void RdmaTransport::MarkDead(Conn* c) {
  if (!c || !c->live_counted || c->rail_index >= devs_.size()) return;
  std::lock_guard<std::mutex> lock(peer_connections_mu_);
  const auto key = std::make_pair(c->metric_peer, c->rail_index);
  const auto it = peer_connections_.find(key);
  if (it != peer_connections_.end() && --it->second == 0)
    peer_connections_.erase(it);
  c->live_counted = false;
}

void RdmaTransport::NoteFailedCompletion(Conn& c,
                                         rdma::RailCompletion completion,
                                         bool peer_rail_charged) {
  const ibv_wc* wc = c.ep.failed_wc();
  if (!wc || c.rail_index >= devs_.size()) return;
  rail_wc_errors_[c.rail_index * rdma::kWcStatusSlots +
                  rdma::WcStatusSlot(wc->status)]
      .fetch_add(1, std::memory_order_relaxed);
  const uint64_t now = rdma::RailPolicy::NowMicros();
  uint64_t suppressed = 0;
  if (!completion_error_log_.Admit(
          LogThrottle::Key(
              c.node, c.rail_index,
              rdma::WcFailureKind(wc->status, wc->vendor_err)),
          now, &suppressed))
    return;
  // A failed completion carries no reliable opcode, but its wr_id is the slot,
  // and the slot's send buffer still holds the request that was in flight.
  const char* op = "unknown";
  if (c.keepalive_probe)
    op = "keepalive";
  else if (wc->wr_id < c.ep.depth())
    op = WireOpLogName(static_cast<uint8_t>(c.ep.sbuf(wc->wr_id)[1]));
  const bool pooled = c.last_application_us != 0;
  DFKV_LOG_WARN(
      "rdma client CQ failure peer=" + c.node +
      " peer_id=" + (c.peer_id.empty() ? std::string("-") : c.peer_id) +
      " dev=" +
      (devs_[c.rail_index].empty() ? std::string("default")
                                   : devs_[c.rail_index]) +
      " op=" + op + " " +
      rdma::WcStatusFields(wc->status, wc->vendor_err) +
      " wr_id=" + std::to_string(wc->wr_id) +
      " qp=" + std::to_string(c.ep.Local().qpn) +
      " lane=" +
      (c.lane == Lane::kControl ? "control"
                                : c.lane == Lane::kSgData ? "sg" : "payload") +
      " pooled=" + (pooled ? "1" : "0") +
      " idle_ms=" +
      std::to_string(pooled && now > c.last_application_us
                         ? (now - c.last_application_us) / 1000
                         : 0) +
      " verdict=" +
      (completion == rdma::RailCompletion::kRailFailure
           ? "rail"
           : completion == rdma::RailCompletion::kEndpointFailure ? "endpoint"
                                                                  : "neutral") +
      " suppressed=" + std::to_string(suppressed) +
      " peer_rail_charged=" + (peer_rail_charged ? "1" : "0"));
}

bool RdmaTransport::Destroy(Conn* c, rdma::RailCompletion completion) {
  if (!c) return false;
  if (c->local_resource_failure)
    completion = rdma::RailCompletion::kAdmission;
  // A pooled connection that fails on reuse, before a single completion has
  // succeeded on it, says that this connection is gone, which is what a peer
  // does when it reclaims an idle connection or restarts. It does not say
  // that the peer cannot be reached over this rail:
  // only a connection made now can say that, and the caller's retry makes
  // one. So the peer rail is not cooled, which would also refuse that retry
  // where the rail is the only usable one. The other idle connections to the
  // peer on this rail are as old as this one and are dropped, so that the
  // next operations do not each find a dead one.
  const bool stale_pooled =
      completion == rdma::RailCompletion::kEndpointFailure &&
      c->reused_from_pool &&
      c->ep.succeeded_completions() == c->completions_at_reuse;
  const std::string stale_peer_id = stale_pooled ? c->health_id : std::string();
  const size_t stale_rail = c->rail_index;
  // A keepalive that fails says no more than that: it probes a pooled
  // connection, and its limit (100 ms) is much shorter than an operation's,
  // so a busy server that answers late fails it too. The connection is
  // dropped; a connection made now judges the rail.
  const bool charge_peer_rail =
      completion == rdma::RailCompletion::kEndpointFailure && !stale_pooled &&
      !c->keepalive_probe;
  NoteFailedCompletion(*c, completion,
                       charge_peer_rail && c->remote_lease_held);
  MarkInactive(c);
  if (completion == rdma::RailCompletion::kEndpointFailure)
    InvalidateCapabilities(c->node, c->peer_id, c->peer_publication);
  MarkDead(c);

  c->lifecycle.BeginDrain();
  const bool credit_held = c->credit_held;
  const rdma::RailLease lease = c->lease;
  const uint64_t lease_started_us = c->lease_started_us;
  const bool release_budget = c->budget_held;
  const rdma::ResourceRequest budget_request = c->budget_request;
  CompleteRemoteLease(c, charge_peer_rail
                             ? RemoteRailOutcome::kEndpointFailure
                             : RemoteRailOutcome::kAbandon);

  // RcEndpoint destroys its QP before deregistering MRs. In particular this
  // fences an outstanding initiator READ before caller memory can be reused.
  delete c;
  if (credit_held) {
    const uint64_t now = rdma::RailPolicy::NowMicros();
    rail_policy_->Complete(lease, now - lease_started_us, completion, now);
  }
  if (release_budget) resource_budget_->Release(budget_request);
  if (stale_pooled) {
    stale_pool_connections_.fetch_add(1, std::memory_order_relaxed);
    RetireIdlePeerRail(stale_peer_id, stale_rail);
  }
  return stale_pooled;
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
      if (scan(pool_) || scan(control_pool_)) break;
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
        nodes, std::max(pool_max_, devs_.size()), depth_, slot_bytes,
        /*floor_endpoints=*/256);
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
    for (auto it = peer_capabilities_.begin();
         it != peer_capabilities_.end();) {
      if (!peer_topologies_->IsCurrent(
              it->first, it->second.peer_id, it->second.publication))
        it = peer_capabilities_.erase(it);
      else
        ++it;
    }
    const auto reap = [&](auto& pools) {
      for (auto found = pools.begin(); found != pools.end();) {
        auto& connections = found->second;
        for (auto it = connections.begin(); it != connections.end();) {
          Conn* conn = *it;
          // Updating one identity can also supersede its old address.
          const bool current =
              (topology.present || found->first != topology.peer_addr) &&
              peer_topologies_->IsCurrent(
                  found->first, conn->peer_id, conn->peer_publication);
          if (current || !conn->lifecycle.RequestRetire()) {
            ++it;
            continue;
          }
          stale.push_back(conn);
          it = connections.erase(it);
        }
        if (connections.empty())
          found = pools.erase(found);
        else
          ++found;
      }
    };
    reap(pool_);
    reap(control_pool_);
  }
  peer_topology_updates_.fetch_add(1, std::memory_order_relaxed);
  if (!topology.present) {
    // A node that left the ring is no longer a probe target (ProbeRails).
    std::lock_guard<std::mutex> lock(known_nodes_mu_);
    known_nodes_.erase(topology.peer_addr);
  }
  for (Conn* conn : stale)
    Destroy(conn, rdma::RailCompletion::kAdmission);
  stale_publication_reaps_.fetch_add(stale.size(), std::memory_order_relaxed);
}

void RdmaTransport::OnPeerIdentities(
    const std::vector<std::string>& live_peer_ids) {
  remote_rail_health_->Reconcile(live_peer_ids);
}

void RdmaTransport::InvalidateCapabilities(const std::string& node,
                                           const std::string& peer_id,
                                           uint64_t publication) {
  std::lock_guard<std::mutex> lock(mu_);
  const auto found = peer_capabilities_.find(node);
  if (found != peer_capabilities_.end() &&
      found->second.peer_id == peer_id &&
      found->second.publication == publication)
    peer_capabilities_.erase(found);
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

bool RdmaTransport::SessionClosed(const Conn& conn) {
  if (conn.session_fd < 0) return false;
  char byte;
  const ssize_t got =
      ::recv(conn.session_fd, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
  // 0: the server closed its end. The server sends nothing on it, so any
  // other answer than "nothing yet" is a dead channel too.
  return got == 0 || (got < 0 && errno != EAGAIN && errno != EWOULDBLOCK &&
                      errno != EINTR) ||
         got > 0;
}

uint64_t RdmaTransport::DeadlineIn(int timeout_ms) {
  return timeout_ms > 0 ? rdma::RailPolicy::NowMicros() +
                              static_cast<uint64_t>(timeout_ms) * 1000
                        : 0;
}

int RdmaTransport::RemainingMs(const OpState* op, int cap_ms) {
  if (!op || op->deadline_us == 0) return cap_ms;
  const uint64_t now = rdma::RailPolicy::NowMicros();
  const uint64_t left_us = op->deadline_us > now ? op->deadline_us - now : 0;
  const int left_ms = static_cast<int>(std::min<uint64_t>(
      (left_us + 999) / 1000, std::numeric_limits<int>::max()));
  return cap_ms < 0 ? left_ms : std::min(cap_ms, left_ms);
}

RdmaTransport::ProbeResult RdmaTransport::ProbeV2(
    const std::string& node, bool* leased_put_supported, int connect_ms,
    int io_ms, bool* session_fd_supported, bool* read_depth_supported) const {
  v2_probe_attempts_.fetch_add(1, std::memory_order_relaxed);
  if (leased_put_supported) *leased_put_supported = false;
  if (session_fd_supported) *session_fd_supported = false;
  if (read_depth_supported) *read_depth_supported = false;
  int fd = DialNode(node, connect_ms > 0 ? connect_ms : connect_ms_,
                    io_ms > 0 ? io_ms : io_ms_);
  if (fd < 0) {
    v2_probe_failures_.fetch_add(1, std::memory_order_relaxed);
    return ProbeResult::kUnreachable;
  }
  char frame[rdma::kDevNameBytes];
  rdma::EncodeDevFrame(rdma::kV2ProbeDevice, declared_, frame,
                       rdma::kDevProtoV2);
  char reply[rdma::kV2ProbeReplyBytes];
  errno = 0;
  const bool answered = net::WriteAll(fd, frame, sizeof(frame)) &&
                        net::ReadAll(fd, reply, sizeof(reply));
  const bool closed = !answered && !SocketTimedOut(errno);
  const bool ok =
      answered &&
      rdma::V2ProbeSupportsDynamicOnly(reply) &&
      rdma::V2ProbeSupportsDynamicPull(reply) &&
      rdma::V2ProbeSupportsPullRead(reply);
  const bool leased = ok && rdma::V2ProbeSupportsLeasedPut(reply);
  const bool session = ok && rdma::V2ProbeSupportsSessionFd(reply);
  const bool read_depth = ok && rdma::V2ProbeSupportsReadDepth(reply);
  ::close(fd);
  if (!ok)
    v2_probe_failures_.fetch_add(1, std::memory_order_relaxed);
  else if (leased_put_supported)
    *leased_put_supported = leased;
  if (ok && session_fd_supported) *session_fd_supported = session;
  if (ok && read_depth_supported) *read_depth_supported = read_depth;
  return ok ? ProbeResult::kSupported
            : closed ? ProbeResult::kClosed : ProbeResult::kRejected;
}

namespace {
// What the operations on this thread did about unreachable peers since the
// record was last taken: refused a connect because the address was held, and
// made a connect that failed. TakeUnreachableRefusal reports a refusal only
// when no connect failed, so a real failure is never taken for a refusal.
thread_local bool tls_refused_held_address = false;
thread_local bool tls_connect_failed = false;
}  // namespace

bool RdmaTransport::TakeUnreachableRefusal() {
  const bool refused_only = tls_refused_held_address && !tls_connect_failed;
  tls_refused_held_address = false;
  tls_connect_failed = false;
  return refused_only;
}

bool RdmaTransport::RecentlyUnreachable(const std::string& node) {
  std::lock_guard<std::mutex> lock(unreachable_mu_);
  const auto found = unreachable_until_us_.find(node);
  if (found == unreachable_until_us_.end()) return false;
  if (rdma::RailPolicy::NowMicros() < found->second) {
    peer_unreachable_skips_.fetch_add(1, std::memory_order_relaxed);
    tls_refused_held_address = true;
    return true;
  }
  unreachable_until_us_.erase(found);
  return false;
}

void RdmaTransport::EnableTestHooks() {
  g_test_hooks.store(true, std::memory_order_relaxed);
}

int RdmaTransport::DialNode(const std::string& node, int connect_ms,
                            int io_ms) const {
  for (int left = fail_dials_for_test_.load(); left > 0;
       left = fail_dials_for_test_.load()) {
    if (fail_dials_for_test_.compare_exchange_weak(left, left - 1)) {
      errno = ETIMEDOUT;
      return -1;
    }
  }
  return net::Dial(node, connect_ms, io_ms);
}

void RdmaTransport::NoteUnreachable(const std::string& node) {
  tls_connect_failed = true;
  std::lock_guard<std::mutex> lock(unreachable_mu_);
  const uint64_t now = rdma::RailPolicy::NowMicros();
  auto& until = unreachable_until_us_[node];
  // Counted, and the interval started, by the connect that failed; the
  // attempts refused during the interval are skips, counted where refused.
  if (until > now) return;
  peer_unreachable_.fetch_add(1, std::memory_order_relaxed);
  until = now + peer_unreachable_hold_us_;
}

void RdmaTransport::NoteLocalRegistrationFailure(Conn* c, size_t bytes) {
  const int err = errno;  // of the registration that just failed
  // What this process or its caller can cause: the locked-memory limit
  // (ENOMEM, EAGAIN) and a buffer that is not there or cannot be pinned
  // (EFAULT, EINVAL). Anything else stays what it was, a fault of the rail.
  if (err != ENOMEM && err != EAGAIN && err != EFAULT && err != EINVAL)
    return;
  c->local_resource_failure = true;
  local_resource_failures_.fetch_add(1, std::memory_order_relaxed);
  uint64_t suppressed = 0;
  if (!local_resource_log_.Admit(
          LogThrottle::Key("register", static_cast<uint64_t>(err), 0),
          rdma::RailPolicy::NowMicros(), &suppressed))
    return;
  DFKV_LOG_WARN(
      "rdma: cannot register a caller buffer of " + std::to_string(bytes) +
      " bytes for a transfer: " + std::strerror(err) +
      ". This is a limit or a buffer of this process, not a rail or peer "
      "fault" +
      (err == ENOMEM || err == EAGAIN
           ? "; most likely the locked-memory limit (ulimit -l, "
             "LimitMEMLOCK) is too low"
           : "; the address cannot be pinned") +
      ". The operation fails with kResourceExhausted and no rail is "
      "charged. suppressed=" + std::to_string(suppressed));
}


uint64_t RdmaTransport::AdmissionWait(const AcquireOptions& options) const {
  if (options.backpressure_us) return *options.backpressure_us;
  if (!options.op) return rail_backpressure_us_;
  const uint64_t now = rdma::RailPolicy::NowMicros();
  if (options.op->admission_deadline_us == 0)
    options.op->admission_deadline_us = now + rail_backpressure_us_;
  uint64_t until = options.op->admission_deadline_us;
  if (options.op->deadline_us != 0)
    until = std::min(until, options.op->deadline_us);
  return until > now ? until - now : 0;
}

void RdmaTransport::NoteConnected(const AcquireOptions& options,
                                  const std::string& node,
                                  const std::string& health_id, size_t rail) {
  {
    std::lock_guard<std::mutex> lock(known_nodes_mu_);
    known_nodes_.insert(node);
  }
  // A peer without an MDS identity takes no remote lease on its operations,
  // so a connection is what tells its rail recovered from a refusal.
  if (RemoteRailHealth::IsAddressIdentity(health_id)) {
    const uint64_t now = rdma::RailPolicy::NowMicros();
    if (const auto lease = remote_rail_health_->TryAcquire(health_id, rail, now))
      CompleteRemote(health_id, rail, lease->generation,
                     RemoteRailOutcome::kSuccess);
  }
  if (!options.op || !options.op->busy_closed_rail) return;
  const size_t closed = *options.op->busy_closed_rail;
  options.op->busy_closed_rail.reset();
  // Without time left the confirmation is skipped, not counted as expired.
  if (closed != rail && RemainingMs(options.op, -1) != 0)
    ConfirmRefusal(node, health_id, closed, options.op->deadline_us);
}

// The peer closed `rail`'s bootstrap before replying and then took one on
// another rail. A server out of handshake slots, or stopping, does that too,
// so the rail is dialed once more; only if that dial is closed the same way
// does the peer refuse the device (a server that does not serve it). The
// rail is then charged as an endpoint failure and cools like one, with a
// recovery probe later, instead of every operation trying it and waiting
// out the busy pause. A peer's last usable rail is never cooled this way.
void RdmaTransport::ConfirmRefusal(const std::string& node,
                                   const std::string& health_id, size_t rail,
                                   uint64_t deadline_us) {
  const auto key = std::make_pair(health_id, rail);
  {
    std::lock_guard<std::mutex> lock(refusal_mu_);
    if (!refusal_checks_.insert(key).second) return;
  }
  AcquireOptions options;
  options.excluded.assign(devs_.size(), 1);
  options.excluded[rail] = 0;
  options.refusal_check = true;
  options.backpressure_us = 0;
  // Within the operation's deadline: each step waits at most
  // kRefusalCheckMs, and all of them together no longer than what is left.
  OpState confirm;
  confirm.deadline_us = deadline_us;
  options.op = &confirm;
  AcquireResult dialed = Acquire(node, Lane::kControl, options);
  bool refused = false;
  if (dialed.conn) {
    Release(node, Lane::kControl, dialed.conn, RemoteRailOutcome::kSuccess,
            std::nullopt);
  } else if (dialed.device_frame_closed) {
    refused = ChargeRefusal(node, health_id, rail);
  }
  if (!refused && dialed.device_frame_closed)
    DFKV_LOG_INFO("rdma: peer " + HealthIdLabel(health_id) +
                  " closed a bootstrap on dev=" + devs_[rail] +
                  " twice; not cooled (its last usable rail, or cooling)");
  std::lock_guard<std::mutex> lock(refusal_mu_);
  refusal_checks_.erase(key);
}

bool RdmaTransport::ChargeRefusal(const std::string& node,
                                  const std::string& health_id, size_t rail) {
  const auto peer = peer_topologies_->Snapshot(node);
  if (!peer) return false;
  // The peer's other rails this client can use now: compatible, enabled
  // here (port up), not locally quarantined.
  const auto candidates = topology_->CandidatesFor(-1, false);
  auto others = rdma::IntersectRailMasks(
      peer->peer_healthy,
      rdma::UnionRailMasks(candidates.allowed, candidates.fallback));
  others.resize(devs_.size(), 0);
  others[rail] = 0;
  const uint64_t now = rdma::RailPolicy::NowMicros();
  const auto local = rail_policy_->Snapshot(now);
  for (size_t i = 0; i < others.size() && i < local.size(); ++i)
    if (local[i].quarantined_until_us > now) others[i] = 0;
  // One charge at a time: two confirmations for two rails of one peer must
  // not each see the other rail usable and cool both.
  std::lock_guard<std::mutex> lock(refusal_mu_);
  if (!rdma::HasRail(remote_rail_health_->AllowedMask(health_id, others, now)))
    return false;
  const auto lease = remote_rail_health_->TryAcquire(health_id, rail, now);
  if (!lease) return false;
  rail_refusals_.fetch_add(1, std::memory_order_relaxed);
  CompleteRemote(health_id, rail, lease->generation,
                 RemoteRailOutcome::kEndpointFailure);
  return true;
}

RdmaTransport::AcquireResult RdmaTransport::Acquire(
    const std::string& node, Lane lane, const AcquireOptions& options) {
  AcquireResult result;
  // Out of the operation's deadline: nothing is dialed or waited for, and
  // neither the peer nor a rail is charged; the operation times out.
  if (RemainingMs(options.op, -1) == 0) {
    op_deadline_expired_.fetch_add(1, std::memory_order_relaxed);
    result.failure = AcquireFailure::kAdmission;
    result.status = Status::kIOError;
    return result;
  }
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
  auto candidates = topology_->CandidatesFor(caller_node, numa_aware_);
  if (preferred_rail_)
    candidates = rdma::PreferPrimaryRail(candidates, *preferred_rail_);
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
  const std::string health_id = HealthId(peer_snapshot->peer_id, node);
  std::optional<rdma::RailLease> lease;
  std::optional<RemoteRailLease> remote_lease;
  uint64_t lease_started = 0;
  size_t ridx = 0;
  for (size_t selection_attempt = 0; selection_attempt < devs_.size();
       ++selection_attempt) {
    const uint64_t now = rdma::RailPolicy::NowMicros();
    const auto remote_compatible = remote_rail_health_->AllowedMask(
        health_id, topology_compatible, now);
    const auto highest_tier =
        rdma::HighestCompatibleTier(rail_tiers_, remote_compatible);
    if (!rdma::HasRail(highest_tier)) break;
    lease = rdma::AcquireHighestTier(
        *rail_policy_, requested, AdmissionWait(options), highest_tier,
        candidates.allowed, candidates.fallback, admission_excluded);
    if (!lease) {
      result.failure = AcquireFailure::kAdmission;
      result.status = Status::kResourceExhausted;
      return result;
    }
    ridx = lease->rail;
    result.attempted_rail = ridx;
    lease_started = rdma::RailPolicy::NowMicros();
    // A peer the MDS does not name takes no remote rail lease: its rails
    // are cooled only for a refused device (NoteConnected), not for
    // ordinary endpoint failures, which it keeps retrying as before.
    // A confirming dial takes none either: it must not charge the rail it
    // checks for anything but a refusal (ChargeRefusal).
    if (peer_snapshot->peer_id.empty() || options.refusal_check) break;
    remote_lease = remote_rail_health_->TryAcquire(health_id, ridx, lease_started);
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
        if (completion == rdma::RailCompletion::kEndpointFailure)
          InvalidateCapabilities(node, peer_snapshot->peer_id,
                                 peer_snapshot->publication);
        if (remote_lease) {
          CompleteRemote(
              health_id, ridx, remote_lease->generation,
              completion == rdma::RailCompletion::kEndpointFailure
                  ? RemoteRailOutcome::kEndpointFailure
                  : RemoteRailOutcome::kAbandon);
        }
        const uint64_t now = rdma::RailPolicy::NowMicros();
        rail_policy_->Complete(*lease, now - lease_started, completion, now);
      };
  const size_t required_depth =
      ConnectionDepth(node, lane, options.requested_credits);
  const auto bound_for = [&](bool leased_put) {
    return lane == Lane::kControl
               ? static_cast<size_t>(rdma::kV2ControlCap)
               : ConnectionBound(
                     options.request_dynamic_pull
                         ? 0
                         : leased_put ? options.leased_inline_bytes
                                      : options.required_data_bytes);
  };

  std::vector<std::pair<void*, size_t>> pools;
  std::vector<Conn*> stale;
  std::vector<Conn*> session_reaped;
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
      auto& idle =
          lane == Lane::kControl ? control_pool_ : pool_;
      auto it = idle.find(node);
      if (it != idle.end()) {
        auto& pool_candidates = it->second;
        for (size_t i = pool_candidates.size(); i > 0; --i) {
          Conn* candidate = pool_candidates[i - 1];
          const bool republished =
              !current_peer ||
              candidate->peer_id != current_peer->peer_id ||
              candidate->peer_publication != current_peer->publication;
          if (republished && candidate->lifecycle.RequestRetire()) {
            stale.push_back(candidate);
            pool_candidates.erase(pool_candidates.begin() +
                                  static_cast<std::ptrdiff_t>(i - 1));
          }
        }
        // Only the connection chosen is checked for a closed session: one
        // peek per reuse, not one per pooled connection under mu_.
        for (;;) {
          size_t best = pool_candidates.size();
          size_t best_bound = std::numeric_limits<size_t>::max();
          size_t best_depth = std::numeric_limits<size_t>::max();
          for (size_t i = 0; i < pool_candidates.size(); ++i) {
            Conn* candidate = pool_candidates[i];
            const bool want_leased_put =
                options.request_leased_put && candidate->leased_put_supported;
            const size_t required_bound = bound_for(want_leased_put);
            if (candidate->peer_id == peer_snapshot->peer_id &&
                candidate->peer_publication == peer_snapshot->publication &&
                candidate->rail_index == ridx &&
                candidate->declared_bytes >= required_bound &&
                candidate->depth >= required_depth &&
                (!want_leased_put || candidate->leased_put) &&
                (candidate->declared_bytes < best_bound ||
                 (candidate->declared_bytes == best_bound &&
                  candidate->depth < best_depth))) {
              best = i;
              best_bound = candidate->declared_bytes;
              best_depth = candidate->depth;
            }
          }
          if (best == pool_candidates.size()) break;
          Conn* chosen = pool_candidates[best];
          // The server ended it and closed the session channel: drop it now,
          // not after the next operation waits out RC retries; pick again.
          if (SessionClosed(*chosen) && chosen->lifecycle.RequestRetire()) {
            session_reaped.push_back(chosen);
            pool_candidates.erase(pool_candidates.begin() +
                                  static_cast<std::ptrdiff_t>(best));
            continue;
          }
          if (chosen->lifecycle.Activate()) {
            pooled = chosen;
            MarkActive(pooled, lane);
            pool_candidates.erase(pool_candidates.begin() +
                                  static_cast<std::ptrdiff_t>(best));
          }
          break;
        }
      }
    }
  }
  for (Conn* conn : stale)
    Destroy(conn, rdma::RailCompletion::kAdmission);
  stale_publication_reaps_.fetch_add(stale.size(), std::memory_order_relaxed);
  // Ended by the server: neither the rail nor the peer is at fault.
  for (Conn* conn : session_reaped)
    Destroy(conn, rdma::RailCompletion::kAdmission);
  session_closed_reaps_.fetch_add(session_reaped.size(),
                                  std::memory_order_relaxed);
  if (!snapshot_current) {
    complete_unowned_lease(rdma::RailCompletion::kAdmission);
    result.failure = AcquireFailure::kAdmission;
    return result;
  }
  if (pooled) {
    pooled->lease = *lease;
    pooled->lease_started_us = lease_started;
    pooled->lease_bytes = 0;
    pooled->credit_held = true;
    pooled->visited = true;
    pooled->remote_lease_generation =
        remote_lease ? remote_lease->generation : 0;
    pooled->remote_lease_held = remote_lease.has_value();
    pooled->remote_recovery_probe =
        remote_lease && remote_lease->recovery_probe;
    pooled->reused_from_pool = true;
    pooled->completions_at_reuse = pooled->ep.succeeded_completions();
    result.from_pool = true;
    if (!pooled->ep.EnsurePoolMrs(pools, /*remote_write=*/false)) {
      Destroy(pooled, rdma::RailCompletion::kRailFailure);
      result.failure = AcquireFailure::kLocalRail;
      return result;
    }
    endpoint_cache_hits_.fetch_add(1, std::memory_order_relaxed);
    if (options.request_leased_put && !pooled->leased_put_supported)
      leaseput_path_fallbacks_.fetch_add(1, std::memory_order_relaxed);
    result.conn = pooled;
    result.failure = AcquireFailure::kNone;
    return result;
  }
  if (options.pooled_only) {
    complete_unowned_lease(rdma::RailCompletion::kAdmission);
    result.failure = AcquireFailure::kAdmission;
    result.status = Status::kResourceExhausted;
    return result;
  }
  // This process could not open an endpoint on the chosen rail for want of
  // descriptors or memory. The device has done nothing wrong and the peer
  // neither: no rail is charged, the peer is not cooled. Another rail can
  // still serve the operation: on a connection already open to the peer
  // (pooled_only: without a descriptor nothing new can be opened), or, for
  // ENOMEM, which may be this device's own limit on QPs or CQs, on a new one.
  auto other_rail = [&](bool pooled_only) -> std::optional<AcquireResult> {
    if (!options.local_resource_fallback) return std::nullopt;
    AcquireOptions other = options;
    other.excluded.resize(devs_.size(), 0);
    other.excluded[ridx] = 1;
    other.pooled_only = pooled_only;
    other.local_resource_fallback = false;
    other.peer = peer_snapshot;
    for (;;) {
      bool untried = false;
      for (size_t r = 0; r < devs_.size(); ++r)
        if (!other.excluded[r] && r < topology_compatible.size() &&
            topology_compatible[r])
          untried = true;
      if (!untried) return std::nullopt;
      AcquireResult alternative = Acquire(node, lane, other);
      if (alternative.conn) {
        local_resource_rail_fallbacks_.fetch_add(1, std::memory_order_relaxed);
        return alternative;
      }
      if (!alternative.attempted_rail) return std::nullopt;
      other.excluded[*alternative.attempted_rail] = 1;
    }
  };
  auto note_local_resource = [&](const char* what, int err) {
    local_resource_failures_.fetch_add(1, std::memory_order_relaxed);
    uint64_t suppressed = 0;
    if (local_resource_log_.Admit(
            LogThrottle::Key(what, static_cast<uint64_t>(err), 0),
            rdma::RailPolicy::NowMicros(), &suppressed))
      DFKV_LOG_WARN(
          std::string("rdma: cannot ") + what + " for " + node + " on " +
          devs_[ridx] + ": " + std::strerror(err) +
          ". This process is out of file descriptors or memory (or the "
          "device out of QPs or CQs); other rails are tried, then the "
          "operation fails with kResourceExhausted. No rail is charged "
          "and the peer is not cooled down. suppressed=" +
          std::to_string(suppressed));
  };

  // No idle connection on this rail, and its last endpoint open failed with
  // ENOMEM a moment ago: no new one is tried until kEnomemHoldUs has passed.
  // Each attempt would cost the peer a connection and a handshake it sees
  // abandoned. The rail's idle connections were usable above all the same.
  const bool enomem_held = rdma::RailPolicy::NowMicros() <
                           enomem_until_us_[ridx].load(std::memory_order_relaxed);
  if (enomem_held) {
    complete_unowned_lease(rdma::RailCompletion::kAdmission);
    if (auto alternative = other_rail(/*pooled_only=*/false))
      return *alternative;
    if (options.local_resource_fallback)
      local_resource_held_.fetch_add(1, std::memory_order_relaxed);
    result.failure = AcquireFailure::kAdmission;
    result.status = Status::kResourceExhausted;
    return result;
  }

  endpoint_cache_misses_.fetch_add(1, std::memory_order_relaxed);
  // A warm QP needs no probe: its own negotiation is authoritative. Sharing
  // observations with a NEW QP requires a stable identified publication.
  // Static peers have no restart/address-reuse signal, so always probe them.
  bool leased_put_supported = false;
  bool session_fd_supported = false;
  bool read_depth_supported = false;
  bool caps_cached = false;
  {
    std::lock_guard<std::mutex> lock(mu_);
    const auto found = peer_capabilities_.find(node);
    if (found != peer_capabilities_.end() &&
        found->second.peer_id == peer_snapshot->peer_id &&
        found->second.publication == peer_snapshot->publication) {
      if (options.force_new) {
        peer_capabilities_.erase(found);
      } else {
        caps_cached = true;
        leased_put_supported = found->second.leased_put;
        session_fd_supported = found->second.session_fd;
        read_depth_supported = found->second.read_depth;
      }
    }
  }
  // A close before any reply is taken as "busy" once: on the first attempt
  // of an operation. The retry that follows sets force_new, and a close on
  // that attempt is charged to the peer rail like any other endpoint failure.
  const bool first_attempt = !options.force_new;
  if (!caps_cached) {
    if (RecentlyUnreachable(node)) {
      complete_unowned_lease(rdma::RailCompletion::kAdmission);
      result.failure = AcquireFailure::kPeerUnreachable;
      return result;
    }
    // A dial cut short by the operation's deadline, or a confirming dial
    // with its short limit (ConfirmRefusal), says nothing about the node:
    // no unreachable hold and no charge from it.
    const int probe_connect_ms = options.refusal_check
                                     ? RemainingMs(options.op, kRefusalCheckMs)
                                     : RemainingMs(options.op, connect_ms_);
    const int probe_io_ms = options.refusal_check
                                ? RemainingMs(options.op, kRefusalCheckMs)
                                : RemainingMs(options.op, io_ms_);
    const bool probe_connect_cut =
        options.refusal_check || probe_connect_ms < connect_ms_;
    const bool probe_cut = probe_connect_cut || probe_io_ms < io_ms_;
    errno = 0;
    const ProbeResult probed = ProbeV2(
        node, &leased_put_supported, std::max(1, probe_connect_ms),
        std::max(1, probe_io_ms), &session_fd_supported,
        &read_depth_supported);
    const int probe_errno = errno;
    if (probed != ProbeResult::kSupported && probe_cut &&
        !options.refusal_check && RemainingMs(options.op, -1) == 0) {
      op_deadline_expired_.fetch_add(1, std::memory_order_relaxed);
      complete_unowned_lease(rdma::RailCompletion::kAdmission);
      result.failure = AcquireFailure::kAdmission;
      result.status = Status::kIOError;
      return result;
    }
    if (probed == ProbeResult::kUnreachable) {
      // A refused or timed-out connect, unless it timed out at a limit the
      // deadline cut short (a refusal is a refusal at any limit).
      if (!options.refusal_check &&
          !(probe_connect_cut && probe_errno == ETIMEDOUT))
        NoteUnreachable(node);
      complete_unowned_lease(rdma::RailCompletion::kAdmission);
      result.failure = AcquireFailure::kPeerUnreachable;
      return result;
    }
    if (probed == ProbeResult::kClosed && first_attempt) {
      if (!options.refusal_check)
        peer_busy_.fetch_add(1, std::memory_order_relaxed);
      complete_unowned_lease(rdma::RailCompletion::kAdmission);
      result.failure = AcquireFailure::kPeerBusy;
      return result;
    }
    if (probed != ProbeResult::kSupported) {
      complete_unowned_lease(rdma::RailCompletion::kEndpointFailure);
      result.failure = AcquireFailure::kEndpoint;
      return result;
    }
  }
  const bool want_leased_put =
      options.request_leased_put && leased_put_supported;
  if (options.request_leased_put && !want_leased_put)
    leaseput_path_fallbacks_.fetch_add(1, std::memory_order_relaxed);
  const size_t required_bound = bound_for(want_leased_put);
  // Open, announce, and budget only the size class this operation needs.
  const size_t conn_depth = required_depth;
  const uint64_t conn_declared =
      lane == Lane::kControl
          ? static_cast<uint64_t>(rdma::kV2ControlCap)
          : static_cast<uint64_t>(required_bound);
  const uint64_t conn_slot_bytes = static_cast<uint64_t>(rdma::V2SlotSize(
      static_cast<size_t>(
          std::max<uint64_t>(conn_declared, rdma::kV2DataOffset))));
  const uint64_t resident_bytes = conn_slot_bytes;
  const rdma::ResourceRequest budget_request{
      1, 1, conn_depth, resident_bytes * conn_depth};
  bool budget_acquired = resource_budget_->TryAcquire(budget_request);
  if (!budget_acquired && EvictOneIdle())
    budget_acquired = resource_budget_->TryAcquire(budget_request);
  if (!budget_acquired)
    budget_acquired = resource_budget_->Acquire(
        budget_request,
        std::chrono::milliseconds(RemainingMs(options.op, resource_acquire_ms_)));
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
  const bool held = RecentlyUnreachable(node);
  const int dial_connect_ms = options.refusal_check
                                  ? RemainingMs(options.op, kRefusalCheckMs)
                                  : RemainingMs(options.op, connect_ms_);
  const int dial_io_ms = options.refusal_check
                             ? RemainingMs(options.op, kRefusalCheckMs)
                             : RemainingMs(options.op, io_ms_);
  // Cut short by the operation's deadline: a connect that times out then
  // says nothing about the node.
  const bool dial_connect_cut =
      !options.refusal_check && dial_connect_ms < connect_ms_;
  int fd = held ? -1
                : DialNode(node, std::max(1, dial_connect_ms),
                           std::max(1, dial_io_ms));
  const int dial_errno = errno;
  if (fd < 0 && !held && (dial_errno == EMFILE || dial_errno == ENFILE)) {
    // socket() itself failed: nothing reached the peer.
    note_local_resource("open a socket", dial_errno);
    complete_unowned_lease(rdma::RailCompletion::kAdmission);
    resource_budget_->Release(budget_request);
    if (auto alternative = other_rail(/*pooled_only=*/true))
      return *alternative;
    result.failure = AcquireFailure::kAdmission;
    result.status = Status::kResourceExhausted;
    return result;
  }
  if (fd < 0) {
    // The bootstrap address is the node's, the same for every rail. A
    // refusal is not a connect that failed: it starts nothing and is not
    // recorded as one (see TakeUnreachableRefusal).
    if (!held && !options.refusal_check &&
        !(dial_connect_cut && dial_errno == ETIMEDOUT))
      NoteUnreachable(node);
    complete_unowned_lease(rdma::RailCompletion::kAdmission);
    resource_budget_->Release(budget_request);
    result.failure = AcquireFailure::kPeerUnreachable;
    return result;
  }
  auto* conn = new Conn();
  conn->lane = lane;
  conn->rail_index = ridx;
  conn->peer_publication = peer_snapshot->publication;
  conn->peer_id = peer_snapshot->peer_id;
  conn->health_id = health_id;
  conn->node = node;
  conn->metric_peer =
      peer_snapshot->peer_id.empty() ? node : peer_snapshot->peer_id;
  conn->remote_lease_generation =
      remote_lease ? remote_lease->generation : 0;
  conn->remote_lease_held = remote_lease.has_value();
  conn->declared_bytes = static_cast<size_t>(conn_declared);
  conn->depth = conn_depth;
  conn->remote_recovery_probe =
      remote_lease && remote_lease->recovery_probe;
  conn->lease = *lease;
  conn->lease_started_us = lease_started;
  conn->credit_held = true;
  conn->budget_request = budget_request;
  conn->budget_held = true;
  const int injected_errno = ridx < fail_open_errno_for_test_.size()
                                 ? fail_open_errno_for_test_[ridx]
                                 : 0;
  conn->ep.set_remote_access(false);  // a client QP is never an RDMA target
  if (injected_errno != 0 ||
      !conn->ep.Open(dev.c_str(), rdma::kV2ControlCap, conn_depth)) {
    ::close(fd);
    const int open_errno =
        injected_errno != 0 ? injected_errno : conn->ep.open_errno();
    if (open_errno == EMFILE || open_errno == ENFILE ||
        open_errno == ENOMEM) {
      note_local_resource("create an endpoint", open_errno);
      if (open_errno == ENOMEM)
        enomem_until_us_[ridx].store(
            rdma::RailPolicy::NowMicros() + kEnomemHoldUs,
            std::memory_order_relaxed);
      Destroy(conn, rdma::RailCompletion::kAdmission);
      if (auto alternative = other_rail(open_errno != ENOMEM))
        return *alternative;
      result.failure = AcquireFailure::kAdmission;
      result.status = Status::kResourceExhausted;
      return result;
    }
    DFKV_LOG_WARN("rdma: device " + dev +
                  " Open failed; rail failure policy will quarantine it");
    if (open_errno == ENODEV || open_errno == EIO) {
      // The device is gone or was reset: stop holding its context, so it
      // closes once this rail's endpoints are gone and a later Open opens
      // the device again instead of reusing the dead context.
      {
        std::lock_guard<std::mutex> lock(rail_devices_mu_);
        if (ridx < rail_devices_.size()) rail_devices_[ridx].reset();
      }
      device_suspects_[ridx * 2 + 1].fetch_add(1, std::memory_order_relaxed);
      // TRN-29: the binding the rail was built on is retired (if no one did
      // yet), and the background thread rebuilds the rail.
      if (device_rebuild_ &&
          rdma::RetireDevice(dev, rail_generation_[ridx].load(
                                      std::memory_order_acquire)))
        keepalive_cv_.notify_all();
    }
    Destroy(conn, rdma::RailCompletion::kRailFailure);
    result.failure = AcquireFailure::kLocalRail;
    return result;
  }
  // TRN-29: an Open can land on a context newer than the one this rail was
  // rebuilt on (another transport saw the reset first, and this one's
  // background thread has not yet). Such a connection would have no
  // registration of this transport's memory: it is dropped, the rail leaves
  // selection until rebuilt, and the operation goes to another rail.
  if (device_rebuild_ &&
      conn->ep.device_generation() !=
          rail_generation_[ridx].load(std::memory_order_acquire)) {
    ::close(fd);
    Destroy(conn, rdma::RailCompletion::kAdmission);
    rail_stale_[ridx].store(1, std::memory_order_release);
    topology_->SetDeviceEnabled(ridx, false);
    keepalive_cv_.notify_all();
    result.failure = AcquireFailure::kLocalRail;
    return result;
  }
  {
    std::lock_guard<std::mutex> lock(rail_devices_mu_);
    if (ridx < rail_devices_.size() && !rail_devices_[ridx])
      rail_devices_[ridx] =
          std::make_unique<rdma::DeviceRef>(conn->ep.ReferenceDevice());
  }
  if (busy_poll_) conn->ep.set_busy_poll(true);
  conn->ep.set_spin_us(spin_us_);

  char devbuf[rdma::kDevNameBytes];
  // Probe requires the dynamic-only contract before a QP bootstrap. There is
  // no legacy handshake fallback and no responder writer token to consume.
  // The session channel only where the probe said the server keeps one:
  // a server that predates it would take the bit for part of the size.
  const bool want_session = session_fd_enabled_ && session_fd_supported;
  // Pipelined READs are asked for only where the probe offered them, and
  // used only once the readiness confirms them: a capability cached from
  // before the server was replaced by an older one then costs nothing.
  const bool want_read_depth = pipelined_reads_ && read_depth_supported;
  const uint64_t bootstrap_declared =
      conn_declared | rdma::kDevFrameRequestDynamicOnly |
      rdma::kDevFrameRequestPullRead | rdma::kDevFrameRequestDynamicPull |
      (want_leased_put ? rdma::kDevFrameRequestLeasedPut : 0) |
      (want_session ? rdma::kDevFrameRequestSessionFd : 0) |
      (want_read_depth ? rdma::kDevFrameRequestReadDepth : 0);
  rdma::EncodeDevFrame(auto_device_ ? std::string() : dev,
                       bootstrap_declared, devbuf, rdma::kDevProtoV2);
  char mine[rdma::kQpInfoBytes], remote_qp_bytes[rdma::kQpInfoBytes];
  rdma::QpInfo my = conn->ep.Local();
  my.depth = static_cast<uint16_t>(std::min<size_t>(conn_depth, 256));
  my.protocol_version = rdma::kDevProtoV2;
  rdma::SerializeQpInfo(my, mine);
  // Each bootstrap read gets one deadline for all its bytes: what is left of
  // the operation, at most DFKV_RDMA_IO_MS (a confirming dial: its own short
  // limit). A read that times out at a limit the operation's deadline cut
  // short says nothing about the peer: nothing is charged for it.
  bool read_cut = false;
  bool read_timed_out = false;
  const auto read_phase = [&](void* buffer, size_t bytes) {
    const int limit_ms = options.refusal_check
                             ? RemainingMs(options.op, kRefusalCheckMs)
                             : RemainingMs(options.op, io_ms_);
    read_cut = !options.refusal_check && limit_ms < io_ms_;
    const auto until = std::chrono::steady_clock::now() +
                       std::chrono::milliseconds(std::max(1, limit_ms));
    errno = 0;
    const bool ok = net::ReadAllWithin(fd, buffer, bytes, timeval{0, 0}, until);
    read_timed_out = !ok && (SocketTimedOut(errno) ||
                             std::chrono::steady_clock::now() >= until);
    return ok;
  };
  const auto deadline_expired = [&]() {
    op_deadline_expired_.fetch_add(1, std::memory_order_relaxed);
    Destroy(conn, rdma::RailCompletion::kAdmission);
    result.failure = AcquireFailure::kAdmission;
    result.status = Status::kIOError;
    return result;
  };
  errno = 0;
  if (!net::WriteAll(fd, devbuf, rdma::kDevNameBytes) ||
      !net::WriteAll(fd, mine, rdma::kQpInfoBytes) ||
      !read_phase(remote_qp_bytes, rdma::kQpInfoBytes)) {
    const bool closed = !read_timed_out && !SocketTimedOut(errno);
    ::close(fd);
    if (closed && first_attempt) {
      // Closed before the peer sent anything. A server at its limit of
      // pending handshakes does this, and so does one that rejects the
      // device; the two look the same from here. Take it as busy once.
      if (!options.refusal_check)
        peer_busy_.fetch_add(1, std::memory_order_relaxed);
      // So does an older server replaced under the same identity and
      // publication, refusing a request bit (session channel, READ depth)
      // that cached capabilities still offer: when this frame carried one,
      // probe it afresh next time. An operation that does not retry its
      // acquire (a whole-object GET) would otherwise be refused on every new
      // connection.
      if (want_session || want_read_depth)
        InvalidateCapabilities(node, peer_snapshot->peer_id,
                               peer_snapshot->publication);
      if (options.op) options.op->busy_closed_rail = ridx;
      result.device_frame_closed = true;
      Destroy(conn, rdma::RailCompletion::kAdmission);
      result.failure = AcquireFailure::kPeerBusy;
      return result;
    }
    // The operation ran out of time; a server under receive-pool pressure
    // may hold its reply for up to 5 s, so this is no evidence against the
    // rail. A read that waited the full DFKV_RDMA_IO_MS is.
    if (!closed && read_cut) return deadline_expired();
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
  // To RTR now; the step to RTS, which fixes the READ depth, waits for the
  // readiness below.
  if (!conn->ep.ConnectRtr(remote)) {
    DFKV_LOG_WARN("rdma: device " + dev +
                  " failed local QP transition during v2 negotiation");
    ::close(fd);
    Destroy(conn, rdma::RailCompletion::kRailFailure);
    result.failure = AcquireFailure::kLocalRail;
    return result;
  }
  conn->ep.set_remote_depth(remote.depth);
  conn->depth = std::min<size_t>(conn_depth, remote.depth);
  if (remote.depth < conn_depth) {
    NoteNegotiatedDepth(node, lane, remote.depth);
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
        resident_bytes * remote.depth};
    resource_budget_->Release(
        rdma::ResourceRequest{0, 0, surplus,
                              resident_bytes * surplus});
    depth_refunds_.fetch_add(1, std::memory_order_relaxed);
  }

  char readiness[rdma::kV2DynamicOnlyReadinessBytes];
  if (!read_phase(readiness, sizeof(readiness))) {
    ::close(fd);
    if (read_timed_out && read_cut) return deadline_expired();
    DFKV_LOG_ERROR("rdma: peer " + node +
                   " did not send its v2 readiness");
    Destroy(conn, rdma::RailCompletion::kEndpointFailure);
    result.failure = AcquireFailure::kEndpoint;
    return result;
  }
  bool session_kept = false;
  bool read_depth_acked = false;
  if (!rdma::DecodeV2DynamicOnlyReadiness(
          readiness, sizeof(readiness), &conn->recv_segment,
          want_session ? &session_kept : nullptr,
          want_read_depth ? &read_depth_acked : nullptr)) {
    DFKV_LOG_ERROR(
        "rdma: peer " + node +
        " did not provide valid negotiated v2 readiness");
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
  // Pipelined READs only on the server's word that this QP accepts them:
  // an older server (responder depth 1) would end it on the second READ.
  if (read_depth_acked) {
    conn->ep.set_read_depth(rdma::kV2ReadDepth);
    pipelined_read_conns_.fetch_add(1, std::memory_order_relaxed);
  }
  if (!conn->ep.ConnectRts()) {
    DFKV_LOG_WARN("rdma: device " + dev +
                  " failed local QP transition during v2 negotiation");
    ::close(fd);
    Destroy(conn, rdma::RailCompletion::kRailFailure);
    result.failure = AcquireFailure::kLocalRail;
    return result;
  }
  if (session_kept) {
    // Kept for the connection's life. net::Dial made it close-on-exec; a
    // plain fork() child still holds a copy until it exits, and while it
    // does the server sees no close and falls back to its idle timeout.
    conn->session_fd = fd;
  } else {
    ::close(fd);
  }
  if (!conn->ep.EnsurePoolMrs(pools, /*remote_write=*/false)) {
    Destroy(conn, rdma::RailCompletion::kRailFailure);
    result.failure = AcquireFailure::kLocalRail;
    return result;
  }
  bool publication_current = false;
  {
    std::lock_guard<std::mutex> lock(mu_);
    publication_current = peer_topologies_->IsCurrent(
        node, conn->peer_id, conn->peer_publication);
    // Never publish a bare probe: only a completed bootstrap proves these
    // capabilities usable, and a late bootstrap cannot republish an old epoch.
    if (!caps_cached && publication_current && !conn->peer_id.empty() &&
        conn->peer_publication != 0) {
      peer_capabilities_[node] = {
          conn->peer_id, conn->peer_publication,
          leased_put_supported, session_fd_supported, read_depth_supported};
    }
  }
  if (!publication_current) {
    stale_publication_reaps_.fetch_add(1, std::memory_order_relaxed);
    Destroy(conn, rdma::RailCompletion::kAdmission);
    result.failure = AcquireFailure::kAdmission;
    return result;
  }
  conn->leased_put = want_leased_put;
  conn->leased_put_supported = leased_put_supported;
  conns_opened_.fetch_add(1, std::memory_order_relaxed);
  MarkClassOpened(conn);
  rail_conns_[ridx].fetch_add(1, std::memory_order_relaxed);
  MarkLive(conn);
  MarkActive(conn, lane);
  NoteConnected(options, node, health_id, ridx);
  // The rail's latency sample is the operation's, not the connection setup's.
  conn->lease_started_us = rdma::RailPolicy::NowMicros();
  conn->lease_bytes = 0;
  result.conn = conn;
  result.failure = AcquireFailure::kNone;
  return result;
}

// A PUT that was posted is not sent again: whether the server stored it is
// not known. Callers pass request_posted false for a PUT whose connection
// Destroy found to be a stale pooled one. Nothing succeeded on that
// connection since it left the pool, not even the acknowledgement of the
// request by the peer's NIC, which a peer that still has the connection
// sends as the request arrives. So the request was almost certainly not
// delivered: the server has reclaimed the connection or restarted. It may
// have been, if only the acknowledgements were lost on the way back, or if
// the completion of the request arrived just as the window's deadline ran
// out (the reap does not poll once more then). The PUT sent again carries
// the same key and bytes, which the server answers with the result of the
// value it has. A first copy that was delivered may still be handled after
// the second, as any PUT whose client gave up on it could be before.
bool RdmaTransport::PrepareRetry(
    int attempt, bool from_pool, std::optional<size_t> attempted_rail,
    AcquireFailure failure, ReplaySafety replay_safety, bool request_posted,
    const std::shared_ptr<const rdma::PeerRailSnapshot>& peer,
    const std::string& node, RailMask* excluded, bool* cross_rail_retry) {
  const std::string health_id = HealthId(peer->peer_id, node);
  if (attempt != 0 || failure == AcquireFailure::kNoCompatibleRail ||
      failure == AcquireFailure::kPeerUnreachable ||
      (replay_safety == ReplaySafety::kUnsafeAfterPost && request_posted))
    return false;
  const bool peer_busy = failure == AcquireFailure::kPeerBusy;
  if (peer_busy && peer_busy_retry_us_ != 0) {
    // Handshake slots at the peer free up as other clients finish theirs.
    // The pause differs per caller so that the callers refused together do
    // not come back together.
    const uint64_t jitter_us =
        (rdma::RailPolicy::NowMicros() ^
         std::hash<std::thread::id>()(std::this_thread::get_id())) %
        peer_busy_retry_us_;
    std::this_thread::sleep_for(
        std::chrono::microseconds(peer_busy_retry_us_ + jitter_us));
  }
  if (failure == AcquireFailure::kLocalRail && attempted_rail &&
      *attempted_rail < devs_.size()) {
    excluded->assign(devs_.size(), 0);
    (*excluded)[*attempted_rail] = 1;
    const int caller_node = numa_aware_ ? numa::CurrentNode() : -1;
    auto candidates = topology_->CandidatesFor(caller_node, numa_aware_);
    if (preferred_rail_)
      candidates = rdma::PreferPrimaryRail(candidates, *preferred_rail_);
    const auto local_enabled =
        rdma::UnionRailMasks(candidates.allowed, candidates.fallback);
    auto compatible =
        rdma::IntersectRailMasks(peer->peer_healthy, local_enabled);
    compatible = remote_rail_health_->AllowedMask(
        health_id, compatible, rdma::RailPolicy::NowMicros());
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
  // A busy peer is retried on the rails an endpoint failure would be retried
  // on: another compatible rail when there is one, since a peer that rejects
  // this device closes the connection in the same way as a full one, and the
  // same rail otherwise.
  if ((failure == AcquireFailure::kEndpoint || peer_busy) && attempted_rail &&
      *attempted_rail < devs_.size()) {
    excluded->assign(devs_.size(), 0);
    (*excluded)[*attempted_rail] = 1;
    const int caller_node = numa_aware_ ? numa::CurrentNode() : -1;
    auto candidates = topology_->CandidatesFor(caller_node, numa_aware_);
    if (preferred_rail_)
      candidates = rdma::PreferPrimaryRail(candidates, *preferred_rail_);
    const auto local_enabled =
        rdma::UnionRailMasks(candidates.allowed, candidates.fallback);
    auto compatible =
        rdma::IntersectRailMasks(peer->peer_healthy, local_enabled);
    compatible = remote_rail_health_->AllowedMask(
        health_id, compatible, rdma::RailPolicy::NowMicros());
    const auto highest_tier =
        rdma::HighestCompatibleTier(rail_tiers_, compatible);
    const auto preferred =
        rdma::IntersectRailMasks(highest_tier, candidates.allowed);
    const auto fallback =
        rdma::IntersectRailMasks(highest_tier, candidates.fallback);
    if (!rdma::HasUnexcludedRail(preferred, fallback, *excluded)) {
      if (!from_pool && !peer_busy) return false;
      excluded->assign(devs_.size(), 0);
      if (from_pool)
        stale_pool_retries_.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
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

  // One declaration at a time. The registrations themselves (anchors opened,
  // every rail's region pinned: hundreds of ms for a GiB) run without mu_,
  // so Acquire and Release on this transport go on with the published
  // generation meanwhile; mu_ is taken only to commit and publish.
  std::lock_guard<std::mutex> declaring(register_mu_);
  std::vector<std::pair<void*, size_t>> candidate;
  {
    std::lock_guard<std::mutex> lk(mu_);
    candidate = pools_;
  }
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
  // anchors_ is only used here, under register_mu_.
  // A rail whose device was reset (TRN-29) is skipped, its slot left
  // empty: its rebuild registers every published region on the new context.
  std::vector<std::unique_ptr<rdma::RcEndpoint>> new_anchors;
  std::vector<rdma::RcEndpoint*> rails;
  if (anchors_.empty()) {
    new_anchors.reserve(devs_.size());
    for (size_t rail = 0; rail < devs_.size(); ++rail) {
      const std::string& device = devs_[rail];
      if (RailStale(rail)) {
        rails.push_back(nullptr);
        new_anchors.push_back(nullptr);
        continue;
      }
      auto endpoint = std::make_unique<rdma::RcEndpoint>();
      endpoint->set_remote_access(false);
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
    for (size_t rail = 0; rail < anchors_.size(); ++rail)
      rails.push_back(RailStale(rail) ? nullptr : anchors_[rail].get());
  }

  // Stage every rail, one after another (registering rails at once was
  // slower: the pinning of the same pages contends). A failure rolls back
  // what was staged; nothing was published.
  std::vector<rdma::RcEndpoint::PoolMrRegistration> registrations(rails.size());
  size_t staged = 0;
  for (; staged < rails.size(); ++staged) {
    // Local access only: since v2.31 servers never WRITE into client
    // memory (GET is a client READ of server staging).
    if (rails[staged] &&
        !rails[staged]->StagePoolMr(base, size, /*remote_write=*/false,
                                    &registrations[staged]))
      break;
  }
  if (staged != rails.size()) {
    while (staged != 0) {
      --staged;
      if (rails[staged]) rails[staged]->RollbackPoolMr(&registrations[staged]);
    }
    mr_registration_rejections_.fetch_add(1, std::memory_order_relaxed);
    DFKV_LOG_ERROR(
        "rdma: explicit memory registration rejected for base=" +
        std::to_string(address) + " size=" + std::to_string(size));
    return false;
  }

  // Superseded generations no endpoint references any more: deregistered
  // after mu_ (tens of ms for a GiB), while the anchors keep the devices;
  // by the holder's destructor if the commit throws.
  rdma::RetiredPoolMrs retired;
  std::vector<Conn*> unusable;
  {
    // Commit and publish together: Acquire sees the complete old generation
    // or the complete new one, never a part of it.
    std::lock_guard<std::mutex> lk(mu_);
    // Room for every idle connection first: one taken out of its pool must
    // not be lost to a failed allocation.
    size_t idle_count = 0;
    for (const auto* idle : {&pool_, &control_pool_})
      for (const auto& entry : *idle) idle_count += entry.second.size();
    unusable.reserve(idle_count);
    for (size_t rail = 0; rail < rails.size(); ++rail)
      if (rails[rail])
        rails[rail]->CommitPoolMr(&registrations[rail], retired.get());
    if (!new_anchors.empty()) anchors_ = std::move(new_anchors);
    // Idle pooled endpoints have no in-flight WRs, so they can safely adopt
    // the new generation now and release their superseded same-base cache
    // reference. An endpoint that cannot resolve the already-anchored
    // declaration is discarded; a future Acquire recreates it from the
    // published generation.
    const auto refresh_idle = [&](auto& idle) {
      for (auto& [node, connections] : idle) {
        (void)node;
        for (auto it = connections.begin(); it != connections.end();) {
          Conn* connection = *it;
          if (connection->ep.EnsurePoolMrs(candidate,
                                           /*remote_write=*/false,
                                           retired.get())) {
            ++it;
            continue;
          }
          it = connections.erase(it);
          unusable.push_back(connection);  // reserved: cannot throw
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
  }
  retired.Deregister();
  for (Conn* connection : unusable) Destroy(connection);
  if (grew) {
    DFKV_LOG_INFO(
        "rdma: explicit memory registration grew for base=" +
        std::to_string(address) + " size=" + std::to_string(size));
  }
  return true;
}

std::vector<size_t> RdmaTransport::IdleDataBounds(
    const std::string& node) const {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<size_t> bounds;
  const auto found = pool_.find(node);
  if (found == pool_.end()) return bounds;
  bounds.reserve(found->second.size());
  for (const Conn* conn : found->second)
    bounds.push_back(conn->declared_bytes);
  std::sort(bounds.begin(), bounds.end());
  return bounds;
}

std::vector<size_t> RdmaTransport::IdleDataDepths(
    const std::string& node) const {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<size_t> depths;
  const auto found = pool_.find(node);
  if (found == pool_.end()) return depths;
  depths.reserve(found->second.size());
  for (const Conn* conn : found->second) depths.push_back(conn->depth);
  std::sort(depths.begin(), depths.end());
  return depths;
}

std::string RdmaTransport::MetricsText() const {
  std::string s;
  s += "# HELP dfkv_rdma_client_conns_opened_total RDMA client connections opened\n";
  s += "# TYPE dfkv_rdma_client_conns_opened_total counter\n";
  s += "dfkv_rdma_client_conns_opened_total " +
       std::to_string(conns_opened_.load(std::memory_order_relaxed)) + "\n";
  std::map<ConnectionClass, ConnectionClassStats> class_stats;
  {
    std::lock_guard<std::mutex> lock(connection_class_mu_);
    class_stats = connection_class_stats_;
  }
  std::map<ConnectionClass, uint64_t> idle_classes;
  {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& [node, connections] : pool_) {
      (void)node;
      for (const Conn* conn : connections)
        ++idle_classes[{conn->declared_bytes, conn->depth}];
    }
  }
  s += "# HELP dfkv_rdma_client_connection_class_opened_total Data QPs opened by adaptive block/depth class\n";
  s += "# TYPE dfkv_rdma_client_connection_class_opened_total counter\n";
  s += "# HELP dfkv_rdma_client_connection_class_active Data QPs currently executing by adaptive block/depth class\n";
  s += "# TYPE dfkv_rdma_client_connection_class_active gauge\n";
  for (const auto& [geometry, stats] : class_stats) {
    const std::string labels =
        "{block_bytes=\"" + std::to_string(geometry.first) +
        "\",depth=\"" + std::to_string(geometry.second) + "\"} ";
    s += "dfkv_rdma_client_connection_class_opened_total" + labels +
         std::to_string(stats.opened) + "\n";
    s += "dfkv_rdma_client_connection_class_active" + labels +
         std::to_string(stats.active) + "\n";
  }
  s += "# HELP dfkv_rdma_client_connection_class_idle Data QPs currently idle by adaptive block/depth class\n";
  s += "# TYPE dfkv_rdma_client_connection_class_idle gauge\n";
  for (const auto& [geometry, idle] : idle_classes) {
    s += "dfkv_rdma_client_connection_class_idle{block_bytes=\"" +
         std::to_string(geometry.first) + "\",depth=\"" +
         std::to_string(geometry.second) + "\"} " + std::to_string(idle) +
         "\n";
  }
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
                 "Process-wide remote resident-slot byte admission budget",
                 resource_used.registered_bytes,
                 resource_limit.registered_bytes);
  std::vector<std::vector<uint64_t>> idle_by_lane_rail(
      3, std::vector<uint64_t>(devs_.size(), 0));
  {
    std::lock_guard<std::mutex> lock(mu_);
    const auto count = [&](const auto& pools) {
      for (const auto& [node, connections] : pools) {
        (void)node;
        for (const Conn* conn : connections) {
          const size_t lane =
              conn->lane == Lane::kData
                  ? 0
                  : (conn->lane == Lane::kSgData ? 1 : 2);
          if (conn->rail_index < devs_.size())
            ++idle_by_lane_rail[lane][conn->rail_index];
        }
      }
    };
    count(pool_);
    count(control_pool_);
  }
  s += "# HELP dfkv_rdma_client_pool_connections Endpoint ownership by operation lane, state, and rail\n";
  s += "# TYPE dfkv_rdma_client_pool_connections gauge\n";
  constexpr const char* lane_names[] = {"payload", "sg", "control"};
  for (size_t lane = 0; lane < idle_by_lane_rail.size(); ++lane) {
    for (size_t rail = 0; rail < devs_.size(); ++rail) {
      const std::string dev =
          devs_[rail].empty() ? std::string("default") : devs_[rail];
      const std::string labels =
          "{lane=\"" + std::string(lane_names[lane]) + "\",state=\"";
      s += "dfkv_rdma_client_pool_connections" + labels + "idle\",dev=\"" +
           dev + "\"} " +
           std::to_string(idle_by_lane_rail[lane][rail]) + "\n";
      s += "dfkv_rdma_client_pool_connections" + labels + "active\",dev=\"" +
           dev + "\"} " +
           std::to_string(
               active_lane_rail_[lane * devs_.size() + rail].load(
                   std::memory_order_relaxed)) +
           "\n";
    }
  }
  s += "# HELP dfkv_rdma_client_peer_connections Live endpoints by peer and local rail\n";
  s += "# TYPE dfkv_rdma_client_peer_connections gauge\n";
  {
    std::lock_guard<std::mutex> lock(peer_connections_mu_);
    for (const auto& [key, count] : peer_connections_) {
      const std::string dev =
          devs_[key.second].empty() ? std::string("default")
                                   : devs_[key.second];
      s += "dfkv_rdma_client_peer_connections{peer=\"" +
           PromLabelEscape(key.first) + "\",dev=\"" + PromLabelEscape(dev) +
           "\"} " + std::to_string(count) + "\n";
    }
  }
  s += "# HELP dfkv_rdma_client_pool_limit Idle endpoints retained per peer and pool\n";
  s += "# TYPE dfkv_rdma_client_pool_limit gauge\n";
  s += "dfkv_rdma_client_pool_limit " + std::to_string(pool_max_) + "\n";
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
  s += "# HELP dfkv_rdma_client_rail_put_ops_total Successful logical PUT objects per local RDMA rail\n";
  s += "# TYPE dfkv_rdma_client_rail_put_ops_total counter\n";
  s += "# HELP dfkv_rdma_client_rail_put_bytes_total Successful logical PUT payload bytes per local RDMA rail\n";
  s += "# TYPE dfkv_rdma_client_rail_put_bytes_total counter\n";
  s += "# HELP dfkv_rdma_client_rail_get_ops_total Successful logical GET objects per local RDMA rail\n";
  s += "# TYPE dfkv_rdma_client_rail_get_ops_total counter\n";
  s += "# HELP dfkv_rdma_client_rail_get_bytes_total Successful logical GET payload bytes per local RDMA rail\n";
  s += "# TYPE dfkv_rdma_client_rail_get_bytes_total counter\n";
  for (size_t i = 0; i < devs_.size(); ++i) {
    const std::string& d = devs_[i].empty() ? std::string("default") : devs_[i];
    s += "dfkv_rdma_client_rail_conns_total{dev=\"" + d + "\"} " +
         std::to_string(rail_conns_[i].load(std::memory_order_relaxed)) + "\n";
    const std::string labels = "{dev=\"" + d + "\"} ";
    s += "dfkv_rdma_client_rail_put_ops_total" + labels +
         std::to_string(rail_put_ops_[i].load(std::memory_order_relaxed)) + "\n";
    s += "dfkv_rdma_client_rail_put_bytes_total" + labels +
         std::to_string(rail_put_bytes_[i].load(std::memory_order_relaxed)) + "\n";
    s += "dfkv_rdma_client_rail_get_ops_total" + labels +
         std::to_string(rail_get_ops_[i].load(std::memory_order_relaxed)) + "\n";
    s += "dfkv_rdma_client_rail_get_bytes_total" + labels +
         std::to_string(rail_get_bytes_[i].load(std::memory_order_relaxed)) + "\n";
  }
  // Lazily populated: a series appears with its first failure, so the label
  // set stays at (rails x statuses that actually occurred).
  s += "# HELP dfkv_rdma_client_rail_wc_errors_total Connections lost to a failed work completion per local RDMA rail and completion status\n";
  s += "# TYPE dfkv_rdma_client_rail_wc_errors_total counter\n";
  for (size_t i = 0; i < devs_.size(); ++i) {
    for (size_t slot = 0; slot < rdma::kWcStatusSlots; ++slot) {
      const uint64_t count =
          rail_wc_errors_[i * rdma::kWcStatusSlots + slot].load(
              std::memory_order_relaxed);
      if (count == 0) continue;
      s += "dfkv_rdma_client_rail_wc_errors_total{dev=\"" +
           PromLabelEscape(devs_[i].empty() ? std::string("default")
                                            : devs_[i]) +
           "\",status=\"" + rdma::WcStatusName(static_cast<int>(slot)) +
           "\"} " + std::to_string(count) + "\n";
    }
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
  s += "# HELP dfkv_rdma_client_declared_max_block_bytes Logical RDMA block safety ceiling\n";
  s += "# TYPE dfkv_rdma_client_declared_max_block_bytes gauge\n";
  s += "dfkv_rdma_client_declared_max_block_bytes " +
       std::to_string(declared_) + "\n";
  s += "# HELP dfkv_rdma_client_connection_min_block_bytes Smallest adaptive data-connection declaration\n";
  s += "# TYPE dfkv_rdma_client_connection_min_block_bytes gauge\n";
  s += "dfkv_rdma_client_connection_min_block_bytes " +
       std::to_string(connection_min_block_bytes_) + "\n";
  s += "# HELP dfkv_rdma_client_oversize_rejects_total Operations rejected above the declared maximum block\n";
  s += "# TYPE dfkv_rdma_client_oversize_rejects_total counter\n";
  s += "dfkv_rdma_client_oversize_rejects_total " +
       std::to_string(oversize_rejects_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_inline_put_max_bytes Leased-PUT inline threshold (env DFKV_RDMA_INLINE_PUT_MAX_BYTES); objects above it use per-op staging leases\n";
  s += "# TYPE dfkv_rdma_client_inline_put_max_bytes gauge\n";
  s += "dfkv_rdma_client_inline_put_max_bytes " +
       std::to_string(inline_put_max_bytes_) + "\n";
  s += "# HELP dfkv_rdma_client_leaseput_ops_total Objects delivered through per-op staging leases\n";
  s += "# TYPE dfkv_rdma_client_leaseput_ops_total counter\n";
  s += "dfkv_rdma_client_leaseput_ops_total " +
       std::to_string(leaseput_ops_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_leaseput_path_fallbacks_total Lease-intent batches that fell back to a traditional largest-object connection\n";
  s += "# TYPE dfkv_rdma_client_leaseput_path_fallbacks_total counter\n";
  s += "dfkv_rdma_client_leaseput_path_fallbacks_total " +
       std::to_string(leaseput_path_fallbacks_.load(std::memory_order_relaxed)) +
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
  s += "# HELP dfkv_rdma_client_session_closed_reaps_total Pooled connections dropped before reuse because the server had closed their session channel (DFKV_RDMA_SESSION_FD)\n";
  s += "# TYPE dfkv_rdma_client_session_closed_reaps_total counter\n";
  s += "dfkv_rdma_client_session_closed_reaps_total " +
       std::to_string(session_closed_reaps_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_pipelined_read_conns_total Connections opened with pipelined RDMA READs (DFKV_RDMA_PIPELINED_READS, server advertising its READ depth)\n";
  s += "# TYPE dfkv_rdma_client_pipelined_read_conns_total counter\n";
  s += "dfkv_rdma_client_pipelined_read_conns_total " +
       std::to_string(pipelined_read_conns_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_stale_pool_connections_total Pooled connections that failed on reuse and were discarded without cooling the peer rail\n";
  s += "# TYPE dfkv_rdma_client_stale_pool_connections_total counter\n";
  s += "dfkv_rdma_client_stale_pool_connections_total " +
       std::to_string(
           stale_pool_connections_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_peer_unreachable_total Connection attempts whose TCP connect to the peer's bootstrap address failed; no rail charged, not retried\n";
  s += "# TYPE dfkv_rdma_client_peer_unreachable_total counter\n";
  s += "dfkv_rdma_client_peer_unreachable_total " +
       std::to_string(peer_unreachable_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_peer_unreachable_skips_total Connection attempts not made because a connect to the same address failed within DFKV_RDMA_REMOTE_RAIL_COOLDOWN_MS\n";
  s += "# TYPE dfkv_rdma_client_peer_unreachable_skips_total counter\n";
  s += "dfkv_rdma_client_peer_unreachable_skips_total " +
       std::to_string(
           peer_unreachable_skips_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_peer_busy_total First connection attempts the peer closed before any reply; no rail charged, retried once after a pause\n";
  s += "# TYPE dfkv_rdma_client_peer_busy_total counter\n";
  s += "dfkv_rdma_client_peer_busy_total " +
       std::to_string(peer_busy_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_rail_background_probes_total Recovery probes of a rail out of its cooldown sent by the background thread instead of an application operation\n";
  s += "# TYPE dfkv_rdma_client_rail_background_probes_total counter\n";
  s += "dfkv_rdma_client_rail_background_probes_total " +
       std::to_string(rail_background_probes_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_port_down_events_total Local rails taken out of selection because their port left ACTIVE\n";
  s += "# TYPE dfkv_rdma_client_port_down_events_total counter\n";
  s += "dfkv_rdma_client_port_down_events_total " +
       std::to_string(port_down_events_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_rails_port_down Local rails out of selection now because their port is not ACTIVE\n";
  s += "# TYPE dfkv_rdma_client_rails_port_down gauge\n";
  s += "dfkv_rdma_client_rails_port_down " +
       std::to_string(rails_port_down_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_op_deadline_expired_total Operations whose deadline (DFKV_RDMA_OP_TIMEOUT_MS, or the batch timeout per attempt) ran out while their connection was being set up; nothing was charged\n";
  s += "# TYPE dfkv_rdma_client_op_deadline_expired_total counter\n";
  s += "dfkv_rdma_client_op_deadline_expired_total " +
       std::to_string(op_deadline_expired_.load(std::memory_order_relaxed)) + "\n";
  // TRN-29: device resets.
  s += "# HELP dfkv_rdma_client_async_events_total Device async events read (IBV_EVENT_*)\n";
  s += "# TYPE dfkv_rdma_client_async_events_total counter\n";
  {
    std::lock_guard<std::mutex> lock(device_events_mu_);
    for (const auto& [key, count] : device_events_)
      s += "dfkv_rdma_client_async_events_total{dev=\"" + key.first +
           "\",event=\"" + key.second + "\"} " + std::to_string(count) + "\n";
  }
  s += "# HELP dfkv_rdma_client_device_suspect_total Signals that a rail's device was reset: DEVICE_FATAL, or an endpoint Open that failed with ENODEV or EIO\n";
  s += "# TYPE dfkv_rdma_client_device_suspect_total counter\n";
  s += "# HELP dfkv_rdma_client_device_dead Rails out of selection until rebuilt on their reset device (DFKV_RDMA_DEVICE_REBUILD)\n";
  s += "# TYPE dfkv_rdma_client_device_dead gauge\n";
  s += "# HELP dfkv_rdma_client_device_rebuilds_total Rails rebuilt on a reset device (DFKV_RDMA_DEVICE_REBUILD)\n";
  s += "# TYPE dfkv_rdma_client_device_rebuilds_total counter\n";
  for (size_t rail = 0; rail < devs_.size(); ++rail) {
    const std::string dev = "{dev=\"" + devs_[rail] + "\"";
    s += "dfkv_rdma_client_device_suspect_total" + dev + ",reason=\"fatal\"} " +
         std::to_string(device_suspects_[rail * 2].load(std::memory_order_relaxed)) + "\n";
    s += "dfkv_rdma_client_device_suspect_total" + dev + ",reason=\"open\"} " +
         std::to_string(device_suspects_[rail * 2 + 1].load(std::memory_order_relaxed)) + "\n";
    s += "dfkv_rdma_client_device_dead" + dev + "} " +
         std::to_string(RailStale(rail) ? 1 : 0) + "\n";
    s += "dfkv_rdma_client_device_rebuilds_total" + dev + "} " +
         std::to_string(device_rebuilds_[rail].load(std::memory_order_relaxed)) + "\n";
  }
  s += "# HELP dfkv_rdma_client_dead_release_failures_total Releases (dereg, dealloc, close) that failed on a device proven gone and were dropped instead of aborting (DFKV_RDMA_DEVICE_REBUILD)\n";
  s += "# TYPE dfkv_rdma_client_dead_release_failures_total counter\n";
  s += "dfkv_rdma_client_dead_release_failures_total " +
       std::to_string(rdma::DeadReleaseFailures()) + "\n";
  s += "# HELP dfkv_rdma_client_rail_refusals_total Peer rails charged as failed because the peer closed a bootstrap on them, took one on another rail, and closed a confirming dial on them again (it does not serve that device)\n";
  s += "# TYPE dfkv_rdma_client_rail_refusals_total counter\n";
  s += "dfkv_rdma_client_rail_refusals_total " +
       std::to_string(rail_refusals_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_local_resource_held_total Operations failed at once because every rail's endpoint open failed with ENOMEM within the last few seconds\n";
  s += "# TYPE dfkv_rdma_client_local_resource_held_total counter\n";
  s += "dfkv_rdma_client_local_resource_held_total " +
       std::to_string(local_resource_held_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_local_resource_rail_fallbacks_total Endpoint opens refused for lack of descriptors or memory whose operation was then served on another rail\n";
  s += "# TYPE dfkv_rdma_client_local_resource_rail_fallbacks_total counter\n";
  s += "dfkv_rdma_client_local_resource_rail_fallbacks_total " +
       std::to_string(local_resource_rail_fallbacks_.load(
           std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_local_resource_failures_total Operations failed because this process could not register a caller buffer or create an endpoint for lack of descriptors or memory; no rail or peer charged\n";
  s += "# TYPE dfkv_rdma_client_local_resource_failures_total counter\n";
  s += "dfkv_rdma_client_local_resource_failures_total " +
       std::to_string(
           local_resource_failures_.load(std::memory_order_relaxed)) +
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
  s += "# HELP dfkv_rdma_client_pull_prepares_total Pull-read PREPARE requests posted\n";
  s += "# TYPE dfkv_rdma_client_pull_prepares_total counter\n";
  s += "dfkv_rdma_client_pull_prepares_total " +
       std::to_string(pull_prepares_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_pull_reads_total Successful initiator READ operations\n";
  s += "# TYPE dfkv_rdma_client_pull_reads_total counter\n";
  s += "dfkv_rdma_client_pull_reads_total " +
       std::to_string(pull_reads_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_unregistered_dest_registrations_total Registrations made for GET destinations outside registered memory (segments that touch share one)\n";
  s += "# TYPE dfkv_rdma_client_unregistered_dest_registrations_total counter\n";
  s += "dfkv_rdma_client_unregistered_dest_registrations_total " +
       std::to_string(dest_registrations_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_unregistered_dest_memlock_fallbacks_total GETs whose unregistered destination the locked-memory limit refused as a whole, registered round by round instead\n";
  s += "# TYPE dfkv_rdma_client_unregistered_dest_memlock_fallbacks_total counter\n";
  s += "dfkv_rdma_client_unregistered_dest_memlock_fallbacks_total " +
       std::to_string(dest_memlock_fallbacks_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_unregistered_dest_span_fallbacks_total Merged GET destination ranges the device would not register, registered segment by segment instead\n";
  s += "# TYPE dfkv_rdma_client_unregistered_dest_span_fallbacks_total counter\n";
  s += "dfkv_rdma_client_unregistered_dest_span_fallbacks_total " +
       std::to_string(dest_span_fallbacks_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_pull_read_wrs_total READ work requests posted (one may scatter into several destination segments)\n";
  s += "# TYPE dfkv_rdma_client_pull_read_wrs_total counter\n";
  s += "dfkv_rdma_client_pull_read_wrs_total " +
       std::to_string(pull_read_wrs_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_pull_read_bytes_total Bytes completed by initiator READ\n";
  s += "# TYPE dfkv_rdma_client_pull_read_bytes_total counter\n";
  s += "dfkv_rdma_client_pull_read_bytes_total " +
       std::to_string(pull_read_bytes_.load(std::memory_order_relaxed)) +
       "\n";
  s += "# HELP dfkv_rdma_client_pull_failures_total Pull-read operations that destroyed their connection\n";
  s += "# TYPE dfkv_rdma_client_pull_failures_total counter\n";
  s += "dfkv_rdma_client_pull_failures_total " +
       std::to_string(pull_failures_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_pull_releases_total Acknowledged dynamic pull releases\n";
  s += "# TYPE dfkv_rdma_client_pull_releases_total counter\n";
  s += "dfkv_rdma_client_pull_releases_total " +
       std::to_string(pull_releases_.load(std::memory_order_relaxed)) + "\n";
  s += "# HELP dfkv_rdma_client_pull_release_failures_total Failed dynamic pull release exchanges\n";
  s += "# TYPE dfkv_rdma_client_pull_release_failures_total counter\n";
  s += "dfkv_rdma_client_pull_release_failures_total " +
       std::to_string(pull_release_failures_.load(std::memory_order_relaxed)) + "\n";
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
                            RemoteRailOutcome remote_outcome,
                            std::optional<uint64_t> application_use_us) {
  MarkInactive(c);
  if (application_use_us) c->last_application_us = *application_use_us;
  CompleteRemoteLease(c, remote_outcome);
  bool refresh_ok = false;
  {
    // A superseded generation this connection held last: deregistered
    // without mu_ (also if the refresh throws), while the connection keeps
    // its device open.
    rdma::RetiredPoolMrs retired;
    {
      std::lock_guard<std::mutex> lk(mu_);
      // Refresh while still active and while pools_ is stable. A refresh
      // failure is local MR evidence and must tear down before returning the
      // lease.
      refresh_ok =
          c->ep.EnsurePoolMrs(pools_, /*remote_write=*/false, retired.get());
    }
  }
  if (!refresh_ok) {
    Destroy(c, rdma::RailCompletion::kRailFailure);
    return;
  }

  if (c->credit_held) {
    const uint64_t now = rdma::RailPolicy::NowMicros();
    rail_policy_->Complete(c->lease,
                           SampleLatency(now - c->lease_started_us,
                                         c->lease_bytes),
                           rdma::RailCompletion::kSuccess, now);
    c->credit_held = false;
  }

  bool reusable = false;
  bool publication_current = false;
  Conn* displaced = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    publication_current = peer_topologies_->IsCurrent(
        node, c->peer_id, c->peer_publication);
    if (publication_current) {
      auto& idle =
          lane == Lane::kControl ? control_pool_ : pool_;
      auto& v = idle[node];
      const size_t pool_cap =
          lane == Lane::kControl ? pool_max_ : data_pool_max_;
      if (v.size() < pool_cap && c->lifecycle.MakeIdle()) {
        c->lane = lane;
        v.push_back(c);
        reusable = true;
      } else if (lane != Lane::kControl && !v.empty()) {
        auto largest = std::max_element(
            v.begin(), v.end(), [](const Conn* left, const Conn* right) {
              return left->budget_request.registered_bytes <
                     right->budget_request.registered_bytes;
            });
        if (largest != v.end() &&
            (*largest)->budget_request.registered_bytes >
                c->budget_request.registered_bytes &&
            (*largest)->lifecycle.RequestRetire()) {
          displaced = *largest;
          if (c->lifecycle.MakeIdle()) {
            c->lane = lane;
            *largest = c;
            reusable = true;
          } else {
            v.erase(largest);
          }
        }
      }
    }
  }
  if (displaced) Destroy(displaced);
  if (!reusable && !publication_current)
    stale_publication_reaps_.fetch_add(1, std::memory_order_relaxed);
  if (!reusable) Destroy(c);
}

Status RdmaTransport::RoundTrip(const std::string& node, WireOp op,
                                const BlockKey& key, uint64_t offset,
                                uint64_t length, const void* payload,
                                uint64_t payload_len, std::string* out,
                                uint64_t* value_len) {
  if (out) out->clear();
  const uint64_t value_bound = static_cast<uint64_t>(OpBound());
  if ((op == WireOp::kCache &&
       (payload_len == 0 || payload_len > value_bound ||
        payload_len > std::numeric_limits<size_t>::max() ||
        !ValidBuffer(payload, static_cast<size_t>(payload_len)))) ||
      (op == WireOp::kMembers && out == nullptr)) {
    return Status::kInvalid;
  }

  const Lane lane = op == WireOp::kCache ? Lane::kData : Lane::kControl;
  const bool local_rail_replay_safe =
      op != WireOp::kCache && op != WireOp::kRemove;
  const ReplaySafety replay_safety =
      op == WireOp::kCache ? ReplaySafety::kUnsafeAfterPost
                           : ReplaySafety::kReplaySafe;
  const auto peer = peer_topologies_->Snapshot(node);
  RailMask excluded(devs_.size(), 0);
  OpState op_state;
  op_state.deadline_us = DeadlineIn(op_timeout_ms_);
  bool cross_rail_retry = false;
  CompletionFaultOperation completion_fault(
      &test_completion_fault_calls_, op == WireOp::kCache);
  for (int attempt = 0; attempt < 2; ++attempt) {
    completion_fault.BeginAttempt(attempt);
    std::string attempt_out;
    uint64_t attempt_value_len = 0;
    AcquireOptions options;
    options.required_data_bytes =
        lane == Lane::kControl
            ? 0
            : static_cast<size_t>(std::max(payload_len, length));
    options.force_new = attempt != 0;
    options.excluded = excluded;
    options.op = &op_state;
    options.peer = peer;
    AcquireResult acquired = Acquire(node, lane, options);
    if (!acquired.conn) {
      if (PrepareRetry(attempt, acquired.from_pool, acquired.attempted_rail,
                       acquired.failure, replay_safety, false, peer, node, &excluded,
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
                       AcquireFailure::kLocalRail, replay_safety, false, peer, node,
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
      } else {
        NoteLocalRegistrationFailure(conn, static_cast<size_t>(payload_len));
      }
    } else {
      // Control ops carry any payload inline after the prefix (kRetain keys).
      if (payload_len > 0 &&
          (payload == nullptr || ep.cap() < kReqPrefix ||
           payload_len > ep.cap() - kReqPrefix)) {
        Release(node, lane, conn, RemoteRailOutcome::kAbandon);
        return Status::kInvalid;
      }
      conn->Encode(ep.sbuf(0), op, key, offset, length, payload_len);
      if (payload_len > 0)
        std::memcpy(ep.sbuf(0) + kReqPrefix, payload,
                    static_cast<size_t>(payload_len));
      ok = ep.PostRecv(0) &&
           ep.PostSend(0, kReqPrefix + static_cast<size_t>(payload_len));
    }
    request_posted = ok;

    std::vector<uint32_t> reply_bytes;
    bool timed_out = false;
    if (ok) {
      ibv_wc_status wc_status = IBV_WC_SUCCESS;
      bool had_wcs = false;
      ok = ReapWindow(ep, 1, &reply_bytes,
                      RemainingMs(&op_state, op_timeout_ms_), &timed_out,
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
        op == WireOp::kMembers || op == WireOp::kRetain ||
                op == WireOp::kRemovedWithin ||
                op == WireOp::kRemovedWithinAges
            ? rdma::kV2ControlResponseMax
            : 0;
    if (ok && !conn->Decode(
                  ep.rbuf(0), &status, &data_len, response_bound,
                  value_len ? &attempt_value_len : nullptr)) {
      ok = false;
      completion = rdma::RailCompletion::kEndpointFailure;
    }
    if (ok && out) {
      if (data_len > rdma::kV2ControlResponseMax ||
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
      if (status == Status::kOk && op == WireOp::kCache)
        RecordRailTransfer(conn, true, 1, payload_len);
      Release(node, lane, conn);
      if (cross_rail_retry)
        cross_rail_retry_successes_.fetch_add(1,
                                              std::memory_order_relaxed);
      return status;
    }

    const size_t failed_rail = conn->rail_index;
    const bool local_resource = conn->local_resource_failure;
    const bool stale_pooled = Destroy(conn, completion);
    if (local_resource) {
      // Another rail or another attempt would fail the same way.
      if (cross_rail_retry)
        cross_rail_retry_exhausted_.fetch_add(1, std::memory_order_relaxed);
      return Status::kResourceExhausted;
    }
    const AcquireFailure failure =
        completion == rdma::RailCompletion::kRailFailure &&
                (local_rail_replay_safe || !request_posted)
            ? AcquireFailure::kLocalRail
            : AcquireFailure::kEndpoint;
    if (PrepareRetry(attempt, acquired.from_pool, failed_rail, failure,
                     replay_safety, request_posted && !stale_pooled, peer, node,
                     &excluded, &cross_rail_retry)) {
      // A stale pooled connection reached no server: its retry gets the
      // whole timeout again (it may have used all of it waiting for a
      // retry-exceeded completion). Any other retry gets what is left.
      if (stale_pooled) op_state.deadline_us = DeadlineIn(op_timeout_ms_);
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
  if (inline_put_max_bytes_ != 0 && len > inline_put_max_bytes_)
    return CacheMany(node, {{key, data, len}})[0];
  if (NoteBlock(len)) return Status::kInvalid;
  return RoundTrip(node, WireOp::kCache, key, 0, 0, data, len, nullptr);
}
Status RdmaTransport::Range(const std::string& node, const BlockKey& key,
                            uint64_t offset, uint64_t length,
                            std::string* out, uint64_t* value_len) {
  if (length > std::numeric_limits<size_t>::max() ||
      NoteBlock(static_cast<size_t>(length)))
    return Status::kInvalid;
  if (out == nullptr) return Status::kInvalid;
  out->clear();
  std::string staged(static_cast<size_t>(length), '\0');
  const RangeDstSegment segment{staged.data(), staged.size()};
  uint64_t received = 0;
  const Status status = PullInto(
      node, key, &segment, 1, staged.size(), Lane::kData, value_len,
      nullptr, PullMode::kSlice, offset, length, &received);
  if (status == Status::kOk) {
    staged.resize(static_cast<size_t>(received));
    *out = std::move(staged);
  }
  return status;
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

Status RdmaTransport::Retain(const std::string& node, uint64_t tenant_hash,
                             uint64_t count, const std::string& payload,
                             std::string* out) {
  if (out == nullptr || count == 0 || count > kRetainMaxKeys)
    return Status::kInvalid;
  return RoundTrip(node, WireOp::kRetain, BlockKey{0, 0, tenant_hash}, 0,
                   count, payload.data(), payload.size(), out);
}

Status RdmaTransport::RemovedWithin(const std::string& node,
                                    uint64_t tenant_hash, uint64_t count,
                                    const std::string& payload,
                                    std::string* out) {
  if (out == nullptr || count == 0 || count > kRemovedWithinMaxKeys)
    return Status::kInvalid;
  return RoundTrip(node, WireOp::kRemovedWithin, BlockKey{0, 0, tenant_hash},
                   0, count, payload.data(), payload.size(), out);
}

Status RdmaTransport::RemovedWithinAges(const std::string& node,
                                        uint64_t tenant_hash, uint64_t count,
                                        const std::string& payload,
                                        std::string* out) {
  if (out == nullptr || count == 0 || count > kRemovedWithinMaxKeys)
    return Status::kInvalid;
  return RoundTrip(node, WireOp::kRemovedWithinAges,
                   BlockKey{0, 0, tenant_hash}, 0, count, payload.data(),
                   payload.size(), out);
}

Status RdmaTransport::Remove(const std::string& node, const BlockKey& key) {
  return RoundTrip(node, WireOp::kRemove, key, 0, 0, nullptr, 0, nullptr);
}

std::vector<Status> RdmaTransport::CacheMany(
    const std::string& node, const std::vector<CacheItem>& items) {
  const size_t count = items.size();
  size_t max_all_len = 0;
  size_t max_inline_len = 0;
  std::vector<Status> result(count, Status::kIOError);
  if (count == 0) return result;
  std::vector<char> bad(count, 0);
  size_t valid_count = 0;
  // Track resident geometry separately from the logical object ceiling.
  bool any_over = false;
  const bool lease_enabled = inline_put_max_bytes_ != 0;
  for (size_t i = 0; i < count; ++i) {
    if (items[i].len == 0 ||
        !ValidBuffer(items[i].data, items[i].len) ||
        NoteBlock(items[i].len)) {
      bad[i] = 1;
      result[i] = Status::kInvalid;
    } else {
      max_all_len = std::max(max_all_len, items[i].len);
      if (lease_enabled && items[i].len > inline_put_max_bytes_) {
        any_over = true;
      } else {
        max_inline_len = std::max(max_inline_len, items[i].len);
      }
      ++valid_count;
    }
  }
  if (valid_count == 0) return result;

  const auto peer = peer_topologies_->Snapshot(node);
  RailMask excluded(devs_.size(), 0);
  OpState op_state;
  bool cross_rail_retry = false;
  CompletionFaultOperation completion_fault(&test_completion_fault_calls_);
  for (int attempt = 0; attempt < 2; ++attempt) {
    completion_fault.BeginAttempt(attempt);
    std::fill(result.begin(), result.end(), Status::kIOError);
    for (size_t i = 0; i < count; ++i)
      if (bad[i]) result[i] = Status::kInvalid;
    AcquireOptions options;
    // Retry safety is unchanged: only a pre-write failure may use a fresh
    // traditional connection. Capability absence is resolved inside Acquire,
    // before geometry selection, and does not consume this retry.
    const bool want_lease = attempt == 0 && any_over && lease_enabled;
    options.required_data_bytes = max_all_len;
    options.leased_inline_bytes = max_inline_len;
    options.request_leased_put = want_lease;
    options.force_new = attempt != 0;
    options.requested_credits = std::min(valid_count, depth_);
    options.excluded = excluded;
    options.op = &op_state;
    // Bounds this attempt's connection; the windows keep their own timeouts.
    op_state.deadline_us = DeadlineIn(BatchTimeout());
    options.peer = peer;
    AcquireResult acquired = Acquire(node, Lane::kData, options);
    if (!acquired.conn) {
      if (PrepareRetry(attempt, acquired.from_pool, acquired.attempted_rail,
                       acquired.failure, ReplaySafety::kUnsafeAfterPost, false,
                       peer, node, &excluded, &cross_rail_retry))
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
    // Bounded contiguous runs preserve first-write-wins for duplicate keys
    // across inline/lease transitions, without allocating index buckets.
    for (size_t base = 0; base < count && conn_ok;) {
      if (bad[base]) {
        ++base;
        continue;
      }
      const bool use_lease =
          conn->leased_put && items[base].len > inline_put_max_bytes_;
      size_t end = base + 1;
      while (end < count && end - base < window &&
             (bad[end] ||
              (conn->leased_put && items[end].len > inline_put_max_bytes_) ==
                  use_lease))
        ++end;
      const size_t width = end - base;
      if (!use_lease) {
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
        if (item.len != 0 && !mr) {
          NoteLocalRegistrationFailure(conn, item.len);
          conn_ok = false;
          break;
        }
        conn->Encode(ep.sbuf(slot), WireOp::kCache, item.key, 0, 0,
                     item.len);
        if (!ep.PostRecv(slot) ||
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
      } else {
      // Three phases: reserve, WRITE_WITH_IMM, then reap commit status.
      struct LeaseSlot {
        size_t item_index = 0;
        uint64_t write_base = 0;
        uint32_t rkey = 0;
        // The server's lease index (its own receive-slot numbering). The
        // WRITE immediate and every lease state lookup happen in the
        // SERVER's slot space, which is not the client's pipeline index.
        uint32_t server_slot = 0;
      };
      std::vector<LeaseSlot> leases(width);
      size_t accepted = 0;
      for (size_t slot = 0; slot < width && conn_ok; ++slot) {
        const size_t item_index = base + slot;
        leases[slot].item_index = item_index;
        if (bad[item_index]) continue;
        const CacheItem& item = items[item_index];
        conn->Encode(ep.sbuf(slot), WireOp::kLeasePut, item.key, 0, 0,
                     item.len);
        if (!ep.PostRecv(slot) || !ep.PostSend(slot, kReqPrefix)) {
          conn_ok = false;
          completion = rdma::RailCompletion::kRailFailure;
          break;
        }
        ++accepted;
      }
      std::vector<uint32_t> reply_bytes;
      bool timed_out = false;
      ibv_wc_status wc_status = IBV_WC_SUCCESS;
      bool had_wcs = false;
      if (conn_ok &&
          !ReapPosted(ep, accepted, width, &reply_bytes, BatchTimeout(),
                      &timed_out, &wc_status, &had_wcs, &completion_fault)) {
        conn_ok = false;
        completion = rdma::ClassifyCompletion(wc_status, had_wcs);
      }
      if (timed_out)
        completion_timeouts_.fetch_add(1, std::memory_order_relaxed);
      if (!conn_ok) break;
      size_t written = 0;
      std::vector<ibv_mr*> payload_mrs(width, nullptr);
      for (size_t slot = 0; slot < width && conn_ok; ++slot) {
        const size_t item_index = leases[slot].item_index;
        if (bad[item_index]) continue;
        Status status;
        uint64_t data_len = 0;
        if (reply_bytes[slot] < kRespPrefix ||
            !conn->Decode(ep.rbuf(slot), &status, &data_len,
                          rdma::kLeasePutReadyBytes) ||
            (status == Status::kOk) !=
                (data_len == rdma::kLeasePutReadyBytes)) {
          conn_ok = false;
          completion = rdma::RailCompletion::kEndpointFailure;
          break;
        }
        // Backpressure (busy pool) and permanent verdicts are per-object
        // outcomes: record them and move on without tearing the connection.
        if (status != Status::kOk) {
          result[item_index] = status;
          continue;
        }
        rdma::LeasePutReady ready;
        // ready.slot names the SERVER's receive slot holding this lease; it
        // is not this client pipeline index and may legitimately differ.
        // Only bound it by the negotiated depth as fail-closed corruption
        // defense — the immediate must reach a server-side active lease.
        if (reply_bytes[slot] < kRespPrefix + rdma::kLeasePutReadyBytes ||
            !rdma::DecodeLeasePutReady(ep.rbuf(slot) + kRespPrefix,
                                       &ready) ||
            ready.slot >= static_cast<uint32_t>(conn->depth) ||
            ready.lease_bytes <
                static_cast<uint64_t>(rdma::kV2DataOffset) +
                    items[item_index].len) {
          conn_ok = false;
          completion = rdma::RailCompletion::kEndpointFailure;
          break;
        }
        leases[slot].write_base = ready.write_base;
        leases[slot].rkey = ready.rkey;
        leases[slot].server_slot = ready.slot;
        ++leaseput_ops_;
        const CacheItem& item = items[item_index];
        ibv_mr* mr = ep.RegisterTransient(const_cast<void*>(item.data),
                                           item.len, /*remote_write=*/false);
        payload_mrs[slot] = mr;
        if (!mr) {
          NoteLocalRegistrationFailure(conn, item.len);
          conn_ok = false;
          completion = rdma::RailCompletion::kRailFailure;
          break;
        }
        // The WRITE window is an ordinary v2 PUT frame aimed at the leased
        // range: re-encode the request header (phase A left a kLeasePut
        // request in this send buffer).
        conn->Encode(ep.sbuf(slot), WireOp::kCache, item.key, 0, 0, item.len);
        if (!ep.PostRecv(slot) ||
            !ep.PostWriteImmScatter(
                slot, kReqPrefix, item.data, item.len, mr,
                leases[slot].write_base + rdma::kV2PutPrefixOffset,
                leases[slot].rkey, leases[slot].server_slot)) {
          conn_ok = false;
          completion = rdma::RailCompletion::kRailFailure;
          break;
        }
        ++written;
        v2_put_writes_.fetch_add(1, std::memory_order_relaxed);
        request_posted = true;
      }
      if (!conn_ok) break;
      reply_bytes.clear();
      timed_out = false;
      wc_status = IBV_WC_SUCCESS;
      had_wcs = false;
      if (!ReapPosted(ep, written, width, &reply_bytes, BatchTimeout(),
                      &timed_out, &wc_status, &had_wcs, &completion_fault)) {
        conn_ok = false;
        completion = rdma::ClassifyCompletion(wc_status, had_wcs);
      }
      if (timed_out)
        completion_timeouts_.fetch_add(1, std::memory_order_relaxed);
      if (!conn_ok) break;
      for (ibv_mr* mr : payload_mrs) ep.ReleaseTransient(mr);
      for (size_t slot = 0; slot < width; ++slot) {
        const size_t item_index = leases[slot].item_index;
        if (bad[item_index] || leases[slot].write_base == 0) continue;
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
      base = end;
    }
    if (conn_ok) {
      uint64_t successful_ops = 0;
      uint64_t successful_bytes = 0;
      for (size_t i = 0; i < count; ++i) {
        if (result[i] != Status::kOk) continue;
        ++successful_ops;
        successful_bytes += items[i].len;
      }
      RecordRailTransfer(conn, true, successful_ops, successful_bytes);
      Release(node, Lane::kData, conn);
      if (cross_rail_retry)
        cross_rail_retry_successes_.fetch_add(1,
                                              std::memory_order_relaxed);
      return result;
    }
    const size_t failed_rail = conn->rail_index;
    const bool local_resource = conn->local_resource_failure;
    const bool stale_pooled = Destroy(conn, completion);
    if (local_resource) {
      // Another rail or another attempt would fail the same way. Items that
      // already have their answer keep it.
      if (cross_rail_retry)
        cross_rail_retry_exhausted_.fetch_add(1, std::memory_order_relaxed);
      MarkClientLocalFailure(&result, Status::kResourceExhausted);
      return result;
    }
    const AcquireFailure failure =
        completion == rdma::RailCompletion::kRailFailure && !request_posted
            ? AcquireFailure::kLocalRail
            : AcquireFailure::kEndpoint;
    if (PrepareRetry(attempt, acquired.from_pool, failed_rail, failure,
                     ReplaySafety::kUnsafeAfterPost,
                     request_posted && !stale_pooled, peer, node,
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
  OpState op_state;
  bool cross_rail_retry = false;
  for (int attempt = 0; attempt < 2; ++attempt) {
    std::fill(result.begin(), result.end(), Status::kIOError);
    std::vector<std::string> attempt_outputs(count);
    std::vector<uint64_t> attempt_value_lens(count, 0);
    AcquireOptions options;
    options.required_data_bytes = static_cast<size_t>(length);
    options.request_dynamic_pull = true;
    options.force_new = attempt != 0;
    options.requested_credits = std::min(count, depth_);
    options.excluded = excluded;
    options.op = &op_state;
    // Bounds this attempt's connection; the windows keep their own timeouts.
    op_state.deadline_us = DeadlineIn(BatchTimeout());
    options.peer = peer;
    AcquireResult acquired = Acquire(node, Lane::kData, options);
    if (!acquired.conn) {
      if (PrepareRetry(attempt, acquired.from_pool, acquired.attempted_rail,
                       acquired.failure, ReplaySafety::kReplaySafe, false,
                       peer, node, &excluded, &cross_rail_retry))
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
    std::vector<RangeDstSegment> segments(window, RangeDstSegment{nullptr, 0});
    std::vector<PullRequest> requests(window);
    for (size_t base = 0; base < count && conn_ok; base += window) {
      const size_t width = std::min(window, count - base);
      for (size_t slot = 0; slot < width; ++slot) {
        std::string& output = attempt_outputs[base + slot];
        output.resize(static_cast<size_t>(length));
        segments[slot] = {output.data(), output.size()};
        requests[slot] = {&keys[base + slot], &segments[slot], 1,
                          output.size(), PullMode::kSlice, offset, length};
      }
      conn_ok = PullWindow(conn, requests.data(), width, BatchTimeout(),
                           &completion);
      if (!conn_ok) break;
      for (size_t slot = 0; slot < width; ++slot) {
        const PullRequest& request = requests[slot];
        result[base + slot] = request.status;
        attempt_value_lens[base + slot] = request.value_len;
        attempt_outputs[base + slot].resize(
            request.status == Status::kOk ? request.data_len : 0);
      }
    }
    if (conn_ok) {
      uint64_t successful_ops = 0;
      uint64_t successful_bytes = 0;
      for (size_t i = 0; i < count; ++i) {
        if (result[i] != Status::kOk) continue;
        ++successful_ops;
        successful_bytes += attempt_outputs[i].size();
      }
      RecordRailTransfer(conn, false, successful_ops, successful_bytes);
      *outputs = std::move(attempt_outputs);
      if (value_lens) *value_lens = std::move(attempt_value_lens);
      Release(node, Lane::kData, conn);
      if (cross_rail_retry)
        cross_rail_retry_successes_.fetch_add(1,
                                              std::memory_order_relaxed);
      return result;
    }
    const size_t failed_rail = conn->rail_index;
    const bool local_resource = conn->local_resource_failure;
    pull_failures_.fetch_add(1, std::memory_order_relaxed);
    // Fence the QP before the operation-owned string destinations are freed.
    Destroy(conn, completion);
    if (local_resource) {
      if (cross_rail_retry)
        cross_rail_retry_exhausted_.fetch_add(1, std::memory_order_relaxed);
      MarkClientLocalFailure(&result, Status::kResourceExhausted);
      return result;
    }
    const AcquireFailure failure =
        completion == rdma::RailCompletion::kRailFailure
            ? AcquireFailure::kLocalRail
            : AcquireFailure::kEndpoint;
    if (PrepareRetry(attempt, acquired.from_pool, failed_rail, failure,
                     ReplaySafety::kReplaySafe, false, peer, node, &excluded,
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
  OpState op_state;
  bool cross_rail_retry = false;
  bool out_dev_set = false;
  for (int attempt = 0; attempt < 2; ++attempt) {
    std::fill(result.begin(), result.end(), Status::kIOError);
    std::fill(exists->begin(), exists->end(), 0);
    AcquireOptions options;
    options.force_new = attempt != 0;
    options.requested_credits = std::min(count, depth_);
    options.excluded = excluded;
    options.op = &op_state;
    // Bounds this attempt's connection; the windows keep their own timeouts.
    op_state.deadline_us = DeadlineIn(BatchTimeout());
    options.peer = peer;
    AcquireResult acquired = Acquire(node, Lane::kControl, options);
    if (!acquired.conn) {
      if (PrepareRetry(attempt, acquired.from_pool, acquired.attempted_rail,
                       acquired.failure, ReplaySafety::kReplaySafe, false,
                       peer, node, &excluded, &cross_rail_retry))
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
                     ReplaySafety::kReplaySafe, false, peer, node, &excluded,
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

bool RdmaTransport::PullWindow(
    Conn* conn, PullRequest* requests, size_t width, int timeout_ms,
    rdma::RailCompletion* failure, bool* window_timed_out) {
  rdma::RcEndpoint& ep = conn->ep;
  *failure = rdma::RailCompletion::kRailFailure;
  if (window_timed_out) *window_timed_out = false;
  // One deadline for the window's request, READs and release together.
  const CompletionDeadline window_deadline(timeout_ms);
  struct ReadState {
    rdma::DynamicPullReady ready{};
    size_t copied = 0;
    size_t segment = 0;
    // The lkey of each destination segment, and the registrations made for
    // this pull (kept until its last READ completes).
    std::vector<uint32_t> lkeys;
    std::vector<ibv_mr*> mrs;
    // Registered round by round instead (a locked-memory limit refused the
    // whole destination at once): mrs then holds one round's.
    bool per_round = false;
    size_t bytes = 0;
    bool grant = false;
  };
  ReadState single_state;
  std::vector<ReadState> batch_states(width > 1 ? width : 0);
  ReadState* states = width == 1 ? &single_state : batch_states.data();
  std::vector<uint32_t> replies;
  bool exchange_timed_out = false;  // the last exchange ran out of time
  const auto exchange = [&](size_t count, size_t request_bytes) {
    bool timed_out = false;
    exchange_timed_out = false;
    ibv_wc_status wc_status = IBV_WC_SUCCESS;
    bool had_wcs = false;
    for (size_t slot = 0; slot < count; ++slot) {
      if (!ep.PostRecv(slot) || !ep.PostSend(slot, request_bytes))
        return false;
    }
    const bool ok =
        ReapWindow(ep, count, &replies, window_deadline.Remaining(),
                   &timed_out, &wc_status, &had_wcs);
    if (timed_out) {
      completion_timeouts_.fetch_add(1, std::memory_order_relaxed);
      exchange_timed_out = true;
      if (window_timed_out) *window_timed_out = true;
    }
    if (!ok) *failure = rdma::ClassifyCompletion(wc_status, had_wcs);
    return ok;
  };
  const auto malformed = [&]() {
    *failure = rdma::RailCompletion::kEndpointFailure;
    return false;
  };
  for (size_t slot = 0; slot < width; ++slot) {
    const auto& request = requests[slot];
    conn->Encode(ep.sbuf(slot), WireOp::kPullRange, *request.key,
                 request.offset, request.length, rdma::kPullPrepareBytes);
    rdma::EncodePullPrepareControl(
        {static_cast<uint32_t>(slot), 0}, ep.sbuf(slot) + kReqPrefix);
  }
  pull_prepares_.fetch_add(width, std::memory_order_relaxed);
  if (!exchange(width, kReqPrefix + rdma::kPullPrepareBytes)) return false;
  for (size_t slot = 0; slot < width; ++slot) {
    auto& request = requests[slot];
    auto& state = states[slot];
    uint64_t bytes = 0;
    if (replies[slot] < kRespPrefix ||
        !conn->Decode(ep.rbuf(slot), &request.status, &bytes,
                      rdma::kDynamicPullReadyBytes, &request.value_len) ||
        bytes != (request.status == Status::kOk
                      ? rdma::kDynamicPullReadyBytes : 0) ||
        replies[slot] != kRespPrefix + bytes)
      return malformed();
    if (request.status != Status::kOk) continue;
    if (!rdma::DecodeDynamicPullReady(
            ep.rbuf(slot) + kRespPrefix, &state.ready))
      return malformed();
    const auto& ready = state.ready;
    // Subtract only after proving offset is valid; offset+length can overflow.
    if (ready.slot_index != slot || ready.value_len != request.value_len ||
        request.offset > ready.value_len)
      return malformed();
    const uint64_t expected = std::min<uint64_t>(
        request.length, ready.value_len - request.offset);
    if (ready.data_len != expected || ready.data_len > request.capacity ||
        ready.data_len > OpBound())
      return malformed();
    state.grant = true;
    request.data_len = ready.data_len;
    if (request.mode == PullMode::kWholeObject &&
        ready.value_len > request.capacity) {
      request.status = Status::kInvalid;
      request.too_large = true;
    }
    // A shorter value an exact reader rejects anyway: report its length,
    // READ nothing into the destination.
    if (request.exact && request.mode == PullMode::kWholeObject &&
        ready.value_len != request.capacity)
      request.data_len = 0;
  }

  // The destination's lkeys. A segment in registered memory (RegisterMemory)
  // uses that registration. The others are registered for this pull, and
  // segments that overlap or touch share one registration: it covers the
  // same pages as theirs, so nothing else of the caller's is pinned (or,
  // with fork support on, kept from a child). A merged range the device
  // refuses is registered segment by segment, counted. If the locked-memory
  // limit refuses the destination as a whole (ENOMEM, EAGAIN), the pull
  // registers it round by round, as each READ goes, instead (counted).
  const auto key_destination = [&](const PullRequest& request,
                                   ReadState& state) -> bool {
    state.lkeys.assign(request.segment_count, 0);
    struct Piece {
      uintptr_t begin;
      uintptr_t end;
      size_t index;
      int kind;
    };
    std::vector<Piece> loose;
    size_t left = request.data_len;
    for (size_t i = 0; i < request.segment_count && left != 0; ++i) {
      const auto& segment = request.segments[i];
      if (segment.second == 0) continue;
      const size_t bytes = std::min<size_t>(segment.second, left);
      left -= bytes;
      if (ibv_mr* pool = ep.RegisterUser(segment.first, bytes)) {
        state.lkeys[i] = pool->lkey;
        continue;
      }
      const auto begin = reinterpret_cast<uintptr_t>(segment.first);
      loose.push_back({begin, begin + bytes, i,
                       static_cast<int>(segment.memory_kind)});
    }
    if (loose.empty()) return true;
    std::sort(loose.begin(), loose.end(), [](const Piece& a, const Piece& b) {
      return a.kind != b.kind ? a.kind < b.kind : a.begin < b.begin;
    });
    const auto register_range = [&](uintptr_t begin, uintptr_t end) {
      void* at = reinterpret_cast<void*>(begin);
      if (ibv_mr* pool = ep.RegisterUser(at, end - begin)) return pool;
      ibv_mr* mr = ep.RegisterTransient(
          at, end - begin, rdma::RcEndpoint::TransientAccess::kLocalWrite);
      if (mr) {
        state.mrs.push_back(mr);
        dest_registrations_.fetch_add(1, std::memory_order_relaxed);
      }
      return mr;
    };
    const auto per_round_instead = [&] {
      for (ibv_mr* mr : state.mrs) ep.ReleaseTransient(mr);
      state.mrs.clear();
      state.per_round = true;
      dest_memlock_fallbacks_.fetch_add(1, std::memory_order_relaxed);
      return true;
    };
    for (size_t first = 0; first < loose.size();) {
      size_t last = first + 1;
      uintptr_t end = loose[first].end;
      while (last < loose.size() && loose[last].kind == loose[first].kind &&
             loose[last].begin <= end) {
        end = std::max(end, loose[last].end);
        ++last;
      }
      ibv_mr* span = last - first > 1 ? register_range(loose[first].begin, end)
                                      : nullptr;
      if (last - first > 1 && !span)
        dest_span_fallbacks_.fetch_add(1, std::memory_order_relaxed);
      for (size_t i = first; i < last; ++i) {
        ibv_mr* mr = span ? span : register_range(loose[i].begin, loose[i].end);
        if (!mr) {
          if (errno == ENOMEM || errno == EAGAIN) return per_round_instead();
          NoteLocalRegistrationFailure(conn, loose[i].end - loose[i].begin);
          return false;
        }
        state.lkeys[loose[i].index] = mr->lkey;
      }
      first = last;
    }
    uint64_t suppressed = 0;
    if (dest_log_.Admit(LogThrottle::Key("unregistered destination", 0, 0),
                        rdma::RailPolicy::NowMicros(), &suppressed))
      DFKV_LOG_WARN("rdma: a GET's destination is not in registered memory, "
                    "so it is registered (and pinned, against the "
                    "locked-memory limit) for each GET: " +
                    std::to_string(state.mrs.size()) + " registrations "
                    "here. Register the buffers once with "
                    "dfkv_register_memory. Similar lines suppressed: " +
                    std::to_string(suppressed));
    return true;
  };
  for (size_t slot = 0; slot < width; ++slot) {
    if (requests[slot].status != Status::kOk || requests[slot].data_len == 0)
      continue;
    if (!key_destination(requests[slot], states[slot])) return false;
  }

  // One READ per live item per round preserves depth-wide pipelining without
  // exceeding the endpoint's send-WR window for scatter-gather destinations.
  const size_t read_sge = ep.ReadSge();
  std::vector<ibv_sge> sges;
  sges.reserve(read_sge);
  const CompletionDeadline& deadline = window_deadline;
  for (;;) {
    size_t posted = 0;
    // A round registration the locked-memory limit refused while READs of
    // this round hold theirs: that item waits for the next round.
    bool deferred = false;
    for (size_t slot = 0; slot < width && !deferred; ++slot) {
      auto& state = states[slot];
      const auto& request = requests[slot];
      if (request.status != Status::kOk ||
          state.copied == request.data_len)
        continue;
      // One READ covers as many of the destination's segments as the device
      // scatters into (ReadSge): the remote range is contiguous.
      state.bytes = 0;
      sges.clear();
      const size_t first_segment = state.segment;
      while (sges.size() < read_sge &&
             state.copied + state.bytes < request.data_len &&
             state.segment < request.segment_count) {
        const size_t index = state.segment++;
        const auto& segment = request.segments[index];
        if (segment.second == 0) continue;
        const size_t bytes = std::min<size_t>(
            segment.second, request.data_len - state.copied - state.bytes);
        uint32_t lkey = state.lkeys[index];
        if (state.per_round) {
          // A READ destination: this side's NIC writes it, no peer does.
          ibv_mr* mr = ep.RegisterTransient(
              segment.first, bytes,
              rdma::RcEndpoint::TransientAccess::kLocalWrite);
          if (!mr && posted != 0 && (errno == ENOMEM || errno == EAGAIN)) {
            for (ibv_mr* held : state.mrs) ep.ReleaseTransient(held);
            state.mrs.clear();
            state.segment = first_segment;
            state.bytes = 0;
            deferred = true;
            break;
          }
          if (!mr) {
            NoteLocalRegistrationFailure(conn, bytes);
            return false;
          }
          if (mr != ep.RegisterUser(segment.first, bytes)) {
            state.mrs.push_back(mr);
            dest_registrations_.fetch_add(1, std::memory_order_relaxed);
          }
          lkey = mr->lkey;
        }
        ibv_sge sge{};
        sge.addr = reinterpret_cast<uintptr_t>(segment.first);
        sge.length = static_cast<uint32_t>(bytes);
        sge.lkey = lkey;
        sges.push_back(sge);
        state.bytes += bytes;
      }
      if (deferred) break;
      if (sges.empty()) return malformed();
      if (!ep.PostReadSg(slot, sges.data(), sges.size(),
                         state.ready.address + state.copied,
                         state.ready.rkey))
        return false;
      pull_read_wrs_.fetch_add(1, std::memory_order_relaxed);
      ++posted;
      // Abandon an actually posted READ so Destroy must fence its destination.
      if (InjectPullReadFailure(&test_pull_reads_)) return malformed();
    }
    if (posted == 0) break;
    for (size_t remaining = posted; remaining != 0; --remaining) {
      ibv_wc wc{};
      const int got = ep.WaitComp(&wc, 1, deadline.Remaining());
      if (got != 1 || wc.status != IBV_WC_SUCCESS ||
          wc.opcode != IBV_WC_RDMA_READ) {
        if (got == 0) {
          completion_timeouts_.fetch_add(1, std::memory_order_relaxed);
          if (window_timed_out) *window_timed_out = true;
        }
        *failure = rdma::ClassifyCompletion(
            got == 1 ? wc.status
                     : got == 0 ? IBV_WC_SUCCESS : kLocalFaultWcStatus,
            got == 1);
        return false;
      }
    }
    for (size_t slot = 0; slot < width; ++slot) {
      auto& state = states[slot];
      state.copied += state.bytes;
      pull_read_bytes_.fetch_add(state.bytes, std::memory_order_relaxed);
      state.bytes = 0;
      // A round's registrations, or the whole destination's once its last
      // READ is in: other items' rounds may need the locked memory.
      if (!state.per_round && state.copied != requests[slot].data_len)
        continue;
      for (ibv_mr* mr : state.mrs) ep.ReleaseTransient(mr);
      state.mrs.clear();
    }
  }
  // Every READ is complete: the destination registrations go.
  for (size_t slot = 0; slot < width; ++slot) {
    for (ibv_mr* mr : states[slot].mrs) ep.ReleaseTransient(mr);
    states[slot].mrs.clear();
  }

  // Pack RELEASE messages in control slots while retaining each grant's
  // original slot and generation. A business miss owns no grant to release.
  size_t releases = 0;
  for (size_t slot = 0; slot < width; ++slot) {
    const auto& state = states[slot];
    if (!state.grant) continue;
    // An exact pull released without its READ is not one.
    if (requests[slot].status == Status::kOk &&
        !(requests[slot].exact && requests[slot].data_len == 0))
      pull_reads_.fetch_add(1, std::memory_order_relaxed);
    conn->Encode(ep.sbuf(releases++), WireOp::kPullRelease,
                 *requests[slot].key, 0, state.ready.slot_generation,
                 static_cast<uint64_t>(state.ready.slot_index) + 1);
  }
  if (releases == 0) return true;
  if (!exchange(releases, kReqPrefix)) {
    pull_release_failures_.fetch_add(1, std::memory_order_relaxed);
    // The data is in; only the release reply missed what was left of the
    // window's deadline. A peer answering a little late is not charged
    // for that.
    if (exchange_timed_out && window_deadline.Remaining() == 0)
      *failure = rdma::RailCompletion::kAdmission;
    return false;
  }
  for (size_t slot = 0; slot < releases; ++slot) {
    Status status = Status::kIOError;
    uint64_t bytes = 0;
    if (replies[slot] != kRespPrefix ||
        !conn->Decode(ep.rbuf(slot), &status, &bytes, 0) ||
        status != Status::kOk || bytes != 0) {
      pull_release_failures_.fetch_add(1, std::memory_order_relaxed);
      return malformed();
    }
  }
  pull_releases_.fetch_add(releases, std::memory_order_relaxed);
  return true;
}

Status RdmaTransport::PullInto(
    const std::string& node, const BlockKey& key,
    const RangeDstSegment* segments, size_t segment_count, size_t capacity,
    Lane lane, uint64_t* value_len, std::string* out_dev, PullMode mode,
    uint64_t offset, uint64_t length, uint64_t* data_len, bool exact) {
  if (mode == PullMode::kWholeObject && capacity == 0)
    return Status::kInvalid;
  const auto peer = peer_topologies_->Snapshot(node);
  RailMask excluded(devs_.size(), 0);
  OpState op_state;
  op_state.deadline_us = DeadlineIn(
      mode == PullMode::kSlice ? op_timeout_ms_ : BatchTimeout());
  bool cross_rail_retry = false;
  // String slices retain their former bounded replay-safe recovery. A whole
  // object read into the caller's destination is tried once more only when
  // it found a stale pooled connection (see Destroy): the READs on it are
  // fenced, and the second attempt writes the whole destination again.
  for (int attempt = 0; attempt < 2; ++attempt) {
    AcquireOptions options;
    options.required_data_bytes = capacity;
    options.request_dynamic_pull = true;
    options.requested_credits = 1;
    options.force_new = attempt != 0;
    options.excluded = excluded;
    options.op = &op_state;
    options.peer = peer;
    AcquireResult acquired = Acquire(node, lane, options);
    if (!acquired.conn) {
      if (mode == PullMode::kSlice &&
          PrepareRetry(attempt, acquired.from_pool, acquired.attempted_rail,
                       acquired.failure, ReplaySafety::kReplaySafe, false,
                       peer, node, &excluded, &cross_rail_retry))
        continue;
      if (cross_rail_retry)
        cross_rail_retry_exhausted_.fetch_add(1, std::memory_order_relaxed);
      return acquired.status;
    }
    Conn* conn = acquired.conn;
    if (out_dev) {
      // One out_dev spans every key of a RangeIntoMulti call: record a rail
      // only when it differs from the last one, so the string names the
      // sequence of rails used, not one entry per key.
      const std::string& dev = devs_[conn->rail_index];
      const size_t arrow = out_dev->rfind("->");
      const size_t last = arrow == std::string::npos ? 0 : arrow + 2;
      if (out_dev->empty()) {
        *out_dev = dev;
      } else if (out_dev->compare(last, std::string::npos, dev) != 0) {
        *out_dev += "->";
        *out_dev += dev;
      }
    }
    PullRequest request{&key, segments, segment_count, capacity, mode,
                        mode == PullMode::kWholeObject ? 0 : offset,
                        mode == PullMode::kWholeObject ? capacity : length};
    request.exact = exact;
    rdma::RailCompletion failure = rdma::RailCompletion::kRailFailure;
    if ((mode == PullMode::kSlice && InjectLocalRailFailure(attempt)) ||
        !PullWindow(conn, &request, 1,
                    RemainingMs(&op_state, mode == PullMode::kSlice
                                               ? op_timeout_ms_
                                               : BatchTimeout()),
                    &failure)) {
      pull_failures_.fetch_add(1, std::memory_order_relaxed);
      const size_t failed_rail = conn->rail_index;
      const bool local_resource = conn->local_resource_failure;
      // Fence the READ before the staged destination can be reused or freed.
      const bool stale_pooled = Destroy(conn, failure);
      if (local_resource) {
        if (cross_rail_retry)
          cross_rail_retry_exhausted_.fetch_add(1,
                                                std::memory_order_relaxed);
        return Status::kResourceExhausted;
      }
      if ((mode == PullMode::kSlice || stale_pooled) &&
          PrepareRetry(attempt, acquired.from_pool, failed_rail,
                       failure == rdma::RailCompletion::kRailFailure
                           ? AcquireFailure::kLocalRail
                           : AcquireFailure::kEndpoint,
                       ReplaySafety::kReplaySafe, false, peer, node, &excluded,
                       &cross_rail_retry)) {
        // As in RoundTrip: the retry after a stale pooled connection gets
        // the whole timeout again; any other retry gets what is left.
        if (stale_pooled)
          op_state.deadline_us = DeadlineIn(
              mode == PullMode::kSlice ? op_timeout_ms_ : BatchTimeout());
        continue;
      }
      if (cross_rail_retry)
        cross_rail_retry_exhausted_.fetch_add(1, std::memory_order_relaxed);
      return Status::kIOError;
    }
    // A node's refusal may carry a length; only a value this transport
    // refused as larger than the destination reports its stored length
    // with kInvalid (see Transport::RangeInto).
    if (value_len)
      *value_len = request.status != Status::kInvalid || request.too_large
                       ? request.value_len : 0;
    if (data_len) *data_len = request.data_len;
    if (request.status == Status::kOk)
      RecordRailTransfer(conn, false, 1, request.data_len);
    Release(node, lane, conn);
    if (cross_rail_retry)
      cross_rail_retry_successes_.fetch_add(1, std::memory_order_relaxed);
    return request.status;
  }
  return Status::kIOError;
}

std::vector<Status> RdmaTransport::RangeInto(
    const std::string& node, const std::vector<BlockKey>& keys,
    const std::vector<RangeDst>& destinations,
    std::vector<uint64_t>* value_lens) {
  const size_t count = keys.size();
  if (value_lens) value_lens->assign(count, 0);
  if (destinations.size() != count) return InvalidStatuses(count);
  std::vector<Status> result(count, Status::kInvalid);
  std::vector<RangeDstSegment> segments;
  segments.reserve(count);
  std::vector<PullItem> items;
  items.reserve(count);
  for (size_t item = 0; item < count; ++item) {
    const RangeDst& destination = destinations[item];
    if (!ValidBuffer(destination.payload, destination.n) ||
        NoteBlock(destination.n))
      continue;
    segments.emplace_back(destination.payload, destination.n,
                          destination.memory_kind);
    items.push_back(
        PullItem{item, nullptr, 1, destination.n, destination.exact});
  }
  for (size_t i = 0; i < items.size(); ++i) items[i].segments = &segments[i];
  PullIntoMany(node, keys, items, Lane::kData, &result, value_lens, nullptr);
  return result;
}

void RdmaTransport::PullIntoMany(const std::string& node,
                                 const std::vector<BlockKey>& keys,
                                 const std::vector<PullItem>& items, Lane lane,
                                 std::vector<Status>* statuses,
                                 std::vector<uint64_t>* value_lens,
                                 std::string* out_dev) {
  if (items.empty()) return;
  if (items.size() == 1) {
    // One key: the single-request path and its retry rules as before.
    const PullItem& item = items.front();
    (*statuses)[item.index] = PullInto(
        node, keys[item.index], item.segments, item.segment_count,
        item.capacity, lane,
        value_lens ? &(*value_lens)[item.index] : nullptr, out_dev,
        PullMode::kWholeObject, 0, 0, nullptr, item.exact);
    return;
  }
  const auto peer = peer_topologies_->Snapshot(node);
  RailMask excluded(devs_.size(), 0);
  OpState op_state;
  bool cross_rail_retry = false;
  size_t done = 0;  // items before this one have their final status
  for (size_t i = 0; i < items.size(); ++i)
    (*statuses)[items[i].index] = Status::kIOError;
  // As for one key: a whole object read into the caller's destination is
  // tried once more only after a stale pooled connection (see PullInto).
  // Windows that completed keep their results; the retry starts at the
  // window that failed.
  for (int attempt = 0; attempt < 2 && done < items.size(); ++attempt) {
    size_t largest = 0;
    for (size_t i = done; i < items.size(); ++i)
      largest = std::max(largest, items[i].capacity);
    AcquireOptions options;
    options.required_data_bytes = largest;
    options.request_dynamic_pull = true;
    options.requested_credits = std::min(items.size() - done, depth_);
    options.force_new = attempt != 0;
    options.excluded = excluded;
    // Bounds this attempt's connection; the windows keep their own timeouts.
    op_state.deadline_us = DeadlineIn(BatchTimeout());
    options.op = &op_state;
    options.peer = peer;
    AcquireResult acquired = Acquire(node, lane, options);
    if (!acquired.conn) {
      if (cross_rail_retry)
        cross_rail_retry_exhausted_.fetch_add(1, std::memory_order_relaxed);
      for (size_t i = done; i < items.size(); ++i)
        (*statuses)[items[i].index] = acquired.status;
      return;
    }
    Conn* conn = acquired.conn;
    if (out_dev) {
      const std::string& dev = devs_[conn->rail_index];
      const size_t arrow = out_dev->rfind("->");
      const size_t last = arrow == std::string::npos ? 0 : arrow + 2;
      if (out_dev->empty()) {
        *out_dev = dev;
      } else if (out_dev->compare(last, std::string::npos, dev) != 0) {
        *out_dev += "->";
        *out_dev += dev;
      }
    }
    const size_t window = std::max<size_t>(
        1, std::min(conn->ep.window(), static_cast<size_t>(conn->lease.credits)));
    std::vector<PullRequest> requests(window);
    rdma::RailCompletion failure = rdma::RailCompletion::kRailFailure;
    bool conn_ok = true;
    bool timed_out = false;
    size_t width = 0;
    uint64_t ops = 0, bytes = 0;
    while (done < items.size()) {
      width = std::min(window, items.size() - done);
      for (size_t slot = 0; slot < width; ++slot) {
        const PullItem& item = items[done + slot];
        requests[slot] = PullRequest{&keys[item.index], item.segments,
                                     item.segment_count, item.capacity,
                                     PullMode::kWholeObject, 0, item.capacity};
        requests[slot].exact = item.exact;
      }
      if (!PullWindow(conn, requests.data(), width, BatchTimeout(), &failure,
                      &timed_out)) {
        conn_ok = false;
        break;
      }
      for (size_t slot = 0; slot < width; ++slot) {
        const PullItem& item = items[done + slot];
        const PullRequest& request = requests[slot];
        (*statuses)[item.index] = request.status;
        if (value_lens)
          (*value_lens)[item.index] =
              request.status != Status::kInvalid || request.too_large
                  ? request.value_len
                  : 0;
        if (request.status == Status::kOk) {
          ++ops;
          bytes += request.data_len;
        }
      }
      done += width;
    }
    if (ops) RecordRailTransfer(conn, false, ops, bytes);
    if (conn_ok) {
      Release(node, lane, conn);
      if (cross_rail_retry)
        cross_rail_retry_successes_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    pull_failures_.fetch_add(1, std::memory_order_relaxed);
    const size_t failed_rail = conn->rail_index;
    const bool local_resource = conn->local_resource_failure;
    // Fence the READs before the destinations can be reused.
    const bool stale_pooled = Destroy(conn, failure);
    if (local_resource) {
      for (size_t i = done; i < items.size(); ++i)
        (*statuses)[items[i].index] = Status::kResourceExhausted;
      return;
    }
    // After a stale pooled connection the failed window is read again;
    // otherwise its keys stay kIOError and only the windows after it, which
    // no READ touched, are tried once more on a fresh connection (another
    // rail when there is one), as each key once had its own connection. Not
    // after a timeout: a dead peer would cost one timeout per window.
    if (!stale_pooled) done += width;
    if (done < items.size() && (stale_pooled || !timed_out)) {
      // Only a stale pooled connection counts as a stale-pool retry.
      const bool retry = PrepareRetry(
          attempt, stale_pooled, failed_rail,
          failure == rdma::RailCompletion::kRailFailure
              ? AcquireFailure::kLocalRail
              : AcquireFailure::kEndpoint,
          ReplaySafety::kReplaySafe, false, peer, node, &excluded,
          &cross_rail_retry);
      if (retry) continue;
      if (!stale_pooled && attempt == 0) {
        // No other rail: the untouched keys get a connection of their own
        // on the same one, as a key read alone would.
        excluded.assign(devs_.size(), 0);
        continue;
      }
    }
    if (cross_rail_retry)
      cross_rail_retry_exhausted_.fetch_add(1, std::memory_order_relaxed);
    return;  // the rest stay kIOError
  }
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
  size_t required_bytes = 0;
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
      required_bytes = std::max(required_bytes, total);
      ++valid_count;
    }
  }
  if (valid_count == 0) return result;

  // Only inline objects contribute to leased connection geometry.
  const bool lease_enabled = inline_put_max_bytes_ != 0;
  bool any_over = false;
  size_t max_inline_len = 0;
  for (size_t i = 0; i < count; ++i) {
    if (bad[i]) continue;
    if (lease_enabled && totals[i] > inline_put_max_bytes_) {
      any_over = true;
    } else {
      max_inline_len = std::max(max_inline_len, totals[i]);
    }
  }

  const auto peer = peer_topologies_->Snapshot(node);
  RailMask excluded(devs_.size(), 0);
  OpState op_state;
  bool cross_rail_retry = false;
  bool out_dev_set = false;
  CompletionFaultOperation completion_fault(&test_completion_fault_calls_);
  for (int attempt = 0; attempt < 2; ++attempt) {
    // A fresh attempt is possible only before any request is posted;
    // validation results survive while connection-local progress restarts.
    completion_fault.BeginAttempt(attempt);
    const bool want_lease = attempt == 0 && any_over && lease_enabled;
    AcquireOptions options;
    options.required_data_bytes = required_bytes;
    options.leased_inline_bytes = max_inline_len;
    options.request_leased_put = want_lease;
    options.force_new = attempt != 0;
    options.requested_credits = 1;
    options.excluded = excluded;
    options.op = &op_state;
    // Bounds this attempt's connection; the windows keep their own timeouts.
    op_state.deadline_us = DeadlineIn(BatchTimeout());
    options.peer = peer;
    AcquireResult acquired = Acquire(node, Lane::kSgData, options);
    if (!acquired.conn) {
      if (PrepareRetry(attempt, acquired.from_pool, acquired.attempted_rail,
                       acquired.failure, ReplaySafety::kUnsafeAfterPost, false,
                       peer, node, &excluded, &cross_rail_retry))
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

    for (size_t item_index = 0; item_index < count && conn_ok; ++item_index) {
      if (bad[item_index] || completed[item_index]) continue;
      const bool use_lease = conn->leased_put &&
                             totals[item_index] > inline_put_max_bytes_;
      const size_t remote_capacity = conn->data_capacity();
      if (!use_lease && totals[item_index] > remote_capacity) {
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

      // Staged-lease items request an exactly-sized receive range first, then
      // aim every window WR at that range. Inline items keep the connection's
      // resident receive slot with immediate 0.
      uint64_t window_base = conn->put_addr(0);
      uint32_t window_rkey = conn->recv_segment.rkey;
      uint32_t window_imm = 0;
      if (use_lease) {
        conn->Encode(ep.sbuf(0), WireOp::kLeasePut, source.key, 0, 0,
                     totals[item_index]);
        std::vector<uint32_t> ready_bytes;
        bool ready_timed_out = false;
        ibv_wc_status ready_status = IBV_WC_SUCCESS;
        bool ready_had_wcs = false;
        bool ready_ok = ep.PostRecv(0) && ep.PostSend(0, kReqPrefix);
        if (ready_ok &&
            !ReapPosted(ep, 1, 1, &ready_bytes, BatchTimeout(),
                        &ready_timed_out, &ready_status, &ready_had_wcs,
                        &completion_fault)) {
          ready_ok = false;
          completion = rdma::ClassifyCompletion(ready_status, ready_had_wcs);
        }
        if (ready_timed_out)
          completion_timeouts_.fetch_add(1, std::memory_order_relaxed);
        if (!ready_ok) {
          conn_ok = false;
          break;
        }
        Status ready_status_code = Status::kIOError;
        uint64_t ready_data_len = 0;
        if (ready_bytes[0] < kRespPrefix ||
            !conn->Decode(ep.rbuf(0), &ready_status_code, &ready_data_len,
                          rdma::kLeasePutReadyBytes) ||
            (ready_status_code == Status::kOk) !=
                (ready_data_len == rdma::kLeasePutReadyBytes)) {
          conn_ok = false;
          completion = rdma::RailCompletion::kEndpointFailure;
          break;
        }
        if (ready_status_code != Status::kOk) {
          result[item_index] = ready_status_code;
          completed[item_index] = 1;
          continue;
        }
        rdma::LeasePutReady ready;
        if (ready_bytes[0] < kRespPrefix + rdma::kLeasePutReadyBytes ||
            !rdma::DecodeLeasePutReady(ep.rbuf(0) + kRespPrefix, &ready) ||
            ready.slot >= static_cast<uint32_t>(conn->depth) ||
            ready.lease_bytes <
                static_cast<uint64_t>(rdma::kV2DataOffset) +
                    totals[item_index]) {
          conn_ok = false;
          completion = rdma::RailCompletion::kEndpointFailure;
          break;
        }
        window_base = ready.write_base + rdma::kV2PutPrefixOffset;
        window_rkey = ready.rkey;
        window_imm = ready.slot;
        ++leaseput_ops_;
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
            NoteLocalRegistrationFailure(conn, payload.second);
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
            window_base +
            (window_index == 0 ? 0 : kReqPrefix + logical_offset);
        if (!ep.PostRecv(0) ||
            !ep.PostWriteImmScatterMulti(
                0, header_len, segments, mrs, remote_addr,
                use_lease ? window_rkey : conn->recv_segment.rkey,
                window_imm)) {
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
      uint64_t successful_ops = 0;
      uint64_t successful_bytes = 0;
      for (size_t i = 0; i < count; ++i) {
        if (!completed[i] || result[i] != Status::kOk) continue;
        ++successful_ops;
        successful_bytes += totals[i];
      }
      RecordRailTransfer(conn, true, successful_ops, successful_bytes);
      Release(node, Lane::kSgData, conn);
      if (cross_rail_retry)
        cross_rail_retry_successes_.fetch_add(1,
                                              std::memory_order_relaxed);
      return result;
    }
    const size_t failed_rail = conn->rail_index;
    const bool local_resource = conn->local_resource_failure;
    const bool stale_pooled = Destroy(conn, completion);
    for (size_t i = 0; i < count; ++i) {
      if (!bad[i] && !completed[i])
        result[i] = local_resource ? Status::kResourceExhausted
                                   : Status::kIOError;
    }
    if (local_resource) {
      if (cross_rail_retry)
        cross_rail_retry_exhausted_.fetch_add(1, std::memory_order_relaxed);
      return result;
    }
    const AcquireFailure failure =
        completion == rdma::RailCompletion::kRailFailure && !request_posted
            ? AcquireFailure::kLocalRail
            : AcquireFailure::kEndpoint;
    if (PrepareRetry(attempt, acquired.from_pool, failed_rail, failure,
                     ReplaySafety::kUnsafeAfterPost,
                     request_posted && !stale_pooled, peer, node,
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

size_t RdmaTransport::ConnectionBound(size_t required) const {
  const size_t logical_max = OpBound();
  size_t wanted = std::max(required, connection_min_block_bytes_);
  if (wanted >= logical_max) return logical_max;
  size_t size_class = 1;
  while (size_class < wanted && size_class <= logical_max / 2)
    size_class <<= 1;
  return size_class < wanted ? logical_max
                             : std::min(size_class, logical_max);
}

size_t RdmaTransport::ConnectionDepth(const std::string& node, Lane lane,
                                      size_t requested_credits) {
  const size_t learned = std::max<size_t>(1, LearnedDepth(node, lane));
  const size_t wanted =
      std::min(learned, std::max<size_t>(1, requested_credits));
  size_t depth_class = 1;
  while (depth_class < wanted && depth_class <= learned / 2)
    depth_class <<= 1;
  return depth_class < wanted ? learned : std::min(depth_class, learned);
}

// Records n as a high-water candidate and reports whether it exceeds the
// logical safety declaration. Physical connection slots use ConnectionBound,
// so a rare large object no longer inflates every data connection.
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
  // Lease sizing changes physical residency, never the operator's explicit
  // logical safety ceiling shared by PUT and GET.
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
  const size_t count = keys.size();
  if (out_lengths) out_lengths->assign(count, 0);
  if (out_dev) out_dev->clear();
  if (destinations.size() != count) return InvalidStatuses(count);
  std::vector<Status> result(count, Status::kInvalid);
  std::vector<PullItem> items;
  items.reserve(count);
  for (size_t item = 0; item < count; ++item) {
    const auto& segments = destinations[item].payloads;
    size_t capacity = 0;
    bool valid = true;
    for (const auto& segment : segments) {
      size_t next = 0;
      if (!ValidBuffer(segment.first, segment.second) ||
          !CheckedAddSize(capacity, segment.second, &next)) {
        valid = false;
        break;
      }
      capacity = next;
    }
    if (!valid || NoteBlock(capacity)) continue;
    items.push_back(
        PullItem{item, segments.data(), segments.size(), capacity});
  }
  std::vector<uint64_t> value_lens(count, 0);
  PullIntoMany(node, keys, items, Lane::kSgData, &result, &value_lens,
               out_dev);
  if (out_lengths) {
    for (const PullItem& item : items) {
      const uint64_t value_len = value_lens[item.index];
      if (result[item.index] == Status::kOk ||
          (result[item.index] == Status::kInvalid &&
           value_len > item.capacity))
        (*out_lengths)[item.index] = static_cast<size_t>(value_len);
    }
  }
  return result;
}

}  // namespace dfkv
