#include "mds/mds_member_poller.h"

#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <utility>

#include "common/status.h"
#include "utils/log.h"
#include "utils/net_util.h"
#include "utils/thread_name.h"
#include "utils/wire_limits.h"
#include "transport/wire.h"

namespace dfkv {

namespace {
// Suspicion threshold, percent of the guard's trusted REFERENCE (not the last
// adopted hop) a view must deviate by — shrink or growth — to trigger
// hysteresis. Anchoring on a reference frozen while under suspicion is what
// stops diffuse mass expiry (heartbeat phases spread over polls) from
// ratcheting the ring down hop by hop, and the symmetric growth arm stops a
// re-registration ramp from rebuilding the ring once per epoch. 50 clears
// any realistic same-poll failure burst (a 64-node ring trips only below 32
// survivors or above 96 members) while catching the etcd NOSPACE /
// lease-mass-expiry signature and its recovery mirror. 0 disables both
// deviation arms; the empty arm always stays on.
int ShrinkGuardPct() {
  // NOTE: this is a CLIENT-side MDS-discovery knob, read when the poller is
  // built during StartMdsDiscovery — after the client's startup config dump has
  // already emitted. It is therefore not recorded here (the record would land
  // in an already-cleared registry). If set via env it still shows in the dump
  // via Emit()'s environ scan; only its unset default is not surfaced.
  if (const char* v = std::getenv("DFKV_MDS_SHRINK_GUARD_PCT")) {
    const long p = std::atol(v);
    if (p >= 0 && p <= 99) return static_cast<int>(p);
  }
  return 50;
}
}  // namespace

MdsMemberPoller::MdsMemberPoller(std::vector<std::string> mds_eps,
                                 std::string group, OnChange cb, int poll_ms,
                                 int io_timeout_ms)
    : eps_(std::move(mds_eps)),
      group_(std::move(group)),
      cb_(std::move(cb)),
      poll_ms_(poll_ms),
      io_ms_(io_timeout_ms),
      guard_(kViewsToAccept, ShrinkGuardPct()) {}

MdsMemberPoller::MdsMemberPoller(std::vector<std::string> mds_eps,
                                 std::string group, OnTopologyChange cb,
                                 int poll_ms, int io_timeout_ms)
    : eps_(std::move(mds_eps)),
      group_(std::move(group)),
      topology_cb_(std::move(cb)),
      poll_ms_(poll_ms),
      io_ms_(io_timeout_ms),
      guard_(kViewsToAccept, ShrinkGuardPct()) {}

MdsMemberPoller::~MdsMemberPoller() { Stop(); }

uint64_t MdsMemberPoller::NowMs() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool MdsMemberPoller::Query(std::vector<MemberInfo>* out, uint64_t* epoch) {
  const uint64_t now = NowMs();
  const std::string ep = eps_.Pick(now);
  if (ep.empty()) return false;

  auto request = [&](WireOp op, Status* status, std::string* data) {
    const int fd = net::Dial(ep, io_ms_, io_ms_);
    if (fd < 0) return false;

    char pre[kReqPrefix];
    EncodeReq(pre, op, BlockKey{}, 0, 0, group_.size());
    bool ok =
        net::WriteAll(fd, pre, kReqPrefix) &&
        (group_.empty() ||
         net::WriteAll(fd, group_.data(), group_.size()));
    uint64_t data_len = 0;
    if (ok) {
      char response[kRespPrefix];
      ok = net::ReadAll(fd, response, kRespPrefix) &&
           DecodeResp(response, status, &data_len,
                      wire_limits::kMdsMaxRespData);
    }
    if (ok && *status == Status::kOk) {
      data->resize(data_len);
      ok = data_len == 0 || net::ReadAll(fd, &(*data)[0], data_len);
    }
    ::close(fd);
    return ok;
  };

  Status status = Status::kInvalid;
  std::string data;
  bool legacy_members = false;
  bool ok = request(WireOp::kListTopology, &status, &data);
  if (ok && status == Status::kInvalid) {
    // kListTopology was appended after the legacy placement RPC. An old MDS
    // reports an unknown operation as kInvalid; retry that same selected
    // endpoint with kListMembers. No transport/framing failure reaches this
    // branch, so an unhealthy modern endpoint still participates in failover.
    data.clear();
    ok = request(WireOp::kListMembers, &status, &data);
    legacy_members = ok && status == Status::kOk;
  }
  if (!ok || status != Status::kOk ||
      !DecodeMembers(data.data(), data.size(), out, epoch)) {
    eps_.MarkFailed(ep, now);
    return false;
  }
  if (legacy_members) {
    // ListMembers is a placement view and may omit degraded peers, so it can
    // never prove complete rail topology even if a transitional MDS happens to
    // append HLT1. Preserve legacy placement while making explicit-tier
    // transports fail closed. Its wire epoch belongs to the placement view;
    // publish the canonical topology generation expected by modern consumers.
    for (auto& member : *out) {
      member.health = MemberHealth{};
      member.has_health = false;
    }
    *epoch = MembersTopologyEpoch(*out);
  }
  eps_.MarkOk(ep);
  return true;
}

bool MdsMemberPoller::PollOnce() {
  std::vector<MemberInfo> topology;
  uint64_t topology_epoch = 0;
  if (!Query(&topology, &topology_epoch))
    return false;  // failed RPC never clears the ring

  std::vector<MemberInfo> placement;
  placement.reserve(topology.size());
  for (const auto& member : topology) {
    if (member.RingEligible()) placement.push_back(member);
  }
  const uint64_t placement_epoch = MembersEpoch(placement);

  // The placement guard still judges the eligible view, exactly as it did when
  // kListMembers filtered degraded members on the server. A rejected placement
  // never advances its epoch, so a persistent view can be adopted later.
  // Topology generations are independent: HLT1 changes must reach the modern
  // callback immediately even while the corresponding placement is held.
  const bool placement_admitted = guard_.Admit(placement.size());
  const bool topology_changed =
      !have_topology_epoch_ || topology_epoch != last_topology_epoch_;
  if (topology_changed) {
    last_topology_epoch_ = topology_epoch;
    have_topology_epoch_ = true;
  }

  bool placement_changed = false;
  if (placement_admitted &&
      (!have_placement_epoch_ || placement_epoch != last_placement_epoch_)) {
    last_placement_epoch_ = placement_epoch;
    have_placement_epoch_ = true;
    placement_changed = true;
    if (!placement.empty()) ever_adopted_ = true;
  }

  if (topology_cb_ && (topology_changed || placement_changed))
    topology_cb_(topology, topology_epoch, placement_admitted);
  if (cb_ && placement_changed) cb_(placement);
  return true;
}

void MdsMemberPoller::ReportHealth(bool query_ok) {
  if (query_ok) {
    reachable_.store(true, std::memory_order_relaxed);
    if (consec_fail_ > 0) {
      DFKV_LOG_INFO("mds-poll: MDS reachable again for group=" + group_ +
                    " after " + std::to_string(consec_fail_) + " failed poll(s)");
      consec_fail_ = 0;
      last_fail_log_ms_ = 0;
    }
    return;
  }
  // Query failed: no MDS endpoint answered ListTopology (unreachable or RPC
  // error). The ring is intentionally NOT cleared, but if it was never
  // populated the client routes every op to an empty ring and silently returns
  // ok=0 — the exact trap where a mistyped mds_endpoint looks like a write/data
  // failure. Log rate-limited so a sustained outage stays a heartbeat.
  ++consec_fail_;
  unreachable_total_.fetch_add(1, std::memory_order_relaxed);
  reachable_.store(false, std::memory_order_relaxed);
  const uint64_t now = NowMs();
  const bool first = (consec_fail_ == 1);
  if (!first && (now - last_fail_log_ms_) < kFailLogEveryMs) return;
  last_fail_log_ms_ = now;
  const std::string base =
      "mds-poll: cannot reach MDS for group=" + group_ + " (mds=" + eps_.Join() +
      ", " + std::to_string(consec_fail_) + " consecutive failed poll(s))";
  if (!ever_adopted_)
    DFKV_LOG_WARN(base +
                  " — ring is EMPTY (never populated); all puts/gets route to "
                  "nowhere and report ok=0. Check mds_endpoints reachability.");
  else
    DFKV_LOG_WARN(base + " — serving stale ring (last known members).");
}

bool MdsMemberPoller::WaitMs(int ms) {
  std::unique_lock<std::mutex> lk(mu_);
  return cv_.wait_for(lk, std::chrono::milliseconds(ms),
                      [this] { return !running_.load(); });
}

void MdsMemberPoller::Loop() {
  while (running_.load()) {
    ReportHealth(PollOnce());
    if (WaitMs(poll_ms_)) return;
  }
}

void MdsMemberPoller::Start() {
  if (running_.exchange(true)) return;
  th_ = std::thread([this] { NameThisThread("mds-poll"); Loop(); });
}

void MdsMemberPoller::Stop() {
  if (!running_.exchange(false)) return;
  cv_.notify_all();
  if (th_.joinable()) th_.join();
}

}  // namespace dfkv
