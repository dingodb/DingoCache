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
// Mass-shrink suspicion threshold, percent of the adopted baseline that must
// disappear IN ONE POLL to trigger hysteresis. 50 clears any realistic
// same-poll failure burst (a 64-node ring trips only below 32 survivors) while
// catching the etcd NOSPACE / lease-mass-expiry signature. 0 disables.
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

MdsMemberPoller::MdsMemberPoller(std::vector<std::string> mds_eps, std::string group,
                                 OnChange cb, int poll_ms, int io_timeout_ms)
    : eps_(std::move(mds_eps)),
      group_(std::move(group)),
      cb_(std::move(cb)),
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
  uint64_t now = NowMs();
  std::string ep = eps_.Pick(now);
  if (ep.empty()) return false;
  int fd = net::Dial(ep, io_ms_, io_ms_);
  if (fd < 0) { eps_.MarkFailed(ep, now); return false; }
  char pre[kReqPrefix];
  EncodeReq(pre, WireOp::kListMembers, BlockKey{}, 0, 0, group_.size());
  bool ok = net::WriteAll(fd, pre, kReqPrefix) &&
            (group_.empty() || net::WriteAll(fd, group_.data(), group_.size()));
  std::string data;
  if (ok) {
    char rp[kRespPrefix];
    Status st = Status::kInvalid;
    uint64_t dlen = 0;
    ok = net::ReadAll(fd, rp, kRespPrefix) && DecodeResp(rp, &st, &dlen, wire_limits::kMdsMaxRespData) &&
         st == Status::kOk;
    if (ok) { data.resize(dlen); ok = (dlen == 0) || net::ReadAll(fd, &data[0], dlen); }
  }
  ::close(fd);
  if (!ok) { eps_.MarkFailed(ep, now); return false; }
  eps_.MarkOk(ep);
  return DecodeMembers(data.data(), data.size(), out, epoch);
}

bool MdsMemberPoller::PollOnce() {
  std::vector<MemberInfo> ms;
  uint64_t epoch = 0;
  if (!Query(&ms, &epoch)) return false;  // failed RPC never clears the ring

  // Adoption guard (MemberViewGuard): a *successful* response that is empty OR
  // sheds most of the adopted baseline at once is almost always etcd-outage
  // recovery (mass lease expiry), not a genuine teardown; adopting it remaps
  // the whole cluster into a miss storm and remaps AGAIN when the members
  // re-register. Suspicious views must persist for kViewsToAccept polls.
  // Rejecting must not advance the placement-content epoch — eventual adoption
  // of that same content would otherwise be suppressed by epoch dedup below.
  if (!guard_.Admit(ms.size()))
    return true;  // keep the last ring; do NOT invoke on_change

  if (!have_epoch_ || epoch != last_epoch_) {
    last_epoch_ = epoch;
    have_epoch_ = true;
    if (!ms.empty()) ever_adopted_ = true;
    cb_(ms);
  }
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
  // Query failed: no MDS endpoint answered ListMembers (unreachable or RPC
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
