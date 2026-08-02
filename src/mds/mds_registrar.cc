#include "mds/mds_registrar.h"

#include <unistd.h>
#include <algorithm>

#include <chrono>
#include <utility>
#include <vector>

#include "common/status.h"
#include "mds/mds_proto.h"
#include "utils/log.h"
#include "utils/net_util.h"
#include "utils/prom_escape.h"
#include "utils/thread_name.h"
#include "utils/wire_limits.h"
#include "transport/wire.h"

namespace dfkv {

MdsRegistrar::MdsRegistrar(std::vector<std::string> mds_eps, std::string group,
                           MemberInfo self, int heartbeat_ms, int io_timeout_ms,
                           bool is_client, int first_registration_timeout_ms)
    : eps_(std::move(mds_eps)),
      group_(std::move(group)),
      self_(std::move(self)),
      hb_ms_(heartbeat_ms),
      io_ms_(io_timeout_ms),
      is_client_(is_client),
      first_registration_timeout_ms_(first_registration_timeout_ms) {}

MdsRegistrar::~MdsRegistrar() { Stop(); }

uint64_t MdsRegistrar::NowMs() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool MdsRegistrar::SendOnce(uint8_t op, int io_timeout_ms,
                            uint64_t deadline_ms) {
  const uint64_t started_ms = NowMs();
  std::string ep = eps_.Pick(started_ms);
  if (ep.empty()) return false;
  int fd = net::Dial(ep, io_timeout_ms, io_timeout_ms);
  if (fd < 0) {
    eps_.MarkFailed(ep, started_ms);
    return false;
  }

  // Re-arm socket I/O before each phase with the absolute startup deadline's
  // remaining budget. Without this, connect + write + read could each consume
  // the full relative timeout and overrun the configured registration deadline.
  auto arm_deadline = [&] {
    if (deadline_ms == 0) return true;
    const uint64_t now = NowMs();
    if (now >= deadline_ms) return false;
    const uint64_t remaining = deadline_ms - now;
    timeval timeout{static_cast<time_t>(remaining / 1000),
                    static_cast<suseconds_t>((remaining % 1000) * 1000)};
    return ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                        sizeof(timeout)) == 0 &&
           ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout,
                        sizeof(timeout)) == 0;
  };

  if (stats_fn_) {  // refresh dynamic stats for this beat (STA1)
    self_.stats = stats_fn_();
    self_.has_stats = true;
  }
  std::string payload = EncodeMemberReq(group_, self_);
  char pre[kReqPrefix];
  EncodeReq(pre, static_cast<WireOp>(op), BlockKey{}, 0, 0, payload.size());
  bool ok = arm_deadline() && net::WriteAll(fd, pre, kReqPrefix) &&
            arm_deadline() &&
            net::WriteAll(fd, payload.data(), payload.size());
  if (ok) {
    char rp[kRespPrefix];
    Status st = Status::kInvalid;
    uint64_t dlen = 0;
    ok = arm_deadline() && net::ReadAll(fd, rp, kRespPrefix) &&
         DecodeResp(rp, &st, &dlen, wire_limits::kMdsMaxRespData) &&
         st == Status::kOk;
    if (ok && dlen) {
      std::vector<char> d(dlen);
      ok = arm_deadline() && net::ReadAll(fd, d.data(), dlen);
    }
  }
  ::close(fd);
  if (ok)
    eps_.MarkOk(ep);
  else
    eps_.MarkFailed(ep, started_ms);
  return ok;
}

bool MdsRegistrar::RegisterOnceBefore(int io_timeout_ms,
                                      uint64_t deadline_ms) {
  const bool ok =
      SendOnce(static_cast<uint8_t>(is_client_ ? WireOp::kClientRegister
                                               : WireOp::kRegister),
               io_timeout_ms, deadline_ms);
  const uint64_t now = NowMs();
  if (!ok || (deadline_ms != 0 && now > deadline_ms)) return false;
  last_success_ms_.store(now, std::memory_order_relaxed);
  if (!registered_.exchange(true, std::memory_order_acq_rel) &&
      registered_fn_) {
    registered_fn_();
  }
  return true;
}

bool MdsRegistrar::RegisterOnce() {
  return RegisterOnceBefore(io_ms_, /*deadline_ms=*/0);
}

bool MdsRegistrar::HeartbeatOnce() {
  const bool ok =
      SendOnce(static_cast<uint8_t>(is_client_ ? WireOp::kClientHeartbeat
                                               : WireOp::kHeartbeat),
               io_ms_, /*deadline_ms=*/0);
  if (registered()) ReportHeartbeatResult(ok);
  return ok;
}

void MdsRegistrar::ReportHeartbeatResult(bool ok) {
  const uint64_t now = NowMs();
  if (ok) {
    last_success_ms_.store(now, std::memory_order_relaxed);
    const uint64_t failed =
        heartbeat_failures_consecutive_.exchange(0, std::memory_order_relaxed);
    last_failure_log_ms_.store(0, std::memory_order_relaxed);
    if (failed != 0) {
      DFKV_LOG_INFO("mds-registrar: heartbeat recovered group=" + group_ +
                    " id=" + self_.id + " after " +
                    std::to_string(failed) + " consecutive failure(s)");
    }
    return;
  }

  const uint64_t consecutive =
      heartbeat_failures_consecutive_.fetch_add(1, std::memory_order_relaxed) +
      1;
  heartbeat_failures_total_.fetch_add(1, std::memory_order_relaxed);
  const uint64_t last_log =
      last_failure_log_ms_.load(std::memory_order_relaxed);
  if (consecutive != 1 && now >= last_log &&
      now - last_log < kFailureLogIntervalMs)
    return;
  last_failure_log_ms_.store(now, std::memory_order_relaxed);
  DFKV_LOG_WARN("mds-registrar: heartbeat failed group=" + group_ +
                " id=" + self_.id + " mds=" + eps_.Join() + " (" +
                std::to_string(consecutive) +
                " consecutive); startup readiness remains latched");
}

uint64_t MdsRegistrar::last_success_age_seconds() const {
  const uint64_t last = last_success_ms_.load(std::memory_order_relaxed);
  if (last == 0) return 0;
  const uint64_t now = NowMs();
  return now >= last ? (now - last) / 1000 : 0;
}

std::string MdsRegistrar::MetricsText() const {
  const std::string labels =
      "{group=\"" + PromLabelEscape(group_) + "\",node=\"" +
      PromLabelEscape(self_.id) + "\"}";
  std::string s;
  auto metric = [&](const char* name, const char* type, const char* help,
                    uint64_t value) {
    s += "# HELP "; s += name; s += " "; s += help; s += "\n";
    s += "# TYPE "; s += name; s += " "; s += type; s += "\n";
    s += name; s += labels; s += " "; s += std::to_string(value); s += "\n";
  };
  metric("dfkv_mds_registration_latched", "gauge",
         "Whether this process has completed its first MDS registration",
         registered() ? 1 : 0);
  metric("dfkv_mds_first_registration_timeout_ms", "gauge",
         "Configured first-registration deadline (0 means unbounded)",
         first_registration_timeout_ms_ > 0
             ? static_cast<uint64_t>(first_registration_timeout_ms_) : 0);
  metric("dfkv_mds_first_registration_timed_out", "gauge",
         "Whether the first-registration deadline expired",
         first_registration_timed_out() ? 1 : 0);
  metric("dfkv_mds_heartbeat_healthy", "gauge",
         "Whether the latest post-registration MDS heartbeat succeeded",
         heartbeat_healthy() ? 1 : 0);
  metric("dfkv_mds_heartbeat_failures_consecutive", "gauge",
         "Current consecutive post-registration MDS heartbeat failures",
         heartbeat_failures_consecutive());
  metric("dfkv_mds_heartbeat_failures_total", "counter",
         "Total post-registration MDS heartbeat failures",
         heartbeat_failures_total());
  metric("dfkv_mds_last_success_age_seconds", "gauge",
         "Age of the latest successful registration or heartbeat",
         last_success_age_seconds());
  return s;
}

bool MdsRegistrar::WaitMs(int ms) {
  std::unique_lock<std::mutex> lk(mu_);
  return cv_.wait_for(lk, std::chrono::milliseconds(ms),
                      [this] { return !running_.load(); });
}

void MdsRegistrar::MarkFirstRegistrationTimedOut() {
  if (first_registration_timed_out_.exchange(true, std::memory_order_acq_rel))
    return;
  DFKV_LOG_ERROR(
      "mds-registrar: first registration deadline expired group=" + group_ +
      " id=" + self_.id + " mds=" + eps_.Join() + " timeout_ms=" +
      std::to_string(first_registration_timeout_ms_));
}

void MdsRegistrar::Loop() {
  const uint64_t deadline =
      first_registration_timeout_ms_ > 0
          ? NowMs() + static_cast<uint64_t>(first_registration_timeout_ms_)
          : 0;
  while (running_.load()) {
    const uint64_t now = NowMs();
    if (deadline != 0 && now >= deadline) {
      MarkFirstRegistrationTimedOut();
      return;
    }
    const int attempt_timeout =
        deadline == 0
            ? io_ms_
            : static_cast<int>(std::min<uint64_t>(
                  static_cast<uint64_t>(io_ms_), deadline - now));
    if (RegisterOnceBefore(attempt_timeout, deadline)) break;
    const uint64_t after = NowMs();
    if (deadline != 0 && after >= deadline) {
      MarkFirstRegistrationTimedOut();
      return;
    }
    const int retry_wait =
        deadline == 0
            ? 1000
            : static_cast<int>(
                  std::min<uint64_t>(1000, deadline - after));
    if (WaitMs(retry_wait)) return;
  }
  while (running_.load()) {
    if (WaitMs(hb_ms_)) return;
    HeartbeatOnce();
  }
}

void MdsRegistrar::Start() {
  if (running_.exchange(true)) return;
  first_registration_timed_out_.store(false, std::memory_order_release);
  th_ = std::thread([this] { NameThisThread(is_client_ ? "mds-reg-cli" : "mds-reg"); Loop(); });
}

void MdsRegistrar::Stop() {
  if (!running_.exchange(false)) return;
  cv_.notify_all();
  if (th_.joinable()) th_.join();
}

}  // namespace dfkv
