#include "cache/write_memory.h"

#include <algorithm>
#include <string>

#include "common/block_key_hash.h"
#include "common/config_dump.h"
#include "common/raw_clock.h"
#include "utils/log.h"
#include "utils/net_util.h"

namespace dfkv {

namespace {
constexpr uint64_t kMinEntries = 1024;
constexpr uint64_t kMaxEntries = 67108864;
constexpr uint64_t kSampleEveryMs = 1000;
}  // namespace

WriteMemory::Offsets WriteMemory::SystemOffsets() {
  // RAW read between the two others: each offset is off by at most the
  // reads' few hundred nanoseconds.
  const int64_t real = ClockNs(CLOCK_REALTIME);
  const int64_t raw = ClockNs(CLOCK_MONOTONIC_RAW);
  const int64_t boot = ClockNs(CLOCK_BOOTTIME);
  return Offsets{(real - raw) / 1000000, (boot - raw) / 1000000};
}

WriteMemory::WriteMemory(uint64_t window_ms, size_t capacity, uint64_t now_ms,
                         OffsetsFn offsets, uint64_t tick_ms)
    : window_ms_(window_ms), offsets_(std::move(offsets)),
      last_now_ms_(now_ms) {
  const size_t per_shard = std::max<size_t>(1, (capacity + kShards - 1) / kShards);
  for (size_t i = 0; i < kShards; ++i)
    shards_.push_back(std::make_unique<RemoveMemory>(
        window_ms, per_shard, now_ms, RemoveMemory::kPresizeAbove / kShards));
  const Offsets cur = offsets_();
  samples_.push_back({now_ms, cur});
  min_ = max_ = cur;
  if (tick_ms > 0)
    tick_ = std::thread([this, tick_ms] { TickLoop(tick_ms); });
}

WriteMemory::~WriteMemory() {
  {
    std::lock_guard<std::mutex> lk(tick_mu_);
    stop_ = true;
  }
  tick_cv_.notify_all();
  if (tick_.joinable()) tick_.join();
}

std::unique_ptr<WriteMemory> WriteMemory::FromEnv(uint64_t window_ms,
                                                  uint64_t now_ms) {
  uint64_t entries = config_dump::EnvU64("DFKV_WRITE_MEMORY_ENTRIES", 0);
  if (entries == 0) return nullptr;
  if (window_ms == 0) {
    config_dump::RecordResolved("DFKV_WRITE_MEMORY_ENTRIES", "0");
    DFKV_LOG_WARN("DFKV_WRITE_MEMORY_ENTRIES needs DFKV_REMOVE_MEMORY_MS (its "
                  "window); the recent-PUT memory stays off");
    return nullptr;
  }
  const uint64_t clamped = std::min(std::max(entries, kMinEntries), kMaxEntries);
  if (clamped != entries) {
    DFKV_LOG_WARN("DFKV_WRITE_MEMORY_ENTRIES=" + std::to_string(entries) +
                  " is outside " + std::to_string(kMinEntries) + "-" +
                  std::to_string(kMaxEntries) + "; using " +
                  std::to_string(clamped));
    entries = clamped;
    config_dump::RecordResolved("DFKV_WRITE_MEMORY_ENTRIES",
                                std::to_string(entries));
  }
  DFKV_LOG_INFO("write memory: on (" + std::to_string(window_ms) + " ms, " +
                std::to_string(entries) + " entries)");
  return std::make_unique<WriteMemory>(window_ms, static_cast<size_t>(entries),
                                       now_ms);
}

RemoveMemory& WriteMemory::ShardOf(const BlockKey& key) {
  // The high bits: the shard's own index uses the low ones.
  return *shards_[BlockKeyMixHash()(key) >> 60];
}

void WriteMemory::Record(const BlockKey& key, uint64_t now_ms) {
  // A write (or the ack of a concurrent earlier write) begins a fresh epoch:
  // old samples and entries cannot prove anything after a possible pause.
  std::lock_guard<std::mutex> lk(check_mu_);
  CheckLocked(now_ms);
  ShardOf(key).Record(key, now_ms);
}

void WriteMemory::RestartLocked(uint64_t now_ms, const Offsets& cur) {
  samples_.clear();
  samples_.push_back({now_ms, cur});
  min_ = max_ = cur;
}

void WriteMemory::CheckLocked(uint64_t now_ms) {
  // Callers sample RAW before waiting for this lock. Serialize that domain
  // time once, then advance every shard's known-completed-write boundary:
  // a later Check may establish that a previously recorded PUT is no longer
  // recent, instead of only knowing up to that PUT's older caller timestamp.
  now_ms = std::max(last_now_ms_, now_ms);
  last_now_ms_ = now_ms;
  for (auto& shard : shards_) shard->AdvanceTo(now_ms);
  const Offsets cur = offsets_();
  // The window's samples, and the newest one before it as its start.
  bool popped = false;
  while (samples_.size() >= 2 && samples_[1].at + window_ms_ <= now_ms) {
    samples_.pop_front();
    popped = true;
  }
  if (popped) {
    min_ = max_ = samples_.front().off;
    for (const Sample& s : samples_) {
      min_.realtime = std::min(min_.realtime, s.off.realtime);
      max_.realtime = std::max(max_.realtime, s.off.realtime);
      min_.boottime = std::min(min_.boottime, s.off.boottime);
      max_.boottime = std::max(max_.boottime, s.off.boottime);
    }
  }
  const int64_t real = std::max(max_.realtime, cur.realtime) -
                       std::min(min_.realtime, cur.realtime);
  const int64_t boot = std::max(max_.boottime, cur.boottime) -
                       std::min(min_.boottime, cur.boottime);
  if (real > kPauseGuardMs || boot > kPauseGuardMs) {
    // The raw clock may have stopped while time passed: every age held
    // could be short by that much. Each shard's time moves to now_ms, so a
    // PUT racing the clear is stamped after it.
    for (auto& shard : shards_) shard->Clear(now_ms);
    const ClearReason reason = real > kPauseGuardMs ? kRealtime : kBoottime;
    cleared_[reason].fetch_add(1, std::memory_order_relaxed);
    DFKV_LOG_WARN(std::string("write memory: cleared (CLOCK_") +
                  (reason == kRealtime ? "REALTIME" : "BOOTTIME") +
                  " moved " + std::to_string(reason == kRealtime ? real : boot) +
                  " ms against CLOCK_MONOTONIC_RAW within the window)");
    RestartLocked(now_ms, cur);
    return;
  }
  if (Entries() == 0) {
    // Reset the sample epoch only after every shard has advanced to this
    // Check. A first PUT after here must be recorded before an overlapping
    // op-26 answer; the publication lock supplies that ordering.
    for (auto& shard : shards_) shard->AdvanceTo(now_ms);
    if (Entries() == 0) {
      RestartLocked(now_ms, cur);
      return;
    }
  }
  if (now_ms >= samples_.back().at + kSampleEveryMs) {
    samples_.push_back({now_ms, cur});
    min_.realtime = std::min(min_.realtime, cur.realtime);
    max_.realtime = std::max(max_.realtime, cur.realtime);
    min_.boottime = std::min(min_.boottime, cur.boottime);
    max_.boottime = std::max(max_.boottime, cur.boottime);
  }
}

void WriteMemory::Check(uint64_t now_ms) {
  std::lock_guard<std::mutex> lk(check_mu_);
  CheckLocked(now_ms);
}

void WriteMemory::Ages(uint64_t tenant_hash, const char* digests,
                       size_t count, uint64_t age_ms, uint64_t now_ms,
                       uint32_t* out) {
  std::lock_guard<std::mutex> lk(check_mu_);
  CheckLocked(now_ms);
  AgesLocked(tenant_hash, digests, count, age_ms, now_ms, out);
}

void WriteMemory::AgesLocked(uint64_t tenant_hash, const char* digests,
                             size_t count, uint64_t age_ms, uint64_t now_ms,
                             uint32_t* out) {
  const uint64_t clears = Cleared(kRealtime) + Cleared(kBoottime);
  for (size_t i = 0; i < count; ++i) {
    const char* d = digests + i * 16;
    const BlockKey key{net::GetU64(d), net::GetU64(d + 8), tenant_hash};
    // W needs no coverage: a write older than what a shard holds is just
    // not there (no recent write).
    RemoveMemory& shard = ShardOf(key);
    shard.QueryAges(tenant_hash, d, 1,
                    std::min(age_ms, shard.CoveredMs(now_ms)), now_ms, &out[i]);
    if (out[i] == RemoveMemory::kAgeUnknown) out[i] = RemoveMemory::kAgeNone;
  }
  // Checked again after the reading: a stop between the two checks clears
  // the table and voids these ages.
  CheckLocked(now_ms);
  if (Cleared(kRealtime) + Cleared(kBoottime) != clears)
    std::fill(out, out + count, RemoveMemory::kAgeNone);
}

uint64_t WriteMemory::AgesWithRemovals(
    RemoveMemory& removes, uint64_t tenant_hash, const char* digests,
    size_t count, uint64_t age_ms, uint64_t now_ms, uint32_t* removed,
    uint32_t* written) {
  // The only nested M/W lock order. PUT and REMOVE take just one of these
  // table locks after their publication bucket, so neither can invert it.
  std::lock_guard<std::mutex> remove_lock(removes.mu_);
  std::lock_guard<std::mutex> write_lock(check_mu_);
  now_ms = std::max({now_ms, removes.last_now_ms_, last_now_ms_});
  CheckLocked(now_ms);
  now_ms = last_now_ms_;
  removes.NowLocked(now_ms);
  const uint64_t covered = removes.QueryAgesLocked(
      tenant_hash, digests, count, age_ms, now_ms, removed);
  AgesLocked(tenant_hash, digests, count, age_ms, now_ms, written);
  return covered;
}

void WriteMemory::TickLoop(uint64_t tick_ms) {
  std::unique_lock<std::mutex> lk(tick_mu_);
  while (!tick_cv_.wait_for(lk, std::chrono::milliseconds(tick_ms),
                            [this] { return stop_; })) {
    lk.unlock();
    Check(RawMs());
    lk.lock();
  }
}

size_t WriteMemory::Entries() {
  size_t n = 0;
  for (auto& shard : shards_) n += shard->Entries();
  return n;
}

uint64_t WriteMemory::Dropped() const {
  uint64_t n = 0;
  for (const auto& shard : shards_) n += shard->Dropped();
  return n;
}

uint64_t WriteMemory::Cleared(ClearReason reason) const {
  return cleared_[reason].load(std::memory_order_relaxed);
}

}  // namespace dfkv
