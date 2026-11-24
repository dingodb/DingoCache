#include "cache/remove_memory.h"

#include <algorithm>

#include "common/config_dump.h"
#include "utils/log.h"
#include "utils/net_util.h"

namespace dfkv {

namespace {
constexpr uint64_t kMaxWindowMs = 3600000;
constexpr uint64_t kDefaultEntries = 1048576;
constexpr uint64_t kMinEntries = 1024;
constexpr uint64_t kMaxEntries = 67108864;
}  // namespace

RemoveMemory::RemoveMemory(uint64_t window_ms, size_t capacity,
                           uint64_t now_ms)
    : window_ms_(window_ms), capacity_(capacity), covered_since_ms_(now_ms) {}

std::unique_ptr<RemoveMemory> RemoveMemory::FromEnv(uint64_t now_ms) {
  uint64_t window = config_dump::EnvU64("DFKV_REMOVE_MEMORY_MS", 0);
  if (window == 0) return nullptr;
  if (window > kMaxWindowMs) {
    DFKV_LOG_WARN("DFKV_REMOVE_MEMORY_MS=" + std::to_string(window) +
                  " is above " + std::to_string(kMaxWindowMs) + "; using " +
                  std::to_string(kMaxWindowMs));
    window = kMaxWindowMs;
  }
  uint64_t entries =
      config_dump::EnvU64("DFKV_REMOVE_MEMORY_ENTRIES", kDefaultEntries);
  const uint64_t clamped = std::min(std::max(entries, kMinEntries), kMaxEntries);
  if (clamped != entries) {
    DFKV_LOG_WARN("DFKV_REMOVE_MEMORY_ENTRIES=" + std::to_string(entries) +
                  " is outside " + std::to_string(kMinEntries) + "-" +
                  std::to_string(kMaxEntries) + "; using " +
                  std::to_string(clamped));
    entries = clamped;
  }
  DFKV_LOG_INFO("remove memory: on (" + std::to_string(window) + " ms, " +
                std::to_string(entries) + " entries)");
  return std::make_unique<RemoveMemory>(window, static_cast<size_t>(entries),
                                        now_ms);
}

uint64_t RemoveMemory::NowLocked(uint64_t now_ms) {
  // Callers read the clock before taking the lock: a thread can arrive with
  // an older time than the last record. The table's time never goes back.
  last_now_ms_ = std::max(last_now_ms_, now_ms);
  return last_now_ms_;
}

void RemoveMemory::ExpireLocked(uint64_t now_ms) {
  while (!order_.empty() && now_ms >= order_.front().second &&
         now_ms - order_.front().second > window_ms_) {
    const auto& [key, at] = order_.front();
    const auto it = last_.find(key);
    if (it != last_.end() && it->second == at) last_.erase(it);
    order_.pop_front();
  }
}

void RemoveMemory::Record(const BlockKey& key, uint64_t now_ms) {
  std::lock_guard<std::mutex> lk(mu_);
  now_ms = NowLocked(now_ms);
  ExpireLocked(now_ms);
  while (order_.size() >= capacity_) {
    // Full: drop the oldest record; this node no longer knows what was
    // removed up to and including its time.
    const auto& [old_key, at] = order_.front();
    const auto it = last_.find(old_key);
    if (it != last_.end() && it->second == at) last_.erase(it);
    covered_since_ms_ = std::max(covered_since_ms_, at + 1);
    order_.pop_front();
    dropped_.fetch_add(1, std::memory_order_relaxed);
  }
  order_.emplace_back(key, now_ms);
  last_[key] = now_ms;
}

uint64_t RemoveMemory::CoveredMs(uint64_t now_ms) {
  std::lock_guard<std::mutex> lk(mu_);
  now_ms = NowLocked(now_ms);
  const uint64_t since =
      now_ms > covered_since_ms_ ? now_ms - covered_since_ms_ : 0;
  return std::min(since, window_ms_);
}

uint64_t RemoveMemory::Query(uint64_t tenant_hash, const char* digests,
                             size_t count, uint64_t age_ms, uint64_t now_ms,
                             uint8_t* out) {
  std::lock_guard<std::mutex> lk(mu_);
  now_ms = NowLocked(now_ms);
  ExpireLocked(now_ms);
  const uint64_t since =
      now_ms > covered_since_ms_ ? now_ms - covered_since_ms_ : 0;
  const uint64_t covered = std::min(since, window_ms_);
  const bool known = age_ms <= covered;
  for (size_t i = 0; i < count; ++i) {
    uint8_t answer = kUnknown;
    if (known) {
      const char* d = digests + i * 16;
      const auto it = last_.find(
          BlockKey{net::GetU64(d), net::GetU64(d + 8), tenant_hash});
      answer = it != last_.end() && now_ms - it->second <= age_ms
                   ? kRemoved
                   : kNotRemoved;
    }
    out[i] = answer;
    answers_[answer].fetch_add(1, std::memory_order_relaxed);
  }
  return covered;
}

size_t RemoveMemory::Entries() {
  std::lock_guard<std::mutex> lk(mu_);
  return last_.size();
}

uint64_t RemoveMemory::Dropped() const {
  return dropped_.load(std::memory_order_relaxed);
}

uint64_t RemoveMemory::QueriedKeys(uint8_t answer) const {
  return answer < 3 ? answers_[answer].load(std::memory_order_relaxed) : 0;
}

}  // namespace dfkv
