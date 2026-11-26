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
                           uint64_t now_ms, size_t presize_above)
    : window_ms_(window_ms), capacity_(capacity), covered_since_ms_(now_ms) {
  if (capacity_ <= presize_above) return;
  // Whole at start: neither the ring nor the index ever grows (the index
  // stays below 70 % full with every record a distinct key).
  ring_.resize(capacity_);
  size_t slots = 1024;
  while (10 * capacity_ > 7 * slots) slots *= 2;
  index_.assign(slots, kEmpty);
}

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
    config_dump::RecordResolved("DFKV_REMOVE_MEMORY_ENTRIES",
                                std::to_string(entries));
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

size_t RemoveMemory::SlotLocked(const BlockKey& key) const {
  const size_t mask = index_.size() - 1;
  size_t slot = BlockKeyMixHash()(key) & mask;
  while (index_[slot] != kEmpty && !(ring_[index_[slot]].key == key))
    slot = (slot + 1) & mask;
  return slot;
}

void RemoveMemory::EraseSlotLocked(size_t slot) {
  // Backward-shift deletion: every later entry of the run that may sit at
  // `slot` moves up, so that no probe stops early at a hole.
  const size_t mask = index_.size() - 1;
  size_t hole = slot;
  for (size_t next = (hole + 1) & mask; index_[next] != kEmpty;
       next = (next + 1) & mask) {
    const size_t home = BlockKeyMixHash()(ring_[index_[next]].key) & mask;
    // `next` may move to `hole` unless its home lies cyclically in
    // (hole, next].
    const bool stays = hole <= next ? (hole < home && home <= next)
                                    : (hole < home || home <= next);
    if (stays) continue;
    index_[hole] = index_[next];
    hole = next;
  }
  index_[hole] = kEmpty;
  --live_;
}

void RemoveMemory::RebuildIndexLocked(size_t slots) {
  index_.assign(slots, kEmpty);
  live_ = 0;
  for (size_t i = 0; i < size_; ++i) {  // oldest first: the latest wins
    size_t pos = head_ + i;
    if (pos >= ring_.size()) pos -= ring_.size();
    const size_t slot = SlotLocked(ring_[pos].key);
    if (index_[slot] == kEmpty) ++live_;
    index_[slot] = static_cast<uint32_t>(pos);
  }
}

void RemoveMemory::PopFrontLocked() {
  const size_t slot = SlotLocked(ring_[head_].key);
  if (index_[slot] == head_) EraseSlotLocked(slot);  // its latest record
  if (++head_ == ring_.size()) head_ = 0;
  --size_;
}

void RemoveMemory::PushBackLocked(const BlockKey& key, uint64_t now_ms) {
  if (size_ == ring_.size()) {
    // Grow (never past capacity_, which Record keeps size_ below): lay the
    // ring out from 0 again, which moves every position.
    std::vector<Rec> grown(std::min(capacity_, std::max<size_t>(1024, 2 * size_)));
    for (size_t i = 0; i < size_; ++i) {
      size_t pos = head_ + i;
      if (pos >= ring_.size()) pos -= ring_.size();
      grown[i] = ring_[pos];
    }
    ring_.swap(grown);
    head_ = 0;
    RebuildIndexLocked(std::max<size_t>(index_.size(), 1024));
  }
  size_t pos = head_ + size_;
  if (pos >= ring_.size()) pos -= ring_.size();
  ring_[pos] = Rec{key, now_ms};
  ++size_;
  size_t slot = SlotLocked(key);
  if (index_[slot] == kEmpty) {
    if (10 * (live_ + 1) > 7 * index_.size()) {
      RebuildIndexLocked(2 * index_.size());
      slot = SlotLocked(key);
    }
    if (index_[slot] == kEmpty) ++live_;
  }
  index_[slot] = static_cast<uint32_t>(pos);
}

void RemoveMemory::ExpireLocked(uint64_t now_ms) {
  while (size_ != 0 && now_ms >= ring_[head_].at &&
         now_ms - ring_[head_].at > window_ms_)
    PopFrontLocked();
}

void RemoveMemory::Record(const BlockKey& key, uint64_t now_ms) {
  std::lock_guard<std::mutex> lk(mu_);
  now_ms = NowLocked(now_ms);
  ExpireLocked(now_ms);
  while (size_ >= capacity_) {
    // Full: drop the oldest record; this node no longer knows what was
    // removed up to and including its time.
    covered_since_ms_ = std::max(covered_since_ms_, ring_[head_].at + 1);
    PopFrontLocked();
    dropped_.fetch_add(1, std::memory_order_relaxed);
  }
  PushBackLocked(key, now_ms);
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
      const BlockKey key{net::GetU64(d), net::GetU64(d + 8), tenant_hash};
      const uint32_t pos = index_.empty() ? kEmpty : index_[SlotLocked(key)];
      answer = pos != kEmpty && now_ms - ring_[pos].at <= age_ms
                   ? kRemoved
                   : kNotRemoved;
    }
    out[i] = answer;
    answers_[answer].fetch_add(1, std::memory_order_relaxed);
  }
  return covered;
}

uint64_t RemoveMemory::QueryAges(uint64_t tenant_hash, const char* digests,
                                 size_t count, uint64_t age_ms,
                                 uint64_t now_ms, uint32_t* out) {
  std::lock_guard<std::mutex> lk(mu_);
  return QueryAgesLocked(tenant_hash, digests, count, age_ms,
                         NowLocked(now_ms), out);
}

uint64_t RemoveMemory::QueryAgesLocked(uint64_t tenant_hash, const char* digests,
                                       size_t count, uint64_t age_ms,
                                       uint64_t now_ms, uint32_t* out) {
  ExpireLocked(now_ms);
  const uint64_t since =
      now_ms > covered_since_ms_ ? now_ms - covered_since_ms_ : 0;
  const uint64_t covered = std::min(since, window_ms_);
  const bool known = age_ms <= covered;
  for (size_t i = 0; i < count; ++i) {
    uint32_t age = kAgeUnknown;
    if (known) {
      const char* d = digests + i * 16;
      const BlockKey key{net::GetU64(d), net::GetU64(d + 8), tenant_hash};
      const uint32_t pos = index_.empty() ? kEmpty : index_[SlotLocked(key)];
      age = kAgeNone;
      if (pos != kEmpty && now_ms - ring_[pos].at <= age_ms)
        age = static_cast<uint32_t>(
            std::min<uint64_t>(now_ms - ring_[pos].at, kAgeUnknown - 1));
    }
    out[i] = age;
    // dfkv_removed_within_queries_total is the per-key op-25 metric. Op 26
    // counts requests outside this table, not these keys a second time.
  }
  return covered;
}

void RemoveMemory::Clear(uint64_t now_ms) {
  std::lock_guard<std::mutex> lk(mu_);
  now_ms = NowLocked(now_ms);
  head_ = size_ = 0;
  std::fill(index_.begin(), index_.end(), kEmpty);
  live_ = 0;
  covered_since_ms_ = std::max(covered_since_ms_, now_ms);
}

void RemoveMemory::AdvanceTo(uint64_t now_ms) {
  std::lock_guard<std::mutex> lk(mu_);
  NowLocked(now_ms);
}

size_t RemoveMemory::Entries() {
  std::lock_guard<std::mutex> lk(mu_);
  return live_;
}

size_t RemoveMemory::RingSlotsForTest() {
  std::lock_guard<std::mutex> lk(mu_);
  return ring_.size();
}

size_t RemoveMemory::IndexSlotsForTest() {
  std::lock_guard<std::mutex> lk(mu_);
  return index_.size();
}

uint64_t RemoveMemory::Dropped() const {
  return dropped_.load(std::memory_order_relaxed);
}

uint64_t RemoveMemory::QueriedKeys(uint8_t answer) const {
  return answer < 3 ? answers_[answer].load(std::memory_order_relaxed) : 0;
}

}  // namespace dfkv
