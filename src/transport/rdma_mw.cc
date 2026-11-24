#include "transport/rdma_mw.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <string>

#include "utils/log.h"

namespace dfkv::rdma {
namespace {

uint64_t NowUs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

}  // namespace

uint64_t MwQuarantineUs(size_t bytes, uint64_t rate_mbps, uint64_t floor_us,
                        uint64_t margin_us) {
  if (rate_mbps == 0) rate_mbps = 10000;
  // bits / (Mbit/s) = microseconds, rounded up; twice the wire time.
  const uint64_t wire_us =
      (static_cast<uint64_t>(bytes) * 8 + rate_mbps - 1) / rate_mbps;
  return std::max(floor_us, 2 * wire_us + margin_us);
}

void BindRegionRegistry::Start(size_t rails, RegisterFn reg,
                               DeregisterFn dereg, uint64_t arena_cap) {
  if (started_) return;
  register_ = std::move(reg);
  deregister_ = std::move(dereg);
  arena_cap_ = arena_cap;
  rails_.assign(rails, {});
  stopping_ = false;
  started_ = true;
  thread_ = std::thread([this] { Run(); });
}

uint64_t BindRegionRegistry::BytesLocked(size_t rail, Kind kind) const {
  uint64_t bytes = 0;
  for (const auto& [base, entry] : rails_[rail])
    if (entry.kind == kind && entry.state != State::kFailed)
      bytes += entry.size;
  return bytes;
}

uint64_t BindRegionRegistry::RegisteredBytes(size_t rail, Kind kind) const {
  std::lock_guard<std::mutex> lock(mu_);
  if (rail >= rails_.size()) return 0;
  uint64_t bytes = 0;
  for (const auto& [base, entry] : rails_[rail])
    if (entry.kind == kind && entry.state == State::kRegistered)
      bytes += entry.size;
  return bytes;
}

ibv_mr* BindRegionRegistry::Find(size_t rail, char* base, size_t size,
                                 Kind kind) {
  std::lock_guard<std::mutex> lock(mu_);
  if (!started_ || stopping_ || rail >= rails_.size()) return nullptr;
  auto& regions = rails_[rail];
  const auto it = regions.find(base);
  if (it != regions.end()) {
    Entry& entry = it->second;
    if (entry.state == State::kRegistered) return entry.mr;
    if (entry.state != State::kFailed ||
        NowUs() - entry.failed_us < kRetryUs)
      return nullptr;
    regions.erase(it);  // failed long enough ago: try again
  }
  if (kind == Kind::kArena && arena_cap_ != 0 &&
      BytesLocked(rail, kind) + size > arena_cap_)
    return nullptr;
  Entry entry;
  entry.kind = kind;
  entry.size = size;
  regions.emplace(base, entry);
  queue_.emplace_back(rail, base);
  work_cv_.notify_one();
  return nullptr;
}

void BindRegionRegistry::Run() {
  std::unique_lock<std::mutex> lock(mu_);
  for (;;) {
    work_cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
    if (stopping_) return;
    const auto [rail, base] = queue_.front();
    queue_.pop_front();
    const auto it = rails_[rail].find(base);
    if (it == rails_[rail].end() || it->second.state != State::kQueued)
      continue;  // forgotten meanwhile
    it->second.state = State::kRegistering;
    const size_t size = it->second.size;
    lock.unlock();
    ibv_mr* mr = register_(rail, base, size);
    lock.lock();
    // Forget waits for kRegistering, so the entry is still there.
    Entry& entry = rails_[rail].at(base);
    if (mr) {
      entry.state = State::kRegistered;
      entry.mr = mr;
      registrations_.fetch_add(1, std::memory_order_relaxed);
    } else {
      entry.state = State::kFailed;
      entry.failed_us = NowUs();
      const uint64_t n =
          register_failures_.fetch_add(1, std::memory_order_relaxed);
      if (n == 0 || (n & 0x3FFu) == 0)
        DFKV_LOG_WARN("rdma: a memory-window region of " +
                      std::to_string(size) + " bytes could not be "
                      "registered on rail " + std::to_string(rail) +
                      " (" + std::strerror(errno) +
                      "); its GETs use per-grant MRs. failures=" +
                      std::to_string(n + 1));
    }
    done_cv_.notify_all();
  }
}

bool BindRegionRegistry::Forget(char* base) {
  std::vector<ibv_mr*> registered;
  {
    std::unique_lock<std::mutex> lock(mu_);
    if (!started_) return true;
    for (auto& regions : rails_) {
      done_cv_.wait(lock, [&] {
        const auto it = regions.find(base);
        return it == regions.end() ||
               it->second.state != State::kRegistering;
      });
      const auto it = regions.find(base);
      if (it == regions.end()) continue;
      if (it->second.mr) registered.push_back(it->second.mr);
      regions.erase(it);
    }
  }
  bool ok = true;
  for (ibv_mr* mr : registered) {
    const int error = deregister_(mr);
    if (error == 0) continue;
    ok = false;
    DFKV_LOG_ERROR("rdma: a memory-window region still has a window bound "
                   "as its memory is released (deregistration error " +
                   std::to_string(error) +
                   "); the memory is kept. This is a bug.");
  }
  return ok;
}

void BindRegionRegistry::Stop() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (!started_) return;
    stopping_ = true;
  }
  work_cv_.notify_all();
  if (thread_.joinable()) thread_.join();
  std::vector<ibv_mr*> registered;
  {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& regions : rails_) {
      for (auto& [base, entry] : regions)
        if (entry.mr) registered.push_back(entry.mr);
      regions.clear();
    }
    queue_.clear();
    started_ = false;
  }
  for (ibv_mr* mr : registered) {
    const int error = deregister_(mr);
    if (error != 0)
      DFKV_LOG_ERROR("rdma: a memory-window region still has a window bound "
                     "at stop (deregistration error " +
                     std::to_string(error) + "); left registered. This is "
                     "a bug.");
  }
}

}  // namespace dfkv::rdma
