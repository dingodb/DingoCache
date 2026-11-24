/* Server memory windows for GET grants (DFKV_RDMA_SERVER_MW, off by default;
 * docs/design/server-memory-windows.md): the quarantine a revoked range
 * waits out, and the registry of the regions windows bind to. Free of
 * verbs calls (registration goes through callbacks), so it is tested
 * without a device. */
#ifndef DFKV_TRANSPORT_RDMA_MW_H_
#define DFKV_TRANSPORT_RDMA_MW_H_

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

struct ibv_mr;

namespace dfkv::rdma {

// How long a revoked GET range stays out of use, in microseconds:
// max(floor, 2 x bytes / rate + margin). rate_mbps is the slowest rail's
// (0: unknown, taken as 10 Gbit/s).
uint64_t MwQuarantineUs(size_t bytes, uint64_t rate_mbps, uint64_t floor_us,
                        uint64_t margin_us);

// The regions windows bind to: receive chunks and RAM-arena extents, each
// registered once per rail (each rail has its own protection domain), with
// local write and window binding and no remote access. A region is
// registered in the background the first time a grant asks for it; until
// then, or after it failed, the grant uses a per-grant MR. No region is
// registered twice on one rail, and none under a lock a grant takes.
class BindRegionRegistry {
 public:
  enum class Kind { kChunk, kArena };
  // Registers [base, base+size) on `rail`; nullptr on failure. Runs on the
  // registry's thread, outside its lock.
  using RegisterFn = std::function<ibv_mr*(size_t rail, char* base,
                                           size_t size)>;
  // 0, or the error of the deregistration (EBUSY: a window is still bound).
  using DeregisterFn = std::function<int(ibv_mr*)>;

  BindRegionRegistry() = default;
  BindRegionRegistry(const BindRegionRegistry&) = delete;
  BindRegionRegistry& operator=(const BindRegionRegistry&) = delete;
  ~BindRegionRegistry() { Stop(); }

  // arena_cap: bytes of arena regions each rail may have registered (or
  // being registered); 0 = no limit.
  void Start(size_t rails, RegisterFn reg, DeregisterFn dereg,
             uint64_t arena_cap);
  bool started() const { return started_; }
  // The region's registration on `rail`, or nullptr: then a registration is
  // started unless one is under way, one failed less than kRetryUs ago, or
  // an arena region would go over the cap.
  ibv_mr* Find(size_t rail, char* base, size_t size, Kind kind);
  // Before a region's memory goes: waits for a registration under way and
  // deregisters it on every rail. False if a deregistration failed (a
  // window still bound: a bug); that memory must then be kept.
  bool Forget(char* base);
  // Deregisters every region (a failure is logged and the region left) and
  // ends the thread. Every window must be gone.
  void Stop();

  uint64_t RegisteredBytes(size_t rail, Kind kind) const;
  uint64_t RegisterFailures() const {
    return register_failures_.load(std::memory_order_relaxed);
  }
  uint64_t Registrations() const {
    return registrations_.load(std::memory_order_relaxed);
  }

  static constexpr uint64_t kRetryUs = 10'000'000;

 private:
  enum class State { kQueued, kRegistering, kRegistered, kFailed };
  struct Entry {
    State state = State::kQueued;
    Kind kind = Kind::kChunk;
    size_t size = 0;
    ibv_mr* mr = nullptr;
    uint64_t failed_us = 0;
  };
  void Run();
  // Bytes of `kind` on `rail` registered or being registered; mu_ held.
  uint64_t BytesLocked(size_t rail, Kind kind) const;

  RegisterFn register_;
  DeregisterFn deregister_;
  uint64_t arena_cap_ = 0;
  bool started_ = false;
  mutable std::mutex mu_;
  std::condition_variable work_cv_;  // the thread: work queued, or stop
  std::condition_variable done_cv_;  // a registration finished
  std::vector<std::unordered_map<char*, Entry>> rails_;  // under mu_
  std::deque<std::pair<size_t, char*>> queue_;           // under mu_
  bool stopping_ = false;                                // under mu_
  std::thread thread_;
  std::atomic<uint64_t> register_failures_{0};
  std::atomic<uint64_t> registrations_{0};
};

}  // namespace dfkv::rdma

#endif  // DFKV_TRANSPORT_RDMA_MW_H_
