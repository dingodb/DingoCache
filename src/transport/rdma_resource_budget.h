#ifndef DFKV_TRANSPORT_RDMA_RESOURCE_BUDGET_H_
#define DFKV_TRANSPORT_RDMA_RESOURCE_BUDGET_H_

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>

namespace dfkv::rdma {

struct ResourceRequest {
  size_t endpoints = 0;
  size_t qps = 0;
  size_t wr_slots = 0;
  uint64_t registered_bytes = 0;
};

// Budget sized from ring topology instead of a constant. Endpoint demand is
// nodes x (payload lane + SG lane, one endpoint per rail each) plus one
// control endpoint per node; +25% headroom keeps eviction cheap while churned
// teardowns still hold their budget. The other three dimensions follow the
// same derivation the constructor defaults use (QP = endpoints, WR =
// endpoints x depth, registered = slot x depth x endpoints). floor_endpoints
// keeps small rings on the documented default. These are admission LIMITS,
// not allocations: raising them consumes nothing until connections exist.
inline ResourceRequest AdaptiveBudgetTarget(size_t nodes, size_t rails,
                                            size_t depth, uint64_t slot_bytes,
                                            size_t floor_endpoints) {
  constexpr size_t kMaxEndpoints = 65536;  // EnvBoundedInt upper bound parity
  const size_t lanes = 2 * (rails ? rails : 1) + 1;
  size_t endpoints = nodes > (kMaxEndpoints / lanes) ? kMaxEndpoints
                                                     : nodes * lanes;
  endpoints = endpoints > kMaxEndpoints - endpoints / 4
                  ? kMaxEndpoints
                  : endpoints + endpoints / 4;
  if (endpoints < floor_endpoints) endpoints = floor_endpoints;
  if (endpoints > kMaxEndpoints) endpoints = kMaxEndpoints;
  const size_t d = depth ? depth : 1;
  ResourceRequest target;
  target.endpoints = endpoints;
  target.qps = endpoints;
  target.wr_slots = endpoints * d;
  target.registered_bytes =
      slot_bytes != 0 && endpoints <= std::numeric_limits<uint64_t>::max() /
                                          slot_bytes / d
          ? slot_bytes * d * endpoints
          : std::numeric_limits<uint64_t>::max();
  return target;
}

class ResourceBudget {
 public:
  explicit ResourceBudget(ResourceRequest limit) : limit_(limit) {}
  bool TryAcquire(const ResourceRequest& request) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!FitsLimit(request) || !FitsAvailable(request)) return false;
    Add(&used_, request);
    ++waiters_served_;
    return true;
  }


  bool Acquire(const ResourceRequest& request,
               std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mu_);
    if (!FitsLimit(request)) return false;
    const auto available = [&] { return FitsAvailable(request); };
    if (!available() &&
        (timeout.count() <= 0 || !cv_.wait_for(lock, timeout, available))) {
      ++timeouts_;
      return false;
    }
    Add(&used_, request);
    ++waiters_served_;
    return true;
  }

  void Release(const ResourceRequest& request) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      Subtract(&used_, request);
    }
    cv_.notify_all();
  }

  // Raise each dimension to max(current, target). Limits only ever grow: a
  // shrink under load would strand used_ > limit_ with no owner to reclaim
  // from. Waiters blocked in Acquire() re-evaluate against the new limit.
  // Returns true when any dimension actually changed.
  bool RaiseLimit(const ResourceRequest& target) {
    bool raised = false;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (target.endpoints > limit_.endpoints) {
        limit_.endpoints = target.endpoints;
        raised = true;
      }
      if (target.qps > limit_.qps) {
        limit_.qps = target.qps;
        raised = true;
      }
      if (target.wr_slots > limit_.wr_slots) {
        limit_.wr_slots = target.wr_slots;
        raised = true;
      }
      if (target.registered_bytes > limit_.registered_bytes) {
        limit_.registered_bytes = target.registered_bytes;
        raised = true;
      }
      if (raised) ++raises_;
    }
    if (raised) cv_.notify_all();
    return raised;
  }

  ResourceRequest limit() const {
    std::lock_guard<std::mutex> lock(mu_);
    return limit_;
  }
  ResourceRequest used() const {
    std::lock_guard<std::mutex> lock(mu_);
    return used_;
  }
  uint64_t timeouts() const {
    std::lock_guard<std::mutex> lock(mu_);
    return timeouts_;
  }
  uint64_t waiters_served() const {
    std::lock_guard<std::mutex> lock(mu_);
    return waiters_served_;
  }
  uint64_t raises() const {
    std::lock_guard<std::mutex> lock(mu_);
    return raises_;
  }

 private:
  template <typename T>
  static bool AddFits(T used, T request, T limit) {
    return request <= limit && used <= limit - request;
  }
  bool FitsLimit(const ResourceRequest& request) const {
    return request.endpoints <= limit_.endpoints && request.qps <= limit_.qps &&
           request.wr_slots <= limit_.wr_slots &&
           request.registered_bytes <= limit_.registered_bytes;
  }
  bool FitsAvailable(const ResourceRequest& request) const {
    return AddFits(used_.endpoints, request.endpoints, limit_.endpoints) &&
           AddFits(used_.qps, request.qps, limit_.qps) &&
           AddFits(used_.wr_slots, request.wr_slots, limit_.wr_slots) &&
           AddFits(used_.registered_bytes, request.registered_bytes,
                   limit_.registered_bytes);
  }
  static void Add(ResourceRequest* value, const ResourceRequest& add) {
    value->endpoints += add.endpoints;
    value->qps += add.qps;
    value->wr_slots += add.wr_slots;
    value->registered_bytes += add.registered_bytes;
  }
  static void Subtract(ResourceRequest* value, const ResourceRequest& sub) {
    value->endpoints -= sub.endpoints;
    value->qps -= sub.qps;
    value->wr_slots -= sub.wr_slots;
    value->registered_bytes -= sub.registered_bytes;
  }

  // Guarded by mu_; mutated only by RaiseLimit (monotonic growth).
  ResourceRequest limit_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  ResourceRequest used_;
  uint64_t timeouts_ = 0;
  uint64_t waiters_served_ = 0;
  uint64_t raises_ = 0;
};

}  // namespace dfkv::rdma

#endif  // DFKV_TRANSPORT_RDMA_RESOURCE_BUDGET_H_
