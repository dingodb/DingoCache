#ifndef DFKV_TRANSPORT_RDMA_RESOURCE_BUDGET_H_
#define DFKV_TRANSPORT_RDMA_RESOURCE_BUDGET_H_

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace dfkv::rdma {

struct ResourceRequest {
  size_t endpoints = 0;
  size_t qps = 0;
  size_t wr_slots = 0;
  uint64_t registered_bytes = 0;
};

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

  ResourceRequest limit() const { return limit_; }
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

  const ResourceRequest limit_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  ResourceRequest used_;
  uint64_t timeouts_ = 0;
  uint64_t waiters_served_ = 0;
};

}  // namespace dfkv::rdma

#endif  // DFKV_TRANSPORT_RDMA_RESOURCE_BUDGET_H_
