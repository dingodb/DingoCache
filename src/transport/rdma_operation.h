#ifndef DFKV_TRANSPORT_RDMA_OPERATION_H_
#define DFKV_TRANSPORT_RDMA_OPERATION_H_

#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>

namespace dfkv::rdma {

enum class OperationState : uint8_t {
  kCreated,
  kSubmitted,
  kPolling,
  kCompleted,
  kCancelled,
  kFailed,
};

// One shard-level RDMA operation context. The scheduler may run many contexts
// concurrently, but one context admits exactly one CQ polling owner. Caller
// buffers and transient MRs remain owned by the executing context until it
// reaches a terminal state.
class OperationContext {
 public:
  bool Submit() {
    OperationState expected = OperationState::kCreated;
    return state_.compare_exchange_strong(expected, OperationState::kSubmitted,
                                          std::memory_order_release,
                                          std::memory_order_relaxed);
  }

  bool ClaimPoller() {
    OperationState expected = OperationState::kSubmitted;
    if (!state_.compare_exchange_strong(expected, OperationState::kPolling,
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
      return false;
    }
    {
      std::lock_guard<std::mutex> lock(owner_mu_);
      poller_ = std::this_thread::get_id();
    }
    return true;
  }

  bool IsPoller() const {
    std::lock_guard<std::mutex> lock(owner_mu_);
    return poller_ == std::this_thread::get_id();
  }

  void RequestCancel() {
    cancel_requested_.store(true, std::memory_order_release);
  }
  bool cancel_requested() const {
    return cancel_requested_.load(std::memory_order_acquire);
  }

  bool Complete(bool success) {
    if (!IsPoller()) return false;
    OperationState expected = OperationState::kPolling;
    const OperationState terminal =
        cancel_requested() ? OperationState::kCancelled
                           : (success ? OperationState::kCompleted
                                      : OperationState::kFailed);
    return state_.compare_exchange_strong(expected, terminal,
                                          std::memory_order_acq_rel,
                                          std::memory_order_acquire);
  }

  OperationState state() const {
    return state_.load(std::memory_order_acquire);
  }
  bool terminal() const {
    const OperationState current = state();
    return current == OperationState::kCompleted ||
           current == OperationState::kCancelled ||
           current == OperationState::kFailed;
  }

 private:
  std::atomic<OperationState> state_{OperationState::kCreated};
  std::atomic<bool> cancel_requested_{false};
  mutable std::mutex owner_mu_;
  std::thread::id poller_;
};

}  // namespace dfkv::rdma

#endif  // DFKV_TRANSPORT_RDMA_OPERATION_H_
