#ifndef DFKV_TRANSPORT_RDMA_CONNECTION_LIFECYCLE_H_
#define DFKV_TRANSPORT_RDMA_CONNECTION_LIFECYCLE_H_

#include <atomic>
#include <cstdint>

namespace dfkv::rdma {

enum class ConnectionState : uint8_t {
  kActive,
  kIdle,
  kRetireRequested,
  kDraining,
};

class ConnectionLifecycle {
 public:
  bool MakeIdle() {
    ConnectionState expected = ConnectionState::kActive;
    return state_.compare_exchange_strong(expected, ConnectionState::kIdle,
                                          std::memory_order_acq_rel,
                                          std::memory_order_acquire);
  }

  bool Activate() {
    ConnectionState expected = ConnectionState::kIdle;
    return state_.compare_exchange_strong(expected, ConnectionState::kActive,
                                          std::memory_order_acq_rel,
                                          std::memory_order_acquire);
  }

  bool RequestRetire() {
    ConnectionState expected = ConnectionState::kIdle;
    return state_.compare_exchange_strong(
        expected, ConnectionState::kRetireRequested,
        std::memory_order_acq_rel, std::memory_order_acquire);
  }

  bool BeginDrain() {
    ConnectionState current = state_.load(std::memory_order_acquire);
    while (current != ConnectionState::kDraining) {
      if (state_.compare_exchange_weak(current, ConnectionState::kDraining,
                                       std::memory_order_acq_rel,
                                       std::memory_order_acquire)) {
        return true;
      }
    }
    return false;
  }

  ConnectionState state() const {
    return state_.load(std::memory_order_acquire);
  }

 private:
  std::atomic<ConnectionState> state_{ConnectionState::kActive};
};

}  // namespace dfkv::rdma

#endif  // DFKV_TRANSPORT_RDMA_CONNECTION_LIFECYCLE_H_
