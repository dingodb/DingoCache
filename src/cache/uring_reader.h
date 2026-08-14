/* uring_reader — single-owner, bounded io_uring reads for the RDMA server.
 *
 * Submission and completion are deliberately separate: the serve loop may keep
 * accepting RDMA receives while disk reads are outstanding.  CQ user_data is a
 * monotonically allocated token, never a caller pointer, so moving a queued
 * request cannot invalidate an in-flight descriptor.
 */
#ifndef DFKV_URING_READER_H_
#define DFKV_URING_READER_H_

#ifdef DFKV_WITH_URING

#include <liburing.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace dfkv {

class UringReader {
 public:
  using Token = uint64_t;
  static constexpr Token kInvalidToken = 0;

  struct ReadDesc {
    int fd = -1;
    void* buf = nullptr;
    unsigned len = 0;
    uint64_t off = 0;
  };

  struct Completion {
    Token token = kInvalidToken;
    long result = 0;
  };

  // A CQE is borrowed from the ring until Reap.  It is intentionally opaque:
  // callers route completed descriptors by Completion::token, not CQE pointers.
  class Event {
   public:
    Event() = default;
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

   private:
    friend class UringReader;
    struct io_uring_cqe* cqe_ = nullptr;
  };

  // Injectable narrow backend used by hardware-free unit tests. Production
  // uses DefaultBackend and pays one virtual dispatch per ring operation.
  class Backend {
   public:
    virtual ~Backend() = default;
    virtual int QueueInit(unsigned entries, struct io_uring* ring) = 0;
    virtual void QueueExit(struct io_uring* ring) = 0;
    virtual struct io_uring_sqe* GetSqe(struct io_uring* ring) = 0;
    virtual void PrepRead(struct io_uring_sqe* sqe, int fd, void* buf,
                          unsigned len, uint64_t off) = 0;
    virtual void SetData(struct io_uring_sqe* sqe, Token token) = 0;
    virtual int Submit(struct io_uring* ring) = 0;
    virtual int PeekCqe(struct io_uring* ring,
                        struct io_uring_cqe** cqe) = 0;
    virtual int WaitCqe(struct io_uring* ring, struct io_uring_cqe** cqe,
                        int timeout_ms) = 0;
    virtual void Seen(struct io_uring* ring, struct io_uring_cqe* cqe) = 0;
  };

  explicit UringReader(unsigned queue_depth, Backend* backend = nullptr)
      : depth_(queue_depth),
        pending_(queue_depth),
        backend_(backend ? backend : &default_backend_) {
    if (queue_depth == 0) return;
    ok_ = backend_->QueueInit(queue_depth, &ring_) == 0;
  }

  ~UringReader() {
    if (!ok_) return;
    // Registered staging must outlive kernel ownership. queue_exit does not
    // cancel accepted reads, so destruction performs the one unbounded drain.
    Drain(/*timeout_ms=*/-1);
    backend_->QueueExit(&ring_);
  }

  UringReader(const UringReader&) = delete;
  UringReader& operator=(const UringReader&) = delete;

  bool ok() const { return ok_; }
  bool poisoned() const { return poisoned_; }
  unsigned depth() const { return depth_; }
  size_t inflight() const { return logical_inflight_; }
  size_t capacity() const { return depth_ - logical_inflight_; }

  // Queue and submit one bounded group. On success every returned token is
  // stable until its logical Completion is reaped. On infrastructure failure
  // the ring is poisoned; accepted reads may still own buffers and Drain is
  // mandatory before their PreparedRead owners are released.
  bool Submit(const ReadDesc* descs, size_t count, Token* tokens) {
    if (!ok_ || poisoned_ || !descs || !tokens || count == 0 ||
        count > capacity())
      return false;
    if (next_token_ >
        std::numeric_limits<Token>::max() - static_cast<Token>(count)) {
      poisoned_ = true;  // never wrap into a token that may have existed
      return false;
    }

    size_t slot = 0;
    for (size_t i = 0; i < count; ++i) {
      while (slot < pending_.size() && pending_[slot].used) ++slot;
      if (slot == pending_.size()) {
        poisoned_ = true;
        return false;
      }
      struct io_uring_sqe* sqe = backend_->GetSqe(&ring_);
      if (!sqe) {
        poisoned_ = true;  // already-prepared SQEs must never be resurrected
        return false;
      }
      Pending& pending = pending_[slot++];
      pending.used = true;
      pending.token = next_token_++;
      pending.desc = descs[i];
      pending.done = 0;
      pending.reaps = 0;
      tokens[i] = pending.token;
      backend_->PrepRead(sqe, pending.desc.fd, pending.desc.buf,
                         pending.desc.len, pending.desc.off);
      backend_->SetData(sqe, pending.token);
    }

    size_t accepted = 0;
    while (accepted < count) {
      int submitted = backend_->Submit(&ring_);
      if (submitted == -EINTR) continue;
      if (submitted <= 0 ||
          static_cast<size_t>(submitted) > count - accepted) {
        poisoned_ = true;
        return false;
      }
      accepted += static_cast<size_t>(submitted);
      kernel_inflight_ += static_cast<size_t>(submitted);
    }
    logical_inflight_ += count;
    return true;
  }

  // Returns 1 with a borrowed event, 0 when no CQE is ready, -1 on ring error.
  int Peek(Event* event) {
    if (!event || event->cqe_ || !ok_ || poisoned_) return -1;
    struct io_uring_cqe* cqe = nullptr;
    const int rc = backend_->PeekCqe(&ring_, &cqe);
    if (rc == -EAGAIN || (rc == 0 && cqe == nullptr)) return 0;
    if (rc < 0 || !cqe) {
      poisoned_ = true;
      return -1;
    }
    event->cqe_ = cqe;
    return 1;
  }

  // Bounded wait. timeout_ms < 0 waits indefinitely. Return values match Peek.
  int Wait(Event* event, int timeout_ms) {
    if (!event || event->cqe_ || !ok_ || poisoned_) return -1;
    struct io_uring_cqe* cqe = nullptr;
    int rc;
    do {
      rc = backend_->WaitCqe(&ring_, &cqe, timeout_ms);
    } while (rc == -EINTR);
    if (rc == -ETIME || rc == -EAGAIN ||
        (rc == 0 && cqe == nullptr))
      return 0;
    if (rc < 0 || !cqe) {
      poisoned_ = true;
      return -1;
    }
    event->cqe_ = cqe;
    return 1;
  }

  // Consume one CQE. Returns 1 for a logical descriptor completion, 0 when a
  // short read was resubmitted under the same token, and -1 on infrastructure
  // failure. Per-read errors remain ordinary Completion::result values.
  int Reap(Event* event, Completion* completion) {
    if (!event || !event->cqe_ || !completion || kernel_inflight_ == 0)
      return -1;
    struct io_uring_cqe* cqe = event->cqe_;
    const Token token = static_cast<Token>(reinterpret_cast<uintptr_t>(
        io_uring_cqe_get_data(cqe)));
    const long result = cqe->res;
    backend_->Seen(&ring_, cqe);
    event->cqe_ = nullptr;
    --kernel_inflight_;

    Pending* pending = Find(token);
    if (!pending) {
      poisoned_ = true;
      return -1;
    }
    ++pending->reaps;
    if (result > 0) {
      pending->done += static_cast<uint64_t>(result);
    }
    if (result > 0 && pending->done < pending->desc.len) {
      // Bound pathological streams of tiny short reads. A normal direct read
      // either completes once or reaches EOF/error on its residual.
      if (pending->reaps > 5 ||
          !PrepResidual(*pending)) {
        poisoned_ = true;
        return -1;
      }
      int submitted;
      do {
        submitted = backend_->Submit(&ring_);
      } while (submitted == -EINTR);
      if (submitted != 1) {
        poisoned_ = true;
        return -1;
      }
      ++kernel_inflight_;
      return 0;
    }

    completion->token = token;
    completion->result =
        result < 0 ? result : static_cast<long>(pending->done);
    pending->used = false;
    --logical_inflight_;
    return 1;
  }

  // Reap and discard all kernel-owned operations exactly once. Logical
  // descriptors (including dormant SQEs after a failed submit) are aborted.
  bool Drain(int timeout_ms = 5000) {
    if (!ok_) return true;
    bool drained = true;
    while (kernel_inflight_ > 0) {
      struct io_uring_cqe* cqe = nullptr;
      int rc;
      do {
        rc = backend_->WaitCqe(&ring_, &cqe, timeout_ms);
      } while (rc == -EINTR);
      if (rc < 0 || !cqe) {
        drained = false;
        break;
      }
      backend_->Seen(&ring_, cqe);
      --kernel_inflight_;
    }
    if (kernel_inflight_ == 0) {
      for (Pending& pending : pending_) pending.used = false;
      logical_inflight_ = 0;
    }
    return drained;
  }

 private:
  class DefaultBackend final : public Backend {
   public:
    int QueueInit(unsigned entries, struct io_uring* ring) override {
      return io_uring_queue_init(entries, ring, 0);
    }
    void QueueExit(struct io_uring* ring) override {
      io_uring_queue_exit(ring);
    }
    struct io_uring_sqe* GetSqe(struct io_uring* ring) override {
      return io_uring_get_sqe(ring);
    }
    void PrepRead(struct io_uring_sqe* sqe, int fd, void* buf, unsigned len,
                  uint64_t off) override {
      io_uring_prep_read(sqe, fd, buf, len, off);
    }
    void SetData(struct io_uring_sqe* sqe, Token token) override {
      io_uring_sqe_set_data(
          sqe, reinterpret_cast<void*>(static_cast<uintptr_t>(token)));
    }
    int Submit(struct io_uring* ring) override {
      return io_uring_submit(ring);
    }
    int PeekCqe(struct io_uring* ring, struct io_uring_cqe** cqe) override {
      return io_uring_peek_cqe(ring, cqe);
    }
    int WaitCqe(struct io_uring* ring, struct io_uring_cqe** cqe,
                int timeout_ms) override {
      if (timeout_ms < 0) return io_uring_wait_cqe(ring, cqe);
      struct __kernel_timespec timeout{
          timeout_ms / 1000,
          static_cast<long long>(timeout_ms % 1000) * 1000000LL};
      return io_uring_wait_cqe_timeout(ring, cqe, &timeout);
    }
    void Seen(struct io_uring* ring, struct io_uring_cqe* cqe) override {
      io_uring_cqe_seen(ring, cqe);
    }
  };

  struct Pending {
    bool used = false;
    Token token = kInvalidToken;
    ReadDesc desc;
    uint64_t done = 0;
    unsigned reaps = 0;
  };

  Pending* Find(Token token) {
    for (Pending& pending : pending_) {
      if (pending.used && pending.token == token) return &pending;
    }
    return nullptr;
  }

  bool PrepResidual(Pending& pending) {
    struct io_uring_sqe* sqe = backend_->GetSqe(&ring_);
    if (!sqe) return false;
    const uint64_t remaining =
        static_cast<uint64_t>(pending.desc.len) - pending.done;
    backend_->PrepRead(
        sqe, pending.desc.fd,
        static_cast<char*>(pending.desc.buf) + pending.done,
        static_cast<unsigned>(remaining), pending.desc.off + pending.done);
    backend_->SetData(sqe, pending.token);
    return true;
  }

  struct io_uring ring_{};
  bool ok_ = false;
  bool poisoned_ = false;
  unsigned depth_ = 0;
  std::vector<Pending> pending_;
  DefaultBackend default_backend_;
  Backend* backend_ = nullptr;
  Token next_token_ = 1;
  size_t logical_inflight_ = 0;
  size_t kernel_inflight_ = 0;
};

}  // namespace dfkv

#endif  // DFKV_WITH_URING
#endif  // DFKV_URING_READER_H_
