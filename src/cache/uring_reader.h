/* uring_reader — a thin, single-owner io_uring wrapper for the RDMA server's
 * async GET (kRange) disk-read path. ADDITIVE + env-gated: only used when the
 * server is built with -DDFKV_WITH_URING and started with DFKV_SERVER_URING=1.
 *
 * Design = Mooncake's batch_read pattern (uring_file.cpp) adapted to dfkv's
 * per-WaitComp completion batch:
 *  - One ring per connection serve loop (single owner => no locking).
 *  - BatchRead() submits up to QUEUE_DEPTH independent O_DIRECT preads at once
 *    (each its own fd/buffer/offset), then waits for the WHOLE batch to complete
 *    before returning. Concurrency comes from QD>1 reads in flight; ordering is
 *    irrelevant inside the ring because every read lands in its own buffer.
 *  - The serve loop only PostSendScatters AFTER BatchRead() returns (all reads
 *    done), iterating the descriptors in arrival order — so replies are strictly
 *    in request order with no reorder buffer.
 *
 * Header-only and #ifdef'd on DFKV_WITH_URING so the rest of the server compiles
 * unchanged when liburing is absent. */
#ifndef DFKV_URING_READER_H_
#define DFKV_URING_READER_H_

#ifdef DFKV_WITH_URING

#include <liburing.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dfkv {

// Single-owner io_uring read ring. Not thread-safe by design (one serve loop
// owns one instance). QD bounds the number of reads in flight per batch.
class UringReader {
 public:
  // One independent read in a batch. `result` is filled by BatchRead (>=0 bytes
  // read, <0 = -errno) so the caller can validate each read after the batch.
  struct ReadDesc {
    int fd = -1;
    void* buf = nullptr;
    unsigned len = 0;
    uint64_t off = 0;
    long result = 0;  // out: cqe->res for this read
  };

  explicit UringReader(unsigned queue_depth) {
    // Default setup flags: portable across the 5.15 kernels in the fleet.
    int ret = io_uring_queue_init(queue_depth, &ring_, 0);
    ok_ = (ret == 0);
    depth_ = queue_depth;
  }

  ~UringReader() {
    if (ok_) {
      // Never let the kernel finish an async read into a buffer after we are
      // gone: queue_exit does NOT cancel in-flight requests, so an undrained
      // read would DMA into whatever lives at the (freed/reused) buffer later.
      Drain(/*timeout_ms=*/-1);
      io_uring_queue_exit(&ring_);
    }
  }

  UringReader(const UringReader&) = delete;
  UringReader& operator=(const UringReader&) = delete;

  bool ok() const { return ok_; }
  unsigned depth() const { return depth_; }

  // True after any BatchRead infrastructure failure. A poisoned ring refuses
  // further BatchRead calls: a failed batch may have left (a) reads in flight
  // in the kernel and (b) prepped-but-unsubmitted SQEs dormant in the SQ ring.
  // A later submit would resurrect those dormant SQEs and execute them against
  // a PREVIOUS batch's buffers (by then rearmed/reused) — silent corruption.
  // The caller keeps the connection alive on the synchronous fallback instead.
  bool poisoned() const { return poisoned_; }

  // Reap-and-discard every read the kernel has accepted. A non-negative timeout
  // bounds each wait and lets the serve loop abandon a connection promptly.
  // timeout_ms < 0 blocks without a deadline and is used by the destructor so
  // destination staging is never released while the kernel still owns it.
  bool Drain(int timeout_ms = 5000) {
    if (!ok_) return true;
    while (inflight_ > 0) {
      struct io_uring_cqe* cqe = nullptr;
      int w;
      if (timeout_ms < 0) {
        w = io_uring_wait_cqe(&ring_, &cqe);
      } else {
        struct __kernel_timespec ts{
            timeout_ms / 1000,
            static_cast<long long>(timeout_ms % 1000) * 1000000LL};
        w = io_uring_wait_cqe_timeout(&ring_, &cqe, &ts);
      }
      if (w == -EINTR) continue;
      if (w < 0 || cqe == nullptr) return false;
      io_uring_cqe_seen(&ring_, cqe);
      --inflight_;
    }
    return true;
  }

  // Submit up to QUEUE_DEPTH reads at a time (each at its own fd/offset/buffer),
  // wait for the whole sub-batch, fill each desc.result, then repeat until all
  // `cnt` descs are done. Returns true if every read was submitted+reaped (a
  // per-read short/EOF/error is recorded in desc.result, NOT a hard failure — the
  // caller validates each). Returns false on a submit/wait infrastructure
  // failure; the ring is then POISONED (see poisoned()) and reads may still be
  // in flight against the descs' buffers — the caller MUST Drain() before it
  // touches or reuses any of them, and may only use the synchronous path from
  // then on.
  bool BatchRead(ReadDesc* descs, int cnt) {
    if (!ok_ || poisoned_ || cnt <= 0) return false;
    ++gen_;  // tags this batch's user_data so stale CQEs can never be misrouted
    // Per-desc bytes already read (for short-read residual re-prep). `descs[i].result`
    // is the FINAL byte count exposed to the caller (>=0) or -errno on hard error.
    // done_[i] tracks accumulated progress while residuals are still in flight.
    done_.assign(static_cast<size_t>(cnt), 0);
    int idx = 0;
    while (idx < cnt) {
      const int batch = std::min(cnt - idx, static_cast<int>(depth_));
      // user_data = (gen_ << 32) | desc index: the index maps each CQE back to
      // its desc regardless of completion order; the generation makes a CQE
      // from any earlier (failed) batch recognizably stale instead of being
      // silently misattributed to a same-numbered desc of THIS batch.
      for (int i = 0; i < batch; ++i) {
        if (!PrepRead(idx + i, descs[idx + i].buf, descs[idx + i].len,
                      descs[idx + i].off, descs[idx + i].fd)) {
          // SQ exhausted unexpectedly (batch <= depth_). The SQEs prepped so
          // far are dormant in the SQ ring; poison so no later submit can
          // resurrect them against reused buffers.
          poisoned_ = true;
          return false;
        }
      }
      // Submit all `batch` SQEs and wait for at least one CQE. Two hazards:
      //  (1) io_uring_submit_and_wait can return -EINTR after the SQEs are already
      //      in the kernel — do NOT resubmit (that would double-queue and desync the
      //      user_data->desc mapping); just re-enter to wait for the pending CQEs.
      //  (2) it may accept fewer than `batch` SQEs (short submit). Any leftover SQEs
      //      stay in the SQ ring; we must flush them with a follow-up submit rather
      //      than leave them to be resurrected by a later residual submit. We loop
      //      until the full batch is accepted so `outstanding == batch` holds.
      int accepted = 0;
      bool waited = false;
      while (accepted < batch) {
        int s = waited ? io_uring_submit(&ring_)
                       : io_uring_submit_and_wait(&ring_, 1);
        if (s < 0) {
          if (s == -EINTR) { waited = true; continue; }  // SQEs already queued
          // Submit can't make progress: reads already accepted are in flight
          // against the descs' buffers and leftover SQEs linger dormant in the
          // SQ ring. Poison and bail; the caller must Drain() before touching
          // any buffer and stays on the synchronous path afterwards.
          poisoned_ = true;
          return false;
        }
        if (s == 0 && waited) { poisoned_ = true; return false; }  // no forward progress
        accepted += s;
        inflight_ += static_cast<uint64_t>(s);  // kernel now owns these reads
        waited = true;  // first call already armed the wait; later calls just submit
      }
      int outstanding = accepted;
      // Reap `outstanding` completions, routing each to its desc by user_data. A
      // short read (0 < res < remaining) re-preps the residual for that desc; a
      // re-prep adds one more outstanding completion. Bound the total reaps to
      // avoid an unbounded loop on a pathological fd.
      int reaped = 0;
      const int kMaxReaps = outstanding + 4 * batch;  // residuals bounded per desc
      while (outstanding > 0) {
        if (reaped >= kMaxReaps) { poisoned_ = true; return false; }
        struct io_uring_cqe* cqe = nullptr;
        int w = io_uring_wait_cqe(&ring_, &cqe);
        if (w == -EINTR) continue;  // wait interrupted: the CQE is still there
        if (w < 0 || cqe == nullptr) { poisoned_ = true; return false; }
        const uint64_t ud = static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(io_uring_cqe_get_data(cqe)));
        const uint32_t g = static_cast<uint32_t>(ud >> 32);
        const uint64_t di = ud & 0xffffffffu;
        const long res = cqe->res;
        io_uring_cqe_seen(&ring_, cqe);
        ++reaped;
        --inflight_;  // the kernel is done with that read's buffer
        if (g != gen_) continue;  // stale CQE from an earlier batch: not ours
        --outstanding;
        if (di >= static_cast<uint64_t>(cnt)) continue;  // foreign user_data (defensive)
        ReadDesc& d = descs[di];
        if (res < 0) { d.result = res; continue; }            // hard error: report -errno
        if (res == 0) { d.result = static_cast<long>(done_[di]); continue; }  // EOF
        done_[di] += static_cast<uint64_t>(res);
        if (done_[di] >= d.len) { d.result = static_cast<long>(done_[di]); continue; }
        // Short read: re-prep the residual [off+done, len-done) into buf+done.
        if (!PrepRead(static_cast<int>(di),
                      static_cast<char*>(d.buf) + done_[di],
                      static_cast<unsigned>(d.len - done_[di]),
                      d.off + done_[di], d.fd)) {
          poisoned_ = true;  // outstanding reads of this batch are still in flight
          return false;
        }
        // Submit the residual SQE. Retry on EINTR (submit does not wait, so it
        // is safe to re-enter); poison on any other short/error — outstanding
        // reads remain in flight and the residual SQE may linger dormant.
        int s;
        do { s = io_uring_submit(&ring_); } while (s == -EINTR);
        if (s < 1) { poisoned_ = true; return false; }
        ++outstanding;  // the residual read is now in flight
        ++inflight_;
      }
      idx += batch;
    }
    return true;
  }

 private:
  // Prep a single read SQE with user_data = (gen_ << 32) | desc index. Returns
  // false only if the SQ ring is unexpectedly exhausted (the caller never
  // queues more than depth_).
  bool PrepRead(int desc_idx, void* buf, unsigned len, uint64_t off, int fd) {
    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (!sqe) return false;
    io_uring_prep_read(sqe, fd, buf, len, off);
    // Use the void* user_data API (not set_data64) so we build against
    // liburing >= 2.0 (the _data64 variants only exist in liburing >= 2.2).
    io_uring_sqe_set_data(
        sqe, reinterpret_cast<void*>(static_cast<uintptr_t>(
                 (static_cast<uint64_t>(gen_) << 32) |
                 static_cast<uint32_t>(desc_idx))));
    return true;
  }

  struct io_uring ring_{};
  bool ok_ = false;
  bool poisoned_ = false;   // a failed batch left in-flight reads / dormant SQEs
  unsigned depth_ = 0;
  uint32_t gen_ = 0;        // per-BatchRead generation tag (CQE routing)
  uint64_t inflight_ = 0;   // reads the kernel has accepted but we haven't reaped
  std::vector<uint64_t> done_;  // per-desc bytes read so far (residual tracking)
};

}  // namespace dfkv

#endif  // DFKV_WITH_URING
#endif  // DFKV_URING_READER_H_
