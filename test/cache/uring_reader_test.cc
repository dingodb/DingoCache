// UringReader unit tests. Compiled to an empty suite unless the tree is built
// with -DDFKV_WITH_URING=ON (which requires -DDFKV_WITH_RDMA=ON); the reader
// itself is env-gated at runtime, but here we drive it directly.
#ifdef DFKV_WITH_URING

#include "cache/uring_reader.h"

#include <gtest/gtest.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <list>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using namespace dfkv;  // NOLINT

namespace {

// Temp file filled with a deterministic byte pattern; returns {fd, path}.
std::pair<int, std::string> MakePatternFile(size_t n) {
  std::string path = "/tmp/dfkv_uring_test_XXXXXX";
  int fd = ::mkstemp(&path[0]);
  EXPECT_GE(fd, 0);
  std::vector<char> data(n);
  for (size_t i = 0; i < n; ++i) data[i] = static_cast<char>('a' + (i % 26));
  EXPECT_EQ(::pwrite(fd, data.data(), n, 0), static_cast<ssize_t>(n));
  return {fd, path};
}

bool PatternOk(const char* buf, size_t off, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (buf[i] != static_cast<char>('a' + ((off + i) % 26))) return false;
  return true;
}

bool RingAvailable() {
  UringReader probe(4);
  return probe.ok();  // io_uring may be unavailable/banned in some sandboxes
}

class ControlledBackend final : public UringReader::Backend {
 public:
  struct Request {
    int fd = -1;
    void* buf = nullptr;
    unsigned len = 0;
    uint64_t off = 0;
    UringReader::Token token = UringReader::kInvalidToken;
  };

  int QueueInit(unsigned entries, io_uring*) override {
    initialized_depth = entries;
    return queue_init_result;
  }

  void QueueExit(io_uring*) override {
    std::lock_guard<std::mutex> lock(mu_);
    queue_exited = true;
    cv_.notify_all();
  }

  io_uring_sqe* GetSqe(io_uring*) override {
    std::lock_guard<std::mutex> lock(mu_);
    if (get_sqe_fail_after >= 0 &&
        get_sqe_calls++ >= get_sqe_fail_after) {
      return nullptr;
    }
    sqes_.emplace_back();
    return &sqes_.back();
  }

  void PrepRead(io_uring_sqe* sqe, int fd, void* buf, unsigned len,
                uint64_t off) override {
    std::lock_guard<std::mutex> lock(mu_);
    prepared_[sqe] = Request{fd, buf, len, off,
                             UringReader::kInvalidToken};
  }

  void SetData(io_uring_sqe* sqe, UringReader::Token token) override {
    std::lock_guard<std::mutex> lock(mu_);
    auto found = prepared_.find(sqe);
    ASSERT_NE(found, prepared_.end());
    found->second.token = token;
    dormant_.push_back(found->second);
    prepared_.erase(found);
  }

  int Submit(io_uring*) override {
    std::lock_guard<std::mutex> lock(mu_);
    ++submit_calls;
    int result = static_cast<int>(dormant_.size());
    if (!submit_results.empty()) {
      result = submit_results.front();
      submit_results.pop_front();
    }
    if (result <= 0) return result;
    if (static_cast<size_t>(result) > dormant_.size()) return result;
    for (int i = 0; i < result; ++i) {
      Request request = dormant_.front();
      dormant_.pop_front();
      accepted_[request.token] = request;
      submission_history.push_back(request);
    }
    max_inflight = std::max(max_inflight, accepted_.size());
    cv_.notify_all();
    return result;
  }

  int PeekCqe(io_uring*, io_uring_cqe** cqe) override {
    std::lock_guard<std::mutex> lock(mu_);
    if (ready_.empty()) {
      *cqe = nullptr;
      return -EAGAIN;
    }
    *cqe = &ready_.front();
    return 0;
  }

  int WaitCqe(io_uring*, io_uring_cqe** cqe, int timeout_ms) override {
    std::unique_lock<std::mutex> lock(mu_);
    const auto ready = [&] { return !ready_.empty() || queue_exited; };
    if (timeout_ms < 0) {
      cv_.wait(lock, ready);
    } else if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                             ready)) {
      *cqe = nullptr;
      return -ETIME;
    }
    if (ready_.empty()) {
      *cqe = nullptr;
      return -ETIME;
    }
    *cqe = &ready_.front();
    return 0;
  }

  void Seen(io_uring*, io_uring_cqe* cqe) override {
    std::lock_guard<std::mutex> lock(mu_);
    ASSERT_FALSE(ready_.empty());
    ASSERT_EQ(cqe, &ready_.front());
    ready_.pop_front();
    ++seen;
  }

  bool Complete(UringReader::Token token, long result) {
    std::lock_guard<std::mutex> lock(mu_);
    auto found = accepted_.find(token);
    if (found == accepted_.end()) return false;
    accepted_.erase(found);
    io_uring_cqe cqe{};
    cqe.user_data = token;
    cqe.res = static_cast<int32_t>(result);
    ready_.push_back(cqe);
    cv_.notify_all();
    return true;
  }

  bool CompleteRead(UringReader::Token token) {
    Request request;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto found = accepted_.find(token);
      if (found == accepted_.end()) return false;
      request = found->second;
    }
    const ssize_t result =
        ::pread(request.fd, request.buf, request.len,
                static_cast<off_t>(request.off));
    return Complete(token, result);
  }

  size_t accepted() const {
    std::lock_guard<std::mutex> lock(mu_);
    return accepted_.size();
  }

  int queue_init_result = 0;
  int get_sqe_fail_after = -1;
  unsigned initialized_depth = 0;
  bool queue_exited = false;
  size_t max_inflight = 0;
  size_t seen = 0;
  int submit_calls = 0;
  std::deque<int> submit_results;
  std::vector<Request> submission_history;

 private:
  mutable std::mutex mu_;
  std::condition_variable cv_;
  int get_sqe_calls = 0;
  std::list<io_uring_sqe> sqes_;
  std::map<io_uring_sqe*, Request> prepared_;
  std::deque<Request> dormant_;
  std::map<UringReader::Token, Request> accepted_;
  std::list<io_uring_cqe> ready_;
};

bool ReadAll(UringReader* ring, UringReader::ReadDesc* descs, size_t count,
             std::vector<long>* results) {
  std::vector<UringReader::Token> tokens(count);
  if (!ring->Submit(descs, count, tokens.data())) return false;
  results->assign(count, 0);
  size_t remaining = count;
  while (remaining != 0) {
    UringReader::Event event;
    if (ring->Wait(&event, -1) != 1) return false;
    UringReader::Completion completion;
    const int reaped = ring->Reap(&event, &completion);
    if (reaped < 0) return false;
    if (reaped == 0) continue;
    auto found = std::find(tokens.begin(), tokens.end(), completion.token);
    if (found == tokens.end()) return false;
    (*results)[static_cast<size_t>(found - tokens.begin())] = completion.result;
    --remaining;
  }
  return true;
}

}  // namespace

TEST(UringReader, SubmitAndReapFillEveryDescAtItsOwnOffset) {
  if (!RingAvailable()) GTEST_SKIP() << "io_uring unavailable";
  auto [fd, path] = MakePatternFile(1 << 20);
  constexpr int kN = 7;
  std::vector<std::vector<char>> bufs(kN, std::vector<char>(4096));
  std::vector<UringReader::ReadDesc> descs(kN);
  for (int i = 0; i < kN; ++i) {
    descs[i].fd = fd;
    descs[i].buf = bufs[i].data();
    descs[i].len = 4096;
    descs[i].off = static_cast<uint64_t>(i) * 8192;
  }
  UringReader ring(8);
  ASSERT_TRUE(ring.ok());
  std::vector<long> results;
  ASSERT_TRUE(ReadAll(&ring, descs.data(), kN, &results));
  EXPECT_FALSE(ring.poisoned());
  for (int i = 0; i < kN; ++i) {
    EXPECT_EQ(results[i], 4096) << "desc " << i;
    EXPECT_TRUE(PatternOk(bufs[i].data(), descs[i].off, 4096)) << "desc " << i;
  }
  ::close(fd);
  ::unlink(path.c_str());
}

TEST(UringReader, PerReadErrorIsRecordedNotInfraFailure) {
  if (!RingAvailable()) GTEST_SKIP() << "io_uring unavailable";
  auto [fd, path] = MakePatternFile(8192);
  int bad = ::open(path.c_str(), O_WRONLY);  // read on O_WRONLY fd => -EBADF/-EACCES
  ASSERT_GE(bad, 0);
  std::vector<char> b0(4096), b1(4096);
  UringReader::ReadDesc descs[2];
  descs[0] = {fd, b0.data(), 4096, 0};
  descs[1] = {bad, b1.data(), 4096, 0};
  UringReader ring(4);
  ASSERT_TRUE(ring.ok());
  // A per-read failure is the caller's to inspect; the submission/reap itself
  // succeeds and the ring stays clean (no poison, nothing left in flight).
  std::vector<long> results;
  ASSERT_TRUE(ReadAll(&ring, descs, 2, &results));
  EXPECT_FALSE(ring.poisoned());
  EXPECT_EQ(results[0], 4096);
  EXPECT_LT(results[1], 0);
  EXPECT_TRUE(ring.Drain());  // no-op: everything reaped
  ::close(fd);
  ::close(bad);
  ::unlink(path.c_str());
}

TEST(UringReader, EofYieldsBytesUpToEof) {
  if (!RingAvailable()) GTEST_SKIP() << "io_uring unavailable";
  auto [fd, path] = MakePatternFile(1000);  // < one 4096 read
  std::vector<char> buf(4096);
  UringReader::ReadDesc d{fd, buf.data(), 4096, 0};
  UringReader ring(4);
  ASSERT_TRUE(ring.ok());
  std::vector<long> results;
  ASSERT_TRUE(ReadAll(&ring, &d, 1, &results));
  EXPECT_EQ(results[0], 1000);
  EXPECT_TRUE(PatternOk(buf.data(), 0, 1000));
  ::close(fd);
  ::unlink(path.c_str());
}

TEST(UringReader, DrainOnIdleRingIsNoop) {
  if (!RingAvailable()) GTEST_SKIP() << "io_uring unavailable";
  UringReader ring(4);
  ASSERT_TRUE(ring.ok());
  EXPECT_FALSE(ring.poisoned());
  EXPECT_TRUE(ring.Drain());
  EXPECT_TRUE(ring.Drain());  // idempotent
}

TEST(UringReader, RepeatedBatchesReuseTheRingCleanly) {
  if (!RingAvailable()) GTEST_SKIP() << "io_uring unavailable";
  auto [fd, path] = MakePatternFile(1 << 16);
  std::vector<char> buf(4096);
  UringReader ring(2);
  ASSERT_TRUE(ring.ok());
  for (int round = 0; round < 5; ++round) {
    UringReader::ReadDesc d{fd, buf.data(), 4096,
                            static_cast<uint64_t>(round) * 4096};
    std::vector<long> results;
    ASSERT_TRUE(ReadAll(&ring, &d, 1, &results)) << "round " << round;
    EXPECT_EQ(results[0], 4096);
    EXPECT_TRUE(PatternOk(buf.data(), d.off, 4096)) << "round " << round;
  }
  EXPECT_FALSE(ring.poisoned());
  ::close(fd);
  ::unlink(path.c_str());
}

TEST(UringReader, ControlledBackendSubmitsSixteenBeforeAnyCompletion) {
  ControlledBackend backend;
  UringReader ring(32, &backend);
  ASSERT_TRUE(ring.ok());

  constexpr size_t kReads = 16;
  std::vector<std::vector<char>> buffers(
      kReads, std::vector<char>(512, '\0'));
  std::vector<UringReader::ReadDesc> descs(kReads);
  for (size_t i = 0; i < kReads; ++i)
    descs[i] = {-1, buffers[i].data(), 512, i * 4096};
  std::vector<UringReader::Token> tokens(kReads);

  ASSERT_TRUE(ring.Submit(descs.data(), descs.size(), tokens.data()));
  EXPECT_EQ(backend.submission_history.size(), kReads);
  EXPECT_GE(backend.max_inflight, kReads);
  EXPECT_EQ(ring.inflight(), kReads);
  EXPECT_EQ(ring.capacity(), 16u);
  UringReader::Event not_ready;
  EXPECT_EQ(ring.Peek(&not_ready), 0)
      << "Submit must not wait for even the first disk completion";

  // Make the last request ready first. It must be observable immediately while
  // the other fifteen logical reads remain outstanding (no wait-all barrier).
  ASSERT_TRUE(backend.Complete(tokens.back(), 512));
  UringReader::Event first_event;
  ASSERT_EQ(ring.Peek(&first_event), 1);
  UringReader::Completion first;
  ASSERT_EQ(ring.Reap(&first_event, &first), 1);
  EXPECT_EQ(first.token, tokens.back());
  EXPECT_EQ(first.result, 512);
  EXPECT_EQ(ring.inflight(), kReads - 1);

  std::vector<UringReader::Token> completion_order{first.token};
  for (size_t i = kReads - 1; i-- > 0;) {
    ASSERT_TRUE(backend.Complete(tokens[i], 512));
    UringReader::Event event;
    ASSERT_EQ(ring.Peek(&event), 1);
    UringReader::Completion completion;
    ASSERT_EQ(ring.Reap(&event, &completion), 1);
    completion_order.push_back(completion.token);
  }
  ASSERT_EQ(completion_order.size(), kReads);
  for (size_t i = 0; i < kReads; ++i)
    EXPECT_EQ(completion_order[i], tokens[kReads - 1 - i]);
  EXPECT_EQ(ring.inflight(), 0u);
  EXPECT_FALSE(ring.poisoned());
}

TEST(UringReader, ControlledBackendOutOfOrderErrorsAndResidualKeepToken) {
  ControlledBackend backend;
  UringReader ring(8, &backend);
  std::vector<char> first(4096), second(2048), third(1024);
  UringReader::ReadDesc descs[] = {
      {-1, first.data(), static_cast<unsigned>(first.size()), 100},
      {-1, second.data(), static_cast<unsigned>(second.size()), 200},
      {-1, third.data(), static_cast<unsigned>(third.size()), 300},
  };
  UringReader::Token tokens[3]{};
  ASSERT_TRUE(ring.Submit(descs, 3, tokens));

  ASSERT_TRUE(backend.Complete(tokens[2], 1024));
  UringReader::Event third_event;
  ASSERT_EQ(ring.Peek(&third_event), 1);
  UringReader::Completion third_completion;
  ASSERT_EQ(ring.Reap(&third_event, &third_completion), 1);
  EXPECT_EQ(third_completion.token, tokens[2]);
  EXPECT_EQ(third_completion.result, 1024);

  ASSERT_TRUE(backend.Complete(tokens[0], 1000));
  UringReader::Event short_event;
  ASSERT_EQ(ring.Peek(&short_event), 1);
  UringReader::Completion ignored;
  EXPECT_EQ(ring.Reap(&short_event, &ignored), 0);
  ASSERT_EQ(backend.submission_history.size(), 4u);
  const ControlledBackend::Request& residual =
      backend.submission_history.back();
  EXPECT_EQ(residual.token, tokens[0]);
  EXPECT_EQ(residual.buf, first.data() + 1000);
  EXPECT_EQ(residual.len, 3096u);
  EXPECT_EQ(residual.off, 1100u);

  ASSERT_TRUE(backend.Complete(tokens[1], -EIO));
  UringReader::Event error_event;
  ASSERT_EQ(ring.Peek(&error_event), 1);
  UringReader::Completion error_completion;
  ASSERT_EQ(ring.Reap(&error_event, &error_completion), 1);
  EXPECT_EQ(error_completion.token, tokens[1]);
  EXPECT_EQ(error_completion.result, -EIO);
  EXPECT_FALSE(ring.poisoned())
      << "per-read errors are data-plane results, not ring failures";

  ASSERT_TRUE(backend.Complete(tokens[0], 3096));
  UringReader::Event residual_event;
  ASSERT_EQ(ring.Peek(&residual_event), 1);
  UringReader::Completion first_completion;
  ASSERT_EQ(ring.Reap(&residual_event, &first_completion), 1);
  EXPECT_EQ(first_completion.token, tokens[0]);
  EXPECT_EQ(first_completion.result, 4096);
  EXPECT_EQ(ring.inflight(), 0u);
}

TEST(UringReader, ControlledBackendSubmitFailureBeforeAdmissionPoisons) {
  ControlledBackend backend;
  backend.submit_results = {-EIO};
  UringReader ring(4, &backend);
  std::vector<char> buffer(4096);
  UringReader::ReadDesc desc{-1, buffer.data(), 4096, 0};
  UringReader::Token token = UringReader::kInvalidToken;

  EXPECT_FALSE(ring.Submit(&desc, 1, &token));
  EXPECT_TRUE(ring.poisoned());
  EXPECT_EQ(backend.accepted(), 0u);
  EXPECT_EQ(ring.inflight(), 0u);
  EXPECT_TRUE(ring.Drain(0));
  EXPECT_EQ(backend.seen, 0u);
  EXPECT_FALSE(ring.Submit(&desc, 1, &token));
}

TEST(UringReader, ControlledBackendPartialAdmissionDrainsBeforeReuse) {
  ControlledBackend backend;
  backend.submit_results = {2, -EIO};
  UringReader ring(8, &backend);
  std::vector<std::vector<char>> buffers(4, std::vector<char>(4096));
  UringReader::ReadDesc descs[4];
  for (size_t i = 0; i < 4; ++i)
    descs[i] = {-1, buffers[i].data(), 4096, i * 4096};
  UringReader::Token tokens[4]{};

  EXPECT_FALSE(ring.Submit(descs, 4, tokens));
  ASSERT_TRUE(ring.poisoned());
  ASSERT_EQ(backend.accepted(), 2u);
  ASSERT_TRUE(backend.Complete(tokens[0], 4096));
  ASSERT_TRUE(backend.Complete(tokens[1], 4096));
  EXPECT_TRUE(ring.Drain(0));
  EXPECT_EQ(backend.seen, 2u);
  EXPECT_EQ(backend.accepted(), 0u);
  EXPECT_EQ(ring.inflight(), 0u);
}

TEST(UringReader, ControlledBackendPartialCqDrainRetainsAllOwners) {
  ControlledBackend backend;
  UringReader ring(8, &backend);
  std::vector<std::vector<char>> buffers(4, std::vector<char>(4096));
  UringReader::ReadDesc descs[4];
  for (size_t i = 0; i < 4; ++i)
    descs[i] = {-1, buffers[i].data(), 4096, i * 4096};
  UringReader::Token tokens[4]{};
  ASSERT_TRUE(ring.Submit(descs, 4, tokens));

  ASSERT_TRUE(backend.Complete(tokens[2], 4096));
  ASSERT_TRUE(backend.Complete(tokens[0], 4096));
  EXPECT_FALSE(ring.Drain(0));
  EXPECT_EQ(backend.seen, 2u);
  EXPECT_EQ(ring.inflight(), 4u)
      << "logical owners must remain retained until every kernel read drains";

  ASSERT_TRUE(backend.Complete(tokens[3], 4096));
  ASSERT_TRUE(backend.Complete(tokens[1], 4096));
  EXPECT_TRUE(ring.Drain(0));
  EXPECT_EQ(backend.seen, 4u);
  EXPECT_EQ(ring.inflight(), 0u);
}

TEST(UringReader, ControlledBackendInitFailureIsExplicitSyncFallbackSignal) {
  ControlledBackend backend;
  backend.queue_init_result = -EPERM;
  UringReader ring(32, &backend);
  EXPECT_FALSE(ring.ok());
  EXPECT_TRUE(ring.Drain(0));
  EXPECT_EQ(backend.initialized_depth, 32u);
}

#endif  // DFKV_WITH_URING
