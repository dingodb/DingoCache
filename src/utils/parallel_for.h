/* Process-wide bounded executor for small internal parallel-for jobs. */
#ifndef DFKV_PARALLEL_FOR_H_
#define DFKV_PARALLEL_FOR_H_

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <string>
#include <vector>

#include "common/config_dump.h"
#include "utils/thread_name.h"

namespace dfkv {

// The caller always participates, so work starts even when every helper is
// occupied. Helpers are created lazily and reused by client fan-out and storage
// batches; no request creates or joins a thread.
class ParallelExecutor {
 public:
  static ParallelExecutor& Instance() {
    static ParallelExecutor executor;
    return executor;
  }

  ~ParallelExecutor() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& worker : workers_)
      if (worker.joinable()) worker.join();
  }

  void Run(size_t count, size_t concurrency,
           const std::function<void(size_t)>& function) {
    if (count == 0) return;
    concurrency = std::min(concurrency, count);
    concurrency = std::min(concurrency, MaxThreads());
    if (concurrency <= 1) {
      for (size_t i = 0; i < count; ++i) function(i);
      return;
    }

    auto job = std::make_shared<Job>();
    job->count = count;
    job->function = function;
    const size_t helpers = concurrency - 1;
    size_t outstanding;
    {
      std::lock_guard<std::mutex> lock(mu_);
      for (size_t i = 0; i < helpers; ++i) queue_.push_back(job);
      outstanding = queue_.size();
    }
    EnsureThreads(outstanding);
    cv_.notify_all();
    Work(*job);
    std::unique_lock<std::mutex> lock(job->mu);
    job->done_cv.wait(lock, [&] {
      return job->done.load(std::memory_order_acquire) == job->count;
    });
  }

  static size_t MaxThreads() {
    static const size_t value = [] {
      size_t result = 32;
      const char* text = std::getenv("DFKV_FANOUT_THREADS");
      if (text && *text) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(text, &end, 10);
        if (end != text && *end == '\0' && parsed >= 1 && parsed <= 1024)
          result = static_cast<size_t>(parsed);
      }
      config_dump::RecordResolved("DFKV_FANOUT_THREADS",
                                  std::to_string(result));
      return result;
    }();
    return value;
  }

 private:
  struct Job {
    size_t count = 0;
    std::function<void(size_t)> function;
    std::atomic<size_t> next{0};
    std::atomic<size_t> done{0};
    std::mutex mu;
    std::condition_variable done_cv;
  };

  static void Work(Job& job) {
    for (size_t i = job.next.fetch_add(1); i < job.count;
         i = job.next.fetch_add(1)) {
      // Internal item functions encode failures in their output slots. An
      // exception must degrade that item rather than terminate a worker/process.
      try {
        job.function(i);
      } catch (...) {
      }
      if (job.done.fetch_add(1, std::memory_order_acq_rel) + 1 == job.count) {
        std::lock_guard<std::mutex> lock(job.mu);
        job.done_cv.notify_all();
      }
    }
  }

  void EnsureThreads(size_t wanted) {
    std::lock_guard<std::mutex> lock(mu_);
    while (workers_.size() < wanted && workers_.size() < MaxThreads()) {
      const size_t index = workers_.size();
      workers_.emplace_back([this, index] {
        NameThisThread("dfkv-work-", index);
        Loop();
      });
    }
  }

  void Loop() {
    for (;;) {
      std::shared_ptr<Job> job;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
        if (stop_ && queue_.empty()) return;
        job = std::move(queue_.front());
        queue_.pop_front();
      }
      Work(*job);
    }
  }

  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<Job>> queue_;
  std::vector<std::thread> workers_;
  bool stop_ = false;
};

inline void RunParallel(size_t count, size_t concurrency,
                        const std::function<void(size_t)>& function) {
  ParallelExecutor::Instance().Run(count, concurrency, function);
}

}  // namespace dfkv

#endif  // DFKV_PARALLEL_FOR_H_
