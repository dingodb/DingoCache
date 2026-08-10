/* Process-wide bounded scheduler for synchronous RDMA batch shards.
 *
 * Public callers still wait for their complete batch, but they never execute a
 * shard themselves. A fixed worker set therefore bounds blocking CQ waiters,
 * active QPs, and server receive-segment leases across every KVClient handle in
 * the process. Jobs are requeued one shard at a time, giving concurrent batches
 * round-robin progress instead of allowing one large batch to monopolize the
 * workers.
 */
#ifndef DFKV_CLIENT_READ_SCHEDULER_H_
#define DFKV_CLIENT_READ_SCHEDULER_H_

#include <chrono>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <mutex>
#include <thread>
#include <vector>

#include "common/config_dump.h"
#include "utils/thread_name.h"

namespace dfkv {

class ReadScheduler {
 public:
  static ReadScheduler& Instance() {
    static ReadScheduler scheduler;
    return scheduler;
  }

  ~ReadScheduler() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    ready_.notify_all();
    for (auto& worker : workers_)
      if (worker.joinable()) worker.join();
  }

  void Run(size_t count, const std::function<void(size_t)>& function) {
    if (count == 0) return;
    auto job = std::make_shared<Job>();
    job->count = count;
    job->function = function;
    job->enqueued_at = Clock::now();
    pending_batches_.fetch_add(1, std::memory_order_relaxed);
    pending_shards_.fetch_add(count, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(mu_);
      queue_.push_back(job);
      EnsureWorkersLocked();
    }
    ready_.notify_one();

    std::unique_lock<std::mutex> lock(job->mu);
    job->done_cv.wait(lock, [&] {
      return job->done.load(std::memory_order_acquire) == count;
    });
  }

  std::string MetricsText() const {
    std::string out;
    const auto gauge = [&](const char* name, const char* help, uint64_t value) {
      out += "# HELP " + std::string(name) + " " + help + "\n";
      out += "# TYPE " + std::string(name) + " gauge\n";
      out += std::string(name) + " " + std::to_string(value) + "\n";
    };
    const auto counter =
        [&](const char* name, const char* help, uint64_t value) {
          out += "# HELP " + std::string(name) + " " + help + "\n";
          out += "# TYPE " + std::string(name) + " counter\n";
          out += std::string(name) + " " + std::to_string(value) + "\n";
        };
    gauge("dfkv_read_scheduler_pending_batches",
          "RDMA read batches waiting or executing", pending_batches_.load());
    gauge("dfkv_read_scheduler_pending_shards",
          "RDMA read shards not yet admitted to a worker",
          pending_shards_.load());
    gauge("dfkv_read_scheduler_active_shards",
          "RDMA read shards currently executing", active_shards_.load());
    counter("dfkv_read_scheduler_queue_delay_us_total",
            "Cumulative enqueue-to-worker delay for RDMA read shards",
            queue_delay_us_.load());
    counter("dfkv_read_scheduler_fairness_yields_total",
            "Shard-level job requeues that yield to another batch",
            fairness_yields_.load());
    return out;
  }

  static size_t WorkerCount() {
    static const size_t value = [] {
      size_t result = 7;
      const char* text = std::getenv("DFKV_RDMA_READ_WORKERS");
      if (text && *text) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(text, &end, 10);
        if (end != text && *end == '\0' && parsed >= 1 && parsed <= 64)
          result = static_cast<size_t>(parsed);
      }
      config_dump::RecordResolved("DFKV_RDMA_READ_WORKERS",
                                  std::to_string(result));
      return result;
    }();
    return value;
  }

 private:
  using Clock = std::chrono::steady_clock;

  struct Job {
    size_t count = 0;
    std::function<void(size_t)> function;
    std::atomic<size_t> next{0};
    std::atomic<size_t> done{0};
    std::mutex mu;
    std::condition_variable done_cv;
    Clock::time_point enqueued_at;
  };

  void EnsureWorkersLocked() {
    while (workers_.size() < WorkerCount()) {
      const size_t index = workers_.size();
      workers_.emplace_back([this, index] {
        NameThisThread("dfkv-read-", index);
        WorkerLoop();
      });
    }
  }

  void WorkerLoop() {
    for (;;) {
      std::shared_ptr<Job> job;
      size_t index = 0;
      {
        std::unique_lock<std::mutex> lock(mu_);
        ready_.wait(lock, [&] { return stop_ || !queue_.empty(); });
        if (stop_ && queue_.empty()) return;
        job = std::move(queue_.front());
        queue_.pop_front();
        index = job->next.fetch_add(1, std::memory_order_relaxed);
        pending_shards_.fetch_sub(1, std::memory_order_relaxed);
        if (index + 1 < job->count) {
          queue_.push_back(job);
          fairness_yields_.fetch_add(1, std::memory_order_relaxed);
          ready_.notify_one();
        }
      }

      queue_delay_us_.fetch_add(
          static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::microseconds>(
                  Clock::now() - job->enqueued_at)
                  .count()),
          std::memory_order_relaxed);
      active_shards_.fetch_add(1, std::memory_order_relaxed);
      try {
        job->function(index);
      } catch (...) {
        // Batch shard functions encode failures in their output slots.
      }
      active_shards_.fetch_sub(1, std::memory_order_relaxed);
      if (job->done.fetch_add(1, std::memory_order_acq_rel) + 1 == job->count) {
        pending_batches_.fetch_sub(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(job->mu);
        job->done_cv.notify_all();
      }
    }
  }

  std::mutex mu_;
  std::condition_variable ready_;
  std::deque<std::shared_ptr<Job>> queue_;
  std::vector<std::thread> workers_;
  bool stop_ = false;
  std::atomic<uint64_t> pending_batches_{0};
  std::atomic<uint64_t> pending_shards_{0};
  std::atomic<uint64_t> active_shards_{0};
  std::atomic<uint64_t> queue_delay_us_{0};
  std::atomic<uint64_t> fairness_yields_{0};
};

inline void RunReadScheduled(size_t count,
                             const std::function<void(size_t)>& function) {
  ReadScheduler::Instance().Run(count, function);
}
}  // namespace dfkv

#endif  // DFKV_CLIENT_READ_SCHEDULER_H_
