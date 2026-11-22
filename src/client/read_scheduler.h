/* Process-wide bounded scheduler for synchronous RDMA batch shards.
 *
 * Public callers still wait for their complete batch, but they never execute a
 * shard themselves. A fixed worker set therefore bounds blocking CQ waiters,
 * active QPs, and server receive-segment leases across every KVClient handle in
 * the process. Jobs are requeued one shard at a time, giving concurrent batches
 * round-robin progress instead of allowing one large batch to monopolize the
 * workers.
 *
 * One cache node may not hold every worker. A shard blocks its worker for as
 * long as its node takes to answer, and a batch reaches one node through up to
 * eight shards: a node that answers slowly would otherwise occupy all workers,
 * and reads of every other node would wait behind it for the length of one of
 * its operations. Shards of one node run on at most WorkerCount() - 1 workers
 * at a time, so one worker is always free for the others, also for a read
 * that arrives while the slow node's batch is running. Shards submitted
 * without a node are not limited.
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
#include <unordered_map>
#include <vector>

#include "common/config_dump.h"
#include "utils/env_parse.h"
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

  // node_of(i) names the cache node shard i talks to; it is what the per-node
  // limit counts by. Without it the shards are not attributed to a node and
  // are not limited.
  void Run(size_t count, const std::function<void(size_t)>& function,
           const std::function<const std::string&(size_t)>& node_of = nullptr) {
    if (count == 0) return;
    auto job = std::make_shared<Job>();
    job->count = count;
    job->function = function;
    job->enqueued_at = Clock::now();
    job->node.assign(count, kNoNode);
    if (node_of) {
      for (size_t i = 0; i < count; ++i) {
        const uint64_t key = std::hash<std::string>{}(node_of(i));
        job->node[i] = key == kNoNode ? 1 : key;
      }
    }
    for (size_t i = 0; i < count; ++i) job->pending.push_back(i);
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
    counter("dfkv_read_scheduler_node_limit_waits_total",
            "Free read-worker wake-ups that found only shards of nodes at "
            "their per-node limit",
            node_limit_waits_.load());
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
      envparse::ReportUnlessPlain("DFKV_RDMA_READ_WORKERS", 1, 64,
                                  std::to_string(result));
      config_dump::RecordResolved("DFKV_RDMA_READ_WORKERS",
                                  std::to_string(result));
      return result;
    }();
    return value;
  }

  // Most workers shards of one node may hold at a time: all but one. With a
  // single worker nothing is reserved.
  static size_t NodeLimit() {
    const size_t workers = WorkerCount();
    return workers > 1 ? workers - 1 : 1;
  }

 private:
  using Clock = std::chrono::steady_clock;
  static constexpr uint64_t kNoNode = 0;

  struct Job {
    size_t count = 0;
    std::function<void(size_t)> function;
    std::vector<uint64_t> node;   // per shard: its node's key, or kNoNode
    std::deque<size_t> pending;   // shards not started yet (guarded by mu_)
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

  // Picks the next shard to run: the first one, in queue order, whose node
  // is below its limit. A job that still has shards left goes to the back of
  // the queue, as before. Caller holds mu_.
  bool TakeLocked(std::shared_ptr<Job>* job, size_t* index, uint64_t* node) {
    const size_t limit = NodeLimit();
    for (auto it = queue_.begin(); it != queue_.end(); ++it) {
      Job& candidate = **it;
      for (auto shard = candidate.pending.begin();
           shard != candidate.pending.end(); ++shard) {
        const uint64_t key = candidate.node[*shard];
        if (key != kNoNode) {
          const auto active = active_by_node_.find(key);
          if (active != active_by_node_.end() && active->second >= limit)
            continue;
        }
        *index = *shard;
        *node = key;
        candidate.pending.erase(shard);
        *job = *it;
        queue_.erase(it);
        if (key != kNoNode) ++active_by_node_[key];
        pending_shards_.fetch_sub(1, std::memory_order_relaxed);
        if (!(*job)->pending.empty()) {
          queue_.push_back(*job);
          fairness_yields_.fetch_add(1, std::memory_order_relaxed);
          ready_.notify_one();
        }
        return true;
      }
    }
    if (!queue_.empty())
      node_limit_waits_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  void WorkerLoop() {
    for (;;) {
      std::shared_ptr<Job> job;
      size_t index = 0;
      uint64_t node = kNoNode;
      {
        std::unique_lock<std::mutex> lock(mu_);
        while (!TakeLocked(&job, &index, &node)) {
          if (stop_ && queue_.empty()) return;
          ready_.wait(lock);
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
      if (node != kNoNode) {
        {
          std::lock_guard<std::mutex> lock(mu_);
          // The entry stays at zero: no allocation per shard under mu_. The
          // keys are the nodes read from, bounded by the ring and its churn.
          --active_by_node_.find(node)->second;
        }
        // A worker may be waiting for exactly this node to drop below its
        // limit.
        ready_.notify_one();
      }
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
  // Shards running now, per node key (guarded by mu_).
  std::unordered_map<uint64_t, size_t> active_by_node_;
  std::atomic<uint64_t> pending_batches_{0};
  std::atomic<uint64_t> pending_shards_{0};
  std::atomic<uint64_t> active_shards_{0};
  std::atomic<uint64_t> queue_delay_us_{0};
  std::atomic<uint64_t> fairness_yields_{0};
  std::atomic<uint64_t> node_limit_waits_{0};
};

inline void RunReadScheduled(
    size_t count, const std::function<void(size_t)>& function,
    const std::function<const std::string&(size_t)>& node_of = nullptr) {
  ReadScheduler::Instance().Run(count, function, node_of);
}
}  // namespace dfkv

#endif  // DFKV_CLIENT_READ_SCHEDULER_H_
