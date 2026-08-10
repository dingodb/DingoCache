/* LatencyHist — fixed-bucket, lock-free latency histogram (Prometheus style) and
 * Sampler — a power-of-two 1-in-N gate. Together they let the server record op
 * latency on the hot path at near-zero cost: Sampler::ShouldSample() is one
 * relaxed atomic add + a mask, and only the sampled (e.g. 1/64) ops read a clock
 * and call Observe() (a bucket scan + relaxed atomic add). No locks, no alloc. */
#ifndef DFKV_LATENCY_HIST_H_
#define DFKV_LATENCY_HIST_H_

#include <atomic>
#include <cstdint>
#include <string>

namespace dfkv {

class LatencyHist {
 public:
  // Cumulative bucket upper bounds in SECONDS (50us .. 60s). Fast data-plane
  // observations still return at the first matching bucket; the long tail keeps
  // control-plane and transport timeouts quantile-observable instead of folding
  // every 5-60s failure into +Inf.
  static constexpr int kNB = 19;
  static constexpr double kBounds[kNB] = {
      50e-6, 100e-6, 250e-6, 500e-6, 1e-3, 2.5e-3, 5e-3, 10e-3,
      25e-3, 50e-3, 100e-3, 250e-3, 500e-3, 1.0, 2.5, 5.0,
      10.0, 30.0, 60.0};

  void Observe(double seconds) {
    if (seconds < 0) seconds = 0;
    // Accumulate nanoseconds (not µs) so sub-µs/fractional-µs cache-hit latencies
    // aren't floored to 0 — at µs granularity _sum was biased low for fast ops.
    sum_ns_.fetch_add(static_cast<uint64_t>(seconds * 1e9), std::memory_order_relaxed);
    count_.fetch_add(1, std::memory_order_relaxed);
    for (int i = 0; i < kNB; ++i) {
      if (seconds <= kBounds[i]) { buckets_[i].fetch_add(1, std::memory_order_relaxed); return; }
    }
    inf_.fetch_add(1, std::memory_order_relaxed);
  }

  uint64_t Count() const { return count_.load(std::memory_order_relaxed); }
  double Sum() const {
    return static_cast<double>(sum_ns_.load(std::memory_order_relaxed)) / 1e9;
  }

  // Full Prometheus block (HELP + TYPE + body). `labels` is the inner label set
  // (e.g. `op="get"`) or empty. Buckets are cumulative (le).
  std::string Render(const std::string& name, const std::string& labels) const {
    return "# HELP " + name + " Operation latency seconds\n" +
           "# TYPE " + name + " histogram\n" + RenderBody(name, labels);
  }

  // Body only (no HELP/TYPE) — for emitting several label sets of the SAME metric
  // (e.g. op="get" and op="put") under a single HELP/TYPE header.
  std::string RenderBody(const std::string& name, const std::string& labels) const {
    auto lbl = [&](const std::string& extra) {
      std::string in = extra;
      if (!labels.empty()) in += (in.empty() ? "" : ",") + labels;
      return in.empty() ? std::string() : "{" + in + "}";
    };
    std::string s;
    uint64_t cum = 0;
    char le[32];
    for (int i = 0; i < kNB; ++i) {
      cum += buckets_[i].load(std::memory_order_relaxed);
      std::snprintf(le, sizeof(le), "le=\"%g\"", kBounds[i]);
      s += name + "_bucket" + lbl(le) + " " + std::to_string(cum) + "\n";
    }
    uint64_t total = cum + inf_.load(std::memory_order_relaxed);
    s += name + "_bucket" + lbl("le=\"+Inf\"") + " " + std::to_string(total) + "\n";
    char sum[64];
    std::snprintf(sum, sizeof(sum), "%.6f", Sum());
    s += name + "_sum" + lbl("") + " " + sum + "\n";
    s += name + "_count" + lbl("") + " " + std::to_string(total) + "\n";
    return s;
  }

 private:
  std::atomic<uint64_t> buckets_[kNB]{};
  std::atomic<uint64_t> inf_{0};
  std::atomic<uint64_t> sum_ns_{0};
  std::atomic<uint64_t> count_{0};
};

// 1-in-N sampler rounded up to a power of two so the gate is a mask (no divide).
class Sampler {
 public:
  explicit Sampler(uint64_t one_in_n) {
    uint64_t p = 1;
    while (p < one_in_n) p <<= 1;
    mask_ = p - 1;
  }
  bool ShouldSample() {
    return (seq_.fetch_add(1, std::memory_order_relaxed) & mask_) == 0;
  }

 private:
  std::atomic<uint64_t> seq_{0};
  uint64_t mask_;
};

}  // namespace dfkv

#endif  // DFKV_LATENCY_HIST_H_
