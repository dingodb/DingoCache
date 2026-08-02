/* OpMetrics — per-op client-side counters + latency histogram, rendered as
 * dfkv_client_op_* in KVClient::MetricsSnapshot().
 *
 * This is the convergent request view: every connector funnels through the
 * KVClient public batch + scalar ops. Each public call records exactly one
 * request and its submitted key count, regardless of transport, retry,
 * rendezvous hit, or early route/health failure.
 *
 * Lock-free: relaxed atomics + the lock-free LatencyHist. `hits` is op-specific
 * (put/remove: keys confirmed; get: cache hits; exist: keys present), so the
 * dashboard derives hit/fail ratio = hits/keys. Latency captures the public
 * call, including rendezvous waits and any bounded native retries. */
#ifndef DFKV_OP_METRICS_H_
#define DFKV_OP_METRICS_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>

#include "utils/latency_hist.h"

namespace dfkv {

class OpMetrics {
 public:
  enum Op { kPut = 0, kGet = 1, kExist = 2, kRemove = 3, kNumOps = 4 };

  void Record(Op op, uint64_t keys, uint64_t hits, uint64_t bytes,
              double seconds) {
    M& m = m_[op];
    m.requests.fetch_add(1, std::memory_order_relaxed);
    m.keys.fetch_add(keys, std::memory_order_relaxed);
    m.hits.fetch_add(hits, std::memory_order_relaxed);
    m.bytes.fetch_add(bytes, std::memory_order_relaxed);
    if (seconds < 0) seconds = 0;
    uint64_t ns = static_cast<uint64_t>(seconds * 1e9);
    uint64_t cur = m.max_ns.load(std::memory_order_relaxed);
    while (ns > cur &&
           !m.max_ns.compare_exchange_weak(cur, ns, std::memory_order_relaxed)) {
    }
    m.lat.Observe(seconds);
  }

  // RAII timer for the single-op paths (many early returns): set .hits/.bytes
  // before the function returns; the dtor records keys=1 + the elapsed time.
  struct Scope {
    OpMetrics* m;
    Op op;
    std::chrono::steady_clock::time_point t0;
    uint64_t keys = 1, hits = 0, bytes = 0;
    Scope(OpMetrics* m_, Op op_)
        : m(m_), op(op_), t0(std::chrono::steady_clock::now()) {}
    ~Scope() {
      double s = std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t0).count();
      m->Record(op, keys, hits, bytes, s);
    }
  };
  Scope Begin(Op op) { return Scope(this, op); }

  std::string Render() const {
    static const char* kName[kNumOps] = {"put", "get", "exist", "remove"};
    std::string s;
    Counter(s, "dfkv_client_op_requests_total", "client ops issued, by op",
            [](const M& m) { return m.requests.load(std::memory_order_relaxed); });
    Counter(s, "dfkv_client_op_keys_total", "keys submitted, by op",
            [](const M& m) { return m.keys.load(std::memory_order_relaxed); });
    Counter(s, "dfkv_client_op_hits_total",
            "keys confirmed (put/remove) / hit (get) / present (exist), by op",
            [](const M& m) { return m.hits.load(std::memory_order_relaxed); });
    Counter(s, "dfkv_client_op_bytes_total", "bytes moved, by op",
            [](const M& m) { return m.bytes.load(std::memory_order_relaxed); });
    s += "# HELP dfkv_client_op_max_seconds peak op latency seconds (lifetime), by op\n";
    s += "# TYPE dfkv_client_op_max_seconds gauge\n";
    for (int o = 0; o < kNumOps; ++o)
      s += std::string("dfkv_client_op_max_seconds{op=\"") + kName[o] + "\"} " +
           FmtSeconds(m_[o].max_ns.load(std::memory_order_relaxed)) + "\n";
    s += "# HELP dfkv_client_op_latency_seconds client op latency seconds, by op\n";
    s += "# TYPE dfkv_client_op_latency_seconds histogram\n";
    for (int o = 0; o < kNumOps; ++o)
      s += m_[o].lat.RenderBody("dfkv_client_op_latency_seconds",
                                std::string("op=\"") + kName[o] + "\"");
    return s;
  }

 private:
  struct M {
    std::atomic<uint64_t> requests{0}, keys{0}, hits{0}, bytes{0}, max_ns{0};
    LatencyHist lat;
  };

  template <class Pick>
  void Counter(std::string& s, const char* name, const char* help,
               Pick pick) const {
    static const char* kName[kNumOps] = {"put", "get", "exist", "remove"};
    s += "# HELP "; s += name; s += " "; s += help; s += "\n";
    s += "# TYPE "; s += name; s += " counter\n";
    for (int o = 0; o < kNumOps; ++o)
      s += std::string(name) + "{op=\"" + kName[o] + "\"} " +
           std::to_string(pick(m_[o])) + "\n";
  }

  static std::string FmtSeconds(uint64_t ns) {
    char b[32];
    std::snprintf(b, sizeof(b), "%.6f", static_cast<double>(ns) / 1e9);
    return b;
  }

  M m_[kNumOps];
};

}  // namespace dfkv

#endif  // DFKV_OP_METRICS_H_
