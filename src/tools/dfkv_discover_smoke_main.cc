/* dfkv_discover_smoke — MDS-based discovery end-to-end smoke client (M4).
 * Constructs a KVClient with NO static members, starts MDS discovery, waits
 * for the ring to populate, then Puts N keys and Gets them back, reporting the
 * hit rate.  Designed to run on a real node against deployed etcd+MDS+dfkv_server.
 *
 * Usage:
 *   dfkv_discover_smoke --mds <ep1[,ep2,...]> [--group <g>] [--count <N>]
 *                        [--size <bytes>] [--poll-ms <ms>] [--timeout-s <s>]
 *
 * Exit codes:
 *   0  PASS (all puts/gets succeeded, no mismatches)
 *   1  FAIL (ring ready but put/get errors or mismatches)
 *   2  ring never became ready within timeout
 */
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "client/kv_client.h"
#include "common/version.h"

using dfkv::KVClient;

static std::vector<std::string> SplitComma(const std::string& s) {
  std::vector<std::string> out;
  for (size_t i = 0, j; i <= s.size(); i = j + 1) {
    j = s.find(',', i);
    if (j == std::string::npos) j = s.size();
    if (j > i) out.push_back(s.substr(i, j - i));
  }
  return out;
}

int main(int argc, char** argv) {
  if (dfkv::WantsVersion(argc, argv)) { std::printf("dfkv_discover_smoke %s\n", dfkv::Version()); return 0; }
  std::string mds_arg;
  std::string group   = "default";
  int         count   = 200;
  int         size    = 1024;
  int         poll_ms = 500;
  int         timeout_s = 15;

  for (int i = 1; i + 1 < argc; i += 2) {
    if      (!std::strcmp(argv[i], "--mds"))       mds_arg   = argv[i + 1];
    else if (!std::strcmp(argv[i], "--group"))     group     = argv[i + 1];
    else if (!std::strcmp(argv[i], "--count"))     count     = std::atoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--size"))      size      = std::atoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--poll-ms"))   poll_ms   = std::atoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--timeout-s")) timeout_s = std::atoi(argv[i + 1]);
  }

  if (mds_arg.empty()) {
    std::fprintf(stderr, "dfkv_discover_smoke: --mds <ep1[,ep2,...]> is required\n");
    return 2;
  }
  if (count <= 0 || size <= 0 || poll_ms <= 0 || timeout_s <= 0) {
    std::fprintf(stderr, "dfkv_discover_smoke: --count/--size/--poll-ms/--timeout-s must be > 0\n");
    return 2;
  }

  std::vector<std::string> mds_eps = SplitComma(mds_arg);

  KVClient c({}, "dfkv/discovery-smoke");
  c.StartMdsDiscovery(mds_eps, group, poll_ms);

  // --- Readiness probe: loop until a Put succeeds or timeout ---
  std::vector<char> probe(static_cast<size_t>(size), 'w');
  auto t0       = std::chrono::steady_clock::now();
  auto deadline = t0 + std::chrono::seconds(timeout_s);
  bool ready    = false;
  while (!ready && std::chrono::steady_clock::now() < deadline) {
    if (c.Put("dfkv_smoke/warmup_k", probe.data(), probe.size())) {
      ready = true;
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
  auto t_ready = std::chrono::steady_clock::now();
  long ready_ms = static_cast<long>(
      std::chrono::duration_cast<std::chrono::milliseconds>(t_ready - t0).count());

  if (!ready) {
    std::fprintf(stderr,
        "dfkv_discover_smoke: ring not ready after %ds (mds=%s group=%s)\n",
        timeout_s, mds_arg.c_str(), group.c_str());
    return 2;
  }

  // --- Put N keys with a per-i byte pattern ---
  int puts_ok = 0;
  std::vector<char> buf(static_cast<size_t>(size));
  for (int i = 0; i < count; ++i) {
    for (int j = 0; j < size; ++j) buf[j] = static_cast<char>(i * 31 + j);
    std::string key = "dfkv_smoke/k_" + std::to_string(i);
    if (c.Put(key, buf.data(), buf.size())) ++puts_ok;
  }

  // --- Get each key back and compare ---
  int gets_hit  = 0;
  int mismatch  = 0;
  std::vector<char> out(static_cast<size_t>(size));
  for (int i = 0; i < count; ++i) {
    std::string key = "dfkv_smoke/k_" + std::to_string(i);
    if (c.Get(key, out.data(), out.size())) {
      ++gets_hit;
      // Verify payload matches the pattern we wrote
      for (int j = 0; j < size; ++j) {
        if (out[j] != static_cast<char>(i * 31 + j)) { ++mismatch; break; }
      }
    }
  }

  bool pass = (puts_ok == count && gets_hit == count && mismatch == 0);

  std::printf(
      "SMOKE mds=%s group=%s count=%d size=%d ready_ms=%ld"
      " puts_ok=%d gets_hit=%d mismatch=%d\n",
      mds_arg.c_str(), group.c_str(), count, size, ready_ms,
      puts_ok, gets_hit, mismatch);
  std::printf("RESULT %s\n", pass ? "PASS" : "FAIL");
  std::fflush(stdout);

  return pass ? 0 : 1;
}
