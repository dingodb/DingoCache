/* Best-effort NUMA placement for RDMA buffers and server serve threads.
 * On multi-socket nodes the RDMA NICs are split across sockets; if a NIC DMAs
 * from memory on the far socket the transfer crosses the inter-socket link and
 * caps aggregate bandwidth (multi-rail then can't beat a single local NIC). This
 * binds a connection's registered buffers — and the server's per-connection serve
 * thread — to the NUMA node of its RDMA device, so each rail stays local.
 *
 * Opt-in via env DFKV_RDMA_NUMA (off by default; only meaningful on multi-socket
 * hosts). When on it ALSO drives NUMA-aware rail selection: the client prefers a
 * rail on the calling thread's NUMA node (see rail_select.h / PickRail). All ops
 * are best-effort: unknown node / sysfs absent => no-op / round-robin all rails.
 * Uses raw syscalls + sysfs so there is no libnuma build dependency. */
#ifndef DFKV_NUMA_UTIL_H_
#define DFKV_NUMA_UTIL_H_

#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cstring>

#include "utils/env_parse.h"

namespace dfkv {
namespace numa {

// DFKV_RDMA_NUMA as the environment has it now. Only "0" is off.
inline bool EnabledFromEnv() {
  return envparse::EnvSwitch("DFKV_RDMA_NUMA", false,
                             envparse::OffWhen::kExactlyZero);
}
inline bool Enabled() {
  static const bool v = EnabledFromEnv();
  return v;
}

// NUMA node of an RDMA device (sysfs), or -1 if unknown / single-node.
inline int DeviceNode(const char* dev) {
  if (!dev || !*dev) return -1;
  char path[256];
  std::snprintf(path, sizeof(path), "/sys/class/infiniband/%s/device/numa_node", dev);
  FILE* f = std::fopen(path, "r");
  if (!f) return -1;
  int n = -1;
  if (std::fscanf(f, "%d", &n) != 1) n = -1;
  std::fclose(f);
  return n;  // -1 on single-node systems
}

// Bind [addr, len) (page-aligned) to `node` so its pages fault in locally.
// MPOL_BIND = 2. Best-effort.
inline void BindMemory(void* addr, size_t len, int node) {
  if (!Enabled() || node < 0 || node >= 1024) return;
  unsigned long mask[1024 / (8 * sizeof(unsigned long))] = {0};
  mask[node / (8 * sizeof(unsigned long))] |= 1UL << (node % (8 * sizeof(unsigned long)));
  ::syscall(SYS_mbind, addr, len, 2 /*MPOL_BIND*/, mask, 1024UL, 0u);
}

// Number of online NUMA nodes (sysfs "online" list like "0-1"), or 1 on any
// parse failure. Ungated: callers decide their own policy.
inline int OnlineNodeCount() {
  FILE* f = std::fopen("/sys/devices/system/node/online", "r");
  if (!f) return 1;
  char buf[256] = {0};
  size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
  std::fclose(f);
  if (n == 0) return 1;
  int count = 0;
  const char* p = buf;
  while (*p) {
    int lo = -1, hi = -1, consumed = 0;
    if (std::sscanf(p, "%d-%d%n", &lo, &hi, &consumed) >= 2) count += hi - lo + 1;
    else if (std::sscanf(p, "%d%n", &lo, &consumed) >= 1) count += 1;
    else break;
    p += consumed;
    while (*p == ',' || *p == ' ' || *p == '\n') ++p;
  }
  return count > 0 ? count : 1;
}

// Spread [addr, len) pages round-robin across nodes [0, nodes) so a large
// shared arena doesn't land entirely on the pre-faulting thread's socket
// (worst case: every remote-socket consumer pays the interconnect on every
// access). MPOL_INTERLEAVE = 3. Ungated + best-effort; call BEFORE the pages
// are first touched (policy only affects new allocations).
inline void InterleaveMemory(void* addr, size_t len, int nodes) {
  if (nodes <= 1 || nodes >= 1024) return;
  unsigned long mask[1024 / (8 * sizeof(unsigned long))] = {0};
  for (int n = 0; n < nodes; ++n)
    mask[n / (8 * sizeof(unsigned long))] |= 1UL << (n % (8 * sizeof(unsigned long)));
  ::syscall(SYS_mbind, addr, len, 3 /*MPOL_INTERLEAVE*/, mask, 1024UL, 0u);
}

// Pin the calling thread to the CPUs of `node` (sysfs cpulist). Best-effort.
inline void PinThreadToNode(int node) {
  if (!Enabled() || node < 0) return;
  char path[256];
  std::snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
  FILE* f = std::fopen(path, "r");
  if (!f) return;
  char buf[4096] = {0};
  size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
  std::fclose(f);
  if (n == 0) return;
  cpu_set_t set;
  CPU_ZERO(&set);
  // cpulist like "0-23,48-71"
  const char* p = buf;
  while (*p) {
    int lo = -1, hi = -1;
    int consumed = 0;
    if (std::sscanf(p, "%d-%d%n", &lo, &hi, &consumed) >= 2) {
      for (int c = lo; c <= hi && c < CPU_SETSIZE; ++c) CPU_SET(c, &set);
    } else if (std::sscanf(p, "%d%n", &lo, &consumed) >= 1) {
      if (lo >= 0 && lo < CPU_SETSIZE) CPU_SET(lo, &set);
    } else {
      break;
    }
    p += consumed;
    while (*p == ',' || *p == ' ' || *p == '\n') ++p;
  }
  if (CPU_COUNT(&set) > 0) ::sched_setaffinity(0, sizeof(set), &set);
}

// True if cpu id is present in a sysfs cpulist like "0-3,8-11" or "5".
inline bool CpuInList(const char* list, int cpu) {
  if (!list) return false;
  const char* p = list;
  while (*p) {
    int lo = -1, hi = -1, consumed = 0;
    if (std::sscanf(p, "%d-%d%n", &lo, &hi, &consumed) >= 2) {
      if (cpu >= lo && cpu <= hi) return true;
    } else if (std::sscanf(p, "%d%n", &lo, &consumed) >= 1) {
      if (cpu == lo) return true;
    } else {
      break;
    }
    p += consumed;
    while (*p == ',' || *p == ' ' || *p == '\n') ++p;
  }
  return false;
}

// Calls fn(cpu) for every CPU id in a sysfs cpulist like "0-3,8-11".
template <typename Fn>
inline void ForEachCpuInList(const char* list, Fn fn) {
  const char* p = list;
  while (*p) {
    int lo = -1, hi = -1, consumed = 0;
    if (std::sscanf(p, "%d-%d%n", &lo, &hi, &consumed) >= 2) {
      for (int cpu = lo; cpu >= 0 && cpu <= hi && cpu < 65536; ++cpu) fn(cpu);
    } else if (std::sscanf(p, "%d%n", &lo, &consumed) >= 1) {
      if (lo >= 0 && lo < 65536) fn(lo);
    } else {
      break;
    }
    p += consumed;
    while (*p == ',' || *p == ' ' || *p == '\n') ++p;
  }
}

// NUMA node of each CPU id from sysfs, -1 where unknown. Read once: a CPU
// does not change nodes, and the scan (an open per possible node, most of
// which do not exist) used to run on every RDMA connection acquire.
inline const std::vector<int>& CpuNodeTable() {
  static const std::vector<int> table = [] {
    std::vector<int> out;
    for (int node = 0; node < 256; ++node) {
      char path[256];
      std::snprintf(path, sizeof(path),
                    "/sys/devices/system/node/node%d/cpulist", node);
      FILE* f = std::fopen(path, "r");
      if (!f) continue;
      char buf[8192] = {0};
      size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
      std::fclose(f);
      // A list that filled the buffer was likely truncated mid-token; leave
      // this node's CPUs unknown rather than parse a severed range (they get
      // -1 -> round-robin all rails, never a wrong NUMA classification).
      if (n == 0 || n == sizeof(buf) - 1) continue;
      ForEachCpuInList(buf, [&](int cpu) {
        if (static_cast<size_t>(cpu) >= out.size()) out.resize(cpu + 1, -1);
        out[static_cast<size_t>(cpu)] = node;
      });
    }
    return out;
  }();
  return table;
}

// NUMA node of the calling thread's current CPU, or -1 if unknown/single.
inline int CurrentNode() {
  const int cpu = ::sched_getcpu();
  if (cpu < 0) return -1;
  const std::vector<int>& table = CpuNodeTable();
  return static_cast<size_t>(cpu) < table.size()
             ? table[static_cast<size_t>(cpu)]
             : -1;
}

}  // namespace numa
}  // namespace dfkv

#endif  // DFKV_NUMA_UTIL_H_
