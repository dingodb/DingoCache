/* dfkv_gpu_dedup_repro — same-host lockstep GET repro against a LIVE server.
 *
 * Reproduces the vLLM PD decode pattern without vLLM: N ranks (separate
 * processes, one CUDA device each) issue the SAME key batch with device
 * destinations at the same instant. With DFKV_CLIENT_NODE_DEDUP=1 +
 * DFKV_CLIENT_NODE_DEDUP_GPU=1 in the environment the ranks rendezvous over
 * CUDA IPC; the tool prints the per-rank dedup split so the server-side read
 * counters can be reconciled key by key. Payloads are content-checked, so a
 * wrong-bytes rendezvous fails loudly here rather than as garbage KV.
 *
 *   dfkv_gpu_dedup_repro --members node=ip:port [--keys 1024] [--size 1048576]
 *                        [--ranks 4] [--namespace schema-id]
 *
 * Master PUTs the keys (host buffers), then forks+execs one child per rank
 * (CUDA does not survive fork). Children barrier on a shared file so their
 * batches truly overlap. Exit code 0 = every rank saw every key with the
 * right bytes.
 */
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "client/cuda_ipc.h"
#include "client/kv_client.h"

using namespace dfkv;

namespace {

struct Args {
  std::string members;
  size_t keys = 1024;
  size_t size = 1 << 20;
  int ranks = 4;
  std::string key_namespace = "dfkv/gpu-dedup-repro";
  int rank = -1;  // >=0: child mode
  std::string barrier;
};

std::string KeyName(size_t i) { return "gpudedup-repro-" + std::to_string(i); }

char PatByte(size_t key, size_t off) {
  return static_cast<char>((key * 131 + off * 7 + 3) & 0xFF);
}


int Child(const Args& a) {
  const CudaLib* cu = CudaLib::Get();
  if (!cu || !cu->BindPrimaryCtx(a.rank)) {
    std::fprintf(stderr, "rank %d: no CUDA ctx on device %d\n", a.rank, a.rank);
    return 2;
  }
  CUdeviceptr pool = 0;
  const size_t pool_bytes = a.keys * a.size;
  if (cu->MemAlloc(&pool, pool_bytes) != kCudaSuccess) {
    std::fprintf(stderr, "rank %d: cuMemAlloc(%zu) failed\n", a.rank, pool_bytes);
    return 2;
  }
  std::vector<std::pair<std::string, std::string>> members;
  {
    const std::string& m = a.members;
    size_t eq = m.find('=');
    members.emplace_back(m.substr(0, eq), m.substr(eq + 1));
  }
  KVClient c(members, a.key_namespace);
  if (!c.RegisterMemory(reinterpret_cast<void*>(pool), pool_bytes)) {
    std::fprintf(stderr, "rank %d: RDMA memory registration failed\n", a.rank);
    return 2;
  }

  std::vector<KvGetItemSg> items(a.keys);
  for (size_t i = 0; i < a.keys; ++i) {
    items[i].key = KeyName(i);
    items[i].dsts = {reinterpret_cast<void*>(pool + i * a.size)};
    items[i].caps = {a.size};
  }

  // Barrier: every rank touches the file, then waits for `ranks` lines.
  {
    FILE* f = std::fopen(a.barrier.c_str(), "a");
    std::fprintf(f, "%d\n", a.rank);
    std::fclose(f);
    for (;;) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      FILE* r = std::fopen(a.barrier.c_str(), "r");
      int lines = 0, ch;
      while ((ch = std::fgetc(r)) != EOF)
        if (ch == '\n') ++lines;
      std::fclose(r);
      if (lines >= a.ranks) break;
    }
  }

  auto t0 = std::chrono::steady_clock::now();
  std::vector<size_t> lens;
  auto res = c.BatchGetAutoSg(items, &lens);
  const double dt = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();

  size_t ok = 0, bad_bytes = 0;
  std::vector<char> host(a.size);
  for (size_t i = 0; i < a.keys; ++i) {
    if (!res[i] || lens[i] != a.size) continue;
    // Verify content (spot-check whole payload; D2H via UVA copy).
    if (cu->Memcpy(reinterpret_cast<CUdeviceptr>(host.data()),
                   pool + i * a.size, a.size) != kCudaSuccess)
      continue;
    bool good = true;
    for (size_t off = 0; off < a.size; off += 4099)
      if (host[off] != PatByte(i, off)) { good = false; break; }
    if (good) ++ok; else ++bad_bytes;
  }
  std::printf("rank %d: ok=%zu/%zu bad_bytes=%zu dt=%.3fs  %s\n", a.rank, ok,
              a.keys, bad_bytes, dt, c.MetricsSnapshot().find("gpu_dedup") !=
              std::string::npos ? "gpu-dedup-active" : "gpu-dedup-INACTIVE");
  // The split lands via DFKV_CLIENT_NODE_DEDUP_LOG=1 (kv_client INFO line).
  std::fflush(stdout);
  return (ok == a.keys && bad_bytes == 0) ? 0 : 3;
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  std::vector<char*> rest;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&] { return std::string(argv[++i]); };
    if (s == "--members") a.members = next();
    else if (s == "--keys") a.keys = std::stoull(next());
    else if (s == "--size") a.size = std::stoull(next());
    else if (s == "--ranks") a.ranks = std::stoi(next());
    else if (s == "--namespace") a.key_namespace = next();
    else if (s == "--rank") a.rank = std::stoi(next());
    else if (s == "--barrier") a.barrier = next();
  }
  if (a.members.empty()) {
    std::fprintf(stderr, "usage: %s --members node=ip:port [--keys N] [--size B] [--ranks R]\n",
                 argv[0]);
    return 1;
  }
  if (a.rank >= 0) return Child(a);

  // Master: PUT the dataset (host buffers), then launch the lockstep ranks.
  {
    std::vector<std::pair<std::string, std::string>> members;
    size_t eq = a.members.find('=');
    members.emplace_back(a.members.substr(0, eq), a.members.substr(eq + 1));
    KVClient c(members, a.key_namespace);
    std::vector<std::string> bufs(a.keys);
    std::vector<KvPutItemSg> puts(a.keys);
    for (size_t i = 0; i < a.keys; ++i) {
      bufs[i].resize(a.size);
      for (size_t off = 0; off < a.size; ++off) bufs[i][off] = PatByte(i, off);
      puts[i].key = KeyName(i);
      puts[i].ptrs = {bufs[i].data()};
      puts[i].sizes = {a.size};
    }
    auto pr = c.BatchPutSg(puts);
    size_t stored = 0;
    for (bool b : pr) stored += b;
    std::printf("master: stored %zu/%zu keys (%zu MB)\n", stored, a.keys,
                stored * a.size >> 20);
    if (stored != a.keys) return 1;
  }

  char bar[] = "/tmp/gpudedup-repro-XXXXXX";
  int bfd = ::mkstemp(bar);
  if (bfd >= 0) ::close(bfd);
  std::vector<pid_t> kids;
  for (int r = 0; r < a.ranks; ++r) {
    pid_t pid = ::fork();
    if (pid == 0) {
      ::execl("/proc/self/exe", argv[0], "--members", a.members.c_str(),
              "--keys", std::to_string(a.keys).c_str(),
              "--size", std::to_string(a.size).c_str(),
              "--ranks", std::to_string(a.ranks).c_str(),
              "--namespace", a.key_namespace.c_str(),
              "--rank", std::to_string(r).c_str(),
              "--barrier", bar, static_cast<char*>(nullptr));
      _exit(97);
    }
    kids.push_back(pid);
  }
  int rc = 0;
  for (pid_t p : kids) {
    int st = 0;
    ::waitpid(p, &st, 0);
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) rc = 4;
  }
  ::unlink(bar);
  std::printf("master: %s\n", rc == 0 ? "ALL RANKS OK" : "FAILURES");
  return rc;
}
