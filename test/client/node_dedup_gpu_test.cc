// GpuNodeDedup: same-host GET rendezvous for device-memory destinations over
// CUDA IPC. Hermetic wrt dfkv (no RDMA/etcd) but NOT wrt hardware: everything
// past the env/name tests needs a CUDA device and skips cleanly without one.
// Cross-process coverage forks AND EXECS this binary — CUDA does not survive a
// bare fork — with the segment name in the environment (see GpuDedupChild).
//
// This file builds two binaries. node_dedup_gpu_test runs against the real
// driver. node_dedup_gpu_sim_test (see that file) runs the same tests against
// the simulated driver in test/client/sim_cuda, on any machine; tests that
// need something only the simulation can do (hold a GPUDirect write back)
// skip on real hardware.
#include "client/node_dedup_gpu.h"

#include <gtest/gtest.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "client/cuda_ipc.h"
#include "client/key_map.h"
#include "client/kv_client.h"
#include "transport/transport.h"

using dfkv::BlockKey;
using dfkv::CudaLib;
using dfkv::CUdeviceptr;
using dfkv::GpuNodeDedup;
using dfkv::kCudaSuccess;

namespace {

// The library never creates CUDA contexts (it rides the framework's); the
// test IS the framework here, so it retains device 0's primary context and
// makes it current on the calling thread. Safe to call per-thread.
bool EnsureCudaCtx() {
  static void* ctx = [] {
    void* h = ::dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!h) return static_cast<void*>(nullptr);
    auto init = reinterpret_cast<int (*)(unsigned)>(::dlsym(h, "cuInit"));
    auto devget = reinterpret_cast<int (*)(int*, int)>(::dlsym(h, "cuDeviceGet"));
    auto retain = reinterpret_cast<int (*)(void**, int)>(
        ::dlsym(h, "cuDevicePrimaryCtxRetain"));
    if (!init || !devget || !retain || init(0) != 0) return static_cast<void*>(nullptr);
    int dev = 0;
    void* c = nullptr;
    if (devget(&dev, 0) != 0 || retain(&c, dev) != 0) return static_cast<void*>(nullptr);
    return c;
  }();
  if (!ctx) return false;
  static auto setcur = reinterpret_cast<int (*)(void*)>(
      ::dlsym(::dlopen("libcuda.so.1", RTLD_NOW), "cuCtxSetCurrent"));
  return setcur && setcur(ctx) == 0;
}

// The simulated driver's controls, or nullptr on real hardware.
template <typename T>
T* SimSymbol(const char* name) {
  return reinterpret_cast<T*>(::dlsym(RTLD_DEFAULT, name));
}
bool Simulated() {
  return ::dlsym(RTLD_DEFAULT, "sim_cuda_gpudirect_write") != nullptr;
}

struct ShmGuard {
  std::string name;
  explicit ShmGuard(const std::string& tag)
      : name("/dfkv-dedup-gputest-" + std::to_string(::getpid()) + "-" + tag) {
    ::shm_unlink(name.c_str());
  }
  ~ShmGuard() { ::shm_unlink(name.c_str()); }
};

GpuNodeDedup::Options Opts(const std::string& name) {
  GpuNodeDedup::Options o;
  o.name = name;
  o.arena_bytes = 32ull << 20;
  o.slots = 1024;
  o.wait_ms = 300;
  o.takeover_ms = 200;
  o.ttl_ms = 30000;  // outlives a child-process exec in the cross-proc test
  return o;
}

// A waiter whose deadline is far beyond any scheduling delay, and claims
// that are never taken over: a test that expects a wait to end early can use
// a bound that is generous and still well short of the deadline.
GpuNodeDedup::Options LongWait(const std::string& name) {
  GpuNodeDedup::Options o = Opts(name);
  o.wait_ms = 5000;
  o.takeover_ms = 600000;
  return o;
}

BlockKey K(uint64_t id) { return BlockKey{id, 0}; }

std::string Val(uint64_t id, size_t n) {
  std::string v(n, '\0');
  for (size_t i = 0; i < n; ++i) v[i] = static_cast<char>((id * 131 + i) & 0xFF);
  return v;
}

// Device buffer with RAII; up/down copies ride the UVA cuMemcpy.
struct DevBuf {
  CUdeviceptr p = 0;
  const CudaLib* cu = CudaLib::Get();
  explicit DevBuf(size_t n) { EXPECT_EQ(cu->MemAlloc(&p, n), kCudaSuccess); }
  ~DevBuf() { if (p) cu->MemFree(p); }
  void Up(const std::string& v) {
    ASSERT_EQ(cu->Memcpy(p, reinterpret_cast<CUdeviceptr>(v.data()), v.size()),
              kCudaSuccess);
  }
  std::string Down(size_t n) {
    std::string v(n, '\0');
    EXPECT_EQ(cu->Memcpy(reinterpret_cast<CUdeviceptr>(v.data()), p, n),
              kCudaSuccess);
    return v;
  }
};

}  // namespace

// ---- no-GPU-required ----

TEST(GpuNodeDedup, EnvSegmentNameCarriesLayoutVersion) {
  const std::string n = GpuNodeDedup::EnvSegmentName(0xabcdef);
  EXPECT_NE(n.find("/dfkv-dedup-gpuv3-"), std::string::npos);
  EXPECT_NE(n.find("0000000000abcdef"), std::string::npos);
  // distinct from the host segment namespace
  EXPECT_EQ(n.find("/dfkv-dedup-v4-"), std::string::npos);
}

TEST(GpuNodeDedup, FromEnvNeedsBothSwitches) {
  ::unsetenv("DFKV_CLIENT_NODE_DEDUP");
  ::unsetenv("DFKV_CLIENT_NODE_DEDUP_GPU");
  EXPECT_EQ(GpuNodeDedup::FromEnv(1, nullptr), nullptr);
  ::setenv("DFKV_CLIENT_NODE_DEDUP", "1", 1);
  EXPECT_EQ(GpuNodeDedup::FromEnv(1, nullptr), nullptr);  // GPU flag still off
  ::unsetenv("DFKV_CLIENT_NODE_DEDUP");
}

// ---- CUDA required below ----

TEST(GpuNodeDedup, FetchPublishThenPeerHitsScattered) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("basic");
  auto a = GpuNodeDedup::Open(Opts(g.name));
  auto b = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  const std::string v = Val(1, 6000);

  // Publisher fetched into TWO device segments (4096 + 4096 caps, 6000 used).
  DevBuf s0(4096), s1(4096);
  s0.Up(v.substr(0, 4096));
  s1.Up(v.substr(4096));
  GpuNodeDedup::Seg src[2] = {{reinterpret_cast<void*>(s0.p), 4096},
                              {reinterpret_cast<void*>(s1.p), 4096}};
  size_t got = 0;
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->ClaimSg(K(1), src, 2, 8192, &got, &token),
            GpuNodeDedup::Role::kFetch);
  a->PublishSg(K(1), token, src, 2, v.size());

  // Peer scatters into a DIFFERENT segment split (3000 + 5192).
  DevBuf d0(3000), d1(5192);
  GpuNodeDedup::Seg dst[2] = {{reinterpret_cast<void*>(d0.p), 3000},
                              {reinterpret_cast<void*>(d1.p), 5192}};
  got = 0;
  ASSERT_EQ(b->ClaimSg(K(1), dst, 2, 8192, &got, &ignored),
            GpuNodeDedup::Role::kHit);
  EXPECT_EQ(got, v.size());
  EXPECT_EQ(d0.Down(3000), v.substr(0, 3000));
  EXPECT_EQ(d1.Down(3000), v.substr(3000));  // 6000-3000 used of 5192 cap
  EXPECT_EQ(b->hits(), 1u);
}

TEST(GpuNodeDedup, WaiterCopiesAfterPublish) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("wait");
  auto a = GpuNodeDedup::Open(Opts(g.name));
  auto b = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  const std::string v = Val(2, 65536);
  DevBuf src(65536), dst(65536);
  src.Up(v);
  GpuNodeDedup::Seg ss{reinterpret_cast<void*>(src.p), 65536};
  GpuNodeDedup::Seg ds{reinterpret_cast<void*>(dst.p), 65536};

  size_t got = 0;
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->ClaimSg(K(2), &ss, 1, 65536, &got, &token),
            GpuNodeDedup::Role::kFetch);
  ASSERT_EQ(b->ClaimSg(K(2), &ds, 1, 65536, &got, &ignored),
            GpuNodeDedup::Role::kWait);
  std::thread pub([&] {
    ASSERT_TRUE(EnsureCudaCtx());
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    a->PublishSg(K(2), token, &ss, 1, v.size());
  });
  EXPECT_TRUE(b->WaitSg(K(2), &ds, 1, 65536, &got));
  pub.join();
  EXPECT_EQ(got, v.size());
  EXPECT_EQ(dst.Down(65536), v);
  EXPECT_EQ(b->wait_hits(), 1u);
}

// The vLLM transfer threads are fresh pthreads with NO current CUDA context
// (the phase-8 PD A/B found the feature silently off because of exactly
// this); every entry point must self-bind the arena device's primary ctx.
TEST(GpuNodeDedup, ThreadWithoutCtxIsServed) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("noctx");
  auto a = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a);
  const std::string v = Val(9, 32768);
  DevBuf src(32768), dst(32768);
  src.Up(v);
  GpuNodeDedup::Seg ss{reinterpret_cast<void*>(src.p), 32768};
  GpuNodeDedup::Seg ds{reinterpret_cast<void*>(dst.p), 32768};
  std::thread t([&] {  // deliberately NO EnsureCudaCtx here
    size_t got = 0;
    uint64_t token = 0;
    ASSERT_EQ(a->ClaimSg(K(9), &ss, 1, 32768, &got, &token),
              GpuNodeDedup::Role::kFetch);
    a->PublishSg(K(9), token, &ss, 1, v.size());
  });
  t.join();
  size_t got = 0;
  uint64_t ignored = 0;
  ASSERT_EQ(a->ClaimSg(K(9), &ds, 1, 32768, &got, &ignored),
            GpuNodeDedup::Role::kHit);
  EXPECT_EQ(got, v.size());
  EXPECT_EQ(dst.Down(32768), v);
}

TEST(GpuNodeDedup, WaitManyAmortizedSync) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("waitmany");
  auto a = GpuNodeDedup::Open(Opts(g.name));
  auto b = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  constexpr size_t kN = 16;
  std::vector<std::string> vals;
  std::vector<std::unique_ptr<DevBuf>> srcs, dsts;
  std::vector<GpuNodeDedup::Seg> ssegs(kN), dsegs(kN);
  std::vector<GpuNodeDedup::WaitItem> wits(kN);
  size_t got = 0;
  std::vector<uint64_t> tokens(kN, 0);
  uint64_t ignored = 0;
  for (size_t i = 0; i < kN; ++i) {
    vals.push_back(Val(100 + i, 8192));
    srcs.push_back(std::make_unique<DevBuf>(8192));
    dsts.push_back(std::make_unique<DevBuf>(8192));
    srcs[i]->Up(vals[i]);
    ssegs[i] = {reinterpret_cast<void*>(srcs[i]->p), 8192};
    dsegs[i] = {reinterpret_cast<void*>(dsts[i]->p), 8192};
    ASSERT_EQ(a->ClaimSg(K(100 + i), &ssegs[i], 1, 8192, &got,
                         &tokens[i]),
              GpuNodeDedup::Role::kFetch);
    ASSERT_EQ(b->ClaimSg(K(100 + i), &dsegs[i], 1, 8192, &got, &ignored),
              GpuNodeDedup::Role::kWait);
    wits[i] = GpuNodeDedup::WaitItem{K(100 + i), &dsegs[i], 1, 8192, false, 0};
  }
  std::thread pub([&] {  // publishes trickle in while the batch wait runs
    ASSERT_TRUE(EnsureCudaCtx());
    for (size_t i = 0; i < kN; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      a->PublishSg(K(100 + i), tokens[i], &ssegs[i], 1,
                   vals[i].size());
    }
  });
  b->WaitManySg(wits.data(), kN);
  pub.join();
  for (size_t i = 0; i < kN; ++i) {
    EXPECT_TRUE(wits[i].ok) << "key " << i;
    EXPECT_EQ(wits[i].got, vals[i].size());
    EXPECT_EQ(dsts[i]->Down(8192), vals[i]) << "key " << i;
  }
  EXPECT_EQ(b->wait_hits(), kN);
}

TEST(GpuNodeDedup, AbortFreesSlotForWaiterFallback) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("abort");
  auto a = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a);
  DevBuf d(4096);
  GpuNodeDedup::Seg s{reinterpret_cast<void*>(d.p), 4096};
  size_t got = 0;
  uint64_t token = 0;
  ASSERT_EQ(a->ClaimSg(K(3), &s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  a->Abort(K(3), token);
  // Slot freed: the next claimant fetches (no stuck FETCHING until takeover).
  ASSERT_EQ(a->ClaimSg(K(3), &s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
}

TEST(GpuNodeDedup, OversizePayloadSkipsRendezvous) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("oversize");
  auto a = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a);
  DevBuf d(4096);
  GpuNodeDedup::Seg s{reinterpret_cast<void*>(d.p), 4096};
  size_t got = 0;
  uint64_t token = 0;
  // cap > arena/2: no slot is claimed, so no publish obligation either.
  EXPECT_EQ(a->ClaimSg(K(4), &s, 1, (32ull << 20), &got, &token),
            GpuNodeDedup::Role::kFetch);
  a->Abort(K(4), token);
  EXPECT_EQ(a->ClaimSg(K(4), &s, 1, (32ull << 20), &got, &token),
            GpuNodeDedup::Role::kFetch);
}

TEST(GpuNodeDedup, InvalidateDropsReadyAndFencesLatePublisher) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("invalidate");
  auto a = GpuNodeDedup::Open(Opts(g.name));
  auto b = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  const std::string value = Val(5, 4096);
  DevBuf src(4096), dst(4096);
  src.Up(value);
  GpuNodeDedup::Seg source{reinterpret_cast<void*>(src.p), 4096};
  GpuNodeDedup::Seg target{reinterpret_cast<void*>(dst.p), 4096};
  size_t got = 0;
  uint64_t token = 0, ignored = 0;

  ASSERT_EQ(a->ClaimSg(K(5), &source, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  a->PublishSg(K(5), token, &source, 1, value.size());
  a->Invalidate(K(5));
  EXPECT_NE(b->ClaimSg(K(5), &target, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kHit);

  ASSERT_EQ(a->ClaimSg(K(6), &source, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  a->Invalidate(K(6));
  a->PublishSg(K(6), token, &source, 1, value.size());
  EXPECT_NE(b->ClaimSg(K(6), &target, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kHit);
}

// Child half of the cross-process test: skipped unless the parent exec'd us
// with the segment name in the environment. Attaches, expects an immediate
// IPC hit on the parent's publish, and verifies the payload byte-for-byte.
TEST(GpuDedupChild, ReadsPeerPublish) {
  const char* seg = ::getenv("DFKV_GPU_DEDUP_TEST_SEG");
  if (!seg) GTEST_SKIP() << "not a child process";
  ASSERT_TRUE(EnsureCudaCtx());
  auto d = GpuNodeDedup::Open(Opts(seg));
  ASSERT_TRUE(d);
  const std::string v = Val(7, 40000);
  DevBuf d0(20000), d1(20000);
  GpuNodeDedup::Seg dst[2] = {{reinterpret_cast<void*>(d0.p), 20000},
                              {reinterpret_cast<void*>(d1.p), 20000}};
  size_t got = 0;
  uint64_t ignored = 0;
  ASSERT_EQ(d->ClaimSg(K(7), dst, 2, 40000, &got, &ignored),
            GpuNodeDedup::Role::kHit);
  ASSERT_EQ(got, v.size());
  EXPECT_EQ(d0.Down(20000), v.substr(0, 20000));
  EXPECT_EQ(d1.Down(20000), v.substr(20000));
}

TEST(GpuNodeDedup, CrossProcessIpcHit) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("xproc");
  auto a = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a);
  const std::string v = Val(7, 40000);
  DevBuf src(40000);
  src.Up(v);
  GpuNodeDedup::Seg s{reinterpret_cast<void*>(src.p), 40000};
  size_t got = 0;
  uint64_t token = 0;
  ASSERT_EQ(a->ClaimSg(K(7), &s, 1, 40000, &got, &token),
            GpuNodeDedup::Role::kFetch);
  a->PublishSg(K(7), token, &s, 1, v.size());

  // CUDA does not survive fork; exec this binary so the child initializes its
  // own driver state and pulls the payload across processes via cuIpc*.
  pid_t pid = ::fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    ::setenv("DFKV_GPU_DEDUP_TEST_SEG", g.name.c_str(), 1);
    ::execl("/proc/self/exe", "node_dedup_gpu_test",
            "--gtest_filter=GpuDedupChild.ReadsPeerPublish",
            static_cast<char*>(nullptr));
    _exit(97);  // exec failed
  }
  int st = 0;
  ASSERT_EQ(::waitpid(pid, &st, 0), pid);
  ASSERT_TRUE(WIFEXITED(st));
  EXPECT_EQ(WEXITSTATUS(st), 0) << "child (IPC reader) failed";
}

// ---- a waiter whose owner gives up ----------------------------------------

TEST(GpuNodeDedup, AbortEndsAWaitAtOnce) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("wait-abort");
  auto a = GpuNodeDedup::Open(LongWait(g.name));
  auto b = GpuNodeDedup::Open(LongWait(g.name));
  ASSERT_TRUE(a && b);
  DevBuf d(4096);
  GpuNodeDedup::Seg s{reinterpret_cast<void*>(d.p), 4096};
  size_t got = 0;
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->ClaimSg(K(40), &s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  ASSERT_EQ(b->ClaimSg(K(40), &s, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kWait);
  a->Abort(K(40), token);
  // Nothing will be published: the waiter falls back now, not after its
  // wait_ms (5 s).
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_FALSE(b->WaitSg(K(40), &s, 1, 4096, &got));
  EXPECT_LT(std::chrono::steady_clock::now() - t0, std::chrono::seconds(2));
  EXPECT_EQ(b->wait_timeouts(), 1u);
}

TEST(GpuNodeDedup, AbortDuringAWaitEndsIt) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("wait-abort-late");
  auto a = GpuNodeDedup::Open(LongWait(g.name));
  auto b = GpuNodeDedup::Open(LongWait(g.name));
  ASSERT_TRUE(a && b);
  DevBuf d(4096);
  GpuNodeDedup::Seg s{reinterpret_cast<void*>(d.p), 4096};
  size_t got = 0;
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->ClaimSg(K(41), &s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  ASSERT_EQ(b->ClaimSg(K(41), &s, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kWait);
  const auto t0 = std::chrono::steady_clock::now();
  std::thread abort([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    a->Abort(K(41), token);
  });
  GpuNodeDedup::WaitItem item{K(41), &s, 1, 4096, false, 0};
  b->WaitManySg(&item, 1);
  const auto waited = std::chrono::steady_clock::now() - t0;
  abort.join();
  EXPECT_FALSE(item.ok);
  EXPECT_GE(waited, std::chrono::milliseconds(50));  // a fetch in flight is waited for
  EXPECT_LT(waited, std::chrono::seconds(2));        // and its abort ends the wait
}

// A batch waits for many keys. The ones whose owner gave up are returned at
// once for the caller to fetch; the published ones are still delivered.
TEST(GpuNodeDedup, WaitManyGivesUpOnAbortedKeysAndDeliversTheRest) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("wait-many-abort");
  auto a = GpuNodeDedup::Open(LongWait(g.name));
  auto b = GpuNodeDedup::Open(LongWait(g.name));
  ASSERT_TRUE(a && b);
  constexpr size_t kN = 16, kLen = 4096;
  std::vector<std::unique_ptr<DevBuf>> src, dst;
  std::vector<GpuNodeDedup::Seg> ssegs(kN), dsegs(kN);
  std::vector<uint64_t> tokens(kN, 0);
  size_t got = 0;
  uint64_t ignored = 0;
  for (size_t i = 0; i < kN; ++i) {
    src.push_back(std::make_unique<DevBuf>(kLen));
    dst.push_back(std::make_unique<DevBuf>(kLen));
    src[i]->Up(Val(200 + i, kLen));
    ssegs[i] = {reinterpret_cast<void*>(src[i]->p), kLen};
    dsegs[i] = {reinterpret_cast<void*>(dst[i]->p), kLen};
    ASSERT_EQ(a->ClaimSg(K(200 + i), &ssegs[i], 1, kLen, &got, &tokens[i]),
              GpuNodeDedup::Role::kFetch);
    ASSERT_EQ(b->ClaimSg(K(200 + i), &dsegs[i], 1, kLen, &got, &ignored),
              GpuNodeDedup::Role::kWait);
  }
  // The owner gets half of its keys and gives the other half up.
  for (size_t i = 0; i < kN; ++i) {
    if (i % 2 == 0)
      a->PublishSg(K(200 + i), tokens[i], &ssegs[i], 1, kLen);
    else
      a->Abort(K(200 + i), tokens[i]);
  }
  std::vector<GpuNodeDedup::WaitItem> items(kN);
  for (size_t i = 0; i < kN; ++i)
    items[i] = GpuNodeDedup::WaitItem{K(200 + i), &dsegs[i], 1, kLen, false, 0};
  const auto t0 = std::chrono::steady_clock::now();
  b->WaitManySg(items.data(), items.size());
  EXPECT_LT(std::chrono::steady_clock::now() - t0, std::chrono::seconds(2))
      << "the batch waited out its deadline for keys nobody is fetching";
  for (size_t i = 0; i < kN; ++i) {
    EXPECT_EQ(items[i].ok, i % 2 == 0) << i;
    if (i % 2 == 0) {
      EXPECT_EQ(items[i].got, kLen);
      EXPECT_EQ(dst[i]->Down(kLen), Val(200 + i, kLen)) << i;
    }
  }
  EXPECT_EQ(b->wait_hits(), kN / 2);
  EXPECT_EQ(b->wait_timeouts(), kN / 2);
}

TEST(GpuNodeDedup, PublishedEntryTheWaiterCannotUseEndsTheWaitAtOnce) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("wait-unusable");
  auto a = GpuNodeDedup::Open(LongWait(g.name));
  auto b = GpuNodeDedup::Open(LongWait(g.name));
  ASSERT_TRUE(a && b);
  DevBuf src(4096), small(1024);
  src.Up(Val(42, 4096));
  GpuNodeDedup::Seg s{reinterpret_cast<void*>(src.p), 4096};
  GpuNodeDedup::Seg d{reinterpret_cast<void*>(small.p), 1024};
  size_t got = 0;
  uint64_t token = 0;
  ASSERT_EQ(a->ClaimSg(K(42), &s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  a->PublishSg(K(42), token, &s, 1, 4096);
  // 4096 bytes do not fit the waiter's 1024: it can never copy this entry.
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_FALSE(b->WaitSg(K(42), &d, 1, 1024, &got));
  EXPECT_LT(std::chrono::steady_clock::now() - t0, std::chrono::seconds(2));
}

// ---- through KVClient: a fetch that lands in device memory -----------------

namespace {

// A transport whose scatter-gather read writes the value into the caller's
// device segments, as a GPUDirect read does. With the simulated driver the
// write is held back until the next CUDA synchronization, which is the
// ordering hazard the load fence exists for; on real hardware it is a plain
// device copy. The client calls the transport on its read workers, threads
// that have no CUDA context: the copy binds one first (a NIC needs none).
class DeviceWritingTransport final : public dfkv::Transport {
 public:
  std::map<std::string, std::string> values;  // by block key file name
  int reads = 0;

  bool pipelined() const override { return true; }
  dfkv::Status Cache(const std::string&, const BlockKey&, const void*,
                     size_t) override {
    return dfkv::Status::kInvalid;
  }
  dfkv::Status Range(const std::string&, const BlockKey&, uint64_t, uint64_t,
                     std::string*, uint64_t*) override {
    return dfkv::Status::kInvalid;
  }
  dfkv::Status Lookup(const std::string&, const BlockKey&, uint64_t*) override {
    return dfkv::Status::kInvalid;
  }
  dfkv::Status Exist(const std::string&, const BlockKey&, bool*) override {
    return dfkv::Status::kInvalid;
  }
  std::vector<dfkv::Status> RangeIntoMulti(
      const std::string&, const std::vector<BlockKey>& keys,
      const std::vector<dfkv::RangeDstMulti>& dsts,
      std::vector<size_t>* out_lens, std::string* out_dev) override {
    using Write = void(void*, const void*, size_t);
    Write* held_back = SimSymbol<Write>("sim_cuda_gpudirect_write");
    std::lock_guard<std::mutex> lock(mu_);
    if (out_dev) out_dev->clear();
    if (out_lens) out_lens->assign(keys.size(), 0);
    std::vector<dfkv::Status> result(keys.size(), dfkv::Status::kNotFound);
    for (size_t i = 0; i < keys.size(); ++i) {
      ++reads;
      const auto found = values.find(keys[i].Filename());
      if (found == values.end()) continue;
      const std::string& v = found->second;
      size_t done = 0;
      for (const auto& seg : dsts[i].payloads) {
        const size_t n = std::min(seg.second, v.size() - done);
        if (n == 0) break;
        if (held_back) {
          held_back(seg.first, v.data() + done, n);
        } else if (!EnsureCudaCtx() ||
                   CudaLib::Get()->Memcpy(
                       reinterpret_cast<CUdeviceptr>(seg.first),
                       reinterpret_cast<CUdeviceptr>(v.data() + done),
                       n) != kCudaSuccess) {
          ADD_FAILURE() << "the transport could not write the destination";
          result[i] = dfkv::Status::kIOError;
          break;
        }
        done += n;
      }
      if (result[i] == dfkv::Status::kIOError) continue;
      if (out_lens) (*out_lens)[i] = v.size();
      result[i] = dfkv::Status::kOk;
    }
    return result;
  }

 private:
  std::mutex mu_;
};

// Two client handles of one process, on the GPU rendezvous of a private
// namespace: the leader fetches, the follower is served by it.
struct TwoClients {
  std::string key_namespace;
  std::string segment;
  DeviceWritingTransport transport;
  std::unique_ptr<dfkv::KVClient> leader, follower;

  explicit TwoClients(const std::string& tag)
      : key_namespace("gpu-dedup-test/" + tag + "/" +
                      std::to_string(::getpid())) {
    ::setenv("DFKV_CLIENT_NODE_DEDUP", "1", 1);
    ::setenv("DFKV_CLIENT_NODE_DEDUP_GPU", "1", 1);
    ::setenv("DFKV_NODE_DEDUP_GPU_ARENA_MB", "16", 1);
    segment = GpuNodeDedup::EnvSegmentName(
        dfkv::ToBlockKey(key_namespace, "").digest_hi);
    ::shm_unlink(segment.c_str());
    using Members = std::vector<std::pair<std::string, std::string>>;
    leader = std::make_unique<dfkv::KVClient>(Members{{"n", "127.0.0.1:1"}},
                                              key_namespace, &transport);
    follower = std::make_unique<dfkv::KVClient>(Members{{"n", "127.0.0.1:1"}},
                                                key_namespace, &transport);
  }
  ~TwoClients() {
    leader.reset();
    follower.reset();
    ::shm_unlink(segment.c_str());
    ::unsetenv("DFKV_CLIENT_NODE_DEDUP");
    ::unsetenv("DFKV_CLIENT_NODE_DEDUP_GPU");
    ::unsetenv("DFKV_NODE_DEDUP_GPU_ARENA_MB");
    ::unsetenv("DFKV_GPU_LOAD_FENCE");
  }
  void Store(const std::string& key, const std::string& value) {
    transport.values[dfkv::ToBlockKey(key_namespace, key).Filename()] = value;
  }
  // One key into two device segments of 20000 bytes each.
  bool Get(dfkv::KVClient* client, const std::string& key, DevBuf* d0,
           DevBuf* d1, size_t* len) {
    std::vector<dfkv::KvGetItemSg> items{
        {key,
         {reinterpret_cast<void*>(d0->p), reinterpret_cast<void*>(d1->p)},
         {20000, 20000}}};
    std::vector<size_t> lens;
    const std::vector<bool> got = client->BatchGetAutoSg(items, &lens);
    *len = lens.empty() ? 0 : lens[0];
    return !got.empty() && got[0];
  }
};

uint64_t ClientCounter(const dfkv::KVClient& client, const std::string& name) {
  const std::string text = client.MetricsSnapshot();
  const size_t at = text.find("\n" + name + " ");
  EXPECT_NE(at, std::string::npos) << name;
  if (at == std::string::npos) return 0;
  return std::stoull(text.substr(at + name.size() + 2));
}

// What the connector does after a load before it uses the destinations.
void ConnectorFence() {
  ASSERT_NE(CudaLib::Get()->CtxSynchronize, nullptr);
  ASSERT_EQ(CudaLib::Get()->CtxSynchronize(), kCudaSuccess);
}

}  // namespace

TEST(GpuNodeDedup, LoadFenceSynchronizesTheDevice) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("fence");
  auto a = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a);
  int* synchronized = SimSymbol<int>("sim_cuda_ctx_synchronize_calls");
  const int before = synchronized ? *synchronized : 0;
  ::unsetenv("DFKV_GPU_LOAD_FENCE");
  EXPECT_TRUE(a->FenceLoads());
  EXPECT_EQ(a->load_fences(), 1u);
  if (synchronized) EXPECT_EQ(*synchronized, before + 1);
  // The switch that turns the connectors' fence off turns this one off too.
  ::setenv("DFKV_GPU_LOAD_FENCE", "0", 1);
  EXPECT_TRUE(a->FenceLoads());
  EXPECT_EQ(a->load_fences(), 1u);
  if (synchronized) EXPECT_EQ(*synchronized, before + 1);
  ::unsetenv("DFKV_GPU_LOAD_FENCE");
}

// The leader's fetch lands in its device destinations and is then published
// to the rendezvous with a device copy. That copy must see the bytes: the
// follower gets what the leader fetched, not what the destinations held
// before the write arrived. Only the simulated driver holds a write back, so
// only there does this test exercise the fence; on real hardware the
// transport's plain copy does not reproduce a late GPUDirect write, and the
// test checks the data path end to end.
TEST(GpuDedupClient, FollowerGetsTheBytesTheLeaderFetched) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  TwoClients c("fence");
  const std::string v = Val(21, 40000);
  c.Store("k", v);
  DevBuf l0(20000), l1(20000), f0(20000), f1(20000);
  const std::string before(20000, '\0');
  l0.Up(before); l1.Up(before); f0.Up(before); f1.Up(before);

  size_t len = 0;
  ASSERT_TRUE(c.Get(c.leader.get(), "k", &l0, &l1, &len));
  EXPECT_EQ(len, v.size());
  ASSERT_TRUE(c.Get(c.follower.get(), "k", &f0, &f1, &len));
  EXPECT_EQ(len, v.size());
  EXPECT_EQ(c.transport.reads, 1) << "the follower was served by the leader";
  EXPECT_EQ(f0.Down(20000), v.substr(0, 20000))
      << "the follower got bytes from before the leader's fetch arrived";
  EXPECT_EQ(f1.Down(20000), v.substr(20000));
  EXPECT_EQ(ClientCounter(*c.leader, "dfkv_client_gpu_dedup_load_fences_total"),
            1u);
  ConnectorFence();
  EXPECT_EQ(l0.Down(20000), v.substr(0, 20000));
  EXPECT_EQ(l1.Down(20000), v.substr(20000));
}

// When the device cannot be synchronized, nothing is published: the follower
// fetches for itself instead of being handed bytes of unknown state.
TEST(GpuDedupClient, NothingIsPublishedWhenTheDeviceCannotBeSynchronized) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  int* fail = SimSymbol<int>("sim_cuda_fail_ctx_synchronize");
  if (!fail) GTEST_SKIP() << "needs the simulated driver to fail a synchronize";
  TwoClients c("fence-fails");
  const std::string v = Val(22, 40000);
  c.Store("k", v);
  DevBuf l0(20000), l1(20000), f0(20000), f1(20000);

  *fail = 1;
  size_t len = 0;
  EXPECT_TRUE(c.Get(c.leader.get(), "k", &l0, &l1, &len));  // its own result
  EXPECT_TRUE(c.Get(c.follower.get(), "k", &f0, &f1, &len));
  *fail = 0;
  EXPECT_EQ(c.transport.reads, 2) << "the follower fetched for itself";
  EXPECT_EQ(ClientCounter(*c.leader,
                          "dfkv_client_gpu_dedup_load_fence_failures_total"),
            1u);
  ConnectorFence();
  EXPECT_EQ(f0.Down(20000), v.substr(0, 20000));
  EXPECT_EQ(f1.Down(20000), v.substr(20000));
}

#ifdef DFKV_GPU_TEST_SIMULATED
int main(int argc, char** argv) {
  // The simulated driver must be the first libcuda.so.1 the loader finds,
  // whatever environment the test was started from: re-execute once with its
  // directory prepended. The cross-process test's child inherits both.
  constexpr char kReexecuted[] = "DFKV_GPU_SIM_TEST_REEXECUTED";
  if (::getenv(kReexecuted) == nullptr) {
    std::string path = DFKV_SIM_CUDA_DIR;
    const char* inherited = ::getenv("LD_LIBRARY_PATH");
    if (inherited != nullptr && *inherited != '\0') {
      path += ':';
      path += inherited;
    }
    ::setenv("LD_LIBRARY_PATH", path.c_str(), 1);
    ::setenv(kReexecuted, "1", 1);
    ::execv("/proc/self/exe", argv);
    std::perror("node_dedup_gpu_sim_test: re-exec");
    return 127;
  }
  if (::access(DFKV_SIM_CUDA_DIR "/libcuda.so.1", R_OK) != 0) {
    std::fprintf(stderr,
                 "node_dedup_gpu_sim_test: " DFKV_SIM_CUDA_DIR
                 "/libcuda.so.1 is missing; run from the build tree it was "
                 "built in\n");
    return 2;
  }
  // Load it now, by name as the code under test does, and refuse to run on
  // anything but the simulation.
  if (::dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL) == nullptr ||
      !Simulated()) {
    std::fprintf(stderr, "node_dedup_gpu_sim_test: the simulated driver is "
                         "not the libcuda.so.1 the loader finds\n");
    return 3;
  }
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif  // DFKV_GPU_TEST_SIMULATED
