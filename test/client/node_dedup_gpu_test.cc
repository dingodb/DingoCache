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
#include <future>
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

namespace dfkv {
// A publisher runs in two stages, and so does an invalidation that takes a
// slot from it; these let a test stop between them.
struct GpuNodeDedupTestAccess {
  struct Hold {
    GpuNodeDedup::PublishHold hold;
    bool taken = false;
  };
  // A publisher that has taken its slot and not yet copied. Never calling
  // PublishFinish is a publisher killed there.
  static Hold PublishBegin(GpuNodeDedup& d, const BlockKey& key, uint64_t token,
                           size_t len) {
    Hold h;
    h.taken = d.PublishBegin(key, token, len, &h.hold);
    return h;
  }
  static void PublishFinish(GpuNodeDedup& d, const Hold& h,
                            const GpuNodeDedup::Seg* segs, size_t nsegs) {
    d.PublishFinish(h.hold, segs, nsegs);
  }
  // A forcer that has taken the slot from h's publisher and not yet
  // released it. Never calling ForceOutFinish is a forcer killed there.
  static bool ForceOutBegin(GpuNodeDedup& d, const Hold& h) {
    return d.ForceOutBegin(h.hold.slot, h.hold.gen);
  }
  static bool ForceOutFinish(GpuNodeDedup& d, const Hold& h) {
    return d.ForceOutFinish(h.hold.slot, h.hold.gen + 1);
  }
  // ... seen in the middle of its commit: owner fields stored, len not yet.
  static void Interleave(GpuNodeDedup& d, const Hold& h, int step) {
    d.InterleaveForTest(h.hold, step);
  }
  static bool Recover(GpuNodeDedup& d, const Hold& h) {
    return d.RecoverForTest(h.hold);
  }
  static bool GenEven(const GpuNodeDedup& d, const Hold& h) {
    return d.GenEvenForTest(h.hold);
  }
  static bool ShiftStamp(GpuNodeDedup& d, const BlockKey& key, int64_t ms) {
    return d.ShiftStampForTest(key, ms);
  }
  static void LookLikeAnOlderRelease(GpuNodeDedup& d, const Hold& h) {
    d.SetFenceableForTest(h.hold, false);
  }
  static void ForgeRegistry(GpuNodeDedup& d, int how) {
    d.ForgeRegistryForTest(how);
  }
  // A reader looks a slot up and then copies out of it: the two stages.
  static auto Find(GpuNodeDedup& d, const BlockKey& key) { return d.Find(key); }
  template <typename SlotPtr>
  static bool CopyOut(GpuNodeDedup& d, SlotPtr slot, const BlockKey& key,
                      const GpuNodeDedup::Seg& seg) {
    d.EnsureThreadCtx();
    size_t got = 0;
    return d.CopyOutSg(slot, key, &seg, 1, seg.cap, &got);
  }
};
}  // namespace dfkv
using dfkv::GpuNodeDedupTestAccess;

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

// ---- a publisher that stops inside PublishSg -------------------------------

namespace {

// Runs Invalidate on its own thread (with a CUDA context, like any caller).
struct InvalidateInBackground {
  std::future<void> done;
  InvalidateInBackground(GpuNodeDedup* d, BlockKey key)
      : done(std::async(std::launch::async, [d, key] {
          EnsureCudaCtx();
          d->Invalidate(key);
        })) {}
  bool ReturnedWithin(std::chrono::milliseconds bound) {
    return done.wait_for(bound) == std::future_status::ready;
  }
};

// One key's worth of device buffers: a source holding `value` and a
// destination of the same size.
struct KeyBufs {
  std::string value;
  DevBuf src, dst;
  GpuNodeDedup::Seg s, d;
  explicit KeyBufs(uint64_t id, size_t n = 4096)
      : value(Val(id, n)), src(n), dst(n),
        s{reinterpret_cast<void*>(src.p), n},
        d{reinterpret_cast<void*>(dst.p), n} {
    src.Up(value);
  }
};

}  // namespace

TEST(GpuNodeDedup, InvalidateDoesNotWaitForAKilledPublisher) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("pub-dead");
  auto a = GpuNodeDedup::Open(Opts(g.name));
  auto b = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  KeyBufs k(60);
  size_t got = 0;
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->ClaimSg(K(60), &k.s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  // The publisher takes its slot and is killed before it commits.
  const auto held = GpuNodeDedupTestAccess::PublishBegin(*a, K(60), token, 4096);
  ASSERT_TRUE(held.taken);

  InvalidateInBackground inv(b.get(), K(60));
  const bool returned = inv.ReturnedWithin(std::chrono::seconds(2));
  EXPECT_TRUE(returned) << "a write or remove of this key would never return";
  if (!returned)  // let the thread go so the test can end
    GpuNodeDedupTestAccess::PublishFinish(*a, held, &k.s, 1);
  inv.done.wait();
  // The slot is usable again.
  EXPECT_EQ(b->ClaimSg(K(60), &k.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kFetch);
}

TEST(GpuNodeDedup, InvalidateFencesAPublisherThatIsStillCopying) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("pub-slow");
  auto a = GpuNodeDedup::Open(Opts(g.name));
  auto b = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  KeyBufs old_value(61), new_value(62);
  size_t got = 0;
  uint64_t token_a = 0, token_b = 0, ignored = 0;
  ASSERT_EQ(a->ClaimSg(K(61), &old_value.s, 1, 4096, &got, &token_a),
            GpuNodeDedup::Role::kFetch);
  const auto held =
      GpuNodeDedupTestAccess::PublishBegin(*a, K(61), token_a, 4096);
  ASSERT_TRUE(held.taken);

  // The key is overwritten while the publisher copies, and fetched again.
  InvalidateInBackground inv(b.get(), K(61));
  const bool returned = inv.ReturnedWithin(std::chrono::seconds(2));
  EXPECT_TRUE(returned);
  if (returned) {
    ASSERT_EQ(b->ClaimSg(K(61), &new_value.s, 1, 4096, &got, &token_b),
              GpuNodeDedup::Role::kFetch);
    b->PublishSg(K(61), token_b, &new_value.s, 1, 4096);
  }
  // The publisher was alive all along and now commits: nothing may come of it.
  GpuNodeDedupTestAccess::PublishFinish(*a, held, &old_value.s, 1);
  inv.done.wait();
  if (returned) {
    ASSERT_EQ(a->ClaimSg(K(61), &old_value.d, 1, 4096, &got, &ignored),
              GpuNodeDedup::Role::kHit);
    EXPECT_EQ(old_value.dst.Down(4096), new_value.value);
  }
}

// A publisher that was fenced and then fails its copy gives up a slot that is
// no longer its own: it must leave the claim that took its place alone.
TEST(GpuNodeDedup, FencedPublisherThatFailsLeavesTheNextClaimAlone) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("pub-abandon");
  auto a = GpuNodeDedup::Open(Opts(g.name));
  auto b = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  KeyBufs old_value(63), new_value(64);
  size_t got = 0;
  uint64_t token_a = 0, token_b = 0, ignored = 0;
  ASSERT_EQ(a->ClaimSg(K(63), &old_value.s, 1, 4096, &got, &token_a),
            GpuNodeDedup::Role::kFetch);
  const auto held =
      GpuNodeDedupTestAccess::PublishBegin(*a, K(63), token_a, 4096);
  ASSERT_TRUE(held.taken);
  // The key is overwritten: the invalidation takes the slot from the
  // publisher, and the next fetch claims it.
  InvalidateInBackground inv(b.get(), K(63));
  if (!inv.ReturnedWithin(std::chrono::seconds(2))) {
    ADD_FAILURE() << "the invalidation waits for the publisher";
    GpuNodeDedupTestAccess::PublishFinish(*a, held, &old_value.s, 1);
    inv.done.wait();
    return;
  }
  ASSERT_EQ(b->ClaimSg(K(63), &new_value.s, 1, 4096, &got, &token_b),
            GpuNodeDedup::Role::kFetch);
  // The old publisher's gather comes up short (its segments hold less than it
  // announced), which makes it give the slot up.
  GpuNodeDedup::Seg short_segment{old_value.s.ptr, 1024};
  GpuNodeDedupTestAccess::PublishFinish(*a, held, &short_segment, 1);
  // The new claim is still there and publishes.
  b->PublishSg(K(63), token_b, &new_value.s, 1, 4096);
  ASSERT_EQ(a->ClaimSg(K(63), &old_value.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kHit)
      << "the fenced publisher dropped the claim that replaced it";
  EXPECT_EQ(old_value.dst.Down(4096), new_value.value);
}

TEST(GpuNodeDedup, ClaimTakesOverFromAKilledPublisher) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("pub-dead-claim");
  auto a = GpuNodeDedup::Open(Opts(g.name));
  auto b = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  KeyBufs k(65);
  size_t got = 0;
  uint64_t token = 0, taken_over = 0;
  ASSERT_EQ(a->ClaimSg(K(65), &k.s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  const auto held = GpuNodeDedupTestAccess::PublishBegin(*a, K(65), token, 4096);
  ASSERT_TRUE(held.taken);
  std::this_thread::sleep_for(std::chrono::milliseconds(250));  // > takeover_ms(200)
  EXPECT_EQ(b->ClaimSg(K(65), &k.d, 1, 4096, &got, &taken_over),
            GpuNodeDedup::Role::kFetch)
      << "every later lookup of this key would wait out its deadline";
}

// The takeover age of a PUBLISHING slot is time spent publishing. A fetch
// that itself took longer than takeover_ms is not taken away the moment its
// owner starts to copy.
TEST(GpuNodeDedup, SlowFetchIsNotForcedOutDuringItsCopy) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("pub-slow-fetch");
  auto a = GpuNodeDedup::Open(LongWait(g.name));  // nobody steals the fetch
  auto o = Opts(g.name);
  o.takeover_ms = 500;
  auto b = GpuNodeDedup::Open(o);
  ASSERT_TRUE(a && b);
  KeyBufs k(66);
  size_t got = 0;
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->ClaimSg(K(66), &k.s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  std::this_thread::sleep_for(std::chrono::milliseconds(600));  // the fetch
  const auto held = GpuNodeDedupTestAccess::PublishBegin(*a, K(66), token, 4096);
  ASSERT_TRUE(held.taken);
  EXPECT_EQ(b->ClaimSg(K(66), &k.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kWait)
      << "a copy that had only just begun was taken away";
  GpuNodeDedupTestAccess::PublishFinish(*a, held, &k.s, 1);
  ASSERT_EQ(b->ClaimSg(K(66), &k.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kHit);
  EXPECT_EQ(k.dst.Down(4096), k.value);
}

// A publisher of an older release commits with unconditional stores: taking
// its slot away would let those stores land on whoever owns the slot next.
// It is waited for, as before, also when another core already sees some of
// its commit stores (here the owner fields) and not yet the others (len).
TEST(GpuNodeDedup, PublisherOfAnOlderReleaseIsNeverForcedOut) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("pub-old");
  auto a = GpuNodeDedup::Open(Opts(g.name));
  auto b = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  KeyBufs k(67);
  size_t got = 0;
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->ClaimSg(K(67), &k.s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  const auto held = GpuNodeDedupTestAccess::PublishBegin(*a, K(67), token, 4096);
  ASSERT_TRUE(held.taken);
  GpuNodeDedupTestAccess::LookLikeAnOlderRelease(*a, held);

  std::this_thread::sleep_for(std::chrono::milliseconds(250));  // > takeover_ms(200)
  EXPECT_EQ(b->ClaimSg(K(67), &k.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kWait);
  InvalidateInBackground inv(b.get(), K(67));
  EXPECT_FALSE(inv.ReturnedWithin(std::chrono::milliseconds(150)));
  // It commits; the invalidation then removes what it published.
  GpuNodeDedupTestAccess::PublishFinish(*a, held, &k.s, 1);
  inv.done.wait();
  EXPECT_NE(b->ClaimSg(K(67), &k.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kHit);
}

// Two invalidations of one key meet at a publisher's slot. Only one may take
// it: the other must not act on the slot until the first is through.
TEST(GpuNodeDedup, TwoInvalidationsDoNotBothTakeAPublishersSlot) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("pub-two-forcers");
  // The first forcer is not stranded here: no recovery within the test.
  GpuNodeDedup::Options o = Opts(g.name);
  o.takeover_ms = 60000;
  auto a = GpuNodeDedup::Open(o);
  auto b = GpuNodeDedup::Open(o);
  auto c = GpuNodeDedup::Open(o);
  ASSERT_TRUE(a && b && c);
  KeyBufs k(68), fresh(69);
  size_t got = 0;
  uint64_t token_a = 0, token_c = 0, ignored = 0;
  ASSERT_EQ(a->ClaimSg(K(68), &k.s, 1, 4096, &got, &token_a),
            GpuNodeDedup::Role::kFetch);
  const auto held =
      GpuNodeDedupTestAccess::PublishBegin(*a, K(68), token_a, 4096);
  ASSERT_TRUE(held.taken);

  // The first invalidation has won the slot and not yet released it.
  ASSERT_TRUE(GpuNodeDedupTestAccess::ForceOutBegin(*b, held));
  InvalidateInBackground second(c.get(), K(68));
  const bool second_acted = second.ReturnedWithin(std::chrono::milliseconds(150));
  EXPECT_FALSE(second_acted) << "a second invalidation took the slot as well";
  if (second_acted) {
    ASSERT_EQ(c->ClaimSg(K(68), &fresh.s, 1, 4096, &got, &token_c),
              GpuNodeDedup::Role::kFetch);
  }
  EXPECT_TRUE(GpuNodeDedupTestAccess::ForceOutFinish(*b, held));
  second.done.wait();
  if (!second_acted) {
    ASSERT_EQ(c->ClaimSg(K(68), &fresh.s, 1, 4096, &got, &token_c),
              GpuNodeDedup::Role::kFetch);
  }
  c->PublishSg(K(68), token_c, &fresh.s, 1, 4096);
  ASSERT_EQ(a->ClaimSg(K(68), &k.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kHit)
      << "the claim made after the invalidations was wiped";
  EXPECT_EQ(k.dst.Down(4096), fresh.value);
}

// A forcer killed after it took a publisher's slot and before it released
// it leaves the slot PUBLISHING with an odd generation. Once that is older
// than the takeover age, an invalidation finishes the release for it. If the
// forcer was only slow, it comes back to a slot that is someone else's and
// leaves it alone.
TEST(GpuNodeDedup, InvalidateFinishesForAForcerThatWasKilled) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("forcer-dead-invalidate");
  auto a = GpuNodeDedup::Open(Opts(g.name));  // takeover_ms 200
  auto b = GpuNodeDedup::Open(Opts(g.name));
  auto c = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b && c);
  KeyBufs k(70), fresh(71);
  size_t got = 0;
  uint64_t token_a = 0, token_c = 0, ignored = 0;
  ASSERT_EQ(a->ClaimSg(K(70), &k.s, 1, 4096, &got, &token_a),
            GpuNodeDedup::Role::kFetch);
  const auto held =
      GpuNodeDedupTestAccess::PublishBegin(*a, K(70), token_a, 4096);
  ASSERT_TRUE(held.taken);
  ASSERT_TRUE(GpuNodeDedupTestAccess::ForceOutBegin(*b, held));

  InvalidateInBackground inv(c.get(), K(70));
  const bool returned = inv.ReturnedWithin(std::chrono::seconds(2));
  EXPECT_TRUE(returned) << "a write or remove of this key would never return";
  if (!returned) GpuNodeDedupTestAccess::ForceOutFinish(*b, held);
  inv.done.wait();
  ASSERT_EQ(c->ClaimSg(K(70), &fresh.s, 1, 4096, &got, &token_c),
            GpuNodeDedup::Role::kFetch);
  // The forcer was only slow: its finish now must not touch the new claim.
  EXPECT_FALSE(GpuNodeDedupTestAccess::ForceOutFinish(*b, held));
  c->PublishSg(K(70), token_c, &fresh.s, 1, 4096);
  ASSERT_EQ(a->ClaimSg(K(70), &k.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kHit)
      << "the late forcer wiped the claim made after the recovery";
  EXPECT_EQ(k.dst.Down(4096), fresh.value);
}

TEST(GpuNodeDedup, ClaimFinishesForAForcerThatWasKilled) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("forcer-dead-claim");
  auto a = GpuNodeDedup::Open(Opts(g.name));  // takeover_ms 200
  auto b = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  KeyBufs k(72);
  size_t got = 0;
  uint64_t token = 0, taken_over = 0;
  ASSERT_EQ(a->ClaimSg(K(72), &k.s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  const auto held = GpuNodeDedupTestAccess::PublishBegin(*a, K(72), token, 4096);
  ASSERT_TRUE(held.taken);
  ASSERT_TRUE(GpuNodeDedupTestAccess::ForceOutBegin(*b, held));
  // Younger than the takeover age: the forcer may still be at work.
  EXPECT_EQ(b->ClaimSg(K(72), &k.d, 1, 4096, &got, &taken_over),
            GpuNodeDedup::Role::kWait);
  std::this_thread::sleep_for(std::chrono::milliseconds(250));  // > takeover_ms
  EXPECT_EQ(b->ClaimSg(K(72), &k.d, 1, 4096, &got, &taken_over),
            GpuNodeDedup::Role::kFetch)
      << "every later lookup of this key would wait out its deadline";
}

// A stranded force-out is finished by one party at a time. While a finisher
// is between its swap and the rest of its stores, a second party that looks
// at the slot must not finish the same release again.
// An invalidation that has to wait for a holder it cannot take the slot from
// (here a publisher of an older release, seen in the middle of its commit)
// still waits, but says so once it has waited longer than the takeover age.
TEST(GpuNodeDedup, AnInvalidationThatWaitsTooLongIsReported) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("invalidate-stuck");
  auto a = GpuNodeDedup::Open(Opts(g.name));  // takeover_ms 200
  auto b = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  KeyBufs k(76);
  size_t got = 0;
  uint64_t token = 0;
  ASSERT_EQ(a->ClaimSg(K(76), &k.s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  const auto held = GpuNodeDedupTestAccess::PublishBegin(*a, K(76), token, 4096);
  ASSERT_TRUE(held.taken);
  GpuNodeDedupTestAccess::LookLikeAnOlderRelease(*a, held);

  InvalidateInBackground inv(b.get(), K(76));
  EXPECT_FALSE(inv.ReturnedWithin(std::chrono::milliseconds(500)))
      << "the invalidation gave up on a holder that may still commit";
  EXPECT_EQ(b->invalidate_stuck(), 1u) << "the long wait was not reported";
  GpuNodeDedupTestAccess::PublishFinish(*a, held, &k.s, 1);
  inv.done.wait();
  EXPECT_EQ(b->invalidate_stuck(), 1u) << "reported more than once";
  EXPECT_NE(b->ClaimSg(K(76), &k.d, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kHit);
}

// A release of a slot taken from a publisher is past its finishing swap: the
// slot can only end EMPTY. Once that has lasted longer than the takeover age
// the finisher is taken to be dead and an invalidation stops waiting, so one
// killed finisher does not hang every write of the key.
TEST(GpuNodeDedup, InvalidateDoesNotWaitForAReleasePastItsFinishingSwap) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("invalidate-finishing");
  auto a = GpuNodeDedup::Open(Opts(g.name));  // takeover_ms 200
  auto b = GpuNodeDedup::Open(Opts(g.name));
  auto c = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b && c);
  KeyBufs k(77);
  size_t got = 0;
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->ClaimSg(K(77), &k.s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  const auto held = GpuNodeDedupTestAccess::PublishBegin(*a, K(77), token, 4096);
  ASSERT_TRUE(held.taken);
  ASSERT_TRUE(GpuNodeDedupTestAccess::ForceOutBegin(*b, held));
  GpuNodeDedupTestAccess::Interleave(*b, held, 4);  // finisher's swap, then killed

  InvalidateInBackground inv(c.get(), K(77));
  EXPECT_FALSE(inv.ReturnedWithin(std::chrono::milliseconds(100)))
      << "gave up on the finisher before the takeover age";
  const bool returned = inv.ReturnedWithin(std::chrono::seconds(2));
  EXPECT_TRUE(returned) << "a write or remove of this key would never return";
  if (!returned) GpuNodeDedupTestAccess::Interleave(*b, held, 5);
  inv.done.wait();
  EXPECT_EQ(c->invalidate_stuck(), 1u);
  EXPECT_NE(c->ClaimSg(K(77), &k.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kHit);
  // The finisher was only slow: it ends the release, and the slot works.
  if (returned) GpuNodeDedupTestAccess::Interleave(*b, held, 5);
  EXPECT_TRUE(GpuNodeDedupTestAccess::GenEven(*a, held));
  uint64_t t2 = 0;
  ASSERT_EQ(a->ClaimSg(K(77), &k.s, 1, 4096, &got, &t2),
            GpuNodeDedup::Role::kFetch);
  a->PublishSg(K(77), t2, &k.s, 1, 4096);
  ASSERT_EQ(c->ClaimSg(K(77), &k.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kHit);
  EXPECT_EQ(k.dst.Down(4096), k.value);
}

// A finisher that is only passing through that state is waited for, and
// the wait is not reported as stuck.
TEST(GpuNodeDedup, InvalidateWaitsForAFinisherAtWorkWithoutReportingIt) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("invalidate-finisher-live");
  auto a = GpuNodeDedup::Open(Opts(g.name));  // takeover_ms 200
  auto b = GpuNodeDedup::Open(Opts(g.name));
  auto c = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b && c);
  KeyBufs k(78);
  size_t got = 0;
  uint64_t token = 0;
  ASSERT_EQ(a->ClaimSg(K(78), &k.s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  const auto held = GpuNodeDedupTestAccess::PublishBegin(*a, K(78), token, 4096);
  ASSERT_TRUE(held.taken);
  ASSERT_TRUE(GpuNodeDedupTestAccess::ForceOutBegin(*b, held));
  GpuNodeDedupTestAccess::Interleave(*b, held, 4);  // finisher's swap

  InvalidateInBackground inv(c.get(), K(78));
  EXPECT_FALSE(inv.ReturnedWithin(std::chrono::milliseconds(50)));
  GpuNodeDedupTestAccess::Interleave(*b, held, 5);  // and its last stores
  EXPECT_TRUE(inv.ReturnedWithin(std::chrono::seconds(2)));
  inv.done.wait();
  EXPECT_EQ(c->invalidate_stuck(), 0u)
      << "a release that was going on normally was reported as stuck";
  EXPECT_TRUE(GpuNodeDedupTestAccess::GenEven(*a, held));
}

// One invalidation reports one slot once. Here the slot is first held by a
// publisher it cannot take the slot from, past the takeover age (reported),
// and then turns into a release past its finishing swap whose finisher dies:
// the invalidation stops waiting without a second report.
TEST(GpuNodeDedup, AnInvalidationReportsASlotOnce) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("invalidate-report-once");
  auto a = GpuNodeDedup::Open(Opts(g.name));  // takeover_ms 200
  auto c = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && c);
  KeyBufs k(79);
  size_t got = 0;
  uint64_t token = 0;
  ASSERT_EQ(a->ClaimSg(K(79), &k.s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  const auto held = GpuNodeDedupTestAccess::PublishBegin(*a, K(79), token, 4096);
  ASSERT_TRUE(held.taken);
  GpuNodeDedupTestAccess::LookLikeAnOlderRelease(*a, held);

  InvalidateInBackground inv(c.get(), K(79));
  EXPECT_FALSE(inv.ReturnedWithin(std::chrono::milliseconds(400)));
  EXPECT_EQ(c->invalidate_stuck(), 1u);
  GpuNodeDedupTestAccess::Interleave(*a, held, 6);  // a forcer takes it
  GpuNodeDedupTestAccess::Interleave(*a, held, 4);  // a finisher's swap, then dies
  const bool returned = inv.ReturnedWithin(std::chrono::seconds(2));
  EXPECT_TRUE(returned);
  if (!returned) GpuNodeDedupTestAccess::Interleave(*a, held, 5);
  inv.done.wait();
  EXPECT_EQ(c->invalidate_stuck(), 1u) << "the same slot was reported twice";
}

TEST(GpuNodeDedup, AStrandedForceOutIsFinishedOnce) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("forcer-dead-twice");
  auto a = GpuNodeDedup::Open(Opts(g.name));  // takeover_ms 200
  auto b = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  KeyBufs k(74);
  size_t got = 0;
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->ClaimSg(K(74), &k.s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  const auto held = GpuNodeDedupTestAccess::PublishBegin(*a, K(74), token, 4096);
  ASSERT_TRUE(held.taken);
  ASSERT_TRUE(GpuNodeDedupTestAccess::ForceOutBegin(*b, held));  // then killed
  std::this_thread::sleep_for(std::chrono::milliseconds(250));    // > takeover_ms

  GpuNodeDedupTestAccess::Interleave(*a, held, 4);  // a finisher's swap
  EXPECT_FALSE(GpuNodeDedupTestAccess::Recover(*b, held))
      << "a second party finished the same force-out";
  GpuNodeDedupTestAccess::Interleave(*a, held, 5);  // the finisher's rest
  EXPECT_TRUE(GpuNodeDedupTestAccess::GenEven(*a, held))
      << "the slot's generation is odd at rest";

  uint64_t t2 = 0;
  ASSERT_EQ(a->ClaimSg(K(74), &k.s, 1, 4096, &got, &t2),
            GpuNodeDedup::Role::kFetch);
  a->PublishSg(K(74), t2, &k.s, 1, 4096);
  ASSERT_EQ(b->ClaimSg(K(74), &k.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kHit);
  EXPECT_EQ(k.dst.Down(4096), k.value);
}

// A slot that went through a force-out is claimed again. While the claim's
// Reserve is between its state swap and the rest of its stores, an
// invalidation of the key the slot still names holds it PUBLISHING with an
// odd generation. That is not a stranded force-out, even once the force-out
// is older than the takeover age: a recovery there would step the
// generation under the other two and leave it odd at rest, after which no
// entry in the slot can be read.
TEST(GpuNodeDedup, ReusedSlotIsNotTakenForAStrandedForceOut) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("forcer-reuse");
  auto a = GpuNodeDedup::Open(Opts(g.name));  // takeover_ms 200
  auto b = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  KeyBufs k(73);
  size_t got = 0;
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->ClaimSg(K(73), &k.s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  const auto held = GpuNodeDedupTestAccess::PublishBegin(*a, K(73), token, 4096);
  ASSERT_TRUE(held.taken);
  ASSERT_TRUE(GpuNodeDedupTestAccess::ForceOutBegin(*b, held));
  ASSERT_TRUE(GpuNodeDedupTestAccess::ForceOutFinish(*b, held));
  std::this_thread::sleep_for(std::chrono::milliseconds(250));  // > takeover_ms

  GpuNodeDedupTestAccess::Interleave(*a, held, 0);  // Reserve swaps the state
  GpuNodeDedupTestAccess::Interleave(*a, held, 1);  // Invalidate holds it
  EXPECT_EQ(b->ClaimSg(K(73), &k.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kWait)
      << "an invalidation in progress was taken for a stranded force-out";
  GpuNodeDedupTestAccess::Interleave(*a, held, 2);
  GpuNodeDedupTestAccess::Interleave(*a, held, 3);

  // The slot works as before.
  uint64_t t2 = 0;
  ASSERT_EQ(a->ClaimSg(K(73), &k.s, 1, 4096, &got, &t2),
            GpuNodeDedup::Role::kFetch);
  a->PublishSg(K(73), t2, &k.s, 1, 4096);
  ASSERT_EQ(b->ClaimSg(K(73), &k.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kHit)
      << "the slot's generation is odd at rest";
  EXPECT_EQ(k.dst.Down(4096), k.value);
}

// ---- a slot that changes hands between a lookup and its use ---------------

// K(id) and Neighbour(id) start their probe at the same slot of the 1024-slot
// test table.
BlockKey Neighbour(uint64_t id) { return K(id + 1024); }

TEST(GpuNodeDedup, CopyOutRefusesASlotRecycledForAnotherKey) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("recycle");
  auto a = GpuNodeDedup::Open(Opts(g.name));
  auto b = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  KeyBufs k(80), other(80 + 1024);
  size_t got = 0;
  uint64_t token = 0;
  ASSERT_EQ(a->ClaimSg(K(80), &k.s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  a->PublishSg(K(80), token, &k.s, 1, 4096);
  // A reader has looked the key up ...
  const auto slot = GpuNodeDedupTestAccess::Find(*a, K(80));
  ASSERT_NE(slot, nullptr);
  // ... and before it copies, the entry goes and another process publishes a
  // different key of the same length into the same slot.
  b->Invalidate(K(80));
  ASSERT_EQ(b->ClaimSg(Neighbour(80), &other.s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  b->PublishSg(Neighbour(80), token, &other.s, 1, 4096);
  ASSERT_EQ(GpuNodeDedupTestAccess::Find(*a, Neighbour(80)), slot);

  EXPECT_FALSE(GpuNodeDedupTestAccess::CopyOut(*a, slot, K(80), k.d))
      << "copied another key's bytes";
  // The slot's present owner is still readable through it.
  EXPECT_TRUE(GpuNodeDedupTestAccess::CopyOut(*a, slot, Neighbour(80), k.d));
  EXPECT_EQ(k.dst.Down(4096), other.value);
}

// A token is good for the slot it was issued for only. Generations are small
// per-slot counters, so the slot a key has moved to can carry the very value
// an older claim of the key holds.
TEST(GpuNodeDedup, LatePublishDoesNotFollowTheKeyToAnotherSlot) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("token-slot");
  auto a = GpuNodeDedup::Open(LongWait(g.name));
  auto b = GpuNodeDedup::Open(LongWait(g.name));
  ASSERT_TRUE(a && b);
  KeyBufs v_old(81), v_new(82), n(81 + 1024);
  size_t got = 0;
  uint64_t neighbour = 0, token_a = 0, token_b = 0, ignored = 0;
  // The neighbour holds the first position: the key lives at the second.
  ASSERT_EQ(a->ClaimSg(Neighbour(81), &n.s, 1, 4096, &got, &neighbour),
            GpuNodeDedup::Role::kFetch);
  // One claim-and-invalidate round on the second position, then the claim
  // this test is about: it is that slot's third generation step.
  ASSERT_EQ(a->ClaimSg(K(81), &v_old.s, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kFetch);
  a->Invalidate(K(81));
  ASSERT_EQ(a->ClaimSg(K(81), &v_old.s, 1, 4096, &got, &token_a),
            GpuNodeDedup::Role::kFetch);
  // The neighbour goes, comes back and goes again: the first position is
  // free and two steps along.
  a->Abort(Neighbour(81), neighbour);
  ASSERT_EQ(a->ClaimSg(Neighbour(81), &n.s, 1, 4096, &got, &neighbour),
            GpuNodeDedup::Role::kFetch);
  a->Abort(Neighbour(81), neighbour);
  // The key is overwritten, and the next fetch of it claims the first
  // position, which thereby reaches the generation the old claim holds.
  b->Invalidate(K(81));
  ASSERT_EQ(b->ClaimSg(K(81), &v_new.s, 1, 4096, &got, &token_b),
            GpuNodeDedup::Role::kFetch);
  // The old fetch completes late with the bytes from before the overwrite.
  a->PublishSg(K(81), token_a, &v_old.s, 1, 4096);
  EXPECT_EQ(a->ClaimSg(K(81), &v_old.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kWait)
      << "bytes from before the invalidation were published";
  b->PublishSg(K(81), token_b, &v_new.s, 1, 4096);
  ASSERT_EQ(a->ClaimSg(K(81), &v_old.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kHit);
  EXPECT_EQ(v_old.dst.Down(4096), v_new.value);
}

// ---- slots nobody can use any more are claimed again ----------------------

// An entry whose owner has gone can never be copied. The next claim of the
// key fetches it again in that slot, and the claims after it wait for that
// fetch, instead of each one fetching for itself until the entry expires.
TEST(GpuNodeDedup, EntryOfAGoneOwnerIsFetchedAgainAndShared) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("gone-owner");
  auto a = GpuNodeDedup::Open(LongWait(g.name));  // ttl_ms 30 s
  auto b = GpuNodeDedup::Open(LongWait(g.name));
  auto c = GpuNodeDedup::Open(LongWait(g.name));
  ASSERT_TRUE(a && b && c);
  KeyBufs k(83);
  size_t got = 0;
  uint64_t token = 0, token_b = 0, ignored = 0;
  ASSERT_EQ(a->ClaimSg(K(83), &k.s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  a->PublishSg(K(83), token, &k.s, 1, 4096);
  a.reset();  // the publisher exits: its arena goes
  ASSERT_EQ(b->ClaimSg(K(83), &k.d, 1, 4096, &got, &token_b),
            GpuNodeDedup::Role::kFetch)
      << "the entry of a gone owner still holds the key";
  EXPECT_NE(token_b, 0u) << "fetched without a slot to publish into";
  ASSERT_EQ(c->ClaimSg(K(83), &k.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kWait);
  b->PublishSg(K(83), token_b, &k.s, 1, 4096);
  ASSERT_EQ(c->ClaimSg(K(83), &k.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kHit);
  EXPECT_EQ(k.dst.Down(4096), k.value);
}

TEST(GpuNodeDedup, StaleFetchesDoNotFillTheProbeWindow) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("stale-window");
  auto a = GpuNodeDedup::Open(Opts(g.name));  // takeover_ms 200
  auto b = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  KeyBufs k(84);
  size_t got = 0;
  uint64_t token = 0, ignored = 0;
  for (uint64_t j = 0; j < 8; ++j)  // every position of the window, then dead
    ASSERT_EQ(a->ClaimSg(K(84 + 1024 * j), &k.s, 1, 4096, &got, &ignored),
              GpuNodeDedup::Role::kFetch);
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  const BlockKey other = K(84 + 1024 * 8);
  ASSERT_EQ(b->ClaimSg(other, &k.s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  EXPECT_NE(token, 0u) << "the window is full of dead fetches";
  EXPECT_EQ(a->ClaimSg(other, &k.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kWait);
  b->PublishSg(other, token, &k.s, 1, 4096);
  ASSERT_EQ(a->ClaimSg(other, &k.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kHit);
  EXPECT_EQ(k.dst.Down(4096), k.value);
}

// ---- a segment whose creation did not finish ------------------------------

TEST(GpuNodeDedup, SegmentLeftUnfinishedByADeadCreatorIsFinished) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("dead-creator");
  const GpuNodeDedup::Options o = Opts(g.name);
  pid_t dead = ::fork();
  if (dead == 0) ::_exit(0);
  int st = 0;
  ::waitpid(dead, &st, 0);
  ASSERT_TRUE(dfkv::dedup_segment::AbandonForTest(
      g.name, GpuNodeDedup::SegmentLayout(GpuNodeDedup::SegmentBytes(o)), dead,
      /*sized=*/true));
  auto a = GpuNodeDedup::Open(o);
  ASSERT_TRUE(a) << "every process would run without GPU dedup";
  auto b = GpuNodeDedup::Open(o);
  ASSERT_TRUE(b);
  KeyBufs k(85);
  size_t got = 0;
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->ClaimSg(K(85), &k.s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  a->PublishSg(K(85), token, &k.s, 1, 4096);
  ASSERT_EQ(b->ClaimSg(K(85), &k.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kHit);
  EXPECT_EQ(k.dst.Down(4096), k.value);
}

// ---- registry entries whose pid is in use by another process --------------

// Every registry entry looks held (pid in use): which of them a new process
// may take. Without taking any, it runs without GPU dedup.
std::unique_ptr<GpuNodeDedup> OpenWithRegistryForged(const std::string& name,
                                                     int how) {
  // The instance whose registry is forged stands for processes that are
  // gone or elsewhere: it must not run its destructor, which would release
  // its own entry under whoever holds it now.
  GpuNodeDedup* first = GpuNodeDedup::Open(Opts(name)).release();
  if (!first) return nullptr;
  GpuNodeDedupTestAccess::ForgeRegistry(*first, how);
  return GpuNodeDedup::Open(Opts(name));
}

// A container restarted in place gets the same pids: the entries of the
// previous processes carry the pids of the new ones. They are reclaimed, not
// skipped as this process's own (7 such restarts of 8 ranks used to fill the
// registry and switch GPU dedup off for good).
TEST(GpuNodeDedup, RegistryReclaimsEntriesOfAPreviousProcessWithTheSamePid) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("reg-same-pid");
  auto b = OpenWithRegistryForged(g.name, 0);
  ASSERT_TRUE(b) << "registry full of a previous process's entries";
  auto c = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(c);
  KeyBufs k(86);
  size_t got = 0;
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(b->ClaimSg(K(86), &k.s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  b->PublishSg(K(86), token, &k.s, 1, 4096);
  ASSERT_EQ(c->ClaimSg(K(86), &k.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kHit);
  EXPECT_EQ(k.dst.Down(4096), k.value);
}

TEST(GpuNodeDedup, RegistryReclaimsEntriesWhosePidAnotherProcessNowHas) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("reg-reused-pid");
  EXPECT_TRUE(OpenWithRegistryForged(g.name, 2));
}

// What cannot be proven gone is left alone: a live process, and an entry of
// an older release (which records no identity) with this process's pid.
TEST(GpuNodeDedup, RegistryKeepsEntriesItCannotProveGone) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  {
    ShmGuard g("reg-live");
    EXPECT_FALSE(OpenWithRegistryForged(g.name, 3)) << "took a live entry";
  }
  {
    ShmGuard g("reg-older");
    EXPECT_FALSE(OpenWithRegistryForged(g.name, 1))
        << "took an older release's entry by its pid";
  }
}

// An entry of a previous process with this pid names an arena address of
// an address space that is gone: it must not be read through.
TEST(GpuNodeDedup, AnEntryOfAPreviousProcessWithThisPidIsNotReadDirectly) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("reg-same-pid-va");
  auto a = GpuNodeDedup::Open(Opts(g.name));
  auto b = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  KeyBufs k(87);
  size_t got = 0;
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->ClaimSg(K(87), &k.s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  a->PublishSg(K(87), token, &k.s, 1, 4096);
  GpuNodeDedupTestAccess::ForgeRegistry(*a, 4);  // a's entry: another image
  EXPECT_NE(b->ClaimSg(K(87), &k.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kHit)
      << "read through the address an entry of another process image gives";
}

// ---- a stamp a little newer than the reader's clock -----------------------

TEST(GpuNodeDedup, EntryStampedAfterTheReadersClockIsFresh) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("stamp-ahead");
  auto a = GpuNodeDedup::Open(Opts(g.name));
  auto b = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  KeyBufs k(88);
  size_t got = 0;
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->ClaimSg(K(88), &k.s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  a->PublishSg(K(88), token, &k.s, 1, 4096);
  ASSERT_TRUE(GpuNodeDedupTestAccess::ShiftStamp(*a, K(88), 500));
  ASSERT_EQ(b->ClaimSg(K(88), &k.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kHit)
      << "a fresh entry was taken for an expired one";
  EXPECT_EQ(k.dst.Down(4096), k.value);
  GpuNodeDedup::WaitItem item{K(88), &k.d, 1, 4096, false, 0};
  b->WaitManySg(&item, 1);
  EXPECT_TRUE(item.ok);
}

TEST(GpuNodeDedup, EntryStampedFarAheadExpires) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("stamp-far-ahead");
  auto a = GpuNodeDedup::Open(Opts(g.name));
  auto b = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  KeyBufs k(90);
  size_t got = 0;
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->ClaimSg(K(90), &k.s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  a->PublishSg(K(90), token, &k.s, 1, 4096);
  ASSERT_TRUE(GpuNodeDedupTestAccess::ShiftStamp(*a, K(90), 600000));
  EXPECT_EQ(b->ClaimSg(K(90), &k.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kFetch)
      << "an entry stamped ten minutes ahead was served";
}

// ---- the precondition: one key, the same bytes in every process ------------

TEST(GpuNodeDedup, LengthDisagreementIsCountedAndNotServed) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("len-disagree");
  auto a = GpuNodeDedup::Open(Opts(g.name));
  auto b = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  KeyBufs k(89);
  size_t got = 0;
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->ClaimSg(K(89), &k.s, 1, 4096, &got, &token),
            GpuNodeDedup::Role::kFetch);
  a->PublishSg(K(89), token, &k.s, 1, 4096);
  GpuNodeDedup::Seg small{k.d.ptr, 1024};
  EXPECT_NE(b->ClaimSg(K(89), &small, 1, 1024, &got, &ignored),
            GpuNodeDedup::Role::kHit);
  EXPECT_EQ(b->length_mismatches(), 1u);
  ASSERT_EQ(b->ClaimSg(K(89), &k.d, 1, 4096, &got, &ignored),
            GpuNodeDedup::Role::kHit);
  EXPECT_EQ(b->length_mismatches(), 1u);
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
