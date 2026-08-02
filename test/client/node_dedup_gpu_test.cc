// GpuNodeDedup: same-host GET rendezvous for device-memory destinations over
// CUDA IPC. Hermetic wrt dfkv (no RDMA/etcd) but NOT wrt hardware: everything
// past the env/name tests needs a CUDA device and skips cleanly without one
// (the CI runners have no GPU; the B200 gate runs the full set). Cross-process
// coverage forks AND EXECS this binary — CUDA does not survive a bare fork —
// with the segment name in the environment (see GpuDedupChild).
#include "client/node_dedup_gpu.h"

#include <gtest/gtest.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "client/cuda_ipc.h"

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
  EXPECT_NE(n.find("/dfkv-dedup-gpuv1-"), std::string::npos);
  EXPECT_NE(n.find("0000000000abcdef"), std::string::npos);
  // distinct from the host segment namespace
  EXPECT_EQ(n.find("/dfkv-dedup-v2-"), std::string::npos);
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
  ASSERT_EQ(a->ClaimSg(K(1), src, 2, 8192, &got), GpuNodeDedup::Role::kFetch);
  a->PublishSg(K(1), src, 2, v.size());

  // Peer scatters into a DIFFERENT segment split (3000 + 5192).
  DevBuf d0(3000), d1(5192);
  GpuNodeDedup::Seg dst[2] = {{reinterpret_cast<void*>(d0.p), 3000},
                              {reinterpret_cast<void*>(d1.p), 5192}};
  got = 0;
  ASSERT_EQ(b->ClaimSg(K(1), dst, 2, 8192, &got), GpuNodeDedup::Role::kHit);
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
  ASSERT_EQ(a->ClaimSg(K(2), &ss, 1, 65536, &got), GpuNodeDedup::Role::kFetch);
  ASSERT_EQ(b->ClaimSg(K(2), &ds, 1, 65536, &got), GpuNodeDedup::Role::kWait);
  std::thread pub([&] {
    ASSERT_TRUE(EnsureCudaCtx());  // publish thread needs its own current ctx
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    a->PublishSg(K(2), &ss, 1, v.size());
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
    ASSERT_EQ(a->ClaimSg(K(9), &ss, 1, 32768, &got), GpuNodeDedup::Role::kFetch);
    a->PublishSg(K(9), &ss, 1, v.size());
  });
  t.join();
  size_t got = 0;
  ASSERT_EQ(a->ClaimSg(K(9), &ds, 1, 32768, &got), GpuNodeDedup::Role::kHit);
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
  for (size_t i = 0; i < kN; ++i) {
    vals.push_back(Val(100 + i, 8192));
    srcs.push_back(std::make_unique<DevBuf>(8192));
    dsts.push_back(std::make_unique<DevBuf>(8192));
    srcs[i]->Up(vals[i]);
    ssegs[i] = {reinterpret_cast<void*>(srcs[i]->p), 8192};
    dsegs[i] = {reinterpret_cast<void*>(dsts[i]->p), 8192};
    ASSERT_EQ(a->ClaimSg(K(100 + i), &ssegs[i], 1, 8192, &got),
              GpuNodeDedup::Role::kFetch);
    ASSERT_EQ(b->ClaimSg(K(100 + i), &dsegs[i], 1, 8192, &got),
              GpuNodeDedup::Role::kWait);
    wits[i] = GpuNodeDedup::WaitItem{K(100 + i), &dsegs[i], 1, 8192, false, 0};
  }
  std::thread pub([&] {  // publishes trickle in while the batch wait runs
    ASSERT_TRUE(EnsureCudaCtx());
    for (size_t i = 0; i < kN; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      a->PublishSg(K(100 + i), &ssegs[i], 1, vals[i].size());
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
  ASSERT_EQ(a->ClaimSg(K(3), &s, 1, 4096, &got), GpuNodeDedup::Role::kFetch);
  a->Abort(K(3));
  // Slot freed: the next claimant fetches (no stuck FETCHING until takeover).
  ASSERT_EQ(a->ClaimSg(K(3), &s, 1, 4096, &got), GpuNodeDedup::Role::kFetch);
}

TEST(GpuNodeDedup, OversizePayloadSkipsRendezvous) {
  if (!EnsureCudaCtx()) GTEST_SKIP() << "no CUDA device";
  ShmGuard g("oversize");
  auto a = GpuNodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a);
  DevBuf d(4096);
  GpuNodeDedup::Seg s{reinterpret_cast<void*>(d.p), 4096};
  size_t got = 0;
  // cap > arena/2: no slot is claimed, so no publish obligation either.
  EXPECT_EQ(a->ClaimSg(K(4), &s, 1, (32ull << 20), &got),
            GpuNodeDedup::Role::kFetch);
  a->Abort(K(4));  // must be a harmless no-op (nothing was reserved)
  EXPECT_EQ(a->ClaimSg(K(4), &s, 1, (32ull << 20), &got),
            GpuNodeDedup::Role::kFetch);
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
  ASSERT_EQ(d->ClaimSg(K(7), dst, 2, 40000, &got), GpuNodeDedup::Role::kHit);
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
  ASSERT_EQ(a->ClaimSg(K(7), &s, 1, 40000, &got), GpuNodeDedup::Role::kFetch);
  a->PublishSg(K(7), &s, 1, v.size());

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
