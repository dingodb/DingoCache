// A simulated libcuda.so.1 for the GPU dedup tests. "Device memory" is host
// memory that this library hands out and recognises, so the code under test
// runs its real paths (contexts, a staging arena, cross-process handles,
// asynchronous copies on a stream) on a machine without a GPU and without
// touching a real driver. It is not installed and not linked into anything:
// the test binary re-executes itself with this directory first on the
// library path.
//
// What is modelled, and only that:
//  - one device (ordinal 0) with one primary context; a context is current
//    per thread, as with the driver;
//  - allocations are shared memory, so a handle exported by one process can
//    be opened by another (cuIpc*);
//  - copies run at once on the calling thread; streams exist but queue nothing;
//  - GPUDirect arrival order: sim_cuda_gpudirect_write() stands for a NIC
//    writing device memory. The bytes are held back and reach the memory at
//    the next CUDA synchronization call (cuCtxSynchronize, cuStreamSynchronize),
//    which is the one thing the GPUDirect RDMA design guide says orders such
//    writes before later CUDA work. A copy issued before that reads what was
//    there before.
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr int kOk = 0;
constexpr int kInvalidValue = 1;     // CUDA_ERROR_INVALID_VALUE
constexpr int kOutOfMemory = 2;      // CUDA_ERROR_OUT_OF_MEMORY
constexpr int kNotInitialized = 3;   // CUDA_ERROR_NOT_INITIALIZED
constexpr int kInvalidContext = 201; // CUDA_ERROR_INVALID_CONTEXT
constexpr int kUnknown = 999;        // CUDA_ERROR_UNKNOWN

struct Region {
  size_t size = 0;
  int fd = -1;       // backing memfd of an allocation made here
  bool owned = false;
};

struct PendingWrite {
  char* dst;
  std::string bytes;
};

std::mutex g_mu;
bool g_initialized = false;
std::map<uintptr_t, Region> g_regions;  // base -> region ("device memory")
std::vector<PendingWrite> g_pending;
int g_context = 0;                      // its address is the primary context
thread_local void* tls_current = nullptr;

// The region containing [p, p+n), or nullptr. Caller holds g_mu.
const Region* FindLocked(uintptr_t p, size_t n, uintptr_t* base = nullptr) {
  auto it = g_regions.upper_bound(p);
  if (it == g_regions.begin()) return nullptr;
  --it;
  if (p < it->first || p - it->first > it->second.size ||
      n > it->second.size - (p - it->first))
    return nullptr;
  if (base) *base = it->first;
  return &it->second;
}

void SettleLocked() {
  for (const PendingWrite& w : g_pending)
    std::memcpy(w.dst, w.bytes.data(), w.bytes.size());
  g_pending.clear();
}

struct IpcHandle {  // fits CUipcMemHandle (64 bytes)
  uint32_t magic;
  int32_t pid;
  int32_t fd;
  uint32_t pad;
  uint64_t size;
};
constexpr uint32_t kHandleMagic = 0x53494d43u;  // "SIMC"

struct CUipcMemHandle {
  char reserved[64];
};

}  // namespace

#define SIM_EXPORT __attribute__((visibility("default")))

extern "C" {

// ---- test controls --------------------------------------------------------

SIM_EXPORT int sim_cuda_ctx_synchronize_calls = 0;
SIM_EXPORT int sim_cuda_memcpy_async_calls = 0;
// Non-zero: cuCtxSynchronize fails (and settles nothing).
SIM_EXPORT int sim_cuda_fail_ctx_synchronize = 0;

// A GPUDirect write: the bytes arrive at the next synchronization call.
SIM_EXPORT void sim_cuda_gpudirect_write(void* dst, const void* src,
                                         size_t n) {
  std::lock_guard<std::mutex> lock(g_mu);
  g_pending.push_back(PendingWrite{
      static_cast<char*>(dst),
      std::string(static_cast<const char*>(src), n)});
}

// ---- driver entry points --------------------------------------------------

SIM_EXPORT int cuInit(unsigned) {
  std::lock_guard<std::mutex> lock(g_mu);
  g_initialized = true;
  return kOk;
}
SIM_EXPORT int cuDeviceGet(int* device, int ordinal) {
  if (!device || ordinal != 0) return kInvalidValue;
  *device = 0;
  return kOk;
}
SIM_EXPORT int cuDevicePrimaryCtxRetain(void** context, int device) {
  if (!context || device != 0) return kInvalidValue;
  *context = &g_context;
  return kOk;
}
SIM_EXPORT int cuDevicePrimaryCtxRelease_v2(int device) {
  return device == 0 ? kOk : kInvalidValue;
}
SIM_EXPORT int cuCtxGetCurrent(void** context) {
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_initialized) return kNotInitialized;
  }
  if (context) *context = tls_current;
  return kOk;
}
SIM_EXPORT int cuCtxSetCurrent(void* context) {
  if (context != nullptr && context != &g_context) return kInvalidContext;
  tls_current = context;
  return kOk;
}
SIM_EXPORT int cuCtxGetDevice(int* device) {
  if (!tls_current) return kInvalidContext;
  if (device) *device = 0;
  return kOk;
}
SIM_EXPORT int cuCtxSynchronize() {
  if (!tls_current) return kInvalidContext;
  std::lock_guard<std::mutex> lock(g_mu);
  ++sim_cuda_ctx_synchronize_calls;
  if (sim_cuda_fail_ctx_synchronize) return kUnknown;
  SettleLocked();
  return kOk;
}

SIM_EXPORT int cuMemAlloc_v2(unsigned long long* pointer, size_t size) {
  if (!pointer || size == 0) return kInvalidValue;
  if (!tls_current) return kInvalidContext;
  const int fd = ::memfd_create("dfkv-sim-cuda", 0);
  if (fd < 0) return kOutOfMemory;
  if (::ftruncate(fd, static_cast<off_t>(size)) != 0) {
    ::close(fd);
    return kOutOfMemory;
  }
  void* base =
      ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (base == MAP_FAILED) {
    ::close(fd);
    return kOutOfMemory;
  }
  std::lock_guard<std::mutex> lock(g_mu);
  g_regions[reinterpret_cast<uintptr_t>(base)] = Region{size, fd, true};
  *pointer = reinterpret_cast<unsigned long long>(base);
  return kOk;
}
SIM_EXPORT int cuMemFree_v2(unsigned long long pointer) {
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_regions.find(static_cast<uintptr_t>(pointer));
  if (it == g_regions.end() || !it->second.owned) return kInvalidValue;
  ::munmap(reinterpret_cast<void*>(it->first), it->second.size);
  ::close(it->second.fd);
  g_regions.erase(it);
  return kOk;
}

SIM_EXPORT int cuMemcpy(unsigned long long dst, unsigned long long src,
                        size_t size) {
  if (!tls_current) return kInvalidContext;
  std::memcpy(reinterpret_cast<void*>(dst), reinterpret_cast<void*>(src), size);
  return kOk;
}
SIM_EXPORT int cuMemcpyAsync(unsigned long long dst, unsigned long long src,
                             size_t size, void* stream) {
  if (!tls_current) return kInvalidContext;
  if (!stream) return kInvalidValue;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    ++sim_cuda_memcpy_async_calls;
    // Like the driver, refuse a device range that is not mapped (any more).
    const auto mapped = [&](unsigned long long p) {
      auto it = g_regions.upper_bound(static_cast<uintptr_t>(p));
      if (it == g_regions.begin()) return true;  // host memory: not ours
      --it;
      const uintptr_t off = static_cast<uintptr_t>(p) - it->first;
      return off >= it->second.size || size <= it->second.size - off;
    };
    if (!mapped(dst) || !mapped(src)) return kInvalidValue;
  }
  std::memcpy(reinterpret_cast<void*>(dst), reinterpret_cast<void*>(src), size);
  return kOk;
}
SIM_EXPORT int cuStreamCreate(void** stream, unsigned) {
  if (!stream) return kInvalidValue;
  if (!tls_current) return kInvalidContext;
  *stream = new int(0);
  return kOk;
}
SIM_EXPORT int cuStreamSynchronize(void* stream) {
  if (!stream) return kInvalidValue;
  if (!tls_current) return kInvalidContext;
  std::lock_guard<std::mutex> lock(g_mu);
  SettleLocked();
  return kOk;
}
SIM_EXPORT int cuStreamDestroy_v2(void* stream) {
  delete static_cast<int*>(stream);
  return kOk;
}

SIM_EXPORT int cuMemHostAlloc(void** pointer, size_t size, unsigned) {
  if (!pointer) return kInvalidValue;
  *pointer = ::operator new(size, std::nothrow);
  return *pointer ? kOk : kOutOfMemory;
}
SIM_EXPORT int cuMemFreeHost(void* pointer) {
  ::operator delete(pointer);
  return kOk;
}
SIM_EXPORT int cuMemHostRegister(void*, size_t, unsigned) { return kOk; }
SIM_EXPORT int cuMemHostUnregister(void*) { return kOk; }

SIM_EXPORT int cuIpcGetMemHandle(CUipcMemHandle* handle,
                                 unsigned long long pointer) {
  if (!handle) return kInvalidValue;
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_regions.find(static_cast<uintptr_t>(pointer));
  if (it == g_regions.end() || !it->second.owned) return kInvalidValue;
  IpcHandle h{kHandleMagic, static_cast<int32_t>(::getpid()), it->second.fd, 0,
              it->second.size};
  std::memset(handle, 0, sizeof(*handle));
  std::memcpy(handle, &h, sizeof(h));
  return kOk;
}
SIM_EXPORT int cuIpcOpenMemHandle_v2(unsigned long long* pointer,
                                     CUipcMemHandle handle, unsigned) {
  if (!pointer) return kInvalidValue;
  if (!tls_current) return kInvalidContext;
  IpcHandle h;
  std::memcpy(&h, &handle, sizeof(h));
  if (h.magic != kHandleMagic) return kInvalidValue;
  // The exporter's descriptor, reached through /proc: the same file whether
  // the exporter is this process or another one of the same user.
  char path[64];
  std::snprintf(path, sizeof(path), "/proc/%d/fd/%d", h.pid, h.fd);
  const int fd = ::open(path, O_RDWR);
  if (fd < 0) return kInvalidValue;
  void* base = ::mmap(nullptr, h.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ::close(fd);
  if (base == MAP_FAILED) return kOutOfMemory;
  std::lock_guard<std::mutex> lock(g_mu);
  g_regions[reinterpret_cast<uintptr_t>(base)] =
      Region{static_cast<size_t>(h.size), -1, false};
  *pointer = reinterpret_cast<unsigned long long>(base);
  return kOk;
}
SIM_EXPORT int cuIpcCloseMemHandle(unsigned long long pointer) {
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_regions.find(static_cast<uintptr_t>(pointer));
  if (it == g_regions.end() || it->second.owned) return kInvalidValue;
  ::munmap(reinterpret_cast<void*>(it->first), it->second.size);
  g_regions.erase(it);
  return kOk;
}

SIM_EXPORT int cuPointerGetAttribute(void* data, int attribute,
                                     unsigned long long pointer) {
  constexpr int kMemoryType = 2;     // CU_POINTER_ATTRIBUTE_MEMORY_TYPE
  constexpr int kDeviceOrdinal = 9;  // CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL
  std::lock_guard<std::mutex> lock(g_mu);
  if (!g_initialized) return kNotInitialized;
  if (!data) return kInvalidValue;
  // Like the driver: memory it does not know fails the query.
  if (!FindLocked(static_cast<uintptr_t>(pointer), 1)) return kInvalidValue;
  if (attribute == kMemoryType) {
    *static_cast<unsigned*>(data) = 2;  // CU_MEMORYTYPE_DEVICE
    return kOk;
  }
  if (attribute == kDeviceOrdinal) {
    *static_cast<int*>(data) = 0;
    return kOk;
  }
  return kInvalidValue;
}
SIM_EXPORT int cuPointerSetAttribute(const void*, int,
                                     unsigned long long pointer) {
  std::lock_guard<std::mutex> lock(g_mu);
  return FindLocked(static_cast<uintptr_t>(pointer), 1) ? kOk : kInvalidValue;
}

}  // extern "C"
