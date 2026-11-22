/* CudaLib — dlopen'd CUDA driver API for GPU rendezvous and final device
 * destination publication.
 *
 * libdfkv.so must not link CUDA: SGLang hosts and CPU-only tools load the same
 * library. Everything here resolves from the libcuda.so.1 the host process
 * loaded and initialized; dfkv never loads or initializes CUDA itself. Optional
 * GPU dedup disables itself when unavailable; an explicitly classified device
 * publication instead returns failure and never reports unpublished bytes.
 * The declarations intentionally mirror cuda.h so no CUDA toolkit is needed at
 * build time; they are ABI, not API (driver exports use fixed C layouts).
 */
#ifndef DFKV_CUDA_IPC_H_
#define DFKV_CUDA_IPC_H_

#include <cstddef>
#include <cstdint>

namespace dfkv {

// Minimal cuda.h mirror (ABI-stable driver types).
using CUresult = int;
using CUdeviceptr = unsigned long long;
using CUcontext = void*;
using CUstream = void*;
struct CUipcMemHandle {
  char reserved[64];
};
constexpr CUresult kCudaSuccess = 0;
constexpr unsigned kCuIpcMemLazyEnablePeerAccess = 0x1;
// CU_POINTER_ATTRIBUTE_SYNC_MEMOPS.
constexpr int kCuPointerAttributeSyncMemops = 6;  // CU_POINTER_ATTRIBUTE_SYNC_MEMOPS
constexpr int kCuPointerAttributeMemoryType = 2;    // CU_POINTER_ATTRIBUTE_MEMORY_TYPE
constexpr int kCuPointerAttributeDeviceOrdinal = 9; // CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL
constexpr unsigned kCuMemoryTypeDevice = 2;         // CU_MEMORYTYPE_DEVICE
constexpr unsigned kCuStreamNonBlocking = 1;        // CU_STREAM_NON_BLOCKING
constexpr unsigned kCuMemHostAllocPortable = 0x1;  // CU_MEMHOSTALLOC_PORTABLE
constexpr unsigned kCuMemHostRegisterPortable = 0x1;  // CU_MEMHOSTREGISTER_PORTABLE

class CudaLib {
 public:
  // The CUDA surface of this process, or nullptr. dfkv only ever attaches to a
  // libcuda.so.1 that the host process has loaded *and* initialized; it never
  // loads the library or calls cuInit itself, so a process that does not use
  // CUDA is never made to open the driver. nullptr => the host has not
  // initialized CUDA (yet), the driver is unusable, or this is a child forked
  // from the attached process. Callers must degrade silently: without CUDA
  // every pointer is host memory.
  //
  // "Not attached" is re-probed, since a framework may initialize CUDA after
  // the client was created. Get() is for per-operation use and probes at most
  // every 100 ms, so it can lag the host's initialization by that much.
  static const CudaLib* Get();
  // Get() without the probe interval: for control-plane calls whose answer is
  // remembered, such as classifying a pool once at registration.
  static const CudaLib* Probe();
  // For a process that is itself the CUDA host -- GPU tools and tests with no
  // framework above them: loads the driver library, initializes it, then
  // attaches. libdfkv never calls this on a caller's behalf.
  static const CudaLib* LoadForHost();

  // True iff p is a CUDA device pointer (UVA query; host/unknown => false).
  bool IsDevicePtr(const void* p) const;
  // Device ordinal owning device pointer p; -1 on failure.
  int DeviceOf(const void* p) const;
  // Current-context access used by scoped publication. GetCurrentCtx succeeds
  // with a null context when the calling thread has no current CUDA context.
  bool GetCurrentCtx(CUcontext* context) const;
  bool SetCurrentCtx(CUcontext context) const;
  // Retain/release are deliberately separate so every scoped retain can be
  // paired on partial setup failure as well as normal teardown.
  bool RetainPrimaryCtx(int dev, CUcontext* context) const;
  bool ReleasePrimaryCtx(int dev) const;
  // Legacy helper for GPU-dedup callers whose context is process-lived.
  bool HasCurrentCtx() const;
  int CurrentDevice() const;  // -1 without a context
  bool BindPrimaryCtx(int dev) const;
  // Arm CU_POINTER_ATTRIBUTE_SYNC_MEMOPS on a GPUDirect destination so CUDA
  // work submissions ordered after an RDMA completion observe the BAR writes.
  // False when the driver lacks cuPointerSetAttribute or rejects the pointer.
  bool SetSyncMemops(const void* p) const;

  CUresult (*MemHostAlloc)(void**, size_t, unsigned) = nullptr;
  CUresult (*MemFreeHost)(void*) = nullptr;
  CUresult (*MemAlloc)(CUdeviceptr*, size_t) = nullptr;
  CUresult (*MemFree)(CUdeviceptr) = nullptr;
  // Unified-addressing copy: any host/device src/dst combination.
  // NOTE (learned live): the synchronous cuMemcpy runs on the LEGACY default
  // stream — it falsely serializes with every compute kernel the framework
  // has in flight (100x slowdowns under load) AND returns before D2D copies
  // complete. The rendezvous therefore uses MemcpyAsync on its own
  // non-blocking stream + StreamSynchronize; plain Memcpy stays for tests.
  CUresult (*Memcpy)(CUdeviceptr, CUdeviceptr, size_t) = nullptr;
  CUresult (*MemcpyAsync)(CUdeviceptr, CUdeviceptr, size_t, CUstream) = nullptr;
  CUresult (*StreamCreate)(CUstream*, unsigned) = nullptr;
  CUresult (*StreamSynchronize)(CUstream) = nullptr;
  CUresult (*StreamDestroy)(CUstream) = nullptr;
  // cuCtxSynchronize on the calling thread's current context. It waits for
  // all work in that context. The GPUDirect RDMA design guide
  // ("Synchronization and Memory Ordering") names CUDA synchronization and
  // work submission calls as what orders GPUDirect RDMA writes before the
  // CUDA work that follows; a synchronization is the one that does not
  // depend on which stream the later work uses. Optional: null when the
  // library lacks it.
  CUresult (*CtxSynchronize)() = nullptr;
  CUresult (*HostRegister)(void*, size_t, unsigned) = nullptr;
  CUresult (*HostUnregister)(void*) = nullptr;
  CUresult (*IpcGetMemHandle)(CUipcMemHandle*, CUdeviceptr) = nullptr;
  CUresult (*IpcOpenMemHandle)(CUdeviceptr*, CUipcMemHandle, unsigned) = nullptr;
  CUresult (*IpcCloseMemHandle)(CUdeviceptr) = nullptr;
  // cuPointerSetAttribute: data points at the attribute's value. Used to arm
  // CU_POINTER_ATTRIBUTE_SYNC_MEMOPS on GPUDirect destinations so a CPU thread
  // that observed an RDMA completion cannot launch a CUDA kernel that reads
  // the destination before the BAR writes are memory-ordered (GPUDirect RDMA
  // design guide, "Synchronization and Memory Ordering").
  CUresult (*PointerSetAttribute)(const void*, int, CUdeviceptr) = nullptr;

 private:
  CudaLib() = default;
  bool Resolve();
  static const CudaLib* Attach(bool throttled);

  CUresult (*ctx_get_current_)(CUcontext*) = nullptr;
  CUresult (*ctx_set_current_)(CUcontext) = nullptr;
  CUresult (*ctx_get_device_)(int*) = nullptr;
  CUresult (*primary_ctx_retain_)(CUcontext*, int) = nullptr;
  CUresult (*primary_ctx_release_)(int) = nullptr;
  CUresult (*pointer_get_attribute_)(void*, int, CUdeviceptr) = nullptr;
  uint64_t owner_process_ = 0;
};

// Process-wide CUDA destination bounce-pool counters. They are intentionally
// cumulative so tests and diagnostics can verify that pinning activity becomes
// flat after lazy warm-up without exposing mutable pool internals.
struct PinnedBouncePoolStats {
  size_t slot_bytes = 0;
  size_t allocated_slots = 0;
  uint64_t allocation_calls = 0;
  uint64_t registration_calls = 0;
  uint64_t wait_count = 0;
  size_t active_leases = 0;
  size_t peak_active_leases = 0;
};

PinnedBouncePoolStats GetPinnedBouncePoolStatsForTest();

}  // namespace dfkv

#endif  // DFKV_CUDA_IPC_H_
