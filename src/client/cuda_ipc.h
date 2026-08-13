/* CudaLib — dlopen'd CUDA driver API for GPU rendezvous and final device
 * destination publication.
 *
 * libdfkv.so must not link CUDA: SGLang hosts and CPU-only tools load the same
 * library. Everything here resolves from libcuda.so.1 at first use. Optional
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
constexpr int kCuPointerAttributeMemoryType = 2;    // CU_POINTER_ATTRIBUTE_MEMORY_TYPE
constexpr int kCuPointerAttributeDeviceOrdinal = 9; // CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL
constexpr unsigned kCuMemoryTypeDevice = 2;         // CU_MEMORYTYPE_DEVICE
constexpr unsigned kCuStreamNonBlocking = 1;        // CU_STREAM_NON_BLOCKING

class CudaLib {
 public:
  // dlopens libcuda.so.1 and resolves the surface exactly once per process.
  // nullptr => no usable driver; callers must degrade silently.
  static const CudaLib* Get();

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
  CUresult (*HostRegister)(void*, size_t, unsigned) = nullptr;
  CUresult (*HostUnregister)(void*) = nullptr;
  CUresult (*IpcGetMemHandle)(CUipcMemHandle*, CUdeviceptr) = nullptr;
  CUresult (*IpcOpenMemHandle)(CUdeviceptr*, CUipcMemHandle, unsigned) = nullptr;
  CUresult (*IpcCloseMemHandle)(CUdeviceptr) = nullptr;

 private:
  CudaLib() = default;
  bool Resolve();

  CUresult (*ctx_get_current_)(CUcontext*) = nullptr;
  CUresult (*ctx_set_current_)(CUcontext) = nullptr;
  CUresult (*ctx_get_device_)(int*) = nullptr;
  CUresult (*primary_ctx_retain_)(CUcontext*, int) = nullptr;
  CUresult (*primary_ctx_release_)(int) = nullptr;
  CUresult (*pointer_get_attribute_)(void*, int, CUdeviceptr) = nullptr;
};

}  // namespace dfkv

#endif  // DFKV_CUDA_IPC_H_
