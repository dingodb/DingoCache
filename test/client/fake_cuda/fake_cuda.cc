// A stand-in for libcuda.so.1 used by cuda_attach_test. It exports the driver
// entry points CudaLib resolves so that "dfkv loaded or initialized CUDA in
// this process" is observable on any machine, with or without a GPU, and so
// the test never touches a real driver. Like the driver, everything reports
// CUDA_ERROR_NOT_INITIALIZED until cuInit ran; after that only the current
// context and memory-type queries succeed.
#include <cstddef>

extern "C" {

// A pointer value the fake reports as CUDA device memory.
__attribute__((visibility("default"))) char fake_cuda_device_byte = 0;
__attribute__((visibility("default"))) int fake_cuda_init_calls = 0;

#define DFKV_FAKE_EXPORT __attribute__((visibility("default")))
constexpr int kOk = 0;
constexpr int kError = 1;
constexpr int kNotInitialized = 3;  // CUDA_ERROR_NOT_INITIALIZED

DFKV_FAKE_EXPORT int cuInit(unsigned) {
  ++fake_cuda_init_calls;
  return kOk;
}
DFKV_FAKE_EXPORT int cuPointerGetAttribute(void* data, int attribute,
                                          unsigned long long pointer) {
  constexpr int kMemoryType = 2;       // CU_POINTER_ATTRIBUTE_MEMORY_TYPE
  constexpr unsigned kDevice = 2;      // CU_MEMORYTYPE_DEVICE
  if (fake_cuda_init_calls == 0) return kNotInitialized;
  if (attribute != kMemoryType || data == nullptr) return kError;
  if (pointer != reinterpret_cast<unsigned long long>(&fake_cuda_device_byte))
    return kError;  // like the driver: unknown host memory fails the query
  *static_cast<unsigned*>(data) = kDevice;
  return kOk;
}
DFKV_FAKE_EXPORT int cuMemHostAlloc(void**, size_t, unsigned) { return kError; }
DFKV_FAKE_EXPORT int cuMemFreeHost(void*) { return kError; }
DFKV_FAKE_EXPORT int cuMemAlloc_v2(unsigned long long*, size_t) { return kError; }
DFKV_FAKE_EXPORT int cuMemFree_v2(unsigned long long) { return kError; }
DFKV_FAKE_EXPORT int cuMemcpy(unsigned long long, unsigned long long, size_t) {
  return kError;
}
DFKV_FAKE_EXPORT int cuMemcpyAsync(unsigned long long, unsigned long long,
                                  size_t, void*) {
  return kError;
}
DFKV_FAKE_EXPORT int cuStreamCreate(void**, unsigned) { return kError; }
DFKV_FAKE_EXPORT int cuStreamSynchronize(void*) { return kError; }
DFKV_FAKE_EXPORT int cuStreamDestroy_v2(void*) { return kError; }
DFKV_FAKE_EXPORT int cuMemHostRegister(void*, size_t, unsigned) { return kError; }
DFKV_FAKE_EXPORT int cuMemHostUnregister(void*) { return kError; }
DFKV_FAKE_EXPORT int cuIpcGetMemHandle(void*, unsigned long long) { return kError; }
DFKV_FAKE_EXPORT int cuIpcOpenMemHandle_v2(unsigned long long*, void*, unsigned) {
  return kError;
}
DFKV_FAKE_EXPORT int cuIpcCloseMemHandle(unsigned long long) { return kError; }
DFKV_FAKE_EXPORT int cuPointerSetAttribute(const void*, int, unsigned long long) {
  return kError;
}
DFKV_FAKE_EXPORT int cuCtxGetCurrent(void** context) {
  if (fake_cuda_init_calls == 0) return kNotInitialized;
  if (context != nullptr) *context = nullptr;
  return kOk;
}
DFKV_FAKE_EXPORT int cuCtxSetCurrent(void*) { return kError; }
DFKV_FAKE_EXPORT int cuCtxGetDevice(int*) { return kError; }
DFKV_FAKE_EXPORT int cuDevicePrimaryCtxRetain(void**, int) { return kError; }
DFKV_FAKE_EXPORT int cuDevicePrimaryCtxRelease_v2(int) { return kError; }

}  // extern "C"
