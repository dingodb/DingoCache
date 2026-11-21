// CudaLib must only attach to a CUDA driver library the host process loaded
// itself. This binary runs with a fake libcuda.so.1 first on LD_LIBRARY_PATH
// (see CMakeLists.txt), so "did dfkv load CUDA?" is observable everywhere and
// no real driver is ever touched.
#include "client/cuda_ipc.h"

#include <dlfcn.h>

#include <chrono>
#include <thread>

#include <gtest/gtest.h>

namespace dfkv {
namespace {

bool CudaLibraryLoaded() {
  void* handle = ::dlopen("libcuda.so.1", RTLD_NOLOAD | RTLD_NOW);
  if (handle == nullptr) return false;
  ::dlclose(handle);
  return true;
}

// One test, three ordered phases: CudaLib keeps process-wide state.
TEST(CudaAttach, NeverLoadsCudaAndAttachesOnlyAfterTheHostDid) {
  ASSERT_FALSE(CudaLibraryLoaded())
      << "precondition: this binary must not link or preload CUDA";

  // Phase 1: a process that never loaded CUDA. Classifying destinations must
  // not bring the driver in -- before, the first query dlopen'ed libcuda and
  // ran cuInit in every dfkv client process on a GPU host.
  EXPECT_EQ(CudaLib::Get(), nullptr);
  EXPECT_EQ(CudaLib::Get(), nullptr);
  EXPECT_FALSE(CudaLibraryLoaded())
      << "dfkv loaded libcuda into a process that had not loaded it";

  // Phase 2: the host framework loads CUDA later than the first query.
  void* host = ::dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
  ASSERT_NE(host, nullptr)
      << "fake libcuda.so.1 is not on LD_LIBRARY_PATH: " << ::dlerror();
  ASSERT_NE(::dlsym(host, "fake_cuda_device_byte"), nullptr)
      << "resolved a real libcuda instead of the test fake";

  // Phase 3: an absent library is re-probed (rate limited), then attached.
  const CudaLib* cuda = nullptr;
  for (int i = 0; i < 400 && cuda == nullptr; ++i) {
    cuda = CudaLib::Get();
    if (cuda == nullptr)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  ASSERT_NE(cuda, nullptr) << "did not attach to the host-loaded library";
  EXPECT_EQ(CudaLib::Get(), cuda);

  char host_byte = 0;
  EXPECT_FALSE(cuda->IsDevicePtr(&host_byte));
  EXPECT_TRUE(cuda->IsDevicePtr(::dlsym(host, "fake_cuda_device_byte")));
}

}  // namespace
}  // namespace dfkv
