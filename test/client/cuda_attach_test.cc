// CudaLib must only attach to a CUDA driver library that the host process
// loaded and initialized itself. The scenarios run against a fake libcuda.so.1
// (test/client/fake_cuda), which makes "did dfkv load or initialize CUDA?"
// observable on any machine. main() re-executes the binary with the fake first
// on the loader path, so not even a regression can reach a real driver.
#include "client/cuda_ipc.h"
#include "client/kv_client.h"
#include "transport/transport.h"

#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#ifndef DFKV_FAKE_CUDA_DIR
#error "DFKV_FAKE_CUDA_DIR must name the directory of the fake libcuda.so.1"
#endif

namespace dfkv {
namespace {

constexpr char kFakeLibrary[] = DFKV_FAKE_CUDA_DIR "/libcuda.so.1";

bool CudaLibraryLoaded() {
  void* handle = ::dlopen("libcuda.so.1", RTLD_NOLOAD | RTLD_NOW);
  if (handle == nullptr) return false;
  ::dlclose(handle);
  return true;
}

// What a host framework does on its own, at a time of its choosing.
void* HostLoadsCuda() { return ::dlopen(kFakeLibrary, RTLD_NOW | RTLD_LOCAL); }

bool HostInitializesCuda(void* library) {
  auto init = reinterpret_cast<int (*)(unsigned)>(::dlsym(library, "cuInit"));
  return init != nullptr && init(0) == 0;
}

int InitCalls(void* library) {
  auto* calls = static_cast<int*>(::dlsym(library, "fake_cuda_init_calls"));
  return calls != nullptr ? *calls : -1;  // -1: not the fake
}

void* DevicePointer(void* library) {
  return ::dlsym(library, "fake_cuda_device_byte");
}

// CudaLib keeps process-wide state and every scenario has to start in a
// process without CUDA, so each one runs in its own forked child.
template <typename Scenario>
void RunInFreshProcess(Scenario&& scenario) {
  ASSERT_FALSE(CudaLibraryLoaded())
      << "the test process itself must stay free of CUDA";
  ::fflush(nullptr);
  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    scenario();
    ::fflush(nullptr);
    ::_exit(::testing::Test::HasFailure() ? 1 : 0);
  }
  int status = 0;
  ASSERT_EQ(::waitpid(child, &status, 0), child);
  EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0)
      << "the scenario failed in its child process; see the failures above";
}

// Only RegisterMemory is reached by these scenarios.
class NoIoTransport : public Transport {
 public:
  Status Cache(const std::string&, const BlockKey&, const void*,
               size_t) override {
    return Status::kIOError;
  }
  Status Range(const std::string&, const BlockKey&, uint64_t, uint64_t,
               std::string*, uint64_t*) override {
    return Status::kIOError;
  }
  Status Lookup(const std::string&, const BlockKey&, uint64_t*) override {
    return Status::kIOError;
  }
  Status Exist(const std::string&, const BlockKey&, bool*) override {
    return Status::kIOError;
  }
};

// Before, the first query dlopen'ed libcuda and ran cuInit in every dfkv
// client process on a GPU host.
TEST(CudaAttach, NeverLoadsCuda) {
  RunInFreshProcess([] {
    EXPECT_EQ(CudaLib::Get(), nullptr);
    EXPECT_EQ(CudaLib::Probe(), nullptr);
    EXPECT_FALSE(CudaLibraryLoaded())
        << "dfkv loaded libcuda into a process that had not loaded it";
  });
}

// A process can map the driver library without ever using CUDA.
TEST(CudaAttach, NeverInitializesCudaForTheHost) {
  RunInFreshProcess([] {
    void* host = HostLoadsCuda();
    ASSERT_NE(host, nullptr) << ::dlerror();
    EXPECT_EQ(CudaLib::Probe(), nullptr)
        << "attached to a library the host has not initialized";
    EXPECT_EQ(InitCalls(host), 0) << "dfkv called cuInit on the host's behalf";
  });
}

TEST(CudaAttach, GetAttachesAfterTheHostInitializedCuda) {
  RunInFreshProcess([] {
    EXPECT_EQ(CudaLib::Get(), nullptr);  // starts the probe interval
    void* host = HostLoadsCuda();
    ASSERT_NE(host, nullptr) << ::dlerror();
    ASSERT_TRUE(HostInitializesCuda(host));

    const CudaLib* cuda = nullptr;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (cuda == nullptr && std::chrono::steady_clock::now() < deadline) {
      cuda = CudaLib::Get();
      if (cuda == nullptr)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ASSERT_NE(cuda, nullptr) << "never attached to the host's library";
    EXPECT_EQ(CudaLib::Get(), cuda);
    EXPECT_EQ(InitCalls(host), 1) << "only the host may call cuInit";

    char host_byte = 0;
    EXPECT_FALSE(cuda->IsDevicePtr(&host_byte));
    EXPECT_TRUE(cuda->IsDevicePtr(DevicePointer(host)));
  });
}

// A registered pool keeps the kind decided at registration, so that decision
// must not come from a probe older than the host's CUDA setup.
TEST(CudaAttach, PoolRegisteredRightAfterCudaCameUpIsDeviceMemory) {
  RunInFreshProcess([] {
    NoIoTransport transport;
    KVClient client({{"n", "10.0.0.1:1"}}, "test/model", &transport);
    char host_byte = 0;
    // An unregistered destination, classified while the process has no CUDA:
    // this starts the probe interval.
    EXPECT_EQ(client.ClassifyDestination(&host_byte, 1),
              DestinationMemoryKind::kHost);

    void* host = HostLoadsCuda();
    ASSERT_NE(host, nullptr) << ::dlerror();
    ASSERT_TRUE(HostInitializesCuda(host));
    void* pool = DevicePointer(host);
    ASSERT_TRUE(client.RegisterMemory(pool, 1));
    EXPECT_EQ(client.ClassifyDestination(pool, 1),
              DestinationMemoryKind::kDevice)
        << "a device pool registered inside the probe interval was "
           "remembered as host memory";
  });
}

TEST(CudaAttach, LoadForHostLoadsInitializesAndAttaches) {
  RunInFreshProcess([] {
    EXPECT_EQ(CudaLib::Get(), nullptr);  // starts the probe interval
    const CudaLib* cuda = CudaLib::LoadForHost();
    void* loaded = ::dlopen("libcuda.so.1", RTLD_NOLOAD | RTLD_NOW);
    ASSERT_NE(loaded, nullptr) << "LoadForHost did not load the library";
    ASSERT_EQ(InitCalls(loaded), 1) << "expected the fake, initialized once";
    ASSERT_NE(cuda, nullptr);
    EXPECT_EQ(CudaLib::Get(), cuda);
  });
}

}  // namespace
}  // namespace dfkv

int main(int argc, char** argv) {
  // The fake must be the first libcuda.so.1 the loader can find, whatever
  // environment the test was started from: re-execute once with it prepended.
  constexpr char kReexecuted[] = "DFKV_CUDA_ATTACH_TEST_REEXECUTED";
  if (::getenv(kReexecuted) == nullptr) {
    std::string path = DFKV_FAKE_CUDA_DIR;
    const char* inherited = ::getenv("LD_LIBRARY_PATH");
    if (inherited != nullptr && *inherited != '\0') {
      path += ':';
      path += inherited;
    }
    ::setenv("LD_LIBRARY_PATH", path.c_str(), 1);
    ::setenv(kReexecuted, "1", 1);
    ::execv("/proc/self/exe", argv);
    std::perror("cuda_attach_test: re-exec");
    return 127;
  }
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
