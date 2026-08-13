#include "client/cuda_ipc.h"
#include "transport/transport.h"

#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <string>

#include "utils/log.h"

namespace dfkv {

namespace {
// Prefer the versioned symbol ONLY where cuda.h itself binds it (the classic
// 32->64-bit _v2 set: cuMemAlloc/cuMemFree/cuIpcOpenMemHandle) — there the
// _v2 signature is the one we declare. NEVER guess _v2 for other entry
// points: drivers export e.g. cuCtxGetDevice_v2 with a DIFFERENT signature
// (an extra CUcontext parameter), and calling it through the plain prototype
// reads a garbage register — a segfault that only fires on some threads
// (found live: one of four vLLM workers died in exactly that call).
void* SymV2(void* h, const char* v2, const char* legacy) {
  if (void* p = ::dlsym(h, v2)) return p;
  return ::dlsym(h, legacy);
}

constexpr size_t kDefaultPinnedPoolBytes = size_t{64} << 20;
constexpr size_t kDefaultPinnedSlotBytes = size_t{4} << 20;
constexpr size_t kMinPinnedSlotBytes = size_t{4} << 10;
constexpr size_t kMaxPinnedSlotBytes = size_t{64} << 20;
constexpr size_t kMaxPinnedPoolBytes = size_t{4} << 30;
constexpr size_t kMaxPinnedSlots = 4096;

void NotePublisherDriverFailure(const std::string& detail) {
  static std::atomic<uint64_t> failures{0};
  const uint64_t count =
      failures.fetch_add(1, std::memory_order_relaxed) + 1;
  if (count == 1 || (count & 0x3ffu) == 0) {
    DFKV_LOG_WARN("cuda destination publication driver failure: " + detail +
                  " failures=" + std::to_string(count));
  }
}

size_t HostPageSize() {
  static const size_t page_size = [] {
    const long value = ::sysconf(_SC_PAGESIZE);
    return value > 0 ? static_cast<size_t>(value) : size_t{0};
  }();
  return page_size;
}

size_t ParsePinnedBytes(const char* name, size_t fallback) {
  const char* value = std::getenv(name);
  if (!value || !*value) return fallback;
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed == 0 ||
      parsed > std::numeric_limits<size_t>::max()) {
    DFKV_LOG_WARN(std::string("invalid ") + name + "='" + value +
                  "'; using " + std::to_string(fallback));
    return fallback;
  }
  return static_cast<size_t>(parsed);
}

struct PublisherCudaState {
  CUstream stream = nullptr;
  CUcontext context = nullptr;
  int retained_device = -1;
  size_t copy_count = 0;
  CUcontext pending_restore = nullptr;
  bool has_pending_restore = false;
  pid_t owner_process = 0;
};

struct BounceLease {
  void* data = nullptr;
  size_t index = 0;
};

class PinnedBouncePool {
 public:
  PinnedBouncePool() : owner_process_(::getpid()) {
    size_t slot_bytes =
        ParsePinnedBytes("DFKV_CUDA_PINNED_SLOT_BYTES",
                         kDefaultPinnedSlotBytes);
    size_t pool_bytes =
        ParsePinnedBytes("DFKV_CUDA_PINNED_POOL_BYTES",
                         kDefaultPinnedPoolBytes);
    const size_t slot_count =
        slot_bytes == 0 ? 0 : pool_bytes / slot_bytes;
    if (slot_bytes < kMinPinnedSlotBytes ||
        slot_bytes > kMaxPinnedSlotBytes ||
        pool_bytes < slot_bytes || pool_bytes > kMaxPinnedPoolBytes ||
        slot_count == 0 || slot_count > kMaxPinnedSlots) {
      DFKV_LOG_WARN(
          "invalid CUDA pinned bounce-pool sizing; using pool_bytes=" +
          std::to_string(kDefaultPinnedPoolBytes) +
          " slot_bytes=" + std::to_string(kDefaultPinnedSlotBytes));
      slot_bytes = kDefaultPinnedSlotBytes;
      pool_bytes = kDefaultPinnedPoolBytes;
    }
    slot_bytes_ = slot_bytes;
    max_slots_ = pool_bytes / slot_bytes;
    slots_ = new (std::nothrow) Slot[max_slots_];
    if (!slots_) {
      max_slots_ = 0;
      allocation_disabled_ = true;
      NotePublisherDriverFailure(
          "pinned bounce-pool metadata allocation failed");
    }
  }

  ~PinnedBouncePool() {
    // A fork child inherited CUDA allocations and possibly a locked mutex from
    // the parent. CUDA is not fork-safe: let process exit reclaim those pages
    // rather than touching either inherited object.
    if (::getpid() != owner_process_) return;

    {
      std::unique_lock<std::mutex> lock(mutex_);
      shutting_down_ = true;
      available_.notify_all();
      available_.wait(lock, [&] {
        return active_leases_ == 0 && allocations_in_progress_ == 0;
      });
    }

    for (size_t i = 0; i < max_slots_; ++i) {
      // A failed stream synchronization never proves DMA completion. Keep
      // such a slot pinned until process exit instead of freeing backing that
      // the driver might still reference.
      if (!slots_[i].data || slots_[i].poisoned) continue;
      FreeAllocation(slots_[i].cuda, slots_[i].data,
                     slots_[i].registered_fallback, "pool teardown");
    }
    delete[] slots_;
  }

  bool Acquire(const CudaLib* cuda, BounceLease* lease) {
    if (!cuda || !lease || ::getpid() != owner_process_) return false;
    bool counted_wait = false;
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      if (shutting_down_) return false;

      for (size_t i = 0; i < max_slots_; ++i) {
        Slot& slot = slots_[i];
        if (slot.data && !slot.in_use && !slot.poisoned) {
          slot.in_use = true;
          BeginLeaseLocked();
          lease->data = slot.data;
          lease->index = i;
          return true;
        }
      }

      size_t allocation_index = max_slots_;
      if (!allocation_disabled_ &&
          allocated_slots_ + allocations_in_progress_ < max_slots_) {
        for (size_t i = 0; i < max_slots_; ++i) {
          if (!slots_[i].data && !slots_[i].allocating) {
            slots_[i].allocating = true;
            allocation_index = i;
            ++allocations_in_progress_;
            break;
          }
        }
      }

      if (allocation_index != max_slots_) {
        lock.unlock();
        Allocation allocation = Allocate(cuda);
        lock.lock();
        Slot& slot = slots_[allocation_index];
        slot.allocating = false;

        if (shutting_down_) {
          lock.unlock();
          if (allocation.data) {
            FreeAllocation(cuda, allocation.data,
                           allocation.registered_fallback,
                           "allocation during teardown");
          }
          lock.lock();
          --allocations_in_progress_;
          available_.notify_all();
          return false;
        }

        --allocations_in_progress_;
        if (allocation.data) {
          slot.data = allocation.data;
          slot.cuda = cuda;
          slot.registered_fallback = allocation.registered_fallback;
          slot.in_use = true;
          ++allocated_slots_;
          ++allocation_calls_;
          ++registration_calls_;
          BeginLeaseLocked();
          available_.notify_all();
          lease->data = slot.data;
          lease->index = allocation_index;
          return true;
        }

        allocation_disabled_ = true;
        available_.notify_all();
      }

      bool can_eventually_acquire = allocations_in_progress_ != 0;
      for (size_t i = 0; i < max_slots_ && !can_eventually_acquire; ++i) {
        can_eventually_acquire =
            slots_[i].data && slots_[i].in_use && !slots_[i].poisoned;
      }
      if (!can_eventually_acquire) return false;
      if (!counted_wait) {
        ++wait_count_;
        counted_wait = true;
      }
      available_.wait(lock);
    }
  }

  void Release(size_t index, bool reusable) {
    if (::getpid() != owner_process_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (index >= max_slots_ || !slots_[index].data ||
        !slots_[index].in_use) {
      NotePublisherDriverFailure("pinned bounce lease bookkeeping underflow");
      return;
    }
    Slot& slot = slots_[index];
    slot.in_use = false;
    if (!reusable) slot.poisoned = true;
    --active_leases_;
    available_.notify_all();
  }

  size_t slot_bytes() const { return slot_bytes_; }

  PinnedBouncePoolStats Snapshot() {
    PinnedBouncePoolStats stats;
    stats.slot_bytes = slot_bytes_;
    if (::getpid() != owner_process_) return stats;
    std::lock_guard<std::mutex> lock(mutex_);
    stats.allocated_slots = allocated_slots_;
    stats.allocation_calls = allocation_calls_;
    stats.registration_calls = registration_calls_;
    stats.wait_count = wait_count_;
    stats.active_leases = active_leases_;
    stats.peak_active_leases = peak_active_leases_;
    return stats;
  }

 private:
  struct Slot {
    void* data = nullptr;
    const CudaLib* cuda = nullptr;
    bool registered_fallback = false;
    bool allocating = false;
    bool in_use = false;
    bool poisoned = false;
  };

  struct Allocation {
    void* data = nullptr;
    bool registered_fallback = false;
  };

  Allocation Allocate(const CudaLib* cuda) const {
    void* data = nullptr;
    if (cuda->MemHostAlloc && cuda->MemFreeHost) {
      const CUresult result = cuda->MemHostAlloc(
          &data, slot_bytes_, kCuMemHostAllocPortable);
      if (result == kCudaSuccess && data) return {data, false};
      NotePublisherDriverFailure("cuMemHostAlloc result=" +
                                 std::to_string(result));
      return {};
    }

    const size_t page_size = HostPageSize();
    if (page_size == 0 ||
        ::posix_memalign(&data, page_size, slot_bytes_) != 0 || !data) {
      NotePublisherDriverFailure(
          "aligned bounce-slot backing allocation failed");
      return {};
    }
    const CUresult result = cuda->HostRegister(
        data, slot_bytes_, kCuMemHostRegisterPortable);
    if (result != kCudaSuccess) {
      std::free(data);
      NotePublisherDriverFailure("cuMemHostRegister result=" +
                                 std::to_string(result));
      return {};
    }
    return {data, true};
  }

  static void FreeAllocation(const CudaLib* cuda, void* data,
                             bool registered_fallback,
                             const char* failure_prefix) {
    if (!cuda || !data) return;
    if (!registered_fallback) {
      const CUresult result = cuda->MemFreeHost(data);
      if (result != kCudaSuccess) {
        NotePublisherDriverFailure(
            std::string(failure_prefix) + " cuMemFreeHost result=" +
            std::to_string(result));
      }
      return;
    }
    const CUresult result = cuda->HostUnregister(data);
    if (result == kCudaSuccess) {
      std::free(data);
    } else {
      // Never free storage while the driver may still hold a physical pin.
      NotePublisherDriverFailure(
          std::string(failure_prefix) + " cuMemHostUnregister result=" +
          std::to_string(result));
    }
  }

  void BeginLeaseLocked() {
    ++active_leases_;
    peak_active_leases_ = std::max(peak_active_leases_, active_leases_);
  }

  const pid_t owner_process_;
  size_t slot_bytes_ = 0;
  size_t max_slots_ = 0;
  Slot* slots_ = nullptr;
  std::mutex mutex_;
  std::condition_variable available_;
  size_t allocated_slots_ = 0;
  size_t allocations_in_progress_ = 0;
  uint64_t allocation_calls_ = 0;
  uint64_t registration_calls_ = 0;
  uint64_t wait_count_ = 0;
  size_t active_leases_ = 0;
  size_t peak_active_leases_ = 0;
  bool allocation_disabled_ = false;
  bool shutting_down_ = false;
};

PinnedBouncePool& ProcessBouncePool() {
  static PinnedBouncePool pool;
  return pool;
}

bool PublishThroughBounce(const CudaLib* cuda, PublisherCudaState* state,
                          void* destination, const void* source, size_t size) {
  PinnedBouncePool& pool = ProcessBouncePool();
  size_t copied = 0;
  size_t chunk_index = 0;
  while (copied < size) {
    BounceLease lease;
    if (!pool.Acquire(cuda, &lease)) {
      NotePublisherDriverFailure("pinned bounce-slot acquisition failed");
      return false;
    }
    const size_t chunk = std::min(size - copied, pool.slot_bytes());
    std::memcpy(lease.data, static_cast<const char*>(source) + copied, chunk);
    const CUresult copy_result = cuda->MemcpyAsync(
        reinterpret_cast<CUdeviceptr>(destination) + copied,
        reinterpret_cast<CUdeviceptr>(lease.data), chunk, state->stream);
    if (copy_result != kCudaSuccess) {
      // A failed enqueue does not prove that the driver rejected the transfer
      // before taking ownership of the pinned source. Fence the stream before
      // making the lease reusable; if the fence itself fails, keep the backing
      // permanently quarantined because DMA completion is unknowable.
      const CUresult recovery_sync_result =
          cuda->StreamSynchronize(state->stream);
      pool.Release(lease.index, recovery_sync_result == kCudaSuccess);
      NotePublisherDriverFailure(
          "cuMemcpyAsync result=" + std::to_string(copy_result) +
          " recovery_sync_result=" +
          std::to_string(recovery_sync_result) +
          " copy=" + std::to_string(state->copy_count) +
          " chunk=" + std::to_string(chunk_index) +
          " bytes=" + std::to_string(chunk));
      return false;
    }

    const CUresult sync_result = cuda->StreamSynchronize(state->stream);
    pool.Release(lease.index, sync_result == kCudaSuccess);
    if (sync_result != kCudaSuccess) {
      NotePublisherDriverFailure(
          "chunk cuStreamSynchronize result=" +
          std::to_string(sync_result) +
          " copy=" + std::to_string(state->copy_count) +
          " chunk=" + std::to_string(chunk_index));
      return false;
    }
    copied += chunk;
    ++chunk_index;
  }
  return true;
}

}  // namespace
PinnedBouncePoolStats GetPinnedBouncePoolStatsForTest() {
  return ProcessBouncePool().Snapshot();
}


bool CudaLib::Resolve() {
  void* h = ::dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
  if (!h) return false;
  using F = void*;
  F init = ::dlsym(h, "cuInit");
  MemHostAlloc = reinterpret_cast<CUresult (*)(void**, size_t, unsigned)>(
      ::dlsym(h, "cuMemHostAlloc"));
  MemFreeHost = reinterpret_cast<CUresult (*)(void*)>(
      ::dlsym(h, "cuMemFreeHost"));
  MemAlloc = reinterpret_cast<CUresult (*)(CUdeviceptr*, size_t)>(
      SymV2(h, "cuMemAlloc_v2", "cuMemAlloc"));
  MemFree = reinterpret_cast<CUresult (*)(CUdeviceptr)>(
      SymV2(h, "cuMemFree_v2", "cuMemFree"));
  Memcpy = reinterpret_cast<CUresult (*)(CUdeviceptr, CUdeviceptr, size_t)>(
      ::dlsym(h, "cuMemcpy"));
  MemcpyAsync = reinterpret_cast<CUresult (*)(CUdeviceptr, CUdeviceptr, size_t,
                                              CUstream)>(
      ::dlsym(h, "cuMemcpyAsync"));
  StreamCreate = reinterpret_cast<CUresult (*)(CUstream*, unsigned)>(
      ::dlsym(h, "cuStreamCreate"));
  StreamSynchronize = reinterpret_cast<CUresult (*)(CUstream)>(
      ::dlsym(h, "cuStreamSynchronize"));
  StreamDestroy = reinterpret_cast<CUresult (*)(CUstream)>(
      SymV2(h, "cuStreamDestroy_v2", "cuStreamDestroy"));
  HostRegister = reinterpret_cast<CUresult (*)(void*, size_t, unsigned)>(
      ::dlsym(h, "cuMemHostRegister"));
  HostUnregister = reinterpret_cast<CUresult (*)(void*)>(
      ::dlsym(h, "cuMemHostUnregister"));
  IpcGetMemHandle = reinterpret_cast<CUresult (*)(CUipcMemHandle*, CUdeviceptr)>(
      ::dlsym(h, "cuIpcGetMemHandle"));
  IpcOpenMemHandle = reinterpret_cast<CUresult (*)(CUdeviceptr*, CUipcMemHandle,
                                                   unsigned)>(
      SymV2(h, "cuIpcOpenMemHandle_v2", "cuIpcOpenMemHandle"));
  IpcCloseMemHandle = reinterpret_cast<CUresult (*)(CUdeviceptr)>(
      ::dlsym(h, "cuIpcCloseMemHandle"));
  ctx_get_current_ = reinterpret_cast<CUresult (*)(CUcontext*)>(
      ::dlsym(h, "cuCtxGetCurrent"));
  ctx_set_current_ = reinterpret_cast<CUresult (*)(CUcontext)>(
      ::dlsym(h, "cuCtxSetCurrent"));
  ctx_get_device_ = reinterpret_cast<CUresult (*)(int*)>(
      ::dlsym(h, "cuCtxGetDevice"));
  primary_ctx_retain_ = reinterpret_cast<CUresult (*)(CUcontext*, int)>(
      ::dlsym(h, "cuDevicePrimaryCtxRetain"));
  primary_ctx_release_ = reinterpret_cast<CUresult (*)(int)>(
      SymV2(h, "cuDevicePrimaryCtxRelease_v2",
            "cuDevicePrimaryCtxRelease"));
  pointer_get_attribute_ =
      reinterpret_cast<CUresult (*)(void*, int, CUdeviceptr)>(
          ::dlsym(h, "cuPointerGetAttribute"));
  if (!MemHostAlloc || !MemFreeHost) {
    MemHostAlloc = nullptr;
    MemFreeHost = nullptr;
  }
  if (!HostRegister || !HostUnregister) {
    HostRegister = nullptr;
    HostUnregister = nullptr;
  }
  if (!init || !MemAlloc || !MemFree || !Memcpy || !MemcpyAsync ||
      !StreamCreate || !StreamSynchronize || !StreamDestroy ||
      (!MemHostAlloc && !HostRegister) || !IpcGetMemHandle ||
      !IpcOpenMemHandle || !IpcCloseMemHandle || !ctx_get_current_ ||
      !ctx_set_current_ || !ctx_get_device_ || !primary_ctx_retain_ ||
      !primary_ctx_release_ || !pointer_get_attribute_) {
    ::dlclose(h);
    return false;
  }
  // cuInit is idempotent; the host framework normally beat us to it. A
  // failure here (no device, driver/library mismatch) disables the surface.
  if (reinterpret_cast<CUresult (*)(unsigned)>(init)(0) != kCudaSuccess) {
    ::dlclose(h);
    return false;
  }
  owner_process_ = static_cast<uint64_t>(::getpid());
  return true;
}

const CudaLib* CudaLib::Get() {
  static CudaLib* lib = [] {
    auto* l = new CudaLib();
    if (l->Resolve()) return l;
    delete l;
    return static_cast<CudaLib*>(nullptr);
  }();
  return lib && lib->owner_process_ == static_cast<uint64_t>(::getpid())
             ? lib
             : nullptr;
}

bool CudaLib::IsDevicePtr(const void* p) const {
  unsigned type = 0;
  if (pointer_get_attribute_(&type, kCuPointerAttributeMemoryType,
                             reinterpret_cast<CUdeviceptr>(p)) != kCudaSuccess)
    return false;  // unregistered host memory: attribute query fails
  return type == kCuMemoryTypeDevice;
}

bool CudaLib::GetCurrentCtx(CUcontext* context) const {
  return context && ctx_get_current_(context) == kCudaSuccess;
}

bool CudaLib::SetCurrentCtx(CUcontext context) const {
  return ctx_set_current_(context) == kCudaSuccess;
}

bool CudaLib::RetainPrimaryCtx(int dev, CUcontext* context) const {
  if (dev < 0 || !context) return false;
  *context = nullptr;
  if (primary_ctx_retain_(context, dev) != kCudaSuccess) return false;
  if (*context) return true;
  primary_ctx_release_(dev);
  return false;
}

bool CudaLib::ReleasePrimaryCtx(int dev) const {
  return dev >= 0 && primary_ctx_release_(dev) == kCudaSuccess;
}

bool CudaLib::HasCurrentCtx() const {
  CUcontext c = nullptr;
  return ctx_get_current_(&c) == kCudaSuccess && c != nullptr;
}

int CudaLib::CurrentDevice() const {
  int dev = -1;
  if (ctx_get_device_(&dev) != kCudaSuccess) return -1;
  return dev;
}

int CudaLib::DeviceOf(const void* p) const {
  int dev = -1;
  if (pointer_get_attribute_(&dev, kCuPointerAttributeDeviceOrdinal,
                             reinterpret_cast<CUdeviceptr>(p)) != kCudaSuccess)
    return -1;
  return dev;
}

bool CudaLib::BindPrimaryCtx(int dev) const {
  CUcontext ctx = nullptr;
  if (!RetainPrimaryCtx(dev, &ctx)) return false;
  if (SetCurrentCtx(ctx)) return true;
  ReleasePrimaryCtx(dev);
  return false;
}

DestinationPublisher::~DestinationPublisher() {
  if (!finished_) Finish();
}

bool DestinationPublisher::Copy(
    void* destination, const void* source, size_t size,
    DestinationMemoryKind memory_kind) {
  if (size == 0) return true;
  if (memory_kind == DestinationMemoryKind::kHost) {
    std::memcpy(destination, source, size);
    return true;
  }
  if (finished_ || failed_) return false;

  const CudaLib* cuda = CudaLib::Get();
  if (!cuda || (cuda_ && cuda_ != cuda)) {
    failed_ = true;
    return false;
  }
  cuda_ = cuda;

  const int destination_device = cuda->DeviceOf(destination);
  if (destination_device < 0 ||
      (device_ >= 0 && destination_device != device_)) {
    // One publication owns one stream/context. Fail closed on a second device
    // rather than publishing only a subset or changing completion semantics.
    failed_ = true;
    return false;
  }

  CUcontext prior_context = nullptr;
  if (!cuda->GetCurrentCtx(&prior_context)) {
    failed_ = true;
    return false;
  }

  auto* state = static_cast<PublisherCudaState*>(stream_);
  if (!state) {
    state = new (std::nothrow) PublisherCudaState();
    if (!state) {
      failed_ = true;
      return false;
    }
    state->owner_process = ::getpid();

    if (prior_context && cuda->CurrentDevice() == destination_device) {
      state->context = prior_context;
    } else if (!cuda->RetainPrimaryCtx(destination_device, &state->context)) {
      delete state;
      failed_ = true;
      return false;
    } else {
      state->retained_device = destination_device;
    }

    const bool changed_context = prior_context != state->context;
    if (changed_context && !cuda->SetCurrentCtx(state->context)) {
      // cuCtxSetCurrent is not allowed to strand a successful primary retain.
      cuda->SetCurrentCtx(prior_context);
      if (state->retained_device >= 0)
        cuda->ReleasePrimaryCtx(state->retained_device);
      delete state;
      failed_ = true;
      return false;
    }

    // Keep the stream publication-owned so context and completion ordering
    // remain independent across concurrent callers.
    const CUresult create_result =
        cuda->StreamCreate(&state->stream, kCuStreamNonBlocking);
    if (create_result != kCudaSuccess || !state->stream) {
      if (state->stream) cuda->StreamDestroy(state->stream);
      if (changed_context) cuda->SetCurrentCtx(prior_context);
      if (state->retained_device >= 0)
        cuda->ReleasePrimaryCtx(state->retained_device);
      delete state;
      failed_ = true;
      return false;
    }

    device_ = destination_device;
    stream_ = state;
  } else if (prior_context != state->context &&
             !cuda->SetCurrentCtx(state->context)) {
    if (!cuda->SetCurrentCtx(prior_context)) {
      state->pending_restore = prior_context;
      state->has_pending_restore = true;
    }
    failed_ = true;
    return false;
  }

  ++state->copy_count;
  bool copy_ok =
      PublishThroughBounce(cuda, state, destination, source, size);

  // A publisher never lends its destination context to its caller, not even
  // between batched Copy calls.
  if (prior_context != state->context &&
      !cuda->SetCurrentCtx(prior_context)) {
    state->pending_restore = prior_context;
    state->has_pending_restore = true;
    copy_ok = false;
  }
  if (!copy_ok) failed_ = true;
  return copy_ok;
}

bool DestinationPublisher::Finish() {
  if (finished_) return !failed_;
  finished_ = true;
  const CudaLib* cuda = static_cast<const CudaLib*>(cuda_);
  auto* state = static_cast<PublisherCudaState*>(stream_);
  if (!cuda || !state) return !failed_;

  if (::getpid() != state->owner_process || CudaLib::Get() != cuda) {
    // CUDA state inherited across fork cannot be destroyed safely in the
    // child. Drop only host bookkeeping and fail the publication closed.
    failed_ = true;
    delete state;
    stream_ = nullptr;
    return false;
  }

  CUcontext prior_context = nullptr;
  const bool have_prior = cuda->GetCurrentCtx(&prior_context);
  if (!have_prior) failed_ = true;

  bool changed_context = false;
  if (have_prior && prior_context != state->context) {
    changed_context = cuda->SetCurrentCtx(state->context);
    if (!changed_context) failed_ = true;
  }

  // Every bounce chunk is synchronized before its exclusive lease is
  // released. This final barrier also covers any driver work reported late
  // after an enqueue failure and precedes stream/context teardown.
  const CUresult sync_result = cuda->StreamSynchronize(state->stream);
  if (sync_result != kCudaSuccess) {
    failed_ = true;
    NotePublisherDriverFailure("cuStreamSynchronize result=" +
                               std::to_string(sync_result));
  }
  const CUresult destroy_result = cuda->StreamDestroy(state->stream);
  if (destroy_result != kCudaSuccess)
    NotePublisherDriverFailure("post-sync cuStreamDestroy result=" +
                               std::to_string(destroy_result));

  if (state->has_pending_restore) {
    if (!cuda->SetCurrentCtx(state->pending_restore)) failed_ = true;
  } else if (changed_context && !cuda->SetCurrentCtx(prior_context)) {
    failed_ = true;
  }
  if (state->retained_device >= 0 &&
      !cuda->ReleasePrimaryCtx(state->retained_device))
    NotePublisherDriverFailure(
        "post-sync cuDevicePrimaryCtxRelease device=" +
        std::to_string(state->retained_device));

  delete state;
  stream_ = nullptr;
  return !failed_;
}

}  // namespace dfkv
