#include "client/cuda_ipc.h"
#include "transport/transport.h"

#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <vector>

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

struct RegisteredHostRange {
  uintptr_t begin;
  uintptr_t end;
};

struct ProcessHostRegistration {
  uintptr_t begin;
  uintptr_t end;
  size_t references;
};

struct HostRegistrationRegistry {
  std::mutex mutex;
  std::vector<ProcessHostRegistration> ranges;
};

HostRegistrationRegistry& ProcessHostRegistry() {
  // Publishers can be destroyed during process teardown. Keep the registry
  // alive until process exit rather than depending on static destruction order.
  static HostRegistrationRegistry* registry = new HostRegistrationRegistry();
  return *registry;
}

struct PublisherCudaState {
  CUstream stream = nullptr;
  CUcontext context = nullptr;
  int retained_device = -1;
  size_t copy_count = 0;
  CUcontext pending_restore = nullptr;
  bool has_pending_restore = false;
  std::vector<RegisteredHostRange> leases;
};

void NotePublisherDriverFailure(const std::string& detail);

bool EnqueueRegisteredCopy(const CudaLib* cuda, PublisherCudaState* state,
                           void* destination, const void* source, size_t size) {
  const uintptr_t source_begin = reinterpret_cast<uintptr_t>(source);
  size_t copied = 0;
  size_t chunk_index = 0;
  while (copied < size) {
    const uintptr_t cursor = source_begin + copied;
    const RegisteredHostRange* containing = nullptr;
    for (const auto& range : state->leases) {
      if (range.begin <= cursor && cursor < range.end) {
        containing = &range;
        break;
      }
    }
    if (!containing) {
      NotePublisherDriverFailure(
          "host registration bookkeeping left source uncovered");
      return false;
    }
    const size_t chunk =
        std::min(size - copied,
                 static_cast<size_t>(containing->end - cursor));
    const CUresult result = cuda->MemcpyAsync(
        reinterpret_cast<CUdeviceptr>(destination) + copied,
        reinterpret_cast<CUdeviceptr>(source) + copied, chunk, state->stream);
    if (result != kCudaSuccess) {
      NotePublisherDriverFailure(
          "cuMemcpyAsync result=" + std::to_string(result) +
          " copy=" + std::to_string(state->copy_count) +
          " chunk=" + std::to_string(chunk_index) +
          " bytes=" + std::to_string(chunk));
      return false;
    }
    copied += chunk;
    ++chunk_index;
  }
  return true;
}

size_t HostPageSize() {
  static const size_t page_size = [] {
    const long value = ::sysconf(_SC_PAGESIZE);
    return value > 0 ? static_cast<size_t>(value) : size_t{0};
  }();
  return page_size;
}

void NotePublisherDriverFailure(const std::string& detail) {
  static std::atomic<uint64_t> failures{0};
  const uint64_t count =
      failures.fetch_add(1, std::memory_order_relaxed) + 1;
  if (count == 1 || (count & 0x3ffu) == 0) {
    DFKV_LOG_WARN("cuda destination publication driver failure: " + detail +
                  " failures=" + std::to_string(count));
  }
}

bool ReleaseHostRangesLocked(
    const CudaLib* cuda, HostRegistrationRegistry* registry,
    const std::vector<RegisteredHostRange>& leases, const char* failure_prefix) {
  bool released = true;
  for (auto lease = leases.rbegin(); lease != leases.rend(); ++lease) {
    auto registration = std::find_if(
        registry->ranges.begin(), registry->ranges.end(),
        [&](const ProcessHostRegistration& range) {
          return range.begin == lease->begin && range.end == lease->end;
        });
    if (registration == registry->ranges.end() ||
        registration->references == 0) {
      NotePublisherDriverFailure(
          "host registration reference bookkeeping underflow");
      released = false;
      continue;
    }

    --registration->references;
    if (registration->references != 0) continue;

    const CUresult result =
        cuda->HostUnregister(reinterpret_cast<void*>(registration->begin));
    if (result == kCudaSuccess) {
      registry->ranges.erase(registration);
    } else {
      // Keep a zero-reference entry so a driver failure never leaves an
      // untracked physical pin. A later acquisition can reuse it and its final
      // release will retry the unregister.
      NotePublisherDriverFailure(std::string(failure_prefix) +
                                 " cuMemHostUnregister result=" +
                                 std::to_string(result));
      released = false;
    }
  }
  return released;
}

bool AcquireHostRange(const CudaLib* cuda, PublisherCudaState* state,
                      const void* source, size_t size) {
  const size_t page_size = HostPageSize();
  const uintptr_t address = reinterpret_cast<uintptr_t>(source);
  if (!source || page_size == 0 ||
      size > std::numeric_limits<uintptr_t>::max() - address)
    return false;

  const uintptr_t requested_end = address + size;
  const uintptr_t begin = address - address % page_size;
  uintptr_t end = requested_end;
  const uintptr_t remainder = end % page_size;
  if (remainder != 0) {
    const uintptr_t padding = page_size - remainder;
    if (padding > std::numeric_limits<uintptr_t>::max() - end) return false;
    end += padding;
  }

  HostRegistrationRegistry& registry = ProcessHostRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);

  // Build a complete acquisition plan before changing either the global
  // reference counts or this publisher's leases. The registry is sorted and
  // non-overlapping; existing physical registrations are acquired whole, while
  // uncovered holes become distinct CUDA registrations.
  std::vector<RegisteredHostRange> plan;
  plan.reserve(registry.ranges.size() + 1);
  size_t holes = 0;
  uintptr_t cursor = begin;
  for (const auto& range : registry.ranges) {
    if (range.end <= cursor) continue;
    if (range.begin >= end) break;
    if (cursor < range.begin) {
      plan.push_back({cursor, std::min(end, range.begin)});
      ++holes;
      cursor = std::min(end, range.begin);
    }
    if (cursor >= end) break;
    if (range.end > cursor) {
      plan.push_back({range.begin, range.end});
      cursor = range.end;
    }
  }
  if (cursor < end) {
    plan.push_back({cursor, end});
    ++holes;
  }

  // Reserve before the first CUDA call so bookkeeping allocation cannot fail
  // after a physical registration has succeeded.
  registry.ranges.reserve(registry.ranges.size() + holes);
  state->leases.reserve(state->leases.size() + plan.size());

  std::vector<RegisteredHostRange> acquired;
  acquired.reserve(plan.size());
  for (const auto& piece : plan) {
    auto registration = std::find_if(
        registry.ranges.begin(), registry.ranges.end(),
        [&](const ProcessHostRegistration& range) {
          return range.begin == piece.begin && range.end == piece.end;
        });
    if (registration != registry.ranges.end()) {
      ++registration->references;
      acquired.push_back(piece);
      continue;
    }

    const CUresult result = cuda->HostRegister(
        reinterpret_cast<void*>(piece.begin), piece.end - piece.begin,
        kCuMemHostRegisterPortable);
    if (result != kCudaSuccess) {
      NotePublisherDriverFailure("cuMemHostRegister result=" +
                                 std::to_string(result));
      ReleaseHostRangesLocked(cuda, &registry, acquired,
                              "acquisition rollback");
      return false;
    }

    const auto insertion = std::lower_bound(
        registry.ranges.begin(), registry.ranges.end(), piece.begin,
        [](const ProcessHostRegistration& range, uintptr_t range_begin) {
          return range.begin < range_begin;
        });
    registry.ranges.insert(
        insertion, {piece.begin, piece.end, /*references=*/1});
    acquired.push_back(piece);
  }

  state->leases.insert(state->leases.end(), acquired.begin(),
                       acquired.end());
  return true;
}

}  // namespace

bool CudaLib::Resolve() {
  void* h = ::dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
  if (!h) return false;
  using F = void*;
  F init = ::dlsym(h, "cuInit");
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
  if (!init || !MemAlloc || !MemFree || !Memcpy || !MemcpyAsync ||
      !StreamCreate || !StreamSynchronize || !StreamDestroy ||
      !HostRegister || !HostUnregister || !IpcGetMemHandle ||
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
  return true;
}

const CudaLib* CudaLib::Get() {
  static CudaLib* lib = [] {
    auto* l = new CudaLib();
    if (l->Resolve()) return l;
    delete l;
    return static_cast<CudaLib*>(nullptr);
  }();
  return lib;
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

  const CudaLib* cuda = static_cast<const CudaLib*>(cuda_);
  if (!cuda) {
    cuda = CudaLib::Get();
    if (!cuda) {
      failed_ = true;
      return false;
    }
    cuda_ = cuda;
  }

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

    // Keep the stream publication-owned: TLS reuse would couple pinned-range
    // lifetime, failure state, and Finish ordering across independent calls.
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

  bool copy_ok = AcquireHostRange(cuda, state, source, size);
  if (copy_ok) {
    ++state->copy_count;
    copy_ok =
        EnqueueRegisteredCopy(cuda, state, destination, source, size);
  }

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

  CUcontext prior_context = nullptr;
  const bool have_prior = cuda->GetCurrentCtx(&prior_context);
  if (!have_prior) failed_ = true;

  bool changed_context = false;
  if (have_prior && prior_context != state->context) {
    changed_context = cuda->SetCurrentCtx(state->context);
    if (!changed_context) failed_ = true;
  }

  // Even after an enqueue/setup failure, synchronization must precede every
  // unregister because earlier copies can still reference operation staging.
  // This synchronization result is the publication barrier: cleanup below
  // cannot undo successfully synchronized destination bytes.
  const CUresult sync_result = cuda->StreamSynchronize(state->stream);
  if (sync_result != kCudaSuccess) {
    failed_ = true;
    NotePublisherDriverFailure("cuStreamSynchronize result=" +
                               std::to_string(sync_result));
  }
  {
    HostRegistrationRegistry& registry = ProcessHostRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    ReleaseHostRangesLocked(cuda, &registry, state->leases, "post-sync");
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
