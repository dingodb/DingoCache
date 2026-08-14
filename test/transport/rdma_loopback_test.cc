// RDMA datapath test over a loopback device (Soft-RoCE / rdma_rxe in CI, or any
// real RDMA NIC). Exercises the native-verbs transport + server + versioned wire
// frame + zero-copy RangeInto + (with depth>1) the pipelined worker pool — the
// code that is otherwise only validated on real 400G hardware. Skips cleanly when
// no RDMA device is present. Built only when DFKV_WITH_RDMA is defined. Run under
// ThreadSanitizer to exercise the worker-pool / QP concurrency.
#include "client/kv_client.h"
#include "client/cuda_ipc.h"
#include "client/node_dedup.h"
#include "client/key_map.h"
#include "cache/kv_node_server.h"
#include "cache/rdma_server.h"
#include "common/config_dump.h"
#include "transport/rdma_transport.h"
#include "transport/rail_select.h"
#include "transport/rdma_topology.h"
#include "transport/rdma_protocol.h"
#include "transport/rdma_verbs.h"
#include "utils/net_util.h"

#include <gtest/gtest.h>
#include <sys/mman.h>  // shm_unlink (node-dedup test)
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cerrno>
#include <algorithm>
#include <cctype>
#include <array>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace dfkv {
class RdmaServerTestPeer {
 public:
  using DiscoverFn = std::function<rdma::RdmaDiscoveryResult(
      const std::vector<std::string>&, rdma::RdmaDiscoveryPolicy)>;
  using InitializeAnchorFn = std::function<std::unique_ptr<rdma::RcEndpoint>(
      const std::string&, const std::vector<std::pair<void*, size_t>>&,
      void*, size_t)>;

  static void SetDiscovery(RdmaServer* server, DiscoverFn discover) {
    server->discover_for_test_ = std::move(discover);
  }
  static void SetInitializeAnchor(RdmaServer* server,
                                  InitializeAnchorFn initialize) {
    server->initialize_anchor_for_test_ = std::move(initialize);
  }
  static size_t AnchorCount(const RdmaServer& server) {
    return server.anchors_.size();
  }
  static size_t RailStatsCount(const RdmaServer& server) {
    return server.rail_stats_.size();
  }
  static size_t RegisteredRailCount(const RdmaServer& server) {
    return server.recv_segment_registered_rails_;
  }
  static uint64_t RegisterWriter(RdmaServer* server) {
    auto writer = std::make_shared<RdmaServer::WriterState>();
    return server->RegisterWriter(writer);
  }
  static bool HasWriter(RdmaServer* server, uint64_t token) {
    std::lock_guard<std::mutex> lock(server->writer_mu_);
    return server->writers_.find(token) != server->writers_.end();
  }
  static void ServeBootstrap(RdmaServer* server, int fd) {
    server->Serve(fd);
  }
#ifdef DFKV_WITH_URING
  static void SetUringBackendFactory(
      RdmaServer* server,
      std::function<UringReader::Backend*()> factory) {
    server->uring_backend_factory_for_test_ = std::move(factory);
  }
#endif
};

class RdmaTransportTestPeer {
 public:
  struct SeededFailure {
    uint64_t now_us = 0;
    RemoteRailCompletion completion;
  };

  struct RetryDecision {
    bool retry = false;
    bool cross_rail = false;
    std::vector<uint8_t> excluded;
  };

  static std::optional<SeededFailure> CoolRemoteRail(
      RdmaTransport* transport, const std::string& peer_id, size_t rail) {
    const uint64_t now = rdma::RailPolicy::NowMicros();
    auto lease = transport->remote_rail_health_->TryAcquire(peer_id, rail, now);
    if (!lease) return std::nullopt;
    return SeededFailure{
        now,
        transport->remote_rail_health_->Complete(
            peer_id, rail, lease->generation,
            RemoteRailOutcome::kEndpointFailure, now)};
  }

  static std::vector<uint8_t> RemoteAllowed(
      const RdmaTransport& transport, const std::string& peer_id,
      const std::vector<uint8_t>& candidates, uint64_t now_us) {
    return transport.remote_rail_health_->AllowedMask(peer_id, candidates,
                                                      now_us);
  }

  static size_t DataPoolSize(RdmaTransport* transport,
                             const std::string& node) {
    std::lock_guard<std::mutex> lock(transport->mu_);
    const auto found = transport->pool_.find(node);
    return found == transport->pool_.end() ? 0 : found->second.size();
  }

  static uint64_t PeerPublication(const RdmaTransport& transport,
                                  const std::string& node) {
    return transport.peer_topologies_->Snapshot(node)->publication;
  }

  static RdmaTransport::Conn* AcquireData(RdmaTransport* transport,
                                          const std::string& node) {
    RdmaTransport::AcquireOptions options;
    return transport->Acquire(node, RdmaTransport::Lane::kData, options).conn;
  }

  static void ReleaseData(RdmaTransport* transport, const std::string& node,
                          RdmaTransport::Conn* conn) {
    transport->Release(node, RdmaTransport::Lane::kData, conn);
  }

  static RetryDecision PrepareEndpointRetry(
      RdmaTransport* transport, bool from_pool, bool replay_safe,
      bool request_posted, size_t attempted_rail) {
    auto peer = std::make_shared<rdma::PeerRailSnapshot>();
    peer->complete = true;
    peer->peer_healthy.assign(transport->devs_.size(), 1);
    peer->compatible.assign(transport->devs_.size(), 1);
    RdmaTransport::RailMask excluded(transport->devs_.size(), 0);
    bool cross_rail = false;
    const bool retry = transport->PrepareRetry(
        /*attempt=*/0, from_pool, attempted_rail,
        RdmaTransport::AcquireFailure::kEndpoint,
        replay_safe ? RdmaTransport::ReplaySafety::kReplaySafe
                    : RdmaTransport::ReplaySafety::kUnsafeAfterPost,
        request_posted, peer, &excluded, &cross_rail);
    return RetryDecision{retry, cross_rail, std::move(excluded)};
  }
};

}  // namespace dfkv

namespace {

// Small per-buffer cap so the test stays well under a modest RLIMIT_MEMLOCK
// (RDMA pins registered memory); the test values are a few KB.
constexpr size_t kMaxMsg = 256 * 1024;

class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) {
      had_old_ = true;
      old_ = old;
    }
    if (value)
      ::setenv(name, value, 1);
    else
      ::unsetenv(name);
  }
  ~ScopedEnv() {
    if (had_old_)
      ::setenv(name_.c_str(), old_.c_str(), 1);
    else
      ::unsetenv(name_.c_str());
  }

  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

 private:
  std::string name_;
  std::string old_;
  bool had_old_ = false;
};

std::string SelfHdr() { return "test/model"; }
void ConfigureTestRecvSegment() {
  // Keep Soft-RoCE CI below modest RLIMIT_MEMLOCK. Production defaults to
  // 2 GiB, but these fixtures use 256-KiB blocks and need only a small segment.
  ::setenv("DFKV_RDMA_RECV_SEGMENT_SIZE", "33554432", 0);
}

struct ChildWriterTokenResult {
  uint64_t token = 0;
  int status = -1;
  bool transferred = false;
  bool stale_token_found = false;
};

ChildWriterTokenResult WriterTokenFromFreshProcess(uint64_t stale_token = 0) {
  int pipe_fds[2];
  if (::pipe(pipe_fds) != 0) return {};
  const pid_t child = ::fork();
  if (child < 0) {
    ::close(pipe_fds[0]);
    ::close(pipe_fds[1]);
    return {};
  }
  if (child == 0) {
    ::close(pipe_fds[0]);
    RdmaServer server(RdmaServer::Handler{}, kMaxMsg);
    const uint64_t token = RdmaServerTestPeer::RegisterWriter(&server);
    std::array<char, sizeof(token) + 1> wire{};
    net::PutU64(wire.data(), token);
    wire[sizeof(token)] = static_cast<char>(
        RdmaServerTestPeer::HasWriter(&server, stale_token));
    size_t written = 0;
    while (written < wire.size()) {
      const ssize_t n =
          ::write(pipe_fds[1], wire.data() + written, wire.size() - written);
      if (n > 0) {
        written += static_cast<size_t>(n);
        continue;
      }
      if (n < 0 && errno == EINTR) continue;
      ::_exit(2);
    }
    ::_exit(0);
  }

  ::close(pipe_fds[1]);
  ChildWriterTokenResult result;
  std::array<char, sizeof(result.token) + 1> wire{};
  size_t received = 0;
  while (received < wire.size()) {
    const ssize_t n =
        ::read(pipe_fds[0], wire.data() + received, wire.size() - received);
    if (n > 0) {
      received += static_cast<size_t>(n);
      continue;
    }
    if (n < 0 && errno == EINTR) continue;
    break;
  }
  ::close(pipe_fds[0]);
  pid_t waited;
  do {
    waited = ::waitpid(child, &result.status, 0);
  } while (waited < 0 && errno == EINTR);
  result.token = net::GetU64(wire.data());
  result.stale_token_found = wire[sizeof(result.token)] != 0;
  result.transferred =
      received == wire.size() && waited == child &&
      WIFEXITED(result.status) && WEXITSTATUS(result.status) == 0;
  return result;
}

// A cache node serving RDMA: KvNodeServer owns the DiskCacheGroup; RdmaServer
// bootstraps QPs and routes requests to it (generic handler + zero-copy range).
struct RdmaNode {
  fs::path dir;
  std::unique_ptr<KvNodeServer> srv;
  std::unique_ptr<RdmaServer> rsrv;
  std::string addr;  // bootstrap "ip:port" for the client member list
  mutable std::mutex observation_mu;
  std::map<std::string, size_t> cache_direct_calls;
  std::map<std::string, size_t> range_direct_calls;

  std::function<void(size_t)> before_range;
  explicit RdmaNode(const std::string& tag, size_t max_msg = kMaxMsg) {
    ConfigureTestRecvSegment();
    dir = fs::temp_directory_path() / ("dfkv_rdma_" + tag);
    fs::remove_all(dir);
    fs::create_directories(dir);
    srv = std::make_unique<KvNodeServer>(dir.string(), 1ull << 30);
    EXPECT_EQ(srv->Start(0), Status::kOk);  // TCP listener owns the cache group
    rsrv = std::make_unique<RdmaServer>(
        [this](uint8_t op, const BlockKey& key, uint64_t off, uint64_t len,
               const char* pl, uint64_t pll, std::string* out,
               size_t* value_len) {
          return srv->ProcessRequestForKey(
              op, key, off, len, pl, pll, out, value_len);
        },
        max_msg);
    rsrv->set_range_handler(
        [this](const BlockKey& key, uint64_t off, uint64_t len,
               char* io_buf, size_t cap, const char** out_data,
               size_t* out_len, size_t* value_len) {
          size_t call = 0;
          {
            std::lock_guard<std::mutex> lock(observation_mu);
            call = ++range_direct_calls[key.Filename()];
          }
          if (before_range) before_range(call);
          return srv->RangeDirectForKey(
              key, off, len, io_buf, cap, out_data, out_len, value_len);
        });
    rsrv->set_cache_direct_handler(
        [this](const BlockKey& key, char* data, size_t len, size_t cap) {
          {
            std::lock_guard<std::mutex> lock(observation_mu);
            ++cache_direct_calls[key.Filename()];
          }
          return srv->CacheDirectForKey(key, data, len, cap);
        });
    EXPECT_EQ(rsrv->Start(0), Status::kOk);
    addr = "127.0.0.1:" + std::to_string(rsrv->port());
  }
  ~RdmaNode() {
    if (rsrv) rsrv->Stop();
    if (srv) srv->Stop();
    fs::remove_all(dir);
  }
  size_t CacheDirectCalls(const BlockKey& key) const {
    std::lock_guard<std::mutex> lock(observation_mu);
    const auto found = cache_direct_calls.find(key.Filename());
    return found == cache_direct_calls.end() ? 0 : found->second;
  }
  size_t RangeDirectCalls(const BlockKey& key) const {
    std::lock_guard<std::mutex> lock(observation_mu);
    const auto found = range_direct_calls.find(key.Filename());
    return found == range_direct_calls.end() ? 0 : found->second;
  }
};

bool HaveRdma() { return RdmaTransport::Available(); }

// Last value of a single-line Prometheus counter (skips the # HELP/# TYPE lines
// via rfind, which lands on the value line emitted after them).
long CounterVal(const std::string& text, const std::string& name) {
  auto p = text.rfind(name);
  if (p == std::string::npos) return -1;
  auto sp = text.find(' ', p);
  if (sp == std::string::npos) return -1;
  try { return std::stol(text.substr(sp + 1)); } catch (...) { return -1; }
}
long DeviceCounterVal(const std::string& text, const std::string& name,
                      const std::string& device) {
  const std::string prefix = name + "{dev=\"" + device + "\"} ";
  const auto pos = text.find(prefix);
  if (pos == std::string::npos) return -1;
  try {
    return std::stol(text.substr(pos + prefix.size()));
  } catch (...) {
    return -1;
  }
}
// Rail-policy configuration is process-global. Keep every two-HCA test on one
// suite-safe value: 32 concurrent callers may each reserve the depth-four
// maximum, even if locality initially selects the same rail for all of them.
constexpr char kRealHcaRailCredits[] = "128";

std::vector<std::string> ConfiguredTwoTestRails() {
  const char* configured = std::getenv("DFKV_RDMA_DEV");
  if (!configured || !*configured) return {};
  std::vector<std::string> rails;
  std::string value(configured);
  for (size_t start = 0; start <= value.size();) {
    const size_t comma = value.find(',', start);
    const size_t end = comma == std::string::npos ? value.size() : comma;
    std::string rail = value.substr(start, end - start);
    rail.erase(std::remove_if(rail.begin(), rail.end(),
                              [](unsigned char c) { return std::isspace(c); }),
               rail.end());
    if (!rail.empty()) rails.push_back(std::move(rail));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return rails.size() == 2 ? rails : std::vector<std::string>{};
}
bool HaveConfiguredActiveRails(const std::vector<std::string>& rails) {
  if (rails.size() != 2 || rails[0] == rails[1]) return false;
  const auto discovered = rdma::RdmaTopology::Discover(
      rails, rdma::RdmaDiscoveryPolicy::kActiveOnly);
  return discovered.status == rdma::RdmaDiscoveryStatus::kOk &&
         discovered.devices.size() == rails.size();
}

void ExpectReleasedRailResources(const std::string& before,
                                 const std::string& after,
                                 const std::vector<std::string>& rails) {
  for (const auto& rail : rails) {
    EXPECT_EQ(DeviceCounterVal(after, "dfkv_rdma_client_rail_inflight", rail),
              0)
        << rail;
    EXPECT_EQ(
        DeviceCounterVal(after, "dfkv_rdma_client_rail_credits_available",
                         rail),
        DeviceCounterVal(before, "dfkv_rdma_client_rail_credits_available",
                         rail))
        << rail;
  }
  EXPECT_EQ(CounterVal(after, "dfkv_rdma_client_transient_user_mr_active"),
            CounterVal(before,
                       "dfkv_rdma_client_transient_user_mr_active"));
}
void ExpectTwoDistinctRailAttempts(
    const std::string& before, const std::string& after,
    const std::vector<std::string>& rails, long expected_local_failures) {
  ASSERT_EQ(rails.size(), 2u);
  long selections = 0;
  long errors = 0;
  long endpoint_errors = 0;
  for (const auto& rail : rails) {
    const long selected =
        DeviceCounterVal(after, "dfkv_rdma_client_rail_selections_total",
                         rail) -
        DeviceCounterVal(before, "dfkv_rdma_client_rail_selections_total",
                         rail);
    EXPECT_EQ(selected, 1) << rail;
    selections += selected;
    errors += DeviceCounterVal(after, "dfkv_rdma_client_rail_errors_total",
                               rail) -
              DeviceCounterVal(before, "dfkv_rdma_client_rail_errors_total",
                               rail);
    endpoint_errors +=
        DeviceCounterVal(after, "dfkv_rdma_client_endpoint_errors_total",
                         rail) -
        DeviceCounterVal(before, "dfkv_rdma_client_endpoint_errors_total",
                         rail);
  }
  EXPECT_EQ(selections, 2);
  EXPECT_EQ(errors, expected_local_failures);
  EXPECT_EQ(endpoint_errors, 0);
}

struct FakePoolRail {
  size_t declared = 0;
  size_t staged = 0;
  bool reject_stage = false;
  size_t commits = 0;
  size_t rollbacks = 0;

  bool Stage(size_t requested) {
    if (reject_stage) return false;
    staged = requested;
    return true;
  }
  void Commit() {
    declared = staged;
    ++commits;
  }
  void Rollback() {
    staged = declared;
    ++rollbacks;
  }
  bool Covers(size_t offset, size_t length) const {
    return offset <= declared && length <= declared - offset;
  }
};

std::string CudaTestBytes(size_t size, uint8_t seed) {
  std::string bytes(size, '\0');
  for (size_t i = 0; i < size; ++i) {
    bytes[i] = static_cast<char>(
        (static_cast<unsigned>(seed) + 131u * i + 17u * (i >> 3)) & 0xffu);
  }
  return bytes;
}

// CUDA allocation with sentinels on both sides of the caller-visible range.
// Upload/download use the dlopen'd driver surface, so this test binary needs
// neither CUDA headers nor a link-time dependency on libcuda.
class CudaGuardedBuffer {
 public:
  static constexpr size_t kGuardBytes = 37;
  static constexpr uint8_t kGuardByte = 0xa7;

  CudaGuardedBuffer(const CudaLib* cuda, size_t payload_size)
      : cuda_(cuda), payload_size_(payload_size) {}
  ~CudaGuardedBuffer() {
    if (base_) cuda_->MemFree(base_);
  }

  bool Allocate() {
    if (cuda_->MemAlloc(&base_, payload_size_ + 2 * kGuardBytes) !=
        kCudaSuccess) {
      base_ = 0;
      return false;
    }
    const std::string initial(payload_size_ + 2 * kGuardBytes,
                              static_cast<char>(kGuardByte));
    return cuda_->Memcpy(base_,
                         reinterpret_cast<CUdeviceptr>(initial.data()),
                         initial.size()) == kCudaSuccess;
  }

  void* payload() const {
    return reinterpret_cast<void*>(base_ + kGuardBytes);
  }

  bool ReadPayload(std::string* out) const {
    out->assign(payload_size_, '\0');
    return cuda_->Memcpy(reinterpret_cast<CUdeviceptr>(out->data()),
                         base_ + kGuardBytes, payload_size_) == kCudaSuccess;
  }

  bool GuardsIntact() const {
    std::string allocation(payload_size_ + 2 * kGuardBytes, '\0');
    if (cuda_->Memcpy(reinterpret_cast<CUdeviceptr>(allocation.data()), base_,
                      allocation.size()) != kCudaSuccess) {
      return false;
    }
    const std::string guard(kGuardBytes, static_cast<char>(kGuardByte));
    return allocation.compare(0, kGuardBytes, guard) == 0 &&
           allocation.compare(kGuardBytes + payload_size_, kGuardBytes,
                              guard) == 0;
  }

 private:
  const CudaLib* cuda_;
  size_t payload_size_;
  CUdeviceptr base_ = 0;
};

const CudaLib* ActiveCudaForTest() {
  const CudaLib* cuda = CudaLib::Get();
  if (!cuda || !cuda->BindPrimaryCtx(0)) return nullptr;
  return cuda;
}

using MemcpyAsyncFn =
    CUresult (*)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
using StreamSynchronizeFn = CUresult (*)(CUstream);

MemcpyAsyncFn cuda_memcpy_async_real = nullptr;
StreamSynchronizeFn cuda_stream_synchronize_real = nullptr;
std::atomic<bool> fail_next_cuda_memcpy_async{false};
std::atomic<bool> fail_next_cuda_stream_synchronize{false};
std::atomic<uint64_t> injected_cuda_memcpy_async_calls{0};
std::atomic<uint64_t> injected_cuda_stream_synchronize_calls{0};
std::atomic<CUdeviceptr> last_cuda_memcpy_async_source{0};

CUresult InjectCudaMemcpyAsync(CUdeviceptr destination, CUdeviceptr source,
                               size_t size, CUstream stream) {
  last_cuda_memcpy_async_source.store(source, std::memory_order_relaxed);
  injected_cuda_memcpy_async_calls.fetch_add(1, std::memory_order_relaxed);
  if (fail_next_cuda_memcpy_async.exchange(false,
                                           std::memory_order_relaxed)) {
    return 999;
  }
  return cuda_memcpy_async_real(destination, source, size, stream);
}

CUresult InjectCudaStreamSynchronize(CUstream stream) {
  injected_cuda_stream_synchronize_calls.fetch_add(
      1, std::memory_order_relaxed);
  if (fail_next_cuda_stream_synchronize.exchange(
          false, std::memory_order_relaxed)) {
    return 999;
  }
  return cuda_stream_synchronize_real(stream);
}

class ScopedCudaAsyncFault {
 public:
  explicit ScopedCudaAsyncFault(const CudaLib* cuda)
      : cuda_(const_cast<CudaLib*>(cuda)),
        memcpy_async_(cuda_->MemcpyAsync),
        stream_synchronize_(cuda_->StreamSynchronize) {
    cuda_memcpy_async_real = memcpy_async_;
    cuda_stream_synchronize_real = stream_synchronize_;
    fail_next_cuda_memcpy_async.store(false, std::memory_order_relaxed);
    fail_next_cuda_stream_synchronize.store(false,
                                            std::memory_order_relaxed);
    injected_cuda_memcpy_async_calls.store(0, std::memory_order_relaxed);
    injected_cuda_stream_synchronize_calls.store(0,
                                                  std::memory_order_relaxed);
    last_cuda_memcpy_async_source.store(0, std::memory_order_relaxed);
    cuda_->MemcpyAsync = InjectCudaMemcpyAsync;
    cuda_->StreamSynchronize = InjectCudaStreamSynchronize;
  }

  ~ScopedCudaAsyncFault() {
    cuda_->MemcpyAsync = memcpy_async_;
    cuda_->StreamSynchronize = stream_synchronize_;
    fail_next_cuda_memcpy_async.store(false, std::memory_order_relaxed);
    fail_next_cuda_stream_synchronize.store(false,
                                            std::memory_order_relaxed);
    cuda_memcpy_async_real = nullptr;
    cuda_stream_synchronize_real = nullptr;
  }

  void FailNextEnqueueAndSynchronize() {
    fail_next_cuda_memcpy_async.store(true, std::memory_order_relaxed);
    fail_next_cuda_stream_synchronize.store(true,
                                            std::memory_order_relaxed);
  }

  uint64_t memcpy_async_calls() const {
    return injected_cuda_memcpy_async_calls.load(std::memory_order_relaxed);
  }

  uint64_t stream_synchronize_calls() const {
    return injected_cuda_stream_synchronize_calls.load(
        std::memory_order_relaxed);
  }

  CUdeviceptr last_source() const {
    return last_cuda_memcpy_async_source.load(std::memory_order_relaxed);
  }

 private:
  CudaLib* cuda_;
  MemcpyAsyncFn memcpy_async_;
  StreamSynchronizeFn stream_synchronize_;
};

class ScopedCudaContextRestore {
 public:
  ScopedCudaContextRestore(const CudaLib* cuda, CUcontext context)
      : cuda_(cuda), context_(context) {}
  ~ScopedCudaContextRestore() { cuda_->SetCurrentCtx(context_); }

 private:
  const CudaLib* cuda_;
  CUcontext context_;
};

}  // namespace

TEST(RdmaWriterToken, LiveWritersHaveUniqueNonzeroTokens) {
  RdmaServer server(RdmaServer::Handler{}, kMaxMsg);
  std::unordered_set<uint64_t> tokens;
  constexpr size_t kWriterCount = 1024;
  for (size_t i = 0; i < kWriterCount; ++i) {
    const uint64_t token = RdmaServerTestPeer::RegisterWriter(&server);
    EXPECT_NE(token, 0u);
    EXPECT_TRUE(tokens.insert(token).second);
  }
}

TEST(RdmaWriterToken, FreshProcessesDoNotReuseStaleTokens) {
  const ChildWriterTokenResult old_process = WriterTokenFromFreshProcess();
  ASSERT_TRUE(old_process.transferred) << "child status=" << old_process.status;
  ASSERT_NE(old_process.token, 0u);

  const ChildWriterTokenResult new_process =
      WriterTokenFromFreshProcess(old_process.token);
  ASSERT_TRUE(new_process.transferred) << "child status=" << new_process.status;
  ASSERT_NE(new_process.token, 0u);
  EXPECT_FALSE(new_process.stale_token_found);
  EXPECT_NE(new_process.token, old_process.token);
}

TEST(RdmaBootstrapCapability, ServerAdvertisesWriterRetirementInProbe) {
  RdmaServer server(RdmaServer::Handler{}, kMaxMsg);
  int sockets[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  std::thread serving(
      [&] { RdmaServerTestPeer::ServeBootstrap(&server, sockets[1]); });

  char request[rdma::kDevNameBytes];
  rdma::EncodeDevFrame(rdma::kV2ProbeDevice, 4u << 20, request);
  const bool wrote = net::WriteAll(sockets[0], request, sizeof(request));
  char reply[rdma::kV2ProbeReplyBytes];
  const bool read = wrote && net::ReadAll(sockets[0], reply, sizeof(reply));

  ::close(sockets[0]);
  serving.join();
  ASSERT_TRUE(read);
  EXPECT_TRUE(rdma::V2ProbeSupportsWriterRetirement(reply));
}

TEST(RdmaBootstrapCapability, UnnegotiatedTokenGetsNoRetirementProof) {
  RdmaServer server(RdmaServer::Handler{}, kMaxMsg);
  int sockets[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  std::thread serving(
      [&] { RdmaServerTestPeer::ServeBootstrap(&server, sockets[1]); });

  char request[rdma::kDevNameBytes];
  rdma::EncodeDevFrame(rdma::kV2RetireWriterDevice, 0x12345678, request);
  const bool wrote = net::WriteAll(sockets[0], request, sizeof(request));
  char proof[rdma::kV2RetireProofBytes];
  const bool read =
      wrote && net::ReadAll(sockets[0], proof, sizeof(proof));

  ::close(sockets[0]);
  serving.join();
  ASSERT_TRUE(wrote);
  EXPECT_FALSE(read);
}

TEST(RdmaServerStartup, AutoModeKeepsFirstActiveDeviceSemantics) {
  ScopedEnv configured_device("DFKV_RDMA_DEV", nullptr);
  ScopedEnv recv_segment("DFKV_RDMA_RECV_SEGMENT_SIZE", "1048576");
  RdmaServer server(
      [](uint8_t, const BlockKey&, uint64_t, uint64_t, const char*, uint64_t,
         std::string*, size_t*) { return Status::kInvalid; },
      kMaxMsg);
  RdmaServerTestPeer::SetDiscovery(
      &server,
      [](const std::vector<std::string>& requested,
         rdma::RdmaDiscoveryPolicy policy) {
        EXPECT_TRUE(requested.empty());
        EXPECT_EQ(policy, rdma::RdmaDiscoveryPolicy::kActiveOnly);
        rdma::RdmaDevInfo first;
        first.name = "mlx5_first";
        first.active = true;
        rdma::RdmaDevInfo second;
        second.name = "mlx5_second";
        second.active = true;
        rdma::RdmaDiscoveryResult result;
        result.devices = {first, second};
        return result;
      });
  std::vector<std::string> initialized;
  RdmaServerTestPeer::SetInitializeAnchor(
      &server,
      [&initialized](
          const std::string& device,
          const std::vector<std::pair<void*, size_t>>&, void*, size_t) {
        initialized.push_back(device);
        return std::make_unique<rdma::RcEndpoint>();
      });

  ASSERT_EQ(server.Start(0), Status::kOk);
  EXPECT_EQ(server.DeviceNames(),
            (std::vector<std::string>{"mlx5_first"}));
  EXPECT_EQ(initialized, server.DeviceNames());
  EXPECT_EQ(RdmaServerTestPeer::AnchorCount(server), 1u);
  EXPECT_EQ(RdmaServerTestPeer::RailStatsCount(server), 1u);
  EXPECT_EQ(RdmaServerTestPeer::RegisteredRailCount(server), 1u);
  server.Stop();
}

TEST(RdmaServerStartup, ExplicitInactiveRailRemainsInFixedTopology) {
  ScopedEnv recv_segment("DFKV_RDMA_RECV_SEGMENT_SIZE", "1048576");
  RdmaServer server(
      [](uint8_t, const BlockKey&, uint64_t, uint64_t, const char*, uint64_t,
         std::string*, size_t*) { return Status::kInvalid; },
      kMaxMsg, "mlx5_active,mlx5_down,mlx5_active");
  RdmaServerTestPeer::SetDiscovery(
      &server,
      [](const std::vector<std::string>& requested,
         rdma::RdmaDiscoveryPolicy policy) {
        EXPECT_EQ(requested,
                  (std::vector<std::string>{"mlx5_active", "mlx5_down"}));
        EXPECT_EQ(policy, rdma::RdmaDiscoveryPolicy::kAllowInactive);
        rdma::RdmaDevInfo active;
        active.name = "mlx5_active";
        active.active = true;
        rdma::RdmaDevInfo inactive;
        inactive.name = "mlx5_down";
        inactive.active = false;
        rdma::RdmaDiscoveryResult result;
        result.devices = {active, inactive};
        return result;
      });
  std::vector<std::string> initialized;
  RdmaServerTestPeer::SetInitializeAnchor(
      &server,
      [&initialized](
          const std::string& device,
          const std::vector<std::pair<void*, size_t>>&, void*, size_t) {
        initialized.push_back(device);
        return std::make_unique<rdma::RcEndpoint>();
      });

  ASSERT_EQ(server.Start(0), Status::kOk);
  EXPECT_EQ(server.DeviceNames(),
            (std::vector<std::string>{"mlx5_active", "mlx5_down"}));
  EXPECT_EQ(initialized, server.DeviceNames());
  EXPECT_EQ(RdmaServerTestPeer::AnchorCount(server), 2u);
  EXPECT_EQ(RdmaServerTestPeer::RailStatsCount(server), 2u);
  EXPECT_EQ(RdmaServerTestPeer::RegisteredRailCount(server), 2u);
  server.Stop();
}

TEST(RdmaServerStartup, PartialAnchorMaterializationFailsAtomically) {
  ScopedEnv recv_segment("DFKV_RDMA_RECV_SEGMENT_SIZE", "1048576");
  RdmaServer server(
      [](uint8_t, const BlockKey&, uint64_t, uint64_t, const char*, uint64_t,
         std::string*, size_t*) { return Status::kInvalid; },
      kMaxMsg, "mlx5_0,mlx5_1");
  RdmaServerTestPeer::SetDiscovery(
      &server,
      [](const std::vector<std::string>&,
         rdma::RdmaDiscoveryPolicy policy) {
        EXPECT_EQ(policy, rdma::RdmaDiscoveryPolicy::kAllowInactive);
        rdma::RdmaDevInfo first;
        first.name = "mlx5_0";
        first.active = true;
        rdma::RdmaDevInfo second;
        second.name = "mlx5_1";
        second.active = false;
        rdma::RdmaDiscoveryResult result;
        result.devices = {first, second};
        return result;
      });
  size_t attempts = 0;
  RdmaServerTestPeer::SetInitializeAnchor(
      &server,
      [&attempts](
          const std::string&, const std::vector<std::pair<void*, size_t>>&,
          void*, size_t) -> std::unique_ptr<rdma::RcEndpoint> {
        if (++attempts == 2) return nullptr;
        return std::make_unique<rdma::RcEndpoint>();
      });

  EXPECT_EQ(server.Start(0), Status::kIOError);
  EXPECT_EQ(attempts, 2u);
  EXPECT_EQ(server.DeviceNames(),
            (std::vector<std::string>{"mlx5_0", "mlx5_1"}));
  EXPECT_EQ(RdmaServerTestPeer::AnchorCount(server), 0u);
  EXPECT_EQ(RdmaServerTestPeer::RailStatsCount(server), 0u);
  EXPECT_EQ(RdmaServerTestPeer::RegisteredRailCount(server), 0u);
}

TEST(RdmaServerStartup, ExplicitResolutionFailureNeverShrinksTopology) {
  RdmaServer server(
      [](uint8_t, const BlockKey&, uint64_t, uint64_t, const char*, uint64_t,
         std::string*, size_t*) { return Status::kInvalid; },
      kMaxMsg, "mlx5_present,mlx5_missing");
  RdmaServerTestPeer::SetDiscovery(
      &server,
      [](const std::vector<std::string>& requested,
         rdma::RdmaDiscoveryPolicy policy) {
        EXPECT_EQ(requested,
                  (std::vector<std::string>{"mlx5_present", "mlx5_missing"}));
        EXPECT_EQ(policy, rdma::RdmaDiscoveryPolicy::kAllowInactive);
        rdma::RdmaDiscoveryResult result;
        result.status =
            rdma::RdmaDiscoveryStatus::kConfiguredDeviceMissing;
        result.failed_device = "mlx5_missing";
        rdma::RdmaDevInfo observed;
        observed.name = "mlx5_present";
        observed.active = true;
        result.observed_devices = {observed};
        return result;
      });
  size_t anchor_attempts = 0;
  RdmaServerTestPeer::SetInitializeAnchor(
      &server,
      [&anchor_attempts](
          const std::string&, const std::vector<std::pair<void*, size_t>>&,
          void*, size_t) -> std::unique_ptr<rdma::RcEndpoint> {
        ++anchor_attempts;
        return std::make_unique<rdma::RcEndpoint>();
      });

  testing::internal::CaptureStderr();
  EXPECT_EQ(server.Start(0), Status::kIOError);
  const std::string logs = testing::internal::GetCapturedStderr();
  EXPECT_NE(logs.find("configured=2 initialized=0 probed=1 "
                      "observed_ACTIVE=1 observed_inactive=(none) "
                      "unresolved=mlx5_missing"),
            std::string::npos)
      << logs;
  EXPECT_EQ(logs.find(" ACTIVE=0 inactive=(none)"), std::string::npos)
      << logs;
  EXPECT_EQ(anchor_attempts, 0u);
  EXPECT_EQ(server.DeviceNames(),
            (std::vector<std::string>{"mlx5_present", "mlx5_missing"}));
  EXPECT_EQ(RdmaServerTestPeer::AnchorCount(server), 0u);
  EXPECT_EQ(RdmaServerTestPeer::RailStatsCount(server), 0u);
  EXPECT_EQ(RdmaServerTestPeer::RegisteredRailCount(server), 0u);
}

TEST(RdmaSafety, LocalRailFailureQuarantinesButPeerFailureDoesNot) {
  rdma::RailPolicyConfig config;
  config.credits_per_rail = 2;
  config.error_threshold = 1;
  config.quarantine_us = 1000;
  config.latency_weight = 1;
  config.error_penalty_us = 100;
  rdma::RailPolicy policy(2, config);

  auto local = policy.TryAcquire(2, 100);
  ASSERT_TRUE(local);
  ASSERT_EQ(local->rail, 0u);
  policy.Complete(*local, 10, rdma::RailCompletion::kRailFailure, 200);
  auto stats = policy.Snapshot(200);
  ASSERT_EQ(stats.size(), 2u);
  EXPECT_EQ(stats[0].errors, 1u);
  EXPECT_EQ(stats[0].endpoint_errors, 0u);
  EXPECT_EQ(stats[0].consecutive_errors, 1u);
  EXPECT_EQ(stats[0].quarantines, 1u);
  EXPECT_TRUE(stats[0].quarantined);
  EXPECT_EQ(stats[0].inflight, 0u);
  EXPECT_EQ(stats[0].credits, 2u);

  auto endpoint = policy.TryAcquire(2, 200);
  ASSERT_TRUE(endpoint);
  ASSERT_EQ(endpoint->rail, 1u);
  policy.Complete(*endpoint, 10, rdma::RailCompletion::kEndpointFailure, 210);
  stats = policy.Snapshot(210);
  EXPECT_EQ(stats[1].errors, 0u);
  EXPECT_EQ(stats[1].endpoint_errors, 1u);
  EXPECT_EQ(stats[1].consecutive_errors, 0u);
  EXPECT_EQ(stats[1].quarantines, 0u);
  EXPECT_FALSE(stats[1].quarantined);
  EXPECT_EQ(stats[1].inflight, 0u);
  EXPECT_EQ(stats[1].credits, 2u);
}

TEST(RdmaSafety, OperationExclusionAppliesToPreferredAndFallbackRails) {
  rdma::RailPolicy policy(
      3, rdma::RailPolicyConfig{/*credits_per_rail=*/2,
                                /*error_threshold=*/3,
                                /*quarantine_us=*/1000,
                                /*latency_weight=*/1,
                                /*error_penalty_us=*/100});
  const std::vector<uint8_t> preferred{1, 1, 0};
  const std::vector<uint8_t> fallback{0, 0, 1};

  auto preferred_retry = rdma::AcquireWithFallback(
      policy, 1, 0, preferred, fallback, /*excluded=*/{1, 0, 0});
  ASSERT_TRUE(preferred_retry);
  EXPECT_EQ(preferred_retry->rail, 1u);
  policy.Complete(*preferred_retry, 1, rdma::RailCompletion::kSuccess, 100);

  auto fallback_retry = rdma::AcquireWithFallback(
      policy, 1, 0, preferred, fallback, /*excluded=*/{1, 1, 0});
  ASSERT_TRUE(fallback_retry);
  EXPECT_EQ(fallback_retry->rail, 2u);
  policy.Complete(*fallback_retry, 1, rdma::RailCompletion::kSuccess, 101);

  EXPECT_TRUE(rdma::HasUnexcludedRail(preferred, fallback, {1, 1, 0}));
  EXPECT_FALSE(rdma::HasUnexcludedRail({1}, {}, {1}));
  EXPECT_FALSE(
      rdma::AcquireWithFallback(policy, 1, 0, {1, 0, 0}, {}, {1, 0, 0}))
      << "admission must not silently bypass an operation-local exclusion";
}

TEST(RdmaPeerRails, NineRailGpuPeerUsesOnlyPrimaryTier) {
  const std::vector<std::string> devices{
      "gpu0", "gpu1", "gpu2", "gpu3", "gpu4",
      "gpu5", "gpu6", "gpu7", "cpu0"};
  std::vector<std::vector<uint8_t>> tiers;
  std::string error;
  ASSERT_TRUE(rdma::ParseRailTiers(
      "gpu0|gpu1|gpu2|gpu3|gpu4|gpu5|gpu6|gpu7;cpu0", devices,
      &tiers, &error))
      << error;
  ASSERT_EQ(tiers.size(), 2u);
  EXPECT_EQ(tiers[0],
            (std::vector<uint8_t>{1, 1, 1, 1, 1, 1, 1, 1, 0}));
  EXPECT_EQ(tiers[1],
            (std::vector<uint8_t>{0, 0, 0, 0, 0, 0, 0, 0, 1}));

  rdma::PeerTopologyStore store(devices, tiers, /*require_complete=*/true);
  PeerTopology topology{
      "gpu-peer", 41, true,
      {{"gpu0", true}, {"gpu1", true}, {"gpu2", true},
       {"gpu3", true}, {"gpu4", true}, {"gpu5", true},
       {"gpu6", true}, {"gpu7", true}, {"cpu0", true}}};
  ASSERT_TRUE(store.Update(topology));
  const auto snapshot = store.Snapshot("gpu-peer");
  ASSERT_TRUE(snapshot->complete);
  EXPECT_EQ(snapshot->generation, 41u);
  EXPECT_EQ(snapshot->compatible,
            (std::vector<uint8_t>{1, 1, 1, 1, 1, 1, 1, 1, 0}));

  rdma::RailPolicy policy(9);
  auto lease = rdma::AcquireHighestTier(
      policy, 1, 0, snapshot->compatible, snapshot->compatible, {});
  ASSERT_TRUE(lease);
  EXPECT_LT(lease->rail, 8u);
  policy.Complete(*lease, 1, rdma::RailCompletion::kSuccess, 101);
  EXPECT_EQ(policy.Snapshot(101)[8].selections, 0u);
}

TEST(RdmaPeerRails, CpuPeerUsesItsFirstSharedRailWithoutRejection) {
  const std::vector<std::string> devices{
      "gpu0", "gpu1", "gpu2", "gpu3", "gpu4",
      "gpu5", "gpu6", "gpu7", "cpu0"};
  std::vector<std::vector<uint8_t>> tiers;
  ASSERT_TRUE(rdma::ParseRailTiers(
      "gpu0|gpu1|gpu2|gpu3|gpu4|gpu5|gpu6|gpu7;cpu0", devices,
      &tiers, nullptr));
  rdma::PeerTopologyStore store(devices, tiers, /*require_complete=*/true);
  ASSERT_TRUE(store.Update(
      PeerTopology{"cpu-peer", 7, true, {{"cpu0", true}}}));
  const auto snapshot = store.Snapshot("cpu-peer");
  ASSERT_EQ(snapshot->compatible,
            (std::vector<uint8_t>{0, 0, 0, 0, 0, 0, 0, 0, 1}));

  rdma::RailPolicy policy(9);
  auto lease = rdma::AcquireHighestTier(
      policy, 1, 0, snapshot->compatible,
      /*preferred=*/{1, 0, 0, 0, 0, 0, 0, 0, 0},
      /*fallback=*/{1, 1, 1, 1, 1, 1, 1, 1, 1});
  ASSERT_TRUE(lease);
  EXPECT_EQ(lease->rail, 8u);
  policy.Complete(*lease, 1, rdma::RailCompletion::kSuccess, 101);
  const auto stats = policy.Snapshot(101);
  EXPECT_EQ(stats[8].selections, 1u);
  EXPECT_EQ(stats[8].errors, 0u);
  EXPECT_EQ(stats[8].endpoint_errors, 0u);
}

TEST(RdmaPeerRails, PrimaryFailureUses200GFallbackAndRecoveryIsStable) {
  const std::vector<std::string> devices{"gpu400_0", "gpu400_1", "ib200"};
  std::vector<std::vector<uint8_t>> tiers;
  ASSERT_TRUE(rdma::ParseRailTiers("gpu400_0|gpu400_1;ib200", devices, &tiers,
                                   nullptr));
  rdma::PeerTopologyStore store(devices, tiers, /*require_complete=*/true);

  ASSERT_TRUE(store.Update(PeerTopology{
      "peer", 10, true,
      {{"ib200", true}, {"gpu400_1", true}, {"gpu400_0", false}}}));
  const auto partial = store.Snapshot("peer");
  EXPECT_EQ(partial->compatible, (std::vector<uint8_t>{0, 1, 0}));

  ASSERT_TRUE(store.Update(PeerTopology{
      "peer", 11, true,
      {{"gpu400_0", false}, {"gpu400_1", false}, {"ib200", true}}}));
  const auto fallback = store.Snapshot("peer");
  EXPECT_EQ(fallback->compatible, (std::vector<uint8_t>{0, 0, 1}));
  EXPECT_EQ(partial->compatible, (std::vector<uint8_t>{0, 1, 0}))
      << "published snapshots must remain immutable";

  ASSERT_TRUE(store.Update(PeerTopology{
      "peer", 12, true,
      {{"gpu400_0", true}, {"gpu400_1", false}, {"ib200", true}}}));
  const auto recovered = store.Snapshot("peer");
  EXPECT_EQ(recovered->compatible, (std::vector<uint8_t>{1, 0, 0}));
  EXPECT_FALSE(store.Update(PeerTopology{
      "peer", 12, true,
      {{"gpu400_0", false}, {"gpu400_1", false}, {"ib200", true}}}))
      << "duplicate generations must not perturb stable recovery";
  EXPECT_EQ(store.Snapshot("peer")->compatible,
            (std::vector<uint8_t>{1, 0, 0}));
}

TEST(RdmaPeerRails, NoIntersectionIsPeerNeutralAndLocallyObservable) {
  const std::vector<std::string> devices{"gpu0", "cpu0"};
  std::vector<std::vector<uint8_t>> tiers;
  ASSERT_TRUE(
      rdma::ParseRailTiers("gpu0;cpu0", devices, &tiers, nullptr));
  rdma::PeerTopologyStore store(devices, tiers, /*require_complete=*/true);
  ASSERT_TRUE(store.Update(
      PeerTopology{"peer", 3, true, {{"remote-only", true}}}));
  EXPECT_TRUE(store.Snapshot("peer")->compatible.empty());
  EXPECT_STREQ(StatusName(Status::kNoCompatibleRail), "NoCompatibleRail");

  rdma::RailPolicy policy(2, rdma::RailPolicyConfig{
                                 /*credits_per_rail=*/1,
                                 /*error_threshold=*/1,
                                 /*quarantine_us=*/1000,
                                 /*latency_weight=*/1,
                                 /*error_penalty_us=*/100});
  const auto stats = policy.Snapshot(100);
  ASSERT_EQ(stats.size(), 2u);
  for (const auto& rail : stats) {
    EXPECT_EQ(rail.selections, 0u);
    EXPECT_EQ(rail.errors, 0u);
    EXPECT_EQ(rail.endpoint_errors, 0u);
    EXPECT_EQ(rail.quarantines, 0u);
    EXPECT_FALSE(rail.quarantined);
  }
}

TEST(RdmaPeerRails, OperationPublicationFencesOlderSnapshots) {
  const std::vector<std::string> devices{"gpu0", "cpu0"};
  std::vector<std::vector<uint8_t>> tiers;
  ASSERT_TRUE(
      rdma::ParseRailTiers("gpu0;cpu0", devices, &tiers, nullptr));
  rdma::PeerTopologyStore store(devices, tiers, /*require_complete=*/true);
  ASSERT_TRUE(store.Update(
      PeerTopology{"peer", 100, true, {{"gpu0", true}}}));
  const auto operation = store.Snapshot("peer");
  ASSERT_EQ(operation->generation, 100u);
  ASSERT_TRUE(store.IsCurrent("peer", operation->publication));

  ASSERT_TRUE(store.Update(
      PeerTopology{"peer", 101, true, {{"cpu0", true}}}));
  EXPECT_FALSE(store.IsCurrent("peer", operation->publication));
  EXPECT_EQ(operation->generation, 100u);
  EXPECT_EQ(operation->compatible, (std::vector<uint8_t>{1, 0}));
  const auto current = store.Snapshot("peer");
  EXPECT_EQ(current->generation, 101u);
  EXPECT_EQ(current->compatible, (std::vector<uint8_t>{0, 1}));
}

TEST(RdmaPeerRails, PrimaryCreditPressureNeverOverflowsToLowerTier) {
  const std::vector<std::vector<uint8_t>> tiers{{1, 0}, {0, 1}};
  const auto compatible =
      rdma::HighestCompatibleTier(tiers, /*peer_healthy=*/{1, 1});
  ASSERT_EQ(compatible, (std::vector<uint8_t>{1, 0}));
  rdma::RailPolicy policy(
      2, rdma::RailPolicyConfig{/*credits_per_rail=*/1,
                                /*error_threshold=*/3,
                                /*quarantine_us=*/1000,
                                /*latency_weight=*/1,
                                /*error_penalty_us=*/100});

  auto primary = rdma::AcquireHighestTier(
      policy, 1, 0, compatible, /*preferred=*/{1, 0},
      /*fallback=*/{0, 1});
  ASSERT_TRUE(primary);
  ASSERT_EQ(primary->rail, 0u);
  EXPECT_FALSE(rdma::AcquireHighestTier(
      policy, 1, 0, compatible, /*preferred=*/{1, 0},
      /*fallback=*/{0, 1}))
      << "healthy Tier-0 pressure must wait or time out, never use Tier-1";
  const auto pressured = policy.Snapshot(101);
  EXPECT_GT(pressured[0].credits_exhausted, 0u);
  EXPECT_EQ(pressured[1].selections, 0u);
  policy.Complete(*primary, 1, rdma::RailCompletion::kSuccess, 102);
}

TEST(RdmaPeerRails, MixedVersionPolicyFailsClosedOnlyForExplicitTiers) {
  const std::vector<std::string> devices{"gpu0", "cpu0"};
  std::vector<std::vector<uint8_t>> explicit_tiers;
  ASSERT_TRUE(rdma::ParseRailTiers("gpu0;cpu0", devices, &explicit_tiers,
                                   nullptr));
  rdma::PeerTopologyStore strict_store(
      devices, explicit_tiers, /*require_complete=*/true);
  EXPECT_FALSE(strict_store.Snapshot("legacy-peer")->complete);
  EXPECT_TRUE(strict_store.Snapshot("legacy-peer")->compatible.empty());
  ASSERT_TRUE(strict_store.Update(
      PeerTopology{"legacy-peer", 1, false, {}}));
  EXPECT_TRUE(strict_store.Snapshot("legacy-peer")->compatible.empty());

  std::vector<std::vector<uint8_t>> homogeneous;
  ASSERT_TRUE(rdma::ParseRailTiers("", devices, &homogeneous, nullptr));
  rdma::PeerTopologyStore legacy_store(
      devices, homogeneous, /*require_complete=*/false);
  EXPECT_TRUE(legacy_store.Snapshot("legacy-peer")->complete);
  EXPECT_EQ(legacy_store.Snapshot("legacy-peer")->compatible,
            (std::vector<uint8_t>{1, 1}));
  ASSERT_TRUE(legacy_store.Update(
      PeerTopology{"legacy-peer", 1, false, {}}));
  EXPECT_EQ(legacy_store.Snapshot("legacy-peer")->compatible,
            (std::vector<uint8_t>{1, 1}));

  std::string error;
  EXPECT_FALSE(rdma::ParseRailTiers("gpu0;missing", devices, &homogeneous,
                                    &error));
  EXPECT_FALSE(error.empty());
}

TEST(RdmaPeerRails, ConcurrentPublicationAcquisitionAndSnapshotTeardown) {
  const std::vector<std::string> devices{"gpu0", "gpu1", "cpu0"};
  std::vector<std::vector<uint8_t>> tiers;
  ASSERT_TRUE(rdma::ParseRailTiers("gpu0|gpu1;cpu0", devices, &tiers,
                                   nullptr));
  auto store = std::make_unique<rdma::PeerTopologyStore>(
      devices, tiers, /*require_complete=*/true);
  ASSERT_TRUE(store->Update(PeerTopology{
      "peer", 2, true,
      {{"gpu0", true}, {"gpu1", false}, {"cpu0", true}}}));
  rdma::RailPolicy policy(3);

  std::atomic<bool> start{false};
  std::atomic<uint64_t> violations{0};
  std::vector<std::thread> threads;
  for (uint64_t writer = 0; writer < 2; ++writer) {
    threads.emplace_back([&, writer] {
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      for (uint64_t i = 0; i < 1000; ++i) {
        const uint64_t generation = 10 + writer * 1000 + i;
        const bool even = generation % 2 == 0;
        store->Update(PeerTopology{
            "peer", generation, true,
            {{"gpu0", even}, {"gpu1", !even}, {"cpu0", true}}});
      }
    });
  }
  for (size_t reader = 0; reader < 4; ++reader) {
    threads.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      for (size_t i = 0; i < 2000; ++i) {
        const auto snapshot = store->Snapshot("peer");
        const bool even = snapshot->generation % 2 == 0;
        const std::vector<uint8_t> expected =
            even ? std::vector<uint8_t>{1, 0, 0}
                 : std::vector<uint8_t>{0, 1, 0};
        auto lease =
            policy.TryAcquire(1, snapshot->generation, snapshot->compatible);
        if (!snapshot->complete || snapshot->compatible != expected || !lease ||
            lease->rail >= snapshot->compatible.size() ||
            snapshot->compatible[lease->rail] == 0) {
          violations.fetch_add(1, std::memory_order_relaxed);
        }
        if (lease) {
          policy.Complete(*lease, 1, rdma::RailCompletion::kSuccess,
                          snapshot->generation);
        }
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (auto& thread : threads) thread.join();
  EXPECT_EQ(violations.load(std::memory_order_relaxed), 0u);
  for (const auto& rail : policy.Snapshot(3000))
    EXPECT_EQ(rail.inflight, 0u);

  const auto retained = store->Snapshot("peer");
  std::atomic<bool> inspect{false};
  std::thread reader([retained, &inspect, &violations] {
    while (!inspect.load(std::memory_order_acquire))
      std::this_thread::yield();
    for (size_t i = 0; i < 10000; ++i) {
      if (!retained->complete || retained->compatible.empty())
        violations.fetch_add(1, std::memory_order_relaxed);
    }
  });
  inspect.store(true, std::memory_order_release);
  store.reset();
  reader.join();
  EXPECT_EQ(violations.load(std::memory_order_relaxed), 0u);
}

TEST(RdmaSafety, QuarantinedRailAllowsOneProbeAndRecoversOnlyOnSuccess) {

  rdma::RailPolicyConfig config;
  config.credits_per_rail = 64;
  config.error_threshold = 1;
  config.quarantine_us = 1000;
  config.latency_weight = 1;
  config.error_penalty_us = 100;
  rdma::RailPolicy policy(2, config);

  auto failed = policy.TryAcquire(1, 100);
  ASSERT_TRUE(failed);
  ASSERT_EQ(failed->rail, 0u);
  policy.Complete(*failed, 10, rdma::RailCompletion::kRailFailure, 200);
  auto before_cooldown = policy.TryAcquire(1, 1199);
  ASSERT_TRUE(before_cooldown);
  EXPECT_EQ(before_cooldown->rail, 1u);
  policy.Complete(*before_cooldown, 10, rdma::RailCompletion::kSuccess, 1199);

  constexpr size_t kCallers = 32;
  std::mutex gate_mu;
  std::condition_variable ready_cv;
  std::condition_variable start_cv;
  std::condition_variable completed_cv;
  size_t ready = 0;
  size_t completed = 0;
  bool start = false;
  std::vector<std::optional<rdma::RailLease>> leases(kCallers);
  std::vector<std::thread> callers;
  callers.reserve(kCallers);
  for (size_t i = 0; i < kCallers; ++i) {
    callers.emplace_back([&, i] {
      std::unique_lock<std::mutex> lock(gate_mu);
      ++ready;
      ready_cv.notify_one();
      start_cv.wait(lock, [&] { return start; });
      lock.unlock();
      leases[i] = policy.TryAcquire(1, 1200);
      lock.lock();
      ++completed;
      completed_cv.notify_one();
    });
  }
  bool all_ready = false;
  {
    std::unique_lock<std::mutex> lock(gate_mu);
    all_ready = ready_cv.wait_for(lock, std::chrono::seconds(2),
                                  [&] { return ready == kCallers; });
    start = true;
  }
  start_cv.notify_all();
  EXPECT_TRUE(all_ready) << "callers did not reach the start barrier";
  {
    std::unique_lock<std::mutex> lock(gate_mu);
    EXPECT_TRUE(completed_cv.wait_for(lock, std::chrono::seconds(2),
                                      [&] { return completed == kCallers; }))
        << "rail-policy callers deadlocked";
  }
  for (auto& caller : callers) caller.join();

  size_t probe_index = kCallers;
  size_t rail0_leases = 0;
  for (size_t i = 0; i < leases.size(); ++i) {
    ASSERT_TRUE(leases[i]) << i;
    if (leases[i]->rail == 0) {
      ++rail0_leases;
      probe_index = i;
    } else {
      EXPECT_EQ(leases[i]->rail, 1u);
    }
  }
  ASSERT_EQ(rail0_leases, 1u);
  auto stats = policy.Snapshot(1200);
  EXPECT_TRUE(stats[0].quarantined);
  EXPECT_TRUE(stats[0].recovery_probe);
  EXPECT_EQ(stats[0].recoveries, 0u);

  for (size_t i = 0; i < leases.size(); ++i) {
    if (i == probe_index) continue;
    policy.Complete(*leases[i], 10, rdma::RailCompletion::kSuccess, 1201);
  }
  policy.Complete(*leases[probe_index], 10,
                  rdma::RailCompletion::kEndpointFailure, 1201);
  stats = policy.Snapshot(1201);
  EXPECT_TRUE(stats[0].quarantined);
  EXPECT_FALSE(stats[0].recovery_probe);
  EXPECT_EQ(stats[0].recoveries, 0u);
  EXPECT_EQ(stats[0].endpoint_errors, 1u);

  auto failed_probe = policy.TryAcquire(1, 1202);
  ASSERT_TRUE(failed_probe);
  ASSERT_EQ(failed_probe->rail, 0u);
  policy.Complete(*failed_probe, 10, rdma::RailCompletion::kRailFailure, 1202);
  stats = policy.Snapshot(1202);
  EXPECT_EQ(stats[0].quarantines, 2u);
  EXPECT_TRUE(stats[0].quarantined);
  EXPECT_FALSE(stats[0].recovery_probe);

  auto other_success = policy.TryAcquire(1, 2201);
  ASSERT_TRUE(other_success);
  ASSERT_EQ(other_success->rail, 1u);
  policy.Complete(*other_success, 10, rdma::RailCompletion::kSuccess, 2201);
  stats = policy.Snapshot(2201);
  EXPECT_EQ(stats[0].recoveries, 0u)
      << "success on another rail must not recover the quarantined rail";

  auto successful_probe = policy.TryAcquire(1, 2202);
  ASSERT_TRUE(successful_probe);
  ASSERT_EQ(successful_probe->rail, 0u);
  policy.Complete(*successful_probe, 10, rdma::RailCompletion::kSuccess, 2203);
  stats = policy.Snapshot(2203);
  EXPECT_EQ(stats[0].recoveries, 1u);
  EXPECT_EQ(stats[0].consecutive_errors, 0u);
  EXPECT_FALSE(stats[0].quarantined);
  EXPECT_FALSE(stats[0].recovery_probe);

  auto admitted_after_recovery = policy.TryAcquire(1, 2204);
  ASSERT_TRUE(admitted_after_recovery);
  EXPECT_EQ(admitted_after_recovery->rail, 0u);
  policy.Complete(*admitted_after_recovery, 10,
                  rdma::RailCompletion::kSuccess, 2205);
}

TEST(RdmaSafety, CompletionDeadlineUsesOneAbsoluteBudget) {
  using Clock = CompletionDeadline::Clock;
  const auto start = Clock::time_point(std::chrono::milliseconds(100));
  CompletionDeadline deadline(50, start);
  EXPECT_EQ(deadline.RemainingAt(start), 50);
  EXPECT_EQ(deadline.RemainingAt(start + std::chrono::milliseconds(30)), 20);
  // A partial completion at +30 ms does not create a fresh 50-ms wait.
  EXPECT_EQ(deadline.RemainingAt(start + std::chrono::milliseconds(49)), 1);
  EXPECT_EQ(deadline.RemainingAt(start + std::chrono::milliseconds(50)), 0);
  CompletionDeadline infinite(-1, start);
  EXPECT_EQ(infinite.RemainingAt(start + std::chrono::hours(24)), -1);
}

TEST(RdmaSafety, PartialMultiRailPoolGrowthRollsBackBeforePublication) {
  std::vector<FakePoolRail> rails(3, FakePoolRail{64, 64});
  rails[1].reject_stage = true;
  size_t published_bytes = 64;
  const bool registered = rdma::RunPoolMrTransaction(
      rails.size(),
      [&](size_t rail) { return rails[rail].Stage(128); },
      [&](size_t rail) { rails[rail].Commit(); },
      [&](size_t rail) { rails[rail].Rollback(); });
  if (registered) published_bytes = 128;

  EXPECT_FALSE(registered);
  EXPECT_EQ(published_bytes, 64u);
  for (const auto& rail : rails) {
    EXPECT_EQ(rail.declared, 64u);
    EXPECT_EQ(rail.staged, 64u);
    EXPECT_EQ(rail.commits, 0u);
  }
  EXPECT_EQ(rails[0].rollbacks, 1u);
  EXPECT_EQ(rails[1].rollbacks, 0u);
  EXPECT_EQ(rails[2].rollbacks, 0u);
}

TEST(RdmaSafety, SuccessfulMultiRailPoolGrowthKeepsOldRangeUsable) {
  std::vector<FakePoolRail> rails(3, FakePoolRail{64, 64});
  size_t published_bytes = 64;
  const bool registered = rdma::RunPoolMrTransaction(
      rails.size(),
      [&](size_t rail) { return rails[rail].Stage(128); },
      [&](size_t rail) { rails[rail].Commit(); },
      [&](size_t rail) { rails[rail].Rollback(); });
  if (registered) published_bytes = 128;

  ASSERT_TRUE(registered);
  EXPECT_EQ(published_bytes, 128u);
  for (const auto& rail : rails) {
    EXPECT_EQ(rail.declared, 128u);
    EXPECT_EQ(rail.commits, 1u);
    EXPECT_EQ(rail.rollbacks, 0u);
    EXPECT_TRUE(rail.Covers(8, 16));
    EXPECT_TRUE(rail.Covers(96, 16));
  }
}

TEST(RdmaSafety, OverridesRejectMalformedVectorsBeforeAcquiringQp) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaTransport transport(kMaxMsg);
  const std::vector<BlockKey> keys = {{1, 2}, {3, 4}};
  std::vector<uint64_t> value_lengths;
  const auto mismatched =
      transport.RangeInto("unused", keys, {{nullptr, 0}}, &value_lengths);
  ASSERT_EQ(mismatched.size(), keys.size());
  EXPECT_EQ(mismatched[0], Status::kInvalid);
  EXPECT_EQ(mismatched[1], Status::kInvalid);

  const auto null_write = transport.CacheMany(
      "unused", {CacheItem{keys[0], nullptr, 1}});
  ASSERT_EQ(null_write.size(), 1u);
  EXPECT_EQ(null_write[0], Status::kInvalid);

  const void* nonnull = reinterpret_cast<const void*>(uintptr_t{1});
  CacheSrcMulti overflow{
      keys[0],
      {{nonnull, std::numeric_limits<size_t>::max()}, {nonnull, 1}}};
  const auto overflow_status =
      transport.CacheFromMulti("unused", {overflow});
  ASSERT_EQ(overflow_status.size(), 1u);
  EXPECT_EQ(overflow_status[0], Status::kInvalid);

  EXPECT_EQ(transport.ExistMany("unused", keys, nullptr),
            std::vector<Status>(keys.size(), Status::kInvalid));
  EXPECT_EQ(transport.Members("unused", nullptr), Status::kInvalid);
  EXPECT_EQ(CounterVal(transport.MetricsText(),
                       "dfkv_rdma_client_conns_opened_total"), 0);
}



// Direct transport ExistMany: windowed batch existence probe on one connection.
// Must be correct across multiple send windows (N > depth) with mixed hit/miss.
TEST(RdmaLoopback, ExistManyWindowedMixedHitMiss) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device (load rdma_rxe for Soft-RoCE)";
  RdmaNode node("exm");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  const int N = 80;  // exceeds a single send window (depth) to exercise looping
  for (int i = 0; i < N; ++i) {
    std::string v = "v" + std::to_string(i);
    ASSERT_TRUE(c.Put("p" + std::to_string(i), v.data(), v.size())) << i;
  }
  // Interleave present (even) and absent (odd) keys using the exact namespace
  // the KVClient used for the writes.
  std::vector<BlockKey> keys;
  for (int i = 0; i < N; ++i) {
    keys.push_back(ToBlockKey(SelfHdr(), "p" + std::to_string(i)));
    keys.push_back(ToBlockKey(SelfHdr(), "absent" + std::to_string(i)));
  }
  std::vector<char> exists;
  auto sts = rt.ExistMany(node.addr, keys, &exists);
  ASSERT_EQ(exists.size(), keys.size());
  ASSERT_EQ(sts.size(), keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    bool want = (i % 2 == 0);
    EXPECT_EQ(exists[i] != 0, want) << "i=" << i;
  }
}

// Client BatchExist over RDMA may expand one node's pool for parallel probe
// shards, but repeated calls must reuse that bounded pool instead of opening a
// fresh connection per key.
TEST(RdmaLoopback, BatchExistReusesExpandedPool) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("bex");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  const int N = 64;  // > batch_concurrency (8): the old per-key fan-out opened many
  for (int i = 0; i < N; ++i) {
    std::string v = "v" + std::to_string(i);
    ASSERT_TRUE(c.Put("e" + std::to_string(i), v.data(), v.size())) << i;
  }
  std::vector<std::string> probe;
  for (int i = 0; i < N; ++i) {
    probe.push_back("e" + std::to_string(i));        // present
    probe.push_back("e" + std::to_string(i) + "_x"); // absent
  }

  // Warm one connection, then let two large batches settle the bounded
  // per-node pool. Thread scheduling need not expose peak fan-out in one call.
  EXPECT_TRUE(c.Exist("e0"));
  const long before =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  ASSERT_GE(before, 1);

  auto er = c.BatchExist(probe);
  ASSERT_EQ(er.size(), probe.size());
  for (size_t i = 0; i < probe.size(); ++i)
    EXPECT_EQ((bool)er[i], (i % 2 == 0)) << probe[i];
  const long expanded =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  EXPECT_LE(expanded - before, 15);  // default max 16, one already warm

  auto again = c.BatchExist(probe);
  ASSERT_EQ(again.size(), probe.size());
  for (size_t i = 0; i < probe.size(); ++i)
    EXPECT_EQ((bool)again[i], (i % 2 == 0)) << probe[i];
  const long settled =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  EXPECT_LE(settled - before, 15);  // default pool cap 16, one already warm

  auto steady = c.BatchExist(probe);
  ASSERT_EQ(steady.size(), probe.size());
  for (size_t i = 0; i < probe.size(); ++i)
    EXPECT_EQ((bool)steady[i], (i % 2 == 0)) << probe[i];
  const long reused =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  EXPECT_EQ(reused, settled)
      << "settled BatchExist did not reuse its bounded connection pool";
}

// Non-SG batch PUT: one oversized item must fail ONLY itself, not poison the
// whole same-node batch. Previously any item over the per-op payload bound
// filled every sibling's status with kInvalid — all lost their cache write.
TEST(RdmaLoopback, BatchPutOversizedFailsOnlyOffender) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("bpo");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  std::string small(4096, 's');
  std::string big(kMaxMsg + 4096, 'b');  // exceeds the per-op payload bound
  std::vector<KvPutItem> items = {
      {"bpo_a", small.data(), small.size()},
      {"bpo_big", big.data(), big.size()},
      {"bpo_c", small.data(), small.size()},
  };
  auto pr = c.BatchPut(items);
  ASSERT_EQ(pr.size(), 3u);
  EXPECT_TRUE(pr[0]) << "sibling poisoned by the oversized item";
  EXPECT_FALSE(pr[1]);
  EXPECT_TRUE(pr[2]) << "sibling poisoned by the oversized item";
  EXPECT_TRUE(c.Exist("bpo_a"));
  EXPECT_FALSE(c.Exist("bpo_big"));
  EXPECT_TRUE(c.Exist("bpo_c"));
}

// Same-host GET rendezvous (phase 5): with DFKV_CLIENT_NODE_DEDUP=1, a second
// client reading the SAME keys must be served from the shm rendezvous, not the
// server — server-side completions stay ~flat while the data stays correct.
TEST(RdmaLoopback, NodeDedupCollapsesSameHostGets) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const std::string nm = NodeDedup::EnvSegmentName(
      ToBlockKey(SelfHdr(), "").digest_hi);
  ::shm_unlink(nm.c_str());
  ::setenv("DFKV_CLIENT_NODE_DEDUP", "1", 1);
  {
    RdmaNode node("ndd");
    RdmaTransport rt1(kMaxMsg), rt2(kMaxMsg);
    KVClient c1({{"n", node.addr}}, SelfHdr(), &rt1);
    KVClient c2({{"n", node.addr}}, SelfHdr(), &rt2);

    const int N = 32;
    std::vector<std::string> vals(N);
    for (int i = 0; i < N; ++i) {
      vals[i].assign(8192, '\0');
      for (size_t b = 0; b < vals[i].size(); ++b)
        vals[i][b] = static_cast<char>((i * 131 + b * 7) & 0xFF);
      ASSERT_TRUE(c1.Put("ndd" + std::to_string(i), vals[i].data(), vals[i].size())) << i;
    }

    auto get_all = [&](KVClient& c, std::vector<std::string>* out) {
      std::vector<KvGetItem> items(N);
      out->assign(N, std::string(8192, '\0'));
      for (int i = 0; i < N; ++i)
        items[i] = {"ndd" + std::to_string(i), &(*out)[i][0], 8192};
      return c.BatchGet(items);
    };

    std::vector<std::string> o1, o2;
    auto r1 = get_all(c1, &o1);  // first reader: remote fetch + publish
    for (int i = 0; i < N; ++i) { ASSERT_TRUE(r1[i]) << i; EXPECT_EQ(o1[i], vals[i]); }
    const long mid = CounterVal(node.rsrv->MetricsText(), "dfkv_rdma_completions_total");
    auto r2 = get_all(c2, &o2);  // peer: rendezvous hits, no server traffic
    for (int i = 0; i < N; ++i) { ASSERT_TRUE(r2[i]) << i; EXPECT_EQ(o2[i], vals[i]); }
    const long after = CounterVal(node.rsrv->MetricsText(), "dfkv_rdma_completions_total");
    EXPECT_LE(after - mid, N / 4)
        << "peer reads reached the server (" << (after - mid)
        << " completions); rendezvous not deduplicating";
  }
  ::unsetenv("DFKV_CLIENT_NODE_DEDUP");
  ::shm_unlink(nm.c_str());
}

TEST(RdmaLoopback, PutGetExistMissOverRdma) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device (load rdma_rxe for Soft-RoCE)";
  RdmaNode node("pgem");
  RdmaTransport rt(kMaxMsg);  // first device (rxe0 in CI)
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  std::string v(4096, '\0');
  for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<char>((i * 31 + 7) & 0xFF);
  ASSERT_TRUE(c.Put("k1", v.data(), v.size()));
  EXPECT_TRUE(c.Exist("k1"));
  std::string out(v.size(), '\0');
  ASSERT_TRUE(c.Get("k1", &out[0], out.size()));
  EXPECT_EQ(out, v);

  // miss: absent key
  std::string m(v.size(), '\0');
  EXPECT_FALSE(c.Get("absent", &m[0], m.size()));
  EXPECT_FALSE(c.Exist("absent"));
}

// Observability counters: the server tallies request completions and the client
// transport tallies connections opened (+ per-rail) and MR regions.
TEST(RdmaLoopback, MetricsCountersTrackOps) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("mco");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  std::vector<char> pool(64 * 1024);
  ASSERT_TRUE(c.RegisterMemory(pool.data(), pool.size()));
  std::string v(2048, 'z');
  ASSERT_TRUE(c.Put("m1", v.data(), v.size()));
  std::string out(v.size(), '\0');
  ASSERT_TRUE(c.Get("m1", &out[0], out.size()));

  // server: at least the PUT + GET requests were completed
  EXPECT_GE(node.rsrv->Completions(), 2u);
  std::string srv_text = node.rsrv->MetricsText();
  EXPECT_NE(srv_text.find("dfkv_rdma_completions_total"), std::string::npos) << srv_text;
  EXPECT_GE(CounterVal(srv_text, "dfkv_rdma_v2_conns_opened_total"), 1);
  EXPECT_GE(CounterVal(srv_text, "dfkv_rdma_v2_put_writes_total"), 1);
  EXPECT_GE(CounterVal(srv_text, "dfkv_rdma_v2_get_writes_total"), 1);
  EXPECT_GT(CounterVal(srv_text, "dfkv_rdma_recv_segment_bytes"), 0);
  EXPECT_NE(srv_text.find("dfkv_rdma_rail_active_conns{dev=\""),
            std::string::npos) << srv_text;
  EXPECT_GE(CounterVal(srv_text, "dfkv_rdma_rail_completions_total"), 2);
  EXPECT_GE(CounterVal(srv_text, "dfkv_rdma_rail_put_writes_total"), 1);
  EXPECT_GE(CounterVal(srv_text, "dfkv_rdma_rail_put_bytes_total"), 2048);
  EXPECT_GE(CounterVal(srv_text, "dfkv_rdma_rail_get_writes_total"), 1);
  EXPECT_GE(CounterVal(srv_text, "dfkv_rdma_rail_get_bytes_total"), 2048);

  // client transport: a connection was opened and the MR region declared
  std::string cli_text = rt.MetricsText();
  EXPECT_NE(cli_text.find("dfkv_rdma_client_conns_opened_total"), std::string::npos) << cli_text;
  EXPECT_NE(cli_text.find("dfkv_rdma_client_rail_conns_total{dev="), std::string::npos) << cli_text;
  EXPECT_NE(cli_text.find("dfkv_rdma_client_mr_regions 1"), std::string::npos) << cli_text;
  EXPECT_GE(CounterVal(cli_text, "dfkv_rdma_client_v2_put_writes_total"), 1);
  EXPECT_GE(CounterVal(cli_text, "dfkv_rdma_client_v2_get_writes_total"), 1);
  EXPECT_GE(CounterVal(cli_text,
                       "dfkv_rdma_client_pool_mr_registrations_total"), 1);
  EXPECT_GE(CounterVal(cli_text, "dfkv_rdma_client_max_block_seen_bytes"),
            2048);
  EXPECT_EQ(CounterVal(cli_text,
                       "dfkv_rdma_client_transient_user_mr_active"), 0);
  EXPECT_GE(CounterVal(cli_text, "dfkv_rdma_client_v2_probe_attempts_total"),
            1);
  EXPECT_GE(CounterVal(cli_text,
                       "dfkv_rdma_client_completion_timeouts_total"), 0);

  // and the client snapshot folds transport metrics in after the health metrics
  std::string snap = c.MetricsSnapshot();
  EXPECT_NE(snap.find("dfkv_client_ops_served_total"), std::string::npos) << snap;
  EXPECT_NE(snap.find("dfkv_rdma_client_conns_opened_total"), std::string::npos) << snap;
}

TEST(RdmaLoopback, MembersReplyHonorsExactBoundAndRejectsOversize) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("large-members");
  const std::string boundary(rdma::kV2ControlResponseMax, 'm');
  node.srv->set_members(boundary);

  RdmaTransport transport(kMaxMsg);
  std::string output;
  ASSERT_EQ(transport.Members(node.addr, &output), Status::kOk);
  EXPECT_EQ(output, boundary);
  EXPECT_GE(node.rsrv->V2Conns(), 1u);

  node.srv->set_members(
      std::string(rdma::kV2ControlResponseMax + 1, 'x'));
  output = "must be cleared";
  EXPECT_EQ(transport.Members(node.addr, &output), Status::kIOError);
  EXPECT_TRUE(output.empty());

  node.srv->set_members(boundary);
  ASSERT_EQ(transport.Members(node.addr, &output), Status::kOk);
  EXPECT_EQ(output, boundary);
}

TEST(RdmaLoopback, StartupFailsWhenV2ReceiveSegmentIsUnavailable) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ::setenv("DFKV_RDMA_RECV_SEGMENT_SIZE", "4096", 1);
  RdmaServer server(
      [](uint8_t, const BlockKey&, uint64_t, uint64_t, const char*, uint64_t,
         std::string*, size_t*) { return Status::kInvalid; },
      kMaxMsg);
  EXPECT_EQ(server.Start(0), Status::kIOError);
  ::unsetenv("DFKV_RDMA_RECV_SEGMENT_SIZE");
}


TEST(RdmaLoopback, V2CacheAcceptsReadOnlySourceMemory) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("readonly-v2");
  RdmaTransport transport(kMaxMsg);
  const BlockKey key = ToBlockKey(SelfHdr(), "readonly-v2");
  constexpr size_t kPayload = 512;
  const size_t stored_len = kPayload;
  void* mapping =
      ::mmap(nullptr, 4096, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(mapping, MAP_FAILED);
  std::memset(mapping, 'r', kPayload);
  ASSERT_EQ(::mprotect(mapping, 4096, PROT_READ), 0);

  ASSERT_EQ(transport.Cache(node.addr, key, mapping, stored_len), Status::kOk);
  std::string output;
  ASSERT_EQ(transport.Range(node.addr, key, 0, stored_len, &output),
            Status::kOk);
  ASSERT_EQ(output.size(), stored_len);
  EXPECT_EQ(std::memcmp(output.data(), mapping, stored_len), 0);
  EXPECT_EQ(::munmap(mapping, 4096), 0);
}

TEST(RdmaLoopback, DirectSingleCacheRangeUsesV2Writes) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("direct-v2");
  RdmaTransport transport(kMaxMsg);
  const BlockKey key = ToBlockKey(SelfHdr(), "direct-v2");
  std::string stored(4096, '\0');
  for (size_t i = 0; i < stored.size(); ++i)
    stored[i] = static_cast<char>((i * 29 + 3) & 0xff);

  ASSERT_EQ(transport.Cache(node.addr, key, stored.data(), stored.size()),
            Status::kOk);
  std::string output;
  ASSERT_EQ(transport.Range(node.addr, key, 0, stored.size(), &output),
            Status::kOk);
  EXPECT_EQ(output, stored);
  EXPECT_GE(node.rsrv->V2PutWrites(), 1u);
  EXPECT_GE(node.rsrv->V2GetWrites(), 1u);
}

TEST(RdmaLoopback, BatchZeroCopyRoundtrip) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("bzc");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  const int N = 20;
  const size_t sz = 4096;
  std::vector<std::string> vals(N), keys(N);
  std::vector<KvPutItem> puts(N);
  for (int i = 0; i < N; ++i) {
    keys[i] = "b" + std::to_string(i);
    vals[i].assign(sz, static_cast<char>((i * 13 + 1) & 0xFF));
    puts[i] = {keys[i], vals[i].data(), sz};
  }
  auto pr = c.BatchPut(puts);
  for (int i = 0; i < N; ++i) EXPECT_TRUE(pr[i]) << i;

  // GET into fresh buffers (RDMA scatters payload straight in = zero copy).
  std::vector<std::string> outs(N);
  std::vector<KvGetItem> gets(N);
  for (int i = 0; i < N; ++i) { outs[i].assign(sz, '\0'); gets[i] = {keys[i], &outs[i][0], sz}; }
  auto gr = c.BatchGet(gets);
  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(gr[i]) << i;
    EXPECT_EQ(outs[i], vals[i]) << i;
  }
}

// Single Put/Get always take the zero-copy fast path on RDMA (register caller
// buffer + scatter-send / scatter-recv) — no size threshold.
TEST(RdmaLoopback, SingleZeroCopyPutGet) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("szc");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  std::string v(8192, '\0');
  for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<char>((i * 37 + 11) & 0xFF);
  // Put twice into the SAME source buffer to exercise the MR-cache hit on re-put.
  ASSERT_TRUE(c.Put("z1", v.data(), v.size()));
  ASSERT_TRUE(c.Put("z1", v.data(), v.size()));

  // Get into the SAME dst buffer twice (scatter-recv straight in; MR-cache hit).
  std::string out(v.size(), '\0');
  ASSERT_TRUE(c.Get("z1", &out[0], out.size()));
  EXPECT_EQ(out, v);
  out.assign(v.size(), '\0');
  ASSERT_TRUE(c.Get("z1", &out[0], out.size()));
  EXPECT_EQ(out, v);

  // Miss on the zero-copy Get path must be a clean miss (kNotFound), not an error.
  std::string m(v.size(), '\0');
  EXPECT_FALSE(c.Get("absent", &m[0], m.size()));
  // Size mismatch (stored payload_len != requested n) => miss, not corruption.
  std::string shorter(v.size() / 2, '\0');
  EXPECT_FALSE(c.Get("z1", &shorter[0], shorter.size()));
}

// A registered memory region (RegisterMemory) is registered once; every buffer
// inside it resolves to that one MR with no per-op ibv_reg_mr. Verified directly
// at the endpoint: two distinct sub-buffers return the SAME MR; an outside buffer
// registers ad-hoc (a different MR).
// Many endpoints on the same device must all Open via the shared per-device
// ibv_context+PD registry (#6 fix: no per-connection ibv_open_device thrash).
// Opening + closing N in waves exercises the refcount get-or-create/free path;
// a leak or double-free here would surface as an Open failure or crash.
TEST(RdmaLoopback, ManyEndpointsShareDeviceContext) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  constexpr int N = 24;  // > typical t16; well past the 1-2 conn happy path
  {
    std::vector<std::unique_ptr<rdma::RcEndpoint>> eps;
    for (int i = 0; i < N; ++i) {
      eps.push_back(std::make_unique<rdma::RcEndpoint>());
      ASSERT_TRUE(eps.back()->Open(nullptr, 64 * 1024, 1)) << "endpoint " << i;
    }
    eps.clear();  // all close -> registry refcount returns to 0, frees ctx+pd
  }
  // A fresh endpoint after the registry drained must still open (re-creates the
  // shared device cleanly).
  rdma::RcEndpoint again;
  ASSERT_TRUE(again.Open(nullptr, 64 * 1024, 1));
}


TEST(RdmaLoopback, RegisterMemoryRejectsInvalidRange) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaTransport rt(kMaxMsg);
  char byte = 0;
  EXPECT_FALSE(rt.RegisterMemory(nullptr, 1));
  EXPECT_FALSE(rt.RegisterMemory(&byte, 0));
  const std::string metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_mr_regions"), 0);
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_mr_registered_bytes"), 0);
  EXPECT_EQ(CounterVal(
                metrics,
                "dfkv_rdma_client_mr_registration_rejections_total"),
            2);
}

// Client-side anchor: RegisterMemory must register the pool MRs at
// DECLARATION time (holding a lifetime device ref), not on the first
// connection's first op — a 141 GB host pool costs ~4 s to pin, which
// belongs in engine startup, not the first lookup.
TEST(RdmaLoopback, RegisterMemoryAnchorsPoolMrBeforeFirstConn) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const uint64_t before = rdma::RcEndpoint::PoolMrRegistrations();
  RdmaTransport rt(kMaxMsg);
  std::vector<char> pool(256 * 1024);
  ASSERT_TRUE(rt.RegisterMemory(pool.data(), 64 * 1024));
  EXPECT_GT(rdma::RcEndpoint::PoolMrRegistrations(), before)
      << "pool MR not registered at declaration time (anchor missing)";
  std::string metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_mr_regions"), 1);
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_mr_registered_bytes"),
            64 * 1024);

  ASSERT_TRUE(rt.RegisterMemory(pool.data(), pool.size()));
  metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_mr_regions"), 1);
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_mr_registered_bytes"),
            static_cast<long>(pool.size()));
  EXPECT_EQ(CounterVal(
                metrics,
                "dfkv_rdma_client_mr_registration_rejections_total"),
            0);

  const uintptr_t base = reinterpret_cast<uintptr_t>(pool.data());
  const size_t wrapping_size =
      std::numeric_limits<uintptr_t>::max() - base + 1;
  EXPECT_FALSE(rt.RegisterMemory(pool.data(), wrapping_size));
  metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_mr_regions"), 1);
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_mr_registered_bytes"),
            static_cast<long>(pool.size()));
  EXPECT_EQ(CounterVal(
                metrics,
                "dfkv_rdma_client_mr_registration_rejections_total"),
            1);
}

TEST(RdmaLoopback, SameBasePoolRegistrationGrowthIsEffective) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  rdma::RcEndpoint endpoint;
  rdma::RcEndpoint old_generation_user;
  ASSERT_TRUE(endpoint.Open(nullptr, 64 * 1024, 1));
  ASSERT_TRUE(old_generation_user.Open(nullptr, 64 * 1024, 1));
  std::vector<char> region(256 * 1024);
  ASSERT_TRUE(endpoint.AddPoolMr(region.data(), 64 * 1024, true));
  ASSERT_TRUE(
      old_generation_user.AddPoolMr(region.data(), 64 * 1024, true));
  ibv_mr* initial = endpoint.RegisterUser(region.data() + 4096, 4096);
  ASSERT_NE(initial, nullptr);
  EXPECT_EQ(old_generation_user.RegisterUser(region.data() + 4096, 4096),
            initial);
  const uint64_t registrations = rdma::RcEndpoint::PoolMrRegistrations();
  const uint64_t active =
      rdma::RcEndpoint::PoolMrActiveRegistrations();

  ASSERT_TRUE(endpoint.AddPoolMr(region.data(), region.size(), true));
  EXPECT_EQ(rdma::RcEndpoint::PoolMrRegistrations(), registrations + 1);
  EXPECT_EQ(rdma::RcEndpoint::PoolMrActiveRegistrations(), active + 1)
      << "the old generation stays leased by its current endpoint";
  ibv_mr* grown =
      endpoint.RegisterUser(region.data() + 128 * 1024, 4096);
  ASSERT_NE(grown, nullptr);
  EXPECT_NE(grown, initial);
  EXPECT_EQ(endpoint.RegisterUser(region.data() + 4096, 4096), grown)
      << "the grown endpoint resolves the old prefix through the new MR";
  EXPECT_EQ(old_generation_user.RegisterUser(region.data() + 4096, 4096),
            initial)
      << "an endpoint using the previous generation keeps its old range valid";
  EXPECT_EQ(old_generation_user.RegisterUser(
                region.data() + 128 * 1024, 4096),
            nullptr);

  ASSERT_TRUE(old_generation_user.AddPoolMr(
      region.data(), region.size(), true));
  EXPECT_EQ(rdma::RcEndpoint::PoolMrActiveRegistrations(), active)
      << "the superseded generation retires after its final endpoint advances";
  EXPECT_EQ(old_generation_user.RegisterUser(region.data() + 4096, 4096),
            grown);
}

TEST(RdmaLoopback, UnregisteredBuffersLeaveNoLiveMrAfterReturn) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("transient-lifetime");
  RdmaTransport transport(kMaxMsg);
  const BlockKey key = ToBlockKey(SelfHdr(), "transient-lifetime");
  const uint64_t baseline = rdma::RcEndpoint::TransientUserMrActive();

  {
    std::vector<char> source(4096, 't');
    const auto statuses = transport.CacheFrom(
        node.addr, {CacheSrc{key, source.data(), source.size()}});
    ASSERT_EQ(statuses.size(), 1u);
    ASSERT_EQ(statuses[0], Status::kOk);
    EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), baseline);
  }  // source may be freed immediately: no cached MR may reference it.

  {
    std::vector<char> destination(4096);
    std::vector<uint64_t> value_lengths;
    const auto statuses = transport.RangeInto(
        node.addr, {key},
        {RangeDst{destination.data(), destination.size()}}, &value_lengths);
    ASSERT_EQ(statuses.size(), 1u);
    ASSERT_EQ(statuses[0], Status::kOk);
    EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), baseline);
  }  // destination may be freed immediately after return as well.
}

TEST(RdmaLoopback, PoolMrSharedAcrossBuffers) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  rdma::RcEndpoint ep;
  ASSERT_TRUE(ep.Open(nullptr, 64 * 1024, 1));
  std::vector<char> region(256 * 1024);
  ASSERT_TRUE(ep.AddPoolMr(region.data(), region.size()));
  ibv_mr* a = ep.RegisterUser(region.data() + 4096, 4096);
  ibv_mr* b = ep.RegisterUser(region.data() + 200000, 4096);
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(a, b);  // both inside the pool -> one shared MR, no per-op registration
  std::vector<char> outside(4096);
  EXPECT_EQ(ep.RegisterUser(outside.data(), outside.size()), nullptr)
      << "out-of-pool stable lookup must fail closed, never cache caller memory";
}

TEST(RdmaLoopback, PoolMrAccessUpgradeWinsRangeLookup) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  rdma::RcEndpoint ep;
  ASSERT_TRUE(ep.Open(nullptr, 64 * 1024, 1));
  std::vector<char> region(256 * 1024);
  ASSERT_TRUE(ep.AddPoolMr(region.data(), region.size()));
  ibv_mr* local_only = ep.RegisterUser(region.data() + 4096, 4096);
  ASSERT_NE(local_only, nullptr);
  ibv_mr* remote_write =
      ep.RegisterRemoteRegion(region.data(), region.size());
  ASSERT_NE(remote_write, nullptr);
  EXPECT_NE(remote_write, local_only);
  EXPECT_EQ(ep.RegisterUser(region.data() + 4096, 4096), remote_write)
      << "range lookup must prefer the remote-write access upgrade";
}

// The host KV pool belongs to the PD, not to a connection: many endpoints on
// the same device must register it ONCE (the #P1-2 fix), not once per connection
// (the pre-fix storm — re-running ibv_reg_mr over a tens-of-GB region on every
// reconnect). Verified via the pool-registration counter delta, and by every
// endpoint resolving an in-pool buffer to the SAME MR (shared lkey).
TEST(RdmaLoopback, PoolMrRegisteredOncePerDeviceNotPerConnection) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  // A distinct region per test run so the counter delta is attributable here.
  std::vector<char> region(512 * 1024);
  const uint64_t regs0 = rdma::RcEndpoint::PoolMrRegistrations();
  const uint64_t adhoc0 = rdma::RcEndpoint::AdhocUserMrTotal();

  constexpr int N = 8;
  std::vector<std::unique_ptr<rdma::RcEndpoint>> eps;
  ibv_mr* first = nullptr;
  for (int i = 0; i < N; ++i) {
    eps.push_back(std::make_unique<rdma::RcEndpoint>());
    ASSERT_TRUE(eps.back()->Open(nullptr, 64 * 1024, 1)) << "endpoint " << i;
    ASSERT_TRUE(eps.back()->AddPoolMr(region.data(), region.size()));
    ibv_mr* m = eps.back()->RegisterUser(region.data() + 4096, 4096);
    ASSERT_NE(m, nullptr);
    if (i == 0) first = m;
    else EXPECT_EQ(m, first) << "all endpoints on one PD share the pool MR";
  }
  EXPECT_EQ(rdma::RcEndpoint::PoolMrRegistrations() - regs0, 1u)
      << N << " endpoints registered the region " << (rdma::RcEndpoint::PoolMrRegistrations() - regs0)
      << " times; must be exactly once per device";
  EXPECT_EQ(rdma::RcEndpoint::AdhocUserMrTotal() - adhoc0, 0u)
      << "in-pool buffers must never take the ad-hoc registration path";

  eps.clear();  // all close; region MR freed when the last device ref drops
  // A fresh endpoint re-registers the (now-freed) region: counter advances by 1.
  rdma::RcEndpoint again;
  ASSERT_TRUE(again.Open(nullptr, 64 * 1024, 1));
  ASSERT_TRUE(again.AddPoolMr(region.data(), region.size()));
}

// End-to-end: register one host pool, then PUT from and GET into sub-buffers.
// Every transfer must use the pool MR; no per-operation registration is legal.
TEST(RdmaLoopback, RegisterMemoryRoundtrip) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("rmr");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  std::vector<char> pool(128 * 1024);
  ASSERT_TRUE(c.RegisterMemory(pool.data(), pool.size()));

  const size_t sz = 4096;
  for (int i = 0; i < 8; ++i) {
    char* src = pool.data() + i * sz * 2;            // distinct sub-buffer per page
    char* dst = pool.data() + i * sz * 2 + sz;       // get into a neighbouring slot
    for (size_t k = 0; k < sz; ++k) src[k] = static_cast<char>((i * 17 + k) & 0xFF);
    std::string key = "rm" + std::to_string(i);
    ASSERT_TRUE(c.Put(key, src, sz)) << i;
    ASSERT_TRUE(c.Get(key, dst, sz)) << i;
    EXPECT_EQ(0, std::memcmp(src, dst, sz)) << i;
  }
}

TEST(RdmaLoopback, ConnectionPoolAndKeepaliveDefaultsResolveAndDisable) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ::unsetenv("DFKV_RDMA_KEEPALIVE_MS");
  ::unsetenv("DFKV_RDMA_POOL_MAX");
  config_dump::ResetForTest();
  testing::internal::CaptureStderr();
  { RdmaTransport rt(kMaxMsg); }
  config_dump::Emit("keepalive-default-test");
  std::string output = testing::internal::GetCapturedStderr();
  EXPECT_NE(output.find("DFKV_RDMA_KEEPALIVE_MS"), std::string::npos);
  EXPECT_NE(output.find(" = 15000  (default)"), std::string::npos);
  const size_t pool_name = output.find("DFKV_RDMA_POOL_MAX");
  ASSERT_NE(pool_name, std::string::npos);
  const std::string pool_line =
      output.substr(pool_name, output.find('\n', pool_name) - pool_name);
  EXPECT_NE(pool_line.find(" = 16  (default)"), std::string::npos);

  ::setenv("DFKV_RDMA_KEEPALIVE_MS", "0", 1);
  ::setenv("DFKV_RDMA_POOL_MAX", "2", 1);
  config_dump::ResetForTest();
  testing::internal::CaptureStderr();
  { RdmaTransport rt(kMaxMsg); }
  config_dump::Emit("keepalive-disabled-test");
  output = testing::internal::GetCapturedStderr();
  EXPECT_NE(output.find("DFKV_RDMA_KEEPALIVE_MS"), std::string::npos);
  EXPECT_NE(output.find(" = 0  (env)"), std::string::npos);
  const size_t pool_override_name = output.find("DFKV_RDMA_POOL_MAX");
  ASSERT_NE(pool_override_name, std::string::npos);
  const std::string pool_override_line =
      output.substr(pool_override_name,
                    output.find('\n', pool_override_name) - pool_override_name);
  EXPECT_NE(pool_override_line.find(" = 2  (env)"), std::string::npos);
  ::unsetenv("DFKV_RDMA_KEEPALIVE_MS");
  ::unsetenv("DFKV_RDMA_POOL_MAX");
}

// A live client must not inherit the server's short dead-client reap interval
// as a first-read reconnect penalty. Idle QPs receive lightweight membership
// probes; a dead process sends none and remains reclaimable by the server.
TEST(RdmaLoopback, KeepalivePreservesIdleDataConnection) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ::setenv("DFKV_RDMA_IDLE_MS", "150", 1);
  ::setenv("DFKV_RDMA_KEEPALIVE_MS", "30", 1);
  {
    RdmaNode node("keepalive");
    RdmaTransport rt(kMaxMsg);
    KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
    std::string value(4096, 'k');
    std::string out(value.size(), '\0');
    EXPECT_TRUE(c.Put("alive", value.data(), value.size()));
    EXPECT_TRUE(c.Get("alive", out.data(), out.size()));
    EXPECT_EQ(out, value);

    const long stale_before = CounterVal(
        rt.MetricsText(), "dfkv_rdma_client_stale_pool_retries_total");
    std::this_thread::sleep_for(std::chrono::milliseconds(450));
    out.assign(value.size(), '\0');
    EXPECT_TRUE(c.Get("alive", out.data(), out.size()));
    EXPECT_EQ(out, value);
    const std::string metrics = rt.MetricsText();
    EXPECT_GT(CounterVal(
                  metrics, "dfkv_rdma_client_keepalive_successes_total"),
              0);
    EXPECT_EQ(CounterVal(
                  metrics, "dfkv_rdma_client_stale_pool_retries_total"),
              stale_before);
    EXPECT_EQ(CounterVal(
                  metrics, "dfkv_rdma_client_keepalive_failures_total"),
              0);
  }
  ::unsetenv("DFKV_RDMA_KEEPALIVE_MS");
  ::unsetenv("DFKV_RDMA_IDLE_MS");
}

// A zero-length item in a mixed RDMA batch fails independently; the valid
// neighbor still completes and no empty object reaches the server.
TEST(RdmaLoopback, BatchPutRejectsEmptyValue) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("bev");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  std::string nonempty(2048, 'x');
  std::vector<KvPutItem> puts = {
      {"e_full", nonempty.data(), nonempty.size()},
      {"e_empty", nonempty.data(), 0},
  };
  auto pr = c.BatchPut(puts);
  ASSERT_TRUE(pr[0]);
  EXPECT_FALSE(pr[1]);

  EXPECT_FALSE(c.Exist("e_empty"));
  std::string out(nonempty.size(), '\0');
  ASSERT_TRUE(c.Get("e_full", &out[0], out.size()));
  EXPECT_EQ(out, nonempty);
}

// Scatter-gather over RDMA: one logical object may span any number of caller
// descriptors. The transport windows that descriptor stream to the negotiated
// HCA SGE limit without changing object identity or byte order.
TEST(RdmaLoopback, ScatterGatherRoundtripOverRdma) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  // The configured ordinary-data depth stays >1; the dedicated SG lane must
  // nevertheless use depth-one windows.
  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");
  RdmaNode node("sg");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  ASSERT_TRUE(rt.pipelined());  // RDMA path, not native TCP

  auto make_chunks = [](const std::string& tag, const std::vector<size_t>& sizes) {
    std::vector<std::string> v(sizes.size());
    for (size_t i = 0; i < sizes.size(); ++i) {
      v[i].resize(sizes[i]);
      for (size_t b = 0; b < sizes[i]; ++b)
        v[i][b] = static_cast<char>((tag.size() + i * 31 + b * 7) & 0xFF);
    }
    return v;
  };
  auto roundtrip = [&](const std::string& key, const std::vector<size_t>& sizes) {
    auto src = make_chunks(key, sizes);
    std::vector<const void*> ptrs;
    for (auto& s : src) ptrs.push_back(s.data());
    KvPutItemSg put{key, ptrs, sizes};
    const auto pr = c.BatchPutSg({put});
    ASSERT_EQ(pr.size(), 1u);
    ASSERT_TRUE(pr[0]) << "put " << key;
    std::vector<std::string> dst(sizes.size());
    std::vector<void*> dptrs;
    for (size_t i = 0; i < sizes.size(); ++i) {
      dst[i].assign(sizes[i], '\0');
      dptrs.push_back(dst[i].data());
    }
    KvGetItemSg get{key, dptrs, sizes};
    std::vector<size_t> lens;
    const auto gr = c.BatchGetAutoSg({get}, &lens);
    ASSERT_EQ(gr.size(), 1u);
    ASSERT_TRUE(gr[0]) << "get " << key;
    size_t total = 0; for (size_t s : sizes) total += s;
    EXPECT_EQ(lens[0], total);
    for (size_t i = 0; i < sizes.size(); ++i) EXPECT_EQ(dst[i], src[i]) << key << " seg " << i;
  };

  const size_t max_sg = rt.MaxSgPayloadSegs();
  ASSERT_GE(max_sg, 2u);
  roundtrip("sg_n1", std::vector<size_t>(1, 4096));
  roundtrip("sg_n2", std::vector<size_t>(2, 2048));
  roundtrip("sg_exact_max", std::vector<size_t>(max_sg, 256));
  roundtrip("sg_max_plus_one", std::vector<size_t>(max_sg + 1, 193));
  roundtrip("sg_multi_window", std::vector<size_t>(max_sg * 3 + 5, 71));
  roundtrip("sg_var", {1, 7, 64, 333, 4096, 11});
  roundtrip("sg_zero_descriptors", {17, 0, 31, 0, 9});

  // Insufficient aggregate capacity is rejected before any destination byte is
  // touched, even when both the value and destination cross WR boundaries.
  {
    std::vector<size_t> sizes(max_sg + 3, 32);
    auto src = make_chunks("sg_small_cap", sizes);
    std::vector<const void*> ptrs;
    for (auto& s : src) ptrs.push_back(s.data());
    const auto put = c.BatchPutSg({{"sg_small_cap", ptrs, sizes}});
    ASSERT_EQ(put.size(), 1u);
    ASSERT_TRUE(put[0]);

    std::vector<size_t> caps = sizes;
    ASSERT_FALSE(caps.empty());
    --caps.back();
    std::vector<std::string> dst(caps.size());
    std::vector<void*> dptrs;
    for (size_t i = 0; i < caps.size(); ++i) {
      dst[i].assign(caps[i], '\x5a');
      dptrs.push_back(dst[i].data());
    }
    std::vector<size_t> lens;
    const auto get = c.BatchGetAutoSg(
        {{"sg_small_cap", dptrs, caps}}, &lens);
    ASSERT_EQ(get.size(), 1u);
    EXPECT_FALSE(get[0]);
    ASSERT_EQ(lens.size(), 1u);
    EXPECT_EQ(lens[0], 0u);
    for (const auto& segment : dst)
      EXPECT_EQ(segment, std::string(segment.size(), '\x5a'));

    std::string contiguous;
    for (const auto& segment : src) contiguous += segment;
    std::string out(contiguous.size(), '\0');
    ASSERT_TRUE(c.Get("sg_small_cap", out.data(), out.size()));
    EXPECT_EQ(out, contiguous);
  }

  // Multi-key fan-out crosses the SG lane's depth-one send window. Pin client
  // concurrency to 1: Soft-RoCE (rdma_rxe) loopback races when many distinct
  // QPs run in parallel (the same flakiness affects contiguous BatchGetAuto).
  // This still exercises the real multi-SGE verbs datapath across many keys.
  {
    c.set_batch_concurrency(1);
    const int N = 24;
    std::vector<std::vector<std::string>> srcs(N);
    std::vector<KvPutItemSg> puts(N);
    for (int i = 0; i < N; ++i) {
      std::vector<size_t> sizes(1 + (i % 4), 64 + (i % 8));
      srcs[i] = make_chunks("sgm" + std::to_string(i), sizes);
      std::vector<const void*> ptrs; for (auto& s : srcs[i]) ptrs.push_back(s.data());
      puts[i] = {"sgm" + std::to_string(i), ptrs, sizes};
    }
    auto pr = c.BatchPutSg(puts);
    ASSERT_EQ(pr.size(), static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) ASSERT_TRUE(pr[i]) << i;
    std::vector<std::vector<std::string>> dsts(N);
    std::vector<KvGetItemSg> gets(N);
    for (int i = 0; i < N; ++i) {
      dsts[i].resize(srcs[i].size());
      std::vector<void*> dptrs; std::vector<size_t> caps(srcs[i].size());
      for (size_t j = 0; j < srcs[i].size(); ++j) {
        dsts[i][j].assign(srcs[i][j].size(), '\0');
        dptrs.push_back(&dsts[i][j][0]); caps[j] = srcs[i][j].size();
      }
      gets[i] = {"sgm" + std::to_string(i), dptrs, caps};
    }
    std::vector<size_t> lens;
    auto gr = c.BatchGetAutoSg(gets, &lens);
    ASSERT_EQ(gr.size(), static_cast<size_t>(N));
    ASSERT_EQ(lens.size(), static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
      ASSERT_TRUE(gr[i]) << i;
      size_t expected_len = 0;
      for (size_t j = 0; j < srcs[i].size(); ++j) {
        expected_len += srcs[i][j].size();
        EXPECT_EQ(dsts[i][j], srcs[i][j]) << i << " " << j;
      }
      EXPECT_EQ(lens[i], expected_len) << i;
    }
  }
}

TEST(RdmaLoopback, MultiWrLaterWindowFailureIsAtomicAndReclaimsState) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv idle_reclaim("DFKV_RDMA_IDLE_MS", "2000");
  RdmaNode node("sgabort");
  const uint64_t transient_baseline =
      rdma::RcEndpoint::TransientUserMrActive();
  const uint64_t active_baseline = node.rsrv->ActiveConns();
  const auto wait_for_active_at_most = [&node](uint64_t limit) {
    for (int i = 0; i < 2000 && node.rsrv->ActiveConns() > limit; ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return node.rsrv->ActiveConns();
  };

  std::vector<std::string> src;
  std::vector<const void*> ptrs;
  std::vector<size_t> sizes;
  {
    RdmaTransport rt(kMaxMsg);
    const size_t max_sg = rt.MaxSgPayloadSegs();
    ASSERT_GE(max_sg, 2u);
    src.resize(max_sg + 7);
    for (size_t i = 0; i < src.size(); ++i) {
      src[i].assign(97 + i % 11, static_cast<char>('a' + i % 26));
      ptrs.push_back(src[i].data());
      sizes.push_back(src[i].size());
    }
    KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
    const uint64_t active_before_fault = node.rsrv->ActiveConns();
    const uint64_t opened_before_fault = node.rsrv->V2Conns();
    std::vector<bool> failed;
    {
      ScopedEnv abort_window("DFKV_RDMA_TEST_ABORT_MULTIWR_WINDOW", "2");
      failed = c.BatchPutSg(
          {{"sg_later_window_abort", ptrs, sizes}});
    }
    ASSERT_EQ(failed.size(), 1u);
    EXPECT_FALSE(failed[0]);
    const uint64_t opened_delta =
        node.rsrv->V2Conns() - opened_before_fault;
    ASSERT_GT(opened_delta, 0u);
    // ActiveConns is a server-side gauge: destroying an RC peer may leave its
    // silent server half alive until the idle reaper runs.  Allow every healthy
    // connection opened by the operation, but require the faulted QP itself to
    // disappear.  One leaked QP therefore exceeds this bound by exactly one.
    const uint64_t max_active_after_fault =
        active_before_fault + opened_delta - 1;
    EXPECT_LE(wait_for_active_at_most(max_active_after_fault),
              max_active_after_fault);
    EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), transient_baseline);
  }

  EXPECT_LE(wait_for_active_at_most(active_baseline), active_baseline);
  EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), transient_baseline);

  EXPECT_EQ(node.srv->Count(), 0u);
  // A failed logical PUT never commits a readable prefix. Once fault injection
  // is removed, the same object succeeds and round-trips on a fresh connection.
  {
    RdmaTransport rt(kMaxMsg);
    KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
    EXPECT_FALSE(c.Exist("sg_later_window_abort"));
    const auto put = c.BatchPutSg(
        {{"sg_later_window_abort", ptrs, sizes}});
    ASSERT_EQ(put.size(), 1u);
    ASSERT_TRUE(put[0]);
    std::vector<std::string> dst(src.size());
    std::vector<void*> dptrs;
    for (size_t i = 0; i < src.size(); ++i) {
      dst[i].assign(src[i].size(), '\0');
      dptrs.push_back(dst[i].data());
    }
    std::vector<size_t> lens;
    const auto get = c.BatchGetAutoSg(
        {{"sg_later_window_abort", dptrs, sizes}}, &lens);
    ASSERT_EQ(get.size(), 1u);
    ASSERT_TRUE(get[0]);
    ASSERT_EQ(lens.size(), 1u);
    size_t expected_len = 0;
    for (size_t i = 0; i < src.size(); ++i) {
      expected_len += src[i].size();
      EXPECT_EQ(dst[i], src[i]) << i;
    }
    EXPECT_EQ(lens[0], expected_len);
  }
  EXPECT_LE(wait_for_active_at_most(active_baseline), active_baseline);
  EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), transient_baseline);
  node.rsrv->Stop();
  EXPECT_EQ(node.rsrv->ActiveConns(), 0u);
}

TEST(RdmaLoopback, ScatterGatherGetUsesDedicatedPool) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ::setenv("DFKV_RDMA_DEPTH", "4", 1);
  RdmaNode node("sggetlane");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  std::string value = "scatter-get-payload";
  ASSERT_TRUE(c.Put("sg_get_seed", value.data(), value.size()));
  const long data_opened =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  ASSERT_EQ(data_opened, 1);

  std::string first(7, '\0');
  std::string second(value.size() - first.size(), '\0');
  KvGetItemSg get{"sg_get_seed", {first.data(), second.data()},
                  {first.size(), second.size()}};
  std::vector<size_t> lengths;
  ASSERT_TRUE(c.BatchGetAutoSg({get}, &lengths)[0]);
  ASSERT_EQ(lengths[0], value.size());
  EXPECT_EQ(first + second, value);
  EXPECT_EQ(CounterVal(rt.MetricsText(),
                       "dfkv_rdma_client_conns_opened_total"),
            data_opened + 1)
      << "SG GET reused the ordinary data pool instead of opening its lane";

  ::unsetenv("DFKV_RDMA_DEPTH");
}

TEST(RdmaLoopback, ScatterGatherPutReturnsToDedicatedPool) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ::setenv("DFKV_RDMA_DEPTH", "4", 1);
  RdmaNode node("sgputlane");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  std::string first = "scatter-";
  std::string second = "put-payload";
  KvPutItemSg put{"sg_put_seed", {first.data(), second.data()},
                  {first.size(), second.size()}};
  ASSERT_TRUE(c.BatchPutSg({put})[0]);
  const long sg_opened =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  ASSERT_EQ(sg_opened, 1);

  std::string ordinary = "ordinary-data";
  ASSERT_TRUE(c.Put("ordinary_after_sg", ordinary.data(), ordinary.size()));
  EXPECT_EQ(CounterVal(rt.MetricsText(),
                       "dfkv_rdma_client_conns_opened_total"),
            sg_opened + 1)
      << "SG PUT returned its depth-one connection to the ordinary data pool";

  ::unsetenv("DFKV_RDMA_DEPTH");
}

// Fix 1 regression: an oversized SG key (total payload > max_payload_, but with a
// legal segment count so it passes the client guard) must fail ONLY itself inside
// CacheFromMulti/RangeIntoMulti, NOT poison its node batch. Previously the up-front
// validation std::fill'd every result kInvalid and returned; now the offender is
// skipped in the window and its siblings on the same node proceed normally.
// Regression: one deep window of SG items where every segment is a distinct
// unregistered buffer. All MRs must remain live until the whole posted window
// completes, then be released before the public call returns. This exercises
// depth * (max_sge-1) simultaneous transient registrations across PUT and GET.
TEST(RdmaLoopback, SgDeepWindowManyUnregisteredSegmentsSafe) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ::setenv("DFKV_RDMA_DEPTH", "4", 1);
  RdmaNode node("sgdw");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  ASSERT_TRUE(rt.pipelined());
  const uint64_t transient_baseline =
      rdma::RcEndpoint::TransientUserMrActive();

  constexpr int kItems = 8;   // > depth: also crosses windows
  constexpr int kSegs = 29;   // max payload SGEs per slot
  constexpr size_t kSeg = 512;
  std::vector<std::vector<std::string>> src(kItems);
  std::vector<KvPutItemSg> puts;
  for (int it = 0; it < kItems; ++it) {
    src[it].resize(kSegs);
    std::vector<const void*> ptrs;
    std::vector<size_t> sizes;
    for (int sg = 0; sg < kSegs; ++sg) {
      src[it][sg].assign(kSeg, static_cast<char>('a' + (it * 7 + sg) % 26));
      ptrs.push_back(src[it][sg].data());
      sizes.push_back(kSeg);
    }
    puts.push_back({"sgdw_" + std::to_string(it), ptrs, sizes});
  }
  auto pr = c.BatchPutSg(puts);
  ASSERT_EQ(pr.size(), puts.size());
  for (int it = 0; it < kItems; ++it) EXPECT_TRUE(pr[it]) << "put item " << it;
  EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), transient_baseline);

  std::vector<std::vector<std::string>> dst(kItems);
  std::vector<KvGetItemSg> gets;
  for (int it = 0; it < kItems; ++it) {
    dst[it].assign(kSegs, std::string(kSeg, '\0'));
    std::vector<void*> dptrs;
    std::vector<size_t> caps;
    for (int sg = 0; sg < kSegs; ++sg) { dptrs.push_back(&dst[it][sg][0]); caps.push_back(kSeg); }
    gets.push_back({"sgdw_" + std::to_string(it), dptrs, caps});
  }
  std::vector<size_t> lens;
  auto gr = c.BatchGetAutoSg(gets, &lens);
  ASSERT_EQ(gr.size(), gets.size());
  for (int it = 0; it < kItems; ++it) {
    EXPECT_TRUE(gr[it]) << "get item " << it;
    EXPECT_EQ(lens[it], kSegs * kSeg) << it;
    for (int sg = 0; sg < kSegs; ++sg)
      EXPECT_EQ(dst[it][sg], src[it][sg]) << "item " << it << " seg " << sg;
  }
  EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), transient_baseline);
  ::unsetenv("DFKV_RDMA_DEPTH");
}

TEST(RdmaLoopback, ScatterGatherOversizedFailsOnlyOffender) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ::setenv("DFKV_RDMA_DEPTH", "4", 1);
  RdmaNode node("sgov");
  RdmaTransport rt(kMaxMsg);  // max_payload_ = 256 KiB
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  c.set_batch_concurrency(1);  // single node, stable on rxe loopback
  ASSERT_TRUE(rt.pipelined());

  auto fill = [](std::string& s, int seed) {
    for (size_t b = 0; b < s.size(); ++b) s[b] = static_cast<char>((seed + b * 7) & 0xFF);
  };

  // Valid sibling A, an oversized item (2 segs * 200 KiB = 400 KiB > 256 KiB max_payload,
  // only 2 segments so it clears the <=29-seg client guard), then valid sibling B —
  // all routed to the same (only) node. The offender must report failure; both
  // siblings must succeed and round-trip byte-exact.
  std::string a0(4096, '\0'); fill(a0, 11);
  std::string b0(8192, '\0'); fill(b0, 23);
  std::string big0(200 * 1024, '\0'), big1(200 * 1024, '\0');
  fill(big0, 31); fill(big1, 37);

  std::vector<KvPutItemSg> puts = {
      {"sgov_a", {a0.data()}, {a0.size()}},
      {"sgov_big", {big0.data(), big1.data()}, {big0.size(), big1.size()}},
      {"sgov_b", {b0.data()}, {b0.size()}},
  };
  auto pr = c.BatchPutSg(puts);
  EXPECT_TRUE(pr[0]) << "sibling A put";
  EXPECT_FALSE(pr[1]) << "oversized put must fail only itself";
  EXPECT_TRUE(pr[2]) << "sibling B put (must not be poisoned by the offender)";

  // The two siblings must have been stored; the oversized key must be absent (never
  // written). Drive the GET-side oversized path too: a get whose total cap exceeds
  // max_payload must miss without poisoning its siblings.
  std::string ga(4096, '\0'), gb(8192, '\0');
  std::string gbig0(200 * 1024, '\0'), gbig1(200 * 1024, '\0');
  std::vector<KvGetItemSg> gets = {
      {"sgov_a", {&ga[0]}, {ga.size()}},
      {"sgov_big", {&gbig0[0], &gbig1[0]}, {gbig0.size(), gbig1.size()}},
      {"sgov_b", {&gb[0]}, {gb.size()}},
  };
  std::vector<size_t> lens;
  auto gr = c.BatchGetAutoSg(gets, &lens);
  EXPECT_TRUE(gr[0]) << "sibling A get";
  EXPECT_FALSE(gr[1]) << "oversized get must miss only itself";
  EXPECT_TRUE(gr[2]) << "sibling B get (must not be poisoned by the offender)";
  if (gr[0]) { EXPECT_EQ(ga, a0); }
  if (gr[2]) { EXPECT_EQ(gb, b0); }

  ::unsetenv("DFKV_RDMA_DEPTH");
}

TEST(RdmaLoopback, PipelinedPoolDepth) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  // depth>1 enables client pipelining + the server GET worker pool (the most
  // concurrency-heavy path). Env is read by both the client ctor and the server
  // serve loop in this single process. Run under TSan to catch races.
  ::setenv("DFKV_RDMA_DEPTH", "4", 1);
  ::setenv("DFKV_RDMA_WORKERS", "4", 1);
  RdmaNode node("ppd");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  const int N = 64;
  const size_t sz = 4096;
  std::vector<std::string> vals(N), keys(N);
  std::vector<KvPutItem> puts(N);
  for (int i = 0; i < N; ++i) {
    keys[i] = "p" + std::to_string(i);
    vals[i].assign(sz, static_cast<char>((i * 7 + 3) & 0xFF));
    puts[i] = {keys[i], vals[i].data(), sz};
  }
  auto pr = c.BatchPut(puts);
  for (int i = 0; i < N; ++i) ASSERT_TRUE(pr[i]) << i;

  std::vector<std::string> outs(N);
  std::vector<KvGetItem> gets(N);
  for (int i = 0; i < N; ++i) { outs[i].assign(sz, '\0'); gets[i] = {keys[i], &outs[i][0], sz}; }
  auto gr = c.BatchGet(gets);
  int hits = 0;
  for (int i = 0; i < N; ++i) { if (gr[i]) { ++hits; EXPECT_EQ(outs[i], vals[i]) << i; } }
  EXPECT_EQ(hits, N);

  ::unsetenv("DFKV_RDMA_DEPTH");
  ::unsetenv("DFKV_RDMA_WORKERS");
}

// Regression for the conn-thread leak (#3). A server Serve thread blocks in
// WaitComp forever after a silent client disconnect (a torn-down RC peer yields
// no completion), so without an idle timeout a long-running server accumulates
// one live thread per lifetime connection — Stop() is the only reaper. The fix:
// an idle timeout reclaims the connection (the thread returns), and ReapDoneLocked
// joins the finished thread on the next accept. This test sets a short idle window
// and verifies the live count drains back to the baseline; without the fix it
// would stay pinned near N and time out.
TEST(RdmaLoopback, ReclaimsAndReapsIdleConnThreads) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ::setenv("DFKV_RDMA_IDLE_MS", "120", 1);  // reclaim idle conns fast (test only)
  constexpr size_t kSmallMsg = 16 * 1024;   // stay well under an 8 MiB memlock
  RdmaNode node("reap", kSmallMsg);          // server buffers must be small too
  // One long-lived client endpoint holds the shared per-device ibv_context open
  // (its server side may be reclaimed when idle and is transparently re-dialed).
  RdmaTransport keep(kSmallMsg);
  KVClient ckeep({{"n", node.addr}}, SelfHdr(), &keep);
  std::string kk(64, 'k');
  ASSERT_TRUE(ckeep.Put("keep", kk.data(), kk.size()));

  const int N = 30;
  for (int i = 0; i < N; ++i) {
    RdmaTransport rt(kSmallMsg);
    KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
    std::string v(512, static_cast<char>(i & 0xFF));
    ASSERT_TRUE(c.Put("k" + std::to_string(i), v.data(), v.size())) << "i=" << i;
    // rt + c leave scope here -> transient client gone -> server conn goes idle.
  }

  // Each round: wait past the idle window so every prior connection's Serve thread
  // has exited, then make ONE connection whose accept runs ReapDoneLocked to join
  // all the finished threads. Only that single in-flight drain thread should then
  // remain. Without reclaim+reap the N transient threads stay in conns_ and the
  // count never falls (the loop exhausts its rounds and the assert fails).
  size_t live = 0;
  for (int round = 0; round < 6; ++round) {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));  // > idle (120 ms)
    { RdmaTransport rt(kSmallMsg); KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
      std::string v(16, 'x'); c.Put("drain", v.data(), v.size()); }
    live = node.rsrv->live_conn_count();
    if (live <= 2) break;
  }
  EXPECT_LE(live, 2u) << "idle conn threads were not reclaimed + reaped (leak)";
  ::unsetenv("DFKV_RDMA_IDLE_MS");
}

// Regression: a BATCH op whose pooled connection the server reclaimed on idle
// must transparently re-dial and succeed — NOT return kIOError for the batch.
// Before the fix only the single-op RoundTrip retried a stale pooled conn; the
// batch paths (CacheMany/CacheFrom/RangeInto/ExistMany/...) gave up on the first
// failed window, which surfaced to SGLang as "Write page to storage: N pages
// failed" on writes and 0-hit prefixes on reads after an idle gap.
TEST(RdmaLoopback, BatchRetriesAfterServerReclaim) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ::setenv("DFKV_RDMA_IDLE_MS", "120", 1);   // server reclaims idle conns fast
  RdmaNode node("brc");
  RdmaTransport rt(kMaxMsg);                   // long-lived: holds the device ctx
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  const int N = 24;
  for (int i = 0; i < N; ++i) {
    std::string v = "val" + std::to_string(i);
    ASSERT_TRUE(c.Put("b" + std::to_string(i), v.data(), v.size())) << "warm i=" << i;
  }
  // Warm the pool so a connection is parked for the node, then snapshot the dial
  // counter — the batch op below must open a NEW one when it finds it stale.
  EXPECT_TRUE(c.Exist("b0"));
  long opened_warm =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  ASSERT_GE(opened_warm, 1);

  // Let the server reclaim the now-idle pooled connection (idle window 120 ms).
  std::this_thread::sleep_for(std::chrono::milliseconds(400));

  // The pooled conn is stale (its server peer was reclaimed). A batch existence
  // probe must detect that on the first window and re-dial a fresh conn, returning
  // CORRECT results — not kIOError for the whole batch. Direct transport call so no
  // KVClient-level health retry can mask a transport that failed to recover.
  // Probe with the same canonical namespace/object identity as the warm writes.
  std::vector<BlockKey> keys;
  for (int i = 0; i < N; ++i) {
    keys.push_back(ToBlockKey(SelfHdr(), "b" + std::to_string(i)));
    keys.push_back(ToBlockKey(SelfHdr(), "nope" + std::to_string(i)));
  }
  std::vector<char> exists;
  auto sts = rt.ExistMany(node.addr, keys, &exists);
  ASSERT_EQ(exists.size(), keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    EXPECT_NE(sts[i], Status::kIOError) << "i=" << i << " (stale conn not retried)";
    EXPECT_EQ(exists[i] != 0, (i % 2 == 0)) << "i=" << i;
  }
  // The retry re-dialed: a new client connection was opened after the warm-up.
  long opened_after =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  EXPECT_GT(opened_after, opened_warm) << "batch op did not re-dial after reclaim";

  ::unsetenv("DFKV_RDMA_IDLE_MS");
}

#ifdef DFKV_WITH_URING
class ControlledRdmaUringBackend final : public UringReader::Backend {
 public:
  struct Request {
    int fd = -1;
    void* buf = nullptr;
    unsigned len = 0;
    uint64_t off = 0;
    UringReader::Token token = UringReader::kInvalidToken;
  };

  int QueueInit(unsigned entries, io_uring*) override {
    std::lock_guard<std::mutex> lock(mu_);
    initialized_depth_ = entries;
    return queue_init_result;
  }

  void QueueExit(io_uring*) override {
    std::lock_guard<std::mutex> lock(mu_);
    exited_ = true;
    cv_.notify_all();
  }

  io_uring_sqe* GetSqe(io_uring*) override {
    std::lock_guard<std::mutex> lock(mu_);
    sqes_.emplace_back();
    return &sqes_.back();
  }

  void PrepRead(io_uring_sqe* sqe, int fd, void* buf, unsigned len,
                uint64_t off) override {
    std::lock_guard<std::mutex> lock(mu_);
    prepared_[sqe] =
        Request{fd, buf, len, off, UringReader::kInvalidToken};
  }

  void SetData(io_uring_sqe* sqe, UringReader::Token token) override {
    std::lock_guard<std::mutex> lock(mu_);
    auto found = prepared_.find(sqe);
    if (found == prepared_.end()) {
      bad_state_ = true;
      return;
    }
    found->second.token = token;
    dormant_.push_back(found->second);
    prepared_.erase(found);
  }

  int Submit(io_uring*) override {
    std::lock_guard<std::mutex> lock(mu_);
    ++submit_calls_;
    int result = static_cast<int>(dormant_.size());
    if (!submit_results_.empty()) {
      result = submit_results_.front();
      submit_results_.pop_front();
    }
    if (result > 0 && static_cast<size_t>(result) <= dormant_.size()) {
      for (int i = 0; i < result; ++i) {
        Request request = dormant_.front();
        dormant_.pop_front();
        accepted_[request.token] = request;
        history_.push_back(request);
      }
      max_inflight_ = std::max(max_inflight_, accepted_.size());
    }
    cv_.notify_all();
    return result;
  }

  int PeekCqe(io_uring*, io_uring_cqe** cqe) override {
    std::lock_guard<std::mutex> lock(mu_);
    if (ready_.empty()) {
      *cqe = nullptr;
      return -EAGAIN;
    }
    *cqe = &ready_.front();
    return 0;
  }

  int WaitCqe(io_uring*, io_uring_cqe** cqe, int timeout_ms) override {
    std::unique_lock<std::mutex> lock(mu_);
    const auto ready = [&] { return !ready_.empty() || exited_; };
    if (timeout_ms < 0) {
      cv_.wait(lock, ready);
    } else if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                             ready)) {
      *cqe = nullptr;
      return -ETIME;
    }
    if (ready_.empty()) {
      *cqe = nullptr;
      return -ETIME;
    }
    *cqe = &ready_.front();
    return 0;
  }

  void Seen(io_uring*, io_uring_cqe* cqe) override {
    std::lock_guard<std::mutex> lock(mu_);
    if (ready_.empty() || cqe != &ready_.front()) {
      bad_state_ = true;
      return;
    }
    ready_.pop_front();
    ++seen_;
    cv_.notify_all();
  }

  void SetSubmitResults(std::deque<int> results) {
    std::lock_guard<std::mutex> lock(mu_);
    submit_results_ = std::move(results);
  }

  bool WaitForSubmitted(size_t count, int timeout_ms = 5000) {
    std::unique_lock<std::mutex> lock(mu_);
    return cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                        [&] { return history_.size() >= count; });
  }

  bool WaitForSubmitCalls(size_t count, int timeout_ms = 5000) {
    std::unique_lock<std::mutex> lock(mu_);
    return cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                        [&] { return submit_calls_ >= count; });
  }

  bool WaitForSeen(size_t count, int timeout_ms = 5000) {
    std::unique_lock<std::mutex> lock(mu_);
    return cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                        [&] { return seen_ >= count; });
  }

  std::vector<Request> History() const {
    std::lock_guard<std::mutex> lock(mu_);
    return history_;
  }

  bool CompleteRead(UringReader::Token token) {
    Request request;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto found = accepted_.find(token);
      if (found == accepted_.end()) return false;
      request = found->second;
    }
    const ssize_t result =
        ::pread(request.fd, request.buf, request.len,
                static_cast<off_t>(request.off));
    return Complete(token, result);
  }

  bool Complete(UringReader::Token token, long result) {
    std::lock_guard<std::mutex> lock(mu_);
    auto found = accepted_.find(token);
    if (found == accepted_.end()) return false;
    accepted_.erase(found);
    io_uring_cqe cqe{};
    cqe.user_data = token;
    cqe.res = static_cast<int32_t>(result);
    ready_.push_back(cqe);
    cv_.notify_all();
    return true;
  }

  size_t MaxInflight() const {
    std::lock_guard<std::mutex> lock(mu_);
    return max_inflight_;
  }

  size_t SeenCount() const {
    std::lock_guard<std::mutex> lock(mu_);
    return seen_;
  }

  size_t AcceptedCount() const {
    std::lock_guard<std::mutex> lock(mu_);
    return accepted_.size();
  }

  bool Exited() const {
    std::lock_guard<std::mutex> lock(mu_);
    return exited_;
  }

  bool BadState() const {
    std::lock_guard<std::mutex> lock(mu_);
    return bad_state_;
  }

  int queue_init_result = 0;

 private:
  mutable std::mutex mu_;
  std::condition_variable cv_;
  unsigned initialized_depth_ = 0;
  size_t submit_calls_ = 0;
  size_t seen_ = 0;
  size_t max_inflight_ = 0;
  bool exited_ = false;
  bool bad_state_ = false;
  std::list<io_uring_sqe> sqes_;
  std::map<io_uring_sqe*, Request> prepared_;
  std::deque<Request> dormant_;
  std::map<UringReader::Token, Request> accepted_;
  std::vector<Request> history_;
  std::list<io_uring_cqe> ready_;
  std::deque<int> submit_results_;
};
#endif

// A cache node that ALSO wires the async-GET prep + complete hooks, so the server
// uses the nonblocking io_uring GET pipeline when DFKV_SERVER_URING=1 (and the
// binary was built with -DDFKV_WITH_URING). With the env off / unbuilt these
// hooks are simply never consulted and the node behaves like a plain RdmaNode.
struct RdmaUringNode {
  fs::path dir;
  std::unique_ptr<KvNodeServer> srv;
  std::unique_ptr<RdmaServer> rsrv;
  std::string addr;
  mutable std::mutex observation_mu;
  std::map<std::string, size_t> prepare_calls;
  std::map<std::string, size_t> range_calls;
#ifdef DFKV_WITH_URING
  mutable std::mutex backend_mu;
  std::condition_variable backend_cv;
  std::vector<std::unique_ptr<ControlledRdmaUringBackend>> backends;
#endif


  explicit RdmaUringNode(
      const std::string& tag, size_t max_msg = kMaxMsg
#ifdef DFKV_WITH_URING
      ,
      std::function<void(ControlledRdmaUringBackend*)> backend_configure = {}
#endif
      ) {
    dir = fs::temp_directory_path() / ("dfkv_uring_" + tag);
    fs::remove_all(dir);
    fs::create_directories(dir);
    srv = std::make_unique<KvNodeServer>(dir.string(), 1ull << 30);
    EXPECT_EQ(srv->Start(0), Status::kOk);
    rsrv = std::make_unique<RdmaServer>(
        [this](uint8_t op, const BlockKey& key, uint64_t off, uint64_t len,
               const char* pl, uint64_t pll, std::string* out,
               size_t* value_len) {
          return srv->ProcessRequestForKey(
              op, key, off, len, pl, pll, out, value_len);
        },
        max_msg);
    rsrv->set_range_handler(  // sync fallback (used when uring path is off)
        [this](const BlockKey& key, uint64_t off, uint64_t len,
               char* io_buf, size_t cap, const char** out_data,
               size_t* out_len, size_t* value_len) {
          {
            std::lock_guard<std::mutex> lock(observation_mu);
            ++range_calls[key.Filename()];
          }
          return srv->RangeDirectForKey(
              key, off, len, io_buf, cap, out_data, out_len, value_len);
        });
    rsrv->set_cache_direct_handler(
        [this](const BlockKey& key, char* data, size_t len, size_t cap) {
          return srv->CacheDirectForKey(key, data, len, cap);
        });
    rsrv->set_prepare_read_handler(
        [this](const BlockKey& key, uint64_t off, uint64_t len,
               char* staging, size_t cap) {
          {
            std::lock_guard<std::mutex> lock(observation_mu);
            ++prepare_calls[key.Filename()];
          }
          return srv->PrepareReadForKey(key, off, len, staging, cap);
        });
#ifdef DFKV_WITH_URING
    if (backend_configure)
      InstallControlledBackendFactory(std::move(backend_configure));
#endif
    EXPECT_EQ(rsrv->Start(0), Status::kOk);
    addr = "127.0.0.1:" + std::to_string(rsrv->port());
  }
#ifdef DFKV_WITH_URING
  void InstallControlledBackendFactory(
      std::function<void(ControlledRdmaUringBackend*)> configure = {}) {
    RdmaServerTestPeer::SetUringBackendFactory(
        rsrv.get(), [this, configure = std::move(configure)] {
          auto backend = std::make_unique<ControlledRdmaUringBackend>();
          if (configure) configure(backend.get());
          ControlledRdmaUringBackend* result = backend.get();
          {
            std::lock_guard<std::mutex> lock(backend_mu);
            backends.push_back(std::move(backend));
          }
          backend_cv.notify_all();
          return result;
        });
  }

  ControlledRdmaUringBackend* WaitForBackend(
      size_t index, int timeout_ms = 5000) {
    std::unique_lock<std::mutex> lock(backend_mu);
    if (!backend_cv.wait_for(
            lock, std::chrono::milliseconds(timeout_ms),
            [&] { return backends.size() > index; }))
      return nullptr;
    return backends[index].get();
  }

  ControlledRdmaUringBackend* WaitForBackendWithSubmitted(
      size_t count, int timeout_ms = 5000) const {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    do {
      for (ControlledRdmaUringBackend* backend : BackendSnapshot()) {
        if (backend->History().size() >= count) return backend;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    return nullptr;
  }

  std::vector<ControlledRdmaUringBackend*> BackendSnapshot() const {
    std::lock_guard<std::mutex> lock(backend_mu);
    std::vector<ControlledRdmaUringBackend*> snapshot;
    snapshot.reserve(backends.size());
    for (const auto& backend : backends) snapshot.push_back(backend.get());
    return snapshot;
  }

  struct ControlledSubmission {
    ControlledRdmaUringBackend* backend = nullptr;
    ControlledRdmaUringBackend::Request request;
  };

  std::vector<ControlledSubmission> AggregateSubmissions() const {
    std::vector<ControlledSubmission> submissions;
    for (ControlledRdmaUringBackend* backend : BackendSnapshot()) {
      for (const auto& request : backend->History())
        submissions.push_back({backend, request});
    }
    return submissions;
  }

  bool WaitForAggregateSubmitted(size_t count, int timeout_ms = 5000) const {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    do {
      if (AggregateSubmissions().size() >= count) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
  }

  size_t AggregateAccepted() const {
    size_t accepted = 0;
    for (ControlledRdmaUringBackend* backend : BackendSnapshot())
      accepted += backend->AcceptedCount();
    return accepted;
  }
#endif
  ~RdmaUringNode() {
    if (rsrv) rsrv->Stop();
    if (srv) srv->Stop();
    fs::remove_all(dir);
  }
  size_t PrepareCalls(const std::string& logical_key) const {
    std::lock_guard<std::mutex> lock(observation_mu);
    const auto found =
        prepare_calls.find(ToBlockKey(SelfHdr(), logical_key).Filename());
    return found == prepare_calls.end() ? 0 : found->second;
  }
  size_t RangeCalls(const std::string& logical_key) const {
    std::lock_guard<std::mutex> lock(observation_mu);
    const auto found =
        range_calls.find(ToBlockKey(SelfHdr(), logical_key).Filename());
    return found == range_calls.end() ? 0 : found->second;
  }
};

struct MultiWindowGet {
  explicit MultiWindowGet(size_t targets_per_window, size_t window_count,
                          size_t seed)
      : sizes(targets_per_window * (window_count - 1) + 1),
        destinations(sizes.size()),
        pointers(sizes.size()) {
    for (size_t i = 0; i < sizes.size(); ++i) {
      sizes[i] = 113 + ((seed + i * 17) % 89);
      destinations[i].assign(sizes[i], '\0');
      pointers[i] = destinations[i].data();
    }
  }

  size_t capacity() const {
    size_t total = 0;
    for (size_t size : sizes) total += size;
    return total;
  }
  size_t window_count(size_t targets_per_window) const {
    return 1 + (sizes.size() - 1) / targets_per_window;
  }
  std::string Flatten() const {
    std::string out;
    out.reserve(capacity());
    for (const auto& destination : destinations) out += destination;
    return out;
  }

  std::vector<size_t> sizes;
  std::vector<std::string> destinations;
  std::vector<void*> pointers;
};

std::string PatternValue(size_t size, size_t seed) {
  std::string value(size, '\0');
  for (size_t i = 0; i < size; ++i)
    value[i] = static_cast<char>((seed * 131 + i * 29 + i / 97) & 0xff);
  return value;
}

TEST(RdmaLoopback, MultiWindowGetDepthFourKeepsLogicalOperationsIsolated) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");
  ScopedEnv recv_segment("DFKV_RDMA_RECV_SEGMENT_SIZE", "33554432");
  ScopedEnv idle_reclaim("DFKV_RDMA_IDLE_MS", "200");
  ScopedEnv sync_prepared_reads("DFKV_SERVER_URING", "0");
  // The malformed-window injection is local to the deliberate fault below.
  // Force it off even when the test process inherited the variable, then
  // restore the inherited value when the test exits.
  ScopedEnv bad_get_op_id_disabled(
      "DFKV_RDMA_TEST_BAD_GET_OP_ID_WINDOW", nullptr);
  RdmaUringNode node("getopid");
  ASSERT_EQ(node.rsrv->PipelineDepth(), 4u);

  const uint64_t transient_baseline =
      rdma::RcEndpoint::TransientUserMrActive();
  const uint64_t active_baseline = node.rsrv->ActiveConns();
  const auto wait_for_active_at_most = [&node](uint64_t limit) {
    for (int i = 0; i < 1000 && node.rsrv->ActiveConns() > limit; ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return node.rsrv->ActiveConns();
  };

  {
    RdmaTransport rt(kMaxMsg);
    KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
    c.set_batch_concurrency(1);  // one QP; concurrency is the negotiated depth
    ASSERT_EQ(
        CounterVal(rt.MetricsText(), "dfkv_rdma_client_pipeline_depth"), 4);

    const size_t targets_per_window = rt.MaxSgPayloadSegs();
    ASSERT_GE(targets_per_window, 2u);
    const size_t segment_count = targets_per_window * 3 + 5;
    std::vector<size_t> sizes(segment_count);
    for (size_t segment = 0; segment < segment_count; ++segment)
      sizes[segment] = 73 + segment % 29;

    constexpr size_t kItems = 2;
    std::vector<std::string> values(kItems);
    for (size_t item = 0; item < kItems; ++item) {
      for (size_t segment = 0; segment < segment_count; ++segment) {
        for (size_t byte = 0; byte < sizes[segment]; ++byte) {
          values[item].push_back(static_cast<char>(
              (item * 151 + segment * 41 + byte * 17) & 0xff));
        }
      }
      ASSERT_TRUE(c.Put("getopid_" + std::to_string(item),
                        values[item].data(), values[item].size()));
    }
    ASSERT_NE(values[0], values[1]);

    std::vector<std::vector<std::string>> dst(
        kItems, std::vector<std::string>(segment_count));
    std::vector<KvGetItemSg> gets;
    for (size_t item = 0; item < kItems; ++item) {
      std::vector<void*> ptrs;
      for (size_t segment = 0; segment < segment_count; ++segment) {
        dst[item][segment].assign(sizes[segment], '\0');
        ptrs.push_back(dst[item][segment].data());
      }
      gets.push_back(
          {"getopid_" + std::to_string(item), ptrs, sizes});
    }

    const long slot_changes_before = CounterVal(
        node.rsrv->MetricsText(),
        "dfkv_rdma_v2_get_continuation_slot_changes_total");
    std::vector<size_t> lengths;
    const auto got = c.BatchGetAutoSg(gets, &lengths);
    ASSERT_EQ(got.size(), kItems);
    ASSERT_EQ(lengths.size(), kItems);
    for (size_t item = 0; item < kItems; ++item) {
      ASSERT_TRUE(got[item]) << "logical GET " << item;
      EXPECT_EQ(lengths[item], values[item].size());
      std::string flattened;
      for (size_t segment = 0; segment < segment_count; ++segment)
        flattened += dst[item][segment];
      EXPECT_EQ(flattened, values[item]) << "logical GET " << item;
      EXPECT_NE(flattened, values[1 - item])
          << "continuation state crossed logical GET identities";
    }
    const long slot_changes_after = CounterVal(
        node.rsrv->MetricsText(),
        "dfkv_rdma_v2_get_continuation_slot_changes_total");
    ASSERT_GE(slot_changes_before, 0);
    EXPECT_GE(slot_changes_after - slot_changes_before, 2)
        << "test did not move continuations across receive WQE slots";
    EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), transient_baseline);

    // Corrupt the stable identity on window two. The server must reject the
    // continuation, destroy that operation's PreparedRead/state, and never
    // issue writes for any later window. The transport retires the failed QP.
    std::vector<std::string> poisoned_dst(
        segment_count, std::string());
    std::vector<void*> poisoned_ptrs;
    for (size_t segment = 0; segment < segment_count; ++segment) {
      poisoned_dst[segment].assign(sizes[segment], '\x5a');
      poisoned_ptrs.push_back(poisoned_dst[segment].data());
    }
    const uint64_t active_before_fault = node.rsrv->ActiveConns();
    const uint64_t opened_before_fault = node.rsrv->V2Conns();
    std::vector<bool> failed;
    std::vector<size_t> failed_lengths;
    {
      ScopedEnv bad_id("DFKV_RDMA_TEST_BAD_GET_OP_ID_WINDOW", "2");
      failed = c.BatchGetAutoSg(
          {{"getopid_0", poisoned_ptrs, sizes}}, &failed_lengths);
    }
    ASSERT_EQ(failed.size(), 1u);
    EXPECT_FALSE(failed[0]);
    ASSERT_EQ(failed_lengths.size(), 1u);
    EXPECT_EQ(failed_lengths[0], 0u);
    size_t first_window_offset = 0;
    for (size_t segment = 0; segment < targets_per_window; ++segment) {
      EXPECT_EQ(poisoned_dst[segment],
                values[0].substr(first_window_offset, sizes[segment]))
          << "fault injection ran before the first window, segment " << segment;
      first_window_offset += sizes[segment];
    }
    for (size_t segment = targets_per_window;
         segment < segment_count; ++segment) {
      EXPECT_EQ(poisoned_dst[segment],
                std::string(sizes[segment], '\x5a'))
          << "server mutated a destination after rejecting bad op id, segment "
          << segment;
    }

    const uint64_t opened_delta =
        node.rsrv->V2Conns() - opened_before_fault;
    ASSERT_GT(opened_delta, 0u);
    ASSERT_GT(active_before_fault, active_baseline);
    const uint64_t cleanup_limit = active_before_fault - 1;
    EXPECT_LE(wait_for_active_at_most(cleanup_limit), cleanup_limit)
        << "malformed continuation connection did not retire";
    EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), transient_baseline);
    ASSERT_EQ(std::getenv("DFKV_RDMA_TEST_BAD_GET_OP_ID_WINDOW"), nullptr);

    // A fresh logical ID on the same key must not inherit rejected sequence
    // state or a stale PreparedRead owner.
    std::vector<std::string> recovered_dst(
        segment_count, std::string());
    std::vector<void*> recovered_ptrs;
    for (size_t segment = 0; segment < segment_count; ++segment) {
      recovered_dst[segment].assign(sizes[segment], '\0');
      recovered_ptrs.push_back(recovered_dst[segment].data());
    }
    std::vector<size_t> recovered_lengths;
    // A transport error marks the node unhealthy in the public KVClient API.
    // Use fresh client health state while retaining the same transport/pool:
    // recovery therefore tests QP retirement/redial rather than an intentional
    // client cooldown, and still catches a poisoned pooled endpoint.
    KVClient recovery_client({{"n", node.addr}}, SelfHdr(), &rt);
    recovery_client.set_batch_concurrency(1);
    const auto recovered = recovery_client.BatchGetAutoSg(
        {{"getopid_0", recovered_ptrs, sizes}}, &recovered_lengths);
    ASSERT_EQ(recovered.size(), 1u);
    ASSERT_TRUE(recovered[0]);
    ASSERT_EQ(recovered_lengths.size(), 1u);
    EXPECT_EQ(recovered_lengths[0], values[0].size());
    std::string recovered_value;
    for (const auto& segment : recovered_dst) recovered_value += segment;
    EXPECT_EQ(recovered_value, values[0]);
    EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), transient_baseline);
  }

  EXPECT_LE(wait_for_active_at_most(active_baseline), active_baseline);
  EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), transient_baseline);
  node.rsrv->Stop();
  EXPECT_EQ(node.rsrv->ActiveConns(), 0u);
}

// Multi-window io_uring contract: one prepared disk read backs every ordered
// continuation, and its source ownership survives through the final SEND.
TEST(RdmaLoopback, UringMultiWindowGetSubmitsOnceAndRetainsSource) {
#ifndef DFKV_WITH_URING
  GTEST_SKIP() << "test requires a DFKV_WITH_URING build";
#else
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "1");
  ScopedEnv uring_depth("DFKV_SERVER_URING_DEPTH", "8");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");
  ScopedEnv recv_segment("DFKV_RDMA_RECV_SEGMENT_SIZE", "33554432");
  ScopedEnv abort_window("DFKV_RDMA_TEST_ABORT_MULTIWR_WINDOW", nullptr);
  ScopedEnv bad_operation("DFKV_RDMA_TEST_BAD_GET_OP_ID_WINDOW", nullptr);

  RdmaUringNode node("multi-window-submit-once", 64 * 1024);
  RdmaTransport transport(64 * 1024);
  KVClient client({{"n", node.addr}}, SelfHdr(), &transport);
  client.set_batch_concurrency(1);
  const size_t targets_per_window = transport.MaxSgPayloadSegs();
  ASSERT_GE(targets_per_window, 2u);

  MultiWindowGet first(targets_per_window, 4, 7);
  ASSERT_EQ(first.window_count(targets_per_window), 4u);
  const std::string first_value = PatternValue(first.capacity(), 11);
  ASSERT_TRUE(
      client.Put("uring_multi_first", first_value.data(), first_value.size()));
  if (node.rsrv->UringInitFallbacks() != 0)
    GTEST_SKIP() << "io_uring unavailable";

  const uint64_t reads_before = node.rsrv->UringReads();
  const uint64_t writes_before = node.rsrv->V2GetWrites();
  std::vector<size_t> lengths;
  const auto result = client.BatchGetAutoSg(
      {{"uring_multi_first", first.pointers, first.sizes}}, &lengths);
  ASSERT_EQ(result.size(), 1u);
  ASSERT_TRUE(result[0]);
  ASSERT_EQ(lengths.size(), 1u);
  EXPECT_EQ(lengths[0], first_value.size());
  EXPECT_EQ(first.Flatten(), first_value);
  EXPECT_EQ(node.PrepareCalls("uring_multi_first"), 1u)
      << "continuation windows must reuse the first prepared read";
  EXPECT_EQ(node.RangeCalls("uring_multi_first"), 0u)
      << "an eligible multi-window read must not fall back to blocking Range";
  EXPECT_EQ(node.rsrv->UringReads() - reads_before, 1u)
      << "one logical object must submit exactly one disk read";
  EXPECT_EQ(node.rsrv->V2GetWrites() - writes_before, 4u)
      << "each non-empty logical window must issue one RDMA WRITE";

  // A second logical GET on the same QP can reuse every receive/source slot only
  // after the first operation's final signaled SEND has retired its owner.
  MultiWindowGet repeated(targets_per_window, 4, 7);
  const uint64_t repeat_reads_before = node.rsrv->UringReads();
  std::vector<size_t> repeated_lengths;
  const auto repeated_result = client.BatchGetAutoSg(
      {{"uring_multi_first", repeated.pointers, repeated.sizes}},
      &repeated_lengths);
  ASSERT_EQ(repeated_result.size(), 1u);
  ASSERT_TRUE(repeated_result[0]);
  ASSERT_EQ(repeated_lengths.size(), 1u);
  EXPECT_EQ(repeated_lengths[0], first_value.size());
  EXPECT_EQ(repeated.Flatten(), first_value);
  EXPECT_EQ(node.PrepareCalls("uring_multi_first"), 2u);
  EXPECT_EQ(node.RangeCalls("uring_multi_first"), 0u);
  EXPECT_EQ(node.rsrv->UringReads() - repeat_reads_before, 1u);

  // Two different operation IDs post their first windows before waiting for a
  // reply. Both must be prepared/submitted independently; their continuations
  // then consume the matching retained source in request order.
  MultiWindowGet left(targets_per_window, 3, 19);
  MultiWindowGet right(targets_per_window, 3, 43);
  const std::string left_value = PatternValue(left.capacity(), 23);
  const std::string right_value = PatternValue(right.capacity(), 47);
  ASSERT_NE(left_value, right_value);
  ASSERT_TRUE(client.Put("uring_multi_left", left_value.data(),
                         left_value.size()));
  ASSERT_TRUE(client.Put("uring_multi_right", right_value.data(),
                         right_value.size()));
  const uint64_t pair_reads_before = node.rsrv->UringReads();
  const uint64_t pair_writes_before = node.rsrv->V2GetWrites();
  std::vector<size_t> pair_lengths;
  const auto pair_result = client.BatchGetAutoSg(
      {{"uring_multi_left", left.pointers, left.sizes},
       {"uring_multi_right", right.pointers, right.sizes}},
      &pair_lengths);
  ASSERT_EQ(pair_result.size(), 2u);
  EXPECT_TRUE(pair_result[0]);
  EXPECT_TRUE(pair_result[1]);
  ASSERT_EQ(pair_lengths.size(), 2u);
  EXPECT_EQ(pair_lengths[0], left_value.size());
  EXPECT_EQ(pair_lengths[1], right_value.size());
  EXPECT_EQ(left.Flatten(), left_value);
  EXPECT_EQ(right.Flatten(), right_value);
  EXPECT_NE(left.Flatten(), right_value)
      << "ordered replies crossed independent operation IDs";
  EXPECT_EQ(node.PrepareCalls("uring_multi_left"), 1u);
  EXPECT_EQ(node.PrepareCalls("uring_multi_right"), 1u);
  EXPECT_EQ(node.RangeCalls("uring_multi_left"), 0u);
  EXPECT_EQ(node.RangeCalls("uring_multi_right"), 0u);
  EXPECT_EQ(node.rsrv->UringReads() - pair_reads_before, 2u);
  EXPECT_EQ(node.rsrv->V2GetWrites() - pair_writes_before, 6u);
#endif
}

TEST(RdmaLoopback, UringMultiWindowDisconnectReleasesPreparedRead) {
#ifndef DFKV_WITH_URING
  GTEST_SKIP() << "test requires a DFKV_WITH_URING build";
#else
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "1");
  ScopedEnv uring_depth("DFKV_SERVER_URING_DEPTH", "8");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");
  ScopedEnv recv_segment("DFKV_RDMA_RECV_SEGMENT_SIZE", "33554432");
  ScopedEnv abort_window_disabled(
      "DFKV_RDMA_TEST_ABORT_MULTIWR_WINDOW", nullptr);
  ScopedEnv bad_operation_disabled(
      "DFKV_RDMA_TEST_BAD_GET_OP_ID_WINDOW", nullptr);

  RdmaUringNode node("multi-window-disconnect", 64 * 1024);
  RdmaTransport seed_transport(64 * 1024);
  KVClient seed_client({{"n", node.addr}}, SelfHdr(), &seed_transport);
  seed_client.set_batch_concurrency(1);
  const size_t targets_per_window = seed_transport.MaxSgPayloadSegs();
  ASSERT_GE(targets_per_window, 2u);
  MultiWindowGet shape(targets_per_window, 4, 61);
  ASSERT_EQ(shape.window_count(targets_per_window), 4u);
  const std::string value = PatternValue(shape.capacity(), 67);
  ASSERT_TRUE(
      seed_client.Put("uring_multi_abort", value.data(), value.size()));
  if (node.rsrv->UringInitFallbacks() != 0)
    GTEST_SKIP() << "io_uring unavailable";

  // Corrupt windows 2, 3, and 4 respectively. The server disconnects after the
  // async first-window submission, an intermediate continuation, and the final
  // continuation request. Every exit must retire the same retained owner once.
  for (size_t corrupt_at : {2u, 3u, 4u}) {
    MultiWindowGet aborted(targets_per_window, 4, 61);
    for (auto& destination : aborted.destinations)
      destination.assign(destination.size(), '\x5a');
    for (size_t i = 0; i < aborted.pointers.size(); ++i)
      aborted.pointers[i] = aborted.destinations[i].data();

    const size_t prepares_before = node.PrepareCalls("uring_multi_abort");
    const uint64_t reads_before = node.rsrv->UringReads();
    std::vector<size_t> aborted_lengths;
    std::vector<bool> aborted_result;
    {
      const std::string window = std::to_string(corrupt_at);
      ScopedEnv bad_operation("DFKV_RDMA_TEST_BAD_GET_OP_ID_WINDOW",
                              window.c_str());
      RdmaTransport failed_transport(64 * 1024);
      KVClient failed_client({{"n", node.addr}}, SelfHdr(), &failed_transport);
      failed_client.set_batch_concurrency(1);
      aborted_result = failed_client.BatchGetAutoSg(
          {{"uring_multi_abort", aborted.pointers, aborted.sizes}},
          &aborted_lengths);
    }
    ASSERT_EQ(aborted_result.size(), 1u);
    EXPECT_FALSE(aborted_result[0]) << "corrupt window " << corrupt_at;
    ASSERT_EQ(aborted_lengths.size(), 1u);
    EXPECT_EQ(aborted_lengths[0], 0u);
    EXPECT_EQ(node.rsrv->UringReads() - reads_before, 1u)
        << "corrupt window " << corrupt_at
        << " reread or failed to submit the first window";
    EXPECT_EQ(node.PrepareCalls("uring_multi_abort") - prepares_before, 1u)
        << "continuations must not prepare after window " << corrupt_at;
    // RangeIntoMulti publishes operation-owned staging only after the logical
    // item completes. A disconnect must therefore expose no torn prefix even
    // though the server has already issued writes for earlier windows.
    for (size_t segment = 0; segment < aborted.sizes.size(); ++segment) {
      EXPECT_EQ(aborted.destinations[segment],
                std::string(aborted.sizes[segment], '\x5a'))
          << "corrupt window " << corrupt_at << ", segment " << segment;
    }
    // Destruction of the failed connection must abort its retained owner and
    // free the source slot exactly once: a fresh operation on the same key must
    // neither join a stale coalescer flight nor inherit continuation state.
    MultiWindowGet recovered(targets_per_window, 4, 61);
    RdmaTransport recovery_transport(64 * 1024);
    KVClient recovery_client(
        {{"n", node.addr}}, SelfHdr(), &recovery_transport);
    recovery_client.set_batch_concurrency(1);
    std::vector<size_t> recovered_lengths;
    const auto recovered_result = recovery_client.BatchGetAutoSg(
        {{"uring_multi_abort", recovered.pointers, recovered.sizes}},
        &recovered_lengths);
    ASSERT_EQ(recovered_result.size(), 1u);
    ASSERT_TRUE(recovered_result[0]) << "recovery after window " << corrupt_at;
    ASSERT_EQ(recovered_lengths.size(), 1u);
    EXPECT_EQ(recovered_lengths[0], value.size());
    EXPECT_EQ(recovered.Flatten(), value);
    EXPECT_EQ(node.PrepareCalls("uring_multi_abort") - prepares_before, 2u)
        << "the failed owner leaked into the recovery operation";
  }
#endif
}

TEST(RdmaLoopback, PreparedReadSyncAndSingleWindowPathsStayUnchanged) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "0");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");
  ScopedEnv recv_segment("DFKV_RDMA_RECV_SEGMENT_SIZE", "33554432");
  RdmaUringNode node("prepared-sync-unchanged", 64 * 1024);
  RdmaTransport transport(64 * 1024);
  KVClient client({{"n", node.addr}}, SelfHdr(), &transport);
  client.set_batch_concurrency(1);

  const std::string single_value = PatternValue(8193, 71);
  ASSERT_TRUE(client.Put("prepared_single", single_value.data(),
                         single_value.size()));
  std::string single_out(single_value.size(), '\0');
  ASSERT_TRUE(client.Get("prepared_single", single_out.data(),
                         single_out.size()));
  EXPECT_EQ(single_out, single_value);
  EXPECT_EQ(node.PrepareCalls("prepared_single"), 1u);
  EXPECT_EQ(node.RangeCalls("prepared_single"), 0u);
  EXPECT_EQ(node.rsrv->UringReads(), 0u);

  const size_t targets_per_window = transport.MaxSgPayloadSegs();
  MultiWindowGet multi(targets_per_window, 3, 73);
  const std::string multi_value = PatternValue(multi.capacity(), 79);
  ASSERT_TRUE(
      client.Put("prepared_multi", multi_value.data(), multi_value.size()));
  std::vector<size_t> lengths;
  const auto result = client.BatchGetAutoSg(
      {{"prepared_multi", multi.pointers, multi.sizes}}, &lengths);
  ASSERT_EQ(result.size(), 1u);
  ASSERT_TRUE(result[0]);
  ASSERT_EQ(lengths.size(), 1u);
  EXPECT_EQ(lengths[0], multi_value.size());
  EXPECT_EQ(multi.Flatten(), multi_value);
  EXPECT_EQ(node.PrepareCalls("prepared_multi"), 1u);
  EXPECT_EQ(node.RangeCalls("prepared_multi"), 0u);
  EXPECT_EQ(node.rsrv->UringReads(), 0u);
}

// Correctness proof for the io_uring async-GET path: many concurrent GETs over a
// SINGLE pooled connection, with the depth high enough that several requests are
// in flight per WaitComp batch (so the server submits a multi-read io_uring batch
// and must reply in arrival order). Every value must come back byte-correct. With
// the flag OFF this still passes via the synchronous path (regression guard); with
// DFKV_SERVER_URING=1 + a URING build it exercises the batch-and-wait reads.
TEST(RdmaLoopback, UringAsyncGetManyConcurrentInOrder) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  // Request the async path (no-op if unbuilt). Use overwrite=0 so a shell-set
  // DFKV_SERVER_URING (e.g. =0 to force the sync path) wins — this lets the CI
  // run the SAME test through both the sync and async serve loops.
  ::setenv("DFKV_SERVER_URING", "1", 0);
  ::setenv("DFKV_SERVER_URING_DEPTH", "32", 1);
  ::setenv("DFKV_RDMA_DEPTH", "8", 1);      // K=8 in-flight => multi-read batches
  ::setenv("DFKV_RDMA_RECV_SEGMENT_SIZE", "4194304", 1);
  // Small per-buffer cap so K=8 slots (rbuf+sbuf+dbuf each) stay under an 8 MiB
  // RLIMIT_MEMLOCK (CI default). Values below are <= 12 KiB, well within 64 KiB.
  constexpr size_t kUringMsg = 64 * 1024;
  RdmaUringNode node("ag", kUringMsg);
  RdmaTransport rt(kUringMsg);
  ASSERT_TRUE(rt.pipelined());
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  // Pin client batch concurrency to 1 QP: the Soft-RoCE (rdma_rxe) loopback used
  // in CI races when many distinct QPs run in parallel (an emulation artifact that
  // hits the plain sync GET path too — see ScatterGatherRoundtripOverRdma). A
  // single connection still pipelines up to `depth` GETs per send window, so the
  // SERVER still forms multi-read io_uring batches; this only removes the rxe
  // cross-QP flakiness, not the concurrency under test.
  c.set_batch_concurrency(1);

  // Distinct content per key (offset + index dependent) so a misrouted reply
  // (wrong buffer / reordered) would mismatch. Mix of sizes incl. sub-page and
  // multi-page to exercise the O_DIRECT aligned-superset trim.
  const int N = 200;
  const size_t kSizes[] = {64, 512, 4096, 8192, 12288};
  std::vector<std::string> vals(N), keys(N);
  std::vector<KvPutItem> puts(N);
  for (int i = 0; i < N; ++i) {
    keys[i] = "ag" + std::to_string(i);
    size_t sz = kSizes[i % 5];
    vals[i].resize(sz);
    for (size_t k = 0; k < sz; ++k)
      vals[i][k] = static_cast<char>((i * 131 + k * 7 + 13) & 0xFF);
    puts[i] = {keys[i], vals[i].data(), sz};
  }
  auto pr = c.BatchPut(puts);
  for (int i = 0; i < N; ++i) ASSERT_TRUE(pr[i]) << i;

  // Exercise exactly one negotiated K=8 client window. The CQ may expose those
  // completions over several nonblocking polls, so require the observable
  // microbatch contract: all eight reads submitted, fewer than eight batches,
  // and at least one submitted batch wider than one. Waiting briefly first
  // keeps the seed PUT's SEND completions out of this controlled snapshot.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  constexpr size_t kControlledReads = 8;
  std::vector<std::string> controlled_outs(kControlledReads);
  std::vector<KvGetItem> controlled_gets(kControlledReads);
  for (size_t i = 0; i < kControlledReads; ++i) {
    controlled_outs[i].assign(vals[i].size(), '\0');
    controlled_gets[i] = {keys[i], controlled_outs[i].data(), vals[i].size()};
  }
#ifdef DFKV_WITH_URING
  const uint64_t controlled_reads_before = node.rsrv->UringReads();
  const uint64_t controlled_batches_before = node.rsrv->UringReadBatches();
#endif
  const auto controlled_result = c.BatchGet(controlled_gets);
  ASSERT_EQ(controlled_result.size(), kControlledReads);
  for (size_t i = 0; i < kControlledReads; ++i) {
    ASSERT_TRUE(controlled_result[i]) << "controlled key " << i;
    EXPECT_EQ(controlled_outs[i], vals[i]) << "controlled key " << i;
  }
#ifdef DFKV_WITH_URING
  const char* controlled_uring = std::getenv("DFKV_SERVER_URING");
  if ((!controlled_uring || std::strcmp(controlled_uring, "0") != 0) &&
      node.rsrv->UringInitFallbacks() == 0) {
    EXPECT_EQ(controlled_reads_before, 0u);
    EXPECT_EQ(controlled_batches_before, 0u);
    EXPECT_EQ(node.rsrv->UringReads(), kControlledReads);
    const uint64_t controlled_batches = node.rsrv->UringReadBatches();
    const uint64_t controlled_max = node.rsrv->UringReadBatchMax();
    EXPECT_GT(controlled_batches, 0u);
    EXPECT_LT(controlled_batches, kControlledReads)
        << "ready drain never submitted more than one read per batch";
    EXPECT_GT(controlled_max, 1u);
    EXPECT_LE(controlled_max, kControlledReads);
    const std::string controlled_metrics = node.rsrv->MetricsText();
    EXPECT_EQ(CounterVal(controlled_metrics, "dfkv_uring_batches_total"),
              static_cast<long>(controlled_batches));
    EXPECT_EQ(CounterVal(controlled_metrics,
                         "dfkv_uring_batch_reads_total"),
              static_cast<long>(kControlledReads));
    EXPECT_EQ(CounterVal(controlled_metrics, "dfkv_uring_batch_max"),
              static_cast<long>(controlled_max));
  }
#endif

  // Run several BatchGet rounds over the one pooled connection; each round fans
  // out N GETs that pipeline K-at-a-time -> the server forms multi-read batches.
  for (int round = 0; round < 3; ++round) {
    std::vector<std::string> outs(N);
    std::vector<KvGetItem> gets(N);
    for (int i = 0; i < N; ++i) { outs[i].assign(vals[i].size(), '\0'); gets[i] = {keys[i], &outs[i][0], vals[i].size()}; }
    auto gr = c.BatchGet(gets);
    for (int i = 0; i < N; ++i) {
      ASSERT_TRUE(gr[i]) << "round " << round << " key " << i;
      EXPECT_EQ(outs[i], vals[i]) << "round " << round << " key " << i;
    }
  }

  // A miss on the async path must still be a clean miss (kNotFound), not an error
  // or a stale-buffer hit; interleave present/absent to mix sync-miss + async-hit
  // replies in the same pipelined window (order-preservation across reply kinds).
  std::vector<std::string> mouts(N);
  std::vector<KvGetItem> mgets;
  std::vector<std::string> mkeys;
  for (int i = 0; i < N; ++i) {
    mkeys.push_back(keys[i]);
    mkeys.push_back("ag_absent" + std::to_string(i));
  }
  std::vector<std::string> mo(mkeys.size());
  std::vector<KvGetItem> mg(mkeys.size());
  for (size_t i = 0; i < mkeys.size(); ++i) {
    size_t cap = (i % 2 == 0) ? vals[i / 2].size() : 4096;
    mo[i].assign(cap, '\0');
    mg[i] = {mkeys[i], &mo[i][0], cap};
  }
  auto mgr = c.BatchGet(mg);
  for (size_t i = 0; i < mkeys.size(); ++i) {
    if (i % 2 == 0) {
      ASSERT_TRUE(mgr[i]) << "present key " << mkeys[i];
      EXPECT_EQ(mo[i], vals[i / 2]) << "present key " << mkeys[i];
    } else {
      EXPECT_FALSE(mgr[i]) << "absent key should miss: " << mkeys[i];
    }
  }

  // The async (uring) read path now feeds op="get" latency: after this many
  // disk-backed GETs the 1/64 sampler must have recorded at least one sample,
  // where before this change the default read path was latency-blind. (When the
  // shell forces DFKV_SERVER_URING=0 the sync RangeDirect samples the same
  // series, so the assertion holds through both serve loops.)
  const std::string mtext = node.srv->MetricsText();
  auto gp = mtext.find("dfkv_op_latency_seconds_count{op=\"get\"}");
  ASSERT_NE(gp, std::string::npos) << mtext;
  long gcnt = std::stol(mtext.substr(
      gp + std::string("dfkv_op_latency_seconds_count{op=\"get\"}").size()));
  EXPECT_GT(gcnt, 0) << "uring GET path recorded no op=\"get\" latency sample";
#ifdef DFKV_WITH_URING
  const char* uring_enabled = std::getenv("DFKV_SERVER_URING");
  if (!uring_enabled || std::strcmp(uring_enabled, "0") != 0) {
    EXPECT_GT(node.rsrv->UringReads(), 0u)
        << "auto-v2 silently bypassed the io_uring read path";
    EXPECT_GT(node.rsrv->V2Conns(), 0u)
        << "test did not exercise the v2 async-GET response path";
  }
#endif

  ::unsetenv("DFKV_SERVER_URING");
  ::unsetenv("DFKV_SERVER_URING_DEPTH");
  ::unsetenv("DFKV_RDMA_DEPTH");
  ::unsetenv("DFKV_RDMA_RECV_SEGMENT_SIZE");
}

TEST(RdmaLoopback, UringReadyDrainSingleCompletionDoesNotWaitAndRecovers) {
#ifndef DFKV_WITH_URING
  GTEST_SKIP() << "test requires a DFKV_WITH_URING build";
#else
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "1");
  ScopedEnv uring_depth("DFKV_SERVER_URING_DEPTH", "8");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "8");
  ScopedEnv recv_segment("DFKV_RDMA_RECV_SEGMENT_SIZE", "4194304");
  // A mistakenly blocking "drain one more" would consume this full idle wait.
  ScopedEnv idle_reclaim("DFKV_RDMA_IDLE_MS", "2000");
  ScopedEnv ambient_completion_fault("DFKV_RDMA_TEST_COMPLETION_FAULT",
                                     nullptr);

  constexpr size_t kUringMsg = 64 * 1024;
  RdmaUringNode node("single-ready-drain", kUringMsg);
  RdmaTransport transport(kUringMsg);
  KVClient client({{"n", node.addr}}, SelfHdr(), &transport);
  client.set_batch_concurrency(1);
  const std::string value = PatternValue(12288, 101);
  ASSERT_TRUE(client.Put("single_ready", value.data(), value.size()));
  if (node.rsrv->UringInitFallbacks() != 0)
    GTEST_SKIP() << "io_uring unavailable";

  // Let the seed PUT's SEND completion retire so the next CQ wake contains only
  // this GET. Polling for another ready WC must return immediately rather than
  // waiting for a completion that does not exist.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  const uint64_t reads_before = node.rsrv->UringReads();
  const uint64_t batches_before = node.rsrv->UringReadBatches();
  std::string output(value.size(), '\0');
  const auto started = std::chrono::steady_clock::now();
  ASSERT_TRUE(client.Get("single_ready", output.data(), output.size()));
  const auto elapsed = std::chrono::steady_clock::now() - started;
  EXPECT_EQ(output, value);
  EXPECT_LT(elapsed, std::chrono::milliseconds(1500))
      << "ready-only drain waited for a nonexistent second completion";
  EXPECT_EQ(node.rsrv->UringReads() - reads_before, 1u);
  EXPECT_EQ(node.rsrv->UringReadBatches() - batches_before, 1u);
  EXPECT_EQ(node.rsrv->UringReadBatchMax(), 1u);

  // A client-side error completion retires the pooled QP after the request was
  // posted. GET is replay-safe: it must reconnect, issue one fresh single-ready
  // request, and still return the exact bytes. This also exercises connection
  // teardown while the server owns completion/source-lifetime state.
  const uint64_t fault_reads_before = node.rsrv->UringReads();
  const uint64_t fault_batches_before = node.rsrv->UringReadBatches();
  const uint64_t conns_before = node.rsrv->V2Conns();
  std::string recovered;
  {
    ScopedEnv completion_fault("DFKV_RDMA_TEST_COMPLETION_FAULT", "1:1:1");
    EXPECT_EQ(transport.Range(
                  node.addr, ToBlockKey(SelfHdr(), "single_ready"), 0,
                  value.size(), &recovered, nullptr),
              Status::kOk);
  }
  EXPECT_EQ(recovered, value);
  const uint64_t fault_reads =
      node.rsrv->UringReads() - fault_reads_before;
  const uint64_t fault_batches =
      node.rsrv->UringReadBatches() - fault_batches_before;
  EXPECT_GE(fault_reads, 1u);
  EXPECT_EQ(fault_batches, fault_reads)
      << "every request in this single-ready recovery must submit alone";
  EXPECT_EQ(node.rsrv->UringReadBatchMax(), 1u);
  EXPECT_GT(node.rsrv->V2Conns(), conns_before)
      << "error completion did not retire and reconnect the pooled QP";
#endif
}


TEST(RdmaLoopback, ControlledUringRangeIntoKeepsDepthFilledWithoutWaitAll) {
#ifndef DFKV_WITH_URING
  GTEST_SKIP() << "test requires a DFKV_WITH_URING build";
#else
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "1");
  ScopedEnv uring_depth("DFKV_SERVER_URING_DEPTH", "32");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "16");
  ScopedEnv recv_segment("DFKV_RDMA_RECV_SEGMENT_SIZE", "8388608");
  ScopedEnv batch_timeout("DFKV_RDMA_BATCH_OP_TIMEOUT_MS", "15000");

  constexpr size_t kMsg = 64 * 1024;
  constexpr size_t kItems = 32;
  RdmaUringNode node(
      "controlled-pipeline", kMsg,
      [](ControlledRdmaUringBackend*) {});
  RdmaTransport transport(kMsg);

  std::vector<BlockKey> keys(kItems);
  std::vector<std::string> values(kItems), outputs(kItems);
  std::vector<RangeDst> destinations(kItems);
  for (size_t i = 0; i < kItems; ++i) {
    keys[i] = ToBlockKey(
        SelfHdr(), "controlled_pipeline_" + std::to_string(i));
    values[i] = PatternValue(32 * 1024, 200 + i);
    std::string ignored;
    ASSERT_EQ(
        node.srv->ProcessRequestForKey(
            static_cast<uint8_t>(WireOp::kCache), keys[i], 0, 0,
            values[i].data(), values[i].size(), &ignored),
        Status::kOk)
        << "direct seed " << i;
    outputs[i].assign(values[i].size(), '\0');
    destinations[i] = {outputs[i].data(), outputs[i].size()};
  }

  const uint64_t writes_before = node.rsrv->V2GetWrites();
  const uint64_t completion_errors_before =
      node.rsrv->CompletionErrors();
  const uint64_t send_fences_before = node.rsrv->UringSendFences();
  const uint64_t send_post_errors_before =
      node.rsrv->UringSendPostErrors();
  const uint64_t replies_posted_before =
      node.rsrv->UringRepliesPosted();
  const auto pipeline_started = std::chrono::steady_clock::now();
  std::vector<Status> statuses;
  std::vector<uint64_t> value_lengths;
  std::thread range_thread([&] {
    statuses =
        transport.RangeInto(node.addr, keys, destinations, &value_lengths);
  });
  const auto stop_and_join = [&] {
    std::atomic<bool> stopped{false};
    std::thread stopper([&] {
      node.rsrv->Stop();
      stopped.store(true, std::memory_order_release);
    });
    for (int attempt = 0;
         attempt < 1000 && !stopped.load(std::memory_order_acquire);
         ++attempt) {
      for (ControlledRdmaUringBackend* active : node.BackendSnapshot()) {
        for (const auto& request : active->History())
          active->CompleteRead(request.token);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    stopper.join();
    if (range_thread.joinable()) range_thread.join();
  };

  ControlledRdmaUringBackend* backend =
      node.WaitForBackendWithSubmitted(16);
  if (backend == nullptr) {
    ADD_FAILURE() << "one RangeInto QP did not submit its first depth window";
    stop_and_join();
    return;
  }
  const auto initial_backends = node.BackendSnapshot();
  const size_t backend_count_after_admission = initial_backends.size();
  const uint64_t conns_after_admission = node.rsrv->V2Conns();
  const uint64_t active_after_admission = node.rsrv->ActiveConns();
  std::vector<ControlledRdmaUringBackend::Request> history =
      backend->History();
  if (history.size() != 16) {
    ADD_FAILURE() << "first RangeInto window expected 16 reads, got "
                  << history.size();
    stop_and_join();
    return;
  }
  EXPECT_GE(backend->MaxInflight(), 16u);
  EXPECT_GE(node.rsrv->UringInflightMax(), 16u);
  EXPECT_EQ(node.rsrv->UringCompletions(), 0u);

  // Reap CQEs 15..1 while sequence zero remains outstanding. The server must
  // consume disk CQEs immediately but emit no noncontiguous wire prefix.
  for (size_t i = 16; i-- > 1;) {
    if (!backend->CompleteRead(history[i].token)) {
      ADD_FAILURE() << "first-window token " << i << " was not retained";
      stop_and_join();
      return;
    }
  }
  for (int attempt = 0;
       attempt < 1000 && node.rsrv->UringCompletions() < 15;
       ++attempt)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  EXPECT_EQ(node.rsrv->UringCompletions(), 15u)
      << "ready disk CQEs were held behind a wait-all barrier";
  EXPECT_FALSE(backend->WaitForSubmitted(17, 20))
      << "a noncontiguous reply escaped before sequence zero";

  if (!backend->CompleteRead(history[0].token)) {
    ADD_FAILURE() << "sequence-zero token was not retained";
    stop_and_join();
    return;
  }
  // A complete first prefix must reach status SEND fences before RangeInto can
  // post its second depth window on the same QP.
  for (int attempt = 0;
       attempt < 500 &&
       (node.rsrv->UringCompletions() < 16 ||
        node.rsrv->V2GetWrites() - writes_before < 16 ||
        node.rsrv->UringRepliesPosted() - replies_posted_before < 16 ||
        node.rsrv->UringSendFences() - send_fences_before < 16);
       ++attempt)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  EXPECT_EQ(node.rsrv->UringCompletions(), 16u);
  EXPECT_EQ(node.rsrv->V2GetWrites() - writes_before, 16u);
  EXPECT_EQ(node.rsrv->UringRepliesPosted() - replies_posted_before, 16u);
  EXPECT_GE(node.rsrv->UringSendFences() - send_fences_before, 16u);
  EXPECT_EQ(node.rsrv->ActiveConns(), active_after_admission);
  EXPECT_EQ(node.rsrv->CompletionErrors(), completion_errors_before);
  EXPECT_EQ(node.rsrv->UringSendPostErrors(), send_post_errors_before);

  if (!backend->WaitForSubmitted(kItems, 2000)) {
    ADD_FAILURE() << "same RangeInto QP did not ingest its second window";
    stop_and_join();
    return;
  }
  const auto backends_after_second = node.BackendSnapshot();
  EXPECT_EQ(backends_after_second.size(), backend_count_after_admission)
      << "controlled RangeInto retried onto a replacement QP";
  EXPECT_EQ(node.rsrv->V2Conns(), conns_after_admission);
  EXPECT_EQ(node.rsrv->ActiveConns(), active_after_admission);
  for (ControlledRdmaUringBackend* initial : initial_backends) {
    EXPECT_NE(std::find(backends_after_second.begin(),
                        backends_after_second.end(), initial),
              backends_after_second.end());
  }
  history = backend->History();
  if (history.size() != kItems) {
    ADD_FAILURE() << "expected exactly " << kItems
                  << " same-QP disk submissions, got " << history.size();
    stop_and_join();
    return;
  }
  for (size_t i = kItems; i-- > 16;) {
    if (!backend->CompleteRead(history[i].token)) {
      ADD_FAILURE() << "second-window token " << i << " was not retained";
      stop_and_join();
      return;
    }
  }

  range_thread.join();
  ASSERT_EQ(statuses.size(), kItems);
  ASSERT_EQ(value_lengths.size(), kItems);
  for (size_t i = 0; i < kItems; ++i) {
    EXPECT_EQ(statuses[i], Status::kOk) << i;
    EXPECT_EQ(value_lengths[i], values[i].size()) << i;
    EXPECT_EQ(outputs[i], values[i]) << "ordered reply " << i;
  }
  EXPECT_EQ(node.rsrv->UringReads(), kItems);
  EXPECT_EQ(node.rsrv->UringCompletions(), kItems);
  EXPECT_EQ(node.rsrv->UringInflight(), 0u);
  EXPECT_GE(node.rsrv->UringInflightMax(), 16u);
  for (int attempt = 0;
       attempt < 500 &&
       node.rsrv->UringSendFences() - send_fences_before < kItems;
       ++attempt)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  EXPECT_GE(node.rsrv->UringSendFences() - send_fences_before, kItems);
  EXPECT_EQ(node.rsrv->UringSendPostErrors(), send_post_errors_before);
  EXPECT_EQ(node.rsrv->UringRepliesPosted() - replies_posted_before, kItems);
  EXPECT_EQ(node.BackendSnapshot().size(), backend_count_after_admission);
  EXPECT_EQ(node.rsrv->V2Conns(), conns_after_admission);
  EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - pipeline_started)
                .count(),
            10000)
      << "controlled pipeline exceeded its deterministic fixture budget";
  for (ControlledRdmaUringBackend* active : node.BackendSnapshot())
    EXPECT_FALSE(active->BadState());
#endif
}


TEST(RdmaLoopback, ControlledUringMultiWindowBytesAndSendLifetimeAreExact) {
#ifndef DFKV_WITH_URING
  GTEST_SKIP() << "test requires a DFKV_WITH_URING build";
#else
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "1");
  ScopedEnv uring_depth("DFKV_SERVER_URING_DEPTH", "16");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "16");
  ScopedEnv recv_segment("DFKV_RDMA_RECV_SEGMENT_SIZE", "8388608");
  ScopedEnv batch_timeout("DFKV_RDMA_BATCH_OP_TIMEOUT_MS", "15000");

  constexpr size_t kMsg = 64 * 1024;
  const std::string logical_key = "controlled_multi";
  const BlockKey key = ToBlockKey(SelfHdr(), logical_key);
  RdmaUringNode node(
      "controlled-multi-window", kMsg,
      [](ControlledRdmaUringBackend*) {});
  RdmaTransport transport(kMsg);
  const size_t targets_per_window = transport.MaxSgPayloadSegs();
  ASSERT_GE(targets_per_window, 2u);
  const auto align_capacity = [](MultiWindowGet* get) {
    constexpr size_t kDiskAlignment = 4096;
    const size_t capacity = get->capacity();
    const size_t aligned =
        (capacity + kDiskAlignment - 1) & ~(kDiskAlignment - 1);
    const size_t extra = aligned - capacity;
    get->sizes.back() += extra;
    get->destinations.back().resize(get->sizes.back(), '\0');
    get->pointers.back() = get->destinations.back().data();
  };
  MultiWindowGet first(targets_per_window, 4, 503);
  align_capacity(&first);
  ASSERT_EQ(first.capacity() % 4096, 0u);
  const std::string value = PatternValue(first.capacity(), 509);
  std::string ignored;
  ASSERT_EQ(
      node.srv->ProcessRequestForKey(
          static_cast<uint8_t>(WireOp::kCache), key, 0, 0, value.data(),
          value.size(), &ignored),
      Status::kOk);
  const auto destination_for = [](MultiWindowGet& get) {
    RangeDstMulti destination;
    destination.payloads.reserve(get.pointers.size());
    for (size_t i = 0; i < get.pointers.size(); ++i)
      destination.payloads.emplace_back(get.pointers[i], get.sizes[i]);
    return destination;
  };
  RangeDstMulti first_destination = destination_for(first);
  const uint64_t writes_before = node.rsrv->V2GetWrites();
  const uint64_t fences_before = node.rsrv->UringSendFences();
  const uint64_t replies_before = node.rsrv->UringRepliesPosted();
  const uint64_t completion_errors_before =
      node.rsrv->CompletionErrors();
  const uint64_t send_post_errors_before =
      node.rsrv->UringSendPostErrors();

  const auto stop_and_join = [&](std::thread* worker) {
    std::atomic<bool> stopped{false};
    std::thread stopper([&] {
      node.rsrv->Stop();
      stopped.store(true, std::memory_order_release);
    });
    for (int attempt = 0;
         attempt < 1000 && !stopped.load(std::memory_order_acquire);
         ++attempt) {
      for (ControlledRdmaUringBackend* active : node.BackendSnapshot()) {
        for (const auto& request : active->History())
          active->CompleteRead(request.token);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    stopper.join();
    if (worker && worker->joinable()) worker->join();
  };

  std::vector<size_t> first_lengths;
  std::vector<Status> first_result;
  std::thread first_thread([&] {
    first_result = transport.RangeIntoMulti(
        node.addr, {key}, {first_destination}, &first_lengths);
  });
  ControlledRdmaUringBackend* backend =
      node.WaitForBackendWithSubmitted(1);
  if (backend == nullptr) {
    ADD_FAILURE() << "direct multi-window read was not submitted";
    stop_and_join(&first_thread);
    return;
  }
  const auto initial_backends = node.BackendSnapshot();
  const uint64_t conns_after_first = node.rsrv->V2Conns();
  auto history = backend->History();
  if (history.size() != 1 ||
      !backend->CompleteRead(history.front().token)) {
    ADD_FAILURE() << "controlled multi-window read was not retained";
    stop_and_join(&first_thread);
    return;
  }
  for (int attempt = 0;
       attempt < 500 &&
       node.rsrv->UringRepliesPosted() - replies_before < 1;
       ++attempt)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  if (node.rsrv->UringRepliesPosted() - replies_before < 1) {
    ADD_FAILURE()
        << "first disk CQE produced no status SEND: writes="
        << node.rsrv->V2GetWrites() - writes_before
        << " replies=" << node.rsrv->UringRepliesPosted() - replies_before
        << " fences=" << node.rsrv->UringSendFences() - fences_before
        << " send_post_errors=" << node.rsrv->UringSendPostErrors()
        << " completion_errors=" << node.rsrv->CompletionErrors()
        << " active=" << node.rsrv->ActiveConns()
        << " v2_conns=" << node.rsrv->V2Conns()
        << " seen=" << backend->SeenCount()
        << " accepted=" << backend->AcceptedCount()
        << " bad_backend=" << backend->BadState();
    stop_and_join(&first_thread);
    return;
  }
  for (int attempt = 0; attempt < 2000; ++attempt) {
    if (node.rsrv->V2GetWrites() - writes_before >= 4 ||
        node.rsrv->CompletionErrors() != completion_errors_before ||
        node.rsrv->ActiveConns() == 0)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (node.rsrv->V2GetWrites() - writes_before < 4) {
    ADD_FAILURE()
        << "multi-window continuation stalled after its first status: writes="
        << node.rsrv->V2GetWrites() - writes_before
        << " replies=" << node.rsrv->UringRepliesPosted() - replies_before
        << " fences=" << node.rsrv->UringSendFences() - fences_before
        << " send_post_errors=" << node.rsrv->UringSendPostErrors()
        << " completion_errors=" << node.rsrv->CompletionErrors()
        << " active=" << node.rsrv->ActiveConns()
        << " v2_conns=" << node.rsrv->V2Conns()
        << " seen=" << backend->SeenCount()
        << " accepted=" << backend->AcceptedCount()
        << " bad_backend=" << backend->BadState();
    stop_and_join(&first_thread);
    return;
  }
  first_thread.join();
  SCOPED_TRACE(
      ::testing::Message()
      << "first multi-window checkpoint: writes="
      << node.rsrv->V2GetWrites() - writes_before
      << " replies=" << node.rsrv->UringRepliesPosted() - replies_before
      << " fences=" << node.rsrv->UringSendFences() - fences_before
      << " send_post_errors_delta="
      << node.rsrv->UringSendPostErrors() - send_post_errors_before
      << " completion_errors_delta="
      << node.rsrv->CompletionErrors() - completion_errors_before
      << " active=" << node.rsrv->ActiveConns()
      << " v2_conns=" << node.rsrv->V2Conns()
      << " history=" << backend->History().size()
      << " seen=" << backend->SeenCount()
      << " accepted=" << backend->AcceptedCount()
      << " max_inflight=" << backend->MaxInflight()
      << " bad_backend=" << backend->BadState());
  ASSERT_EQ(first_result.size(), 1u);
  EXPECT_EQ(first_result[0], Status::kOk);
  ASSERT_EQ(first_lengths.size(), 1u);
  EXPECT_EQ(first_lengths[0], value.size());
  EXPECT_EQ(first.Flatten(), value);
  EXPECT_EQ(node.PrepareCalls(logical_key), 1u);
  EXPECT_EQ(node.RangeCalls(logical_key), 0u);
  EXPECT_EQ(backend->History().size(), 1u)
      << "continuation windows must retain, not resubmit, the source";
  EXPECT_EQ(node.rsrv->V2GetWrites() - writes_before, 4u);
  EXPECT_EQ(node.rsrv->UringRepliesPosted() - replies_before, 4u);

  // Immediate same-transport reuse exercises the final signaled SEND ownership
  // fence. The first source slot/PreparedRead may be recycled only afterward.
  MultiWindowGet second(targets_per_window, 4, 503);
  align_capacity(&second);
  RangeDstMulti second_destination = destination_for(second);
  std::vector<size_t> second_lengths;
  std::vector<Status> second_result;
  std::thread second_thread([&] {
    second_result = transport.RangeIntoMulti(
        node.addr, {key}, {second_destination}, &second_lengths);
  });
  const bool second_submitted = backend->WaitForSubmitted(2, 2000);
  history = backend->History();
  if (!second_submitted || history.size() != 2 ||
      !backend->CompleteRead(history.back().token)) {
    ADD_FAILURE() << "source slot was not reusable after final SEND";
    stop_and_join(&second_thread);
    return;
  }
  EXPECT_GE(node.rsrv->UringSendFences() - fences_before, 4u)
      << "second read reused the source before first final SEND fence";
  EXPECT_EQ(node.BackendSnapshot().size(), initial_backends.size());
  EXPECT_EQ(node.rsrv->V2Conns(), conns_after_first);
  second_thread.join();
  ASSERT_EQ(second_result.size(), 1u);
  EXPECT_EQ(second_result[0], Status::kOk);
  ASSERT_EQ(second_lengths.size(), 1u);
  EXPECT_EQ(second_lengths[0], value.size());
  EXPECT_EQ(second.Flatten(), value);
  EXPECT_EQ(node.PrepareCalls(logical_key), 2u);
  EXPECT_EQ(node.RangeCalls(logical_key), 0u);
  EXPECT_EQ(backend->History().size(), 2u);
  EXPECT_EQ(node.rsrv->V2GetWrites() - writes_before, 8u);
  EXPECT_EQ(node.rsrv->UringRepliesPosted() - replies_before, 8u);
  for (ControlledRdmaUringBackend* active : node.BackendSnapshot())
    EXPECT_FALSE(active->BadState());
#endif
}
TEST(RdmaLoopback, ControlledUringDisconnectDrainsPartialCqBeforeOwners) {
#ifndef DFKV_WITH_URING
  GTEST_SKIP() << "test requires a DFKV_WITH_URING build";
#else
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "1");
  ScopedEnv uring_depth("DFKV_SERVER_URING_DEPTH", "16");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "16");
  ScopedEnv recv_segment("DFKV_RDMA_RECV_SEGMENT_SIZE", "8388608");
  ScopedEnv batch_timeout("DFKV_RDMA_BATCH_OP_TIMEOUT_MS", "15000");

  constexpr size_t kMsg = 64 * 1024;
  constexpr size_t kItems = 16;
  RdmaUringNode node(
      "controlled-disconnect", kMsg,
      [](ControlledRdmaUringBackend*) {});
  RdmaTransport transport(kMsg);
  std::vector<BlockKey> keys(kItems);
  std::vector<std::string> values(kItems), outputs(kItems);
  std::vector<RangeDst> destinations(kItems);
  for (size_t i = 0; i < kItems; ++i) {
    keys[i] = ToBlockKey(
        SelfHdr(), "controlled_disconnect_" + std::to_string(i));
    values[i] = PatternValue(8192, 300 + i);
    std::string ignored;
    ASSERT_EQ(
        node.srv->ProcessRequestForKey(
            static_cast<uint8_t>(WireOp::kCache), keys[i], 0, 0,
            values[i].data(), values[i].size(), &ignored),
        Status::kOk);
    outputs[i].assign(values[i].size(), '\0');
    destinations[i] = {outputs[i].data(), outputs[i].size()};
  }

  std::vector<Status> statuses;
  std::vector<uint64_t> value_lengths;
  std::thread get_thread([&] {
    statuses =
        transport.RangeInto(node.addr, keys, destinations, &value_lengths);
  });
  const auto stop_and_join = [&] {
    std::atomic<bool> stopped{false};
    std::thread stopper([&] {
      node.rsrv->Stop();
      stopped.store(true, std::memory_order_release);
    });
    for (int attempt = 0;
         attempt < 1000 && !stopped.load(std::memory_order_acquire);
         ++attempt) {
      for (ControlledRdmaUringBackend* active : node.BackendSnapshot()) {
        for (const auto& request : active->History())
          active->CompleteRead(request.token);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    stopper.join();
    if (get_thread.joinable()) get_thread.join();
  };
  ControlledRdmaUringBackend* backend =
      node.WaitForBackendWithSubmitted(kItems);
  if (backend == nullptr) {
    ADD_FAILURE() << "direct disconnect fixture did not admit the full depth";
    stop_and_join();
    return;
  }
  const auto history = backend->History();
  if (history.size() != kItems) {
    ADD_FAILURE() << "disconnect fixture admitted " << history.size()
                  << " reads instead of exactly " << kItems;
    stop_and_join();
    return;
  }

  std::atomic<bool> stop_done{false};
  std::thread stop_thread([&] {
    node.rsrv->Stop();
    stop_done.store(true, std::memory_order_release);
  });
  bool first_half_completed = true;
  for (size_t i = 0; i < kItems / 2; ++i)
    first_half_completed &=
        backend->CompleteRead(history[i].token);
  EXPECT_TRUE(first_half_completed);
  const bool partial_seen = backend->WaitForSeen(kItems / 2);
  EXPECT_TRUE(partial_seen);
  EXPECT_FALSE(stop_done.load(std::memory_order_acquire))
      << "disconnect released PreparedRead/staging with reads still in flight";
  EXPECT_EQ(backend->AcceptedCount(), kItems / 2);

  bool second_half_completed = true;
  for (size_t i = kItems / 2; i < kItems; ++i)
    second_half_completed &=
        backend->CompleteRead(history[i].token);
  EXPECT_TRUE(second_half_completed);
  stop_thread.join();
  get_thread.join();
  EXPECT_TRUE(stop_done.load(std::memory_order_acquire));
  EXPECT_EQ(backend->SeenCount(), kItems);
  EXPECT_EQ(backend->AcceptedCount(), 0u);
  EXPECT_TRUE(backend->Exited());
  EXPECT_EQ(node.rsrv->UringInflight(), 0u);
  for (ControlledRdmaUringBackend* active : node.BackendSnapshot())
    EXPECT_FALSE(active->BadState());
#endif
}


TEST(RdmaLoopback, ControlledUringSubmitFailureDrainsAcceptedReadAndAborts) {
#ifndef DFKV_WITH_URING
  GTEST_SKIP() << "test requires a DFKV_WITH_URING build";
#else
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "1");
  ScopedEnv uring_depth("DFKV_SERVER_URING_DEPTH", "8");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");
  ScopedEnv recv_segment("DFKV_RDMA_RECV_SEGMENT_SIZE", "4194304");
  ScopedEnv batch_timeout("DFKV_RDMA_BATCH_OP_TIMEOUT_MS", "15000");

  constexpr size_t kMsg = 64 * 1024;
  constexpr size_t kItems = 4;
  RdmaUringNode node(
      "controlled-partial-submit", kMsg,
      [](ControlledRdmaUringBackend* backend) {
        backend->SetSubmitResults({1, -EIO});
      });
  RdmaTransport transport(kMsg);
  std::vector<std::string> logical_keys(kItems), values(kItems);
  std::vector<BlockKey> keys(kItems);
  std::vector<std::string> outputs(
      kItems, std::string(4096, '\x5a'));
  std::vector<RangeDst> destinations(kItems);
  for (size_t i = 0; i < kItems; ++i) {
    logical_keys[i] = "controlled_partial_" + std::to_string(i);
    keys[i] = ToBlockKey(SelfHdr(), logical_keys[i]);
    values[i] = PatternValue(4096, 600 + i);
    std::string ignored;
    ASSERT_EQ(
        node.srv->ProcessRequestForKey(
            static_cast<uint8_t>(WireOp::kCache), keys[i], 0, 0,
            values[i].data(), values[i].size(), &ignored),
        Status::kOk);
    destinations[i] = {outputs[i].data(), outputs[i].size()};
  }

  std::vector<Status> statuses;
  std::vector<uint64_t> value_lengths;
  std::thread get_thread([&] {
    statuses =
        transport.RangeInto(node.addr, keys, destinations, &value_lengths);
  });
  const auto stop_and_join = [&] {
    std::atomic<bool> stopped{false};
    std::thread stopper([&] {
      node.rsrv->Stop();
      stopped.store(true, std::memory_order_release);
    });
    for (int attempt = 0;
         attempt < 1000 && !stopped.load(std::memory_order_acquire);
         ++attempt) {
      for (ControlledRdmaUringBackend* active : node.BackendSnapshot()) {
        for (const auto& request : active->History())
          active->CompleteRead(request.token);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    stopper.join();
    if (get_thread.joinable()) get_thread.join();
  };
  ControlledRdmaUringBackend* backend =
      node.WaitForBackendWithSubmitted(1);
  if (backend == nullptr) {
    ADD_FAILURE() << "backend did not expose the scripted accepted read";
    stop_and_join();
    return;
  }
  const bool failed_after_partial_admission =
      backend->WaitForSubmitCalls(2);
  const auto history = backend->History();
  if (!failed_after_partial_admission || history.size() != 1) {
    ADD_FAILURE() << "scripted second submit failure did not follow one "
                     "accepted read: calls="
                  << (failed_after_partial_admission ? ">=2" : "<2")
                  << " accepted_history=" << history.size();
    stop_and_join();
    return;
  }

  std::thread stop_thread([&] { node.rsrv->Stop(); });
  EXPECT_TRUE(backend->CompleteRead(history[0].token));
  stop_thread.join();
  get_thread.join();
  EXPECT_EQ(backend->SeenCount(), 1u);
  EXPECT_EQ(backend->AcceptedCount(), 0u);
  EXPECT_TRUE(backend->Exited());
  EXPECT_EQ(node.rsrv->UringInflight(), 0u);
  EXPECT_LE(node.rsrv->UringReads(), 1u)
      << "only a fully admitted first group may publish a logical read";
  ASSERT_EQ(statuses.size(), kItems);
  for (size_t i = 0; i < kItems; ++i) {
    EXPECT_EQ(statuses[i], Status::kIOError) << i;
    EXPECT_EQ(outputs[i], std::string(4096, '\x5a')) << i;
    EXPECT_EQ(node.RangeCalls(logical_keys[i]), 0u) << i;
  }
  EXPECT_GE(node.PrepareCalls(logical_keys[0]), 1u);
  for (ControlledRdmaUringBackend* active : node.BackendSnapshot())
    EXPECT_FALSE(active->BadState());
#endif
}
TEST(RdmaLoopback, ControlledUringInitFailureUsesExactSyncFallback) {
#ifndef DFKV_WITH_URING
  GTEST_SKIP() << "test requires a DFKV_WITH_URING build";
#else
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv uring("DFKV_SERVER_URING", "1");
  ScopedEnv uring_depth("DFKV_SERVER_URING_DEPTH", "32");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "16");
  ScopedEnv recv_segment("DFKV_RDMA_RECV_SEGMENT_SIZE", "8388608");

  constexpr size_t kMsg = 64 * 1024;
  RdmaUringNode node(
      "controlled-init-fallback", kMsg,
      [](ControlledRdmaUringBackend* backend) {
        backend->queue_init_result = -EPERM;
      });
  RdmaTransport transport(kMsg);
  KVClient client({{"n", node.addr}}, SelfHdr(), &transport);
  client.set_batch_concurrency(1);
  const std::string value = PatternValue(16387, 401);
  ASSERT_TRUE(client.Put("controlled_sync", value.data(), value.size()));
  std::string output(value.size(), '\0');
  ASSERT_TRUE(client.Get("controlled_sync", output.data(), output.size()));
  EXPECT_EQ(output, value);
  ControlledRdmaUringBackend* backend = node.WaitForBackend(0);
  ASSERT_NE(backend, nullptr);
  EXPECT_TRUE(backend->History().empty());
  EXPECT_EQ(node.rsrv->UringReads(), 0u);
  EXPECT_GE(node.rsrv->UringInitFallbacks(), 1u);
  EXPECT_EQ(node.PrepareCalls("controlled_sync"), 1u);
#endif
}
// --- P3 B5-3: RAM hot-tier zero-copy RDMA serve --------------------------------
// A cache node with the RAM tier enabled must serve a GET straight from the
// pre-registered arena MR (scatter-send, no copy into the connection buffer, no
// disk), and the bytes must round-trip correctly. dfkv_ram_hit_total proves the
// zero-copy path was taken; the send-in-flight pin is released on IBV_WC_SEND.
namespace {
struct RamRdmaNode {
  fs::path dir;
  std::unique_ptr<KvNodeServer> srv;
  std::unique_ptr<RdmaServer> rsrv;
  std::string addr;
  std::mutex flight_mu;
  std::condition_variable flight_cv;
  std::string barrier_key;
  bool leader_prepared = false;
  bool follower_entered = false;
  bool follower_wait_timed_out = false;


  explicit RamRdmaNode(const std::string& tag, size_t max_msg = kMaxMsg,
                       bool writearound = false) {
    ConfigureTestRecvSegment();
    if (writearound)
      ::setenv("DFKV_RAM_WRITE_MODE", "writearound", 1);
    else
      ::unsetenv("DFKV_RAM_WRITE_MODE");
    ::setenv("DFKV_RAM_TIER", "1", 1);
    ::setenv("DFKV_RAM_TIER_BYTES", "8388608", 1);  // 8 MiB arena
    dir = fs::temp_directory_path() / ("dfkv_ramrdma_" + tag);
    fs::remove_all(dir);
    fs::create_directories(dir);
    srv = std::make_unique<KvNodeServer>(dir.string(), 1ull << 30);
    EXPECT_EQ(srv->Start(0), Status::kOk);
    rsrv = std::make_unique<RdmaServer>(
        [this](uint8_t op, const BlockKey& key, uint64_t off, uint64_t len,
               const char* pl, uint64_t pll, std::string* out,
               size_t* value_len) {
          return srv->ProcessRequestForKey(
              op, key, off, len, pl, pll, out, value_len);
        },
        max_msg);
    rsrv->set_range_handler(
        [this](const BlockKey& key, uint64_t off, uint64_t len,
               char* io_buf, size_t cap, const char** out_data,
               size_t* out_len, size_t* value_len) {
          {
            std::lock_guard<std::mutex> lock(flight_mu);
            if (!barrier_key.empty() && key.Filename() == barrier_key &&
                leader_prepared) {
              follower_entered = true;
              flight_cv.notify_all();
            }
          }
          return srv->RangeDirectForKey(
              key, off, len, io_buf, cap, out_data, out_len, value_len);
        });
    rsrv->set_cache_direct_handler(
        [this](const BlockKey& key, char* data, size_t len, size_t cap) {
          return srv->CacheDirectForKey(key, data, len, cap);
        });
    rsrv->set_prepare_read_handler(
        [this](const BlockKey& key, uint64_t off, uint64_t len,
               char* staging, size_t cap) {
          // An async follower blocks inside PrepareReadForKey while waiting for
          // the leader's published RAM reservation. Signal at handler entry,
          // before that wait, so the held leader can submit and publish.
          {
            std::lock_guard<std::mutex> lock(flight_mu);
            if (!barrier_key.empty() && key.Filename() == barrier_key &&
                leader_prepared && !follower_entered) {
              follower_entered = true;
              flight_cv.notify_all();
            }
          }
          PreparedRead prepared =
              srv->PrepareReadForKey(key, off, len, staging, cap);
          if (prepared.status() == Status::kOk && prepared.needs_io()) {
            std::unique_lock<std::mutex> lock(flight_mu);
            if (!barrier_key.empty() && key.Filename() == barrier_key &&
                !leader_prepared) {
              leader_prepared = true;
              flight_cv.notify_all();
              if (!flight_cv.wait_for(
                      lock, std::chrono::seconds(30),
                      [this] { return follower_entered; })) {
                follower_wait_timed_out = true;
              }
            }
          }
          return prepared;
        });
    // The RAM arena is a registered source pool; per-send pin ownership lives
    // inside PreparedRead.
    if (srv->ram_enabled()) {
      rsrv->RegisterMemory(srv->ram_arena(), srv->ram_arena_bytes());
    }
    EXPECT_EQ(rsrv->Start(0), Status::kOk);
    addr = "127.0.0.1:" + std::to_string(rsrv->port());
  }
  ~RamRdmaNode() {
    if (rsrv) rsrv->Stop();
    if (srv) srv->Stop();
    srv.reset();
    ::unsetenv("DFKV_RAM_TIER");
    ::unsetenv("DFKV_RAM_TIER_BYTES");
    ::unsetenv("DFKV_RAM_WRITE_MODE");
    fs::remove_all(dir);
  }
  void BarrierFirstPreparedRead(const std::string& logical_key) {
    std::lock_guard<std::mutex> lock(flight_mu);
    barrier_key = ToBlockKey(SelfHdr(), logical_key).Filename();
    leader_prepared = false;
    follower_entered = false;
    follower_wait_timed_out = false;
  }
  bool WaitForPreparedLeader() {
    std::unique_lock<std::mutex> lock(flight_mu);
    return flight_cv.wait_for(lock, std::chrono::seconds(30),
                              [this] { return leader_prepared; });
  }
  bool FollowerWaitTimedOut() {
    std::lock_guard<std::mutex> lock(flight_mu);
    return follower_wait_timed_out;
  }
};
}  // namespace

TEST(RdmaLoopback, RamTierZeroCopyServe) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device (load rdma_rxe for Soft-RoCE)";
  RamRdmaNode node("ram");
  ASSERT_TRUE(node.srv->ram_enabled());
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  // PUT then GET a spread of blocks over RDMA; each must round-trip byte-for-byte
  // even though the GET is served zero-copy from the arena MR.
  const int N = 40;
  std::vector<std::string> vals;
  for (int i = 0; i < N; ++i) {
    std::string v = "ram-rdma-" + std::to_string(i) + std::string(300 + i, 'q');
    vals.push_back(v);
    ASSERT_TRUE(c.Put("z" + std::to_string(i), v.data(), v.size())) << i;
  }
  for (int i = 0; i < N; ++i) {
    std::string out(vals[i].size(), '\0');
    ASSERT_TRUE(c.Get("z" + std::to_string(i), &out[0], out.size())) << i;
    EXPECT_EQ(out, vals[i]) << i;
  }

  const std::string m = node.srv->MetricsText();
  EXPECT_GT(CounterVal(m, "dfkv_ram_hit_total"), 0) << "GET must be served from RAM";
  // Every send completed, so no arena slot is left send-pinned: a re-GET still
  // works (pin released on IBV_WC_SEND, not leaked).
  std::string out2(vals[0].size(), '\0');
  ASSERT_TRUE(c.Get("z0", &out2[0], out2.size()));
  EXPECT_EQ(out2, vals[0]);
}

TEST(RdmaLoopback, ColdGetPromotesDirectlyIntoRegisteredRamArena) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device (load rdma_rxe for Soft-RoCE)";
  ::setenv("DFKV_READ_COALESCE", "1", 1);
  ::setenv("DFKV_SERVER_URING", "1", 1);
  {
    RamRdmaNode node("cold-direct-promotion", kMaxMsg,
                     /*writearound=*/true);
    ASSERT_TRUE(node.srv->ram_enabled());
    RdmaTransport rt(kMaxMsg);
    KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

    std::string value(256 * 1024, '\0');
    for (size_t i = 0; i < value.size(); ++i)
      value[i] = static_cast<char>('a' + (i % 19));
    ASSERT_TRUE(c.Put("cold-direct", value.data(), value.size()));
    EXPECT_EQ(CounterVal(node.srv->MetricsText(), "dfkv_ram_objects"), 0);

    std::string first(value.size(), '\0');
    ASSERT_TRUE(c.Get("cold-direct", first.data(), first.size()));
    EXPECT_EQ(first, value);
    const std::string after_cold = node.srv->MetricsText();
    EXPECT_EQ(CounterVal(after_cold, "dfkv_ram_promoted_total"), 1);
    EXPECT_EQ(CounterVal(after_cold, "dfkv_ram_objects"), 1);
#ifdef DFKV_WITH_URING
    EXPECT_GT(node.rsrv->UringReads(), 0u);
#endif

    const long hits_before =
        CounterVal(after_cold, "dfkv_ram_hit_total");
    std::string warm(value.size(), '\0');
    ASSERT_TRUE(c.Get("cold-direct", warm.data(), warm.size()));
    EXPECT_EQ(warm, value);
    EXPECT_GT(CounterVal(node.srv->MetricsText(), "dfkv_ram_hit_total"),
              hits_before);
  }
  ::unsetenv("DFKV_READ_COALESCE");
  ::unsetenv("DFKV_SERVER_URING");
}

TEST(RdmaLoopback, UringMultiWindowCoalescedFollowerPromotesToRam) {
#ifndef DFKV_WITH_URING
  GTEST_SKIP() << "test requires a DFKV_WITH_URING build";
#else
  if (!HaveRdma())
    GTEST_SKIP() << "no RDMA device (load rdma_rxe for Soft-RoCE)";
  ScopedEnv coalesce("DFKV_READ_COALESCE", "1");
  ScopedEnv coalesce_timeout("DFKV_READ_COALESCE_TIMEOUT_MS", "1000");
  ScopedEnv uring("DFKV_SERVER_URING", "1");
  ScopedEnv uring_depth("DFKV_SERVER_URING_DEPTH", "8");
  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");
  ScopedEnv recv_segment("DFKV_RDMA_RECV_SEGMENT_SIZE", "33554432");

  RamRdmaNode node("multi-window-coalesced-promotion", 64 * 1024,
                   /*writearound=*/true);
  ASSERT_TRUE(node.srv->ram_enabled());
  RdmaTransport seed_transport(64 * 1024);
  KVClient seed_client({{"n", node.addr}}, SelfHdr(), &seed_transport);
  seed_client.set_batch_concurrency(1);
  const size_t targets_per_window = seed_transport.MaxSgPayloadSegs();
  MultiWindowGet shape(targets_per_window, 3, 83);
  ASSERT_EQ(shape.window_count(targets_per_window), 3u);
  const std::string value = PatternValue(shape.capacity(), 89);
  ASSERT_TRUE(
      seed_client.Put("coalesced_multi", value.data(), value.size()));
  ASSERT_EQ(CounterVal(node.srv->MetricsText(), "dfkv_ram_objects"), 0);
  if (node.rsrv->UringInitFallbacks() != 0)
    GTEST_SKIP() << "io_uring unavailable";

  MultiWindowGet leader(targets_per_window, 3, 83);
  MultiWindowGet follower(targets_per_window, 3, 83);
  bool leader_ok = false;
  bool follower_ok = false;
  size_t leader_len = 0;
  size_t follower_len = 0;
  // Establish both SG-data QPs before holding the leader in preparation. Real
  // HCAs may spend seconds registering a fresh connection's memory; making that
  // setup part of the convoy race turned the old two-second barrier into a
  // scheduler/hardware-speed test. Missing-key probes warm the exact transport
  // lane without reading or promoting the value under test.
  RdmaTransport leader_transport(64 * 1024);
  KVClient leader_client({{"n", node.addr}}, SelfHdr(), &leader_transport);
  leader_client.set_batch_concurrency(1);
  RdmaTransport follower_transport(64 * 1024);
  KVClient follower_client({{"n", node.addr}}, SelfHdr(), &follower_transport);
  follower_client.set_batch_concurrency(1);
  auto warm_sg_connection = [&](KVClient* client, size_t seed) {
    MultiWindowGet probe(targets_per_window, 1, seed);
    std::vector<size_t> lengths;
    const auto result = client->BatchGetAutoSg(
        {{"coalesced_multi_missing", probe.pointers, probe.sizes}}, &lengths);
    return result.size() == 1 && !result[0] && lengths.size() == 1 &&
           lengths[0] == 0;
  };
  ASSERT_TRUE(warm_sg_connection(&leader_client, 91));
  ASSERT_TRUE(warm_sg_connection(&follower_client, 97));
  node.BarrierFirstPreparedRead("coalesced_multi");

  const uint64_t reads_before = node.rsrv->UringReads();

  std::thread leader_thread([&] {
    std::vector<size_t> lengths;
    const auto result = leader_client.BatchGetAutoSg(
        {{"coalesced_multi", leader.pointers, leader.sizes}}, &lengths);
    leader_ok = result.size() == 1 && result[0];
    if (lengths.size() == 1) leader_len = lengths[0];
  });
  EXPECT_TRUE(node.WaitForPreparedLeader())
      << "leader did not register its async coalescer flight";
  std::thread follower_thread([&] {
    std::vector<size_t> lengths;
    const auto result = follower_client.BatchGetAutoSg(
        {{"coalesced_multi", follower.pointers, follower.sizes}}, &lengths);
    follower_ok = result.size() == 1 && result[0];
    if (lengths.size() == 1) follower_len = lengths[0];
  });
  leader_thread.join();
  follower_thread.join();

  EXPECT_FALSE(node.FollowerWaitTimedOut())
      << "pre-warmed follower never entered the retained leader flight";
  ASSERT_TRUE(leader_ok);
  ASSERT_TRUE(follower_ok);
  EXPECT_EQ(leader_len, value.size());
  EXPECT_EQ(follower_len, value.size());
  EXPECT_EQ(leader.Flatten(), value);
  EXPECT_EQ(follower.Flatten(), value);
  EXPECT_EQ(node.rsrv->UringReads() - reads_before, 1u)
      << "the follower must consume the leader's published read";
  const std::string after_cold = node.srv->MetricsText();
  EXPECT_EQ(CounterVal(after_cold, "dfkv_read_coalesce_leaders_total"), 1);
  EXPECT_EQ(CounterVal(after_cold, "dfkv_read_coalesced_total"), 1);
  EXPECT_EQ(CounterVal(after_cold, "dfkv_ram_promoted_total"), 1);
  EXPECT_EQ(CounterVal(after_cold, "dfkv_ram_objects"), 1);

  // The promoted resident remains a multi-window source, but no longer needs a
  // disk descriptor. Its send pin still spans all continuations until final SEND.
  MultiWindowGet warm(targets_per_window, 3, 83);
  const uint64_t warm_reads_before = node.rsrv->UringReads();
  std::vector<size_t> warm_lengths;
  const auto warm_result = seed_client.BatchGetAutoSg(
      {{"coalesced_multi", warm.pointers, warm.sizes}}, &warm_lengths);
  ASSERT_EQ(warm_result.size(), 1u);
  ASSERT_TRUE(warm_result[0]);
  ASSERT_EQ(warm_lengths.size(), 1u);
  EXPECT_EQ(warm_lengths[0], value.size());
  EXPECT_EQ(warm.Flatten(), value);
  EXPECT_EQ(node.rsrv->UringReads(), warm_reads_before);
  EXPECT_GT(CounterVal(node.srv->MetricsText(), "dfkv_ram_hit_total"), 0);
#endif
}

// DCP2 declared caps: a client that tightens its max block size gets smaller
// shared slots and must still complete every op within the declaration;
// oversized ops fail client-side with kInvalid without touching the wire.
TEST(RdmaLoopback, DeclaredCapsRoundTripAndClientSideBound) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("caps");
  setenv("DFKV_RDMA_MAX_BLOCK_BYTES", "65536", 1);  // declare 64 KiB
  RdmaTransport rt(kMaxMsg);
  unsetenv("DFKV_RDMA_MAX_BLOCK_BYTES");            // don't leak into other tests
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  // Within the declaration: normal round-trip on the right-sized connection.
  std::string v(60 * 1024, 'a');
  ASSERT_TRUE(c.Put("caps-ok", v.data(), v.size()));
  std::string got(v.size(), '\0');
  ASSERT_TRUE(c.Get("caps-ok", got.data(), got.size()));
  EXPECT_EQ(got, v);

  // Over the declaration (but under the transport max): the client bound must
  // reject before sending -- Put returns false, connection stays healthy.
  std::string big(128 * 1024, 'b');
  EXPECT_FALSE(c.Put("caps-over", big.data(), big.size()));
  ASSERT_TRUE(c.Get("caps-ok", got.data(), got.size())) << "conn must survive the rejected op";
  EXPECT_EQ(got, v);
}

// With no explicit override, DCP2 declares the transport's safe global cap;
// blocks right up to that bound still round-trip.
TEST(RdmaLoopback, DefaultDeclarationKeepsGlobalCap) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("nocaps");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);
  std::string v(kMaxMsg - 1, 'c');  // near the global raw-value cap
  ASSERT_TRUE(c.Put("full-size", v.data(), v.size()));
  std::string got(v.size(), '\0');
  ASSERT_TRUE(c.Get("full-size", got.data(), got.size()));
  EXPECT_EQ(got, v);
}

TEST(RdmaLoopback,
     PinnedBouncePoolQuarantinesSlotAfterEnqueueAndRecoverySyncErrors) {
  const CudaLib* cuda = ActiveCudaForTest();
  if (!cuda) GTEST_SKIP() << "no usable CUDA device";

  // Leave healthy capacity after the deliberately poisoned lease so the test
  // can prove both quarantine and forward progress in the process-wide pool.
  ScopedEnv pool_bytes("DFKV_CUDA_PINNED_POOL_BYTES", "3145728");
  ScopedEnv slot_bytes("DFKV_CUDA_PINNED_SLOT_BYTES", "1048576");
  const PinnedBouncePoolStats before = GetPinnedBouncePoolStatsForTest();
  ASSERT_GT(before.slot_bytes, 0u);

  const std::string value = CudaTestBytes(257, 0x42);
  CudaGuardedBuffer failed_destination(cuda, value.size());
  CudaGuardedBuffer healthy_destination(cuda, value.size());
  ASSERT_TRUE(failed_destination.Allocate());
  ASSERT_TRUE(healthy_destination.Allocate());

  CUcontext destination_context = nullptr;
  ASSERT_TRUE(cuda->GetCurrentCtx(&destination_context));
  ASSERT_NE(destination_context, nullptr);
  ScopedCudaContextRestore restore_context(cuda, destination_context);
  ASSERT_TRUE(cuda->SetCurrentCtx(nullptr));
  CUcontext caller_context = destination_context;
  ASSERT_TRUE(cuda->GetCurrentCtx(&caller_context));
  ASSERT_EQ(caller_context, nullptr);

  ScopedCudaAsyncFault fault(cuda);
  DestinationPublisher failed_publisher;
  fault.FailNextEnqueueAndSynchronize();
  EXPECT_FALSE(failed_publisher.Copy(failed_destination.payload(), value.data(),
                                     value.size(),
                                     DestinationMemoryKind::kDevice));
  const CUdeviceptr quarantined_source = fault.last_source();
  ASSERT_NE(quarantined_source, 0u);
  EXPECT_EQ(fault.memcpy_async_calls(), 1u);
  EXPECT_EQ(fault.stream_synchronize_calls(), 1u)
      << "an enqueue error must fence the stream before releasing the lease";

  CUcontext context_after_failure = nullptr;
  ASSERT_TRUE(cuda->GetCurrentCtx(&context_after_failure));
  EXPECT_EQ(context_after_failure, caller_context);
  EXPECT_EQ(GetPinnedBouncePoolStatsForTest().active_leases, 0u);

  EXPECT_FALSE(failed_publisher.Finish());
  const uint64_t calls_after_finish = fault.stream_synchronize_calls();
  EXPECT_FALSE(failed_publisher.Finish());
  EXPECT_EQ(fault.stream_synchronize_calls(), calls_after_finish)
      << "Finish must not repeat teardown or lease handling";
  EXPECT_EQ(GetPinnedBouncePoolStatsForTest().active_leases, 0u);

  DestinationPublisher healthy_publisher;
  ASSERT_TRUE(healthy_publisher.Copy(healthy_destination.payload(), value.data(),
                                     value.size(),
                                     DestinationMemoryKind::kDevice));
  const CUdeviceptr healthy_source = fault.last_source();
  EXPECT_NE(healthy_source, quarantined_source)
      << "the failed recovery fence must permanently quarantine its slot";
  ASSERT_TRUE(healthy_publisher.Finish());
  EXPECT_EQ(GetPinnedBouncePoolStatsForTest().active_leases, 0u);
  CUcontext context_after_finish = destination_context;
  ASSERT_TRUE(cuda->GetCurrentCtx(&context_after_finish));
  EXPECT_EQ(context_after_finish, caller_context);

  ASSERT_TRUE(cuda->SetCurrentCtx(destination_context));

  std::string actual;
  ASSERT_TRUE(healthy_destination.ReadPayload(&actual));
  EXPECT_EQ(actual, value);
  EXPECT_TRUE(healthy_destination.GuardsIntact());
}

TEST(RdmaLoopback,
     PinnedBouncePoolChunksAndBackpressuresConcurrentCudaPublications) {
  const CudaLib* cuda = ActiveCudaForTest();
  if (!cuda) GTEST_SKIP() << "no usable CUDA device";

  // Keep this regression cheap when it is the first CUDA publication in the
  // process. If another test initialized the process-wide pool first, the
  // assertions below derive the already-fixed slot size and remain valid.
  ScopedEnv pool_bytes("DFKV_CUDA_PINNED_POOL_BYTES", "2097152");
  ScopedEnv slot_bytes("DFKV_CUDA_PINNED_SLOT_BYTES", "1048576");
  const PinnedBouncePoolStats before = GetPinnedBouncePoolStatsForTest();
  ASSERT_GT(before.slot_bytes, 0u);

  constexpr size_t kWorkers = 32;
  const size_t payload_size =
      std::max<size_t>(3052800, before.slot_bytes + 8197);
  auto destinations =
      std::make_shared<std::vector<std::unique_ptr<CudaGuardedBuffer>>>();
  destinations->reserve(kWorkers);
  for (size_t i = 0; i < kWorkers; ++i) {
    auto destination =
        std::make_unique<CudaGuardedBuffer>(cuda, payload_size);
    ASSERT_TRUE(destination->Allocate()) << "worker " << i;
    destinations->push_back(std::move(destination));
  }

  struct WaveState {
    std::mutex mutex;
    std::condition_variable cv;
    bool start = false;
    size_t ready = 0;
    size_t done = 0;
    std::vector<uint8_t> published;
    std::vector<uint8_t> context_restored;
  };

  const auto run_wave = [&](uint8_t seed) {
    auto value =
        std::make_shared<const std::string>(CudaTestBytes(payload_size, seed));
    auto state = std::make_shared<WaveState>();
    state->published.assign(kWorkers, 0);
    state->context_restored.assign(kWorkers, 0);
    std::vector<std::thread> workers;
    workers.reserve(kWorkers);
    for (size_t i = 0; i < kWorkers; ++i) {
      workers.emplace_back(
          [cuda, destinations, state, value, pool_slot_bytes = before.slot_bytes,
           i] {
        CUcontext context_before = nullptr;
        CUcontext context_after = nullptr;
        const bool queried_before = cuda->GetCurrentCtx(&context_before);
        {
          std::unique_lock<std::mutex> lock(state->mutex);
          ++state->ready;
          state->cv.notify_all();
          state->cv.wait(lock, [&] { return state->start; });
        }

        DestinationPublisher publisher;
        // The first logical SG segment is itself larger than one bounce slot;
        // the odd following boundaries catch chunk truncation and offset bugs.
        const std::array<size_t, 3> segments{
            pool_slot_bytes + 17,
            2053,
            value->size() - (pool_slot_bytes + 17 + 2053),
        };
        size_t offset = 0;
        bool published = true;
        for (size_t segment : segments) {
          published &=
              publisher.Copy(static_cast<char*>((*destinations)[i]->payload()) +
                                 offset,
                             value->data() + offset, segment,
                             DestinationMemoryKind::kDevice);
          offset += segment;
        }
        published &= publisher.Finish();
        const bool queried_after = cuda->GetCurrentCtx(&context_after);

        {
          std::lock_guard<std::mutex> lock(state->mutex);
          state->published[i] = published;
          state->context_restored[i] =
              queried_before && queried_after &&
              context_before == context_after;
          ++state->done;
        }
        state->cv.notify_all();
      });
    }

    bool all_ready = false;
    {
      std::unique_lock<std::mutex> lock(state->mutex);
      all_ready = state->cv.wait_for(
          lock, std::chrono::seconds(10),
          [&] { return state->ready == kWorkers; });
      state->start = true;
    }
    state->cv.notify_all();

    bool all_done = false;
    {
      std::unique_lock<std::mutex> lock(state->mutex);
      all_done = state->cv.wait_for(
          lock, std::chrono::seconds(30),
          [&] { return state->done == kWorkers; });
    }
    for (auto& worker : workers) {
      if (all_done)
        worker.join();
      else
        worker.detach();
    }

    EXPECT_TRUE(all_ready) << "publication workers missed start deadline";
    EXPECT_TRUE(all_done) << "bounded pool deadlocked under backpressure";
    if (!all_ready || !all_done) return false;
    EXPECT_EQ(state->published,
              std::vector<uint8_t>(kWorkers, 1));
    EXPECT_EQ(state->context_restored,
              std::vector<uint8_t>(kWorkers, 1));
    for (size_t i = 0; i < kWorkers; ++i) {
      std::string actual;
      const bool read_ok = (*destinations)[i]->ReadPayload(&actual);
      EXPECT_TRUE(read_ok) << "worker " << i;
      if (read_ok) EXPECT_EQ(actual, *value) << "worker " << i;
      EXPECT_TRUE((*destinations)[i]->GuardsIntact()) << "worker " << i;
    }
    return true;
  };

  ASSERT_TRUE(run_wave(0x51));
  const PinnedBouncePoolStats warm = GetPinnedBouncePoolStatsForTest();
  EXPECT_GT(warm.allocated_slots, 0u);
  EXPECT_LT(warm.allocated_slots, kWorkers)
      << "the wave must exceed the bounded slot count";
  EXPECT_EQ(warm.active_leases, 0u);
  EXPECT_GT(warm.peak_active_leases, 0u);
  EXPECT_GT(warm.wait_count, before.wait_count)
      << "oversubscription must take the blocking backpressure path";

  ASSERT_TRUE(run_wave(0xb7));
  const PinnedBouncePoolStats reused = GetPinnedBouncePoolStatsForTest();
  EXPECT_EQ(reused.active_leases, 0u);
  EXPECT_EQ(reused.allocated_slots, warm.allocated_slots);
  EXPECT_EQ(reused.allocation_calls, warm.allocation_calls);
  EXPECT_EQ(reused.registration_calls, warm.registration_calls)
      << "all CUDA pinning must be one-time slot warmup, not per publication";
}

TEST(RdmaLoopback, ScalarGetPublishesExactBytesToGuardedCudaDestination) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const CudaLib* cuda = ActiveCudaForTest();
  if (!cuda) GTEST_SKIP() << "no usable CUDA device";
  ScopedEnv host_dedup("DFKV_CLIENT_NODE_DEDUP", nullptr);
  ScopedEnv gpu_dedup("DFKV_CLIENT_NODE_DEDUP_GPU", nullptr);
  ScopedEnv completion_fault("DFKV_RDMA_TEST_COMPLETION_FAULT", nullptr);
  ScopedEnv endpoint_fault("DFKV_RDMA_TEST_ENDPOINT_COMPLETION_FAULT",
                           nullptr);
  ScopedEnv local_failure("DFKV_RDMA_TEST_LOCAL_RAIL_FAILURE_ATTEMPTS",
                          nullptr);

  RdmaNode node("cuda-scalar-get");
  RdmaTransport transport(kMaxMsg);
  KVClient client({{"n", node.addr}}, SelfHdr(), &transport);
  const std::string value = CudaTestBytes(8197, 0x31);
  ASSERT_TRUE(client.Put("cuda-scalar", value.data(), value.size()));

  CudaGuardedBuffer destination(cuda, value.size());
  ASSERT_TRUE(destination.Allocate());
  ASSERT_TRUE(cuda->IsDevicePtr(destination.payload()));
  ASSERT_TRUE(client.Get("cuda-scalar", destination.payload(), value.size()));

  // Read back immediately after Get returns. The API contract is synchronous:
  // its private nonblocking CUDA stream must already be synchronized here.
  std::string actual;
  ASSERT_TRUE(destination.ReadPayload(&actual));
  EXPECT_EQ(actual, value);
  EXPECT_TRUE(destination.GuardsIntact());
}

TEST(RdmaLoopback,
     MultiSegmentGetPublishesUnequalExactBytesToGuardedCudaDestinations) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  const CudaLib* cuda = ActiveCudaForTest();
  if (!cuda) GTEST_SKIP() << "no usable CUDA device";
  ScopedEnv host_dedup("DFKV_CLIENT_NODE_DEDUP", nullptr);
  ScopedEnv gpu_dedup("DFKV_CLIENT_NODE_DEDUP_GPU", nullptr);
  ScopedEnv completion_fault("DFKV_RDMA_TEST_COMPLETION_FAULT", nullptr);
  ScopedEnv endpoint_fault("DFKV_RDMA_TEST_ENDPOINT_COMPLETION_FAULT",
                           nullptr);
  ScopedEnv local_failure("DFKV_RDMA_TEST_LOCAL_RAIL_FAILURE_ATTEMPTS",
                          nullptr);

  // Host first is deliberate: treating the whole SG item as host from its
  // first segment crashes on the following CUDA pointer. The remaining three
  // CUDA segments have unequal, odd capacities and independent guards.
  constexpr std::array<size_t, 4> kSegmentSizes{613, 997, 2051, 4099};
  const size_t total = kSegmentSizes[0] + kSegmentSizes[1] +
                       kSegmentSizes[2] + kSegmentSizes[3];
  const std::string value = CudaTestBytes(total, 0x6d);

  RdmaNode node("cuda-multisegment-get");
  RdmaTransport transport(kMaxMsg);
  KVClient client({{"n", node.addr}}, SelfHdr(), &transport);
  ASSERT_TRUE(client.Put("cuda-multisegment", value.data(), value.size()));

  const std::string host_guard(
      CudaGuardedBuffer::kGuardBytes,
      static_cast<char>(CudaGuardedBuffer::kGuardByte));
  std::string host_allocation(
      kSegmentSizes[0] + 2 * CudaGuardedBuffer::kGuardBytes,
      static_cast<char>(CudaGuardedBuffer::kGuardByte));
  char* const host_payload =
      host_allocation.data() + CudaGuardedBuffer::kGuardBytes;

  std::array<std::unique_ptr<CudaGuardedBuffer>, 3> device_buffers;
  KvGetItemSg get;
  get.key = "cuda-multisegment";
  get.dsts.push_back(host_payload);
  get.caps.push_back(kSegmentSizes[0]);
  for (size_t i = 0; i < device_buffers.size(); ++i) {
    const size_t segment = i + 1;
    device_buffers[i] =
        std::make_unique<CudaGuardedBuffer>(cuda, kSegmentSizes[segment]);
    ASSERT_TRUE(device_buffers[i]->Allocate()) << "segment " << segment;
    ASSERT_TRUE(cuda->IsDevicePtr(device_buffers[i]->payload()))
        << "segment " << segment;
    get.dsts.push_back(device_buffers[i]->payload());
    get.caps.push_back(kSegmentSizes[segment]);
  }

  std::vector<size_t> lengths;
  ASSERT_EQ(client.BatchGetAutoSg({get}, &lengths),
            std::vector<bool>({true}));
  ASSERT_EQ(lengths, std::vector<size_t>({value.size()}));

  // Every immediate read observes API-return state. Odd, unequal boundaries
  // expose word-sized over-copy, and host-first ordering proves memory kind is
  // classified per segment rather than once per SG item.
  EXPECT_EQ(std::string(host_payload, kSegmentSizes[0]),
            value.substr(0, kSegmentSizes[0]));
  EXPECT_EQ(host_allocation.substr(0, CudaGuardedBuffer::kGuardBytes),
            host_guard);
  EXPECT_EQ(host_allocation.substr(CudaGuardedBuffer::kGuardBytes +
                                   kSegmentSizes[0]),
            host_guard);
  size_t offset = kSegmentSizes[0];
  for (size_t i = 0; i < device_buffers.size(); ++i) {
    const size_t segment = i + 1;
    std::string actual;
    ASSERT_TRUE(device_buffers[i]->ReadPayload(&actual))
        << "segment " << segment;
    EXPECT_EQ(actual, value.substr(offset, kSegmentSizes[segment]))
        << "segment " << segment;
    EXPECT_TRUE(device_buffers[i]->GuardsIntact())
        << "segment " << segment;
    offset += kSegmentSizes[segment];
  }
}

TEST(RdmaLoopback,
     PooledPostSubmitFailureDoesNotReplayPutAndGetRemainsReplaySafe) {
  ScopedEnv configured_device("DFKV_RDMA_DEV", nullptr);
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv ambient_completion_fault("DFKV_RDMA_TEST_COMPLETION_FAULT",
                                     nullptr);
  ScopedEnv ambient_local_failure(
      "DFKV_RDMA_TEST_LOCAL_RAIL_FAILURE_ATTEMPTS", nullptr);
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  RdmaNode node("pooled-post-submit");
  const std::string value("pooled\0put-value", 16);

  const auto expect_single_pooled_put_attempt =
      [](const std::string& before, const std::string& after,
         long expected_posts) {
        EXPECT_EQ(CounterVal(after, "dfkv_rdma_endpoint_cache_hits_total") -
                      CounterVal(before,
                                 "dfkv_rdma_endpoint_cache_hits_total"),
                  1);
        EXPECT_EQ(CounterVal(after, "dfkv_rdma_endpoint_cache_misses_total") -
                      CounterVal(before,
                                 "dfkv_rdma_endpoint_cache_misses_total"),
                  0);
        EXPECT_EQ(
            CounterVal(after, "dfkv_rdma_client_v2_put_writes_total") -
                CounterVal(before, "dfkv_rdma_client_v2_put_writes_total"),
            expected_posts);
        EXPECT_EQ(
            CounterVal(after,
                       "dfkv_rdma_client_stale_pool_retries_total") -
                CounterVal(before,
                           "dfkv_rdma_client_stale_pool_retries_total"),
            0);
        EXPECT_EQ(
            CounterVal(after,
                       "dfkv_rdma_client_cross_rail_retries_total") -
                CounterVal(before,
                           "dfkv_rdma_client_cross_rail_retries_total"),
            0);
      };

  {
    RdmaTransport transport(kMaxMsg);
    const BlockKey warm_key =
        ToBlockKey(SelfHdr(), "pooled-scalar-warm");
    const BlockKey failed_key =
        ToBlockKey(SelfHdr(), "pooled-scalar-failed");
    ASSERT_EQ(transport.Cache(node.addr, warm_key, value.data(), value.size()),
              Status::kOk);
    const std::string before = transport.MetricsText();
    {
      ScopedEnv fault("DFKV_RDMA_TEST_COMPLETION_FAULT", "1:1:1");
      EXPECT_EQ(
          transport.Cache(node.addr, failed_key, value.data(), value.size()),
          Status::kIOError);
    }
    const std::string after = transport.MetricsText();
    expect_single_pooled_put_attempt(before, after, 1);
    EXPECT_EQ(node.CacheDirectCalls(failed_key), 1u);
  }

  {
    RdmaTransport transport(kMaxMsg);
    const BlockKey warm_key =
        ToBlockKey(SelfHdr(), "pooled-batch-warm");
    const std::vector<BlockKey> failed_keys{
        ToBlockKey(SelfHdr(), "pooled-batch-failed-0"),
        ToBlockKey(SelfHdr(), "pooled-batch-failed-1"),
    };
    ASSERT_EQ(transport.CacheFrom(
                  node.addr, {{warm_key, value.data(), value.size()}}),
              std::vector<Status>({Status::kOk}));
    const std::vector<CacheSrc> sources{
        {failed_keys[0], value.data(), value.size()},
        {failed_keys[1], value.data(), value.size()},
    };
    const std::string before = transport.MetricsText();
    {
      ScopedEnv fault("DFKV_RDMA_TEST_COMPLETION_FAULT", "1:1:1");
      EXPECT_EQ(transport.CacheFrom(node.addr, sources),
                std::vector<Status>(sources.size(), Status::kIOError));
    }
    const std::string after = transport.MetricsText();
    expect_single_pooled_put_attempt(before, after, sources.size());
    for (const BlockKey& key : failed_keys) {
      EXPECT_EQ(node.CacheDirectCalls(key), 1u);
    }
  }

  {
    RdmaTransport transport(kMaxMsg);
    const BlockKey warm_key =
        ToBlockKey(SelfHdr(), "pooled-sg-warm");
    const BlockKey failed_key =
        ToBlockKey(SelfHdr(), "pooled-sg-failed");
    const std::array<std::string, 2> segments{
        std::string("first\0segment", 13),
        std::string("second\0segment", 14),
    };
    CacheSrcMulti warm;
    warm.key = warm_key;
    warm.payloads = {{value.data(), value.size()}};
    ASSERT_EQ(transport.CacheFromMulti(node.addr, {warm}),
              std::vector<Status>({Status::kOk}));
    CacheSrcMulti source;
    source.key = failed_key;
    source.payloads = {
        {segments[0].data(), segments[0].size()},
        {segments[1].data(), segments[1].size()},
    };
    const std::string before = transport.MetricsText();
    {
      ScopedEnv fault("DFKV_RDMA_TEST_COMPLETION_FAULT", "1:1:1");
      EXPECT_EQ(transport.CacheFromMulti(node.addr, {source}),
                std::vector<Status>({Status::kIOError}));
    }
    const std::string after = transport.MetricsText();
    expect_single_pooled_put_attempt(before, after, 1);
    EXPECT_EQ(node.CacheDirectCalls(failed_key), 1u);
  }

  {
    RdmaTransport transport(kMaxMsg);
    const BlockKey key = ToBlockKey(SelfHdr(), "pooled-get-retry");
    ASSERT_EQ(transport.Cache(node.addr, key, value.data(), value.size()),
              Status::kOk);
    const std::string before = transport.MetricsText();
    std::string output;
    {
      ScopedEnv fault("DFKV_RDMA_TEST_COMPLETION_FAULT", "1:1:1");
      EXPECT_EQ(transport.Range(node.addr, key, 0, value.size(), &output,
                                nullptr),
                Status::kOk);
    }
    EXPECT_EQ(output, value);
    const std::string after = transport.MetricsText();
    EXPECT_EQ(CounterVal(after, "dfkv_rdma_endpoint_cache_hits_total") -
                  CounterVal(before, "dfkv_rdma_endpoint_cache_hits_total"),
              1);
    EXPECT_EQ(CounterVal(after, "dfkv_rdma_endpoint_cache_misses_total") -
                  CounterVal(before,
                             "dfkv_rdma_endpoint_cache_misses_total"),
              1);
    EXPECT_EQ(
        CounterVal(after, "dfkv_rdma_client_v2_get_writes_total") -
            CounterVal(before, "dfkv_rdma_client_v2_get_writes_total"),
        2);
    EXPECT_EQ(node.RangeDirectCalls(key), 2u);
  }
}

TEST(RdmaLoopback, ClientLocalRailFailureRetriesFreshPutAndGet) {
  if (!HaveRdma())
    GTEST_SKIP() << "no RDMA device (load two rdma_rxe devices for this test)";
  const auto rails = ConfiguredTwoTestRails();
  if (!HaveConfiguredActiveRails(rails))
    GTEST_SKIP() << "set DFKV_RDMA_DEV to exactly two ACTIVE test devices";
  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");
  ScopedEnv credits("DFKV_RDMA_RAIL_CREDITS", kRealHcaRailCredits);
  ScopedEnv threshold("DFKV_RDMA_RAIL_ERROR_THRESHOLD", "100");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv local_failure_fault(
      "DFKV_RDMA_TEST_LOCAL_RAIL_FAILURE_ATTEMPTS", nullptr);
  RdmaNode node("cross-rail-scalar");
  RdmaTransport transport(kMaxMsg, rails[0] + "," + rails[1]);
  KVClient client({{"n", node.addr}}, SelfHdr(), &transport);
  const std::string value(64 * 1024, 'p');

  std::string before = transport.MetricsText();
  const long stale_before =
      CounterVal(before, "dfkv_rdma_client_stale_pool_retries_total");
  {
    ScopedEnv fault("DFKV_RDMA_TEST_LOCAL_RAIL_FAILURE_ATTEMPTS", "1");
    ASSERT_TRUE(client.Put("fresh-put", value.data(), value.size()));
  }
  std::string after = transport.MetricsText();
  ExpectTwoDistinctRailAttempts(before, after, rails, 1);
  ExpectReleasedRailResources(before, after, rails);
  EXPECT_EQ(
      CounterVal(after, "dfkv_rdma_endpoint_cache_hits_total") -
          CounterVal(before, "dfkv_rdma_endpoint_cache_hits_total"),
      0)
      << "the first local failure must be exercised on a fresh endpoint";
  EXPECT_EQ(
      CounterVal(after, "dfkv_rdma_endpoint_cache_misses_total") -
          CounterVal(before, "dfkv_rdma_endpoint_cache_misses_total"),
      2);
  EXPECT_EQ(
      CounterVal(after, "dfkv_rdma_client_cross_rail_retries_total") -
          CounterVal(before, "dfkv_rdma_client_cross_rail_retries_total"),
      1);
  EXPECT_EQ(
      CounterVal(after,
                 "dfkv_rdma_client_cross_rail_retry_successes_total") -
          CounterVal(before,
                     "dfkv_rdma_client_cross_rail_retry_successes_total"),
      1);
  EXPECT_EQ(
      CounterVal(after,
                 "dfkv_rdma_client_cross_rail_retry_exhausted_total") -
          CounterVal(before,
                     "dfkv_rdma_client_cross_rail_retry_exhausted_total"),
      0);
  EXPECT_EQ(CounterVal(after,
                       "dfkv_rdma_client_stale_pool_retries_total"),
            stale_before);

  // Attempt 2 of the PUT left a healthy idle QP on its rail. Admission chooses
  // a rail from current NUMA locality and health before consulting that rail's
  // pool, so the GET may start either pooled or fresh. Its injected local
  // failure must still retire that endpoint and retry on the other local rail.
  before = after;
  const long get_stale_before =
      CounterVal(before, "dfkv_rdma_client_stale_pool_retries_total");
  std::string output(value.size(), '\0');
  {
    ScopedEnv fault("DFKV_RDMA_TEST_COMPLETION_FAULT", "1:1:1");
    ASSERT_TRUE(client.Get("fresh-put", output.data(), output.size()));
  }
  EXPECT_EQ(output, value);
  after = transport.MetricsText();
  ExpectTwoDistinctRailAttempts(before, after, rails, 1);
  ExpectReleasedRailResources(before, after, rails);
  const long get_endpoint_misses =
      CounterVal(after, "dfkv_rdma_endpoint_cache_misses_total") -
      CounterVal(before, "dfkv_rdma_endpoint_cache_misses_total");
  EXPECT_GE(get_endpoint_misses, 1);
  EXPECT_LE(get_endpoint_misses, 2)
      << "two rail attempts may require at most one fresh endpoint per rail";
  EXPECT_EQ(
      CounterVal(after, "dfkv_rdma_client_cross_rail_retries_total") -
          CounterVal(before, "dfkv_rdma_client_cross_rail_retries_total"),
      1);
  EXPECT_EQ(
      CounterVal(after,
                 "dfkv_rdma_client_cross_rail_retry_successes_total") -
          CounterVal(before,
                     "dfkv_rdma_client_cross_rail_retry_successes_total"),
      1);
  EXPECT_EQ(CounterVal(after,
                       "dfkv_rdma_client_stale_pool_retries_total"),
            get_stale_before)
      << "a local-rail failure is not a stale-endpoint retry";
}

TEST(RdmaLoopback,
     ReplaySafeTwoRailRetryPublishesExactBytesToCudaDestination) {
  if (!HaveRdma())
    GTEST_SKIP() << "no RDMA device (load two active devices for this test)";
  const auto rails = ConfiguredTwoTestRails();
  if (!HaveConfiguredActiveRails(rails))
    GTEST_SKIP() << "set DFKV_RDMA_DEV to exactly two ACTIVE test devices";
  const CudaLib* cuda = ActiveCudaForTest();
  if (!cuda) GTEST_SKIP() << "no usable CUDA device";

  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");
  ScopedEnv credits("DFKV_RDMA_RAIL_CREDITS", kRealHcaRailCredits);
  ScopedEnv threshold("DFKV_RDMA_RAIL_ERROR_THRESHOLD", "100");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv ambient_completion_fault("DFKV_RDMA_TEST_COMPLETION_FAULT",
                                     nullptr);
  ScopedEnv ambient_endpoint_fault(
      "DFKV_RDMA_TEST_ENDPOINT_COMPLETION_FAULT", nullptr);
  ScopedEnv ambient_local_failure(
      "DFKV_RDMA_TEST_LOCAL_RAIL_FAILURE_ATTEMPTS", nullptr);
  ScopedEnv host_dedup("DFKV_CLIENT_NODE_DEDUP", nullptr);
  ScopedEnv gpu_dedup("DFKV_CLIENT_NODE_DEDUP_GPU", nullptr);

  RdmaNode node("cuda-cross-rail-get");
  RdmaTransport transport(kMaxMsg, rails[0] + "," + rails[1]);
  const BlockKey key = ToBlockKey(SelfHdr(), "cuda-cross-rail");
  const std::string value = CudaTestBytes(64 * 1024 + 3, 0x93);
  ASSERT_EQ(transport.Cache(node.addr, key, value.data(), value.size()),
            Status::kOk);

  CudaGuardedBuffer destination(cuda, value.size());
  ASSERT_TRUE(destination.Allocate());
  ASSERT_TRUE(cuda->IsDevicePtr(destination.payload()));
  CUcontext caller_context = nullptr;
  ASSERT_TRUE(cuda->GetCurrentCtx(&caller_context));
  ASSERT_NE(caller_context, nullptr);

  std::mutex gate_mu;
  std::condition_variable gate_cv;
  bool winning_attempt_entered = false;
  bool release_winner = false;
  node.before_range = [&](size_t call) {
    if (call != 2) return;
    std::unique_lock<std::mutex> lock(gate_mu);
    winning_attempt_entered = true;
    gate_cv.notify_one();
    gate_cv.wait(lock, [&] { return release_winner; });
  };

  const std::string before = transport.MetricsText();
  const long stale_before =
      CounterVal(before, "dfkv_rdma_client_stale_pool_retries_total");
  std::vector<Status> statuses;
  std::vector<uint64_t> value_lens;
  bool context_setup_ok = false;
  bool context_query_ok = false;
  CUcontext context_before = nullptr;
  CUcontext context_after = nullptr;
  ScopedEnv fault("DFKV_RDMA_TEST_COMPLETION_FAULT", "1:1:1");
  std::thread getter([&] {
    context_setup_ok = cuda->SetCurrentCtx(caller_context) &&
                       cuda->GetCurrentCtx(&context_before);
    statuses = transport.RangeInto(
        node.addr, {key},
        {{destination.payload(), value.size(), DestinationMemoryKind::kDevice}},
        &value_lens);
    context_query_ok = cuda->GetCurrentCtx(&context_after);
  });

  bool observed_winner = false;
  {
    std::unique_lock<std::mutex> lock(gate_mu);
    observed_winner = gate_cv.wait_for(
        lock, std::chrono::seconds(10),
        [&] { return winning_attempt_entered; });
  }
  if (observed_winner) {
    // Attempt one completed its direct RDMA write and DONE before the injected
    // local completion fault. Retirement proof makes that committed payload
    // stable while the second rail is still blocked.
    std::string before_winner;
    const bool read_ok = destination.ReadPayload(&before_winner);
    EXPECT_TRUE(read_ok);
    if (read_ok) EXPECT_EQ(before_winner, value);
    EXPECT_TRUE(destination.GuardsIntact());
  }
  {
    std::lock_guard<std::mutex> lock(gate_mu);
    release_winner = true;
  }
  gate_cv.notify_all();
  getter.join();
  ASSERT_TRUE(observed_winner) << "winning rail did not enter before deadline";
  EXPECT_TRUE(context_setup_ok);
  EXPECT_TRUE(context_query_ok);
  EXPECT_EQ(context_before, caller_context);
  EXPECT_EQ(context_after, caller_context);
  EXPECT_EQ(statuses, std::vector<Status>({Status::kOk}));
  EXPECT_EQ(value_lens, std::vector<uint64_t>({value.size()}));

  // Successful return is the publication fence: readback must need no
  // application-side stream synchronization.
  std::string actual;
  ASSERT_TRUE(destination.ReadPayload(&actual));
  EXPECT_EQ(actual, value);
  EXPECT_TRUE(destination.GuardsIntact());
  const std::string after = transport.MetricsText();
  ExpectTwoDistinctRailAttempts(before, after, rails, 1);
  ExpectReleasedRailResources(before, after, rails);
  EXPECT_EQ(
      CounterVal(after, "dfkv_rdma_client_v2_get_writes_total") -
          CounterVal(before, "dfkv_rdma_client_v2_get_writes_total"),
      2);
  EXPECT_EQ(
      CounterVal(after, "dfkv_rdma_client_cross_rail_retries_total") -
          CounterVal(before, "dfkv_rdma_client_cross_rail_retries_total"),
      1);
  EXPECT_EQ(
      CounterVal(after,
                 "dfkv_rdma_client_cross_rail_retry_successes_total") -
          CounterVal(before,
                     "dfkv_rdma_client_cross_rail_retry_successes_total"),
      1);
  EXPECT_EQ(
      CounterVal(after,
                 "dfkv_rdma_client_cross_rail_retry_exhausted_total") -
          CounterVal(before,
                     "dfkv_rdma_client_cross_rail_retry_exhausted_total"),
      0);
  EXPECT_EQ(
      CounterVal(after, "dfkv_rdma_client_stale_pool_retries_total"),
      stale_before);
}

TEST(RdmaLoopback, SingleEnabledRailFallbackIsNotCrossRailRetry) {
  if (!HaveRdma())
    GTEST_SKIP() << "no RDMA device (load two rdma_rxe devices for this test)";
  const auto rails = ConfiguredTwoTestRails();
  if (!HaveConfiguredActiveRails(rails))
    GTEST_SKIP() << "set DFKV_RDMA_DEV to exactly two ACTIVE test devices";
  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");
  ScopedEnv credits("DFKV_RDMA_RAIL_CREDITS", kRealHcaRailCredits);
  ScopedEnv threshold("DFKV_RDMA_RAIL_ERROR_THRESHOLD", "100");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv local_failure_fault(
      "DFKV_RDMA_TEST_LOCAL_RAIL_FAILURE_ATTEMPTS", nullptr);
  RdmaNode node("same-rail-fallback");
  RdmaTransport transport(kMaxMsg, rails[0]);
  KVClient client({{"n", node.addr}}, SelfHdr(), &transport);
  const std::string value(64 * 1024, 's');

  const std::string before = transport.MetricsText();
  {
    ScopedEnv fault("DFKV_RDMA_TEST_LOCAL_RAIL_FAILURE_ATTEMPTS", "1");
    ASSERT_TRUE(client.Put("same-rail", value.data(), value.size()));
  }
  const std::string after = transport.MetricsText();
  EXPECT_EQ(
      DeviceCounterVal(after, "dfkv_rdma_client_rail_selections_total",
                       rails[0]) -
          DeviceCounterVal(before,
                           "dfkv_rdma_client_rail_selections_total",
                           rails[0]),
      2);
  EXPECT_EQ(DeviceCounterVal(after, "dfkv_rdma_client_rail_errors_total",
                             rails[0]) -
                DeviceCounterVal(before,
                                 "dfkv_rdma_client_rail_errors_total",
                                 rails[0]),
            1);
  EXPECT_EQ(
      CounterVal(after, "dfkv_rdma_client_cross_rail_retries_total") -
          CounterVal(before, "dfkv_rdma_client_cross_rail_retries_total"),
      0);
  EXPECT_EQ(
      CounterVal(after,
                 "dfkv_rdma_client_cross_rail_retry_successes_total") -
          CounterVal(before,
                     "dfkv_rdma_client_cross_rail_retry_successes_total"),
      0);
  EXPECT_EQ(
      CounterVal(after,
                 "dfkv_rdma_client_cross_rail_retry_exhausted_total") -
          CounterVal(before,
                     "dfkv_rdma_client_cross_rail_retry_exhausted_total"),
      0);
  ExpectReleasedRailResources(before, after, {rails[0]});
}

TEST(RdmaLoopback,
     ClientCompletionFailurePreservesCompletedMultiWindowSgItems) {
  if (!HaveRdma())
    GTEST_SKIP() << "no RDMA device (load two rdma_rxe devices for this test)";
  const auto rails = ConfiguredTwoTestRails();
  if (!HaveConfiguredActiveRails(rails))
    GTEST_SKIP() << "set DFKV_RDMA_DEV to exactly two ACTIVE test devices";
  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");
  ScopedEnv credits("DFKV_RDMA_RAIL_CREDITS", kRealHcaRailCredits);
  ScopedEnv threshold("DFKV_RDMA_RAIL_ERROR_THRESHOLD", "100");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv local_failure_fault(
      "DFKV_RDMA_TEST_LOCAL_RAIL_FAILURE_ATTEMPTS", nullptr);
  // PUT uses the pre-post local-failure seam, so its retry may safely replay
  // every item. GET advances both items in rounds: item 0 completes in round 2
  // and completion window 3 faults only the still-incomplete item.
  RdmaNode node("cross-rail-sg");
  RdmaTransport transport(kMaxMsg, rails[0] + "," + rails[1]);
  KVClient client({{"n", node.addr}}, SelfHdr(), &transport);

  constexpr size_t kItems = 2;
  constexpr size_t kSegmentBytes = 31;
  const size_t max_sg = transport.MaxSgPayloadSegs();
  ASSERT_GE(max_sg, 2u);
  std::vector<std::vector<std::string>> segments(kItems);
  segments[0].resize(max_sg + 1);
  segments[1].resize(3 * max_sg + 1);
  std::vector<std::string> expected(kItems);
  std::vector<KvPutItemSg> puts;
  puts.reserve(kItems);
  std::vector<BlockKey> block_keys;
  block_keys.reserve(kItems);
  for (size_t i = 0; i < kItems; ++i) {
    std::vector<const void*> pointers;
    std::vector<size_t> sizes;
    for (size_t segment = 0; segment < segments[i].size(); ++segment) {
      // BatchGetAutoSg groups by total capacity. Make both totals equal so the
      // two- and four-window items share one RangeIntoMulti logical call;
      // segment count, rather than byte count, still determines the windows.
      const size_t segment_bytes =
          i == 0 && segment == 0
              ? (2 * max_sg + 1) * kSegmentBytes
              : kSegmentBytes;
      segments[i][segment] =
          std::string(segment_bytes,
                      static_cast<char>('A' + i * 13 + segment % 13));
      expected[i] += segments[i][segment];
      pointers.push_back(segments[i][segment].data());
      sizes.push_back(segments[i][segment].size());
    }
    const std::string key = "sg-" + std::to_string(i);
    puts.push_back({key, std::move(pointers), std::move(sizes)});
    block_keys.push_back(ToBlockKey(SelfHdr(), key));
  }

  std::string before = transport.MetricsText();
  {
    ScopedEnv fault("DFKV_RDMA_TEST_LOCAL_RAIL_FAILURE_ATTEMPTS", "1");
    EXPECT_EQ(client.BatchPutSg(puts), std::vector<bool>(kItems, true));
  }
  std::string after = transport.MetricsText();
  ExpectTwoDistinctRailAttempts(before, after, rails, 1);
  ExpectReleasedRailResources(before, after, rails);
  EXPECT_EQ(node.CacheDirectCalls(block_keys[0]), 1u)
      << "pre-post failure must not duplicate the first PUT item";
  EXPECT_EQ(node.CacheDirectCalls(block_keys[1]), 1u)
      << "pre-post failure must not duplicate the second PUT item";

  std::vector<std::vector<std::string>> output(kItems);
  std::vector<KvGetItemSg> gets;
  gets.reserve(kItems);
  std::vector<size_t> expected_lengths;
  expected_lengths.reserve(kItems);
  for (size_t i = 0; i < kItems; ++i) {
    std::vector<void*> pointers;
    std::vector<size_t> capacities;
    size_t total = 0;
    output[i].resize(segments[i].size());
    for (size_t segment = 0; segment < segments[i].size(); ++segment) {
      output[i][segment].assign(segments[i][segment].size(), '\0');
      pointers.push_back(output[i][segment].data());
      capacities.push_back(output[i][segment].size());
      total += output[i][segment].size();
    }
    gets.push_back(
        {puts[i].key, std::move(pointers), std::move(capacities)});
    expected_lengths.push_back(total);
  }
  ASSERT_EQ(expected_lengths[0], expected_lengths[1])
      << "SG retry items must remain in one transport group";

  before = after;
  std::vector<size_t> lengths;
  {
    ScopedEnv fault("DFKV_RDMA_TEST_COMPLETION_FAULT", "1:1:3");
    EXPECT_EQ(client.BatchGetAutoSg(gets, &lengths),
              std::vector<bool>(kItems, true));
  }
  EXPECT_EQ(lengths, expected_lengths);
  after = transport.MetricsText();
  ExpectTwoDistinctRailAttempts(before, after, rails, 1);
  ExpectReleasedRailResources(before, after, rails);
  EXPECT_EQ(node.RangeDirectCalls(block_keys[0]), 1u)
      << "the completed GET item must not restart on the fresh rail";
  EXPECT_EQ(node.RangeDirectCalls(block_keys[1]), 2u)
      << "only the incomplete GET item must restart from window zero";
  for (size_t i = 0; i < kItems; ++i) {
    std::string actual;
    for (const auto& segment : output[i]) actual += segment;
    EXPECT_EQ(actual, expected[i]) << i;
  }
  EXPECT_EQ(CounterVal(after, "dfkv_rdma_client_cross_rail_retries_total"),
            2);
  EXPECT_EQ(
      CounterVal(after,
                 "dfkv_rdma_client_cross_rail_retry_successes_total"),
      2);
  EXPECT_EQ(
      CounterVal(after,
                 "dfkv_rdma_client_cross_rail_retry_exhausted_total"),
      0);
  EXPECT_EQ(CounterVal(after,
                       "dfkv_rdma_client_stale_pool_retries_total"),
            0);
}

TEST(RdmaLoopback, ConcurrentOperationsRemainIsolatedAcrossLocalRailRetry) {
  if (!HaveRdma())
    GTEST_SKIP() << "no RDMA device (load two rdma_rxe devices for this test)";
  const auto rails = ConfiguredTwoTestRails();
  if (!HaveConfiguredActiveRails(rails))
    GTEST_SKIP() << "set DFKV_RDMA_DEV to exactly two ACTIVE test devices";
  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");
  ScopedEnv credits("DFKV_RDMA_RAIL_CREDITS", kRealHcaRailCredits);
  ScopedEnv threshold("DFKV_RDMA_RAIL_ERROR_THRESHOLD", "100");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv local_failure_fault(
      "DFKV_RDMA_TEST_LOCAL_RAIL_FAILURE_ATTEMPTS", nullptr);
  ScopedEnv ambient_completion_fault("DFKV_RDMA_TEST_COMPLETION_FAULT",
                                     nullptr);
  RdmaNode node("cross-rail-concurrent");
  RdmaTransport transport(kMaxMsg, rails[0] + "," + rails[1]);
  KVClient client({{"n", node.addr}}, "test/cross-rail-concurrent",
                  &transport);

  constexpr size_t kCallers = 32;
  constexpr auto kConcurrencyDeadline = std::chrono::seconds(30);
  constexpr size_t kValueBytes = 64 * 1024;
  std::vector<std::string> values(kCallers);
  for (size_t i = 0; i < kCallers; ++i) {
    values[i].resize(kValueBytes);
    for (size_t byte = 0; byte < kValueBytes; ++byte)
      values[i][byte] =
          static_cast<char>((i * 31 + byte * 17) & 0xff);
  }
  std::vector<std::string> outputs(
      kCallers, std::string(kValueBytes, '\0'));
  for (size_t i = 0; i < kCallers; ++i) {
    ASSERT_TRUE(client.Put("concurrent-" + std::to_string(i),
                           values[i].data(), values[i].size()))
        << i;
  }
  std::vector<char> get_ok(kCallers, 0);
  std::mutex gate_mu;
  std::condition_variable ready_cv;
  std::condition_variable start_cv;
  std::condition_variable completed_cv;
  size_t ready = 0;
  size_t completed = 0;
  bool start = false;
  const auto deadline =
      std::chrono::steady_clock::now() + kConcurrencyDeadline;
  std::vector<std::thread> callers;
  callers.reserve(kCallers);
  const std::string before = transport.MetricsText();
  {
    ScopedEnv fault("DFKV_RDMA_TEST_COMPLETION_FAULT", "1:1:1");
    for (size_t i = 0; i < kCallers; ++i) {
      callers.emplace_back([&, i] {
        std::unique_lock<std::mutex> lock(gate_mu);
        ++ready;
        ready_cv.notify_one();
        if (!start_cv.wait_until(lock, deadline, [&] { return start; })) {
          ++completed;
          completed_cv.notify_one();
          return;
        }
        lock.unlock();
        get_ok[i] =
            client.Get("concurrent-" + std::to_string(i), outputs[i].data(),
                       outputs[i].size())
                ? 1
                : 0;
        lock.lock();
        ++completed;
        completed_cv.notify_one();
      });
    }
    bool all_ready = false;
    {
      std::unique_lock<std::mutex> lock(gate_mu);
      all_ready =
          ready_cv.wait_until(lock, deadline, [&] { return ready == kCallers; });
      start = true;
    }
    start_cv.notify_all();
    EXPECT_TRUE(all_ready) << "callers did not reach the start barrier";
    {
      std::unique_lock<std::mutex> lock(gate_mu);
      EXPECT_TRUE(completed_cv.wait_until(
          lock, deadline, [&] { return completed == kCallers; }))
          << "concurrent RDMA calls deadlocked";
    }
    for (auto& caller : callers) caller.join();
  }
  EXPECT_EQ(get_ok, std::vector<char>(kCallers, 1));

  const std::string after = transport.MetricsText();
  long selection_delta = 0;
  long error_delta = 0;
  for (const auto& rail : rails) {
    selection_delta +=
        DeviceCounterVal(after, "dfkv_rdma_client_rail_selections_total",
                         rail) -
        DeviceCounterVal(before, "dfkv_rdma_client_rail_selections_total",
                         rail);
    error_delta +=
        DeviceCounterVal(after, "dfkv_rdma_client_rail_errors_total", rail) -
        DeviceCounterVal(before, "dfkv_rdma_client_rail_errors_total", rail);
  }
  EXPECT_EQ(selection_delta, static_cast<long>(kCallers + 1))
      << "exactly one logical call must have two rail attempts";
  EXPECT_EQ(error_delta, 1);
  EXPECT_EQ(
      CounterVal(after, "dfkv_rdma_client_cross_rail_retries_total") -
          CounterVal(before, "dfkv_rdma_client_cross_rail_retries_total"),
      1);
  EXPECT_EQ(
      CounterVal(after,
                 "dfkv_rdma_client_cross_rail_retry_successes_total") -
          CounterVal(before,
                     "dfkv_rdma_client_cross_rail_retry_successes_total"),
      1);
  EXPECT_EQ(
      CounterVal(after,
                 "dfkv_rdma_client_cross_rail_retry_exhausted_total") -
          CounterVal(before,
                     "dfkv_rdma_client_cross_rail_retry_exhausted_total"),
      0);
  ExpectReleasedRailResources(before, after, rails);

  for (size_t i = 0; i < kCallers; ++i)
    EXPECT_EQ(outputs[i], values[i]) << i;
  ExpectReleasedRailResources(after, transport.MetricsText(), rails);
}

TEST(RdmaLoopback, AllLocalRailsFailOnceAndReclaimOperationResources) {
  if (!HaveRdma())
    GTEST_SKIP() << "no RDMA device (load two rdma_rxe devices for this test)";
  const auto rails = ConfiguredTwoTestRails();
  if (!HaveConfiguredActiveRails(rails))
    GTEST_SKIP() << "set DFKV_RDMA_DEV to exactly two ACTIVE test devices";
  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");
  ScopedEnv credits("DFKV_RDMA_RAIL_CREDITS", kRealHcaRailCredits);
  ScopedEnv threshold("DFKV_RDMA_RAIL_ERROR_THRESHOLD", "100");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv local_failure_fault(
      "DFKV_RDMA_TEST_LOCAL_RAIL_FAILURE_ATTEMPTS", nullptr);
  ScopedEnv ambient_completion_fault("DFKV_RDMA_TEST_COMPLETION_FAULT",
                                     nullptr);
  RdmaNode node("cross-rail-exhausted");
  RdmaTransport transport(kMaxMsg, rails[0] + "," + rails[1]);
  const std::string key_namespace = "test/cross-rail-exhausted";
  KVClient client({{"n", node.addr}}, key_namespace, &transport);
  const std::string value(64 * 1024, 'x');

  std::string before = transport.MetricsText();
  {
    ScopedEnv fault("DFKV_RDMA_TEST_LOCAL_RAIL_FAILURE_ATTEMPTS", "1,2");
    EXPECT_FALSE(client.Put("failed-put", value.data(), value.size()));
  }
  std::string after = transport.MetricsText();
  ExpectTwoDistinctRailAttempts(before, after, rails, 2);
  ExpectReleasedRailResources(before, after, rails);
  EXPECT_EQ(
      CounterVal(after, "dfkv_rdma_client_cross_rail_retries_total") -
          CounterVal(before, "dfkv_rdma_client_cross_rail_retries_total"),
      1);
  EXPECT_EQ(
      CounterVal(after,
                 "dfkv_rdma_client_cross_rail_retry_successes_total") -
          CounterVal(before,
                     "dfkv_rdma_client_cross_rail_retry_successes_total"),
      0);
  EXPECT_EQ(
      CounterVal(after,
                 "dfkv_rdma_client_cross_rail_retry_exhausted_total") -
          CounterVal(before,
                     "dfkv_rdma_client_cross_rail_retry_exhausted_total"),
      1);
  std::vector<char> exists;
  // Both PUT failures are injected before posting, so retry exhaustion cannot
  // leave an ambiguous server-side commit.
  EXPECT_EQ(transport.ExistMany(
                node.addr, {ToBlockKey(key_namespace, "failed-put")}, &exists),
            std::vector<Status>({Status::kNotFound}));
  EXPECT_EQ(exists, std::vector<char>({0}));

  // Seed a readable value with both fault seams disabled, then fail both local
  // attempts of a fresh public GET. Server DMA lands directly in the caller
  // destination, so a faulted completion may leave partial payload behind; the
  // fence must guarantee the buffer stops changing once the failed GET
  // returns.
  KVClient writer({{"n", node.addr}}, key_namespace, &transport);
  ASSERT_TRUE(writer.Put("failed-get", value.data(), value.size()));
  KVClient reader({{"n", node.addr}}, key_namespace, &transport);
  std::string output(value.size(), static_cast<char>(0x5a));
  before = transport.MetricsText();
  {
    ScopedEnv fault("DFKV_RDMA_TEST_COMPLETION_FAULT",
                    "1:1:1,1:2:1");
    EXPECT_FALSE(reader.Get("failed-get", output.data(), output.size()));
  }
  after = transport.MetricsText();
  ExpectTwoDistinctRailAttempts(before, after, rails, 2);
  ExpectReleasedRailResources(before, after, rails);
  const std::string output_at_return = output;
  EXPECT_EQ(
      CounterVal(after, "dfkv_rdma_client_cross_rail_retries_total") -
          CounterVal(before, "dfkv_rdma_client_cross_rail_retries_total"),
      1);
  EXPECT_EQ(
      CounterVal(after,
                 "dfkv_rdma_client_cross_rail_retry_successes_total") -
          CounterVal(before,
                     "dfkv_rdma_client_cross_rail_retry_successes_total"),
      0);
  EXPECT_EQ(
      CounterVal(after,
                 "dfkv_rdma_client_cross_rail_retry_exhausted_total") -
          CounterVal(before,
                     "dfkv_rdma_client_cross_rail_retry_exhausted_total"),
      1);
  EXPECT_EQ(CounterVal(after,
                       "dfkv_rdma_client_stale_pool_retries_total"),
            CounterVal(before,
                       "dfkv_rdma_client_stale_pool_retries_total"));

  // Drive another synchronous control completion before checking the buffer.
  // Responder-retirement proof must precede retry/return, so no old writer can
  // land bytes after the failed GET returned.
  EXPECT_EQ(transport.ExistMany(
                node.addr, {ToBlockKey(key_namespace, "failed-get")}, &exists),
            std::vector<Status>({Status::kOk}));
  EXPECT_EQ(output, output_at_return);
}

// The SG read path shares the direct-write contract: failed attempts retire the
// responder writer and drain its signaled WRITE CQEs before retry/return. A
// later successful GET rewrites the item in full.
TEST(RdmaLoopback, FailedSgGetLeavesCallerSegmentsStableAfterReturn) {
  if (!HaveRdma())
    GTEST_SKIP() << "no RDMA device (load two rdma_rxe devices for this test)";
  const auto rails = ConfiguredTwoTestRails();
  if (!HaveConfiguredActiveRails(rails))
    GTEST_SKIP() << "set DFKV_RDMA_DEV to exactly two ACTIVE test devices";
  ScopedEnv depth("DFKV_RDMA_DEPTH", "4");
  ScopedEnv credits("DFKV_RDMA_RAIL_CREDITS", kRealHcaRailCredits);
  ScopedEnv threshold("DFKV_RDMA_RAIL_ERROR_THRESHOLD", "100");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  RdmaNode node("sg-failed-get-fence");
  RdmaTransport transport(kMaxMsg, rails[0] + "," + rails[1]);
  KVClient client({{"n", node.addr}}, SelfHdr(), &transport);
  const std::string value(64 * 1024, 'q');
  ASSERT_TRUE(client.Put("sg-failed-get", value.data(), value.size()));

  std::string first(4096, '\x5a');
  std::string second(value.size() - first.size(), '\x5a');
  KvGetItemSg get{"sg-failed-get", {first.data(), second.data()},
                  {first.size(), second.size()}};
  std::vector<size_t> lengths;
  {
    ScopedEnv fault("DFKV_RDMA_TEST_COMPLETION_FAULT", "1:1:1,1:2:1");
    EXPECT_FALSE(client.BatchGetAutoSg({get}, &lengths)[0]);
  }
  // Partial payload may already sit in the segments (unspecified contents on
  // failure); what the fence guarantees is that they stop changing here.
  const std::string first_at_return = first;
  const std::string second_at_return = second;

  // Drive another synchronous control completion; retirement proof must have
  // made every old writer terminal before the failed call returned.
  std::vector<char> exists;
  EXPECT_EQ(transport.ExistMany(
                node.addr, {ToBlockKey(SelfHdr(), "sg-failed-get")}, &exists),
            std::vector<Status>({Status::kOk}));
  EXPECT_EQ(first, first_at_return);
  EXPECT_EQ(second, second_at_return);

  // A clean retry through the same destinations must rewrite the item in
  // full: no first-attempt residue can leak through a successful GET.
  ASSERT_TRUE(client.BatchGetAutoSg({get}, &lengths)[0]);
  ASSERT_EQ(lengths[0], value.size());
  EXPECT_EQ(first + second, value);
}

TEST(RdmaLoopback,
     BlockedResponderWriteRetiresBeforeHostRetryAndReclaimsMr) {
  ScopedEnv configured_device("DFKV_RDMA_DEV", nullptr);
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";

  RdmaNode node("writer-retire-retry");
  RdmaTransport transport(kMaxMsg);
  const BlockKey key = ToBlockKey(SelfHdr(), "writer-retire-retry");
  const std::string value = PatternValue(64 * 1024 + 7, 91);
  ASSERT_EQ(transport.Cache(node.addr, key, value.data(), value.size()),
            Status::kOk);
  const uint64_t mr_baseline = rdma::RcEndpoint::TransientUserMrActive();
  std::string output(value.size(), '\x5a');
  std::vector<Status> statuses;
  std::vector<uint64_t> value_lens;
  std::atomic<bool> returned{false};

  rdma::TestBlockNextResponderWrites(1);
  std::thread getter([&] {
    statuses = transport.RangeInto(
        node.addr, {key}, {{output.data(), output.size()}}, &value_lens);
    returned.store(true, std::memory_order_release);
  });
  const bool blocked =
      rdma::TestWaitForBlockedResponderWrites(1, 10000);
  const bool cancelled =
      blocked && rdma::TestWaitForResponderWriteCancellations(1, 10000);
  EXPECT_TRUE(blocked);
  EXPECT_TRUE(cancelled);
  EXPECT_FALSE(returned.load(std::memory_order_acquire))
      << "client returned before responder-retirement proof";
  EXPECT_GT(rdma::RcEndpoint::TransientUserMrActive(), mr_baseline)
      << "blocked direct attempt did not retain its host MR";
  EXPECT_EQ(output, std::string(output.size(), '\x5a'))
      << "blocked old attempt wrote before retirement";

  rdma::TestReleaseOneBlockedResponderWrite();
  const bool old_writer_exited =
      rdma::TestWaitForReleasedResponderWrites(1, 10000);
  if (!old_writer_exited) rdma::TestReleaseBlockedResponderWrites();
  getter.join();
  rdma::TestReleaseBlockedResponderWrites();

  EXPECT_TRUE(old_writer_exited);
  EXPECT_EQ(statuses, std::vector<Status>({Status::kOk}));
  EXPECT_EQ(value_lens, std::vector<uint64_t>({value.size()}));
  EXPECT_EQ(output, value);
  EXPECT_EQ(node.RangeDirectCalls(key), 2u);
  EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), mr_baseline);

  const std::string at_return = output;
  std::vector<char> exists;
  EXPECT_EQ(transport.ExistMany(node.addr, {key}, &exists),
            std::vector<Status>({Status::kOk}));
  EXPECT_EQ(output, at_return)
      << "retired first-attempt writer modified host memory after return";
  EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), mr_baseline);
}

TEST(RdmaLoopback,
     TwoBlockedResponderAttemptsFailSgGetBeforeReturnAndReclaimMrs) {
  ScopedEnv configured_device("DFKV_RDMA_DEV", nullptr);
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";

  RdmaNode node("writer-retire-terminal-sg");
  RdmaTransport transport(kMaxMsg);
  const BlockKey key = ToBlockKey(SelfHdr(), "writer-retire-terminal-sg");
  const std::string value = PatternValue(96 * 1024 + 11, 37);
  ASSERT_EQ(transport.Cache(node.addr, key, value.data(), value.size()),
            Status::kOk);
  const uint64_t mr_baseline = rdma::RcEndpoint::TransientUserMrActive();

  std::array<std::string, 3> segments{
      std::string(4093, '\x5a'),
      std::string(32771, '\x5a'),
      std::string(value.size() - 4093 - 32771, '\x5a')};
  RangeDstMulti destination{{
      {segments[0].data(), segments[0].size()},
      {segments[1].data(), segments[1].size()},
      {segments[2].data(), segments[2].size()},
  }};
  std::vector<Status> statuses;
  std::vector<size_t> lengths;
  std::atomic<bool> returned{false};

  rdma::TestBlockNextResponderWrites(2);
  std::thread getter([&] {
    statuses =
        transport.RangeIntoMulti(node.addr, {key}, {destination}, &lengths);
    returned.store(true, std::memory_order_release);
  });
  bool fixture_ok = true;
  for (size_t attempt = 1; attempt <= 2; ++attempt) {
    const bool blocked =
        rdma::TestWaitForBlockedResponderWrites(attempt, 10000);
    const bool cancelled =
        blocked &&
        rdma::TestWaitForResponderWriteCancellations(attempt, 10000);
    EXPECT_TRUE(blocked) << attempt;
    EXPECT_TRUE(cancelled) << attempt;
    fixture_ok = fixture_ok && blocked && cancelled;
    EXPECT_FALSE(returned.load(std::memory_order_acquire))
        << "attempt " << attempt << " returned before writer retirement";
    EXPECT_GT(rdma::RcEndpoint::TransientUserMrActive(), mr_baseline)
        << "blocked SG attempt did not retain operation MRs";
    for (const auto& segment : segments)
      EXPECT_EQ(segment, std::string(segment.size(), '\x5a')) << attempt;
    rdma::TestReleaseOneBlockedResponderWrite();
    const bool exited =
        rdma::TestWaitForReleasedResponderWrites(attempt, 10000);
    EXPECT_TRUE(exited) << attempt;
    fixture_ok = fixture_ok && exited;
    if (!fixture_ok) break;
  }
  if (!fixture_ok) rdma::TestReleaseBlockedResponderWrites();
  getter.join();
  rdma::TestReleaseBlockedResponderWrites();

  EXPECT_EQ(statuses, std::vector<Status>({Status::kIOError}));
  EXPECT_EQ(lengths, std::vector<size_t>({0}));
  EXPECT_EQ(node.RangeDirectCalls(key), 2u);
  EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), mr_baseline);
  const auto at_return = segments;
  std::vector<char> exists;
  EXPECT_EQ(transport.ExistMany(node.addr, {key}, &exists),
            std::vector<Status>({Status::kOk}));
  EXPECT_EQ(segments, at_return)
      << "retired terminal writers modified SG memory after return";

  const auto recovered =
      transport.RangeIntoMulti(node.addr, {key}, {destination}, &lengths);
  ASSERT_EQ(recovered, std::vector<Status>({Status::kOk}));
  ASSERT_EQ(lengths, std::vector<size_t>({value.size()}));
  EXPECT_EQ(segments[0] + segments[1] + segments[2], value);
  EXPECT_EQ(rdma::RcEndpoint::TransientUserMrActive(), mr_baseline);
}

TEST(RdmaLoopback,
     EndpointCompletionCoolsOnlyFailedPeerRailAndPublicGetReplays) {
  if (!HaveRdma())
    GTEST_SKIP() << "no RDMA device (load two rdma_rxe devices for this test)";
  const auto rails = ConfiguredTwoTestRails();
  if (!HaveConfiguredActiveRails(rails))
    GTEST_SKIP() << "set DFKV_RDMA_DEV to exactly two ACTIVE test devices";
  ScopedEnv cooldown("DFKV_RDMA_REMOTE_RAIL_COOLDOWN_MS", "30000");
  ScopedEnv max_cooldown("DFKV_RDMA_REMOTE_RAIL_MAX_COOLDOWN_MS", "30000");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv credits("DFKV_RDMA_RAIL_CREDITS", kRealHcaRailCredits);
  ScopedEnv ambient_local_completion_fault(
      "DFKV_RDMA_TEST_COMPLETION_FAULT", nullptr);
  ScopedEnv ambient_endpoint_completion_fault(
      "DFKV_RDMA_TEST_ENDPOINT_COMPLETION_FAULT", nullptr);

  RdmaNode failed_peer("remote-completion-failed-peer");
  RdmaNode independent_peer("remote-completion-independent-peer");
  const BlockKey failed_peer_key =
      ToBlockKey(SelfHdr(), "remote-completion-failed-peer");
  const BlockKey independent_peer_key =
      ToBlockKey(SelfHdr(), "remote-completion-independent-peer");
  const std::string failed_peer_value(64 * 1024, 'f');
  const std::string independent_peer_value(4096, 'i');
  {
    // Seed server state through public operations on a separate transport so
    // the transport under test begins with deterministic rail-zero admission.
    RdmaTransport writer(kMaxMsg, rails[0]);
    ASSERT_EQ(writer.Cache(failed_peer.addr, failed_peer_key,
                           failed_peer_value.data(), failed_peer_value.size()),
              Status::kOk);
    ASSERT_EQ(writer.Cache(independent_peer.addr, independent_peer_key,
                           independent_peer_value.data(),
                           independent_peer_value.size()),
              Status::kOk);
  }

  RdmaTransport transport(kMaxMsg, rails[0] + "," + rails[1]);
  PeerTopology failed_topology;
  failed_topology.peer_addr = failed_peer.addr;
  failed_topology.peer_id = "remote-completion-failed-peer-id";
  failed_topology.generation = 1;
  failed_topology.complete = true;
  for (const auto& rail : rails)
    failed_topology.rails.push_back(PeerRailTopology{rail, true});
  transport.OnPeerTopology(failed_topology);

  PeerTopology independent_topology;
  independent_topology.peer_addr = independent_peer.addr;
  independent_topology.peer_id = "remote-completion-independent-peer-id";
  independent_topology.generation = 1;
  independent_topology.complete = true;
  independent_topology.rails.push_back(PeerRailTopology{rails[0], true});
  independent_topology.rails.push_back(PeerRailTopology{rails[1], false});
  transport.OnPeerTopology(independent_topology);
  KVClient failed_client({{"n", failed_peer.addr}}, SelfHdr(), &transport);
  KVClient independent_client({{"n", independent_peer.addr}}, SelfHdr(),
                              &transport);

  const std::string before_fault = transport.MetricsText();
  std::string output(failed_peer_value.size(), '\0');
  {
    ScopedEnv fault("DFKV_RDMA_TEST_ENDPOINT_COMPLETION_FAULT", "1:1:1");
    ASSERT_TRUE(failed_client.Get("remote-completion-failed-peer",
                                  output.data(), output.size()));
  }
  EXPECT_EQ(output, failed_peer_value);
  const std::string after_fault = transport.MetricsText();

  EXPECT_EQ(CounterVal(after_fault,
                       "dfkv_rdma_client_remote_rail_failures_total") -
                CounterVal(before_fault,
                           "dfkv_rdma_client_remote_rail_failures_total"),
            1)
      << "the failed production lease must complete exactly once";
  EXPECT_EQ(CounterVal(after_fault,
                       "dfkv_rdma_client_remote_rail_cooldowns_total") -
                CounterVal(before_fault,
                           "dfkv_rdma_client_remote_rail_cooldowns_total"),
            1)
      << "one endpoint failure must cause one cooldown transition";
  EXPECT_EQ(
      CounterVal(after_fault,
                 "dfkv_rdma_client_remote_rail_admissions_denied_total"),
      CounterVal(before_fault,
                 "dfkv_rdma_client_remote_rail_admissions_denied_total"))
      << "the injected completion is endpoint evidence, not admission failure";
  EXPECT_EQ(
      DeviceCounterVal(
          after_fault,
          "dfkv_rdma_client_remote_rail_failures_by_rail_total", rails[0]) -
          DeviceCounterVal(
              before_fault,
              "dfkv_rdma_client_remote_rail_failures_by_rail_total", rails[0]),
      1);
  EXPECT_EQ(
      DeviceCounterVal(
          after_fault,
          "dfkv_rdma_client_remote_rail_cooldowns_by_rail_total", rails[0]) -
          DeviceCounterVal(
              before_fault,
              "dfkv_rdma_client_remote_rail_cooldowns_by_rail_total", rails[0]),
      1);
  EXPECT_EQ(
      DeviceCounterVal(
          after_fault,
          "dfkv_rdma_client_remote_rail_failures_by_rail_total", rails[1]),
      DeviceCounterVal(
          before_fault,
          "dfkv_rdma_client_remote_rail_failures_by_rail_total", rails[1]));
  EXPECT_EQ(
      DeviceCounterVal(
          after_fault,
          "dfkv_rdma_client_remote_rail_cooldowns_by_rail_total", rails[1]),
      DeviceCounterVal(
          before_fault,
          "dfkv_rdma_client_remote_rail_cooldowns_by_rail_total", rails[1]));
  EXPECT_EQ(
      DeviceCounterVal(after_fault, "dfkv_rdma_client_rail_errors_total",
                       rails[0]),
      DeviceCounterVal(before_fault, "dfkv_rdma_client_rail_errors_total",
                       rails[0]))
      << "endpoint evidence must not blame the local rail";
  EXPECT_EQ(
      DeviceCounterVal(after_fault, "dfkv_rdma_client_endpoint_errors_total",
                       rails[0]) -
          DeviceCounterVal(before_fault,
                           "dfkv_rdma_client_endpoint_errors_total", rails[0]),
      1);
  for (const auto& rail : rails) {
    EXPECT_EQ(
        DeviceCounterVal(after_fault,
                         "dfkv_rdma_client_rail_selections_total", rail) -
            DeviceCounterVal(before_fault,
                             "dfkv_rdma_client_rail_selections_total", rail),
        1)
        << rail << ": GET must fail once then replay on its sibling rail";
  }
  EXPECT_EQ(
      CounterVal(after_fault, "dfkv_rdma_client_cross_rail_retries_total") -
          CounterVal(before_fault,
                     "dfkv_rdma_client_cross_rail_retries_total"),
      1);
  EXPECT_EQ(
      CounterVal(
          after_fault,
          "dfkv_rdma_client_cross_rail_retry_successes_total") -
          CounterVal(
              before_fault,
              "dfkv_rdma_client_cross_rail_retry_successes_total"),
      1);

  const std::string before_same_peer = after_fault;
  EXPECT_TRUE(failed_client.Exist("remote-completion-failed-peer"));
  const std::string after_same_peer = transport.MetricsText();
  EXPECT_EQ(
      DeviceCounterVal(after_same_peer,
                       "dfkv_rdma_client_rail_selections_total", rails[0]),
      DeviceCounterVal(before_same_peer,
                       "dfkv_rdma_client_rail_selections_total", rails[0]))
      << "the next public operation must avoid the cooled peer/rail pair";
  EXPECT_EQ(
      DeviceCounterVal(after_same_peer,
                       "dfkv_rdma_client_rail_selections_total", rails[1]) -
          DeviceCounterVal(before_same_peer,
                           "dfkv_rdma_client_rail_selections_total", rails[1]),
      1)
      << "the sibling rail remains eligible for the failed peer";

  const std::string before_independent_peer = after_same_peer;
  EXPECT_TRUE(
      independent_client.Exist("remote-completion-independent-peer"));
  const std::string after_independent_peer = transport.MetricsText();
  EXPECT_EQ(
      DeviceCounterVal(after_independent_peer,
                       "dfkv_rdma_client_rail_selections_total", rails[0]) -
          DeviceCounterVal(before_independent_peer,
                           "dfkv_rdma_client_rail_selections_total", rails[0]),
      1)
      << "remote cooldown is scoped by peer and must not disable this local "
         "rail for another peer";
  EXPECT_EQ(
      DeviceCounterVal(after_independent_peer,
                       "dfkv_rdma_client_rail_errors_total", rails[0]),
      DeviceCounterVal(before_fault, "dfkv_rdma_client_rail_errors_total",
                       rails[0]));
  EXPECT_EQ(CounterVal(after_independent_peer,
                       "dfkv_rdma_client_remote_rail_failures_total") -
                CounterVal(before_fault,
                           "dfkv_rdma_client_remote_rail_failures_total"),
            1)
      << "later successful public operations must not double-complete failure";
  EXPECT_EQ(CounterVal(after_independent_peer,
                       "dfkv_rdma_client_remote_rail_cooldowns_total") -
                CounterVal(before_fault,
                           "dfkv_rdma_client_remote_rail_cooldowns_total"),
            1);
}

TEST(RdmaLoopback,
     PostSubmitPutFailureCoolsActualRemoteRailWithoutReplay) {
  if (!HaveRdma())
    GTEST_SKIP() << "no RDMA device (load two rdma_rxe devices for this test)";
  const auto rails = ConfiguredTwoTestRails();
  if (!HaveConfiguredActiveRails(rails))
    GTEST_SKIP() << "set DFKV_RDMA_DEV to exactly two ACTIVE test devices";
  ScopedEnv cooldown("DFKV_RDMA_REMOTE_RAIL_COOLDOWN_MS", "30000");
  ScopedEnv max_cooldown("DFKV_RDMA_REMOTE_RAIL_MAX_COOLDOWN_MS", "30000");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv credits("DFKV_RDMA_RAIL_CREDITS", kRealHcaRailCredits);
  ScopedEnv ambient_local_completion_fault(
      "DFKV_RDMA_TEST_COMPLETION_FAULT", nullptr);
  ScopedEnv ambient_endpoint_completion_fault(
      "DFKV_RDMA_TEST_ENDPOINT_COMPLETION_FAULT", nullptr);
  ScopedEnv ambient_local_failure(
      "DFKV_RDMA_TEST_LOCAL_RAIL_FAILURE_ATTEMPTS", nullptr);

  RdmaNode node("remote-post-submit-put");
  RdmaTransport transport(kMaxMsg, rails[0] + "," + rails[1]);
  PeerTopology topology;
  topology.peer_addr = node.addr;
  topology.peer_id = "remote-post-submit-put-peer";
  topology.generation = 1;
  topology.complete = true;
  for (const auto& rail : rails)
    topology.rails.push_back(PeerRailTopology{rail, true});
  transport.OnPeerTopology(topology);

  const std::string key_namespace = "test/remote-post-submit-put";
  KVClient client({{"n", node.addr}}, key_namespace, &transport);
  const std::string value(64 * 1024, 'f');
  const BlockKey failed_key =
      ToBlockKey(key_namespace, "completion-failed-put");
  const std::string before_fault = transport.MetricsText();
  {
    ScopedEnv fault("DFKV_RDMA_TEST_ENDPOINT_COMPLETION_FAULT", "1:1:1");
    EXPECT_FALSE(
        client.Put("completion-failed-put", value.data(), value.size()));
  }
  const std::string after_fault = transport.MetricsText();

  EXPECT_EQ(CounterVal(after_fault, "dfkv_rdma_client_v2_put_writes_total") -
                CounterVal(before_fault,
                           "dfkv_rdma_client_v2_put_writes_total"),
            1)
      << "an unsafe PUT must not be posted again after ambiguous completion";
  EXPECT_EQ(node.CacheDirectCalls(failed_key), 1u)
      << "the server must observe exactly the single posted PUT";
  EXPECT_EQ(
      CounterVal(after_fault, "dfkv_rdma_client_cross_rail_retries_total") -
          CounterVal(before_fault,
                     "dfkv_rdma_client_cross_rail_retries_total"),
      0)
      << "post-submit PUT failure is not replay-safe";

  size_t failed_rail = rails.size();
  long failed_selections = 0;
  for (size_t rail = 0; rail < rails.size(); ++rail) {
    const long selected =
        DeviceCounterVal(after_fault,
                         "dfkv_rdma_client_rail_selections_total",
                         rails[rail]) -
        DeviceCounterVal(before_fault,
                         "dfkv_rdma_client_rail_selections_total",
                         rails[rail]);
    EXPECT_TRUE(selected == 0 || selected == 1) << rails[rail];
    if (selected == 1) failed_rail = rail;
    failed_selections += selected;
  }
  ASSERT_EQ(failed_selections, 1);
  ASSERT_LT(failed_rail, rails.size());
  const size_t sibling_rail = 1 - failed_rail;
  EXPECT_EQ(
      DeviceCounterVal(
          after_fault,
          "dfkv_rdma_client_remote_rail_failures_by_rail_total",
          rails[failed_rail]) -
          DeviceCounterVal(
              before_fault,
              "dfkv_rdma_client_remote_rail_failures_by_rail_total",
              rails[failed_rail]),
      1);
  EXPECT_EQ(
      DeviceCounterVal(
          after_fault,
          "dfkv_rdma_client_remote_rail_cooldowns_by_rail_total",
          rails[failed_rail]) -
          DeviceCounterVal(
              before_fault,
              "dfkv_rdma_client_remote_rail_cooldowns_by_rail_total",
              rails[failed_rail]),
      1)
      << "the endpoint completion must cool the rail that owned its lease";
  EXPECT_EQ(
      DeviceCounterVal(
          after_fault,
          "dfkv_rdma_client_remote_rail_cooldowns_by_rail_total",
          rails[sibling_rail]) -
          DeviceCounterVal(
              before_fault,
              "dfkv_rdma_client_remote_rail_cooldowns_by_rail_total",
              rails[sibling_rail]),
      0);

  KVClient future_client({{"n", node.addr}}, key_namespace, &transport);
  const BlockKey future_key = ToBlockKey(key_namespace, "future-put");
  ASSERT_TRUE(future_client.Put("future-put", value.data(), value.size()));
  const std::string after_future = transport.MetricsText();
  EXPECT_EQ(
      DeviceCounterVal(after_future,
                       "dfkv_rdma_client_rail_selections_total",
                       rails[failed_rail]) -
          DeviceCounterVal(after_fault,
                           "dfkv_rdma_client_rail_selections_total",
                           rails[failed_rail]),
      0)
      << "future admission must bypass the cooled endpoint rail";
  EXPECT_EQ(
      DeviceCounterVal(after_future,
                       "dfkv_rdma_client_rail_selections_total",
                       rails[sibling_rail]) -
          DeviceCounterVal(after_fault,
                           "dfkv_rdma_client_rail_selections_total",
                           rails[sibling_rail]),
      1);
  EXPECT_EQ(
      CounterVal(after_future, "dfkv_rdma_client_v2_put_writes_total") -
          CounterVal(after_fault, "dfkv_rdma_client_v2_put_writes_total"),
      1);
  EXPECT_EQ(node.CacheDirectCalls(future_key), 1u);
}

TEST(RdmaLoopback, RemoteCooldownBypassesIdlePoolAndUsesSiblingRail) {
  if (!HaveRdma())
    GTEST_SKIP() << "no RDMA device (load two rdma_rxe devices for this test)";
  const auto rails = ConfiguredTwoTestRails();
  if (!HaveConfiguredActiveRails(rails))
    GTEST_SKIP() << "set DFKV_RDMA_DEV to exactly two ACTIVE test devices";
  ScopedEnv cooldown("DFKV_RDMA_REMOTE_RAIL_COOLDOWN_MS", "30000");
  ScopedEnv max_cooldown("DFKV_RDMA_REMOTE_RAIL_MAX_COOLDOWN_MS", "30000");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv credits("DFKV_RDMA_RAIL_CREDITS", kRealHcaRailCredits);

  RdmaNode node("remote-idle-bypass");
  RdmaTransport transport(kMaxMsg, rails[0] + "," + rails[1]);
  PeerTopology topology;
  topology.peer_addr = node.addr;
  topology.peer_id = "remote-idle-bypass-peer";
  topology.generation = 1;
  topology.complete = true;
  for (const auto& rail : rails)
    topology.rails.push_back(PeerRailTopology{rail, true});
  transport.OnPeerTopology(topology);

  const BlockKey key = ToBlockKey(SelfHdr(), "remote-idle-bypass");
  const std::string value(64 * 1024, 'r');
  ASSERT_EQ(transport.Cache(node.addr, key, value.data(), value.size()),
            Status::kOk);
  ASSERT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 1u);
  const std::string warm = transport.MetricsText();
  ASSERT_EQ(
      DeviceCounterVal(warm, "dfkv_rdma_client_rail_selections_total",
                       rails[0]),
      1)
      << "the deterministic first admission must warm local rail zero";

  const auto seeded = RdmaTransportTestPeer::CoolRemoteRail(
      &transport, topology.peer_id, 0);
  ASSERT_TRUE(seeded);
  ASSERT_EQ(seeded->completion.transition, RemoteRailTransition::kCooled);
  EXPECT_EQ(RdmaTransportTestPeer::RemoteAllowed(
                transport, topology.peer_id, {1, 1}, seeded->now_us),
            (std::vector<uint8_t>{0, 1}));

  const std::string before = transport.MetricsText();
  std::string output;
  ASSERT_EQ(transport.Range(node.addr, key, 0, value.size(), &output, nullptr),
            Status::kOk);
  EXPECT_EQ(output, value);
  const std::string after = transport.MetricsText();
  EXPECT_EQ(
      DeviceCounterVal(after, "dfkv_rdma_client_rail_selections_total",
                       rails[0]) -
          DeviceCounterVal(before, "dfkv_rdma_client_rail_selections_total",
                           rails[0]),
      0)
      << "the idle QP on the cooled peer/rail must not bypass admission";
  EXPECT_EQ(
      DeviceCounterVal(after, "dfkv_rdma_client_rail_selections_total",
                       rails[1]) -
          DeviceCounterVal(before, "dfkv_rdma_client_rail_selections_total",
                           rails[1]),
      1);
  EXPECT_EQ(CounterVal(after, "dfkv_rdma_endpoint_cache_hits_total") -
                CounterVal(before, "dfkv_rdma_endpoint_cache_hits_total"),
            0);
  EXPECT_EQ(CounterVal(after, "dfkv_rdma_endpoint_cache_misses_total") -
                CounterVal(before, "dfkv_rdma_endpoint_cache_misses_total"),
            1);
  EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 2u)
      << "the cooled idle QP remains present only to prove it was bypassed";
}

TEST(RdmaLoopback, SoleRemoteCooledRailReturnsExplicitFailure) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv cooldown("DFKV_RDMA_REMOTE_RAIL_COOLDOWN_MS", "30000");
  ScopedEnv max_cooldown("DFKV_RDMA_REMOTE_RAIL_MAX_COOLDOWN_MS", "30000");
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);

  const auto discovered = rdma::RdmaTopology::Discover(
      {}, rdma::RdmaDiscoveryPolicy::kActiveOnly);
  if (discovered.status != rdma::RdmaDiscoveryStatus::kOk ||
      discovered.devices.empty())
    GTEST_SKIP() << "no ACTIVE RDMA device";
  const std::string rail = discovered.devices.front().name;
  RdmaNode node("remote-sole-cooled");
  RdmaTransport transport(kMaxMsg, rail);
  PeerTopology topology;
  topology.peer_addr = node.addr;
  topology.peer_id = "remote-sole-cooled-peer";
  topology.generation = 1;
  topology.complete = true;
  topology.rails.push_back(PeerRailTopology{rail, true});
  transport.OnPeerTopology(topology);

  const BlockKey key = ToBlockKey(SelfHdr(), "remote-sole-cooled");
  const std::string value(4096, 'c');
  ASSERT_EQ(transport.Cache(node.addr, key, value.data(), value.size()),
            Status::kOk);
  ASSERT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 1u);
  const auto seeded = RdmaTransportTestPeer::CoolRemoteRail(
      &transport, topology.peer_id, 0);
  ASSERT_TRUE(seeded);
  ASSERT_EQ(seeded->completion.transition, RemoteRailTransition::kCooled);

  const std::string before = transport.MetricsText();
  std::string output;
  EXPECT_EQ(transport.Range(node.addr, key, 0, value.size(), &output, nullptr),
            Status::kNoCompatibleRail);
  const std::string after = transport.MetricsText();
  EXPECT_TRUE(output.empty());
  EXPECT_EQ(CounterVal(after, "dfkv_rdma_client_no_compatible_rail_total") -
                CounterVal(before,
                           "dfkv_rdma_client_no_compatible_rail_total"),
            1);
  EXPECT_EQ(CounterVal(after, "dfkv_rdma_endpoint_cache_hits_total"),
            CounterVal(before, "dfkv_rdma_endpoint_cache_hits_total"));
  EXPECT_EQ(CounterVal(after, "dfkv_rdma_endpoint_cache_misses_total"),
            CounterVal(before, "dfkv_rdma_endpoint_cache_misses_total"));
  EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 1u);
}

TEST(RdmaLoopback, StableIdentityTopologyFlapRetainsRemoteCooldown) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv cooldown("DFKV_RDMA_REMOTE_RAIL_COOLDOWN_MS", "30000");
  ScopedEnv max_cooldown("DFKV_RDMA_REMOTE_RAIL_MAX_COOLDOWN_MS", "30000");
  const auto discovered = rdma::RdmaTopology::Discover(
      {}, rdma::RdmaDiscoveryPolicy::kActiveOnly);
  if (discovered.status != rdma::RdmaDiscoveryStatus::kOk ||
      discovered.devices.empty())
    GTEST_SKIP() << "no ACTIVE RDMA device";
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  const std::string rail = discovered.devices.front().name;
  RdmaTransport transport(kMaxMsg, rail);
  PeerTopology topology;
  topology.peer_addr = "unused-peer-address";
  topology.peer_id = "stable-flapping-peer";
  topology.generation = 100;
  topology.complete = true;
  topology.rails.push_back(PeerRailTopology{rail, true});
  transport.OnPeerTopology(topology);
  const uint64_t first_publication =
      RdmaTransportTestPeer::PeerPublication(transport, topology.peer_addr);

  const auto seeded = RdmaTransportTestPeer::CoolRemoteRail(
      &transport, topology.peer_id, 0);
  ASSERT_TRUE(seeded);
  topology.generation = 7;
  topology.rails[0].healthy = false;
  transport.OnPeerTopology(topology);
  topology.generation = 999;
  topology.rails[0].healthy = true;
  transport.OnPeerTopology(topology);
  EXPECT_GT(
      RdmaTransportTestPeer::PeerPublication(transport, topology.peer_addr),
      first_publication)
      << "health-only snapshots may republish without resetting rail health";
  EXPECT_EQ(RdmaTransportTestPeer::RemoteAllowed(
                transport, topology.peer_id, {1}, seeded->now_us),
            (std::vector<uint8_t>{0}))
      << "health-only topology generations must not reset stable identity";
}


TEST(RdmaLoopback,
     LateReleaseCannotRepoolAcrossRemoveAndIdenticalReadd) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  const auto discovered = rdma::RdmaTopology::Discover(
      {}, rdma::RdmaDiscoveryPolicy::kActiveOnly);
  if (discovered.status != rdma::RdmaDiscoveryStatus::kOk ||
      discovered.devices.empty())
    GTEST_SKIP() << "no ACTIVE RDMA device";

  const std::string rail = discovered.devices.front().name;
  RdmaNode node("peer-reincarnation");
  RdmaTransport transport(kMaxMsg, rail);
  PeerTopology topology;
  topology.peer_addr = node.addr;
  topology.peer_id = "reincarnated-peer";
  topology.generation = 55;
  topology.complete = true;
  topology.rails.push_back(PeerRailTopology{rail, true});
  transport.OnPeerTopology(topology);
  const uint64_t first_publication =
      RdmaTransportTestPeer::PeerPublication(transport, node.addr);

  auto* old_active =
      RdmaTransportTestPeer::AcquireData(&transport, node.addr);
  ASSERT_NE(old_active, nullptr);
  ASSERT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 0u);

  topology.present = false;
  transport.OnPeerTopology(topology);
  topology.present = true;
  transport.OnPeerTopology(topology);
  EXPECT_GT(RdmaTransportTestPeer::PeerPublication(transport, node.addr),
            first_publication);

  const std::string before_release = transport.MetricsText();
  RdmaTransportTestPeer::ReleaseData(&transport, node.addr, old_active);
  const std::string after_release = transport.MetricsText();
  EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 0u);
  EXPECT_EQ(
      CounterVal(after_release,
                 "dfkv_rdma_client_stale_generation_reaps_total") -
          CounterVal(before_release,
                     "dfkv_rdma_client_stale_generation_reaps_total"),
      1);

  const std::string before_reacquire = transport.MetricsText();
  auto* current =
      RdmaTransportTestPeer::AcquireData(&transport, node.addr);
  ASSERT_NE(current, nullptr);
  const std::string after_reacquire = transport.MetricsText();
  EXPECT_EQ(CounterVal(after_reacquire,
                       "dfkv_rdma_endpoint_cache_hits_total"),
            CounterVal(before_reacquire,
                       "dfkv_rdma_endpoint_cache_hits_total"));
  EXPECT_EQ(CounterVal(after_reacquire,
                       "dfkv_rdma_endpoint_cache_misses_total") -
                CounterVal(before_reacquire,
                           "dfkv_rdma_endpoint_cache_misses_total"),
            1);
  RdmaTransportTestPeer::ReleaseData(&transport, node.addr, current);
  EXPECT_EQ(RdmaTransportTestPeer::DataPoolSize(&transport, node.addr), 1u);
}

TEST(RdmaLoopback,
     ReplaySafeEndpointRetryExcludesRailButPostedPutNeverReplays) {
  if (!HaveRdma())
    GTEST_SKIP() << "no RDMA device (load two rdma_rxe devices for this test)";
  const auto rails = ConfiguredTwoTestRails();
  if (!HaveConfiguredActiveRails(rails))
    GTEST_SKIP() << "set DFKV_RDMA_DEV to exactly two ACTIVE test devices";
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  RdmaTransport transport(kMaxMsg, rails[0] + "," + rails[1]);

  const auto replay_safe = RdmaTransportTestPeer::PrepareEndpointRetry(
      &transport, /*from_pool=*/false, /*replay_safe=*/true,
      /*request_posted=*/true, /*attempted_rail=*/0);
  EXPECT_TRUE(replay_safe.retry);
  EXPECT_TRUE(replay_safe.cross_rail);
  EXPECT_EQ(replay_safe.excluded, (std::vector<uint8_t>{1, 0}))
      << "a fresh replay-safe retry must not revisit the failed endpoint rail";

  const auto posted_put = RdmaTransportTestPeer::PrepareEndpointRetry(
      &transport, /*from_pool=*/true, /*replay_safe=*/false,
      /*request_posted=*/true, /*attempted_rail=*/0);
  EXPECT_FALSE(posted_put.retry);
  EXPECT_FALSE(posted_put.cross_rail);
  EXPECT_EQ(posted_put.excluded, (std::vector<uint8_t>{0, 0}));
}
