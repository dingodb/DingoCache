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

  static size_t ConnectionBound(const RdmaTransport& transport,
                                size_t required) {
    return transport.ConnectionBound(required);
  }

  static std::vector<size_t> IdleDataBounds(
      RdmaTransport* transport, const std::string& node) {
    return transport->IdleDataBounds(node);
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
  // Keep Soft-RoCE CI below modest RLIMIT_MEMLOCK. Production commits a
  // 256-MiB initial chunk under a 16-GiB budget; fixtures need only 32 MiB.
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
  std::atomic<int> handler_delay_ms{0};

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
          const int delay = handler_delay_ms.load(std::memory_order_relaxed);
          if (delay > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(delay));
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

long MetricSum(const std::string& text, const std::string& prefix) {
  long total = 0;
  size_t begin = 0;
  while (begin < text.size()) {
    const size_t end = text.find('\n', begin);
    const size_t length =
        end == std::string::npos ? text.size() - begin : end - begin;
    if (text.compare(begin, prefix.size(), prefix) == 0) {
      const size_t space = text.rfind(' ', begin + length);
      if (space != std::string::npos && space >= begin) {
        try {
          total += std::stol(text.substr(space + 1, begin + length - space - 1));
        } catch (...) {
          return -1;
        }
      }
    }
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return total;
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

TEST(RdmaReadPrimitive, HostDestinationIsByteExact) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  constexpr size_t kBytes = 1024 * 1024;
  const char* dev = std::getenv("DFKV_RDMA_DEV");
  rdma::RcEndpoint source;
  rdma::RcEndpoint initiator;
  if (!source.Open(dev, 4096, 1, 1,
                   /*direct_io_buffers=*/true, kBytes))
    GTEST_SKIP() << "no usable RDMA device";
  if (!initiator.Open(dev, 4096, 1))
    GTEST_SKIP() << "no usable RDMA device";
  const rdma::QpInfo source_info = source.Local();
  const rdma::QpInfo initiator_info = initiator.Local();
  ASSERT_TRUE(source.Connect(initiator_info));
  ASSERT_TRUE(initiator.Connect(source_info));

  for (size_t i = 0; i < kBytes; ++i)
    source.dbuf(0)[i] = static_cast<char>((i * 17 + 31) & 0xff);
  std::vector<char> destination(kBytes, '\0');
  ibv_mr* destination_mr =
      initiator.RegisterTransient(destination.data(), destination.size());
  ASSERT_NE(destination_mr, nullptr);
  ASSERT_TRUE(initiator.PostRead(
      0, destination.data(), destination.size(), destination_mr,
      reinterpret_cast<uint64_t>(source.dbuf(0)), source.dmr(0)->rkey));
  ibv_wc completion{};
  ASSERT_EQ(initiator.WaitComp(&completion, 1, 10000), 1);
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_RDMA_READ);
  EXPECT_EQ(std::memcmp(destination.data(), source.dbuf(0), kBytes), 0);
  initiator.ReleaseTransient(destination_mr);
}
TEST(RdmaReadPrimitive, ConcurrentOneMiBThroughput) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  constexpr size_t kBytes = 1024 * 1024;
  constexpr size_t kThreads = 32;
  constexpr size_t kIterations = 512;
  const char* dev = std::getenv("DFKV_RDMA_DEV");
  struct Pair {
    std::unique_ptr<rdma::RcEndpoint> source;
    std::unique_ptr<rdma::RcEndpoint> initiator;
    std::vector<char> destination;
    ibv_mr* destination_mr = nullptr;
  };
  std::vector<std::unique_ptr<Pair>> pairs;
  pairs.reserve(kThreads);
  for (size_t thread = 0; thread < kThreads; ++thread) {
    auto pair = std::make_unique<Pair>();
    pair->source = std::make_unique<rdma::RcEndpoint>();
    pair->initiator = std::make_unique<rdma::RcEndpoint>();
    if (!pair->source->Open(dev, 4096, 1, 1,
                            /*direct_io_buffers=*/true, kBytes))
      GTEST_SKIP() << "no usable RDMA device";
    if (!pair->initiator->Open(dev, 4096, 1))
      GTEST_SKIP() << "no usable RDMA device";
    const rdma::QpInfo source_info = pair->source->Local();
    const rdma::QpInfo initiator_info = pair->initiator->Local();
    ASSERT_TRUE(pair->source->Connect(initiator_info));
    ASSERT_TRUE(pair->initiator->Connect(source_info));
    std::memset(pair->source->dbuf(0), static_cast<int>(thread + 1), kBytes);
    pair->destination.assign(kBytes, '\0');
    pair->destination_mr = pair->initiator->RegisterTransient(
        pair->destination.data(), pair->destination.size());
    ASSERT_NE(pair->destination_mr, nullptr);
    pairs.push_back(std::move(pair));
  }

  std::atomic<bool> start{false};
  std::atomic<size_t> failures{0};
  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  const auto begin = std::chrono::steady_clock::now();
  for (size_t thread = 0; thread < kThreads; ++thread) {
    workers.emplace_back([&, thread] {
      Pair& pair = *pairs[thread];
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      for (size_t i = 0; i < kIterations; ++i) {
        if (!pair.initiator->PostRead(
                0, pair.destination.data(), pair.destination.size(),
                pair.destination_mr,
                reinterpret_cast<uint64_t>(pair.source->dbuf(0)),
                pair.source->dmr(0)->rkey)) {
          failures.fetch_add(1, std::memory_order_relaxed);
          break;
        }
        ibv_wc completion{};
        if (pair.initiator->WaitComp(&completion, 1, 10000) != 1 ||
            completion.status != IBV_WC_SUCCESS ||
            completion.opcode != IBV_WC_RDMA_READ) {
          failures.fetch_add(1, std::memory_order_relaxed);
          break;
        }
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (auto& worker : workers) worker.join();
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
          .count();
  const double gbps =
      static_cast<double>(kThreads * kIterations * kBytes) / seconds / 1e9;
  std::fprintf(stderr,
               "RDMA_READ_BENCH bytes=%zu threads=%zu iterations=%zu "
               "seconds=%.6f goodput=%.3f_GBps\n",
               kBytes, kThreads, kIterations, seconds, gbps);
  EXPECT_EQ(failures.load(std::memory_order_relaxed), 0u);
  for (size_t thread = 0; thread < kThreads; ++thread) {
    EXPECT_EQ(std::memcmp(pairs[thread]->destination.data(),
                          pairs[thread]->source->dbuf(0), kBytes),
              0);
    pairs[thread]->initiator->ReleaseTransient(
        pairs[thread]->destination_mr);
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
  EXPECT_LE(expanded - before, 7);  // default max 8, one already warm

  auto again = c.BatchExist(probe);
  ASSERT_EQ(again.size(), probe.size());
  for (size_t i = 0; i < probe.size(); ++i)
    EXPECT_EQ((bool)again[i], (i % 2 == 0)) << probe[i];
  const long settled =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  EXPECT_LE(settled - before, 7);  // default pool cap 8, one already warm

  auto steady = c.BatchExist(probe);
  ASSERT_EQ(steady.size(), probe.size());
  for (size_t i = 0; i < probe.size(); ++i)
    EXPECT_EQ((bool)steady[i], (i % 2 == 0)) << probe[i];
  const long reused =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  EXPECT_EQ(reused, settled)
      << "settled BatchExist did not reuse its bounded connection pool";
}

TEST(RdmaLoopback, ConfiguredPoolMaxBoundsIdleControlConnections) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv pool_max("DFKV_RDMA_POOL_MAX", "4");
  RdmaNode node("poolmax4");
  RdmaTransport rt(kMaxMsg);
  KVClient c({{"n", node.addr}}, SelfHdr(), &rt);

  constexpr int kWorkers = 7;
  for (int i = 0; i < kWorkers; ++i) {
    const std::string value = "v" + std::to_string(i);
    ASSERT_TRUE(c.Put("pm" + std::to_string(i), value.data(), value.size()));
  }
  node.handler_delay_ms.store(250, std::memory_order_relaxed);
  auto wave = [&] {
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::vector<char> found(kWorkers, 0);
    std::vector<std::thread> workers;
    for (int i = 0; i < kWorkers; ++i) {
      workers.emplace_back([&, i] {
        ready.fetch_add(1, std::memory_order_relaxed);
        while (!start.load(std::memory_order_acquire))
          std::this_thread::yield();
        found[i] = c.Exist("pm" + std::to_string(i));
      });
    }
    while (ready.load(std::memory_order_relaxed) != kWorkers)
      std::this_thread::yield();
    start.store(true, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(
        MetricSum(
            rt.MetricsText(),
            "dfkv_rdma_client_pool_connections{lane=\"control\",state=\"active\""),
        kWorkers);
    for (auto& worker : workers) worker.join();
    for (const char present : found) EXPECT_TRUE(present);
  };
  wave();
  const long opened_after_first =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  wave();
  const long opened_after_second =
      CounterVal(rt.MetricsText(), "dfkv_rdma_client_conns_opened_total");
  EXPECT_GE(opened_after_second - opened_after_first, 3)
      << "pool max four should reopen the three endpoints it cannot retain";

  const std::string metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_pool_limit"), 4);
  EXPECT_LE(
      MetricSum(
          metrics,
          "dfkv_rdma_client_pool_connections{lane=\"control\",state=\"idle\""),
      4);
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
  EXPECT_GT(CounterVal(srv_text, "dfkv_rdma_recv_segment_bytes"), 0);
  EXPECT_GT(CounterVal(srv_text, "dfkv_rdma_recv_segment_used_bytes"), 0);
  EXPECT_GT(CounterVal(
                srv_text,
                "dfkv_rdma_recv_segment_largest_free_range_bytes"),
            0);
  EXPECT_GE(CounterVal(srv_text, "dfkv_rdma_pull_connections"), 1);
  EXPECT_GT(CounterVal(
                srv_text,
                "dfkv_rdma_connection_bytes{class=\"data\"}"),
            0);
  EXPECT_NE(srv_text.find("dfkv_rdma_rail_active_conns{dev=\""),
            std::string::npos) << srv_text;
  EXPECT_GE(CounterVal(srv_text, "dfkv_rdma_rail_completions_total"), 2);
  EXPECT_GE(CounterVal(srv_text, "dfkv_rdma_rail_put_writes_total"), 1);
  EXPECT_GE(CounterVal(srv_text, "dfkv_rdma_rail_put_bytes_total"), 2048);

  // client transport: a connection was opened and the MR region declared
  std::string cli_text = rt.MetricsText();
  EXPECT_NE(cli_text.find("dfkv_rdma_client_conns_opened_total"), std::string::npos) << cli_text;
  EXPECT_NE(cli_text.find("dfkv_rdma_client_rail_conns_total{dev="), std::string::npos) << cli_text;
  EXPECT_NE(cli_text.find("dfkv_rdma_client_mr_regions 1"), std::string::npos) << cli_text;
  EXPECT_GE(CounterVal(cli_text, "dfkv_rdma_client_v2_put_writes_total"), 1);
  EXPECT_GE(CounterVal(cli_text, "dfkv_rdma_client_pull_reads_total"), 1);
  EXPECT_GE(CounterVal(cli_text,
                       "dfkv_rdma_client_pull_read_bytes_total"), 2048);
  EXPECT_GE(CounterVal(cli_text,
                       "dfkv_rdma_client_pool_mr_registrations_total"), 1);
  EXPECT_EQ(CounterVal(cli_text, "dfkv_rdma_client_pool_limit"), 8);
  EXPECT_GE(MetricSum(
                cli_text,
                "dfkv_rdma_client_pool_connections{lane=\"payload\",state=\"idle\""),
            1);
  EXPECT_GE(MetricSum(cli_text, "dfkv_rdma_client_peer_connections{"), 1);

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

TEST(RdmaLoopback, SegmentOwnershipMetricsReturnToZeroAfterClientTeardown) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv idle_reclaim("DFKV_RDMA_IDLE_MS", "200");
  RdmaNode node("segment-metric-teardown");
  {
    RdmaTransport transport(kMaxMsg);
    KVClient client({{"n", node.addr}}, SelfHdr(), &transport);
    const std::string value(4096, 's');
    ASSERT_TRUE(client.Put("segment-owned", value.data(), value.size()));
    const std::string active = node.rsrv->MetricsText();
    EXPECT_GT(CounterVal(active, "dfkv_rdma_recv_segment_used_bytes"), 0);
    EXPECT_GT(CounterVal(active, "dfkv_rdma_pull_connections"), 0);
    EXPECT_GT(CounterVal(
                  active,
                  "dfkv_rdma_connection_bytes{class=\"data\"}"),
              0);
  }
  for (int i = 0; i < 1000 && node.rsrv->ActiveConns() != 0; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  ASSERT_EQ(node.rsrv->ActiveConns(), 0u);
  const std::string released = node.rsrv->MetricsText();
  EXPECT_EQ(CounterVal(released, "dfkv_rdma_recv_segment_used_bytes"), 0);
  EXPECT_EQ(CounterVal(released, "dfkv_rdma_pull_connections"), 0);
  EXPECT_EQ(CounterVal(
                released,
                "dfkv_rdma_connection_bytes{class=\"data\"}"),
            0);
  EXPECT_EQ(CounterVal(
                released,
                "dfkv_rdma_connection_bytes{class=\"control\"}"),
            0);
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

TEST(RdmaLoopback, ScatterGatherGetReusesUnifiedDataPool) {
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
  const std::string metrics = rt.MetricsText();
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_client_conns_opened_total"),
            data_opened);
  ASSERT_FALSE(node.rsrv->DeviceNames().empty());
  const std::string idle_sg =
      "dfkv_rdma_client_pool_connections{lane=\"sg\",state=\"idle\",dev=\"" +
      node.rsrv->DeviceNames().front() + "\"}";
  EXPECT_EQ(CounterVal(metrics, idle_sg), 1);

  ::unsetenv("DFKV_RDMA_DEPTH");
}

TEST(RdmaLoopback, ScatterGatherPutReturnsToUnifiedDataPool) {
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
            sg_opened);

  std::string actual(ordinary.size(), '\0');
  ASSERT_TRUE(c.Get("ordinary_after_sg", actual.data(), actual.size()));
  EXPECT_EQ(actual, ordinary);
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
  ScopedEnv idle_reclaim("DFKV_RDMA_IDLE_MS", "120");
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

TEST(RdmaLoopback, DataConnectionsUseActualBlockSizeClasses) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv max_block("DFKV_RDMA_MAX_BLOCK_BYTES", "262144");
  ScopedEnv min_block("DFKV_RDMA_CONNECTION_MIN_BLOCK_BYTES", "4096");
  RdmaNode node("adaptive-classes");
  RdmaTransport transport(kMaxMsg);
  KVClient client({{"n", node.addr}}, SelfHdr(), &transport);

  std::string small(4000, 's');
  ASSERT_TRUE(client.Put("small", small.data(), small.size()));
  EXPECT_EQ(RdmaTransportTestPeer::IdleDataBounds(&transport, node.addr),
            (std::vector<size_t>{4096}));

  std::string large(60 * 1024, 'l');
  ASSERT_TRUE(client.Put("large", large.data(), large.size()));
  EXPECT_EQ(RdmaTransportTestPeer::IdleDataBounds(&transport, node.addr),
            (std::vector<size_t>{4096, 65536}));

  std::string got(small.size(), '\0');
  ASSERT_TRUE(client.Get("small", got.data(), got.size()));
  EXPECT_EQ(got, small);
  EXPECT_EQ(RdmaTransportTestPeer::IdleDataBounds(&transport, node.addr),
            (std::vector<size_t>{4096, 65536}));
  EXPECT_EQ(RdmaTransportTestPeer::ConnectionBound(transport, 1), 4096u);
  EXPECT_EQ(RdmaTransportTestPeer::ConnectionBound(transport, 5000), 8192u);
  EXPECT_EQ(RdmaTransportTestPeer::ConnectionBound(transport, 60 * 1024),
            65536u);
}

TEST(RdmaLoopback, ServerReceivePoolCommitsChunksOnDemand) {
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";
  ScopedEnv recv_max("DFKV_RDMA_RECV_SEGMENT_SIZE", "4194304");
  ScopedEnv recv_chunk("DFKV_RDMA_RECV_CHUNK_BYTES", "1048576");
  ScopedEnv max_block("DFKV_RDMA_MAX_BLOCK_BYTES", "65536");
  ScopedEnv min_block("DFKV_RDMA_CONNECTION_MIN_BLOCK_BYTES", "4096");
  RdmaNode node("lazy-recv-pool");

  std::vector<std::unique_ptr<RdmaTransport>> transports;
  std::vector<std::unique_ptr<KVClient>> clients;
  std::string value(60 * 1024, 'p');
  for (int i = 0; i < 4; ++i) {
    auto transport = std::make_unique<RdmaTransport>(kMaxMsg);
    auto client = std::make_unique<KVClient>(
        std::vector<std::pair<std::string, std::string>>{
            {"n", node.addr}},
        SelfHdr(), transport.get());
    ASSERT_TRUE(client->Put("pool-" + std::to_string(i),
                            value.data(), value.size()));
    transports.push_back(std::move(transport));
    clients.push_back(std::move(client));
  }

  const std::string metrics = node.rsrv->MetricsText();
  const long chunks =
      CounterVal(metrics, "dfkv_rdma_recv_segment_chunks");
  EXPECT_GE(chunks, 3);
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_recv_segment_max_bytes"),
            4 * 1024 * 1024);
  EXPECT_EQ(CounterVal(metrics, "dfkv_rdma_recv_segment_bytes"),
            chunks * 1024 * 1024);
  EXPECT_EQ(CounterVal(metrics,
                       "dfkv_rdma_recv_segment_growths_total"),
            chunks - 1);
  EXPECT_EQ(CounterVal(
                metrics,
                "dfkv_rdma_recv_segment_growth_failures_total"),
            0);
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
     PullReadFailureClosesConnectionAndNextGetRecoversWithoutAbort) {
  ScopedEnv configured_device("DFKV_RDMA_DEV", nullptr);
  ScopedEnv rail_tiers("DFKV_RDMA_RAIL_TIERS", nullptr);
  ScopedEnv keepalive("DFKV_RDMA_KEEPALIVE_MS", "0");
  if (!HaveRdma()) GTEST_SKIP() << "no RDMA device";

  RdmaNode node("pull-read-failure");
  RdmaTransport transport(kMaxMsg);
  const BlockKey key = ToBlockKey(SelfHdr(), "pull-read-failure");
  const std::string value = PatternValue(64 * 1024 + 13, 57);
  ASSERT_EQ(transport.Cache(node.addr, key, value.data(), value.size()),
            Status::kOk);

  std::string output(value.size(), '\x5a');
  std::vector<uint64_t> value_lens;
  {
    ScopedEnv read_fault("DFKV_RDMA_TEST_PULL_READ_FAILURE", "1");
    const auto statuses = transport.RangeInto(
        node.addr, {key}, {{output.data(), output.size()}}, &value_lens);
    EXPECT_EQ(statuses, std::vector<Status>({Status::kIOError}));
    EXPECT_EQ(output, std::string(output.size(), '\x5a'));
  }

  const auto recovered = transport.RangeInto(
      node.addr, {key}, {{output.data(), output.size()}}, &value_lens);
  EXPECT_EQ(recovered, std::vector<Status>({Status::kOk}));
  EXPECT_EQ(value_lens, std::vector<uint64_t>({value.size()}));
  EXPECT_EQ(output, value);
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
  const char* configured = std::getenv("DFKV_RDMA_DEV");
  const std::string rail =
      configured && *configured && std::strchr(configured, ',') == nullptr
          ? configured
          : discovered.devices.front().name;
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

  const char* configured = std::getenv("DFKV_RDMA_DEV");
  const std::string rail =
      configured && *configured && std::strchr(configured, ',') == nullptr
          ? configured
          : discovered.devices.front().name;
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
