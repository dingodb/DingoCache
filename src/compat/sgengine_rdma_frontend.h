/* Isolated SGEngine v1 RDMA compatibility listener.
 *
 * Built only with DFKV_WITH_RDMA. The listener owns a distinct bootstrap port
 * and maps every legacy wire key into KeyDomain::kSgEngineV1 before invoking
 * the shared storage service.
 */
#ifndef DFKV_SGENGINE_RDMA_FRONTEND_H_
#define DFKV_SGENGINE_RDMA_FRONTEND_H_

#ifdef DFKV_WITH_RDMA

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "common/status.h"

namespace dfkv {

class KvNodeServer;
class RdmaServer;

namespace compat {

class SgEngineRdmaFrontend {
 public:
  SgEngineRdmaFrontend(KvNodeServer& server, size_t max_message_bytes,
                       std::string device_names = "");
  ~SgEngineRdmaFrontend();

  SgEngineRdmaFrontend(const SgEngineRdmaFrontend&) = delete;
  SgEngineRdmaFrontend& operator=(const SgEngineRdmaFrontend&) = delete;

  Status Start(int port);
  void Stop();
  int port() const;
  size_t PipelineDepth() const;
  bool UseUringPath() const;
  uint64_t RejectedOps() const {
    return rejected_ops_.load(std::memory_order_relaxed);
  }
  std::string MetricsText() const;

 private:
  void Configure();

  KvNodeServer& server_;
  size_t max_message_bytes_;
  std::string device_names_;
  std::unique_ptr<RdmaServer> rdma_;
  std::atomic<uint64_t> rejected_ops_{0};
};

}  // namespace compat
}  // namespace dfkv

#endif  // DFKV_WITH_RDMA
#endif  // DFKV_SGENGINE_RDMA_FRONTEND_H_
