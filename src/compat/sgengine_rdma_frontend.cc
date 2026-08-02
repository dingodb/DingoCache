#include "compat/sgengine_rdma_frontend.h"

#ifdef DFKV_WITH_RDMA

#include <utility>

#include "cache/kv_node_server.h"
#include "cache/rdma_server.h"
#include "transport/wire.h"

namespace dfkv::compat {

SgEngineRdmaFrontend::SgEngineRdmaFrontend(KvNodeServer& server,
                                           size_t max_message_bytes,
                                           std::string device_names)
    : server_(server),
      max_message_bytes_(max_message_bytes),
      device_names_(std::move(device_names)) {}

SgEngineRdmaFrontend::~SgEngineRdmaFrontend() { Stop(); }

void SgEngineRdmaFrontend::Configure() {
  rdma_ = std::make_unique<RdmaServer>(
      [this](uint8_t op_raw, const BlockKey& key, uint64_t offset,
             uint64_t length, const char* payload, uint64_t payload_len,
             std::string* out) {
        const WireOp op = static_cast<WireOp>(op_raw);
        switch (op) {
          case WireOp::kCache:
          case WireOp::kRange:
          case WireOp::kExist:
          case WireOp::kRemove:
          case WireOp::kStats:
            return server_.ProcessRequestForKey(
                op_raw, key, offset, length, payload, payload_len, out);
          default:
            // Never leak native membership addresses across the domain boundary.
            rejected_ops_.fetch_add(1, std::memory_order_relaxed);
            return Status::kInvalid;
        }
      },
      max_message_bytes_, device_names_,
      RdmaServer::ProtocolMode::kSgEngineV1);

  rdma_->set_range_handler(
      [this](const BlockKey& key, uint64_t offset, uint64_t length,
             char* buffer, size_t capacity, const char** out_data,
             size_t* out_len) {
        return server_.RangeDirectForKey(key, offset, length, buffer, capacity,
                                         out_data, out_len);
      });
  rdma_->set_cache_direct_handler(
      [this](const BlockKey& key, char* data, size_t length, size_t capacity) {
        return server_.CacheDirectForKey(key, data, length, capacity);
      });
  rdma_->set_range_prep_handler(
      [this](const BlockKey& key, uint64_t offset, uint64_t length,
             size_t capacity, RdmaServer::RangePrepResult* out) {
        KVStore::RangePrep prep;
        const Status status = server_.RangeDirectPrepForKey(
            key, offset, length, capacity, &prep, &out->flight);
        if (status == Status::kOk) {
          out->fd = prep.fd;
          out->aligned_off = prep.aligned_off;
          out->aligned_len = prep.aligned_len;
          out->head = prep.head;
          out->payload_len = prep.payload_len;
          out->release_token = prep.token;
        }
        return status;
      });
  rdma_->set_range_release_handler(
      [this](uint64_t token) { server_.RangePrepRelease(token); });
  rdma_->set_range_complete_handler(
      [this](bool ok, size_t bytes_read, double elapsed_seconds,
             uint64_t flight, const char* data) {
        server_.RangeDirectComplete(ok, bytes_read, elapsed_seconds, flight,
                                    data);
      });
  rdma_->set_range_flight_abort_handler(
      [this](uint64_t flight) { server_.RangeFlightAbort(flight); });

  if (server_.ram_enabled()) {
    rdma_->RegisterMemory(server_.ram_arena(), server_.ram_arena_bytes());
    rdma_->set_ram_range_handler(
        [this](const BlockKey& key, uint64_t offset, uint64_t length,
               const char** out_ptr, size_t* out_len, uint64_t* out_token) {
          return server_.RamRangePrepForKey(key, offset, length, out_ptr,
                                            out_len, out_token);
        });
    rdma_->set_ram_release_handler(
        [this](uint64_t token) { server_.RamRelease(token); });
  }
}

Status SgEngineRdmaFrontend::Start(int port) {
  if (rdma_ || port < 0 || port > 65535) return Status::kInvalid;
  Configure();
  const Status status = rdma_->Start(port);
  if (status != Status::kOk) rdma_.reset();
  return status;
}

void SgEngineRdmaFrontend::Stop() {
  if (!rdma_) return;
  rdma_->Stop();
  rdma_.reset();
}

int SgEngineRdmaFrontend::port() const { return rdma_ ? rdma_->port() : 0; }

size_t SgEngineRdmaFrontend::PipelineDepth() const {
  return rdma_ ? rdma_->PipelineDepth() : 0;
}

bool SgEngineRdmaFrontend::UseUringPath() const {
  return rdma_ && rdma_->UseUringPath();
}

std::string SgEngineRdmaFrontend::MetricsText() const {
  if (!rdma_) return {};
  std::string metrics = rdma_->MetricsText();
  auto replace_all = [&metrics](const std::string& from,
                                const std::string& to) {
    for (size_t pos = 0; (pos = metrics.find(from, pos)) != std::string::npos;
         pos += to.size())
      metrics.replace(pos, from.size(), to);
  };
  replace_all("dfkv_rdma_", "dfkv_sgengine_rdma_");
  replace_all("dfkv_uring_", "dfkv_sgengine_uring_");

  std::string out;
  out += "# HELP dfkv_sgengine_rdma_info SGEngine compatibility RDMA contract\n";
  out += "# TYPE dfkv_sgengine_rdma_info gauge\n";
  out +=
      "dfkv_sgengine_rdma_info{wire=\"v1\",key_domain=\"sgengine-v1\"} 1\n";
  out += metrics;
  out += "# HELP dfkv_sgengine_rdma_rejected_ops_total Discovery or control "
         "operations rejected at the compatibility boundary\n";
  out += "# TYPE dfkv_sgengine_rdma_rejected_ops_total counter\n";
  out += "dfkv_sgengine_rdma_rejected_ops_total " +
         std::to_string(RejectedOps()) + "\n";
  return out;
}

}  // namespace dfkv::compat

#endif  // DFKV_WITH_RDMA
