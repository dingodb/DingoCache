#include "transport/transport_factory.h"

#include <cstdlib>
#include <exception>
#include <cstring>
#include <string>

#include "transport/tcp_transport.h"

#ifdef DFKV_WITH_RDMA
#include "transport/rdma_transport.h"  // native verbs: device-by-name, 400G-capable
#endif

namespace dfkv {

namespace {
bool EnvTruthy(const char* name) {
  const char* v = std::getenv(name);
  return v && *v && std::strcmp(v, "0") != 0 &&
         std::strcmp(v, "false") != 0 && std::strcmp(v, "no") != 0;
}

bool RequireRdma() { return EnvTruthy("DFKV_REQUIRE_RDMA"); }
bool DeviceFilterConfigured() {
  const char* devices = std::getenv("DFKV_RDMA_DEV");
  return devices && *devices;
}
}  // namespace

std::unique_ptr<Transport> MakeClientTransport(std::string* reason) {
#ifdef DFKV_WITH_RDMA
  if (EnvTruthy("DFKV_RDMA")) {
    if (RdmaTransport::Available()) {  // native verbs (device-by-name, 400G)
      // Availability and construction both discover live HCAs. A link can drop
      // between those probes, so construction failure must follow the same
      // fail-closed/fallback policy instead of escaping through dfkv_open.
      try {
        auto transport = std::make_unique<RdmaTransport>();
        if (reason) *reason = "rdma";
        return transport;
      } catch (const std::exception&) {
        // Fall through to the common policy below.
      }
    }
    if (RequireRdma() || DeviceFilterConfigured()) {
      if (reason) {
        *reason = DeviceFilterConfigured()
                      ? "rdma-configured-devices-not-active"
                      : "rdma-required-but-no-active-device";
      }
      return nullptr;
    }
    if (reason) *reason = "tcp(rdma-requested-but-no-device)";
    return std::make_unique<TcpTransport>();
  }
  if (RequireRdma()) {
    if (reason) *reason = "rdma-required-but-DFKV_RDMA-not-set";
    return nullptr;
  }
  if (reason) *reason = "tcp(rdma-not-requested)";
#else
  if (EnvTruthy("DFKV_RDMA") && DeviceFilterConfigured()) {
    if (reason) *reason = "rdma-configured-devices-but-rdma-not-built";
    return nullptr;
  }
  if (RequireRdma()) {
    if (reason) {
      *reason = EnvTruthy("DFKV_RDMA") ? "rdma-required-but-not-built"
                                       : "rdma-required-but-DFKV_RDMA-not-set";
    }
    return nullptr;
  }
  if (reason) *reason = EnvTruthy("DFKV_RDMA") ? "tcp(rdma-not-built)" : "tcp";
#endif
  return std::make_unique<TcpTransport>();
}

}  // namespace dfkv
