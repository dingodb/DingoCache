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

#ifdef DFKV_WITH_RDMA
bool DeviceFilterConfigured() {
  const char* devices = std::getenv("DFKV_RDMA_DEV");
  return devices && *devices;
}
#endif
}  // namespace

std::unique_ptr<Transport> MakeClientTransport(std::string* reason) {
#ifdef DFKV_WITH_RDMA
  if (EnvTruthy("DFKV_RDMA")) {
    if (RdmaTransport::Available()) {  // native verbs (device-by-name, 400G)
      // Availability and construction both discover live HCAs. A link can drop
      // between those probes, so construction failure is a hard error.
      try {
        auto transport = std::make_unique<RdmaTransport>();
        if (reason) *reason = "rdma";
        return transport;
      } catch (const std::exception&) {
        // Report the same failure policy below.
      }
    }
    if (reason) {
      *reason = DeviceFilterConfigured()
                    ? "rdma-configured-devices-not-active"
                    : "rdma-requested-but-unavailable";
    }
    return nullptr;
  }
  if (reason) *reason = "tcp(rdma-not-requested)";
#else
  if (EnvTruthy("DFKV_RDMA")) {
    if (reason) *reason = "rdma-requested-but-not-built";
    return nullptr;
  }
  if (reason) *reason = "tcp(rdma-not-requested)";
#endif
  return std::make_unique<TcpTransport>();
}

}  // namespace dfkv
