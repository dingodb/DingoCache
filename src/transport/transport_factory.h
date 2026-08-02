/* Client transport selection: RDMA when built with DFKV_WITH_RDMA, requested
 * (DFKV_RDMA=1), and an RDMA device is usable; TCP when RDMA is not requested.
 * A requested but unavailable RDMA transport fails closed. */
#ifndef DFKV_TRANSPORT_FACTORY_H_
#define DFKV_TRANSPORT_FACTORY_H_

#include <memory>

#include "transport/transport.h"

namespace dfkv {

// reason (optional) receives "rdma", "tcp(rdma-not-requested)", or an
// "rdma-requested-..." failure description.
std::unique_ptr<Transport> MakeClientTransport(std::string* reason = nullptr);

}  // namespace dfkv

#endif  // DFKV_TRANSPORT_FACTORY_H_
