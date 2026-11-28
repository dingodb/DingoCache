/* Private RdmaTransport implementation types shared by implementation units.
 * Keep these out of the public transport header.
 */
#ifndef DFKV_RDMA_TRANSPORT_INTERNAL_H_
#define DFKV_RDMA_TRANSPORT_INTERNAL_H_

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "transport/rdma_transport.h"
#include "transport/rdma_connection_lifecycle.h"
#include "transport/rdma_protocol.h"
#include "transport/rdma_verbs.h"

namespace dfkv {

struct RdmaTransport::Conn {
  rdma::RcEndpoint ep;
  rdma::RecvSegmentInfo recv_segment;
  size_t rail_index = 0;
  uint64_t peer_publication = 0;
  std::string peer_id;
  // The identity remote rail health keys this peer by: peer_id, or for a
  // peer the MDS does not name, its address (RemoteRailHealth::AddressIdentity).
  std::string health_id;
  std::string node;  // assigned once at connection creation, never per acquire
  std::string metric_peer;
  uint64_t remote_lease_generation = 0;
  bool remote_lease_held = false;
  bool remote_recovery_probe = false;
  // This acquisition took the connection from the pool, at the given count
  // of successful completions. Until that count moves nothing says the peer
  // still has the other end.
  bool reused_from_pool = false;
  uint64_t completions_at_reuse = 0;
  // The operation stopped because this process could not register a caller
  // buffer. Neither the rail nor the peer had a part in that.
  bool local_resource_failure = false;
  rdma::RailLease lease;
  uint64_t lease_started_us = 0;
  uint64_t lease_bytes = 0;  // payload moved under the current lease
  bool credit_held = false;
  rdma::ResourceRequest budget_request;
  bool budget_held = false;
  size_t declared_bytes = 0;
  size_t depth = 1;
  Lane lane = Lane::kData;
  // Negotiated staged-lease PUT datapath (probe bit + bootstrap request bit).
  // Off for pooled conns predating the opt-in and for peers without support.
  bool leased_put = false;
  bool leased_put_supported = false;
  bool active_counted = false;
  bool live_counted = false;
  bool visited = true;  // guarded by RdmaTransport::mu_ while idle
  uint64_t last_application_us = 0;  // keepalives do not renew idle retention

  rdma::ConnectionLifecycle lifecycle;
  bool keepalive_probe = false;  // a keepalive, not an application op, is in flight
  // The bootstrap socket kept as session channel (DFKV_RDMA_SESSION_FD); -1
  // without one. The server closes its end when it ends the connection.
  int session_fd = -1;
  ~Conn() {
    if (session_fd >= 0) ::close(session_fd);
  }
  void Encode(char* out, WireOp op, const BlockKey& key, uint64_t offset,
              uint64_t length, uint64_t payload_len) const {
    EncodeReqVersion(out, kNativeProtoRdmaV2, op, key, offset, length,
                     payload_len);
  }
  bool Decode(const char* in, Status* status, uint64_t* data_len,
              uint64_t max_data = kMaxFrameLen,
              uint64_t* value_len = nullptr) const {
    return DecodeRespVersion(in, kNativeProtoRdmaV2, status, data_len, max_data,
                             value_len);
  }
  size_t data_capacity() const {
    return recv_segment.slot_size >= rdma::kV2DataOffset
               ? static_cast<size_t>(recv_segment.slot_size -
                                     rdma::kV2DataOffset)
               : 0;
  }
  uint64_t put_addr(size_t slot) const {
    return recv_segment.base_addr +
           slot * recv_segment.slot_size + rdma::kV2PutPrefixOffset;
  }
};

}  // namespace dfkv

#endif  // DFKV_RDMA_TRANSPORT_INTERNAL_H_
