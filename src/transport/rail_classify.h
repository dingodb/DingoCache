/* Datapath failure attribution for RDMA multi-rail health accounting: maps a
 * failed completion window's evidence onto RailCompletion so a dead or slow
 * PEER never drags a healthy LOCAL rail into quarantine. Verbs-dependent
 * (ibv_wc_status); only RDMA-gated compilation units may include this. */
#ifndef DFKV_RAIL_CLASSIFY_H_
#define DFKV_RAIL_CLASSIFY_H_

#include <infiniband/verbs.h>

#include "transport/rail_select.h"

namespace dfkv {
namespace rdma {

// Attribute a failed datapath completion window to the local rail or the
// remote endpoint. `status` is the first non-SUCCESS status reaped while
// draining the window (IBV_WC_SUCCESS when the window produced no error WC),
// and `had_completions` reports whether ANY completion events arrived before
// the window died.
//
// kEndpointFailure (peer/fabric evidence; credits returned, rail health
// untouched):
//   - REM_ACCESS/REM_OP/REM_INV_REQ/REM_ABORT: the peer's QP, MR or servicing
//     logic rejected the request;
//   - RETRY_EXC/RNR_RETRY_EXC/RESP_TIMEOUT: the peer vanished or stalled
//     through the full retry chain;
//   - WR_FLUSH_ERR: outstanding WRs flushed while the QP entered the error
//     state — teardown fallout of a peer-triggered QP error, not a local
//     hardware signal;
//   - deadline expiry with no error WC at all.
// kRailFailure (local evidence; counts toward quarantine):
//   - the LOC_* family (local length/QP-access/EEC/protection/RDD faults),
//     FATAL_ERR and GENERAL_ERR (device-level local faults);
//   - post/CQ/verbs API failures never produce a WC; callers map them to a
//     local status before classifying (see rdma_transport.cc).
//
// Everything else (MW_BIND/BAD_RESP/INV_EEC*/statuses added after this
// table, out-of-range values) defaults to kEndpointFailure: a wrongly spared
// rail merely keeps one connection's credits healthy, while a wrongly blamed
// rail quarantines a healthy HCA and stalls every operation for the cooldown.
// When evidence is ambiguous, spare the local rail.
inline RailCompletion ClassifyCompletion(ibv_wc_status status,
                                         bool had_completions) {
  if (status == IBV_WC_SUCCESS) {
    // No error WC: the window died by its deadline, not by a verbs fault.
    if (had_completions) {
      // Partial progress then silence: completions prove the rail was healthy
      // when last observed, so the stall afterwards is the peer going quiet
      // mid-window.
      return RailCompletion::kEndpointFailure;
    }
    // Zero completions: the peer never answered and the local NIC surfaced no
    // error of its own.
    return RailCompletion::kEndpointFailure;
  }
  switch (status) {
    // Peer-sourced evidence (see header comment).
    case IBV_WC_REM_ACCESS_ERR:
    case IBV_WC_REM_OP_ERR:
    case IBV_WC_REM_INV_REQ_ERR:
    case IBV_WC_REM_ABORT_ERR:
    case IBV_WC_RETRY_EXC_ERR:
    case IBV_WC_RNR_RETRY_EXC_ERR:
    case IBV_WC_RESP_TIMEOUT_ERR:
    case IBV_WC_WR_FLUSH_ERR:
      return RailCompletion::kEndpointFailure;
    // Local-sourced evidence (see header comment).
    case IBV_WC_LOC_LEN_ERR:
    case IBV_WC_LOC_QP_OP_ERR:
    case IBV_WC_LOC_EEC_OP_ERR:
    case IBV_WC_LOC_PROT_ERR:
    case IBV_WC_LOC_ACCESS_ERR:
    case IBV_WC_LOC_RDD_VIOL_ERR:
    case IBV_WC_FATAL_ERR:
    case IBV_WC_GENERAL_ERR:
      return RailCompletion::kRailFailure;
    default:
      // Unknown or ambiguous: spare the local rail (see header comment).
      return RailCompletion::kEndpointFailure;
  }
}

}  // namespace rdma
}  // namespace dfkv

#endif  // DFKV_RAIL_CLASSIFY_H_
