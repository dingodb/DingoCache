/* Stable names for RDMA work-completion statuses. The client and server
 * failure log lines and the per-status error counters share this one table, so
 * a status reads the same in a log line and in a metric label.
 *
 * Deliberately verbs-free: statuses are the plain integers of enum
 * ibv_wc_status, whose values are fixed by the verbs ABI. That keeps the table
 * unit-testable in a build without libibverbs; rdma_verbs.cc pins the
 * numbering against the real enum. */
#ifndef DFKV_RDMA_WC_STATUS_H_
#define DFKV_RDMA_WC_STATUS_H_

#include <cstddef>
#include <cstdint>
#include <string>

namespace dfkv {
namespace rdma {

// One counter slot per known status plus a trailing "other" slot. A status
// added by a newer rdma-core lands in "other", so the label set cannot grow.
constexpr size_t kWcStatusKnown = 24;
constexpr size_t kWcStatusSlots = kWcStatusKnown + 1;

inline size_t WcStatusSlot(int status) {
  return status >= 0 && static_cast<size_t>(status) < kWcStatusKnown
             ? static_cast<size_t>(status)
             : kWcStatusKnown;
}

// The enumerator name without its IBV_WC_ prefix, lowercased; "other" for a
// value outside the table.
inline const char* WcStatusName(int status) {
  static constexpr const char* kNames[kWcStatusSlots] = {
      "success",            // 0
      "loc_len_err",        // 1
      "loc_qp_op_err",      // 2
      "loc_eec_op_err",     // 3
      "loc_prot_err",       // 4
      "wr_flush_err",       // 5
      "mw_bind_err",        // 6
      "bad_resp_err",       // 7
      "loc_access_err",     // 8
      "rem_inv_req_err",    // 9
      "rem_access_err",     // 10
      "rem_op_err",         // 11
      "retry_exc_err",      // 12
      "rnr_retry_exc_err",  // 13
      "loc_rdd_viol_err",   // 14
      "rem_inv_rd_req_err", // 15
      "rem_abort_err",      // 16
      "inv_eecn_err",       // 17
      "inv_eec_state_err",  // 18
      "fatal_err",          // 19
      "resp_timeout_err",   // 20
      "general_err",        // 21
      "tm_err",             // 22
      "tm_rndv_incomplete", // 23
      "other",
  };
  return kNames[WcStatusSlot(status)];
}

// The status part of a failure log line: "status=12 status_name=retry_exc_err
// vendor=129". The number is what operators search for; the name is the
// metric label; the vendor syndrome is provider-specific and kept raw.
inline std::string WcStatusFields(int status, unsigned vendor_err) {
  return "status=" + std::to_string(status) +
         " status_name=" + WcStatusName(status) +
         " vendor=" + std::to_string(vendor_err);
}

// What kind of failure a completion reports, as one number for a log-throttle
// key: the status together with the vendor syndrome, so a different syndrome
// under the same status counts as something new, not as a repeat.
inline uint64_t WcFailureKind(int status, unsigned vendor_err) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(status)) << 32) |
         static_cast<uint32_t>(vendor_err);
}

}  // namespace rdma
}  // namespace dfkv

#endif  // DFKV_RDMA_WC_STATUS_H_
