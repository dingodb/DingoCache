/* Per-key outcome of a batch call, reported by the C ABI's *_st calls.
 * Codes 0..7 are the Status values; the rest are decided by the client
 * without a server answer. Stable ABI: values are never renumbered. */
#ifndef DFKV_CLIENT_ITEM_STATUS_H_
#define DFKV_CLIENT_ITEM_STATUS_H_

#include <cstddef>
#include <cstdint>

#include "common/status.h"

namespace dfkv {

enum class ItemStatus : uint8_t {
  kOk = 0,
  kNotFound = 1,
  kCacheFull = 2,
  kQuotaExceeded = 3,
  kIOError = 4,
  kInvalid = 5,
  kResourceExhausted = 6,
  kNoCompatibleRail = 7,
  // The owner holds the key with another length than the read accepts:
  // shorter than an exact read asks, or larger than the destination. The
  // stored length is known, never guessed from a refusal.
  kLengthMismatch = 8,
  // The owner is in cooldown after failures; nothing was sent.
  kPeerCooling = 9,
  // No owner: the ring is empty, or its owner has no address.
  kNoRoute = 10,
  // The key, buffer or size is not valid; nothing was sent.
  kBadArgument = 11,
};

static_assert(static_cast<int>(Status::kOk) == 0 &&
                  static_cast<int>(Status::kNotFound) == 1 &&
                  static_cast<int>(Status::kCacheFull) == 2 &&
                  static_cast<int>(Status::kQuotaExceeded) == 3 &&
                  static_cast<int>(Status::kIOError) == 4 &&
                  static_cast<int>(Status::kInvalid) == 5 &&
                  static_cast<int>(Status::kResourceExhausted) == 6 &&
                  static_cast<int>(Status::kNoCompatibleRail) == 7,
              "ItemStatus 0..7 are the Status values");

inline ItemStatus ToItemStatus(Status status) {
  return static_cast<ItemStatus>(static_cast<uint8_t>(status));
}

// A read answered `status` with a value of `length` bytes, for a request of
// exactly `size` bytes (exact) or of at most `size`. A transport refuses a
// value larger than the destination with kInvalid and its stored length;
// every other refusal reports length 0 (Transport::RangeInto).
inline ItemStatus ReadItemStatus(Status status, uint64_t length, size_t size,
                                 bool exact) {
  if (status == Status::kInvalid && length > size)
    return ItemStatus::kLengthMismatch;
  if (status != Status::kOk) return ToItemStatus(status);
  return (exact ? length == size : length <= size) ? ItemStatus::kOk
                                                    : ItemStatus::kLengthMismatch;
}

}  // namespace dfkv

#endif  // DFKV_CLIENT_ITEM_STATUS_H_
