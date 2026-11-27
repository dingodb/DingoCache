#include "transport/rdma_transport.h"

#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace dfkv {

Status RdmaTransport::Cache(const std::string& node, const BlockKey& key,
                            const void* data, size_t len) {
  if (inline_put_max_bytes_ != 0 && len > inline_put_max_bytes_)
    return CacheMany(node, {{key, data, len}})[0];
  if (NoteBlock(len)) return Status::kInvalid;
  return RoundTrip(node, WireOp::kCache, key, 0, 0, data, len, nullptr);
}
Status RdmaTransport::Range(const std::string& node, const BlockKey& key,
                            uint64_t offset, uint64_t length,
                            std::string* out, uint64_t* value_len) {
  if (length > std::numeric_limits<size_t>::max() ||
      NoteBlock(static_cast<size_t>(length)))
    return Status::kInvalid;
  if (out == nullptr) return Status::kInvalid;
  out->clear();
  std::string staged(static_cast<size_t>(length), '\0');
  const RangeDstSegment segment{staged.data(), staged.size()};
  uint64_t received = 0;
  const Status status = PullInto(
      node, key, &segment, 1, staged.size(), Lane::kData, value_len,
      nullptr, PullMode::kSlice, offset, length, &received);
  if (status == Status::kOk) {
    staged.resize(static_cast<size_t>(received));
    *out = std::move(staged);
  }
  return status;
}

Status RdmaTransport::Lookup(const std::string& node, const BlockKey& key,
                             uint64_t* value_len) {
  if (value_len == nullptr) return Status::kInvalid;
  *value_len = 0;
  return RoundTrip(node, WireOp::kLookup, key, 0, 0, nullptr, 0, nullptr,
                   value_len);
}
Status RdmaTransport::Exist(const std::string& node, const BlockKey& key,
                            bool* exist) {
  if (exist == nullptr) return Status::kInvalid;
  Status st =
      RoundTrip(node, WireOp::kExist, key, 0, 0, nullptr, 0, nullptr);
  if (st == Status::kOk) {
    *exist = true;
    return Status::kOk;
  }
  if (st == Status::kNotFound) {
    *exist = false;
    return Status::kOk;
  }
  return st;
}

Status RdmaTransport::Retain(const std::string& node, uint64_t tenant_hash,
                             uint64_t count, const std::string& payload,
                             std::string* out) {
  if (out == nullptr || count == 0 || count > kRetainMaxKeys)
    return Status::kInvalid;
  return RoundTrip(node, WireOp::kRetain, BlockKey{0, 0, tenant_hash}, 0,
                   count, payload.data(), payload.size(), out);
}

Status RdmaTransport::RemovedWithin(const std::string& node,
                                    uint64_t tenant_hash, uint64_t count,
                                    const std::string& payload,
                                    std::string* out) {
  if (out == nullptr || count == 0 || count > kRemovedWithinMaxKeys)
    return Status::kInvalid;
  return RoundTrip(node, WireOp::kRemovedWithin, BlockKey{0, 0, tenant_hash},
                   0, count, payload.data(), payload.size(), out);
}

Status RdmaTransport::RemovedWithinAges(const std::string& node,
                                        uint64_t tenant_hash, uint64_t count,
                                        const std::string& payload,
                                        std::string* out) {
  if (out == nullptr || count == 0 || count > kRemovedWithinMaxKeys)
    return Status::kInvalid;
  return RoundTrip(node, WireOp::kRemovedWithinAges,
                   BlockKey{0, 0, tenant_hash}, 0, count, payload.data(),
                   payload.size(), out);
}

Status RdmaTransport::Remove(const std::string& node, const BlockKey& key) {
  return RoundTrip(node, WireOp::kRemove, key, 0, 0, nullptr, 0, nullptr);
}

Status RdmaTransport::Members(const std::string& node, std::string* out) {
  return RoundTrip(node, WireOp::kMembers, BlockKey{}, 0, 0, nullptr, 0, out);
}

}  // namespace dfkv
