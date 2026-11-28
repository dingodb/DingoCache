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

std::vector<Status> RdmaTransport::RangeInto(
    const std::string& node, const std::vector<BlockKey>& keys,
    const std::vector<RangeDst>& destinations,
    std::vector<uint64_t>* value_lens) {
  const size_t count = keys.size();
  if (value_lens) value_lens->assign(count, 0);
  if (destinations.size() != count) return InvalidStatuses(count);
  std::vector<Status> result(count, Status::kInvalid);
  std::vector<RangeDstSegment> segments;
  segments.reserve(count);
  std::vector<PullItem> items;
  items.reserve(count);
  for (size_t item = 0; item < count; ++item) {
    const RangeDst& destination = destinations[item];
    if (!ValidBuffer(destination.payload, destination.n) ||
        NoteBlock(destination.n))
      continue;
    segments.emplace_back(destination.payload, destination.n,
                          destination.memory_kind);
    items.push_back(
        PullItem{item, nullptr, 1, destination.n, destination.exact});
  }
  for (size_t i = 0; i < items.size(); ++i) items[i].segments = &segments[i];
  PullIntoMany(node, keys, items, Lane::kData, &result, value_lens, nullptr);
  return result;
}

std::vector<Status> RdmaTransport::RangeIntoMulti(
    const std::string& node, const std::vector<BlockKey>& keys,
    const std::vector<RangeDstMulti>& destinations,
    std::vector<size_t>* out_lengths, std::string* out_dev) {
  const size_t count = keys.size();
  if (out_lengths) out_lengths->assign(count, 0);
  if (out_dev) out_dev->clear();
  if (destinations.size() != count) return InvalidStatuses(count);
  std::vector<Status> result(count, Status::kInvalid);
  std::vector<PullItem> items;
  items.reserve(count);
  for (size_t item = 0; item < count; ++item) {
    const auto& segments = destinations[item].payloads;
    size_t capacity = 0;
    bool valid = true;
    for (const auto& segment : segments) {
      size_t next = 0;
      if (!ValidBuffer(segment.first, segment.second) ||
          !CheckedAddSize(capacity, segment.second, &next)) {
        valid = false;
        break;
      }
      capacity = next;
    }
    if (!valid || NoteBlock(capacity)) continue;
    items.push_back(
        PullItem{item, segments.data(), segments.size(), capacity});
  }
  std::vector<uint64_t> value_lens(count, 0);
  PullIntoMany(node, keys, items, Lane::kSgData, &result, &value_lens,
               out_dev);
  if (out_lengths) {
    for (const PullItem& item : items) {
      const uint64_t value_len = value_lens[item.index];
      if (result[item.index] == Status::kOk ||
          (result[item.index] == Status::kInvalid &&
           value_len > item.capacity))
        (*out_lengths)[item.index] = static_cast<size_t>(value_len);
    }
  }
  return result;
}

size_t RdmaTransport::ConnectionBound(size_t required) const {
  const size_t logical_max = OpBound();
  size_t wanted = std::max(required, connection_min_block_bytes_);
  if (wanted >= logical_max) return logical_max;
  size_t size_class = 1;
  while (size_class < wanted && size_class <= logical_max / 2)
    size_class <<= 1;
  return size_class < wanted ? logical_max
                             : std::min(size_class, logical_max);
}

size_t RdmaTransport::ConnectionDepth(const std::string& node, Lane lane,
                                      size_t requested_credits) {
  const size_t learned = std::max<size_t>(1, LearnedDepth(node, lane));
  const size_t wanted =
      std::min(learned, std::max<size_t>(1, requested_credits));
  size_t depth_class = 1;
  while (depth_class < wanted && depth_class <= learned / 2)
    depth_class <<= 1;
  return depth_class < wanted ? learned : std::min(depth_class, learned);
}

}  // namespace dfkv
