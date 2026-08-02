#include "transport/transport.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace dfkv {
namespace {

class ValidationTransport final : public Transport {
 public:
  Status Cache(const std::string&, const BlockKey&, const void* data,
               size_t len) override {
    return ValidBuffer(data, len) ? Status::kOk : Status::kInvalid;
  }
  Status Range(const std::string&, const BlockKey&, uint64_t, uint64_t length,
               std::string* out, uint64_t* value_len) override {
    if (out == nullptr || length > std::numeric_limits<size_t>::max())
      return Status::kInvalid;
    out->assign(static_cast<size_t>(length), 'x');
    if (value_len) *value_len = length;
    return Status::kOk;
  }
  Status Lookup(const std::string&, const BlockKey&, uint64_t* value_len) override {
    if (!value_len) return Status::kInvalid;
    *value_len = 0;
    return Status::kNotFound;
  }
  Status Exist(const std::string&, const BlockKey&, bool* exist) override {
    if (!exist) return Status::kInvalid;
    *exist = false;
    return Status::kOk;
  }
};

TEST(TransportValidation, RejectsMismatchedKeyDestinationVectors) {
  ValidationTransport transport;
  std::vector<BlockKey> keys = {{1, 2}, {3, 4}};
  std::vector<RangeDst> one_destination = {{nullptr, 0}};
  std::vector<uint64_t> value_lengths;
  const auto statuses =
      transport.RangeInto("node", keys, one_destination, &value_lengths);
  ASSERT_EQ(statuses.size(), keys.size());
  EXPECT_EQ(statuses[0], Status::kInvalid);
  EXPECT_EQ(statuses[1], Status::kInvalid);
  EXPECT_EQ(value_lengths, std::vector<uint64_t>({0, 0}));

  std::vector<RangeDstMulti> no_destinations;
  std::vector<size_t> out_lengths;
  const auto sg_statuses =
      transport.RangeIntoMulti("node", keys, no_destinations, &out_lengths);
  ASSERT_EQ(sg_statuses.size(), keys.size());
  EXPECT_EQ(sg_statuses[0], Status::kInvalid);
  EXPECT_EQ(sg_statuses[1], Status::kInvalid);
  EXPECT_EQ(out_lengths, std::vector<size_t>({0, 0}));
}

TEST(TransportValidation, RejectsNullBuffersPerItemWithoutPoisoningSiblings) {
  ValidationTransport transport;
  char destination[4] = {};
  const std::vector<BlockKey> keys = {{1, 2}, {3, 4}};
  const std::vector<RangeDst> destinations = {
      {nullptr, 4}, {destination, sizeof(destination)}};
  const auto get_statuses =
      transport.RangeInto("node", keys, destinations, nullptr);
  ASSERT_EQ(get_statuses.size(), 2u);
  EXPECT_EQ(get_statuses[0], Status::kInvalid);
  EXPECT_EQ(get_statuses[1], Status::kOk);
  EXPECT_EQ(std::string(destination, sizeof(destination)), "xxxx");

  const std::vector<CacheItem> items = {
      {{1, 2}, nullptr, 1}, {{3, 4}, destination, sizeof(destination)}};
  const auto put_statuses = transport.CacheMany("node", items);
  ASSERT_EQ(put_statuses.size(), 2u);
  EXPECT_EQ(put_statuses[0], Status::kInvalid);
  EXPECT_EQ(put_statuses[1], Status::kOk);
}

TEST(TransportValidation, SegmentSumsRejectOverflowBeforeResizeOrCopy) {
  ValidationTransport transport;
  const void* nonnull = reinterpret_cast<const void*>(uintptr_t{1});
  CacheSrcMulti overflow_write{
      {1, 2},
      {{nonnull, std::numeric_limits<size_t>::max()}, {nonnull, 1}}};
  const auto write_status =
      transport.CacheFromMulti("node", {overflow_write});
  ASSERT_EQ(write_status.size(), 1u);
  EXPECT_EQ(write_status[0], Status::kInvalid);

  RangeDstMulti overflow_read{{
      {reinterpret_cast<void*>(uintptr_t{1}),
       std::numeric_limits<size_t>::max()},
      {reinterpret_cast<void*>(uintptr_t{1}), 1},
  }};
  std::vector<size_t> out_lengths;
  const auto read_status = transport.RangeIntoMulti(
      "node", {{1, 2}}, {overflow_read}, &out_lengths);
  ASSERT_EQ(read_status.size(), 1u);
  EXPECT_EQ(read_status[0], Status::kInvalid);
  EXPECT_EQ(out_lengths, std::vector<size_t>({0}));
}

TEST(TransportValidation, ScatterGatherRejectsNullNonEmptySegments) {
  ValidationTransport transport;
  CacheSrcMulti bad_write{{1, 2}, {{nullptr, 1}}};
  EXPECT_EQ(transport.CacheFromMulti("node", {bad_write})[0],
            Status::kInvalid);

  RangeDstMulti bad_read{{{nullptr, 1}}};
  EXPECT_EQ(transport.RangeIntoMulti("node", {{1, 2}}, {bad_read}, nullptr)[0],
            Status::kInvalid);
}

}  // namespace
}  // namespace dfkv
