// Recent-REMOVE memory (CLI-02 stage M): what a node answers to "was this key
// removed here within the last age_ms?".
#include "cache/remove_memory.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "utils/net_util.h"

using namespace dfkv;  // NOLINT

namespace {

std::string Digests(const std::vector<BlockKey>& keys) {
  std::string d(keys.size() * 16, '\0');
  for (size_t i = 0; i < keys.size(); ++i) {
    net::PutU64(&d[i * 16], keys[i].digest_hi);
    net::PutU64(&d[i * 16 + 8], keys[i].digest_lo);
  }
  return d;
}

std::vector<uint8_t> Ask(RemoveMemory& m, const std::vector<BlockKey>& keys,
                         uint64_t age, uint64_t now, uint64_t tenant = 7,
                         uint64_t* covered = nullptr) {
  const std::string d = Digests(keys);
  std::vector<uint8_t> out(keys.size(), 9);
  const uint64_t c = m.Query(tenant, d.data(), keys.size(), age, now, out.data());
  if (covered) *covered = c;
  return out;
}

const BlockKey kA{1, 2, 7}, kB{3, 4, 7};

}  // namespace

TEST(RemoveMemory, AnswersWhetherAKeyWasRemovedWithinTheAge) {
  RemoveMemory m(600000, 1024, 1000);
  m.Record(kA, 5000);
  // At 9000: kA removed 4 s ago, kB never.
  EXPECT_EQ(Ask(m, {kA, kB}, 5000, 9000),
            (std::vector<uint8_t>{RemoveMemory::kRemoved,
                                  RemoveMemory::kNotRemoved}));
  // An age that ends after the REMOVE: not removed within it.
  EXPECT_EQ(Ask(m, {kA}, 3000, 9000)[0], RemoveMemory::kNotRemoved);
  // Removed again later: the last REMOVE counts.
  m.Record(kA, 8500);
  EXPECT_EQ(Ask(m, {kA}, 1000, 9000)[0], RemoveMemory::kRemoved);
}

TEST(RemoveMemory, TenantsAreKeptApart) {
  RemoveMemory m(600000, 1024, 1000);
  m.Record(kA, 5000);
  EXPECT_EQ(Ask(m, {kA}, 5000, 9000, /*tenant=*/8)[0],
            RemoveMemory::kNotRemoved);
}

// After a start (and after a drop for room) the node knows only what came
// since: a longer age is unknown.
TEST(RemoveMemory, AnAgeBeforeWhatTheNodeKnowsIsUnknown) {
  RemoveMemory m(600000, 1024, 10000);
  uint64_t covered = 0;
  EXPECT_EQ(Ask(m, {kA, kB}, 5001, 15000, 7, &covered),
            (std::vector<uint8_t>{RemoveMemory::kUnknown,
                                  RemoveMemory::kUnknown}));
  EXPECT_EQ(covered, 5000u);
  EXPECT_EQ(Ask(m, {kA}, 5000, 15000)[0], RemoveMemory::kNotRemoved);
  // ...and at most the window.
  RemoveMemory w(1000, 1024, 0);
  EXPECT_EQ(Ask(w, {kA}, 1001, 50000)[0], RemoveMemory::kUnknown);
  EXPECT_EQ(Ask(w, {kA}, 1000, 50000)[0], RemoveMemory::kNotRemoved);
}

TEST(RemoveMemory, RecordsOlderThanTheWindowAreDropped) {
  RemoveMemory m(1000, 1024, 0);
  m.Record(kA, 100);
  EXPECT_EQ(m.Entries(), 1u);
  m.Record(kB, 1200);  // expires kA's record
  EXPECT_EQ(m.Entries(), 1u);
  EXPECT_EQ(m.Dropped(), 0u) << "expiry is not a drop for room";
}

// Full: the oldest record goes, and the node no longer knows the time up to
// it (a question reaching there is unknown).
TEST(RemoveMemory, AFullMemoryDropsTheOldestAndKnowsLess) {
  RemoveMemory m(600000, 2, 0);
  m.Record(BlockKey{1, 1, 7}, 100);
  m.Record(BlockKey{2, 2, 7}, 200);
  m.Record(BlockKey{3, 3, 7}, 300);  // drops the record at 100
  EXPECT_EQ(m.Dropped(), 1u);
  EXPECT_EQ(m.Entries(), 2u);
  EXPECT_EQ(m.CoveredMs(1000), 899u);
  EXPECT_EQ(Ask(m, {BlockKey{1, 1, 7}}, 900, 1000)[0], RemoveMemory::kUnknown);
  EXPECT_EQ(Ask(m, {BlockKey{2, 2, 7}}, 800, 1000)[0], RemoveMemory::kRemoved);
  EXPECT_EQ(m.QueriedKeys(RemoveMemory::kUnknown), 1u);
  EXPECT_EQ(m.QueriedKeys(RemoveMemory::kRemoved), 1u);
}

// Two threads read the clock before the lock: the one with the older time
// can take the lock second. Its expiry must not take the newer record (an
// underflow of now - at), and the answer must stay "removed".
TEST(RemoveMemory, AnOlderClockReadingDoesNotExpireANewerRecord) {
  RemoveMemory m(600000, 1024, 0);
  m.Record(kB, 101);  // thread B, clock 101, first at the lock
  m.Record(kA, 100);  // thread A, clock 100, second
  EXPECT_EQ(m.Entries(), 2u);
  EXPECT_EQ(Ask(m, {kA, kB}, 50, 120),
            (std::vector<uint8_t>{RemoveMemory::kRemoved,
                                  RemoveMemory::kRemoved}));
  // A query with an older clock than the table's is answered at the table's.
  EXPECT_EQ(Ask(m, {kB}, 50, 99)[0], RemoveMemory::kRemoved);
}

// A record dropped for room was removed at its time: an age reaching that
// time is unknown (coverage starts just after it), not "not removed".
TEST(RemoveMemory, OverflowingByOneMakesTheDroppedTimeUnknown) {
  RemoveMemory m(600000, 1024, 0);
  for (uint64_t i = 1; i <= 1024; ++i) m.Record(BlockKey{i, i, 7}, i);
  m.Record(BlockKey{5000, 5000, 7}, 1025);  // drops the record at time 1
  EXPECT_EQ(m.Dropped(), 1u);
  const uint64_t now = 2000;
  EXPECT_EQ(Ask(m, {BlockKey{1, 1, 7}}, now - 1, now)[0],
            RemoveMemory::kUnknown);
  EXPECT_EQ(Ask(m, {BlockKey{2, 2, 7}}, now - 2, now)[0],
            RemoveMemory::kRemoved);
}

TEST(RemoveMemory, OffUnlessConfigured) {
  ::unsetenv("DFKV_REMOVE_MEMORY_MS");
  EXPECT_EQ(RemoveMemory::FromEnv(0), nullptr);
  ::setenv("DFKV_REMOVE_MEMORY_MS", "600000", 1);
  EXPECT_NE(RemoveMemory::FromEnv(0), nullptr);
  ::unsetenv("DFKV_REMOVE_MEMORY_MS");
}
