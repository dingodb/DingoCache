// Recent-REMOVE memory (CLI-02 stage M): what a node answers to "was this key
// removed here within the last age_ms?".
#include "cache/remove_memory.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <deque>
#include <map>
#include <random>
#include <string>
#include <tuple>
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

// The table against a plain model (record order and each key's latest
// REMOVE, as the first version kept them), through growth, a full ring,
// expiry and keys removed again: every answer, CoveredMs and Entries agree.
// Keys come from a small set, so index runs form, wrap and are shifted on
// deletion.
TEST(RemoveMemory, AgreesWithAPlainModelUnderChurn) {
  struct Model {
    uint64_t window, capacity, covered_since, last_now = 0;
    std::deque<std::pair<BlockKey, uint64_t>> order;
    std::map<std::tuple<uint64_t, uint64_t, uint64_t>, uint64_t> last;
    static std::tuple<uint64_t, uint64_t, uint64_t> T(const BlockKey& k) {
      return {k.digest_hi, k.digest_lo, k.tenant_hash};
    }
    void Pop() {
      const auto it = last.find(T(order.front().first));
      if (it != last.end() && it->second == order.front().second) last.erase(it);
      order.pop_front();
    }
    uint64_t Now(uint64_t now) { return last_now = std::max(last_now, now); }
    void Expire(uint64_t now) {
      while (!order.empty() && now >= order.front().second &&
             now - order.front().second > window)
        Pop();
    }
    void Record(const BlockKey& k, uint64_t now) {
      now = Now(now);
      Expire(now);
      while (order.size() >= capacity) {
        covered_since = std::max(covered_since, order.front().second + 1);
        Pop();
      }
      order.emplace_back(k, now);
      last[T(k)] = now;
    }
    uint8_t Ask(const BlockKey& k, uint64_t age, uint64_t now) {
      now = Now(now);
      Expire(now);
      const uint64_t covered =
          std::min(now > covered_since ? now - covered_since : 0, window);
      if (age > covered) return RemoveMemory::kUnknown;
      const auto it = last.find(T(k));
      return it != last.end() && now - it->second <= age ? RemoveMemory::kRemoved
                                                         : RemoveMemory::kNotRemoved;
    }
  };
  // Grown by doubling, and allocated whole at start (presize_above 0).
  for (const auto& [capacity, presize] :
       std::vector<std::pair<uint64_t, bool>>{{1500, false}, {3000, false},
                                              {3000, true}}) {
    RemoveMemory m(4000, capacity, 0,
                   presize ? 0 : RemoveMemory::kPresizeAbove);
    Model model{4000, capacity, 0};
    std::mt19937_64 rng(capacity);
    std::vector<BlockKey> keys;
    for (uint64_t i = 0; i < 3000; ++i) keys.push_back({i * 0x9e37u, i, 7});
    uint64_t now = 0;
    for (int step = 0; step < 60000; ++step) {
      now += rng() % 500 == 0 ? 1500 : rng() % 3;  // jumps let records expire
      const uint64_t skew = rng() % 4 == 0 ? rng() % 5 : 0;  // an older reading
      const BlockKey& k = keys[rng() % keys.size()];
      m.Record(k, now - std::min(now, skew));
      model.Record(k, now - std::min(now, skew));
      if (step % 7 == 0) {
        const BlockKey& q = keys[rng() % keys.size()];
        const uint64_t age = rng() % 4500;
        ASSERT_EQ(Ask(m, {q}, age, now)[0], model.Ask(q, age, now))
            << "capacity " << capacity << " step " << step;
      }
      if (step % 997 == 0) {
        ASSERT_EQ(m.CoveredMs(now),
                  std::min(now > model.covered_since ? now - model.covered_since : 0,
                           model.window));
        ASSERT_EQ(m.Entries(), model.last.size()) << "step " << step;
      }
    }
    EXPECT_GT(m.Dropped(), 0u) << "the ring never filled";
  }
}

// Above kPresizeAbove the ring and index are allocated whole at start and
// never grow (no copy under the lock on the REMOVE path); below it they
// start small.
TEST(RemoveMemory, ALargeCapacityIsAllocatedAtStartAndNeverGrows) {
  RemoveMemory small(600000, 5000, 0);
  EXPECT_LT(small.RingSlotsForTest(), 5000u);
  RemoveMemory big(600000, 5000, 0, /*presize_above=*/4096);
  EXPECT_EQ(big.RingSlotsForTest(), 5000u);
  const size_t index = big.IndexSlotsForTest();
  EXPECT_GE(index * 7, 5000u * 10) << "the index would grow";
  for (uint64_t i = 0; i < 12000; ++i) big.Record(BlockKey{i, i, 7}, 1 + i / 1000);
  EXPECT_EQ(big.RingSlotsForTest(), 5000u);
  EXPECT_EQ(big.IndexSlotsForTest(), index);
  EXPECT_EQ(big.Entries(), 5000u);
  // Records at 1..12 ms; the oldest dropped was at 7, so at 13 the table
  // knows 5 ms back.
  EXPECT_EQ(Ask(big, {BlockKey{11999, 11999, 7}, BlockKey{0, 0, 7}}, 3, 13),
            (std::vector<uint8_t>{RemoveMemory::kRemoved, RemoveMemory::kNotRemoved}));
  EXPECT_EQ(Ask(big, {BlockKey{11999, 11999, 7}}, 10, 13),
            (std::vector<uint8_t>{RemoveMemory::kUnknown}));
  // The default: above 4 Mi entries.
  EXPECT_EQ(RemoveMemory::kPresizeAbove, size_t{4} << 20);
}
