// RamTier: write-through RAM hot tier on SlabAllocator. Hermetic (no RDMA/disk;
// the flush sink is an injected callback). Covers read-after-write, the
// RAM_ONLY->DURABLE state machine via pins, send-in-flight pin blocking
// eviction, backpressure bypass when flush stalls, flush retry/drop, sub-range
// GetPrep, Remove semantics, and a concurrent TSan stress.
#include "cache/ram_tier.h"

#include <gtest/gtest.h>
#include <sys/mman.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using dfkv::BlockKey;
using dfkv::RamTier;
using dfkv::Status;
using namespace std::chrono_literals;

namespace dfkv {
// Exercise the admission/completion boundary without scheduling a duplicate
// into a sleep-sized race window.
class RamTierTestPeer {
 public:
  using Admission = RamTier::Admission;
  using Completion = std::shared_ptr<RamTier::PutCompletion>;

  static Admission Admit(RamTier& tier, const BlockKey& key,
                         const std::string& value, Completion* completion,
                         bool client_ack = false) {
    return tier.Admit(key, value.data(), value.size(), client_ack, completion);
  }
  static bool Done(const Completion& completion) {
    std::lock_guard<std::mutex> lock(completion->mu);
    return completion->done;
  }
  static bool AwaitDone(const Completion& completion) {
    std::unique_lock<std::mutex> lock(completion->mu);
    return completion->cv.wait_for(lock, 2s, [&] { return completion->done; });
  }
  static Status Wait(const Completion& completion) {
    return RamTier::WaitPut(completion);
  }
  static const char* ArenaBase(const RamTier& tier) { return tier.arena_; }
  // One reclaimer pass over every shard, as the background thread runs it.
  static void ReclaimTick(RamTier& tier) {
    for (size_t i = 0; i < tier.shards_.size(); ++i)
      tier.ReclaimTick(*tier.shards_[i], i);
  }
  // Extents admissions took from colder classes (not background growth).
  static uint64_t ColdSteals(const RamTier& tier) {
    uint64_t n = 0;
    for (const auto& shard : tier.shards_) n += shard->alloc->ColdSteals();
    return n;
  }
};
}  // namespace dfkv
using dfkv::RamTierTestPeer;

namespace {
BlockKey K(uint64_t id) { return BlockKey{id, 0}; }

// A controllable flush sink: records flushed keys, can be gated (block until
// opened) and can be made to fail.
struct FlushSink {
  std::mutex m;
  std::condition_variable cv;
  std::set<std::string> flushed;
  std::unordered_map<std::string, std::string> values;
  bool gate_open = true;      // when false, flush blocks
  bool fail = false;          // when true, flush returns false
  int calls = 0;

  RamTier::FlushFn fn() {
    return [this](const BlockKey& k, char* data, size_t len, size_t) {
      std::unique_lock<std::mutex> lk(m);
      ++calls;
      cv.notify_all();
      cv.wait(lk, [this] { return gate_open; });
      if (fail) return Status::kIOError;
      flushed.insert(k.Filename());
      values.emplace(k.Filename(), std::string(data, len));
      cv.notify_all();
      return Status::kOk;
    };
  }
  void open() { { std::lock_guard<std::mutex> lk(m); gate_open = true; } cv.notify_all(); }
  void close() { std::lock_guard<std::mutex> lk(m); gate_open = false; }
  bool await_calls(int count) {
    std::unique_lock<std::mutex> lk(m);
    return cv.wait_for(lk, 2s, [&] { return calls >= count; });
  }
};

// Declare after RamTier: fatal assertions must open the sink before the tier
// destructor joins its flush workers.
struct OpenFlushOnExit {
  FlushSink& sink;
  ~OpenFlushOnExit() { sink.open(); }
};

// Opens the sink if its scope has not ended after `limit`. A PUT that should
// have stopped waiting then ends with the flush result and fails its
// expectation, instead of hanging the test.
class OpenFlushAfter {
 public:
  OpenFlushAfter(FlushSink& sink, std::chrono::milliseconds limit)
      : thread_([this, &sink, limit] {
          std::unique_lock<std::mutex> lock(mu_);
          if (!cv_.wait_for(lock, limit, [this] { return done_; })) sink.open();
        }) {}
  ~OpenFlushAfter() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      done_ = true;
    }
    cv_.notify_all();
    thread_.join();
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  bool done_ = false;
  std::thread thread_;
};

// Poll a predicate up to ~2s.
template <class F>
bool WaitFor(F pred) {
  for (int i = 0; i < 2000; ++i) {
    if (pred()) return true;
    std::this_thread::sleep_for(1ms);
  }
  return pred();
}

// Set an environment variable for a scope and put back what was there.
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) {
      had_ = true;
      old_ = old;
    }
    setenv(name, value, 1);
  }
  ~ScopedEnv() {
    if (had_)
      setenv(name_, old_.c_str(), 1);
    else
      unsetenv(name_);
  }
  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

 private:
  const char* name_;
  bool had_ = false;
  std::string old_;
};

RamTier::Options Opts(uint64_t bytes, uint32_t gran = 4096) {
  RamTier::Options o;
  o.bytes = bytes;
  o.slot_granularity = gran;
  return o;
}
}  // namespace

TEST(RamTier, ReadAfterWriteBeforeFlush) {
  FlushSink sink;
  sink.close();  // hold flushes so the slot stays RAM_ONLY
  RamTier rt(Opts(64 * 4096), sink.fn());
  ASSERT_TRUE(rt.ok());
  std::string v = "hot-kv-payload";
  ASSERT_TRUE(rt.Put(K(1), v.data(), v.size()));
  // Visible immediately, even though the flush hasn't run.
  RamTier::Hit h;
  ASSERT_TRUE(rt.GetPrep(K(1), 0, v.size(), &h));
  EXPECT_EQ(std::string(h.ptr, h.len), v);
  EXPECT_TRUE(h.in_arena);
  EXPECT_EQ(rt.Hits(), 1u);
  h = RamTier::Hit{};
  sink.open();
}

TEST(RamTier, ShutdownDrainTimeoutAndAckDurabilityLatencyAreObservable) {
  FlushSink sink;
  sink.close();
  RamTier rt(Opts(64 * 4096), sink.fn());
  const std::string value(4096, 'd');
  ASSERT_EQ(rt.PutWriteBack(K(2), value.data(), value.size()).status, Status::kOk);

  EXPECT_FALSE(rt.WaitForDrain(10ms));
  EXPECT_EQ(rt.ShutdownDrainTimeouts(), 1u);
  sink.open();
  ASSERT_TRUE(rt.WaitForDrain(2s));
  EXPECT_EQ(rt.DirtyObjects(), 0u);
  EXPECT_NE(rt.AckToDurableLatencyMetrics("").find(
                "dfkv_ram_ack_to_durable_latency_seconds_count 1"),
            std::string::npos);
}

// A stopping server gives the backlog a time limit. When it runs out, the
// flushers finish what they hold and leave the rest: nothing is written after
// StopBackground returns, and destruction does not resume the drain.
TEST(RamTier, StopBackgroundCanLeaveTheBacklogBehind) {
  FlushSink sink;
  sink.close();
  constexpr int kObjects = 12;
  const std::string value(4096, 'b');
  {
    RamTier rt(Opts(64 * 4096), sink.fn());
    OpenFlushOnExit cleanup{sink};
    ASSERT_TRUE(rt.ok());
    ASSERT_EQ(rt.flusher_count(), 1u);
    // The flusher takes the first object alone and blocks in the sink; the
    // others queue behind it.
    ASSERT_EQ(rt.PutWriteBack(K(100), value.data(), value.size()).status,
              Status::kOk);
    ASSERT_TRUE(sink.await_calls(1));
    for (int i = 1; i < kObjects; ++i)
      ASSERT_EQ(rt.PutWriteBack(K(100 + i), value.data(), value.size()).status,
                Status::kOk);
    ASSERT_EQ(rt.DirtyObjects(), static_cast<uint64_t>(kObjects));

    std::thread stopper([&] { rt.StopBackground(/*abandon_backlog=*/true); });
    const bool stopping = WaitFor([&] { return rt.background_stopping(); });
    sink.open();
    stopper.join();
    ASSERT_TRUE(stopping);

    EXPECT_EQ(sink.calls, 1);  // the object in hand, and only that one
    EXPECT_EQ(rt.DirtyObjects(), static_cast<uint64_t>(kObjects - 1));
    rt.StopBackground(/*abandon_backlog=*/false);  // already stopped: no drain
    EXPECT_EQ(rt.DirtyObjects(), static_cast<uint64_t>(kObjects - 1));
    // The values are still readable from RAM until the tier goes away.
    RamTier::Hit hit;
    ASSERT_TRUE(rt.GetPrep(K(100 + kObjects - 1), 0, value.size(), &hit));
    EXPECT_EQ(std::string(hit.ptr, hit.len), value);
  }
  EXPECT_EQ(sink.calls, 1);  // the destructor wrote nothing either
}

TEST(RamTier, StopBackgroundWritesTheBacklogUnlessToldOtherwise) {
  FlushSink sink;
  sink.close();
  constexpr int kObjects = 12;
  const std::string value(4096, 'w');
  RamTier rt(Opts(64 * 4096), sink.fn());
  OpenFlushOnExit cleanup{sink};
  ASSERT_TRUE(rt.ok());
  for (int i = 0; i < kObjects; ++i)
    ASSERT_EQ(rt.PutWriteBack(K(200 + i), value.data(), value.size()).status,
              Status::kOk);
  std::thread stopper([&] { rt.StopBackground(/*abandon_backlog=*/false); });
  const bool stopping = WaitFor([&] { return rt.background_stopping(); });
  sink.open();
  stopper.join();
  ASSERT_TRUE(stopping);
  EXPECT_EQ(rt.DirtyObjects(), 0u);
  std::lock_guard<std::mutex> lock(sink.m);
  EXPECT_EQ(sink.flushed.size(), static_cast<size_t>(kObjects));
}

TEST(RamTier, TenantIdentityIsPartOfResidentKey) {
  FlushSink sink;
  RamTier rt(Opts(64 * 4096), sink.fn());
  const BlockKey first{7, 11, 101};
  const BlockKey second{7, 11, 202};
  const std::string first_value = "tenant-a";
  const std::string second_value = "tenant-b";

  ASSERT_TRUE(
      rt.PutDurable(first, first_value.data(), first_value.size()));
  ASSERT_TRUE(
      rt.PutDurable(second, second_value.data(), second_value.size()));

  RamTier::Hit hit;
  ASSERT_TRUE(rt.GetPrep(first, 0, 0, &hit));
  EXPECT_EQ(std::string(hit.ptr, hit.len), first_value);
  hit = RamTier::Hit{};
  ASSERT_TRUE(rt.GetPrep(second, 0, 0, &hit));
  EXPECT_EQ(std::string(hit.ptr, hit.len), second_value);
  hit = RamTier::Hit{};

  EXPECT_TRUE(rt.Remove(first));
  EXPECT_FALSE(rt.Contains(first));
  EXPECT_TRUE(rt.Contains(second));
  ASSERT_TRUE(rt.GetPrep(second, 0, 0, &hit));
  EXPECT_EQ(std::string(hit.ptr, hit.len), second_value);
}

TEST(RamTier, WriteBackAcknowledgesVisiblePinnedRamBeforeFlush) {
  FlushSink sink;
  sink.close();
  RamTier rt(Opts(64 * 4096), sink.fn());
  const std::string value(4096, 'w');

  EXPECT_EQ(rt.PutWriteBack(K(101), value.data(), value.size()).status, Status::kOk);
  EXPECT_EQ(rt.RamAcks(), 1u);
  EXPECT_EQ(rt.DirtyObjects(), 1u);
  EXPECT_EQ(rt.DirtyBytes(), 4096u);
  RamTier::Hit hit;
  ASSERT_TRUE(rt.GetPrep(K(101), 0, 0, &hit));
  EXPECT_EQ(std::string(hit.ptr, hit.len), value);
  hit = RamTier::Hit{};

  sink.open();
  ASSERT_TRUE(WaitFor([&] { return rt.Flushed() == 1u; }));
  EXPECT_EQ(rt.DirtyObjects(), 0u);
  EXPECT_EQ(rt.DirtyBytes(), 0u);
}

TEST(RamTier, ZeroWatermarkWaitsForDiskCommit) {
  FlushSink sink;
  sink.close();
  RamTier::Options options = Opts(2 * 4096, 4096);
  options.large_reserve_bytes = 0;
  options.ack_high_watermark_pct = 0;
  RamTier rt(options, sink.fn());
  OpenFlushOnExit cleanup{sink};
  const std::string value(4096, 'b');
  Status result = Status::kInvalid;
  std::thread put([&] {
    result = rt.PutWriteBack(K(103), value.data(), value.size()).status;
  });
  // The flush callback is the barrier: this request has been admitted and
  // cannot be durable until we release it. No scheduling sleeps are needed.
  const bool entered = sink.await_calls(1);
  EXPECT_TRUE(entered);
  if (entered) {
    EXPECT_EQ(rt.RamAcks(), 0u);
    EXPECT_EQ(rt.DirtyObjects(), 1u);
  }
  sink.open();
  put.join();
  EXPECT_EQ(result, Status::kOk);
  EXPECT_EQ(rt.AckBackpressure(), 1u);
  EXPECT_EQ(rt.DirtyObjects(), 0u);
}

TEST(RamTier, PositiveWatermarkRejectsNewKeysButPreservesExistingOwners) {
  FlushSink sink;
  sink.close();
  RamTier::Options options = Opts(4 * 4096);
  options.large_reserve_bytes = 0;
  options.ack_high_watermark_pct = 25;
  RamTier rt(options, sink.fn());
  OpenFlushOnExit cleanup{sink};
  const std::string first(500, 'a'), different(1000, 'b');
  const auto accepted = rt.PutWriteBack(K(105), first.data(), first.size());
  ASSERT_EQ(accepted.status, Status::kOk);
  EXPECT_EQ(accepted.disposition, RamTier::PutDisposition::kComplete);
  ASSERT_TRUE(sink.await_calls(1));
  ASSERT_EQ(rt.DirtyBytes(), 4096u);  // exactly the positive watermark

  // Both payloads resolve to the same pending completion even at saturation.
  RamTierTestPeer::Completion leader, duplicate, rejected;
  ASSERT_EQ(RamTierTestPeer::Admit(rt, K(105), first, &leader, true),
            RamTierTestPeer::Admission::kDuplicate);
  ASSERT_EQ(RamTierTestPeer::Admit(rt, K(105), different, &duplicate, true),
            RamTierTestPeer::Admission::kDuplicate);
  EXPECT_EQ(leader, duplicate);
  EXPECT_FALSE(RamTierTestPeer::Done(duplicate));
  ASSERT_EQ(RamTierTestPeer::Admit(rt, K(106), different, &rejected, true),
            RamTierTestPeer::Admission::kBackpressure);
  EXPECT_FALSE(rejected);  // rejection created no ownership or flush work
  EXPECT_FALSE(rt.Contains(K(106)));
  EXPECT_EQ(rt.DirtyBytes(), 4096u);
  EXPECT_EQ(rt.PutBypass(), 0u);
  EXPECT_EQ(rt.AckBackpressure(), 1u);

  // Tombstones also resolve before the watermark and retain their error.
  ASSERT_TRUE(rt.PutDurable(K(107), first.data(), first.size()));
  RamTier::Hit pin;
  ASSERT_TRUE(rt.GetPrep(K(107), 0, 0, &pin));
  ASSERT_TRUE(rt.Remove(K(107)));
  const auto removed = rt.PutWriteBack(K(107), different.data(), different.size());
  EXPECT_EQ(removed.status, Status::kIOError);
  EXPECT_EQ(removed.disposition, RamTier::PutDisposition::kComplete);
  pin = RamTier::Hit{};
  ASSERT_TRUE(rt.GetPrep(K(105), 0, 0, &pin));
  EXPECT_EQ(std::string(pin.ptr, pin.len), first);
  pin = RamTier::Hit{};

  sink.open();
  ASSERT_TRUE(RamTierTestPeer::AwaitDone(leader));
  EXPECT_EQ(RamTierTestPeer::Wait(duplicate), Status::kOk);
  EXPECT_EQ(rt.DirtyBytes(), 0u);
  const auto recovered = rt.PutWriteBack(K(106), different.data(), different.size());
  EXPECT_EQ(recovered.status, Status::kOk);
  EXPECT_EQ(recovered.disposition, RamTier::PutDisposition::kComplete);
  ASSERT_EQ(rt.PutCommitted(K(106), first.data(), first.size()).status,
            Status::kOk);
  std::lock_guard<std::mutex> lock(sink.m);
  EXPECT_EQ(sink.values.at(K(105).Filename()), first);
  EXPECT_EQ(sink.values.at(K(106).Filename()), different);
}

TEST(RamTier, CapacityBypassRemainsDistinctFromTerminalBackpressure) {
  FlushSink sink;
  RamTier::Options options = Opts(4 * 4096);
  options.large_reserve_bytes = 0;
  RamTier rt(options, sink.fn());
  const std::string too_large(options.bytes + 4096, 'x');
  for (const auto result : {
           rt.PutWriteBack(K(108), too_large.data(), too_large.size()),
           rt.PutCommitted(K(109), too_large.data(), too_large.size())}) {
    EXPECT_EQ(result.status, Status::kCacheFull);
    EXPECT_EQ(result.disposition, RamTier::PutDisposition::kBypass);
  }
  EXPECT_EQ(rt.Count(), 0u);
  EXPECT_EQ(rt.DirtyBytes(), 0u);
  EXPECT_EQ(rt.AckBackpressure(), 0u);
}

TEST(RamTier, WriteBackFlushFailureIsPostAckAndUnhealthy) {
  FlushSink sink;
  sink.fail = true;
  RamTier::Options options = Opts(8 * 4096);
  options.flush_retries = 1;
  RamTier rt(options, sink.fn());
  const std::string value(500, 'f');

  EXPECT_EQ(rt.PutWriteBack(K(104), value.data(), value.size()).status, Status::kOk);
  ASSERT_TRUE(WaitFor([&] { return rt.FlushDropped() == 1u; }));
  EXPECT_EQ(rt.PostAckFlushFailures(), 1u);
  EXPECT_EQ(rt.DirtyObjects(), 0u);
  EXPECT_FALSE(rt.healthy());
  EXPECT_FALSE(rt.Contains(K(104)));
}

TEST(RamTier, FlushMakesDurable) {
  FlushSink sink;
  RamTier rt(Opts(64 * 4096), sink.fn());
  std::string v(1000, 'd');
  ASSERT_TRUE(rt.Put(K(2), v.data(), v.size()));
  EXPECT_TRUE(WaitFor([&] { return rt.Flushed() == 1u; }));
  {
    std::lock_guard<std::mutex> lk(sink.m);
    EXPECT_TRUE(sink.flushed.count(K(2).Filename()));
  }
  EXPECT_EQ(rt.FlushBacklog(), 0u);
}

TEST(RamTier, FlushBacklogIncludesDequeuedInflightItem) {
  FlushSink sink;
  sink.close();
  RamTier rt(Opts(64 * 4096), sink.fn());
  std::string v(1000, 'q');
  ASSERT_TRUE(rt.Put(K(3), v.data(), v.size()));
  ASSERT_TRUE(WaitFor([&] {
    std::lock_guard<std::mutex> lk(sink.m);
    return sink.calls == 1;
  }));
  EXPECT_EQ(rt.FlushBacklog(), 1u);
  sink.open();
  ASSERT_TRUE(WaitFor([&] { return rt.Flushed() == 1u; }));
  EXPECT_EQ(rt.FlushBacklog(), 0u);
}

// Regression (hd04 prod, 2026-07): the RAM tier bound its whole arena to ONE
// SlabAllocator extent, so it could hold only a single size class at a time. A
// real KV working set has assorted value sizes (partial blocks, variable-length
// coalesced chunks); each cross-class Put stole the lone extent and evicted the
// whole tier (ram_objects collapsed to ~1-5 in a 16 GiB arena), so every reuse
// GET missed to slab and load ran at disk speed. With the arena partitioned into
// many extents, distinct size classes coexist and stay resident. Also exercises
// the global-arena-offset math (slots now live in extents > 0).
TEST(RamTier, VaryingSizesCoexistNoChurn) {
  ::setenv("DFKV_RAM_TIER_EXTENT_BYTES", "2097152", 1);  // 2 MiB extents
  FlushSink sink;
  RamTier rt(Opts(256ull << 20), sink.fn());             // 256 MiB arena -> ~128 extents
  ASSERT_TRUE(rt.ok());
  // 6 distinct size classes x 4 keys each (models a real KV mix: a handful of
  // block sizes, many blocks each). 6 classes need far fewer than the 128
  // extents available. Each value < the 2 MiB extent.
  const size_t sizes[6] = {40000, 90000, 200000, 380000, 560000, 730000};
  std::vector<std::pair<uint64_t, size_t>> items;
  for (int i = 0; i < 24; ++i) {
    size_t sz = sizes[i % 6];
    std::vector<char> v(sz, static_cast<char>(i + 1));
    ASSERT_TRUE(rt.Put(K(1000 + i), v.data(), v.size())) << "put i=" << i;
    items.emplace_back(1000 + i, sz);
  }
  // All resident (no cross-class eviction) and byte-correct after the global
  // offset resolves each slot's extent.
  EXPECT_EQ(rt.Count(), items.size());
  EXPECT_EQ(rt.Evictions(), 0u);
  for (auto& [id, sz] : items) {
    RamTier::Hit h;
    ASSERT_TRUE(rt.GetPrep(K(id), 0, sz, &h)) << "get id=" << id;
    EXPECT_EQ(h.len, sz);
    EXPECT_EQ(static_cast<unsigned char>(h.ptr[0]),
              static_cast<unsigned char>((id - 1000) + 1));
    h = RamTier::Hit{};
  }
  ::unsetenv("DFKV_RAM_TIER_EXTENT_BYTES");
}

TEST(RamTier, MissReturnsFalse) {
  FlushSink sink;
  RamTier rt(Opts(64 * 4096), sink.fn());
  RamTier::Hit h;
  EXPECT_FALSE(rt.GetPrep(K(99), 0, 10, &h));
  EXPECT_EQ(rt.Misses(), 1u);
}

TEST(RamTier, SharedRangeContract) {
  FlushSink sink;
  RamTier rt(Opts(64 * 4096), sink.fn());
  const std::string v = "0123456789";
  ASSERT_TRUE(rt.Put(K(3), v.data(), v.size()));
  RamTier::Hit h;

  ASSERT_TRUE(rt.GetPrep(K(3), 3, 0, &h));
  EXPECT_EQ(std::string(h.ptr, h.len), "3456789");
  EXPECT_EQ(h.value_len, v.size());
  h = RamTier::Hit{};

  ASSERT_TRUE(rt.GetPrep(K(3), v.size(), 4, &h));
  EXPECT_EQ(h.len, 0u);
  EXPECT_EQ(h.value_len, v.size());
  h = RamTier::Hit{};

  EXPECT_FALSE(rt.GetPrep(K(3), v.size() + 1, 4, &h));

  ASSERT_TRUE(rt.GetPrep(K(3), 3, 100, &h));
  EXPECT_EQ(std::string(h.ptr, h.len), "3456789");
  EXPECT_EQ(h.value_len, v.size());
  h = RamTier::Hit{};
}

TEST(RamTier, SendPinBlocksEvictionUntilRelease) {
  FlushSink sink;
  RamTier rt(Opts(2 * 4096, 4096), sink.fn());  // exactly 2 slots
  std::string v(4096, 'x');
  ASSERT_TRUE(rt.Put(K(10), v.data(), v.size()));
  ASSERT_TRUE(rt.Put(K(11), v.data(), v.size()));
  EXPECT_TRUE(WaitFor([&] { return rt.Flushed() == 2u; }));  // both durable

  RamTier::Hit h;
  ASSERT_TRUE(rt.GetPrep(K(10), 0, v.size(), &h));  // send-pin on K10
  // Putting a 3rd must evict the *unpinned* durable one (K11), never K10.
  ASSERT_TRUE(rt.Put(K(12), v.data(), v.size()));
  EXPECT_TRUE(WaitFor([&] { return rt.Flushed() == 3u; }));
  EXPECT_TRUE(rt.Contains(K(10))) << "send-pinned slot must survive";
  EXPECT_TRUE(rt.Contains(K(12)));
  EXPECT_FALSE(rt.Contains(K(11)));
  EXPECT_GE(rt.Evictions(), 1u);
  h = RamTier::Hit{};
}

TEST(RamTier, SendPinTokenSurvivesResidentIndexRehash) {
  FlushSink sink;
  RamTier rt(Opts(32ull << 20), sink.fn());
  const std::string value(4096, 'r');
  const BlockKey pinned = K(1'000'000);
  ASSERT_TRUE(rt.PutDurable(pinned, value.data(), value.size()));

  RamTier::Hit hit;
  ASSERT_TRUE(rt.GetPrep(pinned, 0, value.size(), &hit));
  ASSERT_EQ(std::string(hit.ptr, hit.len), value);
  for (uint64_t id = 1; id <= 4096; ++id)
    ASSERT_TRUE(rt.PutDurable(K(id), value.data(), value.size()));

  EXPECT_TRUE(rt.Contains(pinned));
  hit = RamTier::Hit{};
  EXPECT_TRUE(rt.Remove(pinned));
  EXPECT_FALSE(rt.Contains(pinned));
}

TEST(RamTier, BackpressureBypassWhenFlushStalls) {
  FlushSink sink;
  sink.close();  // stall all flushes -> slots stay RAM_ONLY (flush-pinned)
  RamTier rt(Opts(2 * 4096, 4096), sink.fn());
  std::string v(4096, 'p');
  ASSERT_TRUE(rt.Put(K(20), v.data(), v.size()));
  ASSERT_TRUE(rt.Put(K(21), v.data(), v.size()));
  // Arena full of non-evictable slots -> the next Put is declined (bypass).
  EXPECT_FALSE(rt.Put(K(22), v.data(), v.size()));
  EXPECT_EQ(rt.PutBypass(), 1u);
  // Drain: opening the gate lets flushes land, freeing capacity again.
  sink.open();
  EXPECT_TRUE(WaitFor([&] { return rt.Flushed() == 2u; }));
  EXPECT_TRUE(rt.Put(K(22), v.data(), v.size()));
}

TEST(RamTier, FlushFailureRetriesThenDrops) {
  FlushSink sink;
  sink.fail = true;  // every flush attempt fails
  RamTier::Options o = Opts(8 * 4096);
  o.flush_retries = 3;
  RamTier rt(o, sink.fn());
  std::string v(500, 'f');
  ASSERT_TRUE(rt.Put(K(30), v.data(), v.size()));
  // After flush_retries failed attempts the entry is dropped from RAM.
  EXPECT_TRUE(WaitFor([&] { return rt.FlushDropped() == 1u; }));
  EXPECT_FALSE(rt.Contains(K(30)));
  {
    std::lock_guard<std::mutex> lk(sink.m);
    EXPECT_GE(sink.calls, 3);
  }
}
TEST(RamTier, InflightConflictingDuplicateKeepsLeaderPayloadAndCompletion) {
  FlushSink sink;
  sink.close();
  RamTier::Options options = Opts(8 * 4096);
  options.large_reserve_bytes = 0;
  RamTier rt(options, sink.fn());
  OpenFlushOnExit cleanup{sink};
  const std::string leader(500, 'a');
  // A duplicate larger than the entire tier must still join the first value.
  const std::string follower(options.bytes + 4096, 'b');
  RamTierTestPeer::Completion first, second;
  ASSERT_EQ(RamTierTestPeer::Admit(rt, K(31), leader, &first),
            RamTierTestPeer::Admission::kAccepted);
  ASSERT_TRUE(sink.await_calls(1));
  ASSERT_EQ(RamTierTestPeer::Admit(rt, K(31), follower, &second),
            RamTierTestPeer::Admission::kDuplicate);
  ASSERT_EQ(first, second);
  EXPECT_FALSE(RamTierTestPeer::Done(first));
  EXPECT_EQ(rt.PutBypass(), 0u);
  RamTier::Hit hit;
  ASSERT_TRUE(rt.GetPrep(K(31), 0, 0, &hit));
  EXPECT_EQ(std::string(hit.ptr, hit.len), leader);
  hit = RamTier::Hit{};

  sink.open();
  ASSERT_TRUE(RamTierTestPeer::AwaitDone(first));
  EXPECT_EQ(RamTierTestPeer::Wait(first), Status::kOk);
  EXPECT_EQ(RamTierTestPeer::Wait(second), Status::kOk);
  // Both public ACK modes preserve the leader after commit, too.
  EXPECT_EQ(rt.PutCommitted(K(31), follower.data(), follower.size()).status,
            Status::kOk);
  EXPECT_EQ(rt.PutWriteBack(K(31), follower.data(), follower.size()).status,
            Status::kOk);
  std::lock_guard<std::mutex> lock(sink.m);
  EXPECT_EQ(sink.calls, 1);
  EXPECT_EQ(sink.values.at(K(31).Filename()), leader);
}

TEST(RamTier, InflightDuplicateSharesTerminalLeaderFailure) {
  FlushSink sink;
  sink.close();
  sink.fail = true;
  RamTier::Options options = Opts(8 * 4096);
  options.large_reserve_bytes = 0;
  options.flush_retries = 1;
  RamTier rt(options, sink.fn());
  OpenFlushOnExit cleanup{sink};
  const std::string leader(500, 'a');
  const std::string follower(1000, 'b');
  RamTierTestPeer::Completion first, second;
  ASSERT_EQ(RamTierTestPeer::Admit(rt, K(32), leader, &first),
            RamTierTestPeer::Admission::kAccepted);
  ASSERT_TRUE(sink.await_calls(1));
  ASSERT_EQ(RamTierTestPeer::Admit(rt, K(32), follower, &second),
            RamTierTestPeer::Admission::kDuplicate);
  ASSERT_EQ(first, second);
  EXPECT_FALSE(RamTierTestPeer::Done(second));
  sink.open();
  ASSERT_TRUE(RamTierTestPeer::AwaitDone(first));
  EXPECT_EQ(RamTierTestPeer::Wait(first), Status::kIOError);
  EXPECT_EQ(RamTierTestPeer::Wait(second), Status::kIOError);
  EXPECT_EQ(rt.PutBypass(), 0u);
  EXPECT_FALSE(rt.healthy());
  EXPECT_FALSE(rt.Contains(K(32)));
}

// A PUT that waits for the flush of the value owning its key (here a
// disk-ACK duplicate) does so without the caller's admission lock and not
// for longer than put_wait_ms. After the deadline it is answered kCacheFull
// without a bypass (the value that owns the key is still there and still
// being flushed), and a later PUT sees the flush result.
TEST(RamTier, ParkedPutHasADeadlineAndReleasesTheCallersLock) {
  FlushSink sink;
  sink.close();
  RamTier::Options options = Opts(8 * 4096);
  options.large_reserve_bytes = 0;
  options.put_wait_ms = 50;
  RamTier rt(options, sink.fn());
  OpenFlushOnExit cleanup{sink};
  const std::string value(500, 'd'), different(700, 'e');
  ASSERT_TRUE(rt.Put(K(60), value.data(), value.size()));  // first writer
  ASSERT_TRUE(sink.await_calls(1));

  std::mutex admission;
  std::unique_lock<std::mutex> lock(admission);
  const auto t0 = std::chrono::steady_clock::now();
  RamTier::PutResult parked{Status::kInvalid};
  {
    OpenFlushAfter guard(sink, 5s);
    parked = rt.PutCommitted(K(60), different.data(), different.size(), &lock);
  }
  EXPECT_GE(std::chrono::steady_clock::now() - t0, 50ms);
  EXPECT_EQ(parked.status, Status::kCacheFull);
  EXPECT_EQ(parked.disposition, RamTier::PutDisposition::kComplete);
  EXPECT_EQ(parked.flush, RamTier::FlushOutcome::kWaitEnded);
  EXPECT_FALSE(lock.owns_lock());
  EXPECT_EQ(rt.PutWaitTimeouts(), 1u);
  EXPECT_EQ(rt.PutWaitsCanceled(), 0u);
  EXPECT_EQ(rt.ParkedPuts(), 0u);
  EXPECT_TRUE(rt.healthy());
  EXPECT_TRUE(rt.Contains(K(60)));

  sink.open();
  ASSERT_TRUE(WaitFor([&] { return rt.Flushed() == 1u; }));
  EXPECT_EQ(rt.PutCommitted(K(60), different.data(), different.size()).status,
            Status::kOk);
  std::lock_guard<std::mutex> sink_lock(sink.m);
  EXPECT_EQ(sink.values.at(K(60).Filename()), value);  // the first writer's
}

// The lock is released for the wait itself, not merely by the time the PUT
// returns; and the wait ends with the flush result, success or failure.
TEST(RamTier, ParkedPutReleasesItsLockWhileWaitingAndSeesTheFlushResult) {
  for (const bool fail : {false, true}) {
    FlushSink sink;
    sink.close();
    sink.fail = fail;
    RamTier::Options options = Opts(8 * 4096);
    options.large_reserve_bytes = 0;
    options.flush_retries = 1;
    options.put_wait_ms = 0;  // no deadline
    RamTier rt(options, sink.fn());
    OpenFlushOnExit cleanup{sink};
    const std::string value(500, 'f');

    // A disk-ACK first writer and a duplicate of it, each with the lock its
    // caller would hold.
    std::mutex admission;
    RamTier::PutResult results[2] = {{Status::kInvalid}, {Status::kInvalid}};
    std::vector<std::thread> writers;
    for (int i = 0; i < 2; ++i) {
      writers.emplace_back([&, i] {
        std::unique_lock<std::mutex> lock(admission);
        results[i] = rt.PutCommitted(K(61), value.data(), value.size(), &lock);
      });
    }
    ASSERT_TRUE(sink.await_calls(1));
    ASSERT_TRUE(WaitFor([&] { return rt.ParkedPuts() == 2u; }));
    {
      std::unique_lock<std::mutex> probe(admission, std::try_to_lock);
      EXPECT_TRUE(probe.owns_lock()) << "held across the wait";
    }
    sink.open();
    for (auto& writer : writers) writer.join();
    for (const auto& result : results) {
      EXPECT_EQ(result.status, fail ? Status::kIOError : Status::kOk) << fail;
      EXPECT_EQ(result.flush,
                fail ? RamTier::FlushOutcome::kDroppedAfterRetries
                     : RamTier::FlushOutcome::kFlushed)
          << fail;
    }
    EXPECT_EQ(rt.PutWaitTimeouts(), 0u);
  }
}

// CancelPutWaits (a stopping server) answers parked PUTs at once, and every
// PUT that would have to wait from then on; flushing goes on regardless, and
// an answer that needs no wait is still given.
TEST(RamTier, CancelPutWaitsAnswersParkedPutsAndLeavesTheFlushAlone) {
  FlushSink sink;
  sink.close();
  RamTier::Options options = Opts(8 * 4096);
  options.large_reserve_bytes = 0;
  options.put_wait_ms = 0;  // no deadline: only the cancel can end the wait
  RamTier rt(options, sink.fn());
  OpenFlushOnExit cleanup{sink};
  const std::string value(500, 'g');
  // The owner was not acknowledged from RAM, so a RAM-ACK duplicate has to
  // wait for it as well.
  ASSERT_TRUE(rt.Put(K(62), value.data(), value.size()));
  ASSERT_TRUE(sink.await_calls(1));

  RamTier::PutResult write_back{Status::kInvalid};
  RamTier::PutResult committed{Status::kInvalid};
  std::thread first([&] {
    write_back = rt.PutWriteBack(K(62), value.data(), value.size());
  });
  std::thread second([&] {
    committed = rt.PutCommitted(K(62), value.data(), value.size());
  });
  ASSERT_TRUE(WaitFor([&] { return rt.ParkedPuts() == 2u; }));
  {
    OpenFlushAfter guard(sink, 5s);
    rt.CancelPutWaits();
    first.join();
    second.join();
  }
  for (const auto& result : {write_back, committed}) {
    EXPECT_EQ(result.status, Status::kCacheFull);
    EXPECT_EQ(result.disposition, RamTier::PutDisposition::kComplete);
    EXPECT_EQ(result.flush, RamTier::FlushOutcome::kWaitEnded);
  }
  EXPECT_EQ(rt.PutWaitsCanceled(), 2u);
  EXPECT_EQ(rt.PutWaitTimeouts(), 0u);
  EXPECT_EQ(rt.PutCommitted(K(62), value.data(), value.size()).status,
            Status::kCacheFull);  // no parking once the stop has begun
  EXPECT_EQ(rt.PutWaitsCanceled(), 3u);
  // RAM can still acknowledge: a new key, and a duplicate of it.
  for (int i = 0; i < 2; ++i)
    EXPECT_EQ(rt.PutWriteBack(K(63), value.data(), value.size()).status,
              Status::kOk)
        << i;
  EXPECT_EQ(rt.PutWaitsCanceled(), 3u);

  sink.open();
  ASSERT_TRUE(WaitFor([&] { return rt.Flushed() == 2u; }));
  EXPECT_TRUE(rt.healthy());
  // On disk now: answered from the flush result without any wait.
  EXPECT_EQ(rt.PutCommitted(K(62), value.data(), value.size()).status,
            Status::kOk);
}

// The PUTs of one request batch share one wait budget: put_wait_ms counts
// from the batch's arrival, so a PUT that comes after an earlier one of its
// batch has waited only gets what is left, and none once it is used up. A
// wait whose time is up counts as a timeout even if the stop has begun.
TEST(RamTier, ParkedPutsOfOneBatchShareTheWaitFromItsArrival) {
  FlushSink sink;
  sink.close();
  RamTier::Options options = Opts(8 * 4096);
  options.large_reserve_bytes = 0;
  options.put_wait_ms = 400;
  RamTier rt(options, sink.fn());
  OpenFlushOnExit cleanup{sink};
  const std::string value(500, 'h');
  ASSERT_TRUE(rt.Put(K(63), value.data(), value.size()));  // first writer
  ASSERT_TRUE(sink.await_calls(1));
  OpenFlushAfter guard(sink, 5s);

  // 300 ms of the batch's 400 have gone to PUTs before this one.
  auto t0 = std::chrono::steady_clock::now();
  RamTier::PutResult result = rt.PutCommitted(
      K(63), value.data(), value.size(), nullptr, t0 - 300ms);
  auto waited = std::chrono::steady_clock::now() - t0;
  EXPECT_EQ(result.status, Status::kCacheFull);
  EXPECT_EQ(result.flush, RamTier::FlushOutcome::kWaitEnded);
  EXPECT_GE(waited, 90ms);
  EXPECT_LT(waited, 300ms) << "the PUT got a budget of its own";
  EXPECT_EQ(rt.PutWaitTimeouts(), 1u);

  // The batch's budget is used up: answered at once, without parking.
  t0 = std::chrono::steady_clock::now();
  result = rt.PutWriteBack(K(63), value.data(), value.size(), nullptr,
                           t0 - 500ms);
  waited = std::chrono::steady_clock::now() - t0;
  EXPECT_EQ(result.status, Status::kCacheFull);
  EXPECT_EQ(result.flush, RamTier::FlushOutcome::kWaitEnded);
  EXPECT_LT(waited, 100ms);
  EXPECT_EQ(rt.PutWaitTimeouts(), 2u);

  // Out of time and stopping: a timeout, not a cancel.
  rt.CancelPutWaits();
  result = rt.PutCommitted(K(63), value.data(), value.size(), nullptr,
                           std::chrono::steady_clock::now() - 500ms);
  EXPECT_EQ(result.status, Status::kCacheFull);
  EXPECT_EQ(rt.PutWaitTimeouts(), 3u);
  EXPECT_EQ(rt.PutWaitsCanceled(), 0u);
  EXPECT_EQ(rt.ParkedPuts(), 0u);
}

TEST(RamTier, RemoveCancelsQueuedGenerationBeforeReinsert) {
  FlushSink sink;
  sink.close();
  RamTier::Options options = Opts(8 * 4096);
  options.large_reserve_bytes = 0;
  RamTier rt(options, sink.fn());
  OpenFlushOnExit cleanup{sink};
  const std::string old_value(500, 'q');
  const std::string new_value(700, 'n');
  ASSERT_TRUE(rt.Put(K(33), old_value.data(), old_value.size()));
  ASSERT_TRUE(sink.await_calls(1));  // sole flusher owns the blocker

  RamTierTestPeer::Completion old, duplicate, replacement;
  ASSERT_EQ(RamTierTestPeer::Admit(rt, K(34), old_value, &old),
            RamTierTestPeer::Admission::kAccepted);
  ASSERT_EQ(RamTierTestPeer::Admit(rt, K(34), new_value, &duplicate),
            RamTierTestPeer::Admission::kDuplicate);
  ASSERT_EQ(old, duplicate);
  ASSERT_TRUE(rt.Remove(K(34)));  // queued, so cancellation is synchronous
  EXPECT_EQ(RamTierTestPeer::Wait(old), Status::kIOError);
  EXPECT_EQ(RamTierTestPeer::Wait(duplicate), Status::kIOError);
  EXPECT_FALSE(rt.Contains(K(34)));
  ASSERT_EQ(RamTierTestPeer::Admit(rt, K(34), new_value, &replacement),
            RamTierTestPeer::Admission::kAccepted);
  EXPECT_NE(old, replacement);
  EXPECT_FALSE(RamTierTestPeer::Done(replacement));
  sink.open();
  ASSERT_TRUE(RamTierTestPeer::AwaitDone(replacement));
  EXPECT_EQ(RamTierTestPeer::Wait(replacement), Status::kOk);
  EXPECT_EQ(RamTierTestPeer::Wait(old), Status::kIOError);
  std::lock_guard<std::mutex> lock(sink.m);
  EXPECT_EQ(sink.calls, 2);
  EXPECT_EQ(sink.values.at(K(34).Filename()), new_value);
}

TEST(RamTier, RemovedPinnedGenerationRejectsRatherThanBypassing) {
  FlushSink sink;
  RamTier::Options options = Opts(8 * 4096);
  options.large_reserve_bytes = 0;
  RamTier rt(options, sink.fn());
  const std::string first(500, 'a');
  const std::string conflicting(options.bytes + 4096, 'b');
  ASSERT_TRUE(rt.PutDurable(K(35), first.data(), first.size()));
  RamTier::Hit pinned;
  ASSERT_TRUE(rt.GetPrep(K(35), 0, 0, &pinned));
  ASSERT_TRUE(rt.Remove(K(35)));
  EXPECT_FALSE(rt.Contains(K(35)));
  // Physical ownership has not retired: neither payload size nor ACK mode
  // permits a direct-disk fallback while the old generation is still pinned.
  EXPECT_EQ(rt.PutCommitted(K(35), conflicting.data(), conflicting.size()).status,
            Status::kIOError);
  EXPECT_EQ(rt.PutWriteBack(K(35), conflicting.data(), conflicting.size()).status,
            Status::kIOError);
  EXPECT_EQ(rt.PutBypass(), 0u);
  EXPECT_EQ(std::string(pinned.ptr, pinned.len), first);
  pinned = RamTier::Hit{};
  EXPECT_EQ(rt.UsedBytes(), 0u);
  const std::string replacement(700, 'n');
  EXPECT_EQ(rt.PutCommitted(K(35), replacement.data(), replacement.size()).status,
            Status::kOk);
  ASSERT_TRUE(rt.GetPrep(K(35), 0, 0, &pinned));
  EXPECT_EQ(std::string(pinned.ptr, pinned.len), replacement);
}

// A PUT of a key whose cold read holds a hidden reservation does not join
// it (only the reading thread can finish it, and that may be the PUT's own
// thread): it goes to the disk, which holds the key. It used to wait for
// the reservation and share its outcome.
TEST(RamTier, DuplicateOfHiddenReservationGoesToTheDisk) {
  FlushSink sink;
  RamTier::Options options = Opts(8 * 4096);
  options.large_reserve_bytes = 0;
  RamTier rt(options, sink.fn());
  RamTier::DurableReservation reservation;
  ASSERT_TRUE(rt.ReserveDurable(K(36), 4096, &reservation));
  const std::string duplicate(1000, 'd');
  RamTierTestPeer::Completion completion;
  ASSERT_EQ(RamTierTestPeer::Admit(rt, K(36), duplicate, &completion),
            RamTierTestPeer::Admission::kBypass);
  EXPECT_FALSE(completion) << "nothing to wait for";
  EXPECT_FALSE(rt.Contains(K(36)));
  reservation = RamTier::DurableReservation{};
  EXPECT_FALSE(rt.Contains(K(36)));
  EXPECT_EQ(rt.UsedBytes(), 0u);
  EXPECT_EQ(rt.PutBypass(), 0u) << "not an arena-pressure bypass";
}

TEST(RamTier, RemoveHidesActiveFlushAndDefersPinnedRelease) {
  FlushSink sink;
  sink.close();
  RamTier rt(Opts(8 * 4096), sink.fn());
  std::string v(200, 'r');
  ASSERT_TRUE(rt.Put(K(40), v.data(), v.size()));
  EXPECT_TRUE(WaitFor([&] {
    std::lock_guard<std::mutex> lk(sink.m);
    return sink.calls == 1;
  }));

  RamTier::Hit h;
  ASSERT_TRUE(rt.GetPrep(K(40), 0, v.size(), &h));
  std::atomic<bool> removed{false};
  std::thread remover([&] {
    removed.store(rt.Remove(K(40)), std::memory_order_release);
  });
  EXPECT_TRUE(WaitFor([&] { return !rt.Contains(K(40)); }));
  EXPECT_FALSE(removed.load(std::memory_order_acquire));
  EXPECT_EQ(std::string(h.ptr, h.len), v);
  sink.open();
  remover.join();
  EXPECT_TRUE(removed.load(std::memory_order_acquire));
  EXPECT_FALSE(rt.Contains(K(40)));
  h = RamTier::Hit{};
  EXPECT_EQ(rt.Count(), 0u);
}

// Regression: Remove() racing an in-flight PUT must not resurrect the key.
// Admit used to read completion->canceled in one critical section and erase
// writing + install the index entry in another; a Remove landing in between
// flagged the canceled bit too late, the entry was installed untombstoned,
// and the key kept answering GETs after Remove returned true (the disk copy
// is deleted by the caller on remove, so the tier served a deleted value).
// Here a pack of hammer threads keeps the single shard mutex saturated so
// the writer blocks between that read and the install, opening the race
// window; rounds where Remove observed the key must all end with the key
// invisible.
TEST(RamTier, RemoveRacingInFlightPutNeverResurrects) {
  ::setenv("DFKV_RAM_TIER_EXTENT_BYTES", "1048576", 1);
  ::setenv("DFKV_RAM_TIER_SHARDS", "1", 1);
  FlushSink sink;  // gate open: installs flush (and complete) promptly
  RamTier rt(Opts(64ull << 20), sink.fn());
  ASSERT_TRUE(rt.ok());
  const std::string v(1ull << 20, 'v');  // in-arena (== one extent), slow copy

  std::atomic<bool> stop_hammer{false};
  std::vector<std::thread> hammers;
  for (int i = 0; i < 8; ++i) {
    hammers.emplace_back([&, i] {
      // Never admitted; pure lock pressure on the single shard mutex so the
      // writer's read->install transition contends with Remove's phase one.
      while (!stop_hammer.load(std::memory_order_relaxed))
        (void)rt.Contains(K(0xFFF00000u + i));
    });
  }

  int observed = 0;       // rounds where Remove found the key tracked
  int resurrections = 0;  // observed rounds where the key survived Remove
  for (uint64_t r = 0; r < 600; ++r) {
    std::atomic<bool> started{false};
    std::thread writer([&] {
      started.store(true, std::memory_order_release);
      rt.Put(K(r), v.data(), v.size());
    });
    while (!started.load(std::memory_order_acquire))
      std::this_thread::yield();
    // Sweep across the writer's copy window (~100+ us for 1 MiB under
    // contention) so many rounds land mid-flight.
    std::this_thread::sleep_for(std::chrono::microseconds(10 + (r % 28) * 5));
    const bool removed = rt.Remove(K(r));
    writer.join();
    if (removed) {
      ++observed;
      if (rt.Contains(K(r))) ++resurrections;
      RamTier::Hit h;
      if (rt.GetPrep(K(r), 0, 0, &h)) ++resurrections;
    }
    rt.Remove(K(r));  // cleanup: writer may have missed the race window
  }
  stop_hammer.store(true, std::memory_order_relaxed);
  for (auto& t : hammers) t.join();

  EXPECT_GE(observed, 3);  // prove the race regime was actually entered
  EXPECT_EQ(resurrections, 0);
}

// Regression: same cancel/install race through PutDurable (read-promotion),
// which installed the entry without ever consulting completion->canceled. The
// promotion's setup window spans a large posix_memalign + memcpy, so a Remove
// started inside it deterministically flags the shared completion; before the
// fix the entry was still installed (and completed) untombstoned.
TEST(RamTier, RemoveRacingInFlightPutDurableNeverResurrects) {
  ::setenv("DFKV_RAM_TIER_EXTENT_BYTES", "1048576", 1);
  ::setenv("DFKV_RAM_TIER_SHARDS", "1", 1);
  FlushSink sink;  // unused: promoted entries are born durable, never flushed
  RamTier::Options o = Opts(96ull << 20);
  o.large_reserve_bytes = 48ull << 20;  // fit the 8 MiB dedicated values
  RamTier rt(o, sink.fn());
  ASSERT_TRUE(rt.ok());
  const std::string v(8ull << 20, 'p');  // dedicated: ms-wide install window

  int observed = 0;
  int resurrections = 0;
  for (uint64_t r = 0; r < 40; ++r) {
    std::atomic<bool> started{false};
    std::thread writer([&] {
      started.store(true, std::memory_order_release);
      rt.PutDurable(K(r), v.data(), v.size());
    });
    while (!started.load(std::memory_order_acquire))
      std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::microseconds(100 + (r % 9) * 100));
    const bool removed = rt.Remove(K(r));
    writer.join();
    if (removed) {
      ++observed;
      if (rt.Contains(K(r))) ++resurrections;
      RamTier::Hit h;
      if (rt.GetPrep(K(r), 0, 0, &h)) ++resurrections;
    }
    rt.Remove(K(r));  // cleanup
  }

  EXPECT_GE(observed, 3);  // prove the race regime was actually entered
  EXPECT_EQ(resurrections, 0);
}

TEST(RamTier, LargeValueReadRangeFlushAndRemoveIsByteExact) {
  ::setenv("DFKV_RAM_TIER_EXTENT_BYTES", "1048576", 1);
  std::mutex m;
  std::string flushed;
  size_t flushed_len = 0;
  size_t flushed_cap = 0;
  RamTier rt(Opts(4ull << 20),
             [&](const BlockKey&, char* data, size_t len, size_t cap) {
               std::lock_guard<std::mutex> lk(m);
               flushed.assign(data, len);
               flushed_len = len;
               flushed_cap = cap;
               return Status::kOk;
             });
  ::unsetenv("DFKV_RAM_TIER_EXTENT_BYTES");
  ASSERT_TRUE(rt.ok());
  EXPECT_EQ(rt.arena_bytes() + rt.large_budget_bytes(), 4ull << 20);
  EXPECT_EQ(rt.large_budget_bytes(), 2ull << 20);
  EXPECT_LE(rt.arena_bytes(), rt.budget_bytes());
  int fake_mr = 0;
  rt.SetArenaMr(&fake_mr);

  std::string value((1ull << 20) + 137, '\0');
  for (size_t i = 0; i < value.size(); ++i)
    value[i] = static_cast<char>((i * 131 + 17) & 0xff);
  ASSERT_TRUE(rt.Put(K(50), value.data(), value.size()));
  EXPECT_EQ(rt.UsedBytes(), ((value.size() + 4095) / 4096) * 4096);
  EXPECT_EQ(rt.large_used_bytes(), rt.UsedBytes());
  EXPECT_LE(rt.arena_bytes() + rt.large_used_bytes(), rt.budget_bytes());

  RamTier::Hit whole;
  ASSERT_TRUE(rt.GetPrep(K(50), 0, 0, &whole));
  EXPECT_FALSE(whole.in_arena);
  EXPECT_EQ(whole.mr, nullptr);
  EXPECT_EQ(whole.value_len, value.size());
  EXPECT_EQ(std::string(whole.ptr, whole.len), value);
  whole = RamTier::Hit{};

  RamTier::Hit range;
  ASSERT_TRUE(rt.GetPrep(K(50), (1ull << 20) - 19, 91, &range));
  EXPECT_FALSE(range.in_arena);
  EXPECT_EQ(std::string(range.ptr, range.len),
            value.substr((1ull << 20) - 19, 91));
  range = RamTier::Hit{};

  ASSERT_TRUE(WaitFor([&] { return rt.Flushed() == 1; }));
  {
    std::lock_guard<std::mutex> lk(m);
    EXPECT_EQ(flushed_len, value.size());
    EXPECT_GE(flushed_cap, flushed_len);
    EXPECT_EQ(flushed, value);
  }
  EXPECT_TRUE(rt.Remove(K(50)));
  EXPECT_FALSE(rt.Contains(K(50)));
  EXPECT_EQ(rt.UsedBytes(), 0u);
  EXPECT_EQ(rt.large_used_bytes(), 0u);
}

TEST(RamTier, LargeValueSendPinDefersPhysicalRemovalAndCapacityRelease) {
  ::setenv("DFKV_RAM_TIER_EXTENT_BYTES", "1048576", 1);
  FlushSink sink;
  RamTier rt(Opts(2ull << 20), sink.fn());
  ::unsetenv("DFKV_RAM_TIER_EXTENT_BYTES");
  const std::string first((1ull << 20) + 1, 'a');
  const std::string second((1ull << 20) + 1, 'b');
  ASSERT_TRUE(rt.Put(K(51), first.data(), first.size()));
  ASSERT_TRUE(WaitFor([&] { return rt.Flushed() == 1; }));

  RamTier::Hit pinned;
  ASSERT_TRUE(rt.GetPrep(K(51), 0, 0, &pinned));
  ASSERT_FALSE(pinned.in_arena);
  EXPECT_TRUE(rt.Remove(K(51)));
  EXPECT_FALSE(rt.Contains(K(51)));
  EXPECT_FALSE(rt.Put(K(52), second.data(), second.size()))
      << "pinned durable large value must not be reclaimed";
  EXPECT_LE(rt.UsedBytes(), 2ull << 20);

  EXPECT_EQ(std::string(pinned.ptr, pinned.len), first)
      << "logical removal must not invalidate an in-flight pointer";
  pinned = RamTier::Hit{};
  EXPECT_TRUE(rt.Put(K(52), second.data(), second.size()))
      << "final send release must make the physical capacity reusable";
  EXPECT_FALSE(rt.Contains(K(51)));
  EXPECT_TRUE(rt.Contains(K(52)));
  EXPECT_LE(rt.UsedBytes(), 2ull << 20);
}

TEST(RamTier, LargeValueFlushFailureDropsAndReleasesBudget) {
  ::setenv("DFKV_RAM_TIER_EXTENT_BYTES", "1048576", 1);
  FlushSink sink;
  sink.close();
  RamTier::Options o = Opts(2ull << 20);
  o.flush_retries = 1;
  RamTier rt(o, sink.fn());
  ::unsetenv("DFKV_RAM_TIER_EXTENT_BYTES");
  const std::string value((1ull << 20) + 1, 'f');
  ASSERT_TRUE(rt.Put(K(53), value.data(), value.size()));
  RamTier::Hit pinned;
  ASSERT_TRUE(rt.GetPrep(K(53), 0, 0, &pinned));
  ASSERT_FALSE(pinned.in_arena);
  EXPECT_FALSE(rt.Put(K(54), value.data(), value.size()))
      << "non-durable large allocation applies byte-budget backpressure";
  EXPECT_EQ(rt.PutBypass(), 1u);

  {
    std::lock_guard<std::mutex> lk(sink.m);
    sink.fail = true;
  }
  sink.open();
  ASSERT_TRUE(WaitFor([&] { return rt.FlushDropped() == 1; }));
  EXPECT_FALSE(rt.Contains(K(53)));
  EXPECT_GT(rt.UsedBytes(), 0u)
      << "remove-pending allocation stays owned while a reader holds it";
  EXPECT_EQ(std::string(pinned.ptr, pinned.len), value);
  pinned = RamTier::Hit{};
  EXPECT_EQ(rt.UsedBytes(), 0u);

  {
    std::lock_guard<std::mutex> lk(sink.m);
    sink.fail = false;
  }
  EXPECT_TRUE(rt.Put(K(54), value.data(), value.size()));
  EXPECT_LE(rt.UsedBytes(), 2ull << 20);
}

TEST(RamTier, ValueAboveTotalBudgetBypassesWithoutAccountingLeak) {
  ::setenv("DFKV_RAM_TIER_EXTENT_BYTES", "1048576", 1);
  FlushSink sink;
  RamTier rt(Opts(2ull << 20), sink.fn());
  ::unsetenv("DFKV_RAM_TIER_EXTENT_BYTES");
  const std::string too_large((2ull << 20) + 1, 'x');
  EXPECT_FALSE(rt.Put(K(55), too_large.data(), too_large.size()));
  EXPECT_EQ(rt.PutBypass(), 1u);
  EXPECT_EQ(rt.UsedBytes(), 0u);
  EXPECT_EQ(rt.Count(), 0u);
}

TEST(RamTier, ConcurrentPutGetReleaseIsRaceFree) {
  FlushSink sink;
  RamTier rt(Opts(512 * 4096), sink.fn());
  constexpr int T = 8, N = 1500;
  std::atomic<int> hits{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < T; ++t) {
    ts.emplace_back([&, t] {
      std::string v(300, 'c');
      for (int i = 0; i < N; ++i) {
        BlockKey k = K((t * N + i) % 3000);
        rt.Put(k, v.data(), v.size());
        RamTier::Hit h;
        if (rt.GetPrep(k, 0, v.size(), &h)) {
          hits.fetch_add(1);
          h = RamTier::Hit{};
        }
      }
    });
  }
  for (auto& th : ts) th.join();
  EXPECT_GT(hits.load(), 0);
}

// Background reclaimer wiring: once the arena is full of DURABLE slots and new
// writes keep coming, the reclaimer (not the Put path) pre-frees slots, so
// admission rides pop-free-slot and never hits backpressure bypass.
TEST(RamTier, BackgroundReclaimerPreFreesDurableSlots) {
  FlushSink sink;
  auto o = Opts(16 * 4096);  // 16 slots, one extent (pool exhausted at first bind)
  o.reclaim_interval_ms = 1;
  RamTier rt(o, sink.fn());
  ASSERT_TRUE(rt.ok());
  std::string v(4000, 'r');
  for (uint64_t i = 0; i < 16; ++i) ASSERT_TRUE(rt.Put(K(100 + i), v.data(), v.size()));
  ASSERT_TRUE(WaitFor([&] { return rt.FlushBacklog() == 0; }));  // all durable
  // Sustained new-key writes: demand makes the reclaimer top up free slots.
  for (uint64_t i = 0; i < 32; ++i) {
    EXPECT_TRUE(rt.Put(K(200 + i), v.data(), v.size()));
    std::this_thread::sleep_for(1ms);
  }
  EXPECT_TRUE(WaitFor([&] { return rt.Reclaimed() > 0; }))
      << "reclaimer never pre-freed a slot";
  EXPECT_EQ(rt.PutBypass(), 0u);
}

// Class rebalance in the RAM tier: arena full of an idle class, then a burst of
// a NEW size larger than its first extents -- the tick must move extents from
// the idle donor so the burst stays resident (stock retained ~one extent).
TEST(RamTier, ReclaimTickGrowsHotClassFromColdDonor) {
  FlushSink sink;
  RamTier::Options o;
  o.bytes = 16ull << 20;  // 16 MiB arena
  o.large_reserve_bytes = 0;  // isolate fixed-slot rebalance capacity
  o.slot_granularity = 4096;
  o.reclaim_interval_ms = 1;
  setenv("DFKV_RAM_TIER_EXTENT_BYTES", "1048576", 1);  // 16 extents x 1 MiB
  RamTier rt(o, sink.fn());
  unsetenv("DFKV_RAM_TIER_EXTENT_BYTES");
  ASSERT_TRUE(rt.ok());
  std::string a(4000, 'a');
  for (uint64_t i = 0; i < 4096; ++i) ASSERT_TRUE(rt.Put(K(30000 + i), a.data(), a.size()));
  ASSERT_TRUE(WaitFor([&] { return rt.FlushBacklog() == 0; }));  // donor durable
  std::string b(16000, 'b');
  for (uint64_t i = 0; i < 256; ++i) {  // 4 extents' worth of a new class
    ASSERT_TRUE(rt.Put(K(40000 + i), b.data(), b.size()));
    std::this_thread::sleep_for(1ms);
  }
  ASSERT_TRUE(WaitFor([&] { return rt.FlushBacklog() == 0; }));
  size_t resident = 0;
  for (uint64_t i = 0; i < 256; ++i) resident += rt.Contains(K(40000 + i));
  EXPECT_GT(rt.Rebalanced(), 0u) << "growth never kicked in";
  EXPECT_GE(resident, 250u) << "hot class must absorb capacity, not eat itself";
}

// Batched flush sink: a worker drains multiple queued items into ONE batch
// call; per-item retry/drop semantics survive partial batch failure.
TEST(RamTier, BatchFlushDrainsQueueAndRetriesPerItem) {
  std::mutex m;
  std::vector<size_t> batch_sizes;
  std::atomic<int> fail_key{-1};
  std::mutex gate_mu;
  std::condition_variable gate_cv;
  bool gate_open = false;
  RamTier::Options o = Opts(256 * 4096);
  o.flush_threads = 1;  // single worker => deterministic batching
  RamTier rt(o, nullptr);
  rt.set_flush_batch([&](const std::vector<RamTier::FlushItem>& items) {
    {
      std::unique_lock<std::mutex> lk(gate_mu);
      gate_cv.wait(lk, [&] { return gate_open; });
    }
    std::lock_guard<std::mutex> lk(m);
    batch_sizes.push_back(items.size());
    std::vector<Status> ok(items.size(), Status::kOk);
    for (size_t i = 0; i < items.size(); ++i)
      if (fail_key >= 0 &&
          items[i].key.digest_hi ==
              static_cast<uint64_t>(fail_key.load()))
        ok[i] = Status::kIOError;
    return ok;
  });
  ASSERT_TRUE(rt.ok());
  std::string v(4000, 'b');
  bool queued = true;
  for (uint64_t i = 0; i < 40; ++i)
    queued = rt.Put(K(700 + i), v.data(), v.size()) && queued;
  {
    std::lock_guard<std::mutex> lk(gate_mu);
    gate_open = true;
  }
  gate_cv.notify_one();
  ASSERT_TRUE(queued);
  ASSERT_TRUE(WaitFor([&] { return rt.Flushed() == 40u; }));
  {
    std::lock_guard<std::mutex> lk(m);
    size_t mx = 0; for (size_t b : batch_sizes) mx = std::max(mx, b);
    EXPECT_GT(mx, 1u) << "worker never batched";
  }
  // Partial failure: one key keeps failing -> retried then dropped; others flush.
  fail_key = 900;
  ASSERT_TRUE(rt.Put(K(900), v.data(), v.size()));
  ASSERT_TRUE(rt.Put(K(901), v.data(), v.size()));
  ASSERT_TRUE(WaitFor([&] { return rt.FlushDropped() == 1u; }));
  EXPECT_TRUE(WaitFor([&] { return rt.Flushed() >= 41u; }));
  EXPECT_FALSE(rt.Contains(K(900))) << "dropped after retries";
  EXPECT_TRUE(rt.Contains(K(901)));
}


// Sharded tier: keys route by hash to independent shards; the whole lifecycle
// (put -> visible -> flush -> durable -> get/pin/release -> remove) must hold
// with shards > 1 exactly as with the single-lock layout.
TEST(RamTier, ShardedLifecycleAcrossShards) {
  ::setenv("DFKV_RAM_TIER_EXTENT_BYTES", "1048576", 1);  // 1 MiB extents
  ::setenv("DFKV_RAM_TIER_SHARDS", "8", 1);
  FlushSink sink;  // gate open by default
  RamTier::Options o;
  o.bytes = 256ull << 20;  // 256 extents -> 32/shard, keeps 8 shards
  o.large_reserve_bytes = 0;  // this test exercises arena sharding only
  o.slot_granularity = 4096;
  o.flush_threads = 8;
  RamTier t(o, sink.fn());
  ASSERT_TRUE(t.ok());
  EXPECT_EQ(t.shards(), 8u);
  EXPECT_EQ(t.flusher_count(), 8u);

  const int N = 512;  // hash spreads these across all shards
  std::string v(8000, 'x');
  for (int i = 0; i < N; ++i) ASSERT_TRUE(t.Put(K(1000 + i), v.data(), v.size())) << i;
  EXPECT_EQ(t.Count(), static_cast<size_t>(N));
  // Wait for every shard's flushers to drain.
  for (int spin = 0; spin < 500 && t.Flushed() < static_cast<uint64_t>(N); ++spin)
    std::this_thread::sleep_for(10ms);
  EXPECT_EQ(t.Flushed(), static_cast<uint64_t>(N));
  EXPECT_EQ(t.FlushBacklog(), 0u);

  // Every key readable; tokens (shard-encoded) release correctly.
  for (int i = 0; i < N; ++i) {
    RamTier::Hit h;
    ASSERT_TRUE(t.GetPrep(K(1000 + i), 0, 0, &h)) << i;
    EXPECT_EQ(h.len, v.size());
    EXPECT_EQ(std::string(h.ptr, 16), v.substr(0, 16));
    h = RamTier::Hit{};
  }
  // Durable + idle -> removable, on whichever shard the key lives.
  for (int i = 0; i < N; i += 7) EXPECT_TRUE(t.Remove(K(1000 + i))) << i;
  ::unsetenv("DFKV_RAM_TIER_SHARDS");
  ::unsetenv("DFKV_RAM_TIER_EXTENT_BYTES");
}

TEST(RamTier, RejectsUnsupportedNumaMode) {
  ::setenv("DFKV_RAM_TIER_NUMA", "2", 1);
  FlushSink sink;
  RamTier rt(Opts(8 * 4096), sink.fn());
  EXPECT_FALSE(rt.ok());
  ::unsetenv("DFKV_RAM_TIER_NUMA");
}

// Locking the arena is best effort and can be turned off; either way the tier
// comes up.
TEST(RamTier, ArenaLockIsBestEffortAndCanBeTurnedOff) {
  FlushSink sink;
  {
    ::setenv("DFKV_RAM_TIER_LOCK", "off", 1);
    RamTier rt(Opts(8 * 4096), sink.fn());
    EXPECT_TRUE(rt.ok());
    EXPECT_FALSE(rt.arena_locked());
  }
  {
    ::setenv("DFKV_RAM_TIER_LOCK", "0", 1);
    RamTier rt(Opts(8 * 4096), sink.fn());
    EXPECT_TRUE(rt.ok());
    EXPECT_FALSE(rt.arena_locked());
  }
  {
    ::unsetenv("DFKV_RAM_TIER_LOCK");
    RamTier rt(Opts(8 * 4096), sink.fn());
    EXPECT_TRUE(rt.ok());  // locked where the memlock limit allows, fine if not
  }
  ::unsetenv("DFKV_RAM_TIER_LOCK");
}

// Huge pages are asked for, never required: the tier comes up with whatever
// the kernel gives, reports it, and can be told not to ask.
TEST(RamTier, ArenaHugePagesAreBestEffortAndCanBeTurnedOff) {
  FlushSink sink;
  const uint64_t kArena = 8ull << 20;  // four huge pages
  {
    ::unsetenv("DFKV_RAM_TIER_HUGEPAGES");
    RamTier rt(Opts(kArena), sink.fn());
    ASSERT_TRUE(rt.ok());
    EXPECT_TRUE(rt.arena_hugepages_requested());
    EXPECT_LE(rt.arena_hugepage_bytes(), rt.arena_bytes());
    // The arena starts on a huge page boundary, so none is cut in two.
    EXPECT_EQ(reinterpret_cast<uintptr_t>(RamTierTestPeer::ArenaBase(rt)) %
                  (2u << 20),
              0u);
    const std::string value(4096, 'h');
    ASSERT_TRUE(rt.Put(K(1), value.data(), value.size()));
    RamTier::Hit hit;
    ASSERT_TRUE(rt.GetPrep(K(1), 0, value.size(), &hit));
    EXPECT_EQ(std::string(hit.ptr, hit.len), value);
  }
  for (const char* off : {"off", "0"}) {
    ::setenv("DFKV_RAM_TIER_HUGEPAGES", off, 1);
    RamTier rt(Opts(kArena), sink.fn());
    EXPECT_TRUE(rt.ok());
    EXPECT_FALSE(rt.arena_hugepages_requested());
    EXPECT_EQ(rt.arena_hugepage_bytes(), 0u);
  }
  {
    // Smaller than one huge page: nothing to ask for.
    ::unsetenv("DFKV_RAM_TIER_HUGEPAGES");
    RamTier rt(Opts(8 * 4096), sink.fn());
    EXPECT_TRUE(rt.ok());
    EXPECT_FALSE(rt.arena_hugepages_requested());
  }
  ::unsetenv("DFKV_RAM_TIER_HUGEPAGES");
}

namespace {
// An anonymous mapping of `bytes` on a huge page boundary that asks for huge
// pages and has every page touched. The pages around it are mapped without
// the request, so the kernel cannot merge it with a neighbour.
struct HugeRegion {
  explicit HugeRegion(size_t bytes) : size(bytes), raw_size(bytes + (4u << 20)) {
    raw = ::mmap(nullptr, raw_size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) return;
    const uintptr_t aligned =
        (reinterpret_cast<uintptr_t>(raw) + 4096 + (2u << 20) - 1) &
        ~static_cast<uintptr_t>((2u << 20) - 1);
    base = reinterpret_cast<char*>(aligned);
#ifdef MADV_HUGEPAGE
    ::madvise(base, size, MADV_HUGEPAGE);
#endif
    for (size_t at = 0; at < size; at += 4096) base[at] = 1;
  }
  ~HugeRegion() {
    if (raw != MAP_FAILED) ::munmap(raw, raw_size);
  }
  size_t size;
  size_t raw_size;
  void* raw = MAP_FAILED;
  char* base = nullptr;
};
}  // namespace

// The figure is that of the range asked for: other huge-page-backed memory
// of the process, allocated before or after, is not in it, and a range that
// has been split into several mappings is their sum.
TEST(RamTier, HugePageBytesAreCountedInTheGivenRangeOnly) {
  constexpr size_t kRegion = 16u << 20;
  constexpr size_t kOther = 64u << 20;
  HugeRegion region(kRegion);
  ASSERT_NE(region.base, nullptr);
  const uint64_t alone = dfkv::AnonHugePageBytesIn(region.base, kRegion);
  if (alone == 0) GTEST_SKIP() << "no transparent huge pages on this system";
  EXPECT_LE(alone, kRegion);

  HugeRegion other(kOther);  // four times the region, on huge pages too
  ASSERT_NE(other.base, nullptr);
  const uint64_t beside = dfkv::AnonHugePageBytesIn(region.base, kRegion);
  EXPECT_GE(beside, alone);  // the kernel may have completed a huge page
  EXPECT_LE(beside, kRegion) << "memory outside the range was counted";

  // Two mappings now (the second half got another flag): still the sum.
  ASSERT_EQ(::madvise(region.base + kRegion / 2, kRegion / 2, MADV_DONTFORK), 0);
  const uint64_t split = dfkv::AnonHugePageBytesIn(region.base, kRegion);
  EXPECT_GE(split, beside);
  EXPECT_LE(split, kRegion);
  // A length that is not a whole number of pages reaches to the end of its
  // last page, as the kernel's own rounding of the length does: the second
  // mapping ends there and is still inside.
  EXPECT_GE(dfkv::AnonHugePageBytesIn(region.base, kRegion - 100), split);
  EXPECT_GE(dfkv::AnonHugePageBytesIn(region.base, kRegion - 4095), split);
  // Each half is a whole mapping and is counted for itself.
  const uint64_t first = dfkv::AnonHugePageBytesIn(region.base, kRegion / 2);
  const uint64_t second =
      dfkv::AnonHugePageBytesIn(region.base + kRegion / 2, kRegion / 2);
  EXPECT_LE(first, kRegion / 2);
  EXPECT_LE(second, kRegion / 2);
  EXPECT_GE(first + second, split);
  // A range that holds no whole mapping counts nothing.
  EXPECT_EQ(dfkv::AnonHugePageBytesIn(region.base + 4096, 8 * 4096), 0u);
}

// A start that is cancelled while the arena is being faulted in stops there:
// the constructor returns at once, not ready, and has started no thread.
TEST(RamTier, ACancelledStartGivesThePreFaultUp) {
  ::setenv("DFKV_RAM_TIER_LOCK", "off", 1);
  FlushSink sink;
  const uint64_t kArena = 256ull << 20;  // 128 steps of 2 MiB
  std::atomic<uint64_t> asked{0};
  {
    auto options = Opts(kArena);
    // The third step is the last one wanted.
    options.cancel_start = [&asked] { return asked.fetch_add(1) >= 3; };
    RamTier rt(options, sink.fn());
    EXPECT_FALSE(rt.ok());
    EXPECT_TRUE(rt.start_cancelled());
    // Every thread found out at its next step: a few questions, not 128.
    EXPECT_LT(asked.load(), 64u);
  }
  {
    auto options = Opts(8ull << 20);
    asked = 0;
    options.cancel_start = [&asked] {
      asked.fetch_add(1);
      return false;
    };
    RamTier rt(options, sink.fn());
    EXPECT_TRUE(rt.ok());
    EXPECT_FALSE(rt.start_cancelled());
    EXPECT_GE(asked.load(), 4u);  // asked once per 2 MiB, and never said yes
  }
  ::unsetenv("DFKV_RAM_TIER_LOCK");
}

// The arena's figure with other huge-page-backed memory appearing in the
// process while the arena is faulted in, as the disk opens now do.
TEST(RamTier, ArenaHugePageFigureIgnoresConcurrentAllocations) {
  ::unsetenv("DFKV_RAM_TIER_HUGEPAGES");
  FlushSink sink;
  std::atomic<bool> stop{false};
  std::thread other([&stop] {
    std::vector<std::unique_ptr<HugeRegion>> held;
    while (!stop.load()) {
      if (held.size() < 8)
        held.push_back(std::make_unique<HugeRegion>(32u << 20));
      else
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });
  const uint64_t kArena = 32ull << 20;
  RamTier rt(Opts(kArena), sink.fn());
  const uint64_t in_arena = dfkv::AnonHugePageBytesIn(
      RamTierTestPeer::ArenaBase(rt), rt.arena_bytes());
  stop.store(true);
  other.join();
  ASSERT_TRUE(rt.ok());
  if (in_arena == 0) GTEST_SKIP() << "no transparent huge pages on this system";
  EXPECT_LE(rt.arena_hugepage_bytes(), rt.arena_bytes());
  // Read again right after: nothing but the arena itself may be in it.
  EXPECT_LE(rt.arena_hugepage_bytes(), in_arena);
  EXPECT_GT(rt.arena_hugepage_bytes(), 0u);
}

TEST(RamTier, ShardCountCanExceedLegacyCapOf16) {
  // Phase 10: kTokenShardBits 4->6 lifts the cap 16->64 so read/write-heavy
  // multi-connection deployments can shard past the old ceiling. 32 shards must
  // be honored (not clamped to 16), and shard-encoded tokens still round-trip.
  // 1 MiB extents x 1 GiB = 1024 extents => 32 shards sustains the
  // >=32 extents/shard rule. Dedicated residency is disabled here so the test
  // isolates token routing and fixed-arena shard geometry.
  ::setenv("DFKV_RAM_TIER_EXTENT_BYTES", "1048576", 1);
  ::setenv("DFKV_RAM_TIER_SHARDS", "32", 1);
  FlushSink sink;
  RamTier::Options o;
  o.bytes = 1024ull << 20;   // 1024 x 1 MiB extents
  o.large_reserve_bytes = 0;  // preserve all 1024 arena extents for 32 shards
  o.slot_granularity = 4096;
  o.flush_threads = 8;
  RamTier t(o, sink.fn());
  ASSERT_TRUE(t.ok());
  EXPECT_EQ(t.shards(), 32u) << "32 shards must not be clamped to the old 16 cap";
  std::string v(8000, 'y');
  for (int i = 0; i < 256; ++i) ASSERT_TRUE(t.Put(K(5000 + i), v.data(), v.size())) << i;
  for (int i = 0; i < 256; ++i) {
    RamTier::Hit h;
    ASSERT_TRUE(t.GetPrep(K(5000 + i), 0, 0, &h)) << i;  // token from a high shard
    EXPECT_EQ(h.len, v.size());
    h = RamTier::Hit{};
  }
  ::unsetenv("DFKV_RAM_TIER_SHARDS");
  ::unsetenv("DFKV_RAM_TIER_EXTENT_BYTES");
}

// PutDurable: read-promotion entry. The slot is born durable — never enqueued
// for flush (zero DIO write cost) and immediately evictable/removable (no
// flush-pin). Dedup semantics match Put.
TEST(RamTier, PutDurableNoFlushIOAndImmediatelyEvictable) {
  FlushSink sink;
  sink.close();  // any enqueued flush would block forever -> test would hang
  RamTier rt(Opts(64 * 4096), sink.fn());
  ASSERT_TRUE(rt.ok());
  std::string v = "promoted-cold-page";
  ASSERT_TRUE(rt.PutDurable(K(90), v.data(), v.size()));
  EXPECT_EQ(rt.Promoted(), 1u);
  EXPECT_EQ(rt.FlushBacklog(), 0u);  // never touched the flush queue
  {
    std::lock_guard<std::mutex> lk(sink.m);
    EXPECT_EQ(sink.calls, 0);        // zero flush IO
  }
  RamTier::Hit h;
  ASSERT_TRUE(rt.GetPrep(K(90), 0, v.size(), &h));
  EXPECT_EQ(std::string(h.ptr, h.len), v);
  h = RamTier::Hit{};
  // No flush-pin held: Remove (which declines on any pin) succeeds at once.
  EXPECT_TRUE(rt.Remove(K(90)));
  EXPECT_FALSE(rt.Contains(K(90)));
}

TEST(RamTier, DirectDurableReservationPublishesAndPinsUntilRelease) {
  FlushSink sink;
  sink.close();
  RamTier rt(Opts(64 * 4096), sink.fn());
  ASSERT_TRUE(rt.ok());
  const std::string value(8192, 'r');

  RamTier::DurableReservation reservation;
  ASSERT_TRUE(rt.ReserveDurable(K(190), value.size(), &reservation));
  ASSERT_TRUE(reservation);
  ASSERT_GE(reservation.capacity(), value.size());
  EXPECT_FALSE(rt.Contains(K(190)));  // hidden while storage fills the slot
  EXPECT_EQ(rt.UsedBytes(), reservation.capacity());
  std::memcpy(reservation.data(), value.data(), value.size());

  ASSERT_TRUE(reservation.Commit());
  EXPECT_TRUE(rt.Contains(K(190)));
  EXPECT_EQ(rt.Promoted(), 1u);
  EXPECT_EQ(rt.FlushBacklog(), 0u);
  {
    std::lock_guard<std::mutex> lk(sink.m);
    EXPECT_EQ(sink.calls, 0);  // already durable: no write-back
  }

  RamTier::Hit hit;
  ASSERT_TRUE(rt.GetPrep(K(190), 0, value.size(), &hit));
  EXPECT_EQ(std::string(hit.ptr, hit.len), value);
  hit = RamTier::Hit{};

  // Commit retains the reservation pin for an in-flight RDMA send. Remove
  // hides the key immediately, but physical capacity is released only when
  // the send owner drops the reservation.
  EXPECT_TRUE(rt.Remove(K(190)));
  EXPECT_FALSE(rt.Contains(K(190)));
  EXPECT_NE(rt.UsedBytes(), 0u);
  reservation = RamTier::DurableReservation{};
  EXPECT_EQ(rt.UsedBytes(), 0u);
}

TEST(RamTier, AbandonedDirectDurableReservationReleasesHiddenSlot) {
  FlushSink sink;
  RamTier rt(Opts(64 * 4096), sink.fn());
  ASSERT_TRUE(rt.ok());
  {
    RamTier::DurableReservation reservation;
    ASSERT_TRUE(rt.ReserveDurable(K(191), 4096, &reservation));
    std::memset(reservation.data(), 'x', 4096);
    EXPECT_FALSE(rt.Contains(K(191)));
    EXPECT_NE(rt.UsedBytes(), 0u);
  }
  EXPECT_FALSE(rt.Contains(K(191)));
  EXPECT_EQ(rt.UsedBytes(), 0u);
}

TEST(RamTier, PutDurableDedupsAgainstResident) {
  FlushSink sink;
  RamTier rt(Opts(64 * 4096), sink.fn());
  std::string v(500, 'p');
  ASSERT_TRUE(rt.Put(K(91), v.data(), v.size()));
  const uint64_t used = rt.UsedBytes();
  EXPECT_TRUE(rt.PutDurable(K(91), v.data(), v.size()));  // resident -> no-op true
  EXPECT_EQ(rt.UsedBytes(), used);                        // no second slot
  EXPECT_EQ(rt.Promoted(), 0u);                           // dedup is not a promotion
}

// Eviction honours reads: a key read during every turnover of its class stays
// resident while the unread keys admitted around it are evicted. Promoted
// (born-durable) entries are used so no flush pin is involved, and the
// background reclaimer is off so admissions alone drive the hand.
TEST(RamTier, ReadKeepsHotKeyResidentAcrossClassTurnover) {
  FlushSink sink;
  const uint64_t slots = 16;
  auto o = Opts(slots * 4096);  // one extent, one class
  o.reclaim_interval_ms = 0;
  RamTier rt(o, sink.fn());
  ASSERT_TRUE(rt.ok());
  std::string v(4000, 'h');
  const BlockKey hot = K(1);
  ASSERT_TRUE(rt.PutDurable(hot, v.data(), v.size()));  // oldest resident
  for (uint64_t i = 1; i < slots; ++i)
    ASSERT_TRUE(rt.PutDurable(K(100 + i), v.data(), v.size()));
  ASSERT_EQ(rt.Count(), slots);
  ASSERT_EQ(rt.Evictions(), 0u);

  for (uint64_t i = 0; i < 3 * slots; ++i) {  // three full turnovers
    RamTier::Hit h;
    ASSERT_TRUE(rt.GetPrep(hot, 0, v.size(), &h))
        << "hot key evicted after " << i << " admissions";
    h = RamTier::Hit{};
    ASSERT_TRUE(rt.PutDurable(K(1000 + i), v.data(), v.size()));
  }
  EXPECT_TRUE(rt.Contains(hot));
  EXPECT_EQ(rt.Evictions(), 3 * slots);
  for (uint64_t i = 1; i < slots; ++i)
    EXPECT_FALSE(rt.Contains(K(100 + i))) << "unread key " << i;
  EXPECT_FALSE(rt.Contains(K(1000)));  // unread, admitted two turnovers ago
  EXPECT_TRUE(rt.Contains(K(1000 + 3 * slots - 1)));
}

// Second chance, not immunity: after its last read a key survives the
// evictions an unread key of the same age would not, and is gone within one
// pass of the hand over its class.
TEST(RamTier, KeyThatStopsBeingReadIsEvictedWithinOnePass) {
  FlushSink sink;
  const uint64_t slots = 16;
  auto o = Opts(slots * 4096);
  o.reclaim_interval_ms = 0;
  RamTier rt(o, sink.fn());
  ASSERT_TRUE(rt.ok());
  std::string v(4000, 'c');
  const BlockKey once = K(1);
  ASSERT_TRUE(rt.PutDurable(once, v.data(), v.size()));  // oldest resident
  for (uint64_t i = 1; i < slots; ++i)
    ASSERT_TRUE(rt.PutDurable(K(100 + i), v.data(), v.size()));
  ASSERT_EQ(rt.Evictions(), 0u);
  {
    RamTier::Hit h;
    ASSERT_TRUE(rt.GetPrep(once, 0, v.size(), &h));  // the last read
  }

  ASSERT_TRUE(rt.PutDurable(K(1000), v.data(), v.size()));
  EXPECT_TRUE(rt.Contains(once)) << "the oldest key was read: spare it once";
  EXPECT_FALSE(rt.Contains(K(101))) << "the oldest unread key goes instead";
  uint64_t admitted = 1;
  while (rt.Contains(once) && admitted < 2 * slots) {
    ASSERT_TRUE(rt.PutDurable(K(1000 + admitted), v.data(), v.size()));
    ++admitted;
  }
  EXPECT_FALSE(rt.Contains(once));
  EXPECT_LE(admitted, slots) << "one read must not buy more than one pass";
}

// The RAM tier bounds the reference bits one eviction clears, because the
// sweep runs under the shard lock its readers take. With every slot of a
// class larger than the bound read, each admission evicts the slot at the
// hand and moves the hand on by the bound.
TEST(RamTier, EvictionSweepIsBoundedWhenEverySlotWasRead) {
  FlushSink sink;
  const uint64_t limit = RamTier::kEvictClearLimit;
  const uint64_t slots = 4 * limit;
  auto o = Opts(slots * 4096);  // one class, one shard
  o.large_reserve_bytes = 0;
  o.reclaim_interval_ms = 0;
  RamTier rt(o, sink.fn());
  ASSERT_TRUE(rt.ok());
  ASSERT_EQ(rt.shards(), 1u);
  std::string v(4000, 'b');
  for (uint64_t i = 0; i < slots; ++i)
    ASSERT_TRUE(rt.PutDurable(K(1 + i), v.data(), v.size()));
  ASSERT_EQ(rt.Evictions(), 0u);
  for (uint64_t i = 0; i < slots; ++i) {
    RamTier::Hit h;
    ASSERT_TRUE(rt.GetPrep(K(1 + i), 0, v.size(), &h));
  }
  for (uint64_t i = 0; i < 3; ++i) {
    ASSERT_TRUE(rt.PutDurable(K(100000 + i), v.data(), v.size()));
    EXPECT_EQ(rt.Evictions(), i + 1);
    EXPECT_FALSE(rt.Contains(K(1 + i * limit))) << "admission " << i;
  }
  EXPECT_TRUE(rt.Contains(K(2)));  // passed by the first sweep, not evicted
  EXPECT_EQ(rt.Count(), slots);
}

// The reclaimer keeps going when a call ends on the reference-clearing
// bound: a class whose every slot was read still reaches its free-slot target
// in one tick (here a quarter of the class), not one slot per tick.
TEST(RamTier, ReclaimTickReachesTargetInAFullyReadClass) {
  FlushSink sink;
  const uint64_t slots = 4 * RamTier::kEvictClearLimit;
  auto o = Opts(slots * 4096);
  o.large_reserve_bytes = 0;
  o.reclaim_interval_ms = 0;  // ticks are driven by the test
  RamTier rt(o, sink.fn());
  ASSERT_TRUE(rt.ok());
  ASSERT_EQ(rt.shards(), 1u);
  std::string v(4000, 't');
  for (uint64_t i = 0; i < slots; ++i)
    ASSERT_TRUE(rt.PutDurable(K(1 + i), v.data(), v.size()));
  ASSERT_EQ(rt.Evictions(), 0u);
  for (uint64_t i = 0; i < slots; ++i) {
    RamTier::Hit h;
    ASSERT_TRUE(rt.GetPrep(K(1 + i), 0, v.size(), &h));
  }
  RamTierTestPeer::ReclaimTick(rt);
  EXPECT_EQ(rt.Reclaimed(), slots / 4);
  EXPECT_EQ(rt.Count(), slots - slots / 4);
}

// ...but only up to kReclaimClearLimitedCalls calls per class and tick. In a
// fully read class each such call clears kEvictClearLimit bits and frees one
// slot, so a class too large to sweep in one tick gives exactly that many.
TEST(RamTier, ReclaimTickCapsClearLimitedCalls) {
  FlushSink sink;
  const uint64_t cap = RamTier::kReclaimClearLimitedCalls;
  const uint64_t slots = (cap + 16) * RamTier::kEvictClearLimit;
  auto o = Opts(slots * 4096);
  o.large_reserve_bytes = 0;
  o.reclaim_interval_ms = 0;
  RamTier rt(o, sink.fn());
  ASSERT_TRUE(rt.ok());
  ASSERT_EQ(rt.shards(), 1u);
  std::string v(4000, 'u');
  for (uint64_t i = 0; i < slots; ++i)
    ASSERT_TRUE(rt.PutDurable(K(1 + i), v.data(), v.size()));
  ASSERT_EQ(rt.Evictions(), 0u);
  for (uint64_t i = 0; i < slots; ++i) {
    RamTier::Hit h;
    ASSERT_TRUE(rt.GetPrep(K(1 + i), 0, v.size(), &h));
  }
  RamTierTestPeer::ReclaimTick(rt);
  EXPECT_EQ(rt.Reclaimed(), cap);
  EXPECT_EQ(rt.Count(), slots - cap);
}

// The scenario behind the two tests above. A cold class sits next to a class
// that is read in full and then written faster than one slot per tick. If
// the reclaimer freed only one slot per tick while it sweeps the reference
// bits, admissions would run out of free slots, and an admission without a
// free slot takes a whole cold extent of another class before it evicts its
// own. No admission may take one. (The cold class, idle and much colder,
// does give its extents to the background growth, a few per tick.)
TEST(RamTier, WritesAfterFullReadDoNotStealFromAColdClass) {
  FlushSink sink;
  RamTier::Options o;
  o.bytes = 48ull << 20;
  o.large_reserve_bytes = 0;
  o.slot_granularity = 4096;
  o.reclaim_interval_ms = 0;
  std::unique_ptr<RamTier> tier;
  {
    ScopedEnv extent("DFKV_RAM_TIER_EXTENT_BYTES", "1048576");  // 48 x 1 MiB
    tier = std::make_unique<RamTier>(o, sink.fn());
  }
  RamTier& rt = *tier;
  ASSERT_TRUE(rt.ok());
  ASSERT_EQ(rt.shards(), 1u);

  // Cold class: 512 small values, never written again.
  const uint64_t cold = 512;
  std::string small(4000, 'c');
  for (uint64_t i = 0; i < cold; ++i)
    ASSERT_TRUE(rt.PutDurable(K(1 + i), small.data(), small.size()));
  auto cold_resident = [&] {
    uint64_t n = 0;
    for (uint64_t i = 0; i < cold; ++i) n += rt.Contains(K(1 + i));
    return n;
  };

  // Hot class: fill the rest of the arena and turn it over, one tick per 32
  // admissions. The reclaimer keeps up, so nothing is taken from the cold
  // class although it is by now old enough to be a cold-steal victim.
  const uint64_t per_tick = 32;
  std::string big(8000, 'h');
  uint64_t next = 100000;
  auto write_hot = [&](uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
      if (!rt.PutDurable(K(next++), big.data(), big.size())) return false;
      if ((i + 1) % per_tick == 0) RamTierTestPeer::ReclaimTick(rt);
    }
    return true;
  };
  ASSERT_TRUE(write_hot(8000));
  ASSERT_GT(rt.Evictions(), 1000u) << "the hot class never turned over";
  ASSERT_EQ(RamTierTestPeer::ColdSteals(rt), 0u);

  // Read every resident of the hot class, then keep writing at the same
  // rate: the sweep now has to clear a reference bit on every slot.
  uint64_t read = 0;
  for (uint64_t id = 100000; id < next; ++id) {
    RamTier::Hit h;
    if (rt.GetPrep(K(id), 0, big.size(), &h)) ++read;
  }
  ASSERT_GT(read, 8 * uint64_t{RamTier::kEvictClearLimit})
      << "too few slots for the sweep to take several reclaim calls";
  ASSERT_TRUE(write_hot(10 * per_tick));
  EXPECT_EQ(RamTierTestPeer::ColdSteals(rt), 0u)
      << "an admission took a cold extent";
  EXPECT_LT(cold_resident(), cold) << "the idle class gave nothing";
  EXPECT_GT(rt.Rebalanced(), 0u);
}

// The same scenario above the reclaimer's per-tick cap: the read class is too
// large to sweep in one tick, so the tick frees only
// kReclaimClearLimitedCalls slots, and admissions arrive faster than that.
// They evict inline, one slot of their own class per admission, and the cold
// class, read along with the other, keeps every resident.
TEST(RamTier, WritesFasterThanTheReclaimCapEvictTheirOwnClass) {
  FlushSink sink;
  RamTier::Options o;
  o.bytes = 176ull << 20;
  o.large_reserve_bytes = 0;
  o.slot_granularity = 4096;
  o.reclaim_interval_ms = 0;
  std::unique_ptr<RamTier> tier;
  {
    ScopedEnv extent("DFKV_RAM_TIER_EXTENT_BYTES", "1048576");  // 176 x 1 MiB
    ScopedEnv shards("DFKV_RAM_TIER_SHARDS", "1");
    tier = std::make_unique<RamTier>(o, sink.fn());
  }
  RamTier& rt = *tier;
  ASSERT_TRUE(rt.ok());
  ASSERT_EQ(rt.shards(), 1u);

  // Cold class: 512 values in 4 extents, never written again.
  const uint64_t cold = 512;
  std::string big(8000, 'c');
  for (uint64_t i = 0; i < cold; ++i)
    ASSERT_TRUE(rt.PutDurable(K(1 + i), big.data(), big.size()));
  auto cold_resident = [&] {
    uint64_t n = 0;
    for (uint64_t i = 0; i < cold; ++i) n += rt.Contains(K(1 + i));
    return n;
  };

  // Read class: the other 172 extents, 44,032 slots. Fill it and turn it
  // over at 64 admissions per tick; the reclaimer keeps 128 slots free.
  const uint64_t cap = RamTier::kReclaimClearLimitedCalls;
  std::string small(4000, 'h');
  uint64_t next = 100000;
  auto write_hot = [&](uint64_t count, bool tick_after) {
    for (uint64_t i = 0; i < count; ++i)
      if (!rt.PutDurable(K(next++), small.data(), small.size())) return false;
    if (tick_after) RamTierTestPeer::ReclaimTick(rt);
    return true;
  };
  for (int i = 0; i < 704; ++i) ASSERT_TRUE(write_hot(cap, true));
  ASSERT_GT(rt.Evictions(), 1000u) << "the read class never turned over";
  ASSERT_EQ(cold_resident(), cold);

  uint64_t read = 0;
  for (uint64_t id = 100000; id < next; ++id) {
    RamTier::Hit h;
    if (rt.GetPrep(K(id), 0, small.size(), &h)) ++read;
  }
  // The cold class is read as well: otherwise, now much colder than the read
  // class, it would give its extents to the background growth in the ticks
  // below, and their slots would change the counts.
  for (uint64_t i = 0; i < cold; ++i) {
    RamTier::Hit h;
    ASSERT_TRUE(rt.GetPrep(K(1 + i), 0, big.size(), &h));
  }
  ASSERT_GT(read, (cap + 8) * RamTier::kEvictClearLimit)
      << "one tick could sweep the whole class";

  // 128 admissions use up the free slots; the tick that follows wants 256
  // free but stops at its cap with 64. Then 512 admissions without a tick:
  // 64 take a free slot, 448 evict inline.
  const uint64_t before = rt.Evictions();
  const uint64_t reclaimed = rt.Reclaimed();
  ASSERT_TRUE(write_hot(2 * cap, true));
  EXPECT_EQ(rt.Reclaimed() - reclaimed, cap);
  ASSERT_TRUE(write_hot(512, false));
  EXPECT_EQ(cold_resident(), cold) << "a cold extent was taken";
  EXPECT_EQ(rt.Evictions() - before, 512u)
      << "640 admissions, 128 free slots: one eviction for each of the rest";
}

// ---- A flush that fails ------------------------------------------------------
namespace {
// A flush sink that fails as told and records when it was called.
struct FailingSink {
  std::mutex m;
  std::vector<std::chrono::steady_clock::time_point> calls;
  std::function<bool(const BlockKey&, int call)> ok =
      [](const BlockKey&, int) { return false; };
  RamTier::FlushFn fn() {
    return [this](const BlockKey& key, char*, size_t, size_t) {
      std::lock_guard<std::mutex> lk(m);
      calls.push_back(std::chrono::steady_clock::now());
      return ok(key, static_cast<int>(calls.size())) ? Status::kOk
                                                      : Status::kIOError;
    };
  }
  size_t count() {
    std::lock_guard<std::mutex> lk(m);
    return calls.size();
  }
  double gap_ms(size_t i) {  // between call i and call i+1 (1-based)
    std::lock_guard<std::mutex> lk(m);
    return std::chrono::duration<double, std::milli>(calls[i] - calls[i - 1])
        .count();
  }
};
RamTier::Options RetryOpts(uint32_t attempts, uint32_t base_ms) {
  RamTier::Options o = Opts(64 * 4096);
  o.flush_retries = attempts;
  o.flush_retry_base_ms = base_ms;
  o.health_recovery_min_ms = 0;  // the tests that are about it set it
  return o;
}

// A flush sink whose calls block until released and then fail: an attempt
// that is in flight for as long as the test wants.
struct HeldSink {
  std::atomic<int> entered{0};
  std::atomic<int> hold_from{1};  // calls from this one on wait for release
  std::atomic<bool> release{false};
  RamTier::FlushFn fn() {
    return [this](const BlockKey&, char*, size_t, size_t) {
      const int call = entered.fetch_add(1) + 1;
      while (call >= hold_from.load() && !release.load())
        std::this_thread::sleep_for(1ms);
      return Status::kIOError;
    };
  }
};
}  // namespace

// A value that was acknowledged from RAM: its flush is tried again after
// growing pauses, and the value stays readable meanwhile. It used to get
// three attempts within microseconds and was then gone.
TEST(RamTierFlushFailure, AcknowledgedValueIsRetriedWithPausesAndSurvives) {
  FailingSink sink;
  sink.ok = [](const BlockKey&, int call) { return call > 3; };
  RamTier rt(RetryOpts(5, 20), sink.fn());
  const std::string value(500, 'r');
  ASSERT_EQ(rt.PutWriteBack(K(1), value.data(), value.size()).status,
            Status::kOk);
  ASSERT_TRUE(WaitFor([&] { return sink.count() >= 2; }));
  EXPECT_TRUE(rt.Contains(K(1))) << "readable while its flush is retried";
  EXPECT_EQ(rt.DirtyObjects(), 1u);
  EXPECT_GE(rt.FlushBacklog(), 1u);
  ASSERT_TRUE(WaitFor([&] { return rt.Flushed() == 1u; }));
  ASSERT_EQ(sink.count(), 4u);
  // Pauses of 20, 80 and 320 ms (a little less is a scheduler's rounding).
  EXPECT_GE(sink.gap_ms(1), 18.0);
  EXPECT_GE(sink.gap_ms(2), 75.0);
  EXPECT_GE(sink.gap_ms(3), 300.0);
  EXPECT_EQ(rt.FlushRetries(), 3u);
  EXPECT_EQ(rt.FlushDropped(), 0u);
  EXPECT_EQ(rt.PostAckFlushFailures(), 0u);
  EXPECT_EQ(rt.DirtyObjects(), 0u);
  EXPECT_TRUE(rt.healthy());
  EXPECT_TRUE(rt.Contains(K(1)));
}

TEST(RamTierFlushFailure, AfterTheLastAttemptTheValueIsDroppedAndCounted) {
  FailingSink sink;
  RamTier rt(RetryOpts(4, 5), sink.fn());
  const std::string value(500, 'd');
  ASSERT_EQ(rt.PutWriteBack(K(1), value.data(), value.size()).status,
            Status::kOk);
  ASSERT_TRUE(WaitFor([&] { return rt.FlushDropped() == 1u; }));
  EXPECT_EQ(sink.count(), 4u);
  EXPECT_EQ(rt.FlushRetries(), 3u);
  EXPECT_EQ(rt.PostAckFlushFailures(), 1u);
  EXPECT_FALSE(rt.Contains(K(1)));
  EXPECT_EQ(rt.DirtyObjects(), 0u);
  EXPECT_EQ(rt.FlushBacklog(), 0u);
  EXPECT_FALSE(rt.healthy());
}

// A PUT that is still waiting for its flush (disk-ACK) gets its error at
// once: nothing is lost by failing fast, and a handler thread is not held.
TEST(RamTierFlushFailure, WaitingPutIsRetriedWithoutPauses) {
  FailingSink sink;
  RamTier rt(RetryOpts(5, 60000), sink.fn());
  const std::string value(500, 'w');
  const auto started = std::chrono::steady_clock::now();
  const RamTier::PutResult result =
      rt.PutCommitted(K(1), value.data(), value.size());
  EXPECT_EQ(result.status, Status::kIOError);
  EXPECT_EQ(result.flush, RamTier::FlushOutcome::kDroppedAfterRetries);
  EXPECT_LT(std::chrono::steady_clock::now() - started, 5s);
  EXPECT_EQ(sink.count(), 5u);
  EXPECT_EQ(rt.PostAckFlushFailures(), 0u) << "it was never acknowledged";
  EXPECT_EQ(rt.FlushDropped(), 1u);
}

// A disk store that has failed for good: no second attempt.
TEST(RamTierFlushFailure, FlushToAFailedTargetIsNotRetried) {
  FailingSink sink;
  RamTier::Options options = RetryOpts(5, 60000);
  std::atomic<bool> failed{true};
  options.flush_target_failed = [&](const BlockKey&) { return failed.load(); };
  RamTier rt(options, sink.fn());
  const std::string value(500, 't');
  ASSERT_EQ(rt.PutWriteBack(K(1), value.data(), value.size()).status,
            Status::kOk);
  ASSERT_TRUE(WaitFor([&] { return rt.FlushDropped() == 1u; }));
  EXPECT_EQ(sink.count(), 1u);
  EXPECT_EQ(rt.FlushRetries(), 0u);
  EXPECT_EQ(rt.PostAckFlushFailures(), 1u);
  EXPECT_FALSE(rt.healthy());
}

// The tier is unhealthy from a drop until flushes have succeeded again in a
// row. It used to stay unhealthy until the process was restarted.
TEST(RamTierFlushFailure, HealthRecoversWhenFlushesSucceedAgain) {
  FailingSink sink;
  std::atomic<bool> failing{true};
  sink.ok = [&](const BlockKey&, int) { return !failing.load(); };
  RamTier::Options options = RetryOpts(2, 1);
  options.health_recovery_flushes = 8;
  RamTier rt(options, sink.fn());
  const std::string value(500, 'h');
  auto put = [&](uint64_t id) {
    return rt.PutWriteBack(K(id), value.data(), value.size()).status;
  };
  ASSERT_EQ(put(1), Status::kOk);
  ASSERT_TRUE(WaitFor([&] { return rt.FlushDropped() == 1u; }));
  ASSERT_FALSE(rt.healthy());

  failing = false;
  for (uint64_t id = 10; id < 17; ++id) ASSERT_EQ(put(id), Status::kOk);
  ASSERT_TRUE(WaitFor([&] { return rt.Flushed() == 7u; }));
  EXPECT_FALSE(rt.healthy()) << "seven in a row are not eight";

  // A failed attempt in between starts the count again, also when its retry
  // then succeeds.
  failing = true;
  ASSERT_EQ(put(20), Status::kOk);
  ASSERT_TRUE(WaitFor([&] { return rt.FlushRetries() >= 2u; }));
  failing = false;
  ASSERT_TRUE(WaitFor([&] { return rt.DirtyObjects() == 0u; }));
  // (Its retry succeeded or was the last attempt: at most one success so far.)
  for (uint64_t id = 30; id < 36; ++id) ASSERT_EQ(put(id), Status::kOk);
  ASSERT_TRUE(WaitFor([&] { return rt.DirtyObjects() == 0u; }));
  EXPECT_FALSE(rt.healthy());

  ASSERT_EQ(put(40), Status::kOk);
  ASSERT_EQ(put(41), Status::kOk);
  ASSERT_TRUE(WaitFor([&] { return rt.healthy(); }));
  // And a later drop makes it unhealthy again.
  failing = true;
  ASSERT_EQ(put(50), Status::kOk);
  ASSERT_TRUE(WaitFor([&] { return !rt.healthy(); }));
}

// A stop does not wait out a backoff: once the tier is being drained, the
// failing object gets its remaining attempts at once and is given up.
TEST(RamTierFlushFailure, DrainDoesNotWaitOutTheBackoff) {
  FailingSink sink;
  RamTier rt(RetryOpts(5, 60000), sink.fn());
  const std::string value(500, 's');
  ASSERT_EQ(rt.PutWriteBack(K(1), value.data(), value.size()).status,
            Status::kOk);
  ASSERT_TRUE(WaitFor([&] { return sink.count() == 1u; }));
  std::this_thread::sleep_for(20ms);
  ASSERT_EQ(sink.count(), 1u) << "the next attempt is a minute away";
  const auto started = std::chrono::steady_clock::now();
  EXPECT_TRUE(rt.WaitForDrain(10s));
  EXPECT_LT(std::chrono::steady_clock::now() - started, 5s);
  EXPECT_EQ(sink.count(), 5u);
  EXPECT_EQ(rt.FlushDropped(), 1u);
  EXPECT_EQ(rt.DirtyObjects(), 0u);
}

// Removing a value while its flush waits for the next attempt.
TEST(RamTierFlushFailure, RemoveWhileWaitingForTheNextAttempt) {
  FailingSink sink;
  RamTier rt(RetryOpts(5, 60000), sink.fn());
  const std::string value(500, 'x');
  ASSERT_EQ(rt.PutWriteBack(K(1), value.data(), value.size()).status,
            Status::kOk);
  ASSERT_TRUE(WaitFor([&] { return sink.count() == 1u; }));
  ASSERT_TRUE(WaitFor([&] { return rt.FlushRetries() == 1u; }));
  EXPECT_TRUE(rt.Remove(K(1)));
  EXPECT_FALSE(rt.Contains(K(1)));
  EXPECT_EQ(rt.DirtyObjects(), 0u);
  EXPECT_EQ(rt.FlushBacklog(), 0u);
  // The key can be written again, and that one is flushed.
  {
    std::lock_guard<std::mutex> lk(sink.m);
    sink.ok = [](const BlockKey&, int) { return true; };
  }
  ASSERT_EQ(rt.PutWriteBack(K(1), value.data(), value.size()).status,
            Status::kOk);
  ASSERT_TRUE(WaitFor([&] { return rt.Flushed() == 1u; }));
  EXPECT_EQ(sink.count(), 2u);
  EXPECT_EQ(rt.FlushDropped(), 0u);
}

// The same while the retried attempt is in flight: Remove waits for that
// attempt to come back, and it is the last one.
TEST(RamTierFlushFailure, RemoveWhileARetriedAttemptIsInFlight) {
  HeldSink sink;
  sink.hold_from = 2;  // the first attempt fails at once, the second is held
  RamTier rt(RetryOpts(5, 10), sink.fn());
  const std::string value(500, 'y');
  ASSERT_EQ(rt.PutWriteBack(K(1), value.data(), value.size()).status,
            Status::kOk);
  ASSERT_TRUE(WaitFor([&] { return sink.entered.load() == 2; }));
  std::atomic<bool> removed{false};
  std::thread remover([&] {
    EXPECT_TRUE(rt.Remove(K(1)));
    removed = true;
  });
  std::this_thread::sleep_for(50ms);
  EXPECT_FALSE(removed.load()) << "Remove returned under an attempt in flight";
  EXPECT_FALSE(rt.Contains(K(1))) << "hidden from the moment Remove began";
  sink.release = true;
  remover.join();
  EXPECT_FALSE(rt.Contains(K(1)));
  EXPECT_EQ(rt.DirtyObjects(), 0u);
  EXPECT_EQ(rt.FlushBacklog(), 0u);
  std::this_thread::sleep_for(100ms);  // several pauses long
  EXPECT_EQ(sink.entered.load(), 2) << "a removed value was tried again";
  EXPECT_EQ(rt.PostAckFlushFailures(), 0u) << "removed, not lost";
}

// Recovery also needs a minimum time since the last drop: on a busy node the
// flushes alone would end the unhealthy state within milliseconds, before any
// scrape or probe had seen it. Once the streak is complete the time is
// enough: a node that goes idle then does not stay unhealthy.
TEST(RamTierFlushFailure, HealthStaysDownForAMinimumTime) {
  FailingSink sink;
  std::atomic<bool> failing{true};
  sink.ok = [&](const BlockKey&, int) { return !failing.load(); };
  RamTier::Options options = RetryOpts(1, 1);
  options.health_recovery_flushes = 4;
  options.health_recovery_min_ms = 400;
  RamTier rt(options, sink.fn());
  const std::string value(500, 'm');
  auto put = [&](uint64_t id) {
    return rt.PutWriteBack(K(id), value.data(), value.size()).status;
  };
  auto drop_one = [&](uint64_t id) {
    const uint64_t before = rt.FlushDropped();
    failing = true;
    EXPECT_EQ(put(id), Status::kOk);
    EXPECT_TRUE(WaitFor([&] { return rt.FlushDropped() == before + 1; }));
    failing = false;
    return std::chrono::steady_clock::now();
  };
  auto flush = [&](uint64_t first, uint64_t count) {
    const uint64_t before = rt.Flushed();
    for (uint64_t id = first; id < first + count; ++id)
      EXPECT_EQ(put(id), Status::kOk);
    EXPECT_TRUE(WaitFor([&] { return rt.Flushed() == before + count; }));
  };

  // A complete streak, then nothing more is written.
  auto dropped_at = drop_one(1);
  flush(10, 20);
  ASSERT_LT(std::chrono::steady_clock::now() - dropped_at, 350ms)
      << "the machine is too slow for this test's timing";
  EXPECT_FALSE(rt.healthy()) << "twenty good flushes, but only moments later";
  std::this_thread::sleep_for(450ms);
  EXPECT_TRUE(rt.healthy()) << "idle after a complete streak: time is enough";

  // An incomplete streak: time is not enough, the missing flushes are.
  dropped_at = drop_one(2);
  flush(40, 2);
  std::this_thread::sleep_for(450ms);
  EXPECT_FALSE(rt.healthy()) << "two of four, however long ago";
  flush(50, 2);
  EXPECT_TRUE(rt.healthy());

  // A drop after a complete streak, before the time is up: it starts both
  // over. A drop always wins.
  dropped_at = drop_one(3);
  flush(60, 4);
  (void)drop_one(4);
  std::this_thread::sleep_for(450ms);
  EXPECT_FALSE(rt.healthy()) << "the streak before the second drop is void";
  flush(70, 4);
  EXPECT_TRUE(rt.healthy());
}

// A PUT that waits for a flush is told how it ended, not only that it failed:
// a waiting writer must not report success for a value that was dropped, and
// can tell a store that is gone from one that may work again.
TEST(RamTierFlushFailure, WaitingPutLearnsHowTheFlushEnded) {
  const std::string value(500, 'o');
  {  // written
    FailingSink sink;
    sink.ok = [](const BlockKey&, int) { return true; };
    RamTier rt(RetryOpts(5, 1), sink.fn());
    const auto result = rt.PutCommitted(K(1), value.data(), value.size());
    EXPECT_EQ(result.status, Status::kOk);
    EXPECT_EQ(result.flush, RamTier::FlushOutcome::kFlushed);
    // Acknowledged from RAM: this PUT did not wait.
    const auto acked = rt.PutWriteBack(K(2), value.data(), value.size());
    EXPECT_EQ(acked.status, Status::kOk);
    EXPECT_EQ(acked.flush, RamTier::FlushOutcome::kNone);
  }
  {  // the disk store has failed for good
    FailingSink sink;
    RamTier::Options options = RetryOpts(5, 1);
    options.flush_target_failed = [](const BlockKey&) { return true; };
    RamTier rt(options, sink.fn());
    const auto result = rt.PutCommitted(K(1), value.data(), value.size());
    EXPECT_EQ(result.status, Status::kIOError);
    EXPECT_EQ(result.flush, RamTier::FlushOutcome::kDroppedStoreFailed);
  }
  {  // two disk-ACK writers of one key: the second gets the first's outcome
    HeldSink sink;
    RamTier rt(RetryOpts(3, 1), sink.fn());
    RamTier::PutResult first{Status::kOk}, second{Status::kOk};
    std::thread a([&] {
      first = rt.PutCommitted(K(1), value.data(), value.size());
    });
    ASSERT_TRUE(WaitFor([&] { return sink.entered.load() == 1; }));
    std::thread b([&] {
      second = rt.PutCommitted(K(1), value.data(), value.size());
    });
    std::this_thread::sleep_for(50ms);  // b has joined a's flush
    sink.release = true;
    a.join();
    b.join();
    EXPECT_EQ(sink.entered.load(), 3) << "one value, three attempts";
    for (const auto& result : {first, second}) {
      EXPECT_EQ(result.status, Status::kIOError);
      EXPECT_EQ(result.flush, RamTier::FlushOutcome::kDroppedAfterRetries);
    }
    EXPECT_EQ(rt.FlushDropped(), 1u);
  }
  {  // still failing when the tier is drained for a stop
    HeldSink sink;
    RamTier rt(RetryOpts(3, 1), sink.fn());
    RamTier::PutResult result{Status::kOk};
    std::thread writer([&] {
      result = rt.PutCommitted(K(1), value.data(), value.size());
    });
    ASSERT_TRUE(WaitFor([&] { return sink.entered.load() == 1; }));
    std::thread drainer([&] { EXPECT_TRUE(rt.WaitForDrain(10s)); });
    std::this_thread::sleep_for(50ms);  // the drain has begun
    sink.release = true;
    writer.join();
    drainer.join();
    EXPECT_EQ(result.status, Status::kIOError);
    EXPECT_EQ(result.flush, RamTier::FlushOutcome::kDroppedAtShutdown);
  }
}

// A second RAM-ACK writer of a key whose value is waiting for its next flush
// attempt is acknowledged like the first, at once. It used to wait for the
// flush, which with the pauses is seconds -- under a put-key stripe.
TEST(RamTierFlushFailure, SecondWriterOfAnAcknowledgedValueDoesNotWait) {
  FailingSink sink;
  RamTier rt(RetryOpts(3, 2000), sink.fn());  // pauses of 2 s and 8 s
  const std::string value(500, 'a');
  ASSERT_EQ(rt.PutWriteBack(K(1), value.data(), value.size()).status,
            Status::kOk);
  ASSERT_TRUE(WaitFor([&] { return rt.FlushRetries() == 1u; }));
  const auto started = std::chrono::steady_clock::now();
  for (int i = 0; i < 100; ++i) {
    const auto second = rt.PutWriteBack(K(1), value.data(), value.size());
    ASSERT_EQ(second.status, Status::kOk);
    ASSERT_EQ(second.flush, RamTier::FlushOutcome::kNone);
  }
  EXPECT_LT(std::chrono::steady_clock::now() - started, 1s);
  EXPECT_EQ(rt.RamAcks(), 101u);
  EXPECT_TRUE(rt.Contains(K(1)));
  // One value: if it is lost, it is one lost acknowledged value.
  EXPECT_TRUE(rt.WaitForDrain(10s));
  EXPECT_EQ(rt.FlushDropped(), 1u);
  EXPECT_EQ(rt.PostAckFlushFailures(), 1u);
  // Once it is gone, a PUT of the key is a new value, not a duplicate.
  EXPECT_FALSE(rt.Contains(K(1)));
}

// A disk that answers kCacheFull has no slot for the value now; it has not
// failed. Such attempts are neither retries nor failures and do not touch the
// tier's health.
namespace {
struct NoRoomSink {
  std::atomic<int> calls{0};
  std::function<Status(int call)> answer = [](int) {
    return Status::kCacheFull;
  };
  RamTier::FlushFn fn() {
    return [this](const BlockKey&, char*, size_t, size_t) {
      return answer(calls.fetch_add(1) + 1);
    };
  }
};
}  // namespace

// An acknowledged value waits in RAM for room, readable, until the hold time
// is over; then it is dropped like an eviction, not counted as lost.
TEST(RamTierFlushFailure, AcknowledgedValueWaitsForRoomThenIsDroppedNotLost) {
  NoRoomSink sink;
  RamTier::Options options = RetryOpts(3, 10);
  options.flush_no_room_hold_ms = 300;  // attempts at 0, 10, 50, 210, 850 ms
  RamTier rt(options, sink.fn());
  const std::string value(500, 'n');
  ASSERT_EQ(rt.PutWriteBack(K(1), value.data(), value.size()).status,
            Status::kOk);
  ASSERT_TRUE(WaitFor([&] { return sink.calls.load() >= 3; }));
  EXPECT_TRUE(rt.Contains(K(1))) << "readable while it waits for room";
  ASSERT_TRUE(WaitFor([&] { return rt.FlushNoRoomDropped() == 1u; }));
  EXPECT_EQ(sink.calls.load(), 5) << "more attempts than flush_retries";
  EXPECT_EQ(rt.FlushNoRoom(), static_cast<uint64_t>(sink.calls.load()));
  EXPECT_EQ(rt.FlushRetries(), 0u);
  EXPECT_EQ(rt.FlushDropped(), 0u);
  EXPECT_EQ(rt.PostAckFlushFailures(), 0u);
  EXPECT_TRUE(rt.healthy());
  EXPECT_FALSE(rt.Contains(K(1)));
  EXPECT_EQ(rt.DirtyObjects(), 0u);
  EXPECT_EQ(rt.FlushBacklog(), 0u);
}

// Room again before the hold time is over: written, nothing dropped.
TEST(RamTierFlushFailure, AcknowledgedValueIsWrittenOnceThereIsRoom) {
  NoRoomSink sink;
  sink.answer = [](int call) {
    return call > 3 ? Status::kOk : Status::kCacheFull;
  };
  RamTier rt(RetryOpts(2, 5), sink.fn());  // fewer failures than refusals
  const std::string value(500, 'r');
  ASSERT_EQ(rt.PutWriteBack(K(1), value.data(), value.size()).status,
            Status::kOk);
  ASSERT_TRUE(WaitFor([&] { return rt.Flushed() == 1u; }));
  EXPECT_EQ(sink.calls.load(), 4);
  EXPECT_EQ(rt.FlushNoRoom(), 3u);
  EXPECT_EQ(rt.FlushNoRoomDropped(), 0u);
  EXPECT_EQ(rt.FlushDropped(), 0u);
  EXPECT_EQ(rt.DirtyObjects(), 0u);
  EXPECT_TRUE(rt.healthy());
}

// A PUT that waits for its flush (disk-ACK) is answered kCacheFull at once,
// as the disk would answer it, and the value is not kept.
TEST(RamTierFlushFailure, WaitingPutForADiskWithNoRoomIsCacheFull) {
  NoRoomSink sink;
  RamTier rt(RetryOpts(5, 60000), sink.fn());
  const std::string value(500, 'w');
  const RamTier::PutResult result =
      rt.PutCommitted(K(1), value.data(), value.size());
  EXPECT_EQ(result.status, Status::kCacheFull);
  EXPECT_EQ(result.flush, RamTier::FlushOutcome::kNoRoom);
  EXPECT_EQ(sink.calls.load(), 1);
  EXPECT_EQ(rt.FlushNoRoomDropped(), 1u);
  EXPECT_EQ(rt.FlushDropped(), 0u);
  EXPECT_FALSE(rt.Contains(K(1)));
  EXPECT_TRUE(rt.healthy());
}

// A real failure after refusals is still a failure: the attempts that failed
// count against flush_retries, and the value is dropped and counted as lost.
TEST(RamTierFlushFailure, FailuresAfterRefusalsStillEndInALoss) {
  NoRoomSink sink;
  sink.answer = [](int call) {
    return call <= 2 ? Status::kCacheFull : Status::kIOError;
  };
  RamTier rt(RetryOpts(3, 5), sink.fn());
  const std::string value(500, 'f');
  ASSERT_EQ(rt.PutWriteBack(K(1), value.data(), value.size()).status,
            Status::kOk);
  ASSERT_TRUE(WaitFor([&] { return rt.FlushDropped() == 1u; }));
  EXPECT_EQ(sink.calls.load(), 5);
  EXPECT_EQ(rt.FlushNoRoom(), 2u);
  EXPECT_EQ(rt.FlushRetries(), 2u);
  EXPECT_EQ(rt.PostAckFlushFailures(), 1u);
  EXPECT_FALSE(rt.healthy());
}

// While the tier drains for a stop there is no time to wait for room: an
// acknowledged value the disk has no slot for gets flush_retries attempts
// without pauses, as a failing one does, and is then dropped as one the disk
// had no room for, not as lost.
TEST(RamTierFlushFailure, WhileDrainingAValueWithNoRoomGetsUnpacedAttempts) {
  for (bool room_later : {false, true}) {
    SCOPED_TRACE(room_later ? "room on the third attempt" : "never room");
    NoRoomSink sink;
    std::atomic<bool> go{false};
    sink.answer = [&](int call) {
      while (call == 1 && !go.load()) std::this_thread::sleep_for(1ms);
      return room_later && call == 3 ? Status::kOk : Status::kCacheFull;
    };
    RamTier rt(RetryOpts(3, 60000), sink.fn());  // pauses would be minutes
    const std::string value(500, 'd');
    ASSERT_EQ(rt.PutWriteBack(K(1), value.data(), value.size()).status,
              Status::kOk);
    ASSERT_TRUE(WaitFor([&] { return sink.calls.load() == 1; }));
    bool drained = false;
    std::thread drain([&] { drained = rt.WaitForDrain(5s); });
    std::this_thread::sleep_for(50ms);  // the drain has begun
    go = true;
    drain.join();
    EXPECT_TRUE(drained);
    EXPECT_EQ(sink.calls.load(), 3);
    EXPECT_EQ(rt.Flushed(), room_later ? 1u : 0u);
    EXPECT_EQ(rt.FlushNoRoomDropped(), room_later ? 0u : 1u);
    EXPECT_EQ(rt.FlushDropped(), 0u);
    EXPECT_EQ(rt.PostAckFlushFailures(), 0u);
    EXPECT_TRUE(rt.healthy());
  }
}

// flush_given_up is called once for a value dropped without having been
// written, and not for one that was removed while its flush was in flight:
// the remover takes care of that key itself.
TEST(RamTierFlushFailure, GivenUpIsNotCalledForAValueRemovedDuringItsFlush) {
  NoRoomSink sink;
  std::atomic<bool> go{false};
  sink.answer = [&](int call) {
    while (call == 1 && !go.load()) std::this_thread::sleep_for(1ms);
    return Status::kIOError;
  };
  std::mutex mu;
  std::vector<uint64_t> given_up;
  RamTier::Options options = RetryOpts(2, 1);
  options.flush_given_up = [&](const BlockKey& key) {
    std::lock_guard<std::mutex> lk(mu);
    given_up.push_back(key.digest_hi);
  };
  RamTier rt(options, sink.fn());
  const std::string value(500, 'g');
  ASSERT_EQ(rt.PutWriteBack(K(1), value.data(), value.size()).status,
            Status::kOk);
  ASSERT_TRUE(WaitFor([&] { return sink.calls.load() == 1; }));  // in flight
  std::thread remover([&] { EXPECT_TRUE(rt.Remove(K(1))); });
  std::this_thread::sleep_for(50ms);  // the remove waits for that attempt
  go = true;
  remover.join();
  EXPECT_FALSE(rt.Contains(K(1)));
  EXPECT_EQ(rt.FlushDropped(), 0u);
  {
    std::lock_guard<std::mutex> lk(mu);
    EXPECT_TRUE(given_up.empty()) << "a removed value is not given up";
  }

  // A value whose attempts all fail: given up, once.
  ASSERT_EQ(rt.PutWriteBack(K(2), value.data(), value.size()).status,
            Status::kOk);
  ASSERT_TRUE(WaitFor([&] { return rt.FlushDropped() == 1u; }));
  std::lock_guard<std::mutex> lk(mu);
  EXPECT_EQ(given_up, std::vector<uint64_t>{2});
}

// ---- Promotions and the disk (REQ-09) ----------------------------------------
// A promotion installs a copy of what the disk holds. If the disk no longer
// holds the key when the copy is installed (it was removed meanwhile), the
// copy is not installed: it would bring a removed key back.
TEST(RamTier, APromotionOfAValueTheDiskNoLongerHoldsIsNotInstalled) {
  FlushSink sink;
  std::atomic<uint64_t> on_disk{0};  // generation of the disk value, 0 none
  RamTier::Options o = Opts(64 * 4096);
  o.disk_generation = [&](const BlockKey&) { return on_disk.load(); };
  RamTier rt(o, sink.fn());
  const std::string v(5000, 'p');
  // Read generation 7; the disk then lost the key, or holds generation 8.
  for (uint64_t now : {uint64_t{0}, uint64_t{8}}) {
    SCOPED_TRACE(now == 0 ? "removed" : "replaced");
    on_disk = now;
    const uint64_t id = 10 * now;
    rt.PutDurable(K(id + 1), v.data(), v.size(), 7);
    EXPECT_FALSE(rt.Contains(K(id + 1))) << "a copied promotion";
    {
      RamTier::DurableReservation reservation;
      ASSERT_TRUE(rt.ReserveDurable(K(id + 2), v.size(), &reservation, 7));
      std::memcpy(reservation.data(), v.data(), v.size());
      EXPECT_FALSE(reservation.Commit());
    }
    EXPECT_FALSE(rt.Contains(K(id + 2))) << "a reserved promotion";
  }
  on_disk = 7;
  EXPECT_TRUE(rt.PutDurable(K(3), v.data(), v.size(), 7));
  EXPECT_TRUE(rt.Contains(K(3))) << "the disk still holds it: installed";
}

// ---- A cold-read reservation and the operations that meet it (REQ-10) --------
// Only the thread that holds a reservation can finish it, after its disk read
// and reply. A REMOVE or a PUT of the key on that same thread used to wait for
// it for ever (a REMOVE) or for put_wait_ms (a PUT; for ever with 0). Neither
// may wait: the REMOVE cancels the reservation and returns, the PUT goes to
// the disk, which already holds the key.
TEST(RamTier, OperationsOnAKeyWithAColdReadReservationDoNotWaitForIt) {
  FlushSink sink;
  RamTier::Options o = Opts(64 * 4096);
  o.put_wait_ms = 0;
  RamTier rt(o, sink.fn());
  const std::string v(5000, 'r');
  for (bool remove : {true, false}) {
    SCOPED_TRACE(remove ? "remove" : "put");
    RamTier::DurableReservation reservation;
    ASSERT_TRUE(rt.ReserveDurable(K(9), v.size(), &reservation));
    std::atomic<bool> done{false};
    RamTier::PutResult put{Status::kIOError};
    bool removed = false;
    std::thread op([&] {
      if (remove)
        removed = rt.Remove(K(9));
      else
        put = rt.PutWriteBack(K(9), v.data(), v.size());
      done = true;
    });
    const bool returned = WaitFor([&] { return done.load(); });
    EXPECT_TRUE(returned) << "it waited for the reservation";
    std::memcpy(reservation.data(), v.data(), v.size());
    const bool installed = reservation.Commit();  // lets a waiter go
    op.join();
    if (remove) {
      EXPECT_TRUE(removed);
      EXPECT_FALSE(installed) << "the remove canceled the promotion";
      EXPECT_FALSE(rt.Contains(K(9)));
    } else {
      EXPECT_EQ(put.status, Status::kCacheFull);
      EXPECT_EQ(put.disposition, RamTier::PutDisposition::kBypass);
      EXPECT_TRUE(installed);
      rt.Remove(K(9));
    }
  }
}

// ---- A disk that will never take a value (STO-30) ----------------------------
// kInvalid (too large for the disk) and kQuotaExceeded are final answers of a
// healthy disk: no retries, no health change, not counted as lost. A waiting
// PUT gets the disk's answer. They used to be retried and then dropped as a
// failure, an acknowledged value counted as lost and the tier unhealthy.
TEST(RamTierFlushFailure, AValueTheDiskRefusesIsDroppedAtOnceNotLost) {
  for (Status refusal : {Status::kInvalid, Status::kQuotaExceeded}) {
    SCOPED_TRACE(static_cast<int>(refusal));
    NoRoomSink sink;
    sink.answer = [refusal](int) { return refusal; };
    RamTier rt(RetryOpts(5, 10), sink.fn());
    const std::string value(500, 'x');
    ASSERT_EQ(rt.PutWriteBack(K(1), value.data(), value.size()).status,
              Status::kOk);
    ASSERT_TRUE(WaitFor([&] { return rt.FlushRefused() == 1u; }));
    EXPECT_EQ(sink.calls.load(), 1);
    EXPECT_FALSE(rt.Contains(K(1)));
    const RamTier::PutResult waited =
        rt.PutCommitted(K(2), value.data(), value.size());
    EXPECT_EQ(waited.status, refusal);
    EXPECT_EQ(waited.flush, RamTier::FlushOutcome::kRefused);
    EXPECT_EQ(sink.calls.load(), 2);
    EXPECT_EQ(rt.FlushRefused(), 2u);
    EXPECT_EQ(rt.FlushRetries(), 0u);
    EXPECT_EQ(rt.FlushDropped(), 0u);
    EXPECT_EQ(rt.PostAckFlushFailures(), 0u);
    EXPECT_EQ(rt.DirtyObjects(), 0u);
    EXPECT_TRUE(rt.healthy());
  }
}

// A written copy of a key the disk no longer has is dropped by the next PUT,
// which goes to the disk. If a reader still holds the copy, the drop is
// deferred until the reader lets go; a further PUT meanwhile must go to the
// disk as well, not be refused as if the key were being removed (that answer
// is kIOError, on which clients cool the node down).
TEST(RamTier, APutAfterAStaleCopyWasDroppedUnderAReaderGoesToTheDisk) {
  FlushSink sink;
  RamTier::Options o = Opts(64 * 4096);
  o.is_persisted = [](const BlockKey&) { return false; };  // the disk lost it
  RamTier rt(o, sink.fn());
  const std::string v(5000, 's');
  ASSERT_TRUE(rt.PutDurable(K(1), v.data(), v.size()));
  RamTier::Hit reader;
  ASSERT_TRUE(rt.GetPrep(K(1), 0, v.size(), &reader));  // holds a send pin
  RamTier::PutResult first = rt.PutWriteBack(K(1), v.data(), v.size());
  EXPECT_EQ(first.disposition, RamTier::PutDisposition::kBypass);
  RamTier::PutResult second = rt.PutWriteBack(K(1), v.data(), v.size());
  EXPECT_NE(second.status, Status::kIOError);
  EXPECT_EQ(second.disposition, RamTier::PutDisposition::kBypass);
}

// One disk that hangs (slow, not failing) must not hold up the flushes of
// the others. The sink is shaped like the server's: one call writes every
// item it is given, each to its own disk, and returns when all are done.
namespace {
struct TwoDiskSink {
  std::mutex mu;
  std::condition_variable cv;
  bool disk1_open = false;
  static size_t Disk(const BlockKey& key) { return key.digest_hi % 2; }
  std::vector<Status> operator()(const std::vector<RamTier::FlushItem>& items) {
    for (const auto& item : items) {
      if (Disk(item.key) != 1) continue;
      std::unique_lock<std::mutex> lock(mu);
      cv.wait(lock, [&] { return disk1_open; });
      break;
    }
    return std::vector<Status>(items.size(), Status::kOk);
  }
  void Open() {
    {
      std::lock_guard<std::mutex> lock(mu);
      disk1_open = true;
    }
    cv.notify_all();
  }
};
}  // namespace

TEST(RamTier, AHungDiskDoesNotHoldUpTheFlushesOfTheOthers) {
  TwoDiskSink sink;
  RamTier::Options o = Opts(1024 * 4096);
  o.flush_threads = 4;
  o.flush_lanes = 2;
  o.flush_lane = [](const BlockKey& key) { return TwoDiskSink::Disk(key); };
  RamTier rt(o, nullptr);
  rt.set_flush_batch([&](const std::vector<RamTier::FlushItem>& items) {
    return sink(items);
  });
  ASSERT_TRUE(rt.ok());
  const std::string v(4000, 'v');
  for (uint64_t i = 0; i < 64; ++i) ASSERT_TRUE(rt.Put(K(i), v.data(), v.size()));
  const auto disk0_written = [&] {
    for (uint64_t i = 0; i < 64; i += 2) {
      size_t len = 0;
      bool dirty = true;
      if (!rt.Lookup(K(i), &len, &dirty) || dirty) return false;
    }
    return true;
  };
  EXPECT_TRUE(WaitFor(disk0_written))
      << "the values of the healthy disk wait for the hung one";
  sink.Open();
  ASSERT_TRUE(WaitFor([&] { return rt.FlushBacklog() == 0; }));
}

// A disk whose values pile up dirty fills its share of the RAM-ACK
// watermark, not the whole of it: new keys of the other disks are still
// acknowledged from RAM.
TEST(RamTier, AHungDiskOnlyBackpressuresItsOwnKeys) {
  constexpr size_t kDisks = 4;
  std::mutex mu;
  std::condition_variable cv;
  bool open = false;
  RamTier::Options o = Opts(256 * 4096);
  o.flush_threads = 2;
  o.flush_lanes = kDisks;
  o.flush_lane = [](const BlockKey& key) { return key.digest_hi % kDisks; };
  RamTier rt(o, nullptr);
  rt.set_flush_batch([&](const std::vector<RamTier::FlushItem>& items) {
    if (!items.empty() && items[0].key.digest_hi % kDisks == 1) {
      std::unique_lock<std::mutex> lock(mu);
      cv.wait(lock, [&] { return open; });
    }
    return std::vector<Status>(items.size(), Status::kOk);
  });
  ASSERT_TRUE(rt.ok());
  const std::string v(4000, 'v');
  // Disk 1's keys until RAM refuses them: its share is 2/4 of the 80% mark.
  uint64_t id = 1;
  size_t admitted = 0;
  for (; id < 4 * 256; id += kDisks) {
    const auto r = rt.PutWriteBack(K(id), v.data(), v.size());
    if (r.disposition == RamTier::PutDisposition::kBackpressure) break;
    ASSERT_EQ(r.status, Status::kOk);
    ++admitted;
  }
  EXPECT_LT(id, 4u * 256) << "disk 1 was never backpressured";
  EXPECT_LE(rt.DirtyBytes(), 256u * 4096 * 80 / 100 / kDisks * 2 + 4096);
  // Disk 0's keys are still taken and written.
  for (uint64_t k = 0; k < 16; ++k) {
    const auto r = rt.PutWriteBack(K(4 * (1000 + k)), v.data(), v.size());
    EXPECT_EQ(r.disposition, RamTier::PutDisposition::kComplete) << k;
    EXPECT_EQ(r.status, Status::kOk) << k;
  }
  {
    std::lock_guard<std::mutex> lock(mu);
    open = true;
  }
  cv.notify_all();
  ASSERT_TRUE(WaitFor([&] { return rt.FlushBacklog() == 0; }));
  EXPECT_GT(admitted, 0u);
}

// A slow backing-index check (the disk store's lock held across a slow
// metadata write) must not stall the RAM hits of the shard it is asked for.
TEST(RamTier, ASlowPersistedCheckDoesNotStallRamHits) {
  std::mutex mu;
  std::condition_variable cv;
  bool open = false;
  std::atomic<bool> checking{false};
  RamTier::Options o = Opts(256 * 4096);
  o.is_persisted = [&](const BlockKey& key) {
    if (key.digest_hi != 999) return false;
    checking = true;
    std::unique_lock<std::mutex> lock(mu);
    cv.wait_for(lock, std::chrono::milliseconds(300), [&] { return open; });
    return false;
  };
  FlushSink sink;
  RamTier rt(o, sink.fn());
  ASSERT_TRUE(rt.ok());
  const std::string v(4000, 'h');
  ASSERT_TRUE(rt.Put(K(1), v.data(), v.size()));
  std::thread put([&] { rt.Put(K(999), v.data(), v.size()); });
  while (!checking.load()) std::this_thread::sleep_for(1ms);
  double worst_ms = 0;
  for (int i = 0; i < 20; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    RamTier::Hit h;
    ASSERT_TRUE(rt.GetPrep(K(1), 0, v.size(), &h));
    worst_ms = std::max(worst_ms, std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - t0)
                                      .count());
  }
  {
    std::lock_guard<std::mutex> lock(mu);
    open = true;
  }
  cv.notify_all();
  put.join();
  std::printf("[ timing ] slowest RAM hit during a slow check: %.3f ms\n",
              worst_ms);
  EXPECT_LT(worst_ms, 50.0) << "RAM hits waited for the disk index check";
}
