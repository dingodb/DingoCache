// RamTier: write-through RAM hot tier on SlabAllocator. Hermetic (no RDMA/disk;
// the flush sink is an injected callback). Covers read-after-write, the
// RAM_ONLY->DURABLE state machine via pins, send-in-flight pin blocking
// eviction, backpressure bypass when flush stalls, flush retry/drop, sub-range
// GetPrep, Remove semantics, and a concurrent TSan stress.
#include "cache/ram_tier.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

using dfkv::BlockKey;
using dfkv::RamTier;
using namespace std::chrono_literals;

namespace {
BlockKey K(uint64_t id) { return BlockKey{id, 0}; }

// A controllable flush sink: records flushed keys, can be gated (block until
// opened) and can be made to fail.
struct FlushSink {
  std::mutex m;
  std::condition_variable cv;
  std::set<std::string> flushed;
  bool gate_open = true;      // when false, flush blocks
  bool fail = false;          // when true, flush returns false
  int calls = 0;

  RamTier::FlushFn fn() {
    return [this](const BlockKey& k, char*, size_t, size_t) {
      std::unique_lock<std::mutex> lk(m);
      ++calls;
      cv.wait(lk, [this] { return gate_open; });
      if (fail) return false;
      flushed.insert(k.Filename());
      cv.notify_all();
      return true;
    };
  }
  void open() { { std::lock_guard<std::mutex> lk(m); gate_open = true; } cv.notify_all(); }
  void close() { std::lock_guard<std::mutex> lk(m); gate_open = false; }
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
  EXPECT_EQ(rt.Hits(), 1u);
  rt.Release(h.token);
  sink.open();
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
  // block sizes, many blocks each). 6 classes * kStripeWays(8) = 48 extents of
  // headroom vs 128 available. Each value < the 2 MiB extent.
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
    rt.Release(h.token);
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

TEST(RamTier, SubRangeGetPrep) {
  FlushSink sink;
  RamTier rt(Opts(64 * 4096), sink.fn());
  std::string v = "0123456789";
  ASSERT_TRUE(rt.Put(K(3), v.data(), v.size()));
  RamTier::Hit h;
  ASSERT_TRUE(rt.GetPrep(K(3), 3, 4, &h));
  EXPECT_EQ(std::string(h.ptr, h.len), "3456");
  rt.Release(h.token);
  // offset past end -> zero-length hit (still a hit)
  ASSERT_TRUE(rt.GetPrep(K(3), 100, 10, &h));
  EXPECT_EQ(h.len, 0u);
  rt.Release(h.token);
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
  rt.Release(h.token);
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

TEST(RamTier, RemoveOnlyDropsDurableIdle) {
  FlushSink sink;
  sink.close();
  RamTier rt(Opts(8 * 4096), sink.fn());
  std::string v(200, 'r');
  ASSERT_TRUE(rt.Put(K(40), v.data(), v.size()));
  EXPECT_FALSE(rt.Remove(K(40))) << "non-durable (flushing) -> declined";
  sink.open();
  EXPECT_TRUE(WaitFor([&] { return rt.Flushed() == 1u; }));
  RamTier::Hit h;
  ASSERT_TRUE(rt.GetPrep(K(40), 0, v.size(), &h));  // now send-in-flight
  EXPECT_FALSE(rt.Remove(K(40))) << "in-flight -> declined";
  rt.Release(h.token);
  EXPECT_TRUE(rt.Remove(K(40))) << "durable + idle -> removed";
  EXPECT_FALSE(rt.Contains(K(40)));
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
          rt.Release(h.token);
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
  RamTier::Options o = Opts(256 * 4096);
  o.flush_threads = 1;  // single worker => deterministic batching
  RamTier rt(o, nullptr);
  rt.set_flush_batch([&](const std::vector<RamTier::FlushItem>& items) {
    std::lock_guard<std::mutex> lk(m);
    batch_sizes.push_back(items.size());
    std::vector<bool> ok(items.size(), true);
    for (size_t i = 0; i < items.size(); ++i)
      if (fail_key >= 0 &&
          items[i].key.digest_hi ==
              static_cast<uint64_t>(fail_key.load()))
        ok[i] = false;
    return ok;
  });
  ASSERT_TRUE(rt.ok());
  std::string v(4000, 'b');
  for (uint64_t i = 0; i < 40; ++i) ASSERT_TRUE(rt.Put(K(700 + i), v.data(), v.size()));
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
  o.slot_granularity = 4096;
  o.flush_threads = 8;
  RamTier t(o, sink.fn());
  ASSERT_TRUE(t.ok());
  EXPECT_EQ(t.shards(), 8u);

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
    t.Release(h.token);
  }
  // Durable + idle -> removable, on whichever shard the key lives.
  for (int i = 0; i < N; i += 7) EXPECT_TRUE(t.Remove(K(1000 + i))) << i;
  ::unsetenv("DFKV_RAM_TIER_SHARDS");
  ::unsetenv("DFKV_RAM_TIER_EXTENT_BYTES");
}

TEST(RamTier, ShardCountCanExceedLegacyCapOf16) {
  // Phase 10: kTokenShardBits 4->6 lifts the cap 16->64 so read/write-heavy
  // multi-connection deployments can shard past the old ceiling. 32 shards must
  // be honored (not clamped to 16), and shard-encoded tokens still round-trip.
  // 1 MiB extents (the floor) x 1 GiB = 1024 extents => 32 shards sustains the
  // >=32 extents/shard rule (32*32=1024). The arena is mmap'd lazily and the
  // test touches only ~2 MiB, so physical RAM stays tiny despite the 1 GiB size.
  ::setenv("DFKV_RAM_TIER_EXTENT_BYTES", "1048576", 1);
  ::setenv("DFKV_RAM_TIER_SHARDS", "32", 1);
  FlushSink sink;
  RamTier::Options o;
  o.bytes = 1024ull << 20;   // 1024 x 1 MiB extents
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
    t.Release(h.token);
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
  rt.Release(h.token);
  // No flush-pin held: Remove (which declines on any pin) succeeds at once.
  EXPECT_TRUE(rt.Remove(K(90)));
  EXPECT_FALSE(rt.Contains(K(90)));
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
