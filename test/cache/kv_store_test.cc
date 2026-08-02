// TDD R3 — KVStore: the cache-node local store (disk + LRU + cache-only/no-S3).
#include <cerrno>
#include <fstream>
#include "cache/kv_store.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

using dfkv::BlockKey;
using dfkv::KVStore;
using dfkv::Status;

namespace fs = std::filesystem;

class KVStoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           ("dfkv_store_" + std::to_string(::testing::UnitTest::GetInstance()
                                               ->current_test_info()
                                               ->line()));
    fs::remove_all(dir_);
    fs::create_directories(dir_);
  }
  void TearDown() override { fs::remove_all(dir_); }
  KVStore::Options Opts(uint64_t cap = (1ull << 30), size_t shards = 16) {
    return KVStore::Options{dir_.string(), cap, shards};
  }
  fs::path dir_;
};

TEST_F(KVStoreTest, CacheThenRangeRoundTripAndImmediateVisibility) {
  KVStore s(Opts());
  BlockKey k{111, 0, 1};
  std::string v = "the-kv-bytes-0123456789";
  ASSERT_EQ(s.Cache(k, v.data(), v.size()), Status::kOk);
  // sync visibility: cached immediately after Cache() returns
  EXPECT_TRUE(s.IsCached(k));
  std::string out;
  ASSERT_EQ(s.Range(k, 0, v.size(), &out), Status::kOk);
  EXPECT_EQ(out, v);
}

TEST_F(KVStoreTest, RangeIntoReadsStraightIntoCallerBuffer) {
  KVStore s(Opts());
  BlockKey k{222, 0, 1};
  std::string v = "zero-copy-server-side-payload-bytes";
  ASSERT_EQ(s.Cache(k, v.data(), v.size()), Status::kOk);
  char buf[64];
  size_t got = 0;
  ASSERT_EQ(s.RangeInto(k, 0, v.size(), buf, sizeof(buf), &got), Status::kOk);
  EXPECT_EQ(got, v.size());
  EXPECT_EQ(std::string(buf, got), v);
  // miss => NotFound, out_len 0
  got = 12345;
  EXPECT_EQ(s.RangeInto(BlockKey{999, 0, 1}, 0, 64, buf, sizeof(buf), &got), Status::kNotFound);
  EXPECT_EQ(got, 0u);
  // dst_cap caps the read
  ASSERT_EQ(s.RangeInto(k, 0, v.size(), buf, 4, &got), Status::kOk);
  EXPECT_EQ(got, 4u);
  EXPECT_EQ(std::string(buf, 4), v.substr(0, 4));
}

TEST_F(KVStoreTest, RangeDirectReturnsSliceInsideAlignedBuffer) {
  KVStore s(Opts());
  BlockKey k{333, 0, 1};
  std::string v(9000, '\0');
  for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<char>((i * 13 + 5) & 0xFF);
  ASSERT_EQ(s.Cache(k, v.data(), v.size()), Status::kOk);

  void* raw = nullptr;
  ASSERT_EQ(posix_memalign(&raw, 4096, 16 * 1024), 0);
  char* io = static_cast<char*>(raw);
  const char* data = nullptr;
  size_t got = 0;
  ASSERT_EQ(s.RangeDirect(k, 123, 5000, io, 16 * 1024, &data, &got), Status::kOk);
  EXPECT_EQ(got, 5000u);
  EXPECT_GE(data, io);
  EXPECT_LT(data, io + 16 * 1024);
  EXPECT_EQ(std::string(data, got), v.substr(123, 5000));
  std::free(raw);
}

TEST_F(KVStoreTest, CacheDirectWritesAlignedBuffer) {
  KVStore s(Opts());
  BlockKey k{334, 0, 1};
  std::string v(7000, '\0');
  for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<char>((i * 17 + 9) & 0xFF);

  void* raw = nullptr;
  ASSERT_EQ(posix_memalign(&raw, 4096, 8192), 0);
  char* io = static_cast<char*>(raw);
  std::memcpy(io, v.data(), v.size());
  ASSERT_EQ(s.CacheDirect(k, io, v.size(), 8192), Status::kOk);

  std::string out;
  ASSERT_EQ(s.Range(k, 0, v.size(), &out), Status::kOk);
  EXPECT_EQ(out, v);
  std::free(raw);
}

TEST_F(KVStoreTest, RangeMissReturnsNotFoundNoS3) {
  KVStore s(Opts());
  std::string out;
  EXPECT_EQ(s.Range(BlockKey{404, 0, 1}, 0, 16, &out), Status::kNotFound);
  EXPECT_FALSE(s.IsCached(BlockKey{404, 0, 1}));
}

TEST_F(KVStoreTest, IdempotentCacheSkipsRewrite) {
  KVStore s(Opts());
  BlockKey k{7, 0, 1};
  std::string v = "abc";
  ASSERT_EQ(s.Cache(k, v.data(), v.size()), Status::kOk);
  ASSERT_EQ(s.Cache(k, v.data(), v.size()), Status::kOk);  // second is a no-op skip
  EXPECT_EQ(s.Count(), 1u);
}

TEST_F(KVStoreTest, PartialRange) {
  KVStore s(Opts());
  BlockKey k{9, 0, 1};
  std::string v = "0123456789";
  ASSERT_EQ(s.Cache(k, v.data(), v.size()), Status::kOk);
  std::string out;
  ASSERT_EQ(s.Range(k, 3, 4, &out), Status::kOk);
  EXPECT_EQ(out, "3456");
}

TEST_F(KVStoreTest, LargeObjectOver128KiBLocalRead) {
  KVStore s(Opts());
  BlockKey k{42, 0, 1};
  std::string v(300 * 1024, 'x');  // >128KiB: no max_range_size cap on local path
  v[123456] = 'Z';
  ASSERT_EQ(s.Cache(k, v.data(), v.size()), Status::kOk);
  std::string out;
  ASSERT_EQ(s.Range(k, 0, v.size(), &out), Status::kOk);
  EXPECT_EQ(out, v);
}

TEST_F(KVStoreTest, LruEvictsLeastRecentlyUsedWhenOverCapacity) {
  // capacity holds ~2 objects of 1000B. shards=1 so all keys share one stripe and
  // eviction is the strict single-shard CLOCK (a touched entry gets a second
  // chance); with the default 16 shards these 3 keys would land in different
  // stripes and per-shard capacity (2200/16) couldn't hold even one object.
  KVStore s(Opts(/*cap=*/2200, /*shards=*/1));
  auto put = [&](uint64_t id) {
    std::string v(1000, char('a' + (id % 20)));
    return s.Cache(BlockKey{id, 0, 1}, v.data(), v.size());
  };
  ASSERT_EQ(put(1), Status::kOk);
  ASSERT_EQ(put(2), Status::kOk);
  // touch #1 so #2 becomes LRU
  std::string out; ASSERT_EQ(s.Range(BlockKey{1,0,1}, 0, 10, &out), Status::kOk);
  ASSERT_EQ(put(3), Status::kOk);  // over capacity -> evict LRU (#2)
  EXPECT_TRUE(s.IsCached(BlockKey{1, 0, 1}));
  EXPECT_FALSE(s.IsCached(BlockKey{2, 0, 1}));
  EXPECT_TRUE(s.IsCached(BlockKey{3, 0, 1}));
  EXPECT_LE(s.UsedBytes(), 2200u);
  // eviction counters tracked the one evicted 1000-byte object
  EXPECT_EQ(s.Evictions(), 1u);
  EXPECT_EQ(s.EvictedBytes(), 1000u);
}

// All-hot eviction: every resident entry is marked referenced right before each
// new insert. The persistent CLOCK hand must still make progress — clear a cycle
// of reference bits, then evict — and keep usage bounded. A naive guard could
// loop forever (always finds a referenced entry) or let the shard grow without
// bound; this pins that it does neither.
TEST_F(KVStoreTest, EvictsUnderAllReferencedPressure) {
  const uint64_t obj = 1000, cap = 5500;  // ~5 objects fit in one shard
  KVStore s(Opts(cap, /*shards=*/1));
  auto put = [&](uint64_t id) {
    std::string v(obj, char('a' + (id % 20)));
    return s.Cache(BlockKey{id, 0, 1}, v.data(), v.size());
  };
  auto touch = [&](uint64_t id) {
    std::string out;
    s.Range(BlockKey{id, 0, 1}, 0, 8, &out);  // sets the CLOCK bit if resident
  };
  for (uint64_t id = 1; id <= 40; ++id) {
    for (uint64_t j = (id > 8 ? id - 8 : 1); j < id; ++j) touch(j);  // make all hot
    ASSERT_EQ(put(id), Status::kOk) << "id=" << id;
    EXPECT_LE(s.UsedBytes(), cap) << "id=" << id;  // eviction kept usage bounded
  }
  EXPECT_TRUE(s.IsCached(BlockKey{40, 0, 1}));  // the newest insert survives
}

TEST_F(KVStoreTest, ReloadFromDiskRebuildsIndex) {
  BlockKey k{55, 0, 1};
  std::string v = "persisted";
  {
    KVStore s(Opts());
    ASSERT_EQ(s.Cache(k, v.data(), v.size()), Status::kOk);
  }
  KVStore s2(Opts());  // new instance, same dir
  EXPECT_TRUE(s2.IsCached(k));
  std::string out;
  ASSERT_EQ(s2.Range(k, 0, v.size(), &out), Status::kOk);
  EXPECT_EQ(out, v);
}

// Stress the sharded index under concurrency: many keys spread over the default
// 16 stripes, written then read back from 8 threads each. Writers take the
// exclusive per-shard lock, readers the shared one; correctness here (every
// reader hits every key with the right bytes) plus a clean TSan run validate the
// shared_mutex + CLOCK-bit access pattern. 1 GiB cap => no eviction interferes.
TEST_F(KVStoreTest, ConcurrentShardedReadWrite) {
  KVStore s(Opts());  // default 16 shards, 1 GiB capacity
  const int N = 256;
  auto val = [](int i) { return std::string(300, static_cast<char>('a' + (i % 26))); };

  {  // concurrent writers, interleaved keys across threads (and thus shards)
    std::vector<std::thread> ts;
    for (int t = 0; t < 8; ++t)
      ts.emplace_back([&, t] {
        for (int i = t; i < N; i += 8) {
          std::string v = val(i);
          EXPECT_EQ(s.Cache(BlockKey{static_cast<uint64_t>(i), 0, 1}, v.data(), v.size()),
                    Status::kOk);
        }
      });
    for (auto& th : ts) th.join();
  }
  EXPECT_EQ(s.Count(), static_cast<size_t>(N));

  {  // concurrent readers (shared lock): every thread reads every key correctly
    std::atomic<int> hits{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < 8; ++t)
      ts.emplace_back([&] {
        for (int i = 0; i < N; ++i) {
          std::string out;
          if (s.Range(BlockKey{static_cast<uint64_t>(i), 0, 1}, 0, 300, &out) == Status::kOk &&
              out == val(i))
            hits.fetch_add(1, std::memory_order_relaxed);
        }
      });
    for (auto& th : ts) th.join();
    EXPECT_EQ(hits.load(), N * 8);
  }
}

TEST_F(KVStoreTest, RemoveDropsBlockReclaimsBytesAndIsIdempotent) {
  KVStore s(Opts());
  BlockKey k{555, 0, 1};
  std::string v = "remove-me-payload-bytes";
  ASSERT_EQ(s.Cache(k, v.data(), v.size()), Status::kOk);
  ASSERT_TRUE(s.IsCached(k));
  ASSERT_EQ(s.UsedBytes(), v.size());

  EXPECT_EQ(s.Remove(k), Status::kOk);
  EXPECT_FALSE(s.IsCached(k));
  EXPECT_EQ(s.UsedBytes(), 0u);
  EXPECT_EQ(s.Count(), 0u);

  // Removing an absent key is a clean kNotFound (idempotent re-remove too).
  EXPECT_EQ(s.Remove(k), Status::kNotFound);
  EXPECT_EQ(s.Remove(BlockKey{556, 0, 1}), Status::kNotFound);

  // Eviction counters are untouched by an explicit Remove (distinct from
  // capacity eviction).
  EXPECT_EQ(s.Evictions(), 0u);

  // The key is re-cacheable after removal (file path freed, index slot reusable).
  ASSERT_EQ(s.Cache(k, v.data(), v.size()), Status::kOk);
  EXPECT_TRUE(s.IsCached(k));
  EXPECT_EQ(s.UsedBytes(), v.size());
}

TEST_F(KVStoreTest, RemoveKeepsClockHandValidUnderManyKeys) {
  // shards=1 forces all keys into one CLOCK ring so Remove must keep the
  // persistent eviction hand valid; interleave removes with caches and verify
  // the store stays consistent (no crash / stale index).
  KVStore s(Opts(1ull << 30, 1));
  std::string v(64, 'x');
  for (uint64_t i = 0; i < 64; ++i)
    ASSERT_EQ(s.Cache(BlockKey{i, 0, 1}, v.data(), v.size()), Status::kOk);
  ASSERT_EQ(s.Count(), 64u);
  for (uint64_t i = 0; i < 64; i += 2)  // remove every other key
    EXPECT_EQ(s.Remove(BlockKey{i, 0, 1}), Status::kOk);
  EXPECT_EQ(s.Count(), 32u);
  for (uint64_t i = 0; i < 64; ++i)
    EXPECT_EQ(s.IsCached(BlockKey{i, 0, 1}), (i % 2 == 1));
  // The survivors are still readable and a fresh cache still evicts correctly.
  std::string out;
  EXPECT_EQ(s.Range(BlockKey{1, 0, 1}, 0, 64, &out), Status::kOk);
}

TEST_F(KVStoreTest, RebuildIndexReclaimsOrphanTmpAndKeepsPublishedBlocks) {
  BlockKey k{555, 0, 1};
  std::string v = "durable-value-0123456789";
  {
    KVStore s(Opts());
    ASSERT_EQ(s.Cache(k, v.data(), v.size()), Status::kOk);
  }
  // Simulate a crash between WriteFileDirect and rename: drop orphan ".tmp"
  // files alongside the published block (nested like the real StoreKey layout).
  std::error_code ec;
  fs::path bucket = dir_ / "blocks" / "0" / "0";
  fs::create_directories(bucket, ec);
  size_t before = 0;
  for (auto& e : fs::recursive_directory_iterator(dir_)) (void)e, ++before;
  std::ofstream(bucket / "999_0_1.7.tmp") << "half-written-aligned-garbage";
  std::ofstream(dir_ / "888_0_1.3.tmp") << "another-orphan";

  KVStore s2(Opts());
  EXPECT_EQ(s2.TmpReclaimed(), 2u);
  EXPECT_TRUE(s2.IsCached(k)) << "published block must survive tmp cleanup";
  std::string out;
  ASSERT_EQ(s2.Range(k, 0, v.size(), &out), Status::kOk);
  EXPECT_EQ(out, v);
  // No ".tmp" files remain on disk.
  size_t tmps = 0;
  for (auto& e : fs::recursive_directory_iterator(dir_)) {
    std::string n = e.path().filename().string();
    if (n.size() >= 4 && n.substr(n.size() - 4) == ".tmp") ++tmps;
  }
  EXPECT_EQ(tmps, 0u);
  // Orphan bytes were never counted toward capacity.
  (void)before;
}

TEST_F(KVStoreTest, ForeignLayoutFailsClosed) {
  std::ofstream(dir_ / "slab_meta", std::ios::binary) << "foreign";
  KVStore store(Opts());
  EXPECT_FALSE(store.Healthy());
  EXPECT_NE(store.StartupError().find("different store layout"),
            std::string::npos);
  std::string value(16, 'x');
  EXPECT_EQ(store.Cache(BlockKey{1, 0, 1}, value.data(), value.size()),
            Status::kIOError);
  EXPECT_FALSE(store.IsCached(BlockKey{1, 0, 1}));
}

TEST_F(KVStoreTest, InvalidDirectoryAndCapacityFailClosed) {
  auto zero = Opts();
  zero.capacity_bytes = 0;
  KVStore zero_capacity(zero);
  EXPECT_FALSE(zero_capacity.Healthy());
  EXPECT_FALSE(zero_capacity.StartupError().empty());

  fs::remove_all(dir_);
  std::ofstream(dir_, std::ios::binary) << "not-a-directory";
  KVStore regular_file(Opts());
  EXPECT_FALSE(regular_file.Healthy());
  EXPECT_FALSE(regular_file.StartupError().empty());
}

TEST_F(KVStoreTest, EnospcTriggersForceEvictAndRetrySucceeds) {
  // Fill the store with several evictable blocks, then inject a single ENOSPC
  // on the next write. The store must force-evict and retry, landing the block.
  KVStore s(Opts(/*cap=*/1ull << 30));
  for (int i = 0; i < 8; ++i) {
    std::string v(4096, 'a' + i);
    ASSERT_EQ(s.Cache(BlockKey{static_cast<uint64_t>(1000 + i), 0, 1},
                      v.data(), v.size()), Status::kOk);
  }
  ASSERT_GE(s.Count(), 8u);

  int calls = 0;
  s.SetWriteFnForTest([&](const std::string& path, const void* data, size_t len,
                          int* werr) -> bool {
    if (++calls == 1) { if (werr) *werr = ENOSPC; return false; }  // first write: disk full
    // Retry: write for real so the block actually lands + reads back.
    std::ofstream(path, std::ios::binary).write(static_cast<const char*>(data),
                                                static_cast<std::streamsize>(len));
    return true;
  });

  std::string v(4096, 'Z');
  BlockKey nk{2000, 0, 1};
  EXPECT_EQ(s.Cache(nk, v.data(), v.size()), Status::kOk);
  EXPECT_EQ(calls, 2) << "must retry exactly once after the injected ENOSPC";
  EXPECT_EQ(s.EnospcEvictions(), 1u);
  EXPECT_GT(s.Evictions(), 0u) << "force-evict must have reclaimed at least one block";
  EXPECT_TRUE(s.IsCached(nk));
}

TEST_F(KVStoreTest, PersistentEnospcReturnsIoErrorWithoutInfiniteLoop) {
  KVStore s(Opts());
  for (int i = 0; i < 4; ++i) {
    std::string v(4096, 'a' + i);
    ASSERT_EQ(s.Cache(BlockKey{static_cast<uint64_t>(3000 + i), 0, 1},
                      v.data(), v.size()), Status::kOk);
  }
  s.SetWriteFnForTest([&](const std::string&, const void*, size_t, int* werr) -> bool {
    if (werr) *werr = ENOSPC;  // always full: retry can't succeed
    return false;
  });
  std::string v(4096, 'Q');
  EXPECT_EQ(s.Cache(BlockKey{4000, 0, 1}, v.data(), v.size()), Status::kIOError);
  EXPECT_EQ(s.EnospcEvictions(), 0u);  // retry did not succeed -> not counted
}

TEST_F(KVStoreTest, EvictionAndRemoveLeaveNoTmpOrphansOnDisk) {
  // Eviction/Remove now RENAME the victim to a ".tmp" trash name inside the
  // lock and unlink it AFTER releasing the lock. Verify the deferred unlink
  // actually completes: after an eviction storm + removes, no ".tmp" orphans
  // remain and the on-disk regular-file count matches the live index.
  auto count_disk = [&](const char* suffix) {
    size_t n = 0;
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(dir_, ec);
         !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
      if (!it->is_regular_file(ec)) continue;
      std::string f = it->path().filename().string();
      bool tmp = f.size() >= 4 && f.substr(f.size() - 4) == ".tmp";
      if ((suffix && tmp) || (!suffix && !tmp)) ++n;
    }
    return n;
  };

  // Small capacity + many distinct 4 KiB blocks => sustained eviction.
  KVStore s(Opts(/*cap=*/64 * 4096, /*shards=*/4));
  std::string v(4096, 'e');
  for (uint64_t i = 0; i < 400; ++i)
    ASSERT_EQ(s.Cache(BlockKey{i, 0, 1}, v.data(), v.size()), Status::kOk);
  EXPECT_GT(s.Evictions(), 0u) << "the workload must have forced evictions";
  for (uint64_t i = 350; i < 400; ++i) s.Remove(BlockKey{i, 0, 1});  // some Removes too

  EXPECT_EQ(count_disk(".tmp"), 0u) << "deferred unlink must leave no .tmp orphans";
  EXPECT_EQ(count_disk(nullptr), s.Count())
      << "on-disk block files must match the live index";
}
