// RamTier wiring into KvNodeServer: with DFKV_RAM_TIER=1, PUT is write-through
// to RAM (sync-visible, async-flushed) and GET is served from RAM; disabled by
// default the node behaves exactly as before. End-to-end over the TCP transport.
#include "cache/kv_node_server.h"
#include "cache/disk_slab_store.h"
#include "cache/kv_store.h"
#include "client/key_map.h"
#include "transport/tcp_transport.h"
#include "transport/transport.h"
#include "transport/wire.h"

#include <gtest/gtest.h>

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <future>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <atomic>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

using namespace std::chrono_literals;

namespace dfkv {
// Named like the store's test peer (its friend). BreakStateFd breaks the
// state file descriptor, so the next write of slab_state fails.
class DiskSlabStoreTestPeer {
 public:
  static void BreakStateFd(DiskSlabStore* store) {
    if (store->state_fd_ >= 0) ::close(store->state_fd_);
    store->state_fd_ = -1;
  }
  // Fail the store's writes: `record` is the metadata failure that fails a
  // store for good, `payload` a write that fails and may work again.
  static void SetRecordWriteHook(DiskSlabStore* store,
                                 std::function<bool()> hook) {
    store->record_write_hook_for_test_ = std::move(hook);
  }
  static void SetPayloadWriteHook(DiskSlabStore* store,
                                  std::function<bool()> hook) {
    store->payload_write_hook_for_test_ = std::move(hook);
  }
  // A read or write in flight holds its slot like this.
  static bool PinSlot(DiskSlabStore* store, const BlockKey& key) {
    std::lock_guard<std::mutex> lock(store->mu_);
    return store->alloc_->Pin(key);
  }
  static bool UnpinSlot(DiskSlabStore* store, const BlockKey& key) {
    std::lock_guard<std::mutex> lock(store->mu_);
    return store->alloc_->Unpin(key);
  }
};
}  // namespace dfkv

namespace {
// A slab store that opens like any other and whose activation fails where a
// failing disk would make it fail: writing the dirty state.
class CannotBeActivated : public DiskSlabStore {
 public:
  using DiskSlabStore::DiskSlabStore;
  bool Activate() override {
    DiskSlabStoreTestPeer::BreakStateFd(this);
    return DiskSlabStore::Activate();
  }
};
}  // namespace

namespace dfkv {
class KvNodeServerWiringTestPeer {
 public:
  static std::unique_ptr<KvNodeServer> Create(DiskCacheGroup::Options options) {
    return std::unique_ptr<KvNodeServer>(new KvNodeServer(std::move(options)));
  }
  static std::unique_ptr<KvNodeServer> Create(
      DiskCacheGroup::Options options, const std::atomic<bool>* cancel_start) {
    return std::unique_ptr<KvNodeServer>(
        new KvNodeServer(std::move(options), cancel_start));
  }
  static void SetLimit(KvNodeServer& server, size_t limit) {
    server.put_busy_limit_ = limit;
  }
  static void SetCoalescing(KvNodeServer& server, bool enabled) {
    server.coalesce_enabled_ = enabled;
  }
  static bool Begin(KvNodeServer& server) { return server.TryBeginPut(); }
  static void End(KvNodeServer& server) { server.EndPut(); }
  static void SetRam(KvNodeServer& server, RamTier::FlushFn flush,
                     bool ram_ack = false, uint32_t watermark_pct = 80,
                     uint32_t put_wait_ms = 3000,
                     uint32_t backpressure_wait_ms = 0) {
    RamTier::Options options;
    options.put_wait_ms = put_wait_ms;
    options.ack_backpressure_wait_ms = backpressure_wait_ms;
    options.bytes = 16 * 4096;
    options.large_reserve_bytes = 0;
    options.reclaim_interval_ms = 0;
    options.ack_high_watermark_pct = watermark_pct;
    options.is_persisted = [&server](const BlockKey& key) {
      return server.group_.IsCached(key);
    };
    options.disk_generation = [&server](const BlockKey& key) {
      return server.DiskGeneration(key);
    };
    server.ram_ = std::make_unique<RamTier>(options, std::move(flush));
    server.ram_write_back_ = true;
    server.ram_ack_enabled_ = ram_ack;
  }
  static void CancelPutWaits(KvNodeServer& server) {
    server.ram_->CancelPutWaits();
  }
  static auto LockPut(KvNodeServer& server, const BlockKey& key) {
    return std::make_unique<KvNodeServer::PutKeyLock>(server.LockPutKey(key));
  }
  static size_t PublicationBucket(const KvNodeServer& server,
                                  const BlockKey& key) {
    return BlockKeyMixHash{}(key) % server.memory_key_locks_.size();
  }
  static bool HasWriteMemory(const KvNodeServer& server) {
    return server.write_memory_ != nullptr;
  }
  static uint64_t RemovedWithinAgesRequests(const KvNodeServer& server) {
    return server.removed_within_ages_requests_.load(
        std::memory_order_relaxed);
  }
  static void ExpectPublicationWait(KvNodeServer& server,
                                    const BlockKey& key) {
    const size_t bucket = PublicationBucket(server, key);
    {
      std::lock_guard<std::mutex> lk(server.memory_key_test_mu_);
      server.memory_key_arrived_bucket_for_test_ =
          server.memory_key_locks_.size();
      server.memory_key_expected_bucket_for_test_.store(
          bucket, std::memory_order_relaxed);
    }
    server.memory_key_test_cv_.notify_all();
  }
  static bool WaitPublicationWait(KvNodeServer& server,
                                  const BlockKey& key) {
    const size_t bucket = PublicationBucket(server, key);
    std::unique_lock<std::mutex> lk(server.memory_key_test_mu_);
    return server.memory_key_test_cv_.wait_for(
        lk, 2s, [&] { return server.memory_key_arrived_bucket_for_test_ == bucket; });
  }
  static void SetWriteBack(KvNodeServer& server, bool write_back) {
    server.ram_write_back_ = write_back;
  }
  static bool Resident(KvNodeServer& server, const BlockKey& key) {
    return server.ram_->Contains(key);
  }
  static Status Persist(KvNodeServer& server, const BlockKey& key,
                        const char* data, size_t len) {
    return server.group_.Cache(key, data, len);
  }
  static Status DiskRead(KvNodeServer& server, const BlockKey& key,
                         std::string* out) {
    return server.group_.Range(key, 0, 0, out);
  }
  static bool Drain(KvNodeServer& server) {
    return server.ram_->WaitForDrain(2s);
  }
  static bool DropRam(KvNodeServer& server, const BlockKey& key) {
    return server.ram_->Remove(key);
  }
  static bool RamStopping(KvNodeServer& server) {
    return server.ram_->background_stopping();
  }
  static size_t ParkedPuts(KvNodeServer& server) {
    return server.ram_->ParkedPuts();
  }
  static size_t PutKeyStripes() { return KvNodeServer::kPutKeyStripes; }
  static uint64_t RamDirty(KvNodeServer& server) {
    return server.ram_->DirtyObjects();
  }
  static uint64_t UncleanResets(KvNodeServer& server) {
    return server.group_.SlabStats().unclean_resets;
  }
  static uint64_t WarmRestarts(KvNodeServer& server) {
    return server.group_.SlabStats().warm_restarts;
  }
  static size_t PendingRetains(KvNodeServer& server) {
    std::lock_guard<std::mutex> lk(server.pending_retain_mu_);
    return server.pending_retain_.size();
  }
  static uint64_t DrainLimitSeconds(KvNodeServer& server) {
    return server.shutdown_drain_timeout_s_;
  }
};
}  // namespace dfkv

namespace {
// Count actual backend entry, including failed writes; absence of a file alone
// would not prove a terminal rejection avoided the direct-disk handler.
class CountingStore : public KVStore {
 public:
  CountingStore(const std::string& path, uint64_t capacity,
                std::atomic<size_t>& calls)
      : KVStore(Options{path, capacity}), calls_(calls) {}
  Status Cache(const BlockKey& key, const void* data, size_t len) override {
    calls_.fetch_add(1, std::memory_order_relaxed);
    return KVStore::Cache(key, data, len);
  }
  Status CacheDirect(const BlockKey& key, char* data, size_t len,
                     size_t cap) override {
    calls_.fetch_add(1, std::memory_order_relaxed);
    return KVStore::CacheDirect(key, data, len, cap);
  }
  Status RangeDirect(const BlockKey& key, uint64_t offset, uint64_t length,
                     char* io_buf, size_t io_cap, const char** out_data,
                     size_t* out_len, size_t* value_len) override {
    direct_reads.fetch_add(1, std::memory_order_relaxed);
    return KVStore::RangeDirect(key, offset, length, io_buf, io_cap, out_data,
                                out_len, value_len);
  }
  std::atomic<size_t> direct_reads{0};

 private:
  std::atomic<size_t>& calls_;
};

struct FlushBarrier {
  std::mutex mutex;
  std::condition_variable cv;
  bool entered = false;
  bool released = false;
  void Block() {
    std::unique_lock<std::mutex> lock(mutex);
    entered = true;
    cv.notify_all();
    cv.wait(lock, [&] { return released; });
  }
  bool AwaitEntry() {
    std::unique_lock<std::mutex> lock(mutex);
    return cv.wait_for(lock, 2s, [&] { return entered; });
  }
  void Open() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      released = true;
    }
    cv.notify_all();
  }
};

struct OpenFlushOnExit {
  FlushBarrier& barrier;
  ~OpenFlushOnExit() { barrier.Open(); }
};

class BlockingBypassStore : public KVStore {
 public:
  BlockingBypassStore(const std::string& path, uint64_t capacity,
                      FlushBarrier& barrier, bool block_remove = false)
      : KVStore(Options{path, capacity}), barrier_(barrier),
        block_remove_(block_remove) {}
  Status Cache(const BlockKey& key, const void* data, size_t len) override {
    if (len > 64 * 1024) barrier_.Block();
    return KVStore::Cache(key, data, len);
  }
  Status CacheDirect(const BlockKey& key, char* data, size_t len,
                     size_t cap) override {
    if (len > 64 * 1024) barrier_.Block();
    return KVStore::CacheDirect(key, data, len, cap);
  }
  Status Remove(const BlockKey& key) override {
    if (block_remove_) barrier_.Block();
    return KVStore::Remove(key);
  }

 private:
  FlushBarrier& barrier_;
  bool block_remove_;
};

// Extract a single-sample counter/gauge value from Prometheus text (line
// "<name>{...} <v>" or "<name> <v>"); returns -1 if absent.
long MetricVal(const std::string& text, const std::string& name) {
  size_t ls = 0;
  while (ls < text.size()) {
    size_t le = text.find('\n', ls);
    if (le == std::string::npos) le = text.size();
    const std::string line = text.substr(ls, le - ls);
    ls = le + 1;
    if (line.empty() || line[0] == '#') continue;                 // skip HELP/TYPE
    if (line.compare(0, name.size(), name) != 0) continue;        // must start with name
    char after = line.size() > name.size() ? line[name.size()] : '\0';
    if (after != '{' && after != ' ') continue;                  // exact metric, not a prefix
    size_t sp = line.rfind(' ');
    if (sp != std::string::npos) return std::stol(line.substr(sp + 1));
  }
  return -1;
}

std::unique_ptr<KvNodeServer> Start(const fs::path& dir, std::string* addr) {
  fs::remove_all(dir);
  fs::create_directories(dir);
  auto s = std::make_unique<KvNodeServer>(dir.string(), 1ull << 30);
  EXPECT_EQ(s->Start(0), Status::kOk);
  *addr = "127.0.0.1:" + std::to_string(s->port());
  return s;
}

template <class F>
bool WaitFor(F pred) {
  for (int i = 0; i < 2000; ++i) {
    if (pred()) return true;
    std::this_thread::sleep_for(1ms);
  }
  return pred();
}
}  // namespace

TEST(RamTierWiring, WriteThroughAndServeFromRam) {
  ::setenv("DFKV_RAM_TIER", "1", 1);
  ::setenv("DFKV_RAM_TIER_BYTES", "16777216", 1);  // 16 MiB arena
  std::string addr;
  auto dir = fs::temp_directory_path() / "dfkv_ramwire_a";
  auto s = Start(dir, &addr);
  TcpTransport t;

  // PUT 20 blocks, then GET each back -- read-after-write straight from RAM.
  for (int i = 0; i < 20; ++i) {
    std::string v = "ram-block-" + std::to_string(i) + std::string(200, 'z');
    ASSERT_EQ(t.Cache(addr, ToBlockKey("test/model", "k" + std::to_string(i)), v.data(), v.size()),
              Status::kOk) << i;
  }
  for (int i = 0; i < 20; ++i) {
    std::string v = "ram-block-" + std::to_string(i) + std::string(200, 'z');
    std::string out;
    ASSERT_EQ(t.Range(addr, ToBlockKey("test/model", "k" + std::to_string(i)), 0, v.size(), &out),
              Status::kOk) << i;
    EXPECT_EQ(out, v) << i;
  }

  std::string m = s->MetricsText();
  EXPECT_GE(MetricVal(m, "dfkv_ram_put_total"), 20);
  EXPECT_GE(MetricVal(m, "dfkv_ram_hit_total"), 20);
  EXPECT_EQ(MetricVal(m, "dfkv_ram_miss_total"), 0);
  // The async flusher persists every block to disk in the background.
  EXPECT_TRUE(WaitFor([&] { return MetricVal(s->MetricsText(), "dfkv_ram_flushed_total") >= 20; }));

  s.reset();  // stop the flusher before removing the dir (no write race)
  ::unsetenv("DFKV_RAM_TIER");
  ::unsetenv("DFKV_RAM_TIER_BYTES");
  fs::remove_all(dir);
}

TEST(RamTierWiring, SurvivesAsMissWhenRangePastEnd) {
  ::setenv("DFKV_RAM_TIER", "1", 1);
  ::setenv("DFKV_RAM_TIER_BYTES", "8388608", 1);
  std::string addr;
  auto dir = fs::temp_directory_path() / "dfkv_ramwire_b";
  auto s = Start(dir, &addr);
  TcpTransport t;
  std::string v = "0123456789";
  ASSERT_EQ(t.Cache(addr, ToBlockKey("test/model", "p"), v.data(), v.size()), Status::kOk);
  std::string out;
  ASSERT_EQ(t.Range(addr, ToBlockKey("test/model", "p"), 3, 4, &out), Status::kOk);  // RAM sub-range
  EXPECT_EQ(out, "3456");
  // absent key -> miss even with RAM on
  EXPECT_EQ(t.Range(addr, ToBlockKey("test/model", "absent"), 0, 4, &out), Status::kNotFound);
  EXPECT_GE(MetricVal(s->MetricsText(), "dfkv_ram_miss_total"), 1);

  s.reset();  // stop the flusher before removing the dir (no write race)
  ::unsetenv("DFKV_RAM_TIER");
  ::unsetenv("DFKV_RAM_TIER_BYTES");
  fs::remove_all(dir);
}

TEST(RamTierWiring, ExistAndRemoveSeeRam) {
  ::setenv("DFKV_RAM_TIER", "1", 1);
  ::setenv("DFKV_RAM_TIER_BYTES", "8388608", 1);
  std::string addr;
  auto dir = fs::temp_directory_path() / "dfkv_ramwire_d";
  auto s = Start(dir, &addr);
  TcpTransport t;
  std::string v(300, 'e');
  ASSERT_EQ(t.Cache(addr, ToBlockKey("test/model", "live"), v.data(), v.size()), Status::kOk);
  bool e = false;
  ASSERT_EQ(t.Exist(addr, ToBlockKey("test/model", "live"), &e), Status::kOk);
  EXPECT_TRUE(e) << "a RAM-resident block must report as existing";
  // Let it flush to disk so Remove targets a durable block, then remove it.
  EXPECT_TRUE(WaitFor([&] { return MetricVal(s->MetricsText(), "dfkv_ram_flushed_total") >= 1; }));
  ASSERT_EQ(t.Remove(addr, ToBlockKey("test/model", "live")), Status::kOk);
  ASSERT_EQ(t.Exist(addr, ToBlockKey("test/model", "live"), &e), Status::kOk);
  EXPECT_FALSE(e) << "after Remove the block is gone from both tiers";
  std::string out;
  EXPECT_EQ(t.Range(addr, ToBlockKey("test/model", "live"), 0, v.size(), &out), Status::kNotFound);

  s.reset();  // stop the flusher before removing the dir (no write race)
  ::unsetenv("DFKV_RAM_TIER");
  ::unsetenv("DFKV_RAM_TIER_BYTES");
  fs::remove_all(dir);
}

TEST(RamTierWiring, PersistedValueSurvivesConflictingPutAfterRamEviction) {
  ::setenv("DFKV_RAM_TIER", "1", 1);
  ::setenv("DFKV_RAM_TIER_BYTES", "16777216", 1);
  for (const char* ack : {"ram", "disk"}) {
    ::setenv("DFKV_PUT_ACK_MODE", ack, 1);
    std::string addr;
    auto dir = fs::temp_directory_path() /
               (std::string("dfkv_ramwire_retained_") + ack);
    auto server = Start(dir, &addr);
    TcpTransport tcp;
    alignas(4096) char original[4096] = "original";
    alignas(4096) char conflicting[4096] = "modified";
    for (bool direct : {false, true}) {
      const BlockKey key = ToBlockKey("retained-disk", direct ? "direct" : "tcp");
      const auto put = [&](char* data) {
        return direct ? server->CacheDirectForKey(key, data, 8, sizeof(original))
                      : tcp.Cache(addr, key, data, 8);
      };
      ASSERT_EQ(put(original), Status::kOk);
      ASSERT_TRUE(KvNodeServerWiringTestPeer::Drain(*server));
      std::string disk;
      ASSERT_EQ(KvNodeServerWiringTestPeer::DiskRead(*server, key, &disk), Status::kOk);
      ASSERT_EQ(disk, "original");
      ASSERT_TRUE(KvNodeServerWiringTestPeer::DropRam(*server, key));
      ASSERT_FALSE(KvNodeServerWiringTestPeer::Resident(*server, key));

      ASSERT_EQ(put(conflicting), Status::kOk);
      ASSERT_TRUE(KvNodeServerWiringTestPeer::Drain(*server));
      std::string visible;
      ASSERT_EQ(tcp.Range(addr, key, 0, 8, &visible), Status::kOk);
      EXPECT_EQ(visible, "original") << ack << " direct=" << direct;
      ASSERT_EQ(KvNodeServerWiringTestPeer::DiskRead(*server, key, &disk), Status::kOk);
      EXPECT_EQ(disk, visible) << "RAM must not claim durability for different disk bytes";
    }
    server.reset();
    fs::remove_all(dir);
  }
  ::unsetenv("DFKV_PUT_ACK_MODE");
  ::unsetenv("DFKV_RAM_TIER_BYTES");
  ::unsetenv("DFKV_RAM_TIER");
}

TEST(RamTierWiring, CapacityBypassPreservesConcurrentDuplicateValue) {
  for (bool direct : {false, true}) {
    for (bool ram_ack : {false, true}) {
      auto dir = fs::temp_directory_path() /
          ("dfkv_ramwire_bypass_" + std::to_string(direct) +
           "_" + std::to_string(ram_ack));
      fs::remove_all(dir);
      fs::create_directories(dir);
      FlushBarrier disk, flush;
      DiskCacheGroup::Options options{{dir.string()}, 1ull << 30, "file"};
      options.engine_factory = [&](const std::string& path, uint64_t capacity) {
        return std::make_unique<BlockingBypassStore>(path, capacity, disk);
      };
      auto server = KvNodeServerWiringTestPeer::Create(std::move(options));
      auto* node = server.get();
      OpenFlushOnExit disk_cleanup{disk}, flush_cleanup{flush};
      KvNodeServerWiringTestPeer::SetLimit(*server, 0);
      KvNodeServerWiringTestPeer::SetRam(
          *server,
          [&, node](const BlockKey& key, char* data, size_t len, size_t) {
            flush.Block();
            return KvNodeServerWiringTestPeer::Persist(*node, key, data, len);
          }, ram_ack);
      const BlockKey key{8101, 0};
      alignas(4096) char original[128 * 1024], conflicting[4096];
      std::memset(original, 'a', sizeof(original));
      std::memset(conflicting, 'b', sizeof(conflicting));
      auto put = [&](char* value, size_t len) {
        if (direct)
          return server->CacheDirectForKey(key, value, len, len);
        std::string out;
        return server->ProcessRequestForKey(
            static_cast<uint8_t>(WireOp::kCache), key, 0, 0,
            value, len, &out);
      };
      Status first_status = Status::kInvalid, second_status = Status::kInvalid;
      std::thread first([&] { first_status = put(original, sizeof(original)); });
      const bool disk_entered = disk.AwaitEntry();
      EXPECT_TRUE(disk_entered);
      std::thread second([&] { second_status = put(conflicting, sizeof(conflicting)); });
      // Before the fix, the conflicting value is admitted while the oversized
      // owner is in disk I/O. Hold that flush until the owner commits first.
      // Correct serialization cannot enter this flush before disk is opened.
      flush.AwaitEntry();
      disk.Open();
      first.join();
      flush.Open();
      second.join();
      EXPECT_EQ(first_status, Status::kOk);
      EXPECT_EQ(second_status, Status::kOk);
      ASSERT_TRUE(KvNodeServerWiringTestPeer::Drain(*server));
      std::string persisted, visible;
      ASSERT_EQ(KvNodeServerWiringTestPeer::DiskRead(*server, key, &persisted),
                Status::kOk);
      ASSERT_EQ(server->ProcessRequestForKey(
                    static_cast<uint8_t>(WireOp::kRange), key, 0, 0,
                    nullptr, 0, &visible), Status::kOk);
      ASSERT_EQ(persisted.size(), sizeof(original));
      EXPECT_EQ(std::memcmp(persisted.data(), original, sizeof(original)), 0);
      EXPECT_EQ(visible, persisted) << "direct=" << direct
                                    << " ram_ack=" << ram_ack;
      server.reset();
      fs::remove_all(dir);
    }
  }
}

TEST(RamTierWiring, ColdDirectSliceDoesNotReplaceWholeValue) {
  std::string value(8192 + 317, '\0');
  for (size_t i = 0; i < value.size(); ++i)
    value[i] = static_cast<char>('!' + i % 90);
  for (bool coalesce : {false, true}) {
    for (size_t offset : {size_t{0}, size_t{73}, value.size() - 121}) {
      SCOPED_TRACE("coalesce=" + std::to_string(coalesce) +
                   " offset=" + std::to_string(offset));
      const auto dir = fs::temp_directory_path() /
          ("dfkv_direct_slice_" + std::to_string(coalesce) +
           "_" + std::to_string(offset));
      fs::remove_all(dir);
      fs::create_directories(dir);
      std::atomic<size_t> disk_puts{0};
      CountingStore* store = nullptr;
      DiskCacheGroup::Options options{{dir.string()}, 1ull << 30, "file"};
      options.engine_factory = [&](const std::string& path, uint64_t capacity) {
        auto engine = std::make_unique<CountingStore>(path, capacity, disk_puts);
        store = engine.get();
        return engine;
      };
      auto server = KvNodeServerWiringTestPeer::Create(std::move(options));
      KvNodeServerWiringTestPeer::SetCoalescing(*server, coalesce);
      KvNodeServerWiringTestPeer::SetRam(
          *server, [](const BlockKey&, char*, size_t, size_t) {
            return Status::kIOError;
          });
      const BlockKey key{8201, 0};
      // Seed disk directly: the first range is deterministically a cold read.
      ASSERT_EQ(KvNodeServerWiringTestPeer::Persist(
                    *server, key, value.data(), value.size()), Status::kOk);
      ASSERT_FALSE(KvNodeServerWiringTestPeer::Resident(*server, key));
      alignas(4096) char staging[16 * 1024];
      const char* data = nullptr;
      size_t bytes = 0, value_len = 0;
      const size_t length = offset == 0 ? 233 : 5001;
      ASSERT_EQ(server->RangeDirectForKey(
                    key, offset, length, staging, sizeof(staging),
                    &data, &bytes, &value_len), Status::kOk);
      ASSERT_NE(data, nullptr);
      EXPECT_EQ(std::string(data, bytes), value.substr(offset, length));
      EXPECT_EQ(value_len, value.size());
      EXPECT_EQ(store->direct_reads.load(), 1u);
      EXPECT_FALSE(KvNodeServerWiringTestPeer::Resident(*server, key))
          << "Before the fix, a cold slice was installed as the whole value";

      // A complete read must still return the original bytes and stored length,
      // including when the requested length is larger than the stored value.
      ASSERT_EQ(server->RangeDirectForKey(
                    key, 0, value.size() + 4096, staging, sizeof(staging),
                    &data, &bytes, &value_len), Status::kOk);
      ASSERT_NE(data, nullptr);
      EXPECT_EQ(bytes, value.size());
      EXPECT_EQ(std::string(data, bytes), value);
      EXPECT_EQ(value_len, value.size());
      EXPECT_EQ(store->direct_reads.load(), 2u);
      EXPECT_TRUE(KvNodeServerWiringTestPeer::Resident(*server, key));

      // Behavior, not just membership: promotion exposes the complete pinned
      // arena value and subsequent direct reads do not enter the disk backend.
      {
        PreparedRead pinned;
        ASSERT_TRUE(server->RamPinnedHitForKey(key, 0, value.size(), &pinned));
        EXPECT_EQ(pinned.payload_len(), value.size());
        EXPECT_EQ(pinned.value_len(), value.size());
        EXPECT_EQ(std::string(pinned.data(), pinned.payload_len()), value);
      }
      ASSERT_EQ(server->RangeDirectForKey(
                    key, 0, value.size(), staging, sizeof(staging),
                    &data, &bytes, &value_len), Status::kOk);
      EXPECT_EQ(std::string(data, bytes), value);
      EXPECT_EQ(value_len, value.size());
      EXPECT_EQ(store->direct_reads.load(), 2u);
      server.reset();
      fs::remove_all(dir);
    }
  }
}

TEST(RamTierWiring, ColdWholeDirectReadPromotesWithoutLengthOutput) {
  for (bool coalesce : {false, true}) {
    SCOPED_TRACE("coalesce=" + std::to_string(coalesce));
    const auto dir = fs::temp_directory_path() /
        ("dfkv_direct_whole_" + std::to_string(coalesce));
    fs::remove_all(dir);
    fs::create_directories(dir);
    std::atomic<size_t> disk_puts{0};
    CountingStore* store = nullptr;
    DiskCacheGroup::Options options{{dir.string()}, 1ull << 30, "file"};
    options.engine_factory = [&](const std::string& path, uint64_t capacity) {
      auto engine = std::make_unique<CountingStore>(path, capacity, disk_puts);
      store = engine.get();
      return engine;
    };
    auto server = KvNodeServerWiringTestPeer::Create(std::move(options));
    KvNodeServerWiringTestPeer::SetCoalescing(*server, coalesce);
    KvNodeServerWiringTestPeer::SetRam(
        *server, [](const BlockKey&, char*, size_t, size_t) {
          return Status::kIOError;
        });
    const BlockKey key{8202, 0};
    const std::string value(4096 + 111, 'w');
    ASSERT_EQ(KvNodeServerWiringTestPeer::Persist(
                  *server, key, value.data(), value.size()), Status::kOk);
    ASSERT_FALSE(KvNodeServerWiringTestPeer::Resident(*server, key));
    alignas(4096) char staging[8192];
    const char* data = nullptr;
    size_t bytes = 0;
    // value_len is optional; omitting it must not suppress whole promotion.
    ASSERT_EQ(server->RangeDirectForKey(
                  key, 0, value.size(), staging, sizeof(staging),
                  &data, &bytes), Status::kOk);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(std::string(data, bytes), value);
    EXPECT_EQ(store->direct_reads.load(), 1u);
    EXPECT_TRUE(KvNodeServerWiringTestPeer::Resident(*server, key));
    ASSERT_EQ(server->RangeDirectForKey(
                  key, 0, value.size(), staging, sizeof(staging),
                  &data, &bytes), Status::kOk);
    EXPECT_EQ(std::string(data, bytes), value);
    EXPECT_EQ(store->direct_reads.load(), 1u);
    server.reset();
    fs::remove_all(dir);
  }
}

TEST(RamTierWiring, DisabledByDefaultNoRamMetrics) {
  ::unsetenv("DFKV_RAM_TIER");  // ensure off
  std::string addr;
  auto dir = fs::temp_directory_path() / "dfkv_ramwire_c";
  auto s = Start(dir, &addr);
  TcpTransport t;
  std::string v(128, 'd');
  ASSERT_EQ(t.Cache(addr, ToBlockKey("test/model", "x"), v.data(), v.size()), Status::kOk);
  std::string out;
  ASSERT_EQ(t.Range(addr, ToBlockKey("test/model", "x"), 0, v.size(), &out), Status::kOk);  // via disk
  EXPECT_EQ(out, v);
  std::string m = s->MetricsText();
  EXPECT_EQ(MetricVal(m, "dfkv_ram_hit_total"), -1) << "no RAM metrics when disabled";
  EXPECT_EQ(MetricVal(m, "dfkv_ram_put_total"), -1);
  EXPECT_EQ(s->m_cache_put(), 1u);   // normal disk accounting intact
  EXPECT_EQ(s->m_cache_hit(), 1u);
  s.reset();
  fs::remove_all(dir);
}

// Hold all acquired permits until every contender has attempted admission.
// The winner count is exact, independent of disk latency or thread scheduling.
TEST(RamTierWiring, PutAdmissionGateReservesLimitAtomically) {
  ::unsetenv("DFKV_RAM_TIER");
  auto dir = fs::temp_directory_path() / "dfkv_admission_atomic";
  fs::remove_all(dir);
  fs::create_directories(dir);
  auto server = std::make_unique<KvNodeServer>(dir.string(), 1ull << 30);
  KvNodeServerWiringTestPeer::SetLimit(*server, 1);
  std::mutex mutex;
  std::condition_variable cv;
  int attempted = 0;
  int admitted = 0;
  bool release = false;
  std::vector<std::thread> writers;
  for (int i = 0; i < 8; ++i) {
    writers.emplace_back([&] {
      const bool acquired = KvNodeServerWiringTestPeer::Begin(*server);
      {
        std::unique_lock<std::mutex> lock(mutex);
        ++attempted;
        if (acquired) ++admitted;
        cv.notify_all();
        cv.wait(lock, [&] { return release; });
      }
      if (acquired) KvNodeServerWiringTestPeer::End(*server);
    });
  }
  {
    std::unique_lock<std::mutex> lock(mutex);
    EXPECT_TRUE(cv.wait_for(lock, 2s, [&] { return attempted == 8; }));
    release = true;
  }
  cv.notify_all();
  for (auto& writer : writers) writer.join();
  EXPECT_EQ(admitted, 1);
  EXPECT_EQ(server->PutBusyTotal(), 7u);
  const bool acquired = KvNodeServerWiringTestPeer::Begin(*server);
  EXPECT_TRUE(acquired);
  if (acquired) KvNodeServerWiringTestPeer::End(*server);
  server.reset();
  fs::remove_all(dir);
}

TEST(RamTierWiring, PutGateRejectsBeforeRamAdmissionAndLeavesFlushUngated) {
  ::unsetenv("DFKV_RAM_TIER");
  auto dir = fs::temp_directory_path() / "dfkv_admission_ram";
  fs::remove_all(dir);
  fs::create_directories(dir);
  auto server = std::make_unique<KvNodeServer>(dir.string(), 1ull << 30);
  KvNodeServerWiringTestPeer::SetLimit(*server, 1);
  std::mutex mutex;
  std::condition_variable cv;
  bool flushing = false;
  bool release = false;
  int rejected_done = 0;
  KvNodeServerWiringTestPeer::SetRam(
      *server, [&](const BlockKey& key, char* data, size_t len, size_t) {
        {
          std::unique_lock<std::mutex> lock(mutex);
          flushing = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release; });
        }
        return KvNodeServerWiringTestPeer::Persist(*server, key, data, len);
      });
  const BlockKey first{7001, 0}, direct_key{7002, 0}, tcp_key{7003, 0};
  std::string original(500, 'a'), conflicting(1000, 'b');
  auto tcp_put = [&](const BlockKey& key, const std::string& value) {
    std::string out;
    return server->ProcessRequestForKey(
        static_cast<uint8_t>(WireOp::kCache), key, 0, 0, value.data(),
        value.size(), &out);
  };
  Status leader = Status::kInvalid;
  Status direct_rejected = Status::kInvalid;
  Status tcp_rejected = Status::kInvalid;
  std::thread writer([&] {
    leader = server->CacheDirectForKey(first, original.data(),
                                      original.size(), original.size());
  });
  {
    std::unique_lock<std::mutex> lock(mutex);
    EXPECT_TRUE(cv.wait_for(lock, 2s, [&] { return flushing; }));
  }
  std::thread direct([&] {
    direct_rejected = server->CacheDirectForKey(
        direct_key, conflicting.data(), conflicting.size(), conflicting.size());
    std::lock_guard<std::mutex> lock(mutex);
    ++rejected_done;
    cv.notify_all();
  });
  std::thread tcp([&] {
    tcp_rejected = tcp_put(tcp_key, conflicting);
    std::lock_guard<std::mutex> lock(mutex);
    ++rejected_done;
    cv.notify_all();
  });
  {
    std::unique_lock<std::mutex> lock(mutex);
    // This timeout is only a failure guard. The ordering is the explicit
    // flush/permit barrier, not a sleep chosen to catch a race window.
    EXPECT_TRUE(cv.wait_for(lock, 2s, [&] { return rejected_done == 2; }));
  }
  EXPECT_FALSE(KvNodeServerWiringTestPeer::Resident(*server, direct_key));
  EXPECT_FALSE(KvNodeServerWiringTestPeer::Resident(*server, tcp_key));
  {
    std::lock_guard<std::mutex> lock(mutex);
    release = true;
  }
  cv.notify_all();
  writer.join();
  direct.join();
  tcp.join();
  EXPECT_EQ(leader, Status::kOk);
  EXPECT_EQ(direct_rejected, Status::kCacheFull);
  EXPECT_EQ(tcp_rejected, Status::kCacheFull);
  EXPECT_EQ(server->PutBusyTotal(), 2u);
  // The accepted write retained its permit until flush completed, and flush
  // did not itself need a second permit. Both public paths are usable again.
  EXPECT_EQ(tcp_put(first, conflicting), Status::kOk);
  EXPECT_EQ(server->CacheDirectForKey(first, conflicting.data(),
                                      conflicting.size(), conflicting.size()),
            Status::kOk);
  std::string persisted;
  ASSERT_EQ(KvNodeServerWiringTestPeer::DiskRead(*server, first, &persisted),
            Status::kOk);
  EXPECT_EQ(persisted, original);
  EXPECT_EQ(tcp_put(tcp_key, conflicting), Status::kOk);
  server.reset();
  fs::remove_all(dir);
}

TEST(RamTierWiring, DirtyWatermarkRejectsWithoutCallingDiskAndRecovers) {
  ::unsetenv("DFKV_RAM_TIER");
  const auto dir = fs::temp_directory_path() / "dfkv_watermark_terminal";
  fs::remove_all(dir);
  fs::create_directories(dir);
  std::atomic<size_t> disk_calls{0};
  FlushBarrier barrier;
  DiskCacheGroup::Options options{{dir.string()}, 1ull << 30, "file"};
  options.engine_factory = [&](const std::string& path, uint64_t capacity) {
    return std::make_unique<CountingStore>(path, capacity, disk_calls);
  };
  auto server = KvNodeServerWiringTestPeer::Create(std::move(options));
  auto* node = server.get();
  OpenFlushOnExit cleanup{barrier};
  KvNodeServerWiringTestPeer::SetLimit(*server, 0);
  KvNodeServerWiringTestPeer::SetRam(
      *server,
      [&, node](const BlockKey& key, char* data, size_t len, size_t) {
        barrier.Block();
        return KvNodeServerWiringTestPeer::Persist(*node, key, data, len);
      },
      /*ram_ack=*/true, /*watermark_pct=*/25);
  const BlockKey first{7101, 0}, tcp_key{7102, 0}, direct_key{7103, 0};
  std::string original(16 * 1024, 'a'), different(500, 'b');
  auto tcp_put = [&](const BlockKey& key, const std::string& value) {
    std::string out;
    return server->ProcessRequestForKey(
        static_cast<uint8_t>(WireOp::kCache), key, 0, 0, value.data(),
        value.size(), &out);
  };
  ASSERT_EQ(tcp_put(first, original), Status::kOk);  // RAM-ACK, no live PUT permit
  ASSERT_TRUE(barrier.AwaitEntry());
  ASSERT_EQ(disk_calls.load(), 0u);  // durability is deliberately stalled

  Status tcp_status = Status::kInvalid, direct_status = Status::kInvalid;
  int completed = 0;
  std::thread tcp([&] {
    tcp_status = tcp_put(tcp_key, different);
    std::lock_guard<std::mutex> lock(barrier.mutex);
    ++completed;
    barrier.cv.notify_all();
  });
  std::thread direct([&] {
    direct_status = server->CacheDirectForKey(
        direct_key, different.data(), different.size(), different.size());
    std::lock_guard<std::mutex> lock(barrier.mutex);
    ++completed;
    barrier.cv.notify_all();
  });
  {
    std::unique_lock<std::mutex> lock(barrier.mutex);
    // A failure guard, not a timing race: flush stays closed until both NEW
    // requests have had the opportunity to finish without queueing behind it.
    EXPECT_TRUE(barrier.cv.wait_for(lock, 2s, [&] { return completed == 2; }));
  }
  EXPECT_EQ(disk_calls.load(), 0u);
  EXPECT_EQ(server->Count(), 0u);
  EXPECT_EQ(server->PutBusyTotal(), 0u);  // not the foreground-count gate
  EXPECT_TRUE(server->Healthy());
  EXPECT_TRUE(KvNodeServerWiringTestPeer::Resident(*server, first));
  EXPECT_FALSE(KvNodeServerWiringTestPeer::Resident(*server, tcp_key));
  EXPECT_FALSE(KvNodeServerWiringTestPeer::Resident(*server, direct_key));
  std::string visible;
  EXPECT_EQ(server->ProcessRequestForKey(
                static_cast<uint8_t>(WireOp::kRange), first, 0, 0,
                nullptr, 0, &visible),
            Status::kOk);
  EXPECT_EQ(visible, original);
  barrier.Open();
  tcp.join();
  direct.join();
  EXPECT_EQ(tcp_status, Status::kCacheFull);
  EXPECT_EQ(direct_status, Status::kCacheFull);
  ASSERT_TRUE(KvNodeServerWiringTestPeer::Drain(*server));
  EXPECT_EQ(disk_calls.load(), 1u);

  // Rejections never claimed these keys. Both entry points can admit them
  // after drain; conflicting duplicates still preserve the committed leader.
  EXPECT_EQ(tcp_put(first, different), Status::kOk);
  EXPECT_EQ(server->CacheDirectForKey(
                first, different.data(), different.size(), different.size()),
            Status::kOk);
  EXPECT_EQ(disk_calls.load(), 1u);
  std::string persisted;
  ASSERT_EQ(KvNodeServerWiringTestPeer::DiskRead(*server, first, &persisted),
            Status::kOk);
  EXPECT_EQ(persisted, original);
  ASSERT_EQ(tcp_put(tcp_key, different), Status::kOk);
  ASSERT_TRUE(KvNodeServerWiringTestPeer::Drain(*server));
  ASSERT_EQ(server->CacheDirectForKey(
                direct_key, different.data(), different.size(), different.size()),
            Status::kOk);
  ASSERT_TRUE(KvNodeServerWiringTestPeer::Drain(*server));
  EXPECT_EQ(disk_calls.load(), 3u);
  EXPECT_TRUE(server->Healthy());
  server.reset();
  fs::remove_all(dir);
}

// REQ-26: with DFKV_RAM_ACK_BACKPRESSURE_WAIT_MS a PUT of a new key refused
// at the dirty watermark waits for the flush to bring the dirty bytes down.
// The flush is held by a barrier: the first value fills the watermark (25%
// of 64 KiB), and a second key's PUT then has to wait.
namespace {
struct WatermarkNode {
  fs::path dir;
  std::atomic<size_t> disk_calls{0};
  FlushBarrier barrier;
  std::unique_ptr<KvNodeServer> server;
  std::string first_value = std::string(16 * 1024, 'a');
  explicit WatermarkNode(const std::string& tag, uint32_t wait_ms,
                         uint32_t put_wait_ms = 3000) {
    ::unsetenv("DFKV_RAM_TIER");
    dir = fs::temp_directory_path() / ("dfkv_bp_wait_" + tag);
    fs::remove_all(dir);
    fs::create_directories(dir);
    DiskCacheGroup::Options options{{dir.string()}, 1ull << 30, "file"};
    options.engine_factory = [this](const std::string& path, uint64_t cap) {
      return std::make_unique<CountingStore>(path, cap, disk_calls);
    };
    server = KvNodeServerWiringTestPeer::Create(std::move(options));
    KvNodeServerWiringTestPeer::SetLimit(*server, 0);
    KvNodeServer* node = server.get();
    KvNodeServerWiringTestPeer::SetRam(
        *server,
        [this, node](const BlockKey& key, char* data, size_t len, size_t) {
          barrier.Block();
          return KvNodeServerWiringTestPeer::Persist(*node, key, data, len);
        },
        /*ram_ack=*/true, /*watermark_pct=*/25, put_wait_ms, wait_ms);
  }
  ~WatermarkNode() {
    barrier.Open();
    server.reset();
    fs::remove_all(dir);
  }
  Status Put(const BlockKey& key, const std::string& value) {
    std::string out;
    return server->ProcessRequestForKey(static_cast<uint8_t>(WireOp::kCache),
                                        key, 0, 0, value.data(), value.size(),
                                        &out);
  }
  long Metric(const char* name) { return MetricVal(server->MetricsText(), name); }
};
}  // namespace

TEST(RamTierWiring, AWatermarkWaitAdmitsThePutWhenTheFlushDrains) {
  WatermarkNode n("admit", /*wait_ms=*/5000);
  const BlockKey first{7201, 0}, second{7202, 0};
  ASSERT_EQ(n.Put(first, n.first_value), Status::kOk);
  ASSERT_TRUE(n.barrier.AwaitEntry());
  std::atomic<bool> done{false};
  Status st = Status::kInvalid;
  std::thread put([&] {
    st = n.Put(second, std::string(500, 'b'));
    done = true;
  });
  std::this_thread::sleep_for(100ms);
  EXPECT_FALSE(done.load()) << "refused instead of waiting";
  n.barrier.Open();  // the flush drains; the waiting PUT is admitted
  put.join();
  EXPECT_EQ(st, Status::kOk);
  EXPECT_EQ(n.Metric("dfkv_ram_ack_backpressure_waits_total"), 1);
  EXPECT_EQ(n.Metric("dfkv_ram_ack_backpressure_wait_timeouts_total"), 0);
  EXPECT_EQ(n.Metric("dfkv_ram_ack_backpressure_total"), 0);
  EXPECT_GE(n.Metric("dfkv_ram_ack_backpressure_wait_microseconds_total"),
            100000);
}

// The wait ends at DFKV_RAM_ACK_BACKPRESSURE_WAIT_MS, and at the batch's
// DFKV_RAM_PUT_WAIT_MS if that comes first: then the PUT is refused.
TEST(RamTierWiring, AWatermarkWaitIsBounded) {
  for (const auto& [wait_ms, put_wait_ms] :
       std::vector<std::pair<uint32_t, uint32_t>>{{150, 3000}, {5000, 150}}) {
    SCOPED_TRACE(std::to_string(wait_ms) + "/" + std::to_string(put_wait_ms));
    WatermarkNode n("bound" + std::to_string(wait_ms), wait_ms, put_wait_ms);
    const BlockKey first{7211, 0}, second{7212, 0};
    ASSERT_EQ(n.Put(first, n.first_value), Status::kOk);
    ASSERT_TRUE(n.barrier.AwaitEntry());
    const auto t0 = std::chrono::steady_clock::now();
    EXPECT_EQ(n.Put(second, std::string(500, 'b')), Status::kCacheFull);
    const auto waited = std::chrono::steady_clock::now() - t0;
    EXPECT_GE(waited, 140ms);
    EXPECT_LT(waited, 2s);
    EXPECT_EQ(n.Metric("dfkv_ram_ack_backpressure_wait_timeouts_total"), 1);
    EXPECT_EQ(n.Metric("dfkv_ram_ack_backpressure_total"), 1);
    EXPECT_EQ(n.disk_calls.load(), 0u) << "a refusal must not go to the disk";
  }
}

// CacheDirectForKey's batch_arrival is the same absolute budget as the TCP
// path: a pre-aged direct PUT gets only the remainder, and an expired batch
// does not wait through another full watermark interval. Each terminal
// refusal is counted once even though admission has its own wait counters.
TEST(RamTierWiring, WatermarkWaitUsesTheRemainingDirectBatchBudget) {
  WatermarkNode n("direct-budget", /*wait_ms=*/5000, /*put_wait_ms=*/400);
  const BlockKey first{7211, 0}, pre_aged{7212, 0}, expired{7213, 0};
  ASSERT_EQ(n.Put(first, n.first_value), Status::kOk);
  ASSERT_TRUE(n.barrier.AwaitEntry());
  std::vector<char> value(500, 'b');

  auto t0 = std::chrono::steady_clock::now();
  EXPECT_EQ(n.server->CacheDirectForKey(pre_aged, value.data(), value.size(),
                                        value.size(), t0 - 300ms),
            Status::kCacheFull);
  auto waited = std::chrono::steady_clock::now() - t0;
  EXPECT_GE(waited, 90ms);
  EXPECT_LT(waited, 300ms) << "the PUT got a fresh 400 ms batch budget";
  EXPECT_EQ(n.Metric("dfkv_ram_ack_backpressure_waits_total"), 1);
  EXPECT_EQ(n.Metric("dfkv_ram_ack_backpressure_wait_timeouts_total"), 1);
  EXPECT_EQ(n.Metric("dfkv_ram_ack_backpressure_total"), 1);
  EXPECT_EQ(n.disk_calls.load(), 0u);
  EXPECT_FALSE(KvNodeServerWiringTestPeer::Resident(*n.server, pre_aged));

  t0 = std::chrono::steady_clock::now();
  EXPECT_EQ(n.server->CacheDirectForKey(expired, value.data(), value.size(),
                                        value.size(), t0 - 500ms),
            Status::kCacheFull);
  waited = std::chrono::steady_clock::now() - t0;
  EXPECT_LT(waited, 50ms) << "an expired batch waited again";
  EXPECT_EQ(n.Metric("dfkv_ram_ack_backpressure_waits_total"), 2);
  EXPECT_EQ(n.Metric("dfkv_ram_ack_backpressure_wait_timeouts_total"), 2);
  EXPECT_EQ(n.Metric("dfkv_ram_ack_backpressure_total"), 2);
  EXPECT_EQ(n.disk_calls.load(), 0u);
  EXPECT_FALSE(KvNodeServerWiringTestPeer::Resident(*n.server, expired));
}

// While a PUT waits, its key's lock is free, a GET on another thread (another
// connection) is served, and a stop ends the wait at once.
TEST(RamTierWiring, AWatermarkWaitHoldsNoKeyLockBlocksNoGetAndEndsAtStop) {
  WatermarkNode n("lock", /*wait_ms=*/10000);
  const BlockKey first{7221, 0}, second{7222, 0};
  ASSERT_EQ(n.Put(first, n.first_value), Status::kOk);
  ASSERT_TRUE(n.barrier.AwaitEntry());
  std::atomic<bool> done{false};
  Status st = Status::kInvalid;
  std::thread put([&] {
    st = n.Put(second, std::string(500, 'b'));
    done = true;
  });
  ASSERT_TRUE(WaitFor([&] {
    return n.Metric("dfkv_ram_ack_backpressure_waits_total") == 1;
  }));
  {
    // Taken and released on the same thread (a lock is released by the
    // thread that took it).
    auto locked = std::async(std::launch::async, [&] {
      auto lock = KvNodeServerWiringTestPeer::LockPut(*n.server, second);
      return lock != nullptr;
    });
    EXPECT_EQ(locked.wait_for(1s), std::future_status::ready)
        << "the waiting PUT holds its key's lock";
  }
  auto get = std::async(std::launch::async, [&] {
    std::string out;
    const Status s = n.server->ProcessRequestForKey(
        static_cast<uint8_t>(WireOp::kRange), first, 0, 0, nullptr, 0, &out);
    return s == Status::kOk && out == n.first_value;
  });
  ASSERT_EQ(get.wait_for(1s), std::future_status::ready)
      << "a GET waited for the waiting PUT";
  EXPECT_TRUE(get.get());
  EXPECT_FALSE(done.load());
  const auto t0 = std::chrono::steady_clock::now();
  KvNodeServerWiringTestPeer::CancelPutWaits(*n.server);
  put.join();
  EXPECT_LT(std::chrono::steady_clock::now() - t0, 1s);
  EXPECT_EQ(st, Status::kCacheFull);
}

// With the default (0) the PUT is refused at once, as before.
TEST(RamTierWiring, WithoutAWatermarkWaitTheRefusalIsImmediate) {
  WatermarkNode n("none", /*wait_ms=*/0);
  const BlockKey first{7231, 0}, second{7232, 0};
  ASSERT_EQ(n.Put(first, n.first_value), Status::kOk);
  ASSERT_TRUE(n.barrier.AwaitEntry());
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_EQ(n.Put(second, std::string(500, 'b')), Status::kCacheFull);
  EXPECT_LT(std::chrono::steady_clock::now() - t0, 100ms);
  EXPECT_EQ(n.Metric("dfkv_ram_ack_backpressure_waits_total"), 0);
  EXPECT_EQ(n.Metric("dfkv_ram_ack_backpressure_total"), 1);
}

// A tenant with a node quota skips RAM write-back admission. The direct (RDMA)
// PUT entry must still reach the quota-checked disk write: before the fix both
// branches were gated on "no quota", so every such PUT returned kCacheFull and
// stored nothing while the TCP entry worked. Write-around took the same path.
TEST(RamTierWiring, LimitedTenantDirectPutReachesDiskWhenRamEnabled) {
  for (bool write_back : {true, false}) {
    auto dir = fs::temp_directory_path() /
        ("dfkv_ramwire_quota_direct_" + std::to_string(write_back));
    fs::remove_all(dir);
    fs::create_directories(dir);
    ::unsetenv("DFKV_TENANT_QUOTAS_FILE");
    ::setenv("DFKV_TENANT_DEFAULT_QUOTA_BYTES", "65536", 1);
    DiskCacheGroup::Options options{{dir.string()}, 1ull << 30, "file"};
    auto server = KvNodeServerWiringTestPeer::Create(std::move(options));
    ::unsetenv("DFKV_TENANT_DEFAULT_QUOTA_BYTES");
    auto* node = server.get();
    KvNodeServerWiringTestPeer::SetLimit(*server, 0);
    std::atomic<size_t> flushes{0};
    KvNodeServerWiringTestPeer::SetRam(
        *server,
        [&, node](const BlockKey& key, char* data, size_t len, size_t) {
          flushes.fetch_add(1, std::memory_order_relaxed);
          return KvNodeServerWiringTestPeer::Persist(*node, key, data, len);
        });
    KvNodeServerWiringTestPeer::SetWriteBack(*server, write_back);

    const BlockKey key{8201, 0};
    alignas(4096) char value[4096];
    std::memset(value, 'q', sizeof(value));
    EXPECT_EQ(server->CacheDirectForKey(key, value, 1024, sizeof(value)),
              Status::kOk)
        << "write_back=" << write_back;
    std::string out;
    EXPECT_EQ(KvNodeServerWiringTestPeer::DiskRead(*server, key, &out),
              Status::kOk);
    EXPECT_EQ(out, std::string(1024, 'q'));
    // The limited tenant never entered RAM write-back.
    EXPECT_FALSE(KvNodeServerWiringTestPeer::Resident(*server, key));
    EXPECT_EQ(flushes.load(std::memory_order_relaxed), 0u);

    // Exceeding the quota is reported and counted as such, not as a full
    // cache, and stores nothing.
    const BlockKey big{8202, 0};
    alignas(4096) char large[128 * 1024];
    std::memset(large, 'r', sizeof(large));
    EXPECT_EQ(server->CacheDirectForKey(big, large, sizeof(large), sizeof(large)),
              Status::kQuotaExceeded)
        << "write_back=" << write_back;
    EXPECT_EQ(KvNodeServerWiringTestPeer::DiskRead(*server, big, &out),
              Status::kNotFound);
    const std::string metrics = server->MetricsText();
    EXPECT_EQ(MetricVal(metrics, "dfkv_cache_put_total"), 1);
    EXPECT_EQ(MetricVal(metrics, "dfkv_tenant_quota_rejections_total"), 1);

    server.reset();
    fs::remove_all(dir);
  }
}

// PreparedRead unifies RAM and disk ownership. A RAM-resident value is returned
// ready with a send pin; an absent key proceeds to disk after exactly one RAM
// miss. Destroying the ready owner exercises connection-abort cleanup.
TEST(RamTierWiring, PreparedReadCountsRamMissesAndOwnsRamPin) {
  ::setenv("DFKV_RAM_TIER", "1", 1);
  ::setenv("DFKV_RAM_TIER_BYTES", "8388608", 1);
  std::string addr;
  auto dir = fs::temp_directory_path() / "dfkv_ramwire_prep";
  auto s = Start(dir, &addr);
  TcpTransport t;
  std::string v(4096, 'q');
  ASSERT_EQ(t.Cache(addr, ToBlockKey("test/model", "resident"), v.data(), v.size()), Status::kOk);

  const BlockKey rk = ToBlockKey("test/model", "resident");
  const BlockKey ak = ToBlockKey("test/model", "absent-key");
  std::vector<char> staging(1 << 20);

  // Consulted-and-present is not a miss and returns one live transaction.
  const long miss0 = MetricVal(s->MetricsText(), "dfkv_ram_miss_total");
  {
    PreparedRead prep =
        s->PrepareReadForKey(rk, 0, v.size(), staging.data(), staging.size());
    EXPECT_EQ(prep.status(), Status::kOk);
    EXPECT_TRUE(prep.owns_cleanup());
    EXPECT_FALSE(prep.needs_io());
    EXPECT_EQ(prep.payload_len(), v.size());
  }  // destructor-abort releases the RAM send pin
  EXPECT_EQ(MetricVal(s->MetricsText(), "dfkv_ram_miss_total"), miss0);

  // Failure to resolve the arena MR falls back to the owned staging buffer
  // without splitting cleanup ownership or releasing the pin early.
  PreparedRead staged =
      s->PrepareReadForKey(rk, 0, v.size(), staging.data(), staging.size());
  ASSERT_EQ(staged.status(), Status::kOk);
  ASSERT_TRUE(staged.source_registered());
  ASSERT_TRUE(staged.Stage());
  EXPECT_FALSE(staged.source_registered());
  EXPECT_EQ(std::memcmp(staged.data(), v.data(), v.size()), 0);
  EXPECT_TRUE(staged.Commit(Status::kOk, staged.payload_len(), 0.0));
  EXPECT_FALSE(staged.Abort());

  // Absent everywhere: RAM consulted and absent -> exactly one RAM miss.
  PreparedRead absent =
      s->PrepareReadForKey(ak, 0, 16, staging.data(), staging.size());
  EXPECT_EQ(absent.status(), Status::kNotFound);
  EXPECT_FALSE(absent.owns_cleanup());
  EXPECT_EQ(MetricVal(s->MetricsText(), "dfkv_ram_miss_total"), miss0 + 1);

  s.reset();
  ::unsetenv("DFKV_RAM_TIER");
  ::unsetenv("DFKV_RAM_TIER_BYTES");
  fs::remove_all(dir);
}

namespace {
// What a stopping dfkv_server relies on: once Shutdown() has returned, the
// disks are clean whatever happens to the process next. The child leaves
// through _exit, as if it had been killed while releasing its memory.
TEST(RamTierWiring, ShutdownLeavesTheDisksCleanWithoutDestruction) {
  ::unsetenv("DFKV_RAM_TIER");
  ::unsetenv("DFKV_RAM_SHUTDOWN_DRAIN_TIMEOUT_S");
  const auto dir = fs::temp_directory_path() / "dfkv_shutdown_clean";
  fs::remove_all(dir);
  fs::create_directories(dir);
  constexpr int kObjects = 10;
  const std::string value(2000, 's');
  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    // The slab engine, whatever DFKV_STORE_ENGINE says: the file engine has
    // no clean marker to lose.
    auto server = KvNodeServerWiringTestPeer::Create(
        DiskCacheGroup::Options{{dir.string()}, 1ull << 30, "slab"});
    auto* node = server.get();
    KvNodeServerWiringTestPeer::SetRam(
        *server,
        [node](const BlockKey& key, char* data, size_t len, size_t) {
          return KvNodeServerWiringTestPeer::Persist(*node, key, data, len);
        },
        /*ram_ack=*/true);
    bool ok = server->Start(0) == Status::kOk;
    for (int i = 0; ok && i < kObjects; ++i) {
      std::string out;
      ok = server->ProcessRequestForKey(
               static_cast<uint8_t>(WireOp::kCache), BlockKey{7300u + i, 0}, 0,
               0, value.data(), value.size(), &out) == Status::kOk;
    }
    ok = ok && server->Shutdown();
    ok = ok && server->Shutdown();  // idempotent, same answer
    ok = ok && KvNodeServerWiringTestPeer::RamDirty(*server) == 0;
    ::_exit(ok ? 0 : 1);
  }
  int status = 0;
  ASSERT_EQ(::waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);

  // An unclean state would have reset the cache (warm restart is off).
  ::unsetenv("DFKV_SLAB_WARM_RESTART");
  auto reopened = KvNodeServerWiringTestPeer::Create(
      DiskCacheGroup::Options{{dir.string()}, 1ull << 30, "slab"});
  ASSERT_TRUE(reopened->Healthy());
  EXPECT_EQ(KvNodeServerWiringTestPeer::UncleanResets(*reopened), 0u);
  EXPECT_EQ(KvNodeServerWiringTestPeer::WarmRestarts(*reopened), 0u);
  for (int i = 0; i < kObjects; ++i) {
    std::string out;
    ASSERT_EQ(KvNodeServerWiringTestPeer::DiskRead(
                  *reopened, BlockKey{7300u + i, 0}, &out),
              Status::kOk)
        << i;
    EXPECT_EQ(out, value);
  }
  reopened.reset();
  fs::remove_all(dir);
}

// The backlog gets DFKV_RAM_SHUTDOWN_DRAIN_TIMEOUT_S. What is still only in
// RAM after that is dropped, and the disks are closed clean all the same; no
// flush runs after Shutdown() returned.
TEST(RamTierWiring, ShutdownDropsWhatMissesTheDrainTimeout) {
  ::unsetenv("DFKV_RAM_TIER");
  ::unsetenv("DFKV_SLAB_WARM_RESTART");
  ::setenv("DFKV_RAM_SHUTDOWN_DRAIN_TIMEOUT_S", "0", 1);
  struct RestoreEnv {
    ~RestoreEnv() { ::unsetenv("DFKV_RAM_SHUTDOWN_DRAIN_TIMEOUT_S"); }
  } restore_env;
  const auto dir = fs::temp_directory_path() / "dfkv_shutdown_abandon";
  fs::remove_all(dir);
  fs::create_directories(dir);
  constexpr int kObjects = 6;
  const std::string value(2000, 'a');
  FlushBarrier barrier;
  std::atomic<int> flushes{0};
  {
    auto server = KvNodeServerWiringTestPeer::Create(
        DiskCacheGroup::Options{{dir.string()}, 1ull << 30, "slab"});
    auto* node = server.get();
    OpenFlushOnExit cleanup{barrier};
    KvNodeServerWiringTestPeer::SetRam(
        *server,
        [&, node](const BlockKey& key, char* data, size_t len, size_t) {
          barrier.Block();
          flushes.fetch_add(1);
          return KvNodeServerWiringTestPeer::Persist(*node, key, data, len);
        },
        /*ram_ack=*/true);
    auto put = [&](int i) {
      std::string out;
      return server->ProcessRequestForKey(
          static_cast<uint8_t>(WireOp::kCache), BlockKey{7400u + i, 0}, 0, 0,
          value.data(), value.size(), &out);
    };
    // The flusher holds the first object, blocked; the others queue behind it.
    ASSERT_EQ(put(0), Status::kOk);
    ASSERT_TRUE(barrier.AwaitEntry());
    for (int i = 1; i < kObjects; ++i) ASSERT_EQ(put(i), Status::kOk);

    bool clean = false;
    std::thread stopper([&] { clean = server->Shutdown(); });
    const bool stopping = WaitFor(
        [&] { return KvNodeServerWiringTestPeer::RamStopping(*server); });
    barrier.Open();
    stopper.join();
    ASSERT_TRUE(stopping);
    // The disks were marked clean by Shutdown() itself, not by the destructor.
    EXPECT_TRUE(clean);
    EXPECT_EQ(flushes.load(), 1);
    EXPECT_EQ(KvNodeServerWiringTestPeer::RamDirty(*server),
              static_cast<uint64_t>(kObjects - 1));
  }
  EXPECT_EQ(flushes.load(), 1);  // destruction did not resume the drain

  auto reopened = KvNodeServerWiringTestPeer::Create(
      DiskCacheGroup::Options{{dir.string()}, 1ull << 30, "slab"});
  ASSERT_TRUE(reopened->Healthy());
  EXPECT_EQ(KvNodeServerWiringTestPeer::UncleanResets(*reopened), 0u);
  EXPECT_EQ(KvNodeServerWiringTestPeer::WarmRestarts(*reopened), 0u);
  std::string out;
  EXPECT_EQ(KvNodeServerWiringTestPeer::DiskRead(*reopened, BlockKey{7400, 0},
                                                 &out),
            Status::kOk);
  EXPECT_EQ(out, value);
  for (int i = 1; i < kObjects; ++i)
    EXPECT_EQ(KvNodeServerWiringTestPeer::DiskRead(
                  *reopened, BlockKey{7400u + i, 0}, &out),
              Status::kNotFound)
        << i;
  reopened.reset();
  fs::remove_all(dir);
}

// In disk-ACK mode a PUT waits for the flush of its key, as first writer or
// as duplicate. It must not keep the lock that serializes PUTs of its stripe
// while it waits: an unrelated key on the same stripe is admitted.
TEST(RamTierWiring, ParkedPutDoesNotBlockItsStripe) {
  ::unsetenv("DFKV_RAM_TIER");
  const auto dir = fs::temp_directory_path() / "dfkv_parked_put_stripe";
  fs::remove_all(dir);
  fs::create_directories(dir);
  FlushBarrier barrier;
  auto server = KvNodeServerWiringTestPeer::Create(
      DiskCacheGroup::Options{{dir.string()}, 1ull << 30, "file"});
  auto* node = server.get();
  OpenFlushOnExit cleanup{barrier};
  KvNodeServerWiringTestPeer::SetLimit(*server, 0);
  KvNodeServerWiringTestPeer::SetRam(
      *server,
      [&, node](const BlockKey& key, char* data, size_t len, size_t) {
        barrier.Block();
        return KvNodeServerWiringTestPeer::Persist(*node, key, data, len);
      },
      /*ram_ack=*/false, /*watermark_pct=*/80, /*put_wait_ms=*/0);
  const BlockKey first{7600, 0};
  const BlockKey neighbour{
      7600 + KvNodeServerWiringTestPeer::PutKeyStripes(), 0};  // same stripe
  const std::string value(2000, 'p'), other(1000, 'q');
  auto put = [&](const BlockKey& key, const std::string& v) {
    std::string out;
    return server->ProcessRequestForKey(static_cast<uint8_t>(WireOp::kCache),
                                        key, 0, 0, v.data(), v.size(), &out);
  };
  const auto parked = [&](size_t count) {
    return WaitFor([&] {
      return KvNodeServerWiringTestPeer::ParkedPuts(*server) == count;
    });
  };
  Status writer_status = Status::kInvalid;
  std::thread writer([&] { writer_status = put(first, value); });
  ASSERT_TRUE(barrier.AwaitEntry());  // its flush is held
  ASSERT_TRUE(parked(1));
  Status duplicate_status = Status::kInvalid;
  std::thread duplicate([&] { duplicate_status = put(first, value); });
  ASSERT_TRUE(parked(2));

  // The neighbour is admitted and then waits for its own flush, behind the
  // held one. Had it queued on the stripe it would never be counted.
  Status neighbour_status = Status::kInvalid;
  std::thread unrelated([&] {
    neighbour_status = server->CacheDirectForKey(
        neighbour, const_cast<char*>(other.data()), other.size(), other.size());
  });
  EXPECT_TRUE(parked(3))
      << "a PUT of another key waited behind the parked PUTs of its stripe";
  EXPECT_TRUE(KvNodeServerWiringTestPeer::Resident(*server, neighbour));
  barrier.Open();
  unrelated.join();
  duplicate.join();
  writer.join();
  EXPECT_EQ(writer_status, Status::kOk);
  EXPECT_EQ(duplicate_status, Status::kOk);
  EXPECT_EQ(neighbour_status, Status::kOk);
  ASSERT_TRUE(KvNodeServerWiringTestPeer::Drain(*server));
  EXPECT_EQ(MetricVal(server->MetricsText(), "dfkv_ram_parked_puts"), 0);
  server.reset();
  fs::remove_all(dir);
}

// A stop answers parked PUTs at once instead of waiting for their flush:
// stopping the request paths joins the threads those PUTs are parked on, and
// that wait is outside the drain limit. Here the flush never finishes while
// requests are being stopped, and the drain limit is zero.
TEST(RamTierWiring, ShutdownAnswersParkedPutsInsteadOfWaitingForTheirFlush) {
  ::unsetenv("DFKV_RAM_TIER");
  ::unsetenv("DFKV_SLAB_WARM_RESTART");
  ::setenv("DFKV_RAM_SHUTDOWN_DRAIN_TIMEOUT_S", "0", 1);
  struct RestoreEnv {
    ~RestoreEnv() { ::unsetenv("DFKV_RAM_SHUTDOWN_DRAIN_TIMEOUT_S"); }
  } restore_env;
  const auto dir = fs::temp_directory_path() / "dfkv_shutdown_parked_put";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const std::string value(2000, 'w');
  const BlockKey key{7700, 0};
  FlushBarrier barrier;
  {
    auto server = KvNodeServerWiringTestPeer::Create(
        DiskCacheGroup::Options{{dir.string()}, 1ull << 30, "slab"});
    auto* node = server.get();
    OpenFlushOnExit cleanup{barrier};
    KvNodeServerWiringTestPeer::SetRam(
        *server,
        [&, node](const BlockKey& k, char* data, size_t len, size_t) {
          barrier.Block();
          return KvNodeServerWiringTestPeer::Persist(*node, k, data, len);
        },
        /*ram_ack=*/false, /*watermark_pct=*/80, /*put_wait_ms=*/0);
    ASSERT_EQ(server->Start(0), Status::kOk);
    const std::string addr = "127.0.0.1:" + std::to_string(server->port());

    // The first writer is parked on a TCP handler thread, which the stop
    // joins; a duplicate is parked on a thread of another request path.
    std::thread tcp_client([&] {
      TcpTransport transport;
      (void)transport.Cache(addr, key, value.data(), value.size());
    });
    ASSERT_TRUE(barrier.AwaitEntry());
    Status direct_status = Status::kInvalid;
    std::thread direct([&] {
      direct_status = server->CacheDirectForKey(
          key, const_cast<char*>(value.data()), value.size(), value.size());
    });
    ASSERT_TRUE(WaitFor(
        [&] { return KvNodeServerWiringTestPeer::ParkedPuts(*server) == 2; }));

    bool clean = false;
    std::thread stopper([&] {
      server->CancelParkedPuts();  // what the process does before Quiesce()
      clean = server->Shutdown();
    });
    // The flush is still held. Requests stop and the (zero) drain limit
    // passes all the same.
    const bool stopping = WaitFor(
        [&] { return KvNodeServerWiringTestPeer::RamStopping(*server); });
    EXPECT_TRUE(stopping) << "the stop waited for a parked PUT's flush";
    // Cancelling only wakes the parked PUTs; each leaves the waiter list on
    // its own thread, possibly after the stop began.
    EXPECT_TRUE(WaitFor([&] {
      return KvNodeServerWiringTestPeer::ParkedPuts(*server) == 0;
    })) << "a parked PUT was not answered";
    barrier.Open();
    stopper.join();
    direct.join();
    tcp_client.join();
    EXPECT_TRUE(clean);
    EXPECT_EQ(direct_status, Status::kCacheFull);
    EXPECT_EQ(MetricVal(server->MetricsText(),
                        "dfkv_ram_put_waits_canceled_total"), 2);
  }
  fs::remove_all(dir);
}

// The limit decides how much of the backlog a stop drops, so a typo must not
// turn into "drop everything at once".
TEST(RamTierWiring, ShutdownDrainLimitIsParsedStrictly) {
  ::unsetenv("DFKV_RAM_TIER");
  const auto dir = fs::temp_directory_path() / "dfkv_shutdown_limit";
  struct RestoreEnv {
    ~RestoreEnv() { ::unsetenv("DFKV_RAM_SHUTDOWN_DRAIN_TIMEOUT_S"); }
  } restore_env;
  auto limit = [&](const char* value) {
    if (value)
      ::setenv("DFKV_RAM_SHUTDOWN_DRAIN_TIMEOUT_S", value, 1);
    else
      ::unsetenv("DFKV_RAM_SHUTDOWN_DRAIN_TIMEOUT_S");
    fs::remove_all(dir);
    fs::create_directories(dir);
    auto server = KvNodeServerWiringTestPeer::Create(
        DiskCacheGroup::Options{{dir.string()}, 1ull << 30, "file"});
    return KvNodeServerWiringTestPeer::DrainLimitSeconds(*server);
  };
  EXPECT_EQ(limit(nullptr), 30u);
  EXPECT_EQ(limit(""), 30u);
  EXPECT_EQ(limit("0"), 0u);     // "do not wait" is a legitimate choice
  EXPECT_EQ(limit("120"), 120u);
  EXPECT_EQ(limit("3600"), 3600u);
  EXPECT_EQ(limit("7200"), 3600u);  // clamped, not ignored
  EXPECT_EQ(limit("99999999999999999999999999"), 3600u);  // so is overflow
  EXPECT_EQ(limit("abc"), 30u);     // used to be read as 0
  EXPECT_EQ(limit("5m"), 30u);      // used to be read as 5
  EXPECT_EQ(limit("-1"), 30u);
  EXPECT_EQ(limit(" 10"), 30u);
  fs::remove_all(dir);
}

// ---- The arena is faulted in while the disks open --------------------------
namespace {
// Threads of this process right now.
size_t ThreadCount() {
  size_t count = 0;
  for (const auto& entry : fs::directory_iterator("/proc/self/task")) {
    (void)entry;
    ++count;
  }
  return count;
}

struct RamTierEnv {
  RamTierEnv() {
    ::setenv("DFKV_RAM_TIER", "1", 1);
    ::setenv("DFKV_RAM_TIER_BYTES", "8388608", 1);
  }
  ~RamTierEnv() {
    ::unsetenv("DFKV_RAM_TIER");
    ::unsetenv("DFKV_RAM_TIER_BYTES");
    ::unsetenv("DFKV_RAM_TIER_NUMA");
  }
};

std::unique_ptr<StoreEngine> FileEngine(const std::string& dir, uint64_t cap) {
  KVStore::Options options;
  options.cache_dir = dir;
  options.capacity_bytes = cap;
  return std::make_unique<KVStore>(options);
}
}  // namespace

// The disk "opens" until another thread exists in the process: the one that
// builds the RAM tier (and the ones it starts to fault the arena in). One
// directory is opened on the constructing thread itself, so with the tier
// built after the disks, as it used to be, no other thread appears during
// the open.
TEST(RamTierWiring, ArenaIsFaultedInWhileTheDisksOpen) {
  RamTierEnv env;
  const auto dir = fs::temp_directory_path() / "dfkv_ramwire_beside";
  fs::remove_all(dir);
  fs::create_directories(dir);
  std::atomic<bool> tier_started_during_open{false};
  const size_t threads_before = ThreadCount();
  DiskCacheGroup::Options options{{dir.string()}, 1ull << 30, "file"};
  options.engine_factory = [&](const std::string& path, uint64_t cap) {
    for (int i = 0; i < 5000 && !tier_started_during_open; ++i) {
      if (ThreadCount() > threads_before)
        tier_started_during_open = true;
      else
        std::this_thread::sleep_for(1ms);
    }
    return FileEngine(path, cap);
  };
  auto server = KvNodeServerWiringTestPeer::Create(std::move(options));
  EXPECT_TRUE(tier_started_during_open);
  EXPECT_TRUE(server->Healthy());
  EXPECT_TRUE(server->ram_enabled());
  // And the pair works: a value goes through RAM to the disk.
  ASSERT_EQ(server->Start(0), Status::kOk);
  const std::string addr = "127.0.0.1:" + std::to_string(server->port());
  TcpTransport t;
  const std::string v(300, 's');
  ASSERT_EQ(t.Cache(addr, ToBlockKey("test/model", "beside"), v.data(), v.size()),
            Status::kOk);
  EXPECT_TRUE(WaitFor([&] {
    return MetricVal(server->MetricsText(), "dfkv_ram_flushed_total") >= 1;
  }));
  server.reset();
  fs::remove_all(dir);
}

// A disk that fails to open while the arena is being faulted in: the start
// fails as before, and the tier built meanwhile is not kept.
TEST(RamTierWiring, AFailedDiskEndsTheStartWithoutARamTier) {
  RamTierEnv env;
  const auto good = fs::temp_directory_path() / "dfkv_ramwire_faildisk_a";
  const auto bad = fs::temp_directory_path() / "dfkv_ramwire_faildisk_b";
  for (const auto& dir : {good, bad}) {
    fs::remove_all(dir);
    fs::create_directories(dir);
  }
  DiskCacheGroup::Options options{{good.string(), bad.string()}, 1ull << 30,
                                  "file"};
  options.engine_factory =
      [&](const std::string& path, uint64_t cap) -> std::unique_ptr<StoreEngine> {
    if (path == bad.string()) throw std::runtime_error("cannot open this disk");
    return FileEngine(path, cap);
  };
  auto server = KvNodeServerWiringTestPeer::Create(std::move(options));
  EXPECT_FALSE(server->Healthy());
  EXPECT_FALSE(server->DisksHealthy());
  EXPECT_FALSE(server->ram_enabled());
  EXPECT_EQ(server->Start(0), Status::kIOError);
  server.reset();
  fs::remove_all(good);
  fs::remove_all(bad);
}

// The arena fails while the disks open fine: no disk-only start, and the
// disks are closed in order by the destructor.
TEST(RamTierWiring, AFailedArenaEndsTheStartWithTheDisksOpen) {
  RamTierEnv env;
  ::setenv("DFKV_RAM_TIER_NUMA", "neither-off-nor-interleave", 1);
  const auto dir = fs::temp_directory_path() / "dfkv_ramwire_failarena";
  fs::remove_all(dir);
  fs::create_directories(dir);
  auto server = KvNodeServerWiringTestPeer::Create(
      DiskCacheGroup::Options{{dir.string()}, 1ull << 30, "file"});
  EXPECT_TRUE(server->DisksHealthy());
  EXPECT_FALSE(server->Healthy());
  EXPECT_FALSE(server->ram_enabled());
  EXPECT_EQ(server->Start(0), Status::kIOError);
  server.reset();
  fs::remove_all(dir);
}

// ---- A start that is cancelled (a stop signal during startup) --------------
namespace {
DiskSlabStore::Options CancelSlab(const std::string& dir) {
  DiskSlabStore::Options options;
  options.dir = dir;
  options.trust_token_dir = dir + "_tokens";
  fs::create_directories(options.trust_token_dir);
  options.capacity_bytes = 1 << 20;
  options.extent_bytes = 1 << 20;
  options.slot_granularity = 4096;
  return options;
}
}  // namespace

// The cancel flag is set while a disk is being opened, as a signal handler
// would. The disk finishes opening (it is marked dirty by then); the server
// comes back cancelled, refuses to start, and its Shutdown closes the disk
// clean: the next open is a clean start with the data in place.
TEST(RamTierWiring, ACancelledStartClosesTheOpenedDisksClean) {
  RamTierEnv env;
  const auto dir = fs::temp_directory_path() / "dfkv_ramwire_cancel";
  fs::remove_all(dir);
  fs::remove_all(dir.string() + "_tokens");
  const std::string value(3000, 'c');
  {
    bool ok = false;
    DiskSlabStore store(CancelSlab(dir.string()), &ok);
    ASSERT_TRUE(ok) << store.StartupError();
    ASSERT_EQ(store.Cache(BlockKey{7, 7}, value.data(), value.size()),
              Status::kOk);
  }
  std::atomic<bool> cancel{false};
  DiskCacheGroup::Options options{{dir.string()}, 1ull << 20, "slab"};
  options.engine_factory =
      [&](const std::string& path, uint64_t) -> std::unique_ptr<StoreEngine> {
    auto store = std::make_unique<DiskSlabStore>(CancelSlab(path));
    cancel.store(true);  // the signal arrives while the disk is being opened
    return store;
  };
  auto server =
      KvNodeServerWiringTestPeer::Create(std::move(options), &cancel);
  EXPECT_TRUE(server->start_cancelled());
  EXPECT_TRUE(server->DisksHealthy());
  EXPECT_FALSE(server->Healthy());
  EXPECT_FALSE(server->ram_enabled());
  EXPECT_EQ(server->Start(0), Status::kIOError) << "a cancelled start started";
  EXPECT_TRUE(server->Shutdown()) << "the opened disk was not closed clean";
  server.reset();

  bool ok = false;
  DiskSlabStore reopened(CancelSlab(dir.string()), &ok);
  ASSERT_TRUE(ok) << reopened.StartupError();
  const auto stats = reopened.GetStats();
  EXPECT_EQ(stats.unclean_resets, 0u);
  EXPECT_EQ(stats.warm_restarts, 0u);
  std::string out;
  ASSERT_EQ(reopened.Range(BlockKey{7, 7}, 0, value.size(), &out), Status::kOk);
  EXPECT_EQ(out, value);
  fs::remove_all(dir);
  fs::remove_all(dir.string() + "_tokens");
}

// A cancelled start in which a disk failed to open. The group does not keep
// a disk whose open threw, so every disk it has closes clean; Shutdown must
// still not answer "all clean", because dfkv_server takes the exit status
// and the last log line of a cancelled start from it.
TEST(RamTierWiring, ACancelledStartWithAFailedDiskIsNotAllClean) {
  RamTierEnv env;
  const auto good = fs::temp_directory_path() / "dfkv_ramwire_cancel_good";
  const auto bad = fs::temp_directory_path() / "dfkv_ramwire_cancel_bad";
  for (const auto& dir : {good, bad}) {
    fs::remove_all(dir);
    fs::remove_all(dir.string() + "_tokens");
  }
  std::atomic<bool> cancel{false};
  DiskCacheGroup::Options options{{good.string(), bad.string()}, 2ull << 20,
                                  "slab"};
  options.engine_factory =
      [&](const std::string& path, uint64_t) -> std::unique_ptr<StoreEngine> {
    cancel.store(true);
    if (path == bad.string()) throw std::runtime_error("cannot open this disk");
    return std::make_unique<DiskSlabStore>(CancelSlab(path));
  };
  auto server =
      KvNodeServerWiringTestPeer::Create(std::move(options), &cancel);
  EXPECT_TRUE(server->start_cancelled());
  EXPECT_FALSE(server->DisksHealthy());
  EXPECT_FALSE(server->Shutdown()) << "a disk failed to open: not all clean";
  server.reset();

  bool ok = false;
  DiskSlabStore reopened(CancelSlab(good.string()), &ok);
  ASSERT_TRUE(ok) << reopened.StartupError();
  EXPECT_EQ(reopened.GetStats().unclean_resets, 0u);
  EXPECT_EQ(reopened.GetStats().warm_restarts, 0u);
  for (const auto& dir : {good, bad}) {
    fs::remove_all(dir);
    fs::remove_all(dir.string() + "_tokens");
  }
}

// Without the flag, or with the flag never set, nothing changes.
TEST(RamTierWiring, AStartThatIsNotCancelledIsAnOrdinaryStart) {
  RamTierEnv env;
  const auto dir = fs::temp_directory_path() / "dfkv_ramwire_not_cancelled";
  fs::remove_all(dir);
  fs::create_directories(dir);
  std::atomic<bool> cancel{false};
  auto server = KvNodeServerWiringTestPeer::Create(
      DiskCacheGroup::Options{{dir.string()}, 1ull << 30, "file"}, &cancel);
  EXPECT_FALSE(server->start_cancelled());
  EXPECT_TRUE(server->Healthy());
  EXPECT_TRUE(server->ram_enabled());
  EXPECT_EQ(server->Start(0), Status::kOk);
  server.reset();
  fs::remove_all(dir);
}

// ---- The disks are marked in use only when the server is about to serve ----
namespace {
std::string SlabStateFile(const fs::path& dir) {
  std::ifstream in(dir / "slab_state", std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

// A slab directory with one value in it, closed clean; returns its state file.
std::string CleanSlabDir(const fs::path& dir, const BlockKey& key,
                         const std::string& value) {
  fs::remove_all(dir);
  fs::create_directories(dir);
  {
    DiskCacheGroup group({{dir.string()}, 1ull << 30, "slab"});
    EXPECT_TRUE(group.Healthy()) << group.StartupError();
    EXPECT_EQ(group.Cache(key, value.data(), value.size()), Status::kOk);
  }
  return SlabStateFile(dir);
}
}  // namespace

// A server that comes up has its disks activated by the time its constructor
// returns: the state file is the dirty one, and requests are taken.
TEST(RamTierWiring, AServerThatComesUpHasActivatedItsDisks) {
  const auto dir = fs::temp_directory_path() / "dfkv_ramwire_activated";
  const BlockKey key{31, 31};
  const std::string value(3000, 'k');
  const std::string clean = CleanSlabDir(dir, key, value);
  {
    auto server = KvNodeServerWiringTestPeer::Create(
        DiskCacheGroup::Options{{dir.string()}, 1ull << 30, "slab"});
    ASSERT_TRUE(server->Healthy());
    EXPECT_NE(SlabStateFile(dir), clean);
    EXPECT_EQ(server->Count(), 1u);
    ASSERT_EQ(server->Start(0), Status::kOk);
  }
  DiskCacheGroup reopened({{dir.string()}, 1ull << 30, "slab"});
  ASSERT_TRUE(reopened.Healthy()) << reopened.StartupError();
  EXPECT_EQ(reopened.SlabStats().unclean_resets, 0u);
  EXPECT_EQ(reopened.Count(), 1u);
  fs::remove_all(dir);
}

// A start that is cancelled, or fails on its RAM tier, never activates: the
// state file is byte for byte what the previous run wrote, while the server
// object lives and after it is shut down, and Shutdown reports clean.
TEST(RamTierWiring, AStartThatDoesNotGoAheadLeavesTheDisksAsFound) {
  RamTierEnv env;
  const auto dir = fs::temp_directory_path() / "dfkv_ramwire_as_found";
  const BlockKey key{32, 32};
  const std::string value(3000, 'f');
  for (const bool cancelled : {true, false}) {
    SCOPED_TRACE(cancelled ? "cancelled" : "RAM tier failed");
    ::unsetenv("DFKV_RAM_TIER_NUMA");
    const std::string clean = CleanSlabDir(dir, key, value);
    if (!cancelled)
      ::setenv("DFKV_RAM_TIER_NUMA", "neither-off-nor-interleave", 1);
    std::atomic<bool> cancel{cancelled};
    auto server = KvNodeServerWiringTestPeer::Create(
        DiskCacheGroup::Options{{dir.string()}, 1ull << 30, "slab"}, &cancel);
    EXPECT_EQ(server->start_cancelled(), cancelled);
    EXPECT_TRUE(server->DisksHealthy());
    EXPECT_FALSE(server->Healthy());
    EXPECT_EQ(server->Count(), 1u);
    EXPECT_EQ(SlabStateFile(dir), clean) << "the disk was marked in use";
    EXPECT_EQ(server->Start(0), Status::kIOError);
    EXPECT_TRUE(server->Shutdown());
    EXPECT_EQ(SlabStateFile(dir), clean);
    server.reset();
    EXPECT_EQ(SlabStateFile(dir), clean);

    DiskCacheGroup reopened({{dir.string()}, 1ull << 30, "slab"});
    ASSERT_TRUE(reopened.Healthy()) << reopened.StartupError();
    EXPECT_EQ(reopened.SlabStats().unclean_resets, 0u);
    EXPECT_EQ(reopened.SlabStats().warm_restarts, 0u);
    EXPECT_EQ(reopened.Count(), 1u);
  }
  fs::remove_all(dir);
}

// A disk whose activation fails: the server is unhealthy and refuses to
// start; shutting it down does not report every disk clean.
TEST(RamTierWiring, AServerWhoseDiskCannotBeActivatedRefusesToStart) {
  const auto good = fs::temp_directory_path() / "dfkv_ramwire_act_good";
  const auto bad = fs::temp_directory_path() / "dfkv_ramwire_act_bad";
  for (const auto& dir : {good, bad}) {
    fs::remove_all(dir);
    fs::remove_all(dir.string() + "_tokens");
  }
  DiskCacheGroup::Options options{{good.string(), bad.string()}, 2ull << 20,
                                  "slab"};
  options.engine_factory =
      [&](const std::string& path, uint64_t) -> std::unique_ptr<StoreEngine> {
    auto slab = CancelSlab(path);
    slab.defer_activation = true;
    if (path == bad.string()) return std::make_unique<CannotBeActivated>(slab);
    return std::make_unique<DiskSlabStore>(slab);
  };
  auto server = KvNodeServerWiringTestPeer::Create(std::move(options));
  EXPECT_FALSE(server->DisksHealthy());
  EXPECT_FALSE(server->Healthy());
  EXPECT_EQ(server->Start(0), Status::kIOError);
  EXPECT_FALSE(server->Shutdown());
  server.reset();
  // The disk that was activated was closed clean, the ordinary way.
  bool ok = false;
  DiskSlabStore reopened(CancelSlab(good.string()), &ok);
  ASSERT_TRUE(ok) << reopened.StartupError();
  EXPECT_EQ(reopened.GetStats().unclean_resets, 0u);
  EXPECT_EQ(reopened.GetStats().warm_restarts, 0u);
  for (const auto& dir : {good, bad}) {
    fs::remove_all(dir);
    fs::remove_all(dir.string() + "_tokens");
  }
}

// A disk on which nothing can be evicted (every slot held by an operation in
// flight): the server answers a PUT with kCacheFull on the request path the
// TCP handler and the RDMA handler share, and on the direct-write path of
// RDMA; it does not count an I/O error, and it stays healthy. Clients mark a
// node bad on kIOError only.
TEST(RamTierWiring, APutTheDiskHasNoSlotForIsCacheFullNotAnIoError) {
  const auto dir = fs::temp_directory_path() / "dfkv_ramwire_no_slot";
  fs::remove_all(dir);
  fs::remove_all(dir.string() + "_tokens");
  DiskSlabStore* slab = nullptr;
  // The store comes from the factory, so the group renders no per-slab
  // metrics for it; the store is asked directly below.
  DiskCacheGroup::Options options{{dir.string()}, 4 * 4096, "slab"};
  options.engine_factory =
      [&](const std::string& path, uint64_t) -> std::unique_ptr<StoreEngine> {
    auto store_options = CancelSlab(path);
    store_options.capacity_bytes = 4 * 4096;  // one extent, four slots
    store_options.extent_bytes = 4 * 4096;
    store_options.reclaim_interval_ms = 0;
    auto store = std::make_unique<DiskSlabStore>(store_options);
    slab = store.get();
    return store;
  };
  auto server = KvNodeServerWiringTestPeer::Create(std::move(options));
  ASSERT_TRUE(server->Healthy());
  ASSERT_NE(slab, nullptr);
  std::string value(4000, 'n');
  auto put = [&](uint64_t id) {
    std::string out;
    return server->ProcessRequestForKey(
        static_cast<uint8_t>(WireOp::kCache), BlockKey{id, 5}, 0, 0,
        value.data(), value.size(), &out);
  };
  for (uint64_t id = 0; id < 4; ++id) {
    ASSERT_EQ(put(id), Status::kOk);
    ASSERT_TRUE(DiskSlabStoreTestPeer::PinSlot(slab, BlockKey{id, 5}));
  }
  EXPECT_EQ(put(10), Status::kCacheFull);
  EXPECT_EQ(server->CacheDirectForKey(BlockKey{11, 5}, value.data(),
                                      value.size(), value.size()),
            Status::kCacheFull);
  EXPECT_TRUE(server->Healthy());
  EXPECT_EQ(slab->GetStats().put_no_slot, 2u);
  const std::string metrics = server->MetricsText();
  EXPECT_NE(metrics.find("dfkv_errors_total{op=\"put\",status=\"io\"} 0"),
            std::string::npos)
      << "a full disk was counted as an I/O error";
  for (uint64_t id = 0; id < 4; ++id)
    DiskSlabStoreTestPeer::UnpinSlot(slab, BlockKey{id, 5});
  EXPECT_EQ(put(10), Status::kOk);
  server.reset();
  fs::remove_all(dir);
  fs::remove_all(dir.string() + "_tokens");
}
}  // namespace

// ---- A RAM-acknowledging server whose disk refuses writes --------------------
namespace {
// A server with the RAM tier (RAM-ACK, the default) over slab stores whose
// writes the test can fail: `payload` for a write that fails and may work
// again, `record` for the metadata failure that fails a store for good.
struct FlushFailNode {
  RamTierEnv env;
  fs::path root;
  std::vector<DiskSlabStore*> stores;
  std::mutex mu;
  std::unique_ptr<KvNodeServer> server;

  explicit FlushFailNode(const char* name, size_t disks = 1,
                         uint64_t capacity = 64 << 20,
                         uint64_t extent = 1 << 20) {
    root = fs::temp_directory_path() / (std::string("dfkv_flushfail_") + name);
    fs::remove_all(root);
    fs::create_directories(root / "tokens");
    std::vector<std::string> dirs;
    for (size_t i = 0; i < disks; ++i)
      dirs.push_back((root / ("disk" + std::to_string(i))).string());
    stores.assign(disks, nullptr);
    DiskCacheGroup::Options options{dirs, disks * capacity, "slab"};
    options.engine_factory =
        [this, dirs, capacity, extent](
            const std::string& dir, uint64_t) -> std::unique_ptr<StoreEngine> {
      DiskSlabStore::Options o;
      o.dir = dir;
      o.trust_token_dir = (root / "tokens").string();
      o.capacity_bytes = capacity;
      o.extent_bytes = extent;
      o.slot_granularity = 4096;
      o.table_sync_ms = 0;
      o.reclaim_interval_ms = 0;
      auto store = std::make_unique<DiskSlabStore>(o);
      std::lock_guard<std::mutex> lk(mu);
      for (size_t i = 0; i < dirs.size(); ++i)
        if (dirs[i] == dir) stores[i] = store.get();
      return store;
    };
    server = KvNodeServerWiringTestPeer::Create(std::move(options));
  }
  ~FlushFailNode() {
    server.reset();
    fs::remove_all(root);
  }
  Status Put(uint64_t id) {
    const std::string value(4096, 'v');
    std::string out;
    return server->ProcessRequestForKey(static_cast<uint8_t>(WireOp::kCache),
                                        BlockKey{id, 0}, 0, 0, value.data(),
                                        value.size(), &out);
  }
  Status Get(uint64_t id) {
    std::string out;
    return server->ProcessRequestForKey(static_cast<uint8_t>(WireOp::kRange),
                                        BlockKey{id, 0}, 0, 4096, nullptr, 0,
                                        &out);
  }
  long Metric(const char* name) {
    return MetricVal(server->MetricsText(), name);
  }
  // RETAIN SET of one key; the key's code in the reply, or -1.
  int RetainSet(uint64_t id, uint32_t ttl_s) {
    const BlockKey key{id, 0};
    std::string payload(RetainPayloadBytes(1), '\0');
    EncodeRetainPayload(&payload[0], ttl_s, kRetainModeSet, &key, 1);
    std::string out;
    const Status st = server->ProcessRequestForKey(
        static_cast<uint8_t>(WireOp::kRetain), key, 0, 1, payload.data(),
        payload.size(), &out);
    return st == Status::kOk && !out.empty() ? static_cast<uint8_t>(out[0])
                                             : -1;
  }
  bool Settled() {
    return WaitFor([&] { return Metric("dfkv_ram_flush_backlog") == 0; });
  }
};
}  // namespace

// Once a disk's store has failed for good, a PUT for that disk is no longer
// acknowledged from RAM: it gets the store's I/O error. It used to be
// acknowledged, and its value dropped microseconds later.
TEST(RamTierFlushFailure, PutIsRefusedOnceItsDiskStoreHasFailed) {
  FlushFailNode node("refused");
  ASSERT_TRUE(node.server->ram_enabled());
  ASSERT_EQ(node.Put(1), Status::kOk);
  ASSERT_TRUE(node.Settled());
  ASSERT_EQ(node.Metric("dfkv_ram_flushed_total"), 1);

  // The store fails while one acknowledged value is on its way to it. That
  // one cannot be saved: it was acknowledged before anything had failed. It
  // is dropped at once (no retries against a failed store) and counted.
  DiskSlabStoreTestPeer::SetRecordWriteHook(node.stores[0],
                                            [] { return false; });
  ASSERT_EQ(node.Put(2), Status::kOk);
  ASSERT_TRUE(WaitFor([&] { return !node.server->DisksHealthy(); }));
  DiskSlabStoreTestPeer::SetRecordWriteHook(node.stores[0], {});
  ASSERT_TRUE(node.Settled());
  EXPECT_EQ(node.Metric("dfkv_ram_post_ack_flush_failures_total"), 1);
  EXPECT_EQ(node.Metric("dfkv_ram_flush_retries_total"), 0);

  const long acks = node.Metric("dfkv_ram_ack_total");
  for (uint64_t id = 10; id < 30; ++id)
    EXPECT_EQ(node.Put(id), Status::kIOError) << id;
  EXPECT_EQ(node.Metric("dfkv_ram_ack_total"), acks)
      << "nothing more was acknowledged from RAM";
  EXPECT_EQ(node.Metric("dfkv_ram_post_ack_flush_failures_total"), 1)
      << "and so nothing more was lost after an acknowledgement";
  EXPECT_EQ(node.Metric("dfkv_ram_flush_dropped_total"), 1);
  // What was flushed before the failure is still served from RAM.
  EXPECT_EQ(node.Get(1), Status::kOk);
}

// With several disks, a failed one hands its keys to the others: nothing is
// refused, and the value that was on its way to the failed disk when it
// failed is written to another one on its retry instead of being lost.
TEST(RamTierFlushFailure, AFailedDisksKeysGoToTheOtherDisk) {
  FlushFailNode node("otherdisks", 2);
  ASSERT_TRUE(node.server->ram_enabled());
  // The first slots.tbl write on either disk fails: key 1's, on its disk.
  std::atomic<bool> failed_one{false};
  for (DiskSlabStore* store : node.stores)
    DiskSlabStoreTestPeer::SetRecordWriteHook(
        store, [&] { return failed_one.exchange(true); });
  ASSERT_EQ(node.Put(1), Status::kOk);
  ASSERT_TRUE(WaitFor([&] { return node.server->FailedDisks() == 1; }));
  ASSERT_TRUE(node.Settled());
  for (DiskSlabStore* store : node.stores)
    DiskSlabStoreTestPeer::SetRecordWriteHook(store, {});
  ASSERT_NE(node.stores[0]->Healthy(), node.stores[1]->Healthy())
      << "exactly one store failed";
  EXPECT_TRUE(node.server->DisksHealthy());
  EXPECT_TRUE(node.server->DisksDegraded());
  EXPECT_EQ(node.Metric("dfkv_ram_post_ack_flush_failures_total"), 0);
  EXPECT_GE(node.Metric("dfkv_ram_flush_retries_total"), 1);
  EXPECT_EQ(node.Get(1), Status::kOk);

  for (uint64_t id = 100; id < 164; ++id)
    EXPECT_EQ(node.Put(id), Status::kOk) << id;
  ASSERT_TRUE(node.Settled());
  EXPECT_EQ(node.Metric("dfkv_ram_flushed_total"), 65)
      << "every acknowledged value reached a healthy disk";
  EXPECT_EQ(node.Metric("dfkv_ram_post_ack_flush_failures_total"), 0);
  EXPECT_EQ(node.Metric("dfkv_disks_failed"), 1);
  EXPECT_EQ(node.Metric("dfkv_server_degraded"), 1);
}

// A write that fails and then works: the acknowledged value is kept in RAM,
// written on a later attempt, and the node stays healthy. It used to be
// dropped after three attempts in a row, and the node unhealthy for good.
TEST(RamTierFlushFailure, DiskErrorThatPassesLosesNothing) {
  FlushFailNode node("passes");
  ASSERT_TRUE(node.server->ram_enabled());
  std::atomic<int> writes{0};
  // Fails for long enough to outlast three attempts in a row.
  DiskSlabStoreTestPeer::SetPayloadWriteHook(
      node.stores[0], [&] { return writes.fetch_add(1) >= 3; });
  ASSERT_EQ(node.Put(1), Status::kOk);
  ASSERT_TRUE(WaitFor([&] { return writes.load() >= 2; }));
  EXPECT_EQ(node.Get(1), Status::kOk) << "readable while its flush is retried";
  ASSERT_TRUE(WaitFor([&] {
    return node.Metric("dfkv_ram_flushed_total") == 1;
  }));
  EXPECT_EQ(writes.load(), 4);
  EXPECT_EQ(node.Metric("dfkv_ram_flush_retries_total"), 3);
  EXPECT_EQ(node.Metric("dfkv_ram_flush_dropped_total"), 0);
  EXPECT_EQ(node.Metric("dfkv_ram_post_ack_flush_failures_total"), 0);
  EXPECT_EQ(node.Metric("dfkv_ram_healthy"), 1);
  EXPECT_TRUE(node.server->Healthy());
  EXPECT_EQ(node.Get(1), Status::kOk);
}

// A disk that refuses writes, and clients that send the same keys again and
// again: none of those PUTs may be parked for the seconds the flush is being
// retried, and a PUT of another key that shares a put-key stripe with one of
// them must not be held up behind them.
TEST(RamTierFlushFailure, RepeatedPutsOfAKeyDoNotParkBehindItsFailingFlush) {
  FlushFailNode node("parked");
  ASSERT_TRUE(node.server->ram_enabled());
  DiskSlabStoreTestPeer::SetPayloadWriteHook(node.stores[0],
                                             [] { return false; });
  const uint64_t keys[] = {5001, 5002, 5003, 5004};
  for (uint64_t key : keys) ASSERT_EQ(node.Put(key), Status::kOk);
  // Every one of them is now between two attempts.
  ASSERT_TRUE(WaitFor([&] {
    return node.Metric("dfkv_ram_flush_retries_total") >= 4;
  }));

  std::atomic<bool> stop{false};
  std::atomic<long> slowest_us{0}, repeats{0}, not_ok{0};
  std::vector<std::thread> clients;
  for (int t = 0; t < 6; ++t) {
    clients.emplace_back([&, t] {
      for (int i = 0; !stop.load(); ++i) {
        const auto started = std::chrono::steady_clock::now();
        if (node.Put(keys[(t + i) % 4]) != Status::kOk) ++not_ok;
        const long us = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - started)
                            .count();
        long seen = slowest_us.load();
        while (us > seen && !slowest_us.compare_exchange_weak(seen, us)) {
        }
        ++repeats;
      }
    });
  }
  // The stripe is (digest_hi ^ digest_lo ^ tenant) % 1024: these keys share
  // theirs with the four above and are otherwise unrelated.
  long slowest_other_us = 0;
  for (int round = 0; round < 20; ++round) {
    const uint64_t other = keys[round % 4] + 1024ull * (round + 1);
    const auto started = std::chrono::steady_clock::now();
    EXPECT_EQ(node.Put(other), Status::kOk) << other;
    slowest_other_us = std::max<long>(
        slowest_other_us,
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started)
            .count());
    std::this_thread::sleep_for(20ms);
  }
  stop = true;
  for (auto& client : clients) client.join();
  EXPECT_GT(repeats.load(), 100);
  EXPECT_EQ(not_ok.load(), 0)
      << "the value is resident and acknowledged: so is a repeated PUT of it";
  EXPECT_LT(slowest_us.load(), 500000) << "a repeated PUT was parked";
  EXPECT_LT(slowest_other_us, 500000)
      << "a PUT of another key waited behind a parked one on its stripe";
}

// A disk that is healthy but has no slot for the value at the moment: every
// slot of its size class is held by an operation in flight. A value that was
// acknowledged from RAM stays in RAM and is written once a slot is free; it is
// not counted as lost and the tier does not report itself unhealthy. It used
// to be tried five times over 4.25 s like a failing disk and then dropped.
TEST(RamTierFlushFailure, AnAcknowledgedValueWaitsInRamForADiskWithNoSlot) {
  FlushFailNode node("noslot_ack", 1, 4 * 4096, 4 * 4096);  // four slots
  ASSERT_TRUE(node.server->ram_enabled());
  for (uint64_t id = 0; id < 4; ++id) {
    ASSERT_EQ(node.Put(id), Status::kOk);
    ASSERT_TRUE(node.Settled());
    ASSERT_TRUE(DiskSlabStoreTestPeer::PinSlot(node.stores[0], BlockKey{id, 0}));
  }
  ASSERT_EQ(node.Metric("dfkv_ram_flushed_total"), 4);
  ASSERT_EQ(node.Put(10), Status::kOk);  // acknowledged from RAM
  std::this_thread::sleep_for(5s);       // longer than all failure retries
  EXPECT_GT(node.Metric("dfkv_ram_flush_no_room_total"), 0);
  EXPECT_EQ(node.Metric("dfkv_ram_flush_dropped_total"), 0);
  EXPECT_EQ(node.Metric("dfkv_ram_post_ack_flush_failures_total"), 0);
  EXPECT_EQ(node.Metric("dfkv_ram_healthy"), 1);
  EXPECT_TRUE(node.server->Healthy());
  EXPECT_EQ(node.Get(10), Status::kOk) << "still served from RAM";

  for (uint64_t id = 0; id < 4; ++id)
    DiskSlabStoreTestPeer::UnpinSlot(node.stores[0], BlockKey{id, 0});
  bool flushed = false;
  for (int i = 0; i < 5 && !flushed; ++i)
    flushed = WaitFor([&] { return node.Metric("dfkv_ram_flushed_total") == 5; });
  EXPECT_TRUE(flushed) << "written once the disk had a slot again";
  EXPECT_TRUE(node.Settled());
  EXPECT_EQ(node.Metric("dfkv_ram_post_ack_flush_failures_total"), 0);
  EXPECT_EQ(node.Metric("dfkv_ram_healthy"), 1);
}

// The same disk with disk-ACK: the PUT waits for its flush and is answered
// kCacheFull, as the disk answers it without a RAM tier, at once and without
// making the tier unhealthy. It used to get kIOError after five attempts and
// leave the tier reporting itself unhealthy for 30 s.
TEST(RamTierFlushFailure, ADiskAckPutForADiskWithNoSlotIsCacheFull) {
  ::setenv("DFKV_PUT_ACK_MODE", "disk", 1);
  {
    FlushFailNode node("noslot_disk", 1, 4 * 4096, 4 * 4096);
    ASSERT_TRUE(node.server->ram_enabled());
    for (uint64_t id = 0; id < 4; ++id) {
      ASSERT_EQ(node.Put(id), Status::kOk);
      ASSERT_TRUE(
          DiskSlabStoreTestPeer::PinSlot(node.stores[0], BlockKey{id, 0}));
    }
    EXPECT_EQ(node.Put(10), Status::kCacheFull);
    EXPECT_TRUE(node.Settled());
    EXPECT_EQ(node.Metric("dfkv_ram_flush_retries_total"), 0);
    EXPECT_EQ(node.Metric("dfkv_ram_flush_dropped_total"), 0);
    EXPECT_EQ(node.Metric("dfkv_ram_healthy"), 1);
    EXPECT_TRUE(node.server->Healthy());
    EXPECT_EQ(
        MetricVal(node.server->MetricsText(),
                  "dfkv_errors_total{op=\"put\",status=\"io\"}"),
        0);
    for (uint64_t id = 0; id < 4; ++id)
      DiskSlabStoreTestPeer::UnpinSlot(node.stores[0], BlockKey{id, 0});
    EXPECT_EQ(node.Put(10), Status::kOk);
  }
  ::unsetenv("DFKV_PUT_ACK_MODE");
}

// A RETAIN SET for a value that is only in RAM is parked and applied when the
// value is written. A flush attempt the disk refuses for want of a slot, or
// one that fails and is tried again, must leave it parked: the value that
// lands later is the one the client asked to protect. It used to be thrown
// away on the first attempt that did not succeed.
TEST(RamTierFlushFailure, ARetainParkedForAValueWaitingForRoomIsApplied) {
  FlushFailNode node("retain_noroom", 1, 4 * 4096, 4 * 4096);
  ASSERT_TRUE(node.server->ram_enabled());
  for (uint64_t id = 0; id < 4; ++id) {
    ASSERT_EQ(node.Put(id), Status::kOk);
    ASSERT_TRUE(node.Settled());
    ASSERT_TRUE(DiskSlabStoreTestPeer::PinSlot(node.stores[0], BlockKey{id, 0}));
  }
  ASSERT_EQ(node.Put(10), Status::kOk);
  EXPECT_EQ(node.RetainSet(10, 600),
            static_cast<int>(SlabAllocator::RetainCode::kProtectedNew));
  ASSERT_TRUE(WaitFor([&] {
    return node.Metric("dfkv_ram_flush_no_room_total") >= 2;
  }));
  for (uint64_t id = 0; id < 4; ++id)
    DiskSlabStoreTestPeer::UnpinSlot(node.stores[0], BlockKey{id, 0});
  bool flushed = false;
  for (int i = 0; i < 5 && !flushed; ++i)
    flushed = WaitFor([&] { return node.Metric("dfkv_ram_flushed_total") == 5; });
  ASSERT_TRUE(flushed);
  EXPECT_EQ(node.Metric("dfkv_retain_pending_applied_total"), 1);
  EXPECT_NE(node.RetainSet(10, 600),
            static_cast<int>(SlabAllocator::RetainCode::kProtectedNew))
      << "the written value is protected already";
}

TEST(RamTierFlushFailure, ARetainParkedForAValueWhoseFlushIsRetriedIsApplied) {
  FlushFailNode node("retain_retry");
  ASSERT_TRUE(node.server->ram_enabled());
  std::atomic<int> writes{0};
  std::atomic<bool> fail{true};
  DiskSlabStoreTestPeer::SetPayloadWriteHook(node.stores[0], [&] {
    writes.fetch_add(1);
    return !fail.load();
  });
  ASSERT_EQ(node.Put(1), Status::kOk);
  EXPECT_EQ(node.RetainSet(1, 600),
            static_cast<int>(SlabAllocator::RetainCode::kProtectedNew));
  ASSERT_TRUE(WaitFor([&] { return writes.load() >= 2; }));
  fail = false;
  ASSERT_TRUE(WaitFor([&] {
    return node.Metric("dfkv_ram_flushed_total") == 1;
  }));
  EXPECT_EQ(node.Metric("dfkv_ram_post_ack_flush_failures_total"), 0);
  EXPECT_EQ(node.Metric("dfkv_retain_pending_applied_total"), 1);
}

// ---- REMOVE against a read promotion in flight (REQ-09) ---------------------
namespace {
// A file store whose reads and removes the test can hold: a read after its
// bytes are in the caller's buffer, a remove before it touches the disk.
class HoldingStore : public KVStore {
 public:
  HoldingStore(const std::string& path, uint64_t capacity)
      : KVStore(Options{path, capacity}) {}
  Status RangeDirect(const BlockKey& key, uint64_t offset, uint64_t length,
                     char* io_buf, size_t io_cap, const char** out_data,
                     size_t* out_len, size_t* value_len) override {
    const Status st = KVStore::RangeDirect(key, offset, length, io_buf, io_cap,
                                           out_data, out_len, value_len);
    if (hold_read.exchange(false)) read.Block();
    return st;
  }
  Status Remove(const BlockKey& key) override {
    if (hold_remove.exchange(false)) remove.Block();
    return KVStore::Remove(key);
  }
  std::atomic<bool> hold_read{false}, hold_remove{false};
  FlushBarrier read, remove;
};
}  // namespace

// A whole-value read of a key that is only on disk promotes the value into
// RAM when the read is done. A REMOVE of the key that returns while the read
// is in flight must not be undone by that promotion: after it, the key is
// gone from both tiers. The promotion used to install the value anyway, and
// EXISTS and GET then found a removed key. Both orders: the REMOVE completes
// during the disk read, and the promotion lands while the REMOVE is between
// the RAM tier and the disk.
TEST(RamTierWiring, ARemoveIsNotUndoneByAReadPromotionInFlight) {
  for (bool promotion_during_remove : {false, true}) {
    SCOPED_TRACE(promotion_during_remove ? "promotion during the remove"
                                         : "remove during the read");
    const auto dir = fs::temp_directory_path() /
        ("dfkv_remove_promote_" + std::to_string(promotion_during_remove));
    fs::remove_all(dir);
    fs::create_directories(dir);
    HoldingStore* store = nullptr;
    DiskCacheGroup::Options options{{dir.string()}, 1ull << 30, "file"};
    options.engine_factory = [&](const std::string& path, uint64_t capacity) {
      auto engine = std::make_unique<HoldingStore>(path, capacity);
      store = engine.get();
      return engine;
    };
    auto server = KvNodeServerWiringTestPeer::Create(std::move(options));
    KvNodeServerWiringTestPeer::SetCoalescing(*server, false);
    KvNodeServerWiringTestPeer::SetRam(
        *server, [](const BlockKey&, char*, size_t, size_t) {
          return Status::kOk;
        });
    const BlockKey key{8301, 0};
    const std::string value(4096 + 111, 'q');
    ASSERT_EQ(KvNodeServerWiringTestPeer::Persist(*server, key, value.data(),
                                                  value.size()),
              Status::kOk);
    alignas(4096) static char staging[16 * 1024];
    auto read = [&] {
      const char* data = nullptr;
      size_t bytes = 0, value_len = 0;
      return server->RangeDirectForKey(key, 0, value.size(), staging,
                                       sizeof(staging), &data, &bytes,
                                       &value_len);
    };
    auto remove = [&] {
      std::string out;
      return server->ProcessRequestForKey(static_cast<uint8_t>(WireOp::kRemove),
                                          key, 0, 0, nullptr, 0, &out);
    };
    if (!promotion_during_remove) {
      store->hold_read = true;
      std::thread reader([&] { EXPECT_EQ(read(), Status::kOk); });
      ASSERT_TRUE(store->read.AwaitEntry());
      EXPECT_EQ(remove(), Status::kOk);
      store->read.Open();
      reader.join();
    } else {
      store->hold_remove = true;
      Status removed = Status::kIOError;
      std::thread remover([&] { removed = remove(); });
      ASSERT_TRUE(store->remove.AwaitEntry());  // RAM done, disk not yet
      EXPECT_EQ(read(), Status::kOk);           // promotes: disk still has it
      store->remove.Open();
      remover.join();
      EXPECT_EQ(removed, Status::kOk);
    }
    EXPECT_FALSE(KvNodeServerWiringTestPeer::Resident(*server, key))
        << "the removed key came back into RAM";
    std::string out;
    EXPECT_EQ(server->ProcessRequestForKey(static_cast<uint8_t>(WireOp::kRange),
                                           key, 0, 0, nullptr, 0, &out),
              Status::kNotFound);
    server.reset();
    fs::remove_all(dir);
  }
}


// The same read, and between its disk read and its promotion the key is
// removed and written again with a new value that reaches the disk without
// passing through RAM (a quota tenant, a capacity bypass). The promotion
// carries the old value and must not be installed over the new one: a GET
// returns what the disk now holds. It used to return the old value from RAM.
TEST(RamTierWiring, AReadPromotionDoesNotInstallAValueTheDiskHasReplaced) {
  const auto dir = fs::temp_directory_path() / "dfkv_promote_replaced";
  fs::remove_all(dir);
  fs::create_directories(dir);
  HoldingStore* store = nullptr;
  DiskCacheGroup::Options options{{dir.string()}, 1ull << 30, "file"};
  options.engine_factory = [&](const std::string& path, uint64_t capacity) {
    auto engine = std::make_unique<HoldingStore>(path, capacity);
    store = engine.get();
    return engine;
  };
  auto server = KvNodeServerWiringTestPeer::Create(std::move(options));
  KvNodeServerWiringTestPeer::SetCoalescing(*server, false);
  KvNodeServerWiringTestPeer::SetRam(
      *server, [](const BlockKey&, char*, size_t, size_t) {
        return Status::kOk;
      });
  const BlockKey key{8302, 0};
  const std::string v1(4096 + 111, '1'), v2(4096 + 111, '2');
  ASSERT_EQ(KvNodeServerWiringTestPeer::Persist(*server, key, v1.data(),
                                                v1.size()),
            Status::kOk);
  alignas(4096) static char staging[16 * 1024];
  store->hold_read = true;
  std::thread reader([&] {
    const char* data = nullptr;
    size_t bytes = 0, value_len = 0;
    EXPECT_EQ(server->RangeDirectForKey(key, 0, v1.size(), staging,
                                        sizeof(staging), &data, &bytes,
                                        &value_len),
              Status::kOk);
  });
  ASSERT_TRUE(store->read.AwaitEntry());  // has read v1
  std::string out;
  ASSERT_EQ(server->ProcessRequestForKey(static_cast<uint8_t>(WireOp::kRemove),
                                         key, 0, 0, nullptr, 0, &out),
            Status::kOk);
  ASSERT_EQ(KvNodeServerWiringTestPeer::Persist(*server, key, v2.data(),
                                                v2.size()),
            Status::kOk);  // disk only
  store->read.Open();
  reader.join();
  out.clear();
  ASSERT_EQ(server->ProcessRequestForKey(static_cast<uint8_t>(WireOp::kRange),
                                         key, 0, 0, nullptr, 0, &out),
            Status::kOk);
  EXPECT_EQ(out, v2) << "the old value was installed over the new one";
  server.reset();
  fs::remove_all(dir);
}

// The same race through a coalesced read: a second reader of the key arrives
// after the REMOVE and the new value, takes the new value's generation, and
// joins the read that is still running for the old one. It gets the old
// bytes (it overlapped the change), but must not promote them under the new
// generation: a GET afterwards returns what the disk holds.
TEST(RamTierWiring, ACoalescedReaderDoesNotPromoteTheLeadersOldValue) {
  const auto dir = fs::temp_directory_path() / "dfkv_promote_follower";
  fs::remove_all(dir);
  fs::create_directories(dir);
  HoldingStore* store = nullptr;
  DiskCacheGroup::Options options{{dir.string()}, 1ull << 30, "file"};
  options.engine_factory = [&](const std::string& path, uint64_t capacity) {
    auto engine = std::make_unique<HoldingStore>(path, capacity);
    store = engine.get();
    return engine;
  };
  auto server = KvNodeServerWiringTestPeer::Create(std::move(options));
  KvNodeServerWiringTestPeer::SetCoalescing(*server, true);
  KvNodeServerWiringTestPeer::SetRam(
      *server, [](const BlockKey&, char*, size_t, size_t) {
        return Status::kOk;
      });
  const BlockKey key{8303, 0};
  const std::string v1(4096 + 111, '1'), v2(4096 + 111, '2');
  ASSERT_EQ(KvNodeServerWiringTestPeer::Persist(*server, key, v1.data(),
                                                v1.size()),
            Status::kOk);
  auto read = [&](char* buf, size_t cap) {
    const char* data = nullptr;
    size_t bytes = 0, value_len = 0;
    return server->RangeDirectForKey(key, 0, v1.size(), buf, cap, &data,
                                     &bytes, &value_len);
  };
  alignas(4096) static char leader_buf[16 * 1024];
  alignas(4096) static char follower_buf[16 * 1024];
  store->hold_read = true;
  std::thread leader([&] {
    EXPECT_EQ(read(leader_buf, sizeof(leader_buf)), Status::kOk);
  });
  ASSERT_TRUE(store->read.AwaitEntry());  // the leader has read v1
  std::string out;
  ASSERT_EQ(server->ProcessRequestForKey(static_cast<uint8_t>(WireOp::kRemove),
                                         key, 0, 0, nullptr, 0, &out),
            Status::kOk);
  ASSERT_EQ(KvNodeServerWiringTestPeer::Persist(*server, key, v2.data(),
                                                v2.size()),
            Status::kOk);  // disk only
  std::thread follower([&] {
    EXPECT_EQ(read(follower_buf, sizeof(follower_buf)), Status::kOk);
  });
  // The follower joins the flight; it waits up to 500 ms for the leader.
  std::this_thread::sleep_for(100ms);
  store->read.Open();
  leader.join();
  follower.join();
  out.clear();
  ASSERT_EQ(server->ProcessRequestForKey(static_cast<uint8_t>(WireOp::kRange),
                                         key, 0, 0, nullptr, 0, &out),
            Status::kOk);
  EXPECT_EQ(out, v2) << "the leader's old value was promoted by a follower";
  server.reset();
  fs::remove_all(dir);
}

// ---- RETAIN for a value that is in RAM but not dirty (REQ-08) ---------------
// A value that is in RAM as a written (durable) copy while the disk no longer
// has it: a RETAIN SET can protect nothing. It is a hit, unprotected, and
// nothing is parked; no flush will ever come to apply a parked SET. It used
// to be answered kProtectedNew and parked for good.
TEST(RamTierFlushFailure, ARetainForAWrittenRamCopyIsNotParked) {
  FlushFailNode node("retain_durable");
  ASSERT_TRUE(node.server->ram_enabled());
  ASSERT_EQ(node.Put(1), Status::kOk);
  ASSERT_TRUE(node.Settled());
  ASSERT_EQ(node.Metric("dfkv_ram_flushed_total"), 1);
  ASSERT_EQ(node.stores[0]->Remove(BlockKey{1, 0}), Status::kOk);  // disk only
  ASSERT_TRUE(KvNodeServerWiringTestPeer::Resident(*node.server, BlockKey{1, 0}));
  EXPECT_EQ(node.RetainSet(1, 600),
            static_cast<int>(SlabAllocator::RetainCode::kHitUnprotected));
  EXPECT_EQ(KvNodeServerWiringTestPeer::PendingRetains(*node.server), 0u);
}

// ---- Values the disk will never take (STO-30) -------------------------------
// A value larger than the disk's extent can never be stored there. It must
// not be acknowledged from RAM and then dropped as acknowledged data lost
// with the tier unhealthy: the PUT is answered kInvalid, as the disk answers
// it, and the node stays healthy.
TEST(RamTierFlushFailure, AValueTooLargeForTheDiskIsNotAcknowledgedAndLost) {
  FlushFailNode node("too_large");  // 1 MiB extents
  ASSERT_TRUE(node.server->ram_enabled());
  const std::string value((1 << 20) + 4096, 'L');
  std::string out;
  EXPECT_EQ(node.server->ProcessRequestForKey(
                static_cast<uint8_t>(WireOp::kCache), BlockKey{7, 0}, 0, 0,
                value.data(), value.size(), &out),
            Status::kInvalid);
  std::this_thread::sleep_for(5s);  // longer than all failure retries
  EXPECT_EQ(node.Metric("dfkv_ram_post_ack_flush_failures_total"), 0);
  EXPECT_EQ(node.Metric("dfkv_ram_flush_dropped_total"), 0);
  EXPECT_EQ(node.Metric("dfkv_ram_healthy"), 1);
  EXPECT_TRUE(node.server->Healthy());
  EXPECT_EQ(node.Put(8), Status::kOk) << "an ordinary value is still taken";
}

// The production write path end to end: a RAM-acknowledged PUT flushed by the
// RAM tier through DiskCacheGroup to a slab store in direct (O_DIRECT) mode.
// Unit tests run with DFKV_SLAB_WRITE=buffered by default; this one pins
// direct so that path stays covered.
TEST(RamTierWiring, ARamFlushReachesADirectModeSlabThroughTheGroup) {
  RamTierEnv env;
  ::setenv("DFKV_SLAB_WRITE", "direct", 1);
  const auto dir = fs::temp_directory_path() / "dfkv_ramwire_direct_slab";
  fs::remove_all(dir);
  fs::create_directories(dir);
  {
    auto server = KvNodeServerWiringTestPeer::Create(
        DiskCacheGroup::Options{{dir.string()}, 1ull << 30, "slab"});
    ASSERT_TRUE(server->Healthy());
    ASSERT_TRUE(server->ram_enabled());
    const std::string metrics = server->MetricsText();
    if (metrics.find("write_mode=\"direct\"") == std::string::npos) {
      ::unsetenv("DFKV_SLAB_WRITE");
      GTEST_SKIP() << "the test filesystem does not take O_DIRECT";
    }
    const BlockKey key{8401, 0};
    const std::string value(3 * 4096 + 100, 'd');
    std::string out;
    ASSERT_EQ(server->ProcessRequestForKey(static_cast<uint8_t>(WireOp::kCache),
                                           key, 0, 0, value.data(),
                                           value.size(), &out),
              Status::kOk);
    ASSERT_TRUE(WaitFor([&] {
      return MetricVal(server->MetricsText(), "dfkv_ram_flushed_total") == 1;
    }));
    EXPECT_EQ(MetricVal(server->MetricsText(),
                        "dfkv_slab_dio_write_fallback_total"),
              0);
    std::string disk;
    ASSERT_EQ(KvNodeServerWiringTestPeer::DiskRead(*server, key, &disk),
              Status::kOk);
    EXPECT_EQ(disk, value);
  }
  ::unsetenv("DFKV_SLAB_WRITE");
  fs::remove_all(dir);
}

// ---- The RAM tier and a disk that evicted its copy (REQ-17) -----------------
// A value in RAM as a written copy whose disk copy is gone (the disk evicted
// it, or it was removed from the disk alone). A PUT of the key is not a
// duplicate of something stored: the stale copy is dropped and the PUT is
// written to the disk. It used to be answered kOk as a duplicate of the RAM copy and
// never reached the disk, so the key was lost once RAM evicted it.
TEST(RamTierFlushFailure, APutOfAKeyTheDiskNoLongerHasReachesTheDisk) {
  FlushFailNode node("disk_evicted");
  ASSERT_TRUE(node.server->ram_enabled());
  const BlockKey key{1, 0};
  ASSERT_EQ(node.Put(1), Status::kOk);
  ASSERT_TRUE(node.Settled());
  ASSERT_EQ(node.Metric("dfkv_ram_flushed_total"), 1);
  ASSERT_EQ(node.stores[0]->Remove(key), Status::kOk);  // disk only
  ASSERT_TRUE(KvNodeServerWiringTestPeer::Resident(*node.server, key));
  ASSERT_FALSE(node.stores[0]->IsCached(key));
  EXPECT_EQ(node.Put(1), Status::kOk);
  ASSERT_TRUE(node.Settled());
  EXPECT_TRUE(node.stores[0]->IsCached(key)) << "the PUT never reached the disk";
  EXPECT_FALSE(KvNodeServerWiringTestPeer::Resident(*node.server, key))
      << "the stale RAM copy is gone; the PUT was written to the disk itself";
  EXPECT_EQ(node.Get(1), Status::kOk);
}

// The lock that serializes PUTs is per key: a PUT of another key does not
// wait for it, also for keys that shared one of the former 1024 stripes.
TEST(RamTierWiring, PutKeyLockSerializesOnlyItsKey) {
  auto dir = fs::temp_directory_path() / "dfkv_ramwire_putkey";
  fs::remove_all(dir);
  fs::create_directories(dir);
  DiskCacheGroup::Options options{{dir.string()}, 1ull << 30, "file"};
  auto server = KvNodeServerWiringTestPeer::Create(std::move(options));
  auto* node = server.get();
  KvNodeServerWiringTestPeer::SetLimit(*server, 0);
  KvNodeServerWiringTestPeer::SetRam(
      *server,
      [node](const BlockKey& key, char* data, size_t len, size_t) {
        return KvNodeServerWiringTestPeer::Persist(*node, key, data, len);
      },
      /*ram_ack=*/true);
  const BlockKey held{1, 0}, other{1025, 0};  // (hi ^ lo) % 1024 alike
  alignas(4096) static char value[4096];
  std::memset(value, 'v', sizeof(value));
  auto put = [&](const BlockKey& key) {
    return std::async(std::launch::async, [&, key] {
      return server->CacheDirectForKey(key, value, sizeof(value),
                                       sizeof(value));
    });
  };
  // Declared before the lock, so that a failed check releases it first.
  std::future<Status> unrelated, same;
  auto lock = KvNodeServerWiringTestPeer::LockPut(*server, held);
  unrelated = put(other);
  ASSERT_EQ(unrelated.wait_for(2s), std::future_status::ready)
      << "a PUT waited for another key's lock";
  EXPECT_EQ(unrelated.get(), Status::kOk);
  same = put(held);
  EXPECT_EQ(same.wait_for(100ms), std::future_status::timeout);
  lock.reset();
  ASSERT_EQ(same.wait_for(2s), std::future_status::ready);
  EXPECT_EQ(same.get(), Status::kOk);
  EXPECT_EQ(MetricVal(server->MetricsText(), "dfkv_put_key_lock_waits_total"),
            1);
  server.reset();
  fs::remove_all(dir);
}

// ---- CLI-35 publication-lock disk-I/O microbenchmarks -----------------------
// These are measurements, not regression gates: no result depends on a timing
// threshold. Only the deliberate "still blocked" check has a short deadline.
namespace {

bool PublicationBenchEnabled() {
  const char* enabled = std::getenv("DFKV_PUBLICATION_BENCH");
  return enabled != nullptr && std::string(enabled) == "1";
}

class ScopedBenchEnv {
 public:
  ScopedBenchEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) {
      had_old_ = true;
      old_ = old;
    }
    if (value)
      ::setenv(name, value, 1);
    else
      ::unsetenv(name);
  }
  ~ScopedBenchEnv() {
    if (had_old_)
      ::setenv(name_, old_.c_str(), 1);
    else
      ::unsetenv(name_);
  }

 private:
  const char* name_;
  bool had_old_ = false;
  std::string old_;
};

using BenchClock = std::chrono::steady_clock;

double BenchUs(BenchClock::duration elapsed) {
  return std::chrono::duration<double, std::micro>(elapsed).count();
}

struct PublicationNode {
  fs::path dir;
  std::unique_ptr<KvNodeServer> server;
  std::string addr;
  bool started = false;
  ~PublicationNode() {
    server.reset();
    fs::remove_all(dir);
  }
};

using BenchEngineFactory =
    std::function<std::unique_ptr<StoreEngine>(const std::string&, uint64_t)>;

std::unique_ptr<PublicationNode> StartPublicationNode(
    const std::string& tag, bool write_memory,
    BenchEngineFactory factory = {}) {
  auto node = std::make_unique<PublicationNode>();
  node->dir = fs::temp_directory_path() /
              ("dfkv_pubbench_" + tag + "_" + std::to_string(::getpid()));
  fs::remove_all(node->dir);
  fs::create_directories(node->dir);
  ScopedBenchEnv remove_window("DFKV_REMOVE_MEMORY_MS", "600000");
  ScopedBenchEnv write_entries("DFKV_WRITE_MEMORY_ENTRIES",
                               write_memory ? "65536" : nullptr);
  ScopedBenchEnv no_ram("DFKV_RAM_TIER", nullptr);
  ScopedBenchEnv quiet_logs("DFKV_LOG", "ERROR");
  DiskCacheGroup::Options options{{node->dir.string()}, 1ull << 30, "file"};
  options.engine_factory = std::move(factory);
  node->server = KvNodeServerWiringTestPeer::Create(std::move(options));
  if (node->server->Healthy())
    node->started = node->server->Start(0) == Status::kOk;
  if (node->started)
    node->addr = "127.0.0.1:" + std::to_string(node->server->port());
  return node;
}

bool PublicationKeysDistinct(const std::vector<BlockKey>& keys) {
  for (size_t i = 0; i < keys.size(); ++i)
    for (size_t j = 0; j < i; ++j)
      if (keys[i] == keys[j]) return false;
  return true;
}

bool OnePublicationBucket(const KvNodeServer& server,
                          const std::vector<BlockKey>& keys) {
  if (keys.empty()) return false;
  const size_t bucket =
      KvNodeServerWiringTestPeer::PublicationBucket(server, keys[0]);
  for (const auto& key : keys)
    if (KvNodeServerWiringTestPeer::PublicationBucket(server, key) != bucket)
      return false;
  return true;
}

bool SpreadPublicationBuckets(const KvNodeServer& server,
                              const std::vector<BlockKey>& keys) {
  if (keys.empty()) return false;
  std::vector<bool> seen(64, false);
  for (const auto& key : keys) {
    const size_t bucket =
        KvNodeServerWiringTestPeer::PublicationBucket(server, key);
    if (seen[bucket]) return false;
    seen[bucket] = true;
  }
  return true;
}

std::vector<BlockKey> KeysInOneBucket(const KvNodeServer& server,
                                      const std::string& key_namespace,
                                      size_t count) {
  const BlockKey first = ToBlockKey(key_namespace, "collision-0");
  const size_t bucket =
      KvNodeServerWiringTestPeer::PublicationBucket(server, first);
  std::vector<BlockKey> keys{first};
  for (uint64_t serial = 1; keys.size() < count; ++serial) {
    auto key = ToBlockKey(key_namespace,
                          "collision-" + std::to_string(serial));
    if (KvNodeServerWiringTestPeer::PublicationBucket(server, key) == bucket)
      keys.push_back(key);
  }
  return keys;
}

std::vector<BlockKey> KeysAcrossBuckets(const KvNodeServer& server,
                                        const std::string& key_namespace,
                                        size_t count) {
  std::vector<BlockKey> keys;
  std::vector<bool> used(64, false);
  for (uint64_t serial = 0; keys.size() < count; ++serial) {
    auto key = ToBlockKey(key_namespace, "spread-" + std::to_string(serial));
    const size_t bucket =
        KvNodeServerWiringTestPeer::PublicationBucket(server, key);
    if (!used[bucket]) {
      used[bucket] = true;
      keys.push_back(key);
    }
  }
  return keys;
}

struct CallStats {
  std::vector<double> us;
  size_t failures = 0;
};

struct ClientCallStats {
  std::vector<double> put_us;
  std::vector<double> remove_us;
  size_t put_failures = 0;
  size_t remove_failures = 0;
};

double SortedCallPercentile(const std::vector<double>& samples,
                            size_t percentile) {
  if (samples.empty()) return 0;
  const size_t rank = (percentile * samples.size() + 99) / 100;
  return samples[std::min(samples.size() - 1, rank - 1)];
}

void PrintCalls(const std::string& scenario, int repetition, bool write_memory,
                const char* operation, const CallStats& stats, double wall_us,
                size_t keys = 0) {
  std::vector<double> sorted = stats.us;
  std::sort(sorted.begin(), sorted.end());
  const double p50 = SortedCallPercentile(sorted, 50);
  const double p99 = SortedCallPercentile(sorted, 99);
  const double max = sorted.empty() ? 0 : sorted.back();
  const double calls_per_second =
      wall_us > 0 ? stats.us.size() * 1000000.0 / wall_us : 0;
  const double keys_per_second =
      wall_us > 0 ? keys * 1000000.0 / wall_us : 0;
  std::printf(
      "PUBLICATION_BENCH scenario=%s rep=%d W=%s op=%s calls=%zu keys=%zu "
      "p50_us=%.3f p99_us=%.3f max_us=%.3f failures=%zu wall_us=%.3f "
      "calls_per_s=%.3f keys_per_s=%.3f\n",
      scenario.c_str(), repetition, write_memory ? "on" : "off", operation,
      stats.us.size(), keys, p50, p99, max, stats.failures, wall_us,
      calls_per_second, keys_per_second);
}

struct AlternatingResult {
  bool started = false;
  bool write_memory_matches = false;
  bool layout_matches = false;
  bool final_value_matches = false;
  CallStats put;
  CallStats remove;
  double wall_us = 0;
};

AlternatingResult MeasureAlternating(const std::string& scenario,
                                     int repetition, bool write_memory,
                                     const std::vector<BlockKey>& keys,
                                     bool spread) {
  constexpr size_t kClients = 4;
  constexpr size_t kIterations = 128;
  const std::string tag = scenario + "_" + std::to_string(repetition) + "_" +
                          (write_memory ? "on" : "off");
  auto node = StartPublicationNode(tag, write_memory);
  AlternatingResult result;
  result.started = node->started;
  if (!node->started) return result;
  result.write_memory_matches =
      KvNodeServerWiringTestPeer::HasWriteMemory(*node->server) == write_memory;
  result.layout_matches = PublicationKeysDistinct(keys) &&
                          (spread ? SpreadPublicationBuckets(*node->server, keys)
                                  : OnePublicationBucket(*node->server, keys));

  std::vector<ClientCallStats> clients(kClients);
  std::atomic<bool> go{false};
  auto writer = [&](size_t client) {
    clients[client].put_us.reserve(kIterations);
    clients[client].remove_us.reserve(kIterations);
    TcpTransport transport;
    const std::string value(128, 'v');
    while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
    for (size_t i = 0; i < kIterations; ++i) {
      const BlockKey& key = keys[(client * kIterations + i) % keys.size()];
      auto started = BenchClock::now();
      const Status put = transport.Cache(node->addr, key, value.data(), value.size());
      clients[client].put_us.push_back(BenchUs(BenchClock::now() - started));
      if (put != Status::kOk) ++clients[client].put_failures;
      started = BenchClock::now();
      const Status removed = transport.Remove(node->addr, key);
      clients[client].remove_us.push_back(BenchUs(BenchClock::now() - started));
      // Another client may legitimately have removed the shared key first.
      if (removed != Status::kOk && removed != Status::kNotFound)
        ++clients[client].remove_failures;
    }
  };
  std::vector<std::thread> threads;
  threads.reserve(kClients);
  for (size_t client = 0; client < kClients; ++client)
    threads.emplace_back(writer, client);
  const auto wall_started = BenchClock::now();
  go.store(true, std::memory_order_release);
  for (auto& thread : threads) thread.join();
  result.wall_us = BenchUs(BenchClock::now() - wall_started);
  for (auto& client : clients) {
    result.put.us.insert(result.put.us.end(), client.put_us.begin(),
                         client.put_us.end());
    result.remove.us.insert(result.remove.us.end(), client.remove_us.begin(),
                            client.remove_us.end());
    result.put.failures += client.put_failures;
    result.remove.failures += client.remove_failures;
  }

  // End-to-end correctness is not included in the measured loop.
  TcpTransport check;
  const std::string value(128, 'z');
  std::string readback;
  const Status put = check.Cache(node->addr, keys[0], value.data(), value.size());
  const Status range = check.Range(node->addr, keys[0], 0, value.size(), &readback);
  result.final_value_matches = put == Status::kOk && range == Status::kOk &&
                               readback == value;
  if (!result.final_value_matches) ++result.put.failures;
  PrintCalls(scenario, repetition, write_memory, "PUT", result.put,
             result.wall_us);
  PrintCalls(scenario, repetition, write_memory, "REMOVE", result.remove,
             result.wall_us);
  return result;
}

TEST(RamTierWiring, PublicationDiskIoThroughputBenchmark) {
  if (!PublicationBenchEnabled()) {
    GTEST_SKIP() << "set DFKV_PUBLICATION_BENCH=1";
  }
  auto layout = StartPublicationNode("layout", true);
  ASSERT_TRUE(layout->started);
  const std::string ns = "publication/bench";
  const std::vector<BlockKey> same_key{ToBlockKey(ns, "same-key")};
  const auto collisions = KeysInOneBucket(*layout->server, ns, 16);
  const auto spread = KeysAcrossBuckets(*layout->server, ns, 16);
  ASSERT_TRUE(PublicationKeysDistinct(collisions));
  ASSERT_TRUE(PublicationKeysDistinct(spread));
  ASSERT_TRUE(OnePublicationBucket(*layout->server, collisions));
  ASSERT_TRUE(SpreadPublicationBuckets(*layout->server, spread));
  for (int rep = 0; rep < 5; ++rep) {
    for (const bool write_memory : {true, false}) {
      const auto same = MeasureAlternating("same_key", rep, write_memory,
                                           same_key, false);
      EXPECT_TRUE(same.started && same.write_memory_matches &&
                  same.layout_matches && same.final_value_matches);
      EXPECT_EQ(same.put.failures, 0u);
      EXPECT_EQ(same.remove.failures, 0u);
      const auto collision = MeasureAlternating("same_bucket", rep,
                                                write_memory, collisions, false);
      EXPECT_TRUE(collision.started && collision.write_memory_matches &&
                  collision.layout_matches && collision.final_value_matches);
      EXPECT_EQ(collision.put.failures, 0u);
      EXPECT_EQ(collision.remove.failures, 0u);
      const auto separated = MeasureAlternating("spread_buckets", rep,
                                                write_memory, spread, true);
      EXPECT_TRUE(separated.started && separated.write_memory_matches &&
                  separated.layout_matches && separated.final_value_matches);
      EXPECT_EQ(separated.put.failures, 0u);
      EXPECT_EQ(separated.remove.failures, 0u);
    }
  }
}

struct AgesCaseResult {
  bool layout_matches = false;
  uint64_t server_requests = 0;
  uint64_t expected_requests = 0;
  size_t keys = 0;
  double wall_us = 0;
  CallStats calls;
};

AgesCaseResult MeasureAgesCase(KvNodeServer& server, const std::string& addr,
                               const std::string& scenario, int repetition,
                               bool write_memory,
                               const std::vector<BlockKey>& keys,
                               bool duplicate) {
  AgesCaseResult result;
  result.keys = keys.size();
  result.layout_matches =
      !keys.empty() && OnePublicationBucket(server, keys) &&
      (duplicate || PublicationKeysDistinct(keys)) &&
      (!duplicate || std::all_of(keys.begin(), keys.end(), [&](const BlockKey& key) {
        return key == keys[0];
      }));
  result.expected_requests = (keys.size() + kRemovedWithinMaxKeys - 1) /
                             kRemovedWithinMaxKeys;
  const uint64_t requests_before =
      KvNodeServerWiringTestPeer::RemovedWithinAgesRequests(server);
  TcpTransport transport;
  const auto wall_started = BenchClock::now();
  for (size_t begin = 0; begin < keys.size(); begin += kRemovedWithinMaxKeys) {
    const size_t count =
        std::min(kRemovedWithinMaxKeys, keys.size() - begin);
    const std::string payload =
        EncodeRemovedWithinPayload(600000, keys.data() + begin, count);
    std::string out;
    const auto started = BenchClock::now();
    const Status status = transport.RemovedWithinAges(
        addr, keys[begin].tenant_hash, count, payload, &out);
    result.calls.us.push_back(BenchUs(BenchClock::now() - started));
    bool valid = status == Status::kOk &&
                 out.size() == 4 + count * kRemovedWithinAgesKeyBytes;
    for (size_t i = 0; valid && i < count; ++i) {
      const uint32_t removed_age = net::GetU32(out.data() + 4 + i * 8);
      valid = (removed_age == kRemovedAgeNone ||
               removed_age == kRemovedAgeUnknown) &&
              net::GetU32(out.data() + 8 + i * 8) == kRemovedAgeNone;
    }
    if (!valid) ++result.calls.failures;
  }
  result.wall_us = BenchUs(BenchClock::now() - wall_started);
  result.server_requests =
      KvNodeServerWiringTestPeer::RemovedWithinAgesRequests(server) -
      requests_before;
  PrintCalls(scenario + "_n" + std::to_string(keys.size()), repetition,
             write_memory, "OP26", result.calls, result.wall_us, result.keys);
  return result;
}

TEST(RamTierWiring, PublicationAgesBatchThroughputBenchmark) {
  if (!PublicationBenchEnabled()) {
    GTEST_SKIP() << "set DFKV_PUBLICATION_BENCH=1";
  }
  auto layout = StartPublicationNode("batch_layout", true);
  ASSERT_TRUE(layout->started);
  const auto pool = KeysInOneBucket(*layout->server, "publication/batch", 1025);
  ASSERT_TRUE(PublicationKeysDistinct(pool));
  ASSERT_TRUE(OnePublicationBucket(*layout->server, pool));
  for (int rep = 0; rep < 5; ++rep) {
    for (const bool write_memory : {true, false}) {
      auto node = StartPublicationNode(
          "batch_" + std::to_string(rep) + (write_memory ? "_on" : "_off"),
          write_memory);
      ASSERT_TRUE(node->started);
      ASSERT_EQ(KvNodeServerWiringTestPeer::HasWriteMemory(*node->server),
                write_memory);
      for (const size_t count : {size_t{1}, size_t{128}, size_t{1024},
                                 size_t{1025}}) {
        const std::vector<BlockKey> collisions(pool.begin(),
                                               pool.begin() + count);
        const std::vector<BlockKey> duplicates(count, pool[0]);
        const auto same_bucket = MeasureAgesCase(
            *node->server, node->addr, "same_bucket", rep, write_memory,
            collisions, false);
        EXPECT_TRUE(same_bucket.layout_matches);
        EXPECT_EQ(same_bucket.server_requests, same_bucket.expected_requests);
        EXPECT_EQ(same_bucket.calls.failures, 0u);
        const auto repeated = MeasureAgesCase(
            *node->server, node->addr, "duplicates", rep, write_memory,
            duplicates, true);
        EXPECT_TRUE(repeated.layout_matches);
        EXPECT_EQ(repeated.server_requests, repeated.expected_requests);
        EXPECT_EQ(repeated.calls.failures, 0u);
      }
    }
  }
}

bool BenchFlagReady(const std::atomic<bool>& flag,
                    std::chrono::milliseconds timeout) {
  const auto deadline = BenchClock::now() + timeout;
  while (BenchClock::now() < deadline) {
    if (flag.load(std::memory_order_acquire)) return true;
    std::this_thread::yield();
  }
  return flag.load(std::memory_order_acquire);
}

struct JoinBenchThread {
  FlushBarrier* release;
  std::thread thread;
  ~JoinBenchThread() {
    release->Open();
    if (thread.joinable()) thread.join();
  }
};

// Hold the real disk store while its PUT/REMOVE owns the publication bucket.
// With W, op 26 must wait; without W it must complete before the disk opens.
TEST(RamTierWiring, PublicationDiskBarrierWaitBenchmark) {
  if (!PublicationBenchEnabled()) {
    GTEST_SKIP() << "set DFKV_PUBLICATION_BENCH=1";
  }
  for (const bool write_memory : {true, false}) {
    for (const bool remove_operation : {false, true}) {
      FlushBarrier barrier;
      auto node = StartPublicationNode(
          std::string("barrier_") + (remove_operation ? "remove_" : "put_") +
              (write_memory ? "on" : "off"),
          write_memory,
          [&](const std::string& path, uint64_t capacity) {
            return std::make_unique<BlockingBypassStore>(path, capacity, barrier,
                                                         remove_operation);
          });
      ASSERT_TRUE(node->started);
      ASSERT_EQ(KvNodeServerWiringTestPeer::HasWriteMemory(*node->server),
                write_memory);
      // 200 ms below is in this fresh node's known REMOVE history.
      std::this_thread::sleep_for(300ms);
      const BlockKey key = ToBlockKey("publication/barrier", "held");
      const std::string small(128, 's'), large(128 * 1024, 'L');
      TcpTransport writer_transport, query_transport;
      writer_transport.set_timeouts(1000, 2000);
      query_transport.set_timeouts(1000, 2000);
      if (remove_operation) {
        ASSERT_EQ(writer_transport.Cache(node->addr, key, small.data(), small.size()),
                  Status::kOk);
      }
      std::atomic<Status> write_status{Status::kInvalid};
      std::atomic<Status> query_status{Status::kInvalid};
      std::atomic<bool> write_done{false}, query_done{false};
      std::string query_out;
      const auto disk_started = BenchClock::now();
      JoinBenchThread writer{&barrier, std::thread([&] {
        const Status status = remove_operation
                                  ? writer_transport.Remove(node->addr, key)
                                  : writer_transport.Cache(node->addr, key,
                                                           large.data(), large.size());
        write_status.store(status, std::memory_order_release);
        write_done.store(true, std::memory_order_release);
      })};
      ASSERT_TRUE(barrier.AwaitEntry());
      if (write_memory)
        KvNodeServerWiringTestPeer::ExpectPublicationWait(*node->server, key);
      const std::string payload = EncodeRemovedWithinPayload(200, &key, 1);
      JoinBenchThread query{&barrier, std::thread([&] {
        std::string out;
        const Status status = query_transport.RemovedWithinAges(
            node->addr, key.tenant_hash, 1, payload, &out);
        query_status.store(status, std::memory_order_release);
        query_out = std::move(out);
        query_done.store(true, std::memory_order_release);
      })};
      bool query_finished_early = false;
      if (write_memory) {
        ASSERT_TRUE(KvNodeServerWiringTestPeer::WaitPublicationWait(
            *node->server, key)) << "op 26 did not reach the held bucket";
        query_finished_early = query_done.load(std::memory_order_acquire);
        EXPECT_FALSE(query_finished_early)
            << (remove_operation ? "REMOVE" : "PUT")
            << " released op 26 while its disk barrier was still closed";
      } else {
        query_finished_early =
            BenchFlagReady(query_done, std::chrono::milliseconds(2000));
        EXPECT_TRUE(query_finished_early) << "W-off op 26 waited for disk I/O";
      }
      const auto blocked_us = BenchUs(BenchClock::now() - disk_started);
      barrier.Open();
      ASSERT_TRUE(BenchFlagReady(write_done, std::chrono::milliseconds(3000)));
      ASSERT_TRUE(BenchFlagReady(query_done, std::chrono::milliseconds(3000)));
      const auto disk_us = BenchUs(BenchClock::now() - disk_started);
      writer.thread.join();
      query.thread.join();
      EXPECT_EQ(write_status.load(), Status::kOk);
      EXPECT_EQ(query_status.load(), Status::kOk);
      ASSERT_EQ(query_out.size(), 4 + kRemovedWithinAgesKeyBytes);
      const uint32_t removed_age = net::GetU32(query_out.data() + 4);
      const uint32_t written_age = net::GetU32(query_out.data() + 8);
      if (write_memory) {
        EXPECT_GE(disk_us, blocked_us);
        if (remove_operation) {
          EXPECT_NE(removed_age, kRemovedAgeNone);
          EXPECT_NE(removed_age, kRemovedAgeUnknown);
          // The small seed PUT before the gated REMOVE is a real W record.
          EXPECT_NE(written_age, kRemovedAgeNone);
          EXPECT_NE(written_age, kRemovedAgeUnknown);
        } else {
          EXPECT_TRUE(removed_age == kRemovedAgeNone ||
                      removed_age == kRemovedAgeUnknown);
          EXPECT_NE(written_age, kRemovedAgeNone);
        }
      } else {
        EXPECT_TRUE(removed_age == kRemovedAgeNone ||
                    removed_age == kRemovedAgeUnknown);
        EXPECT_EQ(written_age, kRemovedAgeNone);
      }
      const int64_t failed = (write_status.load() != Status::kOk ? 1 : 0) +
                             (query_status.load() != Status::kOk ? 1 : 0);
      std::printf(
          "PUBLICATION_BARRIER W=%s op=%s blocked_us=%.3f disk_call_us=%.3f "
          "query_completed_before_release=%s failures=%lld\n",
          write_memory ? "on" : "off", remove_operation ? "REMOVE" : "PUT",
          blocked_us, disk_us, query_finished_early ? "yes" : "no",
          static_cast<long long>(failed));
    }
  }
}

}  // namespace
