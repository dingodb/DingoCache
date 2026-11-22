// RamTier wiring into KvNodeServer: with DFKV_RAM_TIER=1, PUT is write-through
// to RAM (sync-visible, async-flushed) and GET is served from RAM; disabled by
// default the node behaves exactly as before. End-to-end over the TCP transport.
#include "cache/kv_node_server.h"
#include "cache/disk_slab_store.h"
#include "cache/kv_store.h"
#include "client/key_map.h"
#include "transport/tcp_transport.h"
#include "transport/transport.h"

#include <gtest/gtest.h>

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
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
                     bool ram_ack = false, uint32_t watermark_pct = 80) {
    RamTier::Options options;
    options.bytes = 16 * 4096;
    options.large_reserve_bytes = 0;
    options.reclaim_interval_ms = 0;
    options.ack_high_watermark_pct = watermark_pct;
    options.is_persisted = [&server](const BlockKey& key) {
      return server.group_.IsCached(key);
    };
    server.ram_ = std::make_unique<RamTier>(options, std::move(flush));
    server.ram_write_back_ = true;
    server.ram_ack_enabled_ = ram_ack;
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
  static uint64_t RamDirty(KvNodeServer& server) {
    return server.ram_->DirtyObjects();
  }
  static uint64_t UncleanResets(KvNodeServer& server) {
    return server.group_.SlabStats().unclean_resets;
  }
  static uint64_t WarmRestarts(KvNodeServer& server) {
    return server.group_.SlabStats().warm_restarts;
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
                      FlushBarrier& barrier)
      : KVStore(Options{path, capacity}), barrier_(barrier) {}
  Status Cache(const BlockKey& key, const void* data, size_t len) override {
    if (len > 64 * 1024) barrier_.Block();
    return KVStore::Cache(key, data, len);
  }
  Status CacheDirect(const BlockKey& key, char* data, size_t len,
                     size_t cap) override {
    if (len > 64 * 1024) barrier_.Block();
    return KVStore::CacheDirect(key, data, len, cap);
  }

 private:
  FlushBarrier& barrier_;
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
            return KvNodeServerWiringTestPeer::Persist(*node, key, data, len) ==
                   Status::kOk;
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
          *server, [](const BlockKey&, char*, size_t, size_t) { return false; });
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
        *server, [](const BlockKey&, char*, size_t, size_t) { return false; });
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
        return KvNodeServerWiringTestPeer::Persist(*server, key, data, len) ==
               Status::kOk;
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
        return KvNodeServerWiringTestPeer::Persist(*node, key, data, len) ==
               Status::kOk;
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
          return KvNodeServerWiringTestPeer::Persist(*node, key, data, len) ==
                 Status::kOk;
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
          return KvNodeServerWiringTestPeer::Persist(*node, key, data, len) ==
                 Status::kOk;
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
          return KvNodeServerWiringTestPeer::Persist(*node, key, data, len) ==
                 Status::kOk;
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

  explicit FlushFailNode(const char* name, size_t disks = 1) {
    root = fs::temp_directory_path() / (std::string("dfkv_flushfail_") + name);
    fs::remove_all(root);
    fs::create_directories(root / "tokens");
    std::vector<std::string> dirs;
    for (size_t i = 0; i < disks; ++i)
      dirs.push_back((root / ("disk" + std::to_string(i))).string());
    stores.assign(disks, nullptr);
    DiskCacheGroup::Options options{dirs, disks * (64ull << 20), "slab"};
    options.engine_factory =
        [this, dirs](const std::string& dir,
                     uint64_t) -> std::unique_ptr<StoreEngine> {
      DiskSlabStore::Options o;
      o.dir = dir;
      o.trust_token_dir = (root / "tokens").string();
      o.capacity_bytes = 64 << 20;
      o.extent_bytes = 1 << 20;
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

// With several disks, only the keys of the failed one are refused.
TEST(RamTierFlushFailure, OtherDisksKeepTheirRamAcknowledgement) {
  FlushFailNode node("otherdisks", 2);
  ASSERT_TRUE(node.server->ram_enabled());
  // Fail the store of whichever disk key 1 is written to.
  for (DiskSlabStore* store : node.stores)
    DiskSlabStoreTestPeer::SetRecordWriteHook(store, [] {
      return false;
    });
  ASSERT_EQ(node.Put(1), Status::kOk);
  ASSERT_TRUE(WaitFor([&] { return !node.server->DisksHealthy(); }));
  for (DiskSlabStore* store : node.stores)
    DiskSlabStoreTestPeer::SetRecordWriteHook(store, {});
  ASSERT_TRUE(node.Settled());
  ASSERT_NE(node.stores[0]->Healthy(), node.stores[1]->Healthy())
      << "exactly one store failed";

  int refused = 0, acknowledged = 0;
  for (uint64_t id = 100; id < 164; ++id) {
    const Status st = node.Put(id);
    if (st == Status::kIOError) ++refused;
    if (st == Status::kOk) ++acknowledged;
  }
  EXPECT_EQ(refused + acknowledged, 64);
  EXPECT_GT(refused, 0);
  EXPECT_GT(acknowledged, 0);
  ASSERT_TRUE(node.Settled());
  EXPECT_EQ(node.Metric("dfkv_ram_flushed_total"), acknowledged)
      << "every acknowledged value reached its (healthy) disk";
  EXPECT_EQ(node.Metric("dfkv_ram_post_ack_flush_failures_total"), 1);
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
