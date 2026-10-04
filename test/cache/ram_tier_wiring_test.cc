// RamTier wiring into KvNodeServer: with DFKV_RAM_TIER=1, PUT is write-through
// to RAM (sync-visible, async-flushed) and GET is served from RAM; disabled by
// default the node behaves exactly as before. End-to-end over the TCP transport.
#include "cache/kv_node_server.h"
#include "client/key_map.h"
#include "transport/tcp_transport.h"
#include "transport/transport.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <atomic>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

using namespace std::chrono_literals;

namespace dfkv {
class KvNodeServerWiringTestPeer {
 public:
  static std::unique_ptr<KvNodeServer> Create(DiskCacheGroup::Options options) {
    return std::unique_ptr<KvNodeServer>(new KvNodeServer(std::move(options)));
  }
  static void SetLimit(KvNodeServer& server, size_t limit) {
    server.put_busy_limit_ = limit;
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
