// RamTier wiring into KvNodeServer: with DFKV_RAM_TIER=1, PUT is write-through
// to RAM (sync-visible, async-flushed) and GET is served from RAM; disabled by
// default the node behaves exactly as before. End-to-end over the TCP transport.
#include "cache/kv_node_server.h"
#include "client/key_map.h"
#include "transport/tcp_transport.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <atomic>
#include <thread>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

using namespace std::chrono_literals;

namespace {
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
    ASSERT_EQ(t.Cache(addr, ToBlockKey("k" + std::to_string(i)), v.data(), v.size()),
              Status::kOk) << i;
  }
  for (int i = 0; i < 20; ++i) {
    std::string v = "ram-block-" + std::to_string(i) + std::string(200, 'z');
    std::string out;
    ASSERT_EQ(t.Range(addr, ToBlockKey("k" + std::to_string(i)), 0, v.size(), &out),
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
  ASSERT_EQ(t.Cache(addr, ToBlockKey("p"), v.data(), v.size()), Status::kOk);
  std::string out;
  ASSERT_EQ(t.Range(addr, ToBlockKey("p"), 3, 4, &out), Status::kOk);  // RAM sub-range
  EXPECT_EQ(out, "3456");
  // absent key -> miss even with RAM on
  EXPECT_EQ(t.Range(addr, ToBlockKey("absent"), 0, 4, &out), Status::kNotFound);
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
  ASSERT_EQ(t.Cache(addr, ToBlockKey("live"), v.data(), v.size()), Status::kOk);
  bool e = false;
  ASSERT_EQ(t.Exist(addr, ToBlockKey("live"), &e), Status::kOk);
  EXPECT_TRUE(e) << "a RAM-resident block must report as existing";
  // Let it flush to disk so Remove targets a durable block, then remove it.
  EXPECT_TRUE(WaitFor([&] { return MetricVal(s->MetricsText(), "dfkv_ram_flushed_total") >= 1; }));
  ASSERT_EQ(t.Remove(addr, ToBlockKey("live")), Status::kOk);
  ASSERT_EQ(t.Exist(addr, ToBlockKey("live"), &e), Status::kOk);
  EXPECT_FALSE(e) << "after Remove the block is gone from both tiers";
  std::string out;
  EXPECT_EQ(t.Range(addr, ToBlockKey("live"), 0, v.size(), &out), Status::kNotFound);

  s.reset();  // stop the flusher before removing the dir (no write race)
  ::unsetenv("DFKV_RAM_TIER");
  ::unsetenv("DFKV_RAM_TIER_BYTES");
  fs::remove_all(dir);
}

TEST(RamTierWiring, DisabledByDefaultNoRamMetrics) {
  ::unsetenv("DFKV_RAM_TIER");  // ensure off
  std::string addr;
  auto dir = fs::temp_directory_path() / "dfkv_ramwire_c";
  auto s = Start(dir, &addr);
  TcpTransport t;
  std::string v(128, 'd');
  ASSERT_EQ(t.Cache(addr, ToBlockKey("x"), v.data(), v.size()), Status::kOk);
  std::string out;
  ASSERT_EQ(t.Range(addr, ToBlockKey("x"), 0, v.size(), &out), Status::kOk);  // via disk
  EXPECT_EQ(out, v);
  std::string m = s->MetricsText();
  EXPECT_EQ(MetricVal(m, "dfkv_ram_hit_total"), -1) << "no RAM metrics when disabled";
  EXPECT_EQ(MetricVal(m, "dfkv_ram_put_total"), -1);
  EXPECT_EQ(s->m_cache_put(), 1u);   // normal disk accounting intact
  EXPECT_EQ(s->m_cache_hit(), 1u);
  s.reset();
  fs::remove_all(dir);
}

// I6: the PUT admission gate (DFKV_PUT_INFLIGHT_LIMIT) fast-fails concurrent
// disk writes past the limit with kCacheFull -- a controlled miss, not a queue
// tail -- and counts them. limit=1 with 8 racing 4 MiB writers makes at least
// one rejection all but certain.
TEST(RamTierWiring, PutAdmissionGateRejectsWithCacheFull) {
  ::setenv("DFKV_PUT_INFLIGHT_LIMIT", "1", 1);
  auto dir = fs::temp_directory_path() / "dfkv_admission";
  fs::remove_all(dir);
  fs::create_directories(dir);
  auto s = std::make_unique<KvNodeServer>(dir.string(), 1ull << 30);
  ::unsetenv("DFKV_PUT_INFLIGHT_LIMIT");

  constexpr int T = 8, N = 20;
  std::atomic<int> ok{0}, busy{0}, other{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < T; ++t) {
    ts.emplace_back([&, t] {
      // KVStore::CacheDirect demands a 4 KiB-aligned buffer (O_DIRECT, no
      // buffered fallback in the file engine).
      void* mem = nullptr;
      ASSERT_EQ(posix_memalign(&mem, 4096, 4 << 20), 0);
      char* buf = static_cast<char*>(mem);
      std::memset(buf, 'a' + t, 4 << 20);
      for (int i = 0; i < N; ++i) {
        Status st = s->CacheDirectForKey(
            BlockKey{static_cast<uint64_t>(1000 + t * N + i), 0}, buf,
            4 << 20, 4 << 20);
        if (st == Status::kOk) ok.fetch_add(1);
        else if (st == Status::kCacheFull) busy.fetch_add(1);
        else other.fetch_add(1);
      }
      free(mem);
    });
  }
  for (auto& th : ts) th.join();
  EXPECT_GT(ok.load(), 0);
  EXPECT_GT(busy.load(), 0) << "8 writers vs limit=1: gate never fired?";
  EXPECT_EQ(other.load(), 0);
  const std::string m = s->MetricsText();
  EXPECT_EQ(MetricVal(m, "dfkv_put_busy_total"), busy.load());
  s.reset();
  fs::remove_all(dir);
}

// The uring prep path partitions keys up front: RAM-resident -> decline
// (kInvalid, sync arena serve), absent -> async disk read. The absent branch
// never reaches GetPrep, so it must count the RAM miss itself — without that,
// the default read path reports dfkv_ram_miss_total == 0 forever while hits
// accumulate, and hit/(hit+miss) reads as a fake 100% (seen in production:
// 985k hits / 0 misses across a 15TB read campaign).
TEST(RamTierWiring, UringPrepPathCountsRamMisses) {
  ::setenv("DFKV_RAM_TIER", "1", 1);
  ::setenv("DFKV_RAM_TIER_BYTES", "8388608", 1);
  std::string addr;
  auto dir = fs::temp_directory_path() / "dfkv_ramwire_prep";
  auto s = Start(dir, &addr);
  TcpTransport t;
  std::string v(4096, 'q');
  ASSERT_EQ(t.Cache(addr, ToBlockKey("resident"), v.data(), v.size()), Status::kOk);

  const BlockKey rk = ToBlockKey("resident");
  const BlockKey ak = ToBlockKey("absent-key");
  KVStore::RangePrep prep;

  // RAM-resident: prep declines so the serve loop falls back to the arena.
  // Consulted-and-present is NOT a miss.
  const long miss0 = MetricVal(s->MetricsText(), "dfkv_ram_miss_total");
  EXPECT_EQ(s->RangeDirectPrepForKey(rk, 0, v.size(), 1 << 20, &prep),
            Status::kInvalid);
  EXPECT_EQ(MetricVal(s->MetricsText(), "dfkv_ram_miss_total"), miss0);

  // Absent everywhere: RAM consulted and absent -> exactly one RAM miss,
  // regardless of the disk outcome (kNotFound here).
  EXPECT_EQ(s->RangeDirectPrepForKey(ak, 0, 16, 1 << 20, &prep),
            Status::kNotFound);
  EXPECT_EQ(MetricVal(s->MetricsText(), "dfkv_ram_miss_total"), miss0 + 1);

  s.reset();
  ::unsetenv("DFKV_RAM_TIER");
  ::unsetenv("DFKV_RAM_TIER_BYTES");
  fs::remove_all(dir);
}
