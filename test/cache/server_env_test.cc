// Server settings read from the environment, at the places that read them:
// what each unclean value resolves to (the same as up to 2.38) and that it is
// reported. The parser's own cases are in test/utils/env_parse_test.cc.
#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <functional>
#include <string>

#include "cache/disk_cache_group.h"
#include "cache/disk_slab_store.h"
#include "cache/kv_node_server.h"
#include "cache/rdma_server_env.h"
#include "utils/env_parse.h"
#include "utils/numa_util.h"
#include "utils/wire_limits.h"

namespace fs = std::filesystem;

namespace dfkv {
class KvNodeServerWiringTestPeer {
 public:
  static bool ReadCoalesce(const std::string& dir) {
    fs::remove_all(dir);
    fs::create_directories(dir);
    std::unique_ptr<KvNodeServer> server(new KvNodeServer(
        DiskCacheGroup::Options{{dir}, 1ull << 30, "file"}));
    return server->coalesce_enabled_;
  }
};
class DiskSlabStoreTestPeer {
 public:
  static bool UringWrite(const DiskSlabStore& store) {
    return store.uring_write_enabled_;
  }
  static uint64_t EvictHighBytes(const DiskSlabStore& store) {
    return store.evict_high_bytes_;
  }
};
}  // namespace dfkv

using namespace dfkv;  // NOLINT

namespace {

// Runs `read` with one variable set; returns what was logged meanwhile.
std::string LogOf(const char* name, const char* value,
                  const std::function<void()>& read) {
  envparse::ForgetWarningsForTest();
  if (value != nullptr)
    ::setenv(name, value, 1);
  else
    ::unsetenv(name);
  testing::internal::CaptureStderr();
  read();
  std::string log = testing::internal::GetCapturedStderr();
  ::unsetenv(name);
  return log;
}
bool Names(const std::string& log, const char* name, const char* value) {
  return log.find(std::string(name) + "='" + value + "'") != std::string::npos;
}

struct SwitchCase {
  const char* value;  // nullptr = unset
  bool on;
  bool reported;
};

void CheckSwitch(const char* name, const std::function<bool()>& read,
                 const SwitchCase* cases, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    const SwitchCase& c = cases[i];
    const char* shown = c.value ? c.value : "(unset)";
    bool on = false;
    const std::string log = LogOf(name, c.value, [&] { on = read(); });
    EXPECT_EQ(on, c.on) << name << "=" << shown;
    if (c.reported) {
      EXPECT_TRUE(Names(log, name, c.value)) << shown << ": " << log;
      EXPECT_NE(log.find(c.on ? "read as on, as before"
                              : "read as off, as before"),
                std::string::npos)
          << log;
    } else {
      EXPECT_EQ(log.find(std::string(name) + "="), std::string::npos)
          << shown << ": " << log;
    }
  }
}

// Off only for exactly "0": off, false and no have been switching these ON.
const SwitchCase kExactlyZero[] = {
    {nullptr, false, false}, {"", false, false}, {"0", false, false},
    {"1", true, false},      {"on", true, false}, {"off", true, true},
    {"false", true, true},   {"no", true, true},  {"OFF", true, true},
    {"00", true, true},      {"0abc", true, true}, {"enable", true, true}};

TEST(ServerEnv, RdmaNumaIsOffOnlyForZero) {
  CheckSwitch("DFKV_RDMA_NUMA", [] { return numa::EnabledFromEnv(); },
              kExactlyZero, sizeof(kExactlyZero) / sizeof(kExactlyZero[0]));
}

TEST(ServerEnv, ServerUringIsOffOnlyForZero) {
  SwitchCase cases[sizeof(kExactlyZero) / sizeof(kExactlyZero[0])];
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    cases[i] = kExactlyZero[i];
    // The default of this one is on.
    if (cases[i].value == nullptr || *cases[i].value == '\0') cases[i].on = true;
  }
  CheckSwitch("DFKV_SERVER_URING", [] { return ServerUringFromEnv(); }, cases,
              sizeof(cases) / sizeof(cases[0]));
}

// It is read on every call, not frozen at the first: tests and tools set it
// between servers of one process.
TEST(ServerEnv, ServerUringFollowsTheEnvironmentWithinOneProcess) {
  ::setenv("DFKV_SERVER_URING", "0", 1);
  EXPECT_FALSE(ServerUringFromEnv());
  ::setenv("DFKV_SERVER_URING", "1", 1);
  EXPECT_TRUE(ServerUringFromEnv());
  ::setenv("DFKV_SERVER_URING", "0", 1);
  EXPECT_FALSE(ServerUringFromEnv());
  ::unsetenv("DFKV_SERVER_URING");
  EXPECT_TRUE(ServerUringFromEnv());
}

// Off for anything starting with "0"; on for every other non-empty value.
const SwitchCase kStartsWithZero[] = {
    {nullptr, false, false}, {"", false, false},  {"0", false, false},
    {"1", true, false},      {"on", true, false}, {"off", true, true},
    {"false", true, true},   {"no", true, true},  {"00", false, true},
    {"0abc", false, true},   {"enable", true, true}};

TEST(ServerEnv, ReadCoalesceIsOffForALeadingZero) {
  const std::string dir =
      (fs::temp_directory_path() /
       ("dfkv_server_env_" + std::to_string(::getpid())))
          .string();
  CheckSwitch("DFKV_READ_COALESCE",
              [&] { return KvNodeServerWiringTestPeer::ReadCoalesce(dir); },
              kStartsWithZero,
              sizeof(kStartsWithZero) / sizeof(kStartsWithZero[0]));
  fs::remove_all(dir);
}

class SlabEnv : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           ("dfkv_slab_env_" + std::to_string(::getpid()));
    fs::remove_all(dir_);
  }
  void TearDown() override { fs::remove_all(dir_); }
  DiskSlabStore::Options Opts() {
    fs::remove_all(dir_);
    fs::create_directories(dir_ / "tokens");
    DiskSlabStore::Options o;
    o.dir = (dir_ / "store").string();
    o.trust_token_dir = (dir_ / "tokens").string();
    o.capacity_bytes = 100 << 20;
    o.extent_bytes = 1 << 20;
    o.slot_granularity = 4096;
    o.table_sync_ms = 0;
    o.reclaim_interval_ms = 0;
    return o;
  }
  fs::path dir_;
};

TEST_F(SlabEnv, UringWriteIsOffForALeadingZero) {
  {
    ::unsetenv("DFKV_SLAB_URING_WRITE");
    bool ok = false;
    DiskSlabStore store(Opts(), &ok);
    ASSERT_TRUE(ok) << store.StartupError();
    if (!DiskSlabStoreTestPeer::UringWrite(store))
      GTEST_SKIP() << "built without io_uring";
  }
  SwitchCase cases[sizeof(kStartsWithZero) / sizeof(kStartsWithZero[0])];
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    cases[i] = kStartsWithZero[i];
    if (cases[i].value == nullptr || *cases[i].value == '\0') cases[i].on = true;
  }
  CheckSwitch("DFKV_SLAB_URING_WRITE",
              [&] {
                bool ok = false;
                DiskSlabStore store(Opts(), &ok);
                EXPECT_TRUE(ok) << store.StartupError();
                return DiskSlabStoreTestPeer::UringWrite(store);
              },
              cases, sizeof(cases) / sizeof(cases[0]));
}

// DFKV_SLAB_EVICT_HIGH_PCT, read with an unchecked strtol: 0 switches the
// watermark eviction off, and so does a value that is not a number.
TEST_F(SlabEnv, EvictHighPercentKeepsItsReading) {
  struct Case { const char* value; uint64_t pct; bool reported; };
  const Case cases[] = {{nullptr, 92, false}, {"", 92, false}, {"80", 80, false},
                        {"0", 0, false},      {"abc", 0, true}, {"80%", 80, true},
                        {"-5", 92, true},     {"101", 92, true},
                        {" 80", 80, false},   {"+80", 80, true}};
  for (const Case& c : cases) {
    const char* shown = c.value ? c.value : "(unset)";
    uint64_t high = 1;
    const std::string log = LogOf("DFKV_SLAB_EVICT_HIGH_PCT", c.value, [&] {
      bool ok = false;
      DiskSlabStore store(Opts(), &ok);
      EXPECT_TRUE(ok) << store.StartupError();
      high = DiskSlabStoreTestPeer::EvictHighBytes(store);
    });
    EXPECT_EQ(high, (100ull << 20) / 100 * c.pct) << shown;
    EXPECT_EQ(Names(log, "DFKV_SLAB_EVICT_HIGH_PCT", c.value ? c.value : ""),
              c.reported)
        << shown << ": " << log;
  }
}

// DFKV_RDMA_IDLE_MS, read with an unchecked strtol: everything it read as
// <= 0 disables the reaper (-1 here).
TEST(ServerEnv, RdmaIdleMsKeepsItsReading) {
  struct Case { const char* value; int ms; bool reported; const char* says; };
  const Case cases[] = {
      {nullptr, 600000, false, ""},
      {"", 600000, false, ""},
      {"30000", 30000, false, ""},
      {"0", -1, false, ""},
      {"abc", -1, true, "the idle reaper is off"},
      {"-1", -1, true, "the idle reaper is off"},
      {" -5", -1, true, "a negative number is read as 0"},
      {"30000ms", 30000, true, "using 30000 as before"},
      {"99999999999", 86400000, true, "clamped to the maximum 86400000"},
      {"99999999999999999999999999", 86400000, true, "clamped to the maximum"}};
  for (const Case& c : cases) {
    const char* shown = c.value ? c.value : "(unset)";
    int ms = 0;
    const std::string log =
        LogOf("DFKV_RDMA_IDLE_MS", c.value, [&] { ms = RdmaIdleMsFromEnv(); });
    EXPECT_EQ(ms, c.ms) << shown;
    EXPECT_EQ(Names(log, "DFKV_RDMA_IDLE_MS", c.value ? c.value : ""),
              c.reported)
        << shown << ": " << log;
    EXPECT_NE(log.find(c.says), std::string::npos) << shown << ": " << log;
  }
}

// DFKV_RDMA_GRANT_TIMEOUT_MS: 60 s by default, 0 = no bound (-1 here).
TEST(ServerEnv, RdmaGrantTimeoutMs) {
  struct Case { const char* value; int ms; };
  const Case cases[] = {{nullptr, 60000},
                        {"", 60000},
                        {"1500", 1500},
                        {"0", -1},
                        {"99999999999", 86400000}};
  for (const Case& c : cases) {
    int ms = 0;
    (void)LogOf("DFKV_RDMA_GRANT_TIMEOUT_MS", c.value,
                [&] { ms = RdmaGrantTimeoutMsFromEnv(); });
    EXPECT_EQ(ms, c.ms) << (c.value ? c.value : "(unset)");
  }
}

// DFKV_RDMA_STOP_DRAIN_MS: 2 s by default, 0 = do not wait (-1 here).
TEST(ServerEnv, RdmaStopDrainMs) {
  struct Case { const char* value; int ms; };
  const Case cases[] = {
      {nullptr, 2000}, {"", 2000}, {"500", 500}, {"0", -1}, {"999999", 60000}};
  for (const Case& c : cases) {
    int ms = 0;
    (void)LogOf("DFKV_RDMA_STOP_DRAIN_MS", c.value,
                [&] { ms = RdmaStopDrainMsFromEnv(); });
    EXPECT_EQ(ms, c.ms) << (c.value ? c.value : "(unset)");
  }
}

// The receive-pool sizes: the line names the number that is used.
TEST(ServerEnv, RecvBytesWarningNamesTheValueInUse) {
  constexpr size_t kDefault = 128ull << 30;
  struct Case { const char* value; size_t bytes; bool reported; };
  const Case cases[] = {
      {nullptr, kDefault, false},
      {"", kDefault, false},
      {"1073741824", 1073741824, false},
      {"64G", kDefault, true},           // was never 64: the default
      {"abc", kDefault, true},
      {"0", kDefault, true},
      {"99999999999999999999999999", kDefault, true},  // overflow: the default
      {" 1073741824", 1073741824, true},
      {"5000", 8192, false}};            // aligned up
  for (const Case& c : cases) {
    const char* shown = c.value ? c.value : "(unset)";
    size_t bytes = 0;
    const std::string log = LogOf("DFKV_RDMA_RECV_SEGMENT_SIZE", c.value, [&] {
      bytes = RecvBytesFromEnv("DFKV_RDMA_RECV_SEGMENT_SIZE", kDefault);
    });
    EXPECT_EQ(bytes, c.bytes) << shown;
    EXPECT_EQ(Names(log, "DFKV_RDMA_RECV_SEGMENT_SIZE", c.value ? c.value : ""),
              c.reported)
        << shown << ": " << log;
    if (c.reported) {
      EXPECT_NE(log.find("using " + std::to_string(c.bytes) + ", as before"),
                std::string::npos)
          << shown << ": " << log;
    }
  }
}

}  // namespace

// Server memory windows: off unless asked for; the quarantine floor can only
// be raised above 5 ms and the maximum is never below the floor.
TEST(ServerEnv, ServerMemoryWindowSettings) {
  for (const char* name :
       {"DFKV_RDMA_SERVER_MW", "DFKV_RDMA_SERVER_MW_QUARANTINE_MS",
        "DFKV_RDMA_SERVER_MW_QUARANTINE_MAX_MS",
        "DFKV_RDMA_SERVER_MW_QUARANTINE_BYTES",
        "DFKV_RDMA_SERVER_MW_ARENA_BYTES", "DFKV_RDMA_SERVER_MW_LINK_GBPS"})
    ::unsetenv(name);
  RdmaServerMwSettings out = RdmaServerMwFromEnv();
  EXPECT_FALSE(out.enabled);
  EXPECT_EQ(out.floor_ms, 5u);
  EXPECT_EQ(out.max_ms, 50u);
  EXPECT_EQ(out.quarantine_bytes, 0u);
  EXPECT_EQ(out.arena_bytes, 0u);
  std::string log = LogOf("DFKV_RDMA_SERVER_MW", "1", [&] {
    out = RdmaServerMwFromEnv();
  });
  EXPECT_TRUE(out.enabled);
  log = LogOf("DFKV_RDMA_SERVER_MW_QUARANTINE_MS", "1",
              [&] { out = RdmaServerMwFromEnv(); });
  EXPECT_EQ(out.floor_ms, 5u);
  EXPECT_TRUE(Names(log, "DFKV_RDMA_SERVER_MW_QUARANTINE_MS", "1"));
  ::setenv("DFKV_RDMA_SERVER_MW_QUARANTINE_MS", "80", 1);
  out = RdmaServerMwFromEnv();
  ::unsetenv("DFKV_RDMA_SERVER_MW_QUARANTINE_MS");
  EXPECT_EQ(out.floor_ms, 80u);
  EXPECT_EQ(out.max_ms, 80u) << "the maximum is at least the floor";
}
