// The client library's and the TCP transport's settings, at the places that
// read them: what an unclean value resolves to (the same as it always did)
// and that it is reported once, at WARN, through the library's logger. The
// library runs inside a host process: no value may make it throw or exit.
#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "cache/kv_node_server.h"
#include "client/dfkv_c_api.h"
#include "client/kv_client.h"
#include "client/node_dedup.h"
#include "common/config_dump.h"
#include "mds/mds_member_poller.h"
#include "transport/tcp_transport.h"
#include "transport/transport_factory.h"
#include "utils/env_parse.h"

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {

class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    ::setenv(name, value, 1);
  }
  ~ScopedEnv() { ::unsetenv(name_.c_str()); }

 private:
  std::string name_;
};

std::string LogOf(const std::function<void()>& run) {
  envparse::ForgetWarningsForTest();
  testing::internal::CaptureStderr();
  run();
  return testing::internal::GetCapturedStderr();
}
size_t Count(const std::string& log, const std::string& what) {
  size_t n = 0;
  for (size_t at = log.find(what); at != std::string::npos;
       at = log.find(what, at + 1))
    ++n;
  return n;
}
// One WARN line names the variable with its value and the value in use.
void ExpectReported(const std::string& log, const char* name,
                    const char* value, const std::string& used) {
  const std::string quoted = std::string(name) + "='" + value + "'";
  EXPECT_EQ(Count(log, quoted), 1u) << quoted << " in: " << log;
  const size_t at = log.find(quoted);
  if (at == std::string::npos) return;
  const size_t line_start = log.rfind('\n', at);
  const size_t line_end = log.find('\n', at);
  const std::string line = log.substr(
      line_start == std::string::npos ? 0 : line_start + 1,
      line_end - (line_start == std::string::npos ? 0 : line_start + 1));
  EXPECT_NE(line.find("[WARN]"), std::string::npos) << line;
  EXPECT_NE(line.find(used), std::string::npos) << line;
}

// The value a setting resolved to, as the component recorded it for the
// config dump ("  NAME = value  (source)").
std::string Recorded(const std::string& dump, const std::string& name) {
  size_t at = dump.find("  " + name + " ");
  if (at == std::string::npos) return "(not recorded)";
  at = dump.find("= ", at);
  const size_t end = dump.find("  (", at);
  return dump.substr(at + 2, end - at - 2);
}

TEST(ClientEnv, TcpPoolSettingsKeepTheirReading) {
  struct Case {
    const char* value;
    const char* max_conns;
    bool reported;
  };
  const Case cases[] = {{"", "8", false},   {"16", "16", false},
                        {"64", "64", false}, {"0", "8", true},
                        {"65", "8", true},   {"16x", "8", true},
                        {"abc", "8", true},  {"-1", "8", true},
                        {" 16", "16", true}, {"+16", "16", true}};
  for (const Case& c : cases) {
    ScopedEnv env("DFKV_TCP_POOL_MAX_CONNS", c.value);
    config_dump::ResetForTest();
    const std::string log = LogOf([] {
      TcpTransport transport;
      config_dump::Emit("test");
    });
    EXPECT_EQ(Recorded(log, "DFKV_TCP_POOL_MAX_CONNS"), c.max_conns)
        << "'" << c.value << "'\n" << log;
    if (c.reported)
      ExpectReported(log, "DFKV_TCP_POOL_MAX_CONNS", c.value,
                     std::string("using ") + c.max_conns + ", as before");
    else
      EXPECT_EQ(Count(log, "DFKV_TCP_POOL_MAX_CONNS='"), 0u) << log;
  }
}

// DFKV_RDMA: only "0", "false" and "no" in lower case are off. "off" asks
// for RDMA, as it always did, and now says so.
TEST(ClientEnv, TransportSwitchKeepsItsReading) {
  struct Case {
    const char* value;
    bool rdma_requested;
    bool reported;
  };
  const Case cases[] = {{"", false, false},    {"0", false, false},
                        {"false", false, false}, {"no", false, false},
                        {"off", true, true},   {"FALSE", true, true},
                        {"No", true, true},    {"maybe", true, true}};
  for (const Case& c : cases) {
    ScopedEnv env("DFKV_RDMA", c.value);
    std::string reason;
    const std::string log = LogOf([&] {
      std::unique_ptr<Transport> transport;
      EXPECT_NO_THROW(transport = MakeClientTransport(&reason));
    });
    EXPECT_EQ(reason != "tcp(rdma-not-requested)", c.rdma_requested)
        << "'" << c.value << "': " << reason;
    if (c.reported)
      ExpectReported(log, "DFKV_RDMA", c.value, "read as on, as before");
    else
      EXPECT_EQ(Count(log, "DFKV_RDMA="), 0u) << log;
  }
}

// DFKV_CLIENT_NODE_DEDUP: only "1" is on.
TEST(ClientEnv, NodeDedupSwitchIsOnOnlyForOne) {
  for (const char* value : {"true", "on", "yes", "2", "01"}) {
    ScopedEnv env("DFKV_CLIENT_NODE_DEDUP", value);
    std::unique_ptr<NodeDedup> dedup;
    const std::string log = LogOf([&] { dedup = NodeDedup::FromEnv(0x1234); });
    EXPECT_EQ(dedup, nullptr) << value;
    ExpectReported(log, "DFKV_CLIENT_NODE_DEDUP", value,
                   "read as off, as before");
  }
  for (const char* value : {"", "0", "off", "false"}) {
    ScopedEnv env("DFKV_CLIENT_NODE_DEDUP", value);
    std::unique_ptr<NodeDedup> dedup;
    const std::string log = LogOf([&] { dedup = NodeDedup::FromEnv(0x1234); });
    EXPECT_EQ(dedup, nullptr) << value;
    EXPECT_EQ(Count(log, "DFKV_CLIENT_NODE_DEDUP="), 0u) << log;
  }
}

// DFKV_MDS_SHRINK_GUARD_PCT, read when a member poller is built.
TEST(ClientEnv, ShrinkGuardPercentIsReported) {
  struct Case {
    const char* value;
    const char* used;
    bool reported;
  };
  const Case cases[] = {{"30", "", false},
                        {"0", "", false},
                        {"100", "using 50, as before", true},
                        {"abc", "using 0, as before", true},  // atol: 0
                        {"30%", "using 30, as before", true},
                        {"-1", "using 50, as before", true},
                        // atol(""): 0, the guard off; empty is not unset here.
                        {"", "using 0, as before", true}};
  for (const Case& c : cases) {
    ScopedEnv env("DFKV_MDS_SHRINK_GUARD_PCT", c.value);
    const std::string log = LogOf([] {
      MdsMemberPoller poller({"127.0.0.1:1"}, "g",
                             [](const std::vector<MemberInfo>&) {}, 1000);
    });
    if (c.reported)
      ExpectReported(log, "DFKV_MDS_SHRINK_GUARD_PCT", c.value, c.used);
    else
      EXPECT_EQ(Count(log, "DFKV_MDS_SHRINK_GUARD_PCT"), 0u) << log;
  }
}

dfkv_client_t OpenClient(const std::string& members, const char* mds) {
  dfkv_client_options_v2 options{};
  options.struct_size = sizeof(options);
  options.abi_version = DFKV_CLIENT_ABI_VERSION_V2;
  options.members = members.c_str();
  options.mds_endpoints = mds;
  options.key_namespace = "client-env/test";
  options.key_namespace_len = std::strlen("client-env/test");
  return dfkv_open_v2(&options);
}

// A setting whose 0 is documented to mean something (auto, off, no wait,
// the guard off, ...) is clean at 0: no WARN.
TEST(ClientEnv, ZeroThatMeansSomethingIsNotReported) {
  const fs::path dir = fs::temp_directory_path() /
                       ("dfkv_client_env_zero_" + std::to_string(::getpid()));
  fs::remove_all(dir);
  fs::create_directories(dir);
  KvNodeServer server(dir.string(), 1ull << 30);
  ASSERT_EQ(server.Start(0), Status::kOk);
  const std::string members = "n=127.0.0.1:" + std::to_string(server.port());
  const char* const names[] = {
      "DFKV_BATCH_CONCURRENCY",      // 0 = auto
      "DFKV_PROBE_INTERVAL_MS",      // 0 = no probe
      "DFKV_GET_MISS_RETRIES",       // 0 = no retry
      "DFKV_TCP_POOL_MIN_CONNS",     // 0 = no warm connection
      "DFKV_OPEN_RING_WAIT_MS",      // 0 = do not wait
      "DFKV_MDS_SHRINK_GUARD_PCT",   // 0 = guard off
      "DFKV_RETAIN_PRECHECK_MIN_HEADROOM_PCT",  // 0 % headroom
      "DFKV_RETAIN_CACHE_ENTRIES",
  };
  for (const char* name : names) ::setenv(name, "0", 1);
  const std::string log = LogOf([&] {
    {
      // Without a batch concurrency argument, so the variable is read.
      KVClient client({{"n", "127.0.0.1:" + std::to_string(server.port())}},
                      "client-env/test");
    }
    // With an MDS, so the ring wait and the shrink guard are read too.
    dfkv_client_t client = OpenClient(members, "127.0.0.1:1");
    ASSERT_NE(client, nullptr);
    int ok = 0;
    (void)dfkv_retain_precheck(client, nullptr, nullptr, 0, 0, &ok);
    dfkv_close(client);
  });
  for (const char* name : names) ::unsetenv(name);
  EXPECT_EQ(Count(log, "invalid DFKV_"), 0u) << log;
  server.Stop();
  fs::remove_all(dir);
}

// DFKV_RETAIN_PRECHECK_MIN_HEADROOM_PCT='': strtol("") is 0, not the
// default 5, so empty is reported. (Reported once per process: ctest runs
// each test in its own.)
TEST(ClientEnv, EmptyRetainHeadroomIsReported) {
  ScopedEnv env("DFKV_RETAIN_PRECHECK_MIN_HEADROOM_PCT", "");
  const std::string log = LogOf([] {
    dfkv_client_t client = OpenClient("n=127.0.0.1:1", nullptr);
    ASSERT_NE(client, nullptr);
    int ok = 0;
    (void)dfkv_retain_precheck(client, nullptr, nullptr, 0, 0, &ok);
    dfkv_close(client);
  });
  ExpectReported(log, "DFKV_RETAIN_PRECHECK_MIN_HEADROOM_PCT", "",
                 "set but empty; using 0, as before");
}

// 0 does not turn the peer cooldown or the bad-grace off (no cooldown would
// let every request retry a dead node): it is reported and the default kept.
TEST(ClientEnv, ZeroCooldownIsReportedAndIgnored) {
  ScopedEnv cooldown("DFKV_PEER_COOLDOWN_MS", "0");
  ScopedEnv grace("DFKV_PEER_BAD_GRACE_MS", "0");
  const std::string log = LogOf([] {
    dfkv_client_t client = OpenClient("n=127.0.0.1:1", nullptr);
    ASSERT_NE(client, nullptr);
    dfkv_close(client);
  });
  ExpectReported(log, "DFKV_PEER_COOLDOWN_MS", "0", "using 2000");
  ExpectReported(log, "DFKV_PEER_BAD_GRACE_MS", "0", "using 1000");
}

// The ring wait is passed on as an int: the report names that int, and a
// number that does not fit one is reported.
TEST(ClientEnv, RingWaitReportNamesTheIntWaitedFor) {
  ScopedEnv env("DFKV_OPEN_RING_WAIT_MS", "4294967296");  // 2^32: int 0
  const std::string log = LogOf([] {
    dfkv_client_t client = OpenClient("n=127.0.0.1:1", "127.0.0.1:1");
    ASSERT_NE(client, nullptr);
    dfkv_close(client);
  });
  ExpectReported(log, "DFKV_OPEN_RING_WAIT_MS", "4294967296",
                 "from 0 to 2147483647; using 0, as before");
}

// Every converted variable that a TCP client reads, set to garbage at once:
// the client opens, writes and reads as with clean values, nothing is thrown,
// and each name is reported exactly once however many clients are opened and
// however many operations run.
TEST(ClientEnv, AClientLoadedWithGarbageKeepsWorkingAndReportsEachNameOnce) {
  const fs::path dir = fs::temp_directory_path() /
                       ("dfkv_client_env_" + std::to_string(::getpid()));
  fs::remove_all(dir);
  fs::create_directories(dir);
  KvNodeServer server(dir.string(), 1ull << 30);
  ASSERT_EQ(server.Start(0), Status::kOk);
  const std::string members = "n=127.0.0.1:" + std::to_string(server.port());

  // name, garbage, whether a client that opens, puts, gets and batch-gets
  // over TCP reads it at all (the others are set to garbage too, and are
  // only read on paths this test does not take: MDS discovery, retention,
  // an explicit batch concurrency, the pipelined batch paths). DFKV_RDMA is left out: any garbage there
  // asks for RDMA (covered above), and this test is about a client that works.
  struct Garbage {
    const char* name;
    const char* value;
    bool read;
  };
  const std::vector<Garbage> garbage = {
      {"DFKV_READ_SHARD_KEYS", "sixteen", false},
      {"DFKV_READ_MAX_CONNS", "-8", false},
      {"DFKV_PEER_COOLDOWN_MS", "2s", true},
      {"DFKV_PEER_COOLDOWN_MAX_MS", "30000ms", true},
      {"DFKV_PEER_BAD_GRACE_MS", "1e3", true},
      {"DFKV_GET_MISS_RETRIES", "once", true},
      {"DFKV_PROBE_INTERVAL_MS", "never", true},
      {"DFKV_CLIENT_PERNODE_STATS", "0ff", false},
      {"DFKV_CLIENT_NODE_DEDUP", "true", true},
      {"DFKV_RETAIN_PRECHECK_MIN_HEADROOM_PCT", "5%", true},
      {"DFKV_TCP_POOL_MAX_CONNS", "lots", true},
      {"DFKV_TCP_POOL_MIN_CONNS", "-1", true},
      {"DFKV_TCP_POOL_IDLE_MS", "30s", true},
      {"DFKV_TCP_POOL_BACKOFF_BASE_MS", "0", true},
      {"DFKV_TCP_POOL_BACKOFF_MAX_MS", "5s", true},
      {"DFKV_TCP_POOL_ACQUIRE_TIMEOUT_MS", "10s", true},
      {"DFKV_FANOUT_THREADS", "many", true},
      {"DFKV_BATCH_CONCURRENCY", "4 threads", false},
      {"DFKV_RETAIN_CACHE_ENTRIES", "1M", false},
      {"DFKV_OPEN_RING_WAIT_MS", "30s", false},
  };
  for (const Garbage& g : garbage) ::setenv(g.name, g.value, 1);

  std::string value(4096, 'g');
  const std::string log = LogOf([&] {
    for (int round = 0; round < 3; ++round) {
      dfkv_client_options_v2 options{};
      options.struct_size = sizeof(options);
      options.abi_version = DFKV_CLIENT_ABI_VERSION_V2;
      options.members = members.c_str();
      options.key_namespace = "client-env/test";
      options.key_namespace_len = std::strlen("client-env/test");
      dfkv_client_t client = nullptr;
      ASSERT_NO_THROW(client = dfkv_open_v2(&options));
      ASSERT_NE(client, nullptr);
      std::vector<std::string> names;
      for (int i = 0; i < 20; ++i) {
        const std::string key = "k" + std::to_string(round) + "-" +
                                std::to_string(i);
        names.push_back(key);
        ASSERT_EQ(dfkv_put(client, key.data(), key.size(), value.data(),
                           value.size()),
                  0)
            << key;
        std::string out(value.size(), 0);
        ASSERT_EQ(dfkv_get(client, key.data(), key.size(), &out[0],
                           out.size()),
                  1)
            << key;
        ASSERT_EQ(out, value);
        const void* keys[] = {key.data()};
        const uint64_t lens[] = {key.size()};
        int ok = 0;
        EXPECT_EQ(dfkv_retain_precheck(client, keys, lens, 1, 0, &ok), 0);
      }
      // A batch read of all of them.
      std::vector<const void*> keys;
      std::vector<uint64_t> lens, sizes(names.size(), value.size());
      std::vector<std::string> outs(names.size(), std::string(value.size(), 0));
      std::vector<void*> ptrs;
      std::vector<int> hit(names.size(), 0);
      for (size_t i = 0; i < names.size(); ++i) {
        keys.push_back(names[i].data());
        lens.push_back(names[i].size());
        ptrs.push_back(&outs[i][0]);
      }
      ASSERT_EQ(dfkv_batch_get(client, keys.data(), lens.data(), ptrs.data(),
                               sizes.data(), static_cast<int>(names.size()),
                               hit.data()),
                0);
      for (size_t i = 0; i < names.size(); ++i) {
        EXPECT_EQ(hit[i], 1) << names[i];
        EXPECT_EQ(outs[i], value) << names[i];
      }
      dfkv_close(client);
    }
  });
  for (const Garbage& g : garbage) ::unsetenv(g.name);

  size_t read = 0;
  for (const Garbage& g : garbage) {
    const std::string quoted = std::string(g.name) + "='" + g.value + "'";
    EXPECT_EQ(Count(log, quoted), g.read ? 1u : 0u) << quoted << "\n" << log;
    read += g.read ? 1 : 0;
  }
  // All of it at WARN, through the library's logger (stderr): every line
  // that names a variable.
  EXPECT_EQ(Count(log, "invalid DFKV_"), read) << log;
  EXPECT_EQ(Count(log, "[WARN] dfkv: invalid DFKV_"), read) << log;
  server.Stop();
  fs::remove_all(dir);
}

TEST(ClientEnv, ReadbackRewriteConfigReportsItsResolvedSwitch) {
  struct Case {
    const char* readback_ms;
    const char* fanout;
    const char* rewrite;
    const char* expected;
  };
  const Case cases[] = {{"", "", "", "0"},
                        {"60000", "", "", "0"},
                        {"60000", "", "1", "1"},
                        {"", "", "1", "0"},
                        {"60000", "2", "1", "0"}};
  for (const Case& c : cases) {
    ScopedEnv readback("DFKV_RING_READBACK_MS", c.readback_ms);
    ScopedEnv fanout("DFKV_RING_REMOVE_FANOUT", c.fanout);
    ScopedEnv rewrite("DFKV_RING_READBACK_REWRITE", c.rewrite);
    config_dump::ResetForTest();
    const std::string log = LogOf([] {
      dfkv_client_t client = OpenClient("n=127.0.0.1:1", nullptr);
      ASSERT_NE(client, nullptr);
      config_dump::Emit("client");
      dfkv_close(client);
    });
    EXPECT_EQ(Recorded(log, "DFKV_RING_READBACK_REWRITE"), c.expected)
        << log;
  }
  config_dump::ResetForTest();
}

}  // namespace
