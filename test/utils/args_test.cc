#include "utils/args.h"

#include <gtest/gtest.h>
#include <vector>

using namespace dfkv;  // NOLINT

namespace {
// Build an argv from string literals for Args (which takes char**).
struct Argv {
  std::vector<std::string> store;
  std::vector<char*> ptrs;
  explicit Argv(std::vector<std::string> a) : store(std::move(a)) {
    for (auto& s : store) ptrs.push_back(s.data());
  }
  int argc() const { return static_cast<int>(ptrs.size()); }
  char** argv() { return ptrs.data(); }
};
}  // namespace

TEST(Args, ParsesKnownValuedAndBooleanFlags) {
  Argv a({"prog", "--port", "8080", "--verbose", "--dir", "/x"});
  Args args(a.argc(), a.argv(), {"--port", "--dir"}, {"--verbose"});
  ASSERT_TRUE(args.ok()) << args.error();
  EXPECT_EQ(args.GetInt("--port", 0), 8080);
  EXPECT_EQ(args.Get("--dir", ""), "/x");
  EXPECT_TRUE(args.Has("--verbose"));
  EXPECT_FALSE(args.Has("--nope"));
}

TEST(Args, UnknownFlagFails) {
  Argv a({"prog", "--capp", "5"});
  Args args(a.argc(), a.argv(), {"--cap"});
  EXPECT_FALSE(args.ok());
  EXPECT_NE(args.error().find("--capp"), std::string::npos);
}

TEST(Args, MissingValueFails) {
  Argv a({"prog", "--port"});
  Args args(a.argc(), a.argv(), {"--port"});
  EXPECT_FALSE(args.ok());
  EXPECT_NE(args.error().find("missing value"), std::string::npos);
}

TEST(Args, ValuelessFlagDoesNotEatNextFlag) {
  // The old i+=2 loop would swallow "--port" as --verbose's value. Here
  // --verbose stands alone and --port keeps its own value.
  Argv a({"prog", "--verbose", "--port", "9"});
  Args args(a.argc(), a.argv(), {"--port"}, {"--verbose"});
  ASSERT_TRUE(args.ok()) << args.error();
  EXPECT_TRUE(args.Has("--verbose"));
  EXPECT_EQ(args.GetInt("--port", 0), 9);
}

TEST(Args, TrailingGarbageInNumberFails) {
  Argv a({"prog", "--cap", "5TiB"});
  Args args(a.argc(), a.argv(), {"--cap"});
  uint64_t cap = args.GetU64("--cap", 1);
  EXPECT_FALSE(args.ok());
  EXPECT_EQ(cap, 1u);  // returns the default on error
  EXPECT_NE(args.error().find("--cap"), std::string::npos);
}

TEST(Args, ValidNumbersParse) {
  Argv a({"prog", "--cap", "5368709120"});  // 5 GiB in bytes
  Args args(a.argc(), a.argv(), {"--cap"});
  EXPECT_EQ(args.GetU64("--cap", 0), 5368709120ull);
  EXPECT_TRUE(args.ok());
}

TEST(Args, NonFlagTokenFails) {
  Argv a({"prog", "stray"});
  Args args(a.argc(), a.argv(), {"--port"});
  EXPECT_FALSE(args.ok());
}

TEST(Args, VersionHelpAlwaysAccepted) {
  Argv a({"prog", "--version"});
  Args args(a.argc(), a.argv(), {"--port"});
  EXPECT_TRUE(args.ok());
  EXPECT_TRUE(args.Has("--version"));
}

TEST(HostPort, ValidatesAdvertise) {
  EXPECT_TRUE(IsValidHostPort("192.168.1.5:28100"));
  EXPECT_TRUE(IsValidHostPort("host:1"));
  EXPECT_TRUE(IsValidHostPort("h:65535"));
  EXPECT_FALSE(IsValidHostPort("no-colon"));
  EXPECT_FALSE(IsValidHostPort(":28100"));       // empty host
  EXPECT_FALSE(IsValidHostPort("host:"));        // empty port
  EXPECT_FALSE(IsValidHostPort("host:0"));       // port 0
  EXPECT_FALSE(IsValidHostPort("host:70000"));   // out of range
  EXPECT_FALSE(IsValidHostPort("host:80x"));     // trailing garbage
}

// Regression: the flag→env facades added in dfkv_server_main /
// dfkv_mds_main must all be in the Args valued set, or a typo'd
// addition silently becomes "unknown flag". This mirrors the exact valued
// set the daemons construct; if a new flag is added to main but not here,
// this test breaks as a reminder to keep them in sync.
TEST(Args, ServerFlagFacadesAllAccepted) {
  const std::set<std::string> server_valued = {
      "--dir", "--port", "--cap", "--rdma-port", "--rdma-dev",
      "--mds", "--group", "--id", "--advertise", "--weight",
      "--metrics-port", "--metrics-bind", "--store-engine",
      "--slab-write", "--ram-tier", "--ram-tier-bytes",
      "--slab-granularity", "--put-inflight-limit",
      "--rdma-depth", "--rdma-numa", "--rdma-idle-ms",
      "--rdma-op-timeout-ms", "--server-uring",
      "--server-uring-depth", "--ram-flush-threads",
      "--ram-tier-numa", "--ram-tier-shards", "--slab-table-sync-ms",
      "--slab-reclaim-ms", "--ram-reclaim-ms", "--log",
      "--mds-registration-timeout-ms"};
  // Every facade flag parses cleanly (non-empty value, ok=true).
  for (const auto& f : server_valued) {
    Argv a({"prog", f, "1"});
    Args args(a.argc(), a.argv(), server_valued);
    EXPECT_TRUE(args.ok()) << "flag " << f << ": " << args.error();
    EXPECT_EQ(args.Get(f, ""), "1") << "flag " << f << " lost its value";
  }
}

TEST(Args, MdsFlagFacadesAllAccepted) {
  const std::set<std::string> mds_valued = {
      "--etcd", "--listen", "--metrics-port", "--metrics-bind",
      "--etcd-probe-ms"};
  for (const auto& f : mds_valued) {
    Argv a({"prog", f, "1"});
    Args args(a.argc(), a.argv(), mds_valued);
    EXPECT_TRUE(args.ok()) << "flag " << f << ": " << args.error();
    EXPECT_EQ(args.Get(f, ""), "1") << "flag " << f << " lost its value";
  }
}
