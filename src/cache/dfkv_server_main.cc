/* Standalone cache-node binary for the harness / pre-prod functional bring-up.
 * Usage: dfkv_server --dir <p1[,p2,p3...]> --port <p|0> --cap <bytes>
 *   --dir accepts comma-separated NVMe SSD paths (like dingo-cache --cache_dir);
 *   --cap is the TOTAL capacity, split evenly across the disks.
 * Prints "PORT <port>" on stdout once listening, then runs until SIGTERM/SIGINT. */
#include <unistd.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <atomic>
#include <charconv>
#include <chrono>
#include <functional>
#include <algorithm>
#include <memory>
#include <vector>

#include "utils/env_parse.h"
#include "utils/args.h"
#include "cache/disk_slab_store.h"
#include "cache/kv_node_server.h"
#include "cache/node_member_report.h"
#include "utils/wire_limits.h"
#include "utils/log.h"
#include "common/config_dump.h"
#include "mds/mds_registrar.h"
#include "mds/mds_server.h"
#include "utils/metrics_http.h"
#include "common/version.h"
#ifdef DFKV_WITH_RDMA
#include "cache/store_engine.h"
#include "cache/rdma_server.h"
#include "transport/rdma_health.h"
#endif

using dfkv::KvNodeServer;
using dfkv::Status;

// Sanitizers report at normal process exit (leaks, exit codes); with one of
// them compiled in, main returns and the destructors run.
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define DFKV_EXIT_RUNS_DESTRUCTORS 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || \
    __has_feature(memory_sanitizer)
#define DFKV_EXIT_RUNS_DESTRUCTORS 1
#endif
#endif
#ifndef DFKV_EXIT_RUNS_DESTRUCTORS
#define DFKV_EXIT_RUNS_DESTRUCTORS 0
#endif

// Set by SIGTERM/SIGINT. Read by the main loop, between the stages of a
// start, and by the RAM tier while it faults its arena in (other threads,
// hence atomic; lock-free, so storing to it in a signal handler is allowed).
static std::atomic<bool> g_stop{false};
static_assert(std::atomic<bool>::is_always_lock_free,
              "the stop flag is written from a signal handler");
// Stop signals received; a second one ends a stop-time drain early.
static std::atomic<int> g_stop_signals{0};
static_assert(std::atomic<int>::is_always_lock_free,
              "the signal count is written from a signal handler");
static void OnSig(int) {
  g_stop_signals.fetch_add(1, std::memory_order_relaxed);
  g_stop.store(true, std::memory_order_relaxed);
}

// DFKV_DRAIN: how this node drains. kByStop: it drains while it stops
// (DFKV_STOP_DRAIN_MS) and keeps its record for the restart. kByCtl:
// `dfkvctl drain` asked for it; the node stays draining until undrained,
// and deregisters when it stops (it is being removed).
enum class DrainState : int { kNone = 0, kByStop = 1, kByCtl = 2 };
static std::atomic<bool> g_reload{false};
static void OnHup(int) { g_reload.store(true, std::memory_order_relaxed); }

// A stop signal arrived before the server was up: the start is not carried
// to its end only to be undone. Whatever accepts requests is stopped, the
// disks that were opened are synced and marked clean (the next start is a
// clean one), and the node is never registered with the MDS -- every call
// site is ahead of the registration. Returns the exit status: 0 when every
// configured disk opened and is clean, as for any stop that was asked for
// and carried out; 1 otherwise.
static int EndCancelledStart(KvNodeServer& srv, const char* stage,
                             std::chrono::steady_clock::time_point started,
                             const std::function<void()>& stop_frontends) {
  auto seconds = [started] {
    char text[32];
    std::snprintf(text, sizeof(text), "%.2f s",
                  std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - started).count());
    return std::string(text);
  };
  DFKV_LOG_INFO(std::string("dfkv_server start cancelled by a stop signal (") +
                stage + ", " + seconds() + " into the start); closing the disks");
  if (stop_frontends) stop_frontends();
  // False as well when a disk failed to open: a start in which that happened
  // does not end with "every disk clean" and status 0.
  const bool clean = srv.Shutdown(std::chrono::steady_clock::now());
  // The last line of a cancelled start. From here a kill costs nothing.
  if (clean) {
    DFKV_LOG_INFO("dfkv_server start cancelled: not registered with the MDS, "
                  "every opened disk closed clean, " + seconds() +
                  " after the process started; exiting");
  } else {
    DFKV_LOG_WARN("dfkv_server start cancelled: not registered with the MDS, "
                  "but a disk failed to open or is not clean, " + seconds() +
                  " after the process started; exiting");
  }
  const int exit_code = clean ? 0 : 1;
#if !DFKV_EXIT_RUNS_DESTRUCTORS
  // As at the end of an ordinary stop: nothing left but giving memory back.
  std::fflush(stdout);
  std::fflush(stderr);
  ::_exit(exit_code);
#endif
  return exit_code;
}

// A crash of the server itself: have every slab store take back its trust
// token before the process dies, so the next start resets those caches
// instead of keeping tables a misbehaving process wrote. SIGKILL (OOM, stop
// timeout) never gets here, and does not need to. A stack overflow does not
// either (no alternate stack) and is then treated like a kill.
static struct sigaction g_previous_fatal[NSIG];
static void OnFatalSignal(int sig, siginfo_t* info, void*) {
  dfkv::DiskSlabStore::RevokeAllTrustForCrash();
  // Hand the signal back to whoever had it before (the default action, or a
  // sanitizer's reporter). A fault raised by the kernel re-triggers at the
  // same instruction once this returns, with its real address; one that was
  // sent to the process has to be sent again.
  ::sigaction(sig, &g_previous_fatal[sig], nullptr);
  if (info == nullptr || info->si_code <= 0) ::raise(sig);
}

int main(int argc, char** argv) {
  const auto process_started = std::chrono::steady_clock::now();
  if (dfkv::WantsVersion(argc, argv)) { std::printf("dfkv_server %s\n", dfkv::Version()); return 0; }
  // No args or --help: print usage and exit, rather than silently starting a
  // server with default config (the old behavior — a foot-gun).
  if (argc == 1 || dfkv::WantsHelp(argc, argv)) {
    bool help = dfkv::WantsHelp(argc, argv);
    std::FILE* out = help ? stdout : stderr;
    std::fprintf(out,
      "dfkv_server %s — dfkv cache-node daemon\n"
      "Usage: dfkv_server --dir <p1[,p2,...]> [--port <p|0>] --cap <bytes> [options]\n\n"
      "  --dir <paths>        comma-separated NVMe paths; --cap is the TOTAL, split evenly\n"
      "  --cap <bytes>        total cache capacity (LRU self-limits)\n"
      "  --port <p>           TCP bootstrap/data port (0 = ephemeral)\n"
      "  --rdma-port <p>      RDMA QP-bootstrap port (RDMA build only; enables data path)\n"
      "  --rdma-dev <names>   HCA whitelist; default first ACTIVE local HCA (RDMA build only)\n"
      "  --mds <ip:port,...>  MDS endpoints to register into (with --group/--id/--advertise)\n"
      "  --group <g>          membership group name (default \"default\")\n"
      "  --id <id>            node id; --advertise <ip:port>  address peers reach (rdma-port)\n"
      "  --weight <n>         consistent-hash weight (default 1)\n"
      "  --metrics-port <p>   enable Prometheus /metrics (omit = off); --metrics-bind <addr>\n"
      "  --mds-registration-timeout-ms <n>  first MDS registration deadline (default 60000; env DFKV_MDS_REGISTRATION_TIMEOUT_MS)\n"
      "  --store-engine <e>   storage backend: slab | file (default slab)\n"
      "  --slab-write <m>     slab I/O mode: direct | buffered (default direct)\n"
      "  --ram-tier <on|off>  RAM hot tier + RDMA zero-copy GET (default off)\n"
      "  --ram-tier-bytes <n> RAM arena size in bytes (default 16 GiB; pre-faulted and locked in\n"
      "                       memory at startup, DFKV_RAM_TIER_LOCK=off leaves it unlocked;\n"
      "                       on transparent huge pages where the kernel gives them,\n"
      "                       DFKV_RAM_TIER_HUGEPAGES=off does not ask for them)\n"
      "  --ram-write-mode <m> RAM PUT policy: writeback | writearound (default writeback)\n"
      "  --slab-granularity <n>  slab slot quantum bytes (default 1 MiB; existing mismatched geometry fails startup)\n"
      "  --put-inflight-limit <n>  cap concurrent disk PUTs; excess fast-fail kCacheFull (0 = off)\n"
      "  --tcp-max-conns <n> cap cache TCP handlers (default 512, hard max 4096)\n"
      "  --tcp-io-timeout-s <n>  per-socket TCP read/write timeout seconds (default 60, hard max 3600)\n"
      "  --rdma-depth <n>     RDMA write pipeline depth (default 4; must be >= client's; env DFKV_RDMA_DEPTH)\n"
      "  --rdma-numa <0|1>    NUMA-local rail selection per connection (env DFKV_RDMA_NUMA)\n"
      "  --rdma-idle-ms <n>   idle connection reaper interval ms (env DFKV_RDMA_IDLE_MS)\n"
      "  --rdma-op-timeout-ms <n>  per-op RDMA timeout ms (env DFKV_RDMA_OP_TIMEOUT_MS)\n"
      "  --server-uring <0|1> enable io_uring async-GET path (env DFKV_SERVER_URING; needs -DDFKV_WITH_URING)\n"
      "  --server-uring-depth <n>  io_uring SQ depth (env DFKV_SERVER_URING_DEPTH)\n"
      "  --ram-flush-threads <n>   RAM tier flush thread count (env DFKV_RAM_FLUSH_THREADS)\n"
      "  --ram-tier-numa <m>  RAM arena policy: interleave | off (env DFKV_RAM_TIER_NUMA)\n"
      "  --ram-tier-shards <n>  RAM tier lock shards, 1-64 (env DFKV_RAM_TIER_SHARDS; default 8)\n"
      "  --slab-table-sync-ms <n>  slab table sync cadence ms (env DFKV_SLAB_TABLE_SYNC_MS)\n"
      "  --slab-reclaim-ms <n>  slab background free-slot reclaimer cadence ms, 0 = off (env DFKV_SLAB_RECLAIM_MS)\n"
      "  --ram-reclaim-ms <n>   RAM tier background reclaimer cadence ms, 0 = off (env DFKV_RAM_RECLAIM_MS)\n"
      "  --rdma-recv-segment-size <n>  RDMA receive-pool hard budget (default 128 GiB; env DFKV_RDMA_RECV_SEGMENT_SIZE)\n"
      "  --rdma-recv-chunk-bytes <n>  lazy receive-pool chunk bytes (default 256 MiB; env DFKV_RDMA_RECV_CHUNK_BYTES)\n"
      "  --disk-hash-weight <n>  per-disk vnode multiplier (default 10; env DFKV_DISK_HASH_WEIGHT)\n"
      "  --read-coalesce <0|1>  read-side convoy merge + RAM promotion (default off; env DFKV_READ_COALESCE)\n"
      "  --log <level>        log level: INFO|DEBUG|WARN|ERROR (env DFKV_LOG)\n"
      "  --version, -V        print version and exit\n"
      "  --help, -h           print this help and exit\n"
      "Behavior flags are facades over their DFKV_* env twins; see DEPLOY.md and\n"
      "METRICS.md for the current server matrix. A flag overrides a pre-set env.\n"
      "--rdma-dev is passed directly to the RDMA server (param wins over env).\n",
      dfkv::Version());
    return help ? 0 : 1;
  }
  dfkv::Args args(argc, argv,
                  {"--dir", "--port", "--cap", "--rdma-port", "--rdma-dev",
                   "--mds", "--group",
                   "--id", "--advertise", "--weight",
                   "--metrics-port", "--metrics-bind", "--store-engine",
                   "--slab-write", "--ram-tier", "--ram-tier-bytes", "--ram-write-mode",
                   "--slab-granularity", "--put-inflight-limit",
                   "--tcp-max-conns", "--tcp-io-timeout-s",
                   "--rdma-depth", "--rdma-numa", "--rdma-idle-ms",
                   "--rdma-op-timeout-ms", "--server-uring",
                   "--server-uring-depth", "--ram-flush-threads",
                   "--ram-tier-numa", "--ram-tier-shards", "--slab-table-sync-ms",
                   "--slab-reclaim-ms", "--ram-reclaim-ms", "--log",
                   "--mds-registration-timeout-ms", "--max-msg",
                   "--rdma-recv-segment-size", "--rdma-recv-chunk-bytes",
                   "--disk-hash-weight", "--read-coalesce"});
  std::string dir = args.Get("--dir", "/tmp/dfkv_node");
  std::string rdma_dev = args.Get("--rdma-dev", "");
  std::string mds = args.Get("--mds", "");
  std::string group = args.Get("--group", "default");
  std::string node_id = args.Get("--id", "");
  std::string advertise = args.Get("--advertise", "");
  std::string metrics_bind = args.Get("--metrics-bind", "");
  int weight = args.GetInt("--weight", 1);
  int port = args.GetPort("--port", 0);
  int rdma_port = args.GetPort("--rdma-port", -1);
  int metrics_port = args.GetPort("--metrics-port", -1);
  unsigned long long cap = args.GetU64("--cap", 1ull << 30);
  // Read here, before the one args.ok() check below, in every build: a value
  // that does not parse must stop the start, not become the default.
  const unsigned long long max_msg = args.GetU64("--max-msg", 32ull << 20);
  (void)max_msg;  // the hard payload ceiling of the RDMA server
  // The explicit flag wins over DFKV_STORE_ENGINE. With neither present,
  // DiskCacheGroup resolves the single production default, "slab".
  std::string store_engine = args.Get("--store-engine", "");
  if (!store_engine.empty())
    ::setenv("DFKV_STORE_ENGINE", store_engine.c_str(), 1);
  // Slab write mode: "direct" (default; page-cache-free) | "buffered" opt-out.
  std::string slab_write = args.Get("--slab-write", "");
  if (!slab_write.empty()) ::setenv("DFKV_SLAB_WRITE", slab_write.c_str(), 1);
  // RAM hot tier facade flags. Same layering trick as --store-engine: the tier
  // is constructed deep inside KvNodeServer, so the flag rides the env twin --
  // and setenv(overwrite=1) is what makes a flag beat a pre-set env value.
  std::string ram_tier = args.Get("--ram-tier", "");
  if (!ram_tier.empty()) ::setenv("DFKV_RAM_TIER", ram_tier.c_str(), 1);
  std::string ram_tier_bytes = args.Get("--ram-tier-bytes", "");
  if (!ram_tier_bytes.empty())
    ::setenv("DFKV_RAM_TIER_BYTES", ram_tier_bytes.c_str(), 1);
  // RAM write policy: writeback (default) absorbs PUT bursts in the arena and
  // serves zero-copy GET; writearound writes straight to disk and fills the
  // arena via GET read-promotion.
  std::string ram_write_mode = args.Get("--ram-write-mode", "");
  if (!ram_write_mode.empty())
    ::setenv("DFKV_RAM_WRITE_MODE", ram_write_mode.c_str(), 1);
  // Slot quantum is persistent geometry. An existing store with a different
  // value is rejected fail-closed; operators must explicitly use an empty dir.
  std::string slab_gran = args.Get("--slab-granularity", "");
  if (!slab_gran.empty()) ::setenv("DFKV_SLAB_GRANULARITY", slab_gran.c_str(), 1);
  // PUT admission gate: cap concurrent foreground PUTs before RAM admission
  // or disk I/O. Excess requests fail with kCacheFull without joining a
  // pending write; background flush remains ungated. 0 = off (default).
  std::string put_limit = args.Get("--put-inflight-limit", "");
  if (!put_limit.empty()) ::setenv("DFKV_PUT_INFLIGHT_LIMIT", put_limit.c_str(), 1);
  std::string tcp_max_conns = args.Get("--tcp-max-conns", "");
  if (!tcp_max_conns.empty())
    ::setenv("DFKV_TCP_MAX_CONNS", tcp_max_conns.c_str(), 1);
  std::string tcp_io_timeout_s = args.Get("--tcp-io-timeout-s", "");
  if (!tcp_io_timeout_s.empty())
    ::setenv("DFKV_TCP_IO_TIMEOUT_S", tcp_io_timeout_s.c_str(), 1);
  // --- Remaining env-only knobs: same flag→env facade pattern. Each flag is
  // an empty-by-default string; when set it wins over a pre-set DFKV_* env via
  // setenv(overwrite=1). See DEPLOY.md and METRICS.md for the current matrix.
  // RDMA transport knobs (DFKV_RDMA_DEV already handled via --rdma-dev above,
  // which the RdmaServer constructor takes as a direct arg, so no setenv here).
  std::string rdma_depth = args.Get("--rdma-depth", "");
  if (!rdma_depth.empty()) ::setenv("DFKV_RDMA_DEPTH", rdma_depth.c_str(), 1);
  std::string rdma_numa = args.Get("--rdma-numa", "");
  if (!rdma_numa.empty()) ::setenv("DFKV_RDMA_NUMA", rdma_numa.c_str(), 1);
  std::string rdma_idle_ms = args.Get("--rdma-idle-ms", "");
  if (!rdma_idle_ms.empty()) ::setenv("DFKV_RDMA_IDLE_MS", rdma_idle_ms.c_str(), 1);
  std::string rdma_op_timeout_ms = args.Get("--rdma-op-timeout-ms", "");
  if (!rdma_op_timeout_ms.empty())
    ::setenv("DFKV_RDMA_OP_TIMEOUT_MS", rdma_op_timeout_ms.c_str(), 1);
  // io_uring async-GET path (built -DDFKV_WITH_URING).
  std::string server_uring = args.Get("--server-uring", "");
  if (!server_uring.empty()) ::setenv("DFKV_SERVER_URING", server_uring.c_str(), 1);
  std::string server_uring_depth = args.Get("--server-uring-depth", "");
  if (!server_uring_depth.empty())
    ::setenv("DFKV_SERVER_URING_DEPTH", server_uring_depth.c_str(), 1);
  // RAM tier flusher + NUMA placement.
  std::string ram_flush_threads = args.Get("--ram-flush-threads", "");
  if (!ram_flush_threads.empty())
    ::setenv("DFKV_RAM_FLUSH_THREADS", ram_flush_threads.c_str(), 1);
  std::string ram_tier_numa = args.Get("--ram-tier-numa", "");
  if (!ram_tier_numa.empty()) {
    if (ram_tier_numa != "off" && ram_tier_numa != "interleave") {
      std::fprintf(stderr,
                   "--ram-tier-numa must be 'off' or 'interleave'\n");
      return 1;
    }
    ::setenv("DFKV_RAM_TIER_NUMA", ram_tier_numa.c_str(), 1);
  }
  // RAM tier lock shards (1-64; the tier halves it while a shard would hold
  // <32 extents, so small arenas degrade to fewer shards).
  std::string ram_tier_shards = args.Get("--ram-tier-shards", "");
  if (!ram_tier_shards.empty())
    ::setenv("DFKV_RAM_TIER_SHARDS", ram_tier_shards.c_str(), 1);
  // RDMA v2 receive-pool budget and lazy commit chunk.
  std::string recv_seg = args.Get("--rdma-recv-segment-size", "");
  if (!recv_seg.empty())
    ::setenv("DFKV_RDMA_RECV_SEGMENT_SIZE", recv_seg.c_str(), 1);
  std::string recv_chunk = args.Get("--rdma-recv-chunk-bytes", "");
  if (!recv_chunk.empty())
    ::setenv("DFKV_RDMA_RECV_CHUNK_BYTES", recv_chunk.c_str(), 1);
  // Disk hash weight — per-disk vnode multiplier (default 10; was 1).
  std::string disk_hash_weight = args.Get("--disk-hash-weight", "");
  if (!disk_hash_weight.empty())
    ::setenv("DFKV_DISK_HASH_WEIGHT", disk_hash_weight.c_str(), 1);
  // Read coalesce — convoy merge + RAM promotion (opt-in; default off).
  std::string read_coalesce = args.Get("--read-coalesce", "");
  if (!read_coalesce.empty())
    ::setenv("DFKV_READ_COALESCE", read_coalesce.c_str(), 1);
  // Slab table sync cadence (ms).
  std::string slab_table_sync_ms = args.Get("--slab-table-sync-ms", "");
  if (!slab_table_sync_ms.empty())
    ::setenv("DFKV_SLAB_TABLE_SYNC_MS", slab_table_sync_ms.c_str(), 1);
  // Background free-slot reclaimer cadences (ms; 0 disables).
  std::string slab_reclaim_ms = args.Get("--slab-reclaim-ms", "");
  if (!slab_reclaim_ms.empty())
    ::setenv("DFKV_SLAB_RECLAIM_MS", slab_reclaim_ms.c_str(), 1);
  std::string ram_reclaim_ms = args.Get("--ram-reclaim-ms", "");
  if (!ram_reclaim_ms.empty())
    ::setenv("DFKV_RAM_RECLAIM_MS", ram_reclaim_ms.c_str(), 1);
  // Log level (DFKV_LOG: e.g. "INFO", "DEBUG", "WARN").
  std::string log_level = args.Get("--log", "");
  if (!log_level.empty()) ::setenv("DFKV_LOG", log_level.c_str(), 1);
  // A configured cache node must never remain active-but-unregistered forever.
  // The flag wins over the env twin; both parse strictly and stay inside a hard
  // operational envelope so typos cannot silently disable the fail-closed exit.
  constexpr int kDefaultMdsRegistrationTimeoutMs = 60000;
  constexpr int kMinMdsRegistrationTimeoutMs = 1000;
  constexpr int kMaxMdsRegistrationTimeoutMs = 600000;
  int mds_registration_timeout_ms = kDefaultMdsRegistrationTimeoutMs;
  dfkv::config_dump::Source mds_registration_timeout_source =
      dfkv::config_dump::Source::kDefault;
  if (args.Has("--mds-registration-timeout-ms")) {
    const std::string value =
        args.Get("--mds-registration-timeout-ms", "");
    const auto parsed = std::from_chars(
        value.data(), value.data() + value.size(),
        mds_registration_timeout_ms);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != value.data() + value.size()) {
      std::fprintf(
          stderr,
          "dfkv_server: --mds-registration-timeout-ms must be an integer "
          "in [%d,%d], got '%s'\n",
          kMinMdsRegistrationTimeoutMs, kMaxMdsRegistrationTimeoutMs,
          value.c_str());
      return 2;
    }
    mds_registration_timeout_source = dfkv::config_dump::Source::kFlag;
  } else if (const char* value =
                 std::getenv("DFKV_MDS_REGISTRATION_TIMEOUT_MS")) {
    const char* end = value + std::strlen(value);
    const auto parsed =
        std::from_chars(value, end, mds_registration_timeout_ms);
    if (parsed.ec != std::errc{} || parsed.ptr != end) {
      std::fprintf(
          stderr,
          "dfkv_server: DFKV_MDS_REGISTRATION_TIMEOUT_MS must be an integer "
          "in [%d,%d], got '%s'\n",
          kMinMdsRegistrationTimeoutMs, kMaxMdsRegistrationTimeoutMs, value);
      return 2;
    }
    mds_registration_timeout_source = dfkv::config_dump::Source::kEnv;
  }
  if (args.ok() &&
      (mds_registration_timeout_ms < kMinMdsRegistrationTimeoutMs ||
       mds_registration_timeout_ms > kMaxMdsRegistrationTimeoutMs)) {
    std::fprintf(
        stderr,
        "dfkv_server: MDS registration timeout must be in [%d,%d] ms, got %d\n",
        kMinMdsRegistrationTimeoutMs, kMaxMdsRegistrationTimeoutMs,
        mds_registration_timeout_ms);
    return 2;
  }
  if (!args.ok()) {
    std::fprintf(stderr, "dfkv_server: %s\n(run with --help for usage)\n",
                 args.error().c_str());
    return 2;
  }
  // A mistyped --advertise (no ':') used to wrap rfind(':')+1 into a garbage
  // port; reject it up front rather than register an unreachable address.
  if (!advertise.empty() && !dfkv::IsValidHostPort(advertise)) {
    std::fprintf(stderr, "dfkv_server: --advertise must be host:port (1..65535), "
                 "got '%s'\n", advertise.c_str());
    return 2;
  }
  // Before anything is opened: this used to be found out only after the disks
  // and the RAM tier had been brought up.
  if (!mds.empty() && (node_id.empty() || advertise.empty())) {
    std::fprintf(stderr, "--mds requires --id and --advertise <ip:port>\n");
    return 1;
  }
  // Starting without these is allowed (and used by tests), but in a deployment
  // it is nearly always a unit file that lost a line.
  if (!args.Has("--dir"))
    DFKV_LOG_WARN("--dir not given; using the default " + dir);
  if (!args.Has("--cap"))
    DFKV_LOG_WARN("--cap not given; using the default " + std::to_string(cap) +
                  " bytes");
  if (weight < 1)
    DFKV_LOG_WARN("--weight " + std::to_string(weight) +
                  " is below 1; using 1");
  (void)rdma_port;
  // Record the direct (non-env-facade) flags for the startup config dump.
  // Source is flag ONLY when the flag was actually passed, else default — so the
  // dump truthfully separates externally-specified from built-in-default values.
  // The env-facade flags above already reach environ via setenv, so Emit()'s
  // environ scan folds them in (shown as env with their effective value).
  {
    namespace cd = dfkv::config_dump;
    auto src = [&](const char* flag) {
      return args.Has(flag) ? cd::Source::kFlag : cd::Source::kDefault;
    };
    cd::Record("dir", dir, src("--dir"));
    cd::Record("port", std::to_string(port), src("--port"));
    cd::Record("cap", std::to_string(cap), src("--cap"));
    cd::Record("rdma_dev", rdma_dev.empty() ? "(unset)" : rdma_dev, src("--rdma-dev"));
    cd::Record("mds", mds.empty() ? "(none)" : mds, src("--mds"));
    cd::Record("group", group, src("--group"));
    cd::Record("id", node_id.empty() ? "(auto)" : node_id, src("--id"));
    cd::Record("advertise", advertise.empty() ? "(auto)" : advertise, src("--advertise"));
    cd::Record("weight", std::to_string(weight), src("--weight"));
    cd::Record("metrics_port", std::to_string(metrics_port), src("--metrics-port"));
    cd::Record("metrics_bind", metrics_bind.empty() ? "(any)" : metrics_bind, src("--metrics-bind"));
    cd::Record("DFKV_MDS_REGISTRATION_TIMEOUT_MS",
               std::to_string(mds_registration_timeout_ms),
               mds_registration_timeout_source);
  }
  std::signal(SIGINT, OnSig);
  std::signal(SIGTERM, OnSig);
  std::signal(SIGHUP, OnHup);  // reload the tenant quotas file
  {
    struct sigaction fatal {};
    fatal.sa_sigaction = OnFatalSignal;
    fatal.sa_flags = SA_SIGINFO | SA_NODEFER;  // the handler may re-raise
    sigemptyset(&fatal.sa_mask);
    for (int sig : {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT})
      ::sigaction(sig, &fatal, &g_previous_fatal[sig]);
  }

  // split --dir on commas into one entry per NVMe SSD
  std::vector<std::string> dirs;
  for (size_t i = 0, j; i <= dir.size(); i = j + 1) {
    j = dir.find(',', i);
    if (j == std::string::npos) j = dir.size();
    if (j > i) dirs.push_back(dir.substr(i, j - i));
  }
  if (dirs.empty()) dirs.push_back(dir);

  // The slow part of a start: the disks are opened and the RAM tier arena is
  // faulted in. A stop signal during it gives the arena up at once; the disk
  // opens run to their end (seconds) and are then closed again.
  KvNodeServer srv(dirs, cap, &g_stop);
  if (g_stop)
    return EndCancelledStart(srv, "while opening the disks", process_started,
                             {});
  // TCP request frames share the RDMA path's resolved max-value bound (OOM-DoS
  // guard; see utils/wire_limits.h).
  srv.set_max_request_payload(dfkv::wire_limits::MaxRequestPayload());
  // Identity for Prometheus labels: reuse the MDS --id/--group when present.
  if (!node_id.empty() || group != "default") srv.set_identity(node_id, group);
  if (srv.Start(port) != Status::kOk) {
    std::fprintf(stderr, "failed to start on port %d\n", port);
    return 1;
  }
  std::printf("PORT %d\n", srv.port());
  std::fflush(stdout);


  // Optional Prometheus /metrics endpoint (declared here, started after the RDMA
  // server below so its render callback can fold in the RDMA server counters).
  std::unique_ptr<dfkv::MetricsHttpServer> mhttp;
  DFKV_LOG_INFO("dfkv_server listening on port " + std::to_string(srv.port()) + ", dir=" + dir);

  // Announce the transport build mode loudly so a TCP-only binary (built without
  // -DDFKV_WITH_RDMA=ON) can never be mistaken for an RDMA one at deploy time.
#ifdef DFKV_WITH_RDMA
  DFKV_LOG_INFO("dfkv build: transport=RDMA v2 (libibverbs) + native TCP");
#else
  DFKV_LOG_INFO("dfkv build: transport=TCP-only (built WITHOUT -DDFKV_WITH_RDMA=ON)");
  if (rdma_port >= 0) {
    DFKV_LOG_ERROR(
        "--rdma-port requires an RDMA-enabled binary; rebuild with "
        "-DDFKV_WITH_RDMA=ON (needs libibverbs-dev)");
    srv.Stop();
    return 1;
  }
  if (!rdma_dev.empty())
    DFKV_LOG_WARN("--rdma-dev ignored: this binary has no RDMA support");
#endif

#ifdef DFKV_WITH_RDMA
  // --max-msg is the hard payload ceiling. RDMA v2 leases slots from one
  // process-wide receive segment sized by DFKV_RDMA_RECV_SEGMENT_SIZE.
  std::unique_ptr<dfkv::RdmaServer> rsrv;
  std::unique_ptr<dfkv::rdma::RdmaHealthMonitor> rdma_health;
  if (rdma_port >= 0) {
    rsrv = std::make_unique<dfkv::RdmaServer>(
        [&srv](uint8_t op, const dfkv::BlockKey& key, uint64_t off,
               uint64_t len, const char* pl, uint64_t pll, std::string* out,
               size_t* value_len) {
          return srv.ProcessRequestForKey(op, key, off, len, pl, pll, out,
                                          value_len);
        },
        /*max_msg=*/max_msg, rdma_dev);
    rsrv->set_range_handler(  // disk -> registered dbuf -> RDMA scatter
        [&srv](const dfkv::BlockKey& key, uint64_t off, uint64_t len,
               char* io_buf, size_t cap, const char** out_data,
               size_t* out_len, size_t* value_len) {
          return srv.RangeDirectForKey(key, off, len, io_buf, cap, out_data,
                                       out_len, value_len);
        });
  rsrv->set_cache_direct_handler(  // server-side direct PUT: RDMA dbuf -> O_DIRECT write
        [&srv](const dfkv::BlockKey& key, char* data, size_t len, size_t cap,
               std::chrono::steady_clock::time_point batch_arrival) {
          return srv.CacheDirectForKey(key, data, len, cap, batch_arrival);
        });
    rsrv->set_prepare_read_handler(
        [&srv](const dfkv::BlockKey& key, uint64_t off, uint64_t len,
               char* staging, size_t cap) {
          return srv.PrepareReadForKey(key, off, len, staging, cap);
        });
    rsrv->set_lookup_handler([&srv](const dfkv::BlockKey& key) {
      return srv.LookupForGet(key);
    });
    // B5-3: dynamic-pull GETs of arena-resident values serve the pinned arena
    // address instead of staging a payload-sized copy per pull.
    if (srv.ram_enabled()) {
      rsrv->set_mw_arena(srv.ram_tier());
      rsrv->set_pinned_ram_handler(
          [&srv](const dfkv::BlockKey& key, uint64_t off, uint64_t len,
                 dfkv::PreparedRead* out) {
            return srv.RamPinnedHitForKey(key, off, len, out);
          });
    }
    // The arena as a whole is not registered with the NICs: every pull gets
    // an exact, revocable READ MR over just the value it serves, and nothing
    // else on the server refers to arena memory by MR. Registering the whole
    // arena on every rail pinned it once per rail for no reader -- tens of
    // seconds of startup and shutdown on a large tier.
    if (srv.ram_enabled()) {
      DFKV_LOG_INFO("dfkv_server RAM hot tier: RDMA zero-copy GET enabled (arena " +
                    std::to_string(srv.ram_arena_bytes()) +
                    " bytes, registered per read)");
    }
    const Status rdma_status = rsrv->Start(rdma_port);
    if (rdma_status != Status::kOk) {
      DFKV_LOG_ERROR(
          "dfkv_server RDMA listener failed to start; refusing readiness and "
          "MDS registration");
      srv.Stop();
      return 1;
    }
    DFKV_LOG_INFO("dfkv_server RDMA listening (TCP bootstrap) on port " +
                  std::to_string(rsrv->port()) +
                  (rdma_dev.empty() ? "" : ", dev=" + rdma_dev));
    if (g_stop) {
      const int exit_code = EndCancelledStart(
          srv, "while starting the RDMA listener", process_started,
          [&rsrv] { rsrv->Quiesce(); });
      rsrv->Stop();
      return exit_code;
    }
    // Force-resolve the RDMA depth/uring knobs so their defaults land in the
    // config dump (they are otherwise read lazily on the serve path, after Emit).
    (void)rsrv->PipelineDepth();
    (void)rsrv->UseUringPath();
    const std::vector<std::string> health_devices = rsrv->DeviceNames();
    const char* override_env = std::getenv("DFKV_RDMA_HEALTH_FILE");
    const std::string health_file = override_env ? override_env : "";
    dfkv::envparse::U64Spec recovery_spec;
    recovery_spec.fallback = 3;
    recovery_spec.min = 1;
    recovery_spec.max = 100;
    recovery_spec.legacy = dfkv::envparse::Legacy::kFull;
    recovery_spec.negative = dfkv::envparse::Negative::kWraps;
    const uint32_t recovery_samples = static_cast<uint32_t>(
        dfkv::envparse::EnvU64("DFKV_RDMA_HEALTH_RECOVERY_SAMPLES",
                               recovery_spec));
    rdma_health = std::make_unique<dfkv::rdma::RdmaHealthMonitor>(
        [health_devices, health_file] {
          return dfkv::rdma::ProbeIbDeviceHealth(health_devices, health_file);
        },
        recovery_samples);
    const auto initial_health = rdma_health->Sample();
    DFKV_LOG_INFO("rdma-health: monitoring " +
                  std::to_string(initial_health.ib_devices.size()) +
                  " device(s), ring_eligible=" +
                  (initial_health.ring_eligible ? "true" : "false"));
    dfkv::config_dump::RecordResolved(
        "DFKV_RDMA_HEALTH_RECOVERY_SAMPLES",
        std::to_string(recovery_samples));
    dfkv::config_dump::RecordResolved(
        "DFKV_RDMA_HEALTH_FILE",
        health_file.empty() ? "(sysfs)" : health_file);
  }

#endif

  // Start /metrics now that the (optional) RDMA server exists: the render
  // callback combines the cache-node metrics with the RDMA server counters.
  // /readyz requires startup completion, first successful MDS registration
  // (when configured), and CURRENT local storage health. Later transient MDS
  // heartbeat loss does not erase first-registration history, while a terminal
  // metadata/store/RAM failure immediately removes readiness.
  auto ready_flag = std::make_shared<std::atomic<bool>>(false);
  auto mds_ready =
      std::make_shared<std::atomic<bool>>(mds.empty());
  auto registrar_metrics =
      std::make_shared<std::atomic<dfkv::MdsRegistrar*>>(nullptr);
  srv.set_metrics_extension([&] {
    std::string s;
    // What this node reports to the MDS about its own placement eligibility
    // (cache/node_member_report.h): no failed disk store and, with an RDMA
    // listener, a healthy rail.
    const bool disks_healthy = srv.DisksHealthy();
    bool eligibility_emitted = false;
#ifdef DFKV_WITH_RDMA
    if (rsrv) s += rsrv->MetricsText();
    if (rdma_health) {
      s += rdma_health->MetricsText(disks_healthy);
      eligibility_emitted = true;
    }
#endif
    if (auto* r = registrar_metrics->load(std::memory_order_acquire))
      s += r->MetricsText();
    auto gauge = [&s](const char* name, const char* help, bool value) {
      s += "# HELP "; s += name; s += " "; s += help; s += "\n";
      s += "# TYPE "; s += name; s += " gauge\n";
      s += name; s += value ? " 1\n" : " 0\n";
    };
    if (!eligibility_emitted)
      gauge("dfkv_server_ring_eligible",
            "Whether this node reports itself eligible for placement",
            disks_healthy);
    const bool startup = ready_flag->load(std::memory_order_acquire);
    const bool registered = mds_ready->load(std::memory_order_acquire);
    const bool healthy = srv.Healthy();
    gauge("dfkv_server_startup_complete",
          "Whether every local startup stage completed", startup);
    gauge("dfkv_server_mds_registration_ready",
          "Whether the initial MDS registration gate is satisfied", registered);
    gauge("dfkv_server_healthy",
          "Whether local storage and required RAM tiers are healthy", healthy);
    gauge("dfkv_server_ready",
          "Whether startup, MDS registration, and local health are ready",
          startup && registered && healthy);
    return s;
  });
  // The last point at which a stop signal cancels the start, ahead of the
  // health endpoints (a start that is being cancelled must not answer ready
  // for an instant) and of the registration with the MDS. Past it the node
  // may be registered, and a stop is the ordinary one of the main loop below
  // (the registration loop is stopped; the membership lapses with its lease,
  // there is no explicit deregistration).
  if (g_stop) {
    const int exit_code = EndCancelledStart(
        srv, "before registering with the MDS", process_started, [&] {
#ifdef DFKV_WITH_RDMA
          if (rsrv) rsrv->Quiesce();
#endif
        });
#ifdef DFKV_WITH_RDMA
    if (rsrv) rsrv->Stop();
#endif
    return exit_code;
  }
  if (metrics_port >= 0) {
    mhttp = std::make_unique<dfkv::MetricsHttpServer>(
        [&srv] { return srv.FullMetricsText(); });
    mhttp->set_health_check([&srv] { return srv.Healthy(); });
    mhttp->set_ready_check([ready_flag, mds_ready, &srv] {
      return ready_flag->load(std::memory_order_acquire) &&
             mds_ready->load(std::memory_order_acquire) && srv.Healthy();
    });
    if (mhttp->Start(metrics_port, metrics_bind) == Status::kOk)
      DFKV_LOG_INFO("dfkv_server /metrics on port " + std::to_string(mhttp->port()));
    else
      DFKV_LOG_WARN("dfkv_server /metrics failed to start on port " + std::to_string(metrics_port));
  }

  // DFKV_DRAIN (default off): this node may drain (kDrain from dfkvctl, and
  // DFKV_STOP_DRAIN_MS on a stop); see DrainState. Needs an MDS.
  const bool drain_enabled = dfkv::config_dump::EnvBool("DFKV_DRAIN", false);
  const uint64_t stop_drain_ms = std::min<uint64_t>(
      dfkv::config_dump::EnvU64("DFKV_STOP_DRAIN_MS", 0), 600000);
  constexpr int kDrainBeatTimeoutMs = 15000;
  auto drain_state = std::make_shared<std::atomic<int>>(
      static_cast<int>(DrainState::kNone));
  if (drain_enabled && mds.empty())
    DFKV_LOG_WARN("DFKV_DRAIN=1 without --mds: there is no MDS to tell; draining is off");
  std::unique_ptr<dfkv::MdsRegistrar> registrar;
  if (!mds.empty()) {
    size_t colon = advertise.rfind(':');
    std::string aip = advertise.substr(0, colon);
    uint32_t aport = static_cast<uint32_t>(std::atoi(advertise.c_str() + colon + 1));
    std::vector<std::string> eps;
    for (size_t i = 0, j; i <= mds.size(); i = j + 1) {
      j = mds.find(',', i);
      if (j == std::string::npos) j = mds.size();
      if (j > i) eps.push_back(mds.substr(i, j - i));
    }
    dfkv::MemberInfo self{node_id, aip, aport,
                          static_cast<uint32_t>(weight < 1 ? 1 : weight)};
    self.tcp_port = static_cast<uint32_t>(srv.port());  // TCP wire/stat port -> `dfkvctl stat`
    // Self-description reported on register + every heartbeat, surfaced by
    // `dfkvctl ring` (fleet-wide version/config audit without per-node ssh).
    // Values are the RESOLVED runtime truth (what was actually constructed),
    // not flag intent -- e.g. engine comes from DiskCacheGroup's choice.
    {
      std::string info = std::string("ver=") + dfkv::Version();
      info += ",engine=" + srv.engine_name();
      // Resolved I/O mode (not env intent): tmpfs-style O_DIRECT rejection
      // demotes to buffered, and the fleet audit must see that.
      if ((srv.engine_name() == "slab" || srv.engine_name() == "mixed") &&
          !srv.write_mode().empty())
        info += ",wr=" + srv.write_mode();
      info += ",disks=" + std::to_string(srv.DiskCount());
      info += ",cap=" + std::to_string(cap);
      info += ",ram=" + (srv.ram_enabled()
                             ? std::to_string(srv.ram_arena_bytes()) : std::string("0"));
#ifdef DFKV_WITH_RDMA
      info += ",rdma=" + std::string(rsrv ? (rdma_dev.empty() ? "on" : rdma_dev.c_str()) : "off");
      // Server pipeline depth: clients pipelining deeper than this hit RNR-retry
      // degradation, so the fleet audit must show it next to the transport.
      if (rsrv) info += ",qd=" + std::to_string(rsrv->PipelineDepth());
#else
      info += ",rdma=nobuild";
#endif
      self.info = std::move(info);
    }
    // Heartbeat interval: DFKV_MDS_HEARTBEAT_MS, 1000..10000, default 10000.
    // The rule: a heartbeat plus one registrar I/O timeout must land within
    // the MDS lease, so that one late by a whole timeout still renews it.
    constexpr int kMaxHeartbeatMs = 10000;
    static_assert(kMaxHeartbeatMs + dfkv::MdsRegistrar::kDefaultIoTimeoutMs <
                      dfkv::MdsServer::kTtlSeconds * 1000,
                  "heartbeat + registrar I/O timeout must stay below the lease");
    dfkv::envparse::U64Spec heartbeat_spec;
    heartbeat_spec.fallback = 10000;
    heartbeat_spec.min = 1000;
    heartbeat_spec.max = kMaxHeartbeatMs;
    heartbeat_spec.above = dfkv::envparse::Above::kClamp;
    const int heartbeat_ms = static_cast<int>(
        dfkv::envparse::EnvU64("DFKV_MDS_HEARTBEAT_MS", heartbeat_spec));
    dfkv::config_dump::RecordResolved("DFKV_MDS_HEARTBEAT_MS",
                                      std::to_string(heartbeat_ms));
    registrar = std::make_unique<dfkv::MdsRegistrar>(
        std::move(eps), group, self, heartbeat_ms,
        dfkv::MdsRegistrar::kDefaultIoTimeoutMs, /*is_client=*/false,
        mds_registration_timeout_ms);
    registrar_metrics->store(registrar.get(), std::memory_order_release);
    // Dynamic stats ride every heartbeat (STA1): ring-level capacity/health
    // becomes answerable from the MDS alone (dfkvctl stats / dfkv_mds_group_*),
    // no per-node scrape. All reads are relaxed atomics -- negligible cost.
    registrar->set_stats_provider([&srv, cap]() {
      dfkv::MemberStats st;
      st.capacity_bytes = cap;
      st.used_bytes = srv.UsedBytes();
      st.objects = srv.Count();
      st.hits_total = srv.m_cache_hit();
      st.misses_total = srv.m_cache_miss();
      st.evictions_total = srv.Evictions();
      st.puts_total = srv.m_cache_put();
      st.uptime_seconds = srv.UptimeSeconds();
      st.put_busy_total = srv.PutBusyTotal();
      st.dio_write_fallbacks = srv.DioWriteFallbacks();
      st.ram_used_bytes = srv.RamUsedBytes();
      st.ram_hits_total = srv.RamHits();
      return st;
    });
    // A node whose disk store has failed for good reports itself out of
    // placement on its next heartbeat, with or without RDMA rails.
    std::function<dfkv::MemberHealth()> rails;
#ifdef DFKV_WITH_RDMA
    if (rdma_health) rails = [&rdma_health] { return rdma_health->Sample(); };
#endif
    dfkv::ReportNodeHealth(registrar.get(), &srv, self.info, std::move(rails));
    if (drain_enabled) {
      registrar->set_drain_provider([drain_state] {
        return drain_state->load() != static_cast<int>(DrainState::kNone);
      });
      dfkv::MdsRegistrar* reg = registrar.get();
      srv.set_drain_handler([drain_state, reg, node_id](bool on) {
        drain_state->store(static_cast<int>(on ? DrainState::kByCtl
                                               : DrainState::kNone));
        // Answered once the MDS has the record in its new state; clients
        // pick it up within their poll interval from there.
        const bool written = reg->BeatNow(kDrainBeatTimeoutMs);
        DFKV_LOG_INFO(std::string("dfkv_server ") +
                      (on ? "draining" : "no longer draining") + " id=" +
                      node_id + (written ? "" : " (the MDS has not taken it yet)"));
        return written ? dfkv::Status::kOk : dfkv::Status::kIOError;
      });
    }
#ifdef DFKV_WITH_RDMA
    // A rail's port going down or coming back reaches the MDS with an early
    // heartbeat (whose health sample sees it), not up to an interval later.
    // The listener holds the registrar by pointer: it is cleared, and
    // rsrv->Stop() joins the port watcher, before the registrar is destroyed.
    // Keep it so: no early return between here and that teardown.
    if (rsrv) {
      dfkv::MdsRegistrar* reg = registrar.get();
      rsrv->set_rail_state_listener(
          [reg](const std::string&, bool) { reg->RequestBeat(); });
    }
#endif
    registrar->set_registered_callback([mds_ready, group, node_id] {
      mds_ready->store(true, std::memory_order_release);
      DFKV_LOG_INFO("dfkv_server registered with MDS group=" + group +
                    " id=" + node_id);
    });
    registrar->Start();
    DFKV_LOG_INFO("dfkv_server MDS registration loop started group=" + group +
                  " id=" + node_id + " advertise=" + advertise);
  }

  dfkv::config_dump::Emit("server");
  ready_flag->store(true, std::memory_order_release);
  DFKV_LOG_INFO(mds.empty()
                    ? "dfkv_server ready (all startup stages complete)"
                    : "dfkv_server initialized; readiness awaits initial MDS registration");

  while (!g_stop &&
         !(registrar && registrar->first_registration_timed_out())) {
    struct timespec ts{0, 50 * 1000 * 1000};
    nanosleep(&ts, nullptr);
    if (g_reload.exchange(false, std::memory_order_relaxed)) {
      std::string error;
      if (srv.ReloadTenantQuotas(&error))
        DFKV_LOG_INFO("dfkv_server: tenant quotas reloaded (SIGHUP)");
      else
        DFKV_LOG_WARN("dfkv_server: tenant quotas not reloaded (SIGHUP): " +
                      error + "; the previous quotas stay");
    }
  }
  const bool registration_timed_out =
      registrar && registrar->first_registration_timed_out();
  if (registration_timed_out) {
    DFKV_LOG_ERROR(
        "dfkv_server shutting down after first MDS registration deadline");
  } else {
    DFKV_LOG_INFO("dfkv_server shutting down");
  }
  // Stop-time drain: clients stop writing here and read on as before, while
  // this node keeps serving for DFKV_STOP_DRAIN_MS (a second stop signal
  // ends it). A node `dfkvctl drain` drained already is past that.
  if (drain_enabled && registrar && !registration_timed_out &&
      stop_drain_ms > 0 &&
      drain_state->load() == static_cast<int>(DrainState::kNone)) {
    const int signals = g_stop_signals.load(std::memory_order_relaxed);
    drain_state->store(static_cast<int>(DrainState::kByStop));
    const bool written = registrar->BeatNow(kDrainBeatTimeoutMs);
    DFKV_LOG_INFO("dfkv_server draining for " + std::to_string(stop_drain_ms) +
                  " ms before it stops" +
                  (written ? "" : " (the MDS has not taken it)") +
                  "; a second stop signal ends the wait");
    const auto until = std::chrono::steady_clock::now() +
                       std::chrono::milliseconds(stop_drain_ms);
    while (std::chrono::steady_clock::now() < until &&
           g_stop_signals.load(std::memory_order_relaxed) == signals) {
      struct timespec ts{0, 50 * 1000 * 1000};
      nanosleep(&ts, nullptr);
    }
  }
  srv.set_drain_handler(nullptr);
  // Order: stop everything that calls into the store, get the disks clean,
  // and only then release memory. Releasing a large RAM tier takes tens of
  // seconds; a stop that is killed during that must not cost the cache.
  const auto stop_started = std::chrono::steady_clock::now();
  auto stop_seconds = [stop_started] {
    char text[32];
    std::snprintf(text, sizeof(text), "%.2f s",
                  std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - stop_started).count());
    return std::string(text);
  };
  if (mhttp) mhttp->Stop();
  registrar_metrics->store(nullptr, std::memory_order_release);
#ifdef DFKV_WITH_RDMA
  if (rsrv) rsrv->set_rail_state_listener(nullptr);
#endif
  if (registrar) registrar->Stop();
  // A node drained by `dfkvctl drain` is being removed: its record goes now,
  // and clients drop it on their next poll instead of when its lease ends.
  if (registrar && drain_state->load() == static_cast<int>(DrainState::kByCtl))
    DFKV_LOG_INFO(registrar->Deregister()
                      ? "shutdown: drained node deregistered from the MDS"
                      : "shutdown: could not deregister the drained node; its record ends with its lease");
  // Quiesce() joins the RDMA serve threads; one that is parked in a PUT
  // waiting for a flush must be answered first.
  srv.CancelParkedPuts();
#ifdef DFKV_WITH_RDMA
  if (rsrv) {
    rsrv->Quiesce();
    DFKV_LOG_INFO("shutdown: RDMA requests stopped after " + stop_seconds());
  }
#endif
  const int exit_code = registration_timed_out ? 1 : 0;
  // The last line of a stop. From here a kill costs nothing.
  const char* const next_step =
      DFKV_EXIT_RUNS_DESTRUCTORS ? "; releasing memory" : "; exiting";
  if (srv.Shutdown(stop_started)) {
    DFKV_LOG_INFO("dfkv_server shut down cleanly in " + stop_seconds() +
                  next_step);
  } else {
    DFKV_LOG_WARN("dfkv_server stopped in " + stop_seconds() +
                  ", but not every disk is clean" + next_step);
  }
#if !DFKV_EXIT_RUNS_DESTRUCTORS
  // Everything that has to outlive the process is on disk now, and every
  // thread that touches the store has been joined. What is left is giving
  // memory back: one heap node per cached object in every disk's index, the
  // RAM tier arena, the RDMA registrations. The kernel does that in one go
  // when the process ends; walking the indexes object by object in destructors
  // took seconds with millions of objects. Sanitizer builds keep the orderly
  // teardown: their end-of-process reports hang off it.
  std::fflush(stdout);
  std::fflush(stderr);
  ::_exit(exit_code);
#else
#ifdef DFKV_WITH_RDMA
  if (rsrv) rsrv->Stop();
#endif
  return exit_code;
#endif
}
