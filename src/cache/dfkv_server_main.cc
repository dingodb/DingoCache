/* Standalone cache-node binary for the harness / pre-prod functional bring-up.
 * Usage: dfkv_server --dir <p1[,p2,p3...]> --port <p|0> --cap <bytes>
 *   --dir accepts comma-separated NVMe SSD paths (like dingo-cache --cache_dir);
 *   --cap is the TOTAL capacity, split evenly across the disks.
 * Prints "PORT <port>" on stdout once listening, then runs until SIGTERM/SIGINT. */
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <atomic>
#include <memory>
#include <vector>

#include "utils/args.h"
#include "cache/kv_node_server.h"
#include "compat/sgengine_tcp_frontend.h"
#include "utils/wire_limits.h"
#include "utils/log.h"
#include "common/config_dump.h"
#include "mds/mds_registrar.h"
#include "utils/metrics_http.h"
#include "common/version.h"
#ifdef DFKV_WITH_RDMA
#include "cache/rdma_server.h"
#include "compat/sgengine_rdma_frontend.h"
#endif

using dfkv::KvNodeServer;
using dfkv::Status;

static volatile sig_atomic_t g_stop = 0;
static void OnSig(int) { g_stop = 1; }

int main(int argc, char** argv) {
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
      "  --sgengine-tcp-port <p> isolated SGEngine v1 compatibility TCP port (omit = off)\n"
      "  --sgengine-rdma-port <p> isolated SGEngine v1 compatibility RDMA bootstrap port (omit = off)\n"
      "  --rdma-dev <names>   HCA whitelist; default first ACTIVE local HCA (RDMA build only)\n"
      "  --mds <ip:port,...>  MDS endpoints to register into (with --group/--id/--advertise)\n"
      "  --group <g>          membership group name (default \"default\")\n"
      "  --id <id>            node id; --advertise <ip:port>  address peers reach (rdma-port)\n"
      "  --weight <n>         consistent-hash weight (default 1)\n"
      "  --metrics-port <p>   enable Prometheus /metrics (omit = off); --metrics-bind <addr>\n"
      "  --store-engine <e>   storage backend: file | slab (default file)\n"
      "  --slab-write <m>     slab I/O mode: direct | buffered (default direct)\n"
      "  --ram-tier <on|off>  RAM hot tier: write-through arena + RDMA zero-copy GET (default off)\n"
      "  --ram-tier-bytes <n> RAM arena size in bytes (default 4 GiB; pinned once registered)\n"
      "  --slab-granularity <n>  slab slot quantum bytes (default 1 MiB; CHANGING IT COLD-STARTS the store)\n"
      "  --put-inflight-limit <n>  cap concurrent disk PUTs; excess fast-fail kCacheFull (0 = off)\n"
      "  --rdma-depth <n>     RDMA write pipeline depth (must be <= client's; env DFKV_RDMA_DEPTH)\n"
      "  --rdma-numa <0|1>    NUMA-local rail selection per connection (env DFKV_RDMA_NUMA)\n"
      "  --rdma-idle-ms <n>   idle connection reaper interval ms (env DFKV_RDMA_IDLE_MS)\n"
      "  --rdma-op-timeout-ms <n>  per-op RDMA timeout ms (env DFKV_RDMA_OP_TIMEOUT_MS)\n"
      "  --server-uring <0|1> enable io_uring async-GET path (env DFKV_SERVER_URING; needs -DDFKV_WITH_URING)\n"
      "  --server-uring-depth <n>  io_uring SQ depth (env DFKV_SERVER_URING_DEPTH)\n"
      "  --ram-flush-threads <n>   RAM tier flush thread count (env DFKV_RAM_FLUSH_THREADS)\n"
      "  --ram-tier-numa <n>  RAM tier NUMA node (env DFKV_RAM_TIER_NUMA)\n"
      "  --ram-tier-shards <n>  RAM tier lock shards, 1-64 (env DFKV_RAM_TIER_SHARDS; default 8)\n"
      "  --slab-table-sync-ms <n>  slab table sync cadence ms (env DFKV_SLAB_TABLE_SYNC_MS)\n"
      "  --slab-reclaim-ms <n>  slab background free-slot reclaimer cadence ms, 0 = off (env DFKV_SLAB_RECLAIM_MS)\n"
      "  --ram-reclaim-ms <n>   RAM tier background reclaimer cadence ms, 0 = off (env DFKV_RAM_RECLAIM_MS)\n"
      "  --log <level>        log level: INFO|DEBUG|WARN|ERROR (env DFKV_LOG)\n"
      "  --version, -V        print version and exit\n"
      "  --help, -h           print this help and exit\n"
      "Behavior flags are facades over their DFKV_* env twins (see docs/CONFIG.md\n"
      "for the full matrix); a flag always overrides a pre-set env value.\n"
      "--rdma-dev is passed directly to the RDMA server (param wins over env).\n",
      dfkv::Version());
    return help ? 0 : 1;
  }
  dfkv::Args args(argc, argv,
                  {"--dir", "--port", "--cap", "--rdma-port",
                   "--sgengine-tcp-port", "--sgengine-rdma-port",
                   "--rdma-dev", "--mds", "--group",
                   "--id", "--advertise", "--weight",
                   "--metrics-port", "--metrics-bind", "--store-engine",
                   "--slab-write", "--ram-tier", "--ram-tier-bytes",
                   "--slab-granularity", "--put-inflight-limit",
                   "--rdma-depth", "--rdma-numa", "--rdma-idle-ms",
                   "--rdma-op-timeout-ms", "--server-uring",
                   "--server-uring-depth", "--ram-flush-threads",
                   "--ram-tier-numa", "--ram-tier-shards", "--slab-table-sync-ms",
                   "--slab-reclaim-ms", "--ram-reclaim-ms", "--log",
                   "--max-msg"});
  std::string dir = args.Get("--dir", "/tmp/dfkv_node");
  std::string rdma_dev = args.Get("--rdma-dev", "");
  std::string mds = args.Get("--mds", "");
  std::string group = args.Get("--group", "default");
  std::string node_id = args.Get("--id", "");
  std::string advertise = args.Get("--advertise", "");
  std::string metrics_bind = args.Get("--metrics-bind", "");
  int weight = args.GetInt("--weight", 1);
  int port = args.GetInt("--port", 0);
  int rdma_port = args.GetInt("--rdma-port", -1);
  int sgengine_tcp_port = args.GetInt("--sgengine-tcp-port", -1);
  int sgengine_rdma_port = args.GetInt("--sgengine-rdma-port", -1);
  int metrics_port = args.GetInt("--metrics-port", -1);
  unsigned long long cap = args.GetU64("--cap", 1ull << 30);
  // Storage backend: "file" (default) or "slab". Propagated via env so the
  // DiskCacheGroup (constructed inside KvNodeServer) picks it up; --store-engine
  // wins over a pre-set DFKV_STORE_ENGINE.
  std::string store_engine = args.Get("--store-engine", "");
  if (!store_engine.empty()) ::setenv("DFKV_STORE_ENGINE", store_engine.c_str(), 1);
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
  // Slot quantum: a LAYOUT parameter -- changing it re-inits the store EMPTY
  // (meta mismatch). Facade for small-value deployments (default 1 MiB).
  std::string slab_gran = args.Get("--slab-granularity", "");
  if (!slab_gran.empty()) ::setenv("DFKV_SLAB_GRANULARITY", slab_gran.c_str(), 1);
  // PUT admission gate: cap concurrent disk writes; excess PUTs fast-fail with
  // kCacheFull (client: plain put-failure, no peer cooldown) instead of joining
  // a deep device queue. 0 = off (default).
  std::string put_limit = args.Get("--put-inflight-limit", "");
  if (!put_limit.empty()) ::setenv("DFKV_PUT_INFLIGHT_LIMIT", put_limit.c_str(), 1);
  // --- Remaining env-only knobs: same flag→env facade pattern. Each flag is
  // an empty-by-default string; when set it wins over a pre-set DFKV_* env via
  // setenv(overwrite=1). See docs/CONFIG.md for the full matrix.
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
  if (!ram_tier_numa.empty()) ::setenv("DFKV_RAM_TIER_NUMA", ram_tier_numa.c_str(), 1);
  // RAM tier lock shards (1-64; the tier halves it while a shard would hold
  // <32 extents, so small arenas degrade to fewer shards).
  std::string ram_tier_shards = args.Get("--ram-tier-shards", "");
  if (!ram_tier_shards.empty())
    ::setenv("DFKV_RAM_TIER_SHARDS", ram_tier_shards.c_str(), 1);
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
    cd::Record("sgengine_tcp_port", std::to_string(sgengine_tcp_port),
               src("--sgengine-tcp-port"));
    cd::Record("sgengine_rdma_port", std::to_string(sgengine_rdma_port),
               src("--sgengine-rdma-port"));
    cd::Record("sgengine_tcp_protocol",
               sgengine_tcp_port >= 0 ? "wire-v1/domain-sgengine-v1"
                                      : "(off)",
               src("--sgengine-tcp-port"));
    cd::Record("sgengine_rdma_protocol",
               sgengine_rdma_port >= 0 ? "rdma-v1/domain-sgengine-v1"
                                       : "(off)",
               src("--sgengine-rdma-port"));
    cd::Record("mds", mds.empty() ? "(none)" : mds, src("--mds"));
    cd::Record("group", group, src("--group"));
    cd::Record("id", node_id.empty() ? "(auto)" : node_id, src("--id"));
    cd::Record("advertise", advertise.empty() ? "(auto)" : advertise, src("--advertise"));
    cd::Record("weight", std::to_string(weight), src("--weight"));
    cd::Record("metrics_port", std::to_string(metrics_port), src("--metrics-port"));
    cd::Record("metrics_bind", metrics_bind.empty() ? "(any)" : metrics_bind, src("--metrics-bind"));
  }
  std::signal(SIGINT, OnSig);
  std::signal(SIGTERM, OnSig);

  // split --dir on commas into one entry per NVMe SSD
  std::vector<std::string> dirs;
  for (size_t i = 0, j; i <= dir.size(); i = j + 1) {
    j = dir.find(',', i);
    if (j == std::string::npos) j = dir.size();
    if (j > i) dirs.push_back(dir.substr(i, j - i));
  }
  if (dirs.empty()) dirs.push_back(dir);

  KvNodeServer srv(dirs, cap);
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

  std::unique_ptr<dfkv::compat::SgEngineTcpFrontend> sgengine_tcp;
  if (sgengine_tcp_port >= 0) {
    sgengine_tcp =
        std::make_unique<dfkv::compat::SgEngineTcpFrontend>(srv);
    sgengine_tcp->set_max_request_payload(
        dfkv::wire_limits::MaxRequestPayload());
    if (sgengine_tcp->Start(sgengine_tcp_port) != Status::kOk) {
      std::fprintf(stderr,
                   "failed to start SGEngine compatibility TCP port %d\n",
                   sgengine_tcp_port);
      srv.Stop();
      return 1;
    }
    std::printf("SGENGINE_TCP_PORT %d\n", sgengine_tcp->port());
    std::fflush(stdout);
    DFKV_LOG_INFO("SGEngine v1 compatibility TCP listening on port " +
                  std::to_string(sgengine_tcp->port()));
  }

  // Optional Prometheus /metrics endpoint (declared here, started after the RDMA
  // server below so its render callback can fold in the RDMA server counters).
  std::unique_ptr<dfkv::MetricsHttpServer> mhttp;
  DFKV_LOG_INFO("dfkv_server listening on port " + std::to_string(srv.port()) + ", dir=" + dir);

  // Announce the transport build mode loudly so a TCP-only binary (built without
  // -DDFKV_WITH_RDMA=ON) can never be mistaken for an RDMA one at deploy time.
#ifdef DFKV_WITH_RDMA
  DFKV_LOG_INFO("dfkv build: transport=RDMA (libibverbs) + TCP fallback");
#else
  DFKV_LOG_INFO("dfkv build: transport=TCP-only (built WITHOUT -DDFKV_WITH_RDMA=ON)");
  if (rdma_port >= 0 || sgengine_rdma_port >= 0) {
    DFKV_LOG_ERROR(
        "--rdma-port/--sgengine-rdma-port require an RDMA-enabled binary; "
        "rebuild with -DDFKV_WITH_RDMA=ON (needs libibverbs-dev)");
    srv.Stop();
    return 1;
  }
  if (!rdma_dev.empty())
    DFKV_LOG_WARN("--rdma-dev ignored: this binary has no RDMA support");
#endif

#ifdef DFKV_WITH_RDMA
  // --max-msg is the hard payload ceiling for both protocols. V2 leases slots
  // from one process-wide receive segment, sized independently by
  // DFKV_RDMA_RECV_SEGMENT_SIZE. Automatic v1 fallback retains the legacy
  // per-connection depth * (ValueHeader + max_msg) registered-buffer geometry;
  // clients with an explicit smaller declaration retain that smaller v1 cap.
  const unsigned long long max_msg = args.GetU64("--max-msg", 64ull << 20);
  std::unique_ptr<dfkv::RdmaServer> rsrv;
  if (rdma_port >= 0) {
    rsrv = std::make_unique<dfkv::RdmaServer>(
        [&srv](uint8_t op, const dfkv::BlockKey& key, uint64_t off,
               uint64_t len, const char* pl, uint64_t pll, std::string* out) {
          return srv.ProcessRequestForKey(op, key, off, len, pl, pll, out);
        },
        /*max_msg=*/max_msg, rdma_dev);
    rsrv->set_range_handler(  // server-side direct GET: disk -> registered dbuf -> RDMA scatter
        [&srv](const dfkv::BlockKey& key, uint64_t off, uint64_t len,
               char* io_buf, size_t cap, const char** out_data,
               size_t* out_len) {
          return srv.RangeDirectForKey(key, off, len, io_buf, cap, out_data,
                                       out_len);
        });
    rsrv->set_cache_direct_handler(  // server-side direct PUT: RDMA dbuf -> O_DIRECT write
        [&srv](const dfkv::BlockKey& key, char* data, size_t len, size_t cap) {
          return srv.CacheDirectForKey(key, data, len, cap);
        });
    // Async-GET hooks (used only when built -DDFKV_WITH_URING and started with
    // DFKV_SERVER_URING=1; otherwise the serve loop ignores these and uses the
    // synchronous range_handler above verbatim).
    rsrv->set_range_prep_handler(
        [&srv](const dfkv::BlockKey& key, uint64_t off, uint64_t len,
               size_t cap, dfkv::RdmaServer::RangePrepResult* out) {
          dfkv::KVStore::RangePrep p;
          Status st = srv.RangeDirectPrepForKey(key, off, len, cap, &p,
                                                &out->flight);
          if (st == Status::kOk) {
            out->fd = p.fd;
            out->aligned_off = p.aligned_off;
            out->aligned_len = p.aligned_len;
            out->head = p.head;
            out->payload_len = p.payload_len;
            // Slab slot hold (0 on the file engine). Dropping this token was a
            // real leak: the serve loop can only release what it was handed,
            // so every uring GET left its slot pinned+inflight forever.
            out->release_token = p.token;
          }
          return st;
        });
    rsrv->set_range_release_handler(
        [&srv](uint64_t tok) { srv.RangePrepRelease(tok); });
    rsrv->set_range_complete_handler(
        [&srv](bool ok, size_t bytes_read, double elapsed_sec, uint64_t flight,
               const char* data) {
          srv.RangeDirectComplete(ok, bytes_read, elapsed_sec, flight, data);
        });
    rsrv->set_range_flight_abort_handler(
        [&srv](uint64_t flight) { srv.RangeFlightAbort(flight); });
    // RAM hot tier (P3 B5-3): when enabled, register the arena as a pool MR and
    // wire the zero-copy serve hooks so a RAM hit is scatter-sent straight from
    // the arena (no copy into the connection buffer, no disk). No-op if off.
    if (srv.ram_enabled()) {
      rsrv->RegisterMemory(srv.ram_arena(), srv.ram_arena_bytes());
      rsrv->set_ram_range_handler(
          [&srv](const dfkv::BlockKey& key, uint64_t off, uint64_t len,
                 const char** out_ptr, size_t* out_len, uint64_t* out_token) {
            return srv.RamRangePrepForKey(key, off, len, out_ptr, out_len,
                                          out_token);
          });
      rsrv->set_ram_release_handler([&srv](uint64_t tok) { srv.RamRelease(tok); });
      DFKV_LOG_INFO("dfkv_server RAM hot tier: RDMA zero-copy serve enabled (arena " +
                    std::to_string(srv.ram_arena_bytes()) + " bytes)");
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
    // Force-resolve the RDMA depth/uring knobs so their defaults land in the
    // config dump (they are otherwise read lazily on the serve path, after Emit).
    (void)rsrv->PipelineDepth();
    (void)rsrv->UseUringPath();
  }

  std::unique_ptr<dfkv::compat::SgEngineRdmaFrontend> sgengine_rdma;
  if (sgengine_rdma_port >= 0) {
    sgengine_rdma =
        std::make_unique<dfkv::compat::SgEngineRdmaFrontend>(
            srv, max_msg, rdma_dev);
    if (sgengine_rdma->Start(sgengine_rdma_port) != Status::kOk) {
      DFKV_LOG_ERROR(
          "SGEngine compatibility RDMA listener failed to start; refusing "
          "readiness");
      if (rsrv) rsrv->Stop();
      srv.Stop();
      return 1;
    }
    std::printf("SGENGINE_RDMA_PORT %d\n", sgengine_rdma->port());
    std::fflush(stdout);
    DFKV_LOG_INFO(
        "SGEngine v1 compatibility RDMA listening (TCP bootstrap) on port " +
        std::to_string(sgengine_rdma->port()) +
        (rdma_dev.empty() ? "" : ", dev=" + rdma_dev));
    (void)sgengine_rdma->PipelineDepth();
    (void)sgengine_rdma->UseUringPath();
  }
#endif

  // Start /metrics now that the (optional) RDMA server exists: the render
  // callback combines the cache-node metrics with the RDMA server counters.
  // ready_flag flips once EVERY startup stage below has finished (RDMA
  // listener + MR anchor are already up at this point; MDS registration is
  // still ahead) — /readyz answers 503 until then so rolling upgrades wait
  // for a servable node, not merely an open metrics port.
  auto ready_flag = std::make_shared<std::atomic<bool>>(false);
  if (metrics_port >= 0) {
    mhttp = std::make_unique<dfkv::MetricsHttpServer>([&] {
      std::string s = srv.MetricsText();
      if (sgengine_tcp) s += sgengine_tcp->MetricsText();
#ifdef DFKV_WITH_RDMA
      if (rsrv) s += rsrv->MetricsText();
      if (sgengine_rdma) s += sgengine_rdma->MetricsText();
#endif
      return s;
    });
    mhttp->set_ready_check([ready_flag] { return ready_flag->load(std::memory_order_acquire); });
    if (mhttp->Start(metrics_port, metrics_bind) == Status::kOk)
      DFKV_LOG_INFO("dfkv_server /metrics on port " + std::to_string(mhttp->port()));
    else
      DFKV_LOG_WARN("dfkv_server /metrics failed to start on port " + std::to_string(metrics_port));
  }

  std::unique_ptr<dfkv::MdsRegistrar> registrar;
  if (!mds.empty()) {
    if (node_id.empty() || advertise.empty()) {
      std::fprintf(stderr, "--mds requires --id and --advertise <ip:port>\n");
      srv.Stop();
      return 1;
    }
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
      if (srv.engine_name() == "slab" && !srv.write_mode().empty())
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
    registrar = std::make_unique<dfkv::MdsRegistrar>(std::move(eps), group, self);
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
    registrar->Start();
    DFKV_LOG_INFO("dfkv_server registered with MDS group=" + group + " id=" + node_id +
                  " advertise=" + advertise);
  }

  dfkv::config_dump::Emit("server");
  ready_flag->store(true, std::memory_order_release);
  DFKV_LOG_INFO("dfkv_server ready (all startup stages complete)");

  while (!g_stop) { struct timespec ts{0, 50 * 1000 * 1000}; nanosleep(&ts, nullptr); }
  DFKV_LOG_INFO("dfkv_server shutting down");
  if (mhttp) mhttp->Stop();
  if (registrar) registrar->Stop();
#ifdef DFKV_WITH_RDMA
  if (rsrv) rsrv->Stop();
  if (sgengine_rdma) sgengine_rdma->Stop();
#endif
  if (sgengine_tcp) sgengine_tcp->Stop();
  srv.Stop();
  return 0;
}
