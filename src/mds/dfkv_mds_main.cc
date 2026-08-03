#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <string>

#include <chrono>
#include "utils/args.h"
#include "mds/mds_server.h"
#include "utils/metrics_http.h"
#include "common/version.h"
#include "common/config_dump.h"

namespace {
volatile sig_atomic_t g_stop = 0;
void OnSig(int) { g_stop = 1; }  // async-signal-safe: just set a flag
}  // namespace

int main(int argc, char** argv) {
  if (dfkv::WantsVersion(argc, argv)) { std::printf("dfkv_mds %s\n", dfkv::Version()); return 0; }
  // No args or --help: print usage and exit, rather than starting with defaults.
  if (argc == 1 || dfkv::WantsHelp(argc, argv)) {
    bool help = dfkv::WantsHelp(argc, argv);
    std::FILE* out = help ? stdout : stderr;
    std::fprintf(out,
      "dfkv_mds %s — dfkv Membership Directory Service daemon (stateless; state in etcd)\n"
      "Usage: dfkv_mds --listen <port> [--etcd <host:port>] [options]\n\n"
      "  --listen <port>      TCP port to serve MDS requests on\n"
      "  --etcd <host:port>   etcd endpoint (default 127.0.0.1:2379)\n"
      "  --etcd-probe-ms <n>  etcd reachability probe window ms (default 30000; env DFKV_MDS_ETCD_PROBE_MS)\n"
      "  --metrics-port <p>   enable Prometheus /metrics (omit = off); --metrics-bind <addr>\n"
      "  --version, -V        print version and exit\n"
      "  --help, -h           print this help and exit\n",
      dfkv::Version());
    return help ? 0 : 1;
  }
  dfkv::Args args(argc, argv,
                  {"--etcd", "--listen", "--metrics-port", "--metrics-bind",
                   "--etcd-probe-ms"});
  std::string etcd = args.Get("--etcd", "127.0.0.1:2379");
  std::string metrics_bind = args.Get("--metrics-bind", "");
  int port = args.GetInt("--listen", 0);
  int metrics_port = args.GetInt("--metrics-port", -1);
  // etcd probe window: flag wins over a pre-set DFKV_MDS_ETCD_PROBE_MS.
  std::string etcd_probe_ms = args.Get("--etcd-probe-ms", "");
  if (!etcd_probe_ms.empty()) ::setenv("DFKV_MDS_ETCD_PROBE_MS", etcd_probe_ms.c_str(), 1);
  if (!args.ok()) {
    std::fprintf(stderr, "dfkv_mds: %s\n(run with --help for usage)\n",
                 args.error().c_str());
    return 2;
  }
  {
    namespace cd = dfkv::config_dump;
    auto src = [&](const char* flag) {
      return args.Has(flag) ? cd::Source::kFlag : cd::Source::kDefault;
    };
    cd::Record("etcd", etcd, src("--etcd"));
    cd::Record("listen", std::to_string(port), src("--listen"));
    cd::Record("metrics_port", std::to_string(metrics_port), src("--metrics-port"));
    cd::Record("metrics_bind", metrics_bind.empty() ? "(any)" : metrics_bind,
               src("--metrics-bind"));
  }
  dfkv::MdsServer srv(etcd);
  std::signal(SIGINT, OnSig);
  std::signal(SIGTERM, OnSig);
  if (srv.Start(port) != dfkv::Status::kOk) {
    std::fprintf(stderr, "dfkv_mds: failed to listen on %d\n", port);
    return 1;
  }
  // Fail loud on a bad --etcd: probe until etcd answers within a window, else
  // exit non-zero so the supervisor restarts (and the misconfig is visible)
  // instead of running "up" while every registration silently fails. Window is
  // env-tunable (tests use a short one); default 30 s.
  {
    int probe_ms = static_cast<int>(
        dfkv::config_dump::EnvI64("DFKV_MDS_ETCD_PROBE_MS", 30000));
    if (probe_ms < 0) probe_ms = 30000;  // negative => default (unchanged)
    bool up = false;
    auto t0 = std::chrono::steady_clock::now();
    for (;;) {
      if (srv.ProbeEtcd()) { up = true; break; }
      if (std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - t0).count() >= probe_ms) break;
      struct timespec ts{0, 200 * 1000 * 1000}; nanosleep(&ts, nullptr);
    }
    if (!up) {
      std::fprintf(stderr,
                   "dfkv_mds: etcd unreachable at '%s' after %dms — check --etcd "
                   "(host:port, NO http:// scheme)\n", etcd.c_str(), probe_ms);
      srv.Stop();
      return 1;
    }
  }
  std::printf("dfkv_mds listening on %d, etcd=%s\n", srv.port(), etcd.c_str());
  std::fflush(stdout);
  // Optional Prometheus /metrics on a dedicated port/thread. Absent => no listener.
  std::unique_ptr<dfkv::MetricsHttpServer> mhttp;
  if (metrics_port >= 0) {
    mhttp = std::make_unique<dfkv::MetricsHttpServer>([&srv] { return srv.MetricsText(); });
    // Liveness and readiness are deliberately DIFFERENT predicates:
    // - /healthz is left unset (always 200): it reflects only this process
    //   being alive. A live etcd probe here turns any second-long etcd blip
    //   into kubelet restarting ALL MDS replicas at once — a CrashLoop that
    //   outlasts the blip and further buries etcd under re-registrations.
    // - /readyz keeps the etcd dependency check (a running MDS cannot serve
    //   membership without etcd and must be pulled from routing), but through
    //   a TTL-debounced probe (DFKV_MDS_PROBE_CACHE_MS, default 2500 ms) so
    //   the kubelet scrape cadence can't amplify into etcd read load during
    //   an ongoing incident; a not-ready replica re-probes at most once per TTL.
    mhttp->set_ready_check([&srv] { return srv.ProbeEtcdCached(); });
    if (mhttp->Start(metrics_port, metrics_bind) == dfkv::Status::kOk)
      std::printf("dfkv_mds /metrics on port %d\n", mhttp->port());
    std::fflush(stdout);
  }
  dfkv::config_dump::Emit("mds");
  // poll flag from normal context (matches dfkv_server_main.cc)
  while (!g_stop) { struct timespec ts{0, 50 * 1000 * 1000}; nanosleep(&ts, nullptr); }
  if (mhttp) mhttp->Stop();
  srv.Stop();  // called from the main thread, not the signal handler
  return 0;
}
