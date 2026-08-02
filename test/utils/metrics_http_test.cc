// TDD — tiny embedded HTTP /metrics responder.
#include "utils/metrics_http.h"

#include <gtest/gtest.h>

#include <atomic>
#include <string>

#include "utils/net_util.h"

using namespace dfkv;  // NOLINT

namespace {
// Minimal HTTP/1.0 client: send one request, read the whole response (server
// closes the connection after the body, so recv to EOF).
std::string HttpGet(int port, const std::string& path) {
  int fd = net::Dial("127.0.0.1:" + std::to_string(port), 1000, 2000);
  if (fd < 0) return "";
  std::string req = "GET " + path + " HTTP/1.0\r\n\r\n";
  if (!net::WriteAll(fd, req.data(), req.size())) { ::close(fd); return ""; }
  std::string resp;
  char buf[4096];
  for (;;) {
    ssize_t r = ::recv(fd, buf, sizeof(buf), 0);
    if (r <= 0) break;
    resp.append(buf, static_cast<size_t>(r));
  }
  ::close(fd);
  return resp;
}
}  // namespace

TEST(MetricsHttp, ServesMetricsHealthzAnd404) {
  std::atomic<int> renders{0};
  MetricsHttpServer srv([&renders] {
    renders.fetch_add(1);
    return std::string("dfkv_x 1\n");
  });
  ASSERT_EQ(srv.Start(0), Status::kOk);
  int port = srv.port();
  ASSERT_GT(port, 0);

  std::string m = HttpGet(port, "/metrics");
  EXPECT_NE(m.find("200"), std::string::npos) << m;
  EXPECT_NE(m.find("dfkv_x 1"), std::string::npos) << m;
  EXPECT_NE(m.find("text/plain"), std::string::npos) << m;
  EXPECT_GE(renders.load(), 1);

  std::string h = HttpGet(port, "/healthz");
  EXPECT_NE(h.find("200"), std::string::npos) << h;
  EXPECT_NE(h.find("ok"), std::string::npos) << h;

  std::string nf = HttpGet(port, "/nope");
  EXPECT_NE(nf.find("404"), std::string::npos) << nf;

  srv.Stop();
}

TEST(MetricsHttp, ReapsHandlerThreadsAcrossScrapes) {
  // Prometheus scrapes are Connection: close — one connection per scrape. Without
  // reaping, the handler-thread list grew unbounded. After many sequential
  // scrapes the live (unreaped) count must stay small, not ~N.
  MetricsHttpServer srv([] { return std::string("dfkv_x 1\n"); });
  ASSERT_EQ(srv.Start(0), Status::kOk);
  int port = srv.port();
  for (int i = 0; i < 60; ++i) {
    std::string m = HttpGet(port, "/metrics");
    EXPECT_NE(m.find("dfkv_x 1"), std::string::npos);
  }
  // accept-time reaping joins finished handlers; allow a tiny in-flight slack.
  EXPECT_LE(srv.live_conn_count(), 3u) << "handler threads not reaped";
  srv.Stop();
}

TEST(MetricsHttp, BindAddrRestrictsAndRejectsBad) {
  // loopback bind still serves a loopback client
  MetricsHttpServer srv([] { return std::string("dfkv_x 1\n"); });
  ASSERT_EQ(srv.Start(0, "127.0.0.1"), Status::kOk);
  std::string m = HttpGet(srv.port(), "/metrics");
  EXPECT_NE(m.find("dfkv_x 1"), std::string::npos) << m;
  srv.Stop();
  // a malformed bind address fails cleanly (no listener)
  MetricsHttpServer bad([] { return std::string(""); });
  EXPECT_EQ(bad.Start(0, "not.an.ip"), Status::kInvalid);
}

TEST(MetricsHttp, StopIsIdempotent) {
  MetricsHttpServer srv([] { return std::string("x 1\n"); });
  ASSERT_EQ(srv.Start(0), Status::kOk);
  srv.Stop();
  srv.Stop();  // must not crash / hang
}

TEST(MetricsHttp, HealthCheckPredicateGates503) {
  std::atomic<bool> healthy{true};
  MetricsHttpServer srv([] { return std::string("dfkv_x 1\n"); });
  srv.set_health_check([&] { return healthy.load(); });
  ASSERT_EQ(srv.Start(0), Status::kOk);
  int port = srv.port();

  std::string ok = HttpGet(port, "/healthz");
  EXPECT_NE(ok.find("200"), std::string::npos) << ok;
  EXPECT_NE(ok.find("ok"), std::string::npos) << ok;

  healthy.store(false);
  std::string bad = HttpGet(port, "/healthz");
  EXPECT_NE(bad.find("503"), std::string::npos) << bad;
  EXPECT_NE(bad.find("unavailable"), std::string::npos) << bad;

  srv.Stop();
}

TEST(MetricsHttp, ReadyCheckRecoversAcrossLiveDependencyLoss) {
  std::atomic<bool> ready{false};
  MetricsHttpServer srv([] { return std::string("dfkv_x 1\n"); });
  srv.set_ready_check([&] { return ready.load(); });
  ASSERT_EQ(srv.Start(0), Status::kOk);
  int port = srv.port();

  // Startup window: alive (healthz 200) but NOT ready (readyz 503) — rolling
  // upgrades gate on readiness, not on the port being open.
  std::string h = HttpGet(port, "/healthz");
  EXPECT_NE(h.find("200"), std::string::npos) << h;
  std::string starting = HttpGet(port, "/readyz");
  EXPECT_NE(starting.find("503"), std::string::npos) << starting;
  EXPECT_NE(starting.find("starting"), std::string::npos) << starting;

  ready.store(true);
  std::string ok = HttpGet(port, "/readyz");
  EXPECT_NE(ok.find("200"), std::string::npos) << ok;
  EXPECT_NE(ok.find("ready"), std::string::npos) << ok;

  // Runtime dependency loss must remove scheduler readiness, and restoring the
  // dependency must recover in place: 200 -> 503 -> 200.
  ready.store(false);
  std::string lost = HttpGet(port, "/readyz");
  EXPECT_NE(lost.find("503"), std::string::npos) << lost;
  ready.store(true);
  std::string recovered = HttpGet(port, "/readyz");
  EXPECT_NE(recovered.find("200"), std::string::npos) << recovered;
  // Unset predicate mirrors the always-200 behavior.
  MetricsHttpServer plain([] { return std::string(); });
  ASSERT_EQ(plain.Start(0), Status::kOk);
  std::string dflt = HttpGet(plain.port(), "/readyz");
  EXPECT_NE(dflt.find("200"), std::string::npos) << dflt;
  plain.Stop();
  srv.Stop();
}

TEST(MetricsHttp, StorageHealthDynamicallyRemovesReadiness) {
  std::atomic<bool> startup{false};
  std::atomic<bool> first_registration{false};
  std::atomic<bool> storage_healthy{true};
  MetricsHttpServer srv([] { return std::string("dfkv_x 1\n"); });
  srv.set_health_check([&] { return storage_healthy.load(); });
  srv.set_ready_check([&] {
    return startup.load() && first_registration.load() &&
           storage_healthy.load();
  });
  ASSERT_EQ(srv.Start(0), Status::kOk);

  EXPECT_NE(HttpGet(srv.port(), "/readyz").find("503"), std::string::npos);
  startup.store(true);
  EXPECT_NE(HttpGet(srv.port(), "/readyz").find("503"), std::string::npos);
  first_registration.store(true);
  EXPECT_NE(HttpGet(srv.port(), "/readyz").find("200"), std::string::npos);

  // A terminal local store failure affects both endpoints dynamically.
  storage_healthy.store(false);
  EXPECT_NE(HttpGet(srv.port(), "/healthz").find("503"), std::string::npos);
  EXPECT_NE(HttpGet(srv.port(), "/readyz").find("503"), std::string::npos);

  storage_healthy.store(true);
  EXPECT_NE(HttpGet(srv.port(), "/readyz").find("200"), std::string::npos);
  srv.Stop();
}
