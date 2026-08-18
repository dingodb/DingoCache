// TDD — tiny embedded HTTP /metrics responder.
#include "utils/metrics_http.h"

#include <gtest/gtest.h>

#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <string>
#include <thread>

#include "utils/net_util.h"

using namespace dfkv;  // NOLINT

namespace {
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) {
      had_old_ = true;
      old_ = old;
    }
    ::setenv(name, value, 1);
  }
  ~ScopedEnv() {
    if (had_old_)
      ::setenv(name_.c_str(), old_.c_str(), 1);
    else
      ::unsetenv(name_.c_str());
  }

 private:
  std::string name_;
  std::string old_;
  bool had_old_ = false;
};

bool WaitFor(const std::function<bool()>& predicate, int timeout_ms) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return predicate();
}

int Dial(int port) {
  return net::Dial("127.0.0.1:" + std::to_string(port), 1000, 2000);
}

struct HttpReadResult {
  std::string response;
  bool clean_eof = false;
  int terminal_errno = 0;
};

HttpReadResult ReadResponse(int fd) {
  HttpReadResult result;
  char buffer[4096];
  for (;;) {
    const ssize_t received = ::recv(fd, buffer, sizeof(buffer), 0);
    if (received > 0) {
      result.response.append(buffer, static_cast<size_t>(received));
      continue;
    }
    if (received == 0) {
      result.clean_eof = true;
    } else {
      result.terminal_errno = errno;
    }
    return result;
  }
}

HttpReadResult HttpRequest(int port, const std::string& request,
                           int read_delay_ms = 0) {
  HttpReadResult result;
  const int fd = Dial(port);
  if (fd < 0) {
    result.terminal_errno = errno;
    return result;
  }
  const int receive_bytes = 4096;
  ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &receive_bytes,
               sizeof(receive_bytes));
  if (!net::WriteAll(fd, request.data(), request.size())) {
    result.terminal_errno = errno;
    ::close(fd);
    return result;
  }
  if (read_delay_ms > 0)
    std::this_thread::sleep_for(std::chrono::milliseconds(read_delay_ms));
  result = ReadResponse(fd);
  ::close(fd);
  return result;
}

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

TEST(MetricsHttp, PrometheusHeadersLargeBodyEndsWithCleanEof) {
  const std::string body(256 * 1024, 'm');
  MetricsHttpServer srv([&body] { return body; });
  ASSERT_EQ(srv.Start(0), Status::kOk);
  const std::string request =
      "GET /metrics HTTP/1.1\r\n"
      "Host: 127.0.0.1\r\n"
      "User-Agent: Prometheus/3.5.0\r\n"
      "Accept: application/openmetrics-text;version=1.0.0;q=0.8,*/*;q=0.1\r\n"
      "Accept-Encoding: gzip\r\n"
      "X-Prometheus-Scrape-Timeout-Seconds: 10\r\n"
      "\r\n";
  const HttpReadResult result = HttpRequest(srv.port(), request, 100);

  EXPECT_TRUE(result.clean_eof)
      << "terminal errno=" << result.terminal_errno
      << " bytes=" << result.response.size();
  EXPECT_NE(result.terminal_errno, ECONNRESET);
  EXPECT_NE(result.response.find("HTTP/1.0 200 OK"), std::string::npos);
  EXPECT_NE(result.response.find("Content-Length: " +
                                 std::to_string(body.size())),
            std::string::npos);
  const size_t separator = result.response.find("\r\n\r\n");
  ASSERT_NE(separator, std::string::npos);
  const std::string actual_body = result.response.substr(separator + 4);
  ASSERT_EQ(actual_body.size(), body.size());
  EXPECT_EQ(std::memcmp(actual_body.data(), body.data(), body.size()), 0);
  srv.Stop();
}

TEST(MetricsHttp, WaitsForHeaderTerminatorBeforeResponding) {
  MetricsHttpServer srv([] { return std::string("dfkv_x 1\n"); });
  ASSERT_EQ(srv.Start(0), Status::kOk);
  const int fd = Dial(srv.port());
  ASSERT_GE(fd, 0);
  const std::string partial =
      "GET /metrics HTTP/1.1\r\n"
      "Host: 127.0.0.1\r\n";
  ASSERT_TRUE(net::WriteAll(fd, partial.data(), partial.size()));
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  char byte = 0;
  errno = 0;
  EXPECT_EQ(::recv(fd, &byte, 1, MSG_DONTWAIT), -1);
  EXPECT_TRUE(errno == EAGAIN || errno == EWOULDBLOCK) << errno;

  ASSERT_TRUE(net::WriteAll(fd, "\r\n", 2));
  const HttpReadResult result = ReadResponse(fd);
  EXPECT_TRUE(result.clean_eof);
  EXPECT_NE(result.response.find("HTTP/1.0 200 OK"), std::string::npos);
  EXPECT_NE(result.response.find("dfkv_x 1"), std::string::npos);
  ::close(fd);
  srv.Stop();
}

TEST(MetricsHttp, SlowDripHeadersUseAcceptTimeAbsoluteDeadline) {
  ScopedEnv first_req("DFKV_METRICS_FIRST_REQ_MS", "250");
  MetricsHttpServer srv([] { return std::string("dfkv_x 1\n"); });
  ASSERT_EQ(srv.Start(0), Status::kOk);
  const int fd = Dial(srv.port());
  ASSERT_GE(fd, 0);
  const std::string prefix =
      "GET /metrics HTTP/1.1\r\n"
      "X-Drip: ";
  ASSERT_TRUE(net::WriteAll(fd, prefix.data(), prefix.size()));
  for (char c : std::string("slow")) {
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    if (::send(fd, &c, 1, MSG_NOSIGNAL) <= 0) break;
  }
  EXPECT_TRUE(WaitFor([&] { return srv.ActiveConnections() == 0; }, 3000));
  EXPECT_EQ(srv.DroppedConnections(), 0u);
  ::close(fd);
  EXPECT_NE(HttpGet(srv.port(), "/metrics").find("dfkv_x 1"),
            std::string::npos);
  srv.Stop();
}

TEST(MetricsHttp, OversizeUnterminatedHeadersCloseWithinBound) {
  ScopedEnv first_req("DFKV_METRICS_FIRST_REQ_MS", "5000");
  MetricsHttpServer srv([] { return std::string("dfkv_x 1\n"); });
  ASSERT_EQ(srv.Start(0), Status::kOk);
  const int fd = Dial(srv.port());
  ASSERT_GE(fd, 0);
  const std::string request =
      "GET /metrics HTTP/1.1\r\nX-Oversize: " +
      std::string(33 * 1024, 'h');
  net::WriteAll(fd, request.data(), request.size());
  EXPECT_TRUE(WaitFor([&] { return srv.ActiveConnections() == 0; }, 3000));
  ::close(fd);
  EXPECT_NE(HttpGet(srv.port(), "/metrics").find("dfkv_x 1"),
            std::string::npos);
  srv.Stop();
}

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

// DFKV_METRICS_MAX_CONNS bounds the oneshot port: excess accepts are closed
// IMMEDIATELY (no handler thread), the drop is counted, and the capped
// connections — plus later scrapes after they drain — keep working.
TEST(MetricsHttp, MaxConnsClosesExcessAcceptsImmediately) {
  ScopedEnv max_conns("DFKV_METRICS_MAX_CONNS", "2");
  MetricsHttpServer srv([] { return std::string("dfkv_x 1\n"); });
  ASSERT_EQ(srv.Start(0), Status::kOk);
  const int port = srv.port();

  const int c1 = Dial(port);
  const int c2 = Dial(port);
  ASSERT_GE(c1, 0);
  ASSERT_GE(c2, 0);
  ASSERT_TRUE(WaitFor([&] { return srv.ActiveConnections() == 2; }, 2000));

  const int c3 = Dial(port);
  ASSERT_GE(c3, 0);
  ASSERT_TRUE(WaitFor([&] { return srv.DroppedConnections() == 1; }, 2000));
  EXPECT_EQ(srv.ActiveConnections(), 2u);
  // The dropped peer sees an instant EOF — the accept was closed, not queued.
  char byte = 0;
  EXPECT_LE(::recv(c3, &byte, 1, 0), 0);
  ::close(c3);

  // The two admitted connections still serve normally (request completes well
  // inside the default first-request deadline).
  const std::string req = "GET /healthz HTTP/1.0\r\n\r\n";
  ASSERT_TRUE(net::WriteAll(c1, req.data(), req.size()));
  char buf[128];
  ssize_t r = ::recv(c1, buf, sizeof(buf), 0);
  ASSERT_GT(r, 0);
  EXPECT_NE(std::string(buf, static_cast<size_t>(r)).find("200"),
            std::string::npos);
  ::close(c1);
  ::close(c2);
  // After the surge drains to zero, a fresh scrape is admitted again.
  ASSERT_TRUE(WaitFor([&] { return srv.ActiveConnections() == 0; }, 2000));
  EXPECT_NE(HttpGet(port, "/metrics").find("dfkv_x 1"), std::string::npos);
  srv.Stop();
}

// SO_RCVTIMEO is a per-SYSCALL budget: a drip feeder sending one byte per
// (timeout - epsilon) never timed out and pinned a handler thread forever.
// DFKV_METRICS_FIRST_REQ_MS gives the request line an absolute deadline
// anchored at accept, so the drip connection is closed while the 10s
// per-syscall base alone would never have fired (drip interval < 10s).
TEST(MetricsHttp, SlowDripRequestLineIsClosedByAbsoluteDeadline) {
  ScopedEnv first_req("DFKV_METRICS_FIRST_REQ_MS", "300");
  MetricsHttpServer srv([] { return std::string("dfkv_x 1\n"); });
  ASSERT_EQ(srv.Start(0), Status::kOk);

  const int fd = Dial(srv.port());
  ASSERT_GE(fd, 0);
  ASSERT_TRUE(WaitFor([&] { return srv.ActiveConnections() == 1; }, 2000));
  // Drip a partial request line one byte at a time below the 10s base.
  const char* drip = "GE";
  ::send(fd, drip, 1, MSG_NOSIGNAL);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  ::send(fd, drip + 1, 1, MSG_NOSIGNAL);
  EXPECT_TRUE(WaitFor([&] { return srv.ActiveConnections() == 0; }, 3000))
      << "drip connection outlived its first-request deadline";
  EXPECT_EQ(srv.DroppedConnections(), 0u);  // timeout close is NOT a cap drop
  ::close(fd);
  // Normal scrapes are unaffected on the same server.
  EXPECT_NE(HttpGet(srv.port(), "/metrics").find("dfkv_x 1"),
            std::string::npos);
  srv.Stop();
}

// The active-connection count is +1 at admit and −1 exactly once on EVERY
// exit path — normal EOF after a scrape, abrupt close mid-request, deadline
// expiry, and Stop() interrupting an idle handler. No leak, no double-dec.
TEST(MetricsHttp, ConnCountBalancesAcrossEveryClosePath) {
  ScopedEnv first_req("DFKV_METRICS_FIRST_REQ_MS", "300");
  ScopedEnv max_conns("DFKV_METRICS_MAX_CONNS", "8");
  MetricsHttpServer srv([] { return std::string("dfkv_x 1\n"); });
  ASSERT_EQ(srv.Start(0), Status::kOk);
  const int port = srv.port();

  // path 1: clean one-shot scrape, closed by the server after the response.
  EXPECT_NE(HttpGet(port, "/metrics").find("dfkv_x 1"), std::string::npos);
  // path 2: peer connects then vanishes WITHOUT a request (handler's first
  // recv sees EOF).
  const int gone = Dial(port);
  ASSERT_GE(gone, 0);
  ::close(gone);
  // path 3: drip feeder closed by the absolute deadline (server-side close,
  // NOT a client close — the count must still unwind to zero).
  const int drip = Dial(port);
  ASSERT_GE(drip, 0);
  ::send(drip, "G", 1, MSG_NOSIGNAL);
  ASSERT_TRUE(WaitFor([&] { return srv.ActiveConnections() == 1; }, 3000))
      << "expected only the drip conn to remain; got "
      << srv.ActiveConnections();
  ASSERT_TRUE(WaitFor([&] { return srv.ActiveConnections() == 0; }, 3000))
      << "deadline close leaked the connection count";
  ::close(drip);

  // path 4: Stop() interrupts an idle pre-request handler mid-recv.
  const int idle = Dial(port);
  ASSERT_GE(idle, 0);
  ASSERT_TRUE(WaitFor([&] { return srv.ActiveConnections() == 1; }, 2000));
  srv.Stop();
  EXPECT_EQ(srv.ActiveConnections(), 0u);
  ::close(idle);
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
