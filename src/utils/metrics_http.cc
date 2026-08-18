#include "utils/metrics_http.h"

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <string>

#include "common/config_dump.h"
#include "utils/log.h"
#include "utils/net_util.h"
#include "utils/thread_name.h"

namespace dfkv {

namespace {
// Bounded uint64 env knob: empty => fallback; non-numeric or 0 (unless the
// knob documents 0 as "off") => warn + fallback; above hard_max => warn +
// clamp. Same parse policy as the cache TCP listener knobs (kv_node_server.cc).
uint64_t BoundedEnv(const char* name, uint64_t fallback, uint64_t hard_max,
                    bool allow_zero) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  if (*value < '0' || *value > '9') {
    DFKV_LOG_WARN(std::string("invalid ") + name + "='" + value + "'; using " +
                  std::to_string(fallback));
    return fallback;
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' ||
      (parsed == 0 && !allow_zero)) {
    DFKV_LOG_WARN(std::string("invalid ") + name + "='" + value + "'; using " +
                  std::to_string(fallback));
    return fallback;
  }
  if (parsed > hard_max) {
    DFKV_LOG_WARN(std::string(name) + " clamped to hard maximum " +
                  std::to_string(hard_max));
    return hard_max;
  }
  return parsed;
}
constexpr size_t kMaxHttpRequestHeadBytes = 32 * 1024;

bool ReadRequestHead(
    int fd, bool deadline_on,
    std::chrono::steady_clock::time_point deadline,
    std::string* request_line) {
  if (request_line == nullptr) return false;
  const timeval base_timeout{10, 0};
  std::string line;
  bool first_line = true;
  size_t total = 0;
  while (total < kMaxHttpRequestHeadBytes) {
    char c = 0;
    const bool read =
        deadline_on
            ? net::ReadAllWithin(fd, &c, 1, base_timeout, deadline)
            : ::recv(fd, &c, 1, 0) > 0;
    if (!read) return false;
    ++total;
    if (c != '\n') {
      line.push_back(c);
      continue;
    }
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (first_line) {
      if (line.empty()) return false;
      *request_line = line;
      first_line = false;
    } else if (line.empty()) {
      return true;
    }
    line.clear();
  }
  return false;
}
}  // namespace

MetricsHttpServer::~MetricsHttpServer() { Stop(); }

Status MetricsHttpServer::Start(int port, const std::string& bind_addr) {
  // Connection-lifecycle knobs are resolved at Start so a test/operator can
  // still adjust them per process boot (they are not hot-reloaded).
  first_req_ms_ = BoundedEnv("DFKV_METRICS_FIRST_REQ_MS", kDefaultFirstReqMs,
                             kHardFirstReqMs, /*allow_zero=*/true);
  max_conns_ = static_cast<size_t>(BoundedEnv(
      "DFKV_METRICS_MAX_CONNS", kDefaultMaxConns, kHardMaxConns,
      /*allow_zero=*/false));
  config_dump::RecordResolved("DFKV_METRICS_FIRST_REQ_MS",
                              std::to_string(first_req_ms_));
  config_dump::RecordResolved("DFKV_METRICS_MAX_CONNS",
                              std::to_string(max_conns_));
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) return Status::kIOError;
  int one = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(INADDR_ANY);
  if (!bind_addr.empty() &&
      ::inet_pton(AF_INET, bind_addr.c_str(), &sa.sin_addr) != 1) {
    ::close(listen_fd_); listen_fd_ = -1; return Status::kInvalid;  // bad bind addr
  }
  sa.sin_port = htons(static_cast<uint16_t>(port));
  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) != 0) {
    ::close(listen_fd_); listen_fd_ = -1; return Status::kIOError;
  }
  if (::listen(listen_fd_, 16) != 0) {
    ::close(listen_fd_); listen_fd_ = -1; return Status::kIOError;
  }
  socklen_t sl = sizeof(sa);
  ::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&sa), &sl);
  port_ = ntohs(sa.sin_port);
  running_ = true;
  accept_thread_ = std::thread([this] { NameThisThread("met-accept"); AcceptLoop(); });
  return Status::kOk;
}

void MetricsHttpServer::Stop() {
  if (!running_.exchange(false)) return;  // idempotent
  if (listen_fd_ >= 0) ::shutdown(listen_fd_, SHUT_RDWR);
  if (accept_thread_.joinable()) accept_thread_.join();
  if (listen_fd_ >= 0) { ::close(listen_fd_); listen_fd_ = -1; }
  std::vector<int> fds;
  std::vector<Conn> conns;
  {
    std::lock_guard<std::mutex> lk(conn_mu_);
    fds.assign(conn_fds_.begin(), conn_fds_.end());
    conns.swap(conns_);
  }
  for (int fd : fds) ::shutdown(fd, SHUT_RDWR);
  for (auto& c : conns) if (c.th.joinable()) c.th.join();
}

void MetricsHttpServer::ReapDoneLocked() {
  for (auto it = conns_.begin(); it != conns_.end();) {
    if (it->done->load(std::memory_order_acquire)) {
      if (it->th.joinable()) it->th.join();
      it = conns_.erase(it);
    } else {
      ++it;
    }
  }
}

size_t MetricsHttpServer::live_conn_count() {
  std::lock_guard<std::mutex> lk(conn_mu_);
  return conns_.size();
}

void MetricsHttpServer::AcceptLoop() {
  while (running_) {
    int fd = ::accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) { if (!running_) break; continue; }
    // First-request deadline anchors at ACCEPT time (DFKV_METRICS_FIRST_REQ_MS).
    const auto accepted_at = std::chrono::steady_clock::now();
    // A scrape is one short request/response; a silent peer must not pin the
    // handler thread (Handle's recv loop otherwise blocks indefinitely).
    timeval tv{10, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    std::lock_guard<std::mutex> lk(conn_mu_);
    if (!running_) { ::close(fd); break; }
    ReapDoneLocked();  // join finished scrape handlers (Connection: close => one per scrape)
    if (active_connections_.load(std::memory_order_relaxed) >= max_conns_) {
      // Oneshot, low-frequency port: past DFKV_METRICS_MAX_CONNS the accept is
      // closed IMMEDIATELY instead of occupying a handler thread forever.
      dropped_connections_.fetch_add(1, std::memory_order_relaxed);
      ::close(fd);
      continue;
    }
    // +1 exactly once here (under conn_mu_, so the cap check above is exact);
    // the handler thread's -1 below pairs it on every exit path (clean EOF,
    // recv error, deadline expiry, shutdown) with no double-count.
    active_connections_.fetch_add(1, std::memory_order_relaxed);
    conn_fds_.insert(fd);
    auto done = std::make_shared<std::atomic<bool>>(false);
    conns_.push_back({std::thread([this, fd, done, accepted_at] {
                        NameThisThread("met-conn");
                        Handle(fd, accepted_at);
                        active_connections_.fetch_sub(1, std::memory_order_relaxed);
                        { std::lock_guard<std::mutex> lk(conn_mu_); conn_fds_.erase(fd); }
                        ::close(fd);
                        done->store(true, std::memory_order_release);  // last act
                      }),
                      done});
  }
}

namespace {
// Build an HTTP/1.0 response (Connection: close — we recv to EOF on the client).
std::string Resp(const char* status_line, const char* ctype, const std::string& body) {
  return std::string("HTTP/1.0 ") + status_line + "\r\n" +
         "Content-Type: " + ctype + "\r\n" +
         "Content-Length: " + std::to_string(body.size()) + "\r\n" +
         "Connection: close\r\n\r\n" + body;
}
}  // namespace

void MetricsHttpServer::Handle(
    int fd, std::chrono::steady_clock::time_point accepted_at) {
  // Consume the complete bounded request-head section before replying. Closing
  // a socket with unread Prometheus headers is an abortive close on Linux: the
  // peer sees RST and may lose the tail of a large metrics body even after
  // WriteAll placed it in the local send buffer.
  //
  // The absolute deadline remains anchored at accept and now covers every
  // header byte, not just the request line. This preserves the slow-drip guard:
  // each recv uses min(10 s, deadline remaining), and 0 keeps the legacy
  // no-absolute-deadline behavior.
  const bool deadline_on = first_req_ms_ > 0;
  const auto deadline =
      accepted_at + std::chrono::milliseconds(first_req_ms_);
  std::string line;
  if (!ReadRequestHead(fd, deadline_on, deadline, &line)) return;
  // line == "GET /path HTTP/1.x"
  std::string path;
  {
    size_t s = line.find(' ');
    size_t e = (s == std::string::npos) ? std::string::npos : line.find(' ', s + 1);
    if (s != std::string::npos && e != std::string::npos) path = line.substr(s + 1, e - s - 1);
  }
  std::string out;
  if (path == "/metrics") {
    out = Resp("200 OK", "text/plain; version=0.0.4", render_ ? render_() : "");
  } else if (path == "/healthz") {
    // A registered predicate (e.g. MDS etcd reachability) gates the status; with
    // none set, keep the always-healthy 200 the endpoint had before.
    if (!health_ || health_())
      out = Resp("200 OK", "text/plain", "ok\n");
    else
      out = Resp("503 Service Unavailable", "text/plain", "unavailable\n");
  } else if (path == "/readyz") {
    // Readiness (serving-capable), not liveness: 503 while startup work
    // (arena pre-fault, MR anchor, listeners, MDS registration) is running.
    if (!ready_ || ready_())
      out = Resp("200 OK", "text/plain", "ready\n");
    else
      out = Resp("503 Service Unavailable", "text/plain", "starting\n");
  } else {
    out = Resp("404 Not Found", "text/plain", "not found\n");
  }
  if (!net::WriteAll(fd, out.data(), out.size())) return;
  ::shutdown(fd, SHUT_WR);
}

}  // namespace dfkv
