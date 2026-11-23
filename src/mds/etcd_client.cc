#include "mds/etcd_client.h"

#include <cstdlib>
#include <chrono>

#include "utils/base64.h"
#include "utils/json_min.h"
#include "utils/log.h"

namespace dfkv {

namespace {
// etcd range over a prefix uses range_end = key with last byte incremented.
std::string PrefixRangeEnd(const std::string& prefix) {
  std::string end = prefix;
  for (size_t i = end.size(); i-- > 0;) {
    if (uint8_t(end[i]) < 0xFF) { end[i] = char(uint8_t(end[i]) + 1); end.resize(i + 1); return end; }
  }
  return std::string("\0", 1);  // all 0xFF => use "\0" for scan-to-end (empty would mean single-key get)
}
}  // namespace

constexpr const char* kOpNames[] = {
    "lease_grant", "lease_keepalive", "put", "range", "metrics_range", "get"};

bool EtcdClient::Post(Op op, const std::string& path, const std::string& body,
                      HttpResponse* response) {
  const auto start = std::chrono::steady_clock::now();
  const bool transported = http_->Post(path, body, response);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  const size_t i = static_cast<size_t>(op);
  latency_[i].Observe(seconds);
  requests_[i].fetch_add(1, std::memory_order_relaxed);
  const bool ok = transported && response->status == 200;
  if (!ok) {
    errors_[i].fetch_add(1, std::memory_order_relaxed);
    NoteFailure(op, transported, *response);
  }
  if (op == Op::kLeaseGrant || op == Op::kLeaseKeepAlive || op == Op::kPut) {
    last_write_attempt_ms_.store(NowMs(), std::memory_order_relaxed);
    if (ok) {
      write_failing_since_ms_.store(0, std::memory_order_relaxed);
    } else {
      uint64_t none = 0;
      write_failing_since_ms_.compare_exchange_strong(
          none, NowMs(), std::memory_order_relaxed);
    }
  }
  return ok;
}

uint64_t EtcdClient::NowMs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

void EtcdClient::NoteFailure(Op op, bool transported,
                             const HttpResponse& response) {
  const size_t i = static_cast<size_t>(op);
  const uint64_t now = NowMs();
  uint64_t unlogged = 0;
  {
    std::lock_guard<std::mutex> lk(failure_log_mu_);
    if (failure_logged_ms_[i] != 0 &&
        now - failure_logged_ms_[i] < kFailureLogEveryMs) {
      ++failures_unlogged_[i];
      return;
    }
    failure_logged_ms_[i] = now;
    unlogged = failures_unlogged_[i];
    failures_unlogged_[i] = 0;
  }
  std::string what;
  if (!transported) {
    what = "no answer";
  } else {
    // What etcd said, on one line and bounded.
    std::string body = response.body.substr(0, 256);
    for (char& c : body)
      if (c == '\n' || c == '\r') c = ' ';
    what = "HTTP " + std::to_string(response.status) + ": " + body;
  }
  DFKV_LOG_WARN("mds: etcd " + std::string(kOpNames[i]) + " failed: " + what +
                (unlogged ? " (" + std::to_string(unlogged) +
                                " more failure(s) of this operation since "
                                "the last such line)"
                          : std::string()));
}

std::string EtcdClient::MetricsText() const {
  constexpr const char* latency_name =
      "dfkv_mds_etcd_request_duration_seconds";
  std::string out =
      "# HELP dfkv_mds_etcd_requests_total etcd HTTP requests by operation\n"
      "# TYPE dfkv_mds_etcd_requests_total counter\n"
      "# HELP dfkv_mds_etcd_request_errors_total etcd HTTP transport or status errors by operation\n"
      "# TYPE dfkv_mds_etcd_request_errors_total counter\n"
      "# HELP dfkv_mds_etcd_request_duration_seconds etcd HTTP request latency seconds\n"
      "# TYPE dfkv_mds_etcd_request_duration_seconds histogram\n";
  for (size_t i = 0; i < static_cast<size_t>(Op::kCount); ++i) {
    const std::string labels = "op=\"" + std::string(kOpNames[i]) + "\"";
    out += "dfkv_mds_etcd_requests_total{" + labels + "} " +
           std::to_string(requests_[i].load(std::memory_order_relaxed)) + "\n";
    out += "dfkv_mds_etcd_request_errors_total{" + labels + "} " +
           std::to_string(errors_[i].load(std::memory_order_relaxed)) + "\n";
    out += latency_[i].RenderBody(latency_name, labels);
  }
  // Writes failing for how long: from the first failed write after the last
  // success, while writes are still being attempted. An MDS with nothing left
  // to write (no member heartbeats here) loses nothing, so after the idle
  // window without a write attempt it reads 0 again.
  const uint64_t since = write_failing_since_ms_.load(std::memory_order_relaxed);
  const uint64_t now = NowMs();
  const uint64_t last_attempt =
      last_write_attempt_ms_.load(std::memory_order_relaxed);
  const bool attempting =
      last_attempt != 0 && now - last_attempt <= write_idle_window_ms_;
  out +=
      "# HELP dfkv_mds_etcd_write_failing_seconds How long etcd has been "
      "refusing or failing this MDS's writes (lease grant, keepalive, put); 0 "
      "while they succeed, or when none was attempted for the idle window\n"
      "# TYPE dfkv_mds_etcd_write_failing_seconds gauge\n"
      "dfkv_mds_etcd_write_failing_seconds " +
      std::to_string(since != 0 && attempting && now > since
                         ? (now - since) / 1000
                         : 0) +
      "\n";
  return out;
}

std::optional<int64_t> EtcdClient::LeaseGrant(int ttl_seconds) {
  HttpResponse r;
  std::string body = "{\"TTL\":\"" + std::to_string(ttl_seconds) + "\",\"ID\":\"0\"}";
  if (!Post(Op::kLeaseGrant, "/v3/lease/grant", body, &r)) return std::nullopt;
  JsonValue root;
  if (!JsonParser::Parse(r.body, &root)) return std::nullopt;
  std::string id = root.get_str("ID");
  if (id.empty()) return std::nullopt;
  return std::optional<int64_t>(std::strtoll(id.c_str(), nullptr, 10));
}

bool EtcdClient::LeaseKeepAlive(int64_t lease_id) {
  HttpResponse r;
  std::string body = "{\"ID\":\"" + std::to_string(lease_id) + "\"}";
  if (!Post(Op::kLeaseKeepAlive, "/v3/lease/keepalive", body, &r)) return false;
  JsonValue root;
  if (!JsonParser::Parse(r.body, &root)) return false;
  const JsonValue* res = root.find("result");
  std::string ttl = res ? res->get_str("TTL") : root.get_str("TTL");
  return !ttl.empty() && ttl != "0";
}

bool EtcdClient::Put(const std::string& key, const std::string& value, int64_t lease_id) {
  HttpResponse r;
  std::string body = "{\"key\":\"" + Base64Encode(key) + "\",\"value\":\"" +
                     Base64Encode(value) + "\",\"lease\":\"" + std::to_string(lease_id) + "\"}";
  return Post(Op::kPut, "/v3/kv/put", body, &r);
}

std::optional<RangeResult> EtcdClient::RangePrefixAs(
    const std::string& prefix, Op op) {
  HttpResponse r;
  std::string end = PrefixRangeEnd(prefix);
  std::string body = "{\"key\":\"" + Base64Encode(prefix) +
                     "\",\"range_end\":\"" + Base64Encode(end) + "\"}";
  if (!Post(op, "/v3/kv/range", body, &r)) return std::nullopt;
  JsonValue root;
  if (!JsonParser::Parse(r.body, &root)) return std::nullopt;
  RangeResult out;
  const JsonValue* hdr = root.find("header");
  if (hdr) {
    out.revision = std::strtoll(
        hdr->get_str("revision", "0").c_str(), nullptr, 10);
  }
  const JsonValue* kvs = root.find("kvs");
  if (kvs && kvs->type == JsonValue::kArray) {
    for (const auto& kv : kvs->arr) {
      std::string k, v;
      if (!Base64Decode(kv.get_str("key"), &k)) return std::nullopt;
      if (!Base64Decode(kv.get_str("value"), &v)) return std::nullopt;
      out.kvs.emplace_back(std::move(k), std::move(v));
    }
  }
  return out;
}

std::optional<KeyState> EtcdClient::Get(const std::string& key) {
  HttpResponse r;
  const std::string body = "{\"key\":\"" + Base64Encode(key) + "\"}";
  if (!Post(Op::kGet, "/v3/kv/range", body, &r)) return std::nullopt;
  JsonValue root;
  if (!JsonParser::Parse(r.body, &root)) return std::nullopt;
  KeyState out;
  const JsonValue* kvs = root.find("kvs");
  if (kvs && kvs->type == JsonValue::kArray && !kvs->arr.empty()) {
    const JsonValue& kv = kvs->arr.front();
    if (!Base64Decode(kv.get_str("value"), &out.value)) return std::nullopt;
    out.exists = true;
    out.lease = std::strtoll(kv.get_str("lease", "0").c_str(), nullptr, 10);
  }
  return out;
}

std::optional<RangeResult> EtcdClient::RangePrefix(
    const std::string& prefix) {
  return RangePrefixAs(prefix, Op::kRange);
}

std::optional<RangeResult> EtcdClient::RangePrefixForMetrics(
    const std::string& prefix) {
  return RangePrefixAs(prefix, Op::kMetricsRange);
}

}  // namespace dfkv
