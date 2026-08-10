#include "mds/etcd_client.h"

#include <cstdlib>
#include <chrono>

#include "utils/base64.h"
#include "utils/json_min.h"

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
    "lease_grant", "lease_keepalive", "put", "range", "metrics_range"};

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
  if (!transported || response->status != 200)
    errors_[i].fetch_add(1, std::memory_order_relaxed);
  return transported && response->status == 200;
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

std::optional<RangeResult> EtcdClient::RangePrefix(
    const std::string& prefix) {
  return RangePrefixAs(prefix, Op::kRange);
}

std::optional<RangeResult> EtcdClient::RangePrefixForMetrics(
    const std::string& prefix) {
  return RangePrefixAs(prefix, Op::kMetricsRange);
}

}  // namespace dfkv
