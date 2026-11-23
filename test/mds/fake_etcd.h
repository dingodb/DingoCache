// An in-process stand-in for the etcd JSON gateway, for MDS tests that must
// run without an etcd.
#ifndef DFKV_TEST_MDS_FAKE_ETCD_H_
#define DFKV_TEST_MDS_FAKE_ETCD_H_

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>

#include "utils/base64.h"
#include "utils/json_min.h"
#include "utils/net_util.h"

namespace dfkv {

// The gateway calls the MDS makes, over HTTP/1.1 with Connection: close.
class FakeEtcd {
 public:
  FakeEtcd() {
    lfd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ::bind(lfd_, reinterpret_cast<sockaddr*>(&sa), sizeof(sa));
    ::listen(lfd_, 64);
    socklen_t sl = sizeof(sa);
    ::getsockname(lfd_, reinterpret_cast<sockaddr*>(&sa), &sl);
    port_ = ntohs(sa.sin_port);
    th_ = std::thread([this] { Serve(); });
  }
  ~FakeEtcd() {
    ::shutdown(lfd_, SHUT_RDWR);
    ::close(lfd_);
    if (th_.joinable()) th_.join();
  }
  std::string ep() const { return "127.0.0.1:" + std::to_string(port_); }

  // Out of space: Put and LeaseGrant are refused; KeepAlive and Range answer.
  void RejectWrites(bool on) {
    std::lock_guard<std::mutex> lk(mu_);
    reject_writes_ = on;
  }
  // Every lease runs out: the leases and the keys attached to them are gone.
  void ExpireAllLeases() {
    std::lock_guard<std::mutex> lk(mu_);
    leases_.clear();
    kvs_.clear();
  }
  // Store a value under a key as it is, without a lease (a damaged record).
  void SetRaw(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lk(mu_);
    kvs_[key] = {Base64Encode(value), 0};
  }
  // The lease holding this one key runs out.
  void ExpireKey(const std::string& key) {
    std::lock_guard<std::mutex> lk(mu_);
    const auto found = kvs_.find(key);
    if (found == kvs_.end()) return;
    leases_.erase(found->second.second);
    kvs_.erase(found);
  }

 private:
  struct Reply {
    int status;
    std::string body;
  };

  Reply Handle(const std::string& path, const std::string& body) {
    static const char kNoSpace[] =
        "{\"error\":\"etcdserver: mvcc: database space exceeded\",\"code\":8}";
    static const char kHeader[] = "\"header\":{\"revision\":\"1\"}";
    JsonValue req;
    if (!JsonParser::Parse(body, &req)) return {400, "{}"};
    std::lock_guard<std::mutex> lk(mu_);
    if (path == "/v3/lease/grant") {
      if (reject_writes_) return {429, kNoSpace};
      const int64_t id = next_lease_++;
      leases_.insert(id);
      return {200, std::string("{") + kHeader + ",\"ID\":\"" +
                       std::to_string(id) + "\",\"TTL\":\"30\"}"};
    }
    if (path == "/v3/lease/keepalive") {
      // An expired lease is answered with 200 and no TTL.
      const std::string id = req.get_str("ID");
      const bool alive = leases_.count(std::strtoll(id.c_str(), nullptr, 10));
      return {200, "{\"result\":{\"ID\":\"" + id + "\"" +
                       (alive ? ",\"TTL\":\"30\"" : "") + "}}"};
    }
    if (path == "/v3/kv/put") {
      if (reject_writes_) return {429, kNoSpace};
      const int64_t lease =
          std::strtoll(req.get_str("lease").c_str(), nullptr, 10);
      if (lease != 0 && !leases_.count(lease))
        return {404, "{\"error\":\"etcdserver: requested lease not found\","
                     "\"code\":5}"};
      std::string key;
      if (!Base64Decode(req.get_str("key"), &key)) return {400, "{}"};
      kvs_[key] = {req.get_str("value"), lease};
      return {200, std::string("{") + kHeader + "}"};
    }
    if (path == "/v3/kv/deleterange") {
      std::string key;
      if (!Base64Decode(req.get_str("key"), &key)) return {400, "{}"};
      kvs_.erase(key);
      return {200, std::string("{") + kHeader + "}"};
    }
    if (path == "/v3/kv/range") {
      // Without range_end, the one key named.
      std::string begin, end;
      const bool single = req.get_str("range_end").empty();
      if (!Base64Decode(req.get_str("key"), &begin) ||
          (!single && !Base64Decode(req.get_str("range_end"), &end)))
        return {400, "{}"};
      std::string kvs;
      for (auto it = kvs_.lower_bound(begin);
           it != kvs_.end() && (single ? it->first == begin : it->first < end);
           ++it) {
        if (!kvs.empty()) kvs += ",";
        kvs += "{\"key\":\"" + Base64Encode(it->first) + "\",\"value\":\"" +
               it->second.first + "\",\"lease\":\"" +
               std::to_string(it->second.second) + "\"}";
      }
      return {200, std::string("{") + kHeader +
                       (kvs.empty() ? "" : ",\"kvs\":[" + kvs + "]") + "}"};
    }
    return {404, "{}"};
  }

  void Serve() {
    for (;;) {
      const int fd = ::accept(lfd_, nullptr, nullptr);
      if (fd < 0) return;
      std::string raw;
      char chunk[4096];
      size_t head_end = std::string::npos;
      size_t body_len = 0;
      for (;;) {
        if (head_end == std::string::npos) {
          const size_t at = raw.find("\r\n\r\n");
          if (at != std::string::npos) {
            head_end = at + 4;
            const size_t cl = raw.find("Content-Length: ");
            if (cl != std::string::npos && cl < at)
              body_len = std::strtoul(raw.c_str() + cl + 16, nullptr, 10);
          }
        }
        if (head_end != std::string::npos &&
            raw.size() >= head_end + body_len)
          break;
        const ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) break;
        raw.append(chunk, static_cast<size_t>(n));
      }
      if (head_end != std::string::npos && raw.size() >= head_end + body_len) {
        const size_t path_at = raw.find(' ') + 1;
        const std::string path =
            raw.substr(path_at, raw.find(' ', path_at) - path_at);
        const Reply reply = Handle(path, raw.substr(head_end, body_len));
        const std::string out =
            "HTTP/1.1 " + std::to_string(reply.status) +
            " X\r\nContent-Type: application/json\r\nContent-Length: " +
            std::to_string(reply.body.size()) + "\r\n\r\n" + reply.body;
        net::WriteAll(fd, out.data(), out.size());
      }
      ::close(fd);
    }
  }

  int lfd_ = -1, port_ = 0;
  std::thread th_;
  std::mutex mu_;
  bool reject_writes_ = false;
  int64_t next_lease_ = 1000;
  std::set<int64_t> leases_;
  // key -> (base64 value as received, lease)
  std::map<std::string, std::pair<std::string, int64_t>> kvs_;
};

}  // namespace dfkv

#endif  // DFKV_TEST_MDS_FAKE_ETCD_H_
