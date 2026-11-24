// Test helper shared by the client and transport tests.
#ifndef DFKV_TEST_SCRIPTED_TOPOLOGY_MDS_H_
#define DFKV_TEST_SCRIPTED_TOPOLOGY_MDS_H_

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "common/membership.h"
#include "common/status.h"
#include "transport/wire.h"
#include "utils/net_util.h"

namespace dfkv {

// A local MDS with a mutable topology response. Tests synchronize with the
// client's background poller through what the transport receives or through
// the number of requests this MDS has answered, so they never sleep or depend
// on lease timing.
class ScriptedTopologyMds {
 public:
  explicit ScriptedTopologyMds(std::vector<MemberInfo> members)
      : members_(std::move(members)) {
    lfd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ::bind(lfd_, reinterpret_cast<sockaddr*>(&sa), sizeof(sa));
    ::listen(lfd_, 8);
    socklen_t sl = sizeof(sa);
    ::getsockname(lfd_, reinterpret_cast<sockaddr*>(&sa), &sl);
    port_ = ntohs(sa.sin_port);
    thread_ = std::thread([this] { Serve(); });
  }

  ~ScriptedTopologyMds() {
    ::shutdown(lfd_, SHUT_RDWR);
    ::close(lfd_);
    if (thread_.joinable()) thread_.join();
  }

  std::string endpoint() const {
    return "127.0.0.1:" + std::to_string(port_);
  }

  void SetMembers(std::vector<MemberInfo> members) {
    std::lock_guard<std::mutex> lock(mu_);
    members_ = std::move(members);
  }

  // The client registry kListClients answers with (empty by default), and
  // the payloads of the client registrations received.
  void SetClients(std::vector<MemberInfo> clients) {
    std::lock_guard<std::mutex> lock(mu_);
    clients_ = std::move(clients);
  }
  std::vector<std::string> ClientRegistrations() const {
    std::lock_guard<std::mutex> lock(mu_);
    return registrations_;
  }

  // Answer every request with this status and no member list (an MDS that
  // withholds its list); kOk resumes normal answers.
  void SetStatus(Status status) {
    std::lock_guard<std::mutex> lock(mu_);
    status_ = status;
  }

  // Requests answered so far, and a wait for that count to reach `count`.
  size_t Answered() const {
    std::lock_guard<std::mutex> lock(mu_);
    return answered_;
  }
  bool WaitForAnswered(size_t count) {
    std::unique_lock<std::mutex> lock(mu_);
    return cv_.wait_for(lock, std::chrono::seconds(5),
                        [&] { return answered_ >= count; });
  }

  bool only_topology_requests() const {
    return only_topology_requests_.load();
  }

 private:
  void Serve() {
    for (;;) {
      const int fd = ::accept(lfd_, nullptr, nullptr);
      if (fd < 0) return;
      char prefix[kReqPrefix];
      ReqFields request{};
      bool ok = net::ReadAll(fd, prefix, sizeof(prefix)) &&
                DecodeReq(prefix, &request);
      std::string payload;
      if (ok && request.payload_len != 0) {
        payload.assign(request.payload_len, '\0');
        ok = net::ReadAll(fd, &payload[0], payload.size());
      }
      if (ok && (request.op == static_cast<uint8_t>(WireOp::kClientRegister) ||
                 request.op ==
                     static_cast<uint8_t>(WireOp::kClientHeartbeat))) {
        std::lock_guard<std::mutex> lock(mu_);
        registrations_.push_back(payload);
      }
      if (ok) {
        if (request.op != static_cast<uint8_t>(WireOp::kListTopology))
          only_topology_requests_.store(false);
        std::vector<MemberInfo> members;
        Status status = Status::kOk;
        {
          std::lock_guard<std::mutex> lock(mu_);
          members = request.op == static_cast<uint8_t>(WireOp::kListClients)
                        ? clients_
                        : members_;
          status = status_;
        }
        const std::string data =
            status == Status::kOk
                ? EncodeMembers(members, MembersViewEpoch(members))
                : std::string();
        char response[kRespPrefix];
        EncodeResp(response, status, data.size());
        ok = net::WriteAll(fd, response, sizeof(response)) &&
             (data.empty() ||
              net::WriteAll(fd, data.data(), data.size()));
      }
      (void)ok;
      ::close(fd);
      {
        std::lock_guard<std::mutex> lock(mu_);
        ++answered_;
      }
      cv_.notify_all();
    }
  }

  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::vector<MemberInfo> members_;
  std::vector<MemberInfo> clients_;
  std::vector<std::string> registrations_;
  Status status_ = Status::kOk;
  size_t answered_ = 0;
  std::atomic<bool> only_topology_requests_{true};
  int lfd_ = -1;
  int port_ = 0;
  std::thread thread_;
};

}  // namespace dfkv

#endif  // DFKV_TEST_SCRIPTED_TOPOLOGY_MDS_H_
