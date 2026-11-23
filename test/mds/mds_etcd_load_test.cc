// Not correctness tests: measurements of the load the membership plane puts
// on etcd, against a real etcd (DFKV_TEST_ETCD=host:port). Set
// DFKV_MDS_LOAD_SECONDS (for example 180) to run them; they take that long.
//
// Three MDS share the etcd; cache nodes heartbeat every 10 s through a
// registrar that knows all three, with stats that change on every beat as a
// serving node's do; cache clients poll the member list every 3 s. Reported
// per member (or per client) and minute: etcd requests by operation as the
// MDS made them, and how far the etcd revision moved (each Put, Delete and
// lease-expiry deletion moves it; it is what grows etcd's history).
#include "mds/mds_member_poller.h"
#include "mds/mds_registrar.h"
#include "mds/mds_server.h"
#include "mds/etcd_client.h"
#include "common/membership.h"
#include "utils/http_client.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace dfkv;  // NOLINT

namespace {

const char* EtcdEp() { return std::getenv("DFKV_TEST_ETCD"); }

int LoadSeconds() {
  const char* v = std::getenv("DFKV_MDS_LOAD_SECONDS");
  return v ? std::atoi(v) : 0;
}

int EnvInt(const char* name, int fallback) {
  const char* v = std::getenv(name);
  return v && *v ? std::atoi(v) : fallback;
}

// Sum of `dfkv_mds_etcd_requests_total{op="<op>"}` over the servers.
std::map<std::string, double> EtcdRequests(
    const std::vector<std::unique_ptr<MdsServer>>& servers) {
  std::map<std::string, double> out;
  for (const auto& server : servers) {
    const std::string text = server->MetricsText();
    size_t at = 0;
    const std::string name = "dfkv_mds_etcd_requests_total{op=\"";
    while ((at = text.find(name, at)) != std::string::npos) {
      at += name.size();
      const size_t quote = text.find('"', at);
      const std::string op = text.substr(at, quote - at);
      const size_t value = text.find(' ', quote) + 1;
      out[op] += std::strtod(text.c_str() + value, nullptr);
      at = quote;
    }
  }
  return out;
}

int64_t Revision(const std::string& ep) {
  TcpHttpTransport http(ep, 2000);
  EtcdClient etcd(&http);
  const auto r = etcd.RangePrefix("/dfkv/v1/_load_revision_probe/");
  return r ? r->revision : -1;
}

struct Cluster {
  std::vector<std::unique_ptr<MdsServer>> servers;
  std::vector<std::string> endpoints;
  explicit Cluster(const std::string& etcd, int count) {
    for (int i = 0; i < count; ++i) {
      servers.push_back(std::make_unique<MdsServer>(etcd));
      if (servers.back()->Start(0) != Status::kOk) return;
      endpoints.push_back("127.0.0.1:" +
                          std::to_string(servers.back()->port()));
    }
  }
  ~Cluster() {
    for (auto& s : servers) s->Stop();
  }
};

void Report(const char* what, const std::map<std::string, double>& before,
            const std::map<std::string, double>& after, double per_minute) {
  std::printf("MDS_LOAD %s", what);
  for (const auto& [op, value] : after) {
    const auto b = before.find(op);
    const double delta = value - (b == before.end() ? 0 : b->second);
    std::printf(" %s=%.2f", op.c_str(), delta / per_minute);
  }
  std::printf("\n");
}

}  // namespace

TEST(MdsEtcdLoad, HeartbeatsPerMemberAndMinute) {
  const int seconds = LoadSeconds();
  if (!EtcdEp() || seconds <= 0)
    GTEST_SKIP() << "set DFKV_TEST_ETCD and DFKV_MDS_LOAD_SECONDS to run";
  const int members = EnvInt("DFKV_MDS_LOAD_MEMBERS", 10);
  const int mds = EnvInt("DFKV_MDS_LOAD_MDS", 3);
  Cluster cluster(EtcdEp(), mds);
  ASSERT_EQ(static_cast<int>(cluster.endpoints.size()), mds);
  const std::string group =
      "load-hb-" + std::to_string(::getpid()) + "-" + std::to_string(mds);
  std::vector<std::unique_ptr<MdsRegistrar>> registrars;
  std::atomic<uint64_t> beat{0};
  for (int i = 0; i < members; ++i) {
    MemberInfo self{"node-" + std::to_string(i), "10.7.0." + std::to_string(i),
                    28000, 1};
    auto r = std::make_unique<MdsRegistrar>(cluster.endpoints, group, self);
    r->set_stats_provider([&beat] {
      MemberStats s;
      s.capacity_bytes = 1ull << 40;
      s.puts_total = beat.fetch_add(1);  // changes on every beat
      s.used_bytes = s.puts_total << 20;
      return s;
    });
    registrars.push_back(std::move(r));
  }
  for (auto& r : registrars) r->Start();
  // Let the first registrations land before counting.
  std::this_thread::sleep_for(std::chrono::seconds(5));
  const auto before = EtcdRequests(cluster.servers);
  const int64_t rev_before = Revision(EtcdEp());
  std::this_thread::sleep_for(std::chrono::seconds(seconds));
  const auto after = EtcdRequests(cluster.servers);
  const int64_t rev_after = Revision(EtcdEp());
  const double member_minutes = members * seconds / 60.0;
  std::printf("MDS_LOAD heartbeats members=%d mds=%d seconds=%d\n", members,
              mds, seconds);
  Report("etcd_requests_per_member_minute", before, after, member_minutes);
  std::printf("MDS_LOAD revisions_per_member_minute=%.2f\n",
              (rev_after - rev_before) / member_minutes);
  for (auto& r : registrars) {
    EXPECT_TRUE(r->registered());
    r->Stop();
  }
}

TEST(MdsEtcdLoad, MemberListPollsPerClientAndMinute) {
  const int seconds = LoadSeconds();
  if (!EtcdEp() || seconds <= 0)
    GTEST_SKIP() << "set DFKV_TEST_ETCD and DFKV_MDS_LOAD_SECONDS to run";
  const int clients = EnvInt("DFKV_MDS_LOAD_CLIENTS", 50);
  const int mds = EnvInt("DFKV_MDS_LOAD_MDS", 3);
  Cluster cluster(EtcdEp(), mds);
  ASSERT_EQ(static_cast<int>(cluster.endpoints.size()), mds);
  const std::string group = "load-list-" + std::to_string(::getpid());
  std::vector<std::unique_ptr<MdsRegistrar>> registrars;
  for (int i = 0; i < 8; ++i) {
    MemberInfo self{"node-" + std::to_string(i), "10.8.0." + std::to_string(i),
                    28000, 1};
    registrars.push_back(
        std::make_unique<MdsRegistrar>(cluster.endpoints, group, self));
    registrars.back()->Start();
  }
  std::this_thread::sleep_for(std::chrono::seconds(3));
  std::vector<std::unique_ptr<MdsMemberPoller>> pollers;
  std::atomic<uint64_t> views{0};
  for (int i = 0; i < clients; ++i) {
    pollers.push_back(std::make_unique<MdsMemberPoller>(
        cluster.endpoints, group,
        [&views](const std::vector<MemberInfo>&) { views.fetch_add(1); }));
  }
  for (auto& p : pollers) p->Start();
  std::this_thread::sleep_for(std::chrono::seconds(3));
  const auto before = EtcdRequests(cluster.servers);
  std::this_thread::sleep_for(std::chrono::seconds(seconds));
  const auto after = EtcdRequests(cluster.servers);
  const double client_minutes = clients * seconds / 60.0;
  std::printf("MDS_LOAD polls clients=%d mds=%d seconds=%d\n", clients, mds,
              seconds);
  Report("etcd_requests_per_client_minute", before, after, client_minutes);
  for (auto& p : pollers) p->Stop();
  for (auto& r : registrars) r->Stop();
  EXPECT_GT(views.load(), 0u);
}

// How long a member that stops heartbeating stays listed: its lease's TTL
// (30 s) after its last renewal, whatever the write path does.
TEST(MdsEtcdLoad, DeadMemberLeavesTheListWithinTheLeaseTtl) {
  if (!EtcdEp() || LoadSeconds() <= 0)
    GTEST_SKIP() << "set DFKV_TEST_ETCD and DFKV_MDS_LOAD_SECONDS to run";
  Cluster cluster(EtcdEp(), 3);
  ASSERT_EQ(cluster.endpoints.size(), 3u);
  const std::string group = "load-expiry-" + std::to_string(::getpid());
  MemberInfo self{"node-dead", "10.9.0.1", 28000, 1};
  MdsRegistrar registrar(cluster.endpoints, group, self);
  bool listed = false;
  registrar.Start();
  const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(90);
  while (!registrar.registered() && std::chrono::steady_clock::now() < give_up)
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  // A few heartbeats, so the record is in its steady state.
  std::this_thread::sleep_for(std::chrono::seconds(25));
  registrar.Stop();
  const auto stopped = std::chrono::steady_clock::now();
  for (;;) {
    TcpHttpTransport http(EtcdEp(), 2000);
    EtcdClient etcd(&http);
    const auto r =
        etcd.RangePrefix("/dfkv/v1/groups/" + group + "/members/");
    listed = r && !r->kvs.empty();
    if (!listed || std::chrono::steady_clock::now() > give_up) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  const double seconds = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - stopped)
                             .count();
  std::printf("MDS_LOAD dead_member_listed_for_seconds=%.1f\n", seconds);
  EXPECT_FALSE(listed);
  EXPECT_LE(seconds, MdsServer::kTtlSeconds + 2);
}
