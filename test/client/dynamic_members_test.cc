// TDD R12 — dynamic membership: KVClient.SetMembers() rebuilds the ring at
// runtime so adding a node re-routes new keys without recreating the client.
#include "client/kv_client.h"
#include "scripted_topology_mds.h"
#include "cache/kv_node_server.h"
#include "common/membership.h"
#include "mds/mds_member_poller.h"
#include "transport/rail_select.h"
#include "transport/wire.h"
#include "utils/net_util.h"

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {
std::string Hdr() { return "test/model"; }
struct Node { fs::path dir; std::unique_ptr<KvNodeServer> srv; std::string addr; };
std::unique_ptr<Node> Start(const std::string& tag) {
  auto n = std::make_unique<Node>();
  n->dir = fs::temp_directory_path() / ("dfkv_dyn_" + tag);
  fs::remove_all(n->dir); fs::create_directories(n->dir);
  n->srv = std::make_unique<KvNodeServer>(n->dir.string(), 1ull << 30);
  EXPECT_EQ(n->srv->Start(0), Status::kOk);
  n->addr = "127.0.0.1:" + std::to_string(n->srv->port());
  return n;
}

// Emulates an MDS predating kListTopology. It either returns the legacy
// unknown-op response (kInvalid) or drops that request, and records whether the
// client correctly limits fallback to the former.
class LegacyMembersMds {
 public:
  LegacyMembersMds(std::vector<MemberInfo> members, bool drop_topology)
      : members_(std::move(members)), drop_topology_(drop_topology) {
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

  ~LegacyMembersMds() {
    ::shutdown(lfd_, SHUT_RDWR);
    ::close(lfd_);
    if (thread_.joinable()) thread_.join();
  }

  std::string endpoint() const {
    return "127.0.0.1:" + std::to_string(port_);
  }

  std::vector<WireOp> requests() const {
    std::lock_guard<std::mutex> lock(mu_);
    return requests_;
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
      if (ok && request.payload_len != 0) {
        std::string payload(request.payload_len, '\0');
        ok = net::ReadAll(fd, &payload[0], payload.size());
      }
      if (!ok) {
        ::close(fd);
        continue;
      }

      const WireOp op = static_cast<WireOp>(request.op);
      {
        std::lock_guard<std::mutex> lock(mu_);
        requests_.push_back(op);
      }
      if (op == WireOp::kListTopology && drop_topology_) {
        ::close(fd);
        continue;
      }

      Status status = Status::kInvalid;
      std::string data;
      if (op == WireOp::kListMembers) {
        status = Status::kOk;
        data = EncodeMembers(members_, MembersEpoch(members_));
      }
      char response[kRespPrefix];
      EncodeResp(response, status, data.size());
      ok = net::WriteAll(fd, response, sizeof(response)) &&
           (data.empty() || net::WriteAll(fd, data.data(), data.size()));
      (void)ok;
      ::close(fd);
    }
  }

  const std::vector<MemberInfo> members_;
  const bool drop_topology_;
  mutable std::mutex mu_;
  std::vector<WireOp> requests_;
  int lfd_ = -1;
  int port_ = 0;
  std::thread thread_;
};
}  // namespace

TEST(DynamicMembers, AddingNodeReroutesNewKeys) {
  auto a = Start("a"); auto b = Start("b");
  KVClient c({{"a", a->addr}}, Hdr());  // start with only node a
  std::string v(64, 'v');
  for (int i = 0; i < 60; ++i)
    ASSERT_TRUE(c.Put("p1_" + std::to_string(i), v.data(), v.size()));
  EXPECT_EQ(b->srv->Count(), 0u);  // b not in the ring yet

  c.SetMembers(std::vector<std::pair<std::string,std::string>>{{"a", a->addr}, {"b", b->addr}});  // hot add node b
  for (int i = 0; i < 60; ++i)
    ASSERT_TRUE(c.Put("p2_" + std::to_string(i), v.data(), v.size()));
  EXPECT_GT(b->srv->Count(), 0u);  // some new keys now land on b

  // keys still readable (those routed to their current owner)
  std::string out(v.size(), '\0');
  EXPECT_TRUE(c.Get("p2_0", &out[0], out.size()));
  a->srv->Stop(); b->srv->Stop();
}

// AdoptRing()/MembershipDelta: a membership change must log WHICH node ids
// joined/left, so a scale-up or a node loss is obvious at a glance. The logger
// writes to stderr; capture it around each SetMembers. No servers/ops are
// needed — only the ring rebuild runs (the probe + MDS discovery threads stay
// off unless their env knobs are set), so nothing else pollutes the capture.
TEST(DynamicMembers, SetMembersLogsAddRemoveDelta) {
  using P = std::vector<std::pair<std::string, std::string>>;
  KVClient c(P{{"n1", "127.0.0.1:1"}, {"n2", "127.0.0.1:2"}}, Hdr());

  // add n3
  testing::internal::CaptureStderr();
  c.SetMembers(P{{"n1", "127.0.0.1:1"}, {"n2", "127.0.0.1:2"}, {"n3", "127.0.0.1:3"}});
  std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("3 member(s) +1 -0"), std::string::npos) << log;
  EXPECT_NE(log.find("added: n3"), std::string::npos) << log;
  EXPECT_EQ(log.find("removed:"), std::string::npos) << log;

  // remove n1
  testing::internal::CaptureStderr();
  c.SetMembers(P{{"n2", "127.0.0.1:2"}, {"n3", "127.0.0.1:3"}});
  log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("2 member(s) +0 -1"), std::string::npos) << log;
  EXPECT_NE(log.find("removed: n1"), std::string::npos) << log;

  // rolling replace: n2 out, n4 in
  testing::internal::CaptureStderr();
  c.SetMembers(P{{"n3", "127.0.0.1:3"}, {"n4", "127.0.0.1:4"}});
  log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("+1 -1"), std::string::npos) << log;
  EXPECT_NE(log.find("added: n4"), std::string::npos) << log;
  EXPECT_NE(log.find("removed: n2"), std::string::npos) << log;

  // same ids, only the address changed -> no id churn -> "(unchanged)"
  testing::internal::CaptureStderr();
  c.SetMembers(P{{"n3", "127.0.0.1:33"}, {"n4", "127.0.0.1:44"}});
  log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("(unchanged)"), std::string::npos) << log;

  // empty membership -> WARN (the "ring is empty, ops report ok=0" case)
  testing::internal::CaptureStderr();
  c.SetMembers(P{});
  log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("EMPTY membership"), std::string::npos) << log;
}

// Ring adoptions must feed the transport's topology hint: connection budgets
// scale with nodes x rails, so a growing ring has to reach OnTopologyHint
// both at construction (static member lists never re-adopt) and on every
// later SetMembers (0812-004: a fixed budget starved a 55-node ring).
namespace {
struct HintRecordingTransport : dfkv::Transport {
  void OnTopologyHint(size_t nodes) override {
    std::lock_guard<std::mutex> lock(mu);
    hints.push_back(nodes);
    cv.notify_all();
  }
  void OnPeerTopology(const PeerTopology& topology) override {
    topology_store.Update(topology);
    std::lock_guard<std::mutex> lock(mu);
    peer_topologies.push_back(topology);
    cv.notify_all();
  }
  void OnPeerIdentities(
      const std::vector<std::string>& live_peer_ids) override {
    std::lock_guard<std::mutex> lock(mu);
    peer_identity_sets.push_back(live_peer_ids);
    cv.notify_all();
  }
  bool WaitForPeerTopologies(size_t count) {
    std::unique_lock<std::mutex> lock(mu);
    return cv.wait_for(lock, std::chrono::seconds(5), [&] {
      return peer_topologies.size() >= count;
    });
  }
  bool WaitForPeerIdentitySets(size_t count) {
    std::unique_lock<std::mutex> lock(mu);
    return cv.wait_for(lock, std::chrono::seconds(5), [&] {
      return peer_identity_sets.size() >= count;
    });
  }
  bool WaitForHints(size_t count) {
    std::unique_lock<std::mutex> lock(mu);
    return cv.wait_for(lock, std::chrono::seconds(5),
                       [&] { return hints.size() >= count; });
  }
  std::vector<size_t> Hints() const {
    std::lock_guard<std::mutex> lock(mu);
    return hints;
  }
  std::vector<PeerTopology> PeerTopologies() const {
    std::lock_guard<std::mutex> lock(mu);
    return peer_topologies;
  }
  std::vector<std::vector<std::string>> PeerIdentitySets() const {
    std::lock_guard<std::mutex> lock(mu);
    return peer_identity_sets;
  }
  std::vector<std::string> CachePeers() const {
    std::lock_guard<std::mutex> lock(mu);
    return cache_peers;
  }
  std::shared_ptr<const rdma::PeerRailSnapshot> PeerSnapshot(
      const std::string& address) const {
    return topology_store.Snapshot(address);
  }
  uint64_t Publication(const std::string& address) const {
    return topology_store.Snapshot(address)->publication;
  }
  // What a pooled connection remembers of its peer; the transport drops the
  // connection once this is no longer current.
  bool Current(const std::string& address, const std::string& peer_id,
               uint64_t publication) const {
    return topology_store.IsCurrent(address, peer_id, publication);
  }
  mutable std::mutex mu;
  std::condition_variable cv;
  std::vector<size_t> hints;
  std::vector<PeerTopology> peer_topologies;
  std::vector<std::vector<std::string>> peer_identity_sets;
  std::vector<std::string> cache_peers;
  rdma::PeerTopologyStore topology_store{{"ib0"}, {{0}}, true};
  Status Cache(const std::string& peer, const dfkv::BlockKey&, const void*,
               size_t) override {
    // As RdmaTransport::Acquire: under explicit tiers a peer without a
    // published rail fails closed.
    if (!rdma::HasRail(topology_store.Snapshot(peer)->peer_healthy))
      return Status::kNoCompatibleRail;
    std::lock_guard<std::mutex> lock(mu);
    cache_peers.push_back(peer);
    return Status::kOk;
  }
  Status Range(const std::string&, const dfkv::BlockKey&, uint64_t, uint64_t,
               std::string*, uint64_t*) override {
    return Status::kNotFound;
  }
  Status Lookup(const std::string&, const dfkv::BlockKey&,
                uint64_t* value_len) override {
    if (value_len) *value_len = 0;
    return Status::kNotFound;
  }
  Status Exist(const std::string&, const dfkv::BlockKey&,
               bool* exist) override {
    if (exist) *exist = false;
    return Status::kOk;
  }
};
}  // namespace

TEST(DynamicMembers, AdoptionsFeedTransportTopologyHint) {
  using P = std::vector<std::pair<std::string, std::string>>;
  HintRecordingTransport t;
  KVClient c(P{{"n1", "127.0.0.1:1"}, {"n2", "127.0.0.1:2"}}, Hdr(), &t);
  // The constructor replays the initial adoption once the transport exists.
  ASSERT_EQ(t.hints.size(), 1u);
  EXPECT_EQ(t.hints[0], 2u);

  c.SetMembers(P{{"n1", "127.0.0.1:1"},
                 {"n2", "127.0.0.1:2"},
                 {"n3", "127.0.0.1:3"}});
  ASSERT_EQ(t.hints.size(), 2u);
  EXPECT_EQ(t.hints[1], 3u);

  // An empty adoption routes nowhere and must not hint a zero-sized ring.
  c.SetMembers(P{});
  EXPECT_EQ(t.hints.size(), 2u);
}

TEST(DynamicMembers, LegacyMdsFallbackAdoptsIncompleteTopology) {
  using P = std::vector<std::pair<std::string, std::string>>;
  const MemberInfo peer{"legacy-n1", "127.0.0.1", 28101, 1};
  LegacyMembersMds mds({peer}, false);
  HintRecordingTransport transport;
  KVClient client(P{}, Hdr(), &transport);
  client.StartMdsDiscovery({mds.endpoint()}, "legacy-topology-test", 60000);

  const bool got_topology = transport.WaitForPeerTopologies(1);
  const bool got_ring = client.WaitForRing(5000);
  client.StopMdsDiscovery();

  ASSERT_TRUE(got_topology);
  ASSERT_TRUE(got_ring);
  const std::vector<WireOp> requests = mds.requests();
  ASSERT_EQ(requests.size(), 2u);
  EXPECT_EQ(requests[0], WireOp::kListTopology);
  EXPECT_EQ(requests[1], WireOp::kListMembers);

  // Homogeneous transports still receive placement, while tier-aware
  // transports retain the incomplete signal and fail closed during selection.
  const std::vector<size_t> hints = transport.Hints();
  ASSERT_EQ(hints.size(), 1u);
  EXPECT_EQ(hints[0], 1u);
  const std::vector<PeerTopology> topologies = transport.PeerTopologies();
  ASSERT_EQ(topologies.size(), 1u);
  EXPECT_EQ(topologies[0].peer_addr, "127.0.0.1:28101");
  EXPECT_EQ(topologies[0].peer_id, "legacy-n1");
  EXPECT_TRUE(topologies[0].present);
  EXPECT_EQ(topologies[0].generation,
            MembersTopologyEpoch(std::vector<MemberInfo>{peer}));
  EXPECT_FALSE(topologies[0].complete);
  EXPECT_TRUE(topologies[0].rails.empty());
  EXPECT_EQ(transport.PeerIdentitySets(),
            (std::vector<std::vector<std::string>>{{"legacy-n1"}}));
}

TEST(DynamicMembers, TransportFailureDoesNotTriggerLegacyFallback) {
  const MemberInfo peer{"legacy-n1", "127.0.0.1", 28101, 1};
  LegacyMembersMds mds({peer}, true);
  bool adopted = false;
  MdsMemberPoller poller(
      {mds.endpoint()}, "legacy-topology-test",
      MdsMemberPoller::OnChange(
          [&](const std::vector<MemberInfo>&) { adopted = true; }),
      60000, 1000);

  EXPECT_FALSE(poller.PollOnce());
  EXPECT_FALSE(adopted);
  const std::vector<WireOp> requests = mds.requests();
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_EQ(requests[0], WireOp::kListTopology);
}

TEST(DynamicMembers, RailHealthChangePropagatesWithoutPlacementChurn) {
  using P = std::vector<std::pair<std::string, std::string>>;
  MemberInfo peer{"n1", "127.0.0.1", 28001, 1};
  peer.has_health = true;
  peer.health.ring_eligible = true;
  peer.health.ib_devices = {
      {"ib0", 4, 5, true}, {"ib1", 4, 5, true}};
  ScriptedTopologyMds mds({peer});
  HintRecordingTransport transport;
  KVClient client(P{}, Hdr(), &transport);
  client.StartMdsDiscovery({mds.endpoint()}, "topology-test", 10);

  ASSERT_TRUE(transport.WaitForPeerTopologies(1));
  const std::vector<PeerTopology> initial = transport.PeerTopologies();
  ASSERT_EQ(initial.size(), 1u);
  EXPECT_EQ(initial[0].peer_addr, "127.0.0.1:28001");
  EXPECT_EQ(initial[0].peer_id, "n1");
  EXPECT_TRUE(initial[0].present);
  EXPECT_TRUE(initial[0].complete);
  EXPECT_EQ(initial[0].generation,
            MembersTopologyEpoch(std::vector<MemberInfo>{peer}));
  ASSERT_EQ(initial[0].rails.size(), 2u);
  EXPECT_EQ(initial[0].rails[0].name, "ib0");
  EXPECT_TRUE(initial[0].rails[0].healthy);
  EXPECT_EQ(initial[0].rails[1].name, "ib1");
  EXPECT_TRUE(initial[0].rails[1].healthy);

  // This changes HLT1 and the topology generation, but not placement:
  // n1 remains eligible at the same address and weight.
  peer.health.ib_devices[1] = {"ib1", 2, 2, true};
  mds.SetMembers({peer});
  ASSERT_TRUE(transport.WaitForPeerTopologies(2));
  client.StopMdsDiscovery();

  const std::vector<PeerTopology> changed = transport.PeerTopologies();
  ASSERT_EQ(changed.size(), 2u);
  EXPECT_EQ(changed[1].peer_addr, initial[0].peer_addr);
  EXPECT_EQ(changed[1].peer_id, initial[0].peer_id);
  EXPECT_TRUE(changed[1].present);
  EXPECT_NE(changed[1].generation, initial[0].generation);
  EXPECT_EQ(changed[1].generation,
            MembersTopologyEpoch(std::vector<MemberInfo>{peer}));
  ASSERT_EQ(changed[1].rails.size(), 2u);
  EXPECT_TRUE(changed[1].rails[0].healthy);
  EXPECT_FALSE(changed[1].rails[1].healthy);
  ASSERT_TRUE(transport.WaitForPeerIdentitySets(2));
  EXPECT_EQ(transport.PeerIdentitySets(),
            (std::vector<std::vector<std::string>>{{"n1"}, {"n1"}}));

  // Placement was adopted once. The partial rail transition was independently
  // forwarded to the transport without another ring adoption/scale hint.
  const std::vector<size_t> hints = transport.Hints();
  ASSERT_EQ(hints.size(), 1u);
  EXPECT_EQ(hints[0], 1u);
  EXPECT_TRUE(client.WaitForRing(0));
  EXPECT_TRUE(mds.only_topology_requests());
}

TEST(DynamicMembers, StableIdentityTracksAddressAndIdentityReplacement) {
  using P = std::vector<std::pair<std::string, std::string>>;
  MemberInfo peer{"stable-n1", "127.0.0.1", 28301, 1};
  peer.has_health = true;
  peer.health.ring_eligible = true;
  peer.health.ib_devices = {{"ib0", 4, 5, true}};
  ScriptedTopologyMds mds({peer});
  HintRecordingTransport transport;
  KVClient client(P{}, Hdr(), &transport);
  // A long interval prevents the guarded empty placement from reaching its
  // persistence threshold before the first authoritative removal is asserted.
  client.StartMdsDiscovery({mds.endpoint()}, "identity-test", 1000);

  ASSERT_TRUE(transport.WaitForPeerTopologies(1));
  ASSERT_TRUE(transport.WaitForPeerIdentitySets(1));
  const uint64_t initial_generation =
      MembersTopologyEpoch(std::vector<MemberInfo>{peer});
  const auto initial_snapshot =
      transport.PeerSnapshot("127.0.0.1:28301");
  ASSERT_EQ(initial_snapshot->peer_id, "stable-n1");
  ASSERT_EQ(initial_snapshot->generation, initial_generation);

  // A routing address change explicitly retires the old address, then
  // publishes the same stable identity at its new address.
  peer.port = 28302;
  mds.SetMembers({peer});
  ASSERT_TRUE(transport.WaitForPeerTopologies(3));
  ASSERT_TRUE(transport.WaitForPeerIdentitySets(2));
  const uint64_t address_generation =
      MembersTopologyEpoch(std::vector<MemberInfo>{peer});
  const auto address_snapshot =
      transport.PeerSnapshot("127.0.0.1:28302");
  ASSERT_EQ(address_snapshot->peer_id, "stable-n1");
  ASSERT_EQ(address_snapshot->generation, address_generation);
  EXPECT_EQ(initial_snapshot->peer_id, "stable-n1");
  EXPECT_EQ(initial_snapshot->generation, initial_generation);

  // Reusing an address for a different MDS id is an identity replacement, not
  // an address-derived alias of the old peer.
  peer.id = "replacement-n2";
  mds.SetMembers({peer});
  ASSERT_TRUE(transport.WaitForPeerTopologies(5));
  ASSERT_TRUE(transport.WaitForPeerIdentitySets(3));
  const uint64_t replacement_generation =
      MembersTopologyEpoch(std::vector<MemberInfo>{peer});
  const auto replacement_snapshot =
      transport.PeerSnapshot("127.0.0.1:28302");
  ASSERT_EQ(replacement_snapshot->peer_id, "replacement-n2");
  ASSERT_EQ(replacement_snapshot->generation, replacement_generation);
  EXPECT_EQ(address_snapshot->peer_id, "stable-n1");
  EXPECT_EQ(address_snapshot->generation, address_generation);

  // An empty view is refused by the placement guard, so the ring keeps the
  // final identity, and so does the transport: nothing is retired.
  mds.SetMembers({});
  ASSERT_TRUE(transport.WaitForPeerIdentitySets(4));
  client.StopMdsDiscovery();

  const auto updates = transport.PeerTopologies();
  ASSERT_EQ(updates.size(), 5u);
  EXPECT_EQ(updates[0].peer_id, "stable-n1");
  EXPECT_EQ(updates[0].peer_addr, "127.0.0.1:28301");
  EXPECT_TRUE(updates[0].present);
  EXPECT_EQ(updates[1].peer_id, "stable-n1");
  EXPECT_EQ(updates[1].peer_addr, "127.0.0.1:28301");
  EXPECT_EQ(updates[1].generation, address_generation);
  EXPECT_FALSE(updates[1].present);
  EXPECT_EQ(updates[2].peer_id, "stable-n1");
  EXPECT_EQ(updates[2].peer_addr, "127.0.0.1:28302");
  EXPECT_TRUE(updates[2].present);
  EXPECT_EQ(updates[3].peer_id, "stable-n1");
  EXPECT_EQ(updates[3].peer_addr, "127.0.0.1:28302");
  EXPECT_EQ(updates[3].generation, replacement_generation);
  EXPECT_FALSE(updates[3].present);
  EXPECT_EQ(updates[4].peer_id, "replacement-n2");
  EXPECT_EQ(updates[4].peer_addr, "127.0.0.1:28302");
  EXPECT_TRUE(updates[4].present);
  EXPECT_EQ(transport.PeerIdentitySets(),
            (std::vector<std::vector<std::string>>{
                {"stable-n1"}, {"stable-n1"}, {"replacement-n2"},
                {"replacement-n2"}}));
  EXPECT_EQ(transport.PeerSnapshot("127.0.0.1:28302")->peer_id,
            "replacement-n2");
}

namespace {
MemberInfo HealthyPeer(int index, uint32_t base_port) {
  MemberInfo peer{"n" + std::to_string(index), "127.0.0.1",
                  base_port + static_cast<uint32_t>(index), 1};
  peer.has_health = true;
  peer.health.ring_eligible = true;
  peer.health.ib_devices = {{"ib0", 4, 5, true}};
  return peer;
}
std::string AddressOf(const MemberInfo& peer) {
  return peer.ip + ":" + std::to_string(peer.port);
}
// Wait until the client has applied whatever the MDS was last told to say.
// The MDS counts an answer after it has written it and closed the socket. A
// request already in flight may still carry the previous answer (first
// count); the next one carries the new answer, but it is counted before the
// client has decoded it and run its callback (second count). The poller is
// serial, so it asks a third time only after it has finished with the second.
bool Applied(ScriptedTopologyMds* mds) {
  return mds->WaitForAnswered(mds->Answered() + 3);
}
}  // namespace

// A shrink the guard refuses (the typical case: etcd leases of live nodes
// expired together) keeps the omitted peers on the ring. Their transport
// topology must stay too, or under explicit rail tiers every request to them
// fails kNoCompatibleRail until they register again: the guard would keep
// the routes and lose the data path.
TEST(DynamicMembers, GuardHeldPeersKeepTheirDataPath) {
  using P = std::vector<std::pair<std::string, std::string>>;
  std::vector<MemberInfo> peers;
  for (int i = 0; i < 3; ++i) {
    MemberInfo peer{"n" + std::to_string(i), "127.0.0.1",
                    static_cast<uint32_t>(28211 + i), 1};
    peer.has_health = true;
    peer.health.ring_eligible = true;
    peer.health.ib_devices = {{"ib0", 4, 5, true}};
    peers.push_back(peer);
  }
  ScriptedTopologyMds mds(peers);
  HintRecordingTransport transport;
  KVClient client(P{}, Hdr(), &transport);
  // A second between polls: the shrink is seen once and refused.
  client.StartMdsDiscovery({mds.endpoint()}, "guard-held-test", 1000);
  ASSERT_TRUE(transport.WaitForPeerTopologies(3));
  ASSERT_TRUE(client.WaitForRing(5000));
  ASSERT_TRUE(transport.WaitForPeerIdentitySets(1));

  mds.SetMembers({peers[0]});
  ASSERT_TRUE(transport.WaitForPeerIdentitySets(2));
  client.StopMdsDiscovery();
  ASSERT_EQ(transport.Hints(), std::vector<size_t>({3u}));

  for (const auto& update : transport.PeerTopologies())
    EXPECT_TRUE(update.present) << update.peer_id;
  for (int i = 1; i < 3; ++i) {
    const auto held = transport.PeerSnapshot(
        "127.0.0.1:" + std::to_string(28211 + i));
    EXPECT_EQ(held->peer_id, "n" + std::to_string(i));
    EXPECT_TRUE(rdma::HasRail(held->peer_healthy));
  }
  EXPECT_EQ(transport.PeerIdentitySets().back(),
            (std::vector<std::string>{"n0", "n1", "n2"}));
  const char value = 'v';
  for (int i = 0; i < 300; ++i)
    EXPECT_TRUE(client.Put("held-" + std::to_string(i), &value, 1)) << i;
}

// Once the guard believes the shrink, the omitted peers are retired.
TEST(DynamicMembers, AnAdmittedShrinkRetiresTheOmittedPeers) {
  using P = std::vector<std::pair<std::string, std::string>>;
  std::vector<MemberInfo> peers;
  for (int i = 0; i < 4; ++i) {
    MemberInfo peer{"n" + std::to_string(i), "127.0.0.1",
                    static_cast<uint32_t>(28221 + i), 1};
    peer.has_health = true;
    peer.health.ring_eligible = true;
    peer.health.ib_devices = {{"ib0", 4, 5, true}};
    peers.push_back(peer);
  }
  ScriptedTopologyMds mds(peers);
  HintRecordingTransport transport;
  KVClient client(P{}, Hdr(), &transport);
  client.StartMdsDiscovery({mds.endpoint()}, "admitted-shrink-test", 10);
  ASSERT_TRUE(transport.WaitForPeerTopologies(4));
  ASSERT_TRUE(transport.WaitForHints(1));

  mds.SetMembers({peers[0], peers[1], peers[2]});
  ASSERT_TRUE(transport.WaitForHints(2));
  ASSERT_TRUE(Applied(&mds));
  client.StopMdsDiscovery();
  EXPECT_EQ(transport.Hints(), std::vector<size_t>({4u, 3u}));
  EXPECT_TRUE(transport.PeerSnapshot("127.0.0.1:28224")->peer_id.empty());
  EXPECT_EQ(transport.PeerIdentitySets().back(),
            (std::vector<std::string>{"n0", "n1", "n2"}));
  bool retired = false;
  for (const auto& update : transport.PeerTopologies())
    retired = retired || (update.peer_id == "n3" && !update.present);
  EXPECT_TRUE(retired);
}

// A change of one member must not invalidate what the transport holds for
// the others. The pooled connections of a transport are bound to the
// publication of their peer and are torn down when it moves on; count them.
TEST(DynamicMembers, OneMemberChangeTearsDownOnlyThatPeersConnections) {
  using P = std::vector<std::pair<std::string, std::string>>;
  const int kPeers = 8;
  const int kConnectionsPerPeer = 4;
  std::vector<MemberInfo> peers;
  for (int i = 0; i < kPeers; ++i) peers.push_back(HealthyPeer(i, 28401));
  ScriptedTopologyMds mds(peers);
  HintRecordingTransport transport;
  KVClient client(P{}, Hdr(), &transport);
  client.StartMdsDiscovery({mds.endpoint()}, "one-of-n-test", 10);
  ASSERT_TRUE(transport.WaitForPeerTopologies(kPeers));

  struct PooledConnection {
    std::string address;
    std::string peer_id;
    uint64_t publication;
  };
  std::vector<PooledConnection> pool;
  for (const auto& peer : peers) {
    const uint64_t publication = transport.Publication(AddressOf(peer));
    ASSERT_NE(publication, 0u);
    for (int i = 0; i < kConnectionsPerPeer; ++i)
      pool.push_back({AddressOf(peer), peer.id, publication});
  }

  // One port of n3 goes down. Every member is republished.
  peers[3].health.ib_devices[0] = {"ib0", 1, 2, true};
  mds.SetMembers(peers);
  ASSERT_TRUE(transport.WaitForPeerTopologies(2 * kPeers));
  client.StopMdsDiscovery();

  int torn_down = 0;
  for (const auto& connection : pool) {
    if (!transport.Current(connection.address, connection.peer_id,
                           connection.publication)) {
      ++torn_down;
      EXPECT_EQ(connection.peer_id, "n3");
    }
  }
  EXPECT_EQ(torn_down, kConnectionsPerPeer)
      << "of " << pool.size() << " pooled connections";
  EXPECT_FALSE(transport.PeerSnapshot(AddressOf(peers[3]))->peer_healthy[0]);
}

// The per-peer generation covers what the generation of the whole view did,
// peer by peer.
TEST(DynamicMembers, PeerPublicationFollowsOnlyItsOwnMember) {
  using P = std::vector<std::pair<std::string, std::string>>;
  std::vector<MemberInfo> peers;
  for (int i = 0; i < 4; ++i) peers.push_back(HealthyPeer(i, 28501));
  ScriptedTopologyMds mds(peers);
  HintRecordingTransport transport;
  KVClient client(P{}, Hdr(), &transport);
  client.StartMdsDiscovery({mds.endpoint()}, "per-peer-test", 10);
  ASSERT_TRUE(transport.WaitForPeerTopologies(4));
  ASSERT_TRUE(Applied(&mds));

  std::vector<std::string> address;
  std::vector<uint64_t> publication;
  for (const auto& peer : peers) {
    address.push_back(AddressOf(peer));
    publication.push_back(transport.Publication(address.back()));
    ASSERT_NE(publication.back(), 0u);
  }
  const auto unchanged = [&](int i) {
    return transport.Publication(address[i]) == publication[i];
  };

  // The same members in another order: not a change at all.
  size_t updates = transport.PeerTopologies().size();
  mds.SetMembers({peers[3], peers[1], peers[0], peers[2]});
  ASSERT_TRUE(Applied(&mds));
  EXPECT_EQ(transport.PeerTopologies().size(), updates);
  for (int i = 0; i < 4; ++i) EXPECT_TRUE(unchanged(i)) << i;

  // An MDS that answers an error instead of a list: the last view stays.
  mds.SetStatus(Status::kIOError);
  ASSERT_TRUE(Applied(&mds));
  mds.SetStatus(Status::kOk);
  ASSERT_TRUE(Applied(&mds));
  EXPECT_EQ(transport.PeerTopologies().size(), updates);
  for (int i = 0; i < 4; ++i) EXPECT_TRUE(unchanged(i)) << i;

  // n1 reports one more rail, in another order than before.
  peers[1].health.ib_devices = {{"ib1", 4, 5, true}, {"ib0", 4, 5, true}};
  mds.SetMembers(peers);
  ASSERT_TRUE(Applied(&mds));
  EXPECT_GT(transport.Publication(address[1]), publication[1]);
  publication[1] = transport.Publication(address[1]);
  EXPECT_TRUE(unchanged(0));
  EXPECT_TRUE(unchanged(2));
  EXPECT_TRUE(unchanged(3));

  // n2 leaves and comes back unchanged at the same address: a new
  // incarnation, although its entry is byte for byte what it was.
  mds.SetMembers({peers[0], peers[1], peers[3]});
  ASSERT_TRUE(Applied(&mds));
  EXPECT_TRUE(transport.PeerSnapshot(address[2])->peer_id.empty());
  EXPECT_FALSE(transport.Current(address[2], "n2", publication[2]));
  mds.SetMembers(peers);
  ASSERT_TRUE(Applied(&mds));
  EXPECT_EQ(transport.PeerSnapshot(address[2])->peer_id, "n2");
  EXPECT_GT(transport.Publication(address[2]), publication[2]);
  publication[2] = transport.Publication(address[2]);
  EXPECT_TRUE(unchanged(0));
  EXPECT_TRUE(unchanged(1));
  EXPECT_TRUE(unchanged(3));

  // n3 leaves and comes back at another address.
  mds.SetMembers({peers[0], peers[1], peers[2]});
  ASSERT_TRUE(Applied(&mds));
  const std::string old_address = address[3];
  peers[3].port = 28599;
  address[3] = AddressOf(peers[3]);
  mds.SetMembers(peers);
  ASSERT_TRUE(Applied(&mds));
  client.StopMdsDiscovery();
  EXPECT_TRUE(transport.PeerSnapshot(old_address)->peer_id.empty());
  EXPECT_FALSE(transport.Current(old_address, "n3", publication[3]));
  EXPECT_EQ(transport.PeerSnapshot(address[3])->peer_id, "n3");
  EXPECT_GT(transport.Publication(address[3]), publication[3]);
  EXPECT_TRUE(unchanged(0));
  EXPECT_TRUE(unchanged(1));
  EXPECT_TRUE(unchanged(2));
}
