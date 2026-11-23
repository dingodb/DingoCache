// A cache node whose disk store has failed for good must leave placement,
// unless it has other disks to serve on (DFKV_DEGRADED_MIN_DISKS).
//
// Such a store stores and reads nothing more until the process is restarted.
// The node's registrar thread is unaffected and keeps renewing the lease;
// unless the heartbeat says otherwise, the node stays in every client's ring
// and its share of the keys is lost to them for good.
//
// End to end without an etcd: a real KvNodeServer over a slab store, the
// registrar wiring dfkv_server uses, and a real MdsServer over the in-process
// etcd gateway fake.
#include "cache/node_member_report.h"

#include <gtest/gtest.h>

#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cache/disk_cache_group.h"
#include "cache/disk_slab_store.h"
#include "cache/kv_node_server.h"
#include "cache/ram_tier.h"
#include "common/membership.h"
#include "fake_etcd.h"
#include "mds/mds_proto.h"
#include "mds/mds_registrar.h"
#include "mds/mds_server.h"
#include "transport/wire.h"
#include "utils/net_util.h"

namespace fs = std::filesystem;

namespace dfkv {
class DiskSlabStoreTestPeer {
 public:
  static void SetRecordWriteHook(DiskSlabStore* store,
                                 std::function<bool()> hook) {
    store->record_write_hook_for_test_ = std::move(hook);
  }
};
class KvNodeServerWiringTestPeer {
 public:
  static std::unique_ptr<KvNodeServer> Create(DiskCacheGroup::Options options) {
    return std::unique_ptr<KvNodeServer>(new KvNodeServer(std::move(options)));
  }
  // A write-back RAM tier whose flush to disk never succeeds.
  static void SetRamThatCannotFlush(KvNodeServer& server) {
    RamTier::Options options;
    options.bytes = 16 * 4096;
    options.large_reserve_bytes = 0;
    options.reclaim_interval_ms = 0;
    options.is_persisted = [](const BlockKey&) { return false; };
    server.ram_ = std::make_unique<RamTier>(
        options, [](const BlockKey&, char*, size_t, size_t) {
          return Status::kIOError;
        });
    server.ram_write_back_ = true;
    server.ram_ack_enabled_ = false;
  }
};
}  // namespace dfkv

using namespace dfkv;  // NOLINT

namespace {

bool List(const MdsServer& mds, WireOp op, const std::string& group,
          std::vector<MemberInfo>* members) {
  const int fd =
      net::Dial("127.0.0.1:" + std::to_string(mds.port()), 2000, 5000);
  if (fd < 0) return false;
  char pre[kReqPrefix];
  EncodeReq(pre, op, BlockKey{}, 0, 0, group.size());
  char rp[kRespPrefix];
  Status st = Status::kInvalid;
  uint64_t dlen = 0;
  std::string data;
  bool ok = net::WriteAll(fd, pre, kReqPrefix) &&
            net::WriteAll(fd, group.data(), group.size()) &&
            net::ReadAll(fd, rp, kRespPrefix) && DecodeResp(rp, &st, &dlen) &&
            st == Status::kOk;
  if (ok) {
    data.resize(dlen);
    ok = dlen == 0 || net::ReadAll(fd, &data[0], dlen);
  }
  ::close(fd);
  uint64_t epoch = 0;
  return ok && DecodeMembers(data.data(), data.size(), members, &epoch);
}

// The ids clients build their ring from.
std::set<std::string> Placement(const MdsServer& mds, const std::string& group) {
  std::vector<MemberInfo> members;
  EXPECT_TRUE(List(mds, WireOp::kListMembers, group, &members));
  std::set<std::string> ids;
  for (const auto& m : members) ids.insert(m.id);
  return ids;
}

// One registered member as `dfkvctl ring` sees it.
MemberInfo Registered(const MdsServer& mds, const std::string& group,
                      const std::string& id) {
  std::vector<MemberInfo> members;
  EXPECT_TRUE(List(mds, WireOp::kListTopology, group, &members));
  for (const auto& m : members)
    if (m.id == id) return m;
  ADD_FAILURE() << id << " is not registered";
  return {};
}

class NodeHealthMembership : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           ("dfkv_node_health_" + std::to_string(::getpid()) + "_" +
            ::testing::UnitTest::GetInstance()->current_test_info()->name());
    fs::remove_all(dir_);
    fs::create_directories(dir_ / "tokens");
    ASSERT_EQ(mds_.Start(0), Status::kOk);

    // A second, healthy member, so that the ring is seen to shrink by one.
    MdsRegistrar other({MdsEp()}, kGroup, MemberInfo{"other", "10.0.0.2", 7000, 1});
    ASSERT_TRUE(other.RegisterOnce());

    DiskCacheGroup::Options options;
    for (size_t i = 0; i < Disks(); ++i)
      options.cache_dirs.push_back((dir_ / ("disk" + std::to_string(i))).string());
    stores_.assign(Disks(), nullptr);
    options.engine = "slab";
    options.engine_factory = [this](const std::string& dir,
                                    uint64_t) -> std::unique_ptr<StoreEngine> {
      DiskSlabStore::Options o;
      o.dir = dir;
      o.trust_token_dir = (dir_ / "tokens").string();
      o.capacity_bytes = 4 << 20;
      o.extent_bytes = 1 << 20;
      o.slot_granularity = 4096;
      o.table_sync_ms = 0;
      o.reclaim_interval_ms = 0;
      auto store = std::make_unique<DiskSlabStore>(o);
      for (size_t i = 0; i < Disks(); ++i)
        if (dir == (dir_ / ("disk" + std::to_string(i))).string())
          stores_[i] = store.get();
      return store;
    };
    srv_ = KvNodeServerWiringTestPeer::Create(std::move(options));
    ASSERT_TRUE(srv_->Healthy());
    for (DiskSlabStore* store : stores_) ASSERT_NE(store, nullptr);
    store_ = stores_[0];
  }

  virtual size_t Disks() const { return 1; }

  void TearDown() override {
    srv_.reset();
    mds_.Stop();
    fs::remove_all(dir_);
  }

  std::string MdsEp() const {
    return "127.0.0.1:" + std::to_string(mds_.port());
  }

  // The registrar of this node, wired as dfkv_server wires it.
  std::unique_ptr<MdsRegistrar> Registrar(
      std::function<MemberHealth()> rails) {
    MemberInfo self{"node", "10.0.0.1", 7000, 1};
    self.info = kInfo;
    auto registrar = std::make_unique<MdsRegistrar>(
        std::vector<std::string>{MdsEp()}, kGroup, self);
    ReportNodeHealth(registrar.get(), srv_.get(), self.info, std::move(rails));
    return registrar;
  }

  Status Put(uint64_t id) {
    const std::string value(4096, 'v');
    std::string out;
    return srv_->ProcessRequestForKey(static_cast<uint8_t>(WireOp::kCache),
                                      BlockKey{id, 0}, 0, 0, value.data(),
                                      value.size(), &out);
  }

  // slots.tbl stops taking writes: the store fails for good.
  void FailDiskStore() {
    DiskSlabStoreTestPeer::SetRecordWriteHook(store_, [] { return false; });
    EXPECT_EQ(Put(100), Status::kIOError);
    DiskSlabStoreTestPeer::SetRecordWriteHook(store_, {});
    // The device taking writes again changes nothing.
    EXPECT_EQ(Put(101), Status::kIOError);
    EXPECT_FALSE(srv_->DisksHealthy());
  }

  const std::string kGroup = "g";
  const std::string kInfo = "ver=test,engine=slab";
  fs::path dir_;
  FakeEtcd etcd_;
  MdsServer mds_{etcd_.ep()};
  DiskSlabStore* store_ = nullptr;
  std::vector<DiskSlabStore*> stores_;
  std::unique_ptr<KvNodeServer> srv_;
};

class TwoDiskNodeHealthMembership : public NodeHealthMembership {
 protected:
  size_t Disks() const override { return 2; }
};

TEST_F(NodeHealthMembership, FailedDiskStoreTakesTheNodeOutOfPlacement) {
  auto registrar = Registrar(/*rails=*/{});
  ASSERT_TRUE(registrar->RegisterOnce());
  ASSERT_EQ(Put(1), Status::kOk);
  ASSERT_TRUE(registrar->HeartbeatOnce());
  EXPECT_EQ(Placement(mds_, kGroup), (std::set<std::string>{"node", "other"}));
  // A healthy node without RDMA rails reports what it always did: no HLT1.
  MemberInfo seen = Registered(mds_, kGroup, "node");
  EXPECT_FALSE(seen.has_health);
  EXPECT_EQ(seen.info, kInfo);

  FailDiskStore();
  // The lease is renewed as before: the node does not drop out by expiry.
  ASSERT_TRUE(registrar->HeartbeatOnce());

  EXPECT_EQ(Placement(mds_, kGroup), (std::set<std::string>{"other"}));
  seen = Registered(mds_, kGroup, "node");
  EXPECT_TRUE(seen.has_health);
  EXPECT_FALSE(seen.RingEligible());
  EXPECT_TRUE(seen.health.ib_devices.empty());
  EXPECT_EQ(seen.info, kInfo + ",store=failed");

  // It stays that way: nothing but a restart brings the store back.
  ASSERT_TRUE(registrar->HeartbeatOnce());
  EXPECT_EQ(Placement(mds_, kGroup), (std::set<std::string>{"other"}));
}

TEST_F(NodeHealthMembership, FailedDiskStoreOutranksHealthyRails) {
  const auto rails = [] {
    MemberHealth health;
    health.ring_eligible = true;
    health.ib_devices.push_back(IbDeviceHealth{"ib0", 4, 5, true});
    return health;
  };
  auto registrar = Registrar(rails);
  ASSERT_TRUE(registrar->RegisterOnce());
  MemberInfo seen = Registered(mds_, kGroup, "node");
  EXPECT_TRUE(seen.has_health);
  EXPECT_TRUE(seen.RingEligible());
  EXPECT_EQ(Placement(mds_, kGroup), (std::set<std::string>{"node", "other"}));

  FailDiskStore();
  ASSERT_TRUE(registrar->HeartbeatOnce());

  EXPECT_EQ(Placement(mds_, kGroup), (std::set<std::string>{"other"}));
  seen = Registered(mds_, kGroup, "node");
  EXPECT_FALSE(seen.RingEligible());
  // The rail report itself is passed on unchanged.
  ASSERT_EQ(seen.health.ib_devices.size(), 1u);
  EXPECT_TRUE(seen.health.ib_devices[0].healthy());
  EXPECT_EQ(seen.info, kInfo + ",store=failed");
}

// The RAM tier's own latch (an entry dropped after its flush kept failing)
// makes the node unhealthy for /healthz, but the node still serves every
// request and the cause can be load. It must not cost the node its place.
TEST_F(NodeHealthMembership, UnhealthyRamTierAloneKeepsTheNodeInPlacement) {
  auto registrar = Registrar(/*rails=*/{});
  ASSERT_TRUE(registrar->RegisterOnce());
  KvNodeServerWiringTestPeer::SetRamThatCannotFlush(*srv_);
  EXPECT_NE(Put(1), Status::kOk);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (srv_->Healthy() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  ASSERT_FALSE(srv_->Healthy());
  ASSERT_TRUE(srv_->DisksHealthy());

  ASSERT_TRUE(registrar->HeartbeatOnce());
  EXPECT_EQ(Placement(mds_, kGroup), (std::set<std::string>{"node", "other"}));
  const MemberInfo seen = Registered(mds_, kGroup, "node");
  EXPECT_FALSE(seen.has_health);
  EXPECT_EQ(seen.info, kInfo);
}

// With two disks, one failed store leaves the node in placement: its keys go
// to the other disk, where they miss until written again. INFO says so.
TEST_F(TwoDiskNodeHealthMembership, OneFailedDiskOfTwoKeepsTheNodeInPlacement) {
  auto registrar = Registrar(/*rails=*/{});
  ASSERT_TRUE(registrar->RegisterOnce());
  for (uint64_t id = 1; id <= 32; ++id) ASSERT_EQ(Put(id), Status::kOk);

  DiskSlabStoreTestPeer::SetRecordWriteHook(stores_[0], [] { return false; });
  uint64_t failed_key = 0;
  for (uint64_t id = 1000; id < 1100 && failed_key == 0; ++id)
    if (Put(id) == Status::kIOError) failed_key = id;
  DiskSlabStoreTestPeer::SetRecordWriteHook(stores_[0], {});
  ASSERT_NE(failed_key, 0u) << "no key went to the first disk";
  ASSERT_FALSE(stores_[0]->Healthy());
  EXPECT_TRUE(srv_->DisksHealthy());
  EXPECT_TRUE(srv_->DisksDegraded());
  EXPECT_TRUE(srv_->Healthy()) << "/healthz stays 200";
  // Every key is taken again, the failed disk's on the other one.
  for (uint64_t id = 1; id <= 32; ++id) EXPECT_EQ(Put(id), Status::kOk) << id;
  EXPECT_EQ(Put(failed_key), Status::kOk);

  ASSERT_TRUE(registrar->HeartbeatOnce());
  EXPECT_EQ(Placement(mds_, kGroup), (std::set<std::string>{"node", "other"}));
  const MemberInfo seen = Registered(mds_, kGroup, "node");
  EXPECT_FALSE(seen.has_health);
  EXPECT_EQ(seen.info, kInfo + ",store=degraded");
  const std::string metrics = srv_->MetricsText();
  EXPECT_NE(metrics.find("\ndfkv_disks_failed 1\n"), std::string::npos);
  EXPECT_NE(metrics.find("\ndfkv_server_degraded 1\n"), std::string::npos);
}

// DFKV_DEGRADED_MIN_DISKS at the disk count: one failed disk takes the node
// out of placement, as before.
class TwoDiskNoDegradation : public TwoDiskNodeHealthMembership {
 protected:
  void SetUp() override {
    ::setenv("DFKV_DEGRADED_MIN_DISKS", "2", 1);
    TwoDiskNodeHealthMembership::SetUp();
    ::unsetenv("DFKV_DEGRADED_MIN_DISKS");
  }
};

TEST_F(TwoDiskNoDegradation, OneFailedDiskTakesTheNodeOut) {
  auto registrar = Registrar(/*rails=*/{});
  ASSERT_TRUE(registrar->RegisterOnce());
  DiskSlabStoreTestPeer::SetRecordWriteHook(stores_[1], [] { return false; });
  bool failed = false;
  for (uint64_t id = 1000; id < 1100 && !failed; ++id)
    failed = Put(id) == Status::kIOError;
  DiskSlabStoreTestPeer::SetRecordWriteHook(stores_[1], {});
  ASSERT_TRUE(failed) << "no key went to the second disk";
  EXPECT_FALSE(srv_->DisksHealthy());
  EXPECT_FALSE(srv_->DisksDegraded());
  ASSERT_TRUE(registrar->HeartbeatOnce());
  EXPECT_EQ(Placement(mds_, kGroup), (std::set<std::string>{"other"}));
  EXPECT_EQ(Registered(mds_, kGroup, "node").info, kInfo + ",store=failed");
}

TEST(NodeMemberHealth, FollowsRailsWhileTheDisksAreHealthy) {
  EXPECT_FALSE(NodeMemberHealth(true, {}).has_value());
  const auto down = [] {
    MemberHealth health;
    health.ring_eligible = false;
    return health;
  };
  const auto reported = NodeMemberHealth(true, down);
  ASSERT_TRUE(reported.has_value());
  EXPECT_FALSE(reported->ring_eligible);
  EXPECT_EQ(NodeMemberInfo("ver=x", true, false), "ver=x");
  EXPECT_EQ(NodeMemberInfo("ver=x", true, true), "ver=x,store=degraded");
  EXPECT_EQ(NodeMemberInfo("ver=x", false, false), "ver=x,store=failed");
}

}  // namespace
