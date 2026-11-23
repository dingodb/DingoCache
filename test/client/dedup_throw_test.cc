// A throw on the calling thread between a host-rendezvous claim and its
// publish or abort (an allocation failing, in practice) left the claimed
// slots FETCHING: every other process of the host then waited on those keys
// until the takeover age. The test fails the Nth allocation of the calling
// thread, for every N until the first read is issued (the claims and the
// hand-off to the read workers): after each throw no key of the batch may
// still be claimed.
#include "client/key_map.h"
#include "client/kv_client.h"
#include "client/node_dedup.h"
#include "transport/transport.h"

#include <gtest/gtest.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <vector>

namespace {
thread_local long g_fail_in = 0;  // 0: off; n: the nth allocation fails
}  // namespace

// The whole replaceable set, on malloc and free, so that every new pairs
// with its delete (std::get_temporary_buffer, used by gtest's stable_sort,
// allocates with nothrow new: left to the library, it was freed here).
// Only the throwing forms fail on purpose.
namespace {
void* Allocate(std::size_t n) {
  if (g_fail_in > 0 && --g_fail_in == 0) throw std::bad_alloc();
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void* AlignedOrNull(std::size_t n, std::align_val_t a) {
  const std::size_t align = static_cast<std::size_t>(a);
  return std::aligned_alloc(align, (n ? n + align - 1 : align) / align * align);
}
void* AllocateAligned(std::size_t n, std::align_val_t a) {
  if (g_fail_in > 0 && --g_fail_in == 0) throw std::bad_alloc();
  if (void* p = AlignedOrNull(n, a)) return p;
  throw std::bad_alloc();
}
}  // namespace

void* operator new(std::size_t n) { return Allocate(n); }
void* operator new[](std::size_t n) { return Allocate(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
  return std::malloc(n ? n : 1);
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
  return std::malloc(n ? n : 1);
}
void* operator new(std::size_t n, std::align_val_t a) {
  return AllocateAligned(n, a);
}
void* operator new[](std::size_t n, std::align_val_t a) {
  return AllocateAligned(n, a);
}
void* operator new(std::size_t n, std::align_val_t a,
                   const std::nothrow_t&) noexcept {
  return AlignedOrNull(n, a);
}
void* operator new[](std::size_t n, std::align_val_t a,
                     const std::nothrow_t&) noexcept {
  return AlignedOrNull(n, a);
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept {
  std::free(p);
}
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept {
  std::free(p);
}
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept {
  std::free(p);
}
void operator delete(void* p, std::align_val_t,
                     const std::nothrow_t&) noexcept {
  std::free(p);
}
void operator delete[](void* p, std::align_val_t,
                       const std::nothrow_t&) noexcept {
  std::free(p);
}

namespace dfkv {
namespace {

constexpr size_t kValueLen = 64;

class MapTransport final : public Transport {
 public:
  std::map<std::string, std::string> values;
  std::atomic<int> reads{0};
  Status Cache(const std::string&, const BlockKey&, const void*,
               size_t) override {
    return Status::kOk;
  }
  Status Range(const std::string&, const BlockKey&, uint64_t, uint64_t,
               std::string*, uint64_t*) override {
    return Status::kInvalid;
  }
  Status Lookup(const std::string&, const BlockKey&, uint64_t*) override {
    return Status::kInvalid;
  }
  Status Exist(const std::string&, const BlockKey& key,
               bool* exists) override {
    ++reads;
    std::lock_guard<std::mutex> lock(mu_);
    *exists = values.count(key.Filename()) != 0;
    return *exists ? Status::kOk : Status::kNotFound;
  }
  std::vector<Status> RangeInto(const std::string&,
                                const std::vector<BlockKey>& keys,
                                const std::vector<RangeDst>& dsts,
                                std::vector<uint64_t>* value_lens) override {
    ++reads;
    std::lock_guard<std::mutex> lock(mu_);
    if (value_lens) value_lens->assign(keys.size(), 0);
    std::vector<Status> result(keys.size(), Status::kNotFound);
    for (size_t i = 0; i < keys.size(); ++i) {
      auto found = values.find(keys[i].Filename());
      if (found == values.end()) continue;
      std::memcpy(dsts[i].payload, found->second.data(),
                  std::min(dsts[i].n, found->second.size()));
      if (value_lens) (*value_lens)[i] = found->second.size();
      result[i] = Status::kOk;
    }
    return result;
  }

 private:
  std::mutex mu_;
};

struct DedupEnv {
  std::string name;
  explicit DedupEnv(const std::string& key_namespace) {
    ::setenv("DFKV_CLIENT_NODE_DEDUP", "1", 1);
    ::setenv("DFKV_NODE_DEDUP_ARENA_MB", "1", 1);
    ::setenv("DFKV_NODE_DEDUP_SLOTS", "4096", 1);
    name = NodeDedup::EnvSegmentName(ToBlockKey(key_namespace, "").digest_hi);
    ::shm_unlink(name.c_str());
  }
  ~DedupEnv() {
    ::shm_unlink(name.c_str());
    ::unsetenv("DFKV_CLIENT_NODE_DEDUP");
    ::unsetenv("DFKV_NODE_DEDUP_ARENA_MB");
    ::unsetenv("DFKV_NODE_DEDUP_SLOTS");
  }
  std::unique_ptr<NodeDedup> Peer() const {
    NodeDedup::Options o;
    o.name = name;
    o.arena_bytes = 1ull << 20;
    o.slots = 4096;
    return NodeDedup::Open(o);
  }
};

constexpr size_t kKeys = 8;

// Whether the peer process can claim every key of the batch now; what it
// claims is given back.
bool AllClaimable(NodeDedup* peer, const std::string& key_namespace,
                  bool exist) {
  bool all = true;
  for (size_t i = 0; i < kKeys; ++i) {
    const BlockKey key = ToBlockKey(key_namespace, "k" + std::to_string(i));
    uint64_t token = 0;
    bool val = false;
    char scratch[kValueLen];
    const NodeDedup::Role role =
        exist ? peer->ClaimExist(key, &val, &token)
              : peer->Claim(key, kValueLen, scratch, &token);
    if (role == NodeDedup::Role::kWait) all = false;
    if (token)
      peer->Abort(key, exist ? NodeDedup::Kind::kExist : NodeDedup::Kind::kData,
                  token);
  }
  return all;
}

void Sweep(bool exist) {
  const std::string key_namespace = std::string("dedup-throw/") +
                                    (exist ? "exist/" : "get/") +
                                    std::to_string(::getpid());
  DedupEnv env(key_namespace);
  MapTransport transport;  // every key absent: nothing gets published
  KVClient client({{"n", "test:1"}}, key_namespace, &transport);
  auto peer = env.Peer();
  ASSERT_TRUE(peer);
  std::vector<std::string> keys;
  std::vector<std::string> bufs(kKeys, std::string(kValueLen, '\0'));
  std::vector<KvGetItem> items;
  for (size_t i = 0; i < kKeys; ++i) {
    keys.push_back("k" + std::to_string(i));
    items.push_back({keys.back(), bufs[i].data(), kValueLen});
  }
  int throws = 0;
  for (long n = 1; n < 10000; ++n) {
    bool threw = false;
    const int reads_before = transport.reads.load();
    g_fail_in = n;
    try {
      if (exist)
        (void)client.BatchExist(keys);
      else
        (void)client.BatchGet(items);
    } catch (const std::bad_alloc&) {
      threw = true;
    }
    g_fail_in = 0;
    if (!threw) break;  // the batch ran through: every allocation passed
    // A throw once reads are in flight is outside what this test covers.
    if (transport.reads.load() != reads_before) break;
    ++throws;
    EXPECT_TRUE(AllClaimable(peer.get(), key_namespace, exist))
        << "allocation " << n << " left a key claimed";
  }
  EXPECT_GT(throws, 0);
  std::printf("[ sweep ] %s: %d failing allocations tried\n",
              exist ? "BatchExist" : "BatchGet", throws);
}

TEST(ClientDedupThrow, AThrowingGetLeavesNoKeyClaimed) { Sweep(false); }
TEST(ClientDedupThrow, AThrowingExistLeavesNoKeyClaimed) { Sweep(true); }

}  // namespace
}  // namespace dfkv
