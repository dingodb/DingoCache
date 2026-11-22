// NodeDedup: same-host GET rendezvous over POSIX shm. Hermetic (no RDMA/etcd);
// cross-process behavior via fork. Every test uses its own uniquely-named
// segment and unlinks it, so runs never interfere.
#include "client/node_dedup.h"

#include <gtest/gtest.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <future>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using dfkv::BlockKey;
using dfkv::NodeDedup;
using namespace std::chrono_literals;

namespace dfkv {
// A reader looks a slot up and then copies out of it; these are the two
// stages, so a test can change the slot in between.
struct NodeDedupTestAccess {
  static auto Find(NodeDedup& d, const BlockKey& key, NodeDedup::Kind kind) {
    return d.Find(key, kind);
  }
  // A publisher or aborter after its token check, before it takes the slot.
  static auto OwnedSlot(NodeDedup& d, uint64_t token) {
    return d.OwnedSlot(token);
  }
  template <typename SlotPtr>
  static void PublishInto(NodeDedup& d, SlotPtr slot, const BlockKey& key,
                          NodeDedup::Kind kind, uint64_t token,
                          const std::string& v) {
    NodeDedup::PublishHold hold;
    if (d.PublishBegin(slot, key, kind, token, &hold))
      d.PublishFinish(hold, v.data(), v.size());
  }
  template <typename SlotPtr>
  static void AbortInto(NodeDedup& d, SlotPtr slot, const BlockKey& key,
                        NodeDedup::Kind kind, uint64_t token) {
    d.AbortSlot(slot, key, kind, token);
  }
  // A publisher that has taken its slot and not yet copied: Begin returns a
  // hold that Finish commits. Never calling Finish is a publisher killed
  // there.
  struct Hold {
    NodeDedup::PublishHold hold;
    bool taken = false;
  };
  static Hold PublishBegin(NodeDedup& d, const BlockKey& key,
                           NodeDedup::Kind kind, uint64_t token) {
    Hold h;
    if (auto slot = d.OwnedSlot(token))
      h.taken = d.PublishBegin(slot, key, kind, token, &h.hold);
    return h;
  }
  static void PublishFinish(NodeDedup& d, const Hold& h,
                            const std::string& v) {
    d.PublishFinish(h.hold, v.data(), v.size());
  }
  // An invalidation taking a slot from its publisher, in its two steps.
  static bool ForceOutBegin(NodeDedup& d, const Hold& h) {
    return d.ForceOutBegin(h.hold.slot);
  }
  static void ForceOutFinish(NodeDedup& d, const Hold& h) {
    d.ForceOutFinish(h.hold.slot);
  }
  // Another process's stale Publish/Abort holding a slot for its re-check.
  template <typename SlotPtr>
  static bool StaleLock(NodeDedup& d, SlotPtr slot) {
    return d.LockSlotForTest(slot);
  }
  template <typename SlotPtr>
  static void StaleUnlock(NodeDedup& d, SlotPtr slot) {
    d.UnlockSlotForTest(slot);
  }
  // How long a Publish/Abort waits for a slot held by a stale re-check.
  static void SetLockWait(NodeDedup& d, std::chrono::milliseconds wait) {
    d.lock_wait_ms_ = static_cast<int>(wait.count());
  }
  static void LookLikeAnOlderRelease(NodeDedup& d, const Hold& h) {
    d.SetFenceableForTest(h.hold, false);
  }
  template <typename SlotPtr>
  static bool CopyOut(NodeDedup& d, SlotPtr slot, const BlockKey& key,
                      NodeDedup::Kind kind, size_t n, char* dst) {
    return d.CopyOut(slot, key, kind, n, n, dst, nullptr);
  }
};
}  // namespace dfkv

namespace {

struct ShmGuard {
  std::string name;
  explicit ShmGuard(const std::string& tag)
      : name("/dfkv-dedup-test-" + std::to_string(::getpid()) + "-" + tag) {
    ::shm_unlink(name.c_str());
  }
  ~ShmGuard() { ::shm_unlink(name.c_str()); }
};

NodeDedup::Options Opts(const std::string& name, uint64_t arena = 4 << 20) {
  NodeDedup::Options o;
  o.name = name;
  o.arena_bytes = arena;
  o.slots = 1024;
  o.wait_ms = 300;
  o.takeover_ms = 200;
  o.ttl_ms = 1000;
  return o;
}

// A waiter whose deadline is far beyond any scheduling delay: a test that
// expects the wait to end early can then use a bound that is generous and
// still well short of the deadline.
NodeDedup::Options LongWait(const std::string& name) {
  NodeDedup::Options o = Opts(name);
  o.wait_ms = 5000;
  return o;
}

// For tests that expect kWait from a claim behind a fetch in flight: no
// scheduling delay may turn that into a takeover.
NodeDedup::Options NoTakeover(const std::string& name) {
  NodeDedup::Options o = Opts(name);
  o.takeover_ms = 600000;
  return o;
}

BlockKey K(uint64_t id) { return BlockKey{id, 0}; }

std::string Val(uint64_t id, size_t n) {
  std::string v(n, '\0');
  for (size_t i = 0; i < n; ++i) v[i] = static_cast<char>((id * 131 + i) & 0xFF);
  return v;
}

}  // namespace

TEST(NodeDedup, FetchPublishThenPeersHit) {
  ShmGuard g("basic");
  auto a = NodeDedup::Open(Opts(g.name));
  auto b = NodeDedup::Open(Opts(g.name));  // second attach (peer)
  ASSERT_TRUE(a && b);
  const std::string v = Val(1, 8192);
  std::string dst(v.size(), '\0');
  uint64_t token = 0, ignored = 0;

  ASSERT_EQ(a->Claim(K(1), v.size(), &dst[0], &token),
            NodeDedup::Role::kFetch);
  a->Publish(K(1), NodeDedup::Kind::kData, token, v.data(), v.size());
  ASSERT_EQ(b->Claim(K(1), v.size(), &dst[0], &ignored),
            NodeDedup::Role::kHit);
  EXPECT_EQ(dst, v);
  EXPECT_EQ(b->hits(), 1u);
}

TEST(NodeDedup, WaiterCopiesAfterPublish) {
  ShmGuard g("wait");
  auto a = NodeDedup::Open(Opts(g.name));
  auto b = NodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  const std::string v = Val(2, 65536);
  std::string dst_a(v.size(), '\0'), dst_b(v.size(), '\0');
  uint64_t token = 0, ignored = 0;

  ASSERT_EQ(a->Claim(K(2), v.size(), &dst_a[0], &token),
            NodeDedup::Role::kFetch);
  ASSERT_EQ(b->Claim(K(2), v.size(), &dst_b[0], &ignored),
            NodeDedup::Role::kWait);
  std::thread pub([&] {
    std::this_thread::sleep_for(50ms);
    a->Publish(K(2), NodeDedup::Kind::kData, token, v.data(), v.size());
  });
  EXPECT_TRUE(b->WaitCopy(K(2), v.size(), &dst_b[0]));
  pub.join();
  EXPECT_EQ(dst_b, v);
  EXPECT_EQ(b->wait_hits(), 1u);
}

TEST(NodeDedup, AbortEndsWaitAtOnce) {
  ShmGuard g("abort");
  auto a = NodeDedup::Open(Opts(g.name));
  auto b = NodeDedup::Open(LongWait(g.name));
  ASSERT_TRUE(a && b);
  std::string dst(4096, '\0');
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->Claim(K(3), dst.size(), &dst[0], &token),
            NodeDedup::Role::kFetch);
  ASSERT_EQ(b->Claim(K(3), dst.size(), &dst[0], &ignored),
            NodeDedup::Role::kWait);
  a->Abort(K(3), NodeDedup::Kind::kData, token);
  // Nothing will be published: the waiter must fall back to its own fetch
  // now, not after its wait_ms (5 s).
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_FALSE(b->WaitCopy(K(3), dst.size(), &dst[0]));
  EXPECT_LT(std::chrono::steady_clock::now() - t0, 2s);
  EXPECT_EQ(b->wait_timeouts(), 1u);
}

TEST(NodeDedup, AbortDuringWaitEndsIt) {
  ShmGuard g("abort-late");
  auto a = NodeDedup::Open(Opts(g.name));
  auto b = NodeDedup::Open(LongWait(g.name));
  ASSERT_TRUE(a && b);
  bool val = false;
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->ClaimExist(K(14), &val, &token), NodeDedup::Role::kFetch);
  ASSERT_EQ(b->ClaimExist(K(14), &val, &ignored), NodeDedup::Role::kWait);
  const auto t0 = std::chrono::steady_clock::now();
  std::thread abort([&] {
    std::this_thread::sleep_for(50ms);
    a->Abort(K(14), NodeDedup::Kind::kExist, token);
  });
  EXPECT_FALSE(b->WaitExist(K(14), &val));
  const auto waited = std::chrono::steady_clock::now() - t0;
  abort.join();
  EXPECT_GE(waited, 50ms);  // a fetch in flight is waited for
  EXPECT_LT(waited, 2s);    // and its abort ends the wait (wait_ms 5 s)
}

TEST(NodeDedup, UnusablePublishedEntryEndsWaitAtOnce) {
  ShmGuard g("unusable");
  auto a = NodeDedup::Open(Opts(g.name));
  auto b = NodeDedup::Open(LongWait(g.name));
  ASSERT_TRUE(a && b);
  const std::string v = Val(15, 4096);
  std::string dst(8192, '\0');
  uint64_t token = 0;
  ASSERT_EQ(a->Claim(K(15), v.size(), &dst[0], &token),
            NodeDedup::Role::kFetch);
  a->Publish(K(15), NodeDedup::Kind::kData, token, v.data(), v.size());
  // A strict reader of another length can never copy this entry out.
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_FALSE(b->WaitCopy(K(15), dst.size(), &dst[0]));
  EXPECT_LT(std::chrono::steady_clock::now() - t0, 2s);  // wait_ms 5 s
}

TEST(NodeDedup, SharedDeadlineBoundsWaitAcrossKeys) {
  ShmGuard g("deadline");
  auto a = NodeDedup::Open(Opts(g.name));
  auto b = NodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  const int n = 8;
  const std::string v = Val(20, 4096);
  std::string dst(v.size(), '\0');
  std::vector<uint64_t> tokens(n, 0);
  uint64_t ignored = 0;
  for (int i = 0; i < n; ++i) {
    ASSERT_EQ(a->Claim(K(20 + i), v.size(), &dst[0], &tokens[i]),
              NodeDedup::Role::kFetch);
    ASSERT_EQ(b->Claim(K(20 + i), v.size(), &dst[0], &ignored),
              NodeDedup::Role::kWait);
  }
  // The owner hangs on every key but the last, which it publishes only after
  // the deadline has passed.
  const auto t0 = std::chrono::steady_clock::now();
  const uint64_t deadline = b->WaitDeadline();
  for (int i = 0; i < n - 1; ++i)
    EXPECT_FALSE(b->WaitCopy(K(20 + i), v.size(), &dst[0], deadline));
  const auto waited = std::chrono::steady_clock::now() - t0;
  EXPECT_GE(waited, 280ms);  // wait_ms 300, once
  EXPECT_LT(waited, 600ms);  // not once per key
  a->Publish(K(20 + n - 1), NodeDedup::Kind::kData, tokens[n - 1], v.data(),
             v.size());
  // A key reached after the deadline is still looked at once.
  EXPECT_TRUE(b->WaitCopy(K(20 + n - 1), v.size(), &dst[0], deadline));
  EXPECT_EQ(dst, v);
}

TEST(NodeDedup, DeadFetcherIsTakenOver) {
  ShmGuard g("takeover");
  auto a = NodeDedup::Open(Opts(g.name));
  auto b = NodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  std::string dst(4096, '\0');
  uint64_t token = 0, takeover = 0;
  ASSERT_EQ(a->Claim(K(4), dst.size(), &dst[0], &token),
            NodeDedup::Role::kFetch);
  // a never publishes (simulated crash). After takeover_ms a peer claims the
  // fetch instead of waiting forever.
  std::this_thread::sleep_for(250ms);  // > takeover_ms(200)
  EXPECT_EQ(b->Claim(K(4), dst.size(), &dst[0], &takeover),
            NodeDedup::Role::kFetch);
}

TEST(NodeDedup, TtlExpiryRecyclesEntry) {
  ShmGuard g("ttl");
  auto o = Opts(g.name);
  o.ttl_ms = 100;
  auto a = NodeDedup::Open(o);
  ASSERT_TRUE(a);
  const std::string v = Val(5, 4096);
  std::string dst(v.size(), '\0');
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->Claim(K(5), v.size(), &dst[0], &token),
            NodeDedup::Role::kFetch);
  a->Publish(K(5), NodeDedup::Kind::kData, token, v.data(), v.size());
  ASSERT_EQ(a->Claim(K(5), v.size(), &dst[0], &ignored),
            NodeDedup::Role::kHit);
  std::this_thread::sleep_for(150ms);  // > ttl
  // Expired: a new arrival re-fetches instead of serving stale bytes.
  EXPECT_EQ(a->Claim(K(5), v.size(), &dst[0], &token),
            NodeDedup::Role::kFetch);
}

TEST(NodeDedup, LappedArenaNeverServesOverwrittenBytes) {
  ShmGuard g("lap");
  auto a = NodeDedup::Open(Opts(g.name, /*arena=*/1 << 20));  // 1 MiB ring
  ASSERT_TRUE(a);
  const std::string v = Val(6, 128 * 1024);
  std::string dst(v.size(), '\0');
  uint64_t token = 0;
  ASSERT_EQ(a->Claim(K(6), v.size(), &dst[0], &token),
            NodeDedup::Role::kFetch);
  a->Publish(K(6), NodeDedup::Kind::kData, token, v.data(), v.size());
  // Lap the ring: publish > arena_bytes of other payloads.
  for (uint64_t i = 100; i < 100 + 12; ++i) {
    const std::string w = Val(i, 128 * 1024);
    std::string tmp(w.size(), '\0');
    uint64_t other_token = 0;
    if (a->Claim(K(i), w.size(), &tmp[0], &other_token) ==
        NodeDedup::Role::kFetch)
      a->Publish(K(i), NodeDedup::Kind::kData, other_token, w.data(),
                 w.size());
  }
  // K(6)'s payload region has been overwritten: the entry must NOT hit with
  // stale bytes — any non-kHit outcome (re-fetch) is correct; a kHit MUST
  // still carry the exact original value (possible if the slot was recycled
  // and republished, which this workload doesn't do).
  std::string out(v.size(), '\1');
  uint64_t ignored = 0;
  if (a->Claim(K(6), v.size(), &out[0], &ignored) ==
      NodeDedup::Role::kHit) {
    EXPECT_EQ(out, v);
  }
}

TEST(NodeDedup, CrossProcessRendezvous) {
  ShmGuard g("fork");
  const std::string v = Val(7, 32768);
  auto parent = NodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(parent);
  std::string dst(v.size(), '\0');
  uint64_t token = 0;
  ASSERT_EQ(parent->Claim(K(7), v.size(), &dst[0], &token),
            NodeDedup::Role::kFetch);
  parent->Publish(K(7), NodeDedup::Kind::kData, token, v.data(), v.size());

  pid_t pid = ::fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {  // child: attach and hit
    auto child = NodeDedup::Open(Opts(g.name));
    if (!child) ::_exit(2);
    std::string cdst(v.size(), '\0');
    uint64_t ignored = 0;
    if (child->Claim(K(7), v.size(), &cdst[0], &ignored) !=
        NodeDedup::Role::kHit)
      ::_exit(3);
    if (cdst != v) ::_exit(4);
    ::_exit(0);
  }
  int status = 0;
  ASSERT_EQ(::waitpid(pid, &status, 0), pid);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0) << "child exit code";
}

TEST(NodeDedup, DisabledByDefaultFromEnv) {
  ::unsetenv("DFKV_CLIENT_NODE_DEDUP");
  EXPECT_EQ(NodeDedup::FromEnv(0x51), nullptr);
}

TEST(NodeDedup, AutoAcceptsFittingPayloadStrictDoesNot) {
  ShmGuard g("auto");
  auto a = NodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a);
  const std::string v = Val(8, 5000);  // "unfull" payload
  std::string dst(8192, '\0');
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->Claim(K(8), v.size(), &dst[0], &token),
            NodeDedup::Role::kFetch);
  a->Publish(K(8), NodeDedup::Kind::kData, token, v.data(), v.size());
  // Auto with a larger cap hits and reports the true length.
  size_t got = 0;
  ASSERT_EQ(a->ClaimAuto(K(8), 8192, &dst[0], &got, &ignored),
            NodeDedup::Role::kHit);
  EXPECT_EQ(got, v.size());
  EXPECT_EQ(std::string(dst.data(), got), v);
  // Auto with a too-small cap must NOT hit (falls back to fetch/reserve path).
  std::string small(1024, '\0');
  EXPECT_NE(a->ClaimAuto(K(8), 1024, &small[0], &got, &ignored),
            NodeDedup::Role::kHit);
  // Strict with a different n must NOT hit either.
  EXPECT_NE(a->Claim(K(8), 8192, &dst[0], &ignored),
            NodeDedup::Role::kHit);
}

TEST(NodeDedup, ExistRendezvousBothAnswers) {
  ShmGuard g("exist");
  auto a = NodeDedup::Open(Opts(g.name));
  auto b = NodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  bool val = false;
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->ClaimExist(K(9), &val, &token), NodeDedup::Role::kFetch);
  const char yes = 1;
  a->Publish(K(9), NodeDedup::Kind::kExist, token, &yes, 1);
  ASSERT_EQ(b->ClaimExist(K(9), &val, &ignored), NodeDedup::Role::kHit);
  EXPECT_TRUE(val);
  // Negative answers are published too (valid result, unlike a failed GET).
  ASSERT_EQ(a->ClaimExist(K(10), &val, &token), NodeDedup::Role::kFetch);
  const char no = 0;
  a->Publish(K(10), NodeDedup::Kind::kExist, token, &no, 1);
  ASSERT_EQ(b->ClaimExist(K(10), &val, &ignored), NodeDedup::Role::kHit);
  EXPECT_FALSE(val);
  // The exist namespace never collides with the data namespace of the same key.
  std::string dst(4096, '\0');
  EXPECT_EQ(a->Claim(K(9), dst.size(), &dst[0], &ignored),
            NodeDedup::Role::kFetch);
}


TEST(NodeDedup, InvalidateDropsReadyAndFencesLatePublisher) {
  ShmGuard g("invalidate");
  auto a = NodeDedup::Open(Opts(g.name));
  auto b = NodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  const std::string value = Val(11, 4096);
  std::string dst(value.size(), '\0');
  uint64_t token = 0, ignored = 0;

  ASSERT_EQ(a->Claim(K(11), value.size(), dst.data(), &token),
            NodeDedup::Role::kFetch);
  a->Publish(K(11), NodeDedup::Kind::kData, token, value.data(),
             value.size());
  a->Invalidate(K(11));
  EXPECT_NE(b->Claim(K(11), value.size(), dst.data(), &ignored),
            NodeDedup::Role::kHit);

  ASSERT_EQ(a->Claim(K(12), value.size(), dst.data(), &token),
            NodeDedup::Role::kFetch);
  a->Invalidate(K(12));
  // This is the remove-vs-fetch race: the pre-remove owner completes late.
  a->Publish(K(12), NodeDedup::Kind::kData, token, value.data(),
             value.size());
  EXPECT_NE(b->Claim(K(12), value.size(), dst.data(), &ignored),
            NodeDedup::Role::kHit);

  bool exists = false;
  ASSERT_EQ(a->ClaimExist(K(13), &exists, &token),
            NodeDedup::Role::kFetch);
  const char absent = 0;
  a->Publish(K(13), NodeDedup::Kind::kExist, token, &absent, 1);
  a->Invalidate(K(13));
  EXPECT_NE(b->ClaimExist(K(13), &exists, &ignored),
            NodeDedup::Role::kHit);
}
TEST(NodeDedup, EnvSegmentNameCarriesLayoutVersion) {
  // A layout bump must land in a FRESH segment name: v1.23.0 bumped the header
  // magic behind the same name, the mismatch check refused the stale v1.22
  // segment, and the feature silently disabled itself fleet-wide on upgrade.
  const std::string nm = NodeDedup::EnvSegmentName(0xABCD);
  EXPECT_NE(nm.find("/dfkv-dedup-v4-"), std::string::npos) << nm;
  EXPECT_NE(nm.find("000000000000abcd"), std::string::npos) << nm;
}

TEST(NodeDedup, SegmentIsEagerlyBackedNotSparse) {
  // tmpfs allocates lazily on write: a full /dev/shm would SIGBUS a publisher
  // mid-copy instead of degrading. Open() posix_fallocates the whole segment
  // at creation, converting that into a clean attach-time ENOSPC (nullptr).
  // Verify the backing is real: allocated blocks cover the full length.
  ShmGuard g("backed");
  auto a = NodeDedup::Open(Opts(g.name, /*arena=*/4 << 20));
  ASSERT_TRUE(a);
  int fd = ::shm_open(g.name.c_str(), O_RDONLY, 0600);
  ASSERT_GE(fd, 0);
  struct stat st{};
  ASSERT_EQ(::fstat(fd, &st), 0);
  ::close(fd);
  EXPECT_GE(static_cast<uint64_t>(st.st_blocks) * 512, static_cast<uint64_t>(st.st_size))
      << "segment is sparse: a full tmpfs would SIGBUS at publish time";
}

// K(id) and Neighbour(id) start their probe at the same slot of the 1024-slot
// test table.
BlockKey Neighbour(uint64_t id) { return K(id + 1024); }

// Leaves K(id) published (v_old) at its second probe position with the first
// one free again: the neighbour held the first position while K(id) was
// claimed and has since gone.
void PublishAtSecondPosition(NodeDedup* d, uint64_t id,
                             const std::string& v_old) {
  std::string dst(v_old.size(), '\0');
  uint64_t neighbour = 0, token = 0;
  ASSERT_EQ(d->Claim(Neighbour(id), dst.size(), &dst[0], &neighbour),
            NodeDedup::Role::kFetch);
  ASSERT_EQ(d->Claim(K(id), v_old.size(), &dst[0], &token),
            NodeDedup::Role::kFetch);
  d->Publish(K(id), NodeDedup::Kind::kData, token, v_old.data(),
             v_old.size());
  d->Abort(Neighbour(id), NodeDedup::Kind::kData, neighbour);
}

TEST(NodeDedup, InvalidateRemovesEveryEntryOfTheKey) {
  ShmGuard g("inv-all");
  auto a = NodeDedup::Open(Opts(g.name));
  auto b = NodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  const std::string v_old = Val(30, 4096);
  PublishAtSecondPosition(a.get(), 30, v_old);
  // A reader of another length cannot use that entry and fetches for itself.
  // Its claim takes the free first position: the key now has two entries.
  std::string big(8192, 'n');
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(b->Claim(K(30), big.size(), &big[0], &token),
            NodeDedup::Role::kFetch);
  b->Publish(K(30), NodeDedup::Kind::kData, token, big.data(), big.size());
  // The key is overwritten or removed on the server.
  a->Invalidate(K(30));
  std::string out(v_old.size(), '\0');
  EXPECT_NE(b->Claim(K(30), out.size(), &out[0], &ignored),
            NodeDedup::Role::kHit)
      << "served the entry from before the invalidation";
}

TEST(NodeDedup, InvalidateLeavesOtherKeysOfTheWindowAlone) {
  ShmGuard g("inv-other");
  auto a = NodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a);
  const std::string v = Val(33, 4096), w = Val(34, 4096);
  std::string out(v.size(), '\0');
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->Claim(K(33), v.size(), &out[0], &token),
            NodeDedup::Role::kFetch);
  a->Publish(K(33), NodeDedup::Kind::kData, token, v.data(), v.size());
  ASSERT_EQ(a->Claim(Neighbour(33), w.size(), &out[0], &token),
            NodeDedup::Role::kFetch);
  a->Publish(Neighbour(33), NodeDedup::Kind::kData, token, w.data(),
             w.size());
  a->Invalidate(K(33));
  EXPECT_NE(a->Claim(K(33), out.size(), &out[0], &ignored),
            NodeDedup::Role::kHit);
  ASSERT_EQ(a->Claim(Neighbour(33), out.size(), &out[0], &ignored),
            NodeDedup::Role::kHit);
  EXPECT_EQ(out, w);
}

TEST(NodeDedup, InvalidateRemovesTheEntryBehindAFetchInFlight) {
  ShmGuard g("inv-wait");
  auto a = NodeDedup::Open(Opts(g.name));
  auto b = NodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  const std::string v_old = Val(31, 4096);
  PublishAtSecondPosition(a.get(), 31, v_old);
  // A reader of another length fetches the key for itself; its claim takes
  // the free first position and is still in flight.
  std::string big(8192, '\0'), out(v_old.size(), '\0');
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->Claim(K(31), big.size(), &big[0], &token),
            NodeDedup::Role::kFetch);
  ASSERT_EQ(b->Claim(K(31), out.size(), &out[0], &ignored),
            NodeDedup::Role::kWait);
  // The key is overwritten while that fetch is in flight. The entry behind
  // it is well inside its TTL, so only the invalidation can keep it from the
  // waiter.
  a->Invalidate(K(31));
  EXPECT_FALSE(b->WaitCopy(K(31), out.size(), &out[0]))
      << "the waiter copied the entry from before the invalidation";
}

TEST(NodeDedup, WaiterDoesNotCopyAnExpiredEntry) {
  ShmGuard g("wait-ttl");
  auto o = Opts(g.name);
  o.ttl_ms = 100;
  auto a = NodeDedup::Open(o);
  auto b = NodeDedup::Open(o);
  ASSERT_TRUE(a && b);
  const std::string v_old = Val(32, 4096);
  PublishAtSecondPosition(a.get(), 32, v_old);
  std::this_thread::sleep_for(150ms);  // > ttl
  std::string out(v_old.size(), '\0');
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->Claim(K(32), out.size(), &out[0], &token),
            NodeDedup::Role::kFetch);
  ASSERT_EQ(b->Claim(K(32), out.size(), &out[0], &ignored),
            NodeDedup::Role::kWait);
  // The re-fetch fails. The expired entry behind it is what the waiter now
  // finds; a claimer would refuse it, and so must a waiter.
  a->Abort(K(32), NodeDedup::Kind::kData, token);
  EXPECT_FALSE(b->WaitCopy(K(32), out.size(), &out[0]));
}

TEST(NodeDedup, CopyOutRefusesASlotRecycledForAnotherKey) {
  ShmGuard g("recycle");
  auto a = NodeDedup::Open(Opts(g.name));
  auto b = NodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  const std::string v1 = Val(40, 4096), v2 = Val(41, 4096);
  std::string out(v1.size(), '\0');
  uint64_t token = 0;
  ASSERT_EQ(a->Claim(K(40), v1.size(), &out[0], &token),
            NodeDedup::Role::kFetch);
  a->Publish(K(40), NodeDedup::Kind::kData, token, v1.data(), v1.size());

  // A reader has looked the key up ...
  const auto slot =
      dfkv::NodeDedupTestAccess::Find(*a, K(40), NodeDedup::Kind::kData);
  ASSERT_NE(slot, nullptr);
  // ... and before it copies, the entry goes and another process publishes a
  // different key of the same length into the same slot.
  b->Invalidate(K(40));
  ASSERT_EQ(b->Claim(Neighbour(40), v2.size(), &out[0], &token),
            NodeDedup::Role::kFetch);
  b->Publish(Neighbour(40), NodeDedup::Kind::kData, token, v2.data(),
             v2.size());
  ASSERT_EQ(dfkv::NodeDedupTestAccess::Find(*a, Neighbour(40),
                                            NodeDedup::Kind::kData),
            slot);

  EXPECT_FALSE(dfkv::NodeDedupTestAccess::CopyOut(
      *a, slot, K(40), NodeDedup::Kind::kData, v1.size(), &out[0]))
      << "copied another key's bytes";
  // The slot's present owner is still readable through it.
  EXPECT_TRUE(dfkv::NodeDedupTestAccess::CopyOut(
      *a, slot, Neighbour(40), NodeDedup::Kind::kData, v2.size(), &out[0]));
  EXPECT_EQ(out, v2);
}

// ---- publisher side: a late Publish/Abort must act only on its own claim --

TEST(NodeDedup, PublishRefusesASlotReclaimedAfterItsTokenCheck) {
  ShmGuard g("pub-gap");
  auto a = NodeDedup::Open(NoTakeover(g.name));
  auto b = NodeDedup::Open(NoTakeover(g.name));
  ASSERT_TRUE(a && b);
  const std::string v_old = Val(50, 4096), v_b = Val(51, 4096);
  std::string out(v_old.size(), '\0');
  uint64_t token_a = 0, token_b = 0, ignored = 0;
  ASSERT_EQ(a->Claim(K(50), out.size(), &out[0], &token_a),
            NodeDedup::Role::kFetch);
  // The publisher has checked its token ...
  const auto slot = dfkv::NodeDedupTestAccess::OwnedSlot(*a, token_a);
  ASSERT_NE(slot, nullptr);
  // ... and before it takes the slot, the key is invalidated and another
  // process claims the same slot for another key.
  b->Invalidate(K(50));
  ASSERT_EQ(b->Claim(Neighbour(50), out.size(), &out[0], &token_b),
            NodeDedup::Role::kFetch);
  dfkv::NodeDedupTestAccess::PublishInto(*a, slot, K(50),
                                         NodeDedup::Kind::kData, token_a,
                                         v_old);
  EXPECT_EQ(b->Claim(Neighbour(50), out.size(), &out[0], &ignored),
            NodeDedup::Role::kWait)
      << "the late publish went into another key's claim";
  // The slot's real owner is unaffected.
  b->Publish(Neighbour(50), NodeDedup::Kind::kData, token_b, v_b.data(),
             v_b.size());
  ASSERT_EQ(a->Claim(Neighbour(50), out.size(), &out[0], &ignored),
            NodeDedup::Role::kHit);
  EXPECT_EQ(out, v_b);
}

TEST(NodeDedup, AbortRefusesASlotReclaimedAfterItsTokenCheck) {
  ShmGuard g("abort-gap");
  auto a = NodeDedup::Open(Opts(g.name));
  auto b = NodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  const std::string v_b = Val(53, 4096);
  std::string out(v_b.size(), '\0');
  uint64_t token_a = 0, token_b = 0, ignored = 0;
  ASSERT_EQ(a->Claim(K(52), out.size(), &out[0], &token_a),
            NodeDedup::Role::kFetch);
  const auto slot = dfkv::NodeDedupTestAccess::OwnedSlot(*a, token_a);
  ASSERT_NE(slot, nullptr);
  b->Invalidate(K(52));
  ASSERT_EQ(b->Claim(Neighbour(52), out.size(), &out[0], &token_b),
            NodeDedup::Role::kFetch);
  dfkv::NodeDedupTestAccess::AbortInto(*a, slot, K(52),
                                       NodeDedup::Kind::kData, token_a);
  // The other key's fetch is still in flight and still publishable.
  b->Publish(Neighbour(52), NodeDedup::Kind::kData, token_b, v_b.data(),
             v_b.size());
  ASSERT_EQ(a->Claim(Neighbour(52), out.size(), &out[0], &ignored),
            NodeDedup::Role::kHit)
      << "the late abort dropped another key's claim";
  EXPECT_EQ(out, v_b);
}

// A token is good for the slot it was issued for only. Generations are small
// per-slot counters, so the slot a key has moved to can carry the very value
// an older claim of the key holds.
TEST(NodeDedup, LatePublishDoesNotFollowTheKeyToAnotherSlot) {
  ShmGuard g("token-slot");
  auto a = NodeDedup::Open(NoTakeover(g.name));
  auto b = NodeDedup::Open(NoTakeover(g.name));
  ASSERT_TRUE(a && b);
  const std::string v_old = Val(54, 4096), v_new = Val(55, 4096);
  std::string out(v_old.size(), '\0');
  uint64_t neighbour = 0, token_a = 0, token_b = 0, ignored = 0;
  // The neighbour holds the first position: the key lives at the second.
  ASSERT_EQ(a->Claim(Neighbour(54), out.size(), &out[0], &neighbour),
            NodeDedup::Role::kFetch);
  // One claim-and-invalidate round on the second position, then the claim
  // this test is about: it is that slot's third generation step.
  ASSERT_EQ(a->Claim(K(54), out.size(), &out[0], &ignored),
            NodeDedup::Role::kFetch);
  a->Invalidate(K(54));
  ASSERT_EQ(a->Claim(K(54), out.size(), &out[0], &token_a),
            NodeDedup::Role::kFetch);
  // The neighbour goes, comes back and goes again: the first position is
  // free and two steps along.
  a->Abort(Neighbour(54), NodeDedup::Kind::kData, neighbour);
  ASSERT_EQ(a->Claim(Neighbour(54), out.size(), &out[0], &neighbour),
            NodeDedup::Role::kFetch);
  a->Abort(Neighbour(54), NodeDedup::Kind::kData, neighbour);
  // The key is overwritten, and the next fetch of it claims the first
  // position, which thereby reaches the generation the old claim holds.
  b->Invalidate(K(54));
  ASSERT_EQ(b->Claim(K(54), out.size(), &out[0], &token_b),
            NodeDedup::Role::kFetch);
  // The old fetch completes late with the bytes from before the overwrite.
  a->Publish(K(54), NodeDedup::Kind::kData, token_a, v_old.data(),
             v_old.size());
  EXPECT_EQ(a->Claim(K(54), out.size(), &out[0], &ignored),
            NodeDedup::Role::kWait)
      << "bytes from before the invalidation were published";
  b->Publish(K(54), NodeDedup::Kind::kData, token_b, v_new.data(),
             v_new.size());
  ASSERT_EQ(a->Claim(K(54), out.size(), &out[0], &ignored),
            NodeDedup::Role::kHit);
  EXPECT_EQ(out, v_new);
}

// ---- a publisher that stops inside Publish --------------------------------

// Runs Invalidate on its own thread; true if it returned within the bound.
struct InvalidateInBackground {
  std::future<void> done;
  InvalidateInBackground(NodeDedup* d, BlockKey key)
      : done(std::async(std::launch::async, [d, key] { d->Invalidate(key); })) {}
  bool ReturnedWithin(std::chrono::milliseconds bound) {
    return done.wait_for(bound) == std::future_status::ready;
  }
};

TEST(NodeDedup, InvalidateDoesNotWaitForAKilledPublisher) {
  ShmGuard g("pub-dead");
  auto a = NodeDedup::Open(Opts(g.name));
  auto b = NodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  const std::string v_old = Val(56, 4096);
  std::string out(v_old.size(), '\0');
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->Claim(K(56), out.size(), &out[0], &token),
            NodeDedup::Role::kFetch);
  // The publisher takes its slot and is killed before it commits.
  const auto held = dfkv::NodeDedupTestAccess::PublishBegin(
      *a, K(56), NodeDedup::Kind::kData, token);
  ASSERT_TRUE(held.taken);

  InvalidateInBackground inv(b.get(), K(56));
  const bool returned = inv.ReturnedWithin(2s);
  EXPECT_TRUE(returned) << "a write or remove of this key would never return";
  if (!returned)  // let the thread go so the test can end
    dfkv::NodeDedupTestAccess::PublishFinish(*a, held, v_old);
  inv.done.wait();
  // The slot is usable again.
  EXPECT_EQ(b->Claim(K(56), out.size(), &out[0], &ignored),
            NodeDedup::Role::kFetch);
}

TEST(NodeDedup, InvalidateFencesAPublisherThatIsStillCopying) {
  ShmGuard g("pub-slow");
  auto a = NodeDedup::Open(Opts(g.name));
  auto b = NodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  const std::string v_old = Val(57, 4096), v_new = Val(58, 4096);
  std::string out(v_old.size(), '\0');
  uint64_t token_a = 0, token_b = 0, ignored = 0;
  ASSERT_EQ(a->Claim(K(57), out.size(), &out[0], &token_a),
            NodeDedup::Role::kFetch);
  const auto held = dfkv::NodeDedupTestAccess::PublishBegin(
      *a, K(57), NodeDedup::Kind::kData, token_a);
  ASSERT_TRUE(held.taken);

  // The key is overwritten while the publisher copies, and fetched again.
  InvalidateInBackground inv(b.get(), K(57));
  const bool returned = inv.ReturnedWithin(2s);
  EXPECT_TRUE(returned);
  if (returned) {
    ASSERT_EQ(b->Claim(K(57), out.size(), &out[0], &token_b),
              NodeDedup::Role::kFetch);
    b->Publish(K(57), NodeDedup::Kind::kData, token_b, v_new.data(),
               v_new.size());
  }
  // The publisher was alive all along and now commits: nothing may come of it.
  dfkv::NodeDedupTestAccess::PublishFinish(*a, held, v_old);
  inv.done.wait();
  if (returned) {
    ASSERT_EQ(a->Claim(K(57), out.size(), &out[0], &ignored),
              NodeDedup::Role::kHit);
    EXPECT_EQ(out, v_new);
  }
}

TEST(NodeDedup, ClaimTakesOverFromAKilledPublisher) {
  ShmGuard g("pub-dead-claim");
  auto a = NodeDedup::Open(Opts(g.name));
  auto b = NodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  std::string out(4096, '\0');
  uint64_t token = 0, taken_over = 0;
  ASSERT_EQ(a->Claim(K(59), out.size(), &out[0], &token),
            NodeDedup::Role::kFetch);
  const auto held = dfkv::NodeDedupTestAccess::PublishBegin(
      *a, K(59), NodeDedup::Kind::kData, token);
  ASSERT_TRUE(held.taken);
  std::this_thread::sleep_for(250ms);  // > takeover_ms(200)
  EXPECT_EQ(b->Claim(K(59), out.size(), &out[0], &taken_over),
            NodeDedup::Role::kFetch)
      << "every later lookup of this key would wait out its deadline";
}

// A publisher of an older release commits with unconditional stores: taking
// its slot away would let those stores land on whoever owns the slot next.
// It is waited for, as before.
TEST(NodeDedup, PublisherOfAnOlderReleaseIsNeverForcedOut) {
  ShmGuard g("pub-old");
  auto a = NodeDedup::Open(Opts(g.name));
  auto b = NodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b);
  const std::string v_old = Val(60, 4096);
  std::string out(v_old.size(), '\0');
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->Claim(K(60), out.size(), &out[0], &token),
            NodeDedup::Role::kFetch);
  const auto held = dfkv::NodeDedupTestAccess::PublishBegin(
      *a, K(60), NodeDedup::Kind::kData, token);
  ASSERT_TRUE(held.taken);
  dfkv::NodeDedupTestAccess::LookLikeAnOlderRelease(*a, held);

  std::this_thread::sleep_for(250ms);  // > takeover_ms(200)
  EXPECT_EQ(b->Claim(K(60), out.size(), &out[0], &ignored),
            NodeDedup::Role::kWait);
  InvalidateInBackground inv(b.get(), K(60));
  EXPECT_FALSE(inv.ReturnedWithin(150ms));
  // It commits; the invalidation then removes what it published.
  dfkv::NodeDedupTestAccess::PublishFinish(*a, held, v_old);
  inv.done.wait();
  EXPECT_NE(b->Claim(K(60), out.size(), &out[0], &ignored),
            NodeDedup::Role::kHit);
}

// Two invalidations of one key meet at a publisher's slot. Only one may take
// it: the other must not act on the slot until the first is through, or its
// stores land on whatever the slot has become.
TEST(NodeDedup, TwoInvalidationsDoNotBothTakeAPublishersSlot) {
  ShmGuard g("pub-two-forcers");
  auto a = NodeDedup::Open(Opts(g.name));
  auto b = NodeDedup::Open(Opts(g.name));
  auto c = NodeDedup::Open(Opts(g.name));
  ASSERT_TRUE(a && b && c);
  const std::string v_new = Val(62, 4096);
  std::string out(v_new.size(), '\0');
  uint64_t token_a = 0, token_c = 0, ignored = 0;
  ASSERT_EQ(a->Claim(K(61), out.size(), &out[0], &token_a),
            NodeDedup::Role::kFetch);
  const auto held = dfkv::NodeDedupTestAccess::PublishBegin(
      *a, K(61), NodeDedup::Kind::kData, token_a);
  ASSERT_TRUE(held.taken);

  // The first invalidation has won the slot and not yet released it.
  ASSERT_TRUE(dfkv::NodeDedupTestAccess::ForceOutBegin(*b, held));
  // The second arrives.
  InvalidateInBackground second(c.get(), K(61));
  const bool second_acted = second.ReturnedWithin(150ms);
  EXPECT_FALSE(second_acted) << "a second invalidation took the slot as well";
  // If it did, the slot is free and the next fetch of the key claims it ...
  if (second_acted) {
    ASSERT_EQ(c->Claim(K(61), out.size(), &out[0], &token_c),
              NodeDedup::Role::kFetch);
  }
  // ... before the first invalidation finishes.
  dfkv::NodeDedupTestAccess::ForceOutFinish(*b, held);
  second.done.wait();
  if (!second_acted) {
    ASSERT_EQ(c->Claim(K(61), out.size(), &out[0], &token_c),
              NodeDedup::Role::kFetch);
  }
  c->Publish(K(61), NodeDedup::Kind::kData, token_c, v_new.data(),
             v_new.size());
  ASSERT_EQ(a->Claim(K(61), out.size(), &out[0], &ignored),
            NodeDedup::Role::kHit)
      << "the claim made after the invalidations was wiped";
  EXPECT_EQ(out, v_new);
}

// The takeover age of a PUBLISHING slot is time spent publishing. A fetch
// that itself took longer than takeover_ms is not taken away the moment its
// owner starts to copy.
TEST(NodeDedup, SlowFetchIsNotForcedOutDuringItsCopy) {
  ShmGuard g("pub-slow-fetch");
  auto a = NodeDedup::Open(NoTakeover(g.name));  // nobody steals the fetch
  auto o = Opts(g.name);
  o.takeover_ms = 500;
  o.ttl_ms = 60000;
  auto b = NodeDedup::Open(o);
  ASSERT_TRUE(a && b);
  const std::string v = Val(63, 4096);
  std::string out(v.size(), '\0');
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->Claim(K(63), out.size(), &out[0], &token),
            NodeDedup::Role::kFetch);
  std::this_thread::sleep_for(600ms);  // the fetch: > b's takeover_ms
  const auto held = dfkv::NodeDedupTestAccess::PublishBegin(
      *a, K(63), NodeDedup::Kind::kData, token);
  ASSERT_TRUE(held.taken);
  EXPECT_EQ(b->Claim(K(63), out.size(), &out[0], &ignored),
            NodeDedup::Role::kWait)
      << "a copy that had only just begun was taken away";
  dfkv::NodeDedupTestAccess::PublishFinish(*a, held, v);
  ASSERT_EQ(b->Claim(K(63), out.size(), &out[0], &ignored),
            NodeDedup::Role::kHit);
  EXPECT_EQ(out, v);
}

// A stale Publish or Abort of another process holds the slot for the moment
// of its re-check. The owner's publish, arriving in that moment, must still
// go through.
TEST(NodeDedup, OwnerPublishWaitsOutAStaleRecheck) {
  ShmGuard g("pub-retry");
  auto a = NodeDedup::Open(NoTakeover(g.name));
  auto b = NodeDedup::Open(NoTakeover(g.name));
  ASSERT_TRUE(a && b);
  const std::string v = Val(64, 4096);
  std::string out(v.size(), '\0');
  uint64_t token = 0, ignored = 0;
  ASSERT_EQ(a->Claim(K(64), out.size(), &out[0], &token),
            NodeDedup::Role::kFetch);
  const auto slot = dfkv::NodeDedupTestAccess::OwnedSlot(*a, token);
  ASSERT_NE(slot, nullptr);
  ASSERT_TRUE(dfkv::NodeDedupTestAccess::StaleLock(*b, slot));
  // Far beyond any scheduling delay of the thread below; 50 ms otherwise.
  dfkv::NodeDedupTestAccess::SetLockWait(*a, 10s);
  std::thread stale([&] {
    std::this_thread::sleep_for(2ms);
    dfkv::NodeDedupTestAccess::StaleUnlock(*b, slot);
  });
  a->Publish(K(64), NodeDedup::Kind::kData, token, v.data(), v.size());
  stale.join();
  ASSERT_EQ(b->Claim(K(64), out.size(), &out[0], &ignored),
            NodeDedup::Role::kHit)
      << "the owner gave up and its fetch was never published";
  EXPECT_EQ(out, v);
}
