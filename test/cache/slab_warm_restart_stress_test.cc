// Kill the store in the middle of concurrent PUT / REMOVE / read traffic, start
// it again on the same boot, and check every key against what the killed
// process had been told. This is the property that makes it safe to keep
// slots.tbl after an unclean exit:
//
//   - a key that is readable holds exactly the bytes of a PUT that was issued
//     for it -- never another key's or another version's;
//   - an acknowledged PUT is still there, an acknowledged REMOVE is still gone.
//
// Each key has one writer, plus a remover thread that removes keys of all
// writers, so a PUT and a REMOVE of the same key do overlap. The killed process
// reports "about to" and "returned" for every operation through a pipe, which
// survives it. From that report the test derives, per key, every state a
// linearizable store could be in -- an operation takes effect somewhere between
// its two reports, and one that never returned may or may not have -- and the
// state found after the restart has to be one of them.
//
// Variants: buffered or O_DIRECT payload writes (batches go through io_uring
// when the build has it), and a roomy or an undersized store. Undersized means
// eviction, extent steals and watermark reclaim run under the traffic; a key
// may then also be absent at any time, but what is present must still be right.
//
// Most restarts are inspected by a process that exits without shutting down,
// so the next run again starts on a kept table: kept, killed, kept, ... A kill
// that lands inside a table sync finds the trust token taken back; that
// restart resets, and then every key has to be gone.
//
// DFKV_SLAB_KILL_ITERATIONS sets the number of kills per variant (default: a
// few seconds' worth in total; soak runs use thousands).
#include "cache/disk_slab_store.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace {
namespace fs = std::filesystem;
using dfkv::BlockKey;
using dfkv::DiskSlabStore;
using dfkv::Status;

constexpr int kWriters = 4;
constexpr uint32_t kKeysPerWriter = 48;
constexpr uint32_t kKeys = kWriters * kKeysPerWriter;
constexpr uint32_t kForeignBytes = 0xFFFFFFFFu;  // read back something else

struct Variant {
  bool direct;
  bool undersized;
};

BlockKey Key(uint32_t id) { return BlockKey{0x5eed0000ull + id, id}; }

// Roomy: two size classes. Undersized: thirteen, 16..64 KiB, so that classes
// compete for extents.
size_t SizeOf(uint32_t key, bool undersized) {
  return undersized ? 4096 * (4 + key % 13) : 4096 * (1 + key % 2);
}
// The content names its key and version in every word.
std::string Content(uint32_t key, uint32_t version, bool undersized) {
  std::string value(SizeOf(key, undersized), '\0');
  const uint64_t word = (static_cast<uint64_t>(key) << 32) | version;
  for (size_t at = 0; at + sizeof(word) <= value.size(); at += sizeof(word))
    std::memcpy(&value[at], &word, sizeof(word));
  return value;
}

enum : uint8_t { kPut = 1, kRemove = 2 };
enum : uint8_t { kAboutTo = 0, kReturned = 1, kStoreOpen = 2 };
enum : uint8_t { kWriter = 0, kRemover = 1, kActors = 2 };
struct Note {  // one line of the killed process's report
  uint8_t done;   // kAboutTo / kReturned for an operation, or kStoreOpen
  uint8_t op;
  uint8_t ok;     // meaningful when done
  uint8_t actor;  // a key sees at most one open operation per actor
  uint32_t key;      // kStoreOpen: 1 if the store kept its table at startup
  uint32_t version;
};
static_assert(sizeof(Note) == 12, "written to the pipe in one piece");

// 0 = absent, otherwise the version stored. Keep-first: a PUT onto a present
// key changes nothing.
uint32_t Apply(uint32_t state, const Note& note) {
  if (note.op == kRemove) return 0;
  return state != 0 ? state : note.version;
}

// Every state one key can be in, given the report so far. An entry pairs a
// state with the set of still-open operations already applied to reach it.
class KeyHistory {
 public:
  KeyHistory() { possible_.insert({0, 0}); }

  void set_evictable(bool evictable) { evictable_ = evictable; }

  void AboutTo(const Note& note) {
    open_[note.actor] = note;
    is_open_[note.actor] = true;
    Close();
  }
  // The operation returned: it has taken effect if it succeeded, and has not
  // if it failed. Drop every history that says otherwise.
  void Returned(const Note& note) {
    const uint8_t bit = static_cast<uint8_t>(1u << note.actor);
    std::set<std::pair<uint32_t, uint8_t>> kept;
    for (const auto& [state, applied] : possible_)
      if (((applied & bit) != 0) == (note.ok != 0))
        kept.insert({state, static_cast<uint8_t>(applied & ~bit)});
    possible_ = std::move(kept);
    is_open_[note.actor] = false;
    Close();
  }
  bool Allows(uint32_t state) const {
    for (const auto& entry : possible_)
      if (entry.first == state) return true;
    return false;
  }
  std::string Describe() const {
    std::set<uint32_t> states;
    for (const auto& entry : possible_) states.insert(entry.first);
    std::string text;
    for (uint32_t state : states) text += std::to_string(state) + " ";
    return text;
  }
  // After the restart the key is in `state` and nothing is open any more.
  void Settle(uint32_t state) {
    possible_.clear();
    possible_.insert({state, 0});
    is_open_[kWriter] = is_open_[kRemover] = false;
    Close();
  }

 private:
  // An open operation may take effect at any moment, in any order; in an
  // undersized store a key may also be evicted at any moment.
  void Close() {
    for (bool grew = true; grew;) {
      grew = false;
      const auto snapshot = possible_;
      for (const auto& [state, applied] : snapshot) {
        if (evictable_) grew |= possible_.insert({0, applied}).second;
        for (uint8_t actor = 0; actor < kActors; ++actor) {
          const uint8_t bit = static_cast<uint8_t>(1u << actor);
          if (!is_open_[actor] || (applied & bit) != 0) continue;
          grew |= possible_
                      .insert({Apply(state, open_[actor]),
                               static_cast<uint8_t>(applied | bit)})
                      .second;
        }
      }
    }
  }

  std::set<std::pair<uint32_t, uint8_t>> possible_;
  Note open_[kActors]{};
  bool is_open_[kActors] = {false, false};
  bool evictable_ = false;
};

DiskSlabStore::Options StressOptions(const fs::path& dir, Variant variant) {
  DiskSlabStore::Options options;
  options.dir = dir.string();
  options.direct_writes = variant.direct;  // falls back without O_DIRECT
  options.extent_bytes = 1ull << 20;
  options.slot_granularity = 4096;
  if (variant.undersized) {
    // About 7.7 MiB of keys in 6 MiB, thirteen classes: evictions and steals.
    options.capacity_bytes = 6ull << 20;
    options.reclaim_interval_ms = 5;
  } else {
    // 32 extents for two classes: nothing is ever evicted or stolen.
    options.capacity_bytes = 32ull << 20;
    options.reclaim_interval_ms = 0;
  }
  // Table syncs take the trust token back for their duration, and a kill
  // inside one costs a reset. Frequent enough to be hit, rare enough that most
  // restarts keep the table on a slow disk; DFKV_SLAB_KILL_SYNC_MS overrides.
  options.table_sync_ms = 20;
  if (const char* value = std::getenv("DFKV_SLAB_KILL_SYNC_MS"))
    options.table_sync_ms = static_cast<uint32_t>(std::max(1, std::atoi(value)));
  options.warm_restart = DiskSlabStore::Options::WarmRestart::kAuto;
  std::array<uint8_t, 16> boot_id;
  boot_id.fill(0x42);
  options.boot_id = boot_id;  // "same boot, same mount" on any kernel
  options.mount_id = 0x4242;
  options.trust_token_dir = fs::temp_directory_path().string();
  options.require_local_filesystem = false;
  return options;
}

void Report(int fd, uint8_t done, uint8_t op, bool ok, uint32_t key,
            uint32_t version, uint8_t actor = kWriter) {
  const Note note{done, op, static_cast<uint8_t>(ok), actor, key, version};
  if (::write(fd, &note, sizeof(note)) != static_cast<ssize_t>(sizeof(note)))
    ::_exit(3);
}

// The process that gets killed: never returns.
[[noreturn]] void RunTraffic(const DiskSlabStore::Options& options,
                             Variant variant, int report,
                             uint32_t first_version, unsigned seed) {
  bool ok = false;
  DiskSlabStore store(options, &ok);
  if (!ok) ::_exit(2);
  Report(report, kStoreOpen, 0, true,
         static_cast<uint32_t>(store.GetStats().warm_restarts), 0);
  std::vector<std::thread> threads;
  for (int writer = 0; writer < kWriters; ++writer) {
    threads.emplace_back([&, writer] {
      std::mt19937 rng(seed * 31 + static_cast<unsigned>(writer));
      const uint32_t base = static_cast<uint32_t>(writer) * kKeysPerWriter;
      uint32_t version = first_version;
      for (;;) {
        const uint32_t key = base + rng() % kKeysPerWriter;
        if (rng() % 4 == 0) {
          Report(report, kAboutTo, kRemove, false, key, 0);
          const Status status = store.Remove(Key(key));
          Report(report, kReturned, kRemove,
                 status == Status::kOk || status == Status::kNotFound, key, 0);
          continue;
        }
        if (rng() % 2 == 0) {
          ++version;
          const std::string value = Content(key, version, variant.undersized);
          Report(report, kAboutTo, kPut, false, key, version);
          const Status status =
              store.Cache(Key(key), value.data(), value.size());
          Report(report, kReturned, kPut, status == Status::kOk, key, version);
          continue;
        }
        // A batch of this writer's keys from page-aligned buffers: the path
        // the RAM tier flushes through (O_DIRECT, one io_uring submit).
        const uint32_t count = 1 + rng() % 3;
        struct Item {
          uint32_t key, version;
          char* buffer;
        };
        std::vector<Item> batch;
        std::vector<DiskSlabStore::CacheBatchItem> items;
        for (uint32_t i = 0; i < count; ++i) {
          const uint32_t member = base + (key - base + i * 7) % kKeysPerWriter;
          bool repeated = false;
          for (const Item& earlier : batch) repeated |= earlier.key == member;
          if (repeated) continue;
          ++version;
          const std::string value =
              Content(member, version, variant.undersized);
          char* buffer =
              static_cast<char*>(std::aligned_alloc(4096, value.size()));
          std::memcpy(buffer, value.data(), value.size());
          batch.push_back({member, version, buffer});
          items.push_back({Key(member), buffer, value.size(), value.size()});
        }
        for (const Item& item : batch)
          Report(report, kAboutTo, kPut, false, item.key, item.version);
        const std::vector<Status> statuses = store.CacheDirectBatch(items);
        for (size_t i = 0; i < batch.size(); ++i) {
          Report(report, kReturned, kPut, statuses[i] == Status::kOk,
                 batch[i].key, batch[i].version);
          std::free(batch[i].buffer);
        }
      }
    });
  }
  // Removes that race the writers' PUTs of the same keys.
  threads.emplace_back([&] {
    std::mt19937 rng(seed * 977 + 5);
    for (;;) {
      const uint32_t key = rng() % kKeys;
      Report(report, kAboutTo, kRemove, false, key, 0, kRemover);
      const Status status = store.Remove(Key(key));
      Report(report, kReturned, kRemove,
             status == Status::kOk || status == Status::kNotFound, key, 0,
             kRemover);
      std::this_thread::sleep_for(std::chrono::microseconds(rng() % 200));
    }
  });
  // Readers hold slots across Removes, which is what defers a removal.
  for (int reader = 0; reader < 2; ++reader) {
    threads.emplace_back([&, reader] {
      std::mt19937 rng(seed * 131 + static_cast<unsigned>(reader));
      for (;;) {
        dfkv::ReadLease lease;
        if (store.RangeDirectPrep(Key(rng() % kKeys), 0, 0, 1 << 20, &lease) ==
            Status::kOk)
          std::this_thread::sleep_for(std::chrono::microseconds(rng() % 300));
      }
    });
  }
  for (;;) ::pause();
}

// What a restart finds.
struct Inspection {
  uint32_t opened = 0;         // the store came up
  uint32_t kept = 0;           // ... on a kept table
  uint32_t token_gone = 0;     // ... or reset because the trust token was gone
  uint32_t direct_active = 0;  // ... writing payloads with O_DIRECT
  uint32_t actual[kKeys];      // 0 absent, kForeignBytes, or the version
};

// Opens the store and reads every key. With `result_pipe` >= 0 this is a
// process of its own that sends the result and exits WITHOUT shutting the
// store down, so the next run starts on a kept table again.
void Inspect(const DiskSlabStore::Options& traffic_options, Variant variant,
             Inspection* result, int result_pipe) {
  // No background work: this process only reads, and when it exits without
  // shutting down it must not be in the middle of a table sync of its own.
  DiskSlabStore::Options options = traffic_options;
  options.table_sync_ms = 0;
  options.reclaim_interval_ms = 0;
  bool ok = false;
  DiskSlabStore store(options, &ok);
  result->opened = ok;
  if (ok) {
    result->kept = store.GetStats().warm_restarts == 1;
    result->token_gone = store.UncleanResetReason().rfind(
                             "the previous run's trust token is gone", 0) == 0;
    result->direct_active = store.DirectWritesActive();
    for (uint32_t key = 0; key < kKeys; ++key) {
      std::string found;
      const Status read = store.Range(Key(key), 0, 0, &found);
      uint32_t actual = 0;
      if (read == Status::kOk) {
        uint64_t word = 0;
        if (found.size() >= sizeof(word))
          std::memcpy(&word, found.data(), sizeof(word));
        actual = static_cast<uint32_t>(word);
        if (static_cast<uint32_t>(word >> 32) != key || actual == 0 ||
            found != Content(key, actual, variant.undersized))
          actual = kForeignBytes;
      } else if (read != Status::kNotFound) {
        actual = kForeignBytes;
      }
      result->actual[key] = actual;
    }
  }
  if (result_pipe < 0) return;  // in-process: `store` shuts down cleanly
  const ssize_t n = ::write(result_pipe, result, sizeof(*result));
  ::_exit(n == static_cast<ssize_t>(sizeof(*result)) ? 0 : 4);
}

void KillUnderTraffic(Variant variant) {
  int iterations = 30;
  if (const char* value = std::getenv("DFKV_SLAB_KILL_ITERATIONS"))
    iterations = std::max(1, std::atoi(value));
  const fs::path dir =
      fs::temp_directory_path() /
      ("dfkv_slab_kill_" + std::to_string(::getpid()) +
       (variant.direct ? "_direct" : "_buffered") +
       (variant.undersized ? "_undersized" : "_roomy"));
  fs::remove_all(dir);
  const DiskSlabStore::Options options = StressOptions(dir, variant);

  std::vector<KeyHistory> history(kKeys);  // every key starts absent
  for (KeyHistory& key_history : history)
    key_history.set_evictable(variant.undersized);
  std::mt19937 rng(20261122);
  uint64_t operations = 0, failed_puts = 0, verified = 0, kept_restarts = 0;
  uint64_t resets = 0;
  bool direct_active = false;
  bool last_exit_unclean = false;  // how the previous process on `dir` ended
  for (int iteration = 0; iteration < iterations; ++iteration) {
    SCOPED_TRACE("kill #" + std::to_string(iteration));
    int report[2];
    ASSERT_EQ(::pipe(report), 0);
    const pid_t child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
      ::close(report[0]);
      // Versions never repeat across kills.
      RunTraffic(options, variant, report[1],
                 static_cast<uint32_t>(iteration + 1) * 1000000u,
                 static_cast<unsigned>(iteration));
    }
    ::close(report[1]);

    // Drain the report while the traffic runs; kill at a random moment after
    // the first PUT succeeded (so the kill lands in traffic, not in the
    // store's start), then read what is left in the pipe.
    std::vector<Note> notes;
    const auto kill_delay = std::chrono::milliseconds(rng() % 40);
    auto kill_at = std::chrono::steady_clock::time_point::max();
    bool killed = false;
    std::string pending;
    for (;;) {
      pollfd waiter{report[0], POLLIN, 0};
      ::poll(&waiter, 1, killed ? -1 : 1);
      if (!killed && std::chrono::steady_clock::now() >= kill_at) {
        ASSERT_EQ(::kill(child, SIGKILL), 0);
        killed = true;
      }
      if ((waiter.revents & (POLLIN | POLLHUP)) == 0) continue;
      char buffer[4096];
      const ssize_t n = ::read(report[0], buffer, sizeof(buffer));
      ASSERT_GE(n, 0);
      if (n == 0) break;  // every writer of the pipe is gone
      pending.append(buffer, static_cast<size_t>(n));
      while (pending.size() >= sizeof(Note)) {
        Note note;
        std::memcpy(&note, pending.data(), sizeof(note));
        pending.erase(0, sizeof(note));
        if (note.done == kStoreOpen) {
          ASSERT_EQ(note.key != 0, last_exit_unclean)
              << "after an unclean exit the traffic process must start on a "
                 "kept table, after a clean one on a clean start";
          continue;
        }
        if (note.done == kReturned && note.op == kPut && note.ok != 0 &&
            kill_at == std::chrono::steady_clock::time_point::max())
          kill_at = std::chrono::steady_clock::now() + kill_delay;
        notes.push_back(note);
      }
    }
    ::close(report[0]);
    int status = 0;
    ASSERT_EQ(::waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL)
        << "the traffic process ended on its own, status " << status;

    // Replay the report into the per-key histories. Operations that never
    // returned stay open: both of their outcomes remain possible.
    for (const Note& note : notes) {
      ASSERT_LT(note.key, kKeys);
      ASSERT_LT(note.actor, kActors);
      if (note.done == kAboutTo) {
        history[note.key].AboutTo(note);
        continue;
      }
      if (!note.ok) {
        // Only a PUT into an undersized store may fail (every slot pinned).
        ASSERT_TRUE(variant.undersized && note.op == kPut)
            << "operation failed in the traffic process, op " << int(note.op)
            << " key " << note.key;
        ++failed_puts;
      }
      history[note.key].Returned(note);
      ++operations;
    }

    // Restart. Usually in a process that then exits without shutting down;
    // every fifth time here, with a clean shutdown.
    Inspection found;
    std::memset(&found, 0, sizeof(found));
    const bool inspect_here = iteration % 5 == 4;
    if (inspect_here) {
      Inspect(options, variant, &found, -1);
    } else {
      int result[2];
      ASSERT_EQ(::pipe(result), 0);
      const pid_t inspector = ::fork();
      ASSERT_GE(inspector, 0);
      if (inspector == 0) {
        ::close(result[0]);
        Inspect(options, variant, &found, result[1]);
      }
      ::close(result[1]);
      size_t got = 0;
      while (got < sizeof(found)) {
        const ssize_t n = ::read(result[0], reinterpret_cast<char*>(&found) + got,
                                 sizeof(found) - got);
        if (n <= 0) break;
        got += static_cast<size_t>(n);
      }
      ::close(result[0]);
      int inspector_status = 0;
      ASSERT_EQ(::waitpid(inspector, &inspector_status, 0), inspector);
      ASSERT_TRUE(WIFEXITED(inspector_status) &&
                  WEXITSTATUS(inspector_status) == 0 && got == sizeof(found))
          << "the inspecting process failed";
    }
    last_exit_unclean = !inspect_here;
    ASSERT_TRUE(found.opened) << "the store did not start after the kill";
    direct_active = found.direct_active != 0;
    // A kill that lands inside a table sync finds the trust token taken back,
    // and the restart resets: everything is gone, which is always allowed. Any
    // other reason to reset -- records that contradict each other, say -- is
    // a finding.
    ASSERT_TRUE(found.kept || found.token_gone)
        << "the table was reset, and not because the trust token was gone";
    if (found.kept) ++kept_restarts; else ++resets;
    for (uint32_t key = 0; key < kKeys; ++key) {
      const uint32_t actual = found.actual[key];
      ASSERT_NE(actual, kForeignBytes)
          << "key " << key << " returned torn bytes or another key's bytes";
      if (!found.kept) {
        ASSERT_EQ(actual, 0u) << "key " << key << " survived a reset";
        history[key].Settle(0);
        continue;
      }
      ASSERT_TRUE(history[key].Allows(actual))
          << "key " << key << ": found version " << actual
          << " (0 = absent); the report only allows: "
          << history[key].Describe();
      history[key].Settle(actual);  // the cut-off operations are now decided
      if (actual != 0) ++verified;
    }
  }
  fs::remove_all(dir);
  std::printf("%s payload writes (O_DIRECT active: %s), %s store: kills=%d "
              "kept=%llu reset_in_table_sync=%llu returned_operations=%llu "
              "failed_puts=%llu objects_verified=%llu\n",
              variant.direct ? "direct" : "buffered",
              direct_active ? "yes" : "no",
              variant.undersized ? "undersized" : "roomy", iterations,
              static_cast<unsigned long long>(kept_restarts),
              static_cast<unsigned long long>(resets),
              static_cast<unsigned long long>(operations),
              static_cast<unsigned long long>(failed_puts),
              static_cast<unsigned long long>(verified));
  EXPECT_GT(kept_restarts, 0u) << "no restart ever kept its table";
  EXPECT_GT(verified, 0u) << "no object was ever read back and checked";
}

TEST(SlabWarmRestartStress, KilledUnderBufferedTrafficKeepsWhatItAcknowledged) {
  KillUnderTraffic({/*direct=*/false, /*undersized=*/false});
}

TEST(SlabWarmRestartStress, KilledUnderDirectTrafficKeepsWhatItAcknowledged) {
  KillUnderTraffic({/*direct=*/true, /*undersized=*/false});
}

TEST(SlabWarmRestartStress, KilledWhileEvictingBufferedReturnsOnlyRightBytes) {
  KillUnderTraffic({/*direct=*/false, /*undersized=*/true});
}

TEST(SlabWarmRestartStress, KilledWhileEvictingDirectReturnsOnlyRightBytes) {
  KillUnderTraffic({/*direct=*/true, /*undersized=*/true});
}

}  // namespace
