// Kill the store in the middle of concurrent PUT / REMOVE / read traffic, start
// it again on the same boot, and check every key against what the killed
// process had been told. This is the property that makes it safe to keep
// slots.tbl after an unclean exit:
//
//   - a key that is readable holds exactly the bytes of a PUT that was issued
//     for it -- never another key's or another version's;
//   - an acknowledged PUT is still there, an acknowledged REMOVE is still gone.
//
// Each key is driven by one thread, so its history is a sequence in which only
// the last operation can have been cut off by the kill; both outcomes of that
// one are accepted. The killed process reports "about to" and "done" for every
// operation through a pipe, which survives it.
//
// Runs once with buffered payload writes and once with O_DIRECT (where batches
// go through io_uring when the build has it). DFKV_SLAB_KILL_ITERATIONS sets
// the number of kills per mode (default: a few seconds' worth; soak runs use
// thousands).
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
#include <map>
#include <random>
#include <string>
#include <thread>
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

BlockKey Key(uint32_t id) { return BlockKey{0x5eed0000ull + id, id}; }

// Two size classes; the content names its key and version in every word.
size_t SizeOf(uint32_t key) { return 4096 * (1 + key % 2); }
std::string Content(uint32_t key, uint32_t version) {
  std::string value(SizeOf(key), '\0');
  const uint64_t word = (static_cast<uint64_t>(key) << 32) | version;
  for (size_t at = 0; at + sizeof(word) <= value.size(); at += sizeof(word))
    std::memcpy(&value[at], &word, sizeof(word));
  return value;
}

enum : uint8_t { kPut = 1, kRemove = 2 };
enum : uint8_t { kAboutTo = 0, kReturned = 1, kStoreOpen = 2 };
struct Note {  // one line of the killed process's report
  uint8_t done;  // kAboutTo / kReturned for an operation, or kStoreOpen
  uint8_t op;
  uint8_t ok;    // meaningful when done
  uint8_t pad;
  uint32_t key;
  uint32_t version;
};
static_assert(sizeof(Note) == 12, "written to the pipe in one piece");

// 0 = absent, otherwise the version stored. Keep-first: a PUT onto a present
// key changes nothing.
uint32_t Apply(uint32_t state, const Note& note) {
  if (note.op == kRemove) return 0;
  return state != 0 ? state : note.version;
}

DiskSlabStore::Options StressOptions(const fs::path& dir, bool direct) {
  DiskSlabStore::Options options;
  options.dir = dir.string();
  options.direct_writes = direct;  // falls back where O_DIRECT is unsupported
  options.capacity_bytes = 8ull << 20;  // far more than the keys need: no eviction
  options.extent_bytes = 1ull << 20;
  options.slot_granularity = 4096;
  options.reclaim_interval_ms = 0;
  options.warm_restart = DiskSlabStore::Options::WarmRestart::kAuto;
  std::array<uint8_t, 16> boot_id;
  boot_id.fill(0x42);
  options.boot_id = boot_id;  // "same boot, same mount" on any kernel
  options.mount_id = 0x4242;
  return options;
}

void Report(int fd, uint8_t done, uint8_t op, bool ok, uint32_t key,
            uint32_t version) {
  const Note note{done, op, static_cast<uint8_t>(ok), 0, key, version};
  if (::write(fd, &note, sizeof(note)) != static_cast<ssize_t>(sizeof(note)))
    ::_exit(3);
}

// The process that gets killed: never returns.
[[noreturn]] void RunTraffic(const DiskSlabStore::Options& options, int report,
                             uint32_t first_version, unsigned seed) {
  bool ok = false;
  DiskSlabStore store(options, &ok);
  if (!ok) ::_exit(2);
  Report(report, kStoreOpen, 0, true, 0, 0);
  std::vector<std::thread> threads;
  for (int writer = 0; writer < kWriters; ++writer) {
    threads.emplace_back([&, writer] {
      std::mt19937 rng(seed * 31 + static_cast<unsigned>(writer));
      uint32_t version = first_version;
      for (;;) {
        const uint32_t key = static_cast<uint32_t>(writer) * kKeysPerWriter +
                             rng() % kKeysPerWriter;
        if (rng() % 3 == 0) {
          Report(report, kAboutTo, kRemove, false, key, 0);
          const Status status = store.Remove(Key(key));
          Report(report, kReturned, kRemove,
                 status == Status::kOk || status == Status::kNotFound, key, 0);
          continue;
        }
        if (rng() % 2 == 0) {
          ++version;
          const std::string value = Content(key, version);
          Report(report, kAboutTo, kPut, false, key, version);
          const Status status =
              store.Cache(Key(key), value.data(), value.size());
          Report(report, kReturned, kPut, status == Status::kOk, key, version);
          continue;
        }
        // A batch of this writer's keys from page-aligned buffers: the path
        // the RAM tier flushes through (O_DIRECT, one io_uring submit).
        const uint32_t base = static_cast<uint32_t>(writer) * kKeysPerWriter;
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
          const std::string value = Content(member, version);
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

void KillUnderTraffic(bool direct) {
  int iterations = 40;
  if (const char* value = std::getenv("DFKV_SLAB_KILL_ITERATIONS"))
    iterations = std::max(1, std::atoi(value));
  const fs::path dir = fs::temp_directory_path() /
                       ("dfkv_slab_kill_" + std::to_string(::getpid()) +
                        (direct ? "_direct" : "_buffered"));
  fs::remove_all(dir);
  const DiskSlabStore::Options options = StressOptions(dir, direct);

  std::vector<uint32_t> expected(kKeys, 0);  // what the store must hold now
  std::mt19937 rng(20261122);
  uint64_t operations = 0, kept_objects = 0;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    SCOPED_TRACE("kill #" + std::to_string(iteration));
    int report[2];
    ASSERT_EQ(::pipe(report), 0);
    const pid_t child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
      ::close(report[0]);
      // Versions never repeat across kills.
      RunTraffic(options, report[1],
                 static_cast<uint32_t>(iteration + 1) * 1000000u,
                 static_cast<unsigned>(iteration));
    }
    ::close(report[1]);

    // Drain the report while the traffic runs, kill at a random moment after
    // the store came up (so the state on disk is this run's), then read what
    // is left in the pipe.
    std::vector<Note> notes;
    const auto kill_delay = std::chrono::milliseconds(1 + rng() % 40);
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
        if (note.done == kStoreOpen)
          kill_at = std::chrono::steady_clock::now() + kill_delay;
        else
          notes.push_back(note);
      }
    }
    ::close(report[0]);
    int status = 0;
    ASSERT_EQ(::waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL)
        << "the traffic process ended on its own, status " << status;

    // Replay the report: per key, the state after every acknowledged operation
    // and, if one was cut off, the state had it completed as well.
    std::vector<uint32_t> also_legal = expected;
    std::map<uint32_t, Note> cut_off;
    for (const Note& note : notes) {
      ASSERT_LT(note.key, kKeys);
      if (note.done == kAboutTo) {
        cut_off[note.key] = note;
        continue;
      }
      ASSERT_TRUE(note.ok) << "operation failed in the traffic process, op "
                           << int(note.op) << " key " << note.key;
      expected[note.key] = Apply(expected[note.key], note);
      cut_off.erase(note.key);
      ++operations;
    }
    also_legal = expected;
    for (const auto& [key, note] : cut_off)
      also_legal[key] = Apply(expected[key], note);

    bool ok = false;
    DiskSlabStore store(options, &ok);
    ASSERT_TRUE(ok) << store.StartupError();
    ASSERT_EQ(store.GetStats().warm_restarts, 1u)
        << "the table was reset instead of kept";
    for (uint32_t key = 0; key < kKeys; ++key) {
      std::string found;
      const Status read = store.Range(Key(key), 0, 0, &found);
      ASSERT_TRUE(read == Status::kOk || read == Status::kNotFound);
      uint32_t actual = 0;
      if (read == Status::kOk) {
        ASSERT_GE(found.size(), sizeof(uint64_t));
        uint64_t word = 0;
        std::memcpy(&word, found.data(), sizeof(word));
        ASSERT_EQ(static_cast<uint32_t>(word >> 32), key)
            << "key " << key << " returned another key's bytes";
        actual = static_cast<uint32_t>(word);
        ASSERT_EQ(found, Content(key, actual))
            << "key " << key << " returned torn or foreign bytes";
        ++kept_objects;
      }
      ASSERT_TRUE(actual == expected[key] || actual == also_legal[key])
          << "key " << key << ": found version " << actual << ", expected "
          << expected[key] << " (or " << also_legal[key]
          << " if the cut-off operation completed)";
      expected[key] = actual;  // the cut-off operation is now decided
    }
    // `store` shuts down cleanly here; the next kill starts from this state.
  }
  fs::remove_all(dir);
  std::printf("%s payload writes: kills=%d acknowledged_operations=%llu "
              "objects_verified=%llu\n",
              direct ? "direct" : "buffered", iterations,
              static_cast<unsigned long long>(operations),
              static_cast<unsigned long long>(kept_objects));
  EXPECT_GT(operations, 0u);
}

TEST(SlabWarmRestartStress, KilledUnderBufferedTrafficKeepsWhatItAcknowledged) {
  KillUnderTraffic(/*direct=*/false);
}

TEST(SlabWarmRestartStress, KilledUnderDirectTrafficKeepsWhatItAcknowledged) {
  KillUnderTraffic(/*direct=*/true);
}

}  // namespace
