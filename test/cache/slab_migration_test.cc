#include "cache/slab_migration.h"

#include <gtest/gtest.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "cache/disk_slab_store.h"
#include "cache/log_store.h"
#include "cache/lse_format.h"
#include "utils/crc32.h"
#include "utils/crc32c.h"
#include "utils/net_util.h"
#include "utils/sha256.h"

namespace {

namespace fs = std::filesystem;
using dfkv::BlockKey;
using dfkv::DiskSlabStore;
using dfkv::LogStore;
using dfkv::Status;
using dfkv::slab_migration::MigrateToLse;
using dfkv::slab_migration::SlabSnapshot;
using dfkv::slab_migration::SourceEntry;

using Values = std::vector<std::pair<BlockKey, std::string>>;

std::string Digest(const std::string& bytes) {
  std::array<uint8_t, 32> digest{};
  dfkv::Sha256(bytes.data(), bytes.size(), digest.data());
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  for (uint8_t b : digest) {
    out.push_back(kHex[b >> 4]);
    out.push_back(kHex[b & 15]);
  }
  return out;
}

std::map<std::string, std::string> RootHashes(const fs::path& root) {
  std::map<std::string, std::string> out;
  for (const auto& item : fs::recursive_directory_iterator(root)) {
    if (!item.is_regular_file()) continue;
    std::ifstream in(item.path(), std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
    out[fs::relative(item.path(), root).string()] = Digest(bytes);
  }
  return out;
}

void ExpectRootUnchanged(const fs::path& root,
                         const std::map<std::string, std::string>& before) {
  const auto after = RootHashes(root);
  ASSERT_EQ(after.size(), before.size());
  for (const auto& item : before) {
    const auto found = after.find(item.first);
    ASSERT_NE(found, after.end()) << item.first;
    EXPECT_EQ(found->second, item.second) << item.first;
  }
}

std::vector<uint8_t> ReadBytes(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
}

void WriteBytes(const fs::path& path, const std::vector<uint8_t>& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  ASSERT_TRUE(out.good());
}

std::string ReadText(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

size_t CountSubstring(const std::string& text, const std::string& needle) {
  size_t count = 0;
  for (size_t at = 0; (at = text.find(needle, at)) != std::string::npos;
       at += needle.size())
    ++count;
  return count;
}

fs::path ArmInterposer(const fs::path& root, const std::string& scenario,
                       size_t target_read_bytes = 0) {
  const fs::path log = root / "migration-io-events.log";
  WriteBytes(log, {});
  EXPECT_EQ(::setenv("DFKV_MIGRATION_IO_SCENARIO", scenario.c_str(), 1), 0);
  const std::string target = std::to_string(target_read_bytes);
  EXPECT_EQ(::setenv("DFKV_MIGRATION_IO_TARGET_READ_BYTES", target.c_str(), 1), 0);
  EXPECT_EQ(::setenv("DFKV_MIGRATION_IO_INTERPOSE_LOG", log.c_str(), 1), 0);
  EXPECT_EQ(::unsetenv("DFKV_MIGRATION_IO_WATCH"), 0);
  return log;
}

void PrintInterposerEvidence(const std::string& scenario, const fs::path& log) {
  const std::string evidence = ReadText(log);
  std::cout << "MIGRATION_IO_EVIDENCE_BEGIN " << scenario << "\n"
            << evidence << "MIGRATION_IO_EVIDENCE_END " << scenario << std::endl;
}

bool ReexecuteWithInterposer(const char* exact_filter) {
  const char* active = ::getenv("DFKV_MIGRATION_IO_INTERPOSE_ACTIVE");
  if (active != nullptr && std::string(active) == "1") return false;
  char executable[4096];
  const ssize_t length =
      ::readlink("/proc/self/exe", executable, sizeof(executable) - 1);
  if (length <= 0) {
    EXPECT_NE(length, 0) << "cannot resolve the test executable for scoped preload";
    return true;
  }
  executable[length] = '\0';
  const pid_t child = ::fork();
  if (child < 0) {
    EXPECT_EQ(child, 0) << "cannot create scoped-preload test child";
    return true;
  }
  if (child == 0) {
    ::setenv("LD_PRELOAD", DFKV_MIGRATION_IO_INTERPOSER_PATH, 1);
    ::execl(executable, executable, "--gtest_color=no",
            (std::string("--gtest_filter=") + exact_filter).c_str(),
            static_cast<char*>(nullptr));
    std::perror("execl scoped migration interposer");
    ::_exit(126);
  }
  int status = 0;
  pid_t waited = 0;
  do {
    waited = ::waitpid(child, &status, 0);
  } while (waited < 0 && errno == EINTR);
  EXPECT_EQ(waited, child);
  EXPECT_TRUE(WIFEXITED(status)) << "scoped-preload child status=" << status;
  if (WIFEXITED(status)) EXPECT_EQ(WEXITSTATUS(status), 0);
  return true;
}

void ExpectMigratedRecords(const dfkv::slab_migration::MigrationResult& result,
                           const Values& values, const SlabSnapshot& source) {
  ASSERT_EQ(result.records.size(), values.size());
  ASSERT_EQ(source.entries().size(), values.size());
  std::map<std::string, const std::pair<BlockKey, std::string>*> expected;
  std::map<std::string, const SourceEntry*> source_entries;
  for (const auto& value : values)
    ASSERT_TRUE(expected.emplace(value.first.Filename(), &value).second);
  for (const auto& entry : source.entries())
    ASSERT_TRUE(source_entries.emplace(entry.key.Filename(), &entry).second);
  for (const auto& record : result.records) {
    const auto found = expected.find(record.key.Filename());
    ASSERT_NE(found, expected.end()) << "duplicate/foreign migration record "
                                     << record.key.Filename();
    const auto source_found = source_entries.find(record.key.Filename());
    ASSERT_NE(source_found, source_entries.end()) << record.key.Filename();
    EXPECT_EQ(record.key.tenant_hash, found->second->first.tenant_hash);
    EXPECT_EQ(record.key.digest_hi, found->second->first.digest_hi);
    EXPECT_EQ(record.key.digest_lo, found->second->first.digest_lo);
    EXPECT_EQ(record.value_len, found->second->second.size());
    EXPECT_EQ(record.value_sha256, Digest(found->second->second));
    EXPECT_TRUE(record.source_has_payload_crc);
    EXPECT_EQ(record.source_has_payload_crc, source_found->second->has_payload_crc);
    EXPECT_EQ(record.source_payload_crc, source_found->second->payload_crc);
    EXPECT_EQ(record.source_slot_charge, source_found->second->slot_size);
    EXPECT_EQ(record.payload_crc,
              dfkv::Crc32c(found->second->second.data(), found->second->second.size()));
    EXPECT_GT(record.generation, 0u);
    EXPECT_GE(record.write_lsn, record.generation);
    const bool inline_record =
        dfkv::lse::RecordFitsInline(found->second->second.size());
    dfkv::lse::RecordHeader shape{};
    shape.value_len = found->second->second.size();
    shape.flags = inline_record ? dfkv::lse::kRecordFlagInline : 0;
    EXPECT_EQ(record.footprint, dfkv::lse::RecordFootprint(shape));
    EXPECT_EQ((record.flags & dfkv::lse::kRecordFlagInline) != 0,
              inline_record);
    EXPECT_EQ(record.header_len, dfkv::lse::kRecordHeaderBytes);
    expected.erase(found);
    source_entries.erase(source_found);
  }
  EXPECT_TRUE(expected.empty()) << "at least one source key has no unique record";
  EXPECT_TRUE(source_entries.empty());
}

void ExpectEveryValue(LogStore* store, const Values& values) {
  ASSERT_EQ(store->Count(), values.size());
  std::set<std::string> keys;
  store->ForEachKey([&](const BlockKey& key) { keys.insert(key.Filename()); });
  ASSERT_EQ(keys.size(), values.size());
  for (const auto& value : values) {
    SCOPED_TRACE(value.first.Filename());
    EXPECT_EQ(keys.count(value.first.Filename()), 1u);
    std::string actual;
    size_t actual_len = 0;
    ASSERT_EQ(store->Range(value.first, 0, 0, &actual, &actual_len),
              Status::kOk);
    ASSERT_EQ(actual_len, value.second.size());
    EXPECT_EQ(actual.size(), value.second.size());
    EXPECT_EQ(actual, value.second);
    EXPECT_EQ(Digest(actual), Digest(value.second));
    dfkv::ValueMetadata metadata{};
    ASSERT_EQ(store->Lookup(value.first, &metadata), Status::kOk);
    EXPECT_EQ(metadata.value_len, value.second.size());
    EXPECT_GT(metadata.generation, 0u);
  }
}

size_t OpenFdCount() {
  size_t count = 0;
  for (const auto& ignored : fs::directory_iterator("/proc/self/fd")) {
    (void)ignored;
    ++count;
  }
  return count;
}

class ScopedFileLimit {
 public:
  explicit ScopedFileLimit(rlim_t limit) {
    if (::getrlimit(RLIMIT_NOFILE, &previous_) != 0) return;
    rlimit limited = previous_;
    limited.rlim_cur = limit;
    valid_ = ::setrlimit(RLIMIT_NOFILE, &limited) == 0;
  }

  ~ScopedFileLimit() {
    if (valid_) ::setrlimit(RLIMIT_NOFILE, &previous_);
  }

  bool valid() const { return valid_; }

 private:
  rlimit previous_{};
  bool valid_ = false;
};

class SlabMigrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ = fs::temp_directory_path() /
            ("dfkv_sto14_" + std::to_string(
                ::testing::UnitTest::GetInstance()->current_test_info()->line()));
    fs::remove_all(root_);
    fs::create_directories(root_);
    source_ = root_ / "source";
    destination_ = root_ / "candidate";
  }

  void TearDown() override {
    std::error_code ec;
    if (fs::exists(root_)) {
      for (fs::recursive_directory_iterator it(root_, ec), end; !ec && it != end;
           it.increment(ec)) {
        if (it->is_directory()) fs::permissions(it->path(), fs::perms::owner_all, ec);
        else fs::permissions(it->path(), fs::perms::owner_write, fs::perm_options::add, ec);
      }
      fs::permissions(root_, fs::perms::owner_all, ec);
    }
    fs::remove_all(root_);
  }

  DiskSlabStore::Options SourceOptions() const {
    DiskSlabStore::Options options;
    options.dir = source_.string();
    options.capacity_bytes = 32ull << 20;  // 32 independent 1 MiB STL3 extents.
    options.extent_bytes = 1ull << 20;
    options.slot_granularity = 1ull << 20;
    options.direct_writes = false;
    options.reclaim_interval_ms = 0;
    options.table_sync_ms = 0;
    options.trust_token_dir = (root_ / "tokens").string();
    fs::create_directories(options.trust_token_dir);
    return options;
  }

  void CreateSource(const Values& values, bool retain = false, bool remove_second = false) {
    bool ok = false;
    auto store = std::make_unique<DiskSlabStore>(SourceOptions(), &ok);
    ASSERT_TRUE(ok) << store->StartupError();
    for (const auto& value : values)
      ASSERT_EQ(store->Cache(value.first, value.second.data(), value.second.size()), Status::kOk);
    if (retain) {
      ASSERT_EQ(store->Retain(values.front().first, 60, 0, 1000),
                static_cast<uint8_t>(dfkv::SlabAllocator::RetainCode::kProtectedNew));
    }
    if (remove_second) {
      ASSERT_EQ(store->Remove(values[1].first), Status::kOk);
    }
    ASSERT_TRUE(store->Close());
    store.reset();
  }

  Values SampleValues() const {
    Values values;
    for (uint64_t i = 0; i < 4; ++i)
      values.emplace_back(BlockKey{0x100 + i, 0x1000 + i, 0xaaaa}, std::string(1, '\0'));
    for (uint64_t i = 0; i < 4; ++i) {
      std::string side(4096, static_cast<char>(31 + i));
      side[i] = static_cast<char>(i);
      values.emplace_back(BlockKey{0x200 + i, 0x2000 + i, 0xbbbb}, std::move(side));
    }
    // Same digest_hi as marker zero but a different digest_lo and tenant:
    // exact BlockKey identity is retained, not interpreted as an anchor.
    // digest_lo occupies rec[24:32): its high bits must not be mistaken for
    // reserved zeros. A non-default tenant makes full-tuple identity visible.
    values.emplace_back(BlockKey{0x100, 0xf123456789abcdefULL, 0x123456789abcdef0ULL},
                        std::string(1, '\x7f'));
    values.emplace_back(BlockKey{0x300, 0x3000, 0}, std::string(37, '\0'));
    return values;
  }

  std::unique_ptr<SlabSnapshot> OpenSource(const Values& values) {
    CreateSource(values);
    std::string error;
    auto source = SlabSnapshot::Open(source_.string(), &error);
    EXPECT_TRUE(source != nullptr) << error;
    return source;
  }

  void MakeSourceReadonly() {
    for (const auto& item : fs::recursive_directory_iterator(source_)) {
      if (item.is_directory()) fs::permissions(item.path(), fs::perms::owner_read | fs::perms::owner_exec,
                                               fs::perm_options::replace);
      else fs::permissions(item.path(), fs::perms::owner_read, fs::perm_options::replace);
    }
    fs::permissions(source_, fs::perms::owner_read | fs::perms::owner_exec,
                    fs::perm_options::replace);
  }

  fs::path root_;
  fs::path source_;
  fs::path destination_;
};

TEST_F(SlabMigrationTest, MigratesEveryOrdinaryValueFromReadonlySourceAndReopens) {
  const Values values = SampleValues();
  CreateSource(values);
  MakeSourceReadonly();  // protect before the inspection interface is opened.
  std::string open_error;
  auto source = SlabSnapshot::Open(source_.string(), &open_error);
  ASSERT_NE(source, nullptr) << open_error;
  ASSERT_EQ(source->entries().size(), values.size());
  const auto before = RootHashes(source_);
  auto result = MigrateToLse(*source, destination_.string());
  ASSERT_TRUE(result.ok) << result.error;
  ASSERT_TRUE(result.published);
  ASSERT_TRUE(result.source_unchanged);
  EXPECT_FALSE(fs::exists(destination_.string() + ".sto14-staging"));
  ExpectRootUnchanged(source_, before);
  EXPECT_EQ(result.records.size(), values.size());
  EXPECT_EQ(result.source_slot_charge, values.size() * (1ull << 20));
  EXPECT_EQ(result.source_table_bytes, 32 * 64);
  EXPECT_FALSE(result.destination_files.empty());
  std::map<std::string, std::string> source_manifest;
  for (const auto& file : result.source_before)
    ASSERT_TRUE(source_manifest.emplace(file.path, file.sha256).second) << file.path;
  EXPECT_EQ(source_manifest, before);

  for (const auto& record : result.records) {
    EXPECT_GT(record.generation, 0);
    EXPECT_GE(record.write_lsn, record.generation);
    EXPECT_EQ(record.value_sha256, Digest(std::find_if(values.begin(), values.end(),
        [&](const auto& kv) { return kv.first == record.key; })->second));
    if (record.value_len == 1) {
      EXPECT_EQ(record.flags & dfkv::lse::kRecordFlagInline, dfkv::lse::kRecordFlagInline);
      EXPECT_EQ(record.header_len, dfkv::lse::kRecordHeaderBytes);
      EXPECT_EQ(record.footprint, 4096);  // logical segment footprint, not st_blocks/RSS.
    }
  }

  // Promotion is followed by another independent open for service. All fields,
  // tenants and arbitrary bytes survive; LSE generations are new, not lineage.
  LogStore::Options options;
  options.dir = destination_.string();
  options.capacity_bytes = result.capacity_bytes;
  options.cleaner = false;
  bool ok = false;
  auto reopened = std::make_unique<LogStore>(options, &ok);
  ASSERT_TRUE(ok) << reopened->StartupError();
  EXPECT_EQ(reopened->Count(), values.size());
  for (const auto& kv : values) {
    std::string actual;
    size_t len = 0;
    ASSERT_EQ(reopened->Range(kv.first, 0, 0, &actual, &len), Status::kOk);
    EXPECT_EQ(len, kv.second.size());
    EXPECT_EQ(actual, kv.second);
  }
  ASSERT_TRUE(reopened->Close());
  reopened.reset();

  const auto output_before = RootHashes(destination_);
  auto repeated = MigrateToLse(*source, destination_.string());
  EXPECT_FALSE(repeated.ok);
  EXPECT_FALSE(repeated.published);
  EXPECT_NE(repeated.error.find("refusing to reuse"), std::string::npos);
  EXPECT_EQ(RootHashes(destination_), output_before);
  ExpectRootUnchanged(source_, before);
}

TEST_F(SlabMigrationTest, MissingOrDirtyEpochNeverBecomesColdResetMigration) {
  const Values values{{BlockKey{1, 2, 3}, "value"}};
  CreateSource(values);
  for (bool dirty : {false, true}) {
    if (dirty) {
      CreateSource(values);
      auto state = ReadBytes(source_ / "slab_state");
      dfkv::net::PutU32(reinterpret_cast<char*>(state.data()) + 16, 2);
      dfkv::net::PutU32(reinterpret_cast<char*>(state.data()) + 60,
                        dfkv::Crc32(state.data(), 60));
      WriteBytes(source_ / "slab_state", state);
    } else {
      fs::remove(source_ / "slab_state");
    }
    const auto before = RootHashes(source_);
    std::string error;
    auto source = SlabSnapshot::Open(source_.string(), &error);
    EXPECT_EQ(source, nullptr);
    EXPECT_FALSE(error.empty());
    EXPECT_FALSE(fs::exists(destination_));
    ExpectRootUnchanged(source_, before);
  }
}

TEST_F(SlabMigrationTest, CorruptDuplicateForeignRetentionAndHistoryRowsFailClosed) {
  enum class Damage { kCorrupt, kDuplicate, kForeign, kRetention, kHistory, kReservedTail };
  for (Damage damage : {Damage::kCorrupt, Damage::kDuplicate, Damage::kForeign,
                        Damage::kRetention, Damage::kHistory, Damage::kReservedTail}) {
    SCOPED_TRACE(static_cast<int>(damage));
    TearDown();
    SetUp();
    const Values values{{BlockKey{7, 8, 9}, "first"},
                        {BlockKey{70, 80, 90}, "second"}};
    CreateSource(values, damage == Damage::kRetention, damage == Damage::kHistory);
    switch (damage) {
      case Damage::kCorrupt: {
        auto table = ReadBytes(source_ / "slots.tbl");
        ASSERT_FALSE(table.empty());
        table[8] ^= 1;  // live bit changes without updating the table CRC.
        WriteBytes(source_ / "slots.tbl", table);
        break;
      }
      case Damage::kDuplicate: {
        auto table = ReadBytes(source_ / "slots.tbl");
        size_t first = table.size();
        for (size_t i = 0; i + 64 <= table.size(); i += 64) {
          if (dfkv::net::GetU32(reinterpret_cast<char*>(table.data()) + i) ==
                  0x334C5453u && table[i + 8] == 1) {
            first = i;
            break;
          }
        }
        ASSERT_NE(first, table.size());
        ASSERT_LE(first + 128, table.size());
        std::memcpy(table.data() + first + 64, table.data() + first, 64);
        WriteBytes(source_ / "slots.tbl", table);
        break;
      }
      case Damage::kForeign:
        WriteBytes(source_ / "lse_super", std::vector<uint8_t>(4096, 1));
        break;
      case Damage::kReservedTail: {
        auto table = ReadBytes(source_ / "slots.tbl");
        size_t row = 0;
        while (row + 64 <= table.size() &&
               !(dfkv::net::GetU32(reinterpret_cast<char*>(table.data()) + row) ==
                     0x334C5453u && table[row + 8] == 1))
          row += 64;
        ASSERT_LT(row + 64, table.size());
        table[row + 63] = 1;  // reserved tail byte, not any key or payload field.
        // Recompute the row CRC so rejection must come from the unknown byte,
        // not the ordinary corruption checksum check.
        dfkv::net::PutU32(reinterpret_cast<char*>(table.data()) + row + 4,
                          dfkv::Crc32(table.data() + row + 8, 64 - 8));
        WriteBytes(source_ / "slots.tbl", table);
        break;
      }
      case Damage::kRetention:
      case Damage::kHistory:
        break;
    }
    const auto before = RootHashes(source_);
    std::string error;
    auto source = SlabSnapshot::Open(source_.string(), &error);
    EXPECT_EQ(source, nullptr) << static_cast<int>(damage);
    EXPECT_FALSE(error.empty());
    if (damage == Damage::kReservedTail) {
      EXPECT_NE(error.find("reserved"), std::string::npos) << error;
    }
    EXPECT_FALSE(fs::exists(destination_));
    ExpectRootUnchanged(source_, before);
  }
  // Reset only the values manipulated by the final subcase for TearDown's
  // ordinary ownership; all input mutations above were test-owned copies.
}

TEST_F(SlabMigrationTest, RejectsMixedGeometryAndUnexpectedNonzeroZeroMagicRow) {
  const Values values{{BlockKey{4, 5, 6}, "one"}};
  CreateSource(values);
  auto table = ReadBytes(source_ / "slots.tbl");
  ASSERT_GE(table.size(), 64 * 32);
  table.back() = 1;  // zero magic but nonzero payload: never treat as absent.
  WriteBytes(source_ / "slots.tbl", table);
  std::string error;
  EXPECT_EQ(SlabSnapshot::Open(source_.string(), &error), nullptr);
  EXPECT_NE(error.find("zero-magic"), std::string::npos);

  CreateSource(values);
  auto meta = ReadBytes(source_ / "slab_meta");
  dfkv::net::PutU64(reinterpret_cast<char*>(meta.data()) + 16, 4096);
  WriteBytes(source_ / "slab_meta", meta);
  EXPECT_EQ(SlabSnapshot::Open(source_.string(), &error), nullptr);
  EXPECT_FALSE(fs::exists(destination_));
}

TEST_F(SlabMigrationTest, StagingWriteFailureIsMarkedAndNeverPublished) {
  const Values values = SampleValues();
  auto source = OpenSource(values);
  ASSERT_NE(source, nullptr);
  const auto before = RootHashes(source_);
  const size_t open_fds = OpenFdCount();
  dfkv::slab_migration::MigrationResult result;
  {
    ScopedFileLimit limit(open_fds + 2);
    // Both reserved descriptors fit, so the LSE startup fails instead of the marker.
    ASSERT_TRUE(limit.valid());
    result = MigrateToLse(*source, destination_.string());
  }
  EXPECT_FALSE(result.ok) << result.error;
  EXPECT_FALSE(result.published);
  EXPECT_FALSE(fs::exists(destination_));
  EXPECT_TRUE(fs::exists(destination_.string() + ".sto14-staging/STO14-MIGRATION-INCOMPLETE"))
      << result.error;
  ExpectRootUnchanged(source_, before);
  std::string error;
  EXPECT_TRUE(source->VerifyUnchanged(&error)) << error;
}

TEST_F(SlabMigrationTest, MigratesThousandsWithoutConsumingCleanerReserve) {
  constexpr size_t object_count = 7000;
  Values values;
  values.reserve(object_count);
  for (uint64_t i = 0; i < object_count; ++i)
    values.emplace_back(BlockKey{i, i + 7000, 0x90900},
                        std::string(1, static_cast<char>(i % 251)));
  auto options = SourceOptions();
  // 256 small slots per real 1 MiB extent: 7,000 objects occupy 28 MiB of
  // native slots, avoiding thousands of file descriptors while retaining the
  // same LSE 4 KiB/record and summary-pressure boundary.
  options.slot_granularity = 4096;
  bool ok = false;
  auto writer = std::make_unique<DiskSlabStore>(options, &ok);
  ASSERT_TRUE(ok) << writer->StartupError();
  for (const auto& value : values)
    ASSERT_EQ(writer->Cache(value.first, value.second.data(), value.second.size()), Status::kOk);
  ASSERT_TRUE(writer->Close());
  writer.reset();
  std::string error;
  auto source = SlabSnapshot::Open(source_.string(), &error);
  ASSERT_NE(source, nullptr) << error;
  auto result = MigrateToLse(*source, destination_.string());
  ASSERT_TRUE(result.ok) << result.error;
  EXPECT_TRUE(result.published);
  EXPECT_TRUE(result.source_unchanged);
  ASSERT_EQ(result.records.size(), values.size());
  EXPECT_EQ(result.segment_bytes, 4ull << 20);
  // Seven filled client segments (1,006, then 964 one-byte records) plus the
  // two mandatory cleaner-reserve segments: the old eight-segment estimate
  // cannot hold all 7,000 without reclaiming live data.
  EXPECT_EQ(result.segment_count, 9);
}

TEST_F(SlabMigrationTest, EmptyLiveWritesRemainInvalidAtTheNativeBoundary) {
  const Values values{{BlockKey{1, 2, 3}, "first"}};
  bool ok = false;
  auto store = std::make_unique<DiskSlabStore>(SourceOptions(), &ok);
  ASSERT_TRUE(ok);
  ASSERT_EQ(store->Cache(values.front().first, values.front().second.data(),
                         values.front().second.size()), Status::kOk);
  EXPECT_EQ(store->Cache(BlockKey{9, 9, 9}, "", 0), Status::kInvalid);
  ASSERT_TRUE(store->Close());
  store.reset();
  std::string error;
  auto source = SlabSnapshot::Open(source_.string(), &error);
  ASSERT_TRUE(source != nullptr) << error;
  EXPECT_EQ(source->entries().size(), 1);
}

TEST_F(SlabMigrationTest, RejectsCrcValidNoncanonicalSlotClass) {
  auto options = SourceOptions();
  options.extent_bytes = 4ull << 20;
  bool ok = false;
  auto writer = std::make_unique<DiskSlabStore>(options, &ok);
  ASSERT_TRUE(ok) << writer->StartupError();
  const BlockKey key{4, 5, 6};
  const std::string value(3ull << 20, 'x');
  ASSERT_EQ(writer->Cache(key, value.data(), value.size()), Status::kOk);
  ASSERT_TRUE(writer->Close());
  writer.reset();

  auto table = ReadBytes(source_ / "slots.tbl");
  size_t row = table.size();
  for (size_t offset = 0; offset + 64 <= table.size(); offset += 64) {
    if (dfkv::net::GetU32(reinterpret_cast<char*>(table.data()) + offset) == 0x334C5453u &&
        table[offset + 8] == 1) {
      row = offset;
      break;
    }
  }
  ASSERT_NE(row, table.size());
  const size_t records_per_extent = options.extent_bytes / options.slot_granularity;
  ASSERT_EQ((row / 64) % records_per_extent, 0);
  ASSERT_EQ(dfkv::net::GetU32(reinterpret_cast<char*>(table.data()) + row + 12),
            options.extent_bytes);
  ASSERT_EQ(dfkv::net::GetU32(reinterpret_cast<char*>(table.data()) + row + 32),
            value.size());
  // Its original canonical bucket is 4 MiB at physical slot zero. Altering
  // only slot_size to 3 MiB keeps slot bounds, granularity and payload bounds,
  // so rejection must reach the canonical-class check rather than any earlier
  // physical-slot guard. The 3 MiB row has valid row and payload CRCs.
  dfkv::net::PutU32(reinterpret_cast<char*>(table.data()) + row + 12,
                    static_cast<uint32_t>(3ull << 20));
  dfkv::net::PutU32(reinterpret_cast<char*>(table.data()) + row + 4,
                    dfkv::Crc32(table.data() + row + 8, 64 - 8));
  WriteBytes(source_ / "slots.tbl", table);
  const auto before = RootHashes(source_);
  std::string error;
  EXPECT_EQ(SlabSnapshot::Open(source_.string(), &error), nullptr);
  EXPECT_NE(error.find("noncanonical"), std::string::npos) << error;
  EXPECT_FALSE(fs::exists(destination_));
  ExpectRootUnchanged(source_, before);
}

TEST_F(SlabMigrationTest, ReservesTheMandatorySummaryBlockWhenChoosingSegments) {
  auto options = SourceOptions();
  options.extent_bytes = 8ull << 20;
  bool ok = false;
  auto writer = std::make_unique<DiskSlabStore>(options, &ok);
  ASSERT_TRUE(ok) << writer->StartupError();
  const BlockKey key{40, 41, 42};
  // Fits MaxRecordValue(4 MiB) = S - 3 blocks, but not the ordinary
  // LogStore::MaxValueBytes(4 MiB) = S - 4 blocks admission rule.
  const std::string value((4ull << 20) - 3ull * dfkv::lse::kBlockBytes, 'v');
  ASSERT_EQ(writer->Cache(key, value.data(), value.size()), Status::kOk);
  ASSERT_TRUE(writer->Close());
  writer.reset();

  std::string error;
  auto source = SlabSnapshot::Open(source_.string(), &error);
  ASSERT_NE(source, nullptr) << error;
  ASSERT_EQ(source->entries().size(), 1);
  const auto before = RootHashes(source_);
  auto result = MigrateToLse(*source, destination_.string());
  ASSERT_TRUE(result.ok) << result.error;
  ASSERT_TRUE(result.published);
  ASSERT_TRUE(result.source_unchanged);
  EXPECT_EQ(result.segment_bytes, 8ull << 20);
  EXPECT_EQ(result.segment_count, 8);
  ASSERT_EQ(result.records.size(), 1);
  EXPECT_EQ(result.records[0].value_len, value.size());
  EXPECT_EQ(result.records[0].value_sha256, Digest(value));
  ExpectRootUnchanged(source_, before);
}

TEST(SlabMigrationInventoryTest, AcceptsLexicalDirectoryOrderAcrossSixDigitExtents) {
  const std::vector<std::string> generation_order{
      "E00000", "E00001", "E00002", "E00010", "E99999", "E100000"};
  std::vector<std::string> lexical_order{
      "E00000", "E00001", "E00002", "E00010", "E100000", "E99999"};
  ASSERT_TRUE(std::is_sorted(lexical_order.begin(), lexical_order.end()));
  ASSERT_NE(lexical_order, generation_order);
  EXPECT_TRUE(dfkv::slab_migration::ExtentInventoryMatches(generation_order, lexical_order));
  lexical_order.pop_back();
  EXPECT_FALSE(dfkv::slab_migration::ExtentInventoryMatches(generation_order, lexical_order));
}

TEST_F(SlabMigrationTest, CandidateVerificationFailureStillMarksIncompleteUnderEmfile) {
  auto options = SourceOptions();
  options.capacity_bytes = options.extent_bytes;
  bool ok = false;
  auto writer = std::make_unique<DiskSlabStore>(options, &ok);
  ASSERT_TRUE(ok) << writer->StartupError();
  const BlockKey key{11, 12, 13};
  const std::string value = "small-value";
  ASSERT_EQ(writer->Cache(key, value.data(), value.size()), Status::kOk);
  ASSERT_TRUE(writer->Close());
  writer.reset();
  std::string error;
  auto source = SlabSnapshot::Open(source_.string(), &error);
  ASSERT_NE(source, nullptr) << error;
  const auto before = RootHashes(source_);
  const size_t open_fds = OpenFdCount();
  bool found_candidate_verification_failure = false;
  // Each bound is a distinct deterministic resource-limit scenario, not a
  // retry used to obtain a green result. Require a real read-only candidate
  // verification failure after the writer has completed. The bounded runs also
  // cover creator-startup errors, but do not pretend those are constructor
  // failures in the later verifier, or count a successful migration as proof.
  for (size_t extra = 2; extra <= 24 && !found_candidate_verification_failure; ++extra) {
    const fs::path candidate = root_ / ("fd-limited-" + std::to_string(extra));
    dfkv::slab_migration::MigrationResult result;
    {
      ScopedFileLimit limit(open_fds + extra);
      ASSERT_TRUE(limit.valid());
      result = MigrateToLse(*source, candidate.string());
    }
    std::fprintf(stderr, "extra_fd=%zu result_ok=%d published=%d error=%s\n", extra,
                 static_cast<int>(result.ok), static_cast<int>(result.published),
                 result.error.c_str());
    if (result.error.rfind("open candidate ", 0) != 0 ||
        result.error.find("Too many open files (errno 24)") == std::string::npos)
      continue;
    found_candidate_verification_failure = true;
    EXPECT_FALSE(result.ok) << result.error;
    EXPECT_FALSE(result.published) << result.error;
    EXPECT_TRUE(fs::exists(candidate.string() + ".sto14-staging/STO14-MIGRATION-INCOMPLETE"))
        << "extra_fd=" << extra << ": " << result.error;
  }
  ASSERT_TRUE(found_candidate_verification_failure)
      << "bounded FD scenarios did not reach read-only candidate verification with EMFILE";
  ExpectRootUnchanged(source_, before);
  EXPECT_TRUE(source->VerifyUnchanged(&error)) << error;
}

#ifdef DFKV_WITH_URING
TEST_F(SlabMigrationTest, DirectBatchProducerMigratesOfflineWithExactValues) {
  Values values;
  for (uint64_t i = 0; i < 12; ++i) {
    std::string value(3000 + i * 701, static_cast<char>(31 + i));
    for (size_t p = i; p < value.size(); p += 997)
      value[p] = static_cast<char>((i * 29 + p) % 256);
    values.emplace_back(BlockKey{0x4100 + i, 0x5100 + i * 3,
                                 0xd1ec7000 + i},
                        std::move(value));
  }

  auto options = SourceOptions();
  options.slot_granularity = 4096;
  options.direct_writes = true;
  ASSERT_EQ(::setenv("DFKV_SLAB_URING_WRITE", "1", 1), 0);
  bool ok = false;
  auto writer = std::make_unique<DiskSlabStore>(options, &ok);
  ASSERT_TRUE(ok) << writer->StartupError();
  ASSERT_TRUE(writer->DirectWritesActive())
      << "the runtime trial requires an O_DIRECT-capable source filesystem";

  using Aligned = std::unique_ptr<char, void (*)(void*)>;
  std::vector<Aligned> buffers;
  std::vector<DiskSlabStore::CacheBatchItem> items;
  for (const auto& value : values) {
    const size_t cap = (value.second.size() + 4095) & ~size_t{4095};
    void* memory = nullptr;
    ASSERT_EQ(::posix_memalign(&memory, 4096, cap), 0);
    Aligned buffer(static_cast<char*>(memory), &std::free);
    std::memcpy(buffer.get(), value.second.data(), value.second.size());
    items.push_back({value.first, buffer.get(), value.second.size(), cap});
    buffers.push_back(std::move(buffer));
  }
  const auto statuses = writer->CacheDirectBatch(items);
  ASSERT_EQ(statuses.size(), values.size());
  for (size_t i = 0; i < statuses.size(); ++i)
    ASSERT_EQ(statuses[i], Status::kOk) << i;
  const auto direct_stats = writer->GetStats();
  EXPECT_EQ(direct_stats.dio_write_fallbacks, 0u);
  ASSERT_GE(direct_stats.uring_write_batches, 1u)
      << "no actual io_uring batch was submitted";
  EXPECT_GE(direct_stats.batched_writes, values.size())
      << "not every eligible direct item completed on the CQE path";
  std::cout << "DIRECT_PRODUCER_EVIDENCE odirect_active=1 eligible_items="
            << values.size() << " uring_write_batches="
            << direct_stats.uring_write_batches << " cqe_full_writes="
            << direct_stats.batched_writes << " dio_fallbacks="
            << direct_stats.dio_write_fallbacks << std::endl;
  ASSERT_TRUE(writer->Close());
  writer.reset();
  buffers.clear();
  MakeSourceReadonly();

  std::string error;
  auto source = SlabSnapshot::Open(source_.string(), &error);
  ASSERT_NE(source, nullptr) << error;
  ASSERT_EQ(source->entries().size(), values.size());
  const auto before = RootHashes(source_);
  auto result = MigrateToLse(*source, destination_.string());
  ASSERT_TRUE(result.ok) << result.error;
  ASSERT_TRUE(result.published);
  ASSERT_TRUE(result.source_unchanged);
  ExpectRootUnchanged(source_, before);
  ExpectMigratedRecords(result, values, *source);
  EXPECT_TRUE(fs::exists(destination_));
  EXPECT_FALSE(fs::exists(result.staging_dir));
  std::cout << "DIRECT_PRODUCER_MIGRATION_RESULT "
            << dfkv::slab_migration::ResultJson(result) << std::endl;

  LogStore::Options reopen_options;
  reopen_options.dir = destination_.string();
  reopen_options.capacity_bytes = result.capacity_bytes;
  reopen_options.cleaner = false;
  reopen_options.preopen = false;
  reopen_options.open_segments = 1;
  auto reopened = std::make_unique<LogStore>(reopen_options, &ok);
  ASSERT_TRUE(ok) << reopened->StartupError();
  ExpectEveryValue(reopened.get(), values);
  ASSERT_TRUE(reopened->Close());
  reopened.reset();
  ExpectRootUnchanged(source_, before);
  ASSERT_TRUE(source->VerifyUnchanged(&error)) << error;
}
#endif  // DFKV_WITH_URING

TEST_F(SlabMigrationTest, RealSyscallShortIoRetriesAndTerminalFaultsFailClosed) {
  if (ReexecuteWithInterposer(
          "SlabMigrationTest.RealSyscallShortIoRetriesAndTerminalFaultsFailClosed"))
    return;
  ASSERT_EQ(std::string(::getenv("DFKV_MIGRATION_IO_INTERPOSE_ACTIVE") == nullptr
                            ? ""
                            : ::getenv("DFKV_MIGRATION_IO_INTERPOSE_ACTIVE")),
            "1")
      << "run this case with the bounded slab_migration_io_interposer preloaded";

  const auto make_values = [] {
    return Values{
        {BlockKey{0x6100, 0x7100, 0xf100}, std::string(5001, 'R')},
        {BlockKey{0x6101, 0x7101, 0xf101}, std::string(5002, 'S')},
        {BlockKey{0x6102, 0x7102, 0xf102}, std::string(37, 'T')},
        {BlockKey{0x6103, 0x7103, 0xf103}, std::string(9003, 'U')}};
  };
  const Values values = make_values();
  auto options = SourceOptions();
  options.slot_granularity = 4096;
  bool ok = false;
  auto initial = std::make_unique<DiskSlabStore>(options, &ok);
  ASSERT_TRUE(ok) << initial->StartupError();
  for (const auto& value : values)
    ASSERT_EQ(initial->Cache(value.first, value.second.data(), value.second.size()),
              Status::kOk);
  ASSERT_TRUE(initial->Close());
  initial.reset();
  std::string error;
  auto source = SlabSnapshot::Open(source_.string(), &error);
  ASSERT_NE(source, nullptr) << error;

  const auto before = RootHashes(source_);
  const fs::path retry_log = ArmInterposer(root_, "retry", values[1].second.size());
  auto result = MigrateToLse(*source, destination_.string());
  PrintInterposerEvidence("retry", retry_log);
  const std::string retry_evidence = ReadText(retry_log);
  EXPECT_EQ(CountSubstring(retry_evidence, "event=source_pread_eintr "), 1u);
  EXPECT_EQ(CountSubstring(retry_evidence, "event=source_pread_short "), 1u);
  EXPECT_EQ(CountSubstring(retry_evidence, "event=source_pread_retry_completed "), 1u);
  EXPECT_NE(retry_evidence.find("event=source_pread_short "), std::string::npos);
  EXPECT_NE(retry_evidence.find("bytes=5002 result=5001"), std::string::npos);
  EXPECT_NE(retry_evidence.find("event=source_pread_retry_completed "), std::string::npos);
  EXPECT_NE(retry_evidence.find("bytes=1 result=1"), std::string::npos);
  EXPECT_EQ(CountSubstring(retry_evidence, "event=destination_pwritev2_eintr "), 1u);
  EXPECT_EQ(CountSubstring(retry_evidence, "event=destination_pwritev2_short "), 1u);
  EXPECT_NE(retry_evidence.find("event=destination_pwritev2_short "), std::string::npos);
  EXPECT_NE(retry_evidence.find("bytes=9097 result=4097"), std::string::npos);
  EXPECT_NE(retry_evidence.find("event=destination_pwritev2_retry_completed "),
            std::string::npos);
  EXPECT_NE(retry_evidence.find("offset=8193 bytes=5000 result=5000"),
            std::string::npos);
  ASSERT_TRUE(result.ok) << result.error;
  EXPECT_TRUE(result.published);
  EXPECT_TRUE(result.source_unchanged);
  ExpectMigratedRecords(result, values, *source);
  ExpectRootUnchanged(source_, before);
  std::cout << "RETRY_MIGRATION_RESULT "
            << dfkv::slab_migration::ResultJson(result) << std::endl;
  LogStore::Options reopened_options;
  reopened_options.dir = destination_.string();
  reopened_options.capacity_bytes = result.capacity_bytes;
  reopened_options.cleaner = false;
  auto reopened = std::make_unique<LogStore>(reopened_options, &ok);
  ASSERT_TRUE(ok) << reopened->StartupError();
  ExpectEveryValue(reopened.get(), values);
  ASSERT_TRUE(reopened->Close());
  reopened.reset();

  for (const std::string& terminal : {std::string("source_eof"),
                                      std::string("destination_fail")}) {
    SCOPED_TRACE(terminal);
    TearDown();
    SetUp();
    const Values terminal_values = make_values();
    auto fresh_options = SourceOptions();
    fresh_options.slot_granularity = 4096;
    auto fresh = std::make_unique<DiskSlabStore>(fresh_options, &ok);
    ASSERT_TRUE(ok) << fresh->StartupError();
    for (const auto& value : terminal_values)
      ASSERT_EQ(fresh->Cache(value.first, value.second.data(), value.second.size()),
                Status::kOk);
    ASSERT_TRUE(fresh->Close());
    fresh.reset();
    source = SlabSnapshot::Open(source_.string(), &error);
    ASSERT_NE(source, nullptr) << error;
    const auto terminal_before = RootHashes(source_);
    const fs::path fault_log =
        ArmInterposer(root_, terminal, terminal_values[1].second.size());
    auto failed = MigrateToLse(*source, destination_.string());
    PrintInterposerEvidence(terminal, fault_log);
    const std::string evidence = ReadText(fault_log);
    EXPECT_GE(CountSubstring(evidence, "event=destination_committed_before_fault "), 1u)
        << "the terminal fault must follow at least one complete migrated record";
    if (terminal == "source_eof") {
      EXPECT_EQ(CountSubstring(evidence, "event=source_eof_injected "), 1u);
      EXPECT_EQ(CountSubstring(evidence, "event=destination_pwritev2_eio "), 0u)
          << "source EOF is not a vectored-write/CQE fault";
      EXPECT_NE(failed.error.find("cannot read source value"), std::string::npos)
          << failed.error;
    } else {
      EXPECT_EQ(CountSubstring(evidence, "event=destination_pwritev2_eio "), 1u);
      EXPECT_EQ(CountSubstring(evidence, "event=source_eof_injected "), 0u);
      EXPECT_NE(failed.error.find("ordinary LSE Cache rejected source value"),
                std::string::npos)
          << failed.error;
    }
    EXPECT_FALSE(failed.ok) << failed.error;
    EXPECT_FALSE(failed.published);
    EXPECT_FALSE(failed.source_unchanged);
    EXPECT_FALSE(fs::exists(destination_));
    EXPECT_EQ(failed.staging_dir, destination_.string() + ".sto14-staging");
    const fs::path marker = fs::path(failed.staging_dir) / "STO14-MIGRATION-INCOMPLETE";
    ASSERT_TRUE(fs::exists(marker)) << failed.error;
    EXPECT_EQ(ReadText(marker), failed.error);
    EXPECT_TRUE(fs::exists(fs::path(failed.staging_dir) / "segments"));
    ExpectRootUnchanged(source_, terminal_before);
    ASSERT_TRUE(source->VerifyUnchanged(&error)) << error;
    std::cout << "TERMINAL_MIGRATION_RESULT "
              << dfkv::slab_migration::ResultJson(failed) << std::endl;
  }
}

TEST_F(SlabMigrationTest, MixedLayoutMultiSegmentDirectReopenVerifiesAllRanges) {
  if (ReexecuteWithInterposer(
          "SlabMigrationTest.MixedLayoutMultiSegmentDirectReopenVerifiesAllRanges"))
    return;
  Values values;
  for (uint64_t i = 0; i < 128; ++i) {
    const size_t size = i % 3 == 0 ? 1 : (17 + (i * 37) % 3900);
    std::string value(size, static_cast<char>(i % 251));
    for (size_t p = 0; p < value.size(); p += 113)
      value[p] = static_cast<char>((i + p * 7) % 256);
    values.emplace_back(BlockKey{0x8100 + i, 0x9100 + i * 11,
                                 0xabc000 + i},
                        std::move(value));
  }
  for (uint64_t i = 0; i < 96; ++i) {
    const size_t size = 65539 + (i % 7) * 4093;
    std::string value(size, static_cast<char>(33 + i % 90));
    for (size_t p = i; p < value.size(); p += 4091)
      value[p] = static_cast<char>((i * 13 + p) % 256);
    values.emplace_back(BlockKey{0x8200 + i, 0x9200 + i * 13,
                                 0xabd000 + i},
                        std::move(value));
  }
  auto source_options = SourceOptions();
  source_options.slot_granularity = 4096;
  bool source_ok = false;
  auto source_writer = std::make_unique<DiskSlabStore>(source_options, &source_ok);
  ASSERT_TRUE(source_ok) << source_writer->StartupError();
  for (const auto& value : values)
    ASSERT_EQ(source_writer->Cache(value.first, value.second.data(), value.second.size()),
              Status::kOk);
  ASSERT_EQ(source_writer->Count(), values.size())
      << "the source fixture must not evict any mixed-layout input";
  ASSERT_TRUE(source_writer->Close());
  source_writer.reset();
  std::string error;
  auto source = SlabSnapshot::Open(source_.string(), &error);
  ASSERT_NE(source, nullptr) << error;
  ASSERT_EQ(source->entries().size(), values.size());
  const auto before = RootHashes(source_);
  auto result = MigrateToLse(*source, destination_.string());
  ASSERT_TRUE(result.ok) << result.error;
  ASSERT_TRUE(result.published);
  EXPECT_TRUE(result.source_unchanged);
  ExpectRootUnchanged(source_, before);
  ExpectMigratedRecords(result, values, *source);
  EXPECT_EQ(result.segment_bytes, 4ull << 20);
  std::map<uint32_t, size_t> records_per_segment;
  size_t inline_records = 0;
  size_t noninline_records = 0;
  for (const auto& record : result.records) {
    ++records_per_segment[record.segment];
    if ((record.flags & dfkv::lse::kRecordFlagInline) != 0)
      ++inline_records;
    else
      ++noninline_records;
  }
  EXPECT_GT(records_per_segment.size(), 1u);
  EXPECT_GT(inline_records, 0u);
  EXPECT_GT(noninline_records, 0u);
  size_t maximum_segment_cardinality = 0;
  for (const auto& segment : records_per_segment)
    maximum_segment_cardinality =
        std::max(maximum_segment_cardinality, segment.second);
  EXPECT_GT(maximum_segment_cardinality, 64u)
      << "at least one sealed summary must grow beyond one 4 KiB block";
  std::cout << "MIXED_LAYOUT_EVIDENCE values=" << values.size()
            << " destination_segments=" << records_per_segment.size()
            << " max_segment_cardinality=" << maximum_segment_cardinality
            << " inline=" << inline_records << " noninline=" << noninline_records
            << " capacity=" << result.capacity_bytes << std::endl;

  ASSERT_EQ(std::string(::getenv("DFKV_MIGRATION_IO_INTERPOSE_ACTIVE") == nullptr
                            ? ""
                            : ::getenv("DFKV_MIGRATION_IO_INTERPOSE_ACTIVE")),
            "1");
  const fs::path direct_log = ArmInterposer(root_, "observe");
  ASSERT_EQ(::setenv("DFKV_MIGRATION_IO_WATCH", destination_.c_str(), 1), 0);
  LogStore::Options direct_options;
  direct_options.dir = destination_.string();
  direct_options.capacity_bytes = result.capacity_bytes;
  direct_options.direct_writes = true;
  direct_options.cleaner = false;
  direct_options.preopen = false;
  direct_options.open_segments = 1;
  bool ok = false;
  auto direct = std::make_unique<LogStore>(direct_options, &ok);
  ASSERT_TRUE(ok) << direct->StartupError();
  // Constructor startup reads are not valid range-path proof. The scenario
  // change resets only the read-observation phase while retaining the exact
  // O_DIRECT descriptor inventory established above.
  ASSERT_EQ(::setenv("DFKV_MIGRATION_IO_SCENARIO", "observe_ranges", 1), 0);
  ExpectEveryValue(direct.get(), values);
  for (const auto& value : values) {
    SCOPED_TRACE(value.first.Filename());
    const std::string& expected = value.second;
    const size_t head_len = std::min<size_t>(37, expected.size());
    std::string actual;
    ASSERT_EQ(direct->Range(value.first, 0, head_len, &actual), Status::kOk);
    EXPECT_EQ(actual, expected.substr(0, head_len));
    const size_t middle = expected.size() / 2;
    const size_t middle_len = std::min<size_t>(53, expected.size() - middle);
    ASSERT_EQ(direct->Range(value.first, middle, middle_len, &actual), Status::kOk);
    EXPECT_EQ(actual, expected.substr(middle, middle_len));
    const size_t tail = expected.size() > 29 ? expected.size() - 29 : 0;
    const size_t tail_len = expected.size() - tail;
    ASSERT_EQ(direct->Range(value.first, tail, tail_len, &actual), Status::kOk);
    EXPECT_EQ(actual, expected.substr(tail));
  }
  ASSERT_TRUE(direct->Close());
  direct.reset();
  PrintInterposerEvidence("odirect_mixed_reopen", direct_log);
  const std::string direct_evidence = ReadText(direct_log);
  EXPECT_EQ(CountSubstring(direct_evidence, "event=odirect_open "),
            result.segment_count)
      << "every destination segment must really admit an O_DIRECT descriptor";
  EXPECT_GE(CountSubstring(direct_evidence, "event=odirect_range_pread "),
            records_per_segment.size())
      << "every populated segment must be read through an O_DIRECT fd in the range phase";
  EXPECT_EQ(CountSubstring(direct_evidence,
                           "event=buffered_destination_range_pread "), 0u)
      << "no destination range comparison may fall back to a buffered segment fd";
  ExpectRootUnchanged(source_, before);
  ASSERT_TRUE(source->VerifyUnchanged(&error)) << error;
  std::cout << "MIXED_LAYOUT_MIGRATION_RESULT "
            << dfkv::slab_migration::ResultJson(result) << std::endl;
}

}  // namespace
