#include "cache/slab_migration.h"

#include <gtest/gtest.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "cache/disk_slab_store.h"
#include "cache/log_store.h"
#include "cache/lse_format.h"
#include "utils/crc32.h"
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
  const std::string value = "one";
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
  // A 3 MiB row is in-range and correctly checksummed, but is not in the
  // allocator's [1,2,4] MiB class lattice for a 4 MiB extent.
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

}  // namespace
