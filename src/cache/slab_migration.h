/* Fail-closed, explicit offline migration from a clean STL3 slab to LSE.
 *
 * The source reader is deliberately separate from DiskSlabStore: opening an
 * original with the writer can create/reset metadata, allocate extent blocks,
 * clear rejected records and evict values. This interface never does those
 * things. It requires an absolute, quiesced source and holds the store's
 * cooperative directory lock plus read-only descriptors throughout migration.
 */
#ifndef DFKV_SLAB_MIGRATION_H_
#define DFKV_SLAB_MIGRATION_H_

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/kv_types.h"

namespace dfkv {
namespace slab_migration {

struct FileManifest {
  std::string path;
  uint64_t device = 0;
  uint64_t inode = 0;
  uint64_t size = 0;
  uint64_t blocks_512 = 0;
  uint64_t mtime_ns = 0;
  uint64_t ctime_ns = 0;
  uint64_t links = 0;
  std::string sha256;
};

struct SourceEntry {
  BlockKey key;
  uint32_t extent = 0;
  uint32_t slot = 0;
  uint32_t slot_size = 0;
  uint32_t payload_len = 0;
  bool has_payload_crc = false;
  uint32_t payload_crc = 0;
};

class SlabSnapshot {
 public:
  // Opens only a fully initialized, clean, self-consistent STL3 store.
  // Tombstones, retention, malformed/unknown rows and duplicate live keys are
  // rejected rather than discarded. Empty values are rejected because the
  // ordinary LSE writer/recovery contract cannot preserve them.
  static std::unique_ptr<SlabSnapshot> Open(const std::string& dir,
                                            std::string* error);
  ~SlabSnapshot();
  SlabSnapshot(const SlabSnapshot&) = delete;
  SlabSnapshot& operator=(const SlabSnapshot&) = delete;

  const std::string& dir() const;
  uint64_t extent_bytes() const;
  uint64_t slot_granularity() const;
  uint32_t extent_count() const;
  uint64_t state_epoch() const;
  const std::vector<SourceEntry>& entries() const;

  // One bounded allocation, one pread, and (when present) source CRC checking.
  // sha256 is the exact source value's hexadecimal SHA-256.
  bool ReadValue(size_t entry_index, std::string* value, std::string* sha256,
                 std::string* error) const;
  // Re-enumerates both source directories and rehashes every held descriptor.
  // This detects endpoint changes, not a malicious mutate-and-restore writer;
  // callers must provide the documented offline/quiesced source boundary.
  bool VerifyUnchanged(std::string* error) const;
  const std::vector<FileManifest>& initial_manifest() const;

 private:
  struct Impl;
  explicit SlabSnapshot(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

struct MigratedRecord {
  BlockKey key;
  uint64_t value_len = 0;
  uint64_t source_slot_charge = 0;
  uint32_t source_payload_crc = 0;
  bool source_has_payload_crc = false;
  uint64_t generation = 0;
  uint64_t write_lsn = 0;
  uint64_t incarnation = 0;
  uint32_t segment = 0;
  uint32_t offset_blocks = 0;
  uint16_t header_len = 0;
  uint8_t flags = 0;
  uint32_t payload_crc = 0;
  uint64_t footprint = 0;
  std::string value_sha256;
};

struct MigrationResult {
  bool ok = false;
  bool published = false;
  bool source_unchanged = false;
  std::string error;
  std::string source_dir;
  std::string destination_dir;
  std::string staging_dir;
  uint64_t capacity_bytes = 0;
  uint64_t segment_bytes = 0;
  uint32_t segment_count = 0;
  uint64_t source_slot_charge = 0;
  uint64_t source_table_bytes = 0;
  uint64_t source_allocated_bytes = 0;
  uint64_t destination_allocated_bytes = 0;
  uint64_t elapsed_us = 0;
  std::vector<MigratedRecord> records;
  std::vector<FileManifest> source_before;
  std::vector<FileManifest> source_after;
  std::vector<FileManifest> destination_files;
};

// sorted_actual is ListNames' lexical directory inventory. Compare by
// membership rather than treating the expected generation order as lexical.
// Duplicate expected entries cannot come from the validated E%05u range.
inline bool ExtentInventoryMatches(const std::vector<std::string>& expected,
                                   const std::vector<std::string>& sorted_actual) {
  return expected.size() == sorted_actual.size() &&
         std::all_of(expected.begin(), expected.end(), [&](const std::string& name) {
           return std::binary_search(sorted_actual.begin(), sorted_actual.end(), name);
         });
}

// destination must be absolute and absent, as must destination +
// ".sto14-staging". No destination or staging path is ever reused or removed.
// The writer is the ordinary LogStore::Cache path. Exact source tuples and
// arbitrary values are migrated; values are never classified as markers.
MigrationResult MigrateToLse(const SlabSnapshot& source,
                             const std::string& destination);
std::string ResultJson(const MigrationResult& result);

}  // namespace slab_migration
}  // namespace dfkv

#endif  // DFKV_SLAB_MIGRATION_H_
