// Test-only: preserve a connector-produced source in smaller native 1 MiB
// extents. It uses the real read-only snapshot and DiskSlabStore writers; no
// marker interpretation, key synthesis or raw fixture bytes are involved.
#include "cache/disk_slab_store.h"
#include "cache/slab_migration.h"

#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: dfkv_slab_fixture_repack ABSOLUTE_SOURCE ABSOLUTE_NEW_SMALL_SLAB\n";
    return 2;
  }
  std::error_code ec;
  if (std::filesystem::exists(argv[2], ec) || ec) {
    std::cerr << "new small slab path must be absent\n";
    return 2;
  }
  std::string error;
  auto source = dfkv::slab_migration::SlabSnapshot::Open(argv[1], &error);
  if (!source) {
    std::cerr << error << '\n';
    return 1;
  }
  dfkv::DiskSlabStore::Options options;
  options.dir = argv[2];
  options.capacity_bytes = 64ull << 20;
  options.extent_bytes = 1ull << 20;
  options.slot_granularity = 1ull << 20;
  options.direct_writes = false;
  options.reclaim_interval_ms = 0;
  options.table_sync_ms = 0;
  options.trust_token_dir = std::string(argv[2]) + ".tokens";
  std::filesystem::create_directories(options.trust_token_dir);
  bool ok = false;
  auto writer = std::make_unique<dfkv::DiskSlabStore>(options, &ok);
  if (!ok) {
    std::cerr << writer->StartupError() << '\n';
    return 1;
  }
  for (size_t i = 0; i < source->entries().size(); ++i) {
    std::string value, sha;
    if (!source->ReadValue(i, &value, &sha, &error) ||
        writer->Cache(source->entries()[i].key, value.data(), value.size()) != dfkv::Status::kOk) {
      std::cerr << "cannot repack source entry " << i << ": " << error << '\n';
      return 1;
    }
  }
  if (writer->Count() != source->entries().size() || !writer->Close()) {
    std::cerr << "small native slab did not close with every source value\n";
    return 1;
  }
  writer.reset();
  std::cout << "repacked_values=" << source->entries().size() << '\n';
  return 0;
}
