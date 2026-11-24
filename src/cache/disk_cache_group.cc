#include "cache/disk_cache_group.h"

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <charconv>
#include <cstring>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <cerrno>
#include <system_error>
#include <thread>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "cache/disk_slab_store.h"
#include "cache/log_store.h"
#include "cache/store_dir_class.h"
#include "common/config_dump.h"
#include "utils/log.h"
#include "utils/parallel_for.h"
#include "utils/thread_name.h"

namespace dfkv {

namespace {
bool ParseUnsigned(const char* text, uint64_t max, uint64_t* out) {
  if (text == nullptr || *text == '\0') return false;
  uint64_t value = 0;
  const char* end = text + std::strlen(text);
  const auto parsed = std::from_chars(text, end, value);
  if (parsed.ec != std::errc() || parsed.ptr != end || value > max)
    return false;
  *out = value;
  return true;
}

bool ResolveDiskHashWeight(int* out) {
  const char* text = std::getenv("DFKV_DISK_HASH_WEIGHT");
  if (text == nullptr || *text == '\0') {
    *out = 10;
    return true;
  }
  uint64_t value = 0;
  if (!ParseUnsigned(text, 64, &value) || value == 0) return false;
  *out = static_cast<int>(value);
  return true;
}

bool ParseTenantHash(const std::string& text, uint64_t* out) {
  if (text.size() != 16) return false;
  for (char c : text) {
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
      return false;
  }
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto parsed = std::from_chars(begin, end, *out, 16);
  return parsed.ec == std::errc() && parsed.ptr == end;
}

bool ParseTenantQuotasFile(const std::string& path,
                           std::map<uint64_t, uint64_t>* quotas,
                           std::string* error) {
  std::ifstream input(path);
  if (!input.is_open()) {
    *error = "cannot open DFKV_TENANT_QUOTAS_FILE: " + path;
    return false;
  }
  std::string line;
  size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    const size_t first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') continue;
    std::istringstream fields(line.substr(first));
    std::string hash_text;
    std::string bytes_text;
    std::string extra;
    if (!(fields >> hash_text >> bytes_text) || (fields >> extra)) {
      *error = "malformed tenant quota line " + std::to_string(line_number);
      return false;
    }
    uint64_t hash = 0;
    uint64_t bytes = 0;
    if (!ParseTenantHash(hash_text, &hash) ||
        !ParseUnsigned(bytes_text.c_str(), std::numeric_limits<uint64_t>::max(),
                       &bytes)) {
      *error = "malformed tenant quota line " + std::to_string(line_number);
      return false;
    }
    if (!quotas->emplace(hash, bytes).second) {
      *error = "duplicate tenant hash on quota line " +
               std::to_string(line_number);
      return false;
    }
  }
  if (input.bad()) {
    *error = "cannot read DFKV_TENANT_QUOTAS_FILE: " + path;
    return false;
  }
  return true;
}
}  // namespace

DiskCacheGroup::DiskCacheGroup(Options opt) {
  auto fail = [&](std::string error) {
    healthy_ = false;
    if (startup_error_.empty()) startup_error_ = std::move(error);
  };
  if (opt.cache_dirs.empty()) {
    fail("no cache directories configured");
    return;
  }
  if (opt.capacity_bytes == 0) {
    fail("capacity_bytes must be non-zero");
    return;
  }
  std::set<std::string> unique_dirs;
  for (const auto& dir : opt.cache_dirs) {
    if (dir.empty()) {
      fail("cache directory is empty");
      return;
    }
    if (!unique_dirs.insert(dir).second) {
      fail("duplicate cache directory: " + dir);
      return;
    }
  }
  if (const char* value = std::getenv("DFKV_TENANT_DEFAULT_QUOTA_BYTES")) {
    if (*value != '\0' &&
        !ParseUnsigned(value, std::numeric_limits<uint64_t>::max(),
                       &tenant_default_quota_bytes_)) {
      fail("DFKV_TENANT_DEFAULT_QUOTA_BYTES must be a uint64");
      return;
    }
  }
  if (const char* path = std::getenv("DFKV_TENANT_QUOTAS_FILE");
      path != nullptr && *path != '\0') {
    tenant_quotas_file_ = path;
    std::string error;
    if (!ParseTenantQuotasFile(tenant_quotas_file_, &tenant_quotas_, &error)) {
      config_dump::RecordResolved("DFKV_TENANT_QUOTAS_FILE",
                                  tenant_quotas_file_);
      config_dump::RecordResolved(
          "DFKV_TENANT_DEFAULT_QUOTA_BYTES",
          std::to_string(tenant_default_quota_bytes_));
      fail(std::move(error));
      return;
    }
  }
  for (const auto& quota : tenant_quotas_)
    tenant_quota_rejections_by_hash_.emplace(quota.first, 0);
  config_dump::RecordResolved(
      "DFKV_TENANT_QUOTAS_FILE",
      tenant_quotas_file_.empty() ? "(unset)" : tenant_quotas_file_);
  config_dump::RecordResolved("DFKV_TENANT_DEFAULT_QUOTA_BYTES",
                              std::to_string(tenant_default_quota_bytes_));
  config_dump::RecordResolved("DFKV_TENANT_QUOTAS_COUNT",
                              std::to_string(tenant_quotas_.size()));


  std::string engine = opt.engine;
  if (engine.empty()) {
    const char* value = std::getenv("DFKV_STORE_ENGINE");
    engine = (value && *value) ? value : "slab";
  }
  if (engine != "file" && engine != "slab" && engine != "auto" &&
      engine != "log") {
    fail("DFKV_STORE_ENGINE must be 'file' or 'slab'");
    return;
  }
  if (engine == "auto" || engine == "log") {
    const char* experimental = std::getenv("DFKV_LSE_EXPERIMENTAL");
    if (experimental == nullptr || std::strcmp(experimental, "1") != 0) {
      fail("DFKV_STORE_ENGINE=" + engine +
           " is experimental (the log-structured engine is under "
           "construction); it needs DFKV_LSE_EXPERIMENTAL=1");
      return;
    }
  }
  // The engine of each disk. "auto" and "log" decide it per directory from
  // what the directory holds (docs/design/log-structured-engine.md §10.1);
  // "slab" and "file" use one engine for every disk and leave the checks to
  // the engine, as they always did.
  std::vector<std::string> disk_engine(opt.cache_dirs.size(), engine);
  if ((engine == "auto" || engine == "log") && !opt.engine_factory) {
    for (size_t i = 0; i < opt.cache_dirs.size(); ++i) {
      const std::string& dir = opt.cache_dirs[i];
      StoreDirClassification found;
      std::string error;
      if (!ClassifyStoreDir(dir, &found, &error)) {
        fail(error);
        return;
      }
      std::string names;
      for (const auto& name : found.deciding)
        names += (names.empty() ? "" : ", ") + name;
      const std::string why =
          dir + " holds " + StoreDirClassName(found.cls) + " data (" +
          names + ")";
      switch (found.cls) {
        case StoreDirClass::kEmpty:
          // Until the log engine is accepted, an empty directory under
          // "auto" gets the slab engine, the default.
          disk_engine[i] = engine == "log" ? "log" : "slab";
          break;
        case StoreDirClass::kSlab:
        case StoreDirClass::kFile:
          if (engine == "log") {
            fail("DFKV_STORE_ENGINE=log, but " + why +
                 "; empty the directory or use another one");
            return;
          }
          disk_engine[i] =
              found.cls == StoreDirClass::kSlab ? "slab" : "file";
          break;
        case StoreDirClass::kLog:
          disk_engine[i] = "log";
          break;
        case StoreDirClass::kConflict:
        case StoreDirClass::kForeign:
          fail("cannot choose a storage engine: " + why +
               "; inspect the files, nothing was changed");
          return;
      }
      if (disk_engine[i] == "log" && found.lost_found_in_use) {
        fail("cannot open " + dir + " with the log-structured engine: its "
             "lost+found holds files (or cannot be listed); inspect them, "
             "nothing was changed");
        return;
      }
    }
  }
  engine_ = disk_engine.empty() ? engine : disk_engine.front();
  for (const auto& name : disk_engine)
    if (name != engine_) engine_ = "mixed";
  has_slab_ = std::find(disk_engine.begin(), disk_engine.end(), "slab") !=
              disk_engine.end();
  // Whether slab settings are read and checked: some disk may be slab.
  const bool use_slab = has_slab_;

  int disk_weight = 1;
  if (!ResolveDiskHashWeight(&disk_weight)) {
    fail("DFKV_DISK_HASH_WEIGHT must be an integer in [1,64]");
    return;
  }
  if (const char* text = std::getenv("DFKV_DEGRADED_MIN_DISKS");
      text != nullptr && *text != '\0') {
    uint64_t value = 0;
    if (!ParseUnsigned(text, std::numeric_limits<uint32_t>::max(), &value) ||
        value == 0) {
      fail("DFKV_DEGRADED_MIN_DISKS must be a positive integer");
      return;
    }
    min_healthy_disks_ = static_cast<size_t>(value);
  }
  // A value above this node's disk count means any failure takes it out of
  // placement; one setting can then serve nodes with different disk counts.
  min_healthy_disks_ = std::min(min_healthy_disks_, opt.cache_dirs.size());
  config_dump::RecordResolved("DFKV_DEGRADED_MIN_DISKS",
                              std::to_string(min_healthy_disks_));

  bool slab_direct = true;
  if (const char* mode = std::getenv("DFKV_SLAB_WRITE")) {
    if (std::strcmp(mode, "direct") == 0) {
      slab_direct = true;
    } else if (std::strcmp(mode, "buffered") == 0) {
      slab_direct = false;
    } else if (use_slab ||
               std::find(disk_engine.begin(), disk_engine.end(), "log") !=
                   disk_engine.end()) {
      // The log engine takes the same setting.
      fail("DFKV_SLAB_WRITE must be 'direct' or 'buffered'");
      return;
    }
  }
  // After an unclean exit: always reset the slab table (off, the default), or
  // keep it when the previous process provably died under the same boot and
  // mount (auto; boot also accepts runs without a unique mount id). See
  // DiskSlabStore::Options::WarmRestart.
  using SlabWarmRestart = DiskSlabStore::Options::WarmRestart;
  SlabWarmRestart slab_warm = SlabWarmRestart::kOff;
  if (const char* mode = std::getenv("DFKV_SLAB_WARM_RESTART");
      mode != nullptr && *mode != '\0') {
    if (std::strcmp(mode, "auto") == 0) {
      slab_warm = SlabWarmRestart::kAuto;
    } else if (std::strcmp(mode, "boot") == 0) {
      slab_warm = SlabWarmRestart::kBootOnly;
    } else if (std::strcmp(mode, "off") == 0) {
      slab_warm = SlabWarmRestart::kOff;
    } else if (use_slab) {
      fail("DFKV_SLAB_WARM_RESTART must be 'auto', 'boot' or 'off'");
      return;
    }
  }
  // STO-15: a PUT records its value's CRC32C and a whole-value read checks
  // it. 0 writes records without it and checks none (one CRC pass per PUT
  // and per GET saved); records written meanwhile stay unchecked for good.
  bool slab_payload_crc = true;
  if (const char* mode = std::getenv("DFKV_SLAB_PAYLOAD_CRC");
      mode != nullptr && *mode != '\0') {
    if (std::strcmp(mode, "1") == 0) {
      slab_payload_crc = true;
    } else if (std::strcmp(mode, "0") == 0) {
      slab_payload_crc = false;
    } else if (use_slab) {
      fail("DFKV_SLAB_PAYLOAD_CRC must be 0 or 1");
      return;
    }
  }
  if (use_slab)
    config_dump::RecordResolved("DFKV_SLAB_PAYLOAD_CRC",
                                slab_payload_crc ? "1" : "0");
  if (use_slab)
    config_dump::RecordResolved(
        "DFKV_SLAB_WARM_RESTART",
        slab_warm == SlabWarmRestart::kAuto
            ? "auto"
            : slab_warm == SlabWarmRestart::kBootOnly ? "boot" : "off");
  uint64_t slab_gran = 0;
  if (const char* value = std::getenv("DFKV_SLAB_GRANULARITY")) {
    if (!ParseUnsigned(value, std::numeric_limits<uint32_t>::max(),
                       &slab_gran) ||
        slab_gran == 0) {
      if (use_slab) {
        fail("DFKV_SLAB_GRANULARITY must be a positive uint32");
        return;
      }
      slab_gran = 0;
    }
  }
  uint64_t parsed = 0;
  // The periodic fdatasync of slots.tbl contributes to recovery only under
  // `boot`: there, for a run without a unique mount id, it bounds what a
  // filesystem that died and was mounted again without a reboot can have
  // lost from the table (and a sync in progress takes the trust token back),
  // so it runs every 100 ms. `off` resets the table after every unclean exit,
  // and `auto` keeps it only when the kernel, the mount and with them the
  // page cache survived, which holds every record write whether synced or
  // not. Under `auto` a slow sync (5 s) remains to find a background
  // writeback error of the table during the run, rather than only at the
  // next sync of a stop or a restart. Clean stops and extent rebinds sync
  // the table themselves. DFKV_SLAB_TABLE_SYNC_MS still sets it explicitly.
  uint32_t sync_ms = slab_warm == SlabWarmRestart::kBootOnly ? 100
                     : slab_warm == SlabWarmRestart::kAuto  ? 5000
                                                            : 0;
  if (const char* value = std::getenv("DFKV_SLAB_TABLE_SYNC_MS")) {
    if (!ParseUnsigned(value, std::numeric_limits<uint32_t>::max(), &parsed)) {
      if (use_slab) {
        fail("DFKV_SLAB_TABLE_SYNC_MS must be a uint32");
        return;
      }
    } else {
      sync_ms = static_cast<uint32_t>(parsed);
    }
  }
  // STO-35: move extents between size classes by their tail ages.
  bool slab_balance = false;
  if (const char* mode = std::getenv("DFKV_SLAB_BALANCE");
      mode != nullptr && *mode != '\0') {
    if (std::strcmp(mode, "1") == 0) {
      slab_balance = true;
    } else if (std::strcmp(mode, "0") != 0 && use_slab) {
      fail("DFKV_SLAB_BALANCE must be 0 or 1");
      return;
    }
  }
  if (use_slab)
    config_dump::RecordResolved("DFKV_SLAB_BALANCE", slab_balance ? "1" : "0");
  slab_balance_ = slab_balance && use_slab;
  uint32_t reclaim_ms = 50;
  if (const char* value = std::getenv("DFKV_SLAB_RECLAIM_MS")) {
    if (!ParseUnsigned(value, std::numeric_limits<uint32_t>::max(), &parsed)) {
      if (use_slab) {
        fail("DFKV_SLAB_RECLAIM_MS must be a uint32");
        return;
      }
    } else {
      reclaim_ms = static_cast<uint32_t>(parsed);
    }
  }

  config_dump::RecordResolved("DFKV_STORE_ENGINE", engine_);
  config_dump::RecordResolved("DFKV_DISK_HASH_WEIGHT",
                              std::to_string(disk_weight));
  config_dump::RecordResolved("DFKV_SLAB_WRITE",
                              slab_direct ? "direct" : "buffered");
  config_dump::RecordResolved(
      "DFKV_SLAB_GRANULARITY",
      slab_gran ? std::to_string(slab_gran) : std::string("1048576"));
  config_dump::RecordResolved("DFKV_SLAB_TABLE_SYNC_MS",
                              std::to_string(sync_ms));
  config_dump::RecordResolved("DFKV_SLAB_RECLAIM_MS",
                              std::to_string(reclaim_ms));

  const size_t n = opt.cache_dirs.size();
  const uint64_t base_capacity = opt.capacity_bytes / n;
  const uint64_t capacity_remainder = opt.capacity_bytes % n;
  write_mode_ = use_slab ? "direct" : "";
  // Every disk is opened on its own thread. A slab store reads its whole table
  // back before it can serve (seconds per disk with millions of objects), and
  // the stores share nothing but the synchronized process-wide registries
  // (resolved configuration, crash slots, the logger); one after the other,
  // startup grew with the number of disks.
  struct Opened {
    std::unique_ptr<StoreEngine> store;
    DiskSlabStore* slab = nullptr;
    std::string error;  // what the engine threw while opening, if anything
    bool threw = false;
    double seconds = 0;
  };
  std::vector<Opened> opened(n);
  auto open_one = [&](size_t i) {
    const std::string& dir = opt.cache_dirs[i];
    const uint64_t disk_capacity =
        base_capacity + (i < capacity_remainder ? 1 : 0);
    const auto started = std::chrono::steady_clock::now();
    try {
      if (opt.engine_factory) {
        opened[i].store = opt.engine_factory(dir, disk_capacity);
      } else if (disk_engine[i] == "log") {
        // Experimental (DFKV_LSE_EXPERIMENTAL=1, checked above). Writes are
        // direct or buffered as DFKV_SLAB_WRITE says, as for slab.
        LogStore::Options lo;
        lo.dir = dir;
        lo.capacity_bytes = disk_capacity;
        lo.direct_writes = slab_direct;
        lo.defer_activation = true;
        LogStore::ApplyEnvironment(&lo);
        opened[i].store = std::make_unique<LogStore>(lo);
      } else if (disk_engine[i] == "slab") {
        DiskSlabStore::Options so;
        so.dir = dir;
        so.capacity_bytes = disk_capacity;
        so.direct_writes = slab_direct;
        if (slab_gran) so.slot_granularity = slab_gran;
        so.table_sync_ms = sync_ms;
        so.reclaim_interval_ms = reclaim_ms;
        so.balance = slab_balance;
        so.warm_restart = slab_warm;
        so.payload_crc = slab_payload_crc;
        // Activated by Activate(), with the other disks, not by its own open.
        so.defer_activation = true;
        auto slab = std::make_unique<DiskSlabStore>(so);
        opened[i].slab = slab.get();
        opened[i].store = std::move(slab);
      } else {
        opened[i].store =
            std::make_unique<KVStore>(KVStore::Options{dir, disk_capacity});
      }
    } catch (const std::exception& error) {
      opened[i].threw = true;
      opened[i].error = error.what();
    } catch (...) {
      opened[i].threw = true;
      opened[i].error = "unknown exception";
    }
    opened[i].seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
  };
  const auto open_started = std::chrono::steady_clock::now();
  {
    std::vector<std::thread> helpers;
    helpers.reserve(n - 1);
    for (size_t i = 1; i < n; ++i) {
      // A thread that cannot be created must not end the process with the
      // other disks half open: that disk is opened here instead.
      try {
        helpers.emplace_back([&open_one, i] {
          NameThisThread("disk-open-", i);
          open_one(i);
        });
      } catch (const std::system_error& error) {
        DFKV_LOG_WARN("no thread to open " + opt.cache_dirs[i] + " (" +
                      error.what() + "); opening it in turn");
        open_one(i);
      }
    }
    open_one(0);
    for (auto& helper : helpers) helper.join();
  }
  // Every disk that threw is logged, not only the first one reported below.
  for (size_t i = 0; i < n; ++i)
    if (opened[i].threw)
      DFKV_LOG_ERROR("disk store startup failed: " + opt.cache_dirs[i] + ": " +
                     opened[i].error);
  size_t slowest = 0;
  for (size_t i = 0; i < n; ++i) {
    const std::string& dir = opt.cache_dirs[i];
    if (opened[i].seconds > opened[slowest].seconds) slowest = i;
    if (opened[i].threw) {
      // Reported like any other startup failure, not rethrown: the disks that
      // did open are marked dirty by now, and only an orderly teardown (the
      // rest of `opened` here, disks_ in the destructor) leaves them clean. An
      // exception escaping to a caller that does not catch it would abort the
      // process and cost every one of them its cache.
      fail(dir + ": " + opened[i].error);
      return;
    }
    std::unique_ptr<StoreEngine> store = std::move(opened[i].store);
    slab_of_disk_.push_back(opened[i].slab);
    if (opened[i].slab) {
      slabs_.push_back(opened[i].slab);
      if (!opened[i].slab->DirectWritesActive()) write_mode_ = "buffered";
    }
    if (!store) {
      fail(dir + ": engine factory returned null");
      return;
    }
    if (!store->Healthy()) {
      fail(dir + ": " + store->StartupError());
      DFKV_LOG_ERROR("disk store startup failed: " + dir + ": " +
                     store->StartupError());
    }
    by_id_[dir] = i;
    ids_.push_back(dir);
    canonical_ids_.push_back(CanonicalDiskId(dir));
    disks_.push_back(std::move(store));
    ring_.AddNode(dir, disk_weight);
  }
  if (n > 1 && healthy_) {
    char text[96];
    std::snprintf(text, sizeof(text), " disks in %.2f s (slowest %.2f s: ",
                  std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - open_started).count(),
                  opened[slowest].seconds);
    DFKV_LOG_INFO("disk stores opened: " + std::to_string(n) + text +
                  opt.cache_dirs[slowest] + ")");
  }
  ring_.Build();
  on_record_ = std::make_unique<std::atomic<bool>[]>(disks_.size());
  reset_.assign(disks_.size(), false);
  no_takeover_.assign(disks_.size(), false);
  misplaced_remove_for_test_ = opt.misplaced_remove_for_test;
  sync_removals_for_test_ = opt.sync_removals_for_test;
  record_location_for_test_ = opt.record_location_for_test;
  if (disks_.size() > 1) {
    auto points = ring_.NodePointCounts();
    size_t min_points = SIZE_MAX;
    size_t max_points = 0;
    for (const auto& [_, count] : points) {
      if (count < min_points) min_points = count;
      if (count > max_points) max_points = count;
    }
    DFKV_LOG_INFO("disk ring: " + std::to_string(disks_.size()) +
                  " disks, weight=" + std::to_string(disk_weight) +
                  ", points min=" + std::to_string(min_points) +
                  " max=" + std::to_string(max_points));
  }
  if (!opt.defer_activation) Activate();
}

DiskCacheGroup::~DiskCacheGroup() { Close(); }

bool DiskCacheGroup::Activate() {
  if (activated_) return Healthy();
  // A start in which a disk failed to open does not go on: the disks that
  // did open stay as they were found.
  if (!Healthy() || closed_) return false;
  if (disks_.size() > 1 && !ResetAfterFailedRun()) return false;
  const auto started = std::chrono::steady_clock::now();
  for (const auto& disk : disks_) {
    if (disk->Activate()) continue;
    healthy_ = false;
    const std::string error = disk->Dir() + ": " + disk->StartupError();
    if (startup_error_.empty()) startup_error_ = error;
    DFKV_LOG_ERROR("disk store could not be activated: " + error);
    return false;
  }
  activated_ = true;
  char text[64];
  std::snprintf(text, sizeof(text), " in %.1f ms",
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count());
  DFKV_LOG_INFO("disk stores activated: " + std::to_string(disks_.size()) +
                text + "; an exit without a clean stop is an unclean exit "
                "from here on");
  if (records_found_) DropMisplacedKeys();
  return true;
}

std::vector<DiskCacheGroup::CloseReport> DiskCacheGroup::Close() {
  std::vector<CloseReport> report;
  if (closed_) return report;
  closed_ = true;
  report.resize(disks_.size());
  auto close_one = [this, &report](size_t i) {
    const auto started = std::chrono::steady_clock::now();
    report[i].dir = disks_[i]->Dir();
    report[i].clean = disks_[i]->Close();
    report[i].seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
  };
  std::vector<std::thread> helpers;
  helpers.reserve(disks_.empty() ? 0 : disks_.size() - 1);
  for (size_t i = 1; i < disks_.size(); ++i) {
    // This also runs from the destructor: a thread that cannot be created
    // must not end the process, the disk is closed here instead.
    try {
      helpers.emplace_back(close_one, i);
    } catch (const std::system_error&) {
      close_one(i);
    }
  }
  if (!disks_.empty()) close_one(0);
  for (auto& helper : helpers) helper.join();
  // A run that found failure records and lost no disk, closed clean: what
  // the records asked for is done and on the disks.
  bool all_clean = true;
  for (const auto& r : report) all_clean = all_clean && r.clean;
  if (records_found_ && activated_ && all_clean && !open_drop_incomplete_ &&
      FailedDisks() == 0)
    ClearFailureRecords();
  return report;
}

bool DiskCacheGroup::Healthy() const {
  if (!healthy_ || record_failed_.load(std::memory_order_acquire)) return false;
  const size_t failed = FailedDisks();
  if (failed == 0) return disks_.size() >= min_healthy_disks_;
  // A failed disk's keys go to the healthy disks that take other disks' keys:
  // one excluded from takeover (DropMisplacedKeys) serves only its own, so it
  // does not count. Otherwise those keys would fail while the node looks
  // healthy.
  size_t takers = 0;
  for (size_t i = 0; i < disks_.size(); ++i)
    takers += disks_[i]->Healthy() && !no_takeover_[i] ? 1 : 0;
  return takers >= min_healthy_disks_;
}

size_t DiskCacheGroup::FailedDisks() const {
  size_t failed = 0;
  for (const auto& disk : disks_) failed += disk->Healthy() ? 0 : 1;
  return failed;
}

uint64_t DiskCacheGroup::MaxValueBytes() const {
  uint64_t max = disks_.empty() ? 0 : UINT64_MAX;
  for (const auto& disk : disks_) max = std::min(max, disk->MaxValueBytes());
  return max;
}

bool DiskCacheGroup::DiskFailedFor(const BlockKey& key) const {
  // A key is routed away from a failed disk while any other is healthy, so
  // this holds only once every disk has failed.
  const size_t index = RouteIndex(key);
  return index < disks_.size() && !disks_[index]->Healthy();
}

size_t DiskCacheGroup::HomeIndex(const std::string& name) const {
  std::string id;
  if (!ring_.Lookup(name, &id)) return kInvalidDisk;
  auto it = by_id_.find(id);
  return it == by_id_.end() ? kInvalidDisk : it->second;
}

size_t DiskCacheGroup::RouteIndex(const BlockKey& key) const {
  if (disks_.size() == 1) return 0;
  const std::string name = key.Filename();
  const size_t home = HomeIndex(name);
  if (home == kInvalidDisk || disks_[home]->Healthy()) return home;
  // The key's disk has failed for good. Below DFKV_DEGRADED_MIN_DISKS, or
  // with the failure not on record, the key stays there (and fails).
  if (!Healthy() || !FailureOnRecord(home)) return home;
  // Its keys go to the healthy disks by highest random weight: each failed
  // disk's keys spread evenly, a healthy disk's keys never move, and a key
  // moves again only if the disk it moved to fails too.
  const uint64_t hash = std::hash<std::string>{}(name);
  size_t best = home;
  uint64_t best_score = 0;
  for (size_t i = 0; i < disks_.size(); ++i) {
    if (!disks_[i]->Healthy() || no_takeover_[i]) continue;
    uint64_t score = hash ^ ((i + 1) * 0x9E3779B97F4A7C15ull);
    score = (score ^ (score >> 30)) * 0xBF58476D1CE4E5B9ull;
    score = (score ^ (score >> 27)) * 0x94D049BB133111EBull;
    score ^= score >> 31;
    if (best == home || score > best_score) {
      best = i;
      best_score = score;
    }
  }
  return best;
}

namespace {
// Next to the cache directory, on the same disk, outside what any engine
// reads: "/mnt/disk2/dfkv" -> "/mnt/disk2/dfkv.dfkv-failed-disks". `dir` is
// a canonical id (CanonicalDiskId).
std::string FailureRecordPath(const std::string& dir) {
  return dir + ".dfkv-failed-disks";
}

std::string ParentOf(const std::string& path) {
  std::string parent = std::filesystem::path(path).parent_path().string();
  return parent.empty() ? std::string(".") : parent;
}

bool SyncParentOf(const std::string& path) {
  const int fd =
      ::open(ParentOf(path).c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fd < 0) return false;
  const bool ok = ::fsync(fd) == 0;
  ::close(fd);
  return ok;
}

// A record lives exactly as long as the data it protects only where the
// directory above the cache directory is on the cache directory's own
// filesystem: not for a mount root or a container volume.
bool OnTheDirsFilesystem(const std::string& dir) {
  struct stat d {}, p {};
  return ::stat(dir.c_str(), &d) == 0 && ::stat(ParentOf(dir).c_str(), &p) == 0 &&
         d.st_dev == p.st_dev;
}
}  // namespace

std::string DiskCacheGroup::CanonicalDiskId(const std::string& dir) {
  std::error_code ec;
  const auto canonical = std::filesystem::weakly_canonical(dir, ec);
  std::string id =
      ec ? std::filesystem::path(dir).lexically_normal().string()
         : canonical.string();
  while (id.size() > 1 && id.back() == '/') id.pop_back();
  return id;
}

bool DiskCacheGroup::FailureOnRecord(size_t i) const {
  if (on_record_[i].load(std::memory_order_acquire)) return true;
  if (record_failed_.load(std::memory_order_acquire)) return false;
  std::lock_guard<std::mutex> lock(record_mu_);
  if (on_record_[i].load(std::memory_order_acquire)) return true;
  if (record_failed_.load(std::memory_order_acquire)) return false;
  std::set<std::string> ids = failed_ids_;
  ids.insert(canonical_ids_[i]);
  if (!WriteFailureRecords(ids)) {
    record_failed_.store(true, std::memory_order_release);
    DFKV_LOG_ERROR("disk store failed: " + ids_[i] +
                   "; its failure could not be put on record next to any "
                   "healthy disk (each cache directory must be a "
                   "subdirectory on its disk's own filesystem), so its keys "
                   "do not move to the other disks and the node reports "
                   "itself unhealthy");
    return false;
  }
  failed_ids_ = std::move(ids);
  on_record_[i].store(true, std::memory_order_release);
  DFKV_LOG_ERROR("disk store failed: " + ids_[i] +
                 "; on record, its keys now go to the other disks. The next "
                 "start empties it.");
  return true;
}

bool DiskCacheGroup::WriteFailureRecords(
    const std::set<std::string>& ids) const {
  std::string text;
  for (const auto& id : ids) text += id + "\n";
  size_t written = 0;
  for (size_t j = 0; j < disks_.size(); ++j) {
    if (!disks_[j]->Healthy()) continue;
    const std::string& dir = canonical_ids_[j];
    const bool same_fs = record_location_for_test_
                             ? record_location_for_test_(dir)
                             : OnTheDirsFilesystem(dir);
    if (!same_fs) {
      DFKV_LOG_WARN("no disk failure record next to " + dir + ": the "
                    "directory above it is on another filesystem");
      continue;
    }
    const std::string path = FailureRecordPath(dir);
    const std::string tmp = path + ".tmp";
    const int fd =
        ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) continue;
    bool ok = ::write(fd, text.data(), text.size()) ==
              static_cast<ssize_t>(text.size());
    ok = ::fsync(fd) == 0 && ok;
    ok = ::close(fd) == 0 && ok;
    ok = ok && ::rename(tmp.c_str(), path.c_str()) == 0 && SyncParentOf(path);
    if (ok) ++written;
    else DFKV_LOG_WARN("cannot write the disk failure record " + path);
  }
  return written > 0;
}

// Every copy there is, read in full. A copy that is there but cannot be
// read completely, or that names a directory this node does not have, is
// not usable: what it names cannot be known.
DiskCacheGroup::FailureRecords DiskCacheGroup::ReadFailureRecords() const {
  FailureRecords out;
  for (const auto& dir : canonical_ids_) {
    const std::string path = FailureRecordPath(dir);
    struct stat st {};
    if (::lstat(path.c_str(), &st) != 0) {
      if (errno == ENOENT) continue;
      out.found = true;
      out.usable = false;
      DFKV_LOG_ERROR("cannot inspect the disk failure record " + path + ": " +
                     std::strerror(errno));
      continue;
    }
    out.found = true;
    std::string text;
    bool read_ok = S_ISREG(st.st_mode);
    const int fd = read_ok ? ::open(path.c_str(), O_RDONLY | O_CLOEXEC) : -1;
    if (fd < 0) read_ok = false;
    while (read_ok) {
      char buf[4096];
      const ssize_t n = ::read(fd, buf, sizeof(buf));
      if (n < 0 && errno == EINTR) continue;
      if (n < 0) read_ok = false;
      if (n <= 0) break;
      text.append(buf, static_cast<size_t>(n));
    }
    if (fd >= 0) ::close(fd);
    // The writer never writes an empty record: one is a torn write.
    if (!read_ok || text.empty() || text.back() != '\n') {
      out.usable = false;
      DFKV_LOG_ERROR("the disk failure record " + path + " cannot be read "
                     "completely");
      continue;
    }
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
      if (line.empty()) continue;
      const std::string id = CanonicalDiskId(line);
      if (std::find(canonical_ids_.begin(), canonical_ids_.end(), id) !=
          canonical_ids_.end()) {
        out.ids.insert(id);
      } else {
        out.usable = false;
        DFKV_LOG_ERROR("the disk failure record " + path + " names " + line +
                       ", which is not a cache directory of this node");
      }
    }
  }
  return out;
}

// Before the disks are activated: the disks that failed in an earlier run
// (all of them, if what failed cannot be known) are reset to empty, each by
// its engine. False if one cannot be: the start does not go on.
bool DiskCacheGroup::ResetAfterFailedRun() {
  const FailureRecords records = ReadFailureRecords();
  records_found_ = records.found;
  if (!records.found) return true;
  if (!records.usable)
    DFKV_LOG_ERROR("a disk failure record is not usable: every disk is "
                   "emptied, as what failed in the earlier run cannot be "
                   "known");
  for (size_t i = 0; i < disks_.size(); ++i) {
    if (records.usable && records.ids.count(canonical_ids_[i]) == 0) continue;
    const uint64_t keys = disks_[i]->Count();
    if (!disks_[i]->ResetBeforeActivation()) {
      healthy_ = false;
      const std::string error = ids_[i] + ": cannot reset the disk to empty "
                                "after a failure in an earlier run: " +
                                disks_[i]->StartupError();
      if (startup_error_.empty()) startup_error_ = error;
      DFKV_LOG_ERROR(error);
      return false;
    }
    reset_[i] = true;
    (records.usable ? open_emptied_keys_ : open_unusable_keys_) += keys;
  }
  return true;
}

// After the disks are activated, if a record was found: a disk not reset
// drops the keys whose home is another disk (what an earlier run wrote in
// place of a failed disk).
void DiskCacheGroup::DropMisplacedKeys() {
  const auto started = std::chrono::steady_clock::now();
  const size_t n = disks_.size();
  std::vector<uint64_t> misplaced(n, 0), refused(n, 0);
  std::vector<bool> unsynced(n, false);
  auto drop_one = [&](size_t i) {
    if (reset_[i]) return;
    std::vector<BlockKey> keys;
    disks_[i]->ForEachKey([&](const BlockKey& key) {
      if (HomeIndex(key.Filename()) != i) keys.push_back(key);
    });
    for (const BlockKey& key : keys) {
      const Status st = misplaced_remove_for_test_
                            ? misplaced_remove_for_test_(i, key)
                            : disks_[i]->Remove(key);
      if (st != Status::kOk && st != Status::kNotFound) ++refused[i];
      else ++misplaced[i];
    }
    // The record is cleared at a clean close of this run: by then a power
    // loss must not bring these keys back (the file engine's REMOVE is not
    // durable by itself).
    if (misplaced[i] != 0)
      unsynced[i] = !(sync_removals_for_test_ ? sync_removals_for_test_(i)
                                              : disks_[i]->SyncRemovals());
  };
  std::vector<std::thread> helpers;
  for (size_t i = 1; i < n; ++i) {
    try {
      helpers.emplace_back(drop_one, i);
    } catch (const std::system_error&) {
      drop_one(i);
    }
  }
  drop_one(0);
  for (auto& helper : helpers) helper.join();
  uint64_t refused_total = 0;
  for (size_t i = 0; i < n; ++i) {
    open_misplaced_keys_ += misplaced[i];
    refused_total += refused[i];
    // It still holds a value it must not serve for another disk's key:
    // that key's home failing again would route it here.
    if (refused[i]) {
      no_takeover_[i] = true;
      DFKV_LOG_WARN(ids_[i] + ": " + std::to_string(refused[i]) + " keys of "
                    "other disks could not be removed; it takes no other "
                    "disk's keys in this run");
    }
    // Removed for this run, but a power loss could bring them back: the
    // record stays for the next start.
    if (unsynced[i]) {
      open_drop_incomplete_ = true;
      DFKV_LOG_WARN(ids_[i] + ": the removal of keys of other disks could "
                    "not be synced; the records are kept for the next start");
    }
  }
  if (refused_total != 0) open_drop_incomplete_ = true;
  char text[64];
  std::snprintf(text, sizeof(text), " in %.1f ms",
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count());
  DFKV_LOG_INFO("disk failure records found: emptied " +
                std::to_string(open_emptied_keys_) + " keys of the disks "
                "that failed, " + std::to_string(open_unusable_keys_) +
                " keys of every disk for an unusable record, dropped " +
                std::to_string(open_misplaced_keys_) + " keys found on a "
                "disk that is not their home" + text);
  if (refused_total)
    DFKV_LOG_WARN(std::to_string(refused_total) +
                  " of those keys could not be removed; the records are kept "
                  "for the next start");
}

void DiskCacheGroup::ClearFailureRecords() {
  for (const auto& dir : canonical_ids_) {
    const std::string path = FailureRecordPath(dir);
    struct stat st {};
    if (::lstat(path.c_str(), &st) != 0) continue;
    // Only a file this node writes; anything else there is not ours.
    if (!S_ISREG(st.st_mode)) {
      DFKV_LOG_WARN(path + " is not a file: left as it is (it makes every "
                    "start empty all disks until it is removed)");
      continue;
    }
    if (::unlink(path.c_str()) == 0) SyncParentOf(path);
    else
      DFKV_LOG_WARN("cannot remove the disk failure record " + path + ": " +
                    std::strerror(errno));
  }
  DFKV_LOG_INFO("disk failure records cleared after a clean stop");
}

StoreEngine* DiskCacheGroup::Route(const BlockKey& key) const {
  const size_t index = RouteIndex(key);
  return index == kInvalidDisk ? nullptr : disks_[index].get();
}
uint64_t DiskCacheGroup::TenantLimit(uint64_t tenant_hash) const {
  std::shared_lock<std::shared_mutex> lock(tenant_quotas_mu_);
  const auto it = tenant_quotas_.find(tenant_hash);
  return it == tenant_quotas_.end() ? tenant_default_quota_bytes_ : it->second;
}

bool DiskCacheGroup::ReloadTenantQuotas(std::string* error) {
  if (tenant_quotas_file_.empty()) {
    *error = "DFKV_TENANT_QUOTAS_FILE is not set";
    tenant_quota_reload_failures_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  std::map<uint64_t, uint64_t> quotas;
  if (!ParseTenantQuotasFile(tenant_quotas_file_, &quotas, error)) {
    tenant_quota_reload_failures_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  // The quotas lock only, no tenant stripe. A write between its quota check
  // and its accounting while the quotas change was checked against the old
  // quota: it lands as if it had come just before the reload, and the next
  // write of its tenant is checked against the new one. Writes of one tenant
  // still check and account one at a time under its stripe, so the most a
  // tenant that already had a quota can be over a lowered one is what it
  // held before the reload plus the one write (or batch part) in flight. A
  // tenant that gets its first quota in this reload had no stripe taken by
  // its writes in flight: every one of those lands unchecked. The rejection
  // counts are read and incremented under the shared lock, which this
  // exclusive lock waits out.
  std::unique_lock<std::shared_mutex> lock(tenant_quotas_mu_);
  std::map<uint64_t, uint64_t> rejections;
  for (const auto& quota : quotas) {
    const auto kept = tenant_quota_rejections_by_hash_.find(quota.first);
    rejections.emplace(quota.first,
                       kept == tenant_quota_rejections_by_hash_.end()
                           ? 0
                           : kept->second);
  }
  tenant_quotas_.swap(quotas);
  tenant_quota_rejections_by_hash_.swap(rejections);
  tenant_quota_reloads_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool DiskCacheGroup::QuotaAllowsNewLocked(uint64_t tenant_hash,
                                          uint64_t current,
                                          uint64_t additional) const {
  const uint64_t limit = TenantLimit(tenant_hash);
  return limit == 0 ||
         (current <= limit && additional <= limit - current);
}

void DiskCacheGroup::RecordQuotaRejectionLocked(uint64_t tenant_hash) {
  tenant_quota_rejections_.fetch_add(1, std::memory_order_relaxed);
  std::shared_lock<std::shared_mutex> lock(tenant_quotas_mu_);
  if (tenant_quotas_.find(tenant_hash) == tenant_quotas_.end()) return;
  auto it = tenant_quota_rejections_by_hash_.find(tenant_hash);
  if (it != tenant_quota_rejections_by_hash_.end()) ++it->second;
}


Status DiskCacheGroup::Cache(const BlockKey& key, const void* data, size_t len) {
  if (len == 0 || data == nullptr) return Status::kInvalid;
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  // A tenant without a limit has no admission state to protect, so its
  // writes must not serialize on a stripe held across storage I/O.
  if (!TenantLimited(key.tenant_hash)) return d->Cache(key, data, len);
  std::lock_guard<std::mutex> tenant_lock(
      tenant_stripes_[TenantStripe(key.tenant_hash)]);
  ValueMetadata metadata;
  const Status lookup = d->Lookup(key, &metadata);
  if (lookup != Status::kOk && lookup != Status::kNotFound) return lookup;
  if (lookup == Status::kNotFound &&
      !QuotaAllowsNewLocked(key.tenant_hash,
                            TenantUsedBytes(key.tenant_hash), len)) {
    RecordQuotaRejectionLocked(key.tenant_hash);
    return Status::kQuotaExceeded;
  }
  return d->Cache(key, data, len);
}

Status DiskCacheGroup::Remove(const BlockKey& key) {
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  if (!TenantLimited(key.tenant_hash)) return d->Remove(key);
  std::lock_guard<std::mutex> tenant_lock(
      tenant_stripes_[TenantStripe(key.tenant_hash)]);
  return d->Remove(key);
}
bool DiskCacheGroup::RetainEnabled() const {
  if (disks_.empty()) return false;
  for (const auto& d : disks_)
    if (!d->RetainEnabled()) return false;
  return true;
}

uint8_t DiskCacheGroup::Retain(const BlockKey& key, uint32_t ttl_s,
                               uint8_t mode, uint32_t now_s) {
  StoreEngine* d = Route(key);
  if (d == nullptr) return StoreEngine::kRetainUnsupported;
  return d->Retain(key, ttl_s, mode, now_s);
}

// A failed disk takes no key while another is healthy: its limits do not
// apply to the node.
uint32_t DiskCacheGroup::RetainHeadroomPct() const {
  const bool skip_failed = FailedDisks() < disks_.size();
  uint32_t pct = 100;
  for (const auto& d : disks_)
    if (!skip_failed || d->Healthy())
      pct = std::min(pct, d->RetainHeadroomPct());
  return disks_.empty() ? 0 : pct;
}

uint32_t DiskCacheGroup::RetainMaxTtl() const {
  const bool skip_failed = FailedDisks() < disks_.size();
  uint32_t ttl = UINT32_MAX;
  for (const auto& d : disks_)
    if (!skip_failed || d->Healthy()) ttl = std::min(ttl, d->RetainMaxTtl());
  return disks_.empty() ? 0 : ttl;
}

SlabAllocator::RetentionStats DiskCacheGroup::RetentionStats() const {
  SlabAllocator::RetentionStats sum;
  bool first = true;
  // With disks that keep retention on the node, theirs: an engine without
  // it (file) would report it disabled with no TTL.
  const bool some_retain = std::any_of(
      disks_.begin(), disks_.end(),
      [](const std::unique_ptr<StoreEngine>& d) { return d->RetainEnabled(); });
  for (size_t i = 0; i < disks_.size(); ++i) {
    if (some_retain && !disks_[i]->RetainEnabled()) continue;
    const auto st = disks_[i]->GetRetentionStats();
    sum.enabled = first ? st.enabled : (sum.enabled && st.enabled);
    sum.budget_bytes += st.budget_bytes;
    sum.protected_bytes += st.protected_bytes;
    sum.protected_slots += st.protected_slots;
    sum.extents_pinned += st.extents_pinned;
    sum.expired += st.expired;
    sum.evict_skipped += st.evict_skipped;
    sum.wheel_entries += st.wheel_entries;
    sum.max_ttl_s = first ? st.max_ttl_s : std::min(sum.max_ttl_s, st.max_ttl_s);
    sum.headroom_pct =
        first ? st.headroom_pct : std::min(sum.headroom_pct, st.headroom_pct);
    for (size_t i = 0; i < 7; ++i) sum.results[i] += st.results[i];
    first = false;
  }
  return sum;
}

Status DiskCacheGroup::Lookup(const BlockKey& key, ValueMetadata* out) const {
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->Lookup(key, out);
}


Status DiskCacheGroup::CacheDirect(const BlockKey& key, char* data, size_t len,
                                   size_t cap) {
  if (len == 0 || data == nullptr) return Status::kInvalid;
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  if (!TenantLimited(key.tenant_hash))
    return d->CacheDirect(key, data, len, cap);
  std::lock_guard<std::mutex> tenant_lock(
      tenant_stripes_[TenantStripe(key.tenant_hash)]);
  ValueMetadata metadata;
  const Status lookup = d->Lookup(key, &metadata);
  if (lookup != Status::kOk && lookup != Status::kNotFound) return lookup;
  if (lookup == Status::kNotFound &&
      !QuotaAllowsNewLocked(key.tenant_hash,
                            TenantUsedBytes(key.tenant_hash), len)) {
    RecordQuotaRejectionLocked(key.tenant_hash);
    return Status::kQuotaExceeded;
  }
  return d->CacheDirect(key, data, len, cap);
}

Status DiskCacheGroup::Range(const BlockKey& key, uint64_t offset,
                             uint64_t length, std::string* out,
                             size_t* value_len) {
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->Range(key, offset, length, out, value_len);
}

Status DiskCacheGroup::RangeInto(const BlockKey& key, uint64_t offset,
                                 uint64_t length, char* dst, size_t dst_cap,
                                 size_t* out_len, size_t* value_len) {
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->RangeInto(key, offset, length, dst, dst_cap, out_len, value_len);
}

Status DiskCacheGroup::RangeDirect(
    const BlockKey& key, uint64_t offset, uint64_t length, char* io_buf,
    size_t io_cap, const char** out_data, size_t* out_len, size_t* value_len) {
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->RangeDirect(key, offset, length, io_buf, io_cap, out_data, out_len,
                        value_len);
}

Status DiskCacheGroup::RangeDirectPrep(const BlockKey& key, uint64_t offset,
                                       uint64_t length, size_t io_cap,
                                       ReadLease* out) {
  if (out == nullptr) return Status::kInvalid;
  *out = ReadLease{};
  StoreEngine* d = Route(key);
  if (d == nullptr) return Status::kInvalid;
  return d->RangeDirectPrep(key, offset, length, io_cap, out);
}

std::vector<Status> DiskCacheGroup::CacheDirectBatch(
    const std::vector<StoreEngine::CacheBatchItem>& items,
    const DirectBatchObserver& observe) {
  std::vector<Status> out(items.size(), Status::kInvalid);
  if (items.empty()) return out;

  // Only tenants with a limit take their admission stripe (in ascending order).
  // The common unlimited batch -- every RAM-tier flush of a deployment without
  // quotas -- takes no lock here, so flush workers overlap instead of running
  // one batch at a time behind a stripe held across the disk writes.
  std::array<bool, kTenantStripes> used_stripes{};
  // Once per item: a quota reload in between must not leave an item checked
  // without its stripe.
  std::vector<char> limited(items.size(), 0);
  size_t used = 0;
  for (size_t i = 0; i < items.size(); ++i) {
    limited[i] = TenantLimited(items[i].key.tenant_hash) ? 1 : 0;
    if (!limited[i]) continue;
    bool& stripe_used = used_stripes[TenantStripe(items[i].key.tenant_hash)];
    if (!stripe_used) ++used;
    stripe_used = true;
  }
  if (used > kMaxHeldStripes) {
    // A batch over more stripes than we hold at once (the RAM tier's batches
    // have at most 16 items, so not those) is run in parts of at most
    // kMaxHeldStripes stripes, in ascending stripe order. A tenant's items
    // all fall in one part, so admission within a tenant is unchanged;
    // unlimited items go with the first part. Each part reports to `observe`
    // on its own, so a split batch counts as one batch per part in per-batch
    // metrics.
    std::array<size_t, kTenantStripes> part_of{};
    size_t parts = 0, in_part = kMaxHeldStripes;
    for (size_t stripe = 0; stripe < kTenantStripes; ++stripe) {
      if (!used_stripes[stripe]) continue;
      if (in_part == kMaxHeldStripes) {
        ++parts;
        in_part = 0;
      }
      part_of[stripe] = parts - 1;
      ++in_part;
    }
    for (size_t part = 0; part < parts; ++part) {
      std::vector<size_t> indexes;
      std::vector<StoreEngine::CacheBatchItem> sub;
      for (size_t i = 0; i < items.size(); ++i) {
        const size_t of =
            limited[i] ? part_of[TenantStripe(items[i].key.tenant_hash)] : 0;
        if (of != part) continue;
        indexes.push_back(i);
        sub.push_back(items[i]);
      }
      const std::vector<Status> statuses = CacheDirectBatch(sub, observe);
      for (size_t k = 0; k < indexes.size() && k < statuses.size(); ++k)
        out[indexes[k]] = statuses[k];
    }
    return out;
  }
  std::array<std::unique_lock<std::mutex>, kTenantStripes> locks;
  for (size_t stripe = 0; stripe < kTenantStripes; ++stripe) {
    if (used_stripes[stripe])
      locks[stripe] = std::unique_lock<std::mutex>(tenant_stripes_[stripe]);
  }

  // Admission is evaluated in caller order. Existing keys and batch-internal
  // duplicates are idempotent and consume no additional budget.
  std::vector<uint8_t> charge(items.size(), 0);
  for (size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    const size_t disk = RouteIndex(item.key);
    if (disk == kInvalidDisk || item.len == 0 || item.data == nullptr)
      continue;
    if (!limited[i]) {
      out[i] = Status::kOk;  // no admission: the engine decides
      continue;
    }
    ValueMetadata metadata;
    const Status lookup = disks_[disk]->Lookup(item.key, &metadata);
    if (lookup != Status::kOk && lookup != Status::kNotFound) {
      out[i] = lookup;
      continue;
    }
    if (lookup == Status::kOk) {
      out[i] = Status::kOk;
      continue;
    }
    bool duplicate = false;
    uint64_t additional = item.len;
    for (size_t j = 0; j < i; ++j) {
      if (out[j] != Status::kOk ||
          items[j].key.tenant_hash != item.key.tenant_hash) {
        continue;
      }
      if (items[j].key == item.key) {
        duplicate = true;
        break;
      }
      if (charge[j]) {
        if (additional > std::numeric_limits<uint64_t>::max() - items[j].len) {
          additional = std::numeric_limits<uint64_t>::max();
          break;
        }
        additional += items[j].len;
      }
    }
    if (duplicate) {
      out[i] = Status::kOk;
      continue;
    }
    if (!QuotaAllowsNewLocked(item.key.tenant_hash,
                              TenantUsedBytes(item.key.tenant_hash),
                              additional)) {
      out[i] = Status::kQuotaExceeded;
      RecordQuotaRejectionLocked(item.key.tenant_hash);
      continue;
    }
    charge[i] = 1;
    out[i] = Status::kOk;
  }

  // Stable disk-index groups avoid pointer-order routing and put every result
  // directly back in the caller's original position.
  std::vector<std::vector<size_t>> groups(disks_.size());
  std::vector<size_t> participating;
  for (size_t i = 0; i < items.size(); ++i) {
    if (out[i] != Status::kOk) continue;
    const size_t disk = RouteIndex(items[i].key);
    if (groups[disk].empty()) participating.push_back(disk);
    groups[disk].push_back(i);
    out[i] = Status::kIOError;  // actual engine result replaces this default
  }

  auto run_disk = [&](size_t disk) noexcept {
    const auto started = std::chrono::steady_clock::now();
    const auto& indexes = groups[disk];
    uint64_t bytes = 0;
    for (size_t index : indexes) bytes += items[index].len;
    try {
      std::vector<StoreEngine::CacheBatchItem> sub;
      sub.reserve(indexes.size());
      for (size_t index : indexes) sub.push_back(items[index]);
      const std::vector<Status> statuses =
          disks_[disk]->CacheDirectBatch(sub);
      const size_t count = std::min(indexes.size(), statuses.size());
      for (size_t i = 0; i < count; ++i) out[indexes[i]] = statuses[i];
    } catch (...) {
      // Per-item defaults remain kIOError; other disks still complete.
    }
    if (observe) {
      size_t failures = 0;
      for (size_t index : indexes)
        if (out[index] != Status::kOk) ++failures;
      observe(disk, indexes.size(), bytes, failures,
              std::chrono::duration<double>(
                  std::chrono::steady_clock::now() - started)
                  .count());
    }
  };
  if (participating.empty()) return out;

  if (participating.size() == 1) {
    run_disk(participating[0]);  // common single-disk case: no thread overhead
    return out;
  }

  RunParallel(participating.size(), participating.size(),
              [&](size_t group_index) {
                run_disk(participating[group_index]);
              });
  return out;
}

DiskSlabStore::Stats DiskCacheGroup::SlabStats() const {
  DiskSlabStore::Stats sum;
  for (const auto* d : slabs_) {
    const auto st = d->GetStats();
    sum.dio_write_fallbacks += st.dio_write_fallbacks;
    sum.put_no_slot += st.put_no_slot;
    sum.put_busy += st.put_busy;
    sum.dio_read_fallbacks += st.dio_read_fallbacks;
    sum.table_syncs += st.table_syncs;
    sum.bind_wipes += st.bind_wipes;
    sum.payload_crc_failures += st.payload_crc_failures;
    sum.steals += st.steals;
    sum.cold_steals += st.cold_steals;
    sum.watermark_evictions += st.watermark_evictions;
    sum.watermark_extent_clears += st.watermark_extent_clears;
    sum.watermark_slot_removals += st.watermark_slot_removals;
    sum.watermark_slot_record_clears += st.watermark_slot_record_clears;
    sum.watermark_record_clear_bytes += st.watermark_record_clear_bytes;
    sum.watermark_freed_slot_bytes += st.watermark_freed_slot_bytes;
    sum.watermark_scan_visits += st.watermark_scan_visits;
    sum.watermark_max_victims_per_tick =
        std::max(sum.watermark_max_victims_per_tick,
                 st.watermark_max_victims_per_tick);
    sum.watermark_max_visits_per_tick =
        std::max(sum.watermark_max_visits_per_tick,
                 st.watermark_max_visits_per_tick);
    sum.watermark_ticks += st.watermark_ticks;
    sum.watermark_active =
        std::max(sum.watermark_active, st.watermark_active);
    sum.watermark_last_tick_us =
        std::max(sum.watermark_last_tick_us, st.watermark_last_tick_us);
    sum.watermark_max_tick_us =
        std::max(sum.watermark_max_tick_us, st.watermark_max_tick_us);
    sum.balance_moves += st.balance_moves;
    sum.balance_max_hold_us =
        std::max(sum.balance_max_hold_us, st.balance_max_hold_us);
    sum.extent_returns += st.extent_returns;
    sum.deferred_removes += st.deferred_removes;
    sum.inflight += st.inflight;
    sum.prep_holds += st.prep_holds;
    sum.reclaimed_slots += st.reclaimed_slots;
    // Keep every backend counter in this aggregation. Omitting
    // rebalanced_extents once made dfkv_slab_rebalanced_total read zero
    // fleet-wide despite observed extent movement.
    sum.rebalanced_extents += st.rebalanced_extents;
    sum.batched_writes += st.batched_writes;
    sum.uring_write_batches += st.uring_write_batches;
    sum.metadata_io_errors += st.metadata_io_errors;
    sum.unclean_resets += st.unclean_resets;
    sum.warm_restarts += st.warm_restarts;
    sum.trust_tokens_held += st.trust_tokens_held;
    sum.eviction_record_clears += st.eviction_record_clears;
    sum.record_writes += st.record_writes;
    sum.table_rebuilt += st.table_rebuilt;
    sum.rebuild_corrupt_records += st.rebuild_corrupt_records;
    sum.rebuild_rejected_records += st.rebuild_rejected_records;
    sum.rebuild_scanned_bytes += st.rebuild_scanned_bytes;
    sum.rebuild_scan_chunks += st.rebuild_scan_chunks;
    sum.rebuild_sparse_ranges += st.rebuild_sparse_ranges;
    sum.rebuild_sequential_fallbacks += st.rebuild_sequential_fallbacks;
    sum.rebuild_mmap_scans += st.rebuild_mmap_scans;
    sum.capacity_bytes += st.capacity_bytes;
    sum.allocated_bytes += st.allocated_bytes;
    sum.payload_bytes += st.payload_bytes;
    sum.allocator_objects += st.allocator_objects;
    sum.committed_objects += st.committed_objects;
    sum.class_count += st.class_count;
    sum.bound_extents += st.bound_extents;
    sum.pool_extents += st.pool_extents;
    sum.failed_disks += st.failed_disks;
    sum.failed = sum.failed || st.failed;
  }
  return sum;
}

std::vector<SlabAllocator::ClassStat> DiskCacheGroup::SlabClasses() const {
  std::map<uint32_t, SlabAllocator::ClassStat> by_size;
  for (const auto* disk : slabs_) {
    for (const auto& cls : disk->ClassStats()) {
      auto& sum = by_size[cls.slot_size];
      sum.slot_size = cls.slot_size;
      sum.free_slots += cls.free_slots;
      sum.resident += cls.resident;
      sum.puts += cls.puts;
      sum.extents += cls.extents;
      sum.slots_per_extent =
          std::max(sum.slots_per_extent, cls.slots_per_extent);
      sum.useful_bytes += cls.useful_bytes;
      sum.allocated_bytes += cls.allocated_bytes;
      sum.read_touches += cls.read_touches;
      sum.read_heat += cls.read_heat;
      sum.tail_age = std::max(sum.tail_age, cls.tail_age);  // the oldest disk
    }
  }
  std::vector<SlabAllocator::ClassStat> out;
  out.reserve(by_size.size());
  for (auto& entry : by_size) out.push_back(entry.second);
  return out;
}


bool DiskCacheGroup::IsCached(const BlockKey& key) const {
  StoreEngine* d = Route(key);
  return d != nullptr && d->IsCached(key);
}

uint64_t DiskCacheGroup::UsedBytes() const {
  uint64_t t = 0;
  for (const auto& d : disks_) t += d->UsedBytes();
  return t;
}
uint64_t DiskCacheGroup::TenantUsedBytes(uint64_t tenant_hash) const {
  // What a failed disk holds can no longer be read: it is not charged.
  uint64_t total = 0;
  for (const auto& disk : disks_)
    if (disk->Healthy()) total += disk->TenantUsedBytes(tenant_hash);
  return total;
}

std::vector<DiskCacheGroup::TenantQuotaMetric>
DiskCacheGroup::ConfiguredTenantQuotaMetrics() const {
  std::vector<TenantQuotaMetric> metrics;
  std::map<uint64_t, uint64_t> quotas;  // copied: the stripes come first
  {
    std::shared_lock<std::shared_mutex> lock(tenant_quotas_mu_);
    quotas = tenant_quotas_;
  }
  metrics.reserve(quotas.size());
  for (const auto& quota : quotas) {
    std::lock_guard<std::mutex> tenant_lock(
        tenant_stripes_[TenantStripe(quota.first)]);
    uint64_t rejections = 0;
    {
      std::shared_lock<std::shared_mutex> lock(tenant_quotas_mu_);
      const auto rejected =
          tenant_quota_rejections_by_hash_.find(quota.first);
      if (rejected != tenant_quota_rejections_by_hash_.end())
        rejections = rejected->second;
    }
    metrics.push_back(TenantQuotaMetric{quota.first, quota.second,
                                        TenantUsedBytes(quota.first),
                                        rejections});
  }
  return metrics;
}


size_t DiskCacheGroup::Count() const {
  size_t t = 0;
  for (const auto& d : disks_) t += d->Count();
  return t;
}

uint64_t DiskCacheGroup::Evictions() const {
  uint64_t t = 0;
  for (const auto& d : disks_) t += d->Evictions();
  return t;
}

uint64_t DiskCacheGroup::EvictedBytes() const {
  uint64_t t = 0;
  for (const auto& d : disks_) t += d->EvictedBytes();
  return t;
}

}  // namespace dfkv
