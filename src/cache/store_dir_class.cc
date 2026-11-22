#include "cache/store_dir_class.h"

#include <algorithm>
#include <filesystem>
#include <system_error>

namespace fs = std::filesystem;

namespace dfkv {
namespace {

bool IsSlabMarker(const std::string& name) {
  return name == "slab_meta" || name == "slots.tbl" || name == "slab_state" ||
         name == "slab_initializing" || name == "extents";
}
bool IsLogMarker(const std::string& name) {
  return name == "lse_super" || name == "lse_journal" ||
         name == "lse_initializing" || name == "segments";
}
constexpr size_t kForeignNamesShown = 5;

}  // namespace

const char* StoreDirClassName(StoreDirClass c) {
  switch (c) {
    case StoreDirClass::kEmpty: return "empty";
    case StoreDirClass::kSlab: return "slab";
    case StoreDirClass::kLog: return "log";
    case StoreDirClass::kFile: return "file";
    case StoreDirClass::kConflict: return "slab and log";
    case StoreDirClass::kForeign: return "foreign";
  }
  return "unknown";
}

bool ClassifyStoreDir(const std::string& dir, StoreDirClassification* out,
                      std::string* error) {
  *out = StoreDirClassification{};
  std::error_code ec;
  const auto status = fs::status(dir, ec);
  if (ec && ec != std::errc::no_such_file_or_directory) {
    *error = "cannot inspect cache directory " + dir + ": " + ec.message();
    return false;
  }
  if (!fs::exists(status)) return true;  // the engine creates it: empty
  if (!fs::is_directory(status)) {
    *error = "cache directory " + dir + " is not a directory";
    return false;
  }
  std::vector<std::string> slab, log, file, other;
  for (fs::directory_iterator it(dir, ec), end; !ec && it != end;
       it.increment(ec)) {
    const std::string name = it->path().filename().string();
    if (name == "lost+found") {
      std::error_code lf_ec;
      out->lost_found_in_use =
          !it->is_directory(lf_ec) || lf_ec ||
          !fs::is_empty(it->path(), lf_ec) || lf_ec;
      continue;
    }
    if (IsSlabMarker(name)) slab.push_back(name);
    else if (IsLogMarker(name)) log.push_back(name);
    else if (name == "blocks") file.push_back(name);
    else other.push_back(name);
  }
  if (ec) {
    *error = "cannot list cache directory " + dir + ": " + ec.message();
    return false;
  }
  for (auto* names : {&slab, &log, &other}) std::sort(names->begin(), names->end());
  if (!slab.empty() && !log.empty()) {
    out->cls = StoreDirClass::kConflict;
    out->deciding = slab;
    out->deciding.insert(out->deciding.end(), log.begin(), log.end());
  } else if (!slab.empty()) {
    out->cls = StoreDirClass::kSlab;
    out->deciding = slab;
  } else if (!log.empty()) {
    out->cls = StoreDirClass::kLog;
    out->deciding = log;
  } else if (!file.empty()) {
    out->cls = StoreDirClass::kFile;
    out->deciding = file;
  } else if (!other.empty()) {
    out->cls = StoreDirClass::kForeign;
    other.resize(std::min(other.size(), kForeignNamesShown));
    out->deciding = other;
  }
  return true;
}

}  // namespace dfkv
