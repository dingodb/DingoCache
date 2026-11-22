/* Which storage engine a cache directory belongs to, judged by the names
 * directly inside it (docs/design/log-structured-engine.md §10.1). Used to
 * resolve DFKV_STORE_ENGINE=auto per directory and to refuse an engine that
 * does not own what a directory holds. Advisory: the engine that opens the
 * directory checks its own files again under its own lock. */
#ifndef DFKV_STORE_DIR_CLASS_H_
#define DFKV_STORE_DIR_CLASS_H_

#include <string>
#include <vector>

namespace dfkv {

enum class StoreDirClass {
  kEmpty,     // no entry, or only lost+found (a directory that does not exist
              // yet is empty too)
  kSlab,      // slab markers and no log markers, whatever else is present
  kLog,       // log-engine markers and no slab markers, whatever else
  kFile,      // the file engine's blocks/ and neither of the above
  kConflict,  // slab and log markers both
  kForeign,   // anything else
};
const char* StoreDirClassName(StoreDirClass c);

struct StoreDirClassification {
  StoreDirClass cls = StoreDirClass::kEmpty;
  // The entries that decided the class (markers, or for kForeign the first
  // few unrecognised names), for the refusal message.
  std::vector<std::string> deciding;
  // lost+found holds something, or cannot be listed: a file system check
  // put files there. It does not change the class (the slab and file engines
  // ignore lost+found as they always did); the log engine refuses such a
  // directory, as it does foreign content.
  bool lost_found_in_use = false;
};

// False (with *error set) if the directory exists and cannot be listed, or
// the path is not a directory.
bool ClassifyStoreDir(const std::string& dir, StoreDirClassification* out,
                      std::string* error);

}  // namespace dfkv

#endif  // DFKV_STORE_DIR_CLASS_H_
