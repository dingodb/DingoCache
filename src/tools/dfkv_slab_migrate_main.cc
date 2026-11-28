// Explicit offline STL3 -> LSE migration. This changes no default engine or
// configured directory: an operator must separately choose the published copy.
#include "cache/slab_migration.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace {

bool WriteReport(const fs::path& path, const std::string& bytes) {
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
  if (fd < 0) {
    std::cerr << "cannot create migration report " << path << ": " << std::strerror(errno) << '\n';
    return false;
  }
  size_t done = 0;
  while (done < bytes.size()) {
    const ssize_t n = ::write(fd, bytes.data() + done, bytes.size() - done);
    if (n > 0) done += static_cast<size_t>(n);
    else if (n < 0 && errno == EINTR) continue;
    else {
      std::cerr << "cannot write migration report: " << std::strerror(errno) << '\n';
      ::close(fd);
      return false;
    }
  }
  const bool ok = ::fdatasync(fd) == 0;
  const int saved = errno;
  ::close(fd);
  if (!ok) std::cerr << "cannot sync migration report: " << std::strerror(saved) << '\n';
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0]
              << " ABSOLUTE_CLEAN_STL3_ROOT ABSOLUTE_NEW_LSE_ROOT ABSOLUTE_NEW_REPORT.json\n"
                 "Requires an offline/quiesced source. Refuses tombstones, retention, "
                 "unknown rows and existing output/staging paths. Never switches a service.\n";
    return 2;
  }
  const fs::path report_path(argv[3]);
  if (!report_path.is_absolute() || report_path != report_path.lexically_normal()) {
    std::cerr << "report path must be absolute and lexically normalized\n";
    return 2;
  }
  struct stat report_stat {};
  if (::lstat(report_path.c_str(), &report_stat) == 0 || errno != ENOENT) {
    std::cerr << "report already exists or cannot be inspected\n";
    return 2;
  }
  std::string error;
  auto source = dfkv::slab_migration::SlabSnapshot::Open(argv[1], &error);
  if (!source) {
    std::cerr << "source preflight failed: " << error << '\n';
    return 1;
  }
  std::error_code ec;
  const fs::path report_parent = fs::canonical(report_path.parent_path(), ec);
  const fs::path canonical_source = ec ? fs::path() : fs::canonical(source->dir(), ec);
  auto within = [](const fs::path& path, const fs::path& root) {
    auto p = path.begin(), r = root.begin();
    for (; r != root.end(); ++r, ++p)
      if (p == path.end() || *p != *r) return false;
    return true;
  };
  const fs::path destination = fs::path(argv[2]).lexically_normal();
  if (ec || report_parent != report_path.parent_path() ||
      within(report_path, canonical_source) || within(report_path, destination) ||
      report_path == destination || report_path == destination.string() + ".sto14-staging") {
    std::cerr << "report parent must be canonical and report must not alias source/output paths\n";
    return 2;
  }
  dfkv::slab_migration::MigrationResult result =
      dfkv::slab_migration::MigrateToLse(*source, argv[2]);
  const std::string json = dfkv::slab_migration::ResultJson(result) + "\n";
  if (!WriteReport(report_path, json)) return 3;
  if (!result.ok) {
    std::cerr << (result.published
                      ? "migration candidate published but completion/durability verification failed: "
                      : "migration failed without publishing a partial candidate: ")
              << result.error << "; report=" << report_path << '\n';
    return 1;
  }
  std::cout << json;
  return 0;
}
