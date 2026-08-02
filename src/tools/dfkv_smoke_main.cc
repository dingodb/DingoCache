/* dfkv_smoke — connect to a dfkv cluster and verify put/get/exist round-trips.
 * Usage: dfkv_smoke --members "n1=ip:port,n2=ip:port" [--size BYTES]
 * Exit 0 = OK, non-zero = failure. No GPU/RDMA needed. */
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "client/kv_client.h"
#include "common/version.h"

using namespace dfkv;  // NOLINT

static std::vector<std::pair<std::string, std::string>> ParseMembers(const std::string& s) {
  std::vector<std::pair<std::string, std::string>> out;
  size_t i = 0;
  while (i <= s.size()) {
    size_t c = s.find(',', i);
    if (c == std::string::npos) c = s.size();
    std::string tok = s.substr(i, c - i);
    size_t eq = tok.find('=');
    if (eq != std::string::npos) out.emplace_back(tok.substr(0, eq), tok.substr(eq + 1));
    if (c == s.size()) break;
    i = c + 1;
  }
  return out;
}

int main(int argc, char** argv) {
  if (WantsVersion(argc, argv)) { std::printf("dfkv_smoke %s\n", Version()); return 0; }
  std::string members;
  size_t size = 4096;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--members")) members = argv[i + 1];
    else if (!std::strcmp(argv[i], "--size")) size = std::stoul(argv[i + 1]);
  }
  auto mem = ParseMembers(members);
  if (mem.empty()) { std::fprintf(stderr, "need --members name=ip:port,...\n"); return 2; }

  KVClient c(mem, "dfkv/smoke");
  std::string v(size, '\0');
  for (size_t i = 0; i < size; ++i) v[i] = static_cast<char>(i & 0xFF);
  // Salt the key with the payload size: on a persistent write-once ring, a
  // fixed key re-probed with a different --size read back the OLD payload and
  // reported a spurious data mismatch (bench avoids the same trap via pid+size).
  const std::string key = "dfkv-smoke-key-" + std::to_string(size);

  if (!c.Put(key, v.data(), v.size())) { std::fprintf(stderr, "PUT failed\n"); return 1; }
  if (!c.Exist(key)) { std::fprintf(stderr, "EXIST failed\n"); return 1; }
  std::string out(size, '\0');
  if (!c.Get(key, &out[0], out.size())) { std::fprintf(stderr, "GET miss\n"); return 1; }
  if (out != v) { std::fprintf(stderr, "GET data mismatch\n"); return 1; }
  std::printf("dfkv smoke OK (%zu nodes, %zu bytes)\n", mem.size(), size);
  return 0;
}
