#include "cache/kv_node_server.h"
#include "transport/wire.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace dfkv;  // NOLINT

namespace {

class AlignedBuffer {
 public:
  explicit AlignedBuffer(size_t size) : size_(size) {
    void* p = nullptr;
    EXPECT_EQ(::posix_memalign(&p, 4096, size), 0);
    data_.reset(static_cast<char*>(p));
  }
  char* data() const { return data_.get(); }
  size_t size() const { return size_; }

 private:
  struct Free {
    void operator()(char* p) const { std::free(p); }
  };
  std::unique_ptr<char, Free> data_;
  size_t size_;
};

BlockKey Key(uint64_t n) { return BlockKey{n, n + 1, 7}; }

Status Put(KvNodeServer* server, const BlockKey& key,
           const std::string& value) {
  std::string ignored;
  return server->ProcessRequestForKey(
      static_cast<uint8_t>(WireOp::kCache), key, 0, 0, value.data(),
      value.size(), &ignored);
}

TEST(PreparedRead, CompletionAndFailureAreSingleTerminalActions) {
  ::setenv("DFKV_READ_COALESCE", "1", 1);
  ::setenv("DFKV_READ_COALESCE_TIMEOUT_MS", "25", 1);
  ::unsetenv("DFKV_RAM_TIER");
  const fs::path dir =
      fs::temp_directory_path() / "dfkv_prepared_read_completion";
  fs::remove_all(dir);
  fs::create_directories(dir);
  {
  KvNodeServer server(dir.string(), 1ull << 30);
  ASSERT_TRUE(server.Healthy());

  const std::string value(8192, 'p');
  ASSERT_EQ(Put(&server, Key(1), value), Status::kOk);
  AlignedBuffer staging(1 << 20);

  PreparedRead read =
      server.PrepareReadForKey(Key(1), 0, value.size(), staging.data(),
                               staging.size());
  ASSERT_EQ(read.status(), Status::kOk);
  ASSERT_TRUE(read.owns_cleanup());
  ASSERT_TRUE(read.needs_io());
  const ssize_t got = ::pread(read.fd(), read.staging(), read.aligned_len(),
                              static_cast<off_t>(read.aligned_off()));
  ASSERT_GE(got, 0);
  ASSERT_GE(static_cast<size_t>(got), read.head() + read.payload_len());
  EXPECT_EQ(std::memcmp(read.data(), value.data(), value.size()), 0);
  EXPECT_TRUE(read.Commit(Status::kOk, read.payload_len(), 0.001));
  EXPECT_FALSE(read.Commit(Status::kOk, read.payload_len(), 0.001));
  EXPECT_FALSE(read.Abort());

  PreparedRead failed =
      server.PrepareReadForKey(Key(1), 0, value.size(), staging.data(),
                               staging.size());
  ASSERT_EQ(failed.status(), Status::kOk);
  EXPECT_TRUE(failed.Commit(Status::kIOError, 0, 0.001));
  EXPECT_FALSE(failed.Abort());

  }
  fs::remove_all(dir);
  ::unsetenv("DFKV_READ_COALESCE");
  ::unsetenv("DFKV_READ_COALESCE_TIMEOUT_MS");
}

TEST(PreparedRead, FallbackAbortAndDestructorReleaseFlight) {
  ::setenv("DFKV_READ_COALESCE", "1", 1);
  ::unsetenv("DFKV_RAM_TIER");
  ::setenv("DFKV_READ_COALESCE_TIMEOUT_MS", "25", 1);
  const fs::path dir =
      fs::temp_directory_path() / "dfkv_prepared_read_abort";
  fs::remove_all(dir);
  fs::create_directories(dir);
  {
  KvNodeServer server(dir.string(), 1ull << 30);
  ASSERT_TRUE(server.Healthy());

  const std::string value(4096, 'a');
  ASSERT_EQ(Put(&server, Key(2), value), Status::kOk);
  AlignedBuffer first_staging(1 << 20);
  AlignedBuffer second_staging(1 << 20);

  PreparedRead leader =
      server.PrepareReadForKey(Key(2), 0, value.size(), first_staging.data(),
                               first_staging.size());
  ASSERT_EQ(leader.status(), Status::kOk);
  PreparedRead fallback =
      server.PrepareReadForKey(Key(2), 0, value.size(), second_staging.data(),
                               second_staging.size());
  EXPECT_EQ(fallback.status(), Status::kInvalid);
  EXPECT_FALSE(fallback.owns_cleanup());

  // A follower times out defensively while the RAII leader remains alive, then
  // performs its own synchronous read without stealing cleanup ownership.
  Status follower_status = Status::kIOError;
  std::string follower_value;
  std::thread follower([&] {
    const char* data = nullptr;
    size_t len = 0;
    size_t value_len = 0;
    follower_status = server.RangeDirectForKey(
        Key(2), 0, value.size(), second_staging.data(),
        second_staging.size(), &data, &len, &value_len);
    if (follower_status == Status::kOk && data != nullptr)
      follower_value.assign(data, len);
  });
  follower.join();
  EXPECT_EQ(follower_status, Status::kOk);
  EXPECT_EQ(follower_value, value);
  EXPECT_TRUE(leader.Abort());  // connection-abort path
  EXPECT_FALSE(leader.Abort());

  {
    PreparedRead abandoned =
        server.PrepareReadForKey(Key(2), 0, value.size(), first_staging.data(),
                                 first_staging.size());
    ASSERT_EQ(abandoned.status(), Status::kOk);
    ASSERT_TRUE(abandoned.owns_cleanup());
  }  // destructor-abort must remove the coalescer flight

  PreparedRead retry =
      server.PrepareReadForKey(Key(2), 0, value.size(), second_staging.data(),
                               second_staging.size());
  EXPECT_EQ(retry.status(), Status::kOk);
  EXPECT_TRUE(retry.Abort());

  }
  fs::remove_all(dir);
  ::unsetenv("DFKV_READ_COALESCE");
  ::unsetenv("DFKV_READ_COALESCE_TIMEOUT_MS");
}

}  // namespace
