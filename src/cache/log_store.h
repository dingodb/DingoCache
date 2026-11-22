/* LogStore -- the log-structured storage engine (`log`), as designed in
 * docs/design/log-structured-engine.md. Values are appended to fixed-size
 * segment files as checksummed records; an in-memory index maps keys to
 * records; REMOVE writes durable kill entries to a metadata journal; recovery
 * rebuilds everything from the disk alone.
 *
 * This is implementation stage "LogStore core" (design §14): one open
 * segment, recovery by walking every segment, no sealing, no eviction (a full
 * store answers kCacheFull), no RETAIN. It is selectable only with
 * DFKV_LSE_EXPERIMENTAL=1 and writes the development-format bit. */
#ifndef DFKV_LOG_STORE_H_
#define DFKV_LOG_STORE_H_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cache/lse_format.h"
#include "cache/lse_io.h"
#include "cache/store_engine.h"

namespace dfkv {

class LogStoreTestPeer;

class LogStore : public StoreEngine {
 public:
  struct Options {
    std::string dir;
    uint64_t capacity_bytes = 1ull << 30;
    // O_DIRECT segment I/O with preallocated segments (the deployment mode);
    // false: buffered, segments sparse (tests, tmpfs).
    bool direct_writes = false;
    // 0: derived from capacity_bytes (design §4.7); otherwise a power of two
    // in [4 MiB, 1 GiB]. Used only when the store is created.
    uint64_t segment_bytes = 0;
    // Leave the store opened but not activated; see Activate().
    bool defer_activation = false;
    // The I/O this store does; null = the real system calls.
    std::shared_ptr<lse::IoBackend> io;
    // Test hook, called at named points (see log_store.cc); a test may _exit
    // there to stage a crash.
    std::function<void(const char* point)> crash_hook;
    // Test hook: a PUT stored its value with this generation (called with
    // the store's mutex held, on the PUT's thread, just before it returns).
    std::function<void(const BlockKey&, uint64_t)> on_commit;
    // Test hook: a PUT reserved its space (called with the store's mutex
    // held, on the PUT's thread, right after the reservation).
    std::function<void(const BlockKey&)> on_reserve;
    // Test hook: a REMOVE took the store's mutex (called with it held).
    std::function<void(const BlockKey&)> on_remove;
    // DFKV_LSE_SYNC_MS: the open segment is fdatasynced this often, which
    // bounds what a power loss can take of acknowledged writes (§6.5).
    uint32_t sync_ms = 1000;
  };
  // Reads the DFKV_LSE_* settings into `o` (a malformed value keeps the
  // default, with a warning, as every setting does).
  static void ApplyEnvironment(Options* o);

  // Geometry the design derives from a capacity (§4.7). False if the
  // capacity is below the minimum.
  struct Geometry {
    uint64_t segment_bytes = 0;
    uint32_t segment_count = 0;
  };
  static bool GeometryFor(uint64_t capacity_bytes, uint64_t segment_override,
                          Geometry* out, std::string* error);

  explicit LogStore(Options opt, bool* ok = nullptr);
  ~LogStore() override;
  LogStore(const LogStore&) = delete;
  LogStore& operator=(const LogStore&) = delete;

  bool Healthy() const override;
  const std::string& StartupError() const override { return startup_error_; }
  bool Activate() override;
  bool Close() override;

  Status Cache(const BlockKey& key, const void* data, size_t len) override;
  Status CacheDirect(const BlockKey& key, char* data, size_t len,
                     size_t cap) override;
  Status Range(const BlockKey& key, uint64_t offset, uint64_t length,
               std::string* out, size_t* value_len = nullptr) override;
  Status RangeInto(const BlockKey& key, uint64_t offset, uint64_t length,
                   char* dst, size_t dst_cap, size_t* out_len,
                   size_t* value_len = nullptr) override;
  Status RangeDirect(const BlockKey& key, uint64_t offset, uint64_t length,
                     char* io_buf, size_t io_cap, const char** out_data,
                     size_t* out_len, size_t* value_len = nullptr) override;
  Status RangeDirectPrep(const BlockKey& key, uint64_t offset, uint64_t length,
                         size_t io_cap, ReadLease* out) override;
  bool IsCached(const BlockKey& key) const override;
  Status Lookup(const BlockKey& key, ValueMetadata* out) const override;
  Status Remove(const BlockKey& key) override;

  uint64_t UsedBytes() const override;
  uint64_t TenantUsedBytes(uint64_t tenant_hash) const override;
  size_t Count() const override;
  uint64_t Evictions() const override { return 0; }
  uint64_t EvictedBytes() const override { return 0; }
  const std::string& Dir() const override { return opt_.dir; }
  uint64_t MaxValueBytes() const override;
  const char* EngineName() const override { return "log"; }

  // What the last open recovered (tests, the open log line).
  struct RecoveryStats {
    uint64_t segments_walked = 0;
    uint64_t records_recovered = 0;
    uint64_t records_killed = 0;
    uint64_t records_rejected = 0;   // bad header or payload CRC
    uint64_t records_superseded = 0; // lost to a higher generation
    uint64_t journal_blocks = 0;
    bool journal_damaged = false;
  };
  RecoveryStats GetRecoveryStats() const { return recovery_; }

 private:
  friend class LogStoreTestPeer;
  struct KeyHash {
    size_t operator()(const BlockKey& k) const {
      return static_cast<size_t>(k.digest_hi ^ (k.digest_lo * 0x9e3779b97f4a7c15ULL) ^
                                 (k.tenant_hash << 1));
    }
  };
  // Where a record is: its segment, the incarnation it was written in, and
  // its offset in 4 KiB blocks.
  struct Location {
    uint32_t segment = 0;
    uint64_t incarnation = 0;
    uint32_t offset_blocks = 0;
  };
  struct Entry {
    Location loc;
    uint64_t value_len = 0;
    uint64_t generation = 0;
    uint32_t payload_crc = 0;
    // Where an inline payload starts in the header block (the record's
    // header_len); 0: the payload follows the header block.
    uint16_t inline_at = 0;
    uint64_t write_lsn = 0;
  };
  static uint64_t Footprint(const Entry& e);
  struct Shadow {
    Location loc;
    uint64_t generation = 0;
    bool kill_pending = false;  // its KILL is enqueued, durable at `ticket`
    uint64_t ticket = 0;
  };
  enum class SegState { kFree, kOpen, kFull };
  struct Segment {
    SegState state = SegState::kFree;
    bool header_valid = false;  // its current header is durable and ours
    uint64_t incarnation = 0;
    uint64_t append_off = lse::kBlockBytes;  // next record offset (bytes)
    uint32_t records = 0;       // records placed in this incarnation
    uint32_t live = 0;          // index entries pointing into it
    uint32_t shadows = 0;       // shadows pointing into it
    uint32_t pins = 0;          // reads and writes in flight
    bool dirty = false;         // written by this run: synced at Close
    bool unsynced = false;      // written since the syncer's last pass
    int fd = -1;                // buffered descriptor
    int dio_fd = -1;            // O_DIRECT descriptor (direct mode)
  };
  // A PUT in flight, for keep-first joins and the REMOVE fence.
  struct Flight {
    std::condition_variable cv;
    bool done = false;
    bool cancelled = false;  // a REMOVE took the key while it was written
    Status result = Status::kIOError;
    Location loc;
    uint64_t generation = 0;
  };
  class Journal;

  // ---- open / init / recovery ----
  bool SetError(std::string error);
  bool SetErrno(std::string what, int err);
  bool LockDirectory();
  bool InitStore(const Geometry& g);
  bool OpenStore();
  bool ReadSuperblock();
  bool WriteSuperblock();
  bool OpenSegmentFiles(bool create);
  bool ReadJournal();
  bool WalkSegment(uint32_t index, std::vector<lse::RecordHeader>* records,
                   std::vector<uint32_t>* offsets);
  bool ResetToEmpty();
  bool FreshHeader(uint32_t index);  // with mu_ held or before serving
  std::string SegmentPath(uint32_t index) const;
  bool Serving() const;
  void FailStore(const char* what, int err);
  void Hook(const char* point) const {
    if (opt_.crash_hook) opt_.crash_hook(point);
  }

  // ---- write path ----
  template <typename WriteFn>
  Status Put(const BlockKey& key, size_t len, const WriteFn& write);
  bool ReserveLocked(uint64_t footprint, uint32_t* segment, uint64_t* offset);
  bool OpenNextSegmentLocked();
  uint64_t NextLsnLocked();
  bool WriteRecord(uint32_t segment, uint64_t offset,
                   const lse::RecordHeader& h, const char* payload,
                   size_t len, char* direct_buf, size_t direct_cap);

  // ---- read path ----
  bool AcquireLocked(const BlockKey& key, Entry* out);
  void Unpin(uint32_t segment);
  uint64_t PayloadOffset(const Entry& e) const;
  bool ReadPayload(const Entry& e, uint64_t offset, size_t n, char* dst);
  void DropCorruptLocked(const BlockKey& key, const Entry& e);
  static void ReleaseLeaseThunk(void* owner, uint64_t token) noexcept;

  void AddShadowLocked(const BlockKey& key, const Location& loc,
                       uint64_t generation, uint64_t kill_ticket = 0);
  void EraseEntryLocked(const BlockKey& key);
  bool SegmentIsFreeLocked(uint32_t index) const;

  Options opt_;
  std::shared_ptr<lse::IoBackend> io_;
  std::string startup_error_;
  bool ok_ = false;
  std::atomic<bool> failed_{false};
  std::atomic<bool> activated_{false};
  std::atomic<bool> closed_{false};
  int dir_fd_ = -1;
  int super_fd_ = -1;
  int journal_fd_ = -1;

  lse::Superblock sb_;
  std::vector<Segment> segments_;
  uint32_t open_segment_ = UINT32_MAX;
  std::vector<uint32_t> free_;  // free segments, in the order they are taken

  mutable std::mutex mu_;
  std::unordered_map<BlockKey, Entry, KeyHash> index_;
  std::unordered_multimap<BlockKey, Shadow, KeyHash> shadows_;
  std::unordered_map<BlockKey, std::shared_ptr<Flight>, KeyHash> flights_;
  std::unordered_map<uint64_t, uint64_t> tenant_bytes_;
  uint64_t used_bytes_ = 0;
  uint64_t next_lsn_ = 1;
  uint64_t next_incarnation_ = 1;

  // Recovery results kept until Activate().
  struct KillKey {
    uint32_t segment;
    uint64_t incarnation;
    uint32_t offset_blocks;
    bool operator==(const KillKey& o) const {
      return segment == o.segment && incarnation == o.incarnation &&
             offset_blocks == o.offset_blocks;
    }
  };
  struct KillKeyHash {
    size_t operator()(const KillKey& k) const {
      return static_cast<size_t>(k.incarnation * 0x9e3779b97f4a7c15ULL ^
                                 (uint64_t(k.segment) << 32) ^ k.offset_blocks);
    }
  };
  std::unordered_set<KillKey, KillKeyHash> kills_;
  uint64_t journal_stop_offset_ = 0;
  uint64_t journal_next_seq_ = 1;
  bool needs_reset_ = false;
  RecoveryStats recovery_;

  std::unique_ptr<Journal> journal_;
  // The periodic sync of the open segment (§6.5).
  void SyncerLoop();
  bool stop_syncer_ = false;
  std::condition_variable syncer_cv_;
  std::thread syncer_;
};

}  // namespace dfkv

#endif  // DFKV_LOG_STORE_H_
