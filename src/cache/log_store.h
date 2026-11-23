/* LogStore -- the log-structured storage engine (`log`), as designed in
 * docs/design/log-structured-engine.md. Values are appended to fixed-size
 * segment files as checksummed records; an in-memory index maps keys to
 * records; REMOVE writes durable kill entries to a metadata journal; recovery
 * rebuilds everything from the disk alone.
 *
 * Implemented so far (design §14): the core (stage 3) with one open client
 * segment; sealing with summaries and trailers, summary-based recovery and
 * the clean close (stage 5). No eviction yet (a full store answers
 * kCacheFull), no RETAIN. It is selectable only with
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
    // there to stage a crash. Called without the store's mutex, except in
    // Activate().
    std::function<void(const char* point)> crash_hook;
    // Test hook: a PUT stored its value with this generation (called with
    // the store's mutex held, on the PUT's thread, just before it returns).
    std::function<void(const BlockKey&, uint64_t)> on_commit;
    // Test hook: a PUT reserved its space (called with the store's mutex
    // held, on the PUT's thread, right after the reservation).
    std::function<void(const BlockKey&)> on_reserve;
    // Test hook: a REMOVE took the store's mutex (called with it held).
    std::function<void(const BlockKey&)> on_remove;

    // DFKV_LSE_SYNC_MS: open segments are fdatasynced this often (§6.5).
    uint32_t sync_ms = 1000;
  };
  // Reads the DFKV_LSE_* settings into `o` (a malformed
  // value keeps the default, with a warning, as every setting does).
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
    uint64_t summaries_loaded = 0;
    uint64_t records_recovered = 0;
    uint64_t records_killed = 0;
    uint64_t records_rejected = 0;   // bad header or payload CRC
    uint64_t records_superseded = 0; // lost to a higher generation
    uint64_t journal_blocks = 0;
    bool journal_damaged = false;
  };
  RecoveryStats GetRecoveryStats() const { return recovery_; }

  // Counters since the open (tests).
  struct Stats {
    uint64_t segments_sealed = 0;
    uint64_t cache_full = 0;
    uint64_t read_crc_failures = 0;
    uint64_t journal_resets = 0;
    // Current state.
    uint32_t segments_free = 0;
    uint32_t segments_sealed_now = 0;
    uint32_t sealer_queue = 0;
    uint64_t shadows = 0;
    uint32_t segment_count = 0;
  };
  Stats GetStats() const;

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
    bool operator==(const Location& o) const {
      return segment == o.segment && incarnation == o.incarnation &&
             offset_blocks == o.offset_blocks;
    }
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
  enum class SegState { kFree, kOpen, kSealing, kSealed, kWalked };
  struct Segment {
    SegState state = SegState::kFree;
    bool header_valid = false;  // its current header is durable and ours
    uint64_t incarnation = 0;
    uint64_t append_off = lse::kBlockBytes;  // next record offset (bytes)
    uint32_t records = 0;       // records placed in this incarnation
    uint32_t live = 0;          // index entries pointing into it
    uint32_t shadows = 0;       // shadows pointing into it
    uint32_t pins = 0;          // reads and writes in flight
    uint32_t writers = 0;       // writes in flight
    bool dirty = false;         // written by this run: synced at Close
    bool unsynced = false;      // written since the last periodic sync
    uint64_t last_lsn = 0;      // highest write_lsn placed (seal order)
    // Records written in this incarnation: the summary it gets when sealed.
    std::vector<lse::SummaryEntry> summary;
    int fd = -1;                // buffered descriptor
    int dio_fd = -1;            // O_DIRECT descriptor (direct mode)
  };
  // A PUT in flight, for keep-first joins and the REMOVE fence.
  struct Flight {
    std::condition_variable cv;
    bool done = false;
    bool cancelled = false;  // a REMOVE took the key while it was written
    Status result = Status::kIOError;
    bool has_loc = false;  // space reserved: `loc` names it
    Location loc;
    uint64_t generation = 0;
  };
  class Journal;
  using Lock = std::unique_lock<std::mutex>;

  // ---- open / init / recovery ----
  bool SetError(std::string error);
  bool SetErrno(std::string what, int err);
  bool LockDirectory();
  bool InitStore(const Geometry& g);
  bool OpenStore();
  bool ReadSuperblock();
  bool WriteSuperblock();
  bool OpenSegmentFile(uint32_t index, bool create);
  bool OpenSegmentFiles(bool create);
  bool ReadJournal();
  bool LoadSummary(uint32_t index, std::vector<lse::SummaryEntry>* records,
                   bool* loaded);
  bool WalkSegment(uint32_t index, std::vector<lse::SummaryEntry>* records);
  bool ResetToEmpty();
  // A new incarnation for the segment, durable. `empty`: also a trailer
  // with no records, so that a free segment is not walked at the next open.
  bool FreshHeader(uint32_t index, bool empty = false);  // with mu_ held
  bool SyncDir(const std::string& path);
  std::string SegmentPath(uint32_t index) const;
  bool Serving() const;
  void FailStore(const char* what, int err);
  void Hook(const char* point) const {
    if (opt_.crash_hook) opt_.crash_hook(point);
  }

  // ---- write path ----
  template <typename WriteFn>
  Status Put(const BlockKey& key, size_t len, const WriteFn& write);
  bool ReserveLocked(Lock& lk, uint64_t footprint, uint32_t* segment,
                     uint64_t* offset);
  bool PlaceLocked(uint32_t segment, uint64_t footprint, uint64_t* offset);
  // Takes a free segment and gives it a fresh header.
  bool TakeFreeSegmentLocked(uint32_t* out);
  // Hands an open segment to the sealer; waits while its queue is full.
  void HandToSealerLocked(Lock& lk, uint32_t segment);
  uint64_t NextLsnLocked();
  bool WriteRecord(uint32_t segment, uint64_t offset,
                   const lse::RecordHeader& h, const char* payload,
                   size_t len, char* direct_buf, size_t direct_cap);
  void RecordWrittenLocked(uint32_t segment, const BlockKey& key,
                           const lse::RecordHeader& h, uint64_t offset);
  uint64_t EnqueueKillLocked(const BlockKey& key, const Location& loc,
                             uint64_t generation);

  // ---- sealing (§6.3) ----
  bool SealSegment(Lock& lk, uint32_t segment);
  void SealerLoop();

  // A sealed segment whose records are all killed is free again.
  void RemoveFromFifoLocked(uint32_t segment);
  void MaybeFreeLocked(uint32_t segment);

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

  Options opt_;
  std::shared_ptr<lse::IoBackend> io_;
  std::string startup_error_;
  bool ok_ = false;
  std::atomic<bool> failed_{false};
  std::atomic<bool> activated_{false};
  std::atomic<bool> closed_{false};
  int dir_fd_ = -1;
  int super_fd_ = -1;
  int journal_fd_ = -1;  // owned by journal_ once it exists

  lse::Superblock sb_;
  std::vector<Segment> segments_;
  uint32_t open_segment_ = UINT32_MAX;
  std::vector<uint32_t> free_;   // free segments, in the order they are taken
  std::deque<uint32_t> fifo_;    // sealed segments, oldest first
  std::deque<uint32_t> seal_queue_;

  mutable std::mutex mu_;
  std::condition_variable state_cv_;  // writers, sealer queue, free segments
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
  Stats stats_;

  std::unique_ptr<Journal> journal_;
  bool stop_threads_ = false;
  std::thread sealer_;
};

}  // namespace dfkv

#endif  // DFKV_LOG_STORE_H_
