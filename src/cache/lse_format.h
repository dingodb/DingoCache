/* On-disk format of the log-structured engine (`log`), as specified in
 * docs/design/log-structured-engine.md §4. Encoders and decoders only: no I/O,
 * no state. Every fixed-size structure is one 4 KiB block whose last four
 * bytes hold the CRC-32C of the block with those bytes zero; every integer is
 * little-endian whatever the host. A decoder never trusts its input: it
 * checks the magic, the CRC and every field it can check, and reports which
 * check failed.
 *
 * What a decoder accepts, the matching encoder writes back byte for byte, and
 * an encoder refuses what it cannot write faithfully.
 *
 * The engine itself does not exist yet (implementation stage 1). Stores are
 * created with the development-format incompatible-feature bit set until the
 * format is frozen; a decoder accepts that bit only when asked to. */
#ifndef DFKV_LSE_FORMAT_H_
#define DFKV_LSE_FORMAT_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/kv_types.h"

namespace dfkv {
namespace lse {

constexpr size_t kBlockBytes = 4096;
using Uuid = std::array<uint8_t, 16>;
using Block = std::array<uint8_t, kBlockBytes>;

constexpr uint32_t kFormatMajor = 1;
constexpr uint32_t kFormatMinor = 0;
// Incompatible feature: written by a development build, before the format is
// frozen. A store carrying it is opened only by a build that asks for it.
constexpr uint64_t kIncompatDevelopmentFormat = 1ull << 63;
// Feature bits this build understands (none besides the development bit).
constexpr uint64_t kKnownIncompat = kIncompatDevelopmentFormat;
constexpr uint64_t kKnownRoCompat = 0;

enum class DecodeError {
  kOk = 0,
  kBadMagic,          // not this kind of block (never written, or another one)
  kBadCrc,            // torn or damaged
  kForeignStore,      // another store's uuid
  kUnsupported,       // a format major or incompat feature this build lacks
  kDevelopmentFormat, // the development bit, and the caller did not allow it
  kReadOnlyFeature,   // an unknown ro_compat feature: do not serve
  kBadField,          // a field out of its valid range
};
const char* DecodeErrorName(DecodeError e);

// ---- superblock (lse_super, two copies at offsets 0 and 4096) -------------
struct Superblock {
  uint32_t format_major = kFormatMajor;
  uint32_t format_minor = kFormatMinor;
  uint64_t incompat_features = 0;
  uint64_t ro_compat_features = 0;
  uint64_t compat_features = 0;
  Uuid store_uuid{};
  uint64_t sequence = 0;
  uint64_t segment_bytes = 0;
  uint32_t segment_count = 0;
  uint32_t align = static_cast<uint32_t>(kBlockBytes);
  uint64_t incarnation_reserved = 0;
  uint64_t lsn_reserved = 0;
  uint32_t clean = 0;
  uint32_t reset_in_progress = 0;
  std::string created_by;  // at most 32 bytes are kept
};
Block EncodeSuperblock(const Superblock& sb);
// `allow_development`: accept the development-format bit.
DecodeError DecodeSuperblock(const uint8_t* block, bool allow_development,
                             Superblock* out);

// ---- segment header (offset 0 of a segment file) ---------------------------
struct SegmentHeader {
  Uuid store_uuid{};
  uint32_t segment_index = 0;
  uint64_t incarnation = 0;
};
Block EncodeSegmentHeader(const SegmentHeader& h);
// `store_uuid`: the superblock's; another uuid is kForeignStore.
DecodeError DecodeSegmentHeader(const uint8_t* block, const Uuid& store_uuid,
                                SegmentHeader* out);

// ---- record header block (in front of every record) ------------------------
constexpr uint8_t kRecordValue = 1;
constexpr uint8_t kRecordFlagInline = 1u << 0;
constexpr uint8_t kRecordFlagRelocated = 1u << 1;
// Bytes of the header proper; an inline payload follows it in the block.
constexpr size_t kRecordHeaderBytes = 88;
constexpr size_t kMaxInlinePayload = kBlockBytes - kRecordHeaderBytes;

struct RecordHeader {
  uint8_t type = kRecordValue;
  uint8_t flags = 0;
  // Bytes of the header this record uses, in [kRecordHeaderBytes, 4096]: the
  // header CRC covers them, and an inline payload starts there. A decoder
  // reports what the block says; this build writes kRecordHeaderBytes.
  uint16_t header_len = kRecordHeaderBytes;
  Uuid store_uuid{};
  uint64_t incarnation = 0;
  BlockKey key;
  uint64_t value_len = 0;
  uint64_t generation = 0;
  uint64_t write_lsn = 0;
  uint32_t payload_crc = 0;  // CRC-32C of the value_len payload bytes
};
// A record's payload is inline exactly when it fits in the header block.
inline bool RecordFitsInline(uint64_t value_len,
                             size_t header_len = kRecordHeaderBytes) {
  return header_len <= kBlockBytes && value_len <= kBlockBytes - header_len;
}
// The largest value a record in a segment of `segment_bytes` can hold: the
// segment header, the record's header block and the trailer take one block
// each.
inline uint64_t MaxRecordValue(uint64_t segment_bytes) {
  return segment_bytes < 3 * kBlockBytes ? 0 : segment_bytes - 3 * kBlockBytes;
}
// False, and `out` untouched, unless header_len is in range and the inline
// flag is set exactly when the value fits inline; an inline record needs its
// payload (value_len bytes at `inline_payload`), which is copied into the
// block at header_len.
bool EncodeRecordHeader(const RecordHeader& h, const void* inline_payload,
                        Block* out);
// Checks the header only: magic, header CRC, uuid, incarnation, fields
// (value_len at most MaxRecordValue(segment_bytes), the inline flag set
// exactly when the value fits). The payload, inline or not, is the caller's
// to check against payload_crc.
DecodeError DecodeRecordHeader(const uint8_t* block, const Uuid& store_uuid,
                               uint64_t incarnation, uint64_t segment_bytes,
                               RecordHeader* out);
// Bytes the record takes in its segment: its header block, plus the payload
// padded to 4 KiB unless the payload is inline. `h` is a header the encoder
// accepts or the decoder returned.
uint64_t RecordFootprint(const RecordHeader& h);

// ---- summary (sealed segments) ---------------------------------------------
constexpr size_t kSummaryEntryBytes = 64;
struct SummaryEntry {
  BlockKey key;
  uint64_t value_len = 0;
  uint64_t generation = 0;
  uint64_t write_lsn = 0;
  uint32_t offset_blocks = 0;  // record offset in the segment, in 4 KiB units
  uint32_t payload_crc = 0;
  uint8_t flags = 0;
  // The record header's header_len: where an inline payload starts.
  uint16_t header_len = kRecordHeaderBytes;
};
std::vector<uint8_t> EncodeSummary(const std::vector<SummaryEntry>& entries);
// `bytes` must be a whole number of entries, every record they describe
// must lie inside a segment of `segment_bytes`, before its trailer, and its
// inline flag must be set exactly when the value fits after header_len.
bool DecodeSummary(const uint8_t* bytes, size_t len, uint64_t segment_bytes,
                   std::vector<SummaryEntry>* out);

// ---- trailer (last 4 KiB of a sealed segment) ------------------------------
struct Trailer {
  Uuid store_uuid{};
  uint64_t incarnation = 0;
  uint32_t record_count = 0;
  uint64_t summary_offset = 0;
  uint64_t summary_length = 0;
  uint32_t summary_crc = 0;
  uint64_t last_write_lsn = 0;
};
Block EncodeTrailer(const Trailer& t);
// Valid only for the given incarnation; also checks that the summary length
// matches the record count and that the summary lies inside a segment of
// `segment_bytes`, after the segment header and before the trailer.
DecodeError DecodeTrailer(const uint8_t* block, const Uuid& store_uuid,
                          uint64_t incarnation, uint64_t segment_bytes,
                          Trailer* out);

// ---- metadata journal block (lse_journal) ----------------------------------
constexpr uint8_t kJournalKill = 1;
constexpr uint8_t kJournalRetain = 2;
constexpr size_t kJournalEntryBytes = 64;
constexpr size_t kJournalHeaderBytes = 40;
constexpr size_t kJournalEntriesPerBlock =
    (kBlockBytes - 4 - kJournalHeaderBytes) / kJournalEntryBytes;

struct JournalEntry {
  uint8_t type = 0;
  // KILL: the record at (segment_index, incarnation, offset_blocks).
  uint32_t segment_index = 0;
  uint64_t incarnation = 0;
  uint32_t offset_blocks = 0;
  BlockKey key;              // KILL and RETAIN
  uint64_t generation = 0;   // KILL and RETAIN
  // RETAIN
  uint32_t retain_until = 0;
  uint32_t retain_ttl = 0;
};
struct JournalBlock {
  Uuid store_uuid{};
  uint64_t sequence = 0;  // 1, 2, 3, ... with no gaps
  std::vector<JournalEntry> entries;  // at most kJournalEntriesPerBlock
};
// False, and `out` untouched, for more than kJournalEntriesPerBlock entries:
// a block never holds less than it was given (a KILL left out would bring a
// removed value back).
bool EncodeJournalBlock(const JournalBlock& b, Block* out);
DecodeError DecodeJournalBlock(const uint8_t* block, const Uuid& store_uuid,
                               JournalBlock* out);

}  // namespace lse
}  // namespace dfkv

#endif  // DFKV_LSE_FORMAT_H_
