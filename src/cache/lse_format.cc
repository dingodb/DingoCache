#include "cache/lse_format.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include "utils/crc32c.h"

namespace dfkv {
namespace lse {
namespace {

constexpr size_t kCrcOffset = kBlockBytes - 4;

void PutU16(uint8_t* p, uint16_t v) {
  p[0] = static_cast<uint8_t>(v);
  p[1] = static_cast<uint8_t>(v >> 8);
}
void PutU32(uint8_t* p, uint32_t v) {
  for (int i = 0; i < 4; ++i) p[i] = static_cast<uint8_t>(v >> (8 * i));
}
void PutU64(uint8_t* p, uint64_t v) {
  for (int i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>(v >> (8 * i));
}
uint16_t GetU16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0] | (p[1] << 8));
}
uint32_t GetU32(const uint8_t* p) {
  uint32_t v = 0;
  for (int i = 3; i >= 0; --i) v = (v << 8) | p[i];
  return v;
}
uint64_t GetU64(const uint8_t* p) {
  uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
  return v;
}
void PutKey(uint8_t* p, const BlockKey& k) {
  PutU64(p, k.digest_hi);
  PutU64(p + 8, k.digest_lo);
  PutU64(p + 16, k.tenant_hash);
}
BlockKey GetKey(const uint8_t* p) {
  BlockKey k;
  k.digest_hi = GetU64(p);
  k.digest_lo = GetU64(p + 8);
  k.tenant_hash = GetU64(p + 16);
  return k;
}
void PutUuid(uint8_t* p, const Uuid& u) { std::memcpy(p, u.data(), u.size()); }
Uuid GetUuid(const uint8_t* p) {
  Uuid u;
  std::memcpy(u.data(), p, u.size());
  return u;
}

// CRC-32C of a 4 KiB block with its last four bytes taken as zero.
uint32_t BlockCrc(const uint8_t* block) {
  static const uint8_t kZero[4] = {0, 0, 0, 0};
  return Crc32c(kZero, 4, Crc32c(block, kCrcOffset));
}
void SealBlock(Block* b) { PutU32(b->data() + kCrcOffset, BlockCrc(b->data())); }
bool BlockCrcOk(const uint8_t* block) {
  return GetU32(block + kCrcOffset) == BlockCrc(block);
}
bool MagicIs(const uint8_t* p, const char* magic, size_t n) {
  return std::memcmp(p, magic, n) == 0;
}
bool PowerOfTwo(uint64_t v) { return v != 0 && (v & (v - 1)) == 0; }

// Superblock layout.
constexpr char kSuperMagic[8] = {'D', 'F', 'K', 'V', 'L', 'S', 'E', '1'};
constexpr size_t kSbMajor = 8, kSbMinor = 12, kSbIncompat = 16,
                 kSbRoCompat = 24, kSbCompat = 32, kSbUuid = 40,
                 kSbSequence = 56, kSbSegmentBytes = 64, kSbSegmentCount = 72,
                 kSbAlign = 76, kSbIncarnationReserved = 80,
                 kSbLsnReserved = 88, kSbClean = 96, kSbReset = 100,
                 kSbCreatedBy = 104, kSbCreatedByBytes = 32;
// Segment header layout.
constexpr char kSegmentMagic[8] = {'D', 'F', 'L', 'S', 'S', 'E', 'G', '1'};
constexpr size_t kShUuid = 8, kShIndex = 24, kShIncarnation = 32;
// Record header layout.
constexpr char kRecordMagic[4] = {'D', 'L', 'R', '1'};
constexpr size_t kRhType = 4, kRhFlags = 5, kRhHeaderLen = 6, kRhUuid = 8,
                 kRhIncarnation = 24, kRhKey = 32, kRhValueLen = 56,
                 kRhGeneration = 64, kRhWriteLsn = 72, kRhPayloadCrc = 80,
                 kRhHeaderCrc = 84;
constexpr uint8_t kKnownRecordFlags = kRecordFlagInline | kRecordFlagRelocated;
// Summary entry layout.
constexpr size_t kSeKey = 0, kSeValueLen = 24, kSeGeneration = 32,
                 kSeWriteLsn = 40, kSeOffset = 48, kSePayloadCrc = 52,
                 kSeFlags = 56, kSeHeaderLen = 58;
// Trailer layout.
constexpr char kTrailerMagic[8] = {'D', 'F', 'L', 'S', 'E', 'N', 'D', '1'};
constexpr size_t kTrUuid = 8, kTrIncarnation = 24, kTrCount = 32,
                 kTrSummaryOffset = 40, kTrSummaryLength = 48,
                 kTrSummaryCrc = 56, kTrLastLsn = 64;
// Journal block layout.
constexpr char kJournalMagic[8] = {'D', 'F', 'L', 'S', 'J', 'R', 'N', '1'};
constexpr size_t kJbUuid = 8, kJbSequence = 24, kJbCount = 32;
constexpr size_t kJeType = 0, kJeSegment = 4, kJeIncarnation = 8,
                 kJeOffset = 16, kJeRetainUntil = 20, kJeRetainTtl = 24,
                 kJeKey = 32, kJeGeneration = 56;

constexpr uint64_t kMinSegmentBytes = 4ull << 20;
constexpr uint64_t kMaxSegmentBytes = 1ull << 30;
constexpr uint32_t kMinSegmentCount = 8;

}  // namespace

const char* DecodeErrorName(DecodeError e) {
  switch (e) {
    case DecodeError::kOk: return "ok";
    case DecodeError::kBadMagic: return "bad magic";
    case DecodeError::kBadCrc: return "bad checksum";
    case DecodeError::kForeignStore: return "another store";
    case DecodeError::kUnsupported: return "unsupported format or feature";
    case DecodeError::kDevelopmentFormat: return "development format";
    case DecodeError::kReadOnlyFeature: return "unknown read-only feature";
    case DecodeError::kBadField: return "field out of range";
  }
  return "unknown";
}

// ---- superblock ------------------------------------------------------------
Block EncodeSuperblock(const Superblock& sb) {
  Block b{};
  std::memcpy(b.data(), kSuperMagic, sizeof(kSuperMagic));
  PutU32(&b[kSbMajor], sb.format_major);
  PutU32(&b[kSbMinor], sb.format_minor);
  PutU64(&b[kSbIncompat], sb.incompat_features);
  PutU64(&b[kSbRoCompat], sb.ro_compat_features);
  PutU64(&b[kSbCompat], sb.compat_features);
  PutUuid(&b[kSbUuid], sb.store_uuid);
  PutU64(&b[kSbSequence], sb.sequence);
  PutU64(&b[kSbSegmentBytes], sb.segment_bytes);
  PutU32(&b[kSbSegmentCount], sb.segment_count);
  PutU32(&b[kSbAlign], sb.align);
  PutU64(&b[kSbIncarnationReserved], sb.incarnation_reserved);
  PutU64(&b[kSbLsnReserved], sb.lsn_reserved);
  PutU32(&b[kSbClean], sb.clean);
  PutU32(&b[kSbReset], sb.reset_in_progress);
  std::memcpy(&b[kSbCreatedBy], sb.created_by.data(),
              std::min(sb.created_by.size(), kSbCreatedByBytes));
  SealBlock(&b);
  return b;
}

DecodeError DecodeSuperblock(const uint8_t* b, bool allow_development,
                             Superblock* out) {
  if (!MagicIs(b, kSuperMagic, sizeof(kSuperMagic)))
    return DecodeError::kBadMagic;
  if (!BlockCrcOk(b)) return DecodeError::kBadCrc;
  Superblock sb;
  sb.format_major = GetU32(&b[kSbMajor]);
  sb.format_minor = GetU32(&b[kSbMinor]);
  sb.incompat_features = GetU64(&b[kSbIncompat]);
  sb.ro_compat_features = GetU64(&b[kSbRoCompat]);
  sb.compat_features = GetU64(&b[kSbCompat]);
  if (sb.format_major != kFormatMajor ||
      (sb.incompat_features & ~kKnownIncompat) != 0)
    return DecodeError::kUnsupported;
  if ((sb.incompat_features & kIncompatDevelopmentFormat) != 0 &&
      !allow_development)
    return DecodeError::kDevelopmentFormat;
  if ((sb.ro_compat_features & ~kKnownRoCompat) != 0)
    return DecodeError::kReadOnlyFeature;
  sb.store_uuid = GetUuid(&b[kSbUuid]);
  sb.sequence = GetU64(&b[kSbSequence]);
  sb.segment_bytes = GetU64(&b[kSbSegmentBytes]);
  sb.segment_count = GetU32(&b[kSbSegmentCount]);
  sb.align = GetU32(&b[kSbAlign]);
  sb.incarnation_reserved = GetU64(&b[kSbIncarnationReserved]);
  sb.lsn_reserved = GetU64(&b[kSbLsnReserved]);
  sb.clean = GetU32(&b[kSbClean]);
  sb.reset_in_progress = GetU32(&b[kSbReset]);
  const char* name = reinterpret_cast<const char*>(&b[kSbCreatedBy]);
  sb.created_by.assign(name, strnlen(name, kSbCreatedByBytes));
  for (size_t i = sb.created_by.size(); i < kSbCreatedByBytes; ++i)
    if (name[i] != 0) return DecodeError::kBadField;  // zero after the name
  if (sb.align != kBlockBytes || !PowerOfTwo(sb.segment_bytes) ||
      sb.segment_bytes < kMinSegmentBytes ||
      sb.segment_bytes > kMaxSegmentBytes ||
      sb.segment_count < kMinSegmentCount || sb.clean > 1 ||
      sb.reset_in_progress > 1 || sb.incarnation_reserved == 0 ||
      sb.lsn_reserved == 0)
    return DecodeError::kBadField;
  *out = std::move(sb);
  return DecodeError::kOk;
}

// ---- segment header --------------------------------------------------------
Block EncodeSegmentHeader(const SegmentHeader& h) {
  Block b{};
  std::memcpy(b.data(), kSegmentMagic, sizeof(kSegmentMagic));
  PutUuid(&b[kShUuid], h.store_uuid);
  PutU32(&b[kShIndex], h.segment_index);
  PutU64(&b[kShIncarnation], h.incarnation);
  SealBlock(&b);
  return b;
}

DecodeError DecodeSegmentHeader(const uint8_t* b, const Uuid& store_uuid,
                                SegmentHeader* out) {
  if (!MagicIs(b, kSegmentMagic, sizeof(kSegmentMagic)))
    return DecodeError::kBadMagic;
  if (!BlockCrcOk(b)) return DecodeError::kBadCrc;
  SegmentHeader h;
  h.store_uuid = GetUuid(&b[kShUuid]);
  if (h.store_uuid != store_uuid) return DecodeError::kForeignStore;
  h.segment_index = GetU32(&b[kShIndex]);
  h.incarnation = GetU64(&b[kShIncarnation]);
  if (h.incarnation == 0) return DecodeError::kBadField;
  *out = h;
  return DecodeError::kOk;
}

// ---- record header ---------------------------------------------------------
uint64_t RecordFootprint(const RecordHeader& h) {
  if ((h.flags & kRecordFlagInline) != 0) return kBlockBytes;
  return kBlockBytes + (h.value_len + kBlockBytes - 1) / kBlockBytes * kBlockBytes;
}

bool EncodeRecordHeader(const RecordHeader& h, const void* inline_payload,
                        Block* out) {
  if (h.header_len < kRecordHeaderBytes || h.header_len > kBlockBytes)
    return false;
  const bool is_inline = (h.flags & kRecordFlagInline) != 0;
  if (is_inline != RecordFitsInline(h.value_len, h.header_len)) return false;
  if (is_inline && h.value_len != 0 && inline_payload == nullptr) return false;
  Block& b = *out;
  b.fill(0);
  std::memcpy(b.data(), kRecordMagic, sizeof(kRecordMagic));
  b[kRhType] = h.type;
  b[kRhFlags] = h.flags;
  PutU16(&b[kRhHeaderLen], h.header_len);
  PutUuid(&b[kRhUuid], h.store_uuid);
  PutU64(&b[kRhIncarnation], h.incarnation);
  PutKey(&b[kRhKey], h.key);
  PutU64(&b[kRhValueLen], h.value_len);
  PutU64(&b[kRhGeneration], h.generation);
  PutU64(&b[kRhWriteLsn], h.write_lsn);
  PutU32(&b[kRhPayloadCrc], h.payload_crc);
  PutU32(&b[kRhHeaderCrc], Crc32c(b.data(), h.header_len));
  if (is_inline && h.value_len != 0)
    std::memcpy(&b[h.header_len], inline_payload, h.value_len);
  return true;
}

DecodeError DecodeRecordHeader(const uint8_t* b, const Uuid& store_uuid,
                               uint64_t incarnation, uint64_t segment_bytes,
                               RecordHeader* out) {
  if (!MagicIs(b, kRecordMagic, sizeof(kRecordMagic)))
    return DecodeError::kBadMagic;
  const uint16_t header_len = GetU16(&b[kRhHeaderLen]);
  if (header_len < kRecordHeaderBytes || header_len > kBlockBytes)
    return DecodeError::kBadCrc;  // the length the CRC would cover is unknown
  uint8_t copy[kBlockBytes];
  std::memcpy(copy, b, header_len);
  PutU32(&copy[kRhHeaderCrc], 0);
  if (GetU32(&b[kRhHeaderCrc]) != Crc32c(copy, header_len))
    return DecodeError::kBadCrc;
  // A longer header is a later version's extension, unknown here: zero.
  if (std::any_of(b + kRecordHeaderBytes, b + header_len,
                  [](uint8_t c) { return c != 0; }))
    return DecodeError::kBadField;
  RecordHeader h;
  h.header_len = header_len;
  h.store_uuid = GetUuid(&b[kRhUuid]);
  if (h.store_uuid != store_uuid) return DecodeError::kForeignStore;
  h.incarnation = GetU64(&b[kRhIncarnation]);
  if (h.incarnation != incarnation) return DecodeError::kBadField;
  h.type = b[kRhType];
  h.flags = b[kRhFlags];
  h.key = GetKey(&b[kRhKey]);
  h.value_len = GetU64(&b[kRhValueLen]);
  h.generation = GetU64(&b[kRhGeneration]);
  h.write_lsn = GetU64(&b[kRhWriteLsn]);
  h.payload_crc = GetU32(&b[kRhPayloadCrc]);
  if (h.type != kRecordValue || (h.flags & ~kKnownRecordFlags) != 0 ||
      h.generation == 0 || h.write_lsn < h.generation)
    return DecodeError::kBadField;
  // The bound keeps value_len + 4095 from wrapping in every later sum.
  if (h.value_len > MaxRecordValue(segment_bytes) ||
      ((h.flags & kRecordFlagInline) != 0) !=
          RecordFitsInline(h.value_len, header_len))
    return DecodeError::kBadField;
  *out = h;
  return DecodeError::kOk;
}

// ---- summary ---------------------------------------------------------------
std::vector<uint8_t> EncodeSummary(const std::vector<SummaryEntry>& entries) {
  std::vector<uint8_t> out(entries.size() * kSummaryEntryBytes, 0);
  for (size_t i = 0; i < entries.size(); ++i) {
    uint8_t* p = out.data() + i * kSummaryEntryBytes;
    const SummaryEntry& e = entries[i];
    PutKey(p + kSeKey, e.key);
    PutU64(p + kSeValueLen, e.value_len);
    PutU64(p + kSeGeneration, e.generation);
    PutU64(p + kSeWriteLsn, e.write_lsn);
    PutU32(p + kSeOffset, e.offset_blocks);
    PutU32(p + kSePayloadCrc, e.payload_crc);
    p[kSeFlags] = e.flags;
    PutU16(p + kSeHeaderLen, e.header_len);
  }
  return out;
}

bool DecodeSummary(const uint8_t* bytes, size_t len, uint64_t segment_bytes,
                   std::vector<SummaryEntry>* out) {
  if (len % kSummaryEntryBytes != 0) return false;
  std::vector<SummaryEntry> entries(len / kSummaryEntryBytes);
  for (size_t i = 0; i < entries.size(); ++i) {
    const uint8_t* p = bytes + i * kSummaryEntryBytes;
    SummaryEntry& e = entries[i];
    e.key = GetKey(p + kSeKey);
    e.value_len = GetU64(p + kSeValueLen);
    e.generation = GetU64(p + kSeGeneration);
    e.write_lsn = GetU64(p + kSeWriteLsn);
    e.offset_blocks = GetU32(p + kSeOffset);
    e.payload_crc = GetU32(p + kSePayloadCrc);
    e.flags = p[kSeFlags];
    e.header_len = GetU16(p + kSeHeaderLen);
    if (e.header_len < kRecordHeaderBytes || e.header_len > kBlockBytes ||
        e.offset_blocks == 0 || e.generation == 0 ||
        e.write_lsn < e.generation || (e.flags & ~kKnownRecordFlags) != 0 ||
        e.value_len > MaxRecordValue(segment_bytes))
      return false;
    // Inline exactly when it fits, as in the record header; the record ends
    // before the trailer.
    const bool is_inline = (e.flags & kRecordFlagInline) != 0;
    if (is_inline != RecordFitsInline(e.value_len, e.header_len)) return false;
    RecordHeader h;
    h.flags = e.flags;
    h.value_len = e.value_len;
    const uint64_t end =
        uint64_t(e.offset_blocks) * kBlockBytes + RecordFootprint(h);
    if (segment_bytes < kBlockBytes || end > segment_bytes - kBlockBytes)
      return false;
  }
  *out = std::move(entries);
  return true;
}

// ---- trailer ---------------------------------------------------------------
Block EncodeTrailer(const Trailer& t) {
  Block b{};
  std::memcpy(b.data(), kTrailerMagic, sizeof(kTrailerMagic));
  PutUuid(&b[kTrUuid], t.store_uuid);
  PutU64(&b[kTrIncarnation], t.incarnation);
  PutU32(&b[kTrCount], t.record_count);
  PutU64(&b[kTrSummaryOffset], t.summary_offset);
  PutU64(&b[kTrSummaryLength], t.summary_length);
  PutU32(&b[kTrSummaryCrc], t.summary_crc);
  PutU64(&b[kTrLastLsn], t.last_write_lsn);
  SealBlock(&b);
  return b;
}

DecodeError DecodeTrailer(const uint8_t* b, const Uuid& store_uuid,
                          uint64_t incarnation, uint64_t segment_bytes,
                          Trailer* out) {
  if (!MagicIs(b, kTrailerMagic, sizeof(kTrailerMagic)))
    return DecodeError::kBadMagic;
  if (!BlockCrcOk(b)) return DecodeError::kBadCrc;
  Trailer t;
  t.store_uuid = GetUuid(&b[kTrUuid]);
  if (t.store_uuid != store_uuid) return DecodeError::kForeignStore;
  t.incarnation = GetU64(&b[kTrIncarnation]);
  if (t.incarnation != incarnation) return DecodeError::kBadField;
  t.record_count = GetU32(&b[kTrCount]);
  t.summary_offset = GetU64(&b[kTrSummaryOffset]);
  t.summary_length = GetU64(&b[kTrSummaryLength]);
  t.summary_crc = GetU32(&b[kTrSummaryCrc]);
  t.last_write_lsn = GetU64(&b[kTrLastLsn]);
  if (t.summary_length !=
          static_cast<uint64_t>(t.record_count) * kSummaryEntryBytes ||
      t.summary_offset < kBlockBytes || t.summary_offset % kBlockBytes != 0)
    return DecodeError::kBadField;
  // Between the segment header and the trailer; checked without overflow.
  const uint64_t limit =
      segment_bytes < 2 * kBlockBytes ? 0 : segment_bytes - kBlockBytes;
  if (t.summary_offset > limit || t.summary_length > limit - t.summary_offset)
    return DecodeError::kBadField;
  *out = t;
  return DecodeError::kOk;
}

// ---- journal block ---------------------------------------------------------
bool EncodeJournalBlock(const JournalBlock& jb, Block* out) {
  if (jb.entries.size() > kJournalEntriesPerBlock) return false;
  Block& b = *out;
  b.fill(0);
  std::memcpy(b.data(), kJournalMagic, sizeof(kJournalMagic));
  PutUuid(&b[kJbUuid], jb.store_uuid);
  PutU64(&b[kJbSequence], jb.sequence);
  const size_t n = jb.entries.size();
  PutU16(&b[kJbCount], static_cast<uint16_t>(n));
  for (size_t i = 0; i < n; ++i) {
    uint8_t* p = &b[kJournalHeaderBytes + i * kJournalEntryBytes];
    const JournalEntry& e = jb.entries[i];
    p[kJeType] = e.type;
    PutU32(p + kJeSegment, e.segment_index);
    PutU64(p + kJeIncarnation, e.incarnation);
    PutU32(p + kJeOffset, e.offset_blocks);
    PutU32(p + kJeRetainUntil, e.retain_until);
    PutU32(p + kJeRetainTtl, e.retain_ttl);
    PutKey(p + kJeKey, e.key);
    PutU64(p + kJeGeneration, e.generation);
  }
  SealBlock(&b);
  return true;
}

DecodeError DecodeJournalBlock(const uint8_t* b, const Uuid& store_uuid,
                               JournalBlock* out) {
  if (!MagicIs(b, kJournalMagic, sizeof(kJournalMagic)))
    return DecodeError::kBadMagic;
  if (!BlockCrcOk(b)) return DecodeError::kBadCrc;
  JournalBlock jb;
  jb.store_uuid = GetUuid(&b[kJbUuid]);
  if (jb.store_uuid != store_uuid) return DecodeError::kForeignStore;
  jb.sequence = GetU64(&b[kJbSequence]);
  const size_t n = GetU16(&b[kJbCount]);
  if (jb.sequence == 0 || n > kJournalEntriesPerBlock)
    return DecodeError::kBadField;
  jb.entries.resize(n);
  for (size_t i = 0; i < n; ++i) {
    const uint8_t* p = &b[kJournalHeaderBytes + i * kJournalEntryBytes];
    JournalEntry& e = jb.entries[i];
    e.type = p[kJeType];
    e.segment_index = GetU32(p + kJeSegment);
    e.incarnation = GetU64(p + kJeIncarnation);
    e.offset_blocks = GetU32(p + kJeOffset);
    e.retain_until = GetU32(p + kJeRetainUntil);
    e.retain_ttl = GetU32(p + kJeRetainTtl);
    e.key = GetKey(p + kJeKey);
    e.generation = GetU64(p + kJeGeneration);
    const bool kill = e.type == kJournalKill;
    if ((!kill && e.type != kJournalRetain) || e.generation == 0 ||
        (kill && (e.incarnation == 0 || e.offset_blocks == 0)))
      return DecodeError::kBadField;
  }
  // Slots past the count are zero: a block is written once, whole.
  const uint8_t* rest = &b[kJournalHeaderBytes + n * kJournalEntryBytes];
  const uint8_t* end = &b[kJournalHeaderBytes +
                          kJournalEntriesPerBlock * kJournalEntryBytes];
  if (std::any_of(rest, end, [](uint8_t c) { return c != 0; }))
    return DecodeError::kBadField;
  *out = std::move(jb);
  return DecodeError::kOk;
}

}  // namespace lse
}  // namespace dfkv
