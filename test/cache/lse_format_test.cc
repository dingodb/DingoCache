// The log-structured engine's on-disk format (design §4): every structure
// round-trips, every damaged byte is caught, foreign and unknown blocks are
// refused, and arbitrary bytes never decode or crash a decoder.
#include "cache/lse_format.h"

#include <gtest/gtest.h>

#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "utils/crc32c.h"

using namespace dfkv;        // NOLINT
using namespace dfkv::lse;   // NOLINT

namespace {
Uuid MakeUuid(uint8_t seed) {
  Uuid u;
  for (size_t i = 0; i < u.size(); ++i) u[i] = static_cast<uint8_t>(seed + i);
  return u;
}
const Uuid kStore = MakeUuid(1);
const Uuid kOther = MakeUuid(9);

Superblock SampleSuperblock() {
  Superblock sb;
  sb.incompat_features = kIncompatDevelopmentFormat;
  sb.store_uuid = kStore;
  sb.sequence = 7;
  sb.segment_bytes = 32ull << 20;
  sb.segment_count = 32;
  sb.incarnation_reserved = 1024;
  sb.lsn_reserved = 1u << 20;
  sb.clean = 1;
  sb.created_by = "2.41.0-test";
  return sb;
}

RecordHeader SampleRecord() {
  RecordHeader h;
  h.store_uuid = kStore;
  h.incarnation = 9;
  h.key = BlockKey{0x1111, 0x2222, 0x3333};
  h.value_len = 2'873'000;
  h.generation = 100;
  h.write_lsn = 100;
  h.payload_crc = 0xDEADBEEF;
  return h;
}
constexpr uint64_t kSeg = 32ull << 20;  // the segment size the tests decode for

Block Record(const RecordHeader& h, const void* payload = nullptr) {
  Block b;
  EXPECT_TRUE(EncodeRecordHeader(h, payload, &b));
  return b;
}
Block Journal(const JournalBlock& jb) {
  Block b;
  EXPECT_TRUE(EncodeJournalBlock(jb, &b));
  return b;
}
// Recompute a block's trailing CRC, or a record header's CRC over its
// header_len bytes, after a test changed fields.
void Reseal(Block* b) {
  (*b)[4092] = (*b)[4093] = (*b)[4094] = (*b)[4095] = 0;
  const uint32_t c = Crc32c(b->data(), kBlockBytes);
  for (int i = 0; i < 4; ++i) (*b)[4092 + i] = static_cast<uint8_t>(c >> (8 * i));
}
void ResealRecord(Block* b) {
  const size_t len = (*b)[6] | ((*b)[7] << 8);
  if (len < kRecordHeaderBytes || len > kBlockBytes) return;
  for (int i = 0; i < 4; ++i) (*b)[84 + i] = 0;
  const uint32_t c = Crc32c(b->data(), len);
  for (int i = 0; i < 4; ++i) (*b)[84 + i] = static_cast<uint8_t>(c >> (8 * i));
}
void PutLe(uint8_t* p, uint64_t v, size_t n) {
  for (size_t i = 0; i < n; ++i) p[i] = static_cast<uint8_t>(v >> (8 * i));
}

// Single-bit flips spread over the whole block (every 61st byte, three bits
// of each) must be refused.
template <typename Decode>
void SampledBitFlipsAreRefused(const Block& good, Decode decode) {
  for (size_t byte = 0; byte < kBlockBytes; byte += 61) {
    for (int bit = 0; bit < 8; bit += 3) {
      Block bad = good;
      bad[byte] ^= static_cast<uint8_t>(1u << bit);
      EXPECT_FALSE(decode(bad)) << "byte " << byte << " bit " << bit;
    }
  }
}
}  // namespace

TEST(LseFormat, SuperblockRoundTripsAndChecksEverything) {
  const Superblock sb = SampleSuperblock();
  const Block b = EncodeSuperblock(sb);
  Superblock got;
  ASSERT_EQ(DecodeSuperblock(b.data(), true, &got), DecodeError::kOk);
  EXPECT_EQ(got.store_uuid, sb.store_uuid);
  EXPECT_EQ(got.sequence, 7u);
  EXPECT_EQ(got.segment_bytes, 32ull << 20);
  EXPECT_EQ(got.segment_count, 32u);
  EXPECT_EQ(got.incarnation_reserved, 1024u);
  EXPECT_EQ(got.lsn_reserved, 1u << 20);
  EXPECT_EQ(got.clean, 1u);
  EXPECT_EQ(got.created_by, "2.41.0-test");
  // Little-endian regardless of the host: the sequence is at offset 56.
  EXPECT_EQ(b[56], 7);
  EXPECT_EQ(b[57], 0);

  EXPECT_EQ(DecodeSuperblock(b.data(), false, &got),
            DecodeError::kDevelopmentFormat)
      << "a development store opens only when asked";
  SampledBitFlipsAreRefused(b, [](const Block& x) {
    Superblock s;
    return DecodeSuperblock(x.data(), true, &s) == DecodeError::kOk;
  });
}

TEST(LseFormat, SuperblockRefusesUnknownFeaturesAndBadGeometry) {
  Superblock got;
  Superblock sb = SampleSuperblock();
  sb.incompat_features |= 1ull << 3;
  EXPECT_EQ(DecodeSuperblock(EncodeSuperblock(sb).data(), true, &got),
            DecodeError::kUnsupported);
  sb = SampleSuperblock();
  sb.format_major = kFormatMajor + 1;
  EXPECT_EQ(DecodeSuperblock(EncodeSuperblock(sb).data(), true, &got),
            DecodeError::kUnsupported);
  sb = SampleSuperblock();
  sb.ro_compat_features = 1;
  EXPECT_EQ(DecodeSuperblock(EncodeSuperblock(sb).data(), true, &got),
            DecodeError::kReadOnlyFeature);
  sb = SampleSuperblock();
  sb.compat_features = 1ull << 5;  // unknown compatible features are ignored
  EXPECT_EQ(DecodeSuperblock(EncodeSuperblock(sb).data(), true, &got),
            DecodeError::kOk);
  for (auto bad : std::vector<std::function<void(Superblock*)>>{
           [](Superblock* s) { s->segment_bytes = 3ull << 20; },
           [](Superblock* s) { s->segment_bytes = 2ull << 20; },
           [](Superblock* s) { s->segment_bytes = 2ull << 30; },
           [](Superblock* s) { s->segment_count = 7; },
           [](Superblock* s) { s->align = 512; },
           [](Superblock* s) { s->clean = 2; },
           [](Superblock* s) { s->lsn_reserved = 0; },
           [](Superblock* s) { s->incarnation_reserved = 0; }}) {
    sb = SampleSuperblock();
    bad(&sb);
    EXPECT_EQ(DecodeSuperblock(EncodeSuperblock(sb).data(), true, &got),
              DecodeError::kBadField);
  }
}

TEST(LseFormat, SegmentHeaderRoundTripsAndKnowsItsStore) {
  const Block b = EncodeSegmentHeader({kStore, 12, 345});
  SegmentHeader h;
  ASSERT_EQ(DecodeSegmentHeader(b.data(), kStore, &h), DecodeError::kOk);
  EXPECT_EQ(h.segment_index, 12u);
  EXPECT_EQ(h.incarnation, 345u);
  EXPECT_EQ(DecodeSegmentHeader(b.data(), kOther, &h),
            DecodeError::kForeignStore);
  EXPECT_EQ(DecodeSegmentHeader(EncodeSegmentHeader({kStore, 1, 0}).data(),
                                kStore, &h),
            DecodeError::kBadField)
      << "incarnation 0 is never used";
  Block zero{};
  EXPECT_EQ(DecodeSegmentHeader(zero.data(), kStore, &h),
            DecodeError::kBadMagic)
      << "a never-written header";
  SampledBitFlipsAreRefused(b, [](const Block& x) {
    SegmentHeader s;
    return DecodeSegmentHeader(x.data(), kStore, &s) == DecodeError::kOk;
  });
}

TEST(LseFormat, RecordHeaderRoundTripsWithAndWithoutInlinePayload) {
  RecordHeader h = SampleRecord();
  Block b = Record(h);
  RecordHeader got;
  ASSERT_EQ(DecodeRecordHeader(b.data(), kStore, 9, kSeg, &got),
            DecodeError::kOk);
  EXPECT_EQ(got.key.digest_hi, 0x1111u);
  EXPECT_EQ(got.key.digest_lo, 0x2222u);
  EXPECT_EQ(got.key.tenant_hash, 0x3333u);
  EXPECT_EQ(got.value_len, 2'873'000u);
  EXPECT_EQ(got.payload_crc, 0xDEADBEEFu);
  EXPECT_EQ(got.header_len, kRecordHeaderBytes);
  EXPECT_EQ(DecodeRecordHeader(b.data(), kStore, 10, kSeg, &got),
            DecodeError::kBadField)
      << "a record of an earlier incarnation of its segment";
  EXPECT_EQ(DecodeRecordHeader(b.data(), kOther, 9, kSeg, &got),
            DecodeError::kForeignStore);
  // The header CRC covers the header bytes: a flip there is caught.
  for (size_t byte = 0; byte < kRecordHeaderBytes; ++byte) {
    Block bad = b;
    bad[byte] ^= 0x10;
    EXPECT_NE(DecodeRecordHeader(bad.data(), kStore, 9, kSeg, &got),
              DecodeError::kOk)
        << byte;
  }

  const std::string small = "x";
  h.flags = kRecordFlagInline;
  h.value_len = small.size();
  h.payload_crc = Crc32c(small.data(), small.size());
  b = Record(h, small.data());
  ASSERT_EQ(DecodeRecordHeader(b.data(), kStore, 9, kSeg, &got),
            DecodeError::kOk);
  EXPECT_EQ(b[kRecordHeaderBytes], 'x');
  EXPECT_EQ(Crc32c(&b[kRecordHeaderBytes], got.value_len), got.payload_crc);

  h.flags = kRecordFlagInline | 0x80;
  EXPECT_EQ(DecodeRecordHeader(Record(h, small.data()).data(), kStore, 9, kSeg,
                               &got),
            DecodeError::kBadField)
      << "an unknown flag";
  h.flags = kRecordFlagInline;
  h.write_lsn = 99;
  EXPECT_EQ(DecodeRecordHeader(Record(h, small.data()).data(), kStore, 9, kSeg,
                               &got),
            DecodeError::kBadField)
      << "written before it was first committed";
}

// A payload is inline exactly when it fits, both ways; the encoder writes
// nothing it cannot write whole.
TEST(LseFormat, RecordInlineExactlyWhenTheValueFits) {
  const std::string payload(kMaxInlinePayload + 1, 'p');
  Block b;
  RecordHeader h = SampleRecord();
  h.value_len = kMaxInlinePayload;
  h.flags = 0;
  EXPECT_FALSE(EncodeRecordHeader(h, nullptr, &b)) << "fits, but not inline";
  h.flags = kRecordFlagInline;
  EXPECT_FALSE(EncodeRecordHeader(h, nullptr, &b)) << "inline without payload";
  EXPECT_TRUE(EncodeRecordHeader(h, payload.data(), &b));
  h.value_len = kMaxInlinePayload + 1;
  EXPECT_FALSE(EncodeRecordHeader(h, payload.data(), &b))
      << "inline, but it does not fit";
  h.flags = 0;
  EXPECT_TRUE(EncodeRecordHeader(h, nullptr, &b));

  // The decoder refuses a small non-inline record (CRC-valid, so only the
  // field check can).
  RecordHeader got;
  PutLe(&b[56], 1, 8);  // value_len 1, not inline
  ResealRecord(&b);
  EXPECT_EQ(DecodeRecordHeader(b.data(), kStore, 9, kSeg, &got),
            DecodeError::kBadField);
  // And an inline record whose value does not fit.
  h.flags = kRecordFlagInline;
  h.value_len = 10;
  b = Record(h, payload.data());
  PutLe(&b[56], kMaxInlinePayload + 1, 8);
  ResealRecord(&b);
  EXPECT_EQ(DecodeRecordHeader(b.data(), kStore, 9, kSeg, &got),
            DecodeError::kBadField);
}

TEST(LseFormat, RecordHeaderLengthIsReportedAndBounded) {
  RecordHeader h = SampleRecord();
  RecordHeader got;
  // A longer header (a later minor version's extension): the CRC covers it,
  // an inline payload starts after it, and the decoder reports it.
  h.header_len = 200;
  h.flags = kRecordFlagInline;
  h.value_len = kBlockBytes - 200;
  const std::string payload(h.value_len, 'q');
  Block b = Record(h, payload.data());
  ASSERT_EQ(DecodeRecordHeader(b.data(), kStore, 9, kSeg, &got),
            DecodeError::kOk);
  EXPECT_EQ(got.header_len, 200u);
  EXPECT_EQ(b[200], 'q');
  EXPECT_EQ(b[199], 0);
  h.value_len += 1;
  Block unused;
  EXPECT_FALSE(EncodeRecordHeader(h, payload.data(), &unused));
  // Bytes of a longer header this build does not know are zero.
  b[150] = 1;
  ResealRecord(&b);
  EXPECT_EQ(DecodeRecordHeader(b.data(), kStore, 9, kSeg, &got),
            DecodeError::kBadField);
  // Outside 88..4096 the CRC's extent is unknown: refused as damage.
  for (uint64_t len : {0ull, 87ull, 4097ull, 65535ull}) {
    SCOPED_TRACE(len);
    h = SampleRecord();
    Block x = Record(h);
    PutLe(&x[6], len, 2);
    EXPECT_EQ(DecodeRecordHeader(x.data(), kStore, 9, kSeg, &got),
              DecodeError::kBadCrc);
    h.header_len = static_cast<uint16_t>(len);
    EXPECT_FALSE(EncodeRecordHeader(h, nullptr, &x));
  }
}

// value_len is bounded by the segment, so value_len + 4095 never wraps.
TEST(LseFormat, RecordValueLengthIsBoundedByTheSegment) {
  RecordHeader got;
  for (uint64_t len : std::vector<uint64_t>{MaxRecordValue(kSeg) + 1, kSeg,
                                           ~0ull, ~0ull - 4094}) {
    SCOPED_TRACE(len);
    RecordHeader h = SampleRecord();
    h.value_len = len;
    EXPECT_EQ(DecodeRecordHeader(Record(h).data(), kStore, 9, kSeg, &got),
              DecodeError::kBadField);
  }
  RecordHeader h = SampleRecord();
  h.value_len = MaxRecordValue(kSeg);
  ASSERT_EQ(DecodeRecordHeader(Record(h).data(), kStore, 9, kSeg, &got),
            DecodeError::kOk);
  EXPECT_EQ(RecordFootprint(got), kSeg - 2 * kBlockBytes)
      << "the largest record fills the segment between its header and the "
         "trailer";
}

// The footprint follows the header's inline flag (which the decoder ties to
// the value's size), not a size threshold of its own.
TEST(LseFormat, RecordFootprint) {
  auto footprint = [](uint64_t len) {
    RecordHeader h = SampleRecord();
    h.value_len = len;
    h.flags = RecordFitsInline(len) ? kRecordFlagInline : 0;
    const std::string payload(len <= kBlockBytes ? len : 0, 'f');
    RecordHeader got;
    EXPECT_EQ(DecodeRecordHeader(Record(h, payload.data()).data(), kStore, 9,
                                 kSeg, &got),
              DecodeError::kOk);
    return RecordFootprint(got);
  };
  EXPECT_EQ(footprint(1), 4096u);
  EXPECT_EQ(footprint(kMaxInlinePayload), 4096u);
  EXPECT_EQ(footprint(kMaxInlinePayload + 1), 8192u);
  EXPECT_EQ(footprint(4096), 8192u);
  EXPECT_EQ(footprint(4097), 4096u + 8192u);
  const uint64_t page = 2'873'098;  // a 2.74 MiB value
  EXPECT_LE(footprint(page) - page, 8192u);
  // A longer header leaves less room inline.
  EXPECT_TRUE(RecordFitsInline(kBlockBytes - 200, 200));
  EXPECT_FALSE(RecordFitsInline(kBlockBytes - 199, 200));
}

TEST(LseFormat, SummaryAndTrailerRoundTrip) {
  std::vector<SummaryEntry> entries(3);
  for (size_t i = 0; i < entries.size(); ++i) {
    entries[i].key = BlockKey{i + 1, i + 2, i + 3};
    entries[i].value_len = 5000 * (i + 1);
    entries[i].generation = 10 + i;
    entries[i].write_lsn = 20 + i;
    entries[i].offset_blocks = static_cast<uint32_t>(1 + 5 * i);
    entries[i].payload_crc = static_cast<uint32_t>(0xA0 + i);
    entries[i].flags = i == 1 ? kRecordFlagRelocated : 0;
  }
  entries[0].value_len = 100;
  entries[0].flags = kRecordFlagInline;
  const auto bytes = EncodeSummary(entries);
  ASSERT_EQ(bytes.size(), 3 * kSummaryEntryBytes);
  std::vector<SummaryEntry> got;
  ASSERT_TRUE(DecodeSummary(bytes.data(), bytes.size(), kSeg, &got));
  ASSERT_EQ(got.size(), 3u);
  EXPECT_EQ(got[2].key.tenant_hash, 5u);
  EXPECT_EQ(got[1].flags, kRecordFlagRelocated);
  EXPECT_FALSE(DecodeSummary(bytes.data(), bytes.size() - 1, kSeg, &got));

  Trailer t;
  t.store_uuid = kStore;
  t.incarnation = 9;
  t.record_count = 3;
  t.summary_offset = 64 * 4096;
  t.summary_length = bytes.size();
  t.summary_crc = Crc32c(bytes.data(), bytes.size());
  t.last_write_lsn = 22;
  const Block b = EncodeTrailer(t);
  Trailer tg;
  ASSERT_EQ(DecodeTrailer(b.data(), kStore, 9, kSeg, &tg), DecodeError::kOk);
  EXPECT_EQ(tg.summary_crc, t.summary_crc);
  EXPECT_EQ(tg.last_write_lsn, 22u);
  EXPECT_EQ(DecodeTrailer(b.data(), kStore, 8, kSeg, &tg),
            DecodeError::kBadField)
      << "a trailer is valid only for its incarnation";
  t.summary_length += 1;
  EXPECT_EQ(DecodeTrailer(EncodeTrailer(t).data(), kStore, 9, kSeg, &tg),
            DecodeError::kBadField);
  SampledBitFlipsAreRefused(b, [](const Block& x) {
    Trailer s;
    return DecodeTrailer(x.data(), kStore, 9, kSeg, &s) == DecodeError::kOk;
  });
}

// A CRC-valid summary or trailer still names only places inside its segment.
TEST(LseFormat, SummaryAndTrailerStayInsideTheSegment) {
  auto summary_ok = [](SummaryEntry e) {
    const auto bytes = EncodeSummary({e});
    std::vector<SummaryEntry> got;
    return DecodeSummary(bytes.data(), bytes.size(), kSeg, &got);
  };
  SummaryEntry e;
  e.key = BlockKey{1, 2, 3};
  e.generation = 5;
  e.write_lsn = 5;
  e.offset_blocks = 1;
  e.value_len = MaxRecordValue(kSeg);
  EXPECT_TRUE(summary_ok(e)) << "the largest record, at the first offset";
  e.offset_blocks = 2;
  EXPECT_FALSE(summary_ok(e)) << "runs into the trailer";
  e.value_len = ~0ull;
  e.offset_blocks = 1;
  EXPECT_FALSE(summary_ok(e)) << "a length that would wrap";
  e.value_len = 10;
  e.flags = kRecordFlagInline;
  e.offset_blocks = static_cast<uint32_t>(kSeg / kBlockBytes - 2);
  EXPECT_TRUE(summary_ok(e)) << "an inline record just before the trailer";
  e.offset_blocks += 1;
  EXPECT_FALSE(summary_ok(e)) << "an inline record on the trailer";
  e.offset_blocks = 0xFFFFFFFFu;
  EXPECT_FALSE(summary_ok(e)) << "far outside";
  e.offset_blocks = 1;
  e.value_len = kMaxInlinePayload + 1;
  EXPECT_FALSE(summary_ok(e)) << "inline, but too big to be";
  e.flags = 0;
  e.value_len = 10;
  EXPECT_FALSE(summary_ok(e)) << "small enough to be inline, but is not";
  // The header length is kept: an inline payload after a longer header.
  e.flags = kRecordFlagInline;
  e.header_len = 200;
  e.value_len = kBlockBytes - 200;
  EXPECT_TRUE(summary_ok(e));
  {
    const auto bytes = EncodeSummary({e});
    std::vector<SummaryEntry> got;
    ASSERT_TRUE(DecodeSummary(bytes.data(), bytes.size(), kSeg, &got));
    EXPECT_EQ(got[0].header_len, 200u);
  }
  e.value_len += 1;
  EXPECT_FALSE(summary_ok(e)) << "does not fit after that header";
  for (uint16_t len : {0, 87, 4097}) {
    e.header_len = len;
    e.value_len = 10;
    EXPECT_FALSE(summary_ok(e)) << "header_len " << len;
  }

  auto trailer = [](uint64_t offset, uint32_t count) {
    Trailer t;
    t.store_uuid = kStore;
    t.incarnation = 9;
    t.record_count = count;
    t.summary_offset = offset;
    t.summary_length = uint64_t(count) * kSummaryEntryBytes;
    Trailer got;
    return DecodeTrailer(EncodeTrailer(t).data(), kStore, 9, kSeg, &got);
  };
  EXPECT_EQ(trailer(kSeg - 2 * kBlockBytes, 64), DecodeError::kOk)
      << "a summary filling the block before the trailer";
  EXPECT_EQ(trailer(kSeg - 2 * kBlockBytes, 65), DecodeError::kBadField);
  EXPECT_EQ(trailer(kSeg - kBlockBytes, 0), DecodeError::kOk)
      << "an empty summary, at the trailer";
  EXPECT_EQ(trailer(kSeg, 0), DecodeError::kBadField);
  EXPECT_EQ(trailer(256ull << 30, 1), DecodeError::kBadField)
      << "a summary 256 GiB into a 32 MiB segment";
  EXPECT_EQ(trailer(~0ull & ~4095ull, 1), DecodeError::kBadField);
  EXPECT_EQ(trailer(kBlockBytes, 0xFFFFFFFFu), DecodeError::kBadField)
      << "a count whose summary is larger than the segment";
}

TEST(LseFormat, JournalBlockRoundTripsKillsAndRetains) {
  JournalBlock jb;
  jb.store_uuid = kStore;
  jb.sequence = 5;
  JournalEntry kill;
  kill.type = kJournalKill;
  kill.segment_index = 3;
  kill.incarnation = 44;
  kill.offset_blocks = 17;
  kill.key = BlockKey{1, 2, 3};
  kill.generation = 900;
  JournalEntry retain;
  retain.type = kJournalRetain;
  retain.key = BlockKey{4, 5, 6};
  retain.generation = 901;
  retain.retain_until = 1'800'000'000u;
  retain.retain_ttl = 600;
  jb.entries = {kill, retain};
  const Block b = Journal(jb);
  JournalBlock got;
  ASSERT_EQ(DecodeJournalBlock(b.data(), kStore, &got), DecodeError::kOk);
  EXPECT_EQ(got.sequence, 5u);
  ASSERT_EQ(got.entries.size(), 2u);
  EXPECT_EQ(got.entries[0].incarnation, 44u);
  EXPECT_EQ(got.entries[0].offset_blocks, 17u);
  EXPECT_EQ(got.entries[1].retain_until, 1'800'000'000u);
  EXPECT_EQ(DecodeJournalBlock(b.data(), kOther, &got),
            DecodeError::kForeignStore);
  SampledBitFlipsAreRefused(b, [](const Block& x) {
    JournalBlock s;
    return DecodeJournalBlock(x.data(), kStore, &s) == DecodeError::kOk;
  });

  jb.entries.assign(kJournalEntriesPerBlock, kill);
  ASSERT_EQ(DecodeJournalBlock(Journal(jb).data(), kStore, &got),
            DecodeError::kOk);
  EXPECT_EQ(got.entries.size(), kJournalEntriesPerBlock);
  kill.incarnation = 0;
  jb.entries = {kill};
  EXPECT_EQ(DecodeJournalBlock(Journal(jb).data(), kStore, &got),
            DecodeError::kBadField)
      << "a kill of incarnation 0 names nothing";
}

// A block never holds fewer entries than it was given: a KILL left out would
// bring its value back after the REMOVE returned.
TEST(LseFormat, JournalBlockRefusesMoreEntriesThanFit) {
  ASSERT_EQ(kJournalEntriesPerBlock, 63u);
  JournalBlock jb;
  jb.store_uuid = kStore;
  jb.sequence = 1;
  JournalEntry kill;
  kill.type = kJournalKill;
  kill.incarnation = 1;
  kill.offset_blocks = 1;
  kill.generation = 1;
  jb.entries.assign(kJournalEntriesPerBlock + 1, kill);
  Block b;
  b.fill(0xEE);
  EXPECT_FALSE(EncodeJournalBlock(jb, &b));
  EXPECT_EQ(b[0], 0xEE) << "nothing written";
  // On disk, a count above 63 is refused too (CRC-valid, so by the field
  // check).
  jb.entries.resize(kJournalEntriesPerBlock);
  b = Journal(jb);
  PutLe(&b[32], kJournalEntriesPerBlock + 1, 2);
  Reseal(&b);
  JournalBlock got;
  EXPECT_EQ(DecodeJournalBlock(b.data(), kStore, &got), DecodeError::kBadField);
}

TEST(LseFormat, JournalBlockRefusesUnknownEntriesAndLeftovers) {
  JournalBlock jb;
  jb.store_uuid = kStore;
  jb.sequence = 1;
  JournalEntry e;
  e.type = kJournalKill;
  e.incarnation = 1;
  e.offset_blocks = 1;
  e.generation = 1;
  jb.entries = {e, e};
  JournalBlock got;
  for (uint8_t type : {0, 3, 0x80, 0xFF}) {
    SCOPED_TRACE(int(type));
    jb.entries[1].type = type;
    EXPECT_EQ(DecodeJournalBlock(Journal(jb).data(), kStore, &got),
              DecodeError::kBadField)
        << "an unknown entry type";
  }
  // A slot past the count holds something: not a block this format writes.
  jb.entries[1].type = kJournalKill;
  Block b = Journal(jb);
  PutLe(&b[32], 1, 2);
  Reseal(&b);
  EXPECT_EQ(DecodeJournalBlock(b.data(), kStore, &got), DecodeError::kBadField);
}

TEST(LseFormat, SuperblockRefusesResetStatesAndNameGarbage) {
  Superblock got;
  Superblock sb = SampleSuperblock();
  sb.reset_in_progress = 1;
  EXPECT_EQ(DecodeSuperblock(EncodeSuperblock(sb).data(), true, &got),
            DecodeError::kOk);
  for (uint32_t v : {2u, 3u, 0xFFFFFFFFu}) {
    sb.reset_in_progress = v;
    EXPECT_EQ(DecodeSuperblock(EncodeSuperblock(sb).data(), true, &got),
              DecodeError::kBadField)
        << v;
  }
  sb = SampleSuperblock();
  Block b = EncodeSuperblock(sb);
  b[104 + 20] = 'z';  // after the name's terminating zero
  Reseal(&b);
  EXPECT_EQ(DecodeSuperblock(b.data(), true, &got), DecodeError::kBadField);
}

// ---- per-decoder fuzz (design §11.2) -------------------------------------------
// CRC-valid blocks with random field values: each is refused, or encodes back
// to the same bytes. Fields are changed whole, at their offsets, so padding
// stays zero as the encoders write it.
namespace {
struct Field {
  size_t offset, size;
};
uint64_t RandomValue(std::mt19937_64& rng, uint64_t old, size_t size) {
  const uint64_t mask = size >= 8 ? ~0ull : (1ull << (8 * size)) - 1;
  switch (rng() % 9) {
    case 0: return 0;
    case 1: return 1 + rng() % 3;
    case 2: return rng() % 64;
    case 3: return (old + 1) & mask;
    case 4: return (old - 1) & mask;
    case 5: return (1ull << (rng() % (8 * size))) & mask;
    case 6: return mask;
    case 7: return (rng() % 8192) * 4096 & mask;
    default: return rng() & mask;
  }
}
void MutateFields(std::mt19937_64& rng, const std::vector<Field>& fields,
                  uint8_t* base) {
  const int n = 1 + static_cast<int>(rng() % 3);
  for (int i = 0; i < n; ++i) {
    const Field& f = fields[rng() % fields.size()];
    uint8_t* p = base + f.offset;
    if (f.size > 8) {  // a uuid or a name: another store's, or random bytes
      for (size_t j = 0; j < f.size; ++j)
        p[j] = rng() % 2 ? static_cast<uint8_t>(rng()) : 0;
      continue;
    }
    uint64_t old = 0;
    for (size_t j = 0; j < f.size; ++j) old |= uint64_t(p[j]) << (8 * j);
    PutLe(p, RandomValue(rng, old, f.size), f.size);
  }
}
}  // namespace

TEST(LseFormat, FuzzedBlocksAreRefusedOrReencodeIdentically) {
  std::mt19937_64 rng(1234);
  const int kRounds = 20000;
  int accepted[6] = {0, 0, 0, 0, 0, 0};
  int refused[6] = {0, 0, 0, 0, 0, 0};
  auto count = [&](int kind, bool ok) { (ok ? accepted : refused)[kind]++; };

  const std::vector<Field> sb_fields = {
      {8, 4},  {12, 4}, {16, 8}, {24, 8}, {32, 8},  {40, 16}, {56, 8},
      {64, 8}, {72, 4}, {76, 4}, {80, 8}, {88, 8},  {96, 4},  {100, 4},
      {104, 32}};
  const std::vector<Field> sh_fields = {{8, 16}, {24, 4}, {32, 8}};
  const std::vector<Field> rh_fields = {
      {4, 1},  {5, 1},  {6, 2},  {8, 16}, {24, 8}, {32, 8},
      {40, 8}, {48, 8}, {56, 8}, {64, 8}, {72, 8}, {80, 4}};
  const std::vector<Field> tr_fields = {{8, 16}, {24, 8}, {32, 4}, {40, 8},
                                        {48, 8}, {56, 4}, {64, 8}};
  std::vector<Field> jb_fields = {{8, 16}, {24, 8}, {32, 2}};
  for (size_t i = 0; i < 3; ++i) {
    const size_t at = kJournalHeaderBytes + i * kJournalEntryBytes;
    for (Field f : std::vector<Field>{{0, 1}, {4, 4}, {8, 8}, {16, 4}, {20, 4},
                                      {24, 4}, {32, 8}, {40, 8}, {48, 8},
                                      {56, 8}})
      jb_fields.push_back({at + f.offset, f.size});
  }
  const std::vector<Field> se_fields = {{0, 8},  {8, 8},  {16, 8}, {24, 8},
                                        {32, 8}, {40, 8}, {48, 4}, {52, 4},
                                        {56, 1}, {58, 2}};

  for (int round = 0; round < kRounds; ++round) {
    SCOPED_TRACE(round);
    switch (round % 6) {
      case 0: {
        Block b = EncodeSuperblock(SampleSuperblock());
        MutateFields(rng, sb_fields, b.data());
        Reseal(&b);
        Superblock sb;
        const bool ok = DecodeSuperblock(b.data(), true, &sb) == DecodeError::kOk;
        if (ok) ASSERT_TRUE(EncodeSuperblock(sb) == b);
        count(0, ok);
        break;
      }
      case 1: {
        Block b = EncodeSegmentHeader({kStore, 3, 77});
        MutateFields(rng, sh_fields, b.data());
        Reseal(&b);
        SegmentHeader h;
        const bool ok = DecodeSegmentHeader(b.data(), kStore, &h) == DecodeError::kOk;
        if (ok) ASSERT_TRUE(EncodeSegmentHeader(h) == b);
        count(1, ok);
        break;
      }
      case 2: {
        RecordHeader h = SampleRecord();
        std::string payload(kBlockBytes, '\0');
        for (auto& c : payload) c = static_cast<char>(rng());
        if (rng() % 2) {
          h.flags = kRecordFlagInline;
          h.value_len = rng() % (kMaxInlinePayload + 1);
        }
        Block b = Record(h, payload.data());
        MutateFields(rng, rh_fields, b.data());
        ResealRecord(&b);
        RecordHeader got;
        const bool ok = DecodeRecordHeader(b.data(), kStore, 9, kSeg, &got) ==
                        DecodeError::kOk;
        if (ok) {
          const bool is_inline = (got.flags & kRecordFlagInline) != 0;
          Block again;
          ASSERT_TRUE(EncodeRecordHeader(got, &b[got.header_len], &again));
          // What the record defines: its header, and an inline payload.
          const size_t defined =
              got.header_len + (is_inline ? got.value_len : 0);
          ASSERT_EQ(std::memcmp(again.data(), b.data(), defined), 0);
          ASSERT_LE(got.value_len, MaxRecordValue(kSeg));
          ASSERT_LE(RecordFootprint(got), kSeg - kBlockBytes);
        }
        count(2, ok);
        break;
      }
      case 3: {
        Trailer t;
        t.store_uuid = kStore;
        t.incarnation = 9;
        t.record_count = 12;
        t.summary_offset = 100 * kBlockBytes;
        t.summary_length = 12 * kSummaryEntryBytes;
        t.summary_crc = 0x1234;
        t.last_write_lsn = 50;
        Block b = EncodeTrailer(t);
        MutateFields(rng, tr_fields, b.data());
        Reseal(&b);
        Trailer got;
        const bool ok =
            DecodeTrailer(b.data(), kStore, 9, kSeg, &got) == DecodeError::kOk;
        if (ok) {
          ASSERT_TRUE(EncodeTrailer(got) == b);
          ASSERT_LE(got.summary_offset + got.summary_length,
                    kSeg - kBlockBytes);
        }
        count(3, ok);
        break;
      }
      case 4: {
        JournalBlock jb;
        jb.store_uuid = kStore;
        jb.sequence = 3;
        for (size_t i = 0; i < 1 + rng() % 3; ++i) {
          JournalEntry e;
          e.type = rng() % 2 ? kJournalKill : kJournalRetain;
          e.segment_index = 2;
          e.incarnation = 5;
          e.offset_blocks = 9;
          e.key = BlockKey{rng(), rng(), rng()};
          e.generation = 40;
          e.retain_ttl = 60;
          jb.entries.push_back(e);
        }
        Block b = Journal(jb);
        MutateFields(rng, jb_fields, b.data());
        Reseal(&b);
        JournalBlock got;
        const bool ok = DecodeJournalBlock(b.data(), kStore, &got) == DecodeError::kOk;
        if (ok) {
          Block again;
          ASSERT_TRUE(EncodeJournalBlock(got, &again));
          ASSERT_TRUE(again == b);
        }
        count(4, ok);
        break;
      }
      default: {
        std::vector<SummaryEntry> entries(1 + rng() % 3);
        uint32_t at = 1;
        for (auto& e : entries) {
          e.key = BlockKey{rng(), rng(), rng()};
          e.value_len = rng() % 100000;
          e.flags = RecordFitsInline(e.value_len) ? kRecordFlagInline : 0;
          e.generation = 7;
          e.write_lsn = 8;
          e.offset_blocks = at;
          at += 30;
        }
        std::vector<uint8_t> bytes = EncodeSummary(entries);
        std::vector<Field> fields;
        for (size_t i = 0; i < entries.size(); ++i)
          for (const Field& f : se_fields)
            fields.push_back({i * kSummaryEntryBytes + f.offset, f.size});
        MutateFields(rng, fields, bytes.data());
        std::vector<SummaryEntry> got;
        const bool ok = DecodeSummary(bytes.data(), bytes.size(), kSeg, &got);
        if (ok) ASSERT_TRUE(EncodeSummary(got) == bytes);
        count(5, ok);
        break;
      }
    }
  }
  for (int kind = 0; kind < 6; ++kind) {
    SCOPED_TRACE(kind);
    RecordProperty("accepted_" + std::to_string(kind), accepted[kind]);
    // Both outcomes are reached, so the fuzz tests both directions.
    EXPECT_GT(accepted[kind], 100);
    EXPECT_GT(refused[kind], 100);
  }
}

TEST(LseFormat, ArbitraryBytesNeverDecode) {
  std::mt19937_64 rng(11);
  for (int round = 0; round < 2000; ++round) {
    Block b;
    for (auto& x : b) x = static_cast<uint8_t>(rng());
    // Sometimes start with a valid magic so the later checks are reached.
    switch (round % 5) {
      case 1: std::memcpy(b.data(), "DFKVLSE1", 8); break;
      case 2: std::memcpy(b.data(), "DFLSSEG1", 8); break;
      case 3: std::memcpy(b.data(), "DLR1", 4); break;
      case 4: std::memcpy(b.data(), "DFLSJRN1", 8); break;
      default: break;
    }
    Superblock sb;
    SegmentHeader sh;
    RecordHeader rh;
    Trailer tr;
    JournalBlock jb;
    EXPECT_NE(DecodeSuperblock(b.data(), true, &sb), DecodeError::kOk);
    EXPECT_NE(DecodeSegmentHeader(b.data(), kStore, &sh), DecodeError::kOk);
    EXPECT_NE(DecodeRecordHeader(b.data(), kStore, 1, kSeg, &rh), DecodeError::kOk);
    EXPECT_NE(DecodeTrailer(b.data(), kStore, 1, kSeg, &tr), DecodeError::kOk);
    EXPECT_NE(DecodeJournalBlock(b.data(), kStore, &jb), DecodeError::kOk);
  }
}
